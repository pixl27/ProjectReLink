/* (hookshot) OoT's job barriers: a wait that ends when the work it waits for is done, in place of the game's 20 us sleep
   (HOS_OOT_BARRIER_WAIT; host/oot_barrier.c is the glue, nx2/oot_patches.py patch_barrier_wait puts in the calls).

   Three of the game's parallel-for jobs (f_800998a0, f_800738e0, f_8007e200) run on several threads at once over one job
   block P: each pass over the job's items takes an item and marks its progress with an atomic OR into a bitmask in P, and a
   thread with nothing left to take sleeps 20 us (nn::os::SleepThread) and looks again, until the masks it polls are full.
   On Windows with few logical processors a sleep under a millisecond waits on the timer and wakes ~0.5 ms late
   (host/hos_thread.c sleep_ns), so each look came late: on 4 logical processors, standing in Castle Town's market, the
   main thread was asleep at those sleeps in 11.4 % of its samples (~2 ms a frame). Making the main thread's short sleeps
   yield instead raised its CPU from 22-29 % to 33-37 % of a core and did not shorten the frame (18 Nov, the same scene,
   two rounds an arm): what is wanted is a wake at the moment the work is done, with no CPU spent until then.

   Here a thread with nothing left waits on a word of the host's, a slot of a small table keyed by P, until a mask word
   it polls differs from what its pass saw, or every one is full, or a timeout; a thread that ORs into a polled word wakes
   the slot's waiters right after the OR. The game's own loop is kept: after the wait it looks at the masks again and
   takes what there is, as after its sleep, so a wait that ends early (a wake for another job in the same slot, a timeout)
   costs a look and nothing else.

   No lost wake-up (every load and read-modify-write here sequentially consistent, and the waker fenced after its OR):
     waiter: waiters += 1 (armed); then, in a loop: g = gen; look at the words (they moved: done); wait while gen == g
     waker:  OR into the word (the lifted code); fence; if waiters: gen += 1, wake every waiter on gen
   If the waker reads waiters == 0, its OR comes before the arming in the one order of these operations, so the waiter's
   first look after arming sees it. Otherwise it moves gen and wakes: a waiter whose g was read after the move sees the
   OR in the look that follows it; one whose g was read before finds gen != g when the wait compares (and returns at
   once) or is asleep on gen and is woken. A wake on gen with no move of a polled word (another job of the slot, a
   spurious wake of the OS) is looked at and waited again inside the wait, without going back to the game.

   The includer gives the platform's wait and wake (OOT_BARRIER_WAIT_ON(address of a uint32_t, the value it must still
   hold, ms), which returns at once when the word differs, and OOT_BARRIER_WAKE_ALL(address)) and the clock
   (OOT_BARRIER_NOW_NS()). OOT_BARRIER_ARMING() runs just before the arming and OOT_BARRIER_LOOKED() between a look and
   the wait after it: nothing in the game, a test pauses there to put a wake in each window. */
#ifndef OOT_BARRIER_H
#define OOT_BARRIER_H
#include <stdint.h>

#ifndef OOT_BARRIER_ARMING
#define OOT_BARRIER_ARMING() ((void)0)
#endif
#ifndef OOT_BARRIER_LOOKED
#define OOT_BARRIER_LOOKED() ((void)0)
#endif

enum { OOT_BARRIER_SLOTS = 64, OOT_BARRIER_WORDS = 2 };
/* how a wait ended: the words had moved before it began (no wait), they moved while it waited, or its time ran out */
enum { OOT_BARRIER_AT_ONCE, OOT_BARRIER_MOVED, OOT_BARRIER_TIMED_OUT };

typedef struct OotBarrierSlot {
    _Alignas(64) volatile uint32_t gen;   /* (its own cache line: the jobs' threads move it at their ORs) */
} OotBarrierSlot;
typedef struct OotBarrier {
    OotBarrierSlot slot[OOT_BARRIER_SLOTS];
    _Alignas(64) volatile int32_t waiters;   /* (threads armed in a wait, every slot: the wakers' one load when none waits) */
} OotBarrier;

/* the slot of a job block (P, a guest address: blocks are at least 64 bytes apart) */
static inline OotBarrierSlot *oot_barrier_slot(OotBarrier *b, uint64_t key) {
    return &b->slot[(uint32_t)(key >> 6 ^ key >> 12 ^ key >> 18) % OOT_BARRIER_SLOTS];
}

/* 1 when a polled word differs from what the pass saw (w[i] against seen[i]), or every one is full: the game's loop then
   has something to look at */
static inline int oot_barrier_moved(volatile uint32_t *const *w, const uint32_t *seen, int n, uint32_t full) {
    int all_full = 1;
    for (int i = 0; i < n; i++) {
        const uint32_t v = __atomic_load_n(w[i], __ATOMIC_SEQ_CST);
        if (v != seen[i]) return 1;
        all_full &= v == full;
    }
    return all_full;
}

/* Waits until a polled word moves (oot_barrier_moved) or timeout_ns passes; the caller has looked at the words already and
   given its virtual core up. OOT_BARRIER_MOVED or OOT_BARRIER_TIMED_OUT */
static inline int oot_barrier_wait(OotBarrier *b, uint64_t key, volatile uint32_t *const *w, const uint32_t *seen, int n,
                                   uint32_t full, int64_t timeout_ns) {
    OotBarrierSlot *s = oot_barrier_slot(b, key);
    const int64_t end = OOT_BARRIER_NOW_NS() + timeout_ns;
    int how = OOT_BARRIER_TIMED_OUT;
    OOT_BARRIER_ARMING();
    __atomic_add_fetch(&b->waiters, 1, __ATOMIC_SEQ_CST);
    for (;;) {
        const uint32_t g = __atomic_load_n(&s->gen, __ATOMIC_SEQ_CST);
        if (oot_barrier_moved(w, seen, n, full)) {   /* (the look after arming: a change before it is never missed) */
            how = OOT_BARRIER_MOVED;
            break;
        }
        const int64_t left = end - OOT_BARRIER_NOW_NS();
        if (left <= 0) break;
        OOT_BARRIER_LOOKED();
        OOT_BARRIER_WAIT_ON(&s->gen, g, (uint32_t)((left + 999999) / 1000000));
    }
    __atomic_sub_fetch(&b->waiters, 1, __ATOMIC_SEQ_CST);
    return how;
}

/* After an OR into a polled word of job block key: wakes the slot's waiters (1), or nothing when no thread waits (0) */
static inline int oot_barrier_wake(OotBarrier *b, uint64_t key) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);   /* (the OR before it, whatever order the lift gave it) */
    if (!__atomic_load_n(&b->waiters, __ATOMIC_SEQ_CST)) return 0;
    OotBarrierSlot *s = oot_barrier_slot(b, key);
    __atomic_add_fetch(&s->gen, 1, __ATOMIC_SEQ_CST);
    OOT_BARRIER_WAKE_ALL(&s->gen);
    return 1;
}
#endif
