"""The game's job barriers waiting for their work (host/oot_barrier.h, host/oot_barrier.c; HOS_OOT_BARRIER_WAIT), game-free.

Three of OoT's parallel-for jobs sleep 20 us and look again while the masks they poll are not full; with the switch the sleep
is a wait that ends when a polled mask word moves, and the thread that ORs into such a word wakes the waiters after the OR.
The fixture is the real glue (oot_barrier.c, which includes the real header) over a stub hos.h with a 64 KB guest arena,
real threads and the OS's own wait (WaitOnAddress on Windows, a futex elsewhere), run in two ways:
  - on (unset: the default; 1, yes): a word that moved since the pass looked, or masks already full, end the call at once with
    the virtual core kept; a wait with no move lasts its timeout (the glue's 1 ms, the primitive's 50 ms) and gives the core
    up once and takes it back once; a waiter asleep in the OS's wait is woken by the wake after an OR (one word, and the
    second of two words), not by a wake for another job of its slot or by a wake with no move; a move and its wake put
    just before the arming, or just after the look and before the wait, end the wait at once (no lost wake-up); four
    waiters against a thread that moves the word and wakes, 6,000 waits with random pauses in both windows, none timed out;
    the FPS counter's sleep time counts the wait;
  - off (0, off): the call answers 0 (the game's sleep runs), gives no core up, and the wake moves nothing.
Then broken variants (the header or the glue edited in the temporary copy), each of which must fail the run it breaks: no
look at the words after arming, the waiter not counted, a wake that wakes nobody, a wake that does not move the slot's word,
no timeout, a wake for another job ending the wait, the core not given up, the switch not read, the wrong slot woken.

And the patch (nx2/oot_patches.py patch_barrier_wait) on three small made-up lifted functions written here, their loops'
polls added by the lift's own nx2/poll.py: every sleep in front of its wait, every OR into a polled word followed by its
wake, each once, declared once; each wait told the words its loop's head loads, through the registers that loop compares; a
second run changes nothing; a function whose code differs (a mark or a place: another game version) is left whole while the
others are hooked; the count patch_file reports and the warning without them. The lift keeps each of the three in the
context (nx2/guest_locals.py): its address is in mentioned_addresses(), and touches() says yes for it alone, both as the
lift asks it (before nx2/poll.py adds the polls; patched then polled is polled then patched) and as written. Over a
finished lift (a64_lift.mark_groups, the lift's last step, has put a group's page mark line inside places here) the patch
hooks the three as the lift would have (grouped then patched is patched then grouped), and a second run changes nothing.
Broken copies of the patch (edited in a temporary copy of nx2/oot_patches.py) must fail those checks: hooking code that
differs, a polled OR without its wake, a wait told the wrong word, hooking again on a second run, hooking only the places
that are there, a group's mark line not let through, a back-edge without its poll not let through, what those slots
found not kept.

python -I -B tests/test_oot_barrier.py [--cc gcc]
"""
import argparse
import collections
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cc", "gcc"]

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "third_party" / "a64-kit" / d) for d in ("lift", "tools")] + [str(ROOT / "nx2")]
import a64_lift  # noqa: E402
import oot_patches  # noqa: E402
import poll  # noqa: E402


def check(cond, what):
    if not cond:
        raise SystemExit("FAIL: " + what)


# (the glue's hos.h: the guest memory as a 64 KB arena at 0x80000000; the rest it calls, defined by the fixture)
STUB_HOS = r"""#ifndef HOS_H
#define HOS_H
#include <stdint.h>
#define ARENA_BASE 0x80000000ull
extern uint8_t test_arena[1 << 16];
#define A64_PTR(a) ((void *)(test_arena + ((uint64_t)(a) - ARENA_BASE)))
#define GP(a) ((void *)A64_PTR(a))
int64_t hos_now_ns(void);
void hos_log(const char *fmt, ...);
int hos_vcore_away(void);
void hos_vcore_back(void);
enum { HOS_CHAIN_WAIT = 1, HOS_CHAIN_WOKE, HOS_CHAIN_SIGNAL, HOS_CHAIN_SLEEP, HOS_CHAIN_AWAKE, HOS_CHAIN_PRESENT };
#define HOS_CHAIN(kind, key) ((void)(kind), (void)(key))
static inline int hos_ab_value(const char *name, int normal) { (void)name; return normal; }   /* (host/hos_ab.c: no A/B) */
#endif
"""

FIXTURE = r"""#define _CRT_SECURE_NO_WARNINGS
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <windows.h>   /* (host/posix's: the types; WaitOnAddress and WakeByAddressAll are defined below) */
#include <errno.h>
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif
#include "hos.h"
#include "hookshot_env.h"   /* (env_on: the switches' one rule; the glue then leaves out its own include of it) */

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); exit(1); } } while (0)

uint8_t test_arena[1 << 16];
volatile int hos_fps_timing;
_Thread_local int64_t hos_slept_ns[2];
static volatile long aways, backs;

int64_t hos_now_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER f;
    LARGE_INTEGER t;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (int64_t)((double)t.QuadPart * 1e9 / (double)f.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#endif
}
void hos_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
}
int hos_vcore_away(void) { __atomic_add_fetch(&aways, 1, __ATOMIC_SEQ_CST); return 1; }
void hos_vcore_back(void) { __atomic_add_fetch(&backs, 1, __ATOMIC_SEQ_CST); }

typedef struct Tramp { void (*fn)(void *); void *arg; } Tramp;
#ifdef _WIN32
typedef HANDLE Thread;
static DWORD WINAPI tramp(void *p) { Tramp *t = (Tramp *)p; t->fn(t->arg); return 0; }
static Thread go(Tramp *t) { HANDLE h = CreateThread(NULL, 0, tramp, t, 0, NULL); CHECK(h != NULL); return h; }
static void join(Thread h) { WaitForSingleObject(h, INFINITE); CloseHandle(h); }
static void nap_ms(int ms) { Sleep((DWORD)ms); }
#else
typedef pthread_t Thread;
static void *tramp(void *p) { Tramp *t = (Tramp *)p; t->fn(t->arg); return NULL; }
static Thread go(Tramp *t) { pthread_t h; CHECK(pthread_create(&h, NULL, tramp, t) == 0); return h; }
static void join(Thread h) { pthread_join(h, NULL); }
static void nap_ms(int ms) { usleep((useconds_t)ms * 1000); }
BOOL WaitOnAddress(volatile void *addr, void *compare, SIZE_T size, DWORD ms) {
    (void)size;
    struct timespec ts = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000};
    syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, *(uint32_t *)compare, ms == 0xFFFFFFFFu ? NULL : &ts, NULL, 0);
    return TRUE;
}
void WakeByAddressAll(void *addr) { syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0); }
#endif

/* (the waits' two windows: 1, a pause just before the arming; 2, after a look, just before the OS's wait; 3, a random short
   pause in both. 1 and 2 pause once, until the main thread has made its move and its wake) */
static volatile int race_mode, race_in, race_go;
static _Thread_local uint32_t rng = 0x9e3779b9u;
static void spin_us(int us) { const int64_t end = hos_now_ns() + (int64_t)us * 1000; while (hos_now_ns() < end) {} }
static void race_pause(int where) {
    const int m = __atomic_load_n(&race_mode, __ATOMIC_SEQ_CST);
    if (m == 3) {
        rng ^= rng << 13, rng ^= rng >> 17, rng ^= rng << 5;
        if (rng & 1) spin_us((int)(rng >> 1 & 15));
        return;
    }
    if (m != where) return;
    __atomic_store_n(&race_mode, 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&race_in, 1, __ATOMIC_SEQ_CST);
    while (!__atomic_load_n(&race_go, __ATOMIC_SEQ_CST)) spin_us(20);
}
#define OOT_BARRIER_ARMING() race_pause(1)
#define OOT_BARRIER_LOOKED() race_pause(2)
#include "oot_barrier.c"

#define P1 (ARENA_BASE + 0x1000)
#define P2 (ARENA_BASE + 0x3000)
#define OFF 0x200
#define OFF2 0x208
static volatile uint32_t *word(uint64_t p, uint32_t off) { return (volatile uint32_t *)GP(p + off); }

typedef struct WaitArgs {
    uint64_t p;
    int n;
    uint32_t seen[2], full;
    int64_t timeout, t_start, t_end;
    int how;
} WaitArgs;
static void waiter(void *v) {
    WaitArgs *a = (WaitArgs *)v;
    volatile uint32_t *const w[2] = {word(a->p, OFF), word(a->p, OFF2)};
    a->t_start = hos_now_ns();
    a->how = oot_barrier_wait(&barrier, a->p, w, a->seen, a->n, a->full, a->timeout);
    a->t_end = hos_now_ns();
}
static void armed(int n) {   /* (the waiter counted, so the wake after a move finds it) */
    const int64_t end = hos_now_ns() + 2000000000LL;
    while (__atomic_load_n(&barrier.waiters, __ATOMIC_SEQ_CST) != n && hos_now_ns() < end) nap_ms(1);
    CHECK(__atomic_load_n(&barrier.waiters, __ATOMIC_SEQ_CST) == n);
}

/* a word moved since the pass looked, or every word full: back at once, the core kept */
static void at_once(void) {
    *word(P1, OFF) = 5, *word(P1, OFF2) = 0xf;
    const long a0 = aways;
    const long long n0 = barrier_how[OOT_BARRIER_AT_ONCE];
    CHECK(hos_oot_barrier_wait(P1, OFF, 4, OFF, 4, 0xf) == 1);
    CHECK(hos_oot_barrier_wait(P1, OFF, 5, OFF2, 0xe, 0xf) == 1);
    *word(P1, OFF) = 0xf;
    CHECK(hos_oot_barrier_wait(P1, OFF, 0xf, OFF2, 0xf, 0xf) == 1);
    CHECK(aways == a0 && barrier_how[OOT_BARRIER_AT_ONCE] == n0 + 3);
}

/* no move: the glue's wait lasts its 1 ms and gives the core up once; the primitive's own timeout is kept */
static void timeouts(void) {
    *word(P1, OFF) = 1;
    const long a0 = aways, b0 = backs;
    const long long t_out = barrier_how[OOT_BARRIER_TIMED_OUT];
    hos_fps_timing = 1;
    hos_slept_ns[1] = 0;
    int64_t t0 = hos_now_ns();
    CHECK(hos_oot_barrier_wait(P1, OFF, 1, OFF, 1, 0xf) == 1);
    int64_t took = hos_now_ns() - t0;
    hos_fps_timing = 0;
    CHECK(took >= OOT_BARRIER_TIMEOUT_NS - 20000 && took < 500000000);
    CHECK(aways == a0 + 1 && backs == b0 + 1 && barrier_how[OOT_BARRIER_TIMED_OUT] == t_out + 1);
    CHECK(hos_slept_ns[1] >= OOT_BARRIER_TIMEOUT_NS - 20000);   /* (the FPS counter's Breakdown counts it as a sleep) */
    CHECK(barrier.waiters == 0);
    volatile uint32_t *const w[2] = {word(P1, OFF), word(P1, OFF)};
    const uint32_t seen[2] = {1, 1};
    t0 = hos_now_ns();
    CHECK(oot_barrier_wait(&barrier, P1, w, seen, 1, 0xf, 50000000) == OOT_BARRIER_TIMED_OUT);
    took = hos_now_ns() - t0;
    CHECK(took >= 50000000 - 20000 && took < 2000000000);
}

/* a waiter asleep in the OS's wait: the wake after an OR into its word, or into its second word, ends it at once */
static void wake_on_move(int second) {
    *word(P1, OFF) = second ? 0xf : 1, *word(P1, OFF2) = 1;
    WaitArgs a = {P1, second ? 2 : 1, {second ? 0xf : 1, 1}, 0xf, 2000000000LL, 0, 0, -1};
    Tramp t = {waiter, &a};
    const Thread h = go(&t);
    armed(1);
    nap_ms(20);   /* (asleep in the OS's wait by now) */
    __atomic_fetch_or(word(P1, second ? OFF2 : OFF), 2, __ATOMIC_SEQ_CST);
    const int64_t t0 = hos_now_ns();
    hos_oot_barrier_wake(P1);
    join(h);
    CHECK(a.how == OOT_BARRIER_MOVED);
    CHECK(a.t_end - t0 < 1000000000LL);   /* (by the wake, long before the 2 s) */
    CHECK(barrier.waiters == 0);
}

/* wakes with no move of the waiter's words (another job of its slot, its own job's other words) do not end the wait */
static void no_move_no_end(void) {
    const uint64_t other = P1 + (1ull << 24);   /* (the same slot, another job block) */
    CHECK(oot_barrier_slot(&barrier, other) == oot_barrier_slot(&barrier, P1));
    *word(P1, OFF) = 1;
    WaitArgs a = {P1, 1, {1, 1}, 0xf, 300000000LL, 0, 0, -1};
    Tramp t = {waiter, &a};
    const Thread h = go(&t);
    armed(1);
    for (int i = 0; i < 10; i++) {
        nap_ms(10);
        hos_oot_barrier_wake(other);
        hos_oot_barrier_wake(P1);
    }
    join(h);
    CHECK(a.how == OOT_BARRIER_TIMED_OUT);
    CHECK(a.t_end - a.t_start >= 300000000LL - 20000);
}

/* a move and its wake in each window of the wait: before the arming (the waker sees no waiter and wakes nobody), and
   between a look and the OS's wait (the waker moves the slot's word before the waiter sleeps on it) */
static void race(int where) {
    *word(P1, OFF) = 1;
    __atomic_store_n(&race_in, 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&race_go, 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&race_mode, where, __ATOMIC_SEQ_CST);
    WaitArgs a = {P1, 1, {1, 1}, 0xf, 2000000000LL, 0, 0, -1};
    Tramp t = {waiter, &a};
    const Thread h = go(&t);
    while (!__atomic_load_n(&race_in, __ATOMIC_SEQ_CST)) spin_us(20);
    CHECK(barrier.waiters == (where == 1 ? 0 : 1));
    __atomic_fetch_or(word(P1, OFF), 2, __ATOMIC_SEQ_CST);
    hos_oot_barrier_wake(P1);
    __atomic_store_n(&race_go, 1, __ATOMIC_SEQ_CST);
    join(h);
    CHECK(a.how == OOT_BARRIER_MOVED);
    CHECK(a.t_end - a.t_start < 1000000000LL);   /* (not the 2 s timeout: no lost wake-up) */
}

/* many waiters against a thread that moves the word and wakes, with random pauses in both windows: no wait times out */
enum { STRESS_WAITERS = 4, STRESS_ROUNDS = 1500 };
static volatile int stress_left, stress_timeouts;
static void stress_waiter(void *v) {
    (void)v;
    volatile uint32_t *const w[2] = {word(P2, OFF), word(P2, OFF)};
    for (int r = 0; r < STRESS_ROUNDS && !stress_timeouts; r++) {
        const uint32_t seen[2] = {__atomic_load_n(w[0], __ATOMIC_SEQ_CST), 0};
        if (oot_barrier_wait(&barrier, P2, w, seen, 1, 0xffffffffu, 2000000000LL) == OOT_BARRIER_TIMED_OUT)
            __atomic_add_fetch(&stress_timeouts, 1, __ATOMIC_SEQ_CST);
    }
    __atomic_sub_fetch(&stress_left, 1, __ATOMIC_SEQ_CST);
}
static void stress_waker(void *v) {
    (void)v;
    uint32_t r = 12345;
    while (__atomic_load_n(&stress_left, __ATOMIC_SEQ_CST)) {
        __atomic_fetch_add(word(P2, OFF), 1, __ATOMIC_SEQ_CST);
        hos_oot_barrier_wake(P2);
        r ^= r << 13, r ^= r >> 17, r ^= r << 5;
        spin_us((int)(r % 30));
    }
}
static void stress(void) {
    *word(P2, OFF) = 0;
    stress_left = STRESS_WAITERS, stress_timeouts = 0;
    __atomic_store_n(&race_mode, 3, __ATOMIC_SEQ_CST);
    Tramp tw = {stress_waiter, NULL}, tk = {stress_waker, NULL};
    Thread h[STRESS_WAITERS + 1];
    for (int i = 0; i < STRESS_WAITERS; i++) h[i] = go(&tw);
    h[STRESS_WAITERS] = go(&tk);
    for (int i = 0; i <= STRESS_WAITERS; i++) join(h[i]);
    __atomic_store_n(&race_mode, 0, __ATOMIC_SEQ_CST);
    CHECK(stress_timeouts == 0);
    CHECK(barrier.waiters == 0);
}

/* off: the game's sleep runs (0), no core given up, a wake moves nothing */
static void off(void) {
    CHECK(hos_oot_barrier_on() == 0);
    CHECK(barrier_bind());   /* (so a wait the switch does not stop waits, and fails the checks below, not crashes) */
    *word(P1, OFF) = 1;
    const long a0 = aways;
    CHECK(hos_oot_barrier_wait(P1, OFF, 1, OFF, 1, 0xf) == 0);
    CHECK(aways == a0);
    barrier.waiters = 1;
    const uint32_t g = oot_barrier_slot(&barrier, P1)->gen;
    hos_oot_barrier_wake(P1);
    CHECK(oot_barrier_slot(&barrier, P1)->gen == g);
    barrier.waiters = 0;
}

/* (a wait without its timeout spins for ever: the run then fails here) */
static volatile int finished;
static void watchdog(void *v) {
    (void)v;
    for (int i = 0; i < 200 && !finished; i++) nap_ms(50);
    if (!finished) {
        fprintf(stderr, "FAIL: the fixture did not end in 10 s\n");
        exit(1);
    }
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    static Tramp tw = {watchdog, NULL};
    go(&tw);
    if (!strcmp(argv[1], "off")) {
        off();
        finished = 1;
        printf("barrier off: all passed\n");
        return 0;
    }
    CHECK(!strcmp(argv[1], "on"));
    CHECK(hos_oot_barrier_on() == 1);
    at_once();
    timeouts();
    wake_on_move(0);
    wake_on_move(1);
    no_move_no_end();
    race(1);
    race(2);
    stress();
    CHECK(aways == backs);
    finished = 1;
    printf("barrier on: all passed (under stress %d waits; %lld wakes found a waiter)\n", STRESS_WAITERS * STRESS_ROUNDS,
           barrier_wakes);
    return 0;
}
"""

# (where, old, new, the mode whose run must fail, what it breaks); a list of (old, new) pairs edits more than one place
BROKEN = [
    ("header", [("    OOT_BARRIER_ARMING();\n",
                 "    if (oot_barrier_moved(w, seen, n, full)) return OOT_BARRIER_MOVED;\n    OOT_BARRIER_ARMING();\n"),
                ("        if (oot_barrier_moved(w, seen, n, full)) {", "        if (0) {"),
                ("        OOT_BARRIER_WAIT_ON(&s->gen, g, (uint32_t)((left + 999999) / 1000000));\n",
                 "        OOT_BARRIER_WAIT_ON(&s->gen, g, (uint32_t)((left + 999999) / 1000000));\n"
                 "        if (oot_barrier_moved(w, seen, n, full)) { how = OOT_BARRIER_MOVED; break; }\n")],
     "on", "the waiter looks at the words before arming and after each wait, not between the arming and its first wait"),
    ("header", [("    __atomic_add_fetch(&b->waiters, 1, __ATOMIC_SEQ_CST);\n", ""),
                ("    __atomic_sub_fetch(&b->waiters, 1, __ATOMIC_SEQ_CST);\n", "")],
     "on", "the waiter is not counted, so the waker skips the wake"),
    ("header", [("    OOT_BARRIER_WAKE_ALL(&s->gen);\n", "    (void)s;\n")], "on", "the wake wakes nobody"),
    ("header", [("    __atomic_add_fetch(&s->gen, 1, __ATOMIC_SEQ_CST);\n", "")], "on",
     "the wake does not move the slot's word (a waiter between its look and its sleep sleeps on)"),
    ("header", [("        if (left <= 0) break;\n", "")], "on", "no timeout"),
    ("header", [("        OOT_BARRIER_WAIT_ON(&s->gen, g, (uint32_t)((left + 999999) / 1000000));\n",
                 "        OOT_BARRIER_WAIT_ON(&s->gen, g, (uint32_t)((left + 999999) / 1000000));\n"
                 "        if (__atomic_load_n(&s->gen, __ATOMIC_SEQ_CST) != g) break;\n")],
     "on", "a wake with no move of the waiter's words (another job of the slot) ends the wait"),
    ("glue", [("    const int away = hos_vcore_away();", "    const int away = 0;")], "on", "the virtual core not given up"),
    ("glue", [("int hos_oot_barrier_wait(uint64_t p, uint32_t off1, uint32_t seen1, uint32_t off2, uint32_t seen2, uint32_t full) {\n"
               "    if (!hos_oot_barrier_on()) return 0;\n",
               "int hos_oot_barrier_wait(uint64_t p, uint32_t off1, uint32_t seen1, uint32_t off2, uint32_t seen2, uint32_t full) {\n")],
     "off", "the switch not read by the wait"),
    ("glue", [("    if (oot_barrier_wake(&barrier, p)) {", "    if (oot_barrier_wake(&barrier, p + 64)) {")], "on",
     "the wrong slot woken"),
]


# ---- the patch, on made-up lifted text: the heads, the marks and the places patch_barrier_wait looks for, written here (not
# taken from the patch), between lines of filler; the loops' back-edges as the lift writes them before nx2/poll.py polls them
FILL = "    c->x[9] = 0x1ULL;\n    c->x[9] &= 0xFFFFFFFFULL;\n"
# (two stores through x20: a64_lift.mark_groups gives them one shared page mark line, before the next line that ends their
# run, which is inside a mark or a place wherever they are put below)
STORES = "    recomp_store32(c,c->x[20] + 0,c->x[1]);\n    recomp_store32(c,c->x[20] + 4,c->x[2]);\n"


def _or(base, off, addr, value, between=""):
    return ("    { uint64_t _a=c->x[%d], _b=%dULL; uint64_t _r=_a+_b; c->x[%d]=_r; }\n%s"
            "{ uint64_t _a = c->x[%d]; uint32_t *_p = (uint32_t *)A64_PTR(_a); uint32_t _o = __atomic_fetch_or(_p, (uint32_t)c->x[%d], "
            "__ATOMIC_SEQ_CST); A64_MARK(_a, 4); c->x[8] = (uint64_t)_o; }\n" % (base, off, addr, between, addr, value))


JOB_A = ("void f_800998a0(Ctx *__restrict A64_NOESC c) {\n" + FILL +
         "L_80099a00:;\n" + FILL +
         "L_8009a420:;\nc->x[30] = 0x8009a424ULL; f_85106210(c);\n" + STORES +
         "    c->x[8]=recomp_load32(c,c->x[24] + 500);\n    recomp_barrier();\n"
         "    { uint64_t _b=c->x[29]; int64_t _o=-100; c->x[22]=recomp_load32(c,(_b+_o)); }\n" + FILL +
         "c->x[30] = 0x8009a474ULL; f_851063b0(c);\nc->x[30] = 0x8009a478ULL; f_85106220(c);\ngoto L_80099a00;\n"
         "L_8009a6ac:;\n" + STORES + _or(24, 500, 8, 23) + "    recomp_barrier();\n" + FILL +
         "L_8009a6c0:;\n" + _or(24, 508, 8, 23) + FILL +
         "c->x[30] = 0x8009a6e0ULL; f_80099350(c);\n    recomp_barrier();\n" + _or(24, 512, 8, 23) +
         "L_8009a6f0:;\n    c->x[8]=recomp_load32(c,c->x[24] + 508);\n" + FILL +
         "L_8009a768:;\n    c->x[25]=recomp_load32(c,c->x[24] + 512);\n" + FILL +
         "    recomp_store32_sp(c,c->x[31] + 104,c->x[25]);\n" + FILL +
         "    c->x[25]=recomp_load32(c,c->x[31] + 104);\n" +
         _or(24, 520, 8, 9, "    c->x[11] = 0x1ULL;\n    c->x[11] &= 0xFFFFFFFFULL;\n    c->x[9]=recomp_load32(c,c->x[31] + 112);\n") +
         "L_8009aa1c:;\n    c->x[19]=recomp_load32(c,c->x[24] + 520);\n" + FILL +
         "    recomp_store32_sp(c,c->x[31] + 72,c->x[19]);\n" + FILL +
         "    c->x[19]=recomp_load32(c,c->x[31] + 72);\n"
         "c->x[30] = 0x8009afa4ULL; f_85106210(c);\n"
         "    { uint64_t _b=c->x[29]; int64_t _o=-100; c->x[8]=recomp_load32(c,(_b+_o)); }\n" + FILL +
         "c->x[30] = 0x8009afecULL; f_851063b0(c);\nc->x[30] = 0x8009aff0ULL; f_85106220(c);\ngoto L_8009aa1c;\n" +
         "c->x[30] = 0x8009b230ULL; f_851063b0(c);\nc->x[30] = 0x8009b234ULL; f_85106220(c);\n"
         "    c->x[11] = 0x1ULL;\n    c->x[11] &= 0xFFFFFFFFULL;\ngoto L_8009a768;\n}\n")
JOB_B = ("void f_800738e0(Ctx *__restrict A64_NOESC c) {\n"
         "    { uint64_t _r = ((uint64_t)0 | ((c->x[0]) << 0)); c->x[19] = _r; }\n" + FILL + STORES +
         _or(19, 528, 27, 26, "c->x[30] = 0x80073958ULL; f_80279140(c);\n    recomp_barrier();\n") + FILL +
         _or(19, 536, 8, 9, "    { c->x[9]=(uint64_t)(uint32_t)((uint32_t)c->x[3]<<(c->x[21]&31)); }\n    recomp_barrier();\n") +
         "    recomp_store32_sp(c,c->x[31] + 16,c->x[8]);\n"
         "L_80073e38:;\n    c->x[25]=recomp_load32(c,c->x[19] + 528);\n    recomp_barrier();\n    c->x[28]=recomp_load32(c,c->x[19] + 536);\n" +
         FILL + "    c->x[9]=recomp_load32(c,c->x[31] + 16);\n" + FILL +
         "c->x[30] = 0x800747a8ULL; f_851063b0(c);\nc->x[30] = 0x800747acULL; f_85106220(c);\ngoto L_80073e38;\n}\n")
JOB_C = ("void f_8007e200(Ctx *__restrict A64_NOESC c) {\n" + FILL +
         "    recomp_store32_sp(c,c->x[31] + 68,c->x[9]);\n" + FILL +
         _or(26, 544, 8, 9, "    { c->x[9]=(uint64_t)(uint32_t)((uint32_t)c->x[23]<<(c->x[21]&31)); }\n    recomp_barrier();\n") +
         "L_8007e6c0:;\n    c->x[21]=recomp_load32(c,c->x[26] + 544);\n    recomp_barrier();\n"
         "    c->x[28]=recomp_load32(c,c->x[31] + 68);\n" + FILL +
         "    { uint64_t _r = (c->x[21] ^ ((uint64_t)(uint32_t)(((uint32_t)(c->x[28])) << 0))); _r &= 0xFFFFFFFFULL; c->x[8] = _r; }\n" +
         FILL + "c->x[30] = 0x8007e908ULL; f_851063b0(c);\nc->x[30] = 0x8007e90cULL; f_85106220(c);\ngoto L_8007e6c0;\n}\n")
UNPOLLED = '#include "a64.h"\n\n' + JOB_A + JOB_B + JOB_C   # (as nx2/guest_locals.py asks touches() about each function)
LIFTED, POLLS = poll.add_polls(UNPOLLED)   # (as the patches meet it: each loop's back-edge `{ A64_POLL(c); goto L_x; }`)
GROUPED = a64_lift.mark_groups(LIFTED)     # (a finished lift's: each group's page marks shared, the lift's last step)
# (each sleep: its return address, the loop head it goes back to and the words that head loads into the registers the wait
# must be told, P's register; the ORs into polled words: function, P's register, offset)
SLEEPS = (("f_800998a0", 0x8009a478, None, "c->x[24]", ((500, "(uint32_t)recomp_load32(c,c->x[24] + 500)"),
                                                     (508, "(uint32_t)recomp_load32(c,c->x[24] + 508)")), "(uint32_t)c->x[22]"),
          ("f_800998a0", 0x8009b234, "L_8009a768", "c->x[24]", ((512, "(uint32_t)c->x[25]"),), "(uint32_t)c->x[22]"),
          ("f_800998a0", 0x8009aff0, "L_8009aa1c", "c->x[24]", ((520, "(uint32_t)c->x[19]"),),
           "(uint32_t)recomp_load32(c,c->x[29] - 100)"),
          ("f_800738e0", 0x800747ac, "L_80073e38", "c->x[19]", ((528, "(uint32_t)c->x[25]"), (536, "(uint32_t)c->x[28]")),
           "(uint32_t)recomp_load32(c,c->x[31] + 16)"),
          ("f_8007e200", 0x8007e90c, "L_8007e6c0", "c->x[26]", ((544, "(uint32_t)c->x[21]"),), "~(uint32_t)c->x[28]"))
ORS = (("f_800998a0", 500), ("f_800998a0", 508), ("f_800998a0", 512), ("f_800998a0", 520), ("f_800738e0", 528),
       ("f_800738e0", 536), ("f_8007e200", 544))
WAKE = "if (hos_oot_barrier_on()) hos_oot_barrier_wake(_a - %dULL); }"


def function_text(s, name):
    i = s.index("void %s(Ctx" % name)
    j = s.find("\nvoid f_", i + 1)
    return s[i:j if j >= 0 else len(s)]


def check_hooked(op, s, names=("f_800998a0", "f_800738e0", "f_8007e200")):
    """every sleep of the named functions in front of its wait, told its loop's words; every polled OR with its wake"""
    for fn, lr, head, p, words, full in SLEEPS:
        if fn not in names:
            continue
        body = function_text(s, fn)
        call = "c->x[30] = 0x%xULL; f_85106220(c);" % lr
        check(call not in body, "%s: the sleep at LR %#x is left without its wait" % (fn, lr))
        args = [p] + [x for off, seen in (words if len(words) == 2 else words * 2) for x in (str(off), seen)] + [full]
        hooked = ("c->x[30] = 0x%xULL; if (!hos_oot_barrier_on() || !hos_oot_barrier_wait(%s)) f_85106220(c);" % (lr, ", ".join(args)))
        check(body.count(hooked) == 1, "%s: the sleep at LR %#x is not in front of its wait once, told %s" % (fn, lr, args))
        if head:   # (the words the wait is told are the ones its loop's head loads into those registers, from P)
            loop = body[body.index(head + ":;\n"):]
            for off, seen in words:
                reg = seen[len("(uint32_t)"):]
                check(("%s=recomp_load32(c,%s + %d);" % (reg, p, off)) in loop.split(":;\n", 2)[1],
                      "%s: the wait at LR %#x is told %s at +%d, which its loop's head does not load" % (fn, lr, seen, off))
    for fn, off in ORS:
        if fn not in names:
            continue
        body = function_text(s, fn)
        check(body.count(WAKE % off) == 1, "%s: the OR into P+%#x is not followed by its wake once" % (fn, off))
    check(s.count(op.BARRIER_DECL) == 1 and ('#include "a64.h"\n' + op.BARRIER_DECL) in s, "the hooks are not declared once")


def patch_checks(op):
    check(POLLS == 5 and LIFTED.count("{ A64_POLL(c); goto L_") == 5, "nx2/poll.py did not poll the five loops' back-edges")
    out, n = op.patch_barrier_wait(LIFTED)
    check(n == 3, "patch_barrier_wait found %d of the 3 functions" % n)
    check_hooked(op, out)
    check(op.patch_barrier_wait(out) == (out, 3), "a second run changed a hooked text, or did not count it")
    # (another game version: one mark or one place differs; that function is left whole, the others are hooked)
    for what, old, new in (("a mark", "c->x[19]=recomp_load32(c,c->x[24] + 520);", "c->x[19]=recomp_load32(c,c->x[24] + 524);"),
                           ("a place", "c->x[30] = 0x8009b234ULL; f_85106220(c);", "c->x[30] = 0x8009b238ULL; f_85106220(c);"),
                           ("an OR", "_b=508ULL", "_b=506ULL")):
        other = LIFTED.replace(old, new, 1)
        got, n = op.patch_barrier_wait(other)
        check(n == 2, "with %s of f_800998a0 changed, %d functions were hooked, not 2" % (what, n))
        check(function_text(got, "f_800998a0") == function_text(other, "f_800998a0"),
              "f_800998a0 with %s changed (another game version) was hooked" % what)
        check_hooked(op, got, ("f_800738e0", "f_8007e200"))
    other = LIFTED.replace("    recomp_store32_sp(c,c->x[31] + 68,c->x[9]);\n", "    recomp_store32_sp(c,c->x[31] + 64,c->x[9]);\n")
    got, n = op.patch_barrier_wait(other)
    check(n == 2 and function_text(got, "f_8007e200") == function_text(other, "f_8007e200"),
          "f_8007e200 without its full mask's store was hooked")
    none = '#include "a64.h"\n\nvoid f_80000000(Ctx *__restrict A64_NOESC c) {\n' + FILL + "}\n"
    check(op.patch_barrier_wait(none) == (none, 0), "a file without the functions changed")
    made, counts, lines = op.patch_file(LIFTED, "code_hot_7.c", False)
    check(counts["job barrier waits"] == 3 and any("job barriers" in x for x in lines), "patch_file does not count the barriers")
    again, counts, lines = op.patch_file(made, "code_hot_7.c", False)
    check(again == made and counts["job barrier waits"] == 3 and not any("job barriers" in x for x in lines),
          "patch_file hooked a hooked file again")
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        op.report(["a"], [counts])
    check("job barriers" not in said.getvalue(), "report warns with the three functions hooked")
    with contextlib.redirect_stdout(said):
        op.report(["a"], [collections.Counter({"job barrier waits": 2})])
    check("the job barriers were found in 2 of 3 functions" in said.getvalue(), "report does not warn with two of three")
    # (the lift keeps the three in the context, nx2/guest_locals.py: keep_mentioned by the addresses the patch file names,
    # and touches() asked of the function alone, as localize_regs asks it before nx2/poll.py adds the polls, and as written)
    named = op.mentioned_addresses()
    for fn in ("f_800998a0", "f_800738e0", "f_8007e200"):
        check(int(fn[2:], 16) in named, "%s's address is not in mentioned_addresses(): the lift may give it locals" % fn)
        for what, text in (("as the lift asks it (before the polls)", UNPOLLED), ("as written", LIFTED)):
            check(op.touches(function_text(text, fn)), "touches() says no for %s alone, %s" % (fn, what))
    check(not op.touches(function_text(none, "f_80000000")), "touches() says yes for a function no patch changes")
    unpolled, n = op.patch_barrier_wait(UNPOLLED)
    check(n == 3 and poll.add_polls(unpolled)[0] == out, "the patch before the polls is not the patch after them")
    # (a finished lift, patched by hand: a group's mark line inside the first mark of f_800998a0 and two ORs' places)
    check(GROUPED.count("    A64_MARK((c->x[20] + (uint64_t)(int64_t)(0)), 8);\n") == 3 and _or(24, 500, 8, 23) not in GROUPED,
          "a64_lift.mark_groups put no mark line inside a place here: the case is not made")
    grouped, n = op.patch_barrier_wait(GROUPED)
    check(n == 3 and grouped == a64_lift.mark_groups(out),
          "over a finished lift the patch found %d of the 3 functions, or did not hook them as the lift would have" % n)
    check(op.patch_barrier_wait(grouped) == (grouped, 3), "over a finished lift a second run changed a hooked text, or did "
          "not count it")


# (copies of nx2/oot_patches.py edited: each must fail patch_checks)
BROKEN_PATCH = [
    ("        if not all(m.count(body) == 1 for m in marks) or not all(old.count(body) == 1 for old, _ in places):\n            continue\n",
     "        if not all(old.count(body) == 1 for old, _ in places):\n            continue\n", "hooks code whose marks differ"),
    ("      _barrier_or(24, 500, 8, 23), ", "      ", "a polled OR without its wake"),
    ('"c->x[24], 520, (uint32_t)c->x[19], 520, (uint32_t)c->x[19], ', '"c->x[24], 512, (uint32_t)c->x[19], 512, (uint32_t)c->x[19], ',
     "a wait told the wrong word"),
    ("        if all(new.count(body) == 1 and not old.count(body) for old, new in places):\n            done += 1\n            continue\n",
     "", "a second run hooks again (or does not count a hooked function)"),
    ("not all(old.count(body) == 1 for old, _ in places)", "not any(old.count(body) == 1 for old, _ in places)",
     "hooks the places that are there when one is missing"),
    ('_BARRIER_MARKS = r"((?:', '_BARRIER_MARKS = r"()" or r"((?:', "a group's page mark line inside a place not let through"),
    (r'self.slots.append(r"(\{ A64_POLL\(c\); %s \}|%s)" % (goto, goto))', r'self.slots.append(r"(\{ A64_POLL\(c\); %s \})" % goto)',
     "a loop's back-edge without its poll not let through"),
    ('m.groups() + ("",)', '("",) * (len(m.groups()) + 1)', "the mark lines and gotos a place's slots found not kept"),
]


def broken_patch_checks(tmp):
    source = (ROOT / "nx2" / "oot_patches.py").read_text(encoding="utf-8")
    for i, (old, new, what) in enumerate(BROKEN_PATCH):
        check(source.count(old) == 1, "nx2/oot_patches.py no longer has the text broken patch %d changes: %r" % (i, old))
        path = Path(tmp) / ("oot_patches_broken_%d.py" % i)
        path.write_text(source.replace(old, new, 1), encoding="utf-8", newline="\n")
        spec = importlib.util.spec_from_file_location("oot_patches_broken_%d" % i, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        try:
            patch_checks(mod)
        except SystemExit as e:
            if str(e).startswith("FAIL"):
                continue
            raise
        raise SystemExit("FAIL: broken patch %d (%s) passed the patch checks" % (i, what))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cc", default="gcc")
    a = p.parse_args()
    patch_checks(oot_patches)
    header = (ROOT / "host" / "oot_barrier.h").read_text(encoding="utf-8")
    glue = (ROOT / "host" / "oot_barrier.c").read_text(encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HOS_")}

    def build(tmp, name, h=header, g=glue):
        d = Path(tmp) / name
        d.mkdir()
        (d / "oot_barrier.h").write_text(h, encoding="utf-8", newline="\n")
        (d / "oot_barrier.c").write_text(g, encoding="utf-8", newline="\n")
        (d / "hos.h").write_text(STUB_HOS, encoding="utf-8", newline="\n")
        (d / "fixture.c").write_text(FIXTURE, encoding="utf-8", newline="\n")
        exe = d / "fixture.exe"
        extra = [] if os.name == "nt" else ["-I", str(ROOT / "host" / "posix"), "-pthread"]
        r = subprocess.run([a.cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(d),
                            "-I", str(ROOT / "gpu" / "vk" / "compat")] + extra + [str(d / "fixture.c"), "-o", str(exe)],
                           capture_output=True, text=True)
        check(r.returncode == 0, "the fixture (%s) does not compile:\n%s%s" % (name, r.stdout, r.stderr))
        return exe

    def run(exe, mode, value):
        e = dict(env) if value is None else dict(env, HOS_OOT_BARRIER_WAIT=value)
        try:
            return subprocess.run([str(exe), mode], env=e, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess([str(exe), mode], 1, "", "FAIL: no end in 120 s")

    with tempfile.TemporaryDirectory(prefix="oot-barrier-") as tmp:
        broken_patch_checks(tmp)
        exe = build(tmp, "fixture")
        # (on by default since 18 Nov: unset is on; 0 / off the sleeps as before)
        for mode, value in (("on", "1"), ("on", " yes "), ("on", None), ("off", "0"), ("off", "off")):
            r = run(exe, mode, value)
            check(r.returncode == 0 and ("barrier %s: all passed" % mode) in r.stdout,
                  "%s, HOS_OOT_BARRIER_WAIT=%r:\n%s%s" % (mode, value, r.stdout, r.stderr))
            if mode == "on" and value == "1":
                print(r.stdout.strip().splitlines()[-1])
        for i, (where, edits, mode, what) in enumerate(BROKEN):
            text = header if where == "header" else glue
            for old, _ in edits:
                check(text.count(old) == 1, "host/oot_barrier.%s no longer has the text broken variant %d changes: %r"
                      % ("h" if where == "header" else "c", i, old))
            for old, new in edits:
                text = text.replace(old, new, 1)
            exe = build(tmp, "broken-%d" % i, h=text if where == "header" else header, g=text if where == "glue" else glue)
            r = run(exe, mode, "1" if mode == "on" else "0")
            check(r.returncode != 0 and "FAIL" in r.stderr, "broken variant %d: %s\n%s" % (i, what, r.stdout + r.stderr))
    print("PASS: the job barriers wait until their masks move, are woken by the ORs, lose no wake-up and time out; the patch "
          "hooks the three functions whole or not at all (%d broken variants and %d broken patches caught)"
          % (len(BROKEN), len(BROKEN_PATCH)))


if __name__ == "__main__":
    main()
