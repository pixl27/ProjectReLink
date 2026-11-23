/* Guest threads: one host thread each, with its own Ctx, TLS block and thread-local region.

   TPIDR_EL0 (TLS): a 4 KB block per thread, TP in its middle. TOTK's main reads its one thread-local at TP+0x10
   (local-exec, after AArch64's 16-byte TCB); the area below TP is zeroed room for whatever the SDK's musl keeps
   there. TPIDRRO_EL0 (thread-local region): 0x200 bytes; the SDK's nn::os keeps the current ThreadType* at
   +0x1F8, so lifted SDK code that reads it inline finds the same thread the HLE knows. */
#include "hos.h"
#include "hos_crash_record.h"
#include "hos_report_prefixes.h"
#include "log_redact.h"
#ifndef HOOKSHOT_ENV_H   /* (env_on: the switches' one rule) */
#include "../gpu/vk/compat/hookshot_env.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef TRACY_ENABLE
#define HT_WRAP_LOCKS   /* (HOOKSHOT_TRACY: this file's locks that have to wait are zones) */
#include "hos_tracy.h"
#endif
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION   /* (older Windows SDKs lack it: Windows 10 1803 and later take it) */
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#include <emmintrin.h>
#ifdef __EMSCRIPTEN__
#include <sched.h>
#include <stdatomic.h>
#include <unistd.h>
#endif

struct HosThread {
    Ctx ctx;
    uint64_t guest_tt, entry, arg, stack, stack_size, id;
    uint64_t tls_block, tlr;
    int priority, ideal_core;
    uint64_t core_mask;
    uint64_t tls_slots[HOS_TLS_SLOTS];
    char name[48];
    HANDLE handle;
    int started;
    int serial;   /* one of the frame's serial threads (hos_thread_serial), set with the name */
    /* the kernel side (hos_svc.c): this thread's Horizon handle and the event it blocks on */
    uint32_t kernel_handle;
    HANDLE wake;
    /* virtual cores (HOS_VCORES): the core held + 1 (0: none), the last one, a better waiter's request, queued, and
       whether it has run without one since an escape (vc_acquire; until its next block, hos_vcore_away, or the next
       hos_vcore_back); vc_starved: waiting longer than HOS_VCORE_STARVE_US (it goes first at the next release) */
    int vc_held, vc_last, vc_queued, vc_escaped, vc_starved;
    int vc_boost;   /* (priority steps better while it has the core it took back after the replay: hos_vcore_back_first) */
    volatile int vc_kick;
    int vc_prio_over;   /* (hookshot, HOS_VC_PRIO) the priority the virtual cores schedule it at, 0: its own */
    int64_t vc_since, vc_took;   /* (when it began to wait for a core; when it took the one it holds) */
    int bg_kind; int64_t bg_since;   /* (Hookshot, sprint 5: HOS_BG_THROTTLE, hos_vcore_checkpoint) */
    /* (HOS_CORE_COUNT, TICK-062 A10) calls that ask for the core number (HLE, svc 0x10), waits for a virtual core and their time */
    int64_t cc_hle, cc_svc, cc_vcw, cc_vcw_ns;
    /* (TICK-062 B2) hos_thread_of's per-thread cache */
    /* (hookshot, vc_bg_cap) a background thread: until when it waits for the cores held to be few enough, and until
       when it is then left alone */
    int64_t vc_cap_until, vc_free_until;
    /* (hookshot) a long wait's cause (HOS_VCORE_LOG_MS, vc_long_wait): when it took the core it holds and when a waiter asked
       for it; and, as a grant ends its wait, the thread that let the core go, its priority, how long it had held it and how
       long before it let go it was asked (-1: never) */
    int64_t vc_held_at, vc_kick_at;
    char vc_by[48];
    int vc_by_prio;
    int64_t vc_by_held, vc_by_asked;
    /* (hookshot) the frame trace (RX_VK_FRAME_TRACE, MkwVkHost thread_times): time waiting for a core and time blocked (from
       hos_vcore_away to hos_vcore_back, the core's wait taken out), in ns, counted while hos_fps_timing is on */
    volatile int64_t vc_wait_total, blocked_total;
    int64_t away_at;
    CONDITION_VARIABLE vc_cv;
    /* (hookshot) fibers: the context running now (NULL: ctx, the thread's own), the thread's host fiber once it
       switched to a guest fiber */
    Ctx *volatile run_ctx;
    void *host_fiber;
};

/* The SDK's own mutexes (InternalCriticalSection) tag a lock word with the owner's handle, read from
   ThreadType+0x1B0 through TLR+0x1F8 (sdk+0x13186c). Every thread gets a unique handle there. */
static void thread_gen_bump(void);
#define TT_HANDLE 0x1B0
static volatile LONG next_handle = 0x10000;
uint32_t hos_thread_handle(HosThread *t) { return t->kernel_handle; }

/* nnosInitialize gives the main thread the SDK's own ThreadType (it writes TLR+0x1F8): adopt it, so the HLE and the
   SDK agree on which object is the current thread. */
void hos_thread_adopt_guest(HosThread *t) {
    uint64_t tt = G64(t->tlr + 0x1F8);
    if (tt && tt != t->guest_tt) {
        t->guest_tt = tt;
        thread_gen_bump();
        G32(tt + TT_HANDLE) = t->kernel_handle;
    }
}
int hos_thread_prio(HosThread *t) { return t->priority; }

int hos_trace;
extern const uint64_t hos_thread_start_hook, hos_thread_exit_hook;   /* gen/hos_image.c */
static _Thread_local HosThread *cur;
static SRWLOCK threads_lock = SRWLOCK_INIT;
static HosThread *threads[1024];
static int nthreads;
static volatile LONG64 next_id = 0x50;

HosThread *hos_self(void) { return cur; }
/* (Hookshot native engine, host/hl_host.c) The calling host thread goes on as guest thread t, on t's own guest context
   (hos_ctx()), and back: returns who it was before. t's own host thread must stand still in host code meanwhile, as it
   does inside the bridge's wait; every per-thread thing the OS layer keeps is in the HosThread, so it follows. */
HosThread *hos_thread_act_as(HosThread *t) {
    HosThread *before = cur;
    cur = t;
    return before;
}
Ctx *hos_ctx(void) { return cur ? (cur->run_ctx ? cur->run_ctx : &cur->ctx) : NULL; }
uint64_t hos_thread_guest(HosThread *t) { return t->guest_tt; }
uint64_t hos_thread_id(HosThread *t) { return t->id; }
const char *hos_thread_name(HosThread *t) { return t->name; }
uint64_t *hos_tls_slots(HosThread *t) { return t->tls_slots; }
/* The core a thread runs on. Most of TOTK's threads have no ideal core (-2, "use default") and a one-core affinity
   mask, so the mask decides; the engine keys per-core resources (its command memory arenas, for one) on
   GetCurrentCoreNumber, so threads pinned to different cores must report different ones. */
/* HOS_PIN_BY_CORE=1 (experiment): every game thread runs on the host's logical core 2 x its core number (one per
   physical core on SMT machines), so threads the game puts on the same core never run at the same time, as on the
   Switch, where TOTK keeps per-core data without locks (both MacNativeGames and totk-pc-recomp found it relies on
   that); they are still time-sliced at any instruction, not scheduled by priority as Horizon does. */
static void pin_by_core(HosThread *t) {
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
    static int on = -1;
    if (on < 0) on = env_on("HOS_PIN_BY_CORE");
    if (on && t->handle) SetThreadAffinityMask(t->handle, (DWORD_PTR)1 << (2 * hos_thread_core(t)));
#else
    (void)t;
#endif
}
void hos_thread_affinity(HosThread *t, int ideal, uint64_t mask) {
    if (ideal >= 0) t->ideal_core = ideal;
    if (mask) t->core_mask = mask;
    pin_by_core(t);
}
uint64_t hos_thread_mask(HosThread *t) { return t->core_mask; }
void hos_thread_boost(int level) {
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
    static int off = -1;
    if (off < 0) off = env_on("HOS_NO_PRIORITIES");
    if (!off) SetThreadPriority(GetCurrentThread(), level >= 2 ? THREAD_PRIORITY_HIGHEST : THREAD_PRIORITY_ABOVE_NORMAL);
#else
    (void)level;
#endif
}

int hos_cores(void) {
    static int n;
    if (!n) {
        const char *e = getenv("HOS_CORES");
        n = e && atoi(e) >= 1 && atoi(e) <= 16 ? atoi(e) : 6;   /* (mkw-recomp: a Switch 2 application's six; Hookshot: up to 16) */
        if (!e && hos_game_cores()) n = hos_game_cores();   /* (totk_cores.c: the game set up for more cores) */
    }
    return n;
}
int hos_thread_core(HosThread *t) {
    if (!t) return 0;
    if (t->vc_held) return t->vc_held - 1;   /* (virtual cores: the one it holds) */
    const int n = hos_cores();
    if (t->ideal_core >= 0) return t->ideal_core % n;
    /* the mask's lowest core, wrapped (a thread pinned past the cores there are shares a core number: the extra
       BaseProc executors, totk_cores.c) */
    return t->core_mask ? (int)(__builtin_ctzll(t->core_mask) % (unsigned)n) : 0;
}
void hos_thread_stack(HosThread *t, uint64_t *addr, uint64_t *size) { *addr = t->stack; *size = t->stack_size; }

int hos_thread_priority(HosThread *t, int set, int value) {
    int old = t->priority;
    if (set) t->priority = value;
    return old;
}

#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
static SRWLOCK log_lock = SRWLOCK_INIT, log_write_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE log_cv = CONDITION_VARIABLE_INIT;
static char *log_q[2048];
static int log_head, log_count;
static volatile LONG log_async = -1;
static volatile DWORD log_writer;   /* log_main's thread */

static int log_take_locked(char **out) {
    int n = 0;
    while (log_count) {
        out[n++] = log_q[log_head];
        log_head = (log_head + 1) % 2048;
        log_count--;
    }
    return n;
}

/* (hookshot) A batch goes out in one write. stderr is unbuffered (hos_main.c), so each line was a write of its own: 30-90 a
   second in play, every one extending the log file. With the game on a slow disk (a SATA SSD, 90% full) those writes kept
   the disk busy: its 4 KB reads took 10-200 ms where they take 0.4 ms, and what the game read from it meanwhile (a page of
   its own code, a file of the game's) waited as long: the long frames of a session. */
static void log_write(char **lines, int n) {
    size_t total = 0;
    for (int i = 0; i < n; i++) total += strlen(lines[i]);
    char *all = n > 1 ? (char *)malloc(total + 1) : NULL;
    if (all) {
        size_t at = 0;
        for (int i = 0; i < n; i++) {
            const size_t len = strlen(lines[i]);
            memcpy(all + at, lines[i], len);
            at += len;
            free(lines[i]);
        }
        fwrite(all, 1, total, stderr);
        free(all);
    } else {   /* (one line, or no memory for the batch: line by line) */
        for (int i = 0; i < n; i++) {
            fputs(lines[i], stderr);
            free(lines[i]);
        }
    }
    fflush(stderr);
}

/* (hookshot) and the writer waits a little after the line it is woken for, so the lines of that time go out together: a few
   writes a second however many lines there are. A queue half full goes out at once, before a burst can fill it (a full
   queue has the thread that logs write the line itself). HOS_LOG_BATCH_MS=<ms> (default 250, at most 2000; 0: the lines
   there are at each wake, as before). Nothing waiting is lost: every way out writes the queue first (log_flush,
   log_crash_filter), and the writer holds no lock while it waits */
enum { LOG_BATCH_FULL = 1024 };
static DWORD log_batch_ms(void) {
    static int ms = -1;
    if (ms < 0) {
        const char *e = getenv("HOS_LOG_BATCH_MS");
        const int v = e && *e ? atoi(e) : 250;
        ms = v < 0 ? 0 : v > 2000 ? 2000 : v;
    }
    return (DWORD)ms;
}

static DWORD WINAPI log_main(void *p) {
    (void)p;
    static char *batch[2048];
    log_writer = GetCurrentThreadId();
    for (;;) {
        AcquireSRWLockExclusive(&log_lock);
        while (!log_count) SleepConditionVariableSRW(&log_cv, &log_lock, INFINITE, 0);
        /* (log_queue wakes it at each line: until the time is up or the queue half full) */
        const ULONGLONG until = GetTickCount64() + log_batch_ms();
        for (ULONGLONG now; log_count < LOG_BATCH_FULL && (now = GetTickCount64()) < until;)
            SleepConditionVariableSRW(&log_cv, &log_lock, (DWORD)(until - now), 0);
        const int n = log_take_locked(batch);
        AcquireSRWLockExclusive(&log_write_lock);
        ReleaseSRWLockExclusive(&log_lock);
        log_write(batch, n);
        ReleaseSRWLockExclusive(&log_write_lock);
    }
}

/* the queued lines straight to the handle (crash paths: no CRT stream, no write lock) */
static void log_drain_raw(void) {
    if (!TryAcquireSRWLockExclusive(&log_lock)) return;
    HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
    while (log_count) {
        const char *line = log_q[log_head];
        DWORD written;
        WriteFile(err, line, (DWORD)strlen(line), &written, NULL);
        log_head = (log_head + 1) % 2048;
        log_count--;
    }
    ReleaseSRWLockExclusive(&log_lock);
}

static void log_flush(void) {
    if (log_async <= 0) return;
    /* (a crash report on the writer itself, in its own write: it holds log_write_lock) */
    if (GetCurrentThreadId() == log_writer) { log_drain_raw(); return; }
    static char *batch[2048];
    AcquireSRWLockExclusive(&log_write_lock);
    AcquireSRWLockExclusive(&log_lock);
    const int n = log_take_locked(batch);
    ReleaseSRWLockExclusive(&log_lock);
    log_write(batch, n);
    ReleaseSRWLockExclusive(&log_write_lock);
}

static LPTOP_LEVEL_EXCEPTION_FILTER log_prev_filter;

static LONG WINAPI log_crash_filter(EXCEPTION_POINTERS *e) {
    if (GetCurrentThreadId() == log_writer) log_drain_raw();
    else {
        /* the writer may be in the middle of a batch: it finishes first (at most 2 s: a write stuck on the disk), then the
           rest of the queue goes out (before: the queue was left unwritten whenever the writer was busy) */
        int got = 0;
        for (int i = 0; i < 2000 && !(got = TryAcquireSRWLockExclusive(&log_write_lock)); i++) Sleep(1);
        log_drain_raw();
        if (got) ReleaseSRWLockExclusive(&log_write_lock);
    }
    return log_prev_filter ? log_prev_filter(e) : EXCEPTION_CONTINUE_SEARCH;
}

/* Lines are written by log_main, not by the thread that logs: a write that blocks (a console; the log file while the
   disk is busy: the render thread was seen held 140-310 ms in one line's write) holds no game thread. Nothing queued is
   lost when the game ends or crashes: hos_fatal, report_crash/report_fatal (mkw_glue.c), log_crash_filter,
   svcExitProcess, the window's close and the process's exit (atexit) write the queue first. HOS_LOG_SYNC=1: each line
   written by the thread that logs it, as files and pipes were before */
static int log_queue(const char *line) {
    if (log_async < 0) {
        AcquireSRWLockExclusive(&log_lock);
        if (log_async < 0) {
            int on = !env_on("HOS_LOG_SYNC");
            if (on && !CreateThread(NULL, 0, log_main, NULL, 0, NULL)) on = 0;
            if (on) log_prev_filter = SetUnhandledExceptionFilter(log_crash_filter);
            if (on) atexit(log_flush);
            log_async = on;
        }
        ReleaseSRWLockExclusive(&log_lock);
    }
    if (!log_async) return 0;
    char *s = _strdup(line);
    if (!s) return 0;
    AcquireSRWLockExclusive(&log_lock);
    if (log_count < 2048) {
        log_q[(log_head + log_count) % 2048] = s;
        log_count++;
        ReleaseSRWLockExclusive(&log_lock);
        WakeConditionVariable(&log_cv);
        return 1;
    }
    ReleaseSRWLockExclusive(&log_lock);
    free(s);
    log_flush();
    return 0;
}
#else
static int log_queue(const char *line) { (void)line; return 0; }
static void log_flush(void) {}
#endif

void hos_log_flush(void) { log_flush(); }

/* HOS_LOG_QPC=1: every log line ends with " @qpc <ticks>", the QueryPerformanceCounter value when it was logged, the clock
   PresentMon's TimeInQPC column counts in: a line then falls on the frames of a capture. Off by default, the lines as before */
static long long log_qpc_now(void) {
    static int on = -1;
    if (on < 0) on = env_on("HOS_LOG_QPC");
    if (!on) return -1;
    LARGE_INTEGER q;
    QueryPerformanceCounter(&q);
    return (long long)q.QuadPart;
}

/* (hookshot) A line's level as the log writes it: none for info, so the lines most of the log is stay as they were; "[trace] "
   after the thread's tag for the rest, which the launcher's log reads to show info and above unless asked for more */
static const char *log_level_tag(int level) {
    switch (level) {
    case HOS_LOG_ERROR: return "[error] ";
    case HOS_LOG_WARN: return "[warn] ";
    case HOS_LOG_DEBUG: return "[debug] ";
    case HOS_LOG_TRACE: return "[trace] ";
    default: return "";
    }
}

static void hos_vlog(int level, const char *fmt, va_list ap) {
    char buf[1024], raw[1152], line[2304];   /* (room for the name, the level, the message and the stamp; line: redacted) */
    const long long qpc = log_qpc_now();
    vsnprintf(buf, sizeof buf, fmt, ap);
    if (!strncmp(buf, HOS_REPORT_DEVICE_LOST_PREFIX, sizeof HOS_REPORT_DEVICE_LOST_PREFIX - 1))
        hos_crash_record_fatal(NULL, "device lost");
    else if (!strncmp(buf, HOS_REPORT_FATAL_PREFIX, sizeof HOS_REPORT_FATAL_PREFIX - 1))
        hos_crash_record_fatal(NULL, strstr(buf, "VK_ERROR_DEVICE_LOST") ? "device lost" : "Vulkan fatal");
    if (qpc >= 0)
        snprintf(raw, sizeof raw, "[hos %s] %s%s @qpc %lld\n", cur ? cur->name : "host", log_level_tag(level), buf, qpc);
    else
        snprintf(raw, sizeof raw, "[hos %s] %s%s\n", cur ? cur->name : "host", log_level_tag(level), buf);
    /* (hookshot) the player's user name out before anything has the line: players send their log in (host/log_redact.h) */
    hos_log_redact(line, sizeof line, raw);
    extern volatile int hos_control_on;   /* (control.c: log lines for its subscribers) */
    if (hos_control_on) {
        void hos_control_log(const char *line);
        hos_control_log(line);
    }
    if (log_queue(line)) return;
    fputs(line, stderr);
    fflush(stderr);
}

void hos_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    hos_vlog(HOS_LOG_INFO, fmt, ap);
    va_end(ap);
}

void hos_log_at(int level, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    hos_vlog(level, fmt, ap);
    va_end(ap);
}

void hos_fatal(Ctx *c, const char *fmt, ...) {
    if (!c) c = hos_ctx();
    hos_crash_record_fatal(c ? c->x : NULL, "hos_fatal");
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log_flush();
    char clean[2200];   /* (hookshot) without the player's user name, as every other line (host/log_redact.h) */
    hos_log_redact(clean, sizeof clean, buf);
    fprintf(stderr, "[hos %s] FATAL: %s", cur ? cur->name : "host", clean);
    if (c) fprintf(stderr, " (lr %#llx sp %#llx)", (unsigned long long)c->x[30], (unsigned long long)c->x[31]);
    fprintf(stderr, "\n");
    fflush(stderr);
    void hos_backtrace(void), hos_hostcall_stats(void), hl_trace_on_crash(Ctx *, uint64_t);
    hos_backtrace();
    hl_trace_on_crash(c, 0);   /* (HL_TRACE: hl_trace.c) */
    hos_hostcall_stats();
    ExitProcess(3);
}

/* Traps from the lifted code (rt/a64_rt.c): name the thread, then stop. */
void a64_on_trap(Ctx *c, uint64_t pc, uint32_t insn, const char *why) {
    hos_fatal(c, "%s at %#llx (insn %08x)", why, (unsigned long long)pc, insn);
}

static void setup_guest(HosThread *t) {
    t->tls_block = hos_sys_alloc(0x1000, 0x1000);
    t->tlr = hos_sys_alloc(0x200, 0x200);
    G64(t->tlr + 0x1F8) = t->guest_tt;
    t->kernel_handle = (uint32_t)InterlockedIncrement(&next_handle);
    G32(t->guest_tt + TT_HANDLE) = t->kernel_handle;
    /* (hookshot) SDK 22's InternalCriticalSection::IsLockedByCurrentThread reads the owner handle from TLR+0x110
       directly (SDK 21: ThreadType+0x1B0 through TLR+0x1F8). Left 0, every free mutex looked held by the caller
       and the first heap lock aborted in pthread_mutex_lock. */
    G32(t->tlr + 0x110) = t->kernel_handle;
    t->wake = CreateEventA(NULL, FALSE, FALSE, NULL);   /* auto-reset: one signal, one wake-up */
    Ctx *c = &t->ctx;
    memset(c, 0, sizeof *c);
    c->tpidr_el0 = t->tls_block + 0x800;
    c->tpidrro_el0 = t->tlr;
    c->thread = t;
}

static void track(HosThread *t) {
    AcquireSRWLockExclusive(&threads_lock);
    if (nthreads < 1024) threads[nthreads++] = t;
    thread_gen_bump();
    ReleaseSRWLockExclusive(&threads_lock);
}

/* Host handles of every running guest thread, for the profiler (hos_prof.c). */
int hos_thread_handles(HANDLE *out, const char **names, int max) {
    int n = 0;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads && n < max; i++)
        if (threads[i]->handle) { out[n] = threads[i]->handle; names[n] = threads[i]->name; n++; }
    ReleaseSRWLockShared(&threads_lock);
    return n;
}

/* (Hookshot, hl_mem.c) The guest stacks as {start, end}: every guest thread's (the game gives them, mostly from its heaps)
   and every fiber's that has not returned. At most max; the count is returned. Copied under the lists' locks only. */
static int fiber_stacks(uint64_t (*out)[2], int max);
int hos_guest_stacks(uint64_t (*out)[2], int max) {
    int n = 0;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads && n < max; i++)
        if (threads[i]->stack_size) out[n][0] = threads[i]->stack, out[n][1] = threads[i]->stack + threads[i]->stack_size, n++;
    ReleaseSRWLockShared(&threads_lock);
    return n + fiber_stacks(out + n, max - n);
}

/* (mkw-recomp) Every guest thread's name and guest registers as of its last call out of lifted code, with its
   frame chain from x29: what each thread of a stalled game waits in (mkw_glue.c, HOS_STATS_AFTER). */
void hos_thread_dump(void) {
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads; i++) {
        const Ctx *c = &threads[i]->ctx;
        char bt[400];
        int n = 0;
        uint64_t fp = c->x[29];
        for (int k = 0; k < 12 && fp && !(fp & 15) && fp < HOS_GUEST_SPACE && n < (int)sizeof bt - 24; k++) {
            MEMORY_BASIC_INFORMATION mbi;
            if (!VirtualQuery(GP(fp), &mbi, sizeof mbi) || mbi.State != MEM_COMMIT) break;
            n += snprintf(bt + n, sizeof bt - n, " %#llx", (unsigned long long)G64(fp + 8));
            const uint64_t next = G64(fp);
            if (next <= fp) break;
            fp = next;
        }
        bt[n] = 0;
        hos_log("thread %-24s lr %#llx sp %#llx x0 %#llx; frames%s", threads[i]->name, (unsigned long long)c->x[30],
                (unsigned long long)c->x[31], (unsigned long long)c->x[0], bt);
        if (hos_game_is_mkw() && c->x[30] == 0x825c6844ULL) continue;   /* (MKW's idle thread loop) */
        /* registers (current at the call out), with the text a register (or the word it points to) leads to */
        for (int r = 0; r < 29; r++) {
            uint64_t v = c->x[r], s = 0;
            MEMORY_BASIC_INFORMATION mbi;
            for (int hop = 0; hop < 2 && v > 0x100000 && v < HOS_GUEST_SPACE - 64; hop++) {
                if (!VirtualQuery(GP(v), &mbi, sizeof mbi) || mbi.State != MEM_COMMIT) break;
                const uint8_t *p = (const uint8_t *)GP(v);
                int k = 0;
                while (k < 48 && p[k] >= 0x20 && p[k] < 0x7f) k++;
                if (k >= 6 && (p[k] == 0 || k == 48)) { s = v; break; }
                v = G64(v);
            }
            if (s) hos_log("thread %s:   x%d %#llx -> \"%.48s\"", threads[i]->name, r, (unsigned long long)c->x[r],
                           (const char *)GP(s));
        }
    }
    ReleaseSRWLockShared(&threads_lock);
}

/* HOS_FP_REPORT=1 (debugging): the FPCR / FPSR of the busiest game threads, with the thread stats */
int hos_thread_fp_report(char *out, int cap) {
    int len = 0;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads && len < cap - 64; i++) {
        const char *nm = threads[i]->name;
        if (strstr(nm, "Main") || strstr(nm, "Worker") || strstr(nm, "Executor"))
            len += snprintf(out + len, cap - len, " %s fpcr %llx fpsr %llx;", nm,
                            (unsigned long long)threads[i]->ctx.fpcr, (unsigned long long)threads[i]->ctx.fpsr);
    }
    ReleaseSRWLockShared(&threads_lock);
    return len;
}

HosThread *hos_thread_by_handle(uint32_t h) {
    HosThread *r = NULL;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads; i++)
        if (threads[i]->kernel_handle == h) { r = threads[i]; break; }
    ReleaseSRWLockShared(&threads_lock);
    return r;
}

/* Block on the thread's wake event (the kernel layer's lock must not be held). */
/* (HOS_CHAIN: the kernel's waits, svc mutexes, condition variables and the address arbiter, end in a wake of the
   thread itself: one key per thread, bit 62: never a guest address) */
static uint64_t wake_chain_key(HosThread *t) { return 1ULL << 62 | (uint64_t)(uintptr_t)t; }
void hos_thread_wait(HosThread *t, int64_t timeout_ns, int *timed_out) {
    DWORD ms = timeout_ns < 0 ? INFINITE : (DWORD)((timeout_ns + 999999) / 1000000);
    HOS_CHAIN(HOS_CHAIN_WAIT, wake_chain_key(t));
#ifdef TRACY_ENABLE
    HT_WAIT(tz, "kernel wait");   /* (svc mutexes, condition variables, the address arbiter: host/hos_svc.c) */
    HT_ZONE_TEXTF(tz, "lr %#llx", (unsigned long long)(hos_ctx() ? hos_ctx()->x[30] : 0));
#endif
    const int away = hos_vcore_away();
    *timed_out = WaitForSingleObject(t->wake, ms) == WAIT_TIMEOUT;
    if (away) hos_vcore_back();
    HOS_CHAIN(HOS_CHAIN_WOKE, wake_chain_key(t));
}
void hos_thread_wake(HosThread *t) {
    HOS_CHAIN(HOS_CHAIN_SIGNAL, wake_chain_key(t));
    SetEvent(t->wake);
}

/* TICK-062 B2: hos_thread_of is a linear scan of every thread under a shared lock. HOS_THREAD_OF_CACHE=1: each thread keeps
   a few (ThreadType, thread) pairs, valid while thread_gen has not changed. thread_gen is bumped by everything that can
   change what the scan returns: a thread added (track) or removed (hos_thread_destroy), or a guest_tt rewritten (adopt).
   A reader takes the generation BEFORE the scan and stores the pair under it: a change during the scan leaves the pair
   stale at once, never served. The answer is then the scan's own (the first matching thread in the table), and a cached
   pointer to a destroyed thread cannot be served (destroy bumps the generation before the free). */
static volatile LONG thread_gen;
static void thread_gen_bump(void) { InterlockedIncrement(&thread_gen); }
static HosThread *thread_of_scan(uint64_t guest_tt) {
    HosThread *r = NULL;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads; i++)
        if (threads[i]->guest_tt == guest_tt) { r = threads[i]; break; }
    ReleaseSRWLockShared(&threads_lock);
    return r;
}
HosThread *hos_thread_of(uint64_t guest_tt) {
    static int on = -1;
    if (on < 0) on = getenv("HOS_THREAD_OF_CACHE") && atoi(getenv("HOS_THREAD_OF_CACHE")) != 0;
    if (!on) return thread_of_scan(guest_tt);
    static _Thread_local struct { uint64_t tt; HosThread *t; LONG gen; } cache[8];
    const unsigned i = (unsigned)((guest_tt >> 4) * 0x9E3779B1u >> 7) & 7;
    const LONG g = *(volatile LONG *)&thread_gen;
    if (cache[i].tt == guest_tt && cache[i].gen == g && cache[i].t) return cache[i].t;
    HosThread *r = thread_of_scan(guest_tt);
    cache[i].tt = guest_tt, cache[i].t = r, cache[i].gen = g;
    return r;
}

Ctx *hos_boot_main_thread(uint64_t stack_size) {
    HosThread *t = (HosThread *)calloc(1, sizeof *t);
    t->guest_tt = hos_sys_alloc(0x400, 0x10);      /* a ThreadType for the main thread; the SDK never made one */
    t->stack_size = stack_size;
    t->stack = hos_sys_alloc(stack_size, 0x1000);
    t->id = InterlockedIncrement64(&next_id);
    t->priority = 44;                              /* main thread priority in TOTK's NPDM range */
    t->ideal_core = 0;
    t->started = 1;
    snprintf(t->name, sizeof t->name, "MainThread");
    __atomic_store_n(&t->serial, 1, __ATOMIC_RELAXED);
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &t->handle, 0, FALSE,
                    DUPLICATE_SAME_ACCESS);   /* a real handle, for the profiler */
    SetThreadDescription(GetCurrentThread(), L"MainThread");   /* (HOS_SAMPLE finds threads by it) */
#ifdef TRACY_ENABLE
    HT_THREAD_NAME("MainThread");
#endif
    hos_thread_pin(t->handle, t->name);
    setup_guest(t);
    t->ctx.x[31] = (t->stack + stack_size) & ~15ULL;
    cur = t;
    track(t);
    hos_vcore_back();   /* (virtual cores: before any guest code) */
    return &t->ctx;
}

/* (hookshot) The host's MXCSR follows the guest's FPCR.FZ: results flushed to zero (FTZ) and subnormal operands read as zeros
   (DAZ), which is what Arm's flush-to-zero mode does. The hot functions' floating point runs on the host's own
   instructions (third_party/a64-kit/lift/a64_lift.py, NATIVE_FP), and those flush as the guest expects only with both
   bits set. The exact helpers keep their values under them: with FZ they flush what they compute themselves. Called by
   lifted code after every MSR FPCR, and at a thread's and a fiber's start (MXCSR is a thread's, and a fiber's with
   FIBER_FLAG_FLOAT_SWITCH). HOS_FPCR_SYNC=0: MXCSR left alone, as before */
void hos_fpcr_sync(Ctx *c) {
#if defined(__x86_64__) || defined(_M_X64)
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("HOS_FPCR_SYNC");
        on = !(e && e[0] == '0');
    }
    if (!on) return;
    const unsigned want = ((uint32_t)c->fpcr & (1u << 24)) ? 0x8040u : 0u;
    const unsigned now = _mm_getcsr();
    if ((now & 0x8040u) != want) _mm_setcsr((now & ~0x8040u) | want);
#else
    (void)c;
#endif
}

static DWORD WINAPI thread_main(void *p) {
    HosThread *t = (HosThread *)p;
    cur = t;
#ifdef TRACY_ENABLE
    HT_THREAD_NAME(t->name);   /* (the guest thread's whole name: the system's keeps 15 characters on Linux) */
#endif
    Ctx *c = &t->ctx;
    hos_fpcr_sync(c);
    c->x[0] = t->arg;
    c->x[30] = 0;
    c->x[31] = (t->stack + t->stack_size) & ~15ULL;
    A64Fn f = a64_lookup(t->entry);
    if (!f) hos_fatal(c, "thread %s: entry %#llx is not a function", t->name, (unsigned long long)t->entry);
    hos_vcore_back();   /* (virtual cores: before any guest code) */
    if (hos_trace) hos_log("thread start: entry %#llx arg %#llx", (unsigned long long)t->entry, (unsigned long long)t->arg);
    /* What the SDK's own thread entry does before the thread function (see build.py thread_hooks): nnmusl's start
       hook gives the thread its TLS (thread pointer with the DTV at TP+8, every module's TLS block) and leaves the
       thread pointer at TLR+0x1F0. Without it every thread would share one DTV, and so every thread_local. */
    uint64_t hook = hos_thread_start_hook ? G64(hos_thread_start_hook) : 0;
    if (hook) {
        c->x[0] = t->guest_tt;
        a64_icall(c, hook);
        if (G64(t->tlr + 0x1F0)) c->tpidr_el0 = G64(t->tlr + 0x1F0);
        if (hos_trace || env_on("HOS_TLS_TRACE"))
            hos_log("thread %s: TLS hook gave TP %#llx, DTV %#llx; ideal core %d mask %#llx priority %d", t->name,
                    (unsigned long long)c->tpidr_el0, (unsigned long long)G64(c->tpidr_el0 + 8), t->ideal_core,
                    (unsigned long long)t->core_mask, t->priority);
        c->x[31] = (t->stack + t->stack_size) & ~15ULL;
    }
    c->x[0] = t->arg;
    c->x[30] = 0;
    f(c);
    hook = hos_thread_exit_hook ? G64(hos_thread_exit_hook) : 0;
    if (hook) {
        c->x[0] = t->guest_tt;
        a64_icall(c, hook);
    }
    if (hos_trace) hos_log("thread exit");
    hos_vcore_away();
    return 0;
}

HosThread *hos_thread_create(uint64_t guest_tt, uint64_t entry, uint64_t arg, uint64_t stack, uint64_t stack_size,
                             int priority, int core) {
    HosThread *t = (HosThread *)calloc(1, sizeof *t);
    t->guest_tt = guest_tt;
    t->entry = entry;
    t->arg = arg;
    t->stack = stack;
    t->stack_size = stack_size;
    t->priority = priority;
    t->ideal_core = core;
    t->id = InterlockedIncrement64(&next_id);
    snprintf(t->name, sizeof t->name, "Thread_%llx", (unsigned long long)entry);
    setup_guest(t);   /* FPCR starts at 0, as the kernel does it; TOTK sets FZ itself (nnMain+0x474, thread entries) */
    track(t);
    return t;
}

void hos_thread_start(HosThread *t) {
    if (t->started) return;
    t->started = 1;
    /* Host stack: lifted code nests a C call per guest call, so give it plenty. */
    t->handle = CreateThread(NULL, 64 << 20, thread_main, t, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    if (!t->handle) hos_fatal(NULL, "CreateThread failed for %s", t->name);
    if (t->name[0]) {   /* (named before it started: the OS name too, for profilers and HOS_SAMPLE) */
        wchar_t w[48];
        MultiByteToWideChar(CP_UTF8, 0, t->name, -1, w, 48);
        SetThreadDescription(t->handle, w);
    }
    hos_thread_pin(t->handle, t->name);
    pin_by_core(t);
#ifdef __EMSCRIPTEN__
    { void web_thread_named(HANDLE t, const char *name); web_thread_named(t->handle, t->name); }
#endif
}

void hos_thread_join(HosThread *t) {
    if (!t->handle) return;
#ifdef TRACY_ENABLE
    HT_WAIT(tz, "thread join");
    HT_ZONE_TEXTF(tz, "%s", t->name);
#endif
    const int away = hos_vcore_away();
    /* (EvilOOT, 11-07: a freeze after Load, one thread joining another for good: a join of over 10 s says who joins whom, once) */
    if (WaitForSingleObject(t->handle, 10000) == WAIT_TIMEOUT) {
        HosThread *self = hos_self();
        hos_log("thread %s: joining %s for 10 s", self ? hos_thread_name(self) : "?", t->name);
        WaitForSingleObject(t->handle, INFINITE);
    }
    if (away) hos_vcore_back();
}

/* (hookshot) The frame trace's main-thread columns (RX_VK_FRAME_TRACE, through MkwVkHost thread_times): the thread named name's
   time so far, in ns: [0] waiting for a virtual core, [1] blocked (in a wait, a sleep or a join, the core given up), both counted
   only while hos_fps_timing is on, and [2] its CPU time (the system's, GetThreadTimes). 0 when no thread has that name */
int hos_thread_times_of(const char *name, int64_t out[3]) {
    int found = 0;
    out[0] = out[1] = out[2] = 0;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads && !found; i++) {
        HosThread *t = threads[i];
        if (strcmp(t->name, name)) continue;
        found = 1;
        out[0] = t->vc_wait_total, out[1] = t->blocked_total;
        FILETIME c, e, k, u;
        if (t->handle && GetThreadTimes(t->handle, &c, &e, &k, &u))
            out[2] = (int64_t)(((uint64_t)k.dwHighDateTime << 32 | k.dwLowDateTime) + ((uint64_t)u.dwHighDateTime << 32 | u.dwLowDateTime)) * 100;
    }
    ReleaseSRWLockShared(&threads_lock);
    return found;
}

void hos_thread_destroy(HosThread *t) {
    AcquireSRWLockExclusive(&threads_lock);
    for (int i = 0; i < nthreads; i++)
        if (threads[i] == t) { threads[i] = threads[--nthreads]; break; }
    thread_gen_bump();
    ReleaseSRWLockExclusive(&threads_lock);
    if (t->handle) CloseHandle(t->handle);
    if (t->wake) CloseHandle(t->wake);
    free(t);
}

/* The frame's serial threads: OoT's game thread (sead::MainThread, the boot thread's MainThread before the game names
   it) and its Presentation Thread. A frame waits on each in turn, so a wait of theirs that ends late ends the frame late;
   the host keeps their short waits short (sleep_ns's pollers, the spin before a wait sleeps: hos_spin_ns). By name, as
   the game names them: kept in the thread's record when it is named, not read from the name while it may change. */
static int serial_name(const char *name) { return strstr(name, "MainThread") || !strcmp(name, "Presentation Thread"); }
int hos_thread_serial(const HosThread *t) { return t && __atomic_load_n(&t->serial, __ATOMIC_RELAXED); }

/* The processors this process may run on (its affinity mask, as the game starts with it): Windows with one processor
   group; 0 = not known (more than one group, or elsewhere). Up to SHORT_SLEEP_TIMER_THREADS of them, the host keeps the
   behaviour measured best there (sleep_ns's timer, hos_spin_ns's one spin). */
enum { SHORT_SLEEP_TIMER_THREADS = 6 };
static int process_threads(void) {
    int threads = 0;
#if defined(_WIN32)   /* (measured on Windows only) */
    /* (with more than one processor group the mask is one group's, not the process's: not known then) */
    DWORD_PTR proc = 0, sys = 0;
    if (GetActiveProcessorGroupCount() == 1 && GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys))
        threads = __builtin_popcountll((uint64_t)proc);
#endif
    return threads;
}

/* How long a wait spins before it sleeps (nn_os_sync.c's events, queues and mutexes; hos_svc.c's mutexes), in ns, for the
   calling thread: HOS_SPIN_US (20 us) on the frame's serial threads, HOS_SPIN_OTHER_US (5 us, never more than HOS_SPIN_US)
   on the others, where there are more than SHORT_SLEEP_TIMER_THREADS processors. With 6 or fewer (or not known), every
   thread keeps HOS_SPIN_US unless HOS_SPIN_OTHER_US is set: there the short spin saved no CPU and cost 1.6 % fps (6 pairs,
   Hyrule, 6 of 16 processors), as a 24-thread run with only 2 pollers did (1.7 %). A sleeper runs again 6.6-8.4 us after its signal (HOS_WAKE_LAT, Hyrule), so a spin much longer than that
   mostly burns its thread's time: 90 % of the spin's 15 % of a core in Hyrule was on the job workers and the actor
   executors. 5 us for every thread cut the process's CPU a frame 3.3 % at the same fps (6 pairs, Hyrule, 16 threads); the
   serial threads keep 20 us, as their wait ends are on the frame's path. HOS_SPIN_OTHER_US=20: as before, one spin for all;
   HOS_SPIN_US=0: no spin. */
int64_t hos_spin_ns(void) {
    static int64_t both = -1;   /* (the serial threads' ns << 32 | the others': one atomic word, chosen once) */
    int64_t b = __atomic_load_n(&both, __ATOMIC_RELAXED);
    if (b < 0) {
        const char *e = getenv("HOS_SPIN_US"), *o = getenv("HOS_SPIN_OTHER_US");
        int64_t s = e ? atoll(e) : 20, other = o ? atoll(o) : process_threads() > SHORT_SLEEP_TIMER_THREADS ? 5 : s;   /* (us, clamped before they become ns) */
        s = (s < 0 ? 0 : s > 1000000 ? 1000000 : s) * 1000;
        other = (other < 0 ? 0 : other > 1000000 ? 1000000 : other) * 1000;
        other = other > s ? s : other;
        b = s << 32 | other;
        __atomic_store_n(&both, b, __ATOMIC_RELAXED);
    }
    return hos_thread_serial(hos_self()) ? b >> 32 : b & 0xFFFFFFFF;
}

static void vc_name_prio(HosThread *t);   /* (below) */
void hos_thread_set_name(HosThread *t, const char *name) {
    snprintf(t->name, sizeof t->name, "%s", name);
    __atomic_store_n(&t->serial, serial_name(t->name), __ATOMIC_RELAXED);   /* (atomic: SetThreadName may name another thread) */
    vc_name_prio(t);
#ifdef TRACY_ENABLE
    if (t == cur) HT_THREAD_NAME(t->name);   /* (another thread's: Tracy shows the system's name, set below) */
#endif
    if (t->handle) {
        wchar_t w[48];
        MultiByteToWideChar(CP_UTF8, 0, t->name, -1, w, 48);
        SetThreadDescription(t->handle, w);
        hos_thread_pin(t->handle, t->name);
    }
}

/* HOS_PIN=<name part>=<hex mask>,...[,*=<hex mask>] (experiment): host CPU affinity by thread name (game threads as
   they are named or started, host threads as they register); "*" for every other named thread. The first matching
   entry wins. (2026-10-20: the castle's MainThread and ModuleSystem workers each on a physical core of their own, the
   rest elsewhere, was slower than Windows' own scheduling: 63-68 against 69-71 fps.) */
/* HOS_PRIO=<name part>=<level>,... (experiment, same matching): host thread priority, -2 (lowest) to 2 (highest).
   (10-20: MainThread, ModuleSystem workers and BaseProc executors above normal: no change at the castle) */
typedef struct { char pat[40]; long long value; } NameSpec;
static int parse_name_specs(const char *e, NameSpec *out, int max, int base) {
    int n = 0;
    while (e && *e && n < max) {
        const char *eq = strchr(e, '='), *end = strchr(e, ',');
        if (!end) end = e + strlen(e);
        if (!eq || eq > end) break;
        snprintf(out[n].pat, sizeof out[n].pat, "%.*s", (int)(eq - e), e);
        out[n++].value = strtoll(eq + 1, NULL, base);
        e = *end ? end + 1 : end;
    }
    return n;
}
static const NameSpec *match_name_spec(const NameSpec *s, int n, const char *name) {
    for (int i = 0; i < n; i++)
        if (strcmp(s[i].pat, "*") ? strstr(name, s[i].pat) != NULL : 1) return &s[i];
    return NULL;
}
/* (hookshot) HOS_VC_PRIO=<name part>=<priority>,... (experiment): the priority the virtual cores schedule a thread at
   (who waits for a core, who is asked to give one up), the game's own priority left as it is. OoT's OptimizeChangeBanc
   (a background thread the main thread waits for about every 40 frames on the way to Saria) waited for a core 28% of
   its time, asked to give it up by the frame's threads ~17000 times a second (hs173) */
static void vc_name_prio(HosThread *t) {
    static NameSpec specs[16];
    static int n = -1;
    if (n < 0) n = parse_name_specs(getenv("HOS_VC_PRIO"), specs, 16, 10);
    const NameSpec *p = n ? match_name_spec(specs, n, t->name) : NULL;
    t->vc_prio_over = p ? (int)p->value : 0;
    if (p) hos_log("vcores: %s scheduled at priority %d (its own %d)", t->name, t->vc_prio_over, t->priority);
}
/* The last physical core (its logical processors' mask), kept for the Vulkan back end's replay thread (gpu/vk
   vk_replay.cpp pins it there): the game's threads run on the others. The replay is on the frame's critical path
   (MKW waits for the last frame's GPU work before it submits the next), and on a core shared with game threads
   it ran ~1.7x slower than alone. Only on machines with 8+ logical processors; HOS_RESERVE_CORE=0: none.
   (hookshot) On a CPU with cores of more than one kind, the last core of the fastest kind: the last core of all is an
   efficiency core on Intel's hybrid CPUs (an i7-12700K's logical processors 16-19), the slowest place for the replay
   (an i7-13700K's Kokiri Forest: 48.7 fps with it there, 60 on a performance core). EfficiencyClass is higher the faster
   the core, and the same for every core of a CPU of one kind (the last core, as before). Only with 4+ cores of the
   fastest kind (8+ logical processors always have them on a CPU of one kind): a laptop's 2 performance cores and 8
   efficiency ones (i5-1235U) would keep one of the two from the game; then none is kept. The calling thread's
   processor group's cores: the masks the affinity calls take are its group's. On Linux the same choice from
   /sys/devices/cpu_core/cpus (Intel's hybrid CPUs; last_fastest_core below); without that file, the last core, as before.
   Every way, only among the cores the process may run on wholly (its affinity mask: process_cpus). */
/* (hookshot) The cores that share the largest L3, on a CPU whose cores do not all share one of the same size: AMD's X3D CPUs
   with two CCDs (7900X3D, 7950X3D, 9900X3D, 9950X3D) have the 3D V-Cache on one CCD only (96 MB against 32 MB). Windows
   moves a game there only when its Game Bar knows the program as a game; this one it does not know, so its threads ran on
   both, and the replay's reserved core (the last one) is on the CCD without the cache. On a 9950X3D (SMT off), Castle Town's
   market (2,300 draws): the replay 11.5-12.3 ms a frame and 57-58 fps as it was (the slowest two seconds 54), 10.1-10.5 ms and
   59.9 fps (58.7) with the game's threads kept to the first CCD. So they are kept to those cores. 0 for none: one L3 for all, L3s
   of one size (a 7950X, both CCDs alike), L3s that do not all share as many logical processors (a hybrid CPU's two kinds of
   core: a Ryzen AI 9 HX 370 has 4 Zen 5 cores on a 16 MB L3 and 8 Zen 5c on an 8 MB one, and the game is not squeezed onto the 4;
   an X3D's CCDs are alike but for the cache), or fewer than 8 cores on the largest (the 12-core 7900X3D and 9900X3D have 6 there:
   not measured, left as they were). Measured with SMT off only: with it the 8 cores are 16 logical processors.
   HOS_CACHE_CORES=0: none. Windows only, as the reserved core's choice above: Linux keeps the last core, which on these CPUs is
   on the CCD without the cache.
   (the choice: each L3's logical processor mask and size, n of them, and each core's mask, ncores of them.
   tests/test_reserved_core.py) */
static uint64_t pick_cache_cores(const uint64_t *masks, const uint64_t *sizes, int n, const uint64_t *cores, int ncores) {
    uint64_t best = 0, most = 0, least = ~0ull;
    int width = 0, alike = 1;   /* (how many logical processors an L3 is shared by, and whether all are shared by as many) */
    for (int i = 0; i < n; i++) {
        if (!masks[i] || !sizes[i]) continue;
        if (!width) width = __builtin_popcountll(masks[i]);
        else if (__builtin_popcountll(masks[i]) != width) alike = 0;
        if (sizes[i] > most) most = sizes[i], best = masks[i];
        else if (sizes[i] == most) best |= masks[i];   /* (several of the largest size: all of them) */
        if (sizes[i] < least) least = sizes[i];
    }
    if (!best || least == most || !alike) return 0;
    int physical = 0;
    for (int i = 0; i < ncores; i++) physical += (cores[i] & best) != 0;
    if (physical < 8) return 0;
    return best;
}
/* (hookshot) The logical processors this process may run on (its affinity mask: `start /affinity`, a launcher's, Process
   Lasso's, taskset's), for the reserved core's choice; every one when the mask cannot be read (Windows gives 0 for a process
   with threads in more than one processor group) */
static uint64_t process_cpus(void) {
    DWORD_PTR proc = 0, sys = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) || !proc) return ~0ull;
    return (uint64_t)proc;
}
/* (the choice: the cores' logical processor masks and EfficiencyClass, n of them, among the cores whose logical processors
   are all in allowed (process_cpus); 0 for none. A core outside the process's mask is no candidate and is not counted: the
   replay thread cannot be pinned there (Windows refuses the mask; Linux lets the thread leave the set it was given), and
   the game's threads would lose a core of the mask's for nothing. As the earlier fallback below, which the enhanced choice
   skipped: under 0x0FFF of 16 the last core of all, 0xC000, was kept, the replay's pin was refused and the log said it held.
   tests/test_reserved_core.py, tests/test_reserved_core_mask.py) */
static uint64_t pick_reserved_core(const uint64_t *masks, const unsigned char *classes, int n, uint64_t allowed) {
    uint64_t last = 0;
    int best = -1, fast = 0, logical = 0;
    for (int i = 0; i < n; i++) {
        if (!masks[i] || (masks[i] & ~allowed)) continue;
        logical += __builtin_popcountll(masks[i]);
        if ((int)classes[i] > best) {
            best = classes[i], last = masks[i], fast = 1;
        } else if ((int)classes[i] == best) {
            fast++;
            if (masks[i] > last) last = masks[i];
        }
    }
    if (logical < 8 || fast < 4) return 0;

    // One-time debug logging
    static bool logged = false;
    if (!logged) {
        logged = true;
        hos_log("vk: reserved core selection: %d cores, %d logical, best class=%d, fast cores=%d, selected=%#llx "
                "(of the logical processors %#llx)", n, logical, best, fast, (unsigned long long)last,
                (unsigned long long)allowed);
        for (int i = 0; i < n; i++) {
            if (masks[i]) {
                hos_log("vk:   core %d: mask=%#llx, class=%d, popcount=%d",
                        i, (unsigned long long)masks[i], classes[i], __builtin_popcountll(masks[i]));
            }
        }
    }
    return last;
}
#if defined(_WIN32)
/* (pick_cache_cores over the calling thread's group's L3 caches; 0: no such split, or HOS_CACHE_CORES=0) */
uint64_t hos_cache_cpus(void) {
    static uint64_t mask = ~0ull;
    if (mask != ~0ull) return mask;
    mask = 0;
    if (!env_on_or("HOS_CACHE_CORES", 1)) return mask;
    static uint64_t masks[64], sizes[64], cores[256];
    DWORD len = 0;
    int n = 0, ncores = 0;
    GROUP_AFFINITY ga;
    const WORD group = GetThreadGroupAffinity(GetCurrentThread(), &ga) ? ga.Group : 0;
    if (GetLogicalProcessorInformationEx(RelationCache, NULL, &len) || GetLastError() != ERROR_INSUFFICIENT_BUFFER) return mask;
    char *buf = (char *)malloc(len);
    if (buf && GetLogicalProcessorInformationEx(RelationCache, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len))
        for (DWORD at = 0; at < len && n < 64;) {
            const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + at);
            if (!e->Size) break;
            if (e->Relationship == RelationCache && e->Cache.Level == 3 && e->Cache.GroupMask.Group == group) {
                masks[n] = (uint64_t)e->Cache.GroupMask.Mask;
                sizes[n] = e->Cache.CacheSize;
                n++;
            }
            at += e->Size;
        }
    free(buf);
    len = 0;   /* (and the group's cores: how many of them the largest L3 has) */
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len) || GetLastError() != ERROR_INSUFFICIENT_BUFFER) return mask;
    buf = (char *)malloc(len);
    if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len))
        for (DWORD at = 0; at < len && ncores < 256;) {
            const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + at);
            if (!e->Size) break;
            if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount && e->Processor.GroupMask[0].Group == group)
                cores[ncores++] = (uint64_t)e->Processor.GroupMask[0].Mask;
            at += e->Size;
        }
    free(buf);
    mask = pick_cache_cores(masks, sizes, n, cores, ncores);
    if (mask) {
        uint64_t most = 0, least = ~0ull;
        for (int i = 0; i < n; i++) {
            if (sizes[i] > most) most = sizes[i];
            if (sizes[i] && sizes[i] < least) least = sizes[i];
        }
        hos_log("threads: the game's on logical processors %#llx, the cores that share the largest L3 (%llu MB; the others' is "
                "%llu MB). HOS_CACHE_CORES=0: every core", (unsigned long long)mask, (unsigned long long)(most >> 20),
                (unsigned long long)(least >> 20));
    }
    return mask;
}
/* (pick_reserved_core over the calling thread's group's cores; *cores: how many, 0 when they could not be read) */
static uint64_t last_fastest_core_legacy(int *cores) {
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION info[256];
    DWORD len = sizeof info;
    if (!GetLogicalProcessorInformation(info, &len)) {
        hos_log("vk: GetLogicalProcessorInformation failed: %lu", GetLastError());
        return 0;
    }

    int logical = 0;
    int current_core = 0, listed = 0;
    uint64_t masks[256];
    unsigned char classes[256];
    const uint64_t allowed = process_cpus();   /* (hookshot: the cores the process may run on wholly, as pick_reserved_core) */

    int pcore_candidates = 0;
    int ecore_candidates = 0;

    for (DWORD i = 0; i < len / sizeof info[0] && current_core < 256; i++) {
        if (info[i].Relationship == RelationProcessorCore) {
            uint64_t mask = (uint64_t)info[i].ProcessorMask;
            listed++;
            if (mask & ~allowed) continue;
            int popcount = __builtin_popcountll(mask);
            logical += popcount;

            // Heuristic: cores with 2+ logical processors = P-cores (with HT)
            // cores with 1 logical processor = E-cores (no HT)
            if (popcount >= 2) {
                classes[current_core] = 1;  // P-core
                pcore_candidates++;
            } else {
                classes[current_core] = 0;  // E-core
                ecore_candidates++;
            }
            masks[current_core] = mask;
            current_core++;
        }
    }

    *cores = listed;

    hos_log("vk: legacy API fallback: %d cores, %d logical, P-candidates=%d, E-candidates=%d (of the logical processors %#llx)",
            current_core, logical, pcore_candidates, ecore_candidates, (unsigned long long)allowed);

    // If mixed core types detected, use enhanced picker
    if (pcore_candidates > 0 && ecore_candidates > 0) {
        hos_log("vk: Intel hybrid CPU detected via legacy API heuristic");
        return pick_reserved_core(masks, classes, current_core, allowed);
    }

    // Single CPU type - use original logic (last core)
    uint64_t last = 0;
    for (int i = 0; i < current_core; i++) {
        if ((uint64_t)masks[i] > last) last = (uint64_t)masks[i];
    }
    if (logical >= 8) return last;
    return 0;
}

/* (pick_reserved_core over the calling thread's group's cores; *cores: how many, 0 when they could not be read) */
static uint64_t last_fastest_core(int *cores) {
    static uint64_t masks[256];
    static unsigned char classes[256];
    DWORD len = 0;
    int n = 0;
    *cores = 0;
    GROUP_AFFINITY ga;
    const WORD group = GetThreadGroupAffinity(GetCurrentThread(), &ga) ? ga.Group : 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len) || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        hos_log("vk: GetLogicalProcessorInformationEx failed: %lu", GetLastError());
        return last_fastest_core_legacy(cores);
    }
    char *buf = (char *)malloc(len);
    if (!buf) {
        hos_log("vk: failed to allocate buffer for processor info");
        return last_fastest_core_legacy(cores);
    }

    bool success = false;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) {
        for (DWORD at = 0; at < len && n < 256;) {
            const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + at);
            if (!e->Size) break;
            if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount && e->Processor.GroupMask[0].Group == group) {
                masks[n] = (uint64_t)e->Processor.GroupMask[0].Mask;
                classes[n] = e->Processor.EfficiencyClass;
                n++;
            }
            at += e->Size;
        }
        success = true;
    }

    free(buf);

    if (!success || n == 0) {
        hos_log("vk: extended API failed or no cores found, trying legacy API with Intel hybrid detection");
        return last_fastest_core_legacy(cores);
    }

    *cores = n;
    return pick_reserved_core(masks, classes, n, process_cpus());
}
#endif
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
/* (hookshot) A Linux cpulist ("0-7", "0-11,16-19") as a mask of logical processors 0-63; 0 when it does not parse.
   (tests/test_reserved_core.py) */
static uint64_t parse_cpu_list(const char *s) {
    uint64_t mask = 0;
    while (*s && *s != '\n') {
        char *end;
        long a = strtol(s, &end, 10), b = a;
        if (end == s || a < 0) return 0;
        s = end;
        if (*s == '-') {
            b = strtol(++s, &end, 10);
            if (end == s || b < a) return 0;
            s = end;
        }
        for (long i = a; i <= b && i < 64; i++) mask |= 1ull << i;
        if (*s == ',') s++;
        else if (*s && *s != '\n') return 0;
    }
    return mask;
}
/* (hookshot) Linux: Windows' EfficiencyClass has no file of its own, but on Intel's hybrid CPUs the kernel lists the
   performance cores' logical processors in /sys/devices/cpu_core/cpus (the hybrid PMU; the efficiency cores are in
   cpu_atom). Without it the last core was kept, and on those CPUs that is an efficiency core: a Core Ultra 7 265KF's
   logical processor 19 (P-cores 0-7, E-cores 8-19). Those cores are class 1, the others 0, and pick_reserved_core
   chooses as on Windows. *cores: 0 when the file is not there (a CPU of one kind, AMD, an older kernel): as before. */
static uint64_t last_fastest_core(int *cores) {
    *cores = 0;
    char line[256] = {0};
    FILE *f = fopen("/sys/devices/cpu_core/cpus", "r");
    if (!f) return 0;
    const bool read = fgets(line, sizeof line, f) != NULL;
    fclose(f);
    const uint64_t fast = read ? parse_cpu_list(line) : 0;
    if (!fast) return 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION info[64];
    DWORD len = sizeof info;
    if (!GetLogicalProcessorInformation(info, &len)) return 0;
    uint64_t masks[64];
    unsigned char classes[64];
    int n = 0;
    for (DWORD i = 0; i < len / sizeof info[0] && n < 64; i++)
        if (info[i].Relationship == RelationProcessorCore) {
            masks[n] = (uint64_t)info[i].ProcessorMask;
            classes[n] = (masks[n] & fast) != 0;
            n++;
        }
    *cores = n;
    return pick_reserved_core(masks, classes, n, process_cpus());
}
#endif
uint64_t hos_reserved_cpus(void) {
    static uint64_t mask = ~0ull;
    if (mask != ~0ull) return mask;
    mask = 0;

    // Allow explicit hex mask override for testing
    const char* override = getenv("HOS_RESERVE_CORE_MASK");
    if (override) {
        char* endptr;
        mask = strtoull(override, &endptr, 16);
        if (*endptr != '\0') {
            hos_log("vk: HOS_RESERVE_CORE_MASK '%s' is not a valid hex value, ignoring", override);
            mask = 0;
        } else {
            hos_log("vk: HOS_RESERVE_CORE_MASK override: %#llx", (unsigned long long)mask);
        }
        return mask;
    }

    if (!env_on_or("HOS_RESERVE_CORE", 1)) {
        hos_log("vk: HOS_RESERVE_CORE=0, no core reserved");
        return mask;
    }
#if defined(_WIN32)
    /* (hookshot) none where the game is kept to the cores of the largest L3 (hos_cache_cpus): they are 8 cores on the CPUs
       measured (a 9950X3D, SMT off), and with one of them kept for the replay the game's threads had 7. In the market the replay
       then took 9.8-10.0 ms a frame but the frame rate stayed at 56.9-57.4; with the 8 shared, 10.1-10.5 ms and 59.9 fps.
       HOS_RESERVE_CORE=1: the last core of the fastest kind is kept all the same, as on other CPUs. That is normally a core of
       the other CCD, not the cache core of the 9.8-10.0 ms above, and the game's threads then have all of the cache cores */
    if (hos_cache_cpus() && !env_on("HOS_RESERVE_CORE")) return mask;
    {
        int cores = 0;
        const uint64_t fast = last_fastest_core(&cores);
        if (cores) {
            mask = fast;
            hos_log("vk: reserved core mask (enhanced): %#llx (%d cores; a core the process may run on wholly, of %#llx)",
                    (unsigned long long)mask, cores, (unsigned long long)process_cpus());
            return mask;
        }
    }
#elif !defined(__EMSCRIPTEN__)
    {
        int cores = 0;
        const uint64_t fast = last_fastest_core(&cores);
        if (cores) {
            mask = fast;
            hos_log("vk: reserved core mask (hybrid, /sys/devices/cpu_core): %#llx (%d cores; a core the process may run on "
                    "wholly, of %#llx)", (unsigned long long)mask, cores, (unsigned long long)process_cpus());
            return mask;
        }
    }
#endif
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION info[256];
    DWORD len = sizeof info;
    if (!GetLogicalProcessorInformation(info, &len)) return mask;
    /* (hookshot) of the processors this process may run on. A process kept to some of them (taskset, a launcher's
       affinity) had the PC's last core kept all the same, outside its set, and a thread may leave the set: the replay ran
       there, beside whatever else did. A timed session kept to one CCD of a 7900X (CPUs 0-5,12-17) replayed on CPUs 11 and
       23 of the other, with the builds: frames of 2-8 s whenever one ran, and a 13th core when none did. A core counts
       when all its logical processors are allowed; with every processor allowed, the last core as before. (The choices
       above keep the same rule: pick_reserved_core) */
    const uint64_t proc = process_cpus();
    int logical = 0;
    uint64_t last = 0;
    for (DWORD i = 0; i < len / sizeof info[0]; i++)
        if (info[i].Relationship == RelationProcessorCore && !((uint64_t)info[i].ProcessorMask & ~proc)) {
            logical += __builtin_popcountll((uint64_t)info[i].ProcessorMask);
            if ((uint64_t)info[i].ProcessorMask > last) last = (uint64_t)info[i].ProcessorMask;
        }
    if (logical >= 8) mask = last;
    hos_log("vk: reserved core mask (fallback): %#llx", (unsigned long long)mask);
    return mask;
}

void hos_thread_pin(void *h, const char *name) {
    const uint64_t reserved = hos_reserved_cpus();
#if defined(_WIN32)
    const uint64_t cache = hos_cache_cpus();   /* (hookshot: the cores of the largest L3, when the CPU's differ) */
#else
    const uint64_t cache = 0;
#endif
    if (reserved || cache) {
        DWORD_PTR proc = 0, sys = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
        uint64_t may = (uint64_t)proc;
        if (cache && (may & cache)) may &= cache;
        if (may & ~reserved) SetThreadAffinityMask(h, (DWORD_PTR)(may & ~reserved));
    }
    static NameSpec pins[24], prios[24];
    static int npins = -1, nprios;
    static SRWLOCK lk = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lk);
    if (npins < 0) {
        npins = parse_name_specs(getenv("HOS_PIN"), pins, 24, 16);
        nprios = parse_name_specs(getenv("HOS_PRIO"), prios, 24, 10);
    }
    ReleaseSRWLockExclusive(&lk);
    const NameSpec *p = match_name_spec(pins, npins, name);
    if (p && p->value) SetThreadAffinityMask(h, (DWORD_PTR)p->value);
    static const int levels[5] = {THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL,
                                  THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_HIGHEST};
    if ((p = match_name_spec(prios, nprios, name)) && p->value >= -2 && p->value <= 2)
        SetThreadPriority(h, levels[p->value + 2]);
    else if (strstr(name, "OptimizeChange") && !env_on("HOS_OCB_NORMAL")) {
        /* (OoT's OptimizeChangeBank (guest priority 27, a background thread) walks every placed object of the
           area by name, again and again (Kokiri: ~344 names, ~250k lookups a second). On the console it only gets the
           time nothing more important wants; at the host's normal priority it took the renderer's and the driver's
           cores and the frame waited (Kokiri, RX 6600 + 5600G: 17-20 fps; lowest: 21-28). HOS_OCB_NORMAL=1: as before) */
        SetThreadPriority(h, THREAD_PRIORITY_LOWEST);
    }
}

/* ---- time ------------------------------------------------------------------------------------------------ */
uint64_t hos_ticks(void) { return recomp_cntpct(NULL); }

int64_t hos_now_ns(void) {
    static LARGE_INTEGER freq;
    static int64_t ns_per_tick;
    LARGE_INTEGER now;
    if (!freq.QuadPart) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        ns_per_tick = 1000000000LL % f.QuadPart == 0 ? 1000000000LL / f.QuadPart : 0;
        freq = f;
    }
    QueryPerformanceCounter(&now);
    if (ns_per_tick) return (int64_t)now.QuadPart * ns_per_tick;
    return (int64_t)((now.QuadPart / freq.QuadPart) * 1000000000LL + (now.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
}

static void sleep_ns(int64_t ns, int guest);
/* How sleeps under a millisecond wait: 0, all yield until their time; 1, the game's own (SleepThread) wait on the timer,
   which wakes about 0.5 ms late, and the host's (the frame limiter, the audio renderer) yield; 2, all wait on the timer.
   asked: the value of HOS_SHORT_SLEEP_TIMER, NULL when it is not set; set, it is read by env_on's rule (on: the timer for
   all, 2; off: yielding for all, 0). threads: the logical processors this process may run on, 0 when not known. Not set:
   1 with SHORT_SLEEP_TIMER_THREADS of them or fewer, which is no more than the game's six cores, else 0.
   OoT's idle job workers sleep 20 us and look again, about 7,600 times a second, and a yielding loop keeps a processor
   busy for each: with so few processors that is time the frame's threads need. Measured with the game held to part of a
   5700X3D: on 4 threads Hyrule Field went from 47 to 51 fps and lost its long frames (over 250 ms in 2 of 4 runs
   yielding, in none with the timer), Kokiri's worst frame went from 53-422 ms to 30 ms; on 6 the frame rate stayed and a
   frame's CPU time fell 9 %. On 8 threads and more yielding stays: with the timer for all sleeps there were no long
   frames either way and Kokiri lost 0.2 fps under its cap, and the timer made TOTK's 20 us waits for its job workers 25x
   longer, ~2000 a second on the frame's critical path. */
static int hos_short_sleep_timer(const char *asked, int threads) {
    if (asked) return env_value_off(asked) ? 0 : 2;
    return threads > 0 && threads <= SHORT_SLEEP_TIMER_THREADS ? 1 : 0;
}
/* Where the game's short sleeps yield (more than SHORT_SLEEP_TIMER_THREADS processors), at most this many of its
   threads yield through one at the same time; the game's next short sleep then waits on the timer until one is back.
   OoT's idle job workers all poll their queues in 20 us steps: the first few notice new work within ~20 us, the rest
   stop keeping a processor busy each (as Go's scheduler keeps at most half of its idle threads looking for work, and
   RPCS3 yields below its host quantum only on CPUs with few threads). The frame's serial threads (hos_thread_serial:
   the game thread's wait for its own batch of jobs) and the host's sleeps always yield and are not counted.
   asked: HOS_SHORT_SLEEP_POLLERS, NULL when not set: a number, 0 = no limit (every short sleep yields, as before); not
   a number = not set. Not set: a quarter of the processors (4 of 16), none where they are not known. */
static volatile LONG short_pollers;   /* (the game's threads in the yield loop now) */
static int hos_short_sleep_pollers(const char *asked, int threads) {
    char *end = NULL;
    const long n = asked && *asked ? strtol(asked, &end, 10) : 0;
    if (end && end != asked && *end == 0) return n > 0 ? (int)(n < 1024 ? n : 1024) : 0;
    return threads > SHORT_SLEEP_TIMER_THREADS ? threads / 4 : 0;
}
/* (virtual cores: a sleep gives the core up, SleepThread(0) offers it to a waiter as good) guest: the game's own sleep */
static void sleep_away(int64_t ns, int guest) {
    if (ns <= 0) {
#ifdef TRACY_ENABLE
        HT_WAIT(yz, "sleep 0 (yield)");
#endif
        hos_vcore_yield();
        SwitchToThread();
        return;
    }
#ifdef TRACY_ENABLE
    HT_WAIT(tz, "sleep");
    HT_ZONE_TEXTF(tz, "%s %.3f ms, lr %#llx", guest ? "guest" : "host", ns / 1e6,
                  (unsigned long long)(guest && hos_ctx() ? hos_ctx()->x[30] : 0));
#endif
    HOS_CHAIN(HOS_CHAIN_SLEEP, (uint64_t)ns);
    const int away = hos_vcore_away();
    sleep_ns(ns, guest);
    if (away) hos_vcore_back();
    HOS_CHAIN(HOS_CHAIN_AWAKE, (uint64_t)ns);
}
void hos_sleep_ns(int64_t ns) { sleep_away(ns, 0); }
static void sleep_ns(int64_t ns, int guest) {
    /* A high-resolution waitable timer (one per thread) sleeps to within ~0.5 ms without spinning; Sleep() rounds to
       the timer resolution, so without one the last millisecond is spun (threads that sleep in short steps, the
       vibration thread for one, used 15% of a core spinning). */
    static _Thread_local HANDLE timer;
    static _Thread_local int tried;
    int64_t until = hos_now_ns() + ns;
#ifdef __EMSCRIPTEN__
    /* The web has no high-resolution timer, and its clock is a call into JavaScript: spinning to the deadline below
       burned ~0.8 of a core across TOTK's sleeping threads (cores its game, render and WebGPU threads needed). A
       timed wait (a futex: Atomics.wait) to `slack` before the time, then the rest spun; slack follows how late the
       waits wake (HOS_WEB_SPIN_SLEEP=1: spun, as before). */
    static int web_spin = -1;
    if (web_spin < 0) web_spin = env_on("HOS_WEB_SPIN_SLEEP");
    static _Atomic int64_t slack = 300000;
    const int64_t s = atomic_load_explicit(&slack, memory_order_relaxed);
    if (!web_spin && ns > s + 50000) {
        const int64_t want = until - s;
        usleep((useconds_t)((want - hos_now_ns()) / 1000));
        const int64_t late = hos_now_ns() - want;
        int64_t next = s + (late + 25000 - s) / 8;
        next = next < 50000 ? 50000 : next > 2000000 ? 2000000 : next;
        atomic_store_explicit(&slack, next, memory_order_relaxed);
    }
    while (hos_now_ns() < until) sched_yield();
    return;
#endif
    /* Under a millisecond: yielded away until the time, or the timer for the game's own sleeps where the processors
       are few (hos_short_sleep_timer). HOS_SHORT_SLEEP_TIMER=1: the timer always, 0: never, not set: by the processors */
    /* (the choice, how | the cap << 2: one atomic word, so a thread sees both or neither) */
    static int choice = -1;
    int chosen = __atomic_load_n(&choice, __ATOMIC_RELAXED);
    if (chosen < 0) {
        const int threads = process_threads();   /* (0 elsewhere than Windows: the timer when asked for) */
        int how = hos_short_sleep_timer(getenv("HOS_SHORT_SLEEP_TIMER"), threads);
#if !defined(_WIN32)
        /* (hookshot) Linux: the game's own sleeps wait on the timer whatever the processors, the host's still yield (1).
           With #2602, OoT's five idle job workers and sead::MainThread yielded 3.3-3.6 ms a frame each in Hyrule Field
           (RTX 3090, 7900X, 24 threads), ~9 ms of CPU a frame in all, while one of them waited on a GPU readback. With
           the timer for all sleeps the frame's CPU fell 44.8 -> 35.4 ms at the same median frame (11.30 / 11.32 ms).
           The job workers are not on the frame's critical path there (HOS_CHAIN), so their ~0.5 ms later wake costs
           nothing measured; on 4-8 thread CPUs (the Steam Deck: 8) those yields took processors the frame needs.
           HOS_SHORT_SLEEP_TIMER=0: all yield, as before; =1: all on the timer */
        if (!getenv("HOS_SHORT_SLEEP_TIMER")) how = 1;
#endif
        /* (the line says what was chosen, before any thread has made its timer: where the high-resolution timer cannot
           be made, Windows before 10 1803, a short sleep falls through to the yield loop at the end and nothing changes) */
        static volatile LONG told;
        if (how && !InterlockedExchange(&told, 1))
            hos_log("sleep: %s sleeps under a millisecond wait on the timer where the high-resolution one can be made "
                    "(%d logical processors, 0: not counted; HOS_SHORT_SLEEP_TIMER=0: all yield)",
                    how == 2 ? "all" : "the game's", threads);
        /* (HOS_SHORT_SLEEP_TIMER set says how all short sleeps wait: no limit then) */
        const int cap = how || getenv("HOS_SHORT_SLEEP_TIMER") ? 0 : hos_short_sleep_pollers(getenv("HOS_SHORT_SLEEP_POLLERS"), threads);
        if (cap && !InterlockedExchange(&told, 1))
            hos_log("sleep: at most %d of the game's threads yield through a short sleep at a time, the others wait on the "
                    "timer (%d logical processors; HOS_SHORT_SLEEP_POLLERS=0: all yield)", cap, threads);
        chosen = how | cap << 2;
        __atomic_store_n(&choice, chosen, __ATOMIC_RELAXED);
    }
    const int short_timer = chosen & 3, poll_cap = chosen >> 2;
    if (ns < 1000000 && short_timer < (guest ? 1 : 2)) {
        const int counted = guest && poll_cap && !hos_thread_serial(hos_self());
        if (!counted || InterlockedIncrement(&short_pollers) <= poll_cap) {
        /* TICK-062 B4, HOS_SHORT_SLEEP_PAUSE=1: one yield, then a PAUSE loop to the deadline (SwitchToThread in a loop
           keeps calling into the kernel and bounces the SMT sibling's work) */
        static int pause_spin = -1;
        if (pause_spin < 0) pause_spin = getenv("HOS_SHORT_SLEEP_PAUSE") && atoi(getenv("HOS_SHORT_SLEEP_PAUSE")) != 0;
        if (pause_spin) {
            if (hos_now_ns() < until) SwitchToThread();
            while (hos_now_ns() < until) { YieldProcessor(); YieldProcessor(); YieldProcessor(); YieldProcessor(); }
            return;
        }
            while (hos_now_ns() < until)
                if (!SwitchToThread()) YieldProcessor();
            if (counted) InterlockedDecrement(&short_pollers);
            return;
        }
        InterlockedDecrement(&short_pollers);   /* (enough of them poll already: the timer, below) */
    }
    if (!tried) {
        tried = 1;
        timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    }
    if (timer) {
        LARGE_INTEGER due;
        due.QuadPart = -(ns / 100 > 0 ? ns / 100 : 1);   /* relative, 100 ns units */
        if (SetWaitableTimerEx(timer, &due, 0, NULL, NULL, NULL, 0)) {
            WaitForSingleObject(timer, INFINITE);
            return;
        }
    }
    if (ns > 2000000) Sleep((DWORD)((ns - 1000000) / 1000000));
    while (hos_now_ns() < until) SwitchToThread();
}

/* ---- nn::os threads -------------------------------------------------------------------------------------- */
static HosThread *thread_arg(Ctx *c, uint64_t tt) {
    /* (the calling thread's own ThreadType, the usual case: GetThreadPriority(GetCurrentThread()) ~2,400 times a frame in
       Hyrule from the navigation workers: its record, without the scan of every thread under threads_lock; the SDK reads
       the priority from the ThreadType itself) */
    if (cur && cur->guest_tt == tt) return cur;
    HosThread *t = hos_thread_of(tt);
    if (!t) hos_fatal(c, "unknown nn::os::ThreadType %#llx", (unsigned long long)tt);
    return t;
}

/* nn::os::CreateThread(ThreadType*, void (*)(void*), void* arg, void* stack, unsigned long size, int prio) */
HLE(_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mi) {
    HOS_TRACE_CALL("CreateThread");
    hos_thread_create(ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), (int)ARG(5), -2);
    RET(0);
}

/* nn::os::CreateThread(ThreadType*, void (*)(void*), void* arg, void* stack, unsigned long size, int prio,
   int idealCore) (hookshot: OoT's Porter thread pool uses this overload; left to the guest SDK it reached
   ThreadManagerImplByHorizon::MapAliasStack, which needs svcMapMemory, and aborted) */
HLE(_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mii) {
    HOS_TRACE_CALL("CreateThread");
    hos_thread_create(ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), (int)ARG(5), (int)ARG(6));
    RET(0);
}

HLE(_ZN2nn2os11StartThreadEPNS0_10ThreadTypeE) {
    HOS_TRACE_CALL("StartThread");
    hos_thread_start(thread_arg(c, ARG(0)));
}

HLE(_ZN2nn2os10WaitThreadEPNS0_10ThreadTypeE) {
    HOS_TRACE_CALL("WaitThread");
    hos_thread_join(thread_arg(c, ARG(0)));
}

HLE(_ZN2nn2os13DestroyThreadEPNS0_10ThreadTypeE) {
    HOS_TRACE_CALL("DestroyThread");
    HosThread *t = thread_arg(c, ARG(0));
    hos_thread_join(t);
    hos_thread_destroy(t);
}

HLE(_ZN2nn2os13SetThreadNameEPNS0_10ThreadTypeEPKc) {
    HosThread *t = thread_arg(c, ARG(0));
    hos_thread_set_name(t, ARG(1) ? (const char *)GP(ARG(1)) : "");
    if (hos_trace) hos_log("SetThreadName %s", t->name);
}

HLE(_ZN2nn2os16GetCurrentThreadEv) { RET(cur->guest_tt); }
HLE(_ZN2nn2os11GetThreadIdEPKNS0_10ThreadTypeE) { RET(thread_arg(c, ARG(0))->id); }
HLE(_ZN2nn2os17GetThreadPriorityEPKNS0_10ThreadTypeE) { RET((uint32_t)hos_thread_priority(thread_arg(c, ARG(0)), 0, 0)); }
HLE(_ZN2nn2os20ChangeThreadPriorityEPNS0_10ThreadTypeEi) {
    RET((uint32_t)hos_thread_priority(thread_arg(c, ARG(0)), 1, (int)ARG(1)));
}

/* nn::os::SetThreadCoreMask(ThreadType*, int idealCore, unsigned long mask) */
HLE(_ZN2nn2os17SetThreadCoreMaskEPNS0_10ThreadTypeEim) {
    HosThread *t = thread_arg(c, ARG(0));
    hos_thread_affinity(t, (int)ARG(1), ARG(2));
}

/* TICK-062 A10, HOS_CORE_COUNT=1 (diagnostic, no behaviour change): per guest thread, the calls that ask for the core number
   (nn::os::GetCurrentCoreNumber, svc 0x10) and the waits for a virtual core, logged every 5 s (cumulative) */
int hos_core_count_on(void) {
    static int on = -1;
    if (on < 0) on = env_on("HOS_CORE_COUNT");
    return on;
}
void hos_core_count_svc(void) { if (cur) cur->cc_svc++; }
static void core_count_log(void) {
    static volatile LONG64 last;
    const int64_t now = hos_now_ns();
    const int64_t l = last;
    if (now - l < 5000000000LL || InterlockedCompareExchange64(&last, now, l) != l) return;
    if (!l) return;
    AcquireSRWLockShared(&threads_lock);
    for (int i = 0; i < nthreads; i++) {
        const HosThread *t = threads[i];
        if (t->cc_hle + t->cc_svc + t->cc_vcw)
            hos_log("corecount: %-24s core-number calls %lld (hle %lld, svc %lld), virtual-core waits %lld (%.1f ms)", t->name,
                    (long long)(t->cc_hle + t->cc_svc), (long long)t->cc_hle, (long long)t->cc_svc, (long long)t->cc_vcw,
                    t->cc_vcw_ns / 1e6);
    }
    ReleaseSRWLockShared(&threads_lock);
}
HLE(_ZN2nn2os20GetCurrentCoreNumberEv) {
    if (hos_core_count_on()) { cur->cc_hle++; core_count_log(); }
    RET((uint32_t)hos_thread_core(cur));
}
HLE(_ZN2nn2os26GetThreadAvailableCoreMaskEv) { RET((1u << hos_cores()) - 1); }   /* cores 0-2: what an application gets */

HLE(_ZN2nn2os19GetCurrentStackInfoEPmS1_) {
    if (ARG(0)) G64(ARG(0)) = cur->stack;
    if (ARG(1)) G64(ARG(1)) = cur->stack_size;
}

/* (hookshot) nn::os::YieldThread gives up the virtual core only, not the processor (HOS_OS_YIELD=1: SwitchToThread as
   well, as before; HOS_AB may flip it, each yield deciding alone). OoT yields after every job and actor callback. On a
   PC of 4 cores in Castle Town's market the job threads' CPU time was about their running time plus their time inside
   SwitchToThread (the six BaseProc executors 107 % of a core measured, 45.5 % of it running, 51.6 % in SwitchToThread):
   kernel time handing the processor to another thread and taking it back, about 0.75 of the four processors over the
   job threads. hos_vcore_yield already hands the virtual core to a waiting game thread of better or equal priority, which
   is what the game's yield asks for; the OS still moves threads between processors at the end of their time slices. */
static int os_yield(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("HOS_OS_YIELD");
        v = e && !env_value_off(e);
    }
    return hos_ab_value("HOS_OS_YIELD", v);
}
HLE(_ZN2nn2os11YieldThreadEv) {
#ifdef TRACY_ENABLE
    HT_WAIT(tz, "YieldThread");
    HT_ZONE_TEXTF(tz, "lr %#llx", (unsigned long long)c->x[30]);
#endif
    hos_vcore_yield();
    if (os_yield()) SwitchToThread();
}
/* HOS_SLEEP_LOG=1 (diagnostic): guest sleeps by caller (LR) every 5 s: count, asked, slept */
void hos_sleep_logged(int64_t ns, uint64_t lr) {
    /* HOS_POLL_SLEEP_NS=<ns> (experiment): a guest sleep of up to 100 us lasts at most that long. TOTK's job system
       waits for its workers by sleeping 20 us and looking again (~90 times a frame at the castle, on MainThread and
       the workers): a shorter step notices the job's end sooner (the loop looks again; nothing else changes).
       Measured 2026-10-21 (castle, 2000 ns, 3 rounds): no change. */
    static int64_t poll_cap = -1;
    if (poll_cap < 0) {
        const char *e = getenv("HOS_POLL_SLEEP_NS");
        poll_cap = e ? atoll(e) : 0;
    }
    if (poll_cap && ns > poll_cap && ns <= 100000) ns = poll_cap;
    static int on = -1;
    if (on < 0) on = env_on("HOS_SLEEP_LOG");
    if (!on) { sleep_away(ns, 1); return; }
    const int64_t t0 = hos_now_ns();
    sleep_away(ns, 1);
    const int64_t took = hos_now_ns() - t0;
    static SRWLOCK lk = SRWLOCK_INIT;
    static struct { uint64_t lr; int64_t n, asked, slept; } tab[64];
    static int64_t last;
    AcquireSRWLockExclusive(&lk);
    int i = 0;
    while (i < 64 && tab[i].lr && tab[i].lr != lr) i++;
    if (i < 64) {
        tab[i].lr = lr;
        tab[i].n++;
        tab[i].asked += ns > 0 ? ns : 0;
        tab[i].slept += took;
    }
    const int64_t now = hos_now_ns();
    if (now - last > 5000000000LL) {
        last = now;
        for (int k = 0; k < 64 && tab[k].lr; k++)
            if (tab[k].n)
                hos_log("sleep: lr %#llx: %lld calls, asked %.2f ms, slept %.2f ms (%.1f us each)",
                        (unsigned long long)tab[k].lr, (long long)tab[k].n, tab[k].asked / 1e6, tab[k].slept / 1e6,
                        tab[k].slept / 1e3 / tab[k].n);
        memset(tab, 0, sizeof tab);
    }
    ReleaseSRWLockExclusive(&lk);
}

/* (hookshot) OoT's present waits for its frame's GPU work by polling an NVN event, sleeping ~1.6 ms between looks
   (0x80590500: `while (nvnEventGetValue(e) != 1) SleepThread(...)`, ~3.5 sleeps a frame, 2 ms each on Windows): the
   frame noticed the GPU's signal late, ~7 ms of the 25 ms frame asleep on its critical path. The GPU writes the event
   into guest memory itself, so the sleep looks at it between yields (its virtual core given up) and ends once it is
   1, at the latest when the asked time is up. Only where the poll is nvnEventGetValue. HOS_PRESENT_POLL=1 (an
   experiment: no fps change measured, and it keeps a core busy while it looks) */
static int present_poll(Ctx *c, int64_t ns) {
    static int on = -1;
    static uint64_t fn_ok, fn_bad;
    if (on < 0) on = env_on("HOS_PRESENT_POLL");
    if (!on || c->x[30] != 0x80590520 || ns <= 0 || ns > 20000000) return 0;
    const uint64_t fp = G64(c->x[24]);
    if (fp != fn_ok) {
        if (fp == fn_bad) return 0;
        const char *hos_hostcall_name(uint64_t addr);
        const char *n = hos_hostcall_name(fp);
        if (!n || strcmp(n, "nvnEventGetValue")) { fn_bad = fp; return 0; }
        fn_ok = fp;
        hos_log("sleep: OoT's present poll looks at its event between yields (HOS_PRESENT_POLL)");
    }
    int hos_nvn_event_value(uint64_t guest_event, uint32_t *v);
    const uint64_t ev = c->x[23] + (uint64_t)G32(c->x[22] + 0x164) * 64;
    uint32_t v = 0;
    if (!hos_nvn_event_value(ev, &v)) return 0;
    const int away = hos_vcore_away();
    const int64_t end = hos_now_ns() + ns;
    while (v != 1 && hos_now_ns() < end) {
        SwitchToThread();
        for (int i = 0; i < 64; i++) _mm_pause();
        hos_nvn_event_value(ev, &v);
    }
    if (away) hos_vcore_back();
    return 1;
}
/* (hookshot) The FPS counter's Breakdown (gpu/vk/vk_fps.cpp, through MkwVkHost slept_ns): the time slept in OoT's present poll
   (0x80590520 above: waiting for its frame's GPU work) and in the window's acquire (host/nn_vi.c: the frame pacing), whichever
   thread they are on (OoT simulates on its main thread and records on its presentation thread, which waits for it), and each
   thread's other guest sleeps. A clock read before and after a sleep, only while the Breakdown shows (hos_fps_timing, set by the
   back end through MkwVkHost fps_timing): off, a load of that flag */
volatile int hos_fps_timing;
volatile int64_t hos_poll_slept_ns, hos_acquire_slept_ns;
_Thread_local int64_t hos_slept_ns[2];
static void slept_since(uint64_t lr, int64_t t0) {
    if (!t0) return;
    const int64_t d = hos_now_ns() - t0;
    if ((lr & ~1ull) == 0x80590520) InterlockedAdd64(&hos_poll_slept_ns, d);
    else hos_slept_ns[1] += d;
}
HLE(_ZN2nn2os11SleepThreadENS_8TimeSpanE) {
    const int64_t t0 = hos_fps_timing ? hos_now_ns() : 0;
    if (present_poll(c, (int64_t)ARG(0))) { slept_since(c->x[30], t0); return; }
    if (c->x[30] == 0x80590520 && env_on("HOS_SLEEP_LOG")) {   /* (hookshot, diagnostic: OoT's present poll) */
        static int logged;
        if (logged++ < 3) {
            const char *hos_hostcall_name(uint64_t addr);
            const uint64_t fp = G64(c->x[24]), sync = c->x[23] + (uint64_t)G32(c->x[22] + 0x164) * 64;
            const char *n = hos_hostcall_name(fp);
            hos_log("sleep: OoT's present poll: fn %#llx (%s), arg %#llx, sleep %lld ns", (unsigned long long)fp,
                    n ? n : "guest code", (unsigned long long)sync, (long long)ARG(0));
        }
    }
    hos_sleep_logged((int64_t)ARG(0), c->x[30]);
    slept_since(c->x[30], t0);
}

/* ---- TLS slots --------------------------------------------------------------------------------------------- */
static volatile LONG slot_used[HOS_TLS_SLOTS];

/* nn::os::AllocateTlsSlot(TlsSlot*, void (*)(unsigned long)) -> Result */
HLE(_ZN2nn2os15AllocateTlsSlotEPNS0_7TlsSlotEPFvmE) {
    for (int i = 0; i < HOS_TLS_SLOTS; i++) {
        if (!InterlockedCompareExchange(&slot_used[i], 1, 0)) {
            G32(ARG(0)) = (uint32_t)i;
            RET(0);
            return;
        }
    }
    RET(0x2C01);   /* os::ResultOutOfResource */
}

HLE(_ZN2nn2os11FreeTlsSlotENS0_7TlsSlotE) { if ((uint32_t)ARG(0) < HOS_TLS_SLOTS) slot_used[(uint32_t)ARG(0)] = 0; }
HLE(_ZN2nn2os11GetTlsValueENS0_7TlsSlotE) { RET((uint32_t)ARG(0) < HOS_TLS_SLOTS ? cur->tls_slots[(uint32_t)ARG(0)] : 0); }
HLE(_ZN2nn2os11SetTlsValueENS0_7TlsSlotEm) { if ((uint32_t)ARG(0) < HOS_TLS_SLOTS) cur->tls_slots[(uint32_t)ARG(0)] = ARG(1); }

/* ---- ticks ------------------------------------------------------------------------------------------------- */
volatile uint64_t hos_gfx_module;
/* HOS_TICK_LOG=1 (diagnostic): who reads the clock, by caller (LR), every 5 s */
static void tick_log(uint64_t lr) {
    static CRITICAL_SECTION cs; static volatile LONG once; static uint64_t lrs[512]; static uint32_t counts[512]; static int64_t next;
    if (!InterlockedExchange(&once, 1)) InitializeCriticalSection(&cs);
    EnterCriticalSection(&cs);
    for (int i = (int)((lr >> 2) * 2654435761u >> 8) & 511, n = 0; n < 512; n++, i = (i + 1) & 511)
        if (!lrs[i] || lrs[i] == lr) { lrs[i] = lr; counts[i]++; break; }
    const int64_t now = hos_now_ns();
    if (now > next) {
        next = now + 5000000000LL;
        char line[1500]; int n = 0;
        for (int k = 0; k < 40 && n < 1400; k++) {
            int best = -1;
            for (int i = 0; i < 512; i++) if (counts[i] && (best < 0 || counts[i] > counts[best])) best = i;
            if (best < 0) break;
            n += snprintf(line + n, sizeof line - n, " %llx:%u", (unsigned long long)lrs[best], counts[best]);
            counts[best] = 0;
        }
        memset(counts, 0, sizeof counts);
        hos_log("tick readers in 5 s:%s", line);
    }
    LeaveCriticalSection(&cs);
}
HLE(_ZN2nn2os13GetSystemTickEv) {
    static int on = -1;
    if (on < 0) on = env_on("HOS_TICK_LOG");
    if (on) tick_log(c->x[30]);
    /* (hookshot) The graphics module: x19 at its frame-end clock read (main+0x346c770 is after the bl), for nn_vi.c's
       real-time step */
    if (c->x[30] == 0x8348c770) hos_gfx_module = c->x[19];
    RET(hos_ticks());
}
HLE(_ZN2nn2os22GetSystemTickFrequencyEv) { RET(HOS_TICK_HZ); }
/* (hookshot) At the counter's own rate, HOS_TICK_HZ (31.25 MHz: 32 ns a tick). These converted at the Switch 1's
   19.2 MHz (ticks = ns * 12 / 625) while GetSystemTick counts 31.25 MHz, so a tick span came out 1.63x too long: OoT's
   present slept 1.63 ms for its 1 ms poll step (0x80590500: SleepThread(ConvertToTimeSpan(frequency / 1000))), and
   any of its timing through these was off by as much. HOS_TICK_CONVERT_192=1: as before */
static int tick_convert_192(void) {
    static int on = -1;
    if (on < 0) on = env_on("HOS_TICK_CONVERT_192");
    return on;
}
HLE(_ZN2nn2os13ConvertToTickENS_8TimeSpanE) {
    int64_t ns = (int64_t)ARG(0);
    if (tick_convert_192()) { RET((ns / 625) * 12 + (ns % 625) * 12 / 625); return; }
    RET(ns / (int64_t)(1000000000ULL / HOS_TICK_HZ));
}
HLE(_ZN2nn2os17ConvertToTimeSpanENS0_4TickE) {
    int64_t t = (int64_t)ARG(0);
    if (tick_convert_192()) { RET((t / 12) * 625 + (t % 12) * 625 / 12); return; }
    RET(t * (int64_t)(1000000000ULL / HOS_TICK_HZ));
}

/* ---- fibers (hookshot: nn::async's FiberTaskCoordinator, on OoT's nn.npln threads) ----------------------------
   A guest fiber is a Win32 fiber with its own guest context: the lifted frames on its host stack keep using that
   context wherever it runs, so a fiber may resume on another thread. Switching in, it takes the running thread's
   identity (TLS registers, host thread object); the thread's own context stays in HosThread.ctx. The fiber function
   returns the FiberType to switch to next (nn::os semantics). */
typedef struct HosFiber {
    Ctx ctx;   /* (first: a context's fiber is found from its address) */
    uint64_t ft, fn, arg, stack, size;
    void *host;
    int started, done;
    struct HosFiber *next;
#ifdef TRACY_ENABLE
    char tracy_name[32];   /* (HOOKSHOT_TRACY: the fiber's row in a capture; Tracy keeps the pointer) */
#endif
} HosFiber;
static SRWLOCK fibers_lock = SRWLOCK_INIT;
static HosFiber *fibers;

static HosFiber *fiber_of_ft(uint64_t ft) {
    AcquireSRWLockShared(&fibers_lock);
    HosFiber *f = fibers;
    while (f && f->ft != ft) f = f->next;
    ReleaseSRWLockShared(&fibers_lock);
    return f;
}
static HosFiber *fiber_of_ctx(Ctx *c) {
    AcquireSRWLockShared(&fibers_lock);
    HosFiber *f = fibers;
    while (f && &f->ctx != c) f = f->next;
    ReleaseSRWLockShared(&fibers_lock);
    return f;
}
static void fiber_switch(Ctx *from, HosFiber *to);
static int fiber_stacks(uint64_t (*out)[2], int max) {
    int n = 0;
    AcquireSRWLockShared(&fibers_lock);
    for (HosFiber *f = fibers; f && n < max; f = f->next)
        if (!f->done && f->size) out[n][0] = f->stack, out[n][1] = f->stack + f->size, n++;
    ReleaseSRWLockShared(&fibers_lock);
    return n;
}

static void WINAPI fiber_main(void *p) {
    HosFiber *f = (HosFiber *)p;
    Ctx *c = &f->ctx;
    c->x[0] = f->arg;
    c->x[30] = 0;
    c->x[31] = (f->stack + f->size) & ~15ULL;
    hos_fpcr_sync(c);   /* (a new fiber has the default MXCSR and the FPCR of whoever started it) */
    a64_icall(c, f->fn);   /* FiberType* (*)(void*) */
    const uint64_t next = c->x[0];
    f->done = 1;
    HosFiber *to = next ? fiber_of_ft(next) : NULL;   /* (NULL: back to the thread's own context, as nn::os does) */
    if (next && !to) hos_fatal(c, "fiber %#llx returned %#llx: not an initialized fiber", (unsigned long long)f->ft,
                               (unsigned long long)next);
    fiber_switch(c, to);
    hos_fatal(c, "fiber %#llx: switched to after it returned", (unsigned long long)f->ft);
}

/* from: the context calling (the thread's own or a fiber's); to: NULL for the thread's own context */
static void fiber_switch(Ctx *from, HosFiber *to) {
    HosThread *t = cur;
    if (!t->host_fiber) {
        t->host_fiber = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
        if (!t->host_fiber) hos_fatal(from, "ConvertThreadToFiber failed (%lu)", GetLastError());
    }
    Ctx *dst = to ? &to->ctx : &t->ctx;
    if (dst == from) return;
    if (to) {
        if (to->done) hos_fatal(from, "SwitchToFiber(%#llx): that fiber has returned", (unsigned long long)to->ft);
        if (!to->started) {   /* the thread's state it inherits: FPCR, then the identity below */
            to->started = 1;
            to->ctx.fpcr = from->fpcr;
            to->ctx.fpsr = from->fpsr;
        }
    }
    /* the running thread's identity, whoever ran this context last */
    dst->tpidr_el0 = t->ctx.tpidr_el0;
    dst->tpidrro_el0 = t->ctx.tpidrro_el0;
    dst->thread = t->ctx.thread;
    if (from->poll) dst->poll = 1;   /* (a pending virtual-core request follows the thread) */
    t->run_ctx = to ? &to->ctx : NULL;
#ifdef TRACY_ENABLE
    /* (the zones from here on are the fiber's, or the thread's own again: a fiber's open zones follow it to the thread
       that runs it next, and Tracy checks each one's begins and ends by itself) */
    {
        extern _Thread_local int hos_tracy_calls_open;
        static volatile LONG told;
        if (hos_tracy_calls_open && InterlockedIncrement(&told) <= 5)
            hos_log("tracy: a fiber switch (to %s) with %d game call zones open on %s", to ? to->tracy_name : "the thread's own",
                    hos_tracy_calls_open, t->name);
    }
    if (to) HT_FIBER_ENTER(to->tracy_name);
    else HT_FIBER_LEAVE();
#endif
    SwitchToFiber(to ? to->host : t->host_fiber);
    /* back in `from`, maybe on another thread: take that one's identity (fiber_switch's caller set it already) */
}

/* InitializeFiber(FiberType*, FiberType* (*)(void*), void* arg, void* stack, unsigned long size, int flags) */
HLE(_ZN2nn2os15InitializeFiberEPNS0_9FiberTypeEPFS2_PvES3_S3_mi) {
    HosFiber *f = fiber_of_ft(ARG(0));
    if (!f) {
        f = (HosFiber *)calloc(1, sizeof *f);
        f->ft = ARG(0);
#ifdef TRACY_ENABLE
        snprintf(f->tracy_name, sizeof f->tracy_name, "fiber %llx", (unsigned long long)ARG(0));
#endif
        AcquireSRWLockExclusive(&fibers_lock);
        f->next = fibers, fibers = f;
        ReleaseSRWLockExclusive(&fibers_lock);
    } else if (f->host) {
        DeleteFiber(f->host);
    }
    memset(&f->ctx, 0, sizeof f->ctx);
    f->fn = ARG(1), f->arg = ARG(2), f->stack = ARG(3), f->size = ARG(4);
    f->started = f->done = 0;
    f->host = CreateFiberEx(64 << 10, 64 << 20, FIBER_FLAG_FLOAT_SWITCH, fiber_main, f);
    if (!f->host) hos_fatal(c, "CreateFiberEx failed (%lu)", GetLastError());
    static volatile LONG logged;
    if (InterlockedIncrement(&logged) <= 8 || hos_trace)
        hos_log("fiber %#llx: function %#llx arg %#llx stack %#llx+%#llx flags %d", (unsigned long long)f->ft,
                (unsigned long long)f->fn, (unsigned long long)f->arg, (unsigned long long)f->stack,
                (unsigned long long)f->size, (int)ARG(5));
}

HLE(_ZN2nn2os13FinalizeFiberEPNS0_9FiberTypeE) {
    HosFiber *f = fiber_of_ft(ARG(0));
    if (f && f->host && &f->ctx != c) { DeleteFiber(f->host); f->host = NULL; f->done = 1; }
}

HLE(_ZN2nn2os15GetCurrentFiberEv) {
    HosFiber *f = c == &cur->ctx ? NULL : fiber_of_ctx(c);
    RET(f ? f->ft : 0);
}

/* SwitchToFiber(FiberType*): NULL goes back to the thread's own context */
HLE(_ZN2nn2os13SwitchToFiberEPNS0_9FiberTypeE) {
    HosFiber *to = NULL;
    if (ARG(0) && !(to = fiber_of_ft(ARG(0))))
        hos_fatal(c, "SwitchToFiber(%#llx): not an initialized fiber", (unsigned long long)ARG(0));
    fiber_switch(c, to);
}

/* Every thread's kernel handle (the tag its SDK mutexes hold), for hos_kernel_dump. */
void hos_thread_log_handles(void) {
    AcquireSRWLockShared(&threads_lock);
    /* (lr: the return address of the thread's last call, x0 its first argument: in a wait, which game code waits on
       what; the web has no stack walk) */
    for (int i = 0; i < nthreads; i++)
        hos_log("kernel:   handle %#x = %s: lr %#llx x0 %#llx x19 %#llx x20 %#llx x21 %#llx x22 %#llx",
                threads[i]->kernel_handle, threads[i]->name, (unsigned long long)threads[i]->ctx.x[30],
                (unsigned long long)threads[i]->ctx.x[0], (unsigned long long)threads[i]->ctx.x[19],
                (unsigned long long)threads[i]->ctx.x[20], (unsigned long long)threads[i]->ctx.x[21],
                (unsigned long long)threads[i]->ctx.x[22]);
    ReleaseSRWLockShared(&threads_lock);
}

/* ---- virtual cores (on unless HOS_VCORES=0) -------------------------------------------------------------------
   Horizon runs at most one thread per core, and TOTK keys per-core data (allocators, job queues) by the current core
   number without locks; both MacNativeGames (threads.c) and totk-pc-recomp (runtime/kernel.c, MIT) schedule the game's
   threads on three virtual cores for it. Here a guest thread holds one of hos_cores() cores while it runs guest code
   and gives it up where Horizon would switch threads: in every block (hos_cv_sleep, hos_thread_wait, hos_sleep_ns,
   the NVN waits), at a yield, at exit. A free core goes to the best waiter allowed on it (lower priority value first,
   then the one waiting longest). A waiter better than a holder asks it to give the core up (vc_kick), which it does at
   its next checkpoint (every nn::os call and SVC: hos_vcore_checkpoint), as Horizon would run the better thread at
   once. Guest code with no checkpoint (a spin loop) keeps its core, so a waiter that has waited too long runs anyway
   without one (an "escape", logged with the holders): HOS_VCORE_ESCAPE_US for a better waiter, HOS_VCORE_ESCAPE_EQ_US
   otherwise (both 2000000: 2 s). An escape runs a thread with no core while the holder still runs on it, so two threads
   share per-core state (#1929); it is kept only as a deadlock breaker. 250 ms was the default before: on the 25-minute
   route it escaped 6-8 times a run, at waits that ended by themselves within 1.2 s (0 escapes at 2 s, frame times the
   same or better). A thread that waits only because better threads keep winning its core (they take it in turns at every
   block, or one of them runs on) is starved, not deadlocked, and an escape runs it beside the holder for nothing: once
   it has waited HOS_VCORE_STARVE_US (1000000: 1 s; 0: never) it goes before every other waiter at the next release of a
   core it may use, and asks the worst holder of those cores to give the core up, whatever its priority. It runs a turn,
   the holders ask for the core back as for any worse thread, and a wait that long ends in a grant, not an escape (#262;
   the 5 s line counts these grants). Every 5 s the log has a "vcores:" line. GetCurrentCoreNumber answers the
   core a thread holds. */
#if !defined(__EMSCRIPTEN__)
static SRWLOCK vc_lock = SRWLOCK_INIT;
static HosThread *vc_owner[16];
static HosThread *vc_queue[1024];
static int vc_nq;
static struct { int64_t takes, waits, wait_ns, max_ns, escapes, kicks, yields, boosts, long_waits, since; } vc_st;

int hos_vcores_on(void) {
    static int on = -1;
    if (on < 0) {
        on = env_on_or("HOS_VCORES", 1);   /* (mkw-recomp: on unless HOS_VCORES=0; the game's job queues need it) */
    }
    return on;
}
static int vc_cores(void) { return hos_cores() > 16 ? 16 : hos_cores(); }
static unsigned vc_allowed(HosThread *t) {
    const unsigned all = (1u << vc_cores()) - 1;
    unsigned m = (unsigned)(t->core_mask & all);
    if (!m) m = 1u << hos_thread_core(t);
    return m;
}
/* a runs before b: a starved waiter first (the one waiting longest first among them), then by priority (vcp) */
/* (less its vc_boost while it has the core it took back after the replay: hos_vcore_back_first) */
static int vcp(const HosThread *t) { return (t->vc_prio_over ? t->vc_prio_over : t->priority) - t->vc_boost; }
static int vc_better(HosThread *a, HosThread *b) {
    if (a->vc_starved != b->vc_starved) return a->vc_starved;
    if (!a->vc_starved && vcp(a) != vcp(b)) return vcp(a) < vcp(b);
    return a->vc_since < b->vc_since;
}
/* (hookshot, 12/11) Background threads and the host's cores. On the console a thread on an idle core costs the others
   nothing. Here the six virtual cores run on the host's physical cores, and a 6-core machine has five for the game's
   threads once one is kept for the replay (hos_reserved_cpus): a background thread that runs while the frame's threads
   are busy takes its share of a physical core from one of them (the sibling logical processor). Measured at the Lost
   Woods' heavy spot, Link standing (still.ps1 with HOS_VCORE_WAITS, runs of one build): OptimizeChangeBanc and
   ParallelDecompThread both kept waiting for a core by chance 56.7 fps, one of them 54.3-55.2, neither 52.3-52.7; which
   one waits depends on the core the game gives them, different from run to run.
   So a background thread (priority value over 18: ParallelDecompThread 19 ... OptimizeChangeBanc 27; the frame's threads
   are 16 and 17) takes a free virtual core only while fewer than `cap` are held, and gives its core up when a frame thread
   takes one past the cap. cap: the physical cores the game's threads have (HOS_BG_CAP=<n>; 0: no limit, as before); no
   limit either when that is at least the number of virtual cores (a host with 8 cores or more). It waits at most
   HOS_BG_WAIT_US (8000) for that, then takes any free core and is left alone for HOS_BG_RUN_US (2000): slowed down while
   the frame's threads need the cores, never starved, and never run without a core because of this. */
static int64_t vc_bg_wait_ns = 8000000, vc_bg_run_ns = 2000000;
static int vc_bg_cap(void) {
    static int cap = -1;
    if (cap < 0) {
        const char *e = getenv("HOS_BG_CAP"), *w = getenv("HOS_BG_WAIT_US"), *r = getenv("HOS_BG_RUN_US");
        int n = 0;
        if (!e) {
            n = 0;   /* (off unless asked for: see the measure in the comment above) */
        } else if (strcmp(e, "auto")) {
            n = atoi(e);
        } else {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION info[256];
            DWORD len = sizeof info;
            if (GetLogicalProcessorInformation(info, &len))
                for (DWORD i = 0; i < len / sizeof info[0]; i++) n += info[i].Relationship == RelationProcessorCore;
            if (hos_reserved_cpus()) n--;
        }
        if (w) vc_bg_wait_ns = atoll(w) * 1000;
        if (r) vc_bg_run_ns = atoll(r) * 1000;
        cap = n >= 2 && n < vc_cores() ? n : 0;
        if (cap)
            hos_log("vcores: a background thread (priority over 18) takes a core only while fewer than %d are held, waiting "
                    "%lld us at most (HOS_BG_CAP=0: no limit)", cap, (long long)(vc_bg_wait_ns / 1000));
    }
    return cap;
}
static int vc_held_n(void) {
    int n = 0;
    for (int k = 0; k < vc_cores(); k++) n += vc_owner[k] != NULL;
    return n;
}
static int vcp(const HosThread *t);
/* t may not take a free core now: a background thread, the cap reached, its wait not over; holding vc_lock */
static int vc_capped(const HosThread *t) {
    const int cap = vc_bg_cap();
    return cap && vcp(t) > 18 && vc_held_n() >= cap && hos_now_ns() < t->vc_cap_until;
}
/* a free core t may run on (its last one, its ideal one, then the lowest), or -1; holding vc_lock */
static int vc_free_for(HosThread *t) {
    const unsigned m = vc_allowed(t);
    if (vc_capped(t)) return -1;
    if (t->vc_last > 0 && (m >> (t->vc_last - 1) & 1) && !vc_owner[t->vc_last - 1]) return t->vc_last - 1;
    const int ideal = hos_thread_core(t);
    if ((m >> ideal & 1) && !vc_owner[ideal]) return ideal;
    for (int c = 0; c < vc_cores(); c++)
        if ((m >> c & 1) && !vc_owner[c]) return c;
    return -1;
}
static void vc_take(HosThread *t, int c) {
    vc_owner[c] = t;
    t->vc_held = c + 1;
    t->vc_last = c + 1;
    __atomic_store_n(&t->vc_kick, 0, __ATOMIC_RELAXED);
    t->vc_starved = 0;
    t->vc_took = hos_now_ns();
    /* (hookshot, vc_bg_cap) a frame thread took a core past the cap: the worst background holder not left alone gives
       its own up at its next checkpoint (vc_kick 3: vc_yield lets it go even with nobody waiting for that core) */
    const int cap = vc_bg_cap();
    if (cap && vcp(t) <= 18 && vc_held_n() > cap) {
        HosThread *worst = NULL;
        int64_t now = 0;
        for (int k = 0; k < vc_cores(); k++) {
            HosThread *h = vc_owner[k];
            if (h && vcp(h) > 18 && !h->vc_kick && (!worst || vcp(h) > vcp(worst))) {
                if (!now) now = hos_now_ns();
                if (now >= h->vc_free_until) worst = h;
            }
        }
        if (worst) {
            worst->vc_kick = 3;
            Ctx *rc = worst->run_ctx;
            __atomic_store_n(rc ? &rc->poll : &worst->ctx.poll, 1, __ATOMIC_RELAXED);
            vc_st.kicks++;
        }
    }
    t->vc_held_at = hos_now_ns(), t->vc_kick_at = 0;
}
static void vc_dequeue(HosThread *t) {
    for (int i = 0; i < vc_nq; i++)
        if (vc_queue[i] == t) {
            const int last = vc_nq - 1;
            vc_queue[i] = vc_queue[last];
            __atomic_store_n(&vc_nq, last, __ATOMIC_RELEASE);
            break;
        }
    t->vc_queued = 0;
}
/* (HOS_CHAIN: a thread's wait for a core, and the grant that ends it; bit 63: never a guest address) */
static uint64_t vc_chain_key(HosThread *t) { return 1ULL << 63 | (uint64_t)(uintptr_t)t; }
/* core c is free: to the best waiter allowed on it; holding vc_lock */
static void vc_grant(int c, HosThread *by) {
    HosThread *best = NULL;
    for (int i = 0; i < vc_nq; i++)
        if ((vc_allowed(vc_queue[i]) >> c & 1) && (!best || vc_better(vc_queue[i], best)) && !vc_capped(vc_queue[i]))
            best = vc_queue[i];
    if (!best) return;
    vc_st.boosts += best->vc_starved;
    vc_dequeue(best);
    if (by) {   /* (the long-wait line's cause: vc_long_wait) */
        const int64_t now = hos_now_ns();
        memcpy(best->vc_by, by->name, sizeof best->vc_by);
        best->vc_by_prio = by->priority;
        best->vc_by_held = by->vc_held_at ? now - by->vc_held_at : -1;
        best->vc_by_asked = by->vc_kick_at ? now - by->vc_kick_at : -1;
    }
    vc_take(best, c);
    HOS_CHAIN(HOS_CHAIN_SIGNAL, vc_chain_key(best));
    WakeConditionVariable(&best->vc_cv);
}
static void vc_thread_log(double secs);
/* (hookshot, HOS_VCORE_WAITS=1, diagnostic) who waits for a core: by thread name, the waits and their time since the last
   "vcores:" line, the eight longest printed with it. vc_lock held */
static struct { char name[48]; int64_t ns; int n; } vc_by[96];
static int vc_waits_on(void) {
    static int on = -1;
    if (on < 0) on = env_on("HOS_VCORE_WAITS");
    return on;
}
static void vc_note_wait(const HosThread *t, int64_t ns) {
    if (!vc_waits_on()) return;
    int i = 0;
    while (i < 96 && vc_by[i].name[0] && strcmp(vc_by[i].name, t->name)) i++;
    if (i == 96) return;
    if (!vc_by[i].name[0]) snprintf(vc_by[i].name, sizeof vc_by[i].name, "%s", t->name[0] ? t->name : "?");
    vc_by[i].ns += ns, vc_by[i].n++;
}
static void vc_waits_line(void) {
    if (!vc_waits_on()) return;
    char line[900];
    int len = 0;
    for (int k = 0; k < 8; k++) {
        int best = -1;
        for (int i = 0; i < 96 && vc_by[i].name[0]; i++)
            if (vc_by[i].ns > 0 && (best < 0 || vc_by[i].ns > vc_by[best].ns)) best = i;
        if (best < 0) break;
        len += snprintf(line + len, sizeof line - len, " %s %d waits %.1f ms;", vc_by[best].name, vc_by[best].n, vc_by[best].ns / 1e6);
        vc_by[best].ns = -1;   /* (printed) */
    }
    if (len) hos_log("vcores: waited most:%s", line);
    for (int i = 0; i < 96; i++) vc_by[i].ns = 0, vc_by[i].n = 0;
}
static void vc_stats_locked(int64_t now) {
    if (!vc_st.since) vc_st.since = now;
    if (now - vc_st.since < 5000000000LL) return;
    hos_log("vcores: %lld takes, %lld waits (mean %.0f us, longest %.1f ms, %lld over HOS_VCORE_LOG_MS), %lld kicks, %lld yields, "
            "%lld starved grants, %lld escapes in %.0f s",
            (long long)vc_st.takes, (long long)vc_st.waits, vc_st.waits ? vc_st.wait_ns / 1e3 / vc_st.waits : 0.0,
            vc_st.max_ns / 1e6, (long long)vc_st.long_waits, (long long)vc_st.kicks, (long long)vc_st.yields,
            (long long)vc_st.boosts, (long long)vc_st.escapes, (now - vc_st.since) / 1e9);
    vc_waits_line();
    vc_thread_log((now - vc_st.since) / 1e9);
    memset(&vc_st, 0, sizeof vc_st);
    vc_st.since = now;
}
/* HOS_VCORE_LOG=1 (diagnostic): with the "vcores:" line, each thread that waited for a core: its waits, the time in
   them, its priority and core mask (which threads the six cores hold back, and from whom) */
static struct { HosThread *t; int64_t n, ns; } vc_th[64];
static void vc_thread_note(HosThread *t, int64_t waited) {   /* holding vc_lock */
    static int on = -1;
    if (on < 0) on = env_on("HOS_VCORE_LOG");
    if (!on) return;
    int i = 0;
    while (i < 64 && vc_th[i].t && vc_th[i].t != t) i++;
    if (i == 64) return;
    vc_th[i].t = t;
    vc_th[i].n++;
    vc_th[i].ns += waited;
}
static void vc_thread_log(double secs) {   /* holding vc_lock */
    for (int k = 0; k < 64 && vc_th[k].t; k++) {
        int best = k;
        for (int j = k + 1; j < 64 && vc_th[j].t; j++)
            if (vc_th[j].ns > vc_th[best].ns) best = j;
        const __typeof__(vc_th[0]) tmp = vc_th[k];
        vc_th[k] = vc_th[best];
        vc_th[best] = tmp;
        if (k < 16)
            hos_log("vcores: %s (priority %d, mask %#llx, ideal %d): %lld waits, %.2f ms a second", vc_th[k].t->name,
                    vc_th[k].t->priority, (unsigned long long)vc_th[k].t->core_mask, vc_th[k].t->ideal_core,
                    (long long)vc_th[k].n, vc_th[k].ns / 1e6 / secs);
    }
    memset(vc_th, 0, sizeof vc_th);
}
static int64_t vc_env_ns(const char *name, int64_t us) {
    const char *e = getenv(name);
    return (e ? atoll(e) : us) * 1000;
}
/* Where a core's holder is now, for the escape line: " at <module>+<offset>" of its host thread's instruction pointer
   (Hookshot.exe's link map names the lifted function, f_<guest address>, or the host function around it), sampled with
   the thread stopped for the one GetThreadContext call. Nothing is called while it is stopped: a lock it holds (the heap's,
   the loader's) cannot be waited for here */
static void vc_where(HosThread *h, char *out, size_t cap) {
    out[0] = 0;
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
    if (!h->handle || SuspendThread(h->handle) == (DWORD)-1) return;
    CONTEXT ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.ContextFlags = CONTEXT_CONTROL;
    const BOOL got = GetThreadContext(h->handle, &ctx);
    ResumeThread(h->handle);
    if (!got) return;
    HMODULE mod = NULL;
    char path[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)(uintptr_t)ctx.Rip, &mod))
        GetModuleFileNameA(mod, path, sizeof path);
    const char *base = strrchr(path, '\\');
    snprintf(out, cap, " at %s+%#llx", base ? base + 1 : path, (unsigned long long)(ctx.Rip - (uintptr_t)mod));
#else
    (void)h, (void)cap;
#endif
}
/* (hookshot) HOS_VCORE_LOG_MS=<ms> (50 by default; 0: none): a wait for a core that long is logged with its cause, the thread
   whose release ended it (its priority, how long it had held the core, how long before it let go it was asked to) or how it
   ended otherwise, and the holders of the cores the thread may use as it ends (how long each has held its core, and whether it
   was asked). The first 200 a run; the 5 s "vcores:" line counts them all. Holding vc_lock */
static int64_t vc_log_ns(void) {
    static int64_t v = -1;
    if (v < 0) {
        const char *e = getenv("HOS_VCORE_LOG_MS");
        v = (e && *e ? atoll(e) : 50) * 1000000;
        if (v < 0) v = 0;
    }
    return v;
}
static void vc_long_wait(HosThread *t, int64_t waited, int how, unsigned m) {
    vc_st.long_waits++;
    static int logged;
    if (logged >= 200 && !hos_profiling_dir()) return;   /* (the profiling mode's events.txt takes them all) */
    const int log = logged < 200;
    logged += log;
    const int64_t now = hos_now_ns();
    char by[160] = "", who[320] = "";
    if (how == 0 && t->vc_by[0])
        snprintf(by, sizeof by, "given by %.47s (priority %d), which held it %.1f ms and %s", t->vc_by, t->vc_by_prio,
                 t->vc_by_held / 1e6, t->vc_by_asked < 0 ? "was never asked" : "");
    if (how == 0 && t->vc_by[0] && t->vc_by_asked >= 0) {
        const size_t l = strlen(by);
        snprintf(by + l, sizeof by - l, "let it go %.1f ms after it was asked", t->vc_by_asked / 1e6);
    }
    if (!by[0]) snprintf(by, sizeof by, "%s", how == 0 ? "given at a release" : how == 1 ? "found free" : "ran without one");
    int len = 0;
    for (int k = 0; k < vc_cores() && len < (int)sizeof who - 90; k++)
        if (m >> k & 1) {
            HosThread *h = vc_owner[k];
            len += h && h != t ? snprintf(who + len, sizeof who - len, " core %d: %.40s (%d), %.1f ms%s;", k, h->name, h->priority,
                                          h->vc_held_at ? (now - h->vc_held_at) / 1e6 : 0.0, h->vc_kick ? ", asked" : "")
                 : snprintf(who + len, sizeof who - len, " core %d: %s;", k, h ? "it" : "-");
        }
    if (log)
        hos_log("vcores: %s (priority %d) waited %.1f ms for a core: %s; the cores it may use now:%s", t->name, t->priority,
                waited / 1e6, by, who);
    /* (HOS_PROFILE=1: at its time, and when the wait began, for scripts/profile_report.py to put it in its frame) */
    hos_profiling_event("vcore_wait %.1f ms from %lld: %s (priority %d) waited for a core: %s; the cores it may use now:%s",
                        waited / 1e6, (long long)(now - waited), t->name, t->priority, by, who);
}
static void vc_acquire(HosThread *t) {
    static int64_t esc_better = -1, esc_eq, slice, spin_ns, starve;
    int spun = 0;
    if (esc_better < 0) {
        spin_ns = vc_env_ns("HOS_VCORE_SPIN_US", 0);
        /* (2 s: an escape runs a thread with no core while the holder still runs on it, so it
           is only a deadlock breaker; 250 ms before, reached 20-34 times at every start while the opening movie plays) */
        esc_eq = vc_env_ns("HOS_VCORE_ESCAPE_EQ_US", 2000000);
        esc_better = vc_env_ns("HOS_VCORE_ESCAPE_US", 2000000);
        slice = vc_env_ns("HOS_VCORE_SLICE_US", 2000);
        starve = vc_env_ns("HOS_VCORE_STARVE_US", 1000000);
    }
    AcquireSRWLockExclusive(&vc_lock);
    vc_st.takes++;
    if (vc_bg_cap() && vcp(t) > 18) {   /* (vc_bg_cap: how long it waits for the cores held to be few enough) */
        const int64_t now = hos_now_ns();
        t->vc_cap_until = now < t->vc_free_until ? 0 : now + vc_bg_wait_ns;
    }
    int c = vc_free_for(t);
    if (c >= 0) {
        vc_take(t, c);
        ReleaseSRWLockExclusive(&vc_lock);
        return;
    }
    /* (TICK-058) the record says whom it waits behind: a = its allowed cores | its priority << 8 | the holder's << 16,
       b = the holder of its first allowed core (tools/bench/critpath.py names it) */
    if (__builtin_expect(hos_jt_on, 0)) {
        const unsigned am = vc_allowed(t);
        HosThread *h = am ? vc_owner[__builtin_ctz(am)] : NULL;
        HOS_JT(JT_VC_WAIT_B, am | (uint64_t)(t->priority & 0xff) << 8 | (uint64_t)(h ? h->priority & 0xff : 0xff) << 16,
               (uintptr_t)h);
    }
    const int64_t t0 = hos_now_ns();
#ifdef TRACY_ENABLE
    HT_WAIT(tz, "vcore wait");   /* (a guest thread ready to run, waiting for one of the game's cores) */
#endif
    HOS_CHAIN(HOS_CHAIN_WAIT, vc_chain_key(t));
    t->vc_since = t0;
    t->vc_by[0] = 0;
    int how = 1;   /* (vc_long_wait: 0 a grant, 1 found free, 2 an escape) */
    t->vc_queued = 1;
    if (vc_nq < 1024) {
        vc_queue[vc_nq] = t;
        __atomic_store_n(&vc_nq, vc_nq + 1, __ATOMIC_RELEASE);
    }
    vc_st.waits++;
    for (;;) {
        /* the worst holder worse than t on a core t may use gives it up at its next checkpoint; after
           HOS_VCORE_SLICE_US (2000) of waiting, a holder as good as t does too (a time slice: Horizon would leave
           it running, but its threads never run two at once on one core, which the game's per-core data needs); once
           t is starved (HOS_VCORE_STARVE_US), the worst holder there is, whatever its priority. The holder sees the
           request at its next nn::os call or SVC, or at a loop header (the interrupt polls) */
        const unsigned m = vc_allowed(t);
        HosThread *worst = NULL;
        for (int k = 0; k < vc_cores(); k++) {
            HosThread *h = vc_owner[k];
            if ((m >> k & 1) && h && vcp(h) > vcp(t) && (!worst || vcp(h) > vcp(worst))) worst = h;
        }
        if (!worst && hos_now_ns() - t0 > slice)
            for (int k = 0; k < vc_cores() && !worst; k++)
                if ((m >> k & 1) && vc_owner[k] && vcp(vc_owner[k]) == vcp(t)) worst = vc_owner[k];
        if (!worst && t->vc_starved)
            for (int k = 0; k < vc_cores(); k++) {
                HosThread *h = vc_owner[k];
                if ((m >> k & 1) && h && (!worst || vcp(h) > vcp(worst))) worst = h;
            }
        if (worst && !worst->vc_kick) {
            __atomic_store_n(&worst->vc_kick, vcp(worst) > vcp(t) ? 1 : 2, __ATOMIC_RELAXED);
            worst->vc_kick_at = hos_now_ns();
            Ctx *rc = worst->run_ctx;   /* (a fiber's context while it runs one) */
            __atomic_store_n(rc ? &rc->poll : &worst->ctx.poll, 1, __ATOMIC_RELAXED);
            vc_st.kicks++;
        }
        /* (hookshot, HOS_VCORE_SPIN_US, experiment) before its first sleep the waiter spins a little for a release to give it
           a core: the frame's threads hand cores to each other all the time (4000 waits in 5 s in the Kokiri village, 0.4-0.8 ms
           each), and a sleep and its wake-up are far longer than the hand-over itself. Once a wait: a thread kept waiting
           (background work behind the frame's threads) sleeps as before */
        if (!spun && spin_ns > 0) {
            spun = 1;
            ReleaseSRWLockExclusive(&vc_lock);
            const int64_t until = hos_now_ns() + spin_ns;
            while (!__atomic_load_n(&t->vc_held, __ATOMIC_RELAXED) && hos_now_ns() < until) __builtin_ia32_pause();
            AcquireSRWLockExclusive(&vc_lock);
            if (t->vc_held) break;   /* granted by a release, while spinning */
            if ((c = vc_free_for(t)) >= 0) {
                vc_dequeue(t);
                vc_take(t, c);
                break;
            }
            continue;   /* (the holders asked again, then the sleep) */
        }
        SleepConditionVariableSRW(&t->vc_cv, &vc_lock, 1, 0);
        if (t->vc_held) {   /* granted by a release */
            how = 0;
            break;
        }
        if ((c = vc_free_for(t)) >= 0) {
            vc_dequeue(t);
            vc_take(t, c);
            break;
        }
        const int64_t waited = hos_now_ns() - t0;
        if (starve > 0 && waited > starve) t->vc_starved = 1;
        if (waited > (worst ? esc_better : esc_eq)) {
            vc_dequeue(t);
            t->vc_starved = 0;
            t->vc_escaped = 1;
            t->vc_boost = 0;   /* (no core to keep a place on) */
            how = 2;
            vc_st.escapes++;
            static int logged;
            if (logged < 400) {   /* (40 were all spent in a zone's load: the escapes of steady play went unlogged) */
                logged++;
                /* (each holder with how long it has held its core: a holder that took it after this wait began handed
                   it on in between; one that has held it all along ran that long without an nn::os call, an SVC or a
                   loop poll, or sat blocked in host code with its core: where it is then, vc_where) */
                char who[480] = "";
                int len = 0;
                const int64_t now = hos_now_ns();
                for (int k = 0; k < vc_cores() && len < 360; k++)
                    if (m >> k & 1) {
                        HosThread *h = vc_owner[k];
                        char at[96] = "";
                        if (h && now - h->vc_took > esc_eq / 2) vc_where(h, at, sizeof at);
                        len += snprintf(who + len, sizeof who - len, " core %d: %s (%d)%s, held %.0f ms%s;", k,
                                        h ? h->name : "-", h ? h->priority : 0, h && h->vc_kick ? " asked" : "",
                                        h ? (now - h->vc_took) / 1e6 : 0.0, at);
                    }
                hos_log("vcores: %s (priority %d) ran without a core after %.1f ms:%s", t->name, t->priority, waited / 1e6, who);
            }
            break;
        }
    }
    const int64_t waited = hos_now_ns() - t0;
#ifdef TRACY_ENABLE
    HT_ZONE_TEXTF(tz, "%s%.47s, priority %d", how == 0 ? "given by " : how == 1 ? "found free" : "ran without one",
                  how == 0 ? t->vc_by : "", t->priority);
    HT_ZONE_END(tz);
#endif
    HOS_CHAIN(HOS_CHAIN_WOKE, vc_chain_key(t));
    if (t->vc_cap_until && t0 + waited >= t->vc_cap_until) t->vc_free_until = t0 + waited + vc_bg_run_ns;   /* (vc_bg_cap) */
    vc_st.wait_ns += waited;
    if (waited > vc_st.max_ns) vc_st.max_ns = waited;
    vc_thread_note(t, waited);
    t->cc_vcw++, t->cc_vcw_ns += waited;
    vc_note_wait(t, waited);
    if (vc_log_ns() && waited >= vc_log_ns()) vc_long_wait(t, waited, how, vc_allowed(t));
    if (hos_fps_timing) t->vc_wait_total += waited;
    vc_stats_locked(hos_now_ns());
    ReleaseSRWLockExclusive(&vc_lock);
    HOS_JT(JT_VC_WAIT_E, 0, 0);
}
static void vc_release_locked(HosThread *t) {
    const int c = t->vc_held - 1;
    vc_owner[c] = NULL;
    t->vc_held = 0;
    __atomic_store_n(&t->vc_kick, 0, __ATOMIC_RELAXED);
    t->vc_boost = 0;   /* (its place ahead ends with the core it was for: hos_vcore_back_first) */
    vc_grant(c, t);
    /* (hookshot, vc_bg_cap) fewer cores held: a background thread kept waiting by the cap may take a free core of its own
       mask now, not only this one */
    if (vc_nq && vc_bg_cap())
        for (int k = 0; k < vc_cores(); k++)
            if (!vc_owner[k]) vc_grant(k, t);
}
static void vc_release(HosThread *t) {
    if (!t->vc_held) return;
    AcquireSRWLockExclusive(&vc_lock);
    vc_release_locked(t);
    ReleaseSRWLockExclusive(&vc_lock);
}
/* gives the core to a waiter allowed on it that is better than t (strict) or as good (a yield), or starved */
static void vc_yield(HosThread *t, int strict) {
    if (!t->vc_held) return;
    /* An empty queue has nobody to yield to. A later waiter can still request a checkpoint.
       A stale kick goes through the lock so it is cleared without losing a newer request. */
    if (!__atomic_load_n(&vc_nq, __ATOMIC_ACQUIRE) && !__atomic_load_n(&t->vc_kick, __ATOMIC_RELAXED)) return;
    AcquireSRWLockExclusive(&vc_lock);
    const int c = t->vc_held - 1;
    int any = 0;
    for (int i = 0; i < vc_nq && !any; i++) {
        HosThread *w = vc_queue[i];
        any = (vc_allowed(w) >> c & 1) && (w->vc_starved || (strict ? vcp(w) < vcp(t) : vcp(w) <= vcp(t)));
    }
    /* (hookshot, vc_bg_cap) asked by vc_take: a background thread while too many cores are held */
    if (!any && __atomic_load_n(&t->vc_kick, __ATOMIC_RELAXED) == 3 && vc_held_n() > vc_bg_cap() &&
        hos_now_ns() >= t->vc_free_until) any = 1;
    __atomic_store_n(&t->vc_kick, 0, __ATOMIC_RELAXED);
    if (any) {
        const int boost = t->vc_boost;   /* (a yield to a better waiter is not the block that ends its place ahead) */
        vc_st.yields++;
        vc_release_locked(t);
        t->vc_boost = boost;
    }
    ReleaseSRWLockExclusive(&vc_lock);
    if (any) {
#ifdef TRACY_ENABLE
        HT_SCOPE("vcore yield");
#endif
        vc_acquire(t);
    }
}
/* 1 when the caller is to take a core again after its block (hos_vcore_back): it gave its core up, or it has run without
   one since an escape. An escape lasts until the thread's next block, as every other thread waits for a core after one;
   it took none again before, so an escaped thread ran with no core, and the core number of a core another thread held,
   for the rest of its life (issue 1929) */
int hos_vcore_away(void) {
    HosThread *t = cur;
    if (!hos_vcores_on() || !t) return 0;
    if (!t->vc_held) {
        const int escaped = t->vc_escaped;   /* (only the outermost block of nested ones takes the core again) */
        t->vc_escaped = 0;
        return escaped;
    }
    if (hos_fps_timing) t->away_at = hos_now_ns();   /* (the frame trace's blocked time: to hos_vcore_back) */
    vc_release(t);
    return 1;
}
/* (hookshot) hos_vcore_back after host-only work on the frame's path (the replay in nvnQueueSubmitCommands): the thread
   takes a core ahead of threads up to HOS_VCORE_REPLAY_BOOST priority steps better than it (2), and keeps that place until
   it next gives the core up. OoT's Presentation Thread (priority 17, any core) replays the frame; its six job executors
   and five module workers (priority 16, one core each) run their batches meanwhile, so after each replay it waited
   ~1.1 ms for one of them to finish, to run a few microseconds of game code before the next replay. 0: as before */
void hos_vcore_back_first(void) {
    static int steps = -1;
    if (steps < 0) steps = getenv("HOS_VCORE_REPLAY_BOOST") ? atoi(getenv("HOS_VCORE_REPLAY_BOOST")) : 2;
    HosThread *t = cur;
    if (!hos_vcores_on() || !t) return;
    t->vc_escaped = 0;
    if (t->vc_held) return;
    t->vc_boost = steps > 0 && steps < 64 ? steps : 0;   /* (not queued, no core: no other thread reads it yet) */
    vc_acquire(t);
}
void hos_vcore_back(void) {
    HosThread *t = cur;
    if (!hos_vcores_on() || !t) return;
    /* after it the thread holds a core or has escaped again (vc_acquire sets the flag again): the escape the flag stood
       for has ended, also at a back without its away (a thread's start). A flag left up while a core is held would make
       a later inner block's away answer 1 and take a core in the middle of the outer block */
    t->vc_escaped = 0;
    if (t->away_at) t->blocked_total += hos_now_ns() - t->away_at, t->away_at = 0;
    if (!t->vc_held) vc_acquire(t);
}
/* Lifted code calls this from its interrupt polls with the loop's guest registers still in host registers (a64_poll_cur,
   mkw_glue.c: the poll hands over no context, so the compiler need not spill them). It, and everything it calls, must never
   read or write the thread's guest context (Ctx) or run guest code: what it would see there may be older than the loop. A
   stall dump of a thread spinning in a polled loop shows such older registers for the same reason. */
/* (Hookshot, sprint 5) HOS_BG_THROTTLE=<run us>:<sleep us> (300:3000 by default, as in the fork; =0 off): OoT's OptimizeChangeBanc (guest
   priority 27, the lowest: on the console it gets only the time nothing else wants) runs without end here, 35-44% of a logical
   processor at the Market, at the host's lowest priority, which keeps it off no SMT sibling: the executor, module worker or
   replay thread on the other half of its core runs up to 1.7x slower while it does (the replay's measurement, hos_reserved_cpus).
   After <run> of running it gives its virtual core up and sleeps <sleep>, at a checkpoint (an nn::os call or SVC, where Horizon
   could switch it out too): ~9% of a processor instead. */
static void bg_throttle(HosThread *t) {
    static int64_t run_ns = -1, sleep_ms;
    if (run_ns < 0) {
        const char *e = getenv("HOS_BG_THROTTLE");
        long long r = 300, s = 3000;   /* (on by default, the fork's 300:3000: HOS_BG_THROTTLE=0 off) */
        if (e && sscanf(e, "%lld:%lld", &r, &s) < 1) r = 300;
        sleep_ms = s >= 1000 ? s / 1000 : 1;
        if (r > 0) hos_log("threads: OptimizeChangeBanc runs %lld us, then sleeps %lld ms (HOS_BG_THROTTLE)", r, (long long)sleep_ms);
        run_ns = r > 0 ? r * 1000 : 0;
    }
    if (!t->bg_kind) t->bg_kind = strstr(t->name, "OptimizeChange") ? 1 : 2;
    if (t->bg_kind != 1 || !run_ns) return;
    const int64_t now = hos_now_ns();
    if (!t->bg_since) t->bg_since = now;
    if (now - t->bg_since < run_ns) return;
    const int away = hos_vcore_away();
    Sleep((DWORD)sleep_ms);
    if (away) hos_vcore_back();
    t->bg_since = hos_now_ns();
}
void hos_vcore_checkpoint(void) {
    HosThread *t = cur;
    if (t && t->bg_kind != 2) bg_throttle(t);
    const int kick = t ? __atomic_load_n(&t->vc_kick, __ATOMIC_RELAXED) : 0;
    if (kick) vc_yield(t, kick == 1);
}
void hos_vcore_yield(void) {
    HosThread *t = cur;
    if (hos_vcores_on() && t) vc_yield(t, 0);
}
/* SleepConditionVariableSRW for guest threads: the core given up for the wait; after it the lock is let go while a
   core is taken again (waiting for a core while holding it could stall the thread that will wake the next waiter),
   so callers see a spurious wake-up and look at their condition again, as they do anyway */
int hos_cv_sleep(void *cv, void *lk, unsigned ms) {
#ifdef TRACY_ENABLE
    HT_WAIT(tz, "asleep");   /* (inside a wait's zone: the part of it slept, after its spin; then the core taken again) */
#endif
    const int away = hos_vcore_away();
    const BOOL r = SleepConditionVariableSRW((PCONDITION_VARIABLE)cv, (PSRWLOCK)lk, ms, 0);
#ifdef TRACY_ENABLE
    HT_ZONE_END(tz);
#endif
    if (away) {
        ReleaseSRWLockExclusive((PSRWLOCK)lk);
        hos_vcore_back();
        AcquireSRWLockExclusive((PSRWLOCK)lk);
    }
    return r;
}
#else
int hos_vcores_on(void) { return 0; }
int hos_vcore_away(void) { return 0; }
void hos_vcore_back(void) {}
void hos_vcore_back_first(void) {}
void hos_vcore_checkpoint(void) {}
void hos_vcore_yield(void) {}
int hos_cv_sleep(void *cv, void *lk, unsigned ms) {
    return SleepConditionVariableSRW((PCONDITION_VARIABLE)cv, (PSRWLOCK)lk, ms, 0);
}
#endif
