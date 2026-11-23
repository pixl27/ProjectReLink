/* (hookshot) An A/B within one run: HOS_AB=<switch>=<value>[,<seconds>[,<warm-up seconds>]].

   Two runs of the same build differ by a few fps (the PC's other work, its clocks, where the game's threads land), more than
   the 2-5 % a change is often worth. Here one run measures both sides: after the warm-up (20 s from the start of the process),
   the named switch alternates every period (3 s) between its normal value (side 0, first) and <value> (side 1), and each
   frame is counted for the side it ran on, so whatever drifts slower than a period cancels. The idea is the A/B blocks of
   dkb-pc-recomp's runtime/perf.c (TOTK_AB); the code is Hookshot's own.

   A switch takes part when its reader asks hos_ab_value (host/hos.h) each time it is used, instead of only its value cached
   at the start: host/hos_thread.c's os_yield (HOS_OS_YIELD) and host/oot_barrier.c (HOS_OOT_BARRIER_WAIT) do, and
   tests/test_hos_ab.py lists every reader that
   does. Only a switch that can change between two calls without breaking what the other value built may be wired so; a
   switch whose value sets something up once, at the start or at the lift, may not.

   The side changes on a thread of its own (a sleep to the next flip on the host's clock, hos_now_ns), never at a present.
   The frame trace (RX_VK_FRAME_TRACE) gets each frame's side as its ab_side column (MkwVkHost.ab_frame): -1 outside the A/B,
   0 or 1, and 2 or 3 for that side's frames in a transition (a flip inside the frame, or less than AB_SETTLE_NS before its
   end), which scripts/ab_summary.py leaves out. The host also sums the presents' intervals by side (rx_nvn_present) and logs
   the sums at the end of the run (hos_close_finish): "ab: ...". docs/development.md, "An A/B within one run". */
#include "hos.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

void hos_host_thread_register_self(const char *name);   /* (mkw_glue.c: the thread's name) */

#define AB_SETTLE_NS 300000000LL   /* (a frame that ends this soon after a flip is the transition's) */
#define AB_PERIOD_S 3.0
#define AB_WARM_S 20.0

int hos_ab_side_now = -1;   /* (hos_ab_side: written by the A/B's thread alone) */

typedef struct AbPlan {
    int on, value;
    char name[64];
    int64_t t0, warm, period;   /* (ns: the start, the warm-up, a period) */
    double period_s, warm_s;
} AbPlan;
/* (set before the game's threads start, by hos_ab_start on the main thread, and not changed after: read without a lock) */
static AbPlan ab;
static int ab_read;   /* (a reader named the switch during the A/B: hos_ab_value_at) */
/* the names hos_ab_value_at has seen, by the literal's address: (address << 1) | 1 when it names the A/B's switch. One word a
   slot, so a slot read is whole; a slot two names share is looked up again with strcmp, and nothing else is lost */
static uintptr_t ab_seen[16];
static int64_t ab_last_present;
static int64_t ab_frames[4], ab_ns[4];   /* (the host's presents by hos_ab_frame's answer: 0, 1, 2, 3) */
static int ab_ended;

/* ---- the plan: HOS_AB's text read; 0 and a log line saying what is wrong, the run without an A/B */
static int ab_seconds(const char *s, size_t n, double lo, double hi, double *out) {
    char b[32];
    if (!n || n >= sizeof b) return 0;
    memcpy(b, s, n);
    b[n] = 0;
    char *end;
    const double v = strtod(b, &end);
    if (end != b + n || !(v >= lo && v <= hi)) return 0;   /* (a NaN fails both) */
    *out = v;
    return 1;
}
static int ab_parse(const char *text, AbPlan *p) {
    memset(p, 0, sizeof *p);
    p->period_s = AB_PERIOD_S, p->warm_s = AB_WARM_S;
    const char *why = NULL, *eq = strchr(text, '=');
    const size_t n = eq ? (size_t)(eq - text) : 0;
    if (!eq || !n) why = "not <switch>=<value>[,<seconds>[,<warm-up seconds>]]";
    else if (n >= sizeof p->name) why = "the switch's name is too long";
    else if (text[0] < 'A' || text[0] > 'Z') why = "the switch's name does not start with a capital";
    for (size_t i = 0; !why && i < n; i++)
        if (!((text[i] >= 'A' && text[i] <= 'Z') || (text[i] >= '0' && text[i] <= '9') || text[i] == '_'))
            why = "the switch's name has other than capitals, digits and _";
    const char *f[3] = {eq ? eq + 1 : text, NULL, NULL};   /* (the value, the period, the warm-up) */
    size_t len[3] = {0, 0, 0};
    int fields = 0;
    for (const char *s = f[0]; !why;) {
        const char *c = strchr(s, ',');
        if (fields == 3) {
            why = "more than three fields after the =";
            break;
        }
        f[fields] = s, len[fields] = c ? (size_t)(c - s) : strlen(s), fields++;
        if (!c) break;
        s = c + 1;
    }
    if (!why) {
        char b[24], *end = NULL;
        long v = 0;
        if (len[0] && len[0] < sizeof b) {
            memcpy(b, f[0], len[0]);
            b[len[0]] = 0;
            v = strtol(b, &end, 10);
        }
        if (!len[0] || len[0] >= sizeof b || end != b + len[0] || v < -1000000000L || v > 1000000000L)
            why = "the value is not a whole number (the A/B gives the switch a number)";
        p->value = (int)v;
    }
    if (!why && fields > 1 && !ab_seconds(f[1], len[1], 1, 3600, &p->period_s))
        why = "the period is not a number of seconds from 1 to 3600";
    if (!why && fields > 2 && !ab_seconds(f[2], len[2], 0, 86400, &p->warm_s))
        why = "the warm-up is not a number of seconds from 0 to 86400";
    if (why) {
        hos_log("ab: HOS_AB=%.120s: %s: no A/B in this run", text, why);
        return 0;
    }
    memcpy(p->name, text, n);
    p->name[n] = 0;
    p->period = (int64_t)(p->period_s * 1e9);
    p->warm = (int64_t)(p->warm_s * 1e9);
    return 1;
}

/* ---- the schedule: periods of p->period from the warm-up's end, side 0 first */
static int ab_side_at(const AbPlan *p, int64_t t) {
    if (!__atomic_load_n(&p->on, __ATOMIC_ACQUIRE) || t < p->t0 + p->warm) return -1;
    return (int)(((t - p->t0 - p->warm) / p->period) & 1);
}
static int64_t ab_period_start(const AbPlan *p, int64_t t) {   /* (t at or after the warm-up's end) */
    return p->t0 + p->warm + (t - p->t0 - p->warm) / p->period * p->period;
}
/* the frame from from_ns to to_ns: its side by its end, + 2 when the period began inside it or less than AB_SETTLE_NS before
   its end (from_ns 0: not known, a transition) */
int hos_ab_frame(int64_t from_ns, int64_t to_ns) {
    const int side = ab_side_at(&ab, to_ns);
    if (side < 0) return -1;
    const int64_t start = ab_period_start(&ab, to_ns);
    return side + (from_ns < start || to_ns - start < AB_SETTLE_NS ? 2 : 0);
}
/* the side at now made the one hos_ab_side says; the time of the next change */
static int64_t hos_ab_tick(int64_t now) {
    if (!ab.on) return INT64_MAX;
    const int side = ab_side_at(&ab, now);
    const int was = __atomic_exchange_n(&hos_ab_side_now, side, __ATOMIC_RELAXED);
    if (side != was && was < 0)
        hos_log("ab: the warm-up is over: %s alternates now, its normal value and %d, %.1f s each", ab.name, ab.value, ab.period_s);
    else if (side != was)
        hos_log_at(HOS_LOG_DEBUG, "ab: side %d (%s %s)", side, ab.name, side ? "the A/B's value" : "its normal value");
    return side < 0 ? ab.t0 + ab.warm : ab_period_start(&ab, now) + ab.period;
}
/* the A/B set up from text, the warm-up counted from now (1), or none (0: no text, or a line said what is wrong) */
static int hos_ab_setup(const char *text, int64_t now) {
    AbPlan p;
    if (!text || !*text || !ab_parse(text, &p)) return 0;
    p.t0 = now;
    ab = p;
    __atomic_store_n(&ab.on, 1, __ATOMIC_RELEASE);   /* (the plan whole before a thread that reads it sees it on) */
    __atomic_store_n(&hos_ab_side_now, -1, __ATOMIC_RELAXED);
    hos_log("ab: HOS_AB: after %.0f s of warm-up, %s alternates every %.1f s between its normal value (side 0, first) and %d "
            "(side 1); the frame trace (RX_VK_FRAME_TRACE) has each frame's side in its ab_side column (scripts/ab_summary.py)",
            ab.warm_s, ab.name, ab.period_s, ab.value);
    return 1;
}

/* ---- the switches' side: hos_ab_value (host/hos.h) when an A/B runs */
int hos_ab_value_at(const char *name, int normal, int side) {
    const uintptr_t p = (uintptr_t)name;
    uintptr_t *slot = &ab_seen[(p ^ p >> 5 ^ p >> 11) & 15];
    uintptr_t e = __atomic_load_n(slot, __ATOMIC_RELAXED);
    if (e >> 1 != p) {   /* (a name not seen yet: once a call site, or after another took the slot) */
        const int match = !strcmp(name, ab.name);
        if (match) __atomic_store_n(&ab_read, 1, __ATOMIC_RELAXED);
        e = p << 1 | (uintptr_t)match;
        __atomic_store_n(slot, e, __ATOMIC_RELAXED);
    }
    return (e & 1) && side == 1 ? ab.value : normal;
}
int hos_ab(void) { return hos_ab_side() == 1; }   /* (TOTKPort's hos_prof.c name, which mkw_glue.c stood in for: side 1 runs) */

/* ---- the host's sums: every present's interval by its side (rx_nvn_present, the thread that presents) */
void hos_ab_present(void) {
    if (!__atomic_load_n(&ab.on, __ATOMIC_ACQUIRE)) return;
    const int64_t now = hos_now_ns(), from = __atomic_exchange_n(&ab_last_present, now, __ATOMIC_RELAXED);
    const int s = from ? hos_ab_frame(from, now) : -1;
    if (s < 0) return;
    __atomic_add_fetch(&ab_frames[s], 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&ab_ns[s], now - from, __ATOMIC_RELAXED);
}
/* their line: each side's frames and mean frame time, the transitions left out */
static int hos_ab_summary(char *out, size_t cap) {
    int64_t n[4], ns[4];
    for (int i = 0; i < 4; i++) n[i] = __atomic_load_n(&ab_frames[i], __ATOMIC_RELAXED), ns[i] = __atomic_load_n(&ab_ns[i], __ATOMIC_RELAXED);
    const double m0 = n[0] ? ns[0] / 1e6 / n[0] : 0, m1 = n[1] ? ns[1] / 1e6 / n[1] : 0;
    int at = snprintf(out, cap, "ab: %s=%d against its normal value, %.1f s periods after %.0f s: ", ab.name, ab.value, ab.period_s,
                      ab.warm_s);
    for (int s = 0; s < 2 && at < (int)cap; s++) {
        const double m = s ? m1 : m0;
        at += n[s] ? snprintf(out + at, cap - at, "side %d (%s) %lld frames, %.3f ms a frame (%.1f fps); ", s,
                              s ? "the A/B's value" : "normal", (long long)n[s], m, 1000 / m)
                   : snprintf(out + at, cap - at, "side %d: no frames; ", s);
    }
    if (at < (int)cap && n[0] && n[1])
        at += snprintf(out + at, cap - at, "side 1 %+.2f %% frame time; ", 100 * (m1 - m0) / m0);
    if (at < (int)cap)
        at += snprintf(out + at, cap - at, "%lld frames in transitions left out (the host's presents; scripts/ab_summary.py "
                                           "on the frame trace for medians and a confidence interval)", (long long)(n[2] + n[3]));
    if (at < (int)cap && !__atomic_load_n(&ab_read, __ATOMIC_RELAXED))
        at += snprintf(out + at, cap - at, "; %s was never read during the A/B: no switch of that name takes part, both sides "
                                           "ran the same", ab.name);
    return at;
}
void hos_ab_end(void) {   /* (once, at the end of the run) */
    if (!ab.on || __atomic_exchange_n(&ab_ended, 1, __ATOMIC_RELAXED)) return;
    char line[768];
    hos_ab_summary(line, sizeof line);
    hos_log("%s", line);
}

/* ---- the thread that flips the side */
static DWORD WINAPI ab_main(void *arg) {
    (void)arg;
    hos_host_thread_register_self("A/B");
    for (;;) {
        const int64_t now = hos_now_ns(), next = hos_ab_tick(now), ms = (next - now + 999999) / 1000000;
        Sleep((DWORD)(ms < 1 ? 1 : ms > 60000 ? 60000 : ms));
    }
    return 0;
}
void hos_ab_start(void) {
    if (!hos_ab_setup(getenv("HOS_AB"), hos_now_ns())) return;
    if (!CreateThread(NULL, 64u << 10, ab_main, NULL, 0, NULL)) {
        __atomic_store_n(&ab.on, 0, __ATOMIC_RELEASE);
        hos_log("ab: the A/B's thread did not start: no A/B in this run");
    }
}
