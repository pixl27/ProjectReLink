/* (Hookshot native engine) The OS layer's side of the graphics front end (engine/native, hl/c/hl_gfx.h;
   docs/native-engine/gfx-record-job.md).

   The graphics module's recording job broadcasts to a list of listeners while it holds a global lock; the largest
   listener is the scene's recording pass (guest 0x806602e0, reached only through a pointer), answered here instead of by
   its translation (hl_host.h HL_ANSWER: the lifted code is not touched).
     HL_GFX=0  off: the translated listener, as before
     HL_GFX=1  the native pass (hl::gfx::RecordPass; the default): its own level native, its phases still translated
   HL_GFX_CHECK=1: the pass check. Every NVN procedure the game is given goes through a tap that, while a pass runs on the
   calling thread, adds the call to that pass's recording; with HL_GFX=1 each pass takes the native or the translated path
   at random, and the counters compare the two (hl/gfx/PassCheck.h says what that proves). With HL_GFX=0 every pass is the
   translated one, recorded. The tap is decided when the game asks for its first NVN procedure, so the switch is read
   then; without it the procedures are the NVN layer's own, untouched.
   HL_GFX_CHECK=2: the check's control: the passes drawn for the native path run the translated pass too, so the
   differences it then shows between the paths are the check's own noise.
   HL_GFX_LOG=1: the counters every 10 s (always with the check).

   The draw path's render-state function (guest 0x80077d60, called directly from 254 functions) is answered through an
   entry gate (HL_GATE below; nx2/hl_gates.py puts it in the lifted function at build time):
     HL_GFX_STATE=0  off: the translated function, as before
     HL_GFX_STATE=1  (the default) the native function (hl::gfx::RenderState): it writes the block and calls the NVN procedures itself,
                     through the table of NVN entry points below (the procedures the game's own globals hold, checked
                     once against them); a call that reaches the material's state callback is left to the translation
   With HL_GFX_CHECK=1 (or 2) the gate checks instead, call by call (hl/gfx/StateCheck.h): the native version runs into a
   recording, the translated body runs for real under the tap, and the two lists of NVN calls and the block's bytes are
   compared; the first mismatches are logged in full.

   The draw path's sub-mesh submission (guest 0x80078fc0, a virtual method its callers call directly) is answered through
   an entry gate too (gfx-record-job.md section 9):
     HL_GFX_DRAW=0  off: the translated function, as before
     HL_GFX_DRAW=1  (the default) the native function on its direct paths (hl::gfx::SubmeshDraw): the prologue native,
                    the shared preparation 0x8008fc00 called back on the same context, the submission native
                    (pre-recorded list, indexed instanced draw or mesh tasks, and the block's triangle counter); a call
                    on any other path (the cluster, indirect, query and inline-write paths) is left whole to the
                    translation
   With HL_GFX_CHECK=1 (or 2) it checks instead, in two segments around the preparation: a second gate at 0x8008fc00 is
   the checkpoint, for the call the checked body makes (gfx-record-job.md section 9 says how).

   That gate is also the shared preparation's own (guest 0x8008fc00, 263 callers; gfx-record-job.md section 10):
     HL_GFX_PREP=0  the translated function, as before
     HL_GFX_PREP=1  (the default) native on its cached, program-bound and (HL_GFX_PREP_STATE=1, the default) state paths
                    (hl::gfx::Prepare); every other call is left whole to the translation
   With HL_GFX_CHECK=1 (or 2) every call is checked instead: its route counted, the native version compared call by call
   on the cached path, and the translated body's NVN calls filed by kind in the paths' histogram.

   The recording job's item body (guest 0x80084de0, called directly by the job runners) has a gate too, for its map only:
   with HL_GFX_CHECK=1 each item is observed (its route, draws and NVN calls counted: "loop map" below; its list read at
   the entry: "items by list"; its off-loop draws filed by the variant or method they came from: "variants", "methods");
   nothing of it is native yet, and without the check the gate is off.
   HL_GFX_TIME=1: the three gates' crossings counted and timed (below, "the counters"). */
#include "hl_host.h"
#include "hl/c/hl_gfx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define HL_GFX_DEFAULT HL_GFX_SHELL   /* (on by default, as in the fork; =0 off) */   /* (validated in the three scenes, check and shots clean: gfx-record-job.md section 7) */
#define HL_RECORD_PASS 0x806602e0ULL   /* (hl::bridge::kRecordPass: hl_gfx_on checks the two agree) */
#define HL_GFX_STATE_DEFAULT 1   /* (on by default, as in the fork; =0 off) */ /* the fork: (validated 2026-11-13 on both PCs: the exact check at 0 mismatches over 4.2 M NVN calls in three scenes, native runs clean) */
#define HL_STATE_FN 0x80077d60ULL      /* (hl::bridge::kApplyRenderState: state_init checks the two agree) */

#define HL_VAR_PREPARE_RETURN 0x8008d8ecULL   /* the key-0xa0 function's call of the preparation (hl_gfx_var_prepare_return) */
#define HL_LOOP_FN 0x80084de0ULL             /* the recording job's item body (gfx-record-job.md section 11) */
#define HL_LOOP_SUBMIT_RETURN 0x80085ea4ULL  /* the return address of its call of the sub-mesh submission */
#define HL_LOOP_KEY64_RETURN 0x80087590ULL   /* ... of its two calls of the shared preparation */
#define HL_LOOP_KEY0_RETURN 0x800895bcULL

typedef void (*HostFn)(Ctx *c, void *user);
uint64_t nvn_native_proc(const char *name);                                  /* nvn_native.c */
uint64_t hos_hostcall_target(uint64_t addr, HostFn *fn, void **user);         /* hos_hostcall.c */

static int check_on = -1;   /* HL_GFX_CHECK, read at the first NVN procedure */
static int check_wanted(void) {
    if (check_on < 0) {
        const char *e = getenv("HL_GFX_CHECK");
        check_on = e ? atoi(e) : 0;
        if (check_on < 0) check_on = 0;
    }
    return check_on;
}

/* ---- the check's tap on the NVN procedures ---- */
/* proc: the native numbering; group: the kind of procedure, for the preparation's histogram (HL_GFX_PREP_GROUP_*) */
typedef struct { HostFn fn; void *user; uint32_t op; uint32_t proc; uint32_t group; } Tap;
static _Thread_local int recording;         /* a pass runs on this thread and is being recorded */
static _Thread_local int state_recording;   /* the render-state function's translated body runs here, compared */
static _Thread_local int draw_recording;    /* the sub-mesh submission's translated body runs here, compared (by segment) */
static _Thread_local int prep_recording;    /* the shared preparation's translated body runs here, compared */
static _Thread_local int var_recording;     /* the key-0xa0 function's translated body runs here on one entry, recorded */
static int var_check_wanted(void) {         /* HL_GFX_VAR_CHECK: the taps are needed too */
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("HL_GFX_VAR_CHECK");
        on = e && atoi(e) != 0;
    }
    return on;
}
static volatile LONG taps;
enum { TAP_NAMES = 2048 };
static const char *tap_names[TAP_NAMES];

/* The recording job's item map (gate 80084de0 below): while an item's translated body runs on this thread, the NVN calls
   it makes are counted, and the gates its draws reach file the item's route by the return address they are called with. */
enum {
    LOOP_ROUTE_SUBMIT = 1,    /* the sub-mesh submission called from the job's own loop (return 0x80085ea4) */
    LOOP_ROUTE_KEY64 = 2,     /* the preparation called from the job's second loop (return 0x80087590) */
    LOOP_ROUTE_KEY0 = 4,      /* the preparation called from the job's third loop (return 0x800895bc) */
    LOOP_ROUTE_VARIANT = 8,   /* the submission, the preparation or the render-state function called from elsewhere */
    LOOP_ROUTES = 16
};
static _Thread_local int t_item;           /* an item's translated body runs on this thread, observed */
static _Thread_local uint32_t t_item_route;
static _Thread_local uint64_t t_item_draws, t_item_nvn, t_item_lr;   /* t_item_lr: the first "variant" return address */
static volatile uint64_t loop_other_lr[4];   /* the first return addresses filed as "variant", for the log */
static void loop_seen(uint32_t route, uint64_t lr) {
    t_item_route |= route;
    t_item_draws++;
    if (route == LOOP_ROUTE_VARIANT && !t_item_lr) t_item_lr = lr;
    if (route == LOOP_ROUTE_VARIANT)
        for (int i = 0; i < 4; i++) {
            if (loop_other_lr[i] == lr) break;
            if (!loop_other_lr[i] && InterlockedCompareExchange64((volatile LONG64 *)&loop_other_lr[i], (LONG64)lr, 0) == 0) break;
        }
}

static volatile LONG64 loop_items[LOOP_ROUTES], loop_draws[LOOP_ROUTES], loop_nvn[LOOP_ROUTES], loop_nvn_max;
static volatile LONG64 loop_early;   /* items that return at once (the key's mask, the count or the item's flag) */
enum { LOOP_THREADS = 16 };
static volatile LONG loop_thread_ids[LOOP_THREADS];   /* the host threads items ran on (the first 16) */

static void loop_filed(uint32_t route, uint64_t draws, uint64_t nvn) {
    InterlockedIncrement64(&loop_items[route]);
    InterlockedExchangeAdd64(&loop_draws[route], (LONG64)draws);
    InterlockedExchangeAdd64(&loop_nvn[route], (LONG64)nvn);
    for (LONG64 m = loop_nvn_max; (LONG64)nvn > m; m = loop_nvn_max)
        if (InterlockedCompareExchange64(&loop_nvn_max, (LONG64)nvn, m) == m) break;
    const LONG id = (LONG)GetCurrentThreadId();
    for (int i = 0; i < LOOP_THREADS; i++) {
        if (loop_thread_ids[i] == id) break;
        if (!loop_thread_ids[i] && InterlockedCompareExchange(&loop_thread_ids[i], id, 0) == 0) break;
    }
}

/* The item's list, read at its entry (gfx-record-job.md section 12): class 0 a count below 1, class 1 entries but none
   whose kind mask (entry +8) meets the key's first word (the three inline loops skip such an entry), class 2 at least
   one that does. Filed with whether a gate saw a draw, and the item's NVN calls. */
enum { LIST_EMPTY, LIST_MASKED, LIST_LIVE, LIST_CLASSES };
static volatile LONG64 class_items[LIST_CLASSES][2], class_nvn[LIST_CLASSES][2];

/* The per-variant draws: the table at 0x84c29878 (256 entries, indexed by the key the item computes), read at the
   map's start; an item whose only off-loop draws are a variant's is filed under the variant whose entry is the nearest
   below the first such return address (the variants call the render-state and render-enable functions directly).
   Any other first return address (the drawables' own methods) goes to a small table by address. */
#define HL_VARIANT_TABLE 0x84c29878ULL
enum { VARIANTS = 256, METHOD_LRS = 32 };
static uint64_t variant_start[VARIANTS];   /* sorted */
static uint8_t variant_key[VARIANTS];
static volatile LONG64 variant_items[VARIANTS], variant_nvn[VARIANTS], variant_draws[VARIANTS];
static volatile uint64_t method_lr[METHOD_LRS];
static volatile LONG64 method_items[METHOD_LRS], method_nvn[METHOD_LRS];

static void variants_read(const hl_guest_host *h) {
    const uint64_t *t = (const uint64_t *)h->ptr(h->user, HL_VARIANT_TABLE);
    for (int k = 0; k < VARIANTS; k++) {   /* (insertion sort: once) */
        int i = k;
        while (i > 0 && variant_start[i - 1] > t[k]) {
            variant_start[i] = variant_start[i - 1];
            variant_key[i] = variant_key[i - 1];
            i--;
        }
        variant_start[i] = t[k];
        variant_key[i] = (uint8_t)k;
    }
}

static void variant_filed(uint64_t lr, uint64_t draws, uint64_t nvn) {
    int lo = -1;
    for (int i = 0; i < VARIANTS && variant_start[i] <= lr; i++) lo = i;
    /* (the template bodies are under 0x800 bytes; the two large entries 0x8008b3e0 and 0x8008d640 are 0x2260 and
       0x2200 long, the second ending where the drawables' methods begin) */
    const uint64_t big = lo >= 0 && (variant_start[lo] == 0x8008b3e0ULL || variant_start[lo] == 0x8008d640ULL);
    const int method = lo < 0 || lr - variant_start[lo] >= (big ? 0x2200u : 0x800u);
    if (!method) {
        const int k = variant_key[lo];
        InterlockedIncrement64(&variant_items[k]);
        InterlockedExchangeAdd64(&variant_draws[k], (LONG64)draws);
        InterlockedExchangeAdd64(&variant_nvn[k], (LONG64)nvn);
        return;
    }
    for (int i = 0; i < METHOD_LRS; i++) {
        if (!method_lr[i]) InterlockedCompareExchange64((volatile LONG64 *)&method_lr[i], (LONG64)lr, 0);
        if (method_lr[i] != lr) continue;
        InterlockedIncrement64(&method_items[i]);
        InterlockedExchangeAdd64(&method_nvn[i], (LONG64)nvn);
        return;
    }
}

/* The lines "items by list", "variants" and "methods" (with the loop map) */
static void variant_report(void) {
    char line[1536];
    int n = snprintf(line, sizeof line, "items by list (0 count below 1, 1 all masked, 2 live): ");
    for (int k = 0; k < LIST_CLASSES; k++)
        n += snprintf(line + n, sizeof line - n, "[%d] no draw %lld items %lld NVN, drew %lld items %lld NVN; ", k,
                      (long long)class_items[k][0], (long long)class_nvn[k][0], (long long)class_items[k][1],
                      (long long)class_nvn[k][1]);
    hos_log("hl gfx: %s", line);
    /* the variants, the most NVN calls first (a selection over the 256 counters) */
    uint8_t done[VARIANTS] = {0};
    LONG64 items = 0, nvn = 0;
    int distinct = 0;
    for (int k = 0; k < VARIANTS; k++)
        if (variant_items[k]) items += variant_items[k], nvn += variant_nvn[k], distinct++;
    n = snprintf(line, sizeof line, "variants: %d taken, %lld items, %lld NVN; key:items/draws/NVN: ", distinct,
                 (long long)items, (long long)nvn);
    for (int r = 0; r < 40 && (size_t)n < sizeof line - 48; r++) {
        int best = -1;
        for (int k = 0; k < VARIANTS; k++)
            if (!done[k] && variant_items[k] && (best < 0 || variant_nvn[k] > variant_nvn[best])) best = k;
        if (best < 0) break;
        done[best] = 1;
        n += snprintf(line + n, sizeof line - n, "%#x:%lld/%lld/%lld ", best, (long long)variant_items[best],
                      (long long)variant_draws[best], (long long)variant_nvn[best]);
    }
    hos_log("hl gfx: %s", line);
    n = snprintf(line, sizeof line, "methods (first return address: items/NVN): ");
    for (int i = 0; i < METHOD_LRS && method_lr[i] && (size_t)n < sizeof line - 48; i++)
        n += snprintf(line + n, sizeof line - n, "%#llx:%lld/%lld ", (unsigned long long)method_lr[i],
                      (long long)method_items[i], (long long)method_nvn[i]);
    hos_log("hl gfx: %s", line);
}

/* "loop map: ..." (0: nothing yet) */
static int loop_report(char *line, size_t size) {
    LONG64 items = 0;
    for (int r = 0; r < LOOP_ROUTES; r++) items += loop_items[r];
    if (!items) return 0;
    int threads = 0;
    while (threads < LOOP_THREADS && loop_thread_ids[threads]) threads++;
    int n = snprintf(line, size, "loop map: items %lld (returned at once %lld) on %d threads, max NVN calls an item %lld; "
                     "by route (1 submission loop, 2 key-64 loop, 4 key-0 loop, 8 variant): ",
                     (long long)items, (long long)loop_early, threads, (long long)loop_nvn_max);
    for (int r = 0; r < LOOP_ROUTES && n > 0 && (size_t)n < size; r++)
        if (loop_items[r])
            n += snprintf(line + n, size - n, "[%d] items %lld draws %lld NVN %lld; ", r, (long long)loop_items[r],
                          (long long)loop_draws[r], (long long)loop_nvn[r]);
    if (n > 0 && (size_t)n < size)
        snprintf(line + n, size - n, "variant returns %#llx %#llx %#llx %#llx", (unsigned long long)loop_other_lr[0],
                 (unsigned long long)loop_other_lr[1], (unsigned long long)loop_other_lr[2],
                 (unsigned long long)loop_other_lr[3]);
    return 1;
}

static void tap(Ctx *c, void *user) {
    const Tap *t = (const Tap *)user;
    if (t_item) t_item_nvn++;
    if (recording | state_recording | draw_recording | prep_recording | var_recording) {
        uint64_t x[8];
        uint32_t f[4];
        memcpy(x, c->x, sizeof x);
        for (int i = 0; i < 4; i++) memcpy(&f[i], &c->vreg[i][0], 4);
        if (recording) hl_gfx_check_call(t->op, x, f);
        if (state_recording) hl_gfx_state_check_call(t->proc, x, f);
        if (draw_recording) hl_gfx_draw_check_call(t->proc, x, f);
        if (prep_recording) hl_gfx_prep_check_call(t->proc, t->group, x, f);
        if (var_recording) hl_gfx_var_check_call(t->proc, x, f);
    }
    t->fn(c, t->user);
}

/* The kind of an NVN procedure, by its name, for the preparation's histogram. */
static uint32_t tap_group(const char *name) {
    static const char *const state[] = {"nvnBlendState", "nvnColorState", "nvnChannelMaskState", "nvnDepthStencilState",
                                        "nvnPolygonState", "nvnCommandBufferBindBlendState",
                                        "nvnCommandBufferBindColorState", "nvnCommandBufferSetBlendColor",
                                        "nvnCommandBufferBindChannelMaskState", "nvnCommandBufferBindDepthStencilState",
                                        "nvnCommandBufferSetStencil", "nvnCommandBufferBindPolygonState",
                                        "nvnCommandBufferSetPolygonOffsetClamp", "nvnCommandBufferSetDepthClamp"};
    for (size_t i = 0; i < sizeof state / sizeof state[0]; i++)
        if (!strncmp(name, state[i], strlen(state[i]))) return HL_GFX_PREP_GROUP_STATE;
    if (!strcmp(name, "nvnCommandBufferBindProgram")) return HL_GFX_PREP_GROUP_PROGRAM;
    if (!strncmp(name, "nvnCommandBufferBindUniformBuffer", 33)) return HL_GFX_PREP_GROUP_UNIFORM;
    if (!strncmp(name, "nvnCommandBufferBindStorageBuffer", 33)) return HL_GFX_PREP_GROUP_STORAGE;
    if (strstr(name, "Texture") || strstr(name, "Sampler")) return HL_GFX_PREP_GROUP_TEXTURE;
    if (!strcmp(name, "nvnCommandBufferCallCommands")) return HL_GFX_PREP_GROUP_CALL;
    return HL_GFX_PREP_GROUP_OTHER;
}

/* (nvn_native.c, for every procedure it hands the game, once a procedure) */
void hl_gfx_nvn_proc(const char *name, HostFn *fn, void **user) {
    if (!check_wanted() && !var_check_wanted()) return;
    Tap *t = (Tap *)malloc(sizeof *t);
    t->fn = *fn;
    t->user = *user;
    t->op = (uint32_t)InterlockedIncrement(&taps) - 1;
    const uint32_t n = hl_gfx_nvn_count();
    for (t->proc = 0; t->proc < n && strcmp(hl_gfx_nvn_name(t->proc), name); t->proc++) {}
    t->group = tap_group(name);
    if (t->op < TAP_NAMES) tap_names[t->op] = name;   /* (the NVN layer's table: lives as long as the process) */
    *fn = tap;
    *user = t;
}

/* ---- the counters ---- */
static int log_on;
static volatile LONG64 other_ctx;   /* passes left to the translation because the thread's context was not the one given */

/* HL_GFX_TIME=1 (a measurement, off by default: one predictable branch a gate call when off): the crossings between the
   translated code and the graphics bricks, by gate: calls (translated -> native), how many the native side answered,
   the calls back into the translation (native -> translated, a64_icall), and the time spent from the gate's entry to its
   return, apart for the calls it declined (the bare price of asking: out of lifted code, the native look at the call,
   back). One line every 10 s, per presented frame. */
enum { GT_STATE, GT_DRAW, GT_PREP, GT_N };
static int time_on = -1;
typedef struct { volatile LONG64 calls, answered, declined_ticks, answered_ticks, back, back_ticks; } GateTime;
static GateTime gate_time[GT_N];
static int gfx_time_on(void) {
    if (time_on < 0) {
        const char *e = getenv("HL_GFX_TIME");
        time_on = e && atoi(e) != 0;
        if (time_on) hos_log("hl gfx time: the gates' crossings counted and timed (HL_GFX_TIME=1)");
    }
    return time_on;
}
static int64_t tsc_now(void) {   /* (the time stamp counter: QueryPerformanceCounter's 100 ns steps are a call's length) */
    return (int64_t)__builtin_ia32_rdtsc();
}
static void gate_time_add(int g, int answered, int64_t ticks) {
    GateTime *t = &gate_time[g];
    InterlockedIncrement64(&t->calls);
    if (answered) InterlockedIncrement64(&t->answered), InterlockedAdd64(&t->answered_ticks, ticks);
    else InterlockedAdd64(&t->declined_ticks, ticks);
}
extern volatile LONG64 hos_present_count;   /* display.c */
static void gate_time_report(void) {
    static volatile LONG64 last;
    static int64_t presents0, tsc0;
    static GateTime was[GT_N];
    const int64_t now = hos_now_ns(), l = last;
    if (now - l < 10000000000LL || InterlockedCompareExchange64(&last, now, l) != l) return;
    const int64_t tsc = tsc_now();
    if (!l) { presents0 = hos_present_count, tsc0 = tsc; return; }
    const double frames = (double)(hos_present_count - presents0 > 0 ? hos_present_count - presents0 : 1),
                 ns = (double)(now - l) / (double)(tsc - tsc0 > 0 ? tsc - tsc0 : 1);   /* (ns a tick, over the interval) */
    presents0 = hos_present_count, tsc0 = tsc;
    static const char *const names[GT_N] = {"render state", "sub-mesh draw", "preparation"};
    char line[1024];
    int n = snprintf(line, sizeof line, "hl gfx time: a frame (%.0f frames):", frames);
    for (int g = 0; g < GT_N && n < (int)sizeof line; g++) {
        GateTime d, *t = &gate_time[g];
        d.calls = t->calls - was[g].calls, d.answered = t->answered - was[g].answered;
        d.declined_ticks = t->declined_ticks - was[g].declined_ticks, d.answered_ticks = t->answered_ticks - was[g].answered_ticks;
        d.back = t->back - was[g].back, d.back_ticks = t->back_ticks - was[g].back_ticks;
        was[g].calls = t->calls, was[g].answered = t->answered, was[g].declined_ticks = t->declined_ticks;
        was[g].answered_ticks = t->answered_ticks, was[g].back = t->back, was[g].back_ticks = t->back_ticks;
        const LONG64 dec = d.calls - d.answered;
        n += snprintf(line + n, sizeof line - (size_t)n, " %s: %.0f calls in, %.0f answered (%.3f ms, %.0f ns each), %.0f declined (%.0f ns each), %.0f calls back (%.3f ms);",
                      names[g], d.calls / frames, d.answered / frames, d.answered_ticks * ns / 1e6 / frames,
                      d.answered ? d.answered_ticks * ns / (double)d.answered : 0.0, dec / frames, dec ? d.declined_ticks * ns / (double)dec : 0.0,
                      d.back / frames, d.back_ticks * ns / 1e6 / frames);
    }
    hos_log("%s", line);
}

static void hl_report(void) {
    static volatile LONG64 last;
    const int64_t now = hos_now_ns();
    const int64_t was = last;
    if (now - was < 10000000000LL || InterlockedCompareExchange64(&last, now, was) != was) return;
    char line[1024];
    if (hl_gfx_report(line, sizeof line, tap_names, TAP_NAMES))
        hos_log("hl gfx: %s; passes on another context %lld", line, (long long)other_ctx);
    if (hl_gfx_state_report(line, sizeof line)) hos_log("hl gfx: %s", line);
    if (hl_gfx_draw_report(line, sizeof line)) hos_log("hl gfx: %s", line);
    if (hl_gfx_prep_report(line, sizeof line)) hos_log("hl gfx: %s", line);
    if (hl_gfx_var_report(line, sizeof line)) hos_log("hl gfx: %s", line);
    char shapes[2048];
    if (hl_gfx_prep_shapes(shapes, sizeof shapes)) hos_log("hl gfx: %s", shapes);
    if (loop_report(shapes, sizeof shapes)) {
        hos_log("hl gfx: %s", shapes);
        variant_report();
    }
}

/* ---- the pass ---- */
static int native_on;
static A64Fn translated;

static void hl_gfx_pass(Ctx *A64_NOESC c) {
    const uint64_t listener = c->x[0], args = c->x[1];
    const int label = check_on ? hl_gfx_check_next_native() : native_on;
    int native = check_on == 2 ? 0 : label;
    if (native && hos_ctx() != c) {   /* (the guest calls of the native pass run on hos_ctx: never seen, kept as a guard) */
        InterlockedIncrement64(&other_ctx);
        native = 0;
    }
    if (check_on) {
        hl_gfx_check_begin(args);
        recording = 1;
    }
    if (native) c->x[0] = hl_gfx_record_pass(listener, args);
    else translated(c);
    if (check_on) {
        recording = 0;
        hl_gfx_check_end(label);
    }
    if (log_on) hl_report();
}

/* 1 when the pass's listener is answered here (asked once, at its first lookup) */
static int hl_gfx_on(void) {
    const char *e = getenv("HL_GFX");
    const int mode = e && e[0] ? atoi(e) : HL_GFX_DEFAULT;
    const int check = check_wanted();
    if (mode == HL_GFX_OFF && !check) {
        hos_log("hl gfx: off (HL_GFX=0)");
        return 0;
    }
    if (hl_gfx_pass_address() != HL_RECORD_PASS || !(translated = hl_translated(HL_RECORD_PASS))) {
        hos_log("hl gfx: off: the bridge answers %#llx, not the recording pass", (unsigned long long)hl_gfx_pass_address());
        return 0;
    }
    native_on = hl_gfx_start(hl_host(), mode, check);
    log_on |= check || getenv("HL_GFX_LOG") != NULL;
    hos_log("hl gfx: %s (NVN procedures tapped: %ld)",
            !check ? "the native recording pass"
            : check == 2 ? "the check's control: every pass translated, filed under a path at random"
            : native_on ? "the check: each pass on the native or the translated path at random, recorded"
                        : "the check: the translated recording pass, recorded",
            (long)taps);
    return 1;
}

HL_ANSWER(HL_RECORD_PASS, hl_gfx_pass, hl_gfx_on, "the graphics recording pass's listener")

/* ---- the render-state function ---- */

/* The NVN entry points native code calls (hl_gfx_host.nvn), in the native numbering: the host function and its data
   behind the host call the game's own global holds. */
enum { STATE_PROCS = 64 };
static HostFn nvn_fn[STATE_PROCS];
static void *nvn_user[STATE_PROCS];
static _Thread_local Ctx *t_ctx;   /* the guest context of the call the gate is answering on this thread */

static uint64_t nvn_call(void *user, uint32_t proc, const uint64_t *x, uint32_t nx, const uint32_t *f, uint32_t nf) {
    (void)user;
    Ctx *c = t_ctx;
    for (uint32_t i = 0; i < nx; i++) c->x[i] = x[i];
    for (uint32_t i = 0; i < nf; i++) {   /* (as the lifted code loads a single: the rest of the register cleared) */
        c->vreg[i][0] = f[i];
        c->vreg[i][1] = 0;
    }
    nvn_fn[proc](c, nvn_user[proc]);
    return c->x[0];
}
static void gfx_written(void *user, uint64_t guest, uint64_t n) {
    (void)user, (void)guest, (void)n;   /* (unused when the build tracks no writes) */
    HOS_WRITTEN(GP(guest), (size_t)n);
}
static void gfx_log(void *user, const char *line) {
    (void)user;
    hos_log("hl gfx: %s", line);
}
static const hl_gfx_host gfx_host = {NULL, nvn_call, gfx_written, gfx_log};

/* 1 when every procedure resolves to a host call and the game's global holds that same host call (asked once: the
   render state and the submission share the table) */
static int nvn_resolve_all(void);
static int nvn_resolve(void) {
    static volatile LONG done;   /* 0: not asked, 1: no, 2: yes */
    if (!done) InterlockedCompareExchange(&done, nvn_resolve_all() ? 2 : 1, 0);
    return done == 2;
}
static int nvn_resolve_all(void) {
    const uint32_t n = hl_gfx_nvn_count();
    if (n > STATE_PROCS) {
        hos_log("hl gfx: native producers off: %u NVN procedures, room for %d", n, STATE_PROCS);
        return 0;
    }
    for (uint32_t i = 0; i < n; i++) {
        const char *name = hl_gfx_nvn_name(i);
        const uint64_t a = nvn_native_proc(name);
        uint64_t game;
        memcpy(&game, GP(hl_gfx_nvn_global(i)), 8);
        if (!hos_hostcall_target(a, &nvn_fn[i], &nvn_user[i]) || game != a) {
            hos_log("hl gfx: native producers off: %s is %#llx for the game (its global %#llx), %#llx in the NVN layer", name,
                    (unsigned long long)game, (unsigned long long)hl_gfx_nvn_global(i), (unsigned long long)a);
            return 0;
        }
    }
    return 1;
}

static int state_check;          /* HL_GFX_CHECK set: compare each call, the translation answering */
static A64Fn state_translated;
static volatile LONG state_ready;   /* 0: not asked yet, 1: off, 2: on, 3: being asked */
static _Thread_local int t_state_inside;   /* the gate's own call of the translated body runs on this thread */

/* Asked at the gate's first call (the game's NVN procedures are resolved by then): 1 when the gate answers */
static int state_init(void) {
    for (;;) {
        const LONG s = state_ready;
        if (s == 1 || s == 2) return s == 2;
        if (s == 0 && InterlockedCompareExchange(&state_ready, 3, 0) == 0) break;
        Sleep(0);
    }
    gfx_time_on();
    const char *e = getenv("HL_GFX_STATE");
    const int mode = e && e[0] ? atoi(e) : HL_GFX_STATE_DEFAULT;
    state_check = check_wanted() != 0;
    int on = 0;
    if (mode != 1 && !state_check) {
        hos_log("hl gfx: render state: the translated function (HL_GFX_STATE=%d)", mode);
    } else if (!hl_supported_build()) {
        hos_log("hl gfx: render state off: not the supported build");
    } else if (hl_gfx_state_address() != HL_STATE_FN || !(state_translated = hl_translated(HL_STATE_FN))) {
        hos_log("hl gfx: render state off: the bridge answers %#llx", (unsigned long long)hl_gfx_state_address());
    } else if (nvn_resolve() && hl_gfx_state_start(hl_host(), &gfx_host, state_check)) {
        on = 1;
        log_on |= state_check || getenv("HL_GFX_LOG") != NULL;
        hos_log("hl gfx: render state: %s",
                state_check ? "the check: each call compared with the translated function, which answers"
                            : "the native function (HL_GFX_STATE=1)");
    }
    state_ready = on ? 2 : 1;
    return on;
}

/* (switched on from the start so that the first call asks state_init; off for good once it says so) */
HL_GATE(80077d60, "31664901fd8bc782") {
    if (t_state_inside) return 0;
    if (t_item) loop_seen(LOOP_ROUTE_VARIANT, c->x[30]);
    if (state_ready != 2 && !state_init()) {
        hl_gate_on_80077d60 = 0;
        return 0;
    }
    const int64_t t0 = time_on > 0 ? tsc_now() : 0;
    Ctx *before = t_ctx;
    t_ctx = c;
    int answered;
    if (state_check) {
        answered = hl_gfx_state_check_begin(c->x[0], c->x[1], c->x[31]);
        if (answered) {
            t_state_inside = 1;
            state_recording = 1;
            state_translated(c);
            state_recording = 0;
            t_state_inside = 0;
            hl_gfx_state_check_end();
        }
    } else {
        answered = hl_gfx_state_apply(c->x[0], c->x[1], c->x[31]);
    }
    t_ctx = before;
    if (log_on) hl_report();
    if (t0) gate_time_add(GT_STATE, answered, tsc_now() - t0), gate_time_report();
    return answered;
}
__attribute__((constructor)) static void hl_gfx_state_gate(void) { hl_gate_on_80077d60 = 1; }

/* ---- the sub-mesh submission ---- */

#define HL_GFX_DRAW_DEFAULT 1   /* (on by default, as in the fork; =0 off) */ /* the fork: (the check and the validation clean in the three scenes on both PCs: gfx-record-job.md section 9) */
#define HL_DRAW_FN 0x80078fc0ULL      /* (hl_gfx_draw_address: draw_init checks the two agree) */
#define HL_DRAW_PREPARE 0x8008fc00ULL

static int draw_check;
static A64Fn draw_translated, prepare_translated;
static volatile LONG draw_ready;   /* as state_ready */
static _Thread_local int t_draw_inside;   /* the gate's own call of the translated body runs on this thread */
static _Thread_local int t_draw_armed;    /* that body's call of the preparation is the check's checkpoint */
extern volatile int hl_gate_on_8008fc00;  /* (the checkpoint's gate, below) */

static int draw_init(void) {
    for (;;) {
        const LONG s = draw_ready;
        if (s == 1 || s == 2) return s == 2;
        if (s == 0 && InterlockedCompareExchange(&draw_ready, 3, 0) == 0) break;
        Sleep(0);
    }
    gfx_time_on();
    const char *e = getenv("HL_GFX_DRAW");
    const int mode = e && e[0] ? atoi(e) : HL_GFX_DRAW_DEFAULT;
    draw_check = check_wanted() != 0;
    int on = 0;
    if (mode != 1 && !draw_check) {
        hos_log("hl gfx: draw: the translated sub-mesh submission (HL_GFX_DRAW=%d)", mode);
    } else if (!hl_supported_build()) {
        hos_log("hl gfx: draw off: not the supported build");
    } else if (hl_gfx_draw_address() != HL_DRAW_FN || hl_gfx_draw_prepare_address() != HL_DRAW_PREPARE ||
               !(draw_translated = hl_translated(HL_DRAW_FN)) || !(prepare_translated = hl_translated(HL_DRAW_PREPARE))) {
        hos_log("hl gfx: draw off: the bridge answers %#llx", (unsigned long long)hl_gfx_draw_address());
    } else if (nvn_resolve() && hl_gfx_draw_start(hl_host(), &gfx_host, draw_check)) {
        on = 1;
        log_on |= draw_check || getenv("HL_GFX_LOG") != NULL;
        if (draw_check) hl_gate_on_8008fc00 = 1;   /* (the checkpoint; the gate is also the preparation's own) */
        hos_log("hl gfx: draw: %s",
                draw_check ? "the check: each direct call compared in two segments around its preparation, the "
                             "translated function answering"
                           : "the native sub-mesh submission on its direct paths (HL_GFX_DRAW=1)");
    }
    draw_ready = on ? 2 : 1;
    return on;
}

/* The function itself (x0 drawable, x1 block, w2 sub-mesh, w3 mask; returns nothing). Native: the prologue, the
   preparation called on this context below the function's own frame (where the translation calls it, with the link
   register its call leaves), then the submission. */
HL_GATE(80078fc0, "329b54a385248a34") {
    if (t_draw_inside) return 0;
    if (t_item) loop_seen(c->x[30] == HL_LOOP_SUBMIT_RETURN ? LOOP_ROUTE_SUBMIT : LOOP_ROUTE_VARIANT, c->x[30]);
    if (draw_ready != 2 && !draw_init()) {
        hl_gate_on_80078fc0 = 0;
        return 0;
    }
    const int64_t t0 = time_on > 0 ? tsc_now() : 0;
    Ctx *before = t_ctx;
    t_ctx = c;
    const uint64_t drawable = c->x[0], block = c->x[1];
    const uint32_t sub_mesh = (uint32_t)c->x[2], mask = (uint32_t)c->x[3];
    int answered;
    if (draw_check) {
        answered = hl_gfx_draw_check_begin(drawable, block, sub_mesh, mask);
        if (answered) {
            t_draw_inside = 1;
            t_draw_armed = 1;
            draw_recording = 1;
            draw_translated(c);
            draw_recording = 0;
            t_draw_armed = 0;
            t_draw_inside = 0;
            hl_gfx_draw_check_end();
        }
    } else {
        answered = hl_gfx_draw_begin(drawable, block, sub_mesh, mask);
        if (answered == 2) {
            const uint64_t sp = c->x[31], fp = c->x[29], lr = c->x[30];
            c->x[31] = sp - hl_gfx_draw_frame();
            c->x[0] = block;
            c->x[1] = drawable;
            c->x[2] = 0;
            c->x[30] = hl_gfx_draw_prepare_return();
            const int64_t b0 = t0 ? tsc_now() : 0;
            a64_icall(c, HL_DRAW_PREPARE);
            if (t0) InterlockedIncrement64(&gate_time[GT_DRAW].back), InterlockedAdd64(&gate_time[GT_DRAW].back_ticks, tsc_now() - b0);
            const uint64_t prepared = c->x[0];
            c->x[31] = sp;
            hl_gfx_draw_end(prepared);
            c->x[29] = fp;   /* (as the function's epilogue leaves them) */
            c->x[30] = lr;
        }
    }
    t_ctx = before;
    if (log_on) hl_report();
    if (t0) gate_time_add(GT_DRAW, answered != 0, tsc_now() - t0), gate_time_report();
    return answered != 0;
}
__attribute__((constructor)) static void hl_gfx_draw_gate(void) { hl_gate_on_80078fc0 = 1; }

/* ---- the shared preparation ---- */

#define HL_GFX_PREP_DEFAULT 1   /* (on by default, as in the fork; =0 off) */ /* the fork: (the check at 0 mismatches in Castle Town on the main PC and in seven zones of the save library on the PC 3050, shots right: gfx-record-job.md section 10) */
#define HL_GFX_PREP_STATE_DEFAULT 1   /* (on by default, as in the fork; =0 off) */ /* the fork: (the state step: the check at 0 mismatches in seven zones on the PC 3050, shots right; not run on the main PC) */

static int prep_check;
static volatile LONG prep_ready;          /* as state_ready */
static _Thread_local int t_prep_inside;   /* the gate's own call of the translated body runs on this thread */

static int prep_init(void) {
    for (;;) {
        const LONG s = prep_ready;
        if (s == 1 || s == 2) return s == 2;
        if (s == 0 && InterlockedCompareExchange(&prep_ready, 3, 0) == 0) break;
        Sleep(0);
    }
    gfx_time_on();
    const char *e = getenv("HL_GFX_PREP");
    const int mode = e && e[0] ? atoi(e) : HL_GFX_PREP_DEFAULT;
    prep_check = check_wanted() != 0;
    const char *es = getenv("HL_GFX_PREP_STATE");
    const int state = es && es[0] ? atoi(es) : HL_GFX_PREP_STATE_DEFAULT;
    int on = 0;
    if (mode != 1 && !prep_check) {
        hos_log("hl gfx: preparation: the translated function (HL_GFX_PREP=%d)", mode);
    } else if (!hl_supported_build()) {
        hos_log("hl gfx: preparation off: not the supported build");
    } else if (hl_gfx_prep_address() != HL_DRAW_PREPARE || !(prepare_translated = hl_translated(HL_DRAW_PREPARE))) {
        hos_log("hl gfx: preparation off: the bridge answers %#llx", (unsigned long long)hl_gfx_prep_address());
    } else if (nvn_resolve() && hl_gfx_prep_start(hl_host(), &gfx_host, prep_check, state)) {
        on = 1;
        log_on |= prep_check || getenv("HL_GFX_LOG") != NULL;
        hos_log("hl gfx: preparation: %s",
                prep_check ? "the check: each call's route counted, the calls on the cached path compared with the "
                             "translated function, which answers"
                           : "native on its cached and program-bound paths (HL_GFX_PREP=1)");
        hos_log("hl gfx: preparation: the state step %s (HL_GFX_PREP_STATE=%d)", state ? "native" : "left to the translation", state);
    }
    prep_ready = on ? 2 : 1;
    return on;
}

/* The translated preparation, compared call by call (and its shape filed in the histogram). */
static void prep_checked(Ctx *c) {
    /* (the native side asks NVN queries on this context: the translated body runs with the registers it was given) */
    uint64_t keep[32];
    memcpy(keep, c->x, sizeof keep);
    hl_gfx_prep_check_begin(c->x[0], c->x[1], c->x[31]);
    memcpy(c->x, keep, sizeof keep);
    t_prep_inside = 1;
    prep_recording = 1;
    prepare_translated(c);
    prep_recording = 0;
    t_prep_inside = 0;
    hl_gfx_prep_check_end(c->x[0]);
}

/* The preparation (x0 block, x1 drawable, w2 not read; returns w0, bit 0: go on). Answered natively on its cached path
   (HL_GFX_PREP=1), compared with the check; and the sub-mesh submission check's checkpoint, for the call the checked
   body makes on this thread. */
static int prep_gate(Ctx *c) {
    if (t_prep_inside) return 0;
    if (t_item && !t_draw_armed && c->x[30] != hl_gfx_draw_prepare_return())
        loop_seen(c->x[30] == HL_LOOP_KEY64_RETURN ? LOOP_ROUTE_KEY64
                  : c->x[30] == HL_LOOP_KEY0_RETURN ? LOOP_ROUTE_KEY0 : LOOP_ROUTE_VARIANT, c->x[30]);
    if (!t_draw_armed) {
        if (prep_ready != 2 && !prep_init()) {
            if (!check_wanted()) hl_gate_on_8008fc00 = 0;   /* (the checkpoint needs the gate while checking) */
            return 0;
        }
        const int64_t t0 = time_on > 0 ? tsc_now() : 0;
        Ctx *before = t_ctx;
        t_ctx = c;
        int answered = 1;
        if (prep_check) {
            prep_checked(c);
        } else {
            uint64_t result;
            answered = hl_gfx_prep_apply(c->x[0], c->x[1], c->x[31], &result);
            if (answered) c->x[0] = result;
        }
        t_ctx = before;
        if (log_on) hl_report();
        if (t0) gate_time_add(GT_PREP, answered, tsc_now() - t0), gate_time_report();
        return answered;
    }
    t_draw_armed = 0;
    draw_recording = 0;
    hl_gfx_draw_check_segment();
    prepare_translated(c);
    /* (the native submission asks NVN queries on this context: the translated body goes on with these registers) */
    uint64_t keep[32];
    memcpy(keep, c->x, sizeof keep);
    hl_gfx_draw_check_prepared(c->x[0]);
    memcpy(c->x, keep, sizeof keep);
    draw_recording = 1;
    return 1;
}

/* (the key-0xa0 function's check records each call it makes of the preparation as a segment of its NVN calls) */
HL_GATE(8008fc00, "4df7e218373859d0") {
    const int segment = var_recording && c->x[30] == HL_VAR_PREPARE_RETURN;
    const uint64_t block = c->x[0];
    if (segment) hl_gfx_var_prep_begin(block);
    const int answered = prep_gate(c);
    if (segment) hl_gfx_var_prep_end(c->x[0], answered);
    return answered;
}
__attribute__((constructor)) static void hl_gfx_prep_gate(void) { hl_gate_on_8008fc00 = 1; }

/* ---- the key-0xa0 draw function (guest 0x8008d640) ----
   The recording job's item body calls it through the table at 0x84c29878 (no lifted code names it, so it is answered by
   pointer): the job's own draw loop specialized for one key, 40% of the job's NVN calls (gfx-record-job.md section 13).
   HL_GFX_VAR=1 answers the entries of its plain path natively and runs every other entry through the translation (the
   function keeps nothing between entries but the block, so the translated body is called on a sublist of one entry);
   HL_GFX_VAR=2 only classifies the entries (the line "key loop" says how many take each way) and runs them all
   translated; 0 (the default) leaves the function to the translation. */
#define HL_VAR_FN 0x8008d640ULL
#define HL_VAR_PREPARE 0x8008fc00ULL
static int var_mode;
static A64Fn var_translated;
static volatile LONG var_ready;   /* as state_ready */

static int var_asked(void) {
    const char *e = getenv("HL_GFX_VAR");
    var_mode = e && e[0] ? atoi(e) : 0;
    if (var_check_wanted() && var_mode != 2) var_mode = 1;   /* (the check compares native entries: native on) */
    return var_mode == 1 || var_mode == 2;
}

static int var_init(void) {
    for (;;) {
        const LONG s = var_ready;
        if (s == 1 || s == 2) return s == 2;
        if (s == 0 && InterlockedCompareExchange(&var_ready, 3, 0) == 0) break;
        Sleep(0);
    }
    int on = 0;
    if (!hl_supported_build()) {
        hos_log("hl gfx: key loop off: not the supported build");
    } else if (hl_gfx_var_address() != HL_VAR_FN || !(var_translated = hl_translated(HL_VAR_FN))) {
        hos_log("hl gfx: key loop off: the bridge answers %#llx", (unsigned long long)hl_gfx_var_address());
    } else if (nvn_resolve() && hl_gfx_var_start(hl_host(), &gfx_host, var_mode == 1, var_check_wanted())) {
        on = 1;
        log_on = 1;
        hos_log("hl gfx: key loop: %s (HL_GFX_VAR=%d)",
                var_check_wanted() ? "the check: each plain entry run translated and recorded, then native from the same state"
                : var_mode == 1    ? "the plain entries native, the others translated"
                                   : "every entry translated, classified",
                var_mode);
    }
    var_ready = on ? 2 : 1;
    return on;
}

typedef struct { Ctx *c; uint64_t block, mask, lr; uint32_t flags; } VarCall;

static void var_translate(void *user, uint64_t list, uint32_t count) {
    const VarCall *v = (const VarCall *)user;
    Ctx *c = v->c;
    c->x[0] = list;
    c->x[1] = count;
    c->x[2] = v->block;
    c->x[3] = v->mask;
    c->x[5] = v->flags;
    c->x[30] = v->lr;
    var_translated(c);
}

/* The shared preparation as the function calls it: below its frame, the link register its call leaves. */
static void var_record(void *user, int on) {
    (void)user;
    var_recording = on;
}

static uint64_t var_prepare(void *user, uint64_t block, uint64_t drawable, uint64_t sp) {
    const VarCall *v = (const VarCall *)user;
    Ctx *c = v->c;
    uint64_t result;
    if (prep_ready == 2 && !prep_check && hl_gfx_prep_apply(block, drawable, sp, &result)) return result;
    const uint64_t sp0 = c->x[31], fp = c->x[29], lr = c->x[30];
    c->x[31] = sp;
    c->x[29] = sp + 336;
    c->x[0] = block;
    c->x[1] = drawable;
    c->x[2] = 0;
    c->x[30] = hl_gfx_var_prepare_return();
    a64_icall(c, HL_VAR_PREPARE);
    result = c->x[0];
    c->x[31] = sp0;
    c->x[29] = fp;
    c->x[30] = lr;
    return result;
}

/* x0 the list of draw entries, w1 their count, x2 the block, x3 the key's first word, w5 the item's flags. */
static void hl_gfx_var(Ctx *A64_NOESC c) {
    if (var_ready != 2 && !var_init()) {
        var_translated(c);
        return;
    }
    if ((int32_t)c->x[1] < 1) return;   /* (the function returns at once) */
    if (prep_ready != 2) prep_init();
    VarCall v = {c, c->x[2], c->x[3], c->x[30], (uint32_t)c->x[5]};
    hl_gfx_var_call k = {c->x[0], (uint32_t)c->x[1], c->x[2], c->x[3], (uint32_t)c->x[5], c->x[31]};
    hl_gfx_var_host h = {&v, var_translate, var_prepare, var_record};
    Ctx *before = t_ctx;
    t_ctx = c;
    hl_gfx_var_run(&k, &h);
    t_ctx = before;
    if (log_on) hl_report();
}
HL_ANSWER(HL_VAR_FN, hl_gfx_var, var_asked, "the key-0xa0 draw function")

/* ---- the recording job's item (guest 0x80084de0): its map ----
   One call records one NVN command buffer for one key (gfx-record-job.md section 11). Nothing of it is native yet: with
   HL_GFX_CHECK=1 the gate observes each item, its translated body answering, and files it by the route its draws took
   (the return addresses the submission and preparation gates are called with) with its NVN calls and draws counted:
   the line "loop map: ..." every 10 s. Without the check the gate turns itself off at the first call. */
static volatile LONG loop_ready;   /* as state_ready */
static A64Fn loop_translated;

static int loop_init(void) {
    for (;;) {
        const LONG s = loop_ready;
        if (s == 1 || s == 2) return s == 2;
        if (s == 0 && InterlockedCompareExchange(&loop_ready, 3, 0) == 0) break;
        Sleep(0);
    }
    int on = 0;
    if (!check_wanted()) {
        /* (no line: the item is the translation's, as before) */
    } else if (!hl_supported_build()) {
        hos_log("hl gfx: loop map off: not the supported build");
    } else if (!(loop_translated = hl_translated(HL_LOOP_FN))) {
        hos_log("hl gfx: loop map off: no translated item body");
    } else {
        on = 1;
        log_on = 1;
        variants_read(hl_host());
        hos_log("hl gfx: loop map: each item of the recording job observed, the translated item answering");
    }
    loop_ready = on ? 2 : 1;
    return on;
}

/* x0 the cache of recordings, x1 the pass, x2 the command-memory manager, x3 the 0x70-byte key, x4..x6 the item. */
HL_GATE(80084de0, "3ba04b5feac748d3") {
    if (t_item) return 0;
    if (loop_ready != 2 && !loop_init()) {
        hl_gate_on_80084de0 = 0;
        return 0;
    }
    const hl_guest_host *h = hl_host();
    const uint8_t *cache = (const uint8_t *)h->ptr(h->user, c->x[0]);
    const uint32_t key0 = *(const uint32_t *)h->ptr(h->user, c->x[3]);
    const uint16_t flag = *(const uint16_t *)h->ptr(h->user, c->x[4] + 0xc);
    const int early = !(*(const uint32_t *)(cache + 0x14) & key0) || *(const int32_t *)(cache + 0x2c) <= 0 || (flag & 1);
    if (early) InterlockedIncrement64(&loop_early);
    /* the item's list: (start, count) at x5 over the cache's array of entry pointers (+0x20) */
    const int32_t *pair = (const int32_t *)h->ptr(h->user, c->x[5]);
    int list = LIST_EMPTY;
    if (pair[1] >= 1) {
        const uint64_t *entries = (const uint64_t *)h->ptr(h->user, *(const uint64_t *)(cache + 0x20) + 8 * (int64_t)pair[0]);
        list = LIST_MASKED;
        for (int32_t i = 0; i < pair[1] && list == LIST_MASKED; i++)
            if (*(const uint32_t *)h->ptr(h->user, entries[i] + 8) & key0) list = LIST_LIVE;
    }
    t_item = 1;
    t_item_route = 0;
    t_item_draws = 0;
    t_item_nvn = 0;
    t_item_lr = 0;
    loop_translated(c);
    t_item = 0;
    loop_filed(t_item_route, t_item_draws, t_item_nvn);
    if (!early) {
        InterlockedIncrement64(&class_items[list][t_item_route != 0]);
        InterlockedExchangeAdd64(&class_nvn[list][t_item_route != 0], (LONG64)t_item_nvn);
    }
    if (t_item_lr) variant_filed(t_item_lr, t_item_draws, t_item_nvn);
    hl_report();
    return 1;
}
__attribute__((constructor)) static void hl_gfx_loop_gate(void) { hl_gate_on_80084de0 = 1; }
