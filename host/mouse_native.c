/* (relink) Native mouse look (the Insert menu's "Mouse camera", off by default; HOS_MOUSE_NATIVE=1 or 0 overrides it):
   the captured mouse turns the camera itself, its yaw and pitch in degrees, with no decay, no response curve and no top
   speed, in place of the mouse as the right stick (nn_hid.c's mouse_stick).

   Where the angles are (found by snapshotting the heap while the right stick turned the camera, in Kokiri Forest):
   the player camera's controller holds its pitch limits as {-85, 89, -85, 89} (four floats, degrees), its distance
   after them, its pitch at +0x14 from them and its yaw at +0x2c. Yaw falls as the view turns right and pitch rises as
   it tilts down. The output beside it (pitch and yaw side by side) is recomputed by the game every frame: writing it
   does nothing.

   The limits are looked for in the heap on a thread of their own (about a second over the heap's 2 GB in use, which
   took the game's thread that long in the test build), when the mouse first moves and whenever they are no longer
   where they were (a new scene); at most every 3 s, then less often while none is found (up to 30 s). Until one is
   found the motion is given back and the mouse turns the camera as the stick does.

   The sensitivity is the mouse's (the menu's Mouse sensitivity, - and =): 0.1 degrees a count at 1. HOS_MOUSE_INVERT_Y=1
   or the menu's Invert mouse vertical flips up and down.

   The game uses these angles only while its camera is in its manual state, which a firm push of the right stick starts and
   every arrival, door or dialogue ends: see "the game's manual camera" below, which serves the mouse as the stick too. */
#include "hos.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

int hos_mouse_take(int32_t *dx, int32_t *dy);                                   /* display.c */
void hos_mouse_give_back(int32_t dx, int32_t dy);                               /* display.c */
extern double hos_mouse_sens;

enum { MAX_T = 32, PITCH_AT = 0x14, YAW_AT = 0x2c, PAGE = 4096 };
static const float limits_sig[4] = {-85.0f, 89.0f, -85.0f, 89.0f};
static uint64_t ctl[MAX_T];                 /* the controllers in use: where their limits are */
static size_t nctl;
static uint64_t found[MAX_T];               /* the search's answer, taken up by the next motion */
static size_t nfound;
static volatile LONG searching, answered, busy;
static int64_t next_search, wait_ns = 3000000000LL;
static long long applied, given_back;

/* Which heap pages the process has touched: Windows says which are in its working set; elsewhere every page is read
   (an untouched page of an anonymous mapping reads as the shared zero page there, and zero pages are left out) */
#ifdef _WIN32
typedef struct { void *va; ULONG_PTR attr; } WsInfo;
typedef BOOL(WINAPI *QwsFn)(HANDLE, void *, DWORD);
#endif

/* a byte a heap page: 1 if the process has it (the caller frees it) */
static uint8_t *pages_in_use(uint64_t pages) {
    uint8_t *use = (uint8_t *)calloc((size_t)pages, 1);
    if (!use) return NULL;
#ifdef _WIN32
    QwsFn qws = (QwsFn)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32QueryWorkingSetEx");
    if (qws) {
        enum { BATCH = 65536 };
        WsInfo *info = (WsInfo *)malloc(sizeof(WsInfo) * BATCH);
        for (uint64_t i = 0; info && i < pages; i += BATCH) {
            const uint64_t n = pages - i < BATCH ? pages - i : BATCH;
            for (uint64_t k = 0; k < n; k++) info[k].va = GP(HOS_HEAP_BASE + (i + k) * PAGE), info[k].attr = 0;
            if (qws(GetCurrentProcess(), info, (DWORD)(n * sizeof(WsInfo))))
                for (uint64_t k = 0; k < n; k++) use[i + k] = (uint8_t)(info[k].attr & 1);
        }
        free(info);
    } else
#endif
        memset(use, 1, (size_t)pages);
    return use;
}

/* the 4-byte-aligned places in the heap's used pages that hold len bytes of pat (len <= a page), at
   most max of them; the number found */
static size_t heap_find(const void *pat, size_t len, uint64_t *out, size_t max) {
    const uint64_t pages = hos_heap_size / PAGE;
    uint8_t *use = pages_in_use(pages);
    if (!use || len < 4 || len > PAGE) { free(use); return 0; }
    uint32_t first;
    memcpy(&first, pat, 4);
    size_t n = 0;
    for (uint64_t i = 0; i < pages && n < max; i++) {
        if (!use[i]) continue;
        const uint8_t *p = (const uint8_t *)GP(HOS_HEAP_BASE + i * PAGE);
        for (size_t o = 0; o + len <= PAGE && n < max; o += 4)
            if (*(const uint32_t *)(p + o) == first && !memcmp(p + o, pat, len)) out[n++] = HOS_HEAP_BASE + i * PAGE + o;
    }
    free(use);
    return n;
}

static float getf(uint64_t a) { float f; memcpy(&f, GP(a), 4); return f; }
static void putf(uint64_t a, float f) { memcpy(GP(a), &f, 4); }

static int angles(uint64_t at) {
    const float p = getf(at + PITCH_AT), y = getf(at + YAW_AT);
    return isfinite(p) && isfinite(y) && fabsf(p) <= 90.0f && fabsf(y) <= 360.0f;
}

static int still_there(void) {
    return nctl && !memcmp(GP(ctl[0]), limits_sig, sizeof limits_sig) && angles(ctl[0]);
}

/* (HOS_MOUSE_PROBE=1, a diagnostic: every half second, the words the game changed around each controller in use, as
   offsets from its limits with their value as a float. What told a controller the game still runs from one it left behind) */
static int probe_on(void) {
    static int on = -1;
    if (on < 0) on = getenv("HOS_MOUSE_PROBE") != NULL;
    return on;
}
static void probe(int64_t now) {
    enum { WORDS = 0xC0, BEFORE = 0x100 };
    static int64_t next;
    static uint32_t snap[4][WORDS];
    static uint64_t snap_at[4];
    if (!probe_on() || now < next) return;
    next = now + 500000000;
    for (size_t i = 0; i < nctl && i < 4; i++) {
        const uint64_t page = ctl[i] & ~(uint64_t)(PAGE - 1);
        char line[1000];
        int len = snprintf(line, sizeof line, "mouse probe: %#llx pitch %.2f yaw %.2f |", (unsigned long long)ctl[i],
                           getf(ctl[i] + PITCH_AT), getf(ctl[i] + YAW_AT));
        const int same = snap_at[i] == ctl[i];
        for (int w = 0; w < WORDS; w++) {
            const uint64_t a = ctl[i] - BEFORE + (uint64_t)w * 4;
            if (a < page || a + 4 > page + PAGE) continue;   /* (the controller's own page only: the next may not be there) */
            uint32_t v;
            memcpy(&v, GP(a), 4);
            if (same && v != snap[i][w] && len < (int)sizeof line - 48) {
                float f;
                memcpy(&f, &v, 4);
                len += snprintf(line + len, sizeof line - len, " %+d:%.3f", w * 4 - BEFORE, f);
            }
            snap[i][w] = v;
        }
        snap_at[i] = ctl[i];
        hos_log("%s", line);
    }
}

static DWORD WINAPI search(void *arg) {
    (void)arg;
    const int64_t t0 = hos_now_ns();
    uint64_t at[MAX_T];
    const size_t n = heap_find(limits_sig, sizeof limits_sig, at, MAX_T);
    nfound = 0;
    for (size_t i = 0; i < n; i++)
        if (angles(at[i])) found[nfound++] = at[i];
    if (probe_on())
        for (size_t i = 0; i < n; i++)
            hos_log("mouse probe: limits at %#llx, pitch %.2f yaw %.2f%s", (unsigned long long)at[i], getf(at[i] + PITCH_AT),
                    getf(at[i] + YAW_AT), angles(at[i]) ? "" : " (not angles)");
    hos_log("mouse native: %zu camera controller(s) found (%zu places with the limits) in %.0f ms%s", nfound, n,
            (hos_now_ns() - t0) / 1e6, nfound ? "" : ": the mouse turns the camera as the stick does until one is");
    answered = 1;
    searching = 0;
    return 0;
}

/* (the probe with the mouse as the stick: the same searches and the same lines, nothing written. Whether the game itself
   moves the controller's angles when the stick turns the camera) */
void hos_mouse_probe_tick(void) {
    if (!probe_on() || InterlockedExchange(&busy, 1)) return;
    if (answered) {
        answered = 0;
        memcpy(ctl, found, sizeof ctl);
        nctl = nfound;
    }
    const int64_t now = hos_now_ns();
    if (now >= next_search && !InterlockedExchange(&searching, 1)) {
        next_search = now + 3000000000LL;
        if (!CreateThread(NULL, 0, search, NULL, 0, NULL)) searching = 0;
    }
    if (still_there()) probe(now);
    busy = 0;
}

/* ---- the game's manual camera (relink, issue 2: "no mouse look indoors", "every time I interact with something, mouse
   look just locks up") ----
   OoT's camera has two states. In the automatic one it follows Link and takes nothing from the right stick until the stick
   is pushed well off its centre; from then on, the manual one, the stick turns it down to the smallest deflection, and the
   controller's angles are what the view is made from. An arrival in a place, a door, a dialogue, mounting the horse put it
   back in the automatic state. Measured with the right stick held by a test (HOS_STICK_SWEEP, the camera's yaw read twice a
   second, Hyrule Field, the Market, Kakariko; hs2427 to hs2429):
     - from the automatic state, 15 % held for 3 s: the yaw does not move; neither does it at 20 to 30 % (the mouse's own
       20 %, and a 22 or 30 % held with native look: every capture the same); at 40 % it turns;
     - a pulse of 40 ms at 42 %, then 15 %: it turns (2.3 degrees a second), and goes on after 3 s at rest;
     - once manual, the speed is 110 degrees a second times the square of the deflection (6 %: 0.33; 20 %: 4.2; 40 %: 17.5;
       100 %: 110), with no dead zone.
   So the mouse did nothing where the camera was automatic: with native look always (it wrote the angles with the stick at
   rest: the yaw written at 30 degrees a second for 30 s, every capture the same, hs2421), and as the stick whenever it
   moved gently (60 counts a second: the yaw never moved, hs2426). It "locked up" after each thing that made the camera
   automatic again, until a brisk movement.
   Now the game's own answer is watched: while the mouse moves, the game turns the camera a little each frame when it is in
   the manual state (the stick's deflection, or with native look a small one held for that, NUDGE_PCT), and the angles show
   it. When they have not moved for QUIET_NS, a pulse of PULSE_NS at PULSE_PCT in the motion's direction starts the manual
   state (about 0.9 degree of turn, once). A game that takes no input (a cutscene, a pause) answers no pulse: after three in
   a row the next ones are 2 s apart. With no controller found (hos_mouse_native's search), the pulse is given at the start
   of each movement after a rest instead.
   Native look writes its own running angle, not the game's last one plus the motion, so what the game adds for the small
   deflection does not build up; it starts again from the game's when the motion pauses, when the controller is another
   one, or when the game's angle is not ours within a few degrees (the game moved the camera: an arrival, a lock-on).
   HOS_MOUSE_PULSE=0: no pulse; HOS_MOUSE_NATIVE_NUDGE=<percent, 8>: native look's small deflection (0: none). */
enum { PULSE_PCT = 42, NUDGE_PCT = 8 };
static const int64_t PULSE_NS = 45000000, QUIET_NS = 90000000, PULSE_EVERY_NS = 500000000, PULSE_SLOW_NS = 2000000000,
                     REST_NS = 150000000;
static float own_yaw[MAX_T], own_pitch[MAX_T];
static uint64_t own_at[MAX_T];
static int64_t own_ns[MAX_T];
/* the game's answer: the first controller's angles as last seen or written here, and when the game last changed them */
static uint64_t seen_at;
static float seen_yaw, seen_pitch;
static int64_t seen_changed;
/* the stick's motion (a direction on each axis, -1 0 1) and the pulse */
static int64_t moving_since, moved_last, pulse_last;
static volatile LONG64 pulse_until;
static volatile LONG pulse_x, pulse_y, nudge_x, nudge_y;
static volatile LONG64 nudge_until;
static int pulse_unanswered;
static long long pulses;

static int env_percent(const char *name, int otherwise) {
    const char *e = getenv(name);
    const int v = e ? atoi(e) : otherwise;
    return v < 0 ? 0 : v > 100 ? 100 : v;
}

/* the controllers in use (the search's answer taken up), a search started when there is none and the mouse moves;
   whether there is one. Holding busy */
static int controllers(int64_t now, int moving) {
    if (answered) {
        answered = 0;
        memcpy(ctl, found, sizeof ctl);
        nctl = nfound;
        /* (8 s at most between two looks while none is found, 30 s before: a player who moved the mouse on the title and the
           loading screens then had the stick's way for the first half minute of play, hs2421) */
        wait_ns = nctl ? 3000000000LL : (wait_ns * 2 < 8000000000LL ? wait_ns * 2 : 8000000000LL);
    }
    const int ok = still_there();
    /* (the probe searches every 3 s whatever the controllers in use say: how many places hold the limits, and where) */
    if ((!ok || probe_on()) && moving && now >= next_search && !InterlockedExchange(&searching, 1)) {
        next_search = now + wait_ns;
        if (!CreateThread(NULL, 0, search, NULL, 0, NULL)) searching = 0;
    }
    return ok;
}

/* the game's answer looked at: its angles are not the ones last seen or written here */
static void answer_seen(int64_t now) {
    const uint64_t at = ctl[0];
    const float y = getf(at + YAW_AT), p = getf(at + PITCH_AT);
    if (at != seen_at || y != seen_yaw || p != seen_pitch) seen_changed = now;
    seen_at = at, seen_yaw = y, seen_pitch = p;
}

/* The stick moves in direction (sx, sy), or is at rest (0, 0); known: a controller is in use and its answer was looked at.
   Whether a pulse starts now. Holding busy */
static int pulse_due(int64_t now, int sx, int sy, int known) {
    static int on = -1;
    if (on < 0) on = env_percent("HOS_MOUSE_PULSE", 1) != 0;
    int started = 0;
    if (sx || sy) {
        if (!moving_since || now - moved_last > REST_NS) moving_since = now, started = 1;
        moved_last = now;
    } else if (now - moved_last > REST_NS) moving_since = 0;
    if (!on || !(sx || sy) || now < pulse_until) return 0;
    const int64_t every = pulse_unanswered >= 3 ? PULSE_SLOW_NS : PULSE_EVERY_NS;
    if (now - pulse_last < every) return 0;
    if (known ? (now - seen_changed < QUIET_NS || now - moving_since < QUIET_NS) : !started) return 0;
    /* (the pulse before this one: answered if the game moved the angles since. With no controller there is no telling) */
    pulse_unanswered = !known || seen_changed > pulse_last ? 0 : pulse_unanswered + 1;
    pulse_last = now;
    InterlockedExchange(&pulse_x, sx);
    InterlockedExchange(&pulse_y, sy);
    InterlockedExchange64(&pulse_until, now + PULSE_NS);
    if (!(++pulses & (pulses - 1)))   /* (at each power of two) */
        hos_log("mouse: the camera took nothing from the stick: a pulse to start its manual turn (%lld so far%s)", pulses,
                known ? "" : "; no camera controller found, given at each movement's start");
    return 1;
}

static int pulse_out(int64_t now, int32_t *rx, int32_t *ry) {
    if (now >= pulse_until) return 0;
    *rx = (int32_t)(pulse_x * 32767 * PULSE_PCT / 100);
    *ry = (int32_t)(pulse_y * 32767 * PULSE_PCT / 100);
    return 1;
}

/* (nn_hid.c, the mouse as the right stick, at each read while it is captured) rx, ry: the stick the mouse makes. 1 and the
   pulse's stick in (px, py) while a pulse is on, for the larger of the two to go to the game */
int hos_mouse_stick_pulse(int32_t rx, int32_t ry, int32_t *px, int32_t *py) {
    const int64_t now = hos_now_ns();
    const int sx = rx > 1000 ? 1 : rx < -1000 ? -1 : 0, sy = ry > 1000 ? 1 : ry < -1000 ? -1 : 0;   /* (3 % of the stick) */
    if (!InterlockedExchange(&busy, 1)) {
        const int known = controllers(now, sx || sy);
        if (known) answer_seen(now);
        pulse_due(now, sx, sy, known);
        busy = 0;
    }
    return pulse_out(now, px, py);
}

/* (nn_hid.c, after hos_mouse_native answered 1) the right stick's part of native look: the pulse while one is on, else the
   small deflection while the mouse has moved in the last 40 ms; 0 with the mouse at rest */
int hos_mouse_native_nudge(int32_t *rx, int32_t *ry) {
    static int pct = -1;
    if (pct < 0) pct = env_percent("HOS_MOUSE_NATIVE_NUDGE", NUDGE_PCT);
    const int64_t now = hos_now_ns();
    if (pulse_out(now, rx, ry)) return 1;
    if (!pct || now > nudge_until) return 0;
    *rx = (int32_t)(nudge_x * 32767 * pct / 100);
    *ry = (int32_t)(nudge_y * 32767 * pct / 100);
    return 1;
}

static float wrap180(float y) {
    while (y > 180.0f) y -= 360.0f;
    while (y < -180.0f) y += 360.0f;
    return y;
}

static void turn(size_t i, int64_t now, float dpitch, float dyaw) {
    const uint64_t at = ctl[i];
    const float y = getf(at + YAW_AT), p = getf(at + PITCH_AT), lo = getf(at), hi = getf(at + 4);
    if (own_at[i] != at || now - own_ns[i] > 100000000 || fabsf(wrap180(y - own_yaw[i])) > 3.0f || fabsf(p - own_pitch[i]) > 3.0f)
        own_yaw[i] = y, own_pitch[i] = p;
    own_at[i] = at, own_ns[i] = now;
    own_yaw[i] = wrap180(own_yaw[i] + dyaw);
    own_pitch[i] += dpitch;
    if (own_pitch[i] < lo) own_pitch[i] = lo;
    if (own_pitch[i] > hi) own_pitch[i] = hi;
    putf(at + YAW_AT, own_yaw[i]), putf(at + PITCH_AT, own_pitch[i]);
}

/* With each read of the controller (nn_hid.c, the game's threads): 0 if the mouse is not captured, 1 if its motion
   turned the camera, -1 if no camera was found and the motion was given back (the stick's way takes it) */
int hos_mouse_native(void) {
    static int env_invert = -1;
    if (env_invert < 0) {
        env_invert = getenv("HOS_MOUSE_INVERT_Y") != NULL;
        hos_log("mouse native: on (the menu's Mouse camera, or HOS_MOUSE_NATIVE)");
    }
    int rx_option(int id, int otherwise);
    const int invert = env_invert || rx_option(8, 0);   /* (the menu's Invert mouse vertical) */
    int32_t dx, dy;
    if (!hos_mouse_take(&dx, &dy)) return 0;
    if (InterlockedExchange(&busy, 1)) {   /* (another thread is at it: the motion waits for the next read) */
        hos_mouse_give_back(dx, dy);
        return 1;
    }
    const int64_t now = hos_now_ns();
    if (!controllers(now, dx || dy)) {
        busy = 0;
        if (dx || dy) hos_mouse_give_back(dx, dy), given_back++;
        return -1;
    }
    probe(now);
    answer_seen(now);   /* (before this read's own writes: what the game did since the last ones) */
    /* (the stick's direction for the motion: right is +x; up is +y, the mouse moving away from the player unless inverted.
       The last motion's while the small deflection is still held, so a read with no new count is not a rest) */
    const int32_t my = invert ? dy : -dy;
    int sx = dx > 0 ? 1 : dx < 0 ? -1 : 0, sy = my > 0 ? 1 : my < 0 ? -1 : 0;
    if (dx || dy) {
        const double k = 0.1 * hos_mouse_sens;
        const float dyaw = (float)(-dx * k), dpitch = (float)((invert ? -dy : dy) * k);
        for (size_t i = 0; i < nctl; i++)
            if (!memcmp(GP(ctl[i]), limits_sig, sizeof limits_sig) && angles(ctl[i])) turn(i, now, dpitch, dyaw);
        seen_yaw = getf(ctl[0] + YAW_AT), seen_pitch = getf(ctl[0] + PITCH_AT);   /* (ours: not the game's answer) */
        InterlockedExchange(&nudge_x, sx);
        InterlockedExchange(&nudge_y, sy);
        InterlockedExchange64(&nudge_until, now + 40000000);
        if ((++applied & 2047) == 1)
            hos_log("mouse native: %lld moves applied (%lld given to the stick while no camera was found); pitch %.2f yaw %.2f",
                    applied, given_back, getf(ctl[0] + PITCH_AT), getf(ctl[0] + YAW_AT));
    } else if (now <= nudge_until) sx = (int)nudge_x, sy = (int)nudge_y;
    pulse_due(now, sx, sy, 1);
    busy = 0;
    return 1;
}
