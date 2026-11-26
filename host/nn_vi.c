/* nn::vi for MKW (TOTKPort's nn_vi.c is bound to its renderer and TOTK's frame pacing): the default display, one
   layer, and the display's vsync event, signaled at 60 Hz by a host thread. HDR and VRR are reported unsupported,
   so the game takes its SDR path. Frames are paced at the native NVN layer's window acquire (the end of this file). */
#include "hos.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include "hos_sdl.h"
#endif

void hos_system_event_init(uint64_t key, int autoclear);   /* nn_os_sync.c */
void hos_system_event_signal(uint64_t key);

static uint64_t display, layer;
static uint64_t vsync_events[8];
static volatile LONG nvsync;
static HANDLE vsync_thread;

static DWORD WINAPI vsync_main(void *p) {
    (void)p;
    const int64_t period = 1000000000LL / 60;
    int64_t next = hos_now_ns() + period;
    for (;;) {
        const int64_t now = hos_now_ns();
        if (next > now) hos_sleep_ns(next - now);
        next += period;
        if (next < hos_now_ns()) next = hos_now_ns() + period;   /* (fell behind: no burst of vsyncs) */
        for (LONG i = 0; i < nvsync; i++) hos_system_event_signal(vsync_events[i]);
    }
    return 0;
}

/* nn::vi::Initialize() */
HLE(_ZN2nn2vi10InitializeEv) {}

/* nn::vi::OpenDefaultDisplay(nn::vi::Display**) -> Result */
HLE(_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE) {
    if (!display) display = hos_sys_alloc(0x100, 16);
    G64(ARG(0)) = display;
    RET(0);
}

/* nn::vi::CreateLayer(nn::vi::Layer**, nn::vi::Display*) -> Result */
HLE(_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE) {
    if (!layer) layer = hos_sys_alloc(0x100, 16);
    G64(ARG(0)) = layer;
    RET(0);
}

/* nn::vi::GetDisplayVsyncEvent(nn::os::SystemEventType*, nn::vi::Display*) -> Result */
HLE(_ZN2nn2vi20GetDisplayVsyncEventEPNS_2os15SystemEventTypeEPNS0_7DisplayE) {
    hos_system_event_init(ARG(0), 1);
    if (nvsync < (LONG)(sizeof vsync_events / sizeof vsync_events[0])) {
        vsync_events[nvsync] = ARG(0);
        InterlockedIncrement(&nvsync);
    }
    if (!vsync_thread) vsync_thread = CreateThread(NULL, 0, vsync_main, NULL, 0, NULL);
    RET(0);
}

/* nn::vi::QueryDisplayHdrStatus(bool* on, HdrOutputFormat*, unsigned short* max nits, Display const*) -> Result */
/* (relink) HDR output. The game asks whether its display is HDR; told yes, it makes its window textures RGB10A2 and draws its own
   HDR picture in them (ST 2084 / BT.2020, its interface at the paper white), and sends the layer's metadata below. It is told yes
   when the renderer's window images are HDR10 (gpu/vk/vk_main.cpp hdr_output: the Graphics tab's HDR row, Windows' HDR on for the
   display); the peak is the display's own (DXGI). HOS_HDR=0/1 over it (tests: the game's HDR picture shown in an SDR window is
   grey and washed out), HOS_HDR_FORMAT (1: the value the game took on the first try), HOS_HDR_NITS, HOS_HDR_PAPER_WHITE */
static int hdr_env(const char *name, int dflt) {
    const char *e = getenv(name);
    return e && *e ? atoi(e) : dflt;
}
int rx_option(int id, int otherwise);
static int hdr_on(void) {
    const int env = hdr_env("HOS_HDR", -1);
    return env >= 0 ? env != 0 : rx_option(70, 0) != 0;   /* (HOS_OPT_HDR, gpu/vk/vk_menu.h) */
}
HLE(_ZN2nn2vi21QueryDisplayHdrStatusEPbPNS0_15HdrOutputFormatEPtPKNS0_7DisplayE) {
    static int told;
    const int on = hdr_on();
    const int format = hdr_env("HOS_HDR_FORMAT", 1), nits = hdr_env("HOS_HDR_NITS", rx_option(71, 1000));
    *(uint8_t *)GP(ARG(0)) = (uint8_t)on;
    if (ARG(1)) G32(ARG(1)) = on ? (uint32_t)format : 0;
    if (ARG(2)) *(uint16_t *)GP(ARG(2)) = on ? (uint16_t)(nits < 100 ? 100 : nits > 10000 ? 10000 : nits) : 0;
    if (told++ < 4) hos_log("vi: QueryDisplayHdrStatus -> %s (format %d, %d nits)", on ? "on" : "off", on ? format : 0, on ? nits : 0);
    RET(0);
}
/* nn::vi::QueryDisplayHdrPaperWhite(unsigned short*, Display const*) -> Result */
HLE(_ZN2nn2vi25QueryDisplayHdrPaperWhiteEPtPKNS0_7DisplayE) {
    *(uint16_t *)GP(ARG(0)) = (uint16_t)hdr_env("HOS_HDR_PAPER_WHITE", rx_option(72, 203));
    RET(0);
}
/* The layer's HDR metadata for the next frames (what the game says its picture holds) */
HLE(_ZN2nn2vi35SetLayerNextHdrContentGamutMetadataEPNS0_5LayerENS0_15HdrContentGamutE) {
    static uint64_t last = ~0ull;
    if (ARG(1) != last) last = ARG(1), hos_log("vi: the game's HDR content gamut %llu", (unsigned long long)last);
    RET(0);
}
HLE(_ZN2nn2vi39SetLayerNextHdrToSdrToneMappingMetadataEPNS0_5LayerENS0_14HdrToneMappingE) {
    static uint64_t last = ~0ull;
    if (ARG(1) != last) last = ARG(1), hos_log("vi: the game's HDR-to-SDR tone mapping %llu", (unsigned long long)last);
    RET(0);
}
HLE(_ZN2nn2vi52SetLayerNextHdrExtendedContentLuminanceRangeMetadataEPNS0_5LayerEtt) {
    static uint64_t last = ~0ull;
    const uint64_t v = (ARG(1) & 0xffff) << 16 | (ARG(2) & 0xffff);
    if (v != last) last = v, hos_log("vi: the game's extended content luminance %u..%u", (unsigned)(ARG(1) & 0xffff), (unsigned)(ARG(2) & 0xffff));
    RET(0);
}
HLE(_ZN2nn2vi52SetLayerNextHdrStandardContentLuminanceRangeMetadataEPNS0_5LayerEtt) {
    static uint64_t last = ~0ull;
    const uint64_t v = (ARG(1) & 0xffff) << 16 | (ARG(2) & 0xffff);
    if (v != last) last = v, hos_log("vi: the game's standard content luminance %u..%u", (unsigned)(ARG(1) & 0xffff), (unsigned)(ARG(2) & 0xffff));
    RET(0);
}
/* nn::vi::QueryDisplayVrrSupported(bool*, Display const*) -> Result */
HLE(_ZN2nn2vi24QueryDisplayVrrSupportedEPbPKNS0_7DisplayE) {
    *(uint8_t *)GP(ARG(0)) = 0;
    RET(0);
}
HLE(_ZN2nn2vi23SetDisplayVrrCapabilityEPNS0_7DisplayEb) { RET(0); }
HLE(_ZN2nn2vi28SetLayerExtendedSwapIntervalEPNS0_5LayerEii) { RET(0); }
HLE(_ZN2nn2vi30ClearLayerExtendedSwapIntervalEPNS0_5LayerE) { RET(0); }

/* nn::vi::GetNativeWindow(void**, nn::vi::Layer*) -> Result: a window object (NVN builds its window on it; the
   native NVN layer only keeps the pointer) */
HLE(_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE) {
    static uint64_t window;
    if (!window) window = hos_sys_alloc(0x100, 16);
    G64(ARG(0)) = window;
    hos_log("vi: GetNativeWindow -> %#llx", (unsigned long long)window);
    RET(0);
}

/* ---- frame pacing for the native NVN layer's window (nvn_native.c) ----
   The game acquires a window texture before each frame and presents it after; on the console the acquire blocks
   while every texture waits for the display. Here the acquire waits for the frame's display time: the present
   interval's vsyncs (1: 60 fps, 2: 30) after the last present, so the game runs at the display's rate.
   HOS_NO_PACING=1: as fast as it can. */
static volatile int present_interval = 1;
static int64_t next_frame_at, window_start, frames_in_window;
/* (relink) The frame rate cap: the game's 60, or unlocked (HOS_FPS_UNLOCKED=1):
   up to the display's refresh, game time following the clock (realtime_step below, whatever the Game speed row says: per frame
   at 100 fps the game would run 1.67 times too fast). HOS_FPS_MAX=<fps>: that cap instead of the display's (60 to 240). */
/* (relink, issue 16) On Windows, the refresh of the display the game's window is on, looked at again when the window goes to
   another display. It was the primary display's, read once: the game on a 144 Hz screen beside a 60 Hz primary one stayed
   at 60. Elsewhere, the desktop's refresh as SDL gives it, read once as before */
static int display_hz(void) {
    static int hz;
#ifdef _WIN32
    static int calls;
    static HWND window;
    static HMONITOR monitor;
    if (hz && ++calls % 120) return hz;
    if (!window) window = FindWindowA("ProjectReLink", NULL);   /* (host/display.c's window class) */
    HMONITOR m = window ? MonitorFromWindow(window, MONITOR_DEFAULTTOPRIMARY) : NULL;
    if (hz && m == monitor) return hz;
    monitor = m;
    MONITORINFOEXA mi;
    memset(&mi, 0, sizeof mi);
    mi.cbSize = sizeof mi;
    const char *device = m && GetMonitorInfoA(m, (MONITORINFO *)&mi) ? mi.szDevice : NULL;   /* (NULL: the primary display) */
    DEVMODEA dm;
    memset(&dm, 0, sizeof dm);
    dm.dmSize = sizeof dm;
    const int reported = EnumDisplaySettingsA(device, ENUM_CURRENT_SETTINGS, &dm) ? (int)dm.dmDisplayFrequency : 0;
    int now = reported > 1 ? reported : 60;
    now = now < 60 ? 60 : now > 240 ? 240 : now;
    if (now != hz) hos_log("vi: the game's display (%s) runs at %d Hz", device ? device : "primary, the window not found yet", reported);
    hz = now;
#else
    if (!hz) {
        hz = hos_sdl_desktop_hz();
        if (hz <= 1) hz = 60;
        hz = hz < 60 ? 60 : hz > 240 ? 240 : hz;
    }
#endif
    return hz;
}
/* (relink) HOS_FPS_CAP=<20-59>, which the launcher passes as 30 for its "30, smooth" frame rate: a cap under the game's 60,
   each frame shown for the same number of display refreshes (the renderer presents it again on a vsynced swapchain:
   gpu/vk/vk_replay.cpp present_image) and the next one started once the display has taken the last (hos_vi_native_acquire),
   game time following the clock whatever Game speed says. 0: none */
static int fps_low_cap(void) {
    static int low = -1;
    if (low < 0) {
        low = getenv("HOS_FPS_CAP") ? atoi(getenv("HOS_FPS_CAP")) : 0;
        low = low <= 0 || low >= 60 ? 0 : low < 20 ? 20 : low;
    }
    return low;
}
/* (relink) HOS_FPS_CAP's rate as the display shows it: a 30 cap on a 100 Hz display holds each frame 3 refreshes, 33.3 fps
   (gpu/vk option 14), and a timer at 30.0 left the display a refresh short every other frame: 30 and 40 ms frames in turn.
   The cap's own rate when the renderer holds nothing (no window) */
static double low_cap_effective(void) {
    int rx_option(int id, int otherwise);
    const int mfps = rx_option(14, 0);
    return mfps > 0 ? mfps / 1000.0 : (double)fps_low_cap();
}
static int fps_max(void) {
    static int env = -1;
    if (env < 0) env = getenv("HOS_FPS_MAX") ? atoi(getenv("HOS_FPS_MAX")) : 0;
    static int unlocked = -1;   /* (relink: HOS_FPS_UNLOCKED=1, which the launcher passes for Unlocked; no in-game row) */
    if (unlocked < 0) unlocked = getenv("HOS_FPS_UNLOCKED") && atoi(getenv("HOS_FPS_UNLOCKED")) != 0;
    int cap = 60;
    if (fps_low_cap() && !unlocked && env <= 60) {
        cap = fps_low_cap();
        static int told_low;
        if (!told_low++) hos_log("vi: frame rate cap %d fps (HOS_FPS_CAP), each frame held for the same number of refreshes", cap);
        return cap;
    }
    if (unlocked || env > 60) {
        cap = env > 60 ? env : display_hz();
        cap = cap < 60 ? 60 : cap > 240 ? 240 : cap;
        /* (relink, 0.7.9.3) 60 while a movie plays. The game's events count a movie in frames, one a frame drawn: at 120 fps
           the credits' last event reached its end in half the movie's time, put "The End" over the wrong picture and left a
           black screen after the button (seen on 20/11 with the ending's events run by hand; at 60 fps the same run goes
           through). The movies are 60 fps pictures: nothing is lost. HOS_VIDEO_UNLOCKED=1: as before */
        static int video_unlocked = -1;
        if (video_unlocked < 0) video_unlocked = getenv("HOS_VIDEO_UNLOCKED") != NULL;
        int hos_video_playing(void);
        if (cap > 60 && !video_unlocked && hos_video_playing()) cap = 60;
    }
    static int told;   /* (the cap in the log, at the start and when it changes: a report says which it was) */
    if (cap != told) {
        told = cap;
        hos_log("vi: frame rate cap %d fps (unlocked %s, HOS_FPS_MAX %d%s)", cap, unlocked ? "on" : "off", env,
                unlocked && env <= 60 ? ", the display's refresh" : "");
    }
    return cap;
}
static int64_t frame_period(void) {   /* (the display interval the frames are paced to) */
    const int cap = fps_max();
    if (cap < 60) {   /* (HOS_FPS_CAP: its period, or the game's interval when that is longer) */
        /* (with the display holding the frames, 3% under its period: the display's own clock, through presents_shown in
           hos_vi_native_acquire, is what paces them, and the timer never makes it wait a refresh more) */
        int rx_option(int id, int otherwise);
        const int held = rx_option(14, 0) > 0;
        const int64_t low = (int64_t)(1e9 / low_cap_effective() * (held ? 0.97 : 1.0)), game = 1000000000LL / 60 * present_interval;
        return low > game ? low : game;
    }
    return present_interval == 1 && cap > 60 ? 1000000000LL / cap : 1000000000LL / 60 * present_interval;
}
static volatile LONG64 presents;
int hos_vi_native_interval(Ctx *c, int requested) {
    (void)c;
    /* (relink) OoT's frame-rate controller asks for 2, 3 or 4 vsyncs a frame whenever one runs long (480 changes a minute
       in Kokiri): paced to them, frames snapped to 33, 50 and 67 ms, the stutter, and the frame rate sat far under what the
       PC managed. With game time following the clock (the Insert menu's Real time, realtime_step), a frame is shown as soon
       as it is ready, 60 at most; the game still reads back its own interval. HOS_GAME_INTERVAL=1, or Game speed Per frame:
       paced to the game's interval, as before */
    static int game_interval = -1;
    if (game_interval < 0) game_interval = getenv("HOS_GAME_INTERVAL") && atoi(getenv("HOS_GAME_INTERVAL")) != 0;
    int rx_option(int id, int otherwise);
    const int clamped = requested < 1 ? 1 : requested > 4 ? 4 : requested;
    present_interval = game_interval || !rx_option(0, 1) ? clamped : 1;
    return requested;
}
/* (relink) The window's textures as the console's display queue has them: each present is shown at its display
   time, the later of its present and one interval after the frame shown before it, and a texture is free again once
   the frame presented after it is shown. The acquire of a frame waits for its texture: with `active` textures, the
   texture of the frame `active` back, freed when the frame after that one is shown. With three (OoT) a frame slower
   than the display waits for nothing and a fast one is held to the display's rate, two frames queued at most;
   the acquire waited for one interval after the last present before (two textures' behaviour), which held every
   frame over 16.7 ms to the next vsync step (gameplay at 27 ms frames, 5 ms of them in this sleep). HOS_VI_DOUBLE=1:
   that behaviour */
static int64_t shown_at[16];
static volatile LONG64 shown_count;
void hos_vi_native_acquire(int active) {
    static int no_pacing = -1, double_buffered = -1;
    if (no_pacing < 0) no_pacing = getenv("HOS_NO_PACING") != NULL;
    if (double_buffered < 0) double_buffered = getenv("HOS_VI_DOUBLE") != NULL;
    {   /* (a test) HOS_TEST_SLOW_MS=<ms>: each frame's acquire held that much longer, as on a PC that cannot hold the frame
           rate (two players' endings hung with the movies at 40 to 53 fps) */
        static int64_t slow_ns = -1;
        if (slow_ns < 0) slow_ns = getenv("HOS_TEST_SLOW_MS") ? (int64_t)(atof(getenv("HOS_TEST_SLOW_MS")) * 1e6) : 0;
        if (slow_ns > 0) hos_sleep_ns(slow_ns);
    }
    if (no_pacing || !next_frame_at) return;
    int64_t until = next_frame_at;
    if (!double_buffered && active >= 2) {
        const int64_t n = shown_count, frame = n - active + 1;   /* (the frame whose showing frees this texture) */
        until = frame >= 0 && frame < n ? shown_at[frame & 15] : 0;
    }
    const int64_t now = hos_now_ns();
    if (until > now) hos_sleep_ns(until - now);
    if (fps_low_cap()) {
        /* (HOS_FPS_CAP) The display's own clock paces the frames: the renderer counts the presents the display has taken
           (option 12, gpu/vk/vk_replay.cpp present_image, each held for the same number of refreshes on a vsynced swapchain),
           and the next frame starts once all but one of the game's presents are among them. A timer alone at 30.000 fps
           against a 59.94 Hz display gained a frame every 16 s: a queued frame more of delay, then a dropped one. The two
           counts are compared from where they stood the first time (presents before the renderer started) and again from
           where they stand after a wait gave up (a present the renderer could not make: the window resized, minimized):
           at most 50 ms here, and a renderer that does not count leaves the timer above alone */
        int rx_option(int id, int otherwise);
        static int64_t offset = INT64_MIN;
        static long long waits_gave_up;
        const int64_t mine = presents, give_up = hos_now_ns() + 50000000LL;
        int shown = rx_option(12, -1);
        if (shown >= 0) {
            if (offset == INT64_MIN) offset = mine - 1 - shown;
            while ((int64_t)shown + offset < mine - 1) {
                if (hos_now_ns() >= give_up) {
                    offset = mine - 1 - shown;
                    if (waits_gave_up++ < 8) hos_log("vi: the display did not take a frame in 50 ms: counted again from here");
                    break;
                }
                hos_sleep_ns(500000);
                shown = rx_option(12, -1);
            }
        }
    }
}
/* (mkw-recomp) MKW's frame pacer (0x806bcbb0, x19 when it sets the interval: nvn_native.c): its measured frame time
   (+0x128, ticks), its running excess (+0x218) and interval (+0x214), logged with the frame rate */
volatile uint64_t hos_mkw_pacer;
/* HOS_SCAN_DT=<file prefix> (diagnostic): writes every float in guest memory between ~0.015 and ~0.04, with its
   address, four times 300 presents apart: the game's frame step is either one exact 1/60 or follows the frames */
static void scan_dt(long long presents) {
    static long long first; static const char *prefix; static int scans, read_env;
    if (!read_env) read_env = 1, prefix = getenv("HOS_SCAN_DT");
    if (!prefix) return;
    if (!first) {   /* (once <prefix>go exists) */
        if (presents % 60) return;
        char go[512];
        snprintf(go, sizeof go, "%sgo", prefix);
        FILE *g = fopen(go, "r");
        if (!g) return;
        fclose(g);
        first = presents;
    }
    if (presents < first + scans * 300 || scans >= 4) return;
    char name[512];
    snprintf(name, sizeof name, "%s%d.txt", prefix, scans++);
    FILE *f = fopen(name, "w");
    if (!f) return;
    size_t hits = 0;
    for (uint64_t a = 0x80000000ULL; a < 0x2000000000ULL && hits < 4000000;) {
        MEMORY_BASIC_INFORMATION m;
        if (!VirtualQuery(GP(a), &m, sizeof m)) break;
        const uint64_t size = m.RegionSize;
        if (m.State == MEM_COMMIT && (m.Protect & PAGE_READWRITE)) {
            const uint32_t *w = (const uint32_t *)m.BaseAddress;
            for (size_t i = 0; i < size / 4; i++)
                if (w[i] >= 0x3c800000 && w[i] <= 0x3d230000) {
                    fprintf(f, "%llx %08x\n", (unsigned long long)((uint8_t *)&w[i] - (uint8_t *)GP(0)), w[i]);
                    if (++hits >= 4000000) break;
                }
        }
        a = (uint64_t)((uint8_t *)m.BaseAddress - (uint8_t *)GP(0)) + size;
    }
    fclose(f);
    hos_log("vi: scan %s: %zu floats", name, hits);
}
/* (relink) Game time follows the clock. The engine steps every frame by ratio = 60 / <the graphics module's nominal fps,
   an int at +0x2c8> (main+0x6e6cc4, f_83b6e300 makes seconds = ratio / 60 and the rest), so at 50 fps the game ran at 83%
   speed: walking the same 8 s at 52 and 33 fps went as far as those rates. Each present writes the rate of the last 5
   display intervals there, the rounding error carried to the next frame so that no time is lost; a step is at most
   1/15 s, as TotK Recomp's frame_rate.c (slower frames slow the game down, as on the console), at least 1/60 s (the game's
   own limit holds it at 60). HOS_REALTIME_STEP=0: the shipping whole steps. */
extern volatile uint64_t hos_gfx_module;
volatile int hos_fixed_resolution = 0;   /* (the lifted code reads these: nx2/oot_patches.py) */
volatile uint32_t hos_fixed_scale_bits = 0x3f800000;   /* (HOS_RES_SCALE: the fixed path's scale) */
volatile int hos_dynres_runs = 0;   /* (the dynamic path's frames: it fills fields the 3D reads, +8 +16 +168) */
volatile float hos_text_scale = 1.0f;   /* (the letters' clock of a text pane, f_80b1e140: nx2/oot_patches.py) */
/* (relink, after HeyListen ddb156df) the scene buffer RGBA16F, not R11G11B10F, while the HDR tone mapper row is on PsychoV35:
   f_80672260's format picks it (nx2/oot_patches.py), so the scene PsychoV35 tone maps into has 6 mantissa bits more and its
   highlights do not band */
volatile int hos_scene_rgba16f = 0;
/* (relink) The camera's vertical angle is multiplied by this as it goes into the projection (nx2/oot_fov.py): the Insert menu's
   Field of view in percent (option 13), HOS_FOV=<percent> over it; 1 as the game has it */
volatile float hos_fov_scale = 1.0f;
/* (relink, issues 16 and 67) The frame rate the game's step is made from while the frame rate is unlocked, read by the main
   loop in place of the graphics module's own (nx2/oot_step_rate.py); 0: the module's, as under 60 fps. The game's frame rate
   controller writes 60 / <its display intervals> into the module every frame while the limiter is raised, after the write
   below and never more than 60: unlocked, the step stayed 1/60 s and the whole game ran at fps / 60 times its speed (x1.77
   at 102 fps, x1.24 at 75, measured with HOS_STEP_LOG=1). HOS_STEP_RATE_OLD=1: never set, as before */
volatile int hos_step_fps = 0;
/* (relink, issue 120) The water's ripple steps above 60 fps (nx2/oot_ripples.py): the game cuts the frame's step (bits, a
   float, 1.0 at 60 fps) to an integer, 0 at every frame above 60 fps, and the rings in water stood still. The step is added to
   a remainder kept for each surface (key) and its whole part handed out, so each surface runs at the game's 60 steps a second.
   64 slots, the least recently used taken for a new surface (a scene's surfaces each come every frame). HOS_RIPPLES_CUT=1: the
   game's cut, as before */
uint32_t hos_ripple_steps(uint64_t key, uint32_t bits) {
    static int cut = -1;
    if (cut < 0) cut = getenv("HOS_RIPPLES_CUT") != NULL;
    float step;
    memcpy(&step, &bits, 4);
    if (!(step > 0.0f) || step > 16.0f) return 0;   /* (NaN, none or absurd: nothing, as the cut of a NaN would) */
    if (cut) return (uint32_t)step;
    enum { SLOTS = 64 };
    static uint64_t keys[SLOTS], used[SLOTS], clock;
    static float rem[SLOTS];
    static SRWLOCK lock = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lock);
    int at = -1, oldest = 0;
    for (int i = 0; i < SLOTS; i++) {
        if (keys[i] == key && used[i]) { at = i; break; }
        if (used[i] < used[oldest]) oldest = i;
    }
    if (at < 0) at = oldest, keys[at] = key, rem[at] = 0.0f;
    used[at] = ++clock;
    rem[at] += step;
    const uint32_t whole = (uint32_t)rem[at];
    rem[at] -= (float)whole;
    ReleaseSRWLockExclusive(&lock);
    return whole;
}
/* (relink) The game's other whole-step counters above 60 fps (nx2/oot_step_cuts.py): the pick-up counter above Link's head
   (site 1), the time of day (2) and the Deku Tree's web (3) cut the frame's step times something to an integer, 0 at every
   frame above 60 fps, so the count froze. Same as the ripples: bits is the float they cut, added to a remainder kept for the
   object (key) and its whole part handed out; game is what the game cut, returned for a value no step could be (NaN, none,
   absurd). 96 slots shared by the sites, each keeping its own keys. HOS_PICKUP_CUT=1, HOS_CLOCK_CUT=1, HOS_WEB_CUT=1: the game's
   cut at that site, as before */
uint32_t hos_whole_steps(uint32_t site, uint64_t key, uint32_t bits, uint32_t game) {
    static const char *const names[4] = {NULL, "HOS_PICKUP_CUT", "HOS_CLOCK_CUT", "HOS_WEB_CUT"};
    static int cut[4] = {-1, -1, -1, -1};
    if (site >= 4 || !names[site]) return game;
    if (cut[site] < 0) cut[site] = getenv(names[site]) != NULL;
    float step;
    memcpy(&step, &bits, 4);
    if (cut[site] || !(step > 0.0f) || step > 4096.0f) return game;
    enum { SLOTS = 96 };
    static uint64_t keys[SLOTS], used[SLOTS], clock;
    static uint32_t sites[SLOTS];
    static float rem[SLOTS];
    static SRWLOCK lock = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lock);
    int at = -1, oldest = 0;
    for (int i = 0; i < SLOTS; i++) {
        if (used[i] && keys[i] == key && sites[i] == site) { at = i; break; }
        if (used[i] < used[oldest]) oldest = i;
    }
    if (at < 0) at = oldest, keys[at] = key, sites[at] = site, rem[at] = 0.0f;
    used[at] = ++clock;
    rem[at] += step;
    const uint32_t whole = (uint32_t)rem[at];
    rem[at] -= (float)whole;
    ReleaseSRWLockExclusive(&lock);
    return whole;
}
/* (relink) The physics solver's step count above 60 fps (nx2/oot_step_cuts.py, the two places the game sets it): the game
   cuts step * steps of a 60 fps frame + 0.001 (scaled_bits, a float; per_frame_bits, the steps of a 60 fps frame, a float)
   to an integer, short under a step of 1 and changing frame to frame as the measured step wanders (8 x 0.49 is 3, 8 x 0.5 is
   4), so a moving platform slid out from under Link. Under a step of 1 the count is rounded up instead, with a tenth of a
   step's leeway for the frame time's jitter, at least 1 and never under the game's; a step of 1 or more keeps the game's.
   HOS_SOLVER_CUT=1: the game's count, as before */
uint32_t hos_solver_steps(uint32_t game, uint32_t scaled_bits, uint32_t per_frame_bits) {
    static int cut = -1;
    if (cut < 0) cut = getenv("HOS_SOLVER_CUT") != NULL;
    float scaled, per_frame;
    memcpy(&scaled, &scaled_bits, 4);
    memcpy(&per_frame, &per_frame_bits, 4);
    scaled -= 0.001f;
    if (cut || !(scaled > 0.0f) || !(per_frame > 0.0f) || per_frame > 1000.0f || !(scaled < per_frame)) return game;
    const float want = scaled - 0.1f;
    int32_t steps = (int32_t)want;
    if ((float)steps < want) steps++;
    if (steps < 1) steps = 1;
    return (int32_t)game > steps ? game : (uint32_t)steps;
}
/* (relink) The sound manager's job above 60 fps (nx2/oot_step_cuts.py): it hands the step to the sound engine, whose fade and
   delay timers count updates of at least one whole step; fed less (0.5 at 120 fps) voices were not released when an area
   unloaded, the voices in use climbed to the game's budget and then new sounds and the music stopped. Above 60 fps the job
   runs at 60 Hz: phase 0 (the job's phase, w2) adds the step to a remainder and the job runs once a whole step has built up,
   the engine handed a step of 1 (hos_sound_step); the other phases follow that frame's choice. A step of 1 or more runs as
   the game has it and keeps the remainder (around 60 fps steps of 0.98 and 1.02 alternate: dropping the 0.98 ran the sound
   clock at half the real one); 0, under 0 and NaN run as the game has them. Every phase runs while the step is not
   followed (hos_step_fps 0). HOS_SOUND_CUT=1: the job every frame with the game's step, as before */
static struct { float part; int run, partial; SRWLOCK lock; } sound_gate = {0.0f, 1, 0, SRWLOCK_INIT};
int hos_sound_gate(uint32_t phase, uint32_t step_bits) {
    static int cut = -1;
    if (cut < 0) cut = getenv("HOS_SOUND_CUT") != NULL;
    int run;
    AcquireSRWLockExclusive(&sound_gate.lock);
    if (phase != 0) {
        run = sound_gate.run;
    } else {
        float step;
        memcpy(&step, &step_bits, 4);
        sound_gate.partial = 0;
        if (cut || hos_step_fps <= 60 || !(step > 0.0f)) {
            sound_gate.part = 0.0f;
            sound_gate.run = 1;
        } else if (step >= 1.0f) {
            sound_gate.run = 1;
        } else {
            sound_gate.partial = 1;
            sound_gate.part += step;
            sound_gate.run = sound_gate.part >= 1.0f;
            if (sound_gate.run) sound_gate.part -= 1.0f;
            static int told;
            if (!told) told = 1, hos_log("vi: the sound job runs at 60 Hz above 60 fps (step %.3f)", step);
        }
        run = sound_gate.run;
    }
    ReleaseSRWLockExclusive(&sound_gate.lock);
    return run;
}
/* The step the job hands the engine after its load: a whole one while the gate holds it to 60 Hz */
uint32_t hos_sound_step(uint32_t step_bits) {
    AcquireSRWLockExclusive(&sound_gate.lock);
    const int partial = sound_gate.partial;
    ReleaseSRWLockExclusive(&sound_gate.lock);
    return partial ? 0x3f800000u : step_bits;
}
/* The Insert menu's Text speed (0 Normal, 1 2x, 2 4x, 3 Instant), HOS_TEXT_SPEED=<0..3> over it: what the time of a letter
   animation's step is multiplied by. Instant: the page at once, as the game's own skip on A */
static float text_scale(const char *env, int menu) {
    const int v = env && *env ? atoi(env) : menu;
    return v == 1 ? 2.0f : v == 2 ? 4.0f : v >= 3 ? 1e6f : 1.0f;
}
volatile uint64_t hos_res_module = 0;   /* (f_831f3200's object, whose +0xe5c is the 3D's scale: nx2/oot_patches.py) */
static void realtime_step(int64_t now) {
    int rx_option(int id, int otherwise);
    /* (Resolution 100% from the start: the fixed path skipped the dynamic block from the first frame, its fields stayed
       unset and the 3D came out black. The dynamic path runs a few frames first, as when the menu turns it on in game) */
    hos_fixed_resolution = !rx_option(2, 1) && hos_dynres_runs >= 30;
    hos_scene_rgba16f = rx_option(74, 0) != 0;   /* (relink, after HeyListen ddb156df: HOS_OPT_HDR_TONEMAP, gpu/vk/vk_menu.h) */
    {
        static const char *env;
        static int looked;
        if (!looked) looked = 1, env = getenv("HOS_TEXT_SPEED");
        hos_text_scale = text_scale(env, rx_option(5, 0));
    }
    {   /* (relink) the field of view: nx2/oot_fov.py */
        static int env = -2;
        if (env == -2) env = getenv("HOS_FOV") && *getenv("HOS_FOV") ? atoi(getenv("HOS_FOV")) : -1;
        int pct = env > 0 ? env : rx_option(13, 100);
        pct = pct < 50 ? 50 : pct > 200 ? 200 : pct;
        hos_fov_scale = pct / 100.0f;
        static int told = -1;
        if (pct != told) {
            if (told != -1 || pct != 100) hos_log("vi: field of view %d%% of the game's", pct);
            told = pct;
        }
    }
    {   /* (the Insert menu's Resolution: the fixed path's scale in 1/10000, live; HOS_RES_SCALE
           below stays over it) */
        const int scale = rx_option(3, 0);
        if (scale > 0 && !getenv("HOS_RES_SCALE")) {
            const float v = scale / 10000.0f;
            memcpy((void *)&hos_fixed_scale_bits, &v, 4);
        }
    }
    {
        static int read_scale;
        if (!read_scale) {
            read_scale = 1;
            if (getenv("HOS_RES_SCALE")) { float v = (float)atof(getenv("HOS_RES_SCALE")); memcpy((void *)&hos_fixed_scale_bits, &v, 4); }
        }
    }
    {   /* HOS_RES_LOG=1: the 3D's scale every 300 presents. +0xe20: flags (bit 3: 0.75, bit 4: 0.5 in place of the scale),
           +0xe2c bit 3: recompute asked, +0xe50/+0xe54/+0xe58: the sizes the scale is taken against, +0xe5c: the scale */
        static int on = -1;
        static long long n;
        if (on < 0) on = getenv("HOS_RES_LOG") != NULL;
        const uint64_t m = hos_res_module;
        if (on && m && !(n++ % 300)) {
            float f[4], s;
            memcpy(f, GP(m + 0xe50), sizeof f);
            memcpy(&s, (const void *)&hos_fixed_scale_bits, 4);
            hos_log("vi: res: fixed %d (scale %.3f), +0xe20 %llx, +0xe2c %x, +0xe50 %.0f %.0f %.0f, 3D scale %.3f", hos_fixed_resolution, s,
                    (unsigned long long)*(uint64_t *)GP(m + 0xe20), *(uint32_t *)GP(m + 0xe2c), f[0], f[1], f[2], f[3]);
        }
    }
    const uint64_t mod = hos_gfx_module;
    if (!mod) return;
    {   /* (relink) The frame limiter (f_8042cd20 sleeps to 1 / +0xfc). The engine's adaptive controller writes +0xfc when
           variable refresh is on (+0x231, f_804fc0f0), any rate from 40 up. HOS_FPS_TARGET=<fps>: that rate kept, 0: left alone;
           HOS_FPS_LOG=1: what the game asks for, every 300 presents */
        static int target = -1, log_on = -1;
        static long long n;
        if (target < 0) target = getenv("HOS_FPS_TARGET") ? atoi(getenv("HOS_FPS_TARGET")) : 0, log_on = getenv("HOS_FPS_LOG") != NULL;
        int *limit = (int *)GP(mod + 0xfc);
        if (log_on && !(n++ % 300))
            hos_log("vi: frame limiter %d, vrr %d %d, nominal %d, controller +0x58.. %d %d %d %d", *limit, *(uint8_t *)GP(mod + 0x231),
                    *(uint8_t *)GP(mod + 0x232), *(int *)GP(mod + 0x2c8), *(int *)GP(mod + 0x270 + 0x58), *(int *)GP(mod + 0x270 + 0x28),
                    *(int *)GP(mod + 0x270 + 0x2c), *(int *)GP(mod + 0x270 + 0x30));
        const int cap = fps_max();
        if (cap > 60 && *limit > 0 && *limit < cap) *limit = cap;   /* (unlocked: the game's limiter up to the cap) */
        else if (target && *limit > 0 && *limit < target) *limit = target;
    }
    const int cap = fps_max();
    if (!rx_option(0, 1) && cap == 60) {   /* (the Insert menu: per frame. Not under HOS_FPS_CAP: at 30 the game ran at half speed) */
        int *slot = (int *)GP(mod + 0x2c8);
        if (*slot >= 15 && *slot <= 60) *slot = 60;
        hos_step_fps = 0;
        return;
    }
    static int64_t last;
    static double samples[5], carry;
    static unsigned count, next;
    static long long steps;
    const double dt = (now - last) / 1e9;
    last = now;
    if (dt <= 0 || dt > 0.25) { count = next = 0; carry = 0; hos_step_fps = 0; return; }   /* (a load: the average starts again) */
    /* (HOS_REALTIME_WINDOW=<1-5>: frames averaged for the step; 1: each step the last frame's own time) */
    static int window = -1;
    if (window < 0) {
        window = getenv("HOS_REALTIME_WINDOW") ? atoi(getenv("HOS_REALTIME_WINDOW")) : 5;
        window = window < 1 ? 1 : window > 5 ? 5 : window;
    }
    samples[next++ % window] = dt;
    if (count < (unsigned)window) count++;
    double sum = 0;
    for (unsigned i = 0; i < count; i++) sum += samples[i];
    double ratio = sum / count * 60.0 + carry;
    /* (HOS_FPS_CAP: the rate the display shows, 33.3 for 30 at 100 Hz: held to 30 the game stepped 1/30 s every 30 ms, 11% fast) */
    const double eff = cap < 60 ? low_cap_effective() : (double)cap;
    const int top = (int)(eff + 0.999);
    const double least = 60.0 / eff;   /* (unlocked: steps shorter than 1/60 s) */
    ratio = ratio < least ? least : ratio > 4.0 ? 4.0 : ratio;
    int fps = (int)(60.0 / ratio + 0.5);
    fps = fps < 15 ? 15 : fps > top ? top : fps;
    {   /* (test runs) HOS_TEST_STEP_FPS=<n>: the game's step is 1/n s whatever the frames take: the steps of a frame rate
           the bench cannot reach, in slow motion (issue 158 names 144 fps, the bench draws that room at 100 to 113) */
        static int forced = -1;
        if (forced < 0) forced = getenv("HOS_TEST_STEP_FPS") ? atoi(getenv("HOS_TEST_STEP_FPS")) : 0;
        if (forced >= 15 && forced <= 240) fps = forced;
    }
    int *slot = (int *)GP(mod + 0x2c8);
    if (*slot < 15 || *slot > 240) {
        static int logged;
        if (!logged++) hos_log("vi: the graphics module's frame rate reads %d: no real-time step", *slot);
        hos_step_fps = 0;
        return;
    }
    carry = ratio - 60.0 / fps;
    carry = carry > 0.5 ? 0.5 : carry < -0.5 ? -0.5 : carry;
    {   /* (relink, issues 16 and 67: "above 60 fps everything is faster") HOS_STEP_LOG=1: the game's own clock against the wall
           clock, a line every 5 s. The engine's step object (*[0x84c9e890], what f_83b6e300 fills each frame from the rate
           written here) keeps the step's ratio at +0x10 and its seconds at +0x18: the seconds are summed at each present, with
           the slot's value as the game left it since the last present (what else writes it, if anything) */
        static int on = -1;
        if (on < 0) on = getenv("HOS_STEP_LOG") != NULL;
        const uint64_t holder = on ? *(uint64_t *)GP(0x84c9e890ULL) : 0;
        const uint64_t obj = holder ? *(uint64_t *)GP(holder) : 0;
        if (obj) {
            static double game_s, wall_s, lo = 1e9, hi;
            static int n, slot_changed, last_written;
            float r, s;
            memcpy(&r, GP(obj + 0x10), 4);
            memcpy(&s, GP(obj + 0x18), 4);
            game_s += s, wall_s += dt, n++;
            lo = r < lo ? r : lo, hi = r > hi ? r : hi;
            if (last_written && *slot != last_written) slot_changed++;
            last_written = fps;
            if (wall_s >= 5.0) {
                hos_log("vi: step log: the game's clock advanced %.3f s in %.3f s of wall clock (x%.3f), %d presents (%.1f fps); its "
                        "step's ratio %.3f to %.3f (now %.3f, %.5f s); the rate written here read back changed %d times",
                        game_s, wall_s, game_s / wall_s, n, n / wall_s, lo, hi, r, s, slot_changed);
                game_s = wall_s = 0, n = slot_changed = 0, lo = 1e9, hi = 0;
            }
        }
    }
    *slot = fps;
    {   /* (issues 16 and 67) unlocked: the main loop takes the rate from here, whatever the game's controller writes after this */
        static int old_way = -1;
        if (old_way < 0) old_way = getenv("HOS_STEP_RATE_OLD") != NULL;
        hos_step_fps = cap > 60 && !old_way ? fps : 0;
    }
    if (!(steps++ % 3000)) hos_log("vi: real-time step on: ratio %.3f, game's frame rate %d", ratio, fps);
}
/* HOS_PRESENT_TRACE=<csv>: native game present-call returns, not GPU completion or display times.
   Uses the existing frame ID/clock; a forced stop can lose the last second of buffered present records. */
static void present_trace(int64_t frame, int64_t now) {
    static FILE *file;
    static char buffer[64 * 1024];
    static int looked;
    static int64_t last, flushed;
    if (!looked) {
        const char *path = getenv("HOS_PRESENT_TRACE");
        looked = 1;
        if (path && *path) {
            file = fopen(path, "w");
            if (!file) hos_log("vi: could not open present trace %s", path);
        }
        if (file) {
            setvbuf(file, buffer, _IOFBF, sizeof buffer);
            fputs("frame,return_ns,interval_ns\n", file);
            flushed = now;
        }
    }
    if (!file) return;
    fprintf(file, "%lld,%lld,%lld\n", (long long)frame, (long long)now, (long long)(frame > 1 ? now - last : 0));
    last = now;
    if (now - flushed >= 1000000000LL) {
        fflush(file);
        flushed = now;
    }
}
void hos_vi_native_presented(void) {
    static long long scan_n;
    scan_dt(++scan_n);
    {   /* HOS_FRAME_SLEEP_MS (diagnostic): slower frames, for game speed against frame rate */
        static int ms = -1;
        if (ms < 0) ms = getenv("HOS_FRAME_SLEEP_MS") ? atoi(getenv("HOS_FRAME_SLEEP_MS")) : 0;
        if (ms) hos_sleep_ns(ms * 1000000LL);
    }
    const int64_t period = frame_period(), now = hos_now_ns();
    realtime_step(now);
    /* (behind by more than a frame: the next one is paced from now, with no burst to catch up) */
    next_frame_at = next_frame_at && now - next_frame_at < period ? next_frame_at + period : now + period;
    {   /* (the display queue: this frame is shown one interval after the last, or now when the display waits) */
        const int64_t n = shown_count, last = n ? shown_at[(n - 1) & 15] : 0;
        shown_at[n & 15] = last + period > now ? last + period : now;
        shown_count = n + 1;
    }
    const int64_t n = InterlockedIncrement64(&presents);
    present_trace(n, now);
    {   /* HOS_MKW_INTERVAL=<0|1|2>: the interval MKW's pacer is held at, through its own override (+0x210, any
           non-zero value is used as the interval): 1 (the default): a steady 60 fps; 2: 30; 0: the game's choice,
           which with frames between 16.7 and 33 ms swings between the two (25-27 fps, stutters). Its timing counts
           vsyncs, so an interval the renderer cannot keep slows the game down rather than dropping frames */
        static int forced = -1;
        if (forced < 0) forced = getenv("HOS_MKW_INTERVAL") ? atoi(getenv("HOS_MKW_INTERVAL")) : 1;
        if (hos_mkw_pacer && forced >= 0 && forced <= 2) G32(hos_mkw_pacer + 0x210) = (uint32_t)forced;
    }
    if (!window_start) window_start = now;
    frames_in_window++;
    /* (present-to-present intervals of the window, in buckets: which frames miss their vsync) */
    static int64_t last_present;
    static int hist[6];
    if (last_present) {
        const int64_t d = now - last_present;
        hist[d < 15500000 ? 0 : d < 17800000 ? 1 : d < 22000000 ? 2 : d < 30000000 ? 3 : d < 40000000 ? 4 : 5]++;
    }
    {   /* (relink) HOS_PRESENT_LOG=<file>: every present's time (ns since the first), one per line, for the stutter's pattern */
        static FILE *plog;
        static int plog_read;
        static int64_t plog_t0;
        if (!plog_read) {
            plog_read = 1;
            if (getenv("HOS_PRESENT_LOG")) {
                plog = fopen(getenv("HOS_PRESENT_LOG"), "w"), plog_t0 = now;
                hos_log("vi: present log starts at %lld ns", (long long)now);   /* (aligns it with HOS_SAMPLE_ALL's "t0") */
            }
        }
        if (plog) {
            fprintf(plog, "%lld\n", (long long)(now - plog_t0));
            if (!(n % 60)) fflush(plog);
        }
    }
    last_present = now;
    if (now - window_start >= 5000000000LL) {
        hos_log("vi: %lld frames presented (%.1f fps over the last %.1f s, interval %d)", (long long)n,
                frames_in_window * 1e9 / (double)(now - window_start), (now - window_start) / 1e9, present_interval);
        hos_log("vi: present intervals: <15.5 ms %d, 15.5-17.8 %d, 17.8-22 %d, 22-30 %d, 30-40 %d, over 40 %d", hist[0],
                hist[1], hist[2], hist[3], hist[4], hist[5]);
        memset(hist, 0, sizeof hist);
        window_start = now, frames_in_window = 0;
        const uint64_t pacer = hos_mkw_pacer, freq = pacer ? G64(0x837de940) : 0;   /* (MKW only: nvn_native.c) */
        if (pacer && freq)
            hos_log("vi: the game's pacer: frame work %.1f ms, excess %.1f ms, interval %d",
                    (double)(int64_t)G64(pacer + 0x128) * 1e3 / (double)freq,
                    (double)(int64_t)G64(pacer + 0x218) * 1e3 / (double)freq, (int)G32(pacer + 0x214)),
            hos_log("vi: pacer raw: now %llx f0 %llx 128 %llx e8 %llx 50 %x 210 %x 22c %x freq %llu", (unsigned long long)hos_ticks(),
                    (unsigned long long)G64(pacer + 0xf0), (unsigned long long)G64(pacer + 0x128),
                    (unsigned long long)G64(pacer + 0xe8), G32(pacer + 0x50), G32(pacer + 0x210), *(const uint8_t *)GP(pacer + 0x22c),
                    (unsigned long long)freq);
    }
}
