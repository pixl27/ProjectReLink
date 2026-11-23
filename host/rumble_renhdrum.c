/* (hookshot) RenHDrum's backends: host/rumble_renhdrum.h says what they are and who calls what. */
#include "hos.h"
#include "rumble_renhdrum.h"
#include "hos_sdl.h"
#include "vibration.h"
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../third_party/renhdrum/include/renhdrum.h"

#define RH_ABI 0x000400u            /* include/renhdrum.h's RENHDRUM_VERSION; 0x000401 keeps its ABI */
#define RH_STALE_NS 250000000LL     /* a side the game stopped sending to goes silent: the DSPs hold their last value */
#define RH_TEST_AMP 0.5f
#define RH_FREQ_LOW 160.0f          /* the game's own when it sends none */
#define RH_FREQ_HIGH 320.0f
#define RH_PADS 8

/* ---- RenHDrum, loaded from beside the game by its full path (no folder of ours or PATH searched) ---- */
#define RH_FUNCS(X)                                                                                                    \
    X(uint32_t, version, (void))                                                                                       \
    X(void, set_nx2_mode, (int32_t))                                                                                   \
    X(RenHdcDualSenseDsp *, dualsense_dsp_create, (int32_t, bool))                                                    \
    X(void, dualsense_dsp_destroy, (RenHdcDualSenseDsp *))                                                             \
    X(int32_t, dualsense_dsp_set_half_target, (RenHdcDualSenseDsp *, bool, float, float, float, float))               \
    X(int32_t, dualsense_dsp_fill_f32, (RenHdcDualSenseDsp *, float *, size_t))                                       \
    X(int32_t, dualsense_dsp_build_bt_report, (RenHdcDualSenseDsp *, uint8_t, uint8_t *))                             \
    X(RenHdcXboxRumble *, xbox_rumble_create, (bool))                                                                  \
    X(void, xbox_rumble_destroy, (RenHdcXboxRumble *))                                                                 \
    X(int32_t, xbox_rumble_push_half_sample, (RenHdcXboxRumble *, int64_t, bool, float, float, float, float))         \
    X(int32_t, xbox_rumble_tick, (RenHdcXboxRumble *, int64_t, uint16_t *))                                           \
    X(RenHdcEngine *, engine_create, (bool, bool))                                                                     \
    X(void, engine_destroy, (RenHdcEngine *))                                                                          \
    X(int32_t, engine_push_half_sample, (RenHdcEngine *, int64_t, bool, float, float, float, float))                  \
    X(int32_t, engine_tick, (RenHdcEngine *, int64_t, uint8_t *))
static struct {
#define X(ret, name, args) ret(*name) args;
    RH_FUNCS(X)
#undef X
} rh;
static int rh_tried, rh_ok;
static char rh_reason[256];

static void why(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(rh_reason, sizeof rh_reason, fmt, ap);
    va_end(ap);
    hos_log("vibration: RenHDrum: %s", rh_reason);
}
const char *rh_why(void) { return rh_reason; }

static HMODULE open_library(void) {
#ifdef _WIN32
    wchar_t path[1024];
    DWORD n = GetModuleFileNameW(NULL, path, (DWORD)(sizeof path / sizeof path[0]));
    if (!n || n >= sizeof path / sizeof path[0]) return why("the game's own folder could not be read"), (HMODULE)NULL;
    while (n && path[n - 1] != L'\\' && path[n - 1] != L'/') n--;
    static const wchar_t file[] = L"renhdrum.dll";
    if (n + sizeof file / sizeof file[0] > sizeof path / sizeof path[0]) return why("the game's folder is too long"), (HMODULE)NULL;
    memcpy(path + n, file, sizeof file);
    if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) return why("renhdrum.dll is not beside the game"), (HMODULE)NULL;
    /* (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: its own folder, for what it imports too, and System32; no PATH, no current folder) */
    /* (SEM_FAILCRITICALERRORS: a file that is not a usable DLL fails here, and its controllers keep their motors, rather
       than Windows stopping the game's start with a "Bad Image" box: a release once shipped a Linux library under this
       name, 0xc000012f. The thread's own mode, put back after) */
    DWORD mode = 0;
    BOOL quiet = SetThreadErrorMode(SEM_FAILCRITICALERRORS, &mode);
    HMODULE m = LoadLibraryExW(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    DWORD err = m ? 0 : GetLastError();
    if (quiet) SetThreadErrorMode(mode, NULL);
    if (!m) why("renhdrum.dll would not load (error %lu)", (unsigned long)err);
    return m;
#else
    char path[4096];
    DWORD n = GetModuleFileNameA(NULL, path, (DWORD)sizeof path);
    if (!n || n >= sizeof path) return why("the game's own folder could not be read"), (HMODULE)NULL;
    while (n && path[n - 1] != '/') n--;
    if (n + sizeof "renhdrum.so" > sizeof path) return why("the game's folder is too long"), (HMODULE)NULL;
    memcpy(path + n, "renhdrum.so", sizeof "renhdrum.so");
    FILE *f = fopen(path, "rb");
    if (!f) return why("renhdrum.so is not beside the game"), (HMODULE)NULL;
    fclose(f);
    HMODULE m = LoadLibraryA(path);
    if (!m) why("renhdrum.so would not load (%s)", hos_load_error());
    return m;
#endif
}

int rh_load(void) {
    if (rh_tried) return rh_ok;
    rh_tried = 1;
    HMODULE m = open_library();
    if (!m) return 0;
#define X(ret, name, args)                                                                                             \
    if (!(*(FARPROC *)&rh.name = GetProcAddress(m, "renhdrum_" #name))) {                                              \
        why("it has no renhdrum_" #name ": the plain rumble only");                                                   \
        FreeLibrary(m);                                                                                                \
        return 0;                                                                                                      \
    }
    RH_FUNCS(X)
#undef X
    const uint32_t v = rh.version();
    if (v != RH_ABI && v != 0x000401u) {
        why("it is version %u.%u.%u, this build wants %u.%u.%u", (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF,
            (RH_ABI >> 16) & 0xFF, (RH_ABI >> 8) & 0xFF, RH_ABI & 0xFF);
        FreeLibrary(m);
        return 0;
    }
    rh.set_nx2_mode(2);   /* (a Switch 2 game: RenHDrum's Mode 2 curves, for every pad it plays) */
    hos_log("vibration: RenHDrum %u.%u.%u loaded", (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
    rh_reason[0] = 0;
    return rh_ok = 1;
}

/* ---- what the game sends, and everything that plays it (lk) ---- */
static SRWLOCK lk = SRWLOCK_INIT;
static SRWLOCK life = SRWLOCK_INIT;   /* (opening and closing: the pad thread's, and the game's end; never inside lk) */
static int closed;                    /* (the game is over: nothing opens again) */
static int64_t at[2];                 /* when each side last sent (0 never) */
static int quiet[2] = {1, 1};
static int testing;
static volatile LONG gain_percent = 100;
static RenHdcDualSenseDsp *usb_dsp, *bt_dsp;
static void *usb_pad, *bt_pad;
static struct {
    void *pad;
    int backend;
    RenHdcXboxRumble *xbox;
    RenHdcEngine *engine;
} pads[RH_PADS];

static int64_t now_us(void) { return hos_now_ns() / 1000; }

static void feed(int side, float al, float fl, float ah, float fh) {   /* (lk held) */
    const bool right = side == 1;
    const int64_t us = now_us();
    if (usb_dsp) rh.dualsense_dsp_set_half_target(usb_dsp, right, al, fl, ah, fh);
    if (bt_dsp) rh.dualsense_dsp_set_half_target(bt_dsp, right, al, fl, ah, fh);
    /* (the Switch engine has no gain of its own: the strength scales what it is given, for testers' runs) */
    const float g = vibration_gain((int)gain_percent);
    const float sal = al * g > 1.0f ? 1.0f : al * g, sah = ah * g > 1.0f ? 1.0f : ah * g;
    for (int i = 0; i < RH_PADS; i++) {
        if (pads[i].xbox) rh.xbox_rumble_push_half_sample(pads[i].xbox, us, right, al, fl, ah, fh);
        if (pads[i].engine) rh.engine_push_half_sample(pads[i].engine, us, right, sal, fl, sah, fh);
    }
}

void rh_push(int side, const float v[4]) {
    if (!rh_ok || side < 0 || side > 1 || !v) return;
    float al = v[0], ah = v[2];
    al = al > 0.0f ? (al < 1.0f ? al : 1.0f) : 0.0f;   /* (also a NaN: 0) */
    ah = ah > 0.0f ? (ah < 1.0f ? ah : 1.0f) : 0.0f;
    const float fl = v[1] > 0.0f ? v[1] : RH_FREQ_LOW, fh = v[3] > 0.0f ? v[3] : RH_FREQ_HIGH;
    AcquireSRWLockExclusive(&lk);
    at[side] = hos_now_ns(), quiet[side] = 0;
    if (!testing) feed(side, al, fl, ah, fh);
    ReleaseSRWLockExclusive(&lk);
}

static void recreate_bt_dsp(int percent);
void rh_tick(int64_t now, int test, int percent) {
    if (!rh_ok) return;
    const int was_percent = (int)InterlockedExchange(&gain_percent, percent);
    AcquireSRWLockExclusive(&lk);
    if (test) {   /* (the Test vibration row, or a strength just changed: both bands, half the game's top) */
        testing = 1;
        for (int s = 0; s < 2; s++) feed(s, RH_TEST_AMP, 180.0f, RH_TEST_AMP, RH_FREQ_HIGH), quiet[s] = 0, at[s] = now;
    } else if (testing) {
        testing = 0;
        for (int s = 0; s < 2; s++) feed(s, 0.0f, RH_FREQ_LOW, 0.0f, RH_FREQ_HIGH), quiet[s] = 1, at[s] = 0;
    }
    for (int s = 0; s < 2; s++)   /* (a side gone quiet: the game stopped sending, and the DSP would hold its value) */
        if (at[s] && !quiet[s] && now - at[s] > RH_STALE_NS) feed(s, 0.0f, RH_FREQ_LOW, 0.0f, RH_FREQ_HIGH), quiet[s] = 1;
    ReleaseSRWLockExclusive(&lk);
    if (percent != was_percent && bt_dsp) recreate_bt_dsp(percent);
}

/* ---- a DualSense on a cable: its sound card's third and fourth channels are the voice coils ---- */
static void *usb_stream;
static int usb_token = -1;
static void usb_pull(void *user, uint8_t *out, int len) {   /* (the sound card's thread: lk, for one fill) */
    (void)user;
    float *f = (float *)out;
    const int frames = len / (int)(4 * sizeof(float));
    int filled = 0;
    AcquireSRWLockExclusive(&lk);
    if (usb_dsp) filled = rh.dualsense_dsp_fill_f32(usb_dsp, f, (size_t)frames) >= 0;
    ReleaseSRWLockExclusive(&lk);
    if (!filled) {
        memset(out, 0, (size_t)len);   /* (nothing to play: silence, never a held waveform) */
        return;
    }
    /* the Vibration strength, once, on the samples the DSP made: its law stays the core's (a gain before it would go
       through its curve, and past 100 % its normalising held the output still) */
    const float g = vibration_gain((int)gain_percent);
    for (int i = 0; i < frames; i++)
        for (int ch = 2; ch < 4; ch++) {
            const float v = f[i * 4 + ch] * g;
            f[i * 4 + ch] = v < -1.0f ? -1.0f : v > 1.0f ? 1.0f : v;
        }
}
int rh_usb_find(void) {
    if (!rh_ok) return 0;
    int found = -1, named = 0;
    char id[512], name[256];
    const int n = hos_sdl_output_count();
    for (int i = 0; i < n; i++) {
        if (!hos_sdl_output_info(i, id, (int)sizeof id, name, (int)sizeof name)) continue;
        if (!strstr(name, "DualSense") && !strstr(name, "Wireless Controller")) continue;
        named++;
        if (found < 0 && hos_sdl_output_channels(i) >= 4) found = hos_sdl_output_resolve(id);
    }
    static int said = -2;   /* (said when it changes, not at every look) */
    if (found != said) {
        said = found;
        hos_log(found >= 0 ? "vibration: the DualSense's four-channel sound card is there"
                           : "vibration: %d playback devices, %d named for a DualSense, none with four channels", n, named);
    }
    usb_token = found;
    return found >= 0;
}
int rh_usb_open(void *pad) {
    if (!rh_ok || usb_token < 0) return 0;
    AcquireSRWLockExclusive(&life);
    int ok = 0;
    if (!closed && !usb_pad) {
        RenHdcDualSenseDsp *dsp = rh.dualsense_dsp_create(100, false);   /* (its full gain: the strength is ours) */
        if (dsp) {
            AcquireSRWLockExclusive(&lk);
            usb_dsp = dsp;
            ReleaseSRWLockExclusive(&lk);
            usb_stream = hos_sdl_audio_open((unsigned)usb_token, 48000, 4, 0, usb_pull, NULL);
            if (usb_stream) {
                usb_pad = pad, ok = 1;
                hos_log("vibration: the DualSense's voice coils on its cable (48 kHz, four channels)");
            } else {
                AcquireSRWLockExclusive(&lk);
                usb_dsp = NULL;
                ReleaseSRWLockExclusive(&lk);
                rh.dualsense_dsp_destroy(dsp);
                hos_log("vibration: the DualSense's sound card would not open");
            }
        }
    }
    ReleaseSRWLockExclusive(&life);
    return ok;
}
int rh_usb_alive(void) { return usb_stream && hos_sdl_audio_alive(usb_stream); }
static void usb_close_locked(void) {   /* (life held) */
    void *stream = usb_stream;
    usb_stream = NULL, usb_pad = NULL;
    if (stream) hos_sdl_audio_close(stream);   /* (waits for its callback, which takes lk: lk is not held here) */
    AcquireSRWLockExclusive(&lk);
    RenHdcDualSenseDsp *dsp = usb_dsp;
    usb_dsp = NULL;
    ReleaseSRWLockExclusive(&lk);
    if (dsp) rh.dualsense_dsp_destroy(dsp);
}
void rh_usb_close(void) {
    AcquireSRWLockExclusive(&life);
    usb_close_locked();
    ReleaseSRWLockExclusive(&life);
}
void *rh_usb_owner(void) { return usb_pad; }

/* ---- a DualSense over Bluetooth: report 0x32 every 10.7 ms, written beside SDL's own reports ---- */
static void *bt_hid;
static HANDLE bt_thread;
static volatile LONG bt_run, bt_failed;
static DWORD WINAPI bt_main(LPVOID unused) {
    (void)unused;
    void hos_host_thread_register_self(const char *name);   /* (mkw_glue.c) */
    hos_host_thread_register_self("vibration (DualSense, Bluetooth)");
    uint8_t report[RENHDRUM_DS5_BT_REPORT_SIZE];
    int64_t next = hos_now_ns(), loud_at = 0;
    int fails = 0;
    while (bt_run) {
        next += (int64_t)RENHDRUM_DS5_BT_PERIOD_NS;
        int r = -1;
        AcquireSRWLockExclusive(&lk);
        if (bt_dsp) r = rh.dualsense_dsp_build_bt_report(bt_dsp, RENHDRUM_DS5_BT_BUFFER_LENGTH_DEFAULT, report);
        ReleaseSRWLockExclusive(&lk);
        const int64_t now = hos_now_ns();
        if (r == 1) loud_at = now;
        /* (sent while something plays and two seconds after: the radio stays free for the pad's input otherwise) */
        if (r >= 0 && loud_at && now - loud_at < 2000000000LL) {
            if (hos_sdl_hid_write(bt_hid, report, (int)sizeof report) < 0) {
                if (++fails >= 50) {
                    hos_log("vibration: the DualSense's Bluetooth reports keep failing: its motors instead");
                    InterlockedExchange(&bt_failed, 1);
                    break;
                }
            } else {
                fails = 0;
            }
        }
        if (now > next + 4 * (int64_t)RENHDRUM_DS5_BT_PERIOD_NS) next = now;   /* (far behind: start again from now) */
        if (next > now) Sleep((DWORD)((next - now + 999999) / 1000000));
    }
    return 0;
}
static void recreate_bt_dsp(int percent) {
    /* (Bluetooth's samples are made inside the report: the strength is the DSP's own volume, 10 to 100 %, which RenHDrum
       sets only when one is made. Above 100 % stays at 100 %) */
    const int v = percent < 10 ? 10 : percent > 100 ? 100 : percent;
    AcquireSRWLockExclusive(&life);
    if (bt_dsp) {
        RenHdcDualSenseDsp *dsp = rh.dualsense_dsp_create(v, false), *old;
        if (dsp) {
            AcquireSRWLockExclusive(&lk);
            old = bt_dsp, bt_dsp = dsp;
            ReleaseSRWLockExclusive(&lk);
            rh.dualsense_dsp_destroy(old);
        }
    }
    ReleaseSRWLockExclusive(&life);
}
int rh_bt_open(void *pad, const char *hid_path) {
    if (!rh_ok || !hid_path || !*hid_path) return 0;
    AcquireSRWLockExclusive(&life);
    int ok = 0;
    if (!closed && !bt_pad) {
        void *hid = hos_sdl_hid_open(hid_path);
        const int p = (int)gain_percent;
        RenHdcDualSenseDsp *dsp = hid ? rh.dualsense_dsp_create(p < 10 ? 10 : p > 100 ? 100 : p, false) : NULL;
        if (dsp) {
            AcquireSRWLockExclusive(&lk);
            bt_dsp = dsp;
            ReleaseSRWLockExclusive(&lk);
            bt_hid = hid, bt_pad = pad;
            InterlockedExchange(&bt_run, 1), InterlockedExchange(&bt_failed, 0);
            bt_thread = CreateThread(NULL, 0, bt_main, NULL, 0, NULL);
            ok = bt_thread != NULL;
        }
        if (!ok) {
            hos_log("vibration: the DualSense's HID device would not open for its Bluetooth haptics: its motors instead");
            AcquireSRWLockExclusive(&lk);
            bt_dsp = NULL;
            ReleaseSRWLockExclusive(&lk);
            if (dsp) rh.dualsense_dsp_destroy(dsp);
            if (hid) hos_sdl_hid_close(hid);
            bt_hid = NULL, bt_pad = NULL;
        } else {
            hos_log("vibration: the DualSense's voice coils over Bluetooth (report 0x32)");
        }
    }
    ReleaseSRWLockExclusive(&life);
    return ok;
}
int rh_bt_alive(void) { return bt_thread && !bt_failed; }
static void bt_close_locked(void) {   /* (life held) */
    HANDLE thread = bt_thread;
    bt_thread = NULL;
    InterlockedExchange(&bt_run, 0);
    if (thread) WaitForSingleObject(thread, 2000), CloseHandle(thread);   /* (it takes lk only, never life) */
    if (bt_hid) hos_sdl_hid_close(bt_hid);
    bt_hid = NULL, bt_pad = NULL;
    AcquireSRWLockExclusive(&lk);
    RenHdcDualSenseDsp *dsp = bt_dsp;
    bt_dsp = NULL;
    ReleaseSRWLockExclusive(&lk);
    if (dsp) rh.dualsense_dsp_destroy(dsp);
}
void rh_bt_close(void) {
    AcquireSRWLockExclusive(&life);
    bt_close_locked();
    ReleaseSRWLockExclusive(&life);
}
void *rh_bt_owner(void) { return bt_pad; }

/* ---- an Xbox pad's four motors, a Switch pad's packets ---- */
int rh_pad_open(void *pad, int backend, int bluetooth, int pro) {
    if (!rh_ok || !pad) return 0;
    AcquireSRWLockExclusive(&life);
    int ok = 0;
    for (int i = 0; i < RH_PADS && !closed && !ok; i++) {
        if (pads[i].pad) continue;
        RenHdcXboxRumble *x = backend == VIB_RH_XBOX ? rh.xbox_rumble_create(bluetooth != 0) : NULL;
        RenHdcEngine *e = backend == VIB_RH_SWITCH ? rh.engine_create(bluetooth != 0, pro != 0) : NULL;
        if (!x && !e) break;
        AcquireSRWLockExclusive(&lk);
        pads[i].pad = pad, pads[i].backend = backend, pads[i].xbox = x, pads[i].engine = e;
        ReleaseSRWLockExclusive(&lk);
        ok = 1;
    }
    ReleaseSRWLockExclusive(&life);
    return ok;
}
void rh_pad_close(void *pad) {
    AcquireSRWLockExclusive(&life);
    for (int i = 0; i < RH_PADS; i++) {
        if (!pad || pads[i].pad != pad) continue;
        AcquireSRWLockExclusive(&lk);
        RenHdcXboxRumble *x = pads[i].xbox;
        RenHdcEngine *e = pads[i].engine;
        memset(&pads[i], 0, sizeof pads[i]);
        ReleaseSRWLockExclusive(&lk);
        if (x) rh.xbox_rumble_destroy(x);   /* (out of the list first: nothing else reaches it now) */
        if (e) rh.engine_destroy(e);
    }
    ReleaseSRWLockExclusive(&life);
}
int rh_xbox_poll(void *pad, int64_t now, uint16_t motors[4]) {
    int due = 0;
    AcquireSRWLockShared(&lk);
    for (int i = 0; i < RH_PADS; i++)
        if (pads[i].pad == pad && pads[i].xbox) due = rh.xbox_rumble_tick(pads[i].xbox, now / 1000, motors) == 1;
    ReleaseSRWLockShared(&lk);
    if (due) {   /* (the Vibration strength on each motor's speed) */
        const float g = vibration_gain((int)gain_percent);
        for (int m = 0; m < 4; m++) {
            const float v = motors[m] * g;
            motors[m] = (uint16_t)(v > 65535.0f ? 65535.0f : v);
        }
    }
    return due;
}
int rh_switch_poll(void *pad, int64_t now, uint8_t packet[10]) {
    int due = 0;
    AcquireSRWLockShared(&lk);
    for (int i = 0; i < RH_PADS; i++)
        if (pads[i].pad == pad && pads[i].engine) due = rh.engine_tick(pads[i].engine, now / 1000, packet) == 1;
    ReleaseSRWLockShared(&lk);
    return due;
}

void rh_shutdown(void) {
    if (!rh_ok) return;
    AcquireSRWLockExclusive(&life);
    closed = 1;
    usb_close_locked();
    bt_close_locked();
    ReleaseSRWLockExclusive(&life);
    for (int i = 0; i < RH_PADS; i++)
        if (pads[i].pad) rh_pad_close(pads[i].pad);
}
