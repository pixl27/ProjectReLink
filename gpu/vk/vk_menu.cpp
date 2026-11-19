// The Insert menu. Drawn on the CPU (GDI text into a bitmap) and copied over the window's image at the last step of a present,
// so it costs nothing while closed and it never touches the game's picture, nor what frame interpolation sees. Keys: Insert
// opens and closes it, Up and Down choose, Left and Right change, Esc closes; the game gets no keyboard input while it is open.
// (relink) A gamepad's Home button, or Select and Start together, opens it too; the D-pad chooses and changes, the bottom
// face button changes too, the right one closes. The game gets no gamepad input while it is open either.
// Its Controls page binds the Switch's buttons to gamepad inputs and keys.
// (relink) The mouse is free while it is open (the window lets it go: WM_APP + 8, host/display.c and
// host/posix/display_sdl.c): the row under the cursor is selected, a click on a tab shows it, on a value's left or right
// half changes it down or up, on a toggle flips it, on a slider's track sets it (or drag it); the wheel changes the value
// under the cursor, else moves the selection; a right click is Esc. On the Cheats page and its lists a click is Enter (on a
// choice's or an amount's left or right half, Left or Right) and a right click goes back. vk_menu_hit.h has the layout.
// (relink) The rows that take a restart (Row::restart) say "(after restart)" while their value is not the one the game
// started with, and the Graphics tab lists them. Closing the menu then asks whether to restart now (menu_restart.h):
// Later keeps the saved settings for the next start and is not asked again until they change; Restart now saves and
// closes the window as the player would, and the game starts again (WM_APP + 9: host/hos_relaunch.c).
// (relink, issue 32) The Gameplay tab's Quit game asks the same way (Back, the default, or Quit); Quit saves the options
// and the pipeline cache as Restart now does and closes the window as the player would, and the game exits (WM_APP + 10:
// host/display.c, host/posix/display_sdl.c). The launcher puts the display's refresh rate back when the game's process ends.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <map>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#include "vk_paths.h"
#include "vk_menu.h"
#include "vk_menu_hit.h"
#include "vk_draw.h"   // (pipeline_cache_save: Restart now keeps what was built)
#include "oot_cheat_items.h"
#include "oot_teleport_capture.h"   // (relink: the Travel page's pictures)
#include "hos_volume.h"   // (relink: the Volume row's option id)
#include "oot_tp_places.h"   // (relink: the Travel page's places)
#include "prompt_tex.h"   // (relink: the HUD cross's letters follow the button font)
#include "vk_screenshot.h"   // (relink: F12)
#include "hdr_out.h"   // (relink: the bitmaps in HDR10 when the window is)
#include "vk_startup_paint.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)
#include "vk_dlss.h"
#include "vk_fsr.h"
#include "vk_reflex.h"
#include "upscaler_rows.h"
#include "menu_restart.h"
#ifndef RELINK_VERSION   // (nx2/build.py gives it to this file: nx2/version.py)
#define RELINK_VERSION "unknown"
#endif

namespace mkwvk {
HWND app_window();

Options opts;

namespace {

enum MenuTab {
    TAB_GRAPHICS = 0,
    TAB_GAMEPLAY = 1,
    TAB_CONTROLS = 2,
    TAB_COUNT = 3
};

struct Row {
    MenuTab tab;
    const char* name;
    int* value;
    int lo, hi, step;
    const char* (*text)(int v);
    const char* help;   // (shown under the list while the row is selected: what the row is, the same whatever its value)
    // (a line about the value chosen, under the help, or nullptr: hint(v, -1) is the one to show now for v, hint(v, 0),
    // hint(v, 1), ... every one v can have (those that depend on the other rows or on the GPU), until nullptr, so the
    // menu keeps the height of the tallest: opt_help_most)
    const char* (*hint)(int v, int variant) = nullptr;
    bool restart = false;   // (read once, at the start: a change applies after a restart)
    bool preset_exempt = false;   // (a change here leaves the Quality preset alone: the controls, the window, the counter)
};
/// A row's hints from a table, the value lo first (nullptr: none for that value)
const char* hint_in(const char* const* t, int n, int lo, int v, int variant) {
    return variant <= 0 && v - lo >= 0 && v - lo < n ? t[v - lo] : nullptr;
}
#define MENU_HINTS(fn, lo, ...)                                                                                              \
    const char* fn(int v, int variant) {                                                                                     \
        static const char* const t[] = {__VA_ARGS__};                                                                        \
        return hint_in(t, (int)(sizeof t / sizeof t[0]), lo, v, variant);                                                    \
    }
const char* tx_framegen(int v) {
    static const char* t[] = {"Off", "", "2x", "3x", "4x", "5x", "6x", "7x", "8x"};
    return t[v];
}
const char* tx_onoff(int v) { return v ? "On" : "Off"; }
// (relink) Motion controls: the choice, and whether a connected controller has a gyroscope (host/pad_sdl.c). Either can be
// chosen whatever is connected: a pad plugged in later is read when it comes
const char* tx_gyro(int v) {
    const int status = host && host->gyro_status ? host->gyro_status() : -1;
    if (status == 2) return v ? "On (gyro found)" : "Off (gyro found)";
    if (status == 1) return v ? "On (no gyro on this pad)" : "Off (no gyro on this pad)";
    if (status == 0) return v ? "On (no controller)" : "Off (no controller)";
    return v ? "On" : "Off";
}
const char* tx_mousecam(int v) { return v ? "Native" : "As the stick"; }
const char* tx_sens(int v) {
    static char b[12];
    snprintf(b, sizeof b, "%.2f", v / 100.0);
    return b;
}
// (the game's picture and the window it is shown in, as the last present had them: menu_note_present_sizes)
std::atomic<uint64_t> present_sizes{0};
bool filter_needed() {
    const uint64_t v = present_sizes.load(std::memory_order_relaxed);
    return upscaler_rows::filter_needed((unsigned)(v >> 48), (unsigned)(v >> 32) & 0xffff, (unsigned)(v >> 16) & 0xffff,
                                        (unsigned)v & 0xffff);
}
const char* tx_upscale(int v) {
    static const char* t[] = {"Off", "Catmull-Rom", "Lanczos"};
    if ((unsigned)v >= sizeof t / sizeof t[0]) return t[0];   // (a setting past the row: the label, not the next bytes)
    return t[v];
}
/// The Window scaling filter row's value: its filter, or that none is needed (the saved value stays, for another window size)
const char* tx_filter(int v) { return filter_needed() ? tx_upscale(v) : "Not needed (window = output)"; }
const char* tx_reflex(int v) {
    if (reflex_is_amd()) return v > 0 ? "On (AMD Anti-Lag 2)" : "Off";
    static const char* t[] = {"Off", "On", "On + Boost"};
    return t[reflex_available() ? std::clamp(v, 0, 2) : 0];
}
/// Reflex / Anti-Lag: on the value, or that it is not available here (the row shows Off)
const char* hn_reflex(int v, int variant) {
    static const char* const t[] = {"The game as without it.", "The driver paces the frames for less latency.",
                                    "As On, and the GPU's clocks are kept up as well."};
    static const char* const t_amd[] = {"The game as without it.",
                                        "AMD Anti-Lag 2 (VK_AMD_anti_lag) paces CPU simulation to reduce input latency."};
    static const char* const none = "Not available here: needs VK_NV_low_latency2 (NVIDIA) or VK_AMD_anti_lag (AMD).";
    if (reflex_is_amd()) {
        const char* cur = t_amd[v > 0 ? 1 : 0];
        return variant < 0 || variant == 0 ? cur : variant == 1 ? none : nullptr;
    }
    if (variant < 0) return reflex_available() ? t[std::clamp(v, 0, 2)] : none;
    return variant == 0 ? t[std::clamp(v, 0, 2)] : variant == 1 ? none : nullptr;
}
// The Upscaler (opts.dlss): what scales the 3D up to the output. DLSS and DLAA only where dlss_why() lets them run, (relink)
// AMD FSR and its native AA where fsr_why() does; the menu shows (and steps from) what runs, dlss_mode(), and keeps the saved
// choice for a PC where it can
namespace ur = upscaler_rows;
int upscaler() { return dlss_mode(); }
bool dlss_ok() { return dlss_why() == DLSS_OK; }
bool fsr_ok() { return fsr_why() == FSR_OK; }
/// (relink) FSR 4 runs: AMD's runtime makes it on this GPU and the FSR version row leaves it on Auto
bool fsr4_runs() { return fsr4_available() && !(opts.fsr_version == 1 && fsr31_available()); }
const char* tx_upscaler(int) {
    static const char* t[] = {"Game's own", "NVIDIA DLSS", "NVIDIA DLAA", "AMD FSR", "AMD FSR native AA"};
    static const char* t4[] = {"Game's own", "NVIDIA DLSS", "NVIDIA DLAA", "AMD FSR 4", "AMD FSR 4 native AA"};
    return (fsr4_runs() ? t4 : t)[std::clamp(upscaler(), 0, ur::kUpscalers - 1)];
}
/// The Upscaler: the one shown (what runs, upscaler()), and with the game's own, why DLSS and DLAA, and FSR, are not offered
const char* hn_upscaler(int v, int variant) {
    static const char* const t[] = {
        "The game's own network.",
        "NVIDIA's, from a smaller 3D. Softer than the game's own: set Sharpness to around 80%. Things that move by "
        "themselves can trail for now.",
        "NVIDIA's, the 3D at full size with its edges smoothed; the slowest. Softer than the game's own: set Sharpness to "
        "around 80%. Things that move by themselves can trail for now.",
        "AMD's FSR 3.1, from a smaller 3D, on any GPU. Softer than the game's own: set Sharpness to around 80%. Things "
        "that move by themselves can trail for now.",
        "AMD's FSR 3.1, the 3D at full size with its edges smoothed, on any GPU; slower. A little softer than the game's "
        "own. Things that move by themselves can trail for now.",
    };
    // (relink: FSR 4, AMD's network, where its runtime makes it on this GPU: the RX 6000 series and later)
    static const char* const t4[] = {
        "AMD's FSR 4, its network, from a smaller 3D. Sharper and steadier than FSR 3.1, at more GPU time on an RX 6000 or "
        "7000. The FSR version row can go back to 3.1.",
        "AMD's FSR 4, its network, the 3D at full size with its edges smoothed; the slowest of AMD's. The FSR version row "
        "can go back to 3.1.",
    };
    static const char* const why[DLSS_WHY_COUNT] = {
        nullptr,
        "DLSS and DLAA: not in this build (built without NVIDIA's DLSS SDK, see the readme).",
        "DLSS and DLAA: off for this run (RX_VK_DLSS=0).",
        "DLSS and DLAA: need an NVIDIA RTX GPU.",
        "DLSS and DLAA: need a newer NVIDIA driver.",
        "DLSS and DLAA: DLSS's runtime was not found (the game folder's ngx folder, see the readme).",
    };
    static const char* const fsr_why_text[FSR_WHY_COUNT] = {
        nullptr,
        "AMD FSR: not started yet.",
        "AMD FSR: Windows only.",
        "AMD FSR: off for this run (RX_VK_FSR=0).",
#if defined(__linux__)
        "AMD FSR: libamd_fidelityfx_vk.so is not in the game folder (nor set in RX_VK_FSR_LIB).",
        "AMD FSR: libamd_fidelityfx_vk.so did not load (the log says why).",
        "AMD FSR: libamd_fidelityfx_vk.so has no upscaler for Vulkan.",
#else
        "AMD FSR: AMD's amd_fidelityfx_loader_vk.dll is not in the game folder.",
        "AMD FSR: AMD's DLLs did not load (the log says why; they need Microsoft's Visual C++ runtime).",
        "AMD FSR: AMD's DLLs have no upscaler for Vulkan.",
#endif
        "AMD FSR: this GPU or its driver cannot run it.",
    };
    // (now: the value shown and what runs here; every one: each value, and the game's own with each pair of reasons)
    int w = 0, f = 0;
    if (variant < 0) v = std::clamp(upscaler(), 0, ur::kUpscalers - 1), w = (int)dlss_why(), f = (int)fsr_why();
    else if (v == ur::GAME) w = variant / FSR_WHY_COUNT, f = variant % FSR_WHY_COUNT;
    else if ((v == ur::FSR || v == ur::FSR_AA) && variant == 1) return t4[v - ur::FSR];   // (relink: its FSR 4 line, for the height)
    else if (variant > 0) return nullptr;
    v = std::clamp(v, 0, ur::kUpscalers - 1);
    if (w >= DLSS_WHY_COUNT || f >= FSR_WHY_COUNT) return nullptr;
    if ((v == ur::FSR || v == ur::FSR_AA) && variant < 0 && fsr4_runs()) return t4[v - ur::FSR];
    if (v != ur::GAME || (!why[w] && !fsr_why_text[f])) return t[v];
    static char b[DLSS_WHY_COUNT][FSR_WHY_COUNT][320];
    snprintf(b[w][f], sizeof b[w][f], "%s%s%s%s%s", t[0], why[w] ? " " : "", why[w] ? why[w] : "",
             fsr_why_text[f] ? " " : "", fsr_why_text[f] ? fsr_why_text[f] : "");
    return b[w][f];
}
/// The DLSS model row (opts.dlss_model): NVIDIA's network by its preset, or RX_VK_DLSS_PRESET's for this run
const char* dlss_preset_env() {
    static const char* const e = getenv("RX_VK_DLSS_PRESET");
    return ur::preset_env(e) ? e : nullptr;
}
const char* tx_dlss_model(int v) {
    static const char* const t[] = {"Transformer (K)", "Transformer 2 (M)", "Transformer 2 (L)", "NVIDIA default"};
    if (const char* e = dlss_preset_env()) {
        static char b[48];
        const unsigned p = ur::dlss_preset(v, e);
        if (p) snprintf(b, sizeof b, "Preset %c (RX_VK_DLSS_PRESET)", ur::preset_letter(p));
        else snprintf(b, sizeof b, "NVIDIA default (RX_VK_DLSS_PRESET)");
        return b;
    }
    return t[std::clamp(v, 0, ur::kModels - 1)];
}
/// The DLSS model: the value chosen; variant 1: the game's own upscales (the row waits for DLSS or DLAA); variant 2: set for
/// this run by RX_VK_DLSS_PRESET
const char* hn_dlss_model(int v, int variant) {
    static const char* const t[] = {
        "NVIDIA's transformer model: the one tested most here.",
        "NVIDIA's second-generation transformer, its pick for Performance: sharper, at a little more GPU time.",
        "The second-generation transformer as NVIDIA tuned it for Ultra Performance, the smallest Render resolutions.",
        "Lets DLSS's runtime pick for each mode: what it picks can change with a runtime update.",
    };
    static const char* const unused = "Used by NVIDIA DLSS and DLAA: choose one in the Upscaler row.";
    static const char* const env = "Set for this run by RX_VK_DLSS_PRESET.";
    if (variant < 0) variant = dlss_preset_env() ? 2 : upscaler() != ur::DLSS && upscaler() != ur::DLAA ? 1 : 0;
    if (variant == 1) return unused;
    if (variant == 2) return env;
    return variant == 0 ? t[std::clamp(v, 0, ur::kModels - 1)] : nullptr;
}
/// (relink) The FSR version row (opts.fsr_version): Auto (the newest AMD's runtime makes here) or FSR 3.1
const char* tx_fsr_version(int v) {
    if (fsr_ok() && !fsr4_available()) {   // (FSR 3.1 is all there is here: the row says so)
        static char b[48];
        snprintf(b, sizeof b, "FSR %s", fsr_version());
        return b;
    }
    return v == 1 ? "FSR 3.1" : "Auto (FSR 4)";
}
/// The FSR version: the value chosen; variant 1: another upscaler draws; variant 2: FSR 4 is not made on this GPU
const char* hn_fsr_version(int v, int variant) {
    static const char* const t[] = {
        "FSR 4, AMD's network: sharper and steadier, at more GPU time (about 2 ms a frame at 1080p on an RX 6600).",
        "FSR 3.1: lighter on the GPU, softer, and things that move can shimmer more.",
    };
    static const char* const unused = "Used by AMD FSR and its native AA: choose one in the Upscaler row.";
    static const char* const only31 = "AMD's runtime makes FSR 4 on the RX 6000 series and later, with a recent driver: FSR 3.1 here.";
    if (variant < 0) variant = !fsr_ok() || (upscaler() != ur::FSR && upscaler() != ur::FSR_AA) ? 1 : !fsr4_available() ? 2 : 0;
    if (variant == 1) return unused;
    if (variant == 2) return only31;
    return variant == 0 ? t[std::clamp(v, 0, 1)] : nullptr;
}
const char* tx_percent(int v) {
    static char b[8];
    snprintf(b, sizeof b, "%d%%", v);
    return b;
}
// The output the menu's sizes are shown for: the Output resolution row's (applied at the next start), so the rows below follow a change
// at once; the handheld profile runs the game's own (1536x864)
struct Out {
    uint32_t w, h;
};
constexpr Out kOutputs[] = {{2048, 1152}, {2560, 1440}, {3840, 2160}, {1920, 1080}};
constexpr int kLastOutput = std::size(kOutputs) - 1;
int output_choice(int v) { return v >= 0 && v <= kLastOutput ? v : 0; }
// Display the sizes from smallest to largest while keeping the IDs used by saved settings.
constexpr int kOutputOrder[] = {3, 0, 1, 2};
static_assert(std::size(kOutputOrder) == std::size(kOutputs));
int step_output_choice(int id, int dir) {
    const int* pos = std::find(std::begin(kOutputOrder), std::end(kOutputOrder), output_choice(id));
    const int next = std::clamp((int)(pos - std::begin(kOutputOrder)) + dir, 0, kLastOutput);
    return kOutputOrder[next];
}
Out shown_output() {
    return opts.handheld ? Out{1536, 864} : kOutputs[output_choice(opts.output_res)];
}
// The size the game draws the 3D at for a fixed scale of its output, a twelfth of it: every twelfth asked came out exact at the three
// outputs (2048x1152: 0.58 drew 1195x672, 0.6667 1366x768). So the scale the game is given is the twelfth too: 0.67 drew 2615x1471 at
// 3840x2160, not the 2560x1440 of 8/12.
int twelfths(float scale) { return std::min(std::max((int)(scale * 12.0f + 0.5f), 1), 12); }
void game_size(float scale, uint32_t& w, uint32_t& h) {
    const Out o = shown_output();
    const int k = twelfths(scale);
    w = (o.w * k + 11) / 12, h = o.h * k / 12;
}
const char* tx_resolution(int v) {
    static char b[48];
    const int up = upscaler();
    if (up == ur::DLAA) return "Native (DLAA)";
    if (up == ur::FSR_AA) return "Native (FSR native AA)";
    if (!ur::percent(v)) return "Auto (dynamic)";
    if (const char* mode = ur::temporal_steps(up) ? ur::dlss_mode_name(v) : nullptr) {
        snprintf(b, sizeof b, "%s (%d%%)", mode, ur::percent(v));
        return b;
    }
    uint32_t w, h;
    game_size(ur::percent(v) / 100.0f, w, h);
    snprintf(b, sizeof b, "%d%% (%ux%u)", ur::percent(v), w, h);
    return b;
}
/// The Render resolution: the value chosen, for the Upscaler shown (every one: each value with each Upscaler)
const char* hn_resolution(int v, int variant) {
    const int up = variant < 0 ? std::clamp(upscaler(), 0, ur::kUpscalers - 1) : variant;
    if (up >= ur::kUpscalers) return nullptr;
    if (up == ur::DLAA)
        return "DLAA draws the 3D at full size: nothing to set here. Choose DLSS or the game's own upscaler to draw it "
               "smaller.";
    if (up == ur::FSR_AA)
        return "FSR native AA draws the 3D at full size: nothing to set here. Choose AMD FSR or the game's own upscaler to "
               "draw it smaller.";
    const int p = ur::percent(v);
    if (up == ur::FSR) {   // (relink) FSR's modes are DLSS's shares, by the same names
        switch (p) {
        case 0: return "The game lowers it when the GPU is busy, and FSR follows.";
        case 67: return "FSR's Quality mode, its sharpest.";
        case 58: return "FSR's Balanced mode, between Quality and Performance.";
        case 50: return "FSR's Performance mode: faster, and softer.";
        case 33: return "FSR's Ultra Performance mode: its fastest, and its softest.";
        default: return nullptr;   // (not one of FSR's steps: the row moves to one)
        }
    }
    if (up == ur::DLSS) {
        switch (p) {
        case 0: return "The game lowers it when the GPU is busy, and DLSS follows. Set Sharpness to around 80%.";
        case 67: return "DLSS's sharpest mode. Set Sharpness to around 80%.";
        case 58: return "Between Quality and Performance. Set Sharpness to around 80%.";
        case 50: return "Faster, and softer. Set Sharpness to around 80%.";
        case 33: return "DLSS's fastest mode, and its softest. Set Sharpness to around 80%.";
        default: return nullptr;   // (not one of DLSS's steps: the row moves to one)
        }
    }
    if (!p) return "The game lowers it when the GPU is busy.";
    if (p == 100) return "The 3D at the output's full size: the sharpest, and the slowest.";
    return "A steady size: lower is faster, and softer.";
}
/// A row with nothing else to choose here: shown without its arrows, Left and Right leave it (and the saved value) alone
bool row_fixed(const int* value) {
    return (value == &opts.reflex && !reflex_available()) || (value == &opts.dlss && ur::upscaler_fixed(dlss_ok(), fsr_ok())) ||
           (value == &opts.resolution && ur::resolution_fixed(upscaler())) || (value == &opts.upscale && !filter_needed()) ||
           (value == &opts.dlss_model && ur::model_fixed(upscaler(), dlss_preset_env() != nullptr)) ||
           (value == &opts.fsr_version && (!fsr31_available() || (upscaler() != ur::FSR && upscaler() != ur::FSR_AA)));
}
const char* tx_output_res(int v) {
    static char b[32];
    v = output_choice(v);
    const Out o = kOutputs[v];
    snprintf(b, sizeof b, "%ux%u%s", o.w, o.h, v ? "" : " (game's)");
    return b;
}
const char* tx_speed(int v) { return v ? "Real time" : "Per frame"; }
const char* tx_text_speed(int v) {
    static const char* t[] = {"Normal", "2x", "4x", "Instant"};
    if ((unsigned)v >= sizeof t / sizeof t[0]) return t[0];
    return t[v];
}
const char* tx_ultrawide(int v) { return v ? "Monitor's shape" : "16:9"; }
const char* tx_cycle_items(int v) {   // (relink) the Cycle items row
    static const char* t[] = {"Off", "Last used", "Basic", "+Bombs", "+Tools"};
    return t[(unsigned)v < 5 ? v : 0];
}
const char* tx_gpu(int v) {
    static char buf[300];
    const int idx = v - 1;
    if (v <= 0) snprintf(buf, sizeof buf, "Auto");
    else if (idx < (int)gpu_device_names.size()) snprintf(buf, sizeof buf, "%d: %s", idx, gpu_device_names[idx].c_str());
    else snprintf(buf, sizeof buf, "%d: Device %d", idx, idx);
    // (a name too long for its row is cut short in it: paint_options)
    return buf;
}
const char* tx_pad_layout(int v);
// (relink) the Microphone row: Off, the system's default, or one of the host's microphones (host/nn_audio_in.c: the
// ocarina's input streams), told to the host by its name (host->mic_select) whenever it changes. The chosen one is kept by
// its name, mic_want, also while it is unplugged: the row then has one more entry for it, "<name> (not connected)", the
// host lets the default stand in and opens it again once it is back, and the name stays in the ini. In the ini the name is
// cut to kMicNameKept bytes (at a character's start), so the line fits the 80-byte readers of the file (host/nn_hid.c,
// nn_applet.c, hos_main.c), and a name is matched by that much of it
constexpr int kMaxMics = 16;
constexpr size_t kMicNameKept = 60;
std::string mic_want;      // (the chosen microphone's name, cut to kMicNameKept; empty: Off or the default chosen)
bool mic_missing = false;  // (mic_want is not plugged in: the row's last entry, 2 + mic_listed, stands for it)
int mic_listed = 0;        // (the microphones as the host last listed them)
bool can_mic() { return host && host->mic_count && host->mic_name && host->mic_select; }
std::string mic_key(const char* name) {
    std::string k(name ? name : "");
    if (k.size() <= kMicNameKept) return k;
    size_t cut = kMicNameKept;
    while (cut > 0 && ((unsigned char)k[cut] & 0xC0) == 0x80) cut--;   // (not in the middle of a UTF-8 character)
    return k.substr(0, cut);
}
const char* tx_mic(int v) {
    static char buf[160];
    if (v <= 0) return "Off";
    if (v == 1) return "Default";
    if (mic_missing && v == 2 + mic_listed) {
        snprintf(buf, sizeof buf, "%s (not connected)", mic_want.c_str());
        return buf;
    }
    const char* n = can_mic() ? host->mic_name(v - 2) : "";
    snprintf(buf, sizeof buf, "%s", n && *n ? n : "Microphone");
    return buf;
}
/// The Microphone row: Off, the default, a microphone of the list, or the chosen one while it is unplugged (variant 1)
const char* hn_mic(int v, int variant) {
    if (v <= 0) return variant <= 0 ? "The game hears silence." : nullptr;
    if (v == 1)
        return variant <= 0 ? "The system's default recording device. On Windows, desktop apps must be allowed to use the "
                              "microphone (Settings, Privacy & security)."
                            : nullptr;
    const bool missing = variant < 0 ? mic_missing && v == 2 + mic_listed : variant == 1;
    if (variant > 1) return nullptr;
    return missing ? "Not plugged in: Default stands in until it is back." : "This microphone, by its name. A change applies at once.";
}
const char* tx_rumble(int v) { return v ? tx_percent(v) : "Off"; }
// (relink) Audio output: the speakers the game mixes for, read once at its start (host/nn_audio.c
// au_menu_output). A player with no sound, or with sound from the wrong speakers, can say Stereo here (issue 1)
const char* tx_audio_output(int v) {
    static const char* t[] = {"Auto", "Stereo", "5.1"};
    return (unsigned)v < 3 ? t[v] : t[0];
}
const char* tx_robust(int v) {
    static const char* t[] = {"Auto", "Off", "On"};
    return (unsigned)v < 3 ? t[v] : t[0];
}
const char* tx_preset(int v) {
    static const char* t[] = {"Custom", "Minimum (4GB)", "Low (6GB)", "Balanced (8GB)", "High (12GB)", "Ultra (16GB+)"};
    return t[std::min(std::max(v, 0), 5)];
}
void sync_preset_budgets(int p) {
    switch (p) {
    case 1: opts.vram_mb = 2200;  opts.mirror_mb = 256;  break;
    case 2: opts.vram_mb = 3600;  opts.mirror_mb = 512;  break;
    case 3: opts.vram_mb = 4800;  opts.mirror_mb = 1024; break;
    case 4: opts.vram_mb = 7500;  opts.mirror_mb = 2048; break;
    case 5: opts.vram_mb = 11000; opts.mirror_mb = 3072; break;
    default: opts.vram_mb = 0;    opts.mirror_mb = 0;    break;
    }
}
void apply_preset(int p) {
    switch (p) {
    case 1: // Minimum (4GB VRAM / iGPU)
        opts.resolution = 4; opts.upscale = 2; opts.sharpness = 50;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 1;
        break;
    case 2: // Low (6GB VRAM)
        opts.resolution = 2; opts.upscale = 2; opts.sharpness = 35;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 1;
        break;
    case 3: // Balanced (8GB VRAM)
        opts.resolution = 1; opts.upscale = 1; opts.sharpness = 25;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 0;
        break;
    case 4: // High (12GB VRAM)
        opts.resolution = 1; opts.upscale = 2; opts.sharpness = 30;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 0;
        break;
    case 5: // Ultra (16GB+ VRAM)
        opts.resolution = 1; opts.upscale = 2; opts.sharpness = 40;
        opts.framegen = 3;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 0;
        break;
    default:
        break;
    }
    // (the Upscaler is the player's: a preset never turns DLSS on or off. With DLSS its resolution is DLSS's nearest mode)
    if (p > 0) opts.resolution = ur::snap(upscaler(), opts.resolution);
    sync_preset_budgets(p);
}

MENU_HINTS(hn_preset, 0, "Each row as you set it. Changing most rows yourself comes back to Custom.",
           "Resolution 50% (with DLSS its nearest mode), Lanczos, Sharpness 50%, 2x frames, Handheld profile.",
           "Resolution 75% (with DLSS its nearest mode), Lanczos, Sharpness 35%, 2x frames, Handheld profile.",
           "Resolution 100% (with DLSS its nearest mode), Catmull-Rom, Sharpness 25%, 2x frames.",
           "Resolution 100% (with DLSS its nearest mode), Lanczos, Sharpness 30%, 2x frames.",
           "Resolution 100% (with DLSS its nearest mode), Lanczos, Sharpness 40%, 3x frames.")
MENU_HINTS(hn_gpu, 0, "The best dedicated GPU.")
/// Frame generation: how many frames are made for each of the game's
const char* hn_framegen(int v, int variant) {
    if (variant > 0 || v == 1) return nullptr;
    if (v <= 0) return "Only the game's own frames: no added delay.";
    static char b[9][96];
    snprintf(b[v], sizeof b[v], "%s between each two of the game's.", v == 2 ? "One made frame" : v == 3 ? "Two made frames" : v == 4 ? "Three made frames"
             : v == 5 ? "Four made frames" : v == 6 ? "Five made frames" : v == 7 ? "Six made frames" : "Seven made frames");
    return b[std::clamp(v, 2, 8)];
}
MENU_HINTS(hn_pace, 0, "As soon as it is ready.", "At an even interval: less stutter, at up to one frame of delay.")
MENU_HINTS(hn_vsync, 0, "Each frame is shown as soon as it is ready: the least delay. Where the driver has no mailbox mode "
           "(AMD's on Windows), a line can tear across the picture.",
           "Each frame waits for the display's refresh: no tearing, a little more delay. Below the display's rate, a frame "
           "now and then stays on for two refreshes.")
MENU_HINTS(hn_output_res, 0, "The game's own size.",
           "The sharpest on a 2560x1440 screen: nothing to scale to the window.",
           "The sharpest on a 3840x2160 screen. The game's render targets get a bigger GPU pool, so a full-size 3D fits.",
           "The sharpest on a 1920x1080 screen: the HUD and text pixel for pixel, nothing to scale to the window.")
/// The Window scaling filter: its filter, or that none is needed at this window's size
const char* hn_filter(int v, int variant) {
    static const char* const t[] = {"The plain copy.", "Sharper.", "Sharper still; can ring on edges."};
    static const char* const none = "The window is the Output resolution's size: nothing to scale. The choice is kept for "
                                    "another size.";
    if (variant < 0) return filter_needed() ? t[std::clamp(v, 0, 2)] : none;
    return variant == 0 ? t[std::clamp(v, 0, 2)] : variant == 1 ? none : nullptr;
}
/// Sharpness: with DLSS or DLAA, the amount that suits them
const char* hn_sharpness(int v, int variant) {
    static const char* const dlss = "DLSS and DLAA look softer than the game's own upscaler: around 80% suits them.";
    // (relink) FSR measured as soft as DLSS in the Market (the mean edge strength of a house front at 67%: the game's own
    // 19, DLSS 10.8, FSR 9.1): the same advice. Its native AA is nearer the game's own (15): none
    static const char* const fsr = "AMD FSR looks softer than the game's own upscaler, as DLSS does: around 80% suits it.";
    const int up = upscaler();
    const int which = variant < 0 ? (up == ur::DLSS || up == ur::DLAA ? 1 : up == ur::FSR ? 2 : 0) : variant;
    if (which > 2) return nullptr;
    return which == 1 ? dlss : which == 2 ? fsr : v == 0 ? "Off: the picture as it is." : nullptr;
}
MENU_HINTS(hn_handheld, 0, "Docked: the Output resolution's size.", "Lower resolutions, lighter on the GPU.")
MENU_HINTS(hn_ultrawide, 0, "The game's own shape.",
    "The camera widens to fill a screen wider than 16:9 (21:9, 32:9), the HUD in the middle. A 16:10 or 4:3 screen keeps 16:9.")
MENU_HINTS(hn_robust, 0, "On, on every card.", "Can be faster on some NVIDIA cards.", "Always on.")
MENU_HINTS(hn_fps, 0, nullptr, "With frame generation on, the game's own rate too.")
MENU_HINTS(hn_speed, 0, "A step each frame: the game slows down whenever it runs below 60 fps.",
           "Normal speed down to 15 fps, by taking longer steps.")
MENU_HINTS(hn_text_speed, 0, "The game's own pace, with its pauses.", "Twice as fast.", "Four times as fast.",
           "Each page at once, as pressing A does.")
MENU_HINTS(hn_item_wheel, 0, "R and L scroll the game's own item row.",
           "Pages of 8 items, the same for the child and the adult. Y moves an item; Left/Right or the mouse wheel turn the page.")
MENU_HINTS(hn_cycle_items, 0, "R and L open the item row (or the wheel) only.",
           "A tap of R or L: the item that was on ZR before this one, whatever it is. Two taps: back again.",
           "A tap of R or L: the next or previous arrow, bottle or spell on ZR.", "Basic, and the bombs and bombchus too.",
           "+Bombs, and the hookshot, hammer, slingshot and boomerang too.")
MENU_HINTS(hn_audio_output, 0, "5.1 when the default output device has 5.1 or more speakers, else stereo.",
           "Stereo, whatever the device has.", "5.1, whatever the device has. Mixed down on stereo speakers.")
MENU_HINTS(hn_background, 0, "Only while the game's window has the focus.",
           "Also behind other windows, for streaming or other apps. The keyboard and mouse still need the game's window.")
MENU_HINTS(hn_mousecam, 0, "The mouse stands in for the right stick, with its curve and top speed.",
           "The mouse turns the camera itself, one to one, as in a PC game.")
MENU_HINTS(hn_mouse_invert, 0, nullptr, "Moving the mouse forward tilts the camera down.")
MENU_HINTS(hn_gyro, 0, "The sticks alone aim.",
           "Switch Pro controllers and pads in Switch mode, DualSense and DualShock 4. The value says whether the "
           "connected pad has a gyroscope.")
MENU_HINTS(hn_pad_layout, 0, "The Switch's order: A on the right, B at the bottom, X at the top, Y on the left.",
           "A at the bottom (action and talk), B on the right, X on the left, Y at the top.",
           "A at the bottom, B on the right; X and Y as on the Switch.",
           "Bindings of your own, from the Controls page.")
// (relink) Button prompts: the game's button font, made for the family and the bindings at the start (host/prompts.c)
const char* tx_button_prompts(int v) {
    static const char* t[] = {"Nintendo", "Xbox", "PlayStation"};
    return t[std::clamp(v, 0, 2)];
}
MENU_HINTS(hn_button_prompts, 0, "The game's own: the Switch's letters.",
           "A, B, X, Y, LB, RB, LT, RT, Menu and View, drawn by ReLink, the faded letters around the HUD's action button "
           "too. The ocarina's notes keep the Switch's.",
           "Cross, circle, square, triangle, L1, R1, L2, R2, Options and Create, drawn by ReLink, the faded letters around "
           "the HUD's action button too. The ocarina's notes keep the Switch's.")

// (relink) The keys ReLink's own keyboard actions can take (Menu key, Screenshot key, Fullscreen key), by index: kept in
// relink_options.ini as these indexes: the list must not change (None is its length). F1 opens the menu too while no other of these rows is F1 (keyboards without Insert)
const int kHotkeys[] = {VK_F1, VK_F2, VK_F3, VK_F4, VK_F5, VK_F6, VK_F7, VK_F8, VK_F9, VK_F10, VK_F11, VK_F12, VK_INSERT, VK_PAUSE,
                        VK_SCROLL};
constexpr int kHotkeyNone = (int)(sizeof kHotkeys / sizeof kHotkeys[0]);
const char* tx_hotkey(int v) {
    if (v < 0 || v >= kHotkeyNone) return "None";
    if (kHotkeys[v] == VK_INSERT) return "Insert";
    if (kHotkeys[v] == VK_PAUSE) return "Pause";
    if (kHotkeys[v] == VK_SCROLL) return "Scroll Lock";
    static char b[8];
    return snprintf(b, sizeof b, "F%d", kHotkeys[v] - VK_F1 + 1), b;
}
MENU_HINTS(hn_hotkey, 0, nullptr)

// (relink) HDR: what On does, and what it does not when Windows' HDR is off (hdr_output then stays false: the picture is SDR)
MENU_HINTS(hn_hdr, 0, "SDR, as before.",
           "The game draws in HDR10 at the display's peak brightness; its interface at the paper white (hdr_paper_white in "
           "relink_options.ini, 203 nits). With Windows' HDR off the picture stays SDR.")

Row rows[] = {
    // Graphics (the DLSS model only in a build with DLSS: row_shown)
    {TAB_GRAPHICS, "Quality preset", &opts.preset, 0, 5, 1, tx_preset,
     "Sets the scaling rows, frame generation and memory limits for a GPU with that much memory. The Upscaler stays yours.",
     hn_preset},
    {TAB_GRAPHICS, "Fullscreen", &opts.fullscreen, 0, 1, 1, tx_onoff,
     "Fills the whole screen with the game, without the window's border. The same as F11.", nullptr, false, true},
    {TAB_GRAPHICS, "HDR", &opts.hdr, 0, 1, 1, tx_onoff,
     "The game's own HDR picture, as on an HDR TV. Needs Windows' HDR on for this display. Applies after a restart.", hn_hdr,
     true, true},
    {TAB_GRAPHICS, "GPU device", &opts.gpu_device, 0, 8, 1, tx_gpu,
     "Which graphics card the game runs on. Applies after a restart.", hn_gpu, true, true},
    {TAB_GRAPHICS, "Frame generation", &opts.framegen, 0, 8, 1, tx_framegen,
     "Makes frames between the game's own, for smoother motion, at a little input delay. Fast motion or camera cuts can "
     "show artifacts.", hn_framegen},
    {TAB_GRAPHICS, "Frame pacing", &opts.pace, 0, 1, 1, tx_onoff, "When each frame is shown.", hn_pace},
    {TAB_GRAPHICS, "V-Sync", &opts.vsync, 0, 1, 1, tx_onoff,   // (relink: vk_main.cpp make_swapchain)
     "Shows each frame in step with the display's refresh, so the picture never tears. Applies at once.", hn_vsync, false, true},
    {TAB_GRAPHICS, "Low latency", &opts.reflex, 0, 2, 1, tx_reflex,
     "NVIDIA Reflex / AMD Anti-Lag 2: starts each frame as late as it can, for less input delay. Not used with frame generation.", hn_reflex},
    // (the picture's way, top to bottom: the output's size, what fills it from the 3D, the 3D's size, then the window)
    {TAB_GRAPHICS, "Output resolution", &opts.output_res, 0, kLastOutput, 1, tx_output_res,
     "The game's whole frame, HUD included, before it is scaled to the window. Docked only. Applies after a restart.",
     hn_output_res, true},
    {TAB_GRAPHICS, "Upscaler", &opts.dlss, 0, ur::kUpscalers - 1, 1, tx_upscaler, "What scales the 3D up to the Output resolution.",
     hn_upscaler},
    {TAB_GRAPHICS, "DLSS model", &opts.dlss_model, 0, ur::kModels - 1, 1, tx_dlss_model,
     "Which of NVIDIA's DLSS networks draws the picture.", hn_dlss_model, false, true},
    {TAB_GRAPHICS, "FSR version", &opts.fsr_version, 0, 1, 1, tx_fsr_version,
     "Which of AMD's FSR draws the picture: FSR 4, its network, where this GPU runs it, or FSR 3.1.", hn_fsr_version, false, true},
    {TAB_GRAPHICS, "Render resolution", &opts.resolution, 0, ur::kValues - 1, 1, tx_resolution,
     "The share of the output the 3D is drawn at; the Upscaler fills in the rest.", hn_resolution},
    {TAB_GRAPHICS, "Window scaling filter", &opts.upscale, 0, 2, 1, tx_filter,
     "How the finished picture is scaled to the window, when its size differs from the Output resolution.", hn_filter},
    {TAB_GRAPHICS, "Sharpness", &opts.sharpness, 0, 100, 10, tx_percent,
     "Sharpens the picture as it goes to the window, with any filter. Too much shows halos around edges.", hn_sharpness},
    {TAB_GRAPHICS, "Handheld profile", &opts.handheld, 0, 1, 1, tx_onoff,
     "Runs the game as if the console were undocked. Applies after a restart.", hn_handheld, true},
    {TAB_GRAPHICS, "Ultrawide", &opts.ultrawide, 0, 1, 1, tx_ultrawide,
     "The camera's shape on a screen wider than 16:9. Applies after a restart.", hn_ultrawide, true},
    {TAB_GRAPHICS, "Field of view", &opts.fov, 70, 150, 5, tx_percent,
     "How wide the camera sees, in percent of the game's own angle. Above 100% shows more around Link.", nullptr, false, true},
    {TAB_GRAPHICS, "Robust buffers", &opts.robust, 0, 2, 1, tx_robust,
     "Bounds-checked buffer reads: a read past a buffer's end is dropped instead of stopping the GPU. Applies after a "
     "restart.", hn_robust, true},

    // Gameplay (7 rows here; Quit game and Cheats at the end of the table)
    {TAB_GAMEPLAY, "FPS counter", &opts.show_fps, 0, 1, 1, tx_onoff, "Frames per second in the top right corner.", hn_fps,
     false, true},
    {TAB_GAMEPLAY, "Game speed", &opts.realtime, 0, 1, 1, tx_speed, "How the game keeps time when it runs below 60 fps.",
     hn_speed},
    {TAB_GAMEPLAY, "Text speed", &opts.text_speed, 0, 3, 1, tx_text_speed,
     "How fast the letters of dialogue and other messages appear.", hn_text_speed},
    {TAB_GAMEPLAY, "Volume", &opts.volume, 0, 100, 5, tx_percent,   // (relink: host/nn_audio.c)
     "The game's sound: music, effects and voices together. 0% mutes it.", nullptr, false, true},
    {TAB_GAMEPLAY, "Audio output", &opts.audio_output, 0, 2, 1, tx_audio_output,
     "The speakers the game mixes its sound for. Applies after a restart.", hn_audio_output, true, true},
    {TAB_GAMEPLAY, "Microphone", &opts.mic, 0, 2 + kMaxMics, 1, tx_mic,
     "What the ocarina hears when the game's microphone option is on (playing an instrument, or humming).", hn_mic, false,
     true},
    {TAB_GAMEPLAY, "Item wheel", &opts.item_wheel, 0, 1, 1, tx_onoff,   // (relink: host/oot_wheel.c, vk_wheel.cpp)
     "Hold R or L for a wheel of your items: point at one with either stick and let go to equip it.", hn_item_wheel},
    {TAB_GAMEPLAY, "Cycle items", &opts.cycle_items, 0, 4, 1, tx_cycle_items,
     "Changes the item on ZR with a tap of R or L, within its group. Holding R or L still opens the row.", hn_cycle_items},
    // (relink) the keyboard's own keys of ReLink: kHotkeys
    {TAB_GAMEPLAY, "Menu key", &opts.key_menu, 0, kHotkeyNone - 1, 1, tx_hotkey,
     "The key that opens and closes this menu. A controller's Home button, or Select and Start together, opens it too.",
     hn_hotkey, false, true},
    {TAB_GAMEPLAY, "Screenshot key", &opts.key_shot, 0, kHotkeyNone, 1, tx_hotkey,
     "Saves a picture of the game, without this menu, in the game folder's screenshots folder.", hn_hotkey, false, true},
    {TAB_GAMEPLAY, "Fullscreen key", &opts.key_full, 0, kHotkeyNone, 1, tx_hotkey,
     "Switches between the window and fullscreen. Alt+Enter does too.", hn_hotkey, false, true},

    // Controls (17 rows)
    {TAB_CONTROLS, "Controller in background", &opts.pad_background, 0, 1, 1, tx_onoff,
     "Whether the controller plays the game while another window has the focus.", hn_background},
    {TAB_CONTROLS, "Mouse camera", &opts.mouse_native, 0, 1, 1, tx_mousecam, "How the captured mouse turns the camera.",
     hn_mousecam, false, true},
    {TAB_CONTROLS, "Mouse sensitivity", &opts.mouse_sens, 10, 500, 10, tx_sens,
     "How far the camera turns for a mouse movement, with either Mouse camera: at 1.00 native look turns 0.1 degree a "
     "count. The - and = keys change it in game too.", nullptr, false, true},
    {TAB_CONTROLS, "Invert mouse vertical", &opts.mouse_invert, 0, 1, 1, tx_onoff, "Swaps up and down for the mouse.",
     hn_mouse_invert, false, true},
    {TAB_CONTROLS, "Invert camera X", &opts.invert_cam_x, 0, 1, 1, tx_onoff,
     "Inverts horizontal look (left/right) on the right analog stick.", nullptr, false, true},
    {TAB_CONTROLS, "Invert camera Y", &opts.invert_cam_y, 0, 1, 1, tx_onoff,
     "Inverts vertical look (up/down) on the right analog stick.", nullptr, false, true},
    {TAB_CONTROLS, "Rumble strength", &opts.rumble, 0, 150, 25, tx_rumble,
     "How strongly the controller rumbles, in percent of the game's rumble. Off turns it off.", nullptr, false, true},
    {TAB_CONTROLS, "Left stick dead zone", &opts.deadzone_l, 0, 50, 2, tx_percent,
     "How far the left stick must move before Link walks. Raise it if Link moves on his own (stick drift).", nullptr,
     false, true},
    {TAB_CONTROLS, "Right stick dead zone", &opts.deadzone_r, 0, 50, 2, tx_percent,
     "How far the right stick must move before the camera turns. Raise it if the camera drifts on its own.", nullptr,
     false, true},
    {TAB_CONTROLS, "Motion controls", &opts.gyro, 0, 1, 1, tx_gyro,
     "Aiming and looking with the controller's gyroscope where the game uses motion, as on the Switch.", hn_gyro},
    {TAB_CONTROLS, "Motion sensitivity X", &opts.gyro_sens_x, 10, 300, 10, tx_percent,
     "How far the aim moves left and right for a given turn of the controller. 100% is the console's own.", nullptr, false,
     true},
    {TAB_CONTROLS, "Motion sensitivity Y", &opts.gyro_sens_y, 10, 300, 10, tx_percent,
     "How far the aim moves up and down for a given tilt of the controller. 100% is the console's own.", nullptr, false,
     true},
    {TAB_CONTROLS, "Invert motion X", &opts.gyro_inv_x, 0, 1, 1, tx_onoff, "Turns the motion aim's left and right around.",
     nullptr, false, true},
    {TAB_CONTROLS, "Invert motion Y", &opts.gyro_inv_y, 0, 1, 1, tx_onoff, "Turns the motion aim's up and down around.",
     nullptr, false, true},
    {TAB_CONTROLS, "Layout preset", &opts.pad_layout, 0, 3, 1, tx_pad_layout,
     "Which face button is which. Binding one yourself on the Controls page makes it Custom.", hn_pad_layout},
    {TAB_CONTROLS, "Button prompts", &opts.button_prompts, 0, 2, 1, tx_button_prompts,   // (relink: host/prompts.c)
     "The buttons the game's prompts show. They follow the Layout preset and your own bindings. Applies the next time the "
     "game starts.", hn_button_prompts, true, true},
    {TAB_CONTROLS, "Controls", nullptr, 0, 0, 0, nullptr,   // (relink: opens its page)
     "Which gamepad button and which keys press each of the Switch's buttons."},
    {TAB_GAMEPLAY, "Quit game", nullptr, 0, 0, 0, nullptr,   // (relink, issue 32: asks first, then closes the game)
     "Closes the game, as closing its window does. Your settings are kept; progress since your last save is not."},
    {TAB_GAMEPLAY, "Cheats", nullptr, 0, 0, 0, nullptr,
     "Open the private gameplay tools."},
};
constexpr int kCheatsRow = sizeof rows / sizeof rows[0] - 1;
constexpr int kQuitRow = kCheatsRow - 1;
bool cheats_enabled = false;
std::atomic<int> protect_hearts{-1};
// -1 disables the shortcut, 0 enables it on the ground, and 1 means flying.
std::atomic<int> player_flight{-1};
std::atomic<int> free_camera{0};
std::atomic<int> cheat_action{0};
int clock_hour = 12;
int pouch_item = 0;
int pouch_delta = 1;
int cheat_selected = 0;
constexpr const char* pouch_items[] = {"Deku Seeds", "Deku Nuts", "Deku Sticks", "Bombs", "Arrows"};
constexpr int kPouchItems = sizeof pouch_items / sizeof pouch_items[0];
// (teleport list: generated by work/oot-teleport-tools/write_destinations.py)
// (the order of oot_destinations in host/hos_main.c; Enter sends 1000 + the index)
constexpr const char* teleport_names[] = {
    "Kokiri Forest",
    "Hyrule Field",
    "Kakariko Village",
    "Market",
    "Hyrule Castle",
    "Lost Woods",
    "Sacred Forest Meadow",
    "Lake Hylia",
    "Graveyard",
    "Death Mountain Trail",
    "Death Mountain Crater",
    "Goron City",
    "Zora's River",
    "Zora's Domain",
    "Zora's Fountain",
    "Gerudo Valley",
    "Gerudo Fortress",
    "Haunted Wasteland",
    "Desert Colossus",
    "Inside the Deku Tree",
    "Dodongo's Cavern",
    "Jabu-Jabu's Belly",
    "Forest Temple",
    "Fire Temple",
    "Water Temple",
    "Shadow Temple",
    "Spirit Temple",
    "Bottom of the Well",
    "Ice Cavern",
    "Gerudo Training Ground",
    "Ganon's Castle"};
constexpr const char* teleport_notes[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "adult only", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
struct TeleportSection { const char* name; int first; };
constexpr TeleportSection teleport_sections[] = {{"Overworld", 0}, {"Dungeons", 19}};
constexpr int kTeleportSections = sizeof teleport_sections / sizeof teleport_sections[0];
// (end of the generated teleport list)
constexpr int kTeleports = sizeof teleport_names / sizeof teleport_names[0];
int teleport_choice = 0;
int teleport_scroll = 0;   // (the first list line shown on the Teleport page)
constexpr int kTeleportLines = 20;   // (list lines the Teleport page shows at once)
// the Inventory page (gpu/vk/oot_cheat_items.h): items and their section headings, counts from the game side
#define OOT_MENU_ITEM_LABEL(label, name, kind) label,
#define OOT_MENU_ITEM_KIND(label, name, kind) kind,
#define OOT_MENU_NO_SECTION(name)
constexpr const char* item_names[] = {OOT_CHEAT_ITEMS(OOT_MENU_NO_SECTION, OOT_MENU_ITEM_LABEL)};
constexpr unsigned char item_kinds[] = {OOT_CHEAT_ITEMS(OOT_MENU_NO_SECTION, OOT_MENU_ITEM_KIND)};
constexpr int kItems = sizeof item_names / sizeof item_names[0];
// (each section's first item: a counter that a section heading reads and an item bumps)
constexpr int item_section_count() {
    int n = 0;
#define OOT_MENU_COUNT_SECTION(title) n++;
#define OOT_MENU_SKIP_ITEM(label, name, kind)
    OOT_CHEAT_ITEMS(OOT_MENU_COUNT_SECTION, OOT_MENU_SKIP_ITEM)
    return n;
}
constexpr int kItemSections = item_section_count();
struct ItemSections { TeleportSection s[kItemSections]; };
constexpr ItemSections make_item_sections() {
    ItemSections r{};
    int section = 0, item = 0;
#define OOT_MENU_ADD_SECTION(title) r.s[section].name = title, r.s[section].first = item, section++;
#define OOT_MENU_ADD_ITEM(label, name, kind) item++;
    OOT_CHEAT_ITEMS(OOT_MENU_ADD_SECTION, OOT_MENU_ADD_ITEM)
    return r;
}
constexpr ItemSections item_sections = make_item_sections();
std::atomic<int> item_counts[kItems];   // (-1 until the game side reports: OOT_CHEAT_ITEM_BASE + i)
std::atomic<int> link_age{-1};          // (relink) 0 child, 1 adult, -1 unknown (OOT_CHEAT_AGE_ID: the item wheel's pages)
int item_choice = 0, item_scroll = 0;
constexpr int kCheatRows = 12;
constexpr int kTravelCheat = 11;   // (relink) the Cheats page's Travel row: opens the Travel page, page 6
constexpr int kRows = sizeof rows / sizeof rows[0];
constexpr int kMenuWidth = menu_hit::kWidth;   // (at a scale of 1: 720 lines)

int cur_tab = TAB_GRAPHICS;
int tab_selected[TAB_COUNT] = {0, 0, 0};

// ---- the rows that take a restart: their values at the start (menu_init, the GPU's at menu_set_gpu_count) ----
namespace mr = menu_restart;
mr::State restart_state;
int restart_slot[kRows];             // (a row's slot in restart_state, -1 none)
const char* restart_names[mr::kMax];
void restart_init() {
    static bool done = false;
    if (done) return;
    done = true;
    for (int i = 0; i < kRows; i++) {
        restart_slot[i] = rows[i].restart && rows[i].value ? restart_state.add(*rows[i].value) : -1;
        if (restart_slot[i] >= 0) restart_names[restart_slot[i]] = rows[i].name;
    }
}
int restart_slot_of(const Row& r) { return restart_slot[&r - rows]; }
void restart_values(int* now) {
    for (int i = 0; i < kRows; i++)
        if (restart_slot[i] >= 0) now[restart_slot[i]] = *rows[i].value;
}
/// The rows whose value is not the one the game runs with (a bit a slot)
unsigned restart_pending() {
    int now[mr::kMax];
    restart_values(now);
    return restart_state.pending(now);
}
/// Whether closing the menu now asks to restart: something pending, not declined with Later, and a window to close
bool restart_ask() {
    int now[mr::kMax];
    restart_values(now);
    return app_window() && restart_state.ask(now);
}
std::string restart_list(const char* last) {
    return mr::names(restart_names, restart_state.n, restart_pending(), ", ", last);
}
int ask_sel = 0;   // (the restart question's choice: 0 Later, 1 Restart now)
bool ask_quit = false;   // (page 5 asks Quit game's question instead: 0 Back, 1 Quit)
/// "Restart needed: " and the pending rows, on one line of width: the last ones as "and N more" when they do not fit
std::string restart_line(HDC dc, int width) {
    const unsigned all = restart_pending();
    std::string line;
    for (int keep = restart_state.n; keep >= 0; keep--) {   // (the first `keep` pending rows by name)
        unsigned shown = 0;
        int named = 0, more = 0;
        for (int i = 0; i < restart_state.n; i++)
            if ((all >> i) & 1) (named < keep ? (shown |= 1u << i, named++) : more++);
        line = "Restart needed: " + mr::names(restart_names, restart_state.n, shown);
        if (more) line += (named ? " and " : "") + std::to_string(more) + (named ? " more" : more > 1 ? " settings" : " setting");
        SIZE sz{};
        GetTextExtentPoint32A(dc, line.c_str(), (int)line.size(), &sz);
        if (sz.cx <= width) break;
    }
    return line;
}

/// A row on its tab: the Cheats one only with cheats on, the DLSS model only in a build with DLSS (the Upscaler in every one)
bool row_shown(int i) {
    return (i != kCheatsRow || cheats_enabled) && (rows[i].value != &opts.dlss || ur::upscaler_shown()) &&
           (rows[i].value != &opts.dlss_model || ur::model_shown(dlss_built()));
}
int tab_row_count(int t) {
    int c = 0;
    for (int i = 0; i < kRows; i++) if (rows[i].tab == (MenuTab)t && row_shown(i)) c++;
    return c;
}

/// (relink) The small heading a row sits under on its tab, and that heading's place on the tab ("": the tab's first rows,
/// with no heading). A tab shows its rows heading by heading, in the table's order within each
const char* row_group(int i, int* place) {
    static const struct { const char* name; const char* group; int place; } groups[] = {
        {"Fullscreen", "DISPLAY", 1}, {"HDR", "DISPLAY", 1}, {"GPU device", "DISPLAY", 1}, {"Output resolution", "DISPLAY", 1},
        {"Handheld profile", "DISPLAY", 1}, {"Ultrawide", "DISPLAY", 1}, {"Field of view", "DISPLAY", 1},
        {"Upscaler", "UPSCALING", 2}, {"DLSS model", "UPSCALING", 2}, {"FSR version", "UPSCALING", 2},
        {"Render resolution", "UPSCALING", 2},
        {"Window scaling filter", "UPSCALING", 2}, {"Sharpness", "UPSCALING", 2},
        {"Frame generation", "SMOOTHNESS", 3}, {"Frame pacing", "SMOOTHNESS", 3}, {"V-Sync", "SMOOTHNESS", 3}, {"Low latency", "SMOOTHNESS", 3},
        {"Robust buffers", "ADVANCED", 4},
        {"Volume", "SOUND", 1}, {"Audio output", "SOUND", 1}, {"Microphone", "SOUND", 1},
        {"Item wheel", "ITEMS", 2}, {"Cycle items", "ITEMS", 2},
        {"Menu key", "KEYBOARD", 3}, {"Screenshot key", "KEYBOARD", 3}, {"Fullscreen key", "KEYBOARD", 3},
        {"Quit game", "MORE", 4}, {"Cheats", "MORE", 4},
        {"Controller in background", "PAD", 0}, {"Rumble strength", "PAD", 0}, {"Left stick dead zone", "PAD", 0},
        {"Right stick dead zone", "PAD", 0}, {"Layout preset", "PAD", 0}, {"Button prompts", "PAD", 0}, {"Controls", "PAD", 0},
        {"Mouse camera", "MOUSE", 1}, {"Mouse sensitivity", "MOUSE", 1}, {"Invert mouse vertical", "MOUSE", 1},
        {"Invert camera X", "CAMERA", 2}, {"Invert camera Y", "CAMERA", 2},
        {"Motion controls", "MOTION", 3}, {"Motion sensitivity X", "MOTION", 3}, {"Motion sensitivity Y", "MOTION", 3},
        {"Invert motion X", "MOTION", 3}, {"Invert motion Y", "MOTION", 3},
    };
    for (const auto& g : groups)
        if (!strcmp(g.name, rows[i].name)) return *place = g.place, g.group;
    return *place = 0, "";
}
/// The rows in the order the tabs show them: by tab, then by heading, then as the table has them
const int* display_order() {
    static int order[kRows];
    static const bool made = [] {
        for (int i = 0; i < kRows; i++) order[i] = i;
        std::stable_sort(order, order + kRows, [](int a, int b) {
            int pa = 0, pb = 0;
            row_group(a, &pa), row_group(b, &pb);
            return rows[a].tab != rows[b].tab ? rows[a].tab < rows[b].tab : pa < pb;
        });
        return true;
    }();
    (void)made;
    return order;
}
int tab_row_to_global(int t, int local_idx) {
    int count = 0;
    for (int k = 0; k < kRows; k++) {
        const int i = display_order()[k];
        if (rows[i].tab == (MenuTab)t && row_shown(i)) {
            if (count == local_idx) return i;
            count++;
        }
    }
    return 0;
}

std::atomic<bool> open_now{false};
bool menu_refresh = false;
// (relink) the mouse (menu_tick): the tab under the cursor (-1 none); the slider held with the left button (its row in
// the tab, -1 none)
int hover_tab = -1;
int drag_row = -1;
bool mouse_read = false;   // (the menu has read the mouse since it opened)
bool is_slider(const Row& r) { return r.value == &opts.sharpness || r.value == &opts.mouse_sens || r.value == &opts.gyro_sens_x || r.value == &opts.gyro_sens_y || r.value == &opts.volume; }
/// (a scale: the arrows, a click and the wheel stop at its ends; Rumble strength would go from Off to its strongest)
bool stops_at_ends(const Row& r) { return r.value == &opts.mouse_sens || r.value == &opts.rumble || r.value == &opts.gyro_sens_x || r.value == &opts.gyro_sens_y || r.value == &opts.deadzone_l || r.value == &opts.deadzone_r || r.value == &opts.fov || r.value == &opts.volume; }
/// (the ini's Rumble strength on the menu's steps: 0 to 150, the nearest 25)
int rumble_step(int v) { return (std::min(std::max(v, 0), 150) + 12) / 25 * 25; }
std::string ini_path() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string dir(exe, n);
    return dir.substr(0, dir.find_last_of("\\/") + 1) + "relink_options.ini";
}
const char* ini_key(const Row& r) {
    if (r.value == &opts.output_res) return "output_res";   // (host/nn_applet.c reads it at the start)
    // (rows renamed since their values were first saved keep their keys: an options file from before still loads)
    if (r.value == &opts.resolution) return "resolution";
    if (r.value == &opts.upscale) return "upscaling";
    if (r.value == &opts.dlss) return "dlss";   // (as opts.dlss and RX_VK_DLSS)
    static char k[32];
    size_t n = 0;
    for (const char* p = r.name; *p && n + 1 < sizeof k; p++) k[n++] = *p == ' ' ? '_' : (char)tolower(*p);
    k[n] = 0;
    return k;
}
// (relink) the controls, kept by the host (host/nn_hid.c reads them back from here): pad_<button>, key_<button>
const char* const bind_names[HOS_KEY_N] = {"a", "b", "x", "y", "ls", "rs", "l", "r", "zl", "zr", "plus", "minus",
                                           "left", "up", "right", "down", "move_up", "move_down", "move_left",
                                           "move_right", "cam_up", "cam_down", "cam_left", "cam_right"};
bool can_bind() { return host && host->binding && host->bind && host->pad_raw; }
// (relink, issue 3) a key is down: the host's answer (Windows' state of the keyboard or, on a PC where that says nothing, the
// window's own key messages: host/display.c). A player there could open this menu and not give a control a new key. Asked
// through take_key_press with 0x10000 added to the key code (host/rx_backend.c MKWVK_KEY_HELD says why not a call of its own)
bool key_is_down(int vk) {
    return host && host->take_key_press ? host->take_key_press(0x10000 | (vk & 0xFF)) != 0 : (GetAsyncKeyState(vk) & 0x8000) != 0;
}
const char* tx_pad_layout(int v) {
    static const char* t[] = {"Nintendo (Def)", "Xbox / PC", "Swap Face", "Custom"};
    return t[std::min(std::max(v, 0), 3)];
}
void apply_pad_layout(int layout) {
    if (!can_bind()) return;
    if (layout == 0) { // Nintendo (Default: Bottom=B, Right=A, Top=X, Left=Y)
        host->bind(HOS_BIND_PAD, HOS_SW_A, 0, HOS_PAD_EAST);
        host->bind(HOS_BIND_PAD, HOS_SW_B, 0, HOS_PAD_SOUTH);
        host->bind(HOS_BIND_PAD, HOS_SW_X, 0, HOS_PAD_NORTH);
        host->bind(HOS_BIND_PAD, HOS_SW_Y, 0, HOS_PAD_WEST);
    } else if (layout == 1) { // Xbox / PC Standard (Bottom=A, Right=B, Left=X, Top=Y)
        host->bind(HOS_BIND_PAD, HOS_SW_A, 0, HOS_PAD_SOUTH);
        host->bind(HOS_BIND_PAD, HOS_SW_B, 0, HOS_PAD_EAST);
        host->bind(HOS_BIND_PAD, HOS_SW_X, 0, HOS_PAD_WEST);
        host->bind(HOS_BIND_PAD, HOS_SW_Y, 0, HOS_PAD_NORTH);
    } else if (layout == 2) { // Swap Face Buttons (Bottom=A, Right=B, Top=X, Left=Y)
        host->bind(HOS_BIND_PAD, HOS_SW_A, 0, HOS_PAD_SOUTH);
        host->bind(HOS_BIND_PAD, HOS_SW_B, 0, HOS_PAD_EAST);
        host->bind(HOS_BIND_PAD, HOS_SW_X, 0, HOS_PAD_NORTH);
        host->bind(HOS_BIND_PAD, HOS_SW_Y, 0, HOS_PAD_WEST);
    }
}
int detect_pad_layout() {
    if (!can_bind()) return opts.pad_layout;
    const int a = host->binding(HOS_BIND_PAD, HOS_SW_A, 0);
    const int b = host->binding(HOS_BIND_PAD, HOS_SW_B, 0);
    const int x = host->binding(HOS_BIND_PAD, HOS_SW_X, 0);
    const int y = host->binding(HOS_BIND_PAD, HOS_SW_Y, 0);
    if (a == HOS_PAD_EAST && b == HOS_PAD_SOUTH && x == HOS_PAD_NORTH && y == HOS_PAD_WEST) return 0;
    if (a == HOS_PAD_SOUTH && b == HOS_PAD_EAST && x == HOS_PAD_WEST && y == HOS_PAD_NORTH) return 1;
    if (a == HOS_PAD_SOUTH && b == HOS_PAD_EAST && x == HOS_PAD_NORTH && y == HOS_PAD_WEST) return 2;
    return 3;
}
// (relink) the row's choice to the host: off, the default, or a microphone by its name (and its place in the host's list,
// -1 while it is unplugged: of two with one name, the one chosen). Unplugged, the name is the kept one, cut: the host takes
// a name that starts with it
void mic_tell_host() {
    if (!can_mic()) return;
    if (opts.mic <= 1) host->mic_select(opts.mic <= 0 ? 0 : 1, "", -1);
    else if (mic_missing && opts.mic == 2 + mic_listed) host->mic_select(2, mic_want.c_str(), -1);   // (the kept name, cut)
    else host->mic_select(2, host->mic_name(opts.mic - 2), opts.mic - 2);   // (its whole name: the ini's cut is only the ini's)
}
void mic_row_ends() {
    for (Row& r : rows)
        if (r.value == &opts.mic) r.hi = 1 + mic_listed + (mic_missing ? 1 : 0);
}
// (relink) the microphones listed again (one plugged in since shows up), the row on mic_want's: the one at the row's place
// when it still has that name (two microphones of one name: the second stays chosen), else the first of that name, else
// its "(not connected)" entry while it is unplugged
void refresh_mics() {
    if (!can_mic()) {
        if (opts.mic >= 2) opts.mic = 1;
        return;
    }
    mic_listed = std::min(host->mic_count(), kMaxMics);
    int at = -1;
    if (!mic_want.empty()) {
        if (opts.mic >= 2 && opts.mic - 2 < mic_listed && mic_key(host->mic_name(opts.mic - 2)) == mic_want) at = opts.mic - 2;
        for (int i = 0; i < mic_listed && at < 0; i++)
            if (mic_key(host->mic_name(i)) == mic_want) at = i;
        opts.mic = at >= 0 ? at + 2 : 2 + mic_listed;
    } else if (opts.mic >= 2) {
        opts.mic = 1;
    }
    mic_missing = !mic_want.empty() && at < 0;
    mic_row_ends();
    mic_tell_host();
}
// (the row changed by the player: the choice is now this one, by its name; the "(not connected)" entry is kept only while
// it is the one chosen)
void mic_chosen() {
    if (opts.mic >= 2 && !(mic_missing && opts.mic == 2 + mic_listed)) {
        mic_want = mic_key(host->mic_name(opts.mic - 2)), mic_missing = false;
    } else if (opts.mic < 2) {
        mic_want.clear(), mic_missing = false;
    }
    mic_row_ends();
    mic_tell_host();
}
void save() {
    // (relink) the lines that are not the menu's are kept: the controls, pad_ and key_ (host/nn_hid.c), and every other key
    // this file holds without a row here, the launcher's frame_rate among them (Discord, 18/11: "Fps cap get reverted to 60
    // every time I close the game": the game's save dropped it, and the launcher read the default back)
    std::string kept;
    std::vector<std::pair<std::string, std::string>> others;   // (each key and its line: the last line of a key kept)
    if (FILE* f = fopen(ini_path().c_str(), "r")) {
        char line[512];
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "pad_", 4) || !strncmp(line, "key_", 4)) {
                kept += line;
                continue;
            }
            const char* eq = strchr(line, '=');
            if (!eq || line[0] == '#') continue;
            const std::string key(line, (size_t)(eq - line));
            bool mine = key == "microphone_device" || key == "hdr_paper_white" || key == "options_version";
            for (const Row& r : rows) mine |= r.value && key == ini_key(r);
            if (mine) continue;
            std::string text(line);
            if (text.back() != '\n') text += '\n';
            std::erase_if(others, [&](const auto& o) { return o.first == key; });
            others.emplace_back(key, std::move(text));
        }
        fclose(f);
    }
    if (FILE* f = fopen(ini_path().c_str(), "w")) {
        for (const Row& r : rows)
            if (r.value) fprintf(f, "%s=%d\n", ini_key(r), *r.value);
        if (!mic_want.empty()) fprintf(f, "microphone_device=%s\n", mic_want.c_str());   // (relink: also while it is unplugged)
        fprintf(f, "hdr_paper_white=%d\n", opts.hdr_paper_white);   // (relink: HDR, no row of its own)
        // (2: game_speed below 2 was only ever the old default, see menu_init; 3: the mouse's buttons are bindings, see
        // host/nn_hid.c binds_load)
        fprintf(f, "options_version=3\n");
        for (const auto& o : others) fputs(o.second.c_str(), f);
        if (!can_bind()) fputs(kept.c_str(), f);
        for (int i = 0; i < HOS_KEY_N && can_bind(); i++) {   // (the keys-only controls, HOS_KEY_*, have no pad_ line)
            if (i < HOS_SW_N) fprintf(f, "pad_%s=%d\n", bind_names[i], host->binding(HOS_BIND_PAD, i, 0));
            fprintf(f, "key_%s=%d,%d\n", bind_names[i], host->binding(HOS_BIND_KEY, i, 0), host->binding(HOS_BIND_KEY, i, 1));
        }
        fclose(f);
    }
}

/// A row's new value v, as the arrows set it (dir: 1 up, -1 down, for the rows that skip or wrap), saved unless !keep
/// (a slider's drag: saved once it is let go)
void set_row(Row& r, int v, int dir, bool keep = true) {
    // (nothing else to choose: no change, no save. A saved choice that cannot run here is kept)
    if (row_fixed(r.value)) return;
    // (the scaling rows: only what can run here; the Render resolution row through the Upscaler's own steps)
    if (r.value == &opts.dlss) v = ur::step_upscaler(upscaler(), dir, dlss_ok(), fsr_ok());
    if (r.value == &opts.resolution) v = ur::step(upscaler(), *r.value, dir);
    if (r.value == &opts.output_res) {   // (smallest to largest: kOutputOrder)
        v = step_output_choice(*r.value, dir);
        if (v == *r.value) return;
    }
    if (r.value == &opts.framegen && v == 1) v = dir > 0 ? 2 : 0;   // (off, 2x ... 8x)
    if (stops_at_ends(r)) v = std::min(std::max(v, r.lo), r.hi);   // (a scale: it stops at its ends)
    if (r.value == &opts.pad_layout) {
        if (v > 2) v = dir > 0 ? 0 : 2;
        if (v < 0) v = 2;
        *r.value = v;
        apply_pad_layout(*r.value);
    } else if (r.value == &opts.preset) {
        *r.value = v > r.hi ? r.lo : v < r.lo ? r.hi : v;
        apply_preset(*r.value);
    } else {
        *r.value = v > r.hi ? r.lo : v < r.lo ? r.hi : v;
        // (another Upscaler: the Render resolution moves to the nearest of its steps, DLSS's modes or the game's)
        if (r.value == &opts.dlss) opts.resolution = ur::snap(upscaler(), opts.resolution);
        if (!r.preset_exempt) {   // (the controls, the window and the counter are not the preset's)
            opts.preset = 0; // Custom
            sync_preset_budgets(0);
        }
    }
    if (r.value == &opts.mic && can_mic()) mic_chosen();   // (relink)
    {   // (a row that takes a restart, or a preset that sets one: asked again at the next close)
        int now[mr::kMax];
        restart_values(now);
        restart_state.note(now);
    }
    if (keep) save();
}

// ---- (relink) the Controls page: what presses each of the Switch's buttons, a gamepad input and up to two keys ----
struct Control {
    int button;   // HOS_SW_*, or HOS_KEY_* (keys only)
    const char* name;
};
const Control controls[] = {{HOS_SW_A, "A"}, {HOS_SW_B, "B"}, {HOS_SW_X, "X"}, {HOS_SW_Y, "Y"}, {HOS_SW_L, "L"},
                            {HOS_SW_R, "R"}, {HOS_SW_ZL, "ZL"}, {HOS_SW_ZR, "ZR"}, {HOS_SW_PLUS, "Plus"},
                            {HOS_SW_MINUS, "Minus"}, {HOS_SW_LS, "Left stick press"}, {HOS_SW_RS, "Right stick press"},
                            {HOS_SW_UP, "D-pad up"}, {HOS_SW_DOWN, "D-pad down"}, {HOS_SW_LEFT, "D-pad left"},
                            {HOS_SW_RIGHT, "D-pad right"}, {HOS_KEY_MOVE_UP, "Move forward"},
                            {HOS_KEY_MOVE_DOWN, "Move back"}, {HOS_KEY_MOVE_LEFT, "Move left"},
                            {HOS_KEY_MOVE_RIGHT, "Move right"}, {HOS_KEY_CAM_UP, "Camera up"},
                            {HOS_KEY_CAM_DOWN, "Camera down"}, {HOS_KEY_CAM_LEFT, "Camera left"},
                            {HOS_KEY_CAM_RIGHT, "Camera right"}};
constexpr int kControls = sizeof controls / sizeof controls[0];   // (then "Reset to defaults")
bool keys_only(int i) { return i < kControls && controls[i].button >= HOS_SW_N; }   // (no gamepad input for those)
const char* pad_name(int p) {
    static const char* t[HOS_PAD_N] = {"Bottom button", "Right button", "Left button", "Top button", "Select / Back",
                                       "Home", "Start", "Left stick press", "Right stick press", "L1 / LB", "R1 / RB",
                                       "D-pad up", "D-pad down", "D-pad left", "D-pad right", "L2 / LT", "R2 / RT"};
    return p >= 0 && p < HOS_PAD_N ? t[p] : "None";
}
const char* key_name(int vk) {
    static char b[16];
    if ((vk >= 'A' && vk <= 'Z') || (vk >= '0' && vk <= '9')) return b[0] = (char)vk, b[1] = 0, b;
    if (vk >= VK_F1 && vk <= VK_F12) return snprintf(b, sizeof b, "F%d", vk - VK_F1 + 1), b;
    if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD0 + 9) return snprintf(b, sizeof b, "Num %d", vk - VK_NUMPAD0), b;
    switch (vk) {
    case VK_SPACE: return "Space";
    case VK_SHIFT: return "Shift";
    case VK_CONTROL: return "Ctrl";
    case VK_MENU: return "Alt";
    case VK_TAB: return "Tab";
    case VK_RETURN: return "Enter";
    case VK_BACK: return "Backspace";
    case VK_CAPITAL: return "Caps Lock";
    case VK_UP: return "Up";
    case VK_DOWN: return "Down";
    case VK_LEFT: return "Left";
    case VK_RIGHT: return "Right";
    case VK_HOME: return "Home";
    case VK_END: return "End";
    case VK_PRIOR: return "Page Up";
    case VK_NEXT: return "Page Down";
    case VK_ADD: return "Num +";
    case VK_SUBTRACT: return "Num -";
    case VK_MULTIPLY: return "Num *";
    case VK_DIVIDE: return "Num /";
    case VK_DECIMAL: return "Num .";
    case VK_OEM_1: return ";";
    case VK_OEM_PLUS: return "=";
    case VK_OEM_COMMA: return ",";
    case VK_OEM_MINUS: return "-";
    case VK_OEM_PERIOD: return ".";
    case VK_OEM_2: return "/";
    case VK_OEM_3: return "`";
    case VK_OEM_4: return "[";
    case VK_OEM_5: return "\\";
    case VK_OEM_6: return "]";
    case VK_OEM_7: return "'";
    case VK_LBUTTON: return "Left click";
    case VK_RBUTTON: return "Right click";
    case VK_MBUTTON: return "Middle click";
    case 0x05: return "Mouse 4";   // (VK_XBUTTON1)
    case 0x06: return "Mouse 5";
    }
    return snprintf(b, sizeof b, "Key %d", vk), b;
}
// the keys a binding can take (Esc cancels, Insert and F1 are the menu's, Delete clears)
const std::vector<int>& bindable_keys() {
    static std::vector<int> k;
    if (k.empty()) {
        for (int c = 'A'; c <= 'Z'; c++) k.push_back(c);
        for (int c = '0'; c <= '9'; c++) k.push_back(c);
        for (int c = VK_F2; c <= VK_F12; c++) k.push_back(c);
        for (int c = VK_NUMPAD0; c <= VK_NUMPAD0 + 9; c++) k.push_back(c);
        for (int c : {VK_SPACE, VK_SHIFT, VK_CONTROL, VK_MENU, VK_TAB, VK_RETURN, VK_BACK, VK_CAPITAL, VK_UP, VK_DOWN,
                      VK_LEFT, VK_RIGHT, VK_HOME, VK_END, VK_PRIOR, VK_NEXT, VK_ADD, VK_SUBTRACT, VK_MULTIPLY,
                      VK_DIVIDE, VK_DECIMAL, VK_OEM_1, VK_OEM_PLUS, VK_OEM_COMMA, VK_OEM_MINUS, VK_OEM_PERIOD,
                      VK_OEM_2, VK_OEM_3, VK_OEM_4, VK_OEM_5, VK_OEM_6, VK_OEM_7, VK_LBUTTON, VK_RBUTTON, VK_MBUTTON,
                      0x05, 0x06})   // (the mouse's buttons too: they count while the mouse is captured)
            k.push_back(c);
    }
    return k;
}
int page = 0;          // 0 the options, 1 the controls, 2 the cheats, 3 the teleport list, 4 the inventory, 5 the restart question,
                       // 6 (relink) the Travel page
int csel = 0, ccol = 0;   // the controls' row (kControls: reset) and column (0 the gamepad, 1 the keyboard)
int listening = 0;     // waiting for the new input: 1 a gamepad's, 2 a key
bool listen_armed = false;   // (once what started it was let go)
int64_t listen_end = 0;
std::string keys_text(int button) {
    const int a = host->binding(HOS_BIND_KEY, button, 0), b = host->binding(HOS_BIND_KEY, button, 1);
    std::string t = a ? key_name(a) : "";
    if (b) t += std::string(a ? ", " : "") + key_name(b);
    return t.empty() ? "None" : t;
}

// ---- what is drawn: bitmaps of the menu and the counter, in the window's byte order ----
struct Bitmap {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> px;
    uint64_t version = 0;
    bool keyed = false;                   // (black is see-through: copied as the runs of pixels in between)
    std::vector<VkBufferImageCopy> runs;
};
Bitmap menu_bmp, fps_bmp, notice_bmp;   // (relink: notice_bmp, the Travel page's "picture saved" line)
bool menu_keyed = true;   // (relink: every page of the menu is a cartridge, its round corners see-through)
std::mutex bmp_mu;
int scale_of(uint32_t h) { return std::max(1u, h / 720); }

thread_local uint32_t* paint_bits = nullptr;   // (the bitmap being drawn, for a paint that writes pixels itself)
void gdi_draw(Bitmap& out, uint32_t w, uint32_t h, void (*paint)(HDC, uint32_t, uint32_t, int), int scale) {
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof bi.bmiHeader;
    bi.bmiHeader.biWidth = (LONG)w;
    bi.bmiHeader.biHeight = -(LONG)h;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    HDC dc = CreateCompatibleDC(nullptr);
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!bmp) { DeleteDC(dc); return; }
    HGDIOBJ old = SelectObject(dc, bmp);
    paint_bits = (uint32_t*)bits;   // (the start-up screen blends its logo in: vk_startup_paint.h)
    paint(dc, w, h, scale);
    paint_bits = nullptr;
    GdiFlush();
    if (const char* dir = &out == &menu_bmp ? getenv("RX_VK_MENU_DUMP") : nullptr) {   // (test runs: each drawing of the menu, as menu_<n>.bmp)
        static int n = 0;
        char path[1024];
        snprintf(path, sizeof path, "%s/menu_%03d.bmp", dir, n++);
        if (FILE* f = fopen(path, "wb")) {   // (the 14-byte file header and the 40-byte info header, little-endian: 32-bit, top-down)
            uint8_t hd[54] = {'B', 'M'};
            auto put = [&](int at, uint32_t v, int bytes) { for (int i = 0; i < bytes; i++) hd[at + i] = (uint8_t)(v >> (8 * i)); };
            put(2, 54 + w * h * 4, 4), put(10, 54, 4), put(14, 40, 4), put(18, w, 4), put(22, (uint32_t)-(int32_t)h, 4), put(26, 1, 2),
                put(28, 32, 2);
            fwrite(hd, 1, sizeof hd, f), fwrite(bits, 4, (size_t)w * h, f);
            fclose(f);
        }
    }
    const bool rgba = dev.swap_format == VK_FORMAT_R8G8B8A8_UNORM || dev.swap_format == VK_FORMAT_R8G8B8A8_SRGB;
    std::lock_guard lk(bmp_mu);
    if (&out == &menu_bmp) out.keyed = menu_keyed;   // (with its pixels, under the lock: menu_record reads both together)
    out.w = w, out.h = h;
    out.px.assign((const uint8_t*)bits, (const uint8_t*)bits + (size_t)w * h * 4);
    if (out.keyed) {
        out.runs.clear();
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w;) {
                const uint8_t* p = out.px.data() + ((size_t)y * w + x) * 4;
                if (!(p[0] | p[1] | p[2])) { x++; continue; }
                uint32_t e = x + 1;
                for (; e < w; e++) {
                    const uint8_t* q = out.px.data() + ((size_t)y * w + e) * 4;
                    if (!(q[0] | q[1] | q[2])) break;
                }
                VkBufferImageCopy c{};
                c.bufferOffset = ((uint64_t)y * w + x) * 4;
                c.bufferRowLength = w;
                c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                c.imageOffset = {(int32_t)x, (int32_t)y, 0};
                c.imageExtent = {e - x, 1, 1};
                out.runs.push_back(c);
                x = e;
            }
        }
    }
    if (hdr_output()) {   // (relink) the window's images are HDR10 words: the bitmap as such, at the paper white (hdr_out.h)
        hdr_out::bgra8_to_hdr10(out.px.data(), out.px.size() / 4, hdr_paper_white_nits());
    } else {
        for (size_t i = 0; i < out.px.size(); i += 4) {
            if (rgba) std::swap(out.px[i], out.px[i + 2]);
            out.px[i + 3] = 255;
        }
    }
    out.version++;
    SelectObject(dc, old);
    DeleteObject(bmp);
    DeleteDC(dc);
}
HFONT font(int px, int weight) {
    // (relink) the game's own typeface where it is (vk_startup_paint.h startup_fonts): semibold as Rubik Medium, bold as Rubik's
    const bool own = startup_fonts();
    const char* name = !own ? "Segoe UI" : weight >= FW_BOLD ? "Rubik" : weight >= FW_SEMIBOLD ? "Rubik Medium" : "Rubik";
    return CreateFontA(-px, 0, 0, 0, own ? (weight >= FW_BOLD ? FW_BOLD : FW_NORMAL) : weight, 0, 0, 0, DEFAULT_CHARSET,
                       OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, name);
}
/// The help text word-wrapped to width, a TextOutA a line (the POSIX GDI stand-in has no DrawTextA), drawn at (x, y)
/// when draw: its height in pixels either way
int wrap_help(HDC dc, const char* text, int width, int x, int y, bool draw) {
    SIZE sz{};
    GetTextExtentPoint32A(dc, "Ag", 2, &sz);
    const int line_h = (int)sz.cy;
    int lines = 0;
    for (const char* p = text; *p;) {
        int n = 0;   // (the most whole words that fit; one word at least, however long)
        for (int end = 0;; end++) {
            while (p[end] && p[end] != ' ') end++;
            GetTextExtentPoint32A(dc, p, end, &sz);
            if (n && sz.cx > width) break;
            n = end;
            if (!p[end]) break;
        }
        if (draw) TextOutA(dc, x, y + lines * line_h, p, n);
        lines++;
        p += n;
        while (*p == ' ') p++;
    }
    return lines * line_h;
}
void fill(HDC dc, int x, int y, int w, int h, COLORREF c) {
    RECT r{x, y, x + w, y + h};
    HBRUSH b = CreateSolidBrush(c);
    FillRect(dc, &r, b);
    DeleteObject(b);
}

namespace cart {
constexpr COLORREF kShell = RGB(0xB9, 0xB4, 0xAA), kGroove = RGB(0xA3, 0x9E, 0x93), kLabel = RGB(0xEC, 0xE6, 0xD8),
                   kInk = RGB(0x2A, 0x27, 0x24), kSoft = RGB(0x6B, 0x65, 0x5C), kFaint = RGB(0x8A, 0x83, 0x76),
                   kRule = RGB(0xE2, 0xDB, 0xCB), kOff = RGB(0xCF, 0xC7, 0xB6), kTrack = RGB(0xD6, 0xCD, 0xB9),
                   kHelp = RGB(0x4A, 0x45, 0x3E), kTagInk = RGB(0x8A, 0x6A, 0x10), kTag = RGB(0xF3, 0xE3, 0xB5),
                   kYellow = RGB(0xD4, 0xA5, 0x32), kBlue = RGB(0x3F, 0x5F, 0xB0), kGreen = RGB(0x3E, 0x8A, 0x4E);
struct TabLook { COLORREF strip, on, tint, edge; };
// (a tab's strip, the text on it, the selected row's tint and that row's edge: the strip half over the tint)
constexpr TabLook kTabs[3] = {{RGB(0xC0, 0x47, 0x3D), RGB(255, 255, 255), RGB(0xF2, 0xD6, 0xD0), RGB(0xD9, 0x8E, 0x86)},
                              {RGB(0x3E, 0x8A, 0x4E), RGB(255, 255, 255), RGB(0xD3, 0xE6, 0xD2), RGB(0x88, 0xB8, 0x90)},
                              {RGB(0x3F, 0x5F, 0xB0), RGB(255, 255, 255), RGB(0xD5, 0xDC, 0xEF), RGB(0x8A, 0x9D, 0xD0)}};
COLORREF mix(COLORREF a, COLORREF b, float t) {
    const auto ch = [t](int x, int y) { return (int)std::lround(x + (y - x) * t); };
    return RGB(ch(GetRValue(a), GetRValue(b)), ch(GetGValue(a), GetGValue(b)), ch(GetBValue(a), GetBValue(b)));
}
/// The game's typeface at px pixels: 0 regular, 1 medium, 2 bold, 3 extra bold
HFONT face(float px, int weight) {
    const bool own = startup_fonts();
    const char* name = !own ? "Segoe UI" : weight == 1 ? "Rubik Medium" : weight == 3 ? "Rubik ExtraBold" : "Rubik";
    const int w = own ? (weight == 2 ? FW_BOLD : FW_NORMAL) : weight == 0 ? FW_NORMAL : weight == 1 ? FW_MEDIUM : FW_BOLD;
    return CreateFontA(-std::max(6, (int)std::lround(px)), 0, 0, 0, w, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                       CLEARTYPE_QUALITY, DEFAULT_PITCH, name);
}
int text_w(HDC dc, const std::string& t) {
    SIZE sz{};
    GetTextExtentPoint32A(dc, t.c_str(), (int)t.size(), &sz);
    return (int)sz.cx;
}
int text_h(HDC dc) {
    SIZE sz{};
    GetTextExtentPoint32A(dc, "Ag", 2, &sz);
    return (int)sz.cy;
}
/// A triangle pointing left (dir -1), right (1) or up (0), centred on (cx, cy), size pixels across, row by row
void triangle(HDC dc, int cx, int cy, int size, int dir, COLORREF c) {
    const int half = std::max(1, size / 2);
    for (int i = -half; i <= half; i++) {
        const int len = half - std::abs(i);   // (left and right: rows from the tip; up: rows growing to the base)
        if (dir == 0) startup_gdi::fill(dc, cx - (i + half) / 2, cy - half / 2 + (i + half) * 3 / 4, i + half + 1, 1, c);
        else if (dir > 0) startup_gdi::fill(dc, cx - half / 2, cy + i, len + 1, 1, c);
        else startup_gdi::fill(dc, cx + half / 2 - len, cy + i, len + 1, 1, c);
    }
}
} // namespace cart

/// (relink) The menu's pages other than the options in the same cartridge: the shell, its grooves and the cream label over
/// the whole bitmap (round corners see-through: menu_bmp is keyed), and the page's title on a strip of its colour at the
/// label's top left. The pages keep their own layout under it (vk_menu_hit.h): the strip stays above 38 * s
void cart_page(HDC dc, int w, int h, int s, const char* title, COLORREF strip, COLORREF on) {
    using namespace cart;
    startup_gdi::fill(dc, 0, 0, w, h, RGB(0, 0, 0));
    startup_gdi::round_rect(dc, 0, 0, w, h, 10 * s, kShell);
    for (int x = 40 * s; x + 3 * s < w - 40 * s; x += 9 * s) startup_gdi::fill(dc, x, 2 * s, 2 * s, 3 * s, kGroove);
    startup_gdi::round_rect(dc, 5 * s, 7 * s, w - 10 * s, h - 12 * s, 7 * s, kLabel);
    HFONT f = face(17.0f * (float)s, 3);
    HGDIOBJ old = SelectObject(dc, f);
    const int tw = text_w(dc, title) + 28 * s;
    startup_gdi::round_rect(dc, 5 * s, 7 * s, tw, 31 * s + 7 * s, 7 * s, strip);
    startup_gdi::fill(dc, 5 * s, 38 * s, tw, 7 * s, kLabel);   // (the strip's round bottom hidden: square on the label)
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, on);
    TextOutA(dc, 5 * s + 14 * s, 7 * s + (31 * s - text_h(dc)) / 2, title, (int)strlen(title));
    SelectObject(dc, old);
    DeleteObject(f);
}
/// A selected line on those pages: the page's colour, light, with an edge of it
void cart_selected(HDC dc, int x, int y, int w, int h, int s, COLORREF tint, COLORREF edge) {
    startup_gdi::round_rect(dc, x, y, w, h, 7 * s, edge);
    startup_gdi::round_rect(dc, x + s, y + s, w - 2 * s, h - 2 * s, 6 * s, tint);
}

// ---- (relink) the Travel page: the places of Link's age, grouped by area (gpu/vk/oot_tp_places.h), with a
// picture of each, taken in play the first time Link stays a few seconds at the place (vk_replay.cpp teleport_capture). One
// file a place in the game folder's teleport_pictures folder; nothing is shipped. The files are read and written by a worker
// thread of their own (picture_worker), never by the replay's, the game's or the menu's: a picture is 389 KB, and a present
// must not wait on a disk ----
struct TeleportPicture {
    uint32_t w = 0, h = 0;
    std::vector<unsigned char> rgba;
};
std::map<std::string, TeleportPicture> teleport_captures;   // (by a place's key; std::map: its nodes stay put)
std::map<std::string, int> teleport_files;   // (by key: 1 its file being read, 2 read: in teleport_captures, or none on disk)
std::mutex teleport_captures_mu;   // (both maps: the worker's thread and the menu's)
std::atomic<int> teleport_capture_want{-1};
// (whether pictures are taken at all: only once the player has opened the Travel page, in this session or an earlier one (its
// folder is there). A player who never opens it gets no files and no notice over the game)
std::atomic<bool> travel_pictures_on{false};
std::atomic<int> teleport_map_now{-1};   // (the place nearest Link, as the game side says)
std::atomic<int64_t> notice_until{0};    // (a line over the game while the menu is closed: a picture taken, until then)
std::atomic<bool> notice_new{false};     // (drawn again at the next frame: a new picture)
char notice_text[128];
/// A file in the game folder, or "" (logged) when unknown
std::string game_folder_file(const char* name) {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    const std::string path(exe, n && n < MAX_PATH ? n : 0);
    const size_t slash = path.find_last_of("\\/");
    if (slash == std::string::npos) {
        logf("menu: no game folder for %s", name);
        return "";
    }
    return path.substr(0, slash + 1) + name;
}
/// A picture's file, teleport_pictures/<key>.bin in the game folder, or "" when the folder is unknown
std::string teleport_picture_path(const std::string& key) {
    const std::string dir = game_folder_file("teleport_pictures");
    return dir.empty() ? "" : dir + "/" + key + ".bin";
}
// A picture file: 'FTP2', width, height, then its RGB (no alpha: a picture is opaque). 480x270: 389 KB
constexpr char kPictureMagic[4] = {'F', 'T', 'P', '2'};
bool read_picture_file(const std::string& path, TeleportPicture& p) {
    FILE* f = path.empty() ? nullptr : fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    uint32_t wh[2];
    bool ok = fread(magic, 1, 4, f) == 4 && !memcmp(magic, kPictureMagic, 4) && fread(wh, 4, 2, f) == 2 && wh[0] && wh[0] <= 2048 &&
              wh[1] && wh[1] <= 2048;
    std::vector<unsigned char> rgb;
    if (ok) {
        rgb.resize((size_t)wh[0] * wh[1] * 3);
        ok = fread(rgb.data(), 1, rgb.size(), f) == rgb.size();
    }
    fclose(f);
    if (!ok) {
        logf("menu: %s is no Travel picture, left out", path.c_str());
        return false;
    }
    p.w = wh[0], p.h = wh[1];
    p.rgba.resize((size_t)p.w * p.h * 4);
    for (size_t i = 0, n = (size_t)p.w * p.h; i < n; ++i)
        p.rgba[i * 4] = rgb[i * 3], p.rgba[i * 4 + 1] = rgb[i * 3 + 1], p.rgba[i * 4 + 2] = rgb[i * 3 + 2], p.rgba[i * 4 + 3] = 255;
    return true;
}
bool write_picture_file(const std::string& path, const TeleportPicture& p) {
    if (path.empty()) return false;
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    std::vector<unsigned char> rgb((size_t)p.w * p.h * 3);
    for (size_t i = 0, n = (size_t)p.w * p.h; i < n; ++i)
        rgb[i * 3] = p.rgba[i * 4], rgb[i * 3 + 1] = p.rgba[i * 4 + 1], rgb[i * 3 + 2] = p.rgba[i * 4 + 2];
    FILE* f = fopen((path + ".tmp").c_str(), "wb");
    if (!f) return false;
    const uint32_t wh[2] = {p.w, p.h};
    fwrite(kPictureMagic, 1, 4, f), fwrite(wh, 4, 2, f), fwrite(rgb.data(), 1, rgb.size(), f);
    bool ok = !ferror(f);
    ok = fclose(f) == 0 && ok;
    if (ok) std::filesystem::rename(path + ".tmp", path, ec);   // (over the old one, on either platform)
    return ok && !ec;
}
// The worker's jobs: read a picture's file; Link has stayed near a place (its picture wanted when it has none, here or on
// disk); keep and write a picture taken; forget one (the page's retake: taken again at the next stay there)
struct PictureJob {
    enum Kind { Read, Arrived, Store, Forget, Begin } kind = Read;   // (Begin: the page opened the first time, its folder made)
    int place = -1;     // (Arrived, Store, Forget: a place of oot_tp_places)
    std::string name;   // (Read: a place's key)
    TeleportPicture pic;
};
std::mutex picture_jobs_mu;
std::condition_variable picture_jobs_cv;
std::deque<PictureJob> picture_jobs;
/// The picture of this key read into teleport_captures when its file is there; whether it has one now. On the worker
bool picture_read_now(const std::string& name) {
    {
        std::lock_guard lk(teleport_captures_mu);
        if (teleport_captures.count(name)) return true;
        if (teleport_files[name] == 2) return false;
    }
    TeleportPicture p;
    const bool have = read_picture_file(teleport_picture_path(name), p);
    std::lock_guard lk(teleport_captures_mu);
    if (have && !teleport_captures.count(name)) teleport_captures[name] = std::move(p);
    teleport_files[name] = 2;
    return teleport_captures.count(name) != 0;
}
void picture_worker() {
    for (;;) {
        PictureJob j;
        {
            std::unique_lock lk(picture_jobs_mu);
            picture_jobs_cv.wait(lk, [] { return !picture_jobs.empty(); });
            j = std::move(picture_jobs.front());
            picture_jobs.pop_front();
        }
        if (j.kind == PictureJob::Begin) {
            std::error_code ec;
            const std::string dir = game_folder_file("teleport_pictures");
            if (!dir.empty()) std::filesystem::create_directories(dir, ec);
            logf("menu: Travel pictures are taken from now on, in %s%s", dir.c_str(), ec ? " (the folder could not be made)" : "");
        } else if (j.kind == PictureJob::Read) {
            picture_read_now(j.name);
            menu_refresh = true;   // (the page shows it now, if it is the chosen place's)
        } else if (j.kind == PictureJob::Arrived) {
            if (!picture_read_now(oot_tp_places[j.place].key)) teleport_capture_want.store(j.place, std::memory_order_relaxed);
        } else if (j.kind == PictureJob::Forget) {
            const OotTpPlace& place = oot_tp_places[j.place];
            {
                std::lock_guard lk(teleport_captures_mu);
                teleport_captures.erase(place.key);
                teleport_files[place.key] = 2;
            }
            std::error_code ec;
            std::filesystem::remove(teleport_picture_path(place.key), ec);
            logf("menu: Travel picture of %s (%s) forgotten: taken again at the next stay there", place.label, place.key);
            menu_refresh = true;
        } else {
            const OotTpPlace& place = oot_tp_places[j.place];
            for (size_t i = 3; i < j.pic.rgba.size(); i += 4) j.pic.rgba[i] = 255;   // (opaque, whatever the picture's alpha held)
            const bool saved = write_picture_file(teleport_picture_path(place.key), j.pic);
            int count;
            {
                std::lock_guard lk(teleport_captures_mu);
                teleport_captures[place.key] = std::move(j.pic);
                teleport_files[place.key] = 2;
                count = (int)teleport_captures.size();
            }
            logf("menu: Travel picture of %s (%s) taken (%d in memory)%s", place.label, place.key, count, saved ? "" : "; not saved");
            if (saved) {   // (the line over the game for 2.5 s)
                snprintf(notice_text, sizeof notice_text, "Travel picture saved: %s", place.label);
                notice_new.store(true, std::memory_order_release);
                notice_until.store(prof_now() + 2500000000LL, std::memory_order_relaxed);
            }
        }
    }
}
void picture_job(PictureJob&& j) {
    static std::once_flag started;
    std::call_once(started, [] { std::thread(picture_worker).detach(); });   // (it lives as long as the game)
    {
        std::lock_guard lk(picture_jobs_mu);
        picture_jobs.push_back(std::move(j));
    }
    picture_jobs_cv.notify_one();
}

// the list: the places of one age by area, as list_item and list_mouse take a list (an area's heading is a line too)
struct TravelList {
    int adult = -1;
    std::vector<int> place;               // (the shown places, in order: indices into oot_tp_places)
    std::vector<TeleportSection> areas;   // (each area's title and first shown place)
};
TravelList travel;
int travel_choice = 0, travel_scroll = 0;
int travel_age_l[2], travel_age_r[2];   // (the Child and Adult buttons' left and right edges as last drawn: the mouse's)
int travel_pick = -1;   // (the age whose places the page lists, chosen with Tab or the top face button; -1 Link's own)
/// The age whose places the page lists: the one chosen, else Link's as the game side says it (1 adult, else 0: unknown is the
/// child's). Travelling to the other age's place makes Link that age, as the cheats' Switch age does
int travel_age() { return travel_pick >= 0 ? travel_pick : link_age.load(std::memory_order_relaxed) == 1 ? 1 : 0; }
/// The list for Link's age now (made again when it changed)
void travel_list() {
    const int adult = travel_age();
    if (travel.adult == adult) return;
    travel = {};
    travel.adult = adult;
    for (const OotTpArea& a : oot_tp_areas) {
        if (a.adult != adult) continue;
        travel.areas.push_back({a.title, (int)travel.place.size()});
        for (int i = a.first; i < a.first + a.count; i++) travel.place.push_back(i);
    }
    travel_choice = std::clamp(travel_choice, 0, std::max(0, (int)travel.place.size() - 1));
}
/// Scrolls the list so the chosen place (and its area's heading, at the top) is in view
void travel_reveal() {
    const int sections = (int)travel.areas.size(), n = (int)travel.place.size();
    if (!sections) return;
    int section = 0;
    while (section + 1 < sections && travel_choice >= travel.areas[section + 1].first) section++;
    const int line = travel_choice + section + 1;
    if (travel_choice == travel.areas[section].first) travel_scroll = std::min(travel_scroll, line - 1);
    if (line < travel_scroll) travel_scroll = line;
    if (line >= travel_scroll + kTeleportLines) travel_scroll = line - kTeleportLines + 1;
    travel_scroll = std::max(0, std::min(travel_scroll, n + sections - kTeleportLines));
}
/// The chosen place's picture copied out (under the lock), and whether it has one; its file asked for once when not read yet
bool travel_picture(int place, TeleportPicture* out, bool* reading) {
    *reading = false;
    if (place < 0 || place >= OOT_TP_PLACES) return false;
    const std::string key = oot_tp_places[place].key;
    std::lock_guard lk(teleport_captures_mu);
    const auto it = teleport_captures.find(key);
    if (it != teleport_captures.end()) {
        *out = it->second;
        return true;
    }
    int& state = teleport_files[key];
    if (state == 2) return false;
    *reading = true;
    if (state == 0) {
        state = 1;
        PictureJob j;
        j.kind = PictureJob::Read, j.name = key;
        picture_job(std::move(j));
    }
    return false;
}
/// Whether a place has a picture, as far as the files read so far say (the list's mark)
bool travel_has_picture(int place) {
    std::lock_guard lk(teleport_captures_mu);
    const std::string key = oot_tp_places[place].key;
    if (teleport_captures.count(key)) return true;
    int& state = teleport_files[key];
    if (state == 0) {   // (asked for: the list marks it once it is read)
        state = 1;
        PictureJob j;
        j.kind = PictureJob::Read, j.name = key;
        picture_job(std::move(j));
    }
    return false;
}
/// The Travel page (paint_menu, page 6): the list on the left, the chosen place's picture, name and area on the right

// ---- (relink) the options page (the Graphics, Gameplay and Controls tabs) as a cartridge: a grey shell with its grooves, a
// cream label, the tabs as coloured strips across its top (red, green, blue), the rows under small headings, the selected
// one tinted in its tab's colour with what it does inside it, and the buttons along the bottom. Laid out on a 680-pixel-wide
// frame at 720 lines, times opt_u (the window's height over 720: 1.5 at 1080 lines); paint_options records where each part
// went (opt_hit) for the mouse. The game's own typeface (Rubik, vk_startup_paint.h startup_fonts), else the system's ----

/// (relink) A page's rows scrolled as the PC games' settings do: the wheel scrolls them (not the selection, nor a value),
/// the bar on their right is taken and dragged, or clicked above or below its thumb for a page; the keys and the pad move the
/// selection, and the next drawing brings it into view (reveal). The mouse selects the row it moves onto (each page's own)
struct Scroller {
    int px = 0;              // (how far the rows are scrolled up, in pixels)
    bool reveal = true;      // (the next drawing scrolls the selected row into view: the keys moved it, or the page opened)
    int content = 0, room = 0, top = 0;     // (all the rows' height, the band they show in, its top)
    int bar_l = 0, bar_r = 0, thumb_t = 0, thumb_b = 0;   // (the bar's place for the mouse, its thumb as drawn)
    int drag = -1;           // (the thumb held: where in it the cursor took it; -1 none)
};
Scroller opt_sc, cheat_sc, list_sc[2], travel_sc, controls_sc;
/// The rows' scroll for this drawing: the selected one (sel_top to sel_bottom, from the rows' top) brought into view when the
/// keys moved it, else where the wheel or the bar left it; within the rows either way
void scroll_fit(Scroller& sc, int content, int room, int top, int sel_top, int sel_bottom) {
    sc.content = content, sc.room = room, sc.top = top;
    if (content <= room) sc.px = 0;
    else {
        if (sc.reveal) sc.px = std::max(std::min(sc.px, sel_top), sel_bottom - room);
        sc.px = std::clamp(sc.px, 0, content - room);
    }
    sc.reveal = false;
}
/// The bar at x, when the rows do not all fit: a faint track, its thumb in the page's colour while held (else the shell's)
void scroll_bar(HDC dc, Scroller& sc, int x, float u, COLORREF held) {
    sc.bar_l = sc.bar_r = 0;
    if (sc.content <= sc.room) return;
    const int w = std::max(3, (int)std::lround(6 * u)), r = w / 2;
    const int th = std::max((int)std::lround(28 * u), sc.room * sc.room / sc.content);
    const int ty = sc.top + (sc.room - th) * sc.px / std::max(1, sc.content - sc.room);
    startup_gdi::round_rect(dc, x, sc.top, w, sc.room, r, cart::kRule);
    startup_gdi::round_rect(dc, x, ty, w, th, r, sc.drag >= 0 ? held : cart::kOff);
    sc.thumb_t = ty, sc.thumb_b = ty + th;
    sc.bar_l = x - (int)std::lround(8 * u), sc.bar_r = x + w + (int)std::lround(8 * u);   // (wider than drawn: easy to take)
}
/// The mouse on a page's scroll: the wheel anywhere over the page scrolls (three rows a notch), the bar is dragged or
/// clicked for a page. True while the cursor is on the bar or holds its thumb (the rows then leave the mouse alone)
bool scroll_mouse(Scroller& sc, const MkwVkPointer& ptr, int mx, int my, int mw, int mh, float u, bool& changed) {
    const bool inside = mx >= 0 && my >= 0 && mx < mw && my < mh;
    const int span = sc.content - sc.room;
    if (ptr.wheel && inside && span > 0) {
        const int before = sc.px;
        sc.px = std::clamp(sc.px - ptr.wheel * (int)std::lround(96 * u), 0, span);   // (three 32-pixel rows a notch)
        changed |= sc.px != before;
    }
    if (sc.drag >= 0) {   // (held: the thumb follows the cursor's height until let go)
        if (!(ptr.held & HOS_MOUSE_LEFT)) sc.drag = -1, changed = true;
        else {
            const int th = sc.thumb_b - sc.thumb_t, free = std::max(1, sc.room - th);
            const int want = std::clamp((my - sc.drag - sc.top) * span / free, 0, std::max(0, span));
            if (want != sc.px) sc.px = want, changed = true;
        }
        return true;
    }
    const bool on_bar = span > 0 && sc.bar_r > sc.bar_l && mx >= sc.bar_l && mx < sc.bar_r && my >= sc.top && my < sc.top + sc.room;
    if (on_bar && (ptr.pressed & HOS_MOUSE_LEFT)) {
        if (my >= sc.thumb_t && my < sc.thumb_b) sc.drag = my - sc.thumb_t;
        else sc.px = std::clamp(sc.px + (my < sc.thumb_t ? -sc.room : sc.room), 0, span);   // (a page up or down)
        changed = true;
    }
    return on_bar;
}

float opt_u = 1;   // (the options page's scale: redraw_menu)
int opt_help_box = 0;   // (the height of the box under the rows for what the selected row does: options_size)
int& opt_scroll = opt_sc.px;   // (how far the rows are scrolled up, in pixels, when the window is too short for them all)
/// Where paint_options put each part, in the menu's pixels (the mouse: menu_tick)
struct OptHit {
    int tab_l[TAB_COUNT] = {}, tab_r[TAB_COUNT] = {}, tabs_bottom = 0;
    int body_top = 0, body_bottom = 0;            // (the rows' visible band)
    int rows = 0, row_top[kRows] = {}, row_bottom[kRows] = {};
    int kind[kRows] = {};                         // (0 a list, 1 a switch, 2 choices, 3 a slider, 4 a page)
    int value_l[kRows] = {}, value_r[kRows] = {};   // (the value's part of the row)
    int choices[kRows] = {}, choice_l[kRows][6] = {}, choice_r[kRows][6] = {};
    int track_l[kRows] = {}, track_r[kRows] = {};
} opt_hit;

/// A row's value as text
std::string row_text(const Row& r, int v) { return r.value && r.text ? r.text(v) : ""; }
/// How a row's value is shown: 1 an on/off switch, 2 its few short choices side by side, 3 a slider, 4 a page, 0 a list
int row_kind(const Row& r) {
    if (!r.value) return 4;
    if (r.text == tx_onoff) return 1;
    if (is_slider(r) || stops_at_ends(r)) return 3;
    static const int* const few[] = {&opts.upscale, &opts.realtime, &opts.text_speed, &opts.audio_output, &opts.ultrawide,
                                     &opts.robust, &opts.mouse_native, &opts.button_prompts};
    for (const int* f : few)
        if (r.value == f && (r.hi - r.lo) / std::max(1, r.step) + 1 <= 4) return 2;
    return 0;
}
/// The tallest text of what a row does, over every row and every line its value can have: the box under the rows keeps that room
int opt_help_most(float u) {
    HDC dc = CreateCompatibleDC(nullptr);
    HFONT f = cart::face(13 * u, 0);
    HGDIOBJ old = SelectObject(dc, f);
    const int width = (int)((680 - 20 - 20 - 24) * u);
    int most = 0;
    for (const Row& r : rows) {
        int hint = 0;
        for (int v = r.lo; r.value && r.hint && v <= r.hi; v++)
            for (int k = 0; const char* t = r.hint(v, k); k++) hint = std::max(hint, wrap_help(dc, t, width, 0, 0, false));
        most = std::max(most, wrap_help(dc, r.help, width, 0, 0, false) + (hint ? (int)(4 * u) + hint : 0));
    }
    SelectObject(dc, old), DeleteObject(f), DeleteDC(dc);
    return most;
}
// (the frame's sizes, in its pixels at 720 lines)
constexpr float kOptW = 680, kOptTop = 22, kOptTabH = 36, kOptRowH = 32, kOptHeadH = 24, kOptFootH = 36, kOptPendH = 34;
/// The rows' height on tab t, the selected row's text left out
float opt_rows_h(int t) {
    float hh = 0;
    const char* last = nullptr;
    for (int k = 0, n = tab_row_count(t); k < n; k++) {
        int place = 0;
        const char* g = row_group(tab_row_to_global(t, k), &place);
        if (*g && (!last || strcmp(g, last))) hh += kOptHeadH;
        last = g;
        hh += kOptRowH;
    }
    return hh;
}

void paint_options(HDC dc, uint32_t w, uint32_t h) {
    using namespace cart;
    const float u = opt_u;
    const auto P = [u](float v) { return (int)std::lround(v * u); };
    const int W = (int)w, H = (int)h;
    const TabLook& look = kTabs[cur_tab];
    SetBkMode(dc, TRANSPARENT);
    // the shell, its grooves and the label (the corners outside it black: see-through, menu_bmp is keyed)
    startup_gdi::fill(dc, 0, 0, W, H, RGB(0, 0, 0));
    startup_gdi::round_rect(dc, 0, 0, W, H, P(14), kShell);
    for (int x = P(40); x + P(3) < W - P(40); x += P(9)) startup_gdi::fill(dc, x, P(8), P(3), P(6), kGroove);
    const int lx = P(10), lw = W - 2 * lx, ly = P(kOptTop), lh = H - ly - P(12);
    startup_gdi::round_rect(dc, lx, ly, lw, lh, P(9), kLabel);
    // the tabs: strips across the label's top, the shown one taller
    static const char* const titles[TAB_COUNT] = {"Graphics", "Gameplay", "Controls"};
    const float units = 1.3f + (TAB_COUNT - 1);
    int x = lx;
    for (int t = 0; t < TAB_COUNT; t++) {
        const bool on = t == cur_tab;
        const int tw = (int)std::lround((float)(lw - P(3) * (TAB_COUNT - 1)) * (on ? 1.3f : 1.0f) / units);
        const int top = ly + (on ? 0 : P(7)), right = t == TAB_COUNT - 1 ? lx + lw : x + tw;
        const COLORREF c = on ? kTabs[t].strip : mix(kTabs[t].strip, kLabel, hover_tab == t ? 0.25f : 0.5f);
        startup_gdi::round_rect(dc, x, top, right - x, ly + P(kOptTabH) - top + P(9), P(9), c);   // (its bottom hidden below)
        HFONT f = face(on ? 17 * u : 14 * u, on ? 3 : 1);
        HGDIOBJ old = SelectObject(dc, f);
        SetTextColor(dc, on ? kTabs[t].on : mix(kTabs[t].on, c, 0.15f));
        TextOutA(dc, x + P(14), top + (ly + P(kOptTabH) - top - text_h(dc)) / 2, titles[t], (int)strlen(titles[t]));
        SelectObject(dc, old), DeleteObject(f);
        opt_hit.tab_l[t] = x, opt_hit.tab_r[t] = right;
        x = right + P(3);
    }
    opt_hit.tabs_bottom = ly + P(kOptTabH);
    startup_gdi::fill(dc, lx, ly + P(kOptTabH), lw, P(9), kLabel);   // (the label again under the strips' round bottoms)
    // the bottom: what the selected row does, the restart line when one is pending, then the buttons
    const bool pending = restart_pending() != 0;
    const int foot_top = ly + lh - P(kOptFootH), pend_top = foot_top - (pending ? P(kOptPendH) : 0);
    const int help_top = pend_top - opt_help_box;
    startup_gdi::fill(dc, lx, foot_top, lw, std::max(1, P(1)), kRule);
    HFONT small = face(12.5f * u, 1);
    HGDIOBJ old = SelectObject(dc, small);
    if (pending) {
        startup_gdi::round_rect(dc, lx + P(12), pend_top + P(3), lw - P(24), P(kOptPendH) - P(8), P(8), kTag);
        SetTextColor(dc, RGB(0x6B, 0x52, 0x0C));
        const std::string line = restart_line(dc, lw - P(48)) + " (when the menu closes)";
        TextOutA(dc, lx + P(24), pend_top + P(3) + (P(kOptPendH) - P(8) - text_h(dc)) / 2, line.c_str(), (int)line.size());
    }
    // the rows
    const int body_l = lx + P(10), body_r = lx + lw - P(10);
    const int body_top = ly + P(kOptTabH) + P(6), body_bottom = help_top - P(4);
    opt_hit.body_top = body_top, opt_hit.body_bottom = body_bottom;
    HFONT name_f = face(15 * u, 0), name_b = face(15 * u, 2), value_f = face(14 * u, 0), chip_f = face(13 * u, 0),
          chip_b = face(13 * u, 2), head_f = face(10.5f * u, 2), help_f = face(13 * u, 0), tag_f = face(9.5f * u, 2);
    const int n = tab_row_count(cur_tab), sel = tab_selected[cur_tab];
    const int help_w = body_r - body_l - P(24);
    SelectObject(dc, help_f);
    const Row& selected = rows[tab_row_to_global(cur_tab, sel)];
    // (the rows' places, all one height; scrolled so the selected one shows when they do not all fit)
    int y = 0, sel_top = 0, sel_bottom = 0;
    std::vector<int> head_at(n, -1), top_at(n);
    const char* last = nullptr;
    for (int k = 0; k < n; k++) {
        int place = 0;
        const char* g = row_group(tab_row_to_global(cur_tab, k), &place);
        if (*g && (!last || strcmp(g, last))) head_at[k] = y, y += P(kOptHeadH);
        last = g;
        top_at[k] = y;
        y += P(kOptRowH);
        if (k == sel) sel_top = top_at[k] - (head_at[k] >= 0 ? P(kOptHeadH) : 0), sel_bottom = y;
    }
    const int room = body_bottom - body_top;
    scroll_fit(opt_sc, y, room, body_top, sel_top, sel_bottom);
    HRGN clip = CreateRectRgn(lx, body_top, lx + lw, body_bottom);
    SelectClipRgn(dc, clip);
    opt_hit.rows = n;
    for (int k = 0; k < n; k++) {
        const int gi = tab_row_to_global(cur_tab, k);
        const Row& r = rows[gi];
        const bool on = k == sel;
        if (head_at[k] >= 0) {
            int place = 0;
            SelectObject(dc, head_f);
            SetTextColor(dc, kFaint);
            startup_gdi::spaced(dc, row_group(gi, &place), body_l + P(12), body_top + head_at[k] - opt_scroll + P(10), P(2.5f), false);
        }
        const int top = body_top + top_at[k] - opt_scroll, rh = P(kOptRowH);
        opt_hit.row_top[k] = top, opt_hit.row_bottom[k] = top + rh;
        if (on) {
            startup_gdi::round_rect(dc, body_l, top + P(2), body_r - body_l, rh - P(3), P(10), look.edge);
            startup_gdi::round_rect(dc, body_l + std::max(1, P(1.5f)), top + P(2) + std::max(1, P(1.5f)),
                                    body_r - body_l - 2 * std::max(1, P(1.5f)), rh - P(3) - 2 * std::max(1, P(1.5f)), P(9), look.tint);
        } else {
            const bool next_heads = k + 1 < n && head_at[k + 1] >= 0;
            if (k + 1 < n && k + 1 != sel && !next_heads) startup_gdi::fill(dc, body_l + P(4), top + rh - 1, body_r - body_l - P(8), std::max(1, P(1)), kRule);
        }
        const int mid = top + P(kOptRowH) / 2 + P(1);
        // the name, then its restart tag
        SelectObject(dc, on ? name_b : name_f);
        SetTextColor(dc, kInk);
        TextOutA(dc, body_l + P(12), mid - text_h(dc) / 2, r.name, (int)strlen(r.name));
        int after = body_l + P(12) + text_w(dc, r.name);
        if (r.restart) {
            SelectObject(dc, tag_f);
            const int tw = text_w(dc, "restart") + P(10), th = text_h(dc) + P(2);
            startup_gdi::round_rect(dc, after + P(8), mid - th / 2, tw, th, th / 2, kTag);
            SetTextColor(dc, kTagInk);
            TextOutA(dc, after + P(13), mid - th / 2 + P(1), "restart", 7);
            after += P(8) + tw;
        }
        // the value, from the right
        const int vr = body_r - P(12);
        const int kind = row_kind(r);
        opt_hit.kind[k] = kind, opt_hit.value_r[k] = vr, opt_hit.choices[k] = 0;
        if (kind == 4) {   // (a page)
            cart::triangle(dc, vr - P(5), mid, P(10), 1, look.strip);
            opt_hit.value_l[k] = vr - P(20);
        } else if (kind == 1) {   // (a switch)
            const int sw = P(36), sh = P(20), sx = vr - sw, sy = mid - sh / 2;
            startup_gdi::round_rect(dc, sx, sy, sw, sh, sh / 2, *r.value ? look.strip : kOff);
            const int knob = sh - P(6);
            startup_gdi::round_rect(dc, *r.value ? sx + sw - P(3) - knob : sx + P(3), sy + P(3), knob, knob, knob / 2, RGB(255, 255, 255));
            opt_hit.value_l[k] = sx;
        } else if (kind == 2) {   // (its few choices side by side)
            const int count = std::min(6, (r.hi - r.lo) / std::max(1, r.step) + 1);
            std::vector<std::string> labels;
            int total = 0;
            for (int i = 0; i < count; i++) {
                labels.push_back(row_text(r, r.lo + i * r.step));
                SelectObject(dc, chip_b);
                total += text_w(dc, labels.back()) + P(18) + (i ? P(3) : 0);
            }
            int cx = vr - total;
            const int ch = P(22);
            for (int i = 0; i < count; i++) {
                const bool picked = *r.value == r.lo + i * r.step;
                SelectObject(dc, chip_b);
                const int cw = text_w(dc, labels[i]) + P(18);
                if (picked) startup_gdi::round_rect(dc, cx, mid - ch / 2, cw, ch, ch / 2, look.strip);
                SelectObject(dc, picked ? chip_b : chip_f);
                SetTextColor(dc, picked ? look.on : kSoft);
                TextOutA(dc, cx + (cw - text_w(dc, labels[i])) / 2, mid - text_h(dc) / 2, labels[i].c_str(), (int)labels[i].size());
                opt_hit.choice_l[k][i] = cx, opt_hit.choice_r[k][i] = cx + cw;
                cx += cw + P(3);
            }
            opt_hit.choices[k] = count, opt_hit.value_l[k] = vr - total;
        } else if (kind == 3) {   // (a slider, its value after it)
            SelectObject(dc, value_f);
            const std::string t = row_text(r, *r.value);
            const int vw = std::max(text_w(dc, t), P(44));
            SetTextColor(dc, kInk);
            TextOutA(dc, vr - text_w(dc, t), mid - text_h(dc) / 2, t.c_str(), (int)t.size());
            const int tr = vr - vw - P(12), tl = tr - P(120);
            const int knob_x = tl + (tr - tl) * menu_hit::slider_permille(*r.value, r.lo, r.hi) / 1000;
            startup_gdi::round_rect(dc, tl, mid - P(2), tr - tl, P(4), P(2), kTrack);
            startup_gdi::round_rect(dc, tl, mid - P(2), std::max(P(4), knob_x - tl), P(4), P(2), look.strip);
            const int kd = P(16);
            startup_gdi::round_rect(dc, knob_x - kd / 2, mid - kd / 2, kd, kd, kd / 2, on ? look.tint : kLabel);
            startup_gdi::round_rect(dc, knob_x - kd / 2 + P(2), mid - kd / 2 + P(2), kd - P(4), kd - P(4), (kd - P(4)) / 2, look.strip);
            opt_hit.track_l[k] = tl, opt_hit.track_r[k] = tr, opt_hit.value_l[k] = tl - P(8);
        } else {   // (a list: its value; the selected row's in a pill between two arrows)
            std::string t = row_text(r, *r.value);
            const bool fixed = row_fixed(r.value);
            if (on && !fixed) {
                SelectObject(dc, chip_b);
                const int room_w = vr - after - P(70);
                while (t.size() > 4 && text_w(dc, t) > room_w) t.pop_back();
                const int pw = text_w(dc, t) + P(24), ph = P(24), px = vr - P(16) - pw;
                const bool output_row = r.value == &opts.output_res;
                const bool can_l = !output_row || step_output_choice(*r.value, -1) != *r.value;
                const bool can_r = !output_row || step_output_choice(*r.value, 1) != *r.value;
                cart::triangle(dc, vr - P(5), mid, P(10), 1, can_r ? look.strip : kOff);
                startup_gdi::round_rect(dc, px, mid - ph / 2, pw, ph, ph / 2, look.strip);
                SetTextColor(dc, look.on);
                TextOutA(dc, px + P(12), mid - text_h(dc) / 2, t.c_str(), (int)t.size());
                cart::triangle(dc, px - P(11), mid, P(10), -1, can_l ? look.strip : kOff);
                opt_hit.value_l[k] = px - P(18);
            } else {
                SelectObject(dc, value_f);
                const int room_w = vr - after - P(16);
                if (text_w(dc, t) > room_w) {
                    while (t.size() > 1 && text_w(dc, t + "...") > room_w) t.pop_back();
                    t += "...";
                }
                SetTextColor(dc, fixed ? kFaint : kSoft);
                TextOutA(dc, vr - text_w(dc, t), mid - text_h(dc) / 2, t.c_str(), (int)t.size());
                opt_hit.value_l[k] = vr - text_w(dc, t);
            }
        }
    }
    SelectClipRgn(dc, nullptr);
    DeleteObject(clip);
    {   // what the selected row does, and the line on its value: in its box, a bar of the tab's colour at its left
        startup_gdi::fill(dc, body_l + P(4), help_top + P(2), body_r - body_l - P(8), std::max(1, P(1)), kRule);
        startup_gdi::round_rect(dc, body_l + P(6), help_top + P(12), P(3), opt_help_box - P(20), P(1.5f), look.strip);
        SelectObject(dc, help_f);
        SetTextColor(dc, kHelp);
        const int hy = help_top + P(10);
        const int about = wrap_help(dc, selected.help, help_w, body_l + P(18), hy, true);
        if (const char* hint = selected.value && selected.hint ? selected.hint(*selected.value, -1) : nullptr) {
            SetTextColor(dc, kSoft);
            wrap_help(dc, hint, help_w, body_l + P(18), hy + about + P(4), true);
        }
    }
    scroll_bar(dc, opt_sc, body_r + P(2), u, look.strip);   // (more above or below: the bar on the right)
    // the buttons: what the selected row's buttons do, Back, the tabs, the version
    SelectObject(dc, small);
    const int fy = foot_top + P(kOptFootH) / 2;
    int fx = lx + P(16);
    const auto glyph = [&](const char* letter, COLORREF c) {
        const int d = P(19);
        startup_gdi::round_rect(dc, fx, fy - d / 2, d, d, d / 2, c);
        SetTextColor(dc, RGB(255, 255, 255));
        TextOutA(dc, fx + (d - text_w(dc, letter)) / 2, fy - text_h(dc) / 2, letter, (int)strlen(letter));
        fx += d + P(6);
    };
    const auto say = [&](const char* t) {
        SetTextColor(dc, kHelp);
        TextOutA(dc, fx, fy - text_h(dc) / 2, t, (int)strlen(t));
        fx += text_w(dc, t) + P(20);
    };
    const int skind = row_kind(selected);
    if (skind == 4 || skind == 1) glyph("A", kBlue), say(skind == 4 ? "Open" : "Switch");
    else {
        cart::triangle(dc, fx + P(5), fy, P(10), -1, kYellow), cart::triangle(dc, fx + P(17), fy, P(10), 1, kYellow);
        fx += P(28);
        say("Change");
    }
    glyph("B", kGreen), say("Back");
    {   // (the tabs: the bumpers, Q and E)
        const char* t = "LB RB";
        const int tw = text_w(dc, t) + P(10), th = P(19);
        startup_gdi::round_rect(dc, fx, fy - th / 2, tw, th, P(5), kFaint);
        SetTextColor(dc, RGB(255, 255, 255));
        TextOutA(dc, fx + P(5), fy - text_h(dc) / 2, t, (int)strlen(t));
        fx += tw + P(6);
        say("Tabs");
    }
    SetTextColor(dc, kFaint);
    const std::string ver = version_text(RELINK_VERSION);
    if (fx + text_w(dc, ver) < lx + lw - P(16)) TextOutA(dc, lx + lw - P(16) - text_w(dc, ver), fy - text_h(dc) / 2, ver.c_str(), (int)ver.size());
    SelectObject(dc, old);
    for (HFONT f : {small, name_f, name_b, value_f, chip_f, chip_b, head_f, help_f, tag_f}) DeleteObject(f);
}
/// The options page's size at a window of w x h: the frame at the scale that fits, as tall as its longest tab with the
/// tallest text open, or the window's height less a margin (the rows scroll then)
void options_size(uint32_t w, uint32_t h, uint32_t* mw, uint32_t* mh) {
    float u = std::min((float)h / 720.0f, ((float)w - 32.0f) / kOptW);
    u = std::clamp(u, 0.6f, 4.0f);
    static float measured_at = -1;
    static int help_most = 0;
    if (measured_at != u) help_most = opt_help_most(u), measured_at = u;
    opt_u = u;
    opt_help_box = help_most + (int)std::lround(20 * u);
    float rows_h = 0;
    for (int t = 0; t < TAB_COUNT; t++) rows_h = std::max(rows_h, opt_rows_h(t));
    const float pend = restart_pending() ? kOptPendH : 0;
    const float full = (kOptTop + kOptTabH + 6 + rows_h + 4 + pend + kOptFootH + 12) * u + (float)opt_help_box;
    *mw = (uint32_t)std::lround(kOptW * u);
    *mh = (uint32_t)std::min(full, (float)h - 16.0f * u);
}

// ---- (relink) the Cheats page as the options' cartridge: the Gameplay tab's green, its rows under small headings, each
// with the control it is (a switch, a button for an action, a choice between arrows, a page), what the selected one does in
// the box under them, and a word when an action was done there. The rows keep their numbers (cheat_selected, menu_tick);
// cheat_order is the order they are shown in ----
const char* const cheat_labels[kCheatRows] = {"Heart protection", "No-clip flight", "Mark position", "Return to mark", "Set time",
                                              "Inventory item", "Change quantity", "Free camera", "Teleport", "Switch age",
                                              "Inventory", "Travel"};
const char* const cheat_help[kCheatRows] = {
    "Protects Link's hearts. Hits can still knock him back.",
    "Enable double-tap Jump for takeoff and landing. Hold Jump to climb or L to descend.",
    "Remember Link's position for this loaded scene.",
    "Teleport Link to the marked position in the current scene.",
    "Choose an hour with Left or Right. Enter sets the world clock.",
    "Choose a consumable with Left or Right.",
    "Choose how many to add or remove. Enter applies the change.",
    "Move and look with your usual controls. Page Up / Page Down or R / L changes height.",
    "Enter opens the map list (Left or Right also changes the map here). Link travels through the game's own map change, in his current age.",
    "Swaps Link between child and adult and reloads this map in the other age (the Market when this map has no match).",
    "Enter opens the inventory editor: unlock or relock equipment, items, upgrades and quest items, and change amounts.",
    "Go to a place you know: villages, dungeons' entrances, shops, warps. Each with a picture from your own game."};
constexpr int cheat_order[kCheatRows] = {0, 1, 7, 9, 2, 3, 4, 5, 6, 10, 8, 11};
const char* cheat_group(int row) {
    return row == 0 || row == 1 || row == 7 || row == 9 ? "LINK" : row == 2 || row == 3 ? "POSITION" : row == 4 ? "WORLD"
         : row == 5 || row == 6 || row == 10 ? "ITEMS" : "TRAVEL";
}
/// How a row is shown: 1 a switch, 2 a choice between arrows, 3 a button for its action, 4 a page
int cheat_kind(int row) {
    return row == 0 || row == 1 || row == 7 ? 1 : row == 4 || row == 5 || row == 6 || row == 8 ? 2 : row == 2 || row == 3 || row == 9 ? 3 : 4;
}
std::string cheat_done;   // (what the last action did, said in the box under the rows until cheat_done_until)
int64_t cheat_done_until = 0;
void cheat_say(std::string what) {
    cheat_done = std::move(what);
    cheat_done_until = prof_now() + 3000000000LL;
}
/// Where paint_cheats put each part, by the row's number (the mouse: menu_tick)
struct CheatHit {
    int body_top = 0, body_bottom = 0;
    int top[kCheatRows] = {}, bottom[kCheatRows] = {}, value_l[kCheatRows] = {}, value_r[kCheatRows] = {};
} cheat_hit;
int& cheat_scroll = cheat_sc.px;
int cheat_help_box = 0;
constexpr float kCheatHeadH = 24;

void paint_cheats(HDC dc, uint32_t w, uint32_t h) {
    using namespace cart;
    const float u = opt_u;
    const auto P = [u](float v) { return (int)std::lround(v * u); };
    const int W = (int)w, H = (int)h;
    const TabLook& look = kTabs[1];
    SetBkMode(dc, TRANSPARENT);
    startup_gdi::fill(dc, 0, 0, W, H, RGB(0, 0, 0));
    startup_gdi::round_rect(dc, 0, 0, W, H, P(14), kShell);
    for (int x = P(40); x + P(3) < W - P(40); x += P(9)) startup_gdi::fill(dc, x, P(8), P(3), P(6), kGroove);
    const int lx = P(10), lw = W - 2 * lx, ly = P(kOptTop), lh = H - ly - P(12);
    startup_gdi::round_rect(dc, lx, ly, lw, lh, P(9), kLabel);
    // the strip: the page's name, and where it came from
    HFONT strip_f = face(17 * u, 3), small = face(12.5f * u, 1);
    HGDIOBJ old = SelectObject(dc, strip_f);
    const int sw = text_w(dc, "Cheats") + P(28);
    startup_gdi::round_rect(dc, lx, ly, sw, P(kOptTabH) + P(9), P(9), look.strip);
    startup_gdi::fill(dc, lx, ly + P(kOptTabH), sw, P(9), kLabel);
    SetTextColor(dc, look.on);
    TextOutA(dc, lx + P(14), ly + (P(kOptTabH) - text_h(dc)) / 2, "Cheats", 6);
    SelectObject(dc, small);
    SetTextColor(dc, kFaint);
    const char* from = "Gameplay  >  Cheats";
    TextOutA(dc, lx + sw + P(12), ly + (P(kOptTabH) - text_h(dc)) / 2, from, (int)strlen(from));
    // the rows
    const int foot_top = ly + lh - P(kOptFootH), help_top = foot_top - cheat_help_box;
    const int body_l = lx + P(10), body_r = lx + lw - P(10), body_top = ly + P(kOptTabH) + P(6), body_bottom = help_top - P(4);
    cheat_hit.body_top = body_top, cheat_hit.body_bottom = body_bottom;
    HFONT name_f = face(15 * u, 0), name_b = face(15 * u, 2), value_f = face(14 * u, 0), chip_b = face(13 * u, 2),
          head_f = face(10.5f * u, 2), help_f = face(13 * u, 0);
    char clock_text[12], quantity[12];
    snprintf(clock_text, sizeof clock_text, "%02d:00", clock_hour);
    snprintf(quantity, sizeof quantity, "%+d", pouch_delta);
    int y = 0, sel_top = 0, sel_bottom = 0, top_at[kCheatRows] = {}, head_at[kCheatRows];
    for (int k = 0; k < kCheatRows; k++) {
        const int row = cheat_order[k];
        head_at[k] = -1;
        if (!k || strcmp(cheat_group(row), cheat_group(cheat_order[k - 1]))) head_at[k] = y, y += P(kCheatHeadH);
        top_at[k] = y;
        y += P(kOptRowH);
        if (row == cheat_selected) sel_top = top_at[k] - (head_at[k] >= 0 ? P(kCheatHeadH) : 0), sel_bottom = y;
    }
    const int room = body_bottom - body_top;
    scroll_fit(cheat_sc, y, room, body_top, sel_top, sel_bottom);
    HRGN clip = CreateRectRgn(lx, body_top, lx + lw, body_bottom);
    SelectClipRgn(dc, clip);
    for (int k = 0; k < kCheatRows; k++) {
        const int row = cheat_order[k];
        const bool on = row == cheat_selected;
        if (head_at[k] >= 0) {
            SelectObject(dc, head_f);
            SetTextColor(dc, kFaint);
            startup_gdi::spaced(dc, cheat_group(row), body_l + P(12), body_top + head_at[k] - cheat_scroll + P(10), P(2.5f), false);
        }
        const int top = body_top + top_at[k] - cheat_scroll, rh = P(kOptRowH), mid = top + rh / 2 + P(1);
        cheat_hit.top[row] = top, cheat_hit.bottom[row] = top + rh;
        if (on) {
            startup_gdi::round_rect(dc, body_l, top + P(2), body_r - body_l, rh - P(3), P(10), look.edge);
            const int e = std::max(1, P(1.5f));
            startup_gdi::round_rect(dc, body_l + e, top + P(2) + e, body_r - body_l - 2 * e, rh - P(3) - 2 * e, P(9), look.tint);
        } else if (k + 1 < kCheatRows && cheat_order[k + 1] != cheat_selected && head_at[k + 1] < 0) {
            startup_gdi::fill(dc, body_l + P(4), top + rh - 1, body_r - body_l - P(8), std::max(1, P(1)), kRule);
        }
        SelectObject(dc, on ? name_b : name_f);
        SetTextColor(dc, kInk);
        TextOutA(dc, body_l + P(12), mid - text_h(dc) / 2, cheat_labels[row], (int)strlen(cheat_labels[row]));
        const int vr = body_r - P(12);
        cheat_hit.value_r[row] = vr;
        const int kind = cheat_kind(row);
        if (kind == 1) {   // (a switch)
            const bool is_on = row == 0 ? protect_hearts.load() == 1 : row == 1 ? player_flight.load() >= 0 : free_camera.load() != 0;
            const int sw2 = P(36), sh = P(20), sx = vr - sw2, sy = mid - sh / 2, knob = sh - P(6);
            startup_gdi::round_rect(dc, sx, sy, sw2, sh, sh / 2, is_on ? look.strip : kOff);
            startup_gdi::round_rect(dc, is_on ? sx + sw2 - P(3) - knob : sx + P(3), sy + P(3), knob, knob, knob / 2, RGB(255, 255, 255));
            cheat_hit.value_l[row] = sx;
        } else if (kind == 3) {   // (a button for its action: outlined, filled when selected)
            const char* what = row == 2 ? "Mark" : row == 3 ? "Go back" : "Swap age";
            SelectObject(dc, chip_b);
            const int bw = text_w(dc, what) + P(26), bh = P(24), bx = vr - bw;
            startup_gdi::round_rect(dc, bx, mid - bh / 2, bw, bh, bh / 2, look.strip);
            if (!on) startup_gdi::round_rect(dc, bx + P(2), mid - bh / 2 + P(2), bw - P(4), bh - P(4), (bh - P(4)) / 2, kLabel);
            SetTextColor(dc, on ? look.on : look.strip);
            TextOutA(dc, bx + (bw - text_w(dc, what)) / 2, mid - text_h(dc) / 2, what, (int)strlen(what));
            cheat_hit.value_l[row] = bx;
        } else if (kind == 2) {   // (a choice: between arrows when selected)
            std::string t = row == 4 ? clock_text : row == 5 ? pouch_items[pouch_item] : row == 6 ? std::string(quantity) : teleport_names[teleport_choice];
            if (on) {
                SelectObject(dc, chip_b);
                const int pw = text_w(dc, t) + P(24), ph = P(24), px = vr - P(16) - pw;
                cart::triangle(dc, vr - P(5), mid, P(10), 1, look.strip);
                startup_gdi::round_rect(dc, px, mid - ph / 2, pw, ph, ph / 2, look.strip);
                SetTextColor(dc, look.on);
                TextOutA(dc, px + P(12), mid - text_h(dc) / 2, t.c_str(), (int)t.size());
                cart::triangle(dc, px - P(11), mid, P(10), -1, look.strip);
                cheat_hit.value_l[row] = px - P(18);
            } else {
                SelectObject(dc, value_f);
                SetTextColor(dc, kSoft);
                TextOutA(dc, vr - text_w(dc, t), mid - text_h(dc) / 2, t.c_str(), (int)t.size());
                cheat_hit.value_l[row] = vr - text_w(dc, t);
            }
        } else {   // (a page)
            cart::triangle(dc, vr - P(5), mid, P(10), 1, look.strip);
            cheat_hit.value_l[row] = vr - P(20);
        }
    }
    SelectClipRgn(dc, nullptr);
    DeleteObject(clip);
    scroll_bar(dc, cheat_sc, body_r + P(2), u, look.strip);
    {   // the box: what the selected row does, or for a few seconds what its action just did (with a tick)
        startup_gdi::fill(dc, body_l + P(4), help_top + P(2), body_r - body_l - P(8), std::max(1, P(1)), kRule);
        const bool done = !cheat_done.empty() && prof_now() < cheat_done_until;
        startup_gdi::round_rect(dc, body_l + P(6), help_top + P(12), P(3), cheat_help_box - P(20), P(1.5f), look.strip);
        SelectObject(dc, done ? name_b : help_f);
        SetTextColor(dc, done ? look.strip : kHelp);
        const std::string text = done ? "Done: " + cheat_done : cheat_help[cheat_selected];
        wrap_help(dc, text.c_str(), body_r - body_l - P(30), body_l + P(18), help_top + P(10), true);
    }
    // the buttons: what A does on this row, the arrows where they change it, Back
    startup_gdi::fill(dc, lx, foot_top, lw, std::max(1, P(1)), kRule);
    SelectObject(dc, small);
    const int fy = foot_top + P(kOptFootH) / 2;
    int fx = lx + P(16);
    const auto glyph = [&](const char* letter, COLORREF c) {
        const int d = P(19);
        startup_gdi::round_rect(dc, fx, fy - d / 2, d, d, d / 2, c);
        SetTextColor(dc, RGB(255, 255, 255));
        TextOutA(dc, fx + (d - text_w(dc, letter)) / 2, fy - text_h(dc) / 2, letter, (int)strlen(letter));
        fx += d + P(6);
    };
    const auto say = [&](const char* t) {
        SetTextColor(dc, kHelp);
        TextOutA(dc, fx, fy - text_h(dc) / 2, t, (int)strlen(t));
        fx += text_w(dc, t) + P(20);
    };
    const int kind = cheat_kind(cheat_selected);
    if (kind == 2) {
        cart::triangle(dc, fx + P(5), fy, P(10), -1, kYellow), cart::triangle(dc, fx + P(17), fy, P(10), 1, kYellow);
        fx += P(28);
        say(cheat_selected == 5 ? "Choose" : "Change");
    }
    const char* a = cheat_selected == 4 ? "Set the clock" : cheat_selected == 6 ? "Apply" : cheat_selected == 8 ? "Open the list"
                  : cheat_selected == 5 ? nullptr : kind == 1 ? "Switch" : kind == 3 ? "Do it" : "Open";
    if (a) glyph("A", kBlue), say(a);
    glyph("B", kGreen), say("Back");
    SelectObject(dc, old);
    for (HFONT f : {strip_f, small, name_f, name_b, value_f, chip_b, head_f, help_f}) DeleteObject(f);
}
/// The Cheats page's size: the options' scale, its rows, its box and its bottom, or the window's height less a margin
void cheats_size(uint32_t w, uint32_t h, uint32_t* mw, uint32_t* mh) {
    uint32_t ow = 0, oh = 0;
    options_size(w, h, &ow, &oh);   // (the scale: opt_u)
    const float u = opt_u;
    static float measured_at = -1;
    if (measured_at != u) {
        HDC dc = CreateCompatibleDC(nullptr);
        HFONT f = cart::face(13 * u, 2);
        HGDIOBJ old = SelectObject(dc, f);
        int most = 0;
        for (const char* t : cheat_help) most = std::max(most, wrap_help(dc, t, (int)((kOptW - 20 - 20 - 30) * u), 0, 0, false));
        SelectObject(dc, old), DeleteObject(f), DeleteDC(dc);
        cheat_help_box = most + (int)std::lround(20 * u);
        measured_at = u;
    }
    const float rows_h = kCheatRows * kOptRowH + 5 * kCheatHeadH;
    const float full = (kOptTop + kOptTabH + 6 + rows_h + 4 + kOptFootH + 12) * u + (float)cheat_help_box;
    *mw = (uint32_t)std::lround(kOptW * u);
    *mh = (uint32_t)std::min(full, (float)h - 16.0f * u);
}

// ---- (relink) the Teleport and Inventory pages as the Cheats page: the same green cartridge, where they came from and how
// far down the list is at the top, the list under its sections' small headings, the chosen line tinted with what Enter does
// on it (a Go button for a map; for an item what it is now, its count between arrows), what the page does in the box under
// it, the buttons at the bottom. The keys and the choice are menu_tick's as before (teleport_choice, item_choice) ----
constexpr int kListMost = kItems > kTeleports ? kItems : kTeleports;
/// Where paint_list put each item's line and its value (the mouse: menu_tick)
struct ListHit {
    int body_top = 0, body_bottom = 0, n = 0;
    int top[kListMost] = {}, bottom[kListMost] = {}, value_l[kListMost] = {}, value_r[kListMost] = {};
} list_hit;
int list_help_box = 0;

void paint_list(HDC dc, uint32_t w, uint32_t h, bool inv) {
    using namespace cart;
    const float u = opt_u;
    const auto P = [u](float v) { return (int)std::lround(v * u); };
    const int W = (int)w, H = (int)h;
    const TabLook& look = kTabs[1];
    const int n = inv ? kItems : kTeleports, sections = inv ? kItemSections : kTeleportSections;
    const TeleportSection* sec = inv ? item_sections.s : teleport_sections;
    const int choice = inv ? item_choice : teleport_choice;
    Scroller& lsc = list_sc[inv];
    int& scroll = lsc.px;
    SetBkMode(dc, TRANSPARENT);
    startup_gdi::fill(dc, 0, 0, W, H, RGB(0, 0, 0));
    startup_gdi::round_rect(dc, 0, 0, W, H, P(14), kShell);
    for (int x = P(40); x + P(3) < W - P(40); x += P(9)) startup_gdi::fill(dc, x, P(8), P(3), P(6), kGroove);
    const int lx = P(10), lw = W - 2 * lx, ly = P(kOptTop), lh = H - ly - P(12);
    startup_gdi::round_rect(dc, lx, ly, lw, lh, P(9), kLabel);
    // the strip, the way here, and how far down the list is
    const char* title = inv ? "Inventory" : "Teleport";
    HFONT strip_f = face(17 * u, 3), small = face(12.5f * u, 1);
    HGDIOBJ old = SelectObject(dc, strip_f);
    const int sw = text_w(dc, title) + P(28);
    startup_gdi::round_rect(dc, lx, ly, sw, P(kOptTabH) + P(9), P(9), look.strip);
    startup_gdi::fill(dc, lx, ly + P(kOptTabH), sw, P(9), kLabel);
    SetTextColor(dc, look.on);
    TextOutA(dc, lx + P(14), ly + (P(kOptTabH) - text_h(dc)) / 2, title, (int)strlen(title));
    SelectObject(dc, small);
    SetTextColor(dc, kFaint);
    const std::string from = std::string("Cheats  >  ") + title;
    TextOutA(dc, lx + sw + P(12), ly + (P(kOptTabH) - text_h(dc)) / 2, from.c_str(), (int)from.size());
    char count[32];
    snprintf(count, sizeof count, "%d of %d", choice + 1, n);
    TextOutA(dc, lx + lw - P(16) - text_w(dc, count), ly + (P(kOptTabH) - text_h(dc)) / 2, count, (int)strlen(count));
    // the lines: a heading for each section, then its items
    const int foot_top = ly + lh - P(kOptFootH), help_top = foot_top - list_help_box;
    const int body_l = lx + P(10), body_r = lx + lw - P(10), body_top = ly + P(kOptTabH) + P(6), body_bottom = help_top - P(4);
    list_hit.body_top = body_top, list_hit.body_bottom = body_bottom, list_hit.n = n;
    HFONT name_f = face(15 * u, 0), name_b = face(15 * u, 2), value_f = face(14 * u, 0), chip_b = face(13 * u, 2),
          head_f = face(10.5f * u, 2), help_f = face(13 * u, 0), tag_f = face(9.5f * u, 2);
    std::vector<int> top_at(n), head_at(n, -1);
    int y = 0, sel_top = 0, sel_bottom = 0;
    for (int i = 0, s = 0; i < n; i++) {
        while (s < sections && sec[s].first == i) head_at[i] = y, y += P(kCheatHeadH), s++;
        top_at[i] = y;
        y += P(kOptRowH);
        if (i == choice) sel_top = top_at[i] - (head_at[i] >= 0 ? P(kCheatHeadH) : 0), sel_bottom = y;
    }
    const int room = body_bottom - body_top;
    scroll_fit(lsc, y, room, body_top, sel_top, sel_bottom);
    HRGN clip = CreateRectRgn(lx, body_top, lx + lw, body_bottom);
    SelectClipRgn(dc, clip);
    for (int i = 0, s = 0; i < n; i++) {
        const int top = body_top + top_at[i] - scroll, rh = P(kOptRowH), mid = top + rh / 2 + P(1);
        list_hit.top[i] = top, list_hit.bottom[i] = top + rh;
        if (head_at[i] >= 0) {
            while (s + 1 < sections && sec[s + 1].first <= i) s++;
            std::string head = sec[s].name;
            for (char& c : head) c = (char)toupper((unsigned char)c);
            SelectObject(dc, head_f);
            SetTextColor(dc, kFaint);
            startup_gdi::spaced(dc, head, body_l + P(12), body_top + head_at[i] - scroll + P(10), P(2.5f), false);
        }
        if (top + rh < body_top || top > body_bottom) continue;
        const bool on = i == choice;
        if (on) {
            startup_gdi::round_rect(dc, body_l, top + P(2), body_r - body_l, rh - P(3), P(10), look.edge);
            const int e = std::max(1, P(1.5f));
            startup_gdi::round_rect(dc, body_l + e, top + P(2) + e, body_r - body_l - 2 * e, rh - P(3) - 2 * e, P(9), look.tint);
        } else if (i + 1 < n && i + 1 != choice && head_at[i + 1] < 0) {
            startup_gdi::fill(dc, body_l + P(4), top + rh - 1, body_r - body_l - P(8), std::max(1, P(1)), kRule);
        }
        const char* label = inv ? item_names[i] : teleport_names[i];
        SelectObject(dc, on ? name_b : name_f);
        SetTextColor(dc, kInk);
        TextOutA(dc, body_l + P(12), mid - text_h(dc) / 2, label, (int)strlen(label));
        const int after = body_l + P(12) + text_w(dc, label);
        const int vr = body_r - P(12);
        list_hit.value_r[i] = vr, list_hit.value_l[i] = vr;
        if (!inv) {   // (a map: adult only when it is; on the chosen one, Go)
            if (teleport_notes[i]) {
                SelectObject(dc, tag_f);
                const int tw = text_w(dc, teleport_notes[i]) + P(10), th = text_h(dc) + P(2);
                startup_gdi::round_rect(dc, after + P(8), mid - th / 2, tw, th, th / 2, kTag);
                SetTextColor(dc, kTagInk);
                TextOutA(dc, after + P(13), mid - th / 2 + P(1), teleport_notes[i], (int)strlen(teleport_notes[i]));
            }
            if (on) {
                SelectObject(dc, chip_b);
                const int bw = text_w(dc, "Go") + P(30), bh = P(24), bx = vr - bw;
                startup_gdi::round_rect(dc, bx, mid - bh / 2, bw, bh, bh / 2, look.strip);
                SetTextColor(dc, look.on);
                TextOutA(dc, bx + (bw - text_w(dc, "Go")) / 2, mid - text_h(dc) / 2, "Go", 2);
                list_hit.value_l[i] = bx;
            } else {
                cart::triangle(dc, vr - P(5), mid, P(10), 1, kOff);
            }
            continue;
        }
        // (an item: its count between arrows when it has one and is chosen; else Have or Not yet; ? until the game says)
        const int have = item_counts[i].load(std::memory_order_relaxed);
        if (have < 0) {
            SelectObject(dc, value_f);
            SetTextColor(dc, kFaint);
            TextOutA(dc, vr - text_w(dc, "?"), mid - text_h(dc) / 2, "?", 1);
            list_hit.value_l[i] = vr - text_w(dc, "?");
        } else if (item_kinds[i]) {
            char v[16];
            snprintf(v, sizeof v, "%d", have);
            if (on) {
                SelectObject(dc, chip_b);
                const int pw = std::max(text_w(dc, v) + P(24), P(44)), ph = P(24), px = vr - P(16) - pw;
                cart::triangle(dc, vr - P(5), mid, P(10), 1, look.strip);
                startup_gdi::round_rect(dc, px, mid - ph / 2, pw, ph, ph / 2, look.strip);
                SetTextColor(dc, look.on);
                TextOutA(dc, px + (pw - text_w(dc, v)) / 2, mid - text_h(dc) / 2, v, (int)strlen(v));
                cart::triangle(dc, px - P(11), mid, P(10), -1, look.strip);
                list_hit.value_l[i] = px - P(18);
            } else {
                SelectObject(dc, value_f);
                SetTextColor(dc, have ? kInk : kFaint);
                TextOutA(dc, vr - text_w(dc, v), mid - text_h(dc) / 2, v, (int)strlen(v));
                list_hit.value_l[i] = vr - text_w(dc, v);
            }
        } else {
            const char* t = have > 0 ? "Have" : "Not yet";
            SelectObject(dc, chip_b);
            const int pw = text_w(dc, t) + P(22), ph = P(22), px = vr - pw;
            if (have > 0) startup_gdi::round_rect(dc, px, mid - ph / 2, pw, ph, ph / 2, look.strip);
            else {
                startup_gdi::round_rect(dc, px, mid - ph / 2, pw, ph, ph / 2, kOff);
                startup_gdi::round_rect(dc, px + P(1.5f), mid - ph / 2 + P(1.5f), pw - P(3), ph - P(3), (ph - P(3)) / 2, on ? look.tint : kLabel);
            }
            SetTextColor(dc, have > 0 ? look.on : kSoft);
            TextOutA(dc, px + (pw - text_w(dc, t)) / 2, mid - text_h(dc) / 2, t, (int)strlen(t));
            list_hit.value_l[i] = px;
        }
    }
    SelectClipRgn(dc, nullptr);
    DeleteObject(clip);
    scroll_bar(dc, lsc, body_r + P(2), u, look.strip);
    {   // the box: what the page does with the chosen line
        startup_gdi::fill(dc, body_l + P(4), help_top + P(2), body_r - body_l - P(8), std::max(1, P(1)), kRule);
        startup_gdi::round_rect(dc, body_l + P(6), help_top + P(12), P(3), list_help_box - P(20), P(1.5f), look.strip);
        SelectObject(dc, help_f);
        SetTextColor(dc, kHelp);
        const char* text = !inv ? (teleport_notes[choice] ? "Link travels through the game's own map change, in his current age. This "
                                                            "place is there for the adult only."
                                                          : "Link travels through the game's own map change, in his current age, and "
                                                            "arrives at the map's own entrance.")
                         : item_kinds[choice] ? "Enter unlocks it or takes it away. Left and Right change how many you carry, up "
                                                "to what your bag holds."
                                              : "Enter unlocks it or takes it away again: equipment, items, upgrades and quest items.";
        wrap_help(dc, text, body_r - body_l - P(30), body_l + P(18), help_top + P(10), true);
    }
    // the buttons
    startup_gdi::fill(dc, lx, foot_top, lw, std::max(1, P(1)), kRule);
    SelectObject(dc, small);
    const int fy = foot_top + P(kOptFootH) / 2;
    int fx = lx + P(16);
    const auto glyph = [&](const char* letter, COLORREF c, bool round) {
        SetTextColor(dc, RGB(255, 255, 255));
        const int d = P(19), gw = round ? d : text_w(dc, letter) + P(10);
        startup_gdi::round_rect(dc, fx, fy - d / 2, gw, d, round ? d / 2 : P(5), c);
        TextOutA(dc, fx + (gw - text_w(dc, letter)) / 2, fy - text_h(dc) / 2, letter, (int)strlen(letter));
        fx += gw + P(6);
    };
    const auto say = [&](const char* t) {
        SetTextColor(dc, kHelp);
        TextOutA(dc, fx, fy - text_h(dc) / 2, t, (int)strlen(t));
        fx += text_w(dc, t) + P(20);
    };
    if (inv && item_kinds[choice]) {
        cart::triangle(dc, fx + P(5), fy, P(10), -1, kYellow), cart::triangle(dc, fx + P(17), fy, P(10), 1, kYellow);
        fx += P(28);
        say("How many");
    }
    glyph("A", kBlue, true), say(inv ? (item_counts[choice].load() > 0 ? "Take away" : "Unlock") : "Go there");
    if (!inv) {
        cart::triangle(dc, fx + P(5), fy, P(10), -1, kYellow), cart::triangle(dc, fx + P(17), fy, P(10), 1, kYellow);
        fx += P(28);
        say("Section");
    } else glyph("LB RB", kFaint, false), say("Section");
    glyph("B", kGreen, true), say("Back");
    SelectObject(dc, old);
    for (HFONT f : {strip_f, small, name_f, name_b, value_f, chip_b, head_f, help_f, tag_f}) DeleteObject(f);
}
/// The list pages' size: the options' scale, their rows as the window allows (they scroll), the box and the bottom
void list_size(uint32_t w, uint32_t h, bool inv, uint32_t* mw, uint32_t* mh) {
    uint32_t ow = 0, oh = 0;
    options_size(w, h, &ow, &oh);   // (the scale: opt_u)
    const float u = opt_u;
    list_help_box = (int)std::lround(58 * u);   // (two lines of help and the margins)
    const int n = inv ? kItems : kTeleports, sections = inv ? kItemSections : kTeleportSections;
    const float rows_h = n * kOptRowH + sections * kCheatHeadH;
    const float full = (kOptTop + kOptTabH + 6 + rows_h + 4 + kOptFootH + 12) * u + (float)list_help_box;
    *mw = (uint32_t)std::lround(kOptW * u);
    *mh = (uint32_t)std::min(full, (float)h - 16.0f * u);
}

// ---- (relink) the Travel page as the Cheats page's cartridge, wider: at the top the way here, the age the places are for
// (Child | Adult) and how many have pictures; on the left the places by area, the chosen one tinted, a dot on those with their
// picture; on the right the chosen place's picture, its name, its area and age, and Go. The keys are menu_tick's as before;
// the mouse (travel_mouse) chooses with a click and travels with a second one, or with Go ----
constexpr float kTravelW = 1000, kTravelListR = 400, kTravelCardL = 420;
struct TravelHit {
    int body_top = 0, body_bottom = 0, list_r = 0;
    std::vector<int> top, bottom;
    int age_t = 0, age_b = 0;
    int go_l = 0, go_r = 0, go_t = 0, go_b = 0;
} travel_hit;
int& travel_px_scroll = travel_sc.px;

void paint_travel_cart(HDC dc, uint32_t w, uint32_t h) {
    using namespace cart;
    travel_list();
    const float u = opt_u;
    const auto P = [u](float v) { return (int)std::lround(v * u); };
    const int W = (int)w, H = (int)h;
    const TabLook& look = kTabs[1];
    const int n = (int)travel.place.size(), sections = (int)travel.areas.size();
    SetBkMode(dc, TRANSPARENT);
    startup_gdi::fill(dc, 0, 0, W, H, RGB(0, 0, 0));
    startup_gdi::round_rect(dc, 0, 0, W, H, P(14), kShell);
    for (int x = P(40); x + P(3) < W - P(40); x += P(9)) startup_gdi::fill(dc, x, P(8), P(3), P(6), kGroove);
    const int lx = P(10), lw = W - 2 * lx, ly = P(kOptTop), lh = H - ly - P(12);
    startup_gdi::round_rect(dc, lx, ly, lw, lh, P(9), kLabel);
    // the strip, the way here, the ages, how many places and pictures
    HFONT strip_f = face(17 * u, 3), small = face(12.5f * u, 1), chip_b = face(13 * u, 2), chip_f = face(13 * u, 0);
    HGDIOBJ old = SelectObject(dc, strip_f);
    const int sw = text_w(dc, "Travel") + P(28), mid_top = ly + P(kOptTabH) / 2;
    startup_gdi::round_rect(dc, lx, ly, sw, P(kOptTabH) + P(9), P(9), look.strip);
    startup_gdi::fill(dc, lx, ly + P(kOptTabH), sw, P(9), kLabel);
    SetTextColor(dc, look.on);
    TextOutA(dc, lx + sw / 2 - text_w(dc, "Travel") / 2, mid_top - text_h(dc) / 2, "Travel", 6);
    SelectObject(dc, small);
    SetTextColor(dc, kFaint);
    const char* from = "Cheats  >  Travel";
    TextOutA(dc, lx + sw + P(12), mid_top - text_h(dc) / 2, from, (int)strlen(from));
    {   // (Child | Adult: a pill each, the listed one filled; Tab or X on a pad changes it)
        int x = lx + sw + P(24) + text_w(dc, from);
        const char* const ages[2] = {"Child", "Adult"};
        const int bh = P(24);
        travel_hit.age_t = mid_top - bh / 2, travel_hit.age_b = mid_top + bh / 2;
        for (int a = 0; a < 2; a++) {
            const bool on = a == travel.adult;
            SelectObject(dc, on ? chip_b : chip_f);
            const int bw = text_w(dc, ages[a]) + P(24);
            startup_gdi::round_rect(dc, x, mid_top - bh / 2, bw, bh, bh / 2, on ? look.strip : kRule);
            SetTextColor(dc, on ? look.on : kSoft);
            TextOutA(dc, x + (bw - text_w(dc, ages[a])) / 2, mid_top - text_h(dc) / 2, ages[a], (int)strlen(ages[a]));
            travel_age_l[a] = x, travel_age_r[a] = x + bw;
            x += bw + P(4);
        }
    }
    int with = 0;
    for (int i = 0; i < n; i++) with += travel_has_picture(travel.place[i]);
    char count[64];
    snprintf(count, sizeof count, "%d of %d  \xB7  %d with pictures", n ? travel_choice + 1 : 0, n, with);
    SelectObject(dc, small);
    SetTextColor(dc, kFaint);
    TextOutA(dc, lx + lw - P(16) - text_w(dc, count), mid_top - text_h(dc) / 2, count, (int)strlen(count));
    // the places, by area
    const int foot_top = ly + lh - P(kOptFootH);
    const int body_l = lx + P(10), list_r = lx + P(kTravelListR), body_top = ly + P(kOptTabH) + P(6), body_bottom = foot_top - P(6);
    travel_hit.body_top = body_top, travel_hit.body_bottom = body_bottom, travel_hit.list_r = list_r;
    travel_hit.top.assign(n, 0), travel_hit.bottom.assign(n, 0);
    HFONT name_f = face(15 * u, 0), name_b = face(15 * u, 2), head_f = face(10.5f * u, 2), help_f = face(13 * u, 0),
          big = face(21 * u, 3), go_f = face(15 * u, 3);
    std::vector<int> top_at(n), head_at(n, -1);
    int y = 0, sel_top = 0, sel_bottom = 0;
    for (int i = 0, s = 0; i < n; i++) {
        while (s < sections && travel.areas[s].first == i) head_at[i] = y, y += P(kCheatHeadH), s++;
        top_at[i] = y;
        y += P(kOptRowH);
        if (i == travel_choice) sel_top = top_at[i] - (head_at[i] >= 0 ? P(kCheatHeadH) : 0), sel_bottom = y;
    }
    const int room = body_bottom - body_top;
    scroll_fit(travel_sc, y, room, body_top, sel_top, sel_bottom);
    HRGN clip = CreateRectRgn(lx, body_top, list_r + P(10), body_bottom);
    SelectClipRgn(dc, clip);
    for (int i = 0, s = 0; i < n; i++) {
        const int top = body_top + top_at[i] - travel_px_scroll, rh = P(kOptRowH), mid = top + rh / 2 + P(1);
        travel_hit.top[i] = top, travel_hit.bottom[i] = top + rh;
        if (head_at[i] >= 0) {
            while (s + 1 < sections && travel.areas[s + 1].first <= i) s++;
            std::string head = travel.areas[s].name;
            for (char& c : head) c = (char)toupper((unsigned char)c);
            SelectObject(dc, head_f);
            SetTextColor(dc, kFaint);
            startup_gdi::spaced(dc, head, body_l + P(12), body_top + head_at[i] - travel_px_scroll + P(10), P(2.5f), false);
        }
        if (top + rh < body_top || top > body_bottom) continue;
        const bool on = i == travel_choice;
        if (on) {
            startup_gdi::round_rect(dc, body_l, top + P(2), list_r - body_l, rh - P(3), P(10), look.edge);
            const int e = std::max(1, P(1.5f));
            startup_gdi::round_rect(dc, body_l + e, top + P(2) + e, list_r - body_l - 2 * e, rh - P(3) - 2 * e, P(9), look.tint);
        } else if (i + 1 < n && i + 1 != travel_choice && head_at[i + 1] < 0) {
            startup_gdi::fill(dc, body_l + P(4), top + rh - 1, list_r - body_l - P(8), std::max(1, P(1)), kRule);
        }
        const char* label = oot_tp_places[travel.place[i]].label;
        SelectObject(dc, on ? name_b : name_f);
        SetTextColor(dc, kInk);
        std::string t = label;
        const int room_w = list_r - body_l - P(46);
        if (text_w(dc, t) > room_w) {
            while (t.size() > 1 && text_w(dc, t + "...") > room_w) t.pop_back();
            t += "...";
        }
        TextOutA(dc, body_l + P(12), mid - text_h(dc) / 2, t.c_str(), (int)t.size());
        if (travel_has_picture(travel.place[i])) {   // (a dot: this place has its picture)
            const int d = P(8);
            startup_gdi::round_rect(dc, list_r - P(16) - d, mid - d / 2, d, d, d / 2, look.strip);
        }
    }
    SelectClipRgn(dc, nullptr);
    DeleteObject(clip);
    scroll_bar(dc, travel_sc, list_r + P(2), u, look.strip);
    // the card: the picture, the place, Go
    const int cl = lx + P(kTravelCardL), cr = lx + lw - P(16), cw = cr - cl, ph = cw * 9 / 16, py = body_top + P(6);
    const int place = n ? travel.place[travel_choice] : -1;
    TeleportPicture pic;
    bool reading = false;
    const bool have = travel_picture(place, &pic, &reading);
    startup_gdi::round_rect(dc, cl - P(3), py - P(3), cw + P(6), ph + P(6), P(10), kShell);   // (the picture in a frame of the shell)
    GdiFlush();
    if (have && paint_bits && pic.w && pic.h) {
        for (int yy = 0; yy < ph && py + yy < H; yy++) {
            const uint32_t sy = (uint32_t)((uint64_t)yy * pic.h / ph);
            uint32_t* row = paint_bits + (size_t)(py + yy) * W;
            for (int xx = 0; xx < cw && cl + xx < W; xx++) {
                const unsigned char* q = &pic.rgba[((size_t)sy * pic.w + (uint32_t)((uint64_t)xx * pic.w / cw)) * 4];
                row[cl + xx] = (uint32_t)q[0] << 16 | (uint32_t)q[1] << 8 | q[2];
            }
        }
    } else {   // (none yet: soft stripes, and how one comes)
        for (int yy = 0; yy < ph && paint_bits && py + yy < H; yy++) {
            uint32_t* row = paint_bits + (size_t)(py + yy) * W;
            for (int xx = 0; xx < cw && cl + xx < W; xx++) row[cl + xx] = ((xx + yy) / P(14)) % 2 ? 0xE2DBCB : 0xE8E2D4;
        }
        SelectObject(dc, help_f);
        SetTextColor(dc, kSoft);
        const char* l1 = reading ? "Reading the picture..." : "No picture yet";
        const char* l2 = reading ? "" : "Stay here a few seconds with the menu closed.";
        TextOutA(dc, cl + (cw - text_w(dc, l1)) / 2, py + ph / 2 - text_h(dc), l1, (int)strlen(l1));
        TextOutA(dc, cl + (cw - text_w(dc, l2)) / 2, py + ph / 2 + P(2), l2, (int)strlen(l2));
    }
    if (place >= 0) {
        const OotTpPlace& p = oot_tp_places[place];
        int ty = py + ph + P(16);
        SelectObject(dc, big);
        SetTextColor(dc, kInk);
        TextOutA(dc, cl, ty, p.label, (int)strlen(p.label));
        ty += text_h(dc) + P(2);
        const char* area = "";
        for (const TeleportSection& a : travel.areas)
            if (a.first <= travel_choice) area = a.name;
        char where[128];
        const bool other = link_age.load(std::memory_order_relaxed) >= 0 && link_age.load(std::memory_order_relaxed) != travel.adult;
        snprintf(where, sizeof where, "%s  \xB7  %s%s", area, travel.adult ? "adult" : "child", other ? ": Link becomes that age" : "");
        SelectObject(dc, help_f);
        SetTextColor(dc, kSoft);
        TextOutA(dc, cl, ty, where, (int)strlen(where));
        ty += text_h(dc) + P(8);
        SetTextColor(dc, kHelp);
        ty += wrap_help(dc, "Link arrives here through the game's own map change, as when he walks in. A dot in the list: a place "
                            "with its picture.", cw - P(130), cl, ty, true);
        // Go, at the card's bottom right
        SelectObject(dc, go_f);
        const int bw = text_w(dc, "Go there") + P(40), bh = P(38), bx = cr - bw, by = std::max(py + ph + P(16), body_bottom - bh);
        startup_gdi::round_rect(dc, bx, by + P(3), bw, bh, P(12), RGB(0x2C, 0x64, 0x38));
        startup_gdi::round_rect(dc, bx, by, bw, bh, P(12), look.strip);
        SetTextColor(dc, look.on);
        TextOutA(dc, bx + (bw - text_w(dc, "Go there")) / 2, by + (bh - text_h(dc)) / 2, "Go there", 8);
        travel_hit.go_l = bx, travel_hit.go_r = bx + bw, travel_hit.go_t = by, travel_hit.go_b = by + bh + P(3);
    } else travel_hit.go_l = travel_hit.go_r = 0;
    // the buttons
    startup_gdi::fill(dc, lx, foot_top, lw, std::max(1, P(1)), kRule);
    SelectObject(dc, small);
    const int fy = foot_top + P(kOptFootH) / 2;
    int fx = lx + P(16);
    const auto glyph = [&](const char* letter, COLORREF c, bool round) {
        SetTextColor(dc, RGB(255, 255, 255));
        const int d = P(19), gw = round ? d : text_w(dc, letter) + P(10);
        startup_gdi::round_rect(dc, fx, fy - d / 2, gw, d, round ? d / 2 : P(5), c);
        TextOutA(dc, fx + (gw - text_w(dc, letter)) / 2, fy - text_h(dc) / 2, letter, (int)strlen(letter));
        fx += gw + P(6);
    };
    const auto say = [&](const char* t) {
        SetTextColor(dc, kHelp);
        TextOutA(dc, fx, fy - text_h(dc) / 2, t, (int)strlen(t));
        fx += text_w(dc, t) + P(20);
    };
    glyph("A", kBlue, true), say("Go there");
    cart::triangle(dc, fx + P(5), fy, P(10), -1, kYellow), cart::triangle(dc, fx + P(17), fy, P(10), 1, kYellow);
    fx += P(28);
    say("Area");
    glyph("X", kFaint, true), say("Child / Adult");
    glyph("Y", kYellow, true), say("Picture again");
    glyph("B", kGreen, true), say("Back");
    SelectObject(dc, old);
    for (HFONT f : {strip_f, small, chip_b, chip_f, name_f, name_b, head_f, help_f, big, go_f}) DeleteObject(f);
}
/// The Travel page's size: the options' scale, wider for the picture, as tall as the window allows (the list scrolls)
void travel_size(uint32_t w, uint32_t h, uint32_t* mw, uint32_t* mh) {
    uint32_t ow = 0, oh = 0;
    options_size(w, h, &ow, &oh);   // (the scale: opt_u)
    const float u = opt_u;
    *mw = (uint32_t)std::min((float)kTravelW * u, (float)w - 32.0f);
    *mh = (uint32_t)std::min((kOptTop + kOptTabH + 6 + 560 + kOptFootH + 12) * u, (float)h - 16.0f * u);
}
/// The mouse over the Travel page: a place clicked is chosen, clicked again it is travelled to, as Go is; the wheel moves the
/// choice (stopping at the ends); a right click goes back
void travel_mouse(const MkwVkPointer& ptr, int mx, int my, int mw, int mh, bool moved, int n, bool* press, bool& changed) {
    const bool inside = mx >= 0 && my >= 0 && mx < mw && my < mh;
    MkwVkPointer sp = ptr;
    if (mx >= travel_hit.list_r + (int)std::lround(16 * opt_u)) sp.wheel = 0;   // (the wheel scrolls the list when over it)
    const bool on_bar = scroll_mouse(travel_sc, sp, mx, my, mw, mh, opt_u, changed);
    int over = -1;
    if (inside && !on_bar && mx < travel_hit.list_r && my >= travel_hit.body_top && my < travel_hit.body_bottom)
        for (int i = 0; i < n && i < (int)travel_hit.top.size(); i++)
            if (my >= travel_hit.top[i] && my < travel_hit.bottom[i]) over = i;
    const bool on_go = inside && mx >= travel_hit.go_l && mx < travel_hit.go_r && my >= travel_hit.go_t && my < travel_hit.go_b;
    if ((ptr.pressed & HOS_MOUSE_LEFT) && over >= 0) {
        if (over == travel_choice) press[7] = true;
        else travel_choice = over, changed = true;
    } else if ((ptr.pressed & HOS_MOUSE_LEFT) && on_go) press[7] = true;
    (void)moved;
    if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;
}

// ---- (relink) the Controls page (the bindings) as the Controls tab's blue cartridge: the way here and the layout preset at
// the top, the columns' names, the Switch's buttons under small headings (their short names as round marks), the chosen row
// tinted with its chosen binding filled (yellow while it waits for the new input), Reset to defaults as a button, what Enter
// and Delete do in the box under the rows. The keys and the choice are menu_tick's (csel, ccol, listening) ----
const char* control_group(int i) {
    return i < 4 ? "FACE BUTTONS" : i < 8 ? "SHOULDERS" : i < 10 ? "MENU" : i < 12 ? "STICKS" : i < 16 ? "D-PAD" : i < 20 ? "MOVE (KEYS)"
         : i < kControls ? "CAMERA (KEYS)" : "";
}
struct ControlsHit {
    int body_top = 0, body_bottom = 0;
    int top[kControls + 1] = {}, bottom[kControls + 1] = {};
    int pad_l = 0, pad_r = 0, key_l = 0, key_r = 0, reset_l = 0, reset_r = 0;
} controls_hit;
int& controls_scroll = controls_sc.px;
constexpr float kCtlPadX = 250, kCtlKeyX = 455, kCtlHelpH = 58;

void paint_controls_cart(HDC dc, uint32_t w, uint32_t h) {
    using namespace cart;
    const float u = opt_u;
    const auto P = [u](float v) { return (int)std::lround(v * u); };
    const int W = (int)w, H = (int)h;
    const TabLook& look = kTabs[2];
    SetBkMode(dc, TRANSPARENT);
    startup_gdi::fill(dc, 0, 0, W, H, RGB(0, 0, 0));
    startup_gdi::round_rect(dc, 0, 0, W, H, P(14), kShell);
    for (int x = P(40); x + P(3) < W - P(40); x += P(9)) startup_gdi::fill(dc, x, P(8), P(3), P(6), kGroove);
    const int lx = P(10), lw = W - 2 * lx, ly = P(kOptTop), lh = H - ly - P(12);
    startup_gdi::round_rect(dc, lx, ly, lw, lh, P(9), kLabel);
    HFONT strip_f = face(17 * u, 3), small = face(12.5f * u, 1), name_f = face(15 * u, 0), name_b = face(15 * u, 2),
          cell_f = face(14 * u, 0), cell_b = face(13.5f * u, 2), head_f = face(10.5f * u, 2), help_f = face(13 * u, 0), mark_f = face(12 * u, 3);
    HGDIOBJ old = SelectObject(dc, strip_f);
    const int sw = text_w(dc, "Controls") + P(28), mid_top = ly + P(kOptTabH) / 2;
    startup_gdi::round_rect(dc, lx, ly, sw, P(kOptTabH) + P(9), P(9), look.strip);
    startup_gdi::fill(dc, lx, ly + P(kOptTabH), sw, P(9), kLabel);
    SetTextColor(dc, look.on);
    TextOutA(dc, lx + P(14), mid_top - text_h(dc) / 2, "Controls", 8);
    SelectObject(dc, small);
    SetTextColor(dc, kFaint);
    const char* from = "Controls  >  Buttons";
    TextOutA(dc, lx + sw + P(12), mid_top - text_h(dc) / 2, from, (int)strlen(from));
    const std::string layout = std::string("Layout: ") + tx_pad_layout(opts.pad_layout);
    TextOutA(dc, lx + lw - P(16) - text_w(dc, layout), mid_top - text_h(dc) / 2, layout.c_str(), (int)layout.size());
    // the columns' names
    const int body_l = lx + P(10), body_r = lx + lw - P(10), pad_x = lx + P(kCtlPadX), key_x = lx + P(kCtlKeyX);
    const int cols_y = ly + P(kOptTabH) + P(8);
    SelectObject(dc, head_f);
    SetTextColor(dc, kFaint);
    startup_gdi::spaced(dc, "BUTTON", body_l + P(12), cols_y, P(2.5f), false);
    startup_gdi::spaced(dc, "GAMEPAD", pad_x, cols_y, P(2.5f), false);
    startup_gdi::spaced(dc, "KEYBOARD", key_x, cols_y, P(2.5f), false);
    controls_hit.pad_l = pad_x - P(10), controls_hit.pad_r = key_x - P(10), controls_hit.key_l = key_x - P(10), controls_hit.key_r = body_r;
    // the rows
    const int foot_top = ly + lh - P(kOptFootH), help_top = foot_top - P(kCtlHelpH);
    const int body_top = cols_y + P(16), body_bottom = help_top - P(4);
    controls_hit.body_top = body_top, controls_hit.body_bottom = body_bottom;
    int y = 0, sel_top = 0, sel_bottom = 0, top_at[kControls + 1], head_at[kControls + 1];
    for (int i = 0; i <= kControls; i++) {
        head_at[i] = -1;
        if (i < kControls && (!i || strcmp(control_group(i), control_group(i - 1)))) head_at[i] = y, y += P(kCheatHeadH);
        if (i == kControls) y += P(8);
        top_at[i] = y;
        y += P(kOptRowH) + (i == kControls ? P(4) : 0);
        if (i == csel) sel_top = top_at[i] - (head_at[i] >= 0 ? P(kCheatHeadH) : 0), sel_bottom = y;
    }
    const int room = body_bottom - body_top;
    scroll_fit(controls_sc, y, room, body_top, sel_top, sel_bottom);
    HRGN clip = CreateRectRgn(lx, body_top, lx + lw, body_bottom);
    SelectClipRgn(dc, clip);
    for (int i = 0; i <= kControls; i++) {
        const int top = body_top + top_at[i] - controls_scroll, rh = P(kOptRowH), mid = top + rh / 2 + P(1);
        controls_hit.top[i] = top, controls_hit.bottom[i] = top + rh;
        if (head_at[i] >= 0) {
            SelectObject(dc, head_f);
            SetTextColor(dc, kFaint);
            startup_gdi::spaced(dc, control_group(i), body_l + P(12), body_top + head_at[i] - controls_scroll + P(10), P(2.5f), false);
        }
        if (top + rh < body_top || top > body_bottom) continue;
        const bool on = i == csel;
        if (i == kControls) {   // (Reset to defaults: a button, red outlined, filled when chosen)
            const char* t = "Reset to defaults";
            SelectObject(dc, cell_b);
            const int bw = text_w(dc, t) + P(30), bh = P(26), bx = body_l + P(8);
            const TabLook& red = kTabs[0];
            startup_gdi::round_rect(dc, bx, mid - bh / 2, bw, bh, bh / 2, red.strip);
            if (!on) startup_gdi::round_rect(dc, bx + P(2), mid - bh / 2 + P(2), bw - P(4), bh - P(4), (bh - P(4)) / 2, kLabel);
            SetTextColor(dc, on ? red.on : red.strip);
            TextOutA(dc, bx + (bw - text_w(dc, t)) / 2, mid - text_h(dc) / 2, t, (int)strlen(t));
            controls_hit.reset_l = bx, controls_hit.reset_r = bx + bw;
            continue;
        }
        if (on) {
            startup_gdi::round_rect(dc, body_l, top + P(2), body_r - body_l, rh - P(3), P(10), look.edge);
            const int e = std::max(1, P(1.5f));
            startup_gdi::round_rect(dc, body_l + e, top + P(2) + e, body_r - body_l - 2 * e, rh - P(3) - 2 * e, P(9), look.tint);
        } else if (i + 1 < kControls && i + 1 != csel && head_at[i + 1] < 0) {
            startup_gdi::fill(dc, body_l + P(4), top + rh - 1, body_r - body_l - P(8), std::max(1, P(1)), kRule);
        }
        // the Switch's button: its short name in a round mark, a longer one as words
        const Control& c = controls[i];
        if (strlen(c.name) <= 2) {
            SelectObject(dc, mark_f);
            const int mw2 = std::max(P(24), text_w(dc, c.name) + P(14)), mh2 = P(24);
            startup_gdi::round_rect(dc, body_l + P(10), mid - mh2 / 2, mw2, mh2, mh2 / 2, kInk);
            SetTextColor(dc, kLabel);
            TextOutA(dc, body_l + P(10) + (mw2 - text_w(dc, c.name)) / 2, mid - text_h(dc) / 2, c.name, (int)strlen(c.name));
        } else {
            SelectObject(dc, on ? name_b : name_f);
            SetTextColor(dc, kInk);
            TextOutA(dc, body_l + P(12), mid - text_h(dc) / 2, c.name, (int)strlen(c.name));
        }
        // its two bindings
        for (int col = 0; col < 2; col++) {
            const int x = col ? key_x : pad_x;
            std::string v = col == 0 ? (keys_only(i) ? "-" : pad_name(host->binding(HOS_BIND_PAD, c.button, 0))) : keys_text(c.button);
            const bool chosen = on && col == ccol;
            if (chosen && listening) v = col == 0 ? "Press a button..." : "Press a key...";
            if (chosen) {
                SelectObject(dc, cell_b);
                const int pw = text_w(dc, v) + P(22), ph = P(24);
                startup_gdi::round_rect(dc, x - P(10), mid - ph / 2, pw, ph, ph / 2, listening ? kYellow : look.strip);
                SetTextColor(dc, listening ? kInk : look.on);
                TextOutA(dc, x + P(1), mid - text_h(dc) / 2, v.c_str(), (int)v.size());
            } else {
                SelectObject(dc, cell_f);
                SetTextColor(dc, col == 0 && keys_only(i) ? kOff : on ? kInk : kSoft);
                TextOutA(dc, x, mid - text_h(dc) / 2, v.c_str(), (int)v.size());
            }
        }
    }
    SelectClipRgn(dc, nullptr);
    DeleteObject(clip);
    scroll_bar(dc, controls_sc, body_r + P(2), u, look.strip);
    {   // the box: what Enter and Delete do here, or what the page waits for
        startup_gdi::fill(dc, body_l + P(4), help_top + P(2), body_r - body_l - P(8), std::max(1, P(1)), kRule);
        startup_gdi::round_rect(dc, body_l + P(6), help_top + P(12), P(3), P(kCtlHelpH) - P(20), P(1.5f), listening ? kYellow : look.strip);
        SelectObject(dc, listening ? name_b : help_f);
        SetTextColor(dc, kHelp);
        std::string text;
        if (listening) text = std::string(listening == 1 ? "Press the gamepad button" : "Press the key") + " for " + controls[csel].name +
                              " now. Esc cancels; it stops by itself after a few seconds.";
        else if (csel == kControls) text = "Puts every gamepad button back as the Layout preset has them.";
        else if (keys_only(csel)) text = "The keyboard's key for this one (the gamepad's stick does it there). Enter binds a new key; Delete clears it.";
        else text = "Enter binds a new one in the chosen column: press the button or the key. Delete clears it. Left and Right change the column.";
        wrap_help(dc, text.c_str(), body_r - body_l - P(30), body_l + P(18), help_top + P(10), true);
    }
    // the buttons
    startup_gdi::fill(dc, lx, foot_top, lw, std::max(1, P(1)), kRule);
    SelectObject(dc, small);
    const int fy = foot_top + P(kOptFootH) / 2;
    int fx = lx + P(16);
    const auto glyph = [&](const char* letter, COLORREF c, bool round) {
        SetTextColor(dc, RGB(255, 255, 255));
        const int d = P(19), gw = round ? d : text_w(dc, letter) + P(10);
        startup_gdi::round_rect(dc, fx, fy - d / 2, gw, d, round ? d / 2 : P(5), c);
        TextOutA(dc, fx + (gw - text_w(dc, letter)) / 2, fy - text_h(dc) / 2, letter, (int)strlen(letter));
        fx += gw + P(6);
    };
    const auto say = [&](const char* t) {
        SetTextColor(dc, kHelp);
        TextOutA(dc, fx, fy - text_h(dc) / 2, t, (int)strlen(t));
        fx += text_w(dc, t) + P(20);
    };
    if (listening) glyph("Esc", kFaint, false), say("Cancel");
    else {
        glyph("A", kBlue, true), say(csel == kControls ? "Reset" : "Bind");
        if (csel < kControls) {
            if (!keys_only(csel)) {
                cart::triangle(dc, fx + P(5), fy, P(10), -1, kYellow), cart::triangle(dc, fx + P(17), fy, P(10), 1, kYellow);
                fx += P(28);
                say("Column");
            }
            glyph("Del", kFaint, false), say("Clear");
        }
        glyph("B", kGreen, true), say("Back");
    }
    SelectObject(dc, old);
    for (HFONT f : {strip_f, small, name_f, name_b, cell_f, cell_b, head_f, help_f, mark_f}) DeleteObject(f);
}
/// The Controls page's size: the options' scale, as tall as its rows or the window allows (they scroll)
void controls_size(uint32_t w, uint32_t h, uint32_t* mw, uint32_t* mh) {
    uint32_t ow = 0, oh = 0;
    options_size(w, h, &ow, &oh);   // (the scale: opt_u)
    const float u = opt_u;
    const float rows_h = (kControls + 1) * kOptRowH + 7 * kCheatHeadH + 12;
    const float full = (kOptTop + kOptTabH + 8 + 16 + rows_h + 4 + kCtlHelpH + kOptFootH + 12) * u;
    *mw = (uint32_t)std::lround(kOptW * u);
    *mh = (uint32_t)std::min(full, (float)h - 16.0f * u);
}

void paint_menu(HDC dc, uint32_t w, uint32_t h, int s) {
    if (page == 0) return paint_options(dc, w, h);   // (relink: the cartridge)
    if (page == 2) return paint_cheats(dc, w, h);
    if (page == 3 || page == 4) return paint_list(dc, w, h, page == 4);
    if (page == 6) return paint_travel_cart(dc, w, h);
    if (page == 1) return paint_controls_cart(dc, w, h);
    SetBkMode(dc, TRANSPARENT);
    HFONT title = font(20 * s, FW_SEMIBOLD), body = font(16 * s, FW_NORMAL);
    HGDIOBJ old = SelectObject(dc, title);
    struct Owned { HFONT f; ~Owned() { DeleteObject(f); } } hints{font(13 * s, FW_NORMAL)};   // (the lines of keys and help: smaller,
                                                                                                // as the options page's)
    // (relink) the cartridge (cart_page), each page in a colour: the Gameplay tab's green for the cheats and their lists,
    // the Controls tab's blue for the controls, yellow for the restart question and red for Quit game's
    const int look = page == 5 ? (ask_quit ? 0 : -1) : page == 1 ? 2 : 1;
    const COLORREF strip = look < 0 ? cart::kYellow : cart::kTabs[look].strip, tint = look < 0 ? cart::kTag : cart::kTabs[look].tint,
                   edge = look < 0 ? RGB(0xE3, 0xC4, 0x74) : cart::kTabs[look].edge, on = look < 0 ? cart::kInk : cart::kTabs[look].on;
    const char* heading = page == 6 ? "Travel" : page == 5 ? (ask_quit ? "Quit game" : "Restart needed") : page == 4 ? "Inventory" : page == 3 ? "Teleport"
                        : page == 2 ? "Cheats" : "Controls";
    cart_page(dc, (int)w, (int)h, s, heading, strip, on);
    {   // (which build, at the top right, so a report can say it without opening the log; the release alone when the whole
        // line would run into the title)
        HFONT small = font(12 * s, FW_NORMAL);
        SelectObject(dc, small);
        SetTextColor(dc, cart::kFaint);
        std::string ver = RELINK_VERSION;
        const int room = (int)w / 2 - 16 * s;
        SIZE sz{};
        GetTextExtentPoint32A(dc, ver.c_str(), (int)ver.size(), &sz);
        if (sz.cx > room && ver.find(" (") != std::string::npos) {
            ver.resize(ver.find(" ("));
            GetTextExtentPoint32A(dc, ver.c_str(), (int)ver.size(), &sz);
        }
        if (sz.cx <= room && page != 6 && page != 3 && page != 4) TextOutA(dc, (int)w - 16 * s - sz.cx, 18 * s, ver.c_str(), (int)ver.size());
        SelectObject(dc, title);
        DeleteObject(small);
    }
    SelectObject(dc, body);
    if (page == 5) {   // (relink) the restart question, as the menu closes with a setting that takes a restart, or Quit game's
        SetTextColor(dc, cart::kInk);
        const std::string ask = ask_quit ? std::string("Quit the game?") : "Restart to apply " + restart_list(" and ") + "?";
        const int asked = wrap_help(dc, ask.c_str(), (int)w - 32 * s, 16 * s, 46 * s, true);   // (one line or two)
        HFONT small = font(14 * s, FW_NORMAL);
        SelectObject(dc, small);
        SetTextColor(dc, cart::kHelp);
        const char* lost = "Progress since your last save is lost.";
        TextOutA(dc, 16 * s, std::min(46 * s + asked + 4 * s, 94 * s), lost, (int)strlen(lost));
        SelectObject(dc, body);
        const char* const labels[2] = {ask_quit ? "Back" : "Later", ask_quit ? "Quit" : "Restart now"};
        for (int i = 0; i < 2; i++) {
            const int x = menu_hit::ask_button_x(i, s), y = menu_hit::kAskButtonY * s;
            const int bw = menu_hit::kAskButtonW * s, bh = menu_hit::kAskButtonH * s;
            const bool sel = i == ask_sel;
            startup_gdi::round_rect(dc, x, y, bw, bh, bh / 2, sel ? strip : cart::kRule);
            SIZE sz{};
            GetTextExtentPoint32A(dc, labels[i], (int)strlen(labels[i]), &sz);
            SetTextColor(dc, sel ? on : cart::kInk);
            TextOutA(dc, x + (bw - (int)sz.cx) / 2, y + (bh - (int)sz.cy) / 2, labels[i], (int)strlen(labels[i]));
        }
        SelectObject(dc, small);
        SetTextColor(dc, cart::kSoft);
        const char* hint = ask_quit ? "Left/Right choose   Enter confirms   Esc: Back" : "Left/Right choose   Enter confirms   Esc: Later";
        TextOutA(dc, 16 * s, 168 * s, hint, (int)strlen(hint));
        SelectObject(dc, old);
        DeleteObject(small);
        DeleteObject(title);
        DeleteObject(body);
        return;
    }
    SelectObject(dc, old);   // (page 0, the options, is paint_options')
    DeleteObject(title);
    DeleteObject(body);
}
char fps_text[48] = "";
char fps_sub[32] = "";
/// The usual overlay: a bold green number with a dark edge, nothing behind it. Hard-edged letters: see-through pixels are copied or not.
void outlined(HDC dc, int x, int y, const char* text, COLORREF colour, int edge) {
    SetTextColor(dc, RGB(6, 8, 6));
    const int n = (int)strlen(text);
    for (int dy = -edge; dy <= edge; dy++)
        for (int dx = -edge; dx <= edge; dx++)
            if (dx || dy) TextOutA(dc, x + dx, y + dy, text, n);
    SetTextColor(dc, colour);
    TextOutA(dc, x, y, text, n);
}
HFONT hard_font(int px, int weight) {
    return CreateFontA(-px, 0, 0, 0, weight, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, NONANTIALIASED_QUALITY,
                       DEFAULT_PITCH, "Segoe UI");
}
/// (relink) The Travel page's notice over the game: a picture was saved
void paint_notice(HDC dc, uint32_t w, uint32_t h, int s) {
    // (relink) a small label of the cartridge: the shell's edge, the cream, a green strip at its left (see-through corners)
    fill(dc, 0, 0, (int)w, (int)h, RGB(0, 0, 0));
    startup_gdi::round_rect(dc, 0, 0, (int)w, (int)h, 8 * s, cart::kShell);
    startup_gdi::round_rect(dc, 3 * s, 3 * s, (int)w - 6 * s, (int)h - 6 * s, 6 * s, cart::kLabel);
    startup_gdi::round_rect(dc, 3 * s, 3 * s, 12 * s, (int)h - 6 * s, 6 * s, cart::kTabs[1].strip);
    fill(dc, 9 * s, 3 * s, 6 * s, (int)h - 6 * s, cart::kLabel);
    SetBkMode(dc, TRANSPARENT);
    HFONT f = font(16 * s, FW_SEMIBOLD);
    HGDIOBJ old = SelectObject(dc, f);
    SetTextColor(dc, cart::kInk);
    SIZE sz{};
    GetTextExtentPoint32A(dc, notice_text, (int)strlen(notice_text), &sz);
    TextOutA(dc, ((int)w - (int)sz.cx) / 2, ((int)h - (int)sz.cy) / 2 + s, notice_text, (int)strlen(notice_text));
    SelectObject(dc, old);
    DeleteObject(f);
}
void paint_fps(HDC dc, uint32_t w, uint32_t h, int s) {
    fill(dc, 0, 0, (int)w, (int)h, RGB(0, 0, 0));
    SetBkMode(dc, TRANSPARENT);
    SetTextAlign(dc, TA_RIGHT | TA_TOP);
    const COLORREF green = RGB(72, 255, 110);
    const int right = (int)w - 2 * s;
    HFONT label = hard_font(13 * s, FW_BOLD), big = hard_font(28 * s, FW_BOLD);
    HGDIOBJ old = SelectObject(dc, label);
    outlined(dc, right, 14 * s, "FPS", green, s);
    SelectObject(dc, big);
    outlined(dc, right - 30 * s, 0, fps_text, green, s);
    if (fps_sub[0]) {
        SelectObject(dc, label);
        outlined(dc, right, 32 * s, fps_sub, RGB(226, 232, 226), s);
    }
    SelectObject(dc, old);
    DeleteObject(label);
    DeleteObject(big);
}
void redraw_menu(int s) {
    if (page == 6) {   // (relink) the Travel page: the list and the picture, as the Cheats page's cartridge
        uint32_t mw = 0, mh = 0;
        travel_size(dev.swap_extent.width, dev.swap_extent.height, &mw, &mh);
        gdi_draw(menu_bmp, mw, mh, paint_menu, s);
        return;
    }
    if (page == 5) {
        gdi_draw(menu_bmp, kMenuWidth * s, (unsigned)menu_hit::kAskHeight * s, paint_menu, s);
        return;
    }
    if (page == 3 || page == 4) {   // (relink: the Teleport and Inventory pages, as the Cheats page)
        uint32_t mw = 0, mh = 0;
        list_size(dev.swap_extent.width, dev.swap_extent.height, page == 4, &mw, &mh);
        gdi_draw(menu_bmp, mw, mh, paint_menu, s);
        return;
    }
    if (page == 2) {   // (relink: the Cheats page, as the options' cartridge)
        uint32_t mw = 0, mh = 0;
        cheats_size(dev.swap_extent.width, dev.swap_extent.height, &mw, &mh);
        gdi_draw(menu_bmp, mw, mh, paint_menu, s);
        return;
    }
    if (page == 1) {   // (relink) the bindings, as the Controls tab's cartridge
        uint32_t mw = 0, mh = 0;
        controls_size(dev.swap_extent.width, dev.swap_extent.height, &mw, &mh);
        gdi_draw(menu_bmp, mw, mh, paint_menu, s);
        return;
    }
    // (relink) the options page: the cartridge (paint_options), its round corners see-through
    uint32_t mw = 0, mh = 0;
    options_size(dev.swap_extent.width, dev.swap_extent.height, &mw, &mh);
    gdi_draw(menu_bmp, mw, mh, paint_menu, s);
}

std::atomic<int64_t> game_frames{0}, shown_frames{0};
int cur_scale = 1;

struct Upload {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    void* map = nullptr;
    uint64_t size = 0, version = 0;
};
Upload ups[4][3];   // (the menu's, the counter's, the start-up screen's, (relink) the Travel notice's; a frame in flight's each)

void copy_over(VkCommandBuffer cmd, VkImage dst, Bitmap& b, Upload& u, int x, int y, uint32_t w, uint32_t h) {
    std::lock_guard lk(bmp_mu);
    // (relink, issue 96) the part inside the window: a bitmap that went past its edge was not drawn at all, and in a 1280x720
    // window the Insert menu (726 lines at a scale of 1 since 0.7.9's rows) never showed
    const int cx0 = std::max(0, -x), cy0 = std::max(0, -y);
    const int cx1 = std::min((int)b.w, (int)w - x), cy1 = std::min((int)b.h, (int)h - y);
    if (!b.w || cx0 >= cx1 || cy0 >= cy1) return;
    const uint64_t size = (uint64_t)b.w * b.h * 4;
    if (u.size < size) {
        // (the slot's own frame is over before the slot is used again: its smaller buffer can go. It stayed for good before, which
        // mattered little for a menu and is the window's size for the start-up screen at each resize)
        if (u.buf) destroy_buffer(u.buf, u.alloc);
        u = {};
        if (!create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, u.buf, u.alloc, &u.map)) return;
        u.size = size, u.version = 0;
    }
    if (u.version != b.version) std::memcpy(u.map, b.px.data(), size), u.version = b.version;
    if (b.keyed) {
        static thread_local std::vector<VkBufferImageCopy> at;
        at.clear();
        for (const auto& r : b.runs) {   // (each run one line of the bitmap: the lines and the parts of lines outside left out)
            const int ry = r.imageOffset.y, rx0 = std::max(r.imageOffset.x, cx0);
            const int rx1 = std::min(r.imageOffset.x + (int)r.imageExtent.width, cx1);
            if (ry < cy0 || ry >= cy1 || rx0 >= rx1) continue;
            VkBufferImageCopy c = r;
            c.bufferOffset += (uint64_t)(rx0 - r.imageOffset.x) * 4;
            c.imageOffset.x = rx0 + x, c.imageOffset.y = ry + y;
            c.imageExtent.width = (uint32_t)(rx1 - rx0);
            at.push_back(c);
        }
        if (!at.empty()) vkCmdCopyBufferToImage(cmd, u.buf, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)at.size(), at.data());
        return;
    }
    VkBufferImageCopy c{};
    c.bufferOffset = ((uint64_t)cy0 * b.w + cx0) * 4;
    c.bufferRowLength = b.w;
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageOffset = {x + cx0, y + cy0, 0};
    c.imageExtent = {(uint32_t)(cx1 - cx0), (uint32_t)(cy1 - cy0), 1};
    vkCmdCopyBufferToImage(cmd, u.buf, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
}

// ---- (relink) the start-up screen (vk_startup_paint.h): the whole window, a picture that moves ----
StartupView startup;
Bitmap startup_bmp;
std::atomic<bool> startup_on{false};
void paint_startup(HDC dc, uint32_t w, uint32_t h, int pct) { paint_startup_view(dc, w, h, pct, startup, paint_bits); }

} // namespace

void startup_screen(const StartupView* view) {
    // (one at a time, and nothing after the end: a painter whose last present waited for a hidden window comes back here once
    // the game has started, and would paint the landscape again for nobody)
    static std::mutex mu;
    static bool ended = false;
    std::lock_guard guard(mu);
    if (!view) {
        startup_on.store(false, std::memory_order_release);
        ended = true;
        // (the picture is the whole window: the kept landscape and the bitmap are 8 MB each at 1080 lines, on PCs short of memory.
        // The three upload buffers go a few presents later, menu_record)
        startup_backdrop = {};
        std::lock_guard lk(bmp_mu);
        startup_bmp.w = startup_bmp.h = 0;
        std::vector<uint8_t>().swap(startup_bmp.px);
        return;
    }
    if (ended) return;
    static uint32_t drawn_w = 0, drawn_h = 0;
    static double drawn_at = -1;
    const uint32_t w = dev.swap_extent.width, h = dev.swap_extent.height;
    StartupView v = *view;
    if (v.version.empty()) v.version = version_text(RELINK_VERSION);   // ("v2.0.5"; "v2.0.5 dev" between releases)
    // (the picture moves: drawn again 30 times a second up to 1080 lines, 15 up to 1440, 10 above. Each drawing is the window's
    // pixels copied three times and sent to the card, and the start's own work has the processor)
    const uint64_t pixels = (uint64_t)w * h;
    // (the ending, outro, each time it is asked, about 60 a second: the work is done then, and the closing must be smooth)
    const double every = v.outro >= 0 ? 0 : pixels > 4200000 ? 0.099 : pixels > 2300000 ? 0.066 : 0.033;
    StartupView still = v, was = startup;
    still.anim = was.anim = 0;
    if (w && h && (w != drawn_w || h != drawn_h || !(still == was) || v.anim < drawn_at || v.anim - drawn_at >= every - 0.004)) {
        startup = v;
        drawn_w = w, drawn_h = h, drawn_at = v.anim;
        gdi_draw(startup_bmp, w, h, paint_startup, 100);
    }
    startup_on.store(true, std::memory_order_release);
}

int hotkey_vk(int index) { return index >= 0 && index < kHotkeyNone ? kHotkeys[index] : 0; }
int hdr_paper_white_nits() {
    static const int env = getenv("HOS_HDR_PAPER_WHITE") && *getenv("HOS_HDR_PAPER_WHITE") ? atoi(getenv("HOS_HDR_PAPER_WHITE")) : 0;
    return std::clamp(env > 0 ? env : opts.hdr_paper_white, 80, 500);
}
void menu_init() {
    static bool done;
    if (done) return;
    done = true;
    for (auto& count : item_counts) count.store(-1, std::memory_order_relaxed);
    bool has_options = false;
    const char* cheats = getenv("HOS_OOT_CHEATS");
    cheats_enabled = !cheats || strcmp(cheats, "0") != 0;   // (relink: shown unless HOS_OOT_CHEATS=0 hides them)
    if (cheats_enabled) {   // (relink) the Travel page's pictures: taken when the page was opened before (its folder is there),
        // and only while the page can be opened: it is on the Cheats page
        std::error_code ec;
        const std::string dir = game_folder_file("teleport_pictures");
        travel_pictures_on.store(!dir.empty() && std::filesystem::is_directory(dir, ec), std::memory_order_relaxed);
    }
    if (cheats_enabled) {
        const char* hearts = getenv("HOS_OOT_PROTECT_HEARTS");
        protect_hearts.store(hearts && !strcmp(hearts, "1"));
    }
    if (FILE* f = fopen(ini_path().c_str(), "r")) {
        has_options = true;
        char line[80];
        bool has_resolution = false;
        bool v2 = false;
        int old_dynres = -1;   // (the row before Resolution: 0, the full scale, is its 100%; 1 is Auto)
        while (fgets(line, sizeof line, f)) {
            char* eq = strchr(line, '=');
            if (!eq) continue;
            *eq = 0;
            for (Row& r : rows) {
                if (!r.value || strcmp(line, ini_key(r))) continue;
                const int v = atoi(eq + 1);
                *r.value = r.value == &opts.output_res ? output_choice(v) : std::min(std::max(v, r.lo), r.hi);
            }
            has_resolution |= !strcmp(line, "resolution");
            v2 |= !strcmp(line, "options_version");
            if (!strcmp(line, "dynamic_resolution")) old_dynres = atoi(eq + 1);
            if (!strcmp(line, "hdr_paper_white")) opts.hdr_paper_white = std::clamp(atoi(eq + 1), 80, 500);   // (relink: HDR)
            if (!strcmp(line, "microphone_device")) {   // (relink: the microphone, by its name)
                mic_want = mic_key(eq + 1);
                while (!mic_want.empty() && (mic_want.back() == '\n' || mic_want.back() == '\r')) mic_want.pop_back();
            }
        }
        fclose(f);
        // (the saved microphone by its name, not its place; the host told the row's choice. Off or the default: no list, no SDL
        // audio at the start)
        if (!mic_want.empty()) refresh_mics();
        else {
            if (opts.mic >= 2) opts.mic = 1;
            mic_tell_host();
        }
        if (!has_resolution && old_dynres >= 0) opts.resolution = old_dynres ? 0 : 1;
        opts.rumble = rumble_step(opts.rumble);   // (a hand-edited 60: 50, on the arrows' steps)
        opts.deadzone_l = std::min(std::max(opts.deadzone_l, 0), 50) / 2 * 2;   // (relink: host/hos_stick.h)
        opts.deadzone_r = std::min(std::max(opts.deadzone_r, 0), 50) / 2 * 2;
        opts.fov = std::min(std::max(opts.fov, 70), 150) / 5 * 5;   // (relink: nx2/oot_fov.py)
        opts.volume = std::min(std::max(opts.volume, 0), 100) / 5 * 5;   // (relink: host/nn_audio.c)
        // (Game speed is Real time by default now: Per frame played at half speed at 30 fps, the "slow motion" players
        // reported. An options file from before kept the old default whether chosen or not: Real time for it too)
        if (!v2) opts.realtime = 1;
    }
    // An opt-in profile for 4 GB cards: keep existing options authoritative, but make a first run usable on
    // low-VRAM hardware without requiring the menu to render first.
    if (!has_options && getenv("RX_VK_LOW_SPEC") && atoi(getenv("RX_VK_LOW_SPEC")) != 0) {
        apply_preset(1);
        opts.framegen = 0;
        opts.upscale = 0;
        opts.sharpness = 0;
    }
    if (opts.preset > 0) sync_preset_budgets(opts.preset);
    if (opts.framegen == 1) opts.framegen = 2;
    if (const char* e = getenv("RX_VK_FRAMEGEN_MULT")) opts.framegen = std::min(std::max(atoi(e), 2), 8);
    if (const char* e = getenv("RX_VK_PACE")) opts.pace = atoi(e) != 0;
    if (const char* e = getenv("RX_VK_UPSCALE"))
        for (Row& r : rows)
            if (r.value == &opts.upscale) *r.value = std::min(std::max(atoi(e), r.lo), r.hi);
    if (const char* e = getenv("HOS_REALTIME_STEP")) opts.realtime = atoi(e) != 0;
    if (const char* e = getenv("HOS_FPS_UNLOCKED")) opts.framerate = atoi(e) != 0;
    if (const char* e = getenv("HOS_DYNRES")) opts.resolution = atoi(e) != 0 ? 0 : 1;
    if (const char* e = getenv("HOS_OUTPUT_RES")) opts.output_res = output_choice(atoi(e));
    restart_init();   // (what this run started with: the rows that take a restart differ from it once changed)
}

bool resolution_auto() { return opts.resolution == 0; }
int resolution_scale() {
    const float s = upscaler_rows::percent(opts.resolution) / 100.0f;
    return s > 0 ? (int)(twelfths(s) * 10000.0f / 12.0f + 0.5f) : 0;   // (the twelfth game_size labels)
}

void menu_set_gpu_count(int count) {
    for (Row& r : rows) {
        if (r.value == &opts.gpu_device) {
            r.hi = count;
            if (opts.gpu_device > count) opts.gpu_device = 0;
        }
    }
    restart_init();
    for (int i = 0; i < kRows; i++)   // (the row's value the game started with, the cards counted)
        if (rows[i].value == &opts.gpu_device) restart_state.restart_value(restart_slot[i], opts.gpu_device);
}

bool menu_is_open() { return open_now.load(std::memory_order_relaxed); }
int menu_take_capture_request() {   // (relink: not while the menu covers the picture)
    const int want = teleport_capture_want.exchange(-1, std::memory_order_relaxed);
    return open_now.load(std::memory_order_relaxed) ? -1 : want;
}
/// (relink) The replay's thread, at a present: the picture's pixels copied and handed to the picture worker, which keeps and
/// writes them. No file is touched here
void menu_store_capture(int place, const uint8_t* rgba, uint32_t w, uint32_t h) {
    if (place < 0 || place >= OOT_TP_PLACES) return;
    PictureJob j;
    j.kind = PictureJob::Store, j.place = place;
    j.pic.w = w, j.pic.h = h;
    j.pic.rgba.assign(rgba, rgba + (size_t)w * h * 4);
    picture_job(std::move(j));
}
bool menu_teleport_captures_on() { return travel_pictures_on.load(std::memory_order_relaxed); }   // (relink)
/// (relink) F12 (menu_tick): the next present's picture is kept. With the menu open too: the picture is the game's, the
/// menu is drawn over it afterwards and is not in it
std::atomic<bool> screenshot_want{false};
bool menu_take_screenshot_request() { return screenshot_want.exchange(false, std::memory_order_relaxed); }
void menu_store_screenshot(std::vector<uint8_t>&& rgba, uint32_t w, uint32_t h) {
    // (a thread of its own: compressing and writing a 3840x2160 picture takes a few hundred ms, never the replay's)
    std::thread([px = std::move(rgba), w, h] {
        const std::string path = screenshot_path(game_folder_file("screenshots"));
        const auto t0 = std::chrono::steady_clock::now();
        const bool saved = screenshot_write(path, px.data(), w, h);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const size_t slash = path.find_last_of("\\/");
        const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
        logf("menu: screenshot %ux%u %s%s (%.0f ms)", w, h, saved ? "written to " : "NOT written: ", path.empty() ? "no game folder" : path.c_str(), ms);
        if (saved) snprintf(notice_text, sizeof notice_text, "Screenshot saved: screenshots\\%s", name.c_str());
        else snprintf(notice_text, sizeof notice_text, "The screenshot could not be saved");
        notice_new.store(true, std::memory_order_release);
        notice_until.store(prof_now() + 2500000000LL, std::memory_order_relaxed);
    }).detach();
}
int menu_heart_protection() { return protect_hearts.load(std::memory_order_relaxed); }
int menu_player_flight() { return cheats_enabled ? player_flight.load(std::memory_order_relaxed) : -1; }
int menu_free_camera() { return free_camera.load(std::memory_order_relaxed); }
int menu_take_cheat_action() { return cheat_action.exchange(0, std::memory_order_relaxed); }

void menu_note_present_sizes(uint32_t picture_w, uint32_t picture_h, uint32_t window_w, uint32_t window_h) {
    const auto c = [](uint32_t v) { return (uint64_t)std::min(v, 0xffffu); };
    present_sizes.store(c(picture_w) << 48 | c(picture_h) << 32 | c(window_w) << 16 | c(window_h), std::memory_order_relaxed);
}
void menu_note_frame(bool game_frame) {
    shown_frames.fetch_add(1, std::memory_order_relaxed);
    if (game_frame) game_frames.fetch_add(1, std::memory_order_relaxed);
}

// The Fullscreen row and F11 are one setting: the row asks the window's thread (host/display.c), F11 shows in the row
void fullscreen_tick() {
    static int applied = 0, settle = 0;
    const HWND main_window = app_window();
    if (!main_window) return;
    static const bool env = getenv("HOS_FULLSCREEN") != nullptr;   // (a test's own choice: the row is left alone)
    if (env) return;
    if (opts.fullscreen != applied) {
        applied = opts.fullscreen;
        settle = 60;
        PostMessageA(main_window, WM_APP + 7, (WPARAM)applied, 0);
        return;
    }
    if (settle > 0) { settle--; return; }
    const int now = !(GetWindowLongA(main_window, GWL_STYLE) & WS_CAPTION);
    if (now != applied) {
        opts.fullscreen = applied = now;
        menu_refresh = true;
        save();
    }
}

/// A rate over the half-second that just ended. The first call only starts that interval: the frames since launch,
/// divided by one second, were shown as the rate the moment the counter was turned on.
bool fps_take(int64_t now, int64_t shown, int64_t game, int64_t* next, int64_t* last_shown, int64_t* last_game,
              double* shown_rate, double* game_rate) {
    if (now < *next) return false;
    const bool have = *next != 0;
    if (have) {
        const double secs = (now - *next + 500000000) / 1e9;
        *shown_rate = (shown - *last_shown) / secs;
        *game_rate = (game - *last_game) / secs;
    }
    *last_shown = shown;
    *last_game = game;
    *next = now + 500000000;
    return have;
}

int menu_item_count(int i) { return i >= 0 && i < kItems ? item_counts[i].load(std::memory_order_relaxed) : -1; }
int menu_link_age() { return link_age.load(std::memory_order_relaxed); }
void menu_option_set(int id, int value) {
    if (id == HOS_PROMPTS_TEX_ID) return prompt_tex_announce(value);   // (relink: the button font's family and bindings, host/prompts.c)
    if (id == OOT_CHEAT_AGE_ID) {   // (relink: the Travel page's list is of Link's age)
        if (link_age.exchange(value, std::memory_order_relaxed) != value && page == 6) menu_refresh = true;
        return;
    }
    if (id == OOT_TP_MAP_ID) return teleport_map_now.store(value, std::memory_order_relaxed);   // (relink: the place nearest Link)
    if (id == OOT_TP_CAPTURE_ID) {   // (relink: Link stayed near a place: its picture wanted when it has none)
        if (value >= 0 && value < OOT_TP_PLACES) {
            PictureJob j;
            j.kind = PictureJob::Arrived, j.place = value;
            picture_job(std::move(j));
        }
        return;
    }
    if (id >= OOT_CHEAT_ITEM_BASE && id < OOT_CHEAT_ITEM_BASE + kItems) {   // (the game side's inventory counts)
        if (item_counts[id - OOT_CHEAT_ITEM_BASE].exchange(value, std::memory_order_relaxed) != value && page == 4)
            menu_refresh = true;
        return;
    }
    if (id == HOS_OPT_CHEAT_FLIGHT) {
        if (cheats_enabled) {
            int expected = value ? 0 : 1;
            player_flight.compare_exchange_strong(expected, value != 0, std::memory_order_relaxed);
        }
        return;
    }
    if (id != 7) return;
    opts.mouse_sens = std::min(std::max(value, 10), 500);   // (the keys work while the menu is closed: it shows it when opened)
    save();
}

/// (relink) The mouse over the Teleport or Inventory page: where paint_list put each line (list_hit). The line under it chosen,
/// a click as Enter, or on an amount's left or right half as Left or Right; the wheel moves the choice; a right click goes back
void list_mouse_cart(const MkwVkPointer& ptr, int mx, int my, int mw, int mh, bool moved, bool amounts, int n, int& choice, bool* press,
                     bool& changed) {
    const int grab = (int)std::lround(6 * opt_u);
    const bool inside = mx >= 0 && my >= 0 && mx < mw && my < mh;
    const bool on_bar = scroll_mouse(list_sc[amounts], ptr, mx, my, mw, mh, opt_u, changed);   // (amounts: the inventory's)
    int over = -1, side = 0;
    if (inside && !on_bar && my >= list_hit.body_top && my < list_hit.body_bottom)
        for (int i = 0; i < n && i < list_hit.n; i++)
            if (my >= list_hit.top[i] && my < list_hit.bottom[i]) over = i;
    if (over >= 0 && mx >= list_hit.value_l[over] && mx < list_hit.value_r[over] + grab)
        side = mx < (list_hit.value_l[over] + list_hit.value_r[over]) / 2 ? -1 : 1;
    if (over >= 0 && (moved || ptr.pressed) && over != choice) choice = over, changed = true;
    if ((ptr.pressed & HOS_MOUSE_LEFT) && over >= 0) {
        if (amounts && item_kinds[over] && side != 0) press[side < 0 ? 3 : 4] = true;
        else press[7] = true;
    }
    if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;
}
/// Scrolls the teleport list so the chosen map (and its section heading, at the top) is in view
void teleport_reveal() {
    int section = 0;
    while (section + 1 < kTeleportSections && teleport_choice >= teleport_sections[section + 1].first) section++;
    const int line = teleport_choice + section + 1;
    if (teleport_choice == teleport_sections[section].first) teleport_scroll = std::min(teleport_scroll, line - 1);
    if (line < teleport_scroll) teleport_scroll = line;
    if (line >= teleport_scroll + kTeleportLines) teleport_scroll = line - kTeleportLines + 1;
    teleport_scroll = std::max(0, std::min(teleport_scroll, kTeleports + kTeleportSections - kTeleportLines));
}
/// Scrolls the inventory so the chosen item (and its section heading, at the top) is in view
void item_reveal() {
    int section = 0;
    while (section + 1 < kItemSections && item_choice >= item_sections.s[section + 1].first) section++;
    const int line = item_choice + section + 1;
    if (item_choice == item_sections.s[section].first) item_scroll = std::min(item_scroll, line - 1);
    if (line < item_scroll) item_scroll = line;
    if (line >= item_scroll + kTeleportLines) item_scroll = line - kTeleportLines + 1;
    item_scroll = std::max(0, std::min(item_scroll, kItems + kItemSections - kTeleportLines));
}
void menu_tick() {
    fullscreen_tick();
    {   // (relink) V-Sync changed: the swapchain made again at its next present, in the mode it now asks for (vk_main.cpp)
        static int applied = -1;
        if (applied < 0) applied = opts.vsync;
        else if (opts.vsync != applied && dev.swap_extent.width && dev.swap_extent.height) {
            applied = opts.vsync;
            logf("menu: V-Sync %s", opts.vsync ? "on" : "off");
            dev.swap_resize.store((uint64_t)dev.swap_extent.width << 32 | dev.swap_extent.height, std::memory_order_release);
        }
    }
    enum { K_N = 17 };
    static bool held[K_N];
    static int keys[K_N] = {VK_INSERT, VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_ESCAPE, VK_F1,   // (F1: for keyboards without Insert)
                            VK_RETURN, VK_DELETE, VK_BACK,
                            'Q', 'E', VK_TAB,   // (the tabs: Q and E, Tab and Shift+Tab)
                            VK_PRIOR, VK_NEXT, VK_HOME, VK_END};   // (the long lists: a page, the ends)
    {   // (relink) the Menu key row; F1 as well unless another of ReLink's keys is F1 (then a key nothing is: Insert's place
        // left as it is would open the menu twice when the Menu key row is Insert)
        const int menu = hotkey_vk(opts.key_menu), f1_taken = hotkey_vk(opts.key_shot) == VK_F1 || hotkey_vk(opts.key_full) == VK_F1;
        keys[0] = menu ? menu : VK_INSERT;
        keys[6] = keys[0] == VK_F1 || f1_taken ? 0xFF : VK_F1;   // (0xFF: no key)
    }
    HWND fg = GetForegroundWindow();
    DWORD pid = 0;
    if (fg) GetWindowThreadProcessId(fg, &pid);
    const bool mine = pid == GetCurrentProcessId();
    bool press[K_N] = {};
    for (int i = 0; i < K_N; i++) {
        const bool down = mine && key_is_down(keys[i]);
        const bool queued = host && host->take_key_press && host->take_key_press(keys[i]);
        press[i] = host && host->take_key_press ? mine && queued : down && !held[i];
        held[i] = down;
    }
    // (relink) F12: a screenshot of the game's picture (vk_screenshot.h), menu open or not; not while a control waits for
    // its key in the menu (the key is that control's then)
    if (hotkey_vk(opts.key_shot) && host && host->take_key_press && host->take_key_press(hotkey_vk(opts.key_shot)) && mine && !listening)
        screenshot_want.store(true, std::memory_order_relaxed);
    // (relink) the gamepads, by position: Home, or Select and Start together, as Insert; the D-pad as the arrows,
    // the bottom face button as Enter, the right one as Esc (not while a binding waits for its button)
    static uint32_t pad_held = 0;
    const uint32_t raw = mine && host && host->pad_raw ? host->pad_raw() : 0, pad_press = raw & ~pad_held;
    const uint32_t chord = 1u << HOS_PAD_BACK | 1u << HOS_PAD_START;
    const bool chord_now = (raw & chord) == chord && (pad_held & chord) != chord;
    pad_held = raw;
    auto pad = [&](int p) { return listening != 1 && (pad_press >> p & 1) != 0; };
    if (pad(HOS_PAD_GUIDE) || chord_now) press[0] = true;
    {   // holding an arrow or the D-pad repeats it: after 0.35 s, 12 a second, then 30 a second after 1.5 s held
        static int64_t held_since[5], next_repeat[5];
        const int pads[5] = {0, HOS_PAD_UP, HOS_PAD_DOWN, HOS_PAD_LEFT, HOS_PAD_RIGHT};
        const int64_t t = prof_now();
        for (int i = 1; i <= 4; i++) {
            const bool down = held[i] || (open_now.load() && listening != 1 && (raw >> pads[i] & 1));
            if (!down) { held_since[i] = 0; continue; }
            if (!held_since[i]) { held_since[i] = t, next_repeat[i] = t + 350000000LL; continue; }
            if (t >= next_repeat[i] && open_now.load() && listening == 0) {
                press[i] = true;
                next_repeat[i] = t + (t - held_since[i] > 1500000000LL ? 33000000LL : 83000000LL);
            }
        }
    }
    if (open_now.load() && listening != 1) {
        press[1] |= pad(HOS_PAD_UP), press[2] |= pad(HOS_PAD_DOWN), press[3] |= pad(HOS_PAD_LEFT);
        press[4] |= pad(HOS_PAD_RIGHT), press[5] |= pad(HOS_PAD_EAST), press[7] |= pad(HOS_PAD_SOUTH);
    }
    {   // RX_VK_MENU_SCRIPT (test runs, which never have the focus): timed menu keys, "<seconds>:<INS|UP|DOWN|LEFT|RIGHT|ESC|ENTER|TABL|TABR>,..."
        // (SHOT: F12, the screenshot)
        static const char* next = getenv("RX_VK_MENU_SCRIPT");
        static const int64_t start = prof_now();
        double at;
        char key[8];
        int n = 0;
        if (next && sscanf(next, "%lf:%7[A-Z]%n", &at, key, &n) == 2 && (prof_now() - start) / 1e9 >= at) {
            static const char* names[] = {"INS", "UP", "DOWN", "LEFT", "RIGHT", "ESC", "", "ENTER", "", "", "TABL", "TABR"};
            for (int i = 0; i < 12; i++) press[i] |= *names[i] && !strcmp(key, names[i]);
            if (!strcmp(key, "SHOT")) screenshot_want.store(true, std::memory_order_relaxed);
            next = next[n] == ',' ? next + n + 1 : nullptr;
        }
    }
    bool tab_left = press[10] || pad(HOS_PAD_LSHOULDER) || pad(HOS_PAD_LTRIGGER);
    bool tab_right = press[11] || press[12] || pad(HOS_PAD_RSHOULDER) || pad(HOS_PAD_RTRIGGER);
    if (press[12] && key_is_down(VK_SHIFT)) tab_left = true, tab_right = false;
    const int scale = scale_of(dev.swap_extent.height);
    bool changed = scale != cur_scale || menu_refresh;
    menu_refresh = false;
    cur_scale = scale;
    // (relink) the mouse, while the menu is open: the cursor in the menu's pixels (mx, my), whether it moved since the
    // last frame, and its clicks and wheel since then (vk_menu_hit.h)
    MkwVkPointer ptr{};
    int mx = -1, my = -1, mw = 0, mh = 0;
    bool mouse = false, moved = false;
    if (open_now.load() && host && host->pointer) {
        mouse = host->pointer(&ptr) != 0;
        {
            std::lock_guard lk(bmp_mu);
            mw = (int)menu_bmp.w, mh = (int)menu_bmp.h;
        }
        int ix = 0, iy = 0, x0 = 0, y0 = 0;
        menu_hit::to_image(ptr.x, ptr.y, ptr.width, ptr.height, (int)dev.swap_extent.width, (int)dev.swap_extent.height,
                           &ix, &iy);
        menu_hit::origin((int)dev.swap_extent.width, (int)dev.swap_extent.height, mw, mh, &x0, &y0);
        if (mouse) mx = ix - x0, my = iy - y0;
        static int last_x = -1, last_y = -1;
        moved = mouse && mouse_read && (mx != last_x || my != last_y);   // (not the first read: a cursor resting over the
        last_x = mx, last_y = my, mouse_read = true;                     // menu as it opens leaves the selection alone)
        if (!mouse) ptr.pressed = 0, ptr.wheel = 0;
    }
    // (closing with a setting that takes a restart changed: the restart question first, Later already chosen on it)
    auto later = [&] {
        int now[mr::kMax];
        restart_values(now);
        restart_state.choose_later(now);
        open_now = false, page = 0, changed = true;
    };
    if ((press[0] || press[6]) && open_now.load() && page == 5 && ask_quit) {
        ask_quit = false, open_now = false, page = 0, changed = true;   // (Quit game's question: the menu closes, nothing else)
    } else if ((press[0] || press[6]) && open_now.load() && page == 5) {
        later();
    } else if ((press[0] || press[6]) && open_now.load() && restart_ask()) {
        page = 5, ask_sel = 0, ask_quit = false, listening = 0, changed = true;
    } else if (press[0] || press[6]) {
        open_now = !open_now.load(), changed = true;
        for (Scroller* sc : {&opt_sc, &cheat_sc, &list_sc[0], &list_sc[1], &travel_sc, &controls_sc}) sc->reveal = true, sc->drag = -1;
        page = 0, listening = 0;
        // (the layout row shows the controls as they are: host/nn_hid.c keeps them, the ini's pad_ lines, and a saved
        // layout is not applied over them at start)
        if (open_now.load()) opts.pad_layout = detect_pad_layout();
        if (open_now.load()) dlss_probe();   // (whether the Upscaler row can offer DLSS here: NGX asked once)
        if (open_now.load()) refresh_mics();   // (the microphones as they are now)
    }
    if (open_now.load() && page == 5) {   // (relink) the restart question: Later (the default) or Restart now
        if (mouse) {   // (a button under the cursor chosen, a click on it pressed; the wheel changes it, a right click is Later)
            const menu_hit::Hit hit = menu_hit::ask(mx, my, scale, mw, mh);
            if (hit.part == menu_hit::kButton && (moved || ptr.pressed) && hit.index != ask_sel) ask_sel = hit.index, changed = true;
            if ((ptr.pressed & HOS_MOUSE_LEFT) && hit.part == menu_hit::kButton) press[7] = true;
            if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;
            if (ptr.wheel) ask_sel ^= 1, changed = true;
        }
        if (press[1] || press[2] || press[3] || press[4] || tab_left || tab_right) ask_sel ^= 1, changed = true;
        if (ask_quit && (press[5] || (press[7] && ask_sel == 0))) {   // (Back: the menu's list again)
            ask_quit = false, page = 0, changed = true;
        } else if (ask_quit && press[7]) {   // (Quit: everything saved, then the window closes as the player would close it)
            logf("menu: quit game");
            save();
            pipeline_cache_save(true);
            ask_quit = false, open_now = false, page = 0, changed = true;
            PostMessageA(app_window(), WM_APP + 10, 0, 0);
        } else if (press[5] || (press[7] && ask_sel == 0)) {
            later();
        } else if (press[7]) {   // (Restart now: everything saved, then the window closes as the player would close it)
            logf("menu: restart now, to apply %s", restart_list(", ").c_str());
            save();
            pipeline_cache_save(true);
            open_now = false, page = 0, changed = true;
            PostMessageA(app_window(), WM_APP + 9, 0, 0);
        }
        if (changed && open_now.load()) redraw_menu(scale);
    } else if (open_now.load() && page == 1) {   // (relink) the Controls page
        if (!can_bind()) page = 0;
        else if (listening) {
            if (press[5] || prof_now() > listen_end) listening = 0, changed = true;   // (Esc, or nothing for a while)
            else if (listening == 1) {
                if (!listen_armed) listen_armed = (raw & ~(1u << HOS_PAD_GUIDE)) == 0;
                else if (const uint32_t p = pad_press & ~(1u << HOS_PAD_GUIDE)) {
                    int bit = 0;
                    while (!(p >> bit & 1)) bit++;
                    host->bind(HOS_BIND_PAD, controls[csel].button, 0, bit);
                    listening = 0, changed = true;
                    opts.pad_layout = detect_pad_layout();
                    save();
                }
            } else {
                int down = 0;
                for (int k : bindable_keys())
                    if (mine && key_is_down(k)) {
                        down = k;
                        break;
                    }
                if (!listen_armed) listen_armed = !down;
                else if (down) {
                    host->bind(HOS_BIND_KEY, controls[csel].button, 0, down);
                    host->bind(HOS_BIND_KEY, controls[csel].button, 1, 0);
                    listening = 0, changed = true;
                    save();
                }
            }
        } else {
            if (mouse) {   // (the mouse: the row under it selected, a click on a binding listens for its new input, on
                           // Reset resets; the wheel moves the selection, a right click goes back)
                // (where paint_controls_cart put each row and column: controls_hit)
                const ControlsHit& ch = controls_hit;
                const bool inside = mx >= 0 && my >= 0 && mx < mw && my < mh;
                const bool on_bar = scroll_mouse(controls_sc, ptr, mx, my, mw, mh, opt_u, changed);
                int over = -1, col = -1;
                if (inside && !on_bar && my >= ch.body_top && my < ch.body_bottom)
                    for (int i = 0; i <= kControls; i++)
                        if (my >= ch.top[i] && my < ch.bottom[i]) over = i;
                if (over >= 0 && over < kControls) col = mx >= ch.key_l && mx < ch.key_r ? 1 : mx >= ch.pad_l && mx < ch.pad_r ? 0 : -1;
                const bool on_reset = over == kControls && mx >= ch.reset_l && mx < ch.reset_r;
                if (over >= 0 && (moved || ptr.pressed)) {
                    if (over != csel) csel = over, changed = true;
                    if (col >= 0 && !keys_only(csel) && ccol != col) ccol = col, changed = true;
                }
                if ((ptr.pressed & HOS_MOUSE_LEFT) && (col >= 0 || on_reset)) press[7] = true;   // (as Enter)
                if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;
            }
            if (press[5]) page = 0, changed = true;
            if (press[1] || press[2]) controls_sc.reveal = true;
            if (press[1]) csel = (csel + kControls) % (kControls + 1), changed = true;
            if (press[2]) csel = (csel + 1) % (kControls + 1), changed = true;
            if ((press[3] || press[4]) && csel < kControls && !keys_only(csel)) ccol ^= 1, changed = true;
            if (keys_only(csel)) ccol = 1;   // (the stick's and the camera's directions: keys only)
            if (press[7]) {
                if (csel == kControls) {
                    host->bind(HOS_BIND_PAD, -1, 0, 0);
                    opts.pad_layout = 0;
                    save();
                }
                else listening = ccol ? 2 : 1, listen_armed = false, listen_end = prof_now() + 6000000000LL;
                changed = true;
            }
            if ((press[8] || press[9]) && csel < kControls) {
                if (ccol == 0) {
                    host->bind(HOS_BIND_PAD, controls[csel].button, 0, -1);
                    opts.pad_layout = detect_pad_layout();
                }
                else for (int k = 0; k < 2; k++) host->bind(HOS_BIND_KEY, controls[csel].button, k, 0);
                changed = true;
                save();
            }
        }
        if (changed) redraw_menu(scale);
    } else if (open_now.load() && page == 2) {
        // (the Cheats rows that choose with Left and Right: Set time, Inventory item, Change quantity, Teleport)
        auto chooses = [](int row) { return row == 4 || row == 5 || row == 6 || row == 8; };
        // (the rows as shown, cheat_order: the arrows and the wheel go through them in that order)
        auto step = [&](int dir, bool wrap) {
            int k = 0;
            while (k < kCheatRows && cheat_order[k] != cheat_selected) k++;
            k = wrap ? (k + dir + kCheatRows) % kCheatRows : std::clamp(k + dir, 0, kCheatRows - 1);
            cheat_selected = cheat_order[k], changed = true;
            cheat_sc.reveal = true;   // (the keys moved it: in view)
        };
        if (!cheat_done.empty() && prof_now() >= cheat_done_until) cheat_done.clear(), changed = true;   // (its word said)
        if (mouse) {   // (the mouse, over the cartridge: where paint_cheats put each row and its value, cheat_hit)
            const int grab = (int)std::lround(6 * opt_u);
            const bool inside = mx >= 0 && my >= 0 && mx < mw && my < mh;
            const bool on_bar = scroll_mouse(cheat_sc, ptr, mx, my, mw, mh, opt_u, changed);
            int over = -1, side = 0;
            if (inside && !on_bar && my >= cheat_hit.body_top && my < cheat_hit.body_bottom)
                for (int r = 0; r < kCheatRows; r++)
                    if (my >= cheat_hit.top[r] && my < cheat_hit.bottom[r]) over = r;
            if (over >= 0 && mx >= cheat_hit.value_l[over] && mx < cheat_hit.value_r[over] + grab)
                side = mx < (cheat_hit.value_l[over] + cheat_hit.value_r[over]) / 2 ? -1 : 1;
            if (over >= 0 && (moved || ptr.pressed) && over != cheat_selected) cheat_selected = over, changed = true;
            const bool on_choice = over >= 0 && over == cheat_selected && chooses(over) && side != 0;
            if ((ptr.pressed & HOS_MOUSE_LEFT) && over >= 0) {
                if (on_choice) press[side < 0 ? 3 : 4] = true;
                else press[7] = true;
            }
            if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;
        }
        if (press[5]) page = 0, changed = true;
        else {
            if (press[1] || press[2]) step(press[2] ? 1 : -1, true);
            if (cheat_selected < 2 && (press[3] || press[4] || press[7])) {
                if (cheat_selected == 0) protect_hearts.store(!protect_hearts.load());
                else player_flight.store(player_flight.load() < 0 ? 0 : -1);
                changed = true;
            }
            if (cheat_selected == 4 && (press[3] || press[4])) {
                clock_hour = (clock_hour + (press[4] ? 1 : 23)) % 24;
                changed = true;
            }
            if (cheat_selected == 5 && (press[3] || press[4])) {
                pouch_item = (pouch_item + kPouchItems + (press[4] ? 1 : -1)) % kPouchItems;
                changed = true;
            }
            if (cheat_selected == 6 && (press[3] || press[4])) {
                const int step = press[4] ? 1 : -1;
                pouch_delta = std::clamp(pouch_delta + step, -10, 10);
                if (!pouch_delta) pouch_delta = step;
                changed = true;
            }
            if (press[7] && cheat_selected >= 2 && cheat_selected <= 4) {
                cheat_action.store(cheat_selected == 4 ? 100 + clock_hour : cheat_selected);
                char said[48];
                snprintf(said, sizeof said, "the clock is set to %02d:00.", clock_hour);
                cheat_say(cheat_selected == 2 ? "Link's position is marked." : cheat_selected == 3 ? "Link goes back to the mark." : said);
                changed = true;
            }
            if (press[7] && cheat_selected == 6) {
                cheat_action.store(200 + pouch_item * 32 + pouch_delta + 10);
                char said[64];
                snprintf(said, sizeof said, "%s %d %s.", pouch_delta > 0 ? "gave" : "took", pouch_delta > 0 ? pouch_delta : -pouch_delta,
                         pouch_items[pouch_item]);
                cheat_say(said);
                changed = true;
            }
            if (press[7] && cheat_selected == 7) {
                free_camera.store(!free_camera.load());
                changed = true;
            }
            if (cheat_selected == 8 && (press[3] || press[4])) {
                teleport_choice = (teleport_choice + kTeleports + (press[4] ? 1 : -1)) % kTeleports;
                changed = true;
            }
            if (press[7] && cheat_selected == 8) page = 3, teleport_reveal(), changed = true;
            if (press[7] && cheat_selected == 10) page = 4, item_reveal(), changed = true;
            if (press[7] && cheat_selected == kTravelCheat) {   // (relink: the Travel page, Link's own age first)
                page = 6, travel_pick = -1, travel_list(), travel_reveal(), changed = true;
                if (!travel_pictures_on.exchange(true, std::memory_order_relaxed)) {   // (pictures are taken from now on)
                    PictureJob j;
                    j.kind = PictureJob::Begin;
                    picture_job(std::move(j));
                }
            }
            if (press[7] && cheat_selected == 9) {
                cheat_action.store(999);   // (host/hos_main.c OOT_AGE_ACTION)
                open_now = false;
                changed = true;
            }
        }
        if (changed) redraw_menu(scale);
    } else if (open_now.load() && page == 4) {   // the inventory
        if (mouse)
            list_mouse_cart(ptr, mx, my, mw, mh, moved, true, kItems, item_choice, press, changed);
        if (press[5]) page = 2, changed = true;
        else {
            if (press[1] || press[2]) {
                item_choice = (item_choice + kItems + (press[2] ? 1 : -1)) % kItems;
                changed = true;
            }
            if (press[13] || press[14] || press[15] || press[16]) {   // (Page Up / Page Down, Home / End)
                item_choice = press[15] ? 0 : press[16] ? kItems - 1
                            : std::clamp(item_choice + (press[14] ? 1 : -1) * (kTeleportLines - 2), 0, kItems - 1);
                changed = true;
            }
            if (tab_left || tab_right) {
                int section = 0;
                while (section + 1 < kItemSections && item_choice >= item_sections.s[section + 1].first) section++;
                if (tab_right || item_choice == item_sections.s[section].first)
                    section = (section + kItemSections + (tab_left ? -1 : 1)) % kItemSections;
                item_choice = item_sections.s[section].first;
                changed = true;
            }
            if (press[1] || press[2] || press[13] || press[14] || press[15] || press[16] || tab_left || tab_right) list_sc[1].reveal = true;
            if (press[7]) cheat_action.store(OOT_CHEAT_ITEM_ACTION + item_choice * 256 + 200);
            else if (press[3] || press[4]) cheat_action.store(OOT_CHEAT_ITEM_ACTION + item_choice * 256 + 64 + (press[4] ? 1 : -1));
            if (changed) item_reveal();
        }
        if (changed) redraw_menu(scale);
    } else if (open_now.load() && page == 3) {   // the teleport list
        if (mouse)
            list_mouse_cart(ptr, mx, my, mw, mh, moved, false, kTeleports, teleport_choice, press, changed);
        if (press[5]) page = 2, changed = true;
        else if (press[7]) {
            cheat_action.store(1000 + teleport_choice);
            open_now = false, page = 2;   // (back to the game: the map change starts with an event)
            changed = true;
        } else {
            if (press[1] || press[2]) {
                teleport_choice = (teleport_choice + kTeleports + (press[2] ? 1 : -1)) % kTeleports;
                changed = true;
            }
            if (press[13] || press[14] || press[15] || press[16]) {   // (Page Up / Page Down, Home / End)
                teleport_choice = press[15] ? 0 : press[16] ? kTeleports - 1
                                : std::clamp(teleport_choice + (press[14] ? 1 : -1) * (kTeleportLines - 2), 0, kTeleports - 1);
                changed = true;
            }
            if (press[3] || press[4] || tab_left || tab_right) {   // (to the first map of the previous or next section)
                int section = 0;
                while (section + 1 < kTeleportSections && teleport_choice >= teleport_sections[section + 1].first) section++;
                const bool back = press[3] || tab_left;
                if (!back || teleport_choice == teleport_sections[section].first)
                    section = (section + kTeleportSections + (back ? -1 : 1)) % kTeleportSections;
                teleport_choice = teleport_sections[section].first;
                changed = true;
            }
            if (press[1] || press[2] || press[3] || press[4] || press[13] || press[14] || press[15] || press[16] || tab_left || tab_right)
                list_sc[0].reveal = true;
            if (changed) teleport_reveal();
        }
        if (changed) redraw_menu(scale);
    } else if (open_now.load() && page == 6) {   // (relink) the Travel page
        travel_list();
        const int n = (int)travel.place.size(), sections = (int)travel.areas.size();
        press[9] |= pad(HOS_PAD_WEST);   // (Y on a pad, as Backspace: the picture taken again)
        bool switch_age = press[12] || pad(HOS_PAD_NORTH);   // (Tab, or X on a pad: the other age's places)
        if (mouse && (ptr.pressed & HOS_MOUSE_LEFT) && my >= travel_hit.age_t && my < travel_hit.age_b)
            for (int a = 0; a < 2; a++)
                if (mx >= travel_age_l[a] && mx < travel_age_r[a] && a != travel.adult) switch_age = true;
        if (switch_age) {
            travel_pick = !travel.adult, travel_choice = 0, travel_scroll = 0;
            travel_list(), travel_reveal(), changed = true;
            ptr.pressed = 0;
            if (press[12]) tab_left = tab_right = false;   // (Tab is the tabs' key elsewhere: here the age's alone)
        }
        if (mouse && n) travel_mouse(ptr, mx, my, mw, mh, moved, n, press, changed);
        else if (mouse && (ptr.pressed & HOS_MOUSE_RIGHT))
            press[5] = true;
        if (press[5]) page = 2, changed = true;   // (back to the Cheats page, as the teleport list)
        else if (press[7] && n) {
            cheat_action.store(OOT_PLACE_ACTION + travel.place[travel_choice] * 2 + travel.adult);
            open_now = false, page = 0;   // (back to the game: the map change starts with an event)
            changed = true;
        } else if (press[9] && n) {
            PictureJob j;
            j.kind = PictureJob::Forget, j.place = travel.place[travel_choice];
            picture_job(std::move(j));
        } else if (n) {
            if (press[1] || press[2]) travel_choice = (travel_choice + n + (press[2] ? 1 : -1)) % n, changed = true;
            if (press[13] || press[14] || press[15] || press[16]) {   // (Page Up / Page Down, Home / End)
                travel_choice = press[15] ? 0 : press[16] ? n - 1 : std::clamp(travel_choice + (press[14] ? 1 : -1) * (kTeleportLines - 2), 0, n - 1);
                changed = true;
            }
            if (press[3] || press[4] || tab_left || tab_right) {   // (to the first place of the previous or next area)
                int section = 0;
                while (section + 1 < sections && travel_choice >= travel.areas[section + 1].first) section++;
                const bool back = press[3] || tab_left;
                if (!back || travel_choice == travel.areas[section].first) section = (section + sections + (back ? -1 : 1)) % sections;
                travel_choice = travel.areas[section].first;
                changed = true;
            }
            if (press[1] || press[2] || press[3] || press[4] || press[13] || press[14] || press[15] || press[16] || tab_left || tab_right)
                travel_sc.reveal = true;
            if (changed) travel_reveal();
        }
        if (changed) redraw_menu(scale);
    } else if (open_now.load()) {
        if (mouse) {   // (the mouse, over the cartridge: where paint_options put each part, opt_hit)
            const int count = tab_row_count(cur_tab);
            const OptHit& oh = opt_hit;
            const int grab = (int)std::lround(6 * opt_u);   // (a click this far past a track's ends still takes them)
            const bool inside = mx >= 0 && my >= 0 && mx < mw && my < mh;
            int over_tab = -1, over_row = -1, side = 0, choice = -1, permille = 0;
            bool on_track = false;
            if (inside && my < oh.tabs_bottom) {
                for (int t = 0; t < TAB_COUNT; t++)
                    if (mx >= oh.tab_l[t] && mx < oh.tab_r[t]) over_tab = t;
            } else if (inside && my >= oh.body_top && my < oh.body_bottom) {
                for (int k = 0; k < oh.rows && k < count; k++)
                    if (my >= oh.row_top[k] && my < oh.row_bottom[k]) over_row = k;
            }
            if (over_row >= 0) {
                const int k = over_row;
                if (mx >= oh.value_l[k] && mx < oh.value_r[k] + grab) side = mx < (oh.value_l[k] + oh.value_r[k]) / 2 ? -1 : 1;
                for (int i = 0; oh.kind[k] == 2 && i < oh.choices[k]; i++)
                    if (mx >= oh.choice_l[k][i] && mx < oh.choice_r[k][i]) choice = i;
                if (oh.kind[k] == 3 && mx >= oh.track_l[k] - grab && mx < oh.track_r[k] + grab) {
                    on_track = true;
                    permille = std::clamp((mx - oh.track_l[k]) * 1000 / std::max(1, oh.track_r[k] - oh.track_l[k]), 0, 1000);
                }
            }
            {   // (the wheel scrolls the rows, over the tabs it changes them; the bar is dragged or clicked)
                MkwVkPointer sp = ptr;
                if (over_tab >= 0) sp.wheel = 0;
                if (scroll_mouse(opt_sc, sp, mx, my, mw, mh, opt_u, changed)) over_row = -1, side = 0, choice = -1, on_track = false;
            }
            if (over_tab != hover_tab) hover_tab = over_tab, changed = true;
            int& sel = tab_selected[cur_tab];
            if (drag_row >= 0) {   // (a slider held: it follows the cursor's x until let go)
                Row& d = rows[tab_row_to_global(cur_tab, drag_row)];
                if (!(ptr.held & HOS_MOUSE_LEFT) || !d.value) {
                    drag_row = -1;
                    save();
                } else {
                    const int p = std::clamp((mx - oh.track_l[drag_row]) * 1000 / std::max(1, oh.track_r[drag_row] - oh.track_l[drag_row]), 0, 1000);
                    const int v = menu_hit::slider_value(p, d.lo, d.hi, d.step);
                    if (v != *d.value) set_row(d, v, v > *d.value ? 1 : -1, false), changed = true;
                }
            } else if (over_row >= 0 && (moved || ptr.pressed) && over_row != sel) {
                sel = over_row, changed = true;
            }
            if ((ptr.pressed & HOS_MOUSE_LEFT) && drag_row < 0) {
                if (over_tab >= 0 && over_tab != cur_tab) cur_tab = over_tab, changed = true;
                else if (over_row >= 0) {
                    Row& d = rows[tab_row_to_global(cur_tab, over_row)];
                    if (!d.value) press[7] = true;   // (a page: as Enter)
                    else if (choice >= 0) {          // (one of its choices: that one)
                        const int v = d.lo + choice * d.step;
                        if (v != *d.value) set_row(d, v, v > *d.value ? 1 : -1);
                        changed = true;
                    } else if (on_track) {           // (a slider's track: there, and held it follows the cursor)
                        drag_row = over_row;
                        const int v = menu_hit::slider_value(permille, d.lo, d.hi, d.step);
                        if (v != *d.value) set_row(d, v, v > *d.value ? 1 : -1, false);
                        changed = true;
                    } else if (side >= 0) press[4] = true;   // (a switch flips; a value's right half, or the name, as Right)
                    else press[3] = true;                    // (the left half as Left)
                }
            }
            if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;   // (as Esc)
            if (ptr.wheel && drag_row < 0 && over_tab >= 0) {   // (over the tabs: the previous or the next)
                if (ptr.wheel > 0) tab_left = true;
                else tab_right = true;
            }
        }
        if (press[5] && restart_ask()) page = 5, ask_sel = 0, ask_quit = false, changed = true;   // (the restart question first)
        else if (press[5]) open_now = false;
        // (the tabs: LB and RB, Q and E, or Tab)
        if (press[1] || press[2] || press[3] || press[4]) opt_sc.reveal = true;   // (the keys moved or changed: show the row)
        if (tab_left) cur_tab = (cur_tab + TAB_COUNT - 1) % TAB_COUNT, changed = true;
        else if (tab_right) cur_tab = (cur_tab + 1) % TAB_COUNT, changed = true;
        {
            static int scrolled_tab = -1;
            if (scrolled_tab != cur_tab) scrolled_tab = cur_tab, opt_sc.px = 0, opt_sc.reveal = true;
        }
        const int tab_count = tab_row_count(cur_tab);
        int& selected = tab_selected[cur_tab];
        if (press[1]) selected = (selected + tab_count - 1) % tab_count, changed = true;
        if (press[2]) selected = (selected + 1) % tab_count, changed = true;
        Row& r = rows[tab_row_to_global(cur_tab, selected)];
        if (press[7] && &r == &rows[kCheatsRow]) page = 2, changed = true;
        else if (press[7] && &r == &rows[kQuitRow]) {   // (Quit game: asked first, Back the default)
            if (app_window()) page = 5, ask_sel = 0, ask_quit = true, changed = true;
        } else if (press[7] && !r.value && can_bind()) page = 1, csel = 0, ccol = 0, changed = true;
        if ((press[3] || press[4] || press[7]) && r.value) {   // (Enter, or the bottom face button, as Right)
            if (press[7]) press[4] = true;
            set_row(r, *r.value + (press[4] ? r.step : -r.step), press[4] ? 1 : -1);
            changed = true;
        }
        if (changed) redraw_menu(scale);
    }
    {   // (relink) the window lets the mouse go while the menu is open, and takes it back after (host/display.c)
        static bool told = false;
        const bool now_open = open_now.load();
        if (!now_open) {
            if (drag_row >= 0) save();   // (a slider let go off the window: kept now)
            hover_tab = -1, drag_row = -1, mouse_read = false;
        }
        if (now_open != told && app_window()) {
            told = now_open;
            PostMessageA(app_window(), WM_APP + 8, (WPARAM)now_open, 0);
        }
    }
    if (opts.show_fps) {
        static int64_t next = 0, last_game = 0, last_shown = 0;
        const int64_t now = prof_now();
        double shown = 0, game = 0;
        if (fps_take(now, shown_frames.load(), game_frames.load(), &next, &last_shown, &last_game, &shown, &game)) {
            snprintf(fps_text, sizeof fps_text, "%.0f", shown);
            if (opts.framegen >= 2) snprintf(fps_sub, sizeof fps_sub, "game %.0f", game);
            else fps_sub[0] = 0;
            fps_bmp.keyed = true;
            gdi_draw(fps_bmp, 120u * scale, 54u * scale, paint_fps, scale);
        }
    }
    if (notice_new.exchange(false, std::memory_order_acq_rel)) {   // (relink: the Travel page's "picture saved" line)
        HDC dc = CreateCompatibleDC(nullptr);
        HFONT f = font(16 * scale, FW_SEMIBOLD);
        HGDIOBJ old = SelectObject(dc, f);
        SIZE sz{};
        GetTextExtentPoint32A(dc, notice_text, (int)strlen(notice_text), &sz);
        SelectObject(dc, old);
        DeleteObject(f);
        DeleteDC(dc);
        notice_bmp.keyed = true;   // (its round corners see-through: set before its first drawing, and never changed)
        gdi_draw(notice_bmp, (uint32_t)sz.cx + 40u * scale, 40u * scale, paint_notice, scale);
    }
}

void menu_record(VkCommandBuffer cmd, VkImage dst, int slot, uint32_t w, uint32_t h) {
    if (startup_on.load(std::memory_order_acquire)) {   // (relink) under the counter and the menu
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        std::unique_lock lk(bmp_mu);
        const uint32_t sw = startup_bmp.w, sh = startup_bmp.h;
        lk.unlock();
        if (sw) copy_over(cmd, dst, startup_bmp, ups[2][slot], (int)(w - sw) / 2, (int)(h - sh) / 2, w, h);
    } else {   // (the start-up screen's upload buffers, the window's size each: let go once no frame in flight can still read them)
        static int presents_since = 0;
        if ((ups[2][0].buf || ups[2][1].buf || ups[2][2].buf) && ++presents_since > 12) {
            for (Upload& u : ups[2]) {
                if (u.buf) destroy_buffer(u.buf, u.alloc);
                u = {};
            }
        }
    }
    if (opts.show_fps && !startup_on.load(std::memory_order_relaxed)) {   // (no game frames yet: not "0 FPS" over the start-up screen)
        // The game image was just blitted (or cleared) into dst. Its write must
        // finish before the FPS pixels overwrite it; command order alone does
        // not order these transfer writes. The menu below has its own barrier.
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        std::unique_lock lk(bmp_mu);
        const uint32_t fw = fps_bmp.w;
        lk.unlock();
        if (fw) copy_over(cmd, dst, fps_bmp, ups[1][slot], (int)w - (int)fw - (int)(12 * cur_scale), (int)(10 * cur_scale), w, h);
    }
    if (!open_now.load(std::memory_order_relaxed) && prof_now() < notice_until.load(std::memory_order_relaxed)) {   // (relink)
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        std::unique_lock lk(bmp_mu);
        const uint32_t nw = notice_bmp.w, nh = notice_bmp.h;
        lk.unlock();   // (low in the middle, above the item row's place: the HUD keeps its corners)
        if (nw && nh + 60 * cur_scale < h) copy_over(cmd, dst, notice_bmp, ups[3][slot], (int)(w - nw) / 2, (int)(h - nh - 60 * cur_scale), w, h);
    }
    if (open_now.load(std::memory_order_relaxed)) {
        std::unique_lock lk(bmp_mu);
        const uint32_t mw = menu_bmp.w, mh = menu_bmp.h;
        lk.unlock();
        // (one copy finished before the next: on a Radeon the menu came out with its top torn away beside the FPS counter)
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        int x0 = 0, y0 = 0;   // (the place the mouse's hits are read against: menu_tick)
        menu_hit::origin((int)w, (int)h, (int)mw, (int)mh, &x0, &y0);
        if (mw) copy_over(cmd, dst, menu_bmp, ups[0][slot], x0, y0, w, h);
    }
}

} // namespace mkwvk
