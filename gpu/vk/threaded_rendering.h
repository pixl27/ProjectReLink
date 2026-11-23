// (hookshot) Threaded rendering, the Insert menu's row (vk_menu.cpp, threaded_rendering= in hookshot_options.ini): Auto, On or
// Off, as one rule every reader shares: the menu (vk_menu.cpp threaded_choice resolves it once, at the start, and hands each
// reader its half: menu_render_thread, menu_recorder_thread), the replay (vk_replay.cpp render_thread_mode), the recorder
// (vk_recorder.h mode_from_env), the host's present poll (HOS_OPT_RENDER_THREAD, host/nvn_native.c, through async_on) and the
// checks (tests/test_threaded_rendering.py compiles this file alone).
//
//   Off    the replay on the game's submitting thread, the recorder as RX_VK_RECORDER chooses it: as before the row
//   On     the render thread (what RX_VK_ASYNC=1 gives: the replay on a thread of its own, the game a frame ahead of it) and
//          the recorder thread (what RX_VK_RECORDER=thread gives) together, whatever the CPU
//   Auto   (the default) decided from the physical cores among the logical processors the process may run on (its
//          affinity mask, not the PC's count; a core with two usable SMT siblings counts once: vk_progcache.cpp usable_cores,
//          cores_of below), in three tiers:
//            kAutoRecorderCores (6) or more: the render thread and the recorder thread, as On;
//            kAutoRenderCores (4) or 5: the render thread alone, the recorder as with the row Off;
//            fewer: off, as Off.
//          And off where the mode is not safe, whatever the CPU (the caller says so: the capture off, pools with no
//          mirrors, RX_VK_LOW_SPEC, a frame dump, a benchmark replay). All of that behind kAutoEnabled, below, which is on:
//          turned off, Auto is Off on every PC.
//
// The tiers are measured (18 Nov 2026: a Ryzen 9 9950X3D with SMT off, held to N cores by its affinity, Castle Town's market
// entrance, two runs each; fps, then the share of frames of 16.7 ms or more):
//
//   cores   serial replay (Off)        the render thread alone     the render thread and the recorder thread
//   4       65.5 / 65.0, 12.7 / 15.0%  67.6 / 67.9, 7.7 / 8.7%     60.9 / 62.1, 34.7 / 31.5%
//   6       67.4 / 68.4, 6.9 / 4.5%    69.5 / 69.0, 3.3 / 4.2%     72.9 / 72.4, 1.9 / 2.4%
//   8       67.0 / 66.9, 5.5 / 6.6%    69.9 / 69.9, 3.1 / 2.9%     72.8 / 73.5, 1.4 / 1.4%
//
// So on 4 cores the render thread alone is the best of the three and the recorder thread beside it the worst (a third of the
// frames long); from 6 cores both threads are the best. 16 cores ran both at about 75 fps. Before, Auto was both threads
// from 12 logical processors and off below, which left CPUs of 6 to 11 cores without SMT serial and counted a 4-core,
// 8-thread CPU (a Steam Deck) as 8: it is 4 here, the tier measured on 4 cores.
//
// The environment's switches stay the overrides they were, for testing: RX_VK_ASYNC set chooses the render thread and
// RX_VK_RECORDER set the recorder, each over its half of the row; unset, the row decides. Read once, at the start: a change of
// the row takes effect at the next start (the replay's async_on and the recorder's mode are read once).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace mkwvk::threaded {

enum { AUTO = 0, ON = 1, OFF = 2 };   // (the row's values, in V-Sync's order: Auto, On, Off)

/// THE GATE: Auto's one switch. True: Auto follows the rule below (its tiers by cores, and off where unsafe), so a PC with 4
/// cores for the game runs the render thread by default, and one with 6 the recorder thread too. False: Auto is Off
/// everywhere and the row changes nobody's default; nothing else needs to change to turn Auto off again. Still open with it
/// on: the game's large GPU-written tables (Hyrule Field's culling tables) can be read a frame old behind the game.
constexpr bool kAutoEnabled = true;
/// (hookshot) Auto's tiers, in physical cores the process may run on (the game's own threads, the replay, the recorder and the
/// parallel replay's workers beside one another): from kAutoRenderCores the render thread (on 4 cores 67.6 / 67.9 fps against
/// 65.5 / 65.0 serial, and 7.7 / 8.7% of the frames long against 12.7 / 15.0%); from kAutoRecorderCores the recorder thread
/// too (on 6 cores 72.9 / 72.4 fps against 69.5 / 69.0 with the render thread alone; on 4 it was 60.9 / 62.1, the worst)
constexpr unsigned kAutoRenderCores = 4;
constexpr unsigned kAutoRecorderCores = 6;

enum class Why {
    On,                 // the row says On
    Off,                // the row says Off
    AutoGate,           // Auto, kAutoEnabled off: off everywhere
    AutoUnsafe,         // Auto, and the mode is not safe on this run
    AutoFewCores,       // Auto, fewer than kAutoRenderCores cores (or their count unknown: 0): off
    AutoRenderThread,   // Auto, kAutoRenderCores to kAutoRecorderCores - 1 cores: the render thread alone
    AutoBothThreads,    // Auto, kAutoRecorderCores or more: the render thread and the recorder thread
};
/// The row's answer in two halves, each read by its own reader. The recorder thread never comes without the render thread
struct Choice {
    bool render_thread;     // the render thread, RX_VK_ASYNC=1's (vk_replay.cpp render_thread_mode; the host's poll by it)
    bool recorder_thread;   // the recorder thread, RX_VK_RECORDER=thread's (vk_recorder.h mode_from_env)
    Why why;
};

/// What Auto picks for `cores` physical cores where the mode is safe, the gate aside
constexpr Choice auto_picks(unsigned cores) {
    if (cores >= kAutoRecorderCores) return {true, true, Why::AutoBothThreads};
    if (cores >= kAutoRenderCores) return {true, false, Why::AutoRenderThread};
    return {false, false, Why::AutoFewCores};
}

/// The rule, from the row's value, the physical cores the process may run on and whether the mode is unsafe on this run: no
/// environment, no clock, for the checks. `auto_enabled`: the gate (the checks pass it both ways). A value that is none of
/// the row's is Auto
constexpr Choice choose(int setting, unsigned cores, bool unsafe, bool auto_enabled = kAutoEnabled) {
    if (setting == ON) return {true, true, Why::On};
    if (setting == OFF) return {false, false, Why::Off};
    if (!auto_enabled) return {false, false, Why::AutoGate};
    if (unsafe) return {false, false, Why::AutoUnsafe};
    return auto_picks(cores);
}

/// (hookshot) Auto picked a thread, in either tier: the caller's last safety check, one that reads the device (the buffer
/// pools' mirrors), is asked only then
constexpr bool auto_picked(const Choice& c) { return c.why == Why::AutoRenderThread || c.why == Why::AutoBothThreads; }

/// (hookshot) Auto's count: the physical cores among the logical processors the process may run on. `core_masks`: one mask
/// a physical core, of its logical processors (GetLogicalProcessorInformation's RelationProcessorCore entries, an SMT core's
/// two siblings in one mask; on Linux host/posix's, from sysfs core_id and physical_package_id), `n` of them; `usable`: the
/// affinity mask. A core counts once when any of its logical processors is usable, so a 4-core, 8-thread CPU is 4. 0: none
/// counted (nothing read, or no core in the mask): the caller then counts logical processors, as before
constexpr unsigned cores_of(const uint64_t* core_masks, unsigned n, uint64_t usable) {
    unsigned cores = 0;
    for (unsigned i = 0; i < n; i++) cores += (core_masks[i] & usable) != 0;
    return cores;
}

/// (hookshot) The start's line for the choice (vk_menu.cpp logs it once, after "vk: Threaded rendering: "): `cores` Auto's
/// count, `cpus` the logical processors the process may run on, `unsafe` what made Auto not safe (AutoUnsafe only)
inline void describe(char* out, size_t size, const Choice& c, unsigned cores, unsigned cpus, const char* unsafe) {
    switch (c.why) {
    case Why::On: std::snprintf(out, size, "On (the render thread and the recorder thread)"); return;
    case Why::Off: std::snprintf(out, size, "Off"); return;
    case Why::AutoGate:
        std::snprintf(out, size, "Auto: off (Auto is turned off in this build; %u cores of %u logical processors, its rule is "
                      "the render thread from %u cores and the recorder thread too from %u)", cores, cpus, kAutoRenderCores,
                      kAutoRecorderCores);
        return;
    case Why::AutoUnsafe: std::snprintf(out, size, "Auto: off (not safe on this run: %s)", unsafe ? unsafe : "?"); return;
    case Why::AutoFewCores:
        std::snprintf(out, size, "Auto: off (%u cores of %u logical processors; the render thread from %u cores)", cores, cpus,
                      kAutoRenderCores);
        return;
    case Why::AutoRenderThread:
        std::snprintf(out, size, "Auto: the render thread only (%u cores of %u logical processors; the recorder thread too "
                      "from %u cores)", cores, cpus, kAutoRecorderCores);
        return;
    case Why::AutoBothThreads:
        std::snprintf(out, size, "Auto: on, the render thread and the recorder thread (%u cores of %u logical processors)",
                      cores, cpus);
        return;
    }
    std::snprintf(out, size, "?");
}

} // namespace mkwvk::threaded
