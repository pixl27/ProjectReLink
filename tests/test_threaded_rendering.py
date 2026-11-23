"""The Insert menu's Threaded rendering (Auto / On / Off): its rule (gpu/vk/threaded_rendering.h), Auto's count of cores
(gpu/vk/vk_progcache.cpp usable_cores) and how the row reaches the replay, the recorder and the host. Compiled checks of
the rule, the count and the readers, plus source checks; no game, no GPU.

The rule is one function every reader shares: On is the render thread and the recorder thread together, Off neither, and
Auto picks by the physical cores among the logical processors the process may run on (a core's SMT siblings counted
once), in three tiers measured on 18 Nov 2026 (a Ryzen 9 9950X3D with SMT off held to N cores, Castle Town's market):
6 or more cores both threads (72.9 / 72.4 fps on 6 against 69.5 / 69.0 with the render thread alone), 4 or 5 the render
thread alone (67.6 / 67.9 fps on 4 against 65.5 / 65.0 serial; both threads made 60.9 / 62.1 there, a third of the frames
16.7 ms or longer), fewer off. Both Auto tiers are off where the mode is not safe, and all of Auto is behind one constant,
threaded::kAutoEnabled, which is on: turned off, Auto is Off on every PC and the row changes nobody's default.

Checked:
 - the rule, compiled: the tiers at every count to 256, the recorder thread never without the render thread, On and Off
   whatever the CPU, the unsafe gate and the gate over both Auto tiers, the start's log line for each answer;
 - the count, compiled: threaded::cores_of on a Steam Deck's 4 cores / 8 threads (4), one sibling of each core usable, the
   Linux and the Windows numbering of siblings, a hybrid CPU, SMT off; and usable_cores itself, cut from vk_progcache.cpp,
   on a stand-in GetLogicalProcessorInformation and affinity mask: the SMT siblings once, the fall-backs to the logical
   count where the cores cannot be read, never more than the logical count;
 - the readers, compiled from their source (vk_replay.cpp render_thread_mode, vk_recorder.h mode_from_env and the hook
   vk_replay.cpp sets) and run in made-up environments: each takes its own half of the row, and RX_VK_ASYNC and
   RX_VK_RECORDER, when set, are over their halves;
 - the source: the gate is the only one and on; the row is Auto by default and takes a restart; the menu resolves the row
   once, from usable_cores, asks the pools' mirrors in both Auto tiers and logs describe's line; RX_VK_ASYNC is read in one
   place; the host's present poll asks the renderer (HOS_OPT_RENDER_THREAD); the device policy and the facts count cores;
   the row's help, the readme, the development guide and the switch table say the tiers;
 - and the checks have teeth: broken copies (the recorder thread on at 4 cores, SMT siblings counted as cores, the 4-core
   tier missing, the logical count back, the recorder hooked to the render thread's half, the mirrors asked in one tier
   only, and others) must each fail.

python -I -B tests/test_threaded_rendering.py [--cxx clang++]
"""
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"

# The rule and the count: threaded_rendering.h alone, and usable_cores cut from vk_progcache.cpp on stand-ins
RULE_FIXTURE = r'''
#include "threaded_rendering.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)
using namespace mkwvk::threaded;

// (the stand-ins usable_cores is built on: the affinity mask, GetLogicalProcessorInformation, usable_cpus)
typedef uintptr_t DWORD_PTR;
typedef uintptr_t ULONG_PTR;
typedef uint32_t DWORD;
typedef void* HANDLE;
typedef int BOOL;
enum LOGICAL_PROCESSOR_RELATIONSHIP { RelationProcessorCore = 0, RelationNumaNode = 1, RelationCache = 2, RelationProcessorPackage = 3 };
struct SYSTEM_LOGICAL_PROCESSOR_INFORMATION {
    ULONG_PTR ProcessorMask;
    LOGICAL_PROCESSOR_RELATIONSHIP Relationship;
    uint64_t Reserved[2];
};
static uint64_t fake_mask = 0;
static bool fake_mask_ok = true;
static std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> fake_info;
static int fake_info_fails = 0;   // (0 none; 1 the size asked for fails; 2 the size given, the call with it fails)
static unsigned fake_cpus = 8;
static HANDLE GetCurrentProcess() { return (HANDLE)(intptr_t)-1; }
static BOOL GetProcessAffinityMask(HANDLE, DWORD_PTR* proc, DWORD_PTR* sys) {
    *proc = (DWORD_PTR)fake_mask, *sys = ~(DWORD_PTR)0;
    return fake_mask_ok;
}
static BOOL GetLogicalProcessorInformation(SYSTEM_LOGICAL_PROCESSOR_INFORMATION* info, DWORD* len) {
    if (fake_info_fails == 1) return 0;
    const DWORD need = (DWORD)(fake_info.size() * sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!info || *len < need) { *len = need; return 0; }
    if (fake_info_fails == 2) return 0;
    for (size_t i = 0; i < fake_info.size(); i++) info[i] = fake_info[i];
    *len = need;
    return 1;
}
static unsigned usable_cpus() { return fake_cpus; }
namespace threaded = mkwvk::threaded;
@@USABLE_CORES@@
// (a CPU for the stand-ins: one entry a core, its logical processors in its mask, and the caches and the package beside)
static void cpu(std::initializer_list<uint64_t> cores) {
    fake_info.clear();
    uint64_t all = 0;
    for (uint64_t m : cores) {
        fake_info.push_back({(ULONG_PTR)m, RelationProcessorCore, {0, 0}});
        fake_info.push_back({(ULONG_PTR)m, RelationCache, {0, 0}});   // (its L1 and L2: not cores)
        all |= m;
    }
    fake_info.push_back({(ULONG_PTR)all, RelationCache, {0, 0}});
    fake_info.push_back({(ULONG_PTR)all, RelationProcessorPackage, {0, 0}});
    fake_info.push_back({(ULONG_PTR)all, RelationNumaNode, {0, 0}});
}

static bool same(const Choice& c, bool render, bool recorder, Why why) {
    return c.render_thread == render && c.recorder_thread == recorder && c.why == why;
}
static std::string line_of(const Choice& c, unsigned cores, unsigned cpus, const char* unsafe) {
    char b[256];
    describe(b, sizeof b, c, cores, cpus, unsafe);
    return b;
}

int main() {
    // (the row's values, in V-Sync's order)
    CHECK(AUTO == 0 && ON == 1 && OFF == 2);
    // (Auto's tiers, measured on 18 Nov: the render thread from 4 cores, the recorder thread too from 6)
    CHECK(kAutoRenderCores == 4 && kAutoRecorderCores == 6);
    for (unsigned cores = 0; cores <= 256; cores++) {
        const Choice c = auto_picks(cores);
        if (cores < 4) CHECK(same(c, false, false, Why::AutoFewCores));
        else if (cores < 6) CHECK(same(c, true, false, Why::AutoRenderThread));
        else CHECK(same(c, true, true, Why::AutoBothThreads));
    }
    // (the recorder thread never without the render thread; Auto picked a thread exactly in its two tiers)
    for (const int setting : {-1, (int)AUTO, (int)ON, (int)OFF, 7})
        for (const bool gate : {false, true})
            for (const bool unsafe : {false, true})
                for (unsigned cores = 0; cores <= 16; cores++) {
                    const Choice c = choose(setting, cores, unsafe, gate);
                    CHECK(!c.recorder_thread || c.render_thread);
                    CHECK(auto_picked(c) == (c.why == Why::AutoRenderThread || c.why == Why::AutoBothThreads));
                    CHECK(auto_picked(c) == (setting != ON && setting != OFF && c.render_thread));
                }
    // (On and Off are the player's word: whatever the CPU, whatever is unsafe, whatever the gate)
    for (const bool gate : {false, true})
        for (const bool unsafe : {false, true})
            for (const unsigned cores : {0u, 2u, 4u, 5u, 6u, 12u, 64u}) {
                CHECK(same(choose(ON, cores, unsafe, gate), true, true, Why::On));
                CHECK(same(choose(OFF, cores, unsafe, gate), false, false, Why::Off));
            }
    // (Auto with the gate on: the three tiers)
    for (const unsigned cores : {0u, 1u, 2u, 3u}) CHECK(same(choose(AUTO, cores, false, true), false, false, Why::AutoFewCores));
    for (const unsigned cores : {4u, 5u}) CHECK(same(choose(AUTO, cores, false, true), true, false, Why::AutoRenderThread));
    for (const unsigned cores : {6u, 7u, 8u, 11u, 12u, 16u, 64u})
        CHECK(same(choose(AUTO, cores, false, true), true, true, Why::AutoBothThreads));
    // (not safe: off in both tiers, and below them)
    for (const unsigned cores : {0u, 3u, 4u, 5u, 6u, 8u, 16u, 64u})
        CHECK(same(choose(AUTO, cores, true, true), false, false, Why::AutoUnsafe));
    // (the gate off: off on every PC, in both tiers, and said so)
    for (const bool unsafe : {false, true})
        for (const unsigned cores : {0u, 3u, 4u, 5u, 6u, 12u, 64u})
            CHECK(same(choose(AUTO, cores, unsafe, false), false, false, Why::AutoGate));
    // (a value that is none of the row's: Auto)
    CHECK(choose(7, 4, false, true).why == Why::AutoRenderThread && choose(-1, 64, false, true).why == Why::AutoBothThreads);
    CHECK(choose(-1, 64, false, false).why == Why::AutoGate);
    // (as built: the gate is on, so Auto is its rule)
    CHECK(kAutoEnabled);
    CHECK(same(choose(AUTO, 3, false), false, false, Why::AutoFewCores));
    CHECK(same(choose(AUTO, 4, false), true, false, Why::AutoRenderThread));
    CHECK(same(choose(AUTO, 6, false), true, true, Why::AutoBothThreads));
    CHECK(same(choose(AUTO, 6, true), false, false, Why::AutoUnsafe) && same(choose(AUTO, 4, true), false, false, Why::AutoUnsafe));

    // (the count: the physical cores among the usable logical processors, a core's SMT siblings once)
    {
        const uint64_t deck[] = {0x3, 0xC, 0x30, 0xC0};   // (4 cores, 8 threads, Windows' numbering: a core's siblings side by side)
        CHECK(cores_of(deck, 4, 0xFF) == 4);
        CHECK(cores_of(deck, 4, 0x55) == 4);   // (one sibling of each core)
        CHECK(cores_of(deck, 4, 0x0F) == 2);   // (4 logical processors: 2 cores)
        CHECK(cores_of(deck, 4, 0x01) == 1 && cores_of(deck, 4, 0) == 0 && cores_of(deck, 0, 0xFF) == 0);
        CHECK(cores_of(deck, 4, ~0ull) == 4);
        CHECK(same(choose(AUTO, cores_of(deck, 4, 0xFF), false, true), true, false, Why::AutoRenderThread));
        uint64_t linux_smt[8];   // (8 cores, 16 threads, Linux's numbering: a core's siblings i and i + 8)
        for (int i = 0; i < 8; i++) linux_smt[i] = 1ull << i | 1ull << (i + 8);
        CHECK(cores_of(linux_smt, 8, 0xFFFF) == 8 && cores_of(linux_smt, 8, 0x000F) == 4 && cores_of(linux_smt, 8, 0x0F0F) == 4);
        CHECK(cores_of(linux_smt, 8, 0x0707) == 3);
        uint64_t smt_off[16];   // (a 9950X3D with SMT off: one logical processor a core, held to N of them as measured)
        for (int i = 0; i < 16; i++) smt_off[i] = 1ull << i;
        CHECK(cores_of(smt_off, 16, 0xF) == 4 && cores_of(smt_off, 16, 0x3F) == 6 && cores_of(smt_off, 16, 0xFF) == 8);
        CHECK(cores_of(smt_off, 16, 0xFFFF) == 16);
        uint64_t hybrid[14];   // (6 performance cores with SMT and 8 efficiency cores without: 14 cores, 20 threads)
        for (int i = 0; i < 6; i++) hybrid[i] = 3ull << (2 * i);
        for (int i = 0; i < 8; i++) hybrid[6 + i] = 1ull << (12 + i);
        CHECK(cores_of(hybrid, 14, (1ull << 20) - 1) == 14 && cores_of(hybrid, 14, 0xFFF) == 6);
        uint64_t wide[32];   // (64 logical processors: the mask's top bit too)
        for (int i = 0; i < 32; i++) wide[i] = 3ull << (2 * i);
        CHECK(cores_of(wide, 32, ~0ull) == 32 && cores_of(wide, 32, 1ull << 63) == 1);
    }
    // (usable_cores as vk_progcache.cpp has it, on the stand-ins)
    {
        cpu({0x3, 0xC, 0x30, 0xC0}), fake_mask = 0xFF, fake_cpus = 8;   // (a Steam Deck: 4 cores, 8 threads)
        CHECK(usable_cores() == 4);
        CHECK(same(choose(AUTO, usable_cores(), false, true), true, false, Why::AutoRenderThread));
        fake_mask = 0x0F, fake_cpus = 4;   // (held to 4 logical processors: 2 cores)
        CHECK(usable_cores() == 2);
        fake_mask = 0x55, fake_cpus = 4;   // (one sibling of each core: 4)
        CHECK(usable_cores() == 4);
        fake_mask_ok = false, fake_cpus = 8;   // (the mask not read: every core)
        CHECK(usable_cores() == 4);
        fake_mask_ok = true, fake_mask = 0;   // (an empty mask: every core)
        CHECK(usable_cores() == 4);
        fake_mask = 0xFF;
        fake_info_fails = 1;   // (the cores not read: the logical count, as before)
        CHECK(usable_cores() == 8);
        fake_info_fails = 2;
        CHECK(usable_cores() == 8);
        fake_info_fails = 0;
        fake_info.clear();   // (no core listed)
        CHECK(usable_cores() == 8);
        fake_info.push_back({(ULONG_PTR)0xFF, RelationCache, {0, 0}});   // (caches only: no core)
        CHECK(usable_cores() == 8);
        cpu({0x3, 0xC}), fake_mask = 0x300, fake_cpus = 2;   // (no listed core in the mask: the logical count)
        CHECK(usable_cores() == 2);
        cpu({0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80}), fake_mask = 0xFF, fake_cpus = 4;   // (never more than the logical count)
        CHECK(usable_cores() == 4);
        const uint64_t one = 1;   // (the 9950X3D with SMT off, as measured: 4, 6 and 8 of its 16)
        cpu({one, one << 1, one << 2, one << 3, one << 4, one << 5, one << 6, one << 7, one << 8, one << 9, one << 10, one << 11,
             one << 12, one << 13, one << 14, one << 15});
        fake_mask = 0xF, fake_cpus = 4;
        CHECK(usable_cores() == 4 && same(choose(AUTO, usable_cores(), false, true), true, false, Why::AutoRenderThread));
        fake_mask = 0x3F, fake_cpus = 6;
        CHECK(usable_cores() == 6 && same(choose(AUTO, usable_cores(), false, true), true, true, Why::AutoBothThreads));
        fake_mask = 0xFF, fake_cpus = 8;
        CHECK(usable_cores() == 8 && same(choose(AUTO, usable_cores(), false, true), true, true, Why::AutoBothThreads));
        cpu({3ull << 0, 3ull << 2, 3ull << 4, 3ull << 6, 3ull << 8, 3ull << 10}), fake_mask = 0xFFF, fake_cpus = 12;
        CHECK(usable_cores() == 6);   // (6 cores, 12 threads: both threads, which Auto gave before only from 12 logical)
    }

    // (the start's line, after "vk: Threaded rendering: ", for each answer)
    CHECK(line_of(choose(ON, 2, true), 2, 4, nullptr) == "On (the render thread and the recorder thread)");
    CHECK(line_of(choose(OFF, 16, false), 16, 16, nullptr) == "Off");
    CHECK(line_of(choose(AUTO, 4, false, false), 4, 8, nullptr) ==
          "Auto: off (Auto is turned off in this build; 4 cores of 8 logical processors, its rule is the render thread from 4 "
          "cores and the recorder thread too from 6)");
    CHECK(line_of(choose(AUTO, 8, true), 8, 8, "RX_VK_DUMP") == "Auto: off (not safe on this run: RX_VK_DUMP)");
    CHECK(line_of(Choice{false, false, Why::AutoUnsafe}, 6, 12, "the buffer pools have no mirrors") ==
          "Auto: off (not safe on this run: the buffer pools have no mirrors)");
    CHECK(line_of(choose(AUTO, 2, false), 2, 4, nullptr) ==
          "Auto: off (2 cores of 4 logical processors; the render thread from 4 cores)");
    CHECK(line_of(choose(AUTO, 4, false), 4, 8, nullptr) ==
          "Auto: the render thread only (4 cores of 8 logical processors; the recorder thread too from 6 cores)");
    CHECK(line_of(choose(AUTO, 5, false), 5, 5, nullptr) ==
          "Auto: the render thread only (5 cores of 5 logical processors; the recorder thread too from 6 cores)");
    CHECK(line_of(choose(AUTO, 6, false), 6, 12, nullptr) ==
          "Auto: on, the render thread and the recorder thread (6 cores of 12 logical processors)");
    {   // (a short buffer: cut, never overrun)
        char b[16];
        std::memset(b, 'x', sizeof b);
        describe(b, 8, choose(AUTO, 4, false), 4, 8, nullptr);
        CHECK(std::strlen(b) == 7 && b[8] == 'x');
    }
    std::printf("PASS: the Threaded rendering rule, Auto's count of cores and the start's line\n");
}
'''

# The readers, cut from their source: the replay's half (render_thread_mode), the recorder's (mode_from_env, through the
# hook vk_replay.cpp sets), on the row as the menu resolves it for the arguments' setting, cores and safety
READERS_FIXTURE = r'''
#define _CRT_SECURE_NO_WARNINGS   // (getenv, as the renderer reads it: MSVC's library calls it unsafe)
#include "threaded_rendering.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
static mkwvk::threaded::Choice row;   // (the menu's threaded_choice, for the arguments)
[[maybe_unused]] static bool menu_render_thread() { return row.render_thread; }
[[maybe_unused]] static bool menu_recorder_thread() { return row.recorder_thread; }
namespace rec {
@@MODE_ENUM@@
inline bool (*g_setting_thread)() = nullptr;
@@PARALLEL@@
@@MODE_FROM_ENV@@
}
@@RENDER_THREAD_MODE@@
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    row = mkwvk::threaded::choose(std::atoi(argv[1]), (unsigned)std::atoi(argv[2]), std::atoi(argv[3]) != 0);
    rec::g_setting_thread = @@HOOK@@;
    bool refused = false;
    const int recorder = rec::mode_from_env(&refused);
    std::printf("render=%d recorder=%d refused=%d\n", render_thread_mode() != 0 ? 1 : 0, recorder, refused ? 1 : 0);
    return 0;
}
'''
REC_OFF, SYNC, DEFER, THREAD = 0, 1, 2, 3   # (vk_recorder.h's Mode)
AUTO, ON, OFF = 0, 1, 2                     # (the row's values)
# (the row's setting, cores, unsafe, the environment's switches; then the render thread, the recorder's mode, refused)
READER_CASES = [
    (AUTO, 2, 0, {}, 0, REC_OFF, 0),
    (AUTO, 3, 0, {}, 0, REC_OFF, 0),
    (AUTO, 4, 0, {}, 1, REC_OFF, 0),     # (the render thread alone: the recorder as with the row Off)
    (AUTO, 5, 0, {}, 1, REC_OFF, 0),
    (AUTO, 6, 0, {}, 1, THREAD, 0),
    (AUTO, 16, 0, {}, 1, THREAD, 0),
    (AUTO, 8, 1, {}, 0, REC_OFF, 0),     # (not safe)
    (AUTO, 4, 1, {}, 0, REC_OFF, 0),
    (ON, 2, 1, {}, 1, THREAD, 0),
    (OFF, 16, 0, {}, 0, REC_OFF, 0),
    # (the switches over their halves)
    (AUTO, 4, 0, {"RX_VK_RECORDER": "thread"}, 1, THREAD, 0),
    (AUTO, 6, 0, {"RX_VK_RECORDER": "sync"}, 1, SYNC, 0),
    (AUTO, 6, 0, {"RX_VK_RECORDER": "off"}, 1, REC_OFF, 0),
    (AUTO, 6, 0, {"RX_VK_RECORDER": "defer"}, 1, DEFER, 0),
    (AUTO, 6, 0, {"RX_VK_ASYNC": "0"}, 0, THREAD, 0),
    (AUTO, 4, 0, {"RX_VK_ASYNC": "0"}, 0, REC_OFF, 0),
    (AUTO, 2, 0, {"RX_VK_ASYNC": "1"}, 1, REC_OFF, 0),
    (OFF, 16, 0, {"RX_VK_ASYNC": "1", "RX_VK_RECORDER": "thread"}, 1, THREAD, 0),
    (ON, 16, 0, {"RX_VK_ASYNC": "0", "RX_VK_RECORDER": "sync"}, 0, SYNC, 0),
    # (the recorder thread refused beside the parallel replay, the row's as the switch's; nothing to refuse with 4 cores)
    (AUTO, 6, 0, {"RX_VK_PARALLEL": "2"}, 1, REC_OFF, 1),
    (AUTO, 4, 0, {"RX_VK_PARALLEL": "2"}, 1, REC_OFF, 0),
]
SWITCHES = ("RX_VK_ASYNC", "RX_VK_RECORDER", "RX_VK_PARALLEL")


def block(text, marker):
    """From marker to the brace closing its first one"""
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def read_sources():
    paths = dict(rule=VK / "threaded_rendering.h", menu=VK / "vk_menu.cpp", menu_h=VK / "vk_menu.h", replay=VK / "vk_replay.cpp",
                 recorder=VK / "vk_recorder.h", api=VK / "vk_api.h", progcache=VK / "vk_progcache.cpp",
                 internal=VK / "vk_internal.h", policy=VK / "device_policy.h", main=VK / "vk_main.cpp",
                 switches=VK / "switches.def", native=ROOT / "host" / "nvn_native.c", readme=ROOT / "readme.md",
                 docs=ROOT / "docs" / "development.md")
    return {k: p.read_text(encoding="utf-8") for k, p in paths.items()}


def constant(rule, name):
    m = re.search(r"constexpr unsigned %s = (\d+);" % name, rule)
    return int(m.group(1)) if m else None


def check_sources(s):
    """The source checks: the failures, an empty list when none"""
    bad = []
    rule, menu = s["rule"], s["menu"]
    render_cores, recorder_cores = constant(rule, "kAutoRenderCores"), constant(rule, "kAutoRecorderCores")
    if render_cores is None or recorder_cores is None:
        return ["threaded_rendering.h: no kAutoRenderCores or kAutoRecorderCores, Auto's two tiers"]
    tiers = (recorder_cores, render_cores, recorder_cores - 1)

    # (the gate: one constant, on, and both of Auto's tiers behind it)
    if len(re.findall(r"constexpr bool \w+ = ", rule)) != 1 or "constexpr bool kAutoEnabled = true;" not in rule:
        bad.append("threaded_rendering.h: Auto's gate is not the one constant kAutoEnabled, on (turning it off changes the "
                   "default for every PC with 4 cores: that is its own change)")
    choose = block(rule, "constexpr Choice choose(")
    if "bool auto_enabled = kAutoEnabled" not in choose or "if (!auto_enabled) return {false, false, Why::AutoGate};" not in choose:
        bad.append("threaded_rendering.h: choose() no longer keeps Auto behind the gate")
    if "kAutoCpus" in rule:
        bad.append("threaded_rendering.h: kAutoCpus, the logical processors' threshold, is back")

    # (the row: Auto by default, read once so it takes a restart)
    if "    int threaded_rendering = 0;" not in s["menu_h"]:
        bad.append("vk_menu.h: the Threaded rendering option is not Auto (0) out of the box")
    row = re.search(r'\{TAB_DEBUG, "Threaded rendering", &opts\.threaded_rendering, 0, 2, 1, tx_vsync,[^{}]*\}', menu)
    if not row or not row.group(0).endswith("hn_threaded, true, true}"):
        bad.append("vk_menu.cpp: no Threaded rendering row (Auto, On, Off), or it no longer takes a restart")
    elif "after a restart" not in row.group(0).replace('"\n     "', ""):
        bad.append("vk_menu.cpp: the Threaded rendering row's help does not say it applies after a restart")
    # (the row's line for Auto says the tiers as the header has them)
    hint = block(menu, "const char* hn_threaded(int v, int variant) {")
    end = hint.find(': "The same as Off, on every PC."')
    auto_line = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', hint[:end])) if end > 0 else ""
    for need in ("%d or more processor cores" % tiers[0], "the render thread alone with %d or %d" % tiers[1:]):
        if need not in auto_line:
            bad.append("vk_menu.cpp: hn_threaded's Auto line does not say " + need)

    # (resolved once, from the cores, the mirrors asked in both Auto tiers, describe's line logged; each reader its half)
    marker = "static const threaded::Choice& threaded_choice() {"
    resolved = block(menu, marker) if marker in menu else ""
    if not resolved:
        bad.append("vk_menu.cpp: no threaded_choice(), the row resolved once")
    for need in ("static const threaded::Choice choice = [] {", "menu_init();", "const unsigned cpus = usable_cpus(), cores = usable_cores();",
                 "threaded::choose(opts.threaded_rendering, cores, unsafe != nullptr)", '"RX_VK_CAPTURE"', '"RX_VK_LOW_SPEC"',
                 '"RX_VK_DUMP"', '"RX_VK_BENCH"', "if (threaded::auto_picked(c) && !pools_mirrored()) {",
                 "c = threaded::Choice{false, false, threaded::Why::AutoUnsafe};",
                 "threaded::describe(line, sizeof line, c, cores, cpus, unsafe);", 'logf("vk: Threaded rendering: %s", line);'):
        if resolved and need not in resolved:
            bad.append("vk_menu.cpp: threaded_choice() lost " + need + " (resolved once, from the cores the process may run on, "
                       "off for Auto in both tiers where the mode is not safe, and logged)")
    if "RX_VK_ASYNC" in resolved or "RX_VK_RECORDER" in resolved:
        bad.append("vk_menu.cpp: threaded_choice() is the row's answer alone: the switches are read over it by their own readers")
    for need in ("bool menu_render_thread() { return threaded_choice().render_thread; }",
                 "bool menu_recorder_thread() { return threaded_choice().recorder_thread; }"):
        if need not in menu:
            bad.append("vk_menu.cpp: no " + need.split(" {")[0] + " giving its own half of the row")
    if "bool menu_render_thread();" not in s["menu_h"] or "bool menu_recorder_thread();" not in s["menu_h"]:
        bad.append("vk_menu.h: menu_render_thread and menu_recorder_thread are not declared")

    # (the replay: RX_VK_ASYNC in one place, the switch over the row's render thread)
    mode = block(s["replay"], "int render_thread_mode() {")
    if 'const char* e = getenv("RX_VK_ASYNC");' not in mode or "return e ? atoi(e) : menu_render_thread() ? 1 : 0;" not in mode:
        bad.append("vk_replay.cpp: render_thread_mode() is no longer RX_VK_ASYNC when set, else the row's render thread as 1 or 0")
    if "render_thread_mode() != 0" not in block(s["replay"], "bool async_on() {"):
        bad.append("vk_replay.cpp: async_on() no longer reads render_thread_mode()")
    if 'getenv("RX_VK_ASYNC")' not in block(menu, "const char* threaded_env() {"):
        bad.append("vk_menu.cpp: RX_VK_ASYNC is read outside threaded_env(), the row's note that the switch is set")
    # (the recorder: the switch when set; unset, the row's recorder thread, through the pointer vk_replay.cpp sets)
    if "inline bool (*g_setting_thread)() = nullptr;" not in s["recorder"] or \
            "(rec::g_setting_thread = menu_recorder_thread, true)" not in s["replay"]:
        bad.append("the recorder's question to the row is not its own half (vk_recorder.h g_setting_thread, vk_replay.cpp: "
                   "menu_recorder_thread)")

    # (the host: the present poll asks the renderer, which answers async_on(), the render thread's half)
    if "enum { HOS_OPT_RENDER_THREAD = 58 };" not in s["api"]:
        bad.append("vk_api.h: HOS_OPT_RENDER_THREAD is not option 58")
    if "if (id == HOS_OPT_RENDER_THREAD) return async_on();" not in block(s["replay"], "t.option = [](int id) -> int {"):
        bad.append("vk_replay.cpp: the option callback no longer answers HOS_OPT_RENDER_THREAD with async_on()")
    poll = block(s["native"], "NV(nvnEventGetValue) {")
    if "now = e ? !env_value_off(e) : rx_option(HOS_OPT_RENDER_THREAD, -1);" not in poll or 'getenv("HOS_PRESENT_NOWAIT")' not in poll:
        bad.append("nvn_native.c: the present poll is not HOS_PRESENT_NOWAIT when set, else the renderer's HOS_OPT_RENDER_THREAD")

    # (the count: usable_cores beside usable_cpus, by the cores' masks, declared; the facts and the policy count cores too)
    marker = "unsigned usable_cores() {"
    cores = block(s["progcache"], marker) if marker in s["progcache"] else ""
    if not cores:
        bad.append("vk_progcache.cpp: no usable_cores()")
    for need in ("GetLogicalProcessorInformation(", "RelationProcessorCore", "threaded::cores_of(", "usable_cpus()",
                 "GetProcessAffinityMask("):
        if cores and need not in cores:
            bad.append("vk_progcache.cpp: usable_cores() lost " + need)
    if "unsigned usable_cores();" not in s["internal"]:
        bad.append("vk_internal.h: usable_cores is not declared")
    if "threaded::auto_picks(f.cores_usable)" not in s["policy"] or "pf.cores_usable = usable_cores();" not in s["main"]:
        bad.append("the device policy's Threaded rendering flags are not by the cores (device_policy.h f.cores_usable, "
                   "vk_main.cpp pf.cores_usable = usable_cores())")

    # (no reader left on the old one-half answer)
    for key in ("menu", "menu_h", "replay", "recorder", "policy", "native", "rule"):
        if "menu_threaded_rendering" in s[key] or re.search(r"\bWhy::Auto(Cpus|FewCpus)\b", s[key]):
            bad.append("%s still names menu_threaded_rendering or Why::AutoCpus: the row has two halves now" % key)

    # (said where the settings and the switches are)
    readme, docs = s["readme"], s["docs"]
    table = [line for line in readme.splitlines() if line.startswith("| | Developer: Threaded rendering | Auto, On, Off")]
    start = readme.find("- **Threaded rendering:**")
    para = readme[start:start + 900].replace("\n  ", " ") if start >= 0 else ""
    if len(table) != 1 or not para:
        bad.append("readme.md: no Threaded rendering line in the option table, or no paragraph on it")
    else:
        for where, text in (("the option table", table[0]), ("the paragraph", para)):
            if "%d or more" % tiers[0] not in text or "%d or %d" % tiers[1:] not in text or "12 or more" in text:
                bad.append("readme.md: %s does not say Auto's tiers (%d or more cores, %d or %d)" % ((where,) + tiers))
    rows = [line for line in docs.splitlines() if line.startswith("| Threaded rendering (the Insert menu's row")]
    if len(rows) != 1 or "threaded::kAutoEnabled" not in rows[0]:
        bad.append("docs/development.md: no row for Threaded rendering, or it does not name the gate")
    elif "%d or more cores, the render thread and the recorder thread" % tiers[0] not in rows[0] or \
            "%d or %d, the render thread alone" % tiers[1:] not in rows[0] or "usable_cores" not in rows[0]:
        bad.append("docs/development.md: the Threaded rendering row does not say Auto's tiers by cores (usable_cores)")
    for name, need in (("RX_VK_ASYNC", "on from %d physical cores" % render_cores),
                       ("RX_VK_RECORDER", "Auto from %d physical cores" % recorder_cores)):
        m = re.search(r'^SW\(%s, \w+, \w+, "([^"]*)"' % name, s["switches"], re.M)
        if not m or need not in m.group(1):
            bad.append("switches.def: %s's default does not say %s" % (name, need))
    return bad


def compile_and_run(cxx, source, include, tmp, name, runs):
    """Build the fixture against `include` (threaded_rendering.h's folder) and run it once for each (arguments, environment)
    of runs: (built, the compiler's output or the completed runs)"""
    cpp, exe = Path(tmp) / (name + ".cpp"), Path(tmp) / (name + (".exe" if os.name == "nt" else ""))
    cpp.write_text(source, encoding="utf-8")
    r = subprocess.run([cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I" + str(include), str(cpp), "-o", str(exe)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode:
        return False, r.stdout + r.stderr
    return True, [subprocess.run([str(exe)] + [str(a) for a in args], capture_output=True, text=True, timeout=60, env=env)
                  for args, env in runs]


def with_rule(s, tmp, name):
    """A folder of its own with this copy's threaded_rendering.h"""
    inc = Path(tmp) / (name + "-include")
    inc.mkdir(exist_ok=True)
    with open(inc / "threaded_rendering.h", "w", encoding="utf-8", newline="\n") as f:
        f.write(s["rule"])
    return inc


def check_rule(cxx, s, tmp, name="rule"):
    """The rule's and the count's fixture: None when it passes, else why not ('build: ...' when it did not build)"""
    source = RULE_FIXTURE.replace("@@USABLE_CORES@@", block(s["progcache"], "unsigned usable_cores() {"))
    built, out = compile_and_run(cxx, source, with_rule(s, tmp, name), tmp, name, [([], None)])
    if not built:
        return "build: " + out
    return None if out[0].returncode == 0 else (out[0].stdout + out[0].stderr).strip()


def check_readers(cxx, s, tmp, name="readers"):
    """The readers' fixture in each of READER_CASES' environments: None when every one answers as expected"""
    hook = re.search(r"\(rec::g_setting_thread = (\w+), true\)", s["replay"])
    enum = re.search(r"enum Mode : int \{[^}]*\};", s["recorder"])
    source = READERS_FIXTURE.replace("@@MODE_ENUM@@", enum.group(0) if enum else "") \
                            .replace("@@PARALLEL@@", block(s["recorder"], "inline bool parallel_configured() {")) \
                            .replace("@@MODE_FROM_ENV@@", block(s["recorder"], "inline int mode_from_env(bool* refused) {")) \
                            .replace("@@RENDER_THREAD_MODE@@", block(s["replay"], "int render_thread_mode() {")) \
                            .replace("@@HOOK@@", hook.group(1) if hook else "nullptr")
    base = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    runs = [([setting, cores, unsafe], dict(base, **environment)) for setting, cores, unsafe, environment, *_ in READER_CASES]
    built, out = compile_and_run(cxx, source, with_rule(s, tmp, name), tmp, name, runs)
    if not built:
        return "build: " + out
    for (setting, cores, unsafe, environment, render, recorder, refused), run in zip(READER_CASES, out):
        want = "render=%d recorder=%d refused=%d" % (render, recorder, refused)
        if run.returncode or run.stdout.strip() != want:
            return "the row %d, %d cores, unsafe %d, %s: %s, expected %s" % (setting, cores, unsafe, environment or "no switch",
                                                                            (run.stdout + run.stderr).strip(), want)
    return None


# Broken copies, each of which a check must catch: (what, the source, its text, the text in its place, the check)
BROKEN = [
    ("the recorder thread on at 4 cores", "rule", "    if (cores >= kAutoRecorderCores) return {true, true, Why::AutoBothThreads};",
     "    if (cores >= kAutoRenderCores) return {true, true, Why::AutoBothThreads};", "rule"),
    ("the recorder thread in the 4-core tier", "rule", "return {true, false, Why::AutoRenderThread};",
     "return {true, true, Why::AutoRenderThread};", "rule"),
    ("the 4-core tier missing", "rule", "    if (cores >= kAutoRenderCores) return {true, false, Why::AutoRenderThread};\n", "",
     "rule"),
    ("the recorder thread without the render thread", "rule", "return {true, false, Why::AutoRenderThread};",
     "return {false, true, Why::AutoRenderThread};", "rule"),
    ("the old threshold, both threads from 12", "rule", "constexpr unsigned kAutoRecorderCores = 6;",
     "constexpr unsigned kAutoRecorderCores = 12;", "rule"),
    ("SMT siblings counted as cores", "rule", "cores += (core_masks[i] & usable) != 0;",
     "cores += (unsigned)__builtin_popcountll(core_masks[i] & usable);", "rule"),
    ("the unsafe gate on the both-threads tier only", "rule", "if (unsafe) return {false, false, Why::AutoUnsafe};",
     "if (unsafe && cores >= kAutoRecorderCores) return {false, false, Why::AutoUnsafe};", "rule"),
    ("Auto's gate over the both-threads tier only", "rule", "if (!auto_enabled) return {false, false, Why::AutoGate};",
     "if (!auto_enabled && cores >= kAutoRecorderCores) return {false, false, Why::AutoGate};", "rule"),
    ("the render-thread tier logged as on", "rule", '"Auto: the render thread only (', '"Auto: on (', "rule"),
    ("usable_cores counting logical processors", "progcache",
     "threaded::cores_of(masks.data(), (unsigned)masks.size(), (uint64_t)proc)", "usable_cpus()", "rule"),
    ("usable_cores with no fall-back to the logical count", "progcache", "return cores ? std::min(cores, cpus) : cpus;",
     "return cores ? std::min(cores, cpus) : 0;", "rule"),
    ("usable_cores counting every core of the PC, not the mask's", "progcache",
     "threaded::cores_of(masks.data(), (unsigned)masks.size(), (uint64_t)proc)",
     "threaded::cores_of(masks.data(), (unsigned)masks.size(), ~0ull)", "rule"),
    ("the recorder hooked to the render thread's half", "replay", "(rec::g_setting_thread = menu_recorder_thread, true)",
     "(rec::g_setting_thread = menu_render_thread, true)", "readers"),
    ("the replay on the recorder thread's half", "replay", "return e ? atoi(e) : menu_render_thread() ? 1 : 0;",
     "return e ? atoi(e) : menu_recorder_thread() ? 1 : 0;", "readers"),
    ("the row over RX_VK_RECORDER", "recorder", 'if (!e && g_setting_thread && g_setting_thread()) e = "thread";',
     'if (g_setting_thread && g_setting_thread()) e = "thread";', "readers"),
    ("the menu counting logical processors", "menu", "threaded::choose(opts.threaded_rendering, cores, unsafe != nullptr)",
     "threaded::choose(opts.threaded_rendering, cpus, unsafe != nullptr)", "source"),
    ("the mirrors asked in the both-threads tier only", "menu", "if (threaded::auto_picked(c) && !pools_mirrored()) {",
     "if (c.why == threaded::Why::AutoBothThreads && !pools_mirrored()) {", "source"),
    ("the menu's recorder half from the render thread", "menu",
     "bool menu_recorder_thread() { return threaded_choice().recorder_thread; }",
     "bool menu_recorder_thread() { return threaded_choice().render_thread; }", "source"),
    ("the device policy by logical processors", "policy", "threaded::auto_picks(f.cores_usable)",
     "threaded::auto_picks(f.cpus_usable)", "source"),
    ("the row's help with the old threshold", "menu", '"On with 6 or more processor cores', '"On with 12 or more processor cores',
     "source"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    s = read_sources()

    bad = check_sources(s)
    if bad:
        raise SystemExit("FAIL: " + "\nFAIL: ".join(bad))
    for path in sorted(list(VK.glob("*.cpp")) + list(VK.glob("*.h")) + list((ROOT / "host").glob("*.c"))):
        text = path.read_text(encoding="utf-8", errors="replace")
        reads = len(re.findall(r'getenv\("RX_VK_ASYNC"\)', text))
        # (vk_menu.cpp's one is threaded_env, the row's line that says the switch is set: it decides nothing)
        if reads != (1 if path.name in ("vk_replay.cpp", "vk_menu.cpp") else 0):
            raise SystemExit("FAIL: %s reads RX_VK_ASYNC itself (%d times): only render_thread_mode() decides by it, once, so "
                             "the row and the switch cannot disagree" % (path.relative_to(ROOT).as_posix(), reads))

    with tempfile.TemporaryDirectory(prefix="threaded-rendering-") as tmp:
        why = check_rule(a.cxx, s, tmp)
        if why:
            raise SystemExit("FAIL: the rule's fixture: " + why)
        print("PASS: the Threaded rendering rule, Auto's count of cores and the start's line")
        why = check_readers(a.cxx, s, tmp)
        if why:
            raise SystemExit("FAIL: the readers: " + why)
        print("PASS: the readers take their halves, the switches over them (%d environments)" % len(READER_CASES))

        for i, (what, key, old, new, check) in enumerate(BROKEN):
            if s[key].count(old) != 1:
                raise SystemExit("FAIL: broken copy %r: its text is not once in %s" % (what, key))
            broken = dict(s, **{key: s[key].replace(old, new)})
            if check == "source":
                caught = bool(check_sources(broken))
            else:
                why = (check_rule if check == "rule" else check_readers)(a.cxx, broken, tmp, "broken%d" % i)
                if why and why.startswith("build: "):
                    raise SystemExit("FAIL: broken copy %r did not build, so it proves nothing:\n%s" % (what, why))
                caught = why is not None
            if not caught:
                raise SystemExit("FAIL: broken copy %r passed the %s check" % (what, check))
        print("PASS: %d broken copies caught" % len(BROKEN))
    print("PASS: Threaded rendering: the rule's three tiers by cores, the gate (on), and the row's way to the replay, the "
          "recorder and the host")


if __name__ == "__main__":
    main()
