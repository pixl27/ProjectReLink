"""The presents on a queue of a compute family of their own only when the Debug tab's Present queue row or
RX_VK_PRESENT_QUEUE asks for it (off by default on every PC).

On the replay's queue a present waits behind the game's next frame, so frame generation's frames, each presented at its own
time, reached the screen together: with V-Sync Off at 3x a third of them under 2 ms after the one before (an RTX 5090 on
621.24), 0.1% or less on a queue of their own. A change made the queue the default with NVIDIA's driver 621.24 and newer on
Windows; a tester's RTX 5080 on that driver then lost the GPU with frame generation on (2x and 3x, Built-in FG and DLSS
FG, mailbox and FIFO), and the crashes stopped with RX_VK_PRESENT_QUEUE=0 (another RTX 5080 looks the same).
v2.0.9 had turned it off after an RTX 4090 on 621.14 had a black screen with it (Hookshot issue 1161). So it is opt-in again.

The behaviour (gpu/vk/vk_main.cpp create_device's queue choice, cut out and compiled, its Windows branch included on any
host): with the switch unset and no row saved no family is chosen, for NVIDIA's driver 621.24 and newer too; the row's
present_queue=1 in hookshot_options.ini (read at the start, as robust_buffers=) chooses the first compute family without
graphics, =0 none; RX_VK_PRESENT_QUEUE over the row either way (1 or any on value: chosen; 0, false, off, no: none; empty:
the row); with no such family none. The row: on the Debug tab, On or Off, applied after a restart, its key present_queue.
The text checks: the device-lost message names the switch while the queue is in use, DLSS Frame Gen's exit wait bounds
the queues' own locks too (a present on the presents' queue lets dev.present_mu go), and docs/development.md and
gpu/vk/switches.def have the switch's row with its default.

    python -I -B tests/test_present_queue_default.py --cxx clang++ --headers <Vulkan-Headers include>
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import between  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r"""
#define VK_NO_PROTOTYPES   // (the device call below is the fixture's own)
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "hookshot_env.h"

static VkDriverId driver_id = VK_DRIVER_ID_NVIDIA_PROPRIETARY;
static struct { VkPhysicalDevice physical = nullptr; VkPhysicalDeviceProperties props{}; } dev;
static void logf(const char*, ...) {}
static std::string ini;   // (hookshot_options.ini as the fixture writes it; empty: no file)
static std::string options_path() { return ini; }
static bool reshade_compat() { return false; }   // (Hookshot's ReShade rule, vk_main.cpp: no ReShade here)
static void vkGetPhysicalDeviceProperties2(VkPhysicalDevice, VkPhysicalDeviceProperties2* p) {
    auto* d = static_cast<VkPhysicalDeviceDriverProperties*>(p->pNext);
    if (d) d->driverID = driver_id;
}
static uint32_t choose(const std::vector<VkQueueFamilyProperties>& families) {
    const uint32_t n = (uint32_t)families.size();
    /* CHOICE */
    return compute_family;
}
static void set_env(const char* v) {
#ifdef _WIN32
    _putenv_s("RX_VK_PRESENT_QUEUE", v ? v : "");
#else
    if (v) setenv("RX_VK_PRESENT_QUEUE", v, 1); else unsetenv("RX_VK_PRESENT_QUEUE");
#endif
}
static void set_ini(const char* line) {   // (nullptr: no options file)
    static int n = 0;
    ini = line ? std::string("present-queue-") + std::to_string(++n) + ".ini" : std::string("no-such-options-file.ini");
    if (line)
        if (FILE* f = std::fopen(ini.c_str(), "w")) std::fprintf(f, "vsync=0\n%s\nrobust_buffers=0\n", line), std::fclose(f);
}
static int fails = 0;
static void expect(const char* what, uint32_t got, uint32_t want) {
    if (got != want) {
        std::printf("FAIL: %s: family %d, expected %d (-1: the replay's queue)\n", what, (int)got, (int)want);
        fails++;
    }
}
int main() {
    const VkQueueFlags gfx = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    std::vector<VkQueueFamilyProperties> nvidia(3), no_compute(2);
    nvidia[0].queueFlags = gfx, nvidia[1].queueFlags = VK_QUEUE_TRANSFER_BIT,
    nvidia[2].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;   // (an NVIDIA GPU's: graphics, transfer, compute)
    no_compute[0].queueFlags = gfx, no_compute[1].queueFlags = VK_QUEUE_TRANSFER_BIT;
    const uint32_t none = UINT32_MAX;
    set_ini(nullptr);
    for (uint32_t version : {(621u << 22) | (14u << 14), (621u << 22) | (24u << 14), (630u << 22)}) {
        dev.props.driverVersion = version;
        driver_id = VK_DRIVER_ID_NVIDIA_PROPRIETARY;
        set_env(nullptr);
        expect("NVIDIA, the switch unset", choose(nvidia), none);
        set_env("");
        expect("NVIDIA, the switch empty", choose(nvidia), none);
        for (const char* on : {"1", "on", "yes", "true"}) set_env(on), expect("NVIDIA, the switch on", choose(nvidia), 2);
        for (const char* off : {"0", "off", "no", "false"}) set_env(off), expect("NVIDIA, the switch off", choose(nvidia), none);
    }
    driver_id = VK_DRIVER_ID_AMD_PROPRIETARY;
    set_env(nullptr);
    expect("AMD, the switch unset", choose(nvidia), none);
    set_env("1");
    expect("AMD, the switch on", choose(nvidia), 2);
    expect("no compute-only family, the switch on", choose(no_compute), none);
    driver_id = VK_DRIVER_ID_NVIDIA_PROPRIETARY, dev.props.driverVersion = (621u << 22) | (24u << 14);
    set_env(nullptr);
    set_ini("present_queue=1");
    expect("the Debug tab's Present queue On, the switch unset", choose(nvidia), 2);
    set_ini("present_queue=0");
    expect("the Debug tab's Present queue Off, the switch unset", choose(nvidia), none);
    set_ini("present_queue=1"), set_env("0");
    expect("the row On, RX_VK_PRESENT_QUEUE=0 over it", choose(nvidia), none);
    set_ini("present_queue=0"), set_env("1");
    expect("the row Off, RX_VK_PRESENT_QUEUE=1 over it", choose(nvidia), 2);
    set_ini("present_queue=1"), set_env("");
    expect("the row On, the switch empty", choose(nvidia), 2);
    set_ini("present_queue=1"), set_env(nullptr);
    expect("the row On, no compute-only family", choose(no_compute), none);
    if (fails) return 1;
    std::puts("PASS: the presents' own queue only when the Present queue row or RX_VK_PRESENT_QUEUE asks for it");
    return 0;
}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    p.add_argument("--headers", type=Path, required=True)
    a = p.parse_args()
    main_cpp = (ROOT / "gpu" / "vk" / "vk_main.cpp").read_text(encoding="utf-8")
    choice = between(main_cpp, "uint32_t compute_family = UINT32_MAX;", "// (hookshot) The presenter thread's own queue",
                     where="gpu/vk/vk_main.cpp create_device")
    # (the Windows branch, where there is one, on any host: the default it sets is what is checked)
    fixture = FIXTURE.replace("/* CHOICE */", choice.replace("#ifdef _WIN32", "#if 1"))
    with tempfile.TemporaryDirectory(prefix="present-queue-default-") as tmp:
        cpp, exe = Path(tmp) / "fixture.cpp", Path(tmp) / ("fixture.exe" if os.name == "nt" else "fixture")
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-Wno-unused-parameter", "-Wno-missing-field-initializers", "-I" + str(a.headers), "-I" + str(ROOT / "gpu" / "vk" / "compat"),
                        str(cpp), "-o", str(exe)], check=True)
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20, cwd=tmp)
        sys.stdout.write(run.stdout)
        assert run.returncode == 0, "create_device's queue choice: the presents' own queue only when the Present queue row or RX_VK_PRESENT_QUEUE asks"
    menu = (ROOT / "gpu" / "vk" / "vk_menu.cpp").read_text(encoding="utf-8")
    row = between(menu, '{TAB_DEBUG, "Present queue", &opts.present_queue, 0, 1, 1, tx_onoff,', "},", where="gpu/vk/vk_menu.cpp")
    assert row.rstrip().endswith("true, true"),         "the Debug tab's Present queue row: Off or On, applied after a restart (restart), the Quality preset left alone"
    assert "int present_queue = 0;" in (ROOT / "gpu" / "vk" / "vk_menu.h").read_text(encoding="utf-8"),         "vk_menu.h: Options::present_queue, 0 Off by default"
    assert 'if (const char* e = getenv("RX_VK_PRESENT_QUEUE"); e && *e) env_override(&opts.present_queue, !env_value_off(e));'         in menu and '(value == &opts.present_queue && getenv("RX_VK_PRESENT_QUEUE") && *getenv("RX_VK_PRESENT_QUEUE"))' in menu,         "vk_menu.cpp: with RX_VK_PRESENT_QUEUE set the row shows the switch's value and is fixed (as ASTC as RGBA8's)"
    dev = (ROOT / "docs" / "development.md").read_text(encoding="utf-8")
    row = [l for l in dev.splitlines() if l.startswith("| `RX_VK_PRESENT_QUEUE=1` |")]
    assert len(row) == 1 and "off by default on every PC" in row[0], \
        "docs/development.md: the RX_VK_PRESENT_QUEUE row and its default"
    sw = (ROOT / "gpu" / "vk" / "switches.def").read_text(encoding="utf-8")
    assert "SW(RX_VK_PRESENT_QUEUE, player, bool, \"follows present_queue= in hookshot_options.ini" in sw, \
        "gpu/vk/switches.def: a player's switch over the Present queue row, off on every PC"
    replay = (ROOT / "gpu" / "vk" / "vk_replay.cpp").read_text(encoding="utf-8")
    lost = between(replay, "[[noreturn]] void device_lost(const char* why) {", "\n}\n", where="gpu/vk/vk_replay.cpp")
    assert "dev.present_queue_supported ?" in lost and "RX_VK_PRESENT_QUEUE=0" in lost and "replay_fatal(lost);" in lost, \
        "device_lost: the way back named while the presents are on their own queue"
    dlss = (ROOT / "gpu" / "vk" / "vk_dlss.cpp").read_text(encoding="utf-8")
    close = between(dlss, "void dlss_fg_close() {\n    // (not only where it is asked for now", "\n}\n", where="gpu/vk/vk_dlss.cpp")
    assert "!free_by_then(dev.present_queue_mu) || !free_by_then(dev.presenter_queue_mu)" in close and \
        close.index("free_by_then(dev.present_queue_mu)") < close.index("device_wait_idle_locked();"), \
        "dlss_fg_close: the presents' own queues free before the idle wait, within the exit's 2 s"
    print("PASS: the presents on their own compute queue only with RX_VK_PRESENT_QUEUE=1")


if __name__ == "__main__":
    main()
