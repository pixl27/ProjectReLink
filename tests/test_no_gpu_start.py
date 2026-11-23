"""A start where Vulkan lists no GPU at all: the game ends at once, and the window after it says why.

A Linux player on a real-time kernel (CachyOS's rt-bore, PREEMPT_RT) had NVIDIA's driver not loaded: the Vulkan loader
listed no device, the renderer did not start, and the game played on behind a black window with its sound, until the
player closed it. The start-failure window then said to update the graphics driver. Checked here, with no game, GPU or
window:

- The renderer's failure ends the game at once (host/posix/display_sdl.c renderer_main, host/display.c display_main's
  start of the renderer, both cut from the source): the line the crash window reads, the log flushed, the run's marker
  closed as a clean end, exit code 1. A renderer that starts goes on.
- What the PC says of its graphics driver when no device is listed (gpu/vk/compat/hookshot_no_gpu.h, on made-up /sys and
  /proc trees): an NVIDIA card with NVIDIA's driver not loaded on a real-time kernel, not loaded on another kernel, loaded
  with no Vulkan device, and anything else not known; the cards' PCI ids in the line.
- The crash window (host/hos_crash_dialog.h) takes the line emitted by pick_device and suggests troubleshooting for the
  observations, or "found no graphics card" when nothing more is known. A
  log without the line keeps the Vulkan 1.3 message.

    python -I -B tests/test_no_gpu_start.py --cc clang --cxx clang++
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import between, block  # noqa: E402

RUN_ALL_ARGS = ["--cc", "gcc", "--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
NO_GPU_H = ROOT / "gpu" / "vk" / "compat" / "hookshot_no_gpu.h"

# (the renderer's start as both displays have it, with stand-ins: ExitProcess goes back to the case through longjmp)
DISPLAY = r"""
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
typedef void *HANDLE;
typedef unsigned long DWORD;
typedef unsigned int UINT;
#define WINAPI
static jmp_buf ended;
static int exit_code, flushed, cleaned_after_flush, signalled, rx_ok, resized;
static char last_log[512];
static void hos_log(const char *format, ...) {
    va_list a;
    va_start(a, format);
    vsnprintf(last_log, sizeof last_log, format, a);
    va_end(a);
}
static void hos_log_flush(void) { flushed = 1; }
void hos_crash_clean_end(void) { cleaned_after_flush = flushed; }
static void ExitProcess(UINT code) { exit_code = (int)code; longjmp(ended, 1); }
static void SetEvent(HANDLE h) { (void)h; signalled++; }
static int rx_start(void *w) { (void)w; return rx_ok; }
static void rx_resize(int w, int h) { (void)w, (void)h; resized++; }
typedef struct { int unused; } Pointer;
static Pointer pointer;
static void hos_pointer_size(Pointer *p, int *w, int *h) { (void)p; *w = 1280, *h = 720; }
static int vk_window;
static HANDLE renderer_up;
static volatile int renderer;
static int cancelled;
static int rx_start_failure_claim(void) { return !cancelled; }
static void *hwnd;
__attribute__((unused)) static void *use[] = {(void *)hos_log_flush, (void *)ExitProcess, (void *)rx_resize, (void *)hos_pointer_size,
                                             &vk_window, &pointer};

#include "linux_renderer_main.inc"
static void windows_start(void) {
#include "windows_start.inc"
}
int main(int argc, char **argv) {
    (void)argc;
    const int windows = !strcmp(argv[1], "windows");
    rx_ok = !strcmp(argv[2], "ok");
    cancelled = !strcmp(argv[2], "cancelled");
    exit_code = -1;
    if (!setjmp(ended)) {
        if (windows) windows_start();
        else renderer_main(NULL);
    }
    printf("exit=%d flushed=%d clean=%d signalled=%d renderer=%d resized=%d\n%s\n", exit_code, flushed, cleaned_after_flush,
           signalled, (int)renderer, resized, last_log);
    return 0;
}
"""

# (pick_device itself, with Vulkan and menu stand-ins; its log goes to the crash window)
PICK = r"""
#include "hookshot_no_gpu.h"
#include <cstdint>
#include <cstdlib>
#include <cstdarg>
#include <string>
#include <vector>
using VkPhysicalDevice = int;
struct VkPhysicalDeviceProperties { char deviceName[256]; uint32_t apiVersion, driverVersion, vendorID, deviceID; int deviceType; };
struct Device { int instance, physical, memory; VkPhysicalDeviceProperties props; } dev;
struct Card { uint32_t vendor, device; };
static std::vector<std::string> gpu_device_names;
static std::vector<Card> gpu_device_cards;
static struct { int gpu_device; } opts;
namespace gpu_choice { static int menu_device(int, int) { return -1; } }
#define VK_API_VERSION_MAJOR(v) ((v) >> 22)
#define VK_API_VERSION_MINOR(v) (((v) >> 12) & 1023)
#define VK_API_VERSION_1_3 ((1u << 22) | (3u << 12))
#define VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU 2
#define VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU 1
static void menu_init() {}
static void menu_set_gpu_count(int) {}
static void vkEnumeratePhysicalDevices(int, uint32_t *n, VkPhysicalDevice *) { *n = 0; }
static void vkGetPhysicalDeviceProperties(int, VkPhysicalDeviceProperties *) {}
static void vkGetPhysicalDeviceMemoryProperties(int, int *) {}
static void logf(const char *format, ...) {
    va_list a; va_start(a, format); vprintf(format, a); va_end(a); putchar('\n');
}
static const char *fixture_root;
#define hookshot_no_gpu_probe(root, g) hookshot_no_gpu_probe(fixture_root, g)
#include "pick_device.inc"
int main(int argc, char **argv) { (void)argc; fixture_root = argv[1]; return pick_device() ? 1 : 0; }
"""
DIALOG = r"""
#include "hos_crash_dialog.h"
int main(int argc, char **argv) {
    HosCrashLog f;
    memset(&f, 0, sizeof f);
    for (int i = 1; i < argc; i++) hos_crash_log_line(&f, argv[i]);
    HosCrashRecord r;
    hos_crash_record_parse("", &r);
    const int v = hos_crash_verdict(&r, &f, 1, 1, 0), cause = hos_crash_cause(v, &r, &f, 1, 1);
    char message[2000];
    hos_crash_message(message, sizeof message, v, hos_crash_stage(&f), cause);
    printf("vulkan=%d failed=%d verdict=%d\n%s\n", f.vulkan, hos_crash_start_failed(&f), v, message);
    return 0;
}
"""
HEAD = ["[hos host] hookshot nightly-680-9dd41326 (source e287b3d8)", "[hos host] run_id: 0123456789abcdef0123456789abcdef"]
START_SHOWN = 3   # (HOS_CRASH_SHOW_START)


def write(root, files):
    """A made-up file system: {path: text}, a path ending in / a folder"""
    for name, text in files.items():
        path = root / name
        if name.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def card(slot, vendor, device, cls):
    # (a PCI slot's folder is named 0000:01:00.0; Windows takes no ':' in a name, and the probe reads none of it)
    base = "sys/bus/pci/devices/%s/" % (slot.replace(":", "_") if os.name == "nt" else slot)
    return {base + "vendor": vendor + "\n", base + "device": device + "\n", base + "class": cls + "\n"}


RTX_5060 = {**card("0000:01:00.0", "0x10de", "0x2d05", "0x030000"),
            **card("0000:01:00.1", "0x10de", "0x22eb", "0x040300"),   # (its HDMI sound: no graphics card)
            **card("0000:00:14.0", "0x8086", "0x7ae0", "0x0c0330")}   # (a USB controller)
PCS = {
    "rt": ({**RTX_5060, "sys/kernel/realtime": "1\n", "proc/sys/kernel/version": "#1 SMP PREEMPT_RT Thu Nov 13 2026\n",
            "sys/module/nvidia_drm/": ""}, "nvidia-rt"),
    "rt-version": ({**RTX_5060, "proc/sys/kernel/version": "#1 SMP PREEMPT_RT Thu Nov 13 2026\n"}, "nvidia-rt"),
    "unloaded": ({**RTX_5060, "proc/sys/kernel/version": "#1 SMP PREEMPT_DYNAMIC Thu Nov 13 2026\n"}, "nvidia-unloaded"),
    "not-rt": ({**RTX_5060, "sys/kernel/realtime": "0\n"}, "nvidia-unloaded"),
    "loaded": ({**RTX_5060, "sys/kernel/realtime": "1\n", "sys/module/nvidia/": ""}, "nvidia-vulkan"),
    "amd": ({**card("0000:03:00.0", "0x1002", "0x744c", "0x030000"), "sys/kernel/realtime": "1\n"}, "unknown"),
    "hybrid": ({**RTX_5060, **card("0000:03:00.0", "0x1002", "0x744c", "0x030000"),
                "sys/kernel/realtime": "1\n"}, "unknown"),
    "empty": ({}, "unknown"),
}
MESSAGES = {   # (what the window must say for each cause, and what it must not)
    "nvidia-rt": (("real-time", "standard kernel"), "Update your graphics driver"),
    "nvidia-unloaded": (("NVIDIA's kernel module isn't loaded", "Restart your PC"), "real-time"),
    "nvidia-vulkan": (("NVIDIA's kernel module is loaded", "Vulkan"), "real-time"),
    "unknown": (("found no graphics card",), "real-time"),
}


def run(exe, *args):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True, timeout=30, creationflags=flags)
    if r.returncode:
        raise SystemExit("FAIL: %s %s exited %d: %s" % (exe.name, " ".join(map(str, args)), r.returncode, r.stderr))
    return r.stdout


def compile_c(cc, source, exe, *includes):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run([cc, "-std=gnu11", "-D_CRT_SECURE_NO_WARNINGS", "-Wall", "-Wextra", "-Werror",
                        *("-I" + str(i) for i in includes), str(source), "-o", str(exe)],
                       capture_output=True, text=True, timeout=120, creationflags=flags)
    if r.returncode:
        raise SystemExit("FAIL: %s does not compile:\n%s" % (source.name, r.stdout + r.stderr))


def check_display(cxx, tmp):
    linux = (ROOT / "host/posix/display_sdl.c").read_text(encoding="utf-8")
    windows = (ROOT / "host/display.c").read_text(encoding="utf-8")
    (tmp / "linux_renderer_main.inc").write_text(block(linux, "static DWORD WINAPI renderer_main(void *p)") + "\n",
                                                 encoding="utf-8")
    (tmp / "windows_start.inc").write_text(between(windows, "    renderer = rx_start(hwnd);", "    SetTimer(hwnd, 1, 250, NULL);"),
                                           encoding="utf-8")
    (tmp / "display.c").write_text(DISPLAY, encoding="utf-8")
    exe = tmp / ("display" + (".exe" if os.name == "nt" else ""))
    compile_c(cxx, tmp / "display.c", exe, tmp)
    lines = []
    for system in ("linux", "windows"):
        head, line = run(exe, system, "fail").splitlines()[:2]
        if not head.startswith("exit=1 flushed=1 clean=1 signalled=0 renderer=0 "):
            raise SystemExit("FAIL: %s: the renderer did not start and the game went on (%s): a black window with sound until "
                             "the player closes it, and only then the window that says why" % (system, head))
        if not line.startswith("display: the renderer did not start"):
            raise SystemExit("FAIL: %s: the renderer's failure logged %r, not the line the crash window reads" % (system, line))
        lines.append("[hos host] " + line)
        head = run(exe, system, "ok").splitlines()[0]
        if not head.startswith("exit=-1 ") or "signalled=1" not in head or "renderer=1" not in head:
            raise SystemExit("FAIL: %s: a renderer that started did not go on: %s" % (system, head))
    head = run(exe, "linux", "cancelled").splitlines()[0]
    if not head.startswith("exit=-1 flushed=0 clean=0 signalled=0 renderer=0 "):
        raise SystemExit("FAIL: cancelled startup reported a failure: " + head)
    print("PASS: the renderer's failure ends the game at once, on Linux and Windows; a renderer that starts goes on")
    return lines


def check_probe_and_window(cc, cxx, tmp, display_lines):
    if not NO_GPU_H.is_file():
        raise SystemExit("FAIL: nothing asks the PC why Vulkan listed no GPU (%s missing): the window after a start where "
                         "NVIDIA's driver is not loaded (a real-time kernel) can only say to update the graphics driver"
                         % NO_GPU_H.relative_to(ROOT).as_posix())
    suffix = ".exe" if os.name == "nt" else ""
    (tmp / "dialog.c").write_text(DIALOG, encoding="utf-8")
    probe, dialog = tmp / ("probe" + suffix), tmp / ("dialog" + suffix)
    main_src = (ROOT / "gpu/vk/vk_main.cpp").read_text(encoding="utf-8")
    (tmp / "pick_device.inc").write_text(block(main_src, "bool pick_device() {") + "\n", encoding="utf-8")
    (tmp / "pick.cpp").write_text(PICK, encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run([cxx, "-std=c++17", "-D_CRT_SECURE_NO_WARNINGS", "-I" + str(NO_GPU_H.parent),
                             str(tmp / "pick.cpp"), "-o", str(probe)], capture_output=True, text=True,
                            timeout=120, creationflags=flags)
    if result.returncode:
        raise SystemExit("FAIL: pick_device does not compile:\n" + result.stdout + result.stderr)
    compile_c(cc, tmp / "dialog.c", dialog, ROOT / "host")
    for name, (files, want) in PCS.items():
        root = tmp / ("pc-" + name)
        root.mkdir()
        write(root, files)
        output = run(probe, root).splitlines()
        line = next((s for s in output if s.startswith("vk: no Vulkan device (cause ")), "")
        if not line.startswith("vk: no Vulkan device (cause %s)" % want):
            raise SystemExit("FAIL: %s: the line does not start with its cause: %r" % (name, line))
        cards = "1002:744c" if name == "amd" else "10de:2d05" if files else ""
        if cards not in line or "10de:22eb" in line or "8086:7ae0" in line:
            raise SystemExit("FAIL: %s: the graphics cards in the line are wrong: %r" % (name, line))
        log = HEAD + ["[hos host] " + s for s in output] + ["[hos host] vk: pick_device failed",
                      "[hos host] rx: mkwvk_init returned NULL: nothing is drawn"] + display_lines[:1]
        out = run(dialog, *log)
        if not out.startswith("vulkan=1 failed=1 verdict=%d" % START_SHOWN):
            raise SystemExit("FAIL: %s: the window does not show a start that failed: %s" % (name, out))
        need, never = MESSAGES[want]
        if any(n not in out for n in need) or never in out:
            raise SystemExit("FAIL: %s (%s): the window does not name the cause (wants %s, never %r):\n%s"
                             % (name, want, need, never, out))
    print("PASS: the PC's driver for %d made-up PCs, and the window names each cause" % len(PCS))
    out = run(dialog, *HEAD, "[hos host] vk: no Vulkan 1.3 GPU")
    if "Vulkan 1.3" not in out or "found no graphics card" in out:
        raise SystemExit("FAIL: a log without the line (devices, none with Vulkan 1.3) lost its message:\n" + out)
    for line in display_lines:
        out = run(dialog, *HEAD, line)
        if not out.startswith("vulkan=1 failed=1 verdict=%d" % START_SHOWN):
            raise SystemExit("FAIL: the crash window does not read %r as a start that failed: %s" % (line, out))
    print("PASS: devices without Vulkan 1.3 keep their message; both displays' line is a start that failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", ""))
    args = parser.parse_args()
    cxx = (shutil.which(args.cxx) if args.cxx else
           next((p for name in ("clang++", "g++", "c++") if (p := shutil.which(name))), None))
    cc = shutil.which(args.cc)
    if not cc:
        raise SystemExit("C compiler not found: " + args.cc)
    if not cxx:
        raise SystemExit("C++ compiler not found" + (": " + args.cxx if args.cxx else "; use --cxx to specify one"))
    # The synthetic sysfs reader uses narrow paths on Windows. Public is normally ASCII and writable even when
    # the user's temporary directory contains characters outside the active code page.
    temp_root = os.environ.get("PUBLIC") if os.name == "nt" else None
    if os.name == "nt" and (not temp_root or not temp_root.isascii()):
        raise SystemExit("An ASCII-only PUBLIC directory is needed for the Windows probe fixture")
    with tempfile.TemporaryDirectory(prefix="hookshot-no-gpu-", dir=temp_root) as folder:
        tmp = Path(folder)
        lines = check_display(cc, tmp)
        check_probe_and_window(cc, cxx, tmp, lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
