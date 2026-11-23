"""Whether the pipeline cache's file is read at the start (gpu/vk/pcache_load.h, used by vk_draw.cpp's pcache_load_off), issue
388: not read by default on Linux with NVIDIA's own driver, 20 GB of RAM or less and programs loaded at their create calls,
where NVIDIA's own shader cache is on and its folder holds 512 MB or more; on Windows with NVIDIA's own driver and programs
loaded at create, 20 GB of RAM or less, where %LOCALAPPDATA%/NVIDIA/GLCache holds 512 MB or more; read as before elsewhere;
RX_VK_PCACHE_LOAD says either way. Read, too, where NVIDIA's cache may not grow to 4 GB (gpu/vk/nvidia_cache_limit.h: Windows'
Shader Cache Size, 1 GB on the players' PCs whose caches sat at 1025 MB, or a smaller __GL_SHADER_DISK_CACHE_SIZE on Linux): a
"warm" 1 GB cache drops the pipelines a start builds as it fills, and cannot answer them.

Two parts, no game or GPU:
- the rule's table (pcache_load::choose), compiled and run;
- vk_draw.cpp's pcache_load_off itself, cut out of the source and compiled with stand-ins for the device (its driver ID), the
  load mode and the RAM, then run once a case (its answer is latched for the process) against real folders: NVIDIA's cache
  found where its driver keeps it (__GL_SHADER_DISK_CACHE_PATH's GLCache, XDG_CACHE_HOME, HOME, the earlier drivers' .nv),
  its size alone counted (an empty NVIDIA folder beside a full Mesa folder reads the file), a cache turned off, missing or
  not readable read as cold, NVK left alone, and the log line. The 600 MB files are sparse: they take no disk space.
  POSIX: the Linux folder cases; Windows: the Windows ones (LOCALAPPDATA's NVIDIA/GLCache; AMD's driver reads the file).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

TABLE = r'''
#include "pcache_load.h"
#include <cstdio>
#include <initializer_list>
using namespace pcache_load;
static int fails;
#define NEED(c, what) do { if (!(c)) { fails++; std::printf("FAIL: %s\n", what); } } while (0)
constexpr Cache M = Cache::kMeasured;
// (the rule is constexpr: the table holds at build time too)
static_assert(!choose(-1, false, true, true, 15757, M, 1200).load, "a 16 GB Linux NVIDIA PC with a warm driver cache leaves the file");
static_assert(choose(-1, false, true, true, 32000, M, 1200).load, "a 32 GB PC reads the file");
int main() {
    const uint32_t pc16 = 15757;   // (a 16 GB PC: what it reports)
    // the small PC: not read
    Choice c = choose(-1, false, true, true, pc16, M, 1200);
    NEED(!c.load && c.why == Why::kSmallPc, "16 GB, Linux NVIDIA, create, warm driver cache: not read");
    NEED(!choose(-1, false, true, true, kSmallRamMb, M, kWarmMb).load, "at the limits: not read");
    NEED(needs_cache(-1, false, true, true, pc16), "the small PC looks at NVIDIA's cache");
    // read as before everywhere else
    NEED(choose(-1, false, true, true, kSmallRamMb + 1, M, 1200).why == Why::kRam, "over 20 GB: read");
    NEED(choose(-1, false, true, true, 0, M, 1200).why == Why::kRam, "RAM not known: read");
    NEED(choose(-1, false, true, false, pc16, M, 1200).why == Why::kNotLinuxNvidia, "Windows, AMD, Intel, NVK: read");
    NEED(choose(-1, false, false, true, pc16, M, 1200).why == Why::kStart, "programs loaded before the title: read");
    NEED(choose(-1, true, true, true, pc16, M, 1200).why == Why::kPreparing, "a --prepare-shaders run: read");
    NEED(choose(-1, false, true, true, pc16, M, kWarmMb - 1).why == Why::kColdDriverCache, "a nearly empty driver cache: read");
    NEED(choose(-1, false, true, true, pc16, Cache::kOff, 1200).why == Why::kDriverCacheOff, "NVIDIA's cache off: read");
    NEED(choose(-1, false, true, true, pc16, Cache::kUnknown, 1200).why == Why::kDriverCacheUnknown, "not measured: read");
    for (bool prep : {false, true})
        for (bool create : {false, true})
            for (bool nv : {false, true})
                for (uint32_t ram : {0u, pc16, 32000u})
                    NEED(needs_cache(-1, prep, create, nv, ram) == (!prep && create && nv && ram == pc16),
                         "NVIDIA's folder looked at only where the rest would leave the file");
    // the switch, whatever the rest says
    NEED(choose(1, false, true, true, pc16, M, 1200).load, "RX_VK_PCACHE_LOAD=1 reads it on the small PC");
    NEED(!choose(0, false, false, false, 64000, Cache::kOff, 0).load, "RX_VK_PCACHE_LOAD=0 does not read it anywhere");
    NEED(!choose(0, true, true, true, pc16, M, 1200).load, "RX_VK_PCACHE_LOAD=0 holds in a preparing run too, as before");
    NEED(!needs_cache(0, false, true, true, pc16) && !needs_cache(1, false, true, true, pc16), "the switch walks no folder");
    // Windows with NVIDIA's own driver: not read on a small PC where its GLCache folder is warm (Linux's RAM rule)
    c = choose(-1, false, true, false, pc16, M, 1200, true);
    NEED(!c.load && c.why == Why::kWindowsNvidia, "Windows NVIDIA, 16 GB, create, warm GLCache: not read");
    NEED(!choose(-1, false, true, false, kSmallRamMb, M, kWarmMb, true).load, "Windows NVIDIA at the limits: not read");
    NEED(choose(-1, false, true, false, 32000, M, 1200, true).why == Why::kRam, "Windows NVIDIA with 32 GB: read");
    NEED(choose(-1, false, true, false, 0, M, 1200, true).why == Why::kRam, "Windows NVIDIA, RAM not known: read");
    NEED(needs_cache(-1, false, true, false, pc16, true) && !needs_cache(-1, false, true, false, 32000, true) &&
         !needs_cache(-1, false, true, false, 0, true), "Windows NVIDIA looks at its GLCache folder on a small PC only");
    NEED(choose(-1, false, true, false, pc16, M, kWarmMb - 1, true).why == Why::kColdDriverCache, "Windows: a nearly empty GLCache: read");
    NEED(choose(-1, false, true, false, pc16, Cache::kUnknown, 0, true).why == Why::kDriverCacheUnknown, "Windows: GLCache not measured: read");
    NEED(choose(-1, false, false, false, pc16, M, 1200, true).why == Why::kStart, "Windows: programs loaded before the title: read");
    NEED(choose(-1, true, true, false, pc16, M, 1200, true).why == Why::kPreparing, "Windows: a --prepare-shaders run: read");
    NEED(choose(1, false, true, false, pc16, M, 1200, true).load, "Windows: RX_VK_PCACHE_LOAD=1 reads it");
    NEED(!needs_cache(1, false, true, false, pc16, true) && !needs_cache(-1, false, false, false, pc16, true) &&
         !needs_cache(-1, true, true, false, pc16, true), "Windows: the switch, a start or a preparation walk no folder");
    NEED(choose(-1, false, true, false, pc16, M, 1200, false).why == Why::kNotLinuxNvidia, "Windows AMD or Intel: read");
    // a cache kept small (the 1 GB default players' logs show, 1025 MB "warm"): read, on Windows and Linux
    c = choose(-1, false, true, false, pc16, M, 1025, true, 1024);
    NEED(c.load && c.why == Why::kDriverCacheSmall, "Windows NVIDIA, 16 GB, a 1 GB Shader Cache Size: read");
    NEED(choose(-1, false, true, true, pc16, M, 1025, false, 1024).why == Why::kDriverCacheSmall, "Linux NVIDIA, a 1 GB limit: read");
    NEED(choose(-1, false, true, false, pc16, M, 1200, true, kDriverRoomMb - 1).why == Why::kDriverCacheSmall, "just under 4 GB: read");
    NEED(!choose(-1, false, true, false, pc16, M, 1200, true, kDriverRoomMb).load, "4 GB: not read");
    NEED(!choose(-1, false, true, false, pc16, M, 1200, true, 10240).load, "10 GB (what the launcher sets): not read");
    NEED(!choose(-1, false, true, false, pc16, M, 1200, true, 0xFFFFFFFFull).load, "Unlimited: not read");
    NEED(!choose(-1, false, true, false, pc16, M, 1200, true, 0).load, "a limit not known: as before, not read");
    NEED(!choose(0, false, true, false, pc16, M, 1200, true, 1024).load, "RX_VK_PCACHE_LOAD=0 wins over a small limit");
    for (Why w : {Why::kSwitch, Why::kPreparing, Why::kStart, Why::kNotLinuxNvidia, Why::kRam, Why::kDriverCacheOff,
                  Why::kDriverCacheUnknown, Why::kColdDriverCache, Why::kSmallPc, Why::kWindowsNvidia, Why::kDriverCacheSmall})
        NEED(why_text(w)[0] != 0, "every reason has its words for the log");
    if (!fails) std::printf("ok\n");
    return fails != 0;
}
'''

# vk_draw.cpp's pcache_load_off, with stand-ins: the device's driver (FAKE_DRIVER=nvk: Mesa's NVK), the load mode
# (FAKE_AT_CREATE=0: before the title) and the RAM (FAKE_RAM_MB, through sysconf). Exit 10: not read, 11: read
ADAPTER = r'''
#include "pcache_load.h"
#include "nvidia_cache_limit.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#ifndef _WIN32
#include "compute_gate.h"
#include <unistd.h>
// Redirect only the kernel paths. Exercise the production membership and ancestor traversal.
namespace fixture {
compute_gate::CgroupMemory cgroup_memory() {
    return compute_gate::cgroup_memory("membership", "cgroups");
}
}
#define compute_gate fixture
static long fake_sysconf(int name) {
    const char* m = getenv("FAKE_RAM_MB");
    const long mb = m ? atol(m) : 0;
    return name == _SC_PAGE_SIZE ? 4096 : name == _SC_PHYS_PAGES ? mb * 256 : -1;
}
#define sysconf fake_sysconf
#else
#include <windows.h>
// (the RAM, as pcache_load_off asks Windows for it: FAKE_RAM_MB, none when unset)
static BOOL fake_memory_status(MEMORYSTATUSEX* m) {
    const char* e = getenv("FAKE_RAM_MB");
    if (!e) return FALSE;
    m->ullTotalPhys = (DWORDLONG)atoll(e) << 20;
    return TRUE;
}
#define GlobalMemoryStatusEx fake_memory_status
#endif
enum VkDriverId { VK_DRIVER_ID_AMD_PROPRIETARY = 1, VK_DRIVER_ID_NVIDIA_PROPRIETARY = 4, VK_DRIVER_ID_MESA_NVK = 24 };
struct { VkDriverId driver_id = VK_DRIVER_ID_NVIDIA_PROPRIETARY; } dev;
// (NVIDIA's cache limit: Linux's __GL_SHADER_DISK_CACHE_SIZE read as vk_draw.cpp's nvidia_limit reads it; Windows' Shader Cache
// Size, from the driver's settings there, stood in by FAKE_LIMIT_MB)
static nvidia_cache_limit::Limit nvidia_limit() {
#ifdef _WIN32
    nvidia_cache_limit::Limit l;
    l.where = "Shader Cache Size in the NVIDIA Control Panel";
    if (const char* m = getenv("FAKE_LIMIT_MB")) l.mb = strtoull(m, nullptr, 10);
    return l;
#else
    return nvidia_cache_limit::linux_limit(getenv("__GL_SHADER_DISK_CACHE_SIZE"));
#endif
}
bool programs_load_at_create() { return !env_value_off(getenv("FAKE_AT_CREATE") ? getenv("FAKE_AT_CREATE") : "1"); }
void logf(const char* fmt, ...) {
    va_list a;
    va_start(a, fmt);
    std::vprintf(fmt, a);
    va_end(a);
    std::printf("\n");
}
#include "pcache_load_off.inc"
int main() {
    if (const char* d = getenv("FAKE_DRIVER"); d && !std::strcmp(d, "nvk")) dev.driver_id = VK_DRIVER_ID_MESA_NVK;
    if (const char* d = getenv("FAKE_DRIVER"); d && !std::strcmp(d, "amd")) dev.driver_id = VK_DRIVER_ID_AMD_PROPRIETARY;
    const bool off = pcache_load_off();
    return off == pcache_load_off() ? (off ? 10 : 11) : 2;   // (decided once)
}
'''

MB600 = 600 << 20


def function(src, head):
    if src.count(head) != 1:
        raise SystemExit("FAIL: vk_draw.cpp has no single " + head)
    fn = src[src.index(head):]
    return fn[:fn.index("\n}\n") + 3]


def check_source(src):
    fn = function(src, "bool pcache_load_off() {")
    for needle, why in (
            ('#include "pcache_load.h"', "vk_draw.cpp does not include pcache_load.h"),
            ("dev.driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY", "NVIDIA's own driver is not told apart (NVK has Mesa's cache)"),
            ("pcache_load::choose(sw, preparing, at_create, linux_nvidia, ram_mb, cache.cache,\n"
             "                                                               cache.bytes >> 20, windows_nvidia, limit_mb)",
             "pcache_load_off does not decide through the rule (with Windows NVIDIA's arm and NVIDIA's cache limit)"),
            ("looked ? nvidia_limit()", "pcache_load_off does not ask for NVIDIA's cache limit where it looks at the cache"),
            ('pcache_load::nvidia_cache_windows(getenv("LOCALAPPDATA")', "Windows does not look at NVIDIA's GLCache folder"),
            ("return !choice.load;", "pcache_load_off does not return the rule's answer"),
            ("static const bool off", "the answer is not decided once a run")):
        if needle not in (src if needle.startswith("#include") else fn):
            raise SystemExit(why + " (looked for: " + needle + ")")
    # (NVIDIA's folder alone counts: driver_cache_bytes adds Mesa's to it)
    if "driver_cache_bytes" in fn:
        raise SystemExit("pcache_load_off counts the other drivers' cache folders")
    # (the rest of the pipeline cache still goes by the one answer: not saved when not read)
    init = function(src, "void pcache_init() {")
    if "const bool off = pcache_load_off();" not in init or "pcache_no_save = off;" not in init:
        raise SystemExit("pcache_init no longer takes pcache_load_off's answer for reading and saving")
    return fn


def sparse(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


def cases():
    """(name, env, files {relative path: size}, dirs [relative], expected exit, words the log line has, words it lacks)"""
    nv, mesa = ".cache/nvidia/GLCache/0123abcd/entry", ".cache/mesa_shader_cache_db/part0/mesa_cache.db"
    small = {"FAKE_RAM_MB": "15757"}
    return [
        ("warm", dict(small, HOME="h"), {"h/" + nv: MB600}, [], 10, ["not read", "600 MB in", "RX_VK_PCACHE_LOAD=1 reads it"], []),
        # (hos_main.c's 10 GiB, and a player's 1 GiB: a cache that drops what a start builds)
        ("warm-10g", dict(small, HOME="h", __GL_SHADER_DISK_CACHE_SIZE="10737418240"), {"h/" + nv: MB600}, [], 10,
         ["not read", "up to 10240 MB (__GL_SHADER_DISK_CACHE_SIZE)"], []),
        ("warm-1g", dict(small, HOME="h", __GL_SHADER_DISK_CACHE_SIZE="1073741824"), {"h/" + nv: MB600}, [], 11,
         ["read", "under 4 GB", "up to 1024 MB"], ["not read"]),
        # (the review's case: NVIDIA's folder empty and fresh, Mesa's full: the file is read)
        ("mesa-only", dict(small, HOME="h", __GL_SHADER_DISK_CACHE_PATH="p", MESA_SHADER_CACHE_DIR="m"),
         {"h/" + mesa: MB600, "m/part0/mesa_cache.db": MB600}, ["p/GLCache"], 11, ["read", "nearly empty", "0 MB in"], []),
        ("path-wins", dict(small, HOME="h", __GL_SHADER_DISK_CACHE_PATH="p"), {"h/" + nv: MB600}, ["p/GLCache"], 11,
         ["nearly empty"], []),
        ("path-warm", dict(small, HOME="h", __GL_SHADER_DISK_CACHE_PATH="p"), {"p/GLCache/0123abcd/entry": MB600}, [], 10,
         ["not read"], []),
        ("path-glcache-only", dict(small, __GL_SHADER_DISK_CACHE_PATH="p"), {"p/elsewhere/entry": MB600}, ["p/GLCache"], 11,
         ["nearly empty"], []),
        ("xdg", dict(small, HOME="h", XDG_CACHE_HOME="x"), {"x/nvidia/GLCache/0123abcd/entry": MB600}, [], 10, ["not read"], []),
        ("xdg-not-home", dict(small, HOME="h", XDG_CACHE_HOME="x"), {"h/" + nv: MB600}, [], 11, ["nearly empty"], []),
        ("earlier-driver", dict(small, HOME="h"), {"h/.nv/GLCache/0123abcd/entry": MB600}, ["h/.cache/nvidia/GLCache"], 10,
         ["not read", ".nv"], []),
        ("earlier-driver-empty", dict(small, HOME="h"), {"h/" + nv: MB600}, ["h/.nv/GLCache"], 11, ["nearly empty"], []),
        ("missing", dict(small, HOME="h"), {}, ["h"], 11, ["nearly empty", "0 MB in"], []),
        ("not-a-folder", dict(small, HOME="h"), {"h/.cache/nvidia/GLCache": 10}, [], 11, ["could not be measured"], []),
        ("switched-off", dict(small, HOME="h", __GL_SHADER_DISK_CACHE="0"), {"h/" + nv: MB600}, [], 11, ["is off"], ["MB in"]),
        ("no-folder", dict(small), {}, [], 11, ["is off"], []),
        ("nvk", dict(small, HOME="h", FAKE_DRIVER="nvk"), {"h/" + nv: MB600}, [], 11, ["NVIDIA's own driver"], ["MB in"]),
        ("ram", dict(HOME="h", FAKE_RAM_MB="32000"), {"h/" + nv: MB600}, [], 11, ["32000 MB of RAM"], ["MB in"]),
        ("limited-parent", dict(HOME="h", FAKE_RAM_MB="32000", FAKE_CGROUP_MB="9216"),
         {"h/" + nv: MB600}, [], 10, ["not read", "9216 MB", "600 MB in"], []),
        ("limited-cold", dict(HOME="h", FAKE_RAM_MB="32000", FAKE_CGROUP_MB="9216"),
         {}, [], 11, ["nearly empty", "9216 MB"], []),
        ("larger-parent", dict(HOME="h", FAKE_RAM_MB="15757", FAKE_CGROUP_MB="32768"),
         {"h/" + nv: MB600}, [], 10, ["not read", "15757 MB"], []),
        ("plenty-of-capacity", dict(HOME="h", FAKE_RAM_MB="32000", FAKE_CGROUP_MB="30000"),
         {"h/" + nv: MB600}, [], 11, ["30000 MB", "more than 20 GB"], ["MB in"]),
        ("limited-forced", dict(HOME="h", FAKE_RAM_MB="32000", FAKE_CGROUP_MB="9216", RX_VK_PCACHE_LOAD="1"),
         {"h/" + nv: MB600}, [], 11, ["RX_VK_PCACHE_LOAD=1"], ["MB in"]),
        ("start", dict(small, HOME="h", FAKE_AT_CREATE="0"), {"h/" + nv: MB600}, [], 11, ["before the title"], ["MB in"]),
        ("prepare", dict(small, HOME="h", HOOKSHOT_PREPARE_ONLY="1"), {"h/" + nv: MB600}, [], 11, ["--prepare-shaders"], ["MB in"]),
        ("switch-on", dict(small, HOME="h", RX_VK_PCACHE_LOAD="1"), {"h/" + nv: MB600}, [], 11, ["RX_VK_PCACHE_LOAD=1"],
         ["MB in", "MB of RAM"]),
        # (the switch wins over a preparation, and prints no cache size it did not measure)
        ("switch-off-prepare", dict(small, HOME="h", RX_VK_PCACHE_LOAD="0", HOOKSHOT_PREPARE_ONLY="1"), {}, ["h"], 10,
         ["not read", "RX_VK_PCACHE_LOAD=0"], ["MB in", "reads it"]),
    ]


def windows_cases():
    """(name, env, files, dirs, expected exit, words the line has, words it lacks): LOCALAPPDATA's NVIDIA/GLCache"""
    gl = "NVIDIA/GLCache/dae675479948b9c5/4738267965884b64/b4217564bad97491.bin"
    small = {"FAKE_RAM_MB": "15757"}
    return [
        ("win-warm", dict(small, LOCALAPPDATA="l"), {"l/" + gl: MB600}, [], 10,
         ["not read", "Windows with NVIDIA's own driver", "15757 MB of RAM", "600 MB in", "RX_VK_PCACHE_LOAD=1 reads it"], []),
        ("win-1gb", dict(small, LOCALAPPDATA="l", FAKE_LIMIT_MB="1024"), {"l/" + gl: MB600}, [], 11,
         ["read", "under 4 GB", "up to 1024 MB (Shader Cache Size in the NVIDIA Control Panel)"], ["not read"]),
        ("win-10gb", dict(small, LOCALAPPDATA="l", FAKE_LIMIT_MB="10240"), {"l/" + gl: MB600}, [], 10,
         ["not read", "up to 10240 MB"], []),
        ("win-32gb", dict(LOCALAPPDATA="l", FAKE_RAM_MB="32000"), {"l/" + gl: MB600}, [], 11,
         ["read", "more than 20 GB", "32000 MB of RAM"], ["MB in"]),
        ("win-ram-unknown", dict(LOCALAPPDATA="l"), {"l/" + gl: MB600}, [], 11, ["read", "RAM not known"], ["MB in"]),
        ("win-cold", dict(small, LOCALAPPDATA="l"), {"l/" + gl: 64 << 20}, [], 11, ["read", "nearly empty", "64 MB in"], []),
        ("win-none", dict(small, LOCALAPPDATA="l"), {}, ["l"], 11, ["nearly empty", "0 MB in"], []),
        ("win-dx-only", dict(small, LOCALAPPDATA="l"), {"l/NVIDIA/DXCache/x.bin": MB600}, [], 11, ["nearly empty"], []),
        ("win-no-localappdata", dict(small), {}, [], 11, ["could not be measured"], []),
        ("win-amd", dict(small, LOCALAPPDATA="l", FAKE_DRIVER="amd"), {"l/" + gl: MB600}, [], 11, ["not NVIDIA's own driver"],
         ["MB in"]),
        ("win-start", dict(small, LOCALAPPDATA="l", FAKE_AT_CREATE="0"), {"l/" + gl: MB600}, [], 11, ["before the title"],
         ["MB in"]),
        ("win-prepare", dict(small, LOCALAPPDATA="l", HOOKSHOT_PREPARE_ONLY="1"), {"l/" + gl: MB600}, [], 11,
         ["--prepare-shaders"], ["MB in"]),
        ("win-switch-on", dict(small, LOCALAPPDATA="l", RX_VK_PCACHE_LOAD="1"), {"l/" + gl: MB600}, [], 11,
         ["RX_VK_PCACHE_LOAD=1"], ["MB in"]),
        ("win-switch-off", dict(LOCALAPPDATA="l", FAKE_RAM_MB="32000", RX_VK_PCACHE_LOAD="0"), {}, ["l"], 10,
         ["not read", "RX_VK_PCACHE_LOAD=0"], ["MB in", "reads it"]),
    ]


def run_windows(exe, root):
    failed = []
    keep = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TMP", "TEMP")}
    for name, env, files, dirs, want, has, lacks in windows_cases():
        base = root / name
        base.mkdir()
        for rel, size in files.items():
            sparse(base / rel, size)
        for rel in dirs:
            (base / rel).mkdir(parents=True, exist_ok=True)
        e = dict(keep)
        for k, v in env.items():
            e[k] = str(base / v) if k == "LOCALAPPDATA" else v
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30, env=e, cwd=base)
        line = r.stdout.strip()
        ok = r.returncode == want and line.count("\n") == 0 and all(w in line for w in has) and not any(w in line for w in lacks)
        print(("ok   " if ok else "FAIL ") + name + ": " + line)
        if not ok:
            failed.append(name)
        for rel in files:   # (NTFS keeps no holes by default: the space goes back at once)
            (base / rel).unlink()
    return failed


def run_adapter(exe, root):
    failed = []
    keep = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TMP", "TEMP")}
    for name, env, files, dirs, want, has, lacks in cases():
        base = root / name
        base.mkdir()
        for rel, size in files.items():
            sparse(base / rel, size)
        for rel in dirs:
            (base / rel).mkdir(parents=True, exist_ok=True)
        if "FAKE_CGROUP_MB" in env:
            group = base / "cgroups/group"
            leaf = group / "game"
            leaf.mkdir(parents=True)
            (base / "membership").write_text("0::/group/game\n")
            (leaf / "memory.max").write_text("max\n")
            limit = int(env["FAKE_CGROUP_MB"])
            (group / "memory.max").write_text(str(limit << 20))
            # The compile gate sees 1 MiB left. The file choice needs the hard capacity.
            (group / "memory.current").write_text(str((limit - 1) << 20))
        e = dict(keep)
        for k, v in env.items():
            e[k] = str(base / v) if k in ("HOME", "XDG_CACHE_HOME", "__GL_SHADER_DISK_CACHE_PATH", "MESA_SHADER_CACHE_DIR") else v
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30, env=e, cwd=base)
        line = r.stdout.strip()
        ok = r.returncode == want and line.count("\n") == 0 and all(w in line for w in has) and not any(w in line for w in lacks)
        print(("ok   " if ok else "FAIL ") + name + ": " + line)
        if not ok:
            failed.append(name)
    # (a folder in NVIDIA's cache that cannot be read: not measured, read; not as root, which reads it all the same)
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        base = root / "unreadable"
        sparse(base / "h/.cache/nvidia/GLCache/0123abcd/entry", MB600)
        locked = base / "h/.cache/nvidia/GLCache/0123abcd"
        locked.chmod(0)
        try:
            e = dict(keep, HOME=str(base / "h"), FAKE_RAM_MB="15757")
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30, env=e)
        finally:
            locked.chmod(0o755)
        line = r.stdout.strip()
        ok = r.returncode == 11 and "could not be measured" in line and "could not be walked through" in line
        print(("ok   " if ok else "FAIL ") + "unreadable: " + line)
        if not ok:
            failed.append("unreadable")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++") or "g++")
    args = parser.parse_args()
    fn = check_source((ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8"))
    flags = ["-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "gpu/vk")]
    if os.name == "nt":
        flags += ["-D_CRT_SECURE_NO_WARNINGS"]
    with tempfile.TemporaryDirectory(prefix="pcache-load-") as tmp:
        tmp = Path(tmp)
        cpp, exe = tmp / "table.cpp", tmp / "table.exe"
        cpp.write_text(TABLE, encoding="utf-8")
        subprocess.run([args.cxx, *flags, str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
        (tmp / "pcache_load_off.inc").write_text(fn, encoding="utf-8")
        cpp, exe = tmp / "adapter.cpp", tmp / "adapter.exe"
        cpp.write_text(ADAPTER, encoding="utf-8")
        subprocess.run([args.cxx, *flags, "-I", str(tmp), "-include", str(ROOT / "gpu/vk/compat/hookshot_env.h"), str(cpp),
                        "-o", str(exe)], check=True)
        folders = tmp / "folders"
        folders.mkdir()
        if os.name == "nt":
            failed = run_windows(exe, folders)
        else:
            failed = run_adapter(exe, folders)
    if failed:
        print("FAIL: " + ", ".join(failed))
        return 1
    print("pipeline cache file: not read on a small PC (Linux or Windows) with NVIDIA's own driver and its warm cache, "
          "programs loaded at create; read as before elsewhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
