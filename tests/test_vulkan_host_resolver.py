"""Check that the Windows Vulkan loader uses the executable's resolver, with an old-host fallback."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srccut import block

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cxx", default="clang++")
args = parser.parse_args()

source = block((ROOT / "gpu/vk/vk_main.cpp").read_text(encoding="utf-8"), "bool load_global()")
harness = r"""
#include <cstdint>
#include <cstring>
#define _WIN32 1
using HMODULE = void*;
using PFN_vkGetInstanceProcAddr = void* (*)(void*, const char*);
using PFN_vkCreateInstance = void (*)();
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x800
#define VKX_GLOBAL(X) X(vkCreateInstance)

static int direct_loads, host_resolves;
static void direct_create() {}
static void wrapped_create() {}
static void* direct_gipa(void*, const char*) { return reinterpret_cast<void*>(&direct_create); }
static void* wrapped_gipa(void*, const char*) { return reinterpret_cast<void*>(&wrapped_create); }
static HMODULE LoadLibraryExA(const char*, void*, unsigned flags) {
    if (flags != LOAD_LIBRARY_SEARCH_SYSTEM32) return nullptr;
    ++direct_loads;
    return reinterpret_cast<HMODULE>(1);
}
static void* GetProcAddress(HMODULE, const char* name) {
    if (std::strcmp(name, "vkGetInstanceProcAddr")) return nullptr;
    return reinterpret_cast<void*>(&direct_gipa);
}
struct Host { uintptr_t (*vulkan_loader_proc)(); };
namespace mkwvk { [[maybe_unused]] static const Host* host; }
static uintptr_t resolve() {
    ++host_resolves;
    return reinterpret_cast<uintptr_t>(&wrapped_gipa);
}
static uintptr_t fail_resolve() { return 0; }

static PFN_vkGetInstanceProcAddr get_instance_proc;
static PFN_vkCreateInstance vkCreateInstance;
""" + source + r"""
int main() {
    Host h{resolve};
    mkwvk::host = &h;
    if (!load_global() || vkCreateInstance != wrapped_create || host_resolves != 1 || direct_loads != 0)
        return 1;
    h.vulkan_loader_proc = nullptr;
    if (!load_global() || vkCreateInstance != direct_create || direct_loads != 1)
        return 2;
    h.vulkan_loader_proc = fail_resolve;
    if (load_global() || direct_loads != 1)
        return 3;
}
"""
with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "vulkan_loader.cpp"
    exe = Path(tmp) / ("vulkan_loader.exe" if os.name == "nt" else "vulkan_loader")
    src.write_text(harness, encoding="utf-8")
    subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(src), "-o", str(exe)], check=True)
    status = subprocess.run([str(exe)], check=False).returncode
    if status:
        sys.exit({
            1: "the renderer bypassed the executable's Vulkan resolver",
            2: "the previous host's System32 fallback failed",
            3: "a failed host resolve silently changed resolver paths",
        }.get(status, f"Vulkan loader harness exited with {status}"))
print("Vulkan host resolver, previous-host fallback, and missing resolver checked")
