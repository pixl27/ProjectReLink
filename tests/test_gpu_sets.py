"""Each GPU's shader caches are kept apart (gpu/vk/gpu_sets.h; no game or GPU required): a start on another GPU sets the caches
made on the GPU used before aside in cache/gpu_set_<stamp>/ instead of starting them over, and a start back on that GPU takes
them back (renames, no copy). Another build's sets go, as do sets past kKeep; files that are no GPU's (the notes, the state
lists, the computes) stay where they are. And integrated GPUs prepare nothing before the game (vk_main.cpp
igpu_prepares_in_play), a Steam Deck excepted, with RX_VK_PREPARE_IGPU=1 to prepare there as before.

The header is compiled and run against a temporary folder; the renderer's sources are checked for where it is called.

python -I -B tests/test_gpu_sets.py [--cxx g++]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "g++"]

ROOT = Path(__file__).resolve().parents[1]

DRIVER = r'''
#include "gpu_sets.h"
#include <cstdio>
#include <fstream>
#include <string>
static int bad;
static void check(bool ok, const char* what) { if (!ok) { std::printf("FAIL: %s\n", what); bad = 1; } }
static void put(const std::string& path, const std::string& text) { std::ofstream(path, std::ios::binary) << text; }
static std::string get(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static bool there(const std::string& path) { return std::filesystem::exists(path); }
int main(int, char** argv) {
    const std::string dir = std::string(argv[1]) + "/";
    using namespace gpu_sets;
    const uint64_t build = 0xB0, nv = 0x1111, amd = 0x2222, other_build = 0xB1;
    // (a folder from before the record: whatever is there is left to the caches' own checks; the record is written)
    put(dir + "hookshot_programs.bin", "nv programs");
    put(dir + "hookshot_programs.idx", "nv idx");
    put(dir + "hookshot_vk_pipelines.bin", "nv pipelines");
    put(dir + "hookshot_vk_pipelines_play_3.bin", "nv play");
    put(dir + "hookshot_vk_pipelines.stamp", "nvstamp!");
    put(dir + "hookshot_prepared.bin", "nv prepared");
    put(dir + "hookshot_module_ids.bin", "nv ids");
    put(dir + "hookshot_pipelines.bin", "notes");             // (the notes: no GPU's)
    put(dir + "hookshot_pipelines.bin.built", "nv built");
    put(dir + "hookshot_pipeline_states.bin", "states");      // (the state list: no GPU's)
    put(dir + "hookshot_computes.bin", "computes");
    put(dir + "hookshot_driver_faults.txt", "faults");
    Result r = switch_to(dir, build, nv, "NVIDIA");
    check(r.what == What::kFirst && get(dir + "hookshot_programs.bin") == "nv programs", "a first start leaves the files");
    check(switch_to(dir, build, nv, "NVIDIA").what == What::kSame, "the same GPU again: nothing moves");
    // (the other GPU: NVIDIA's set aside, none for AMD yet)
    r = switch_to(dir, build, amd, "AMD");
    const std::string nv_dir = aside_dir(dir, nv);
    check(r.what == What::kSetAside && r.from == nv && r.aside == 8, "another GPU: the set aside (8 files)");
    check(!there(dir + "hookshot_programs.bin") && get(nv_dir + "hookshot_programs.bin") == "nv programs" &&
          get(nv_dir + "hookshot_vk_pipelines_play_3.bin") == "nv play" && get(nv_dir + "hookshot_pipelines.bin.built") == "nv built",
          "the set's files in cache/gpu_set_<stamp>/");
    check(get(dir + "hookshot_pipelines.bin") == "notes" && get(dir + "hookshot_pipeline_states.bin") == "states" &&
          get(dir + "hookshot_computes.bin") == "computes" && get(dir + "hookshot_driver_faults.txt") == "faults",
          "the notes, the state list, the computes and the driver's faults stay");
    // (AMD makes its own)
    put(dir + "hookshot_programs.bin", "amd programs");
    put(dir + "hookshot_vk_pipelines.bin", "amd pipelines");
    put(dir + "hookshot_prepared.bin", "amd prepared");
    // (back on NVIDIA: its set back, AMD's aside)
    r = switch_to(dir, build, nv, "NVIDIA");
    check(r.what == What::kRestored && r.back == 8 && r.aside == 3, "back on the first GPU: its set back, the other's aside");
    check(get(dir + "hookshot_programs.bin") == "nv programs" && get(dir + "hookshot_vk_pipelines.stamp") == "nvstamp!" &&
          get(dir + "hookshot_prepared.bin") == "nv prepared", "the first GPU's files as they were");
    check(!there(nv_dir) && get(aside_dir(dir, amd) + "hookshot_programs.bin") == "amd programs", "the other GPU's set aside");
    // (and to AMD again: its set back)
    r = switch_to(dir, build, amd, "AMD");
    check(r.what == What::kRestored && get(dir + "hookshot_programs.bin") == "amd programs" &&
          get(aside_dir(dir, nv) + "hookshot_programs.bin") == "nv programs", "the second GPU's set back again");
    // (a third GPU: at most kKeep sets aside, the newest)
    r = switch_to(dir, build, 0x3333, "Intel");
    check(r.what == What::kSetAside && r.dropped_sets == 1 && there(aside_dir(dir, amd)) && !there(aside_dir(dir, nv)),
          "a third GPU: the oldest set aside goes (kKeep 1)");
    // (a new build: its sets aside are no use and go; the files in cache/ stay for the caches' own checks)
    put(dir + "hookshot_programs.bin", "intel programs");
    r = switch_to(dir, other_build, amd, "AMD");
    check(r.what == What::kNew && r.dropped_sets == 1 && !there(aside_dir(dir, amd)) &&
          get(dir + "hookshot_programs.bin") == "intel programs", "a new build: the old build's sets go, cache/ left as it is");
    Owner o;
    check(read_owner(dir, &o) && o.build == other_build && o.stamp == amd, "the record says the build and the stamp");
    // (a set aside of another build is let go at any start, and an unknown folder is not touched)
    std::filesystem::create_directories(dir + "gpu_set_00000000000000ff");
    write_owner(dir + "gpu_set_00000000000000ff/", {build, 0xff}, "");
    std::filesystem::create_directories(dir + "patch-builds");
    put(dir + "patch-builds/x", "x");
    r = switch_to(dir, other_build, amd, "AMD");
    check(r.what == What::kSame && r.dropped_sets == 1 && !there(dir + "gpu_set_00000000000000ff") && there(dir + "patch-builds/x"),
          "another build's set aside goes; other folders stay");
    if (!bad) std::puts("ok");
    return bad;
}
'''


def block(text, marker):
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def check_source():
    prog = (ROOT / "gpu/vk/vk_progcache.cpp").read_text(encoding="utf-8")
    draw = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    main = (ROOT / "gpu/vk/vk_main.cpp").read_text(encoding="utf-8")
    sw = block(prog, "void cache_sets_switch()")
    for needle in ('env_on("RX_VK_NO_GPU_SETS")', "translator_build_stamp()", "gpu_sets::switch_to(dir, build, program_cache_stamp()"):
        if needle not in sw:
            raise SystemExit("vk_progcache.cpp: cache_sets_switch() no longer has: " + needle)
    # (before each set file is first opened: the program cache, the pipeline cache, the prepare record, the warm marker)
    disk = block(prog, "void disk_init()")
    if disk.index("cache_sets_switch();") > disk.index('cache_path(disk_zstd'):
        raise SystemExit("vk_progcache.cpp: disk_init() opens the program cache before the sets are switched")
    drop = block(draw, "bool pcache_drop_stale()")
    if not drop.index("cache_sets_switch();") < drop.index("pcache_stamp::check("):
        raise SystemExit("vk_draw.cpp: pcache_drop_stale() checks the pipeline cache before the sets are switched")
    for name in ("std::string prepared_path()", "std::string precompile_built_path()"):
        line = draw[draw.index(name):draw.index("\n", draw.index(name))]
        if "cache_sets_switch()" not in line:
            raise SystemExit("vk_draw.cpp: %s does not switch the sets first" % name)
    # (the build stamp leaves the GPU out: no dev., profile() or host-> in it)
    shader = (ROOT / "gpu/vk/vk_shader.cpp").read_text(encoding="utf-8")
    bs = block(shader, "uint64_t translator_build_stamp()")
    if "HOOKSHOT_SHADER_STAMP" not in bs or any(n in bs for n in ("dev.", "profile()", "host->")):
        raise SystemExit("vk_shader.cpp: translator_build_stamp() is not the build alone")
    # (integrated GPUs: nothing before the game, a Deck excepted, RX_VK_PREPARE_IGPU=1 as before)
    ig = block(main, "bool igpu_prepares_in_play()")
    for needle in ("VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU", 'env_on("RX_VK_PREPARE_IGPU")', '"SteamDeck"', "Custom GPU 0405",
                   "Custom GPU 0932"):
        if needle not in ig:
            raise SystemExit("vk_main.cpp: igpu_prepares_in_play() no longer has: " + needle)
    up = block(main, "bool started_up()")
    at = up.index("if (!prepare_all && igpu_prepares_in_play() && precompile_wanted()) {")
    if not (up.index("if (prepare_only() && igpu_prepares_in_play())") < up.index("if (prepare_only()) {") and
            "precompile_index_only();" in up[at:at + 400] and at < up.index("if (precompile_wanted()) {")):
        raise SystemExit("vk_main.cpp: started_up() prepares before the game on an integrated GPU")
    sw_def = (ROOT / "gpu/vk/switches.def").read_text(encoding="utf-8")
    doc = (ROOT / "docs/development.md").read_text(encoding="utf-8")
    for name in ("RX_VK_NO_GPU_SETS", "RX_VK_PREPARE_IGPU"):
        if "SW(%s," % name not in sw_def or name not in doc:
            raise SystemExit("%s is not in gpu/vk/switches.def and docs/development.md" % name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="gpu-sets-") as tmp:
        src, exe, work = Path(tmp) / "t.cpp", Path(tmp) / "t.exe", Path(tmp) / "cache"
        work.mkdir()
        src.write_text(DRIVER, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "gpu/vk"), str(src), "-o",
                        str(exe)], check=True)
        r = subprocess.run([str(exe), str(work)], capture_output=True, text=True)
        if r.returncode or "ok" not in r.stdout:
            raise SystemExit("FAIL: the GPU sets:\n" + r.stdout + r.stderr)
    print("PASS: another GPU's caches are kept aside and taken back, another build's let go, kKeep held; integrated GPUs "
          "prepare in play")


if __name__ == "__main__":
    main()
