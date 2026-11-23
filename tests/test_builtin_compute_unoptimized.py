"""The renderer's own compute pipelines are made without the driver's optimizer where it refuses them with it
(gpu/vk/vk_draw.cpp extern_guarded_compute, gpu/vk/vk_texture.cpp texture_compute_pipeline).

Intel's Windows driver on the Arc A-series (an A580 on 101.9175) answers VK_ERROR_UNKNOWN (-13) for some optimized
compute builds that it makes without the optimizer: the game's own dispatches show it in the log ("the optimized build was
not made (-13); the one without the optimizer stays"). The renderer's own compute pipelines - the GPU ASTC decoder, the BC
encoders, the GPU unswizzle, the upscale and mask helpers, the write-back diff, the late pages' fix, FSR 3's inputs - were
asked once, with the optimizer, and given up on. On the A580 the ASTC decoder was never made and every ASTC texture was
decoded on the CPU (a frame's uploads 30.9 s, two 1365 x 1365 x 90 arrays 11 s each): the screen black after the opening
cutscene and at a save's load. The typed upscale helper standing in for a network left out was not made either (the
network's output a blit of RGB9E5 bits read as B10G11R11), nor the write-back diff and the late pages' fix.

The check runs the real texture_compute_pipeline and extern_guarded_compute against a driver that refuses every optimized
build (the A580's) and needs the pipeline made, without the optimizer, by both; against one that makes it optimized
(NVIDIA, AMD), that build is kept and no other asked for; against one that refuses both, an error and no pipeline. It needs
vk_draw.cpp's typed upscale helper to make its pipeline through texture_compute_pipeline (it called the driver itself,
once), and the ASTC decoder's failure to be logged with its result (the A580's log had no line for it).

python -I -B tests/test_builtin_compute_unoptimized.py --headers <Vulkan-Headers/include> [--cxx g++]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "gpu" / "vk" / "vk_texture.cpp"
DRAW = ROOT / "gpu" / "vk" / "vk_draw.cpp"

FIXTURE = r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct IntelDriverGate { IntelDriverGate() {} };
void logf(const char*, ...) {}
// (the driver: 0 refuses an optimized compute build with VK_ERROR_UNKNOWN, as Intel's on the A580; 1 makes every build;
// 2 refuses every build)
int driver = 0, asks = 0, plain_asks = 0;
VkResult ask(const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
    asks++;
    const bool plain = (ci->flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT) != 0;
    plain_asks += plain;
    if (driver == 2 || (driver == 0 && !plain)) return *pl = VK_NULL_HANDLE, VK_ERROR_UNKNOWN;
    *pl = (VkPipeline)(uintptr_t)(plain ? 0x2 : 0x1);
    return VK_SUCCESS;
}
VkResult guarded_compute(VkPipelineCache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) { return ask(ci, pl); }
// (the pipeline cache's path: the driver's answer as well, through the play cache)
VkResult shared_compute_pipeline(const VkComputePipelineCreateInfo* ci, VkPipeline* out) { return ask(ci, out); }
/*EXTERN*/
/*TEXTURE*/
int check(const char* who, VkResult r, VkPipeline out) {
    std::printf("%s: result %d, pipeline %p, %d asks (%d without the optimizer)\n", who, (int)r, (void*)out, asks, plain_asks);
    if (driver == 0 && (r != VK_SUCCESS || out != (VkPipeline)(uintptr_t)0x2))
        return std::puts("FAIL: a driver that refuses only the optimized build is left with no pipeline"), 1;
    if (driver == 1 && (r != VK_SUCCESS || out != (VkPipeline)(uintptr_t)0x1 || plain_asks))
        return std::puts("FAIL: a driver that makes the optimized build does not keep it alone"), 1;
    if (driver == 2 && (r == VK_SUCCESS || out != VK_NULL_HANDLE))
        return std::puts("FAIL: a driver that refuses every build is said to have made one"), 1;
    return 0;
}
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    driver = atoi(argv[1]);
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkPipeline out = (VkPipeline)(uintptr_t)0x99;
    if (!strcmp(argv[2], "texture")) return check("texture_compute_pipeline", texture_compute_pipeline(ci, out), out);
    const VkResult r = extern_guarded_compute(VK_NULL_HANDLE, &ci, &out);
    return check("extern_guarded_compute", r, out);
}
"""


def need(ok, why):
    if not ok:
        raise SystemExit("FAIL: " + why)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headers", type=Path, required=True)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    tex = TEX.read_text(encoding="utf-8")
    draw = DRAW.read_text(encoding="utf-8")
    fixture = FIXTURE.replace("/*EXTERN*/", block(
        draw, "VkResult extern_guarded_compute(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {",
        where=DRAW))
    fixture = fixture.replace("/*TEXTURE*/", block(
        tex, "VkResult texture_compute_pipeline(const VkComputePipelineCreateInfo& ci, VkPipeline& out) {", where=TEX))
    with tempfile.TemporaryDirectory(prefix="builtin-compute-") as temp:
        src, exe = Path(temp) / "make.cpp", Path(temp) / "make.exe"
        src.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
                        "-Wno-unused-parameter", "-I" + str(a.headers), str(src), "-o", str(exe)], check=True)
        for path in ("extern", "texture"):
            for driver, name in ((0, "Intel's on the Arc A-series"), (1, "one that makes it optimized"), (2, "one that refuses all")):
                r = subprocess.run([str(exe), str(driver), path], text=True, capture_output=True, timeout=20)
                need(r.returncode == 0, f"{name}: {r.stdout.strip()}")
    upscale = block(draw, "const char* keys, bool temporal) {", where=DRAW)
    need("texture_compute_pipeline(ci, pipe)" in upscale,
         "the typed upscale helper asks the driver for its pipeline itself, once, with the optimizer")
    astc = block(tex, "bool astc_ready() {", where=TEX)
    need("ASTC decoder" in astc and "(int)r" in astc, "the GPU ASTC decoder's pipeline can fail with nothing in the log")
    print("PASS: the renderer's own compute pipelines are made without the optimizer where the driver refuses them with it, "
          "and kept optimized elsewhere")


if __name__ == "__main__":
    main()
