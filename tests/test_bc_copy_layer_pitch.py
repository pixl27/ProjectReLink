"""The copy out of the ASTC-to-BC encoder's buffer: explicit block pitch, and per-layer regions on AMD's Windows driver (gpu/vk/vk_texture.cpp).

An ASTC array kept as BC3/BC7 is decoded into a scratch a run of layers at a time, encoded into a buffer
(host_shaders/bc_encoder.comp: block (z * blocks_y + y) * blocks_x + x, 16 bytes each) and copied into the image by
bc_encode_into. On AMD's Windows driver (diagnosed on RDNA 2: RX 6700 XT, RX 6600), a single multi-layer
VkBufferImageCopy (layerCount > 1) targeting a non-4-aligned compressed mip-tail level (such as 5x5 mip 8 on the
1365 x 1365 x 90 terrain material arrays) truncates the copy to floor(w/4) x floor(h/4) blocks per layer, leaving the
right and bottom edge blocks zeroed (pure black) even when bufferRowLength and bufferImageHeight are explicit.

When dev.policy.bc_copy_per_layer is on (AMD's Windows driver by default; RX_VK_BC_COPY_PER_LAYER), bc_encode_into splits
non-4-aligned multi-layer copies into n single-layer VkBufferImageCopy regions (chunked by 8, so 1 call for normal runs
of up to 8 layers) while keeping 8-layer decode/encode runs intact. When it is off (every other driver) or the level is
4-aligned or single-layer, it keeps the single VkBufferImageCopy region with explicit whole-block pitch.

python -I -B tests/test_bc_copy_layer_pitch.py --headers <Vulkan-Headers/include> [--cxx g++]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "gpu" / "vk" / "vk_texture.cpp"

FIXTURE = r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
struct Texture { VkImage image = (VkImage)1; };
struct { struct { bool bc_copy_per_layer = false; } policy; } dev;
VkPipeline bc_pipe = (VkPipeline)1, bc7_pipe = (VkPipeline)2;
VkPipelineLayout bc_layout = (VkPipelineLayout)1;
VkImageView bc_scratch_view = (VkImageView)1;
VkBuffer bc_out = (VkBuffer)1;
bool bc7_enabled() { return false; }
void full_barrier(VkCommandBuffer) {}
void descriptors_disturbed() {}
void vkCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
void vkCmdPushDescriptorSetKHR(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkWriteDescriptorSet*) {}
void vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t) {}
/*BCPUSH*/
BcPush pushed{};
std::vector<VkBufferImageCopy> copied;
int copy_calls = 0;
void vkCmdPushConstants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, uint32_t, uint32_t size, const void* p) {
    if (size == sizeof pushed) std::memcpy(&pushed, p, size);
}
void vkCmdCopyBufferToImage(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, uint32_t count, const VkBufferImageCopy* c) {
    copy_calls++;
    for (uint32_t i = 0; i < count; ++i) copied.push_back(c[i]);
}
/*ENCODE*/
int main(int argc, char** argv) {
    if (argc != 7) return 2;
    const uint32_t w = (uint32_t)atoi(argv[1]), h = (uint32_t)atoi(argv[2]), n = (uint32_t)atoi(argv[3]);
    const int dx = atoi(argv[4]), dy = atoi(argv[5]);
    const bool per_layer = atoi(argv[6]) != 0;
    dev.policy.bc_copy_per_layer = per_layer;
    Texture t;
    constexpr uint32_t kLevel = 8, kBaseLayer = 3;
    bc_encode_into((VkCommandBuffer)1, w, h, n, t, kLevel, kBaseLayer, dx, dy);
    // Independently derive expected block counts and byte layout from (w, h).
    const uint32_t exp_bx = (w + 3u) / 4u, exp_by = (h + 3u) / 4u;
    const uint64_t exp_row = (uint64_t)exp_bx * 16u, exp_layer = exp_row * exp_by;
    if (pushed.blocks_x != exp_bx || pushed.blocks_y != exp_by)
        return std::puts("FAIL: encoder push constants do not match ceil(w/4) x ceil(h/4)"), 1;
    const bool expect_split = per_layer && n > 1 && ((w & 3u) || (h & 3u));
    const size_t want_regions = expect_split ? (size_t)n : 1u;
    if (copied.size() != want_regions) {
        std::printf("FAIL: got %zu copy regions, want %zu (per_layer=%d, %ux%u x%u)\n", copied.size(), want_regions,
                    (int)per_layer, w, h, n);
        return 1;
    }
    const int want_calls = expect_split ? (int)((n + 7u) / 8u) : 1;
    if (copy_calls != want_calls) {
        std::printf("FAIL: got %d vkCmdCopyBufferToImage calls, want %d\n", copy_calls, want_calls);
        return 1;
    }
    for (size_t i = 0; i < copied.size(); ++i) {
        const VkBufferImageCopy& c = copied[i];
        if (!c.bufferRowLength || !c.bufferImageHeight)
            return std::puts("FAIL: bufferRowLength or bufferImageHeight left at 0"), 1;
        if (c.bufferRowLength != exp_bx * 4u || c.bufferImageHeight != exp_by * 4u)
            return std::puts("FAIL: buffer pitch does not match ceil(w/4)*4 x ceil(h/4)*4"), 1;
        if (c.imageOffset.x != dx || c.imageOffset.y != dy || c.imageOffset.z != 0)
            return std::puts("FAIL: copy imageOffset does not match (dx, dy, 0)"), 1;
        if (c.imageExtent.width != w || c.imageExtent.height != h || c.imageExtent.depth != 1)
            return std::puts("FAIL: copy extent does not match level dimensions"), 1;
        if (c.imageSubresource.mipLevel != kLevel)
            return std::puts("FAIL: copy mipLevel mismatch"), 1;
        if (expect_split) {
            if (c.imageSubresource.layerCount != 1 || c.imageSubresource.baseArrayLayer != kBaseLayer + (uint32_t)i)
                return std::puts("FAIL: per-layer region does not target layer base+i with layerCount=1"), 1;
            if (c.bufferOffset != (uint64_t)i * exp_layer)
                return std::puts("FAIL: per-layer region bufferOffset does not match i * layer_bytes"), 1;
        } else {
            if (c.imageSubresource.layerCount != n || c.imageSubresource.baseArrayLayer != kBaseLayer || c.bufferOffset != 0)
                return std::puts("FAIL: single-region copy does not cover [base, base+n) at offset 0"), 1;
        }
    }
    return 0;
}
"""


def need(ok, why):
    if not ok:
        raise SystemExit("FAIL: " + why)


def default_cxx():
    return os.environ.get("CXX") or ("clang++" if shutil.which("clang++") and not shutil.which("g++") else "g++")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headers", type=Path, required=True)
    p.add_argument("--cxx", default=default_cxx())
    a = p.parse_args()
    tex = TEX.read_text(encoding="utf-8")
    fixture = FIXTURE.replace("/*BCPUSH*/", block(tex, "struct BcPush {", where=TEX) + ";")
    fixture = fixture.replace("/*ENCODE*/", block(tex, "void bc_encode_into(VkCommandBuffer cmd,", where=TEX))
    with tempfile.TemporaryDirectory(prefix="bc-copy-pitch-") as temp:
        src, exe = Path(temp) / "pitch.cpp", Path(temp) / "pitch.exe"
        src.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
                        "-Wno-unused-parameter", "-I" + str(a.headers), str(src), "-o", str(exe)], check=True)
        # Terrain 1365x1365 mips (including 5x5 mip 8 that triggers the AMD Windows truncation), other
        # non-4-aligned mips (6x6, 9x9, 683x170), 4-aligned 1024x1024, single-layer 1365x1365, non-zero (dx, dy)
        # sub-region copy (as texture_copy_into_bc), and n > 8 to exercise the chunked vkCmdCopyBufferToImage loop.
        cases = (
            (682, 682, 8, 0, 0), (341, 341, 8, 0, 0), (85, 85, 8, 0, 0), (5, 5, 8, 0, 0), (6, 6, 8, 0, 0),
            (9, 9, 8, 0, 0), (2, 2, 8, 0, 0), (1, 1, 8, 0, 0), (1024, 1024, 8, 0, 0), (1365, 1365, 1, 0, 0),
            (683, 170, 3, 0, 0), (5, 5, 4, 4, 8), (5, 5, 10, 0, 0),
        )
        for per_layer in (0, 1):
            for w, h, n, dx, dy in cases:
                r = subprocess.run([str(exe), str(w), str(h), str(n), str(dx), str(dy), str(per_layer)],
                                   text=True, capture_output=True, timeout=20)
                need(r.returncode == 0, f"{w} x {h}, {n} layers ({dx},{dy}, per_layer={per_layer}): {r.stdout.strip()}")
    print("PASS: bc_encode_into uses explicit block pitch everywhere and splits non-4-aligned multi-layer copies only when bc_copy_per_layer is on")


if __name__ == "__main__":
    main()
