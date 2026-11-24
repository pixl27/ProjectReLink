"""Exercise the small depth-clear upload against submitted bytes."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srccut import block

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r"""
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
enum { VK_IMAGE_ASPECT_DEPTH_BIT = 2, VK_IMAGE_TYPE_2D = 1, VK_SAMPLE_COUNT_1_BIT = 1,
       VK_IMAGE_LAYOUT_GENERAL = 1 };
constexpr uint32_t VK_REMAINING_MIP_LEVELS = ~0u, VK_REMAINING_ARRAY_LAYERS = ~0u;
using VkCommandBuffer = int;
struct VkImageSubresourceRange { uint32_t aspect, mip, levels, layer, layers; };
struct VkClearDepthStencilValue { float depth; uint32_t stencil; };
struct Texture {
    uint32_t aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    int image = 1, type = VK_IMAGE_TYPE_2D, samples = VK_SAMPLE_COUNT_1_BIT;
    bool sparse = false, gpu_written = false;
    uint64_t gpu = 0;
};
struct Ranges {};
std::vector<uint8_t> submitted((1u << 20) + 3);
unsigned reads, clears;
bool env_on(const char*) { return true; }
const uint8_t* guest_bytes(uint64_t at, uint64_t n, std::vector<uint8_t>&) {
    assert(at + n <= submitted.size());
    ++reads;
    return submitted.data() + at;
}
void texture_resident_flush(Texture*) {}
int upload_cmd() { return 1; }
void checkpoint_mark(int, uint64_t, char) {}
void logf(const char*, ...) {}
void vkCmdClearDepthStencilImage(int, int, int, const VkClearDepthStencilValue* zero,
                                int count, const VkImageSubresourceRange* range) {
    assert(count == 1 && zero->depth == 0.0f && zero->stencil == 0);
    assert(range->aspect == VK_IMAGE_ASPECT_DEPTH_BIT && range->layers == VK_REMAINING_ARRAY_LAYERS);
    ++clears;
}
@FUNCTION@
int main() {
    Texture tex;
    const Ranges partial{};
    assert(!upload_small_zero_depth(tex, submitted.size(), &partial));
    tex.gpu_written = true;
    assert(!upload_small_zero_depth(tex, submitted.size(), nullptr));
    tex.gpu_written = false;
    assert(reads == 0 && clears == 0);
    submitted.back() = 1;
    assert(!upload_small_zero_depth(tex, submitted.size(), nullptr));
    assert(reads == 2 && clears == 0);
    submitted.back() = 0;
    assert(upload_small_zero_depth(tex, submitted.size(), nullptr));
    assert(reads == 4 && clears == 1);
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    source = (ROOT / "gpu/vk/vk_texture.cpp").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="small-zero-depth-") as temp:
        cpp = Path(temp) / "fixture.cpp"
        exe = Path(temp) / "fixture.exe"
        cpp.write_text(FIXTURE.replace("@FUNCTION@", block(source, "bool upload_small_zero_depth(")), encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
