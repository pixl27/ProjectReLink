"""Compile and run decoded ASTC's reuse rules (gpu/vk/astc_reuse.h: the blocks' hash, the shape, the list; no game/GPU), and
check that vk_texture.cpp still lists finalized textures, copies from them and retires them."""
import argparse
import subprocess
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srccut import block

RUN_ALL_ARGS = ["--cxx", "g++"]

ROOT = Path(__file__).resolve().parents[1]

CPU_CACHE_PROBE = r"""
#include "astc_reuse.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <mutex>
#include <span>
#include <vector>
static unsigned decodes = 0;
void decode(std::span<const uint8_t> blocks, uint32_t w, uint32_t, uint32_t, uint32_t, uint32_t,
            std::span<uint8_t> out) {
    ++decodes;
    std::fill(out.begin(), out.end(), blocks.front() + w);
}
using AstcCpuDecoder = decltype(&decode);
%s
int main() {
    std::vector<uint8_t> encoded(16384, 7), decoded(256 * 1024);
    auto run = [&](uint32_t width) {
        decode_astc_vega_cached(decode, encoded, width, 256, 1, 8, 8, decoded);
        assert(std::all_of(decoded.begin(), decoded.end(), [&](uint8_t v) { return v == uint8_t(encoded.front() + width); }));
    };
    run(256);
    run(256);
    run(256);
    assert(decodes == 2);
    run(257);
    assert(decodes == 3);
    encoded.front() = 8;
    run(257);
    assert(decodes == 4);

    // Two different textures that hash to the same bucket both stay warm.
    encoded.front() = 40;
    const uint64_t first = astc_reuse::xxh64(encoded.data(), encoded.size(), 0);
    uint8_t other = 41;
    for (;; ++other) {
        assert(other != 40);
        encoded.front() = other;
        const uint64_t hash = astc_reuse::xxh64(encoded.data(), encoded.size(), 0);
        if (hash %% 16 == first %% 16 && hash %% 512 != first %% 512) break;
    }
    for (uint8_t value : {uint8_t(40), other}) {
        encoded.front() = value;
        run(260);
        run(260);
    }
    const unsigned warmed = decodes;
    for (uint8_t value : {uint8_t(40), other, uint8_t(40), other}) {
        encoded.front() = value;
        run(260);
    }
    assert(decodes == warmed);
}
"""

HASH_MEMO_PROBE = r"""
#include "astc_reuse.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <mutex>
#include <vector>
struct Host { uint64_t guest_size() const { return 4096; } };
static Host fixture;
static Host* host = &fixture;
static uint32_t gen = 1;
static bool shadow = false, gpu_written = false;
static uint64_t reads = 0;
static std::atomic<int64_t> st_reuse_hash_bytes{0};
static std::array<uint8_t, 128> bytes{};
bool env_on(const char*) { return true; }
bool memory_has_shadow(uint64_t, uint64_t) { return shadow; }
bool memory_writeback_ever(uint64_t, uint64_t) { return gpu_written; }
bool memory_written_since(uint64_t, uint64_t, uint32_t prior) { return prior < gen; }
uint32_t memory_generation() { return gen; }
const uint8_t* guest_bytes(uint64_t, uint64_t, std::vector<uint8_t>&) {
    ++reads;
    return bytes.data();
}
%s
int main() {
    const auto first = reuse_blocks_hash(64, 64);
    assert(first == reuse_blocks_hash(64, 64) && reads == 1);
    ++gen;
    bytes[0] = 1;
    const auto changed = reuse_blocks_hash(64, 64);
    assert(changed != first && reads == 2);
    shadow = true;
    bytes[0] = 2;
    assert(reuse_blocks_hash(64, 64) != changed && reads == 3);
    shadow = false;
    gpu_written = true;
    assert(reuse_blocks_hash(64, 64) != changed && reads == 4);
}
"""


AHEAD_COPY_PROBE = r"""
#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
struct LevelGeo { uint32_t w = 256, h = 256; };
void logf(const char*, ...) {}
bool astc_ahead_on() { return true; }
std::atomic<bool> ahead_memory_pressure{false};
%s
int main() {
    auto& slot = astc_ahead_queue().jobs[0];
    auto result = std::make_shared<AstcAheadJob>();
    result->bw = result->bh = 8;
    result->blocks.assign(16384, 7);
    result->rgba.assign(256 * 1024, 31);
    slot = result;
    std::vector<uint8_t> block = result->blocks, out(result->rgba.size(), 0);
    auto copy = [&] { return astc_ahead_copy(block, 256, 256, 1, 8, 8, out); };
    assert(!copy() && slot && out.front() == 0);
    result->state = AstcAheadJob::ready;
    block.back() ^= 1;
    assert(!copy() && slot && out.front() == 0);
    block.back() ^= 1;
    ahead_memory_pressure.store(true);
    assert(!copy() && slot && out.front() == 0);
    ahead_memory_pressure.store(false);
    assert(copy() && !slot && out == result->rgba);
    assert(!copy());
}
"""


def check_source():
    src = (ROOT / "gpu/vk/vk_texture.cpp").read_text(encoding="utf-8")
    for needle in ('#include "astc_reuse.h"',
                   "if (!dirty && !tex.uploaded && reuse_upload(tex, total)) return up_route = UP_ASTC, true;",
                   "freed.list(x->blocks_hash, x);",
                   "if (tex->freed_listed || tex->reuse_retire) reuse_drop(*tex);",
                   "if (tex->freed_listed || tex->reuse_retire) return false;",
                   "vkCmdCopyImage(cmd, from->image, VK_IMAGE_LAYOUT_GENERAL, tex.image, VK_IMAGE_LAYOUT_GENERAL",
                   "from->last_value = std::max(from->last_value, rec_value);",
                   "went.swap(reuse_gone);"):
        if needle not in src:
            raise SystemExit("vk_texture.cpp no longer has: " + needle)
    forget = block(src, "void forget_gpu(Texture* tex) {")
    if "reuse_drop(*tex);" not in forget:
        raise SystemExit("forget_gpu no longer takes a destroyed texture out of the reuse list")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="astc-reuse-") as tmp:
        exe = Path(tmp) / "astc-reuse"
        subprocess.run([args.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/test_astc_reuse.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
        src = (ROOT / "gpu/vk/vk_texture.cpp").read_text(encoding="utf-8")
        probe = Path(tmp) / "hash-memo.cpp"
        memo = src[src.index("std::atomic<int64_t> st_hash_memo_hits"):src.index("uint64_t reuse_blocks_hash(")]
        memo += block(src, "uint64_t reuse_blocks_hash(")
        probe.write_text(HASH_MEMO_PROBE % memo, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-I", str(ROOT / "gpu/vk"), str(probe), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
        cache = block(src, "struct AstcCpuCacheEntry {") + ";\n"
        cache += block(src, "void decode_astc_vega_cached(")
        probe = Path(tmp) / "cpu-cache.cpp"
        probe.write_text(CPU_CACHE_PROBE % cache, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-I", str(ROOT / "gpu/vk"), str(probe), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)

        ahead = block(src, "struct AstcAheadJob {") + ";\n"
        ahead += block(src, "struct AstcAheadQueue {") + ";\n"
        ahead += block(src, "AstcAheadQueue& astc_ahead_queue()")
        ahead += block(src, "bool astc_ahead_copy(")
        probe = Path(tmp) / "ahead-copy.cpp"
        probe.write_text(AHEAD_COPY_PROBE % ahead, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                        str(probe), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
