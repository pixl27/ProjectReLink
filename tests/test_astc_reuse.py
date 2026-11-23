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
        cache = block(src, "struct AstcCpuCacheEntry {") + ";\n"
        cache += block(src, "void decode_astc_vega_cached(")
        probe = Path(tmp) / "cpu-cache.cpp"
        probe.write_text(CPU_CACHE_PROBE % cache, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-I", str(ROOT / "gpu/vk"), str(probe), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
