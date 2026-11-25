"""Check the ASTC CPU partition selector against the original 2D formula."""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]

REFERENCE = r"""
u32 reference(u32 seed, u32 count, bool small, u32 x, u32 y) {
    if (count == 1) return 0;
    if (small) x *= 2, y *= 2;
    seed += (count - 1) * 1024;
    const u32 hash = hash52(seed);
    const u32 even = (seed & 2) ? 4 : 5;
    const u32 odd = count == 3 ? 6 : 5;
    const u32 xshift = (seed & 1) ? even : odd;
    const u32 yshift = (seed & 1) ? odd : even;
    u32 s[8];
    for (u32 n = 0; n < 8; ++n) {
        const u32 digit = (hash >> (4 * n)) & 15;
        s[n] = (digit * digit) >> ((n & 1) ? yshift : xshift);
    }
    const u32 a = (s[0] * x + s[1] * y + (hash >> 14)) & 63;
    const u32 b = (s[2] * x + s[3] * y + (hash >> 10)) & 63;
    const u32 c = count >= 3 ? (s[4] * x + s[5] * y + (hash >> 6)) & 63 : 0;
    const u32 d = count == 4 ? (s[6] * x + s[7] * y + (hash >> 2)) & 63 : 0;
    if (a >= b && a >= c && a >= d) return 0;
    if (b >= c && b >= d) return 1;
    if (c >= d) return 2;
    return 3;
}

int main() {
    for (u32 count = 1; count <= 4; ++count)
        for (u32 seed = 0; seed < 1024; ++seed)
            for (bool small : {false, true}) {
                const Partition2D partitions(seed, count, small);
                for (u32 y = 0; y < 12; ++y)
                    for (u32 x = 0; x < 12; ++x)
                        assert(partitions.at(x, y) == reference(seed, count, small, x, y));
            }
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    path = ROOT / "gpu/third_party/suyu/video_core/textures/astc.cpp"
    decoder = path.read_text(encoding="utf-8")
    selector = decoder[decoder.index("static inline u32 hash52"):
                       decoder.index("static constexpr bool IsHDRColorEndpointMode")]
    includes = "#include <array>\n#include <cassert>\n#include <cstdint>\nusing u32 = uint32_t;\n"
    with tempfile.TemporaryDirectory(prefix="test-astc-cpu-partition-") as directory:
        source = Path(directory) / "partition.cpp"
        executable = source.with_suffix(".exe" if os.name == "nt" else "")
        source.write_text(includes + selector + REFERENCE, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++17", "-O2", str(source), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
