"""The Linux AV1 path can preserve planar 4:2:0 as NV12 without RGBA work.

Run with: python3 -I -B nx2/test_video_nv12.py [--cc clang]
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()

    source = (ROOT / "host" / "nn_video.c").read_text(encoding="utf-8")
    start = source.index("static void queue_frame(")
    end = source.index("static void mf_picture(", start)
    queue_frame = source[start:end]
    assert "f->format == AV_PIX_FMT_YUV420P || f->format == AV_PIX_FMT_YUVA420P" in queue_frame
    assert "const int lazy_rgba = nv12 || f->format == AV_PIX_FMT_YUV420P" in queue_frame
    assert "p.rgba = lazy_rgba ? NULL" in queue_frame
    assert "video_yuv420p_to_nv12" in queue_frame

    # Decoder pools currently hold each acquired picture at index zero. The
    # existing lazy conversion deliberately follows that same ownership rule.
    rgba_reader = source[source.index("ReadDecoderOutputImageRgba") : source.index("ReadDecoderOutputImageNv12")]
    assert "if (i == 0 && o->rgba_stale) pool_rgba(o)" in rgba_reader

    program = r'''\
#include <assert.h>
#include <stdint.h>
#include "host/video_nv12.h"

int main(void) {
    const uint8_t y[18] = {
        1, 2, 3, 4, 99, 99,
        5, 6, 7, 8, 99, 99,
        9, 10, 11, 12, 99, 99
    };
    const uint8_t u[4] = {21, 22, 99, 99};
    const uint8_t v[6] = {31, 32, 99, 99, 99, 99};
    uint8_t out[18] = {0};
    const uint8_t expected[18] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        21, 31, 22, 32,
        0, 0
    };
    video_yuv420p_to_nv12(out, 4, 3, y, 6, u, 4, v, 6);
    for (unsigned i = 0; i < sizeof out; i++) assert(out[i] == expected[i]);
    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="video-nv12-") as temporary:
        directory = Path(temporary)
        c_file = directory / "test.c"
        executable = directory / "test"
        c_file.write_text(program, encoding="utf-8")
        subprocess.run(
            [args.cc, "-std=c11", "-Wall", "-Wextra", "-Werror", f"-I{ROOT}", str(c_file), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)

    print("video_nv12: planar packing and lazy-RGBA contract ok")


if __name__ == "__main__":
    main()
