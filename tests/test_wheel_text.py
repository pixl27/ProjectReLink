"""The item wheel's letters off Windows (gpu/vk/vk_wheel_text.h: stb_truetype from Dear ImGui, the bundled Libertinus Sans Bold),
compiled alone and run on the bundled font (tests/test_wheel_text.cpp): UTF-8 read letter by letter, every label measured and
drawn inside its mask, sizes that scale, and a font that does not read gives the 0 by 0 the wheel skips. Linux lost the GDI
stand-in the wheel's labels were written for (host/posix/gdi_posix.c, removed in 70391859), so vk_wheel.cpp did not compile there.

    python -I -B tests/test_wheel_text.py [--cxx g++]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "third_party/fonts/libertinus/LibertinusSans-Bold.ttf"   # (nx2/ui_fonts.py's hookshot_font_strong)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="wheel-text-") as tmp:
        exe = Path(tmp) / "wheel-text"
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-D_CRT_SECURE_NO_WARNINGS",
                        "-I" + str(ROOT / "third_party/imgui"), str(ROOT / "tests/test_wheel_text.cpp"), "-o", str(exe)],
                       check=True)
        subprocess.run([str(exe), str(FONT)], check=True)


if __name__ == "__main__":
    main()
