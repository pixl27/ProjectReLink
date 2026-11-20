"""Build and run the game-free temporal-upscaler camera-math regression.

Run with: python3 -I -B nx2/test_dlss_math.py [--cxx clang++]
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    source = ROOT / "gpu" / "vk" / "tests" / "dlss_math_test.cpp"
    with tempfile.TemporaryDirectory(prefix="dlss-math-") as temporary:
        executable = Path(temporary) / "dlss_math_test"
        subprocess.run(
            [args.cxx, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)

    dlss = (ROOT / "gpu" / "vk" / "vk_dlss.cpp").read_text(encoding="utf-8")
    assert "dlss_math::camera_projection(" in dlss
    assert "if (!reset && !motion) reset = true" in dlss
    print("dlss_math: ok")


if __name__ == "__main__":
    main()
