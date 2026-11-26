"""The ASTC LDR decoder (gpu/vk/texdecode/astc.cpp) equals ARM's astcenc on the committed vectors, block by block, in the linear and sRGB modes, error blocks included. The vectors are random blocks, not game data; scripts/gen_texdecode_astc_vectors.py regenerates them with an astcenc of your own, the test needs none.
python -I -B tests/test_texdecode_astc.py [--cxx c++]
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    cxx = shutil.which(args.cxx)
    if not cxx:
        raise SystemExit("C++ compiler not found: " + args.cxx)
    with tempfile.TemporaryDirectory(prefix="longshot-texdecode-") as folder:
        exe = Path(folder) / "t"
        r = subprocess.run([cxx, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "gpu" / "vk"),
                            str(ROOT / "tests" / "texdecode_astc_test.cpp"), str(ROOT / "gpu" / "vk" / "texdecode" / "astc.cpp"),
                            "-o", str(exe)], capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise SystemExit("FAIL: texdecode_astc_test.cpp does not compile\n" + r.stdout + r.stderr)
        r = subprocess.run([str(exe), str(ROOT / "tests" / "texdecode_astc_vectors.vec")], capture_output=True, text=True, timeout=300)
        print(r.stdout, end="")
        if r.returncode:
            raise SystemExit("FAIL: texdecode_astc_test failed\n" + r.stderr)


if __name__ == "__main__":
    main()
