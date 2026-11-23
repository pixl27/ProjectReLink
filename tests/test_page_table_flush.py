"""Exercise the replay's page-table transfers and scratch allocation with no GPU or game.

Run python -I tests/test_page_table_flush.py [--cxx clang++].
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from srccut import block

RUN_ALL_ARGS = ["--cxx", "{cxx}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    source = (ROOT / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    fixture = Path(__file__).with_suffix(".cpp").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        cpp = Path(tmp) / "flush.cpp"
        exe = Path(tmp) / "flush.exe"
        cpp.write_text(fixture.replace("/* FLUSH */", block(source, "bool memory_table_flush()")), encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-O2", str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
