"""Exercise the menu transition without a game or Vulkan device."""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]

PROGRAM = r"""
#include "gpu/vk/menu_fade.h"
#include <cassert>

using mkwvk::menu_fade::Action;
using mkwvk::menu_fade::Transition;
constexpr int64_t ms = 1000000;

int main() {
    Transition fade;
    assert(fade.step(0, false, false).action == Action::Native);  // startup/loading
    assert(fade.step(10 * ms, false, true).action == Action::Capture);
    fade.captured_at = 10 * ms;  // replay only sets this after recording the copy
    assert(fade.step(80 * ms, false, true).action == Action::Native);
    assert(fade.step(90 * ms, true, false).action == Action::Native);
    assert(fade.step(2000 * ms, true, false).action == Action::Native);
    assert(fade.step(2100 * ms, false, false).action == Action::Hold);
    assert(fade.step(2300 * ms, false, false).action == Action::Hold);
    const auto first = fade.step(2350 * ms, false, true);
    assert(first.action == Action::Blend && first.opacity == 1000);
    const auto half = fade.step(2470 * ms, false, true);
    assert(half.action == Action::Blend && half.opacity == 500);
    assert(fade.step(2590 * ms, false, true).action == Action::Capture);

    fade.captured_at = 2600 * ms;
    assert(fade.step(2700 * ms, true, true).action == Action::Native);  // wheel over the world
    assert(fade.step(2800 * ms, false, false).action == Action::Hold);
    assert(fade.step(4300 * ms, false, false).action == Action::Native);  // bounded wait
    assert(fade.step(4400 * ms, false, true).action == Action::Capture);

    fade.captured_at = 4400 * ms;
    assert(fade.step(6600 * ms, true, false).action == Action::Native);
    assert(fade.step(6700 * ms, false, false).action == Action::Native);  // stale capture
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="test-menu-fade-") as directory:
        source = Path(directory) / "menu_fade.cpp"
        executable = source.with_suffix(".exe" if os.name == "nt" else "")
        source.write_text(PROGRAM, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++17", "-I", str(ROOT), str(source), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
