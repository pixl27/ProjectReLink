"""Compile and run the lanes a mesh fallback compute stage runs at (gpu/vk/fallback_lanes.h; no game/GPU)."""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "g++"]

ROOT = Path(__file__).resolve().parents[1]


def check_source():
    """The renderer still notes whether each stage is lane-free beside its phis, and asks the rule for a fallback stage's lanes"""
    needles = {
        "gpu/vk/vk_progcache.cpp": ['#include "fallback_lanes.h"',
                                    "p.stages[k].lane_free = fallback_lanes::spirv_lane_free(spirv[k].data(), spirv[k].size());",
                                    "st.lane_free = fallback_lanes::spirv_lane_free(spirv.data(), spirv.size());"],
        "gpu/vk/vk_shader.h": ["mutable bool lane_free = false;"],
        "gpu/vk/vk_main.cpp": ["dev.subgroup_min = sc.minSubgroupSize;"],
        "gpu/vk/vk_draw.cpp": ['#include "fallback_lanes.h"', "RX_VK_MESH_LANES", "fallback_lanes::parse_lanes(e)",
                               "fallback_lanes::wanted(mesa_intel, asked, dev.subgroup_min, dev.subgroup_max)",
                               "r.requiredSubgroupSize = lanes_wanted;",
                               "if (si.pNext && fallback_lanes::lanes_of(st.lane_free, lanes_wanted) != 32) si.pNext = &lanes_size;"],
    }
    for path, wanted in needles.items():
        src = (ROOT / path).read_text(encoding="utf-8")
        for needle in wanted:
            if needle not in src:
                raise SystemExit(path + " no longer has: " + needle)
    cache = (ROOT / "gpu/vk/vk_progcache.cpp").read_text(encoding="utf-8")
    # (noted wherever the phis are: a stage whose code was read and not scanned would stay at 32 lanes for no reason)
    if cache.count("lane_free = fallback_lanes::spirv_lane_free(") != cache.count("phis = fallback_size::spirv_phis("):
        raise SystemExit("vk_progcache.cpp: lane_free is not noted at every place the phis are")
    draw = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    fn = draw[draw.index("bool fallback_compute_pipelines(Program& p) {"):]
    fn = fn[:fn.index("cached_compute(&ci, &made[i])")]
    # (the stage's own lanes replace the 32 pin_subgroup asked for, and only where the device can be asked at all)
    if not fn.index("pin_subgroup(si);") < fn.index("si.pNext = &lanes_size;") < fn.index("ci.stage = si;"):
        raise SystemExit("vk_draw.cpp: the fallback stage's lanes must be chosen after pin_subgroup and before the pipeline is asked for")
    # (the default stays 32: a stage is narrowed only when its SPIR-V was read and found lane-free)
    header = (ROOT / "gpu/vk/fallback_lanes.h").read_text(encoding="utf-8")
    if "inline uint32_t lanes_of(bool lane_free, uint32_t wanted_lanes) { return lane_free ? wanted_lanes : 32; }" not in header:
        raise SystemExit("fallback_lanes.h: a stage that is not lane-free must run at 32 lanes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="fallback-lanes-") as tmp:
        exe = Path(tmp) / "fallback-lanes"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/test_fallback_lanes.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
