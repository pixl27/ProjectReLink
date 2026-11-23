"""Check the 32-lane subgroup pin against maxComputeWorkgroupSubgroups without a GPU.

pin_subgroup (vk_draw.cpp) chains VkPipelineShaderStageRequiredSubgroupSizeCreateInfo with requiredSubgroupSize 32 on the
stages the GPU can pin. For compute, task and mesh stages VUID-VkPipelineShaderStageCreateInfo-pNext-02756 asks that the
local workgroup size (x * y * z) be at most requiredSubgroupSize * maxComputeWorkgroupSubgroups, a limit the spec sets no
minimum for. The fixture compiles pin_subgroup and pin_compute's call of it (every game compute pipeline's), cut from the source, with a
fake device, and checks that a workgroup 32-lane subgroups cannot cover is not pinned, while one they cover still is.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu/vk"

HEAD = r'''
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
using VkShaderStageFlags = uint32_t;
using VkShaderStageFlagBits = uint32_t;
constexpr uint32_t VK_SHADER_STAGE_VERTEX_BIT = 1, VK_SHADER_STAGE_FRAGMENT_BIT = 16, VK_SHADER_STAGE_COMPUTE_BIT = 32,
                   VK_SHADER_STAGE_TASK_BIT_EXT = 64, VK_SHADER_STAGE_MESH_BIT_EXT = 128;
constexpr uint32_t VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT = 2;
constexpr int VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO = 1000225001;
struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo {
    int sType;
    void* pNext = nullptr;
    uint32_t requiredSubgroupSize = 0;
};
struct VkPipelineShaderStageCreateInfo {
    int sType = 0;
    const void* pNext = nullptr;
    uint32_t flags = 0;
    VkShaderStageFlagBits stage = 0;
};
struct VkComputePipelineCreateInfo {
    VkPipelineShaderStageCreateInfo stage;
};
struct {
    VkShaderStageFlags sg32_stages = 0;
    uint32_t max_workgroup_subgroups = 0;
    struct { bool computeFullSubgroups = true; } vk13;
} dev;
struct Program {
    std::array<uint32_t, 3> workgroup{1, 1, 1};
};
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "CHECK failed (%s): %s\n", where, #c); std::exit(1); } } while (0)
'''

CASES = r'''
/// A compute pipeline of this workgroup through each of the source's calls: pinned (with full subgroups or not) or left alone
void compute(const char* where, uint32_t x, uint32_t y, uint32_t z, bool pinned, bool full) {
    for (Call call : calls) {
        Program q;
        q.workgroup = {x, y, z};
        VkComputePipelineCreateInfo ci{};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        call(q, ci);
        CHECK((ci.stage.pNext != nullptr) == pinned);
        if (pinned) CHECK(static_cast<const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*>(ci.stage.pNext)->requiredSubgroupSize == 32);
        CHECK(((ci.stage.flags & VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) != 0) == full);
    }
}

/// A stage made without its workgroup (the graphics pipelines' stages and the mesh fallback's compute stages: pin_subgroup(si))
void stage(const char* where, VkShaderStageFlagBits bit, bool pinned) {
    VkPipelineShaderStageCreateInfo si{};
    si.stage = bit;
    pin_subgroup(si);
    CHECK((si.pNext != nullptr) == pinned);
    CHECK(si.flags == 0);
}

int main() {
    dev.sg32_stages = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    // 16 subgroups of 32 lanes: 512 invocations at most
    dev.max_workgroup_subgroups = 16;
    compute("1024 x 1 x 1, 16 subgroups", 1024, 1, 1, false, false);
    compute("32 x 8 x 4, 16 subgroups", 32, 8, 4, false, false);
    compute("512 x 1 x 1, 16 subgroups", 512, 1, 1, true, true);
    compute("64 x 4 x 2, 16 subgroups", 64, 4, 2, true, true);
    compute("48 x 1 x 1, 16 subgroups", 48, 1, 1, true, false);
    stage("mesh, 16 subgroups", VK_SHADER_STAGE_MESH_BIT_EXT, true);
    stage("task, 16 subgroups", VK_SHADER_STAGE_TASK_BIT_EXT, true);
    stage("fallback compute, 16 subgroups", VK_SHADER_STAGE_COMPUTE_BIT, true);
    stage("fragment not pinnable, 16 subgroups", VK_SHADER_STAGE_FRAGMENT_BIT, false);
    // one subgroup: 32 invocations; a task or mesh workgroup has up to 63 threads (a 6-bit field)
    dev.max_workgroup_subgroups = 1;
    stage("mesh, 1 subgroup", VK_SHADER_STAGE_MESH_BIT_EXT, false);
    stage("task, 1 subgroup", VK_SHADER_STAGE_TASK_BIT_EXT, false);
    stage("fallback compute, 1 subgroup", VK_SHADER_STAGE_COMPUTE_BIT, false);
    compute("32 x 1 x 1, 1 subgroup", 32, 1, 1, true, true);
    compute("16 x 2 x 2, 1 subgroup", 16, 2, 2, false, false);
    dev.sg32_stages |= VK_SHADER_STAGE_FRAGMENT_BIT;
    stage("fragment, 1 subgroup (no workgroup)", VK_SHADER_STAGE_FRAGMENT_BIT, true);
    // 32 subgroups: the guest's 1024 invocations
    dev.max_workgroup_subgroups = 32;
    compute("1024 x 1 x 1, 32 subgroups", 1024, 1, 1, true, true);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    draw = (VK / "vk_draw.cpp").read_text(encoding="utf-8")
    internal = (VK / "vk_internal.h").read_text(encoding="utf-8")
    helper = block(internal, "inline bool pins_32(") + "\n" if "inline bool pins_32(" in internal else ""
    pin = block(draw, "void pin_subgroup(")
    # (the compute pipelines' call: pin_compute's, which compute_pipeline_info makes every game compute pipeline through)
    lines = re.findall(r"^[ \t]*if \(compute_pin_needed\(p\)\) (pin_subgroup\(st, [^\n]*\);)[ \t]*$", draw, re.M)
    assert len(lines) == 1, "pin_compute's pin_subgroup call: %r" % lines
    assert draw.count("pin_subgroup(si);") == 2, "the graphics and mesh fallback stages' pin_subgroup calls"
    calls = "".join("void call_%d(Program& p, VkComputePipelineCreateInfo& ci) { auto& st = ci.stage; %s }\n"
                    % (i, line.strip()) for i, line in enumerate(lines))
    calls += "using Call = void (*)(Program&, VkComputePipelineCreateInfo&);\n"
    calls += "const Call calls[] = {%s};\n" % ", ".join("call_%d" % i for i in range(len(lines)))
    fixture = HEAD + helper + pin + "\n" + calls + CASES
    with tempfile.TemporaryDirectory(prefix="hookshot-subgroup-pin-") as tmp:
        cpp, exe = Path(tmp) / "fixture.cpp", Path(tmp) / "fixture.exe"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", str(cpp), "-o", str(exe)], check=True, timeout=120)
        subprocess.run([str(exe)], check=True, timeout=30)
    # (the translator is told the same: a compute program its pipeline does not pin is not translated for 32 lanes)
    shader = (VK / "vk_shader.cpp").read_text(encoding="utf-8")
    assert "EmitSPIRV(profile_pinned(), rt, t->program, bindings)" not in shader, \
        "the compute translation takes the pinned profile whatever its workgroup"
    main_cpp = (VK / "vk_main.cpp").read_text(encoding="utf-8")
    assert "sc.maxComputeWorkgroupSubgroups" in main_cpp, "maxComputeWorkgroupSubgroups is never read"
    print("32-lane pins follow maxComputeWorkgroupSubgroups for compute, task and mesh stages ok")


if __name__ == "__main__":
    main()
