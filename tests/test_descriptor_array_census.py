"""The indexed descriptor arrays the programs declare are counted by stage, kind and size, and the ones near a limit are named.

host_info() gives the translator no device limits, so a dynamically indexed texture array is sized with its defaults, up to
1024 elements, on every GPU; Apple's M1 and M2 reach 996 samplers a stage through an argument buffer. Whether the game declares
such arrays (or image, image-buffer and texture-buffer arrays at all) was never counted. Checked here:
- gpu/vk/descriptor_array_census.h's cap is the translator's (HostTranslateInfo::DEFAULT_DESCRIPTOR_LIMIT, texture_pass.cpp's
  DESCRIPTOR_MAX_COUNT), the four lists it reads are Shader::Info's, each descriptor with a count, and vk_programs.cpp holds its
  stage bits to Vulkan's (a static_assert, which the object compile checks);
- its census, run against stand-in program stages: a stage's arrays (count above 1) are counted by stage, kind and size, once a
  program stage whatever its variants; one of 1024 elements is named as at the cap, one of 996 as near it, the texture arrays
  of a stage with more than 996 samplers as past the M1's and M2's; with a device's per-stage limits (MoltenVK's here) the
  arrays of a stage past them are named; single descriptors, stages with no arrays and stages it does not know are not; past
  30 the arrays are counted and not named; the summary gives the counts and a line for each stage with arrays;
- it is compiled with what the Windows headers define as macros (windef.h's near, far, IN and OUT, rpcndr.h's small, basetyps.h's
  interface) defined before it, as vk_programs.cpp includes it after windows.h: a member named near stopped the Windows build;
- every program the program cache sees is counted, translated (program_cache_save, before it returns for a cache that is off)
  or read (program_cache_read, program_cache_read_light), the summary is logged after the preparation and when the window
  closes, and none of it is in the translator's stamp (nx2/translator_files.py): no player's programs are translated again
  for it.

No Vulkan, GPU or game. python -I -B tests/test_descriptor_array_census.py [--cxx clang++] [--source <tree>]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


FIXTURE = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <utility>
#include <vector>
// (what windows.h defines, which vk_programs.cpp has included when it reaches the census header)
#define near
#define far
#define IN
#define OUT
#define small char
#define interface struct
#include "descriptor_array_census.h"
#include <cstdio>
#include <vector>
using namespace mkwvk;
struct D { unsigned count; };
struct Info {   // (Shader::Info's four lists; each descriptor has a count)
    std::vector<D> texture_descriptors, texture_buffer_descriptors, image_descriptors, image_buffer_descriptors;
};
constexpr uint32_t VERTEX = 0x1, GEOMETRY = 0x8, FRAGMENT = 0x10, COMPUTE = 0x20, TASK = 0x40, MESH = 0x80;
Info make(std::vector<unsigned> t, std::vector<unsigned> tb = {}, std::vector<unsigned> i = {}, std::vector<unsigned> ib = {}) {
    Info info;
    for (unsigned c : t) info.texture_descriptors.push_back({c});
    for (unsigned c : tb) info.texture_buffer_descriptors.push_back({c});
    for (unsigned c : i) info.image_descriptors.push_back({c});
    for (unsigned c : ib) info.image_buffer_descriptors.push_back({c});
    return info;
}
void run(ArrayCensus& c, bool many) {
    std::vector<std::string> lines;
    c.note(0xa, 0, FRAGMENT, make({1, 1024, 64}, {2}, {3}, {2}), lines);   // (a dynamic array at the cap, a handle table, F1's three)
    c.note(0xa, 0, FRAGMENT, make({1, 1024, 64, 1024}), lines);            // (another variant of the same stage: not counted again)
    c.note(0xa, 1, VERTEX, make({1, 1}), lines);                           // (no array)
    c.note(0xb, 0, COMPUTE, make({996}, {}, {1}), lines);                  // (near the cap; 996 samplers is the M1's and M2's own)
    c.note(0xc, 0, COMPUTE, make({8}), lines);                             // (a mesh fallback's task stage, as compute)
    c.note(0xc, 1, COMPUTE, make({}, {}, {}, {4}), lines);                 // (its mesh stage)
    c.note(0xc, 2, VERTEX, make({}), lines);                               // (its generated vertex stage: no descriptors)
    c.note(0xc, 3, FRAGMENT, make({17}), lines);                           // (17 samplers: past MoltenVK's 16)
    c.note(0xd, 0, TASK, make({2}), lines);
    c.note(0xd, 1, MESH, make({300, 700}), lines);                         // (1000 samplers: past the M1's and M2's 996)
    c.note(0xe, 0, GEOMETRY, make({1024}), lines);                         // (a stage the census does not know: left out)
    if (many)
        for (uint64_t k = 0; k < 40; k++) c.note(0x100 + k, 0, FRAGMENT, make({1000}), lines);
    for (const std::string& line : lines) std::printf("LOG %s\n", line.c_str());
    for (const std::string& line : c.summary("close")) std::printf("SUMMARY %s\n", line.c_str());
}
int main() {
    std::printf("CAP %u\n", ArrayCensus::kCap);
    for (int s = 0; s < ArrayCensus::kStages; s++) std::printf("STAGE %s %#x\n", ArrayCensus::kStageNames[s], ArrayCensus::kStageBits[s]);
    std::printf("PART none\n");
    {
        ArrayCensus none;
        std::vector<std::string> lines;
        none.note(0x1, 0, VERTEX, make({1}, {1}), lines);
        none.note(0x1, 1, FRAGMENT, make({1, 1}), lines);
        for (const std::string& line : lines) std::printf("LOG %s\n", line.c_str());
        for (const std::string& line : none.summary("the preparation")) std::printf("SUMMARY %s\n", line.c_str());
    }
    std::printf("PART any GPU\n");
    {
        ArrayCensus any;
        run(any, false);
    }
    std::printf("PART MoltenVK\n");
    {
        ArrayCensus mvk;
        mvk.limits = {16, 256, 256};   // (maxPerStageDescriptorSamplers, ...SampledImages, ...StorageImages: MoltenVK 1.4.2 on this Mac)
        run(mvk, true);
    }
    return 0;
}
"""

HEAD = "LOG vk: indexed descriptor array, program %016x's %s stage: "
CAP = "at the translator's cap of 1024 elements, its default limit (the game's may be longer)"
NEAR = "near the translator's cap of 1024 elements"
APPLE = "the stage's %d samplers pass the 996 that an Apple M1 or M2 stage reaches through an argument buffer"
SAMPLERS = "the stage's %d samplers pass this GPU's maxPerStageDescriptorSamplers (16)"
SAMPLED = "the stage's %d sampled images pass this GPU's maxPerStageDescriptorSampledImages (256)"
STAGES = [
    "SUMMARY vk: indexed descriptor arrays (close), fragment stages: %s1 texture buffer (2-16: 1), 1 storage image (2-16: 1), 1 "
    "image buffer (2-16: 1); the longest 1024",
    "SUMMARY vk: indexed descriptor arrays (close), compute stages: 2 texture (2-16: 1, 996-1023: 1), 1 image buffer (2-16: 1); "
    "the longest 996",
    "SUMMARY vk: indexed descriptor arrays (close), task stages: 1 texture (2-16: 1); the longest 2",
    "SUMMARY vk: indexed descriptor arrays (close), mesh stages: 2 texture (257-995: 2); the longest 700",
]
WANT = {
    "none": ["SUMMARY vk: indexed descriptor arrays (the preparation): none in 2 program stages"],
    "any GPU": [
        HEAD % (0xa, "fragment") + "texture[1024]: " + CAP + "; " + APPLE % 1089,
        HEAD % (0xa, "fragment") + "texture[64]: " + APPLE % 1089,
        HEAD % (0xb, "compute") + "texture[996]: " + NEAR,
        HEAD % (0xd, "mesh") + "texture[300]: " + APPLE % 1000,
        HEAD % (0xd, "mesh") + "texture[700]: " + APPLE % 1000,
        "SUMMARY vk: indexed descriptor arrays (close): 12 in 7 of 9 program stages, 5 of them near a limit (named above)",
        STAGES[0] % "3 texture (17-64: 2, 1024 and more: 1), ", STAGES[1], STAGES[2], STAGES[3],
    ],
    "MoltenVK": [
        HEAD % (0xa, "fragment") + "texture[1024]: " + "; ".join([CAP, APPLE % 1089, SAMPLERS % 1089, SAMPLED % 1091]),
        HEAD % (0xa, "fragment") + "texture[64]: " + "; ".join([APPLE % 1089, SAMPLERS % 1089, SAMPLED % 1091]),
        HEAD % (0xa, "fragment") + "texture buffer[2]: " + SAMPLED % 1091,
        HEAD % (0xb, "compute") + "texture[996]: " + "; ".join([NEAR, SAMPLERS % 996, SAMPLED % 996]),
        HEAD % (0xc, "fragment") + "texture[17]: " + SAMPLERS % 17,
        HEAD % (0xd, "mesh") + "texture[300]: " + "; ".join([APPLE % 1000, SAMPLERS % 1000, SAMPLED % 1000]),
        HEAD % (0xd, "mesh") + "texture[700]: " + "; ".join([APPLE % 1000, SAMPLERS % 1000, SAMPLED % 1000]),
    ] + [HEAD % (0x100 + k, "fragment") + "texture[1000]: " + "; ".join([NEAR, APPLE % 1000, SAMPLERS % 1000, SAMPLED % 1000])
         for k in range(23)] + [
        "SUMMARY vk: indexed descriptor arrays (close): 52 in 47 of 49 program stages, 47 of them near a limit (the first 30 named "
        "above)",
        STAGES[0] % "43 texture (17-64: 2, 996-1023: 40, 1024 and more: 1), ", STAGES[1], STAGES[2], STAGES[3],
    ],
}
VK_STAGES = [("vertex", 0x1, "VERTEX_BIT"), ("fragment", 0x10, "FRAGMENT_BIT"), ("compute", 0x20, "COMPUTE_BIT"),
             ("task", 0x40, "TASK_BIT_EXT"), ("mesh", 0x80, "MESH_BIT_EXT")]


def constant(text, name, where, fails):
    m = re.search(r"\b%s = (\d+);" % name, text)
    if not m:
        fails.append("%s: no %s" % (where, name))
        return None
    return int(m.group(1))


def source_checks(tree, vk):
    fails = []
    eden = vk / "eden" / "shader_recompiler"
    info = (eden / "shader_info.h").read_text(encoding="utf-8")
    for kind, field in (("Texture", "texture_descriptors"), ("TextureBuffer", "texture_buffer_descriptors"),
                        ("Image", "image_descriptors"), ("ImageBuffer", "image_buffer_descriptors")):
        if not re.search(r"\b%sDescriptors %s;" % (kind, field), info) or "u32 count;" not in block(info, "struct %sDescriptor {" % kind):
            fails.append("shader_info.h: Shader::Info's %s is not a list of %sDescriptor with a count" % (field, kind))
    caps = [constant((eden / "host_translate_info.h").read_text(encoding="utf-8"), "DEFAULT_DESCRIPTOR_LIMIT", "host_translate_info.h", fails),
            constant((eden / "ir_opt" / "texture_pass.cpp").read_text(encoding="utf-8"), "DESCRIPTOR_MAX_COUNT", "texture_pass.cpp", fails)]
    programs = (vk / "vk_programs.cpp").read_text(encoding="utf-8")
    for i, (_, _, bit) in enumerate(VK_STAGES):
        if "ArrayCensus::kStageBits[%d] == VK_SHADER_STAGE_%s" % (i, bit) not in programs:
            fails.append("vk_programs.cpp: no static_assert of the census's stage %d against VK_SHADER_STAGE_%s" % (i, bit))
    note = block(programs, "void descriptor_arrays_note(const Program& p)")
    if "array_census.note(p.key, (uint32_t)k, p.stages[k].vk_stage, p.stages[k].info, lines)" not in note or \
            "{l.maxPerStageDescriptorSamplers, l.maxPerStageDescriptorSampledImages, l.maxPerStageDescriptorStorageImages}" not in note:
        fails.append("vk_programs.cpp descriptor_arrays_note: not each stage of the program, with the device's per-stage limits")
    if "array_census.summary(when)" not in block(programs, "void descriptor_arrays_log(const char* when)"):
        fails.append("vk_programs.cpp descriptor_arrays_log: not the census's summary")
    cache = (vk / "vk_progcache.cpp").read_text(encoding="utf-8")
    save = block(cache, "void program_cache_save(")
    if not 0 <= save.find("descriptor_arrays_note(p);") < save.find("disk_init();"):
        fails.append("vk_progcache.cpp program_cache_save: a translated program is not counted before the cache may be off")
    for name in ("void program_cache_read(", "std::unique_ptr<Program> program_cache_read_light("):
        body = block(cache, name)
        if not 0 <= body.find("descriptor_arrays_note(*p);") < body.find("p->ready.store(true"):
            fails.append("vk_progcache.cpp %s: a program read is not counted before it is published" % name[:-1].split()[-1])
    prepare = block((vk / "vk_draw.cpp").read_text(encoding="utf-8"), "bool prepare_run(")
    prepared, logged = prepare.find('logf("vk: prepared %zu shader programs'), prepare.find("descriptor_arrays_log(")
    if prepared < 0 or logged < prepared or "the preparation" not in prepare[logged:logged + 120]:
        fails.append("vk_draw.cpp prepare_run: the census is not logged after the preparation's line")
    if 'descriptor_arrays_log("close");' not in block((vk / "vk_main.cpp").read_text(encoding="utf-8"), "void mkwvk_close_asked(void)"):
        fails.append("vk_main.cpp mkwvk_close_asked: the census is not logged when the window closes")
    # (none of it in the translator's stamp: nx2/translator_files.py, which nx2/build.py stamps with since #370, hashes eden/,
    # sirit/, compat/ and three files of gpu/vk)
    listing = tree / "nx2" / "translator_files.py"
    text = listing.read_text(encoding="utf-8") if listing.exists() else ""
    top_list, file_list = re.search(r"^DIRS = \(([^)]*)\)", text, re.M), re.search(r"^FILES = \(([^)]*)\)", text, re.M)
    if not (top_list and file_list):
        fails.append("nx2/translator_files.py: the translator's stamp list was not found")
    else:
        tops = re.findall(r'"([^"]+)"', top_list.group(1))
        stamped = set(re.findall(r'"([^"]+)"', file_list.group(1)))
        mine = {"descriptor_array_census.h", "vk_programs.cpp", "vk_programs.h", "vk_progcache.cpp", "vk_draw.cpp", "vk_main.cpp"}
        if stamped & mine:
            fails.append("the translator's stamp hashes %s" % sorted(stamped & mine))
        for name in sorted(stamped):
            if "descriptor_array" in (vk / name).read_text(encoding="utf-8", errors="replace"):
                fails.append("gpu/vk/%s (in the translator's stamp) has a part of the census" % name)
        for top in tops:
            for d, _, files in os.walk(vk / top):
                for f in files:
                    p = Path(d) / f
                    if p.suffix in (".h", ".cpp", ".inc") and "descriptor_array_census" in p.read_text(encoding="utf-8", errors="replace"):
                        fails.append("%s (in the translator's stamp) reads the census" % p.relative_to(vk))
    return fails, caps


def fixture_checks(cxx, vk, caps):
    fails = []
    with tempfile.TemporaryDirectory(prefix="descriptor-array-census-") as tmp:
        src, exe = Path(tmp) / "fixture.cpp", Path(tmp) / ("fixture" + (".exe" if os.name == "nt" else ""))
        src.write_text(FIXTURE, encoding="utf-8")
        r = subprocess.run([cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-I", str(vk), str(src), "-o", str(exe)],
                           capture_output=True, text=True, timeout=180)
        if r.returncode:
            print(r.stdout + r.stderr)
            return ["the fixture did not compile"]
        got = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        if got.returncode:
            return ["the fixture failed: " + got.stderr.strip()]
    out = got.stdout.splitlines()
    cap = [int(line.split()[1]) for line in out if line.startswith("CAP ")]
    if None not in caps and (cap != [caps[0]] or cap != [caps[1]]):
        fails.append("the census's cap %s is not the translator's %s" % (cap, caps))
    stages = [(line.split()[1], int(line.split()[2], 16)) for line in out if line.startswith("STAGE ")]
    if stages != [(name, bit) for name, bit, _ in VK_STAGES]:
        fails.append("the census's stages %s are not Vulkan's %s" % (stages, [(n, b) for n, b, _ in VK_STAGES]))
    parts, at = {}, None
    for line in out:
        if line.startswith("PART "):
            at = line[5:]
            parts[at] = []
        elif at is not None and line.startswith(("LOG ", "SUMMARY ")):
            parts[at].append(line)
    for name, want in WANT.items():
        got_lines = parts.get(name, [])
        if got_lines != want:
            diff = [i for i in range(max(len(got_lines), len(want)))
                    if i >= len(got_lines) or i >= len(want) or got_lines[i] != want[i]][:1]
            i = diff[0] if diff else 0
            fails.append("%s: %d lines, %d wanted; line %d:\n  got  %s\n  want %s" % (
                name, len(got_lines), len(want), i, got_lines[i] if i < len(got_lines) else "(none)", want[i] if i < len(want) else "(none)"))
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cxx", default="clang++")
    ap.add_argument("--source", type=Path, default=ROOT, help="the tree to check (default: this one)")
    a = ap.parse_args()
    vk = a.source / "gpu" / "vk"
    fails = []
    if not (vk / "descriptor_array_census.h").is_file():
        fails.append("gpu/vk/descriptor_array_census.h: no census of the indexed descriptor arrays the programs declare")
    else:
        source_fails, caps = source_checks(a.source, vk)
        fails += source_fails + fixture_checks(a.cxx, vk, caps)
    for f in fails:
        print("FAIL: " + f)
    if fails:
        return 1
    print("PASS: the programs' indexed descriptor arrays are counted by stage, kind and size, translated or read, and the ones near "
          "a limit are named")
    return 0


if __name__ == "__main__":
    sys.exit(main())
