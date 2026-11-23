"""The mesh fallback's warp in workgroup memory on Intel's Mesa driver (mma_layout.h mesh_fallback_shared_warp).

NVN's mesh programs are written for 32-lane warps, so a fallback stage with a ballot, a vote or a shuffle was held to 32 lanes, and
on an Iris Xe Hyrule Castle's colour mesh stages spent most of their time spilling there. The translator can instead give such a
stage a warp made of its 32-thread workgroup, with the ballots, votes and shuffles going through words of workgroup memory
(emit_spirv_warp.cpp): the stage then has no subgroup instruction and the renderer runs it at fewer lanes (fallback_lanes.h,
tests/test_fallback_lanes.py). This checks the switch (Intel's Mesa driver, RX_VK_MESH_SHARED_WARP over it) and the program cache's
stamp; which stages the planner takes; that the words are a Workgroup array of a type of their own (no ArrayStride, which
spirv-val refuses there); that every warp operation has its shared form and the stage declares none of the subgroup's
capabilities or inputs; that a region is opened before its if's merge instruction and closed at its end; that the words are
zeroed before the prologue's barrier; and that the host makes, clears and reads the counters past the records.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]
SPIRV = ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv"
ENV_H = ROOT / "gpu/vk/compat/hookshot_env.h"   # (env_value_off)
DRIVER_IDS = {"VK_DRIVER_ID_MESA_RADV": 3, "VK_DRIVER_ID_NVIDIA_PROPRIETARY": 4,   # (as the Vulkan headers number them)
              "VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS": 5, "VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA": 6}


def block(text, marker, start=0):
    start = text.index(marker, start)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def need(text, what, why):
    if what not in text:
        raise SystemExit(f"{why} ({what} not found)")


def switch(cxx):
    """mesh_pages_wanted, which sets this switch too, cut from vk_coopmat.cpp and run for the GPUs and settings that matter"""
    coopmat = (ROOT / "gpu/vk/vk_coopmat.cpp").read_text(encoding="utf-8")
    fn = block(coopmat, "bool mesh_pages_wanted(")
    init = block(coopmat, "void coopmat_init(")
    call = re.search(r"mesh_fallback_shared_warp = mesh_pages_wanted\(([^;]+)\);", init)
    if not call:
        raise SystemExit("coopmat_init does not set mesh_fallback_shared_warp from mesh_pages_wanted")
    args = " ".join(call.group(1).split())
    for needed in ("dev.driver_id", '"RX_VK_MESH_SHARED_WARP"'):
        if needed not in args:
            raise SystemExit(f"mesh_pages_wanted is not given {needed} for mesh_fallback_shared_warp")
    check = re.search(r"mesh_fallback_shared_warp_check =\s*([^;]+);", init)
    if not check or "mesh_fallback_shared_warp &&" not in " ".join(check.group(1).split()) or \
            '"RX_VK_MESH_SHARED_WARP_CHECK"' not in check.group(1):
        raise SystemExit("the check is not RX_VK_MESH_SHARED_WARP_CHECK over the switch")
    mesa, windows = "VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA", "VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS"
    nvidia, radv = "VK_DRIVER_ID_NVIDIA_PROPRIETARY", "VK_DRIVER_ID_MESA_RADV"
    cases = [(mesa, None, True, "Intel, Mesa"), (mesa, "0", False, "Intel, Mesa, =0"), (windows, None, False, "Intel, Windows"),
             (nvidia, None, False, "NVIDIA"), (radv, None, False, "AMD, RADV"), (radv, "1", True, "AMD, RADV, =1")]
    checks = "\n".join(
        f'    if (mesh_pages_wanted({v}, {"nullptr" if e is None else chr(34) + e + chr(34)}) != {str(w).lower()}) '
        f'{{ std::puts("FAIL: {name}"); return 1; }}'
        for v, e, w, name in cases)
    driver_enum = "enum VkDriverId { " + ", ".join(f"{k} = {n}" for k, n in DRIVER_IDS.items()) + " };\n"
    source = (ENV_H.read_text(encoding="utf-8") + "\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n" + driver_enum + fn +
              "\nint main() {\n" + checks + "\n    return 0;\n}\n")
    with tempfile.TemporaryDirectory(prefix="mesh-shared-warp-") as folder:
        folder = Path(folder)
        (folder / "t.cpp").write_text(source, encoding="utf-8")
        exe = folder / "t.exe"
        subprocess.run([cxx, "-std=c++20", "-w", str(folder / "t.cpp"), "-o", str(exe)], check=True)
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
        if run.returncode:
            raise SystemExit(run.stdout.strip() or "the switch check failed")
    return len(cases)


def shifted_overlap(flags, bit):
    """a field shifted into the flags (the software multiplies' form, `mma_software_form << 8`, values 0-2) takes two bits
    there that no `? Nu : 0u` shows: a switch's bit must not be one of them"""
    return any(bit >> int(s) & 3 for s in re.findall(r"<< (\d+)", flags))


def stamp():
    """both the switch and its check change the SPIR-V: each its own bit of the translator's stamp"""
    shader = (ROOT / "gpu/vk/vk_shader.cpp").read_text(encoding="utf-8")
    body = block(shader, "uint64_t translator_stamp(")
    flags = body[body.index("const uint32_t flags"):body.index("add(&flags")]
    bits = [int(b) for b in re.findall(r"\? (\d+)u : 0u", flags)]
    for name in ("mesh_fallback_shared_warp", "mesh_fallback_shared_warp_check"):
        bit = re.search(name + r" \? (\d+)u : 0u", flags)
        if not bit:
            raise SystemExit(f"translator_stamp does not take {name}")
        if bits.count(int(bit.group(1))) != 1 or shifted_overlap(flags, int(bit.group(1))):
            raise SystemExit(f"{name} shares its stamp bit {bit.group(1)}")


def planner():
    ctx = (SPIRV / "spirv_emit_context.cpp").read_text(encoding="utf-8")
    ctor = ctx[ctx.index("EmitContext::EmitContext("):ctx.index("EmitContext::~EmitContext()")]   # (its initializers have braces)
    # (the inputs depend on the plan: a stage on the shared warp declares none of the subgroup's)
    if not ctor.index("PlanSharedWarp(program);") < ctor.index("DefineInterfaces(program);"):
        raise SystemExit("PlanSharedWarp must run before DefineInterfaces, whose inputs depend on it")
    plan = block(ctx, "void EmitContext::PlanSharedWarp(")
    head = plan[:plan.index("std::vector<const IR::Block*> open;")]
    for needed, why in (("runtime_info.mesh_fallback", "only the fallback"), ("Stage::Task", "its task stages"),
                        ("Stage::Mesh", "its mesh stages"), ("mesh_fallback_shared_warp", "the switch"),
                        ("program.workgroup_size[0] != 32", "one 32-lane warp a workgroup"),
                        ("program.workgroup_size[1] != 1 || program.workgroup_size[2] != 1", "a workgroup of one row")):
        need(head, needed, f"the planner is not limited to {why}")
    # (what the shared warp cannot stand for: lanes that run in one thread in program order, or reads of the stage's outputs)
    refused = plan[plan.index("case IR::Opcode::QuadBroadcast:"):]
    refused = refused[:refused.index("return;")]
    for op in ("QuadBroadcast", "QuadSwap", "FSwizzleAdd", "MatrixMulAddF16", "SubgroupBarrier", "IsbeReadOutput"):
        need(refused, f"case IR::Opcode::{op}:", f"the planner takes a stage with {op}")
    for op in ("SubgroupBallot", "VoteAll", "VoteAny", "VoteEqual", "ShuffleIndex", "ShuffleUp", "ShuffleDown", "ShuffleButterfly"):
        need(plan, f"case IR::Opcode::{op}:", f"the planner gives {op} no words")
    # (a site's words serve one pass: none in a loop; a lane that entered a region reaches its sites: no return inside one)
    need(plan, "if (loops) {\n                    return;", "the planner takes a site inside a loop")
    ret = plan[plan.index("case IR::AbstractSyntaxNode::Type::Return:"):]
    need(ret[:ret.index("break;")], "if (!open.empty() || loops) {", "the planner takes a return inside an if or a loop")
    need(plan, "warp_word_count = (std::max)(words + WARP_REGION_WORDS * static_cast<u32>(warp_region_ifs.size()), 1U);",
         "the words planned are not the sites' and the regions'")
    define = block(ctx, "void EmitContext::DefineSharedWarp(")
    # (Sirit gives back a type already made from the same words: a u32 array of a u32 length may be a block's, with a stride)
    need(define, "TypeArray(U32[1], SConst(static_cast<s32>(warp_word_count)))", "the words' array type may be a block's")
    need(define, "spv::StorageClass::Workgroup", "the words are not in workgroup memory")
    for needed, why in (("OpLoopMerge(done, next, spv::LoopControlMask::DontUnroll)", "a loop"),
                        ("Const(WARP_WAIT)", "bounded"), ("OpAtomicLoad(U32[1], pointer", "an atomic read of the word"),
                        ("MeshFallbackWarpCounter(WARP_COUNTER_WAITS)", "counted when it gives up")):
        need(define, needed, f"the wait is not {why}")
    mesh = block(ctx, "void EmitContext::DefineMesh(")
    need(mesh, "DefineSharedWarp();", "DefineMesh does not make the shared warp")
    # (with it, no subgroup capability or input: the stage can run at any lane count; its check keeps them)
    inputs = block(ctx, "void EmitContext::DefineInputs(")
    need(inputs, "const bool subgroup_free{warp_shared && !warp_shared_check};", "DefineInputs does not know the stage is subgroup-free")
    if inputs.count("&& !subgroup_free") < 2:
        raise SystemExit("DefineInputs still declares the subgroup's masks or invocation id for a stage on the shared warp")
    emit = (SPIRV / "emit_spirv.cpp").read_text(encoding="utf-8")
    caps = block(emit, "void SetupCapabilities(")
    need(caps, "const bool subgroup_free{ctx.warp_shared && !ctx.warp_shared_check};", "SetupCapabilities does not know it either")
    need(caps, "profile.SupportsSubgroupStage(ctx.stage) && !subgroup_free", "the subgroup's capabilities are still added")


def emission():
    warp = (SPIRV / "emit_spirv_warp.cpp").read_text(encoding="utf-8")
    for fn, shared in (("Id EmitLaneId(", "ctx.warp_lane"), ("Id EmitVoteAll(", "SharedBallot(ctx, pred)"),
                       ("Id EmitVoteAny(", "SharedBallot(ctx, pred)"), ("Id EmitVoteEqual(", "SharedBallot(ctx, pred)"),
                       ("Id EmitSubgroupBallot(", "SharedBallot(ctx, pred)"), ("Id EmitSubgroupEqMask(", "ctx.warp_bit"),
                       ("Id EmitSubgroupLtMask(", "SharedLtMask(ctx)"), ("Id EmitSubgroupLeMask(", "SharedLeMask(ctx)"),
                       ("Id EmitSubgroupGtMask(", "SharedLeMask(ctx)"), ("Id EmitSubgroupGeMask(", "SharedLtMask(ctx)"),
                       ("Id EmitShuffleIndex(", "SharedShuffle(ctx, value, src_thread_id)"),
                       ("Id EmitShuffleUp(", "SharedShuffle(ctx, value, src_thread_id)"),
                       ("Id EmitShuffleDown(", "SharedShuffle(ctx, value, src_thread_id)"),
                       ("Id EmitShuffleButterfly(", "SharedShuffle(ctx, value, src_thread_id)")):
        body = block(warp, fn)
        need(body, "if (ctx.warp_shared) {", f"{fn[3:-1]} has no shared form")
        need(body, shared, f"{fn[3:-1]}'s shared form")
    need(block(warp, "Id EmitShuffleIndex("), "SharedBroadcast(ctx, value, lane.U32() & 31)", "a constant lane's shuffle is not one word")
    # (a ballot or a shuffle waits for the lanes of its region that it reads, never for lanes outside it)
    for fn in ("Id SharedBallot(", "Id SharedShuffle(", "Id SharedBroadcast("):
        body = block(warp, fn)
        need(body, "WarpWait(ctx, word", f"{fn[3:-1]} does not wait for the lanes it reads")
        need(body, "ctx.warp_region", f"{fn[3:-1]} does not wait within its region")
    region = block(warp, "void WarpOpenRegion(")
    for needed, why in (("ctx.warp_region_ifs.contains(body)", "only the ifs planned"),
                        ("WarpWait(ctx, word + 1, ctx.warp_region);", "after every lane around has said"),
                        ("ctx.warp_regions.push_back(ctx.warp_region);", "keeping the region around")):
        need(region, needed, f"a region is not opened {why}")
    need(block(warp, "void WarpCloseRegion("), "ctx.warp_region = ctx.warp_regions.back();", "a region's end does not restore the one around")
    traverse = block((SPIRV / "emit_spirv.cpp").read_text(encoding="utf-8"), "void Traverse(")
    cond_if = traverse[traverse.index("case IR::AbstractSyntaxNode::Type::If: {"):]
    # (a merge instruction must stand right before its branch: the region's words and wait come before it)
    if not cond_if.index("WarpOpenRegion(ctx, node.data.if_node.body") < cond_if.index("ctx.OpSelectionMerge(endif_label"):
        raise SystemExit("the region is opened after its if's merge instruction")
    end_if = traverse[traverse.index("case IR::AbstractSyntaxNode::Type::EndIf:"):]
    need(end_if[:end_if.index("break;")], "WarpCloseRegion(ctx);", "an if's end does not close its region")
    prologue = block(warp, "void WarpPrologue(")
    for needed, why in (("ctx.local_invocation_id", "the lane is the invocation's place in the workgroup"),
                        ("ctx.Const(0xffffffffU)", "the whole warp is the first region"),
                        ("ctx.OpStore(WarpWord(ctx, at), ctx.u32_zero_value);", "the words start at zero")):
        need(prologue, needed, f"the shared warp's start: {why}")
    mesh = (SPIRV / "emit_spirv_mesh.cpp").read_text(encoding="utf-8")
    body = block(mesh, "void EmitMeshPrologue(")
    # (zeroed before the barrier both stages' prologues end with: every lane's zero lands before any lane writes a word)
    first_barrier = body.index("Barrier(ctx);")
    if not body.index("WarpPrologue(ctx);") < first_barrier:
        raise SystemExit("the shared warp's words are zeroed after the prologue's first barrier")
    # (the epilogue's loop index is a function variable: those open the entry block, before any other instruction)
    if not body.index("ctx.mesh_fallback_epilogue_word = ctx.AddLocalVariable(") < body.index("WarpPrologue(ctx);"):
        raise SystemExit("the epilogue's variable comes after the shared warp's instructions in the entry block")


def constants():
    text = (SPIRV / "emit_spirv.h").read_text(encoding="utf-8")
    v = {name: int(value) for name, value in re.findall(r"constexpr u32 (WARP_\w+) = (\d+);", text)}
    for name in ("WARP_REGION_WORDS", "WARP_BALLOT_WORDS", "WARP_SHUFFLE_WORDS", "WARP_BROADCAST_WORDS", "WARP_COUNTER_BYTES",
                 "WARP_COUNTER_WAITS", "WARP_COUNTER_COMPARED", "WARP_COUNTER_DIFFERENT", "WARP_COUNTER_POOL_WAITS"):
        if name not in v:
            raise SystemExit(f"emit_spirv.h has no {name}")
    if v["WARP_SHUFFLE_WORDS"] != 33:   # (a value a lane and the arrivals)
        raise SystemExit("a shuffle's words are not 32 values and an arrival word")
    counters = (v["WARP_COUNTER_WAITS"], v["WARP_COUNTER_COMPARED"], v["WARP_COUNTER_DIFFERENT"], v["WARP_COUNTER_POOL_WAITS"])
    if len(set(counters)) != 4 or any(c % 4 or c + 4 > v["WARP_COUNTER_BYTES"] for c in counters):
        raise SystemExit("the shared warp's counters overlap or pass their bytes")
    need(text, "constexpr u32 WARP_WAIT = 1u << 16;", "the wait's bound changed")
    need(block(text, "inline u64 MeshFallbackRecordsBytes("), "mesh_fallback_global_attributes",
         "the counters' place does not follow the records' end with the attribute pool")


def host():
    draw = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    buffers = block(draw, "bool fallback_buffers_of(")
    need(buffers, "MeshFallbackCounters() ? Shader::Backend::SPIRV::WARP_COUNTER_BYTES : 0", "the records buffer has no room for the counters")
    need(buffers, "(warp_counters ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0)", "the counters cannot be copied out")
    record = block(draw, "bool fallback_record(")
    clear = record.index("vkCmdFillBuffer(cmd, f->records, Shader::Backend::SPIRV::MeshFallbackRecordsBytes()")
    first_send = record.index("send(stages[0]")
    barriers = [m.start() for m in re.finditer(r"full_barrier\(cmd\);", record)]
    if not any(clear < b < first_send for b in barriers):
        raise SystemExit("the counters' clear is not ordered before the draw's stages by a barrier")
    read = record.index("fallback_warp_counters(cmd, *f);")
    if not any(first_send < b < read for b in barriers):
        raise SystemExit("the counters are copied out before the barrier after the draw's stages")
    counters = block(draw, "void fallback_warp_counters(")
    need(counters, "spirv::MeshFallbackRecordsBytes()", "the counters are copied from somewhere other than where the stages count")
    need(counters, "f.warp_counters_cleared", "the counters are read before they were cleared")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    n = switch(args.cxx)
    stamp()
    planner()
    emission()
    constants()
    host()
    print(f"PASS: mesh fallback shared warp switch ({n} cases), cache stamp, planner, words and wait, warp operations, regions, "
          "prologue, counters")


if __name__ == "__main__":
    main()
