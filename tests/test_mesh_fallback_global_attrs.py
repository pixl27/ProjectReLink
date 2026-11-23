"""The mesh fallback's attribute regions in a pool of the records buffer on Intel's Mesa driver (mma_layout.h
mesh_fallback_global_attributes).

Intel gives workgroup memory in powers of two: Hyrule Castle's colour mesh stages, 9 to 18 KB of it, took 16 or 32 KB each, few
workgroups fit on a subslice, and nothing hid their spill traffic (739acd67: 5.3-6.9 ms a dispatch, 2.6-2.8 ms with its region
in device memory). This checks the switch (Intel's Mesa driver, RX_VK_MESH_GLOBAL_ATTRS over it) and the program cache's stamp;
that every fallback access to the region goes through the accessors that choose workgroup memory or the slot; that the stages
with the region in a slot see it across the workgroup at their barriers, take a slot by a bounded wait in a function of its own
(the prologue's block must not branch), and hand it back at the epilogue's end; and that the host makes the pool and clears its
header before each draw.
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
    call = re.search(r"mesh_fallback_global_attributes = mesh_pages_wanted\(([^;]+)\);", init)
    if not call:
        raise SystemExit("coopmat_init does not set mesh_fallback_global_attributes from mesh_pages_wanted")
    args = " ".join(call.group(1).split())
    for needed in ("dev.driver_id", '"RX_VK_MESH_GLOBAL_ATTRS"'):
        if needed not in args:
            raise SystemExit(f"mesh_pages_wanted is not given {needed} for mesh_fallback_global_attributes")
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
    with tempfile.TemporaryDirectory(prefix="mesh-global-attrs-") as folder:
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
    shader = (ROOT / "gpu/vk/vk_shader.cpp").read_text(encoding="utf-8")
    body = block(shader, "uint64_t translator_stamp(")
    flags = body[body.index("const uint32_t flags"):body.index("add(&flags")]
    bit = re.search(r"mesh_fallback_global_attributes \? (\d+)u : 0u", flags)
    if not bit:
        raise SystemExit("translator_stamp does not take mesh_fallback_global_attributes")
    if [int(b) for b in re.findall(r"\? (\d+)u : 0u", flags)].count(int(bit.group(1))) != 1 or \
            shifted_overlap(flags, int(bit.group(1))):
        raise SystemExit(f"mesh_fallback_global_attributes shares its stamp bit {bit.group(1)}")


def constants():
    """the pool's layout: emit_spirv.h's constants, evaluated"""
    text = (SPIRV / "emit_spirv.h").read_text(encoding="utf-8")
    values = {}
    for name, expr in re.findall(r"constexpr u(?:32|64) (MESH_FALLBACK_\w+) = ([^;]+);", text):
        expr = re.sub(r"(\d+)u(?:ll)?\b", r"\1", re.sub(r"u64\{([^}]*)\}", r"(\1)", expr))
        try:
            values[name] = eval(expr, {}, dict(values))
        except (NameError, SyntaxError):   # (the push offsets, from names this check does not read)
            pass
    v = values
    if v["MESH_FALLBACK_POOL_OFFSET"] < (v["MESH_FALLBACK_HEADER_WORDS"] + v["MESH_FALLBACK_RECORD_WORDS"]) * 4:
        raise SystemExit("the pool starts inside the records")
    if v["MESH_FALLBACK_POOL_SLOTS"] != 1 << v["MESH_FALLBACK_POOL_SLOT_SHIFT"]:
        raise SystemExit("the pool's slot count is not 1 << its shift: a ticket's slot is its low bits")
    if v["MESH_FALLBACK_POOL_HEADER_BYTES"] < 4 * (1 + v["MESH_FALLBACK_POOL_SLOTS"]):
        raise SystemExit("the pool's header has no room for the counter and a word a slot")
    if v["MESH_FALLBACK_POOL_OFFSET"] + v["MESH_FALLBACK_POOL_BYTES"] + 16 > v["MESH_FALLBACK_RECORDS_LIMIT"]:
        raise SystemExit("the records buffer with its pool and counters passes fallback_buffers_of's limit")
    if v["MESH_FALLBACK_CAPACITY"] >= 1 << 32:   # (the counter is cleared before each draw: a draw's tickets must not wrap it)
        raise SystemExit("a draw's tickets would wrap the pool's counter")
    return v


def emission():
    ctx = (SPIRV / "spirv_emit_context.cpp").read_text(encoding="utf-8")
    define = block(ctx, "void EmitContext::DefineMesh(")
    cond = re.search(r"mesh_attributes_global = ([^;]+);", define)
    if not cond:
        raise SystemExit("DefineMesh does not decide mesh_attributes_global")
    c = " ".join(cond.group(1).split())
    for needed, why in (("mesh_fallback_global_attributes", "the switch"), ("mesh_fallback", "only the fallback"),
                        ("stage == Stage::Mesh", "only its mesh stages"), ("<= MESH_FALLBACK_POOL_SLOT_BYTES", "a region a slot holds")):
        need(c, needed, f"mesh_attributes_global is not limited to {why}")
    shared = define.index("if (!mesh_attributes_global) {")
    if define.index('Name(mesh_attributes, "isbe_attributes");') < shared:
        raise SystemExit("isbe_attributes is still made in workgroup memory for stages whose region is in a slot")
    # (with the switch off, the region and the variables after it are made as before: other GPUs' SPIR-V does not change)
    if not (define.index("mesh_map = DefineVariable") > define.index('Name(mesh_attributes, "isbe_attributes");') and
            define.index("if (mesh_attributes_global) {") > define.index('Name(mesh_primitive_count, "isbe_primitive_count");')):
        raise SystemExit("DefineMesh made its variables in another order: the SPIR-V of stages without a slot changes")
    claim = define[define.index("mesh_attribute_claim = OpFunction"):define.index("OpFunctionEnd();")]
    for needed, why in (("const Id is_first{OpIEqual(U1, local, u32_zero_value)}", "only thread 0 takes a ticket"),
                        ("OpBranchConditional(is_first, first_label, end_label)", "its test made before the merge"),
                        ("OpAtomicIAdd(U32[1], pool_word(u32_zero_value)", "a ticket from the pool's counter"),
                        ("Const(MESH_FALLBACK_POOL_SLOTS - 1)", "its slot from its low bits"),
                        ("Const(MESH_FALLBACK_POOL_SLOT_SHIFT)", "its generation from the rest"),
                        ("OpAtomicLoad(U32[1], slot_word", "the slot's word read until handed on"),
                        ("Const(MESH_FALLBACK_POOL_WAIT)", "a bounded wait"),
                        ("MeshFallbackWarpCounter(WARP_COUNTER_POOL_WAITS)", "a wait that gives up counted"),
                        ("OpLoopMerge(done, next, spv::LoopControlMask::DontUnroll)", "the wait a loop"),
                        ("OpStore(mesh_attribute_ticket_shared, ticket)", "the ticket handed to the workgroup")):
        need(claim, needed, f"the slot's claim: {why}")
    mesh = (SPIRV / "emit_spirv_mesh.cpp").read_text(encoding="utf-8")
    barrier = block(mesh, "void Barrier(EmitContext& ctx)")
    if "if (ctx.mesh_attributes_global)" not in barrier or "UniformMemory" not in barrier:
        raise SystemExit("the barriers of stages with the region in a slot do not make its writes seen across the workgroup")
    # (every fallback access to the region chooses: CopyLines and ConvertDepth are the native mesh path's)
    for name in ("Id LoadAttribute(", "void StoreAttribute("):
        need(block(mesh, name), "ctx.mesh_attributes_global", f"{name.split('(')[0].split()[-1]} does not choose the slot")
    rest = mesh
    for marker in ("void CopyLines(", "void ConvertDepth(", "Id LoadAttribute(", "void StoreAttribute("):
        rest = rest.replace(block(mesh, marker), "")
    raw = re.findall(r"SharedWord\(ctx, ctx\.mesh_attributes,", rest)
    if raw:
        raise SystemExit(f"{len(raw)} access(es) to the attribute region in workgroup memory that do not go through the accessors")
    for marker, needed, why in (
            ("void MeshFallbackEpilogue(", "LoadAttribute(ctx, index)", "the epilogue reads the region through the accessor"),
            ("Id EmitIsbeReadOutput(", "LoadAttribute(", "the program's reads of its outputs"),
            ("void EmitIsbeWriteAttribute(", "StoreAttribute(", "the program's writes"),
            ("void EmitMeshPrologue(", "ctx.OpFunctionCall(ctx.void_id, ctx.mesh_attribute_claim", "the prologue takes a slot by a call"),
            ("void EmitMeshPrologue(", "u64{MESH_FALLBACK_POOL_OFFSET} + MESH_FALLBACK_POOL_HEADER_BYTES", "the slots after the header"),
            ("void EmitMeshPrologue(", "StoreAttribute(ctx, Clamped(", "the region in its slot zeroed as in workgroup memory")):
        need(block(mesh, marker), needed, why)
    prologue = block(mesh, "void EmitMeshPrologue(")
    if prologue.index("mesh_attribute_claim") > prologue.index("StoreAttribute(ctx, Clamped("):
        raise SystemExit("the region is zeroed before the workgroup has its slot")
    epilogue = block(mesh, "void MeshFallbackEpilogue(")
    release = epilogue[epilogue.rindex("if (ctx.mesh_attributes_global)"):]
    for needed, why in (("Barrier(ctx);", "after every thread has read the region"),
                        ("ctx.OpAtomicExchange(ctx.U32[1], PoolWord(", "the slot's word set"),
                        ("ctx.Const(MESH_FALLBACK_POOL_SLOT_SHIFT)", "to the next generation")):
        need(release, needed, f"the epilogue's end does not hand the slot back {why}")
    if epilogue.rindex("if (ctx.mesh_attributes_global)") < epilogue.rindex("ctx.AddLabel(done);"):
        raise SystemExit("the slot is handed back before the records are copied from it")


def host():
    draw = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    buffers = block(draw, "bool fallback_buffers_of(")
    need(buffers, "mesh_fallback_global_attributes", "fallback_buffers_of makes no pool")
    need(buffers, "MESH_FALLBACK_POOL_OFFSET + Shader::Backend::SPIRV::MESH_FALLBACK_POOL_BYTES", "the records buffer is not sized for the pool")
    record = block(draw, "bool fallback_record(")
    fill = re.search(r"vkCmdFillBuffer\(cmd, f->records, Shader::Backend::SPIRV::MESH_FALLBACK_POOL_OFFSET,\s*"
                     r"Shader::Backend::SPIRV::MESH_FALLBACK_POOL_HEADER_BYTES, 0\)", record)
    if not fill:
        raise SystemExit("fallback_record does not clear the pool's header before the draw")
    barriers = [m.start() for m in re.finditer(r"full_barrier\(cmd\);", record)]
    if not any(b > fill.start() for b in barriers) or not any(b < fill.start() for b in barriers):
        raise SystemExit("the pool's clear is not between the barriers that order it after the draw before and before this one")
    # (before the draw's stages, with a barrier between: a clear after them, unordered, could land while the mesh stage takes and
    # hands back slots)
    stages = record.index("vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE")
    first_send = record.index("send(stages[0]")
    if fill.start() > stages or fill.start() > first_send or not any(fill.start() < b < first_send for b in barriers):
        raise SystemExit("the pool's clear is not ordered before the draw's task and mesh stages by a barrier")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    n = switch(args.cxx)
    stamp()
    constants()
    emission()
    host()
    print(f"PASS: mesh fallback attribute pool switch ({n} cases), cache stamp, pool layout, the region's accesses, slot claim "
          "and hand-back, and the host's pool")


if __name__ == "__main__":
    main()
