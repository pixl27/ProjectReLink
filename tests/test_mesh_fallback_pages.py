"""The mesh fallback's global loads through the guest page table on Intel's Mesa driver (mma_layout.h mesh_fallback_pages).

The fallback's task and mesh stages are held to 32 lanes; on an Iris Xe its colour mesh stages sit at the register file's limit,
and each global load's storage buffer search, inlined at every load, filled them with spill code (Hyrule Castle's 739acd67:
12.3 ms a frame, 5.7 ms through the table). This checks the switch (Intel's Mesa driver, RX_VK_MESH_PAGES over it), the program
cache's stamp, that only the fallback's task and mesh stages take the table, and that its lookup is the page table the renderer
writes: the same two windows, the same page numbering, 0 where a page is not mapped.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]
ENV_H = ROOT / "gpu/vk/compat/hookshot_env.h"   # (env_value_off)
EMIT = ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/spirv_emit_context.cpp"
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


def switch(cxx):
    """mesh_pages_wanted cut from vk_coopmat.cpp and run for the GPUs and settings that matter"""
    coopmat = (ROOT / "gpu/vk/vk_coopmat.cpp").read_text(encoding="utf-8")
    fn = block(coopmat, "bool mesh_pages_wanted(")
    init = block(coopmat, "void coopmat_init(")
    call = re.search(r"mesh_fallback_pages = mesh_pages_wanted\(([^;]+)\);", init)
    if not call:
        raise SystemExit("coopmat_init does not set mesh_fallback_pages from mesh_pages_wanted")
    args = " ".join(call.group(1).split())
    for needed in ("dev.driver_id", '"RX_VK_MESH_PAGES"'):
        if needed not in args:
            raise SystemExit(f"mesh_pages_wanted is not given {needed}")
    mesa, windows = "VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA", "VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS"
    nvidia, radv = "VK_DRIVER_ID_NVIDIA_PROPRIETARY", "VK_DRIVER_ID_MESA_RADV"
    cases = [  # driver, RX_VK_MESH_PAGES, wanted
        (mesa, None, True, "Intel, Mesa"),
        (mesa, "0", False, "Intel, Mesa, RX_VK_MESH_PAGES=0"),
        (mesa, "off", False, "Intel, Mesa, RX_VK_MESH_PAGES=off"),
        (windows, None, False, "Intel, Windows driver"),
        (nvidia, None, False, "NVIDIA"),
        (radv, None, False, "AMD, RADV"),
        (radv, "1", True, "AMD, RADV, RX_VK_MESH_PAGES=1"),
    ]
    checks = "\n".join(
        f'    if (mesh_pages_wanted({v}, {"nullptr" if e is None else chr(34) + e + chr(34)}) != {str(w).lower()}) '
        f'{{ std::puts("FAIL: {name}"); return 1; }}'
        for v, e, w, name in cases)
    driver_enum = "enum VkDriverId { " + ", ".join(f"{k} = {n}" for k, n in DRIVER_IDS.items()) + " };\n"
    source = (ENV_H.read_text(encoding="utf-8") + "\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n" + driver_enum + fn +
              "\nint main() {\n" + checks + "\n    return 0;\n}\n")
    with tempfile.TemporaryDirectory(prefix="mesh-pages-") as folder:
        folder = Path(folder)
        (folder / "t.cpp").write_text(source, encoding="utf-8")
        exe = folder / "t.exe"
        subprocess.run([cxx, "-std=c++20", "-Wall", "-Wextra", str(folder / "t.cpp"), "-o", str(exe)], check=True)
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
    body = block(shader, "uint64_t translator_stamp(")   # (the program cache's stamp: vk_progcache.cpp's disk_stamp takes it)
    bit = re.search(r"mesh_fallback_pages \? (\d+)u : 0u", body)
    if not bit:
        raise SystemExit("translator_stamp does not take mesh_fallback_pages: programs cached in one form are used in the other")
    flags = body[body.index("const uint32_t flags"):body.index("add(&flags")]
    others = [int(b) for b in re.findall(r"\? (\d+)u : 0u", flags)]
    if others.count(int(bit.group(1))) != 1 or shifted_overlap(flags, int(bit.group(1))):
        raise SystemExit(f"mesh_fallback_pages shares its stamp bit {bit.group(1)} with another switch")


def emission():
    emit = EMIT.read_text(encoding="utf-8")
    fn = block(emit, "void EmitContext::DefineGlobalMemoryFunctions(")
    cond = re.search(r"const bool fallback_pages\{([^;]+)\};", fn)
    if not cond:
        raise SystemExit("DefineGlobalMemoryFunctions has no fallback_pages condition")
    c = " ".join(cond.group(1).split())
    for needed, why in (("mesh_fallback_pages", "the switch"), ("mesh_fallback", "only the fallback's stages"),
                        ("stage == Stage::Task || stage == Stage::Mesh", "only its task and mesh stages"),
                        ("wants_guest_pages", "a stage whose push block has the table's address"),
                        ("rescaling_push_constants.value != 0", "a stage with the push block made"),
                        ("profile.supported_spirv >= 0x00010400", "a vector's OpSelect on one condition (SPIR-V 1.4)")):
        if needed not in c:
            raise SystemExit(f"fallback_pages is not limited to {why} ({needed} not found)")
    page = block(fn, "const auto page_load = [&](Id type, Id addr)")
    guest = block(fn, "const auto guest_lookup = [&](Id addr, u32 size, auto&& hit)")
    # (the table's own lookup: the same address, windows and page numbering as the compute stages' guest_lookup)
    for needed in ("Const(guest_pages_member_index)",
                   "OpISub(U64, addr, Constant(U64, u64{kGuestPagesBase}))",
                   "OpISub(U64, addr, Constant(U64, u64{kGuestPagesBase2}))",
                   "OpULessThan(U1, offset_a, Constant(U64, u64{kGuestPagesBytes}))",
                   "OpULessThan(U1, offset_b, Constant(U64, u64{kGuestPagesBytes2}))",
                   "OpIAdd(U64, OpShiftRightLogical(U64, offset_b, Const(12U)), Constant(U64, u64{kGuestPagesBytes >> 12}))",
                   "OpShiftLeftLogical(U64, page_index, Const(3U))",
                   "OpBitwiseAnd(U64, addr, Constant(U64, u64{0xfff}))"):
        if needed not in page:
            raise SystemExit(f"page_load does not look the address up as the page table is written: {needed}")
        if needed not in guest:
            raise SystemExit(f"guest_lookup changed ({needed}): check page_load against it and update this check")
    for needed, why in (("psb_load(U64, slot, 8U)", "the page's word is read"),
                        ("OpINotEqual(U1, page, u64_zero)", "an unmapped page is told apart"),
                        ("OpSelect(type, mapped, psb_load(type, target, 4U), ConstantNull(type))", "0 where nothing is mapped")):
        if needed not in page:
            raise SystemExit(f"page_load: {why} ({needed} not found)")
    load = block(fn, "const auto define_load{")
    at = load.index("if (fallback_pages")
    if load.index("OpReturnValue(page_load(type, addr));", at) > load.index("define_body(ssbo_member", at):
        raise SystemExit("define_load searches the storage buffers before it returns the page table's answer")
    write = block(fn, "const auto define_write{")
    if "page_load" in write or "fallback_pages" in write:
        raise SystemExit("the stores now go through the table: the check and mma_layout.h say they keep the search")
    # (the windows the shader reads are the ones vk_memory.cpp writes, page_table_census.h kPageWindows)
    consts = dict(re.findall(r"constexpr unsigned long long (kGuestPages\w+) = ([^;]+);", emit))
    value = {k: eval(v.replace("ull", "")) for k, v in consts.items()}
    memory = (ROOT / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    if "for (const PageWindow& w : kPageWindows)" not in block(memory, "void guest_pages_set("):
        raise SystemExit("vk_memory.cpp guest_pages_set no longer writes kPageWindows: check the shader's windows against it")
    census = (ROOT / "gpu/vk/page_table_census.h").read_text(encoding="utf-8")
    want = re.search(r"kPageWindows\[\] = \{\{(0x[0-9a-f]+)ull, (\d+)ull << (\d+), 0\}, \{(0x[0-9a-f]+)ull, (\d+)ull << (\d+),", census)
    if not want:
        raise SystemExit("page_table_census.h kPageWindows not found")
    a, a_n, a_s, b, b_n, b_s = want.groups()
    if (value["kGuestPagesBase"], value["kGuestPagesBytes"], value["kGuestPagesBase2"], value["kGuestPagesBytes2"]) != (
            int(a, 16), int(a_n) << int(a_s), int(b, 16), int(b_n) << int(b_s)):
        raise SystemExit("the shader's page table windows are not the ones vk_memory.cpp writes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    n = switch(args.cxx)
    stamp()
    emission()
    print(f"PASS: mesh fallback page table switch ({n} cases), cache stamp, and the table's lookup as the renderer writes it")


if __name__ == "__main__":
    main()
