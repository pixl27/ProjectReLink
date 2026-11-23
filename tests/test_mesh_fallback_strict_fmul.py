"""AMD's older Windows proprietary shader compiler (RDNA 1 and 2, RX 5000 and RX 6000) reassociates NoContraction
OpFMul inside compute-fallback mesh stages even with DISABLE_OPTIMIZATION_BIT, causing grass blade tips to differ
by 1 ULP between depth pre-pass and colour pass and flash white under EQUAL depth testing.

EmitFPMul32 works around this by emitting OpFma(a, b, -0.0f) when ctx.mesh_fallback, mesh_fallback_strict_fmul, and
no_contraction are all true. This game-free check verifies:
1. EmitFPMul32 compiled against Sirit emits GLSL.std.450 Fma with -0.0f (0x80000000) and NoContraction when all three
   conditions hold, and plain OpFMul whenever any of the three is false.
2. coopmat_init() wires dev.policy.mesh_fallback_strict_fmul to Shader::Backend::SPIRV::mesh_fallback_strict_fmul.
3. translator_stamp() folds mesh_fallback_strict_fmul into a unique, non-overlapping bit of flags.

    python -I -B tests/test_mesh_fallback_strict_fmul.py --cxx <g++/clang++> --headers <SPIRV-Headers/include>
"""
import argparse
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srccut import block  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{spirv_headers}"]

ROOT = Path(__file__).resolve().parents[1]
SPIRV = ROOT / "gpu" / "vk" / "eden" / "shader_recompiler" / "backend" / "spirv"

FIXTURE = r"""
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <sirit/sirit.h>

using u32 = std::uint32_t;
using f32 = float;
using Sirit::Id;

bool mesh_fallback_strict_fmul = false;

namespace IR {
struct FpControl {
    bool no_contraction = false;
};
struct Inst {
    FpControl ctrl{};
    template <typename T>
    T Flags() const { return ctrl; }
};
} // namespace IR

struct EmitContext : Sirit::Module {
    using Sirit::Module::Module;
    bool mesh_fallback = false;
    std::array<Id, 5> F32{};
    Id Const(f32 v) { return Constant(F32[1], v); }
};

@@DECORATE@@
@@EMIT_FPMUL32@@

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    const bool fb = argv[1][0] == '1';
    mesh_fallback_strict_fmul = argv[2][0] == '1';
    const bool nc = argv[3][0] == '1';

    EmitContext ctx{0x00010600};
    ctx.mesh_fallback = fb;
    ctx.AddCapability(spv::Capability::Shader);
    ctx.SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    const Id f32t{ctx.TypeFloat(32)};
    ctx.F32[1] = f32t;
    const Id voidt{ctx.TypeVoid()};
    const Id main_fn{ctx.OpFunction(voidt, spv::FunctionControlMask::MaskNone, ctx.TypeFunction(voidt))};
    ctx.AddLabel();
    IR::Inst inst{{nc}};
    (void)EmitFPMul32(ctx, &inst, ctx.Const(1.5f), ctx.Const(2.5f));
    ctx.OpReturn();
    ctx.OpFunctionEnd();
    ctx.AddEntryPoint(spv::ExecutionModel::GLCompute, main_fn, "main");
    ctx.AddExecutionMode(main_fn, spv::ExecutionMode::LocalSize, 1u, 1u, 1u);

    const std::vector<u32> code{ctx.Assemble()};
    FILE* f = std::fopen(argv[4], "wb");
    if (!f) return 3;
    std::fwrite(code.data(), 4, code.size(), f);
    std::fclose(f);
    return 0;
}
"""

OP_EXT_INST = 12
OP_CONSTANT = 43
OP_DECORATE = 71
OP_FMUL = 133
DEC_NO_CONTRACTION = 42
GLSL_FMA = 50


def inspect_spv(path):
    data = path.read_bytes()
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    consts = {}
    no_contract = set()
    fmuls = []
    fmas = []
    i = 5
    while i < len(words):
        op = words[i] & 0xFFFF
        cnt = words[i] >> 16
        a = words[i + 1:i + cnt]
        if op == OP_CONSTANT and len(a) == 3:
            consts[a[1]] = a[2]
        elif op == OP_DECORATE and len(a) >= 2 and a[1] == DEC_NO_CONTRACTION:
            no_contract.add(a[0])
        elif op == OP_FMUL:
            fmuls.append(a[1])
        elif op == OP_EXT_INST and len(a) == 7 and a[3] == GLSL_FMA:
            fmas.append((a[1], a[4], a[5], a[6]))
        i += cnt
    return consts, no_contract, fmuls, fmas


def check_wiring():
    coop = (ROOT / "gpu" / "vk" / "vk_coopmat.cpp").read_text(encoding="utf-8")
    init = block(coop, "void coopmat_init(")
    if "Shader::Backend::SPIRV::mesh_fallback_strict_fmul = dev.policy.mesh_fallback_strict_fmul;" not in init:
        raise SystemExit("FAIL: coopmat_init does not set Shader::Backend::SPIRV::mesh_fallback_strict_fmul")

    shader = (ROOT / "gpu" / "vk" / "vk_shader.cpp").read_text(encoding="utf-8")
    stamp = block(shader, "uint64_t translator_stamp(")
    flags = stamp[stamp.index("const uint32_t flags"):stamp.index("add(&flags")]
    bits = [int(b) for b in re.findall(r"\? (\d+)u : 0u", flags)]
    m = re.search(r"mesh_fallback_strict_fmul \? (\d+)u : 0u", flags)
    if not m:
        raise SystemExit("FAIL: translator_stamp does not fold mesh_fallback_strict_fmul")
    bit = int(m.group(1))
    if bits.count(bit) != 1 or any((bit >> int(s)) & 3 for s in re.findall(r"<< (\d+)", flags)):
        raise SystemExit(f"FAIL: mesh_fallback_strict_fmul stamp bit {bit} overlaps another flag")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()

    check_wiring()

    fp = (SPIRV / "emit_spirv_floating_point.cpp").read_text(encoding="utf-8")
    decorate = block(fp, "Id Decorate(")
    emit_mul = block(fp, "Id EmitFPMul32(")
    fixture = FIXTURE.replace("@@DECORATE@@", decorate).replace("@@EMIT_FPMUL32@@", emit_mul)
    sirit = ROOT / "gpu" / "vk" / "sirit"

    with tempfile.TemporaryDirectory(prefix="strict-fmul-") as tmp:
        tmp = Path(tmp)
        cpp, exe = tmp / "fixture.cpp", tmp / "fixture.exe"
        cpp.write_text(fixture, encoding="utf-8")
        r = subprocess.run(
            [args.cxx, "-std=c++20", "-O1", "-I" + str(sirit / "include"), "-I" + str(sirit / "src"),
             "-I" + str(args.headers), str(cpp), *map(str, sorted((sirit / "src").rglob("*.cpp"))), "-o", str(exe)],
            capture_output=True, text=True, timeout=600)
        if r.returncode:
            print(r.stdout + r.stderr)
            raise SystemExit("FAIL: fixture did not build")

        cases = [
            (1, 1, 1, True),   # mesh fallback + strict_fmul + NoContraction -> OpFma(a, b, -0.0f)
            (0, 1, 1, False),  # non-fallback stage -> OpFMul
            (1, 0, 1, False),  # strict_fmul off -> OpFMul
            (1, 1, 0, False),  # no NoContraction -> OpFMul
        ]
        for fb, strict, nc, expect_fma in cases:
            spv = tmp / f"c_{fb}{strict}{nc}.spv"
            subprocess.run([str(exe), str(fb), str(strict), str(nc), str(spv)], check=True, timeout=30)
            consts, no_contract, fmuls, fmas = inspect_spv(spv)
            if expect_fma:
                if fmuls or len(fmas) != 1:
                    raise SystemExit(f"FAIL: case ({fb},{strict},{nc}) expected 1 Fma and 0 FMul, got fmas={fmas} fmuls={fmuls}")
                rid, _, _, c_id = fmas[0]
                if consts.get(c_id) != 0x80000000:
                    raise SystemExit(f"FAIL: case ({fb},{strict},{nc}) Fma addend is {consts.get(c_id):#x}, expected -0.0f (0x80000000)")
                if rid not in no_contract:
                    raise SystemExit(f"FAIL: case ({fb},{strict},{nc}) Fma missing NoContraction")
            else:
                if fmas or len(fmuls) != 1:
                    raise SystemExit(f"FAIL: case ({fb},{strict},{nc}) expected 1 FMul and 0 Fma, got fmuls={fmuls} fmas={fmas}")
                if bool(nc) != (fmuls[0] in no_contract):
                    raise SystemExit(f"FAIL: case ({fb},{strict},{nc}) unexpected NoContraction state")

    print("ok: EmitFPMul32 emits OpFma(a, b, -0.0f) with NoContraction only when mesh_fallback && mesh_fallback_strict_fmul && no_contraction; wiring and stamp verified")


if __name__ == "__main__":
    main()
