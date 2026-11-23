"""Verify subgroup admission reduces CAS contention without combining operands.

Compute invocations take turns entering the CAS loop, which receives each original operand
and returns its own old value. The fixture checks RED and ATOM, subgroup support and fragment
fallback, and validates generated modules when spirv-val is available.

    python -I -B tests/test_float_add_subgroup.py --headers <SPIRV-Headers/include> [--cxx g++] [--spirv-val spirv-val]
"""
import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{spirv_headers}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"
SPIRV = VK / "eden" / "shader_recompiler" / "backend" / "spirv"
sys.path.insert(0, str(ROOT / "third_party" / "a64-kit" / "tools"))
import toolchain


def block(text, marker, start=0):
    at = text.index(marker, start)
    end = text.index("{", at) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[at:end]


def between(text, start, end, what):
    if start not in text or end not in text[text.index(start):]:
        raise SystemExit("FAIL: %s: no '%s' ... '%s'" % (what, start, end))
    at = text.index(start)
    return text[at:text.index(end, at)]


def cut(text, marker, what):
    if marker not in text:
        raise SystemExit("FAIL: %s: no '%s'" % (what, marker))
    return block(text, marker)


FIXTURE = r"""
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include <sirit/sirit.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/stage.h"
using Sirit::Id;
using Shader::Stage;
struct NotImplementedException : std::runtime_error {
    template <typename... A> explicit NotImplementedException(const char* what, A&&...) : std::runtime_error(what) {}
};
static bool g_result_read = false;   // (the add's result read: ATOM; not read: RED)
namespace IR {
struct Value {   // (immediate binding and offset only: what the fixture's one add uses)
    u32 value;
    bool IsImmediate() const { return true; }
    u32 U32() const { return value; }
};
struct Inst {
    bool HasUses() const { return g_result_read; }
};
}
enum class Operation { Increment, Decrement, FPAdd, FPMin, FPMax };
struct StorageTypeDefinition { Id array{}; Id element{}; };
struct StorageTypes { StorageTypeDefinition U32, F32; };
struct StorageDefinitions { Id U32{}; Id F32{}; };
struct Info { bool uses_atomic_f32_add{}; };
struct EmitContext : Sirit::Module {
    using Sirit::Module::Module;
    Shader::Profile profile{};
    Stage stage{Stage::Compute};
    bool mesh_fallback{};
    Id U1{};
    std::array<Id, 5> U32{}, F32{};
    Id u32_zero_value{};
    bool uses_explicit_workgroup_layout{};
    Id shared_memory_u32{};
    StorageTypes storage_types{};
    std::array<StorageDefinitions, 1> ssbos{};
    Id f32_add_cas{};
    Id f32_red_cas{};
    bool native_f32_add{};
    Id Const(u32 value) { return Constant(U32[1], value); }
    Id Def(const IR::Value&) { return u32_zero_value; }
    void define_storage_buffers(const Info& info);
};
@@CAS@@
void EmitContext::define_storage_buffers(const Info& info) {
@@DECIDE@@
@@ADDS@@
}
@@ATOMIC@@
int main(int argc, char** argv) {
    // (output, stage, subgroup support, result read, mesh fallback)
    if (argc < 6) return 2;
    EmitContext ctx{0x00010600};
    ctx.stage = std::strcmp(argv[2], "fragment") ? Stage::Compute : Stage::Fragment;
    ctx.mesh_fallback = argv[5][0] == '1';
    ctx.profile.supported_spirv = 0x00010600;
    ctx.profile.support_descriptor_aliasing = true;
    ctx.profile.supported_subgroup_stages = argv[3][0] == '1' ? 0x7F : 0;
    g_result_read = argv[4][0] == '1';
    ctx.AddCapability(spv::Capability::Shader);
    ctx.AddCapability(spv::Capability::VariablePointersStorageBuffer);   // (as DefineStorageBuffers for a program with a float add)
    ctx.AddExtension("SPV_KHR_storage_buffer_storage_class");
    ctx.SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    const Id void_t{ctx.TypeVoid()};
    ctx.U1 = ctx.TypeBool();
    ctx.U32[1] = ctx.TypeInt(32, false);
    ctx.F32[1] = ctx.TypeFloat(32);
    ctx.F32[2] = ctx.TypeVector(ctx.F32[1], 2);
    ctx.u32_zero_value = ctx.Constant(ctx.U32[1], 0u);
    std::vector<Id> interfaces;
    {   // (the u32 view, as DefineSsbos)
        const Id array{ctx.TypeRuntimeArray(ctx.U32[1])};
        ctx.Decorate(array, spv::Decoration::ArrayStride, 4U);
        const Id block{ctx.TypeStruct(array)};
        ctx.Decorate(block, spv::Decoration::Block);
        ctx.MemberDecorate(block, 0, spv::Decoration::Offset, 0U);
        ctx.storage_types.U32.array = ctx.TypePointer(spv::StorageClass::StorageBuffer, block);
        ctx.storage_types.U32.element = ctx.TypePointer(spv::StorageClass::StorageBuffer, ctx.U32[1]);
        const Id var{ctx.AddGlobalVariable(ctx.storage_types.U32.array, spv::StorageClass::StorageBuffer)};
        ctx.Decorate(var, spv::Decoration::Binding, 0U);
        ctx.Decorate(var, spv::Decoration::DescriptorSet, 0U);
        ctx.ssbos[0].U32 = var;
        interfaces.push_back(var);
    }
    Info info{true};
    ctx.define_storage_buffers(info);
    const Id main_id{ctx.OpFunction(void_t, spv::FunctionControlMask::MaskNone, ctx.TypeFunction(void_t))};
    ctx.AddLabel();
    IR::Inst add{};
    Shader::Backend::SPIRV::EmitStorageAtomicAddF32(ctx, @@INST@@IR::Value{0}, IR::Value{8}, ctx.Constant(ctx.F32[1], 1.0f));
    ctx.OpReturn();
    ctx.OpFunctionEnd();
    (void)add;
    // (a compute module whatever the stage the decision was told: the module is only for spirv-val)
    ctx.AddEntryPoint(spv::ExecutionModel::GLCompute, main_id, "main", interfaces);
    ctx.AddExecutionMode(main_id, spv::ExecutionMode::LocalSize, 64u, 1u, 1u);
    const std::vector<u32> code{ctx.Assemble()};
    FILE* f = std::fopen(argv[1], "wb");
    if (!f) return 3;
    std::fwrite(code.data(), 4, code.size(), f);
    std::fclose(f);
    return 0;
}
"""

OP_CAPABILITY, OP_ENTRY_POINT, OP_FUNCTION, OP_FUNCTION_END, OP_FUNCTION_CALL = 17, 15, 54, 56, 57
OP_ATOMIC_COMPARE_EXCHANGE, OP_ELECT, OP_BROADCAST_FIRST, OP_GROUP_FADD = 230, 333, 338, 350
CAP_GROUP_NON_UNIFORM, CAP_ARITHMETIC, CAP_BALLOT = 61, 63, 64
GROUP_OPS = (OP_ELECT, OP_BROADCAST_FIRST, OP_GROUP_FADD)


def instructions(words):
    i = 5
    while i < len(words):
        n = words[i] >> 16
        yield words[i] & 0xFFFF, words[i + 1:i + n]
        i += n


def functions(ops):
    """Each function's id, the ops in it and the functions it calls; and the entry point's id."""
    funcs, current, entry = {}, None, None
    for op, operands in ops:
        if op == OP_ENTRY_POINT:
            entry = operands[1]
        elif op == OP_FUNCTION:
            current = operands[1]
            funcs[current] = {"ops": [], "calls": []}
        elif op == OP_FUNCTION_END:
            current = None
        elif current is not None:
            funcs[current]["ops"].append((op, operands))
            if op == OP_FUNCTION_CALL:
                funcs[current]["calls"].append(operands[2])
    return funcs, entry


def check(name, ops, serialized):
    funcs, entry = functions(ops)
    caps = {operands[0] for op, operands in ops if op == OP_CAPABILITY}
    count = {op: sum(o == op for o, _ in ops) for op in GROUP_OPS + (OP_ATOMIC_COMPARE_EXCHANGE,)}
    if count[OP_ATOMIC_COMPARE_EXCHANGE] != 1:
        raise SystemExit("FAIL: %s: %d compare-exchanges (want the loop's one)" % (name, count[OP_ATOMIC_COMPARE_EXCHANGE]))
    loop = [f for f, body in funcs.items() if any(op == OP_ATOMIC_COMPARE_EXCHANGE for op, _ in body["ops"])][0]
    if serialized:
        if count[OP_ELECT] != 1 or count[OP_BROADCAST_FIRST] or count[OP_GROUP_FADD] or CAP_GROUP_NON_UNIFORM not in caps:
            raise SystemExit("FAIL: %s: expected subgroup election without a reduction" % name)
        wrappers = [f for f, body in funcs.items() if any(op == OP_ELECT for op, _ in body["ops"])]
        if len(wrappers) != 1 or funcs[entry]["calls"] != wrappers or funcs[wrappers[0]]["calls"] != [loop]:
            raise SystemExit("FAIL: %s: CAS loop is not reached through subgroup admission" % name)
        body = funcs[wrappers[0]]["ops"]
        params = [args[1] for op, args in body if op == 55]
        calls = [args for op, args in body if op == OP_FUNCTION_CALL]
        elect = [args[1] for op, args in body if op == OP_ELECT][0]
        branches = [args for op, args in body if op == 250 and args[0] == elect]
        if len(params) != 3 or len(calls) != 1 or calls[0][3:] != params or len(branches) != 2 or not any(op == 246 for op, _ in body):
            raise SystemExit("FAIL: %s: election must guard the CAS and loop exit, forwarding each operand" % name)
        label = None
        call_label = None
        for op, args in body:
            if op == 248:
                label = args[0]
            if op == OP_FUNCTION_CALL:
                call_label = label
        if branches[0][1] != call_label or branches[0][2] == call_label:
            raise SystemExit("FAIL: %s: CAS is not restricted to the elected invocation" % name)
        return "one CAS caller at a time per subgroup"
    if any(count[op] for op in GROUP_OPS) or caps & {CAP_GROUP_NON_UNIFORM, CAP_ARITHMETIC, CAP_BALLOT}:
        raise SystemExit("FAIL: %s: subgroup operations where the add must be the plain loop" % name)
    if funcs[entry]["calls"] != [loop] or any(op in (249, 250, 251) for op, _ in funcs[entry]["ops"]):
        raise SystemExit("FAIL: %s: the entry point does not call the loop unconditionally" % name)
    calls = [args for op, args in funcs[entry]["ops"] if op == OP_FUNCTION_CALL]
    # OpConstant operands are result type, result id, literal bits. The original 1.0 operand
    # must reach the loop directly, without a reduction or any other transformation.
    one = {args[1] for op, args in ops if op == 43 and len(args) == 3 and args[2] == 0x3f800000}
    if len(calls) != 1 or len(calls[0]) != 6 or calls[0][4] not in one:
        raise SystemExit("FAIL: %s: the original operand does not reach the CAS loop" % name)
    return "the plain loop"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--spirv-val", default=None)
    args = parser.parse_args()
    context = (SPIRV / "spirv_emit_context.cpp").read_text(encoding="utf-8")
    atomic = (SPIRV / "emit_spirv_atomic.cpp").read_text(encoding="utf-8")
    # (the CAS helpers, CasFunction to the descriptors' names, in the source's order; on main without SubgroupCasLoop)
    cas = between(context, "Id CasFunction(EmitContext& ctx, Operation operation, Id value_type)",
                  "template <typename Desc>\nstd::string NameOf", "spirv_emit_context.cpp")
    storage = cut(context, "void EmitContext::DefineStorageBuffers(", "spirv_emit_context.cpp")
    decide = storage[storage.index("    native_f32_add = "):storage.index("if (info.storage_buffers_descriptors.empty())")]
    adds = between(storage, "    if (info.uses_atomic_f32_add && !native_f32_add) {", "    if (info.uses_atomic_f16x2_add) {",
                   "DefineStorageBuffers")
    emit = cut(atomic, "Id EmitStorageAtomicAddF32(EmitContext& ctx", "emit_spirv_atomic.cpp")
    helpers = "\n".join(cut(atomic, marker, "emit_spirv_atomic.cpp") for marker in (
        "Id StorageIndex(EmitContext& ctx", "Id StoragePointer(EmitContext& ctx", "std::pair<Id, Id> AtomicArgs(EmitContext& ctx"))
    helpers = "namespace Shader::Backend::SPIRV {\n" + helpers + "\n" + emit + "\n}\n"
    # (main did not pass the instruction to the add emitter)
    takes_inst = "IR::Inst* inst" in emit[:emit.index("{")]
    fixture = (FIXTURE.replace("@@CAS@@", "namespace {\n" + cas + "\n}\n").replace("@@DECIDE@@", decide)
               .replace("@@ADDS@@", adds).replace("@@ATOMIC@@", helpers)
               .replace("@@INST@@", "&add, " if takes_inst else ""))
    sirit = VK / "sirit"
    with tempfile.TemporaryDirectory(prefix="float-add-subgroup-") as tmp:
        tmp = Path(tmp)
        cpp, exe = tmp / "fixture.cpp", tmp / "fixture.exe"
        cpp.write_text(fixture, encoding="utf-8")
        flags = []
        if os.name == "nt" and "clang" in Path(args.cxx).name.lower():
            flags = toolchain.native_target(host=True)
            if toolchain.mingw(host=True):
                flags += toolchain.native_link(host=True)
            else:
                here = str(Path(args.cxx).resolve().parent)
                link = shutil.which("lld-link", path=here) or shutil.which("lld-link")
                if not link:
                    lld = shutil.which("lld", path=here) or shutil.which("lld")
                    if lld:
                        link = str(tmp / "lld-link.exe")
                        shutil.copyfile(lld, link)
                if link:
                    flags += ["-fuse-ld=lld", "-B" + str(Path(link).parent), "-Wl,/nodefaultlib:oldnames"]
        r = subprocess.run([args.cxx, *flags, "-std=c++20", "-O1", "-I" + str(sirit / "include"), "-I" + str(sirit / "src"),
                            "-I" + str(args.headers), "-I" + str(VK / "eden"), "-I" + str(VK / "compat"), str(cpp),
                            *map(str, sorted((sirit / "src").rglob("*.cpp"))), "-o", str(exe)],
                           capture_output=True, text=True, timeout=600)
        if r.returncode:
            print(r.stdout + r.stderr)
            raise SystemExit("FAIL: the fixture did not build")
        # (name, stage, subgroup support, result read, mesh fallback)
        for name, stage, subgroup, read, mesh in (("RED", "compute", "1", "0", "0"),
                                                  ("ATOM", "compute", "1", "1", "0"),
                                                  ("RED, no subgroup", "compute", "0", "0", "0"),
                                                  ("RED in a fragment stage", "fragment", "1", "0", "0"),
                                                  ("RED mesh fallback", "compute", "1", "0", "1")):
            spv = tmp / ("%s.spv" % "".join(c if c.isalnum() else "_" for c in name))
            r = subprocess.run([str(exe), str(spv), stage, subgroup, read, mesh], capture_output=True, text=True, timeout=30)
            if r.returncode:
                print(r.stdout + r.stderr)
                raise SystemExit("FAIL: %s: the fixture stopped" % name)
            data = spv.read_bytes()
            ops = list(instructions(list(struct.unpack("<%dI" % (len(data) // 4), data))))
            what = check(name, ops, stage == "compute" and subgroup == "1" and mesh == "0")
            validator = args.spirv_val or shutil.which("spirv-val")
            if validator:
                r = subprocess.run([validator, "--target-env", "vulkan1.3", str(spv)], capture_output=True, text=True,
                                   timeout=60)
                if r.returncode:
                    print(r.stdout + r.stderr)
                    raise SystemExit("FAIL: %s: spirv-val" % name)
            print("%s: %s, %s" % (name, what, "spirv-val ok" if validator else "spirv-val unavailable"))
    print("PASS: subgroup admission preserves each operand and reduces CAS contention")


if __name__ == "__main__":
    main()
