"""The compute fallback's records area holds the floor grass at the Grass distance's 4x, without game files or a GPU.

Every mesh program on the fallback takes its records from one fixed area (emit_spirv.h MESH_FALLBACK_RECORD_WORDS), so a program
with long records fits fewer: the grass's blades (records of 22 words) were held at 762528 records a frame at 4x while their
partner program (7 words) drew ~1.02M, and the blades it left out were drawn white. tests/mesh_fallback_capacity_test.cpp
gives both programs the same workgroups at the measured load and at the load the caps reach, with the real
MeshFallbackFit, and checks they draw the same records, every one. Then it calls vk_draw.cpp's real fallback_buffers_of (cut
out here), with the attribute pool and the shared warp's counters off and on, and checks the records buffer that area needs is
made: past its limit fallback_buffers_of refuses it, and every program's fallback with it.

    python -I -B tests/test_mesh_fallback_capacity.py [--cxx clang++]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block, between   # noqa: E402
from test_mesh_fallback_layout import flags   # noqa: E402  (the recompiler's include paths and C++23 mode)

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
CASES = ("grass_pair_measured", "grass_pair_cap", "buffers", "buffers_pool", "buffers_warp", "buffers_pool_warp")


def cut(source):
    """vk_draw.cpp's thread-owned objects, FallbackBuffers, its thread's list and fallback_buffers_of with what it calls"""
    tls = between(source, "struct TlsOwned {", "\n\n")
    tls_macro = between(source, "#define TLS_OBJECT(", "\n\n")
    if tls_macro not in tls:
        tls += "\n" + tls_macro
    return "\n\n".join((tls, block(source, "struct FallbackBuffers {") + ";",
                        next(line for line in source.splitlines() if line.startswith("TLS_OBJECT(fallback_buffers,")),
                        block(source, "uint64_t fallback_buffer_address("), block(source, "bool fallback_buffers_of("))) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    p.add_argument("--source", type=Path, default=ROOT / "gpu/vk/vk_draw.cpp", help="another vk_draw.cpp, for a control")
    a = p.parse_args()
    cxx = shutil.which(a.cxx)
    if not cxx:
        sys.exit("FAIL: C++ compiler not found: " + a.cxx)
    env = dict(os.environ)
    env["PATH"] = str(Path(cxx).parent) + os.pathsep + env.get("PATH", "")   # (llvm-mingw's runtime DLLs beside it)
    creation = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    with tempfile.TemporaryDirectory(prefix="mesh-fallback-capacity-") as tmp:
        tmp = Path(tmp)
        (tmp / "fallback_actual.inc").write_text(cut(a.source.read_text(encoding="utf-8")), encoding="utf-8")
        exe = tmp / ("capacity.exe" if os.name == "nt" else "capacity")
        # (the recompiler's flags; no SPIR-V header is read, so its folder is one already on the list)
        try:
            f = flags(argparse.Namespace(cxx=cxx, headers=ROOT / "gpu" / "vk"))
        except SystemExit as e:   # (a compiler with no C++23 mode: skipped, as tests/test_mesh_fallback_layout.py is there)
            if "no C++23 mode" not in str(e):
                raise
            print("SKIP: %s" % str(e)[6:])
            return
        made = subprocess.run([cxx, *f, "-I" + str(tmp), str(ROOT / "tests" / "mesh_fallback_capacity_test.cpp"),
                               "-o", str(exe)], capture_output=True, text=True, env=env, creationflags=creation)
        if made.returncode:
            print(made.stdout, made.stderr, end="")
            sys.exit("FAIL: the check does not compile")
        failed = 0
        for case in CASES:
            r = subprocess.run([str(exe), case], capture_output=True, text=True, env=env, timeout=60, creationflags=creation)
            print((r.stdout + r.stderr).rstrip())
            failed += r.returncode != 0
    print("mesh fallback capacity: %d cases, %d failed" % (len(CASES), failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
