"""Compile the virtual cores' actual ordering functions (host/hos_thread.c) with a harness that never blocks, and check
the replaying thread's place ahead (hos_vcore_back_first): who a free core goes to, who a holder yields to, and that
the place ends when the core is given up.

python -I -B tests/test_vcore_replay_first.py [--cxx clang++]
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

RUN_ALL_ARGS = ["--cxx", "g++"]

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = ("static int vcp(const HosThread *t) {", "static int vc_better(", "static unsigned vc_allowed(", "static void vc_take(",
             "static void vc_dequeue(", "static uint64_t vc_chain_key(", "static void vc_grant(",
             "static void vc_release_locked(", "static void vc_release(", "static void vc_yield(")


def block(source, marker):
    start = source.index(marker)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    threads = (ROOT / "host/hos_thread.c").read_text(encoding="utf-8")
    native = "\n".join(block(threads, marker) for marker in FUNCTIONS)
    # (the entry point itself: it must set the place before it waits, and only when it holds no core)
    entry = block(threads, "void hos_vcore_back_first(")
    assert entry.index("if (t->vc_held) return;") < entry.index("t->vc_boost =") < entry.index("vc_acquire(t);"), entry
    compiler = shutil.which(args.cxx)
    if not compiler:
        raise SystemExit("C++ compiler not found: " + args.cxx)
    msvc = Path(compiler).name.lower() in ("cl", "cl.exe")
    env = os.environ.copy()
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        if not msvc:
            for name in ("INCLUDE", "LIB", "LIBPATH"):
                env.pop(name, None)
            windows = Path(os.environ["SystemRoot"])
            env["PATH"] = os.pathsep.join(map(str, (Path(compiler).resolve().parent, windows / "System32", windows)))

    def run(command, timeout):
        result = subprocess.run(command, env=env, creationflags=flags, timeout=timeout,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()

    with tempfile.TemporaryDirectory(prefix="vcore-replay-first-") as temporary:
        folder = Path(temporary)
        (folder / "actual_vcore.inc").write_text(native + "\n", encoding="utf-8")
        binary = folder / "fixture.exe"
        fixture = ROOT / "tests/vcore_replay_first_test.cpp"
        if msvc:
            command = [compiler, "/nologo", "/std:c++17", "/EHsc", "/W4", "/WX", "/I" + str(folder),
                       "/Fo" + str(folder / "fixture.obj"), "/Fe" + str(binary), str(fixture)]
        else:
            command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I" + str(folder), str(fixture),
                       "-o", str(binary)]
        run(command, 60)
        run([str(binary)], 10)


if __name__ == "__main__":
    main()
