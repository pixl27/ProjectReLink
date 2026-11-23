"""HOOKSHOT_TRACY on Windows: the executable's link line names every system library the Tracy client asks for.

The client asks for its libraries with `#pragma comment(lib, ...)` (ws2_32, advapi32, user32, secur32, dbghelp), and all
but dbghelp's are under _MSC_VER, which the clang of llvm-mingw does not define: there nothing reads them, and a library
missing from the link line is an undefined symbol at the link (GetUserNameExA, without secur32). So every library the
pragmas in third_party/tracy/public name must be on the Windows link of the executable: in HOST_LIBS, or in the libraries
nx2/build.py adds when HOOKSHOT_TRACY is set. Text only: no compiler, no Tracy build, no game.

    python -I -B tests/test_tracy_win_libs.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import between, find   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRACY = ROOT / "third_party" / "tracy" / "public"
PRAGMA = re.compile(r'#\s*pragma\s+comment\s*\(\s*lib\s*,\s*"([\w.]+?)(?:\.lib)?"\s*\)', re.I)
LIB = re.compile(r'"-l([\w.]+)"')


def pragma_libs():
    """library -> the client's files whose pragmas ask for it"""
    libs = {}
    for f in sorted(TRACY.rglob("*")):
        if f.suffix in (".c", ".cpp", ".h", ".hpp"):
            for lib in PRAGMA.findall(f.read_text(encoding="utf-8", errors="replace")):
                libs.setdefault(lib.lower(), []).append(f.relative_to(ROOT).as_posix())
    return libs


def linked_libs():
    """The -l libraries of the executable's Windows link: HOST_LIBS' Windows half and the HOOKSHOT_TRACY ones"""
    src = (ROOT / "nx2" / "build.py").read_text(encoding="utf-8")
    host = between(src, "HOST_LIBS = ", "\nGPU = ", where="nx2/build.py")
    windows = host.split("if LINUX else", 1)[1]
    at = find(src, '"/implib:"', where="nx2/build.py")   # (the Tracy part of the link: the client's import library)
    tracy = src[src.rfind("\n", 0, at) + 1:src.find("\n", at)]
    if "if TRACY" not in tracy:
        sys.exit("FAIL: the line with /implib: in nx2/build.py is no longer the HOOKSHOT_TRACY part of the link: %r" % tracy)
    return {l.lower() for l in LIB.findall(windows)}, {l.lower() for l in LIB.findall(tracy)}


def main():
    asked = pragma_libs()
    if not asked:
        sys.exit("FAIL: no #pragma comment(lib, ...) in %s: moved? update the check" % TRACY.relative_to(ROOT).as_posix())
    host, tracy = linked_libs()
    missing = sorted(l for l in asked if l not in host | tracy)
    for l in missing:
        print("FAIL: %s (asked for by %s) is not on the Windows link of the executable: neither in HOST_LIBS nor with "
              "HOOKSHOT_TRACY in nx2/build.py" % (l, ", ".join(asked[l])))
    if missing:
        sys.exit(1)
    print("OK: the %d libraries Tracy's pragmas ask for are linked (%s)" % (len(asked), ", ".join(sorted(asked))))


if __name__ == "__main__":
    main()
