"""A thread's stack at Windows' commit limit. A thread's stack is only reserved (the guest's 64 MB,
host/hos_thread.c), each page committed as it grows. At the commit limit Windows can't commit the next one and raises a stack
overflow (0xc00000fd) with the stack nearly unused: a 24 GB PC at its limit crashed so on PreActorModelCompose. Guest memory
already waits there for the renderer to give commit back (commit_wait); a stack now does too, and goes on.

Compiles the real code of host/mkw_glue.c from commit_wait to the crash report (the commit waits, the out-of-memory notice,
the first-touch handler and what sits beside it) with tests/stack_commit_test.c, registering the vectored handlers
hos_commit_init registers from that code, as it does. Each case in its own process, which puts itself in a Job object with a
commit limit and fills it:

- recovers: commit comes back 300 ms after a thread with a reserved 64 MB stack starts growing it: it gets 6.4 MB deep and
  returns (on main it died with 0xc00000fd);
- never: nothing comes back: after the wait, the out-of-memory line, then 0xc00000fd (the crash report's, as before);
- both again with 512-byte frames ("small"), whose stack pointer is in the last committed page when the next is refused, as
  in the game: the handler has under 4 KB to run on, and the stand-in hos_log takes the real one's 4.3 KB, so a handler
  that logs before it has committed room below the stack dies there with no line;
- ran-out: a stack that really runs out (1.6 GB deep): 0xc00000fd at once, never tried again forever.

Windows only (Job objects, the stack's guard page); skipped elsewhere.

python -I -B tests/test_stack_commit.py [--cc clang]
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUN_ALL_SKIP = {"linux": "Windows only: a Job object's commit limit and a thread stack's guard page"}
STACK_OVERFLOW = 0xC00000FD
OOM_LINE = "mem: Windows is out of memory (RAM and page file)"


def cut(glue):
    """mkw_glue.c from commit_wait to the comment before the crash report: the commit waits and their handlers"""
    start = glue.find("static int commit_wait(void *p, SIZE_T size)")
    end = glue.find("/* A crash nothing handles")
    if start < 0 or end < start:
        raise SystemExit("FAIL: mkw_glue.c has no commit_wait ... 'A crash nothing handles' stretch")
    at = glue.rfind("\n/*", 0, start)   # (with commit_wait's own comment)
    return glue[at if at >= 0 else start:end]


def handlers(glue, code):
    """hos_commit_init's lines for the cut code, in its order: the AddVectoredExceptionHandler calls whose handler is in it,
    and what it sets of the cut code's own statics (stack_spare, the commit held back for a stack that never grew)"""
    init = glue.find("void hos_commit_init(void)")
    body = glue[init:glue.index("\n}", init)] if init >= 0 else ""
    kept = []
    for line in body.splitlines():
        line = line.strip()
        call = re.fullmatch(r"AddVectoredExceptionHandler\((\d+), (\w+)\);", line)
        if call and re.search(r"static LONG CALLBACK %s\(" % call.group(2), code):
            kept.append(line)
        static = re.fullmatch(r"(\w+) = .*;", line)
        if static and re.search(r"^static [^;(]*\b%s;" % static.group(1), code, re.M):
            kept.append(line)
    if not any(line.startswith("AddVectoredExceptionHandler") for line in kept):
        raise SystemExit("FAIL: hos_commit_init registers no handler from the cut code")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()
    if os.name != "nt":
        print("SKIP: " + RUN_ALL_SKIP["linux"])
        return
    cc = shutil.which(args.cc) or shutil.which("gcc")
    if not cc:
        raise SystemExit("test_stack_commit: no C compiler %r (pass --cc)" % args.cc)
    glue = (ROOT / "host/mkw_glue.c").read_text(encoding="utf-8")
    code = cut(glue)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    with tempfile.TemporaryDirectory(prefix="stack-commit-") as temporary:
        folder = Path(temporary)
        (folder / "stack_commit.inc").write_text(code, encoding="utf-8")
        (folder / "stack_commit_handlers.inc").write_text("\n".join(handlers(glue, code)) + "\n", encoding="utf-8")
        executable = folder / "stack_commit_test.exe"
        subprocess.run([cc, "-std=gnu11", "-O1", "-Wall", "-Wno-unused-function", "-I", str(folder),
                        str(ROOT / "tests/stack_commit_test.c"), "-o", str(executable), "-luser32"], check=True,
                       creationflags=flags)

        def run(case, *more):
            result = subprocess.run([str(executable), case, *more], capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=60, creationflags=flags)
            code = result.returncode & 0xFFFFFFFF
            print("%s: exit %#x\n%s" % (" ".join((case,) + more), code, result.stderr), end="")
            if code == 4:
                raise SystemExit("SKIP: " + result.stderr.strip())
            return code, result.stderr

        code, err = run("recovers")
        if code == STACK_OVERFLOW:
            raise SystemExit("FAIL: a thread whose stack page was refused at the commit limit died with a stack overflow "
                             "(0xc00000fd), with commit given back 300 ms later and most of its 64 MB unused")
        if code or "returned 0" not in err:
            raise SystemExit("FAIL: recovers: the thread did not grow its stack and return (exit %#x)" % code)
        code, err = run("recovers", "small")
        if code or "returned 0" not in err:
            raise SystemExit("FAIL: recovers small: with 512-byte frames (under 4 KB of stack left to the handler) the thread "
                             "did not grow its stack and return (exit %#x)" % code)
        for more in ((), ("small",)):
            code, err = run("never", *more)
            if code != STACK_OVERFLOW or OOM_LINE not in err:
                raise SystemExit("FAIL: never%s: wanted the out-of-memory line, then 0xc00000fd (exit %#x)"
                                 % ("".join(" " + m for m in more), code))
        code, err = run("ran-out")
        if code != STACK_OVERFLOW:
            raise SystemExit("FAIL: ran-out: a stack that really ran out did not end with 0xc00000fd (exit %#x)" % code)
    print("PASS: a stack page refused at the commit limit waits for commit and the thread goes on; one that never comes, "
          "or a stack that really ran out, still ends as a stack overflow")


if __name__ == "__main__":
    main()
