"""HOOKSHOT_TRACY's zones in the lifted game code (nx2/tracy_funcs.py), without the game and without Tracy.

The list is read as written (addresses with or without f_ or 0x, `calls`, names, comments, a later line replacing an earlier
one, a bad line refused). Then the mechanism itself on two small C files shaped as lifted chunks are: one defines f_1000 and
calls it, the other calls it too. With the defining object patched and a wrapper named f_1000 linked in, every call (the
defining object's own, the other object's and one through a function pointer) reaches the wrapper once and the function's code
once; without the patch the same program never reaches a wrapper. And the header for `calls` turns an A64_ICALL whose return
address lies in a listed function into a call the hook sees, and leaves one outside it alone.

    python -I -B tests/test_tracy_funcs.py [--cc cc]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ALL_SKIP = {"windows": "the listed functions' zones patch ELF objects: Linux"}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import tracy_funcs  # noqa: E402

CHUNK_A = """
typedef struct Ctx Ctx;
extern int ran, wrapped;
void f_1000(Ctx *c);
void f_2000(Ctx *c);
void f_1000(Ctx *__restrict c) { ran++; }
void f_2000(Ctx *__restrict c) { f_1000(c); }
"""
CHUNK_B = """
typedef struct Ctx Ctx;
void f_1000(Ctx *c);
void f_3000(Ctx *__restrict c) { f_1000(c); }
void (*table[])(Ctx *) = { f_1000 };
"""
WRAPPER = """
typedef struct Ctx Ctx;
int ran, wrapped;
void __real_f_1000(Ctx *c);
void f_1000(Ctx *c) { wrapped++; __real_f_1000(c); }
"""
MAIN = """
#include <stdio.h>
typedef struct Ctx Ctx;
extern int ran, wrapped;
void f_2000(Ctx *c); void f_3000(Ctx *c);
extern void (*table[])(Ctx *);
int main(void) { f_2000(0); f_3000(0); table[0](0); printf("%d %d\\n", ran, wrapped); return 0; }
"""
PLAIN_GLOBALS = "int ran, wrapped;\n"
CALLS = """
#include <stdint.h>
#include <stdio.h>
typedef struct Ctx { uint64_t x[32]; } Ctx;
static int zones, ended, calls;
#define A64_SITE_CALL(c, t, slow) do { calls++; } while (0)
#define A64_ICALL(c, t) A64_SITE_CALL(c, t, a64_icall_site)
%s
HtCallZone hos_tracy_call_begin(Ctx *c, uint64_t target) { HtCallZone z = { { 1, 1 } }; (void)c; (void)target; zones++; return z; }
void hos_tracy_call_end(HtCallZone *z) { ended += (int)z->w[1]; }
int main(void) {
    Ctx ctx, *c = &ctx;
    c->x[30] = 0x8000100cULL; A64_ICALL(c, 0x9000);   /* (inside the listed function at 0x80001000, 0x40 long) */
    c->x[30] = 0x80001040ULL; A64_ICALL(c, 0x9000);   /* (the function after it) */
    c->x[30] = 0x80000ffcULL; A64_ICALL(c, 0x9000);   /* (the one before) */
    printf("%%d %%d %%d\\n", zones, ended, calls);
    return 0;
}
"""


def run(cc, folder, sources, name):
    exe = folder / name
    subprocess.run([cc, "-O1", "-w", *map(str, sources), "-o", str(exe)], check=True)
    return subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", default="cc")
    cc = p.parse_args().cc
    failures = []
    with tempfile.TemporaryDirectory(prefix="tracy-funcs-") as tmp:
        folder = Path(tmp)
        # the list
        a, b = folder / "a.txt", folder / "b.txt"
        a.write_text("# a comment\n80001000 calls  job loop\nf_80002000\n0x80003000   a \"name\"\n", encoding="utf-8")
        b.write_text("80002000 calls again   # later wins\n", encoding="utf-8")
        got = tracy_funcs.read_list([str(a), str(b)])
        want = [(0x80001000, True, "job loop"), (0x80002000, True, "again"), (0x80003000, False, "a 'name'")]
        if got != want:
            failures.append("the list reads as %r, not %r" % (got, want))
        b.write_text("not an address\n", encoding="utf-8")
        try:
            tracy_funcs.read_list([str(b)])
            failures.append("a line that is no address was taken")
        except SystemExit:
            pass
        if tracy_funcs.read_list([tracy_funcs.SEED]) == []:
            failures.append("nx2/tracy_funcs.txt lists nothing")
        # where a function is defined, and the wrappers' text
        (folder / "code_000.c").write_text(CHUNK_A, encoding="utf-8")
        (folder / "code_001.c").write_text(CHUNK_B, encoding="utf-8")
        where = tracy_funcs.defined_in([str(folder / "code_000.c"), str(folder / "code_001.c")], ["f_1000", "f_3000", "f_4000"])
        if where != {"f_1000": str(folder / "code_000.c"), "f_3000": str(folder / "code_001.c")}:
            failures.append("defined_in found %r" % where)
        text = tracy_funcs.wrappers([(0x80001000, True, "job loop")])
        if 'void f_80001000(Ctx *__restrict A64_NOESC c) {' not in text or '{ "f_80001000 job loop", "f_80001000",' not in text \
                or "hos_tracy_fn_begin(&loc, c, 0x80001000ULL); __real_f_80001000(c); }" not in text:
            failures.append("the wrapper's text is not a zone around __real_f_<address>:\n" + text)
        # the object patch
        for n, src in (("wrapper.c", WRAPPER), ("main.c", MAIN), ("plain.c", PLAIN_GLOBALS)):
            (folder / n).write_text(src, encoding="utf-8")
        for n in ("code_000", "code_001"):
            subprocess.run([cc, "-O1", "-w", "-ffunction-sections", "-fPIE", "-c", str(folder / (n + ".c")), "-o", str(folder / (n + ".o"))], check=True)
        done = tracy_funcs.patch_object(str(folder / "code_000.o"), str(folder / "patched.o"), ["f_1000", "f_9999"])
        if done != ["f_1000"]:
            failures.append("patch_object renamed %r, not f_1000 alone" % done)
        plain = run(cc, folder, [folder / "code_000.o", folder / "code_001.o", folder / "plain.c", folder / "main.c"], "plain")
        if plain != ["3", "0"]:
            failures.append("unpatched, the program printed %r, not 3 runs and no wrapper" % plain)
        patched = run(cc, folder, [folder / "patched.o", folder / "code_001.o", folder / "wrapper.c", folder / "main.c"], "patched")
        if patched != ["3", "3"]:
            failures.append("patched, the three calls (same object, another object, a pointer) gave %r runs and wrapper calls, "
                            "not 3 and 3" % patched)
        # the header for `calls`
        header = tracy_funcs.calls_header([(0x80001000, 0x40)])
        body = "\n".join(l for l in header.splitlines() if not l.startswith("#include"))
        (folder / "calls.c").write_text(CALLS % body, encoding="utf-8")
        calls = run(cc, folder, [folder / "calls.c"], "calls")
        if calls != ["1", "1", "3"]:
            failures.append("the calls header made %r zones, ends and calls, not 1 zone begun and ended (the call inside the "
                            "listed function) and 3 calls" % calls)
        if "tracy/" in header or "hos_tracy.h" in header:
            failures.append("the calls header brings Tracy's headers into the lifted chunk")
    # no other build: the option's parts are behind HOOKSHOT_TRACY in nx2/build.py, and the zones behind TRACY_ENABLE
    build = (ROOT / "nx2" / "build.py").read_text(encoding="utf-8")
    for needle in ('TRACY = os.environ.get("HOOKSHOT_TRACY", "") == "1"', '(f != "hos_tracy_game.c" or TRACY)',
                   "if TRACY and LINUX and host_srcs", "if TRACY and host_srcs:"):
        if needle not in build:
            failures.append("nx2/build.py no longer has %r: is the option still off by default?" % needle)
    for rel in ("host/hos_thread.c", "host/nn_os_sync.c", "host/nvn_native.c", "gpu/vk/vk_replay.cpp", "gpu/vk/vk_draw.cpp",
                "gpu/vk/vk_memory.cpp", "gpu/vk/vk_texture.cpp", "gpu/vk/vk_jobs.cpp", "gpu/vk/vk_recorder.h"):
        guarded, bare = False, []
        for n, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith("#ifdef TRACY_ENABLE"):
                guarded = True
            elif s.startswith("#endif"):
                guarded = False
            elif not guarded and ("HT_" in s.split("//")[0].split("/*")[0]) and "HT_" in line:
                bare.append(n)
        if bare:
            failures.append("%s uses a zone macro outside #ifdef TRACY_ENABLE (lines %s): a build without Tracy, and the checks "
                            "that compile a function cut from the file, must not see it" % (rel, bare[:8]))
    for f in failures:
        print("FAIL: " + f)
    if failures:
        sys.exit(1)
    print("ok: the list, the wrappers, the object patch (3 calls, 3 zones), the calls header, and the zones behind TRACY_ENABLE")


if __name__ == "__main__":
    main()
