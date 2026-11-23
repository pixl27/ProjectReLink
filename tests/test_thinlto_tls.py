"""Production profiler/recorder TLS across translation units, natively and with ThinLTO.

Compiles the real profiler scopes and recorder core (through Enc), not a model of
Arena or its consumers. No game, GPU, Vulkan headers or Vulkan library required.
--source-ref runs this same harness against a git revision, including pre-accessor
main, so a COFF link regression can be compared without changing the checkout.
python -I -B tests/test_thinlto_tls.py [--cxx clang++] [--source-ref <commit>]
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def source(path, ref):
    if ref:
        return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{ref}:{path}"], encoding="utf-8")
    return (ROOT / path).read_text(encoding="utf-8")


def state_header(rec, prof):
    # Keep the complete production core and its real Packet, Enc, fallback and
    # drain implementations; only the Vulkan-dependent routing/packers are out.
    includes = rec[rec.index("#include <algorithm>"):rec.index('#include "vk_loader.h"')]
    core = rec[rec.index("#define RX_REC_CALLS"):rec.index("// ---- The wrapper each vk_work.h macro calls:")]
    scopes = prof[prof.index("// (hookshot) The per-draw phases (PROF_HOT)"):prof.index("} // namespace mkwvk", prof.index("struct ProfThread {"))]
    # Test-only adapter lets the identical consumers exercise the old variable
    # and the new accessor, without rewriting either production implementation.
    arena = "rec::t_arena()" if "inline Arena& t_arena()" in rec else "rec::t_arena"
    return "#pragma once\n#include <cassert>\n" + includes + '''
#include "hookshot_env.h"
#include "vk_new.h"
namespace mkwvk {
extern std::atomic<bool> par_active;
void logf(const char*, ...);
inline std::atomic<int64_t> clock_ticks{0}, prof_ns[2]{};
inline int64_t prof_now() { return clock_ticks.fetch_add(10); }
''' + scopes + "}\n" + core + '''
} // namespace mkwvk::rec
using namespace mkwvk;
struct Owner { rec::Arena* arena; ProfThread* profiler; };
inline rec::Arena& arena() { return ''' + arena + '''; }
Owner from_a();
Owner from_b();
void queue_a(int);
void queue_b(int);
void drain_b();
void record(int);
'''


A = r'''#include "state.h"
[[gnu::noinline]] Owner from_a() {
    ProfScope exact(0);
    return {&arena(), &prof_t};
}
[[gnu::noinline]] void queue_a(int value) {
    rec::Enc encoder;
    const auto call = [value] { record(value); };
    encoder.emit(rec::OP_vkCmdDraw, call, call);
}
'''

B = r'''#include "state.h"
[[gnu::noinline]] Owner from_b() {
    ProfScopeHot hot(1);
    ProfScope exact(0);
    return {&arena(), &prof_t};
}
[[gnu::noinline]] void queue_b(int value) {
    rec::Enc encoder;
    const auto call = [value] { record(value); };
    encoder.emit(rec::OP_vkCmdDraw, call, call);
}
[[gnu::noinline]] void drain_b() { rec::boundary(rec::B_vkEndCommandBuffer); }
'''

DRIVER = r'''#include "state.h"
namespace mkwvk {
std::atomic<bool> par_active{false};
void logf(const char*, ...) {}
}
std::atomic<int> calls{0};
void record(int value) { calls.store(calls.load() * 10 + value); }
int main() {
    prof_full.store(1);
    rec::set_mode(rec::DEFER);
    assert(prof_t.rng == 0 && prof_t.open == 0 && prof_t.k == 0 && prof_t.exact == 0);
    const Owner owner = from_a(), other = from_b();
    assert(owner.arena == other.arena && owner.profiler == other.profiler);
    assert(owner.profiler == &prof_t && prof_t.exact == 20 && prof_t.open == 0);
    assert(owner.arena->tail == &owner.arena->head);
    queue_a(1);
    queue_b(2);
    assert(calls.load() == 0 && owner.arena->packets == 2);
    drain_b();
    assert(calls.load() == 12 && owner.arena->head == nullptr);
    assert(owner.arena->tail == &owner.arena->head && owner.arena->used == 0);
    std::thread worker([owner] {
        assert(prof_t.rng == 0 && prof_t.open == 0 && prof_t.k == 0 && prof_t.exact == 0);
        const Owner mine = from_a(), again = from_b();
        assert(mine.arena == again.arena && mine.profiler == again.profiler);
        assert(mine.arena != owner.arena && mine.profiler != owner.profiler);
        assert(prof_t.exact == 20 && prof_t.open == 0 && mine.arena->tail == &mine.arena->head);
        queue_a(3);
        queue_b(4);
        assert(calls.load() == 12 && mine.arena->head && mine.arena->packets == 2);
        // No explicit drain: the real Arena destructor runs both Enc packets.
    });
    worker.join();
    assert(calls.load() == 1234);
    assert(prof_t.exact == 20 && prof_t.open == 0 && owner.arena->head == nullptr);
    assert(owner.arena->tail == &owner.arena->head && owner.arena->packets == 2);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++", help="native Clang with LLD and ThinLTO support")
    parser.add_argument("--source-ref", help="read production sources from this git revision instead of the working tree")
    args = parser.parse_args()
    # (hookshot) the run_all's {cxx} is GCC where no Clang is installed: ThinLTO is Clang's, so nothing to check there
    try:
        probe = subprocess.run([args.cxx, "--version"], capture_output=True, text=True)
    except OSError:
        probe = None
    if probe is None or probe.returncode != 0 or "clang" not in probe.stdout.lower():
        print("SKIP: %s is not Clang (ThinLTO, -flto=thin, is Clang's)" % args.cxx)
        return
    header = state_header(source("gpu/vk/vk_recorder.h", args.source_ref), source("gpu/vk/vk_internal.h", args.source_ref))
    with tempfile.TemporaryDirectory(prefix="hookshot-thinlto-tls-") as directory:
        path = Path(directory)
        (path / "state.h").write_text(header, encoding="utf-8")
        (path / "hookshot_env.h").write_text(source("gpu/vk/compat/hookshot_env.h", args.source_ref), encoding="utf-8")
        (path / "vk_new.h").write_text(source("gpu/vk/vk_new.h", args.source_ref), encoding="utf-8")
        for name, text in (("a.cpp", A), ("b.cpp", B), ("main.cpp", DRIVER)):
            (path / name).write_text(text, encoding="utf-8")
        for mode, flags in (("native", []), ("thin", ["-flto=thin", "-fuse-ld=lld"])):
            print(f"---- {mode}: {args.source_ref or 'working tree'}", flush=True)
            common = [args.cxx, "-std=c++20", "-O2", "-pthread", *flags]
            objects = []
            for name in ("a", "b", "main"):
                obj = path / f"{mode}-{name}.o"
                subprocess.run([*common, "-c", str(path / f"{name}.cpp"), "-o", str(obj)], check=True)
                objects.append(str(obj))
            exe = path / (mode + ".exe")
            subprocess.run([*common, *(["-static"] if os.name == "nt" else []), *objects, "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True, timeout=15)
    print("PASS: native/ThinLTO production cross-TU TLS identity, thread isolation and Enc exit drain")


if __name__ == "__main__":
    main()
