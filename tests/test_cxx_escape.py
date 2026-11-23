"""A C++ exception that leaves the renderer. Nothing above hookshot_vk.dll catches one: the host is C, and a renderer
thread's body is the top of its stack. A std::length_error, a std::system_error from a thread that could not be started or a
std::bad_alloc then ended the game with exception 0x20474343 ("An unexpected error"), saying nothing of what was thrown; v3.0.0
still did after the renderer's new_handler. Now every function the renderer gives the host (gpu/vk/renderer_guard.h),
mkwvk_init, and the bodies of the replay's threads catch a std::exception and stop the game with its type and what() in the log
(gpu/vk/cxx_escape.h escaped): out of memory through the host's line and notice for std::bad_alloc and a thread that could not be
started, the renderer's FATAL line for any other. The texture decode's helper threads go through fork_join, where a helper that
cannot be started leaves its part to the caller (before, std::terminate).

Builds tests/cxx_escape_test.cpp with the real headers and the real MkwVkRenderer table behind guard_renderer, and runs each case
in its own process, the hooks printing what the game would say:

- pass: a call that throws nothing gets its arguments and result through, nothing said, a null member left null, no double wrap;
- entry-*: std::bad_alloc out of option or nvn_submit: out of memory naming the function; std::length_error (a vector's reserve
  past any PC's memory) and std::bad_array_new_length: FATAL, a size gone wrong; std::out_of_range (.at): FATAL with its what();
  a std::system_error of no resources: out of memory; another std::system_error: FATAL;
- thread, thread-memory: an exception out of a spawn_needed thread's body (the replay, the presenter) stops the game from there;
- fork-join, fork-join-throws, fork-join-caller-throws: every part made once and the helpers' start run; a helper's exception
  stops the game; part(0)'s comes out once the helpers are joined;
- fork-join-no-threads, spawn-no-threads (glibc: a default stack past the address space, so pthread_create says EAGAIN, as at the
  commit limit): the caller makes every part, said once; spawn says false; spawn_needed stops as out of memory.

And by text: renderer_guard.h guards every function member of MkwVkRenderer (vk_api.h); mkwvk_init catches around init_body,
which sets the hooks once it has the host and guards the table after fill_replay_table; the replay, presenter and order threads
are started through spawn_needed; vk_texture.cpp's helper threads all go through fork_join; and cxx_escape.h has no catch (...)
(a foreign exception, an access violation or a driver's 0xe06d7363, stays the host crash record's).

On main this fails: there is no gpu/vk/cxx_escape.h, and nothing guards the table.

python -I -B tests/test_cxx_escape.py [--cxx g++]
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu/vk"

# (case: exit code, lines the output must have)
CASES = {
    "pass": (0, ["PASS"]),
    "entry-bad-alloc": (4, ["LOG vk: a C++ exception came out of option: std::bad_alloc: ",
                            "OOM the renderer's C++ allocations (std::bad_alloc out of option)"]),
    "entry-submit": (4, ["OOM the renderer's C++ allocations (std::bad_alloc out of nvn_submit)"]),
    "entry-length": (3, ["LOG vk: a C++ exception came out of option: std::length_error: ",
                         "FATAL std::length_error came out of option: ", "(a size past any PC's memory: one gone wrong)"]),
    "entry-array-length": (3, ["FATAL std::bad_array_new_length came out of nvn_submit: "]),
    "entry-range": (3, ["FATAL std::out_of_range came out of option: "]),
    "entry-system-memory": (4, ["OOM the renderer's C++ allocations (std::system_error out of option)"]),
    "entry-system-other": (3, ["FATAL std::system_error came out of option: a lock"]),
    "thread": (3, ["LOG vk: a C++ exception came out of replay: std::out_of_range: ", "FATAL std::out_of_range came out of replay"]),
    "thread-memory": (4, ["OOM the renderer's C++ allocations (std::bad_alloc out of presenter)"]),
    "fork-join": (0, ["PASS"]),
    "fork-join-throws": (3, ["FATAL std::out_of_range came out of texture unswizzle: part 2"]),
    "fork-join-caller-throws": (0, ["PASS"]),
    "fork-join-no-threads": (0, ["LOG vk: the texture gather thread could not be started (", "PASS"]),
    "spawn-no-threads": (4, ["LOG vk: the upscaler probe thread could not be started (", "LOG vk: the replay thread could not",
                             "OOM the renderer's threads (the replay thread)"]),
}


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def renderer_members(api):
    """The function members of MkwVkRenderer, in vk_api.h's order"""
    body = api[api.index("typedef struct MkwVkRenderer {"):api.index("} MkwVkRenderer;")]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    members = []
    for statement in body.split(";"):   # (the first (*name) of each: a callback's own parameter, shot's done, is not a member)
        m = re.search(r"\(\s*\*\s*(\w+)\s*\)\s*\(", statement)
        if m:
            members.append(m.group(1))
    return members


def check_sites():
    problems = []
    if not (VK / "cxx_escape.h").exists():
        return ["gpu/vk/cxx_escape.h is missing: a C++ exception out of the renderer ends the game with 0x20474343, unsaid"]
    members = renderer_members(read("gpu/vk/vk_api.h"))
    if len(members) < 30:
        problems.append("vk_api.h: MkwVkRenderer's members not found (%d)" % len(members))
    guard = read("gpu/vk/renderer_guard.h")
    guarded = re.findall(r"^\s*MKWVK_GUARD\((\w+)\);", guard, re.M)
    for m in members:
        if m not in guarded:
            problems.append("renderer_guard.h: MkwVkRenderer::%s is not guarded: an exception out of it goes on into the host's C"
                            % m)
    for m in guarded:
        if m not in members:
            problems.append("renderer_guard.h guards %s, which MkwVkRenderer does not have" % m)
    main = read("gpu/vk/vk_main.cpp")
    init = block(main, "const MkwVkRenderer* mkwvk_init(const MkwVkHost* h) {")
    if not re.search(r"try \{\s*return init_body\(h\);\s*\} catch \(const std::exception& e\) \{\s*"
                     r"mkwvk::cxx_escape::escaped\(\"mkwvk_init\", e\);", init):
        problems.append("vk_main.cpp: mkwvk_init does not catch around init_body")
    body = block(main, "static const MkwVkRenderer* init_body(const MkwVkHost* h) {")
    for needle in ("cxx_escape::g_hooks.log =", "cxx_escape::g_hooks.out_of_memory =", "cxx_escape::g_hooks.fatal =",
                   'logf("vk: FATAL: %s", why);', "std::_Exit(3);", "guard_renderer(table);"):
        if needle not in body:
            problems.append("vk_main.cpp init_body: no " + needle)
    if "guard_renderer(table);" in body:
        if body.index("host = h;") > body.index("cxx_escape::g_hooks.out_of_memory ="):
            problems.append("vk_main.cpp init_body: the hooks are set before the host is kept")
        if body.index("fill_replay_table(table);") > body.index("guard_renderer(table);"):
            problems.append("vk_main.cpp init_body: the table is guarded before it is filled")
    if "host->out_of_memory(what)" not in body[body.index("cxx_escape::g_hooks.out_of_memory ="):]:
        problems.append("vk_main.cpp init_body: the out-of-memory hook is not the host's out_of_memory")
    replay = read("gpu/vk/vk_replay.cpp")
    for name, fn in (("presenter", "presenter_loop"), ("replay", "replay_thread_main"), ("order", "sync_watch")):
        if '::mkwvk::cxx_escape::spawn_needed("%s", %s);' % (name, fn) not in replay:
            problems.append("vk_replay.cpp: the %s thread (%s) is not started through spawn_needed" % (name, fn))
        if "std::thread(%s)" % fn in replay:
            problems.append("vk_replay.cpp: %s is still started as a bare std::thread" % fn)
    texture = read("gpu/vk/vk_texture.cpp")
    if re.search(r"\.emplace_back\(\[&part", texture):
        problems.append("vk_texture.cpp: a helper thread started outside fork_join (one refused at the commit limit: terminate)")
    if texture.count("::mkwvk::cxx_escape::fork_join(") != 3:
        problems.append("vk_texture.cpp: not three fork_joins (the gather, the check, the unswizzle)")
    escape = read("gpu/vk/cxx_escape.h")
    if re.search(r"catch\s*\(\s*\.\.\.\s*\)", re.sub(r"//[^\n]*", "", escape)):
        problems.append("cxx_escape.h has a catch (...): it would take a foreign exception from the host's crash record")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    problems = check_sites()
    if problems:
        raise SystemExit("FAIL:\n  " + "\n  ".join(problems))
    cxx = shutil.which(args.cxx)
    if not cxx:
        raise SystemExit("test_cxx_escape: no C++ compiler %r (pass --cxx)" % args.cxx)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    with tempfile.TemporaryDirectory(prefix="cxx-escape-") as temporary:
        executable = Path(temporary) / "fixture.exe"
        subprocess.run([cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(ROOT),
                        str(ROOT / "tests/cxx_escape_test.cpp"), "-o", str(executable)], check=True, creationflags=flags)
        for case, (code, want) in CASES.items():
            result = subprocess.run([str(executable), case], capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=60, creationflags=flags)
            out = result.stdout
            print("%s: exit %d" % (case, result.returncode))
            if "SKIP " in out and result.returncode == 0:
                print("  " + out.strip())
                continue
            missing = [w for w in want if w not in out]
            if result.returncode != code or missing or "FAIL" in out:
                print(out, end="")
                print(result.stderr, end="", file=sys.stderr)
                raise SystemExit("FAIL: case %s: exit %d (want %d)%s" % (case, result.returncode, code,
                                 "; missing " + repr(missing) if missing else ""))
    print("PASS: a C++ exception out of the renderer is said and stops the game as out of memory or a FATAL line, not 0x20474343")


if __name__ == "__main__":
    main()
