"""One rule for on/off switches: env_on (gpu/vk/compat/hookshot_env.h) in C and C++, kit.env_on in Python.

Unset, empty, 0, false, off and no (any case, spaces around them not counted) are off; 1 and anything else are on, so
RX_VK_NO_PROGRAM_CACHE=0 no longer turns the feature on. env_int reads a number with a default. Compile the header as C
and as C++ and check both, and the Python twin, against the same table. Also check that no file in host/ or gpu/vk reads
an on/off switch the old way, getenv(name) != NULL, for a name it reads nowhere else for a value. No game or GPU is needed.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import kit  # noqa: E402

CASES = [(None, 0), ("", 0), ("0", 0), (" 0 ", 0), ("false", 0), ("FALSE", 0), ("Off", 0), ("no", 0), ("0\r\n", 0),
         ("1", 1), ("2", 1), ("yes", 1), ("on", 1), ("true", 1), ("00", 1), ("2c56dc1006942f74", 1), ("dir/x", 1)]

FIXTURE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hookshot_env.h"
#ifdef _WIN32
static void put(const char* name, const char* value) {
    char s[256];
    snprintf(s, sizeof s, "%s=%s", name, value ? value : "");
    _putenv(s);
}
#else
static void put(const char* name, const char* value) {
    if (value) setenv(name, value, 1);
    else unsetenv(name);
}
#endif
int main(int argc, char** argv) {
    /* argv: [value|-unset] want ... */
    int i;
    for (i = 1; i + 1 < argc; i += 2) {
        const char* v = strcmp(argv[i], "-unset") ? argv[i] : NULL;
        put("HOOKSHOT_T_SWITCH", v);
        if (env_on("HOOKSHOT_T_SWITCH") != atoi(argv[i + 1])) {
            fprintf(stderr, "FAIL: env_on with [%s] is %d\n", v ? v : "(unset)", env_on("HOOKSHOT_T_SWITCH"));
            return 1;
        }
    }
    /* env_on_or: the default only when unset; set, as env_on reads it (empty: off) */
    put("HOOKSHOT_T_SWITCH", NULL);
    if (!env_on_or("HOOKSHOT_T_SWITCH", 1) || env_on_or("HOOKSHOT_T_SWITCH", 0)) return 5;
    put("HOOKSHOT_T_SWITCH", "false");
    if (env_on_or("HOOKSHOT_T_SWITCH", 1)) return 6;
    put("HOOKSHOT_T_SWITCH", "yes");
    if (!env_on_or("HOOKSHOT_T_SWITCH", 0)) return 7;
#ifndef _WIN32   /* (Windows has no variable that is set and empty) */
    put("HOOKSHOT_T_SWITCH", "");
    if (env_on_or("HOOKSHOT_T_SWITCH", 1)) return 8;
#endif
    put("HOOKSHOT_T_NUMBER", "12");
    if (env_int("HOOKSHOT_T_NUMBER", 5) != 12) return 2;
    put("HOOKSHOT_T_NUMBER", "x");
    if (env_int("HOOKSHOT_T_NUMBER", 5) != 5) return 3;
    put("HOOKSHOT_T_NUMBER", NULL);
    if (env_int("HOOKSHOT_T_NUMBER", 5) != 5) return 4;
    puts("PASS");
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()

    for value, want in CASES:   # (Python's twin)
        env = os.environ.copy()
        os.environ.pop("HOOKSHOT_T_SWITCH", None)
        if value is not None:
            os.environ["HOOKSHOT_T_SWITCH"] = value
        got = kit.env_on("HOOKSHOT_T_SWITCH")
        os.environ.clear(), os.environ.update(env)
        if got != bool(want):
            raise SystemExit("FAIL: kit.env_on with [%r] is %s" % (value, got))

    argv = []
    for value, want in CASES:
        if value is not None and ("\r" in value or "\n" in value):
            continue   # (not passed through a command line)
        argv += ["-unset" if value is None else value, str(want)]
    include = ROOT / "gpu" / "vk" / "compat"
    with tempfile.TemporaryDirectory(prefix="env-on-") as tmp:
        for lang, suffix, std in (("C", ".c", "-std=c11"), ("C++", ".cpp", "-std=c++20")):
            src = Path(tmp) / ("fixture" + suffix)
            exe = Path(tmp) / ("fixture_%s%s" % (suffix[1:], ".exe" if os.name == "nt" else ""))
            src.write_text(FIXTURE, encoding="utf-8")
            flags = ["-x", "c", "-D_POSIX_C_SOURCE=200809L"] if lang == "C" else []   # (C++'s compiler, told it is C)
            # (_CRT_SECURE_NO_WARNINGS: the fixture's stdio.h comes before hookshot_env.h, and Windows' CRT then calls getenv
            # deprecated for the rest of the file)
            subprocess.run([args.cxx, std, "-Wall", "-Wextra", "-Werror", "-D_CRT_SECURE_NO_WARNINGS", *flags, "-I", str(include),
                            str(src), "-o", str(exe)], check=True)
            subprocess.run([str(exe), *argv], check=True)

    # (no on/off switch read the old way, set-means-on: a game switch (RX_, HOS_, HOOKSHOT_, EDEN_, NVN_) whose every read in
    # host/ or gpu/ only asks whether it is set: getenv(name) != NULL / == NULL, !getenv(name), getenv(name) followed by ?,
    # && or ||, or alone between ( or && / || and ) or && / || or the line's end. A line that reads the name twice reads its
    # value too (getenv(x) ? atoi(getenv(x)) : ...), as does one that also asks env_on(x) (getenv(x) ? env_on(x) : a default),
    # and a getenv inside a call (atoi(getenv(x)), parse(getenv(x))) is a value)
    name = r'(?:std::)?getenv\("((?:RX|HOS|HOOKSHOT|EDEN|NVN)_[A-Z0-9_]+)"\)'
    forms = [re.compile(name + r"\s*[!=]=\s*(?:nullptr|NULL)"), re.compile(r"!\s*" + name),
             re.compile(name + r"\s*(?:\?|&&|\|\|)"),
             re.compile(r"(?:(?<![A-Za-z0-9_])\(|(?<=\bif)\(|(?<=while)\(|&&|\|\|)\s*" + name + r"\s*(?:\)|&&|\|\||$)")]
    # (asked only whether it is set, on purpose, where the value is read by the rule elsewhere)
    set_at_all = {"HOS_FULLSCREEN": "gpu/vk/vk_menu.cpp: a run that names it at all keeps the menu's Fullscreen row out of it"}
    reads, presence = {}, {}
    for path in sorted(p for d in ("host", "gpu") for p in (ROOT / d).rglob("*") if p.suffix in (".c", ".cpp", ".h")):
        rel = path.relative_to(ROOT).as_posix()
        if "gpu/render" in rel or "third_party" in rel or "/tests/" in rel or "sirit" in rel:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
            if line.strip().startswith(("//", "/*", "*")):
                continue
            for m in re.finditer(name, line):
                n = m.group(1)
                reads[n] = reads.get(n, 0) + 1
                if line.count('getenv("%s")' % n) == 1 and 'env_on("%s")' % n not in line and \
                        any(f.group(1) == n for form in forms for f in form.finditer(line)):
                    presence.setdefault(n, []).append("%s:%d" % (rel, number))
    old = sorted(n for n, where in presence.items() if len(where) == reads.get(n, 0) and n not in set_at_all)
    if old:
        raise SystemExit("FAIL: on/off switches still read as set-means-on (set to 0 they are on): use env_on:\n  " +
                         "\n  ".join("%s (%s)" % (n, ", ".join(presence[n])) for n in old))
    # (no on/off switch of host/ read as a number: atoi says off for true, yes and on. Seen as atoi(getenv(name)), and as
    # atoi(e) within a few lines of e = getenv(name). A switch that is a number says so here)
    numbers = {"HOS_CORES", "HOS_PROFILE", "HOS_COND_RECORD", "HOS_FPS_TARGET", "HOS_REALTIME_WINDOW", "HOS_FRAME_SLEEP_MS",
               "HOS_MKW_INTERVAL", "RX_NVN_WATCH_AFTER", "HOS_MUTEX_TRACE", "HOS_FPS_MAX", "HOS_AUDIO_CHANNELS",
               "HOS_AUDIO_DEVICE", "HOS_CRASH_TEST", "HOS_LOG_BATCH_MS", "HOS_NVN_REC_SHIFT", "HOS_NVN_WATCH_DRAW",
               "HOS_SDL_VERSION", "HOS_STATS_AFTER", "HOS_TEXT_SIZE", "HOS_THREAD_TIMES", "HOS_VIDEO_INPUT_LIMIT",
               "HOS_WIDE_UI_LOG", "HOS_PROFILE_EVERY", "HOS_SHADOW_BUCKET"}
    switch = r'"((?:RX|HOS|HOOKSHOT|EDEN|NVN)_[A-Z0-9_]+)"'
    direct = re.compile(r"atoi\(getenv\(" + switch + r"\)\)")
    taken = re.compile(r"\b(\w+)\s*=\s*getenv\(" + switch + r"\)")
    atoi_read = set()
    for path in sorted((ROOT / "host").rglob("*.c")):
        if "win32_posix" in path.name:
            continue
        rel = path.relative_to(ROOT).as_posix()
        held = {}   # (a variable: the switch it was last given, and the line)
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
            found = [m.group(1) for m in direct.finditer(line)]
            for m in taken.finditer(line):
                held[m.group(1)] = (m.group(2), number)
            for var, (n, at) in held.items():
                if number - at <= 6 and re.search(r"atoi\(%s\b" % re.escape(var), line):
                    found.append(n)
            atoi_read.update("%s (%s:%d)" % (n, rel, number) for n in found if n not in numbers)
    if atoi_read:
        raise SystemExit("FAIL: switches of host/ read with atoi: an on/off switch reads env_on, env_on_or or env_value_off, a "
                         "number env_int or a place in this check's list:\n  " + "\n  ".join(sorted(atoi_read)))
    print("PASS: env_on's rule in C, C++ and Python (%d values); no on/off switch read the old way" % len(CASES))


if __name__ == "__main__":
    main()
