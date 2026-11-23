"""(hookshot) The A/B within one run (host/hos_ab.c, HOS_AB) and its summary (scripts/ab_summary.py), game-free.

The real host/hos_ab.c is compiled over a stub hos.h (the real one's A/B declarations, a clock the fixture moves, the log kept
in memory) and a stub windows.h (no thread is started: the fixture calls hos_ab_tick at the times it chooses), with the wired
reader cut from its source (hos_thread.c's os_yield: HOS_OS_YIELD).
It checks:
- the parser: HOS_AB=<switch>=<value>[,<seconds>[,<warm-up seconds>]], the defaults (3 s, 20 s), and bad values (no =, a
  name in lower case or too long, a value that is no whole number such as check, a period of 0 or over an hour, a negative
  warm-up, a fourth field) each refused with a log line, no A/B and no crash;
- the schedule on the fake clock, every 10 ms over a minute: -1 through the warm-up, then side 0 first and the sides
  alternating every period; hos_ab_tick's next change;
- hos_ab_frame: -1 in the warm-up, a frame across the warm-up's end or a flip, or ending less than 0.3 s after one, a
  transition (side + 2), a frame from an unknown start a transition, a 4 s stall over a whole period a transition;
- the cheap path: with no A/B and in the warm-up, hos_ab_value answers the normal value without looking at a name (the table
  of names seen stays empty); during the A/B both sides look the name up alike, and only the named switch moves (40 names
  that share the table's slots, copies of the name at other addresses);
- the wired switch flips with the side: HOS_OS_YIELD (1 and 0), both ways round (the switch set in the environment and
  an A/B value of 1);
- the host's sums: frames of 16 ms on side 0 and 18 ms on side 1 give exactly those means and +12.50 %, transitions counted
  apart, the end line written once, and its note when no reader named the switch.
Then broken variants of host/hos_ab.c must fail: the warm-up ignored, the sides swapped, no settle time after a flip, a frame
across a flip not a transition, the name not checked, a period of 0 accepted, a value that is no number accepted.

The places of the calls (hos_ab_start before the game's threads, hos_ab_present in rx_nvn_present, hos_ab_end in
hos_close_finish, hos_ab_frame last in vk_host) and the list of wired switches are read from the source.

scripts/ab_summary.py on made-up traces: known means per side (16 and 18 ms), the warm-up's and the transitions' frames (50 and
100 ms) left out, a last period cut short left out and one more for the same side at both ends, two files together, and a
steady drift that moves both sides alike giving no difference. Its broken variants (transitions kept, sides swapped, the
warm-up counted, the cut-short period kept, the parity trim gone) must fail.

python -I -B tests/test_hos_ab.py [--cc clang]
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

RUN_ALL_ARGS = ["--cc", "clang"]

ROOT = Path(__file__).resolve().parents[1]


def fail(what):
    raise SystemExit("FAIL: " + what)


def block(text, head):
    """The source text of the function starting at head, braces balanced"""
    if head not in text:
        fail("no '%s' in the source: update the test" % head)
    at = text.index(head)
    depth, i = 0, text.index("{", at)
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[at:j + 1]
    fail("unbalanced: " + head)


def between(text, first, last_line_has):
    """From first to the end of the first line after it that holds last_line_has"""
    if first not in text:
        fail("no '%s' in the source: update the test" % first)
    start = text.index(first)
    end = text.index(last_line_has, start)
    return text[start:text.index("\n", end) + 1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STUB_WINDOWS = r"""#pragma once
#include <stddef.h>
#include <stdint.h>
typedef uint32_t DWORD;
typedef void *HANDLE;
#define WINAPI
typedef DWORD (*LPTHREAD_START_ROUTINE)(void *);
static int threads_started;
static HANDLE CreateThread(void *attr, size_t stack, LPTHREAD_START_ROUTINE fn, void *arg, DWORD flags, DWORD *id) {
    (void)attr, (void)stack, (void)fn, (void)arg, (void)flags, (void)id;
    threads_started++;
    return (HANDLE)1;
}
static void Sleep(DWORD ms) { (void)ms; }
"""

STUB_HOS = r"""#pragma once
#include <stddef.h>
#include <stdint.h>
enum { HOS_LOG_ERROR, HOS_LOG_WARN, HOS_LOG_INFO, HOS_LOG_DEBUG, HOS_LOG_TRACE };
void hos_log(const char *fmt, ...);
void hos_log_at(int level, const char *fmt, ...);
int64_t hos_now_ns(void);
/* (host/hos.h's own A/B declarations) */
@AB_DECLS@
"""

FIXTURE = r"""#define _CRT_SECURE_NO_WARNINGS
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hookshot_env.h"
#include "hos.h"
#include "hos_ab.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL:%d %s (%s)\n", __LINE__, #x, step); exit(1); } } while (0)
static const char *step = "start";
static char logbuf[1 << 16];
static size_t loglen, last_at;   /* (last_at: where the last line starts) */
static int loglines;
static void vlog(const char *fmt, va_list a) {
    last_at = loglen;
    if (loglen < sizeof logbuf - 1) loglen += (size_t)vsnprintf(logbuf + loglen, sizeof logbuf - loglen - 1, fmt, a);
    if (loglen > sizeof logbuf - 2) loglen = sizeof logbuf - 2;
    logbuf[loglen++] = '\n', logbuf[loglen] = 0;
    loglines++;
}
void hos_log(const char *fmt, ...) { va_list a; va_start(a, fmt); vlog(fmt, a); va_end(a); }
void hos_log_at(int level, const char *fmt, ...) { (void)level; va_list a; va_start(a, fmt); vlog(fmt, a); va_end(a); }
static int64_t fake_now;
int64_t hos_now_ns(void) { return fake_now; }
void hos_host_thread_register_self(const char *name) { (void)name; }

/* ---- the wired readers, cut from their sources */
/* (hookshot: HOS_OS_YIELD, host/hos_thread.c; the wide walk of the write marks it was shown with upstream is gone here) */
@OS_YIELD@
#define T0 5000000000LL
#define S(x) ((int64_t)((x) * 1e9 + 0.5))
static void ab_reset(void) {
    memset(&ab, 0, sizeof ab), memset(ab_seen, 0, sizeof ab_seen), memset(ab_frames, 0, sizeof ab_frames);
    memset(ab_ns, 0, sizeof ab_ns);
    ab_read = 0, ab_last_present = 0, ab_ended = 0, hos_ab_side_now = -1;
}
static int seen_any(void) {
    for (int i = 0; i < 16; i++)
        if (ab_seen[i]) return 1;
    return 0;
}
static void at(double s) { fake_now = T0 + S(s), hos_ab_tick(fake_now); }
static int expected_side(double s, double warm, double period) {
    if (s < warm - 1e-9) return -1;
    return (int)((s - warm + 1e-9) / period) & 1;
}

static void parser(void) {
    step = "the parser";
    AbPlan p;
    CHECK(ab_parse("HOS_OS_YIELD=0", &p) && !strcmp(p.name, "HOS_OS_YIELD") && p.value == 0);
    CHECK(p.period == S(3) && p.warm == S(20));
    CHECK(ab_parse("HOS_X2_Y=-2,2.5,0", &p) && p.value == -2 && p.period == S(2.5) && p.warm == 0);
    CHECK(ab_parse("HOS_X1=0,1,5", &p) && p.value == 0 && p.period == S(1) && p.warm == S(5));
    static const char *const bad[] = {
        "HOS_OS_YIELD", "=1", "hos_x=1", "HOS-X=1", "1HOS=1", "HOS_X=", "HOS_X=check", "HOS_X=1.5", "HOS_X=1 ", "HOS_X=0x10",
        "HOS_X=99999999999", "HOS_X=1,", "HOS_X=1,0", "HOS_X=1,0.5", "HOS_X=1,3601", "HOS_X=1,abc", "HOS_X=1,nan", "HOS_X=1,3,-1",
        "HOS_X=1,3,", "HOS_X=1,3,20,4", "HOS_X=1,,20",
        "HOS_A_NAME_FAR_TOO_LONG_FOR_ANY_SWITCH_OF_THIS_BUILD_AT_ALL_REALLY_X=1"};
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        const int lines = loglines;
        CHECK(!ab_parse(bad[i], &p));
        CHECK(loglines == lines + 1 && strstr(logbuf + last_at, "no A/B in this run"));
    }
    ab_reset();
    const int lines = loglines;
    CHECK(!hos_ab_setup(NULL, T0) && !hos_ab_setup("", T0) && loglines == lines);   /* (unset: nothing said) */
    CHECK(!hos_ab_setup("HOS_X=1,0", T0) && hos_ab_side() == -1 && hos_ab_frame(T0, T0 + S(30)) == -1);
    CHECK(hos_ab_tick(T0 + S(30)) == INT64_MAX && hos_ab_side() == -1);
}

static void schedule(void) {
    step = "the schedule";
    ab_reset();
    CHECK(hos_ab_setup("HOS_OS_YIELD=0,3,20", T0) && strstr(logbuf, "HOS_OS_YIELD alternates every 3.0 s"));
    CHECK(hos_ab_side() == -1);
    fake_now = T0;
    CHECK(hos_ab_tick(T0) == T0 + S(20));
    CHECK(hos_ab_tick(T0 + S(19.999)) == T0 + S(20) && hos_ab_side() == -1);
    CHECK(hos_ab_tick(T0 + S(20)) == T0 + S(23) && hos_ab_side() == 0);
    CHECK(hos_ab_tick(T0 + S(23)) == T0 + S(26) && hos_ab_side() == 1);
    CHECK(hos_ab_tick(T0 + S(25.99)) == T0 + S(26) && hos_ab_side() == 1);
    for (int ms = 0; ms <= 60000; ms += 10) {
        at(ms / 1000.0);
        CHECK(hos_ab_side() == expected_side(ms / 1000.0, 20, 3));
    }
    CHECK(threads_started == 0);   /* (only hos_ab_start starts one) */
    /* (another plan: 1.5 s periods, no warm-up) */
    ab_reset();
    CHECK(hos_ab_setup("HOS_OS_YIELD=0,1.5,0", T0));
    for (int ms = 0; ms <= 20000; ms += 10) {
        at(ms / 1000.0);
        CHECK(hos_ab_side() == expected_side(ms / 1000.0, 0, 1.5));
    }
}

static void frames(void) {
    step = "hos_ab_frame";
    ab_reset();
    CHECK(hos_ab_setup("HOS_OS_YIELD=0,3,20", T0));
    CHECK(hos_ab_frame(T0 + S(5), T0 + S(5.016)) == -1);            /* (the warm-up) */
    CHECK(hos_ab_frame(T0 + S(19.99), T0 + S(20.006)) == 2);         /* (across the warm-up's end) */
    CHECK(hos_ab_frame(T0 + S(20.1), T0 + S(20.116)) == 2);          /* (the first 0.3 s) */
    CHECK(hos_ab_frame(T0 + S(20.284), T0 + S(20.299)) == 2);
    CHECK(hos_ab_frame(T0 + S(20.29), T0 + S(20.306)) == 0);         /* (ends past the settle) */
    CHECK(hos_ab_frame(T0 + S(22.99), T0 + S(23.006)) == 3);         /* (across a flip) */
    CHECK(hos_ab_frame(T0 + S(23.31), T0 + S(23.326)) == 1);
    CHECK(hos_ab_frame(T0 + S(25.97), T0 + S(25.986)) == 1);
    CHECK(hos_ab_frame(0, T0 + S(21)) == 2);                          /* (its start not known) */
    CHECK(hos_ab_frame(T0 + S(22), T0 + S(26.5)) == 2);              /* (a stall over a whole period) */
    CHECK(hos_ab_frame(T0 + S(26.4), T0 + S(26.5)) == 0);
}

static void values(void) {
    step = "hos_ab_value";
    ab_reset();
    /* (no A/B: the normal value, no name looked at) */
    for (int i = 0; i < 1000; i++) CHECK(hos_ab_value("HOS_OS_YIELD", 0) == 0 && hos_ab_value("HOS_OS_YIELD", 5) == 5);
    CHECK(!seen_any() && !ab_read);
    CHECK(hos_ab_setup("HOS_OS_YIELD=7,3,20", T0));
    at(10);   /* (the warm-up: the same) */
    for (int i = 0; i < 1000; i++) CHECK(hos_ab_value("HOS_OS_YIELD", 0) == 0);
    CHECK(!seen_any() && !ab_read);
    at(21);   /* (side 0: the name looked up as on side 1, the normal value) */
    CHECK(hos_ab_value("HOS_OS_YIELD", 0) == 0 && seen_any() && ab_read);
    CHECK(hos_ab_value("HOS_OTHER", 3) == 3);
    at(24);   /* (side 1) */
    CHECK(hos_ab_value("HOS_OS_YIELD", 0) == 7 && hos_ab_value("HOS_OTHER", 3) == 3);
    static char copy[32], other[32];   /* (the name at other addresses; a name's text never changes at its address) */
    strcpy(copy, "HOS_OS_YIELD");
    CHECK(hos_ab_value(copy, 0) == 7);
    strcpy(other, "HOS_OS_YIELD2");
    CHECK(hos_ab_value(other, 0) == 0);
    /* (40 names, 2 of them the switch's, over the 16 slots, asked in turn: each answer right) */
    static char names[40][24];
    for (int i = 0; i < 40; i++)
        if (i % 20) snprintf(names[i], sizeof names[i], "HOS_N%d", i);
        else strcpy(names[i], "HOS_OS_YIELD");
    for (int round = 0; round < 50; round++)
        for (int i = 0; i < 40; i++) CHECK(hos_ab_value(names[(i * 7 + round) % 40], 1) == ((i * 7 + round) % 40 % 20 ? 1 : 7));
    at(27);
    CHECK(hos_ab_value("HOS_OS_YIELD", 0) == 0);
}

static void wired(const char *mode) {
    step = "the wired switches";
    ab_reset();
    const int inverse = !strcmp(mode, "inverse");
    /* (the normal values, cached by each reader at its first call: before any A/B) */
    const int yield0 = inverse;   /* (HOS_OS_YIELD unset: 0; the inverse run sets it to 1) */
    CHECK(os_yield() == yield0);
    /* (a row a wired switch: its plan, its reader, its two sides) */
    struct { const char *plan; int (*read)(void); int side0, side1; } w[] = {
        {inverse ? "HOS_OS_YIELD=0" : "HOS_OS_YIELD=1", os_yield, yield0, !yield0},
    };
    const int nw = (int)(sizeof w / sizeof w[0]);
    for (int k = 0; k < nw; k++) {
        ab_reset();
        CHECK(hos_ab_setup(w[k].plan, T0));
        for (int ms = 0; ms <= 40000; ms += 250) {
            at(ms / 1000.0);
            const int side = expected_side(ms / 1000.0, 20, 3);
            CHECK(w[k].read() == (side == 1 ? w[k].side1 : w[k].side0));
            for (int o = 0; o < nw; o++)   /* (the other switches stay) */
                if (o != k) CHECK(w[o].read() == w[o].side0);
        }
    }
}

static void sums(void) {
    step = "the host's sums";
    ab_reset();
    CHECK(hos_ab_setup("HOS_OS_YIELD=0,3,20", T0));
    for (fake_now = T0; fake_now < T0 + S(50);) {   /* (16 ms frames on side 0 and in the warm-up, 18 ms on side 1) */
        fake_now += ab_side_at(&ab, fake_now) == 1 ? S(0.018) : S(0.016);
        hos_ab_present();
    }
    CHECK(ab_frames[0] && ab_frames[1] && ab_frames[2] && ab_frames[3]);
    char line[768];
    hos_ab_summary(line, sizeof line);
    printf("%s\n", line);
    CHECK(strstr(line, "ab: HOS_OS_YIELD=0 against its normal value, 3.0 s periods after 20 s: "));
    CHECK(strstr(line, "16.000 ms a frame (62.5 fps)") && strstr(line, "18.000 ms a frame (55.6 fps)"));
    CHECK(strstr(line, "side 1 +12.50 % frame time"));
    CHECK(strstr(line, "was never read during the A/B"));
    char want[64];
    snprintf(want, sizeof want, "%lld frames in transitions", (long long)(ab_frames[2] + ab_frames[3]));
    CHECK(strstr(line, want));
    at(24);
    CHECK(os_yield() == 0);   /* (a reader named it) */
    hos_ab_summary(line, sizeof line);
    CHECK(!strstr(line, "never read"));
    const char *before = logbuf + loglen;
    hos_ab_end();
    hos_ab_end();
    const char *first = strstr(before, "ab: HOS_OS_YIELD=0 against");
    CHECK(first && !strstr(first + 1, "ab: HOS_OS_YIELD=0 against"));
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "main";
    if (!strcmp(mode, "main")) {
        parser();
        schedule();
        frames();
        values();
        sums();
    }
    wired(mode);
    printf("hos_ab %s: all passed\n", mode);
    return 0;
}
"""

# (each must make the fixture fail)
BROKEN_C = [
    ("warm-up ignored", "t < p->t0 + p->warm) return -1;", "t < p->t0) return -1;"),
    ("sides swapped", "/ p->period) & 1);", "/ p->period) & 1) ^ 1;"),
    ("no settle after a flip", "to_ns - start < AB_SETTLE_NS ? 2 : 0", "to_ns - start < 0 ? 2 : 0"),
    ("a frame across a flip not a transition", "from_ns < start || ", ""),
    ("the name not checked", "return (e & 1) && side == 1 ? ab.value : normal;", "return side == 1 ? ab.value : normal;"),
    ("a period of 0 accepted", "ab_seconds(f[1], len[1], 1, 3600,", "ab_seconds(f[1], len[1], 0, 3600,"),
    ("a value that is no number accepted", "|| end != b + len[0] ||", "||"),
]
# (the switches wired to hos_ab_value, one a line in name order, each with why a flip between two of its reads is safe)
WIRED = {
    # (read at each barrier sleep and each OR's wake; the game's loop looks at its masks again after a wait or a sleep alike,
    # and a wake skipped while off costs a waiter its 1 ms timeout at most)
    "HOS_OOT_BARRIER_WAIT": "host/oot_barrier.c",
    # (read at each nn::os::YieldThread; nothing is kept between two yields)
    "HOS_OS_YIELD": "host/hos_thread.c",
}


def wiring_checks():
    hos = (ROOT / "host/hos.h").read_text(encoding="utf-8")
    if "static inline int hos_ab_side(void) { return __atomic_load_n(&hos_ab_side_now, __ATOMIC_RELAXED); }" not in hos:
        fail("host/hos.h: hos_ab_side is not one relaxed load, inline")
    main = (ROOT / "host/hos_main.c").read_text(encoding="utf-8")
    if not 0 <= main.index("hos_ab_start();") < main.index("hos_display_start();") < main.index('call_guest(c, hos_entry_init_libc0'):
        fail("host/hos_main.c: hos_ab_start is not called before the display and the game's threads start")
    backend = (ROOT / "host/rx_backend.c").read_text(encoding="utf-8")
    present = block(backend, "void rx_nvn_present(const nvn_tex *t, const int crop[4]) {")
    if not 0 <= present.index("hos_ab_present();") < present.index("vk->nvn_present(t, crop)"):
        fail("host/rx_backend.c: rx_nvn_present does not count the frame (hos_ab_present) before the renderer presents")
    table = re.sub(r"/\*.*?\*/", "", block(backend, "static const MkwVkHost vk_host = {"), flags=re.S)
    if [v.strip() for v in table.split(",") if v.strip()][-2] != "hos_ab_frame":
        fail("host/rx_backend.c: vk_host does not end with hos_ab_frame (MkwVkHost.ab_frame, version 34)")
    api = (ROOT / "gpu/vk/vk_api.h").read_text(encoding="utf-8")
    if int(re.search(r"#define MKWVK_API_VERSION (\d+)", api).group(1)) < 35 or \
            "int (*ab_frame)(int64_t from_ns, int64_t to_ns);\n} MkwVkHost;" not in api:
        fail("gpu/vk/vk_api.h: MkwVkHost does not end with ab_frame at version 34 or later")
    replay = (ROOT / "gpu/vk/vk_replay.cpp").read_text(encoding="utf-8")
    if "host->version >= 35 && host->ab_frame ? host->ab_frame(l_at, at) : -1" not in replay:
        fail("gpu/vk/vk_replay.cpp: the frame trace does not ask the host for ab_side (only from a version 34 host)")
    finish = block((ROOT / "host/hos_relaunch.c").read_text(encoding="utf-8"), "void hos_close_finish(void) {")
    if not 0 <= finish.index("hos_ab_end();") < finish.index("hos_log_flush();"):
        fail("host/hos_relaunch.c: hos_close_finish does not log the A/B's sums before the log's flush")
    # (the wired switches: each one's safety under a flip is argued at its call and run below; a new one is added here)
    found = {}
    for path in sorted((ROOT / "host").rglob("*.c")):
        for m in re.finditer(r'hos_ab_value\("([A-Z0-9_]+)"', path.read_text(encoding="utf-8")):
            found.setdefault(m.group(1), []).append(path.relative_to(ROOT).as_posix())
    if {k: v for k, v in found.items()} != {k: [v] for k, v in WIRED.items()}:
        fail("the switches wired to hos_ab_value are %r, not %r (a switch whose value sets something up once may not be)"
             % (found, WIRED))


def c_fixture(cc, tmp):
    hos = (ROOT / "host/hos.h").read_text(encoding="utf-8")
    decls = between(hos, "extern int hos_ab_side_now;", "void hos_ab_end(void);")
    backend = (ROOT / "host/rx_backend.c").read_text(encoding="utf-8")
    threads = (ROOT / "host/hos_thread.c").read_text(encoding="utf-8")
    yield_reader = re.search(r"^static int os_yield\(void\) \{\n.*?^\}\n", threads, re.M | re.S)
    if not yield_reader:
        fail("host/hos_thread.c: no os_yield() to cut (HOS_OS_YIELD's reader)")
    fixture = FIXTURE.replace("@OS_YIELD@", yield_reader.group(0))
    source = (ROOT / "host/hos_ab.c").read_text(encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HOS_")}
    runs = (("main", {}), ("inverse", {"HOS_OS_YIELD": "1"}))

    def build(name, ab_source, broken=False):
        d = Path(tmp) / name
        d.mkdir()
        (d / "hos.h").write_text(STUB_HOS.replace("@AB_DECLS@", decls), encoding="utf-8")
        (d / "windows.h").write_text(STUB_WINDOWS, encoding="utf-8")
        (d / "hos_ab.c").write_text(ab_source, encoding="utf-8")
        (d / "fixture.c").write_text(fixture, encoding="utf-8")
        exe = d / "fixture.exe"
        # (a broken variant may leave a parameter or a variable unused)
        quiet = ["-Wno-unused-parameter", "-Wno-unused-variable", "-Wno-unused-but-set-variable"] if broken else []
        r = subprocess.run([cc, "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function"] + quiet +
                           ["-I", str(d), "-I", str(ROOT / "gpu/vk/compat"), str(d / "fixture.c"), "-o", str(exe)],
                           capture_output=True, text=True)
        if r.returncode:
            fail("the fixture (%s) does not compile:\n%s%s" % (name, r.stdout, r.stderr))
        return exe

    exe = build("fixture", source)
    for mode, switches in runs:
        r = subprocess.run([str(exe), mode], env=dict(env, **switches), capture_output=True, text=True, timeout=120)
        if r.returncode or ("hos_ab %s: all passed" % mode) not in r.stdout:
            fail("hos_ab.c, %s:\n%s%s" % (mode, r.stdout, r.stderr))
        print(r.stdout.strip().replace("\n", " / "))
    for i, (what, old, new) in enumerate(BROKEN_C):
        if source.count(old) != 1:
            fail("host/hos_ab.c no longer has the text the broken variant '%s' changes: %r" % (what, old))
        broken = build("broken-%d" % i, source.replace(old, new), broken=True)
        r = subprocess.run([str(broken), "main"], env=env, capture_output=True, text=True, timeout=120)
        if r.returncode == 0 or "FAIL" not in r.stderr:
            fail("broken variant '%s' passed" % what)
        print("broken variant %s: caught (%s)" % (what, r.stderr.strip().splitlines()[0]))


# ---- scripts/ab_summary.py
BROKEN_PY = [
    ("transitions kept", '        if v in TRANSITION:\n            p["transitions"] += 1\n        else:\n            p["ms"].append(ms)\n',
     '        p["ms"].append(ms)\n'),
    ("sides swapped", "        side = v & 1\n", "        side = (v & 1) ^ 1\n"),
    ("the warm-up counted", "        if v not in STEADY and v not in TRANSITION:   # (outside the A/B: its warm-up)\n",
     "        if False:\n"),
    ("the cut-short period kept", '        if periods[-1]["total"] < 0.8 * median:\n', "        if False:\n"),
    ("no parity trim", '    if len(periods) > 1 and periods[0]["side"] != periods[-1]["side"]:\n', "    if False:\n"),
]


def trace(path, rows):
    t = 0
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("frame,ms,game_ms,at_ns,ab_side\n")
        for i, (ms, side) in enumerate(rows):
            t += int(ms * 1e6)
            f.write("%d,%.3f,1.00,%d,%d\n" % (i + 1, ms, t, side))
        f.write("%d,16.0" % (len(rows) + 1))   # (a line cut short by the run's end)


def made_up(path):
    """20 s of 50 ms warm-up frames; 12 whole periods of 3 s (three 100 ms transition frames, then side 0's 15/17 ms or side 1's
    17/19 ms); a 13th, side 0's, cut short at 1.5 s (30 ms frames)"""
    rows = [(50.0, -1)] * 400
    for k in range(12):
        side = k & 1
        pair, n = ((15.0, 17.0), 168) if side == 0 else ((17.0, 19.0), 150)
        rows += [(100.0, side + 2)] * 3
        rows += [(pair[i & 1], side) for i in range(n)]
    rows += [(100.0, 2)] * 3 + [(30.0, 0)] * 40
    trace(path, rows)


def drifting(path):
    """20 whole periods whose frames slow steadily (16 ms + 0.1 ms a second), alike on both sides"""
    rows, t = [], 0.0
    for k in range(20):
        side, end = k & 1, (k + 1) * 3.0
        first = True
        while t < end - 1e-9:
            ms = 16.0 + 0.1 * t
            t += ms / 1000
            rows.append((ms, side + (2 if first else 0)))
            first = False
    trace(path, rows)


def summary_checks(tmp):
    script = (ROOT / "scripts/ab_summary.py").read_text(encoding="utf-8")
    a, b = Path(tmp) / "frames.csv", Path(tmp) / "drift.csv"
    made_up(a)
    drifting(b)

    def run(text, *args):
        p = Path(tmp) / "ab_summary_run.py"
        p.write_text(text, encoding="utf-8")
        r = subprocess.run([sys.executable, "-I", "-B", str(p), "--json", *map(str, args)], capture_output=True, text=True,
                           timeout=120)
        if r.returncode:
            return None, r.stderr
        return json.loads(r.stdout), ""

    def check(text):
        """the problems with the script's numbers, [] when right"""
        bad = []
        res, err = run(text, a, a)
        if res is None:
            return ["did not run: " + err]
        one, both = res[0], res[-1]
        s0, s1 = one["sides"]["0"], one["sides"]["1"]
        if one["periods"] != 11 or one["periods_left_out"] != 2 or s0["periods"] != 6 or s1["periods"] != 5:
            bad.append("periods %d (%d and %d), %d left out: 11 (6 and 5), 2 left out expected"
                       % (one["periods"], s0["periods"], s1["periods"], one["periods_left_out"]))
        if one["transition_frames"] != 39 or one["outside_frames"] != 400:
            bad.append("%d transition frames and %d outside: 39 and 400 expected" % (one["transition_frames"], one["outside_frames"]))
        for st, frames, mean, median, slow in ((s0, 6 * 168, 16.0, 16.0, 0.5), (s1, 5 * 150, 18.0, 18.0, 1.0)):
            if st.get("frames") != frames or abs(st.get("mean_ms", 0) - mean) > 1e-9 or abs(st.get("median_ms", 0) - median) > 1e-9 \
                    or abs(st.get("slow_share", -1) - slow) > 1e-9:
                bad.append("a side: %r, expected %d frames, mean %.1f, median %.1f, share >= 16.7 ms %.1f" % (st, frames, mean, median, slow))
        d = one.get("diff", {})
        if abs(d.get("mean_ms", 0) - 2.0) > 1e-9 or abs(d.get("mean_pct", 0) - 12.5) > 1e-9 or not d.get("ci_ms") or d["ci_ms"][0] <= 0:
            bad.append("the difference %r: +2 ms, +12.5 %%, an interval above 0 expected" % d)
        if both["name"] != "all 2 files" or both["periods"] != 22 or abs(both["sides"]["1"].get("mean_ms", 0) - 18.0) > 1e-9:
            bad.append("two files together: %r" % both)
        res, err = run(text, b)
        if res is None:
            return bad + ["the drift did not run: " + err]
        dd = res[0].get("diff", {}).get("mean_ms", 9)
        if res[0]["periods"] != 19 or abs(dd) > 0.1:
            bad.append("a steady drift: %d periods, side 1 %+.3f ms: 19 periods and no difference expected" % (res[0]["periods"], dd))
        return bad

    bad = check(script)
    if bad:
        fail("scripts/ab_summary.py:\n  " + "\n  ".join(bad))
    r = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts/ab_summary.py"), str(a)], capture_output=True, text=True,
                       timeout=120)
    if r.returncode or "side 1 is slower" not in r.stdout:
        fail("scripts/ab_summary.py's text:\n" + r.stdout + r.stderr)
    print(r.stdout.rstrip())
    for what, old, new in BROKEN_PY:
        if script.count(old) != 1:
            fail("scripts/ab_summary.py no longer has the text the broken variant '%s' changes: %r" % (what, old))
        found = check(script.replace(old, new))
        if not found:
            fail("broken variant of scripts/ab_summary.py '%s' passed" % what)
        print("broken variant %s: caught (%s)" % (what, found[0][:110]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cc", default="clang")
    a = p.parse_args()
    cc = shutil.which(a.cc)
    if not cc:
        fail("no C compiler %r (pass --cc)" % a.cc)
    wiring_checks()
    with tempfile.TemporaryDirectory(prefix="hos-ab-") as tmp:
        c_fixture(cc, tmp)
        summary_checks(tmp)
    print("PASS: the A/B's parser, schedule, frames, cheap path and sums; the wired switch flips with it; "
          "ab_summary.py's sides, transitions, warm-up, trims and interval")


if __name__ == "__main__":
    main()
