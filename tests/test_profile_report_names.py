"""The profiling mode's report names a Windows build's code and threads as it names a Linux build's (scripts/prof.py,
scripts/profile_report.py), without the game:

  - a Windows linker map's C++ symbols (llvm-mingw: Itanium names, _ZN...) come out demangled, as nm -C gives them on
    Linux; a name that isn't mangled, and a lifted import's (imp__ZN...: scripts/cpu_report.py reads those), stay as they
    are. Skipped when no demangler is on this PC (llvm-cxxfilt or c++filt);
  - a sample asleep under a wait counts as waiting, not running, with a C++ name's parameters (as nm -C and the Windows
    demangling give it): std::condition_variable's and libc++'s waits, host/hos_thread.c's sleep_away, the renderer's
    wait for a GPU fence;
  - the Threads table puts a thread's CPU % from the log (`prof: threads in` names it with spaces, "mkw_vk replay") on the
    same row as its samples (the samplers write "mkw_vk_replay").

python -I -B tests/test_profile_report_names.py
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prof  # noqa: E402

WAIT = "_ZNSt3__118condition_variable4waitERNS_11unique_lockINS_5mutexEEE"
MAP = """ Address         Publics by Value              Rva+Base               Lib:Object

 0001:00000000       _ZN5mkwvk12_GLOBAL__N_14waitEyy 0000000180001000     vk_vk_replay.o
 0001:00000100       %s 0000000180001100     libc++:condition_variable.cpp.obj
 0001:00000200       vk_record_frame            0000000180001200     vk_vk_replay.o
 0001:00000300       imp__ZN2nn2os13WaitLightEventEPNS0_14LightEventTypeE 0000000180001300     vk_vk_replay.o
""" % WAIT


def need(ok, what):
    if not ok:
        raise SystemExit("profile report names: " + what)


def demangled():
    with tempfile.TemporaryDirectory() as t:
        mp = Path(t) / "hookshot_vk.map"
        mp.write_text(MAP, encoding="utf-8")
        _, names = prof.load_map(str(mp), 0x180000000)
    need("vk_record_frame" in names, "a plain C name changed: %r" % names)
    need("imp__ZN2nn2os13WaitLightEventEPNS0_14LightEventTypeE" in names, "a lifted import's name changed: %r" % names)
    find = getattr(prof, "demangler", lambda: shutil.which("llvm-cxxfilt") or shutil.which("c++filt"))
    if not find():
        print("profile report names: no llvm-cxxfilt or c++filt here: the demangling is not checked")
        return
    need("mkwvk::(anonymous namespace)::wait(unsigned long long, unsigned long long)" in names,
         "a renderer function's Itanium name is not demangled: %r" % names)
    wait = next((n for n in names if "condition_variable" in n), "")
    need(wait.startswith("std::__1::condition_variable::wait("), "libc++'s wait is not demangled: %r" % wait)


def waits():
    """a sample asleep in one of these counts as waiting, whichever system named it"""
    wait = "std::__1::condition_variable::wait(std::__1::unique_lock<std::__1::mutex>&)"
    # (a sample of a thread asleep in it: the system's frame the leaf, libc++'s wait (linked into hookshot_vk.dll) above it)
    need(prof.waiting("ntdll.dll", ["ntdll.dll", "hookshot_vk:" + wait, "hookshot_vk:vk_record_frame"]),
         "a thread asleep in std::condition_variable::wait counts as running")
    for frame in ("hookshot_vk:std::__1::__libcpp_condvar_timedwait(void**, void**, timespec*)", "hookshot:sleep_away",
                  "hookshot_vk:mkwvk::(anonymous namespace)::wait(unsigned long long, unsigned long long)"):
        need(prof.waiting("ntdll.dll", ["ntdll.dll", frame]), "a thread asleep in %s counts as running" % frame)
    need(not prof.waiting("ntdll.dll", ["ntdll.dll", "hookshot_vk:mkwvk::(anonymous namespace)::record(int)"]),
         "a thread in the system's code under a function that doesn't wait counts as waiting")


def threads_table():
    """a made-up profile folder: two threads sampled, the log naming one of them with a space"""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "game" / "logs" / "profile-20261117-020000"
        d.mkdir(parents=True)
        t0 = 1_000_000_000
        pres = ["%d,%d,%d,0,0,0,0" % (i + 1, t0 + (i + 1) * 16_700_000, 16_700_000) for i in range(30)]
        (d / "presents.csv").write_text("frame,return_ns,interval_ns,present_begin_ns,renderer_begin_ns,renderer_end_ns,"
                                        "vcore_back_ns\n" + "\n".join(pres) + "\n", encoding="utf-8")
        lines = ["# t0_ns %d" % t0, "s 1 hookshot_vk.dll+10", "s 2 hookshot.exe+20"]
        for ms in range(18, 500, 2):
            lines += ["%d mkw_vk_replay|=1" % ms, "%d sead::MainThread|=2" % ms]
        (d / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (d / "process.json").write_text(json.dumps({"env": {"HOS_PROFILE": "1"}}), encoding="utf-8")
        (d / "game.log").write_text("[hos host] prof: threads in 5 s: sead::MainThread 34%; mkw_vk replay 36%;\n",
                                    encoding="utf-8")
        out = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts/profile_report.py"), str(d), "--maps", t,
                              "--no-trace"], capture_output=True, text=True, encoding="utf-8")
        need(out.returncode == 0, "profile_report.py failed:\n" + out.stdout + out.stderr)
        rep = (d / "report.md").read_text(encoding="utf-8")
        rows = [r for r in rep.splitlines() if r.startswith("| mkw_vk")]
        need(len(rows) == 1, "the renderer's replay thread has %d rows in Threads, not one: %r" % (len(rows), rows))
        cells = [c.strip() for c in rows[0].strip("|").split("|")]
        need(cells[1] == "36" and cells[3] != "0", "its row lacks the log's CPU share or its samples: %r" % rows[0])


if __name__ == "__main__":
    demangled()
    waits()
    threads_table()
    print("profile report names: ok")
