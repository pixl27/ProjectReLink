"""Test the POSIX Win32 layer's process affinity and thread-priority handoff.

No game data, guest memory, or renderer is used.

python nx2/test_win32_posix_threading.py [--cc clang]
"""
import argparse
from pathlib import Path
import platform
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        print("SKIP: win32_posix.c currently targets Linux/x86_64")
        return
    fixture = r'''
#define _GNU_SOURCE
#include <windows.h>

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); exit(1); } } while (0)

static DWORD record_nice(void *arg) {
    *(int *)arg = getpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid));
    return 0;
}

static DWORD_PTR mask64(const cpu_set_t *set) {
    DWORD_PTR mask = 0;
    for (int i = 0; i < 64; i++)
        if (CPU_ISSET(i, set)) mask |= (DWORD_PTR)1 << i;
    return mask;
}

int main(void) {
    cpu_set_t launch, one;
    CHECK(sched_getaffinity(0, sizeof launch, &launch) == 0);
    const DWORD_PTR launch_mask = mask64(&launch);
    CHECK(launch_mask != 0);

    int cpu = 0;
    while (cpu < 64 && !CPU_ISSET(cpu, &launch)) cpu++;
    CHECK(cpu < 64);
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    CHECK(sched_setaffinity(0, sizeof one, &one) == 0);

    DWORD_PTR process_mask = 0, system_mask = 0;
    CHECK(GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask));
    CHECK(process_mask == launch_mask);
    CHECK((system_mask & process_mask) == process_mask);

    /* The parent and children share one CPU, so CreateThread returns before the child can publish its tid. */
    for (int i = 0; i < 32; i++) {
        int nice = -1;
        HANDLE thread = CreateThread(NULL, 0, record_nice, &nice, 0, NULL);
        CHECK(thread != NULL);
        CHECK(SetThreadPriority(thread, THREAD_PRIORITY_LOWEST));
        CHECK(WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0);
        CHECK(nice == 10);
        CHECK(CloseHandle(thread));
    }

    CHECK(sched_setaffinity(0, sizeof launch, &launch) == 0);
    puts("PASS: launch affinity survives thread pinning and early priorities reach new threads");
}
'''
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="win32-posix-threading-", dir=work) as temp:
        directory = Path(temp).resolve()
        directory.relative_to(work.resolve())
        source = directory / "fixture.c"
        executable = directory / "fixture"
        source.write_text(fixture, encoding="utf-8")
        subprocess.run(
            [
                args.cc,
                "-std=gnu11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-I",
                str(ROOT / "host" / "posix"),
                str(ROOT / "host" / "posix" / "win32_posix.c"),
                str(source),
                "-ldl",
                "-pthread",
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
