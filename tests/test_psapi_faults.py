"""The Linux psapi shim's page-fault count (host/posix/psapi.h K32GetProcessMemoryInfo).

Windows' PROCESS_MEMORY_COUNTERS.PageFaultCount counts the process's soft and hard faults together; the shim filled it with
0, so no log line could carry faults on Linux. It now reads getrusage (ru_minflt + ru_majflt). This builds a tiny program
against the shim as it is: the count is not 0, and touching 64 MB of new memory raises it by about one fault a page (at
least a quarter of the pages: transparent huge pages may take some in one fault); the working set and private bytes are
still filled as before.

Linux only.  python -I -B tests/test_psapi_faults.py [--cc gcc]
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

PROGRAM = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <psapi.h>
int main(void) {
    PROCESS_MEMORY_COUNTERS_EX a, b;
    if (!K32GetProcessMemoryInfo(0, (PROCESS_MEMORY_COUNTERS *)&a, sizeof a)) return 2;
    const size_t n = 64u << 20;
    char *p = (char *)mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 3;
    madvise(p, n, MADV_NOHUGEPAGE);
    for (size_t k = 0; k < n; k += 4096) p[k] = 1;
    if (!K32GetProcessMemoryInfo(0, (PROCESS_MEMORY_COUNTERS *)&b, sizeof b)) return 4;
    printf("%lu %lu %llu %llu\n", (unsigned long)a.PageFaultCount, (unsigned long)b.PageFaultCount,
           (unsigned long long)b.WorkingSetSize, (unsigned long long)b.PrivateUsage);
    return 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc", default="gcc")
    a = p.parse_args()
    if os.name == "nt" or sys.platform == "darwin":
        print("SKIP: the posix layer's checks run on Linux")
        return
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / "faults.c", Path(d) / "faults"
        src.write_text(PROGRAM, encoding="utf-8")
        subprocess.run([a.cc, "-std=gnu11", "-I", str(ROOT / "host/posix"), "-Wno-unused-function", "-Wno-attributes", str(src),
                        "-o", str(exe)],
                       check=True)
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise SystemExit(f"FAIL: the program exited {r.returncode}: {r.stderr}")
        before, after, ws, private = map(int, r.stdout.split())
    pages = (64 << 20) // 4096
    if before == 0:
        raise SystemExit("FAIL: PageFaultCount is 0")
    if after - before < pages // 4:
        raise SystemExit(f"FAIL: touching {pages} new pages raised PageFaultCount by {after - before}")
    if ws < (64 << 20) or private < (64 << 20):
        raise SystemExit(f"FAIL: the working set ({ws}) or private bytes ({private}) lost the 64 MB touched")
    print(f"PASS: PageFaultCount follows getrusage ({before} -> {after} over {pages} new pages)")


if __name__ == "__main__":
    main()
