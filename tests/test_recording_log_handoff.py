"""Check the recording table and guest core are available while the periodic log sink is blocked.

Run python -I -B tests/test_recording_log_handoff.py [--cc clang]. The fixture compiles
the publication section of EndRecording and holds its log call until the parent replies.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srccut import between, block

ROOT = Path(__file__).resolve().parents[1]
RUN_ALL_ARGS = ["--cc", "gcc"]

FIXTURE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct rec_node { uint64_t ctl; struct rec_node *next; } rec_node;
static rec_node *slots[1], **rec_buckets = slots;
enum { REC_BUCKETS = 1, SWEEP_EVERY = 1024 };
static unsigned rec_shift;
static size_t ncopies, npins;
static long long copy_bytes, seg_count, seg_bytes;
static int copy_lock, core, away_calls, back_calls, debug_sweep;
static unsigned rec_bucket(uint64_t ctl) { (void)ctl; return 0; }
static void live_handle_add(uint64_t h) { (void)h; }
static size_t sweep_overwritten_copies(void) { return debug_sweep ? 3 : 0; }
#define HOS_LOG_DEBUG 0
#define hos_log_at(level, ...) hos_log(__VA_ARGS__)
static void AcquireSRWLockExclusive(int *lock) { if ((*lock)++) abort(); }
static void ReleaseSRWLockExclusive(int *lock) { if (--*lock) abort(); }
static int hos_vcore_away(void) {
    if (copy_lock) abort();
    int held = core;
    core = 0;
    away_calls++;
    return held;
}
static void hos_vcore_back(void) { if (copy_lock || core++) abort(); back_calls++; }
static void hos_log(const char *fmt, ...) {
    (void)fmt;
    printf("%d %d\n", copy_lock, core);
    fflush(stdout);
    if (getchar() == EOF) abort();
}
static void publish(rec_node *n, void *copy) {
    /* PUBLISH */
}
int main(int argc, char **argv) {
    if (argc != 3) return 1;
    core = atoi(argv[1]);
    debug_sweep = atoi(argv[2]);
    const int limit = debug_sweep ? SWEEP_EVERY : 50000;
    rec_node node = {0};
    for (int i = 1; i <= limit; i++) {
        publish(&node, &node);
        if (core != atoi(argv[1]) || copy_lock) return 2;
        if (i < limit && (away_calls || back_calls)) return 3;
    }
    return ncopies != (size_t)limit || away_calls != 1 || back_calls != atoi(argv[1]);
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    args = parser.parse_args()
    source = (ROOT / "host/nvn_native.c").read_text()
    func = block(source, "NV(nvnCommandBufferEndRecording)")
    publish = between(func, "    AcquireSRWLockExclusive(&copy_lock);", "    if (watch_draw_cmd")
    with tempfile.TemporaryDirectory() as tmp:
        c, exe = Path(tmp) / "fixture.c", Path(tmp) / "fixture.exe"
        c.write_text(FIXTURE.replace("/* PUBLISH */", publish))
        subprocess.run([args.cc, "-std=gnu11", str(c), "-o", str(exe)], check=True)
        for held, sweep in ((1, 0), (0, 0), (1, 1), (0, 1)):
            with subprocess.Popen([str(exe), str(held), str(sweep)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True) as process:
                # The sink reports ownership, then blocks until the parent lets the call finish.
                deadline = Timer(10, process.kill)
                deadline.start()
                try:
                    state = process.stdout.readline().strip()
                    process.communicate("\n")
                finally:
                    deadline.cancel()
                    deadline.join()
                if state != "0 0":
                    raise SystemExit(f"FAIL: recording lock and core at blocked sink: {state!r}, expected '0 0'")
                if process.returncode:
                    raise SystemExit(f"FAIL: handoff/restore for initial core {held}: exit {process.returncode}")
    print("PASS: blocked recording log releases both resources and restores only a borrowed core")


if __name__ == "__main__":
    main()
