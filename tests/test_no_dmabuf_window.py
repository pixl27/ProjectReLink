"""RX_VK_NO_DMABUF makes the guest window anonymous memory, so the renderer's pools can be host-pointer imports.

The switch tells the renderer to make no dma-bufs (gpu/vk/vk_memory.cpp, import_dma_buf) and import host pointers instead,
but host/hos_main.c still mapped the window from a memfd wherever /dev/udmabuf opened, and a shared mapping's pages
cannot be host-pointer imports (RADV: VK_ERROR_INVALID_EXTERNAL_HANDLE): no pool was imported, every one was copied,
2-5 fps for ~15 s after the save loaded (RX 9070 XT), and a teleport waited out the long arrival event.

The check cuts the decision out of host/hos_main.c by its text and runs it for each of /dev/udmabuf opened or not,
RX_VK_NO_DMABUF set or not and HOS_GUEST_MEMFD unset, 0 or 1.

    python -I -B tests/test_no_dmabuf_window.py [--cc gcc]
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cc", "gcc"]
RUN_ALL_SKIP = {"windows": "the Linux guest window's start-up check"}
ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include <stdio.h>
#include <stdlib.h>
#define hos_log(...) (fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))
int main(int argc, char **argv) {
    (void)argc;
    const char *e = getenv("HOS_GUEST_MEMFD");
    int usable = atoi(argv[1]);
@@CODE@@
        printf("memfd\n");
        return 0;
    }
    printf("anonymous\n");
    return 0;
}
'''


def run(exe, usable, no_dmabuf, memfd):
    env = {"PATH": "/usr/bin:/bin"}
    if no_dmabuf:
        env["RX_VK_NO_DMABUF"] = "1"
    if memfd is not None:
        env["HOS_GUEST_MEMFD"] = memfd
    out = subprocess.run([str(exe), "1" if usable else "0"], capture_output=True, text=True, check=True, env=env)
    return out.stdout.strip(), out.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="gcc")
    args = parser.parse_args()
    src = (ROOT / "host/hos_main.c").read_text(encoding="utf-8")
    begin = src.index('const int no_dmabuf = env_on("RX_VK_NO_DMABUF")')
    decide = src.index("if (e ? !env_value_off(e) :", begin)
    code = src[begin:src.index("{", decide) + 1]
    with tempfile.TemporaryDirectory(prefix="no-dmabuf-window-") as tmp:
        c, exe = Path(tmp) / "fixture.c", Path(tmp) / "fixture"
        c.write_text(FIXTURE.replace("@@CODE@@", code), encoding="utf-8")
        subprocess.run([args.cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-include", str(ROOT / "gpu/vk/compat/hookshot_env.h"), str(c),
                        "-o", str(exe)], check=True)

        want = {
            # (udmabuf opens, RX_VK_NO_DMABUF, HOS_GUEST_MEMFD) -> window
            (True, False, None): "memfd",
            (True, True, None): "anonymous",   # the bug: was memfd, so no pool could be imported
            (False, False, None): "anonymous",
            (False, True, None): "anonymous",
            (True, True, "1"): "memfd",        # asked for all the same
            (True, False, "0"): "anonymous",
            (False, True, "1"): "memfd",
        }
        for (usable, no_dmabuf, memfd), window in want.items():
            got, log = run(exe, usable, no_dmabuf, memfd)
            assert got == window, f"udmabuf {usable}, RX_VK_NO_DMABUF {no_dmabuf}, HOS_GUEST_MEMFD {memfd}: {got}, want {window}"
            said = "RX_VK_NO_DMABUF" in log
            assert said == (usable and no_dmabuf and memfd is None), (usable, no_dmabuf, memfd, log)
    print("PASS: RX_VK_NO_DMABUF -> anonymous guest window (host-pointer imports) unless HOS_GUEST_MEMFD=1")


if __name__ == "__main__":
    main()
