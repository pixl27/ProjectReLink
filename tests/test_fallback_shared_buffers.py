"""Serial replay threads share fallback scratch; parallel recorders retain private buffers.

Compiles the production selector, then changes producer threads in submission order.
Run: python -I -B tests/test_fallback_shared_buffers.py.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from srccut import block

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    source = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    selector = block(source, "std::vector<FallbackBuffers>& fallback_buffers() {")
    code = r'''
#include "gpu/vk/compat/hookshot_env.h"
#include <atomic>
#include <vector>
#include <thread>
#include <cassert>
#include <cstdio>
struct FallbackBuffers { int tag; };
std::atomic<bool> par_active{false};
std::vector<FallbackBuffers> shared_fallback_buffers;
std::vector<FallbackBuffers>& parallel_fallback_buffers() {
    static thread_local std::vector<FallbackBuffers> v;
    return v;
}
@SELECTOR@
int main(int argc, char**) {
    std::vector<FallbackBuffers>* first=nullptr;
    std::thread a([&] { first=&fallback_buffers(); first->push_back({42}); }); a.join();
    if (argc==1) {
        std::thread b([&] { assert(&fallback_buffers()==first); assert(fallback_buffers().at(0).tag==42); }); b.join();
        par_active=true;
        std::thread c([&] { assert(&fallback_buffers()!=first); assert(fallback_buffers().empty()); fallback_buffers().push_back({99}); }); c.join();
        par_active=false; assert(fallback_buffers().at(0).tag==42);
    } else {
        assert(shared_fallback_buffers.empty());
        assert(fallback_buffers().empty());
    }
    puts("PASS: serial scratch survives producer changes; parallel and disabled modes stay private");
}
'''.replace("@SELECTOR@", selector)
    import os
    with tempfile.TemporaryDirectory(prefix="shared-fallback-") as tmp:
        cpp, exe = Path(tmp) / "test.cpp", Path(tmp) / "test.exe"
        cpp.write_text(code, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(ROOT), str(cpp), "-o", str(exe)], check=True)
        for setting in ("1", "0", None):
            env = dict(os.environ)
            env.pop("RX_VK_MFB_SHARED", None)
            if setting is not None:
                env["RX_VK_MFB_SHARED"] = setting
            subprocess.run([str(exe)] + ([] if setting=="1" else ["off"]), env=env, check=True)

if __name__ == "__main__":
    main()
