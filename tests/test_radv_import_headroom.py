"""Exercise system-heap headroom before a large RADV dma-buf import."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    source = (ROOT / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    factory = block(source, "bool import_dma_buf(")
    assert "radv_import_headroom(hi - lo);" in factory, "large imports do not make room before allocating"
    assert factory.index("radv_import_headroom(hi - lo);") < factory.index("!import_dma_buf_pieces(")
    harness = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <vector>

constexpr uint64_t MB = 1ull << 20;
constexpr int VK_DRIVER_ID_MESA_RADV = 1;
struct { int driver_id = VK_DRIVER_ID_MESA_RADV; } dev;
struct FreeBuf { unsigned buffer; bool alloc; void* map; uint64_t size; };
std::mutex cap_free_mu;
std::vector<FreeBuf> cap_free;
uint64_t cap_free_bytes, used, budget;
bool budget_known = true;
unsigned probes;
std::vector<unsigned> destroyed;
bool allocation_in_vram(bool device_local) { return device_local; }
void destroy_buffer(unsigned buffer, bool) { destroyed.push_back(buffer); }
void logf(const char*, ...) {}
bool system_heap_usage(uint64_t* usage, uint64_t* limit) {
    ++probes;
    *usage = used;
    *limit = budget;
    return budget_known;
}
""" + block(source, "void radv_import_headroom(uint64_t bytes) {") + r"""
int main() {
    budget = 1936 * MB;
    used = 1000 * MB;
    cap_free = {{1, false, nullptr, 224 * MB}, {2, false, nullptr, 64 * MB},
                {3, true, nullptr, 128 * MB}, {4, false, nullptr, 32 * MB}};
    cap_free_bytes = 448 * MB;

    radv_import_headroom(32 * MB);
    dev.driver_id = 0;
    radv_import_headroom(224 * MB);
    assert(probes == 0 && destroyed.empty());

    dev.driver_id = VK_DRIVER_ID_MESA_RADV;
    radv_import_headroom(224 * MB);
    assert(probes == 1 && destroyed.empty());
    budget_known = false;
    used = 1934 * MB;
    radv_import_headroom(224 * MB);
    assert(destroyed.empty());

    budget_known = true;
    radv_import_headroom(224 * MB);
    assert((destroyed == std::vector<unsigned>{4, 2, 1}));
    assert(cap_free.size() == 1 && cap_free.front().buffer == 3 && cap_free_bytes == 128 * MB);

    // Exact room for a 64 MiB import and 64 MiB of submit reserve.
    cap_free.push_back({5, false, nullptr, 64 * MB});
    cap_free_bytes += 64 * MB;
    used = budget - 128 * MB;
    radv_import_headroom(64 * MB);
    assert(cap_free.size() == 2 && destroyed.back() == 1);
    used += MB;
    radv_import_headroom(64 * MB);
    assert(cap_free.size() == 1 && destroyed.back() == 5);
}
"""
    with tempfile.TemporaryDirectory(prefix="radv-import-headroom-") as temp:
        cpp = Path(temp) / "fixture.cpp"
        exe = Path(temp) / "fixture.exe"
        cpp.write_text(harness, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True, timeout=15)
    print("RADV import headroom cases passed")


if __name__ == "__main__":
    main()
