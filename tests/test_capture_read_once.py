"""Run the renderer's scalar and batched reads against distinct live and captured pages.

python -I -B tests/test_capture_read_once.py --cxx g++
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srccut import block

ROOT = Path(__file__).resolve().parents[1]
RUN_ALL_ARGS = ["--cxx", "{cxx}"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    source = (ROOT / "gpu/vk/vk_memory.cpp").read_text()
    funcs = ["int submitted_read_unlocked(", "bool memory_read_as_submitted(", "void memory_read_as_submitted_many("]
    fixture = r"""
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>
constexpr uint64_t kPageShift = 12, kPages = 3;
constexpr uint64_t virt = 1ull << 40;
uint8_t ram[3*4096];
unsigned live_reads;
struct Host {
    uint8_t* guest_base() { return ram; }
    uint64_t guest_size() { return sizeof ram; }
    uint64_t gpu_guest(uint64_t a) {
        if (a >= virt) {
            uint64_t off = a-virt;
            if (off < 4096) return 8192+off;
            if (off < 8192) return off;
            return 0;
        }
        return a < sizeof ram ? a : 0;
    }
    bool gpu_read(uint64_t a, void* dst, uint64_t n) {
        ++live_reads;
        for (uint64_t i = 0; i < n; ++i) {
            auto physical = gpu_guest(a+i);
            if (!physical) return false;
            static_cast<uint8_t*>(dst)[i] = ram[physical];
        }
        return true;
    }
} api, *host = &api;
std::mutex pools_mu;
std::atomic<bool> captures_seen{true};
std::atomic<uint64_t> cap_view_gen{1};
bool capture_on() { return true; }
struct SubmittedPage { uint64_t page, pgen, view; } entry{};
uint32_t page_gen(uint64_t p) { assert(p < kPages); return 1; }
SubmittedPage& submitted_page_for(uint64_t) { return entry; }
bool submitted_page_reuse(uint64_t p,uint64_t g,uint64_t v,uint64_t q,uint64_t h,uint64_t w) {
    return p==q && g==h && v==w;
}
// The current capture covers page 1. Multi-page requests overlay only that part.
int64_t submitted_read_locked(uint64_t a, void *dst, uint64_t n) {
    uint64_t lo = std::max<uint64_t>(a,4096), hi = std::min<uint64_t>(a+n,8192);
    if (lo>=hi) return 0;
    std::memset(static_cast<uint8_t*>(dst)+(lo-a), 42, hi-lo);
    return 1;
}
void submitted_read_counted(int64_t, int64_t) {}
struct SubmittedRead { uint64_t gpu; void* dst; uint64_t n; };
"""
    if "int64_t mapped_submitted_read_locked(" in source:
        funcs.insert(1, "int64_t mapped_submitted_read_locked(")
    funcs.append("uint32_t memory_page_stamp(")
    fixture += "\n".join(block(source, f) for f in funcs)
    fixture += r"""
int main() {
    std::memset(ram, 19, sizeof ram);
    uint8_t b[8]{};
    assert(memory_read_as_submitted(4100,b,8) && b[0]==42 && b[7]==42);
    assert(live_reads==0 && "captured scalar read copied live bytes");
    uint8_t batch[65][8]{};
    std::vector<SubmittedRead> jobs;
    for (auto& item : batch) jobs.push_back({4096,item,8});
    memory_read_as_submitted_many(jobs.data(), jobs.size());
    for (auto& item : batch) assert(item[0]==42 && item[7]==42);
    assert(live_reads==0 && "captured batch read copied live bytes");
    assert(memory_read_as_submitted(8196,b,8) && b[0]==19 && b[7]==19);
    assert(memory_read_as_submitted(8188,b,8) && b[0]==42 && b[3]==42 && b[4]==19);
    assert(memory_read_as_submitted(virt+4100,b,8) && b[0]==42);
    assert(memory_read_as_submitted(virt+4092,b,8) && b[0]==19 && b[4]==42);
    assert(memory_page_stamp(1ull<<39)==0 && memory_page_stamp(4100)==1);
    assert(!memory_read_as_submitted(1ull<<39,b,4));
    const SubmittedRead beyond{1ull<<39,b,4};
    memory_read_as_submitted_many(&beyond,1);
    std::memset(b,0,sizeof b);
    assert(!memory_read_as_submitted(3*4096-3,b,8) && b[2]==19 && b[3]==0);
    std::memset(batch, 0, sizeof batch);
    const SubmittedRead mixed[]={{8196,batch[0],8},{4100,batch[1],8},{8188,batch[2],8}};
    memory_read_as_submitted_many(mixed,3);
    assert(batch[0][0]==19 && batch[1][0]==42 && batch[2][3]==42 && batch[2][4]==19);
    captures_seen=false;
    assert(memory_read_as_submitted(4100,b,8) && b[0]==19);
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = Path(tmp)/"fixture.cpp", Path(tmp)/"fixture.exe"
        src.write_text(fixture)
        subprocess.run([args.cxx, "-std=c++17", str(src), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True, cwd=tmp)
    print("PASS: captured reads avoid live copies, scalar and batched fallbacks preserve bytes")


if __name__ == "__main__":
    main()
