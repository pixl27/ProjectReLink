"""Check the actual sparse queue submission with a fake Vulkan queue and semaphore."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from srccut import block

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]

PREFIX = r"""
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

struct { VkDevice device{}; VkQueue queue{}; std::mutex present_mu; } dev;
VkSemaphore sparse_tl{};
std::atomic<uint64_t> sparse_tl_value{0};
unsigned st_sparse_binds = 0, flushes = 0, calls = 0;
double st_sparse_bind_ms = 0;
void queue_flush_locked() { ++flushes; }
void logf(const char*, ...) {}
VkResult vkCreateSemaphore(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* out) {
    *out = (VkSemaphore)(uintptr_t)1;
    return VK_SUCCESS;
}

constexpr VkDeviceSize page = 65536;
std::vector<std::vector<VkSparseMemoryBind>> expected;
bool fail_bind = false;
VkResult vkQueueBindSparse(VkQueue, uint32_t batches, const VkBindSparseInfo* batch, VkFence) {
    ++calls;
    assert(flushes == calls && batches == 1 && batch->bufferBindCount == expected.size());
    assert(batch->signalSemaphoreCount == 1 && *batch->pSignalSemaphores == sparse_tl);
    auto* timeline = static_cast<const VkTimelineSemaphoreSubmitInfo*>(batch->pNext);
    assert(timeline->signalSemaphoreValueCount == 1 && timeline->pSignalSemaphoreValues[0] == sparse_tl_value+1);
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto& info = batch->pBufferBinds[i];
        assert(info.buffer == (VkBuffer)(uintptr_t)(i+1));
        assert(info.bindCount == expected[i].size());
        for (size_t j = 0; j < expected[i].size(); ++j) {
            const auto& got = info.pBinds[j];
            const auto& want = expected[i][j];
            assert(got.resourceOffset == want.resourceOffset && got.size == want.size && got.memory == want.memory);
            assert(got.flags == want.flags && (!want.memory || got.memoryOffset == want.memoryOffset));
        }
    }
    return fail_bind ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
"""

MAIN = r"""
int main() {
    VkDeviceMemory a = (VkDeviceMemory)(uintptr_t)2, b = (VkDeviceMemory)(uintptr_t)3;
    std::vector<VkSparseMemoryBind> p = {
        {0,page,a,0,0}, {page,page,a,page,0}, {2*page,page,a,4*page,0},
        {3*page,page,b,5*page,0}, {5*page,page,b,6*page,0},
        {6*page,page,{},0,0}, {7*page,page,{},0,0}, {8*page,page,a,0,0},
        {9*page,page,a,page,1}   // Synthetic flag mismatch checks that packing preserves flags.
    };
    std::vector<VkSparseMemoryBind> q = {{0,page,a,2*page,0}, {page,page,a,3*page,0}};
    expected = {
        {{0,2*page,a,0,0}, p[2], p[3], p[4], {6*page,2*page,{},0,0}, p[7], p[8]},
        {{0,2*page,a,2*page,0}}
    };
    std::vector<VkSparseBufferMemoryBindInfo> inputs = {
        {(VkBuffer)(uintptr_t)1, (uint32_t)p.size(), p.data()},
        {(VkBuffer)(uintptr_t)2, (uint32_t)q.size(), q.data()}
    };
    uint64_t signaled = 0;
    assert(sparse_bind_infos(inputs,&signaled) && signaled == 1);
    fail_bind = true;
    assert(!sparse_bind_infos(inputs,&signaled) && signaled == 0 && sparse_tl_value == 1);
    assert(inputs[0].pBinds == p.data() && inputs[0].bindCount == p.size());
    assert(inputs[1].pBinds == q.data() && inputs[1].bindCount == q.size());
    assert(p.front().size == page && q.front().size == page);
    fail_bind = false;
    assert(sparse_bind_infos(inputs,&signaled) && signaled == 2 && calls == 3);
    expected = {{p[0]}};
    inputs = {{(VkBuffer)(uintptr_t)1, 1, p.data()}};
    assert(sparse_bind_infos(inputs,&signaled) && signaled == 3);
    inputs.clear();
    assert(sparse_bind_infos(inputs) && calls == 4);
}
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--headers", required=True)
    args = parser.parse_args()
    source = (ROOT / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        cpp = Path(tmp) / "binds.cpp"
        exe = Path(tmp) / "binds.exe"
        cpp.write_text(PREFIX + block(source, "bool sparse_bind_infos(") + MAIN, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
                        "-I" + args.headers, str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)

if __name__ == "__main__":
    main()
