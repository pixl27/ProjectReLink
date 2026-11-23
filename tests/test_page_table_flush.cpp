#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <new>
#include <random>
#include <set>
#include <vector>

static size_t largest;
void* operator new(size_t n) {
    largest = std::max(largest, n);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

struct TableRun { uint64_t first, count, address; };
std::vector<TableRun> table_pending;
bool table_defer;
using VkCommandBuffer = int;
int guest_pages_buffer = 11;
VkCommandBuffer upload_command_buffer() { return 17; }
struct GprofSpan { GprofSpan(int, const char*) {} int items = 0; uint64_t bytes = 0; };
static uint64_t* guest_pages = nullptr;   // (the CPU's table: main's out-of-memory guard writes it at once)
static void logf(const char*, ...) {}
constexpr uint64_t untouched = UINT64_MAX;
static std::vector<uint64_t> actual(180000, untouched);
static size_t last_end;

void vkCmdUpdateBuffer(int cmd, int buffer, size_t off, size_t bytes, const void* src) {
    assert(cmd == 17 && buffer == 11 && bytes > 0 && bytes <= 65536);
    assert(off % 8 == 0 && bytes % 8 == 0 && off >= last_end && off + bytes <= actual.size() * 8);
    std::copy_n(static_cast<const uint64_t*>(src), bytes / 8, actual.begin() + off / 8);
    last_end = off + bytes;
}

/* FLUSH */

void check(std::vector<TableRun> writes, bool limit_scratch = false) {
    table_pending = writes;
    std::fill(actual.begin(), actual.end(), untouched);
    auto expected = actual;
    for (const auto& r : writes)
        for (uint64_t i = 0; i < r.count; ++i)
            expected[r.first + i] = r.address ? r.address + (i << 12) : 0;
    table_defer = true;
    last_end = largest = 0;
    const bool changed = memory_table_flush();
    const size_t scratch = largest;
    assert(changed == !writes.empty() && !table_defer && table_pending.empty());
    assert(actual == expected);
    if (limit_scratch) {
        std::printf("Largest scratch allocation for four overlapping mappings: %zu bytes\n", scratch);
        if (scratch > 128 * 1024) {
            std::fprintf(stderr, "Flushing four mappings allocated %zu bytes in one block\n", scratch);
            std::exit(1);
        }
    }
}

int main() {
    check({});
    check({{97000, 12, 0}, {8190, 16387, 0x101000}, {8192, 1, 0}, {8190, 2, 0},
           {8193, 2, 0x800000}, {24577, 8, 0x10000}, {24577, 8, 0x20000}});
    std::mt19937 rng(7041);
    for (unsigned trial = 0; trial < 50; ++trial) {
        std::vector<TableRun> writes;
        for (unsigned i = 0; i < 128; ++i)
            writes.push_back({rng() % 16384, 1 + rng() % 2048, i % 5 ? uint64_t(rng()) << 12 : 0});
        check(writes);
    }
    check({{100, 150000, 0x500000}, {90000, 40000, 0x800000}, {1234, 120000, 0},
           {8000, 10000, 0xa00000}}, true);
}
