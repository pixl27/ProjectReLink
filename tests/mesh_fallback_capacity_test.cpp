// (hookshot) The compute fallback's records area against the grass's load, with emit_spirv.h's real words and vk_draw.cpp's
// real fallback_buffers_of (cut by tests/test_mesh_fallback_capacity.py into fallback_actual.inc), no game, GPU or driver.
//
// The floor grass draws through two mesh programs a frame, the blades in one and their partner in the other, the same
// workgroups each: 739acd678ae08655 (21 vertex lines, records of 22 words) and a6a688bae3c9b80c (6 vertex lines, 7 words).
// Every program takes its records from the same fixed area, so a 22-word program fits a third as many as a 7-word one: past
// that fit, the epilogue (emit_spirv_mesh.cpp MeshFallbackEpilogue) drops the workgroups that no longer fit, and the blades'
// program draws fewer primitives than its partner. Measured on the fallback with the Grass distance at 4x: 739a held
// at 762528 records a frame, a6a6 at 1002492 to 1023009, and the blades white where the partner drew alone. (The model's
// workgroups are all 288 records, the game's are not: it stops at 762336, the last whole workgroup under the same fit, 762600.)
//
// grass_pair_*: both programs given the same full workgroups of 96 triangles (288 records) up to a load must draw the same
// records, every one: at the load measured at 4x, and at the load the caps let through (~1.29M records, extrapolated
// from the measured one).
// buffers_*: the records buffer that area needs is made, with mma_layout.h's attribute pool and the shared warp's counters
// off and on: fallback_buffers_of refuses a records buffer past its limit, every program's fallback then.
//   mesh_fallback_capacity_test <case>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "shader_recompiler/backend/spirv/emit_spirv.h"

namespace SPV = Shader::Backend::SPIRV;

// (stand-ins for what fallback_buffers_of calls: a buffer is a number, its size remembered)
using VkBuffer = uint64_t;
using VkBufferUsageFlags = uint32_t;
struct Allocation { unsigned id = 0; };
using VmaAllocation = Allocation*;
constexpr VkBuffer VK_NULL_HANDLE = 0;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 1;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_TRANSFER_DST_BIT = 2;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 4;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT = 8;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 16;
constexpr int VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO = 1;
struct VkBufferDeviceAddressInfo { int sType; VkBuffer buffer = 0; };
struct { int device = 1; } dev;
struct Program {
    uint32_t fallback_vertices_per_workgroup = 96 * 3;
    uint32_t fallback_record_words = 22;
    uint32_t fallback_payload_words = 32;
    uint32_t fallback_task_threads = 32;
    bool fallback_task = true;
};

static std::vector<uint64_t> made;   // (each buffer's size, in the order made)
static bool create_buffer(uint64_t size, VkBufferUsageFlags, bool, VkBuffer& buffer, VmaAllocation& allocation, void**) {
    static Allocation allocations[16];
    if (made.size() >= 15) return false;
    made.push_back(size);
    buffer = made.size();
    allocation = &allocations[made.size()];
    return true;
}
[[maybe_unused]] static void destroy_buffer(VkBuffer, VmaAllocation) {}
static uint64_t vkGetBufferDeviceAddress(int, const VkBufferDeviceAddressInfo* info) { return info->buffer << 32; }

#include "fallback_actual.inc"

namespace {
void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

/// The records a program's draw counts after workgroups taking `taken` records each, up to `load` in all (the last one what
/// is left), take theirs from the area one after another: a workgroup's are written and drawn only where all of them fit, as
/// MeshFallbackEpilogue's atomic add and its test (base <= fit && taken <= fit - base)
uint64_t Drawn(uint32_t words, uint64_t load, uint32_t taken) {
    const uint32_t fit{SPV::MeshFallbackFit(words)};
    uint64_t alloc{0}, drawn{0};
    for (uint64_t left = load; left; ) {
        const uint32_t t{static_cast<uint32_t>(left < taken ? left : taken)};
        left -= t;
        const uint64_t base{alloc};
        alloc += t;
        if (base <= fit && t <= fit - base) drawn += t;
    }
    return drawn;
}

constexpr uint32_t kBladeWords = 22;     // (739acd678ae08655: 21 vertex lines and the record's valid word)
constexpr uint32_t kPartnerWords = 7;    // (a6a688bae3c9b80c: 6 vertex lines and the valid word)
constexpr uint32_t kWorkgroup = 96 * 3;  // (a full meshlet: 96 triangles, a record a corner)

void GrassPair(uint64_t load) {
    const uint64_t blades{Drawn(kBladeWords, load, kWorkgroup)}, partner{Drawn(kPartnerWords, load, kWorkgroup)};
    std::printf("  load %llu: the 22-word program draws %llu, the 7-word one %llu (fits %u and %u)\n",
                static_cast<unsigned long long>(load), static_cast<unsigned long long>(blades),
                static_cast<unsigned long long>(partner), SPV::MeshFallbackFit(kBladeWords), SPV::MeshFallbackFit(kPartnerWords));
    Require(partner == load, "the 7-word program dropped records");
    Require(blades == partner, "the 22-word program draws fewer records than its partner: the blades are drawn white");
}

void Buffers(bool pool, bool warp) {
    SPV::mesh_fallback_global_attributes = pool;
    SPV::mesh_fallback_shared_warp = warp;
    for (const uint32_t words : {kBladeWords, kPartnerWords}) {
        Program p;
        p.fallback_record_words = words;
        FallbackBuffers* out = nullptr;
        Require(fallback_buffers_of(p, out) && out && out->records_address, "fallback_buffers_of refused the records buffer");
        const uint64_t area{(uint64_t{SPV::MESH_FALLBACK_HEADER_WORDS} + uint64_t{SPV::MeshFallbackFit(words)} * words) * 4};
        Require(out->records_bytes >= area, "the records buffer is smaller than the area the programs write");
        const uint64_t counters{SPV::MeshFallbackCounters() ? SPV::WARP_COUNTER_BYTES : 0u};
        Require(out->records_bytes >= SPV::MeshFallbackRecordsBytes() + counters,
                "the records buffer has no room for the pool or the counters past the records");
    }
    std::printf("  pool %d, shared warp %d: records buffer %llu bytes\n", pool, warp,
                static_cast<unsigned long long>(made.empty() ? 0 : made.front()));
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string name{argv[1]};
    try {
        if (name == "grass_pair_measured") GrassPair(1023009);   // (a6a6's most a frame at 4x, standing in the field)
        else if (name == "grass_pair_cap") GrassPair(1290000);   // (the most the caps let through: extrapolated)
        else if (name == "buffers") Buffers(false, false);
        else if (name == "buffers_pool") Buffers(true, false);
        else if (name == "buffers_warp") Buffers(false, true);
        else if (name == "buffers_pool_warp") Buffers(true, true);
        else throw std::runtime_error("unknown case");
        std::printf("PASS %s\n", argv[1]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s: %s\n", argv[1], e.what());
        return 1;
    }
}
