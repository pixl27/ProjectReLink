// SPDX-License-Identifier: GPL-3.0-or-later
// Game-free tests for the load at create letting a loaded program's shader modules go once its noted pipelines are built
// (gpu/vk/vk_progcache.cpp program_cache_drop_loaded, remade by program_cache_complete_graphics; compiled and run by
// tests/test_loaded_modules_drop.py). The stand-ins are tests/linux_pressure_test.cpp's.
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "../gpu/vk/compat/hookshot_env.h"

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)

#if !defined(_WIN32)
#define _fseeki64 fseeko
#define _ftelli64 ftello
#endif

// Minimal stand-ins so vk_progcache.cpp's real functions compile and run without a GPU or game.
using VkShaderStageFlagBits = uint32_t;
using VkShaderModule = uintptr_t;
using VkDescriptorSetLayout = uintptr_t;
using VkPipelineLayout = uintptr_t;
using VkDescriptorType = uint32_t;
using VkPipeline = uintptr_t;
using VkShaderStageFlags = uint32_t;
using VkPrimitiveTopology = uint32_t;

constexpr VkShaderModule VK_NULL_HANDLE = 0;
constexpr VkShaderStageFlagBits VK_SHADER_STAGE_VERTEX_BIT = 0x01;
constexpr VkShaderStageFlagBits VK_SHADER_STAGE_FRAGMENT_BIT = 0x10;
constexpr VkShaderStageFlagBits VK_SHADER_STAGE_COMPUTE_BIT = 0x20;
constexpr VkShaderStageFlagBits VK_SHADER_STAGE_TASK_BIT_EXT = 0x40;
constexpr VkShaderStageFlagBits VK_SHADER_STAGE_MESH_BIT_EXT = 0x80;
constexpr VkPrimitiveTopology VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3;

namespace Shader {
namespace IR {
enum class Attribute : uint64_t { Position = 0 };
}
struct Descriptor {
    uint32_t count = 1;
};
struct Info {
    uint32_t tag = 0;
    std::vector<Descriptor> constant_buffer_descriptors;
    std::vector<Descriptor> storage_buffers_descriptors;
    std::vector<Descriptor> texture_buffer_descriptors;
    std::vector<Descriptor> image_buffer_descriptors;
    std::vector<Descriptor> texture_descriptors;
    std::vector<Descriptor> image_descriptors;
    std::map<IR::Attribute, IR::Attribute> legacy_stores_mapping;
    std::vector<std::vector<uint32_t>> handle_programs;
};
} // namespace Shader

namespace fallback_size {
inline uint32_t spirv_phis(const uint32_t* words, size_t n) { return n ? words[0] : 0; }
inline uint32_t spirv_calls(const uint32_t*, size_t) { return 0; }
}
namespace fallback_lanes {
inline bool spirv_lane_free(const uint32_t*, size_t) { return false; }
}

namespace mkwvk {

struct Dev {
    uintptr_t device = 1;
} dev;

struct UseCounters {
    int64_t loads = 0, load_ns = 0;
    bool background = false;
};
thread_local UseCounters use_counters;
std::atomic<bool> programs_in_play{false};
struct Census {
    std::atomic<int64_t> play_loads{0}, play_load_ns{0};
} census;

void logf(const char*, ...) {}

static size_t next_module_id = 0;
static size_t active_modules = 0;
static size_t make_module_calls = 0;
static uint64_t last_make_module_program = 0;
static bool fail_make_module = false;
static std::unordered_map<VkShaderModule, std::vector<uint32_t>> module_words;

void vkDestroyShaderModule(uintptr_t, VkShaderModule m, const void*) {
    const auto it = module_words.find(m);
    CHECK(it != module_words.end() && active_modules > 0);
    module_words.erase(it);
    --active_modules;
}

#include "cache_types.inc"

std::mutex disk_mu;
FILE* disk_in = nullptr;
std::string disk_path;
std::unordered_map<uint64_t, std::vector<DiskRecord>> disk_records;
std::unordered_map<uint64_t, std::vector<DiskRecord>> disk_taken;
FILE* disk_out = nullptr;
FILE* disk_idx = nullptr;
bool disk_listing = true;
bool disk_on = true;
bool disk_zstd = false;
std::atomic<int64_t> st_cache_hits{0};
constexpr uint32_t kCompressedKind = 16;
constexpr size_t kZstdHead = 20;
constexpr size_t kIdxEntry = 20;

std::mutex shader_mu;
std::unordered_map<uint64_t, std::vector<std::unique_ptr<Program>>> programs;

void disk_init() {}
void programs_load_wait(uint64_t) {}
void module_id_learned(uint64_t, uint64_t, VkShaderModule) {}
void descriptor_arrays_note(const Program&) {}   // (the census of indexed descriptor arrays, vk_programs.cpp: none here)
bool env_on(const char*) { return false; }
#define PROF_EXACT() ((void)0)

size_t ZSTD_compressBound(size_t src_size) { return src_size + 32; }
size_t ZSTD_compress(void* dst, size_t dst_cap, const void* src, size_t src_size, int) {
    if (dst_cap < src_size + 4) return ~size_t{0};
    auto* out = static_cast<uint8_t*>(dst);
    out[0] = 'Z'; out[1] = 'S'; out[2] = 'T'; out[3] = 'D';
    for (size_t i = 0; i < src_size; ++i) out[4 + i] = static_cast<const uint8_t*>(src)[i] ^ 0x5Au;
    return src_size + 4;
}
size_t ZSTD_decompress(void* dst, size_t dst_cap, const void* src, size_t src_size) {
    if (src_size < 4) return ~size_t{0};
    const auto* in = static_cast<const uint8_t*>(src);
    if (in[0] != 'Z' || in[1] != 'S' || in[2] != 'T' || in[3] != 'D' || dst_cap != src_size - 4) return ~size_t{0};
    auto* out = static_cast<uint8_t*>(dst);
    for (size_t i = 0; i < dst_cap; ++i) out[i] = in[4 + i] ^ 0x5Au;
    return dst_cap;
}
bool ZSTD_isError(size_t code) { return code == ~size_t{0}; }

VkShaderModule make_module(const std::vector<uint32_t>& spirv, uint64_t program = 0, bool* /*busy*/ = nullptr) {
    ++make_module_calls;
    last_make_module_program = program;
    if (fail_make_module || spirv.empty()) return VK_NULL_HANDLE;
    const VkShaderModule h = ++next_module_id;
    module_words[h] = spirv;
    ++active_modules;
    return h;
}

#include "cache_functions.inc"

} // namespace mkwvk

namespace {

void test_drop_loaded(const std::string& dir) {
    using namespace mkwvk;
    const std::string path = dir + "/hookshot_programs.bin";
    disk_out = std::fopen(path.c_str(), "wb");
    CHECK(disk_out != nullptr);
    const uint64_t header[2] = {0x31505348u, 0x123456789abcdef0ull};
    std::fwrite(header, 8, 2, disk_out);
    std::fflush(disk_out);
    Program gfx;   // (vertex + fragment)
    gfx.key = 0x2001;
    gfx.stages.resize(2);
    gfx.stages[0].nvn_stage = 0, gfx.stages[0].vk_stage = VK_SHADER_STAGE_VERTEX_BIT;
    gfx.stages[1].nvn_stage = 1, gfx.stages[1].vk_stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    const std::vector<uint32_t> vs_spv(64, 0x11112222u), fs_spv(96, 0x33334444u), cs_spv(128, 0x55556666u);
    program_cache_save(gfx, {vs_spv, fs_spv});
    Program fb;   // (a mesh fallback program: compute + vertex + fragment)
    fb.key = 0x2002, fb.fallback = true;
    fb.stages.resize(3);
    fb.stages[0].nvn_stage = 0, fb.stages[0].vk_stage = VK_SHADER_STAGE_COMPUTE_BIT;
    fb.stages[1].nvn_stage = 0, fb.stages[1].vk_stage = VK_SHADER_STAGE_VERTEX_BIT;
    fb.stages[2].nvn_stage = 1, fb.stages[2].vk_stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    program_cache_save(fb, {cs_spv, vs_spv, fs_spv});
    Program comp;
    comp.key = 0x2003, comp.compute = true;
    comp.stages.resize(1);
    comp.stages[0].nvn_stage = 5, comp.stages[0].vk_stage = VK_SHADER_STAGE_COMPUTE_BIT;
    program_cache_save(comp, {cs_spv});
    std::fclose(disk_out);
    disk_out = nullptr;
    disk_path = path;
    disk_in = std::fopen(path.c_str(), "rb");
    CHECK(disk_in != nullptr);
    // (read as the load at create reads them: into the job's own list, not `programs` yet)
    std::vector<std::unique_ptr<Program>> made;
    for (uint64_t k : {0x2001ull, 0x2002ull, 0x2003ull}) program_cache_read(disk_in, &disk_mu, k, disk_records[k], made);
    CHECK(made.size() == 3 && active_modules == 6);
    Program& g = *made[0];
    Program& f = *made[1];
    Program& c = *made[2];
    CHECK(g.key == 0x2001 && f.key == 0x2002 && c.key == 0x2003);
    CHECK(program_cache_remade() == 0);

    // a fallback program and a compute program keep theirs (their compute stages are other pipelines' modules)
    CHECK(program_cache_drop_loaded(f) == 0 && program_cache_drop_loaded(c) == 0 && active_modules == 6);
    // a stage that cannot be remade from the file (translated in this run, or compressed: code_words 0) keeps the program's
    {
        const uint64_t w = g.stages[1].code_words;
        g.stages[1].code_words = 0;
        CHECK(program_cache_drop_loaded(g) == 0 && active_modules == 6 && !program_modules_dropped(g));
        g.stages[1].code_words = w;
    }
    // the graphics program whose noted pipelines are built: both modules go, and no pipeline is built from it until remade
    CHECK(program_cache_drop_loaded(g) == 2);
    CHECK(active_modules == 4 && g.stages[0].module == VK_NULL_HANDLE && g.stages[1].module == VK_NULL_HANDLE);
    CHECK(program_modules_dropped(g));
    // a second program of the same key and variant (a borrowed one) keeps its own, and is not taken for the one let go
    {
        Program copy;
        copy.key = g.key, copy.variant = g.variant;
        copy.stages = g.stages;
        copy.stages[0].module = make_module(vs_spv, copy.key);
        CHECK(!program_modules_dropped(copy));
        CHECK(program_cache_drop_loaded(copy) == 0 && copy.stages[0].module != VK_NULL_HANDLE);
        vkDestroyShaderModule(0, copy.stages[0].module, nullptr);
    }
    // a new pipeline state in play: the worker remakes them from the file, the very SPIR-V, and counts it
    last_make_module_program = 0;
    const size_t calls = make_module_calls;
    CHECK(program_cache_complete_graphics(g));
    CHECK(last_make_module_program == 0x2001 && active_modules == 6 && !program_modules_dropped(g));
    CHECK(make_module_calls == calls + 2);   // (its two stages, no more)
    CHECK(module_words[g.stages[0].module] == vs_spv && module_words[g.stages[1].module] == fs_spv);
    CHECK(program_cache_remade() == 1);
    // and a second call has nothing to make
    CHECK(program_cache_complete_graphics(g) && program_cache_remade() == 1 && active_modules == 6);
    CHECK(make_module_calls == calls + 2);
    // let go again after its new pipeline: fine, and remade again
    CHECK(program_cache_drop_loaded(g) == 2 && active_modules == 4);
    CHECK(program_cache_complete_graphics(g) && program_cache_remade() == 2 && active_modules == 6);
    std::fclose(disk_in);
    disk_in = nullptr;
    disk_path.clear();
}

} // namespace

int main(int argc, char** argv) {
    CHECK(argc == 2);
    const std::string arg = argv[1];
    if (arg == "--switch") {   // (RX_VK_DROP_LOADED_MODULES as the process has it: 1 on, 0 off)
        std::printf("%d\n", (int)mkwvk::drop_loaded_modules_on());
        return 0;
    }
    test_drop_loaded(arg);
    std::printf("PASS: a loaded program's shader modules let go once its noted pipelines are built, and remade from the file\n");
    return 0;
}
