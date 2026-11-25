// Draws and dispatches: a program's pipeline layout (its stages' descriptors in Eden's order, as the SPIR-V back end
// numbered them, one push-descriptor set when they fit), graphics pipelines (whatever Vulkan 1.3 and
// VK_EXT_extended_dynamic_state3 take as dynamic state is dynamic; the rest is the key), each draw's descriptors
// (driver banks and c1 from the recording's stream, user uniform and storage buffers in the imported pools, textures
// and storage images as views of vk_texture.cpp's images, texel buffers), passes kept open while the targets stay,
// and the draws themselves.
//
// Pipelines persist in a VkPipelineCache in the game folder's cache/ (hookshot_vk_pipelines.bin).
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <tuple>
#include <atomic>
#include <chrono>
#include <memory>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <deque>
#include <exception>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <list>
#include <thread>
#include <setjmp.h>
#include <signal.h>
#if defined(__GLIBC__)
#include <malloc.h>   // (malloc_trim: compute_gate.h trims)
#endif

#include "shader_recompiler/backend/spirv/emit_spirv.h"

#include "vk_paths.h"
#include "mesh_capture.h"
#include "host_shaders/mesh_storage_compare_comp_spv.h"
#include "vk_progress.h"   // (PrepareWork: prepare_run's count of the programs, each at its cost)
#include "pipeline_cache_file.h"
#include "vk_draw.h"
#ifdef TRACY_ENABLE
#include "../../host/hos_tracy.h"   // (HOOKSHOT_TRACY: the profiler's zones; nothing in any other build)
#endif
#include "draw_key.h"     // (the draw path's questions answered from their arguments alone: pipe_same, dyn_same, ...)
#include "flat_map.h"
#include "vk_menu.h"          // (opts.anisotropy: the Anisotropic filtering row; resolution_scale)
#include "upscaler_rows.h"    // (upscale_lod_bias: which upscalers draw the 3D smaller)
#include "vk_framegen.h"   // (hookshot: the gameplay upscale's output, for the item wheel's backdrop)
#include "dlss_actor.h"
#include "hidden_mesh.h"      // (hookshot: a hidden mesh's depth draws left out, hidden_depth_draw)
#include "vk_temporal.h"
#include "vk_framegen.h"   // (framegen_hud_before: a gameplay upscale's picture, for FSR 3.1 frame interpolation's HUD mark)
#include "vk_fsr3fi.h"     // (frames_wanted)
#include "vk_fsr3.h"          // (AMD FSR 3.1 in the gameplay upscale's place)
#include "vk_fsr3_inputs.h"
#include "reactive_policy.h"
#include "upscaler_rows.h"
#include "linux_pressure.h"
#include "pcache_stamp.h"
#include "pcache_load.h"     // (hookshot: whether the pipeline cache file is read)
#include "nvidia_cache_limit.h"
#ifdef _WIN32
#include <io.h>              // (_chsize_s: trim incomplete pipeline journal entries)
#else
#include <unistd.h>          // (sysconf: the RAM, pcache_load_off)
#endif
#include "vk_reactive.h"   // (the upscalers' reactive mask: the scene copied before its transparent draws)
#include "vk_dump.h"
#include "vk_gprof.h"
#include "vk_jobs.h"
#include "vk_shader.h"
#include "vk_texture.h"
#include "vk_ledger.h"   // (RX_VK_LEDGER)
#include "ubo_memo.h"    // (HL_UBO_MEMO)
#include "predict_limits.h"   // (what the prediction of pipelines holds itself to)
#include "movie_upscale_policy.h"
#include "skip_code.h"
#include "driver_faults.h"
#include "driver_cache_env.h"   // (the drivers' own caches as the environment chooses them, Linux)
#include "fallback_size_policy.h"
#include "fallback_lanes.h"
#include "translate_wait.h"
#include "precompile_save.h"   // (the pipeline cache written before the title after a long precompile)
#include "bounded_wait.h"   // (the load-at-create start's wait for its builds, bounded by their progress)
#include "start_steps.h"    // (a first start's warm and load at create on the bar as one count)
#include "quick_pipe.h"

#include "vk_builtin_spv.h"   // (RX_VK_SHADER_DIR)
#include "shaders/clear_f_frag_spv.h"
#include "shaders/clear_i_frag_spv.h"
#include "shaders/clear_u_frag_spv.h"
#include "shaders/clear_vert_spv.h"
#include "shaders/taau_upscale_comp_spv.h"
#include "shaders/taau_upscale_rgba_comp_spv.h"
#include "shaders/shadow_stab_comp_spv.h"
#include "shaders/movie_upscale_comp_spv.h"
#include "shader_recompiler/backend/spirv/mma_layout.h"
#ifndef HOOKSHOT_ENV_H   // (env_on: the switches' one rule)
#include "hookshot_env.h"
#endif
#include "vk_programs.h"
#include "vk_pipelines.h"
#include "warm_bounded.h"
#include "crash_markers.h"   // (the files that count the starts that crashed in a step: a close puts them back)
#include "compute_gate.h"
#include "intel_compiles.h"   // (intel_prepare_workers: the preparation's threads where Intel's compiler takes them side by side)
#include "compute_pin.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

namespace mkwvk {
// (EvilOOT, RX_VK_DRAW_STATS: the pointer table's regions, vk_bank.cpp)
struct PointerEntry { uint64_t addr, size; };
const PointerEntry* pointer_table(const NvnState& s, int stage);
uint32_t stage_cbuf_word(const NvnState& s, const Stage& st, uint32_t index, uint32_t offset);   // (vk_shader.cpp)
uint64_t sampler_registrations_generation();   // (vk_bank.cpp: sampler_for's cache)
namespace {
// ---- (hookshot) this file's per-thread objects that need a constructor or a destructor ----
// With llvm-mingw every use of such a thread_local calls the file's TLS init function, which saves registers, finds the
// thread's TLS block and checks a guard byte however long ago the thread set them up: ~2.4% of the replay thread on a
// Ryzen 9 3900X in Kokiri Forest (HOS_SAMPLE). Each one is reached through a plain thread_local pointer instead, which needs
// no init function: TLS_OBJECT(name, type) makes name(), the calling thread's object, made on its first use and deleted when
// the thread ends. Only the list that owns them has a destructor, and only a thread's first use of each object touches it.
// A new per-thread object of a type with a constructor or destructor belongs here too (tests/test_tls_objects.py checks).
// (They go when the thread ends, with the list: no other thread_local's destructor may use them.)
struct TlsOwned {
    void* object;
    void** slot;
    void (*del)(void*);
};
std::vector<TlsOwned>& tls_owned() {
    struct List {
        std::vector<TlsOwned> v;
        ~List() {
            for (auto it = v.rbegin(); it != v.rend(); ++it) it->del(it->object), *it->slot = nullptr;
        }
    };
    static thread_local List list;
    return list.v;
}
template <class T>
T& tls_make(T*& slot) {
    slot = new T();   // (value-initialized: zeroed as a thread_local's static storage was)
    tls_owned().push_back({slot, reinterpret_cast<void**>(&slot), [](void* p) { delete static_cast<T*>(p); }});
    return *slot;
}
#define TLS_OBJECT(name, ...)                                                         \
    thread_local __VA_ARGS__* name##_tls = nullptr;                                   \
    inline __VA_ARGS__& name() {                                                      \
        __VA_ARGS__* p = name##_tls;                                                  \
        return p ? *p : tls_make(name##_tls);                                         \
    }

// ---- (hookshot) RX_VK_PROF_FINE=1: where a draw's time goes, finer than the frame line's phases, and what each draw does ----
// Off, each site is a test of one constant. On, a timed site reads the clock twice (~15 ns; the frame line's phases grow by the same, so
// compare a run with it on to another with it on) and the sums are logged with the frame statistics, "a frame (fine)": the times in ms a
// frame, the counts a draw. FT_*: total preparation, descriptor and command phases, plus the sections of the descriptor build
// (banks, c1, user uniform buffers, storage buffers, textures) and of the commands (hazards and the pass, pipeline bind,
// descriptor push, push constants, dynamic state, vertex and index binds, the draw). A total minus its named sections exposes
// loop, lookup and diagnostic overhead that was previously outside every fine timer.
enum FineT { FT_PREP, FT_TARGETS, FT_PROGRAM_PICK, FT_PIPE_LOOKUP, FT_DESC_TOTAL, FT_BANK, FT_C1, FT_UBO, FT_STORAGE, FT_TEX,
             FT_CMD_TOTAL, FT_HAZARD, FT_CMD_PREPASS, FT_CMD_TRACK, FT_BINDPL, FT_BINDDESC, FT_PUSHCALL, FT_PUSHC, FT_DYN,
             FT_CMD_POSTSTATE, FT_MESH_DRAW, FT_MESH_COUNT_BIND, FT_MESH_ARGS_BIND, FT_MESH_PUSH, FT_MESH_CALL, FT_VERTEX, FT_DRAWCALL,
             FT_TEXSTORE, FT_TEXHANDLE, FT_TEXVIEW, FT_TEXFP, FT_TEXMISS, FT_COUNT };
const char* const kFineT[] = {"draw preparation", "targets+types", "program pick", "pipeline lookup", "descriptor total", "banks",
                                      "c1", "user uniform buffers", "storage buffers", "textures", "command total", "hazards+pass",
                                      "command pre-pass", "sampled-target tracking", "bind pipeline", "bind descriptors",
                                      "of it the push call", "push constants", "dynamic state", "post-state setup",
                                      "mesh draw setup+call", "mesh count bind", "mesh arguments bind", "mesh first push", "mesh command",
                                      "vertex+index binds", "draw call", "of tex: storage-pointer test", "of tex: handle",
                                      "of tex: view", "of tex: fingerprint",
                                      "of textures: slot misses (view and sampler lookups)"};
enum FineC { FC_DRAWS, FC_BUILDS, FC_DESCS, FC_WRITES, FC_BANK_CALLS, FC_BANK_GEN, FC_BANK_FILLS, FC_BANK_FILL_BYTES, FC_BANK_SAME,
             FC_BANK_NEW, FC_BANK_NEW_BYTES, FC_C1_CALLS, FC_UBO_CALLS, FC_UBO_OVERLAY, FC_UBO_STREAM, FC_PUSH_FULL, FC_PUSH_DELTA,
             FC_PUSH_NONE, FC_PUSH_WRITES, FC_PUSH_DESCS, FC_BIND_PIPE, FC_DYN_FAST, FC_DYN_FULL, FC_DYN_CALLS, FC_VBIND, FC_IBIND,
             FC_PUSHC, FC_PIPE_FAST, FC_PIPE_MAP, FC_POOL_SETS, FC_POOL_DESCS, FC_SSBO_DESCS, FC_TEX_HIT, FC_TEX_MISS,
             FC_MESH_DRAWS, FC_MESH_INDIRECT, FC_MESH_COUNTED, FC_MESH_FIRST, FC_COUNT };
const char* const kFineC[] = {"draws", "descriptor builds", "descriptors", "descriptor writes", "bank calls",
                                      "bank kept by its stage's generation", "bank fills", "bank bytes filled", "bank refills equal",
                                      "bank copies into the stream", "bank bytes copied", "c1 calls", "user uniform buffer calls",
                                      "uniform ranges copied with an update's overlay", "uniform ranges copied into the stream",
                                      "full pushes", "pushes of changed writes only", "pushes of nothing (all as pushed)",
                                      "writes pushed", "descriptors pushed", "pipeline binds", "dynamic state from the draw before",
                                      "dynamic state worked out", "dynamic state commands", "vertex stream binds", "index buffer binds",
                                      "push constants", "pipeline keys from the fast pair", "pipeline keys from the thread's map",
                                      "sets allocated", "descriptors in them", "storage buffer descriptors built",
                                      "texture slots kept (a handle's view, tex and sampler)", "texture slots resolved again",
                                      "mesh draws", "indirect mesh draws", "counted mesh draws", "mesh draws with a nonzero first task"};
static_assert(std::size(kFineT) == FT_COUNT && std::size(kFineC) == FC_COUNT, "every section and count is named");
const bool g_fine = env_on("RX_VK_PROF_FINE");
int64_t fine_ns[FT_COUNT], fine_n[FC_COUNT];
struct FineScope {
    int ph;
    int64_t t0;
    explicit FineScope(int p) : ph(p), t0(g_fine ? prof_now() : 0) {}
    void finish() {
        if (g_fine && t0) fine_ns[ph] += prof_now() - t0, t0 = 0;
    }
    ~FineScope() { finish(); }
    FineScope(const FineScope&) = delete;
    FineScope& operator=(const FineScope&) = delete;
};
#define FINE_SCOPE(ph) FineScope fine_scope_##ph(ph)
#define FINE_N(c, n) do { if (g_fine) fine_n[c] += (n); } while (0)
#define FINE_X(c, n) (g_fine ? (void)(fine_n[c] += (n)) : (void)0)   // (the same, as an expression)
/// The sums since the last report, as a frame's times and a draw's counts (the caller: every 120 frames)
void fine_report(double frames) {
    if (!g_fine) return;
    static int64_t last_ns[FT_COUNT], last_n[FC_COUNT];
    char line[1400];
    int k = snprintf(line, sizeof line, "vk: a frame (fine): ms:");
    for (int i = 0; i < FT_COUNT; i++) {
        k += snprintf(line + k, sizeof line - k, " %s %.2f%s", kFineT[i], (fine_ns[i] - last_ns[i]) / frames / 1e6, i + 1 < FT_COUNT ? "," : "");
        last_ns[i] = fine_ns[i];
    }
    logf("%s", line);
    const double draws = std::max<double>(1.0, (double)(fine_n[FC_DRAWS] - last_n[FC_DRAWS]));
    k = snprintf(line, sizeof line, "vk: a frame (fine): %.0f draws; a draw:", draws / frames);
    for (int i = 1; i < FC_COUNT; i++) {
        k += snprintf(line + k, sizeof line - k, " %s %.2f%s", kFineC[i], (fine_n[i] - last_n[i]) / draws, i + 1 < FC_COUNT ? "," : "");
        if (k > 700) {
            logf("%s", line);
            k = snprintf(line, sizeof line, "vk: a frame (fine): a draw:");
        }
    }
    logf("%s", line);
    for (int i = 0; i < FC_COUNT; i++) last_n[i] = fine_n[i];
}

/// 32 lanes, where the stage can be held to it (workgroup_x: a compute stage's width, which full subgroups must divide)
/// (invocations: its whole workgroup, x * y * z, which 32-lane subgroups must cover: pins_32)
void pin_subgroup(VkPipelineShaderStageCreateInfo& st, uint32_t workgroup_x = 0, uint64_t invocations = 64) {
    static const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo size = [] {
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo r{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        r.requiredSubgroupSize = 32;
        return r;
    }();
    if (!pins_32(st.stage, invocations)) return;
    st.pNext = &size;
    if (workgroup_x && !(workgroup_x % 32) && dev.vk13.computeFullSubgroups)
        st.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
}

/// (hookshot) What a compute stage has that can tell the lanes it runs on (compute_pin.h): its Shader::Info's flags, which
/// every way a program is made keeps (translated, read from the program cache, read from it without its code), and the
/// program's shared memory
compute_pin::Uses compute_stage_uses(const Stage& st, uint32_t shared_memory) {
    compute_pin::Uses u;
    u.vote = st.info.uses_subgroup_vote, u.mask = st.info.uses_subgroup_mask;
    u.invocation_id = st.info.uses_subgroup_invocation_id, u.shuffles = st.info.uses_subgroup_shuffles;
    u.quad_shuffles = st.info.uses_quad_shuffles, u.matrix = st.info.uses_cooperative_matrix, u.fswzadd = st.info.uses_fswzadd;
    u.shared_memory = shared_memory;
    return u;
}
compute_pin::Uses compute_program_uses(const Program& p, const Stage& st) {
    compute_pin::Uses u = compute_stage_uses(st, p.shared_memory);
    u.invocations = (uint64_t)p.workgroup[0] * p.workgroup[1] * p.workgroup[2];
    return u;
}
/// (hookshot) Whether a game compute program's stage is held to 32 lanes: every one, but on Mesa's Intel driver only those that
/// can tell their lanes (compute_pin.h: held to 32 lanes that compiler takes minutes and gigabytes over a large program, and
/// the preparation never ended on an Iris Xe). RX_VK_COMPUTE_PIN=all or lanes says so on any GPU
bool compute_pin_needed(const Program& p) {
    static const compute_pin::Mode mode =
        compute_pin::mode_for(dev.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, getenv("RX_VK_COMPUTE_PIN"));
    const bool known = !p.stages.empty();
    return compute_pin::needed(mode, known, known ? compute_program_uses(p, p.stages[0]) : compute_pin::Uses{});
}
/// (hookshot) The program whose compute pipeline this thread described last (compute_pipeline_info, the mesh fallback's stages):
/// what a compile under way is logged by (ComputeTurn)
struct ComputeDescribed {
    uint64_t key = 0;
    compute_pin::Uses uses;
    uint32_t workgroup[3] = {0, 0, 0};
    bool fallback = false;
    uint32_t phis = 0;   // (its SPIR-V's: Stage::phis, for compute_refused)
};
thread_local ComputeDescribed compute_described;
void compute_describe(const Program& p, const Stage& st) {
    compute_described.key = p.key, compute_described.uses = compute_program_uses(p, st);
    for (int i = 0; i < 3; i++) compute_described.workgroup[i] = p.workgroup[i];
    compute_described.fallback = p.fallback;
    compute_described.phis = st.phis;
}
/// (hookshot) A game compute program's stage: held to 32 lanes where the GPU can and the program needs it, and described for
/// the log. Every maker of a compute program's pipeline comes through here (compute_pipeline_info), so that each asks the
/// caches for the pipeline the others built
void pin_compute(VkPipelineShaderStageCreateInfo& st, const Program& p) {
    if (compute_pin_needed(p)) pin_subgroup(st, p.workgroup[0], uint64_t(p.workgroup[0]) * p.workgroup[1] * p.workgroup[2]);
    compute_describe(p, p.stages[0]);
}

// ---- environment switches, read once ----
struct Switches {
    bool no_depth = env_on("RX_VK_NODEPTH");
    bool no_cull = env_on("RX_VK_NOCULL");
    bool ignore_discard = env_on("RX_VK_IGNORE_DISCARD");
    bool log_draws = env_on("RX_VK_LOG_DRAWS");
};
const Switches& sw() {
    static const Switches s;
    return s;
}

// (EvilOOT, RX_VK_DRAW_STATS) the storage buffers bound in the measured frame, with their programs: who writes what
struct StatsBind {
    uint64_t prog, addr, size;
    int stage;
    bool written, table;
    uint32_t cbuf = 0, off = 0;
    uint64_t raw = 0;
    uint64_t vkbuf = 0, vkoff = 0;   // (the Vulkan buffer and offset the binding resolved to: two pools over one range show here)
};
std::mutex stats_bind_mu;
std::vector<StatsBind> stats_binds;
// (and the GPU addresses the small storage buffers of each dispatch hold: pointers its program may follow)
struct StatsPtr { uint64_t prog, in_buffer, value; };
std::vector<StatsPtr> stats_ptrs;
std::vector<StatsPtr> stats_ptrs2;   // (second level: in the pointer table's regions, covered by nothing the dispatch has)
// (RX_VK_DRAW_STATS_PROGS: those programs' uniform and storage buffers as the dispatch saw them, hex, once each)
std::vector<std::string> stats_dumps;
// (and as the GPU held them after the wanted draws: 32 slots of up to 1 MB each; StatsQuery::gpu_dump)
constexpr uint32_t kGpuDumpSlot = 1u << 20, kGpuDumpSlots = 32;
VkBuffer stats_gpu = VK_NULL_HANDLE;
VmaAllocation stats_gpu_mem = nullptr;
uint8_t* stats_gpu_map = nullptr;
struct GpuDumpRec { uint64_t prog, addr; uint32_t slot, bytes; int stage, index; };
std::vector<GpuDumpRec> stats_gpu_recs;
// (RX_VK_MISS_LOG=1: each program's slice of one host-visible buffer, its c0+0xff0 descriptor: the global memory
// accesses no binding held, counted, the first 63 addresses kept; translate_program.cpp, spirv_emit_context.cpp)
// (RX_VK_BLOCK_PROBE: 16 KB slices, the per-block entry counters at word 128 + block, emit_spirv.cpp Traverse)
const uint32_t kMissSlice = getenv("RX_VK_BLOCK_PROBE") ? 16384u : 512u, kMissSlices = getenv("RX_VK_BLOCK_PROBE") ? 1024u : 4096u;
VkBuffer miss_buf = VK_NULL_HANDLE;
VmaAllocation miss_mem = nullptr;
uint32_t* miss_map = nullptr;
std::mutex miss_mu;
std::unordered_map<uint64_t, uint32_t> miss_slot;   // program key -> slice
VkDescriptorBufferInfo miss_binding(uint64_t prog) {
    std::lock_guard lk(miss_mu);
    if (!miss_buf) {
        void* m = nullptr;
        if (!create_buffer((uint64_t)kMissSlice * kMissSlices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, miss_buf, miss_mem, &m))
            return {VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
        miss_map = (uint32_t*)m;
        std::memset(miss_map, 0, (size_t)kMissSlice * kMissSlices);
    }
    auto it = miss_slot.find(prog);
    if (it == miss_slot.end()) {
        if (miss_slot.size() >= kMissSlices) return {miss_buf, 0, kMissSlice};   // (full: all share the first)
        it = miss_slot.emplace(prog, (uint32_t)miss_slot.size()).first;
    }
    return {miss_buf, (VkDeviceSize)it->second * kMissSlice, kMissSlice};
}
/// (RX_VK_DRAW_STATS_PROGS) a stage's uniform and storage buffers as recorded, hex (first 1 KB each), once per program
void stats_dump_stage(uint64_t prog, const NvnState& s, int stage) {
    static std::mutex dm;
    static std::unordered_set<uint64_t> done;
    static int64_t done_frame = -1;   // (once per program per measured frame)
    {
        std::lock_guard lk(dm);
        const int64_t f = (int64_t)texture_frame_no();
        if (f != done_frame) done.clear(), done_frame = f;
        if (!done.insert(prog ^ ((uint64_t)stage << 56)).second) return;
    }
    const auto dump = [&](const char* kind, int slot, const nvn_p_bind_buf& b) {
        if (!b.addr || !b.size) return;
        // (uniform buffers whole up to 2 KB and with the stream's inline updates, ub_read: what the shader reads)
        const bool ubo = kind[0] == 'u';
        // (RX_VK_DUMP_LIVE=1: storage buffers as guest memory holds them now, not as submitted: what the CPU wrote since)
        static const bool live = env_on("RX_VK_DUMP_LIVE");
        const auto read = [&](uint64_t a, void* dst, uint64_t bytes) {
            return ubo ? ub_read(a, dst, bytes) : live ? host->gpu_read(a, dst, bytes) : memory_read_as_submitted(a, dst, bytes);
        };
        const uint64_t n = std::min<uint64_t>(b.size, ubo ? 0x800 : 0x400) & ~15ull;
        std::vector<uint32_t> w(n / 4);
        if (w.empty() || !read(b.addr, w.data(), n)) return;
        uint64_t nonzero = 0, total = 0;   // (over the whole binding, up to 16 MB: is anything in it at all)
        uint64_t first_nz = UINT64_MAX;    // (and where the data past the first 64 bytes starts, 64 words of it shown)
        std::vector<uint32_t> at_nz;
        {
            const uint64_t all = std::min<uint64_t>(b.size, 16ull << 20) & ~3ull;
            std::vector<uint32_t> big(all / 4);
            if (!big.empty() && read(b.addr, big.data(), all))
                for (size_t i = 0; i < big.size(); i++) {
                    nonzero += big[i] != 0, total++;
                    if (big[i] && i >= 16 && first_nz == UINT64_MAX) first_nz = i * 4;
                }
            if (first_nz != UINT64_MAX && first_nz >= 0x400)
                at_nz.assign(big.begin() + first_nz / 4, big.begin() + std::min<size_t>(big.size(), first_nz / 4 + 64));
        }
        std::string text;
        char line[240];
        snprintf(line, sizeof line, "%016llx s%d %s%d at %#llx size %#llx (game memory: %llu of %llu words non-zero; first past "
                 "+0x40 at %s%llx):\n", (unsigned long long)prog, stage, kind, slot, (unsigned long long)b.addr,
                 (unsigned long long)b.size, (unsigned long long)nonzero, (unsigned long long)total, first_nz == UINT64_MAX ? "none " : "+",
                 (unsigned long long)(first_nz == UINT64_MAX ? 0 : first_nz));
        text += line;
        for (size_t i = 0; i < w.size(); i += 4) {
            snprintf(line, sizeof line, "  +%03zx: %08x %08x %08x %08x\n", i * 4, w[i], w[i + 1], w[i + 2], w[i + 3]);
            text += line;
        }
        for (size_t i = 0; i + 4 <= at_nz.size(); i += 4) {
            snprintf(line, sizeof line, "  +%06llx: %08x %08x %08x %08x\n", (unsigned long long)(first_nz + i * 4), at_nz[i], at_nz[i + 1],
                     at_nz[i + 2], at_nz[i + 3]);
            text += line;
        }
        std::lock_guard bl(stats_bind_mu);
        if (stats_dumps.size() < 256) stats_dumps.push_back(text);
    };
    for (int i = 0; i < 16; i++) dump("ubo", i, s.ubo[stage][i]);
    for (int i = 0; i < 16; i++) dump("ssbo", i, s.ssbo[stage][i]);
}
bool stats_prog_wanted(uint64_t key) {
    static const std::vector<uint64_t> want = [] {
        std::vector<uint64_t> v;
        for (const char* q = getenv("RX_VK_DRAW_STATS_PROGS"); q && *q;) {
            char* end;
            const uint64_t k = strtoull(q, &end, 16);
            if (end == q) break;
            v.push_back(k);
            q = *end ? end + 1 : end;
        }
        return v;
    }();
    return std::find(want.begin(), want.end(), key) != want.end();
}
bool stats_frame_now();   // (below: the frame being measured is the current one)
/// (EvilOOT) RX_VK_SKIP_FILE=<file>: the draws of the programs listed in it (hex keys, comma or newline separated) left out,
/// the file read again whenever it changes (checked every 300 ms): a bisection in one run, from the scene being looked at
bool skip_live(uint64_t key) {
    static const char* path = getenv("RX_VK_SKIP_FILE");
    if (!path || !*path) return false;
    static std::mutex mu;
    static std::vector<uint64_t> keys;
    static std::atomic<int64_t> next_check{0};
    static int64_t last_size = -1, last_time = -1;
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard lk(mu);
    if (now >= next_check.load(std::memory_order_relaxed)) {
        next_check.store(now + 300, std::memory_order_relaxed);
        // (size and write time, portably: the Linux build has no GetFileAttributesExA)
        std::error_code ec;
        const auto fsize = std::filesystem::file_size(path, ec);
        const auto ftime = ec ? std::filesystem::file_time_type{} : std::filesystem::last_write_time(path, ec);
        if (!ec) {
            const int64_t size = (int64_t)fsize;
            const int64_t time = (int64_t)ftime.time_since_epoch().count();
            if (size != last_size || time != last_time) {
                last_size = size, last_time = time;
                keys.clear();
                if (FILE* f = fopen(path, "rb")) {
                    std::string text;
                    char buf[4096];
                    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
                    fclose(f);
                    for (const char* q = text.c_str(); *q;) {
                        char* e;
                        const uint64_t k = strtoull(q, &e, 16);
                        if (e == q) { q++; continue; }
                        keys.push_back(k);
                        q = e;
                    }
                    std::sort(keys.begin(), keys.end());
                }
                logf("vk: RX_VK_SKIP_FILE: %zu programs left out now", keys.size());
            }
        } else if (!keys.empty()) {
            keys.clear(), last_size = last_time = -1;
        }
    }
    return std::binary_search(keys.begin(), keys.end(), key);
}
void stats_note_bind(uint64_t prog, int stage, uint64_t addr, uint64_t size, bool written, bool table, uint32_t cbuf = 0,
                     uint32_t off = 0, uint64_t raw = 0, uint64_t vkbuf = 0, uint64_t vkoff = 0) {
    if (!stats_frame_now()) return;
    std::lock_guard lk(stats_bind_mu);
    if (stats_binds.size() < 200000) stats_binds.push_back({prog, addr, size, stage, written, table, cbuf, off, raw, vkbuf, vkoff});
}

/// The recording being built (each recording thread's own)
thread_local VkCommandBuffer rec_cmd = VK_NULL_HANDLE;
thread_local uint64_t rec_value = 0;

// ---- per recording: a descriptor pool and an upload stream (reset when the recording is reused) ----
struct RecordingRes {
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> extra;   // (pools added when the first ran out)
    // (hookshot) what the recording took of its pools: sets, and descriptors by make_pool's types (ub, sb, cis, si, utb, stb): the most
    // a recording took is logged with the streams, which is what make_pool's sizes can be set from
    uint32_t pool_sets = 0, pool_descs[6] = {};
    uint32_t pool_shift = 0;   // (the slot's pool's size: make_pool's shift, pool_shift_first at first)
    // (the stream in use: the first, or one a recording that outgrew it was moved to. Descriptors written before the move hold
    // the old buffer's range, so the old ones stay until the slot's next recording, which the GPU's use of this one is done by)
    VkBuffer stream = VK_NULL_HANDLE;
    VmaAllocation stream_alloc = nullptr;
    uint8_t* mapped = nullptr;
    uint64_t used = 0, size = 0;
    struct Stream { VkBuffer buffer; VmaAllocation alloc; uint8_t* mapped; uint64_t size; };
    std::vector<Stream> streams;   // (the first, then the ones made on demand: kept for the slot's next recordings)
    size_t si = 0;                 // (the one in use)
    uint64_t before = 0;           // (what the recording used of the streams before this one)
    uint32_t idle = 0;             // (recordings of the slot that stayed in the first stream)
    // (flat (flat_map.h): an unordered_map put a malloc into every c1 upload kept, and a free into the next recording's clear)
    FlatMap<uintptr_t, std::pair<VkBuffer, uint64_t>> c1_at;   // (c1 of a stage, by the stage's address: uploaded once a recording)
    uint64_t ub_snap_epoch = 0;   // (this recording's number among all: its uniform copies in ub_snaps)
};
// (a recording's first stream: one for each of the kRecordings slots. RX_VK_STREAM_MB=<n>: that size, 1 by default; a recording
// that needs more is given a bigger stream for the rest of it; the most a recording used is logged: "the recording streams".
// (hookshot) 1 MB, not 2: the most a recording held was 339-381 KB in Castle Town and Hyrule Field at 1080p and 1440p, and the
// 48 streams are host-visible VRAM where the card has resizable BAR (a stream's uniforms, vertices and indices): 48 MB given
// back to the textures)
const uint64_t kStream = (getenv("RX_VK_STREAM_MB") ? (uint64_t)std::max(1, atoi(getenv("RX_VK_STREAM_MB"))) : 1u) << 20;
std::atomic<uint64_t> stream_peak{0}, stream_extra{0}, stream_grown{0};
std::atomic<uint64_t> pools_extra{0};   // (descriptor pools made because a recording's own ran out: make_pool)
std::atomic<uint64_t> pools_grown{0};   // (slots whose pool was made whole after one ran out: pool_shift_first)
std::deque<RecordingRes> res;   // (a deque: a recording thread keeps its own in place while another is added)
thread_local RecordingRes* cur = nullptr;

// (hookshot) A slot's pool starts at half the size below (shift 1) and is made at the whole size (shift 0) from the recording after one
// that ran out of it: a recording took at most 289 sets and 13,015 sampled images of the 8,192 and 32,768 (Castle Town and Hyrule
// Field at 1440p), and RADV (the Steam Deck) keeps a pool's descriptors in GPU memory, ~4 MB a whole pool for the 48 slots. The pools
// added when one ran out are whole. RX_VK_DESC_POOL_GROW=0: every pool whole, as before
uint32_t pool_shift_first() {
    static const uint32_t v = getenv("RX_VK_DESC_POOL_GROW") && atoi(getenv("RX_VK_DESC_POOL_GROW")) == 0 ? 0u : 1u;
    return v;
}
VkDescriptorPool make_pool(uint32_t shift = 0) {
    // (each recording's pool: 8192 sets of up to 32768 descriptors a type, for the programs with more descriptors than the push limit,
    // 24 or 48 pools at once, and the driver may keep host memory for all of that. A pool that runs out is followed by another of the same
    // size (cur->extra), so a smaller one is safe, and costs a pool made and destroyed per recording in a scene that needs more: the
    // streams' line counts those (pools_extra), which is what says how small is too small.
    // RX_VK_DESC_POOL_DIV=<n> (1..32): a n-th of both)
    static const uint32_t div0 = getenv("RX_VK_DESC_POOL_DIV") ? (uint32_t)std::clamp(atoi(getenv("RX_VK_DESC_POOL_DIV")), 1, 32) : 1;
    const uint32_t div = div0 << std::min(shift, 4u);
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32768 / div}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384 / div},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32768 / div}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4096 / div},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2048 / div}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 2048 / div}};
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets = 8192 / div;
    ci.poolSizeCount = (uint32_t)std::size(sizes);
    ci.pPoolSizes = sizes;
    VkDescriptorPool p = VK_NULL_HANDLE;
    return vkCreateDescriptorPool(dev.device, &ci, nullptr, &p) == VK_SUCCESS ? p : VK_NULL_HANDLE;
}

constexpr VkBufferUsageFlags kStreamUse = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;   // (vertex too: a binding served from a copy, gpu_buffer_bind)
// Commit ownership only after the complete mapping and its container entry exist.
bool stream_insert(RecordingRes& r, size_t index, uint64_t bytes) {
    if (!bytes || index > r.streams.size()) return false;
    RecordingRes::Stream s{};
    s.size = bytes;
    void* mapped = nullptr;
    if (!create_buffer(bytes, kStreamUse, true, s.buffer, s.alloc, &mapped, true) || !s.buffer || !s.alloc || !mapped) {
        if (s.buffer || s.alloc) destroy_buffer(s.buffer, s.alloc);
        return false;
    }
    s.mapped = (uint8_t*)mapped;
    try {
        r.streams.insert(r.streams.begin() + (ptrdiff_t)index, s);
    } catch (const std::exception&) {
        destroy_buffer(s.buffer, s.alloc);
        return false;
    }
    return true;
}
/// The recording goes on in its next stream (made when the slot has none yet, twice the last one's size up to 32 MB, or the size asked)
bool stream_next(uint64_t need) {
    constexpr uint64_t mb = 1ull << 20;
    if (!cur || need > UINT64_MAX - (mb - 1)) return false;
    RecordingRes& r = *cur;
    if (r.si >= r.streams.size()) return false;
    if (r.si + 1 < r.streams.size() && r.streams[r.si + 1].size >= need) {
        r.si++;
    } else {
        const uint64_t bytes = std::max<uint64_t>((need + mb - 1) & ~(mb - 1), std::min<uint64_t>(r.size, 16ull << 20) * 2);
        const size_t next = r.si + 1;
        // Keep old bindings alive; a fitting bigger stream goes before smaller retained ones.
        if (!stream_insert(r, next, bytes)) return false;
        r.si = next;
        stream_extra += bytes, stream_grown++;
        static int logged;
        if (logged++ < 8)
            logf("vk: a recording needed more than its %llu MB stream: one of %llu MB made (%llu MB of them in all)",
                 (unsigned long long)(r.size >> 20), (unsigned long long)(bytes >> 20), (unsigned long long)(stream_extra.load() >> 20));
    }
    r.before += r.used;
    r.stream = r.streams[r.si].buffer, r.stream_alloc = r.streams[r.si].alloc, r.mapped = r.streams[r.si].mapped;
    r.size = r.streams[r.si].size, r.used = 0;
    return true;
}
/// Room for n bytes at align in the recording's stream, moving it to a bigger one when there is none (false: no room)
bool stream_reserve(uint64_t n, uint64_t align) {
    if (!cur || !cur->stream || !cur->mapped || !align || (align & (align - 1))) return false;
    if (cur->used <= cur->size) {
        const uint64_t padding = (0 - cur->used) & (align - 1);
        const uint64_t left = cur->size - cur->used;
        if (padding <= left && n <= left - padding) return true;
    }
    if (n > UINT64_MAX - align) return false;
    if (stream_next(n + align)) return true;
    static int logged;
    if (logged++ < 4) logf("vk: the recording's stream is full (%llu MB)", (unsigned long long)(cur->size >> 20));
    return false;
}
/// Space in the recording's stream (UINT64_MAX when there is none; the buffer is cur->stream and cur->mapped after the call)
uint64_t stream_alloc(uint64_t n, uint64_t align) {
    if (!stream_reserve(n, align)) return UINT64_MAX;
    const uint64_t at = (cur->used + align - 1) & ~(align - 1);
    cur->used = at + n;
    return at;
}

VkBuffer zero_buffer = VK_NULL_HANDLE;   // (attributes the program reads that no stream supplies; 64 KB of zeros)
VmaAllocation zero_alloc = nullptr;

// ---- NVN state to Vulkan ----
/// 3-component 8/16-bit vertex formats are not vertex formats everywhere: read 4 (the 4th past the attribute)
VkFormat supported_vertex_format(VkFormat f) {
    static std::mutex mu;   // (pipelines are built on workers)
    std::lock_guard lk(mu);
    static std::unordered_map<int, VkFormat> known;
    auto it = known.find((int)f);
    if (it != known.end()) return it->second;
    VkFormatProperties p{};
    vkGetPhysicalDeviceFormatProperties(dev.physical, f, &p);
    VkFormat out = f;
    if (!(p.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) {
        switch (f) {
        case VK_FORMAT_R8G8B8_UNORM: out = VK_FORMAT_R8G8B8A8_UNORM; break;
        case VK_FORMAT_R8G8B8_SNORM: out = VK_FORMAT_R8G8B8A8_SNORM; break;
        case VK_FORMAT_R8G8B8_UINT: out = VK_FORMAT_R8G8B8A8_UINT; break;
        case VK_FORMAT_R8G8B8_SINT: out = VK_FORMAT_R8G8B8A8_SINT; break;
        case VK_FORMAT_R8G8B8_USCALED: out = VK_FORMAT_R8G8B8A8_USCALED; break;
        case VK_FORMAT_R8G8B8_SSCALED: out = VK_FORMAT_R8G8B8A8_SSCALED; break;
        case VK_FORMAT_R16G16B16_SFLOAT: out = VK_FORMAT_R16G16B16A16_SFLOAT; break;
        case VK_FORMAT_R16G16B16_UNORM: out = VK_FORMAT_R16G16B16A16_UNORM; break;
        case VK_FORMAT_R16G16B16_SNORM: out = VK_FORMAT_R16G16B16A16_SNORM; break;
        case VK_FORMAT_R16G16B16_UINT: out = VK_FORMAT_R16G16B16A16_UINT; break;
        case VK_FORMAT_R16G16B16_SINT: out = VK_FORMAT_R16G16B16A16_SINT; break;
        case VK_FORMAT_R16G16B16_USCALED: out = VK_FORMAT_R16G16B16A16_USCALED; break;
        case VK_FORMAT_R16G16B16_SSCALED: out = VK_FORMAT_R16G16B16A16_SSCALED; break;
        default: break;
        }
    }
    known[(int)f] = out;
    return out;
}

// ---- pipeline layouts and pipelines ----
// (+ the guest page table's address, a task draw's first, and the fallback's two buffer addresses)
constexpr uint32_t kPushSize = Shader::Backend::SPIRV::MESH_FALLBACK_PUSH_END;

void ensure_layout(Program& p) {
    if (p.layout) return;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    uint64_t total = 0;   // (64-bit: a sum of 32-bit counts)
    auto add = [&](VkDescriptorType type, VkShaderStageFlags stage, const auto& descriptors) {
        for (const auto& d : descriptors) {
            bindings.push_back({(uint32_t)bindings.size(), type, d.count, stage, nullptr});
            p.binding_types.push_back(type);
            total += d.count;
        }
    };
    for (const Stage& st : p.stages) {   // (Eden's order, as the SPIR-V back end numbered them)
        add(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, st.vk_stage, st.info.constant_buffer_descriptors);
        add(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, st.vk_stage, st.info.storage_buffers_descriptors);
        add(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, st.vk_stage, st.info.texture_buffer_descriptors);
        add(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, st.vk_stage, st.info.image_buffer_descriptors);
        add(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, st.vk_stage, st.info.texture_descriptors);
        add(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, st.vk_stage, st.info.image_descriptors);
    }
    // Compare the driver's push cost with the existing per-recording pools. Keep helper passes' push layouts unchanged.
    static const bool descriptor_sets = env_on("RX_VK_DESCRIPTOR_SETS");
    p.push = !descriptor_sets && dev.push_descriptor && vkCmdPushDescriptorSetKHR && total <= dev.max_push_descriptors;
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    if (p.push) li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    li.bindingCount = (uint32_t)bindings.size();
    li.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &p.set_layout);
    VkShaderStageFlags stages = 0;
    for (const Stage& st : p.stages) stages |= st.vk_stage;
    const VkPushConstantRange range{stages, 0, kPushSize};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &p.set_layout;
    pi.pushConstantRangeCount = 1;
    pi.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(dev.device, &pi, nullptr, &layout);
    p.push_stages = stages;
    for (const Stage& st : p.stages) p.uses_render_area |= st.info.uses_render_area;
    p.uses_push = p.uses_render_area;
    for (const Stage& st : p.stages) p.uses_push |= st.info.uses_rescaling_uniform;
    // (the page table: compute, and (EvilOOT) the graphics stages, which follow global memory as the fallback)
    for (const Stage& st : p.stages) p.uses_push |= st.info.uses_global_memory;
    for (const Stage& st : p.stages) p.uses_push |= (st.vk_stage & VK_SHADER_STAGE_TASK_BIT_EXT) != 0;   // (its first workgroup)
    if (p.fallback) p.uses_push = true;   // (its records' and payloads' addresses)
    std::atomic_thread_fence(std::memory_order_release);   // (the layout last: a thread that sees it sees the rest)
    p.layout = layout;
}

// ---- the pipeline cache, on disk ----
VkPipelineCache pcache = VK_NULL_HANDLE;
namespace { void ledger_drop_all(); }
bool pcache_loaded = false;   // (it began with this device's and driver's data from the file)
std::string pcache_path;
/// (hookshot) What the last save on this thread did: a forced save runs on its caller's thread, which reads it (precompile_save.h)
thread_local precompile_save::Outcome pcache_save_outcome;
std::atomic<int64_t> pcache_built_since_save{0};
// (Hookshot) RX_VK_PCACHE_MISSES=0: every pipeline made counts toward a save of the cache, as before
bool pcache_misses_only() {
    static const bool v = !getenv("RX_VK_PCACHE_MISSES") || atoi(getenv("RX_VK_PCACHE_MISSES")) != 0;
    return v;
}
// (hookshot) The size of the cache's data when it was last read or written. The count above takes in every pipeline made
// with the cache, the ones it already held too: the precompile's ~13,000 at each start, all found in the cache, made the
// first check in play (present 300) write the whole cache again, 1.1 GB, at every start, the file the same 1146 MB each
// time. On a SATA SSD that write took 4-10 s and left the disk's reads slow (10-450 ms) for the minutes after, the game's
// own reads too. A save asks the driver the data's size first (no data is copied for that) and writes only a different
// size: data the cache took in since. The pipeline creation feedback cannot tell the new ones apart: NVIDIA's driver reports
// most precompiled pipelines as not found in the app's cache (~460 of ~12,900 here), which it then serves from its own.
std::atomic<size_t> pcache_saved_size{0};
bool pcache_no_save = false;   // (RX_VK_PCACHE_LOAD=0: pcache_init)
bool pcache_file_rejected = false;   // (pcache_init: the cache's file was there but not for this device and driver)
/// A pipeline cache file's data, or nothing when it cannot be read or was made by another device or driver
PipelineCacheFile pcache_file_read(const std::string& path) {
    return PipelineCacheFile::read(path, [](const uint8_t* head, uint64_t size) {
        uint32_t head_size = 0, version = 0, vendor = 0, device = 0;
        std::memcpy(&head_size, head, 4);
        std::memcpy(&version, head + 4, 4);
        std::memcpy(&vendor, head + 8, 4);
        std::memcpy(&device, head + 12, 4);
        return head_size >= 32 && head_size <= size && version == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
               vendor == dev.props.vendorID && device == dev.props.deviceID &&
               !std::memcmp(head + 16, dev.props.pipelineCacheUUID, VK_UUID_SIZE);
    });
}
std::string pcache_dir;

// (hookshot) Intel's Windows driver faults in vkGetPipelineCacheData (igvk64.dll+0x79ddf3, an access violation) when it fetches
// a cache's data while other threads build pipelines with that cache: the game closes with no message
// (https://github.com/NotRexed/Hookshot/issues/999: the Arc 140V, and 4 runs of 6 on a Core Ultra 7 155H, driver 101.9175).
// Vulkan allows it; on Intel the fetches and the builds take turns: a build holds this shared, a fetch alone.
// The fetch holds it around the vkGetPipelineCacheData calls only, never across a file's write nor across a wait for the
// builds (play_cache_save's wait for a generation's users: a builder counted in there may be between two of its builds, waiting
// for this lock, and would never count itself out). The background save does not queue behind the builds: it tries a while,
// then leaves the save to the next one; after kFetchPutOffs saves put off in a row it queues, for a bounded time. With the lock
// on, the window's close waits a bounded time too: a build stuck in the driver costs the play file, not the close. With it off,
// every wait is as before (a build on AMD's Windows driver can take 5-13 s: the close waits for it rather than lose the file).
// RX_VK_PCACHE_FETCH_LOCK=0: off, =1: on any GPU
std::shared_timed_mutex pcache_fetch_mu;
bool pcache_fetch_lock_on() {
    static const bool on = dev.policy.pcache_fetch_lock;   // (device_policy.h: Intel's Windows driver, then the switch; once)
    return on;
}
/// (Intel) compute pipelines built one at a time, on a thread of their own (jobs_push_serial): by default with Intel's
/// Windows driver, whose compiler made this needed (device_policy.h compute_builds_serial; it was every Intel GPU, Mesa's
/// driver too). RX_VK_COMPUTE_SERIAL=1 or 0 says so on any GPU
bool serial_builds_on() {
    static const bool on = dev.policy.compute_builds_serial;
    return on;
}
thread_local int pcache_fetch_depth = 0;   // (a fetch inside a fetch on the same thread: pcache_saved_now's save)
constexpr int kFetchPollMs = 2000;    // (the background save's patience, polled)
constexpr int kFetchPutOffs = 3;      // (saves put off in a row before the next one queues)
constexpr int kFetchQueueMs = 10000;  // (how long a queued fetch waits: the window's close, a save put off kFetchPutOffs times)
/// A fetch of a pipeline cache's data, the builds held off while it lasts (held() says whether it was had).
/// wait_ms < 0: waits until the builds under way are done. queue: waits in the lock, as a writer (new builds wait behind it), at
/// most wait_ms; not queue: polled for wait_ms, had only at a moment no build is under way (a writer waiting in the lock would
/// stall the builds behind it, the replay thread's draw-time asks among them)
struct PcacheFetch {
    bool owns = false, outer = true;
    explicit PcacheFetch(int wait_ms, bool queue = true) {
        if (!pcache_fetch_lock_on()) return;
        if (pcache_fetch_depth > 0) {
            outer = false, pcache_fetch_depth++;
            return;
        }
        if (wait_ms < 0) pcache_fetch_mu.lock(), owns = true;
        else if (queue) owns = pcache_fetch_mu.try_lock_for(std::chrono::milliseconds(wait_ms));
        else
            for (const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
                 !(owns = pcache_fetch_mu.try_lock()) && std::chrono::steady_clock::now() < until;)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (owns) pcache_fetch_depth = 1;
    }
    ~PcacheFetch() {
        if (!outer) pcache_fetch_depth--;
        else if (owns) pcache_fetch_depth = 0, pcache_fetch_mu.unlock();
    }
    bool held() const { return !pcache_fetch_lock_on() || !outer || owns; }
};
/// A pipeline built with a cache, while no fetch of one is under way (PcacheFetch)
struct PcacheUse {
    bool on = false;
    explicit PcacheUse(VkPipelineCache cache) : on(cache && pcache_fetch_lock_on() && pcache_fetch_depth == 0) {
        if (on) pcache_fetch_mu.lock_shared();
    }
    ~PcacheUse() {
        if (on) pcache_fetch_mu.unlock_shared();
    }
};

// ---- (hookshot) What play makes goes into a cache of its own, one per save ----
// pcache takes what the start-up builds and is then left as it was read. Every save in play wrote it whole again for what play had
// added: NVIDIA's driver copies into the cache a pipeline is made with each pipeline it takes from its own disk cache, so a first
// visit grew it at each new pipeline, and the save every 32 of them wrote 0.5-1.2 GB to add a few MB (497-539 MB written six times
// in a 4-minute walk, 1.4-3.2 s each, 3 GB in all to keep 42 MB). In play a pipeline goes into the generation's own cache (or comes
// from pcache, create_cached), and a save writes that generation alone, to hookshot_vk_pipelines_play_<n>.bin, while play goes on into
// a new one. The next start merges those files into pcache as it reads it; at kPlayFilesKept of them they are written into
// hookshot_vk_pipelines.bin there, once, and removed. RX_VK_PLAY_CACHE=0: everything into pcache, as before
struct PlayCache {
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::atomic<int> users{0};       // (pipelines being made with it now)
    std::atomic<int64_t> built{0};   // (pipelines put into it)
    // (of play's pipelines while it was the generation: from the driver's own cache, compiled, from pcache)
    std::atomic<int64_t> from_driver{0}, compiled{0}, from_start{0};
};
std::atomic<PlayCache*> play_cache{nullptr};   // (null before play, and without pipelineCreationCacheControl)
uint32_t play_seq = 0;                         // (the highest play file number read or written)
constexpr size_t kPlayFilesKept = 16;
std::string play_path(uint32_t n) {
    return pcache_dir + "hookshot_vk_pipelines_play_" + std::to_string(n) + ".bin";
}
/// A cache's data written whole to path, through a .tmp moved over it: its size, 0 when it is not in place (also when the
/// fetch lock was not had within wait_ms, PcacheFetch: the file's write is outside it)
size_t play_file_write(VkPipelineCache cache, const std::string& path, int wait_ms = -1, bool queue = true) {
    size_t n = 0;
    {
        PcacheFetch fetch(wait_ms, queue);
        if (!fetch.held() || vkGetPipelineCacheData(dev.device, cache, &n, nullptr) != VK_SUCCESS || !n) return 0;
    }
    auto data = PipelineCacheFile::writable(path, n);
    if (data.empty()) return 0;
    {   // (a cache grown since its size was asked does not fit: not written, tried again at the next save)
        PcacheFetch fetch(wait_ms, queue);
        if (!fetch.held() || vkGetPipelineCacheData(dev.device, cache, &n, data.data()) != VK_SUCCESS) return 0;
    }
    return data.publish(n) ? n : 0;
}
extern bool pcache_no_save;   // (pcache_init, below: the file not read, RX_VK_PCACHE_LOAD or pcache_load.h)
/// (pcache_init, before any other thread has pcache) the play files the last starts wrote, merged into pcache; at kPlayFilesKept
/// of them pcache is written into its file and they are removed. A file of another driver is of no use: removed. The number merged
size_t play_files_merge() {
    // (hookshot) The file not read: nothing merged into the cache, which started empty, and above all that cache never
    // written over the file at kPlayFilesKept (a session without it wrote its 38 MB over a 1.2 GB file a start loading programs
    // before the title then lacked). The play files stay for a start that reads the file
    if (pcache_no_save) return 0;
    std::vector<std::string> merged;
    size_t merged_kb = 0;
    // (hookshot, issue 2579) and into one cache of their own, written as one play file in their place: each save of a session wrote a
    // file of its own (every 32 new pipelines), and a game folder held 22 of them before the 16 that fold them into pcache
    VkPipelineCache together = VK_NULL_HANDLE;
    VkPipelineCacheCreateInfo tci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (vkCreatePipelineCache(dev.device, &tci, nullptr, &together) != VK_SUCCESS) together = VK_NULL_HANDLE;
    std::error_code ec;
    static const std::string pre = "hookshot_vk_pipelines_play_", suf = ".bin";
    for (const auto& e : std::filesystem::directory_iterator(pcache_dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.size() <= pre.size() + suf.size() || name.compare(0, pre.size(), pre) ||
            name.compare(name.size() - suf.size(), suf.size(), suf))
            continue;
        const std::string num = name.substr(pre.size(), name.size() - pre.size() - suf.size());
        if (num.size() > 9 || num.find_first_not_of("0123456789") != std::string::npos) continue;
        play_seq = std::max(play_seq, (uint32_t)std::stoul(num));
        const std::string path = e.path().string();
        auto part = pcache_file_read(path);
        VkPipelineCache tmp = VK_NULL_HANDLE;
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = part.size(), ci.pInitialData = part.data();
        if (part.empty()) {
            if (part.rejected()) remove(path.c_str()); // Only a proven incompatible header can discard a generation.
            continue;
        }
        if (vkCreatePipelineCache(dev.device, &ci, nullptr, &tmp) != VK_SUCCESS) {
            if (part.reset()) remove(path.c_str()); // Release a mapped Windows view before removing rejected bytes.
            continue;
        }
        if (vkMergePipelineCaches(dev.device, pcache, 1, &tmp) == VK_SUCCESS) {
            merged.push_back(path), merged_kb += part.size() >> 10;
            if (together && vkMergePipelineCaches(dev.device, together, 1, &tmp) != VK_SUCCESS)
                vkDestroyPipelineCache(dev.device, together, nullptr), together = VK_NULL_HANDLE;
        }
        vkDestroyPipelineCache(dev.device, tmp, nullptr);
    }
    if (!merged.empty()) logf("vk: pipeline cache: %zu play files merged (%zu KB)", merged.size(), merged_kb);
    if (merged.size() >= kPlayFilesKept) {   // (into the one file, here at the start, and out of the play files)
        if (const size_t wrote = play_file_write(pcache, pcache_path)) {
            for (const std::string& path : merged) remove(path.c_str());
            logf("vk: pipeline cache: the play files written into it (%zu KB)", wrote >> 10);
        }
    } else if (merged.size() > 1 && together) {   // (into one play file, the others removed once it is in place)
        if (const size_t wrote = play_file_write(together, play_path(play_seq + 1))) {
            play_seq++;
            for (const std::string& path : merged) remove(path.c_str());
            logf("vk: pipeline cache: %zu play files put together into play file %u (%zu KB)", merged.size(), play_seq, wrote >> 10);
        }
    }
    if (together) vkDestroyPipelineCache(dev.device, together, nullptr);
    return merged.size();
}
PlayCache* play_cache_make() {
    auto* g = new PlayCache;   // (never freed: a thread that read the pointer before a save took it may still count itself in)
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (vkCreatePipelineCache(dev.device, &ci, nullptr, &g->cache) != VK_SUCCESS) {
        delete g;   // (never published)
        return nullptr;
    }
    return g;
}
/// (the start-up done, vk_main's start) play begins: what is made from now on goes into the play generations
void play_cache_begin() {
    static const bool off = getenv("RX_VK_PLAY_CACHE") && atoi(getenv("RX_VK_PLAY_CACHE")) == 0;
    // RX_VK_PCACHE_LOAD=0 is the A/B of the cache with nothing kept: no play generations either. The file
    // left unread by the automatic rule (pcache_load.h) keeps them: what play makes goes into them and play_cache_save lets
    // each go at a save, not written
    static const bool switched_off = [] { const char* e = getenv("RX_VK_PCACHE_LOAD"); return e && *e && env_value_off(e); }();
    if ((pcache_no_save && switched_off) || off || !pcache || !dev.vk13.pipelineCreationCacheControl || play_cache.load()) return;
    if (PlayCache* g = play_cache_make()) {
        play_cache.store(g);
        logf("vk: pipeline cache: what play makes goes into play files (RX_VK_PLAY_CACHE=0: into the one cache)");
    }
}
void predicted_use_log();   // (below, with the predicted pipelines)
/// (pipeline_cache_save's thread, or the window's close) play's generation taken away once nothing is being made with it, a new one
/// in its place, and written alone. One not written is kept and tried again at the next save. wait_ms and queue: the fetch lock's
/// (PcacheFetch), which is never held across the wait for a generation's users. With the fetch lock on, a bounded save (the
/// window's close, the background save's poll) waits for the users that long too: a build stuck in the driver leaves its
/// generation for a next save, and neither the save nor a close behind it (mu) hangs. With the lock off the users are waited
/// for as before. The close's worst case: wait_ms for the users, then wait_ms for each of play_file_write's two fetches
void play_cache_save(int wait_ms = -1, bool queue = true) {
    static std::mutex mu;   // (a save's thread and the window's close, pipeline_cache_close)
    std::lock_guard lk(mu);
    static std::vector<PlayCache*> unsaved;
    // (hookshot, issue 2579) the session's generations merged into one cache, written each save to one play file: the session's own
    // number, taken at its first save. A file a save each piled up in the game folder (22 of them)
    static VkPipelineCache session = VK_NULL_HANDLE;
    static uint32_t session_n = 0;
    static int64_t session_built = 0;
    static bool unwritten = false;   // (the session's cache holds more than its file: a write that failed)
    if (PlayCache* g = play_cache.load(); g && g->built.load()) {
        unsaved.reserve(unsaved.size() + 1);   // (before it is taken away: it cannot then be lost to a failed allocation)
        if (PlayCache* next = play_cache_make()) {
            play_cache.store(next);
            unsaved.push_back(g);
        }
    }
    const bool bounded = pcache_fetch_lock_on() && wait_ms >= 0;   // (queue or not: a polled save holds mu too)
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(bounded ? wait_ms : 0);
    const auto idle = [&](PlayCache* old) {   // (written once nothing is being made with it)
        while (old->users.load() && (!bounded || std::chrono::steady_clock::now() < until)) Sleep(1);
        if (!old->users.load()) return true;
        static std::atomic<int> told{0};
        if (told++ < 4) logf("vk: pipeline cache: a play file not written yet: a pipeline was still being built in it");
        return false;
    };
    // (the file not read, pcache_no_save: no session cache, so each generation is let go below, nothing merged or written)
    if (!session && !unsaved.empty() && !pcache_no_save) {
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        if (vkCreatePipelineCache(dev.device, &ci, nullptr, &session) != VK_SUCCESS) session = VK_NULL_HANDLE;
    }
    bool grew = false;
    for (auto it = unsaved.begin(); session && it != unsaved.end();) {   // (each generation into the session's cache, then let go)
        PlayCache* old = *it;
        if (!idle(old) || vkMergePipelineCaches(dev.device, session, 1, &old->cache) != VK_SUCCESS) {
            ++it;
            continue;
        }
        session_built += old->built.load();
        logf("vk: pipeline cache: play saved: %lld pipelines, %lld from the driver's cache, %lld compiled; %lld more from the start's cache",
             (long long)old->built.load(), (long long)old->from_driver.load(), (long long)old->compiled.load(),
             (long long)old->from_start.load());
        vkDestroyPipelineCache(dev.device, old->cache, nullptr);
        old->cache = VK_NULL_HANDLE;
        it = unsaved.erase(it);
        grew = true;
    }
    if (session && (grew || unwritten)) {   // (one not written stays in the session's cache: written with the next save)
        if (!session_n) session_n = play_seq + 1;
        const size_t wrote = play_file_write(session, play_path(session_n), wait_ms, queue);
        unwritten = !wrote;
        if (wrote) {
            play_seq = std::max(play_seq, session_n);
            logf("vk: pipeline cache: play file %u saved (%zu KB, %lld pipelines this session)", session_n, wrote >> 10,
                 (long long)session_built);
            predicted_use_log();
        }
    }
    for (auto it = unsaved.begin(); !session && it != unsaved.end();) {   // (no session cache: a file each, as before)
        PlayCache* old = *it;
        if (!idle(old)) {
            ++it;
            continue;
        }
        if (pcache_no_save) {   // (hookshot: the file not read, nor saved: what play made is in the driver's own cache)
            static std::atomic<int> told{0};
            if (const int n = ++told; n <= 4 || n % 16 == 0)
                logf("vk: pipeline cache: play's generation %d let go, not written (the file not read): %lld pipelines, %lld from the "
                     "driver's cache, %lld compiled", n, (long long)old->built.load(), (long long)old->from_driver.load(),
                     (long long)old->compiled.load());
            vkDestroyPipelineCache(dev.device, old->cache, nullptr);
            old->cache = VK_NULL_HANDLE;
            it = unsaved.erase(it);
            continue;
        }
        if (const size_t wrote = play_file_write(old->cache, play_path(play_seq + 1), wait_ms, queue)) {
            play_seq++;
            logf("vk: pipeline cache: play file %u saved (%zu KB): %lld pipelines, %lld from the driver's cache, %lld compiled; %lld "
                 "more from the start's cache",
                 play_seq, wrote >> 10, (long long)old->built.load(), (long long)old->from_driver.load(), (long long)old->compiled.load(),
                 (long long)old->from_start.load());
            predicted_use_log();
            vkDestroyPipelineCache(dev.device, old->cache, nullptr);
            old->cache = VK_NULL_HANDLE;
            it = unsaved.erase(it);
        } else {
            ++it;
        }
    }
}
/// Before play, a pipeline made with pcache. In play pcache is left as it was read: asked of it first, a pipeline new to it would
/// grow it all the same (NVIDIA copies in what it takes from its own cache). Asked without compiling
/// (VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT), in this order:
///  1. the generation's cache: one the driver has in its own cache comes back, copied into the generation;
///  2. pcache: one it holds comes back from it (the computes the prepare warmed, on a driver without a cache of its own);
///  3. else it is compiled, into the generation's cache.
/// One the caller already asks without compiling (a dispatch's compute pipeline) takes steps 1 and 2 alone: asked of pcache
/// first it too would grow it in play, NVIDIA answering from its own cache.
/// A stage library asked without compiling is the exception: pcache first, then the generation, never compiled. A draw waits
/// for it on the replay thread, and pcache is where the prepare's warm pass put the stage libraries: found there the ask is an
/// in-memory hit, while asked of the empty generation first every one came from the driver's disk cache, about five times as
/// slow (a 10-minute first visit on an RTX 4070 Ti SUPER: 7.3 s of links against 1.3 s, 2.2 ms a pipeline against 0.4).
/// pcache grows by it only for a library the warm pass did not leave there (a folder prepared before the warm pass), and
/// only in memory: it is not written in play for that (pipeline_cache_save) until it has grown 64 MB.
/// The load when the game makes a program (vk_programs.cpp) asks the same way as a stage library, and for the same reason: what
/// it asks for is what the prepare built into pcache, an in-memory hit there; asked of the generation first, each would come
/// from the driver's disk cache and be copied into the generation, and a session's play files would hold every pipeline again
thread_local bool ask_start_first = false;
template <class CI>
VkResult create_cached(VkResult (*create)(VkPipelineCache, const CI*, VkPipeline*), const CI* ci, VkPipeline* pl) {
    PlayCache* g = play_cache.load();
    if (!g) return create(pcache, ci, pl);
    const bool ask_only = (ci->flags & VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT) != 0;
    // (counted in before it is used: a save takes the generation away only once its count is back to 0)
    for (;;) {
        g = play_cache.load();
        g->users.fetch_add(1);
        if (play_cache.load() == g) break;
        g->users.fetch_sub(1);
    }
    CI ask = *ci;
    ask.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    if (ask_only && ((ci->flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) || ask_start_first)) {
        VkResult r = create(pcache, &ask, pl);
        // (the census counts what a draw or dispatch asked for; what the load at create asks for is counted there)
        if (r == VK_SUCCESS) {
            g->from_start.fetch_add(1), census.play_from_start += !ask_start_first, use_counters.made_how = 2;
        } else if (r == VK_PIPELINE_COMPILE_REQUIRED) {
            *pl = VK_NULL_HANDLE;
            r = create(g->cache, &ask, pl);   // (one play compiled into this generation)
            if (r == VK_SUCCESS)
                g->built.fetch_add(1), g->from_driver.fetch_add(1), census.play_from_driver += !ask_start_first, use_counters.made_how = 1;
        }
        g->users.fetch_sub(1);
        return r;
    }
    VkResult r = create(g->cache, &ask, pl);
    if (r == VK_SUCCESS) {
        g->built.fetch_add(1), g->from_driver.fetch_add(1), census.play_from_driver++, use_counters.made_how = 1;
    } else if (r == VK_PIPELINE_COMPILE_REQUIRED) {
        *pl = VK_NULL_HANDLE;
        r = create(pcache, &ask, pl);
        if (r == VK_SUCCESS) {
            g->from_start.fetch_add(1), census.play_from_start++, use_counters.made_how = 2;
        } else if (r == VK_PIPELINE_COMPILE_REQUIRED && !ask_only) {
            *pl = VK_NULL_HANDLE;
            r = create(g->cache, ci, pl);
            if (r == VK_SUCCESS) g->built.fetch_add(1), g->compiled.fetch_add(1), census.play_compiled++, use_counters.made_how = 3;
        }
    }
    g->users.fetch_sub(1);
    return r;
}
/// (pipeline_cache_save) play has begun and what it makes goes into the play generations
bool play_cache_in_use() { return play_cache.load() != nullptr; }
// ---- (end of the play caches) ----

/// (hookshot) The size NVIDIA's own driver lets its shader cache grow to (nvidia_cache_limit.h), read once a run: Windows' Shader
/// Cache Size through the driver's settings, Linux's __GL_SHADER_DISK_CACHE_SIZE. Not known for other drivers
static nvidia_cache_limit::Limit nvidia_limit() {
    static const nvidia_cache_limit::Limit limit = [] {
        if (dev.driver_id != VK_DRIVER_ID_NVIDIA_PROPRIETARY) return nvidia_cache_limit::Limit{};
#ifdef _WIN32
        return nvidia_cache_limit::windows_limit();
#else
        return nvidia_cache_limit::linux_limit(getenv("__GL_SHADER_DISK_CACHE_SIZE"));
#endif
    }();
    return limit;
}
// (RX_VK_PCACHE_LOAD=0, an A/B of what the file costs in memory: its 2.5 GB read whole and held by the driver. The cache
// starts empty and is not saved; the driver's own shader cache is all there is.) (hookshot) Unset, the file is
// not read either on a Linux PC with NVIDIA's own driver, 20 GB of RAM or less and programs loaded at their create calls,
// where NVIDIA's own shader cache is on and not nearly empty: the driver's copy of the file was most of a gigabyte of a 16 GB
// PC's swap (pcache_load.h). RX_VK_PCACHE_LOAD=1 reads it there too. One log line gives the answer and why, on every PC
bool pcache_load_off() {
    static const bool off = [] {
        const char* e = getenv("RX_VK_PCACHE_LOAD");
        const int sw = e && *e ? (env_value_off(e) ? 0 : 1) : -1;   // (hookshot_env.h: 0/off/false/no)
        const bool preparing = env_on("HOOKSHOT_PREPARE_ONLY"), at_create = programs_load_at_create();
        bool cgroup_limited = false;
#ifdef _WIN32
        const bool linux_nvidia = false;
        // (Windows with NVIDIA's own driver: its GLCache folder answers, with Linux's RAM rule: pcache_load.h)
        const bool windows_nvidia = dev.driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        const uint32_t ram_mb = GlobalMemoryStatusEx(&ms) ? (uint32_t)std::min<uint64_t>(ms.ullTotalPhys >> 20, UINT32_MAX) : 0;
#else
        const bool windows_nvidia = false;
        const bool linux_nvidia = dev.driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY;   // (not NVK: Mesa's, with Mesa's cache)
        const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
        uint64_t capacity = pages > 0 && page > 0 ? ((uint64_t)pages * (uint64_t)page) >> 20 : UINT64_MAX;
        if (linux_nvidia && sw < 0 && !preparing && at_create) {
            const uint64_t limit = compute_gate::cgroup_memory().limit_mb;
            cgroup_limited = limit < capacity;
            if (cgroup_limited) capacity = std::max<uint64_t>(1, limit);   // zero means unknown to the cache policy
        }
        const uint32_t ram_mb = capacity == UINT64_MAX ? 0 : (uint32_t)std::min<uint64_t>(capacity, UINT32_MAX);
#endif
        // (NVIDIA's folder walked only where everything else says not to read the file; its size alone counts, not Mesa's)
        pcache_load::Measured cache{pcache_load::Cache::kUnknown, 0};
        std::string dir;
        const bool looked = pcache_load::needs_cache(sw, preparing, at_create, linux_nvidia, ram_mb, windows_nvidia);
        if (looked && windows_nvidia) {
            cache = pcache_load::nvidia_cache_windows(getenv("LOCALAPPDATA"), &dir);
        } else if (looked) {
            const char* c = getenv("__GL_SHADER_DISK_CACHE");
            cache = pcache_load::nvidia_cache(c && *c && env_value_off(c), getenv("__GL_SHADER_DISK_CACHE_PATH"),
                                              getenv("XDG_CACHE_HOME"), getenv("HOME"), &dir);
        }
        // (the size NVIDIA's cache may grow to: under 4 GB it drops pipelines as it fills, and the file answers them)
        const nvidia_cache_limit::Limit limit = looked ? nvidia_limit() : nvidia_cache_limit::Limit{};
        const uint64_t limit_mb = limit.off ? 0 : limit.mb;
        const pcache_load::Choice choice = pcache_load::choose(sw, preparing, at_create, linux_nvidia, ram_mb, cache.cache,
                                                               cache.bytes >> 20, windows_nvidia, limit_mb);
        std::string seen;   // (what was looked at: the switch, or the RAM and NVIDIA's cache where they were)
        if (sw >= 0) seen = std::string("RX_VK_PCACHE_LOAD=") + e;
        else if (choice.why == pcache_load::Why::kRam || looked)
            seen = std::to_string(ram_mb) + (cgroup_limited ? " MB cgroup memory limit" : " MB of RAM");
        if (looked) {
            seen += ", NVIDIA's own shader cache ";
            seen += cache.cache == pcache_load::Cache::kOff       ? std::string("off (__GL_SHADER_DISK_CACHE, or no folder for it)")
                    : cache.cache == pcache_load::Cache::kUnknown ? "not measured: " + dir + " could not be walked through"
                                                                  : std::to_string(cache.bytes >> 20) + " MB in " + dir;
            if (limit_mb) seen += ", up to " + std::to_string(limit_mb) + " MB (" + limit.where + ")";
        }
        if (!choice.load && sw < 0) seen += "; RX_VK_PCACHE_LOAD=1 reads it";
        logf("vk: pipeline cache file %s: %s%s%s%s", choice.load ? "read" : "not read", pcache_load::why_text(choice.why),
             seen.empty() ? "" : " (", seen.c_str(), seen.empty() ? "" : ")");
        return !choice.load;
    }();
    return off;
}
/// (hookshot) The pipeline cache's files go when they were made for another translator than this one (program_cache_stamp,
/// which the program cache and the prepare record start over by): their pipelines can no longer be hit, and the file only grew,
/// read whole at every start and held by the driver (pcache_stamp.h). Once a run, before the first of pcache_init and
/// prepare_run: whether files went, so that prepare_run makes the stage libraries the warm pass kept in them again
/// (prepared_for.warmed), whatever its record says. (hookshot, issue
/// 388) With the file not read (pcache_load_off) they go too: a start that reads it would drop them, and kept they were 1.2 GB
/// of disk after every update that changed the translator
bool pcache_drop_stale() {
    cache_sets_switch();   // (another GPU's caches aside first, this one's back: gpu_sets.h)
    static const bool dropped = [] {
        const std::string dir = cache_path("");   // (where pcache_init keeps them: the game folder's cache/, vk_paths.h)
        const pcache_stamp::Dropped gone = pcache_stamp::check(dir, program_cache_stamp());
        if (gone.files)
            logf("vk: pipeline cache: %zu files made for another translator dropped (%llu MB); it starts empty", gone.files,
                 (unsigned long long)(gone.bytes >> 20));
        return gone.files > 0;
    }();
    return dropped;
}
void pcache_init() {
    if (pcache) return;
    const std::string dir = pcache_dir = cache_path("");   // (the game folder's cache/, with its separator: vk_paths.h)
    if (const size_t removed = PipelineCacheFile::cleanup_temporaries(dir))
        logf("vk: pipeline cache: %zu old temporary files removed", removed);
    pcache_path = dir + "hookshot_vk_pipelines.bin";
    const bool off = pcache_load_off();
    pcache_no_save = off;
    pcache_drop_stale();   // (before the files are read)
    auto data = off ? PipelineCacheFile{} : pcache_file_read(pcache_path);
    // (a file there that this device and driver cannot take: another driver made it (a driver update changes the
    // pipelineCacheUUID), another GPU, or a write cut short. The cache starts empty, and what the pipelines' marker says is in
    // it is gone: precompile_run warms again before the game rather than compile the noted pipelines in play. After
    // ProjectReLink db54a4bc and aabc7ad8)
    FILE* was = !off && data.empty() ? fopen(pcache_path.c_str(), "rb") : nullptr;   // (as pcache_file_read opens it)
    const bool refused = was && fgetc(was) != EOF;
    if (was) fclose(was);
    if (refused) {
        pcache_file_rejected = true;
        logf("vk: pipeline cache: %s was made for another driver or GPU (or is damaged): it starts empty, and the noted "
             "pipelines are built again", pcache_path.c_str());
    }
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (vkCreatePipelineCache(dev.device, &ci, nullptr, &pcache) != VK_SUCCESS) {
        ci.initialDataSize = 0, ci.pInitialData = nullptr;
        vkCreatePipelineCache(dev.device, &ci, nullptr, &pcache);
    }
    logf("vk: pipeline cache %s (%zu KB)", pcache_path.c_str(), data.size() >> 10);
    data = {};
    if (pcache && !off) play_files_merge();
    // (what it holds as read, the play files too: a save of data that size has nothing to add to the files)
    size_t held = 0;
    PcacheFetch fetch(-1);
    if (pcache && vkGetPipelineCacheData(dev.device, pcache, &held, nullptr) == VK_SUCCESS) pcache_saved_size = held;
}

struct Targets {
    Texture* color[8]{};
    VkImageView color_view[8]{};
    VkImage block_image[8]{};
    VkFormat color_format[8]{};
    int color_view_nvn[8]{};   // (hookshot) the NVN format each is drawn in (view_nvn_format): the pipeline state list's
    uint32_t color_level[8]{}, color_layer[8]{}, depth_level = 0, depth_layer = 0;   // (the levels and layers viewed)
    Texture* depth = nullptr;
    VkImageView depth_view = VK_NULL_HANDLE;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    int n = 0;
    bool has_block_images = false;
    uint32_t width = 0, height = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};

/// (hookshot) The colour output types a program is translated for: each target's, by its format (Shader::AttributeType). The draw's,
/// and the pipeline state list's from the formats it maps the game's to (state_targets)
void color_types_of(const Targets& at, uint8_t types[8]) {
    for (int i = 0; i < 8; i++) {
        const VkFormat f = i < at.n ? at.color_format[i] : VK_FORMAT_UNDEFINED;
        switch (f) {
        case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
            types[i] = (uint8_t)Shader::AttributeType::UnsignedInt;
            break;
        case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32B32A32_SINT:
            types[i] = (uint8_t)Shader::AttributeType::SignedInt;
            break;
        default: types[i] = (uint8_t)Shader::AttributeType::Float; break;
        }
    }
}
/// (hookshot) A mesh draw's topology (it has none of its own) and the topology a draw's pipeline is made with: the drawn one, or for
/// the mesh fallback's program the points, lines or triangles its mesh program makes. The draw's, and the pipeline state list's
constexpr VkPrimitiveTopology kMeshDrawTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
VkPrimitiveTopology pipeline_topology(const Program& p, VkPrimitiveTopology drawn) {
    return p.fallback ? p.fallback_topology : drawn;
}

/// A graphics pipeline: built on a worker (vk_jobs.h); ready once pipeline is set (null: it failed)
struct PipeEntry {
    std::atomic<bool> ready{false};
    std::atomic<VkPipeline> pipeline{VK_NULL_HANDLE};   // (a fast-linked one first, the worker's full build over it)
    std::atomic<uint8_t> how{0};   // (the census: how play made it, UseCounters::made_how; 0: before play, or not made)
    std::atomic<uint8_t> quick{0};   // (hookshot: quick_pipeline's, before the full build: quick_pipe.h's kNone, kMaking, kTried)
    std::atomic<const Program*> prog{nullptr};   // (the program object whose stages and layout it was built from: graphics_pipeline)
};
std::unordered_map<uint64_t, std::unique_ptr<PipeEntry>> pipelines;   // (the replay thread's)
// (hookshot) Pipelines made ahead of their first draw by the load at a program's create call (vk_programs.cpp), by their keys.
// `pipelines` is the replay's own and has no lock outside the parallel replay, so a background job puts what it made here and
// the draw's lookup takes it the first time it asks for that key (graphics_pipeline)
std::mutex kept_ready_mu;
std::unordered_map<uint64_t, std::pair<VkPipeline, const Program*>> kept_ready;   // (and the program object it was made from)
std::atomic<size_t> kept_ready_n{0};
// (st_draws to st_mesh are bumped at every draw by every recording thread: a cache line a thread, split_count.h)
SplitCount st_draws, st_skipped, st_dispatches, st_passes, st_mesh;
int64_t st_pending_skips, st_waits;
int64_t st_dispatch_unordered;   // (hookshot: dispatches right after another left without a barrier, RX_VK_GAME_ORDER)
// (hookshot) skips by the line that made them, reported with the frame statistics (draw_skip_report)
namespace {
struct SkipSite { std::atomic<int> line{0}; std::atomic<int64_t> n{0}; int64_t last = 0; };
SkipSite skip_sites[48];
int skip_slot(int line) {
    for (auto& k : skip_sites) {
        int z = 0;
        if (k.line.load() == line || k.line.compare_exchange_strong(z, line) || k.line.load() == line)
            return (int)(&k - skip_sites);
    }
    return 47;
}
}  // namespace
thread_local bool occ_skipped_here = false;   // (a draw left out after its occlusion query began)
#define SKIP_HERE() do { static const int slot_ = skip_slot(__LINE__); skip_sites[slot_].n.fetch_add(1, std::memory_order_relaxed); st_skipped++; occ_skipped_here = true; return; } while (0)
void skip_report_impl(double frames) {
    char line[600];
    int n = snprintf(line, sizeof line, "vk: skips a frame by vk_draw.cpp line:");
    bool any = false;
    for (auto& k : skip_sites) {
        const int64_t v = k.n.load(std::memory_order_relaxed);
        if (k.line.load() && v != k.last && n < (int)sizeof line - 32) {
            n += snprintf(line + n, sizeof line - n, " %d: %.1f", k.line.load(), (v - k.last) / frames);
            any = true;
        }
        k.last = v;
    }
    if (any) logf("%s", line);
}
std::atomic<int64_t> st_pipelines, st_links, st_link_us, st_lib_hits, st_lib_cold, st_lib_built, st_lib_build_us;
std::atomic<int64_t> st_pipe_fresh_skips;   // draws left out for a pipeline in the building, into targets made this frame
std::atomic<int64_t> st_pipeline_us;
double st_wait_ms;
double st_shader_wait_ms;   // (st_wait_ms's waits for a program's translation; the rest are for its pipeline)

bool depth_aspect(VkFormat f) {
    return is_depth_format(f) && f != VK_FORMAT_S8_UINT;
}

// (hookshot) The programs the driver faulted on (driver_faults.h): not asked of it again in this run, nor in later runs on the
// same driver (hookshot_driver_faults.txt beside the executable). One program's builds go to the driver one at a time on AMD,
// where a build behind a faulted one hung (issue 996); other drivers keep building a program's variants in parallel.
// RX_VK_DRIVER_FAULTS=0: off; =1: the default; =2: one at a time on every driver
constexpr VkResult kDriverFault = (VkResult)-77;   // (what the guards below answer for a fault in the driver)
int driver_faults_mode() {
    static const int mode = getenv("RX_VK_DRIVER_FAULTS") ? std::clamp(atoi(getenv("RX_VK_DRIVER_FAULTS")), 0, 2) : 1;
    return mode;
}
bool driver_faults_on() { return driver_faults_mode() != 0; }
std::string driver_faults_path() { return cache_path("hookshot_driver_faults.txt"); }
driver_faults::Quarantine& fault_quarantine() {
    static driver_faults::Quarantine q;
    static std::once_flag once;
    std::call_once(once, [] {   // (the earlier runs' faults, this device's driver only)
        std::string text;
        if (FILE* f = fopen(driver_faults_path().c_str(), "rb")) {
            char b[4096];
            size_t n;
            while ((n = fread(b, 1, sizeof b, f)) > 0) text.append(b, n);
            fclose(f);
        }
        const auto keys = driver_faults::parse(text, dev.props.vendorID, dev.props.driverVersion);
        q.load(keys);
        // (device_policy.h fault_gate_serial: AMD's Windows driver, whose hang issue 996 is; it was every AMD GPU, RADV's too)
        q.set_serial(driver_faults_mode() == 2 || dev.policy.fault_gate_serial);
        if (!keys.empty())
            logf("vk: %zu programs the driver faulted on in earlier runs are not asked of it (%s; RX_VK_DRIVER_FAULTS=0: they are)",
                 keys.size(), driver_faults_path().c_str());
        if (q.serial())
            logf("vk: one build of a program goes to the driver at a time (%s; RX_VK_DRIVER_FAULTS=0: as many as ask)",
                 driver_faults_mode() == 2 ? "RX_VK_DRIVER_FAULTS=2" : "AMD");
    });
    return q;
}
void driver_fault_note(uint64_t key) {
    static std::mutex mu;
    std::lock_guard lk(mu);
    const std::string path = driver_faults_path();
    FILE* f = fopen(path.c_str(), "ab");
    if (!f) {   // (a folder the game cannot write to: Program Files)
        static bool told;
        if (!told) told = true, logf("vk: %s could not be written: the faulted programs are kept out for this run only", path.c_str());
        return;
    }
    fseek(f, 0, SEEK_END);   // (in append mode ftell says 0 until the first write, on Windows: the file's own end)
    if (ftell(f) == 0)
        fputs("# Hookshot: programs the graphics driver's compiler faulted on, not asked of it again on that driver\n"
              "# (gpu/vk/driver_faults.h). <vendor> <driver version> <program key>. Delete this file to try them again.\n", f);
    fputs(driver_faults::line(dev.props.vendorID, dev.props.driverVersion, key).c_str(), f);
    fclose(f);
}

// (hookshot) A driver's compiler that faults on a shader (amdvlk64.dll on a mesh program, Hookshot issue 26) costs that one draw, not the game
std::atomic<int> st_driver_faults{0};
/// (the faults caught below, counted: a driver can stop answering after one, and the preparation's watchdog says so, vk_main.cpp
/// driver_faults)
void note_driver_fault() { st_driver_faults.fetch_add(1, std::memory_order_relaxed); }
#ifdef _WIN32
__declspec(noinline) VkResult guarded_graphics(VkPipelineCache cache, const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl) {
    __try {
        return vkCreateGraphicsPipelines(dev.device, cache, 1, ci, nullptr, pl);
    } __except (1) {
        *pl = VK_NULL_HANDLE;   // (whatever the driver wrote before it faulted is not a pipeline)
        note_driver_fault();
        logf("vk: the driver faulted compiling a graphics pipeline (exception %#lx): that one is left out", (unsigned long)GetExceptionCode());
        return (VkResult)-77;   // (a fault in the driver, as guarded_compute's: driver_faults.h keeps the program out)
    }
}
__declspec(noinline) VkResult guarded_compute_once(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
    __try {
        return vkCreateComputePipelines(dev.device, cache, 1, ci, nullptr, pl);
    } __except (1) {
        *pl = VK_NULL_HANDLE;   // (whatever the driver wrote before it faulted is not a pipeline)
        note_driver_fault();
        logf("vk: the driver faulted compiling a compute pipeline (exception %#lx): that one is left out", (unsigned long)GetExceptionCode());
        return (VkResult)-77;   // (a fault in the driver, told apart from what it returns)
    }
}
#else
// (linux: no structured exceptions, so the fault arrives as a signal on the compiling thread: an armed window covers the driver
// call alone, and a fault in it is reported as the call's failure, as the Windows guard above reports an access violation. A fault
// outside it goes to whoever had the signal before, so the game's own crash report is unchanged)
static thread_local sigjmp_buf fault_jmp;
static thread_local volatile sig_atomic_t fault_armed;
static struct sigaction fault_previous[2];
static const int fault_signals[2] = {SIGSEGV, SIGBUS};

static void on_driver_fault(int sig, siginfo_t* si, void* ctx) {
    if (fault_armed) {
        fault_armed = 0;
        siglongjmp(fault_jmp, 1);
    }
    const struct sigaction* p = &fault_previous[sig == SIGSEGV ? 0 : 1];
    if ((p->sa_flags & SA_SIGINFO) && p->sa_sigaction) {
        p->sa_sigaction(sig, si, ctx);
    } else if (p->sa_handler && p->sa_handler != SIG_IGN && p->sa_handler != SIG_DFL) {
        p->sa_handler(sig);
    } else {
        signal(sig, SIG_DFL);
    }
}

static void install_driver_fault_guard() {
    static std::once_flag once;
    std::call_once(once, [] {
        for (int i = 0; i < 2; i++) {
            struct sigaction sa;
            memset(&sa, 0, sizeof sa);
            sa.sa_sigaction = on_driver_fault;
            sa.sa_flags = SA_SIGINFO;
            sigemptyset(&sa.sa_mask);
            sigaction(fault_signals[i], &sa, &fault_previous[i]);
        }
    });
}

VkResult guarded_graphics(VkPipelineCache cache, const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl) {
    install_driver_fault_guard();
    if (sigsetjmp(fault_jmp, 1) != 0) {
        *pl = VK_NULL_HANDLE;   // (whatever the driver wrote before it faulted is not a pipeline)
        note_driver_fault();
        logf("vk: the driver faulted compiling a graphics pipeline: that one is left out");
        return (VkResult)-77;
    }
    fault_armed = 1;
    const VkResult r = vkCreateGraphicsPipelines(dev.device, cache, 1, ci, nullptr, pl);
    fault_armed = 0;
    return r;
}
VkResult guarded_compute_once(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
    install_driver_fault_guard();
    if (sigsetjmp(fault_jmp, 1) != 0) {
        *pl = VK_NULL_HANDLE;   // (whatever the driver wrote before it faulted is not a pipeline)
        note_driver_fault();
        logf("vk: the driver faulted compiling a compute pipeline: that one is left out");
        return (VkResult)-77;
    }
    fault_armed = 1;
    const VkResult r = vkCreateComputePipelines(dev.device, cache, 1, ci, nullptr, pl);
    fault_armed = 0;
    return r;
}
#endif
VkPipeline build_graphics(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, int part = -1,
                          bool* cold = nullptr);
/// (hookshot) A compute pipeline the driver's compiler fails on (AMD's crashes, -13, or faults, -77, on some programs) is built
/// again without its optimizer, as a graphics one is (build_graphics): run a little slower rather than left out. Not for the
/// memory errors, which the optimizer does not cause, nor for a cold probe's answer (VK_PIPELINE_COMPILE_REQUIRED: positive)
VkResult guarded_compute(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
    const VkResult r = guarded_compute_once(cache, ci, pl);
    if (r >= 0 || r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
        (ci->flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT))
        return r;
    VkComputePipelineCreateInfo again = *ci;
    again.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    *pl = VK_NULL_HANDLE;
    const VkResult second = guarded_compute_once(cache, &again, pl);
    static std::atomic<int> told{0};
    if (told++ < 20)
        logf("vk: a compute pipeline failed (%d) with the driver's optimizer: %s", (int)r,
             second == VK_SUCCESS ? "made without it" : "not made without it either");
    return second;
}
bool gpl_libs_cached() {
    static const bool on = !env_on("RX_VK_GPL_NO_CACHE");
    return on;
}
// (hookshot) vkCreateShaderModule has no guard of its own. Translation workers call it, and on Windows their stack into
// igvk64 is short: an access violation there is not the pipeline guards' "driver faulted compiling" line, and it ends
// the process. The same answer as those guards: that program is left out.
#ifdef _WIN32
__declspec(noinline) VkResult guarded_shader_module(const VkShaderModuleCreateInfo* ci, VkShaderModule* m) {
    __try {
        return vkCreateShaderModule(dev.device, ci, nullptr, m);
    } __except (1) {
        *m = VK_NULL_HANDLE;   // (whatever the driver wrote before it faulted is not a module)
        note_driver_fault();
        logf("vk: the driver faulted creating a shader module (exception %#lx): that program is left out",
             (unsigned long)GetExceptionCode());
        return (VkResult)-77;
    }
}
#else
VkResult guarded_shader_module(const VkShaderModuleCreateInfo* ci, VkShaderModule* m) {
    install_driver_fault_guard();
    if (sigsetjmp(fault_jmp, 1) != 0) {
        *m = VK_NULL_HANDLE;
        note_driver_fault();
        logf("vk: the driver faulted creating a shader module: that program is left out");
        return (VkResult)-77;
    }
    fault_armed = 1;
    const VkResult r = vkCreateShaderModule(dev.device, ci, nullptr, m);
    fault_armed = 0;
    return r;
}
#endif
thread_local bool driver_cache_only = false;   // (predict_run's: pipelines for the driver's own cache, not into ours)
thread_local bool quick_build = false;   // (hookshot: quick_pipeline's, without the driver's optimizer and into no cache)
thread_local bool gate_busy = false;   // (hookshot: set by build_graphics when the driver-fault gate answered kBusy: quick_pipeline's)
thread_local bool slow_full_build = false;   // (hookshot: the worker's full build in graphics_pipeline's job: RX_VK_TEST_SLOW_MS's)
/// (hookshot, a test from Mako) RX_VK_TEST_SLOW_MS=<ms> holds each full build that long, as AMD's optimizer on a cold cache, to
/// see quick pipelines and the brief wait at work on any GPU. Off unless set; a value not a number is ignored. Read once
int test_slow_ms() {
    static const int slow_ms = [] {
        const char* e = getenv("RX_VK_TEST_SLOW_MS");
        bool bad = false;
        const int ms = quick_pipe::slow_ms(e, &bad);
        if (bad) logf("vk: RX_VK_TEST_SLOW_MS=%s is not a number of milliseconds: ignored", e);
        if (ms > 0) logf("vk: every full pipeline build is held %d ms (RX_VK_TEST_SLOW_MS, a test)", ms);
        return ms;
    }();
    return slow_ms;
}
// (with driver_cache_only: the cache handed to the driver instead of none, predicted_check's empty one, as play hands its generation's)
thread_local VkPipelineCache driver_ask_cache = VK_NULL_HANDLE;
/// (with the pipeline cache: from it, or in play into the play generation's, create_cached)
/// (the guards with the cache held from a fetch, PcacheFetch)
VkResult note_out_of_memory(VkResult r) {
    if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY)
        st_driver_out_of_memory.fetch_add(1, std::memory_order_relaxed);
    return r;
}
VkResult guarded_graphics_held(VkPipelineCache cache, const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl) {
    PcacheUse use(cache);
    return note_out_of_memory(guarded_graphics(cache, ci, pl));
}
VkResult guarded_compute_held(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
    PcacheUse use(cache);
    return note_out_of_memory(guarded_compute(cache, ci, pl));
}
bool compiles_nothing(VkPipelineCreateFlags flags) { return (flags & VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT) != 0; }
/// (hookshot) Windows Intel: an ask, or a dispatch's module, left to a worker because another thread held Intel's compiler gate
/// (IntelDriverGate: a worker's compile). There is no lookup of the driver's cache or ours that does not enter the driver, so
/// the pipeline may be in a cache all the same: its draw or dispatch is left out, as a cold one is, until a worker has made it.
/// Counted and logged (the first 8, then at each power of two), so the Intel runs show how often it happens
std::atomic<int64_t> st_intel_gate_busy{0};
VkResult intel_gate_busy() {
    gate_busy = true;   // quick_pipeline can retry when the compiler is free
    const int64_t n = ++st_intel_gate_busy;
    if (n <= 8 || (n & (n - 1)) == 0)
        logf("vk: asked cold: Intel's gate busy (a worker compiles); left to a worker, %lld so far", (long long)n);
    return VK_PIPELINE_COMPILE_REQUIRED;
}
VkResult cached_graphics(const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl) {
#ifdef TRACY_ENABLE
    HT_SCOPE("cached_graphics (pipeline compile)");
#endif
    IntelDriverGate gate(compiles_nothing(ci->flags), true);   // (a preparation thread's: side by side, intel_prepare_open)
    if (gate.busy) return intel_gate_busy();
    return create_cached(guarded_graphics_held, ci, pl);
}
/// (hookshot) The driver's compute compiles, as many at once as the memory holds (compute_gate.h): on Mesa's Intel driver each
/// of the game's large compute programs holds gigabytes while it compiles, and the preparation's workers ran one each, which
/// an Iris Xe PC with 32 GB did not hold. On RADV with an integrated GPU too: a Radeon 760M PC with 12 GB was killed for memory
/// while the first start made the mesh fallback's compute pipelines on every CPU. RX_VK_COMPUTE_AT_ONCE=<n>: at most
/// n on any GPU, 0: no limit, as before
uint64_t memory_available_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? (uint64_t)(m.ullAvailPhys >> 20) : compute_gate::kUnknownMB;
#else
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return compute_gate::kUnknownMB;
    unsigned long long kb = 0, v = 0;
    char key[64];
    bool found = false;
    while (!found && fscanf(f, "%63s %llu kB\n", key, &v) == 2)
        if (!strcmp(key, "MemAvailable:")) kb = v, found = true;
    fclose(f);
    // (the least of the system's and the cgroup's: kUnknownMB is the largest, so either one alone stands when the other is not known)
    return std::min(found ? (uint64_t)(kb >> 10) : compute_gate::kUnknownMB, compute_gate::cgroup_memory().available_mb);
#endif
}
compute_gate::Gate& compute_compiles() {
    static compute_gate::Gate gate;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* e = getenv("RX_VK_COMPUTE_AT_ONCE");
        if (e && *e) {
            const unsigned n = (unsigned)std::clamp(atoi(e), 0, 64);
            gate.set(n, false);
            if (n) logf("vk: compute pipelines: at most %u compiled at once (RX_VK_COMPUTE_AT_ONCE)", n);
            else logf("vk: compute pipelines: no limit on how many are compiled at once (RX_VK_COMPUTE_AT_ONCE=0)");
        } else if (dev.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA) {
            const uint64_t avail = memory_available_mb();
            const uint64_t budget = compute_gate::budget_for(avail);
            gate.set(0, true, budget);
            if (!budget)
                logf("vk: compute pipelines: one compiled at a time (Mesa's Intel compiler holds a lot for a large one, and %s; "
                     "RX_VK_COMPUTE_AT_ONCE=<n> or 0 says otherwise)",
                     avail == compute_gate::kUnknownMB ? "the system does not say how much memory is available"
                                                       : "little memory is available");
            else
                logf("vk: compute pipelines: as many compiled at once as %llu MB hold (%llu MB available now, %llu left to the PC), "
                     "%llu MB for one held to 32 lanes (%llu MB with %u phis or more, %llu MB for a mesh fallback stage) and %llu MB for "
                     "another, at most %u (Mesa's Intel compiler; RX_VK_COMPUTE_AT_ONCE=<n> or 0 says otherwise)",
                     (unsigned long long)budget, (unsigned long long)avail, (unsigned long long)compute_gate::kReserveMB,
                     (unsigned long long)compute_gate::kPinnedMB, (unsigned long long)compute_gate::kHeavyMB,
                     compute_gate::kHeavyPhis, (unsigned long long)compute_gate::kMeshMB,
                     (unsigned long long)compute_gate::kFreeMB, compute_gate::kMost);
        } else if (dev.facts.radv() && compute_gate::radv_by_memory(dev.facts.integrated(), dev.facts.ram >> 20)) {
            // (the GPU's memory is the system's: the compiles, the game and the GPU's own allocations all take from the RAM. Only
            // under 13 GB of it: the Steam Deck and the 16 GB APUs keep main's rule, nobody has run this there)
            const uint64_t avail = memory_available_mb();
            const uint64_t budget = compute_gate::budget_for(avail);
            gate.set(0, true, budget);
            if (!budget)
                logf("vk: compute pipelines: one compiled at a time (RADV, an integrated GPU, and %s; RX_VK_COMPUTE_AT_ONCE=<n> or 0 "
                     "says otherwise)", avail == compute_gate::kUnknownMB ? "the system does not say how much memory is available"
                                                                          : "little memory is available");
            else
                logf("vk: compute pipelines: as many compiled at once as %llu MB hold (%llu MB available now, %llu left to the PC), "
                     "%llu MB for one of %u phis or more, %llu MB of %u or more, %llu MB for another, at most %u; one at a time without "
                     "the memory (RADV, an integrated GPU, %llu MB of RAM; RX_VK_COMPUTE_AT_ONCE=<n> or 0 says otherwise)",
                     (unsigned long long)budget, (unsigned long long)(avail == compute_gate::kUnknownMB ? 0 : avail),
                     (unsigned long long)compute_gate::kReserveMB, (unsigned long long)compute_gate::kRadvBigMB,
                     compute_gate::kRadvBigPhis, (unsigned long long)compute_gate::kRadvMidMB, compute_gate::kRadvMidPhis,
                     (unsigned long long)compute_gate::kRadvMB, compute_gate::kMost, (unsigned long long)(dev.facts.ram >> 20));
        }
    });
    return gate;
}
/// (hookshot) The driver's compute compiles under way, by their program: a preparation that stands still says which program the
/// driver is on and for how long (compute_compiles_log), and a compile that took kSlowS seconds or more is named when it ends.
/// Before, a stall said only "waiting on the GPU driver": on an Iris Xe it was one program at 32 lanes for over five minutes,
/// and nothing said which. RX_VK_COMPUTE_LOG_MS=<ms>: every compile that long is named when it ends, however many
struct ComputeCompiling {
    ComputeDescribed of;
    bool pinned;
    std::chrono::steady_clock::time_point began;
};
std::mutex compiling_mu;
std::list<ComputeCompiling> compiling;
std::atomic<unsigned> sdk_compiles{0};
bool prepare_compiling_seen() {
    std::lock_guard lk(compiling_mu);
    return !compiling.empty() || sdk_compiles.load(std::memory_order_acquire) != 0;
}
bool compute_pinned(const VkComputePipelineCreateInfo& ci) {
    for (auto n = (const VkBaseInStructure*)ci.stage.pNext; n; n = n->pNext)
        if (n->sType == VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO) return true;
    return false;
}
void compute_compiled_log(const ComputeCompiling& c, int64_t ms) {
    static const int asked_ms = getenv("RX_VK_COMPUTE_LOG_MS") ? std::max(0, atoi(getenv("RX_VK_COMPUTE_LOG_MS"))) : -1;
    static std::atomic<int> told{0};
    constexpr int64_t kSlowMs = 10000;
    if (asked_ms >= 0 ? ms < asked_ms : ms < kSlowMs || told++ >= 60) return;
    const compute_pin::Uses& u = c.of.uses;
    const bool ops = compute_pin::sees_lanes_ops(u);
    logf("vk: compute pipeline of %016llx%s: the driver compiled it in %.1f s (%s; workgroup %ux%ux%u, shared memory %u bytes)",
         (unsigned long long)c.of.key, c.of.fallback ? " (mesh fallback)" : "", ms / 1000.0,
         !c.pinned ? "at the lanes the driver chose" : ops ? "held to 32 lanes, for its subgroup operations"
         : u.shared_memory ? "held to 32 lanes, for its shared memory in one warp" : "held to 32 lanes",
         c.of.workgroup[0], c.of.workgroup[1], c.of.workgroup[2], u.shared_memory);
    if (asked_ms >= 0) {   // (diagnosis: when it ended, by the boot clock /proc/uptime reads, to match memory samples to it)
#ifndef _WIN32
        timespec ts{};
        clock_gettime(CLOCK_BOOTTIME, &ts);
        logf("vk: compute pipeline of %016llx: ended at %.2f after %.2f s", (unsigned long long)c.of.key, ts.tv_sec + ts.tv_nsec / 1e9,
             ms / 1000.0);
#endif
    }
    if (asked_ms >= 0)   // (diagnosis: what of it tells its lanes)
        logf("vk: compute pipeline of %016llx: uses%s%s%s%s%s%s%s", (unsigned long long)c.of.key, u.vote ? " votes" : "",
             u.mask ? " lane-masks" : "", u.invocation_id ? " lane-index" : "", u.shuffles ? " shuffles" : "",
             u.quad_shuffles ? " quad-shuffles" : "", u.matrix ? " matrices" : "", u.fswzadd ? " fswzadd" : "");
}
/// The compiles that have run over_s seconds or more, in one line, at most once every 10 s (the preparation's steps call it
/// as they wait)
void compute_compiles_log(int64_t over_s = 30) {
    static const bool asked = getenv("RX_VK_COMPUTE_LOG_MS") != nullptr;   // (diagnosis: from 5 s)
    if (asked) over_s = std::min<int64_t>(over_s, 5);
    static std::atomic<int64_t> next_ms{0};
    const auto now = std::chrono::steady_clock::now();
    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    int64_t due = next_ms.load(std::memory_order_relaxed);
    if (now_ms < due || !next_ms.compare_exchange_strong(due, now_ms + 10000)) return;
    char line[1024];
    int n = 0, long_ones = 0;
    size_t running = 0;
    {
        std::lock_guard lk(compiling_mu);
        running = compiling.size();
        for (const ComputeCompiling& c : compiling) {
            const int64_t s = std::chrono::duration_cast<std::chrono::seconds>(now - c.began).count();
            if (s < over_s) continue;
            if (long_ones++ < (asked ? 24 : 6) && n < (int)sizeof line - 64)
                n += snprintf(line + n, sizeof line - n, "%s%016llx %lld s%s", n ? ", " : "", (unsigned long long)c.of.key,
                              (long long)s, c.pinned ? " (32 lanes)" : "");
        }
    }
    if (long_ones)
        logf("vk: compute pipelines: the driver is compiling %zu, %d of them for %lld s or longer: %s%s", running, long_ones,
             (long long)over_s, line, long_ones > 6 ? ", ..." : "");
}
/// (hookshot) Whether the compile described (compute_describe) is one the driver is not given: a game compute program held to
/// 32 lanes on Mesa's Intel driver with more phis than its compiler finishes (compute_pin.h too_big). Named in the log, a few
/// times. RX_VK_COMPUTE_PHI_LIMIT=<n>: another limit, 0 none
bool compute_refused(const ComputeDescribed& d, bool pinned) {
    static const uint32_t limit = [] {
        uint32_t v = compute_pin::kIntelPhiLimit;
        const char* e = getenv("RX_VK_COMPUTE_PHI_LIMIT");
        if (e && !fallback_size::parse_limit(e, v)) {
            v = compute_pin::kIntelPhiLimit;
            logf("vk: RX_VK_COMPUTE_PHI_LIMIT=%s is not a number from 0 to 4294967295: ignored", e);
        }
        return v;
    }();
    if (d.fallback ||
        !compute_pin::too_big(dev.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, pinned, d.phis, limit))
        return false;
    static std::atomic<int> told{0};
    if (told++ < 8)
        logf("vk: compute pipeline of %016llx is not built: held to 32 lanes with %u phis, more than Mesa's Intel compiler "
             "finishes (the most %u, RX_VK_COMPUTE_PHI_LIMIT); its dispatches are left out",
             (unsigned long long)d.key, d.phis, limit);
    return true;
}
/// (hookshot) What a compile weighs at the gate (compute_gate.h): on RADV by its phis (weight_radv), elsewhere by what Mesa's
/// Intel compiler was measured to take (weight: held to 32 lanes or not, its phis, a mesh fallback stage or not)
uint64_t compute_weight(const VkComputePipelineCreateInfo& ci) {
    if (dev.facts.radv()) return compute_gate::weight_radv(compute_described.phis);
    return compute_gate::weight(compute_pinned(ci), compute_described.phis, compute_described.fallback);
}
/// A compile's turn at the gate, for as long as it lives. A create call that only asks the caches compiles nothing and takes
/// none. ok(): false when the preparation was stopped (the window closed) while this one waited, or the program is one the
/// driver is not given (compute_refused): it is not compiled.
/// A long compile hands the memory it freed back to the system as it leaves (compute_gate.h trims)
struct ComputeTurn {
    explicit ComputeTurn(const VkComputePipelineCreateInfo& ci)
        : gate((ci.flags & VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT) ? nullptr : &compute_compiles()),
          weight(compute_weight(ci)) {
        if (gate && compute_refused(compute_described, compute_pinned(ci))) gate = nullptr, stopped = true;
        if (gate && !gate->enter(weight, memory_available_mb, [] { return prepare_stopped(); })) gate = nullptr, stopped = true;
        began = std::chrono::steady_clock::now();   // (the turn is taken: the compile starts)
        if (gate) {
            std::lock_guard lk(compiling_mu);
            compiling.push_front({compute_described, compute_pinned(ci), began});
            mine = compiling.begin();
        }
    }
    ~ComputeTurn() {
        if (!gate) return;
        const int64_t held_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - began).count();
        ComputeCompiling was;
        {
            std::lock_guard lk(compiling_mu);
            was = *mine;
            prepare_compiled();   // (before it leaves the list: the watchdog cannot fall back to the short limit in between)
            compiling.erase(mine);
        }
        compute_compiled_log(was, held_ms);
        const bool gated = gate->on();
        gate->leave(weight);
#if defined(__GLIBC__)
        // (after the turn is given back: the next compile does not wait for it)
        if (compute_gate::trims(gated, held_ms)) malloc_trim(0);
#else
        (void)gated, (void)held_ms;
#endif
    }
    ComputeTurn(const ComputeTurn&) = delete;
    ComputeTurn& operator=(const ComputeTurn&) = delete;
    bool ok() const { return !stopped; }

private:
    compute_gate::Gate* gate;
    uint64_t weight;
    bool stopped = false;
    std::chrono::steady_clock::time_point began;
    std::list<ComputeCompiling>::iterator mine;
};
VkResult cached_compute(const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
#ifdef TRACY_ENABLE
    HT_SCOPE("cached_compute (pipeline compile)");
#endif
    // (the turn first, Intel's lock inside it: a worker that held the lock while it waited for a turn would block the compiles
    // that have theirs, and those hold the memory it waits for)
    const ComputeTurn turn(*ci);
    if (!turn.ok()) return *pl = VK_NULL_HANDLE, VK_ERROR_OUT_OF_HOST_MEMORY;   // (stopped while it waited, or refused: not compiled)
    IntelDriverGate gate(compiles_nothing(ci->flags));
    if (gate.busy) return intel_gate_busy();
    return create_cached(guarded_compute_held, ci, pl);
}
/// (hookshot) What a compute program's pipeline is made from: its one stage, pinned to 32 lanes where the GPU can and the
/// program needs it (compute_pin_needed), and its layout (ensure_layout first). noopt: RX_VK_NOOPT is followed, as play's builds follow it (the precompile's do not)
VkComputePipelineCreateInfo compute_pipeline_info(const Program& q, bool noopt) {
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = q.stages[0].module;
    ci.stage.pName = "main";
    pin_compute(ci.stage, q);
    ci.layout = q.layout;
    static const bool noopt_on = env_on("RX_VK_NOOPT");
    static const uint64_t amd_noopt_key = []() -> uint64_t {
        const char* value = getenv("RX_VK_AMD_COMPUTE_NOOPT_KEY");
        if (!value || std::strlen(value) != 16) return 0;
        for (const char* p = value; *p; ++p)
            if (!std::strchr("0123456789abcdefABCDEF", *p)) return 0;
        return std::strtoull(value, nullptr, 16);
    }();
    const bool targeted = dev.props.vendorID == 0x1002 && amd_noopt_key && amd_noopt_key == q.key;
    if (targeted || (noopt && noopt_on)) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    return ci;
}
/// (hookshot) Before play on Intel's one-at-a-time builds (precompile_run, precompile_warm): ci asked without the optimizer, the
/// pipeline a dispatch's first build asks for (Intel quick compute builds), so that a dispatch finds it in the cache. With the
/// optimizer, one of the game's compute pipelines took Intel's compiler up to 389.5 s on a UHD Graphics (i3-1115G4), past the
/// shader preparation's 300 s with nothing finished: the preparation was stopped (hookshot_prepare_skipped) and every later
/// start built each shader when first drawn. true: asked so (the optimized build is then owed to its first dispatch)
bool precompile_compute_quick(VkComputePipelineCreateInfo& ci) {
    if (!serial_builds_on() || !intel_quick_compute() || (ci.flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT)) return false;
    ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    return true;
}
/// (hookshot) That pipeline asked of the caches without compiling (create_cached: the driver's own, then the start's): null when
/// neither has it, and where the driver cannot be asked that way
VkPipeline compute_pipeline_cached(const Program& q, bool noopt) {
    if (!dev.vk13.pipelineCreationCacheControl) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo asked = compute_pipeline_info(q, noopt);
    asked.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    VkPipeline cached = VK_NULL_HANDLE;
    return cached_compute(&asked, &cached) == VK_SUCCESS ? cached : VK_NULL_HANDLE;
}
/// (a graphics pipeline made with the cache chosen for it: pcache through the play caches, another or none as it is)
VkResult graphics_with(VkPipelineCache cache, const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl) {
    if (cache && cache == pcache) return cached_graphics(ci, pl);
    IntelDriverGate gate(quick_build || compiles_nothing(ci->flags), true);
    if (gate.busy) return intel_gate_busy();
    return guarded_graphics_held(cache, ci, pl);
}

/// A target's write mask: the state's channel mask, nothing for a target the fragment shader leaves unwritten
/// (Maxwell keeps its texels; Vulkan writes undefined values: TOTK's maxwell.md 1.1)
uint32_t target_write_mask(const Program& p, const NvnState& s, int i) {
    const Stage* fs = nullptr;
    for (const Stage& st : p.stages)
        if (st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT) fs = &st;
    if (!fs || i >= 8 || !fs->info.stores_frag_color[i]) return 0;
    // (amd-fix, Lichii #353) and none of the components it does not write: Vulkan writes undefined values there, AMD its
    // export registers' garbage (NaN and saturated pixels above OoT's horizon, which the bloom spread)
    const uint32_t written = fs->info.frag_color_components[i];
    return (s.chmask.mask >> (4 * i)) & 0xF & (!written ? 0xFu : (uint32_t)written);
}
/// (hookshot) What target_write_mask reads of a program, as one word: whether its fragment stage stores each output and which
/// components it writes of it (6 bits each). Two programs with the same word make the same masks from the same state
uint64_t program_wm_sig(const Program& p) {
    if (const int64_t known = p.wm_sig.load(std::memory_order_relaxed); known >= 0) return (uint64_t)known;
    const Stage* fs = nullptr;
    for (const Stage& st : p.stages)
        if (st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT) fs = &st;
    uint64_t sig = 0;
    for (int i = 0; fs && i < 8; i++)
        if (fs->info.stores_frag_color[i]) {
            const uint32_t written = fs->info.frag_color_components[i];
            sig |= (uint64_t)(1u | (written ? 2u : 0u) | (written & 0xFu) << 2) << (6 * i);
        }
    p.wm_sig.store((int64_t)sig, std::memory_order_relaxed);   // (replay threads may ask at once: the same word)
    return sig;
}

// ---- (hookshot) The precompile, as TotK Recomp's (runtime/vk_pipeline.c vk_pipeline_prewarm, keys in pipeline-keys.bin): each
// pipeline the game asks for is noted in hookshot_pipelines.bin, by what built it; a release build, before the game starts,
// makes every kept program ready and builds every noted pipeline on all the CPUs (the driver's own cache, hookshot_vk_pipelines.bin,
// makes that quick), so that nothing is built during play. RX_VK_PRECOMPILE=1: in any build; =0: off. A build with
// RX_RELEASE=1 (nx2/build.py) has it on. The notes are written in every build ----
std::mutex note_mu;
FILE* note_file = nullptr;
std::unordered_set<uint64_t> noted;   // (a pipeline once in the file, however many runs meet it)
std::string pipe_notes_path() { return cache_path("hookshot_pipelines.bin"); }
/// (hookshot) hookshot_pipelines.bin.built: a precompile that got to its end built every note there was into the pipeline cache
/// (its stamp, note count and compute coverage). RX_VK_LOAD=create loads pipelines from the caches alone, and only once this says they are there.
/// What adds notes without building them (state_list_prepare) removes it; a note play adds was built by play, into its cache
std::string precompile_built_path() { return cache_sets_switch(), pipe_notes_path() + ".built"; }   // (gpu_sets.h first)
void precompile_built_write(size_t notes, uint64_t compute_coverage) {
    // (not by a --prepare-shaders run: on NVIDIA the pipeline cache that process writes does not answer the play process
    // ("kept 0 of 32"), so the first play after a preparation found none of the noted pipelines in a cache and compiled them
    // in play with draws left out. Left unwritten, the first play start builds them itself, and its cache does answer)
    if (env_on("HOOKSHOT_PREPARE_ONLY")) return;
    if (FILE* f = fopen(precompile_built_path().c_str(), "wb")) {
        const uint64_t rec[] = {program_cache_stamp(), (uint64_t)notes, compute_coverage};
        fwrite(rec, sizeof rec, 1, f);
        fclose(f);
    }
}

// The notes file, format 2. A note is a program (key, variant) with the draw state its pipeline was made for; it holds no
// compiled code, so a change of the shader translation does not make it wrong, only its pipelines stale. The header says what the
// notes were written for:
// format      this layout of the header and of a record
// key_version bumped by hand when pipeline_keyor the way a note is built changes meaning
// note_size   sizeof(PipeNote), and state: HOOKSHOT_STATE_STAMP (nx2/build.py: the hash of vk_state.h, vk_api.h, vk_draw.h, rx_nvn.h)
// translator  program_cache_stampwhen it was written (HOOKSHOT_SHADER_STAMP: the shader translation)
// A file whose format, key_version, note_size or state differs is of no use and is started again. One whose translator differs is
// CARRIED: its notes are kept, every pipeline is made again from them with the new translation (never trusted as built: the
// precompile recomputes each key from the state) and the file is rewritten with the new translator stamp. The old format (the
// translator stamp and the size, 16 bytes) is read when its stamp is the current one. RX_VK_NOTES_STRICT=1: a translator
// that differs drops the notes, as before.
constexpr uint32_t kNoteFormat = 2, kPipeKeyVersion = 1;
#ifdef HOOKSHOT_STATE_STAMP
constexpr uint64_t kStateStamp = HOOKSHOT_STATE_STAMP;
#else
constexpr uint64_t kStateStamp = 0;   // (a build that does not say: notes are never carried to another translator)
#endif
struct NoteHead {
    char magic[8];
    uint32_t format, key_version;
    uint64_t note_size, state, translator;
};
NoteHead note_head_now() {
    NoteHead h{};
    std::memcpy(h.magic, "HKPIPES2", 8);
    h.format = kNoteFormat, h.key_version = kPipeKeyVersion, h.note_size = sizeof(PipeNote), h.state = kStateStamp;
    h.translator = program_cache_stamp();
    return h;
}
enum class NoteFile { Missing, Current, Carried, Old, Unusable };
/// What the file at f (positioned at its start) is; first is where its records begin
NoteFile note_file_state(FILE* f, long* first) {
    uint8_t buf[sizeof(NoteHead)]{};
    const size_t got = fread(buf, 1, sizeof buf, f);
    const NoteHead now = note_head_now();
    if (got >= sizeof(NoteHead) && !std::memcmp(buf, now.magic, 8)) {
        NoteHead h;
        std::memcpy(&h, buf, sizeof h);
        *first = (long)sizeof(NoteHead);
        if (h.format != now.format || h.key_version != now.key_version || h.note_size != now.note_size || h.state != now.state ||
            (now.state == 0 && h.translator != now.translator))
            return NoteFile::Unusable;
        if (h.translator == now.translator) return NoteFile::Current;
        return getenv("RX_VK_NOTES_STRICT") && atoi(getenv("RX_VK_NOTES_STRICT")) == 0 ? NoteFile::Carried : NoteFile::Unusable;   // (carried only with =0: off by default here)
    }
    if (got >= 16) {   // (the old header: the translator stamp, sizeof(PipeNote))
        uint64_t head[2];
        std::memcpy(head, buf, 16);
        *first = 16;
        if (head[0] == now.translator && head[1] == sizeof(PipeNote)) return NoteFile::Old;
    }
    return NoteFile::Unusable;
}
/// (hookshot) Targets a pipeline can be made with: at most 8 colour targets, a sample count Vulkan has, a topology it has.
/// The notes and the state list are files: what they say is checked before a pipeline is made from it
bool targets_ok(uint32_t n, uint32_t samples, uint32_t topo) {
    return n <= 8 && samples >= 1 && samples <= 64 && !(samples & (samples - 1)) && topo <= VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
}
/// (hookshot) Formats a pipeline can be made with: each colour target's none or one this device draws to, the depth target's
/// one it uses as depth. They come from the notes and the shareable state list, and any number reached the driver as a
/// VkFormat (a crash in the shader preparation, at every start, until the file was deleted)
bool formats_ok(uint32_t n, const uint32_t color[8], uint32_t has_depth, uint32_t depth) {
    if (!dev.physical) return true;   // (no device to ask: the pipeline is not made either)
    auto supports = [](uint32_t f, VkFormatFeatureFlags need) {
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(dev.physical, (VkFormat)f, &fp);
        return (fp.optimalTilingFeatures & need) == need;
    };
    for (uint32_t i = 0; i < n && i < 8; i++)
        if (color[i] != VK_FORMAT_UNDEFINED && !supports(color[i], VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) return false;
    return !has_depth || depth == VK_FORMAT_UNDEFINED ||
           (is_depth_format((VkFormat)depth) && supports(depth, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT));
}
/// (hookshot) a note read from hookshot_pipelines.bin: its counts in range, its booleans 0 or 1, and no host pointers (the
/// render targets' objects of the run that wrote it mean nothing here)
bool note_sane(PipeNote& n) {
    if (!targets_ok(n.n, n.samples, n.topo) || n.s.nvattr < 0 || n.s.nvattr > 16 ||
        !formats_ok(n.n, n.color_format, n.has_depth, n.depth_format))
        return false;
    n.s.rt = nvn_p_set_rt{};
    for (bool* b : {&n.s.has_prog, &n.s.restart, &n.s.depth_clamp, &n.s.raster_discard, &n.s.render_enabled}) {
        unsigned char v;
        std::memcpy(&v, b, 1);
        v = v != 0;
        std::memcpy(b, &v, 1);
    }
    return n.s.prog.nstage <= 6;
}
/// The records of a notes file from where they start, each pipeline (by the key it was written with) once
std::vector<std::unique_ptr<PipeNote>> note_records(FILE* f, long first) {
    std::vector<std::unique_ptr<PipeNote>> notes;
    std::unordered_set<uint64_t> seen;
    fseek(f, first, SEEK_SET);
    for (;;) {
        auto rec = std::make_unique<PipeNote>();
        if (fread(rec.get(), sizeof(PipeNote), 1, f) != 1) break;
        if (!note_sane(*rec)) continue;
        if (seen.insert(rec->key).second) notes.push_back(std::move(rec));
    }
    return notes;
}
/// The notes of this build (hookshot_pipelines.bin), each pipeline once; carried is set when they come from another translator
std::vector<std::unique_ptr<PipeNote>> pipe_notes_read(bool* carried = nullptr) {
    std::vector<std::unique_ptr<PipeNote>> notes;
    if (carried) *carried = false;
    if (FILE* f = fopen(pipe_notes_path().c_str(), "rb")) {
        long first = 0;
        const NoteFile st = note_file_state(f, &first);
        if (st == NoteFile::Current || st == NoteFile::Carried || st == NoteFile::Old) {
            notes = note_records(f, first);
            if (carried) *carried = st == NoteFile::Carried;
        }
        fclose(f);
    }
    return notes;
}
/// The file open for appending, in the current format (note_mu held). A file of the old format or another translator is rewritten
/// first with its notes (in a file beside it, then moved over it); an unusable one is started again
void notes_open_carried_locked() {
    if (note_file) return;
    const std::string path = pipe_notes_path();
    long first = 0;
    NoteFile st = NoteFile::Missing;
    std::vector<std::unique_ptr<PipeNote>> keep;
    if (FILE* f = fopen(path.c_str(), "rb")) {
        st = note_file_state(f, &first);
        if (st == NoteFile::Carried || st == NoteFile::Old || st == NoteFile::Current) keep = note_records(f, first);
        if (st == NoteFile::Current) {   // (a record cut short by a killed run: the file is rewritten whole, or the next note would be misaligned)
            fseek(f, 0, SEEK_END);
            if ((ftell(f) - first) % (long)sizeof(PipeNote)) st = NoteFile::Old;
        }
        fclose(f);
    }
    for (const auto& n : keep) noted.insert(n->key);
    if (st == NoteFile::Current) {
        note_file = fopen(path.c_str(), "ab");
        if (note_file) return;
    }
    const NoteHead head = note_head_now();
    const std::string tmp = path + ".new";
    if (FILE* f = fopen(tmp.c_str(), "wb")) {
        fwrite(&head, sizeof head, 1, f);
        for (const auto& n : keep) fwrite(n.get(), sizeof(PipeNote), 1, f);
        const bool ok = fflush(f) == 0;
        fclose(f);
        if (ok && MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
            if (st == NoteFile::Carried || st == NoteFile::Old)
                logf("vk: pipeline notes: %zu carried over to this build (%s)", keep.size(),
                     st == NoteFile::Old ? "old file format" : "the shader translation changed: their pipelines are made again");
            note_file = fopen(path.c_str(), "ab");
            return;
        }
        remove(tmp.c_str());
    }
    noted.clear();   // (could not rewrite: started again)
    note_file = fopen(path.c_str(), "wb");
    if (note_file) fwrite(&head, sizeof head, 1, note_file);
}
/// A note's targets, as its draw had them
Targets note_targets(const PipeNote& n) {
    Targets at{};
    at.n = (int)n.n;
    at.samples = (VkSampleCountFlagBits)n.samples;
    at.depth_format = (VkFormat)n.depth_format;
    at.depth = n.has_depth ? (Texture*)1 : nullptr;   // (never read through: whether there is one)
    for (int k = 0; k < 8; k++) at.color_format[k] = (VkFormat)n.color_format[k];
    return at;
}
/// A note, appended (on a worker: the file is never touched on the replay thread)
void state_record(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, const uint8_t color_types[8]);
uint64_t pipeline_key(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo);
/// A list a crash or a full disk left ending in part of a record: cut back to its last whole record, which are all kept. The
/// records added after a part one were never read back (their pipelines built again in play at each start)
static void cut_part_record(const std::string& path, uint64_t head, uint64_t rec) {
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(path, ec);
    if (ec || size < head) return;
    const uint64_t whole = head + (size - head) / rec * rec;
    if (whole == size) return;
    std::filesystem::resize_file(path, whole, ec);
    logf("vk: %s ended in part of a record: %s", path.c_str(), ec ? "it could not be cut back" : "cut back to its last whole one");
}
/// (note_mu held) the notes file opened for this run's new pipelines, the keys it holds noted. known: its notes as pipe_notes_read
/// has just read them (precompile_run, before play; a note it left out as not sane is written again when its pipeline is met,
/// and the next read takes that one). Without them the records are read here in order. The first new pipeline in play opened
/// the file and read its keys back with a seek and an 8-byte read each: 142 ms for 1150 notes on the thread replaying
void notes_open_locked(const std::vector<std::unique_ptr<PipeNote>>* known) {
    static const uint64_t stamp = program_cache_stamp();
    const std::string path = pipe_notes_path();
    uint64_t head[2] = {0, 0};
    if (FILE* f = fopen(path.c_str(), "rb")) {
        if (fread(head, 8, 2, f) != 2) head[0] = 0;
        fclose(f);
    }
    const bool same = head[0] == stamp && head[1] == sizeof(PipeNote);
    if (same) cut_part_record(path, sizeof head, sizeof(PipeNote));
    if (same && known) {
        for (const auto& n : *known) noted.insert(n->key);
    } else if (same) {
        if (FILE* f = fopen(path.c_str(), "rb")) {
            auto rec = std::make_unique<PipeNote>();
            if (_fseeki64(f, 16, SEEK_SET) == 0)
                while (fread(rec.get(), sizeof(PipeNote), 1, f) == 1) noted.insert(rec->key);
            fclose(f);
        }
    }
    note_file = fopen(path.c_str(), same ? "ab" : "wb");
    if (note_file && !same) {
        const uint64_t h[2] = {stamp, sizeof(PipeNote)};
        fwrite(h, 8, 2, note_file);
    }
}
/// Whether key was new to the notes (play meets it for the first time on this install)
bool pipeline_note(uint64_t key, const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo,
                   const uint8_t* color_types = nullptr) {
    static const bool off = env_on("RX_VK_NO_PROGRAM_CACHE");
    if (off) return false;
    if (color_types) state_record(p, s, at, topo, color_types);
    auto rec = std::make_unique<PipeNote>();
    rec->key = key, rec->prog = p.key, rec->variant = p.variant, rec->topo = (uint32_t)topo, rec->n = (uint32_t)at.n;
    rec->samples = (uint32_t)at.samples, rec->depth_format = (uint32_t)at.depth_format, rec->has_depth = at.depth != nullptr;
    for (int i = 0; i < 8; i++) rec->color_format[i] = (uint32_t)at.color_format[i];
    rec->s = s;
    std::lock_guard lk(note_mu);
    if (!note_file) notes_open_locked(nullptr);
    if (!note_file || !noted.insert(key).second) return false;
    fwrite(rec.get(), sizeof(PipeNote), 1, note_file), fflush(note_file);
    return true;
}

// ---- (hookshot) The compute programs play dispatches, each key once, in hookshot_computes.bin beside the executable. The precompile
// made every kept compute program ready and kept a pipeline for each: 10,328 of them, a few GB of RAM that grew a 16 GB PC out of memory
// ("memory busted", out-of-memory crashes, Discord and issues 1228/1245, 2026-11-09), where play dispatches a few hundred. With the list
// only those are kept ready; the rest stay on disk and are made when first dispatched, from the driver's and our pipeline cache (the
// prepare warmed both: milliseconds, at the dispatch). Keys only, no stamp: a key no longer kept is just made in play.
// This file is what this install's own play dispatched. The shipped pipeline state list names compute programs too
// (StateRec2::kCompute: from the pipeline tour and the contributed lists), and the precompile keeps ready those and these, each
// once (computes_plan): a program this install's play has not reached yet is ready the first time it is dispatched.
// RX_VK_PRECOMPILE_COMPUTES=all: every kept compute program, as before; =none: none; =learned: this file's alone, the shipped
// list's left out; =listed: the shipped list's and this file's on a first start too, where every kept one is the default ----
constexpr char kComputesMagic[8] = {'H', 'K', 'C', 'O', 'M', 'P', '1', 0};
std::mutex compute_note_mu;
FILE* compute_note_file = nullptr;
std::unordered_set<uint64_t> computes_noted;
std::string computes_path() { return cache_path("hookshot_computes.bin"); }
std::unordered_set<uint64_t> computes_read() {
    std::unordered_set<uint64_t> keys;
    if (FILE* f = fopen(computes_path().c_str(), "rb")) {
        char magic[8] = {};
        if (fread(magic, 1, 8, f) == 8 && !memcmp(magic, kComputesMagic, 8))
            for (uint64_t k = 0; fread(&k, 8, 1, f) == 1;) keys.insert(k);
        fclose(f);
    }
    return keys;
}
void compute_note(uint64_t key) {
    static const bool off = env_on("RX_VK_NO_PROGRAM_CACHE");
    if (off) return;
    std::lock_guard lk(compute_note_mu);
    if (!compute_note_file) {
        computes_noted = computes_read();
        const bool append = !computes_noted.empty();   // (none read: a new file, or one that is not a list)
        compute_note_file = fopen(computes_path().c_str(), append ? "ab" : "wb");
        if (compute_note_file && !append) fwrite(kComputesMagic, 1, 8, compute_note_file);
    }
    if (compute_note_file && computes_noted.insert(key).second) fwrite(&key, 8, 1, compute_note_file), fflush(compute_note_file);
}

// ---- (hookshot) The variants of compute programs play dispatches, each once, in hookshot_compute_variants.bin beside the
// executable: a program's key and the variant's hash (Program::variant, the hash of its texture answers). A compute program
// has up to 64 kept variants and which one a dispatch runs is only known from what is bound then; with RX_VK_LOAD=create the
// variants this install's play ran before get their code and pipeline when the game makes the program (vk_programs.cpp), the
// others only what a pick needs. Hashes only, this install's own, never sent: a pair that names no kept variant does nothing ----
constexpr char kComputeVariantsMagic[8] = {'H', 'K', 'C', 'V', 'A', 'R', '1', 0};
std::mutex compute_variant_mu;
FILE* compute_variant_file = nullptr;
std::unordered_set<uint64_t> compute_variants_noted;   // (mix(key, variant))
std::string compute_variants_path() { return computes_path().substr(0, computes_path().size() - strlen("hookshot_computes.bin")) + "hookshot_compute_variants.bin"; }
std::vector<std::pair<uint64_t, uint64_t>> compute_variants_read() {
    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    if (FILE* f = fopen(compute_variants_path().c_str(), "rb")) {
        char magic[8] = {};
        if (fread(magic, 1, 8, f) == 8 && !memcmp(magic, kComputeVariantsMagic, 8))
            for (uint64_t r[2]; fread(r, 8, 2, f) == 2;) pairs.push_back({r[0], r[1]});
        fclose(f);
    }
    return pairs;
}
void compute_variant_note(uint64_t key, uint64_t variant) {
    static const bool off = env_on("RX_VK_NO_PROGRAM_CACHE");
    if (off) return;
    std::lock_guard lk(compute_variant_mu);
    if (!compute_variant_file) {
        const auto had = compute_variants_read();
        for (const auto& [k, v] : had) compute_variants_noted.insert(mix(k, v));
        compute_variant_file = fopen(compute_variants_path().c_str(), had.empty() ? "wb" : "ab");
        if (compute_variant_file && had.empty()) fwrite(kComputeVariantsMagic, 1, 8, compute_variant_file);
    }
    if (compute_variant_file && compute_variants_noted.insert(mix(key, variant)).second) {
        const uint64_t r[2] = {key, variant};
        fwrite(r, 8, 2, compute_variant_file), fflush(compute_variant_file);
    }
}

// ---- (hookshot) The pipeline state list: each pipeline the game asks for, by what it is made of and nothing of the game's code
// (its stages' code keys, the vertex input and colour output types its program was translated for, the draw state baked into it), in
// hookshot_pipeline_states.bin beside the executable. Unlike hookshot_pipelines.bin it is good for any build of the back end: when
// the shaders are prepared for a new one (prepare_run), every pipeline in it is noted for the precompile, its program translated
// first when the seed's are not that one, so that a new release, or a new install with a list from someone else's play
// (hookshot_pipeline_states_shipped.bin, nx2/make_game.py: merged from assets/pipeline_states/), builds them before the game and not in
// play. Only hashes, formats and state: the stages themselves come from the player's own game (hookshot_shaders.bin).
// Version 2 (HKSTATE2) holds what the game asked for, not what this GPU made of it: the targets' NVN formats (version 1 had the
// Vulkan formats this device chose: decoded ASTC, widened formats), the draw's own topology and whether it was a mesh draw (not the
// mesh fallback's topology), and the program variant's answers with its stages numbered as every GPU numbers them (the mesh
// fallback adds a vertex stage: answers_portable). The prepare maps each record through the rules the draw itself follows
// (state_targets: attachment_format; pipeline_topology; color_types_of), so one list made on one PC prepares the pipelines another
// GPU's draws will ask for, the mesh fallback's included. A record with variant 0 and no answers or questions stands for the variant
// nothing bound makes (seed_plain_of: each texture as its instruction declares it), as lists converted from records that hold no
// texture answers have them. Version 1 lists (a play folder's own from an older build) are still read.
// A version 2 record of kind kCompute names a compute program play dispatched, by its compute stage's code key in code[0] and
// nothing else (every other field 0): no pipeline state, a compute pipeline has none. It is the key hookshot_computes.bin holds,
// a hash of the game's stage like the others, the same on every GPU and with any settings. The record's size and the magic are
// unchanged: lists without such records are what they were, and a build from before them drops one (its sample count, 0, is
// not one a pipeline can have: targets_ok), so it never takes one for a graphics pipeline ----
constexpr char kStateMagic[8] = {'H', 'K', 'S', 'T', 'A', 'T', 'E', '1'};
constexpr char kStateMagic2[8] = {'H', 'K', 'S', 'T', 'A', 'T', 'E', '2'};
/// A version 1 record as the file has it, and in memory any record resolved for this device (state_list_read): the Vulkan formats
/// and colour output types this GPU draws with, the answers and variant numbered as answers_portable numbers them
struct StateRec {
    uint64_t code[4];        // the stages' code keys: vertex (mesh), fragment, task, the header-only stage (0: none)
    uint64_t variant;        // the program's (its answers about the textures)
    uint32_t mesh, topo, n, samples, depth_format, has_depth;
    uint32_t color_format[8];
    uint8_t color_types[8];  // (Shader::AttributeType of each target: what the fragment outputs are translated for)
    uint64_t blend[8];       // (nvn_blend_st)
    uint32_t vattr[16], nvattr;   // (nvn_va_st: the vertex attributes' formats)
    uint32_t stride[16], divisor[16];
    uint32_t blend_en, chmask, poly_mode, depth_clamp, a2c, pad;
    // the program variant's texture answers: by descriptor (Program::answers: stage << 28 | descriptor << 12 | image << 8 | type), and
    // as the translation asked them (Program::questions); a variant with more is noted without them, and made in play
    uint32_t nanswers, nquestions;
    uint32_t answers[24];
    uint8_t questions[32];
};
static_assert(sizeof(StateRec) == 528 && std::is_trivially_copyable_v<StateRec>, "the state list's record is a file format");
/// A version 2 record: what the game asked for, the same on every GPU and with any settings
struct StateRec2 {
    uint64_t code[4];        // as version 1
    uint64_t variant;        // portable_variant: the hash of the answers below
    uint32_t mesh;           // a mesh draw (task / mesh stages), whether this GPU runs it natively or through the fallback
    uint32_t topo;           // the draw's topology (a mesh draw's: kMeshDrawTopology; pipeline_topology makes the pipeline's)
    uint32_t n, samples, has_depth;
    uint32_t depth_nvn;      // the depth target's NVN format
    uint32_t color_nvn[8];   // each colour target's NVN texture format, 0: none
    uint32_t view_nvn[8];    // and the format it is drawn in (view_nvn_format: its view's)
    uint64_t blend[8];
    uint32_t vattr[16], nvattr;
    uint32_t stride[16], divisor[16];
    uint32_t blend_en, chmask, poly_mode, depth_clamp, a2c;
    uint32_t nanswers, nquestions;
    uint32_t answers[24];    // (stages numbered as answers_portable numbers them)
    uint8_t questions[32];
    uint32_t kind;           // 0: a graphics or mesh pipeline (all the lists held before it); kCompute: a compute program
    uint32_t pad;
    static constexpr uint32_t kCompute = 1;   // (code[0]: the compute stage's code key, Program::key; the other fields 0)
};
static_assert(sizeof(StateRec2) == 552 && std::is_trivially_copyable_v<StateRec2>, "the state list's record is a file format");
std::string state_list_path(const char* name) {
    // (the shipped list is the build's, beside the game; the ones play adds to are this PC's, in the game folder's cache/)
    return strstr(name, "_shipped") ? game_folder_path() + name : cache_path(name);
}
template <class R>
uint64_t state_hash(const R& r) {
    uint64_t h = 0xCBF29CE484222325ull;
    const uint8_t* b = (const uint8_t*)&r;
    for (size_t i = 0; i < sizeof r; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
/// The targets a record's draw renders into on this device: each NVN format as the draw's own texture_get and texture_view make it
/// (attachment_format), the sample count and whether there is depth as the game had them
Targets state_targets(uint32_t n, uint32_t samples, const uint32_t color_nvn[8], const uint32_t view_nvn[8], uint32_t has_depth,
                      uint32_t depth_nvn) {
    Targets at{};
    at.n = (int)std::min(n, 8u), at.samples = (VkSampleCountFlagBits)samples;
    for (int k = 0; k < at.n; k++)
        at.color_format[k] = color_nvn[k] ? attachment_format((int)color_nvn[k], (int)view_nvn[k]) : VK_FORMAT_UNDEFINED;
    at.depth = has_depth ? (Texture*)1 : nullptr;   // (never read through: whether there is one)
    at.depth_format = has_depth ? attachment_format((int)depth_nvn, (int)depth_nvn) : VK_FORMAT_UNDEFINED;
    return at;
}
/// A version 1 record's answers, numbered as answers_portable numbers them: a mesh record made with the fallback has a fragment
/// stage one further (after the generated vertex stage) and the fallback's topology. False: it cannot be told apart (its answers left
/// out) and is not used
bool state_v1_portable(StateRec& r) {
    if (!r.mesh) return true;
    const uint32_t g = r.code[2] ? 2 : 1;   // (where the fallback's generated vertex stage is: after (task,) mesh)
    bool fallback = r.topo != (uint32_t)kMeshDrawTopology;
    for (uint32_t i = 0; i < r.nanswers && i < 24; i++) fallback |= (r.answers[i] >> 28) > g;
    if (!fallback) return true;
    if (!r.nanswers) return false;
    std::vector<Program::Answer> answers(r.nanswers);
    for (uint32_t i = 0; i < r.nanswers; i++) {
        uint32_t stage = r.answers[i] >> 28;
        if (stage > g) stage--, r.answers[i] = (r.answers[i] & 0x0FFFFFFFu) | stage << 28;
        answers[i] = {(uint16_t)stage, (uint16_t)(r.answers[i] >> 12 & 0xFFFF), (uint8_t)(r.answers[i] & 0xFF), (r.answers[i] >> 8 & 1) != 0};
    }
    r.variant = answers_hash_of(answers), r.topo = (uint32_t)kMeshDrawTopology;
    return true;
}
/// A version 2 record for this device (false: it cannot be, its counts out of range or a format this GPU does not draw to)
bool state_v2_resolve(const StateRec2& v, StateRec& r) {
    if (v.nanswers > std::size(v.answers) || v.nquestions > std::size(v.questions) || v.nvattr > 16 || !targets_ok(v.n, v.samples, v.topo))
        return false;
    r = StateRec{};
    std::memcpy(r.code, v.code, sizeof r.code);
    r.variant = v.variant, r.mesh = v.mesh != 0, r.topo = v.topo;
    const Targets at = state_targets(v.n, v.samples, v.color_nvn, v.view_nvn, v.has_depth, v.depth_nvn);
    r.n = (uint32_t)at.n, r.samples = (uint32_t)at.samples, r.has_depth = at.depth != nullptr, r.depth_format = (uint32_t)at.depth_format;
    for (int k = 0; k < 8; k++) r.color_format[k] = (uint32_t)at.color_format[k];
    color_types_of(at, r.color_types);
    std::memcpy(r.blend, v.blend, sizeof r.blend);
    std::memcpy(r.vattr, v.vattr, sizeof r.vattr), r.nvattr = v.nvattr;
    std::memcpy(r.stride, v.stride, sizeof r.stride), std::memcpy(r.divisor, v.divisor, sizeof r.divisor);
    r.blend_en = v.blend_en, r.chmask = v.chmask, r.poly_mode = v.poly_mode, r.depth_clamp = v.depth_clamp, r.a2c = v.a2c;
    r.nanswers = v.nanswers, r.nquestions = v.nquestions;
    std::memcpy(r.answers, v.answers, sizeof r.answers), std::memcpy(r.questions, v.questions, sizeof r.questions);
    return formats_ok(r.n, r.color_format, r.has_depth, r.depth_format);
}
/// The records of a state list of either version, each resolved for this device (none when it is not one). *version: the file's
/// (0: not a list). Only its pipelines: the compute programs it names (StateRec2::kCompute) have nothing to resolve and are read by
/// state_list_computes; a kind this build does not know is left out
std::vector<StateRec> state_list_read(const std::string& path, int* version = nullptr) {
    std::vector<StateRec> out;
    if (version) *version = 0;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return out;
    char magic[8];
    uint32_t size = 0;
    if (fread(magic, 8, 1, f) == 1 && fread(&size, 4, 1, f) == 1) {
        if (!memcmp(magic, kStateMagic, 8) && size == sizeof(StateRec)) {
            if (version) *version = 1;
            StateRec r;
            while (fread(&r, sizeof r, 1, f) == 1) {
                if (r.nanswers > std::size(r.answers) || r.nquestions > std::size(r.questions) || r.nvattr > 16 ||
                    !targets_ok(r.n, r.samples, r.topo) || !formats_ok(r.n, r.color_format, r.has_depth, r.depth_format) ||
                    !state_v1_portable(r))
                    continue;   // (hookshot: a pipeline is made from the rest: its counts must be ones it can have)
                out.push_back(r);
            }
        } else if (!memcmp(magic, kStateMagic2, 8) && size == sizeof(StateRec2)) {
            if (version) *version = 2;
            StateRec2 v;
            StateRec r;
            while (fread(&v, sizeof v, 1, f) == 1)
                if (!v.kind && state_v2_resolve(v, r)) out.push_back(r);
        }
    }
    fclose(f);
    return out;
}
/// The compute programs a state list names, as the file has them (each maybe more than once), and nothing else of it: no record
/// resolved, no device asked (precompile_run reads the shipped list at every start for these alone, some 20 MB of it pipelines)
std::vector<uint64_t> state_list_computes(const std::string& path) {
    std::vector<uint64_t> keys;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return keys;
    char magic[8];
    uint32_t size = 0;
    if (fread(magic, 8, 1, f) == 1 && fread(&size, 4, 1, f) == 1 && !memcmp(magic, kStateMagic2, 8) && size == sizeof(StateRec2)) {
        std::vector<StateRec2> some(256);   // (whole records only: one cut short at the end is not read)
        for (size_t n; (n = fread(some.data(), sizeof(StateRec2), some.size(), f)) > 0;)
            for (size_t i = 0; i < n; i++)
                if (some[i].kind == StateRec2::kCompute && some[i].code[0]) keys.push_back(some[i].code[0]);
    }
    fclose(f);
    return keys;
}
/// The compute programs the precompile keeps ready before play (precompile_run). which: RX_VK_PRECOMPILE_COMPUTES, null when not set.
/// keys, each once: those this install's play dispatched (hookshot_computes.bin: learned) and those the shipped state list names
/// (every one of them: a list is bounded where it is made, nx2/pipeline_states.py check and read_sent, never cut short here, where
/// a program left out would only show as a stall in play; not when state_lists is false or which is "learned").
/// all: every kept compute program instead:
///  - a first start, with nothing learned yet (and whenever nothing names any, or which is "all"). The shipped list names what
///    someone's play dispatched, not what this player's will: a program it misses would be loaded and its pipeline made at its
///    first dispatch, which a first start has never done. It costs that start the memory of every program (the comment at
///    hookshot_computes.bin); from the second start on only the listed ones are kept, as before;
///  - which "listed" takes the shipped list's alone on a first start too: far less memory there, and a program the list
///    misses is made at its first dispatch.
/// none: which is "none"
struct ComputePlan {
    std::unordered_set<uint64_t> keys;
    bool all = false, none = false, learned_only = false;
    size_t shipped = 0, learned = 0;   // (the keys the shipped list names, and those play dispatched here)
    size_t fresh = 0;                  // (the shipped list's that this install's play has not dispatched)

    // The play process's .built marker can satisfy a later Builder request if it
    // warmed every compute, or the same set. The order of the list is immaterial.
    static constexpr uint64_t every = 1;
    uint64_t coverage() const {
        if (all) return every;
        uint64_t stamp = 0;
        if (!none) {
            std::vector<uint64_t> sorted(keys.begin(), keys.end());
            std::sort(sorted.begin(), sorted.end());
            for (const auto key : sorted) stamp = mix(stamp, key);
        }
        return stamp << 1;
    }
};
// Most translates the whole seed and warms compute pipelines for programs seen in play. Without a usable list, computes_plan("listed") falls back to every kept program.
bool compile_level_all();
const char* precompile_computes_which() {
    if (const char* e = getenv("RX_VK_PRECOMPILE_COMPUTES")) return e;
    if (dev.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA || !compile_level_all()) return "listed";
    return nullptr;
}
ComputePlan computes_plan(const char* which, bool state_lists) {
    ComputePlan plan;
    plan.learned_only = which && !strcmp(which, "learned");
    plan.keys = computes_read();
    plan.learned = plan.keys.size();
    if (state_lists && !plan.learned_only) {
        std::unordered_set<uint64_t> seen;
        for (const uint64_t k : state_list_computes(state_list_path("hookshot_pipeline_states_shipped.bin")))
            if (seen.insert(k).second) plan.shipped++, plan.fresh += plan.keys.insert(k).second;
    }
    const bool listed = which && !strcmp(which, "listed");
    plan.all = which && !plan.learned_only && !listed ? !strcmp(which, "all") : listed ? plan.keys.empty() : !plan.learned;
    plan.none = !plan.all && which && !strcmp(which, "none");
    return plan;
}
// Warming for the driver and keeping pipelines resident during play are separate.
// On All the driver's warm includes every compute, even after this install has played.
const ComputePlan& compute_warm_plan() {
    static const ComputePlan plan = [] {
        const char* which = precompile_computes_which();
        return computes_plan(which ? which : "all", !env_on("RX_VK_NO_STATE_LIST"));
    }();
    return plan;
}
std::mutex state_mu;
FILE* state_file = nullptr;
std::unordered_set<uint64_t> state_seen;
/// (state_mu held) the state list opened for this run's new pipelines, the records it holds hashed. precompile_run opens it before
/// play: the first new pipeline in play read and hashed the whole list on the thread replaying, 8-18 ms. A version 1 list (an older
/// build's) is kept as hookshot_pipeline_states_v1.bin, which the prepare reads too, and this build's records start a version 2 one.
/// RX_VK_NO_STATE_LIST=1: never called
void state_open_locked() {
    const std::string path = state_list_path("hookshot_pipeline_states.bin");
    bool keep = false;
    if (FILE* f = fopen(path.c_str(), "rb")) {
        char magic[8];
        uint32_t size = 0;
        const bool v2 = fread(magic, 8, 1, f) == 1 && fread(&size, 4, 1, f) == 1 && !memcmp(magic, kStateMagic2, 8) &&
                        size == sizeof(StateRec2);
        const bool v1 = !v2 && !memcmp(magic, kStateMagic, 8) && size == sizeof(StateRec);
        if (v2) {
            StateRec2 r;
            while (fread(&r, sizeof r, 1, f) == 1) state_seen.insert(state_hash(r)), keep = true;
        }
        fclose(f);
        if (v1) {
            const std::string old = state_list_path("hookshot_pipeline_states_v1.bin");
            remove(old.c_str());
            if (rename(path.c_str(), old.c_str()) == 0) logf("vk: state list: the version 1 list kept as %s", old.c_str());
        }
    }
    if (keep) cut_part_record(path, 12, sizeof(StateRec2));   // (the magic and the record size)
    state_file = fopen(path.c_str(), keep ? "ab" : "wb");
    if (state_file && !keep) {
        const uint32_t size = sizeof(StateRec2);
        fwrite(kStateMagic2, 8, 1, state_file), fwrite(&size, 4, 1, state_file);
    }
    // (the compute programs this install's play dispatched before, hookshot_computes.bin: in the list too, each once, so that
    // the list a player sends (Save my pipeline list) names them all, not only those of the runs since this build)
    if (state_file) {
        bool added = false;
        for (const uint64_t key : computes_read()) {
            StateRec2 r{};
            r.code[0] = key, r.kind = StateRec2::kCompute;
            if (key && state_seen.insert(state_hash(r)).second) fwrite(&r, sizeof r, 1, state_file), added = true;
        }
        if (added) fflush(state_file);
    }
}
/// (the draw's state; topo: the pipeline's, pipeline_topology's) the pipeline as the game asked for it, in the list once
void state_record(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, const uint8_t color_types[8]) {
    static const bool off = env_on("RX_VK_NO_STATE_LIST");
    if (off) return;
    (void)color_types;   // (made again from the formats: color_types_of)
    StateRec2 r{};
    std::memcpy(r.code, p.codes, sizeof r.code);   // (kept with the program: the stages' lock is not taken on the draw's path)
    if (!r.code[0] || p.compute) return;
    r.variant = portable_variant(p);
    r.mesh = p.mesh || p.fallback ? 1 : 0;
    r.topo = r.mesh ? (uint32_t)kMeshDrawTopology : (uint32_t)topo, r.n = (uint32_t)at.n, r.samples = (uint32_t)at.samples;
    for (int i = 0; i < at.n && i < 8; i++)
        if (at.color[i]) r.color_nvn[i] = (uint32_t)at.color[i]->nvn_format, r.view_nvn[i] = (uint32_t)at.color_view_nvn[i];
    r.has_depth = at.depth != nullptr;
    if (at.depth) r.depth_nvn = (uint32_t)at.depth->nvn_format;
    for (int i = 0; i < 16; i++) r.vattr[i] = s.vattr[i], r.stride[i] = s.vstream[i].stride, r.divisor[i] = s.vstream[i].divisor;
    r.nvattr = (uint32_t)s.nvattr;
    for (int i = 0; i < 8; i++) std::memcpy(&r.blend[i], &s.blend[i], 8);
    r.blend_en = (uint32_t)s.color.blend_en, r.chmask = s.chmask.mask, r.poly_mode = (uint32_t)s.poly.mode;
    r.depth_clamp = s.depth_clamp ? 1 : 0, r.a2c = s.ms.a2c ? 1 : 0;
    if (p.answers.size() <= 24 && p.questions.size() <= 32) {
        const std::vector<Program::Answer> answers = answers_portable(p);
        r.nanswers = (uint32_t)answers.size(), r.nquestions = (uint32_t)p.questions.size();
        for (size_t i = 0; i < answers.size(); i++) {
            const Program::Answer& a = answers[i];
            r.answers[i] = (uint32_t)(a.stage & 0xF) << 28 | (uint32_t)(a.desc & 0xFFFF) << 12 | (a.image ? 1u : 0u) << 8 | a.type;
        }
        std::memcpy(r.questions, p.questions.data(), p.questions.size());
    }
    const uint64_t h = state_hash(r);
    std::lock_guard lk(state_mu);
    if (!state_file) state_open_locked();
    if (state_file && state_seen.insert(h).second) fwrite(&r, sizeof r, 1, state_file), fflush(state_file);
}
/// (RX_VK_SHIPPED_STATES) The shipped list's contents by a hash, and the hash state_list_prepare last went through, in
/// hookshot_pipeline_states_shipped.done: a list that came after the shaders were prepared (a list shipped later, or a folder
/// prepared before it had one) was never noted, the prepare being done for that translator and seed, and its pipelines were all
/// made in play. 0: no list
uint64_t shipped_list_hash() {
    uint64_t h = 0xCBF29CE484222325ull;
    FILE* f = fopen(state_list_path("hookshot_pipeline_states_shipped.bin").c_str(), "rb");
    if (!f) return 0;
    uint8_t buf[65536];
    size_t total = 0;
    for (size_t got; (got = fread(buf, 1, sizeof buf, f)) > 0; total += got)
        for (size_t i = 0; i < got; i++) h = (h ^ buf[i]) * 1099511628211ull;
    fclose(f);
    return total > 12 ? h : 0;
}
uint64_t shipped_list_done() {
    uint64_t h = 0;
    if (FILE* f = fopen(state_list_path("hookshot_pipeline_states_shipped.done").c_str(), "rb")) {
        if (fread(&h, 8, 1, f) != 1) h = 0;
        fclose(f);
    }
    return h;
}
bool shipped_list_pending() {
    static const bool on = !getenv("RX_VK_SHIPPED_STATES") || atoi(getenv("RX_VK_SHIPPED_STATES")) != 0;
    if (!on || env_on("RX_VK_NO_STATE_LIST")) return false;
    const uint64_t h = shipped_list_hash();
    const bool pending = h && h != shipped_list_done();
    if (pending) logf("vk: the shipped pipeline state list is new since the shaders were prepared: its pipelines are noted now");
    return pending;
}
/// (a compute program's first dispatch of this run: dispatch) the program in the list once, by its key alone, as a kCompute record
void state_record_compute(uint64_t key) {
    static const bool off = env_on("RX_VK_NO_STATE_LIST");
    if (off || !key) return;
    StateRec2 r{};
    r.code[0] = key, r.kind = StateRec2::kCompute;
    const uint64_t h = state_hash(r);
    std::lock_guard lk(state_mu);
    if (!state_file) state_open_locked();
    if (state_file && state_seen.insert(h).second) fwrite(&r, sizeof r, 1, state_file), fflush(state_file);
}
/// (hookshot) The player's own state list as it is now, whole records only (state_mu held, so no record is half written): what
/// the menu's Save my pipeline list zips (vk_menu.cpp). *present: whether there is one
std::string state_list_bytes(bool* present) {
    std::lock_guard lk(state_mu);
    if (state_file) fflush(state_file);
    std::string bytes;
    *present = false;
    if (FILE* f = fopen(state_list_path("hookshot_pipeline_states.bin").c_str(), "rb")) {
        *present = true;
        char b[1 << 16];
        for (size_t n; (n = fread(b, 1, sizeof b, f)) > 0;) bytes.append(b, n);
        fclose(f);
    }
    return bytes;
}
/// Every pipeline of the state lists (the player's own and the shipped one): its program translated
/// when the program cache lacks it, then noted in hookshot_pipelines.bin for precompile_run to build. RX_VK_NO_STATE_LIST=1: off
void state_list_prepare(void (*progress)(size_t done, size_t total), bool open_seed = false) {
    if (env_on("RX_VK_NO_STATE_LIST")) return;
    remove((pipe_notes_path().substr(0, pipe_notes_path().find_last_of("\\/") + 1) + "hookshot_load_start.txt").c_str());
    remove((pipe_notes_path().substr(0, pipe_notes_path().find_last_of("\\/") + 1) + "hookshot_load_start.txt.waited").c_str());   // (a preparation: loads at create are tried again, load_default.h load_fallback_path)
    const auto t0 = std::chrono::steady_clock::now();
    int shipped_version = 0;
    // (the shipped list's compute programs are not pipelines to note: precompile_run keeps them ready, computes_plan)
    std::vector<StateRec> recs = state_list_read(state_list_path("hookshot_pipeline_states_shipped.bin"), &shipped_version);
    const size_t shipped = recs.size();
    if (const uint64_t h = shipped ? shipped_list_hash() : 0) {   // (gone through: state_list_pending is false until it changes)
        if (FILE* f = fopen(state_list_path("hookshot_pipeline_states_shipped.done").c_str(), "wb")) fwrite(&h, 8, 1, f), fclose(f);
    }
    // (the player's own: this build's, and an older build's version 1 list, state_open_locked)
    for (const char* own : {"hookshot_pipeline_states.bin", "hookshot_pipeline_states_v1.bin"})
        for (const StateRec& r : state_list_read(state_list_path(own))) recs.push_back(r);
    if (recs.empty() || (open_seed && !seed_open())) return;
    // the programs: each once (by its stages and the types it was translated for)
    struct Want { const StateRec* r; uint64_t key = 0; };
    std::vector<Want> want;
    std::unordered_map<uint64_t, size_t> by_program;
    std::vector<size_t> rec_program(recs.size());
    for (size_t i = 0; i < recs.size(); i++) {
        const StateRec& r = recs[i];
        uint64_t h = 0xCBF29CE484222325ull;
        const auto add = [&h](const void* d, size_t n) {
            for (size_t k = 0; k < n; k++) h = (h ^ ((const uint8_t*)d)[k]) * 1099511628211ull;
        };
        add(r.code, sizeof r.code), add(&r.mesh, 4), add(r.color_types, 8);
        for (int k = 0; k < 16; k++) {   // (the input types, not the formats: what the program key holds)
            const uint8_t t = (uint8_t)attribute_type_of(r.vattr[k]);
            add(&t, 1);
        }
        const auto [it, fresh] = by_program.emplace(h, want.size());
        if (fresh) want.push_back({&r});
        rec_program[i] = it->second;
    }
    // The texture variants that need translation are only known after loading the base programs.
    size_t done = 0, total = want.size() + recs.size();
    std::mutex progress_mu;
    const auto report_done = [&] {
        if (progress) {
            std::lock_guard<std::mutex> lock(progress_mu);
            progress(++done, total);
        }
    };
    if (progress) progress(0, total);
    std::atomic<size_t> next{0}, made{0}, missing{0};
    const auto worker = [&] {
        ub_thread_off();   // (a background thread: see ub_thread_off; the pipeline state list's builds raced the replay's updates)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        for (;;) {
            const size_t i = next++;
            if (i >= want.size()) return;
            const StateRec& r = *want[i].r;
            std::unique_ptr<Program> p;
            // (a mesh program: native or the mesh fallback's, as this GPU draws it (seed_translate); one with a header-only stage is
            // made in play)
            if (!r.code[3]) p = seed_program_of(r.code, r.mesh != 0, r.vattr, r.color_types, &want[i].key);
            if (!want[i].key) missing++;
            if (p) {
                if (!p->failed) made++;
                for (Stage& st : p->stages)
                    if (st.module) vkDestroyShaderModule(dev.device, st.module, nullptr);
            }
            report_done();
        }
    };
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < std::clamp(half_cpus(), 1u, 32u); i++) threads.emplace_back(worker);
    for (std::thread& t : threads) t.join();
    // then each pipeline, noted as a draw would note it
    std::unordered_set<uint64_t> keys;
    for (const Want& w : want)
        if (w.key) keys.insert(w.key);
    // (these keys only: the kept compute programs stay on disk for precompile_run, which builds their pipelines)
    program_cache_load_some(keys, [](Program& p) { ensure_layout(p); }, false);
    // the variants made by other textures (the seed's are for nothing bound): translated with the answers the game's gave
    // (a record with variant 0 and no answers or questions stands for the variant nothing bound makes: a list converted from records
    // that hold no texture answers has those; a program without textures has variant 0 already, and is found as it is)
    const auto plain = [](const StateRec& r) { return !r.variant && !r.nanswers && !r.nquestions; };
    std::vector<size_t> variant_of;   // (a record of each variant the cache lacks)
    size_t plain_wanted = 0;
    {
        std::unordered_set<uint64_t> seen;
        for (size_t i = 0; i < recs.size(); i++) {
            const StateRec& r = recs[i];
            const uint64_t key = want[rec_program[i]].key;
            // (a plain record: the variant nothing bound makes has the hash of its answers, not 0, so it is looked up by them:
            // loaded from the program cache already, it is not translated, and saved, again)
            if (key && (r.nanswers || r.nquestions || plain(r)) && !program_find_portable(key, r.variant) &&
                !(plain(r) && program_find_plain(key)) && seen.insert(mix(key, r.variant)).second)
                variant_of.push_back(i), plain_wanted += plain(r);
        }
    }
    total = want.size() + variant_of.size();
    if (progress) progress(done, total);
    std::atomic<size_t> variants_made{0};
    std::vector<const Program*> made_plain(variant_of.size());   // (each worker writes its own records' only)
    next = 0;
    const auto variant_worker = [&] {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        for (;;) {
            const size_t i = next++;
            if (i >= variant_of.size()) return;
            const StateRec& r = recs[variant_of[i]];
            const Program* p = nullptr;
            if (plain(r)) {
                p = made_plain[i] = seed_plain_of(r.code, r.mesh != 0, r.vattr, r.color_types);
            } else {
                std::vector<uint8_t> questions(r.questions, r.questions + r.nquestions);
                std::vector<Program::Answer> answers(r.nanswers);
                for (uint32_t k = 0; k < r.nanswers; k++)
                    answers[k] = {(uint16_t)(r.answers[k] >> 28), (uint16_t)(r.answers[k] >> 12 & 0xFFFF), (uint8_t)(r.answers[k] & 0xFF),
                                  (r.answers[k] >> 8 & 1) != 0};
                p = seed_variant_of(r.code, r.mesh != 0, r.vattr, r.color_types, questions, answers);
                if (p && portable_variant(*p) == r.variant) variants_made++;
            }
            if (p) ensure_layout(const_cast<Program&>(*p));
            report_done();
        }
    };
    threads.clear();
    for (unsigned i = 0; i < std::clamp(half_cpus(), 1u, 32u); i++) threads.emplace_back(variant_worker);
    for (std::thread& t : threads) t.join();
    std::unordered_map<uint64_t, const Program*> plain_of;   // (by program key: the variant nothing bound makes)
    for (size_t i = 0; i < variant_of.size(); i++)
        if (made_plain[i]) plain_of.emplace(want[rec_program[variant_of[i]]].key, made_plain[i]);
    size_t noted_n = 0, no_variant = 0, noted_mesh = 0, noted_fallback = 0, noted_plain = 0, added = 0;
    auto s = std::make_unique<NvnState>();
    for (size_t i = 0; i < recs.size(); i++) {
        const StateRec& r = recs[i];
        const uint64_t key = want[rec_program[i]].key;
        if (!key) continue;
        const Program* p = program_find_portable(key, r.variant);
        if (!p && plain(r)) {   // (made now, or loaded from the program cache)
            const auto it = plain_of.find(key);
            p = it != plain_of.end() ? it->second : program_find_plain(key);
            noted_plain += p != nullptr;
        }
        if (!p) {
            no_variant++;
            continue;
        }
        *s = NvnState{};
        for (int k = 0; k < 16; k++) s->vattr[k] = r.vattr[k], s->vstream[k].stride = r.stride[k], s->vstream[k].divisor = r.divisor[k];
        s->nvattr = (int)r.nvattr;
        for (int k = 0; k < 8; k++) std::memcpy(&s->blend[k], &r.blend[k], 8);
        s->color.blend_en = r.blend_en, s->chmask.mask = r.chmask, s->poly.mode = r.poly_mode;
        s->depth_clamp = r.depth_clamp != 0, s->ms.a2c = (uint8_t)r.a2c;
        Targets at{};
        at.n = (int)r.n, at.samples = (VkSampleCountFlagBits)r.samples, at.depth_format = (VkFormat)r.depth_format;
        at.depth = r.has_depth ? (Texture*)1 : nullptr;   // (never read through: whether there is one)
        for (int k = 0; k < 8; k++) at.color_format[k] = (VkFormat)r.color_format[k];
        if (!p->layout) ensure_layout(const_cast<Program&>(*p));
        const VkPrimitiveTopology topo = pipeline_topology(*p, (VkPrimitiveTopology)r.topo);   // (the draw's: the fallback's own)
        added += pipeline_note(pipeline_key(*p, *s, at, topo), *p, *s, at, topo);
        noted_n++;
        noted_mesh += p->mesh, noted_fallback += p->fallback;
    }
    logf("vk: state list: %zu pipelines (%zu shipped, version %d), %zu programs (%zu translated now, %zu with stages not in the seed), "
         "%zu variants by their textures (%zu made), %zu by nothing bound (%zu made), %zu noted (%zu native mesh, %zu mesh fallback, "
         "%zu by nothing bound), %zu whose program variant was not made, in %.1f s", recs.size(), shipped, shipped_version, want.size(),
         made.load(), missing.load(), variant_of.size() - plain_wanted, variants_made.load(), plain_wanted, plain_of.size(), noted_n,
         noted_mesh, noted_fallback, noted_plain, no_variant, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    // (hookshot) Only new notes take the marker that every note's pipeline is in the pipeline cache (precompile_covers_notes):
    // they are added here without their pipelines, and precompile_run builds them. It was removed at every call, and the
    // preparation after every build calls this: each update, even one that changed no shader, left the next start to build
    // every noted pipeline again ("Loading pipelines", 44,000 here, minutes on a slower PC) after the preparation had
    // already built them in its own process. A list that notes nothing new leaves the marker, and the start loads at create
    if (added) {
        remove(precompile_built_path().c_str());
        logf("vk: state list: %zu pipelines not noted before: the next start builds them into the pipeline cache", added);
    }
    if (open_seed && !env_on("RX_VK_SEED_CHECK")) seed_close();
}

// ---- (hookshot) Stage libraries, as TotK Recomp's runtime/vk_pipeline.c: the part of a pipeline that costs is its shaders, and with
// VK_EXT_graphics_pipeline_library they are built apart from what a draw adds (vertex layout, target formats). prepare_run builds
// each program's two shader libraries once, for the driver's own cache to keep; a draw that needs a pipeline nothing has built gets
// them back from that cache, links them on the spot, and draws at once. The worker's full build then takes over. Needs the blend,
// polygon, clamp and alpha-to-coverage state dynamic, or those would be in the libraries too. RX_VK_GPL=1: on
// (a library the driver does not have compiled is not compiled in the draw: asked for with
// VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT, it comes back at once either way. Cold, the draw is left out or waits
// for the worker's full build, as without libraries, and a worker builds the library after it, for the program's next pipeline.
// Needs pipelineCreationCacheControl: without it, compiled in the draw) ----
bool gpl_wanted() {
    static const bool on = dev.graphics_pipeline_library && dev.eds3_blend && dev.eds3_polygon && dev.eds3_clamp && dev.eds3_a2c &&
                           env_on("RX_VK_GPL");   // (off until the driver step is safe: linked cold, the libraries stall the draw)
    return on;
}
bool gpl_ask_only() {
    static const bool on = dev.vk13.pipelineCreationCacheControl;
    return on;
}
std::mutex gpl_mu;
std::unordered_map<uint64_t, VkPipeline> gpl_libs;   // (null: cold and a worker's to build, or it failed. Not asked for again)
/// cold_key: set to the library's key when the driver would have had to compile it (the caller has a worker build it: gpl_build_cold)
VkPipeline gpl_part(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, int part, uint64_t key,
                    uint64_t* cold_key) {
    key = mix(key, (uint64_t)part);
    {
        std::lock_guard lk(gpl_mu);
        if (auto it = gpl_libs.find(key); it != gpl_libs.end()) return it->second;
    }
    const bool ask = (part == 1 || part == 2) && gpl_ask_only();   // (the two with shaders in them)
    bool cold = false;
    VkPipeline made = build_graphics(p, s, at, topo, part, ask ? &cold : nullptr);
    if (ask && made) st_lib_hits++;
    st_lib_cold += cold;
    std::lock_guard lk(gpl_mu);
    const auto [it, fresh] = gpl_libs.emplace(key, made);
    if (!fresh && made) vkDestroyPipeline(dev.device, made, nullptr);
    if (fresh && cold && cold_key) *cold_key = key;
    return it->second;
}
/// (a worker's: a library gpl_part found cold. One that fails stays null, so it is built once at most)
void gpl_build_cold(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, int part, uint64_t key) {
    const auto t0 = std::chrono::steady_clock::now();
    const VkPipeline made = build_graphics(p, s, at, topo, part);
    // (what the cold libraries cost the workers: their jobs are in the one queue with the pipelines and translations of the
    // draws left out, and nothing else says how long those waited behind them)
    st_lib_build_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    if (!made) return;
    st_lib_built++;
    std::lock_guard lk(gpl_mu);
    gpl_libs[key] = made;
}
/// cold: the keys of the pre-rasterization and fragment libraries this call found cold (0: not)
VkPipeline gpl_link(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, uint64_t cold[2]) {
#ifdef TRACY_ENABLE
    HT_SCOPE("gpl_link (pipeline)");
#endif
    if (at.n < 0 || at.n > 8) return VK_NULL_HANDLE;   // (hookshot: at most 8 colour targets)
    const uint64_t program = mix(p.key, p.variant);
    uint64_t input = (uint64_t)topology_class(topo) + 1;
    if (!p.mesh && !p.fallback) {
        const Shader::Info& vs = p.stages[0].info;
        for (int i = 0; i < 16; i++)
            input = mix(input, vs.loads.Generic(i) ? (uint64_t)s.vattr[i] << 8 | (i < s.nvattr) : 0);
        for (int i = 0; i < 16; i++) input = mix(input, (uint64_t)s.vstream[i].divisor);
    }
    uint64_t output = (uint64_t)at.samples << 8 | (uint64_t)at.n;
    for (int i = 0; i < at.n; i++) output = mix(output, (uint64_t)at.color_format[i]);
    output = mix(output, at.depth ? (uint64_t)at.depth_format : 0);
    VkPipeline libs[4];
    uint32_t n = 0;
    if (!p.mesh) libs[n++] = gpl_part(p, s, at, topo, 0, input, nullptr);
    libs[n++] = gpl_part(p, s, at, topo, 1, program, &cold[0]);
    libs[n++] = gpl_part(p, s, at, topo, 2, mix(program, (uint64_t)at.samples), &cold[1]);
    libs[n++] = gpl_part(p, s, at, topo, 3, output, nullptr);
    for (uint32_t i = 0; i < n; i++)
        if (!libs[i]) return VK_NULL_HANDLE;
    VkPipelineLibraryCreateInfoKHR li{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
    li.libraryCount = n;
    li.pLibraries = libs;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.pNext = &li;
    ci.layout = p.layout;
    VkPipeline pl = VK_NULL_HANDLE;
    IntelDriverGate gate;
    if (guarded_graphics(VK_NULL_HANDLE, &ci, &pl) != VK_SUCCESS) return VK_NULL_HANDLE;
    return pl;
}

/// The pipeline entry of a draw (a new one is queued for a worker; RX_VK_SYNC_PIPELINES=1 builds it here)
/// What a graphics pipeline was built from, so the next draw of the same material does not hash it again
struct PipeFast {
    bool on = false;
    PipeEntry* pe = nullptr;
    uint64_t prog = 0, variant = 0;
    int topo = 0, n = 0, nvattr = 0;
    uint32_t samples = 0, depth = 0, blend_en = 0, chmask = 0, poly = 0, clamp = 0, a2c = 0;
    uint32_t formats[8]{}, vattr[16]{}, div[16]{};
    uint64_t blend[8]{};
};
thread_local PipeFast pipe_fast[2];

/// (hookshot) the stage a graphics pipeline's vertex input comes from (a fallback program's first stage is a compute one)
const Shader::Info* vertex_info(const Program& p) {
    for (const Stage& st : p.stages)
        if (st.vk_stage == VK_SHADER_STAGE_VERTEX_BIT) return &st.info;
    return p.stages.empty() ? nullptr : &p.stages[0].info;
}

/// What a draw's pipeline key is made from (draw_key.h KeyInputs), read out of the program, the state, the targets and the
/// device. Each word is the one the key always mixed in, cast as it was there: the keys are in players' pipeline lists and caches
KeyInputs key_inputs(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo) {
    KeyInputs k;
    k.prog = p.key, k.variant = p.variant;
    k.topo_class = (uint64_t)topology_class(topo), k.samples = (uint64_t)at.samples;
    k.vertex_input = !p.mesh && !p.fallback;
    if (k.vertex_input) {
        const Shader::Info& vs = *vertex_info(p);
        for (int i = 0; i < 16; i++) {
            if (vs.loads.Generic(i)) k.loads |= 1u << i;
            k.vattr[i] = (uint64_t)s.vattr[i], k.divisor[i] = (uint64_t)s.vstream[i].divisor;
        }
        k.nvattr = s.nvattr;
    }
    k.n = at.n;
    for (int i = 0; i < at.n; i++) {
        k.color_format[i] = (uint64_t)at.color_format[i];
        std::memcpy(&k.blend[i], &s.blend[i], 8);
    }
    k.depth_format = (uint64_t)at.depth_format;
    k.blend_en = s.color.blend_en;
    k.eds3_blend = dev.eds3_blend, k.eds3_polygon = dev.eds3_polygon, k.eds3_clamp = dev.eds3_clamp, k.eds3_a2c = dev.eds3_a2c;
    if (!k.eds3_blend)
        for (int i = 0; i < at.n; i++) k.write_masks |= (uint64_t)target_write_mask(p, s, i) << (4 * i);
    k.poly_mode = (uint64_t)s.poly.mode, k.depth_clamp = (uint64_t)s.depth_clamp, k.a2c = (uint64_t)s.ms.a2c;
    return k;
}
/// The identity of a graphics pipeline: what is baked into it (the rest is dynamic state). The precompile makes it again from a note's
/// state and does not trust the key the note was written with
/// (the key from the program's own seed: pipeline_key's, or the ledger's, whose seed is the program's fingerprint)
static uint64_t pipeline_key_from(uint64_t seed, const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo) {
    if (at.n < 0 || at.n > 8) return 0;   // (hookshot: at most 8 colour targets)
    return pipeline_key_seeded(key_inputs(p, s, at, topo), seed);   // (draw_key.h's arithmetic)
}
uint64_t pipeline_key(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo) {
    return pipeline_key_from(mix(p.key, p.variant), p, s, at, topo);
}

// ---- (hookshot, RX_VK_SHADER_KEY) What the driver has compiled, hookshot_compiled.bin ----
// A change of the shader translation drops every kept program (the stamp, vk_shader.cpp program_cache_stamp), and every program is translated
// again; most of the change's programs come out the same. What the driver was handed for them last time (stage libraries, compute
// pipelines, predicted pipelines) is still in its own cache (the driver keys that on the SPIR-V and the state, not on our stamp) and
// in mkw_vk_pipelines.bin, so building it again only to find it there is time spent for nothing: the ledger lists, by a hash of
// what the driver sees (Program::fp, plus the state of a pipeline), the things already built here. An entry is the 64-bit id; the
// file's header says what the ids are good for (the device, the driver, the layout of the pipeline's state: kStateStamp and
// kPipeKeyVersion, the pipeline cache's own data being there); anything else starts the list again. An entry added is written when
// the prepare or the pipeline cache save commits it (ledger_commit): the cache's data is on disk by then too
namespace {
std::mutex ledger_mu;
std::unordered_set<uint64_t> ledger_have;
std::vector<uint64_t> ledger_pending;
FILE* ledger_out = nullptr;
bool ledger_ready = false;
std::atomic<int64_t> ledger_reused{0}, ledger_built{0};
constexpr char kLedgerMagic[8] = {'H', 'K', 'C', 'O', 'M', 'P', 'L', '1'};
std::string ledger_path() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string dir(exe, n);
    return dir.substr(0, dir.find_last_of("\\/") + 1) + "hookshot_compiled.bin";
}
uint64_t ledger_context() {
    uint64_t h = 0xCBF29CE484222325ull;
    const auto add = [&h](const void* d, size_t n) {
        for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)d)[i]) * 1099511628211ull;
    };
    const uint64_t fixed[] = {kStateStamp, kPipeKeyVersion, dev.props.vendorID, dev.props.deviceID, dev.props.driverVersion,
                              dev.props.apiVersion, (uint64_t)dev.mesh_shader | (uint64_t)dev.mesh_fallback << 1 |
                              (uint64_t)dev.eds3_blend << 2 | (uint64_t)dev.eds3_polygon << 3 | (uint64_t)dev.eds3_clamp << 4 |
                              (uint64_t)dev.eds3_a2c << 5 | (uint64_t)dev.push_descriptor << 6};
    add(fixed, sizeof fixed);
    add(dev.props.pipelineCacheUUID, VK_UUID_SIZE);
    return h;
}
/// (ledger_mu held) the file read, or started again when it is another device's, driver's or layout's
void ledger_open_locked() {
    if (ledger_ready) return;
    ledger_ready = true;
    const std::string path = ledger_path();
    const uint64_t context = ledger_context();
    int64_t good = 0;
    if (FILE* f = fopen(path.c_str(), "rb")) {
        char magic[8];
        uint64_t ctx = 0;
        if (fread(magic, 8, 1, f) == 1 && !memcmp(magic, kLedgerMagic, 8) && fread(&ctx, 8, 1, f) == 1 && ctx == context) {
            uint64_t id;
            good = 16;
            while (fread(&id, 8, 1, f) == 1) ledger_have.insert(id), good += 8;
        }
        fclose(f);
    }
    if (good) {
        if (FILE* f = fopen(path.c_str(), "r+b")) _chsize_s(_fileno(f), good), fclose(f);   // (a half entry cut off)
        ledger_out = fopen(path.c_str(), "ab");
    } else {
        ledger_out = fopen(path.c_str(), "wb");
        if (ledger_out) fwrite(kLedgerMagic, 8, 1, ledger_out), fwrite(&context, 8, 1, ledger_out), fflush(ledger_out);
    }
    logf("vk: shader key: %zu things the driver has built are listed (hookshot_compiled.bin)", ledger_have.size());
}
bool ledger_has(uint64_t id) {
    if (!id) return false;
    std::lock_guard lk(ledger_mu);
    ledger_open_locked();
    return ledger_have.count(id) != 0;
}
size_t ledger_size() {
    std::lock_guard lk(ledger_mu);
    ledger_open_locked();
    return ledger_have.size();
}
void ledger_add(uint64_t id) {
    if (!id) return;
    std::lock_guard lk(ledger_mu);
    ledger_open_locked();
    if (ledger_have.insert(id).second) ledger_pending.push_back(id), ledger_built++;
}
/// The pipeline cache has no data of ours (a new file, or another driver's): what the list says was built is not in it
void ledger_drop_all() {
    std::lock_guard lk(ledger_mu);
    ledger_open_locked();
    if (ledger_have.empty() && ledger_pending.empty()) return;
    ledger_have.clear(), ledger_pending.clear();
    if (ledger_out) fclose(ledger_out), ledger_out = nullptr;
    ledger_out = fopen(ledger_path().c_str(), "wb");
    const uint64_t context = ledger_context();
    if (ledger_out) fwrite(kLedgerMagic, 8, 1, ledger_out), fwrite(&context, 8, 1, ledger_out), fflush(ledger_out);
}
void ledger_commit() {
    std::lock_guard lk(ledger_mu);
    if (!ledger_out || ledger_pending.empty()) return;
    fwrite(ledger_pending.data(), 8, ledger_pending.size(), ledger_out);
    fflush(ledger_out);
    ledger_pending.clear();
}
uint64_t ledger_id_seen(uint64_t fp) { return fp ? mix(fp, 0x7365656Eull) : 0; }
thread_local std::vector<uint64_t>* fingerprint_probe = nullptr;   // (update_changed_share: the fingerprints met, not listed)
uint64_t ledger_id_libs(const Program& p) { return p.fp ? mix(p.fp, 0x6C6962ull) : 0; }
uint64_t ledger_id_compute(const Program& p) { return p.fp ? mix(p.fp, 0x6373ull) : 0; }
uint64_t ledger_id_pipeline(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo) {
    return p.fp ? mix(pipeline_key_from(mix(p.fp, 0x706970ull), p, s, at, topo), 1) : 0;
}
}  // namespace

bool predicted_key(uint64_t key);   // (the predicted pipelines' keys, below)
void predicted_use(int how);
PipeEntry* graphics_pipeline(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, const uint8_t color_types[8]) {
    // the key: what is baked into it (the rest is dynamic state)
    // (hookshot: and which variant: p.key is the same for all the variants of a program by its textures' types, and a draw of
    // one ran another's shaders under its own descriptors)
    uint32_t formats[8]{};
    uint32_t vattr[16]{}, divs[16]{};
    uint64_t blend[8]{};
    for (int i = 0; i < 8 && i < at.n; i++) formats[i] = (uint32_t)at.color_format[i];
    for (int i = 0; i < 16; i++) vattr[i] = s.vattr[i], divs[i] = s.vstream[i].divisor;
    for (int i = 0; i < 8; i++) std::memcpy(&blend[i], &s.blend[i], 8);
    const int topo_c = topology_class(topo);
    const uint32_t blend_en = (uint32_t)s.color.blend_en, chmask = s.chmask.mask, poly = (uint32_t)s.poly.mode;
    const uint32_t clamp = s.depth_clamp ? 1u : 0u, a2c = s.ms.a2c ? 1u : 0u;
    for (const PipeFast& f : pipe_fast)
        if (f.on && f.pe &&
            pipe_same((uint64_t)(uintptr_t)&p, p.variant, topo_c, (uint32_t)at.samples, at.n, (uint32_t)at.depth_format,
                      formats, s.nvattr, vattr, divs, blend_en, blend, chmask, poly, clamp, a2c, f.prog, f.variant, f.topo,
                      f.samples, f.n, f.depth, f.formats, f.nvattr, f.vattr, f.div, f.blend_en, f.blend, f.chmask, f.poly,
                      f.clamp, f.a2c)) {
            FINE_N(FC_PIPE_FAST, 1);
            return f.pe;
        }
    auto remember = [&](PipeEntry* e) {
        if (!e) return e;
        if (!(pipe_fast[0].on && pipe_fast[0].pe == e)) pipe_fast[1] = pipe_fast[0];
        PipeFast& f = pipe_fast[0];
        f.on = true, f.pe = e, f.prog = (uint64_t)(uintptr_t)&p, f.variant = p.variant, f.topo = topo_c;
        f.samples = (uint32_t)at.samples, f.n = at.n, f.depth = (uint32_t)at.depth_format, f.nvattr = s.nvattr;
        f.blend_en = blend_en, f.chmask = chmask, f.poly = poly, f.clamp = clamp, f.a2c = a2c;
        std::memcpy(f.formats, formats, sizeof formats);
        std::memcpy(f.vattr, vattr, sizeof vattr);
        std::memcpy(f.div, divs, sizeof divs);
        std::memcpy(f.blend, blend, sizeof blend);
        return e;
    };
    // (hookshot, 2026-11-15) A pipeline belongs to the program object it was built from. The key names a program by its code and
    // its variant's hash, and two program objects can have both the same: the one the precompile or a program's create call
    // built a noted pipeline from, and the one a later draw picks. A draw then ran one program's shaders under the other's
    // layout, descriptors and constants; with the scene's lit programs a light loop read garbage for its count and the GPU
    // never finished the draw: the device lost near water on NVIDIA cards, six runs of six in a reproduction at Lake Hylia,
    // none with this. A pipeline made for another object with the same key is not taken: this object gets its own, under a key
    // of its own (not noted: it holds an address). RX_VK_PIPE_BY_PROGRAM=0: shared by the key alone, as before
    static const bool by_program = [] { const char* e = getenv("RX_VK_PIPE_BY_PROGRAM"); return !(e && env_value_off(e)); }();
    uint64_t key = pipeline_key(p, s, at, topo);
    const uint64_t own = mix(key, (uint64_t)(uintptr_t)&p);
    const uint64_t mkey = by_program ? own : key;   // (a thread's map: by key and object)
    static thread_local FlatMap<uint64_t, PipeEntry*> mine;   // (entries live as long as the cache)
    if (auto m = mine.find(mkey); m != mine.end()) return FINE_X(FC_PIPE_MAP, 1), remember(m->second);
    PROF_EXACT();   // (new to this thread, or new: rare, and a pipeline made can take milliseconds)
    ParLock lk;
    auto it = pipelines.find(key);
    bool own_key = false;
    if (by_program && it != pipelines.end()) {
        const Program* made_for = it->second->prog.load(std::memory_order_acquire);
        if (made_for && made_for != &p) {
            static std::atomic<int> told{0};
            if (told++ < 12)
                logf("vk: pipeline %016llx was made for another program object of %016llx (variant %016llx): this one gets its own",
                     (unsigned long long)key, (unsigned long long)p.key, (unsigned long long)p.variant);
            key = own, own_key = true;
            it = pipelines.find(key);
        }
    }
    if (it != pipelines.end()) return remember(mine[mkey] = it->second.get());
    pcache_init();
    PipeEntry* e = pipelines.emplace(key, std::make_unique<PipeEntry>()).first->second.get();
    e->prog.store(&p, std::memory_order_release);
    if (!own_key && kept_ready_n.load(std::memory_order_acquire)) {   // (made at its program's create call: ready, as one the precompile built)
        VkPipeline kept = VK_NULL_HANDLE;
        bool deferred = false;
        {
            std::lock_guard kl(kept_ready_mu);
            // (only the object it was made from takes it: another one of the same key builds its own, see above)
            if (const auto k = kept_ready.find(key); k != kept_ready.end() && (!by_program || !k->second.second || k->second.second == &p)) {
                kept = k->second.first;
                deferred = !kept;
                kept_ready.erase(k);
                kept_ready_n.store(kept_ready.size(), std::memory_order_release);
            }
        }
        if (kept) {
            e->pipeline = kept;
            e->ready.store(true, std::memory_order_release);
            census.pipes_taken++;
            return remember(e);
        }
        // The loader yielded this noted state to first use. Try its cache before queuing a build.
        if (deferred && dev.vk13.pipelineCreationCacheControl) {
            bool cold = false;
            const quick_pipe::Raise asking(ask_start_first);
            if (const VkPipeline pl = build_graphics(p, s, at, topo, -1, &cold)) {
                e->pipeline = pl;
                e->ready.store(true, std::memory_order_release);
                return remember(e);
            }
            census.load_pipes_cold++;
        }
    }
    use_counters.pipes_new++, census.play_pipes_new++;
    // (new to the notes and one the preparation predicted: only the prediction put it in a cache. Counted once made, by how)
    const bool fresh = pipeline_note(key, p, s, at, topo, color_types);
    const bool predicted = fresh && predicted_key(key);
    static const bool sync = env_on("RX_VK_SYNC_PIPELINES");
    if (sync) {
        if (program_cache_complete_graphics(p)) e->pipeline = build_graphics(p, s, at, topo);
        e->ready = true;
        return remember(e);
    }
    // (linked here, from the libraries: the draw goes ahead, and the worker's full build replaces it when it is done. Not linked
    // when a library is cold: the entry stays not ready, and the draw is left out or waits for the full build)
    uint64_t cold[2]{};
    if (gpl_wanted() && !p.fallback) {
        if (!program_cache_complete_graphics(p)) {
            e->ready.store(true, std::memory_order_release);
            return remember(e);
        }
        const auto t0 = std::chrono::steady_clock::now();
        const VkPipeline linked = gpl_link(p, s, at, topo, cold);
        st_link_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        st_links++;
        if (linked) {
            e->pipeline = linked;
            e->ready.store(true, std::memory_order_release);
        }
    }
    struct Job {
        PipeEntry* e;
        const Program* p;
        NvnState s;
        Targets at;
        VkPrimitiveTopology topo;
        uint64_t cold[2];
        bool predicted;
    };
    auto job = std::make_shared<Job>();
    job->e = e, job->p = &p, job->s = s, job->at = at, job->topo = topo, job->cold[0] = cold[0], job->cold[1] = cold[1];
    job->predicted = predicted;
    jobs_push([job] {
        // (over a linked one or a quick one, which are left alone: draws recorded with them may still be on their way. Neither is
        // destroyed: nothing here knows when the recordings that read the handle are done on the GPU, so a replaced quick
        // pipeline stays for the session, one more per entry that got one, counted in quick_pipeline's log line)
        if (!program_cache_complete_graphics(*job->p)) {
            job->e->ready.store(true, std::memory_order_release);
            return;
        }
        use_counters.made_how = 0;
        VkPipeline full;
        {   // (RX_VK_TEST_SLOW_MS: held inside the driver-fault gate, build_graphics)
            const quick_pipe::Raise slow(slow_full_build);
            full = build_graphics(*job->p, job->s, job->at, job->topo);
        }
        if (full) {
            job->e->pipeline = full;
        } else if (job->e->pipeline.load(std::memory_order_acquire)) {
            // (hookshot, deliberate) a full build that failed, or that the driver faulted on, leaves a quick pipeline (no optimizer)
            // the entry's for the session: its draws are drawn with it, where without one they would be left out (SKIP_HERE)
            static std::atomic<int> told{0};
            if (told++ < 10)
                logf("vk: the full pipeline of %016llx was not made: the one made ahead of it is drawn with for the session",
                     (unsigned long long)job->p->key);
        }
        job->e->how.store(use_counters.made_how, std::memory_order_relaxed);
        if (job->predicted) predicted_use(use_counters.made_how);
        job->e->ready.store(true, std::memory_order_release);
        // (the cold libraries after it and behind what is queued, with no flag: the draw waits for the pipeline above, and
        // jobs_wait runs one job)
        for (int i = 0; i < 2; i++)
            if (job->cold[i])
                jobs_push([job, i] { gpl_build_cold(*job->p, job->s, job->at, job->topo, i + 1, job->cold[i]); });
    }, false, &e->ready, [job]() noexcept {
        // Keep a linked pipeline if one was already usable; otherwise null means the completed build failed.
        job->e->ready.store(true, std::memory_order_release);
    });   // (a draw that waits for it runs it at once if it is still queued: jobs_wait)
    return remember(e);
}

/// part: -1 the whole pipeline, or the graphics pipeline library it stands for: 0 vertex input, 1 pre-rasterization, 2 fragment
/// shader, 3 fragment output
/// cold: made only if the driver has it compiled already; set, and nothing made, when it would have had to compile
VkPipeline build_graphics(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, int part, bool* cold) {
#ifdef TRACY_ENABLE
    HT_SCOPE("build_graphics (pipeline)");
#endif
    if (at.n < 0 || at.n > 8 || program_modules_dropped(p)) return VK_NULL_HANDLE;   // (hookshot: at most 8 colour targets)
    const Shader::Info& vs = *vertex_info(p);
    const auto t0 = std::chrono::steady_clock::now();
    static const bool trace = env_on("RX_VK_PIPELINE_TRACE");
    if (trace) logf("vk: graphics pipeline begin: program %016llx, part %d", (unsigned long long)p.key, part);
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    for (const Stage& st : p.stages) {
        // (hookshot: the fallback's dispatches are not graphics stages. Only a fallback program's: with native mesh shaders
        // the task and mesh stages are the pipeline's own, and a mesh program built without them was a fragment stage
        // alone, which drew none of its geometry and then lost the device, issue 640)
        if (p.fallback && (st.vk_stage & (VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)))
            continue;
        VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        si.stage = st.vk_stage;
        si.module = st.module;
        si.pName = "main";
        pin_subgroup(si);
        stages.push_back(si);
    }
    // vertex input: the attributes the vertex stage reads; those without a stream read zeros (binding 15)
    std::vector<VkVertexInputBindingDescription> vb;
    std::vector<VkVertexInputAttributeDescription> va;
    bool stream_used[16]{};
    bool zero_used = false;
    if (!p.mesh && !p.fallback)
        for (uint32_t i = 0; i < 16; i++) {
            if (!vs.loads.Generic(i)) continue;
            const nvn_va_st a = *reinterpret_cast<const nvn_va_st*>(&s.vattr[i]);
            const VkFormat f = (int)i < s.nvattr ? vertex_format(a.fmt) : VK_FORMAT_UNDEFINED;
            if (f == VK_FORMAT_UNDEFINED || a.stream >= 15) {
                va.push_back({i, 15, VK_FORMAT_R32_SFLOAT, 0});   // (an unset attribute reads (0, 0, 0, 1))
                zero_used = true;
                continue;
            }
            va.push_back({i, a.stream, supported_vertex_format(f), a.offset & 0x3FFF});
            stream_used[a.stream] = true;
        }
    for (uint32_t b = 0; b < 15; b++)
        if (stream_used[b])
            vb.push_back({b, s.vstream[b].stride,
                          s.vstream[b].divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
    if (zero_used) vb.push_back({15, 0, VK_VERTEX_INPUT_RATE_VERTEX});
    // (a divisor over 1: one element per that many instances (VK_EXT_vertex_attribute_divisor); it read one per
    // instance before)
    std::vector<VkVertexInputBindingDivisorDescriptionEXT> vdivs;
    for (uint32_t b = 0; b < 15; b++)
        if (stream_used[b] && s.vstream[b].divisor > 1) {
            static std::atomic<int> logged{0};
            if (logged++ < 8)
                logf("vk: vertex stream %u divisor %u%s", b, s.vstream[b].divisor, dev.vertex_divisor ? "" : " (not supported: 1)");
            if (dev.vertex_divisor) vdivs.push_back({b, std::min(s.vstream[b].divisor, dev.max_vertex_divisor)});
        }
    {   // (each NVN vertex format once: what the game draws with)
        static std::mutex mu;
        static bool seen[256];
        std::lock_guard lk(mu);
        for (const auto& d : va)
            if (d.binding != 15) {
                const nvn_va_st a = *reinterpret_cast<const nvn_va_st*>(&s.vattr[d.location]);
                if (a.fmt < 256 && !seen[a.fmt]) seen[a.fmt] = true, logf("vk: vertex format NVN %#x -> vk %d", (unsigned)a.fmt, (int)d.format);
            }
    }
    VkPipelineVertexInputDivisorStateCreateInfoEXT vdi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT};
    vdi.vertexBindingDivisorCount = (uint32_t)vdivs.size();
    vdi.pVertexBindingDivisors = vdivs.data();
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = (uint32_t)vb.size();
    vi.pVertexBindingDescriptions = vb.data();
    vi.vertexAttributeDescriptionCount = (uint32_t)va.size();
    vi.pVertexAttributeDescriptions = va.data();
    if (!vdivs.empty()) vi.pNext = &vdi;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = topo;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    // (dynamic: left out, or each value of them would be another library)
    rs.depthClampEnable = !dev.eds3_clamp && s.depth_clamp;
    rs.polygonMode = dev.eds3_polygon || s.poly.mode > 1 ? VK_POLYGON_MODE_FILL : s.poly.mode == 0 ? VK_POLYGON_MODE_POINT : VK_POLYGON_MODE_LINE;
    rs.lineWidth = 1.0f;
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pvs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT};
    pvs.provokingVertexMode = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT;   // (NVN's default, as OpenGL's)
    if (dev.provoking_vertex) rs.pNext = &pvs;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = at.samples;
    ms.alphaToCoverageEnable = s.ms.a2c && !dev.eds3_a2c;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cb[8]{};
    for (int i = 0; i < at.n; i++) {
        const nvn_blend_st& b = s.blend[i];
        cb[i].blendEnable = (s.color.blend_en >> i & 1) && at.color_format[i] ? VK_TRUE : VK_FALSE;
        cb[i].srcColorBlendFactor = blend_factor(b.src_rgb);
        cb[i].dstColorBlendFactor = blend_factor(b.dst_rgb);
        cb[i].colorBlendOp = blend_op(b.eq_rgb);
        cb[i].srcAlphaBlendFactor = blend_factor(b.src_a);
        cb[i].dstAlphaBlendFactor = blend_factor(b.dst_a);
        cb[i].alphaBlendOp = blend_op(b.eq_a);
        cb[i].colorWriteMask = target_write_mask(p, s, i);
    }
    VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    bs.attachmentCount = (uint32_t)at.n;
    bs.pAttachments = cb;
    std::vector<VkDynamicState> dyn = {
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE, VK_DYNAMIC_STATE_LINE_WIDTH,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE};
    if (!p.mesh) {
        dyn.push_back(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY);
        dyn.push_back(VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE);
        // The fallback vertex stage reads records by vertex index, with no vertex buffers bound.
        if (!p.fallback) dyn.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);
    }
    if (dev.eds3_blend) {
        dyn.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT);
        dyn.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT);
        dyn.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT);
    }
    if (dev.eds3_polygon) dyn.push_back(VK_DYNAMIC_STATE_POLYGON_MODE_EXT);
    if (dev.eds3_clamp) dyn.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
    if (dev.eds3_a2c) dyn.push_back(VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT);
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = (uint32_t)dyn.size();
    dy.pDynamicStates = dyn.data();
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = (uint32_t)at.n;
    ri.pColorAttachmentFormats = at.color_format;
    if (at.depth) {
        if (depth_aspect(at.depth_format)) ri.depthAttachmentFormat = at.depth_format;
        if (has_stencil(at.depth_format)) ri.stencilAttachmentFormat = at.depth_format;
    }
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.pNext = &ri;
    ci.stageCount = (uint32_t)stages.size();
    ci.pStages = stages.data();
    ci.pVertexInputState = p.mesh ? nullptr : &vi;
    ci.pInputAssemblyState = p.mesh ? nullptr : &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &bs;
    ci.pDynamicState = &dy;
    ci.layout = p.layout;
    VkGraphicsPipelineLibraryCreateInfoEXT library{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT};
    std::vector<VkPipelineShaderStageCreateInfo> part_stages;
    if (part >= 0) {
        static const VkGraphicsPipelineLibraryFlagsEXT parts[4] = {
            VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
            VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
        library.flags = parts[part];
        ri.pNext = &library;
        ci.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
        for (const VkPipelineShaderStageCreateInfo& si : stages)
            if ((part == 2 && si.stage == VK_SHADER_STAGE_FRAGMENT_BIT) || (part == 1 && si.stage != VK_SHADER_STAGE_FRAGMENT_BIT))
                part_stages.push_back(si);
        ci.stageCount = (uint32_t)part_stages.size();
        ci.pStages = part_stages.data();
        ci.pVertexInputState = part == 0 ? &vi : nullptr;
        ci.pInputAssemblyState = part == 0 ? &ia : nullptr;
        ci.pViewportState = part == 1 ? &vp : nullptr;
        ci.pRasterizationState = part == 1 ? &rs : nullptr;
        ci.pMultisampleState = part >= 2 ? &ms : nullptr;
        ci.pDepthStencilState = part == 2 ? &ds : nullptr;
        ci.pColorBlendState = part == 3 ? &bs : nullptr;
        ci.layout = part == 1 || part == 2 ? p.layout : VK_NULL_HANDLE;
        if (part != 3) {
            ri.colorAttachmentCount = 0, ri.pColorAttachmentFormats = nullptr;
            ri.depthAttachmentFormat = ri.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
        }
    }
    VkPipeline pl = VK_NULL_HANDLE;
    {   // (local) RX_VK_NOOPT=1: build every pipeline without the driver's optimizer. AMD's proprietary compiler crashes
        // (-13 here) or takes minutes on some Ocarina shaders it builds in seconds unoptimized (Lichii #200, #210).
        static const bool noopt = env_on("RX_VK_NOOPT");
        if (noopt) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        // (from PR 430, YunaraDebugs: on AMD the task and mesh programs (OoT's foliage culling, following pointers through its
        // tables) hung the GPU when the driver's optimizer built them: a device lost one to four minutes into Kokiri, each time with
        // another of them first (RX 6600, issues 461 and 484); never in runs where they were built without it. They are built
        // without it on AMD. RX_VK_MESH_OPT=1: with it, as the others)
        // (hookshot) Not on Mesa's RADV: there the task programs built without it run long enough to reset the GPU while the
        // save loads at the Great Deku Tree (Steam Deck OLED, RX 6800 XT; issue 1000). Built with it, that reset is gone; other
        // RADV hangs remain (issue 1000).
        static const bool amd = dev.props.vendorID == 0x1002 && dev.driver_id != VK_DRIVER_ID_MESA_RADV;
        static const bool mesh_opt = env_on("RX_VK_MESH_OPT");
        if (amd && !mesh_opt) {
            bool mesh = false;
            for (const Stage& st : p.stages) mesh |= (st.vk_stage & (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) != 0;
            if (mesh) {
                ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
                static std::atomic<int> told{0};
                if (told++ < 1) logf("vk: task and mesh programs built without the driver's optimizer (AMD; RX_VK_MESH_OPT=1: with it)");
            }
        }
    }
#ifdef _WIN32
    // AMD's optimizer can leave this cloth color pipeline pending after its depth pass has drawn (issues #461, #1540).
    if (p.key == 0x5fa56485a6b07619ull) {
        VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props.pNext = &driver;
        vkGetPhysicalDeviceProperties2(dev.physical, &props);
        if (driver.driverID == VK_DRIVER_ID_AMD_PROPRIETARY) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
#endif
    if (quick_build) {   // (hookshot) quick_pipeline's: none when the full build is made without the optimizer already
        if (part >= 0 || (ci.flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT)) return VK_NULL_HANDLE;
        ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
    if (cold) ci.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    // (Hookshot, RX_VK_PCACHE_MISSES: whether the pipeline cache had it. A hit adds nothing to the cache: counted as built, the
    // 1.2 GB cache was fetched and written again at every scene load, whose pipelines all come from it)
    VkPipelineCreationFeedback feedback{};
    VkPipelineCreationFeedbackCreateInfo feedback_ci{VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO};
    feedback_ci.pPipelineCreationFeedback = &feedback;
    if (pcache_misses_only() && part < 0 && !driver_cache_only) feedback_ci.pNext = ci.pNext, ci.pNext = &feedback_ci;
    // (the two shader libraries go into our cache, as whole pipelines do: asked for under the flag, the driver answers from a cache
    // it is handed and not from its own on disk, so a library the warm pass built and let go was cold again at its first draw
    // (measured: 21 of 21). gpl_libs_cached: RX_VK_GPL_NO_CACHE=1 keeps them out, as before. With our cache: through
    // graphics_with, so in play one compiled goes into the play generation, and an ask under the flag still goes to pcache)
    const bool lib_cached = (part == 1 || part == 2) && gpl_libs_cached();
    // (hookshot) a program the driver faulted on is not asked of it again, and on AMD one program's builds go to it one at a
    // time, so a second build waits here and not inside the driver behind a faulted one (driver_faults.h, issue 996). A cold
    // probe does not wait: another build of its program under way is, for it, a compile that is required. Nor does a quick
    // build: behind the program's full build it would be no quicker; the draw goes on without it
    const bool gated = driver_faults_on();
    if (gated) {
        const auto gate = fault_quarantine().begin(p.key, !cold && !quick_build);
        if (gate == driver_faults::Quarantine::kBusy) return gate_busy = true, cold ? (*cold = true, VK_NULL_HANDLE) : VK_NULL_HANDLE;
        if (gate == driver_faults::Quarantine::kFaulted) {
            static std::atomic<int> told{0};
            if (told++ < 10)
                logf("vk: a pipeline of %016llx left out: the driver faulted on that program (%s)", (unsigned long long)p.key,
                     driver_faults_path().c_str());
            return VK_NULL_HANDLE;
        }
    }
    // (hookshot, RX_VK_TEST_SLOW_MS) the worker's full build held here, inside the gate, where AMD's optimizer holds it: a quick
    // build meets the gate busy as on AMD, where the gate is serial (RX_VK_DRIVER_FAULTS=2 makes it so on every driver)
    if (slow_full_build && test_slow_ms() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(test_slow_ms()));
    // (hookshot) a quick build into no cache: an unoptimized pipeline saved would be what a later run's full build is answered with
    const VkPipelineCache cache = quick_build ? VK_NULL_HANDLE
                                  : lib_cached || (part < 0 && !driver_cache_only) ? pcache : driver_ask_cache;
    VkResult r = graphics_with(cache, &ci, &pl);
    // The AMD Windows compiler can reject ordinary graphics pipelines with its optimizer on. (A fault caught in it is kDriverFault
    // now, VK_ERROR_UNKNOWN before: the retry stays for both, under the gate.) Any program and any driver, libraries too, not
    // only where the task + mesh fallback runs: the Market's and Kakariko's dogs were white on an RX 6600 ("graphics pipeline for
    // 4e5ce183a33b0855 not made (-13)"), translated for 64-lane groups. Built without the optimizer, the draws are drawn, a
    // little slower, instead of left out. Not for the memory errors, which the optimizer does not cause, nor for a cold
    // probe's answer (VK_PIPELINE_COMPILE_REQUIRED, positive)
    if (r < 0 && r != VK_ERROR_OUT_OF_HOST_MEMORY && r != VK_ERROR_OUT_OF_DEVICE_MEMORY &&
        !(ci.flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT)) {
        const VkResult first = r;
        ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        pl = VK_NULL_HANDLE;
        r = graphics_with(cache, &ci, &pl);
        static std::atomic<int> told{0};
        if (told++ < 20)
            logf("vk: %s %016llx failed (%d) with the driver's optimizer: %s", part >= 0 ? "a library of" : "graphics pipeline for",
                 (unsigned long long)p.key, (int)first, r == VK_SUCCESS ? "made without it" : "not made without it either");
    }
    if (gated && fault_quarantine().end(p.key, r == kDriverFault)) {
        logf("vk: the driver faulted building a pipeline of %016llx: that program is not asked of it again (%s)",
             (unsigned long long)p.key, driver_faults_path().c_str());
        driver_fault_note(p.key);
    }
    if (cold && r == VK_PIPELINE_COMPILE_REQUIRED) return *cold = true, VK_NULL_HANDLE;
    if (part >= 0) {
        if (r != VK_SUCCESS) {
            static std::atomic<int> logged;
            if (logged++ < 10) logf("vk: library %d of %016llx not made (%d)", part, (unsigned long long)p.key, (int)r);
            return VK_NULL_HANDLE;
        }
        pcache_built_since_save += lib_cached && !cold;   // (one that was compiled, here or by a worker: the cache has something to save)
        return pl;
    }
    if (r != VK_SUCCESS) {
        static std::atomic<int> logged;
        if (logged++ < 10) logf("vk: graphics pipeline for %016llx not made (%d)", (unsigned long long)p.key, (int)r);
        pl = VK_NULL_HANDLE;
    }
    // (a pipeline the cache had already adds nothing to it: RX_VK_PCACHE_MISSES)
    if (!(pcache_misses_only() && (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) &&
          (feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT)))
        pcache_built_since_save += !driver_cache_only && !quick_build;
    if (!quick_build) {   // (hookshot: quick builds are counted apart, in quick_pipeline's log line)
        st_pipelines++;
        st_pipeline_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    return pl;
}

/// (hookshot) RX_VK_QUICK_MS, read once and logged: quick_pipe.h
double quick_budget_ms(bool must = false) {
    static const char* setting = getenv("RX_VK_QUICK_MS");
    static const bool required_only = [] {
        if (setting && strcmp(setting, "required") != 0) return false;
        logf("vk: quick pipelines only for draws that must wait (RX_VK_QUICK_MS=required)");
        return true;
    }();
    // take() exempts mandatory draws from the frame budget. A positive value enables that path.
    if (required_only) return must ? quick_pipe::kQuickSuggestedMs : 0;
    static const double ms = [] {
        const char* e = setting;
        bool bad = false;
        const double v = quick_pipe::quick_ms(e, &bad);
        if (bad)
            logf("vk: RX_VK_QUICK_MS=%s is not a number of milliseconds: ignored, quick pipelines off (%.0f is the suggested value)",
                 e, quick_pipe::kQuickSuggestedMs);
        if (v > 0)
            logf("vk: a draw whose pipeline is still being built gets one from the caches or made without the driver's optimizer, "
                 "%.0f ms of those a frame on each recording thread (RX_VK_QUICK_MS)", v);
        return v;
    }();
    return ms;
}
/// (hookshot) RX_VK_PIPE_WAIT_MS, read once and logged: quick_pipe.h
double pipe_wait_budget_ms() {
    static const double ms = [] {
        const char* e = getenv("RX_VK_PIPE_WAIT_MS");
        bool bad = false;
        const double v = quick_pipe::wait_ms(e, &bad);
        if (bad) logf("vk: RX_VK_PIPE_WAIT_MS=%s is not a number of milliseconds: ignored", e);
        if (v > 0)
            logf("vk: a pipeline still being built, for a draw on earlier frames' targets, is moved to the queue's front and waited "
                 "for at most %.0f ms a frame on each recording thread before the draw is left out (RX_VK_PIPE_WAIT_MS)", v);
        return v;
    }();
    return ms;
}
/// (hookshot, ported from Mako, by sharkiix, 8292d363) A pipeline for a draw whose full build is not done (quick_pipe::take):
/// the full one if a cache has it compiled, asked without compiling, else one made here without the driver's optimizer and
/// into no cache, set as the entry's pipeline and not ready: the worker's full build replaces it. At most RX_VK_QUICK_MS a frame
/// on each recording thread, but for a draw that cannot be left out (must). true: the entry has one to draw with
bool quick_pipeline(PipeEntry& e, const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, bool must) {
    const double budget_ms = quick_budget_ms(must);
    if (budget_ms <= 0 || program_modules_dropped(p)) return false;
    static thread_local quick_pipe::FrameBudget budget;
    const auto now_ms = [] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    // (busy: the driver-fault gate had another build of the program in the driver (AMD): the entry is tried again later)
    const auto ask = [&](bool& busy) -> VkPipeline {   // (needs pipelineCreationCacheControl, as pipeline_from_caches)
        if (!dev.vk13.pipelineCreationCacheControl) return VK_NULL_HANDLE;
        bool cold = false;
        gate_busy = false;
        const quick_pipe::Raise asking(ask_start_first);
        const VkPipeline pl = build_graphics(p, s, at, topo, -1, &cold);
        busy = gate_busy;
        return pl;
    };
    double quick_took = 0;
    const auto build = [&](bool& busy) -> VkPipeline {
        // Unoptimized native task/mesh shaders can reset RADV during a load (issue 1000).
        if (dev.driver_id == VK_DRIVER_ID_MESA_RADV && p.mesh && !p.fallback) return VK_NULL_HANDLE;
        const double t0 = now_ms();
        gate_busy = false;
        const quick_pipe::Raise quick(quick_build);
        const VkPipeline pl = build_graphics(p, s, at, topo);
        busy = gate_busy;
        quick_took = now_ms() - t0;
        return pl;
    };
    const auto destroy = [](VkPipeline pl) { vkDestroyPipeline(dev.device, pl, nullptr); };
    const auto poll = [] { std::this_thread::sleep_for(std::chrono::milliseconds(1)); };
    const quick_pipe::Got got = quick_pipe::take(e, must, budget_ms, budget, texture_frame_no(), ask, build, destroy, now_ms, poll);
    // (each one made here stays for the session once the full build replaces it: graphics_pipeline's job. The counts below
    // are those kept pipelines)
    if (got == quick_pipe::Got::kCached) {
        static std::atomic<int64_t> hits{0};
        const int64_t h = ++hits;
        if (h <= 5 || h % 200 == 0)
            logf("vk: pipeline from the caches for a draw ahead of its build (%lld so far, each kept for the session)", (long long)h);
    } else if (got == quick_pipe::Got::kQuick) {
        static std::atomic<int64_t> n{0}, total_us{0};
        const int64_t k = ++n;
        total_us += (int64_t)(quick_took * 1000);
        if (k <= 20 || k % 50 == 0)
            logf("vk: quick pipeline (no optimizer) for %016llx in %.1f ms, drawn with until its full build is done (%lld so far, "
                 "%.0f ms in all, each kept for the session)", (unsigned long long)p.key, quick_took, (long long)k,
                 total_us.load() / 1000.0);
    }
    return got != quick_pipe::Got::kNothing;
}

// ---- samplers (by NVN sampler parameters) ----
/// (hookshot) What vkCreateSampler reads of the create info sampler_make fills in, less what no sample can reach: the anisotropy
/// of a sampler without anisotropic filtering (Vulkan ignores it), the border colour of one no address mode clamps to its border.
/// Two keys whose infos are equal get samplers that sample alike, so they share one (sampler_book). The comparison of a sampler
/// that does not compare stays in it: a driver may compare anyway (the amd-fix note below). 32-bit
/// words without padding, compared as bytes (a bias of -0 apart from one of +0: two samplers, as two keys were)
struct SamplerInfo {
    uint32_t mag, min, mip, u, v, w;
    float bias, aniso, min_lod, max_lod;
    uint32_t aniso_on, compare_on, compare_op, border;
    float custom[4];      // (a custom border colour's, else 0)
    uint32_t reduction;   // (VK_SAMPLER_REDUCTION_MODE_* + 1; 0: none)
    bool operator==(const SamplerInfo& o) const { return !std::memcmp(this, &o, sizeof o); }
};
static_assert(sizeof(SamplerInfo) == 19 * 4, "SamplerInfo is compared as bytes: no padding");
struct SamplerInfoHash {
    size_t operator()(const SamplerInfo& i) const {
        uint32_t w[sizeof i / 4];
        std::memcpy(w, &i, sizeof i);
        uint64_t h = 0x5A;
        for (const uint32_t x : w) h = mix(h, x);
        return (size_t)h;
    }
};
SamplerInfo sampler_info(const VkSamplerCreateInfo& ci) {
    SamplerInfo i{};
    i.mag = ci.magFilter, i.min = ci.minFilter, i.mip = ci.mipmapMode;
    i.u = ci.addressModeU, i.v = ci.addressModeV, i.w = ci.addressModeW;
    i.bias = ci.mipLodBias, i.min_lod = ci.minLod, i.max_lod = ci.maxLod;
    i.aniso_on = ci.anisotropyEnable, i.aniso = ci.anisotropyEnable ? ci.maxAnisotropy : 0.0f;
    i.compare_on = ci.compareEnable, i.compare_op = ci.compareOp;
    const VkSamplerAddressMode b = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    const bool border = ci.addressModeU == b || ci.addressModeV == b || ci.addressModeW == b;
    i.border = border ? (uint32_t)ci.borderColor : 0u;
    for (auto* n = static_cast<const VkBaseInStructure*>(ci.pNext); n; n = n->pNext) {
        if (n->sType == VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT && border)
            std::memcpy(i.custom, reinterpret_cast<const VkSamplerCustomBorderColorCreateInfoEXT*>(n)->customBorderColor.float32,
                        sizeof i.custom);
        else if (n->sType == VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO)
            i.reduction = (uint32_t)reinterpret_cast<const VkSamplerReductionModeCreateInfo*>(n)->reductionMode + 1u;
    }
    return i;
}
/// (hookshot) The samplers sampler_make made: one for each SamplerInfo, and the keys that resolve to each, under the replay's lock
/// (ParLock; the atomics are read without it). A driver makes at most maxSamplerAllocationCount samplers (NVIDIA 4000, NVK 4096):
/// past that vkCreateSampler fails, and a draw with a failed sampler is refused (bind_descriptors). So before a new one would go
/// past the room sampler_room keeps, the least recently resolved one is retired: out of every table at once, and destroyed once
/// the GPU is past every recording made so far (vk_retirement_value), since a thread's own tables (sampler_for's, mine) may still
/// hand it to the draws it is recording until it reads retired_gen again. Those tables hide a sampler's use from the book, so
/// from half the retiring point on (watching) their entries last a present: each sampler in use resolves here once a present
struct SamplerBook {
    struct Made { VkSampler s; uint64_t used; };       // (used: tick when a key last resolved to it here; only the order to retire in)
    struct Retired { VkSampler s; uint64_t after; };   // (destroyed once the timeline reaches after; 0: never)
    std::unordered_map<SamplerInfo, Made, SamplerInfoHash> made;   // (the owner)
    std::unordered_map<uint64_t, Made*> keys;   // (sampler_make's key -> its sampler in made; several keys may share one)
    std::vector<Retired> retired;
    uint64_t tick = 0, retirements = 0;
    std::atomic<uint64_t> retired_gen{0};   // (moves at each retirement: each thread's sampler_make drops its mine)
    std::atomic<bool> watching{false};      // (near the device's limit, for good: the threads' tables last a present)
};
SamplerBook sampler_book;
/// (hookshot) The retired samplers the GPU is done with (the timeline at completed), destroyed
void samplers_reap(uint64_t completed) {
    std::erase_if(sampler_book.retired, [completed](const SamplerBook::Retired& r) {
        if (!r.after || r.after > completed) return false;
        vkDestroySampler(dev.device, r.s, nullptr);
        return true;
    });
}
/// (hookshot) The least recently resolved sampler retired (sampler_book). false: none, or no room to note it (it stays made: one
/// not noted would never be destroyed)
bool sampler_retire() {
    SamplerBook& b = sampler_book;
    if (b.made.empty()) return false;
    auto lru = b.made.begin();
    for (auto it = b.made.begin(); it != b.made.end(); ++it)
        if (it->second.used < lru->second.used) lru = it;
    try {
        b.retired.push_back({lru->second.s, vk_retirement_value()});
    } catch (const std::exception&) {
        return false;
    }
    const SamplerBook::Made* gone = &lru->second;
    std::erase_if(b.keys, [gone](const auto& kv) { return kv.second == gone; });
    b.made.erase(lru);
    b.retired_gen.fetch_add(1, std::memory_order_acq_rel);
    samplers_changed();   // (sampler_for's tables and the descriptors kept by reuse resolve theirs again)
    if (++b.retirements == 1 || b.retirements % 1000 == 0)
        logf("vk: samplers: %zu made, the device allows %u: the least recently used retired (%llu so far), each destroyed once "
             "the GPU is past the recordings that may use it", b.made.size() + 1, dev.props.limits.maxSamplerAllocationCount,
             (unsigned long long)b.retirements);   // (made + 1: as many as there were)
    return true;
}
/// (hookshot) Whether sampler_make may make one more sampler. The room: maxSamplerAllocationCount (4000, Vulkan's least, when not
/// known) less a reserve of 64 (a quarter of a smaller limit) for the renderer's other samplers (the blits', the upscalers', frame
/// generation's) and an upscaler SDK's. From 3/4 of it the least recently resolved ones are retired, one for each made; the last
/// 1/4 is for those made while retired ones are still the GPU's. Full only when the GPU may still use every one of those: then
/// nothing is retired and the draw waits for a later present (the failed key's rule), as one whose creation failed does
bool sampler_room() {
    SamplerBook& b = sampler_book;
    const uint64_t limit = dev.props.limits.maxSamplerAllocationCount ? dev.props.limits.maxSamplerAllocationCount : 4000;
    const uint64_t room = limit - std::min<uint64_t>(64, limit / 4), keep = room - room / 4;
    if (!b.retired.empty()) samplers_reap(vk_timeline_completed());
    if (b.made.size() + b.retired.size() >= room) {
        static std::atomic<int> told{0};
        if (told++ < 8)
            logf("vk: samplers: %zu made and %zu retired that the GPU may still use fill the %llu this device allows "
                 "(maxSamplerAllocationCount %u): a draw needing a new one waits for a later present", b.made.size(),
                 b.retired.size(), (unsigned long long)room, dev.props.limits.maxSamplerAllocationCount);
        return false;
    }
    if (b.made.size() >= keep / 2) b.watching.store(true, std::memory_order_relaxed);
    while (b.made.size() >= keep && sampler_retire()) {}
    return true;
}
VkSamplerAddressMode wrap(uint8_t w) {
    switch (w) {
    case NVN_WRAP_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case NVN_WRAP_MODE_MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case NVN_WRAP_MODE_CLAMP_TO_BORDER: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case NVN_WRAP_MODE_MIRROR_CLAMP: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}
// (hookshot) The Anisotropic filtering row (opts.anisotropy): the samplers that filter linearly between texels and have mip
// levels get at least n-times anisotropic filtering (the game's value if it asks for more; the device's limit if n is above it):
// surfaces seen at a slant, the ground ahead above all, keep their detail. OoT asks for 1x or 2x (its log line below), which
// at a PC's resolutions smears paths and floors a few metres off. 16x by default; Game's: what the game asks for
uint32_t anisotropy_row() {
    static const uint32_t times[] = {0, 2, 4, 8, 16};
    const int v = opts.anisotropy;
    return v >= 0 && v < 5 ? times[v] : 16u;
}
// RX_VK_ANISO=<n> over the row for a run (0: the game's own). Part of every sampler's key below, so no cached sampler is one made
// for another n; a change of the row moves the samplers' generation (samplers_changed), so sampler_for's tables and the
// descriptors kept by reuse take the new ones
uint32_t aniso_override() {
    static const int env = getenv("RX_VK_ANISO") ? std::clamp(atoi(getenv("RX_VK_ANISO")), 0, 16) : -1;
    return env >= 0 ? (uint32_t)env : anisotropy_row();
}
// (hookshot) The upscalers' texture LOD bias. OoT draws its 3D at 3/4 of the output and gives its world's materials a LOD bias of
// -0.5 for that (sampler_make's log line). An upscaler that draws from another share adds to the samplers the game biases below 0
// (upscaler_rows.h upscale_lod_bias_for): DLSS and DLAA NVIDIA's log2(render / output) - 1 in all (Quality -1.17 more, DLAA -0.58,
// Auto -1), FSR 3.1 and FSR 4 log2(share / 0.75) below 75 %; 0 with the game's own upscale, Off and FSR 4's native AA.
// RX_VK_UPSCALE_LOD_BIAS=0: off. RX_VK_DLSS_LOD_OFFSET=<steps>: DLSS's -1 for a run (0: the render size's own detail, as
// before). Part of every sampler's key; the menu's Upscaler, Render resolution and Quality preset rows move the samplers' generation
float upscale_lod_bias() {
    static const bool off = getenv("RX_VK_UPSCALE_LOD_BIAS") && atoi(getenv("RX_VK_UPSCALE_LOD_BIAS")) == 0;
    if (off) return 0.0f;
    static const float offset = [] {
        const char* e = getenv("RX_VK_DLSS_LOD_OFFSET");
        const float v = e && *e ? (float)atof(e) : -1.0f;
        return std::isfinite(v) ? std::clamp(v, -3.0f, 0.0f) : -1.0f;
    }();
    const int up = fsr3_wanted() ? (int)upscaler_rows::FSR3 : temporal_mode();
    return upscaler_rows::upscale_lod_bias_for(up, resolution_scale(), offset);   // (share in 1/10000 of the output; 0: Auto)
}
/// compare: the shader reads depth with a reference (a sampler without one cannot compare). one_level: the image bound with it
/// has a single level, and the Anisotropic filtering row's anisotropy is not added (the game's own stays)
VkSampler sampler_make(int id, bool compare, bool one_level);
// (each texture of each draw found its sampler through reg_mu, a copy of the registration, a hash of it and a map lookup:
// ~48000 times a frame at 3000 draws. Now a thread's table by id and compare, good while no sampler registration's contents
// changed (sampler_registrations_generation). RX_VK_SAMPLER_CACHE=0 turns off both this table and the parameters' cache inside
// sampler_make; =1 or the Sampler cache row turns the parameters' cache on (this table stays on, as it is without the option)
bool sampler_cache_on();   // (hookshot, RX_VK_SAMPLER_CACHE and the Sampler cache row: defined below with the descriptor fingerprint's)
VkSampler sampler_for(int id, bool compare, bool one_level) {
    static const bool on = !getenv("RX_VK_SAMPLER_CACHE") || atoi(getenv("RX_VK_SAMPLER_CACHE")) != 0;
    if (!on || id < 0 || id >= 4096) return sampler_make(id, compare, one_level);
    struct Entry {
        uint64_t gen, frame, retired;
        VkSampler s;
        uint32_t slot_version, aniso;
        float lod_bias;
    };
    static thread_local Entry table[2][2][4096];   // (gen 0: never filled; the generation starts at 1)
    const uint64_t gen = sampler_registrations_generation();   // (read first: a registration after it makes the next call miss)
    // (near the device's sampler limit, an entry lasts a present: the book sees the samplers in use, sampler_book)
    const uint64_t frame = sampler_book.watching.load(std::memory_order_relaxed) ? texture_frame_no() : 0;
    Entry& e = table[compare ? 1 : 0][one_level ? 1 : 0][id];
    static const bool reuse = env_on("RX_VK_BINDING_REUSE");
    // Registrations publish the global generation before the slot version. Check both
    // on the opt-in path so a lookup between those publications cannot keep a stale sampler.
    const uint32_t version = reuse ? sampler_slot_version(id) : 0;
    if (e.gen == gen && e.frame == frame && (!reuse || e.slot_version == version)) return e.s;
    const uint64_t retired = reuse ? sampler_book.retired_gen.load(std::memory_order_acquire) : 0;
    const uint32_t aniso = reuse && !one_level ? aniso_override() : 0;
    const float bias = reuse ? upscale_lod_bias() : 0.0f;
    if (reuse && e.s && e.frame == frame && e.retired == retired && e.slot_version == version &&
        e.aniso == aniso && e.lod_bias == bias) {
        e.gen = gen;
        return e.s;
    }
    const VkSampler made = sampler_make(id, compare, one_level);
    if (made) e = {gen, frame, retired, made, version, aniso, bias};   // (a failed creation is retryable)
    return made;
}
VkSampler sampler_make(int id, bool compare, bool one_level) {
    nvn_samp_par p{};
    bool known = false;
    if (sampler_cache_on()) {   // (hookshot: what an id holds stays cached with its version on this thread: resolving an id
        static thread_local std::unordered_map<int, std::pair<uint32_t, nvn_samp_par>> seen;   // (that was not registered
        const uint32_t v = sampler_slot_version(id);                                            // (anew takes no lock)
        if (auto it = seen.find(id); it != seen.end() && it->second.first == v)
            p = it->second.second, known = true;
        else if (sampler_of(id, &p)) {
            known = true;
            try { seen[id] = {v, p}; } catch (const std::exception&) {}   // (only an acceleration cache)
        }
    } else
        known = sampler_of(id, &p);
    if (!known) p.minf = NVN_MIN_FILTER_LINEAR, p.magf = NVN_MAG_FILTER_LINEAR, p.lod_max = 1000.0f;
    const float up_bias = p.lod_bias < 0.0f ? upscale_lod_bias() : 0.0f;   // (the game's upscaled materials only)
    uint32_t up_bits;
    std::memcpy(&up_bits, &up_bias, sizeof up_bits);
    // (one_level: the row's anisotropy is not added, so it is not part of the key either)
    const uint32_t row_aniso = one_level ? 0u : aniso_override();
    uint64_t key = mix(mix(compare ? 0xC0 : 0xAB, row_aniso), up_bits);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&p);
    for (size_t i = 0; i < sizeof p; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, b + i, std::min<size_t>(8, sizeof p - i));
        key = mix(key, w);
    }
    static thread_local std::unordered_map<uint64_t, VkSampler> mine;   // (dropped at a retirement, its samplers may be going, and
    static thread_local uint64_t mine_gen = 0, mine_frame = 0;          // at each present near the limit: sampler_for's note)
    const uint64_t g = sampler_book.retired_gen.load(std::memory_order_acquire);
    const uint64_t f = sampler_book.watching.load(std::memory_order_relaxed) ? texture_frame_no() : 0;
    if (g != mine_gen || f != mine_frame) mine.clear(), mine_gen = g, mine_frame = f;
    if (auto m = mine.find(key); m != mine.end()) return m->second;
    ParLock lk;
    SamplerBook& book = sampler_book;
    if (auto it = book.keys.find(key); it != book.keys.end()) {
        it->second->used = ++book.tick;
        try { mine.emplace(key, it->second->s); } catch (const std::exception&) {}   // (the book still owns it)
        return it->second->s;
    }
    // Keep valid cache hits ahead of failure tracking. A bad key gets at most one failed attempt per present;
    // unrelated keys can still work, and a later present can retry without changing the registration.
    static uint64_t failed_frame = UINT64_MAX;
    static std::unordered_set<uint64_t> failed;
    static bool tracking_failed = false;
    const uint64_t frame = texture_frame_no();
    if (failed_frame != frame) {
        failed.clear();
        failed_frame = frame;
        tracking_failed = false;
    }
    if (tracking_failed || failed.count(key)) return VK_NULL_HANDLE;
    auto remember_failure = [&] {
        try { failed.insert(key); } catch (const std::exception&) { tracking_failed = true; }
    };
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = p.magf == NVN_MAG_FILTER_NEAREST ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    ci.minFilter = (p.minf == NVN_MIN_FILTER_NEAREST || p.minf == NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST ||
                    p.minf == NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    ci.mipmapMode = (p.minf == NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR || p.minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR)
                        ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    {   // (hookshot, issue 1117: blocky halos, smoke and fog shafts since v2.0.6, gone with every sampler linear. The nearest ones
        // named, with the game's own words for them; RX_VK_FORCE_LINEAR=1: every sampler linear, for the A/B)
        static const bool force_linear = env_on("RX_VK_FORCE_LINEAR");
        if (ci.magFilter == VK_FILTER_NEAREST || ci.minFilter == VK_FILTER_NEAREST) {
            static std::atomic<int> told{0};
            if (told++ < 24)
                logf("vk: sampler %d is nearest: min filter %u, mag filter %u, wrap %u/%u/%u, lod %g..%g bias %g, aniso %g, compare %d%s", id,
                     (unsigned)p.minf, (unsigned)p.magf, (unsigned)p.ws, (unsigned)p.wt, (unsigned)p.wr, (double)p.lod_min,
                     (double)p.lod_max, (double)p.lod_bias, (double)p.aniso, (int)compare, force_linear ? " (made linear: RX_VK_FORCE_LINEAR)" : "");
            if (force_linear) ci.magFilter = ci.minFilter = VK_FILTER_LINEAR;
        }
    }
    ci.addressModeU = wrap(p.ws), ci.addressModeV = wrap(p.wt), ci.addressModeW = wrap(p.wr);
    ci.mipLodBias = std::clamp(p.lod_bias + up_bias, -dev.props.limits.maxSamplerLodBias, dev.props.limits.maxSamplerLodBias);
    float aniso = p.aniso;
    {   // (what the game asks for is told once for each anisotropy, filter, LOD bias and one level or not, 16 at most: not known
        // otherwise. one_level is in the key: a sampler kind first bound with an image of one level would else tell 0 for all)
        static std::mutex am;
        static std::unordered_set<uint64_t> asked;
        uint32_t bits, bias;
        std::memcpy(&bits, &p.aniso, sizeof bits);
        std::memcpy(&bias, &p.lod_bias, sizeof bias);
        std::lock_guard ag(am);
        if (asked.size() < 16 && asked.insert((uint64_t)bits << 32 ^ (uint64_t)bias << 3 ^ p.minf ^ (uint64_t)one_level << 63).second)
            logf("vk: sampler anisotropy: the game asks for %g (min filter %u, mip filter %s, LOD bias %g; made with %u%s)", (double)p.aniso,
                 (unsigned)p.minf, ci.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? "linear" : "nearest", (double)p.lod_bias, row_aniso,
                 one_level ? ", an image of one level: not the row's" : "");
    }
    // (a linear filter with mip levels, linear or nearest between them: OoT asks for both, 2x with linear levels and 1x with
    // nearest ones, its world's materials with a LOD bias of -0.5 either way. Not for an image of one level: the game's own
    // pictures it reads back (a half-size fog and particle layer, the scene's depth), where a 2x2 block of pixels across an
    // object's edge has texture coordinates far apart. An anisotropic filter then takes texels along that whole span, from
    // the sky behind a mountain's ridge into the ridge: dashed bright lines along every edge on AMD's RX 7900 XTX. With
    // the row's 16x the game's mipmapped sampler did that to OoT's composite of its half-size layer, 1a7482c0ec9991ea)
    if (row_aniso && ci.minFilter == VK_FILTER_LINEAR && p.minf != NVN_MIN_FILTER_LINEAR)
        aniso = std::max(aniso, (float)row_aniso);
    ci.anisotropyEnable = aniso > 1.0f && dev.features.samplerAnisotropy;
    ci.maxAnisotropy = std::clamp(aniso, 1.0f, dev.props.limits.maxSamplerAnisotropy);
    ci.compareEnable = compare && p.cmp_mode != 0;
    ci.compareOp = compare_op(p.cmp_func);
    {   // (amd-fix) A depth-reference sample through a sampler that does not compare is undefined in Vulkan: NVIDIA's driver
        // compares anyway, AMD's does not (OoT samples this way: compare mode 0, function 0). A shader that compares gets
        // a comparing sampler: the game's function when it set one, LEQUAL otherwise
        if (compare && !ci.compareEnable) {
            ci.compareEnable = VK_TRUE;
            ci.compareOp = p.cmp_func >= 1 && p.cmp_func <= 8 ? compare_op(p.cmp_func) : VK_COMPARE_OP_LESS_OR_EQUAL;
        }
        static std::mutex lm;
        static std::unordered_set<uint32_t> told;
        std::lock_guard lg(lm);
        if (compare && told.size() < 16 && told.insert((uint32_t)p.cmp_mode << 8 | p.cmp_func).second)
            logf("vk: depth-compare sampler: the game's compare mode %u function %u -> compare %s op %d (filters min %d mag %d)",
                 p.cmp_mode, p.cmp_func, ci.compareEnable ? "on" : "off", (int)ci.compareOp, (int)ci.minFilter, (int)ci.magFilter);
    }
    {   // (hookshot) A depth comparison through a nearest filter is one test a pixel: shadow edges in stair steps, the shadow map's
        // texels as squares over Hyrule Field's hills (Discord, 2026-11-09). Linear, the GPU tests the four nearest texels and blends
        // them (percentage-closer filtering), the soft edge the console's picture has
        if (ci.compareEnable) ci.magFilter = ci.minFilter = VK_FILTER_LINEAR;
    }
    ci.minLod = std::max(0.0f, p.lod_min);
    // (no mipmap filter: the base level only, as NVN samples it)
    ci.maxLod = (p.minf == NVN_MIN_FILTER_NEAREST || p.minf == NVN_MIN_FILTER_LINEAR) ? ci.minLod + 0.25f
                                                                                       : std::max(ci.minLod, p.lod_max);
    VkSamplerCustomBorderColorCreateInfoEXT bc{VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    const bool black = !p.border[0] && !p.border[1] && !p.border[2];
    const bool white = p.border[0] == 1.0f && p.border[1] == 1.0f && p.border[2] == 1.0f;
    if (black && !p.border[3]) ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    else if (black && p.border[3] == 1.0f) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    else if (white && p.border[3] == 1.0f) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    else if (dev.custom_border_color) {
        ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
        std::memcpy(bc.customBorderColor.float32, p.border, 16);
        bc.format = VK_FORMAT_UNDEFINED;
        ci.pNext = &bc;
    } else {
        ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
    if (p.reduction && !compare) {   // (min / max filtering)
        static VkSamplerReductionModeCreateInfo rm{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO};
        rm.reductionMode = p.reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX;
        rm.pNext = (void*)ci.pNext;
        ci.pNext = &rm;
    }
    const SamplerInfo info = sampler_info(ci);
    if (auto it = book.made.find(info); it != book.made.end()) {   // (another key's sampler samples alike: shared, not made again)
        it->second.used = ++book.tick;
        try { book.keys.emplace(key, &it->second); } catch (const std::exception&) {}   // (only the keys' table: made owns it)
        return it->second.s;
    }
    if (!sampler_room()) {   // (the device's samplers all in the GPU's hands: none made past its limit)
        remember_failure();
        return VK_NULL_HANDLE;
    }
    VkSampler sm = VK_NULL_HANDLE;
    if (vkCreateSampler(dev.device, &ci, nullptr, &sm) != VK_SUCCESS) {
        remember_failure();   // (an error's output handle is undefined, not a resource to cache or destroy)
        return VK_NULL_HANDLE;
    }
    SamplerBook::Made* made = nullptr;
    try {
        made = &book.made.emplace(info, SamplerBook::Made{sm, ++book.tick}).first->second;
    } catch (const std::exception&) {
        vkDestroySampler(dev.device, sm, nullptr);   // (only the new sampler whose ownership was not committed)
        remember_failure();
        return VK_NULL_HANDLE;
    }
    try { book.keys.emplace(key, made); } catch (const std::exception&) {}   // (only the keys' table: the next look finds it by info)
    return sm;
}

VkImageViewType view_type(Shader::TextureType t) {
    switch (t) {
    case Shader::TextureType::Color1D: return VK_IMAGE_VIEW_TYPE_1D;
    case Shader::TextureType::ColorArray1D: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case Shader::TextureType::ColorArray2D: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case Shader::TextureType::Color3D: return VK_IMAGE_VIEW_TYPE_3D;
    case Shader::TextureType::ColorCube: return VK_IMAGE_VIEW_TYPE_CUBE;
    case Shader::TextureType::ColorArrayCube: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}
VkFormat image_format(Shader::ImageFormat f) {
    switch (f) {
    case Shader::ImageFormat::R8_UINT: return VK_FORMAT_R8_UINT;
    case Shader::ImageFormat::R8_SINT: return VK_FORMAT_R8_SINT;
    case Shader::ImageFormat::R16_UINT: return VK_FORMAT_R16_UINT;
    case Shader::ImageFormat::R16_SINT: return VK_FORMAT_R16_SINT;
    case Shader::ImageFormat::R32_UINT: return VK_FORMAT_R32_UINT;
    case Shader::ImageFormat::R32G32_UINT: return VK_FORMAT_R32G32_UINT;
    case Shader::ImageFormat::R32G32B32A32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

// Dummy descriptors for what does not resolve (the null descriptor where the device has it)
struct Dummies {
    Texture* tex[7]{};
    VkImageView view[7]{};
} dummies;

/// A handle word pair as the shader reads it (Eden's TexturePair: primary, optionally ORed with a secondary)
template <class D>
uint32_t handle_of(const NvnState& s, const Stage& st, const D& d, uint32_t k) {
    if constexpr (requires { d.handle_program; }) {   // (read from a storage buffer: its handle program's; of a table, element k's)
        if (d.handle_program != ~0U) return stage_handle(s, st, d.handle_program, k);
    }
    const uint32_t step = 1u << d.size_shift;
    uint32_t h = stage_cbuf_word(s, st, d.cbuf_index, d.cbuf_offset + k * step);
    if constexpr (requires { d.has_secondary; }) {
        if (d.has_secondary) {
            const uint32_t a = h << d.shift_left;
            const uint32_t b = stage_cbuf_word(s, st, d.secondary_cbuf_index, d.secondary_cbuf_offset + k * step)
                               << d.secondary_shift_left;
            h = a | b;
        }
    }
    return h;
}

/// (Fritchii d603d98bb, HandleIsStoragePointer) The handle word of texture k is where a storage buffer of the same program
/// keeps its 8-byte address (and the handle is not computed per draw). (Lichii 282a585a) Or the translation saw the word
/// reach the handle only as a load's address (TextureDescriptor::handle_is_pointer_word), storage buffer or not
bool handle_is_storage_pointer(const Shader::Info& info, const Shader::TextureDescriptor& d, uint32_t k) {
    if (d.handle_program != ~0U) return false;
    if (d.handle_is_pointer_word) return true;
    const uint32_t at = d.cbuf_offset + (k << d.size_shift);
    for (const auto& sb : info.storage_buffers_descriptors) {
        if (sb.cbuf_index != d.cbuf_index) continue;
        for (uint32_t j = 0; j < sb.count; j++) {
            const uint32_t a = sb.cbuf_offset + j * 16;
            if (at >= a && at < a + 8) return true;
        }
    }
    return false;
}

/// The view a texture handle samples (null: nothing resolves)
/// The handles a recording resolved, with what they gave (draws bind the same textures again and again: the slot
/// lookup, the cache's key and the view's are made once a recording, and again when a slot is registered anew)
struct HandleHit {
    VkImageView view;
    Texture* tex;
    uint64_t destroy_gen;
    uint32_t level;   // (the image level the view starts at)
    uint32_t version; // (the slot's, when resolved)
    uint32_t tex_epoch;   // (the texture epoch it was last touched in)
    bool sampling;    // (its use reads the texture's contents)
};
TLS_OBJECT(handle_hits, FlatMap<uint64_t, HandleHit>)
/* (hookshot) a direct-mapped front for handle_hits. The same texture bindings are resolved again and again (14.9 descriptors
   a draw, 725 draws, 10,600 a frame) and a hash map find for each was 0.9 ms a frame. The map stays the authority: this
   answers only a lookup whose key, slot version, destroy generation and texture epoch are all the ones it was filled with,
   so a slot re-registered or a texture destroyed misses it at once and takes the map's path. RX_VK_VIEW_FAST=0: as before. */
struct ViewFast {
    uint64_t key = ~0ull;
    VkImageView view = VK_NULL_HANDLE;
    Texture* tex = nullptr;
    uint32_t level = 0, version = 0;
    uint64_t destroy_gen = 0, tex_epoch = 0;
    bool sampling = false;
};
TLS_OBJECT(view_fast, std::array<ViewFast, 1024>)
// (hookshot) RX_VK_TRACE_TEX=<w>x<h>,...: each draw sampling a texture of that size is logged with its state, every 30th frame, 600 lines: a
// model that does not show is found by its textures, and whether the game draws it, and how, read from the line
thread_local const Texture* trace_tex_hit = nullptr;
thread_local uint32_t trace_tex_mask = 0;
bool trace_tex_wanted(const Texture* t) {
    static const std::vector<std::pair<uint32_t, uint32_t>> want = [] {
        std::vector<std::pair<uint32_t, uint32_t>> v;
        if (const char* e = getenv("RX_VK_TRACE_TEX"))
            for (const char* q = e; *q;) {
                uint32_t w = 0, h = 0;
                if (sscanf(q, "%ux%u", &w, &h) == 2) v.push_back({w, h});
                q = strchr(q, ',');
                if (!q) break;
                q++;
            }
        return v;
    }();
    for (size_t i = 0; i < want.size(); i++)
        if (t->width == want[i].first && t->height == want[i].second) trace_tex_mask |= 1u << i;
    return !want.empty() && trace_tex_mask == (1u << want.size()) - 1;   // (a draw with all of them)
}
thread_local VkFormat last_view_format = VK_FORMAT_UNDEFINED;   // (hookshot: RX_VK_LOG_IMG, the view last made)
bool log_img_wanted(uint64_t key) {   // (hookshot) RX_VK_LOG_IMG=<key prefix,...>: its texture and image bindings, 8 times
    static const char* want = getenv("RX_VK_LOG_IMG");
    if (!want) return false;
    char hex[24];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)key);
    for (const char* q = want; *q;) {
        const char* e = q;
        while (*e && *e != ',') e++;
        if (e > q && !strncmp(hex, q, (size_t)(e - q))) return true;
        q = *e ? e + 1 : e;
    }
    return false;
}

VkImageView texture_view_of(uint32_t handle, VkImageViewType want, ViewUse use, VkFormat storage_as = VK_FORMAT_UNDEFINED,
                            Texture** tex_out = nullptr, uint32_t* level_out = nullptr) {
    const uint32_t version = texture_slot_version((int)(handle & 0xFFFFF));
    const uint64_t dgen = texture_destroy_generation();
    const uint64_t key = (uint64_t)(handle & 0xFFFFF) | (uint64_t)want << 20 | (uint64_t)use << 28 | (uint64_t)storage_as << 32;
    static const bool view_fast_on = !getenv("RX_VK_VIEW_FAST") || atoi(getenv("RX_VK_VIEW_FAST")) != 0;
    ViewFast* vf = view_fast_on ? &view_fast()[(key * 0x9E3779B97F4A7C15ull) >> 54] : nullptr;
    // Keep the existing view through a small number of unrelated evictions. The gone ring
    // rejects the texture's own removal, compaction, and any history it cannot prove.
    static const bool reuse = env_on("RX_VK_BINDING_REUSE");
    if (reuse && vf && vf->key == key && vf->version == version && vf->view && vf->destroy_gen != dgen &&
        dgen - vf->destroy_gen <= 8 && texture_kept_since(vf->tex, vf->destroy_gen, dgen))
        vf->destroy_gen = dgen;
    if (vf && vf->key == key && vf->version == version && vf->destroy_gen == dgen) {
        // (the epoch's touch is done here, as the map's path does it: the map's own copy of the epoch may go stale, and then
        // the first lookup of that key through the map touches the texture again, which only marks it used twice)
        if (vf->tex && vf->tex_epoch != texture_epoch()) {
            if (!texture_touch_quick(vf->tex, vf->sampling)) {
                ParLock lk;
                texture_touch(vf->tex, vf->sampling);
            }
            vf->tex_epoch = texture_epoch();
        }
        if (tex_out) *tex_out = vf->tex;
        if (level_out) *level_out = vf->level;
        return vf->view;
    }
    static const bool log_hits = env_on("RX_VK_LOG_HITS");
    static int64_t st_hit, st_new, st_ver, st_gen, st_n;
    const auto it = handle_hits().find(key);
    // A surviving view can be reused after an unrelated eviction. Misses still resolve under ParLock.
    if (it != handle_hits().end() && it->second.version == version && it->second.destroy_gen != dgen &&
        it->second.view && texture_gone_ring_on() && texture_kept_since(it->second.tex, it->second.destroy_gen, dgen))
        it->second.destroy_gen = dgen;
    if (log_hits) {   // (the counters were a second search of this map on every binding, with the log off)
        if (++st_n % 200000 == 0)
            logf("vk: handle hits %lld, new %lld, slot re-registered %lld, a texture destroyed %lld", (long long)st_hit,
                 (long long)st_new, (long long)st_ver, (long long)st_gen);
        if (it == handle_hits().end()) st_new++;
        else if (it->second.version != version) st_ver++;
        else if (it->second.destroy_gen != dgen) st_gen++;
        else st_hit++;
    }
    if (it != handle_hits().end() && it->second.version == version && it->second.destroy_gen == dgen) {
        if (it->second.tex && it->second.tex_epoch != texture_epoch()) {   // (once a submission: its age and contents)
            if (!texture_touch_quick(it->second.tex, it->second.sampling)) {
                ParLock lk;
                texture_touch(it->second.tex, it->second.sampling);
            }
            it->second.tex_epoch = texture_epoch();
        }
        if (vf)
            vf->key = key, vf->view = it->second.view, vf->tex = it->second.tex, vf->level = it->second.level,
            vf->version = it->second.version, vf->destroy_gen = it->second.destroy_gen, vf->tex_epoch = it->second.tex_epoch,
            vf->sampling = it->second.sampling;
        if (tex_out) *tex_out = it->second.tex;
        if (level_out) *level_out = it->second.level;
        return it->second.view;
    }
    ParLock lk;
    nvn_tex t;
    nvn_view v;
    bool has_view = false;
    VkImageView view = VK_NULL_HANDLE;
    Texture* tex = nullptr;
    uint32_t level = 0;
    const bool known = texture_of((int)(handle & 0xFFFFF), &t, &v, &has_view);
    if (known) {
        tex = texture_get(&t, use == ViewUse::Sample || use == ViewUse::Storage);
        if (tex) view = texture_view(tex, &t, has_view ? &v : nullptr, want, use, &last_view_format, storage_as);
        if (tex) level = image_level(tex, has_view ? view_level(&v) : 0);
    }
    if (!view) {   // (what a shader reads as the dummy instead: once per texture slot)
        static std::unordered_map<uint32_t, int> told;
        if (told.size() < 64 && !told[handle & 0xFFFFF]++) {
            if (!known)
                logf("vk: texture slot %#x (handle %#x) is not registered: sampled as the dummy", handle & 0xFFFFF, handle);
            else
                logf("vk: texture slot %#x (handle %#x): %s; NVN target %d format %#x %ux%ux%u levels %u gpu %#llx, view %s "
                     "(format %#x target %d), wanted view type %d use %d",
                     handle & 0xFFFFF, handle, tex ? "no view of it" : "no image for it", t.d.target, t.d.format, t.d.width,
                     t.d.height, t.d.depth, t.d.levels, (unsigned long long)t.gpu, has_view ? "yes" : "no",
                     has_view && v.set_format ? v.format : 0, has_view && v.set_target ? v.target : -1, (int)want, (int)use);
        }
    }
    handle_hits()[key] = {view, tex, dgen, level, version, texture_epoch(), use == ViewUse::Sample || use == ViewUse::Storage};
    if (vf) vf->key = key, vf->view = view, vf->tex = tex, vf->level = level, vf->version = version, vf->destroy_gen = dgen,
            vf->tex_epoch = texture_epoch(), vf->sampling = use == ViewUse::Sample || use == ViewUse::Storage;
    if (tex_out) *tex_out = tex;
    if (level_out) *level_out = level;
    return view;
}

/// A uniform buffer's descriptor range: the bank uploaded (driver banks, c1) or the user buffer in its pool
struct BankCache {
    const Program* prog = nullptr;
    int stage = -1;
    uint32_t index = 0, size = 0;
    uint64_t at = 0;
    VkBuffer buf = VK_NULL_HANDLE;   // (the stream it is in: a recording may have moved to a bigger one since)
    uint64_t bind_gen = ~0ull;   // (the bindings it was made for: the same and no new grid, the same bank)
    uint32_t grid[3]{};
    std::vector<uint8_t> bytes;
};
TLS_OBJECT(bank_cache, std::array<std::array<BankCache, kNvnStages>, 2>)   // [c0 or c2][NVN stage]: the last upload (draws in a row share most of it)
// (AyyBeeDee20738, PR 252) An indirect dispatch's grid as its program reads it from c0+0..0xc. dispatch() fills it from guest
// memory as the recording saw it, but OoT's foliage culling chains write the next pass's group counts on the GPU in the same
// frame: the program read stale counts (its own work bounds) and culled plants or kept last frame's. grid_patch_want: c0 gets an
// upload of its own, grid_patch_at, into which dispatch() copies the 12 bytes of arguments on the GPU just before the dispatch,
// as the console's hardware sees them
thread_local bool grid_patch_want = false;
thread_local uint64_t grid_patch_at = UINT64_MAX;
thread_local VkBuffer grid_patch_buf = VK_NULL_HANDLE;   // (the stream c0 went to: PR 585's streams can move on)

int desc_reuse_level();   // (below: RX_VK_DESC_REUSE and the Descriptor reuse row)
/// (hookshot) Whether a uniform buffer is copied into the stream as submitted rather than bound where the game keeps it
/// (RX_VK_UB_SNAPSHOT; the reason in uniform_range). On by default on Intel, its iGPUs and its discrete Arc cards alike
bool ub_snapshot_wanted() {
    // (hookshot: and any integrated GPU whose pools have no VRAM mirror, unless RX_VK_UB_SNAPSHOT says: there the GPU reads
    // the game's memory as it runs the frame. On a Steam Deck bound in place, 1 of 3 starts froze in a conversation where 0
    // of ~25 did with the copy, ~1 fps)
    if (dev.policy.set(Sw::ub_snapshot)) return dev.policy.ub_snapshot;   // (device_policy.h: Intel, then the switch)
    return dev.policy.ub_snapshot ||
           (dev.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && !pools_mirrored());
}
/// (hookshot) A draw's uniform copies (uniform_range) are read from guest memory together once its descriptors are built, with
/// pools_mu taken once for all of them (memory_read_as_submitted_many) rather than once a copy: the lock and its release were
/// two thirds of a copy's time on an Arc B580 (HOS_SAMPLE, Castle Town's market: about 10,000 copies a frame, 4 a draw). The
/// stream's bytes they fill are read by the GPU alone, after the recording is submitted. RX_VK_UB_READ_BATCH=0: each read where
/// it is asked, as before; 2 (a test): both, compared, the differences counted in the log
int ub_read_batch_mode() {
    static const int mode = [] {
        const char* e = getenv("RX_VK_UB_READ_BATCH");
        return e && *e ? std::clamp(atoi(e), 0, 2) : 1;
    }();
    return mode;
}
TLS_OBJECT(ub_reads_pending, std::vector<SubmittedRead>)   // (the build's reads not made yet: ub_reads_flush)
void ub_reads_flush() {
    std::vector<SubmittedRead>& p = ub_reads_pending();
    if (p.empty()) return;
    if (ub_read_batch_mode() == 2) {
        static thread_local std::vector<uint8_t> again;
        static std::atomic<int64_t> asked{0}, differ{0};
        for (const SubmittedRead& r : p) {
            again.resize(r.n);
            const SubmittedRead one{r.gpu, again.data(), r.n};
            memory_read_as_submitted_many(&one, 1);
            if (std::memcmp(again.data(), r.dst, r.n)) differ++;
            if (++asked % 200000 == 0)
                logf("vk: ub read batch (check): %lld reads made again at their draw's end, %lld differed", (long long)asked.load(),
                     (long long)differ.load());
        }
    } else {
        memory_read_as_submitted_many(p.data(), p.size());
    }
    p.clear();
}
struct UbReadsScope {   // (build_descriptors: its reads made when it is over, whichever way it ends)
    ~UbReadsScope() { ub_reads_flush(); }
};
/// (hookshot) A recording's uniform copies, by address and size: a draw that binds the range an earlier draw of the recording had
/// copied into the stream, with the same captures in view (memory_capture_view), is given that copy instead of one of its own. In
/// Castle Town's market on an Arc B580 the snapshot made 10,400 copies a frame, 19 MB, of 4,200 distinct ranges, 2.4 MB, and a
/// range's contents were the same in every copy of it within a recording (RX_VK_UB_SNAP_DEDUP=2 in the market: 9.9 million
/// compared, none differed). By size as well as address: an address is bound with different sizes (a program's used size of
/// it). The stream's own updates to a range (ub_overlay) come before this, as before. RX_VK_UB_SNAP_DEDUP=0: a copy for every
/// binding, as before; 2 (a test): a copy for every binding all the same, each compared with the recording's first copy of its
/// range, the differences counted in the log
int ub_snap_dedup_mode() {
    static const int mode = [] {
        const char* e = getenv("RX_VK_UB_SNAP_DEDUP");
        if (e && *e) return std::clamp(atoi(e), 0, 2);
        // (Hookshot's earlier switch for the same reuse: RX_VK_UB_SNAPSHOT_REUSE=0, a copy for every binding)
        const char* old = getenv("RX_VK_UB_SNAPSHOT_REUSE");
        return old && env_value_off(old) ? 0 : 1;
    }();
    return mode;
}
struct UbSnap {
    uint64_t addr = 0, n = 0, view = 0, epoch = 0, at = 0;
    VkBuffer buf = VK_NULL_HANDLE;
    std::vector<uint8_t> bytes;   // (RX_VK_UB_SNAP_DEDUP=2: the first copy's contents)
};
constexpr size_t kUbSnaps = 4096;   // (a thread's: a recording in the market held about 130 distinct ranges)
TLS_OBJECT(ub_snaps, std::vector<UbSnap>)
/// The slot of (addr, n) in the thread's table: its own (hit: this recording's copy, with these captures in view), or one to fill
/// (another recording's, or this one's of the range with other captures in view); nullptr when 16 probes found neither
UbSnap* ub_snap_slot(uint64_t addr, uint64_t n, uint64_t view, bool& hit) {
    hit = false;
    if (!cur) return nullptr;
    std::vector<UbSnap>& snaps = ub_snaps();
    if (snaps.empty()) snaps.resize(kUbSnaps);
    uint64_t h = (addr ^ n * 0x9E3779B97F4A7C15ull) * 0xBF58476D1CE4E5B9ull;
    h ^= h >> 31;
    const uint64_t epoch = cur->ub_snap_epoch;
    for (size_t i = 0; i < 16; i++) {
        UbSnap& e = snaps[(h + i) & (kUbSnaps - 1)];
        if (e.epoch != epoch) return &e;
        if (e.addr == addr && e.n == n) {
            hit = e.view == view;
            return &e;
        }
    }
    return nullptr;
}
void ub_snap_count(bool hit, uint64_t n) {
    static thread_local int64_t asked, given, saved;
    asked++;
    if (hit) given++, saved += (int64_t)n;
    if (asked % 4000000 == 0)
        logf("vk: uniform copies (RX_VK_UB_SNAP_DEDUP=%d): %lld of %lld given the recording's earlier copy of their range, %.1f MB not "
             "copied", ub_snap_dedup_mode(), (long long)given, (long long)asked, saved / 1048576.0);
}
/// (RX_VK_UB_SNAP_DEDUP=2) The copy made now, as with the switch off, and compared with the recording's first of the range
void ub_snap_check(UbSnap* memo, bool hit, uint64_t addr, uint8_t* dst, uint64_t n) {
    static thread_local std::vector<uint8_t> now;
    now.assign(n, 0);
    memory_read_as_submitted(addr, now.data(), n);
    std::memcpy(dst, now.data(), n);   // (the stream is written, never read back: write-combined)
    if (hit && memo && memo->bytes.size() == n) {
        static std::atomic<int64_t> compared{0}, differ{0};
        if (std::memcmp(memo->bytes.data(), now.data(), n)) {
            static std::atomic<int> shown{0};
            if (shown++ < 8)
                logf("vk: uniform copies (check): %#llx +%#llx differs from the recording's first copy of it",
                     (unsigned long long)addr, (unsigned long long)n);
            differ++;
        }
        if (++compared % 100000 == 0)
            logf("vk: uniform copies (check): %lld compared with the recording's first copy of their range, %lld differed",
                 (long long)compared.load(), (long long)differ.load());
    } else if (memo) {
        memo->bytes.assign(now.begin(), now.end());
    }
}
// (Hookshot, HL_UBO_MEMO=1, default off; ubo_memo.h) a uniform slot's answer kept per thread while the pools' stamp, the update list's
// view and the replay segment stand. HL_UBO_MEMO_CHECK=1 (implies the switch): every hit is also worked out and compared (the buffer, the
// offset, the range), the mismatches counted, the first twelve logged, a line every 10 s. Left off where a hit would skip work the
// lookup does: RX_VK_LEDGER's marks, RX_VK_CAP_ONLY_GPU's exclusion notes, the copies as submitted (RX_VK_UB_SNAPSHOT, on by default
// on Intel and integrated GPUs without VRAM mirrors: ub_snapshot_wanted) and RX_VK_UB_CUR_ONLY
int ubo_memo_mode() {
    static const int mode = [] {
        const int v = env_on("HL_UBO_MEMO_CHECK") ? 2 : env_on("HL_UBO_MEMO") ? 1 : 0;
        if (!v) return 0;
        // (RX_VK_CAP_ONLY_GPU is on by default, vk_memory.cpp cap_only_gpu: the memo runs only where it is set to 0)
        const char* cap = getenv("RX_VK_CAP_ONLY_GPU");
        if (ledger_on() || !cap || !env_value_off(cap) || ub_snapshot_wanted() || env_on("RX_VK_UB_CUR_ONLY")) {
            logf("hl ubo memo: left off (RX_VK_LEDGER, RX_VK_CAP_ONLY_GPU, RX_VK_UB_SNAPSHOT or RX_VK_UB_CUR_ONLY is on)");
            return 0;
        }
        logf("hl ubo memo: the uniform slots' answers kept while the pools, the update list and the segment stand%s (HL_UBO_MEMO=1)",
             v == 2 ? ", every hit compared with the full lookup (HL_UBO_MEMO_CHECK=1)" : "");
        return v;
    }();
    return mode;
}
using UboMemoSlots = ubomemo::Entry[kNvnStages][16];
UboMemoSlots& ubo_memo_slots() {
    thread_local UboMemoSlots slots;
    return slots;
}
void ubo_memo_compare(const VkDescriptorBufferInfo& memo, const VkDescriptorBufferInfo& full, uint64_t addr, uint64_t n, int stage,
                      uint32_t index) {
    static std::atomic<uint64_t> compared{0}, bad{0}, last_log{0};
    compared++;
    if (memo.buffer != full.buffer || memo.offset != full.offset || memo.range != full.range) {
        if (bad++ < 12)
            logf("hl ubo memo check: MISMATCH stage %d slot %u addr %#llx +%#llx: memo %p+%#llx (%#llx), full %p+%#llx (%#llx)", stage,
                 index, (unsigned long long)addr, (unsigned long long)n, (void*)memo.buffer, (unsigned long long)memo.offset,
                 (unsigned long long)memo.range, (void*)full.buffer, (unsigned long long)full.offset, (unsigned long long)full.range);
    }
    const uint64_t now = GetTickCount64();
    uint64_t prev = last_log.load();
    if (prev == 0) last_log.compare_exchange_strong(prev, now);
    else if (now - prev >= 10000 && last_log.compare_exchange_strong(prev, now))
        logf("hl ubo memo check: %llu memo hits compared, %llu mismatches", (unsigned long long)compared.load(),
             (unsigned long long)bad.load());
}

VkDescriptorBufferInfo uniform_range(const NvnState& s, const Stage& st, uint32_t index) {
    VkDescriptorBufferInfo bi{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
    const uint64_t align = std::max<uint64_t>(dev.props.limits.minUniformBufferOffsetAlignment, 16);
    uint32_t used = index < Shader::Info::MAX_CBUFS ? st.info.constant_buffer_used_sizes[index] : 0;
    // Global fallbacks read descriptor words added after the IR's constant-bank usage was collected.
    if (st.info.uses_global_memory)
        for (const auto& d : st.info.storage_buffers_descriptors)
            if (d.cbuf_index == index) used = std::max(used, d.cbuf_offset + d.count * 16u);
    if (index == 0 || index == 2) {   // the driver's banks: only what the program reads of them
        FINE_SCOPE(FT_BANK);
        FINE_N(FC_BANK_CALLS, 1);
        const uint32_t n = std::clamp<uint32_t>((used + 15) & ~15u, 16, 0xA00);
        BankCache& c = bank_cache()[index == 2][std::clamp(st.nvn_stage, 0, kNvnStages - 1)];
        if (index == 0 && st.nvn_stage == 5 && grid_patch_want) {   // (PR 252) an upload of its own, patched on the GPU
            const uint64_t at = stream_alloc(n, align);
            if (at == UINT64_MAX) return bi;
            driver_bank_fill(s, 0, 5, cur->mapped + at, n);
            grid_patch_at = at, grid_patch_buf = cur->stream;
            c.at = UINT64_MAX;   // (not shared: its first words change on the GPU)
            return {cur->stream, at, n};
        }
        const bool grid_same = st.nvn_stage != 5 || !std::memcmp(c.grid, s.grid, sizeof c.grid);
        static const bool no_bank_cache = desc_reuse_level() == 0;   // (RX_VK_DESC_REUSE=never: filled every time)
        // (hookshot) a graphics stage's c2 is made from its textures, samplers and images, its c0 from its storage buffers: it stands while
        // those do, whatever else was bound (a uniform buffer for each draw))
        const int bank_stage = std::clamp(st.nvn_stage, 0, kNvnStages - 1);
        const uint64_t gen = bank_stage > 4 ? s.bind_gen : index == 2 ? s.tex_gen[bank_stage] : s.ssbo_gen[bank_stage];
        if (!no_bank_cache && c.at != UINT64_MAX && c.bind_gen == gen && grid_same && c.size >= n) {
            FINE_N(FC_BANK_GEN, 1);
            return {c.buf, c.at, n};
        }
        static thread_local std::vector<uint8_t> tmp;
        tmp.resize(n);
        FINE_N(FC_BANK_FILLS, 1);
        FINE_N(FC_BANK_FILL_BYTES, n);
        driver_bank_fill(s, (int)index, st.nvn_stage, tmp.data(), n);
        static const bool check = env_on("RX_VK_CHECK_BANKS");   // (the fill against the word path)
        if (check)
            for (uint32_t off = 0; off < n; off += 4) {
                const uint32_t w = driver_bank_word(s, (int)index, st.nvn_stage, off);
                static int bad;
                if (std::memcmp(&w, tmp.data() + off, 4) && bad++ < 20)
                    logf("vk: bank c%u stage %d word %#x: fill %#x, word path %#x", index, st.nvn_stage, off,
                         *(const uint32_t*)(tmp.data() + off), w);
            }
        if (c.at != UINT64_MAX && c.size >= n && c.bytes.size() >= n && !std::memcmp(c.bytes.data(), tmp.data(), n) &&
            c.prog != nullptr) {
            c.bind_gen = gen;
            std::memcpy(c.grid, s.grid, sizeof c.grid);
            FINE_N(FC_BANK_SAME, 1);
            return {c.buf, c.at, n};
        }
        const uint64_t at = stream_alloc(n, align);
        if (at == UINT64_MAX) return bi;
        FINE_N(FC_BANK_NEW, 1);
        FINE_N(FC_BANK_NEW_BYTES, n);
        std::memcpy(cur->mapped + at, tmp.data(), n);
        c.bind_gen = gen;
        std::memcpy(c.grid, s.grid, sizeof c.grid);
        c.prog = (const Program*)1, c.at = at, c.buf = cur->stream, c.size = n, c.bytes = tmp;
        return {cur->stream, at, n};
    }
    if (index == 1) {
        FINE_SCOPE(FT_C1);
        FINE_N(FC_C1_CALLS, 1);
        const uintptr_t key = (uintptr_t)&st;
        auto it = cur->c1_at.find(key);
        const uint64_t n = std::max<uint64_t>((st.c1.size() + 15) & ~15ull, 16);
        if (it != cur->c1_at.end()) return {it->second.first, it->second.second, n};
        const uint64_t at = stream_alloc(n, align);
        if (at == UINT64_MAX) return bi;
        std::memset(cur->mapped + at, 0, n);
        std::memcpy(cur->mapped + at, st.c1.data(), st.c1.size());
        cur->c1_at[key] = {cur->stream, at};
        return {cur->stream, at, n};
    }
    if (index >= 3 && index < 3 + 16) {
        FINE_SCOPE(FT_UBO);
        FINE_N(FC_UBO_CALLS, 1);
        const nvn_p_bind_buf& u = s.ubo[std::clamp(st.nvn_stage, 0, kNvnStages - 1)][index - 3];
        if (!u.addr) return bi;
        const uint64_t want = used ? std::max<uint64_t>(used, 16) : std::max<uint64_t>(u.size, 16);
        const uint64_t n = std::min<uint64_t>(std::min<uint64_t>(want, u.size ? std::max<uint64_t>(u.size, 16) : want),
                                              dev.props.limits.maxUniformBufferRange);
        // (Hookshot, HL_UBO_MEMO: the answer of this slot's last build, while nothing that decided it moved: ubo_memo.h. The answer
        // is kept only when it was the pool's buffer itself, which `direct` says below)
        const int memo_stage = std::clamp(st.nvn_stage, 0, kNvnStages - 1);
        const int memo_mode = ubo_memo_mode();
        ubomemo::Entry* memo = nullptr;
        ubomemo::Key memo_key;
        VkDescriptorBufferInfo memo_hit{};
        bool memo_hit_found = false;
        if (memo_mode && u.addr < (1ull << 40) && !s.ubo_bytes[memo_stage][index - 3]) {
            memo = &ubo_memo_slots()[memo_stage][index - 3];
            uint64_t limit = 0;
            memo_key = {u.addr, n, gpu_bind_stamp(), ub_view(&limit), limit, texture_segment()};
            uint64_t a, b;
            if (memo->hit(memo_key, &a, &b)) {
                memo_hit = {(VkBuffer)(uintptr_t)a, b, n};
                if (memo_mode == 1) return memo_hit;
                memo_hit_found = true;   // (HL_UBO_MEMO_CHECK: the answer is worked out too, and compared)
            }
        }
        bool direct = false;   // (the answer below is the pool's buffer, no copy: the one kind the memo keeps)
        const auto full = [&]() -> VkDescriptorBufferInfo {
        const LedgerUse ledger_scope_(kUseUniform);   // (RX_VK_LEDGER: the block's memory, whichever way it reaches the GPU below)
        ledger_mark(u.addr, n, kUseUniform);
        // (Hookshot, HL_UBO_INLINE: a slot bound with its bytes: those, with the stream's updates over them as ub_overlay
        // does, copied into the stream; past the bytes the bind carried, zeros. No guest read, no mirror)
        if (const uint8_t* bytes = s.ubo_bytes[std::clamp(st.nvn_stage, 0, kNvnStages - 1)][index - 3]) {
            const uint64_t have = s.ubo_bytes_n[std::clamp(st.nvn_stage, 0, kNvnStages - 1)][index - 3];
            const uint64_t at = stream_alloc(n, align);
            if (at == UINT64_MAX) return bi;
            uint8_t* p = cur->mapped + at;
            std::memcpy(p, bytes, std::min(n, have));
            if (n > have) std::memset(p + have, 0, n - have);
            ub_overlay_onto(u.addr, n, p);
            FINE_N(FC_UBO_STREAM, 1);
            return {cur->stream, at, n};
        }
        // (updates the stream made to this range in this submission or ones still in flight: a copy with them in)
        // (room first: the overlay copies into the stream's mapping, which a move to a bigger stream in the middle would change)
        if (stream_reserve(n, align))
            if (const uint64_t at = ub_overlay(u.addr, n, cur->mapped, stream_alloc, align); at != UINT64_MAX) {
                FINE_N(FC_UBO_OVERLAY, 1);
                return {cur->stream, at, n};
            }
        // (RX_VK_UB_SNAPSHOT=1: the contents as they are now, in the stream, not the pool the GPU reads later)
        // (on by default on Intel: when the GPU reads the game's memory itself as it runs the frame, the game has by then
        // written the next frame's uniforms over the ones submitted (the per-object matrices are in memory it reuses frame
        // after frame). The depth prepass drew the Kokiri ground with other objects' matrices, out of sight: the depth there
        // stayed far, and the fog pass painted sky over the ground (the white forest on Intel Core Ultra iGPUs, still there
        // after the split-submission fence fix #2419). An Arc B580 (101.9154), whose pools have VRAM mirrors, showed the same
        // (the forest floor and the treehouses gone to sky, white glowing blocks and pots that came and went) for another reason:
        // bind_descriptors' delta push. There a push that carries some of a draw's uniform buffer writes and leaves others out
        // as unchanged draws with wrong data; the snapshot's copies are new at every draw, so every one of them is pushed (with
        // the snapshot off, RX_VK_PUSH_ALL=1 draws it right too: ub_snapshot_wanted). Integrated GPUs without a mirror have it
        // too (the Steam Deck); discrete ones with no mirror (RX_VK_MIRROR=0) keep it off until it is measured there.
        // RX_VK_UB_SNAPSHOT=0: off)
        static const bool snapshot = [] {
            const bool on = ub_snapshot_wanted();
            logf("vk: uniform buffers %s (RX_VK_UB_SNAPSHOT=%d)", on ? "copied as submitted" : "bound in place", on ? 1 : 0);
            return on;
        }();
        // (but a range the GPU writes earlier in this submission, in any of its parts, or in an earlier submission the GPU
        // has not done yet, is bound in place, where the GPU reads it after that write: guest memory and the captures, which
        // the copy is made from, do not have the GPU's bytes yet. Not asked while a parallel replay records segments
        // (RX_VK_PARALLEL), whose writes are not noted: memory_gpu_written)
        const int gpu_written = snapshot && memory_gpu_writes_any() && !par_active.load(std::memory_order_relaxed)
                                    ? memory_gpu_written(u.addr < (1ull << 40) ? u.addr : host->gpu_guest(u.addr), n) : 0;
        BufferRef b = snapshot && !gpu_written ? BufferRef{} : gpu_buffer_bind(u.addr, n, false);
        // (the guard's: after a write in this submission, then in an earlier one; bound in place, copied all the same)
        static std::atomic<int64_t> after_gpu[2][2];
        if (gpu_written) after_gpu[gpu_written - 1][b && !(b.offset & (align - 1)) ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
        if (b && !(b.offset & (align - 1)))   // (a power of two: Vulkan's, and stream_reserve asks it)
            return direct = true, VkDescriptorBufferInfo{b.buffer, b.offset, n};
        // (the sharing and the batched reads are the snapshot's alone: with it off, a range that can't be bound in place, unaligned
        // or outside the pools, is read and copied at its binding as before, and bind_descriptors keeps its delta push)
        static const int dedup = snapshot ? ub_snap_dedup_mode() : 0;   // (the recording's earlier copy of the range: ub_snap_slot)
        bool hit = false;
        const uint64_t view = dedup ? memory_capture_view() : 0;
        UbSnap* memo = dedup ? ub_snap_slot(u.addr, n, view, hit) : nullptr;
        if (dedup) ub_snap_count(hit && dedup == 1, n);
        if (hit && dedup == 1) return {memo->buf, memo->at, n};
        const uint64_t at = stream_alloc(n, align);   // (unaligned or outside the pools: copied into the stream)
        if (at == UINT64_MAX) return bi;
        FINE_N(FC_UBO_STREAM, 1);
        // (as the submission had it, not as the game has rewritten it since: the captures' pages. A range the GPU wrote that
        // can't be bound in place, unaligned or outside the pools, is copied all the same, without that write, as before)
        static const int batch = snapshot ? ub_read_batch_mode() : 0;
        if (dedup == 2) {
            ub_snap_check(memo, hit, u.addr, cur->mapped + at, n);
        } else {
            if (batch != 1) memory_read_as_submitted(u.addr, cur->mapped + at, n);
            if (batch != 0) ub_reads_pending().push_back({u.addr, cur->mapped + at, n});   // (read at the build's end: UbReadsScope)
        }
        if (memo && !hit)
            memo->addr = u.addr, memo->n = n, memo->view = view, memo->epoch = cur->ub_snap_epoch, memo->at = at, memo->buf = cur->stream;
        static std::atomic<int64_t> copied{0};   // (the recording threads' together)
        if (copied.fetch_add(1, std::memory_order_relaxed) % 5000 == 0)
            logf("vk: uniform buffer %#llx +%#llx copied into the stream (%s; %lld so; after a GPU write in the submission: "
                 "%lld bound in place, %lld copied all the same; after one in a submission still on the GPU: %lld bound in "
                 "place, %lld copied all the same)", (unsigned long long)u.addr, (unsigned long long)n,
                 b ? "unaligned" : snapshot && !gpu_written ? "a snapshot" : "outside the pools", (long long)copied.load(std::memory_order_relaxed),
                 (long long)after_gpu[0][0].load(std::memory_order_relaxed), (long long)after_gpu[0][1].load(std::memory_order_relaxed),
                 (long long)after_gpu[1][0].load(std::memory_order_relaxed), (long long)after_gpu[1][1].load(std::memory_order_relaxed));
        return {cur->stream, at, n};
        };
        const VkDescriptorBufferInfo r = full();
        if (memo) {
            if (memo_hit_found) ubo_memo_compare(memo_hit, r, u.addr, n, memo_stage, index);
            // (the key as read before the answer: a list that moved meanwhile, or a stamp the answer itself moved, is not kept)
            uint64_t limit = 0;
            const uint64_t epoch_after = ub_view(&limit);
            if (direct && ubomemo::storable(memo_key.epoch, epoch_after) && limit == memo_key.limit)
                memo->store({u.addr, n, gpu_bind_stamp(), memo_key.epoch, memo_key.limit, memo_key.seg}, (uint64_t)(uintptr_t)r.buffer,
                            r.offset);
            else
                memo->drop();
        }
        return r;
    }
    return bi;
}

struct DescWriter {
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufs;
    std::vector<VkDescriptorImageInfo> imgs;
    std::vector<VkBufferView> views;
    const Program* prog = nullptr;   // (the program they were built for: a dispatch builds its own over a draw's)
    void reset() {
        writes.clear(), bufs.clear(), imgs.clear(), views.clear();
    }
};
TLS_OBJECT(dw, DescWriter)
TLS_OBJECT(draw_sampled, std::vector<const Texture*>)   // (the textures the descriptors being built sample or store)
TLS_OBJECT(gfx_sampled, std::vector<const Texture*>)    // (the last draw's: kept while its descriptors are reused)

/// Fills dw with the program's descriptors for the bound state (bindings in Eden's order). Storage the program
/// writes is noted: textures it stores into are GPU-written, buffers it stores into are stamped (vk_memory.cpp).
struct DumpWritten { Texture* tex; uint32_t level; };
TLS_OBJECT(dump_written, std::vector<DumpWritten>)
// (RX_VK_DUMP_SAMPLED=prog[,prog...]: in a dump, the textures those programs sample too, each once, 96 at most)
TLS_OBJECT(dump_sampled, std::vector<Texture*>)
bool dump_sampled_wanted(uint64_t key) {
    static const std::vector<uint64_t> want = [] {
        std::vector<uint64_t> v;
        if (const char* e = getenv("RX_VK_DUMP_SAMPLED"))
            for (const char* q = e; *q;) {
                char* end;
                const uint64_t k = strtoull(q, &end, 16);
                if (end == q) break;
                v.push_back(k);
                q = *end ? end + 1 : end;
            }
        return v;
    }();
    return std::find(want.begin(), want.end(), key) != want.end();
}   // (RX_VK_DUMP: the storage images the descriptors let the program write)
TLS_OBJECT(dump_inputs, std::string)               // (and what it samples)
int64_t st_dummy_textures;               // (textures sampled as the dummy: no image for the handle)

void dump_input(char kind, uint32_t index, uint32_t handle, const Texture* tex) {
    char b[160];
    if (!tex)
        snprintf(b, sizeof b, " %c%u:none(%#x)", kind, index, handle);
    else
    {
        nvn_tex t;
        nvn_view v;
        bool has_view = false;
        const int nf = texture_of((int)(handle & 0xFFFFF), &t, &v, &has_view)
                           ? (has_view && v.set_format ? (int)v.format : (int)t.d.format) : -1;
        snprintf(b, sizeof b, " %c%u:#%u:%llx/f%d/n%#x/%ux%ux%u%s%s", kind, index, tex->id, (unsigned long long)tex->gpu,
                 (int)tex->format, nf, tex->width, tex->height, tex->type == VK_IMAGE_TYPE_3D ? tex->depth : tex->layers,
                 tex->gpu_written ? "/G" : "", tex->uploaded ? "/U" : "");
    }
    dump_inputs() += b;
}

/// A descriptor over 2 ms is logged, 40 times, and only while RX_VK_SLOW_DESC is set. Off, the clock is not read.
bool slow_desc_timing() {
    static const int on = env_on("RX_VK_SLOW_DESC");
    return on;
}

// (hookshot) one switch read two ways, like the Robust buffers row of vk_main.cpp: the environment variable says it for test
// runs (any value, 0 off); else the row's line in hookshot_options.ini in the user folder; else off.
/// (the row's value in hookshot_options.ini, read before the menu has loaded it; the environment's switch first. def: no
/// file, no line, or a file older than options_version since (one whose line was only ever the old default, vk_menu.cpp
/// load))
bool menu_env_on(const char* env_name, const char* ini_key_name, bool def = false, int since = 0) {
    if (const char* e = getenv(env_name)) return !env_value_off(e);   // (hookshot_env.h: the switches' one rule)
    const std::string ini = options_path();   // (vk_paths.h)
    const size_t klen = strlen(ini_key_name);
    int v = -1, version = 0;
    if (FILE* f = fopen(ini.c_str(), "r")) {
        char line[80];
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, ini_key_name, klen) && line[klen] == '=') v = atoi(line + klen + 1);
            if (!strncmp(line, "options_version=", 16)) version = atoi(line + 16);
        }
        fclose(f);
    }
    if (v < 0 || version < since) return def;
    return v != 0;
}
/// (hookshot) How far a draw keeps the descriptors (and the banks) of the draw before it, one switch for the three it had:
///   2  the Descriptor reuse row's On (the default since options_version 4): kept through registrations of the texture slots
///      it does not sample. What it does sample is checked by slot version, and the samplers by their registrations'
///      generation, instead of the global one every registration moves
///   1  the row's Off: kept while no registration at all was made since
///   0  never: every draw builds its own and fills its banks (an A/B against the reuse itself)
/// RX_VK_DESC_REUSE goes over the row for a run, with the row's values: 0 (or off) its Off, 1 (or on) its On; never: level 0.
/// (It replaces RX_VK_DESC_SLOTGEN, same values, RX_VK_NO_DESC_REUSE=1, now never, and RX_VK_DESC_REUSE_DGEN, which turned
/// off the texture-destroy check of issue 1126 and is gone: host/hos_main.c names the replacement when one is still set)
/// The row applies in play (menu_debug_rows): a draw keeps descriptors only from a build made at the same level, desc_fp_on.
/// RX_VK_DESC_REUSE holds the level for the run
std::atomic<int> desc_reuse_row{-1}, sampler_cache_row{-1};   // (the rows as the menu has them now; -1 before it has loaded)
} // namespace
void menu_debug_rows(int descriptor_reuse, int sampler_cache) {   // (vk_draw.h: the menu, when it loads and when either row changes)
    desc_reuse_row.store(descriptor_reuse != 0, std::memory_order_relaxed);
    sampler_cache_row.store(sampler_cache != 0, std::memory_order_relaxed);
}
namespace {
int desc_reuse_level() {
    static int start = -1;
    static bool env = false;
    if (start < 0) {
        const char* e = getenv("RX_VK_DESC_REUSE");
        env = e != nullptr;
        start = e && !strcmp(e, "never") ? 0 : menu_env_on("RX_VK_DESC_REUSE", "descriptor_reuse", true, 4) ? 2 : 1;
        if (start == 2)
            logf("vk: descriptor slotgen ready: draws keep their descriptors through the registrations of slots they do not "
                 "sample (menu: Descriptor reuse)");
        if (start == 0) logf("vk: RX_VK_DESC_REUSE=never: every draw builds its own descriptors");
    }
    const int row = env ? -1 : desc_reuse_row.load(std::memory_order_relaxed);
    const int level = row < 0 ? start : row ? 2 : 1;
    static std::atomic<int> last{-1};
    if (last.load(std::memory_order_relaxed) != level && last.exchange(level, std::memory_order_relaxed) >= 0)
        logf("vk: Descriptor reuse %s (menu)", level == 2 ? "on: draws keep their descriptors through other slots' registrations"
                                                          : "off: kept only while no registration at all was made");
    return level;
}
bool desc_slotgen_on() { return desc_reuse_level() == 2; }
/// (hookshot) the Sampler cache row / RX_VK_SAMPLER_CACHE=1: what a sampler id holds is kept next to its version on the replay
/// thread, so resolving an id that was not registered anew takes no lock (the registrations' mutex contends with the texture streaming)
/// The row applies in play (menu_debug_rows): what is kept is checked by the slot's version, so entries from before it was
/// turned off and on again are still good while the slot was not registered anew. RX_VK_SAMPLER_CACHE holds it for the run
bool sampler_cache_on() {
    static int start = -1;
    static bool env = false;
    if (start < 0) {
        env = getenv("RX_VK_SAMPLER_CACHE") != nullptr;
        start = menu_env_on("RX_VK_SAMPLER_CACHE", "sampler_cache");
        if (start) logf("vk: sampler cache ready: sampler ids answer their parameters without the registrations' lock (menu: Sampler cache)");
    }
    const int row = env ? -1 : sampler_cache_row.load(std::memory_order_relaxed);
    return row < 0 ? start != 0 : row != 0;
}
// (Descriptor reuse) What the descriptors being built read of the registrations: the version of every slot whose handle they
// resolved. The next draw of the same bindings keeps them while none of those versions moved. A handle read as 0 is noted: of a
// slot that is not registered, only a registration (the global generation) can tell us that it is one now.
TLS_OBJECT(desc_fp, std::vector<uint64_t>)   // ((version << 20 | slot), a handle read)
thread_local bool desc_fp_dummy = false;      // (some handle read 0)
thread_local bool desc_fp_on = false;         // (the build being made records its handles: Descriptor reuse on when it began)
thread_local uint64_t desc_samp_seen = 0;     // (the samplers' generation when the build began)
thread_local uint64_t desc_destroy_seen = 0;  // (the eviction generation: destroy_later retires a view without touching any slot or sampler generation)
void desc_fp_note(uint32_t handle) {
    if (!desc_fp_on) return;
    const uint32_t slot = handle & 0xFFFFF;
    desc_fp().push_back((uint64_t)texture_slot_version((int)slot) << 20 | slot);
    desc_fp_dummy |= handle == 0;
}

/// (a descriptor that took over 2 ms to resolve: logged, 40 of them)
void slow_descriptor(std::chrono::steady_clock::time_point t0, const char* what, uint64_t addr, uint64_t size) {
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static int logged;
    if (ms >= 2.0 && logged++ < 40)
        logf("vk: slow descriptor %.1f ms: %s %#llx size %#llx", ms, what, (unsigned long long)addr, (unsigned long long)size);
}

// ---- RX_VK_WATCH_PROG: a program's bindings, the last 40 kept and logged when the device is lost ----
bool watched(uint64_t key) {   // (RX_VK_WATCH_PROG=<key>[,<key>...])
    static const std::vector<uint64_t> keys = [] {
        std::vector<uint64_t> v;
        for (const char* q = getenv("RX_VK_WATCH_PROG"); q && *q;) {
            v.push_back(strtoull(q, nullptr, 16));
            while (*q && *q != ',') q++;
            if (*q) q++;
        }
        return v;
    }();
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}
uint64_t watch_prog() {
    static const uint64_t w = getenv("RX_VK_WATCH_PROG") ? strtoull(getenv("RX_VK_WATCH_PROG"), nullptr, 16) : 0;
    return w;
}
std::mutex watch_mu;
std::string watch_ring[40];
int watch_at = 0;
void watch_note(const char* fmt, ...) {
    char line[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    std::lock_guard lk(watch_mu);
    watch_ring[watch_at++ % 40] = line;
}

/// (RX_VK_DUMP_SSBO=<prog>[,...]: in a dump, those programs' storage buffers, in guest memory and as the GPU has them)
bool dump_ssbo_wanted(uint64_t key) {
    static const std::vector<uint64_t> want = [] {
        std::vector<uint64_t> v;
        for (const char* q = getenv("RX_VK_DUMP_SSBO"); q && *q;) {
            char* end;
            const uint64_t k = strtoull(q, &end, 16);
            if (end == q) break;
            v.push_back(k);
            q = *end ? end + 1 : end;
        }
        return v;
    }();
    return std::find(want.begin(), want.end(), key) != want.end();
}
/// (and the GPU's copies of them, before or after the dispatch)
void dump_ssbos(VkCommandBuffer cmd, const Program& p, const char* when) {
    if (!dump_on() || !dump_ssbo_wanted(p.key)) return;
    uint32_t sb = 0;
    for (const VkWriteDescriptorSet& w : dw().writes) {
        if (w.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || !w.pBufferInfo) continue;
        for (uint32_t i = 0; i < w.descriptorCount; i++, sb++) {
            const VkDescriptorBufferInfo& b = w.pBufferInfo[i];
            if (!b.buffer || b.range == VK_WHOLE_SIZE) continue;
            char what[128];
            snprintf(what, sizeof what, "%03d prog %016llx sb%u %s", dump_seq(), (unsigned long long)p.key, sb, when);
            dump_buffer(cmd, b.buffer, b.offset, b.range, what);
        }
    }
}

// A program's texture descriptors as this thread last resolved them: a draw whose handle for a slot is the one before, with
// the texture slot, the sampler slot and the texture set unchanged since (their versions and the destroy generation), takes the view,
// the texture and the sampler from here instead of handle_hits' search and sampler_for's (which takes reg_mu, shared by all the
// recording threads, and hashes the sampler's parameters). The handle itself is read again every draw (from the driver bank, a
// uniform buffer or a handle program): only its resolution to Vulkan objects is kept, for 4 handles a slot. Per thread (parallel recording), no lock.
// RX_VK_NO_SLOT_CACHE=1: every slot resolved as before
struct TexSlot {
    uint32_t handle = 0, version = 0, samp_version = 0, tex_epoch = 0;
    uint64_t destroy_gen = ~0ull;
    VkImageView view = VK_NULL_HANDLE;
    Texture* tex = nullptr;
    VkSampler sampler = VK_NULL_HANDLE;
    bool valid = false;
};
constexpr size_t kTexWays = 4;   // (the handles kept a slot)
struct ProgSlots {
    uint64_t key = 0;
    std::vector<TexSlot> tex;
};
TLS_OBJECT(prog_slots, std::unordered_map<const Program*, ProgSlots>)

// (hookshot, issue 14, RX_VK_NULL_WRITE) A storage buffer a program writes through whose address resolved to nothing: the descriptor was
// a null one (Vulkan drops its writes). RADV hangs the graphics ring on a submission that writes at a tiny GPU address (0 or 0x5000:
// "GPUVM fault", a texture-unit write, Steam Deck and RX 9070 XT), where NVIDIA's driver drops such a write silently. Counted by why
// it resolved to nothing, one log line for each program and descriptor. With the switch on, a buffer of our own is bound there
// instead of the null descriptor, so every write of a draw or dispatch lands in memory (or past its 64 bytes, where the robust access
// drops it) and no descriptor has base address 0. 64 bytes only: a program's search of its bindings takes the binding's length as the
// end of the range it serves, and a longer one would take over reads the page table serves better.
// RX_VK_NULL_WRITE=1: the scratch buffer; unset or 0: the null descriptor as before (the cases still counted and logged)
enum NullWriteWhy { kNwAddressZero, kNwNotBuffer, kNwNotBound, kNwWhys };
std::atomic<int64_t> null_writes[kNwWhys];
constexpr uint64_t kNullScratchBytes = 64;
VkBuffer null_scratch = VK_NULL_HANDLE;
bool null_write_on() {
    // (off by default for now: a slot a program also reads would read the scratch's last writes instead of the null
    // descriptor's zeros, and it is not shown yet to cure the RADV hang; RX_VK_NULL_WRITE=1 for the Deck's test)
    static const bool on = getenv("RX_VK_NULL_WRITE") && atoi(getenv("RX_VK_NULL_WRITE")) != 0;
    return on;
}
/// The scratch buffer (made on first use, host-visible and zeroed, never read back), or no buffer if it could not be made
VkDescriptorBufferInfo null_write_binding() {
    static std::once_flag once;
    static VmaAllocation mem = nullptr;
    std::call_once(once, [] {
        void* mapped = nullptr;
        if (!create_buffer(kNullScratchBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, null_scratch, mem, &mapped) || !mapped)
            null_scratch = VK_NULL_HANDLE;
        else std::memset(mapped, 0, kNullScratchBytes);
    });
    return {null_scratch, 0, kNullScratchBytes};
}
void null_write_note(const Program& p, int stage, uint32_t cbuf, uint32_t off, uint32_t k, uint32_t count, uint64_t addr, uint64_t size,
                     NullWriteWhy why) {
    null_writes[why]++;
    static std::mutex mu;
    static std::unordered_set<uint64_t> told;
    std::lock_guard lk(mu);
    if (!told.insert(p.key ^ ((uint64_t)off << 40) ^ ((uint64_t)cbuf << 32) ^ ((uint64_t)k << 24) ^ ((uint64_t)stage << 56)).second ||
        told.size() > 200)
        return;
    static const char* const names[kNwWhys] = {"its address is 0 (the slot was never set)",
                                               "its address and size do not name a buffer (not guest memory, or a table entry past its table)",
                                               "its pool has no buffer for it yet (not imported, or the range does not fit)"};
    logf("vk: written storage buffer resolves to nothing: program %016llx stage %d ssbo %u of %u (c%u+%#x) addr %#llx size %#llx: %s; %s",
         (unsigned long long)p.key, stage, k, count, cbuf, off, (unsigned long long)addr, (unsigned long long)size, names[why],
         null_write_on() ? "a scratch buffer is bound" : "left null");
}

// (hookshot, a diagnosis) RX_VK_NULL_TEX=<kinds>: the textures of those kinds that draws sample read as the
// null texture (zeros; the dummy where the device has no null descriptors) instead of their images. Kinds, comma-separated: 3d,
// cube, shadow (a depth compare), 2d, array, all. Dispatches keep theirs: what the compute passes produce stays as it is. On
// RTX 50 cards the GPU hangs in draws that read the 128 x 72 x 96 volumes, a cube and the full-size depth; if an image of one
// kind is what hangs it, the freeze stops when that kind is left out (the picture is wrong then: only a test)
int null_tex_kinds() {
    static const int kinds = [] {
        const char* e = getenv("RX_VK_NULL_TEX");
        if (!e || !*e) return 0;
        int k = 0;
        static const struct { const char* name; int bit; } names[] = {
            {"3d", 1}, {"cube", 2}, {"shadow", 4}, {"2d", 8}, {"array", 16}, {"all", 31}};
        for (const char* q = e; *q;) {
            const char* end = q;
            while (*end && *end != ',') end++;
            for (const auto& n : names)
                if ((size_t)(end - q) == strlen(n.name) && !strncmp(q, n.name, end - q)) k |= n.bit;
            q = *end ? end + 1 : end;
        }
        logf("vk: RX_VK_NULL_TEX=%s: draws sample the null texture in place of%s%s%s%s%s (a diagnosis: the picture will be wrong)",
             e, k & 1 ? " 3D textures" : "", k & 2 ? " cubes" : "", k & 4 ? " depth compares" : "", k & 8 ? " 2D textures" : "",
             k & 16 ? " arrays" : "");
        return k;
    }();
    return kinds;
}
bool null_tex_wanted(const Stage& st, const Shader::TextureDescriptor& d) {
    const int kinds = null_tex_kinds();
    if (!kinds || st.vk_stage == VK_SHADER_STAGE_COMPUTE_BIT) return false;
    if (d.is_depth) return (kinds & 4) != 0;
    switch (d.type) {
    case Shader::TextureType::Color3D: return (kinds & 1) != 0;
    case Shader::TextureType::ColorCube: case Shader::TextureType::ColorArrayCube: return (kinds & 2) != 0;
    case Shader::TextureType::ColorArray1D: case Shader::TextureType::ColorArray2D: return (kinds & 16) != 0;
    default: return (kinds & 8) != 0;
    }
}
// Project ReLink 54af18c identified the two exposure accumulation programs.
bool exposure_meter_program(uint64_t key) {
    return key == 0x435a0138b9013040ULL || key == 0x592079442d7129daULL;
}
thread_local BufferRef exposure_meter_target;
thread_local uint32_t exposure_meter_bindings = 0;

void build_descriptors(const NvnState& s, const Program& p) {
    static const bool exposure_part = env_on("RX_VK_EXPOSURE_METER_PART");
    const bool track_exposure = exposure_part && p.compute && exposure_meter_program(p.key) && readback_early_on();
    if (track_exposure) exposure_meter_target = {}, exposure_meter_bindings = 0;
#ifdef TRACY_ENABLE
    HT_SCOPE_DRAW("build_descriptors");
#endif
    // (hookshot) the thread's writer asked for once: each dw() is a thread-local lookup, a call in a shared library, and a
    // build made dozens of them
    const UbReadsScope ub_reads;   // (the uniform copies' reads made together when the build is over: ub_read_batch_mode)
    DescWriter& out = dw();
    out.reset();
    out.prog = &p;
    desc_fp().clear(), desc_fp_dummy = false;   // (Descriptor reuse: the samplers' generation is read before the first handle,
    desc_fp_on = desc_slotgen_on();             // (the row read once for the whole build: it changes in play)
    desc_samp_seen = sampler_registrations_generation();   // (so one registered during the build still forces the next rebuild)
    desc_destroy_seen = texture_destroy_generation();      // (an eviction retires views behind the slots' backs: it bumps only this)
    draw_sampled().clear();
    dump_written().clear();
    dump_inputs().clear();
    // (hookshot) how many of each the program has: its own, the same at every build. Counted once and kept with the program
    // (desc_counts): every build walked six lists a stage only for these three numbers
    size_t nbuf = 0, nimg = 0, nview = 0, ntex = 0;
    const uint64_t kept = p.desc_counts.load(std::memory_order_relaxed);
    if (kept != ~0ull) {
        desc_counts_unpack(kept, &nbuf, &nimg, &nview);
    } else {
        for (const Stage& st : p.stages) {
            for (const auto& d : st.info.constant_buffer_descriptors) nbuf += d.count;
            for (const auto& d : st.info.storage_buffers_descriptors) nbuf += d.count;
            for (const auto& d : st.info.texture_buffer_descriptors) nview += d.count;
            for (const auto& d : st.info.image_buffer_descriptors) nview += d.count;
        for (const auto& d : st.info.texture_descriptors) nimg += d.count, ntex += d.count;
            for (const auto& d : st.info.image_descriptors) nimg += d.count;
        }
        // (kept only once the program is whole: its stages are filled in by a worker before `ready`)
        if (p.ready.load(std::memory_order_acquire)) {
            const uint64_t packed = desc_counts_pack(nbuf, nimg, nview);
            if (packed != ~0ull) p.desc_counts.store(packed, std::memory_order_relaxed);
        }
    }
    out.bufs.reserve(nbuf), out.imgs.reserve(nimg), out.views.reserve(nview);   // (stable pointers below)
    static const bool no_slot_cache = env_on("RX_VK_NO_SLOT_CACHE");
    TexSlot* slots = nullptr;
    if (!no_slot_cache && ntex) {
        auto& table = prog_slots();
        ProgSlots& ps = table[&p];
        if (ps.key != p.key || ps.tex.size() != ntex * kTexWays) ps.key = p.key, ps.tex.assign(ntex * kTexWays, TexSlot{});
        slots = ps.tex.data();
    }
    size_t tslot = 0;
    const uint64_t dgen_now = slots ? texture_destroy_generation() : 0;
    FINE_N(FC_DESCS, (int64_t)(nbuf + nimg + nview));
    uint32_t binding = 0;
    auto write = [&](VkDescriptorType type, uint32_t count) {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstBinding = binding++;
        w.descriptorCount = count;
        w.descriptorType = type;
        return w;
    };
    const uint64_t salign = storage_binding_align();   // (the translator's: vk_internal.h)
    for (const Stage& st : p.stages) {
        const Shader::Info& info = st.info;
        const int ns = st.nvn_stage;
        for (const auto& d : info.constant_buffer_descriptors) {
            VkWriteDescriptorSet w = write(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, d.count);
            const size_t first = out.bufs.size();
            for (uint32_t k = 0; k < d.count; k++) {
                const auto t0 = slow_desc_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                out.bufs.push_back(uniform_range(s, st, d.index + k));
                if (slow_desc_timing()) slow_descriptor(t0, "uniform buffer", d.index + k, out.bufs.back().range);
            }
            w.pBufferInfo = &out.bufs[first];
            out.writes.push_back(w);
        }
        for (const auto& d : info.storage_buffers_descriptors) {
            FINE_SCOPE(FT_STORAGE);
            FINE_N(FC_SSBO_DESCS, d.count);
            VkWriteDescriptorSet w = write(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d.count);
            const size_t first = out.bufs.size();
            for (uint32_t k = 0; k < d.count; k++) {
                const uint32_t off = d.cbuf_offset + k * 16;
                if (d.count == 1 && d.cbuf_index == 0 && d.cbuf_offset == 0xff0) {   // (EvilOOT, RX_VK_MISS_LOG)
                    out.bufs.push_back(miss_binding(p.key));
                    continue;
                }
                // (AyyBeeDee20738, PR 138; EvilOOT the same before it, for OoT's culling tables in c4) through the stage's banks as
                // its program reads them (c1 from the program, c3+ from the bound uniform buffer with the stream's inline updates):
                // a storage buffer named there read 0 from the driver banks and was bound to nothing
                const uint64_t addr = stage_cbuf_word(s, st, d.cbuf_index, off) |
                                      (uint64_t)stage_cbuf_word(s, st, d.cbuf_index, off + 4) << 32;
                uint64_t size = stage_cbuf_word(s, st, d.cbuf_index, off + 8);
                // (EvilOOT) a table entry that is a bare pointer (its size word 0, as OoT's culling setup keeps them): bound up to
                // 1 MB of its pool, as the mesh pointer table's regions are; the shader takes the binding's length as its end.
                // Not marked written: the 1 MB is a guess past the real buffer, and its copy back to guest memory overwrote
                // the CPU's own data beside it (a pool header's pointer, the game crashed). What the GPU writes there (the
                // culling passes' group counts) is read by later GPU work, which binds the same mirror.
                bool bare = false;
                // (Fritchii 8193f926f, jpkto, Lichii #351) A read-only one is the base of an object pool the shader reads
                // anywhere in: bound to the pool's end, as far as a storage buffer reaches (at most 1 GB). OoT's froxel fog
                // finds its light probes this way, up to ~130 MB past the pointer: the probes past the binding read 0 and the
                // fog took an overbright constant from them (the white haze, zones overexposed or too dark). Written ones keep
                // 1 MB. RX_VK_BARE_READ_MB=<MB> caps it (1: as before)
                static const uint64_t bare_read = [] {
                    const uint64_t reach = std::min<uint64_t>(dev.props.limits.maxStorageBufferRange, 1ull << 30);
                    const uint64_t cap = getenv("RX_VK_BARE_READ_MB") ? (uint64_t)atoll(getenv("RX_VK_BARE_READ_MB")) << 20 : reach;
                    return std::max<uint64_t>(1ull << 20, std::min(cap, reach - std::min<uint64_t>(reach, 64ull << 10)));
                }();
                if (d.count > 1 && !size && addr && !(addr & 3) && host->gpu_guest(addr))
                    if (const uint64_t end = gpu_pool_end(addr); end > addr)
                        size = std::min<uint64_t>(end - addr, d.is_written ? (1ull << 20) : bare_read) & ~3ull, bare = true;
                // (EvilOOT) the pointer table's regions (c0+0x880, vk_bank.cpp pointer_table) are guesses too: GPU-only, as above
                if (d.cbuf_index == 0 && d.cbuf_offset == 0x880 && d.count == 3) bare = true;
                // (Fritchii's storage buffer pass, Info::native_global_bounds) a program whose accesses pick among candidate
                // pointers at run time: those are bare 8-byte arguments, the word after one another argument (a stride, a
                // handle), not an NVN size. Unless that word looks like this pointer's size (its high half 0, inside the
                // pool), bound to the pool's end, 8 MB at most (Fritchii's limit), 1 MB and GPU-only if written, as the
                // bare table entries above; the shaders take the binding's length as the end
                if (info.native_global_bounds && d.count == 1 && d.cbuf_index != 0 && addr && !(addr & 3) &&
                    host->gpu_guest(addr)) {
                    const uint64_t end = gpu_pool_end(addr);
                    const bool sized = size && !stage_cbuf_word(s, st, d.cbuf_index, off + 12) && end > addr && size <= end - addr;
                    if (!sized && end > addr)
                        size = std::min<uint64_t>(end - addr, d.is_written ? (1ull << 20) : (8ull << 20)) & ~3ull, bare = true;
                }
                if (d.is_written) memory_writer_key(p.key);   // (RX_VK_LOG_RB_WRITERS: the program a written page is from)
                const uint64_t base = addr & ~(salign - 1);
                uint64_t n = std::max<uint64_t>(((size + (addr - base)) + 3) & ~3ull, 16);
                {   // (hookshot, RX_VK_SSBO_STRETCH=1: a read-only buffer bound to its pool's end, up to 64 MB. Lichii bf3f395c9: the grass's
                    // task and mesh programs read records 12-18 MB past a base whose size word said less, read 0 and culled every blade)
                    static const bool stretch = env_on("RX_VK_SSBO_STRETCH");
                    if (stretch && !d.is_written && addr && d.count == 1)
                        if (const uint64_t end = gpu_pool_end(base); end > base + n)
                            n = std::max<uint64_t>(n, std::min<uint64_t>(end - base, 64ull << 20) & ~uint64_t{3});
                }
                // (EvilOOT) RX_VK_SSBO_WIDE=<MB>, experiment: single storage buffers bound from their base to the end of their
                // pool (that many MB at most), so that a pointer loaded through the buffer and used with the buffer's binding
                // (the pointer pass's split accesses) reads its target instead of 0 past the descriptor's size. Only the
                // descriptor's own range is marked written (copied back to guest memory): the rest stays the GPU's.
                uint64_t wide_n = 0;
                {
                    static const uint64_t wide = getenv("RX_VK_SSBO_WIDE") ? (uint64_t)atoll(getenv("RX_VK_SSBO_WIDE")) << 20 : 0;
                    if (wide && d.count == 1 && addr)
                        if (const uint64_t end = gpu_pool_end(base); end > base + n) wide_n = std::min(end - base, wide) & ~3ull;
                }
                const auto t0 = slow_desc_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const LedgerUse ledger_scope_(kUseStorage);   // (RX_VK_LEDGER)
                // (hookshot) a storage buffer table's entries (count > 1, Info::global_tables): the ones past the table's
                // end are whatever the constant buffer holds there, bound only when they look like a buffer
                const bool plausible = d.count == 1 || (size && size <= (256ull << 20) && host->gpu_guest(base));
                constexpr uint64_t exposure_bytes = 0x1040;
                const bool meter = track_exposure && addr && addr == base && size == exposure_bytes &&
                                   d.count == 1 && d.is_written && !bare && !wide_n;
                if (meter)
                    if (const uint64_t guest = host->gpu_guest(addr))
                        memory_rb_waited(guest, exposure_bytes);  // prime the existing exact writer-part tracking
                BufferRef b = addr && plausible ? gpu_buffer_bind(base, n, d.is_written && !bare) : BufferRef{};
                if (meter && b && b.size >= exposure_bytes) {
                    exposure_meter_target = b;
                    ++exposure_meter_bindings;
                }
                // (AyyBeeDee20738, PR 138) a table entry's region that does not fit its pool this moment bound nothing, and the next
                // frame bound it again: its reads flipped between the data and 0, the picture between bright and dark. Smaller,
                // down to 1 MB, rather than nothing (and past what is bound the shaders' search now goes on to the page table)
                if (!b.buffer && d.count > 1 && addr && plausible) {
                    uint64_t m = n;
                    if (const uint64_t end = gpu_pool_end(base); end > base) m = std::min<uint64_t>(m, end - base);
                    for (; !b.buffer && m >= (1ull << 20); m = (m / 2) & ~3ull) {
                        b = gpu_buffer_bind(base, m, d.is_written && !bare);
                        if (b.buffer) {   // (EvilOOT: the descriptor's range what was bound, so the search ends there)
                            n = m;
                            break;
                        }
                    }
                }
                // (EvilOOT) a bare entry the program writes through: its first 64 bytes marked written (the same mirror), so
                // the counts and records the culling setup writes there survive the mirror's syncs from guest memory and go
                // back to it, as on the console; the rest of the guessed 1 MB stays the GPU's (copied back whole, it
                // overwrote the CPU's own data beside it)
                if (b && bare && d.is_written && d.cbuf_offset != 0x880) {
                    const uint64_t head = std::min<uint64_t>(n, (addr - base) + 0x40);
                    (void)gpu_buffer_bind(base, (head + 3) & ~3ull, true);
                }
                // (RX_VK_PROTECT_BARE=1: the whole range kept for the GPU against the CPU's page syncs, vk_memory.cpp prot_pages)
                if (b && bare && d.is_written)
                    if (const uint64_t guest = host->gpu_guest(base)) memory_protect_gpu(guest, n);
                if (b && wide_n) {   // (the same buffer, its binding longer, nothing more marked written)
                    const BufferRef w = gpu_buffer_bind(base, wide_n, false);
                    if (w && w.buffer == b.buffer && w.offset == b.offset) b = w, n = wide_n;
                }
                // (hookshot) all a program may write through it, past what is marked written (a bare entry's guessed range, the
                // pointer table's, a wide binding's): not copied back, but a GPU write hazard for the readback refresh of a later
                // recording in the submission (vk_memory.cpp memory_rb_refresh)
                if (b && d.is_written && (bare || wide_n)) memory_note_gpu_write(base, n);
                {   // (EvilOOT) the wanted programs' storage descriptors in order, once: descriptor n = the IR's "#n"
                    static std::mutex lm;
                    static std::unordered_set<uint64_t> told;
                    static const bool stats_env = getenv("RX_VK_DRAW_STATS") != nullptr;
                    if (stats_env && stats_prog_wanted(p.key)) {
                        const size_t n_desc = (size_t)(&d - info.storage_buffers_descriptors.data());
                        std::lock_guard lk(lm);
                        const uint64_t tk = p.key ^ ((uint64_t)ns << 56) ^ ((uint64_t)n_desc << 48) ^ ((uint64_t)k << 40);
                        if (told.insert(tk).second)
                            logf("vk: descriptors of %016llx stage %d: #%zu[%u] c%u+%#x%s -> %#llx size %#llx (bound %s)", (unsigned long long)p.key,
                                 ns, n_desc, k, (unsigned)d.cbuf_index, off, d.is_written ? " written" : "", (unsigned long long)addr,
                                 (unsigned long long)size, b ? "yes" : "NO");
                    }
                }
                // (census: every single descriptor, and table entries that hold something, with where they came from)
                if (d.count == 1 || addr) {
                    // (unresolved: the uniform buffer the word was read from, its address and size, in the vk fields)
                    const nvn_p_bind_buf* ub = d.cbuf_index >= 3 && d.cbuf_index < 3 + 16
                                                   ? &s.ubo[std::clamp(ns, 0, kNvnStages - 1)][d.cbuf_index - 3] : nullptr;
                    const bool none = !(addr && plausible);
                    stats_note_bind(p.key, ns, none ? 0 : base, none ? size : n, d.is_written, d.count > 1, d.cbuf_index, off, addr,
                                    none && ub ? ub->addr : (uint64_t)(uintptr_t)b.buffer, none && ub ? ub->size : b.offset);
                }
                static std::atomic<int> table_told{0}, table_seen{0};
                static const int table_unlimited = env_on("RX_VK_LOG_TABLE");
                if (d.count > 1 && table_track_open(table_told.load(std::memory_order_relaxed),
                                                    table_seen.load(std::memory_order_relaxed), table_unlimited)) {
                    table_seen.fetch_add(1, std::memory_order_relaxed);   // (hookshot) a table entry bound one frame and not the next, 60 lines, then it stops looking
                    static std::mutex tm;
                    static std::unordered_map<uint64_t, std::pair<bool, uint64_t>> was;   // -> bound, its address
                    static int told;
                    const uint64_t key = p.key ^ ((uint64_t)off << 48) ^ ((uint64_t)ns << 60);
                    const bool bound = b.buffer != VK_NULL_HANDLE;
                    std::lock_guard lk(tm);
                    auto it = was.find(key);
                    if (it != was.end() && it->second.first != bound && told < 60) {
                        told++, table_told++;
                        logf("vk: table entry %s at frame %llu: prog %016llx ssbo %u of %u (c%u+%#x) addr %#llx size %#llx "
                             "(before %#llx)", bound ? "bound again" : "UNBOUND", (unsigned long long)texture_frame_no(),
                             (unsigned long long)p.key, k, d.count, (unsigned)d.cbuf_index, off, (unsigned long long)addr,
                             (unsigned long long)size, (unsigned long long)it->second.second);
                    }
                    was[key] = {bound, addr};
                }
                {   // (hookshot) RX_VK_LOG_WRITEBACK=1: the programs that bind a big written storage buffer (each written
                    // range is copied back to guest memory at the submission's end), 40 of them
                    static const bool log_wb = env_on("RX_VK_LOG_WRITEBACK");
                    static int logged;
                    if (log_wb && d.is_written && n >= (8ull << 20) && logged < 40) {
                        static std::vector<std::pair<uint64_t, uint32_t>> seen;
                        const std::pair<uint64_t, uint32_t> key{p.key, d.cbuf_offset + k * 16};
                        if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
                            seen.push_back(key);
                            logged++;
                            logf("vk: written storage buffer of %.1f MB: prog %016llx ssbo %u of %u (c%u+%#x) at %#llx, stage %d",
                                 n / 1048576.0, (unsigned long long)p.key, k, d.count, (unsigned)d.cbuf_index, off,
                                 (unsigned long long)addr, ns);
                        }
                    }
                }
                if (watch_prog() && p.key == watch_prog()) {   // (RX_VK_WATCH_PROG=<key>: that program's bindings)
                        watch_note("vk: watch %016llx: ssbo %u (c%u+%#x%s) addr %#llx size %#llx -> buffer %p offset %#llx range %#llx; grid %u %u %u",
                             (unsigned long long)p.key, k, (unsigned)d.cbuf_index, off, d.is_written ? ", written" : "",
                             (unsigned long long)addr, (unsigned long long)size, (void*)b.buffer,
                             (unsigned long long)b.offset, (unsigned long long)n, s.grid[0], s.grid[1], s.grid[2]);
                }
                if (slow_desc_timing()) slow_descriptor(t0, d.is_written ? "storage buffer (written)" : "storage buffer", base, n);
                if (dump_on() && dump_ssbo_wanted(p.key)) {   // (RX_VK_DUMP_SSBO: where it is, and what the CPU's memory holds)
                    std::vector<uint32_t> words((size_t)std::min<uint64_t>(n, 65536) / 4);
                    size_t nz = 0;
                    const bool read = b && addr && !words.empty() && host->gpu_read(base, words.data(), words.size() * 4);
                    if (read)
                        for (uint32_t w : words) nz += w != 0;
                    logf("vk: dump ssbo %03d prog %016llx stage %d sb%u (c%u+%#x%s): addr %#llx size %#llx bound %s; guest "
                         "memory: %s%zu of %zu words not zero", dump_seq(), (unsigned long long)p.key, ns, k,
                         (unsigned)d.cbuf_index, off, d.is_written ? ", written" : "", (unsigned long long)addr,
                         (unsigned long long)size, b ? "yes" : "NO", read ? "" : "(unreadable) ", nz, words.size());
                }
                {   // (RX_VK_PEEK_PROG=<key prefix>: that program's storage buffers' first words in guest memory, and what the
                    // first 64-bit word points at: the foliage culling's inputs)
                    static const char* peek = getenv("RX_VK_PEEK_PROG");
                    static std::atomic<int> peeked{0};
                    char hex[24];
                    if (peek) snprintf(hex, sizeof hex, "%016llx", (unsigned long long)p.key);   // (only with the switch: once per buffer per draw)
                    if (peek && addr && !strncmp(hex, peek, strlen(peek)) && peeked++ < 60) {
                        uint32_t w[8]{}, q[8]{};
                        const bool ok = host->gpu_read(addr, w, sizeof w);
                        const uint64_t ptr = (uint64_t)w[1] << 32 | w[0];
                        const bool ok2 = ptr && host->gpu_read(ptr, q, sizeof q);
                        logf("vk: peek %s sb%u (c%u+%#x%s) addr %#llx size %#llx: %s%08x %08x %08x %08x %08x %08x %08x %08x; at %#llx: %s%08x %08x %08x %08x %08x %08x",
                             hex, k, (unsigned)d.cbuf_index, off, d.is_written ? ", written" : "", (unsigned long long)addr,
                             (unsigned long long)size, ok ? "" : "(unreadable) ", w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
                             (unsigned long long)ptr, ok2 ? "" : "(unreadable) ", q[0], q[1], q[2], q[3], q[4], q[5]);
                    }
                }
                if (addr && plausible && !b) {   // (a storage buffer that could not be bound: the program reads zeros from it)
                    static std::atomic<int> unbound{0};
                    if (unbound++ < 40)
                        logf("vk: storage buffer NOT bound (reads zeros): program %016llx stage %d c%u+%#x addr %#llx size "
                             "%#llx%s", (unsigned long long)p.key, ns, (unsigned)d.cbuf_index, off,
                             (unsigned long long)addr, (unsigned long long)size, d.is_written ? " (written)" : "");
                }
                if (!b && d.is_written && !(d.count == 1 && d.cbuf_index == 0 && d.cbuf_offset == 0xff0))
                    null_write_note(p, ns, d.cbuf_index, off, k, d.count, addr, size,
                                    !addr ? kNwAddressZero : plausible ? kNwNotBound : kNwNotBuffer);
                VkDescriptorBufferInfo scratch{};
                if (!b && d.is_written && null_write_on()) scratch = null_write_binding();
                out.bufs.push_back(b ? VkDescriptorBufferInfo{b.buffer, b.offset, n}
                                    : scratch.buffer ? scratch
                                    : VkDescriptorBufferInfo{dev.null_descriptor ? VK_NULL_HANDLE : zero_buffer, 0,
                                                             dev.null_descriptor ? VK_WHOLE_SIZE : 16});
                {   // (RX_VK_LOG_WRITTEN=1: every storage buffer range a program writes, once, 300 of them)
                    static const bool on = env_on("RX_VK_LOG_WRITTEN");
                    if (on && d.is_written) {
                        static std::mutex wm;
                        static std::unordered_set<uint64_t> seen;
                        std::lock_guard lk(wm);
                        if (seen.size() < 300 && seen.insert(p.key ^ (addr >> 12) * 0x9E3779B97F4A7C15ull).second)
                            logf("vk: writes storage: prog %016llx c%u+%#x addr %#llx size %#llx frame %llu", (unsigned long long)p.key,
                                 (unsigned)d.cbuf_index, off, (unsigned long long)addr, (unsigned long long)size, (unsigned long long)texture_frame_no());
                    }
                }
                if (b && d.is_written) {
                    const auto t1 = slow_desc_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    const uint64_t guest = host->gpu_guest(base);
                    if (guest) { ParLock lk; memory_stamp_gpu(guest, n); }
                    if (slow_desc_timing()) slow_descriptor(t1, "stamp", base, n);
                }
            }
            w.pBufferInfo = &out.bufs[first];
            out.writes.push_back(w);
        }
        auto texel_views = [&](const auto& list, VkDescriptorType type, bool storage) {
            for (const auto& d : list) {
                VkWriteDescriptorSet w = write(type, d.count);
                const size_t first = out.views.size();
                for (uint32_t k = 0; k < d.count; k++) {
                    const uint32_t handle = handle_of(s, st, d, k);
                    desc_fp_note(handle);   // (Descriptor reuse)
                    nvn_tex t;
                    nvn_view v;
                    bool has_view = false;
                    VkBufferView bv = VK_NULL_HANDLE;
                    if (handle && texture_of((int)(handle & 0xFFFFF), &t, &v, &has_view)) {
                        ParLock lk;
                        bv = texel_buffer_view(&t, has_view ? &v : nullptr, storage);
                    }
                    out.views.push_back(bv);
                }
                w.pTexelBufferView = &out.views[first];
                out.writes.push_back(w);
            }
        };
        texel_views(info.texture_buffer_descriptors, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, false);
        texel_views(info.image_buffer_descriptors, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, true);
        for (const auto& d : info.texture_descriptors) {
            FINE_SCOPE(FT_TEX);
            VkWriteDescriptorSet w = write(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, d.count);
            const size_t first = out.imgs.size();
            const VkImageViewType vt = view_type(d.type);
            // (Lichii bd0171f7) A handle table's array: the shader indexes it per pixel with an index it read itself (OoT's
            // clustered lights: a light index from the pixel's list), and element k is table entry k's texture (that light's
            // shadow map). The handles of all elements at once: the constant words and storage addresses the program reads
            // are read once
            static thread_local std::vector<uint32_t> table;
            const bool is_table = d.handle_program != ~0U && d.count > 1;
            if (is_table) table.resize(d.count), stage_handles(s, st, d.handle_program, d.count, table.data(), p.key);
            uint32_t prev_handle = 0;
            VkImageView prev_view = VK_NULL_HANDLE;
            Texture* prev_tex = nullptr;
            bool have_prev = false;
            for (uint32_t k = 0; k < d.count; k++) {
                // (Fritchii d603d98bb) A handle word where a storage buffer of the same program keeps its address is not a
                // handle: OoT's water samples each light's texture with a handle loaded per pixel from the light buffer, and
                // the translation falls back to the first constant word on its path, c0[0x520], the light buffer's address.
                // Its low bits named an unrelated texture that changed with the double-buffered buffer: the water flickered
                // every other frame. No texture there now
                // (fragment stages only, the water's case: in other stages (the foliage culling's) the textures went missing and
                // the foliage was culled whenever Link swung his sword)
                // (Handle tables now give those samples their own textures: this is for what the translation still cannot
                // compute per draw)
                const uint32_t handle = is_table ? table[k]
                                        : (st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT && handle_is_storage_pointer(info, d, k))
                                            ? 0u : handle_of(s, st, d, k);
                desc_fp_note(handle);   // (Descriptor reuse)
                Texture* tex = nullptr;
                VkImageView view = VK_NULL_HANDLE;
                if (is_table && have_prev && handle == prev_handle) {   // (a table repeats entries: unused ones are 0)
                    view = prev_view, tex = prev_tex;
                } else {
                    upload_defer_ok = st.vk_stage != VK_SHADER_STAGE_COMPUTE_BIT;   // (a draw's: its new ASTC textures may wait a frame)
                    mip_stream_draw = s.rt.depth && st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT && vt == VK_IMAGE_VIEW_TYPE_2D &&
                                      !is_table && !d.is_integer && !d.is_depth && d.type_fixed;
                    view = handle ? texture_view_of(handle, vt, ViewUse::Sample, VK_FORMAT_UNDEFINED, &tex) : VK_NULL_HANDLE;
                    mip_stream_draw = upload_defer_ok = false;
                    // (Lichii bd0171f7) A depth-compare sample of a colour texture is undefined and differs between vendors: a
                    // handle read from memory (an unused or stale table entry) that names one binds nothing
                    if (is_table && view && tex && d.is_depth && !is_depth_format(tex->format)) {
                        static std::atomic<int> told{0};
                        if (told++ < 8)
                            logf("vk: handle table: element %u of program %016llx names texture #%u (format %d) for a depth "
                                 "comparison: bound as nothing", k, (unsigned long long)p.key, tex->id, (int)tex->format);
                        view = VK_NULL_HANDLE, tex = nullptr;
                    }
                    if (is_table) prev_handle = handle, prev_view = view, prev_tex = tex, have_prev = true;
                }
                // (the shader clamps its index to the last element: a table that names a texture there may run past the
                // array, and its lights from that index on all take this one's texture. Said once)
                if (is_table && k + 1 == d.count && view) {
                    static std::atomic<bool> told{false};
                    if (!told.exchange(true))
                        logf("vk: handle table: last element %u of program %016llx names texture #%u: the table may have more "
                             "entries than the array, and those take this texture (said once)", k, (unsigned long long)p.key,
                             tex ? tex->id : 0);
                }
                if (view && null_tex_wanted(st, d)) view = VK_NULL_HANDLE, tex = nullptr;   // (RX_VK_NULL_TEX)
                if (dump_on()) dump_input('t', (uint32_t)(out.imgs.size() - 0), handle, view ? tex : nullptr);
                if (view && tex && trace_tex_wanted(tex)) trace_tex_hit = tex;
                if (log_img_wanted(p.key)) {
                    static std::atomic<int> told{0};
                    if (told++ < 40)
                        logf("vk: img prog %016llx texture %u: handle %#x, shader integer %d type %d depth %d, view format "
                             "%d, texture #%d format %d %ux%ux%u x%u levels %u sparse %d view %p", (unsigned long long)p.key, k,
                             handle, (int)d.is_integer, (int)d.type, (int)d.is_depth, (int)last_view_format,
                             tex ? (int)tex->id : -1, tex ? (int)tex->format : -1, tex ? tex->width : 0, tex ? tex->height : 0,
                             tex ? tex->depth : 0, tex ? tex->layers : 0, tex ? tex->levels : 0, tex ? (int)tex->sparse : 0,
                             (void*)view);
                }
                if (dump_on() && view && tex && dump_sampled_wanted(p.key)) {
                    static std::unordered_set<uint32_t> done;
                    static int last_seq = 0;
                    if (dump_seq() < last_seq) done.clear();
                    last_seq = dump_seq();
                    if (done.size() < 96 && done.insert(tex->id).second) dump_sampled().push_back(tex);
                }
                if (watch_prog() && p.key == watch_prog())
                    watch_note("vk: watch %016llx: texture %u handle %#x -> #%u %#llx %ux%ux%u image %p view %p%s",
                               (unsigned long long)p.key, k, handle, tex ? tex->id : 0, (unsigned long long)(tex ? tex->gpu : 0),
                               tex ? tex->width : 0, tex ? tex->height : 0, tex ? tex->layers : 0, tex ? (void*)tex->image : nullptr,
                               (void*)view, view ? "" : " (dummy)");
                if (view && tex && tex->gpu_written) draw_sampled().push_back(tex);
                if (!view) view = dummies.view[(int)vt], st_dummy_textures += handle != 0;
                // (an image of one level: no anisotropy of the row's, sampler_make)
                out.imgs.push_back({sampler_for((int)(handle >> 20), d.is_depth, tex && tex->levels <= 1), view,
                                    VK_IMAGE_LAYOUT_GENERAL});
            }
            w.pImageInfo = &out.imgs[first];
            out.writes.push_back(w);
        }
        for (const auto& d : info.image_descriptors) {
            VkWriteDescriptorSet w = write(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, d.count);
            const size_t first = out.imgs.size();
            const VkImageViewType vt = view_type(d.type);
            for (uint32_t k = 0; k < d.count; k++) {
                const uint32_t handle = handle_of(s, st, d, k);
                desc_fp_note(handle);   // (Descriptor reuse)
                Texture* tex = nullptr;
                uint32_t level = 0;
                VkImageView view = handle ? texture_view_of(handle, vt, ViewUse::Storage, image_format(d.format), &tex, &level)
                                          : VK_NULL_HANDLE;
                if (view && tex && d.is_written) { ParLock lk; texture_written(tex, "a storage image of a draw"); }
                if (view && tex) draw_sampled().push_back(tex);
                if (log_img_wanted(p.key)) {
                    static std::atomic<int> told{0};
                    if (told++ < 16)
                        logf("vk: img prog %016llx image %u: handle %#x, shader integer %d format %d (view as %d), "
                             "view format %d, texture format %d", (unsigned long long)p.key, k, handle,
                             (int)d.is_integer, (int)d.format, (int)image_format(d.format),
                             (int)last_view_format, tex ? (int)tex->format : -1);
                }
                if (tex && tex->sparse && d.is_written) {
                    static int logged;
                    if (logged++ < 20)
                        logf("vk: program %016llx stores into sparse texture %#llx (handle %#x, view format %d, %s)",
                             (unsigned long long)p.key, (unsigned long long)tex->gpu, handle, (int)image_format(d.format),
                             view ? "view made" : "NO view");
                }
                if (view && tex && d.is_written && (dump_on() || probe_on())) dump_written().push_back({tex, level});
                if (dump_on()) dump_input(d.is_written ? 'W' : 'R', (uint32_t)out.imgs.size(), handle, view ? tex : nullptr);
                if (!view && !dev.null_descriptor) view = dummies.view[(int)vt];
                out.imgs.push_back({VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL});
            }
            w.pImageInfo = &out.imgs[first];
            out.writes.push_back(w);
        }
    }
}

/// Binds the descriptors built: pushed, or a set from the recording's pool
// (push descriptors keep what is not pushed again: a program drawn again pushes only the bindings that changed,
// while nothing else was bound at its bind point in this command buffer)
struct Pushed {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    size_t n = 0;                                // (the writes as pushed)
    // (hookshot) each write's descriptors, as pushed, in one vector of words and where each write's begin: a program change was a
    // vector of vectors resized to the new program's writes, each new one grown word by word (3 allocations a write, and the ones cut
    // off freed: a dozen of each a draw)
    std::vector<uint64_t> words;
    std::vector<uint32_t> at;
};
TLS_OBJECT(pushed, std::array<Pushed, 2>)   // [graphics, compute]
} // namespace
std::string state_list_snapshot(bool* present) { return state_list_bytes(present); }   // (vk_draw.h: the menu's)
bool uniform_snapshot_on() {   // (vk_internal.h: ub_snapshot_wanted's answer for vk_memory.cpp's guard and vk_replay.cpp's parallel replay)
    static const bool on = ub_snapshot_wanted();
    return on;
}
/// (hookshot) A compute pipeline of the back end's own, made elsewhere (vk_fsr3.cpp), through the pipeline cache the game's are kept
/// in across runs (hookshot_vk_pipelines.bin), as theirs are: in play through the play caches (create_cached), so that it does not grow
/// the start's cache; before the cache is loaded, with none. (Outside the unnamed namespace: vk_fsr3.cpp links to it)
VkResult shared_compute_pipeline(const VkComputePipelineCreateInfo* ci, VkPipeline* out) {
    // (not a game program: ComputeTurn must not weigh, refuse or name it as the game program this thread described last)
    compute_described = {};
    return cached_compute(ci, out);
}
/// (hookshot) A pipeline cache of the caller's own in the game folder's cache/ (vk_fsr4.cpp: AMD FSR 4's unrolled layers). Apart
/// from the translator's files, which pcache_drop_stale drops when the translator changes (most updates): what goes in it is not
/// the translator's, and the launcher moves the whole cache/ folder into an update's. Its file's data when this device and
/// driver can take it (pcache_file_read), else empty; *read: the bytes read, *refused: a file there that was not taken. Not read
/// (nor saved, below) where the pipeline cache's file is not (RX_VK_PCACHE_LOAD=0 or its rule: pcache_no_save, pcache_init)
VkPipelineCache own_pipeline_cache(const char* name, size_t* read, bool* refused) {
    PipelineCacheFile data = pcache_load_off() ? PipelineCacheFile{} : pcache_file_read(cache_path(name));
    *read = data.size(), *refused = data.rejected();
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size(), ci.pInitialData = data.data();
    VkPipelineCache cache = VK_NULL_HANDLE;
    if (vkCreatePipelineCache(dev.device, &ci, nullptr, &cache) == VK_SUCCESS) return cache;
    *read = 0;   // (data the driver would not take: an empty cache, as for a file of another driver)
    ci.initialDataSize = 0, ci.pInitialData = nullptr;
    return vkCreatePipelineCache(dev.device, &ci, nullptr, &cache) == VK_SUCCESS ? cache : VK_NULL_HANDLE;
}
/// (hookshot) That cache written whole to its file (through a .tmp moved over it, as the play files are) when its data is not the
/// size last read or written (*saved, updated): a pipeline the driver put in it since, whether it compiled it or took it from its
/// own disk cache, which an update's folder (another path) does not share. false: it could not be written
bool own_pipeline_cache_save(VkPipelineCache cache, const char* name, size_t* saved) {
    if (!cache || pcache_load_off()) return true;
    size_t n = 0;
    {
        PcacheFetch fetch(-1, true);
        if (!fetch.held() || vkGetPipelineCacheData(dev.device, cache, &n, nullptr) != VK_SUCCESS) return false;
    }
    if (!n || n == *saved) return true;
    const size_t wrote = play_file_write(cache, cache_path(name));
    if (wrote) *saved = wrote;
    return wrote != 0;
}
/// (hookshot) A compute pipeline built as shared_compute_pipeline's (the same turn and gates), through the caller's own cache
VkResult own_cache_compute_pipeline(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* out) {
    compute_described = {};
    const ComputeTurn turn(*ci);
    if (!turn.ok()) return *out = VK_NULL_HANDLE, VK_ERROR_OUT_OF_HOST_MEMORY;   // (stopped while it waited, or refused: not compiled)
    IntelDriverGate gate(compiles_nothing(ci->flags));
    if (gate.busy) return intel_gate_busy();
    return guarded_compute_held(cache, ci, out);
}
// Preserve the SDK's batch and allocator while using the same cache fetch lock as the renderer.
VkResult own_cache_sdk_pipelines(PFN_vkCreateComputePipelines create, VkDevice device, VkPipelineCache cache, uint32_t count,
                                 const VkComputePipelineCreateInfo* infos, const VkAllocationCallbacks* allocator, VkPipeline* pipelines) {
    const PcacheUse use(cache);
    sdk_compiles.fetch_add(1, std::memory_order_acq_rel);
    const VkResult result = create(device, cache, count, infos, allocator, pipelines);
    prepare_compiled();
    sdk_compiles.fetch_sub(1, std::memory_order_release);
    return result;
}
/// (hookshot) The driver faults the pipeline compiles caught so far (guarded_graphics, guarded_compute): the preparation's
/// watchdog says so (vk_main.cpp). Outside the unnamed namespace: vk_main.cpp links to it
int driver_faults() { return st_driver_faults.load(std::memory_order_relaxed); }
std::atomic<int64_t> st_driver_out_of_memory{0};
/// (hookshot) The guarded driver calls, for vk_shader.cpp and vk_temporal.cpp. Declaring guarded_* itself in the header
/// is a second function (the definitions sit in the unnamed namespace) and every call in this file is ambiguous.
VkResult extern_guarded_graphics(VkPipelineCache cache, const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl) {
    return guarded_graphics(cache, ci, pl);
}
/// extern_guarded_compute makes the renderer's own compute pipelines (the texture pipelines, the upscale and mask helpers,
/// the write-back diff, the late pages' fix, FSR 3's inputs, frame generation's passes): one the driver refuses with its
/// optimizer is asked once more without it. Intel's Windows driver on the Arc A-series (an A580 on 101.9175) answered
/// VK_ERROR_UNKNOWN (-13) for optimized builds it made without the optimizer: the ASTC decoder refused so, every ASTC texture
/// was decoded on the CPU (a frame's uploads 30.9 s, the screen black after the opening cutscene); the write-back diff, the
/// late pages' fix and the upscale helper standing in for a network left out were off. Where the optimized build is made
/// (NVIDIA, AMD) nothing else is asked. The game's programs have their own path (compute_optimize)
VkResult extern_guarded_compute(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
    *pl = VK_NULL_HANDLE;
    VkResult r = guarded_compute(cache, ci, pl);
    if (r == VK_SUCCESS && *pl) return VK_SUCCESS;
    *pl = VK_NULL_HANDLE;
    if (r == VK_SUCCESS) r = VK_ERROR_UNKNOWN;   // (no pipeline is no success)
    if (ci->flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT) return r;
    VkComputePipelineCreateInfo plain = *ci;
    plain.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    const VkResult p = guarded_compute(cache, &plain, pl);
    const bool made = p == VK_SUCCESS && *pl;
    if (!made) *pl = VK_NULL_HANDLE;
    static std::atomic<int> told{0};
    if (told++ < 8)
        logf("vk: one of the renderer's compute pipelines was not made with the driver's optimizer (%d): %s", (int)r,
             made ? "made without it" : "nor without it");
    return made ? VK_SUCCESS : p == VK_SUCCESS ? r : p;
}
VkResult extern_guarded_shader_module(const VkShaderModuleCreateInfo* ci, VkShaderModule* m) {
    return guarded_shader_module(ci, m);
}
/// (hookshot) A program whose shader module the driver faulted on (vk_shader.cpp make_module) is kept out as one whose pipeline
/// it faulted on: not asked of it again in this run, nor in later runs on the same driver (driver_faults.h)
bool driver_faulted_program(uint64_t key) { return key && driver_faults_on() && fault_quarantine().contains(key); }
void driver_faulted_module(uint64_t key) {
    if (!key || !driver_faults_on() || fault_quarantine().contains(key)) return;
    fault_quarantine().load({key});
    logf("vk: the driver faulted making a shader module of %016llx: that program is not asked of it again (%s)",
         (unsigned long long)key, driver_faults_path().c_str());
    driver_fault_note(key);
}
/// (hookshot) Whether predict_run or precompile_run went on and left workers inside the driver: the pipeline cache's save can wait
/// on them, and so can exit() unloading the driver (vk_main.cpp, the --prepare-shaders run)
std::atomic<bool> st_left_workers{false};
void prepare_left_workers_note() { st_left_workers.store(true); }
bool prepare_left_workers() { return st_left_workers.load(); }
void draw_skip_report(double frames) { skip_report_impl(frames), fine_report(frames); }
void descriptors_disturbed() {
    pushed()[0].cmd = pushed()[1].cmd = VK_NULL_HANDLE;
}
bool stream_holds(const uint8_t* base, uint64_t at, uint64_t n) {
    return cur && cur->mapped == base && n && at + n >= at && at + n <= cur->used;
}
bool mesh_args_ready();             // (below: the task programs' firsts)
extern uint64_t mesh_args_address;
namespace {

bool bind_descriptors(VkCommandBuffer cmd, VkPipelineBindPoint bp, const Program& p) {
#ifdef TRACY_ENABLE
    HT_SCOPE_DRAW("bind_descriptors");
#endif
    DescWriter& wr = dw();   // (once: the accessor is a call and a thread-local read at each use. The same object for the thread's life: build_descriptors resets it in place)
    if (wr.writes.empty()) return true;
    // A failed sampler is retryable, but cannot be published to either descriptor path.
    // Check the entire array before updating the pushed-state cache or allocating a set.
    for (const auto& w : wr.writes)
        if (w.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            for (uint32_t k = 0; k < w.descriptorCount; k++)
                if (!w.pImageInfo[k].sampler) return false;
    Pushed& ps = pushed()[bp == VK_PIPELINE_BIND_POINT_COMPUTE];
    if (p.push) {
        static const bool whole = env_on("RX_VK_PUSH_ALL");   // (A/B: every binding every time)
        if (whole || ps.cmd != cmd || ps.layout != p.layout || ps.n != wr.writes.size()) {
            if (g_fine) {
                fine_n[FC_PUSH_FULL]++, fine_n[FC_PUSH_WRITES] += (int64_t)wr.writes.size();
                for (const VkWriteDescriptorSet& w : wr.writes) fine_n[FC_PUSH_DESCS] += w.descriptorCount;
            }
            {
                FINE_SCOPE(FT_PUSHCALL);
                vkCmdPushDescriptorSetKHR(cmd, bp, p.layout, 0, (uint32_t)wr.writes.size(), wr.writes.data());
            }
            ps.cmd = cmd, ps.layout = p.layout, ps.n = wr.writes.size();
            ps.words.clear(), ps.at.clear();
            for (const VkWriteDescriptorSet& w : wr.writes) ps.at.push_back((uint32_t)ps.words.size()), words_append(w, ps.words);
            return true;
        }
        static thread_local std::vector<VkWriteDescriptorSet> changed;
        changed.clear();
        // (hookshot) With the uniform copies shared within a recording (ub_snap_dedup_mode), a draw's uniform buffer writes are
        // often the ones pushed for the draw before. An Arc B580 (driver 32.0.101.9154) draws wrong when a push carries some of a
        // draw's uniform buffer writes and leaves others out as unchanged: objects with other objects' matrices (Castle Town's
        // market), as with the snapshot off (RX_VK_UB_SNAPSHOT=0: main's delta push there). Pushing those left out along with them
        // puts it right; adding them to a push with no uniform buffer write, or to a draw that pushes nothing, does not (one
        // session each, PR 952). The banks and c1, left out unchanged beside a draw's own new copies, draw right: what tells the
        // two apart is not known. Before the sharing each draw had copies of its own, so every one of its uniform buffer writes
        // had changed and was pushed: they still all are, the textures and storage buffers only when changed, as before
        static const bool ub_every = ub_snapshot_wanted() && ub_snap_dedup_mode() == 1;
        const auto again = [&](const VkWriteDescriptorSet& w) { return ub_every && w.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; };
        // (the same layout, so the same count of words in each write: compared and kept in place)
        for (size_t i = 0; i < wr.writes.size(); i++) {
            const uint64_t* end = ps.words.data() + (i + 1 < ps.at.size() ? ps.at[i + 1] : ps.words.size());
            const int same = words_same_store(wr.writes[i], ps.words.data() + ps.at[i], end);
            if (same < 0) {   // (not so: nothing is known of what is pushed; the whole set again)
                ps.cmd = VK_NULL_HANDLE;
                return bind_descriptors(cmd, bp, p);
            }
            if (!same || again(wr.writes[i])) changed.push_back(wr.writes[i]);
        }
        if (!changed.empty()) {
            if (g_fine) {
                fine_n[FC_PUSH_DELTA]++, fine_n[FC_PUSH_WRITES] += (int64_t)changed.size();
                for (const VkWriteDescriptorSet& w : changed) fine_n[FC_PUSH_DESCS] += w.descriptorCount;
            }
            FINE_SCOPE(FT_PUSHCALL);
            vkCmdPushDescriptorSetKHR(cmd, bp, p.layout, 0, (uint32_t)changed.size(), changed.data());
        } else {
            FINE_N(FC_PUSH_NONE, 1);
        }
        return true;
    }
    ps.cmd = VK_NULL_HANDLE;   // (a set bound: what was pushed is not known to stay)
    if (g_fine) {
        fine_n[FC_POOL_SETS]++;
        for (const VkWriteDescriptorSet& w : wr.writes) fine_n[FC_POOL_DESCS] += w.descriptorCount;
    }
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = cur->extra.empty() ? cur->pool : cur->extra.back();
    if (!ai.descriptorPool) return false;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(dev.device, &ai, &set) != VK_SUCCESS) {
        const VkDescriptorPool pool = make_pool();
        if (!pool) return false;
        ai.descriptorPool = pool;
        if (vkAllocateDescriptorSets(dev.device, &ai, &set) != VK_SUCCESS) {
            vkDestroyDescriptorPool(dev.device, pool, nullptr);
            return false;
        }
        try {
            cur->extra.push_back(pool);
            pools_extra.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception&) {
            vkDestroyDescriptorPool(dev.device, pool, nullptr);
            return false;
        }
    }
    cur->pool_sets++;
    for (auto& w : wr.writes) {
        w.dstSet = set;
        switch (w.descriptorType) {   // (make_pool's order)
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: cur->pool_descs[0] += w.descriptorCount; break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: cur->pool_descs[1] += w.descriptorCount; break;
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: cur->pool_descs[2] += w.descriptorCount; break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: cur->pool_descs[3] += w.descriptorCount; break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: cur->pool_descs[4] += w.descriptorCount; break;
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: cur->pool_descs[5] += w.descriptorCount; break;
            default: break;
        }
    }
    vkUpdateDescriptorSets(dev.device, (uint32_t)wr.writes.size(), wr.writes.data(), 0, nullptr);
    vkCmdBindDescriptorSets(cmd, bp, p.layout, 0, 1, &set, 0, nullptr);
    return true;
}

// ---- RX_VK_CHECKPOINTS: a marker before each draw and dispatch (VK_NV_device_diagnostic_checkpoints); when the
// device is lost, the markers the GPU reached say which program it was running ----
struct Checkpoint { uint64_t prog; uint64_t frame; uint32_t seq; char kind; };
Checkpoint checkpoints[1 << 16];
std::atomic<uint32_t> checkpoint_n{0};
// (hookshot) with the buffer markers, each draw's targets and the GPU-written textures it samples, kept beside
// its marker: the report then says which pass the hung draw was in and what it read, which a program key alone did not (the
// same program draws in several passes, and the RTX 50 reports named a different program almost every time).
// Made only when the buffer markers are (RX_VK_CHECKPOINTS=marker): 8 MB
struct CheckpointDraw {
    uint32_t width = 0, height = 0, c0 = 0, c0_format = 0, depth = 0, depth_format = 0, n = 0;
    struct { uint32_t id, format; uint16_t w, h, d; uint8_t type, layers; } s[6];
};
std::unique_ptr<CheckpointDraw[]> checkpoint_draws;

// (VK_AMD_buffer_marker, where the NV checkpoints are missing: marker i's sequence number goes to word 2i when the GPU
// reaches it and to word 2i + 1 once all the work before it is done. Host memory, not VRAM: a GPU reset loses VRAM.)
VkBuffer marker_buffer = VK_NULL_HANDLE;
volatile uint32_t* marker_words = nullptr;

// (RX_VK_CHECKPOINTS=ext: the extension on, no markers: which of the two keeps the device from being lost)
bool checkpoint_markers() {
    static const bool on = !getenv("RX_VK_CHECKPOINTS") || strcmp(getenv("RX_VK_CHECKPOINTS"), "ext") != 0;
    return on;
}
bool checkpoints_on() {
    return ((dev.checkpoints && vkCmdSetCheckpointNV) || marker_words) && checkpoint_markers();
}
void set_checkpoint(VkCommandBuffer cmd, uint32_t i, uint32_t seq) {
    if (marker_words) {
        vkCmdWriteBufferMarkerAMD(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, marker_buffer, 8ull * i, seq);
        vkCmdWriteBufferMarkerAMD(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, marker_buffer, 8ull * i + 4, seq);
    } else {
        vkCmdSetCheckpointNV(cmd, (void*)(uintptr_t)(i + 1));
    }
}
// (RX_VK_WORK_LOCAL=1) Every draw and dispatch took the next marker with ++checkpoint_n, one atomic the five recording
// threads of a parallel replay all bumped, and wrote its 24-byte entry next to another thread's. With the switch, a recording thread
// inside a parallel segment (work_seg_local) takes 64 numbers at a time (one fetch_add) and uses them in order: each marker still
// gets a number of its own (the blocks never overlap), its entry is written by one thread, and set_checkpoint records the same
// call with it. A block is used only in the frame it was taken in (texture_frame_no), and a frame sets ~3,000 markers of the
// ring's 65,536, so a block's entries are not reused by another thread while it holds it. What changes is the device-lost report
// only: neighbouring entries of the ring are then the same thread's draws, not the last ones of all threads in time. The NV
// checkpoints stay on as they were (VK_NV_device_diagnostic_checkpoints: the device was lost every run without them). The AMD buffer
// markers (marker_words) keep the global order: their report reads the numbers as a time order.
uint32_t checkpoint_seq() {
    if (!work_seg_local || marker_words) return ++checkpoint_n;
    thread_local uint32_t next = 0, end = 0;
    thread_local uint64_t frame = ~0ull;
    const uint64_t f = texture_frame_no();
    if (next == end || frame != f) {
        next = checkpoint_n.fetch_add(64) + 1, end = next + 64;
        frame = f;
    }
    return next++;
}
void checkpoint(VkCommandBuffer cmd, const Program& p, char kind, const Targets* at = nullptr) {
    if (!checkpoints_on()) return;
    const uint32_t seq = checkpoint_seq(), i = (seq - 1) & 0xFFFF;
    checkpoints[i] = {p.key, texture_frame_no(), seq, kind};
    if (checkpoint_draws) {   // (the slot is this marker's alone: no lock)
        CheckpointDraw& c = checkpoint_draws[i];
        c = CheckpointDraw{};
        if (at) {
            c.width = at->width, c.height = at->height, c.c0_format = at->n ? (uint32_t)at->color_format[0] : 0;
            c.c0 = at->n && at->color[0] ? at->color[0]->id : 0;
            c.depth = at->depth ? at->depth->id : 0, c.depth_format = (uint32_t)at->depth_format;
            for (const Texture* t : gfx_sampled()) {   // (this draw's: set from its descriptors just before)
                if (c.n == 6) break;
                c.s[c.n++] = {t->id, (uint32_t)t->format, (uint16_t)t->width, (uint16_t)t->height, (uint16_t)t->depth,
                              (uint8_t)t->type, (uint8_t)std::min<uint32_t>(t->layers, 255)};
            }
        }
    }
    set_checkpoint(cmd, i, seq);
}
const char* checkpoint_kind(char c) {
    return c == 'd' ? "draw of program" : c == 'c' ? "dispatch of program" : c == 'i' ? "INDIRECT dispatch of program"
         : c == 'm' ? "mirror sync of pool"
         : c == 'b' ? "write-back of pool" : c == 'w' ? "write-back diff pass, chunk count" : c == 'u' ? "upload of texture"
         : c == 'g' ? "GPU unswizzle of texture"
         : c == 's' ? "mirror syncs' pass, chunk count"
         : "?";
}

thread_local uint64_t fallback_payload_address = 0, fallback_records_address = 0;   // (the draw's fallback buffers)
thread_local uint64_t fallback_indirect_input = 0;

void push_constants(VkCommandBuffer cmd, const Program& p, const Targets* at, uint32_t fallback_first = 0) {
    checkpoint(cmd, p, at ? 'd' : 'c', at);
    {   // (frame overlap, vk_replay.cpp: each draw and dispatch counted here, where every one comes by whatever it pushes.
        // A dispatch, the mesh fallback's compute stages, and a draw whose program follows pointers through the page table
        // (a pool never bound has no mirror there) read the game's memory at the GPU's time as far as the replay knows)
        bool follows = !at;
        if (at && p.uses_push)
            for (const Stage& st : p.stages) follows |= st.info.uses_global_memory;
        frame_overlap_op(follows);
    }
    if (!p.uses_push) return;   // (most programs read none: a push a draw was ~5% of the replay's commands)
    FINE_N(FC_PUSHC, 1);
    uint8_t data[kPushSize]{};
    const float one = 1.0f;
    std::memcpy(data + Shader::Backend::SPIRV::RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, &one, 4);
    {   // (guest memory's page table, for the pointers compute, task and mesh stages follow)
        static const uint64_t pages = memory_guest_pages();
        std::memcpy(data + Shader::Backend::SPIRV::GUEST_PAGES_PUSH_OFFSET, &pages, 8);
    }
    if (p.fallback_task) {
        std::memcpy(data + Shader::Backend::SPIRV::MESH_FIRST_PUSH_OFFSET, &fallback_first, sizeof fallback_first);
    } else if (at && mesh_args_ready()) {   // (a task program's firsts: entry 0's 0 until its draw says otherwise, never address 0)
        const uint64_t firsts = mesh_args_address + 12;
        std::memcpy(data + Shader::Backend::SPIRV::MESH_FIRST_PUSH_OFFSET, &firsts, 8);
    }
    if (p.fallback) {   // (the fallback's records and payloads, pushed as addresses)
        std::memcpy(data + Shader::Backend::SPIRV::MESH_FALLBACK_PAYLOAD_PUSH_OFFSET, &fallback_payload_address, 8);
        std::memcpy(data + Shader::Backend::SPIRV::MESH_FALLBACK_RECORDS_PUSH_OFFSET, &fallback_records_address, 8);
        std::memcpy(data + Shader::Backend::SPIRV::MESH_FALLBACK_INDIRECT_INPUT_PUSH_OFFSET, &fallback_indirect_input, 8);
    }
    if (p.uses_render_area && at) {
        // (at the block's own offset, 0, as Eden pushes it: the SPIR-V back end declares RenderAreaInfo at offset 0,
        // over the rescaling words. Placed after RescalingLayout, the shaders read zeros: every full-screen pass
        // that takes its texel size from it (SMAA's edges, weights and blending) sampled at infinity, and the
        // title's SMAA flagged 94% of the picture as edges: the red and streaked frames)
        const float ra[4] = {(float)at->width, (float)at->height, 0.0f, 0.0f};
        std::memcpy(data + Shader::Backend::SPIRV::RENDERAREA_LAYOUT_OFFSET, ra, sizeof ra);
    }
    vkCmdPushConstants(cmd, p.layout, p.push_stages, 0, kPushSize, data);
}

} // namespace

void device_fault_report() {
    if (!dev.device_fault || !vkGetDeviceFaultInfoEXT) {
        logf("vk: device fault: VK_EXT_device_fault not available");
        return;
    }
    VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    if (const VkResult r = vkGetDeviceFaultInfoEXT(dev.device, &counts, nullptr); r != VK_SUCCESS) {
        logf("vk: device fault: no information (%d)", (int)r);
        return;
    }
    std::vector<VkDeviceFaultAddressInfoEXT> addrs(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor(counts.vendorInfoCount);
    VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
    info.pAddressInfos = addrs.data(), info.pVendorInfos = vendor.data();
    counts.vendorBinarySize = 0;
    if (vkGetDeviceFaultInfoEXT(dev.device, &counts, &info) != VK_SUCCESS) return;
    logf("vk: device fault: %s", info.description);
    static const char* kinds[] = {"none", "read invalid", "write invalid", "execute invalid", "instruction pointer unknown",
                                  "instruction pointer invalid", "instruction pointer fault"};
    for (const auto& a : addrs)
        logf("vk:   %s at %#llx (precision %#llx)", (unsigned)a.addressType < 7 ? kinds[a.addressType] : "?",
             (unsigned long long)a.reportedAddress, (unsigned long long)a.addressPrecision);
    for (const auto& v : vendor)
        logf("vk:   vendor: %s (code %#llx, data %#llx)", v.description, (unsigned long long)v.vendorFaultCode,
             (unsigned long long)v.vendorFaultData);
}

void checkpoint_mark(VkCommandBuffer cmd, uint64_t key, char kind) {
    if (!checkpoints_on()) return;
    const uint32_t seq = checkpoint_seq(), i = (seq - 1) & 0xFFFF;   // (see checkpoint_seq)
    checkpoints[i] = {key, texture_frame_no(), seq, kind};
    if (checkpoint_draws) checkpoint_draws[i] = CheckpointDraw{};
    set_checkpoint(cmd, i, seq);
}

namespace { void buffer_markers_watch(); }

void buffer_markers_init() {
    if (!vkCmdWriteBufferMarkerAMD) return;
    const VkDeviceSize size = 8ull << 16;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buffer{};
    if (vkCreateBuffer(dev.device, &bi, nullptr, &buffer) != VK_SUCCESS) return;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev.device, buffer, &req);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (int pass = 0; pass < 2 && type == UINT32_MAX; pass++)   // (system memory first: it outlasts a reset)
        for (uint32_t i = 0; i < dev.memory.memoryTypeCount && type == UINT32_MAX; i++) {
            const VkMemoryPropertyFlags f = dev.memory.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits >> i & 1) && (f & want) == want && (pass || !(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
                type = i;
        }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    VkDeviceMemory memory{};
    void* words = nullptr;
    if (type == UINT32_MAX || vkAllocateMemory(dev.device, &ai, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(dev.device, buffer, nullptr);
        logf("vk: RX_VK_CHECKPOINTS: no host memory for the buffer markers");
        return;
    }
    vkBindBufferMemory(dev.device, buffer, memory, 0);
    if (vkMapMemory(dev.device, memory, 0, size, 0, &words) != VK_SUCCESS) {
        vkDestroyBuffer(dev.device, buffer, nullptr);
        vkFreeMemory(dev.device, memory, nullptr);
        return;
    }
    std::memset(words, 0, size);
    checkpoint_draws.reset(new CheckpointDraw[1 << 16]);
    marker_buffer = buffer;
    marker_words = (volatile uint32_t*)words;
    logf("vk: RX_VK_CHECKPOINTS: VK_AMD_buffer_marker, a marker before each draw and dispatch (memory type %u)", type);
    buffer_markers_watch();
}

namespace {
// (the markers the GPU reached whose earlier work never finished: the hang is in the work after the last marker that did)
void buffer_markers_report() {
    const uint32_t n = checkpoint_n.load(), first = n > 0xFFFF ? n - 0xFFFF : 1;
    uint32_t reached = 0, last_done = 0;
    std::vector<uint32_t> stuck;
    // (0: not reached; 1: reached, the work before it not finished; 2: the work before it finished)
    const auto state = [](uint32_t s) -> int {
        const uint32_t i = (s - 1) & 0xFFFF;
        return marker_words[2 * i + 1] == s ? 2 : marker_words[2 * i] == s ? 1 : 0;
    };
    for (uint32_t s = first; s && s <= n; s++) {
        const int st = state(s);
        reached += st != 0;
        if (st == 2) last_done = s;
        else if (st == 1) stuck.push_back(s);
    }
    logf("vk: device lost: %u markers set, %u reached; the last whose earlier work all finished: %u", n, reached, last_done);
    const auto line = [](const char* what, uint32_t s) {
        const Checkpoint& k = checkpoints[(s - 1) & 0xFFFF];
        logf("vk:   %s: marker %u (%s %016llx, frame %llu)", what, s, checkpoint_kind(k.kind), (unsigned long long)k.prog,
             (unsigned long long)k.frame);
    };
    for (uint32_t s = last_done > 6 ? last_done - 6 : 1; last_done && s < last_done; s++) line("before", s);
    // (from the last finished marker to the first reached one whose earlier work did not finish: what the GPU was running)
    const uint32_t start = last_done ? last_done : first;
    const uint32_t end = std::max(start, stuck.empty() ? start + 8 : std::min(stuck.front(), start + 16));
    for (uint32_t s = start; s && s <= end && s <= n; s++) line(s == last_done ? "RUNNING after" : "after", s);
    for (size_t j = 0; j < stuck.size() && j < 12; j++) line("reached, earlier work unfinished", stuck[j]);
}

/// (issue 14) The markers read while the GPU is stuck, not only once the device is lost. On a Steam Deck the ring timed out
/// and the submission never came back (no VK_ERROR_DEVICE_LOST, so checkpoints_report never ran), though the markers sit
/// in host memory the CPU can read at any time. A thread looks at them four times a second: when markers the GPU reached
/// have work unfinished and no marker has finished for RX_VK_CHECKPOINTS_STALL_S seconds (default 2: under amdgpu's
/// lockup timeout, so before the reset), buffer_markers_report is logged then, once a stall (three at most).
/// RX_VK_CHECKPOINTS_STALL_S=0: no watch, the report at the loss alone as before
struct MarkerScan {
    uint32_t done = 0, last_done = 0, pending = 0, n = 0;   // (done: in the last 65535; pending: reached, earlier work unfinished)
};
MarkerScan marker_scan() {
    MarkerScan m;
    m.n = checkpoint_n.load();
    const uint32_t first = m.n > 0xFFFF ? m.n - 0xFFFF : 1;
    for (uint32_t s = first; s && s <= m.n; s++) {
        const uint32_t i = (s - 1) & 0xFFFF;
        const bool r = marker_words[2 * i] == s, d = marker_words[2 * i + 1] == s;
        if (d) m.done++, m.last_done = s;
        else if (r) m.pending++;
    }
    return m;
}
// (the watch's call on each look: true when the GPU has been stuck stall_s seconds, then not again until a marker finishes.
// Progress is the GPU's alone: more markers finished, or a later one. The CPU recording more is not (it can go on a while
// with the GPU stuck), and the markers it records push the oldest out of the 65535, so fewer finished is not either)
struct MarkerStall {
    double stall_s = 2.0, since = 0;
    MarkerScan last;
    bool told = false;
    bool step(const MarkerScan& m, double now) {
        const bool moved = m.done > last.done || m.last_done > last.last_done;
        last = m;
        if (moved || !m.pending) {   // (progress, or nothing under way: the clock starts again)
            since = now, told = false;
            return false;
        }
        if (told || now - since < stall_s) return false;
        return told = true;
    }
};
double marker_stall_seconds() {
    const char* e = getenv("RX_VK_CHECKPOINTS_STALL_S");
    return e ? atof(e) : 2.0;
}
void buffer_markers_watch() {
    const double stall_s = marker_stall_seconds();
    if (!(stall_s > 0) || !marker_words) return;
    logf("vk: RX_VK_CHECKPOINTS: the markers are logged too when the GPU finishes none for %.1f s (RX_VK_CHECKPOINTS_STALL_S)",
         stall_s);
    std::thread([stall_s] {
        SetThreadDescription(GetCurrentThread(), L"mkw_vk marker watch");
        MarkerStall w;
        w.stall_s = stall_s;
        const auto t0 = std::chrono::steady_clock::now();
        for (int reports = 0; reports < 3;) {
            Sleep(250);
            const MarkerScan m = marker_scan();
            const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!w.step(m, now)) continue;
            reports++;
            logf("vk: GPU stalled: no marker finished for %.1f s, %u reached with earlier work unfinished (the markers now, "
                 "before a reset)", now - w.since, m.pending);
            buffer_markers_report();
        }
    }).detach();
}
} // namespace

void draw_records_log(uint64_t lo, uint64_t hi);
bool pass_trace_on();   // (below)
// (hookshot) RX_VK_PASS_TRACE=1: when the GPU hangs, the last frames' passes and dispatches in stream order go to the log,
// once (Hookshot's pass_traces_log): the pass the hung draw was in, and what wrote the textures it read
static void hang_pass_trace() {
    static std::atomic<bool> written{false};
    if (!pass_trace_on() || written.exchange(true)) return;
    const uint64_t f = texture_frame_no();
    pass_traces_log(f > 7 ? f - 7 : 0, f);
}
void device_lost_notice();   // (vk_main.cpp)
void checkpoints_report() {
    static bool done;
    if (done) return;
    done = true;
    device_lost_notice();
    if (env_on("RX_VK_FAULT_INFO")) device_fault_report();   // (now, while the process is still alive: the faulting address)
    {   // (RX_VK_FLASH_RECORD=<program,...>: those programs' last draws, sizes and all)
        const uint64_t f = texture_frame_no();
        draw_records_log(f > 6 ? f - 6 : 0, f);
    }
    hang_pass_trace();
    // (checkpoints first: on NVIDIA vkGetDeviceFaultInfoEXT has not come back after a fault, which lost the report)
    // The host's crash/freeze records read this prefix (host/hos_report_prefixes.h); keep it stable.
    logf("vk: device lost: reading %s", dev.checkpoints ? "the checkpoints"
                                        : marker_words ? "the buffer markers"
                                                       : "the fault info (RX_VK_CHECKPOINTS=1 for markers)");
    struct FaultLast { ~FaultLast() { if ((!dev.checkpoints && !marker_words) || env_on("RX_VK_FAULT_INFO")) device_fault_report(); } } fault_last;
    const auto watched = [] {
        std::lock_guard lk(watch_mu);
        for (int i = 0; i < 40; i++)
            if (!watch_ring[(watch_at + i) % 40].empty()) logf("%s", watch_ring[(watch_at + i) % 40].c_str());
    };
    if (marker_words) {
        buffer_markers_report();
        watched();
        return;
    }
    if (!dev.checkpoints || !vkGetQueueCheckpointDataNV) return;
    uint32_t n = 0;
    vkGetQueueCheckpointDataNV(dev.queue, &n, nullptr);
    std::vector<VkCheckpointDataNV> cps(n, VkCheckpointDataNV{VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
    vkGetQueueCheckpointDataNV(dev.queue, &n, cps.data());
    logf("vk: device lost: %u checkpoints (%u markers set)", n, checkpoint_n.load());
    watched();
    for (const VkCheckpointDataNV& c : cps) {
        const uint32_t i = (uint32_t)(uintptr_t)c.pCheckpointMarker - 1;
        if (i >= (1u << 16)) continue;
        const Checkpoint& k = checkpoints[i];
        const auto kind = checkpoint_kind;
        logf("vk:   stage %#x: marker %u (%s %016llx, frame %llu)", (unsigned)c.stage, k.seq, kind(k.kind),
             (unsigned long long)k.prog, (unsigned long long)k.frame);
        // (the ones before it, for context)
        for (uint32_t b = 1; b <= 6; b++) {
            const Checkpoint& o = checkpoints[(i - b) & 0xFFFF];
            logf("vk:     before: %s %016llx", kind(o.kind), (unsigned long long)o.prog);
        }
        // (and the ones after it: what the GPU never reached)
        for (uint32_t b = 1; b <= 8; b++) {
            const Checkpoint& o = checkpoints[(i + b) & 0xFFFF];
            logf("vk:     after: marker %u %s %016llx, frame %llu", o.seq, kind(o.kind), (unsigned long long)o.prog, (unsigned long long)o.frame);
        }
    }
}

namespace {

// ---- the open render pass ----
struct Pass {
    bool open = false;
    VkImageView color[8]{};
    VkImage block_image[8]{};
    bool has_block_images = false;
    Texture* tex[8]{};   // (RX_VK_DUMP: what the pass renders into, and the last program it drew with)
    uint32_t level[8]{}, layer[8]{};
    Texture* depth_tex = nullptr;
    uint32_t depth_level = 0, depth_layer = 0;
    uint64_t prog = 0;
    VkImageView depth = VK_NULL_HANDLE;
    int n = 0;
    uint32_t width = 0, height = 0;
    int draws = 0;           // (RX_VK_GPU_PROFILE: its draws, and its label's targets)
    uint32_t wm_or = 0, wm_n = 0;   // (RX_VK_PASS_TRACE: the colour write masks its draws set, target 0, and the targets)
    int clears = 0;          // (RX_VK_PASS_TRACE: its clears, and the first program it drew with)
    uint64_t first_prog = 0;
    VkFormat format0 = VK_FORMAT_UNDEFINED;
    std::vector<const Texture*> sampled;   // First six distinct samples for pass tracing only.
    uint16_t sampled_targets = 0;          // (colour i bit i, depth bit 8: targets earlier draws sampled)
    uint16_t written = 0;                  // (its targets written so far: colour i bit i, depth bit 8)
};
TLS_OBJECT(pass, Pass)
thread_local bool work_since_barrier = false;   // (GPU work recorded since the last full barrier: a dispatch orders after it)
thread_local VkCommandBuffer last_dispatch_cmd = VK_NULL_HANDLE;   // (the last direct or indirect dispatch's command buffer, and the
thread_local uint64_t last_dispatch_epoch = 0;                     // work count right after it: game_order's back-to-back test)
// what the command buffer has bound (draws skip what is already there): CmdBound, below, with the types it is made of
/// The descriptors the graphics bind point has: what they were built from
struct DescBound {
    bool valid = false;
    const Program* prog = nullptr;
    uint64_t bind = 0, reg = 0, ub = 0, dgen = 0;
    // (Descriptor reuse: the versions of the slots the last build resolved, the samplers' generation it saw, and whether it
    // sampled an unregistered slot; the same draw keeps its descriptors while none of those moved)
    std::vector<uint64_t> fp;
    bool fp_dummy = false;
    bool fp_on = false;   // (the last build recorded its handles: one made with the row off has none to check)
    uint64_t samp = 0, destroy = 0;
};
int64_t st_desc_slotgen_saves;   // (reuses the global registrations' generation check would have rebuilt for)
struct LastVb { VkBuffer b[16]; VkDeviceSize o[16], z[16], s[16]; };
// (the guest streams and index buffer already bound on this command buffer; a new recording drops them)
struct VertexSet {
    uint64_t stamp = 0;
    uint64_t addr[16]{};
    uint64_t size[16]{};
    uint32_t stride[16]{};
    bool on = false;
};
struct IndexBind {
    uint64_t buffer = 0, offset = 0;
    uint32_t type = 0;
    bool on = false;
};
SplitCount st_desc_reused;   // (split_count.h)

thread_local uint64_t targets_serial = ~0ull;   // (the state's render targets resolved once per NOP_SET_RT)
thread_local Targets cached_targets;

Targets targets(NvnState& s) {
    if (targets_serial == s.rt_serial) return cached_targets;
    ParLock lk;
    Targets at;
    at.n = std::clamp(s.rt.n, 0, 8);
    for (int i = 0; i < at.n; i++) {
        const nvn_tex* t = s.rt.color[i];
        Texture* tex = t ? texture_get(t, false) : nullptr;
        if (tex && !(tex->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) tex = nullptr;
        if (!tex) continue;
        const nvn_view* v = s.rt.has_cview[i] ? &s.rt.cview[i] : nullptr;
        const auto [view, f, blocks] = texture_attachment(tex, t, v);
        if (!view) continue;
        at.color[i] = tex, at.color_view[i] = view, at.color_format[i] = f, at.color_view_nvn[i] = view_nvn_format(t, v);
        at.block_image[i] = blocks;
        at.has_block_images |= blocks != VK_NULL_HANDLE;
        const uint32_t level = image_level(tex, view_level(v));
        at.color_level[i] = level, at.color_layer[i] = v ? view_layer(v) : 0;
        uint32_t w = std::max(1u, tex->width >> level), h = std::max(1u, tex->height >> level);
        // (a raw view of a compressed target is one texel a block: its extent in blocks. The texels' 64 x 64 over the 16 x 16
        // view of OoT's noise volumes put the render area past the attachment)
        const auto bc = [](VkFormat x) { return x >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && x <= VK_FORMAT_BC7_SRGB_BLOCK; };
        if (bc(tex->format) && !bc(f)) w = (w + 3) / 4, h = (h + 3) / 4;
        at.width = at.width ? std::min(at.width, w) : w;
        at.height = at.height ? std::min(at.height, h) : h;
        at.samples = tex->samples;
    }
    // (a texture at two slots: the later one left unattached, as nxnative does for BotW's G-buffer)
    for (int i = 0; i < at.n; i++)
        for (int j = 0; j < i; j++)
            if (at.color_view[i] && at.color_view[i] == at.color_view[j]) {
                at.color[i] = nullptr, at.color_view[i] = VK_NULL_HANDLE, at.color_format[i] = VK_FORMAT_UNDEFINED, at.color_view_nvn[i] = 0;
                at.block_image[i] = VK_NULL_HANDLE;
                break;
            }
    while (at.n > 0 && !at.color[at.n - 1]) at.n--;   // (trailing empty targets)
    if (s.rt.depth) {
        Texture* tex = texture_get(s.rt.depth, false);
        if (tex && (tex->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) {
            const nvn_view* v = s.rt.has_dview ? &s.rt.dview : nullptr;
            VkFormat f = VK_FORMAT_UNDEFINED;
            const VkImageView view = texture_view(tex, s.rt.depth, v, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Attachment, &f);
            if (view) {
                at.depth = tex, at.depth_view = view, at.depth_format = tex->format;
                const uint32_t level = image_level(tex, view_level(v));
                at.depth_level = level, at.depth_layer = v ? view_layer(v) : 0;
                const uint32_t w = std::max(1u, tex->width >> level), h = std::max(1u, tex->height >> level);
                at.width = at.width ? std::min(at.width, w) : w;
                at.height = at.height ? std::min(at.height, h) : h;
                if (!at.n) at.samples = tex->samples;
            }
        }
    }
    targets_serial = s.rt_serial;
    cached_targets = at;
    return at;
}

/// A draw with no target at all (a fragment shader that only writes storage): the pass is the viewport's area
Targets bare_targets(const NvnState& s, Targets at) {
    if (at.n || at.depth || at.width) return at;
    at.width = (uint32_t)std::max(1.0f, s.viewport[0][0] + s.viewport[0][2]);
    at.height = (uint32_t)std::max(1.0f, s.viewport[0][1] + s.viewport[0][3]);
    at.width = std::min(at.width, 16384u), at.height = std::min(at.height, 16384u);
    return at;
}

/// Whether the targets of a draw kept a picture from an earlier frame (a draw left out of them shows for one frame)
/// (hookshot, 17/11, issue 47 and the Discord's striped ground) Whether a draw left out of these targets is drawn again in the next
/// frames: every target was drawn in the frame before (the scene's own, redrawn each frame) and the draw covers it, its viewport
/// and scissor over nine tenths of it. Not a part the game draws once into a target it keeps: a tile, a decal, a shadow of a
/// cache it updates a piece at a time. Left out, such a piece was never drawn again, and that part of the target kept what was
/// there before until the area was loaded again: rectangular patches of the wrong texels over the ground, a few minutes into a
/// place never seen in this session (Flix's route in the Fire Temple, every time after a fresh start; logs with draws left out
/// in all four reports). Those wait for their pipeline instead
bool targets_redrawn(const Targets& at, const NvnState& s) {
    const uint64_t now = texture_frame_no();
    const auto recent = [&](const Texture* t) { return !t || t->last_written + 1 >= now; };
    for (int i = 0; i < at.n; i++)
        if (!recent(at.color[i])) return false;
    if (!recent(at.depth)) return false;
    if (!at.width || !at.height) return false;
    const float w = (float)at.width * 0.9f, h = (float)at.height * 0.9f;
    if (std::fabs(s.viewport[0][2]) < w || std::fabs(s.viewport[0][3]) < h) return false;
    if (s.scissor[2] > 0 && s.scissor[3] > 0 && ((float)s.scissor[2] < w || (float)s.scissor[3] < h)) return false;
    return true;
}
bool targets_persistent(const Targets& at) {
    const uint64_t now = texture_frame_no();
    for (int i = 0; i < at.n; i++)
        if (at.color[i] && at.color[i]->first_render_frame && at.color[i]->first_render_frame < now) return true;
    return at.depth && at.depth->first_render_frame && at.depth->first_render_frame < now;
}

void begin_pass(VkCommandBuffer cmd, const Targets& at) {
    Pass& ps = pass();   // (once: the accessor is a call and a thread-local read at each use. The same object for the thread's life: draw_begin_recording resets it in place)
    bool same = ps.open && ps.depth == at.depth_view && ps.n == at.n && ps.width == at.width &&
                ps.height == at.height;
    for (int i = 0; i < at.n && same; i++) same = ps.color[i] == at.color_view[i];
    if (same) return;
    end_pass(cmd);
    if (at.has_block_images) {
        full_barrier(cmd, "barrier (before BC slice loads)");
        for (int i = 0; i < at.n; i++)
            if (at.block_image[i])
                copy_block_attachment(cmd, *at.color[i], at.block_image[i], at.color_level[i], at.color_layer[i], true);
        full_barrier(cmd, "barrier (after BC slice loads)");
    } else {
        pass_start_barrier(cmd);   // (what the pass before wrote, and the transfers before it: vk_texture.cpp)
    }
    work_since_barrier = true;
    VkRenderingAttachmentInfo ca[8]{};
    bool marks = !texture_marked(at.depth);
    for (int i = 0; i < at.n && !marks; i++) marks = !texture_marked(at.color[i]);
    ParLock lk(marks);
    for (int i = 0; i < at.n; i++) {
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        ca[i].imageView = at.color_view[i];
        ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ps.color[i] = at.color_view[i];
        ps.block_image[i] = at.block_image[i];
        ps.tex[i] = at.color[i], ps.level[i] = at.color_level[i], ps.layer[i] = at.color_layer[i];
        texture_written(at.color[i], "a pass's colour target");
        texture_rendered(at.color[i]);
    }
    for (int i = at.n; i < 8; i++) ps.color[i] = VK_NULL_HANDLE;
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    if (at.depth) {
        da.imageView = at.depth_view;
        da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        texture_written(at.depth, "a pass's depth target");
        texture_rendered(at.depth);
    }
    ps.depth = at.depth_view;
    ps.has_block_images = at.has_block_images;
    ps.depth_tex = at.depth, ps.depth_level = at.depth_level, ps.depth_layer = at.depth_layer;
    ps.n = at.n, ps.width = at.width, ps.height = at.height;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {at.width, at.height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = (uint32_t)at.n;
    ri.pColorAttachments = ca;
    if (at.depth && depth_aspect(at.depth_format)) ri.pDepthAttachment = &da;
    if (at.depth && has_stencil(at.depth_format)) ri.pStencilAttachment = &da;
    gprof_begin(cmd);
    ps.draws = 0, ps.format0 = at.n ? at.color_format[0] : at.depth_format;
    ps.wm_or = 0, ps.wm_n = 0;
    ps.clears = 0, ps.first_prog = 0;
    ps.sampled.clear(), ps.sampled_targets = 0, ps.written = 0;
    vkCmdBeginRendering(cmd, &ri);
    ps.open = true;
    st_passes++;
}

/// The dynamic state a draw sets, compared with what the command buffer has (only what changed is set again: most
/// draws repeat their neighbours' state, and these calls were a third of the replay's command time)
struct DynState {
    VkViewport vp;
    VkRect2D sc;
    float blend[4];
    uint32_t sref[2], scmp[2], swm[2];
    float bias[3];
    VkBool32 bias_en;
    VkCullModeFlags cull;
    VkFrontFace front;
    VkPrimitiveTopology topo;
    VkBool32 restart, dt, dw, st, discard, clamp, a2c;
    VkCompareOp dop;
    VkStencilOp sops[2][3];
    VkCompareOp scmpop[2];
    VkPolygonMode poly;
    int n;
    VkBool32 en[8];
    VkColorBlendEquationEXT eq[8];
    VkColorComponentFlags wm[8];
};
// (cb: the thread's CmdBound, in dynamic_state)
#define DYN_CHANGED(f) ((!cb.dyn_valid || std::memcmp(&d.f, &cb.dyn.f, sizeof d.f)) && (FINE_X(FC_DYN_CALLS, 1), true))

constexpr int kDynF = 13, kDynU = 35;
struct DynFast {
    bool on = false;
    float fp[kDynF]{};
    uint32_t up[kDynU]{};
    uint8_t raw[160]{};
    int nb = 0;
    uint32_t wm0 = 0, wm_n = 0;
};
/// What this thread's command buffer has bound, which the next draw does not bind or set again: the pipeline, the vertex
/// streams, the index buffer, the dynamic state and the graphics descriptors. One object a recording thread (bound()), so
/// that what forgets it is one call. (The pushed descriptors' words are pushed(), forgotten by descriptors_disturbed(), which
/// the other files call; the buffers kept_vb resolved are no binding and last a submission.)
struct CmdBound {
    VkPipeline pipeline = VK_NULL_HANDLE;
    LastVb vb{};
    bool vb_valid = false;
    VertexSet vertex_set;   // (the guest streams bound: a draw with the same ones looks no buffer up)
    IndexBind ib;
    DynState dyn{};
    bool dyn_valid = false;
    DynFast dyn_fast;       // (the inputs dyn was made from; read only while dyn_valid)
    DescBound desc;
    /// Commands recorded by something else bound a pipeline and set state of their own, but no index buffer (a masked
    /// clear): the next draw binds its pipeline, vertex streams and descriptors and sets its dynamic state again
    void reset_keeping_index() {
        dyn_valid = false, desc.valid = false, vb_valid = false, pipeline = VK_NULL_HANDLE;
        vertex_set.on = false;   // (the next draw binds its vertex streams again, as vb_valid above asks)
    }
    /// A new command buffer, or commands recorded by something else that may have bound anything (an upscaler's stand-in):
    /// nothing is known to be bound
    void reset() {
        reset_keeping_index();
        ib.on = false;
    }
};
TLS_OBJECT(bound, CmdBound)
/// (Descriptor reuse) Whether every slot the last build resolved still holds what it did: no registration since touched a
/// slot this draw samples, so its descriptors stand through the streaming's churn. A build that sampled an unregistered slot
/// keeps the old check: a registration is the only thing that could make such a handle resolve, and no slot version moved.
bool desc_slots_kept(const DescBound& k, uint64_t reg_gen) {
    if (k.samp != sampler_registrations_generation()) return false;
    if (k.destroy != texture_destroy_generation()) return false;   // (an eviction sweep: destroy_later retires the
    if (k.fp_dummy && k.reg != reg_gen) return false;      // (views without bumping any slot, so they alone stay silent)
    for (const uint64_t e : k.fp)
        if (texture_slot_version((int)(e & 0xFFFFF)) != (uint32_t)(e >> 20)) return false;
    return true;
}
/// wm_sig: what the draw's program makes of the write masks (program_wm_sig)
int dyn_pack(const NvnState& s, const Targets& at, VkPrimitiveTopology topo, bool mesh, uint64_t wm_sig, float* fp, uint32_t* up,
             uint8_t* raw) {
    fp[0] = s.viewport[0][0], fp[1] = s.viewport[0][1], fp[2] = s.viewport[0][2], fp[3] = s.viewport[0][3];
    fp[4] = s.depth_near, fp[5] = s.depth_far;
    std::memcpy(fp + 6, s.blend_color, sizeof s.blend_color);
    fp[10] = s.poly_offset[0], fp[11] = s.poly_offset[1], fp[12] = s.poly_offset[2];
    up[0] = (uint32_t)s.scissor[0], up[1] = (uint32_t)s.scissor[1], up[2] = (uint32_t)s.scissor[2], up[3] = (uint32_t)s.scissor[3];
    up[4] = s.stencil_ref[0], up[5] = s.stencil_ref[1];
    up[6] = s.stencil_value_mask[0], up[7] = s.stencil_value_mask[1];
    up[8] = s.stencil_mask[0], up[9] = s.stencil_mask[1];
    up[10] = at.width, up[11] = at.height, up[12] = (uint32_t)at.n, up[13] = (uint32_t)at.depth_format;
    up[14] = at.depth != nullptr;
    up[15] = (uint32_t)s.color.blend_en, up[16] = s.chmask.mask;
    up[17] = (uint32_t)topo, up[18] = mesh ? 1u : 0u;
    up[19] = s.restart ? 1u : 0u, up[20] = s.raster_discard ? 1u : 0u, up[21] = s.depth_clamp ? 1u : 0u, up[22] = s.ms.a2c ? 1u : 0u;
    up[23] = (uint32_t)s.poly.cull, up[24] = (uint32_t)s.poly.front, up[25] = (uint32_t)s.poly.mode, up[26] = (uint32_t)s.poly.offset_en;
    for (int i = 0; i < 8; i++) up[27 + i] = (uint32_t)at.color_format[i];
    static_assert(sizeof(s.ds) + sizeof(s.blend) + 8 <= 160);
    int atb = 0;
    std::memcpy(raw + atb, &s.ds, sizeof s.ds);
    atb += (int)sizeof s.ds;
    std::memcpy(raw + atb, s.blend, sizeof s.blend);
    atb += (int)sizeof s.blend;
    // (hookshot) what the program makes of the write masks, not the program: draws change program far more often than they change that,
    // and each program change worked the whole state out again)
    std::memcpy(raw + atb, &wm_sig, 8);
    return atb + 8;
}

/// p: the draw's program, for its dynamic write masks
void dynamic_state(VkCommandBuffer cmd, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, bool mesh, const Program& p) {
    Pass& ps = pass();   // (once: the accessor is a call and a thread-local read at each use. The same object for the thread's life: draw_begin_recording resets it in place)
    CmdBound& cb = bound();   // (once, as the pass)
    float df[kDynF];
    uint32_t du[kDynU];
    uint8_t db[160];
    const int nb = dyn_pack(s, at, topo, mesh, program_wm_sig(p), df, du, db);
    if (!dump_on() && cb.dyn_valid && cb.dyn_fast.on && cb.dyn_fast.nb == nb &&
        dyn_same(df, kDynF, du, kDynU, db, nb, cb.dyn_fast.fp, cb.dyn_fast.up, cb.dyn_fast.raw)) {
        ps.wm_or |= cb.dyn_fast.wm0, ps.wm_n = cb.dyn_fast.wm_n;   // (the pass trace still sees the masks)
        FINE_N(FC_DYN_FAST, 1);
        return;
    }
    FINE_N(FC_DYN_FULL, 1);
    DynState d;
    std::memset(&d, 0, sizeof d);   // (padding compared too)
    // (RX_VK_FOOTPRINT=<key>,...: those programs' draws write depth wherever they rasterize, both faces, whatever is
    // there already: with RX_VK_DUMP_EACH, each one's footprint (diagnostics))
    static const std::vector<uint64_t> footprint = [] {
        std::vector<uint64_t> v;
        for (const char* q = getenv("RX_VK_FOOTPRINT"); q && *q;) {
            char* end;
            const uint64_t k = strtoull(q, &end, 16);
            if (end == q) break;
            v.push_back(k);
            q = *end ? end + 1 : end;
        }
        return v;
    }();
    const bool fp = dump_on() && at.depth && depth_aspect(at.depth_format) && !footprint.empty() &&
                    std::find(footprint.begin(), footprint.end(), p.key) != footprint.end();
    // the viewport upside down (negative height): NVN's NDC has +y up, Vulkan's down
    float vx = s.viewport[0][0], vy = s.viewport[0][1], vw = s.viewport[0][2], vh = s.viewport[0][3];
    if (vw == 0.0f && vh == 0.0f) vx = 0, vy = 0, vw = (float)at.width, vh = (float)at.height;
    d.vp = {vx, vy + vh, vw, -vh, std::clamp(s.depth_near, 0.0f, 1.0f), std::clamp(s.depth_far, 0.0f, 1.0f)};
    d.sc = {{0, 0}, {at.width, at.height}};
    if (s.scissor[2] > 0 && s.scissor[3] > 0) {
        const int32_t x0 = std::max(0, s.scissor[0]), y0 = std::max(0, s.scissor[1]);
        const int32_t x1 = std::min<int32_t>(s.scissor[0] + s.scissor[2], (int32_t)at.width);
        const int32_t y1 = std::min<int32_t>(s.scissor[1] + s.scissor[3], (int32_t)at.height);
        d.sc = {{x0, y0}, {(uint32_t)std::max(0, x1 - x0), (uint32_t)std::max(0, y1 - y0)}};
    }
    std::memcpy(d.blend, s.blend_color, sizeof d.blend);
    for (int f = 0; f < 2; f++)
        d.sref[f] = s.stencil_ref[f] & 0xFF, d.scmp[f] = s.stencil_value_mask[f] & 0xFF, d.swm[f] = s.stencil_mask[f] & 0xFF;
    d.bias[0] = s.poly_offset[1], d.bias[1] = s.poly_offset[2], d.bias[2] = s.poly_offset[0];
    d.bias_en = s.poly.offset_en != 0;
    {   // (hookshot) The characters' own shadow maps (square, depth only, 768 and less; 1e26755d projects them on the screen): their
        // stored depth is moved away from the light, or they shadow themselves whole: Link and the ferns black in the shade, while
        // the ground behind them keeps their shadow. RX_VK_SHADOW_BIAS=<units>,<slope>; 0: as the game asks
        // (read once, thread-safely: parallel recording threads reach here at once)
        static float units = 250000.0f, slope = 0.0f;
        static const bool bias_read = [] {
            if (const char* e = getenv("RX_VK_SHADOW_BIAS")) sscanf(e, "%f,%f", &units, &slope);
            return true;
        }();
        (void)bias_read;
        // (hookshot, an experiment) The broken pots' shards go dark in blocks with straight screen-aligned edges, the ground
        // around them lit: a shadow projected over a screen rectangle, the shards shadowing themselves inside it, as Link did before
        // this rule. If the shards' maps are outside it, these take them: RX_VK_SHADOW_BIAS_MAX=<px>, the largest side of a depth
        // target it applies to (768); RX_VK_SHADOW_BIAS_ANY=1: not square too, 2: and with colour targets (main passes too: their
        // EQUAL depth tests may break). The log names each target they add it to that the rule above does not
        static const uint32_t max_side =
            getenv("RX_VK_SHADOW_BIAS_MAX") ? (uint32_t)std::max(1, atoi(getenv("RX_VK_SHADOW_BIAS_MAX"))) : 768u;
        static const int any_shape = getenv("RX_VK_SHADOW_BIAS_ANY") ? atoi(getenv("RX_VK_SHADOW_BIAS_ANY")) : 0;
        const bool rule = at.n == 0 && at.width == at.height && at.width <= 768 && at.depth;
        const bool wider = at.depth && std::max(at.width, at.height) <= max_side && (at.n == 0 || any_shape >= 2) &&
                           (at.width == at.height || any_shape >= 1);
        if (units > 0 && (rule || wider)) {
            d.bias[0] += units, d.bias[2] += slope;
            d.bias_en = true;
            if (!rule) {
                static std::mutex m;
                static std::unordered_set<uint64_t> told;
                const uint64_t k = (uint64_t)at.width << 40 ^ (uint64_t)at.height << 16 ^ (uint64_t)at.depth_format << 4 ^ (uint64_t)at.n;
                std::lock_guard lk(m);
                if (told.size() < 64 && told.insert(k).second)
                    logf("vk: shadow bias also added (RX_VK_SHADOW_BIAS_MAX / _ANY) to a depth target %ux%u format %d with %d colour "
                         "targets, program %016llx (the game's bias %s, units %g slope %g)", at.width, at.height, (int)at.depth_format,
                         at.n, (unsigned long long)p.key, s.poly.offset_en ? "on" : "off",
                         s.poly_offset[1], s.poly_offset[0]);
            }
        }
        // (amd-fix, 11-07) the sun's shadow cascades (square, depth only, past 768): RX_VK_CASCADE_BIAS=<units>,<slope> added;
        // RX_VK_LOG_CASCADE_BIAS=1: what the game asks for them
        if (at.n == 0 && at.width == at.height && at.width > 768 && at.depth) {
            static float cu = 0, cs = 0;
            static const bool parsed = [] {
                if (const char* e = getenv("RX_VK_CASCADE_BIAS")) sscanf(e, "%f,%f", &cu, &cs);
                return true;
            }();
            (void)parsed;
            static const bool log_on = env_on("RX_VK_LOG_CASCADE_BIAS");
            static int logged;
            if (log_on && logged < 40) {
                logged++;
                logf("vk: cascade %ux%u bias: enabled %d units %g clamp %g slope %g", at.width, at.height, (int)d.bias_en,
                     d.bias[0], d.bias[1], d.bias[2]);
            }
            if (cu != 0 || cs != 0) d.bias[0] += cu, d.bias[2] += cs, d.bias_en = true;
        }
    }
    {   // (hookshot, a diagnosis) RX_VK_LOG_DEPTH_TARGETS=1: each depth-only target a program draws into, once per size,
        // format, program and bias: the game's own depth bias and the one drawn with (the characters' shadow maps above). Which maps
        // the broken pots' shards are drawn into, and whether the rule above takes them
        static const bool on = getenv("RX_VK_LOG_DEPTH_TARGETS") && atoi(getenv("RX_VK_LOG_DEPTH_TARGETS")) != 0;
        if (on && at.n == 0 && at.depth) {
            const uint64_t prog = p.key;
            const uint64_t k = prog * 0x9e3779b97f4a7c15ull ^ (uint64_t)at.width << 44 ^ (uint64_t)at.height << 24 ^
                               (uint64_t)at.depth_format << 2 ^ (uint64_t)(s.poly.offset_en != 0) << 1 ^ (uint64_t)(d.bias_en != 0);
            static std::mutex m;
            static std::unordered_set<uint64_t> seen;
            std::lock_guard lk(m);
            if (seen.size() < 300 && seen.insert(k).second)
                logf("vk: depth-only target %ux%u format %d, program %016llx: the game's bias %s (units %g clamp %g slope %g), "
                     "drawn with bias %s (units %g slope %g)", at.width, at.height, (int)at.depth_format, (unsigned long long)prog,
                     s.poly.offset_en ? "on" : "off", s.poly_offset[1], s.poly_offset[2], s.poly_offset[0], d.bias_en ? "on" : "off",
                     d.bias[0], d.bias[2]);
        }
    }
    d.cull = sw().no_cull ? VK_CULL_MODE_NONE
           : s.poly.cull == NVN_FACE_FRONT ? VK_CULL_MODE_FRONT_BIT
           : s.poly.cull == NVN_FACE_BACK ? VK_CULL_MODE_BACK_BIT
           : s.poly.cull == NVN_FACE_FRONT_AND_BACK ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE;
    // The viewport is flipped (negative height), which reverses the screen-space winding, so NVN's front face maps
    // straight across: telling Vulkan the opposite is what culled every triangle (the frame was black).
    d.front = s.poly.front == NVN_FRONT_FACE_CCW ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    if (fp) d.cull = VK_CULL_MODE_NONE;
    d.topo = mesh ? VK_PRIMITIVE_TOPOLOGY_MAX_ENUM : topo;
    d.restart = !mesh && s.restart && topo != VK_PRIMITIVE_TOPOLOGY_POINT_LIST && topo != VK_PRIMITIVE_TOPOLOGY_LINE_LIST &&
                topo != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST && topo != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    const bool depth = at.depth && depth_aspect(at.depth_format);
    const bool stencil = at.depth && has_stencil(at.depth_format);
    d.dt = !sw().no_depth && depth && s.ds.dt;
    d.dw = !sw().no_depth && depth && s.ds.dw;
    d.dop = compare_op(s.ds.df);
    if (fp) d.dt = true, d.dw = true, d.dop = VK_COMPARE_OP_ALWAYS;
    {   // (hookshot) A colour pass after a depth pass tests EQUAL against what another program wrote. The compiler of Intel's Windows
        // driver does not give two programs the same position to the last bit, and the ground came out in holes: the sky through
        // them, a white forest. A small margin towards the camera instead. It also passes what is in front of the depth pass, which
        // EQUAL would not: not on Mesa's driver, where EQUAL holds. RX_VK_EQ_BIAS=<units>, 0: as the game asks
        // Not for a draw that blends: the game keeps such a draw (a haze sheet across Death Mountain Trail's entrance, program
        // 701ae221d88b2908) where the depth is exactly its own, and LESS_OR_EQUAL drew the whole sheet over the mountain behind:
        // a flat khaki veil on Intel (Arc 155H). The prepass's colour passes the margin was made for do not blend: some have blending
        // on with ONE, ZERO (the same as off), and those keep the margin (without it, Death Mountain Trail's ground had holes).
        static const float eq_bias = dev.policy.equal_depth_bias;   // (device_policy.h: 64 on Intel's Windows driver but on Battlemage, then the switch)
        bool blends = false;
        for (int i = 0; i < at.n && !blends; i++) {
            const nvn_blend_st& b = s.blend[i];
            blends = (s.color.blend_en >> i & 1) && at.color_format[i] && target_write_mask(p, s, i) &&
                     (blend_factor(b.src_rgb) != VK_BLEND_FACTOR_ONE || blend_factor(b.dst_rgb) != VK_BLEND_FACTOR_ZERO);
        }
        if (eq_bias > 0 && d.dt && !d.dw && d.dop == VK_COMPARE_OP_EQUAL && !blends) {
            d.dop = VK_COMPARE_OP_LESS_OR_EQUAL;
            d.bias_en = VK_TRUE;
            d.bias[0] = -eq_bias, d.bias[1] = 0.0f, d.bias[2] = 0.0f;
        }
    }
    d.st = stencil && s.ds.st;
    d.sops[0][0] = stencil_op(s.ds.fail_f), d.sops[0][1] = stencil_op(s.ds.zpass_f), d.sops[0][2] = stencil_op(s.ds.zfail_f);
    d.sops[1][0] = stencil_op(s.ds.fail_b), d.sops[1][1] = stencil_op(s.ds.zpass_b), d.sops[1][2] = stencil_op(s.ds.zfail_b);
    d.scmpop[0] = compare_op(s.ds.sf_f), d.scmpop[1] = compare_op(s.ds.sf_b);
    d.discard = s.raster_discard && !sw().ignore_discard;
    {   // (diagnosis, default off) RX_VK_DISCARD_PROG=<key start>[,...]: those programs' draws run their geometry stages only (rasterizer
        // discard), so with RX_VK_PASS_TIMING=2 the run's time minus the normal run's is the pixel work: the picture is wrong
        static const std::vector<std::string> keys = [] {
            std::vector<std::string> v;
            for (const char* q = getenv("RX_VK_DISCARD_PROG"); q && *q;) {
                const char* e = q;
                while (*e && *e != ',') e++;
                v.emplace_back(q, e);
                q = *e ? e + 1 : e;
            }
            return v;
        }();
        if (!keys.empty()) {
            char k[20];
            snprintf(k, sizeof k, "%016llx", (unsigned long long)p.key);
            for (const std::string& w : keys)
                if (!w.empty() && strncmp(k, w.c_str(), w.size()) == 0) d.discard = VK_TRUE;
        }
    }
    d.poly = s.poly.mode == 0 ? VK_POLYGON_MODE_POINT : s.poly.mode == 1 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    static const bool fp_lines = env_on("RX_VK_FOOTPRINT_LINES");   // (the footprints as wireframes)
    if (fp && fp_lines) d.poly = VK_POLYGON_MODE_LINE;
    d.clamp = s.depth_clamp, d.a2c = s.ms.a2c != 0;
    d.n = dev.eds3_blend ? at.n : 0;
    for (int i = 0; i < d.n; i++) {
        const nvn_blend_st& b = s.blend[i];
        d.en[i] = (s.color.blend_en >> i & 1) && at.color_format[i] ? VK_TRUE : VK_FALSE;
        d.eq[i] = {blend_factor(b.src_rgb), blend_factor(b.dst_rgb), blend_op(b.eq_rgb), blend_factor(b.src_a),
                   blend_factor(b.dst_a), blend_op(b.eq_a)};
        d.wm[i] = target_write_mask(p, s, i);
    }
    if (dump_on()) {   // (RX_VK_DUMP: the state a program draws with, once per change)
        char line[400];
        int k = snprintf(line, sizeof line, "prog %016llx n%d depth %s dt%u dw%u dop%d st%u cull%u front%u vp %.0f,%.0f %.0fx%.0f z %.2f-%.2f sc %d,%d %ux%u:",
                         (unsigned long long)p.key, at.n,
                         at.depth ? (depth ? "yes" : "stencil") : "no", d.dt, d.dw, (int)d.dop, d.st, d.cull, d.front,
                         d.vp.x, d.vp.y, d.vp.width, d.vp.height, d.vp.minDepth, d.vp.maxDepth, d.sc.offset.x, d.sc.offset.y,
                         d.sc.extent.width, d.sc.extent.height);
        for (int i = 0; i < at.n && k < (int)sizeof line - 40; i++)
            k += snprintf(line + k, sizeof line - k, " c%d:%s wm%x en%u %d/%d/%d", i, at.color[i] ? "" : "none", d.wm[i],
                          d.en[i], (int)d.eq[i].srcColorBlendFactor, (int)d.eq[i].dstColorBlendFactor,
                          (int)d.eq[i].colorBlendOp);
        static std::string last;
        if (last != line) last = line, logf("vk: dump-st %03d %s", dump_seq(), line);
    }
    if (DYN_CHANGED(vp)) vkCmdSetViewportWithCount(cmd, 1, &d.vp);
    if (DYN_CHANGED(sc)) vkCmdSetScissorWithCount(cmd, 1, &d.sc);
    if (DYN_CHANGED(blend)) vkCmdSetBlendConstants(cmd, d.blend);
    if (DYN_CHANGED(sref)) {
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_BIT, d.sref[0]);
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_BACK_BIT, d.sref[1]);
    }
    if (DYN_CHANGED(scmp)) {
        vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_BIT, d.scmp[0]);
        vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_BACK_BIT, d.scmp[1]);
    }
    if (DYN_CHANGED(swm)) {
        vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_BIT, d.swm[0]);
        vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_BACK_BIT, d.swm[1]);
    }
    if (DYN_CHANGED(bias)) vkCmdSetDepthBias(cmd, d.bias[0], d.bias[1], d.bias[2]);
    if (DYN_CHANGED(bias_en)) vkCmdSetDepthBiasEnable(cmd, d.bias_en);
    if (!cb.dyn_valid) vkCmdSetLineWidth(cmd, 1.0f), vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
    if (DYN_CHANGED(cull)) vkCmdSetCullMode(cmd, d.cull);
    if (DYN_CHANGED(front)) vkCmdSetFrontFace(cmd, d.front);
    if (!mesh) {
        if (DYN_CHANGED(topo)) vkCmdSetPrimitiveTopology(cmd, d.topo);
        if (DYN_CHANGED(restart) || cb.dyn.topo == VK_PRIMITIVE_TOPOLOGY_MAX_ENUM) vkCmdSetPrimitiveRestartEnable(cmd, d.restart);
    }
    if (DYN_CHANGED(dt)) vkCmdSetDepthTestEnable(cmd, d.dt);
    if (DYN_CHANGED(dw)) vkCmdSetDepthWriteEnable(cmd, d.dw);
    if (DYN_CHANGED(dop)) vkCmdSetDepthCompareOp(cmd, d.dop);
    if (DYN_CHANGED(st)) vkCmdSetStencilTestEnable(cmd, d.st);
    if (DYN_CHANGED(sops) || DYN_CHANGED(scmpop)) {
        vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_BIT, d.sops[0][0], d.sops[0][1], d.sops[0][2], d.scmpop[0]);
        vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_BACK_BIT, d.sops[1][0], d.sops[1][1], d.sops[1][2], d.scmpop[1]);
    }
    if (DYN_CHANGED(discard)) vkCmdSetRasterizerDiscardEnable(cmd, d.discard);
    if (dev.eds3_polygon && DYN_CHANGED(poly)) vkCmdSetPolygonModeEXT(cmd, d.poly);
    if (dev.eds3_clamp && DYN_CHANGED(clamp)) vkCmdSetDepthClampEnableEXT(cmd, d.clamp);
    if (dev.eds3_a2c && DYN_CHANGED(a2c)) vkCmdSetAlphaToCoverageEnableEXT(cmd, d.a2c);
    if (d.n) {
        const bool same_n = cb.dyn_valid && cb.dyn.n == d.n;
        if (!same_n || std::memcmp(d.en, cb.dyn.en, sizeof(VkBool32) * d.n))
            FINE_X(FC_DYN_CALLS, 1), vkCmdSetColorBlendEnableEXT(cmd, 0, (uint32_t)d.n, d.en);
        if (!same_n || std::memcmp(d.eq, cb.dyn.eq, sizeof(VkColorBlendEquationEXT) * d.n))
            FINE_X(FC_DYN_CALLS, 1), vkCmdSetColorBlendEquationEXT(cmd, 0, (uint32_t)d.n, d.eq);
        if (!same_n || std::memcmp(d.wm, cb.dyn.wm, sizeof(VkColorComponentFlags) * d.n))
            FINE_X(FC_DYN_CALLS, 1), vkCmdSetColorWriteMaskEXT(cmd, 0, (uint32_t)d.n, d.wm);
    }
    cb.dyn = d;
    cb.dyn_valid = true;
    cb.dyn_fast.on = true, cb.dyn_fast.nb = nb;
    std::memcpy(cb.dyn_fast.fp, df, sizeof df);
    std::memcpy(cb.dyn_fast.up, du, sizeof du);
    std::memcpy(cb.dyn_fast.raw, db, (size_t)nb);
    cb.dyn_fast.wm0 = d.n ? (uint32_t)d.wm[0] : 0, cb.dyn_fast.wm_n = (uint32_t)d.n;
    ps.wm_or |= cb.dyn_fast.wm0, ps.wm_n = cb.dyn_fast.wm_n;
}
#undef DYN_CHANGED

/// Without robustness2's nullDescriptor a descriptor's image view must not be VK_NULL_HANDLE
/// (VUID-VkWriteDescriptorSet-descriptorType-02997): 1x1 black images (1D, 2D/cube, 3D) and a view of each type, made once
/// in the first recording (rec_cmd). Images that could not all be made are given back: the views stay null, as before
void make_dummies() {
    static VkImage img[3]{};
    static VmaAllocation mem[3]{};
    if (img[0] || dev.null_descriptor || !rec_cmd) return;
    const VkImageType type[3] = {VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D};
    const uint32_t layers[3] = {1, 6, 1};
    for (int i = 0; i < 3; i++) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.flags = i == 1 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
        ci.imageType = type[i];
        ci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ci.extent = {1, 1, 1};
        ci.mipLevels = 1, ci.arrayLayers = layers[i];
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!create_image(ci, img[i], mem[i])) {
            for (int j = 0; j <= i; j++) {
                if (img[j]) destroy_image(img[j], mem[j]);
                img[j] = VK_NULL_HANDLE, mem[j] = nullptr;
            }
            static int logged;
            if (logged++ < 4) logf("vk: the dummy images (no nullDescriptor): not made");
            return;
        }
    }
    VkImageMemoryBarrier2 b[3]{};
    for (int i = 0; i < 3; i++) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b[i].srcStageMask = VK_PIPELINE_STAGE_2_NONE, b[i].dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        b[i].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].image = img[i];
        b[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers[i]};
    }
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 3, d.pImageMemoryBarriers = b;
    vkCmdPipelineBarrier2(rec_cmd, &d);
    const VkClearColorValue black{};
    for (int i = 0; i < 3; i++) {
        const VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers[i]};
        vkCmdClearColorImage(rec_cmd, img[i], VK_IMAGE_LAYOUT_GENERAL, &black, 1, &r);
    }
    for (int i = 0; i < 3; i++) {
        b[i].srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT, b[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b[i].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b[i].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        b[i].oldLayout = b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    vkCmdPipelineBarrier2(rec_cmd, &d);
    // (VkImageViewType order: 1D, 2D, 3D, CUBE, 1D_ARRAY, 2D_ARRAY, CUBE_ARRAY)
    const int src[7] = {0, 1, 2, 1, 0, 1, 1};
    const uint32_t view_layers[7] = {1, 1, 1, 6, 1, 6, 6};
    for (int v = 0; v < 7; v++) {
        if (v == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY && !dev.features.imageCubeArray) continue;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img[src[v]];
        vi.viewType = (VkImageViewType)v;
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, view_layers[v]};
        if (vkCreateImageView(dev.device, &vi, nullptr, &dummies.view[v]) != VK_SUCCESS) dummies.view[v] = VK_NULL_HANDLE;
    }
}


// ---- clears: between the draws of a pass, cut by scissor 0 (as deko3d programs every clear), a channel mask with
// a full-screen triangle (vkCmdClearAttachments has no write mask; TOTK's maxwell.md 1.3) ----
VkRect2D clear_rect(const NvnState& s, uint32_t w, uint32_t h) {
    VkRect2D r{{0, 0}, {w, h}};
    if (s.scissor[2] > 0 && s.scissor[3] > 0) {
        const int32_t x0 = std::clamp(s.scissor[0], 0, (int32_t)w), y0 = std::clamp(s.scissor[1], 0, (int32_t)h);
        const int32_t x1 = std::clamp(s.scissor[0] + s.scissor[2], 0, (int32_t)w);
        const int32_t y1 = std::clamp(s.scissor[1] + s.scissor[3], 0, (int32_t)h);
        r = {{x0, y0}, {(uint32_t)std::max(0, x1 - x0), (uint32_t)std::max(0, y1 - y0)}};
    }
    return r;
}

/// The SPIR-V of a module with its Location 0 output moved to location (masked clears of target n)
std::vector<uint32_t> spirv_location(const uint32_t* w, size_t n, uint32_t location) {
    std::vector<uint32_t> out(w, w + n);
    for (size_t i = 5; i < out.size();) {
        const uint32_t op = out[i] & 0xFFFF, len = out[i] >> 16;
        if (!len) break;
        if (op == 71 && len >= 4 && out[i + 2] == 30) out[i + 3] = location;   // OpDecorate %x Location
        i += len;
    }
    return out;
}

std::unordered_map<uint64_t, VkPipeline> clear_pipes;
struct ClearFailure { int tries = 0, waited = 0; };   // (a key whose pipeline could not be made: its tries, its clears since)
std::unordered_map<uint64_t, ClearFailure> clear_failures;
constexpr int kClearTries = 3;          // (failed this often, a key backs off: no try a frame)
constexpr int kClearRetryEvery = 600;   // (a key that backed off is tried once in this many of its clears)
VkPipelineLayout clear_layout = VK_NULL_HANDLE;

/// (#1787) The pipeline of a masked clear, made on its first use. A layout, module or pipeline the driver did not make is not
/// used and not kept: the key is tried again on its next clears, kClearTries times, then once every kClearRetryEvery of its
/// clears, so a device short of memory for a while gets the clear back and one that never makes it costs no try a frame
VkPipeline clear_pipeline(const Targets& at, int index, uint32_t mask, int kind) {
    uint64_t key = mix((uint64_t)index << 8 | mask << 4 | (uint64_t)kind, (uint64_t)at.samples);
    for (int i = 0; i < at.n; i++) key = mix(key, (uint64_t)at.color_format[i]);
    key = mix(key, (uint64_t)at.depth_format);
    auto it = clear_pipes.find(key);
    if (it != clear_pipes.end()) return it->second;
    ClearFailure& failed = clear_failures[key];
    if (failed.tries >= kClearTries && ++failed.waited < kClearRetryEvery) return VK_NULL_HANDLE;
    failed.waited = 0;
    auto fail = [&](const char* what, VkResult r) -> VkPipeline {
        if (++failed.tries == kClearTries)
            logf("vk: masked clear: no %s (VkResult %d); this clear is left out, tried again every %d clears", what, (int)r,
                 kClearRetryEvery);
        return VK_NULL_HANDLE;
    };
    if (!clear_layout) {
        const VkPushConstantRange r{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &r;
        VkPipelineLayout made = VK_NULL_HANDLE;
        const VkResult lr = vkCreatePipelineLayout(dev.device, &li, nullptr, &made);
        if (lr != VK_SUCCESS || !made) return fail("pipeline layout", lr);
        clear_layout = made;
    }
    const auto fs = kind == 1 ? BUILTIN_SPV(CLEAR_U_FRAG_SPV) : kind == 2 ? BUILTIN_SPV(CLEAR_I_FRAG_SPV) : BUILTIN_SPV(CLEAR_F_FRAG_SPV);
    const uint32_t* fw = fs.code;
    const size_t fn = fs.size / 4;
    const std::vector<uint32_t> frag = spirv_location(fw, fn, (uint32_t)index);
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE;
    IntelDriverGate gate;   // (the modules and the pipeline: one thread in Intel's compiler on Windows, a fault leaves it out)
    VkResult vr = VK_ERROR_UNKNOWN, fr = VK_ERROR_UNKNOWN;
    {
        IntelDriverGate gate;   // (released before cached_graphics, which takes the same lock)
        const auto spv = BUILTIN_SPV(CLEAR_VERT_SPV);
        mi.codeSize = spv.size, mi.pCode = spv.code;
        vr = guarded_shader_module(&mi, &vm);
        if (vr != VK_SUCCESS || !vm) return fail("vertex module", vr);
        mi.codeSize = frag.size() * 4, mi.pCode = frag.data();
        fr = guarded_shader_module(&mi, &fm);
    }
    if (fr != VK_SUCCESS || !fm) {
        vkDestroyShaderModule(dev.device, vm, nullptr);
        return fail("fragment module", fr);
    }
    VkPipelineShaderStageCreateInfo st[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                          {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT, st[0].module = vm, st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, st[1].module = fm, st[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL, rs.cullMode = VK_CULL_MODE_NONE, rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = at.samples;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cb[8]{};
    for (int i = 0; i < at.n; i++) cb[i].colorWriteMask = i == index ? mask : 0;
    VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    bs.attachmentCount = (uint32_t)at.n, bs.pAttachments = cb;
    const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2, dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = (uint32_t)at.n;
    ri.pColorAttachmentFormats = at.color_format;
    if (at.depth) {
        if (depth_aspect(at.depth_format)) ri.depthAttachmentFormat = at.depth_format;
        if (has_stencil(at.depth_format)) ri.stencilAttachmentFormat = at.depth_format;
    }
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.pNext = &ri;
    ci.stageCount = 2, ci.pStages = st;
    ci.pVertexInputState = &vi, ci.pInputAssemblyState = &ia, ci.pViewportState = &vp, ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms, ci.pDepthStencilState = &ds, ci.pColorBlendState = &bs, ci.pDynamicState = &dy;
    ci.layout = clear_layout;
    VkPipeline pl = VK_NULL_HANDLE;
    const VkResult pr = cached_graphics(&ci, &pl);
    vkDestroyShaderModule(dev.device, vm, nullptr);
    vkDestroyShaderModule(dev.device, fm, nullptr);
    if (pr != VK_SUCCESS || !pl) {
        if (pl) vkDestroyPipeline(dev.device, pl, nullptr);
        return fail("pipeline", pr);
    }
    clear_failures.erase(key);
    clear_pipes[key] = pl;
    return pl;
}

int format_kind(VkFormat f) {   // 0 float, 1 unsigned integer, 2 signed integer
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32: return 1;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT: return 2;
    default: return 0;
    }
}

int64_t st_clears, st_masked_clears;

// doc: docs/design.md#4-the-lessons-from-mkw-as-rules-here (rule 5: clears after draws get a pass of their own)
/// A clear after draws of the open pass: in a pass of its own, after a full barrier (begin_pass makes it). In the pass
/// the draws before it may still read what it clears (a target they sample: the depth the fog and the lighting read):
/// vkCmdClearAttachments is ordered with the attachments' own writes only, and the GPU cleared tiles while a draw
/// before it still sampled them: a band of the picture, below or above a line that moved, for a frame (the fog's flat
/// colour over a Grand Prix intro, the shadows' dark band in its race).
void clear_after_draws(VkCommandBuffer cmd) {
    if (!pass().open || pass().draws == 0) return;
    static int logged;
    if (logged++ < 6)
        logf("vk: a clear after %d draws of its pass (%ux%u, %d colour targets%s, last program %016llx): a pass of its own",
             pass().draws, pass().width, pass().height, pass().n, pass().depth ? " + depth" : "", (unsigned long long)pass().prog);
    end_pass(cmd);
}

/// (hookshot) The temporal upscalers' reactive mask (vk_reactive.h, RX_VK_REACTIVE=1): before the scene's transparent draws
/// (reactive_policy.h: blended over it, tested against its depth without writing it) the scene's colour is copied, in a
/// break of the pass. The scene is the B10G11R11 target with a depth of the size the upscaler last took, as its jitter is
/// given (temporal_jitter_size, fsr3_inputs::jitter_size: 0 while neither draws). Not while AMD FSR 4 or Intel XeSS draws on
/// the frame chain: neither is given a mask, so nothing is copied for it
void reactive_scene_draw(VkCommandBuffer cmd, const Program& p, const NvnState& s, const Targets& at) {
    if (!reactive::wanted()) return;
    if (!at.n || !at.color[0] || !at.depth || !temporal_scene_format(at.color_format[0])) return;
    const uint64_t now = texture_frame_no();
    struct Seen { uint64_t frame; uint32_t w, h; };
    static thread_local Seen seen{~0ull, 0, 0};
    if (seen.frame != now) {   // (once a frame on each thread, not once a draw)
        const bool chain = temporal_jitter_size(0) != 0;   // (one upscaler at a time: the chain's engine, else FSR 3.1)
        if (chain && (upscaler_rows::is_fsr4(temporal_mode()) || upscaler_rows::is_xess(temporal_mode())))
            seen = {now, 0, 0};
        else
            seen = {now, chain ? temporal_jitter_size(0) : fsr3_inputs::jitter_size(0),
                    chain ? temporal_jitter_size(1) : fsr3_inputs::jitter_size(1)};
    }
    if (!seen.w || at.width != seen.w || at.height != seen.h) return;
    static thread_local reactive_policy::Run run;
    const reactive_policy::Draw kind =
        reactive_policy::classify(target_write_mask(p, s, 0) != 0, (s.color.blend_en & 1) != 0, s.ds.dt != 0, s.ds.dw != 0);
    const bool capped = run.capped && run.frame == now;
    if (!reactive_policy::copy_before(run, now, (uint64_t)(uintptr_t)at.color[0], kind)) {
        static std::atomic<int> told{0};
        if (run.capped && !capped && told.fetch_add(1, std::memory_order_relaxed) < 4)
            logf("vk: reactive mask: the scene's transparent draws come in more than %d runs (frame %llu): the mask holds what "
                 "was drawn after the last copy too",
                 reactive_policy::kCopies, (unsigned long long)now);
        return;
    }
    end_pass(cmd);   // (a copy cannot be made in a pass)
    reactive::copy_scene(cmd, at.color[0], at.color_level[0], at.color_layer[0], at.width, at.height);
}
// doc: docs/design.md#4-the-lessons-from-mkw-as-rules-here (rule 5: a draw that samples what its pass wrote gets a pass of its own)
/// A draw that samples a target this pass already wrote, or writes a target an earlier draw of this pass sampled: in
/// a pass of its own, after a full barrier (begin_pass makes it), as for the clears above: in one pass the GPU orders
/// the targets' writes with each other only, not with the draws' texture reads of them. (A draw that samples a target
/// it writes itself is left as it is.) Returns the targets the draw writes.
uint16_t draw_hazards(VkCommandBuffer cmd, const Program& p, const NvnState& s, const Targets& at, uint16_t& samples) {
    Pass& ps = pass();   // (once: the accessor is a call and a thread-local read at each use. The same object for the thread's life: draw_begin_recording resets it in place)
    uint16_t writes = 0;
    for (int i = 0; i < at.n; i++)
        if (at.color[i] && target_write_mask(p, s, i)) writes |= (uint16_t)(1u << i);
    if (at.depth && s.ds.dw) writes |= 1u << 8;
    samples = 0;
    for (const Texture* t : gfx_sampled()) {
        if (!t) continue;
        for (int i = 0; i < at.n; ++i)
            if (at.color[i] == t) samples |= (uint16_t)(1u << i);
        if (at.depth == t) samples |= 1u << 8;
    }
    const uint16_t feedback = samples & writes;
    static std::atomic<int> self_logged{0};
    if (feedback && self_logged.load(std::memory_order_relaxed) < 12 &&
        self_logged.fetch_add(1, std::memory_order_relaxed) < 12)
        logf("vk: a draw (program %016llx) samples targets it writes (mask %#x, %ux%u)",
             (unsigned long long)p.key, (unsigned)feedback, at.width, at.height);
    if (!ps.open) return writes;
    bool same = ps.depth == at.depth_view && ps.n == at.n && ps.width == at.width && ps.height == at.height;
    for (int i = 0; i < at.n && same; i++) same = ps.color[i] == at.color_view[i];
    if (!same) return writes;   // (other targets: begin_pass ends this pass with a barrier anyway)
    const char* kind = (samples & ps.written & ~writes) ? "samples a target the pass wrote" : nullptr;
    if (!kind && writes_sampled_target(writes, ps.sampled_targets))
        kind = "writes a target an earlier draw sampled";
    if (!kind) return writes;
    static int logged;
    if (logged++ < 8)
        logf("vk: a draw (program %016llx) %s, after %d draws of the pass (%ux%u): a pass of its own", (unsigned long long)p.key,
             kind, ps.draws, ps.width, ps.height);
    end_pass(cmd);
    return writes;
}

} // namespace

void clear_color_target(VkCommandBuffer cmd, NvnState& s, int index, const float rgba[4], uint32_t mask) {
#ifdef TRACY_ENABLE
    HT_SCOPE("clear_color_target");
#endif
    ParLock lk;
    mask &= 0xF;
    if (!mask) return;
    const Targets at = targets(s);
    if (index < 0 || index >= at.n || !at.color_view[index]) return;
    clear_after_draws(cmd);
    begin_pass(cmd, at);
    const uint32_t lw = at.width, lh = at.height;
    const VkRect2D rect = clear_rect(s, lw, lh);
    if (!rect.extent.width || !rect.extent.height) return;
    st_clears++, pass().clears++;
    if (mask == 0xF) {
        VkClearAttachment ca{};
        ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ca.colorAttachment = (uint32_t)index;
        std::memcpy(&ca.clearValue.color, rgba, 16);
        const VkClearRect cr{rect, 0, 1};
        vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
        pass().written |= (uint16_t)(1u << index);
        return;
    }
    pass().written |= (uint16_t)(1u << index);
    pcache_init();
    const int kind = format_kind(at.color_format[index]);
    const VkPipeline pl = clear_pipeline(at, index, mask, kind);
    if (!pl) return;
    st_masked_clears++;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl);
    bound().reset_keeping_index();   // (the clear binds no index buffer: the draws' one stays bound)
    descriptors_disturbed();   // (the next draw pushes all its descriptors, not only those that changed)
    const VkViewport v{0, 0, (float)lw, (float)lh, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &v);
    vkCmdSetScissor(cmd, 0, 1, &rect);
    vkCmdPushConstants(cmd, clear_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, rgba);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void clear_depth_stencil(VkCommandBuffer cmd, NvnState& s, float depth, bool depth_mask, uint32_t stencil,
                         uint32_t stencil_mask) {
    ParLock lk;
    const Targets at = targets(s);
    if (!at.depth) return;
    VkImageAspectFlags aspect = 0;
    if (depth_mask && depth_aspect(at.depth_format)) aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (stencil_mask && has_stencil(at.depth_format)) aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!aspect) return;
    if (stencil_mask && (stencil_mask & 0xFF) != 0xFF) {
        static int logged;
        if (logged++ < 4) logf("vk: stencil clear with write mask %#x: cleared whole", stencil_mask);
    }
    clear_after_draws(cmd);
    begin_pass(cmd, at);
    const VkRect2D rect = clear_rect(s, at.width, at.height);
    if (!rect.extent.width || !rect.extent.height) return;
    VkClearAttachment ca{};
    ca.aspectMask = aspect;
    ca.clearValue.depthStencil = {std::clamp(depth, 0.0f, 1.0f), stencil & 0xFF};
    const VkClearRect cr{rect, 0, 1};
    vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
    pass().written |= 1u << 8;
    st_clears++, pass().clears++;
}

void clear_stats(int64_t* clears, int64_t* masked) {
    *clears = st_clears, *masked = st_masked_clears;
}

namespace {
} // namespace

void draw_flush_recording(int slot) {
    ParLock lk;
    const RecordingRes& r = res[slot];
    // Earlier descriptors still name streams we outgrew; later retained streams were not used this recording.
    // (none when the slot's first stream could not be made: draw_begin_recording)
    for (size_t i = 0; i <= r.si && i < r.streams.size(); i++) {
        const RecordingRes::Stream& s = r.streams[i];
        if (s.alloc) flush_mapped(s.alloc, i < r.si ? s.size : r.used);
    }
}

void draw_begin_recording(int slot, VkCommandBuffer cmd, uint64_t value) {
    rec_cmd = cmd;
    rec_value = value;
    ParLock lk;
    if ((size_t)slot >= res.size()) res.resize((size_t)slot + 1);
    RecordingRes& r = res[slot];
    if (!r.pool) {
        r.pool_shift = pool_shift_first();
        r.pool = make_pool(r.pool_shift);
    } else {
        {
            rec::OwnSlot own;   // (RX_VK_RECORDER_OWN: the slot's own pools, vk_recorder.h; begin_recording's slot_begin)
            vkResetDescriptorPool(dev.device, r.pool, 0);
            for (VkDescriptorPool p : r.extra) vkDestroyDescriptorPool(dev.device, p, nullptr);
            // (the last recording of the slot ran out of a pool smaller than whole: the slot's pool is made whole now)
            if (!r.extra.empty() && r.pool_shift > 0) {
                if (const VkDescriptorPool whole = make_pool(0)) {
                    vkDestroyDescriptorPool(dev.device, r.pool, nullptr);
                    r.pool = whole, r.pool_shift = 0;
                    pools_grown.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        r.extra.clear();
        // (the streams the last recording of the slot grew into: given back after 120 recordings that stayed in the first, the
        // slot's last one being done (recording_wait))
        if (r.streams.size() > 1) {
            if (r.si > 0) r.idle = 0;
            else if (++r.idle >= 120) {
                for (size_t i = 1; i < r.streams.size(); i++) {
                    stream_extra -= r.streams[i].size;
                    destroy_buffer(r.streams[i].buffer, r.streams[i].alloc);
                }
                r.streams.resize(1);
            }
        }
    }
    if (r.streams.empty()) stream_insert(r, 0, kStream);
    if (!r.streams.empty()) {
        const RecordingRes::Stream& s = r.streams[0];
        r.si = 0, r.stream = s.buffer, r.stream_alloc = s.alloc, r.mapped = s.mapped, r.size = s.size;
    } else {
        r.si = 0, r.stream = VK_NULL_HANDLE, r.stream_alloc = nullptr, r.mapped = nullptr, r.size = 0;
    }
    if (r.before + r.used > stream_peak.load(std::memory_order_relaxed)) stream_peak.store(r.before + r.used, std::memory_order_relaxed);
    static int64_t begun;
    static std::atomic<size_t> c1_peak{0};   // (the most c1 stages a recording uploaded: c1_at's size before its reset)
    if (r.c1_at.size() > c1_peak.load(std::memory_order_relaxed)) c1_peak.store(r.c1_at.size(), std::memory_order_relaxed);
    static std::atomic<uint32_t> sets_peak{0}, descs_peak[6]{};   // (the most a recording took of its pools: RecordingRes)
    if (r.pool_sets > sets_peak.load(std::memory_order_relaxed)) sets_peak.store(r.pool_sets, std::memory_order_relaxed);
    for (int i = 0; i < 6; i++)
        if (r.pool_descs[i] > descs_peak[i].load(std::memory_order_relaxed)) descs_peak[i].store(r.pool_descs[i], std::memory_order_relaxed);
    r.pool_sets = 0;
    for (uint32_t& n : r.pool_descs) n = 0;
    if (++begun % 3000 == 0)
        logf("vk: the recording streams: %zu of %llu MB, the most one held %llu KB; %llu made on demand (%llu MB kept now); "
             "c1 stages: at most %zu a recording; %llu descriptor pools made when a recording's own ran out (%llu slots' pools made "
             "whole); a recording's pool took at most %u sets (ub %u, sb %u, cis %u, si %u, utb %u, stb %u)", res.size(),
             (unsigned long long)(kStream >> 20), (unsigned long long)(stream_peak.load(std::memory_order_relaxed) >> 10),
             (unsigned long long)stream_grown.load(), (unsigned long long)(stream_extra.load() >> 20),
             c1_peak.load(std::memory_order_relaxed), (unsigned long long)pools_extra.load(std::memory_order_relaxed),
             (unsigned long long)pools_grown.load(std::memory_order_relaxed), sets_peak.load(), descs_peak[0].load(), descs_peak[1].load(), descs_peak[2].load(), descs_peak[3].load(),
             descs_peak[4].load(), descs_peak[5].load());
    r.used = 0, r.before = 0;
    r.c1_at.reset();   // (its array kept: emptied at every recording's start, it fills again to about the same size)
    static std::atomic<uint64_t> ub_snap_epochs{0};   // (every recording a number of its own: ub_snap_slot gives out only its copies)
    r.ub_snap_epoch = ++ub_snap_epochs;   // (the uniform copies of the slot's last recording are no longer given out)
    cur = &r;
    pass() = Pass{};
    work_since_barrier = false;
    bound().reset();
    descriptors_disturbed();
    static thread_local uint64_t hits_value = ~0ull;   // (resolved targets last a submission, through its parts: no CPU write is
    if (value != hits_value) {            // collected between them; resolved handles last while their slot and
        hits_value = value;               // texture do, and are touched again once a submission)
        targets_serial = ~0ull;
        if (handle_hits().size() > 65536) {
            handle_hits().clear();
            for (ViewFast& f : view_fast()) f.key = ~0ull;   // (the front cache holds what the map held)
        }
        if (prog_slots().size() > 8192) prog_slots().clear();
    }
    for (auto& row : bank_cache())
        for (BankCache& c : row) c = BankCache{}, c.at = UINT64_MAX;
    if (!zero_buffer) {
        void* mapped = nullptr;
        create_buffer(65536, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, zero_buffer, zero_alloc, &mapped);
        if (mapped) std::memset(mapped, 0, 65536);
        make_dummies();
    }
}

// ---- RX_VK_PASS_TRACE=1: the passes and dispatches of the last frames in stream order, logged around a flash (the
// one-frame glitches: what the frame that showed one did differently from its neighbours) ----
bool pass_trace_on() {
    static const bool v = env_on("RX_VK_PASS_TRACE");
    return v;
}
std::mutex trace_mu;
std::map<uint64_t, std::vector<std::pair<uint64_t, std::string>>> pass_traces;   // texture frame -> (order, line)
thread_local uint64_t trace_seq = 0;
void trace_line(const char* fmt, ...) {
    char line[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    const uint64_t key = (texture_segment() << 24) | (trace_seq++ & 0xFFFFFF);   // (segments: on several threads)
    std::lock_guard lk(trace_mu);
    auto& v = pass_traces[texture_frame_no()];
    if (v.size() < 6000) v.emplace_back(key, line);
    while (pass_traces.size() > 8) pass_traces.erase(pass_traces.begin());
}

// (hookshot) RX_VK_PROBE=1 (replay is serial then): the centre texel of every colour target after each pass and of
// every written image after each dispatch, kept for the last 16 frames. At a flash (RX_VK_FLASH) the entries of that
// frame that differ from both neighbours' are logged in order: the first is the pass that went wrong.
namespace {
constexpr uint32_t kProbeFrames = 16, kProbePerFrame = 1024;
struct ProbeEntry { uint64_t prog, gpu; uint32_t fmt, w, h; char kind; int draws; };
VkBuffer probe_buf = VK_NULL_HANDLE;
VmaAllocation probe_mem = nullptr;
void* probe_map = nullptr;
ProbeEntry probe_meta[kProbeFrames][kProbePerFrame];
uint32_t probe_n[kProbeFrames];
uint64_t probe_frame[kProbeFrames];
} // namespace
bool probe_on() {
    static const bool on = getenv("RX_VK_PROBE") != nullptr;
    return on;
}
// (RX_VK_PROBE=2: at speed: replay stays parallel, big images only, the picture's chain; the frame log's white frames dump
// the last 16 frames, probe_all_<n>.txt)
bool probe_lite() {
    static const bool on = getenv("RX_VK_PROBE") && atoi(getenv("RX_VK_PROBE")) == 2;
    return on;
}
bool probe_serial() { return probe_on() && !probe_lite(); }
std::mutex probe_mu;
void probe_image(VkCommandBuffer cmd, Texture* tex, uint32_t level, uint32_t layer, uint64_t prog, char kind, int draws = -1) {
    if (!probe_on() || !tex || !tex->image || tex->samples != VK_SAMPLE_COUNT_1_BIT || tex->astc || tex->bc ||
        !(tex->aspect & VK_IMAGE_ASPECT_COLOR_BIT))
        return;
    {   // (RX_VK_PROBE_MIN=<pixels>: smaller images too, the ambient cubemap's for one)
        static const uint32_t min_w = getenv("RX_VK_PROBE_MIN") ? (uint32_t)atoi(getenv("RX_VK_PROBE_MIN")) : 384;
        if (probe_lite() && (tex->width >> std::min(level, tex->levels - 1)) < min_w) return;
    }
    std::unique_lock lite_lk(probe_mu, std::defer_lock);
    if (probe_lite()) lite_lk.lock();
    if (!probe_buf && !create_buffer(uint64_t{kProbeFrames} * kProbePerFrame * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                                     probe_buf, probe_mem, &probe_map))
        return;
    const uint64_t f = texture_frame_no();
    const uint32_t slot = (uint32_t)(f % kProbeFrames);
    if (probe_frame[slot] != f) probe_frame[slot] = f, probe_n[slot] = 0;
    if (probe_n[slot] >= kProbePerFrame) return;
    const uint32_t i = probe_n[slot]++;
    level = std::min(level, tex->levels - 1);
    const uint32_t w = std::max(1u, tex->width >> level), h = std::max(1u, tex->height >> level);
    probe_meta[slot][i] = {prog, tex->gpu, (uint32_t)tex->format, w, h, kind, draws};
    std::memset((uint8_t*)probe_map + (uint64_t{slot} * kProbePerFrame + i) * 16, 0, 16);
    full_barrier(cmd);
    const VkBufferImageCopy c{(uint64_t{slot} * kProbePerFrame + i) * 16, 0, 0,
                              {VK_IMAGE_ASPECT_COLOR_BIT, level, std::min(layer, tex->layers - 1), 1},
                              {(int32_t)(w / 2), (int32_t)(h / 2), 0}, {1, 1, 1}};
    vkCmdCopyImageToBuffer(cmd, tex->image, VK_IMAGE_LAYOUT_GENERAL, probe_buf, 1, &c);
    full_barrier(cmd);
}
// (the uniform buffers of each pass's first draw: non-finite floats, a global constant gone wrong)
std::vector<std::string> probe_ubos[kProbeFrames];
uint64_t probe_ubo_frame[kProbeFrames];
void probe_draw(const NvnState& s, const Program& p, int pass_draws) {
    if (!probe_on() || pass_draws != 0) return;
    if (probe_lite()) {   // (RX_VK_PROBE_UBO=<key prefix,...>: at speed, those programs' constants only)
        static const char* want = getenv("RX_VK_PROBE_UBO");
        if (!want) return;
        char hex[24];
        snprintf(hex, sizeof hex, "%016llx", (unsigned long long)p.key);
        bool hit = false;
        for (const char* q = want; *q && !hit;) {
            const char* e = q;
            while (*e && *e != ',') e++;
            hit = e > q && !strncmp(hex, q, (size_t)(e - q));
            q = *e ? e + 1 : e;
        }
        if (!hit) return;
    }
    std::lock_guard ubo_lk(probe_mu);
    const uint64_t f = texture_frame_no();
    const uint32_t slot = (uint32_t)(f % kProbeFrames);
    if (probe_ubo_frame[slot] != f) probe_ubo_frame[slot] = f, probe_ubos[slot].clear();
    for (const Stage& st : p.stages) {
        const int ns = std::clamp(st.nvn_stage, 0, kNvnStages - 1);
        for (int i = 0; i < 16; i++) {
            const nvn_p_bind_buf& u = s.ubo[ns][i];
            if (!u.addr || !(i + 3 < Shader::Info::MAX_CBUFS && st.info.constant_buffer_used_sizes[i + 3])) continue;
            static thread_local std::vector<uint8_t> b;
            const uint64_t k = std::min<uint64_t>(u.size ? u.size : 4096, 16384);
            b.resize(k);
            if (!memory_read_as_submitted(u.addr, b.data(), k)) continue;
            int nan = 0, inf = 0, zero = 0, first_bad = -1;
            float big = 0;
            for (uint64_t j = 0; j + 4 <= k; j += 4) {
                uint32_t w;
                std::memcpy(&w, b.data() + j, 4);
                const uint32_t e = w & 0x7f800000u;
                if (e == 0x7f800000u) {
                    ((w & 0x7fffffu) ? nan : inf)++;
                    if (first_bad < 0) first_bad = (int)j;
                }
                else if (e >= 0x5f000000u) big++;
                zero += w == 0;
            }
            char line[200];
            snprintf(line, sizeof line, "U %llu %016llx s%d c%d %#llx +%#llx nan %d inf %d huge %.0f zero %d first %#x", (unsigned long long)f,
                     (unsigned long long)p.key, ns, i + 3, (unsigned long long)u.addr, (unsigned long long)k, nan, inf, big, zero, first_bad);
            probe_ubos[slot].push_back(line);
            static const bool hex = env_on("RX_VK_PROBE_UBO_HEX");   // (the words themselves: which constant went wrong)
            if (hex) {
                std::string h = "H " + std::to_string(f) + " " + std::string(line + 2 + std::to_string(f).size() + 1, 16) + " c" + std::to_string(i + 3) + " ";
                char w8[12];
                for (uint64_t j = 0; j + 4 <= k; j += 4) {
                    uint32_t w;
                    std::memcpy(&w, b.data() + j, 4);
                    snprintf(w8, sizeof w8, "%08x", w);
                    h += w8;
                }
                probe_ubos[slot].push_back(std::move(h));
            }
        }
        // (the storage buffers too: where each is, how long, and what guest memory holds of it)
        for (const auto& d : st.info.storage_buffers_descriptors) {
            for (uint32_t k = 0; k < d.count && k < 8; k++) {
                const uint32_t off = d.cbuf_offset + k * 16;
                // (amd-fix) through the stage's banks as its program reads them: a storage buffer named in c1 (the program's
                // own constants) read 0 from the driver banks and was bound to nothing
                const uint64_t addr = stage_cbuf_word(s, st, d.cbuf_index, off) |
                                      (uint64_t)stage_cbuf_word(s, st, d.cbuf_index, off + 4) << 32;
                const uint64_t size = stage_cbuf_word(s, st, d.cbuf_index, off + 8);
                static thread_local std::vector<uint8_t> sb;
                const uint64_t n = std::min<uint64_t>(size, 4ull << 20);
                sb.resize((size_t)n);
                uint64_t hash = 0xcbf29ce484222325ull, nz = 0;
                const bool read = addr && n && memory_read_as_submitted(addr, sb.data(), n);
                if (read)
                    for (uint64_t j = 0; j < n; j++) nz += sb[j] != 0, hash = (hash ^ sb[j]) * 0x100000001b3ull;
                char line[200];
                snprintf(line, sizeof line, "S %llu %016llx s%d c%u+%#x%s addr %#llx size %#llx read %d nz %llu hash %016llx", (unsigned long long)f,
                         (unsigned long long)p.key, ns, (unsigned)d.cbuf_index, off, d.is_written ? "w" : "r", (unsigned long long)addr,
                         (unsigned long long)size, (int)read, (unsigned long long)nz, (unsigned long long)hash);
                probe_ubos[slot].push_back(line);
            }
        }
    }
}
void probe_dump_all(const char* why) {
    if (!probe_lite() || !probe_map) return;
    static int left = 8, n = 0;
    if (left <= 0) return;
    left--;
    char path[64];
    snprintf(path, sizeof path, "probe_all_%d.txt", n++);
    logf("vk: %s: %s, frame %llu", why, path, (unsigned long long)texture_frame_no());
    std::lock_guard lk(probe_mu);
    if (FILE* o = fopen(path, "w")) {
        const uint64_t now = texture_frame_no();
        for (uint64_t f = now >= 15 ? now - 15 : 0; f <= now; f++) {
            const uint32_t slot = (uint32_t)(f % kProbeFrames);
            if (probe_frame[slot] != f) continue;
            for (uint32_t i = 0; i < probe_n[slot]; i++) {
                const ProbeEntry& e = probe_meta[slot][i];
                uint32_t w[4];
                std::memcpy(w, (uint8_t*)probe_map + (uint64_t{slot} * kProbePerFrame + i) * 16, 16);
                fprintf(o, "%llu %u %c %016llx %#llx %ux%u %u %08x %08x %08x %08x d%d\n", (unsigned long long)f, i, e.kind,
                        (unsigned long long)e.prog, (unsigned long long)e.gpu, e.w, e.h, e.fmt, w[0], w[1], w[2], w[3], e.draws);
            }
        }
        for (uint32_t slot = 0; slot < kProbeFrames; slot++)
            for (const std::string& l : probe_ubos[slot]) fprintf(o, "%s\n", l.c_str());
        fclose(o);
    }
}
void probe_log(uint64_t f) {
    if (!probe_on() || !probe_map) return;
    static int left = 30;
    if (left <= 0) return;
    const uint32_t a = (uint32_t)((f - 1) % kProbeFrames), b = (uint32_t)(f % kProbeFrames), c = (uint32_t)((f + 1) % kProbeFrames);
    if (probe_frame[a] != f - 1 || probe_frame[b] != f || probe_frame[c] != f + 1) {
        logf("vk: probe of frame %llu: its neighbours are gone", (unsigned long long)f);
        return;
    }
    left--;
    logf("vk: probe of flash frame %llu: %u / %u / %u entries", (unsigned long long)f, probe_n[a], probe_n[b], probe_n[c]);
    const auto word = [](uint32_t slot, uint32_t i, int k) {
        uint32_t w;
        std::memcpy(&w, (uint8_t*)probe_map + (uint64_t{slot} * kProbePerFrame + i) * 16 + k * 4, 4);
        return w;
    };
    char path[64];
    snprintf(path, sizeof path, "probe_%llu.txt", (unsigned long long)f);
    if (FILE* o = fopen(path, "w")) {   // (scripts/probe.py aligns the frames by program and target)
        for (const uint32_t slot : {a, b, c})
            for (uint32_t i = 0; i < probe_n[slot]; i++) {
                const ProbeEntry& e = probe_meta[slot][i];
                fprintf(o, "%llu %u %c %016llx %#llx %ux%u %u %08x %08x %08x %08x\n", (unsigned long long)probe_frame[slot], i, e.kind,
                        (unsigned long long)e.prog, (unsigned long long)e.gpu, e.w, e.h, e.fmt, word(slot, i, 0), word(slot, i, 1),
                        word(slot, i, 2), word(slot, i, 3));
            }
        for (const uint32_t slot : {a, b, c})
            for (const std::string& l : probe_ubos[slot]) fprintf(o, "%s%c", l.c_str(), 10);
        fclose(o);
    }
}

// (amd-fix, diagnosis) in a frame dump, the indirect mesh draws' arguments as the GPU has them: noted at the draw (inside
// the pass, where no copy can be recorded) and copied out once the pass ends ("dump buffer ... mesh indirect")
namespace {
TLS_OBJECT(dump_mesh_indirect, std::vector<std::pair<uint64_t, uint64_t>>)   // (program key, indirect address)
} // namespace
void end_pass(VkCommandBuffer cmd) {
    Pass& ps = pass();   // (once: the accessor is a call and a thread-local read at each use. The same object for the thread's life: draw_begin_recording resets it in place)
    if (!ps.open) return;
    vkCmdEndRendering(cmd);
    ps.open = false;
    note_pass_end(cmd);   // (before the dumps and probes below: they record work, and the next pass's barrier is then a full one)
    if (ps.has_block_images) {
        full_barrier(cmd, "barrier (before BC slice stores)");
        for (int i = 0; i < ps.n; i++)
            if (ps.block_image[i])
                copy_block_attachment(cmd, *ps.tex[i], ps.block_image[i], ps.level[i], ps.layer[i], false);
        full_barrier(cmd, "barrier (after BC slice stores)");
    }
    if (dump_on() && !dump_mesh_indirect().empty()) {
        for (const auto& [key, addr] : dump_mesh_indirect()) {
            const BufferRef r = gpu_buffer_bind(addr, 16, false);
            char what[80];
            snprintf(what, sizeof what, "mesh indirect prog %016llx at %#llx", (unsigned long long)key, (unsigned long long)addr);
            if (r.buffer) dump_buffer(cmd, r.buffer, r.offset, 16, what);
        }
        dump_mesh_indirect().clear();
    }
    if (probe_on())
        for (int i = 0; i < ps.n; i++)
            if (ps.color[i] && ps.tex[i]) probe_image(cmd, ps.tex[i], ps.level[i], ps.layer[i], ps.prog, 'p', ps.draws);
    if (pass_trace_on()) {
        char sampled[160] = "";
        int k = 0;
        for (size_t i = 0; i < ps.sampled.size() && i < 6 && k < 140; i++)
            k += snprintf(sampled + k, sizeof sampled - k, " %#llx", (unsigned long long)ps.sampled[i]->gpu);
        trace_line("pass %ux%u f%d %dc%s wm%x/%u: %d draws %d clears, c0 %#llx/%u, depth %#llx/%u, progs %016llx..%016llx, "
                   "sampled%s", ps.width, ps.height, (int)ps.format0, ps.n, ps.depth ? "+d" : "", ps.wm_or, ps.wm_n, ps.draws,
                   ps.clears, (unsigned long long)(ps.n && ps.tex[0] ? ps.tex[0]->gpu : 0), ps.layer[0],
                   (unsigned long long)(ps.depth_tex ? ps.depth_tex->gpu : 0), ps.depth_layer,
                   (unsigned long long)ps.first_prog, (unsigned long long)ps.prog, sampled);
    }
    if (gprof_on()) {
        char label[160];
        snprintf(label, sizeof label, "pass %ux%u f%d %dc%s, %s draws, %zu sampled, first prog %016llx, last prog %016llx", ps.width, ps.height,
                 (int)ps.format0, ps.n, ps.depth ? "+d" : "", ps.draws > 256 ? ">256" : ps.draws > 32 ? "33-256" : ps.draws > 4 ? "5-32" : "0-4",
                 ps.sampled.size(), (unsigned long long)ps.first_prog, (unsigned long long)ps.prog);
        gprof_end(cmd, label, ps.draws);
    }
    if (dump_on())
        for (int i = 0; i < ps.n; i++)
            if (ps.color[i] && ps.tex[i]) {
                char what[64];
                snprintf(what, sizeof what, "pass c%d prog %016llx", i, (unsigned long long)ps.prog);
                dump_image(cmd, ps.tex[i], ps.level[i], ps.layer[i], what);
            }
    if (dump_on() && ps.depth && ps.depth_tex) {
        char what[64];
        snprintf(what, sizeof what, "pass depth prog %016llx", (unsigned long long)ps.prog);
        dump_image(cmd, ps.depth_tex, ps.depth_level, ps.depth_layer, what);
    }
    if (dump_on())
        for (Texture* t : dump_sampled()) {
            char what[64];
            snprintf(what, sizeof what, "sampled by %016llx", (unsigned long long)ps.prog);
            for (uint32_t l = 0; l < std::min<uint32_t>(t->layers, 2); l++) dump_image(cmd, t, 0, l, what);
            if (t->guest_size >= (8u << 20)) texture_log_guest(t);   // (and what the game's memory holds of it)
            for (uint32_t lv = 1; t->layers == 1 && lv < std::min<uint32_t>(t->levels, 8); lv++) {   // (every level)
                snprintf(what, sizeof what, "sampled by %016llx level %u", (unsigned long long)ps.prog, lv);
                dump_image(cmd, t, lv, 0, what);
            }
        }
    dump_sampled().clear();
    ps.prog = 0;
}

const std::vector<uint64_t>& record_progs();
void record_draw(const NvnState& s, const Program& p, const DrawArgs& d, const Targets& at);
void probe_draw(const NvnState& s, const Program& p, int pass_draws);
/// (RX_VK_DUMP_ARGS=<key>[,<key>...] (hex; 1: every draw): in a dump, each draw of those programs logged with its
/// arguments, its indirect arguments as guest memory holds them, its vertex buffers and the first words of the vertex
/// stage's uniform buffers)
void log_dump_draw(const NvnState& s, const Program& p, const DrawArgs& d) {
    static const std::vector<uint64_t> want = [] {
        std::vector<uint64_t> v;
        for (const char* q = getenv("RX_VK_DUMP_ARGS"); q && *q;) {
            char* end;
            const uint64_t k = strtoull(q, &end, 16);
            if (end == q) break;
            v.push_back(k);
            q = *end ? end + 1 : end;
        }
        return v;
    }();
    if (want.empty() || (want[0] != 1 && std::find(want.begin(), want.end(), p.key) == want.end())) return;
    char line[4096];
    int n = snprintf(line, sizeof line, "vk: dump-draw %03d prog %016llx prim %d count %d first %d inst %d bv %d bi %d idx %d/%d ibuf %#llx",
                     dump_seq(), (unsigned long long)p.key, d.prim, d.count, d.first, d.instances, d.base_vertex, d.base_instance,
                     (int)d.indexed, d.itype, (unsigned long long)d.ibuf);
    if (d.indirect) {
        uint32_t a[5] = {};
        host->gpu_read(d.indirect, a, sizeof a);
        uint32_t c = 0;
        if (d.count_addr) host->gpu_read(d.count_addr, &c, 4);
        n += snprintf(line + n, sizeof line - n, " ind %#llx [%u %u %u %u %u] cnt %#llx=%u max %d stride %d",
                      (unsigned long long)d.indirect, a[0], a[1], a[2], a[3], a[4], (unsigned long long)d.count_addr, c, d.max,
                      d.stride);
    }
    if (d.indexed && d.ibuf && s.nvattr > 0) {   // (the first indices, and attribute 0 of the vertices they take)
        const nvn_va_st a0 = *reinterpret_cast<const nvn_va_st*>(&s.vattr[0]);
        const uint32_t stride = s.vstream[a0.stream & 15].stride;
        const uint64_t isz = d.itype == 0 ? 1 : d.itype == 1 ? 2 : 4;
        uint8_t ib[64] = {};
        host->gpu_read(d.ibuf + (uint64_t)d.first * isz, ib, 12 * isz);
        n += snprintf(line + n, sizeof line - n, " IDX");
        for (int k = 0; k < 12; k++) {
            uint32_t ix = isz == 1 ? ib[k] : isz == 2 ? ((uint16_t*)ib)[k] : ((uint32_t*)ib)[k];
            if (k < 6 && ix != 0xFFFF && ix != 0xFFFFFFFF && s.vbuf[a0.stream & 15].address && stride) {
                float v[2] = {};
                host->gpu_read(s.vbuf[a0.stream & 15].address + (uint64_t)((int64_t)ix + d.base_vertex) * stride + a0.offset, v, 8);
                n += snprintf(line + n, sizeof line - n, " %u(%.4g,%.4g)", ix, v[0], v[1]);
            } else n += snprintf(line + n, sizeof line - n, " %u", ix);
        }
    }
    for (int i : {12, 11}) {   // (the vertex stage's c15 and c14 first: MKW's terrain patches' placement and camera)
        const nvn_p_bind_buf& u = s.ubo[0][i];
        if (!u.addr) continue;
        float f[16] = {};
        const bool upd = ub_overlay(u.addr, sizeof f, (uint8_t*)f, [](uint64_t, uint64_t) { return (uint64_t)0; }, 0) != UINT64_MAX;
        if (!upd) host->gpu_read(u.addr, f, sizeof f);
        n += snprintf(line + n, sizeof line - n, " C%d%s %#llx+%#llx [", i + 3, upd ? "*" : "", (unsigned long long)u.addr,
                      (unsigned long long)u.size);
        for (int k = 0; k < 16; k++) n += snprintf(line + n, sizeof line - n, k ? " %.6g" : "%.6g", f[k]);
        n += snprintf(line + n, sizeof line - n, "]");
    }
    for (int i = 0; i < 4 && n < 1500; i++)
        if (s.vbuf[i].address) n += snprintf(line + n, sizeof line - n, " vb%d %#llx+%#llx", i, (unsigned long long)s.vbuf[i].address,
                                             (unsigned long long)s.vbuf[i].size);
    for (int i = 0; i < s.nvattr && i < 4 && n < 1500; i++) {   // (the attributes: format, offset, stream; the streams' strides)
        const nvn_va_st a = *reinterpret_cast<const nvn_va_st*>(&s.vattr[i]);
        n += snprintf(line + n, sizeof line - n, " va%d f%u+%u s%u/%u", i, (unsigned)a.fmt, (unsigned)a.offset, (unsigned)a.stream,
                      s.vstream[a.stream & 15].stride);
    }
    for (int i = 0; i < 16 && n < 1500; i++) {   // (the vertex stage's storage buffers: their first 8 words)
        const nvn_p_bind_buf& b = s.ssbo[0][i];
        if (!b.addr) continue;
        uint32_t w[8] = {};
        host->gpu_read(b.addr, w, sizeof w);
        n += snprintf(line + n, sizeof line - n, " sb%d %#llx+%#llx [%x %x %x %x %x %x %x %x]", i, (unsigned long long)b.addr,
                      (unsigned long long)b.size, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
    }
    for (int i = 0; i < 16 && n < 1500; i++) {   // (the vertex stage's uniform buffers: their first 16 words)
        const nvn_p_bind_buf& u = s.ubo[0][i];
        if (!u.addr) continue;
        float f[16] = {};
        // (as the draw reads it: guest memory with the stream's uniform updates over it; '*': updated)
        const bool upd = ub_overlay(u.addr, sizeof f, (uint8_t*)f, [](uint64_t, uint64_t) { return (uint64_t)0; }, 0) != UINT64_MAX;
        if (!upd) host->gpu_read(u.addr, f, sizeof f);
        n += snprintf(line + n, sizeof line - n, " c%d%s %#llx+%#llx [", i + 3, upd ? "*" : "", (unsigned long long)u.addr,
                      (unsigned long long)u.size);
        for (int k = 0; k < 16 && n < 1900; k++) n += snprintf(line + n, sizeof line - n, k ? " %.5g" : "%.5g", f[k]);
        n += snprintf(line + n, sizeof line - n, "]");
    }
    logf("%s", line);
}
extern int64_t presents_now;   // (vk_replay.cpp)
// The last present whose frame drew the world (its lighting composite, 1a7482c0). A pause menu, the map, a loading screen or a
// movie draws none: then ultrawide's sides are blurred or black (vk_framegen.cpp wide_sides)
std::atomic<int64_t> world_drawn_at{0};
std::atomic<int64_t> menu_fade_world_at{-1}, menu_fade_pause_at{-1};
// The last present whose frame drew the item wheel's background (02ff9920: the last frame of play, blurred, over the whole picture;
// drawn behind the wheel only, not the pause menu or the map): no sides of ours then (vk_framegen.cpp wide_sides)
std::atomic<int64_t> wheel_drawn_at{0};
uint32_t stage_cbuf_word(const NvnState& s, const Stage& st, uint32_t index, uint32_t offset);   // (vk_shader.cpp)
// (hookshot) RX_VK_EXPOSURE_LOG=1: the final pass's exposure (c3+0x38 of its fragment stage, OoT's 73320e720a317817) at each
// frame, kept for exposure_log_dump when the frame log sees a white picture
struct ExposureRing { int64_t present; uint32_t word; };
std::mutex exposure_mu;
ExposureRing exposure_ring[32];
uint32_t exposure_at = 0;
bool exposure_log_on() {
    static const bool on = getenv("RX_VK_EXPOSURE_LOG") != nullptr;
    return on;
}
void exposure_note(const NvnState& s, const Program& p) {
    static std::unordered_map<uint64_t, bool> known;
    bool hit = false;
    {
        std::lock_guard lk(exposure_mu);
        auto it = known.find(s.prog.serial);
        if (it != known.end()) hit = it->second;
        else hit = known[s.prog.serial] = program_code_keys(s).find("73320e720a317817") != std::string::npos;
    }
    if (!hit) return;
    for (const Stage& st : p.stages)
        if (st.nvn_stage == 1) {
            const uint32_t w = stage_cbuf_word(s, st, 3, 0x38);
            std::lock_guard lk(exposure_mu);
            exposure_ring[exposure_at++ % 32] = {presents_now, w};
            static const bool every = getenv("RX_VK_EXPOSURE_LOG") && atoi(getenv("RX_VK_EXPOSURE_LOG")) == 2;
            if (every) {
                float f;
                std::memcpy(&f, &w, 4);
                logf("vk: exposure %llu %g", (unsigned long long)presents_now, f);
            }
        }
}
void exposure_log_dump(const char* why) {
    std::lock_guard lk(exposure_mu);
    char line[900];
    int n = snprintf(line, sizeof line, "vk: %s: exposure by present:", why);
    for (uint32_t i = 0; i < 32 && n < 860; i++) {
        const ExposureRing& e = exposure_ring[(exposure_at + i) % 32];
        float v;
        std::memcpy(&v, &e.word, 4);
        n += snprintf(line + n, sizeof line - n, " %lld:%.3g", (long long)e.present, v);
    }
    logf("%s", line);
}

// ---- (EvilOOT) RX_VK_DRAW_STATS=<dir>: what each program's draws produce on the GPU, for one frame. When <dir>/draw_stats.now
// appears (checked every half second; the file is removed), every draw of the next frame runs inside a pipeline statistics
// query: vertex / task / mesh shader invocations, primitives into and out of clipping, fragment shader invocations. Eight
// frames later the results are read and written to <dir>/draw_stats_<frame>.txt, one line per program, the programs whose
// draws run but output nothing first (a draw whose mesh threads run and whose primitives never reach the clipper is one that
// made no triangles: NaN positions, culled lists, empty records). ----
namespace {
constexpr uint32_t kStatsQueries = 32768;
struct StatsMeta {
    uint64_t prog;
    uint8_t mesh, indirect, indexed, args_read;
    int32_t count, instances, max, stride;
    uint64_t count_addr, args_addr;
    uint32_t guest[4], guest_count;   // (the same words as the game's memory held them when the draw was recorded)
    uint8_t dispatch;                 // (a compute dispatch: its grid, or its indirect arguments as above)
    uint32_t grid[3];
    uint64_t args_vkbuf, args_vkoff;  // (the Vulkan buffer and offset the arguments were read from)
    uint64_t codes[4]{};             // native vertex/mesh/compute, fragment, task, header-only keys at recording time
};
VkQueryPool stats_pool = VK_NULL_HANDLE;
// (indirect draws: after the draw, the first 16 bytes of its arguments and its count word as the GPU holds them, 32 bytes a
// query: args[4], count)
VkBuffer stats_args = VK_NULL_HANDLE;
VmaAllocation stats_args_mem = nullptr;
uint32_t* stats_args_map = nullptr;
// (EvilOOT) RX_VK_WATCH_ADDRS=<gpu address>[,...] (hex, at most 4): 16 bytes at each, as the GPU holds them after every
// dispatch of the measured frame; the report lists the dispatches that changed them. Which pass writes a launch count.
constexpr uint32_t kWatchMax = 4;
VkBuffer stats_watch = VK_NULL_HANDLE;
VmaAllocation stats_watch_mem = nullptr;
uint32_t* stats_watch_map = nullptr;
std::vector<uint64_t> parse_hex_list(const char* q, size_t max) {
    std::vector<uint64_t> a;
    while (q && *q && a.size() < max) {
        char* end;
        const uint64_t x = strtoull(q, &end, 16);
        if (end == q) break;
        a.push_back(x);
        q = *end ? end + 1 : end;
    }
    return a;
}
// (RX_VK_WATCH_ARGS_OF=<program>[,<program>]: the launch arguments of those dispatches instead: their addresses differ
// from start to start, and the game alternates two sets frame by frame: the last two each program used, noted at every
// dispatch, watched from the measured frame on)
std::mutex argwatch_mu;
std::vector<std::pair<uint64_t, std::array<uint64_t, 2>>> watch_seen;   // (program -> its last two argument addresses)
std::vector<uint64_t> watch_live;   // (what the measured frame watches: set when it is armed)
void watch_note(uint64_t prog, uint64_t indirect) {
    static const std::vector<uint64_t> progs = parse_hex_list(getenv("RX_VK_WATCH_ARGS_OF"), 2);
    if (progs.empty() || !indirect || std::find(progs.begin(), progs.end(), prog) == progs.end()) return;
    std::lock_guard lk(argwatch_mu);
    for (auto& [p, a] : watch_seen)
        if (p == prog) {
            if (a[0] != indirect) a[1] = a[0], a[0] = indirect;
            return;
        }
    watch_seen.push_back({prog, {indirect, 0}});
}
// (RX_VK_WATCH_SB=<program>:<slot>:<offset>[,...], hex: a compute program's storage buffer slot + offset, as it binds it)
std::vector<uint64_t> watch_sb_seen;
void watch_note_sb(uint64_t prog, const NvnState& s) {
    struct Want { uint64_t prog; uint32_t slot; uint64_t off; };
    static const std::vector<Want> want = [] {
        std::vector<Want> v;
        for (const char* q = getenv("RX_VK_WATCH_SB"); q && *q && v.size() < 2;) {
            char* e;
            Want w{};
            w.prog = strtoull(q, &e, 16);
            if (*e != ':') break;
            w.slot = (uint32_t)strtoul(e + 1, &e, 16);
            if (*e != ':') break;
            w.off = strtoull(e + 1, &e, 16);
            v.push_back(w);
            q = *e ? e + 1 : e;
        }
        return v;
    }();
    for (const Want& w : want) {   // (slot 0x10 and up: uniform buffer slot - 0x10)
        const nvn_p_bind_buf* b = w.slot < 16 ? &s.ssbo[5][w.slot] : w.slot < 32 ? &s.ubo[5][w.slot - 16] : nullptr;
        if (w.prog != prog || !b || !b->addr) continue;
        const uint64_t a = b->addr + w.off;
        std::lock_guard lk(argwatch_mu);
        if (std::find(watch_sb_seen.begin(), watch_sb_seen.end(), a) == watch_sb_seen.end()) {
            watch_sb_seen.insert(watch_sb_seen.begin(), a);
            if (watch_sb_seen.size() > 2) watch_sb_seen.pop_back();
        }
    }
}
void watch_arm() {   // (stats_tick, as a frame is armed)
    static const std::vector<uint64_t> fixed = parse_hex_list(getenv("RX_VK_WATCH_ADDRS"), kWatchMax);
    std::lock_guard lk(argwatch_mu);
    watch_live.clear();
    for (const uint64_t a : fixed) watch_live.push_back(a & ~15ull);
    for (const uint64_t a : watch_sb_seen)
        if (watch_live.size() < kWatchMax) watch_live.push_back(a & ~15ull);
    for (const auto& [p, a] : watch_seen)
        for (const uint64_t x : a)
            if (x && watch_live.size() < kWatchMax) watch_live.push_back(x & ~15ull);
}
const std::vector<uint64_t>& watch_addrs() { return watch_live; }
bool watch_wanted() { return getenv("RX_VK_WATCH_ADDRS") || getenv("RX_VK_WATCH_ARGS_OF") || getenv("RX_VK_WATCH_SB"); }
bool stats_mesh_bits = false;
StatsMeta stats_meta[kStatsQueries];
std::atomic<uint32_t> stats_used{0};
std::atomic<int64_t> stats_frame{-1};   // (the frame measured; -1 none)
std::atomic<bool> stats_written{false};
std::mutex stats_mu;
const char* stats_dir() {
    static const char* d = getenv("RX_VK_DRAW_STATS");
    return d && *d ? d : nullptr;
}
bool stats_frame_now() {
    if (!stats_dir()) return false;
    const int64_t at = stats_frame.load(std::memory_order_relaxed);
    return at >= 0 && (int64_t)texture_frame_no() == at && !stats_written.load(std::memory_order_relaxed);
}
bool stats_make_pool() {
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
    qi.queryCount = kStatsQueries;
    const VkQueryPipelineStatisticFlags base = VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
                                               VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                                               VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
                                               VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
                                               VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
    qi.pipelineStatistics = base;
    if (dev.mesh_queries) {   // (not dev.mesh_shader: see where it is set)
        qi.pipelineStatistics = base | VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT |
                                VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT;
        if (vkCreateQueryPool(dev.device, &qi, nullptr, &stats_pool) == VK_SUCCESS) {
            stats_mesh_bits = true;
            return true;
        }
        qi.pipelineStatistics = base;
    }
    return vkCreateQueryPool(dev.device, &qi, nullptr, &stats_pool) == VK_SUCCESS;
}
void stats_code_keys(FILE* f, uint32_t n) {
    // Keep distinct tuples: a program hash alone is not the recorded native stage identity.
    std::map<std::array<uint64_t, 7>, uint32_t> groups;
    for (uint32_t i = 0; i < n; i++) {
        const StatsMeta& m = stats_meta[i];
        groups[{m.prog, m.dispatch, m.mesh, m.codes[0], m.codes[1], m.codes[2], m.codes[3]}]++;
    }
    fprintf(f, "\n# native code keys: kind program queries | vertex/mesh/compute fragment task header\n");
    for (const auto& [k, count] : groups)
        fprintf(f, "code %s %016llx %u | %016llx %016llx %016llx %016llx\n", k[1] ? "comp" : k[2] ? "mesh" : "draw",
                (unsigned long long)k[0], count, (unsigned long long)k[3], (unsigned long long)k[4],
                (unsigned long long)k[5], (unsigned long long)k[6]);
}
/// (each draw) arms a capture when the trigger file is there; writes the results once the measured frame is long done
void stats_tick() {
    const char* dir = stats_dir();
    if (!dir) return;
    const int64_t frame = (int64_t)texture_frame_no();
    const int64_t at = stats_frame.load(std::memory_order_relaxed);
    if (at < 0) {
        static std::atomic<int64_t> next_check{0};
        const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now < next_check.load(std::memory_order_relaxed)) return;
        next_check.store(now + 500, std::memory_order_relaxed);
        const std::string trigger = std::string(dir) + "/draw_stats.now";   // ('/': both systems take it; '\\' made a file beside the folder on Linux)
        FILE* f = fopen(trigger.c_str(), "rb");
        if (!f) return;
        fclose(f);
        std::remove(trigger.c_str());
        std::lock_guard lk(stats_mu);
        if (stats_frame.load() >= 0) return;
        if (!stats_pool && !stats_make_pool()) {
            logf("vk: draw stats: no pipeline statistics query pool");
            return;
        }
        if (!stats_args) {
            void* m = nullptr;
            if (create_buffer((uint64_t)kStatsQueries * 32, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, stats_args, stats_args_mem, &m))
                stats_args_map = (uint32_t*)m;
        }
        if (stats_args_map) std::memset(stats_args_map, 0xEE, (size_t)kStatsQueries * 32);   // (0xEEEEEEEE: never copied)
        watch_arm();
        if (watch_wanted() && !stats_watch) {
            void* m = nullptr;
            if (create_buffer((uint64_t)kStatsQueries * kWatchMax * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, stats_watch, stats_watch_mem, &m))
                stats_watch_map = (uint32_t*)m;
        }
        if (stats_watch_map) std::memset(stats_watch_map, 0xEE, (size_t)kStatsQueries * kWatchMax * 16);
        vkResetQueryPool(dev.device, stats_pool, 0, kStatsQueries);
        {
            std::lock_guard bl(stats_bind_mu);
            stats_binds.clear();
            stats_ptrs.clear();
            stats_ptrs2.clear();
            stats_dumps.clear();
            stats_gpu_recs.clear();
        }
        {
            std::lock_guard ml(miss_mu);   // (the misses from here on: older frames' in-flight work may still add some)
            if (miss_map) std::memset(miss_map, 0, (size_t)kMissSlice * kMissSlices);
        }
        stats_used.store(0);
        stats_written.store(false);
        stats_frame.store(frame + 2);   // (a frame not yet started on any replay thread)
        logf("vk: draw stats: measuring frame %lld%s", (long long)(frame + 2), stats_mesh_bits ? " (with task and mesh counts)" : "");
        return;
    }
    if (frame < at + 8 || stats_written.load()) return;
    std::lock_guard lk(stats_mu);
    if (stats_written.exchange(true)) return;
    const uint32_t n = std::min(stats_used.load(), kStatsQueries);
    const uint32_t words = stats_mesh_bits ? 7 : 5;   // (in bit order: vs, clip in, clip out, fs, cs, [task, mesh])
    std::vector<uint64_t> r((size_t)n * words);
    VkResult res = n ? vkGetQueryPoolResults(dev.device, stats_pool, 0, n, r.size() * 8, r.data(), (VkDeviceSize)words * 8,
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
                     : VK_SUCCESS;
    struct Sum {
        uint64_t draws = 0, vs = 0, clip_in = 0, clip_out = 0, fs = 0, cs = 0, task = 0, mesh = 0;
        bool is_mesh = false, indirect = false, is_dispatch = false;
        std::string args;   // (indirect: the GPU's argument words and count of its first draws)
    };
    std::map<uint64_t, Sum> by;
    for (uint32_t i = 0; i < n && res == VK_SUCCESS; i++) {
        const StatsMeta& m = stats_meta[i];
        Sum& x = by[m.prog];
        const uint64_t* v = &r[(size_t)i * words];
        x.draws++, x.vs += v[0], x.clip_in += v[1], x.clip_out += v[2], x.fs += v[3], x.cs += v[4];
        if (stats_mesh_bits) x.task += v[5], x.mesh += v[6];
        x.is_mesh |= m.mesh != 0, x.indirect |= m.indirect != 0, x.is_dispatch |= m.dispatch != 0;
        if (m.dispatch && x.draws == 1) x.args += " #" + std::to_string(i);   // (its place in the frame's recording)
        if (m.dispatch && !m.indirect && x.args.size() < 200) {
            char b[64];
            snprintf(b, sizeof b, " [grid %u %u %u]", m.grid[0], m.grid[1], m.grid[2]);
            x.args += b;
        }
        if (m.indirect && x.args.size() < 200) {
            char b[160];
            const uint32_t* a = stats_args_map ? stats_args_map + (size_t)i * 8 : nullptr;
            const uint32_t* g = m.guest;
            if (!a || !m.args_read)
                snprintf(b, sizeof b, " [@%llx gpu not read, mem %u %u %u %u]", (unsigned long long)m.args_addr, g[0], g[1], g[2], g[3]);
            else if (m.count_addr)
                snprintf(b, sizeof b, " [@%llx gpu %u %u %u %u cnt=%u | mem %u %u %u %u cnt=%u]", (unsigned long long)m.args_addr, a[0],
                         a[1], a[2], a[3], a[4], g[0], g[1], g[2], g[3], m.guest_count);
            else
                snprintf(b, sizeof b, " [@%llx gpu %u %u %u %u | mem %u %u %u %u]", (unsigned long long)m.args_addr, a[0], a[1], a[2],
                         a[3], g[0], g[1], g[2], g[3]);
            x.args += b;
        }
    }
    std::vector<std::pair<uint64_t, Sum>> rows(by.begin(), by.end());
    // (no output first: shaders ran and nothing reached the clipper, then nothing shaded)
    const auto rank = [](const Sum& x) {
        if (x.is_dispatch) return x.cs ? 5 : 4;   // (dispatches after the draws: those that ran no thread first)
        const bool ran = x.vs || x.mesh || x.task;
        return !ran ? 2 : !x.clip_in ? 0 : !x.fs ? 1 : 3;
    };
    std::stable_sort(rows.begin(), rows.end(), [&](const auto& a, const auto& b) { return rank(a.second) < rank(b.second); });
    const std::string path = std::string(dir) + "/draw_stats_" + std::to_string(at) + ".txt";
    if (FILE* f = fopen(path.c_str(), "wb")) {
        fprintf(f, "# frame %lld: %u draws, %zu programs (query result %d). kind prog draws | vs task mesh | clip_in clip_out | fs | note\n",
                (long long)at, n, rows.size(), (int)res);
        for (const auto& [prog, x] : rows) {
            const int k = rank(x);
            if (x.is_dispatch) {   // (compute: its dispatches, the threads they ran)
                fprintf(f, "comp %016llx %6llu | cs threads %12llu | %s%s\n", (unsigned long long)prog, (unsigned long long)x.draws,
                        (unsigned long long)x.cs, k == 4 ? "NO THREADS" : "", (x.indirect ? " (indirect)" + x.args : x.args).c_str());
                continue;
            }
            fprintf(f, "%s %016llx %6llu | %10llu %8llu %10llu | %10llu %10llu | %11llu | %s%s\n", x.is_mesh ? "mesh" : "draw",
                    (unsigned long long)prog, (unsigned long long)x.draws, (unsigned long long)x.vs, (unsigned long long)x.task,
                    (unsigned long long)x.mesh, (unsigned long long)x.clip_in, (unsigned long long)x.clip_out,
                    (unsigned long long)x.fs, k == 0 ? "RAN, NO PRIMITIVES" : k == 1 ? "primitives, no fragments" : k == 2 ? "no shader work" : "",
                    x.indirect ? (" (indirect)" + x.args).c_str() : "");
        }
        if (res == VK_SUCCESS) stats_code_keys(f, n);
        // who binds each indirect draw's argument words in this frame: written (W) or only read (r), by which programs
        std::vector<StatsBind> binds;
        {
            std::lock_guard bl(stats_bind_mu);
            binds = stats_binds;
        }
        // (RX_VK_DRAW_STATS_PROGS=<key>,...: every storage buffer those programs bound in this frame)
        if (const char* want = getenv("RX_VK_DRAW_STATS_PROGS")) {
            for (const char* q = want; *q;) {
                char* end;
                const uint64_t k = strtoull(q, &end, 16);
                if (end == q) break;
                q = *end ? end + 1 : end;
                fprintf(f, "\n# bindings of %016llx:\n", (unsigned long long)k);
                std::vector<std::pair<uint64_t, uint64_t>> seen;
                for (const StatsBind& b : binds) {
                    const std::pair<uint64_t, uint64_t> slot{b.addr ^ ((uint64_t)b.off << 40) ^ ((uint64_t)b.cbuf << 56), b.size};
                    if (b.prog != k || std::find(seen.begin(), seen.end(), slot) != seen.end()) continue;
                    seen.push_back(slot);
                    fprintf(f, "  s%d %s %#llx +%#llx  from c%u+%#x (raw %#llx)%s%s\n", b.stage, b.written ? "W" : "r",
                            (unsigned long long)b.addr, (unsigned long long)b.size, b.cbuf, b.off, (unsigned long long)b.raw,
                            b.table ? " (table entry)" : "", !b.addr ? "  <-- UNRESOLVED" : "");
                }
            }
        }
        {
            std::lock_guard bl(stats_bind_mu);
            for (const std::string& d : stats_dumps) fprintf(f, "\n%s", d.c_str());
            for (const GpuDumpRec& r : stats_gpu_recs) {   // (as the GPU held them after the wanted draws)
                if (!stats_gpu_map) break;
                const uint32_t* w = (const uint32_t*)(stats_gpu_map + (size_t)r.slot * kGpuDumpSlot);
                uint64_t nonzero = 0;
                uint32_t first = UINT32_MAX;
                for (uint32_t i = 0; i < r.bytes / 4; i++)
                    if (w[i]) nonzero++, first = std::min(first, i);
                fprintf(f, "\n%016llx s%d ssbo%d at %#llx AS THE GPU HELD IT after the %s (%llu of %u words non-zero, first at +%x):\n",
                        (unsigned long long)r.prog, r.stage, r.index, (unsigned long long)r.addr, r.stage == 5 ? "dispatch" : "draw",
                        (unsigned long long)nonzero, r.bytes / 4, first == UINT32_MAX ? 0 : first * 4);
                const uint32_t from = first == UINT32_MAX ? 0 : first & ~3u;
                for (uint32_t i = from; i + 4 <= r.bytes / 4 && i < from + 64; i += 4)
                    fprintf(f, "  +%05x: %08x %08x %08x %08x\n", i * 4, w[i], w[i + 1], w[i + 2], w[i + 3]);
            }
        }
        // (second level) pointers found in compute programs' pointer table regions that no binding or region of theirs covers
        {
            std::lock_guard bl(stats_bind_mu);
            std::map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> by;   // program -> (region, pointer)
            for (const StatsPtr& q : stats_ptrs2) by[q.prog].emplace_back(q.in_buffer, q.value);
            fprintf(f, "\n# second-level pointers no binding covers: %zu programs\n", by.size());
            for (auto& [prog, v] : by) {
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
                std::string t;
                for (size_t i = 0; i < v.size() && i < 10; i++) {
                    char b[96];
                    snprintf(b, sizeof b, " %#llx(in %#llx, pool end %#llx)", (unsigned long long)v[i].second,
                             (unsigned long long)v[i].first, (unsigned long long)gpu_pool_end(v[i].second));
                    t += b;
                }
                fprintf(f, "  %016llx %zu pointers:%s\n", (unsigned long long)prog, v.size(), t.c_str());
            }
        }
        // (RX_VK_MISS_LOG) the global memory accesses no binding held, since the capture was armed, by program: how many,
        // and the first addresses with the pool each lies in
        {
            std::lock_guard ml(miss_mu);
            std::vector<std::pair<uint32_t, uint64_t>> rows;   // (misses, program)
            if (miss_map)
                for (const auto& [prog, slot] : miss_slot)
                    if (const uint32_t m = miss_map[(size_t)slot * (kMissSlice / 4)]) rows.emplace_back(m, prog);
            std::sort(rows.rbegin(), rows.rend());
            fprintf(f, "\n# global memory misses (RX_VK_MISS_LOG): %zu programs\n", rows.size());
            for (const auto& [m, prog] : rows) {
                const uint32_t* w = miss_map + (size_t)miss_slot[prog] * (kMissSlice / 4);
                std::string addrs;
                std::vector<uint64_t> uniq;
                for (uint32_t i = 0; i < std::min<uint32_t>(m, 63); i++) {
                    const uint64_t a = w[1 + 2 * i] | (uint64_t)w[2 + 2 * i] << 32;
                    if (std::find(uniq.begin(), uniq.end(), a) != uniq.end() || uniq.size() >= 12) continue;
                    uniq.push_back(a);
                    char t[64];
                    snprintf(t, sizeof t, " %#llx(pool end %#llx)", (unsigned long long)a, (unsigned long long)gpu_pool_end(a));
                    addrs += t;
                }
                fprintf(f, "  %016llx %u misses:%s\n", (unsigned long long)prog, m, addrs.c_str());
            }
        }
        if (getenv("RX_VK_BLOCK_PROBE") && miss_map) {   // (EvilOOT) the probed programs' blocks: invocations that entered each
            std::lock_guard ml(miss_mu);
            for (const auto& [prog, slot] : miss_slot) {
                const uint32_t* w = miss_map + (size_t)slot * (kMissSlice / 4);
                fprintf(f, "\n# block probe %016llx (invocations entering IR Block $n, since the frame was armed)\n", (unsigned long long)prog);
                for (uint32_t b = 0; 128 + b < kMissSlice / 4; b++)
                    if (w[128 + b]) fprintf(f, "  $%u %u\n", b, w[128 + b]);
            }
        }
        if (const std::vector<uint64_t>& w = watch_addrs(); !w.empty() && stats_watch_map) {   // (RX_VK_WATCH_ADDRS)
            fprintf(f, "\n# watched addresses: bindings over them this frame (W written; tables left out)\n");
            for (const uint64_t a : w) {
                std::vector<std::string> seen;
                for (const StatsBind& b : binds) {
                    if (b.table || !b.addr || a < b.addr || a >= b.addr + b.size) continue;
                    char t[96];
                    snprintf(t, sizeof t, "%s%016llx@s%d[%#llx+%#llx]", b.written ? "W:" : "r:", (unsigned long long)b.prog, b.stage,
                             (unsigned long long)b.addr, (unsigned long long)b.size);
                    if (std::find(seen.begin(), seen.end(), t) == seen.end() && seen.size() < 40) seen.push_back(t);
                }
                fprintf(f, "  @%llx:", (unsigned long long)a);
                for (const std::string& s2 : seen) fprintf(f, " %s", s2.c_str());
                fprintf(f, "\n");
            }
            fprintf(f, "\n# watched addresses, after each dispatch (shown where they changed; #n = its place in the frame)\n");
            std::vector<std::array<uint32_t, 4>> last(w.size(), {0xEEEEEEEEu, 0xEEEEEEEEu, 0xEEEEEEEEu, 0xEEEEEEEEu});
            for (uint32_t i = 0; i < n; i++) {
                if (!stats_meta[i].dispatch) continue;
                for (size_t k = 0; k < w.size(); k++) {
                    const uint32_t* v = stats_watch_map + ((size_t)i * kWatchMax + k) * 4;
                    if (v[0] == 0xEEEEEEEEu && v[1] == 0xEEEEEEEEu) continue;   // (not copied)
                    std::array<uint32_t, 4> now{v[0], v[1], v[2], v[3]};
                    if (now == last[k]) continue;
                    fprintf(f, "  #%u %016llx: @%llx = %u %u %u %u (was %u %u %u %u)\n", i, (unsigned long long)stats_meta[i].prog,
                            (unsigned long long)w[k], now[0], now[1], now[2], now[3], last[k][0], last[k][1], last[k][2], last[k][3]);
                    last[k] = now;
                }
            }
        }
        // (census) single storage buffer descriptors that resolved to nothing in this frame, by program and slot: a shader
        // that reads or writes a buffer it was never given. Empty table entries are left out (tables have holes).
        {
            std::map<std::tuple<uint64_t, int, uint32_t, uint32_t>, std::tuple<uint64_t, bool, uint64_t, uint64_t, int>> unresolved;
            for (const StatsBind& b : binds)   // (-> raw, written, the uniform buffer's address and size, times seen)
                if (!b.table && !b.addr) {
                    auto& e = unresolved[{b.prog, b.stage, b.cbuf, b.off}];
                    e = {b.raw, b.written, b.vkbuf, b.vkoff, std::get<4>(e) + 1};
                }
            fprintf(f, "\n# unresolved single descriptors: %zu\n", unresolved.size());
            for (const auto& [k, v] : unresolved)
                fprintf(f, "  %016llx s%d c%u+%#x raw %#llx %s (x%d; from the uniform buffer at %#llx size %#llx)\n",
                        (unsigned long long)std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k),
                        (unsigned long long)std::get<0>(v), std::get<1>(v) ? "WRITTEN" : "read", std::get<4>(v),
                        (unsigned long long)std::get<2>(v), (unsigned long long)std::get<3>(v));
        }
        fprintf(f, "\n# indirect arguments: the storage buffers of this frame that hold them (%zu bindings noted)\n", binds.size());
        std::vector<uint64_t> done;
        for (uint32_t i = 0; i < n; i++) {
            const StatsMeta& m = stats_meta[i];
            if (!m.indirect || std::find(done.begin(), done.end(), m.args_addr) != done.end()) continue;
            done.push_back(m.args_addr);
            std::string who;
            std::vector<std::pair<uint64_t, bool>> seen;
            for (const StatsBind& b : binds) {
                if (m.args_addr < b.addr || m.args_addr >= b.addr + b.size) continue;
                const std::pair<uint64_t, bool> k{b.prog, b.written};
                if (std::find(seen.begin(), seen.end(), k) != seen.end()) continue;
                seen.push_back(k);
                char t[160];
                // (vk: the Vulkan buffer and the offset in it this binding gives the argument words; the read's own after the
                // list: different ones mean two copies of the same memory)
                snprintf(t, sizeof t, " %s%016llx@s%d[%#llx+%#llx]%s vk %llx+%#llx", b.written ? "W:" : "r:", (unsigned long long)b.prog,
                         b.stage, (unsigned long long)b.addr, (unsigned long long)b.size, b.table ? "(table)" : "",
                         (unsigned long long)b.vkbuf, (unsigned long long)(b.vkoff + (m.args_addr - b.addr)));
                if (who.size() < 3000) who += t;
            }
            fprintf(f, "args %#llx (draw prog %016llx, read from vk %llx+%#llx):%s\n", (unsigned long long)m.args_addr,
                    (unsigned long long)m.prog, (unsigned long long)m.args_vkbuf, (unsigned long long)m.args_vkoff,
                    who.empty() ? " NO BINDING HOLDS THEM" : who.c_str());
            // (and the dispatches whose small buffers hold a pointer into the buffer around them: 1 MB either side)
            std::string ptr_who;
            std::vector<uint64_t> pseen;
            for (const StatsPtr& q : stats_ptrs) {
                if (q.value + (1ull << 20) < m.args_addr || q.value > m.args_addr + (1ull << 20)) continue;
                if (std::find(pseen.begin(), pseen.end(), q.prog ^ q.value) != pseen.end()) continue;
                pseen.push_back(q.prog ^ q.value);
                char t[96];
                snprintf(t, sizeof t, " %016llx(%#llx in %#llx)", (unsigned long long)q.prog, (unsigned long long)q.value,
                         (unsigned long long)q.in_buffer);
                if (ptr_who.size() < 1200) ptr_who += t;
            }
            if (!ptr_who.empty()) fprintf(f, "     pointed to by:%s\n", ptr_who.c_str());
        }
        fclose(f);
    }
    logf("vk: draw stats: frame %lld, %u draws of %zu programs -> %s", (long long)at, n, rows.size(), path.c_str());
    stats_frame.store(-1);   // (ready for another trigger)
}
/// One draw's query, ended on every way out of draw()
struct StatsQuery {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    uint32_t index = 0;
    DrawArgs own{};
    const DrawArgs* draw = &own;
    StatsQuery(VkCommandBuffer c, const Program& p, const DrawArgs& d) { begin(c, p, d, false, nullptr); }
    /// a compute dispatch: its grid, or its indirect arguments (12 bytes, read back as a draw's)
    StatsQuery(VkCommandBuffer c, const Program& p, uint64_t indirect, uint32_t x, uint32_t y, uint32_t z) {
        if (stats_dir()) watch_note(p.key, indirect);
        DrawArgs d{};
        d.indirect = indirect;
        const uint32_t grid[3] = {x, y, z};
        begin(c, p, d, true, grid);
    }
    void begin(VkCommandBuffer c, const Program& p, const DrawArgs& d, bool dispatch, const uint32_t* grid) {
        if (!stats_dir() || !stats_pool) return;
        const int64_t at = stats_frame.load(std::memory_order_relaxed);
        if (at < 0 || (int64_t)texture_frame_no() != at || stats_written.load(std::memory_order_relaxed)) return;
        const uint32_t i = stats_used.fetch_add(1);
        if (i >= kStatsQueries) return;
        stats_meta[i] = {p.key, (uint8_t)d.mesh, (uint8_t)(d.indirect != 0), (uint8_t)d.indexed, 0, d.count, d.instances, d.max,
                         d.stride, d.count_addr, d.indirect, {}, 0, (uint8_t)dispatch, {}, 0, 0};
        std::memcpy(stats_meta[i].codes, p.codes, sizeof stats_meta[i].codes);
        if (grid) std::memcpy(stats_meta[i].grid, grid, sizeof stats_meta[i].grid);
        if (d.indirect) {
            host->gpu_read(d.indirect, stats_meta[i].guest, dispatch ? 12 : 16);
            if (d.count_addr) host->gpu_read(d.count_addr, &stats_meta[i].guest_count, 4);
        }
        own = d;
        is_dispatch = dispatch;
        if (dispatch) copy_args(c, own, i);   // (before the dispatch: outside any pass already)
        vkCmdBeginQuery(c, stats_pool, i, 0);
        cmd = c, index = i;
    }
    bool is_dispatch = false;
    const NvnState* state = nullptr;   // (a wanted draw: its mesh stage's storage buffers copied from the GPU after it)
    uint64_t prog = 0;
    ~StatsQuery() {
        if (!cmd) return;
        vkCmdEndQuery(cmd, stats_pool, index);
        if (!is_dispatch) copy_args(cmd, *draw, index);   // (a dispatch's were copied before it ran: begin)
        if (is_dispatch) copy_watch(cmd, index);
        if (state) gpu_dump(cmd, *state, prog, is_dispatch);
    }
    /// (RX_VK_DRAW_STATS_PROGS) the first 1 KB of each storage buffer of the mesh stage, as the GPU holds it after the draw,
    /// once per program: into stats_gpu (host-visible), printed with the report beside the memory view
    static void gpu_dump(VkCommandBuffer cmd, const NvnState& s, uint64_t prog, bool is_dispatch_dump = false);
    static void copy_args(VkCommandBuffer cmd, const DrawArgs& d, uint32_t index);
    static void copy_watch(VkCommandBuffer cmd, uint32_t index);
};
void StatsQuery::copy_watch(VkCommandBuffer cmd, uint32_t index) {
    const std::vector<uint64_t>& w = watch_addrs();
    if (w.empty() || !stats_watch) return;
    end_pass(cmd);
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
    for (size_t k = 0; k < w.size(); k++) {
        const BufferRef r = gpu_buffer_bind(w[k], 16, false);
        if (!r) continue;
        VkBufferCopy c{r.offset, ((VkDeviceSize)index * kWatchMax + k) * 16, 16};
        vkCmdCopyBuffer(cmd, r.buffer, stats_watch, 1, &c);
    }
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_HOST_READ_BIT;
    vkCmdPipelineBarrier2(cmd, &di);
}
void StatsQuery::gpu_dump(VkCommandBuffer cmd, const NvnState& s, uint64_t prog, bool is_dispatch_dump) {
    std::lock_guard bl(stats_bind_mu);
    for (const GpuDumpRec& r : stats_gpu_recs)
        if (r.prog == prog) return;
    if (!stats_gpu) {
        void* m = nullptr;
        if (!create_buffer((uint64_t)kGpuDumpSlot * kGpuDumpSlots, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, stats_gpu, stats_gpu_mem, &m))
            return;
        stats_gpu_map = (uint8_t*)m;
    }
    end_pass(cmd);
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
    for (const int stage : {7, 6, 5})
        for (int i = 0; i < 16; i++) {
            if ((stage == 5) != is_dispatch_dump) continue;   // (EvilOOT: a dispatch's own, stage 5)
            const nvn_p_bind_buf& b = s.ssbo[stage][i];
            if (!b.addr || !b.size || stats_gpu_recs.size() >= kGpuDumpSlots) continue;
            const uint32_t bytes = (uint32_t)std::min<uint64_t>(b.size, kGpuDumpSlot) & ~3u;
            const BufferRef r = gpu_buffer_bind(b.addr, bytes, false);
            if (!r) continue;
            const uint32_t slot = (uint32_t)stats_gpu_recs.size();
            VkBufferCopy c{r.offset, (VkDeviceSize)slot * kGpuDumpSlot, bytes};
            vkCmdCopyBuffer(cmd, r.buffer, stats_gpu, 1, &c);
            stats_gpu_recs.push_back({prog, b.addr, slot, bytes, stage, i});
        }
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_HOST_READ_BIT;
    vkCmdPipelineBarrier2(cmd, &di);
}
/// The arguments as the GPU holds them: copied outside the pass, which the next draw opens again, as RX_VK_DUMP_EACH
/// does. A dispatch's before the dispatch (what it launched with: it may reset them itself), a draw's after it.
// (hookshot) A draw's occlusion query (vk_replay.cpp, RX_VK_OCC_QUERIES). A query only says something when its draw drew: it
// is begun just before the draw command is recorded (issue), inside the draw's pass, and ended as draw() returns. A draw
// that returns before its command, or whose vertex input could not all be bound (zero_buffer in its place: drop), never
// begins it: the query has no result, and its report stays "visible" (occ_collect). An empty query ended would read 0, an
// object hidden that may be in view. Made after StatsQuery, so it ends first: StatsQuery's end copies outside the pass.
enum OccNotBegun { kOccReturned, kOccVertexInput, kOccTaskBatches };
std::atomic<uint64_t> occ_not_begun[3];
struct OccGuard {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkQueryPool pool = VK_NULL_HANDLE;
    uint32_t q = 0;
    bool begun = false;
    OccGuard(VkCommandBuffer c, const DrawArgs& d) : cmd(c), pool(d.occ_pool), q(d.occ_query) {}
    OccGuard(const OccGuard&) = delete;
    OccGuard& operator=(const OccGuard&) = delete;
    /// the draw command comes next: its query begins
    void issue() {
        if (!pool || begun) return;
        vkCmdBeginQuery(cmd, pool, q, dev.occlusion_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
        begun = true;
    }
    /// the draw's count would not be the game's (why: OccNotBegun): its query is never begun
    void drop(OccNotBegun why) {
        if (!pool || begun) return;
        occ_not_begun[why].fetch_add(1, std::memory_order_relaxed);
        pool = VK_NULL_HANDLE;
    }
    ~OccGuard() {
        if (begun) vkCmdEndQuery(cmd, pool, q);
        else drop(kOccReturned);
    }
};

void StatsQuery::copy_args(VkCommandBuffer cmd, const DrawArgs& d, uint32_t index) {
    {
        if (!d.indirect || !stats_args) return;
        const BufferRef args = gpu_buffer_bind(d.indirect, 16, false);
        const BufferRef cnt = d.count_addr ? gpu_buffer_bind(d.count_addr, 4, false) : BufferRef{};
        if (!args) return;
        stats_meta[index].args_vkbuf = (uint64_t)(uintptr_t)args.buffer, stats_meta[index].args_vkoff = args.offset;
        end_pass(cmd);
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        VkBufferCopy c{args.offset, (VkDeviceSize)index * 32, 16};
        vkCmdCopyBuffer(cmd, args.buffer, stats_args, 1, &c);
        if (cnt) {
            VkBufferCopy cc{cnt.offset, (VkDeviceSize)index * 32 + 16, 4};
            vkCmdCopyBuffer(cmd, cnt.buffer, stats_args, 1, &cc);
        }
        VkMemoryBarrier2 back{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        back.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, back.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        back.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
        back.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_HOST_READ_BIT;
        di.pMemoryBarriers = &back;
        vkCmdPipelineBarrier2(cmd, &di);
        stats_meta[index].args_read = 1;
    }
}
} // namespace
/// (hookshot) the occlusion queries draw() left unbegun since the last call, by why (vk_draw.h)
void occ_draw_counts(uint64_t out[3]) {
    for (int i = 0; i < 3; i++) out[i] = occ_not_begun[i].exchange(0, std::memory_order_relaxed);
}

// ---- (Hookshot) real occlusion counts: RX_VK_OCC_REAL=<mode> ----
// The samples-passed reports are written on the CPU in stream order with RX_VK_OCC_FIXED (4096, "everything visible"). Here
// every draw the replay records also runs inside a Vulkan occlusion query, one slot a draw (or dispatch) of the stream (its
// serial number in the stream, the same on the dry pass and on the parallel segments' threads), in a host-reset ring of 64K
// slots. A report's count is the sum of the slots between the reset before it and it, read once its submission is done.
// 0 (default): off, no queries.  1: measured and logged only (the fixed value is still written: the picture is unchanged).
// 2: the fixed value, or 0 when the same report address read 0 the last time it was measured (one frame late).
// 3: the last measured count itself (fixed until the address has one).
// 4: same frame: a render condition on a report not read yet ends the recording there (a parallel replay's dry pass stops
// there, and its segments up to it are recorded), submits it and waits for the GPU; every report's count is then written
// over the fixed value in guest memory before the condition reads it. OoT: one wait a frame (its ~900 test draws come
// first, the first condition 550+ draws after the last of them).
// 5: one frame late, by the report's place in the frame: OoT's report addresses differ from frame to frame (none repeats
// within 4 frames; modes 2 and 3 read some old frame's count of another object: the washed-out picture), but the k-th
// report of a frame tests the same object as the k-th of the frame before (99% of zero / non-zero agree, standing).
// Key: the program bound at the report, its window's draws, and its place among the frame's reports with those two.
// The count measured for that key at most 3 frames before is written; the fixed value until there is one.
// RX_VK_OCC_LOG=<dir>: RX_VK_OCC_LOG_FRAMES frames (default 2) every RX_VK_OCC_LOG_EVERY presents (default 600) written to <dir>/occ_<present>.txt: every
// report (window, draws, real count), every render condition, every draw in a window with its pipeline statistics.
thread_local int64_t occ_draw_slot = -1;   // (the replay: the serial of the draw it hands to draw; -1: none)
int occ_mode() {
    static const int m = getenv("RX_VK_OCC_REAL") ? atoi(getenv("RX_VK_OCC_REAL")) : 0;
    return m;
}
namespace {
constexpr uint32_t kOccRing = 1u << 16, kOccChunk = 1u << 12;
enum : uint8_t { OCC_BEGUN = 1, OCC_MESH = 2, OCC_INDIRECT = 4, OCC_SKIPPED = 8, OCC_STATS = 16, OCC_INDEXED = 32 };
struct OccMeta { uint64_t prog; uint8_t flags; };
OccMeta occ_meta[kOccRing];
VkQueryPool occ_pool = VK_NULL_HANDLE, occ_stats_pool = VK_NULL_HANDLE;
bool occ_precise = false, occ_stats_mesh = false, occ_failed = false;
std::once_flag occ_once;
uint64_t occ_chunk_value[kOccRing / kOccChunk];   // (the last submission value whose draws used each chunk: the replay thread's)
std::atomic<uint64_t> occ_log_lo{UINT64_MAX}, occ_log_hi{UINT64_MAX};   // (serials of the frames being logged)
const char* occ_log_dir() {
    static const char* d = getenv("RX_VK_OCC_LOG");
    return d && *d ? d : nullptr;
}
void occ_init() {
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    vkGetPhysicalDeviceFeatures2(dev.physical, &f);
    occ_precise = f.features.occlusionQueryPrecise != VK_FALSE;
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_OCCLUSION;
    qi.queryCount = kOccRing;
    if (vkCreateQueryPool(dev.device, &qi, nullptr, &occ_pool) != VK_SUCCESS) {
        occ_failed = true;
        logf("vk: occlusion: no query pool; the fixed value stays");
        return;
    }
    vkResetQueryPool(dev.device, occ_pool, 0, kOccRing);
    if (occ_log_dir() && f.features.pipelineStatisticsQuery) {
        VkQueryPoolCreateInfo si{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        si.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        si.queryCount = kOccRing;
        const VkQueryPipelineStatisticFlags base =
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
        si.pipelineStatistics = base;
        if (dev.mesh_shader) {
            si.pipelineStatistics = base | VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT |
                                    VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT;
            occ_stats_mesh = vkCreateQueryPool(dev.device, &si, nullptr, &occ_stats_pool) == VK_SUCCESS;
        }
        if (!occ_stats_mesh) {
            si.pipelineStatistics = base;
            vkCreateQueryPool(dev.device, &si, nullptr, &occ_stats_pool);
        }
        if (occ_stats_pool) vkResetQueryPool(dev.device, occ_stats_pool, 0, kOccRing);
    }
    logf("vk: occlusion: real counts, mode %d (1 log only, 2 zero-or-fixed, 3 counts), %u query ring%s%s", occ_mode(), kOccRing,
         occ_precise ? ", precise" : "", occ_stats_pool ? ", pipeline statistics in logged frames" : "");
}
}  // namespace
/// One draw's occlusion query (and in logged frames its pipeline statistics), ended on every way out of draw
struct OccQuery {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    uint32_t slot = 0;
    bool stats = false;
    OccQuery(VkCommandBuffer c, uint64_t prog, const DrawArgs& d, bool stats_busy) {
        if (occ_draw_slot < 0 || !occ_pool) return;
        const uint64_t serial = (uint64_t)occ_draw_slot;
        slot = (uint32_t)(serial % kOccRing);
        uint8_t fl = OCC_BEGUN | (d.mesh ? OCC_MESH : 0) | (d.indirect ? OCC_INDIRECT : 0) | (d.indexed ? OCC_INDEXED : 0);
        stats = occ_stats_pool && !stats_busy && serial >= occ_log_lo.load(std::memory_order_relaxed) &&
                serial < occ_log_hi.load(std::memory_order_relaxed);
        if (stats) fl |= OCC_STATS;
        occ_meta[slot] = {prog, fl};
        occ_skipped_here = false;
        vkCmdBeginQuery(c, occ_pool, slot, occ_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
        if (stats) vkCmdBeginQuery(c, occ_stats_pool, slot, 0);
        cmd = c;
    }
    ~OccQuery() {
        if (!cmd) return;
        if (stats) vkCmdEndQuery(cmd, occ_stats_pool, slot);
        vkCmdEndQuery(cmd, occ_pool, slot);
        if (occ_skipped_here) occ_meta[slot].flags |= OCC_SKIPPED;
    }
};
namespace {
struct OccPending {
    uint64_t addr, lo, hi, value, key;
    int64_t present;
    uint64_t seq;
    bool logged;
};
std::deque<OccPending> occ_pending;   // (the replay thread's, in stream order)
std::unordered_map<uint64_t, int> occ_pending_at;     // (report address -> reports pending there)
std::unordered_map<uint64_t, uint64_t> occ_latest;    // (report address -> the seq of its latest report)
struct OccKeyed { uint64_t count; int64_t present; };
std::unordered_map<uint64_t, OccKeyed> occ_keyed;     // (mode 5: key -> its last measured count)
std::unordered_map<uint64_t, uint32_t> occ_key_ord;   // (mode 5: this frame's reports so far by program and window)
std::unordered_map<uint64_t, uint64_t> occ_last;   // (report address -> its last measured count)
FILE* occ_log = nullptr;
uint64_t occ_seq = 0;
int64_t occ_presents = 0;
// (every 600 presents: reports a frame, of them 0, and the 0s by what their windows held)
struct OccTally {
    int64_t reports = 0, zero = 0, empty = 0, unrecorded = 0, skipped = 0, mesh = 0, indirect = 0, plain = 0, draws = 0;
    int64_t unknown_write = 0, zero_write = 0;
} occ_t;
void occ_resolve(size_t n) {   // (the first n pending: their submissions are done)
    if (!n) return;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (size_t i = 0; i < n; i++) lo = std::min(lo, occ_pending[i].lo), hi = std::max(hi, occ_pending[i].hi);
    std::vector<uint64_t> res;   // (per slot: count, available)
    std::vector<uint64_t> st;    // (logged slots: clip, fs, [task, mesh], available)
    const uint32_t sw = occ_stats_mesh ? 5 : 3;
    const bool span_ok = hi > lo && hi - lo <= kOccRing / 2;
    if (span_ok) {
        res.assign((size_t)(hi - lo) * 2, 0);
        for (uint64_t at = lo; at < hi;) {
            const uint32_t first = (uint32_t)(at % kOccRing), cnt = (uint32_t)std::min<uint64_t>(hi - at, kOccRing - first);
            vkGetQueryPoolResults(dev.device, occ_pool, first, cnt, (size_t)cnt * 16, &res[(size_t)(at - lo) * 2], 16,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            at += cnt;
        }
        if (occ_log && occ_stats_pool) {
            st.assign((size_t)(hi - lo) * sw, 0);
            for (uint64_t at = lo; at < hi;) {
                const uint32_t first = (uint32_t)(at % kOccRing), cnt = (uint32_t)std::min<uint64_t>(hi - at, kOccRing - first);
                vkGetQueryPoolResults(dev.device, occ_stats_pool, first, cnt, (size_t)cnt * sw * 8, &st[(size_t)(at - lo) * sw],
                                      (VkDeviceSize)sw * 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
                at += cnt;
            }
        }
    }
    static uint64_t logged_upto = 0;   // (draw lines written once)
    for (size_t i = 0; i < n; i++) {
        const OccPending& r = occ_pending[i];
        uint64_t sum = 0;
        int draws = 0, unrec = 0, skipped = 0, mesh = 0, indirect = 0;
        if (span_ok)
            for (uint64_t k = r.lo; k < r.hi; k++) {
                const uint64_t* v = &res[(size_t)(k - lo) * 2];
                const OccMeta& m = occ_meta[k % kOccRing];
                draws++;
                if (!v[1] || !(m.flags & OCC_BEGUN)) { unrec++; continue; }
                sum += v[0];
                if (m.flags & OCC_SKIPPED) skipped++;
                if (m.flags & OCC_MESH) mesh++;
                if (m.flags & OCC_INDIRECT) indirect++;
            }
        occ_last[r.addr] = sum;
        if (r.key) occ_keyed[r.key] = {sum, r.present};
        if (auto pa = occ_pending_at.find(r.addr); pa != occ_pending_at.end() && --pa->second <= 0) occ_pending_at.erase(pa);
        if (occ_mode() >= 4 && span_ok && occ_latest[r.addr] == r.seq) guest_cpu_write(r.addr, &sum, 8);   // (over the fixed value)
        occ_t.reports++;
        occ_t.draws += draws;
        if (!sum) {
            occ_t.zero++;
            if (!draws) occ_t.empty++;
            else if (unrec == draws) occ_t.unrecorded++;
            else if (skipped) occ_t.skipped++;
            else if (mesh) occ_t.mesh++;
            else if (indirect) occ_t.indirect++;
            else occ_t.plain++;
        }
        if (occ_log && r.logged) {
            fprintf(occ_log, "R %llu p%lld addr %#llx win %llu %llu draws %d unrec %d skipped %d mesh %d indirect %d count %llu\n",
                    (unsigned long long)r.seq, (long long)r.present, (unsigned long long)r.addr, (unsigned long long)r.lo,
                    (unsigned long long)r.hi, draws, unrec, skipped, mesh, indirect, (unsigned long long)sum);
            if (span_ok)
                for (uint64_t k = std::max(r.lo, logged_upto); k < r.hi; k++) {
                    const uint64_t* v = &res[(size_t)(k - lo) * 2];
                    const OccMeta& m = occ_meta[k % kOccRing];
                    const uint64_t* q = st.empty() ? nullptr : &st[(size_t)(k - lo) * sw];
                    const bool qa = q && q[sw - 1] && (m.flags & OCC_STATS);
                    fprintf(occ_log, "D %llu prog %016llx flags %#x avail %d samples %llu clip %lld fs %lld task %lld mesh %lld\n",
                            (unsigned long long)k, (unsigned long long)m.prog, (unsigned)m.flags, (int)v[1], (unsigned long long)v[0],
                            qa ? (long long)q[0] : -1, qa ? (long long)q[1] : -1, qa && occ_stats_mesh ? (long long)q[2] : -1,
                            qa && occ_stats_mesh ? (long long)q[3] : -1);
                }
            logged_upto = std::max(logged_upto, r.hi);
        }
    }
    occ_pending.erase(occ_pending.begin(), occ_pending.begin() + (ptrdiff_t)n);
    if (occ_log) fflush(occ_log);
}
void occ_poll(uint64_t done) {
    size_t n = 0;
    while (n < occ_pending.size() && occ_pending[n].value <= done) n++;
    occ_resolve(n);
}
}  // namespace
/// (the replay thread, at every draw and dispatch of the stream) the ring's next chunk reset from the host before its first use
void occ_advance(uint64_t serial, uint64_t value) {
    if (!occ_mode()) return;
    std::call_once(occ_once, occ_init);
    if (!occ_pool) return;
    const uint32_t chunk = (uint32_t)((serial % kOccRing) / kOccChunk);
    occ_chunk_value[chunk] = std::max(occ_chunk_value[chunk], value);
    if (serial % kOccChunk) return;
    // (entering a chunk: its last lap's draws done and its reports read, then reset)
    const uint64_t last = occ_chunk_value[chunk];
    if (last && last != value) vk_timeline_wait(last);
    size_t n = 0;
    while (n < occ_pending.size() && occ_pending[n].lo + kOccRing < serial + kOccChunk) n++;
    if (n) {
        if (occ_pending[n - 1].value > vk_timeline_completed()) vk_timeline_wait(occ_pending[n - 1].value);
        occ_resolve(n);
    }
    vkResetQueryPool(dev.device, occ_pool, chunk * kOccChunk, kOccChunk);
    if (occ_stats_pool) vkResetQueryPool(dev.device, occ_stats_pool, chunk * kOccChunk, kOccChunk);
    for (uint32_t i = 0; i < kOccChunk; i++) occ_meta[chunk * kOccChunk + i].flags = 0;
    occ_chunk_value[chunk] = value;
}
/// (the replay thread, a samples-passed report in stream order) the value to write: fixed, or from the last measured count
uint64_t occ_report(uint64_t addr, uint64_t lo, uint64_t hi, uint64_t value, uint64_t fixed, uint64_t prog) {
    const int mode = occ_mode();
    if (!mode || !occ_pool) return fixed;
    const bool logged = occ_log && lo >= occ_log_lo.load(std::memory_order_relaxed) && hi <= occ_log_hi.load(std::memory_order_relaxed);
    uint64_t key = 0;
    if (mode == 5) {
        const uint64_t pk = prog * 0x9E3779B97F4A7C15ull ^ (hi - lo) * 0xC2B2AE3D27D4EB4Full;
        const uint32_t ord = occ_key_ord[pk]++;
        key = (pk ^ ((uint64_t)ord * 0x165667B19E3779F9ull)) | 1;
    }
    occ_pending.push_back({addr, lo, hi, value, key, occ_presents, occ_seq, logged});
    occ_latest[addr] = occ_seq++;
    occ_pending_at[addr]++;
    if (mode == 4) return fixed;   // (the count written once read, before a condition reads it)
    if (mode == 1) return fixed;
    if (mode == 5) {
        const auto k = occ_keyed.find(key);
        if (k == occ_keyed.end() || occ_presents - k->second.present > 3) {
            occ_t.unknown_write++;
            return fixed;
        }
        if (!k->second.count) occ_t.zero_write++;
        return k->second.count;
    }
    const auto it = occ_last.find(addr);
    if (it == occ_last.end()) {
        occ_t.unknown_write++;
        return fixed;
    }
    if (!it->second) occ_t.zero_write++;
    if (mode == 2) return it->second ? fixed : 0;
    return it->second;
}
/// (mode 4) a render condition at addr reads a report not read yet
bool occ_cond_pending(uint64_t addr) {
    return occ_mode() == 4 && (occ_pending_at.count(addr) || occ_pending_at.count(addr + 16));
}
/// (mode 4, after the GPU finished everything recorded so far) every pending report read and written
void occ_resolve_all() {
    occ_resolve(occ_pending.size());
}
/// (mode 4: no submit was possible there) the conditions read what is written (the fixed value); the counts come later
void occ_give_up() {
    static int told;
    if (told++ < 8) logf("vk: occlusion: a condition on a report not read yet could not wait for it (the fixed value read)");
    occ_pending_at.clear();
}
/// (the replay thread, logged frames) a render condition as it was decided
void occ_cond(uint64_t addr, int mode, uint64_t x, uint64_t y, bool on, uint64_t serial) {
    if (!occ_log || serial < occ_log_lo.load(std::memory_order_relaxed) || serial >= occ_log_hi.load(std::memory_order_relaxed)) return;
    fprintf(occ_log, "C %llu p%lld at %llu addr %#llx mode %d x %llu y %llu %s\n", (unsigned long long)occ_seq++, (long long)occ_presents,
            (unsigned long long)serial, (unsigned long long)addr, mode, (unsigned long long)x, (unsigned long long)y, on ? "on" : "off");
}
/// (the replay thread, at each present) reports whose submissions are done read; the logged frames opened and closed
void occ_present(int64_t presents, uint64_t serial, uint64_t done) {
    if (!occ_mode() || !occ_pool) return;
    occ_presents = presents;
    occ_key_ord.clear();
    occ_poll(done);
    if (presents % 600 == 0 && occ_keyed.size() > 200000) occ_keyed.clear();
    if (const char* dir = occ_log_dir()) {
        static const int64_t every = getenv("RX_VK_OCC_LOG_EVERY") ? std::max(4, atoi(getenv("RX_VK_OCC_LOG_EVERY"))) : 600;
        static const int64_t frames = getenv("RX_VK_OCC_LOG_FRAMES") ? std::clamp(atoi(getenv("RX_VK_OCC_LOG_FRAMES")), 1, 16) : 2;
        if (presents % every == 0) {
            if (occ_log) fclose(occ_log);
            char path[600];
            snprintf(path, sizeof path, "%s/occ_%lld.txt", dir, (long long)presents);
            occ_log = fopen(path, "w");
            occ_log_hi.store(UINT64_MAX), occ_log_lo.store(serial);
        } else if (presents % every == frames && occ_log) {
            occ_log_hi.store(serial);
        }
    }
    if (presents % 600 == 0 && occ_t.reports) {
        const double f = 600.0;
        logf("vk: occlusion: %.0f reports a frame, %.0f read 0 (%.0f with no draws, %.0f none recorded, %.0f a draw skipped, %.0f with mesh "
             "draws, %.0f indirect, %.0f plain draws); %.1f draws a report; written: %.0f zero, %.0f not yet measured",
             occ_t.reports / f, occ_t.zero / f, occ_t.empty / f, occ_t.unrecorded / f, occ_t.skipped / f, occ_t.mesh / f, occ_t.indirect / f,
             occ_t.plain / f, occ_t.reports ? (double)occ_t.draws / occ_t.reports : 0.0, occ_t.zero_write / f, occ_t.unknown_write / f);
        occ_t = {};
    }
}

// ---- (EvilOOT) NVN2's indirect mesh draws in Vulkan's layout ----
// NVN2 keeps an indirect mesh draw as {task count, first task} (as its direct DrawMeshTasks(first, count) and NVIDIA's
// mesh shading always did); vkCmdDrawMeshTasksIndirectEXT reads {x, y, z} workgroups. Read as they were, "7 tasks from 0"
// was 7 x 0 x 1: no work, so every GPU-built mesh draw (OoT's foliage, grass, falling leaves, fireflies) drew nothing even
// once its counts were right. Before each, outside the pass, its records go to a ring as {count, 1, 1} (a fill of 1s,
// then each record's count word copied over x). RX_VK_MESH_IND_COUNT_WORD=1 takes the second word for the count
// instead. A first task other than 0 has no Vulkan
// counterpart (as for the direct draws: drawn from 0).
namespace {
constexpr uint64_t kMeshRing = 4ull << 20;
VkBuffer mesh_ring = VK_NULL_HANDLE;
VmaAllocation mesh_ring_mem = nullptr;
std::mutex mesh_ring_mu;
uint64_t mesh_ring_at = 0;
constexpr uint64_t kMeshSingleRing = 1ull << 20;
VkBuffer mesh_single_ring = VK_NULL_HANDLE;
VmaAllocation mesh_single_mem = nullptr;
uint64_t mesh_single_at = 0;
VkBuffer mesh_dbg = VK_NULL_HANDLE;   // (RX_VK_G3_MESHCOUNT)
VmaAllocation mesh_dbg_mem = nullptr;
void* mesh_dbg_ptr = nullptr;
thread_local VkBuffer mesh_conv_buf = VK_NULL_HANDLE;   // (the converted records for the draw in progress)
thread_local uint64_t mesh_conv_off = 0;
bool mesh_indirect_convert(VkCommandBuffer cmd, const DrawArgs& d) {
    mesh_conv_buf = VK_NULL_HANDLE;
    if (!d.indirect) return false;
    static const uint32_t word = getenv("RX_VK_MESH_IND_COUNT_WORD") ? (uint32_t)atoi(getenv("RX_VK_MESH_IND_COUNT_WORD")) & 3 : 0;
    // (amd-fix: the rewrite with the firsts in the draw does it, RX_VK_MESH_ARGS; this one drew every draw from its first 0)
    static const bool rewrite = getenv("RX_VK_MESH_ARGS") && atoi(getenv("RX_VK_MESH_ARGS")) != 0;   // (sharkiix version: off, as upstream)
    if (rewrite) return false;
    PROF_EXACT();
    // The compute fallback dispatches from the first record. Native mesh draws may consume the whole list.
    const uint32_t n = dev.mesh_shader ? (uint32_t)std::clamp(d.max, 1, 4096) : 1u;
    const uint32_t stride = d.stride > 0 ? (uint32_t)d.stride : 8;
    const BufferRef args = gpu_buffer_bind(d.indirect, (uint64_t)(n - 1) * stride + 4 * (word + 1), false);
    if (!args) return false;
    static const bool scoped_on = env_on("RX_VK_MESH_CONVERT_SCOPED");
    static const bool trace_counts = env_on("RX_VK_G3_MESHCOUNT");
    const bool scoped = !dev.mesh_shader && scoped_on && !trace_counts && policy_rule::windows_vega_compat(dev.facts);
    VkBuffer converted;
    uint64_t off;
    bool fixed_yz = false;
    {
        std::lock_guard lk(mesh_ring_mu);
        if (scoped) {
            // Each slot is permanently {x, 1, 1, padding}. Keeping a separate
            // ring prevents a variable-size native draw from overwriting y or z.
            if (!mesh_single_ring) {
                void* mapped = nullptr;
                if (create_buffer(kMeshSingleRing, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  true, mesh_single_ring, mesh_single_mem, &mapped)) {
                    std::fill_n(static_cast<uint32_t*>(mapped), kMeshSingleRing / sizeof(uint32_t), 1u);
                    flush_mapped_range(mesh_single_mem, 0, kMeshSingleRing);
                }
            }
            fixed_yz = mesh_single_ring != VK_NULL_HANDLE;
        }
        VkBuffer& ring = fixed_yz ? mesh_single_ring : mesh_ring;
        if (!ring) {
            void* unused = nullptr;
            if (!create_buffer(kMeshRing, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, ring,
                               mesh_ring_mem, &unused))
                return false;
        }
        uint64_t& at = fixed_yz ? mesh_single_at : mesh_ring_at;
        const uint64_t capacity = fixed_yz ? kMeshSingleRing : kMeshRing;
        const uint64_t bytes = ((uint64_t)n * 12 + 15) & ~15ull;
        if (at + bytes > capacity) at = 0;
        off = at;
        at += bytes;
        converted = ring;
    }
    end_pass(cmd);
    GprofSpan gp(cmd, "meshconv (an indirect mesh draw's arguments -> {count, 1, 1}, with its barriers)");   // (RX_VK_PASS_TIMING)
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    // Vega's fallback consumes one indirect record. Keep the dependency on
    // the source count and scratch slot without making other buffers visible.
    VkBufferMemoryBarrier2 buffers[2]{{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2},
                                     {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2}};
    if (scoped) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true, std::memory_order_relaxed))
            logf("vk: Vega indirect mesh conversion: scoped barriers, fixed y/z %s", fixed_yz ? "active" : "allocation unavailable");
        for (auto& barrier : buffers) {
            barrier.srcStageMask = mb.srcStageMask;
            barrier.dstStageMask = mb.dstStageMask;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        }
        buffers[0].srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        buffers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        buffers[0].buffer = args.buffer, buffers[0].offset = args.offset + 4 * word, buffers[0].size = 4;
        buffers[1].srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        buffers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        buffers[1].buffer = converted, buffers[1].offset = off, buffers[1].size = 12;
        di.memoryBarrierCount = 0;
        di.bufferMemoryBarrierCount = 2;
        di.pBufferMemoryBarriers = buffers;
    }
    vkCmdPipelineBarrier2(cmd, &di);
    VkBufferCopy first{args.offset + 4 * word, off, 4};
    const VkBufferCopy* regions = &first;
    std::vector<VkBufferCopy> list;
    if (n == 1) {
        // The fill writes y and z and the copy writes x. Both become visible at the indirect-read barrier below.
        if (!fixed_yz) vkCmdFillBuffer(cmd, converted, off + 4, 8, 1u);
    } else {
        vkCmdFillBuffer(cmd, converted, off, (uint64_t)n * 12, 1u);
        mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier2(cmd, &di);
        list.resize(n);
        for (uint32_t i = 0; i < n; i++)
            list[i] = {first.srcOffset + (uint64_t)i * stride, off + (uint64_t)i * 12, 4};
        regions = list.data();
    }
    vkCmdCopyBuffer(cmd, args.buffer, converted, n, regions);
    {   // (diagnosis, default off) RX_VK_G3_MESHCOUNT=1: the same counts into a host-visible twin of the ring, for mesh_count_note
        if (trace_counts) {
            if (!mesh_dbg && !create_buffer(kMeshRing, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, mesh_dbg, mesh_dbg_mem, &mesh_dbg_ptr)) mesh_dbg = VK_NULL_HANDLE;
            if (mesh_dbg) vkCmdCopyBuffer(cmd, args.buffer, mesh_dbg, n, regions);
        }
    }
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    if (scoped) {
        buffers[1].srcStageMask = mb.srcStageMask;
        buffers[1].srcAccessMask = mb.srcAccessMask;
        buffers[1].dstStageMask = mb.dstStageMask;
        buffers[1].dstAccessMask = mb.dstAccessMask;
        di.bufferMemoryBarrierCount = 1;
        di.pBufferMemoryBarriers = &buffers[1];
    }
    vkCmdPipelineBarrier2(cmd, &di);
    mesh_conv_buf = converted, mesh_conv_off = off;
    return true;
}
/// (RX_VK_G3_MESHCOUNT=1, diagnosis) an indirect mesh draw of program key put its record at ring offset off: 2000 draws on, the count the GPU
/// wrote there is read, and every 4000 of them the programs with the most workgroups are logged (draws, mean, largest, sum a frame)
void mesh_count_note(uint64_t key, uint64_t off) {
    static std::mutex mu;
    static std::deque<std::pair<uint64_t, uint64_t>> q;
    struct Agg { uint64_t n = 0, sum = 0, most = 0; };
    static std::unordered_map<uint64_t, Agg> agg;
    static uint64_t reads = 0;
    std::lock_guard lk(mu);
    q.emplace_back(key, off);
    if (q.size() <= 2000 || !mesh_dbg_ptr) return;
    const auto [k, o] = q.front();
    q.pop_front();
    const uint32_t v = *(volatile const uint32_t*)((const uint8_t*)mesh_dbg_ptr + o);
    Agg& a = agg[k];
    a.n++, a.sum += v, a.most = std::max<uint64_t>(a.most, v);
    if (++reads % 4000 != 0) return;
    std::vector<std::pair<uint64_t, uint64_t>> top;
    for (const auto& [kk, aa] : agg) top.emplace_back(aa.sum, kk);
    std::sort(top.rbegin(), top.rend());
    for (size_t i = 0; i < top.size() && i < 14; i++) {
        const Agg& aa = agg[top[i].second];
        logf("vk: mesh counts prog %016llx: %llu draws, %.1f workgroups mean, %llu most, %llu in all", (unsigned long long)top[i].second,
             (unsigned long long)aa.n, (double)aa.sum / aa.n, (unsigned long long)aa.most, (unsigned long long)aa.sum);
    }
    agg.clear();
}
} // namespace
// (hookshot) NVN's indirect mesh draws hold (count, first) and Vulkan's (x, y, z): given as they were, every one drew y = first = 0 workgroups,
// with no error. They are most of the game's vegetation: the forest's trees, ferns and grass (GPU culling writes their counts, so they can only
// be moved on the GPU): each count is copied into this buffer of (count, 1, 1) just before its draw. The GPU runs the copies and draws in
// order, so a slot used again is safe. A first other than 0 has no Vulkan counterpart.
constexpr uint32_t kMeshArgs = 1u << 19;   // (a ring: entries are reused 262144 draws on, frames after the GPU read them; ranges start in its first half)
VkBuffer mesh_args_buffer = VK_NULL_HANDLE;
VmaAllocation mesh_args_alloc = nullptr;
uint32_t* mesh_args_words = nullptr;
uint64_t mesh_args_address = 0;   // (its device address: a task program reads its draw's first workgroup at + 12, + 16 per draw)
// (hookshot) Indirect mesh draws' arguments, 16 bytes each as Vulkan takes them with the first after: {count, 1, 1, first}. NVN's are
// {count, 0, 0, first}: as they lay, every foliage draw had no workgroups in y and z and drew nothing. Entry 0 stays {0, 1, 1, 0}: the
// first of draws that have none
bool mesh_args_ready() {
    static const bool ok = [] {
        void* mapped = nullptr;
        if (!create_buffer((uint64_t)kMeshArgs * 16,
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true,
                           mesh_args_buffer, mesh_args_alloc, &mapped))
            return false;
        const auto discard = [] {
            destroy_buffer(mesh_args_buffer, mesh_args_alloc);
            mesh_args_buffer = VK_NULL_HANDLE;
            mesh_args_alloc = nullptr;
            mesh_args_words = nullptr;
            mesh_args_address = 0;
            return false;
        };
        if (!mapped) return discard();
        uint32_t* w = (uint32_t*)mapped;
        mesh_args_words = w;
        for (uint32_t i = 0; i < kMeshArgs; i++) w[i * 4] = 0, w[i * 4 + 1] = 1, w[i * 4 + 2] = 1, w[i * 4 + 3] = 0;
        VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        da.buffer = mesh_args_buffer;
        mesh_args_address = vkGetBufferDeviceAddress(dev.device, &da);
        if (!mesh_args_address) return discard();
        return true;
    }();
    return ok;
}
uint32_t mesh_args_take(uint32_t n) {   // (n entries of the ring, never entry 0; n at most kMeshArgs / 2)
    // (a fixed modulus: with one that changed with n, two draws close together could be given overlapping entries)
    static std::atomic<uint32_t> next{1};
    n = std::clamp<uint32_t>(n, 1, kMeshArgs / 2);
    const uint32_t at = next.fetch_add(n) % (kMeshArgs / 2);
    return at + 1;
}

struct KeptVertex {
    uint64_t addr = 0, size = 0, stamp = 0;
    BufferRef ref{};
    int ok = 0;
};
thread_local KeptVertex kept_vb[16];

// ---- (hookshot, issue 640) a GPU without mesh shaders: the task and mesh stages as compute dispatches and a
// generated vertex stage, drawn through an ordinary graphics pipeline. Records and payloads live in two buffers a
// recording thread reuses draw after draw (one pair, not one a program: see below); the commands are ordered, so each
// dispatch runs after the last draw that read them.
namespace {
// (one pair for every fallback program a recording thread draws, not a pair a program: 95 programs' pairs were ~2.2 GB of
// VRAM on an RX 5500M, and when VRAM ran out a program's pair could not be made and its draws stopped for good. Each draw
// clears the slots it runs and a barrier orders it after the draw before, so every program can use the same records.)
struct FallbackBuffers {
    VkBuffer records = VK_NULL_HANDLE, payload = VK_NULL_HANDLE;
    VmaAllocation records_mem = nullptr, payload_mem = nullptr;
    uint64_t records_address = 0, payload_address = 0;
    uint64_t records_bytes = 0, payload_bytes = 0;
    bool warp_counters_cleared = false;   // (mma_layout.h mesh_fallback_shared_warp: its counters past the records, at zero once)
};
// (grown pairs stay: a command buffer recorded before the growth may still read them)
TLS_OBJECT(fallback_buffers, std::vector<FallbackBuffers>)
thread_local VkBuffer fallback_draw_records = VK_NULL_HANDLE;   // (the last fallback_record's: its header is the draw's command)

uint32_t fallback_task_groups(const Program& p) {
    return Shader::Backend::SPIRV::MESH_FALLBACK_CAPACITY / std::max(1u, p.fallback_task_threads);
}

uint64_t fallback_buffer_address(VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(dev.device, &info);
}

bool geometry_capture_on() {
    static const bool on = env_on("RX_VK_GEOMETRY_CAPTURE");
    return on;
}
bool geometry_reuse_on() {
    static const bool on = env_on("RX_VK_GEOMETRY_REUSE") && policy_rule::windows_vega_compat(dev.facts);
    return on;
}

uint32_t geometry_dependencies(const Program& p, const DrawArgs& d) {
    using namespace mesh_capture;
    uint32_t flags = (p.fallback_task ? task : 0) | (d.indirect ? indirect : 0);
    for (const Stage& st : p.stages) {
        if (st.vk_stage != VK_SHADER_STAGE_COMPUTE_BIT) continue;
        const auto& i = st.info;
        if (!i.storage_buffers_descriptors.empty()) flags |= storage;
        if (!i.image_descriptors.empty() || !i.image_buffer_descriptors.empty() ||
            !i.texture_descriptors.empty() || !i.texture_buffer_descriptors.empty()) flags |= image;
        if (i.uses_global_memory || i.num_global_tables) flags |= global;
        if (i.stores_global_memory || i.uses_atomic_image_u32 ||
            std::any_of(i.storage_buffers_descriptors.begin(), i.storage_buffers_descriptors.end(),
                        [](const auto& buffer) { return buffer.is_written; })) flags |= mesh_capture::write;
    }
    return flags;
}

struct GeometryStorage {
    struct Range {
        VkDescriptorBufferInfo* binding = nullptr;
        VkDescriptorBufferInfo original{};
        uint32_t offset = 0;
    };
    static constexpr uint32_t capacity = 64u << 10;
    static constexpr uint32_t input_args = capacity, output_args = capacity + 16, snapshot_bytes = capacity + 32;
    std::array<Range, 4> ranges{};
    uint32_t count = 0, used = 0;
};

struct GeometryStorageGpu {
    struct Snapshot { VkBuffer buffer = VK_NULL_HANDLE; VmaAllocation memory = nullptr; uint64_t address = 0; };
    Snapshot snapshots[2];
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    int next = 0;
    bool failed = false;
};
TLS_OBJECT(geometry_storage_gpu, GeometryStorageGpu)

bool geometry_storage_ready(GeometryStorageGpu& gpu) {
    if (gpu.failed) return false;
    if (gpu.pipeline) return true;
    const auto fail = [&] {
        for (auto& snapshot : gpu.snapshots)
            if (snapshot.buffer) destroy_buffer(snapshot.buffer, snapshot.memory);
        if (gpu.layout) vkDestroyPipelineLayout(dev.device, gpu.layout, nullptr);
        gpu.failed = true;
        return false;
    };
    constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    for (auto& snapshot : gpu.snapshots) {
        void* mapped = nullptr;
        if (!create_buffer(GeometryStorage::snapshot_bytes, usage, false, snapshot.buffer, snapshot.memory, &mapped)) return fail();
        snapshot.address = fallback_buffer_address(snapshot.buffer);
        if (!snapshot.address) return fail();
    }
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 68};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(dev.device, &layout, nullptr, &gpu.layout) != VK_SUCCESS) return fail();
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    const auto code = BUILTIN_SPV(MESH_STORAGE_COMPARE_COMP_SPV);
    info.codeSize = code.size;
    info.pCode = code.code;
    VkShaderModule module = VK_NULL_HANDLE;
    {
        IntelDriverGate gate;
        if (guarded_shader_module(&info, &module) != VK_SUCCESS) return fail();
    }
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.layout = gpu.layout;
    pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                      module, "main", nullptr};
    const VkResult result = cached_compute(&pipeline, &gpu.pipeline);
    vkDestroyShaderModule(dev.device, module, nullptr);
    return result == VK_SUCCESS || fail();
}

// Snapshot the Vulkan resources (including earlier GPU writes). The mesh stage uses this same copy if it runs.
void geometry_storage_record(VkCommandBuffer cmd, GeometryStorageGpu& gpu, const GeometryStorage& storage,
                             const FallbackBuffers& f, int current, int previous, bool indirect) {
    const auto& now = gpu.snapshots[current];
    for (uint32_t i = 0; i < storage.count; ++i) {
        const auto& range = storage.ranges[i];
        const VkBufferCopy copy{range.original.offset, range.offset, range.original.range};
        vkCmdCopyBuffer(cmd, range.original.buffer, now.buffer, 1, &copy);
    }
    if (indirect) {
        const VkBufferCopy copy{mesh_conv_off, GeometryStorage::input_args, sizeof(VkDispatchIndirectCommand)};
        vkCmdCopyBuffer(cmd, mesh_conv_buf, now.buffer, 1, &copy);
    }
    if (previous < 0) return;
    full_barrier(cmd);
    struct Push {
        uint64_t previous, current, records, args;
        uint32_t offsets[4]{}, lengths[4]{};
        uint32_t indirect = 0;
    } push{gpu.snapshots[previous].address, now.address, f.records_address, now.address + GeometryStorage::output_args};
    constexpr uint32_t push_bytes = offsetof(Push, indirect) + sizeof(uint32_t);
    static_assert(push_bytes == 68);
    push.indirect = indirect;
    for (uint32_t i = 0; i < storage.count; ++i) {
        push.offsets[i] = storage.ranges[i].offset / 4;
        push.lengths[i] = static_cast<uint32_t>(storage.ranges[i].original.range / 4);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gpu.pipeline);
    vkCmdPushConstants(cmd, gpu.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, &push);
    vkCmdDispatch(cmd, 1, 1, 1);
    // The fallback's pre-dispatch barrier also makes the comparison's writes visible.
}

bool geometry_inputs(const Program& p, const DrawArgs& d, uint32_t dependencies, mesh_capture::Inputs& inputs, GeometryStorage& storage) {
    // The fallback's workgroups allocate output records atomically. With exactly one workgroup,
    // its result does not depend on allocation order. Uniforms come from immutable CPU uploads.
    const bool indirect = (dependencies & mesh_capture::indirect) != 0;
    if ((dependencies & ~(mesh_capture::storage | mesh_capture::indirect)) || d.first_task ||
        (indirect ? !mesh_conv_buf : (d.groups[0] != 1 || d.groups[1] != 1 || d.groups[2] != 1)) ||
        p.uses_render_area || Shader::Backend::SPIRV::MeshFallbackCounters() ||
        !cur || !cur->mapped || par_active.load(std::memory_order_relaxed)) return false;
    if (p.stages.empty() || p.stages.front().vk_stage != VK_SHADER_STAGE_COMPUTE_BIT ||
        !p.stages.front().lane_free || p.stages.front().nvn_stage != 7) return false;
    const auto& bindings = p.stages.front().info.constant_buffer_descriptors;
    const auto& writes = dw().writes;
    if (bindings.size() > writes.size()) return false;
    for (size_t i = 0; i < bindings.size(); ++i) {
        const auto& w = writes[i];
        if (w.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || !w.pBufferInfo ||
            w.descriptorCount != bindings[i].count) return false;
        for (uint32_t k = 0; k < w.descriptorCount; ++k) {
            const auto& b = w.pBufferInfo[k];
            if (!b.buffer || b.buffer != cur->stream || !b.range || b.offset > cur->used || b.range > cur->used - b.offset ||
                !inputs.append({cur->mapped + b.offset, static_cast<size_t>(b.range)})) return false;
        }
    }
    const auto& storage_descriptors = p.stages.front().info.storage_buffers_descriptors;
    if (bindings.size() + storage_descriptors.size() > writes.size()) return false;
    for (size_t i = 0; i < storage_descriptors.size(); ++i) {
        const auto& desc = storage_descriptors[i];
        const auto& w = writes[bindings.size() + i];
        if (desc.is_written || (desc.cbuf_index == 0 && desc.cbuf_offset == 0xff0) ||
            w.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || !w.pBufferInfo ||
            w.descriptorCount != desc.count || w.descriptorCount > storage.ranges.size() - storage.count) return false;
        for (uint32_t k = 0; k < w.descriptorCount; ++k) {
            const auto& b = w.pBufferInfo[k];
            const uint64_t alignment = storage_binding_align();
            const uint64_t offset = (storage.used + alignment - 1) & ~(alignment - 1);
            if (!b.buffer || b.buffer == zero_buffer || !b.range || (b.offset & 3) || (b.range & 3) ||
                offset >= GeometryStorage::capacity || b.range > GeometryStorage::capacity - offset) return false;
            // w points into the current descriptor writer's mutable buffer-info array.
            storage.ranges[storage.count++] = {const_cast<VkDescriptorBufferInfo*>(&b), b, static_cast<uint32_t>(offset)};
            storage.used = static_cast<uint32_t>(offset + b.range);
            if (!inputs.append({reinterpret_cast<const uint8_t*>(&b.range), sizeof(b.range)})) return false;
        }
    }
    return true;
}

struct RecentGeometry {
    const Program* program = nullptr;
    VkBuffer records = VK_NULL_HANDLE;
    uint64_t recording = 0;
    int snapshot = -1;
    bool indirect = false;
    mesh_capture::Inputs inputs;
};
TLS_OBJECT(recent_geometry, RecentGeometry)
TLS_OBJECT(geometry_candidate, mesh_capture::Inputs)

struct GeometryCapture {
    std::mutex mutex;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    uint32_t* mapped = nullptr;
    std::FILE* file = nullptr;
    uint64_t value = 0, frame = 0, polled_frame = ~0ull;
    uint64_t key = 0, variant = 0;
    uint32_t flags = 0, words = 0, fit = 0, source_vertices = 0, written = 0, draws_this_frame = 0;
    bool failed = false;
};
GeometryCapture geometry_capture;

// Read only finished GPU work. This one-slot sample never waits on pending work or reads live guest memory.
void geometry_capture_drain(GeometryCapture& capture) {
    namespace spirv = Shader::Backend::SPIRV;
    if (!capture.value || !vk_timeline_poll(capture.value)) return;
    const uint64_t value = std::exchange(capture.value, 0);
    invalidate_mapped(capture.allocation, mesh_capture::staging_words * sizeof(uint32_t));
    const uint32_t* header = capture.mapped;
    const bool task = (capture.flags & mesh_capture::task) != 0;
    if (header[0] > capture.fit || (task && header[5] > spirv::MESH_FALLBACK_CAPACITY)) return;
    const uint32_t vertices = mesh_capture::stored_vertices(header[0], capture.fit);
    const uint32_t slots = task ? mesh_capture::stored_slots(header[5], spirv::MESH_FALLBACK_CAPACITY) : 0;
    if (!capture.file) {
        const auto path = log_path("mesh_geometry.rtm");
        capture.file = fopen(path.c_str(), "wb");
        if (!capture.file || fwrite("RTMESH01", 1, 8, capture.file) != 8) {
            capture.failed = true;
            return;
        }
        logf("vk: capturing mesh records and task visibility in %s", path.c_str());
    }
    const uint64_t identity[]{capture.frame, capture.key, capture.variant, value};
    const uint32_t counts[]{capture.flags, capture.words, vertices, slots};
    const auto put = [&](const void* data, size_t size, size_t count) {
        if (!capture.failed && fwrite(data, size, count, capture.file) != count) capture.failed = true;
    };
    put(identity, sizeof(uint64_t), std::size(identity));
    put(counts, sizeof(uint32_t), std::size(counts));
    put(header, sizeof(uint32_t), mesh_capture::header_words);
    const uint32_t* records = header + mesh_capture::header_words;
    if (capture.flags & mesh_capture::soa) {
        for (uint32_t word = 0; word < capture.words; ++word)
            put(records + word * capture.source_vertices, sizeof(uint32_t), vertices);
    } else {
        put(records, sizeof(uint32_t), size_t(vertices) * capture.words);
    }
    put(header + mesh_capture::header_words + mesh_capture::max_words * mesh_capture::sample_vertices,
        sizeof(uint32_t), slots);
    if (fflush(capture.file)) capture.failed = true;
    if (!capture.failed) ++capture.written;
}

void geometry_capture_draw(VkCommandBuffer cmd, const Program& p, const FallbackBuffers& f, uint32_t flags) {
    namespace spirv = Shader::Backend::SPIRV;
    if (!rec_value || par_active.load(std::memory_order_relaxed)) return;
    std::lock_guard lk(geometry_capture.mutex);
    auto& capture = geometry_capture;
    const uint64_t frame = texture_frame_no();
    if (capture.failed || capture.written >= 48) return;
    if (capture.polled_frame != frame) {
        capture.polled_frame = frame;
        capture.draws_this_frame = 0;
        geometry_capture_drain(capture);
    }
    // Sample later draws too. Every fourth sample keeps the first draw for comparisons over time.
    const uint32_t selected = capture.written % 4 ? (capture.written * 7) % 32 : 0;
    if (capture.value || capture.failed || capture.written >= 48 || (capture.written && frame < capture.frame + 8) ||
        !p.fallback_record_words || p.fallback_record_words > mesh_capture::max_words || capture.draws_this_frame++ != selected)
        return;
    if (!capture.buffer && !create_buffer_cached(mesh_capture::staging_words * sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, capture.buffer, capture.allocation, (void**)&capture.mapped)) return;
    if (!capture.mapped) return;
    const bool soa = mesh_fallback_layout() == static_cast<uint32_t>(spirv::MeshFallbackLayout::Soa);
    const uint32_t words = p.fallback_record_words;
    const uint32_t fit = spirv::MeshFallbackFit(words);
    const uint32_t vertices = std::min(fit, mesh_capture::sample_vertices);
    const VkDeviceSize head = mesh_capture::header_words * sizeof(uint32_t);
    std::array<VkBufferCopy, 1 + mesh_capture::max_words> copies{};
    copies[0] = {0, 0, head};
    if (soa) {
        for (uint32_t word = 0; word < words; ++word)
            copies[word + 1] = {head + VkDeviceSize(word) * fit * 4, head + VkDeviceSize(word) * vertices * 4, VkDeviceSize(vertices) * 4};
    } else {
        copies[1] = {head, head, VkDeviceSize(vertices) * words * 4};
    }
    vkCmdCopyBuffer(cmd, f.records, capture.buffer, soa ? 1 + words : 2, copies.data());
    if (p.fallback_task && f.payload) {
        const VkBufferCopy slots{0, head + VkDeviceSize(mesh_capture::max_words) * mesh_capture::sample_vertices * 4,
                                 mesh_capture::sample_slots * sizeof(uint32_t)};
        vkCmdCopyBuffer(cmd, f.payload, capture.buffer, 1, &slots);
    }
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
    capture.value = rec_value;
    capture.frame = frame;
    capture.key = p.key;
    capture.variant = p.variant;
    capture.flags = flags | (soa ? mesh_capture::soa : 0);
    capture.words = words;
    capture.fit = fit;
    capture.source_vertices = vertices;
}

bool fallback_buffers_of(const Program& p, FallbackBuffers*& out) {
    if (!p.fallback_vertices_per_workgroup || !p.fallback_record_words) return false;
    const uint64_t capacity = Shader::Backend::SPIRV::MESH_FALLBACK_CAPACITY;
    // (the header, the draw's command, the allocator and the mesh dispatch, and then a fixed area every program takes records
    // from, a primitive's corners at a time: only the primitives the mesh workgroups output, not a slot's worst case each.
    // Slots a primitive could fill were ~650 MB for the floor grass's ~13900 slots; it outputs ~17 MB)
    // (and with mma_layout.h mesh_fallback_global_attributes, the attribute regions' pool past them: ~32 MB)
    // (and with mma_layout.h mesh_fallback_shared_warp or mesh_fallback_global_attributes, their counters past those:
    // emit_spirv.h WARP_COUNTER_BYTES, where the stages find them at MeshFallbackRecordsBytes())
    const uint64_t warp_counters = Shader::Backend::SPIRV::MeshFallbackCounters() ? Shader::Backend::SPIRV::WARP_COUNTER_BYTES : 0;
    const uint64_t records = (Shader::Backend::SPIRV::mesh_fallback_global_attributes
                                  ? Shader::Backend::SPIRV::MESH_FALLBACK_POOL_OFFSET + Shader::Backend::SPIRV::MESH_FALLBACK_POOL_BYTES
                                  : (uint64_t)(Shader::Backend::SPIRV::MESH_FALLBACK_HEADER_WORDS +
                                               Shader::Backend::SPIRV::MESH_FALLBACK_RECORD_WORDS) * 4) + warp_counters;
    // (the most a recording thread's records buffer takes: its size follows the switches only, and emit_spirv.h's static_assert
    // keeps every layout under it)
    if (records > Shader::Backend::SPIRV::MESH_FALLBACK_RECORDS_LIMIT) return false;
    // (the map, a slot a started mesh workgroup, and a payload a task workgroup, its threads' meshlets sharing it)
    const uint64_t threads = std::max(1u, p.fallback_task_threads);
    const uint64_t payload = p.fallback_task ? capacity * 4 + (capacity + threads - 1) / threads * (uint64_t)p.fallback_payload_words * 4 : 0;
    auto& sets = fallback_buffers();
    for (auto it = sets.rbegin(); it != sets.rend(); ++it) {
        if (it->records && it->records_bytes >= records && (!payload || (it->payload && it->payload_bytes >= payload))) {
            out = &*it;
            return true;
        }
    }
    FallbackBuffers f;
    void* mapped = nullptr;
    // (TRANSFER_SRC: the shared warp's and the pool's counters are copied out of the records, fallback_warp_counters)
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                     ((warp_counters || geometry_capture_on()) ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0);
    // (only the payload outgrown: the newest pair's records, which every program takes from, go with the bigger payload. Made
    // again, each growth left another records area on the thread for the session, that no draw used again)
    const bool shared = !sets.empty() && sets.back().records && sets.back().records_bytes >= records;
    const auto drop_records = [&] {
        if (!shared) destroy_buffer(f.records, f.records_mem);
    };
    if (shared) {
        const FallbackBuffers& last = sets.back();
        f.records = last.records, f.records_mem = last.records_mem;
        f.records_address = last.records_address, f.records_bytes = last.records_bytes;
        f.warp_counters_cleared = last.warp_counters_cleared;   // (the shared warp's counters are in those records: not zeroed again)
    } else {
        // (the records' header is the vertex stage's indirect draw and the mesh stage's indirect dispatch)
        if (!create_buffer(records, usage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false, f.records, f.records_mem, &mapped)) return false;
        f.records_address = fallback_buffer_address(f.records);
        f.records_bytes = records;
        if (!f.records_address) {
            drop_records();
            return false;
        }
    }
    if (payload) {
        if (!create_buffer(payload, usage, false, f.payload, f.payload_mem, &mapped)) {
            drop_records();
            return false;
        }
        f.payload_address = fallback_buffer_address(f.payload);
        f.payload_bytes = payload;
        if (!f.payload_address) {
            destroy_buffer(f.payload, f.payload_mem);
            drop_records();
            return false;
        }
    }
    sets.push_back(f);
    out = &sets.back();
    return true;
}

bool fallback_compute_pipelines(Program& p) {
    static std::mutex mu;
    // (the programs whose compute pipeline the driver refused or faulted on, by key and variant: not tried again. Each try
    // was one more driver fault, on every draw of the program: 12,160 faults for one program in a Kokiri Forest run on
    // RADV, the frame down to 2-3 fps (#1001); on a GTX 1070 the replay stopped presenting after the first one. Left out,
    // the program costs its draws, as the first failure already did)
    static std::set<std::pair<uint64_t, uint64_t>> failed;
    {
        // (the lock covers the set and the program's pipelines, not the compiles: on the workers and in the precompile several
        // programs compile at once, which one lock held through AMD's seconds-long compiles made one at a time. One program is
        // built by one caller: fallback_pipelines_ready and precompile_run take its compute_state first)
        std::lock_guard lk(mu);
        if (!p.fallback_compute.empty()) return true;
        if (failed.count({p.key, p.variant})) return false;
    }
    // (hookshot) none of its stages given to a compiler that would never finish one: RADV on GCN 1-4 grew past 25 GB on the two
    // biggest mesh programs and kept the memory on its thread (fallback_size_policy.h). Left out, as a refused one is below.
    // (hookshot) Mesa's Intel compiler as well, with its own limit: alone it took the 80,000- and the 29,000-phi ones to 14 GB in
    // minutes, unfinished. RX_VK_FALLBACK_PHI_LIMIT=<n>: another limit, 0 none
    const bool mesa_intel = dev.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA;
    static const uint32_t phi_limit = [mesa_intel] {
        const uint32_t standard = mesa_intel ? fallback_size::kIntelPhiLimit : fallback_size::kRadvGcnPhiLimit;
        uint32_t limit = standard;
        const char* e = getenv("RX_VK_FALLBACK_PHI_LIMIT");
        if (e && !fallback_size::parse_limit(e, limit)) {
            limit = standard;
            logf("vk: RX_VK_FALLBACK_PHI_LIMIT=%s is not a number from 0 to 4294967295: ignored", e);
        }
        if (dev.radv_gcn || mesa_intel || e)
            logf("vk: mesh fallback compute stages with more than %u phis are not built on %s%s", limit,
                 mesa_intel ? "Mesa's Intel driver" : "RADV GCN 1-4 cards", limit ? "" : " (0: none left out)");
        return limit;
    }();
    for (size_t i = 0; i < p.stages.size(); i++) {
        const Stage& st = p.stages[i];
        if (st.vk_stage != VK_SHADER_STAGE_COMPUTE_BIT || !fallback_size::too_big(dev.radv_gcn || mesa_intel, st.phis, phi_limit))
            continue;
        {
            std::lock_guard lk(mu);
            failed.insert({p.key, p.variant});
        }
        static std::atomic<int> told{0};
        if (told++ < 8)
            logf("vk: fallback: the compute pipeline of %016llx (stage %zu of %zu, %u phis, the most %u) is not built: %s would not "
                 "finish it (RX_VK_FALLBACK_PHI_LIMIT); its draws are left out",
                 (unsigned long long)p.key, i, p.stages.size(), st.phis, phi_limit,
                 mesa_intel ? "Mesa's Intel compiler" : "RADV on this GCN card");
        return false;
    }
    // (hookshot) nor to Mesa's Intel compiler a stage of more calls than it finishes: one of the four took it to a 14 GB cap in 27
    // minutes unfinished (fallback_size_policy.h). RX_VK_FALLBACK_CALL_LIMIT=<n>: another limit, 0 none
    static const uint32_t call_limit = [] {
        uint32_t limit = fallback_size::kIntelCallLimit;
        const char* e = getenv("RX_VK_FALLBACK_CALL_LIMIT");
        if (e && !fallback_size::parse_limit(e, limit)) {
            limit = fallback_size::kIntelCallLimit;
            logf("vk: RX_VK_FALLBACK_CALL_LIMIT=%s is not a number from 0 to 4294967295: ignored", e);
        }
        if (dev.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA || e)
            logf("vk: mesh fallback compute stages with more than %u calls are not built on Mesa's Intel driver%s", limit,
                 limit ? "" : " (0: none left out)");
        return limit;
    }();
    for (size_t i = 0; i < p.stages.size(); i++) {
        const Stage& st = p.stages[i];
        if (st.vk_stage != VK_SHADER_STAGE_COMPUTE_BIT ||
            !fallback_size::too_many_calls(dev.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, st.calls, call_limit))
            continue;
        {
            std::lock_guard lk(mu);
            failed.insert({p.key, p.variant});
        }
        static std::atomic<int> told{0};
        if (told++ < 8)
            logf("vk: fallback: the compute pipeline of %016llx (stage %zu of %zu, %u calls, the most %u) is not built: Mesa's Intel "
                 "compiler would not finish it (RX_VK_FALLBACK_CALL_LIMIT); its draws are left out",
                 (unsigned long long)p.key, i, p.stages.size(), st.calls, call_limit);
        return false;
    }
    std::vector<VkPipeline> made(p.stages.size(), VK_NULL_HANDLE);
    bool any = false;
    for (size_t i = 0; i < p.stages.size(); i++) {
        const Stage& st = p.stages[i];
        if (st.vk_stage != VK_SHADER_STAGE_COMPUTE_BIT) continue;
        any = true;
        VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        si.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        si.module = st.module;
        si.pName = "main";
        pin_subgroup(si);
        // (hookshot, fallback_lanes.h) a stage whose warp is in workgroup memory (mma_layout.h mesh_fallback_shared_warp) has
        // no subgroup instruction: it runs at 8 lanes on Mesa's Intel driver, where 32 lanes leave a lane too few registers.
        // RX_VK_MESH_LANES=8, 16 or 32 (32: as every other stage)
        static const uint32_t lanes_wanted = [mesa_intel] {
            const char* e = getenv("RX_VK_MESH_LANES");
            const uint32_t asked = fallback_lanes::parse_lanes(e);
            if (e && !asked) logf("vk: RX_VK_MESH_LANES=%s is not 8, 16 or 32: ignored", e);
            const uint32_t lanes = fallback_lanes::wanted(mesa_intel, asked, dev.subgroup_min, dev.subgroup_max);
            if (lanes != 32) logf("vk: mesh fallback stages with no subgroup instruction run at %u lanes (RX_VK_MESH_LANES=32: at 32)", lanes);
            return lanes;
        }();
        static const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo lanes_size = [] {
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo r{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
            r.requiredSubgroupSize = lanes_wanted;
            return r;
        }();
        if (si.pNext && fallback_lanes::lanes_of(st.lane_free, lanes_wanted) != 32) si.pNext = &lanes_size;
        compute_describe(p, st);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = si;
        ci.layout = p.layout;
        if (dev.task_mesh_fallback || env_on("RX_VK_NOOPT"))
            ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        if (const VkResult r = cached_compute(&ci, &made[i]); r != VK_SUCCESS) {
            {
                std::lock_guard lk(mu);
                failed.insert({p.key, p.variant});   // (#1317: not compiled again at every draw)
            }
            // (what failed and against which limits: a GTX 1080 Ti's black screen after the intro on v2.0.11 had only "could not be
            // made", the result not said. -77: the driver faulted. The workgroup and shared memory are the mesh stage's, against the GPU's)
            static std::atomic<int> told{0};
            if (told++ < 8)
                logf("vk: fallback: the compute pipeline of %016llx (stage %zu of %zu, workgroup %ux%ux%u, shared memory %u bytes, the GPU's "
                     "most %u bytes and %u invocations) could not be made: %d; its draws are left out",
                     (unsigned long long)p.key, i, p.stages.size(), p.workgroup[0], p.workgroup[1], p.workgroup[2], p.shared_memory,
                     dev.props.limits.maxComputeSharedMemorySize, dev.props.limits.maxComputeWorkGroupInvocations, (int)r);
            for (VkPipeline m : made)
                if (m) vkDestroyPipeline(dev.device, m, nullptr);
            return false;
        }
        pcache_built_since_save++;
    }
    if (!any) return false;
    std::lock_guard lk(mu);
    p.fallback_compute = std::move(made);
    return true;
}

/// (hookshot) A fallback program's compute pipelines, compiled on a worker as a compute program's are (see dispatch()): made in
/// the draw, AMD's compiler took up to 12 s on one, every recording thread stopped behind it, and the game froze in the first
/// room with foliage (RX 5500M). The draw that starts the build waits up to 50 ms for it, which covers NVIDIA's compiles; a draw
/// while it still compiles is left out, as a dispatch is. compute_state, which only compute programs use otherwise: 0 before, 1
/// while it compiles, 2 after (made, or refused and left out for good: fallback_compute_pipelines)
bool fallback_pipelines_ready(Program& p) {
    if (p.compute_state.load(std::memory_order_acquire) == 2) return !p.fallback_compute.empty();
    int none = 0;
    const bool fresh = p.compute_state.compare_exchange_strong(none, 1);
    if (fresh) {
        { ParLock lk; pcache_init(); }
        Program* job = &p;
        auto build = [job] {
            fallback_compute_pipelines(*job);
            job->compute_state.store(2, std::memory_order_release);
        };
        // (one at a time on Intel, as compute programs are: serial_builds_on)
        static const bool serial = serial_builds_on();
        auto failed = [job]() noexcept {
            // fallback_compute is published only after the whole build succeeds; empty means failed.
            job->compute_state.store(2, std::memory_order_release);
        };
        if (serial) jobs_push_serial(std::move(build), failed);
        else jobs_push(std::move(build), true, nullptr, failed);
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (fresh && p.compute_state.load(std::memory_order_acquire) != 2 &&
           std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(50))
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    if (p.compute_state.load(std::memory_order_acquire) != 2) {
        static std::atomic<int> told{0};
        if (told++ < 40) logf_at(log_debug, "vk: fallback draw of %016llx left out: its compute pipelines are still compiling", (unsigned long long)p.key);
        st_pending_skips++;
        return false;
    }
    return !p.fallback_compute.empty();
}

/// (hookshot, mma_layout.h mesh_fallback_shared_warp) The shared warp's counters, the words past the records (emit_spirv.h
/// WARP_COUNTER_*): the waits for other lanes that gave up, after which a ballot or a shuffle may have had a wrong answer, and
/// with RX_VK_MESH_SHARED_WARP_CHECK the answers compared with the subgroup's and those that differed, and the attribute pool's
/// waits for a slot that gave up (mma_layout.h mesh_fallback_global_attributes). Copied out after one
/// fallback draw of a frame in every 600 and logged 30 frames later: each time with the check, otherwise only when more waits
/// gave up since the last line. Called after the barrier that follows the draw's stages.
void fallback_warp_counters(VkCommandBuffer cmd, const FallbackBuffers& f) {
    namespace spirv = Shader::Backend::SPIRV;
    if (!spirv::MeshFallbackCounters() || !f.warp_counters_cleared) return;
    thread_local VkBuffer copy = VK_NULL_HANDLE;
    thread_local VmaAllocation copy_mem = nullptr;
    thread_local uint32_t* words = nullptr;
    thread_local uint64_t copied = 0;   // (the frame of the last copy)
    thread_local bool pending = false;
    thread_local uint32_t waits_told = 0;
    thread_local uint32_t pool_waits_told = 0;
    const uint64_t frame = texture_frame_no();
    if (pending && frame > copied + 30) {
        pending = false;
        invalidate_mapped(copy_mem, spirv::WARP_COUNTER_BYTES);   // (host memory that is cached and not coherent: the GPU's copy)
        const uint32_t waits = words[spirv::WARP_COUNTER_WAITS / 4];
        if (spirv::mesh_fallback_shared_warp_check)
            logf("vk: the shared warp's check: %u answers compared with the subgroup's so far, %u different; %u waits gave up",
                 words[spirv::WARP_COUNTER_COMPARED / 4], words[spirv::WARP_COUNTER_DIFFERENT / 4], waits);
        else if (waits != waits_told)
            logf("vk: the mesh fallback's shared warp: %u waits for other lanes gave up so far, and the ballots and shuffles after "
                 "them may be wrong (RX_VK_MESH_SHARED_WARP=0: the subgroup's warp)", waits);
        waits_told = waits;
        const uint32_t pool_waits = words[spirv::WARP_COUNTER_POOL_WAITS / 4];
        if (pool_waits != pool_waits_told)
            logf("vk: the mesh fallback's attribute pool: %u waits for a slot gave up so far, and those workgroups shared a "
                 "region with another, whose geometry may be wrong (RX_VK_MESH_GLOBAL_ATTRS=0: the regions in workgroup memory)", pool_waits);
        pool_waits_told = pool_waits;
    }
    if (pending || frame % 600 != 300 || frame == copied) return;
    if (!copy && !create_buffer_cached(spirv::WARP_COUNTER_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, copy, copy_mem, (void**)&words)) return;
    const VkBufferCopy region{spirv::MeshFallbackRecordsBytes(), 0, spirv::WARP_COUNTER_BYTES};
    vkCmdCopyBuffer(cmd, f.records, copy, 1, &region);
    copied = frame, pending = true;
}

/// The dispatches of a mesh draw, outside any render pass, and the vertex count its graphics draw then covers.
bool fallback_record(VkCommandBuffer cmd, Program& p, const DrawArgs& d) {
    if (d.count_addr) {   // (more than one draw in the records: each its own dispatch, not one)
        static std::atomic<int> told{0};
        if (told++ < 2) logf("vk: fallback: a mesh draw with a count buffer is left out (issue 640)");
        return false;
    }
    if (dw().prog != &p) {   // (the descriptors built are another program's: draw() rebuilds them for a fallback draw)
        static std::atomic<int> told{0};
        if (told++ < 2) logf("vk: fallback: the descriptors built are not %016llx's: its draw is left out", (unsigned long long)p.key);
        return false;
    }
    FallbackBuffers* f = nullptr;
    if (!fallback_buffers_of(p, f) || !fallback_pipelines_ready(p)) return false;
    const BufferRef input = p.fallback_task && d.indirect ? gpu_buffer_bind(d.indirect, 8, false) : BufferRef{};
    const VkDeviceAddress input_base = input ? fallback_buffer_address(input.buffer) : 0;
    fallback_indirect_input = input_base ? input_base + input.offset : 0;
    fallback_payload_address = f->payload_address;
    fallback_records_address = f->records_address;
    const bool capture = geometry_capture_on();
    const bool reuse = geometry_reuse_on();
    uint32_t dependencies = 0;
    bool eligible = false;
    GeometryStorage storage;
    GeometryStorageGpu* storage_gpu = nullptr;
    int current = -1, previous = -1;
    if (capture || reuse) {
        dependencies = geometry_dependencies(p, d);
        auto& inputs = geometry_candidate();
        inputs.size = 0;
        eligible = geometry_inputs(p, d, dependencies, inputs, storage);
        auto& recent = recent_geometry();
        const bool same = reuse && eligible && recent.program == &p && recent.recording == cur->ub_snap_epoch &&
                          recent.records == f->records && recent.indirect == bool(d.indirect) &&
                          recent.inputs.same(inputs);
        if (same && !storage.count && !d.indirect) {
            fallback_draw_records = f->records;
            frame_overlap_op(true);
            static std::atomic<uint64_t> hits{0};
            const uint64_t count = ++hits;
            if (count <= 4 || count % 1024 == 0)
                logf("vk: mesh geometry reused in its recording (%llu so far)", (unsigned long long)count);
            return true;
        }
        if (reuse && eligible && (storage.count || d.indirect)) {
            auto& gpu = geometry_storage_gpu();
            if (geometry_storage_ready(gpu)) {
                storage_gpu = &gpu;
                previous = same ? recent.snapshot : -1;
                current = gpu.next;
                gpu.next ^= 1;
            } else {
                eligible = false;
            }
        }
        // The new dispatch can overwrite the scratch even if it fails after the first clear.
        recent.program = nullptr;
    }

    const uint32_t capacity = Shader::Backend::SPIRV::MESH_FALLBACK_CAPACITY;
    end_pass(cmd);
    // (RX_VK_PASS_TIMING: the fills, barriers and dispatches below as one span of their own, "fallback prog <key>", not glue
    // between the spans; its barriers are in it, not among the "barrier" spans. After end_pass, which closes the pass's span)
    char label[32] = "";
    if (gprof_on()) snprintf(label, sizeof label, "fallback prog %016llx", (unsigned long long)p.key);
    GprofSpan span(cmd, label);
    // (the shared pair: the last fallback draw on it, of any program of this thread, has read it before it is reset)
    full_barrier(cmd);
    const uint32_t groups{(uint32_t)std::clamp(d.groups[0], 0, (int)capacity)};
    // The four draw words, the record allocator and the mesh dispatch's x/y/z.
    static_assert(Shader::Backend::SPIRV::MESH_FALLBACK_DISPATCH_WORD == 5 && Shader::Backend::SPIRV::MESH_FALLBACK_HEADER_WORDS == 8);
    if (storage_gpu) {
        geometry_storage_record(cmd, *storage_gpu, storage, *f, current, previous, d.indirect != 0);
        static std::atomic<uint64_t> snapshots{0};
        const uint64_t count = ++snapshots;
        if (count <= 4)
            logf("vk: mesh geometry recorded a %s snapshot (%u storage ranges, %u bytes)",
                 d.indirect ? "GPU-counted" : "direct", storage.count, storage.used);
    }
    if (previous < 0) {
        constexpr uint32_t empty_header[]{0, 1, 0, 0, 0, 0, 1, 1};
        vkCmdUpdateBuffer(cmd, f->records, 0, sizeof(empty_header), empty_header);
    }
    // (the attribute regions' pool, mma_layout.h mesh_fallback_global_attributes: its ticket counter at 0 and every slot free for
    // the first generation. The draw before handed every slot back, but a wait that gave up may not have: each draw starts clean.
    // Here, between the barrier after the draw before and the one before this draw's stages: after the stages, with no barrier
    // between, the clear could land while the mesh stage still takes and hands back slots)
    if (f->records_bytes >= Shader::Backend::SPIRV::MESH_FALLBACK_POOL_OFFSET + Shader::Backend::SPIRV::MESH_FALLBACK_POOL_BYTES)
        vkCmdFillBuffer(cmd, f->records, Shader::Backend::SPIRV::MESH_FALLBACK_POOL_OFFSET,
                        Shader::Backend::SPIRV::MESH_FALLBACK_POOL_HEADER_BYTES, 0);
    // (the shared warp's and the pool's counters past the records, emit_spirv.h WARP_COUNTER_*: at zero once, when the buffer
    // is first used; from then on they only grow)
    if (Shader::Backend::SPIRV::MeshFallbackCounters() && !f->warp_counters_cleared &&
        f->records_bytes >= Shader::Backend::SPIRV::MeshFallbackRecordsBytes() + Shader::Backend::SPIRV::WARP_COUNTER_BYTES) {
        vkCmdFillBuffer(cmd, f->records, Shader::Backend::SPIRV::MeshFallbackRecordsBytes(), Shader::Backend::SPIRV::WARP_COUNTER_BYTES, 0);
        f->warp_counters_cleared = true;
    }
    fallback_draw_records = f->records;
    full_barrier(cmd);
    if (storage_gpu) {
        const VkBuffer snapshot = storage_gpu->snapshots[current].buffer;
        for (uint32_t i = 0; i < storage.count; ++i) {
            const auto& range = storage.ranges[i];
            *range.binding = {snapshot, range.offset, range.original.range};
        }
    }
    const bool descriptors_bound = bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    if (storage_gpu)
        for (uint32_t i = 0; i < storage.count; ++i) *storage.ranges[i].binding = storage.ranges[i].original;
    if (!descriptors_bound) return false;
    push_constants(cmd, p, nullptr, d.first_task);
    size_t stages[2] = {p.stages.size(), p.stages.size()};
    for (size_t i = 0; i < p.stages.size(); i++) {
        if (p.stages[i].vk_stage != VK_SHADER_STAGE_COMPUTE_BIT) continue;
        if (stages[0] == p.stages.size()) stages[0] = i;
        else stages[1] = i;
    }
    const bool indirect{d.indirect && mesh_conv_buf != VK_NULL_HANDLE};
    auto send = [&](size_t stage, bool mesh_stage) {
        if (stage >= p.fallback_compute.size() || !p.fallback_compute[stage]) return false;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.fallback_compute[stage]);
        if (mesh_stage && p.fallback_task) {
            // (the mesh workgroups the task's started threads counted into the records' header, each its slot in the map:
            // direct or indirect, the CPU need not know the task count (DrawMeshTasksIndirect records none))
            vkCmdDispatchIndirect(cmd, f->records, (VkDeviceSize)Shader::Backend::SPIRV::MESH_FALLBACK_DISPATCH_WORD * 4);
        } else if (previous >= 0) {
            vkCmdDispatchIndirect(cmd, storage_gpu->snapshots[current].buffer, GeometryStorage::output_args);
            static std::atomic<uint64_t> comparisons{0};
            const uint64_t count = ++comparisons;
            if (count <= 4 || count % 1024 == 0)
                logf("vk: mesh geometry compared storage snapshots on the GPU (%llu so far)", (unsigned long long)count);
        } else if (d.indirect) {
            if (!indirect) {
                static std::atomic<int> told{0};
                if (told++ < 2) logf("vk: fallback: an indirect mesh draw with no converted arguments is left out");
                return false;
            }
            if (storage_gpu) vkCmdDispatchIndirect(cmd, storage_gpu->snapshots[current].buffer, GeometryStorage::input_args);
            else vkCmdDispatchIndirect(cmd, mesh_conv_buf, mesh_conv_off);
        } else {
            const uint32_t most{p.fallback_task ? (uint32_t)std::max(1u, capacity / std::max(1u, p.fallback_task_threads))
                                                : groups};
            const uint32_t n{mesh_stage ? groups : std::min(groups, most)};
            if (!n) return true;
            vkCmdDispatch(cmd, n, 1, 1);
        }
        return true;
    };
    if (p.fallback_task) {
        if (!send(stages[0], false)) return false;
        full_barrier(cmd);
        if (!send(stages[1], true)) return false;
    } else {
        if (!send(stages[0], true)) return false;
    }
    full_barrier(cmd);   // (the records and the draw's command, which the draw reads as indirect arguments)
    fallback_warp_counters(cmd, *f);
    if (capture) geometry_capture_draw(cmd, p, *f, dependencies | (eligible ? mesh_capture::candidate : 0));
    if (reuse && eligible) {
        auto& recent = recent_geometry();
        const auto& inputs = geometry_candidate();
        recent.program = &p;
        recent.records = f->records;
        recent.recording = cur->ub_snap_epoch;
        recent.snapshot = current;
        recent.indirect = d.indirect != 0;
        recent.inputs.size = inputs.size;
        std::memcpy(recent.inputs.bytes.data(), inputs.bytes.data(), inputs.size);
    }
    return true;
}
} // namespace

void bind_index(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
    const uint64_t b = (uint64_t)(uintptr_t)buffer;
    CmdBound& cb = bound();
    if (index_bind_same(b, (uint64_t)offset, (uint32_t)type, cb.ib.buffer, cb.ib.offset, cb.ib.type, cb.ib.on))
        return;
    vkCmdBindIndexBuffer(cmd, buffer, offset, type);
    cb.ib = {b, (uint64_t)offset, (uint32_t)type, true};
}

/// (hookshot, a diagnostic for the temporal upscaler's motion: vk_temporal.cpp) RX_VK_DLSS_TRACE=<frames>: every that many frames, one frame's
/// draws into targets of a size the upscaler can take as the scene (temporal_scene_size), logged: the program, its targets' formats, the geometry and the uniform buffers
static void dlss_trace_draw(const NvnState& s, const Program& p, const DrawArgs& d, const Targets& at) {
    static const int every = getenv("RX_VK_DLSS_TRACE") ? atoi(getenv("RX_VK_DLSS_TRACE")) : 0;
    if (every <= 0 || !temporal_scene_size(at.width, at.height)) return;
    const uint64_t f = texture_frame_no();
    if (f % (uint64_t)every != (uint64_t)every / 2) return;
    char line[3072];
    int n = snprintf(line, sizeof line, "vk: dlss-trace f%llu prog %016llx%s%s %ux%u c[", (unsigned long long)f,
                     (unsigned long long)p.key, d.mesh ? " mesh" : "", p.fallback ? "(fb)" : "", at.width, at.height);
    for (int i = 0; i < at.n && n < 1800; i++) n += snprintf(line + n, sizeof line - n, "%s%d", i ? "," : "", (int)at.color_format[i]);
    n += snprintf(line + n, sizeof line - n, "] d%d %s n%d i%d bv%d inst%d ib%llx g%d,%d,%d ind%llx", (int)at.depth_format,
                  d.indexed ? "idx" : "arr", d.count, d.first, d.base_vertex, d.instances, (unsigned long long)d.ibuf,
                  d.groups[0], d.groups[1], d.groups[2], (unsigned long long)d.indirect);
    for (int v = 0; v < 4 && n < 1800; v++)
        if (s.vbuf[v].address) n += snprintf(line + n, sizeof line - n, " vb%d=%llx/%llu/s%u", v, (unsigned long long)s.vbuf[v].address,
                                             (unsigned long long)s.vbuf[v].size, s.vstream[v].stride);
    n += snprintf(line + n, sizeof line - n, " va");
    for (int i = 0; i < s.nvattr && i < 16 && n < 1800; i++) {
        const nvn_va_st a = *reinterpret_cast<const nvn_va_st*>(&s.vattr[i]);
        n += snprintf(line + n, sizeof line - n, "%s%x@%u:%u", i ? "," : "=", a.fmt, a.offset, a.stream);
    }
    for (const Stage& st : p.stages) {
        const int ns = std::clamp(st.nvn_stage, 0, kNvnStages - 1);
        n += snprintf(line + n, sizeof line - n, " st%d:", ns);
        for (int u = 0; u < 16 && n < 1900; u++)
            if (s.ubo[ns][u].addr) n += snprintf(line + n, sizeof line - n, " u%d=%llx/%llu", u, (unsigned long long)s.ubo[ns][u].addr,
                                                 (unsigned long long)s.ubo[ns][u].size);
        for (int u = 0; u < 16 && n < 1900; u++)
            if (s.ssbo[ns][u].addr) n += snprintf(line + n, sizeof line - n, " s%d=%llx/%llu", u, (unsigned long long)s.ssbo[ns][u].addr,
                                                  (unsigned long long)s.ssbo[ns][u].size);
    }
    logf("%s", line);
}

/// (hookshot) A skinned draw into the scene, for the temporal upscaler's actor motion (vk_temporal.cpp: DLSS's and FSR's). OoT's characters and the things that
/// move by themselves are indexed draws of one vertex stream: the position at attribute 0 (RGB32F or RGBA16F), the weights at 4
/// (and 5, RGBA8), the joints at 6 (and 7, RGBA16UI or RGBA8UI), the bone palette in the vertex stage's uniform buffer 6 (NVN's
/// binding 3: 3 x 4 matrices, a whole number of them) and the object's origin in its buffer 4 (binding 1, at 0x40).
/// (hookshot) Such a draw is described in `k` (its depth set; left empty otherwise), and draw() gives it to actor
/// motion only after hidden_depth_draw: a draw left out there has no motion to draw again
static void skin_check(const NvnState& s, const Program& p, const DrawArgs& d, const Targets& at, SkinnedDraw& k) {
    k = SkinnedDraw{};
    if (d.mesh || !d.indexed || d.indirect || d.instances != 1 || !at.n || !at.color[0] || !at.depth ||
        !temporal_scene_size(at.width, at.height) || !temporal_scene_format(at.color_format[0]) || s.nvattr < 7 ||
        (!temporal_wanted() && !temporal_fg_inputs()))   // (hookshot: and for frame generation's inputs alone)
        return;
    const auto attr = [&](int i) { return *reinterpret_cast<const nvn_va_st*>(&s.vattr[i]); };
    const nvn_va_st pos = attr(0);
    if ((pos.fmt != NVN_FORMAT_RGB32F && pos.fmt != NVN_FORMAT_RGBA16F) || pos.stream != 0) return;
    const nvn_p_bind_buf& bones = s.ubo[0][3];
    if (!bones.addr || bones.size < 48 || bones.size % 48) return;
    k.formats = pos.fmt == NVN_FORMAT_RGBA16F ? dlss_actor::kHalfPosition : 0;
    for (int set = 0; set < 2 && 6 + set < s.nvattr; set++) {
        const nvn_va_st w = attr(4 + set), j = attr(6 + set);
        // (hookshot) a second set of one influence: an R8 weight and an R8UI or R16UI joint (Link's draws: a fifth bone,
        // his cap's among them; left out, those vertices were skinned with part of their weight and their motion was wrong)
        if (set == 1 && w.fmt == NVN_FORMAT_R8 && (j.fmt == NVN_FORMAT_R8UI || j.fmt == NVN_FORMAT_R16UI) && !w.stream && !j.stream) {
            k.formats |= dlss_actor::kSingle << set;
            if (j.fmt == NVN_FORMAT_R8UI) k.formats |= dlss_actor::kByteJoints << set;
            k.weights[set] = w.offset, k.joints[set] = j.offset, k.sets = set + 1;
            break;
        }
        // (hookshot) a rigid model on one bone: a joint and no weight (Navi's body, props, held items: a palette of one
        // matrix, sometimes more); and two-bone skinning, RG8 weights with RG8UI joints (Navi's wings among them)
        if (set == 0 && !w.fmt && (j.fmt == NVN_FORMAT_R8UI || j.fmt == NVN_FORMAT_R16UI) && !j.stream) {
            k.formats |= (dlss_actor::kSingle | dlss_actor::kNoWeight) << set;
            if (j.fmt == NVN_FORMAT_R8UI) k.formats |= dlss_actor::kByteJoints << set;
            k.weights[set] = 0, k.joints[set] = j.offset, k.sets = set + 1;
            break;
        }
        if (w.fmt == NVN_FORMAT_RG8 && j.fmt == NVN_FORMAT_RG8UI && !w.stream && !j.stream) {
            k.formats |= (dlss_actor::kPair | dlss_actor::kByteJoints) << set;
            k.weights[set] = w.offset, k.joints[set] = j.offset, k.sets = set + 1;
            continue;
        }
        if (w.fmt != NVN_FORMAT_RGBA8 || (j.fmt != NVN_FORMAT_RGBA16UI && j.fmt != NVN_FORMAT_RGBA8UI) || w.stream || j.stream ||
            w.offset % 4 || j.offset % 4)
            break;
        if (j.fmt == NVN_FORMAT_RGBA8UI) k.formats |= dlss_actor::kByteJoints << set;
        k.weights[set] = w.offset, k.joints[set] = j.offset, k.sets = set + 1;
    }
    const uint32_t stride = s.vstream[0].stride;
    if (!k.sets || !stride || stride % 4 || pos.offset % 4 || !s.vbuf[0].address) return;
    k.depth = at.depth, k.vbuf = s.vbuf[0].address, k.vsize = s.vbuf[0].size, k.stride = stride, k.pos = pos.offset;
    k.ibuf = d.ibuf, k.itype = d.itype, k.count = d.count, k.base_vertex = d.base_vertex;
    k.bones = bones.addr, k.bones_size = bones.size, k.object = s.ubo[0][1].addr, k.object_size = s.ubo[0][1].size;
    k.program = p.key;
}

// ---- (hookshot) the temporal upscalers' depth with a hidden character: a hidden mesh's depth draws are left out ----
// When the camera is pushed into Link (the crawl tunnels, the top of a vine wall) the game stops drawing his colour, but goes
// on drawing his body into the scene's depth: his 30 meshes' z-prepass draws, which write depth and no colour (the first
// skinned draws of a frame). The temporal upscalers and frame generation take that depth, and actor motion draws his motion
// over it, so they follow an invisible body where the screen shows the room behind it: copies of things and a drag over the
// picture, under DLSS and DLAA alike (RX_VK_DLSS_DEBUG=motion shows solid shapes at the camera). The game's own upscale has
// no history and hardly shows it.
// So those draws are left out: the game's own draws, nothing added. The rule is gpu/vk/hidden_mesh.h's: a mesh (a vertex
// buffer) is hidden after kHiddenAfter frames in a row with such depth draws and no colour draw from it into any target,
// following colour; its depth draws are drawn again from the frame after its colour comes back. Never left out: a draw the
// game asks about with an occlusion query (OoT's culling counts its z-prepass: vk_replay.cpp occ_take), a draw into another
// depth target (shadows, other views), a mesh that never had colour.
// Asked right after skin_check, before the draw is given to actor motion and before its descriptors are built. Nothing here
// runs without a temporal upscaler or frame generation's inputs (skin_check's own rule): no mesh is followed then. With
// them, the game's own picture changes too where it reads that depth: later depth-tested draws, the occlusion counts of what
// is behind the body, effects made from the depth no longer see the hidden body.
// One table for every drawing thread: the game submits from several threads, the replay draws one submission at a time
// (vk_replay.cpp replay_mu), and a mesh's depth and colour can come from two of them. With the parallel replay
// (RX_VK_PARALLEL=2 and more) the workers draw at once, and nothing here runs (temporal_hidden_mesh_mode).
// RX_VK_DLSS_FADED=0: nothing left out; RX_VK_DLSS_FADED_LOG=1: the detailed lines (temporal_hidden_mesh_mode). From the
// patch magicbeanenjoyerguyperson posted, who found the cause.
namespace {
struct HiddenMeshes {
    hidden_mesh::Tracker t;
    bool said_first = false, said_out = false, said_query = false, said_gone = false;   // (the detailed lines, once a run)
    // (what was left out: one line for each stretch of kHiddenStretch frames that had any, kHiddenStretches lines a run)
    uint64_t stretch_start = 0, stretch_last = 0, stretch_draws = 0, stretch_key = 0, stretch_program = 0;
    uint64_t stretch_stencil = 0, stretch_back = 0;
    uint32_t stretch_frames = 0, stretch_most = 0;
    int stretches = 0;
};
constexpr int kHiddenStretches = 300;
constexpr uint64_t kHiddenStretch = 600;
HiddenMeshes* hidden_meshes_one = nullptr;   // (made at the first skinned draw; under replay_mu, as every draw)
} // namespace

/// A frame is over: its stretch line when one is due, and with the detailed lines on (mode's bit 1) the counts of the first
/// frames and why a hidden mesh's draws were drawn
static void hidden_meshes_log(HiddenMeshes& h, const hidden_mesh::Summary& s, int mode) {
    if (!s.f) return;
    const hidden_mesh::Frame& f = *s.f;
    if (f.left_out) {   // (into the stretch that is being counted)
        if (!h.stretch_draws) h.stretch_start = f.frame, h.stretch_key = f.out_key, h.stretch_program = f.out_program;
        h.stretch_draws += f.left_out, h.stretch_frames++, h.stretch_last = f.frame;
        h.stretch_most = std::max(h.stretch_most, f.left_out);
        h.stretch_stencil += f.stencil, h.stretch_back += f.back;
    }
    if (h.stretch_draws && f.frame - h.stretch_start >= kHiddenStretch) {
        // (back: a mesh's colour came in a frame with its depth draws left out, which the game's EQUAL colour test then
        // does not show: once for each of its meshes each time the camera leaves Link, 30; more is a mesh that blinks)
        if (h.stretches < kHiddenStretches)
            logf("vk: upscaler: faded: frames %llu to %llu: %llu depth draws of hidden meshes left out, in %u frames, %u in a "
                 "frame at most, %llu of them with a stencil test; %llu times a mesh's colour came back in a frame with them "
                 "left out; the first of vertex buffer %llx, program %016llx%s",
                 (unsigned long long)h.stretch_start, (unsigned long long)h.stretch_last, (unsigned long long)h.stretch_draws,
                 h.stretch_frames, h.stretch_most, (unsigned long long)h.stretch_stencil, (unsigned long long)h.stretch_back,
                 (unsigned long long)h.stretch_key, (unsigned long long)h.stretch_program,
                 ++h.stretches == kHiddenStretches ? " (the last of these lines for this run)" : "");
        h.stretch_draws = 0, h.stretch_frames = 0, h.stretch_most = 0, h.stretch_stencil = 0, h.stretch_back = 0;
    }
    if (!(mode & 2)) return;
    // (once: the first frame with a followed mesh; and once: the first frame with a draw left out)
    const bool first = !h.said_first && (s.meshes || s.colour_only), first_out = !h.said_out && f.left_out;
    if (first || first_out) {
        h.said_first = true, h.said_out |= f.left_out != 0;
        logf("vk: upscaler: faded: frame %llu: %u meshes with %u depth draws into the scene; %u of them got colour in the frame "
             "(%u colour draws on the scene's depth target, %u on others), %u got none, %u of those hidden in the next frame; "
             "%u more meshes had colour and no depth draw; %u depth draws left out, %u drawn for a query; %u draws of more "
             "meshes than the table holds; one %s: vertex buffer %llx, program %016llx",
             (unsigned long long)f.frame, s.meshes, f.depth_draws, s.coloured, f.scene_colour, f.other_colour, s.bare, s.hidden,
             s.colour_only, f.left_out, f.kept_query, f.over, s.hidden ? "hidden" : "without colour",
             (unsigned long long)(s.one ? s.one->key : 0), (unsigned long long)(s.one ? s.one->program : 0));
    }
    // (meshes the frame before left hidden, and none of their depth draws left out in this one: why not, each answer once)
    if ((mode & 1) && s.inherited && !f.left_out) {
        if (f.kept_query && !h.said_query) {
            h.said_query = true;
            logf("vk: upscaler: faded: frame %llu: %u meshes were hidden, and no depth draw of theirs was left out: %u were "
                 "drawn because the game asks about them with an occlusion query",
                 (unsigned long long)f.frame, s.inherited, f.kept_query);
        } else if (!f.kept_query && !h.said_gone) {
            h.said_gone = true;
            logf("vk: upscaler: faded: frame %llu: %u meshes were hidden, and no depth draw of theirs was left out: none of "
                 "them had a depth draw into the scene in it", (unsigned long long)f.frame, s.inherited);
        }
    }
}

/// A draw, after skin_check (`skinned`: a skinned draw into the scene): counted for the hidden-mesh rule. True: it is a
/// hidden mesh's depth draw and is to be left out
static bool hidden_depth_draw(const Program& p, const NvnState& s, const DrawArgs& d, const Targets& at, bool skinned) {
    static const int mode = temporal_hidden_mesh_mode();   // (bit 0: leave out; bit 1: the detailed lines; 0: nothing runs)
    if (!mode) return false;
    // (no mesh followed in this frame or the one before: nothing to look up; always so without an upscaler)
    if (!skinned && (!hidden_meshes_one || !hidden_meshes_one->t.active())) return false;
    if (!hidden_meshes_one) hidden_meshes_one = new HiddenMeshes();   // (kept for the run)
    HiddenMeshes& h = *hidden_meshes_one;
    const uint64_t frame = texture_frame_no();
    if (h.t.now.frame != frame) hidden_meshes_log(h, h.t.begin(frame), mode);
    hidden_mesh::Draw x;
    // (the mesh: vertex stream 0, where the draw reads a vertex from it. A mesh-shader draw, or one with no vertex input
    // from stream 0, keeps the buffer the draw before bound: its colour is not that mesh's)
    bool reads_stream0 = false;
    for (int i = 0; i < s.nvattr && i < 16 && !reads_stream0; i++) {
        const nvn_va_st a = *reinterpret_cast<const nvn_va_st*>(&s.vattr[i]);
        reads_stream0 = a.fmt && a.stream == 0;
    }
    x.key = !d.mesh && reads_stream0 ? s.vbuf[0].address : 0;
    x.program = p.key;
    x.target = (uint64_t)(uintptr_t)at.depth;
    x.skinned = skinned;
    for (int i = 0; i < at.n && !x.colour; i++) x.colour = target_write_mask(p, s, i) != 0;
    x.depth_write = at.depth && s.ds.dw;
    x.stencil = at.depth && has_stencil(at.depth_format) && s.ds.st;
    x.query = d.occ_pool != VK_NULL_HANDLE;
    return h.t.draw(x, (mode & 1) != 0, [](uint64_t target) {
        return temporal_scene_depth_is(reinterpret_cast<const Texture*>((uintptr_t)target));
    });
}

/// OoT's tree-shadow pass (3427a1dbefd5e02c: a full-screen draw that writes G and B of an RGBA8 mask at the scene's size, MIN
/// blended) offsets each pixel's shadow-map lookup by a normal it reconstructs from a gather of the neighbouring depths. On
/// grass blades and leaf edges those neighbours flip between the blade and the ground behind it with the camera's sub-pixel
/// shift, so pixels flip between lit and shadowed every frame: fast thin streaks over sunlit ground (the console's TAAU
/// averages them away). The mask is evened out over frames right after that draw (shaders/shadow_stab.comp).
/// RX_VK_MASK_BLEND=<weight> (0.85 was the default; 0: off), RX_VK_MASK_RADIUS=<r> (the history's bound, a (2r+1)^2
/// neighbourhood; default 0: unbounded. A 3x3 bound kept nothing: the streaks move as blocks wider than that).
/// OFF BY DEFAULT since 2026-11-11: with it on, the GPU hung on the mesh draw before this pass (b0b3b40244c71ec7) near Hyrule
/// Castle, device lost after ~8 minutes on the owner's save, twice, and in 2 of the night's route runs. The same build with
/// RX_VK_MASK_BLEND=0 ran 14 minutes there clean, as v2.1.0 (before this pass) did. Opt-in until the hang is found
float mask_weight() {
    static const float w = std::clamp(getenv("RX_VK_MASK_BLEND") ? (float)atof(getenv("RX_VK_MASK_BLEND")) : 0.0f, 0.0f, 0.95f);
    return w;
}
int mask_radius_env() {
    static const int r = std::clamp(getenv("RX_VK_MASK_RADIUS") ? atoi(getenv("RX_VK_MASK_RADIUS")) : 0, 0, 4);
    return r;
}
/// (hookshot, RX_VK_MASK_BLEND) What mask_stabilize no longer uses: the history images a new mask size replaced, and the views of
/// textures that went (texture_destroy_generation). Given back 120 presents later, when no work in flight can use them, as the TAAU's
/// replaced histories are. They were overwritten where they stood: an image, its memory and its view lost at every size change (the
/// dynamic resolution makes one every few frames), and a view kept for each texture image the mask was blended into, after the image went,
/// and found again under its handle for another image
struct MaskOld {
    VkImage image;
    VmaAllocation mem;
    VkImageView view;
    uint64_t frame;
};
std::vector<MaskOld> mask_old;
void mask_retire(VkImage& image, VmaAllocation& mem, VkImageView& view, uint64_t frame) {
    if (image || view) mask_old.push_back({image, mem, view, frame});
    image = VK_NULL_HANDLE, mem = nullptr, view = VK_NULL_HANDLE;
}
void mask_reap(uint64_t frame) {
    for (size_t i = 0; i < mask_old.size();) {
        if (frame - mask_old[i].frame < 120) {
            i++;
            continue;
        }
        if (mask_old[i].view) vkDestroyImageView(dev.device, mask_old[i].view, nullptr);
        if (mask_old[i].image) destroy_image(mask_old[i].image, mask_old[i].mem);
        mask_old.erase(mask_old.begin() + (ptrdiff_t)i);
    }
}
void mask_stabilize(VkCommandBuffer cmd, Texture* tex, uint32_t level, uint32_t layer, float weight) {
    if (!tex || !tex->image || tex->format != VK_FORMAT_R8G8B8A8_UNORM || !(tex->usage & VK_IMAGE_USAGE_STORAGE_BIT)) return;
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return;
    static VkPipeline pipe = VK_NULL_HANDLE;
    static VkPipelineLayout layout = VK_NULL_HANDLE;
    static bool tried = false;
    if (!tried) {
        tried = true;
        VkDescriptorSetLayoutBinding b[3] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                             {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                             {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        li.bindingCount = 3, li.pBindings = b;
        VkDescriptorSetLayout set{};
        vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &set);
        const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
        VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pi.setLayoutCount = 1, pi.pSetLayouts = &set, pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &pr;
        vkCreatePipelineLayout(dev.device, &pi, nullptr, &layout);
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        const auto spv = BUILTIN_SPV(SHADOW_STAB_COMP_SPV);
        mi.codeSize = spv.size, mi.pCode = spv.code;
        VkShaderModule m{};
        {
            IntelDriverGate gate;   // (the module's only: texture_compute_pipeline takes the same lock)
            if (guarded_shader_module(&mi, &m) != VK_SUCCESS) m = VK_NULL_HANDLE;
        }
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
        ci.layout = layout;
        // (vk_texture.cpp: through the pipeline cache, and without the optimizer where the driver refuses it. Asked once,
        // optimized, here: an Arc A580 on 101.9175 refused the typed upscale helper's, a network's output a blit)
        if (!m || texture_compute_pipeline(ci, pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
        vkDestroyShaderModule(dev.device, m, nullptr);
        logf("vk: tree-shadow mask blend %s", pipe ? "ready" : "not made");
    }
    if (!pipe) return;
    const uint32_t w = std::max(1u, tex->width >> level), h = std::max(1u, tex->height >> level);
    // (the mask's own storage view at that level and layer, made once a texture and level)
    static std::map<std::tuple<VkImage, uint32_t, uint32_t>, VkImageView> views;
    static uint64_t views_gen = 0;
    mask_reap(texture_frame_no());
    if (views_gen != texture_destroy_generation()) {   // (a texture went: every view is given back, the ones still wanted made again)
        for (auto& kv : views) {
            VkImage none = VK_NULL_HANDLE;
            VmaAllocation no_mem = nullptr;
            mask_retire(none, no_mem, kv.second, texture_frame_no());
        }
        views.clear();
        views_gen = texture_destroy_generation();
    }
    VkImageView& mv = views[{tex->image, level, layer}];
    if (!mv) {
        VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vc.image = tex->image, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = VK_FORMAT_R8G8B8A8_UNORM;
        vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, layer, 1};
        if (vkCreateImageView(dev.device, &vc, nullptr, &mv) != VK_SUCCESS) { mv = VK_NULL_HANDLE; return; }
    }
    static VkImage hist[2] = {};
    static VmaAllocation hist_mem[2] = {};
    static VkImageView hist_view[2] = {};
    static uint32_t hist_w = 0, hist_h = 0, cur = 0;
    static uint64_t hist_frame = 0;
    end_pass(cmd); // History image transitions must be outside dynamic rendering.
    if (hist_w != w || hist_h != h) {
        for (int i = 0; i < 2; i++) mask_retire(hist[i], hist_mem[i], hist_view[i], texture_frame_no());   // (the old size's, given back later)
        for (int i = 0; i < 2; i++) {
            VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ic.imageType = VK_IMAGE_TYPE_2D, ic.format = VK_FORMAT_R8G8B8A8_UNORM;
            ic.extent = {w, h, 1}, ic.mipLevels = 1, ic.arrayLayers = 1, ic.samples = VK_SAMPLE_COUNT_1_BIT;
            ic.tiling = VK_IMAGE_TILING_OPTIMAL, ic.usage = VK_IMAGE_USAGE_STORAGE_BIT;
            if (!create_image(ic, hist[i], hist_mem[i])) { hist_w = hist_h = 0; return; }
            VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vc.image = hist[i], vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = ic.format;
            vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCreateImageView(dev.device, &vc, nullptr, &hist_view[i]);
            VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            ib.srcStageMask = VK_PIPELINE_STAGE_2_NONE, ib.srcAccessMask = VK_ACCESS_2_NONE;
            ib.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            ib.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib.image = hist[i], ib.subresourceRange = vc.subresourceRange;
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &ib;
            vkCmdPipelineBarrier2(cmd, &di);
        }
        hist_w = w, hist_h = h, hist_frame = 0, cur = 0;
    }
    const uint64_t now_frame = texture_frame_no();
    const int valid = hist_frame && now_frame - hist_frame <= 3 ? 1 : 0;
    hist_frame = now_frame;
    const uint32_t prev = cur, nxt = cur ^ 1u;
    cur = nxt;
    const VkDescriptorImageInfo infos[3] = {{VK_NULL_HANDLE, mv, VK_IMAGE_LAYOUT_GENERAL},
                                            {VK_NULL_HANDLE, hist_view[prev], VK_IMAGE_LAYOUT_GENERAL},
                                            {VK_NULL_HANDLE, hist_view[nxt], VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[3]{};
    for (int i = 0; i < 3; i++) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, wr[i].dstBinding = (uint32_t)i, wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, wr[i].pImageInfo = &infos[i];
    }
    struct { int32_t size[2]; int32_t valid, mode; float weight; int32_t radius; } pc{{(int32_t)w, (int32_t)h}, valid, 0, weight, mask_radius_env()};
    full_barrier(cmd);   // (after the pass's write of the mask)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 3, wr);
    descriptors_disturbed();   // (what bind_descriptors remembers pushing is not what is bound now: the next dispatch pushes all again; unmarked, a dispatch of the same layout kept these bindings and read our images: the device lost)
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cmd);
    pc.mode = 1;
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cmd);   // (before the next pass reads or draws the mask)
    static std::atomic<int> told{0};
    if (told++ == 0) logf("vk: tree-shadow mask blend on #%u %ux%u, weight %.2f", tex->id, w, h, weight);
}

// ---- RX_VK_OCC_PROXY_SKIP=1 (off by default): draws that cannot change anything, left out of the command buffer ----
// OoT starts each frame with ~900 occlusion proxy tests: a draw between a samples-passed reset and its report. With the default
// occlusion (RX_VK_OCC_REAL off, RX_VK_OCC_FIXED >= 0) the report's count is the fixed value, written on the CPU in stream order
// (vk_replay.cpp NOP_REPORT_COUNTER), whatever the draws did, and the render conditions read those words: the draws' own GPU
// work is not observed. A draw is left out here only when it provably writes nothing and reads nothing a later command depends on:
// - no colour write: every bound colour target's write mask is 0 (target_write_mask: the same masks the pipeline and the dynamic
// state use; no fragment stage writes no colour);
// - no depth write (no depth target, or its write off) and no stencil test (no stencil aspect, or the test off: then it writes no
// stencil whatever its ops);
// - its program binds no storage buffer, storage image, image buffer, texture or texture buffer in any stage, touches no global
// memory, and is neither a mesh nor a mesh fallback program (program_inert): a shader with no such resource has no memory to
// write, and with no texture it samples nothing (so draw_hazards has no hazard to find for it and adds nothing to the pass);
// - no diagnosis that watches draws (dumps).
// What it keeps, so that the command buffer's structure and every texture's state stay as they were: everything before the
// draw's own commands (the program pick, the pipeline lookup and its not-ready rule: a draw whose pipeline is not ready is left out
// earlier, as before), and begin_pass: the pass
// ends and begins where it did, with the same barrier and the same texture_written / texture_rendered marks of its targets, and
// the pass counts the draw (pass.draws: a clear after it still gets a pass of its own). What it leaves out: the draw's
// descriptor build and bind, pipeline bind, push constants (and its NV checkpoint marker), dynamic state, vertex / index binds and
// the vkCmdDraw. The caches that skip repeated binds (last_pipeline, the descriptor reuse, dynamic state, vertex sets) describe
// what the command buffer has bound, and are left as the last recorded draw set them, which is what it still has bound.
bool occ_proxy_skip_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_OCC_PROXY_SKIP");
        const char* fixed = getenv("RX_VK_OCC_FIXED");
        const bool v = e && atoi(e) != 0 && occ_mode() == 0 && (!fixed || atoll(fixed) >= 0);
        logf("vk: draws that write nothing and sample nothing: %s (RX_VK_OCC_PROXY_SKIP)", v ? "left out (their pass kept)" : "recorded, as before");
        return v;
    }();
    return on;
}
// (the answer kept by the program's key and variant, each thread its own: not in Program, whose size is part of the program
// cache's stamp (vk_shader.cpp program_cache_stamp: sizeof(Program)); a field there made every program cache on disk start again.
// A key and variant name one translation, so the same stages and the same answer)
bool program_inert(const Program& p) {
    thread_local std::unordered_map<uint64_t, bool> known;
    const uint64_t id = p.key ^ (p.variant * 0x9E3779B97F4A7C15ull) ^ (p.compute ? 1 : 0);
    if (const auto it = known.find(id); it != known.end()) return it->second;
    bool inert = !p.compute && !p.mesh && !p.fallback && !p.stages.empty();
    for (const Stage& st : p.stages) {
        const Shader::Info& i = st.info;
        if (!i.storage_buffers_descriptors.empty() || !i.image_descriptors.empty() || !i.image_buffer_descriptors.empty() ||
            !i.texture_descriptors.empty() || !i.texture_buffer_descriptors.empty() || i.stores_global_memory || i.uses_global_memory ||
            i.uses_atomic_image_u32 || i.num_global_tables)
            inert = false;
    }
    if (known.size() > 65536) known.clear();
    known.emplace(id, inert);
    return inert;
}
bool draw_writes_nothing(const Program& p, const NvnState& s, const Targets& at) {
    for (int i = 0; i < at.n; i++)
        if (at.color[i] && target_write_mask(p, s, i)) return false;
    if (at.depth && s.ds.dw) return false;
    if (at.depth && has_stencil(at.depth_format) && s.ds.st) return false;
    return true;
}
static bool writes_memory(const Program& p) {
    for (const Stage& st : p.stages) {
        const auto& info = st.info;
        const auto written = [](const auto& ds) {
            return std::any_of(ds.begin(), ds.end(), [](const auto& d) { return d.is_written; });
        };
        if (info.stores_global_memory || written(info.storage_buffers_descriptors) ||
            written(info.image_buffer_descriptors) || written(info.image_descriptors))
            return true;
    }
    return false;
}

// A mesh fallback's colour draws are left out until their pipelines build (minutes each on an Iris Xe), while the smaller
// depth-only draws of the same culled list (same indirect arguments) are already built. Depth with no colour lets the sky show
// through the grass, so a depth-only draw is held while its list's colour draw is.
enum class FallbackRole { None, DepthOnly, Colour };
static FallbackRole fallback_role_of(const Program& p, const NvnState& s, const Targets& at, const DrawArgs& d) {
    if (!d.indirect || writes_memory(p)) return FallbackRole::None;
    uint32_t wm = 0;
    for (int i = 0; i < at.n; i++)
        if (at.color[i]) wm |= target_write_mask(p, s, i);
    if (wm) return FallbackRole::Colour;
    return at.depth && s.ds.dw ? FallbackRole::DepthOnly : FallbackRole::None;
}
static std::mutex fallback_held_mu;
// Indirect-argument address -> frame its colour draw was last left out; an address is assumed to stay with one list for the hold.
static std::unordered_map<uint64_t, uint64_t> fallback_held;
/// Marks the list's colour draw as left out (held) or recorded (!held) this frame.
static void fallback_list_hold(uint64_t list, bool held) {
    std::lock_guard lk(fallback_held_mu);
    if (held) fallback_held[list] = texture_frame_no();
    else fallback_held.erase(list);
}
/// True while the list's colour draw was left out in the last 120 frames (so a list no colour draw uses any more is released).
static bool fallback_list_held(uint64_t list) {
    std::lock_guard lk(fallback_held_mu);
    const auto it = fallback_held.find(list);
    if (it == fallback_held.end()) return false;
    if (texture_frame_no() - it->second <= 120) return true;
    fallback_held.erase(it);
    return false;
}

std::atomic<int64_t> st_occ_proxy_skips{0};
int64_t occ_proxy_skips() { return st_occ_proxy_skips.load(std::memory_order_relaxed); }

void draw(VkCommandBuffer cmd, NvnState& s, const DrawArgs& d) {
    FineScope fine_prep(FT_PREP);
    FineScope fine_targets(FT_TARGETS);
    if (par_stats_on) par_stats_draw();
    handle_memo_next();   // (hookshot: a new draw or dispatch: the handles kept for the last one go)
#ifdef TRACY_ENABLE
    HT_SCOPE_DRAW("draw");
#endif
    Pass& ps = pass();   // (once: the accessor is a call and a thread-local read at each use. The same object for the thread's life: draw_begin_recording resets it in place)
    CmdBound& cb = bound();   // (once, as the pass: what the command buffer has bound)
    PROF_HOT(PROF_DRAW);
    FINE_N(FC_DRAWS, 1);
    stats_tick();
    if (d.mesh) mesh_indirect_convert(cmd, d);   // (before the pass opens; sets mesh_conv_buf for this mesh draw)
    VkPrimitiveTopology topo = kMeshDrawTopology;
    if (!d.mesh && !topology(d.prim, topo)) {
        static int logged[32];
        if (d.prim >= 0 && d.prim < 32 && logged[d.prim]++ < 2) logf("vk: draws of NVN primitive %d are left out", d.prim);
        SKIP_HERE();
    }
    if (!d.mesh && d.prim == NVN_DRAW_PRIMITIVE_PATCHES) {   // (said here: topology() is draw_key.h's, which logs nothing)
        static int logged;
        if (logged++ < 5) logf("vk: a draw with patches");
    }
    const Targets at = bare_targets(s, targets(s));
    if (!at.width || !at.height) SKIP_HERE();
    // (hookshot) the scene's camera and depth for the temporal upscalers: NVIDIA DLSS and AMD FSR 4 (vk_temporal.cpp) and AMD
    // FSR 3.1 (vk_fsr3_inputs.cpp). The vertex stage's uniform buffer 3 (NVN's binding 0)
    if (at.n && at.color[0] && at.depth) {
        temporal_note_scene_draw(at.color[0], at.width, at.height, at.depth, s.ubo[0][0].addr, s.ubo[0][0].size);
        fsr3_inputs::note_scene_draw(at.color[0], at.width, at.height, at.depth, s.ubo[0][0].addr, s.ubo[0][0].size);
    }
    uint8_t types[8];
    color_types_of(at, types);
    const Program* cp;
    ProgramUse use(s.prog, s.has_prog, false);   // (the program's first draw, for the census: vk_programs.h)
    fine_targets.finish();
    {
        FINE_SCOPE(FT_PROGRAM_PICK);
        PROF_HOT(PROF_PROGRAM);
        ParLock lk(d.mesh);   // (graphics programs: each thread's picks, program_pick's own lock)
        program_load_async_here(true);   // (hookshot: a cached program's read goes to the workers: vk_shader.cpp)
#ifdef TRACY_ENABLE
        HT_SCOPE_DRAW("program_for");   // (here, not in vk_shader.cpp: that file is part of the shader caches' stamp)
#endif
        cp = program_for(s, d.mesh ? ProgramKind::Mesh : ProgramKind::Graphics, types);
        program_load_async_here(false);
    }
    if (use.on) use.program(cp, cp && cp->ready.load(std::memory_order_acquire));
    {   // (hookshot, a test) RX_VK_NO_MESH=1: task / mesh draws left out, to see what they draw; =2: the others left out
        static const int no_mesh = getenv("RX_VK_NO_MESH") ? atoi(getenv("RX_VK_NO_MESH")) : 0;
        if ((no_mesh == 1 && d.mesh) || (no_mesh == 2 && !d.mesh)) return;
    }
    if (!cp) {   // (hookshot) which draw has no program, 12 times: its stages' native code keys
        static std::atomic<int> told{0};
        if (told++ < 12) logf("vk: a draw without a program (frame %llu); its code:%s", (unsigned long long)texture_frame_no(),
                              program_code_keys(s).c_str());
        SKIP_HERE();
    }
    // (ultrawide's menu sides: the world's lighting composite was asked for, even if a skip or a pipeline not yet made leaves this
    // draw out below: the world is drawn, it is no menu. vk_framegen.cpp wide_sides)
    if (cp->key == 0x1a7482c0ec9991eaULL) world_drawn_at.store(presents_now, std::memory_order_relaxed);
    if (cp->key == 0x02ff9920c2571cc2ULL) wheel_drawn_at.store(presents_now, std::memory_order_relaxed);
    if (cp->key == 0xbecef27180cc822fULL) menu_fade_pause_at.store(presents_now, std::memory_order_relaxed);
    if (skip_live(cp->key)) return;   // (EvilOOT, RX_VK_SKIP_FILE)
    {   // (hookshot, RX_VK_SKIP_PROG=<key prefix,...>: draws too, their native code keys logged once; =-: none)
        // (OoT's 53509044401d78d2, by default only while its targets have no view: 128 draws, once while an area loads, that render
        // the slices of two 64^3 BC4 noise volumes through raw RGBA16UI views. Those views were not made (no attachment usage) and
        // the pass ran over no target at the texels' 64 x 64: on about half the first starts the device was lost (7 of 14), and it
        // was left out always. Left out, the volumes stayed zero and the foliage's LOD crossfade left trees bare at mid distance
        // (#1151). Where the driver takes the usage (vk_texture.cpp) the views are made and the pass is sized in blocks
        // (targets()); it runs there, and is still left out where they are not)
        static const bool default_list = !getenv("RX_VK_SKIP_PROG");
        if (default_list && cp->key == 0x53509044401d78d2ull && !at.n) {
            static std::atomic<int> told{0};
            if (told++ < 2)
                logf("vk: draws of program %016llx left out: its targets have no view (RX_VK_SKIP_PROG=- runs them anyway)",
                     (unsigned long long)cp->key);
            return;
        }
        // (hookshot, 2026-11-14) OoT's 2b37a33d189014a8 and 0948a49a3c3068a8: two small box draws (24 indices) into a
        // half-size HDR target that read the 128 x 72 x 96 volume textures, every frame. On an RTX 5070 Ti the device was lost as
        // soon as Link entered water, and both of a player's reports named these two as the only draws in flight both times, the
        // second the last one begun (NV checkpoints, knoch23, v0.6.6). On this RTX 3060 Ti they never hang, but they are wrong
        // too: in the Market fountain Link's skin, hair and belt came out white (hs2864), and normal without them (hs2865).
        // Standing in the Market and Kokiri Forest without them, no difference beyond the anti-aliasing's jitter (hs2866 /
        // hs2867). Left out until what they compute is translated right; RX_VK_FOG_BOXES=1 runs them as before
        static const bool fog_boxes = getenv("RX_VK_FOG_BOXES") && atoi(getenv("RX_VK_FOG_BOXES")) != 0;
        if (!fog_boxes && (cp->key == 0x2b37a33d189014a8ull || cp->key == 0x0948a49a3c3068a8ull)) {
            static std::atomic<int> told{0};
            if (told++ < 2)
                logf("vk: draws of program %016llx left out: a GPU hang in water on RTX 50 cards, Link white in water elsewhere "
                     "(RX_VK_FOG_BOXES=1 runs them)", (unsigned long long)cp->key);
            return;
        }
        // (the list parsed once into key prefixes: it was a snprintf and a string compare on every draw)
        static const std::vector<std::pair<uint64_t, uint64_t>> skip_keys = [] {   // (value, mask) of each prefix
            std::vector<std::pair<uint64_t, uint64_t>> v;
            const char* list = getenv("RX_VK_SKIP_PROG") && strcmp(getenv("RX_VK_SKIP_PROG"), "-") ? getenv("RX_VK_SKIP_PROG") : "";
            for (const char* q = list; *q;) {
                const char* e = q;
                while (*e && *e != ',') e++;
                const size_t n = std::min<size_t>((size_t)(e - q), 16);
                if (n) {
                    const uint64_t value = strtoull(std::string(q, n).c_str(), nullptr, 16) << (4 * (16 - n));
                    const uint64_t mask = n == 16 ? ~0ull : ~(~0ull >> (4 * n));
                    v.push_back({value, mask});
                }
                q = *e ? e + 1 : e;
            }
            return v;
        }();
        for (const auto& [value, mask] : skip_keys)
            if ((cp->key & mask) == value) {
                static std::atomic<int> told{0};
                if (told++ < 8) logf("vk: draws of program %016llx left out (RX_VK_SKIP_PROG); its code:%s", (unsigned long long)cp->key,
                                     program_code_keys(s).c_str());
                return;
            }
        // (hookshot, a diagnosis) RX_VK_SKIP_3D_DRAWS=1: every draw whose program samples a 3D texture left out.
        // The RTX 50 hangs moved from program to program as programs were left out, always to another draw that reads the
        // volumes; this leaves them all out at once (much of the lighting goes with them: only a test)
        static const bool skip_3d = getenv("RX_VK_SKIP_3D_DRAWS") && atoi(getenv("RX_VK_SKIP_3D_DRAWS")) != 0;
        if (skip_3d) {
            bool reads_3d = false;
            for (const Stage& st : cp->stages)
                for (const auto& td : st.info.texture_descriptors) reads_3d |= td.type == Shader::TextureType::Color3D;
            if (reads_3d) {
                static std::atomic<int> told{0};
                if (told++ < 8) logf("vk: draws of program %016llx left out (RX_VK_SKIP_3D_DRAWS: it samples a 3D texture)",
                                     (unsigned long long)cp->key);
                return;
            }
        }
        // (hookshot, a workaround) RX_VK_SKIP_HALF_PASS=1: the draws of the game's half-resolution HDR pass that
        // sample a 3D texture left out. Every RTX 50 report made with one recording thread put the hung draw in that pass (an
        // R11G11B10F target with depth, half the size of the scene's: 768x432 at 1536x864, 960x540 at 1920x1080): Link's body
        // under the water surface and the two volume boxes already left out. The scene's size is the widest R11G11B10F target
        // with depth drawn this frame or the one before (the render scale and the upscalers change it)
        static const bool skip_half = getenv("RX_VK_SKIP_HALF_PASS") && atoi(getenv("RX_VK_SKIP_HALF_PASS")) != 0;
        if (skip_half && at.n == 1 && at.depth && at.color_format[0] == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
            static std::atomic<uint64_t> w_frame{0};
            static std::atomic<uint32_t> w_now{0}, w_before{0};
            const uint64_t f = texture_frame_no();
            if (w_frame.load(std::memory_order_relaxed) != f && w_frame.exchange(f) != f)
                w_before.store(w_now.exchange(0));
            uint32_t seen = w_now.load(std::memory_order_relaxed);
            while (at.width > seen && !w_now.compare_exchange_weak(seen, at.width)) {}
            const uint32_t scene_w = std::max(w_now.load(std::memory_order_relaxed), w_before.load(std::memory_order_relaxed));
            bool reads_3d = false;
            for (const Stage& st : cp->stages)
                for (const auto& td : st.info.texture_descriptors) reads_3d |= td.type == Shader::TextureType::Color3D;
            if (reads_3d && at.width * 2 <= scene_w + 2 && at.width * 3 >= scene_w) {   // (half the scene's width, give or take
                static std::atomic<int> told{0};                                            // a texel of rounding)
                if (told++ < 8)
                    logf("vk: draws of program %016llx left out (RX_VK_SKIP_HALF_PASS: the %ux%u pass, the scene %u wide)",
                         (unsigned long long)cp->key, at.width, at.height, scene_w);
                return;
            }
        }
    }
    {   // (hookshot, RX_VK_SKIP_CODE=<code key prefix,...>: the draws of every program with a stage of that native code left out,
        // decided once a program. RX_VK_SKIP_PROG names whole programs; a shared stage is in many. On the Steam Deck (RADV VANGOGH)
        // the GPU runs for seconds in the task/mesh programs that share the task shader 2c56dc1006942f74, a nested loop over a 2D
        // grid (a culling pass), until the kernel resets it at 10 s, in the Great Deku Tree and Kokiri Forest (issue 817). About
        // 30 programs use it; RX_VK_SKIP_CODE=2c56dc1006942f74 leaves them out and the Deck plays without the hang, but without
        // the models they draw too: a workaround, opt-in)
        static const char* skip_code_list = getenv("RX_VK_SKIP_CODE");
        if (skip_code_list && *skip_code_list) {
            static std::mutex memo_mu;
            static std::unordered_map<uint64_t, bool> memo;
            bool skip;
            {
                std::lock_guard lk(memo_mu);
                const auto it = memo.find(cp->key);
                if (it != memo.end()) {
                    skip = it->second;
                } else {
                    const std::string codes = program_code_keys(s);
                    skip = skip_code::matches(codes, skip_code_list);
                    memo.emplace(cp->key, skip);
                    static int told = 0;
                    if (skip && told++ < 32)
                        logf("vk: draws of program %016llx left out (RX_VK_SKIP_CODE); its code:%s", (unsigned long long)cp->key,
                             codes.c_str());
                }
            }
            if (skip) return;
        }
    }
    // Wait for translation before reading its stages. A pending pipeline may be left out only when no stage writes
    // global memory or storage buffers/images and its targets hold an earlier frame's picture. RX_VK_PIPE_SKIP=0:
    // pipelines are always waited for too.
    // Waiting on unknown translations preserves writes but can restore first-visit pauses (previously 0.2-3 s).
    // RX_VK_PIPE_SKIP=2 restores the old translation skip for persistent targets; after translation, memory-writing pipelines wait.
    static const bool always_wait = getenv("RX_VK_PIPE_SKIP") && atoi(getenv("RX_VK_PIPE_SKIP")) == 0;
    static const bool legacy_translation_skip = getenv("RX_VK_PIPE_SKIP") && atoi(getenv("RX_VK_PIPE_SKIP")) == 2;
    // (hookshot, 11-17: and only where the next frame draws it again, targets_redrawn: every target drawn in the frame
    // before and the draw's viewport and scissor over nine tenths of it. A piece the game draws once into a target it
    // keeps (a tile, a decal, a cached shadow), left out, was never drawn again: the striped ground until the area was
    // loaded again. Those wait for their program or pipeline now. RX_VK_SKIP_PARTS=1: any target holding an earlier
    // frame's picture, as before)
    static const bool skip_parts = env_on("RX_VK_SKIP_PARTS");
    const bool skippable = targets_persistent(at) && (skip_parts || targets_redrawn(at, s));
    auto wait_for = [&](const std::atomic<bool>& ready) {
        PROF_EXACT();
        const auto t0 = std::chrono::steady_clock::now();
        const int queued = jobs_pending();
        const bool here = jobs_wait(ready);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        st_waits++;
        st_wait_ms += ms;
        static std::atomic<int> told{0};
        if (here && told++ < 10)   // (hookshot: that the steal ran, with the queue the wait would have stood behind)
            logf("vk: a draw's %s ran where it waited, ahead of the queue (%d other jobs queued or running), in %.1f ms",
                 &ready == &cp->ready ? "shader translation" : "pipeline build", queued > 0 ? queued - 1 : 0, ms);
    };
    // (hookshot) RX_VK_TRANSLATE_WAIT_MS=<ms> (off unless set): a translation not ready, for a draw whose targets hold an
    // earlier frame's picture, is waited for at most this long a frame in all, on each recording thread, without taking the
    // job onto this thread, and the draw is then left out of the frame. Waiting it out (#1573, the default) can bring back
    // the 0.2-3 s pauses at an area's first new programs on a cold program bank; leaving it out at once (RX_VK_PIPE_SKIP=2)
    // lost the writes of translations that finish in a moment. With the bound, a translation slower than it can still lose a
    // memory write (its writes are not known before it is translated): so it is opt-in until it has been tested on more
    // machines. A target made this frame still waits it out. A value that is not a number is ignored with a warning.
    static const double translate_wait_ms = [] {
        double ms = translate_wait::kOff;
        const char* e = getenv("RX_VK_TRANSLATE_WAIT_MS");
        if (e && !translate_wait::parse(e, ms)) {
            ms = translate_wait::kOff;
            logf("vk: RX_VK_TRANSLATE_WAIT_MS=%s is not a number of milliseconds: ignored", e);
        }
        if (e && ms >= 0)
            logf("vk: a shader still translating is waited for at most %.1f ms a frame on each recording thread, then its draws "
                 "on earlier frames' targets are left out (RX_VK_TRANSLATE_WAIT_MS)", ms);
        else if (e)
            logf("vk: shaders still translating are waited out (RX_VK_TRANSLATE_WAIT_MS off)");
        return ms;
    }();
    if (!cp->ready.load(std::memory_order_acquire)) {
        if (legacy_translation_skip && skippable) return use.skipped(), void(st_pending_skips++);
        if (!always_wait && translate_wait_ms >= 0 && skippable) {
            static thread_local uint64_t budget_frame = ~0ull;
            static thread_local double budget_left = 0;
            const uint64_t frame = texture_frame_no();
            if (frame != budget_frame) budget_frame = frame, budget_left = translate_wait_ms;
            bool ready;
            if (budget_left > 0) {   // (counted as a wait only when it can wait: the stat #1573 was judged on stays honest)
                PROF_EXACT();
                const auto t0 = std::chrono::steady_clock::now();
                ready = jobs_wait_for(cp->ready, budget_left);
                const double took = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                budget_left -= took;
                st_waits++;
                st_wait_ms += took;
            } else {
                ready = cp->ready.load(std::memory_order_acquire);
            }
            if (!ready) return use.skipped(), void(st_pending_skips++);
        } else {
        wait_for(cp->ready);
        }
    }
    // (hookshot) a cached program's stand-in, its records now read (vk_shader.cpp): the pick again finds the variant read, or
    // starts its translation as before
    if (cp->failed && program_is_stand_in(cp)) {
        {
            ParLock lk(d.mesh);
            cp = program_for(s, d.mesh ? ProgramKind::Mesh : ProgramKind::Graphics, types);
        }
        if (!cp) SKIP_HERE();
        if (!cp->ready.load(std::memory_order_acquire)) {   // (waited for as the first pick's: Hookshot's rules above)
            if (legacy_translation_skip && skippable) return use.skipped(), void(st_pending_skips++);
            wait_for(cp->ready);
        }
    }
    if (exposure_log_on() && cp) exposure_note(s, *cp);
    if (cp->failed) SKIP_HERE();
    Program& p = const_cast<Program&>(*cp);
    if (p.fallback_task && !d.indirect) {
        const int batch_size = (int)fallback_task_groups(p);
        if (!batch_size) SKIP_HERE();
        if (d.groups[0] > batch_size) {
            DrawArgs batch = d;
            // (hookshot) one draw of the game's in several: one query cannot be begun in each (and a count of the last batch
            // alone could be 0 while an earlier one drew). Not counted: its report stays "visible"
            if (batch.occ_pool) OccGuard(cmd, batch).drop(kOccTaskBatches), batch.occ_pool = VK_NULL_HANDLE;
            for (int remaining = d.groups[0]; remaining > 0; remaining -= batch.groups[0]) {
                batch.groups[0] = std::min(remaining, batch_size);
                draw(cmd, s, batch);
                batch.first_task += (uint32_t)batch.groups[0];
            }
            return;
        }
    }
    topo = pipeline_topology(p, topo);   // (the mesh fallback's: its mesh program's points, lines or triangles)
    PipeEntry* pe;
    {
        FINE_SCOPE(FT_PIPE_LOOKUP);
        PROF_HOT(PROF_PIPE);
        if (!p.layout) {
            PROF_EXACT();
            ParLock lk;
            ensure_layout(p);
        }
        const int64_t pipes_new = use.on ? use_counters.pipes_new : 0;   // (another file's thread-local: read for the life log only)
        pe = graphics_pipeline(p, s, at, topo, types);
        if (use.on) use.pipeline(pe->ready.load(std::memory_order_acquire), use_counters.pipes_new != pipes_new);
    }
    if (!pe->ready.load(std::memory_order_acquire)) {
        const bool may_skip = !always_wait && skippable && !writes_memory(p);
        // (hookshot, from Mako) one to draw with until the full build is done: from the caches, or quick (quick_pipeline). Else a
        // draw that may be left out first waits briefly for its build, moved to the queue's front (RX_VK_PIPE_WAIT_MS, off unless
        // set), and one that may not waits for it, running it here if no worker has (jobs_wait). RX_VK_PIPE_SKIP=0 waits for
        // every full pipeline, as it says: no quick one
        if (always_wait) {
            wait_for(pe->ready);
        } else if (!pe->pipeline.load(std::memory_order_acquire) && !quick_pipeline(*pe, p, s, at, topo, !may_skip)) {
            if (may_skip) {
                static thread_local quick_pipe::FrameBudget wait_budget;
                const auto now_ms = [] {
                    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
                };
                double waited = -1;
                const bool ready = quick_pipe::brief(
                    pe->ready, pipe_wait_budget_ms(), wait_budget, texture_frame_no(), [&] { jobs_hurry(pe->ready); },
                    [&](double ms) {
                        PROF_EXACT();
                        const double t0 = now_ms();
                        const bool ok = jobs_wait_for(pe->ready, ms);
                        waited = now_ms() - t0;
                        return ok;
                    },
                    now_ms);
                if (waited >= 0) st_waits++, st_wait_ms += waited;
                if (!ready) return use.skipped(), void(st_pending_skips++);
            } else {
                wait_for(pe->ready);
            }
        }
    }
    if (use.on) use.pipeline_made(pe->how.load(std::memory_order_relaxed));
    const VkPipeline pl = pe->pipeline;
    if (!pl) SKIP_HERE();
    dlss_trace_draw(s, p, d, at);
    SkinnedDraw skinned;
    skin_check(s, p, d, at, skinned);
    // (hookshot) a hidden mesh's depth draw is left out here: before actor motion takes it, before its descriptors
    if (hidden_depth_draw(p, s, d, at, skinned.depth != nullptr)) SKIP_HERE();
    if (skinned.depth) temporal_note_skinned_draw(skinned);
    // (after OoT's tree-shadow pass has drawn its mask: the mask evened out over frames, at this draw's end: mask_stabilize)
    struct MaskGuard {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        Texture* tex = nullptr;
        uint32_t level = 0, layer = 0;
        float w = 0;
        ~MaskGuard() { if (cmd) mask_stabilize(cmd, tex, level, layer, w); }
    } mask_guard;
    if (p.key == 0x3427a1dbefd5e02cull && at.n > 0 && at.color[0])
        if (const float mw = mask_weight(); mw > 0) {
            // Assign fields without a temporary whose destructor would dispatch early.
            mask_guard.cmd = cmd; mask_guard.tex = at.color[0];
            mask_guard.level = at.color_level[0]; mask_guard.layer = at.color_layer[0]; mask_guard.w = mw;
        }
    if (!record_progs().empty() && std::find(record_progs().begin(), record_progs().end(), p.key) != record_progs().end())
        record_draw(s, p, d, at);
    if (occ_proxy_skip_on() && !dump_on() && !d.mesh && !(program_inert(p) && draw_writes_nothing(p, s, at))) {
        // (diagnosis: the first 24 programs a frame draws most whose draws F2 keeps, and why, logged once each)
        // (each thread's own count, no lock: the log is once a program on the thread that reaches 2000 first, 24 at most)
        thread_local std::unordered_map<uint64_t, int64_t> seen;
        if (seen.size() < 4096 && ++seen[p.key] == 2000) {
            static std::atomic<int> told{0};
            if (told.fetch_add(1) < 24) {
                int sb = 0, img = 0, tex = 0, tb = 0, gm = 0, fs = 0;
                for (const Stage& st : p.stages) {
                    sb += (int)st.info.storage_buffers_descriptors.size(), img += (int)(st.info.image_descriptors.size() + st.info.image_buffer_descriptors.size());
                    tex += (int)st.info.texture_descriptors.size(), tb += (int)st.info.texture_buffer_descriptors.size();
                    gm += st.info.stores_global_memory || st.info.uses_global_memory;
                    fs += st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT;
                }
                uint32_t wm = 0;
                for (int i = 0; i < at.n; i++) wm |= (at.color[i] ? target_write_mask(p, s, i) : 0u) << (4 * i);
                logf("vk: F2 keeps the draws of program %016llx (2000 so far): storage %d, images %d, textures %d, texel buffers %d, "
                     "global memory %d, fragment stage %d; colour masks %#x, depth %d write %d, stencil test %d", (unsigned long long)p.key,
                     sb, img, tex, tb, gm, fs, wm, at.depth ? 1 : 0, (int)s.ds.dw, (int)(at.depth && has_stencil(at.depth_format) && s.ds.st));
            }
        }
    }
    if (!d.mesh && occ_proxy_skip_on() && !dump_on() && program_inert(p) && draw_writes_nothing(p, s, at)) {   //
        {
            // (draw_hazards is not called: for this draw it does nothing. Its writes are 0 by draw_writes_nothing, and what it
            // samples (gfx_sampled) is empty in the old path: its program has no texture, so its own descriptor build set it
            // empty, or it reused the descriptors of the same program, whose build set it empty. With no writes and nothing
            // sampled draw_hazards neither logs nor ends the pass. Here gfx_sampled still holds the last recorded draw's, which is
            // what a later draw reusing that draw's descriptors needs, so it is neither read nor changed)
            FINE_SCOPE(FT_HAZARD);
            begin_pass(cmd, at);
        }
        ps.prog = p.key;
        if (!ps.first_prog) ps.first_prog = p.key;
        st_draws++, ps.draws++;
        if (const int64_t k = st_occ_proxy_skips.fetch_add(1, std::memory_order_relaxed) + 1; k % 200000 == 0)
            logf("vk: draws left out, writing and sampling nothing (RX_VK_OCC_PROXY_SKIP): %lld so far, frame %llu", (long long)k,
                 (unsigned long long)texture_frame_no());
        return;
    }
    // (RX_VK_DUMP_EACH=<key>,...: in a dump, the pass ends after each draw of those programs, so every one of them is
    // read back on its own)
    struct EachGuard {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        ~EachGuard() { if (cmd) end_pass(cmd); }
    } each_guard;
    {
        static const std::vector<uint64_t> each = [] {
            std::vector<uint64_t> v;
            if (const char* e = getenv("RX_VK_DUMP_EACH"))
                for (const char* q = e; *q;) {
                    char* end;
                    const uint64_t k = strtoull(q, &end, 16);
                    if (end == q) break;
                    v.push_back(k);
                    q = *end ? end + 1 : end;
                }
            return v;
        }();
        // (a key of 1: every draw into a target at least 2048 wide; 2,<width in hex>: every draw into targets that wide)
        if (!each.empty() && dump_on() &&
            (std::find(each.begin(), each.end(), p.key) != each.end() ||
             (each[0] == 1 && at.width >= 2048) || (each[0] == 2 && each.size() > 1 && at.width == each[1])))
            each_guard.cmd = cmd;
    }
    if (sw().log_draws) {
        static int logged;
        if (logged++ < 40)
            logf("vk: draw: program %016llx, %d stages, %d targets %ux%u, colour 0 %#llx format %d, topology %d, count %d",
                 (unsigned long long)p.key, (int)p.stages.size(), at.n, at.width, at.height,
                 (unsigned long long)(at.color[0] ? at.color[0]->gpu : 0), (int)at.color_format[0], (int)topo, d.count);
    }
    // (the descriptors depend on the program, the bindings, the texture slots and the stream's uniform updates:
    // when none changed since the last draw, the set it pushed or bound stands)
    // (hookshot) and no texture went since (texture_destroy_generation: an image evicted, or made again at another size or format,
    // in the same recording): the set it pushed or bound named that texture's old views, which go once the GPU is past the texture's
    // last use, a use the reused set no longer counted. The draws after it sampled freed memory, whatever the driver put there next:
    // plants and fences glowing white on a Legion Go S, which RX_VK_DESC_REUSE=never took away (issue 1126)
    const uint64_t reg_gen = registrations_generation(), ub_gen = ub_generation();
    static const bool no_reuse = desc_reuse_level() == 0;   // (RX_VK_DESC_REUSE=never: every draw builds its own)
    const uint64_t dgen = texture_destroy_generation();
    // (hookshot, issue 640: a mesh fallback draw also pushes the writes at the compute bind point, for its task and mesh
    // dispatches, so they must be its own. A dispatch since its last draw built the compute program's over them: pushed
    // with the fallback's layout, its task and mesh stages read another program's buffers, and descriptors the push left
    // as they were: stale addresses, like the one RADV faulted on in Hyrule Field (#792))
    const bool slotgen = desc_slotgen_on();
    // (RX_VK_READBACK_EARLY, once a read of pages written in place has waited: a program that writes a storage buffer builds its
    // descriptors at each draw, so each write is noted
    // with the part it is recorded in (vk_memory.cpp early_in_place_write); a draw that kept the set before noted nothing, and a
    // read could have gone on after an earlier writer's part)
    static const bool early = readback_early_on();
    const bool early_rebind = early && memory_readback_early_active() && [&] {
        for (const Stage& st : p.stages)
            for (const auto& d : st.info.storage_buffers_descriptors)
                if (d.is_written) return true;
        return false;
    }();
    const bool same_desc = !no_reuse && !early_rebind && cb.desc.valid && cb.desc.prog == &p && cb.desc.bind == s.bind_gen &&
                           cb.desc.ub == ub_gen && cb.desc.dgen == dgen &&
                           (slotgen ? cb.desc.fp_on && desc_slots_kept(cb.desc, reg_gen) : cb.desc.reg == reg_gen) &&
                           (!p.fallback || dw().prog == &p);
    if (p.fallback && !same_desc && dw().prog != &p && cb.desc.valid && cb.desc.prog == &p) {
        static std::atomic<int> told{0};
        if (told++ < 3) logf("vk: fallback: %016llx's descriptors built again after a dispatch", (unsigned long long)p.key);
    }
    if (same_desc && slotgen && cb.desc.reg != reg_gen) st_desc_slotgen_saves++;   // (the global check would have rebuilt)
    fine_prep.finish();
    if (!same_desc) {
        FINE_SCOPE(FT_DESC_TOTAL);
        PROF_HOT(PROF_DESC);
        FINE_N(FC_BUILDS, 1);
        trace_tex_hit = nullptr, trace_tex_mask = 0;
        build_descriptors(s, p);   // (textures first used here are uploaded by the recording's upload command buffer)
        FINE_N(FC_WRITES, (int64_t)dw().writes.size());
        static std::string last_logged;
        if (dump_on()) {
            char head[256];
            int hn = snprintf(head, sizeof head, "prog %016llx %ux%u:", (unsigned long long)p.key, at.width, at.height);
            for (int i = 0; i < at.n && hn < 200; i++)   // (the targets' NVN formats and the views' Vulkan ones)
                if (at.color[i])
                    hn += snprintf(head + hn, sizeof head - hn, " c%d:#%u/n%#x/v%d", i, at.color[i]->id,
                                   s.rt.has_cview[i] && s.rt.cview[i].set_format ? (int)s.rt.cview[i].format
                                                                                 : (int)s.rt.color[i]->d.format,
                                   (int)at.color_format[i]);
            if (last_logged != head + dump_inputs())
                last_logged = head + dump_inputs(), logf("vk: dump-in %03d draw %s", dump_seq(), last_logged.c_str());
        }
    }
    if (dump_on()) log_dump_draw(s, p, d);
    PROF_HOT(PROF_CMDS);
    FINE_SCOPE(FT_CMD_TOTAL);
    texture_watch_prog(p.key);
    if (!same_desc) gfx_sampled() = draw_sampled();
    // (hookshot) NVIDIA DLAA before the game's bloom and tone map: the frame's draws into the scene and those that read it, and
    // at the bloom's first draw DLAA on the scene, before this draw reads it (vk_temporal.cpp, pre_bloom.h). Its passes and
    // its actor motion's draws bound a pipeline, a viewport, vertex streams and descriptors of their own: this draw sets
    // all of its again, as after the network's stand-in (temporal_stand_in). Kept, a draw of the scene's size (the tone
    // map, where DLAA ran before it) took the actor motion's upright viewport for the game's upside-down one (dynamic_state)
    // and drew the 3D upside down
    const bool pre_recorded = temporal_pre_bloom(cmd, p.key, gfx_sampled(), at.color, at.n);
    if (pre_recorded) cb.reset();
    static const bool barrier_every = env_on("RX_VK_BARRIER_EVERY_DRAW");   // (an experiment: a pass each)
    if (barrier_every) end_pass(cmd);
    reactive_scene_draw(cmd, p, s, at);
    uint16_t writes, samples;
    {
        FINE_SCOPE(FT_HAZARD);
        writes = draw_hazards(cmd, p, s, at, samples);
    }
    FineScope fine_prepass(FT_CMD_PREPASS);
    BufferRef mesh_args{};
    uint32_t mesh_args_at = 0;
    // (RX_VK_MESH_ARGS=1: rewritten {count, 1, 1, first}. Off: it did not bring the canopies back, and it ends the pass at each draw)
    // (amd-fix, 11-07: on by default with the firsts (RX_VK_MESH_FIRST), the canopies back; RX_VK_MESH_ARGS=0: off. A single indirect
    // draw has no stride of its own: its entry is NVN's 16 bytes, taken as 8 the first was never copied)
    static const bool mesh_rewrite = getenv("RX_VK_MESH_ARGS") && atoi(getenv("RX_VK_MESH_ARGS")) != 0;   // (sharkiix version: off, as upstream)
    if (mesh_rewrite && d.mesh && d.indirect) {   // (out of the pass: a copy cannot be made in one)
        const uint32_t n = (uint32_t)std::clamp(d.max, 1, (int)(kMeshArgs / 2)), stride = d.stride ? (uint32_t)d.stride : 16u;
        BufferRef src = gpu_buffer_bind(d.indirect, (uint64_t)n * stride, false);
        if (!src) SKIP_HERE();
        if (mesh_args_ready()) {
            static const uint32_t word = getenv("RX_VK_MESH_COUNT_WORD") ? (uint32_t)atoi(getenv("RX_VK_MESH_COUNT_WORD")) : 0;
            end_pass(cmd);
            full_barrier(cmd);
            static thread_local std::vector<VkBufferCopy> copies;
            copies.clear();
            mesh_args_at = mesh_args_take(n);
            static const bool show = env_on("RX_VK_LOG_MESH_INDIRECT");
            if (show && mesh_args_at > 4096 && (mesh_args_at & 4095) < n) {   // (the counts the GPU copied, of the last 4096)
                uint32_t nonzero = 0, most = 0;
                uint64_t sum = 0;
                for (uint32_t i = mesh_args_at - 4096; i < mesh_args_at; i++) {
                    const uint32_t v = mesh_args_words[i * 4];
                    nonzero += v != 0, most = std::max(most, v), sum += v;
                }
                logf("vk: mesh counts: %u of 4096 not zero, largest %u, sum %llu", nonzero, most, (unsigned long long)sum);
            }
            {   // (amd-fix, RX_VK_LOG_MESH_PERPROG=1, diagnosis: per mesh program, the counts and firsts the GPU wrote, 4096 to 8192 entries ago)
                static const bool per_prog = env_on("RX_VK_LOG_MESH_PERPROG");
                if (per_prog) {
                    static std::vector<uint64_t> prog_at(kMeshArgs);
                    static uint32_t last_window = 0;
                    static int windows_told = 0;
                    const uint32_t window = mesh_args_at / 4096;
                    if (window != last_window && mesh_args_at >= 8192 && windows_told < 40) {
                        last_window = window;
                        windows_told++;
                        struct Agg { uint32_t entries = 0, nonzero = 0, most = 0, first_most = 0; uint64_t sum = 0; };
                        std::map<uint64_t, Agg> agg;
                        for (uint32_t i = mesh_args_at - 8192; i < mesh_args_at - 4096; i++) {
                            Agg& a = agg[prog_at[i]];
                            const uint32_t v = mesh_args_words[i * 4];
                            a.entries++, a.nonzero += v != 0, a.most = std::max(a.most, v), a.sum += v;
                            a.first_most = std::max(a.first_most, mesh_args_words[i * 4 + 3]);
                        }
                        for (const auto& [k, a] : agg)
                            logf("vk: mesh perprog window %u prog %016llx: %u entries, %u not zero, largest %u, sum %llu, largest first %u", window,
                                 (unsigned long long)k, a.entries, a.nonzero, a.most, (unsigned long long)a.sum, a.first_most);
                    }
                    for (uint32_t i = 0; i < n; i++) prog_at[(mesh_args_at + i) % kMeshArgs] = p.key;
                }
            }
            for (uint32_t i = 0; i < n; i++) {
                copies.push_back({src.offset + i * stride + word * 4, (VkDeviceSize)(mesh_args_at + i) * 16, 4});
                // (the first: NVN2's entry is {count, first}. Word 3 is the culling pass's own (an atomic max, read back as "70 0 0 70",
                // "60 0 0 950"): taken as the first, it sent the draws past the records. RX_VK_MESH_FIRST_WORD=<n>)
                static const uint32_t first_word = getenv("RX_VK_MESH_FIRST_WORD") ? (uint32_t)atoi(getenv("RX_VK_MESH_FIRST_WORD")) : 1u;
                if (stride >= 8) copies.push_back({src.offset + i * stride + first_word * 4, (VkDeviceSize)(mesh_args_at + i) * 16 + 12, 4});
            }
            vkCmdCopyBuffer(cmd, src.buffer, mesh_args_buffer, (uint32_t)copies.size(), copies.data());
            full_barrier(cmd);
            mesh_args = {mesh_args_buffer, (VkDeviceSize)mesh_args_at * 16, (VkDeviceSize)n * 16, 0};
        }
    }
    const FallbackRole fallback_role = p.fallback && d.mesh ? fallback_role_of(p, s, at, d) : FallbackRole::None;
    if (fallback_role == FallbackRole::DepthOnly && fallback_list_held(d.indirect)) SKIP_HERE();
    if (p.fallback && d.mesh && !fallback_record(cmd, p, d)) {
        if (fallback_role == FallbackRole::Colour && p.compute_state.load(std::memory_order_acquire) == 1)
            fallback_list_hold(d.indirect, true);
        SKIP_HERE();
    }
    if (fallback_role == FallbackRole::Colour) fallback_list_hold(d.indirect, false);
    fine_prepass.finish();
    {
        FINE_SCOPE(FT_HAZARD);
        begin_pass(cmd, at);
    }
    probe_draw(s, p, ps.draws);   // (hookshot: RX_VK_PROBE)
    gprof_run(cmd, p.key);   // (RX_VK_PASS_TIMING=2)
    FineScope fine_track(FT_CMD_TRACK);
    ps.prog = p.key;
    if (!ps.first_prog) ps.first_prog = p.key;
    if (pass_trace_on() && ps.sampled.size() < 6) {
        for (const Texture* t : gfx_sampled()) {
            if (!t || std::find(ps.sampled.begin(), ps.sampled.end(), t) != ps.sampled.end()) continue;
            ps.sampled.push_back(t);
            if (ps.sampled.size() == 6) break;
        }
    }
    ps.sampled_targets |= samples;
    ps.written |= writes;
    fine_track.finish();
    if (pl != cb.pipeline) {
        FINE_SCOPE(FT_BINDPL);
        FINE_N(FC_BIND_PIPE, 1);
        if (res_log_on()) res_note_pipeline_use(pl);   //
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl), cb.pipeline = pl;
        if (p.fallback) {
            // A static-stride pipeline invalidates the dynamic strides, even with no vertex bindings.
            // Drop both bind-skip caches; the resolved buffer references themselves are still reusable.
            cb.vb_valid = false;
            cb.vertex_set.on = false;
        }
    }
    if (!same_desc || pre_recorded) {   // (hookshot: after DLAA's commands, its pushes are gone: temporal_pre_bloom)
        FINE_SCOPE(FT_BINDDESC);
        if (!bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p)) SKIP_HERE();
        cb.desc.valid = true, cb.desc.prog = &p, cb.desc.bind = s.bind_gen, cb.desc.reg = reg_gen;
        cb.desc.ub = ub_gen, cb.desc.dgen = dgen;
        cb.desc.fp_on = desc_fp_on;
        if (desc_fp_on)   // (Descriptor reuse: what its handles resolved; the next such draw may keep it through other slots' churn)
            cb.desc.fp = desc_fp(), cb.desc.fp_dummy = desc_fp_dummy, cb.desc.samp = desc_samp_seen,
            cb.desc.destroy = desc_destroy_seen;
    } else {
        st_desc_reused++;
    }
    {
        FINE_SCOPE(FT_PUSHC);
        push_constants(cmd, p, &at);
    }
    if (watched(p.key)) {   // (RX_VK_WATCH_PROG: the draw's state, 80 lines)
        static std::atomic<int> n{0};
        if (n++ < 80) {
            char line[600];
            int k = snprintf(line, sizeof line, "vk: watch draw %016llx %ux%u targets %d: blend_en %#x logic %u wm", (unsigned long long)p.key,
                             at.width, at.height, at.n, (unsigned)s.color.blend_en, (unsigned)s.color.logic_op);
            for (int i = 0; i < at.n && k < 500; i++) {
                const nvn_blend_st& b = s.blend[i];
                k += snprintf(line + k, sizeof line - k, " [%d: wm %x src %#x dst %#x eq %u / a %#x %#x %u fmt %d]", i,
                              (unsigned)(s.chmask.mask >> (4 * i) & 15), (unsigned)b.src_rgb, (unsigned)b.dst_rgb, (unsigned)b.eq_rgb,
                              (unsigned)b.src_a, (unsigned)b.dst_a, (unsigned)b.eq_a, (int)at.color_format[i]);
            }
            snprintf(line + k, sizeof line - k, " blendc %.2f %.2f %.2f %.2f depth %d", s.blend_color[0], s.blend_color[1],
                     s.blend_color[2], s.blend_color[3], at.depth ? 1 : 0);
            logf("%s", line);
        }
    }
    {
        FINE_SCOPE(FT_DYN);
        dynamic_state(cmd, s, at, topo, d.mesh && !p.fallback, p);
    }
    FineScope fine_poststate(FT_CMD_POSTSTATE);
    if (trace_tex_hit && texture_frame_no() % 30 == 0) {
        static std::atomic<int> told{0};
        if (told++ < 600) {
            char ds[80] = "", poly[40] = "";
            int k = 0;
            for (size_t i = 0; i < sizeof s.ds && k < 70; i++) k += snprintf(ds + k, sizeof ds - k, "%02x", ((const uint8_t*)&s.ds)[i]);
            k = 0;
            for (size_t i = 0; i < sizeof s.poly && k < 30; i++) k += snprintf(poly + k, sizeof poly - k, "%02x", ((const uint8_t*)&s.poly)[i]);
            logf("vk: trace frame %llu tex #%u %ux%u prog %016llx target %ux%u x%d depth %d fmt0 %d wm %x blend_en %x topo %d count %d inst %d first %d base %d "
                 "%s%s%s groups %d ds %s poly %s vp0 %.0f %.0f %.0f %.0f sc %d %d %d %d cull-ish vs0 %#llx",
                 (unsigned long long)texture_frame_no(), trace_tex_hit->id, trace_tex_hit->width, trace_tex_hit->height, (unsigned long long)p.key, at.width,
                 at.height, at.n, at.depth ? 1 : 0, (int)at.color_format[0], (unsigned)s.chmask.mask, (unsigned)s.color.blend_en, (int)topo, d.count,
                 d.instances, d.first, d.base_vertex, d.indexed ? "indexed " : "", d.indirect ? "indirect " : "", d.mesh ? "mesh " : "", d.groups[0], ds, poly,
                 s.viewport[0][0], s.viewport[0][1], s.viewport[0][2], s.viewport[0][3], s.scissor[0], s.scissor[1], s.scissor[2], s.scissor[3],
                 (unsigned long long)s.vbuf[0].address);
            if (d.mesh) {   // (what the task and mesh stages have bound, and the first words of the small tables: pointers)
                for (int st : {6, 7})
                    for (int i = 0; i < 4; i++) {
                        const nvn_p_bind_buf& b = s.ssbo[st][i];
                        if (!b.addr) continue;
                        uint64_t w[10]{};
                        memory_read_as_submitted(b.addr, w, std::min<uint64_t>(sizeof w, b.size));
                        logf("vk: trace   stage %d ssbo %d %#llx size %#llx: %llx %llx %llx %llx %llx %llx %llx %llx", st, i, (unsigned long long)b.addr,
                             (unsigned long long)b.size, (unsigned long long)w[0], (unsigned long long)w[1], (unsigned long long)w[2], (unsigned long long)w[3],
                             (unsigned long long)w[4], (unsigned long long)w[5], (unsigned long long)w[6], (unsigned long long)w[7]);
                    }
            }
        }
    }
    if (d.indexed && s.restart) {   // (RX_VK_LOG_RESTART=1: the restarts draws ask for, by topology, index type and index)
        static const bool log_restart = env_on("RX_VK_LOG_RESTART");
        if (log_restart) {
            static std::mutex m;
            static std::map<uint64_t, int64_t> seen;
            const uint64_t key = (uint64_t)topo << 40 | (uint64_t)d.itype << 32 | s.restart_index;
            std::lock_guard l(m);
            const int64_t n = ++seen[key];
            if ((n == 1 && seen.size() <= 60) || n == 1000 || n == 100000)
                logf("vk: restart (%lld draws so far): prog %016llx topology %d index type %d restart index %#x%s count %d",
                     (long long)n, (unsigned long long)p.key, (int)topo, d.itype, s.restart_index,
                     d.indirect ? " indirect" : "", d.count);
        }
    }
    StatsQuery stats_query(cmd, p, d);   // (EvilOOT, RX_VK_DRAW_STATS)
    OccGuard occ(cmd, d);   // (hookshot: after stats_query, so it ends first, inside the pass; begun by occ.issue())
    OccQuery occ_query(cmd, p.key, d, stats_query.cmd != VK_NULL_HANDLE);   // (RX_VK_OCC_REAL)
    if (stats_frame_now() && stats_prog_wanted(p.key)) {   // (and the wanted programs' buffers: a mesh draw's stages, fragment)
        for (const int stage : {6, 7, 1}) stats_dump_stage(p.key, s, stage);
        // (and as the GPU holds them, after it: RX_VK_GPU_DUMP=1 only; its bindings can sync a pool's mirror mid-frame, and
        // the culling chain then measured as broken)
        static const bool gpu_dump_on = env_on("RX_VK_GPU_DUMP");
        if (stats_query.cmd && gpu_dump_on) stats_query.state = &s, stats_query.prog = p.key;
    }
    fine_poststate.finish();
    if (d.mesh) {
        FINE_SCOPE(FT_MESH_DRAW);
        FINE_N(FC_MESH_DRAWS, 1);
        FINE_N(FC_MESH_INDIRECT, d.indirect != 0);
        FINE_N(FC_MESH_COUNTED, d.count_addr != 0);
        FINE_N(FC_MESH_FIRST, d.first_task != 0);
        if (p.fallback) {   // (hookshot: the generated vertex stage reads the records; one vertex a corner)
            occ.issue();
            vkCmdDrawIndirect(cmd, fallback_draw_records, 0, 1, sizeof(VkDrawIndirectCommand));   // (its count: the records taken)
            st_mesh++;
            st_draws++, ps.draws++;
            return;
        }
        {   // (amd-fix, diagnosis) RX_VK_LOG_MESH_PROGS=1: each mesh program's first draw and every 600th, with what its
            // indirect arguments hold in guest memory (0 groups: the culling pass before it left nothing)
            static const bool per_prog = env_on("RX_VK_LOG_MESH_PROGS");
            if (dump_on() && d.indirect && dump_mesh_indirect().size() < 256) dump_mesh_indirect().push_back({p.key, d.indirect});
            if (per_prog) {
                static std::mutex mm;
                static std::unordered_map<uint64_t, uint64_t> seen;
                uint64_t n;
                {
                    std::lock_guard lk(mm);
                    n = seen[p.key]++;
                }
                if (n % 600 == 0 && seen.size() < 400) {
                    uint32_t w[3] = {(uint32_t)d.groups[0], (uint32_t)d.groups[1], (uint32_t)d.groups[2]}, draws = (uint32_t)std::max(d.max, 1);
                    if (d.indirect) host->gpu_read(d.indirect, w, sizeof w);
                    if (d.count_addr) host->gpu_read(d.count_addr, &draws, 4);
                    logf("vk: mesh prog %016llx draw #%llu frame %llu: groups %u %u %u, %u draws (max %d), indirect %#llx count %#llx; codes%s",
                         (unsigned long long)p.key, (unsigned long long)n, (unsigned long long)texture_frame_no(), w[0], w[1], w[2],
                         draws, d.max, (unsigned long long)d.indirect, (unsigned long long)d.count_addr, program_code_keys(s).c_str());
                }
            }
        }
        {   // (RX_VK_LOG_MESH_DRAWS=1: how large each is, the largest so far, 400 lines)
            static const bool log_sizes = env_on("RX_VK_LOG_MESH_DRAWS");
            if (log_sizes) {
                static std::atomic<int> told{0};
                static std::atomic<uint32_t> most{0};
                uint32_t w[3] = {(uint32_t)d.groups[0], (uint32_t)d.groups[1], (uint32_t)d.groups[2]}, draws = (uint32_t)std::max(d.max, 1);
                if (d.indirect) host->gpu_read(d.indirect, w, sizeof w);
                if (d.count_addr) host->gpu_read(d.count_addr, &draws, 4);
                const uint64_t size = (uint64_t)w[0] * w[1] * w[2] * draws;
                uint32_t was = most.load();
                if (size > was && told++ < 400) {
                    most = (uint32_t)std::min<uint64_t>(size, 0xFFFFFFFFu);
                    logf("vk: mesh draw prog %016llx frame %llu: groups %u %u %u, %u draws (max %d, stride %d), indirect %#llx count %#llx", (unsigned long long)p.key,
                         (unsigned long long)texture_frame_no(), w[0], w[1], w[2], draws, d.max, d.stride, (unsigned long long)d.indirect, (unsigned long long)d.count_addr);
                }
            }
        }
        {   // (RX_VK_LOG_MESH_INDIRECT=1: what indirect mesh draws hold, 300 lines)
            static const bool on = env_on("RX_VK_LOG_MESH_INDIRECT");
            static std::atomic<int> told{0};
            if (on && d.indirect && told < 300) {
                uint32_t w[12]{}, cnt = 0;
                host->gpu_read(d.indirect, w, sizeof w);
                if (d.count_addr) host->gpu_read(d.count_addr, &cnt, 4);
                told++;
                logf("vk: mesh indirect prog %016llx: stride %d max %d count %u (at %#llx): %u %u %u %u %u %u %u %u", (unsigned long long)p.key, d.stride, d.max, cnt,
                     (unsigned long long)d.count_addr, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
            }
        }
        if (d.indirect) {
            ledger_mark(d.indirect, (uint64_t)std::max(d.max, 1) * (uint64_t)std::max(d.stride, 12), kUseIndirect);   // (RX_VK_LEDGER)
            if (d.count_addr) ledger_mark(d.count_addr, 4, kUseIndirect);
            BufferRef cnt;
            {
                FINE_SCOPE(FT_MESH_COUNT_BIND);
                cnt = d.count_addr ? gpu_buffer_bind(d.count_addr, 4, false) : BufferRef{};
            }
            if (mesh_conv_buf) {   // (EvilOOT: the records as {count, 1, 1}, mesh_indirect_convert)
                // (hookshot: as many records as mesh_indirect_convert wrote, not the game's maximum)
                const uint32_t conv_n = (uint32_t)std::clamp(d.max, 1, 4096);
                occ.issue();
                if (mesh_dbg) mesh_count_note(p.key, mesh_conv_off);
                {
                    FINE_SCOPE(FT_MESH_CALL);
                    if (cnt && vkCmdDrawMeshTasksIndirectCountEXT)
                        vkCmdDrawMeshTasksIndirectCountEXT(cmd, mesh_conv_buf, mesh_conv_off, cnt.buffer, cnt.offset, conv_n, 12);
                    else
                        vkCmdDrawMeshTasksIndirectEXT(cmd, mesh_conv_buf, mesh_conv_off, conv_n, 12);
                }
                mesh_conv_buf = VK_NULL_HANDLE;
                st_mesh++;
                st_draws++, ps.draws++;
                return;
            }
            if (d.max < 0 || d.stride < 0) SKIP_HERE();   // (hookshot: the game's ints; a negative count was ~4e9 draws)
            if (!mesh_args.buffer) {
                FINE_SCOPE(FT_MESH_ARGS_BIND);
                mesh_args = gpu_buffer_bind(d.indirect, (uint64_t)std::max(d.max, 1) * std::max(d.stride, 12), false);
                if (!mesh_args) SKIP_HERE();
            }
            const uint32_t step = mesh_args.buffer == mesh_args_buffer && mesh_args_buffer ? 16u : (uint32_t)std::max(d.stride, 12);
            if (mesh_args.buffer == mesh_args_buffer && mesh_args_buffer) {   // (the task programs' firsts: this draw's entries)
                FINE_SCOPE(FT_MESH_PUSH);
                const uint64_t firsts = mesh_args_address + mesh_args.offset + 12;
                vkCmdPushConstants(cmd, p.layout, p.push_stages, Shader::Backend::SPIRV::MESH_FIRST_PUSH_OFFSET, 8, &firsts);
            }
            occ.issue();
            {
                FINE_SCOPE(FT_MESH_CALL);
                if (cnt && vkCmdDrawMeshTasksIndirectCountEXT)
                    vkCmdDrawMeshTasksIndirectCountEXT(cmd, mesh_args.buffer, mesh_args.offset, cnt.buffer, cnt.offset, d.max, step);
                else
                    vkCmdDrawMeshTasksIndirectEXT(cmd, mesh_args.buffer, mesh_args.offset, (uint32_t)std::max(d.max, 1), step);
            }
        } else {
            if (mesh_args_ready()) {   // (a direct draw's first: in an entry of its own, or entry 0's 0)
                FINE_SCOPE(FT_MESH_PUSH);
                uint32_t at = 0;
                if (d.first_task) {
                    at = mesh_args_take(1);
                    mesh_args_words[at * 4 + 3] = (uint32_t)d.first_task;
                }
                const uint64_t firsts = mesh_args_address + (uint64_t)at * 16 + 12;
                vkCmdPushConstants(cmd, p.layout, p.push_stages, Shader::Backend::SPIRV::MESH_FIRST_PUSH_OFFSET, 8, &firsts);
            }
            occ.issue();
            {
                FINE_SCOPE(FT_MESH_CALL);
                vkCmdDrawMeshTasksEXT(cmd, (uint32_t)std::max(d.groups[0], 0), (uint32_t)std::max(d.groups[1], 1),
                                      (uint32_t)std::max(d.groups[2], 1));
            }
        }
        st_mesh++;
        st_draws++, ps.draws++;
        return;
    }
    // vertex streams (from their pools, strides dynamic) and the zero stream
    // (the same streams are bound for draw after draw: the pool lookup, which was ~8% of a race under the lock, is
    // skipped until the submission or the mirror changes. The whole set, already bound on this command buffer, skips
    // the lookup and the bind)
    const int64_t fine_vertex_t0 = g_fine ? prof_now() : 0;
    uint64_t vaddr[16]{};
    uint64_t vsize[16]{};
    uint32_t vstride_key[16]{};
    for (int i = 0; i < 16; i++) {
        if (i == 15 || !s.vbuf[i].address) continue;
        vaddr[i] = s.vbuf[i].address, vsize[i] = s.vbuf[i].size, vstride_key[i] = s.vstream[i].stride;
    }
    const uint64_t vstamp = gpu_bind_stamp();
    bool vertex_complete = true;   // (hookshot: every stream bound; else a zero_buffer stands in, and a count means nothing)
    if (!vertex_set_same(vstamp, vaddr, vsize, vstride_key, cb.vertex_set.stamp, cb.vertex_set.addr, cb.vertex_set.size,
                         cb.vertex_set.stride, cb.vertex_set.on)) {
        VkBuffer vbufs[16];
        VkDeviceSize voffs[16], vsizes[16], vstrides[16];
        bool complete = true;
        for (uint32_t b = 0; b < 16; b++) {
            vbufs[b] = zero_buffer, voffs[b] = 0, vsizes[b] = 65536, vstrides[b] = 0;
            if (b == 15 || !s.vbuf[b].address) continue;
            const uint64_t vb_addr = s.vbuf[b].address, vb_size = std::max<uint64_t>(s.vbuf[b].size, 1);
            const uint64_t vb_stamp = gpu_bind_stamp();
            ledger_mark(vb_addr, vb_size, kUseVertex);   // (RX_VK_LEDGER)
            BufferRef r;
            if (vertex_bind_reuse(vb_addr, vb_size, vb_stamp, kept_vb[b].addr, kept_vb[b].size, kept_vb[b].stamp, kept_vb[b].ok))
                r = kept_vb[b].ref;
            else {
                r = gpu_buffer_bind(vb_addr, vb_size, false);
                kept_vb[b] = {vb_addr, vb_size, gpu_bind_stamp(), r, r ? 1 : 0};
            }
            if (!r) {
                complete = false;
                continue;
            }
            vbufs[b] = r.buffer, voffs[b] = r.offset, vsizes[b] = s.vbuf[b].size ? s.vbuf[b].size : VK_WHOLE_SIZE;
            vstrides[b] = s.vstream[b].stride;
        }
        // (only the bindings that changed: vertex_bind_range)
        uint32_t vb_first = 0, vb_count = 0;
        vertex_bind_range(vbufs, voffs, vsizes, vstrides, cb.vb.b, cb.vb.o, cb.vb.z, cb.vb.s, cb.vb_valid, &vb_first, &vb_count);
        if (vb_count) {
            FINE_N(FC_VBIND, 1);
            vkCmdBindVertexBuffers2(cmd, vb_first, vb_count, vbufs + vb_first, voffs + vb_first, vsizes + vb_first, vstrides + vb_first);
            std::memcpy(cb.vb.b, vbufs, sizeof vbufs), std::memcpy(cb.vb.o, voffs, sizeof voffs);
            std::memcpy(cb.vb.z, vsizes, sizeof vsizes), std::memcpy(cb.vb.s, vstrides, sizeof vstrides);
            cb.vb_valid = true;
        }
        vertex_complete = complete;
        if (complete && gpu_bind_stamp() == vstamp) {
            cb.vertex_set.stamp = vstamp;
            for (int i = 0; i < 16; i++) {
                const bool used = i != 15 && vaddr[i];
                cb.vertex_set.addr[i] = used ? vaddr[i] : 0;
                cb.vertex_set.size[i] = used ? (vsize[i] ? vsize[i] : 1) : 0;
                cb.vertex_set.stride[i] = used ? vstride_key[i] : 0;
            }
            cb.vertex_set.on = true;
        } else
            cb.vertex_set.on = false;
    }
    if (g_fine) fine_ns[FT_VERTEX] += prof_now() - fine_vertex_t0;
    if (!vertex_complete) occ.drop(kOccVertexInput);
    const VkIndexType itype = d.itype == 0 ? VK_INDEX_TYPE_UINT8_EXT : d.itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    // (quads read their indices on the CPU and draw with 32-bit ones: they need no uint8 index support)
    if (d.indexed && d.itype == 0 && !dev.index_uint8 && d.prim != NVN_DRAW_PRIMITIVE_QUADS) SKIP_HERE();
    if (d.prim == NVN_DRAW_PRIMITIVE_QUADS) {   // (quads as two triangles each: indices made in the stream)
        if (d.indirect || d.count < 4) SKIP_HERE();
        if (d.count > (1 << 24)) SKIP_HERE();   // (hookshot: 4M quads at most; the game's count sized host vectors of up to 16 GB)
        const uint32_t quads = (uint32_t)d.count / 4, unit = d.itype == 0 ? 1 : d.itype == 1 ? 2 : 4;
        std::vector<uint32_t> src(d.indexed ? (size_t)quads * 4 : 0);
        if (d.indexed) {
            std::vector<uint8_t> raw((size_t)quads * 4 * unit);
            if (!memory_read_as_submitted(d.ibuf, raw.data(), raw.size())) SKIP_HERE();
            for (size_t i = 0; i < src.size(); i++) {
                uint32_t v = 0;
                std::memcpy(&v, raw.data() + i * unit, unit);
                src[i] = v;
            }
        }
        const uint64_t at6 = stream_alloc((uint64_t)quads * 6 * 4, 16);
        if (at6 == UINT64_MAX) SKIP_HERE();
        uint32_t* out = reinterpret_cast<uint32_t*>(cur->mapped + at6);
        for (uint32_t q = 0; q < quads; q++) {
            const uint32_t b = q * 4;
            const uint32_t v[4] = {d.indexed ? src[b] : (uint32_t)d.first + b, d.indexed ? src[b + 1] : (uint32_t)d.first + b + 1,
                                   d.indexed ? src[b + 2] : (uint32_t)d.first + b + 2, d.indexed ? src[b + 3] : (uint32_t)d.first + b + 3};
            const uint32_t tri[6] = {v[0], v[1], v[2], v[0], v[2], v[3]};
            std::memcpy(out + q * 6, tri, sizeof tri);
        }
        bind_index(cmd, cur->stream, at6, VK_INDEX_TYPE_UINT32);
        occ.issue();
        vkCmdDrawIndexed(cmd, quads * 6, (uint32_t)std::max(1, d.instances), 0, d.indexed ? d.base_vertex : 0,
                         (uint32_t)d.base_instance);
        st_draws++, ps.draws++;
        return;
    }
    if (d.indirect) {
        // (hookshot) the records the draw reads, at the stride it reads them with (20 for indexed draws), all bound; a
        // negative count or stride from the game is no draw
        if (d.max < 0 || d.stride < 0) SKIP_HERE();
        const int ind_stride = std::max(d.stride, d.indexed ? 20 : 16);
        ledger_mark(d.indirect, (uint64_t)std::max(d.max, 1) * (uint64_t)ind_stride, kUseIndirect);   // (RX_VK_LEDGER)
        if (d.count_addr) ledger_mark(d.count_addr, 4, kUseIndirect);
        BufferRef args = gpu_buffer_bind(d.indirect, (uint64_t)std::max(d.max, 1) * (uint64_t)ind_stride, false);
        BufferRef cnt = d.count_addr ? gpu_buffer_bind(d.count_addr, 4, false) : BufferRef{};
        if (!args) SKIP_HERE();
        if (d.indexed) {
            ledger_mark(d.ibuf, 4, kUseIndex);
            BufferRef ib = gpu_buffer_bind(d.ibuf, 4, false);
            if (!ib) SKIP_HERE();
            bind_index(cmd, ib.buffer, ib.offset, itype);
            occ.issue();
            if (cnt) vkCmdDrawIndexedIndirectCount(cmd, args.buffer, args.offset, cnt.buffer, cnt.offset, std::max(d.max, 1), ind_stride);
            else vkCmdDrawIndexedIndirect(cmd, args.buffer, args.offset, std::max(d.max, 1), ind_stride);
        } else {
            occ.issue();
            if (cnt) vkCmdDrawIndirectCount(cmd, args.buffer, args.offset, cnt.buffer, cnt.offset, std::max(d.max, 1), ind_stride);
            else vkCmdDrawIndirect(cmd, args.buffer, args.offset, std::max(d.max, 1), ind_stride);
        }
    } else if (d.indexed) {
        const uint32_t unit = d.itype == 0 ? 1 : d.itype == 1 ? 2 : 4;
        BufferRef ib;
        {
            FINE_SCOPE(FT_VERTEX);
            ledger_mark(d.ibuf, (uint64_t)std::max(d.count, 1) * unit, kUseIndex);   // (RX_VK_LEDGER)
            ib = gpu_buffer_bind(d.ibuf, (uint64_t)std::max(d.count, 1) * unit, false);
            if (!ib) SKIP_HERE();
            FINE_N(FC_IBIND, 1);
            bind_index(cmd, ib.buffer, ib.offset, itype);
        }
        FINE_SCOPE(FT_DRAWCALL);
        occ.issue();
        vkCmdDrawIndexed(cmd, (uint32_t)d.count, (uint32_t)std::max(1, d.instances), 0, d.base_vertex, (uint32_t)d.base_instance);
    } else {
        FINE_SCOPE(FT_DRAWCALL);
        occ.issue();
        vkCmdDraw(cmd, (uint32_t)d.count, (uint32_t)std::max(1, d.instances), (uint32_t)d.first, (uint32_t)d.base_instance);
    }
    // Mark recorded gameplay draws, so an awaited PSO cannot make an empty frame look ready.
    if (cp->key == 0x1a7482c0ec9991eaULL) menu_fade_world_at.store(presents_now, std::memory_order_relaxed);
    st_draws++, ps.draws++;
}

/// A dispatch over 2 ms is logged, 60 times, and only while RX_VK_SLOW_DISPATCH is set. Off, the clock is not read.
bool slow_dispatch_timing() {
    static const int on = env_on("RX_VK_SLOW_DISPATCH");
    return on;
}

struct DispatchClock {   // (a dispatch over 2 ms: logged, with where its time went)
    std::chrono::steady_clock::time_point t0 = slow_dispatch_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}, t_prog{}, t_pipe{}, t_desc{};
    uint64_t key = 0;
    bool made_pipeline = false;
    ~DispatchClock() {
        if (!slow_dispatch_timing()) return;
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        static int logged;
        if (ms < 2.0 || logged >= 60) return;
        logged++;
        auto at = [&](std::chrono::steady_clock::time_point t) {
            return t.time_since_epoch().count() ? std::chrono::duration<double, std::milli>(t - t0).count() : -1.0;
        };
        logf("vk: slow dispatch %.1f ms: program %016llx (picked at %.1f), pipeline%s at %.1f, descriptors at %.1f",
             ms, (unsigned long long)key, at(t_prog), made_pipeline ? " made" : "", at(t_pipe), at(t_desc));
    }
};

/// (hookshot) OoT's neural upscaler (about 13 tensor-core compute passes a frame over storage-buffer feature maps,
/// 1920x1080 in, 2560x1440 out) does not translate: its matrix multiplies and a few other native opcodes have no
/// lowering. Its last pass reads the network's result, the input colour and a noise texture and writes the output
/// image; left out, the output stays black under the UI drawn over it. Here a rejected compute pass that writes a
/// storage image larger than the largest colour texture it samples (and does not also write) is replaced by a linear
/// blit of that texture into the image: the frame at its full size, without the network's detail. (An image the
/// pass also samples is one of its inputs, never the output.) The gameplay networks' passes also when the image is the
/// texture's size: the full-size pass at 100% Render resolution, an anti-aliasing of the 3D at the output's size.
/// The first replacement of each program is logged.
/// (shaders/movie_upscale.comp) The upscale in a shader: it reads and writes the two images as the network's own passes do, where a blit came
/// out black on an RX 9070 XT. False: no way to, the blit is tried
bool movie_upscale(VkCommandBuffer cmd, uint32_t in_handle, uint32_t out_handle, const Texture* out) {
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return false;
    Texture* a = nullptr;
    Texture* b = nullptr;
    uint32_t la = 0, lb = 0;
    const VkImageView in = texture_view_of(in_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, VK_FORMAT_UNDEFINED, &a, &la);
    const VkImageView to = texture_view_of(out_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, VK_FORMAT_R8G8B8A8_UNORM, &b, &lb);
    if (!in || !to || !b) return false;
    static VkPipeline pipe = VK_NULL_HANDLE;
    static VkPipelineLayout layout = VK_NULL_HANDLE;
    static VkSampler linear = VK_NULL_HANDLE;
    static const bool made = [] {
        VkDescriptorSetLayoutBinding bind[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                                {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        li.bindingCount = 2, li.pBindings = bind;
        VkDescriptorSetLayout set{};
        vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &set);
        const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pi.setLayoutCount = 1, pi.pSetLayouts = &set, pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &pr;
        vkCreatePipelineLayout(dev.device, &pi, nullptr, &layout);
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        const auto spv = BUILTIN_SPV(MOVIE_UPSCALE_COMP_SPV);
        mi.codeSize = spv.size, mi.pCode = spv.code;
        VkShaderModule m{};
        {
            IntelDriverGate gate;   // (the module's only: texture_compute_pipeline takes the same lock)
            if (guarded_shader_module(&mi, &m) != VK_SUCCESS) m = VK_NULL_HANDLE;
        }
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
        ci.layout = layout;
        // (vk_texture.cpp: through the pipeline cache, and without the optimizer where the driver refuses it. Asked once,
        // optimized, here: an Arc A580 on 101.9175 refused the typed upscale helper's, a network's output a blit)
        if (!m || texture_compute_pipeline(ci, pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
        vkDestroyShaderModule(dev.device, m, nullptr);
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.25f;
        vkCreateSampler(dev.device, &si, nullptr, &linear);
        return pipe != VK_NULL_HANDLE;
    }();
    if (!made) return false;
    const VkDescriptorImageInfo in_info{linear, in, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo out_info{VK_NULL_HANDLE, to, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    w[0].dstBinding = 0, w[0].descriptorCount = 1, w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, w[0].pImageInfo = &in_info;
    w[1].dstBinding = 1, w[1].descriptorCount = 1, w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, w[1].pImageInfo = &out_info;
    const uint32_t size[2] = {std::max(1u, out->width >> lb), std::max(1u, out->height >> lb)};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, w);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, size);
    vkCmdDispatch(cmd, (size[0] + 7) / 8, (size[1] + 7) / 8, 1);
    descriptors_disturbed();
    return true;
}

// (hookshot, issue 109) The movie upscaler's networks cost 12 ms a frame on an RTX 3050 and 83-93 ms on an RTX 4060 with a newer
// driver: the intro movie ran at 10 fps and the game never got past it. When a frame of few draws (a movie) takes over 40 ms
// of GPU time (vk_replay.cpp), they are left out from then on, as on AMD: the linear upscale below, the same picture a little
// softer. The gameplay networks are never left out. RX_VK_NETWORK_KEEP=all: never
std::atomic<bool> movie_networks_slow{false};
bool gameplay_network(uint64_t key) { return key == 0xfd029a317a9a4b94ull || key == 0x9dc98871eb7fc101ull; }

static movie_upscale_policy::Mode movie_upscale_mode() {
    // Read on first dispatch, after host/hos_main.c has loaded hookshot_env.txt.
    static const auto mode = movie_upscale_policy::parse(getenv("RX_VK_MOVIE_UPSCALE"));
    return mode;
}

static bool movie_keep_all_networks() {
    static const bool keep = getenv("RX_VK_NETWORK_KEEP") && !strcmp(getenv("RX_VK_NETWORK_KEEP"), "all");
    return keep;
}

static bool typed_upscale(VkCommandBuffer cmd, VkImageView input_view, VkImageView output_view,
                          Texture* out, uint32_t level, uint64_t key, const char* keys, bool temporal);
static bool temporal_rejected_network(VkCommandBuffer cmd, uint64_t key, Texture* src, const nvn_tex* in, Texture* dst,
                                      const nvn_tex* out);

thread_local Texture* stub_picture = nullptr;   // (upscale_stub's output, when it is the scene's format: framegen_hud_before)
bool upscale_stub(VkCommandBuffer cmd, const NvnState& s, uint64_t key) {
    stub_picture = nullptr;
    nvn_tex out{}, in{};
    uint64_t out_area = 0, in_area = 0, in_h = 0, out_h = 0;
    uint64_t img_gpu[32]{}, tex_gpu[128]{};
    int nimg = 0, ntex = 0;
    uint32_t ntex_slots = 0, nimg_slots = 0;   // (the program's own slots: later ones hold other passes' bindings)
    cs_slot_counts(s, &ntex_slots, &nimg_slots);
    ntex_slots = std::min(ntex_slots, 128u), nimg_slots = std::min(nimg_slots, 32u);
    for (uint32_t i = 0; i < ntex_slots; i++) {   // (what the pass samples: an image it also samples is an input, not the output)
        nvn_tex t;
        nvn_view v;
        bool hv = false;
        if (s.tex[5][i] && texture_of((int)(s.tex[5][i] & 0xFFFFF), &t, &v, &hv)) tex_gpu[ntex++] = t.gpu;
    }
    for (uint32_t i = 0; i < nimg_slots; i++) {
        nvn_tex t;
        nvn_view v;
        bool hv = false;
        if (!s.img[5][i] || !texture_of((int)(s.img[5][i] & 0xFFFFF), &t, &v, &hv)) continue;
        img_gpu[nimg++] = t.gpu;
        if (std::find(tex_gpu, tex_gpu + ntex, t.gpu) != tex_gpu + ntex) continue;
        const uint64_t a = (uint64_t)t.d.width * t.d.height;
        if (t.d.depth <= 1 && a > out_area) out = t, out_area = a, out_h = s.img[5][i];
    }
    if (!out_area) return false;
    for (uint32_t i = 0; i < ntex_slots; i++) {
        nvn_tex t;
        nvn_view v;
        bool hv = false;
        if (!s.tex[5][i] || !texture_of((int)(s.tex[5][i] & 0xFFFFF), &t, &v, &hv)) continue;
        if (t.d.depth > 1 || t.d.target != out.d.target) continue;
        if (std::find(img_gpu, img_gpu + nimg, t.gpu) != img_gpu + nimg) continue;   // (the network's own buffers)
        const uint64_t a = (uint64_t)t.d.width * t.d.height;
        if (a > in_area) in = t, in_area = a, in_h = s.tex[5][i];
    }
    // (an output no larger than the input is no upscale, except for the gameplay networks: at 100% Render resolution the
    // full-size pass, 9dc98871eb7fc101, reads the 3D at the output's own size and writes the picture. Left out on a GPU
    // without f16 shaders (vk_shader.cpp: GTX 10, Polaris on RADV) and turned away here, it wrote nothing: the 3D was black
    // under the HUD at 100%, and only there)
    if (!in_area || out_area < in_area || (out_area == in_area && !gameplay_network(key))) return false;
    Texture* src = texture_get(&in, true);
    Texture* dst = texture_get(&out, false);
    if (!src || !dst || !src->image || !dst->image || (src->aspect | dst->aspect) != VK_IMAGE_ASPECT_COLOR_BIT || src->bc ||
        dst->bc || src->samples != VK_SAMPLE_COUNT_1_BIT || dst->samples != VK_SAMPLE_COUNT_1_BIT)
        return false;
    // (the gameplay upscale's picture, for frame generation's HUD mark once this returns: the scene's format into the same,
    // larger. framegen_hud_before takes only the picture's formats; the movies' RGBA8 is not one)
    if (src->format == dst->format) stub_picture = dst;
    // (hookshot) an upscale network that did not translate here (the Render resolution's fixed shares have one each, and on a
    // GPU without cooperative matrices they are left out): the temporal upscaler is offered it first, as temporal_stand_in
    // offers the networks that did translate. It takes only the scene into the picture; anything else goes on below
    if (temporal_wanted() && temporal_rejected_network(cmd, key, src, &in, dst, &out)) return true;
    // (hookshot) an upscale network that did not translate here (the Render resolution's fixed shares have one each, and on a
    // GPU without cooperative matrices they are left out): the temporal upscaler is offered it first, as temporal_stand_in
    // offers the networks that did translate, and so is FSR 3.1, as fsr3_stand_in. It takes only the scene into the picture;
    // anything else goes on below
    if ((temporal_wanted() || temporal_off() || fsr3_inputs::wanted()) && temporal_rejected_network(cmd, key, src, &in, dst, &out))
        return true;
    VkFormatProperties fs{}, fd{};
    vkGetPhysicalDeviceFormatProperties(dev.physical, src->format, &fs);
    vkGetPhysicalDeviceFormatProperties(dev.physical, dst->format, &fd);
    if (!(fs.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(fd.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) ||
        !(fs.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        return false;
    end_pass(cmd);
    full_barrier(cmd);
    if (dst->format == VK_FORMAT_R8G8B8A8_UNORM && movie_upscale(cmd, in_h, out_h, dst)) {
        full_barrier(cmd);
        texture_written(dst);
        static std::atomic<bool> logged_movie_upscale{false};
        if (!logged_movie_upscale.exchange(true, std::memory_order_relaxed))
            logf("vk: movie upscaling uses the linear compute path");
        return true;
    }
    // Failed upscales can receive RGB9E5 bits in a shared R11 image, including the gameplay networks (the 75% TAAU and the
    // full-size pass). Match the observed base-level 2D case only; other fallbacks keep their existing blit.
    const bool gameplay_taau = key == 0xfd029a317a9a4b94ull;
    const bool gameplay_full = key == 0x9dc98871eb7fc101ull;
    if (gameplay_taau && scene_tonemap_on() &&
        (src->format == VK_FORMAT_R16G16B16A16_SFLOAT || dst->format == VK_FORMAT_R16G16B16A16_SFLOAT)) {
        if (src->format != VK_FORMAT_R16G16B16A16_SFLOAT || dst->format != VK_FORMAT_R16G16B16A16_SFLOAT ||
            in.d.format != NVN_FORMAT_RGB9E5F || out.d.format != NVN_FORMAT_R11G11B10F ||
            !(fd.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ||
            src->type != VK_IMAGE_TYPE_2D || dst->type != VK_IMAGE_TYPE_2D || src->depth != 1 || dst->depth != 1 ||
            src->layers != 1 || dst->layers != 1 || src->levels != 1 || dst->levels != 1 || src->skip || dst->skip ||
            in.d.swizzle_r != NVN_TEXTURE_SWIZZLE_R || in.d.swizzle_g != NVN_TEXTURE_SWIZZLE_G ||
            in.d.swizzle_b != NVN_TEXTURE_SWIZZLE_B) return false;
        nvn_tex selected_in{}, selected_out{};
        nvn_view iv{}, ov{};
        bool ihv = false, ohv = false;
        if (texture_of((int)(in_h & 0xFFFFF), &selected_in, &iv, &ihv) &&
            texture_of((int)(out_h & 0xFFFFF), &selected_out, &ov, &ohv) && !ihv && !ohv &&
            selected_in.serial == in.serial && selected_out.serial == out.serial) {
            VkFormat input_format = VK_FORMAT_UNDEFINED, output_format = VK_FORMAT_UNDEFINED;
            const VkImageView input_view = texture_view(src, &in, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample,
                                                         &input_format, VK_FORMAT_R16G16B16A16_SFLOAT);
            const VkImageView output_view = texture_view(dst, &out, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage,
                                                          &output_format, VK_FORMAT_R16G16B16A16_SFLOAT);
            if (input_view && output_view && input_format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                output_format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                typed_upscale(cmd, input_view, output_view, dst, 0, key, "PsychoV35 TAAU fallback", true)) {
                full_barrier(cmd);
                texture_written(dst);
                return true;
            }
        }
        return false; // Do not blit perceptually encoded PsychoV35 input into a linear output.
    }
    if (in.d.format == NVN_FORMAT_RGB9E5F && out.d.format == NVN_FORMAT_R11G11B10F &&
        src->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && dst->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 &&
        (fd.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) &&
        src->type == VK_IMAGE_TYPE_2D && dst->type == VK_IMAGE_TYPE_2D && src->depth == 1 && dst->depth == 1 &&
        src->layers == 1 && dst->layers == 1 && src->levels == 1 && dst->levels == 1 && !src->skip && !dst->skip &&
        in.d.swizzle_r == NVN_TEXTURE_SWIZZLE_R && in.d.swizzle_g == NVN_TEXTURE_SWIZZLE_G &&
        in.d.swizzle_b == NVN_TEXTURE_SWIZZLE_B) {
        VkFormatProperties e5{};
        vkGetPhysicalDeviceFormatProperties(dev.physical, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, &e5);
        const VkFormatFeatureFlags sampled = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        nvn_tex selected_in{}, selected_out{};
        nvn_view iv{}, ov{};
        bool ihv = false, ohv = false;
        if ((e5.optimalTilingFeatures & sampled) == sampled &&
            texture_of((int)(in_h & 0xFFFFF), &selected_in, &iv, &ihv) &&
            texture_of((int)(out_h & 0xFFFFF), &selected_out, &ov, &ohv) && !ihv && !ohv &&
            selected_in.serial == in.serial && selected_out.serial == out.serial) {
            VkFormat input_format = VK_FORMAT_UNDEFINED, output_format = VK_FORMAT_UNDEFINED;
            const VkImageView input_view = texture_view(src, &in, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, &input_format);
            const VkImageView output_view = texture_view(dst, &out, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage,
                                                         &output_format, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
            if (input_view && output_view && input_format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 &&
                output_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 &&
                typed_upscale(cmd, input_view, output_view, dst, 0, key, "generic RGB9E5 fallback", gameplay_network(key))) {
                full_barrier(cmd);
                texture_written(dst);
                if (gameplay_taau) {
                    static std::atomic<bool> logged_gameplay_upscale{false};
                    if (!logged_gameplay_upscale.exchange(true, std::memory_order_relaxed))
                        logf("vk: gameplay RGB9E5 upscale uses the typed fallback");
                }
                if (gameplay_full) {
                    static std::atomic<bool> logged_full{false};
                    if (!logged_full.exchange(true, std::memory_order_relaxed))
                        logf("vk: the full-size network 9dc98871eb7fc101 is drawn by the typed fallback (%ux%u -> %ux%u)",
                             in.d.width, in.d.height, out.d.width, out.d.height);
                }
                return true;
            }
        }
    }
    if ((gameplay_taau || gameplay_full) && in.d.format == NVN_FORMAT_RGB9E5F && out.d.format == NVN_FORMAT_R11G11B10F &&
        src->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && dst->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
        static std::atomic<bool> warned_gameplay_upscale{false};
        if (!warned_gameplay_upscale.exchange(true, std::memory_order_relaxed))
            logf("vk: gameplay RGB9E5 upscale unavailable; the fallback blit may have incorrect colors");
    }
    VkImageBlit bl{};
    bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bl.srcOffsets[1] = {(int32_t)src->width, (int32_t)src->height, 1};
    bl.dstOffsets[1] = {(int32_t)dst->width, (int32_t)dst->height, 1};
    vkCmdBlitImage(cmd, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
    full_barrier(cmd);
    texture_written(dst);
    if (dump_on()) {   // (in a frame dump: what it read and what it made)
        dump_image(cmd, src, 0, 0, "upscale stub source");
        dump_image(cmd, dst, 0, 0, "upscale stub result");
    }
    static std::mutex mu;
    static std::unordered_set<uint64_t> seen;
    std::lock_guard lk(mu);
    if (seen.size() < 32 && seen.insert(key).second)
        logf("vk: program %016llx does not translate: its dispatch replaced by a linear upscale of %#llx %ux%u (vk %d) "
             "into %#llx %ux%u (vk %d)", (unsigned long long)key, (unsigned long long)in.gpu, in.d.width, in.d.height,
             (int)src->format, (unsigned long long)out.gpu, out.d.width, out.d.height, (int)dst->format);
    (void)in_h, (void)out_h;
    return true;
}

/// (hookshot) OoT's gameplay TAAU network (fd029a317a9a4b94) stood in for: its input upscaled into its output
/// (shaders/taau_upscale.comp). The network's translation loses the blue channel, the yellow cast over every
/// scene; with this the picture is right, without the network's temporal anti-aliasing.
/// RX_VK_TAAU_KEYS=<program key prefixes,...> (default fd029a317a9a4b94; "-": none, the network runs)
bool taau_key(uint64_t key) {
    // (the real network where the card runs it as the console does; on the software multiplies it cost most of a frame on AMD's cards, and a
    // locked 60 with this in its place: Hookshot issue 52)
    static const char* keys = getenv("RX_VK_TAAU_KEYS") ? getenv("RX_VK_TAAU_KEYS")
                              : Shader::Backend::SPIRV::mma_software || !float16_shaders()
                                    ? "fd029a317a9a4b94" : "-";   // (no f16 shaders: vk_shader.cpp leaves the network out)
    char hex[24];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)key);
    bool match = false;
    for (const char* q = keys; *q && !match;) {
        const char* e = q;
        while (*e && *e != ',') e++;
        match = e > q && *q != '-' && !strncmp(hex, q, (size_t)(e - q));
        q = *e ? e + 1 : e;
    }
    return match;
}
static bool typed_upscale(VkCommandBuffer cmd, VkImageView input_view, VkImageView output_view,
                          Texture* out, uint32_t level, uint64_t key, const char* keys, bool temporal) {
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR || !input_view || !output_view || !out) return false;
    const bool rgba = out->format == VK_FORMAT_R16G16B16A16_SFLOAT;
    if (out->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && !(rgba && temporal && scene_tonemap_on())) return false;
    struct Pipeline {
        VkPipeline pipe = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkSampler linear = VK_NULL_HANDLE;
        bool tried = false;
    };
    static Pipeline pipelines[2];
    Pipeline& pipeline = pipelines[rgba ? 1 : 0];
    auto& pipe = pipeline.pipe;
    auto& layout = pipeline.layout;
    auto& linear = pipeline.linear;
    auto& tried = pipeline.tried;
    if (!tried) {
        tried = true;
        VkDescriptorSetLayoutBinding b[3] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                             {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                             {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        li.bindingCount = 3, li.pBindings = b;
        VkDescriptorSetLayout set{};
        vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &set);
        const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
        VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pi.setLayoutCount = 1, pi.pSetLayouts = &set, pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &pr;
        vkCreatePipelineLayout(dev.device, &pi, nullptr, &layout);
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        const auto spv = rgba ? BUILTIN_SPV(TAAU_UPSCALE_RGBA_COMP_SPV) : BUILTIN_SPV(TAAU_UPSCALE_COMP_SPV);
        mi.codeSize = spv.size, mi.pCode = spv.code;
        VkShaderModule m{};
        {
            IntelDriverGate gate;   // (the module's only: texture_compute_pipeline takes the same lock)
            if (guarded_shader_module(&mi, &m) != VK_SUCCESS) m = VK_NULL_HANDLE;
        }
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
        ci.layout = layout;
        // (vk_texture.cpp: through the pipeline cache, and without the optimizer where the driver refuses it. Asked once,
        // optimized, here: an Arc A580 on 101.9175 refused the typed upscale helper's, a network's output a blit)
        if (!m || texture_compute_pipeline(ci, pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
        vkDestroyShaderModule(dev.device, m, nullptr);
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.25f;
        vkCreateSampler(dev.device, &si, nullptr, &linear);
        logf("vk: typed upscale helper %s, %s output%s", keys, rgba ? "RGBA16F" : "packed",
             pipe ? "" : ": its pipeline was not made");
    }
    if (!pipe) return false;
    uint32_t size[3] = {std::max(1u, out->width >> level), std::max(1u, out->height >> level), 0};
    // (the stand-in's last output, which bounds this one's pixels: shaders/taau_upscale.comp. Made again at a new size,
    // the first frame unbounded; RX_VK_TAAU_NO_CLAMP=1: never bounded)
    struct History {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation memory = nullptr;
        VkImageView view = VK_NULL_HANDLE;
        uint32_t width = 0, height = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint64_t frame = 0;
    };
    static History histories[2];
    History& history = histories[temporal ? 0 : 1];
    auto& hist = history.image;
    auto& hist_mem = history.memory;
    auto& hist_view = history.view;
    auto& hist_w = history.width;
    auto& hist_h = history.height;
    auto& hist_frame = history.frame;
    // (off by default: bounding a pixel by the last frame's smeared everything that moves (the user: blurry, warping)
    // and only dimmed the flashes; RX_VK_TAAU_CLAMP=1 turns it on)
    static const bool no_clamp = !env_on("RX_VK_TAAU_CLAMP");
    // (the ones a new size replaced: given back 120 presents later, when no work in flight can use them. They were left, the
    // output's size changing with the dynamic resolution)
    struct OldHist { VkImage image; VmaAllocation mem; VkImageView view; uint64_t frame; };
    static std::vector<OldHist> old_hist;
    for (size_t i = 0; i < old_hist.size();) {
        if (texture_frame_no() - old_hist[i].frame < 120) { i++; continue; }
        if (old_hist[i].view) vkDestroyImageView(dev.device, old_hist[i].view, nullptr);
        destroy_image(old_hist[i].image, old_hist[i].mem);
        old_hist.erase(old_hist.begin() + (ptrdiff_t)i);
    }
    if (hist_w != size[0] || hist_h != size[1] || history.format != out->format) {
        // (the one before is kept for the while: work in flight may still use it)
        if (hist) old_hist.push_back({hist, hist_mem, hist_view, texture_frame_no()});
        hist = VK_NULL_HANDLE, hist_view = VK_NULL_HANDLE;
        VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ic.imageType = VK_IMAGE_TYPE_2D, ic.format = out->format;
        ic.extent = {size[0], size[1], 1}, ic.mipLevels = 1, ic.arrayLayers = 1, ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling = VK_IMAGE_TILING_OPTIMAL, ic.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        if (!create_image(ic, hist, hist_mem)) { hist = VK_NULL_HANDLE; return false; }
        VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vc.image = hist, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = ic.format;
        vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(dev.device, &vc, nullptr, &hist_view);
        VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};   // (UNDEFINED -> GENERAL, once)
        ib.srcStageMask = VK_PIPELINE_STAGE_2_NONE, ib.srcAccessMask = VK_ACCESS_2_NONE;
        ib.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        ib.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = hist, ib.subresourceRange = vc.subresourceRange;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &ib;
        vkCmdPipelineBarrier2(cmd, &di);
        hist_w = size[0], hist_h = size[1], history.format = out->format, hist_frame = 0;
    }
    // (bounded when the history is last frame's or so: a pause of the TAAU, a menu, starts again unbounded)
    const uint64_t now_frame = texture_frame_no();
    size[2] = temporal && !no_clamp && hist_frame && now_frame - hist_frame <= 3 ? 1u : 0u;
    // (the gameplay stand-in's edges: FXAA in the shader. RX_VK_TAAU_FXAA=0: the plain linear upscale as before; 2: the
    // pixels it changed drawn red)
    static const int fxaa = getenv("RX_VK_TAAU_FXAA") ? atoi(getenv("RX_VK_TAAU_FXAA")) : 1;
    if (temporal && fxaa) size[2] |= fxaa == 2 ? 6u : 2u;
    hist_frame = now_frame;
    const VkDescriptorImageInfo in_info{linear, input_view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo out_info{VK_NULL_HANDLE, output_view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo hist_info{VK_NULL_HANDLE, hist_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[3]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
                              {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    w[0].dstBinding = 0, w[0].descriptorCount = 1, w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &in_info;
    w[1].dstBinding = 1, w[1].descriptorCount = 1, w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[1].pImageInfo = &out_info;
    w[2].dstBinding = 2, w[2].descriptorCount = 1, w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[2].pImageInfo = &hist_info;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 3, w);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, size);
    vkCmdDispatch(cmd, (size[0] + 7) / 8, (size[1] + 7) / 8, 1);
    descriptors_disturbed();   // (the program's own pushes are gone from the bind point)
    if (temporal) probe_image(cmd, out, level, 0, key, 't');
    return true;
}
/// (hookshot) AMD FSR 3.1 in place of the gameplay upscale network (vk_fsr3.cpp), and its frame interpolation's inputs from it
/// (vk_fsr3_inputs.cpp): the same dispatches as temporal_stand_in below, found the same way. true: FSR 3.1 wrote the output
bool fsr3_stand_in(VkCommandBuffer cmd, const NvnState& s, const Program& p) {
    if (p.stages.empty() || !fsr3_inputs::wanted()) return false;
    const Stage& st = p.stages[0];
    if (!gameplay_network(p.key) && !(st.info.uses_cooperative_matrix && st.info.texture_descriptors.size() == 1 &&
                                      st.info.image_descriptors.size() == 1))
        return false;
    if (st.info.texture_descriptors.empty() || st.info.image_descriptors.empty()) return false;
    size_t ntex = 0;   // (build_descriptors' image infos: the textures', then the images')
    for (const auto& d : st.info.texture_descriptors) ntex += d.count;
    if (dw().imgs.size() <= ntex || !dw().imgs[0].imageView || !dw().imgs[ntex].imageView) return false;
    Texture* in = nullptr;
    const uint32_t in_handle = handle_of(s, st, st.info.texture_descriptors[0], 0);
    if (in_handle) texture_view_of(in_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, VK_FORMAT_UNDEFINED, &in);
    const auto& od = st.info.image_descriptors[0];
    Texture* out = nullptr;
    uint32_t level = 0;
    const uint32_t out_handle = handle_of(s, st, od, 0);
    if (out_handle) texture_view_of(out_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, image_format(od.format), &out, &level);
    VkImageView input_view = dw().imgs[0].imageView, output_view = dw().imgs[ntex].imageView;
    if (in && out && fsr3_inputs::colour_pair(in->format, out->format) && out->format == VK_FORMAT_R16G16B16A16_SFLOAT) {
        input_view = texture_view_of(in_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, VK_FORMAT_R16G16B16A16_SFLOAT);
        output_view = texture_view_of(out_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, VK_FORMAT_R16G16B16A16_SFLOAT);
    }
    return fsr3_upscale(cmd, p.key, gameplay_network(p.key), in, input_view, out, output_view, level);
}
/// (hookshot) A temporal upscaler (NVIDIA DLSS, AMD FSR 4: vk_temporal.cpp) in place of the gameplay upscale network, when
/// one is on and its inputs are as expected: the network's input as it samples it, its output's storage view
bool temporal_stand_in(VkCommandBuffer cmd, const NvnState& s, const Program& p) {
    // (the 4/3 network and the full-size pass by their keys; the game's other gameplay upscales, one network for each fixed
    // share of the Render resolution row (67%: 6676619641320ba7, 58% f10782f9e58b6a7c, 50% 72ba090c70fe2614, 33%
    // 0775215f8da720e6 at 2048x1152), by their shape: a network sampling one texture into one image. temporal_upscale takes those
    // only when the texture is this frame's scene, in the gameplay networks' input format, and the image the picture)
    // (Off: the network's stretch; with no engine, frame generation's inputs alone: temporal_fg_inputs, the network runs)
    if ((!temporal_wanted() && !temporal_off() && !temporal_fg_inputs()) || p.stages.empty()) return false;
    const Stage& st = p.stages[0];
    if (!gameplay_network(p.key) && !(st.info.uses_cooperative_matrix && st.info.texture_descriptors.size() == 1 &&
                                      st.info.image_descriptors.size() == 1))
        return false;
    if (st.info.texture_descriptors.empty() || st.info.image_descriptors.empty()) return false;
    size_t ntex = 0;   // (build_descriptors' image infos: the textures', then the images')
    for (const auto& d : st.info.texture_descriptors) ntex += d.count;
    if (dw().imgs.size() <= ntex || !dw().imgs[0].imageView || !dw().imgs[ntex].imageView) return false;
    Texture* in = nullptr;
    const uint32_t in_handle = handle_of(s, st, st.info.texture_descriptors[0], 0);
    if (in_handle) texture_view_of(in_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, VK_FORMAT_UNDEFINED, &in);
    const auto& od = st.info.image_descriptors[0];
    Texture* out = nullptr;
    uint32_t level = 0;
    const uint32_t out_handle = handle_of(s, st, od, 0);
    if (out_handle) texture_view_of(out_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, image_format(od.format), &out, &level);
    VkImageView in_view = dw().imgs[0].imageView, out_view = dw().imgs[ntex].imageView;
    if (in && out && temporal_colour_pair(in->format, out->format) && out->format == VK_FORMAT_R16G16B16A16_SFLOAT) {
        in_view = texture_view_of(in_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, VK_FORMAT_R16G16B16A16_SFLOAT);
        out_view = texture_view_of(out_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, VK_FORMAT_R16G16B16A16_SFLOAT);
    }
    const bool done = temporal_upscale(cmd, p.key, gameplay_network(p.key), in, in_view, out, out_view, level);
    // (its actor motion may have drawn with a pipeline, viewport and descriptors of its own: the next draw sets all of its again)
    bound().reset();
    return done;
}
/// (hookshot) The picture a gameplay upscale writes (its output, found as temporal_stand_in finds it), for frame generation's
/// HUD mark (vk_framegen.h framegen_hud_before: copied once the upscale wrote it, whichever drew it). Null unless AMD FSR
/// 3.1's frame interpolation is asked for, or for any other dispatch
static Texture* upscale_picture(const NvnState& s, const Program& p, uint32_t* level) {
    if (p.stages.empty() || !fsr3fi::frames_wanted()) return nullptr;
    const Stage& st = p.stages[0];
    const bool known = gameplay_network(p.key);
    if (!known && !(st.info.uses_cooperative_matrix && st.info.texture_descriptors.size() == 1 &&
                    st.info.image_descriptors.size() == 1))
        return nullptr;
    if (st.info.texture_descriptors.empty() || st.info.image_descriptors.empty()) return nullptr;
    Texture* in = nullptr;
    const uint32_t in_handle = handle_of(s, st, st.info.texture_descriptors[0], 0);
    if (in_handle) texture_view_of(in_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, VK_FORMAT_UNDEFINED, &in);
    const auto& od = st.info.image_descriptors[0];
    Texture* out = nullptr;
    const uint32_t out_handle = handle_of(s, st, od, 0);
    if (out_handle) texture_view_of(out_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, image_format(od.format), &out, level);
    if (!in || !out || !temporal_colour_pair(in->format, out->format)) return nullptr;
    // At 100% render resolution the known gameplay pass writes the picture at the scene's own size.
    const uint32_t width = out->width >> *level;
    if (width <= in->width && !(known && width == in->width && (out->height >> *level) == in->height)) return nullptr;
    return out;
}
/// (hookshot) The same for an upscale network that did not translate on this GPU (upscale_stub found its input, the largest
/// texture it samples, and its output, the largest image it writes): the Render resolution row's fixed shares (67%
/// 6676619641320ba7, 58%, 50% 72ba090c70fe2614, 33%) are cooperative-matrix networks, left out on a GPU that is not NVIDIA's or multiplies in
/// software (vk_shader.cpp, RX_VK_NETWORK_KEEP), so
/// without this DLSS and FSR would only ever see the game's 3/4. True: FSR 3.1 or the temporal upscaler wrote the output
static bool temporal_rejected_network(VkCommandBuffer cmd, uint64_t key, Texture* src, const nvn_tex* in, Texture* dst,
                                      const nvn_tex* out) {
    end_pass(cmd);
    VkFormat in_format = VK_FORMAT_UNDEFINED, out_format = VK_FORMAT_UNDEFINED;
    const VkImageView in_view = texture_view(src, in, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, &in_format);
    const bool rgba_pair = (temporal_colour_pair(src->format, dst->format) || fsr3_inputs::colour_pair(src->format, dst->format)) &&
                           dst->format == VK_FORMAT_R16G16B16A16_SFLOAT;
    const VkImageView out_view = texture_view(dst, out, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, &out_format,
                                              rgba_pair ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    // (the gameplay networks' own formats: their input is the square root of the scene colour, which the fixed shares' networks
    // sample as RGB9E5 from the B10G11R11 image it is kept in (shaders/taau_upscale.comp), the 3/4 one as B10G11R11. Decoded,
    // both are what the temporal upscaler's prep pass takes. The movies' (RGBA8) are not the scene)
    const bool packed_pair = out_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 &&
                             (in_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || in_format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32);
    if (!in_view || !out_view || (!packed_pair && !temporal_colour_pair(in_format, out_format) &&
                                !fsr3_inputs::colour_pair(in_format, out_format)))
        return false;
    // (FSR 3.1 first, as fsr3_stand_in comes before temporal_stand_in: its upscale, or its frame interpolation's inputs while
    // another upscaler draws. Without it, on a card that leaves these networks out, FSR 3.1's own Render resolution steps
    // were a plain stretch: issue #1829)
    const bool fsr3 = fsr3_upscale(cmd, key, false, src, in_view, dst, out_view, 0);
    const bool done = fsr3 || ((temporal_wanted() || temporal_off() || temporal_fg_inputs()) &&
                               temporal_upscale(cmd, key, false, src, in_view, dst, out_view, 0));
    bound().reset();
    if (!done) return false;
    full_barrier(cmd);
    texture_written(dst);
    static std::mutex mu;
    static std::unordered_set<uint64_t> seen;
    std::lock_guard lk(mu);
    if (seen.size() < 32 && seen.insert(key).second)
        logf("vk: network %016llx does not translate here: %s draws its upscale (%ux%u -> %ux%u)", (unsigned long long)key,
             fsr3 ? "FSR 3.1"
                  : temporal_mode() == upscaler_rows::FSR3 ? "FSR 3.1 (on the temporal chain)" : "the temporal upscaler",
             src->width, src->height, dst->width, dst->height);
    return true;
}
/// (hookshot) After the gameplay upscale wrote its output (the game's network, or DLSS in its place: stood_in),
/// with the Item wheel row on: that picture before the game draws its HUD on it, the wheel's backdrop
/// (vk_framegen.h framegen_wheel_scene). With the row off (the default) it returns at once
void wheel_scene_note(VkCommandBuffer cmd, const NvnState& s, const Program& p, bool stood_in) {
    if (!framegen_wheel_on() || p.stages.empty()) return;
    const Stage& st = p.stages[0];
    // (the game's gameplay upscales: the 4/3 network and the full-size pass by their keys, the Render resolution row's fixed
    // shares by their shape, a network sampling one texture into one image, as dlss_stand_in takes them)
    if (!stood_in && !gameplay_network(p.key) &&
        !(st.info.uses_cooperative_matrix && st.info.texture_descriptors.size() == 1 && st.info.image_descriptors.size() == 1))
        return;
    if (st.info.image_descriptors.empty()) return;
    const auto& od = st.info.image_descriptors[0];
    Texture* out = nullptr;
    uint32_t level = 0;
    const uint32_t handle = handle_of(s, st, od, 0);
    if (handle) texture_view_of(handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, image_format(od.format), &out, &level);
    if (!out || !out->image || !(out->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || out->samples != VK_SAMPLE_COUNT_1_BIT) return;
    level = std::min(level, out->levels - 1);
    framegen_wheel_scene(cmd, out->image, out->format, std::max(1u, out->width >> level), std::max(1u, out->height >> level), level);
}
bool taau_stand_in(VkCommandBuffer cmd, const NvnState& s, const Program& p) {
    if (!taau_key(p.key) || !dev.push_descriptor || !vkCmdPushDescriptorSetKHR || p.stages.empty()) return false;
    const Stage& st = p.stages[0];
    if (st.info.texture_descriptors.empty() || st.info.image_descriptors.empty()) return false;
    size_t ntex = 0;   // (build_descriptors' image infos: the textures', then the images')
    for (const auto& d : st.info.texture_descriptors) ntex += d.count;
    if (dw().imgs.size() <= ntex || !dw().imgs[0].imageView || !dw().imgs[ntex].imageView) return false;
    const auto& od = st.info.image_descriptors[0];
    Texture* out = nullptr;
    uint32_t level = 0;
    const uint32_t handle = handle_of(s, st, od, 0);
    if (!handle) return false;
    texture_view_of(handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, image_format(od.format), &out, &level);
    if (!out) return false;
    VkImageView input_view = dw().imgs[0].imageView, output_view = dw().imgs[ntex].imageView;
    if (out->format == VK_FORMAT_R16G16B16A16_SFLOAT && scene_tonemap_on()) {
        const uint32_t input_handle = handle_of(s, st, st.info.texture_descriptors[0], 0);
        Texture* input = nullptr;
        if (!input_handle) return false;
        texture_view_of(input_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, VK_FORMAT_UNDEFINED, &input);
        if (!input || input->format != VK_FORMAT_R16G16B16A16_SFLOAT) return false;
        input_view = texture_view_of(input_handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample, VK_FORMAT_R16G16B16A16_SFLOAT);
        output_view = texture_view_of(handle, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage, VK_FORMAT_R16G16B16A16_SFLOAT);
        if (!input_view || !output_view) return false;
    } else if (out->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
        return false;
    }
    const char* keys = getenv("RX_VK_TAAU_KEYS") ? getenv("RX_VK_TAAU_KEYS") : "fd029a317a9a4b94";
    return typed_upscale(cmd, input_view, output_view, out, level, p.key, keys, true);
}

// ---- resources: the pipelines the game binds (RX_VK_RES_LOG=1), and compute programs' shader modules let go ----
namespace {
std::mutex res_use_mu;
std::unordered_set<VkPipeline> res_used;
}
void res_note_pipeline_use(VkPipeline pl) {
    std::lock_guard lk(res_use_mu);
    res_used.insert(pl);
}
size_t res_pipelines_used() {
    std::lock_guard lk(res_use_mu);
    return res_used.size();
}
/// RX_VK_CS_MODULE_DROP: a compute program's shader module is let go once its pipeline is made (nothing reads it after: the
/// dispatch builds from it only while compute_pipeline is null). The driver keeps each module's SPIR-V; 0: kept, as before
bool cs_module_drop() {
    static const bool v = env_on("RX_VK_CS_MODULE_DROP");   // (off by default here: =1 on)
    return v;
}
std::atomic<int64_t> cs_modules_dropped{0};
std::atomic<int64_t> cs_cache_hits{0}, cs_cache_misses{0};   // (compute pipelines built in play: in the pipeline cache or not)
void res_pipeline_counts(size_t* graphics, size_t* used, int64_t* modules_dropped) {
    *graphics = pipelines.size();   // (the replay's map, read for a count only)
    *used = res_pipelines_used();
    *modules_dropped = cs_modules_dropped.load();
    logf("vk: compute pipelines built in play: %lld from the pipeline cache (or under 30 ms), %lld compiled", (long long)cs_cache_hits.load(),
         (long long)cs_cache_misses.load());
    if (const int64_t z = null_writes[kNwAddressZero].load(), nb = null_writes[kNwNotBuffer].load(), np = null_writes[kNwNotBound].load(); z + nb + np)
        logf("vk: written storage buffers that resolved to nothing (RX_VK_NULL_WRITE %s): %lld in all, address 0 %lld, not a buffer %lld, pool not bound %lld",
             null_write_on() ? "on: scratch bound" : "off", (long long)(z + nb + np), (long long)z, (long long)nb, (long long)np);
}
void cs_module_release(Program& p) {
    if (!cs_module_drop() || !p.compute || !std::atomic_ref<VkPipeline>(p.compute_pipeline).load(std::memory_order_acquire) ||
        p.stages.empty() || !p.stages[0].module)
        return;
    vkDestroyShaderModule(dev.device, p.stages[0].module, nullptr);
    p.stages[0].module = VK_NULL_HANDLE;
    const int64_t total = ++cs_modules_dropped;
    if (total <= 4 || total % 128 == 0)
        logf("vk: compute modules: %lld released after pipeline creation", (long long)total);
}

bool precompile_wanted() {
#if defined(HOOKSHOT_RELEASE)
    bool on = true;
#else
    bool on = false;
#endif
    if (const char* e = getenv("RX_VK_PRECOMPILE")) on = atoi(e) != 0;
    return on;
}
namespace {
std::atomic<bool> prepare_halt{false};
std::atomic<bool> prepare_saving_now{false};   // (precompile_warm: the pipeline cache written out, prepare_saving)
}
void prepare_stop() { prepare_halt.store(true); }
bool prepare_stopped() { return prepare_halt.load(std::memory_order_relaxed); }
bool prepare_saving() { return prepare_saving_now.load(std::memory_order_relaxed); }
std::string prepared_path() { return cache_sets_switch(), cache_path("hookshot_prepared.bin"); }   // (gpu_sets.h first)
// (hookshot) What the last prepare here was made for, in hookshot_prepared.bin: the translator (program_cache_stamp), and since the
// seed grows apart from it (4221050d: networks, the movie upscaler, the particle sets, 19 compute programs more) the seed it went
// through (hookshot_shaders.bin), by a hash of its contents. By the translator alone, a folder prepared before such an update never
// prepared the new programs: each was translated in play when first met, 200-400 ms apiece. The seed is 500 MB and more: its hash is
// taken again only when its size or write time is not the recorded one, and a hash that is still the same records the new time.
// A write time says nothing of a write that came within its resolution of the one recorded (whole seconds from libstdc++'s
// std::filesystem on Windows, 2 s on FAT, a timer tick at best): as git does for its index, the shortcut is only taken for a seed
// written well before the record was (2 s); one written closer to it is hashed
// And "prepared" is two things, so the record has a sixth word. The stamp: every seed program is in the program cache, good for
// this translator. warmed: the driver's part of them was made, good for this driver and for what that part is (warmed_stamp).
// With the stamp alone the driver's part was made only together with translation, so never for a program kept already: a player
// who prepared without stage libraries and then got them had "0 new, 0 stage libraries", and after a driver update, which
// empties the driver's cache, nothing made them again; each was then compiled in the draw that first needed it. A record of
// before the sixth word reads as warmed for nothing
namespace {
struct PreparedFor {
    uint64_t stamp = 0, seed_hash = 0, seed_size = 0;
    int64_t seed_time = 0, recorded_at = 0;   // (Windows: 100 ns FILETIME ticks; elsewhere: filesystem-clock nanoseconds)
    bool timed = false;   // (recorded_at read: records of four words had none)
    uint64_t kept = 0;   // (the program keys the cache held when the prepare ended; 0: a record from before it was written)
    uint64_t warmed = 0;   // (what the driver's part was made for: warmed_stamp. 0: not made, or a record of five words)
};
#ifdef _WIN32
constexpr int64_t kSeedTimeSlack = 20'000'000;   // (two seconds in FILETIME ticks)
#else
constexpr int64_t kSeedTimeSlack = 2'000'000'000;
#endif
enum class PrepareWhy { None, First, Translator, Seed, Cache };
/// The seed's size and write time (false: there is none). Windows keeps native 100 ns FILETIME ticks: converting that
/// epoch to nanoseconds overflows int64_t. Other platforms keep filesystem-clock nanoseconds
bool seed_file_id(const std::string& path, uint64_t* size, int64_t* time) {
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &a) || (a.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) return false;
    *size = (uint64_t)a.nFileSizeHigh << 32 | a.nFileSizeLow;
    *time = (int64_t)((uint64_t)a.ftLastWriteTime.dwHighDateTime << 32 | a.ftLastWriteTime.dwLowDateTime);
#else
    std::error_code ec;
    const auto fsize = std::filesystem::file_size(path, ec);
    if (ec) return false;
    const auto ftime = std::filesystem::last_write_time(path, ec);
    if (ec) return false;
    *size = (uint64_t)fsize;
    *time = (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(ftime.time_since_epoch()).count();
#endif
    return true;
}
/// Now, on the clock seed_file_id's times are of
int64_t file_clock_now() {
#ifdef _WIN32
    FILETIME t;
    GetSystemTimeAsFileTime(&t);
    return (int64_t)((uint64_t)t.dwHighDateTime << 32 | t.dwLowDateTime);
#else
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::filesystem::file_time_type::clock::now().time_since_epoch()).count();
#endif
}
/// The seed's contents, hashed four words at a time: ~0.1 s for its 540 MB read from memory, 0.3-0.6 s from the disk (a byte at a
/// time, as hash_bytes, 0.45 s from memory). Only after the file was written again
bool seed_file_hash(const std::string& path, uint64_t* out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    constexpr uint64_t p1 = 0x9E3779B185EBCA87ull, p2 = 0xC2B2AE3D27D4EB4Full;
    const auto mix = [](uint64_t h, uint64_t w) {
        h += w * p2;
        return ((h << 31) | (h >> 33)) * p1;
    };
    uint64_t lane[4] = {p1 + p2, p2, 0, 0 - p1}, total = 0, tail = 0xCBF29CE484222325ull;
    std::vector<uint8_t> buf(1 << 20);   // (a multiple of 32: only the last read leaves bytes over)
    for (size_t n; (n = fread(buf.data(), 1, buf.size(), f)) > 0;) {
        total += n;
        size_t at = 0;
        for (; at + 32 <= n; at += 32) {
            uint64_t w[4];
            std::memcpy(w, buf.data() + at, 32);
            for (int k = 0; k < 4; k++) lane[k] = mix(lane[k], w[k]);
        }
        for (; at < n; at++) tail = (tail ^ buf[at]) * 1099511628211ull;
    }
    const bool ok = !ferror(f);
    fclose(f);
    uint64_t h = total * p1 ^ tail;
    for (const uint64_t l : lane) h = mix(h ^ mix(0, l), p1);
    *out = h;
    return ok;
}
/// The record: 0 none, 1 the translator alone (written before the seed was), 2 both. A record without the time it was written
/// at (four words) never takes the shortcut: its seed is hashed, and the record written again with it
int prepared_read(const std::string& path, PreparedFor* had) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return 0;
    uint64_t r[7]{};
    const size_t n = fread(r, 8, 7, f);
    fclose(f);
    had->stamp = r[0], had->seed_hash = r[1], had->seed_size = r[2], had->seed_time = (int64_t)r[3];
    had->recorded_at = (int64_t)r[4], had->timed = n >= 5;
    had->warmed = n >= 6 ? r[5] : 0;
    had->kept = n >= 7 ? r[6] : 0;   // (hookshot: the program keys the cache held when the prepare ended)
    return n >= 4 ? 2 : n >= 1 ? 1 : 0;
}
void prepared_write(const std::string& path, const PreparedFor& now) {
    const uint64_t r[7] = {now.stamp, now.seed_hash, now.seed_size, (uint64_t)now.seed_time, (uint64_t)file_clock_now(), now.warmed,
                           now.kept};
    if (FILE* f = fopen(path.c_str(), "wb")) fwrite(r, 8, 7, f), fclose(f);
}
/// Whether to prepare (and why), and what for: *now, written once the prepare is done. No seed: the translator alone, as before.
/// *hash_ms: how long the seed's hash took (0: not taken). now->warmed is the record's, for this translator: the caller's to renew
PrepareWhy prepare_check(const std::string& path, const std::string& seed, uint64_t stamp, PreparedFor* now, double* hash_ms) {
    *hash_ms = 0;
    PreparedFor had;
    const int read = prepared_read(path, &had);
    now->stamp = stamp;
    now->kept = had.kept;
    now->warmed = had.stamp == stamp ? had.warmed : 0;   // (kept where the record is only written again for the seed's new time)
    const bool have_seed = seed_file_id(seed, &now->seed_size, &now->seed_time);
    bool same_seed = !have_seed;
    if (have_seed && read == 2 && had.seed_size == now->seed_size && had.seed_time == now->seed_time &&
        had.timed && had.seed_time < had.recorded_at - kSeedTimeSlack) {
        now->seed_hash = had.seed_hash, same_seed = true;   // (not read again)
    } else if (have_seed) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool hashed = seed_file_hash(seed, &now->seed_hash);
        *hash_ms = std::max(1e-3, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        same_seed = hashed && read == 2 && had.seed_hash == now->seed_hash &&
                    had.seed_size == now->seed_size;
        // (rewritten as it was, by a build: the new time kept, so the next start does not hash it again; a seed written just
        // before the record is hashed again until a record is written 2 s after it)
        if (same_seed && had.stamp == stamp) prepared_write(path, *now);
    }
    if (!read) return PrepareWhy::First;
    if (had.stamp != stamp) return PrepareWhy::Translator;
    if (!same_seed) return PrepareWhy::Seed;
    // (the record says prepared, but the program cache may not have kept it: a start with other shader settings (RX_VK_OOT_*, ...,
    // program_cache_stamp) makes the cache again under its stamp, and closed before it prepared, the next start under the
    // usual settings found an empty cache and a record that said prepared: every program then translated in play, 200-400 ms
    // each. Less than half of what the last prepare left, or none for a record from before the count: prepared again)
    const size_t kept = program_cache_kept();
    if (kept != SIZE_MAX && (had.kept ? kept < had.kept / 2 : kept == 0)) return PrepareWhy::Cache;
    return PrepareWhy::None;
}
} // namespace
namespace {
std::atomic<bool> lazy_update{false};
/// The guard of a lazy update: ninety-odd of the seed's graphics programs, evenly spread, are translated now (the program cache they go to
/// is the new translator's, empty), and how many have a fingerprint the ledger has not seen is counted. *measured: how many were
/// translated (the others were kept already, from an earlier start of this update, or do not translate)
double update_changed_share(size_t* measured) {
    *measured = 0;
    const size_t programs = seed_open();
    if (!programs) return 0;
    std::vector<uint64_t> fps;
    fingerprint_probe = &fps;
    const size_t want = 96, stride = std::max<size_t>(1, programs / want);
    for (size_t i = stride / 2; i < programs && fps.size() < want; i += stride) {
        if (seed_is_compute(i)) continue;
        if (std::unique_ptr<Program> p = seed_program(i, 0))
            for (Stage& st : p->stages)
                if (st.module) vkDestroyShaderModule(dev.device, st.module, nullptr);
    }
    fingerprint_probe = nullptr;
    size_t changed = 0;
    for (const uint64_t fp : fps) {
        if (!ledger_has(ledger_id_seen(fp))) changed++;
        ledger_add(ledger_id_seen(fp));
    }
    seed_close();
    *measured = fps.size();
    return fps.empty() ? 0 : (double)changed / fps.size();
}
}  // namespace
bool prepare_is_lazy_update() { return lazy_update.load(); }
/// What the driver's part is good for: these programs, this driver on this GPU (its version too: the cache's UUID is the driver's
/// to choose), with the stage libraries or without, and what warm_libraries builds of them (kWarmRecipe, to be counted up when that
/// changes: vk_draw.cpp is not in the translator's stamp)
constexpr uint32_t kWarmRecipe = 2;   // (2: the libraries are in our pipeline cache)
uint64_t warmed_stamp(bool libraries) {
    uint64_t h = program_cache_stamp();
    const auto add = [&h](const void* d, size_t n) {
        for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)d)[i]) * 1099511628211ull;
    };
    add(dev.props.pipelineCacheUUID, VK_UUID_SIZE);
    const uint32_t v[] = {dev.props.vendorID, dev.props.deviceID, dev.props.driverVersion,
                          libraries ? (gpl_libs_cached() ? 1u : 3u) : 0u, kWarmRecipe};
    add(v, sizeof v);
    return h;
}
/// (hookshot) An update: every shader was prepared once here, for another translator. Each release changes the translator, and every
/// update then held the game's start for the whole seed again (116k variants, minutes on most PCs: "shaders need to be prepared every
/// update"). Now only what the notes name is built before the game (precompile_run), the rest in play behind it (prepare_background):
/// with a lazy update (RX_VK_SHADER_KEY on, see below) nothing is translated before the game, a program a draw needs is translated by that
/// draw, and the rest behind it, the driver given only what its fingerprint says it has not built
bool prepare_is_update() {
    // (opt-in, RX_VK_PREPARE_BACKGROUND=1: when the translator changed, the program cache is gone too, and the first area then
    // translated thousands of programs in play: 1 fps and black frames for minutes, which looked broken. Before the game is the default)
    if (env_on("RX_VK_PREPARE_AGAIN") || !env_on("RX_VK_PREPARE_BACKGROUND")) return false;
    FILE* f = fopen(prepared_path().c_str(), "rb");
    if (!f) return false;
    uint64_t have = 0;
    const bool read = fread(&have, 8, 1, f) == 1;   // (the record's first word, the translator's stamp: see PreparedFor)
    fclose(f);
    return read && have != program_cache_stamp();
}
/// (hookshot) The driver's part of the graphics and mesh programs: the two shader libraries of every kept program variant, whoever
/// made it (the seed, a state list, play), each built once and let go, for the driver's own cache (see prepare_run). It walks the
/// program cache, not the seed, so it needs no translation: the same pass serves a first prepare, stage libraries switched on
/// later and a new driver. The programs the notes name go first (what play is known to use: a pass that is closed, or a driver
/// cache that fills, has those), the rest in the file's order. False: stopped (the window was closed)
/// The pipeline cache written now, and whether the file then holds what the cache did (a warm pass's libraries have to be on disk
/// before the stamp says they are: a start that ended before the save would otherwise find them cold for good)
bool pcache_saved_now() {
    size_t before = 0;
    // (RX_VK_PCACHE_LOAD=0: the save does nothing, and the file left from before, read against the cache that started empty,
    // would pass for it: the record is not written as done)
    if (!pcache || pcache_no_save) return false;
    {
        PcacheFetch fetch(-1);
        if (vkGetPipelineCacheData(dev.device, pcache, &before, nullptr) != VK_SUCCESS) return false;
    }
    pipeline_cache_save(true);
    std::error_code ec;
    const auto size = std::filesystem::file_size(pcache_path, ec);
    return !ec && size >= before;
}
/// (hookshot) The threads a preparation step builds graphics pipelines on: want, or where Intel's compiler takes them side by
/// side only while its gate holds play's (intel_prepare_open: an Arc A-series or Core Ultra on Auto, driver 101.9175 or
/// newer), as many as the free memory allows (intel_compiles.h intel_prepare_workers: 16 workers side by side on a 16 GB
/// A730M ran into the commit limit). Said once a step
unsigned prepare_pipeline_threads(unsigned want, const char* step) {
    if (!intel_prepare_open()) return want;
    const uint64_t avail = memory_available_mb();
    const unsigned n = intel_prepare_workers(want, avail == compute_gate::kUnknownMB ? UINT64_MAX : avail);
    logf("vk: preparation (%s): Intel's compiler takes the graphics pipelines side by side on %u threads of %u (%llu MB "
         "free)", step, n, want, avail == compute_gate::kUnknownMB ? 0ull : (unsigned long long)avail);
    return n;
}
bool warm_libraries(void (*progress)(size_t done, size_t total)) {
    const auto t0 = std::chrono::steady_clock::now();
    pcache_init();   // (before the workers: the libraries are built into it, and the start that only warms has not made it yet)
    // (a state list's programs are in memory by now, state_list_prepare read them in for the precompile: those as they are)
    std::vector<Program*> loaded;
    std::vector<uint64_t> keys = program_cache_keys(&loaded);
    size_t named = loaded.size();
    {
        std::unordered_set<uint64_t> by_notes;
        for (const auto& n : pipe_notes_read()) by_notes.insert(n->prog);
        named += std::stable_partition(keys.begin(), keys.end(), [&](uint64_t k) { return by_notes.count(k) != 0; }) - keys.begin();
    }
    // (a variant a state list made is in memory and has a record too, seed_variant_of: once)
    std::unordered_set<uint64_t> in_memory;
    for (const Program* p : loaded) in_memory.insert(mix(p->key, p->variant));
    auto blank = std::make_unique<NvnState>();
    blank->nvattr = 16;
    const size_t total = loaded.size() + keys.size();
    std::atomic<size_t> next{0}, done{0}, libs{0}, variants{0};
    const auto warm = [&](Program& p) {
        // (a fallback program's pipelines are never linked from libraries: graphics_pipeline)
        if (p.failed || p.compute || p.fallback || p.stages.empty()) return;
        ensure_layout(p);
        Targets at{};
        for (int part = 1; part <= 2; part++)
            if (const VkPipeline lib = build_graphics(p, *blank, at, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, part)) {
                vkDestroyPipeline(dev.device, lib, nullptr);
                libs++;
            }
        variants++;
    };
    const auto worker = [&] {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);   // (as prepare_run's, and half the CPUs)
        const IntelPrepareLane lane;   // (a preparation thread: vk_internal.h)
        for (;;) {
            if (prepare_stopped()) return;
            const size_t i = next++;
            if (i >= total) return;
            if (i < loaded.size()) warm(*loaded[i]);
            else
                program_cache_borrow(keys[i - loaded.size()], [&](Program& p) {
                    if (!in_memory.count(mix(p.key, p.variant))) warm(p);
                });
            done++;
        }
    };
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < std::clamp(half_cpus(), 1u, 32u); i++) threads.emplace_back(worker);
    while (done.load() < total && !prepare_stopped()) {
        if (progress) progress(done.load(), total);
        Sleep(100);
        if (next.load() >= total + threads.size()) break;
    }
    for (std::thread& t : threads) t.join();
    if (prepare_stopped()) {   // (what was built is in the driver's cache: the next start goes through those quickly)
        logf("vk: the stage libraries stopped at %zu of %zu programs (the window was closed): the next start makes them again",
             done.load(), total);
        return false;
    }
    logf("vk: warmed %zu stage libraries of %zu kept shader program variants (%zu programs, the %zu the notes name first) in %.1f s",
         libs.load(), variants.load(), total, named, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (gpl_libs_cached()) {
        const auto t1 = std::chrono::steady_clock::now();
        const bool saved = pcache_saved_now();
        std::error_code ec;
        const auto size = std::filesystem::file_size(pcache_path, ec);
        logf("vk: the pipeline cache with the stage libraries: %s, %llu MB, in %.1f s", saved ? "saved" : "not saved: warmed again at the next start",
             (unsigned long long)(ec ? 0 : size >> 20), std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
        if (!saved) return false;
    }
    return true;
}
/// (hookshot) Every kept compute program's variants given to warm, each variant an item of its own on every CPU (prepare_run, where
/// the driver's compute compiles are gated): a thread takes the next program, borrows all its variants at once and queues them,
/// and every thread takes queued variants before it takes another program, so a program's variants are compiled side by side
/// and only a few programs are read in at once. The gate (compute_gate.h) decides how many compiles run. progress: in programs.
/// only: the programs to warm, the precompile's plan (computes_plan), null for every kept one. The programs done
size_t warm_computes_apart(const std::function<void(size_t done, size_t total)>& progress, const std::function<void(Program&)>& warm,
                           const std::unordered_set<uint64_t>* only) {
    std::vector<uint64_t> keys = program_cache_compute_keys();
    if (only) keys.erase(std::remove_if(keys.begin(), keys.end(), [&](uint64_t k) { return !only->count(k); }), keys.end());
    struct Batch {
        size_t left = 0;
    };
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<Program*, Batch*>> items;
    size_t open_batches = 0;
    std::atomic<size_t> next_key{0}, keys_done{0};
    // (mu held: one queued variant run, the lock let go meanwhile. False when none is queued)
    const auto run_one = [&](std::unique_lock<std::mutex>& lk) {
        if (items.empty()) return false;
        const auto [p, batch] = items.front();
        items.pop_front();
        lk.unlock();
        warm(*p);
        lk.lock();
        if (!--batch->left) cv.notify_all();
        return true;
    };
    const auto worker = [&] {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        for (;;) {
            {
                std::unique_lock lk(mu);
                if (run_one(lk)) continue;
                if (next_key.load() >= keys.size() || prepare_stopped()) {
                    if (!open_batches) return;   // (nothing queued, nothing more to come)
                    cv.wait_for(lk, std::chrono::milliseconds(100));
                    continue;
                }
            }
            const size_t k = next_key++;
            if (k >= keys.size()) continue;
            program_cache_borrow_all(keys[k], [&](std::vector<std::unique_ptr<Program>>& variants) {
                Batch batch;
                std::unique_lock lk(mu);
                for (auto& p : variants)
                    if (p->compute && !p->failed && !p->stages.empty() && p->stages[0].module) items.push_back({p.get(), &batch}), batch.left++;
                open_batches++;
                cv.notify_all();
                while (batch.left)   // (its own variants and others', until its own are compiled: they die when this returns)
                    if (!run_one(lk)) cv.wait_for(lk, std::chrono::milliseconds(100));
                open_batches--;
                cv.notify_all();
            });
            keys_done++;
        }
    };
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < std::max(1u, usable_cpus()); i++) threads.emplace_back(worker);
    for (;;) {
        if (progress) progress(keys_done.load(), keys.size());
        {
            std::unique_lock lk(mu);
            if (keys_done.load() >= keys.size() || (prepare_stopped() && !open_batches)) break;
        }
        compute_compiles_log();
        Sleep(100);
    }
    for (std::thread& t : threads) t.join();
    if (progress) progress(keys_done.load(), keys.size());
    return keys_done.load();
}

/// Every shader of the game, before its first start (vk_shader.cpp, the seed): translated once for this translator, and the
/// driver's part made once for this driver (see PreparedFor). True: prepared now, either of them.
/// background: in play, behind the game: a few workers at idle priority, translation only (the driver's part is left to first use:
/// its compiler crashing in here would take the game with it)
/// refresh_states: Builder preparation also imports state lists when the seed is already prepared
bool prepare_run(void (*progress)(size_t done, size_t total), bool background, bool refresh_states) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string path = prepared_path();
    PreparedFor prepared_for;
    double hash_ms = 0;
    const PrepareWhy why = prepare_check(path, seed_path(), program_cache_stamp(), &prepared_for, &hash_ms);
    // (hookshot) the pipeline cache's files dropped at this start (pcache_drop_stale: another translator's, or a folder from
    // before the stamp file) took the stage libraries the warm pass had kept in them: made again, whatever the record says,
    // once a run, by the start's own preparation (the background one in play leaves the driver's part to first use)
    static std::atomic<bool> rewarm_once{true};
    if (!background && pcache_drop_stale() && rewarm_once.exchange(false)) prepared_for.warmed = 0;   // (read as not made)
    const bool libraries = gpl_wanted();
    const uint64_t warmed = warmed_stamp(libraries);
    const std::string warming = path + ".warming";
    if (hash_ms > 0)
        logf("vk: the shader list was hashed (%.0f MB in %.0f ms): written since the last check, or too close to it to go by its time",
             prepared_for.seed_size / 1048576.0, hash_ms);
    if (why == PrepareWhy::None && !env_on("RX_VK_PREPARE_AGAIN")) {
        // A Builder run can add pipeline states without changing the shader seed.
        if (env_on("RX_VK_SEED_CHECK")) {
            if (seed_open() && refresh_states) state_list_prepare(progress);
        } else if (refresh_states) {
            state_list_prepare(progress, true);
        }
        if (prepared_for.warmed == warmed || background) return false;
        // (translated, and the driver's part is another driver's, or of before the libraries were wanted: that part alone, no
        // shader translated. Without libraries it is predict_run's pipelines, which follow a prepare that prepared)
        logf("vk: the shaders are prepared, but their driver's part was made for another driver or other settings: made again%s",
             libraries ? "" : " (no stage libraries: the predicted pipelines alone)");
        if (libraries) {
            if (FILE* f = fopen(warming.c_str(), "rb")) {
                fclose(f);
                logf("vk: the last start did not finish the driver's part of the shaders: left out");
            } else {
                crash_markers::begin(warming, 1, crash_markers::none);
                if (!warm_libraries(progress)) {
                    crash_markers::end(warming);
                    return false;
                }
            }
            crash_markers::end(warming);
        }
        prepared_for.warmed = warmed;
        prepared_write(path, prepared_for);
        return true;
    }
    // (the translator is the same: what the program cache keeps stays, and seed_translate and seed_compute leave it out. The
    // programs the seed has gained are what is made)
    if (why == PrepareWhy::Seed)
        logf("vk: the game's shader list (hookshot_shaders.bin) is not the one the shaders were prepared from: preparing again "
             "(only the programs not kept yet)");
    if (why == PrepareWhy::Cache)
        logf("vk: the program cache holds %zu program keys of the %llu the last preparation left: preparing again",
             program_cache_kept(), (unsigned long long)prepared_for.kept);
    const size_t programs = seed_open();
    if (!programs) return false;
    // (the record goes while the shaders are prepared and is written again when they are: a preparation closed or killed part
    // way left the old record standing, and the next start trusted it whenever the cache held more than half of what it had held,
    // or anything at all under a record from before the count: thousands of programs were then translated in play. With no
    // record the next start prepares again, and only what the cache does not hold yet)
    remove(path.c_str());
    const size_t total = programs * 3;   // (each program for float, one and two integer bone inputs)
    // The driver's part: each program's stage libraries, built once and let go, for the driver's own disk cache. What stays with us is
    // nothing: a kept library costs memory, and our pipeline cache would be read in whole at every start. A driver whose compiler
    // crashes on one of them (it happens: TotK Recomp's issue 30) would crash here at every start: the file says this step began and
    // did not end, and the next start goes without it. Those libraries are then built when a draw first needs them
    bool warm = libraries && !background;
    bool warm_cs = !background;
    if (background) {
    } else if (FILE* f = fopen(warming.c_str(), "rb")) {
        fclose(f);
        warm = warm_cs = false;
        logf("vk: the last start did not finish the driver's part of the shaders: left out");
    } else if (warm || warm_cs) {   // (a start closed in them leaves no file: crash_markers::close)
        crash_markers::begin(warming, 1, crash_markers::none);
    }
    pcache_init();   // (once, here: the workers below build compute pipelines into it)
    std::atomic<size_t> next{0}, done{0}, made{0}, libs{0}, variants{0}, asking{0}, capped{0};
    // (hookshot) The compute programs first: each is its texture variants, up to 64 translations and as many driver compiles, seconds
    // to tens of seconds on one worker, where a graphics program is a few milliseconds. In the seed's order they were spread over its
    // second half, the last of them the seed's very last program: the other workers ran out of work and waited on the last few
    // (~20 s of a ~165 s prepare with fewer and fewer CPUs busy, 12 workers on a 24-thread PC). Taken first, the graphics programs
    // fill in behind them
    std::vector<uint32_t> order(programs);
    size_t computes = 0;
    {
        size_t at = 0;
        for (size_t k = 0; k < programs; k++)
            if (seed_is_compute(k)) order[at++] = (uint32_t)k;
        computes = at;
        for (size_t k = 0; k < programs; k++)
            if (!seed_is_compute(k)) order[at++] = (uint32_t)k;
    }
    // (hookshot) compute pipelines too, for the driver's own cache: built in play they stopped the game while the driver compiled
    // them, seconds each for the culling programs (53caac5f: 6.5 s, 41 s inlined), which looked like a hang
    // (hookshot) Where the driver's compute compiles are gated (Mesa's Intel driver, compute_gate.h), a program's variants are not
    // compiled here, one after the other on the worker that translated them (up to 64 of them, 15-25 s each for one held to 32
    // lanes: four such programs held four workers for half an hour), but after the translation, each variant its own item on
    // every CPU (warm_computes_apart). RX_VK_COMPUTE_WARM_APART=0 or 1 says so on any GPU
    static const bool apart = [] {
        const char* e = getenv("RX_VK_COMPUTE_WARM_APART");
        return e ? !env_value_off(e) : dev.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA;
    }();
    // Use the same selection for a compile during translation and for the separate driver pass.
    const std::unordered_set<uint64_t> none;
    const ComputePlan* plan = warm_cs ? &compute_warm_plan() : nullptr;
    const auto* only = !plan || plan->all ? nullptr : plan->none ? &none : &plan->keys;
    // (hookshot) The bar's count, in programs, each at its cost (PrepareWork, vk_progress.h): the compute programs first, with their
    // driver compiles, held it near 0% for as long as they took. Those the plan compiles count at that cost, as many as it
    // names (the seed's own keys are the translator's, not read here); the rest are translated alone. work: in thirds of a graphics
    // program; warm_credited: what this thread's compute program has counted of its compiles so far, warm_seen: whether it compiled
    const size_t warmed_computes = only ? std::min(only->size(), computes) : computes;
    const PrepareWork work_of(programs, warmed_computes, warm_cs);
    std::atomic<uint64_t> work{0};
    static thread_local uint64_t warm_credited = 0;
    static thread_local bool warm_seen = false;
    bool warming_apart = false;   // (the pass after the translation runs: warm_compute compiles)
    const auto warm_compute = [&](Program& p) {
        if (!warm_cs || taau_key(p.key) || (apart && !warming_apart)) return;
        if (!apart && only && !only->count(p.key)) return;
        ensure_layout(p);
        // (as a dispatch, a load and the precompile ask for it: the same pin, so the same pipeline in the caches)
        const VkComputePipelineCreateInfo ci = compute_pipeline_info(p, true);
        VkPipeline pl = VK_NULL_HANDLE;
        // (into the app's pipeline cache, saved to hookshot_vk_pipelines.bin: without it the next start compiled them again in play
        // where the driver's own cache missed them, AMD's 15 s culling compiles each time)
        {
            const ComputeTurn turn(ci);   // (as cached_compute: as many of the driver's compiles at once as the memory holds)
            IntelDriverGate gate;   // (inside the turn, as cached_compute)
            if (turn.ok() && guarded_compute_held(pcache, &ci, &pl) == VK_SUCCESS && pl) {
                vkDestroyPipeline(dev.device, pl, nullptr);
                libs++;
                pcache_built_since_save++;
            }
        }
        if (p.layout) vkDestroyPipelineLayout(dev.device, p.layout, nullptr), p.layout = VK_NULL_HANDLE;
        if (p.set_layout) vkDestroyDescriptorSetLayout(dev.device, p.set_layout, nullptr), p.set_layout = VK_NULL_HANDLE;
        // (compiled on the worker that translated it: each variant moves the bar, its program's share at most. The pass apart
        // counts its own, below)
        warm_seen = true;
        if (!warming_apart && warm_credited < work_of.warm_each) {
            const uint64_t step = std::min<uint64_t>(PrepareWork::kGraphics, work_of.warm_each - warm_credited);
            warm_credited += step, work += step;
        }
    };
    const auto worker = [&] {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        // (below normal, and half the CPUs, below: on all of them at High the driver's compiles froze the whole PC)
        SetThreadPriority(GetCurrentThread(), background ? THREAD_PRIORITY_IDLE : THREAD_PRIORITY_BELOW_NORMAL);
        // (a preparation thread, its modules side by side where intel_prepare_open; in play, background, as play's: vk_internal.h)
        const IntelPrepareLane lane(!background);
        for (;;) {
            if (prepare_stopped()) return;
            const size_t i = next++;
            if (i >= total) return;
            const size_t k = order[i / 3];
            if (seed_is_compute(k)) {   // (hookshot) a compute program with each of its texture variants, each warmed
                if (i % 3 == 0) {
                    warm_credited = 0, warm_seen = false;
                    const SeedVariants v = seed_compute(k, 64, warm_compute);
                    made += v.variants != 0, variants += v.variants, asking += v.variants > 1, capped += v.capped && v.variants;
                    if (!apart && warm_seen)   // (its compiles, all counted)
                        work += work_of.warm_each - std::min(warm_credited, work_of.warm_each);
                }
                done++, work++;
                continue;
            }
            // (its stage libraries are not built here any more, but by warm_libraries below, from the program cache: here
            // they were built only for a program translated now)
            const auto t_prog = std::chrono::steady_clock::now();
            std::unique_ptr<Program> seeded = seed_program(k, (int)(i % 3));
            {   // (hookshot: a program that took over a second to prepare is named, a few dozen times: the no-mesh fallback's first
                // start ran about 100 times slower than a mesh GPU's)
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prog).count();
                static std::atomic<int> slow_logged{0};
                if (ms > 1000 && slow_logged.fetch_add(1, std::memory_order_relaxed) < 40)
                    logf("vk: preparing program %016llx (part %d) took %.0f ms%s%s",
                         (unsigned long long)(seeded ? seeded->key : 0), (int)(i % 3), ms,
                         seeded && seeded->fallback ? " (mesh fallback)" : "", seeded && seeded->failed ? " (failed)" : "");
            }
            if (std::unique_ptr<Program> p = std::move(seeded)) {
                if (!p->failed) made++;
                for (Stage& st : p->stages)
                    if (st.module) vkDestroyShaderModule(dev.device, st.module, nullptr);
            }
            done++, work++;
        }
    };
    std::vector<std::thread> threads;
    const unsigned cpus = usable_cpus();
    // (in play: a quarter of the CPUs, two to four, the game's own threads before them)
    // (device_policy.h program_workers_in_play; and never more than the process may use now)
    const unsigned workers = background ? std::min(dev.policy.program_workers_in_play, cpus) : prepare_cpus();
    if (!background)
        logf("vk: preparation: %u workers for the shader programs (%u CPUs usable; %s)", workers, cpus,
             segment_heap() ? "the segment heap: as many as the free memory allows" : "half the CPUs");
    for (unsigned i = 0; i < workers; i++) threads.emplace_back(worker);
    while (done.load() < total && !prepare_stopped()) {
        if (progress) progress(work_of.done(work.load()), programs);   // (in programs' worth, as the log counts them, PrepareWork)
        compute_compiles_log();
        Sleep(100);
        if (next.load() >= total + threads.size()) break;
    }
    for (std::thread& t : threads) t.join();
    if (!background) state_list_prepare(nullptr);   // (in play it is too late for the precompile: an update's is state_list_run, before)
    if (!env_on("RX_VK_SEED_CHECK")) seed_close();
    if (prepare_stopped()) {   // (closed: what was made is in the program cache already, the rest is made at the next start)
        logf("vk: shader preparation stopped at %zu of %zu (the window was closed): the next start goes on from there", done.load(),
             total);
        // (the file as the close leaves it, crash_markers::close: a close neither counts as a crash nor clears one)
        return false;
    }
    int64_t translated = 0, failed = 0;
    shader_stats(&translated, &failed);
    logf("vk: prepared %zu shader programs%s (%zu new, %lld failed, %lld networks left out, %zu compute pipelines; %zu compute variants, "
         "%zu programs with texture variants, %zu at the limit) in %.1f s", programs, background ? " in play" : "", made.load(), (long long)failed,
         (long long)shader_left_out(), libs.load(), variants.load(),
         asking.load(), capped.load(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    shader_fail_lines();   // (the failed by their "not translated" lines, all of them: those lines stop at 30 programs each)
    if (warm_cs && apart) {
        warming_apart = true;
        const auto w0 = std::chrono::steady_clock::now();
        const size_t before = libs.load();
        // (the programs the precompile keeps ready, and no others: on Mesa's Intel driver the listed ones, precompile_computes_which.
        // The rest are compiled at their first dispatch)
        // (its count, in the programs it warms, onto the share of the bar the compute programs' compiles have: not a count of its
        // own on the programs' part, which held the bar at what the translation had reached until it passed it)
        const uint64_t translated_work = work.load(), warm_share = work_of.warm_each * warmed_computes;
        const auto apart_progress = [&](size_t d, size_t t) {
            if (!progress) return;
            work.store(translated_work + (t ? warm_share * std::min(d, t) / t : warm_share));
            progress(work_of.done(work.load()), programs);
        };
        const size_t programs_warmed = warm_computes_apart(apart_progress, warm_compute, only);
        logf("vk: preparation: the compute pipelines of %zu compute programs compiled for the driver's cache (%s): %zu variants on "
             "%u threads in %.1f s%s", programs_warmed, only ? "those the precompile keeps" : "every kept one",
             libs.load() - before, usable_cpus(), std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(),
             prepare_stopped() ? "; stopped before the end" : "");
        // (closed in it: the file as the close leaves it, crash_markers::close, as for the programs above: a close neither counts
        // as a crash nor clears one)
        if (prepare_stopped()) return false;
    }
    if (warm) {
        // (translated, whatever becomes of the libraries: a start that is closed in them goes on with them alone)
        prepared_for.warmed = 0;
        prepared_write(path, prepared_for);
        if (!warm_libraries(progress)) {
            crash_markers::end(warming);
            return false;
        }
    }
    descriptor_arrays_log(background ? "the preparation in play" : "the preparation");   // (the programs' indexed arrays: vk_programs.cpp)
    if (!background) crash_markers::end(warming);
    // (the driver's part left out, by a start that ended in it or in play, is not tried again for this driver: as before)
    prepared_for.warmed = warmed;
    if (const size_t kept = program_cache_kept(); kept != SIZE_MAX) prepared_for.kept = kept;
    prepared_write(path, prepared_for);
    return true;
}
bool state_list_pending() { return shipped_list_pending(); }
/// (hookshot) An update (prepare_is_update): the seed is prepared in play, behind the game, too late for the precompile; the pipelines
/// of the state list are made now, before it, so that precompile_run builds them before the game as on a first start
void state_list_run() {
    state_list_prepare(nullptr, true);
}
void prepare_background() {
    std::thread([] {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
        Sleep(20000);   // (past the title screen's own loading)
        logf("vk: an update: the shaders not met yet are translated in play, behind the game");
        prepare_run(nullptr, true);
    }).detach();
}

/// (hookshot) A line every 10 s while the preparation builds pipelines: their count and an estimate of the time left. The
/// pipeline steps wrote nothing to the log for 10-30 minutes on some AMD cards, and players pasted a log whose last lines (the
/// vertex formats) looked like a hang (Discord 2026-11-10, "stuck on preparing shaders"); the Builder's bar moved, the log did not
/// (hookshot) The pipeline cache written while a warm builds, every kWarmSaveEvery, not only at its end. A preparation on a cold
/// driver cache is hours of compiles (AMD: nightly-622's, 80 minutes in and going, with no hookshot_vk_pipelines.bin written yet),
/// and one killed, crashed, or stopped before its end (make_game's silence timeout, the player closing it) kept none of them: the
/// file was written once, after the last build. Now what was built up to the last save is in the file, and the next start's warm
/// finds it in the cache. warm_save_due: whether a save is due at `now` (the first call only sets the next one); warm_save_tick: a
/// background save when one is (pipeline_cache_save(false): its own thread, the builds go on, nothing written when nothing new)
constexpr auto kWarmSaveEvery = std::chrono::minutes(10);
bool warm_save_due(std::chrono::steady_clock::time_point now, std::chrono::steady_clock::time_point& next) {
    if (next == std::chrono::steady_clock::time_point{}) {
        next = now + kWarmSaveEvery;
        return false;
    }
    if (now < next) return false;
    next = now + kWarmSaveEvery;
    return true;
}
void warm_save_tick(std::chrono::steady_clock::time_point& next) {
    if (warm_save_due(std::chrono::steady_clock::now(), next)) pipeline_cache_save(false);
}

void prepare_progress_log(const char* what, size_t done, size_t total, std::chrono::steady_clock::time_point t0,
                          std::chrono::steady_clock::time_point& next) {
    const auto now = std::chrono::steady_clock::now();
    if (now < next || !total) return;
    next = now + std::chrono::seconds(10);
    const double s = std::chrono::duration<double>(now - t0).count();
    const double left = done ? s * (double)(total - std::min(done, total)) / (double)done : 0.0;
    if (done) logf("vk: %s: %zu of %zu pipelines in %.0f s, about %.0f s left", what, done, total, s, left);
    else logf("vk: %s: 0 of %zu pipelines in %.0f s (the driver compiles the first ones)", what, total, s);
    compute_compiles_log();
}
/// (hookshot) The notes' references, walked once a start. notes_lost_by_driver, precompile_warm and precompile_at_create each
/// walked hookshot_pipelines.bin for them (37,474 notes in 1.0 GB on one PC: about 0.25 s a walk with the file in the OS's
/// cache), two walks on a normal start and three when the driver had lost the notes. All three run one after
/// the other on precompile_run's thread, before the game, and none of them writes the notes file: build_graphics notes nothing
/// (pipeline_note is play's, graphics_pipeline, and the state list's, state_list_prepare, which runs before precompile_run), and
/// the file is opened for appending (notes_open_locked) in precompile_at_create, after its references are taken. The references
/// are kept with the file's size and time as they were before the walk, and a use that finds either changed (a file grown,
/// cut or rewritten by anything not foreseen here) walks again: a stale place is never handed out. A walk is kept only when it
/// read the whole file and the file's size, time and head could be read (not when the file was missing, held by another
/// program, or a read or seek failed): a walk that failed is not handed to the next use, and a use that cannot read the file
/// walks again. Not seen: a rewrite at the same size that keeps the time (another program, or
/// within one tick of a coarse file system); main's references, kept for the session, were as exposed. take: the last use, which
/// takes them (precompile_at_create, for programs_index); the next use after it walks again
struct StartNoteRefs {
    std::vector<NoteRef> refs;
    int64_t size = -1, time = 0;   // (the file's, before the walk; size -1: there was none)
    bool kept = false;
    int walks = 0;                 // (this start's, said in the precompile's lines)
    double walk_ms = 0;
};
StartNoteRefs start_note_refs_kept;   // (precompile_run's thread alone)
static void notes_file_id(int64_t* size, int64_t* time) {
    std::error_code ec, ec_time;
    const std::string path = pipe_notes_path();
    const uintmax_t n = std::filesystem::file_size(path, ec);
    const auto t = std::filesystem::last_write_time(path, ec_time);
    // (and whether its head reads, as notes_open_locked reads it: a file another program has since locked against reads is
    // walked again, as main's walk at each use saw it)
    uint64_t head[2];
    FILE* f = fopen(path.c_str(), "rb");
    const bool reads = f && fread(head, 8, 2, f) == 2;
    if (f) fclose(f);
    *size = ec || ec_time || !reads ? -1 : (int64_t)n;   // (-1: no file, or one not read; a walk is never kept on it)
    *time = ec_time ? 0 : (int64_t)t.time_since_epoch().count();
}
std::vector<NoteRef> start_note_refs(bool take) {
    StartNoteRefs& k = start_note_refs_kept;
    int64_t size = -1, time = 0;
    notes_file_id(&size, &time);
    if (!k.kept || k.size != size || k.time != time) {
        const auto t0 = std::chrono::steady_clock::now();
        bool whole = false;
        k.refs = pipe_notes_refs(&whole);
        k.walks++;
        k.walk_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        int64_t after = -1, after_time = 0;   // (changed while it was walked: used this once, not kept)
        notes_file_id(&after, &after_time);
        // (kept only from a walk that read the whole file it found: one that could not open it (another program holding it
        // with reads denied) or had a read fail part way is used this once and the next use walks again, as main's walks did)
        k.kept = whole && size >= 0 && after == size && after_time == time;
        k.size = size, k.time = time;
    }
    if (!take) return k.refs;
    k.kept = false;
    return std::move(k.refs);
}

/// (hookshot, RX_VK_LOAD=create) The first start after a preparation: every noted pipeline built into the pipeline cache and let
/// go, so that the loads at the programs' create calls find each one there and never meet a pipeline the prepare should have
/// compiled. The precompile does the same by keeping them all ready, which is the memory RX_VK_LOAD=create is there to save
/// and more than a 16 GB PC has (its precompile was killed at 23,407 of 35,870 pipelines under a 12 GB cap). Here each
/// worker borrows one program, builds its notes' pipelines one after the other and destroys each at once: never more
/// pipelines alive than workers (warm_bounded.h). Before the game, with the precompile's progress; a start closed in it begins
/// again at the next one (the close puts the file back as the start found it, crash_markers.h), and after two that crashed in it
/// (or were killed), the notes are taken as they are (what a cache then lacks is made at its first draw, as on an install that
/// was never prepared)
void precompile_warm(void (*progress)(size_t done, size_t total)) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string warming = precompile_built_path() + ".warming";
    int ended_inside = 0;
    if (FILE* f = fopen(warming.c_str(), "rb")) {
        if (fscanf(f, "%d", &ended_inside) != 1) ended_inside = 1;
        fclose(f);
    }
    pcache_init();
    const std::vector<NoteRef> refs = start_note_refs(false);
    if (ended_inside >= 2) {
        remove(warming.c_str());
        precompile_built_write(refs.size(), compute_warm_plan().coverage());
        logf("vk: precompile: the last two starts did not get through building the noted pipelines into the cache: left as they "
             "are, a pipeline the caches lack is made when it is first drawn");
        return;
    }
    crash_markers::begin(warming, ended_inside + 1, ended_inside ? ended_inside : crash_markers::none);
    // (by program, in the notes' order: a worker's borrowed program serves all its notes)
    std::vector<uint64_t> progs;
    std::unordered_map<uint64_t, std::vector<const NoteRef*>> by_prog;
    for (const NoteRef& r : refs) {
        auto& list = by_prog[r.prog];
        if (list.empty()) progs.push_back(r.prog);
        list.push_back(&r);
    }
    const size_t total = refs.size();
    // (the compute pipelines after them, the plan's compute programs', counted in the progress from the start: reported as the
    // notes' total once those were done, the screen read "32,623 / 32,623", 100%, "a few seconds left" while they were built, #714)
    // (hookshot) Only those the precompile keeps where RX_VK_PRECOMPILE_COMPUTES says which, or its default does: on Mesa's
    // Intel driver the listed ones (precompile_computes_which), as the preparation's own warm builds them. Every kept one there
    // was 4,059 programs at a game's first start on an Iris Xe with 16 GB, the driver's compiles on 12 threads, and Linux killed
    // the game for memory at 1,852 of them. A program left out is made at its first dispatch, as with "listed".
    // The seed warm's selection here too (compute_warm_plan): compiling the rest at the next start would move the same
    // work to the title screen.
    std::vector<uint64_t> computes = program_cache_compute_keys();
    const ComputePlan& plan = compute_warm_plan();
    if (!plan.all)
        computes.erase(std::remove_if(computes.begin(), computes.end(),
                                      [&](uint64_t k) { return plan.none || !plan.keys.count(k); }),
                       computes.end());
    const size_t all_n = total + computes.size();
    std::atomic<size_t> done{0}, built{0}, failed{0}, no_program{0};
    const std::string notes_path = pipe_notes_path();
    auto log_at = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::chrono::steady_clock::time_point save_at{};   // (warm_save_tick: the cache written as it goes)
    const unsigned threads = prepare_pipeline_threads(std::min(usable_cpus(), 16u), "the noted pipelines");
    warm_bounded::run(progs.size(), threads, [&](size_t i) {
        static thread_local bool off = (ub_thread_off(), true);   // (a background thread: see ub_thread_off)
        static thread_local FILE* in = fopen(notes_path.c_str(), "rb");
        static thread_local std::unique_ptr<PipeNote> note = std::make_unique<PipeNote>();
        (void)off;
        const IntelPrepareLane lane;   // (a preparation thread: its graphics pipelines side by side where intel_prepare_open)
        const auto& list = by_prog.at(progs[i]);
        size_t found = 0;
        program_cache_borrow(progs[i], [&](Program& p) {
            ensure_layout(p);
            for (const NoteRef* r : list) {
                if (r->variant != p.variant || prepare_stopped() || !pipe_note_at(in, r->at, note.get())) continue;
                found++;
                const VkPipeline made = build_graphics(p, note->s, note_targets(*note), (VkPrimitiveTopology)note->topo);
                if (made) vkDestroyPipeline(dev.device, made, nullptr);   // (it is in the cache now: that is all it was built for)
                (made ? built : failed)++;
            }
        });
        no_program += list.size() - std::min(found, list.size());
        done += list.size();
    }, [] { return prepare_stopped(); }, [&](size_t) {
        if (progress) progress(done.load(), all_n);
        prepare_progress_log("noted pipelines", done.load(), total, t0, log_at);
        warm_save_tick(save_at);
    });
    // Build the selected compute pipelines into the cache after the graphics notes.
    std::atomic<size_t> computes_done{0}, computes_built{0}, computes_failed{0};
    const auto c0 = std::chrono::steady_clock::now();
    warm_bounded::run(done == total && !prepare_stopped() ? computes.size() : 0, threads, [&](size_t i) {
        static thread_local bool off = (ub_thread_off(), true);
        (void)off;
        program_cache_borrow(computes[i], [&](Program& p) {
            if (!p.compute || p.failed || p.stages.empty() || !p.stages[0].module || prepare_stopped() || taau_key(p.key)) return;
            ensure_layout(p);
            VkComputePipelineCreateInfo ci = compute_pipeline_info(p, true);   // (as a dispatch and a load ask for it)
            precompile_compute_quick(ci);
            VkPipeline made = VK_NULL_HANDLE;
            if (cached_compute(&ci, &made) == VK_SUCCESS && made) {
                vkDestroyPipeline(dev.device, made, nullptr);
                computes_built++;
            } else {
                computes_failed++;
            }
        });
        computes_done++;
    }, [] { return prepare_stopped(); }, [&](size_t) {
        if (progress) progress(total + computes_done.load(), all_n);
        prepare_progress_log("compute pipelines", computes_done.load(), computes.size(), c0, log_at);
        warm_save_tick(save_at);
    });
    const bool all = done == total && computes_done == computes.size();
    static const bool prepare_only = env_on("HOOKSHOT_PREPARE_ONLY");   // (a --prepare-shaders run: vk_main.cpp)
    if (all && !prepare_stopped()) {
        // (the whole cache written, a GB and more, with every pipeline counted: the start-up screen says so, not "a few seconds
        // left", #714)
        const auto s0 = std::chrono::steady_clock::now();
        prepare_saving_now.store(true, std::memory_order_relaxed);
        pipeline_cache_save(true);   // (in the file before the marker says they are there)
        prepare_saving_now.store(false, std::memory_order_relaxed);
        logf("vk: precompile: the pipeline cache %s, in %.0f ms", precompile_save::said(pcache_save_outcome).c_str(),
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
        // (not by the preparation's own process: warm_bounded.h leaves_marker)
        if (warm_bounded::leaves_marker(prepare_only, all, prepare_stopped())) precompile_built_write(total, compute_warm_plan().coverage());
        driver_env_note();
        crash_markers::end(warming);
    }
    logf("vk: precompile: %zu noted pipelines of %zu programs and %zu compute pipelines of %zu compute programs built into the "
         "pipeline cache and let go (%zu and %zu not made, %zu whose program variant is not kept), on %u threads in %.1f s (the "
         "compute ones %.1f s; the notes' file walked %d times this start so far)%s", built.load(), progs.size(),
         computes_built.load(), computes.size(), failed.load(), computes_failed.load(), no_program.load(), threads,
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count(), start_note_refs_kept.walks,
         !all || prepare_stopped() ? "; stopped before the end: the next start goes on"
         : prepare_only ? "; the preparation's own process: the first start of the game builds them again, from the driver's "
                          "cache, into the cache it then loads from" : "");
}

/// (hookshot, RX_VK_LOAD=create) The start with nothing kept ready by a list: each note is only found, by its place in the file,
/// and its pipeline is made when the game makes its program (vk_programs.cpp); the compute programs the lists name are only
/// named. What a start must still do itself is done: the caches and the two lists opened before play, and on a GPU without
/// mesh shaders the mesh fallback's programs and their compute pipelines, which stay as they were (every one, before the title).
/// Those are built in the driver here, behind the start-up screen, with its progress (a fallback program at a time) and the
/// precompile's bound (bounded_wait.h, RX_VK_PRECOMPILE_STALL_S): nothing finished for that long, the game starts and the
/// builds still in the driver go on behind it, their programs' draws left out until they are done, as a first draw's are.
/// Without them every start of rc.7 on an RX 5600 XT and an RX 5700 XT (AMD's Windows driver, the compute fallback, a new
/// translator: every fallback pipeline compiled again) sat at 0%, "Loading the shader cache", until the window was closed,
/// and the close lost what was built (Hookshot #1373, #1405). The workers' state is on the heap: let go once they are done,
/// never freed when the game went on without them. A pass that took RX_VK_PRECOMPILE_SAVE_S writes the pipeline cache before
/// the title, as precompile_run's does (precompile_save.h)
void precompile_at_create(void (*progress)(size_t done, size_t total)) {
    const auto t0 = std::chrono::steady_clock::now();
    pcache_init();
    std::vector<NoteRef> refs = start_note_refs(true);   // (its last use this start: programs_index takes them)
    if (!env_on("RX_VK_NO_PROGRAM_CACHE")) {
        std::lock_guard lk(note_mu);
        if (!note_file) {
            for (const NoteRef& r : refs) noted.insert(r.key);
            const std::vector<std::unique_ptr<PipeNote>> none;
            notes_open_locked(&none);
        }
    }
    if (!env_on("RX_VK_NO_STATE_LIST")) {
        std::lock_guard lk(state_mu);
        if (!state_file) state_open_locked();
    }
    const ComputePlan plan = computes_plan(precompile_computes_which(), !env_on("RX_VK_NO_STATE_LIST"));
    struct Pass {
        std::vector<Program*> fallbacks, kept;
        std::unordered_map<uint64_t, std::vector<NoteRef>> notes_of;   // (copies: the workers may outlive refs)
        std::string notes_path;
        std::atomic<size_t> next{0}, next_kept{0}, done{0}, kept_pipes{0}, kept_cold{0};
        std::atomic<bool> gone{false};   // (the game went on: no new item is taken, the builds under way end on their own)
    };
    Pass* const pass = new Pass;
    // (every program read in here, kept from now on: its records are taken as they are read, so the load its create call
    // queues finds none and makes none of its noted pipelines. Those are asked of the caches here, as that load would)
    program_cache_load_some({}, [&](Program& p) {
        ensure_layout(p);
        pass->kept.push_back(&p);
        if (p.fallback) pass->fallbacks.push_back(&p);
    }, false, true);
    if (!pass->kept.empty()) {
        std::unordered_set<uint64_t> graphics;
        for (const Program* p : pass->kept)
            if (!p->compute) graphics.insert(p->key);
        for (const NoteRef& r : refs)
            if (graphics.count(r.prog)) pass->notes_of[r.prog].push_back(r);
    }
    pass->notes_path = pipe_notes_path();
    const auto taking = [pass] { return !prepare_stopped() && !pass->gone.load(std::memory_order_relaxed); };
    const auto worker = [pass, taking] {
        ub_thread_off();
        for (size_t i; taking() && (i = pass->next++) < pass->fallbacks.size();) {
            int none = 0;
            if (pass->fallbacks[i]->compute_state.compare_exchange_strong(none, 1)) {
                fallback_compute_pipelines(*pass->fallbacks[i]);
                pass->fallbacks[i]->compute_state.store(2, std::memory_order_release);
            }
            pass->done++;
        }
        FILE* in = nullptr;
        const std::unique_ptr<PipeNote> note = std::make_unique<PipeNote>();
        for (size_t i; taking() && (i = pass->next_kept++) < pass->kept.size();) {
            const Program& p = *pass->kept[i];
            const auto list = pass->notes_of.find(p.key);
            if (!p.compute && list != pass->notes_of.end()) {
                if (!in) in = fopen(pass->notes_path.c_str(), "rb");
                for (const NoteRef& r : list->second) {
                    if (r.variant != p.variant || !pipe_note_at(in, r.at, note.get())) continue;
                    bool cold = false;
                    // (from the caches alone, as a load at a create call: what none has is made when it is first drawn)
                    if (const VkPipeline pl = pipeline_from_caches(p, *note, &cold)) pass->kept_pipes += pipeline_keep_ready(r.key, pl, &p);
                    else pass->kept_cold++;
                }
            }
            pass->done++;
        }
        if (in) fclose(in);
    };
    const size_t total = pass->fallbacks.size() + pass->kept.size();
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < std::min<size_t>(usable_cpus(), pass->kept.size()); i++) threads.emplace_back(worker);
    static const int stall_s = getenv("RX_VK_PRECOMPILE_STALL_S") ? atoi(getenv("RX_VK_PRECOMPILE_STALL_S")) : 30;
    const auto built0 = std::chrono::steady_clock::now();
    auto log_at = built0 + std::chrono::seconds(10);
    const bounded_wait::End end = threads.empty() ? bounded_wait::End::kDone : bounded_wait::wait(
        pass->done, total, stall_s, [] { return std::chrono::steady_clock::now(); },
        [&](size_t done) {
            if (progress) progress(done, total);
            const auto now = std::chrono::steady_clock::now();
            if (now < log_at) return;
            log_at = now + std::chrono::seconds(10);
            logf("vk: precompile: load at create: %zu of %zu steps in %.0f s (each mesh fallback program's compute pipelines, then "
                 "the noted pipelines of each program kept, from the caches)", done, total,
                 std::chrono::duration<double>(now - built0).count());
            compute_compiles_log();   // (the compiles running long by name: on a first start the mesh fallback's are compiled, not
                                      // found, minutes of it on an integrated GPU)
        },
        [] { return prepare_stopped(); }, [] { Sleep(100); },
        // (a --prepare-shaders run reaches here when a marker says its notes are built: as precompile_run, a compute compile at
        // work (a CPU busy) is left to the preparation's watchdog, vk_main.cpp, since one left behind here dies with the run's
        // end. A game start goes on after stall_s whatever the driver does, and so does a compile with no CPU at work)
        [] {
            static const bool prepare_only_run = env_on("HOOKSHOT_PREPARE_ONLY");
            return prepare_only_run && prepare_compiling() && prepare_compiler_busy();
        });
    const bool gone_on = end != bounded_wait::End::kDone;
    if (gone_on) {
        pass->gone.store(true);
        prepare_left_workers_note();   // (stalled or closed: builds may still be in the driver, vk_main.cpp's watchdog and exit)
        if (end == bounded_wait::End::kStalled) {
            logf("vk: precompile: load at create: %zu of %zu steps done, nothing finished for %d s: "
                 "the game starts, the builds still in the driver go on behind it (RX_VK_PRECOMPILE_STALL_S=0: wait for all)",
                 pass->done.load(), total, stall_s);
        }
        for (std::thread& t : threads) t.detach();
    } else {
        for (std::thread& t : threads) t.join();
    }
    const size_t notes_n = refs.size(), kept_n = pass->kept.size(), kept_pipes = pass->kept_pipes.load(),
                 kept_cold = pass->kept_cold.load();
    if (!gone_on) delete pass;
    programs_index(std::move(refs), plan.none ? std::unordered_set<uint64_t>{} : plan.keys);
    logf("vk: precompile: load at create (RX_VK_LOAD=create): %zu noted pipelines found, none built; %zu compute programs "
         "named (%zu in the shipped list, %zu dispatched in play here); %zu mesh fallback programs kept ready, %zu of their "
         "noted pipelines with them (%zu in no cache)%s; in %.0f ms. No other program is kept ready by a list: each is loaded "
         "when the game makes it (the notes' file walked %d times this start, in %.0f ms: start_note_refs)",
         notes_n, plan.keys.size(), plan.shipped, plan.learned, kept_n, kept_pipes, kept_cold,
         end == bounded_wait::End::kStalled ? ", some still being built" : "",
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
         start_note_refs_kept.walks, start_note_refs_kept.walk_ms);
    // (fallback pipelines that took this long were compiled, not found: written before the title, the start-up screen saying
    // so, as precompile_run does; a start closed before then kept none of them, and the next one compiled them all again)
    static const int save_s = precompile_save::after_s(getenv("RX_VK_PRECOMPILE_SAVE_S"));
    const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - built0).count();
    if (precompile_save::due(build_s, save_s, end == bounded_wait::End::kStalled, prepare_stopped(),
                             env_on("HOOKSHOT_PREPARE_ONLY"))) {
        const auto s0 = std::chrono::steady_clock::now();
        prepare_saving_now.store(true, std::memory_order_relaxed);
        pipeline_cache_save(true);
        prepare_saving_now.store(false, std::memory_order_relaxed);
        logf("vk: precompile: load at create: its pipelines took %.0f s (compiled, not found in the caches): the pipeline cache "
             "%s, in %.0f ms, before the title (RX_VK_PRECOMPILE_SAVE_S=0: not)", build_s,
             precompile_save::said(pcache_save_outcome).c_str(),
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
    }
    predicted_check();
}

// Skipping startup compilation must not leave load-at-create requests without an
// index. Initialize the shared cache before starting its workers, then index only
// existing notes/programs. Do not import state lists, translate shaders, warm cold
// pipelines, or eagerly load mesh fallback programs here.
void precompile_index_only() {
    if (!programs_load_at_create()) return;
    pcache_init();
    const ComputePlan plan = computes_plan(precompile_computes_which(), !env_on("RX_VK_NO_STATE_LIST"));
    programs_index(pipe_notes_refs(), plan.none ? std::unordered_set<uint64_t>{} : plan.keys);
    logf("vk: startup compilation skipped; cached load-at-create index initialized");
}

/// The precompile waits for its pipelines (the start shows a console until they are built) only when asked: RX_VK_PRECOMPILE_WAIT=1.
/// Otherwise it queues them and the game starts; the draws that meet one still being built are left out of their frame
bool precompile_blocking() {
    static const bool wait = getenv("RX_VK_PRECOMPILE_WAIT") && atoi(getenv("RX_VK_PRECOMPILE_WAIT")) != 0;
    return wait;
}

namespace {
/// The precompile's work, on the heap for good: its threads may go on after precompile_run returns (the background, or a start that
/// gave up waiting for a pipeline that never finishes: two of 4926 on an RX 5700 XT, "stuck at 99%", Discord 11-08)
struct Pre {
    struct Item {
        std::unique_ptr<PipeNote> note;
        const Program* program;
        PipeEntry* entry;
        Targets at;
    };
    std::vector<Item> items;          // graphics pipelines, in the order the notes were written (the order the game first used them)
    std::vector<Program*> computes;   // the compute programs the cache keeps
    std::atomic<size_t> next{0}, done{0}, made{0}, failed{0};
    std::atomic<unsigned> running{0};
    size_t total = 0;
    size_t carried = 0, missing = 0;
    bool background = false;
    std::chrono::steady_clock::time_point t0;
    std::string marker;
};
void precompile_finished(Pre* pre) {
    logf("vk: precompile: %zu graphics (%zu made, %zu not) and %zu compute pipelines built in %.0f ms%s", pre->items.size(),
         pre->made.load(), pre->failed.load(), pre->computes.size(),
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pre->t0).count(),
         pre->background ? " (in the background, while the game ran)" : "");
    remove(pre->marker.c_str());
}
}  // namespace


/// (hookshot) Whether the driver's own shader cache has lost what the notes name, at a start that would load at create calls: a
/// sample of the notes asked without a compile, of the driver alone (an empty scratch pipeline cache: on NVIDIA ours only refers
/// to the driver's). NVIDIA's cache is shared with every other game and evicts what it holds past its size, and a driver update
/// empties it; the marker (precompile_built_write) still said every note had been built, so each listed pipeline was compiled in
/// play with its draws left out (the gate route with NVIDIA's cache pointed at an empty folder: 10,263 in no cache, 702,312 draws
/// left out in 140 s, 21 s of waits), and only the next start built them before the title (load_default's marker, which keeps
/// them all for good). Now the warm pass runs again at this start, behind its progress
bool notes_lost_by_driver() {
    if (!dev.vk13.pipelineCreationCacheControl) return false;
    constexpr size_t kSample = 48;
    const std::vector<NoteRef> refs = start_note_refs(false);
    if (refs.size() < kSample) return false;
    FILE* f = fopen(pipe_notes_path().c_str(), "rb");
    if (!f) return false;
    const auto t0 = std::chrono::steady_clock::now();
    VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache scratch = VK_NULL_HANDLE;
    if (vkCreatePipelineCache(dev.device, &cci, nullptr, &scratch) != VK_SUCCESS) scratch = VK_NULL_HANDLE;
    // (the thread's driver-only asking, the scratch cache and the file put back and let go however this ends: a throw from a
    // borrow or a build would otherwise leave the thread asking the scratch cache's driver alone from then on)
    struct Restore {
        FILE* f;
        VkPipelineCache scratch;
        ~Restore() {
            driver_cache_only = false, driver_ask_cache = VK_NULL_HANDLE;
            if (scratch) vkDestroyPipelineCache(dev.device, scratch, nullptr);
            fclose(f);
        }
    } restore{f, scratch};
    driver_cache_only = true, driver_ask_cache = scratch;
    size_t asked = 0, cold_n = 0;
    auto note = std::make_unique<PipeNote>();
    for (size_t i = 0; i < kSample; i++) {
        const NoteRef& r = refs[i * refs.size() / kSample];
        if (!pipe_note_at(f, r.at, note.get())) continue;
        program_cache_borrow(note->prog, [&](Program& p) {
            if (p.variant != note->variant) return;
            ensure_layout(p);
            bool cold = false;
            const VkPipeline pl = build_graphics(p, note->s, note_targets(*note),
                                                 pipeline_topology(p, (VkPrimitiveTopology)note->topo), -1, &cold);
            if (pl) asked++, vkDestroyPipeline(dev.device, pl, nullptr);
            else if (cold) asked++, cold_n++;
        });
    }
    const bool lost = asked >= kSample / 3 && cold_n * 2 > asked;
    logf("vk: the driver's shader cache has %zu of %zu noted pipelines asked (a sample, in %.0f ms)%s", asked - cold_n, asked,
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
         lost ? ": it lost them (a driver update, or evicted): they are built again before the title"
              : "");
    return lost;
}

void precompile_run(void (*progress)(size_t done, size_t total)) {
    // (the texture pipelines first, into the start's pipeline cache: not compiled inside a frame of play, vk_texture.cpp)
    { ParLock lk; pcache_init(); }
    texture_pipelines_warm();
    // (hookshot, RX_VK_LOAD=create: only once every note has been built into the pipeline cache, so that a load from the
    // caches alone never meets a pipeline the prepare should have compiled. The first start after a preparation builds them
    // there and lets each go, precompile_warm: before the game, and never more of them alive than workers)
    if (programs_load_at_create()) {
        // (the pipeline cache's files dropped at this start, pcache_drop_stale, or its file refused as another driver's,
        // pcache_init: what the marker says is in them is gone. The file read first, so the marker is judged on what it holds)
        pcache_init();
        if (pcache_drop_stale() || pcache_file_rejected) remove(precompile_built_path().c_str());
        // (and where the driver's own cache lost what the marker says was built: notes_lost_by_driver)
        if (precompile_covers_notes() && !env_on("HOOKSHOT_PREPARE_ONLY") && notes_lost_by_driver())
            remove(precompile_built_path().c_str());
        // (a first start's two passes on the bar as one count: start_steps. The second's steps counted before the warm, which only
        // borrows the programs; a --prepare-shaders run has no second pass, and its warm ends the bar)
        const bool warms = !precompile_covers_notes();
        const bool then = warms && !env_on("HOOKSHOT_PREPARE_ONLY");
        start_steps::begin(progress, then, then ? program_cache_fallback_steps() : 0);
        if (warms) precompile_warm(progress ? start_steps::warm : nullptr);
        if (precompile_covers_notes()) {
            precompile_at_create(progress ? start_steps::create : nullptr);
            return start_steps::end();
        }
        // (no load at create this start: what start_note_refs kept for it is let go here, as each walk's were before)
        start_note_refs_kept = StartNoteRefs{};
        // (the window was closed in it; or this is the preparation's own process, which leaves no marker: its warm filled
        // the driver's cache, the first start of the game warms again and loads at create calls from then on)
        if (prepare_stopped() || env_on("HOOKSHOT_PREPARE_ONLY")) return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    // (a start that dies in here would die in here every time, the notes being what it builds from: a player saw the console reach
    // 542 of 543 and the game close. The file says this step began and did not end: the next start lets the notes go and plays)
    // (two starts in a row, not one: closing the window while it ran counted as a crash, and the next start then skipped the
    // precompile and compiled everything in play: "stuck at 99%, then fine on the second start", 2026-11-08. And a close does
    // not count at all: a quit ends the process before this returns, so two quits in it still dropped the notes. The close puts
    // the file back as the start found it, crash_markers.h)
    const std::string running = pipe_notes_path() + ".running";
    int ended_inside = 0;
    if (FILE* f = fopen(running.c_str(), "rb")) {
        if (fscanf(f, "%d", &ended_inside) != 1) ended_inside = 1;
        fclose(f);
        if (ended_inside >= 2) {
            remove(running.c_str());
            remove(pipe_notes_path().c_str());
            logf("vk: the last two starts did not get through the precompile: its notes are dropped");
            return;
        }
        logf("vk: the last start did not get through the precompile (it crashed there, or was killed): trying it once more");
    }
    crash_markers::begin(running, ended_inside + 1, ended_inside ? ended_inside : crash_markers::none);
    // (stopped: the window was closed, and the close puts the file back, crash_markers::close)
    struct Done { const std::string& path; ~Done() { if (!prepare_stopped()) crash_markers::end(path); } } done_marker{running};
    pcache_init();
    std::vector<std::unique_ptr<PipeNote>> notes = pipe_notes_read();
    // (the notes file and the state list opened now, before play, with the notes just read: pipeline_note and state_record opened
    // them at the first new pipeline in play, on the thread replaying, 150 ms and more)
    if (!env_on("RX_VK_NO_PROGRAM_CACHE")) {
        std::lock_guard lk(note_mu);
        if (!note_file) notes_open_locked(&notes);
    }
    if (!env_on("RX_VK_NO_STATE_LIST")) {
        std::lock_guard lk(state_mu);
        if (!state_file) state_open_locked();
    }
    std::unordered_set<uint64_t> wanted, wanted_variants;
    for (const auto& n : notes) wanted.insert(n->prog), wanted_variants.insert(mix(n->prog, n->variant));
    // (the compute programs: those this install's play dispatched (hookshot_computes.bin) and those the shipped state list names,
    // each once; every kept one on a first start, with nothing dispatched yet: computes_plan)
    const ComputePlan plan = env_on("HOOKSHOT_PREPARE_ONLY") ? compute_warm_plan()
        : computes_plan(precompile_computes_which(), !env_on("RX_VK_NO_STATE_LIST"));
    if (!plan.all && !plan.none) wanted.insert(plan.keys.begin(), plan.keys.end());
    std::vector<Program*> computes, fallbacks;
    std::unordered_set<uint64_t> listed_here;   // (the listed keys this install has a program of)
    const size_t programs_n = program_cache_load_some(wanted, [&](Program& p) {
        ensure_layout(p);
#ifdef __linux__
        if (!p.compute && wanted_variants.count(mix(p.key, p.variant))) program_cache_note_built(p);   // (pipeline_cache_play)
#endif
        if (p.compute) {
            computes.push_back(&p);
            if (plan.keys.count(p.key)) listed_here.insert(p.key);
        } else if (p.fallback) fallbacks.push_back(&p);   // (hookshot: the mesh fallback's compute pipelines too, before play)
    }, plan.all, true);
    // (a listed key with no program here: another game version's, or a program this install did not keep. Nothing is made of it)
    logf("vk: precompile: compute programs: %s (%zu keys listed: %zu in the shipped list, %zu dispatched in play here, %zu of the "
         "shipped not among those; %zu of the listed have a program here, %zu variants)",
         plan.all ? (plan.learned ? "every kept one" : "every kept one (nothing dispatched in play here yet: a first start)")
                  : plan.none ? "none" : plan.learned_only ? "those play dispatched here" : "those listed",
         plan.keys.size(), plan.shipped, plan.learned, plan.fresh, listed_here.size(), computes.size());
    {   // (how many of them keep the 32 lanes, and what for: compute_pin.h)
        size_t pinned = 0, shared_alone = 0;
        for (const Program* p : computes) {
            if (!compute_pin_needed(*p)) continue;
            pinned++;
            if (p->stages.empty()) continue;
            shared_alone += !compute_pin::sees_lanes_ops(compute_program_uses(*p, p->stages[0]));
        }
        if (!computes.empty())
            logf("vk: precompile: compute programs: %zu held to 32 lanes (%zu of them for their shared memory alone), %zu at the lanes "
                 "the driver chooses (RX_VK_COMPUTE_PIN=all or lanes)", pinned, shared_alone, computes.size() - pinned);
    }
    const double read_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const size_t total = notes.size() + computes.size() + fallbacks.size();
    // (the workers' state on the heap: a start whose last pipelines never finish (two of 4926 on an RX 5700 XT, "stuck at
    // 99%", Discord 11-08) goes on to the game after PRE_STALL_S seconds with no progress, the stragglers left running; it is
    // shared with the workers and freed after the last one returns. RX_VK_PRECOMPILE_STALL_S=<s>, 0: wait for all. In a
    // --prepare-shaders run a compute compile still at work is left to the watchdog: a 32-lane compile on an Iris Xe was
    // measured at 531 s, and one left behind here died with the run's end, its work lost)
    struct Pre {
        std::vector<std::unique_ptr<PipeNote>> notes;
        std::vector<Program*> computes, fallbacks;
        std::atomic<size_t> next{0}, done{0};
        std::atomic<size_t> computes_cached{0}, computes_compiled{0}, computes_failed{0};
        std::mutex mu, first_graphics;
        std::atomic<bool> graphics_begun{false};
        bool abandoned = false;
    };
    const size_t graphics_n = notes.size(), computes_n = computes.size(), fallbacks_n = fallbacks.size();
    auto pre = std::make_shared<Pre>();
    pre->notes = std::move(notes), pre->computes = std::move(computes), pre->fallbacks = std::move(fallbacks);
    auto& next = pre->next;
    auto& done = pre->done;
    const auto worker = [pre, total] {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        const IntelPrepareLane lane;   // (a preparation thread: its graphics pipelines side by side where intel_prepare_open)
        auto& notes = pre->notes;
        auto& computes = pre->computes;
        auto& fallbacks = pre->fallbacks;
        for (;;) {
            if (prepare_stopped()) return;
            {   // (the game went on without them: no new one, the GPU and the CPU left to it; the ones in flight are let go of)
                std::lock_guard lk(pre->mu);
                if (pre->abandoned) return;
            }
            const size_t i = pre->next++;
            if (i >= total) return;
            if (i < computes.size()) {
                Program& p = *computes[i];
                VkComputePipelineCreateInfo ci = compute_pipeline_info(p, false);
                // (asked of the caches first, as a dispatch asks: the prepare left it there, and the log's line can then say how
                // many were compiled now. Where the driver cannot say, each is just made, and counted as compiled)
                VkPipeline made = compute_pipeline_cached(p, false);
                bool quick = false;
                if (made) {
                    pre->computes_cached++;
                } else {
                    quick = precompile_compute_quick(ci);
                    if (cached_compute(&ci, &made) == VK_SUCCESS && made) pre->computes_compiled++;
                    else made = VK_NULL_HANDLE, pre->computes_failed++;
                }
                std::lock_guard lk(pre->mu);
                if (!pre->abandoned) {
                    std::atomic_ref<VkPipeline>(p.compute_pipeline).store(made, std::memory_order_release);
                    if (made && quick) p.optimize_owed.store(true, std::memory_order_release);
                    else if (made) cs_module_release(p);
                } else if (made) vkDestroyPipeline(dev.device, made, nullptr);   // (the game went on: it builds its own)
            } else if (i < computes.size() + fallbacks.size()) {
                // (a program a draw is already building is left to it: compute_state, as in fallback_pipelines_ready)
                Program& p = *fallbacks[i - computes.size()];
                int none = 0;
                if (p.compute_state.compare_exchange_strong(none, 1)) {
                    fallback_compute_pipelines(p);
                    p.compute_state.store(2, std::memory_order_release);
                }
            } else {
                PipeNote& n = *notes[i - computes.size() - fallbacks.size()];
                const Program* p = program_find(n.prog, n.variant);
                if (p) {
                    // (the first graphics pipeline alone, the rest in parallel: after the library warm, NVIDIA's driver (621.14,
                    // RTX 4070 Ti Super) faulted (nvoglv64+0xde2515) when all the workers began theirs at once, in 14 of 100
                    // starts, and the start hung on the lock it held then)
                    std::unique_lock first(pre->first_graphics, std::defer_lock);
                    if (!pre->graphics_begun.load()) {
                        first.lock();
                        if (pre->graphics_begun.load()) first.unlock();
                    }
                    const VkPipeline made = build_graphics(*p, n.s, note_targets(n), (VkPrimitiveTopology)n.topo);
                    pre->graphics_begun = true;
                    if (first.owns_lock()) first.unlock();
                    // (hookshot, 2026-11-15) The build's preparation run (HOOKSHOT_PREPARE_ONLY) plays nothing: the pipeline is in the
                    // pipeline cache now, all it was built for there, and is let go at once. Kept, every noted pipeline (32,713) stayed
                    // alive to the run's end on this path (start mode: GPUs without mesh shaders, or an install the safety net took
                    // back to it), over 12 GB on its own (a precompile was killed at 23,407 of 35,870 under a 12 GB cap): a 16 GB PC
                    // paged to disk while it built. The game's first start keeps them ready from the caches, as before.
                    // RX_VK_PREPARE_KEEP=1: kept, as before (tests/test_prepare_lets_go.py)
                    static const bool let_go = env_on("HOOKSHOT_PREPARE_ONLY") && !env_on("RX_VK_PREPARE_KEEP");
                    if (let_go) {
                        if (made) vkDestroyPipeline(dev.device, made, nullptr);
                    } else {
                        std::lock_guard lk(note_mu);
                        auto it = pipelines.find(n.key);
                        if (it == pipelines.end()) {
                            auto e = std::make_unique<PipeEntry>();
                            e->pipeline = made;
                            e->prog.store(p, std::memory_order_release);   // (whose pipeline it is: graphics_pipeline)
                            e->ready.store(true, std::memory_order_release);
                            pipelines.emplace(n.key, std::move(e));
                        } else if (made) vkDestroyPipeline(dev.device, made, nullptr);   // (unpublished: play or another note already owns the entry)
                    }
                }
            }
            pre->done++;
        }
    };
    std::vector<std::thread> threads;
    const unsigned cpus = usable_cpus();
    const unsigned workers = prepare_pipeline_threads(std::min(cpus, 64u), "the precompile");
    for (unsigned i = 0; i < workers; i++) threads.emplace_back(worker);
    static const int stall_s = getenv("RX_VK_PRECOMPILE_STALL_S") ? atoi(getenv("RX_VK_PRECOMPILE_STALL_S")) : 30;
    size_t last_done = 0;
    auto last_change = std::chrono::steady_clock::now();
    bool gone_on = false;
    auto log_at = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::chrono::steady_clock::time_point save_at{};   // (warm_save_tick: the cache written as it goes)
    const auto built0 = std::chrono::steady_clock::now();
    while (done.load() < total && !prepare_stopped()) {
        if (progress) progress(done.load(), total);
        prepare_progress_log("noted pipelines", done.load(), total, built0, log_at);
        warm_save_tick(save_at);
        Sleep(100);
        if (next.load() >= total + threads.size()) break;
        const auto now = std::chrono::steady_clock::now();
        if (done.load() != last_done) last_done = done.load(), last_change = now;
        else {
            // (the preparation's own process waits for a compute compile under way while the CPU is at work: its watchdog waits
            // longer for it and stops it if it stalls (vk_main.cpp), and one left behind here died with the run's end. A compile
            // that uses no CPU is stuck, and is left behind after stall_s as before. A game start goes on to play after stall_s
            // whatever the driver does, the compile behind it: its dispatches are left out until play has built the pipeline.
            // Whether or not every pipeline was handed out: with every worker inside the driver none was, and the start waited
            // for good on AMD's RX 6600 / 6600 XT, the CPU idle)
            static const bool prepare_only_run = env_on("HOOKSHOT_PREPARE_ONLY");
            const int limit_s =
                prepare_only_run && next.load() >= total && prepare_compiling() && prepare_compiler_busy() ? 0 : stall_s;
            if (limit_s > 0 && now - last_change > std::chrono::seconds(limit_s)) {
                std::lock_guard lk(pre->mu);
                pre->abandoned = true;
                gone_on = true;
                prepare_left_workers_note();
                logf("vk: precompile: %zu of %zu done, nothing finished for %d s: the game starts, the rest are made in play",
                     done.load(), total, limit_s);
                break;
            }
        }
    }
    size_t cs_cached = 0, cs_compiled = 0, cs_failed = 0;
    const auto compute_counts = [&] {
        cs_cached = pre->computes_cached.load(), cs_compiled = pre->computes_compiled.load(), cs_failed = pre->computes_failed.load();
    };
    if (gone_on) {
        compute_counts();   // (as far as they got: the stragglers go on)
        for (std::thread& t : threads) t.detach();
    } else {
        for (std::thread& t : threads) t.join();
        compute_counts();
    }
    logf("vk: precompile: %zu programs, %zu graphics and %zu compute pipelines, %zu mesh fallback programs in %.0f ms (%.0f ms "
         "of it the caches opened, the notes and programs read)", programs_n, graphics_n, computes_n, fallbacks_n,
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), read_ms);
    logf("vk: precompile: compute pipelines: %zu kept ready (%zu from the pipeline caches, %zu compiled now), %zu not made",
         cs_cached + cs_compiled, cs_cached, cs_compiled, cs_failed);
    if (!gone_on && !prepare_stopped()) driver_env_note();   // (the environment the notes were built in, for later starts)
    // (every note built, by a run that got to its end: RX_VK_LOAD=create may load them from the caches from the next start on)
    if (!gone_on && !prepare_stopped()) precompile_built_write(graphics_n, plan.coverage());
    // (pipelines that took this long were compiled, not found: written before the title, the start-up screen saying so, as
    // precompile_warm does under RX_VK_LOAD=create; in play it would wait for 64 MB of growth: precompile_save.h)
    static const int save_s = precompile_save::after_s(getenv("RX_VK_PRECOMPILE_SAVE_S"));
    const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - built0).count();
    if (precompile_save::due(build_s, save_s, gone_on, prepare_stopped(), env_on("HOOKSHOT_PREPARE_ONLY"))) {
        const auto s0 = std::chrono::steady_clock::now();
        prepare_saving_now.store(true, std::memory_order_relaxed);
        pipeline_cache_save(true);
        prepare_saving_now.store(false, std::memory_order_relaxed);
        logf("vk: precompile: its pipelines took %.0f s (compiled, not found in the caches): the pipeline cache %s, in %.0f ms, "
             "before the title (RX_VK_PRECOMPILE_SAVE_S=0: not)", build_s, precompile_save::said(pcache_save_outcome).c_str(),
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
    }
    predicted_check();
}

// (hookshot) The pipelines the notes predict for the programs no note names (vk_predict.cpp), after a prepare that prepared: each built
// once for the driver's own cache and let go. Not into ours (hookshot_vk_pipelines.bin, read whole at every start: ~40000 more pipelines made
// it 1.7 GB), nor kept: each program is borrowed from hookshot_programs.bin for its pipelines and let go
/// (hookshot) The preparation's three parts as the files on disk count them before it runs: PrepareBar's guesses, so that the bar
/// has every part in its total from the start. The seed's programs (its header); the pipelines the notes will predict, about one and
/// a half a seed program (56,588 for 38,809 programs with the shipped list of 25,251 states); the pipelines the notes and state
/// lists name, and the compute programs play dispatched
void prepare_guess(size_t out[3]) {
    out[0] = out[1] = out[2] = 0;
    const auto size_of = [](const std::string& path) -> int64_t {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return 0;
        _fseeki64(f, 0, SEEK_END);
        const int64_t n = _ftelli64(f);
        fclose(f);
        return n;
    };
    if (FILE* f = fopen(seed_path().c_str(), "rb")) {
        char head[16]{};
        uint32_t programs = 0;
        if (fread(head, 1, 16, f) == 16 && !memcmp(head, "HKSEED01", 8)) std::memcpy(&programs, head + 8, 4);
        fclose(f);
        // (the predicted pipelines: about one and a half a seed program with the shipped list, 56,588 for 38,809 programs, not one;
        // at one the bar ran ahead through the programs and then slowed for minutes. The count replaces the guess once decided)
        out[0] = programs, out[1] = programs + programs / 2;
    }
    const int64_t notes = size_of(pipe_notes_path()), computes = size_of(computes_path());
    const int64_t lists = std::max<int64_t>(0, size_of(state_list_path("hookshot_pipeline_states_shipped.bin")) - 12) +
                          std::max<int64_t>(0, size_of(state_list_path("hookshot_pipeline_states.bin")) - 12);
    out[2] = (size_t)(std::max<int64_t>(0, notes - 16) / (int64_t)sizeof(PipeNote) + lists / (int64_t)sizeof(StateRec) +
                      std::max<int64_t>(0, computes - 8) / 8);
}

void note_color_types(const PipeNote& n, uint8_t types[8]) {
    color_types_of(note_targets(n), types);
}

void predicted_sample_write(const std::vector<std::unique_ptr<PipeNote>>& predicted);
void predicted_keys_write(std::vector<uint64_t> keys);
// (hookshot) The prediction under way: hookshot_pipelines.bin.predicting from its start to its end. A start closed (or ended) in it
// left the rest of the predicted pipelines unbuilt for good: the next start found the programs prepared, so it skipped the
// prediction, and play compiled them as they were drawn. With the marker there, the next start runs it again; what the driver
// kept from the first try comes back from its cache at once (a second prediction over a full driver cache: 7.9 s for 43,414)
std::string predict_pending_path() { return pipe_notes_path() + ".predicting"; }
bool predict_pending() {
    FILE* f = fopen(predict_pending_path().c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}
// (hookshot) Compile shaders, the Builder's, make_game.py's and the launcher's choice (compile_shaders= in the game folder's
// hookshot_options.ini, where make_game.py writes it: the build's, not the player's settings in the user folder, vk_paths.h
// options_path; 0 Most, the default; 1 All; any other number is Most). All builds the predicted pipelines too: about 50,000 for the driver's own cache, over 40 % more time (a full preparation on an RTX
// 3090 with 24 threads: 415 and 413 s against 292 s without them), for pipelines play almost never draws ("vk: predicted
// pipelines met in play"). RX_VK_COMPILE_SHADERS=all or most over the file (the game's --compile-shaders sets it, host/hos_main.c).
// Compute selection also reads the level. Only the prediction path (compile_all below) clears an old prediction.
namespace {
bool compile_level_all() {
    static const bool all = [] {
        const char* why = "the default";
        int v = 0;
        if (const char* e = getenv("RX_VK_COMPILE_SHADERS")) {
            v = !strcmp(e, "all") || !strcmp(e, "1") ? 1 : 0;
            why = "RX_VK_COMPILE_SHADERS";
        } else {
            char exe[MAX_PATH];
            const DWORD n = game_module_path(exe, MAX_PATH);
            std::string ini(exe, n);
            ini = ini.substr(0, ini.find_last_of("\\/") + 1) + "hookshot_options.ini";
            if (FILE* f = fopen(ini.c_str(), "r")) {
                char line[80];
                while (fgets(line, sizeof line, f))
                    if (!strncmp(line, "compile_shaders=", 16)) {
                        v = atoi(line + 16) == 1 ? 1 : 0;
                        why = "compile_shaders in hookshot_options.ini";
                        break;
                    }
                fclose(f);
            }
        }
        if (v) {
            logf("vk: Compile shaders: All (%s): the predicted pipelines are built too", why);
        } else {
            logf("vk: Compile shaders: Most (%s): the predicted pipelines are left out; a pipeline no list names is made when it "
                 "is first drawn", why);
        }
        return v != 0;
    }();
    return all;
}
}  // namespace
namespace {
/// The commit the system still has to give (MB): its limit, RAM and the page file, less what every process has committed. A C++
/// allocation that finds none is refused ("0 MB of commit left"): the game's own, and any other program's. The driver's compiles
/// ask the process's heaps for what they need, and a program of a million SPIR-V words took gigabytes of them. Windows' rule: on
/// Linux the kernel overcommits, and this says there is no limit
unsigned long long commit_left_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? (unsigned long long)(m.ullAvailPageFile >> 20) : ~0ull;
#else
    return ~0ull;
#endif
}
}  // namespace
bool compile_all() {
    static const bool all = [] {
        if (compile_level_all()) return true;
        remove(predict_pending_path().c_str());
        remove((pipe_notes_path() + ".predicted").c_str());
        remove((pipe_notes_path() + ".predicted_keys").c_str());
        return false;
    }();
    return all;
}
void predict_run(void (*progress)(size_t done, size_t total)) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<PipeNote>> notes = pipe_notes_read();
    if (notes.empty() || env_on("RX_VK_NO_PREDICT")) {
        remove(predict_pending_path().c_str());
        return;
    }
    // (the marker counts the starts that began it: two that did not get through it (a driver that hangs or ends the process on one
    // of these pipelines) and it is let go, the rest left to play, as precompile_run does with its notes)
    // A start closed in it has its count written back unchanged by the close (crash_markers::close), whether its workers stopped in
    // time or not: only one that crashed or hung there counts
    int began = 0;
    bool resumed = false;
    if (FILE* f = fopen(predict_pending_path().c_str(), "rb")) {
        if (fscanf(f, "%d", &began) != 1) began = 1;
        fclose(f);
        resumed = true;
    }
    if (began >= 2) {
        remove(predict_pending_path().c_str());
        logf("vk: the last two starts did not get through the predicted pipelines: the rest are built in play");
        return;
    }
    if (resumed) logf("vk: the last start did not finish building the predicted pipelines (%s): they are built again (what the "
                      "driver kept comes from its cache)", began ? "it ended in them" : "closed");
    crash_markers::begin(predict_pending_path(), began + 1, began);
    size_t programs = 0;
    PredictCounts left_out;
    std::vector<std::unique_ptr<PipeNote>> predicted = predict_pipelines(notes, &programs, &left_out);
    const double decide_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const size_t notes_n = notes.size(), predicted_n = predicted.size();
    // (the workers' state on the heap, as precompile_run's: a pipeline the driver never finishes (builds stuck after "vertex format
    // NVN 0xe -> vk 17" on players' cards of all kinds, AMD and NVIDIA, #2107) holds its worker for good, and the preparation goes
    // on without it once nothing has finished for RX_VK_PREDICT_STALL_S. Workers share ownership of the state: a stuck worker still
    // uses it)
    struct Pre {
        std::vector<std::unique_ptr<PipeNote>> notes, predicted;
        std::unordered_map<uint64_t, std::vector<const PipeNote*>> by_program;
        std::vector<uint64_t> keys;
        std::atomic<size_t> next{0}, done{0}, made{0}, same{0};
        std::atomic<bool> out_of_time{false}, out_of_memory{false}, abandoned{false}, graphics_begun{false};
        std::atomic<size_t> too_big{0}, too_big_pipelines{0};   // (programs left out for their size, and their pipelines)
        std::mutex first_graphics, keys_mu;
        std::vector<uint64_t> made_keys;   // (each made pipeline's key, as play looks it up: predicted_keys_write)
    };
    auto pre = std::make_shared<Pre>();
    pre->notes = std::move(notes), pre->predicted = std::move(predicted);
    for (const auto& n : pre->predicted) pre->by_program[n->prog].push_back(n.get());
    for (const auto& kv : pre->by_program) pre->keys.push_back(kv.first);
    auto& next = pre->next;
    auto& done = pre->done;
    auto& made = pre->made;
    auto& out_of_time = pre->out_of_time;
    // (hookshot) No time budget by default: every predicted pipeline is built here, in the preparation, under its progress bar (the
    // start-up screen's and the builder's), not in play. RX_VK_PREDICT_S=<seconds> sets one: those not built by then are built in
    // play when first drawn. The budget (90 s) came when #1051 let the prediction find its programs again: ~28,000 pipelines, many
    // minutes before the game's first frame (20 with the CPU busy) with nothing on screen to say so, on half the CPUs below normal
    // priority. Now the bar counts them from the start (PrepareBar), closing the window stops them (prepare_stopped, kept for the
    // next start), and they take every CPU: nothing else runs before the game
    static const int budget_s = getenv("RX_VK_PREDICT_S") ? atoi(getenv("RX_VK_PREDICT_S")) : 0;
    // (The pipelines are made for the driver's own cache and let go. On an RTX 3050 with 32 GB, a program of 502k SPIR-V words and one
    // of 1014k (the game's two largest) each took the driver's compiler over three minutes and, still growing past 11 GB of the
    // process's commit, all the PC had: C++ allocations were refused at the commit limit, the compile failed and the memory stayed
    // taken (start-up 532 s, 36-39 GB at its peak; without them 246 s and 4.3 GB). The others, up to 133k words, take 25 s and a few
    // hundred MB. Over RX_VK_PREDICT_MAX_KWORDS (300: 300k words; 0: no limit) a program is left to the draw that needs it, as it was
    // before the prediction; and the prediction stops, the rest left to play, when the system has less than
    // RX_VK_PREDICT_RESERVE_MB (3072) of commit left: predict_limits.h)
    static const unsigned long long max_kwords = getenv("RX_VK_PREDICT_MAX_KWORDS") ? strtoull(getenv("RX_VK_PREDICT_MAX_KWORDS"), nullptr, 10)
                                                                                    : predictlimits::kDefaultMaxKWords;
    static const unsigned long long reserve_mb = getenv("RX_VK_PREDICT_RESERVE_MB") ? strtoull(getenv("RX_VK_PREDICT_RESERVE_MB"), nullptr, 10)
                                                                                    : predictlimits::kDefaultReserveMB;
    const auto worker = [pre, t0] {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        const IntelPrepareLane lane;   // (a preparation thread: its graphics pipelines side by side where intel_prepare_open)
        driver_cache_only = true;
        for (;;) {
            if (prepare_stopped() || pre->abandoned.load()) return;
            if (budget_s > 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(budget_s)) {
                pre->out_of_time = true;
                return;
            }
            if (predictlimits::low_commit(commit_left_mb(), reserve_mb)) {
                if (!pre->out_of_memory.exchange(true))
                    logf("vk: predicted pipelines: %llu MB of commit left (RX_VK_PREDICT_RESERVE_MB %llu): the prediction stops here, "
                         "the builds in hand are let finish, the rest are built in play", commit_left_mb(), reserve_mb);
                return;
            }
            const size_t i = pre->next++;
            if (i >= pre->keys.size()) return;
            const auto& list = pre->by_program[pre->keys[i]];
            size_t counted = 0;
            // (each pipeline once, by the key play looks it up by (pipeline_key): the prediction's vertex classes tell apart the
            // exact topology and each stream's stride, which a pipeline takes as dynamic state, and a mesh program takes no vertex
            // layout at all, so several predictions can be the one pipeline. Over the whole key, not each record: a variant kept
            // twice is borrowed twice. The program's form (mesh, mesh fallback) goes with it: pipeline_key does not hold it)
            std::unordered_set<uint64_t> built;
            std::vector<uint64_t> made_keys;
            program_cache_borrow(pre->keys[i], [&](Program& p) {
                ensure_layout(p);
                uint64_t words = 0;
                for (const Stage& st : p.stages) words += st.spirv_words;
                if (predictlimits::too_big(words, max_kwords)) {   // (counted done below, with the notes no variant answered)
                    pre->too_big++, pre->too_big_pipelines += list.size();
                    logf("vk: predicted pipelines: program %016llx left out, %llu k words of SPIR-V (RX_VK_PREDICT_MAX_KWORDS %llu): its "
                         "driver compile takes minutes and gigabytes; a draw that needs it builds it", (unsigned long long)p.key,
                         (unsigned long long)(words / 1000), max_kwords);
                    return;
                }
                for (const PipeNote* n : list)
                    if (n->variant == p.variant && !pre->abandoned.load()) {   // (gone on without it: no more of its list)
                        // (a mesh fallback program's own topology: its donor's may be an ordinary draw's, pipeline_topology)
                        const VkPrimitiveTopology topo = pipeline_topology(p, (VkPrimitiveTopology)n->topo);
                        const Targets at = note_targets(*n);
                        if (!built.insert(mix(pipeline_key(p, n->s, at, topo), (uint64_t)p.mesh | (uint64_t)p.fallback << 1)).second) {
                            pre->same++, pre->done++, counted++;
                            continue;
                        }
                        // (the first graphics pipeline alone, the rest in parallel, as in precompile_run: after the library warm,
                        // NVIDIA's driver (621.14, RTX 4070 Ti Super) faulted when all the workers began theirs at once, and the
                        // start hung on the lock it held then. This runs before precompile_run, so its first pipeline is the first)
                        std::unique_lock first(pre->first_graphics, std::defer_lock);
                        if (!pre->graphics_begun.load()) {
                            first.lock();
                            if (pre->graphics_begun.load()) first.unlock();
                        }
                        const VkPipeline pl = build_graphics(p, n->s, at, topo);
                        pre->graphics_begun = true;
                        if (first.owns_lock()) first.unlock();
                        if (pl) vkDestroyPipeline(dev.device, pl, nullptr), pre->made++, made_keys.push_back(pipeline_key(p, n->s, at, topo));
                        pre->done++, counted++;   // (each pipeline as it finishes: the stall check below sees a slow program move)
                    }
            });
            if (!made_keys.empty()) {
                std::lock_guard lk(pre->keys_mu);
                pre->made_keys.insert(pre->made_keys.end(), made_keys.begin(), made_keys.end());
            }
            if (counted < list.size()) pre->done += list.size() - counted;   // (and the notes no kept variant answered)
        }
    };
    // (the driver's faults before this run's workers start: one in their first milliseconds counts as this run's, below)
    const int faults_before = driver_faults();
    std::vector<std::thread> threads;
    const unsigned cpus = usable_cpus();
    const unsigned workers = prepare_pipeline_threads(std::min(cpus, 64u), "the predicted pipelines");
    for (unsigned i = 0; i < workers; i++) threads.emplace_back(worker);
    // (hookshot) Nothing finished for this long, on any worker (each pipeline counts as it finishes): the run goes on, the stuck
    // workers left in the driver, the rest of the predicted pipelines built in play when first drawn. When:
    // - the driver faulted in this run (driver_faults: caught, and a driver can stop answering after one): wherever the run is;
    // - otherwise only once every program has been handed out, as precompile_run's guard: the last stragglers, never the bulk. A
    //   slow driver that still works can sit up to 30 s on one pipeline (#2107), and what is given up here is built in play, at the
    //   game's own start too, where no watchdog is; a real hang before then is the --prepare-shaders watchdog's (vk_main.cpp).
    // Before, it waited for ever. RX_VK_PREDICT_STALL_S=<s>, 0: wait for all
    static const int stall_s = getenv("RX_VK_PREDICT_STALL_S") ? atoi(getenv("RX_VK_PREDICT_STALL_S")) : 30;
    size_t last_done = 0;
    auto last_change = std::chrono::steady_clock::now();
    bool gone_on = false;
    auto log_at = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    const auto built0 = std::chrono::steady_clock::now();
    while (next.load() < pre->keys.size() + threads.size() && !prepare_stopped() && !out_of_time.load() && !pre->out_of_memory.load()) {
        if (progress) progress(done.load(), predicted_n);
        prepare_progress_log("predicted pipelines", done.load(), predicted_n, built0, log_at);
        Sleep(100);
        const auto now = std::chrono::steady_clock::now();
        if (done.load() != last_done) last_done = done.load(), last_change = now;
        else if (const bool faulted = driver_faults() > faults_before;
                 stall_s > 0 && (faulted || next.load() >= pre->keys.size()) && now - last_change > std::chrono::seconds(stall_s)) {
            pre->abandoned = true;
            gone_on = true;
            prepare_left_workers_note();
            logf("vk: predicted pipelines: %zu of %zu done, nothing finished for %d s (%s): the rest are built in play", done.load(),
                 predicted_n, stall_s, faulted ? "the driver faulted, and is stuck since" : "the last ones stuck in the driver");
            break;
        }
    }
    if (gone_on) {
        for (std::thread& t : threads) t.detach();
    } else {
        for (std::thread& t : threads) t.join();
    }
    if (progress) progress(gone_on ? predicted_n : done.load(), predicted_n);
    const double all_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    logf("vk: predicted %zu pipelines from %zu notes for %zu program variants no note names: %zu made, %zu the same pipeline as "
         "another, in %.1f s (%.1f s deciding them, %.1f s building them on %zu threads); left out as no draw could ask for them: %zu "
         "by their vertex input types, %zu by their colour output types (%zu program variants with none left; %zu notes whose own "
         "types do not fit; %zu took a less common layout or targets that fit)%s", predicted_n, notes_n, programs, made.load(),
         pre->same.load(), all_s, decide_s, all_s - decide_s, threads.size(), left_out.vertex_types, left_out.output_types,
         left_out.variants_unfit, left_out.notes_unfit, left_out.refit,
         gone_on ? " (gone on without the pipelines the driver did not finish)"
         : out_of_time.load() ? " (the time budget, RX_VK_PREDICT_S, ran out: the rest are built in play)"
         : pre->out_of_memory.load() ? " (the system's commit ran low, RX_VK_PREDICT_RESERVE_MB: the rest are built in play)"
         : prepare_stopped() ? " (stopped: the window was closed; the driver keeps what was built)" : "");
    if (pre->too_big.load())
        logf("vk: predicted pipelines: %zu programs (%zu pipelines) left out for their size: built when a draw needs them", pre->too_big.load(),
             pre->too_big_pipelines.load());
    const bool cut = out_of_time.load() || pre->out_of_memory.load();   // (a budget ran out: what was built is not the whole)
    if (!prepare_stopped() && !cut && !gone_on) {
        predicted_sample_write(pre->predicted);
        predicted_keys_write(pre->made_keys);
        driver_env_note();   // (they are in the driver's cache alone: found only in this folder)
    }
    // (gone on: what is stuck is left to play, as the stall guard says; a start that does it again would only stall again)
    if (!prepare_stopped()) crash_markers::end(predict_pending_path());   // (closed: the close writes `began` back, not counted)
    driver_cache_log("after the prediction");
}

// ---- (hookshot) Whether the driver kept the predicted pipelines. predict_run builds them for the driver's own disk cache alone and
// lets them go: on a driver without one, or with one too small (Windows NVIDIA takes its size from the control panel), minutes of the
// preparation leave nothing, and nothing said so. A sample of them (hookshot_pipelines.bin.predicted) is asked of the driver at the
// next start, without compiling, as play asks: "the driver kept N of M". Also the driver cache folder's size in the log ----
namespace {
bool predicted_this_run = false;   // (asked in the process that built them, the driver answers from memory: no check then)
std::string predicted_sample_path() { return pipe_notes_path() + ".predicted"; }
}
void predicted_sample_write(const std::vector<std::unique_ptr<PipeNote>>& predicted) {
    predicted_this_run = true;
    constexpr size_t kSample = 32;
    std::vector<PipeNote> pick;
    for (size_t i = 0; i < predicted.size() && pick.size() < kSample; i += std::max<size_t>(1, predicted.size() / kSample))
        pick.push_back(*predicted[i]);
    const std::string path = predicted_sample_path();
    if (FILE* f = fopen(path.c_str(), "wb")) {
        const uint64_t head[3] = {program_cache_stamp(), sizeof(PipeNote), pick.size()};
        const bool ok = fwrite(head, 8, 3, f) == 3 && fwrite(pick.data(), sizeof(PipeNote), pick.size(), f) == pick.size();
        if (fclose(f) != 0 || !ok) remove(path.c_str());
    }
}
// ---- (hookshot) How often play meets a pipeline only the prediction made: hookshot_pipelines.bin.predicted_keys holds the keys of
// the pipelines the last prediction built (sorted, as graphics_pipeline looks them up). A pipeline play makes that no note named and
// whose key is there is counted once, by how it was made: from the driver's cache (the prediction saved its compile), compiled
// (the driver did not keep it), or from the start's cache. The line "vk: predicted pipelines met in play" follows each play file
// saved, for the session so far. What decides whether the prediction's minutes in the preparation are worth it ----
namespace {
std::string predicted_keys_path() { return pipe_notes_path() + ".predicted_keys"; }
std::atomic<int64_t> predicted_met[4];   // (by UseCounters::made_how: 0 not made, 1 from the driver's cache, 2 the start's, 3 compiled)
}
void predicted_keys_write(std::vector<uint64_t> keys) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const std::string path = predicted_keys_path();
    if (FILE* f = fopen(path.c_str(), "wb")) {
        const uint64_t head[2] = {program_cache_stamp(), keys.size()};
        const bool ok = fwrite(head, 8, 2, f) == 2 && fwrite(keys.data(), 8, keys.size(), f) == keys.size();
        if (fclose(f) != 0 || !ok) remove(path.c_str());
    }
}
/// Read the keys and retain whether the record was valid: a completed prediction can have zero keys.
bool predicted_keys_read(std::vector<uint64_t>& keys) {
    bool ok = false;
    if (FILE* f = fopen(predicted_keys_path().c_str(), "rb")) {
        uint64_t head[2]{};
        if (fread(head, 8, 2, f) == 2 && head[0] == program_cache_stamp() && head[1] <= (64u << 20) &&
            fseek(f, 0, SEEK_END) == 0 && ftell(f) == 16 + (long)head[1] * 8 && fseek(f, 16, SEEK_SET) == 0) {
            keys.resize((size_t)head[1]);
            ok = fread(keys.data(), 8, keys.size(), f) == keys.size();
        }
        fclose(f);
    }
    if (!ok) keys.clear();
    return ok;
}
bool predicted_complete() {
    std::vector<uint64_t> keys;
    return predicted_keys_read(keys);   // Read again: a preparation in this process may have just written it.
}
namespace {   // (declared in the anonymous namespace of graphics_pipeline and the play caches)
/// (read once, at the first pipeline play makes: a few hundred KB)
const std::vector<uint64_t>& predicted_keys() {
    static const std::vector<uint64_t> keys = [] {
        std::vector<uint64_t> k;
        predicted_keys_read(k);
        return k;
    }();
    return keys;
}
bool predicted_key(uint64_t key) {
    const std::vector<uint64_t>& k = predicted_keys();
    return std::binary_search(k.begin(), k.end(), key);
}
void predicted_use(int how) {
    predicted_met[how >= 0 && how < 4 ? how : 0]++;
}
void predicted_use_log() {
    const size_t known = predicted_keys().size();
    if (!known) return;
    logf("vk: predicted pipelines met in play (this session): %lld of the %zu the preparation predicted; %lld from the driver's cache, "
         "%lld compiled, %lld from the start's cache, %lld not made",
         (long long)(predicted_met[0] + predicted_met[1] + predicted_met[2] + predicted_met[3]), known, (long long)predicted_met[1].load(),
         (long long)predicted_met[3].load(), (long long)predicted_met[2].load(), (long long)predicted_met[0].load());
}
}  // namespace
/// The driver's shader cache folders this process uses (NVIDIA's, Mesa's layouts: driver_cache_env.h; on Windows NVIDIA's, AMD's,
/// Intel's), in bytes. The walk is bounded: Mesa's folder is shared by every program the player runs, and a many-file cache can
/// hold hundreds of thousands of files; past the bound the count is a floor (*partial)
uint64_t driver_cache_bytes(std::string* where, bool* partial) {
    namespace fs = std::filesystem;
    std::vector<fs::path> dirs;
#ifdef _WIN32
    const auto env = [](const char* n) -> std::string { const char* v = getenv(n); return v ? v : ""; };
    const std::string local = env("LOCALAPPDATA");
    if (!local.empty())
        for (const char* d : {"NVIDIA\\GLCache", "AMD\\VkCache", "Intel\\ShaderCache"}) dirs.push_back(fs::path(local) / d);
#else
    for (const std::string& d : driver_cache_env::folders(getenv)) dirs.push_back(fs::path(d));
#endif
    constexpr size_t kMaxEntries = 50000;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    size_t entries = 0;
    bool cut = false;
    uint64_t total = 0;
    for (const fs::path& d : dirs) {
        std::error_code ec;
        if (!fs::is_directory(d, ec)) continue;
        uint64_t n = 0;
        bool cut_here = false;
        for (auto it = fs::recursive_directory_iterator(d, fs::directory_options::skip_permission_denied, ec);
             !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (++entries > kMaxEntries || ((entries & 255) == 0 && std::chrono::steady_clock::now() > deadline)) {
                cut = cut_here = true;
                break;
            }
            if (it->is_regular_file(ec)) n += it->file_size(ec);
        }
        total += n;
        if (where)
            *where += (where->empty() ? "" : ", ") + d.string() + (cut_here ? " at least " : " ") + std::to_string(n >> 20) + " MB";
        if (cut) break;
    }
    if (partial) *partial = cut;
    return total;
}
#ifndef _WIN32
namespace {
/// (Linux) Whether the driver keeps its cache as Mesa does (RADV, ANV, NVK, ...) rather than as NVIDIA's does
bool driver_is_mesa() { return dev.driver_id != VK_DRIVER_ID_NVIDIA_PROPRIETARY && dev.driver_id != VK_DRIVER_ID_AMD_PROPRIETARY; }
/// RADV on a GPU where it can force VRS: it has VK_KHR_fragment_shading_rate from gfx10.3 on, where radv_device.c reads its
/// VRS switches (driver_cache_env.h)
bool radv_vrs_capable() {
    if (dev.driver_id != VK_DRIVER_ID_MESA_RADV || !dev.physical) return false;
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(dev.physical, nullptr, &n, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> exts(n);
    if (vkEnumerateDeviceExtensionProperties(dev.physical, nullptr, &n, exts.data()) != VK_SUCCESS) return false;
    for (const VkExtensionProperties& e : exts)
        if (!strcmp(e.extensionName, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) return true;
    return false;
}
driver_cache_env::Fingerprint driver_fingerprint(bool* radv_vrs = nullptr, bool* forced = nullptr) {
    static const bool vrs = radv_vrs_capable();
    const bool f = driver_cache_env::radv_vrs_forced(getenv, vrs, [](const std::string& path) {
        std::error_code ec;
        return std::filesystem::exists(path, ec);
    });
    if (radv_vrs) *radv_vrs = vrs;
    if (forced) *forced = f;
    return driver_cache_env::fingerprint(getenv, driver_is_mesa(), f);
}
/// hookshot_driver_env.txt: the environments the starts that built the noted pipelines here ran with (driver_cache_env.h records)
std::string driver_env_path() { return cache_path("hookshot_driver_env.txt"); }
std::string driver_env_read() {
    std::string text;
    if (FILE* f = fopen(driver_env_path().c_str(), "rb")) {
        char buf[4096];
        const size_t n = fread(buf, 1, sizeof buf, f);
        text.assign(buf, n);
        fclose(f);
    }
    return text;
}
}  // namespace
#endif
/// (Linux) The environment the noted (or predicted) pipelines were just built in, added to the records for later starts
void driver_env_note() {
#ifndef _WIN32
    if (!dev.device) return;
    const std::string text = driver_cache_env::records_text(driver_cache_env::records_read(driver_env_read()), driver_fingerprint());
    const std::string path = driver_env_path(), part = path + ".part";
    if (FILE* f = fopen(part.c_str(), "wb")) {
        const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
        if (fclose(f) == 0 && ok) rename(part.c_str(), path.c_str());
        else remove(part.c_str());
    }
#endif
}
void driver_cache_log(const char* when) {
    std::string where;
    bool partial = false;
    const uint64_t n = driver_cache_bytes(&where, &partial);
    logf("vk: driver shader cache %s: %s%llu MB (%s)", when, partial ? "at least " : "", (unsigned long long)(n >> 20),
         where.empty() ? "no folder found" : where.c_str());
    // (NVIDIA's limit: under 4 GB the pipelines a start builds do not all stay, and are compiled again in play; the launcher
    // raises Windows' to 10 GB, hos_main.c driver_cache_room Linux's)
    if (strcmp(when, "at the start") == 0 && dev.device && dev.driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
        const nvidia_cache_limit::Limit limit = nvidia_limit();
        if (limit.off) logf("vk: driver shader cache: NVIDIA's is off (%s)", limit.where);
        else if (!limit.mb) logf("vk: driver shader cache: NVIDIA's limit not known");
        else if (nvidia_cache_limit::small(limit.mb))
            logf_at(log_warn, "vk: driver shader cache: NVIDIA's holds at most %llu MB (%s), under the %llu MB a start's pipelines "
                              "need: the ones it drops are compiled again in play. Raise it to 10 GB or Unlimited (the launcher "
                              "does it before a build or a start)", (unsigned long long)limit.mb, limit.where,
                    (unsigned long long)nvidia_cache_limit::kRoomMb);
        else if (limit.mb == nvidia_cache_limit::kUnlimited) logf("vk: driver shader cache: NVIDIA's has no limit (%s)", limit.where);
        else logf("vk: driver shader cache: NVIDIA's holds at most %llu MB (%s)", (unsigned long long)limit.mb, limit.where);
    }
#ifndef _WIN32
    if (strcmp(when, "at the start") != 0 || !dev.device) return;
    bool radv_vrs = false, forced = false;
    const driver_cache_env::Fingerprint now = driver_fingerprint(&radv_vrs, &forced);
    logf("vk: driver shader cache: %s", driver_cache_env::describe(getenv, driver_is_mesa(), radv_vrs, forced).c_str());
    const driver_cache_env::Verdict v = driver_cache_env::compare(driver_cache_env::records_read(driver_env_read()), now);
    if (v.other_keys) {
        logf_at(log_warn, "vk: driver shader cache: this start's pipeline keys (%s) are not those of any start that built the "
                          "pipelines here (the last: %s): neither our pipeline cache nor the driver's has its pipelines, and each "
                          "is compiled again, before the title or in play", now.keys.c_str(), v.keys_before.c_str());
    } else if (v.other_folder) {
        std::error_code ec;
        if (std::filesystem::exists(predicted_keys_path(), ec))
            logf("vk: driver shader cache: the predicted pipelines were built into %s; this start uses %s, so each is compiled "
                 "when it is first drawn", v.folder_before.c_str(), now.folder.c_str());
    }
#endif
}
/// (the start after a preparation, behind the game on a thread of its own) the sample asked of the driver, logged
void predicted_check() {
    if (predicted_this_run) return;
    // (not in a --prepare-shaders run (vk_main.cpp prepare_only): it exits right after, and a detached thread still asking the
    // driver then would race the process's exit, its static destructors and the driver's teardown; the game's next start checks)
    if (env_on("HOOKSHOT_PREPARE_ONLY")) return;
    // (asking without compiling, VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT, needs pipelineCreationCacheControl, as
    // the play caches' and the stage libraries' asks do: without it the flag is invalid and the "ask" would compile every one)
    if (!dev.vk13.pipelineCreationCacheControl) return;
    std::vector<PipeNote> sample;
    if (FILE* f = fopen(predicted_sample_path().c_str(), "rb")) {
        uint64_t head[3]{};
        if (fread(head, 8, 3, f) == 3 && head[0] == program_cache_stamp() && head[1] == sizeof(PipeNote) && head[2] <= 64) {
            sample.resize((size_t)head[2]);
            if (fread(sample.data(), sizeof(PipeNote), sample.size(), f) != sample.size()) sample.clear();
        }
        fclose(f);
    }
    if (sample.empty()) return;
    std::thread([sample] {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
        const auto t0 = std::chrono::steady_clock::now();
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        VkPipelineCache scratch = VK_NULL_HANDLE;
        if (vkCreatePipelineCache(dev.device, &ci, nullptr, &scratch) != VK_SUCCESS) scratch = VK_NULL_HANDLE;
        driver_cache_only = true, driver_ask_cache = scratch;
        size_t asked = 0, kept = 0;
        for (const PipeNote& n : sample)
            program_cache_borrow(n.prog, [&](Program& p) {
                if (p.variant != n.variant) return;
                ensure_layout(p);
                bool cold = false;
                const VkPipeline pl = build_graphics(p, n.s, note_targets(n), pipeline_topology(p, (VkPrimitiveTopology)n.topo), -1, &cold);
                asked++;
                if (pl) kept++, vkDestroyPipeline(dev.device, pl, nullptr);
            });
        driver_cache_only = false, driver_ask_cache = VK_NULL_HANDLE;
        if (scratch) vkDestroyPipelineCache(dev.device, scratch, nullptr);
        logf("vk: the driver's shader cache kept %zu of %zu predicted pipelines asked (a sample of the last preparation's, in %.0f ms)%s",
             kept, asked, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
             asked && !kept ? ": none, so the predicted pipelines are compiled when first drawn on this driver" : "");
    }).detach();
}

/// (hookshot) Intel's one-at-a-time compute builds (serial_builds_on): a dispatch's pipeline is made first without the optimizer
/// (the Debug tab's Intel quick compute builds row, RX_VK_COMPUTE_QUICK_FIRST), then with it here, behind the builds queued
/// first. With the optimizer, Intel's compiler took 0.3 s to 6.5 min for each of 37 of the game's dispatches on a UHD Graphics
/// (i3-1115G4), 22 minutes on its one thread, each pass over the scene (its light, its exposure) left out until its turn: the
/// Deku Tree's dialogue drawn blown out. The dispatch binds this one once made; the first stays for the session, as a quick
/// graphics pipeline does (a recording may hold it)
void compute_optimize(Program& q) {
    VkComputePipelineCreateInfo ci = compute_pipeline_info(q, true);
    const auto t0 = std::chrono::steady_clock::now();
    VkPipeline made = VK_NULL_HANDLE;
    if (const VkResult r = cached_compute(&ci, &made); r != VK_SUCCESS || !made) {
        static std::atomic<int> told{0};
        if (told++ < 20)
            logf("vk: compute pipeline of %016llx: the optimized build was not made (%d); the one without the optimizer stays",
                 (unsigned long long)q.key, (int)r);
        return;
    }
    q.compute_optimized.store(made, std::memory_order_release);
    pcache_built_since_save++;
    st_pipelines++;
    const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    static std::atomic<int> told{0};
    if (ms >= 200 || told++ < 20)
        logf("vk: compute pipeline of %016llx: optimized in %lld ms, bound from now on", (unsigned long long)q.key, (long long)ms);
}


void dispatch(VkCommandBuffer cmd, NvnState& s, uint32_t x, uint32_t y, uint32_t z, uint64_t indirect) {
    if (par_stats_on) par_stats_draw();
    PROF(PROF_DISPATCH);
    handle_memo_next();   // (hookshot: a new draw or dispatch: the handles kept for the last one go)
    DispatchClock dclock;
    s.grid[0] = indirect ? 1 : x, s.grid[1] = indirect ? 1 : y, s.grid[2] = indirect ? 1 : z;
    if (indirect) {   // (the program may read its grid from c0: the counts as the GPU will see them, best effort)
        uint32_t g[3] = {1, 1, 1};
        if (memory_read_as_submitted(indirect, g, 12)) s.grid[0] = g[0], s.grid[1] = g[1], s.grid[2] = g[2];
    }
    const uint8_t types[8]{};
    ProgramUse use(s.prog, s.has_prog, true);   // (the program's first dispatch, for the census: vk_programs.h)
    const Program* cp = program_for(s, ProgramKind::Compute, types);
    if (use.on) use.program(cp, cp && cp->ready.load(std::memory_order_acquire));
    dclock.t_prog = slow_dispatch_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // (a program still being translated is waited for, as a draw's is: a dispatch left out is work never done again.
    // MKW updates its terrain height clipmaps a region at a time as the camera moves, and makes its sky's tables
    // once; a skipped update left holes and flat patches in the terrain, garbage in the sky)
    if (cp && !cp->ready.load(std::memory_order_acquire)) {
        const auto t0 = std::chrono::steady_clock::now();
        jobs_wait(cp->ready);
        st_waits++;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        st_wait_ms += ms, st_shader_wait_ms += ms;
    }
    if (!cp || cp->failed) {   // (hookshot) the program's resources, once a control section: what a rejected dispatch reads/writes
        static std::mutex mu2;
        static std::unordered_set<uint64_t> seen2;
        const uint64_t id = s.prog.sd[0].control;
        std::lock_guard lk2(mu2);
        if (seen2.size() < 64 && seen2.insert(id).second) {
            char line[1400];
            int n = snprintf(line, sizeof line, "vk: rejected compute %016llx (control %#llx) grid %ux%ux%u:",
                             (unsigned long long)compute_code_key(s), (unsigned long long)id, s.grid[0], s.grid[1], s.grid[2]);
            auto desc = [&](const char* what, uint64_t h) {
                nvn_tex t;
                nvn_view v;
                bool hv = false;
                if (!h || !texture_of((int)(h & 0xFFFFF), &t, &v, &hv)) return;
                if (n < (int)sizeof line - 96)
                    n += snprintf(line + n, sizeof line - n, " %s %#llx %ux%ux%u fmt %#x;", what,
                                  (unsigned long long)t.gpu, t.d.width, t.d.height, t.d.depth, (unsigned)t.d.format);
            };
            for (int i = 0; i < 8; i++) desc("img", s.img[5][i]);
            for (int i = 0; i < 16; i++) desc("tex", s.tex[5][i]);
            for (int i = 0; i < 16; i++)
                if (s.ssbo[5][i].addr && n < (int)sizeof line - 64)
                    n += snprintf(line + n, sizeof line - n, " ssbo%d %#llx+%#llx;", i, (unsigned long long)s.ssbo[5][i].addr,
                                  (unsigned long long)s.ssbo[5][i].size);
            for (int i = 0; i < 8; i++)
                if (s.ubo[5][i].addr && n < (int)sizeof line - 64)
                    n += snprintf(line + n, sizeof line - n, " ubo%d %#llx+%#llx;", i, (unsigned long long)s.ubo[5][i].addr,
                                  (unsigned long long)s.ubo[5][i].size);
            logf("%s", line);
        }
    }
    if (cp && cp->failed) {   // (hookshot) which programs' dispatches are left out, once each, with their grids
        static std::mutex mu;
        static std::unordered_set<uint64_t> seen;
        std::lock_guard lk(mu);
        if (seen.size() < 64 && seen.insert(cp->key).second)
            logf("vk: dispatches of program %016llx are left out (it did not translate): grid %ux%ux%u%s",
                 (unsigned long long)cp->key, s.grid[0], s.grid[1], s.grid[2], indirect ? " (indirect)" : "");
    }
    // (RX_VK_FORCE_UPSCALE_STUB=1, a test: every upscale network takes the fallback, as on a GPU where they do not translate. The
    // neon colours of issue 425/508 come from that path, and an NVIDIA PC never took it)
    // (=1: the networks other than gameplay's, which on AMD take this path; gameplay's has its stand-in there. =2: all)
    static const int force_stub = getenv("RX_VK_FORCE_UPSCALE_STUB") ? atoi(getenv("RX_VK_FORCE_UPSCALE_STUB")) : 0;
    const bool slow_network = cp && !cp->failed && !cp->stages.empty() &&
                              (movie_upscale_policy::use_linear(movie_upscale_mode(),
                                  movie_networks_slow.load(std::memory_order_relaxed), movie_keep_all_networks(),
                                  cp->stages[0].info.uses_cooperative_matrix, gameplay_network(cp->key)) ||
                               (force_stub && cp->stages[0].info.uses_cooperative_matrix &&
                                (force_stub >= 2 || !gameplay_network(cp->key))));
    if ((!cp || cp->failed || slow_network) && upscale_stub(cmd, s, compute_code_key(s))) {
        // (a gameplay upscale's picture, as the dispatch below copies its own: frame generation's HUD mark)
        if (stub_picture && fsr3fi::frames_wanted()) framegen_hud_before(cmd, stub_picture, 0);
        return void(st_dispatches++);
    }
    if (slow_network) SKIP_HERE();   // (an earlier pass of the network: the stub is its last pass)
    if (!cp || !cp->ready.load(std::memory_order_acquire) || cp->failed) SKIP_HERE();
    {   // (hookshot, RX_VK_SKIP_PROG=<key prefix,...>: those programs' dispatches left out, for diagnosis)
        static const char* skip = getenv("RX_VK_SKIP_PROG");
        if (skip) {
            char hex[24];
            snprintf(hex, sizeof hex, "%016llx", (unsigned long long)cp->key);
            for (const char* q = skip; *q;) {
                const char* e = q;
                while (*e && *e != ',') e++;
                if (e > q && !strncmp(hex, q, (size_t)(e - q))) return;
                q = *e ? e + 1 : e;
            }
        }
    }
    dclock.key = cp->key;
    texture_watch_prog(cp->key);
    {   // (hookshot, RX_VK_LOG_PUSH=<key prefix>: once, that program's push constants and the c0 words it reads)
        static const char* want = getenv("RX_VK_LOG_PUSH");
        static bool done;
        char hex[24];
        if (want && !done) snprintf(hex, sizeof hex, "%016llx", (unsigned long long)cp->key);   // (only with the switch: once per dispatch)
        if (want && !done && !strncmp(hex, want, strlen(want))) {
            done = true;
            char line[1600];
            int n = snprintf(line, sizeof line, "vk: program %s: %u bytes pushed; c0:", hex, s.cs_push_n);
            for (uint32_t off = 0; off < 0x100 && n < (int)sizeof line - 16; off += 4)
                n += snprintf(line + n, sizeof line - n, " %08x", driver_bank_word(s, 0, 5, off));
            logf("%s", line);
        }
    }
    Program& p = const_cast<Program&>(*cp);
    if (!p.layout) {
        ParLock lk;
        ensure_layout(p);
    }
    // (the compute pipeline is compiled on a worker, not here with the recording lock held: AMD's compiler took 15-23 s on the
    // culling programs, every recording thread stopped behind it and the screen stayed black after the opening (6950 XT, RX 6600,
    // 2026-11-08). The dispatch waits up to 50 ms for it, which covers NVIDIA's compiles; one still compiling after that is left
    // out until it is ready, as graphics draws are)
    // (compute_pipeline is written by the build worker and the precompile's threads while dispatches read it: always atomically)
    const auto cpipe = [&p] { return std::atomic_ref<VkPipeline>(p.compute_pipeline).load(std::memory_order_acquire); };
    if (use.on) use.pipeline(cpipe() != VK_NULL_HANDLE, false);
    if (!cpipe() && !taau_key(p.key) && p.compute_state.load(std::memory_order_acquire) != 2) {
        int none = 0;
        // (a program read from the cache without its code, vk_progcache.cpp: its module made now, by the dispatch that starts its
        // pipeline. One whose code does not read is failed, and left out below. On Windows Intel the compiler's gate is only
        // tried for it (program_cache_complete_ask, IntelDriverGate): free, the module is made here and the pipeline asked of
        // the caches below, as on other GPUs; held by a worker's compile, both are left to the worker (on_worker) rather than
        // waited for here. Its code is looked at once the CAS is won, under the cache's lock: before that, a worker started by
        // another dispatch may be clearing it, complete_from)
        bool on_worker = false;
        const bool fresh = p.compute_state.compare_exchange_strong(none, 1) &&
                           (program_cache_complete_ask(p, &on_worker) || on_worker ||
                            (p.compute_state.store(2, std::memory_order_release), false));
        if (on_worker) intel_gate_busy();
        std::shared_ptr<std::atomic<uint8_t>> made_how;   // (on_worker: how the worker made it, for the census)
        if (fresh) {
            { ParLock lk; pcache_init(); }
            const Program* job = &p;
            // (one at a time on Intel: serial_builds_on)
            static const bool serial = serial_builds_on();
            // (one the prepare left in our cache is only asked for here: it comes back at once. Queued, it waited for a worker
            // behind their graphics builds, and its dispatch was left out)
            const VkPipeline cached = on_worker ? VK_NULL_HANDLE : compute_pipeline_cached(p, true);
            if (on_worker) made_how = std::make_shared<std::atomic<uint8_t>>(0);
            auto build = [job, on_worker, made_how] {
                Program& q = const_cast<Program&>(*job);
                use_counters.made_how = 0;
                if (on_worker) {
                    if (!program_cache_complete(q)) {
                        q.failed = true;
                        q.compute_state.store(2, std::memory_order_release);
                        return;
                    }
                    if (const VkPipeline c = compute_pipeline_cached(q, true)) {
                        std::atomic_ref<VkPipeline>(q.compute_pipeline).store(c, std::memory_order_release);
                        st_pipelines++;
                        // (from a cache: the play caches say which, create_cached; before play, the start's, as below)
                        made_how->store(use_counters.made_how ? use_counters.made_how : 2, std::memory_order_relaxed);
                        cs_module_release(q);
                        q.compute_state.store(2, std::memory_order_release);
                        return;
                    }
                }
                VkComputePipelineCreateInfo ci = compute_pipeline_info(q, true);
                // (hookshot) on Intel's one-at-a-time builds, without the optimizer first: compute_optimize
                const bool quick = serial && intel_quick_compute() && !(ci.flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT);
                if (quick) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
                const auto t0 = std::chrono::steady_clock::now();
                static const bool trace = env_on("RX_VK_PIPELINE_TRACE");
                if (trace) logf("vk: compute pipeline begin: program %016llx", (unsigned long long)q.key);
                VkPipeline made = VK_NULL_HANDLE;
                if (const VkResult r = cached_compute(&ci, &made); r != VK_SUCCESS || !made) {
                    static std::atomic<int> told{0};
                    if (told++ < 20) logf("vk: compute pipeline of %016llx not made (%d)", (unsigned long long)q.key, (int)r);
                    q.failed = true;
                } else {
                    std::atomic_ref<VkPipeline>(q.compute_pipeline).store(made, std::memory_order_release);
                    if (quick) jobs_push_serial([job] { compute_optimize(const_cast<Program&>(*job)); });
                    else cs_module_release(q);
                }
                const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
                if (ms >= 200)
                    logf("vk: compute pipeline of %016llx: SLOW compile, %lld ms (on a worker%s)", (unsigned long long)q.key, (long long)ms,
                         quick ? ", without the optimizer" : "");
                st_pipeline_us += ms * 1000;
                st_pipelines++;
                pcache_built_since_save++;
                if (made_how) made_how->store(use_counters.made_how ? use_counters.made_how : 3, std::memory_order_relaxed);
                q.compute_state.store(2, std::memory_order_release);
            };
            // (from a cache, or compiled on a worker; on_worker: as the worker made it, once it has, below)
            if (use.on) use.pipeline(false, true);
            if (use.on && !on_worker) use.pipeline_made(cached ? 2 : 3);
            if (cached) {
                std::atomic_ref<VkPipeline>(p.compute_pipeline).store(cached, std::memory_order_release);
                st_pipelines++;
                cs_module_release(p);
                p.compute_state.store(2, std::memory_order_release);
            } else {
                auto failed = [job]() noexcept {
                    Program& q = const_cast<Program&>(*job);
                    std::atomic_ref<VkPipeline>(q.compute_pipeline).store(VK_NULL_HANDLE, std::memory_order_release);
                    q.failed = true;
                    q.compute_state.store(2, std::memory_order_release);
                };
                // (first: a dispatch is a pass over the scene (its light, its exposure), left out while it builds; the
                // mesh fallback's are a model each. Kokiri Forest's first dispatch on a UHD Graphics waited minutes behind them)
                if (serial) jobs_push_serial(std::move(build), failed, true);
                else jobs_push(std::move(build), true, nullptr, failed);
            }
        }
        // (only the dispatch that started the build waits: later ones while it still compiles are left out at once, not 50 ms
        // each, which stalled every frame by 50 ms per compute program still building)
        const auto t0 = std::chrono::steady_clock::now();
        while (fresh && p.compute_state.load(std::memory_order_acquire) != 2 &&
               std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(50))
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        if (use.on && made_how && p.compute_state.load(std::memory_order_acquire) == 2)
            use.pipeline_made(made_how->load(std::memory_order_relaxed));
        if (p.compute_state.load(std::memory_order_acquire) != 2) {
            static std::atomic<int> told{0};
            if (told++ < 40) logf_at(log_debug, "vk: dispatch of %016llx left out: its compute pipeline is still compiling", (unsigned long long)p.key);
            st_pending_skips++;
            use.skipped();
            return;
        }
        if (p.failed || !cpipe()) SKIP_HERE();
    }
    ParLock plk(!cpipe());
    if (!cpipe() && !taau_key(p.key)) {
        if (!program_cache_complete(p)) SKIP_HERE();   // (read without its code: see above)
        pcache_init();
        VkComputePipelineCreateInfo ci = compute_pipeline_info(p, true);   // (RX_VK_NOOPT=1, as for graphics pipelines above)
        const auto t0 = std::chrono::steady_clock::now();
        {   // (EvilOOT, 11-07: a freeze after Load was the driver's compiler on one compute pipeline, here, on the submit's thread)
            static std::atomic<int> told{0};
            if (told++ < 400) logf_at(log_trace, "vk: compute pipeline of %016llx: compiling", (unsigned long long)p.key);
        }
        VkPipeline built = VK_NULL_HANDLE;   // (published only once made: other threads read it meanwhile)
        if (const VkResult made = cached_compute(&ci, &built); made != VK_SUCCESS) {
            static std::atomic<int> told{0};
            if (told++ < 20) logf("vk: compute pipeline of %016llx not made (%d)", (unsigned long long)p.key, (int)made);
            p.failed = true;
            SKIP_HERE();
        }
        std::atomic_ref<VkPipeline>(p.compute_pipeline).store(built, std::memory_order_release);
        {
            const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
            if (ms >= 200) logf("vk: compute pipeline of %016llx: SLOW compile, %lld ms", (unsigned long long)p.key, (long long)ms);
        }
        st_pipeline_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        st_pipelines++;
        pcache_built_since_save++;
        if (std::atomic_ref<VkPipeline>(p.compute_pipeline).load(std::memory_order_acquire)) cs_module_release(p);   // (RX_VK_CS_MODULE_DROP)
    }
    if (plk.on) par_mutex().unlock(), plk.on = false;
    dclock.t_pipe = slow_dispatch_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    end_pass(cmd);
    grid_patch_want = indirect, grid_patch_at = UINT64_MAX;
    exposure_meter_target = {};
    exposure_meter_bindings = 0;
    build_descriptors(s, p);
    grid_patch_want = false;
    dclock.t_desc = slow_dispatch_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (dump_on())
        logf("vk: dump-in %03d dispatch %ux%ux%u prog %016llx:%s", dump_seq(), x, y, z, (unsigned long long)p.key,
             dump_inputs().c_str());
    if (dump_on()) {   // (amd-fix, diagnosis) what a dispatch can reach: its storage and uniform buffers, its pointer table
        char line[1800];
        int n = snprintf(line, sizeof line, "vk: dump-reach %03d prog %016llx%s:", dump_seq(), (unsigned long long)p.key,
                         !p.stages.empty() && p.stages[0].info.stores_global_memory ? " (stores global)" : "");
        for (int i = 0; i < 16 && n < 1700; i++)
            if (s.ssbo[5][i].addr)
                n += snprintf(line + n, sizeof line - n, " sb%d %#llx+%#llx", i, (unsigned long long)s.ssbo[5][i].addr,
                              (unsigned long long)s.ssbo[5][i].size);
        for (int i = 0; i < 16 && n < 1700; i++)
            if (s.ubo[5][i].addr)
                n += snprintf(line + n, sizeof line - n, " ub%d %#llx+%#llx", i, (unsigned long long)s.ubo[5][i].addr,
                              (unsigned long long)s.ubo[5][i].size);
        if (!p.stages.empty()) {   // the translated program's own storage bindings: which constant-buffer word names each
            const auto& sd = p.stages[0].info.storage_buffers_descriptors;
            for (size_t i = 0; i < sd.size() && n < 1700; i++) {
                const uint32_t off = sd[i].cbuf_offset;
                const uint64_t a = stage_cbuf_word(s, p.stages[0], sd[i].cbuf_index, off) |
                                   (uint64_t)stage_cbuf_word(s, p.stages[0], sd[i].cbuf_index, off + 4) << 32;
                n += snprintf(line + n, sizeof line - n, " ssbo%zu=c%u+%#x%s%s->%#llx", i, sd[i].cbuf_index, off,
                              sd[i].is_written ? "w" : "", sd[i].count > 1 ? "[t]" : "", (unsigned long long)a);
            }
        }
        logf("%s", line);
    }
    // (dispatches after other work wait for it: the WebGPU renderer's dispatches were ordered the same way by
    // wgpu's own tracking, and MKW's compute chains read what the one before wrote)
    // (hookshot, RX_VK_GAME_ORDER: a direct dispatch right after another, nothing recorded between, gets none: game_order)
    const bool after_dispatch = !indirect && cmd == last_dispatch_cmd &&
                                work_epoch.load(std::memory_order_relaxed) == last_dispatch_epoch && game_order() &&
                                !par_active.load(std::memory_order_relaxed);
    if (work_since_barrier && !after_dispatch) full_barrier(cmd, "barrier (dispatch)");
    else if (after_dispatch && ++st_dispatch_unordered % 50000 == 1)
        logf("vk: %lld dispatches right after another ordered by the game's barriers so far, with none of their own (RX_VK_GAME_ORDER)",
             (long long)st_dispatch_unordered);
    work_since_barrier = true;
    if (!p.stages.empty() && p.stages[0].info.uses_cooperative_matrix) {   // (the game's networks, once each: what they read and write)
        static std::mutex mu;
        static std::unordered_map<uint64_t, int> seen;
        std::lock_guard lk(mu);
        if (seen[p.key]++ == 0) {
            char line[900];
            int n = snprintf(line, sizeof line, "vk: network %016llx: dispatch %ux%ux%u, %zu textures %zu images:", (unsigned long long)p.key, x, y, z,
                             p.stages[0].info.texture_descriptors.size(), p.stages[0].info.image_descriptors.size());
            for (const Texture* t : draw_sampled())
                if (t && n < 860) n += snprintf(line + n, sizeof line - n, " %ux%ux%u/%u", t->width, t->height, t->depth, (unsigned)t->format);
            logf("%s", line);
        }
    }
    // (a gameplay upscale's picture, copied for frame generation's HUD mark once it is written, whoever writes it)
    uint32_t picture_level = 0;
    Texture* const picture = upscale_picture(s, p, &picture_level);
    if (fsr3_stand_in(cmd, s, p))   // (the item wheel's scene, and frame generation's HUD mark)
        return wheel_scene_note(cmd, s, p, true), framegen_hud_before(cmd, picture, picture_level), void(st_dispatches++);
    if (temporal_stand_in(cmd, s, p))   // (the item wheel's scene, and frame generation's HUD mark)
        return wheel_scene_note(cmd, s, p, true), framegen_hud_before(cmd, picture, picture_level), void(st_dispatches++);
    if (taau_stand_in(cmd, s, p))   // (the item wheel's scene, and frame generation's HUD mark)
        return wheel_scene_note(cmd, s, p, true), framegen_hud_before(cmd, picture, picture_level), void(st_dispatches++);
    const VkPipeline pipe = cpipe();
    if (!pipe) SKIP_HERE();
    dump_ssbos(cmd, p, "before");
    if (!p.dispatched.load(std::memory_order_relaxed)) compute_variant_note(p.key, p.variant);   // (which variant: this install's own)
    if (!p.dispatched.exchange(true, std::memory_order_relaxed)) compute_note(p.key), state_record_compute(p.key);
    if (p.optimize_owed.load(std::memory_order_relaxed) && p.optimize_owed.exchange(false, std::memory_order_acq_rel)) {
        Program* q = &p;
        jobs_push_serial([q] { compute_optimize(*q); });
    }
    const VkPipeline optimized = p.compute_optimized.load(std::memory_order_acquire);   // (compute_optimize)
    if (res_log_on()) res_note_pipeline_use(optimized ? optimized : pipe);   // (RX_VK_RES_LOG)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, optimized ? optimized : pipe);
    if (!bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p)) SKIP_HERE();
    push_constants(cmd, p, nullptr);
    gprof_begin(cmd);
    if (stats_frame_now()) {   // (EvilOOT, RX_VK_DRAW_STATS) the pointers this dispatch's small buffers hold
        const auto scan = [&](const nvn_p_bind_buf& b, uint64_t most) {
            if (!b.addr || !b.size) return;
            const uint64_t n = std::min<uint64_t>(b.size, most) & ~7ull;
            std::vector<uint64_t> w(n / 8);
            if (w.empty() || !memory_read_as_submitted(b.addr, w.data(), n)) return;
            std::lock_guard bl(stats_bind_mu);
            for (const uint64_t v : w)
                if (!(v & 3) && v >= (1ull << 32) && v < (1ull << 40) && host->gpu_guest(v) && stats_ptrs.size() < 400000)
                    stats_ptrs.push_back({p.key, b.addr, v});
        };
        for (const nvn_p_bind_buf& b : s.ssbo[5]) scan(b, 0x400);
        for (const nvn_p_bind_buf& b : s.ubo[5]) scan(b, 0x1000);
        {   // (second level) pointers inside the pointer table's regions that nothing this dispatch has covers
            const PointerEntry* e = pointer_table(s, 5);
            const auto covered = [&](uint64_t v) {
                for (const nvn_p_bind_buf& b : s.ssbo[5])
                    if (b.addr && v >= b.addr && v < b.addr + b.size) return true;
                for (int i = 0; i < 3; i++)
                    if (e[i].addr && v >= e[i].addr && v < e[i].addr + e[i].size) return true;
                return false;
            };
            for (int i = 0; i < 3; i++) {
                if (!e[i].addr || !e[i].size) continue;
                const uint64_t n = std::min<uint64_t>(e[i].size, 256u << 10) & ~7ull;
                std::vector<uint64_t> w(n / 8);
                if (w.empty() || !memory_read_as_submitted(e[i].addr, w.data(), n)) continue;
                uint32_t found = 0;
                std::lock_guard bl(stats_bind_mu);
                for (const uint64_t v : w)
                    if (!(v & 3) && v >= (1ull << 32) && v < (1ull << 40) && host->gpu_guest(v) && gpu_pool_end(v) && !covered(v) &&
                        found++ < 64 && stats_ptrs2.size() < 400000)
                        stats_ptrs2.push_back({p.key, e[i].addr, v});
            }
        }
        if (stats_prog_wanted(p.key)) stats_dump_stage(p.key, s, 5);   // (with each buffer's non-zero words)
    }
    if (stats_dir()) watch_note_sb(p.key, s);   // (RX_VK_WATCH_SB)
    {
        StatsQuery stats_query(cmd, p, indirect, x, y, z);   // (EvilOOT, RX_VK_DRAW_STATS)
        {   // (RX_VK_GPU_DUMP=1 with RX_VK_DRAW_STATS_PROGS: the wanted dispatches' storage buffers as the GPU holds them after)
            static const bool gpu_dump_on = env_on("RX_VK_GPU_DUMP");
            if (gpu_dump_on && stats_query.cmd && stats_prog_wanted(p.key)) stats_query.state = &s, stats_query.prog = p.key;
        }
        if (indirect) {
            ledger_mark(indirect, 12, kUseIndirect);   // (RX_VK_LEDGER)
            BufferRef args = gpu_buffer_bind(indirect, 12, false);
            if (!args) SKIP_HERE();
            if (grid_patch_at != UINT64_MAX) {   // (PR 252) the grid as the GPU has it, into the program's c0 (the full barrier
                // above: the pass that wrote the arguments is done)
                const VkBufferCopy gc{args.offset, grid_patch_at, 12};
                vkCmdCopyBuffer(cmd, args.buffer, grid_patch_buf, 1, &gc);
                VkMemoryBarrier2 gb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                gb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, gb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                gb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, gb.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
                VkDependencyInfo gd{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                gd.memoryBarrierCount = 1, gd.pMemoryBarriers = &gb;
                vkCmdPipelineBarrier2(cmd, &gd);
                grid_patch_at = UINT64_MAX;
            }
            checkpoint(cmd, p, 'i');   // (a device lost's report names it: a grid read from GPU memory)
            if (dump_on() && dump_ssbo_wanted(p.key)) {   // (RX_VK_DUMP_SSBO: the grid as the GPU holds it, and where)
                char what[128];
                snprintf(what, sizeof what, "%03d prog %016llx indirect grid at %#llx", dump_seq(), (unsigned long long)p.key,
                         (unsigned long long)indirect);
                dump_buffer(cmd, args.buffer, args.offset, 12, what);
            }
            vkCmdDispatchIndirect(cmd, args.buffer, args.offset);
        } else {
            // The game's exposure writer accumulates four sums. Clear only an
            // unambiguous binding of that known meter, then keep the exact wait.
            if (x && y && z && exposure_meter_bindings == 1 && exposure_meter_target &&
                !(exposure_meter_target.offset & 3)) {
                full_barrier(cmd, "exposure meter before clear");
                vkCmdFillBuffer(cmd, exposure_meter_target.buffer, exposure_meter_target.offset, 16, 0);
                full_barrier(cmd, "exposure meter before accumulation");
                static std::atomic<uint64_t> resets{0};
                const uint64_t n = ++resets;
                if (n <= 4 || n % 300 == 0)
                    logf("vk: exposure meter: %llu GPU resets with exact writer-part tracking", (unsigned long long)n);
            }
            vkCmdDispatch(cmd, x, y, z);
        }
        last_dispatch_cmd = cmd, last_dispatch_epoch = work_epoch.load(std::memory_order_relaxed);
    }
    wheel_scene_note(cmd, s, p, false);   // (hookshot: the game's own gameplay upscale, for the item wheel)
    framegen_hud_before(cmd, picture, picture_level);   // (the game's own network drew the gameplay upscale)
    if (dump_on() && dump_ssbo_wanted(p.key)) dump_ssbos(cmd, p, "after");
    if (gprof_on()) {
        char label[128];   // (G2: the grid, and "mma" for the game's matrix networks, native or software multiplies)
        snprintf(label, sizeof label, "dispatch %s prog %016llx%s grid %ux%ux%u", indirect ? "indirect" : "", (unsigned long long)p.key,
                 !p.stages.empty() && p.stages[0].info.uses_cooperative_matrix ? " mma" : "", indirect ? 0u : x, indirect ? 0u : y, indirect ? 0u : z);
        gprof_end(cmd, label);
    }
    st_dispatches++;
    if (pass_trace_on())
        trace_line("dispatch %ux%ux%u%s prog %016llx", x, y, z, indirect ? " indirect" : "", (unsigned long long)p.key);
    if (probe_on())
        for (size_t i = 0; i < dump_written().size(); i++) probe_image(cmd, dump_written()[i].tex, dump_written()[i].level, 0, p.key, 'd');
    if (dump_on())
        for (size_t i = 0; i < dump_written().size(); i++) {
            char what[64];
            snprintf(what, sizeof what, "dispatch %ux%ux%u prog %016llx img%zu", x, y, z, (unsigned long long)p.key, i);
            dump_image(cmd, dump_written()[i].tex, dump_written()[i].level, 0, what);
        }
}

// (hookshot) In the parallel replay's dry pass, in stream order on the replay thread: the images this draw or dispatch
// writes marked as the GPU's (texture_written) before any worker runs. Workers record out of order, and that mark was
// made as each reached its draw: one ahead of it sampled a render target or compute output as never written and
// uploaded stale guest memory over it, an upload that runs after the earlier segment's work. Targets change address
// and size as the camera moves: flashes of a lit scene gone black or white in motion (22 per 1000 frames moving, 45
// against 2 with serial replay)
thread_local bool dry_last_valid = false;   // (dry_mark_written's last targets are this dry pass's)
void dry_marks_begin() { dry_last_valid = false; }
void dry_mark_written(NvnState& s, bool compute, bool first) {
    if (!compute) {
        // (hookshot) the targets as the draw before had them are skipped, within one dry pass only: `first` is the pass's first draw OR
        // dispatch, so when a command buffer began with a dispatch its first draw was compared with the last draw of the dry pass
        // before, a frame ago. With the camera still the game's targets come back at the same addresses, the state matched, and
        // a target made again since (its texture retired at the present between) went unmarked: the later segment uploaded
        // stale memory over it and the whole scene was black for a frame. dry_marks_begin() forgets the last targets at each
        // dry pass
        static thread_local decltype(s.rt) last;
        if (!first && dry_last_valid && !std::memcmp(&last, &s.rt, sizeof last)) return;
        dry_last_valid = true;
        std::memcpy((void*)&last, (const void*)&s.rt, sizeof last);
        const Targets at = targets(s);
        ParLock lk;   // (RX_VK_SEG_EARLY: segments recorded meanwhile read these marks; nothing when none is)
        for (int i = 0; i < at.n; i++)
            if (at.color[i]) texture_written(at.color[i], "the dry pass (a draw's target)");
        if (at.depth) texture_written(at.depth, "the dry pass (a draw's depth)");
        return;
    }
    const uint8_t types[8]{};
    const Program* p = program_for(s, ProgramKind::Compute, types);
    if (!p || !p->ready.load(std::memory_order_acquire) || p->failed) return;
    for (const Stage& st : p->stages)
        for (const auto& d : st.info.image_descriptors) {
            if (!d.is_written) continue;
            const VkImageViewType vt = view_type(d.type);
            for (uint32_t k = 0; k < d.count; k++) {
                const uint32_t handle = handle_of(s, st, d, k);
                Texture* tex = nullptr;
                uint32_t level = 0;
                if (handle && texture_view_of(handle, vt, ViewUse::Storage, image_format(d.format), &tex, &level) && tex)
                    texture_written(tex, "the dry pass (a dispatch's storage image)");
            }
        }
}

void draw_stats(int64_t* draws, int64_t* skipped, int64_t* dispatches, int64_t* pipes) {
    *draws = st_draws, *skipped = st_skipped, *dispatches = st_dispatches, *pipes = st_pipelines.load();
}
void draw_stats2(int64_t* passes, int64_t* mesh, double* pipeline_ms) {
    *passes = st_passes, *mesh = st_mesh, *pipeline_ms = st_pipeline_us.load() / 1000.0;
}
int64_t draw_pipe_fresh_skips() { return st_pipe_fresh_skips.load(std::memory_order_relaxed); }
void draw_stats3(int64_t* pending_skips, int64_t* waits, double* wait_ms) {
    *pending_skips = st_pending_skips, *waits = st_waits, *wait_ms = st_wait_ms;
}
double draw_shader_wait_ms() { return st_shader_wait_ms; }
bool draw_stats4(int64_t* links, double* link_ms, int64_t* lib_hits, int64_t* lib_cold, int64_t* lib_built, double* lib_build_ms) {
    *links = st_links.load(), *link_ms = st_link_us.load() / 1000.0, *lib_hits = st_lib_hits.load(), *lib_cold = st_lib_cold.load();
    *lib_built = st_lib_built.load(), *lib_build_ms = st_lib_build_us.load() / 1000.0;
    return gpl_wanted();
}
std::mutex rec_mu;
std::map<uint64_t, std::vector<std::string>> draw_records;   // texture frame -> its recorded draws
const std::vector<uint64_t>& record_progs() {
    static const std::vector<uint64_t> v = [] {
        std::vector<uint64_t> r;
        if (const char* e = getenv("RX_VK_FLASH_RECORD"))
            for (const char* q = e; *q;) {
                char* end;
                const uint64_t k = strtoull(q, &end, 16);
                if (end == q) break;
                r.push_back(k);
                q = *end ? end + 1 : end;
            }
        return r;
    }();
    return v;
}
void record_draw(const NvnState& s, const Program& p, const DrawArgs& d, const Targets& at) {
    char line[1024];
    int n = snprintf(line, sizeof line, "prog %016llx count %d first %d inst %d bv %d bi %d idx %d ibuf %#llx ind %#llx/%#llx",
                     (unsigned long long)p.key, d.count, d.first, d.instances, d.base_vertex, d.base_instance,
                     (int)d.indexed, (unsigned long long)d.ibuf, (unsigned long long)d.indirect,
                     (unsigned long long)d.count_addr);
    n += snprintf(line + n, sizeof line - n, " target %ux%u n%d samples %d c0 %d depth %d vp %g %g %g %g scissor %d %d %d %d", at.width, at.height, at.n,
                  (int)at.samples, (int)at.color_format[0], (int)at.depth_format, s.viewport[0][0], s.viewport[0][1], s.viewport[0][2], s.viewport[0][3],
                  s.scissor[0], s.scissor[1], s.scissor[2], s.scissor[3]);
    n += snprintf(line + n, sizeof line - n, "%s", program_textures(s, p).c_str());
    if (d.indirect) {   // (what the GPU will take the draw's size from)
        uint32_t w[5]{};
        if (host->gpu_read(d.indirect, w, sizeof w))
            n += snprintf(line + n, sizeof line - n, " args %u %u %u %d %u", w[0], w[1], w[2], (int)w[3], w[4]);
    }
    for (const Stage& st : p.stages) {   // (each stage's uniform buffers: address, size, a hash of the first 512 bytes)
        const int ns = std::clamp(st.nvn_stage, 0, kNvnStages - 1);
        for (int i = 0; i < 16 && n < 900; i++) {
            const nvn_p_bind_buf& u = s.ubo[ns][i];
            if (!u.addr || !(i + 3 < Shader::Info::MAX_CBUFS && st.info.constant_buffer_used_sizes[i + 3])) continue;
            uint8_t b[512];
            const uint64_t k = std::min<uint64_t>(sizeof b, u.size ? u.size : sizeof b);
            uint64_t h = 1469598103934665603ull;
            if (host->gpu_read(u.addr, b, k))
                for (uint64_t j = 0; j < k; j++) h = (h ^ b[j]) * 1099511628211ull;
            n += snprintf(line + n, sizeof line - n, " s%d.c%d %#llx+%#llx h%016llx", ns, i + 3, (unsigned long long)u.addr,
                          (unsigned long long)u.size, (unsigned long long)h);
        }
    }
    {   // (hookshot) RX_VK_RECORD_WORDS=<cbuf>:<word>,...: those uniform words, as the draw's stages have them
        // bound (c15:452 is the light count the hung RTX 50 draws' only loop is clamped to; c6:452 the same in the 3070's): a
        // loop that ran to its 8192-turn guard would need such a word in the thousands
        static const std::vector<std::pair<int, int>> words = [] {
            std::vector<std::pair<int, int>> v;
            if (const char* e = getenv("RX_VK_RECORD_WORDS"))
                for (const char* q = e; *q;) {
                    int c = 0, w = 0;
                    if (sscanf(q, "%d:%d", &c, &w) == 2 && c >= 3 && c < 19 && w >= 0) v.push_back({c, w});
                    while (*q && *q != ',') q++;
                    if (*q) q++;
                }
            return v;
        }();
        for (const Stage& st : p.stages) {
            const int ns = std::clamp(st.nvn_stage, 0, kNvnStages - 1);
            for (const auto& [c, w] : words) {
                const nvn_p_bind_buf& u = s.ubo[ns][c - 3];
                uint32_t v = 0;
                if (n < 960 && u.addr && (!u.size || 4ull * w + 4 <= u.size) && host->gpu_read(u.addr + 4ull * w, &v, 4))
                    n += snprintf(line + n, sizeof line - n, " s%d.c%d[%d]=%u", ns, c, w, v);
            }
        }
    }
    {
        static std::atomic<int> told{0};
        if (told++ < 300) logf("vk: record: %s", line);
    }
    // (RX_VK_RECORD_FLOATS=<words>: the fragment stage's first uniform buffer as floats, each draw, one line: which constant does what)
    static const int words = getenv("RX_VK_RECORD_FLOATS") ? atoi(getenv("RX_VK_RECORD_FLOATS")) : 0;
    if (words > 0) {
        const nvn_p_bind_buf& u = s.ubo[1][0];
        std::vector<float> f((size_t)std::min(words, 128));
        if (u.addr && host->gpu_read(u.addr, f.data(), f.size() * 4)) {
            std::string o = "vk: floats " + std::to_string(texture_frame_no());
            char b[24];
            for (const float v : f) snprintf(b, sizeof b, " %.4g", v), o += b;
            logf("%s", o.c_str());
        }
    }
    std::lock_guard lk(rec_mu);
    const uint64_t f = texture_frame_no();
    draw_records[f].push_back(line);
    while (draw_records.size() > 6) draw_records.erase(draw_records.begin());
}
void pass_traces_log(uint64_t lo, uint64_t hi) {
    std::lock_guard lk(trace_mu);
    for (uint64_t f = lo; f <= hi; f++) {
        auto it = pass_traces.find(f);
        if (it == pass_traces.end()) continue;
        std::vector<std::pair<uint64_t, std::string>> v = it->second;
        std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        logf("vk: passes of frame %llu (%zu):", (unsigned long long)f, v.size());
        for (const auto& [key, l] : v) logf("vk:   [%llu] %s", (unsigned long long)(key >> 24), l.c_str());
    }
}
void draw_records_log(uint64_t lo, uint64_t hi) {
    std::lock_guard lk(rec_mu);
    for (uint64_t f = lo; f <= hi; f++) {
        auto it = draw_records.find(f);
        if (it == draw_records.end()) continue;
        logf("vk: draws of frame %llu (%zu):", (unsigned long long)f, it->second.size());
        for (const std::string& l : it->second) logf("vk:   %s", l.c_str());
    }
}
int64_t draw_desc_reused() {
    return st_desc_reused;
}
int64_t draw_slotgen_saves() {   // (Descriptor reuse: of those, what the global registrations' generation check would have rebuilt)
    return st_desc_slotgen_saves;
}

// (Hookshot) the process's private commit now, in MB (0 where it cannot be read): the cache save logs it around its fetch, to show
// what the fetch cost in memory (the vector: its whole size; the mapped file: none)
uint64_t pcache_private_mb() {
#ifdef _WIN32
    struct Counters {
        DWORD cb, PageFaultCount;
        SIZE_T PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage, QuotaPeakNonPagedPoolUsage,
            QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage, PrivateUsage;
    };
    using MemInfo = BOOL(WINAPI*)(HANDLE, Counters*, DWORD);
    static const MemInfo info = (MemInfo)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32GetProcessMemoryInfo");
    Counters c{};
    c.cb = sizeof c;
    return info && info(GetCurrentProcess(), &c, sizeof c) ? (uint64_t)(c.PrivateUsage >> 20) : 0;
#else
    return 0;
#endif
}
void pcache_log_save(const char* how, size_t n, uint64_t private_before_mb, uint64_t private_during_mb, int64_t ms) {
    logf("vk: pipeline cache saved (%s): %llu MB in %lld ms; process private commit %llu MB before the fetch, %llu MB with its data in hand",
         how, (unsigned long long)(n >> 20), (long long)ms, (unsigned long long)private_before_mb, (unsigned long long)private_during_mb);
}

// (Hookshot, RX_VK_PCACHE_MAP, default on) The cache's data fetched straight into the file's own pages: the driver writes into a
// mapped view of the temporary file, so the 1.2 GB never exists as private memory (the vector it replaces was 1.2 GB of commit and
// of working set until the write ended; mapped file pages are the system's to write out and to drop). The driver's data is still
// made whole in one call: Vulkan has no way to fetch it in pieces. False: nothing written (the caller then uses a vector).
bool pcache_save_mapped(const std::string& tmp, size_t n) {
#ifdef _WIN32
    static const bool on = !getenv("RX_VK_PCACHE_MAP") || atoi(getenv("RX_VK_PCACHE_MAP")) != 0;
    if (!on || !n) return false;
    HANDLE f = CreateFileA(tmp.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    bool ok = false;
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t before = pcache_private_mb();
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READWRITE, (DWORD)((uint64_t)n >> 32), (DWORD)n, nullptr);
    void* view = m ? MapViewOfFile(m, FILE_MAP_WRITE, 0, 0, n) : nullptr;
    size_t got = n;
    if (view && vkGetPipelineCacheData(dev.device, pcache, &got, view) == VK_SUCCESS && got <= n) ok = true;
    if (ok)
        pcache_log_save("mapped file", got, before, pcache_private_mb(),
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
    if (view) UnmapViewOfFile(view);
    if (m) CloseHandle(m);
    if (ok && got < n) {   // (the cache shrank between the two calls: the file ends where the data does)
        LARGE_INTEGER end;
        end.QuadPart = (LONGLONG)got;
        ok = SetFilePointerEx(f, end, nullptr, FILE_BEGIN) && SetEndOfFile(f);
    }
    CloseHandle(f);
    if (!ok) DeleteFileA(tmp.c_str());
    return ok;
#else
    (void)tmp, (void)n;
    return false;
#endif
}

void program_fingerprinted(uint64_t fp) {
    if (fingerprint_probe) fingerprint_probe->push_back(fp);
    else ledger_add(ledger_id_seen(fp));
}

void pipeline_cache_play() {
    pcache_init();   // (the play cache's generations need the pipeline cache's dir, also where no preparation or index ran)
    play_cache_begin();
    programs_play_begin();   // (the census: what is made from here on is made in play)
#ifdef __linux__
    // (hookshot) On Linux with <= 20 GB of RAM (the Steam Deck's 16 GB shared memory and 16 GB PCs) or when memory pressure is
    // already active (or RX_VK_DROP_MODULES=1), the VkShaderModules of ready programs whose pipelines were built in precompile_run
    // (~37,000 modules, ~1.5-1.8 GB of uncompressed SPIR-V in the driver's heap) have done their work once precompile_run finishes.
    // Any graphics program whose noted pipelines were built remakes its module from hookshot_programs.bin
    // (program_cache_complete_graphics) if play ever builds a new state variant; programs without a built graphics pipeline (not
    // noted by program_cache_note_built) keep their vertex and fragment modules so in-play pipeline builds do not read the disk.
    // 32-64 GB Linux desktops keep their modules by default so new pipeline variants in play never touch the disk.
    // RX_VK_KEEP_MODULES=1 (or RX_VK_DROP_MODULES=0) keeps all
    if (!env_on("HOOKSHOT_PREPARE_ONLY") && !prepare_left_workers()) {
        uint64_t avail_bytes = 0, swap_free_bytes = 0, total_bytes = 0;
        bool has_swap = true;
        uint64_t over = 0;
        if (linux_pressure::read_meminfo(&avail_bytes, &swap_free_bytes, nullptr, &total_bytes, &has_swap)) {
            over = linux_pressure::shortfall(avail_bytes, swap_free_bytes,
                                             linux_pressure::avail_reserve(getenv("RX_VK_AVAIL_RESERVE_MB")),
                                             linux_pressure::commit_reserve(getenv("RX_VK_COMMIT_RESERVE_MB")),
                                             has_swap);
        }
        size_t dropped_modules = 0;
        if (linux_pressure::should_drop_modules(total_bytes, over, getenv("RX_VK_KEEP_MODULES"), getenv("RX_VK_DROP_MODULES"))) {
            dropped_modules = program_cache_drop_modules();
            if (dropped_modules)
                logf("vk: programs: %zu shader modules let go after the precompile (RX_VK_KEEP_MODULES=1 keeps them)",
                     dropped_modules);
        }
#if defined(__GLIBC__)
        if (dropped_modules) malloc_trim(0);
#endif
    }
#endif
}

// ---- (hookshot) What the programs' loader asks of the pipelines (vk_pipelines.h) ----
bool precompile_covers_notes() {
    uint64_t rec[3]{};
    FILE* f = fopen(precompile_built_path().c_str(), "rb");
    if (!f) return false;
    const size_t n = fread(rec, sizeof(*rec), 3, f);
    fclose(f);
    if (n < 2 || rec[0] != program_cache_stamp()) return false;
    if (n == 3 && rec[2] == ComputePlan::every) return true;
    // Play learns more keys as it dispatches them. Its usual start needn't repeat the warm
    // each time that happens. A Builder must cover its current list, including a Most -> All upgrade.
    if (env_on("HOOKSHOT_PREPARE_ONLY")) return n == 3 && rec[2] == compute_warm_plan().coverage();
    return n == 2 || !compute_warm_plan().all;
}
std::string pipe_notes_file() { return pipe_notes_path(); }
std::vector<NoteRef> pipe_notes_refs(bool* whole) {
    std::vector<NoteRef> refs;
    if (whole) *whole = false;
    FILE* f = fopen(pipe_notes_path().c_str(), "rb");
    if (!f) return refs;
    uint64_t head[2] = {0, 0};
    if (fread(head, 8, 2, f) == 2 && head[0] == program_cache_stamp() && head[1] == sizeof(PipeNote)) {
        // Read each complete record sequentially. Only its reference stays in memory.
        std::unordered_set<uint64_t> seen;
        PipeNote probe{};
        for (int64_t at = 16; fread(&probe, sizeof probe, 1, f) == 1; at += sizeof(PipeNote)) {
            NoteRef r{};
            std::memcpy(r.vattr, probe.s.vattr, sizeof r.vattr);
            if (!targets_ok(probe.n, probe.samples, probe.topo) ||
                !formats_ok(probe.n, probe.color_format, probe.has_depth, probe.depth_format))
                continue;
            r.key = probe.key, r.prog = probe.prog, r.variant = probe.variant, r.at = at;
            note_color_types(probe, r.color_types);
            if (seen.insert(r.key).second) refs.push_back(r);
        }
    }
    if (whole) *whole = feof(f) && !ferror(f);   // (it ended at the file's end: not at a read or a seek that failed)
    fclose(f);
    return refs;
}
bool pipe_note_at(FILE* f, int64_t at, PipeNote* out) {
    return f && _fseeki64(f, at, SEEK_SET) == 0 && fread((void*)out, sizeof(PipeNote), 1, f) == 1 && note_sane(*out);
}
VkPipeline pipeline_from_caches(const Program& p, const PipeNote& n, bool* cold) {
    ask_start_first = true;
    *cold = false;
    const VkPipeline pl = build_graphics(p, n.s, note_targets(n), (VkPrimitiveTopology)n.topo, -1, cold);
    ask_start_first = false;
    return pl;
}
VkPipeline compute_pipeline_from_caches(const Program& p) {
    ask_start_first = true;
    const VkPipeline pl = compute_pipeline_cached(p, true);
    ask_start_first = false;
    return pl;
}
VkPipeline compute_pipeline_by_module_id(const Program& p, const uint8_t* id, uint32_t size) {
    if (!dev.vk13.pipelineCreationCacheControl || !size) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo ci = compute_pipeline_info(p, true);
    // (VK_EXT_shader_module_identifier: no module, its identifier in the stage; only ever without compiling, which is all
    // the driver can do with an identifier: it answers from its cache or says a compile would be needed)
    VkPipelineShaderStageModuleIdentifierCreateInfoEXT by{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO_EXT};
    by.identifierSize = size, by.pIdentifier = id;
    by.pNext = ci.stage.pNext, ci.stage.pNext = &by, ci.stage.module = VK_NULL_HANDLE;
    ci.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    VkPipeline pl = VK_NULL_HANDLE;
    ask_start_first = true;
    const bool made = cached_compute(&ci, &pl) == VK_SUCCESS;
    ask_start_first = false;
    return made ? pl : VK_NULL_HANDLE;
}
void pipeline_defer(uint64_t key) {
    std::lock_guard lk(kept_ready_mu);
    kept_ready.try_emplace(key, std::make_pair(VkPipeline(VK_NULL_HANDLE), (const Program*)nullptr));
    kept_ready_n.store(kept_ready.size(), std::memory_order_release);
}
bool pipeline_keep_ready(uint64_t key, VkPipeline pl, const Program* made_from) {
    if (!pl) return false;
    {
        std::lock_guard lk(kept_ready_mu);
        auto [it, inserted] = kept_ready.try_emplace(key, std::make_pair(pl, made_from));
        if (inserted || !it->second.first) {   // (a deferred note, pipeline_defer, takes the pipeline made for it)
            it->second = {pl, made_from};
            kept_ready_n.store(kept_ready.size(), std::memory_order_release);
            return true;
        }
    }
    vkDestroyPipeline(dev.device, pl, nullptr);
    return false;
}
std::string load_fallback_path() {
    const std::string notes = pipe_notes_path();
    return notes.substr(0, notes.find_last_of("\\/") + 1) + "hookshot_load_start.txt";
}
void program_layout(Program& p) { ensure_layout(p); }
std::vector<std::pair<uint64_t, uint64_t>> compute_variants_learned() { return compute_variants_read(); }
void pipeline_cache_close() {
    try {   // (with the fetch lock on, bounded: a build stuck in the driver costs the play file, not the close)
        if (play_cache.load()) play_cache_save(kFetchQueueMs);
    } catch (const std::exception& e) {
        logf("vk: pipeline cache save failed: %s", e.what());
    }
}
/// What play made since the last save, into a play file now (vk_fsr4.cpp: its tuned layers, minutes to build the first time,
/// kept even when the game does not close normally)
void pipeline_cache_save_play() {
    try {
        if (play_cache.load()) play_cache_save();
    } catch (const std::exception& e) {
        logf("vk: pipeline cache save failed: %s", e.what());
    }
}

// (hookshot) a save put off for builds under way, owed to the serial builds' thread (pipeline_cache_between_builds), and
// that thread's save there
std::atomic<bool> pcache_save_owed{false};
thread_local bool pcache_save_between = false;
/// (hookshot) Intel's compute builds are one at a time (jobs_push_serial), and its compiler took 30 s to 6.5 minutes on some of
/// the game's compute programs on a UHD Graphics: one build or another was under way the whole of a 15-minute run, the queued
/// fetch's 10 s never found a moment without one, and every save in play was put off. Nothing play built was kept, and the next
/// start built it all again (each start read the same 182 KB). Between two of the serial thread's builds, the save runs there:
/// no compute build is under way, and the graphics builds it queues behind take milliseconds
void pipeline_cache_between_builds() {
    if (!pcache_save_owed.exchange(false)) return;
    pcache_save_between = true;
    struct Done { ~Done() { pcache_save_between = false; } } done;
    pipeline_cache_save(false);
}
// A save in play, by RX_VK_PCACHE_PLAY_SAVE: 2 (the default) every time it is asked, as before; 1 only with room for it and
// not more often than every kPlaySaveMinS, on a thread in Windows' background mode; 0 never (a forced save, at exit, the
// prepare or Restart now, writes what play built). Measured where the heap's cold pages are taken out of the working set
// (HL_MEM_COLD, off by default here): a 1.27 GB cache saved 7 times in play (RTX 5070, 31 GB of RAM, 28 GB of private
// commit), and after two of the saves the game stopped for 3.5 to 14.2 s while 200 to 270 MB of those heap pages came back
// from the page file, pushed out of RAM by the saves' new file pages. Room: the physical memory available is at least twice
// the cache plus kPlaySaveSpareMB. 0 save, 1 put off for room, 2 put off for the interval, 3 never in play
constexpr uint64_t kPlaySaveSpareMB = 4096;
constexpr int64_t kPlaySaveMinS = 300;
int pcache_play_save_gate(int mode, uint64_t avail_bytes, uint64_t cache_bytes, int64_t since_last_s) {
    if (mode >= 2) return 0;
    if (mode <= 0) return 3;
    if (avail_bytes < 2 * cache_bytes + (kPlaySaveSpareMB << 20)) return 1;
    if (since_last_s >= 0 && since_last_s < kPlaySaveMinS) return 2;
    return 0;
}
std::atomic<int64_t> pcache_saved_at_s{-1};   // (steady clock seconds of the last save in play; -1: none yet)
int pcache_play_save_mode() {
    static const int mode = [] {
        const char* e = getenv("RX_VK_PCACHE_PLAY_SAVE");
        return e && *e ? atoi(e) : 2;
    }();
    return mode;
}
uint64_t pcache_available_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? (uint64_t)m.ullAvailPhys : UINT64_MAX;
#else
    return UINT64_MAX;
#endif
}
uint64_t pcache_file_bytes() {   // (the cache's size on disk: what the next save writes again, about)
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA fa{};
    if (GetFileAttributesExA(pcache_path.c_str(), GetFileExInfoStandard, &fa)) return (uint64_t)fa.nFileSizeHigh << 32 | fa.nFileSizeLow;
#endif
    return 0;
}
void pipeline_cache_save(bool force) {
    using Outcome = precompile_save::Outcome;
    pcache_save_outcome = {};
    // (hookshot) The file not read (pcache_no_save): the cache's own part is skipped, never written, but play's
    // generations are still taken away and freed at each save (play_cache_save writes none then), as they are written and freed
    // where the file is read. Returning here, what play made would pile up in one generation, in memory, until the close
    const bool between = pcache_save_between;
    if (!pcache) pcache_save_outcome.what = Outcome::kNoCache;
    else if (pcache_no_save) pcache_save_outcome.what = Outcome::kOff;
    // (RX_VK_PCACHE_SAVE_ALWAYS=1, for measurements: every save asked in play goes ahead to the gate even with fewer than 32
    // pipelines built since the last one, as play with many new pipelines would have it)
    static const bool always = env_on("RX_VK_PCACHE_SAVE_ALWAYS");
    if (!pcache || (!force && !between && !always && pcache_built_since_save < 32)) return;
    if (!force && !between && pcache_play_save_mode() < 2) {   // (a save asked in play: RX_VK_PCACHE_PLAY_SAVE's gate)
        const int64_t now_s =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const int64_t last = pcache_saved_at_s.load();
        const uint64_t avail = pcache_available_bytes(), size = pcache_file_bytes();
        const int gate = pcache_play_save_gate(pcache_play_save_mode(), avail, size, last < 0 ? -1 : now_s - last);
        if (gate) {
            static int64_t told;
            if (gate != 2 && (++told <= 4 || told % 32 == 0))
                logf("vk: pipeline cache save put off in play (%s; %lld pipelines owed): written at exit or later",
                     gate == 3 ? "RX_VK_PCACHE_PLAY_SAVE=0" : "too little free memory for it", (long long)pcache_built_since_save.load());
            if (gate == 1 && (told <= 4 || told % 32 == 0))
                logf("vk:   (%llu MB available, %llu MB of cache: a save in play wants twice the cache and %llu MB more)",
                     (unsigned long long)(avail >> 20), (unsigned long long)(size >> 20), (unsigned long long)kPlaySaveSpareMB);
            return;
        }
        pcache_saved_at_s = now_s;
    }
    const int64_t built = pcache_built_since_save.exchange(0);
    // (on a thread of its own unless forced, at exit: ~100 MB fetched and written took the replay thread 100+ ms at
    // each scene change; the cache is internally synchronized, pipelines go on being built meanwhile)
    static std::atomic<bool> busy{false};
    if (busy.exchange(true)) {
        if (between) {   // (a background save under way: owed again, for after the next build)
            pcache_built_since_save += built;
            pcache_save_owed = true;
            return;
        }
        if (!force) return;
        while (busy.load()) Sleep(1);   // (the one under way, then this one)
        busy = true;
    }
    auto save = [force, between, built] {
        struct Done { ~Done() { busy = false; } } done;
#ifdef _WIN32
        // (RX_VK_PCACHE_PLAY_SAVE=1: a save in play runs in Windows' background mode, on its own thread: its disk writes at
        // low I/O priority and the pages it touches, the file's among them, at low memory priority, so that RAM gives them up
        // before the game's)
        if (!force && !between && pcache_play_save_mode() == 1) SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
#endif
        // (Intel: no build under way while the caches' data is fetched, PcacheFetch, the lock held around each fetch alone. A
        // forced save waits for the builds; the background one polls, and is put off when builds ran throughout; after
        // kFetchPutOffs of those in a row it queues for kFetchQueueMs, so a stretch of back-to-back builds cannot put it off for good.
        // Put off, it is owed to the serial builds' thread, which queues for it between two of its builds)
        static std::atomic<int> put_off{0};
        const bool queue = force || between || put_off.load() >= kFetchPutOffs;
        const int wait_ms = force ? -1 : queue ? kFetchQueueMs : kFetchPollMs;
        bool had = true;
        auto fetch = [&](size_t* n, void* data) {
            PcacheFetch lock(wait_ms, queue);
            had = lock.held();
            return had && vkGetPipelineCacheData(dev.device, pcache, n, data) == VK_SUCCESS;
        };
        if (!pcache_no_save) try {
            size_t n = 0;
            // (the size last read or written: nothing new, not fetched nor written again; see pcache_saved_size)
            // (in play only once it has grown by a good deal: what play makes is in the play files, and the little pcache still
            // grows by there (a stage library the driver's own cache answered, copied into it) had it written whole at every
            // save all the same, 2.4 GB 25 times in a 10-minute first visit for 9 MB more. Past the threshold it is written,
            // so a folder prepared before the warm pass keeps its libraries through a crash. The window's close does not write it
            // (pipeline_cache_close saves the generation only); a forced save does: Restart now, the prepare. RX_VK_PLAY_SAVE_START_MB=<n>: the growth it is written at in play (64; 0: any, as before))
            static const size_t play_grown = (size_t)(getenv("RX_VK_PLAY_SAVE_START_MB") ? std::max<long long>(0, atoll(getenv("RX_VK_PLAY_SAVE_START_MB"))) : 64) << 20;
            const auto put_off_now = [&] {
                const int times = ++put_off;
                if (times <= 4 || times % 16 == 0)
                    logf("vk: pipeline cache save put off (%d in a row): pipelines were being built throughout%s", times,
                         serial_builds_on() ? "; owed to the serial builds' thread, between two of its builds" : "");
                pcache_built_since_save += built;   // (still to save: the next request writes them, not only after 32 more)
                if (serial_builds_on()) pcache_save_owed = true;
            };
            const bool sized = fetch(&n, nullptr);
            bool write = sized && n && n != pcache_saved_size;
            if (!had) return put_off_now();
            pcache_save_outcome.what = !sized || !n ? Outcome::kFetchFailed : write ? Outcome::kNotRun : Outcome::kSame;
            pcache_save_outcome.bytes = n;
            if (write && !force && play_grown && play_cache_in_use() && n < pcache_saved_size + play_grown) {
                write = false;
                static std::atomic<int> told{0};
                if (told++ < 4)
                    logf("vk: pipeline cache %zu KB larger in play: written once it is %zu MB larger, or by a forced save",
                         (n - std::min<size_t>(n, pcache_saved_size)) >> 10, play_grown >> 20);
            }
            if (write) {
                auto data = PipelineCacheFile::writable(pcache_path, n);
                const bool fetched = !data.empty() && fetch(&n, data.data());
                if (fetched && data.publish(n)) {
                    pcache_saved_size = n;
                    pcache_save_outcome.what = Outcome::kWritten, pcache_save_outcome.bytes = n;
                    logf("vk: pipeline cache saved (%zu KB)%s", n >> 10, between ? ", between two serial builds" : "");
                } else if (!had) {
                    return put_off_now();   // (the size had, the data not: put off as well, not lost)
                } else {
                    pcache_save_outcome.what = data.empty() || fetched ? Outcome::kWriteFailed : Outcome::kFetchFailed;
                    pcache_save_outcome.detail = pcache_path;
                }
            }
            put_off = 0;
        } catch (const std::exception& e) {
            pcache_save_outcome.what = Outcome::kError, pcache_save_outcome.detail = e.what();
            logf("vk: pipeline cache save failed: %s", e.what());
        }
        try {
            play_cache_save(wait_ms, queue);   // (what play made since the last save, alone)
        } catch (const std::exception& e) {
            logf("vk: pipeline cache save failed: %s", e.what());
        }
    };
    if (force || between) save();
    else {
        try {
            std::thread([save] { helper_thread_start(), save(); }).detach();   // (do_present asks, inside its ReplayPin)
        } catch (const std::exception& e) {
            busy = false;
            logf("vk: could not start pipeline cache save: %s", e.what());
        }
    }
}

/// (vk_internal.h: a compile under way, for the preparation's watchdog)
bool prepare_compiling() { return prepare_compiling_seen(); }
/// (vk_internal.h: Mesa's Intel driver, whose compiler takes minutes over the game's largest programs while it works)
bool prepare_long_compiles() { return dev.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA; }

} // namespace mkwvk

namespace mkwvk {
/// (hookshot) n bytes copied into the recording's stream, as a buffer binding (vk_memory.cpp gpu_buffer_bind: a read whose pool is
/// still being imported in the background); empty when there is no recording or no room
BufferRef stream_copy(const void* src, uint64_t n) {
    if (!cur || !n) return {};
    const uint64_t align = std::max<uint64_t>({256, dev.props.limits.minUniformBufferOffsetAlignment,
                                               dev.props.limits.minStorageBufferOffsetAlignment});
    const uint64_t at = stream_alloc(n, align);
    if (at == UINT64_MAX) return {};
    std::memcpy(cur->mapped + at, src, n);
    return {cur->stream, at, n, 0};
}

UniformUpload uniform_upload(uint64_t size) {
    ParLock lk;
    if (!cur || !cur->stream || !cur->mapped || !size) return {};
    const uint64_t align = std::max<uint64_t>(256, dev.props.limits.minUniformBufferOffsetAlignment);
    const uint64_t at = stream_alloc(size, align);
    if (at == UINT64_MAX) return {};
    return {{cur->stream, at, size, 0}, cur->mapped + at, cur->stream_alloc};
}

} // namespace mkwvk
