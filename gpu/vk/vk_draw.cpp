// Draws and dispatches: a program's pipeline layout (its stages' descriptors in Eden's order, as the SPIR-V back end
// numbered them, one push-descriptor set when they fit), graphics pipelines (whatever Vulkan 1.3 and
// VK_EXT_extended_dynamic_state3 take as dynamic state is dynamic; the rest is the key), each draw's descriptors
// (driver banks and c1 from the recording's stream, user uniform and storage buffers in the imported pools, textures
// and storage images as views of vk_texture.cpp's images, texel buffers), passes kept open while the targets stay,
// and the draws themselves.
//
// Pipelines persist in the driver's own disk cache. The full start (RX_VK_LEAN_PIPELINES=0) also keeps a VkPipelineCache
// next to the executable (mkw_vk_pipelines.bin); the lean start, every PC's since 0.7.6, keeps none (pipelines_lean, below).
#include <algorithm>
#include <filesystem>
#include <iterator>
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
#include <condition_variable>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <vector>
#include <thread>
#include <setjmp.h>
#include <signal.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"

#include "vk_paths.h"
#include "vk_draw.h"
#include "vk_framegen.h"   // (relink: the gameplay upscale's output, for the item wheel's backdrop)
#include "dlss_actor.h"
#include "vk_dlss.h"
#include "vk_dump.h"
#include "vk_gprof.h"
#include "vk_jobs.h"
#include "vk_mesh_split.h"
#include "vk_shader.h"
#include "vk_texture.h"
#include "movie_upscale_policy.h"
#include "skip_code.h"
#include "direct_ub_snapshot.h"

#include "shaders/clear_f_frag_spv.h"
#include "shaders/clear_i_frag_spv.h"
#include "shaders/clear_u_frag_spv.h"
#include "shaders/clear_vert_spv.h"
#include "shaders/taau_upscale_comp_spv.h"
#include "shaders/movie_upscale_comp_spv.h"
#include "shader_recompiler/backend/spirv/mma_layout.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

namespace mkwvk {
// (EvilOOT, RX_VK_DRAW_STATS: the pointer table's regions, vk_bank.cpp)
struct PointerEntry { uint64_t addr, size; };
const PointerEntry* pointer_table(const NvnState& s, int stage);
uint32_t stage_cbuf_word(const NvnState& s, const Stage& st, uint32_t index, uint32_t offset);   // (vk_shader.cpp)
uint64_t sampler_registrations_generation();   // (vk_bank.cpp: sampler_for's cache)
namespace {
// ---- (relink) this file's per-thread objects that need a constructor or a destructor ----
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

// (relink) set by draw() while it builds the descriptors of a draw into a single colour target with no depth (the HUD and the
// other 2D passes): the small buffers it binds are noted for the submit-time capture (memory_note_ui_buffer)
thread_local bool ui_draw_now = false;
// (relink, 10/11 08:30) and set by dispatch() while it builds a compute dispatch's descriptors: its small buffers, uniform and
// storage, are captured at the submit too. The exposure moved in steps while the camera turned (the picture's brightness held two
// or three frames, then jumped ~8 %: 9 to 11 frames in a thousand standing 1.5 levels off their neighbours' middle, hs51 / hs53
// / hs58): a compute pass reads a small block the game rewrites in place every frame, and the replay, one to three frames behind
// the game, found a later frame's values in it, the same ones for the frames it replayed in a row. Captured, 1.8 a thousand
// (hs59_dispatch), as with the game held one frame ahead (HOS_FRAMES_AHEAD=1, which costs 59.0 -> 47.5 frames a second in the
// village, hs540 / hs541); 36 pages copied at a submit, 0.03 ms. A 2D pass's or every depth-less draw's uniform buffers do
// nothing for it (hs52_uiubo 10.2, hs59_nodepth 7.5). RX_VK_CAPTURE_COMPUTE=0: off
thread_local bool compute_now = false;

// ---- (relink) RX_VK_PROF_FINE=1: where a draw's time goes, finer than the frame line's phases, and what each draw does ----
// Off, each site is a test of one constant. On, a timed site reads the clock twice (~15 ns; the frame line's phases grow by the same, so
// compare a run with it on to another with it on) and the sums are logged with the frame statistics, "a frame (fine)": the times in ms a
// frame, the counts a draw. FT_*: total preparation, descriptor and command phases, plus the sections of the descriptor build
// (banks, c1, user uniform buffers, storage buffers, textures) and of the commands (hazards and the pass, pipeline bind,
// descriptor push, push constants, dynamic state, vertex and index binds, the draw). A total minus its named sections exposes
// loop, lookup and diagnostic overhead that was previously outside every fine timer.
enum FineT { FT_PREP, FT_TARGETS, FT_PROGRAM_PICK, FT_PIPE_LOOKUP, FT_DESC_TOTAL, FT_BANK, FT_C1, FT_UBO, FT_STORAGE, FT_TEX,
             FT_CMD_TOTAL, FT_HAZARD, FT_CMD_PREPASS, FT_CMD_TRACK, FT_BINDPL, FT_BINDDESC, FT_PUSHCALL, FT_PUSHC, FT_DYN,
             FT_CMD_POSTSTATE, FT_MESH_DRAW, FT_MESH_COUNT_BIND, FT_MESH_ARGS_BIND, FT_MESH_PUSH, FT_MESH_CALL, FT_VERTEX, FT_DRAWCALL,
             FT_COUNT };
const char* const kFineT[] = {"draw preparation", "targets+types", "program pick", "pipeline lookup", "descriptor total", "banks",
                                      "c1", "user uniform buffers", "storage buffers", "textures", "command total", "hazards+pass",
                                      "command pre-pass", "sampled-target tracking", "bind pipeline", "bind descriptors",
                                      "of it the push call", "push constants", "dynamic state", "post-state setup",
                                      "mesh draw setup+call", "mesh count bind", "mesh arguments bind", "mesh first push", "mesh command",
                                      "vertex+index binds", "draw call"};
enum FineC { FC_DRAWS, FC_BUILDS, FC_DESCS, FC_WRITES, FC_BANK_CALLS, FC_BANK_GEN, FC_BANK_FILLS, FC_BANK_FILL_BYTES, FC_BANK_SAME,
             FC_BANK_NEW, FC_BANK_NEW_BYTES, FC_C1_CALLS, FC_UBO_CALLS, FC_UBO_OVERLAY, FC_UBO_STREAM, FC_UBO_REUSE, FC_PUSH_FULL,
             FC_PUSH_DELTA, FC_PUSH_NONE, FC_PUSH_WRITES, FC_PUSH_DESCS, FC_BIND_PIPE, FC_DYN_FAST, FC_DYN_FULL, FC_DYN_CALLS,
             FC_VBIND, FC_IBIND, FC_PUSHC, FC_PIPE_FAST, FC_PIPE_MAP, FC_POOL_SETS, FC_POOL_DESCS, FC_SSBO_DESCS, FC_MESH_DRAWS,
             FC_MESH_INDIRECT, FC_MESH_COUNTED, FC_MESH_FIRST, FC_COUNT };
const char* const kFineC[] = {"draws", "descriptor builds", "descriptors", "descriptor writes", "bank calls",
                                      "bank kept by its stage's generation", "bank fills", "bank bytes filled", "bank refills equal",
                                      "bank copies into the stream", "bank bytes copied", "c1 calls", "user uniform buffer calls",
                                      "uniform ranges copied with an update's overlay", "uniform ranges copied into the stream",
                                      "direct uniform snapshots reused", "full pushes", "pushes of changed writes only",
                                      "pushes of nothing (all as pushed)",
                                      "writes pushed", "descriptors pushed", "pipeline binds", "dynamic state from the draw before",
                                      "dynamic state worked out", "dynamic state commands", "vertex stream binds", "index buffer binds",
                                      "push constants", "pipeline keys from the fast pair", "pipeline keys from the thread's map",
                                      "sets allocated (over the push limit)", "descriptors in them", "storage buffer descriptors built",
                                      "mesh draws", "indirect mesh draws", "counted mesh draws", "mesh draws with a nonzero first task"};
static_assert(std::size(kFineT) == FT_COUNT && std::size(kFineC) == FC_COUNT, "every section and count is named");
const bool g_fine = getenv("RX_VK_PROF_FINE") != nullptr;
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
    char line[900];
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
        if (k > (int)sizeof line - 80) {
            logf("%s", line);
            k = snprintf(line, sizeof line, "vk: a frame (fine): a draw:");
        }
    }
    logf("%s", line);
    for (int i = 0; i < FC_COUNT; i++) last_n[i] = fine_n[i];
}

bool no_a2c() {   // (RX_VK_NO_A2C, read once: it was a getenv on every alpha-to-coverage draw, the foliage's)
    static const bool v = getenv("RX_VK_NO_A2C") != nullptr;
    return v;
}

/// 32 lanes, where the stage can be held to it (workgroup_x: a compute stage's width, which full subgroups must divide)
void pin_subgroup(VkPipelineShaderStageCreateInfo& st, uint32_t workgroup_x = 0) {
    static const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo size = [] {
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo r{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        r.requiredSubgroupSize = 32;
        return r;
    }();
    if (!(dev.sg32_stages & st.stage)) return;
    st.pNext = &size;
    if (workgroup_x && !(workgroup_x % 32) && dev.vk13.computeFullSubgroups)
        st.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
}

inline uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h * 0xff51afd7ed558ccdull;
}

// ---- environment switches, read once ----
struct Switches {
    bool no_depth = getenv("RX_VK_NODEPTH") != nullptr;
    bool no_cull = getenv("RX_VK_NOCULL") != nullptr;
    bool ignore_discard = getenv("RX_VK_IGNORE_DISCARD") != nullptr;
    bool log_draws = getenv("RX_VK_LOG_DRAWS") != nullptr;
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
        static const bool live = getenv("RX_VK_DUMP_LIVE") != nullptr;
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
    std::unordered_map<const void*, std::pair<VkBuffer, uint64_t>> c1_at;   // (c1 of a stage: uploaded once a recording)
    // Direct (non-mirrored) uniform ranges copied once for each submitted-memory
    // view and again after an inline update or another submit in this recording.
    mkwvk::direct_ub::Cache<VkBuffer> direct_ub_at;
};
// (a recording's first stream: one for each of the kRecordings slots. RX_VK_STREAM_MB=<n>: that size, 2 by default; a recording
// that needs more is given a bigger stream for the rest of it (RX_VK_STREAM_GROW_OFF=1: draws are skipped instead, and the default
// is 24 MB as before); the most a recording used is logged: "the recording streams")
bool stream_grow_off() {
    static const bool off = getenv("RX_VK_STREAM_GROW_OFF") != nullptr;
    return off;
}
const uint64_t kStream = (getenv("RX_VK_STREAM_MB") ? (uint64_t)std::max(1, atoi(getenv("RX_VK_STREAM_MB"))) : stream_grow_off() ? 24u : 2u) << 20;
std::atomic<uint64_t> stream_peak{0}, stream_extra{0}, stream_grown{0};
std::deque<RecordingRes> res;   // (a deque: a recording thread keeps its own in place while another is added)
thread_local RecordingRes* cur = nullptr;

VkDescriptorPool make_pool() {
    // (each recording's pool: 8192 sets of up to 32768 descriptors a type, for the programs with more descriptors than the push limit,
    // 24 or 48 pools at once, and the driver may keep host memory for all of that. A pool that runs out is followed by another of the same
    // size (cur->extra), so a smaller one is safe, and costs a pool made and destroyed per recording in a scene that needs more.
    // RX_VK_DESC_POOL_DIV=<n> (1..32): a n-th of both)
    static const uint32_t div = getenv("RX_VK_DESC_POOL_DIV") ? (uint32_t)std::clamp(atoi(getenv("RX_VK_DESC_POOL_DIV")), 1, 32) : 1;
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32768 / div}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384 / div},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32768 / div}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4096 / div},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2048 / div}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 2048 / div}};
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets = 8192 / div;
    ci.poolSizeCount = (uint32_t)std::size(sizes);
    ci.pPoolSizes = sizes;
    VkDescriptorPool p = VK_NULL_HANDLE;
    vkCreateDescriptorPool(dev.device, &ci, nullptr, &p);
    return p;
}

constexpr VkBufferUsageFlags kStreamUse = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
/// The recording goes on in its next stream (made when the slot has none yet, twice the last one's size up to 32 MB, or the size asked)
bool stream_next(uint64_t need) {
    RecordingRes& r = *cur;
    if (r.si + 1 < r.streams.size() && r.streams[r.si + 1].size >= need) {
        r.si++;
    } else {
        if (stream_grow_off()) return false;
        RecordingRes::Stream s{};
        s.size = std::max<uint64_t>((need + (1u << 20) - 1) & ~((1ull << 20) - 1), std::min<uint64_t>(r.size * 2, 32ull << 20));
        void* mapped = nullptr;
        if (!create_buffer(s.size, kStreamUse, true, s.buffer, s.alloc, &mapped, true) || !mapped) return false;
        s.mapped = (uint8_t*)mapped;
        stream_extra += s.size, stream_grown++;
        static int logged;
        if (logged++ < 8)
            logf("vk: a recording needed more than its %llu MB stream: one of %llu MB made (%llu MB of them in all)",
                 (unsigned long long)(r.size >> 20), (unsigned long long)(s.size >> 20), (unsigned long long)(stream_extra.load() >> 20));
        // (one made while the old ones stay: a bigger one that fits here goes before the smaller ones kept after it)
        r.streams.insert(r.streams.begin() + (ptrdiff_t)++r.si, s);
    }
    r.before += r.used;
    r.stream = r.streams[r.si].buffer, r.stream_alloc = r.streams[r.si].alloc, r.mapped = r.streams[r.si].mapped;
    r.size = r.streams[r.si].size, r.used = 0;
    return true;
}
/// Room for n bytes at align in the recording's stream, moving it to a bigger one when there is none (false: no room)
bool stream_reserve(uint64_t n, uint64_t align) {
    if (((cur->used + align - 1) & ~(align - 1)) + n <= cur->size) return true;
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
VkCompareOp compare_op(uint32_t f) {
    return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS;   // (NEVER = 1 .. ALWAYS = 8)
}
VkStencilOp stencil_op(uint32_t o) {
    return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP;   // (KEEP = 1 .. DECR_WRAP = 8)
}
VkBlendFactor blend_factor(uint32_t f) {
    switch (f) {
    case NVN_BLEND_FUNC_ZERO: return VK_BLEND_FACTOR_ZERO;
    case NVN_BLEND_FUNC_ONE: return VK_BLEND_FACTOR_ONE;
    case NVN_BLEND_FUNC_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
    case NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case NVN_BLEND_FUNC_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
    case NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case NVN_BLEND_FUNC_DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
    case NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case NVN_BLEND_FUNC_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
    case NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case NVN_BLEND_FUNC_SRC_ALPHA_SATURATE: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case NVN_BLEND_FUNC_CONSTANT_COLOR: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case NVN_BLEND_FUNC_CONSTANT_ALPHA: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    case NVN_BLEND_FUNC_SRC1_COLOR: return VK_BLEND_FACTOR_SRC1_COLOR;
    case NVN_BLEND_FUNC_SRC1_ALPHA: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    default: return VK_BLEND_FACTOR_ONE;
    }
}
VkBlendOp blend_op(uint32_t e) {
    switch (e) {
    case NVN_BLEND_EQUATION_SUB: return VK_BLEND_OP_SUBTRACT;
    case NVN_BLEND_EQUATION_REVERSE_SUB: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case NVN_BLEND_EQUATION_MIN: return VK_BLEND_OP_MIN;
    case NVN_BLEND_EQUATION_MAX: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}
/// Vulkan topology (false: one Vulkan has none of: quads, quad strips, polygons, line loops)
bool topology(int prim, VkPrimitiveTopology& t) {
    switch (prim) {
    case NVN_DRAW_PRIMITIVE_POINTS: t = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; return true;
    case NVN_DRAW_PRIMITIVE_LINES: t = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; return true;
    case NVN_DRAW_PRIMITIVE_LINE_STRIP: t = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; return true;
    case NVN_DRAW_PRIMITIVE_TRIANGLES: t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; return true;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP: t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; return true;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_FAN: t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; return true;
    case NVN_DRAW_PRIMITIVE_POLYGON: t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; return true;
    case NVN_DRAW_PRIMITIVE_LINES_ADJACENCY: t = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; return true;
    case NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY: t = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY; return true;
    case NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY: t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; return true;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY: t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY; return true;
    case NVN_DRAW_PRIMITIVE_PATCHES: {
        static int logged;
        if (logged++ < 5) logf("vk: a draw with patches");
        t = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        return true;
    }
    case NVN_DRAW_PRIMITIVE_QUADS: t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; return true;   // (indices made: draw())
    default: return false;
    }
}
int topology_class(VkPrimitiveTopology t) {
    switch (t) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return 0;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: return 1;
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST: return 3;
    default: return 2;
    }
}
/// NVN vertex attribute format -> Vulkan (gpu/render/rx_draw.cpp nvn_attribute's table)
VkFormat vertex_format(int f) {
    static const VkFormat r8[] = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT};
    static const VkFormat r16[] = {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT};
    static const VkFormat r32[] = {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT};
    static const VkFormat rg8[] = {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT};
    static const VkFormat rg16[] = {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT};
    static const VkFormat rg32[] = {VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT};
    static const VkFormat rgb8[] = {VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT};
    static const VkFormat rgb16[] = {VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16_SINT};
    static const VkFormat rgb32[] = {VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT};
    static const VkFormat rgba8[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT};
    static const VkFormat rgba16[] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT};
    static const VkFormat rgba32[] = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT};
    static const struct { int lo; const VkFormat* f; } ranges[] = {
        {0x01, r8}, {0x05, r16}, {0x0A, r32}, {0x0D, rg8}, {0x11, rg16}, {0x16, rg32},
        {0x19, rgb8}, {0x1D, rgb16}, {0x22, rgb32}, {0x25, rgba8}, {0x29, rgba16}, {0x2E, rgba32}};
    static const int counts[] = {4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3};
    for (int i = 0; i < 12; i++)
        if (f >= ranges[i].lo && f < ranges[i].lo + counts[i]) return ranges[i].f[f - ranges[i].lo];
    if (f >= 0x51 && f <= 0x68) {   // (*_UI2F / *_I2F: integers converted to float by the shader (UnsignedScaled /
        // SignedScaled attributes): the 32-bit ones were read as float bits before, and R32 has no scaled format)
        static const VkFormat us[12] = {VK_FORMAT_R8_UINT, VK_FORMAT_R16_UINT, VK_FORMAT_R32_UINT, VK_FORMAT_R8G8_UINT,
                                        VK_FORMAT_R16G16_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R8G8B8_UINT,
                                        VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R8G8B8A8_UINT,
                                        VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R32G32B32A32_UINT};
        static const VkFormat ss[12] = {VK_FORMAT_R8_SINT, VK_FORMAT_R16_SINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_SINT,
                                        VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R8G8B8_SINT,
                                        VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R8G8B8A8_SINT,
                                        VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SINT};
        return ((f - 0x51) & 1 ? ss : us)[(f - 0x51) / 2];
    }
    switch (f) {
    case NVN_FORMAT_RGB10A2: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case NVN_FORMAT_RGB10A2UI: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case NVN_FORMAT_RGB10A2SN: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    case NVN_FORMAT_RGB10A2I: return VK_FORMAT_A2B10G10R10_SINT_PACK32;
    case NVN_FORMAT_RGB10A2_UI2F: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case NVN_FORMAT_RGB10A2_I2F: return VK_FORMAT_A2B10G10R10_SINT_PACK32;
    case NVN_FORMAT_R11G11B10F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case NVN_FORMAT_BGRA8: case NVN_FORMAT_BGRX8: return VK_FORMAT_B8G8R8A8_UNORM;
    case NVN_FORMAT_RGBX8: return VK_FORMAT_R8G8B8A8_UNORM;
    default: return VK_FORMAT_UNDEFINED;
    }
}
/// (relink) Whether a colour target of this format may blend. NVN leaves the blend bits as the game set them on integer targets,
/// where the hardware ignores them; Vulkan forbids blending there (formats without COLOR_ATTACHMENT_BLEND), and with it on, OoT's
/// load-time pass into a 64^3 R16G16B16A16_UINT volume is where the device was lost (RX_VK_SKIP_PROG's 53509044401d78d2).
/// RX_VK_BLEND_INT=1: as before
bool format_blends(VkFormat f) {
    static const bool as_before = getenv("RX_VK_BLEND_INT") != nullptr;
    if (as_before || f == VK_FORMAT_UNDEFINED) return true;
    static std::atomic<uint8_t> known[256];   // (0: not asked, 1: no, 2: yes; the core formats)
    if ((uint32_t)f < 256)
        if (const uint8_t k = known[(uint32_t)f].load(std::memory_order_relaxed)) return k == 2;
    VkFormatProperties p{};
    vkGetPhysicalDeviceFormatProperties(dev.physical, f, &p);
    const bool b = (p.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
    if ((uint32_t)f < 256) known[(uint32_t)f].store(b ? 2 : 1, std::memory_order_relaxed);
    return b;
}

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
    uint32_t total = 0;
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
    p.push = dev.push_descriptor && vkCmdPushDescriptorSetKHR && total <= dev.max_push_descriptors;
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
    static const bool follow_graphics = !getenv("RX_VK_FOLLOW_GRAPHICS") || atoi(getenv("RX_VK_FOLLOW_GRAPHICS")) != 0;
    for (const Stage& st : p.stages)   // (the page table; EvilOOT: and graphics stages with RX_VK_FOLLOW_GRAPHICS)
        p.uses_push |= ((st.vk_stage & (VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) ||
                        follow_graphics) &&
                       st.info.uses_global_memory;
    for (const Stage& st : p.stages) p.uses_push |= (st.vk_stage & VK_SHADER_STAGE_TASK_BIT_EXT) != 0;   // (its first workgroup)
    if (p.fallback) p.uses_push = true;   // (its records' and payloads' addresses)
    std::atomic_thread_fence(std::memory_order_release);   // (the layout last: a thread that sees it sees the rest)
    p.layout = layout;
}

// ---- the pipeline cache, on disk ----
VkPipelineCache pcache = VK_NULL_HANDLE;
std::string pcache_path;
std::atomic<int64_t> pcache_built_since_save{0};
// (relink, 13/11) The cache is let go once the start's precompile has made its pipelines from it (pipeline_cache_slim). The
// driver keeps what a VkPipelineCache holds in its own memory, several times the file: with OoT's 865 MB file, 4.3 GB of the
// process's private bytes for as long as the cache lived (vmmap.ps1, 40 s after the start: 20.6 GB with it, 16.3 GB with an empty
// one, both without the precompile, whose 18900 pipelines cost 2.7 GB more). On a 16 GB PC the game then ran out of RAM in its
// first area: 0.9 GB available 40 s after the start, 10 to 400 MB from the second area on, Windows holding 9 GB of the process
// compressed and the imports of the next areas taking 15 to 480 ms each on the replay thread (memwatch.ps1, hs1895 / hs1896).
// The pipelines the precompile made keep their own code; what play builds after it goes into a small cache of its own, written
// to <file>.new and merged into the big one at the next start, the file made again whole only when that grows past 64 MB or the
// precompile built something the file lacked. RX_VK_KEEP_PIPELINE_CACHE=1: the cache kept for the whole run, as before.
//
// (relink, 13/11, later) LEAN PIPELINES, then the default on a PC with less than 20 GB of RAM (on every PC since 0.7.6, below):
// letting the cache go gave back 1 GB
// of the 4.3 (the driver keeps the code of the 18900 pipelines made from it: 6.2 GB of its arenas instead of 7.3). What costs
// is having every pipeline the game ever drew alive from the start. Lean: no cache read from the file and none kept, the
// graphics pipelines made when a draw first wants them (as the ones the notes do not name always were: on the workers, the
// draw left out until its pipeline is ready), from the driver's own disk cache; of the compute pipelines only those of the
// programs play has used are made before the game (computes_used, below), the others at their first dispatch on a worker.
// Measured on a 16 GB PC, three adult areas in a row (Gerudo Fortress, Water Temple, Kakariko; memwatch.ps1): RAM available
// 220-1150 MB instead of 24-400, the process's working set steady at 5-6 GB instead of falling to 0.3, private bytes 21-24 GB
// instead of 27-30, hard-fault reads mostly 0 instead of thousands a second; the third area 55.9 fps with one 0.6 s stall
// instead of 36-51 with 0.9-1.4 s ones (hs1902 against hs1890-1896); 0.0-0.1 draws a frame left out for a pipeline.
// RX_VK_LEAN_PIPELINES=0 / 1 chooses; RX_VK_PRECOMPILE set (either way) means the full cache and that choice, as before.
//
// (relink, 15/11, 0.7.6) LEAN ON EVERY PC. The freeze near water (issues 6, 7, 27, 35, 39, 46, 53: the picture stops, then the
// device is lost, on RTX 20 to 50 cards) came in every log from a PC with 24 GB of RAM or more, that is from the full start;
// the bench PC has 16 GB, ran lean, and never froze. With a player's save at Lake Hylia (issue 7) and the full start forced
// there (RX_VK_LEAN_PIPELINES=0, RTX 3060 Ti, 100 % render): the device lost at frame 2925 and at frame 3147, two starts of
// two, a draw of the scene's lit programs running each time; lean, the same save and the same moves: 130 s in the water and
// nothing lost (hs3103, hs3105 against hs3104). Why the full start ends there is not known yet. Lean is also what every
// measure and validation of this port was made with. RX_VK_LEAN_PIPELINES=0: the full start, as PCs with 20 GB and more
// had it until 0.7.5.
bool pipelines_lean() {
    static const bool lean = [] {
        bool on = true;
        if (const char* e = getenv("RX_VK_LEAN_PIPELINES")) on = atoi(e) != 0;
        else if (getenv("RX_VK_PRECOMPILE")) on = false;
        if (on)
            logf("vk: lean pipelines: no pipeline cache kept in memory, graphics pipelines made at their first draw, the compute "
                 "pipelines play has used (this PC's list and the shipped one) before the game (RX_VK_LEAN_PIPELINES=0: all of "
                 "them before the game, as PCs with 20 GB "
                 "of RAM and more had until 0.7.5)");
        else
            logf("vk: full pipelines (RX_VK_LEAN_PIPELINES=0 or RX_VK_PRECOMPILE): every noted pipeline made before the game. "
                 "This start froze near water on NVIDIA cards (issues 6 and 7)");
        return on;
    }();
    return lean;
}
// (relink, lean pipelines) The compute programs play has made a pipeline for, by key, in relink_computes.bin beside the
// executable (8 bytes each, added as they come): what a lean start makes before the game. The full start makes a pipeline for
// every compute variant the program cache keeps, 10328 of them in OoT, most of them texture variants no area uses: 2.8 GB of
// the driver's memory (lean with all of them: arenas 3.8 GB against 1.0 GB with none, hs1910). A program not in the list yet
// is built at its first dispatch on a worker, the dispatch waiting 50 ms at most (dispatch(), below), and is in it from then on.
//
// (relink, 0.7.8) THE SHIPPED LIST. A new install has no list of its own, and neither has a PC that ran the full start until
// 0.7.5 (it made every compute pipeline before the game, so play never noted one): their first minutes of play built each
// compute program's pipeline at its first dispatch, cold on a driver whose cache lacks it (0.5 to 6 s each on NVIDIA), and
// left its dispatches out meanwhile (lighting volumes, culling, exposure). The release ships the list of a play of the game,
// relink_computes_shipped.bin beside the executable (nx2/make_game.py: work/computes.bin, a copy of a played folder's
// relink_computes.bin, the same format), and a lean start builds the programs of both lists before the game. A key is the
// compute stage's code key (vk_shader.cpp code_key, program_pick: a hash of the game's own control section and code), so it is
// the same on every PC and in every build of the back end; the precompile builds every variant the program cache keeps of a
// listed program (2406 keys made 5030 pipelines on the bench). A key the program cache does not hold (a program the seed does
// not have, another version of the game) is passed over at no cost: that program is translated and built at its first
// dispatch, as before. relink_computes.bin stays this PC's own: play notes in it what it builds, whatever the shipped list
// says. RX_VK_NO_SHIPPED_COMPUTES=1: the shipped list is not read
std::mutex computes_used_mu;
std::unordered_set<uint64_t> computes_used;
bool computes_used_read = false;
std::string computes_list_path(const char* name) {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string dir(exe, n);
    return dir.substr(0, dir.find_last_of("\\/") + 1) + name;
}
std::string computes_used_path() { return computes_list_path("relink_computes.bin"); }
/// A compute list's keys (8 bytes each) into keys; how many the file held (0: none, or no file)
size_t computes_list_add(const std::string& path, std::unordered_set<uint64_t>& keys) {
    size_t n = 0;
    if (FILE* f = fopen(path.c_str(), "rb")) {
        uint64_t key = 0;
        while (fread(&key, 8, 1, f) == 1) keys.insert(key), n++;
        fclose(f);
    }
    return n;
}
void computes_used_load() {   // (computes_used_mu held)
    if (computes_used_read) return;
    computes_used_read = true;
    computes_list_add(computes_used_path(), computes_used);
}
void compute_used_note(uint64_t key) {
    std::lock_guard lk(computes_used_mu);
    computes_used_load();
    if (!computes_used.insert(key).second) return;
    if (FILE* f = fopen(computes_used_path().c_str(), "ab")) {
        fwrite(&key, 8, 1, f);
        fclose(f);
    }
}
size_t pcache_loaded = 0;        // (the cache's size, as the driver gives it back, once the files were read into it)
std::vector<uint8_t> pcache_new; // (<file>.new as read at this start: what the small cache starts from)
bool pcache_slimmed = false;     // (the file is the base; pcache holds what was built since, and <file>.new's)
std::vector<uint8_t> pcache_read(const std::string& path) {
    std::vector<uint8_t> data;
    if (FILE* f = fopen(path.c_str(), "rb")) {
        _fseeki64(f, 0, SEEK_END);
        const int64_t sz = _ftelli64(f);
        _fseeki64(f, 0, SEEK_SET);
        if (sz > 32) {
            data.resize((size_t)sz);
            if (fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
        }
        fclose(f);
    }
    // (data of another device or driver is dropped before Vulkan sees it)
    if (data.size() >= 32) {
        uint32_t vendor, device;
        std::memcpy(&vendor, data.data() + 8, 4);
        std::memcpy(&device, data.data() + 12, 4);
        if (vendor != dev.props.vendorID || device != dev.props.deviceID ||
            std::memcmp(data.data() + 16, dev.props.pipelineCacheUUID, VK_UUID_SIZE))
            data.clear();
    }
    return data;
}
void pcache_init() {
    if (pcache || pipelines_lean()) return;   // (lean: no cache of ours, VK_NULL_HANDLE to every pipeline made)
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string dir(exe, n);
    dir = dir.substr(0, dir.find_last_of("\\/") + 1);
    pcache_path = dir + "mkw_vk_pipelines.bin";
    // (relink, RX_VK_PIPELINE_CACHE_PATH=<file>: another pipeline cache, for cold-start tests that must not touch the player's)
    if (const char* e = getenv("RX_VK_PIPELINE_CACHE_PATH")) pcache_path = e;
    const std::vector<uint8_t> data = pcache_read(pcache_path);
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (vkCreatePipelineCache(dev.device, &ci, nullptr, &pcache) != VK_SUCCESS) {
        ci.initialDataSize = 0, ci.pInitialData = nullptr;
        vkCreatePipelineCache(dev.device, &ci, nullptr, &pcache);
    }
    // (what play built since the file was last written whole: with the rest for this start's precompile)
    pcache_new = pcache_read(pcache_path + ".new");
    if (!pcache_new.empty()) {
        VkPipelineCacheCreateInfo mi{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        mi.initialDataSize = pcache_new.size();
        mi.pInitialData = pcache_new.data();
        VkPipelineCache more = VK_NULL_HANDLE;
        if (vkCreatePipelineCache(dev.device, &mi, nullptr, &more) != VK_SUCCESS || vkMergePipelineCaches(dev.device, pcache, 1, &more) != VK_SUCCESS)
            pcache_new.clear();
        if (more) vkDestroyPipelineCache(dev.device, more, nullptr);
    }
    if (vkGetPipelineCacheData(dev.device, pcache, &pcache_loaded, nullptr) != VK_SUCCESS) pcache_loaded = 0;
    logf("vk: pipeline cache %s (%zu KB%s)", pcache_path.c_str(), data.size() >> 10,
         pcache_new.empty() ? "" : (", and " + std::to_string(pcache_new.size() >> 10) + " KB built since").c_str());
}

struct Targets {
    Texture* color[8]{};
    VkImageView color_view[8]{};
    VkFormat color_format[8]{};
    uint32_t color_level[8]{}, color_layer[8]{}, depth_level = 0, depth_layer = 0;   // (the levels and layers viewed)
    Texture* depth = nullptr;
    VkImageView depth_view = VK_NULL_HANDLE;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    int n = 0;
    uint32_t width = 0, height = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    uint8_t stand_in = 0;   // (relink, issue 21: colour i bit i, the view a plain image's standing in for a compressed one)
};

/// A graphics pipeline: built on a worker (vk_jobs.h); ready once pipeline is set (null: it failed)
struct PipeEntry {
    std::atomic<bool> ready{false};
    std::atomic<VkPipeline> pipeline{VK_NULL_HANDLE};   // (a fast-linked one first, the worker's full build over it)
    std::atomic<const Program*> prog{nullptr};          // (the program object whose stages and layout it was built from)
};
std::unordered_map<uint64_t, std::unique_ptr<PipeEntry>> pipelines;   // (the replay thread's)
// (relink, 0.7.8, RX_VK_PIPE_SHARED=1) A parallel replay's thread that meets a pipeline its own map lacks found it in `pipelines`
// under the recording lock (par_mu), and that lock is held for whole texture uploads, image creations and pool imports by the
// thread doing them: in Hyrule Castle's arrival (hs3052) "pipelines" summed 248 ms of the slow submits, for 390 new pipelines
// that each of the five threads met once. With it, a pipeline another thread made is found under this reader lock, without
// par_mu; making one still takes par_mu, and the insertion takes this one too. Lean pipelines only: the full start's precompile
// inserts from its own threads without it
std::shared_mutex pipes_rw;
std::atomic<int64_t> st_pipe_shared_hits{0}, st_pipe_locked{0};
bool pipe_shared_on() {
    static const bool on = [] {
        const bool v = getenv("RX_VK_PIPE_SHARED") && atoi(getenv("RX_VK_PIPE_SHARED")) != 0 && pipelines_lean();
        if (v)
            logf("vk: pipelines another replay thread made are found without the recording lock (RX_VK_PIPE_SHARED=1)");
        return v;
    }();
    return on;
}
int64_t st_draws, st_skipped, st_dispatches, st_passes, st_mesh, st_pending_skips, st_waits;
// (relink) skips by the line that made them, reported with the frame statistics (draw_skip_report)
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
#define SKIP_HERE() do { static const int slot_ = skip_slot(__LINE__); skip_sites[slot_].n.fetch_add(1, std::memory_order_relaxed); st_skipped++; return; } while (0)
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
std::atomic<int64_t> st_pipelines, st_links, st_link_us;
std::atomic<int64_t> st_pipeline_us;
double st_wait_ms;

bool depth_aspect(VkFormat f) {
    return is_depth_format(f) && f != VK_FORMAT_S8_UINT;
}

// (relink) A driver's compiler that faults on a shader (amdvlk64.dll on a mesh program) costs that one draw, not the game
#ifdef _WIN32
__declspec(noinline) VkResult guarded_graphics(VkPipelineCache cache, const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl) {
    __try {
        return vkCreateGraphicsPipelines(dev.device, cache, 1, ci, nullptr, pl);
    } __except (1) {
        return VK_ERROR_UNKNOWN;
    }
}
__declspec(noinline) VkResult guarded_compute_once(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
    __try {
        return vkCreateComputePipelines(dev.device, cache, 1, ci, nullptr, pl);
    } __except (1) {
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
        logf("vk: the driver faulted compiling a graphics pipeline: that one is left out");
        return VK_ERROR_UNKNOWN;
    }
    fault_armed = 1;
    const VkResult r = vkCreateGraphicsPipelines(dev.device, cache, 1, ci, nullptr, pl);
    fault_armed = 0;
    return r;
}
VkResult guarded_compute_once(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
    install_driver_fault_guard();
    if (sigsetjmp(fault_jmp, 1) != 0) {
        logf("vk: the driver faulted compiling a compute pipeline: that one is left out");
        return (VkResult)-77;
    }
    fault_armed = 1;
    const VkResult r = vkCreateComputePipelines(dev.device, cache, 1, ci, nullptr, pl);
    fault_armed = 0;
    return r;
}
#endif
/// (relink) A compute pipeline the driver's compiler fails on (AMD's crashes, -13, or faults, -77, on some programs) is built
/// again without its optimizer, as a graphics one is (build_graphics): run a little slower rather than left out
VkResult guarded_compute(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) {
    const VkResult r = guarded_compute_once(cache, ci, pl);
    if (r == VK_SUCCESS || r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
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
VkPipeline build_graphics(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, int part = -1);
thread_local bool driver_cache_only = false;   // (predict_run's: pipelines for the driver's own cache, not into ours)

/// A target's write mask: the state's channel mask, nothing for a target the fragment shader leaves unwritten
/// (Maxwell keeps its texels; Vulkan writes undefined values: TOTK's maxwell.md 1.1)
uint32_t target_write_mask(const Program& p, const NvnState& s, int i) {
    const Stage* fs = nullptr;
    for (const Stage& st : p.stages)
        if (st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT) fs = &st;
    if (!fs || i >= 8 || !fs->info.stores_frag_color[i]) return 0;
    // (amd-fix, Lichii #353) and none of the components it does not write: Vulkan writes undefined values there, AMD its
    // export registers' garbage (NaN and saturated pixels above OoT's horizon, which the bloom spread). RX_VK_ALL_COMPONENTS=1: off
    static const bool all = getenv("RX_VK_ALL_COMPONENTS") != nullptr;
    const uint32_t written = fs->info.frag_color_components[i];
    return (s.chmask.mask >> (4 * i)) & 0xF & (all || !written ? 0xFu : (uint32_t)written);
}
thread_local const Program* cur_program = nullptr;   // (the draw's, for its dynamic write masks)
/// (relink) What target_write_mask reads of a program, as one word: whether its fragment stage stores each output and which
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

// ---- (relink) The precompile, as TotK Recomp's (runtime/vk_pipeline.c vk_pipeline_prewarm, keys in pipeline-keys.bin): each
// pipeline the game asks for is noted in relink_pipelines.bin, by what built it; a release build, before the game starts,
// makes every kept program ready and builds every noted pipeline on all the CPUs (the driver's own cache, mkw_vk_pipelines.bin,
// makes that quick), so that nothing is built during play. RX_VK_PRECOMPILE=1: in any build; =0: off. A build with
// RX_RELEASE=1 (nx2/build.py) has it on. The notes are written in every build ----
std::mutex note_mu;
FILE* note_file = nullptr;
std::unordered_set<uint64_t> noted;   // (a pipeline once in the file, however many runs meet it)
std::string pipe_notes_path() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string dir(exe, n);
    return dir.substr(0, dir.find_last_of("\\/") + 1) + "relink_pipelines.bin";
}
/// The notes of this build (relink_pipelines.bin), each pipeline once
std::vector<std::unique_ptr<PipeNote>> pipe_notes_read() {
    std::vector<std::unique_ptr<PipeNote>> notes;
    if (FILE* f = fopen(pipe_notes_path().c_str(), "rb")) {
        uint64_t head[2] = {0, 0};
        if (fread(head, 8, 2, f) == 2 && head[0] == program_cache_stamp() && head[1] == sizeof(PipeNote)) {
            std::unordered_set<uint64_t> seen;
            for (;;) {
                auto rec = std::make_unique<PipeNote>();
                if (fread(rec.get(), sizeof(PipeNote), 1, f) != 1) break;
                if (seen.insert(rec->key).second) notes.push_back(std::move(rec));
            }
        }
        fclose(f);
    }
    return notes;
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
void state_record(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, const uint8_t color_types[8]);
uint64_t pipeline_key(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo);
void pipeline_note(uint64_t key, const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo,
                   const uint8_t* color_types = nullptr) {
    static const bool off = getenv("RX_VK_NO_PROGRAM_CACHE") != nullptr;
    if (off) return;
    if (color_types) state_record(p, s, at, topo, color_types);
    auto rec = std::make_unique<PipeNote>();
    rec->key = key, rec->prog = p.key, rec->variant = p.variant, rec->topo = (uint32_t)topo, rec->n = (uint32_t)at.n;
    rec->samples = (uint32_t)at.samples, rec->depth_format = (uint32_t)at.depth_format, rec->has_depth = at.depth != nullptr;
    for (int i = 0; i < 8; i++) rec->color_format[i] = (uint32_t)at.color_format[i];
    rec->s = s;
    std::lock_guard lk(note_mu);
    static const uint64_t stamp = program_cache_stamp();
    if (!note_file) {
        const std::string path = pipe_notes_path();
        uint64_t head[2] = {0, 0};
        if (FILE* f = fopen(path.c_str(), "rb")) {
            if (fread(head, 8, 2, f) != 2) head[0] = 0;
            fclose(f);
        }
        const bool same = head[0] == stamp && head[1] == sizeof(PipeNote);
        if (same) {
            if (FILE* f = fopen(path.c_str(), "rb")) {
                for (int64_t at = 16;; at += sizeof(PipeNote)) {
                    uint64_t k = 0;
                    _fseeki64(f, at, SEEK_SET);
                    if (fread(&k, 8, 1, f) != 1) break;
                    noted.insert(k);
                }
                fclose(f);
            }
        }
        note_file = fopen(path.c_str(), same ? "ab" : "wb");
        if (note_file && !same) {
            const uint64_t h[2] = {stamp, sizeof(PipeNote)};
            fwrite(h, 8, 2, note_file);
        }
    }
    if (note_file && noted.insert(key).second) fwrite(rec.get(), sizeof(PipeNote), 1, note_file), fflush(note_file);
}

// ---- (relink) The pipeline state list: each pipeline the game asks for, by what it is made of and nothing of the game's code
// (its stages' code keys, the vertex input and colour output types its program was translated for, the draw state baked into it), in
// relink_pipeline_states.bin beside the executable. Unlike relink_pipelines.bin it is good for any build of the back end: when
// the shaders are prepared for a new one (prepare_run), every pipeline in it is noted for the precompile, its program translated
// first when the seed's are not that one, so that a new release, or a new install with a list from someone else's play
// (relink_pipeline_states_shipped.bin, nx2/make_game.py: work/pipeline_states.bin), builds them before the game and not in
// play. Only hashes, formats and state: the stages themselves come from the player's own game (relink_shaders.bin) ----
constexpr char kStateMagic[8] = {'H', 'K', 'S', 'T', 'A', 'T', 'E', '1'};
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
std::string state_list_path(const char* name) {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string dir(exe, n);
    return dir.substr(0, dir.find_last_of("\\/") + 1) + name;
}
uint64_t state_hash(const StateRec& r) {
    uint64_t h = 0xCBF29CE484222325ull;
    const uint8_t* b = (const uint8_t*)&r;
    for (size_t i = 0; i < sizeof r; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
/// The records of a state list (none when it is not one)
std::vector<StateRec> state_list_read(const std::string& path) {
    std::vector<StateRec> out;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return out;
    char magic[8];
    uint32_t size = 0;
    if (fread(magic, 8, 1, f) == 1 && !memcmp(magic, kStateMagic, 8) && fread(&size, 4, 1, f) == 1 && size == sizeof(StateRec)) {
        StateRec r;
        while (fread(&r, sizeof r, 1, f) == 1) {
            if (r.nanswers > std::size(r.answers) || r.nquestions > std::size(r.questions)) continue;
            out.push_back(r);
        }
    }
    fclose(f);
    return out;
}
std::mutex state_mu;
FILE* state_file = nullptr;
std::unordered_set<uint64_t> state_seen;
void state_record(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, const uint8_t color_types[8]) {
    static const bool off = getenv("RX_VK_NO_STATE_LIST") != nullptr;
    if (off) return;
    StateRec r{};
    std::memcpy(r.code, p.codes, sizeof r.code);   // (kept with the program: the stages' lock is not taken on the draw's path)
    if (!r.code[0] || p.compute) return;
    r.variant = p.variant;
    r.mesh = p.mesh || p.fallback ? 1 : 0, r.topo = (uint32_t)topo, r.n = (uint32_t)at.n, r.samples = (uint32_t)at.samples;
    r.depth_format = (uint32_t)at.depth_format, r.has_depth = at.depth != nullptr;
    for (int i = 0; i < 8; i++) r.color_format[i] = (uint32_t)at.color_format[i], r.color_types[i] = color_types[i];
    for (int i = 0; i < 16; i++) r.vattr[i] = s.vattr[i], r.stride[i] = s.vstream[i].stride, r.divisor[i] = s.vstream[i].divisor;
    r.nvattr = (uint32_t)s.nvattr;
    for (int i = 0; i < 8; i++) std::memcpy(&r.blend[i], &s.blend[i], 8);
    r.blend_en = (uint32_t)s.color.blend_en, r.chmask = s.chmask.mask, r.poly_mode = (uint32_t)s.poly.mode;
    r.depth_clamp = s.depth_clamp ? 1 : 0, r.a2c = s.ms.a2c ? 1 : 0;
    if (p.answers.size() <= 24 && p.questions.size() <= 32) {
        r.nanswers = (uint32_t)p.answers.size(), r.nquestions = (uint32_t)p.questions.size();
        for (size_t i = 0; i < p.answers.size(); i++) {
            const Program::Answer& a = p.answers[i];
            r.answers[i] = (uint32_t)(a.stage & 0xF) << 28 | (uint32_t)(a.desc & 0xFFFF) << 12 | (a.image ? 1u : 0u) << 8 | a.type;
        }
        std::memcpy(r.questions, p.questions.data(), p.questions.size());
    }
    const uint64_t h = state_hash(r);
    std::lock_guard lk(state_mu);
    if (!state_file) {
        const std::string path = state_list_path("relink_pipeline_states.bin");
        const std::vector<StateRec> had = state_list_read(path);
        for (const StateRec& o : had) state_seen.insert(state_hash(o));
        FILE* probe = fopen(path.c_str(), "rb");
        const bool keep = probe && !had.empty();
        if (probe) fclose(probe);
        state_file = fopen(path.c_str(), keep ? "ab" : "wb");
        if (state_file && !keep) {
            const uint32_t size = sizeof(StateRec);
            fwrite(kStateMagic, 8, 1, state_file), fwrite(&size, 4, 1, state_file);
        }
    }
    if (state_file && state_seen.insert(h).second) fwrite(&r, sizeof r, 1, state_file), fflush(state_file);
}
/// (relink) The workers of the shader preparation before the game (the first start, and after an update that changed the translator):
/// RX_VK_PREPARE_THREADS=<n>, else every CPU thread but one, at most 32. They were half of them, for the PC's sake (on all of them
/// at High the driver's compiles froze it); they run below normal, so the loading screen and the desktop still get the CPU they
/// ask for, and the half left idle made a 4-thread PC's first start twice as long
unsigned prepare_threads() {
    static const unsigned n = [] {
        const unsigned cpus = std::max(1u, std::thread::hardware_concurrency());
        if (const char* e = getenv("RX_VK_PREPARE_THREADS"); e && atoi(e) > 0) return std::min((unsigned)atoi(e), 64u);
        return std::clamp(cpus > 1 ? cpus - 1 : 1u, 1u, 32u);
    }();
    return n;
}
/// (prepare_run, the seed open) every pipeline of the state lists (the player's own and the shipped one): its program translated
/// when the program cache lacks it, then noted in relink_pipelines.bin for precompile_run to build. RX_VK_NO_STATE_LIST=1: off
void state_list_prepare(void (*progress)(size_t done, size_t total)) {
    if (getenv("RX_VK_NO_STATE_LIST")) return;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<StateRec> recs = state_list_read(state_list_path("relink_pipeline_states_shipped.bin"));
    const size_t shipped = recs.size();
    for (const StateRec& r : state_list_read(state_list_path("relink_pipeline_states.bin"))) recs.push_back(r);
    if (recs.empty()) return;
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
    std::atomic<size_t> next{0}, made{0}, missing{0};
    const auto worker = [&] {
        ub_thread_off();   // (a background thread: see ub_thread_off; the pipeline state list's builds raced the replay's updates)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        for (;;) {
            const size_t i = next++;
            if (i >= want.size()) return;
            const StateRec& r = *want[i].r;
            std::unique_ptr<Program> p;
            // (a mesh program on a GPU without mesh shaders is not made; one with a header-only stage is made in play)
            if (!r.code[3] && (!r.mesh || dev.mesh_shader)) p = seed_program_of(r.code, r.mesh != 0, r.vattr, r.color_types, &want[i].key);
            if (!want[i].key) missing++;
            if (p) {
                if (!p->failed) made++;
                for (Stage& st : p->stages)
                    if (st.module) vkDestroyShaderModule(dev.device, st.module, nullptr);
            }
            if (progress) progress(i, want.size());
        }
    };
    std::vector<std::thread> threads;
    const unsigned cpus = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned i = 0; i < prepare_threads(); i++) threads.emplace_back(worker);
    for (std::thread& t : threads) t.join();
    // then each pipeline, noted as a draw would note it
    std::unordered_set<uint64_t> keys;
    for (const Want& w : want)
        if (w.key) keys.insert(w.key);
    // (these keys only: the kept compute programs stay on disk for precompile_run, which builds their pipelines)
    program_cache_load_some(keys, [](Program& p) { ensure_layout(p); }, false);
    // the variants made by other textures (the seed's are for nothing bound): translated with the answers the game's gave
    std::vector<size_t> variant_of;   // (a record of each variant the cache lacks)
    {
        std::unordered_set<uint64_t> seen;
        for (size_t i = 0; i < recs.size(); i++) {
            const StateRec& r = recs[i];
            const uint64_t key = want[rec_program[i]].key;
            if (key && (r.nanswers || r.nquestions) && !program_find(key, r.variant) && seen.insert(mix(key, r.variant)).second)
                variant_of.push_back(i);
        }
    }
    std::atomic<size_t> variants_made{0};
    next = 0;
    const auto variant_worker = [&] {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        for (;;) {
            const size_t i = next++;
            if (i >= variant_of.size()) return;
            const StateRec& r = recs[variant_of[i]];
            std::vector<uint8_t> questions(r.questions, r.questions + r.nquestions);
            std::vector<Program::Answer> answers(r.nanswers);
            for (uint32_t k = 0; k < r.nanswers; k++)
                answers[k] = {(uint16_t)(r.answers[k] >> 28), (uint16_t)(r.answers[k] >> 12 & 0xFFFF), (uint8_t)(r.answers[k] & 0xFF),
                              (r.answers[k] >> 8 & 1) != 0};
            if (const Program* p = seed_variant_of(r.code, r.mesh != 0, r.vattr, r.color_types, questions, answers)) {
                if (p->variant == r.variant) variants_made++;
                ensure_layout(const_cast<Program&>(*p));
            }
        }
    };
    threads.clear();
    for (unsigned i = 0; i < prepare_threads(); i++) threads.emplace_back(variant_worker);
    for (std::thread& t : threads) t.join();
    size_t noted_n = 0, no_variant = 0;
    auto s = std::make_unique<NvnState>();
    for (size_t i = 0; i < recs.size(); i++) {
        const StateRec& r = recs[i];
        const uint64_t key = want[rec_program[i]].key;
        if (!key) continue;
        const Program* p = program_find(key, r.variant);
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
        pipeline_note(pipeline_key(*p, *s, at, (VkPrimitiveTopology)r.topo), *p, *s, at, (VkPrimitiveTopology)r.topo);
        noted_n++;
    }
    logf("vk: state list: %zu pipelines (%zu shipped), %zu programs (%zu translated now, %zu with stages not in the seed), %zu variants "
         "by their textures (%zu made), %zu noted, %zu whose program variant was not made, in %.1f s", recs.size(), shipped,
         want.size(), made.load(), missing.load(), variant_of.size(), variants_made.load(), noted_n, no_variant,
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

// ---- (relink) Stage libraries, as TotK Recomp's runtime/vk_pipeline.c: the part of a pipeline that costs is its shaders, and with
// VK_EXT_graphics_pipeline_library they are built apart from what a draw adds (vertex layout, target formats). prepare_run builds
// each program's two shader libraries once, for the driver's own cache to keep; a draw that needs a pipeline nothing has built gets
// them back from that cache, links them on the spot, and draws at once. The worker's full build then takes over. Needs the blend,
// polygon, clamp and alpha-to-coverage state dynamic, or those would be in the libraries too. RX_VK_GPL=1: on ----
bool gpl_wanted() {
    static const bool on = dev.graphics_pipeline_library && dev.eds3_blend && dev.eds3_polygon && dev.eds3_clamp && dev.eds3_a2c &&
                           getenv("RX_VK_GPL") != nullptr;   // (off until the driver step is safe: linked cold, the libraries stall the draw)
    return on;
}
std::mutex gpl_mu;
std::unordered_map<uint64_t, VkPipeline> gpl_libs;
VkPipeline gpl_part(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, int part, uint64_t key) {
    key = mix(key, (uint64_t)part);
    {
        std::lock_guard lk(gpl_mu);
        if (auto it = gpl_libs.find(key); it != gpl_libs.end()) return it->second;
    }
    VkPipeline made = build_graphics(p, s, at, topo, part);
    if (!made) return VK_NULL_HANDLE;
    std::lock_guard lk(gpl_mu);
    const auto [it, fresh] = gpl_libs.emplace(key, made);
    if (!fresh) vkDestroyPipeline(dev.device, made, nullptr);
    return it->second;
}
VkPipeline gpl_link(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo) {
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
    if (!p.mesh) libs[n++] = gpl_part(p, s, at, topo, 0, input);
    libs[n++] = gpl_part(p, s, at, topo, 1, program);
    libs[n++] = gpl_part(p, s, at, topo, 2, mix(program, (uint64_t)at.samples));
    libs[n++] = gpl_part(p, s, at, topo, 3, output);
    for (uint32_t i = 0; i < n; i++)
        if (!libs[i]) return VK_NULL_HANDLE;
    VkPipelineLibraryCreateInfoKHR li{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
    li.libraryCount = n;
    li.pLibraries = libs;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.pNext = &li;
    ci.layout = p.layout;
    VkPipeline pl = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pl) != VK_SUCCESS) return VK_NULL_HANDLE;
    return pl;
}

/// The pipeline entry of a draw (a new one is queued for a worker; RX_VK_SYNC_PIPELINES=1 builds it here)
/// What a graphics pipeline was built from, so the next draw of the same material does not hash it again
bool pipe_same(uint64_t prog, uint64_t variant, int topo, uint32_t samples, int n, uint32_t depth, const uint32_t* formats,
               int nvattr, const uint32_t* vattr, const uint32_t* div, uint32_t blend_en, const uint64_t* blend,
               uint32_t chmask, uint32_t poly, uint32_t clamp, uint32_t a2c, uint64_t cprog, uint64_t cvariant, int ctopo,
               uint32_t csamples, int cn, uint32_t cdepth, const uint32_t* cformats, int cnvattr, const uint32_t* cvattr,
               const uint32_t* cdiv, uint32_t cblend_en, const uint64_t* cblend, uint32_t cchmask, uint32_t cpoly,
               uint32_t cclamp, uint32_t ca2c) {
    if (prog != cprog || variant != cvariant || topo != ctopo || samples != csamples || n != cn || depth != cdepth ||
        nvattr != cnvattr || blend_en != cblend_en || chmask != cchmask || poly != cpoly || clamp != cclamp || a2c != ca2c)
        return false;
    for (int i = 0; i < 8; i++)
        if (formats[i] != cformats[i] || blend[i] != cblend[i]) return false;
    for (int i = 0; i < 16; i++)
        if (vattr[i] != cvattr[i] || div[i] != cdiv[i]) return false;
    return true;
}
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

/// (relink) the stage a graphics pipeline's vertex input comes from (a fallback program's first stage is a compute one)
const Shader::Info* vertex_info(const Program& p) {
    for (const Stage& st : p.stages)
        if (st.vk_stage == VK_SHADER_STAGE_VERTEX_BIT) return &st.info;
    return p.stages.empty() ? nullptr : &p.stages[0].info;
}

// (relink, RX_VK_LOG_COMPILES=1: each pipeline compile over 300 ms, its kind, time and how many compiles ran beside it: whether
// compiles wait on each other, or some shaders simply take that long)
std::atomic<int> compiles_running{0};
void compile_note(const char* kind, uint64_t key, std::chrono::steady_clock::time_point t0) {
    static const bool on = getenv("RX_VK_LOG_COMPILES") != nullptr;
    if (!on) return;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static std::atomic<int> told{0};
    if (ms >= 300 && told++ < 400) logf("vk: compile %s %016llx: %.0f ms (%d compiles running now)", kind, (unsigned long long)key, ms, compiles_running.load());
}
/// A graphics pipeline's key: what is baked into it (the rest is dynamic state)
uint64_t pipeline_key(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo) {
    uint64_t key = mix(mix(p.key, p.variant), (uint64_t)topology_class(topo) << 8 | (uint64_t)at.samples);
    const Shader::Info& vs = *vertex_info(p);
    if (!p.mesh && !p.fallback) {
        for (int i = 0; i < 16; i++)
            if (vs.loads.Generic(i)) key = mix(key, (uint64_t)s.vattr[i] << 8 | (i < s.nvattr));
        for (int i = 0; i < 16; i++) key = mix(key, (uint64_t)s.vstream[i].divisor);
    }
    for (int i = 0; i < at.n; i++) {
        key = mix(key, (uint64_t)at.color_format[i]);
        if (!dev.eds3_blend) {
            uint64_t b;
            std::memcpy(&b, &s.blend[i], 8);
            key = mix(key, (s.color.blend_en >> i & 1) ? b : 0);
        }
    }
    key = mix(key, (uint64_t)at.depth_format);
    if (!dev.eds3_blend) {
        uint64_t wm = 0;
        for (int i = 0; i < at.n; i++) wm |= (uint64_t)target_write_mask(p, s, i) << (4 * i);
        key = mix(key, wm << 8 | s.color.blend_en);
    }
    if (!dev.eds3_polygon) key = mix(key, (uint64_t)s.poly.mode);
    if (!dev.eds3_clamp) key = mix(key, (uint64_t)s.depth_clamp);
    if (!dev.eds3_a2c) key = mix(key, (uint64_t)s.ms.a2c);
    return key;
}

PipeEntry* graphics_pipeline(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, const uint8_t color_types[8]) {
    // the key: what is baked into it (the rest is dynamic state)
    // (relink: and which variant: p.key is the same for all the variants of a program by its textures' types, and a draw of
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
    // (relink, 15/11, issues 6 and 7: THE CAUSE OF THE FREEZE NEAR WATER) The key names a program by its code and its variant's
    // hash, and two program objects can have both the same: the one the precompile found for a note (program_find) and the one a
    // draw picks. In a Lake Hylia run with every noted pipeline made before the game, seven pipelines were made from one object
    // and drawn with the other, whose shader modules and layout were not the same: all of them the scene's lit programs
    // (b7a09a2173859e73, 0d9fbc089d68dd5b, 20d950f3be10efc4..., the ones every freeze report named). The draw ran one program's
    // shaders under the other's descriptors and constants, its light loop took what it read for a count, and the GPU never
    // finished the draw: device lost in six runs of six (hs3103 to hs3112), none with this (hs3113), same save, same moves.
    // Lean pipelines (0.7.6) only made it rare: there the first draw of a key builds the pipeline, from its own program.
    // So a pipeline belongs to the program object it was built from: another object with the same key gets its own, under a key
    // of its own (a thread's map is by key and object too). RX_VK_PIPE_BY_PROGRAM=0: shared by the key alone, as before
    static const bool by_program = !getenv("RX_VK_PIPE_BY_PROGRAM") || atoi(getenv("RX_VK_PIPE_BY_PROGRAM")) != 0;
    uint64_t key = pipeline_key(p, s, at, topo);
    const uint64_t own = mix(key, (uint64_t)(uintptr_t)&p);
    const uint64_t mkey = by_program ? own : key;
    static thread_local std::unordered_map<uint64_t, PipeEntry*> mine;   // (entries live as long as the cache)
    if (auto m = mine.find(mkey); m != mine.end()) return FINE_X(FC_PIPE_MAP, 1), remember(m->second);
    PROF_EXACT();   // (new to this thread, or new: rare, and a pipeline made can take milliseconds)
    const bool shared = pipe_shared_on() && par_active.load(std::memory_order_relaxed);
    if (shared) {   // (RX_VK_PIPE_SHARED: one made already, by any thread, found without par_mu; the same choice as below)
        PipeEntry* found = nullptr;
        {
            std::shared_lock rl(pipes_rw);
            auto it = pipelines.find(key);
            if (by_program && it != pipelines.end()) {
                const Program* made_for = it->second->prog.load(std::memory_order_acquire);
                if (made_for && made_for != &p) it = pipelines.find(own);
            }
            if (it != pipelines.end()) found = it->second.get();
        }
        const int64_t n = found ? ++st_pipe_shared_hits : ++st_pipe_locked;
        if (n % 2000 == 0) {
            static std::atomic<int> told{0};
            if (told++ < 30)
                logf("vk: pipelines (RX_VK_PIPE_SHARED): %lld found by a replay thread without the recording lock, %lld looked up "
                     "or made under it", (long long)st_pipe_shared_hits.load(), (long long)st_pipe_locked.load());
        }
        if (found) return remember(mine[mkey] = found);
    }
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
    PipeEntry* e;
    {   // (RX_VK_PIPE_SHARED: the map changes under the readers' lock too; the program is set before any reader sees the entry)
        std::unique_lock wl(pipes_rw, std::defer_lock);
        if (pipe_shared_on()) wl.lock();
        auto made = std::make_unique<PipeEntry>();
        made->prog.store(&p, std::memory_order_release);
        e = pipelines.emplace(key, std::move(made)).first->second.get();
    }
    e->prog.store(&p, std::memory_order_release);
    if (!own_key) pipeline_note(key, p, s, at, topo, color_types);   // (its own key holds an address: nothing to note)
    static const bool sync = getenv("RX_VK_SYNC_PIPELINES") != nullptr;
    if (sync) {
        e->pipeline = build_graphics(p, s, at, topo);
        e->ready = true;
        return remember(e);
    }
    // (linked here, from the libraries: the draw goes ahead, and the worker's full build replaces it when it is done)
    if (gpl_wanted() && !p.fallback) {
        const auto t0 = std::chrono::steady_clock::now();
        const VkPipeline linked = gpl_link(p, s, at, topo);
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
    };
    auto job = std::make_shared<Job>();
    job->e = e, job->p = &p, job->s = s, job->at = at, job->topo = topo;
    jobs_push([job] {
        // (over a linked one, which is left alone: draws recorded with it may still be on their way)
        const auto t0 = std::chrono::steady_clock::now();
        compiles_running++;
        if (const VkPipeline full = build_graphics(*job->p, job->s, job->at, job->topo)) job->e->pipeline = full;
        compiles_running--;
        compile_note("graphics", job->p->key, t0);
        job->e->ready.store(true, std::memory_order_release);
    }, false, &e->ready);   // (a draw that waits for it runs it at once if it is still queued: jobs_wait)
    return remember(e);
}

/// part: -1 the whole pipeline, or the graphics pipeline library it stands for: 0 vertex input, 1 pre-rasterization, 2 fragment
/// shader, 3 fragment output
VkPipeline build_graphics(const Program& p, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, int part) {
    const Shader::Info& vs = *vertex_info(p);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    for (const Stage& st : p.stages) {
        // (relink: the fallback's dispatches are not graphics stages. Only a fallback program's: with native mesh shaders
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
            if (dev.vertex_divisor) vdivs.push_back({b, s.vstream[b].divisor});
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
    ms.alphaToCoverageEnable = s.ms.a2c && !dev.eds3_a2c && !no_a2c();
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cb[8]{};
    for (int i = 0; i < at.n; i++) {
        const nvn_blend_st& b = s.blend[i];
        cb[i].blendEnable = (s.color.blend_en >> i & 1) && at.color_format[i] && format_blends(at.color_format[i]) ? VK_TRUE : VK_FALSE;
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
        dyn.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);
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
        static const bool noopt = getenv("RX_VK_NOOPT") != nullptr;
        if (noopt) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        // (from PR 430, YunaraDebugs: on AMD the task and mesh programs (OoT's foliage culling, following pointers through its
        // tables) hung the GPU when the driver's optimizer built them: a device lost one to four minutes into Kokiri, each time with
        // another of them first (RX 6600, issues 461 and 484); never in runs where they were built without it. They are built
        // without it on AMD. RX_VK_MESH_OPT=1: with it, as the others)
        static const bool amd = [] {
            VkPhysicalDeviceProperties pr{};
            vkGetPhysicalDeviceProperties(dev.physical, &pr);
            return pr.vendorID == 0x1002;
        }();
        static const bool mesh_opt = getenv("RX_VK_MESH_OPT") != nullptr;
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
    VkResult r = guarded_graphics(part >= 0 || driver_cache_only ? VK_NULL_HANDLE : pcache, &ci, &pl);
    // (relink, Discord: Artspear, the Market's and Kakariko's dogs white on an RX 6600 in 0.7.9.1: "graphics pipeline for
    // 4e5ce183a33b0855 not made (-13)") A driver whose compiler fails on a program (AMD's proprietary one crashes, -13, on some
    // Ocarina shaders, the dogs' among them since they are translated for 64-lane groups; 5fa56485a6b07619 and 6eaa646d07ea4e43 on
    // an RX 6600 XT since 0.7.8) builds it again without its optimizer, which it does: the draws are then drawn, a little slower,
    // instead of left out. Not for the memory errors, which the optimizer does not cause
    if (r != VK_SUCCESS && r != VK_ERROR_OUT_OF_HOST_MEMORY && r != VK_ERROR_OUT_OF_DEVICE_MEMORY &&
        !(ci.flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT)) {
        const VkResult first = r;
        ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        pl = VK_NULL_HANDLE;
        r = guarded_graphics(part >= 0 || driver_cache_only ? VK_NULL_HANDLE : pcache, &ci, &pl);
        static std::atomic<int> told{0};
        if (told++ < 20)
            logf("vk: %s %016llx failed (%d) with the driver's optimizer: %s", part >= 0 ? "a library of" : "graphics pipeline for",
                 (unsigned long long)p.key, (int)first, r == VK_SUCCESS ? "made without it" : "not made without it either");
    }
    if (part >= 0) {
        if (r != VK_SUCCESS) {
            static std::atomic<int> logged;
            if (logged++ < 10) logf("vk: library %d of %016llx not made (%d)", part, (unsigned long long)p.key, (int)r);
            return VK_NULL_HANDLE;
        }
        return pl;
    }
    if (r != VK_SUCCESS) {
        static std::atomic<int> logged;
        if (logged++ < 10) logf("vk: graphics pipeline for %016llx not made (%d)", (unsigned long long)p.key, (int)r);
        pl = VK_NULL_HANDLE;
    }
    st_pipelines++;
    pcache_built_since_save += !driver_cache_only;
    st_pipeline_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    return pl;
}

// ---- samplers (by NVN sampler parameters) ----
std::unordered_map<uint64_t, VkSampler> samplers;
VkSamplerAddressMode wrap(uint8_t w) {
    switch (w) {
    case NVN_WRAP_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case NVN_WRAP_MODE_MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case NVN_WRAP_MODE_CLAMP_TO_BORDER: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case NVN_WRAP_MODE_MIRROR_CLAMP: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}
// (relink) RX_VK_ANISO=<n> (default 0: the game's own): the samplers that filter linearly between texels and between mip levels
// get at least n-times anisotropic filtering (the game's value if it asks for more; the device's limit if n is above it): surfaces seen
// at a slant keep their detail, for some GPU time. Part of every sampler's key below, so no cached sampler is one made for another n
uint32_t aniso_override() {
    static const uint32_t n = getenv("RX_VK_ANISO") ? (uint32_t)std::clamp(atoi(getenv("RX_VK_ANISO")), 0, 16) : 0u;
    return n;
}
/// compare: the shader reads depth with a reference (a sampler without one cannot compare)
VkSampler sampler_make(int id, bool compare);
// (each texture of each draw found its sampler through reg_mu, a copy of the registration, a hash of it and a map lookup:
// ~48000 times a frame at 3000 draws. Now a thread's table by id and compare, good while no sampler registration's contents
// changed (sampler_registrations_generation). RX_VK_SAMPLER_CACHE=0: off)
VkSampler sampler_for(int id, bool compare) {
    static const bool on = !getenv("RX_VK_SAMPLER_CACHE") || atoi(getenv("RX_VK_SAMPLER_CACHE")) != 0;
    if (!on || id < 0 || id >= 4096) return sampler_make(id, compare);
    struct Entry { uint64_t gen; float bias; VkSampler s; };
    static thread_local Entry table[2][4096];   // (gen 0: never filled; the generation starts at 1)
    const uint64_t gen = sampler_registrations_generation();   // (read first: a registration after it makes the next call miss)
    const float bias = compare ? 0.0f : upscaler_lod_bias();   // (relink: the upscaler's mip bias changes the sampler too)
    Entry& e = table[compare ? 1 : 0][id];
    if (e.gen == gen && e.bias == bias) return e.s;
    const VkSampler made = sampler_make(id, compare);
    e = {gen, bias, made};
    return made;
}
VkSampler sampler_make(int id, bool compare) {
    nvn_samp_par p{};
    if (!sampler_of(id, &p)) p.minf = NVN_MIN_FILTER_LINEAR, p.magf = NVN_MAG_FILTER_LINEAR, p.lod_max = 1000.0f;
    // (relink) DLSS / FSR drawing: the mipmapped textures' bias lowered by upscaler_lod_bias (vk_dlss.cpp), so the upscaled
    // picture gets the output size's texture detail. Not the depth compares (shadow maps), nor samplers without mip levels (the
    // render targets the passes read, the HUD's flat textures): the bias changes nothing there but the key
    const bool mipmapped = p.minf == NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST || p.minf == NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR ||
                           p.minf == NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST || p.minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR;
    if (!compare && mipmapped) p.lod_bias += upscaler_lod_bias();
    uint64_t key = mix(compare ? 0xC0 : 0xAB, aniso_override());
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&p);
    for (size_t i = 0; i < sizeof p; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, b + i, std::min<size_t>(8, sizeof p - i));
        key = mix(key, w);
    }
    static thread_local std::unordered_map<uint64_t, VkSampler> mine;   // (samplers are never destroyed)
    if (auto m = mine.find(key); m != mine.end()) return m->second;
    ParLock lk;
    auto it = samplers.find(key);
    if (it != samplers.end()) return mine[key] = it->second;
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = p.magf == NVN_MAG_FILTER_NEAREST ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    ci.minFilter = (p.minf == NVN_MIN_FILTER_NEAREST || p.minf == NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST ||
                    p.minf == NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    ci.mipmapMode = (p.minf == NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR || p.minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR)
                        ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = wrap(p.ws), ci.addressModeV = wrap(p.wt), ci.addressModeW = wrap(p.wr);
    ci.mipLodBias = std::clamp(p.lod_bias, -dev.props.limits.maxSamplerLodBias, dev.props.limits.maxSamplerLodBias);
    float aniso = p.aniso;
    {   // (RX_VK_ANISO: what the game asks for is told once for each value, 16 at most: what it sets is not known otherwise)
        static std::mutex am;
        static std::unordered_set<uint32_t> asked;
        uint32_t bits;
        std::memcpy(&bits, &p.aniso, sizeof bits);
        std::lock_guard ag(am);
        if (asked.size() < 16 && asked.insert(bits).second)
            logf("vk: sampler anisotropy: the game asks for %g (min filter %u, mip filter %s; RX_VK_ANISO=%u)", (double)p.aniso, (unsigned)p.minf,
                 ci.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? "linear" : "nearest", aniso_override());
    }
    if (aniso_override() && ci.minFilter == VK_FILTER_LINEAR && ci.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR)
        aniso = std::max(aniso, (float)aniso_override());
    ci.anisotropyEnable = aniso > 1.0f && dev.features.samplerAnisotropy;
    ci.maxAnisotropy = std::clamp(aniso, 1.0f, dev.props.limits.maxSamplerAnisotropy);
    ci.compareEnable = compare && p.cmp_mode != 0;
    ci.compareOp = compare_op(p.cmp_func);
    {   // (amd-fix) A depth-reference sample through a sampler that does not compare is undefined in Vulkan: NVIDIA's driver
        // compares anyway, AMD's does not (OoT samples this way: compare mode 0, function 0). A shader that compares gets
        // a comparing sampler: the game's function when it set one, LEQUAL otherwise. RX_VK_DREF_FROM_SAMPLER=1: as before
        static const bool as_before = getenv("RX_VK_DREF_FROM_SAMPLER") != nullptr;
        if (compare && !as_before && !ci.compareEnable) {
            ci.compareEnable = VK_TRUE;
            ci.compareOp = p.cmp_func >= 1 && p.cmp_func <= 8 ? compare_op(p.cmp_func) : VK_COMPARE_OP_LESS_OR_EQUAL;
        }
        static std::mutex lm;
        static std::unordered_set<uint32_t> told;
        std::lock_guard lg(lm);
        if (compare && told.size() < 16 && told.insert((uint32_t)p.cmp_mode << 8 | p.cmp_func).second)
            logf("vk: depth-compare sampler: the game's compare mode %u function %u -> compare %s op %d", p.cmp_mode, p.cmp_func,
                 ci.compareEnable ? "on" : "off", (int)ci.compareOp);
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
    VkSampler sm = VK_NULL_HANDLE;
    vkCreateSampler(dev.device, &ci, nullptr, &sm);
    samplers.emplace(key, sm);
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
/// (relink: a word of any constant buffer: the driver's banks c0 / c2, the program's c1, the game's uniform buffers c3..; the driver
/// bank alone answered 0 for the others)
template <class D>
uint32_t handle_of(const NvnState& s, const Stage& st, const D& d, uint32_t k) {
    if constexpr (requires { d.handle_program; }) {   // (read from a storage buffer: its handle program's)
        if (d.handle_program != ~0U) return stage_handle(s, st, d.handle_program);
    }
    const int nvn_stage = st.nvn_stage;
    const uint32_t step = 1u << d.size_shift;
    (void)nvn_stage;
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
/// keeps its 8-byte address (and the handle is not computed per draw)
bool handle_is_storage_pointer(const Shader::Info& info, const Shader::TextureDescriptor& d, uint32_t k) {
    static const bool off = getenv("RX_VK_HANDLE_POINTER") && getenv("RX_VK_HANDLE_POINTER")[0] == '1';
    if (off || d.handle_program != ~0U) return false;
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

/// (relink, after Fritchii 8193f926f / Lichii #351) how far a read-only pointer window reaches: its pool's
/// end, as far as a storage buffer does (1 GB at most). OoT's froxel fog finds its light probes up to ~130 MB past the pointer:
/// past a 64 MB window they read 0 and the fog took an overbright constant from them (white haze, zones too bright or too dark).
/// Written windows keep their small caps. RX_VK_BARE_READ_MB=<MB> caps it (64: as before)
uint64_t pointer_read_reach() {
    static const uint64_t r = [] {
        const uint64_t reach = std::min<uint64_t>(dev.props.limits.maxStorageBufferRange, 1ull << 30);
        const uint64_t cap = getenv("RX_VK_BARE_READ_MB") ? (uint64_t)atoll(getenv("RX_VK_BARE_READ_MB")) << 20 : reach;
        return std::max<uint64_t>(1ull << 20, std::min(cap, reach - std::min<uint64_t>(reach, 64ull << 10))) & ~255ull;
    }();
    return r;
}

/// The view a texture handle samples (null: nothing resolves)
/// The handles a recording resolved, with what they gave (draws bind the same textures again and again: the slot
/// lookup, the cache's key and the view's are made once a recording, and again when a slot is registered anew)
struct HandleHit {
    VkImageView view;
    Texture* tex;
    uint32_t level;   // (the image level the view starts at)
    uint32_t version; // (the slot's, when resolved)
    uint64_t destroy_gen;
    bool sampling;    // (its use reads the texture's contents)
    uint64_t tex_epoch;   // (the texture epoch it was last touched in)
};
/// (relink) An open-addressing table for the handle cache: the draws' texture bindings looked it up through MSVC's
/// unordered_map (FNV over the key's bytes, a node a lookup), 5% of the replay thread outdoors. Linear probing, power of
/// two, grown at half full; no erase (entries are overwritten, the whole table cleared)
struct HandleTable {
    struct Slot { uint64_t first; HandleHit second; };
    std::vector<Slot> slots;
    std::vector<uint8_t> used;
    size_t n = 0;
    struct Iter {
        Slot* s;
        Slot* operator->() const { return s; }
        bool operator==(const Iter& o) const { return s == o.s; }
        bool operator!=(const Iter& o) const { return s != o.s; }
    };
    static uint64_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; return k; }
    Iter end() { return {nullptr}; }
    Iter find(uint64_t key) {
        if (slots.empty()) return end();
        const size_t mask = slots.size() - 1;
        for (size_t i = mix(key) & mask;; i = (i + 1) & mask) {
            if (!used[i]) return end();
            if (slots[i].first == key) return {&slots[i]};
        }
    }
    void grow() {
        std::vector<Slot> old = std::move(slots);
        std::vector<uint8_t> old_used = std::move(used);
        const size_t cap = old.empty() ? 4096 : old.size() * 2;
        slots.assign(cap, Slot{});
        used.assign(cap, 0);
        n = 0;
        for (size_t i = 0; i < old.size(); i++)
            if (old_used[i]) (*this)[old[i].first] = old[i].second;
    }
    HandleHit& operator[](uint64_t key) {
        if ((n + 1) * 2 > slots.size()) grow();
        const size_t mask = slots.size() - 1;
        for (size_t i = mix(key) & mask;; i = (i + 1) & mask) {
            if (!used[i]) {
                used[i] = 1, slots[i].first = key, slots[i].second = HandleHit{}, n++;
                return slots[i].second;
            }
            if (slots[i].first == key) return slots[i].second;
        }
    }
    size_t size() const { return n; }
    void clear() { std::fill(used.begin(), used.end(), 0), n = 0; }
};
TLS_OBJECT(handle_hits, HandleTable)
// (relink) RX_VK_TRACE_TEX=<w>x<h>,...: each draw sampling a texture of that size is logged with its state, every 30th frame, 600 lines: a
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
thread_local VkFormat last_view_format = VK_FORMAT_UNDEFINED;   // (relink: RX_VK_LOG_IMG, the view last made)
bool log_img_new() {   // (relink) RX_VK_LOG_IMG_NEW=1: RX_VK_LOG_IMG's lines for each new texture pairing, the whole run
    static const bool on = getenv("RX_VK_LOG_IMG_NEW") != nullptr;
    return on;
}
bool log_img_wanted(uint64_t key) {   // (relink) RX_VK_LOG_IMG=<key prefix,...>: its texture and image bindings, 8 times
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
    static const bool log_hits = getenv("RX_VK_LOG_HITS") != nullptr;
    static int64_t st_hit, st_new, st_ver, st_gen, st_n;
    const auto it = handle_hits().find(key);
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
    handle_hits()[key] = {view, tex, level, version, dgen, use == ViewUse::Sample || use == ViewUse::Storage, texture_epoch()};
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
// as the console's hardware sees them. RX_VK_NO_GRID_PATCH=1: as before
thread_local bool grid_patch_want = false;
thread_local uint64_t grid_patch_at = UINT64_MAX;
thread_local VkBuffer grid_patch_buf = VK_NULL_HANDLE;   // (the stream c0 went to: PR 585's streams can move on)
// (relink, issues 6 and 7, 15/11, a diagnosis; RX_VK_CHECKPOINTS=marker only) Every hung draw in the water reports is a forward-lit
// program whose one loop runs min(light grid word, uniform c15 word 452) + 1 times. What the marker report could not say: that
// word, and where the fragment stage's storage buffers (the light grid among them) were bound. Noted as the descriptors are
// built (c15[452] from guest memory as it is then, not an inline update of the buffer), kept with the draw's marker
struct LightProbe {
    uint32_t c15_452 = UINT32_MAX;   // (or c6's: the scenery's programs keep the same constants there)
    uint32_t cbuf = 0, nb = 0;
    struct { uint64_t addr, n; bool bound; } b[3];
};
thread_local LightProbe light_probe_now, light_probe_draw;
bool light_probe_on() {
    static const bool on = getenv("RX_VK_CHECKPOINTS") && !strcmp(getenv("RX_VK_CHECKPOINTS"), "marker");
    return on;
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
        static const bool no_bank_cache = getenv("RX_VK_NO_DESC_REUSE") != nullptr;   // (A/B: filled every time)
        // (relink) a graphics stage's c2 is made from its textures, samplers and images, its c0 from its storage buffers: it stands while
        // those do, whatever else was bound (a uniform buffer for each draw). RX_VK_BANK_GEN_OFF=1: while nothing at all was bound anew)
        static const bool stage_gen_off = getenv("RX_VK_BANK_GEN_OFF") != nullptr;
        const int bank_stage = std::clamp(st.nvn_stage, 0, kNvnStages - 1);
        const uint64_t gen = stage_gen_off || bank_stage > 4 ? s.bind_gen : index == 2 ? s.tex_gen[bank_stage] : s.ssbo_gen[bank_stage];
        if (!no_bank_cache && c.at != UINT64_MAX && c.bind_gen == gen && grid_same && c.size >= n) {
            FINE_N(FC_BANK_GEN, 1);
            return {c.buf, c.at, n};
        }
        static thread_local std::vector<uint8_t> tmp;
        tmp.resize(n);
        FINE_N(FC_BANK_FILLS, 1);
        FINE_N(FC_BANK_FILL_BYTES, n);
        driver_bank_fill(s, (int)index, st.nvn_stage, tmp.data(), n);
        static const bool check = getenv("RX_VK_CHECK_BANKS") != nullptr;   // (the fill against the word path)
        if (check)
            for (uint32_t off = 0; off < n; off += 4) {
                const uint32_t w = driver_bank_word(s, (int)index, st.nvn_stage, off);
                static int bad;
                if (std::memcmp(&w, tmp.data() + off, 4) && bad++ < 20)
                    logf("vk: bank c%u stage %d word %#x: fill %#x, word path %#x", index, st.nvn_stage, off,
                         *(const uint32_t*)(tmp.data() + off), w);
            }
        c.bind_gen = gen;
        std::memcpy(c.grid, s.grid, sizeof c.grid);
        if (c.at != UINT64_MAX && c.size >= n && c.bytes.size() >= n && !std::memcmp(c.bytes.data(), tmp.data(), n) &&
            c.prog != nullptr) {
            FINE_N(FC_BANK_SAME, 1);
            return {c.buf, c.at, n};
        }
        const uint64_t at = stream_alloc(n, align);
        if (at == UINT64_MAX) return bi;
        FINE_N(FC_BANK_NEW, 1);
        FINE_N(FC_BANK_NEW_BYTES, n);
        std::memcpy(cur->mapped + at, tmp.data(), n);
        c.prog = (const Program*)1, c.at = at, c.buf = cur->stream, c.size = n, c.bytes = tmp;
        return {cur->stream, at, n};
    }
    if (index == 1) {
        FINE_SCOPE(FT_C1);
        FINE_N(FC_C1_CALLS, 1);
        const void* key = &st;
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
        if ((index == 15 || index == 6) && st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT && light_probe_on() &&
            used >= 453 * 4 && used < 0x10000) {   // (the marker report's c15[452], or c6[452])
            uint32_t v = 0;
            if ((!u.size || u.size >= 453 * 4) && host->gpu_read(u.addr + 452 * 4, &v, 4))
                light_probe_now.c15_452 = v, light_probe_now.cbuf = index;
        }
        const uint64_t want = used ? std::max<uint64_t>(used, 16) : std::max<uint64_t>(u.size, 16);
        const uint64_t n = std::min<uint64_t>(std::min<uint64_t>(want, u.size ? std::max<uint64_t>(u.size, 16) : want),
                                              dev.props.limits.maxUniformBufferRange);
        // (updates the stream made to this range in this submission or ones still in flight: a copy with them in)
        // (room first: the overlay copies into the stream's mapping, which a move to a bigger stream in the middle would change)
        if (stream_reserve(n, align))
            if (const uint64_t at = ub_overlay(u.addr, n, cur->mapped, [&](uint64_t bytes) { return stream_alloc(bytes, align); });
                at != UINT64_MAX) {
                FINE_N(FC_UBO_OVERLAY, 1);
                return {cur->stream, at, n};
            }
        // A direct import aliases guest RAM. The CPU may start its next frame
        // after present while this frame is still on the GPU, so a descriptor
        // cannot safely retain that live range. A VRAM mirror is queue-ordered
        // and remains zero-copy. RX_VK_UB_SNAPSHOT=0 / 1 overrides automatic.
        static const mkwvk::direct_ub::Mode snapshot = mkwvk::direct_ub::mode(getenv("RX_VK_UB_SNAPSHOT"));
        // (relink: a 2D draw's uniform buffer too, with RX_VK_CAPTURE_UI_UBO=1 only: they are many small ones spread over a
        // transient arena (~900 ranges, 12 MB noted in a minute on the platform, hs460), and the HUD's records are storage buffers)
        static const bool ui_ubo = getenv("RX_VK_CAPTURE_UI_UBO") != nullptr;
        if ((ui_draw_now && ui_ubo) || compute_now) memory_note_ui_buffer(u.addr, n);
        BufferRef b = snapshot == mkwvk::direct_ub::Mode::forced ? BufferRef{} : gpu_buffer_bind(u.addr, n, false);
        const bool keep_copy = mkwvk::direct_ub::should_snapshot(snapshot, b.direct);
        if (b && b.offset % align == 0 && !keep_copy) return {b.buffer, b.offset, n};
        const uint64_t generation = ub_generation(), capture_view = memory_capture_view();
        if (keep_copy) {
            static std::atomic_flag told = ATOMIC_FLAG_INIT;
            if (!told.test_and_set())
                logf("vk: direct uniform buffers copied into the recording stream (RX_VK_UB_SNAPSHOT=0: direct, 1: all)");
            VkBuffer buffer = VK_NULL_HANDLE;
            uint64_t at = 0;
            if (cur->direct_ub_at.find(u.addr, n, generation, capture_view, &buffer, &at)) {
                FINE_N(FC_UBO_REUSE, 1);
                return {buffer, at, n};
            }
        }
        const uint64_t at = stream_alloc(n, align);   // (unaligned or outside the pools: copied into the stream)
        if (at == UINT64_MAX) return bi;
        FINE_N(FC_UBO_STREAM, 1);
        // (as the submission had it, not as the game has rewritten it since: the captures' pages)
        memory_read_as_submitted(u.addr, cur->mapped + at, n);
        if (keep_copy && !cur->direct_ub_at.keep(u.addr, n, generation, capture_view, cur->stream, at)) {
            static std::atomic_flag full_told = ATOMIC_FLAG_INIT;
            if (!full_told.test_and_set())
                logf("vk: direct uniform-buffer snapshot cache full; extra bindings will be copied again");
        }
        static int64_t copied;
        if (!keep_copy && ++copied % 5000 == 1)
            logf("vk: uniform buffer %#llx +%#llx copied into the stream (%s; %lld so)", (unsigned long long)u.addr,
                 (unsigned long long)n, b ? "unaligned" : "outside the pools", (long long)copied);
        return {cur->stream, at, n};
    }
    return bi;
}

struct DescWriter {
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufs;
    std::vector<VkDescriptorImageInfo> imgs;
    std::vector<VkBufferView> views;
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
    static const int on = getenv("RX_VK_SLOW_DESC") != nullptr;
    return on;
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
bool no_bare_table_ptrs() {   // (RX_VK_NO_BARE_TABLE_PTRS=1: storage buffers without a size word (table entries, pointers outside
    // NVN's slots) bound as their words say, as before)
    static const bool v = getenv("RX_VK_NO_BARE_TABLE_PTRS") != nullptr;
    return v;
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

// The table-entry flip log is diagnostic. It stops after 60 lines, and after this many looks: a game that
// never flips was taking a mutex and updating a map for every storage-buffer table entry of every draw.
// unlimited (RX_VK_LOG_TABLE=1) keeps looking until the 60 lines.
bool table_track_open(int logged, int seen, int unlimited) {
    constexpr int kTableTrackLooks = 65536;
    return logged < 60 && (unlimited || seen < kTableTrackLooks);
}

struct PeekBind { VkBuffer buffer; VkDeviceSize offset, range; uint32_t stage, cbuf, off, indirect; uint64_t addr, base; };
constexpr uint32_t kPeekBytes = 256, kPeekSlots = 512;
VkBuffer peek_buffer = VK_NULL_HANDLE;
VmaAllocation peek_alloc = nullptr;
const uint8_t* peek_mapped = nullptr;
struct PeekPending { uint64_t key, frame; uint32_t slot; PeekBind b; };
std::mutex peek_mu;
std::vector<PeekPending> peek_pending;
uint32_t peek_next = 0;
void peek_log_ready() {
    std::lock_guard lk(peek_mu);
    const uint64_t now = texture_frame_no();
    for (auto it = peek_pending.begin(); it != peek_pending.end();) {
        if (now < it->frame + 4) {
            ++it;
            continue;
        }
        char line[1500];
        int k = snprintf(line, sizeof line, "vk: peek %016llx frame %llu stage %u c%u+%#x%s at %#llx (%llu bytes bound):",
                         (unsigned long long)it->key, (unsigned long long)it->frame, it->b.stage, it->b.cbuf, it->b.off,
                         it->b.indirect != ~0U ? " (loaded pointer)" : "", (unsigned long long)it->b.addr, (unsigned long long)it->b.range);
        const uint8_t* w = peek_mapped + (size_t)it->slot * kPeekBytes;
        for (uint32_t i = 0; i < kPeekBytes / 4 && k < 1440; i++) {
            uint32_t v;
            std::memcpy(&v, w + i * 4, 4);
            k += snprintf(line + k, sizeof line - k, " %08x", v);
        }
        logf("%s", line);
        it = peek_pending.erase(it);
    }
}
thread_local std::vector<PeekBind> peek_binds;   // (build_descriptors fills it for the peeked programs)
// (relink) OoT's sky light accumulators: the window build_descriptors bound for the program's pointer (499dbfd0721d2858's c0+0x20,
// ab7699ee7f536755's c0+0x30, the same block), for the dispatch's reset
struct SkySums { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceSize offset = 0, size = 0; };
thread_local SkySums sky_sums;
thread_local SkySums meter_out;   // (relink, RX_VK_METER_ZERO: where an exposure program's sums were bound)
constexpr uint64_t kSkySumsProgram = 0x499dbfd0721d2858ull, kSkyRgblProgram = 0xab7699ee7f536755ull;
bool peek_prog(uint64_t key) {   // (RX_VK_PEEK_PROG=<key prefix>,... or *; RX_VK_PEEK_FRAME=<n>: that frame only)
    static const char* want = getenv("RX_VK_PEEK_PROG");
    if (!want || !*want) return false;
    // (RX_VK_PEEK_FRAME=<n> or <from>-<to>)
    static int64_t frame_lo = -1, frame_hi = -1;
    static const bool frames_read = [] {
        if (const char* e = getenv("RX_VK_PEEK_FRAME")) {
            frame_lo = frame_hi = atoll(e);
            if (const char* d = strchr(e, '-')) frame_hi = atoll(d + 1);
        }
        return true;
    }();
    (void)frames_read;
    const int64_t now = (int64_t)texture_frame_no();
    if (frame_lo >= 0 && (now < frame_lo || now > frame_hi)) {
        if (now > frame_hi) peek_log_ready();   // (what is left to log, after the frames)
        return false;
    }
    if (!strcmp(want, "*")) return true;
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
// (relink, issues 6, 7 and 27, a diagnosis) RX_VK_NULL_TEX=<kinds>: the textures of those kinds that draws sample read as the
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
void build_descriptors(const NvnState& s, const Program& p) {
    if (peek_prog(p.key)) peek_binds.clear();
    if (p.key == kSkySumsProgram || p.key == kSkyRgblProgram) sky_sums = {};   // (relink: bound again below)
    if (p.key == 0x435a0138b9013040ull || p.key == 0x592079442d7129daull) meter_out = {};
    dw().reset();
    draw_sampled().clear();
    light_probe_now = LightProbe{};
    dump_written().clear();
    dump_inputs().clear();
    size_t nbuf = 0, nimg = 0, nview = 0;
    for (const Stage& st : p.stages) {
        for (const auto& d : st.info.constant_buffer_descriptors) nbuf += d.count;
        for (const auto& d : st.info.storage_buffers_descriptors) nbuf += d.count;
        for (const auto& d : st.info.texture_buffer_descriptors) nview += d.count;
        for (const auto& d : st.info.image_buffer_descriptors) nview += d.count;
        for (const auto& d : st.info.texture_descriptors) nimg += d.count;
        for (const auto& d : st.info.image_descriptors) nimg += d.count;
    }
    dw().bufs.reserve(nbuf), dw().imgs.reserve(nimg), dw().views.reserve(nview);   // (stable pointers below)
    FINE_N(FC_DESCS, (int64_t)(nbuf + nimg + nview));
    uint32_t binding = 0;
    auto write = [&](VkDescriptorType type, uint32_t count) {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstBinding = binding++;
        w.descriptorCount = count;
        w.descriptorType = type;
        return w;
    };
    const uint64_t salign = std::max<uint64_t>(dev.props.limits.minStorageBufferOffsetAlignment, 4);
    for (const Stage& st : p.stages) {
        const Shader::Info& info = st.info;
        const int ns = st.nvn_stage;
        for (const auto& d : info.constant_buffer_descriptors) {
            VkWriteDescriptorSet w = write(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, d.count);
            const size_t first = dw().bufs.size();
            for (uint32_t k = 0; k < d.count; k++) {
                const auto t0 = slow_desc_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                dw().bufs.push_back(uniform_range(s, st, d.index + k));
                if (slow_desc_timing()) slow_descriptor(t0, "uniform buffer", d.index + k, dw().bufs.back().range);
            }
            w.pBufferInfo = &dw().bufs[first];
            dw().writes.push_back(w);
        }
        for (const auto& d : info.storage_buffers_descriptors) {
            FINE_SCOPE(FT_STORAGE);
            FINE_N(FC_SSBO_DESCS, d.count);
            VkWriteDescriptorSet w = write(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d.count);
            const size_t first = dw().bufs.size();
            for (uint32_t k = 0; k < d.count; k++) {
                const uint32_t off = d.cbuf_offset + k * 16;
                if (d.count == 1 && d.cbuf_index == 0 && d.cbuf_offset == 0xff0) {   // (EvilOOT, RX_VK_MISS_LOG)
                    dw().bufs.push_back(miss_binding(p.key));
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
                const uint64_t base = addr & ~(salign - 1);
                uint64_t n = std::max<uint64_t>(((size + (addr - base)) + 3) & ~3ull, 16);
                {   // (relink, RX_VK_SSBO_STRETCH=1: a read-only buffer bound to its pool's end, up to 64 MB. Lichii bf3f395c9: the grass's
                    // task and mesh programs read records 12-18 MB past a base whose size word said less, read 0 and culled every blade)
                    static const bool stretch = getenv("RX_VK_SSBO_STRETCH") != nullptr;
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
                // (relink) a storage buffer table's entries (count > 1, Info::global_tables): the ones past the table's
                // end are whatever the constant buffer holds there, bound only when they look like a buffer
                const bool plausible = d.count == 1 || (size && size <= (256ull << 20) && host->gpu_guest(base));
                BufferRef b = addr && plausible ? gpu_buffer_bind(base, n, d.is_written && !bare) : BufferRef{};
                if (light_probe_on() && st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT && light_probe_now.nb < 3)   // (the marker report's)
                    light_probe_now.b[light_probe_now.nb++] = {addr, n, b.buffer != VK_NULL_HANDLE};
                if ((ui_draw_now || compute_now) && b && d.count == 1) memory_note_ui_buffer(base, n);   // (relink: the HUD's records, a compute pass's blocks)
                // (relink: OoT's exposure meter, where its two programs bind their block: vk_memory.cpp meter_bases)
                if ((p.key == 0x435a0138b9013040ull || p.key == 0x592079442d7129daull) && d.is_written && b && n == 0x1040) {
                    memory_meter_range(host->gpu_guest(base), n);
                    meter_out = SkySums{b.buffer, b.offset + (addr - base), n - (addr - base)};
                    if (fast_sync_part("RX_VK_METER_ZERO")) memory_no_late(host->gpu_guest(base), n);
                }
                // (relink) OoT's sky light accumulators: where they were bound, for the dispatch's reset (kSkySumsProgram)
                if (((p.key == kSkySumsProgram && d.cbuf_offset == 0x20) || (p.key == kSkyRgblProgram && d.cbuf_offset == 0x30)) &&
                    d.cbuf_index == 0 && d.count == 1)
                    sky_sums = b ? SkySums{b.buffer, b.offset + (addr - base), n - (addr - base)} : SkySums{};
                if (((p.key == kSkySumsProgram && d.cbuf_offset == 0x20) || (p.key == kSkyRgblProgram && d.cbuf_offset == 0x30)) &&
                    d.cbuf_index == 0 && d.count == 1 && b && !getenv("RX_VK_NO_SKY_ZERO"))
                    memory_no_late(host->gpu_guest(base) + (addr - base), std::min<uint64_t>(n - (addr - base), 0x100));
                // (AyyBeeDee20738, PR 138) a table entry's region that does not fit its pool this moment bound nothing, and the next
                // frame bound it again: its reads flipped between the data and 0, the picture between bright and dark. Smaller,
                // down to 1 MB, rather than nothing (and past what is bound the shaders' search now goes on to the page table)
                static const bool no_shrink = getenv("RX_VK_NO_TABLE_SHRINK") != nullptr;   // (A/B: unbound as before)
                if (!b.buffer && d.count > 1 && addr && plausible && !no_shrink) {
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
                // (relink) all a program may write through it, past what is marked written (a bare entry's guessed range, the
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
                static const int table_unlimited = getenv("RX_VK_LOG_TABLE") != nullptr;
                if (d.count > 1 && table_track_open(table_told.load(std::memory_order_relaxed),
                                                    table_seen.load(std::memory_order_relaxed), table_unlimited)) {
                    table_seen.fetch_add(1, std::memory_order_relaxed);   // (relink) a table entry bound one frame and not the next, 60 lines, then it stops looking
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
                {   // (relink) RX_VK_LOG_WRITEBACK=1: the programs that bind a big written storage buffer (each written
                    // range is copied back to guest memory at the submission's end), 40 of them
                    static const bool log_wb = getenv("RX_VK_LOG_WRITEBACK") != nullptr;
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
                if (b && peek_prog(p.key)) peek_binds.push_back({b.buffer, b.offset, n, (uint32_t)ns, d.cbuf_index, off, ~0U, addr, base});
                dw().bufs.push_back(b ? VkDescriptorBufferInfo{b.buffer, b.offset, n}
                                    : VkDescriptorBufferInfo{dev.null_descriptor ? VK_NULL_HANDLE : zero_buffer, 0,
                                                             dev.null_descriptor ? VK_WHOLE_SIZE : 16});
                {   // (RX_VK_LOG_WRITTEN=1: every storage buffer range a program writes, once, 300 of them)
                    static const bool on = getenv("RX_VK_LOG_WRITTEN") != nullptr;
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
            w.pBufferInfo = &dw().bufs[first];
            dw().writes.push_back(w);
        }
        auto texel_views = [&](const auto& list, VkDescriptorType type, bool storage) {
            for (const auto& d : list) {
                VkWriteDescriptorSet w = write(type, d.count);
                const size_t first = dw().views.size();
                for (uint32_t k = 0; k < d.count; k++) {
                    const uint32_t handle = handle_of(s, st, d, k);
                    nvn_tex t;
                    nvn_view v;
                    bool has_view = false;
                    VkBufferView bv = VK_NULL_HANDLE;
                    if (handle && texture_of((int)(handle & 0xFFFFF), &t, &v, &has_view)) {
                        ParLock lk;
                        bv = texel_buffer_view(&t, has_view ? &v : nullptr, storage);
                    }
                    dw().views.push_back(bv);
                }
                w.pTexelBufferView = &dw().views[first];
                dw().writes.push_back(w);
            }
        };
        texel_views(info.texture_buffer_descriptors, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, false);
        texel_views(info.image_buffer_descriptors, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, true);
        for (const auto& d : info.texture_descriptors) {
            FINE_SCOPE(FT_TEX);
            VkWriteDescriptorSet w = write(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, d.count);
            const size_t first = dw().imgs.size();
            const VkImageViewType vt = view_type(d.type);
            for (uint32_t k = 0; k < d.count; k++) {
                // (Fritchii d603d98bb) A handle word where a storage buffer of the same program keeps its address is not a
                // handle: OoT's water samples each light's texture with a handle loaded per pixel from the light buffer, and
                // the translation falls back to the first constant word on its path, c0[0x520], the light buffer's address.
                // Its low bits named an unrelated texture that changed with the double-buffered buffer: the water flickered
                // every other frame. No texture there now. RX_VK_HANDLE_POINTER=1: the old handle
                // (fragment stages only, the water's case: in other stages (the foliage culling's) the textures went missing and
                // the foliage was culled whenever Link swung his sword. RX_VK_HANDLE_POINTER_ALL=1: every stage, as before)
                static const bool all_stages = getenv("RX_VK_HANDLE_POINTER_ALL") != nullptr;
                const uint32_t handle = ((all_stages || st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT) && handle_is_storage_pointer(info, d, k))
                                            ? 0u : handle_of(s, st, d, k);
                Texture* tex = nullptr;
                VkImageView view = handle ? texture_view_of(handle, vt, ViewUse::Sample, VK_FORMAT_UNDEFINED, &tex)
                                          : VK_NULL_HANDLE;
                if (view && null_tex_wanted(st, d)) view = VK_NULL_HANDLE, tex = nullptr;   // (RX_VK_NULL_TEX)
                if (dump_on()) dump_input('t', (uint32_t)(dw().imgs.size() - 0), handle, view ? tex : nullptr);
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
                static const bool log_3d = getenv("RX_VK_LOG_3D") != nullptr;   // (relink, diagnosis: who samples 3D textures)
                if (log_3d && tex && tex->type == VK_IMAGE_TYPE_3D) {
                    ParLock lk;
                    static std::unordered_set<uint64_t> seen3;
                    if (seen3.size() < 600 && seen3.insert(mix(p.key, tex->width << 16 | tex->depth)).second)
                        logf("vk: 3d frame %llu prog %016llx samples #%u %ux%ux%u fmt %d (shader type %d)%s",
                             (unsigned long long)texture_frame_no(), (unsigned long long)p.key, tex->id, tex->width, tex->height,
                             tex->depth, (int)tex->format, (int)d.type, view ? "" : " DUMMY");
                }
                if (log_img_wanted(p.key) && log_img_new()) {
                    // (RX_VK_LOG_IMG_NEW=1 with RX_VK_LOG_IMG: every new pairing of a program's texture slot with what it got,
                    // at its first frame, for the whole run: what a program samples in one scene and not in another)
                    const uint64_t sig = mix(mix(p.key, (uint64_t)k << 32 | (uint32_t)d.type),
                                             tex ? ((uint64_t)tex->width << 40 | (uint64_t)tex->height << 20 | tex->depth) ^
                                                       ((uint64_t)tex->layers << 56 | (uint64_t)tex->format << 48 | tex->type)
                                                 : (view ? 1u : 2u));
                    ParLock lk;
                    static std::unordered_set<uint64_t> seen;
                    nvn_samp_par sp{};
                    const bool has_sp = sampler_of((int)(handle >> 20), &sp);
                    if (seen.size() < 4000 && seen.insert(mix(sig, handle >> 20)).second)
                        logf("vk: img new frame %llu prog %016llx tex %u: shader type %d depth %d, handle %#x -> %s #%d %ux%ux%u x%u "
                             "levels %u fmt %d imgtype %d cube %d%s; sampler %d%s min %u mag %u wrap %u %u %u aniso %g "
                             "reduction %u cmp %u/%u lod %g..%g bias %g border %g %g %g %g",
                             (unsigned long long)texture_frame_no(),
                             (unsigned long long)p.key, k, (int)d.type, (int)d.is_depth, handle, view ? "view" : "DUMMY",
                             tex ? (int)tex->id : -1, tex ? tex->width : 0, tex ? tex->height : 0, tex ? tex->depth : 0,
                             tex ? tex->layers : 0, tex ? tex->levels : 0, tex ? (int)tex->format : -1, tex ? (int)tex->type : -1,
                             tex ? (int)tex->cube : -1, tex && tex->gpu_written ? " gpu-written" : "", (int)(handle >> 20),
                             has_sp ? "" : " (none: default)", (unsigned)sp.minf, (unsigned)sp.magf, (unsigned)sp.ws,
                             (unsigned)sp.wt, (unsigned)sp.wr, (double)sp.aniso, (unsigned)sp.reduction, (unsigned)sp.cmp_mode,
                             (unsigned)sp.cmp_func, (double)sp.lod_min, (double)sp.lod_max, (double)sp.lod_bias,
                             (double)sp.border[0], (double)sp.border[1], (double)sp.border[2], (double)sp.border[3]);
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
                dw().imgs.push_back({sampler_for((int)(handle >> 20), d.is_depth), view, VK_IMAGE_LAYOUT_GENERAL});
            }
            w.pImageInfo = &dw().imgs[first];
            dw().writes.push_back(w);
        }
        for (const auto& d : info.image_descriptors) {
            VkWriteDescriptorSet w = write(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, d.count);
            const size_t first = dw().imgs.size();
            const VkImageViewType vt = view_type(d.type);
            for (uint32_t k = 0; k < d.count; k++) {
                const uint32_t handle = handle_of(s, st, d, k);
                Texture* tex = nullptr;
                uint32_t level = 0;
                VkImageView view = handle ? texture_view_of(handle, vt, ViewUse::Storage, image_format(d.format), &tex, &level)
                                          : VK_NULL_HANDLE;
                if (view && tex && d.is_written) { ParLock lk; texture_written(tex); }
                if (view && tex) draw_sampled().push_back(tex);
                static const bool log_3d_w = getenv("RX_VK_LOG_3D") != nullptr;   // (relink, diagnosis: who stores into 3D)
                if (log_3d_w && tex && tex->type == VK_IMAGE_TYPE_3D) {
                    ParLock lk;
                    static std::unordered_set<uint64_t> seen3w;
                    if (seen3w.size() < 200 && seen3w.insert(mix(p.key, tex->width << 16 | tex->depth)).second)
                        logf("vk: 3d frame %llu prog %016llx image %s #%u %ux%ux%u fmt %d as vk %d%s",
                             (unsigned long long)texture_frame_no(), (unsigned long long)p.key,
                             d.is_written ? "STORES" : "loads", tex->id, tex->width, tex->height, tex->depth, (int)tex->format,
                             (int)image_format(d.format), view ? "" : " NO VIEW");
                }
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
                if (dump_on()) dump_input(d.is_written ? 'W' : 'R', (uint32_t)dw().imgs.size(), handle, view ? tex : nullptr);
                if (!view && !dev.null_descriptor) view = VK_NULL_HANDLE;
                dw().imgs.push_back({VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL});
            }
            w.pImageInfo = &dw().imgs[first];
            dw().writes.push_back(w);
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
    std::vector<std::vector<uint64_t>> writes;   // (each write's descriptors, as pushed; RX_VK_PUSH_STATE_OLD=1)
    // (relink) the same in one vector of words and where each write's begin: a program change was a vector of vectors resized to the
    // new program's writes, each new one grown word by word (3 allocations a write, and the ones cut off freed: a dozen of each a draw)
    std::vector<uint64_t> words;
    std::vector<uint32_t> at;
};
TLS_OBJECT(pushed, std::array<Pushed, 2>)   // [graphics, compute]
} // namespace
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

void write_words(const VkWriteDescriptorSet& w, std::vector<uint64_t>& out) {
    out.clear();
    out.push_back((uint64_t)w.descriptorType << 32 | w.dstBinding);
    for (uint32_t k = 0; k < w.descriptorCount; k++) {
        if (w.pBufferInfo) {
            const VkDescriptorBufferInfo& b = w.pBufferInfo[k];
            out.push_back((uint64_t)b.buffer), out.push_back(b.offset), out.push_back(b.range);
        } else if (w.pImageInfo) {
            const VkDescriptorImageInfo& i = w.pImageInfo[k];
            out.push_back((uint64_t)i.sampler), out.push_back((uint64_t)i.imageView);
        } else if (w.pTexelBufferView) {
            out.push_back((uint64_t)w.pTexelBufferView[k]);
        }
    }
}

/// write_words' words appended to out (the words of the pushed state, a write after another)
void words_append(const VkWriteDescriptorSet& w, std::vector<uint64_t>& out) {
    out.push_back((uint64_t)w.descriptorType << 32 | w.dstBinding);
    for (uint32_t k = 0; k < w.descriptorCount; k++) {
        if (w.pBufferInfo) {
            const VkDescriptorBufferInfo& b = w.pBufferInfo[k];
            out.push_back((uint64_t)b.buffer), out.push_back(b.offset), out.push_back(b.range);
        } else if (w.pImageInfo) {
            const VkDescriptorImageInfo& i = w.pImageInfo[k];
            out.push_back((uint64_t)i.sampler), out.push_back((uint64_t)i.imageView);
        } else if (w.pTexelBufferView) {
            out.push_back((uint64_t)w.pTexelBufferView[k]);
        }
    }
}
/// Whether the words in [o, end) are write_words' of w (1), or not (0: they are after the call). -1: w is not the write they were made for
/// (its words do not fill them exactly), and nothing was touched
int words_same_store(const VkWriteDescriptorSet& w, uint64_t* o, const uint64_t* end) {
    const size_t per = w.pBufferInfo ? 3 : w.pImageInfo ? 2 : w.pTexelBufferView ? 1 : 0;
    if ((size_t)(end - o) != 1 + (size_t)w.descriptorCount * per) return -1;
    bool same = true;
    auto put = [&](uint64_t v) {
        same &= *o == v;
        *o++ = v;
    };
    put((uint64_t)w.descriptorType << 32 | w.dstBinding);
    for (uint32_t k = 0; k < w.descriptorCount; k++) {
        if (w.pBufferInfo) {
            const VkDescriptorBufferInfo& b = w.pBufferInfo[k];
            put((uint64_t)b.buffer), put(b.offset), put(b.range);
        } else if (w.pImageInfo) {
            const VkDescriptorImageInfo& i = w.pImageInfo[k];
            put((uint64_t)i.sampler), put((uint64_t)i.imageView);
        } else if (w.pTexelBufferView) {
            put((uint64_t)w.pTexelBufferView[k]);
        }
    }
    return same ? 1 : 0;
}

bool bind_descriptors(VkCommandBuffer cmd, VkPipelineBindPoint bp, const Program& p) {
    DescWriter& wr = dw();   // One TLS accessor for the writer used throughout this bind.
    if (wr.writes.empty()) return true;
    Pushed& ps = pushed()[bp == VK_PIPELINE_BIND_POINT_COMPUTE];
    if (p.push) {
        static const bool whole = getenv("RX_VK_PUSH_ALL") != nullptr;   // (A/B: every binding every time)
        static const bool old_state = getenv("RX_VK_PUSH_STATE_OLD") != nullptr;   // (A/B: a vector of words a write, as before)
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
            if (old_state) {
                ps.writes.resize(wr.writes.size());
                for (size_t i = 0; i < wr.writes.size(); i++) write_words(wr.writes[i], ps.writes[i]);
            } else {
                ps.words.clear(), ps.at.clear();
                for (const VkWriteDescriptorSet& w : wr.writes) ps.at.push_back((uint32_t)ps.words.size()), words_append(w, ps.words);
            }
            return true;
        }
        static thread_local std::vector<VkWriteDescriptorSet> changed;
        static thread_local std::vector<uint64_t> words;
        changed.clear();
        if (old_state) {
            for (size_t i = 0; i < wr.writes.size(); i++) {
                write_words(wr.writes[i], words);
                if (words != ps.writes[i]) changed.push_back(wr.writes[i]), ps.writes[i].swap(words);
            }
        } else {   // (the same layout, so the same count of words in each write: compared and kept in place)
            for (size_t i = 0; i < wr.writes.size(); i++) {
                const uint64_t* end = ps.words.data() + (i + 1 < ps.at.size() ? ps.at[i + 1] : ps.words.size());
                const int same = words_same_store(wr.writes[i], ps.words.data() + ps.at[i], end);
                if (same < 0) {   // (not so: nothing is known of what is pushed; the whole set again)
                    ps.cmd = VK_NULL_HANDLE;
                    return bind_descriptors(cmd, bp, p);
                }
                if (!same) changed.push_back(wr.writes[i]);
            }
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
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(dev.device, &ai, &set) != VK_SUCCESS) {
        cur->extra.push_back(make_pool());
        ai.descriptorPool = cur->extra.back();
        if (vkAllocateDescriptorSets(dev.device, &ai, &set) != VK_SUCCESS) return false;
    }
    for (auto& w : wr.writes) w.dstSet = set;
    vkUpdateDescriptorSets(dev.device, (uint32_t)wr.writes.size(), wr.writes.data(), 0, nullptr);
    vkCmdBindDescriptorSets(cmd, bp, p.layout, 0, 1, &set, 0, nullptr);
    return true;
}

// ---- RX_VK_CHECKPOINTS: a marker before each draw and dispatch (VK_NV_device_diagnostic_checkpoints); when the
// device is lost, the markers the GPU reached say which program it was running ----
struct Checkpoint { uint64_t prog; uint64_t frame; uint32_t seq; char kind; };
Checkpoint checkpoints[1 << 16];
std::atomic<uint32_t> checkpoint_n{0};
// (relink, issues 6, 7 and 27) with the buffer markers, each draw's targets and the GPU-written textures it samples, kept beside
// its marker: the report then says which pass the hung draw was in and what it read, which a program key alone did not (the
// same program draws in several passes, and the RTX 50 reports named a different program almost every time).
// Made only when the buffer markers are (RX_VK_CHECKPOINTS=marker): 8 MB
struct CheckpointDraw {
    uint32_t width = 0, height = 0, c0 = 0, c0_format = 0, depth = 0, depth_format = 0, n = 0;
    struct { uint32_t id, format; uint16_t w, h, d; uint8_t type, layers; } s[6];
    LightProbe light;   // (relink, 15/11: its fragment stage's c15[452] and storage buffers, as bound)
};
std::unique_ptr<CheckpointDraw[]> checkpoint_draws;
// (igpu-fix) a line of text beside a marker (checkpoint_note): which texture, level and layer a decode or an encode works on.
// Made with the buffer markers: 8 MB
struct CheckpointNote { char text[128]; };
std::unique_ptr<CheckpointNote[]> checkpoint_notes;

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
void checkpoint(VkCommandBuffer cmd, const Program& p, char kind, const Targets* at = nullptr) {
    if (!checkpoints_on()) return;
    const uint32_t seq = ++checkpoint_n, i = (seq - 1) & 0xFFFF;
    checkpoints[i] = {p.key, texture_frame_no(), seq, kind};
    if (checkpoint_notes) checkpoint_notes[i].text[0] = 0;
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
            c.light = light_probe_draw;   // (this draw's: set with gfx_sampled)
        }
    }
    set_checkpoint(cmd, i, seq);
}
const char* checkpoint_kind(char c) {
    return c == 'd' ? "draw of program" : c == 'c' ? "dispatch of program" : c == 'i' ? "INDIRECT dispatch of program"
         : c == 'm' ? "mirror sync of pool"
         : c == 'b' ? "write-back of pool" : c == 'u' ? "upload of texture" : c == 'g' ? "GPU unswizzle of texture"
         : c == 'A' ? "ASTC decode of texture" : c == 'E' ? "BC3 encode into texture" : c == 'R' ? "resize blit for texture"
         : c == 'Z' ? "end of the uploads of texture"
         : c == 'P' ? "PRESENT to the window, present number" : c == 'B' ? "UPLOADS: readback refresh, pools"
         : c == 'T' ? "UPLOADS: page table, pages" : c == 'K' ? "READBACK of a GPU-drawn texture to guest memory, texture"
         : c == 'S' ? "START of command buffer of submission" : c == 'F' ? "END of command buffer of submission"
         : "?";
}

thread_local uint64_t fallback_payload_address = 0, fallback_records_address = 0;   // (the draw's fallback buffers)

void push_constants(VkCommandBuffer cmd, const Program& p, const Targets* at) {
    checkpoint(cmd, p, at ? 'd' : 'c', at);
    if (!p.uses_push) return;   // (most programs read none: a push a draw was ~5% of the replay's commands)
    FINE_N(FC_PUSHC, 1);
    uint8_t data[kPushSize]{};
    const float one = 1.0f;
    std::memcpy(data + Shader::Backend::SPIRV::RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, &one, 4);
    {   // (guest memory's page table, for the pointers compute, task and mesh stages follow)
        static const uint64_t pages = memory_guest_pages();
        std::memcpy(data + Shader::Backend::SPIRV::GUEST_PAGES_PUSH_OFFSET, &pages, 8);
    }
    if (at && mesh_args_ready()) {   // (a task program's firsts: entry 0's 0 until its draw says otherwise, never address 0)
        const uint64_t firsts = mesh_args_address + 12;
        std::memcpy(data + Shader::Backend::SPIRV::MESH_FIRST_PUSH_OFFSET, &firsts, 8);
    }
    if (p.fallback) {   // (the fallback's records and payloads, pushed as addresses)
        std::memcpy(data + Shader::Backend::SPIRV::MESH_FALLBACK_PAYLOAD_PUSH_OFFSET, &fallback_payload_address, 8);
        std::memcpy(data + Shader::Backend::SPIRV::MESH_FALLBACK_RECORDS_PUSH_OFFSET, &fallback_records_address, 8);
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
    const uint32_t seq = ++checkpoint_n, i = (seq - 1) & 0xFFFF;
    checkpoints[i] = {key, texture_frame_no(), seq, kind};
    if (checkpoint_draws) checkpoint_draws[i] = CheckpointDraw{};
    if (checkpoint_notes) checkpoint_notes[i].text[0] = 0;
    set_checkpoint(cmd, i, seq);
}
// (igpu-fix) a marker with a line of text kept beside it, printed in the device-lost report
void checkpoint_note(VkCommandBuffer cmd, uint64_t key, char kind, const char* fmt, ...) {
    if (!checkpoints_on()) return;
    const uint32_t seq = ++checkpoint_n, i = (seq - 1) & 0xFFFF;
    checkpoints[i] = {key, texture_frame_no(), seq, kind};
    if (checkpoint_draws) checkpoint_draws[i] = CheckpointDraw{};
    if (checkpoint_notes) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(checkpoint_notes[i].text, sizeof checkpoint_notes[i].text, fmt, ap);
        va_end(ap);
    }
    set_checkpoint(cmd, i, seq);
}

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
    checkpoint_notes.reset(new CheckpointNote[1 << 16]());
    marker_buffer = buffer;
    marker_words = (volatile uint32_t*)words;
    logf("vk: RX_VK_CHECKPOINTS: VK_AMD_buffer_marker, a marker before each draw and dispatch (memory type %u)", type);
}

namespace {
// (the markers the GPU reached whose earlier work never finished: the hang is in the work after the last marker that did)
void buffer_markers_report(const char* why = "device lost") {
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
    logf("vk: %s: %u markers set, %u reached; the last whose earlier work all finished: %u", why, n, reached, last_done);
    // (relink, issue 7: a report listed the markers after the last finished one as "running" though none was reached, and
    // read as if the two dispatches after the hung draw were running too. Each line now says what its marker's words say, and
    // a draw's line its targets and the GPU-written textures it samples (#id, as RX_VK_LOG_IMG and the pass traces name them))
    const auto line = [&state](const char* what, uint32_t s) {
        const uint32_t i = (s - 1) & 0xFFFF;
        const Checkpoint& k = checkpoints[i];
        static const char* const states[] = {"not reached", "reached, earlier work NOT done", "earlier work done"};
        char extra[720] = "";
        if (checkpoint_draws && k.kind == 'd' && checkpoint_draws[i].width) {
            const CheckpointDraw& c = checkpoint_draws[i];
            int m = snprintf(extra, sizeof extra, "; into %ux%u, colour #%u (vk %u), depth #%u (vk %u)", c.width, c.height, c.c0,
                             c.c0_format, c.depth, c.depth_format);
            for (uint32_t j = 0; j < c.n && m > 0 && m < 400; j++)
                m += snprintf(extra + m, sizeof extra - m, "%s #%u %ux%ux%u%s vk %u", j ? "," : "; samples", c.s[j].id,
                              (unsigned)c.s[j].w, (unsigned)c.s[j].h, c.s[j].type == VK_IMAGE_TYPE_3D ? (unsigned)c.s[j].d
                                                                                                   : (unsigned)c.s[j].layers,
                              c.s[j].type == VK_IMAGE_TYPE_3D ? " 3D" : "", c.s[j].format);
            // (relink, 15/11) the forward-lit programs' loop bound and light grid: a c15[452] in the thousands would make the
            // loop run to its 8192-turn guard on every pixel; a storage buffer "not bound" reads 0 (robust access)
            if (c.light.c15_452 != UINT32_MAX && m > 0 && m < 560)
                m += snprintf(extra + m, sizeof extra - m, "; c%u[452] %u", c.light.cbuf, c.light.c15_452);
            for (uint32_t j = 0; j < c.light.nb && m > 0 && m < 640; j++)
                m += snprintf(extra + m, sizeof extra - m, "%s %#llx+%#llx%s", j ? "," : "; storage",
                              (unsigned long long)c.light.b[j].addr, (unsigned long long)c.light.b[j].n,
                              c.light.b[j].bound ? "" : " (not bound)");
        }
        if (checkpoint_notes && checkpoint_notes[i].text[0] && !extra[0]) snprintf(extra, sizeof extra, "; %s", checkpoint_notes[i].text);
        logf("vk:   %s: marker %u [%s] (%s %016llx, frame %llu)%s", what, s, states[state(s)], checkpoint_kind(k.kind),
             (unsigned long long)k.prog, (unsigned long long)k.frame, extra);
    };
    for (uint32_t s = last_done > 6 ? last_done - 6 : 1; last_done && s < last_done; s++) line("before", s);
    // (from the last finished marker to the first reached one whose earlier work did not finish: what the GPU was running)
    const uint32_t start = last_done ? last_done : first;
    const uint32_t end = std::max(start, stuck.empty() ? start + 8 : std::min(stuck.front(), start + 16));
    for (uint32_t s = start; s && s <= end && s <= n; s++) line(s == last_done ? "RUNNING after" : "after", s);
    for (size_t j = 0; j < stuck.size() && j < 12; j++) line("reached, earlier work unfinished", stuck[j]);
    if (stuck.empty() && last_done)
        logf("vk:   no marker was reached with its earlier work unfinished: the GPU stopped in the work recorded right after marker "
             "%u in its command buffer (the draw or dispatch it names, then any barrier, copy or end of pass before the next "
             "marker)", last_done);
    {   // (igpu-fix) every marker numbered after the last finished one that the GPU reached, and the upload-side ones (a recording's
        // uploads are numbered among its draws but run before them): the start and end of each command buffer, the decodes
        uint32_t shown = 0, more = 0;
        for (uint32_t s = last_done + 1; last_done && s <= n; s++) {
            const char k = checkpoints[(s - 1) & 0xFFFF].kind;
            if (!state(s) && (k == 'd' || k == 'c' || k == 'i')) continue;
            if (shown < 120) shown++, line("after (reached, or not a draw)", s);
            else more++;
        }
        if (more) logf("vk:   ... and %u more such markers", more);
    }
    {   // (markers numbered below the last finished one and never reached. With the parallel replay the workers number their
        // segments' markers side by side, so the highest finished marker can be a segment the GPU ran before the one that hung:
        // the hang may then be just before the lowest of these, in a segment of its own (the draws of one pass share their
        // targets: the #ids tell the segments apart). In a serial replay: another command buffer, submitted later)
        uint32_t below = 0;
        std::vector<uint32_t> lowest;
        for (uint32_t s = first; s && s < last_done; s++)
            if (!state(s) && below++ < 4) lowest.push_back(s);
        if (below) {
            logf("vk:   %u markers numbered below %u were never reached (RX_VK_PARALLEL=0 numbers them in the order the GPU runs "
                 "them); the six before the lowest, then the lowest:", below, last_done);
            for (uint32_t s = lowest[0] > 6 ? lowest[0] - 6 : 1; s < lowest[0]; s++) line("before the lowest never reached", s);
            for (const uint32_t s : lowest) line("never reached", s);
        }
    }
    // (relink, 15/11) With the parallel replay the markers are numbered as each worker records, so "the last whose earlier work
    // all finished" can be another command buffer's (an RTX 5080's report: the last done 1710626, reached and unfinished from
    // 1709970). Within one command buffer the numbers do run in order: the lowest marker reached and not finished is the most
    // telling, and the work before it is what hung. Its eight before, with what each marker says
    if (!stuck.empty()) {
        const uint32_t s0 = stuck.front();
        for (uint32_t s = s0 > 8 ? s0 - 8 : 1; s < s0; s++) {
            const uint32_t i = (s - 1) & 0xFFFF;
            const bool r = marker_words[2 * i] == s, d = marker_words[2 * i + 1] == s;
            line(d ? "before the first unfinished: done" : r ? "before the first unfinished: reached, NOT done"
                                                            : "before the first unfinished: not reached", s);
        }
    }
}
} // namespace

void draw_records_log(uint64_t lo, uint64_t hi);
// (relink, issues 6, 7 and 27) RX_VK_PASS_TRACE=1: when the GPU hangs, the last frames' passes and dispatches in stream order go to
// relink_passes_hang.txt beside the game, once: the pass the hung draw was in, and what wrote the textures it read
static void hang_pass_trace() {
    static std::atomic<bool> written{false};
    if (!pass_trace_on() || written.exchange(true)) return;
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string path(exe, n);
    path = path.substr(0, path.find_last_of("\\/") + 1) + "relink_passes_hang.txt";
    const uint64_t f = texture_frame_no();
    pass_traces_write(path.c_str(), f > 7 ? f - 7 : 0, f);
    logf("vk: the passes of frames %llu to %llu written to relink_passes_hang.txt (RX_VK_PASS_TRACE)",
         (unsigned long long)(f > 7 ? f - 7 : 0), (unsigned long long)f);
}
// (relink, issue 7) the GPU finished nothing for `seconds` while work waited on it, the device not lost (yet): an RTX 5070 Ti
// stopped this way in the water and never reported a loss, so neither report above ever ran. Once a run: with the buffer
// markers (RX_VK_CHECKPOINTS=marker) what the GPU was running; without them, the hint that turns them on
void gpu_stall_report(double seconds, uint64_t done, uint64_t waited) {
    static std::atomic<bool> done_once{false};
    if (done_once.exchange(true)) return;
    logf("vk: GPU STALLED: no work finished for %.1f s (finished %llu, waiting for %llu)", seconds, (unsigned long long)done,
         (unsigned long long)waited);
    {
        const uint64_t f = texture_frame_no();
        draw_records_log(f > 6 ? f - 6 : 0, f);
    }
    hang_pass_trace();
    if (marker_words) {
        buffer_markers_report("GPU stalled");
        std::lock_guard lk(watch_mu);
        for (int i = 0; i < 40; i++)
            if (!watch_ring[(watch_at + i) % 40].empty()) logf("%s", watch_ring[(watch_at + i) % 40].c_str());
    } else {
        logf("vk: GPU stalled: no markers to read (RX_VK_CHECKPOINTS=marker in relink_env.txt names the work that hung)");
    }
}
void device_lost_notice();   // (vk_main.cpp)
void checkpoints_report() {
    static bool done;
    if (done) return;
    done = true;
    device_lost_notice();
    if (getenv("RX_VK_FAULT_INFO")) device_fault_report();   // (now, while the process is still alive: the faulting address)
    {   // (RX_VK_FLASH_RECORD=<program,...>: those programs' last draws, sizes and all)
        const uint64_t f = texture_frame_no();
        draw_records_log(f > 6 ? f - 6 : 0, f);
    }
    hang_pass_trace();
    // (checkpoints first: on NVIDIA vkGetDeviceFaultInfoEXT has not come back after a fault, which lost the report)
    // (relink, issue 66: six logs said "reading the fault info" and then nothing, RX_VK_FAULT_INFO not being set; and
    // RX_VK_CHECKPOINTS=1 is the NV checkpoints where the driver has them, not the buffer markers)
    logf("vk: device lost: reading %s", dev.checkpoints ? "the checkpoints"
                                        : marker_words ? "the buffer markers"
                                        : getenv("RX_VK_FAULT_INFO") ? "the fault info"
                                        : "nothing: no markers on (RX_VK_CHECKPOINTS=marker, RX_VK_PARALLEL=0 and RX_VK_FAULT_INFO=1 "
                                          "in relink_env.txt name the work that stopped)");
    struct FaultLast { ~FaultLast() { if (std::getenv("RX_VK_FAULT_INFO")) device_fault_report(); } } fault_last;
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
    std::vector<const Texture*> sampled;   // (rendered textures its draws sampled: a target among them is a hazard)
    uint16_t sampled_targets = 0;          // (colour i bit i, depth bit 8: targets earlier draws sampled)
    uint16_t written = 0;                  // (its targets written so far: colour i bit i, depth bit 8)
    uint8_t stand_in = 0;                  // (relink, issue 21: colour i bit i, a stand-in copied back at its end)
};
TLS_OBJECT(pass, Pass)
thread_local bool work_since_barrier = false;   // (GPU work recorded since the last full barrier: a dispatch orders after it)
// what the command buffer has bound (draws skip what is already there)
thread_local bool last_desc_valid = false;
thread_local const Program* last_desc_prog = nullptr;
thread_local uint64_t last_desc_bind = 0, last_desc_reg = 0, last_desc_ub = 0;
thread_local VkPipeline last_pipeline = VK_NULL_HANDLE;
struct LastVb { VkBuffer b[16]; VkDeviceSize o[16], z[16], s[16]; };
thread_local LastVb last_vb;
thread_local bool last_vb_valid = false;
// (the guest streams and index buffer already bound on this command buffer; a new recording drops them)
struct VertexSet {
    uint64_t stamp = 0;
    uint64_t addr[16]{};
    uint64_t size[16]{};
    uint32_t stride[16]{};
    bool on = false;
};
thread_local VertexSet vertex_set;
struct IndexBind {
    uint64_t buffer = 0, offset = 0;
    uint32_t type = 0;
    bool on = false;
};
thread_local IndexBind last_ib;
int64_t st_desc_reused;

thread_local uint64_t targets_serial = ~0ull;   // (the state's render targets resolved once per NOP_SET_RT)
thread_local Targets cached_targets;
thread_local uint64_t targets_dgen = 0;   // (relink: and while no texture went or moved: their views are the images')

Targets targets(NvnState& s) {
    if (targets_serial == s.rt_serial && targets_dgen == texture_destroy_generation()) return cached_targets;
    ParLock lk;
    Targets at;
    at.n = std::clamp(s.rt.n, 0, 8);
    for (int i = 0; i < at.n; i++) {
        const nvn_tex* t = s.rt.color[i];
        Texture* tex = t ? texture_get(t, false) : nullptr;
        if (tex && !(tex->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) tex = nullptr;
        if (!tex) continue;
        const nvn_view* v = s.rt.has_cview[i] ? &s.rt.cview[i] : nullptr;
        VkFormat f = VK_FORMAT_UNDEFINED;
        const VkImageView view = texture_view(tex, t, v, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Attachment, &f);
        if (!view) continue;
        at.color[i] = tex, at.color_view[i] = view, at.color_format[i] = f;
        const uint32_t level = image_level(tex, view_level(v));
        at.color_level[i] = level, at.color_layer[i] = v ? view_layer(v) : 0;
        uint32_t w = std::max(1u, tex->width >> level), h = std::max(1u, tex->height >> level);
        // (a raw view of a compressed target is one texel a block: its extent in blocks. The texels'
        // 64 x 64 over the 16 x 16 view of OoT's noise volumes put the render area past the attachment)
        const auto bc = [](VkFormat x) { return x >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && x <= VK_FORMAT_BC7_SRGB_BLOCK; };
        if (bc(tex->format) && !bc(f)) {
            w = (w + 3) / 4, h = (h + 3) / 4;
            // (relink, issue 21: a plain image of that size stands in for it, copied in and out around the pass)
            if (texture_is_stand_in(tex, view)) at.stand_in |= (uint8_t)(1u << i);
        }
        at.width = at.width ? std::min(at.width, w) : w;
        at.height = at.height ? std::min(at.height, h) : h;
        at.samples = tex->samples;
    }
    // (a texture at two slots: the later one left unattached, as nxnative does for BotW's G-buffer)
    for (int i = 0; i < at.n; i++)
        for (int j = 0; j < i; j++)
            if (at.color_view[i] && at.color_view[i] == at.color_view[j]) {
                at.color[i] = nullptr, at.color_view[i] = VK_NULL_HANDLE, at.color_format[i] = VK_FORMAT_UNDEFINED;
                at.stand_in &= (uint8_t)~(1u << i);
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
    targets_dgen = texture_destroy_generation();
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
/// (relink, 17/11, issue 47 and the Discord's striped ground) Whether a draw left out of these targets is drawn again in the next
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
    Pass& ps = pass();   // One TLS accessor for this recording thread's pass state.
    bool same = ps.open && ps.depth == at.depth_view && ps.n == at.n && ps.width == at.width &&
                ps.height == at.height;
    for (int i = 0; i < at.n && same; i++) same = ps.color[i] == at.color_view[i];
    if (same) return;
    end_pass(cmd);
    full_barrier(cmd, "barrier (pass start)");   // (what the pass before wrote, and the transfers before it)
    work_since_barrier = true;
    if (at.stand_in) {   // (relink, issue 21) what the compressed targets hold, into the images that stand in for them
        for (int i = 0; i < at.n; i++)
            if (at.stand_in >> i & 1) texture_stand_in_copy(cmd, at.color[i], at.color_view[i], true);
        full_barrier(cmd, "barrier (stand-in in)");
    }
    ps.stand_in = at.stand_in;
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
        ps.tex[i] = at.color[i], ps.level[i] = at.color_level[i], ps.layer[i] = at.color_layer[i];
        texture_written(at.color[i]);
        texture_rendered(at.color[i]);
    }
    for (int i = at.n; i < 8; i++) ps.color[i] = VK_NULL_HANDLE;
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    if (at.depth) {
        da.imageView = at.depth_view;
        da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        texture_written(at.depth);
        texture_rendered(at.depth);
    }
    ps.depth = at.depth_view;
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
thread_local DynState last_dyn;
thread_local bool last_dyn_valid = false;
#define DYN_CHANGED(f) ((!last_dyn_valid || std::memcmp(&d.f, &last_dyn.f, sizeof d.f)) && (FINE_X(FC_DYN_CALLS, 1), true))

// A renderer helper may bind its own graphics pipeline, dynamic state, vertex/index streams and push descriptors.
// Forget every cached guest draw binding before the next draw; otherwise it can skip commands whose Vulkan state the
// helper replaced. This is deliberately separate from descriptors_disturbed(), which also covers compute-only helpers.
void graphics_state_disturbed() {
    last_dyn_valid = false, last_desc_valid = false, last_vb_valid = false, last_pipeline = VK_NULL_HANDLE;
    vertex_set.on = false, last_ib.on = false;
    descriptors_disturbed();
}

/// The inputs a draw's dynamic state is built from. Most draws repeat the draw before them.
bool dyn_same(const float* fp, int nf, const uint32_t* up, int nu, const uint8_t* raw, int nb, const float* cfp,
              const uint32_t* cup, const uint8_t* craw) {
    if (nf <= 0 || nu <= 0 || nb <= 0) return false;
    for (int i = 0; i < nf; i++)
        if (fp[i] != cfp[i]) return false;
    for (int i = 0; i < nu; i++)
        if (up[i] != cup[i]) return false;
    for (int i = 0; i < nb; i++)
        if (raw[i] != craw[i]) return false;
    return true;
}
constexpr int kDynF = 13, kDynU = 35;
struct DynFast {
    bool on = false;
    float fp[kDynF]{};
    uint32_t up[kDynU]{};
    uint8_t raw[160]{};
    int nb = 0;
    uint32_t wm0 = 0, wm_n = 0;
};
thread_local DynFast dyn_fast;
int dyn_pack(const NvnState& s, const Targets& at, VkPrimitiveTopology topo, bool mesh, float* fp, uint32_t* up,
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
    // (relink) what the program makes of the write masks, not the program: draws change program far more often than they change that,
    // and each program change worked the whole state out again. RX_VK_DYN_PROG_KEY=1: the program, as before)
    static const bool by_program = getenv("RX_VK_DYN_PROG_KEY") != nullptr;
    const uint64_t prog = by_program || !cur_program ? (uint64_t)(uintptr_t)cur_program : program_wm_sig(*cur_program);
    std::memcpy(raw + atb, &prog, 8);
    return atb + 8;
}

void dynamic_state(VkCommandBuffer cmd, const NvnState& s, const Targets& at, VkPrimitiveTopology topo, bool mesh) {
    Pass& ps = pass();   // One TLS accessor for this recording thread's pass state.
    float df[kDynF];
    uint32_t du[kDynU];
    uint8_t db[160];
    const int nb = dyn_pack(s, at, topo, mesh, df, du, db);
    if (!dump_on() && last_dyn_valid && dyn_fast.on && dyn_fast.nb == nb &&
        dyn_same(df, kDynF, du, kDynU, db, nb, dyn_fast.fp, dyn_fast.up, dyn_fast.raw)) {
        ps.wm_or |= dyn_fast.wm0, ps.wm_n = dyn_fast.wm_n;   // (the pass trace still sees the masks)
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
    const bool fp = dump_on() && at.depth && depth_aspect(at.depth_format) && cur_program && !footprint.empty() &&
                    std::find(footprint.begin(), footprint.end(), cur_program->key) != footprint.end();
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
    {   // (relink) The characters' own shadow maps (square, depth only, 768 and less; 1e26755d projects them on the screen): their
        // stored depth is moved away from the light, or they shadow themselves whole: Link and the ferns black in the shade, while
        // the ground behind them keeps their shadow. RX_VK_SHADOW_BIAS=<units>,<slope>; 0: as the game asks
        static float units = -1, slope = 0;
        if (units < 0) {
            units = 250000.0f, slope = 0.0f;
            if (const char* e = getenv("RX_VK_SHADOW_BIAS")) sscanf(e, "%f,%f", &units, &slope);
        }
        // (relink, issue 48, an experiment) The broken pots' shards go dark in blocks with straight screen-aligned edges, the ground
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
                         at.n, (unsigned long long)(cur_program ? cur_program->key : 0), s.poly.offset_en ? "on" : "off",
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
            static const bool log_on = getenv("RX_VK_LOG_CASCADE_BIAS") != nullptr;
            static int logged;
            if (log_on && logged < 40) {
                logged++;
                logf("vk: cascade %ux%u bias: enabled %d units %g clamp %g slope %g", at.width, at.height, (int)d.bias_en,
                     d.bias[0], d.bias[1], d.bias[2]);
            }
            if (cu != 0 || cs != 0) d.bias[0] += cu, d.bias[2] += cs, d.bias_en = true;
        }
    }
    {   // (relink, issue 48, a diagnosis) RX_VK_LOG_DEPTH_TARGETS=1: each depth-only target a program draws into, once per size,
        // format, program and bias: the game's own depth bias and the one drawn with (the characters' shadow maps above). Which maps
        // the broken pots' shards are drawn into, and whether the rule above takes them
        static const bool on = getenv("RX_VK_LOG_DEPTH_TARGETS") && atoi(getenv("RX_VK_LOG_DEPTH_TARGETS")) != 0;
        if (on && at.n == 0 && at.depth) {
            const uint64_t prog = cur_program ? cur_program->key : 0;
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
    {   // (relink) A colour pass after a depth pass tests EQUAL against what another program wrote. Intel's compiler does not give two
        // programs the same position to the last bit, and the ground came out in holes: the sky through them, a white forest.
        // A small margin towards the camera instead. RX_VK_EQ_BIAS=<units>, 0: as the game asks
        static const float eq_bias = getenv("RX_VK_EQ_BIAS") ? (float)atof(getenv("RX_VK_EQ_BIAS")) : dev.props.vendorID == 0x8086 ? 64.0f : 0.0f;
        if (eq_bias > 0 && d.dt && !d.dw && d.dop == VK_COMPARE_OP_EQUAL) {
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
    d.poly = s.poly.mode == 0 ? VK_POLYGON_MODE_POINT : s.poly.mode == 1 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    static const bool fp_lines = getenv("RX_VK_FOOTPRINT_LINES") != nullptr;   // (the footprints as wireframes)
    if (fp && fp_lines) d.poly = VK_POLYGON_MODE_LINE;
    d.clamp = s.depth_clamp, d.a2c = s.ms.a2c != 0 && !no_a2c();
    d.n = dev.eds3_blend ? at.n : 0;
    for (int i = 0; i < d.n; i++) {
        const nvn_blend_st& b = s.blend[i];
        d.en[i] = (s.color.blend_en >> i & 1) && at.color_format[i] && format_blends(at.color_format[i]) ? VK_TRUE : VK_FALSE;
        d.eq[i] = {blend_factor(b.src_rgb), blend_factor(b.dst_rgb), blend_op(b.eq_rgb), blend_factor(b.src_a),
                   blend_factor(b.dst_a), blend_op(b.eq_a)};
        d.wm[i] = target_write_mask(*cur_program, s, i);
    }
    if (dump_on()) {   // (RX_VK_DUMP: the state a program draws with, once per change)
        char line[400];
        int k = snprintf(line, sizeof line, "prog %016llx n%d depth %s dt%u dw%u dop%d st%u cull%u front%u vp %.0f,%.0f %.0fx%.0f z %.2f-%.2f sc %d,%d %ux%u:",
                         (unsigned long long)(cur_program ? cur_program->key : 0), at.n,
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
    if (!last_dyn_valid) vkCmdSetLineWidth(cmd, 1.0f), vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
    if (DYN_CHANGED(cull)) vkCmdSetCullMode(cmd, d.cull);
    if (DYN_CHANGED(front)) vkCmdSetFrontFace(cmd, d.front);
    if (!mesh) {
        if (DYN_CHANGED(topo)) vkCmdSetPrimitiveTopology(cmd, d.topo);
        if (DYN_CHANGED(restart) || last_dyn.topo == VK_PRIMITIVE_TOPOLOGY_MAX_ENUM) vkCmdSetPrimitiveRestartEnable(cmd, d.restart);
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
        const bool same_n = last_dyn_valid && last_dyn.n == d.n;
        if (!same_n || std::memcmp(d.en, last_dyn.en, sizeof(VkBool32) * d.n))
            FINE_X(FC_DYN_CALLS, 1), vkCmdSetColorBlendEnableEXT(cmd, 0, (uint32_t)d.n, d.en);
        if (!same_n || std::memcmp(d.eq, last_dyn.eq, sizeof(VkColorBlendEquationEXT) * d.n))
            FINE_X(FC_DYN_CALLS, 1), vkCmdSetColorBlendEquationEXT(cmd, 0, (uint32_t)d.n, d.eq);
        if (!same_n || std::memcmp(d.wm, last_dyn.wm, sizeof(VkColorComponentFlags) * d.n))
            FINE_X(FC_DYN_CALLS, 1), vkCmdSetColorWriteMaskEXT(cmd, 0, (uint32_t)d.n, d.wm);
    }
    last_dyn = d;
    last_dyn_valid = true;
    dyn_fast.on = true, dyn_fast.nb = nb;
    std::memcpy(dyn_fast.fp, df, sizeof df);
    std::memcpy(dyn_fast.up, du, sizeof du);
    std::memcpy(dyn_fast.raw, db, (size_t)nb);
    dyn_fast.wm0 = d.n ? (uint32_t)d.wm[0] : 0, dyn_fast.wm_n = (uint32_t)d.n;
    ps.wm_or |= dyn_fast.wm0, ps.wm_n = dyn_fast.wm_n;
}
#undef DYN_CHANGED

void make_dummies() {
    if (dummies.view[0] || dev.null_descriptor) return;
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
VkPipelineLayout clear_layout = VK_NULL_HANDLE;

VkPipeline clear_pipeline(const Targets& at, int index, uint32_t mask, int kind) {
    uint64_t key = mix((uint64_t)index << 8 | mask << 4 | (uint64_t)kind, (uint64_t)at.samples);
    for (int i = 0; i < at.n; i++) key = mix(key, (uint64_t)at.color_format[i]);
    key = mix(key, (uint64_t)at.depth_format);
    auto it = clear_pipes.find(key);
    if (it != clear_pipes.end()) return it->second;
    if (!clear_layout) {
        const VkPushConstantRange r{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &r;
        vkCreatePipelineLayout(dev.device, &li, nullptr, &clear_layout);
    }
    const uint32_t* fw = kind == 1 ? CLEAR_U_FRAG_SPV : kind == 2 ? CLEAR_I_FRAG_SPV : CLEAR_F_FRAG_SPV;
    const size_t fn = (kind == 1 ? sizeof CLEAR_U_FRAG_SPV : kind == 2 ? sizeof CLEAR_I_FRAG_SPV : sizeof CLEAR_F_FRAG_SPV) / 4;
    const std::vector<uint32_t> frag = spirv_location(fw, fn, (uint32_t)index);
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE;
    mi.codeSize = sizeof CLEAR_VERT_SPV, mi.pCode = CLEAR_VERT_SPV;
    vkCreateShaderModule(dev.device, &mi, nullptr, &vm);
    mi.codeSize = frag.size() * 4, mi.pCode = frag.data();
    vkCreateShaderModule(dev.device, &mi, nullptr, &fm);
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
    vkCreateGraphicsPipelines(dev.device, pcache, 1, &ci, nullptr, &pl);
    vkDestroyShaderModule(dev.device, vm, nullptr);
    vkDestroyShaderModule(dev.device, fm, nullptr);
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

/// A clear after draws of the open pass: in a pass of its own, after a full barrier (begin_pass makes it). In the pass
/// the draws before it may still read what it clears (a target they sample: the depth the fog and the lighting read):
/// vkCmdClearAttachments is ordered with the attachments' own writes only, and the GPU cleared tiles while a draw
/// before it still sampled them: a band of the picture, below or above a line that moved, for a frame (the fog's flat
/// colour over a Grand Prix intro, the shadows' dark band in its race). RX_VK_CLEAR_IN_PASS=1: in the pass, as before.
void clear_after_draws(VkCommandBuffer cmd) {
    static const bool in_pass = getenv("RX_VK_CLEAR_IN_PASS") != nullptr;
    if (in_pass || !pass().open || pass().draws == 0) return;
    static int logged;
    if (logged++ < 6)
        logf("vk: a clear after %d draws of its pass (%ux%u, %d colour targets%s, last program %016llx): a pass of its own",
             pass().draws, pass().width, pass().height, pass().n, pass().depth ? " + depth" : "", (unsigned long long)pass().prog);
    end_pass(cmd);
}

bool writes_sampled_target(uint16_t writes, uint16_t sampled_targets) {
    return (writes & sampled_targets) != 0;
}
/// A draw that samples a target this pass already wrote, or writes a target an earlier draw of this pass sampled: in
/// a pass of its own, after a full barrier (begin_pass makes it), as for the clears above: in one pass the GPU orders
/// the targets' writes with each other only, not with the draws' texture reads of them. (A draw that samples a target
/// it writes itself is left as it is.) Returns the targets the draw writes. RX_VK_NO_PASS_HAZARDS=1: not checked.
uint16_t draw_hazards(VkCommandBuffer cmd, const Program& p, const NvnState& s, const Targets& at) {
    Pass& ps = pass();   // One TLS accessor for this recording thread's pass state.
    uint16_t writes = 0;
    for (int i = 0; i < at.n; i++)
        if (at.color[i] && target_write_mask(p, s, i)) writes |= (uint16_t)(1u << i);
    if (at.depth && s.ds.dw) writes |= 1u << 8;
    static const bool off = getenv("RX_VK_NO_PASS_HAZARDS") != nullptr;
    if (off || !ps.open) return writes;
    bool same = ps.depth == at.depth_view && ps.n == at.n && ps.width == at.width && ps.height == at.height;
    for (int i = 0; i < at.n && same; i++) same = ps.color[i] == at.color_view[i];
    if (!same) return writes;   // (other targets: begin_pass ends this pass with a barrier anyway)
    auto slot_of = [&](const Texture* t) {
        for (int i = 0; i < at.n; i++)
            if (at.color[i] == t) return i;
        return at.depth == t ? 8 : -1;
    };
    const char* kind = nullptr;
    for (const Texture* t : gfx_sampled()) {
        const int k = slot_of(t);
        if (k >= 0 && (ps.written >> k & 1) && !(writes >> k & 1)) { kind = "samples a target the pass wrote"; break; }
    }
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
    last_dyn_valid = false, last_desc_valid = false, last_vb_valid = false, last_pipeline = VK_NULL_HANDLE;
    vertex_set.on = false;   // (the next draw binds its vertex streams again, as last_vb_valid above asks)
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

void draw_begin_recording(int slot, VkCommandBuffer cmd, uint64_t value) {
    rec_cmd = cmd;
    rec_value = value;
    ParLock lk;
    if ((size_t)slot >= res.size()) res.resize((size_t)slot + 1);
    RecordingRes& r = res[slot];
    if (!r.pool) {
        r.pool = make_pool();
        void* mapped = nullptr;
        RecordingRes::Stream s{};
        s.size = kStream;
        create_buffer(s.size, kStreamUse, true, s.buffer, s.alloc, &mapped, true);
        s.mapped = (uint8_t*)mapped;
        r.streams.push_back(s);
    } else {
        vkResetDescriptorPool(dev.device, r.pool, 0);
        for (VkDescriptorPool p : r.extra) vkDestroyDescriptorPool(dev.device, p, nullptr);
        r.extra.clear();
        // (the streams the last recording of the slot grew into: given back after 120 recordings that stayed in the first, the
        // slot's last one being done (recording_wait). RX_VK_STREAM_KEEP=1: kept)
        static const bool keep = getenv("RX_VK_STREAM_KEEP") != nullptr;
        if (r.streams.size() > 1 && !keep) {
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
    {
        const RecordingRes::Stream& s = r.streams[0];
        r.si = 0, r.stream = s.buffer, r.stream_alloc = s.alloc, r.mapped = s.mapped, r.size = s.size;
    }
    if (r.before + r.used > stream_peak.load(std::memory_order_relaxed)) stream_peak.store(r.before + r.used, std::memory_order_relaxed);
    static int64_t begun;
    if (++begun % 3000 == 0)
        logf("vk: the recording streams: %zu of %llu MB, the most one held %llu KB; %llu made on demand (%llu MB kept now)", res.size(),
             (unsigned long long)(kStream >> 20), (unsigned long long)(stream_peak.load(std::memory_order_relaxed) >> 10),
             (unsigned long long)stream_grown.load(), (unsigned long long)(stream_extra.load() >> 20));
    r.used = 0, r.before = 0;
    r.c1_at.clear();
    r.direct_ub_at.reset();
    cur = &r;
    pass() = Pass{};
    work_since_barrier = false;
    last_dyn_valid = false, last_desc_valid = false, last_vb_valid = false, last_pipeline = VK_NULL_HANDLE;
    vertex_set.on = false, last_ib.on = false;
    descriptors_disturbed();
    static thread_local uint64_t hits_value = ~0ull;   // (resolved targets last a submission, through its parts: no CPU write is
    if (value != hits_value) {            // collected between them; resolved handles last while their slot and
        hits_value = value;               // texture do, and are touched again once a submission)
        targets_serial = ~0ull;
        if (handle_hits().size() > 65536) handle_hits().clear();
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
    static const bool v = getenv("RX_VK_PASS_TRACE") != nullptr;
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

// (relink) RX_VK_PROBE=1 (replay is serial then): the centre texel of every colour target after each pass and of
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
            static const bool hex = getenv("RX_VK_PROBE_UBO_HEX") != nullptr;   // (the words themselves: which constant went wrong)
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
    Pass& ps = pass();   // One TLS accessor for this recording thread's pass state.
    if (!ps.open) return;
    vkCmdEndRendering(cmd);
    ps.open = false;
    if (ps.stand_in) {   // (relink, issue 21) what the pass rendered into stand-ins, back into the compressed targets
        full_barrier(cmd, "barrier (stand-in out)");
        for (int i = 0; i < ps.n; i++)
            if (ps.stand_in >> i & 1) texture_stand_in_copy(cmd, ps.tex[i], ps.color[i], false);
        full_barrier(cmd, "barrier (stand-in out)");
        ps.stand_in = 0;
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
        snprintf(label, sizeof label, "pass %ux%u f%d %dc%s, %s draws, first prog %016llx, last prog %016llx", ps.width, ps.height,
                 (int)ps.format0, ps.n, ps.depth ? "+d" : "", ps.draws > 256 ? ">256" : ps.draws > 32 ? "33-256" : ps.draws > 4 ? "5-32" : "0-4",
                 (unsigned long long)ps.first_prog, (unsigned long long)ps.prog);
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
        const bool upd = ub_overlay(u.addr, sizeof f, (uint8_t*)f, [](uint64_t) { return (uint64_t)0; }) != UINT64_MAX;
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
        const bool upd = ub_overlay(u.addr, sizeof f, (uint8_t*)f, [](uint64_t) { return (uint64_t)0; }) != UINT64_MAX;
        if (!upd) host->gpu_read(u.addr, f, sizeof f);
        n += snprintf(line + n, sizeof line - n, " c%d%s %#llx+%#llx [", i + 3, upd ? "*" : "", (unsigned long long)u.addr,
                      (unsigned long long)u.size);
        for (int k = 0; k < 16 && n < 1900; k++) n += snprintf(line + n, sizeof line - n, k ? " %.5g" : "%.5g", f[k]);
        n += snprintf(line + n, sizeof line - n, "]");
    }
    logf("%s", line);
}
extern int64_t presents_now;   // (vk_replay.cpp)
uint32_t stage_cbuf_word(const NvnState& s, const Stage& st, uint32_t index, uint32_t offset);   // (vk_shader.cpp)
// (relink) RX_VK_EXPOSURE_LOG=1: the final pass's exposure (c3+0x38 of its fragment stage, OoT's 73320e720a317817) at each
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
    {   // (relink) the final pass's program and what it samples, once a second: where the exposure comes from
        static int64_t last_told = -1000;
        if (presents_now - last_told >= 60) {
            last_told = presents_now;
            const nvn_p_bind_buf& u = s.ubo[1][0];
            logf("vk: exposure pass: program %016llx, c3 %#llx +%#llx;%s", (unsigned long long)p.key, (unsigned long long)u.addr,
                 (unsigned long long)u.size, program_textures(s, p).c_str());
        }
    }
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
    if (dev.mesh_shader) {
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
    StatsQuery(VkCommandBuffer c, uint64_t prog, const DrawArgs& d) { begin(c, prog, d, false, nullptr); }
    /// a compute dispatch: its grid, or its indirect arguments (12 bytes, read back as a draw's)
    StatsQuery(VkCommandBuffer c, uint64_t prog, uint64_t indirect, uint32_t x, uint32_t y, uint32_t z) {
        if (stats_dir()) watch_note(prog, indirect);
        DrawArgs d{};
        d.indirect = indirect;
        const uint32_t grid[3] = {x, y, z};
        begin(c, prog, d, true, grid);
    }
    void begin(VkCommandBuffer c, uint64_t prog, const DrawArgs& d, bool dispatch, const uint32_t* grid) {
        if (!stats_dir() || !stats_pool) return;
        const int64_t at = stats_frame.load(std::memory_order_relaxed);
        if (at < 0 || (int64_t)texture_frame_no() != at || stats_written.load(std::memory_order_relaxed)) return;
        const uint32_t i = stats_used.fetch_add(1);
        if (i >= kStatsQueries) return;
        stats_meta[i] = {prog, (uint8_t)d.mesh, (uint8_t)(d.indirect != 0), (uint8_t)d.indexed, 0, d.count, d.instances, d.max,
                         d.stride, d.count_addr, d.indirect, {}, 0, (uint8_t)dispatch, {}, 0, 0};
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

// ---- (EvilOOT) NVN2's indirect mesh draws in Vulkan's layout ----
// NVN2 keeps an indirect mesh draw as {task count, first task} (as its direct DrawMeshTasks(first, count) and NVIDIA's
// mesh shading always did); vkCmdDrawMeshTasksIndirectEXT reads {x, y, z} workgroups. Read as they were, "7 tasks from 0"
// was 7 x 0 x 1: no work, so every GPU-built mesh draw (OoT's foliage, grass, falling leaves, fireflies) drew nothing even
// once its counts were right. Before each, outside the pass, its records go to a ring as {count, 1, 1} (a fill of 1s,
// then each record's count word copied over x). RX_VK_MESH_IND_COUNT_WORD=1 takes the second word for the count
// instead; RX_VK_MESH_IND_RAW=1 draws from the records as they are, as before. A first task other than 0 has no Vulkan
// counterpart (as for the direct draws: drawn from 0).
namespace {
constexpr uint64_t kMeshRing = 4ull << 20;
VkBuffer mesh_ring = VK_NULL_HANDLE;
VmaAllocation mesh_ring_mem = nullptr;
std::mutex mesh_ring_mu;
uint64_t mesh_ring_at = 0;
thread_local VkBuffer mesh_conv_buf = VK_NULL_HANDLE;   // (the converted records for the draw in progress)
thread_local uint64_t mesh_conv_off = 0;
thread_local uint32_t mesh_conv_n = 0;   // (how many records were converted: the draw asks for no more)
bool mesh_indirect_convert(VkCommandBuffer cmd, const DrawArgs& d) {
    mesh_conv_buf = VK_NULL_HANDLE;
    static const bool raw = getenv("RX_VK_MESH_IND_RAW") && atoi(getenv("RX_VK_MESH_IND_RAW")) != 0;
    static const uint32_t word = getenv("RX_VK_MESH_IND_COUNT_WORD") ? (uint32_t)atoi(getenv("RX_VK_MESH_IND_COUNT_WORD")) & 3 : 0;
    // (amd-fix: the rewrite with the firsts in the draw does it, RX_VK_MESH_ARGS; this one drew every draw from its first 0)
    static const bool rewrite = getenv("RX_VK_MESH_ARGS") && atoi(getenv("RX_VK_MESH_ARGS")) != 0;   // (sharkiix version: off)
    if (raw || rewrite || !d.mesh || !d.indirect) return false;
    PROF_EXACT();   // (a pass ended and three barriers: not one of the cheap per-draw steps PROF_DRAW samples)
    const uint32_t n = (uint32_t)std::clamp(d.max, 1, 4096);
    if (d.max < 1 || d.max > 4096) {   // (relink, issue 109: the draw asked the driver for d.max records where n were converted)
        static std::atomic<int> told{0};
        if (told++ < 10)
            logf("vk: a mesh indirect draw of %d records: %u converted and drawn", d.max, n);
    }
    const uint32_t stride = d.stride > 0 ? (uint32_t)d.stride : 8;
    const BufferRef args = gpu_buffer_bind(d.indirect, (uint64_t)(n - 1) * stride + 4 * (word + 1), false);
    if (!args) return false;
    uint64_t off;
    {
        std::lock_guard lk(mesh_ring_mu);
        if (!mesh_ring) {
            void* unused = nullptr;
            if (!create_buffer(kMeshRing, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false, mesh_ring,
                               mesh_ring_mem, &unused))
                return false;
        }
        const uint64_t bytes = ((uint64_t)n * 12 + 15) & ~15ull;
        if (mesh_ring_at + bytes > kMeshRing) mesh_ring_at = 0;   // (4 MB: frames of draws before it comes round)
        off = mesh_ring_at;
        mesh_ring_at += bytes;
    }
    end_pass(cmd);
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier2(cmd, &di);
    vkCmdFillBuffer(cmd, mesh_ring, off, (uint64_t)n * 12, 1u);
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier2(cmd, &di);
    std::vector<VkBufferCopy> regions(n);
    for (uint32_t i = 0; i < n; i++) regions[i] = {args.offset + (uint64_t)i * stride + 4 * word, off + (uint64_t)i * 12, 4};
    vkCmdCopyBuffer(cmd, args.buffer, mesh_ring, n, regions.data());
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier2(cmd, &di);
    mesh_conv_buf = mesh_ring, mesh_conv_off = off, mesh_conv_n = n;
    return true;
}
} // namespace
// (relink) NVN's indirect mesh draws hold (count, first) and Vulkan's (x, y, z): given as they were, every one drew y = first = 0 workgroups,
// with no error. They are most of the game's vegetation: the forest's trees, ferns and grass (GPU culling writes their counts, so they can only
// be moved on the GPU): each count is copied into this buffer of (count, 1, 1) just before its draw. The GPU runs the copies and draws in
// order, so a slot used again is safe. A first other than 0 has no Vulkan counterpart.
constexpr uint32_t kMeshArgs = 1u << 18;   // (a ring: entries are reused 262144 draws on, frames after the GPU read them)
VkBuffer mesh_args_buffer = VK_NULL_HANDLE;
VmaAllocation mesh_args_alloc = nullptr;
uint32_t* mesh_args_words = nullptr;
uint64_t mesh_args_address = 0;   // (its device address: a task program reads its draw's first workgroup at + 12, + 16 per draw)
// (relink) Indirect mesh draws' arguments, 16 bytes each as Vulkan takes them with the first after: {count, 1, 1, first}. NVN's are
// {count, 0, 0, first}: as they lay, every foliage draw had no workgroups in y and z and drew nothing. Entry 0 stays {0, 1, 1, 0}: the
// first of draws that have none
bool mesh_args_ready() {
    static const bool ok = [] {
        void* mapped = nullptr;
        if (!create_buffer((uint64_t)kMeshArgs * 16,
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true,
                           mesh_args_buffer, mesh_args_alloc, &mapped) || !mapped)
            return false;
        uint32_t* w = (uint32_t*)mapped;
        mesh_args_words = w;
        for (uint32_t i = 0; i < kMeshArgs; i++) w[i * 4] = 0, w[i * 4 + 1] = 1, w[i * 4 + 2] = 1, w[i * 4 + 3] = 0;
        VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        da.buffer = mesh_args_buffer;
        mesh_args_address = vkGetBufferDeviceAddress(dev.device, &da);
        return mesh_args_address != 0;
    }();
    return ok;
}
uint32_t mesh_args_take(uint32_t n) {   // (n entries of the ring, never entry 0)
    static std::atomic<uint32_t> next{1};
    const uint32_t at = next.fetch_add(n) % (kMeshArgs - n);
    return at + 1;
}

/// A vertex stream bound earlier this submission, still the same buffer (a mirror made since, or a new submission, is not)
bool vertex_bind_reuse(uint64_t addr, uint64_t size, uint64_t stamp, uint64_t cached_addr, uint64_t cached_size,
                       uint64_t cached_stamp, int cached) {
    return cached && addr && size && addr == cached_addr && size == cached_size && stamp == cached_stamp;
}
struct KeptVertex {
    uint64_t addr = 0, size = 0, stamp = 0;
    BufferRef ref{};
    int ok = 0;
};
thread_local KeptVertex kept_vb[16];

// ---- (relink, issue 640) a GPU without mesh shaders: the task and mesh stages as compute dispatches and a
// generated vertex stage, drawn through an ordinary graphics pipeline. Records and payloads live in two buffers a
// program reuses draw after draw; the commands are ordered, so each dispatch runs after the last draw that read them.
namespace {
struct FallbackBuffers {
    VkBuffer records = VK_NULL_HANDLE, payload = VK_NULL_HANDLE, args = VK_NULL_HANDLE;
    VmaAllocation records_mem = nullptr, payload_mem = nullptr, args_mem = nullptr;
    uint64_t records_address = 0, payload_address = 0, args_address = 0;
    bool ok = false, tried = false;
};
// (relink, issue 114) One set of buffers for every program and every replay thread, as large as the largest program asked. They were a
// program's own on each replay thread: ~45 MB of records a mesh program (1024 slots of 378 vertices of 29 words), five threads, a
// hundred programs in Kokiri Forest: 8.3 GB of video memory used of 7.2 allowed on an 8 GB card (forced on an RTX 3060 Ti, gx01), the
// textures then held to 471 MB and spilled to system memory, and a GTX 1080 fell from 30 to 1 frame a second. Sharing them is what a
// program's draws already did: each batch clears and writes them between full barriers, which order it after every earlier command
// on the queue, of any thread's command buffer. Outgrown buffers are kept: commands recorded with them may not have run yet
struct FallbackPool {
    std::mutex mu;
    FallbackBuffers cur;
    uint64_t records = 0, payload = 0;   // (cur's sizes)
    uint64_t failed = 0;                 // (records of this size or more could not be made: not tried again at each draw)
    std::vector<FallbackBuffers> old;
    std::atomic<uint64_t> gen{0};        // (cur's generation: a thread's copy is good while it is the same)
};
FallbackPool& fallback_pool() {
    static FallbackPool* pool = new FallbackPool;   // (never freed: the buffers live as long as the device)
    return *pool;
}
thread_local FallbackBuffers fallback_cur;   // (the draw's: a copy, the same for all its batches)
thread_local uint64_t fallback_cur_gen = ~0ull, fallback_cur_records = 0, fallback_cur_payload = 0;
// (relink, issue 640) A draw asks for more workgroups than the records' slots hold (a task draw of up to 444 task workgroups where a
// dispatch holds 32: Kokiri's cliffs and the rocks round the village's entrance were cut): it runs in batches, each the compute
// dispatches and the vertex draw of as many as the slots hold, the program reading its workgroup index plus the batch's first
// (MESH_FALLBACK_BASE_PUSH_OFFSET). An indirect draw's count is the GPU's: its batches' arguments are made there (vk_mesh_split.h),
// as many batches as the count the CPU sees could need, with room to spare, up to kFallbackBatches
constexpr uint32_t kFallbackBatches = 16;
constexpr uint32_t kFallbackDirectBatches = 64;   // (a direct draw's, counted on the CPU: 2048 task workgroups; past them it is cut)
struct FallbackPlan {
    FallbackBuffers* f = nullptr;
    uint32_t batches = 0, batch = 0;
    uint32_t per_batch = 0;   // (task workgroups, or mesh workgroups without a task, a batch runs)
    uint32_t threads = 1;     // (slots a workgroup takes)
    uint32_t groups = 0;      // (a direct draw's workgroups)
    bool indirect = false;
};
thread_local FallbackPlan fallback_plan;

uint64_t fallback_buffer_address(VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(dev.device, &info);
}

bool fallback_buffers_of(const Program& p, FallbackBuffers*& out) {
    const uint64_t capacity = Shader::Backend::SPIRV::MESH_FALLBACK_CAPACITY;
    if (!p.fallback_vertices_per_workgroup || !p.fallback_record_words) return false;
    const uint64_t records = capacity * 4 +
                             capacity * (uint64_t)p.fallback_vertices_per_workgroup * p.fallback_record_words * 4;
    if (records > (512ull << 20)) return false;
    const uint64_t payload = p.fallback_task ? capacity * 4 + capacity * (uint64_t)p.fallback_payload_words * 4 : 0;
    FallbackPool& pool = fallback_pool();
    if (fallback_cur.ok && fallback_cur_gen == pool.gen.load(std::memory_order_acquire) && records <= fallback_cur_records &&
        payload <= fallback_cur_payload) {
        out = &fallback_cur;
        return true;
    }
    std::lock_guard lk(pool.mu);
    if (!pool.cur.ok || records > pool.records || payload > pool.payload) {
        const uint64_t want_r = std::max(records, pool.records), want_p = std::max(payload, pool.payload);
        if (pool.failed && want_r >= pool.failed) return false;
        FallbackBuffers made;
        void* mapped = nullptr;
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bool ok = create_buffer(want_r, usage, false, made.records, made.records_mem, &mapped);
        if (ok) made.records_address = fallback_buffer_address(made.records);
        if (ok && want_p) {
            ok = create_buffer(want_p, usage, false, made.payload, made.payload_mem, &mapped);
            if (ok) made.payload_address = fallback_buffer_address(made.payload);
        }
        if (ok) {   // (an indirect draw's batches' arguments, vk_mesh_split.h)
            const uint64_t args = ((uint64_t)kMeshSplitHeadWords + (uint64_t)kFallbackBatches * kMeshSplitBatchWords) * 4;
            ok = create_buffer(args, usage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false, made.args, made.args_mem, &mapped);
            if (ok) made.args_address = fallback_buffer_address(made.args);
        }
        if (!ok || !made.records_address || !made.args_address) {
            static std::atomic<int> told{0};
            if (told++ < 4) logf("vk: fallback: buffers of %llu MB of records could not be made", (unsigned long long)(want_r >> 20));
            pool.failed = pool.failed ? std::min(pool.failed, want_r) : want_r;
            return false;   // (what was made of them is let go with the device)
        }
        made.ok = true;
        if (pool.cur.ok) pool.old.push_back(pool.cur);
        pool.cur = made, pool.records = want_r, pool.payload = want_p;
        pool.gen.fetch_add(1, std::memory_order_release);
        logf("vk: fallback: records %llu MB, payloads %llu KB, shared by every mesh program (%zu outgrown sets kept)",
             (unsigned long long)(want_r >> 20), (unsigned long long)(want_p >> 10), pool.old.size());
    }
    fallback_cur = pool.cur;
    fallback_cur_gen = pool.gen.load(std::memory_order_acquire);
    fallback_cur_records = pool.records, fallback_cur_payload = pool.payload;
    out = &fallback_cur;
    return true;
}

bool fallback_compute_pipelines(Program& p) {
    static std::mutex mu;
    std::lock_guard lk(mu);
    if (!p.fallback_compute.empty()) return true;
    // (relink, issue 60) A program whose compile failed was compiled again at each of its draws, on the recording thread, the
    // driver's state after a fault caught in it unknown (-77): the GTX 1070's log stopped for good right after the first
    // failure. Its draws are now left out without another try. RX_VK_FALLBACK_RETRY=1: tried again at each draw, as before
    static std::unordered_set<const Program*> failed;
    static const bool retry = getenv("RX_VK_FALLBACK_RETRY") && atoi(getenv("RX_VK_FALLBACK_RETRY")) != 0;
    if (!retry && failed.count(&p)) return false;
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
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = si;
        ci.layout = p.layout;
        {   // (relink, issue 60) RX_VK_NOOPT=1 here too, as for every other pipeline: the test a GTX 1070 can run
            static const bool noopt = getenv("RX_VK_NOOPT") != nullptr;
            if (noopt) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        }
        // (relink, issue 60) These are compiled here, on the recording thread, at the program's first draw. A GTX 1070's log
        // ended on "could not be made" for one of them and then nothing: no present, no other line. Each compile now says it
        // starts, how long it took and what the driver answered, so a log that stops says whether a compile was under way
        static std::atomic<int> told_start{0};
        if (told_start++ < 64) logf("vk: fallback: compiling stage %zu of %016llx", i, (unsigned long long)p.key);
        const auto t0 = std::chrono::steady_clock::now();
        const VkResult r = guarded_compute(pcache, &ci, &made[i]);
        const long long ms = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        if (r != VK_SUCCESS) {
            static std::atomic<int> told{0};
            if (told++ < 8)
                logf("vk: fallback: the compute pipeline of %016llx could not be made (%d, %lld ms%s): its draws are left out",
                     (unsigned long long)p.key, (int)r, ms, r == (VkResult)-77 ? ", a fault in the driver" : "");
            for (VkPipeline& m : made)
                if (m) vkDestroyPipeline(dev.device, m, nullptr), m = VK_NULL_HANDLE;
            failed.insert(&p);
            return false;
        }
        static std::atomic<int> told_slow{0};
        if (ms >= 200 && told_slow++ < 64)
            logf("vk: fallback: stage %zu of %016llx compiled in %lld ms", i, (unsigned long long)p.key, ms);
        pcache_built_since_save++;
    }
    if (!any) return false;
    p.fallback_compute = std::move(made);
    return true;
}

/// (relink, issue 640, RX_VK_LOG_FALLBACK=1) What the fallback's mesh draws ask for against what their batches run: task workgroups
/// (or mesh workgroups without a task) asked and runnable, the draws cut, the indirect draws' entries past the first. An indirect
/// draw's count is read from guest memory as the CPU sees it (the GPU's culling may have written another since), which is also what
/// its batches are counted from. 30 cut draws, then a line every 4000 fallback draws
void fallback_note(const Program& p, const DrawArgs& d, const FallbackPlan& plan, uint32_t hint) {
    static const bool on = getenv("RX_VK_LOG_FALLBACK") && atoi(getenv("RX_VK_LOG_FALLBACK")) != 0;
    if (!on) return;
    uint64_t asked = 0;
    uint32_t entries = 1;
    if (d.indirect) {
        entries = (uint32_t)std::max(d.max, 1);
        const uint32_t stride = d.stride > 0 ? (uint32_t)d.stride : 8;
        asked = hint;
        for (uint32_t i = 1; i < std::min(entries, 64u); i++) {
            uint32_t count = 0;
            host->gpu_read(d.indirect + (uint64_t)i * stride, &count, 4);
            asked += count;
        }
    } else {
        asked = plan.groups;
    }
    const uint64_t run = std::min<uint64_t>(d.indirect ? hint : asked, (uint64_t)plan.batches * plan.per_batch);
    static std::atomic<uint64_t> draws{0}, cut{0}, indirect{0}, multi{0}, asked_all{0}, run_all{0}, most_asked{0}, batches{0};
    const uint64_t n = ++draws;
    indirect += d.indirect != 0, multi += entries > 1, asked_all += asked, run_all += run, batches += plan.batches;
    uint64_t was = most_asked.load();
    while (asked > was && !most_asked.compare_exchange_weak(was, asked)) {}
    if (asked > run) {
        static std::atomic<int> told{0};
        cut++;
        if (told++ < 30)
            logf("vk: fallback: draw of %016llx cut: %s%s, %llu %s workgroups asked in %u entr%s, %llu run (%u batches of %u)",
                 (unsigned long long)p.key, d.indirect ? "indirect" : "direct", p.fallback_task ? ", task" : "", (unsigned long long)asked,
                 p.fallback_task ? "task" : "mesh", entries, entries == 1 ? "y" : "ies", (unsigned long long)run, plan.batches, plan.per_batch);
    }
    if (n % 4000 == 0)
        logf("vk: fallback: %llu mesh draws in %llu batches, %llu cut, %llu indirect (%llu with more than one entry); workgroups asked "
             "%llu, run %llu, largest draw %llu", (unsigned long long)n, (unsigned long long)batches.load(), (unsigned long long)cut.load(),
             (unsigned long long)indirect.load(), (unsigned long long)multi.load(), (unsigned long long)asked_all.load(),
             (unsigned long long)run_all.load(), (unsigned long long)most_asked.load());
}

/// The words of batch b's arguments in an indirect draw's buffer (vk_mesh_split.h), in bytes
VkDeviceSize fallback_args_at(uint32_t b, uint32_t word) {
    return ((VkDeviceSize)kMeshSplitHeadWords + (VkDeviceSize)b * kMeshSplitBatchWords + word) * 4;
}

/// Batch b of the draw fallback_record planned: its dispatches, outside any render pass, the program reading its workgroup index plus
/// the batch's first. The records and payloads are the program's, used again by each batch: the last batch's vertex draw read them
/// before they are cleared
bool fallback_batch(VkCommandBuffer cmd, Program& p, uint32_t b) {
    const FallbackPlan& plan = fallback_plan;
    FallbackBuffers* f = plan.f;
    if (!f || b >= plan.batches) return false;
    fallback_payload_address = f->payload_address;
    fallback_records_address = f->records_address;
    const uint32_t capacity = Shader::Backend::SPIRV::MESH_FALLBACK_CAPACITY;
    end_pass(cmd);
    full_barrier(cmd);
    vkCmdFillBuffer(cmd, f->records, 0, (VkDeviceSize)capacity * 4, 0);   // (the slots this batch runs)
    // (the payloads too: a mesh workgroup the task did not start still runs, and reads its payload's words)
    if (f->payload && p.fallback_task) vkCmdFillBuffer(cmd, f->payload, 0, VK_WHOLE_SIZE, 0);
    full_barrier(cmd);
    if (!bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p)) return false;
    push_constants(cmd, p, nullptr);
    const uint32_t base = b * plan.per_batch;
    vkCmdPushConstants(cmd, p.layout, p.push_stages, Shader::Backend::SPIRV::MESH_FALLBACK_BASE_PUSH_OFFSET, 4, &base);
    size_t stages[2] = {p.stages.size(), p.stages.size()};
    for (size_t i = 0; i < p.stages.size(); i++) {
        if (p.stages[i].vk_stage != VK_SHADER_STAGE_COMPUTE_BIT) continue;
        if (stages[0] == p.stages.size()) stages[0] = i;
        else stages[1] = i;
    }
    const uint32_t n = plan.indirect ? 0 : std::min(plan.groups - std::min(plan.groups, base), plan.per_batch);
    const auto send = [&](size_t stage, uint32_t word, uint32_t count) {
        if (stage >= p.fallback_compute.size() || !p.fallback_compute[stage]) return false;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.fallback_compute[stage]);
        if (plan.indirect) vkCmdDispatchIndirect(cmd, f->args, fallback_args_at(b, word));
        else if (count) vkCmdDispatch(cmd, count, 1, 1);
        return true;
    };
    if (p.fallback_task) {   // (a task workgroup's slots are its index in the batch * threads + thread)
        if (!send(stages[0], kMeshSplitTaskWord, n)) return false;
        full_barrier(cmd);
        if (!send(stages[1], kMeshSplitMeshWord, n * plan.threads)) return false;   // (a slot each; the task's flags cull those it did not start)
    } else if (!send(stages[0], kMeshSplitMeshWord, n)) {
        return false;
    }
    full_barrier(cmd);
    return true;
}

/// Batch b's vertex draw, in the open pass: one vertex a corner of each primitive its slots may hold
void fallback_draw(VkCommandBuffer cmd, const Program& p, uint32_t b) {
    const FallbackPlan& plan = fallback_plan;
    if (plan.indirect) {
        vkCmdDrawIndirect(cmd, plan.f->args, fallback_args_at(b, kMeshSplitDrawWord), 1, 16);
        return;
    }
    const uint32_t base = b * plan.per_batch;
    const uint32_t n = std::min(plan.groups - std::min(plan.groups, base), plan.per_batch);
    const uint64_t vertices = (uint64_t)n * plan.threads * p.fallback_vertices_per_workgroup;
    if (vertices) vkCmdDraw(cmd, (uint32_t)std::min<uint64_t>(vertices, 0xFFFFFFFFu), 1, 0, 0);
}

/// A mesh draw's plan (its batches; an indirect draw's arguments made on the GPU from its count), and its first batch's dispatches
bool fallback_record(VkCommandBuffer cmd, Program& p, const DrawArgs& d) {
    fallback_plan = {};
    if (d.count_addr) {   // (more than one draw in the records: each its own dispatch, not one)
        static std::atomic<int> told{0};
        if (told++ < 2) logf("vk: fallback: a mesh draw with a count buffer is left out (issue 640)");
        return false;
    }
    FallbackBuffers* f = nullptr;
    if (!fallback_buffers_of(p, f) || !fallback_compute_pipelines(p)) return false;
    const uint32_t capacity = Shader::Backend::SPIRV::MESH_FALLBACK_CAPACITY;
    FallbackPlan plan;
    plan.f = f;
    plan.threads = p.fallback_task ? std::max(1u, p.fallback_task_threads) : 1u;
    plan.per_batch = std::max(1u, capacity / plan.threads);
    plan.indirect = d.indirect != 0;
    uint32_t hint = 0;
    if (plan.indirect) {
        VkPipeline split = VK_NULL_HANDLE;
        VkPipelineLayout split_layout = VK_NULL_HANDLE;
        if (!mesh_split_pipeline(&split, &split_layout)) return false;
        const BufferRef src = gpu_buffer_bind(d.indirect, 4, false);
        if (!src) return false;
        host->gpu_read(d.indirect, &hint, 4);   // (the count as the CPU sees it, a frame or so old: a quarter more, and a batch)
        const uint64_t want = (uint64_t)hint + hint / 4 + plan.per_batch;
        plan.batches = (uint32_t)std::clamp<uint64_t>((want + plan.per_batch - 1) / plan.per_batch, 1, kFallbackBatches);
        end_pass(cmd);
        full_barrier(cmd);   // (the count's writer before the copy, and the last draw's reads of the arguments)
        const VkBufferCopy count{src.offset, 0, 4};
        vkCmdCopyBuffer(cmd, src.buffer, f->args, 1, &count);
        full_barrier(cmd);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, split);
        const MeshSplitPush push{f->args_address, plan.per_batch, plan.threads, p.fallback_vertices_per_workgroup, plan.batches};
        vkCmdPushConstants(cmd, split_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
        vkCmdDispatch(cmd, (plan.batches + 63) / 64, 1, 1);
        full_barrier(cmd);
    } else {
        plan.groups = (uint32_t)std::max(d.groups[0], 0);
        plan.batches = std::min((plan.groups + plan.per_batch - 1) / plan.per_batch, kFallbackDirectBatches);
    }
    fallback_note(p, d, plan, hint);
    fallback_plan = plan;
    if (!plan.batches) return true;   // (no workgroup: nothing to draw)
    return fallback_batch(cmd, p, 0);
}
} // namespace

/// The vertex streams already bound on this command buffer (a new recording, or a mirror made since, is not)
bool vertex_set_same(uint64_t stamp, const uint64_t* addr, const uint64_t* size, const uint32_t* stride,
                     uint64_t cached_stamp, const uint64_t* cached_addr, const uint64_t* cached_size,
                     const uint32_t* cached_stride, int on) {
    if (!on || stamp != cached_stamp) return false;
    for (int i = 0; i < 16; i++) {
        const bool used = i != 15 && addr[i];
        if (!used) {
            if (cached_addr[i] || cached_stride[i]) return false;
            continue;
        }
        const uint64_t sz = size[i] ? size[i] : 1;
        if (cached_addr[i] != addr[i] || cached_size[i] != sz || cached_stride[i] != stride[i]) return false;
    }
    return true;
}
/// The index buffer already bound on this command buffer (a new recording is not)
bool index_bind_same(uint64_t buffer, uint64_t offset, uint32_t type, uint64_t cached_buffer, uint64_t cached_offset,
                     uint32_t cached_type, int on) {
    return on && buffer && buffer == cached_buffer && offset == cached_offset && type == cached_type;
}
void bind_index(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
    const uint64_t b = (uint64_t)(uintptr_t)buffer;
    if (index_bind_same(b, (uint64_t)offset, (uint32_t)type, last_ib.buffer, last_ib.offset, last_ib.type, last_ib.on))
        return;
    vkCmdBindIndexBuffer(cmd, buffer, offset, type);
    last_ib = {b, (uint64_t)offset, (uint32_t)type, true};
}

/// (relink, a diagnostic for NVIDIA DLSS's motion: vk_dlss.cpp) RX_VK_DLSS_TRACE=<frames>: every that many frames, one frame's
/// draws into targets at least 640 wide, logged: the program, its targets' formats, the geometry and the uniform buffers
static void dlss_trace_draw(const NvnState& s, const Program& p, const DrawArgs& d, const Targets& at) {
    static const int every = getenv("RX_VK_DLSS_TRACE") ? atoi(getenv("RX_VK_DLSS_TRACE")) : 0;
    if (every <= 0 || at.width < 640) return;
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

/// (relink) A skinned draw into the scene, for NVIDIA DLSS's actor motion (vk_dlss.cpp). OoT's characters and the things that
/// move by themselves are indexed draws of one vertex stream: the position at attribute 0 (RGB32F or RGBA16F), the weights at 4
/// (and 5, RGBA8), the joints at 6 (and 7, RGBA16UI or RGBA8UI), the bone palette in the vertex stage's uniform buffer 6 (NVN's
/// binding 3: 3 x 4 matrices, a whole number of them) and the object's origin in its buffer 4 (binding 1, at 0x40)
static void dlss_skin_check(const NvnState& s, const Program& p, const DrawArgs& d, const Targets& at) {
    if (d.mesh || !d.indexed || d.indirect || d.instances != 1 || !at.n || !at.color[0] || !at.depth || at.width < 640 ||
        at.color_format[0] != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || s.nvattr < 7 || !dlss_wanted())
        return;
    const auto attr = [&](int i) { return *reinterpret_cast<const nvn_va_st*>(&s.vattr[i]); };
    const nvn_va_st pos = attr(0);
    if ((pos.fmt != NVN_FORMAT_RGB32F && pos.fmt != NVN_FORMAT_RGBA16F) || pos.stream != 0) return;
    const nvn_p_bind_buf& bones = s.ubo[0][3];
    if (!bones.addr || bones.size < 48 || bones.size % 48) return;
    DlssSkin k;
    k.formats = pos.fmt == NVN_FORMAT_RGBA16F ? dlss_actor::kHalfPosition : 0;
    for (int set = 0; set < 2 && 6 + set < s.nvattr; set++) {
        const nvn_va_st w = attr(4 + set), j = attr(6 + set);
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
    dlss_note_skinned_draw(k);
}

// (relink) RX_VK_PEEK_PROG=<key prefix>: the storage buffers a program binds as the GPU holds them when it runs (what it reads
// may never reach the game's memory: the GPU culling's windows are for the GPU alone): before each of its draws, the first 256 bytes
// of each copied to a host-visible buffer, logged a few frames later; 96 buffers in all
// (RX_VK_PEEK_AT=<address>,...: only the buffers holding one of these, 256 bytes from it; RX_VK_PEEK_FOLLOW=<key prefix>:<cbuf>:<offset>
// adds the address that program binds there, each new one, 8 at most)
std::vector<uint64_t>& peek_at() {
    static std::vector<uint64_t> at = [] {
        std::vector<uint64_t> v;
        for (const char* q = getenv("RX_VK_PEEK_AT"); q && *q;) {
            v.push_back(strtoull(q, nullptr, 0));
            while (*q && *q != ',') q++;
            if (*q == ',') q++;
        }
        return v;
    }();
    return at;
}
bool peek_follows() {
    static const bool on = getenv("RX_VK_PEEK_FOLLOW") != nullptr;
    return on;
}
void peek_follow(uint64_t key) {
    static const char* f = getenv("RX_VK_PEEK_FOLLOW");
    if (!f) return;
    char want[32] = {};
    unsigned cbuf = 0, off = 0;
    if (sscanf(f, "%31[0-9a-f]:%u:%i", want, &cbuf, &off) != 3) return;
    char hex[24];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)key);
    if (strncmp(hex, want, strlen(want))) return;
    std::lock_guard lk(peek_mu);
    for (const PeekBind& b : peek_binds)
        if (b.cbuf == cbuf && b.off == (uint32_t)off && b.indirect == ~0U && b.addr &&
            std::find(peek_at().begin(), peek_at().end(), b.addr) == peek_at().end() && peek_at().size() < 8) {
            peek_at().push_back(b.addr);
            logf("vk: peek: following %#llx (%016llx c%u+%#x)", (unsigned long long)b.addr, (unsigned long long)key, cbuf, (unsigned)off);
        }
}
void peek_draw(VkCommandBuffer cmd, uint64_t key, bool in_pass = true) {
    static std::atomic<int> left{getenv("RX_VK_PEEK_MAX") ? atoi(getenv("RX_VK_PEEK_MAX")) : 160};
    peek_log_ready();
    peek_follow(key);
    if (left.load() <= 0 || peek_binds.empty()) return;
    if (peek_follows() && peek_at().empty()) return;
    static const bool ok = [] {
        void* mapped = nullptr;
        if (!create_buffer((uint64_t)kPeekSlots * kPeekBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, peek_buffer, peek_alloc, &mapped) || !mapped)
            return false;
        peek_mapped = (const uint8_t*)mapped;
        return true;
    }();
    if (!ok) return;
    if (in_pass) end_pass(cmd);
    full_barrier(cmd);
    {
        std::lock_guard lk(peek_mu);
        std::vector<uint64_t> done;   // (each address once a call: table entries' windows overlap)
        for (const PeekBind& b : peek_binds) {
            uint64_t from = b.base;
            if (!peek_at().empty()) {
                from = 0;
                for (uint64_t a : peek_at())
                    if (a >= b.base && a < b.base + b.range && std::find(done.begin(), done.end(), a) == done.end()) from = a;
                if (!from) continue;
                done.push_back(from);
            }
            if (left.fetch_sub(1) <= 0) break;
            const uint32_t slot = peek_next++ % kPeekSlots;
            const VkDeviceSize at = b.offset + (from - b.base);
            const VkBufferCopy region{at, (VkDeviceSize)slot * kPeekBytes, std::min<VkDeviceSize>(b.base + b.range - from, kPeekBytes)};
            vkCmdCopyBuffer(cmd, b.buffer, peek_buffer, 1, &region);
            PeekBind shown = b;
            shown.addr = from;
            peek_pending.push_back({key, texture_frame_no(), slot, shown});
        }
    }
    full_barrier(cmd);
}

// (relink) NVN's indirect mesh draws hold (count, first) and Vulkan's (x, y, z): given as they were, every one drew y = first = 0 workgroups,
// with no error. They are most of the game's vegetation: the forest's trees, ferns and grass (GPU culling writes their counts, so they can only
// be moved on the GPU): each count is copied into this buffer of (count, 1, 1) just before its draw. The GPU runs the copies and draws in
// order, so a slot used again is safe. A first other than 0 has no Vulkan counterpart.
// (relink) The copies' barriers, no wider than they need: before, the shaders' writes (the culling's counts) for the copy;
// after, the copy for the draw's indirect read. Full barriers there (ALL_COMMANDS both sides) waited for all the pass's work at
// each of the ~40 indirect mesh draws a frame: the terrace's repeated frames went 0.15 -> 0.83 a second (hs295).
// RX_VK_MESH_ARGS_FULL=1: full barriers, as before
void mesh_args_barrier(VkCommandBuffer cmd, bool before) {
    static const bool full = getenv("RX_VK_MESH_ARGS_FULL") != nullptr;
    if (full) return full_barrier(cmd);
    VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    if (before) {
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                         VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                         VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;   // (and a slot's earlier indirect read before it is written again)
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    } else {
        b.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        b.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    }
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.memoryBarrierCount = 1;
    d.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &d);
}
/// (relink) RX_VK_LOG_GLOBAL=1: each program whose translation kept global memory accesses (read through the run-time search
/// of the bound storage buffers: a miss reads 0 and an atomic is lost), once, with its grid or targets
void global_note(const NvnState& s, const Program& p, const Targets* at) {
    static const bool on = getenv("RX_VK_LOG_GLOBAL") != nullptr;
    if (!on) return;
    bool any = false;
    for (const Stage& st : p.stages) any |= st.info.uses_global_memory;
    if (!any) return;
    static std::mutex mu;
    static std::unordered_set<uint64_t> seen;
    {
        std::lock_guard lk(mu);
        if (seen.size() > 400 || !seen.insert(p.key).second) return;
    }
    std::string st_text;
    for (const Stage& st : p.stages) {
        char b[160];
        snprintf(b, sizeof b, " [stage %d: global %d, stores %d, tables %u, ssbos %zu, atomics f32add %d]", st.nvn_stage, (int)st.info.uses_global_memory,
                 (int)st.info.stores_global_memory, st.info.num_global_tables, st.info.storage_buffers_descriptors.size(), (int)st.info.uses_atomic_f32_add);
        st_text += b;
    }
    logf("vk: global memory: program %016llx %s %ux%ux%u frame %llu:%s", (unsigned long long)p.key, p.compute ? "compute" : "graphics",
         at ? at->width : s.grid[0], at ? at->height : s.grid[1], at ? (uint32_t)at->n : s.grid[2], (unsigned long long)texture_frame_no(), st_text.c_str());
}

/// (relink) RX_VK_TEX_OF: a watched program's textures (and its colour targets, or the storage images a dispatch writes)
void tex_of_note(const NvnState& s, const Program& p, const Targets* at, const NvnState* compute) {
    static std::mutex mu;
    static std::unordered_map<uint64_t, int> told;
    static const std::vector<uint64_t> keys = [] {
        std::vector<uint64_t> v;
        for (const char* q = getenv("RX_VK_TEX_OF"); q && *q;) {
            char* end;
            const uint64_t k = strtoull(q, &end, 16);
            if (end == q) break;
            v.push_back(k);
            q = *end ? end + 1 : end;
        }
        return v;
    }();
    static const bool depth_or_gpu = getenv("RX_VK_TEX_DEPTH") || getenv("RX_VK_TEX_GPU");
    if (!depth_or_gpu && std::find(keys.begin(), keys.end(), p.key) == keys.end()) return;   // (before any work: a check at every draw)
    // (each sampled texture with its GPU address: "gpu w x h x d nvn-format")
    std::string list;
    bool has_depth = false, has_gpu = false;
    static const uint32_t depth_want = getenv("RX_VK_TEX_DEPTH") ? (uint32_t)atoi(getenv("RX_VK_TEX_DEPTH")) : 0;   // (any program sampling a texture that deep)
    static const uint64_t gpu_want = getenv("RX_VK_TEX_GPU") ? strtoull(getenv("RX_VK_TEX_GPU"), nullptr, 16) : 0;   // (any program sampling that storage)
    for (const Stage& st : p.stages)
        for (const auto& d : st.info.texture_descriptors)
            for (uint32_t k = 0; k < std::min(d.count, 4u); k++) {
                const uint32_t h = handle_of(s, st, d, k);
                nvn_tex t;
                nvn_view v;
                bool hv = false;
                if (!h || !texture_of((int)(h & 0xFFFFF), &t, &v, &hv)) continue;
                char b[120];
                snprintf(b, sizeof b, " s%d %#llx %ux%ux%u f%#x;", st.nvn_stage, (unsigned long long)t.gpu, t.d.width, t.d.height, t.d.depth,
                         (unsigned)t.d.format);
                list += b;
                has_depth |= depth_want && t.d.depth == depth_want;
                has_gpu |= gpu_want && t.gpu == gpu_want;
            }
    if (std::find(keys.begin(), keys.end(), p.key) == keys.end()) {
        if (!has_depth && !has_gpu) return;
        std::lock_guard lk(mu);
        if (told[p.key] >= 3) return;
    }
    {
        std::lock_guard lk(mu);
        if (told[p.key]++ >= 12) return;
    }
    char tgt[200] = "";
    if (at && at->color[0])
        snprintf(tgt, sizeof tgt, " -> colour %#llx %ux%ux%u x%u layer %u fmt %d", (unsigned long long)at->color[0]->gpu, at->color[0]->width,
                 at->color[0]->height, at->color[0]->depth, at->color[0]->layers, at->color_layer[0], (int)at->color_format[0]);
    std::string imgs;
    if (compute)
        for (int i = 0; i < 8; i++) {
            const uint32_t h = (uint32_t)compute->img[5][i];
            nvn_tex t;
            nvn_view v;
            bool hv = false;
            if (h && texture_of((int)(h & 0xFFFFF), &t, &v, &hv)) {
                char b[120];
                snprintf(b, sizeof b, " img%d %#llx %ux%ux%u fmt %#x;", i, (unsigned long long)t.gpu, t.d.width, t.d.height, t.d.depth, (unsigned)t.d.format);
                imgs += b;
            }
        }
    std::string sb;   // (the storage buffers it binds: where, how big, written)
    for (const Stage& st : p.stages)
        for (const auto& d : st.info.storage_buffers_descriptors)
            for (uint32_t k = 0; k < std::min(d.count, 2u); k++) {
                const uint32_t off = d.cbuf_offset + k * 16;
                const uint64_t addr = stage_cbuf_word(s, st, d.cbuf_index, off) | (uint64_t)stage_cbuf_word(s, st, d.cbuf_index, off + 4) << 32;
                const uint32_t size = stage_cbuf_word(s, st, d.cbuf_index, off + 8);
                char b[120];
                snprintf(b, sizeof b, " ssbo c%u+%#x%s %#llx +%#x;", (unsigned)d.cbuf_index, off, d.is_written ? " w" : "", (unsigned long long)addr, size);
                sb += b;
            }
    logf("vk: tex-of %016llx frame %llu:%s%s%s%s", (unsigned long long)p.key, (unsigned long long)texture_frame_no(), list.c_str(), tgt, imgs.c_str(),
         sb.c_str());
}

void draw(VkCommandBuffer cmd, NvnState& s, const DrawArgs& d) {
    FineScope fine_prep(FT_PREP);
    FineScope fine_targets(FT_TARGETS);
    Pass& ps = pass();   // One TLS accessor for this recording thread's pass state.
    PROF_HOT(PROF_DRAW);
    FINE_N(FC_DRAWS, 1);
    stats_tick();
    // (EvilOOT: before the pass opens; sets mesh_conv_buf for the mesh draw below. relink, issue 640: not with the no-mesh fallback,
    // whose batches read the count themselves: it was a pass ended, three barriers and a copy for nothing at each indirect draw)
    if (!dev.mesh_fallback) mesh_indirect_convert(cmd, d);
    handle_memo_next();
    VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    if (!d.mesh && !topology(d.prim, topo)) {
        static int logged[32];
        if (d.prim >= 0 && d.prim < 32 && logged[d.prim]++ < 2) logf("vk: draws of NVN primitive %d are left out", d.prim);
        SKIP_HERE();
    }
    const Targets at = bare_targets(s, targets(s));
    if (!at.width || !at.height) SKIP_HERE();
    // (relink) the scene's camera and depth for NVIDIA DLSS (vk_dlss.cpp): the vertex stage's uniform buffer 3 (NVN's binding 0)
    if (at.n && at.color[0] && at.depth)
        dlss_note_scene_draw(at.color[0], at.width, at.height, at.depth, s.ubo[0][0].addr, s.ubo[0][0].size);
    uint8_t types[8];
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
    const Program* cp;
    fine_targets.finish();
    {
        FINE_SCOPE(FT_PROGRAM_PICK);
        PROF_HOT(PROF_PROGRAM);
        ParLock lk(d.mesh);   // (graphics programs: each thread's picks, program_pick's own lock)
        program_load_async_here(true);   // (relink: a cached program's read goes to the workers: vk_shader.cpp)
        cp = program_for(s, d.mesh ? ProgramKind::Mesh : ProgramKind::Graphics, types);
        program_load_async_here(false);
    }
    {   // (relink, a test) RX_VK_NO_MESH=1: task / mesh draws left out, to see what they draw; =2: the others left out
        static const int no_mesh = getenv("RX_VK_NO_MESH") ? atoi(getenv("RX_VK_NO_MESH")) : 0;
        if ((no_mesh == 1 && d.mesh) || (no_mesh == 2 && !d.mesh)) return;
    }
    if (!cp) {   // (relink) which draw has no program, 12 times: its stages' native code keys
        static std::atomic<int> told{0};
        if (told++ < 12) logf("vk: a draw without a program (frame %llu); its code:%s", (unsigned long long)texture_frame_no(),
                              program_code_keys(s).c_str());
        SKIP_HERE();
    }
    if (skip_live(cp->key)) return;   // (EvilOOT, RX_VK_SKIP_FILE)
    {   // (relink, RX_VK_SKIP_PROG=<key prefix,...>: draws too, their native code keys logged once; =-: none)
        // (OoT's 53509044401d78d2, by default only while its targets have no view: 128 draws, once
        // while an area loads, that render the slices of two 64^3 BC4 noise volumes through raw RGBA16UI views. Those views were
        // not made (no attachment usage) and the pass ran over no target at the texels' 64 x 64: on about half the first starts
        // the device was lost (7 of 14), and it was left out always. Left out, the volumes stayed zero and the foliage's LOD
        // crossfade, which is dithered by that noise, cut hard: trees bare at mid distance. Where the driver takes the usage
        // (vk_texture.cpp) the views are made and the pass is sized in blocks (targets()); it runs there, and is still left out
        // where they are not. (relink, issue 21) Those raw views lost the device on an RTX 4070 at every new game: plain images
        // stand in for the volumes' slices now, copied into them after each pass (vk_texture.cpp stand_in_view), so the views
        // are made on every driver and the pass runs; RX_VK_RAW_TARGET_VIEWS=1: the raw views as before)
        static const bool default_list = !getenv("RX_VK_SKIP_PROG");
        if (default_list && cp->key == 0x53509044401d78d2ull && !at.n) {
            static std::atomic<int> told{0};
            if (told++ < 2)
                logf("vk: draws of program %016llx left out: its targets have no view (RX_VK_SKIP_PROG=- runs them anyway)",
                     (unsigned long long)cp->key);
            return;
        }
        // (relink, issues 6 and 7, 2026-11-14) OoT's 2b37a33d189014a8 and 0948a49a3c3068a8: two small box draws (24 indices) into a
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
        // (relink, issues 6, 7 and 27, a diagnosis) RX_VK_SKIP_3D_DRAWS=1: every draw whose program samples a 3D texture left out.
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
        // (relink, issues 6, 7 and 27, a workaround) RX_VK_SKIP_HALF_PASS=1: the draws of the game's half-resolution HDR pass that
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
    {   // (relink, RX_VK_SKIP_CODE=<code key prefix,...>: the draws of every program with a stage of that native code left out,
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
    // Not ready (translating, or its pipeline building): left out of the frame when its targets hold an earlier
    // frame's picture already, which shows for a frame without it; waited for otherwise (a pass drawn once, a target
    // made this frame), as totk-pc-recomp's TOTK_PIPE_SKIP rule. RX_VK_PIPE_WAIT=1: always waited for.
    // (waited for by default: a draw left out showed as missing geometry in races, where many programs are new;
    // RX_VK_PIPE_SKIP=1: left out when its targets hold an earlier frame's picture, as before)
    static const bool always_wait = getenv("RX_VK_PIPE_SKIP") == nullptr;
    // (relink, issue 47: and only where the next frame draws it again, targets_redrawn. RX_VK_SKIP_PARTS=1: as before, any draw
    // into targets that hold an earlier frame's picture)
    static const bool skip_parts = getenv("RX_VK_SKIP_PARTS") && atoi(getenv("RX_VK_SKIP_PARTS")) != 0;
    const bool persistent = targets_persistent(at) && (skip_parts || targets_redrawn(at, s));
    const bool may_skip = !always_wait && persistent;
    // (relink) a bounded wait (RX_VK_PIPE_WAIT_MS, default 8 a frame, 0: as before, unbounded): a draw into targets that already hold
    // an earlier frame's picture waits for its program or pipeline while this frame's waits stay under it, and is left out after
    // that (drawn from the next frame its pipeline is ready in, 1-3 frames later). Waiting for each in turn was the player's
    // freezes of 0.3-4.7 s in places never seen before (231 programs and 460 pipelines met in 16 minutes, each ~86 ms, built one
    // after the other as the draws asked). Targets drawn first this frame (a pass drawn once: a table baked at load) still wait
    // (night 10/11, 01:20: 0 by default, the draw waits and the picture is exact: left out, a pass missing for a frame showed as a
    // white picture at first views. 04:40: 8 again, for the scene's geometry only. Waiting for each program in turn is one build at
    // a time: the first visit of Jabu-Jabu met ~40 new programs, ~1 s each (translation, then the driver's build), and the game
    // stood at 0.1-0.7 frames a second for 40 s; the Forest, Water and Shadow temples froze 8 to 16 s the same way (hs491). Left
    // out past the frame's 8 ms, every program the frame needs is asked for at once and the six workers build them side by side;
    // the draws come in as theirs are ready. A pass with no depth target (post-processing, the 2D of the HUD, blits: the ones whose
    // absence was a white or black picture) still waits without bound: RX_VK_PIPE_WAIT_FLAT=1 bounds those too, as before 01:20)
    // (relink, 19/11, issues 29 and 54: the stutter while the camera turns. 2 ms, not 8. The graphics pipelines are made at their
    // first draw in every run (lean pipelines), and turning the camera meets new ones every frame: the frame spent its whole 8 ms
    // on them each time, and left the draws out after all. MEASURED on an RX 6600 XT (Windows, fast sync on), the Kokiri Forest
    // save at 67 % with AMD FSR, the camera turned 3 s into a view not seen yet then 3 s back over it, frame times from
    // HOS_PRESENT_LOG (5 runs at 8 ms, 4 at 2): time lost in frames over 25 ms into the new view 1057 -> 859 ms (its rate 27-38
    // -> 39-41 fps), back over the view seen its frames over 25 ms 36 -> 18.8 (408 -> 352 ms lost, one run of the four with a
    // 300 ms texture lookup right after a defragmentation pass, as one of the reference's); standing still unchanged (31 / 28 ms);
    // the waits 4.6-5.6 -> 1.6-1.7 ms a frame; the exposure's steps (RX_VK_EXPOSURE_LOG=2) 0 in every run either way. The draws
    // left out are the ones 8 left out (a 128x128 pass of two targets nearly all; the 1024x576 scene twice in three runs), now
    // for a frame or two longer)
    static const int wait_ms = getenv("RX_VK_PIPE_WAIT_MS") ? std::max(0, atoi(getenv("RX_VK_PIPE_WAIT_MS"))) : 2;
    static const bool wait_flat = getenv("RX_VK_PIPE_WAIT_FLAT") != nullptr;
    const bool bounded = wait_ms > 0 && persistent && (at.depth || wait_flat);
    static thread_local uint64_t wait_frame = ~0ull;
    static thread_local double waited_ms = 0;
    if (texture_frame_no() != wait_frame) wait_frame = texture_frame_no(), waited_ms = 0;
    bool left_out = false;
    auto wait_for = [&](const std::atomic<bool>& ready) {
        PROF_EXACT();
        const auto t0 = std::chrono::steady_clock::now();
        if (bounded) {
            const double left = wait_ms - waited_ms;
            if (left <= 0 || !jobs_wait_until(ready, t0 + std::chrono::microseconds((int64_t)(left * 1000)))) left_out = true;
        } else {
            jobs_wait(ready);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        st_waits++;
        st_wait_ms += ms, waited_ms += ms;
        // (relink, issue 47) the draws left out and those the rule above waited for instead, named: 40 lines of each
        static std::atomic<int> told_out{0}, told_kept{0};
        // (waited for here, left out after the frame's wait under the rule before: a part of a kept target, with depth)
        const bool kept_part = !bounded && !skip_parts && wait_ms > 0 && (at.depth || wait_flat) && targets_persistent(at);
        if ((left_out && told_out.load(std::memory_order_relaxed) < 40) || (kept_part && told_kept.load(std::memory_order_relaxed) < 40)) {
            (left_out ? told_out : told_kept)++;
            const Texture* t0t = at.n ? at.color[0] : at.depth;
            logf("vk: draw of program %016llx %s (%.1f ms waited): %d targets, the first #%u %ux%u NVN %#x drawn at frame %llu (now %llu), "
                 "pass %ux%u, viewport %.0f %.0f %.0f %.0f, scissor %d %d %d %d",
                 (unsigned long long)cp->key, left_out ? "left out (not ready)" : "waited for: a part of a target kept from earlier frames",
                 ms, at.n + (at.depth ? 1 : 0), t0t ? t0t->id : 0u, t0t ? t0t->width : 0u, t0t ? t0t->height : 0u,
                 t0t ? (unsigned)t0t->nvn_format : 0u, (unsigned long long)(t0t ? t0t->last_written : 0),
                 (unsigned long long)texture_frame_no(), at.width, at.height, s.viewport[0][0], s.viewport[0][1], s.viewport[0][2],
                 s.viewport[0][3], s.scissor[0], s.scissor[1], s.scissor[2], s.scissor[3]);
        }
    };
    if (!cp->ready.load(std::memory_order_acquire)) {
        if (may_skip) return void(st_pending_skips++);
        wait_for(cp->ready);
        if (left_out) return void(st_pending_skips++);
    }
    // (relink) a cached program's stand-in, its records now read (vk_shader.cpp): the pick again finds the variant read, or
    // starts its translation as before
    if (cp->failed && program_is_stand_in(cp)) {
        {
            ParLock lk(d.mesh);
            cp = program_for(s, d.mesh ? ProgramKind::Mesh : ProgramKind::Graphics, types);
        }
        if (!cp) SKIP_HERE();
        if (!cp->ready.load(std::memory_order_acquire)) {
            if (may_skip) return void(st_pending_skips++);
            wait_for(cp->ready);
            if (left_out) return void(st_pending_skips++);
        }
    }
    if (exposure_log_on() && cp) exposure_note(s, *cp);
    {   // (relink) RX_VK_TEX_OF=<program key,...>: what those programs' draws sample and draw into, 12 lines each at most
        static const bool want = getenv("RX_VK_TEX_OF") || getenv("RX_VK_TEX_DEPTH") || getenv("RX_VK_TEX_GPU");
        if (want && cp && !cp->failed) tex_of_note(s, *cp, &at, nullptr);
        if (cp && !cp->failed) global_note(s, *cp, &at);
    }
    if (cp->failed) SKIP_HERE();
    Program& p = const_cast<Program&>(*cp);
    if (p.fallback) topo = p.fallback_topology;   // (the mesh program's points, lines or triangles)
    PipeEntry* pe;
    {
        FINE_SCOPE(FT_PIPE_LOOKUP);
        PROF_HOT(PROF_PIPE);
        if (!p.layout) {
            PROF_EXACT();
            ParLock lk;
            ensure_layout(p);
        }
        pe = graphics_pipeline(p, s, at, topo, types);
    }
    if (!pe->ready.load(std::memory_order_acquire)) {
        if (may_skip) return void(st_pending_skips++);
        wait_for(pe->ready);
        if (left_out) return void(st_pending_skips++);
    }
    const VkPipeline pl = pe->pipeline;
    if (!pl) SKIP_HERE();
    cur_program = &p;
    dlss_trace_draw(s, p, d, at);
    dlss_skin_check(s, p, d, at);
    if (!record_progs().empty() && std::find(record_progs().begin(), record_progs().end(), p.key) != record_progs().end())
        record_draw(s, p, d, at);
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
    const uint64_t reg_gen = registrations_generation(), ub_gen = ub_generation();
    static const bool no_reuse = getenv("RX_VK_NO_DESC_REUSE") != nullptr;   // (A/B: every draw builds its own)
    const bool same_desc = !no_reuse && last_desc_valid && last_desc_prog == &p && last_desc_bind == s.bind_gen &&
                           last_desc_reg == reg_gen && last_desc_ub == ub_gen;
    fine_prep.finish();
    if (!same_desc) {
        FINE_SCOPE(FT_DESC_TOTAL);
        PROF_HOT(PROF_DESC);
        FINE_N(FC_BUILDS, 1);
        trace_tex_hit = nullptr, trace_tex_mask = 0;
        ui_draw_now = !at.depth && at.n == 1 && at.width >= 1280;   // (relink: a full-size 2D pass's draw, see ui_draw_now)
        texture_write_targets(at.color, at.n, at.depth);   // (relink: a target among the draw's textures: vk_texture.cpp texture_touch)
        build_descriptors(s, p);   // (textures first used here are uploaded by the recording's upload command buffer)
        texture_write_targets(nullptr, 0, nullptr);
        ui_draw_now = false;
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
    if (!same_desc) gfx_sampled() = draw_sampled(), light_probe_draw = light_probe_now;
    static const bool barrier_every = getenv("RX_VK_BARRIER_EVERY_DRAW") != nullptr;   // (an experiment: a pass each)
    if (barrier_every) end_pass(cmd);
    uint16_t writes;
    {
        FINE_SCOPE(FT_HAZARD);
        writes = draw_hazards(cmd, p, s, at);
    }
    FineScope fine_prepass(FT_CMD_PREPASS);
    static const bool log_self = getenv("RX_VK_LOG_SELF_SAMPLE") != nullptr;   // (relink, diagnosis: a draw sampling its own target)
    if (log_self && log_img_wanted(p.key)) {   // (and RX_VK_LOG_IMG's programs: their targets, at each new pairing)
        static std::mutex mu2;
        static std::unordered_set<uint64_t> seen2;
        std::lock_guard lk(mu2);
        uint64_t sig = mix(p.key, at.depth ? at.depth->id : 0);
        for (int i = 0; i < at.n; i++) sig = mix(sig, at.color[i] ? at.color[i]->id : 0);
        sig = mix(sig, (uint64_t)s.ds.dt << 1 | s.ds.dw);
        if (seen2.size() < 400 && seen2.insert(sig).second) {
            char line[300];
            int k = snprintf(line, sizeof line, "vk: targets frame %llu prog %016llx pass draw %d: depth #%d dt %d dw %d, colour",
                             (unsigned long long)texture_frame_no(), (unsigned long long)p.key, ps.draws,
                             at.depth ? (int)at.depth->id : -1, (int)s.ds.dt, (int)s.ds.dw);
            for (int i = 0; i < at.n && k < 250; i++)
                k += snprintf(line + k, sizeof line - k, " #%d", at.color[i] ? (int)at.color[i]->id : -1);
            k += snprintf(line + k, sizeof line - k, "; pass first prog %016llx", (unsigned long long)ps.first_prog);
            logf("%s", line);
        }
    }
    if (log_self) {
        for (const Texture* t : gfx_sampled()) {
            int slot = -1;
            for (int i = 0; i < at.n; i++)
                if (at.color[i] == t) slot = i;
            if (at.depth == t) slot = 8;
            if (slot < 0) continue;
            static std::mutex mu;
            static std::unordered_set<uint64_t> seen;
            std::lock_guard lk(mu);
            const uint64_t sig = mix(p.key, (uint64_t)slot << 8 | (writes >> slot & 1));
            if (seen.size() < 200 && seen.insert(sig).second)
                logf("vk: self-sample frame %llu prog %016llx samples its own %s %u (#%u %ux%u fmt %d), %s it; depth test %d write %d",
                     (unsigned long long)texture_frame_no(), (unsigned long long)p.key, slot == 8 ? "depth target" : "colour target",
                     slot == 8 ? 0u : (unsigned)slot, t->id, t->width, t->height, (int)t->format,
                     (writes >> slot & 1) ? "WRITES" : "does not write", (int)s.ds.dt, (int)s.ds.dw);
        }
    }
    BufferRef mesh_args{};
    uint32_t mesh_args_at = 0;
    // (RX_VK_MESH_ARGS=1: rewritten {count, 1, 1, first}. Off: it did not bring the canopies back, and it ends the pass at each draw)
    // (amd-fix, 11-07: on by default with the firsts (RX_VK_MESH_FIRST), the canopies back; RX_VK_MESH_ARGS=0: off. A single indirect
    // draw has no stride of its own: its entry is NVN's 16 bytes, taken as 8 the first was never copied)
    static const bool mesh_rewrite = getenv("RX_VK_MESH_ARGS") && atoi(getenv("RX_VK_MESH_ARGS")) != 0;   // (sharkiix version: off)
    if (mesh_rewrite && d.mesh && d.indirect) {   // (out of the pass: a copy cannot be made in one)
        const uint32_t n = (uint32_t)std::clamp(d.max, 1, (int)kMeshArgs), stride = d.stride ? (uint32_t)d.stride : 16u;
        BufferRef src = gpu_buffer_bind(d.indirect, (uint64_t)n * stride, false);
        if (!src) SKIP_HERE();
        if (mesh_args_ready()) {
            static const uint32_t word = getenv("RX_VK_MESH_COUNT_WORD") ? (uint32_t)atoi(getenv("RX_VK_MESH_COUNT_WORD")) : 0;
            end_pass(cmd);
            mesh_args_barrier(cmd, true);
            static thread_local std::vector<VkBufferCopy> copies;
            copies.clear();
            mesh_args_at = mesh_args_take(n);
            static const bool show = getenv("RX_VK_LOG_MESH_INDIRECT") != nullptr;
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
                static const bool per_prog = getenv("RX_VK_LOG_MESH_PERPROG") != nullptr;
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
    if (peek_prog(p.key)) peek_draw(cmd, p.key);   // (relink: RX_VK_PEEK_PROG)
    if (p.fallback && d.mesh && !fallback_record(cmd, p, d)) SKIP_HERE();
    fine_prepass.finish();
    {
        FINE_SCOPE(FT_HAZARD);
        begin_pass(cmd, at);
    }
    // the draw's occlusion query (vk_replay.cpp, RX_VK_OCC_QUERIES): begun in its pass and ended as draw()
    // returns, before each_guard can end the pass. A draw that returns before this point leaves its query unused: its report
    // stays "visible"
    struct OccGuard {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkQueryPool pool = VK_NULL_HANDLE;
        uint32_t q = 0;
        ~OccGuard() { if (pool) vkCmdEndQuery(cmd, pool, q); }
    } occ_guard;
    if (d.occ_pool) {
        vkCmdBeginQuery(cmd, d.occ_pool, d.occ_query, dev.occlusion_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
        occ_guard.cmd = cmd, occ_guard.pool = d.occ_pool, occ_guard.q = d.occ_query;
    }
    probe_draw(s, p, ps.draws);   // (relink: RX_VK_PROBE)
    gprof_run(cmd, p.key);   // (RX_VK_PASS_TIMING=2)
    FineScope fine_track(FT_CMD_TRACK);
    ps.prog = p.key;
    if (!ps.first_prog) ps.first_prog = p.key;
    for (const Texture* t : gfx_sampled()) {
        if (std::find(ps.sampled.begin(), ps.sampled.end(), t) != ps.sampled.end()) continue;
        ps.sampled.push_back(t);
        int slot = -1;
        for (int i = 0; i < at.n; i++)
            if (at.color[i] == t) { slot = i; break; }
        if (slot < 0 && at.depth == t) slot = 8;
        if (slot >= 0) ps.sampled_targets |= (uint16_t)(1u << slot);
    }
    ps.written |= writes;
    fine_track.finish();
    if (pl != last_pipeline) {
        FINE_SCOPE(FT_BINDPL);
        FINE_N(FC_BIND_PIPE, 1);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl), last_pipeline = pl;
    }
    if (!same_desc) {
        FINE_SCOPE(FT_BINDDESC);
        if (!bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p)) SKIP_HERE();
        last_desc_valid = true, last_desc_prog = &p, last_desc_bind = s.bind_gen, last_desc_reg = reg_gen;
        last_desc_ub = ub_gen;
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
        dynamic_state(cmd, s, at, topo, d.mesh && !p.fallback);
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
        static const bool log_restart = getenv("RX_VK_LOG_RESTART") != nullptr;
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
    StatsQuery stats_query(cmd, p.key, d);   // (EvilOOT, RX_VK_DRAW_STATS)
    if (stats_frame_now() && stats_prog_wanted(p.key)) {   // (and the wanted programs' buffers: a mesh draw's stages, fragment)
        for (const int stage : {6, 7, 1}) stats_dump_stage(p.key, s, stage);
        // (and as the GPU holds them, after it: RX_VK_GPU_DUMP=1 only; its bindings can sync a pool's mirror mid-frame, and
        // the culling chain then measured as broken)
        static const bool gpu_dump_on = getenv("RX_VK_GPU_DUMP") != nullptr;
        if (stats_query.cmd && gpu_dump_on) stats_query.state = &s, stats_query.prog = p.key;
    }
    fine_poststate.finish();
    if (d.mesh) {
        FINE_SCOPE(FT_MESH_DRAW);
        FINE_N(FC_MESH_DRAWS, 1);
        FINE_N(FC_MESH_INDIRECT, d.indirect != 0);
        FINE_N(FC_MESH_COUNTED, d.count_addr != 0);
        FINE_N(FC_MESH_FIRST, d.first_task != 0);
        if (p.fallback) {   // (relink: the generated vertex stage reads the records; one vertex a corner)
            // (issue 640: batch by batch, each batch's dispatches out of the pass, then its draw in it. The graphics pipeline and
            // descriptors stay bound across the passes; the push constants, which the dispatches set, are set again. The draw's
            // occlusion query ends with the first batch's pass: a query cannot span two)
            for (uint32_t b = 0; b < fallback_plan.batches; b++) {
                if (b) {
                    if (occ_guard.pool) vkCmdEndQuery(cmd, occ_guard.pool, occ_guard.q), occ_guard.pool = VK_NULL_HANDLE;
                    if (!fallback_batch(cmd, p, b)) break;
                    begin_pass(cmd, at);
                    push_constants(cmd, p, &at);
                }
                fallback_draw(cmd, p, b);
            }
            st_mesh++;
            st_draws++, ps.draws++;
            return;
        }
        {   // (amd-fix, diagnosis) RX_VK_LOG_MESH_PROGS=1: each mesh program's first draw and every 600th, with what its
            // indirect arguments hold in guest memory (0 groups: the culling pass before it left nothing)
            static const bool per_prog = getenv("RX_VK_LOG_MESH_PROGS") != nullptr;
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
            {   // (relink) RX_VK_LOG_TASK_PTRS=1: for a mesh program whose task stage keeps global memory accesses, once (60 of them):
                // its task storage buffers (constant buffer place, address, size), their first 4 doublewords, and for each one that
                // looks like a pointer whether it falls in one of those buffers (the global fallback finds only those)
                static const bool log_ptrs = getenv("RX_VK_LOG_TASK_PTRS") != nullptr;
                if (log_ptrs) {
                    static std::mutex m;
                    static std::unordered_set<uint64_t> seen;
                    bool tell = false;
                    {
                        std::lock_guard l(m);
                        tell = seen.size() < 60 && seen.insert(p.key).second;
                    }
                    for (const Stage& st : p.stages) {
                        if (!tell || st.nvn_stage != 6 || !st.info.uses_global_memory) continue;
                        struct R { uint32_t ci, off; uint64_t a, n; };
                        std::vector<R> rs;
                        for (const auto& sd : st.info.storage_buffers_descriptors) {
                            if (sd.count != 1) continue;
                            const uint64_t a = stage_cbuf_word(s, st, sd.cbuf_index, sd.cbuf_offset) |
                                               (uint64_t)stage_cbuf_word(s, st, sd.cbuf_index, sd.cbuf_offset + 4) << 32;
                            rs.push_back({sd.cbuf_index, sd.cbuf_offset, a, stage_cbuf_word(s, st, sd.cbuf_index, sd.cbuf_offset + 8)});
                        }
                        for (const R& r : rs) {
                            uint64_t w[4]{};
                            if (r.a) memory_read_as_submitted(r.a, w, sizeof w);
                            char line[700];
                            int k = snprintf(line, sizeof line, "vk: task ptrs %016llx frame %llu: c%u+%#x addr %#llx size %#llx:", (unsigned long long)p.key,
                                             (unsigned long long)texture_frame_no(), r.ci, r.off, (unsigned long long)r.a, (unsigned long long)r.n);
                            for (uint64_t q : w) {
                                const char* where = "";
                                if (q >= (1ull << 32) && q < (1ull << 48)) {
                                    where = " (outside)";
                                    for (const R& o : rs)
                                        if (q >= o.a && q < o.a + o.n) where = " (inside)";
                                    if (!*where || !strcmp(where, " (outside)")) where = gpu_pool_end(q) > q ? " (outside, in a pool)" : " (outside, no pool)";
                                    for (const R& o : rs)
                                        if (q >= o.a && q < o.a + o.n) where = " (inside)";
                                }
                                k += snprintf(line + k, sizeof line - k, " %#llx%s", (unsigned long long)q, where);
                            }
                            logf("%s", line);
                        }
                    }
                }
            }
            static const bool log_sizes = getenv("RX_VK_LOG_MESH_DRAWS") != nullptr;
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
            static const bool on = getenv("RX_VK_LOG_MESH_INDIRECT") != nullptr;
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
            BufferRef cnt;
            {
                FINE_SCOPE(FT_MESH_COUNT_BIND);
                cnt = d.count_addr ? gpu_buffer_bind(d.count_addr, 4, false) : BufferRef{};
            }
            if (mesh_conv_buf) {   // (EvilOOT: the records as {count, 1, 1}, mesh_indirect_convert)
                {
                    FINE_SCOPE(FT_MESH_CALL);
                    if (cnt && vkCmdDrawMeshTasksIndirectCountEXT)
                        vkCmdDrawMeshTasksIndirectCountEXT(cmd, mesh_conv_buf, mesh_conv_off, cnt.buffer, cnt.offset, mesh_conv_n, 12);
                    else
                        vkCmdDrawMeshTasksIndirectEXT(cmd, mesh_conv_buf, mesh_conv_off, mesh_conv_n, 12);
                }
                mesh_conv_buf = VK_NULL_HANDLE;
                st_mesh++;
                st_draws++, ps.draws++;
                return;
            }
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
    if (!vertex_set_same(vstamp, vaddr, vsize, vstride_key, vertex_set.stamp, vertex_set.addr, vertex_set.size,
                         vertex_set.stride, vertex_set.on)) {
        VkBuffer vbufs[16];
        VkDeviceSize voffs[16], vsizes[16], vstrides[16];
        bool complete = true;
        for (uint32_t b = 0; b < 16; b++) {
            vbufs[b] = zero_buffer, voffs[b] = 0, vsizes[b] = 65536, vstrides[b] = 0;
            if (b == 15 || !s.vbuf[b].address) continue;
            const uint64_t vb_addr = s.vbuf[b].address, vb_size = std::max<uint64_t>(s.vbuf[b].size, 1);
            const uint64_t vb_stamp = gpu_bind_stamp();
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
        if (!last_vb_valid || std::memcmp(vbufs, last_vb.b, sizeof vbufs) || std::memcmp(voffs, last_vb.o, sizeof voffs) ||
            std::memcmp(vsizes, last_vb.z, sizeof vsizes) || std::memcmp(vstrides, last_vb.s, sizeof vstrides)) {
            FINE_N(FC_VBIND, 1);
            vkCmdBindVertexBuffers2(cmd, 0, 16, vbufs, voffs, vsizes, vstrides);
            std::memcpy(last_vb.b, vbufs, sizeof vbufs), std::memcpy(last_vb.o, voffs, sizeof voffs);
            std::memcpy(last_vb.z, vsizes, sizeof vsizes), std::memcpy(last_vb.s, vstrides, sizeof vstrides);
            last_vb_valid = true;
        }
        if (complete && gpu_bind_stamp() == vstamp) {
            vertex_set.stamp = vstamp;
            for (int i = 0; i < 16; i++) {
                const bool used = i != 15 && vaddr[i];
                vertex_set.addr[i] = used ? vaddr[i] : 0;
                vertex_set.size[i] = used ? (vsize[i] ? vsize[i] : 1) : 0;
                vertex_set.stride[i] = used ? vstride_key[i] : 0;
            }
            vertex_set.on = true;
        } else
            vertex_set.on = false;
    }
    if (g_fine) fine_ns[FT_VERTEX] += prof_now() - fine_vertex_t0;
    const VkIndexType itype = d.itype == 0 ? VK_INDEX_TYPE_UINT8_EXT : d.itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    if (d.indexed && d.itype == 0 && !dev.index_uint8) SKIP_HERE();
    if (d.prim == NVN_DRAW_PRIMITIVE_QUADS) {   // (quads as two triangles each: indices made in the stream)
        if (d.indirect || d.count < 4) SKIP_HERE();
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
        vkCmdDrawIndexed(cmd, quads * 6, (uint32_t)std::max(1, d.instances), 0, d.indexed ? d.base_vertex : 0,
                         (uint32_t)d.base_instance);
        st_draws++, ps.draws++;
        return;
    }
    if (d.indirect) {
        BufferRef args = gpu_buffer_bind(d.indirect, (uint64_t)std::max(d.max, 1) * std::max(d.stride, 16), false);
        BufferRef cnt = d.count_addr ? gpu_buffer_bind(d.count_addr, 4, false) : BufferRef{};
        if (!args) SKIP_HERE();
        if (d.indexed) {
            BufferRef ib = gpu_buffer_bind(d.ibuf, 4, false);
            if (!ib) SKIP_HERE();
            bind_index(cmd, ib.buffer, ib.offset, itype);
            if (cnt) vkCmdDrawIndexedIndirectCount(cmd, args.buffer, args.offset, cnt.buffer, cnt.offset, d.max, std::max(d.stride, 20));
            else vkCmdDrawIndexedIndirect(cmd, args.buffer, args.offset, std::max(d.max, 1), std::max(d.stride, 20));
        } else {
            if (cnt) vkCmdDrawIndirectCount(cmd, args.buffer, args.offset, cnt.buffer, cnt.offset, d.max, std::max(d.stride, 16));
            else vkCmdDrawIndirect(cmd, args.buffer, args.offset, std::max(d.max, 1), std::max(d.stride, 16));
        }
    } else if (d.indexed) {
        const uint32_t unit = d.itype == 0 ? 1 : d.itype == 1 ? 2 : 4;
        BufferRef ib;
        {
            FINE_SCOPE(FT_VERTEX);
            ib = gpu_buffer_bind(d.ibuf, (uint64_t)std::max(d.count, 1) * unit, false);
            if (!ib) SKIP_HERE();
            FINE_N(FC_IBIND, 1);
            bind_index(cmd, ib.buffer, ib.offset, itype);
        }
        FINE_SCOPE(FT_DRAWCALL);
        vkCmdDrawIndexed(cmd, (uint32_t)d.count, (uint32_t)std::max(1, d.instances), 0, d.base_vertex, (uint32_t)d.base_instance);
    } else {
        FINE_SCOPE(FT_DRAWCALL);
        vkCmdDraw(cmd, (uint32_t)d.count, (uint32_t)std::max(1, d.instances), (uint32_t)d.first, (uint32_t)d.base_instance);
    }
    st_draws++, ps.draws++;
}

/// A dispatch over 2 ms is logged, 60 times, and only while RX_VK_SLOW_DISPATCH is set. Off, the clock is not read.
bool slow_dispatch_timing() {
    static const int on = getenv("RX_VK_SLOW_DISPATCH") != nullptr;
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

/// (relink) OoT's neural upscaler (about 13 tensor-core compute passes a frame over storage-buffer feature maps,
/// 1920x1080 in, 2560x1440 out) does not translate: its matrix multiplies and a few other native opcodes have no
/// lowering. Its last pass reads the network's result, the input colour and a noise texture and writes the output
/// image; left out, the output stays black under the UI drawn over it. Here a rejected compute pass that writes a
/// storage image larger than the largest colour texture it samples (and does not also write) is replaced by a linear
/// blit of that texture into the image: the frame at its full size, without the network's detail. (An image the
/// pass also samples is one of its inputs, never the output.)
/// RX_VK_NO_UPSCALE_STUB=1: left out, as before. The first replacement of each program is logged.
/// (shaders/movie_upscale.comp) The upscale in a shader: it reads and writes the two images as the network's own passes do, where a blit came
/// out black on an RX 9070 XT. False: no way to, the blit is tried
bool movie_upscale(VkCommandBuffer cmd, uint32_t in_handle, uint32_t out_handle, const Texture* out) {
    static const bool off = getenv("RX_VK_MOVIE_BLIT") != nullptr;
    if (off || !dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return false;
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
        mi.codeSize = sizeof MOVIE_UPSCALE_COMP_SPV, mi.pCode = MOVIE_UPSCALE_COMP_SPV;
        VkShaderModule m{};
        vkCreateShaderModule(dev.device, &mi, nullptr, &m);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
        ci.layout = layout;
        if (vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
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

// (relink, issue 109) The movie upscaler's networks cost 12 ms a frame on an RTX 3050 and 83-93 ms on an RTX 4060 with a newer
// driver: the intro movie ran at 10 fps and the game never got past it. When a frame of few draws (a movie) takes over 40 ms
// of GPU time (vk_replay.cpp), they are left out from then on, as on AMD: the linear upscale below, the same picture a little
// softer. The gameplay networks are never left out. RX_VK_NETWORK_KEEP=all: never
std::atomic<bool> movie_networks_slow{false};
bool gameplay_network(uint64_t key) { return key == 0xfd029a317a9a4b94ull || key == 0x9dc98871eb7fc101ull; }

static movie_upscale_policy::Mode movie_upscale_mode() {
    // Read on first dispatch, after host/hos_main.c has loaded relink_env.txt.
    static const auto mode = movie_upscale_policy::parse(getenv("RX_VK_MOVIE_UPSCALE"));
    return mode;
}

static bool movie_keep_all_networks() {
    static const bool keep = getenv("RX_VK_NETWORK_KEEP") && !strcmp(getenv("RX_VK_NETWORK_KEEP"), "all");
    return keep;
}

static bool typed_upscale(VkCommandBuffer cmd, VkImageView input_view, VkImageView output_view,
                          Texture* out, uint32_t level, uint64_t key, const char* keys, bool temporal);

bool upscale_stub(VkCommandBuffer cmd, const NvnState& s, uint64_t key) {
    static const bool off = getenv("RX_VK_NO_UPSCALE_STUB") != nullptr;
    if (off) return false;
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
    if (!in_area || out_area <= in_area) return false;
    Texture* src = texture_get(&in, true);
    Texture* dst = texture_get(&out, false);
    if (!src || !dst || !src->image || !dst->image || (src->aspect | dst->aspect) != VK_IMAGE_ASPECT_COLOR_BIT || src->bc ||
        dst->bc || src->samples != VK_SAMPLE_COUNT_1_BIT || dst->samples != VK_SAMPLE_COUNT_1_BIT)
        return false;
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
    // The generic failed-network path can receive RGB9E5 bits in a shared R11 image.
    // Match the observed base-level 2D case only; other fallbacks keep their existing blit.
    if (!gameplay_network(key) && in.d.format == NVN_FORMAT_RGB9E5F && out.d.format == NVN_FORMAT_R11G11B10F &&
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
            if (input_format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 && output_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
                // A gameplay network that did not translate reaches this stub before dispatch() can call
                // dlss_stand_in. Give the requested external upscaler the same scene input/output here; if it cannot
                // take this frame, keep the existing typed fallback as the bounded, visible recovery path.
                bool external = false;
                if (dlss_wanted()) {
                    external = dlss_upscale(cmd, key, gameplay_network(key), src, input_view, dst, output_view, 0);
                    // dlss_upscale may have drawn actor motion before returning false; either result replaces graphics state.
                    graphics_state_disturbed();
                }
                if (external || typed_upscale(cmd, input_view, output_view, dst, 0, key, "generic RGB9E5 fallback", false)) {
                    full_barrier(cmd);
                    texture_written(dst);
                    return true;
                }
            }
        }
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

/// (relink) OoT's gameplay TAAU network (fd029a317a9a4b94) stood in for: its input upscaled into its output
/// (shaders/taau_upscale.comp). The network's translation loses the blue channel, the yellow cast over every
/// scene; with this the picture is right, without the network's temporal anti-aliasing.
/// RX_VK_TAAU_KEYS=<program key prefixes,...> (default fd029a317a9a4b94; "-": none, the network runs)
bool taau_key(uint64_t key) {
    // (the real network where the card runs it as the console does; on the software multiplies it cost most of a frame on AMD's cards, and a
    // locked 60 with this in its place)
    static const char* keys = getenv("RX_VK_TAAU_KEYS") ? getenv("RX_VK_TAAU_KEYS")
                              : Shader::Backend::SPIRV::mma_software || getenv("RX_VK_NO_F16")
                                    ? "fd029a317a9a4b94" : "-";   // (RX_VK_NO_F16: vk_shader.cpp leaves the network out)
    if (keys[0] == '-' && !keys[1]) return false;   // (relink: the default on cards with native multiplies, without a snprintf a dispatch)
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
    static VkPipeline pipe = VK_NULL_HANDLE;
    static VkPipelineLayout layout = VK_NULL_HANDLE;
    static VkSampler linear = VK_NULL_HANDLE;
    static bool tried = false;
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
        mi.codeSize = sizeof TAAU_UPSCALE_COMP_SPV, mi.pCode = TAAU_UPSCALE_COMP_SPV;
        VkShaderModule m{};
        vkCreateShaderModule(dev.device, &mi, nullptr, &m);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
        ci.layout = layout;
        if (vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
        vkDestroyShaderModule(dev.device, m, nullptr);
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.25f;
        vkCreateSampler(dev.device, &si, nullptr, &linear);
        logf("vk: typed upscale helper %s%s", keys,
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
    static const bool no_clamp = getenv("RX_VK_TAAU_CLAMP") == nullptr;
    // (the ones a new size replaced: given back 120 presents later, when no work in flight can use them. They were left, the
    // output's size changing with the dynamic resolution. RX_VK_TAAU_HIST_LEAK=1: as before)
    struct OldHist { VkImage image; VmaAllocation mem; VkImageView view; uint64_t frame; };
    static std::vector<OldHist> old_hist;
    static const bool leak = getenv("RX_VK_TAAU_HIST_LEAK") != nullptr;
    for (size_t i = 0; i < old_hist.size();) {
        if (texture_frame_no() - old_hist[i].frame < 120) { i++; continue; }
        if (old_hist[i].view) vkDestroyImageView(dev.device, old_hist[i].view, nullptr);
        destroy_image(old_hist[i].image, old_hist[i].mem);
        old_hist.erase(old_hist.begin() + (ptrdiff_t)i);
    }
    if (hist_w != size[0] || hist_h != size[1]) {
        // (the one before is kept for the while: work in flight may still use it)
        if (hist && !leak) old_hist.push_back({hist, hist_mem, hist_view, texture_frame_no()});
        hist = VK_NULL_HANDLE, hist_view = VK_NULL_HANDLE;
        VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ic.imageType = VK_IMAGE_TYPE_2D, ic.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
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
        hist_w = size[0], hist_h = size[1], hist_frame = 0;
    }
    // (bounded when the history is last frame's or so: a pause of the TAAU, a menu, starts again unbounded)
    const uint64_t now_frame = texture_frame_no();
    size[2] = temporal && !no_clamp && hist_frame && now_frame - hist_frame <= 3 ? 1u : 0u;
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
/// (relink) NVIDIA DLSS in place of the gameplay upscale network (vk_dlss.cpp), when it is on and its inputs are as expected:
/// the network's input as it samples it, its output's storage view
bool dlss_stand_in(VkCommandBuffer cmd, const NvnState& s, const Program& p) {
    // (the 4/3 network and the full-size pass by their keys; the game's other gameplay upscales, one network for each fixed
    // share of the Render resolution row (67%: 6676619641320ba7, 58% f10782f9e58b6a7c, 50% 72ba090c70fe2614, 33%
    // 0775215f8da720e6 at 2048x1152), by their shape: a network sampling one texture into one image. dlss_upscale takes those
    // only when the texture is this frame's scene, in the gameplay networks' input format, and the image the picture)
    if (!dlss_wanted() || p.stages.empty()) return false;
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
    const bool done = dlss_upscale(cmd, p.key, gameplay_network(p.key), in, dw().imgs[0].imageView, out, dw().imgs[ntex].imageView, level);
    // (its actor motion may have drawn with a pipeline, viewport and descriptors of its own: the next draw sets all of its again)
    graphics_state_disturbed();
    return done;
}
/// (relink) After the gameplay upscale wrote its output (the game's network, or DLSS in its place: stood_in),
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
    if (!out || out->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32) return false;
    const char* keys = getenv("RX_VK_TAAU_KEYS") ? getenv("RX_VK_TAAU_KEYS") : "fd029a317a9a4b94";
    return typed_upscale(cmd, dw().imgs[0].imageView, dw().imgs[ntex].imageView, out, level, p.key, keys, true);
}

bool precompile_wanted() {
#if defined(RELINK_RELEASE)
    bool on = true;
#else
    bool on = false;
#endif
    if (const char* e = getenv("RX_VK_PRECOMPILE")) on = atoi(e) != 0;
    return on;
}
namespace {
std::atomic<bool> prepare_halt{false};
}
void prepare_stop() { prepare_halt.store(true); }
bool prepare_stopped() { return prepare_halt.load(std::memory_order_relaxed); }
std::string prepared_path() {
    char exe[MAX_PATH];
    const DWORD exe_n = game_module_path(exe, MAX_PATH);
    std::string path(exe, exe_n);
    return path.substr(0, path.find_last_of("\\/") + 1) + "relink_prepared.bin";
}
/// (relink) An update: every shader was prepared once here, for another translator. Each release changes the translator, and every
/// update then held the game's start for the whole seed again (116k variants, minutes on most PCs: "shaders need to be prepared every
/// update"). With RX_VK_PREPARE_BACKGROUND=1 only what the notes name is built before the game (precompile_run), the rest in play
/// behind it (prepare_background)
bool prepare_is_update() {
    // (opt-in, RX_VK_PREPARE_BACKGROUND=1: when the translator changed, the program cache is gone too, and the first area then
    // translated thousands of programs in play: 1 fps and black frames for minutes, which looked broken. Before the game is the default)
    if (getenv("RX_VK_PREPARE_AGAIN") || !getenv("RX_VK_PREPARE_BACKGROUND")) return false;
    FILE* f = fopen(prepared_path().c_str(), "rb");
    if (!f) return false;
    uint64_t have = 0;
    const bool read = fread(&have, 8, 1, f) == 1;
    fclose(f);
    return read && have != program_cache_stamp();
}
/// Every shader of the game, before its first start (vk_shader.cpp, the seed): once for this translator. True: prepared now.
/// background: in play, behind the game: a few workers at idle priority, translation only (the driver's part is left to first use:
/// its compiler crashing in here would take the game with it)
bool prepare_run(void (*progress)(size_t done, size_t total), bool background) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string path = prepared_path();
    const uint64_t stamp = program_cache_stamp();
    if (FILE* f = fopen(path.c_str(), "rb")) {
        uint64_t have = 0;
        const bool same = fread(&have, 8, 1, f) == 1 && have == stamp;
        fclose(f);
        if (same && !getenv("RX_VK_PREPARE_AGAIN")) {
            if (getenv("RX_VK_SEED_CHECK")) seed_open();
            return false;
        }
    }
    const size_t programs = seed_open();
    if (!programs) return false;
    const size_t total = programs * 3;   // (each program for float, one and two integer bone inputs)
    // The driver's part: each program's stage libraries, built once and let go, for the driver's own disk cache. What stays with us is
    // nothing: a kept library costs memory, and our pipeline cache would be read in whole at every start. A driver whose compiler
    // crashes on one of them (it happens: TotK Recomp's issue 30) would crash here at every start: the file says this step began and
    // did not end, and the next start goes without it. Those libraries are then built when a draw first needs them
    const std::string warming = path + ".warming";
    bool warm = gpl_wanted() && !getenv("RX_VK_NO_WARM") && !background;
    bool warm_cs = !getenv("RX_VK_NO_WARM") && !background;
    if (background) {
    } else if (FILE* f = fopen(warming.c_str(), "rb")) {
        fclose(f);
        warm = warm_cs = false;
        logf("vk: the last start did not finish the driver's part of the shaders: left out");
    } else if (warm || warm_cs) {
        if (FILE* w = fopen(warming.c_str(), "wb")) fclose(w);
    }
    auto blank = std::make_unique<NvnState>();
    blank->nvattr = 16;
    pcache_init();   // (once, here: the workers below build compute pipelines into it)
    std::atomic<size_t> next{0}, done{0}, made{0}, libs{0}, variants{0}, asking{0}, capped{0};
    // (relink) compute pipelines too, for the driver's own cache: built in play they stopped the game while the driver compiled
    // them, seconds each for the culling programs (53caac5f: 6.5 s, 41 s inlined), which looked like a hang
    const auto warm_compute = [&](Program& p) {
        if (!warm_cs || taau_key(p.key)) return;
        ensure_layout(p);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = p.stages[0].module;
        ci.stage.pName = "main";
        pin_subgroup(ci.stage, p.workgroup[0]);
        ci.layout = p.layout;
        if (getenv("RX_VK_NOOPT")) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        VkPipeline pl = VK_NULL_HANDLE;
        // (into the app's pipeline cache, saved to mkw_vk_pipelines.bin: without it the next start compiled them again in play
        // where the driver's own cache missed them, AMD's 15 s culling compiles each time)
        if (guarded_compute(pcache, &ci, &pl) == VK_SUCCESS && pl) {
            vkDestroyPipeline(dev.device, pl, nullptr);
            libs++;
            pcache_built_since_save++;
        }
        if (p.layout) vkDestroyPipelineLayout(dev.device, p.layout, nullptr), p.layout = VK_NULL_HANDLE;
        if (p.set_layout) vkDestroyDescriptorSetLayout(dev.device, p.set_layout, nullptr), p.set_layout = VK_NULL_HANDLE;
    };
    const auto worker = [&] {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        // (below normal, and half the CPUs, below: on all of them at High the driver's compiles froze the whole PC)
        SetThreadPriority(GetCurrentThread(), background ? THREAD_PRIORITY_IDLE : THREAD_PRIORITY_BELOW_NORMAL);
        for (;;) {
            if (prepare_stopped()) return;
            const size_t i = next++;
            if (i >= total) return;
            if (seed_is_compute(i / 3)) {   // (relink) a compute program with each of its texture variants, each warmed
                if (i % 3 == 0) {
                    const SeedVariants v = seed_compute(i / 3, 64, warm_compute);
                    made += v.variants != 0, variants += v.variants, asking += v.variants > 1, capped += v.capped && v.variants;
                }
                done++;
                continue;
            }
            if (std::unique_ptr<Program> p = seed_program(i / 3, (int)(i % 3))) {
                if (!p->failed) made++;
                if (!p->failed && warm && !p->compute) {
                    ensure_layout(*p);
                    Targets at{};
                    for (int part = 1; part <= 2; part++)
                        if (const VkPipeline lib = build_graphics(*p, *blank, at, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, part)) {
                            vkDestroyPipeline(dev.device, lib, nullptr);
                            libs++;
                        }
                }
                for (Stage& st : p->stages)
                    if (st.module) vkDestroyShaderModule(dev.device, st.module, nullptr);
                if (p->layout) vkDestroyPipelineLayout(dev.device, p->layout, nullptr);
                if (p->set_layout) vkDestroyDescriptorSetLayout(dev.device, p->set_layout, nullptr);
            }
            done++;
        }
    };
    std::vector<std::thread> threads;
    const unsigned cpus = std::max(1u, std::thread::hardware_concurrency());
    // (in play: a quarter of the CPUs, two to four, the game's own threads before them)
    const unsigned workers = background ? std::clamp(cpus / 4, 2u, 4u) : prepare_threads();
    for (unsigned i = 0; i < workers; i++) threads.emplace_back(worker);
    while (done.load() < total && !prepare_stopped()) {
        if (progress) progress(done.load() / 3, programs);   // (in programs, as the log counts them, not programs times variants)
        Sleep(100);
        if (next.load() >= total + threads.size()) break;
    }
    for (std::thread& t : threads) t.join();
    if (!background) state_list_prepare(nullptr);   // (in play it is too late for the precompile: an update's is state_list_run, before)
    if (!getenv("RX_VK_SEED_CHECK")) seed_close();
    if (prepare_stopped()) {   // (closed: what was made is in the program cache already, the rest is made at the next start)
        logf("vk: shader preparation stopped at %zu of %zu (the window was closed): the next start goes on from there", done.load(),
             total);
        remove(warming.c_str());
        return false;
    }
    int64_t translated = 0, failed = 0;
    shader_stats(&translated, &failed);
    logf("vk: prepared %zu shader programs%s (%zu new, %lld failed, %zu stage libraries; %zu compute variants, %zu programs with "
         "texture variants, %zu at the limit) in %.1f s", programs, background ? " in play" : "", made.load(), (long long)failed, libs.load(), variants.load(),
         asking.load(), capped.load(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (!background) remove(warming.c_str());
    if (FILE* f = fopen(path.c_str(), "wb")) fwrite(&stamp, 8, 1, f), fclose(f);
    return true;
}
/// (relink) An update (prepare_is_update): the seed is prepared in play, behind the game, too late for the precompile; the pipelines
/// of the state list are made now, before it, so that precompile_run builds them before the game as on a first start
void state_list_run() {
    if (getenv("RX_VK_NO_STATE_LIST") || !seed_open()) return;
    state_list_prepare(nullptr);
    if (!getenv("RX_VK_SEED_CHECK")) seed_close();
}
void prepare_background() {
    std::thread([] {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
        Sleep(20000);   // (past the title screen's own loading)
        logf("vk: an update: the shaders not met yet are translated in play, behind the game");
        prepare_run(nullptr, true);
    }).detach();
}

// (relink) Lean pipelines leave the noted graphics pipelines to their first draw, which is quick only when the driver's own cache
// has them. It has none after a first install, after a driver update, or when the game's program moved or changed its name: NVIDIA
// keeps its cache by program, and night68's new name emptied it (the Market ran at 29 fps on its first visit, with up to 221 draws a
// frame left out while their pipelines were made, against under 10 before). So when the driver's cache has not seen this build,
// each noted pipeline is built once for that cache and let go, as predict_run does for the predicted ones: its program is borrowed
// from relink_programs.bin for it and nothing stays in memory. relink_pipelines_warm.txt says which driver, build and program were
// warmed. RX_VK_WARM=0: never; =1: at this start whatever the file says. RX_VK_WARM_S: a time budget in seconds (600; 0: none)
std::string warm_marker_path() {
    const std::string notes = pipe_notes_path();
    return notes.substr(0, notes.find_last_of("\\/") + 1) + "relink_pipelines_warm.txt";
}
std::string warm_marker_now() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    char line[MAX_PATH + 128];
    snprintf(line, sizeof line, "%08x %04x:%04x %016llx %.*s", (unsigned)dev.props.driverVersion, (unsigned)dev.props.vendorID,
             (unsigned)dev.props.deviceID, (unsigned long long)program_cache_stamp(), (int)n, exe);
    return line;
}
// (relink, 15/11) The marker says the driver's cache was warmed for this driver, build and program, but the cache itself can go
// meanwhile, the marker unchanged: NVIDIA's emptied itself on the bench between two runs (hs2988 -> hs2989: the start's 5030
// compute pipelines 2.7 s -> 76.6 s), nothing warmed it again, and play made each graphics pipeline at its first draw (the Market
// turned at 34 fps, nvgpucomp64.dll in nearly every long frame's samples, hs2991). So a start the marker says is warm probes the
// cache first: kProbe notes spread over the file, at an offset that changes from start to start (the same ones each time were
// in the cache from the probe before: a warm-up stopped halfway then probed warm, hs2994), are built for the driver's cache one
// by one and timed. A quarter of them over RX_VK_WARM_PROBE_MS (12) means the driver compiles them: the cache lost them, and the
// start warms it again (the marker removed first, so that a warm-up cut short goes on at the next start). Measured on the bench
// (RTX 3060 Ti): from the cache, a median of 1.5 to 2.7 ms and 8.5 ms at most; compiled, a median of 63 ms (3 ms called a warm
// cache cold once, hs2997: 5 s of warm-up for nothing). Warm, the probe costs kProbe pipelines from the cache and their programs
// read (0.1 s). RX_VK_WARM_PROBE_MS=0: no probe (the marker alone, as before)
bool driver_cache_lost(const std::vector<std::unique_ptr<PipeNote>>& notes) {
    static const double limit_ms = getenv("RX_VK_WARM_PROBE_MS") ? atof(getenv("RX_VK_WARM_PROBE_MS")) : 12.0;
    if (limit_ms <= 0 || notes.empty()) return false;
    constexpr size_t kProbe = 16;
    const size_t n = notes.size(), step = std::max<size_t>(1, n / kProbe),
                 off = (size_t)(std::chrono::steady_clock::now().time_since_epoch().count() / 1000) % step;
    std::vector<const PipeNote*> pick;
    for (size_t k = 0; k < kProbe && k < n; k++) pick.push_back(notes[(k * step + off) % n].get());
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> ms;
    std::thread([&] {   // (as warm_run's workers: a thread of its own, its pipelines for the driver's cache only)
        ub_thread_off();
        driver_cache_only = true;
        std::unordered_set<uint64_t> wanted;
        for (const PipeNote* n : pick) wanted.insert(n->prog);
        program_cache_load_some(wanted, [](Program& p) { ensure_layout(p); }, false);
        for (const PipeNote* n : pick) {
            const Program* p = program_find(n->prog, n->variant);
            if (!p) continue;
            const auto b0 = std::chrono::steady_clock::now();
            const VkPipeline pl = build_graphics(*p, n->s, note_targets(*n), (VkPrimitiveTopology)n->topo);
            if (!pl) continue;
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count());
            vkDestroyPipeline(dev.device, pl, nullptr);
        }
    }).join();
    if (ms.size() < kProbe / 2) {   // (too few built to tell: the marker decides, as before)
        logf("vk: the driver's cache not probed: %zu of %zu pipelines built", ms.size(), pick.size());
        return false;
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    const size_t cold = (size_t)(ms.end() - std::upper_bound(ms.begin(), ms.end(), limit_ms));
    const bool lost = cold * 4 >= ms.size();
    logf("vk: the driver's cache probed: %zu pipelines in %.2f s, median %.2f ms, longest %.2f ms, %zu over %.1f ms: %s", ms.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), med, ms.back(), cold, limit_ms,
         lost ? "it no longer has them, warmed again" : "warm");
    return lost;
}
/// The warm-up; its progress counts after more items to come (the start's compute pipelines: precompile_run). How many it
/// counted: its notes when it ran, 0 when it had nothing to do
size_t warm_run(void (*progress)(size_t done, size_t total), size_t after) {
    static const char* const sw = getenv("RX_VK_WARM");
    if (sw && !atoi(sw)) return 0;
    const std::string now = warm_marker_now(), path = warm_marker_path();
    bool marked = false;
    if (!sw) {
        char had[MAX_PATH + 128] = "";
        if (FILE* f = fopen(path.c_str(), "rb")) {
            const size_t n = fread(had, 1, sizeof had - 1, f);
            fclose(f);
            had[n] = 0;
            if (char* end = strpbrk(had, "\r\n")) *end = 0;
        }
        marked = now == had;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<std::unique_ptr<PipeNote>> notes = pipe_notes_read();
    if (notes.empty()) return 0;   // (nothing noted yet: play notes them, and the driver's cache has them from their first draw)
    if (marked) {
        if (!driver_cache_lost(notes)) return 0;
        remove(path.c_str());   // (a warm-up cut short is not marked: the next start warms again without a probe)
    }
    // (the notes' programs read in one pass, as the full precompile reads them: borrowing each from the file for its pipelines,
    // as predict_run does, made 8535 pipelines in 370 s cold and still 310 s with the driver's cache warm, one program at a time;
    // play asks for these programs anyway, the notes being what it drew before)
    std::unordered_set<uint64_t> wanted;
    for (const auto& n : notes) wanted.insert(n->prog);
    const size_t programs_n = program_cache_load_some(wanted, [](Program& p) { ensure_layout(p); }, false);
    const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // (relink, issue 80) A pipeline the driver never returns from held the start for good: AMD's on an RX 6600 / 6600 XT stopped
    // at "pipelines 8517 of 13329" (3 short of the warm-up's 8520) and at "1139 of 5951", the CPU idle, for over an hour. The
    // workers looked at the time budget only between pipelines, and the start joined them all. Now the workers' state is on the
    // heap and this thread watches: nothing finished for RX_VK_WARM_STALL_S seconds (30; 0: wait, as before), or the budget spent,
    // and the start goes on without the ones still inside the driver (let go, never freed). Those are named in the log and kept in
    // <marker>.stuck, which the next starts leave out of the warm-up (they are built in play when first drawn, as any other)
    struct Warm {
        std::vector<std::unique_ptr<PipeNote>> notes;
        std::atomic<size_t> next{0}, done{0}, made{0};
        std::atomic<int> alive{0};
        std::atomic<bool> stop{false};
        std::unique_ptr<std::atomic<int64_t>[]> current;   // (each worker's note, -1 between them)
        std::unique_ptr<std::atomic<int64_t>[]> since;     // (when it went into the driver for it, steady ns)
    };
    const std::string stuck_path = path + ".stuck";
    std::unordered_set<uint64_t> stuck_before;
    if (FILE* f = fopen(stuck_path.c_str(), "rb")) {
        unsigned long long k;
        while (fscanf(f, "%llx", &k) == 1) stuck_before.insert((uint64_t)k);
        fclose(f);
    }
    auto w = std::make_shared<Warm>();
    size_t skipped_stuck = 0;
    for (const auto& n : notes) {   // (the notes are read again for the heap: the warm-up's own copy, kept by stuck workers)
        if (stuck_before.count(n->key)) { skipped_stuck++; continue; }
        w->notes.push_back(std::make_unique<PipeNote>(*n));
    }
    if (skipped_stuck) logf("vk: warm-up: %zu pipelines left out, the driver never returned from them before (%s)", skipped_stuck,
                            stuck_path.c_str());
    static const int budget_s = getenv("RX_VK_WARM_S") ? atoi(getenv("RX_VK_WARM_S")) : 600;
    static const int stall_s = getenv("RX_VK_WARM_STALL_S") ? atoi(getenv("RX_VK_WARM_STALL_S")) : 30;
    // (one pipeline in the driver for RX_VK_WARM_PIPE_S seconds (60; 0: no such limit) is taken as stuck at once: noted in
    // <marker>.stuck and a new worker started in its place. On this RX 6600 XT six of eight workers were stuck so: the two left
    // kept the progress going, slowly, so the stall rule never fired and the warm-up spent its whole budget at a quarter speed)
    static const int pipe_s = getenv("RX_VK_WARM_PIPE_S") ? atoi(getenv("RX_VK_WARM_PIPE_S")) : 60;
    const unsigned cpus = std::max(1u, std::thread::hardware_concurrency());
    const unsigned nthreads = std::min(cpus, 64u), slots = nthreads * 4;   // (room for the workers that replace stuck ones)
    w->current.reset(new std::atomic<int64_t>[slots]);
    w->since.reset(new std::atomic<int64_t>[slots]);
    for (unsigned k = 0; k < slots; k++) w->current[k] = -1, w->since[k] = 0;
    const auto steady_ns = [] {
        return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const auto worker = [w, steady_ns](unsigned k) {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        driver_cache_only = true;
        for (;;) {
            if (prepare_stopped() || w->stop.load(std::memory_order_relaxed)) break;
            const size_t i = w->next++;
            if (i >= w->notes.size()) break;
            const PipeNote& n = *w->notes[i];
            w->since[k] = steady_ns();
            w->current[k] = (int64_t)i;
            {   // (a test, on a PC whose driver returns from everything: RX_VK_WARM_TEST_STUCK=<n> makes every n-th noted
                // pipeline one the "driver" never returns from, so the rules above can be seen at work)
                static const long long test_stuck = getenv("RX_VK_WARM_TEST_STUCK") ? atoll(getenv("RX_VK_WARM_TEST_STUCK")) : 0;
                if (test_stuck > 0 && i % (size_t)test_stuck == (size_t)test_stuck - 1)
                    for (;;) Sleep(1000);
            }
            if (const Program* p = program_find(n.prog, n.variant))
                if (const VkPipeline pl = build_graphics(*p, n.s, note_targets(n), (VkPrimitiveTopology)n.topo))
                    vkDestroyPipeline(dev.device, pl, nullptr), w->made++;
            w->current[k] = -1;
            w->done++;
        }
        w->alive--;
    };
    std::vector<std::thread> threads;
    std::vector<bool> stuck(slots, false);
    std::vector<uint64_t> stuck_keys;
    std::string stuck_names;
    const auto note_stuck = [&](unsigned k) {   // (named in the log and kept in the file at once: a crash later keeps it too)
        const int64_t i = w->current[k].load();
        if (i < 0 || stuck[k]) return false;
        stuck[k] = true;
        const PipeNote& n = *w->notes[(size_t)i];
        stuck_keys.push_back(n.key);
        char one[96];
        snprintf(one, sizeof one, "%s%016llx (pipeline %016llx)", stuck_names.empty() ? "" : ", ", (unsigned long long)n.prog,
                 (unsigned long long)n.key);
        if (stuck_names.size() < 1500) stuck_names += one;
        if (FILE* f = fopen(stuck_path.c_str(), "ab")) fprintf(f, "%016llx\n", (unsigned long long)n.key), fclose(f);
        return true;
    };
    w->alive = (int)nthreads;
    for (unsigned k = 0; k < nthreads; k++) threads.emplace_back(worker, k);
    size_t last_done = 0, stuck_n = 0;
    auto last_change = std::chrono::steady_clock::now();
    bool out_of_time = false, stalled = false;
    while (w->alive.load() > (int)stuck_n && !prepare_stopped()) {
        if (progress) progress(w->done.load(), w->notes.size() + skipped_stuck + after);
        Sleep(100);
        const auto t = std::chrono::steady_clock::now();
        if (pipe_s > 0) {
            const int64_t now_ns = steady_ns();
            for (unsigned k = 0; k < (unsigned)threads.size(); k++)
                if (!stuck[k] && w->current[k].load() >= 0 && now_ns - w->since[k].load() > pipe_s * 1000000000LL && note_stuck(k)) {
                    stuck_n++;
                    logf("vk: warm-up: pipeline %016llx in the driver for over %d s: left behind, a new worker in its place",
                         (unsigned long long)stuck_keys.back(), pipe_s);
                    if (threads.size() < slots) w->alive++, threads.emplace_back(worker, (unsigned)threads.size());
                }
        }
        if (budget_s > 0 && t - t0 > std::chrono::seconds(budget_s)) { out_of_time = true; break; }
        if (w->done.load() != last_done) last_done = w->done.load(), last_change = t;
        else if (stall_s > 0 && t - last_change > std::chrono::seconds(stall_s)) { stalled = true; break; }
    }
    w->stop = true;
    {   // (the workers between pipelines stop at once; give the others a moment, then let go of those still in the driver)
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (w->alive.load() > 0 && std::chrono::steady_clock::now() < until) Sleep(20);
    }
    const bool all_out = w->alive.load() == 0;
    if (all_out) {
        for (std::thread& t : threads) t.join();
    } else {
        for (std::thread& t : threads) t.detach();
        for (unsigned k = 0; k < (unsigned)threads.size(); k++) note_stuck(k);   // (those the budget or the stall rule left)
    }
    if (!stuck_keys.empty())
        logf("vk: warm-up: %zu pipeline%s the driver did not return from, left behind (program %s); the game starts, and the next "
             "starts leave them out%s", stuck_keys.size(), stuck_keys.size() == 1 ? "" : "s", stuck_names.c_str(),
             out_of_time ? " (the warm-up's time budget, RX_VK_WARM_S, ran out)" : stalled ? " (nothing finished for RX_VK_WARM_STALL_S)" : "");
    const bool whole = !prepare_stopped() && !out_of_time && !stalled && all_out && stuck_keys.empty();
    if (whole)
        if (FILE* f = fopen(path.c_str(), "wb")) fprintf(f, "%s\n", now.c_str()), fclose(f);
    logf("vk: warmed the driver's cache: %zu of %zu noted pipelines made in %.1f s (%.1f s of it %zu programs read)%s",
         w->made.load(), notes.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), read_s,
         programs_n, whole ? "" : " (stopped before the end: the rest are made in play, and the next start goes on)");
    return notes.size();
}

// (relink, 0.7.8) The full start's pipeline cache files, mkw_vk_pipelines.bin and its .new (800 MB and more on the PCs that
// had the full start until 0.7.5), are read by nothing in a lean start (pcache_init returns before) and written by nothing
// either: a lean start removes them, said once in the log. Only the files beside the game, never RX_VK_PIPELINE_CACHE_PATH's,
// and never in the full start (RX_VK_LEAN_PIPELINES=0, RX_VK_PRECOMPILE), which is the one that reads them. A player who
// turns the full start on later loses nothing but time: pcache_init starts from an empty cache, as on a new install, that
// start builds every noted pipeline from the driver's own cache (or compiles it), and the file is written again.
// RX_VK_KEEP_OLD_PIPELINE_CACHE=1: kept (a test folder that goes from one start to the other)
void old_pipeline_cache_drop() {
    if (getenv("RX_VK_KEEP_OLD_PIPELINE_CACHE")) return;
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string dir(exe, n);
    const std::string base = dir.substr(0, dir.find_last_of("\\/") + 1) + "mkw_vk_pipelines.bin";
    for (const std::string& path : {base, base + ".new"}) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) continue;   // (not there: the usual case)
        _fseeki64(f, 0, SEEK_END);
        const int64_t size = _ftelli64(f);
        fclose(f);
        if (remove(path.c_str()) == 0)
            logf("vk: lean pipelines: %s removed (%lld MB): only the full start reads it (RX_VK_LEAN_PIPELINES=0 makes it again)",
                 path.c_str(), (long long)(size >> 20));
        else
            logf("vk: lean pipelines: %s, which only the full start reads, could not be removed (in use?): left for the next start",
                 path.c_str());
    }
}

void precompile_run(void (*progress)(size_t done, size_t total)) {
    const auto t0 = std::chrono::steady_clock::now();
    if (pipelines_lean()) old_pipeline_cache_drop();
    // (a start that dies in here would die in here every time, the notes being what it builds from: a player saw the console reach
    // 542 of 543 and the game close. The file says this step began and did not end: the next start lets the notes go and plays)
    // (two starts in a row, not one: closing the window while it ran counted as a crash, and the next start then skipped the
    // precompile and compiled everything in play: "stuck at 99%, then fine on the second start", 2026-11-08)
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
        logf("vk: the last start did not get through the precompile (closed, or it crashed there): trying it once more");
    }
    if (FILE* f = fopen(running.c_str(), "wb")) fprintf(f, "%d\n", ended_inside + 1), fclose(f);
    struct Done { const std::string& path; ~Done() { remove(path.c_str()); } } done_marker{running};
    pcache_init();
    // (relink, lean pipelines: the notes' graphics pipelines are left to their first draw; only the compute programs are read
    // and their pipelines made here)
    const bool lean = pipelines_lean();
    std::vector<std::unique_ptr<PipeNote>> notes;
    if (!lean) notes = pipe_notes_read();
    std::unordered_set<uint64_t> wanted;
    for (const auto& n : notes) wanted.insert(n->prog);
    size_t listed_here = 0, listed_shipped = 0;
    if (lean) {   // (the compute programs play has used, not every kept one: compute_used_note; and the shipped list's)
        {
            std::lock_guard lk(computes_used_mu);
            computes_used_load();
            wanted = computes_used;
        }
        listed_here = wanted.size();
        static const bool no_shipped = getenv("RX_VK_NO_SHIPPED_COMPUTES") != nullptr;
        if (!no_shipped) {
            std::unordered_set<uint64_t> shipped;
            computes_list_add(computes_list_path("relink_computes_shipped.bin"), shipped);
            listed_shipped = shipped.size();
            wanted.insert(shipped.begin(), shipped.end());
        }
    }
    std::vector<Program*> computes;
    const size_t programs_n = program_cache_load_some(wanted, [&](Program& p) {
        ensure_layout(p);
        if (p.compute) computes.push_back(&p);
    }, !lean);
    const double read_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (lean) {
        std::unordered_set<uint64_t> found;
        for (const Program* p : computes) found.insert(p->key);
        logf("vk: compute lists: %zu programs (%zu shipped, %zu from this PC's play), %zu of them in the program cache: their %zu "
             "pipelines made before the game, the others' at their first dispatch%s", wanted.size(), listed_shipped, listed_here,
             found.size(), computes.size(), getenv("RX_VK_NO_SHIPPED_COMPUTES") ? " (RX_VK_NO_SHIPPED_COMPUTES: the shipped list not read)" : "");
    }
    // (lean: the noted graphics pipelines into the driver's own cache first, once a driver, build and program. Its pipelines and
    // the compute ones below are one count on the start-up screen and in the builder's progress: as two, the percentage went
    // back from 99 to 0 between them, and a first start's dawn went back with it. The compute programs are read first for it)
    const size_t warmed = lean ? warm_run(progress, computes.size()) : 0;
    const size_t total = notes.size() + computes.size();
    // (the workers' state on the heap: a start whose last pipelines never finish (two of 4926 on an RX 5700 XT, "stuck at
    // 99%", Discord 11-08) goes on to the game after PRE_STALL_S seconds with no progress, the stragglers left running; it is
    // let go then, never freed. RX_VK_PRECOMPILE_STALL_S=<s>, 0: wait for all, as before)
    struct Pre {
        std::vector<std::unique_ptr<PipeNote>> notes;
        std::vector<Program*> computes;
        std::atomic<size_t> next{0}, done{0};
        std::mutex mu;
        bool abandoned = false;
    };
    const size_t graphics_n = notes.size(), computes_n = computes.size();
    Pre* pre = new Pre;
    pre->notes = std::move(notes), pre->computes = std::move(computes);
    auto& next = pre->next;
    auto& done = pre->done;
    const auto worker = [pre, total] {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        auto& notes = pre->notes;
        auto& computes = pre->computes;
        for (;;) {
            if (prepare_stopped()) return;
            {   // (relink: the game went on without them: the others stop too, and leave the GPU and CPU to it)
                std::lock_guard lk(pre->mu);
                if (pre->abandoned) return;
            }
            const size_t i = pre->next++;
            if (i >= total) return;
            if (i < computes.size()) {
                Program& p = *computes[i];
                VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                ci.stage.module = p.stages[0].module;
                ci.stage.pName = "main";
                pin_subgroup(ci.stage, p.workgroup[0]);
                ci.layout = p.layout;
                VkPipeline made = VK_NULL_HANDLE;
                if (guarded_compute(pcache, &ci, &made) != VK_SUCCESS) made = VK_NULL_HANDLE;
                std::lock_guard lk(pre->mu);
                if (!pre->abandoned) p.compute_pipeline = made;
                else if (made) vkDestroyPipeline(dev.device, made, nullptr);   // (the game went on: it builds its own)
            } else {
                PipeNote& n = *notes[i - computes.size()];
                const Program* p = program_find(n.prog, n.variant);
                if (p) {
                    const VkPipeline made = build_graphics(*p, n.s, note_targets(n), (VkPrimitiveTopology)n.topo);
                    std::lock_guard lk(note_mu);
                    auto it = pipelines.find(n.key);
                    if (it == pipelines.end()) {
                        auto e = std::make_unique<PipeEntry>();
                        e->pipeline = made;
                        e->prog.store(p, std::memory_order_release);   // (see graphics_pipeline: whose pipeline it is)
                        e->ready.store(true, std::memory_order_release);
                        pipelines.emplace(n.key, std::move(e));
                    }
                }
            }
            pre->done++;
        }
    };
    std::vector<std::thread> threads;
    const unsigned cpus = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned i = 0; i < std::min(cpus, 64u); i++) threads.emplace_back(worker);
    static const int stall_s = getenv("RX_VK_PRECOMPILE_STALL_S") ? atoi(getenv("RX_VK_PRECOMPILE_STALL_S")) : 30;
    size_t last_done = 0;
    auto last_change = std::chrono::steady_clock::now();
    bool gone_on = false;
    while (done.load() < total && !prepare_stopped()) {
        if (progress) progress(warmed + done.load(), warmed + total);
        Sleep(100);
        if (next.load() >= total + threads.size()) break;
        const auto now = std::chrono::steady_clock::now();
        if (done.load() != last_done) last_done = done.load(), last_change = now;
        // (relink, issue 80: whether or not every pipeline was handed out. With every worker inside the driver none was, and the
        // start waited for good)
        else if (stall_s > 0 && now - last_change > std::chrono::seconds(stall_s)) {
            std::lock_guard lk(pre->mu);
            pre->abandoned = true;
            gone_on = true;
            logf("vk: precompile: %zu of %zu done, nothing finished for %d s: the game starts, the rest are made in play", done.load(),
                 total, stall_s);
            break;
        }
    }
    if (gone_on) {
        for (std::thread& t : threads) t.detach();
    } else {
        for (std::thread& t : threads) t.join();
        delete pre;
        pipeline_cache_slim();   // (relink: every pipeline the cache was read for is made)
    }
    logf("vk: precompile: %zu programs, %zu graphics and %zu compute pipelines in %.0f ms (%.0f ms of it the caches opened, the notes "
         "and programs read)", programs_n, graphics_n, computes_n,
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), read_ms);
}

// (relink) The pipelines the notes predict for the programs no note names (vk_predict.cpp), after a prepare that prepared: each built
// once for the driver's own cache and let go. Not into ours (mkw_vk_pipelines.bin, read whole at every start: ~40000 more pipelines made
// it 1.7 GB), nor kept: each program is borrowed from relink_programs.bin for its pipelines and let go
void predict_run(void (*progress)(size_t done, size_t total)) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<std::unique_ptr<PipeNote>> notes = pipe_notes_read();
    if (notes.empty() || getenv("RX_VK_NO_PREDICT")) return;
    size_t programs = 0;
    const std::vector<std::unique_ptr<PipeNote>> predicted = predict_pipelines(notes, &programs);
    std::unordered_map<uint64_t, std::vector<const PipeNote*>> by_program;
    for (const auto& n : predicted) by_program[n->prog].push_back(n.get());
    std::vector<uint64_t> keys;
    for (const auto& kv : by_program) keys.push_back(kv.first);
    std::atomic<size_t> next{0}, done{0}, made{0};
    // (relink) A time budget, RX_VK_PREDICT_S (90 s by default; 0: none): once #1051 let the prediction find its programs again it
    // predicted ~28000 pipelines, and a first start after an update spent many minutes on them before the game's first frame (20 with
    // the CPU busy). Those not built by then are built in play when first drawn, as before the prediction worked
    static const int budget_s = getenv("RX_VK_PREDICT_S") ? atoi(getenv("RX_VK_PREDICT_S")) : 90;
    std::atomic<bool> out_of_time{false};
    const auto worker = [&] {
        ub_thread_off();   // (a background thread: see ub_thread_off)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);   // (as prepare_run's)
        driver_cache_only = true;
        for (;;) {
            if (prepare_stopped()) return;
            if (budget_s > 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(budget_s)) {
                out_of_time = true;
                return;
            }
            const size_t i = next++;
            if (i >= keys.size()) return;
            const auto& list = by_program[keys[i]];
            program_cache_borrow(keys[i], [&](Program& p) {
                ensure_layout(p);
                for (const PipeNote* n : list)
                    if (n->variant == p.variant)
                        if (const VkPipeline pl = build_graphics(p, n->s, note_targets(*n), (VkPrimitiveTopology)n->topo))
                            vkDestroyPipeline(dev.device, pl, nullptr), made++;
            });
            done += list.size();
        }
    };
    std::vector<std::thread> threads;
    const unsigned cpus = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned i = 0; i < prepare_threads(); i++) threads.emplace_back(worker);
    while (next.load() < keys.size() + threads.size() && !prepare_stopped() && !out_of_time.load()) {
        if (progress) progress(done.load(), predicted.size());
        Sleep(100);
    }
    for (std::thread& t : threads) t.join();
    logf("vk: predicted %zu pipelines from %zu notes for %zu program variants no note names: %zu made in %.1f s%s", predicted.size(),
         notes.size(), programs, made.load(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
         out_of_time.load() ? " (the time budget, RX_VK_PREDICT_S, ran out: the rest are built in play)" : "");
}

// (relink) Compute pipelines made ahead. A dispatch's compute pipeline was made where the dispatch was, the first time: 0.5-6 s each
// on NVIDIA from a cold cache (hs224: 60 of them, 24.6 s), the player's multi-second freezes in places never seen before; a dispatch is
// never left out (its work is not done again). The game sets its programs' shaders long before it dispatches them (the engine's and the
// foliage's compute shaders at load): each compute program set is translated and its pipeline made on a worker then (vk_jobs), into
// the pipeline cache kept on disk. RX_VK_EAGER_COMPUTE=0: off. Both paths go through make_compute_pipeline: one build a program,
// the dispatch waits for one under way instead of starting another
std::mutex compute_mu;
std::condition_variable compute_cv;
std::unordered_set<const Program*> compute_building;
std::atomic<int64_t> st_eager_compute{0}, st_eager_compute_ms{0};
bool make_compute_pipeline(Program& p, bool eager) {
    {
        std::unique_lock lk(compute_mu);
        if (eager) {
            if (p.compute_pipeline || compute_building.count(&p)) return true;
        } else {
            compute_cv.wait(lk, [&] { return !compute_building.count(&p); });
        }
        ensure_layout(p);
        if (p.compute_pipeline) return true;
        compute_building.insert(&p);
    }
    pcache_init();
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = p.stages[0].module;
    ci.stage.pName = "main";
    pin_subgroup(ci.stage, p.workgroup[0]);
    ci.layout = p.layout;
    VkPipeline made = VK_NULL_HANDLE;
    const auto t0 = std::chrono::steady_clock::now();
    compiles_running++;
    const VkResult r = guarded_compute(pcache, &ci, &made);
    compiles_running--;
    compile_note(eager ? "compute (ahead)" : "compute (at its dispatch)", p.key, t0);
    {
        std::lock_guard lk(compute_mu);
        compute_building.erase(&p);
        if (r == VK_SUCCESS && made && !p.compute_pipeline) p.compute_pipeline = made;
    }
    compute_cv.notify_all();
    if (r == VK_SUCCESS) pcache_built_since_save++, compute_used_note(p.key);
    if (eager && r == VK_SUCCESS) {
        st_eager_compute++;
        st_eager_compute_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    return r == VK_SUCCESS;
}
void eager_compute(const nvn_prog& prog) {
    auto s = std::make_unique<NvnState>();
    s->prog = prog;
    s->has_prog = prog.nstage > 0;
    const uint8_t types[8]{};
    const Program* cp = program_pick(*s, ProgramKind::Compute, types);   // (compute: translated right there, under the shaders' lock)
    if (!cp || cp->failed || !cp->ready.load(std::memory_order_acquire) || !cp->compute || taau_key(cp->key)) return;
    make_compute_pipeline(const_cast<Program&>(*cp), true);
    static std::atomic<int> told{0};
    if (const int64_t n = st_eager_compute.load(); n && n % 25 == 0 && told++ < 40)
        logf("vk: compute pipelines made ahead of their first dispatch: %lld (%lld ms of compiles off the frame)", (long long)n,
             (long long)st_eager_compute_ms.load());
}
void eager_compute_request(const nvn_prog* p) {
    static const bool on = !getenv("RX_VK_EAGER_COMPUTE") || atoi(getenv("RX_VK_EAGER_COMPUTE")) != 0;
    if (!on || !p || p->nstage != 1) return;
    const nvn_prog copy = *p;
    jobs_push_background([copy] { eager_compute(copy); });   // (behind the pipelines draws wait on)
}

void dispatch(VkCommandBuffer cmd, NvnState& s, uint32_t x, uint32_t y, uint32_t z, uint64_t indirect) {
    PROF(PROF_DISPATCH);
    handle_memo_next();
    DispatchClock dclock;
    s.grid[0] = indirect ? 1 : x, s.grid[1] = indirect ? 1 : y, s.grid[2] = indirect ? 1 : z;
    if (indirect) {   // (the program may read its grid from c0: the counts as the GPU will see them, best effort)
        uint32_t g[3] = {1, 1, 1};
        if (memory_read_as_submitted(indirect, g, 12)) s.grid[0] = g[0], s.grid[1] = g[1], s.grid[2] = g[2];
    }
    const uint8_t types[8]{};
    const Program* cp = program_for(s, ProgramKind::Compute, types);
    dclock.t_prog = slow_dispatch_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // (a program still being translated is waited for, as a draw's is: a dispatch left out is work never done again.
    // MKW updates its terrain height clipmaps a region at a time as the camera moves, and makes its sky's tables
    // once; a skipped update left holes and flat patches in the terrain, garbage in the sky)
    if (cp && !cp->ready.load(std::memory_order_acquire)) {
        const auto t0 = std::chrono::steady_clock::now();
        jobs_wait(cp->ready);
        st_waits++;
        st_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    if (!cp || cp->failed) {   // (relink) the program's resources, once a control section: what a rejected dispatch reads/writes
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
    if (cp && cp->failed) {   // (relink) which programs' dispatches are left out, once each, with their grids
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
    if ((!cp || cp->failed || slow_network) && upscale_stub(cmd, s, compute_code_key(s))) return void(st_dispatches++);
    if (slow_network) SKIP_HERE();   // (an earlier pass of the network: the stub is its last pass)
    if (!cp || !cp->ready.load(std::memory_order_acquire) || cp->failed) SKIP_HERE();
    {   // (relink) RX_VK_TEX_OF for dispatches: what they sample and store into
        static const bool want = getenv("RX_VK_TEX_OF") || getenv("RX_VK_TEX_DEPTH") || getenv("RX_VK_TEX_GPU");
        if (want) tex_of_note(s, *cp, nullptr, &s);
        global_note(s, *cp, nullptr);
    }
    {   // (relink, RX_VK_SKIP_PROG=<key prefix,...>: those programs' dispatches left out, for diagnosis)
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
    {   // (relink, RX_VK_LOG_PUSH=<key prefix>: once, that program's push constants and the c0 words it reads)
        static const char* want = getenv("RX_VK_LOG_PUSH");
        static bool done;
        char hex[24];
        snprintf(hex, sizeof hex, "%016llx", (unsigned long long)cp->key);
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
    // out until it is ready, as graphics draws are. RX_VK_COMPUTE_WAIT=1: waited for, as before)
    static const bool compute_wait = getenv("RX_VK_COMPUTE_WAIT") != nullptr;
    if (!compute_wait && !p.compute_pipeline && !taau_key(p.key) && p.compute_state.load(std::memory_order_acquire) != 2) {
        int none = 0;
        const bool fresh = p.compute_state.compare_exchange_strong(none, 1);
        if (fresh) {
            { ParLock lk; pcache_init(); }
            const Program* job = &p;
            // (issue 496: one at a time on Intel. RX_VK_COMPUTE_SERIAL=1 or 0 says so on any GPU)
            static const bool serial = getenv("RX_VK_COMPUTE_SERIAL") ? getenv("RX_VK_COMPUTE_SERIAL")[0] == '1'
                                                                     : dev.props.vendorID == 0x8086;
            auto build = [job] {
                Program& q = const_cast<Program&>(*job);
                VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                ci.stage.module = q.stages[0].module;
                ci.stage.pName = "main";
                pin_subgroup(ci.stage, q.workgroup[0]);
                ci.layout = q.layout;
                static const bool noopt = getenv("RX_VK_NOOPT") != nullptr;
                if (noopt) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
                const auto t0 = std::chrono::steady_clock::now();
                VkPipeline made = VK_NULL_HANDLE;
                if (const VkResult r = guarded_compute(pcache, &ci, &made); r != VK_SUCCESS || !made) {
                    static std::atomic<int> told{0};
                    if (told++ < 20) logf("vk: compute pipeline of %016llx not made (%d)", (unsigned long long)q.key, (int)r);
                    q.failed = true;
                } else {
                    q.compute_pipeline = made;
                    compute_used_note(q.key);
                }
                const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
                if (ms >= 200) logf("vk: compute pipeline of %016llx: SLOW compile, %lld ms (on a worker)", (unsigned long long)q.key, (long long)ms);
                st_pipeline_us += ms * 1000;
                st_pipelines++;
                pcache_built_since_save++;
                q.compute_state.store(2, std::memory_order_release);
            };
            if (serial) jobs_push_serial(std::move(build));
            else jobs_push(std::move(build), true);
        }
        // (only the dispatch that started the build waits: later ones while it still compiles are left out at once, not 50 ms
        // each, which stalled every frame by 50 ms per compute program still building)
        const auto t0 = std::chrono::steady_clock::now();
        while (fresh && p.compute_state.load(std::memory_order_acquire) != 2 &&
               std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(50))
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        if (p.compute_state.load(std::memory_order_acquire) != 2) {
            static std::atomic<int> told{0};
            if (told++ < 40) logf("vk: dispatch of %016llx left out: its compute pipeline is still compiling", (unsigned long long)p.key);
            st_pending_skips++;
            return;
        }
        if (p.failed || !p.compute_pipeline) SKIP_HERE();
    }
    ParLock plk(!p.compute_pipeline);
    if (!p.compute_pipeline && !taau_key(p.key)) {   // (a stood-in network is never compiled)
        pcache_init();
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = p.stages[0].module;
        ci.stage.pName = "main";
        pin_subgroup(ci.stage, p.workgroup[0]);
        ci.layout = p.layout;
        {   // (local) RX_VK_NOOPT=1, as for graphics pipelines above
            static const bool noopt = getenv("RX_VK_NOOPT") != nullptr;
            if (noopt) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        }
        const auto t0 = std::chrono::steady_clock::now();
        {   // (EvilOOT, 11-07: a freeze after Load was the driver's compiler on one compute pipeline, here, on the submit's thread)
            static std::atomic<int> told{0};
            if (told++ < 400) logf("vk: compute pipeline of %016llx: compiling", (unsigned long long)p.key);
        }
        if (const VkResult made = guarded_compute(pcache, &ci, &p.compute_pipeline); made != VK_SUCCESS) {
            static std::atomic<int> told{0};
            if (told++ < 20) logf("vk: compute pipeline of %016llx not made (%d)", (unsigned long long)p.key, (int)made);
            p.compute_pipeline = VK_NULL_HANDLE;
            p.failed = true;
            SKIP_HERE();
        }
        {
            const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
            if (ms >= 200) logf("vk: compute pipeline of %016llx: SLOW compile, %lld ms", (unsigned long long)p.key, (long long)ms);
        }
        st_pipeline_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        st_pipelines++;
        pcache_built_since_save++;
        compute_used_note(p.key);
    }
    if (plk.on) par_mutex().unlock(), plk.on = false;
    dclock.t_pipe = slow_dispatch_timing() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    end_pass(cmd);
    static const bool no_grid_patch = getenv("RX_VK_NO_GRID_PATCH") != nullptr;
    grid_patch_want = indirect && !no_grid_patch, grid_patch_at = UINT64_MAX;
    static const bool capture_compute = !getenv("RX_VK_CAPTURE_COMPUTE") || atoi(getenv("RX_VK_CAPTURE_COMPUTE")) != 0;
    // (RX_VK_CAPTURE_COMPUTE_MAX=<bytes>: the largest of them noted. Every one up to 64 KB was 140 to 220 ranges over 22 MB in
    // the village, whose pools' mirror syncs then took 2.68 ms a frame for 0.65: 52.9 -> 46.9 frames a second where the frame
    // has ~2850 draws, hs641 / hs640)
    static const uint64_t compute_max = getenv("RX_VK_CAPTURE_COMPUTE_MAX") ? (uint64_t)std::max(0, atoi(getenv("RX_VK_CAPTURE_COMPUTE_MAX"))) : 64u << 10;
    // (and only the exposure's own passes by default: the sky sums, the picture's RGBL sums and the loop that clears them. A
    // limit of 32 bytes still fixed the steps (2.8 a thousand, hs65_max32) but still noted 591 ranges, 13 MB: every dispatch's
    // 16 and 32 byte blocks sit in the game's transient arena, a page each. RX_VK_CAPTURE_COMPUTE_PROGS=all: every dispatch)
    static const bool capture_all = getenv("RX_VK_CAPTURE_COMPUTE_PROGS") && !strcmp(getenv("RX_VK_CAPTURE_COMPUTE_PROGS"), "all");
    // (RX_VK_CAPTURE_COMPUTE_KEYS=<16 hex digits>,...: those programs' dispatches instead; RX_VK_LOG_COMPUTE=1: each compute
    // program's key the first time it is dispatched, to find the one that reads the exposure's block)
    static const std::vector<uint64_t> capture_keys = [] {
        std::vector<uint64_t> v;
        for (const char* q = getenv("RX_VK_CAPTURE_COMPUTE_KEYS"); q && *q;) {
            char* e = nullptr;
            const uint64_t k = strtoull(q, &e, 16);
            if (e == q) break;
            v.push_back(k);
            q = *e == ',' ? e + 1 : e;
        }
        return v;
    }();
    {
        static const bool log_compute = getenv("RX_VK_LOG_COMPUTE") != nullptr;
        if (log_compute) {
            static std::mutex mu;
            static std::vector<uint64_t> seen;
            std::lock_guard lk(mu);
            if (std::find(seen.begin(), seen.end(), p.key) == seen.end()) {
                seen.push_back(p.key);
                logf("vk: compute program %016llx (%zu so far, present %lld)", (unsigned long long)p.key, seen.size(), (long long)texture_frame_no());
            }
        }
    }
    const bool listed = !capture_keys.empty() && std::find(capture_keys.begin(), capture_keys.end(), p.key) != capture_keys.end();
    // (RX_VK_CAPTURE_COMPUTE_PROGS=exposure: the three passes above alone, which does nothing for the steps, 9.2 a thousand,
    // hs66_expo. Every dispatch by default. RX_VK_CAPTURE_COMPUTE_RB=1: of their buffers only those on a page the game reads
    // GPU results in, which does nothing either, 9.0, hs71_rbonly.
    // WHAT THE CAPTURE DOES FOR THE STEPS (10/11 10:45): it puts pages a submission on the GPU still writes back into the
    // captures, and each capture then waits for that submission (take_capture's late pages). The wait is the cure: with
    // RX_VK_CAPTURE_WAIT=0 the steps are there, everything captured all the same (7.4 and 12.4 a thousand, hs73), and with the
    // wait kept for any two of those pages they are gone (1.8, hs75). So the replay may not run more than one submission
    // ahead of the GPU, and the 1 to 2.5 ms of "memory sync" a frame in heavy views is that wait, not the copies)
    static const bool capture_expo = getenv("RX_VK_CAPTURE_COMPUTE_PROGS") && !strcmp(getenv("RX_VK_CAPTURE_COMPUTE_PROGS"), "exposure");
    static const bool rb_only = getenv("RX_VK_CAPTURE_COMPUTE_RB") && atoi(getenv("RX_VK_CAPTURE_COMPUTE_RB")) != 0;
    (void)capture_all;
    compute_now = capture_compute && (listed || (capture_keys.empty() && (!capture_expo || p.key == kSkySumsProgram || p.key == kSkyRgblProgram ||
                                                                          p.key == 0x6ccba775ea111ee9ull)));
    ui_note_max = compute_max;
    ui_note_rb_only = rb_only;
    build_descriptors(s, p);
    ui_note_rb_only = false;
    ui_note_max = 64u << 10;
    compute_now = false;
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
    static const bool no_order = getenv("RX_VK_NO_DISPATCH_ORDER") != nullptr;
    if (work_since_barrier && !no_order) full_barrier(cmd, "barrier (dispatch)");
    work_since_barrier = true;
    if (p.key == kSkySumsProgram || p.key == kSkyRgblProgram || p.key == 0x6ccba775ea111ee9ull) replay_saw_exposure = true;
    {   // (relink, RX_VK_METER_ZERO=1) OoT's exposure meter: 435a0138b9013040 and 592079442d7129da add a frame's
        // sums into the first 16 bytes of their block, which the game reads and sets back to zero. With the game ahead of the
        // GPU its zeros reach the mirror late, or a write-back lands over them first and the old sums come back: the next
        // frame's are added on top (twice the light). Cleared here right before each, as the sky light's below
        static const bool meter_zero = fast_sync_part("RX_VK_METER_ZERO");
        if (meter_zero && (p.key == 0x435a0138b9013040ull || p.key == 0x592079442d7129daull) && meter_out.buffer && meter_out.size >= 16 &&
            !(meter_out.offset & 3)) {
            vkCmdFillBuffer(cmd, meter_out.buffer, meter_out.offset, 16, 0);
            full_barrier(cmd);
        }
    }
    if ((p.key == kSkySumsProgram || p.key == kSkyRgblProgram) && sky_sums.buffer && !p.stages.empty()) {
        // (relink) OoT's sky light: 6ccba775ea111ee9 clears 7 x 16 bytes of a block from +0x30, then 499dbfd0721d2858 adds the
        // sky probe into 4 x RGB at c0[0x20] + c1[0] (+0x30) + 16 k with float atomics, and ab7699ee7f536755 its RGBL at
        // c0[0x30] + 0x80; the game's CPU reads them back for its exposure. The same 4 KB page holds the frame's constants, which
        // the CPU writes for its next frame while the replay runs a frame or two behind: a mirror sync of that page between the
        // clear and the adds (a new submission part or segment binding the pool) put back the last frame's sums, the adds went
        // on top, and that frame's ambient light came out doubled: the whole scenery a step brighter (+10 of luma, bluish) for
        // one frame, 40-90 times in 150 s on the terrace (hs297, hs303, hs308: all 53 flashes on frames whose sums had not been
        // cleared). Cleared again right before each, they hold this frame's sums only. RX_VK_NO_SKY_ZERO=1: as before
        static const bool off = getenv("RX_VK_NO_SKY_ZERO") != nullptr;
        const uint32_t at = p.key == kSkySumsProgram ? stage_cbuf_word(s, p.stages[0], 1, 0) : 0x80u;
        const uint32_t n = p.key == kSkySumsProgram ? 0x40u : 0x10u;
        if (!off && !(at & 3) && at + n <= sky_sums.size) {
            vkCmdFillBuffer(cmd, sky_sums.buffer, sky_sums.offset + at, n, 0);
            full_barrier(cmd);
        } else if (!off) {
            static std::atomic<int> told{0};
            if (told++ < 4)
                logf("vk: sky light reset skipped for %016llx: offset %#x, window %#llx bytes", (unsigned long long)p.key, at,
                     (unsigned long long)sky_sums.size);
        }
    }
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
    if (dlss_stand_in(cmd, s, p)) return wheel_scene_note(cmd, s, p, true), void(st_dispatches++);   // (relink: the item wheel's scene)
    if (taau_stand_in(cmd, s, p)) return wheel_scene_note(cmd, s, p, true), void(st_dispatches++);
    if (!p.compute_pipeline) SKIP_HERE();
    dump_ssbos(cmd, p, "before");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.compute_pipeline);
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
        StatsQuery stats_query(cmd, p.key, indirect, x, y, z);   // (EvilOOT, RX_VK_DRAW_STATS)
        {   // (RX_VK_GPU_DUMP=1 with RX_VK_DRAW_STATS_PROGS: the wanted dispatches' storage buffers as the GPU holds them after)
            static const bool gpu_dump_on = getenv("RX_VK_GPU_DUMP") != nullptr;
            if (gpu_dump_on && stats_query.cmd && stats_prog_wanted(p.key)) stats_query.state = &s, stats_query.prog = p.key;
        }
        if (indirect) {
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
            vkCmdDispatch(cmd, x, y, z);
        }
    }
    wheel_scene_note(cmd, s, p, false);   // (relink: the game's own gameplay upscale, for the item wheel)
    if (dump_on() && dump_ssbo_wanted(p.key)) dump_ssbos(cmd, p, "after");
    if (peek_prog(p.key)) peek_draw(cmd, p.key, false);   // (relink: RX_VK_PEEK_PROG, what the dispatch left)
    if (gprof_on()) {
        char label[96];
        snprintf(label, sizeof label, "dispatch %s prog %016llx", indirect ? "indirect" : "", (unsigned long long)p.key);
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

// (relink) In the parallel replay's dry pass, in stream order on the replay thread: the images this draw or dispatch
// writes marked as the GPU's (texture_written) before any worker runs. Workers record out of order, and that mark was
// made as each reached its draw: one ahead of it sampled a render target or compute output as never written and
// uploaded stale guest memory over it, an upload that runs after the earlier segment's work. Targets change address
// and size as the camera moves: flashes of a lit scene gone black or white in motion (22 per 1000 frames moving, 45
// against 2 with serial replay). RX_VK_NO_DRY_MARKS=1: as before
void dry_mark_written(NvnState& s, bool compute, bool first) {
    static const bool off = getenv("RX_VK_NO_DRY_MARKS") != nullptr;
    if (off) return;
    if (!compute) {
        static thread_local decltype(s.rt) last;
        if (!first && !std::memcmp(&last, &s.rt, sizeof last)) return;
        std::memcpy((void*)&last, (const void*)&s.rt, sizeof last);
        const Targets at = targets(s);
        for (int i = 0; i < at.n; i++)
            if (at.color[i]) texture_written(at.color[i]);
        if (at.depth) texture_written(at.depth);
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
                    texture_written(tex);
            }
        }
}

// (relink, 12/11) The dry pass, at a clear: the target it clears is written too. dry_mark_written is called at draws and
// dispatches only, and a target the game clears and never draws into had no mark: it was written for the segments only once
// the segment that holds the clear recorded it. OoT clears a 768 x 432 target (no draw) and samples it two passes later; its
// storage is the transient heap's, another image's (1536 x 864) earlier in the frame, finalized every frame. With a segment
// boundary between that clear and the pass that samples it, the later segment's worker found it as the game left it last
// frame (drawn, finalized, its memory written by the CPU since) whenever it got there first, uploaded it from the game's
// memory over the clear, and the lighting that reads it went out: the whole scene black for a frame but the lights' own
// glow, 4 to 24 times in 115 s at Hyrule Castle and in bursts while the boundary stayed there (hs1570-hs1586, py/darkflash.py;
// found with the passes' trace: the pass after the boundary sampled 3 drawn textures, not 4). None with the replay on one
// thread, with one segment, or with 2 or 3 segments at most, whose boundaries fall elsewhere. index: the colour target,
// -1: the depth and stencil. RX_VK_NO_DRY_CLEAR_MARKS=1: as before
void dry_mark_cleared(NvnState& s, int index) {
    static const bool off = getenv("RX_VK_NO_DRY_MARKS") != nullptr || getenv("RX_VK_NO_DRY_CLEAR_MARKS") != nullptr;
    if (off) return;
    const Targets at = targets(s);
    if (index < 0) {
        if (at.depth) texture_written(at.depth);
    } else if (index < at.n && at.color[index] && at.color_view[index]) {
        texture_written(at.color[index]);
    }
}

void draw_stats(int64_t* draws, int64_t* skipped, int64_t* dispatches, int64_t* pipes) {
    *draws = st_draws, *skipped = st_skipped, *dispatches = st_dispatches, *pipes = st_pipelines.load();
}
void draw_stats2(int64_t* passes, int64_t* mesh, double* pipeline_ms) {
    *passes = st_passes, *mesh = st_mesh, *pipeline_ms = st_pipeline_us.load() / 1000.0;
}
void draw_stats3(int64_t* pending_skips, int64_t* waits, double* wait_ms) {
    *pending_skips = st_pending_skips, *waits = st_waits, *wait_ms = st_wait_ms;
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
    {   // (relink, issues 6, 7 and 27) RX_VK_RECORD_WORDS=<cbuf>:<word>,...: those uniform words, as the draw's stages have them
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
// (relink, 12/11) the same into a file: eight frames of the Castle are 3300 lines, more than the host's log queue holds
// (the log stopped in the middle of the first frame, hs1581)
void pass_traces_write(const char* path, uint64_t lo, uint64_t hi) {
    FILE* o = fopen(path, "w");
    if (!o) return;
    std::lock_guard lk(trace_mu);
    for (uint64_t f = lo; f <= hi; f++) {
        auto it = pass_traces.find(f);
        if (it == pass_traces.end()) continue;
        std::vector<std::pair<uint64_t, std::string>> v = it->second;
        std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        fprintf(o, "frame %llu (%zu)%c", (unsigned long long)f, v.size(), 10);
        for (const auto& [key, l] : v) fprintf(o, "  [%llu] %s%c", (unsigned long long)(key >> 24), l.c_str(), 10);
    }
    fclose(o);
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

/// The cache's content to `path` (through a temporary file beside it). False: not written
bool pipeline_cache_write(const std::string& path) {
    try {
        size_t n = 0;
        if (vkGetPipelineCacheData(dev.device, pcache, &n, nullptr) != VK_SUCCESS || !n) return false;
        std::vector<uint8_t> data(n);
        if (vkGetPipelineCacheData(dev.device, pcache, &n, data.data()) != VK_SUCCESS) return false;
        const std::string tmp = path + ".tmp";
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) return false;
        const bool ok = fwrite(data.data(), 1, n, f) == n;
        fclose(f);
        return ok && MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
    } catch (const std::exception& e) {
        logf("vk: pipeline cache save failed: %s", e.what());
        return false;
    }
}

/// (relink: see pcache_loaded) After the start's precompile, no other thread building pipelines yet: the big cache goes, a
/// small one takes its place
void pipeline_cache_slim() {
    static const bool keep = getenv("RX_VK_KEEP_PIPELINE_CACHE") != nullptr;
    if (keep || !pcache || pcache_slimmed || pcache_loaded < (64u << 20)) return;
    size_t now = 0;
    if (vkGetPipelineCacheData(dev.device, pcache, &now, nullptr) != VK_SUCCESS) return;
    // (the file made again whole when the precompile built what it lacked, or what was built since has grown: then nothing is
    // left over for the small cache to start from)
    const bool fold = now != pcache_loaded || pcache_new.size() >= (64u << 20);
    if (fold) {
        if (!pipeline_cache_write(pcache_path)) return;   // (not written: kept as it is, for the saves to try again)
        remove((pcache_path + ".new").c_str());
        pcache_new.clear();
    }
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = pcache_new.size();
    ci.pInitialData = pcache_new.empty() ? nullptr : pcache_new.data();
    VkPipelineCache small = VK_NULL_HANDLE;
    if (vkCreatePipelineCache(dev.device, &ci, nullptr, &small) != VK_SUCCESS) return;
    vkDestroyPipelineCache(dev.device, pcache, nullptr);
    pcache = small;
    pcache_slimmed = true;
    pcache_built_since_save = 0;
    logf("vk: pipeline cache: %zu MB let go after the precompile%s; what play builds is kept in %s.new (%zu KB now)", now >> 20,
         fold ? ", the file written again whole first" : "", pcache_path.c_str(), pcache_new.size() >> 10);
    pcache_new = {};
}

void pipeline_cache_save(bool force) {
    if (!pcache || (!force && pcache_built_since_save < 32)) return;
    if (pcache_slimmed && !pcache_built_since_save) return;   // (nothing built since the small cache was last written)
    pcache_built_since_save = 0;
    // (on a thread of its own unless forced, at exit: ~100 MB fetched and written took the replay thread 100+ ms at
    // each scene change; the cache is internally synchronized, pipelines go on being built meanwhile)
    static std::atomic<bool> busy{false};
    if (busy.exchange(true)) {
        if (!force) return;
        while (busy.load()) Sleep(1);   // (the one under way, then this one)
        busy = true;
    }
    auto save = [] {
        struct Done { ~Done() { busy = false; } } done;
        // (the small cache to its own file; a cache never let go is the whole of it, and what <file>.new had is in it)
        if (pcache_slimmed) pipeline_cache_write(pcache_path + ".new");
        else if (pipeline_cache_write(pcache_path)) remove((pcache_path + ".new").c_str());
    };
    if (force) save();
    else {
        try {
            std::thread(save).detach();
        } catch (const std::exception& e) {
            busy = false;
            logf("vk: could not start pipeline cache save: %s", e.what());
        }
    }
}

} // namespace mkwvk

// (relink) the host's call when the game sets a compute program's shaders (host/nvn_native.c set_shaders, looked up by name in
// host/rx_backend.c, so the back end's interface (vk_api.h, part of the shader cache's stamp) stays as it is)
extern "C" __declspec(dllexport) void relink_program_set(const nvn_prog* p) { mkwvk::eager_compute_request(p); }
