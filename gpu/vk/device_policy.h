// (hookshot) What the renderer decides by the PC it runs on, in one place: the GPU's vendor and driver, its memory, the
// PC's RAM and CPU count, the OS. DeviceFacts is what create_device (vk_main.cpp) learns of the PC, once, before the device
// is made. device_policy() turns the facts into the defaults, with no switch read: what each rule gives this PC unset.
// device_policy_log() prints both at the start ("vk: policy:" lines), so one log shows everything a PC got for being what
// it is. Plain code with no Vulkan calls (the headers for the enums only), so tests/test_device_policy.py runs it on
// made-up PCs and pins every field on each.
//
// The rules are moving here in steps. A field marked "read" is what its site uses: the default by the facts, then its
// switch, read once by device_policy_env() with one rule for every switch (there). One marked "site" is computed here by
// the site's rule, for the log and the test, while the site still decides with its own switch read; the step named moves
// it: 2, the sites whose rule is already a tested header of its own; 3, the sizes; 4, the Intel and AMD rules of the
// draws; 5, device creation; 7, the mirrors' family and what is known only after the device is made. (Step 6, the shader
// translation's, is done: vk_shader.cpp reads its fields here. That file is in the translator's stamp and this one is
// not, so a rule changed here reaches the program cache only through the values translator_stamp folds: a field the
// translation reads must be folded there, and tests/test_device_policy.py checks that each one is.)
// The sizes that go by the driver's allowance are worked out from the allowance at the start (allowed_at_start); their
// sites ask the same rule (policy_rule) with the allowance at their own first call, after the renderer has allocated, so
// a site's number can be lower than the log's.
// docs/development/device-policy.md is the guide.
#pragma once

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vulkan/vulkan_core.h>

#ifndef HOOKSHOT_ENV_H   // (env_value_off: the words that say off)
#include "compat/hookshot_env.h"
#endif
#include "astc_bc_policy.h"
#include "auto_settings.h"
#include "fallback_size_policy.h"
#include "gpu_arch.h"
#include "gpu_clock.h"
#include "intel_compiles.h"
#include "mesh_mode.h"
#include "mirror_budget.h"
#include "retired_give_back.h"
#include "target_pool.h"
#include "texture_budget.h"
#include "threaded_rendering.h"
#include "vram_compact.h"

namespace mkwvk {

struct DeviceFacts {
    // who
    uint32_t vendor_id = 0, device_id = 0, driver_version = 0;
    VkDriverId driver_id{};
    VkPhysicalDeviceType device_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    const char* driver_info = "";   // (VkPhysicalDeviceDriverProperties::driverInfo: AMD's Windows driver names its compiler here)
    const char* device_name = "";   // (the Steam Deck's APU, RTX cards, and fallback RDNA 4 detection by name)
    bool windows = false;           // (the build's OS)
    bool steam_deck_env = false;    // (SteamDeck=1, as auto_profile reads it)
    // memory
    uint64_t vram_heap = 0;         // (the largest device-local heap, RX_VK_TEST_VRAM_MB applied: device_vram_bytes)
    // (what the driver allows the process on its device-local heaps before the renderer allocates, as vram_usage would answer
    // then: VK_EXT_memory_budget's budgets, or without the extension VMA's own estimate, 80% of the heaps; 0: not known)
    uint64_t allowed_at_start = 0;
    uint64_t ram = 0;               // (physical RAM; 0: not known)
    bool rebar = false;             // (a discrete card with a DEVICE_LOCAL | HOST_VISIBLE memory type on its VRAM heap; no rule reads it yet)
    // CPU
    unsigned cpus_all = 0, cpus_usable = 0;   // (hardware_concurrency; usable_cpus, the affinity mask's count)
    unsigned cores_usable = 0;   // (hookshot: usable_cores, the physical cores among those, SMT siblings counted once; 0 unknown)
    // what Vulkan says the card can do, whatever this run turns on
    bool mesh_shader = false;       // (task and mesh shaders: dev.mesh_capable)
    bool mesh_queries = false;      // (their counts in pipeline statistics, as this run has them: off with the Compute mode)
    bool buffer_device_address = false;
    bool float32_atomic_add = false;   // (VK_EXT_shader_atomic_float's shaderBufferFloat32AtomicAdd)
    bool coop_matrix_ext = false;      // (VK_KHR_cooperative_matrix is listed)
    bool nv_checkpoints_ext = false;   // (VK_NV_device_diagnostic_checkpoints is listed)
    bool storage_io16 = false;         // (storageInputOutput16: RADV lacks it on GCN 1-4)
    bool storage_buffer_array_dynamic_indexing = false, robust_image_access = false;
    uint32_t subgroup_size = 32, subgroup_max = 32;
    VkShaderStageFlags sg32_stages = 0;   // (stages that can be pinned to 32 lanes; 0 with RX_VK_NO_SUBGROUP_PIN, as dev.sg32_stages)
    // derived, so no rule spells them twice
    bool integrated() const { return device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU; }
    bool discrete() const { return device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU; }
    bool nvidia() const { return vendor_id == 0x10DE; }
    bool amd() const { return vendor_id == 0x1002; }
    bool intel() const { return vendor_id == 0x8086; }
    bool amd_win() const { return windows && driver_id == VK_DRIVER_ID_AMD_PROPRIETARY; }
    bool radv() const { return driver_id == VK_DRIVER_ID_MESA_RADV; }
    bool radv_gcn() const { return radv() && !storage_io16; }
    bool deck_apu() const { return gpu_clock::deck_apu(device_name); }
    bool handheld() const { return (steam_deck_env && integrated()) || deck_apu(); }   // (auto_profile's hw.handheld)
    GpuArch arch() const { return classify_gpu_arch(vendor_id, device_id, device_name, radv_gcn()); }
    const char* arch_name() const { return gpu_arch_name(arch()); }
    bool rdna1() const { return arch() == GpuArch::AmdRdna1; }
    bool rdna2() const { return arch() == GpuArch::AmdRdna2; }
    bool rdna3() const { return arch() == GpuArch::AmdRdna3; }
    bool rdna4() const { return arch() == GpuArch::AmdRdna4; }
};

// The rules that are a number or a yes from one or two facts, each as its site has it. device_policy() calls them with
// the facts at the start; a site that has to ask later (the allowance at its first call) calls the rule itself.
namespace policy_rule {

inline bool windows_vega_compat(const DeviceFacts& facts) {
    return facts.windows && facts.amd() && facts.device_id == 0x15E7;
}


constexpr uint64_t kMB = 1ull << 20, kGB = 1ull << 30;

/// RX_VK_NATIVE_FADD unset: the drivers the GPU's own float add was run on (vk_main.cpp native_fadd_wanted has the runs).
/// NVIDIA's driver; Intel's Windows driver on Meteor Lake and Arrow Lake (0x7Dxx) from 101.9175, Battlemage (0xE2xx) from
/// 101.9154, Panther Lake (0xB0xx) from 101.9150 and Lunar Lake (0x64xx: 0x6420, 0x64A0, 0x64B0) from 101.8991 (with the loop
/// Arc 140Vs lost the device, at 499dbfd0 among other programs: Hookshot issues 844, 878, 1027 and 1225). Intel's Windows driverVersion is
/// 101 << 14 | build on every family; Mesa's Intel driver numbers its versions otherwise and is never listed
inline bool native_fadd_listed(VkDriverId driver, uint32_t device, uint32_t version) {
    if (driver == VK_DRIVER_ID_NVIDIA_PROPRIETARY) return true;
    if (driver != VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS) return false;
    return ((device & 0xFF00u) == 0x7D00u && version >= (101u << 14 | 9175u)) ||
           ((device & 0xFF00u) == 0xE200u && version >= (101u << 14 | 9154u)) ||
           ((device & 0xFF00u) == 0xB000u && version >= (101u << 14 | 9150u)) ||
           ((device & 0xFF00u) == 0x6400u && version >= (101u << 14 | 8991u));
}

/// Default depth-bias margin for EQUAL colour passes after a Z-prepass (vk_draw.cpp dynamic_state).
/// Intel's Windows driver needs 64 units on older/integrated families where depth and colour programs do not match to the
/// last bit (issue 17, Arc B390). On discrete Battlemage (0xE2xx, e.g. Arc B570 / B580) the margin's LESS_OR_EQUAL lets
/// alpha-tested Z-prepass cutouts (fallen leaves, Link's hair, distant tree cards) pass depth where the prepass
/// discarded them, rendering opaque rectangles; keep it 0 there.
inline float equal_depth_bias(VkDriverId driver, uint32_t device) {
    if (driver != VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS) return 0.0f;
    if ((device & 0xFF00u) == 0xE200u) return 0.0f;
    return 64.0f;
}

/// The mirrors' budget (vk_memory.cpp mirror_budget): a third of the card, at most 1700 MB, on a card of 6.5 GB or less,
/// to hold open-area peaks without falling back to PCIe reads; else mirror_budget.h's rule
inline uint64_t mirror_budget(uint64_t vram) {
    if (vram && vram <= 6 * kGB + 512 * kMB) return std::min<uint64_t>(1700 * kMB, vram / 3);
    return mirror_rule::auto_bytes(vram);
}

/// An integrated GPU's texture budget from its share of the allowance (texture_budget.h integrated_bytes) and the PC's
/// RAM, which it draws from (vk_texture.cpp texture_frame): a third of the RAM is left to the game and the system (8 GB at
/// least, 16 at most), then a cap by the RAM (1.5 GB at 8, 2 at 12, 2.5 at 16, 3.5 at 24, 4 at 32, 6 above; when ASTC is
/// decoded on the CPU, 6 GB at 24 and 8 above, as RGBA8 textures take ~4x the space of BC3), 1 GB at least, and never
/// more than the share. RAM not known: taken as 8 GB
inline uint64_t integrated_reserved(uint64_t ram) {   // (what of the RAM is left to the game, its targets and the system)
    return std::max<uint64_t>(std::min<uint64_t>(ram / 3, 16 * kGB), 8 * kGB);
}
inline uint64_t integrated_texture_budget(uint64_t share, uint64_t ram, bool astc_on_cpu = false) {
    if (!ram) ram = 8 * kGB;
    const uint64_t reserved = integrated_reserved(ram);
    const uint64_t available = ram > reserved ? ram - reserved : 0;
    uint64_t b = std::min(available, share);
    const uint64_t cap = ram <= 8 * kGB ? 1536 * kMB : ram <= 12 * kGB ? 2048 * kMB : ram <= 16 * kGB ? 2560 * kMB
                       : astc_on_cpu ? (ram <= 24 * kGB ? 6144 * kMB : 8192 * kMB)
                       : ram <= 24 * kGB ? 3584 * kMB : ram <= 32 * kGB ? 4096 * kMB : 6144 * kMB;
    b = std::min(b, cap);
    b = std::max<uint64_t>(b, 1024 * kMB);
    return std::min(b, share);
}

/// The texture cache's budget at the first look (texture_frame): texture_budget.h by the card, an integrated GPU's by the RAM too
inline uint64_t texture_budget(uint64_t allowed, uint64_t vram, uint64_t ram, bool integrated, bool astc_on_cpu = false) {
    if (!integrated) return texture_rule::dedicated_bytes(allowed, vram, ram);
    return integrated_texture_budget(texture_rule::integrated_bytes(allowed, vram, astc_on_cpu), ram, astc_on_cpu);
}

/// Whether the card counts as small for the textures' top levels (vk_texture.cpp low_vram_texture_mips, with Minimum's
/// settings at the site): allowed 4.5 GiB or less, or an integrated GPU beside 16 GiB of RAM or less (an i3-1115G4's UHD
/// Graphics was allowed 7.4 GB of 16 while Windows paged the game out)
inline bool small_vram_textures(uint64_t allowed, bool integrated, uint64_t ram) {
    return (allowed && allowed <= 4 * kGB + 512 * kMB) || (integrated && ram && ram <= 16 * kGB);
}

/// The write-back shadows' VRAM at most (vk_memory.cpp shadow_room): an eighth of the allowance, 512 MB when it is not known,
/// and 512 MB at most on an integrated GPU, whose VRAM is the system's RAM
inline uint64_t wb_shadow_cap(uint64_t allowed, bool integrated) {
    const uint64_t eighth = allowed ? allowed / 8 : 512 * kMB;
    return integrated ? std::min<uint64_t>(eighth, 512 * kMB) : eighth;
}

/// What the spare capture buffers may hold (vk_memory.cpp cap_spare_max): 32 MB where the driver allows 6.5 GB or less
/// (after a load the retired buffers, 40-60 MB each, stayed as spares for good), no limit elsewhere or with the allowance unknown
inline uint64_t cap_spare_max(uint64_t allowed) {
    return allowed && allowed <= 6 * kGB + 512 * kMB ? 32 * kMB : ~0ull;
}

/// The idle staging arenas kept (vk_texture.cpp texture_frame): 128 MB where the driver allows 6.5 GB or less (with resizable
/// BAR the arenas are VRAM: an RTX 4050 Laptop held 256 MB of its 5.7 GB idle), else 256 MB
inline uint64_t staging_keep(uint64_t allowed) {
    return (allowed && allowed <= 6 * kGB + 512 * kMB ? 128 : 256) * kMB;
}

/// VMA's blocks by the card's heap (vk_vma.cpp): 32 MB large-heap blocks and 16 MB render target blocks on a card of 8 GB
/// or less, to halve the fragmentation there; 64 and 32 MB above
inline uint64_t vma_block(uint64_t heap) { return heap <= vram_compact::GLOBAL_MIN_GRAPHICS_REQ ? 32 * kMB : 64 * kMB; }
inline uint64_t target_block(uint64_t heap) { return heap <= vram_compact::GLOBAL_MIN_GRAPHICS_REQ ? 16 * kMB : 32 * kMB; }

/// The background workers (vk_jobs.cpp start): 6 from 16 hardware threads, else a third of them, 2 to 4
inline unsigned workers(unsigned cpus) { return cpus >= 16 ? 6u : std::clamp(cpus / 3, 2u, 4u); }
/// The capture's copy helpers (vk_memory.cpp capture_copy_helpers; none with RX_VK_LOW_SPEC): four where the threads are spare
inline unsigned capture_helpers(unsigned cpus) { return cpus >= 12 ? 4u : 2u; }
/// The threads that scan and unswizzle a big texture (vk_texture.cpp upload_all_zero, upload_inner) and gather a big sparse
/// one's pages (guest_bytes): half the CPUs, 1 to 8 and 1 to 6
inline unsigned texture_threads(unsigned cpus) { return std::clamp(cpus / 2, 1u, 8u); }
inline unsigned texture_gather_threads(unsigned cpus) { return std::clamp(cpus / 2, 1u, 6u); }
/// The preparation's workers in play (vk_draw.cpp prepare_run): a quarter of the CPUs, 2 to 4, the game's own threads before
/// them, and never more than the process may use
inline unsigned program_workers_in_play(unsigned cpus, unsigned usable) {
    return std::min(std::clamp(std::max(1u, cpus) / 4, 2u, 4u), usable);
}

/// Whether the textures' squeeze at the count line takes a quarter of the excess and not all of it (texture_frame): where
/// the driver allows under 8 GB, each frame's own allowance
inline bool squeeze_damped(uint64_t allowed) { return allowed < vram_compact::GLOBAL_MIN_GRAPHICS_REQ; }

/// A number switch's value: decimal digits only (no sign, no space, nothing after), from lo to hi. false for anything
/// else, which device_policy_env ignores and says so: atoi read "off" as 0 and a negative size as a huge one
inline bool parse_number(const char* text, uint64_t lo, uint64_t hi, uint64_t& out) {
    if (!text || !*text) return false;
    uint64_t v = 0;
    for (const char* c = text; *c; c++) {
        if (*c < '0' || *c > '9') return false;
        const uint64_t digit = (uint64_t)(*c - '0');
        if (digit > hi || v > (hi - digit) / 10) return false;   // (past hi, whatever the type would wrap to)
        v = v * 10 + digit;
    }
    if (v < lo) return false;
    out = v;
    return true;
}
/// The same for a number with a fraction (RX_VK_EQ_BIAS): digits, at most one point with digits on both sides, 0 to hi
inline bool parse_decimal(const char* text, double hi, double& out) {
    if (!text || *text < '0' || *text > '9') return false;
    double v = 0, place = 0;
    for (const char* c = text; *c; c++) {
        if (*c == '.') {
            if (place != 0 || c[1] < '0' || c[1] > '9') return false;
            place = 1;
            continue;
        }
        if (*c < '0' || *c > '9') return false;
        if (place != 0) place /= 10, v += (*c - '0') * place;
        else v = v * 10 + (*c - '0');
        if (v > hi) return false;
    }
    out = v;
    return true;
}

}  // namespace policy_rule

/// The fields a switch can set (device_policy_env), for DevicePolicy::set and the log's mark
enum class Sw : unsigned {
    astc_on_cpu, batch_submits, compute_builds_serial, pcache_fetch_lock, split_before_draw, ub_snapshot, split_ub_flush,
    occ_copy, mirrors, fsr4_tune, position_invariant, equal_depth_bias, cap_spare_max, staging_keep, workers,
    mesh_fallback_strict_fmul, bc_copy_per_layer,
    fetch_bounds, nan_bits, gather_offset,   // (device_policy_fixes_env's three)
    count
};

// Each field: what it is, the PCs that get it and why (the site's own comment has the runs), its switch, and who decides
// (and the driver version the rule was seen on, where one is known). "read": the site uses the field, its switch read by
// device_policy_env. "site, step N": the site still decides by the same rule with its own switch read, and step N of the
// conversion moves it here.
struct DevicePolicy {
    uint32_t set_by_switch = 0;     // (a bit for each Sw whose switch set its field: set())
    char switch_notes[320] = "";    // (the switches that set a field, and the values that were ignored, for the log)
    bool set(Sw s) const { return (set_by_switch >> (unsigned)s) & 1u; }
    // ---- one driver's bug ----
    // The translation's three fixes (they were vk_vendor.h's), chosen from who made the GPU, whose driver it runs and, where
    // a driver names its compiler, its driverInfo. NVIDIA's driver gets none of them (the GPU every row of the game was
    // checked on). Each has a switch that forces it on or off on any GPU, to test it where it is not needed, read by
    // device_policy_fixes_env.
    // Every GPU but NVIDIA's: out-of-bounds texel fetches answered with zero in the shader (emit_spirv_image.cpp
    // EmitImageFetch). Vulkan leaves such a fetch undefined without robustImageAccess, which Hookshot does not turn on;
    // NVIDIA's GPUs (the Switch 2's among them) return zero (measured on an RTX 3090: every texture type, past each edge,
    // the last level and the last sample). The ground fog's shader reads depth at points projected off screen near the
    // edges, and on AMD (RX 9070) those pixels took the other branch: a pale stepped box at the left, right and top
    // edges. RX_VK_FETCH_BOUNDS. Read (vk_coopmat.cpp coopmat_init)
    bool fetch_bounds = false;
    // AMD's older Windows shader compiler (the driver branch for RDNA 1 and 2, RX 5000 and RX 6000: its driverInfo names
    // the "AMD proprietary shader compiler" where the RDNA 3 and 4 driver names LLPC; the test of driverInfo is
    // totk-pc-recomp's vk_caps.c, MIT): isnan(x) as a test of x's bits, not OpIsNan (emit_spirv_floating_point.cpp IsNan).
    // Players of OoT ports on those cards saw black and white blocks where a NaN test did not hold. The colour outputs'
    // NaN guard is such a test too. RX_VK_NAN_BITS. Read (coopmat_init)
    bool nan_bits = false;
    // AMD's older Windows shader compiler (RDNA 1 and 2, RX 5000 and RX 6000: same driverInfo check as nan_bits): in
    // compute-fallback mesh stages, where gl_Position cannot be decorated Invariant, the compiler ignores NoContraction
    // (even with DISABLE_OPTIMIZATION_BIT) and reassociates fma(t, v * rsq, add) into fma(rsq * t, v, add) in depth-only
    // passes where v * rsq has one use while keeping v * rsq in the colour pass. The 1-ULP shift on grass blade tips
    // fails the colour pass's EQUAL depth test and flashes white (sky) triangles (issues 1710, 1799, 1934; seen on
    // 0x800161 / 26.11.2). Emitting NoContraction f32 multiplies as fma(a, b, -0.0f) (IEEE-754 bit-exact, single
    // v_fmaak_f32) blocks the reassociation. RX_VK_MESH_STRICT_FMUL. Read (coopmat_init; in the program cache's stamp)
    bool mesh_fallback_strict_fmul = false;
    // AMD's Windows driver (diagnosed on RDNA 2: RX 6700 XT, RX 6600, driver 0x800161): a single multi-layer
    // VkBufferImageCopy (layerCount > 1) into a block-compressed (BC3/BC7) mip-tail level whose extent is not a multiple
    // of 4 (e.g. 5x5 mip 8 on the 1365 x 1365 x 90 terrain material arrays, 6x6 on 1536/768, 9x9 on 300) truncates the
    // copy to floor(w/4) x floor(h/4) blocks per layer, leaving the right and bottom edge blocks zeroed (pure black) even
    // with explicit bufferRowLength/bufferImageHeight. Splitting non-4-aligned multi-layer copies into one
    // VkBufferImageCopy region per layer inside the same call uses the single-layer copy path at zero measurable cost while
    // keeping 8-layer decode/encode runs intact; enabled across AMD's Windows driver (matching bc_small_runs) so untested
    // AMD Windows generations (RDNA 1, RDNA 3, GCN) cannot regress there. RX_VK_BC_COPY_PER_LAYER. Read
    // (vk_texture.cpp bc_encode_into)
    bool bc_copy_per_layer = false;
    // AMD's and Intel's drivers and the mobile ones Eden runs on: gathers nudged by 1/512 of a texel, so that the four
    // texels a gather picks match Maxwell's at texel edges (emit_spirv_image.cpp ImageGatherSubpixelOffset, Eden's, which
    // Hookshot had never turned on). Eden and Lichii turn it on for these drivers: shadow edges drawn as dashed lines
    // without it. RX_VK_GATHER_OFFSET. Read (vk_shader.cpp make_profile; in the program cache's stamp)
    bool gather_offset = false;
    // AMD's Windows driver: its compiler gets fragment-stage subgroup shuffles wrong, a basket in Link's house drew as a
    // white blob (Lichii #273); broadcasts and selects there instead. RADV too (seen so far on RDNA1, other RADV GPUs
    // untested): objects flickered between right, speckled and glowing white while moving (the ladder of Link's house). RX_VK_FRAG_SHUFFLE_FALLBACK, read at the site. Read
    // (vk_shader.cpp make_profile; in the program cache's stamp)
    bool frag_shuffle_fallback = false;
    // Every vendor but NVIDIA: positions invariant, as AMD's and Intel's compilers do not give a depth pass and its colour
    // pass the same position. By the vendor, so NVK goes without them as NVIDIA's own compiler does: whether Mesa's
    // compiler needs them wants a run on NVK. RX_VK_NO_INVARIANT turns them off. Read (vk_coopmat.cpp coopmat_init)
    bool position_invariant = false;
    // Intel's Windows driver (except discrete Battlemage, 0xE2xx): a colour pass tests EQUAL against what another
    // program wrote, and its compiler does not give the two the same position to the last bit (holes in the ground,
    // issue 17, Arc B390); 64 units towards the camera. Not Mesa's (Iris Xe, Mesa 26.2.7: Kakariko, the Death Mountain
    // gate and trail without a hole) and not Battlemage (Arc B570, 101.9175: Castle Town Market and Hyrule Field without
    // a hole), where the margin's LESS_OR_EQUAL passes discarded alpha-tested Z-prepass cutouts as opaque rectangles
    // or draws terrain in front of rocks. RX_VK_EQ_BIAS=<units>, 0 off. Read (vk_draw.cpp dynamic_state)
    float equal_depth_bias = 0;
    // AMD but RADV: task and mesh programs built without the driver's optimizer, which hung the GPU (RX 6600, issues 461
    // and 484); on RADV they reset it built without (issue 1000). RX_VK_MESH_OPT. Site (vk_draw.cpp), step 4
    bool mesh_no_optimizer = false;
    // AMD with the task + mesh fallback (Navi 21-23 on Windows): a graphics build the compiler rejected is tried again
    // without its optimizer. No switch. Site (vk_draw.cpp), step 4
    bool amd_retry_no_optimizer = false;
    // AMD's Windows driver: the cloth program 5fa56485a6b07619 built without the optimizer, which left its colour pipeline
    // pending (issues 461, 1540). No switch. Site (vk_draw.cpp, which asks the driver again), step 4
    bool amd_cloth_no_optimizer = false;
    // AMD's Windows driver: one program's builds go to it one at a time, so a second waits outside it and not behind a
    // faulted one (Hookshot issue 996, that driver's hang). It asked the vendor until this rule moved here, which held
    // RADV's and AMDVLK's builds too. RX_VK_DRIVER_FAULTS=2 on any GPU, read at the site. Read (vk_draw.cpp)
    bool fault_gate_serial = false;
    // AMD's Windows driver on Navi 21 to 23 (mesh_mode.h's list): task + mesh draws through the compute fallback, as the
    // native path timed out (Hookshot 996; seen on driver 0x800161, and no later driver is known to end it). Where the
    // card has mesh shaders and buffer device addresses. RX_VK_TASK_MESH_FALLBACK. Site (vk_main.cpp create_device), step 5
    bool task_mesh_fallback = false;
    // 0 Automatic is native where the card has mesh shaders, 2 Compute: RADV, whose native task and mesh path deadlocks
    // its gfx and compute rings (issue 2245: RX 6800, RX 6600 XT, Steam Deck OLED, Ally X; no Mesa version is known to end
    // it). It asked for AMD off Windows until this rule moved here, which took AMDVLK too, a driver the issue does not
    // name. RX_VK_MESH_SHADERS, read at the site. Read (create_device, which asks the driver itself: the facts come later)
    int mesh_mode = 0;
    // RADV on GCN 1-4: fallback compute stages over this many phis are not built, as RADV's compiler grew past 25 GB on
    // them (an RX 580); Mesa's Intel driver its own (20000); 0 elsewhere, no limit. RX_VK_FALLBACK_PHI_LIMIT. Site (vk_draw.cpp), step 5
    uint32_t fallback_phi_limit = 0;
    // Intel on Windows: one thread at a time in igvk64's compiler, which faulted with two (an Arc A750, issue 496; a UHD).
    // Not where the row's Auto is side by side (intel_side_by_side_auto); the Intel shader compiles row and
    // RX_VK_NO_INTEL_SERIAL open it. Site (vk_internal.h intel_gate_on), step 4
    bool intel_compile_gate = false;
    // Intel's Windows driver: compute pipelines built one at a time on a thread of their own, as igvk64 froze the PC with
    // four at once (an Arc A750, issue 496). It asked the vendor until this rule moved here, which held Mesa's driver
    // too. RX_VK_COMPUTE_SERIAL. Read (vk_draw.cpp serial_builds_on)
    bool compute_builds_serial = false;
    // Intel's Windows driver: builds held off while a pipeline cache's data is fetched, as igvk64 faulted in the fetch
    // beside them (Hookshot 999, an Arc 140V; seen on driver 101.9175). RX_VK_PCACHE_FETCH_LOCK. Read (vk_draw.cpp
    // pcache_fetch_lock_on)
    bool pcache_fetch_lock = false;
    // Intel: a part of a submission ends before a plain draw rather than after its 120th, as a part that began with mesh
    // draws drew exploded triangles (an Arc B580 on Windows; seen on driver 101.9154). By the vendor: the cause was not
    // found, so whether Mesa's driver or a GPU without mesh shaders needs it wants a run. RX_VK_SPLIT_BEFORE_DRAW. Read
    // (vk_replay.cpp part_ends_before_draw)
    bool split_before_draw = false;
    // AMD's Windows driver on the graphics in Ryzen 7000 and 9000 desktop CPUs (0x164E, 0x13C0) and the Radeon 860M
    // (0x1114): ASTC decoded on the CPU, as the GPU decoder's one long submission had Windows reset the device (a Ryzen 9 7900X,
    // 7 runs of 8; seen on driver 26.10.1; the 860M lost it on the same Kokiri Forest arrays). On the Ryzen 5 5625U
    // Vega iGPU (0x15E7, AMD driver 26.5.2), CPU decoding restores Water Temple walls and the minimap.
    // RX_VK_ASTC_CPU. Read (vk_texture.cpp astc_on_cpu)
    bool astc_on_cpu = false;
    // NVIDIA: the Deku Tree web's vertex code b449bcdb69b36a8a has its global loads inlined, as NVIDIA's compiler
    // substitutes a load's result there. No switch. Read (vk_shader.cpp translate_into; in the program cache's stamp)
    bool inline_deku_web_loads = false;
    // Every vendor but NVIDIA: the movie upscaler's networks are left out (the last pass becomes a linear upscale of the
    // movie), as their multiplies took minutes each for AMD's compiler and the movies ran at 10 fps (issues 26, 38, 43).
    // The site leaves them out where the multiplies run in software too, on any vendor, which is known only after the
    // device is made. RX_VK_NETWORK_KEEP names the ones that still run, read at the site. Read (vk_shader.cpp
    // translate_into; in the program cache's stamp)
    bool networks_left_out = false;
    // Guest memory tried as dma-bufs: off Windows, and not on NVIDIA's own driver, which refuses the mapping for every
    // pool. RX_VK_NO_DMABUF turns it off. Site (vk_memory.cpp import_dma_buf), step 4
    bool dma_buf_imports = false;
    // Where VK_NV_device_diagnostic_checkpoints is listed (NVIDIA): a marker before each draw, without which the device
    // was lost every run; the race they hide is not found. RX_VK_CHECKPOINTS=0. Site (create_device), step 5
    bool nv_checkpoints = false;
    // Retired imports: 1 given back at once; 2 after a wait, on Windows with AMD's and Intel's drivers, which lost the
    // device right after give-backs (issues 17, 309, 571); 0 kept is only by the switch (retired_give_back.h Mode).
    // RX_VK_KEEP_RETIRED. Site (vk_memory.cpp retired_mode), step 2
    int retired_mode = 1;
    // Idle imports given back past this share of the system heap's budget: 85, and 0 (off) on Windows with Intel's driver,
    // for the same losses (AMD's gives them back too since 11-17: kept, new imports were refused past ~7 GB). RX_VK_IMPORT_HIGH. Site (vk_memory.cpp), step 2
    int idle_import_high_pct = 85;
    // An allowance the driver gives as the count itself is taken a tenth lower (AMD's Windows driver, #933, #1025). On
    // every PC: vram_usage sees the driver do it, and no fact here chooses. Site (vk_vma.cpp), step 2
    bool allowance_at_count_rule = true;

    // ---- what the device can do ----
    // A discrete card: its buffer pools get a copy in VRAM, as it reads the game's memory over PCIe (110 ms of GPU time a
    // frame on a 3090 Ti in TOTK; an RX 580 28 to 54% faster). RX_VK_MIRROR. Read (vk_memory.cpp mirror_on)
    bool mirrors = false;
    // Imports by chunks of 4 MB with mirrors (a mirror is its import's whole range), 16 without. RX_VK_IMPORT_CHUNK_MB.
    // Site (vk_memory.cpp import_chunk), step 7
    uint32_t import_chunk_mb = 16;
    // Intel: uniform buffers copied as submitted, as a GPU without mirrors reads memory the game has rewritten (and the
    // B580 for the renderer's own delta push). By the vendor until a GPU without mirrors that is not Intel's has been
    // run with it. RX_VK_UB_SNAPSHOT. Read (vk_draw.cpp ub_snapshot_wanted)
    bool ub_snapshot = false;
    // Uniform updates written at each part's end; not on Intel without mirrors, where the game saw them mid-frame (a
    // white flash every 8 frames on an Arc 140V, Hookshot 999). The mirrors as the run has them, RX_VK_MIRROR counted.
    // RX_VK_SPLIT_UBFLUSH. Read (vk_replay.cpp)
    bool split_ub_flush = true;
    // The GPU's own float add on storage buffers, where the card has it and a run has shown it right
    // (policy_rule::native_fadd_listed, the one list: NVIDIA's driver; Intel's Windows driver from 101.9175 on Meteor Lake
    // and Arrow Lake, 101.9154 on Battlemage, 101.9150 on Panther Lake, 101.8991 on Lunar Lake); elsewhere the compare-exchange loop.
    // RX_VK_NATIVE_FADD, read at the site. Site (create_device, which asks the list itself), step 5
    bool native_float_add = false;
    // A subgroup can have more than 32 lanes: the long warp path for the stages the GPU does not pin to 32 (the site's
    // stage profiles). RX_VK_WARP64, read at the site. Read (vk_shader.cpp make_profile; in the program cache's stamp)
    bool warp64 = false;
    // The mesh fallback's ballots, votes and shuffles through workgroup memory (mma_layout.h mesh_fallback_shared_warp): a
    // fallback stage the planner takes then has no subgroup instruction and runs alike at any lane count. Mesa's Intel
    // driver, where 32 lanes spilled (fallback_lanes.h); and AMD's Windows driver on a GPU whose compute stages cannot be
    // held to 32 lanes but run wider (GCN: RX 400/500 and Vega, 64 lanes only, sg32_stages without compute), where every
    // such stage otherwise takes the 64-lane emulation of NVN's 32-lane warp: on driver 24.9.1 an RX 580 compiled each
    // fallback stage in 53 to 310 s, its draws left out meanwhile, and foliage, hair and Link drew as specks (Longshot
    // #1204, #1836). Not RADV on the same cards: an RX 580 there (Mesa 26.2.6) compiled them in seconds and drew them
    // right on the subgroup, and the shared warp gave the same answers and the same fps, so it would only cost a GCN RADV
    // player one translation of the fallback. RX_VK_MESH_SHARED_WARP, read at the site. Site (vk_coopmat.cpp
    // coopmat_init; translator_stamp folds what it gives)
    bool mesh_shared_warp = false;

    // ---- measured on one card ----
    // RADV but on GCN 1-4 and on a discrete RDNA 4 card: the replay's submits held and sent as one (a Radeon 8060S on Bazzite, the submit 70 -> 6 ms a
    // frame, issue 413; an RX 580 lost 4 fps). GCN is DeviceFacts::radv_gcn, the one definition; it was "no stage pinned
    // to 32 lanes and 64-lane waves" until this rule moved here, which RX_VK_NO_SUBGROUP_PIN made true of every 64-lane
    // card and which took Vega (GFX9) for GCN too. On an RX 9070 XT the gaps between held submits let the card's
    // clock drop (median 1715 MHz held, 2589 MHz sent at once) and the frame was a quarter longer (62 -> 81 fps sent at
    // once). RX_VK_BATCH. Read (vk_replay.cpp batch_on)
    bool batch_submits = false;
    // RADV and AMD RDNA 4 on AMD's proprietary driver: the software matrix multiplies' unrolled forms are tried (measured on the
    // Steam Deck and an RX 9060 XT); a test shader on the device then says which is used. RX_VK_MMA_UNROLL. Site
    // (vk_coopmat.cpp), step 7
    bool mma_unroll_try = false;
    // Mesa's Intel driver, with dynamic indexing of storage buffer arrays: the tables searched in loops, as Mesa inlines
    // the written-out form (Windows not tried). RX_VK_GM_LOOP. Site (vk_coopmat.cpp table_loop_wanted), step 7
    bool global_table_loop = false;
    // Windows AMD integrated graphics and Intel discrete graphics: copy occlusion counts to host memory.
    // An Arc B580 went from 34.8 to 45.8 fps in the Market on driver 101.9154. Off with parallel replay, at the site.
    // RX_VK_OCC_COPY. Read (vk_replay.cpp occ_copy_wanted)
    bool occ_copy = false;
    // Intel on Windows, the Arc B-series from driver 101.9154: the Intel shader compiles row's Auto is side by side (a
    // B580 froze 248-409 s at an area's first visit one at a time). Site (intel_compiles.h, vk_menu.cpp), step 4
    bool intel_side_by_side_auto = false;
    // AMD: FSR 4's passes tuned as the library makes them (an RX 6700 XT, 8.97 -> 4.04 ms; AMD's outputs bit for bit on
    // Adrenalin 26.10.1). RX_VK_FSR4_TUNE can only turn it off: it rewrites AMD's shaders for AMD's GPUs. Read (vk_fsr4.cpp
    // tune::on)
    bool fsr4_tune = false;
    // The Steam Deck's own APU: its peak clock state while its power level is auto, which the site reads later (+8% to
    // +46% for about 4 W). RX_VK_GPU_CLOCK. Site (gpu_clock.h, vk_gpu_clock.cpp), step 2
    bool gpu_clock_peak = false;
    // NVIDIA: a queue kept for the presenter thread, which is itself off. The study of these rules doubts it: nothing
    // uses the queue. RX_VK_PRESENT_THREAD. Site (create_device), step 5
    bool presenter_queue = false;
    // Linux with NVIDIA's own driver and 20 GB of RAM or less: the pipeline cache's file may be left unread, where
    // NVIDIA's own cache is warm and programs load at create (pcache_load.h decides with those). RX_VK_PCACHE_LOAD. Site
    // (vk_draw.cpp pcache_load_off), step 4
    bool pcache_small_pc = false;

    // ---- sizes ----
    // The hardware half of the smaller top levels (policy_rule::small_vram_textures); the site adds Minimum's settings.
    // RX_VK_LOW_VRAM_MIPS=0. Site (vk_texture.cpp low_vram_texture_mips), step 3
    bool small_vram_textures = false;
    // Every large ASTC texture kept as BC3: allowed under 8.5 GB, or an integrated GPU (astc_bc_policy.h).
    // RX_VK_ASTC_BC_ALL. Site (vk_texture.cpp), step 3
    bool astc_bc_all = false;
    // The texture cache's budget at the first look (policy_rule::texture_budget). RX_VK_VRAM_MB. Site (texture_frame), step 3
    uint64_t texture_budget = 0;
    // The mirrors' budget (policy_rule::mirror_budget). The study doubts the third on cards of 6.5 GB or less: a 4 GB card
    // gets 1365 MB where mirror_budget.h argues for 256. RX_VK_MIRROR_MB. Site (vk_memory.cpp mirror_budget), step 3
    uint64_t mirror_budget = 0;
    // The render target pool's bound in MB: a twelfth of the allowance, 256 to 768 (target_pool.h). RX_VK_TARGET_POOL_MB.
    // Site (vk_texture.cpp pool_cap), step 2
    uint64_t target_pool_mb = 256;
    // The textures' free space a compaction starts from: a sixteenth of the allowance, 64 to 512 MB (vram_compact.h), read
    // again at each pass at the site. RX_VK_COMPACT_MB. Site (vk_texture.cpp), step 2
    uint64_t compact_min_free = 0;
    // The write-back shadows' VRAM at most (policy_rule::wb_shadow_cap). RX_VK_WB_SHADOW_MB. Site (vk_memory.cpp shadow_room), step 3
    uint64_t wb_shadow_cap = 0;
    // What the spare capture buffers may hold (policy_rule::cap_spare_max; ~0: no limit). RX_VK_CAP_SPARE_MB=<MB>, 0 no
    // limit. Read where the switch set it (set(Sw::cap_spare_max)); else the site (vk_memory.cpp cap_spare_max) asks the
    // rule with the allowance at its own first call
    uint64_t cap_spare_max = ~0ull;
    // The idle staging arenas kept (policy_rule::staging_keep). RX_VK_STAGING_KEEP_MB=<MB>. Read where the switch set it;
    // else the site (texture_frame) asks the rule with the allowance at its first frame
    uint64_t staging_keep = 0;
    // The textures' pool's blocks made at the start, of 32 MB: none on an integrated GPU, 8% of the heap (2 to 12) under
    // 8 GB, else 15% (2 to 24), as each made in play is ~170 ms on AMD's Windows driver (vram_compact.h blocks_at_start).
    // RX_VK_IMAGE_PREALLOC. Site (vk_vma.cpp), step 3
    uint32_t image_blocks_at_start = 0;
    // VMA's block sizes by the heap (policy_rule::vma_block, target_block). No switch. Site (vk_vma.cpp), step 3
    uint64_t vma_block = 0, target_block = 0;
    // The driver's count, in percent of its allowance, past which the textures give up the excess: 97, and 90 on an
    // integrated GPU, which has nowhere to page to (vram_compact.h count_line_pct). No switch. Site (texture_frame), step 3
    uint32_t count_line_pct = 97;
    // Room in VMA's blocks lets that squeeze wait; not on an integrated GPU (vram_compact.h room_in_blocks). Site
    // (texture_frame), step 3
    bool room_in_blocks_rule = true;
    // The squeeze takes a quarter of the excess where the driver allows under 8 GB (policy_rule::squeeze_damped). No
    // switch. The site (texture_frame) asks the rule with each frame's own allowance
    bool squeeze_damped = false;
    // Linux's memory hold caps the texture budget only on an integrated GPU, whose textures are that memory. No switch.
    // Site (texture_frame), step 3
    bool linux_hold_caps_budget = false;
    // DLSS's model K row draws with the CNN preset E on a card under 7 GB (upscaler_rows.h cnn_for_vram). RX_VK_DLSS_PRESET.
    // Site (vk_dlss.cpp, vk_menu.cpp), step 2
    bool dlss_cnn = false;
    // The Quality preset Auto (this PC)'s tier, 1 Minimum to 5 Ultra (auto_settings.h tier_of): by the heap, Minimum on an
    // integrated GPU, a handheld or a card without mesh shaders, Low at most under 12 GB of RAM. Site (auto_profile), step 2
    int auto_tier = 0;
    // An integrated GPU beside 4 threads or fewer: Auto's Render resolution is 33% (auto_settings.h choose). Site, step 2
    bool auto_render_third = false;
    // An NVIDIA card named RTX, or an AMD RDNA 4 card (is_rdna4): Auto picks DLSS or FSR 4 where the build has it. Site
    // (auto_profile), step 2
    bool rtx = false, rdna4 = false;
    // The background workers (policy_rule::workers). RX_VK_WORKERS=<n>, 1 to 16. Read (vk_jobs.cpp start)
    unsigned workers = 2;
    // The capture's copy helpers (policy_rule::capture_helpers). RX_VK_CAPTURE_THREADS. Site (vk_memory.cpp), step 3
    unsigned capture_helpers = 2;
    // The texture uploads' threads (policy_rule::texture_threads, texture_gather_threads). No switch. Sites (vk_texture.cpp:
    // three functions that checks compile on their own, with no device), step 3
    unsigned texture_threads = 1, texture_gather_threads = 1;
    // The preparation's workers in play (policy_rule::program_workers_in_play). No switch. Read (vk_draw.cpp prepare_run)
    unsigned program_workers_in_play = 2;
    // Threaded rendering's Auto by the cores the process may use, where the mode is safe on this run (threaded_rendering.h):
    // the render thread from 4 cores, the recorder thread too from 6. Site (vk_menu.cpp), step 7
    bool threaded_auto_render = false, threaded_auto_recorder = false;
    // Linux with 20 GB of RAM or less: the shader modules are let go after the precompile (1.5-1.8 GB); a memory shortfall
    // then does the same on any PC, at the site. RX_VK_DROP_MODULES, RX_VK_KEEP_MODULES. Site (vk_draw.cpp), step 2
    bool drop_modules_small_ram = false;
    // An unset HOS_NATIVE_ZSTD keeps the zstd decode's virtual core on an Intel integrated GPU (its load freezes) and gives
    // it up elsewhere. Site (host/hos_zstd.c, through HOS_OPT_NATIVE_ZSTD)
    bool native_zstd_keep_core = false;
};

/// The defaults this PC gets, with no switch read
inline DevicePolicy device_policy(const DeviceFacts& f) {
    using namespace policy_rule;
    DevicePolicy p;
    const uint64_t allowed = f.allowed_at_start, heap = f.vram_heap;
    const bool integrated = f.integrated();

    // one driver's bug
    p.fetch_bounds = !f.nvidia();
    p.nan_bits = f.driver_id == VK_DRIVER_ID_AMD_PROPRIETARY && f.driver_info && std::strstr(f.driver_info, "proprietary shader compiler");
    p.mesh_fallback_strict_fmul = p.nan_bits;
    p.bc_copy_per_layer = f.amd_win();
    switch (f.driver_id) {
    case VK_DRIVER_ID_AMD_PROPRIETARY:
    case VK_DRIVER_ID_AMD_OPEN_SOURCE:
    case VK_DRIVER_ID_MESA_RADV:
    case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
    case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
    case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:
    case VK_DRIVER_ID_MESA_TURNIP:
        p.gather_offset = true;
        break;
    default:
        break;
    }
    p.frag_shuffle_fallback = f.driver_id == VK_DRIVER_ID_AMD_PROPRIETARY || f.driver_id == VK_DRIVER_ID_MESA_RADV;
    p.position_invariant = !f.nvidia();
    p.equal_depth_bias = policy_rule::equal_depth_bias(f.driver_id, f.device_id);
    p.mesh_no_optimizer = f.amd() && !f.radv();
    // (corrected: the driver the deadlock is in, RADV, and not "AMD off Windows", which also took AMDVLK. To go back:
    // !f.windows && f.amd(), here and in create_device)
    p.mesh_mode = f.radv() ? 2 : 0;
    p.task_mesh_fallback = f.windows && mesh_rule::task_mesh_fallback_default(f.vendor_id, f.device_id) && f.mesh_shader &&
                           f.buffer_device_address;
    p.amd_retry_no_optimizer = f.amd() && p.task_mesh_fallback;
    p.amd_cloth_no_optimizer = f.windows && f.driver_id == VK_DRIVER_ID_AMD_PROPRIETARY;
    // (corrected: AMD's Windows driver, whose hang issue 996 is, and not the vendor, which also held RADV's and AMDVLK's
    // builds to one at a time. To go back: f.amd())
    p.fault_gate_serial = f.driver_id == VK_DRIVER_ID_AMD_PROPRIETARY;
    // (Mesa's Intel driver has its own limit: its compiler did not finish the 29,000- and 80,000-phi stages, fallback_size_policy.h)
    p.fallback_phi_limit = f.radv_gcn() ? fallback_size::kRadvGcnPhiLimit
                           : f.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ? fallback_size::kIntelPhiLimit : 0;
    p.intel_side_by_side_auto = f.windows && f.intel() && intel_side_by_side_measured(f.device_id, f.driver_version);
    p.intel_compile_gate = f.windows && f.intel() && !p.intel_side_by_side_auto;
    // (corrected: Intel's Windows driver, whose compiler issue 496 is, and not the vendor, which also held ANV's compute
    // builds to one at a time. To go back: f.intel())
    p.compute_builds_serial = f.driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS;
    p.pcache_fetch_lock = f.windows && f.intel();
    p.split_before_draw = f.intel();
    p.astc_on_cpu = f.windows && f.amd() && (f.device_id == 0x164E || f.device_id == 0x13C0 || f.device_id == 0x1114 ||
                                         f.device_id == 0x15E7);
    p.inline_deku_web_loads = f.nvidia();
    p.networks_left_out = !f.nvidia();
    p.dma_buf_imports = !f.windows && f.driver_id != VK_DRIVER_ID_NVIDIA_PROPRIETARY;
    p.nv_checkpoints = f.nv_checkpoints_ext;
    p.retired_mode = (int)retired_give_back::mode(false, false, f.windows, f.nvidia());
    p.idle_import_high_pct = f.windows && !f.nvidia() && !f.amd() ? 0 : 85;
    p.allowance_at_count_rule = true;

    // what the device can do
    p.mirrors = f.discrete();
    p.import_chunk_mb = p.mirrors ? 4 : 16;
    p.ub_snapshot = f.intel();
    p.split_ub_flush = !f.intel() || p.mirrors;
    p.native_float_add = native_fadd_listed(f.driver_id, f.device_id, f.driver_version) && f.float32_atomic_add;
    p.warp64 = f.subgroup_max > 32;
    p.mesh_shared_warp = f.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                         (f.windows && f.driver_id == VK_DRIVER_ID_AMD_PROPRIETARY && f.subgroup_max > 32 &&
                          !(f.sg32_stages & VK_SHADER_STAGE_COMPUTE_BIT));

    // measured on one card
    // (corrected: GCN by the one definition, radv_gcn. To go back: f.radv() && !(f.sg32_stages == 0 && f.subgroup_size > 32))
    p.batch_submits = f.radv() && !f.radv_gcn() && !(f.rdna4() && f.discrete());
    p.mma_unroll_try = f.radv() || (f.rdna4() && f.driver_id == VK_DRIVER_ID_AMD_PROPRIETARY);
    p.global_table_loop = f.driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA && f.storage_buffer_array_dynamic_indexing;
    p.occ_copy = f.windows && ((f.amd() && f.integrated()) ||
                              (f.driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS && f.discrete()));
    p.fsr4_tune = f.amd();
    p.gpu_clock_peak = f.amd() && f.deck_apu();
    p.presenter_queue = f.nvidia();
    p.pcache_small_pc = !f.windows && f.driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY && (f.ram >> 20) && (f.ram >> 20) <= 20480;

    // sizes
    p.small_vram_textures = small_vram_textures(allowed, integrated, f.ram);
    p.astc_bc_all = astc_bc_policy::all(allowed, integrated);
    p.texture_budget = texture_budget(allowed, heap, f.ram, integrated, p.astc_on_cpu);
    p.mirror_budget = mirror_budget(heap);
    p.target_pool_mb = target_pool::default_mb(allowed);
    p.compact_min_free = vram_compact::default_min_free(allowed);
    p.wb_shadow_cap = wb_shadow_cap(allowed, integrated);
    p.cap_spare_max = cap_spare_max(allowed);
    p.staging_keep = staging_keep(allowed);
    p.image_blocks_at_start = (uint32_t)vram_compact::blocks_at_start(heap, 32 * kMB, true, integrated);
    p.vma_block = vma_block(heap), p.target_block = target_block(heap);
    p.count_line_pct = (uint32_t)vram_compact::count_line_pct(integrated);
    p.room_in_blocks_rule = !integrated;
    p.squeeze_damped = squeeze_damped(allowed);
    p.linux_hold_caps_budget = !f.windows && integrated;
    p.dlss_cnn = heap && heap < 7 * kGB;
    {
        ::auto_settings::Hardware hw;
        hw.integrated = integrated, hw.handheld = f.handheld(), hw.mesh_native = f.mesh_shader;
        hw.vram_mb = (uint32_t)(heap >> 20), hw.ram_mb = (uint32_t)(f.ram >> 20);
        p.auto_tier = ::auto_settings::tier_of(hw);
    }
    p.auto_render_third = integrated && f.cpus_all > 0 && f.cpus_all <= 4;
    p.rtx = f.nvidia() && std::strstr(f.device_name, "RTX");
    p.rdna4 = f.rdna4();
    p.workers = workers(f.cpus_all);
    p.capture_helpers = capture_helpers(f.cpus_all);
    p.texture_threads = texture_threads(f.cpus_all);
    p.texture_gather_threads = texture_gather_threads(f.cpus_all);
    p.program_workers_in_play = program_workers_in_play(f.cpus_all, f.cpus_usable);
    {
        const threaded::Choice cpu = threaded::auto_picks(f.cores_usable);
        p.threaded_auto_render = cpu.render_thread, p.threaded_auto_recorder = cpu.recorder_thread;
    }
    p.drop_modules_small_ram = !f.windows && f.ram > 0 && f.ram <= 20 * kGB;
    p.native_zstd_keep_core = f.intel() && f.integrated();
    return p;
}

/// The switches over the defaults, read once (create_device), by one rule for all of them.
/// A yes/no switch: unset or empty, the PC's default; 0, false, off or no (any case, spaces around them not counted), off;
/// anything else, on. A number: decimal digits only, in the field's range; anything else is ignored, the default stays,
/// and the log says so (switch_notes). Two can only turn their rule off, and say so where they are read.
/// Only the fields marked "read" whose switch is named here: a site that still decides reads its own. get: getenv, or a test's
template <class Get> void device_policy_env(DevicePolicy& p, const DeviceFacts& f, Get get) {
    const auto note = [&](const char* name, const char* value, const char* what) {
        const size_t n = std::strlen(p.switch_notes);
        std::snprintf(p.switch_notes + n, sizeof p.switch_notes - n, "%s%s=%.24s%s", n ? "; " : "", name, value, what);
    };
    const auto mark = [&](Sw s, const char* name, const char* value) {
        p.set_by_switch |= 1u << (unsigned)s;
        note(name, value, "");
    };
    // (-1: unset or empty, 0: off, 1: on; e: the value)
    const auto word = [&](const char* name, const char*& e) -> int {
        e = get(name);
        if (!e || !*e) return -1;
        return env_value_off(e) ? 0 : 1;
    };
    const auto flag = [&](Sw s, const char* name, bool& v) {
        const char* e = nullptr;
        const int w = word(name, e);
        if (w < 0) return;
        v = w != 0;
        mark(s, name, e);
    };
    const auto number = [&](Sw s, const char* name, uint64_t lo, uint64_t hi, uint64_t& out) -> bool {
        const char* e = get(name);
        if (!e || !*e) return false;
        if (!policy_rule::parse_number(e, lo, hi, out)) {
            note(name, e, " ignored (not a whole number in its range)");
            return false;
        }
        mark(s, name, e);
        return true;
    };
    flag(Sw::mirrors, "RX_VK_MIRROR", p.mirrors);
    // (what goes by the mirrors goes by them as the run has them)
    p.import_chunk_mb = p.mirrors ? 4 : 16;
    p.split_ub_flush = !f.intel() || p.mirrors;
    flag(Sw::split_ub_flush, "RX_VK_SPLIT_UBFLUSH", p.split_ub_flush);
    flag(Sw::ub_snapshot, "RX_VK_UB_SNAPSHOT", p.ub_snapshot);
    flag(Sw::astc_on_cpu, "RX_VK_ASTC_CPU", p.astc_on_cpu);
    // (the integrated GPU's texture budget follows whether ASTC is decoded on the CPU)
    p.texture_budget = policy_rule::texture_budget(f.allowed_at_start, f.vram_heap, f.ram, f.integrated(), p.astc_on_cpu);
    flag(Sw::batch_submits, "RX_VK_BATCH", p.batch_submits);
    flag(Sw::compute_builds_serial, "RX_VK_COMPUTE_SERIAL", p.compute_builds_serial);
    flag(Sw::pcache_fetch_lock, "RX_VK_PCACHE_FETCH_LOCK", p.pcache_fetch_lock);
    flag(Sw::split_before_draw, "RX_VK_SPLIT_BEFORE_DRAW", p.split_before_draw);
    flag(Sw::occ_copy, "RX_VK_OCC_COPY", p.occ_copy);
    flag(Sw::mesh_fallback_strict_fmul, "RX_VK_MESH_STRICT_FMUL", p.mesh_fallback_strict_fmul);
    flag(Sw::bc_copy_per_layer, "RX_VK_BC_COPY_PER_LAYER", p.bc_copy_per_layer);
    {   // (off only: the tuning rewrites AMD's shaders for AMD's GPUs, and a value that says on leaves the default)
        const char* e = nullptr;
        if (word("RX_VK_FSR4_TUNE", e) == 0) p.fsr4_tune = false, mark(Sw::fsr4_tune, "RX_VK_FSR4_TUNE", e);
    }
    {   // (a switch named for what it turns off: on takes the invariant positions away, off leaves the default)
        const char* e = nullptr;
        if (word("RX_VK_NO_INVARIANT", e) == 1) p.position_invariant = false, mark(Sw::position_invariant, "RX_VK_NO_INVARIANT", e);
    }
    if (const char* e = get("RX_VK_EQ_BIAS"); e && *e) {
        double v = 0;
        if (policy_rule::parse_decimal(e, 65536, v)) p.equal_depth_bias = (float)v, mark(Sw::equal_depth_bias, "RX_VK_EQ_BIAS", e);
        else note("RX_VK_EQ_BIAS", e, " ignored (not a number from 0 to 65536)");
    }
    uint64_t n = 0;
    if (number(Sw::cap_spare_max, "RX_VK_CAP_SPARE_MB", 0, 1048576, n)) p.cap_spare_max = n ? n << 20 : ~0ull;   // (0: no limit)
    if (number(Sw::staging_keep, "RX_VK_STAGING_KEEP_MB", 0, 1048576, n)) p.staging_keep = n << 20;
    if (number(Sw::workers, "RX_VK_WORKERS", 1, 16, n)) p.workers = (unsigned)n;
}

/// The translation's three fixes' switches over them (RX_VK_FETCH_BOUNDS, RX_VK_NAN_BITS, RX_VK_GATHER_OFFSET), read once
/// after device_policy_env, with the reading they have had since vk_vendor.h and not yet the one rule above: unset or
/// empty, the PC's default; a number that is 0, or a word that is no number, off; any other number on (so "on" is off).
/// get: getenv, or a test's
template <class Get> void device_policy_fixes_env(DevicePolicy& p, Get get) {
    const auto over = [&](Sw s, const char* name, bool& v) {
        const char* e = get(name);
        if (!e || !*e) return;
        v = std::atoi(e) != 0;
        p.set_by_switch |= 1u << (unsigned)s;
        const size_t n = std::strlen(p.switch_notes);
        std::snprintf(p.switch_notes + n, sizeof p.switch_notes - n, "%s%s=%.24s", n ? "; " : "", name, e);
    };
    over(Sw::fetch_bounds, "RX_VK_FETCH_BOUNDS", p.fetch_bounds);
    over(Sw::nan_bits, "RX_VK_NAN_BITS", p.nan_bits);
    over(Sw::gather_offset, "RX_VK_GATHER_OFFSET", p.gather_offset);
}

/// One line of the block, built in a buffer under 900 characters whatever it is given (a long driverInfo is cut short)
struct PolicyLine {
    char text[880];
    size_t n = 0;
    explicit PolicyLine(const char* what) {
        text[0] = 0;
        add("vk: policy: %s", what);
    }
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    void add(const char* fmt, ...) {
        if (n >= sizeof text - 1) return;
        va_list ap;
        va_start(ap, fmt);
        const int k = std::vsnprintf(text + n, sizeof text - n, fmt, ap);
        va_end(ap);
        if (k > 0) n = std::min(n + (size_t)k, sizeof text - 1);
    }
    /// name=0 or 1; a * after it when its switch set it; and, on, the driver version the rule was seen on, where one is known
    void flag(const char* name, bool on, bool by_switch = false, const char* seen_on = nullptr) {
        add(" %s=%d%s", name, on ? 1 : 0, by_switch ? "*" : "");
        if (on && seen_on) add("(seen on %s)", seen_on);
    }
};

/// The block at the start: the facts, then what this PC gets by them, a * after a value its switch set (the last line names
/// the switches read and the values ignored). A field a site still decides shows the default: the site's own switch is
/// not in it. A driver's fix that is on says the driver version it was seen on, beside this driver's, so a report shows a
/// workaround still applied on a newer driver. line: void(const char*), a log line
template <class Line> void device_policy_log(const DevicePolicy& p, const DeviceFacts& f, Line line) {
    const auto mb = [](uint64_t bytes) { return (unsigned long long)(bytes >> 20); };
    {
        PolicyLine l("facts");
        l.add(" %04x:%04x (arch %s), driver id %d, version %#x (%.100s), type %d, %s; VRAM %llu MB, allowed at the start %llu MB, RAM %llu MB,"
              " resizable BAR %d; CPUs %u (%u usable, %u cores); mesh shaders %d, float add %d, cooperative matrices %d, 16-bit stage"
              " inputs %d, lanes %u to %u (32 on stages %#x)",
              (unsigned)f.vendor_id, (unsigned)f.device_id, f.arch_name(), (int)f.driver_id, (unsigned)f.driver_version, f.driver_info, (int)f.device_type,
              f.windows ? "Windows" : "not Windows", mb(f.vram_heap), mb(f.allowed_at_start), mb(f.ram), (int)f.rebar, f.cpus_all,
              f.cpus_usable, f.cores_usable, (int)f.mesh_shader, (int)f.float32_atomic_add, (int)f.coop_matrix_ext, (int)f.storage_io16,
              (unsigned)f.subgroup_size, (unsigned)f.subgroup_max, (unsigned)f.sg32_stages);
        line(l.text);
    }
    {
        PolicyLine l("driver fixes");
        l.add(" on driver version %#x:", (unsigned)f.driver_version);
        l.flag("fetch_bounds", p.fetch_bounds, p.set(Sw::fetch_bounds));
        l.flag("nan_bits", p.nan_bits, p.set(Sw::nan_bits));
        l.flag("mesh_fallback_strict_fmul", p.mesh_fallback_strict_fmul, p.set(Sw::mesh_fallback_strict_fmul), "0x800161");
        l.flag("bc_copy_per_layer", p.bc_copy_per_layer, p.set(Sw::bc_copy_per_layer), "0x800161");
        l.flag("gather_offset", p.gather_offset, p.set(Sw::gather_offset));
        l.flag("frag_shuffle_fallback", p.frag_shuffle_fallback);
        l.flag("position_invariant", p.position_invariant, p.set(Sw::position_invariant));
        l.add(" equal_depth_bias=%g%s", (double)p.equal_depth_bias, p.set(Sw::equal_depth_bias) ? "*" : "");
        l.flag("mesh_no_optimizer", p.mesh_no_optimizer);
        l.flag("amd_retry_no_optimizer", p.amd_retry_no_optimizer);
        l.flag("amd_cloth_no_optimizer", p.amd_cloth_no_optimizer);
        l.flag("fault_gate_serial", p.fault_gate_serial);
        l.flag("task_mesh_fallback", p.task_mesh_fallback, false, "0x800161");
        l.add(" mesh_mode=%s fallback_phi_limit=%u", p.mesh_mode == 2 ? "compute" : "automatic", (unsigned)p.fallback_phi_limit);
        l.flag("intel_compile_gate", p.intel_compile_gate);
        l.flag("compute_builds_serial", p.compute_builds_serial, p.set(Sw::compute_builds_serial));
        l.flag("pcache_fetch_lock", p.pcache_fetch_lock, p.set(Sw::pcache_fetch_lock), "101.9175");
        l.flag("split_before_draw", p.split_before_draw, p.set(Sw::split_before_draw), "101.9154");
        l.flag("astc_on_cpu", p.astc_on_cpu, p.set(Sw::astc_on_cpu), "26.10.1");
        l.flag("inline_deku_web_loads", p.inline_deku_web_loads);
        l.flag("networks_left_out", p.networks_left_out);
        l.flag("dma_buf_imports", p.dma_buf_imports);
        l.flag("nv_checkpoints", p.nv_checkpoints);
        l.add(" retired=%s idle_import_high=%d%%", p.retired_mode == 2 ? "after a wait" : p.retired_mode == 1 ? "at once" : "kept",
              p.idle_import_high_pct);
        l.flag("allowance_at_count_rule", p.allowance_at_count_rule);
        line(l.text);
    }
    {
        PolicyLine l("can and measured:");
        l.flag("mirrors", p.mirrors, p.set(Sw::mirrors));
        l.add(" import_chunk=%u MB", (unsigned)p.import_chunk_mb);
        l.flag("ub_snapshot", p.ub_snapshot, p.set(Sw::ub_snapshot));
        l.flag("split_ub_flush", p.split_ub_flush, p.set(Sw::split_ub_flush));
        l.flag("native_float_add", p.native_float_add, false, f.nvidia() ? nullptr : "Intel 101.8991 to 101.9175 by family");
        l.flag("warp64", p.warp64);
        l.flag("mesh_shared_warp_default", p.mesh_shared_warp);
        l.flag("batch_submits", p.batch_submits, p.set(Sw::batch_submits));
        l.flag("mma_unroll_try", p.mma_unroll_try);
        l.flag("global_table_loop", p.global_table_loop);
        l.flag("occ_copy", p.occ_copy, p.set(Sw::occ_copy), "101.9154");
        l.flag("intel_side_by_side_auto", p.intel_side_by_side_auto, false, "101.9154");
        l.flag("fsr4_tune", p.fsr4_tune, p.set(Sw::fsr4_tune));
        l.flag("gpu_clock_peak", p.gpu_clock_peak);
        l.flag("presenter_queue", p.presenter_queue);
        l.flag("pcache_small_pc", p.pcache_small_pc);
        line(l.text);
    }
    {
        PolicyLine l("sizes by the allowance at the start (the sites read it later):");
        l.add(" textures %llu MB, mirrors %llu, target pool %llu, compaction from %llu, shadows %llu, staging kept %llu%s, spare"
              " captures ",
              mb(p.texture_budget), mb(p.mirror_budget), (unsigned long long)p.target_pool_mb, mb(p.compact_min_free),
              mb(p.wb_shadow_cap), mb(p.staging_keep), p.set(Sw::staging_keep) ? "*" : "");
        if (p.cap_spare_max == ~0ull) l.add("no limit");
        else l.add("%llu", mb(p.cap_spare_max));
        if (p.set(Sw::cap_spare_max)) l.add("*");
        l.add("; blocks at the start %u, of %llu MB (targets %llu); count line %u%%", (unsigned)p.image_blocks_at_start, mb(p.vma_block),
              mb(p.target_block), (unsigned)p.count_line_pct);
        l.flag("room_in_blocks_rule", p.room_in_blocks_rule);
        l.flag("squeeze_damped", p.squeeze_damped);
        l.flag("linux_hold_caps_budget", p.linux_hold_caps_budget);
        l.flag("small_vram_textures", p.small_vram_textures);
        l.flag("astc_bc_all", p.astc_bc_all);
        l.flag("dlss_cnn", p.dlss_cnn);
        line(l.text);
    }
    {
        PolicyLine l("auto settings and threads:");
        l.add(" auto_tier=%d", p.auto_tier);
        l.flag("auto_render_third", p.auto_render_third);
        l.flag("rtx", p.rtx);
        l.flag("rdna4", p.rdna4);
        l.add("; workers %u%s, capture helpers %u, texture threads %u (gather %u), program workers in play %u", p.workers,
              p.set(Sw::workers) ? "*" : "", p.capture_helpers, p.texture_threads, p.texture_gather_threads, p.program_workers_in_play);
        l.flag("threaded_auto_render", p.threaded_auto_render);
        l.flag("threaded_auto_recorder", p.threaded_auto_recorder);
        l.flag("drop_modules_small_ram", p.drop_modules_small_ram);
        l.flag("native_zstd_keep_core", p.native_zstd_keep_core);
        line(l.text);
    }
    {
        PolicyLine l("switches (* above: set by its switch; the other values are this PC's defaults):");
        l.add(" %s", p.switch_notes[0] ? p.switch_notes : "none of these is set");
        line(l.text);
    }
}

}  // namespace mkwvk
