// Guest memory for the GPU: each NVN memory pool imported as host memory (VK_EXT_external_memory_host) with one
// buffer over it. The GPU reads the game's memory where it is and its writes land there: no mirrored copies, no
// write-backs, no upload streams. A GPU address (a guest address for the native pools) maps to (buffer, offset).
//
// Pools are imported the first time something on the GPU touches them, not all at once: MKW has ~2,400 pools at the
// title and importing each is a buffer, a memory object and a device address (tens of seconds taken at one submit,
// on the render thread, in the first frame). The table of pools is cheap to keep current (host->gpu_pools: two
// numbers each) and is refreshed when its generation changes, at a submit.
#include <cstring>
#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <set>
#include <memory>
#include <new>
#include <mutex>
#include <vector>
#include <condition_variable>
#include "vk_menu.h"
#include "auto_settings.h"
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#if defined(__AVX2__)
#include <immintrin.h>   // (the capture's streaming copy)
#endif

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "vk_internal.h"
#ifdef TRACY_ENABLE
#include "../../host/hos_tracy.h"   // (HOOKSHOT_TRACY: the profiler's zones; nothing in any other build)
#endif
#include "vk_new.h"   // (NewNothrow)
#include "readback_early.h"  // (RX_VK_READBACK_EARLY's and RX_VK_LATE_DEFER's rules)
#include "vk_texture.h"   // (upload_command_buffer)
#include "vk_draw.h"      // (extern_guarded_shader_module, extern_guarded_compute)
#include "vk_ledger.h"    // (RX_VK_LEDGER)
#include "vk_lag.h"       // (RX_VK_LAG_LOG)
#include "vk_draw.h"      // (linux_pressure_shortfall)
#include "vk_gprof.h"      // (RX_VK_PASS_TIMING)
#include "import_hold.h"
#include "import_fence.h"
#include "import_reach.h"
#include "mirror_budget.h"   // (mirror_budget's rule, by the card)
#include "mirror_trim.h"
#include "wb_learn.h"
#include "capture_merge.h"    // (the capture copied ahead)
#include "mirror_inherit.h"   // (/ #1127: what a new mirror takes from the retired pools' mirrors)
#include "msync_diff.h"       // (the mirror syncs' diff)
#include "gone_ring.h"        // (the recording threads' pool table)
#include "shaders/msync_scatter_comp_spv.h"   // (the mirror syncs' scatter copy)
#include "retired_give_back.h"
#include "sparse_mirror.h"
#include "sparse_spare.h"
#include "linux_pressure.h"
#include "page_marks.h"   // (wb_ever: invalidate_quick)
#include "page_table_census.h"   // (the page table's windows, and the census of the pools outside them)
#include <atomic>
#include <unordered_set>
#ifdef __linux__
#include "pcie_link.h"        // (the link speed from sysfs)
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "vk_builtin_spv.h"   // (RX_VK_SHADER_DIR)
#include "shaders/wb_diff_comp_spv.h"
#include "shaders/late_fix_comp_spv.h"
#include "vram_pressure.h"
#ifndef HOOKSHOT_ENV_H   // (env_on: the switches' one rule)
#include "hookshot_env.h"
#endif
#include "vk_draw.h"   // (extern_guarded_shader_module, extern_guarded_compute)
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

namespace mkwvk {
// (RX_VK_POOL_FINALIZE: the captures of the game's submits queued, and those the replay took: a finalize waits for the ones before it)
std::atomic<uint64_t> submits_queued{0}, submits_taken{0};
namespace {

// Every raw imported-buffer destruction goes through here, including unpublished allocation rollback.
// Published imports reach it only at the existing completion/eviction gates; no additional GPU wait.
void destroy_import_buffer(VkBuffer buffer) {
    release_texel_buffer_views(buffer);
    vkDestroyBuffer(dev.device, buffer, nullptr);
}

struct Pool {
    uint64_t lo = 0, hi = 0;   // guest range imported (aligned outwards)
    uint64_t range_lo = 0, range_hi = 0;   // (the game's pool range it was found for: pool_of's last hit)
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::vector<VkDeviceMemory> pieces;   // (a pool imported as several dma-bufs: the ones after `memory`)
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    // its VRAM mirror (see "the VRAM mirror" below)
    enum : uint8_t { MIRROR_NONE, MIRROR_WANTED, MIRROR_ON, MIRROR_REFUSED } mstate = MIRROR_NONE;
    VkBuffer mirror = VK_NULL_HANDLE;
    VmaAllocation mirror_alloc = nullptr;
    uint64_t mirror_retry_epoch = 0;   // (non-zero when refused due to transient VRAM budget: retry once past this epoch)
    uint32_t synced_gen = 0;     // every page stamped up to this generation is in the mirror
    uint64_t synced_epoch = 0;   // the submission that last brought it up to date
    uint64_t sync_seg = 0;       // (and the parallel replay's segment whose uploads did)
    uint64_t last_value = 0;     // (the newest submission that bound it: its mirror is in use until that one is done)
    uint64_t direct_write_epoch = 0;   // (the part that last wrote it in guest memory, before it had a mirror)
    uint64_t retired_value = 0;  // (retired: the submission that may still use it)
    bool was_mirrored = false;   // (its mirror was given back: its pages stay in the captures, for the next one)
    // (hookshot) the bytes bound from it, at most 1 MB a binding (RX_VK_MIRROR_HEAT_CAP_KB), decayed every 16 submissions: which
    // pools the GPU reads most, for mirrors_by_heat. Added to without the lock (__atomic), read under it. Neither the whole bound
    // ranges nor the bindings alone: on a 4 GB RTX 3050 Ti a 160 MB pool bound ~0.6 times a submission, 57 MB a binding (21 GB of
    // ranges every 600 submissions), and a 132 MB one bound ~7 times, 1.1 MB a binding (4.7 GB); with the 132 MB one's mirror the
    // frame ran at 59-60 fps against 55 with the 160 MB one's (the GPU 12.3-13.6 ms against 15.0: a wide binding is not read
    // whole), and small pools bound a thousand times a frame for a few bytes each held the room by their count
    uint64_t heat = 0;
    // (hookshot) its mirror's VRAM refused by the driver (a full or cut-up 4 GB card): tried again from this submission on, as the
    // texture cache gives VRAM back, instead of never (RX_VK_MIRROR_HEAT=0: never, as before)
    uint64_t retry_epoch = 0;
    uint8_t refusals = 0;
    static uint64_t heat_of(uint64_t size) {
        static const uint64_t cap = [] {
            const char* e = getenv("RX_VK_MIRROR_HEAT_CAP_KB");
            return e && atoll(e) > 0 ? (uint64_t)atoll(e) << 10 : 1ull << 20;
        }();
        return std::min(size, cap);
    }
    // (hookshot) the GPU wrote it past what is copied back (memory_note_gpu_write: a bare entry's guessed range, the pointer
    // table's, a wide binding's): that is in its mirror alone, so the mirror is never given back (mirrors_give_back)
    bool gpu_only = false;
    // (and the GPU wrote it through a binding of its mirror, copied back or not: after a save reload, the mirrors of such pools
    // given back and made again left cyan in play and over the Start menu (RX 5500M, gone with every mirror kept); kept too)
    bool gpu_written = false;
    bool inherit_lost = false;   // (retired, and the pool that took it in made no mirror in time to copy its GPU contents)
    // (EvilOOT) retired with its mirror on: the pool that took it in copies the GPU's contents from it when its own mirror is
    // made (mirror_make), and until then the mirror stays (mirrors_give_back)
    bool inherit_pending = false;
    uint64_t retired_epoch = 0;
    int64_t retired_ms = 0;      // (retired: when, import_now_ms; where its import waits before going back, the wait counts from here)
    bool prefetched = false;     // (amd-fix) imported ahead by the background thread, not bound yet
    // (hookshot, mirror_trim.h) mostly outside every game pool since this submission (0: not); how often its range was trimmed
    // before; and, retired by a trim, its mirror's contents wanted by each of the pools made from its live parts
    uint64_t dead_since = 0;
    uint8_t trims = 0;
    uint64_t trim_epoch = 0;   // (the submission its range was last trimmed in)
    bool trimmed = false;
    // (hookshot) the write-back's diff: what guest memory holds of the pool, in VRAM, and the pages of it known to be so
    VkBuffer shadow = VK_NULL_HANDLE;
    VmaAllocation shadow_alloc = nullptr;
    VkDeviceAddress shadow_address = 0;
    std::vector<uint64_t> shadow_known;   // (a bit a page)
    // (hookshot, RX_VK_MIRROR_SPARSE: sparse_mirror.h) a sparse mirror: memory only in the blocks a game pool is in
    bool sparse = false;
    bool any_wanted = false;              // (in sparse_wanted: a binding reached a block with none)
    uint64_t sparse_block_bytes = 0, sparse_size = 0;   // (its block, and its buffer's size in whole blocks)
    uint32_t sparse_types = 0;            // (the memory types its blocks can be of)
    uint64_t sparse_made = 0;             // (the size its buffer was created with: the next mirror of it may take the buffer, sparse_spare.h)
    std::vector<VmaAllocation> blocks;    // (a block's memory; nullptr: none)
    std::vector<uint8_t> has;             // (1: it has memory. Read without the lock by the bindings' fast path: atomic_ref)
    std::vector<uint8_t> wanted;          // (blocks given memory at the next submission)
    std::vector<uint64_t> block_dead_since;     // (the submission a block with memory was first seen with no game pool in it; 0: one is)
    std::vector<VmaAllocation> released;  // (a block's memory given back but not freed yet: still bound, taken back as it is)
    std::vector<uint32_t> release_gen;    // (a block's releases and take-backs counted: only the latest dead_blocks entry acts)
    std::vector<uint64_t> last_bound;     // (the submission a binding last reached a block: atomic_ref, the bindings' fast path too)
    uint64_t resident_bytes = 0;
    std::vector<std::pair<uint64_t, uint64_t>> refill;   // (offset, size: written in guest memory where blocks had memory)
    bool shadow_wanted = false;   // (its write-back was worth one: one given back under VRAM pressure is made again)
};
void diff_gpu_wrote(uint64_t guest, uint64_t size);   // (below: the GPU may have written there)
uint64_t mem_rec_value = 0, mem_completed = 0;   // (the recording being made, and the GPU's progress)

struct GuestPool {
    uint64_t lo = 0, hi = 0;   // (aligned outwards the same way, so an import covers what the lookup found)
};

std::vector<GuestPool> guest_pools;   // by lo, from the host
std::atomic<uint64_t> guest_pools_gen{1};   // (bumped when that table is replaced: a span cached without the lock goes stale)
std::map<uint64_t, Pool> imported;    // by lo, made on first use
Pool* last_pool = nullptr;            // (the last pool_of hit: draws in a row bind buffers of the same pools)
extern Pool* pool_cache[4096];
void retire_pool(std::map<uint64_t, Pool>::iterator it);
size_t retired_pools_count();   // (import diagnostics, RX_VK_IMPORT_DIAG)
Pool* pool_lookup(uint64_t guest, uint64_t size);
// (RX_VK_NO_HOST_IMPORT: a pool whose import was given back, imported again where guest memory is needed through it; below)
enum { kAgainBind, kAgainWrite, kAgainSync, kAgainMake, kAgainRb, kAgainTexture, kAgainFree, kAgainWb, kAgainKinds };
bool import_again(Pool& p, int why);
// (RX_VK_POOL_MIRROR: pools made with no import; see "a mirror per pool, never imported" below)
int64_t st_pm_made = 0, st_pm_fills = 0, st_pm_fill_bytes = 0, st_pm_live_bytes = 0, st_pm_again[kAgainKinds] = {};
int64_t st_pm_held = 0, st_pm_held_bytes = 0;   // (B4: pools the drop was told about)
int64_t st_pm_why[6] = {};   // (a binding imported: no recording, refused, parallel segment, deferred, written here, the mirror not made)
bool staging_into_mirror(Pool& p, std::vector<VkBufferCopy>& runs, VkCommandBuffer cmd = VK_NULL_HANDLE);
// (hookshot) The spans of the imports in `imported`, for the scans down the table by start (import_reach.h): how far below an
// address one that reaches it may start. Kept by imported_add and imported_take, the only ways in and out (pools_mu held)
import_reach::Reach live_reach;
/// An import into the table, counted in live_reach (pools_mu held)
std::map<uint64_t, Pool>::iterator imported_add(const Pool& p) {
    const auto [it, added] = imported.emplace(p.lo, p);
    if (added) live_reach.add(p.hi - p.lo);
    return it;
}
/// An import out of the table, its node (and the Pool in it) kept where it is, no longer counted (pools_mu held)
std::map<uint64_t, Pool>::node_type imported_take(std::map<uint64_t, Pool>::iterator it) {
    live_reach.remove(it->second.hi - it->second.lo);
    return imported.extract(it);
}
#ifdef _WIN32
struct SrwMutex {   // (a slim reader/writer lock: half std::mutex's cost, taken for every buffer a draw binds)
    SRWLOCK l = SRWLOCK_INIT;
    void lock() { AcquireSRWLockExclusive(&l); }
    void unlock() { ReleaseSRWLockExclusive(&l); }
};
#else
// (off Windows SRWLOCK is a pthread rwlock, host/posix/windows.h: more atomic steps than a mutex for a lock only ever taken whole)
using SrwMutex = std::mutex;
#endif
SrwMutex pools_mu;
uint32_t generation = ~0u;
uint64_t align = 4096;
uint64_t max_import_size = UINT64_MAX;
int failures = 0;
ImportHolds import_holds;          // (issue 218: a failed import is not tried again for ImportHolds::kGapMs)
std::mutex import_holds_mu;
thread_local bool import_cooling = false;
thread_local VkResult import_result = VK_SUCCESS;

int64_t import_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
bool import_on_hold(uint64_t lo, uint64_t hi) {
    std::lock_guard lk(import_holds_mu);
    import_cooling = !import_holds.allow(lo, hi, import_now_ms());
    return import_cooling;
}
void import_remember_failure(uint64_t lo, uint64_t hi) {
    std::lock_guard lk(import_holds_mu);
    import_holds.deny_for(lo, hi, import_now_ms());
}

constexpr VkBufferUsageFlags kUsage =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;

#ifdef __linux__
// Guest memory handed to the GPU as a dma-buf over the game's own pages: the window is a memfd's (host/hos_main.c),
// and /dev/udmabuf makes a dma-buf of a range of it, which the device imports like any other. The pages are the same ones
// the game reads and writes (nothing is copied), pinned once when it is made. A host-pointer import (VK_EXT_external_memory_host)
// is a userptr buffer object to amdgpu, whose pages the kernel looks up again at every command submission, and RADV puts every
// buffer object in every submission: with ~1.5 GB of pools imported that was 15-45 ms a submission in Kokiri Forest,
// 3-8 fps with 11 ms of GPU work. A pool larger than udmabuf's size limit (/sys/module/udmabuf/parameters/size_limit_mb,
// 64 MB unless raised) is made of several dma-bufs of at most that size, bound one after another into a sparse buffer, so the
// pool is still one buffer with one device address and nobody has to raise the limit. RX_VK_NO_DMABUF=1: host pointers only,
// as before; RX_VK_DMABUF_PIECE_MB=<n>: pieces of at most n MB (to try the sparse path with the limit raised)
struct UdmabufCreate { uint32_t memfd, flags; uint64_t offset, size; };
constexpr unsigned long kUdmabufCreate = (1ul << 30) | (sizeof(UdmabufCreate) << 16) | ('u' << 8) | 0x42;   // (_IOW('u', 0x42, ...))
int dmabuf_memfd = -2, dmabuf_dev = -1, dmabuf_failures = 0;
uint64_t dmabuf_limit = 0, dmabuf_succeeded = 0;

void disable_dmabuf_imports(const char* why) {
    if (dmabuf_memfd >= 0) logf("vk: disabling udmabuf pool imports (%s); falling back to host-pointer imports", why);
    dmabuf_memfd = -1;
}

/// The memory of guest [lo, lo + size) as an imported dma-buf; VK_NULL_HANDLE when udmabuf or the import refused it
VkDeviceMemory dma_buf_memory(uint64_t lo, uint64_t size) {
    UdmabufCreate c{(uint32_t)dmabuf_memfd, 1 /* UDMABUF_FLAGS_CLOEXEC */, lo, size};
    const int fd = ioctl(dmabuf_dev, kUdmabufCreate, &c);
    if (fd < 0) {
        const int err = errno;
        if (dmabuf_failures++ < 8)
            logf("vk: guest %#llx..%#llx: no udmabuf (errno %d)", (unsigned long long)lo, (unsigned long long)(lo + size), err);
        if (!dmabuf_succeeded && (err == EINVAL || err == EBADFD || err == EPERM))
            disable_dmabuf_imports("udmabuf rejected guest memfd");
        return VK_NULL_HANDLE;
    }
    static uint32_t type = UINT32_MAX;   // (the same for every dma-buf of the window: asked once)
    if (type == UINT32_MAX) {
        VkMemoryFdPropertiesKHR fp{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
        if (vkGetMemoryFdPropertiesKHR(dev.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp) == VK_SUCCESS) {
            for (uint32_t i = 0; i < dev.memory.memoryTypeCount && type == UINT32_MAX; i++)   // (host visible, coherent, first)
                if ((fp.memoryTypeBits >> i & 1) &&
                    (dev.memory.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                    type = i;
            for (uint32_t i = 0; i < dev.memory.memoryTypeCount && type == UINT32_MAX; i++)
                if (fp.memoryTypeBits >> i & 1) type = i;
        }
        if (type == UINT32_MAX) {
            close(fd);
            disable_dmabuf_imports("driver reports no memory type for udmabuf");
            return VK_NULL_HANDLE;
        }
    }
    VkImportMemoryFdInfoKHR im{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    im.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    im.fd = fd;
    VkMemoryAllocateFlagsInfo fl{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    im.pNext = &fl;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &im;
    ai.allocationSize = size;
    ai.memoryTypeIndex = type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    const VkResult r = vkAllocateMemory(dev.device, &ai, nullptr, &memory);
    if (r != VK_SUCCESS) {   // (the fd is the caller's again when the import fails)
        close(fd);
        if (dmabuf_failures++ < 8)
            logf("vk: guest %#llx..%#llx: its dma-buf not imported (VkResult %d)", (unsigned long long)lo,
                 (unsigned long long)(lo + size), (int)r);
        if (!dmabuf_succeeded && r != VK_ERROR_OUT_OF_DEVICE_MEMORY)
            disable_dmabuf_imports("driver rejected udmabuf import");
        return VK_NULL_HANDLE;
    }
    dmabuf_succeeded++;
    return memory;
}

/// A pool larger than a dma-buf may be: a sparse buffer, each `piece` bytes of it bound to its own dma-buf
bool import_dma_buf_pieces(uint64_t lo, uint64_t hi, uint64_t piece, Pool& p) {
    VkExternalMemoryBufferCreateInfo eb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    eb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = &eb;
    bi.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT;
    bi.size = hi - lo;
    bi.usage = kUsage;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(dev.device, &bi, nullptr, &buffer) != VK_SUCCESS) {
        if (dmabuf_failures++ < 8) logf("vk: pool %#llx..%#llx: no sparse buffer for its dma-bufs", (unsigned long long)lo, (unsigned long long)hi);
        return false;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev.device, buffer, &req);
    // (each piece starts at a multiple of the sparse block size; the last runs to the end of what the buffer needs, past hi
    // when the size rounds up: the window's next pages, which the buffer never reaches)
    const uint64_t align = std::max<uint64_t>(req.alignment, 4096);
    piece = piece / align * align;
    const uint64_t end = lo + req.size;
    if (!piece || (req.size & 4095) || end > host->guest_size()) {
        destroy_import_buffer(buffer);
        if (dmabuf_failures++ < 8)
            logf("vk: pool %#llx..%#llx: sparse block %llu, size %llu: not in dma-bufs of %llu", (unsigned long long)lo,
                 (unsigned long long)hi, (unsigned long long)req.alignment, (unsigned long long)req.size, (unsigned long long)piece);
        return false;
    }
    std::vector<VkDeviceMemory> memories;
    std::vector<VkSparseMemoryBind> binds;
    using clk = std::chrono::steady_clock;   // (where the import's time goes, logged below: issue 14's 975 ms was the fence wait)
    const clk::time_point t0 = clk::now();
    clk::time_point t1 = t0, t2 = t0;
    for (uint64_t off = 0; off < req.size; off += piece) {
        const uint64_t n = std::min(piece, req.size - off);
        const VkDeviceMemory m = dma_buf_memory(lo + off, n);
        if (!m) break;
        memories.push_back(m);
        binds.push_back({off, n, m, 0, 0});
    }
    VkResult res = VK_ERROR_UNKNOWN;
    t1 = clk::now();
    if (binds.size() == (req.size + piece - 1) / piece) {
        VkSparseBufferMemoryBindInfo bb{buffer, (uint32_t)binds.size(), binds.data()};
        VkBindSparseInfo si{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
        si.bufferBindCount = 1;
        si.pBufferBinds = &bb;
        static VkFence fence = VK_NULL_HANDLE;   // (the background importer and the recording thread can both import)
        static std::mutex fence_mu;
        std::lock_guard fk(fence_mu);
        if (!fence) {
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(dev.device, &fi, nullptr, &fence);
        }
        if (dev.sparse_queue && dev.sparse_queue != dev.queue) {
            // (a queue of the driver's sparse family, RADV's: this function is its only user and fence_mu its lock. Nothing uses
            // the new buffer yet and the fence is waited for below, so the replay's queue and its held submits are left alone)
            res = vkQueueBindSparse(dev.sparse_queue, 1, &si, fence);
        } else {   // (on the queue, as the sparse textures' binds: nothing uses the new buffer yet, so no wait for earlier work)
            std::lock_guard qk(dev.present_mu);
            queue_flush_locked();
            vk_tl_note("bind-sparse-pool+");
            res = vkQueueBindSparse(dev.queue, 1, &si, fence);
            vk_tl_note("bind-sparse-pool-");
        }
        t2 = clk::now();
        if (res == VK_SUCCESS) {
            res = vkWaitForFences(dev.device, 1, &fence, VK_TRUE, UINT64_MAX);
            vkResetFences(dev.device, 1, &fence);
        }
    }
    {
        const clk::time_point t3 = clk::now();
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        static int logged;
        if (logged++ < 16)
            logf("vk: pool %#llx..%#llx: %zu dma-bufs bound in %.1f ms (udmabuf and imports %.1f, the bind %.1f, its fence %.1f) on %s",
                 (unsigned long long)lo, (unsigned long long)hi, memories.size(), ms(t0, t3), ms(t0, t1), ms(t1, t2), ms(t2, t3),
                 dev.sparse_queue ? "a queue of its own" : "the replay's queue");
    }
    if (res != VK_SUCCESS) {
        destroy_import_buffer(buffer);
        for (VkDeviceMemory m : memories) vkFreeMemory(dev.device, m, nullptr);
        if (dmabuf_failures++ < 8)
            logf("vk: pool %#llx..%#llx: its %zu dma-bufs not bound (VkResult %d)", (unsigned long long)lo, (unsigned long long)hi,
                 memories.size(), (int)res);
        return false;
    }
    p.lo = lo, p.hi = hi, p.buffer = buffer, p.memory = memories[0];
    p.pieces.assign(memories.begin() + 1, memories.end());
    return true;
}

void radv_import_headroom(uint64_t bytes);

bool import_dma_buf(uint64_t lo, uint64_t hi, Pool& p) {
    static uint64_t piece = 0;
    if (dmabuf_memfd == -2) {
        dmabuf_memfd = -1;
        typedef int (*MemfdFn)(void);
        const MemfdFn fn = env_on("RX_VK_NO_DMABUF") ? nullptr : (MemfdFn)dlsym(RTLD_DEFAULT, "hos_posix_guest_memfd");
        const int fd = fn ? fn() : -1;
        // (NVIDIA's driver offers a dma-buf of the window two memory types, host visible and coherent (dma_buf_memory's choice)
        // and one with no flags, and refuses the first: VK_ERROR_OUT_OF_DEVICE_MEMORY for every pool, the sparse ones unbound,
        // and its kernel module logs "NVRM: memMap_IMPL: CPU mapping not supported for addressSpace: 0x1" at each start. Every
        // pool was a host-pointer import after all, which on NVIDIA is not slow, so the attempt is not made there. RTX 4090,
        // 615.78.08, Kokiri Forest uncapped, two runs each in turn: 81.3 and 80.4 fps with the attempt, 81.5 and 79.6 without
        // it, no difference beyond the 1 to 2 % between identical runs)
        const bool nvidia = dev.driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
        int open_errno = 0;
        if (fd >= 0 && dev.dma_buf_import && vkGetMemoryFdPropertiesKHR && !nvidia) {
            dmabuf_dev = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
            if (dmabuf_dev < 0) open_errno = errno;
        }
        if (fd >= 0 && dmabuf_dev >= 0) {
            dmabuf_memfd = fd;
            dmabuf_limit = 64ull << 20;
            if (FILE* f = fopen("/sys/module/udmabuf/parameters/size_limit_mb", "r")) {
                unsigned long long mb = 0;
                if (fscanf(f, "%llu", &mb) == 1 && mb) dmabuf_limit = mb << 20;
                fclose(f);
            }
            piece = dmabuf_limit;
            if (const char* e = getenv("RX_VK_DMABUF_PIECE_MB"))
                if (atoll(e) > 0) piece = std::min<uint64_t>(piece, (uint64_t)atoll(e) << 20);
        }
        logf("vk: guest memory as dma-bufs: %s", dmabuf_memfd >= 0 ? "yes" : !fn ? "no (no memfd window, or RX_VK_NO_DMABUF)"
                 : fd < 0 ? "no (the window is anonymous memory)" : !dev.dma_buf_import ? "no (VK_EXT_external_memory_dma_buf)"
                 : open_errno == ENOENT ? "no (/dev/udmabuf is missing: the udmabuf kernel module is not loaded)"
                 : open_errno == EACCES || open_errno == EPERM ? "no (/dev/udmabuf is not open to this user)"
                 : nvidia ? "no (NVIDIA's driver refuses them: host pointers)" : "no (/dev/udmabuf cannot be opened)");
        // (without it every pool is a host-pointer import, which amdgpu looks up again at every submission: 3-8 fps in Kokiri.
        // Distributions differ: systemd's uaccess rule gives the logged-in user the device on some, others leave it to the kvm
        // group, root's alone. Said plainly, since this is the difference between playable and not)
        if (dmabuf_memfd < 0 && (open_errno == EACCES || open_errno == EPERM))
            logf("vk: WARNING: the game will run very slowly. Give your user /dev/udmabuf: sudo usermod -aG kvm $USER, then log "
                 "out and back in");
        else if (dmabuf_memfd < 0 && open_errno == ENOENT)
            logf("vk: WARNING: the game will run very slowly. Load the udmabuf kernel module: sudo modprobe udmabuf");
        // (without it every pool is a host-pointer import, which amdgpu looks up again at every submission: 3-8 fps in Kokiri.
        // Distributions differ: systemd's uaccess rule gives the logged-in user the device on some, others leave it to the kvm
        // group, root's alone. Said plainly, since this is the difference between playable and not)
        if (dmabuf_memfd < 0 && (open_errno == EACCES || open_errno == EPERM))
            logf("vk: WARNING: the game will run very slowly. Give your user /dev/udmabuf: sudo usermod -aG kvm $USER, then log "
                 "out and back in (docs/USER_GUIDE.md, Linux)");
        else if (dmabuf_memfd < 0 && open_errno == ENOENT)
            logf("vk: WARNING: the game will run very slowly. Load the udmabuf kernel module: sudo modprobe udmabuf "
                 "(docs/USER_GUIDE.md, Linux)");
        if (dmabuf_memfd >= 0)
            logf("vk: dma-bufs of at most %llu MB (udmabuf's limit %llu MB); larger pools %s", (unsigned long long)(piece >> 20),
                 (unsigned long long)(dmabuf_limit >> 20),
                 dev.sparse_buffers ? "are several, in a sparse buffer"
                                    : "CANNOT be imported (no sparse binding, and the window's shared pages are no host-pointer import)");
    }
    if (dmabuf_memfd < 0 || (lo & 4095) || (hi & 4095)) return false;
    if (hi - lo > piece && !dev.sparse_buffers) {
        // (hookshot: said once, with what helps: such a pool is no import at all, its buffers bind nothing and whatever reads or
        // writes them, the foliage culling first, gets zeros: grass and small plants flickering in and out)
        static std::atomic<int> told{0};
        if (told++ < 1)
            logf("vk: WARNING: pool %#llx..%#llx (%llu MB) is over udmabuf's %llu MB limit and this driver binds no sparse buffers: it "
                 "cannot be imported, and what reads or writes it gets zeros (plants flickering or missing). Close the game and raise "
                 "the limit, then start it again: echo 2048 | sudo tee /sys/module/udmabuf/parameters/size_limit_mb",
                 (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)((hi - lo) >> 20),
                 (unsigned long long)(dmabuf_limit >> 20));
        return false;
    }
    radv_import_headroom(hi - lo);
    if (hi - lo > piece) {
        if (!import_dma_buf_pieces(lo, hi, piece, p)) return false;
    } else {
        VkExternalMemoryBufferCreateInfo eb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        eb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.pNext = &eb;
        bi.size = hi - lo;
        bi.usage = kUsage;
        VkBuffer buffer = VK_NULL_HANDLE;
        if (vkCreateBuffer(dev.device, &bi, nullptr, &buffer) != VK_SUCCESS) return false;
        const VkDeviceMemory memory = dma_buf_memory(lo, hi - lo);
        // Allocation alone does not make the buffer usable. A failed bind owns both resources,
        // but must leave no partial import for the host-pointer fallback to publish.
        if (!memory || vkBindBufferMemory(dev.device, buffer, memory, 0) != VK_SUCCESS) {
            destroy_import_buffer(buffer);
            if (memory) vkFreeMemory(dev.device, memory, nullptr);
            return false;
        }
        p.lo = lo, p.hi = hi, p.buffer = buffer, p.memory = memory;
    }
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = p.buffer;
    p.address = vkGetBufferDeviceAddress(dev.device, &da);
    if (env_on("RX_VK_LOG_ADDR"))   // (as import_pool's: which buffer a device fault's address is in)
        logf("vk: addr dma-buf %#llx..%#llx at gpu %#llx..%#llx", (unsigned long long)lo, (unsigned long long)hi,
             (unsigned long long)p.address, (unsigned long long)(p.address + (hi - lo)));
    static int64_t n = 0, bytes = 0, sparse = 0;
    n++, bytes += (int64_t)(hi - lo), sparse += !p.pieces.empty();
    if (n <= 4 || n % 64 == 0 || (!p.pieces.empty() && sparse <= 4))
        logf("vk: pool %#llx..%#llx a dma-buf%s (%lld so, %lld in pieces, %lld MB)", (unsigned long long)lo, (unsigned long long)hi,
             p.pieces.empty() ? "" : "s in a sparse buffer", (long long)n, (long long)sparse, (long long)(bytes >> 20));
    return true;
}
#endif

int64_t st_imports_asked = 0;   // (every import asked for, whatever its path: RX_VK_POOL_MIRROR's must stay 0)
bool import_pool(uint64_t lo, uint64_t hi, Pool& p) {
    st_imports_asked++;
    import_cooling = false;
    import_result = VK_SUCCESS;
    if (import_on_hold(lo, hi)) return false;   // (tried recently and failed: not again until the hold ends)
    if (hi <= lo || hi - lo > max_import_size) {
        logf("vk: import %#llx..%#llx exceeds the device's allocation/buffer limit", (unsigned long long)lo,
             (unsigned long long)hi);
        import_remember_failure(lo, hi);
        return false;
    }
    p.lo = lo;
    p.hi = hi;
    uint8_t* ptr = host->guest_base() + lo;
    const uint64_t bytes = hi - lo;
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    clk::time_point t1, t2, t3, t4;
    struct Report {   // (where an import's time goes: logged for slow ones, 12 times)
        const clk::time_point &a, &b, &c, &d, &e;
        ~Report() {
            if (e.time_since_epoch().count() == 0) return;
            const double total = std::chrono::duration<double, std::milli>(e - a).count();
            static int logged;
            if (total < 2.0 || logged++ >= 12) return;
            auto ms = [](clk::time_point x, clk::time_point y) { return std::chrono::duration<double, std::milli>(y - x).count(); };
            logf("vk: import %.1f ms: commit+properties %.1f, buffer %.1f, allocate %.1f, bind+address %.1f", total, ms(a, b),
                 ms(b, c), ms(c, d), ms(d, e));
        }
    } report{t0, t1, t2, t3, t4};
    {   // (HOS_POOL_TRAP, host/hos_memsplit.c: pages a measurement made NOACCESS get their access back before the driver
        // locks them, which does not come through the host's handler. An exe without the export: nothing)
        using Release = void (*)(uint64_t, uint64_t, const char*);
        static const Release release = (Release)(void*)GetProcAddress(GetModuleHandleA(nullptr), "hos_pool_trap_release");
        if (release) release(lo, hi - lo, "import");
        // (HOS_NOACCESS_RAM: the host never gives back pages of a range imported, from before the commit below on: the driver
        // locks them, and a page decommitted under the import would leave the GPU on another page than the CPU)
        using Note = void (*)(uint64_t, uint64_t);
        static const Note note = (Note)(void*)GetProcAddress(GetModuleHandleA(nullptr), "hos_noaccess_import");
        if (note) note(lo, hi);
    }
    // (pages the game has not touched yet may be only reserved: committed now, as the pool's memory is usable)
    // (hookshot) One commit avoids the slow query walk, but resets protection on committed pages too. Keep the walk
    // for CPU write watches: commit only reserved pages, leaving the watched pages read-only.
    // ponytail: add new watch switches here; query the host if watches stop requiring a startup switch.
    static const bool page_watches = getenv("RX_NVN_WATCH_TEX") || getenv("RX_NVN_WATCH_TEXELS_SRC") ||
                                     getenv("RX_VK_WATCH_CPU") || getenv("RX_VK_VERIFY_DIM");
    // (HOS_GUEST_PAGES, host/hos_guestpages.c: not while a drop splits the window's reservation, which fails commits in it
    // meanwhile. An exe without the export, or the switch off: nothing)
    using Hold = void (*)(int);
    static const Hold hold = (Hold)(void*)GetProcAddress(GetModuleHandleA(nullptr), "hos_gp_hold");
    if (hold) hold(1);
    if (page_watches || VirtualAlloc(ptr, (SIZE_T)bytes, MEM_COMMIT, PAGE_READWRITE) != ptr)
        for (uint8_t* q = ptr; q < ptr + bytes;) {
            MEMORY_BASIC_INFORMATION mi{};
            if (VirtualQuery(q, &mi, sizeof mi) != sizeof mi) break;
            uint8_t* end = std::min((uint8_t*)mi.BaseAddress + mi.RegionSize, ptr + bytes);
            if (mi.State == MEM_RESERVE) VirtualAlloc(q, (size_t)(end - q), MEM_COMMIT, PAGE_READWRITE);
            q = end;
        }
    if (hold) hold(0);
#ifdef __linux__
    if (import_dma_buf(lo, hi, p)) {
        t1 = t2 = t3 = t4 = clk::now();
        return true;
    }
#endif
    // (the memory types this pointer can be imported as: asked for each pointer, the driver answers per pointer)
    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    // (each driver refusal leaves its VkResult in import_result, for pool_lookup's "not imported" line)
    if (const VkResult r = vkGetMemoryHostPointerPropertiesEXT(dev.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                                               ptr, &hp);
        r != VK_SUCCESS) {
        import_result = r;
        import_remember_failure(lo, hi);
        return false;
    }
    t1 = clk::now();
    VkExternalMemoryBufferCreateInfo eb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    eb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = &eb;
    bi.size = bytes;
    bi.usage = kUsage;
    if (const VkResult r = vkCreateBuffer(dev.device, &bi, nullptr, &p.buffer); r != VK_SUCCESS) {
        import_result = r;
        import_remember_failure(lo, hi);
        return false;
    }
    // (the type must also be one the buffer allows, and the buffer must fit in the imported bytes)
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev.device, p.buffer, &req);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < dev.memory.memoryTypeCount && type == UINT32_MAX; i++)   // (host visible, coherent)
        if ((hp.memoryTypeBits & req.memoryTypeBits) >> i & 1 &&
            (dev.memory.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            type = i;
    if (type == UINT32_MAX || req.size > bytes) {
        destroy_import_buffer(p.buffer);
        p.buffer = VK_NULL_HANDLE;
        import_remember_failure(lo, hi);
        return false;
    }
    t2 = clk::now();
    VkImportMemoryHostPointerInfoEXT im{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    im.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    im.pHostPointer = ptr;
    VkMemoryAllocateFlagsInfo fl{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    im.pNext = &fl;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &im;
    ai.allocationSize = bytes;
    ai.memoryTypeIndex = type;
    // (hookshot, import_fence.h) the driver's question about the pointer walks the heap from here to the end of its run of
    // read-write pages: the page after the range ends the run while it is asked. Not with the CPU write watches, which
    // change protections themselves. RX_VK_IMPORT_FENCE=0 (or off, false, no): off
    static const bool fence_on = [] {
        const char* v = getenv("RX_VK_IMPORT_FENCE");
        return !v || !env_value_off(v);
    }();
    VkResult allocated;
    {
        const ImportFence fence(ptr, bytes, fence_on && !page_watches);
        allocated = vkAllocateMemory(dev.device, &ai, nullptr, &p.memory);
    }
    t3 = clk::now();
    const VkResult bound = allocated == VK_SUCCESS ? vkBindBufferMemory(dev.device, p.buffer, p.memory, 0) : allocated;
    if (allocated != VK_SUCCESS && env_on("RX_VK_IMPORT_DIAG")) {
        // (diagnostics, RX_VK_IMPORT_DIAG=1: every refused import, with what overlaps it and the pages' Windows state)
        size_t live_over = 0;
        uint64_t live_bytes = 0;
        for (const auto& [k, q] : imported) {
            live_bytes += q.hi - q.lo;
            if (q.lo < hi && q.hi > lo) live_over++;
        }
        char regions[400];
        int rn = 0, nreg = 0;
        regions[0] = 0;
        for (uint8_t* q = ptr; q < ptr + bytes && nreg < 64;) {
            MEMORY_BASIC_INFORMATION mi{};
            if (VirtualQuery(q, &mi, sizeof mi) != sizeof mi) break;
            uint8_t* end = std::min((uint8_t*)mi.BaseAddress + mi.RegionSize, ptr + bytes);
            if (nreg < 6 && rn < (int)sizeof regions - 80)
                rn += snprintf(regions + rn, sizeof regions - rn, " [%llu KB state %#lx prot %#lx type %#lx]",
                               (unsigned long long)((end - q) >> 10), mi.State, mi.Protect, mi.Type);
            nreg++;
            q = end;
        }
        logf("vk: import diag: %#llx..%#llx (%llu KB, ptr %% 64K = %#llx) refused (%d); %zu live imports overlap it, %zu live "
             "(%llu MB), %zu retired; %d regions:%s", (unsigned long long)lo, (unsigned long long)hi,
             (unsigned long long)(bytes >> 10), (unsigned long long)((uintptr_t)ptr & 0xFFFF), (int)allocated, live_over,
             imported.size(), (unsigned long long)(live_bytes >> 20), retired_pools_count(), nreg, regions);
    }
    if (bound != VK_SUCCESS) {
        import_result = bound;
        destroy_import_buffer(p.buffer);
        if (allocated == VK_SUCCESS) vkFreeMemory(dev.device, p.memory, nullptr);
        p.buffer = VK_NULL_HANDLE, p.memory = VK_NULL_HANDLE;
        import_remember_failure(lo, hi);
        return false;
    }
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = p.buffer;
    p.address = vkGetBufferDeviceAddress(dev.device, &da);
    if (env_on("RX_VK_LOG_ADDR"))   // (diagnostics: which buffer a device fault's address is in)
        logf("vk: addr import %#llx..%#llx at gpu %#llx..%#llx", (unsigned long long)lo, (unsigned long long)hi,
             (unsigned long long)p.address, (unsigned long long)(p.address + (hi - lo)));
    t4 = clk::now();
    return true;
}

} // namespace

// (hookshot) guest memory's page table, for the pointers shaders follow (spirv_emit_context.cpp): a page's device address, 0 for none. Written as
// pools come and go, before the submissions that can reach them
uint64_t* guest_pages = nullptr;
uint64_t guest_pages_address = 0;
VkBuffer guest_pages_buffer = VK_NULL_HANDLE;
// (hookshot, 11-07) A change to the table made while a submission is recorded goes into that submission's upload commands, after
// the mirror copies it records there, instead of into the table at once: the table is read by the GPU, and work submitted
// before (still running, or queued) read the new entries early. A pool's range zeroed at its merge read as unmapped (zero
// counts: the culling's foliage gone for a frame), and a mirror made but not yet filled read as garbage: the foliage
// culling 53caac5f looped on a count from it until Windows reset the driver (the device lost of ~1 start in 5, always
// after pool merges, 2 s after the GPU stopped)
struct TableRun { uint64_t first, count, address; };
thread_local std::vector<TableRun> table_pending;
thread_local bool table_defer = false;
void guest_pages_set(uint64_t lo, uint64_t hi, uint64_t address) {
    if (!guest_pages) return;
    // (on by default since 11-08: 14 of 14 starts without the device lost, against ~1 in 7 written at once; one-frame flashes
    // 0-1 bright / 4 dark in two 40 s Kokiri runs, as many as without it, and the same fps. One earlier run with 21 dark ones was
    // the runs' spread, not this)
    // (two windows, as spirv_emit_context.cpp reads them: page_table_census.h)
    for (const PageWindow& w : kPageWindows) {
        const uint64_t a0 = std::max(lo, w.base) & ~0xFFFull, a1 = std::min(hi, w.base + w.bytes);
        if (a0 >= a1) continue;
        const uint64_t first = w.first + ((a0 - w.base) >> 12), count = (a1 - a0 + 0xFFF) >> 12;
        const uint64_t at = address ? address + (a0 - lo) : 0;
        if (table_defer) {
            table_pending.push_back({first, count, at});
            continue;
        }
        for (uint64_t i = 0; i < count; i++) guest_pages[first + i] = at ? at + (i << 12) : 0;
    }
}
void memory_table_begin() {
    if (!table_pending.empty()) {   // (left by a recording that never ended: written now)
        table_defer = false;
        for (const TableRun& r : table_pending)
            for (uint64_t i = 0; i < r.count; i++) guest_pages[r.first + i] = r.address ? r.address + (i << 12) : 0;
        table_pending.clear();
    }
    table_defer = guest_pages != nullptr;
}
// (at the end of the recording's uploads: the changes, the last one for each page, as transfers into the table; the
// upload command buffer's barriers order them after the work submitted before and before this submission's)
bool memory_table_flush() {
#ifdef TRACY_ENABLE
    HT_SCOPE("memory_table_flush");
#endif
    table_defer = false;
    if (table_pending.empty()) return false;
    VkCommandBuffer cmd = upload_command_buffer();
    GprofSpan gp(cmd, "ptable update (guest page table)");
    gp.items = 0;
    // (hookshot) a heap that has no room left (Windows' commit limit: 32 GB of RAM with a 2 GB page file, Hookshot at 24 GB as
    // Hyrule Field loaded) threw std::bad_alloc here, on the presenting thread, and nothing between it and the game's code
    // catches: the game closed. Then the changes are written into the table at once, as RX_VK_TABLE_NOW does: work
    // submitted before may read them early, against the game closing
    try {
        std::vector<uint64_t> batch;
        uint64_t batch_first = 0;
        const auto emit = [&] {
            vkCmdUpdateBuffer(cmd, guest_pages_buffer, batch_first * 8, batch.size() * 8, batch.data());
            gp.items++;
            gp.bytes += batch.size() * 8;
            batch.clear();
        };
        const auto append = [&](const TableRun& r, uint64_t first, uint64_t end) {
            if (!batch.empty() && batch_first + batch.size() != first) emit();
            while (first < end) {
                if (batch.empty()) batch_first = first;
                const size_t n = (size_t)std::min<uint64_t>(8192 - batch.size(), end - first);
                const size_t before = batch.size();
                if (before + n > batch.capacity())
                    batch.reserve(std::min<size_t>(8192, std::max(before + n, batch.capacity() * 2)));
                batch.resize(before + n);   // zero mappings stay zero
                if (r.address) {
                    const uint64_t address = r.address + ((first - r.first) << 12);
                    for (size_t i = 0; i < n; ++i) batch[before + i] = address + ((uint64_t)i << 12);
                }
                first += n;
                if (batch.size() == 8192) emit();   // vkCmdUpdateBuffer takes at most 64 KiB
            }
        };
        const bool ordered = std::adjacent_find(table_pending.begin(), table_pending.end(),
            [](const TableRun& a, const TableRun& b) { return a.first + a.count > b.first; }) == table_pending.end();
        if (ordered) {
            for (const TableRun& r : table_pending) append(r, r.first, r.first + r.count);
        } else {
            // Walk range starts in page order. Among overlapping writes the highest original
            // index owns the pages until it ends or another write starts.
            std::vector<size_t> starts;
            starts.reserve(table_pending.size());
            for (size_t i = 0; i < table_pending.size(); ++i) starts.push_back(i);
            std::sort(starts.begin(), starts.end(), [](size_t a, size_t b) {
                return table_pending[a].first < table_pending[b].first;
            });
            std::vector<size_t> active;
            size_t next = 0;
            uint64_t page = 0;
            while (next < starts.size() || !active.empty()) {
                if (active.empty()) page = table_pending[starts[next]].first;
                while (next < starts.size() && table_pending[starts[next]].first == page) {
                    active.push_back(starts[next++]);
                    std::push_heap(active.begin(), active.end());
                }
                const TableRun& r = table_pending[active.front()];
                const uint64_t end = r.first + r.count;
                if (end <= page) {
                    std::pop_heap(active.begin(), active.end());
                    active.pop_back();
                    continue;
                }
                const uint64_t until = next == starts.size() ? end : std::min(end, table_pending[starts[next]].first);
                append(r, page, until);
                page = until;
            }
        }
        if (!batch.empty()) emit();
    } catch (const std::bad_alloc&) {
        static std::atomic<int> said{0};
        uint64_t n = 0;
        for (const TableRun& r : table_pending) {
            n += r.count;
            for (uint64_t i = 0; i < r.count; i++) guest_pages[r.first + i] = r.address ? r.address + (i << 12) : 0;
        }
        table_pending.clear();
        if (said.fetch_add(1) < 8)
            logf("vk: page table: no memory for this recording's %llu page changes (the commit limit?): written at once",
                 (unsigned long long)n);
        return gp.items != 0;
    }
    table_pending.clear();
    return true;
}

uint64_t memory_guest_pages() {
    static std::mutex mu;
    std::lock_guard lk(mu);
    static bool tried = false;
    if (tried) return guest_pages_address;
    tried = true;
    // (the table is for Kokiri's lighting. The menus' blank text was the pointer rule in graphics stages, now compute only)
    VkBuffer b = VK_NULL_HANDLE;
    VmaAllocation a = nullptr;
    void* mapped = nullptr;
    const uint64_t bytes = kPageTableEntries * 8;
    if (!create_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       true, b, a, &mapped, true) || !mapped) {
        logf("vk: no page table of guest memory: pointers in no bound buffer read as 0");
        return 0;
    }
    std::memset(mapped, 0, bytes);
    log_alloc_where("page table of guest memory (read by shaders following pointers)", a);
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = b;
    guest_pages_address = vkGetBufferDeviceAddress(dev.device, &da);
    guest_pages_buffer = b;
    std::lock_guard pk(pools_mu);
    guest_pages = (uint64_t*)mapped;
    for (auto& kv : imported) guest_pages_set(kv.second.lo, kv.second.hi, kv.second.address);
    logf("vk: page table of guest memory at %#llx (%zu pools)", (unsigned long long)guest_pages_address, imported.size());
    return guest_pages_address;
}

bool memory_init() {
    if (!dev.external_memory_host) {
        logf("vk: no VK_EXT_external_memory_host: the guest memory cannot be imported (needed for now)");
        return false;
    }
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    hp.pNext = &p11;
    p11.pNext = &p13;
    p2.pNext = &hp;
    vkGetPhysicalDeviceProperties2(dev.physical, &p2);
    align = std::max<uint64_t>(hp.minImportedHostPointerAlignment, 4096);
    max_import_size = std::min(p11.maxMemoryAllocationSize, p13.maxBufferSize);
    logf("vk: guest memory imported in place (alignment %llu)", (unsigned long long)align);
    return true;
}

// (hookshot) The pools the page table cannot reach (page_table_census.h): refreshed with the pool table, the count at close
namespace {
std::mutex page_census_mu;
PageTableCensus page_census;
} // namespace
void page_table_census_log(const char* when) {
    std::lock_guard lk(page_census_mu);
    logf("%s", page_census.summary(when).c_str());
}

/// Refresh the guest's pool table (no Vulkan work: pools are imported as they are used)
std::atomic<int64_t> memsync_lock_wait_ns{0};
std::atomic<int64_t> memsync_part_ns[3];
void memory_sync() {
#ifdef TRACY_ENABLE
    HT_SCOPE("memory_sync");
#endif
    static std::mutex sync_mu;   // (called from every thread that touches memory)
    const auto w0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> sync_lk(sync_mu);
    struct Waited {   // (diagnostics: only the lock waits, added up; nothing else changes)
        std::chrono::steady_clock::time_point t;
        ~Waited() {}
    } waited{w0};
    memsync_lock_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waited.t).count();
    const uint32_t g = host->maps_generation();
    if (g == generation) return;
    generation = g;
    static thread_local std::vector<std::array<uint64_t, 2>> list(8192);
    const int n = host->gpu_pools(reinterpret_cast<uint64_t(*)[2]>(list.data()), (int)list.size());
    std::vector<GuestPool> next;
    next.reserve(n > 0 ? (size_t)n : 0);
    for (int i = 0; i < n; i++) {
        if (!list[i][1]) continue;
        GuestPool p;
        p.lo = list[i][0] & ~(align - 1);
        p.hi = (list[i][0] + list[i][1] + align - 1) & ~(align - 1);
        next.push_back(p);
    }
    std::sort(next.begin(), next.end(), [](const GuestPool& a, const GuestPool& b) { return a.lo < b.lo; });
    {   // (hookshot) a line for each pool that came outside the page table's windows, the first 30 of them
        std::lock_guard ck(page_census_mu);
        for (const std::string& line : page_census.refresh(next)) logf("%s", line.c_str());
    }
    const auto w1 = std::chrono::steady_clock::now();
    std::lock_guard lk(pools_mu);
    memsync_lock_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w1).count();
    guest_pools.swap(next);
    guest_pools_gen.fetch_add(1, std::memory_order_relaxed);
    last_pool = nullptr;
    std::fill(std::begin(pool_cache), std::end(pool_cache), nullptr);
    static int logged;
    if (logged++ < 8) logf_at(log_debug, "vk: %zu pools mapped", guest_pools.size());
}

// ---- page write generations ----
// Two levels over the guest's 512 GB (2^27 pages of 4 KB): chunks of 64K page generations made when a page of the
// chunk is first stamped, and the newest generation of each 2 MB region (512 pages), which lets a check skip a big
// texture's clean regions without looking at their pages. 0: never written.
namespace {
constexpr uint64_t kPageShift = 12, kChunkPages = 1u << 16, kRegionPages = 512;
constexpr uint64_t kPages = 1ull << 27;
std::vector<std::unique_ptr<uint32_t[]>> gen_chunks(kPages / kChunkPages);
std::vector<uint32_t> region_gen(kPages / kRegionPages);
// (and the generation every page of a region was written at, at least: a large stamp (a storage buffer a dispatch may
// write, hundreds of MB) marks whole regions, not 512 pages each; it took dispatches 8-150 ms)
std::vector<uint32_t> region_full(kPages / kRegionPages);
inline uint32_t load_u32(uint32_t& x) {
    return std::atomic_ref<uint32_t>(x).load(std::memory_order_acquire);
}
inline void store_u32(uint32_t& x, uint32_t v) {
    std::atomic_ref<uint32_t>(x).store(v, std::memory_order_release);
}
inline uint32_t page_gen(uint64_t p) {
    const auto& c = gen_chunks[p / kChunkPages];
    return std::max(c ? load_u32(c[p % kChunkPages]) : 0u, load_u32(region_full[p / kRegionPages]));
}
uint32_t gen_now = 1;
uint64_t pages_collected = 0;
// (hookshot) memory_collect_written takes every page's write mark, then marks the pages in the captured ranges again
// for the next capture. A submit's capture made on the game's thread in between (capture_scan) found those pages
// unmarked and left them out, and the work it was taken for read an older capture's copy of them: Kokiri Forest's
// ground mist gone for a frame in 16 to 89 of 3,600 frames standing there with the render thread replaying jobs as they
// come (the default), where a third of the captures came in that window; with whole frames (RX_VK_FRAME_GATE=1) the two
// met while an area loaded (26 and 36 of ~650 captures after a teleport to Kokiri Forest, 0 or 1 of ~700 standing
// there). The collection holds collect_mu over the window and the capture's scan takes it: a capture that comes then
// waits for the collection. On an Arc B580, in sessions with four area loads (~110,000 collections each): with whole
// frames the captures waited ~900 times, 0.25 to 0.32 s in all, the longest 1 and 12 ms (at a load); with jobs
// replayed as they come ~34,000 times, ~2.6 s, the longest ~21 ms at each load. Lock order: collect_mu, then
// cap_ranges_mu (capture_ranges_merged) and the host's map lock (collect_written's ranges, gpu_ptr). capture_scan takes
// collect_mu under its capture's order lock (submit_order_mu, sync_order_mu with RX_VK_ASYNC=0, or replay_mu from
// resume_queue), none of which is taken under collect_mu, and takes no lock under it
std::mutex collect_mu;

// (RX_VK_LOG_INHERIT=<hex guest address>: that page's stamps and the retired mirrors a new mirror takes over it)
uint64_t inherit_watch_page() {
    static const uint64_t v = getenv("RX_VK_LOG_INHERIT") ? strtoull(getenv("RX_VK_LOG_INHERIT"), nullptr, 16) >> kPageShift : ~0ull;
    return v;
}
void stamp_page(uint64_t p, uint32_t g) {
    if (p >= kPages) return;
    if (p == inherit_watch_page()) logf("vk: inherit watch: page %#llx stamped, gen %u", (unsigned long long)(p << kPageShift), g);
    auto& c = gen_chunks[p / kChunkPages];
    if (!c) {
        c.reset(new uint32_t[kChunkPages]);
        std::fill(c.get(), c.get() + kChunkPages, 0u);
    }
    store_u32(c[p % kChunkPages], g);
    region_gen[p / kRegionPages] = g;
}
} // namespace

namespace {
/// (hookshot) Whether addresses asked in ascending order fall in sorted, disjoint [lo, hi) ranges: a cursor that only
/// moves forward, one pass over the ranges for a whole scan, where a binary search for every page cost log2(ranges)
/// compares each (the collections give their pages in address order: host/rx_backend.c's collect_in walks sorted, merged
/// ranges, hos_gpu_guest_ranges' for memory_collect_written, and capture_make's own `mirrored` list for the pages of a
/// Capture, which capture_make's staging and take_capture walk). An address below the last one asked searches again: any
/// order, the same answer
struct RangeWalk {
    const std::vector<std::pair<uint64_t, uint64_t>>& r;
    size_t next = 0;   // (the first range that starts past the last address asked)
    uint64_t last = 0;
    bool in(uint64_t a) {
        if (a < last) next = (size_t)(std::upper_bound(r.begin(), r.end(), std::make_pair(a, ~uint64_t(0))) - r.begin());
        else
            while (next < r.size() && r[next].first <= a) next++;
        last = a;
        return next && a < r[next - 1].second;
    }
};
void cap_view_bump();   // (defined with cap_view_gen below)
}
namespace { int cap_only_gpu(); bool cap_excl_hit(uint64_t lo, uint64_t hi); void cap_born_take(std::vector<std::pair<uint64_t, uint64_t>>& out); }   // (RX_VK_CAP_ONLY_GPU: below)
// (RX_VK_CAP_ONLY_GPU's report of a read into a left-out range: the back end's record of the game's pool at g, below)
bool nvn_pool_record_at(uint64_t g, uint64_t* lo, uint64_t* hi, uint32_t* flags, uint64_t* made_ns);
// (RX_VK_CAP_ONLY_GPU: every pool the back end has a record of, page-aligned, appended to out; below)
void nvn_pool_record_ranges(std::vector<std::pair<uint64_t, uint64_t>>& out);
const std::vector<std::pair<uint64_t, uint64_t>>& capture_ranges_merged();
void shadow_written(const uint32_t* pages, int64_t n);   // (the write-back's diff, below: pages the captures took)
void shadow_written_collected(const uint32_t* pages, int64_t n);   // (the write-back's diff, below: pages the collection found)
void memory_collect_written() {
    if (cap_only_gpu() == 1) {   // (pools made since: their pages' earlier writes were dropped as heap memory's, see below)
        static thread_local std::vector<std::pair<uint64_t, uint64_t>> born;
        cap_born_take(born);
        for (const auto& [lo, hi] : born) memory_stamp(lo, hi - lo);
        born.clear();
    }
    uint32_t* pages = nullptr;
    int64_t n = 0;
    uint32_t ec_gen = 0;
    {   // (from taking the marks to marking the captured pages again: no capture scans in between, collect_mu)
        std::lock_guard lk(collect_mu);
        ec_gen = early_check_collect_begin();   // (RX_VK_LOG_EARLY_CHECK: the CPU's writes, from here on)
        n = host->collect_written(&pages);
        if (n > 0) {
            const uint32_t g = ++gen_now;
            // (the captured ranges once a call: a lock, a pool-list refresh and three searches for every dirty page
            // cost the replay thousands of lock cycles a submit, the cost that retired gpu_pages' lock below)
            RangeWalk in_capture{capture_ranges_merged()};
            const bool only_gpu = cap_only_gpu() == 1;
            for (int64_t i = 0; i < n; i++) {
                const uint64_t guest = (uint64_t)pages[i] << kPageShift;
                // (RX_VK_CAP_ONLY_GPU=1: heap memory between the pools of an imported chunk, which no GPU address reaches: not
                // stamped, so no mirror sync copies it either)
                if (only_gpu && cap_excl_hit(guest, guest + 4096)) continue;
                // (written after its submit's capture: the next capture takes it, as it is then)
                if (in_capture.in(guest)) {
                    uint64_t avail = 0;
                    if (const uint8_t* q = host->gpu_ptr(guest, &avail)) host->mark_written(q, 4096);
                    continue;
                }
                stamp_page(pages[i], g);
            }
        }
    }
    if (ec_gen) early_check_cpu_pages(pages, n, true, ec_gen);   // (outside collect_mu: it takes ec_mu)
    if (n > 0) {
        pages_collected += (uint64_t)n;
        cap_view_bump();
        // (hookshot) every page the CPU wrote, captured later or not: unknown to the write-back diff's shadows before this
        // recording's write-back. With RX_VK_CAPTURE=0 this is the only report of the game's writes (capture_make returns
        // first); with captures, it is the pages written after this submission's capture (PR 1109's review)
        shadow_written_collected(pages, n);
    }
    if (pages) host->free_pages(pages);
}
uint32_t memory_generation() {
    return gen_now;
}
bool memory_written_since(uint64_t guest, uint64_t size, uint32_t gen) {
    if (!size) return false;
    const uint64_t p0 = guest >> kPageShift, p1 = std::min((guest + size - 1) >> kPageShift, kPages - 1);
    for (uint64_t r = p0 / kRegionPages; r <= p1 / kRegionPages; r++) {
        if (region_gen[r] <= gen) continue;
        if (load_u32(region_full[r]) > gen) return true;
        const uint64_t a = std::max(p0, r * kRegionPages), b = std::min(p1, r * kRegionPages + kRegionPages - 1);
        for (uint64_t p = a; p <= b; p++)
            if (page_gen(p) > gen) return true;
    }
    return false;
}
void memory_written_runs(uint64_t guest, uint64_t size, uint32_t gen, std::vector<std::pair<uint64_t, uint64_t>>& out) {
    if (!size) return;
    const uint64_t p0 = guest >> kPageShift, p1 = std::min((guest + size - 1) >> kPageShift, kPages - 1);
    for (uint64_t r = p0 / kRegionPages; r <= p1 / kRegionPages; r++) {
        if (region_gen[r] <= gen) continue;
        const uint64_t a = std::max(p0, r * kRegionPages), b = std::min(p1, r * kRegionPages + kRegionPages - 1);
        for (uint64_t p = a; p <= b; p++) {
            if (page_gen(p) <= gen) continue;
            const uint64_t lo = std::max(p << kPageShift, guest) - guest;
            const uint64_t hi = std::min((p + 1) << kPageShift, guest + size) - guest;
            if (!out.empty() && out.back().second == lo) out.back().second = hi;
            else out.push_back({lo, hi});
        }
    }
}
void memory_drop_shadow(uint64_t guest, uint64_t size);
// (a GPU write: RX_VK_GPU_PAGES=1 notes the pages, and the mirror syncs that copy the CPU's version of one over the
// mirror are counted (the GPU's bytes in that page replaced by what guest memory or a capture held: a page the CPU
// and the GPU both write). ~1M a race, no effect seen; off by default: its lock on every dirty page cost the replay)
std::mutex gpu_pages_mu;
std::unordered_map<uint32_t, uint32_t> gpu_pages;   // page -> the generation of the GPU's last write
int64_t st_sync_gpu_pages, st_sync_gpu_live;
bool gpu_pages_on() {
    static const bool v = env_on("RX_VK_GPU_PAGES");
    return v;
}
// (hookshot, RX_VK_UB_SNAPSHOT's guard) The GPU writes this thread's recordings have noted that guest memory may not have
// yet: memory_stamp_gpu's (storage buffers a draw or a dispatch writes, buffer copies and fills, texture-to-buffer copies)
// and memory_note_gpu_write's (bare and wide storage ranges). A uniform range among them is bound in place, where the GPU
// reads it after that write in the queue's order, and not copied from guest memory and the captures, which do not have
// the GPU's bytes (uniform_range). The submission being recorded keeps its own in all its parts: split_part runs
// memory_begin_submission at every part, and a part's GPU work has not necessarily run when the next part is recorded; a
// burst's next submit goes on in the same recording. At a submission's start (memory_gpu_writes_begin: run_submit's new
// recording, the present's, bench_replay's) the last one's take its timeline value, and each is let go once the GPU has
// done its submission (mem_completed, as of the part being recorded): its bytes are in guest memory by then, copied back
// from the mirrors at each part's end or written there in place. Not covered: a range memory_note_gpu_write noted where
// the GPU writes past what is copied back (its pool keeps its mirror: gpu_only), once its submission is done. Kept for good,
// such a range would bind in place memory the game fills again itself, the rewritten uniforms the snapshot is for. Noted
// only with the snapshot on (uniform_snapshot_on: nothing else asks). A parallel replay is not made then (RX_VK_PARALLEL,
// off by default: par_snapshot_serial in vk_replay.cpp records each command buffer on the replay thread); a segment's
// thread notes nothing all the same, and the guard is not asked while segments record
struct GpuWrite { uint64_t lo, hi, value; };   // ([lo, hi) in guest memory; its submission's timeline value: UINT64_MAX while it is the one recorded)
thread_local std::vector<GpuWrite> gpu_writes;   // (by lo, apart: merged where they meet, under the later submission's value)
thread_local uint64_t gpu_writes_lo = UINT64_MAX, gpu_writes_hi = 0;   // (all of them within: most uniforms are elsewhere)
thread_local uint64_t gpu_writes_value = 0;   // (the timeline value the notes of the submission being recorded saw)
void memory_gpu_writes_begin() {
    uint64_t lo = UINT64_MAX, hi = 0;
    size_t kept = 0;
    for (GpuWrite w : gpu_writes) {
        if (w.value == UINT64_MAX) w.value = gpu_writes_value;   // (the last submission's own)
        if (w.value <= mem_completed) continue;                  // (done on the GPU: guest memory has the bytes)
        lo = std::min(lo, w.lo), hi = std::max(hi, w.hi);
        gpu_writes[kept++] = w;
    }
    gpu_writes.resize(kept);
    gpu_writes_lo = lo, gpu_writes_hi = hi, gpu_writes_value = 0;
}
void memory_gpu_writes_note(uint64_t guest, uint64_t size) {
    static const bool on = uniform_snapshot_on();
    if (!on || !guest || !size || par_active.load(std::memory_order_relaxed)) return;
    uint64_t lo = guest, hi = guest + size;
    gpu_writes_lo = std::min(gpu_writes_lo, lo), gpu_writes_hi = std::max(gpu_writes_hi, hi);
    gpu_writes_value = std::max(gpu_writes_value, mem_rec_value);
    // (the first that reaches lo, and those after it that begin by hi: one range with it, the submission being recorded's)
    const auto it = std::lower_bound(gpu_writes.begin(), gpu_writes.end(), lo, [](const GpuWrite& w, uint64_t at) { return w.hi < at; });
    auto end = it;
    for (; end != gpu_writes.end() && end->lo <= hi; ++end) lo = std::min(lo, end->lo), hi = std::max(hi, end->hi);
    if (it == end) {
        gpu_writes.insert(it, {lo, hi, UINT64_MAX});
    } else {
        *it = {lo, hi, UINT64_MAX};
        gpu_writes.erase(it + 1, end);
    }
}
/// 0: no GPU write guest memory may not have; 1: one in the submission being recorded; 2: one in an earlier submission the
/// GPU has not done yet
int memory_gpu_written(uint64_t guest, uint64_t size) {
    if (!guest || !size || guest >= gpu_writes_hi || guest + size <= gpu_writes_lo) return 0;
    int got = 0;   // (the first that ends past guest, and those after it that begin before the range's end)
    for (auto it = std::upper_bound(gpu_writes.begin(), gpu_writes.end(), guest, [](uint64_t at, const GpuWrite& w) { return at < w.hi; });
         it != gpu_writes.end() && it->lo < guest + size; ++it) {
        if (it->value == UINT64_MAX) return 1;
        if (it->value > mem_completed) got = 2;
    }
    return got;
}
bool memory_gpu_writes_any() { return !gpu_writes.empty(); }
void memory_stamp_gpu(uint64_t guest, uint64_t size) {
    if (!size) return;
    memory_gpu_writes_note(guest, size);
    memory_stamp(guest, size);
    if (!gpu_pages_on()) return;
    std::lock_guard lk(gpu_pages_mu);
    const uint64_t p0 = guest >> kPageShift, p1 = (guest + size - 1) >> kPageShift;
    if (p1 - p0 > 4096) return;   // (big copies: not tracked)
    for (uint64_t p = p0; p <= p1; p++) gpu_pages[(uint32_t)p] = gen_now;
    if (gpu_pages.size() > 2000000) gpu_pages.clear();
}
bool gpu_page(uint64_t page) {
    if (!gpu_pages_on()) return false;
    std::lock_guard lk(gpu_pages_mu);
    return gpu_pages.count((uint32_t)page) != 0;
}

// (RX_VK_PROTECT_BARE=1, an experiment for the one-frame foliage flicker and the floating trees) The ranges a shader writes
// through a table pointer (vk_draw.cpp: a bare entry, only its first 64 bytes marked written) are GPU-written pages the
// mirror syncs know nothing of: a CPU write anywhere on such a page copied the CPU's whole page over the GPU's records
// (trees drawn from records lost that way stretched into columns reaching the sky, and culled foliage drew garbage for
// a frame). Pages noted here keep the GPU's version when the CPU also wrote them since the
// pool's last sync. Nothing is copied back to guest memory (that overwrote the CPU's own data beside them once).
std::mutex prot_mu;
std::unordered_map<uint32_t, uint32_t> prot_pages;   // page -> gen_now when a shader last wrote through it
std::atomic<int64_t> st_prot_kept{0};
bool protect_bare_on() {
    static const bool v = getenv("RX_VK_PROTECT_BARE") && getenv("RX_VK_PROTECT_BARE")[0] == '1';
    return v;
}
bool prot_page_since(uint64_t page, uint32_t synced) {
    std::lock_guard lk(prot_mu);
    const auto it = prot_pages.find((uint32_t)page);
    return it != prot_pages.end() && it->second >= synced;
}
void memory_protect_gpu(uint64_t guest, uint64_t size) {
    if (!size || !protect_bare_on()) return;
    const uint64_t p0 = guest >> kPageShift, p1 = (guest + size - 1) >> kPageShift;
    if (p1 - p0 > 16384) return;   // (64 MB at most: a guess past that is not the GPU's records)
    std::lock_guard lk(prot_mu);
    for (uint64_t p = p0; p <= p1; p++) prot_pages[(uint32_t)p] = gen_now;
    if (prot_pages.size() > 4000000) prot_pages.clear();
}

void memory_stamp(uint64_t guest, uint64_t size) {
    if (!size) return;
    {   // (RX_VK_LOG_STAMP=<lo>:<hi> (hex guest addresses): the stamps there, 300 lines, with their callers)
        static const std::pair<uint64_t, uint64_t> range = [] {
            unsigned long long a = 0, b = 0;
            if (const char* e = getenv("RX_VK_LOG_STAMP")) sscanf(e, "%llx:%llx", &a, &b);
            return std::pair<uint64_t, uint64_t>{a, b};
        }();
        static int logged;
        if (range.second && guest < range.second && guest + size > range.first && logged++ < 300)
            logf("vk: stamp %#llx +%#llx (gen %u) from %p", (unsigned long long)guest, (unsigned long long)size, gen_now + 1,
                 __builtin_return_address(0));
    }
    memory_drop_shadow(guest, size);   // (a write the capture did not take: guest memory is newer than its copy)
    const uint32_t g = ++gen_now;
    const uint64_t p0 = guest >> kPageShift, p1 = std::min((guest + size - 1) >> kPageShift, kPages - 1);
    for (uint64_t p = p0; p <= p1;) {
        if (p % kRegionPages == 0 && p + kRegionPages - 1 <= p1) {   // (a whole region)
            if (inherit_watch_page() / kRegionPages == p / kRegionPages)
                logf("vk: inherit watch: region of page %#llx stamped whole, gen %u", (unsigned long long)(inherit_watch_page() << kPageShift), g);
            store_u32(region_full[p / kRegionPages], g);
            region_gen[p / kRegionPages] = g;
            p += kRegionPages;
            continue;
        }
        stamp_page(p++, g);
    }
    cap_view_bump();   // (a replay read's cached page is this generation's answer, not the one from before the write)
}
uint64_t memory_pages_collected() {
    return pages_collected;
}

// ---- submit-time capture ----
// MKW writes some of a frame's data in place right after its submit (the console's GPU has read it by then). The
// replay records a submit later, and the mirrors were filled from guest memory as it was then: a frame's terrain
// tiles read the next one's constants and flashed as walls. The pages are now copied as they are at the submit.
namespace {
struct Capture {
    std::vector<uint32_t> pages;     // (written since the last submit)
    std::vector<uint32_t> copied;    // (those copied, in order)
    std::vector<uint32_t> slots;     // (and where each is in the buffer: 4 KB a slot, the page's index in pages)
    // (copied by the submitting thread straight into the buffer the GPU copies the mirrors' pages from, in cached host
    // memory: the replay's own copy of them into one was ~0.6 ms a submit once the game's small pools were captured)
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    void* map = nullptr;
    uint64_t bsize = 0;
    size_t nbytes = 0;
    std::vector<uint32_t> deferred;   // (pages left to the GPU's copy from guest memory, capture_make)
    std::vector<uint32_t> late;      // (indices into pages: copied again by the replay once late_value is done)
    std::vector<uint64_t> late_pv;   // (each late page's own write-back, beside late: RX_VK_LATE_DEFER, late_defer_split)
    uint64_t late_value = 0;
    // (RX_VK_CAPTURE_HASH) the chunk hashes of each slot's bytes, made by the copy itself (16 a slot), and which slots have
    // them (0: none, the mirror sync hashes the slot as before)
    std::vector<uint64_t> hashes;
    std::vector<uint8_t> hok;
    // (RX_VK_CAPTURE_GRAN) slot i's written granules, bit k = its k-th 512 bytes hold the page's (0xff: all of it);
    // empty: every slot whole
    std::vector<uint8_t> gmask;
    uint64_t late_done = 0;          // (the GPU's progress when it was made: the late pages' fix)
    // (hookshot) "shadow staging": the texture pages of this capture, copied from guest memory into ordinary cached RAM too, for
    // take_capture's texture shadow to read instead of the buffer above (often write-combined here: CPU reads ~0.3 GB/s).
    // sidx[i] = the staging slot of pages[i] (4 KB a slot), or -1: that page is read from the buffer. Empty sidx: no staging.
    std::unique_ptr<uint8_t[]> stage;
    size_t stage_bytes = 0;
    std::vector<int32_t> sidx;
    bool buf_cached = false;   // (its capture buffer's memory is HOST_CACHED: for the staging log line)
};
// (the staging blocks, kept for the next capture: a fresh 20 MB block is ~5000 page faults on the game's thread)
std::mutex stage_mu;   // (a leaf lock)
std::vector<std::pair<std::unique_ptr<uint8_t[]>, size_t>> stage_free;
size_t stage_free_bytes = 0;
bool stage_get(Capture& c, size_t need) {   // (a block of at least need bytes into c: a kept one, else a new one)
    {
        std::lock_guard lk(stage_mu);
        size_t best = stage_free.size();
        for (size_t i = 0; i < stage_free.size(); i++)
            if (stage_free[i].second >= need && (best == stage_free.size() || stage_free[i].second < stage_free[best].second)) best = i;
        if (best < stage_free.size()) {
            c.stage = std::move(stage_free[best].first), c.stage_bytes = stage_free[best].second;
            stage_free_bytes -= c.stage_bytes;
            stage_free.erase(stage_free.begin() + best);
            return true;
        }
    }
    const size_t sz = (need + (1u << 20) - 1) & ~size_t((1u << 20) - 1);   // (rounded to 1 MB: blocks of near sizes fit each other)
    {
        const NewNothrow nothrow;   // (a refusal: no staging block, as before the new_handler, not its wait and stop)
        c.stage.reset(new (std::nothrow) uint8_t[sz]);
    }
    c.stage_bytes = c.stage ? sz : 0;
    return c.stage != nullptr;
}
void stage_put(Capture& c) {   // (the block back to the kept ones, at most four and 96 MB; else freed)
    c.sidx.clear();
    if (!c.stage) return;
    std::lock_guard lk(stage_mu);
    static const bool low_spec = env_on("RX_VK_LOW_SPEC");   // (RAM the tight resource there: no block kept)
    if (!low_spec && stage_free.size() < 4 && stage_free_bytes + c.stage_bytes <= (96ull << 20)) {
        stage_free_bytes += c.stage_bytes;
        stage_free.emplace_back(std::move(c.stage), c.stage_bytes);
    }
    c.stage.reset(), c.stage_bytes = 0;
}
std::mutex cap_mu;
std::deque<std::unique_ptr<Capture>> captures;
std::vector<std::unique_ptr<Capture>> spare;   // (taken ones, their buffers kept)
// ---- the captures' pages, for the mirror syncs ----
// A capture takes the pages written since the last submit; a pool its submission does not bind is synced at a later
// one, whose capture does not have those pages: copied from guest memory as it was at that replay, by when MKW had
// rewritten them for a later frame (it writes several submits ahead of the replay), a frame's objects drew with
// another one's transforms and instance data (the one-frame flashes). Each page a capture took is in a table by page:
// the capture (the generation its take stamped the pages with) and the page's slot in that capture's buffer. The
// entry holds while the page's generation is still that one (a later capture re-takes the page, a write the captures
// did not take (memory_stamp) makes guest memory newer) and the capture's buffer is kept (the last kKeep captures, and
// until the work that copies from it is done); a sync copies a dirty page from there, else from guest memory.
// (A flat table and generations: the hash maps' node allocations took ~0.7 ms a submit, the replay's.)
struct CapEntry { uint32_t gen, slot; };
constexpr uint64_t kCapChunkPages = 1u << 16;   // (a chunk of the table: 256 MB of guest memory, made when first used)
// the slot's top bit: a late page (a write-back still on the GPU when the capture copied it) merged at its sync,
// see "the late pages merged" below; cap_entry gives the slot without it, cap_entry_late the bit
constexpr uint32_t kCapLateBit = 0x80000000u;   // (a slot is an index into a capture's pages: under 65536)
std::vector<std::unique_ptr<CapEntry[]>> cap_table;
inline CapEntry cap_entry(uint64_t page) {
    const uint64_t c = page / kCapChunkPages;
    if (c >= cap_table.size() || !cap_table[c]) return {0, 0};
    CapEntry e = cap_table[c][page % kCapChunkPages];
    e.slot &= ~kCapLateBit;
    return e;
}
inline bool cap_entry_late(uint64_t page) {
    const uint64_t c = page / kCapChunkPages;
    if (c >= cap_table.size() || !cap_table[c]) return false;
    return (cap_table[c][page % kCapChunkPages].slot & kCapLateBit) != 0;
}
inline void cap_entry_set(uint64_t page, CapEntry e) {
    const uint64_t c = page / kCapChunkPages;
    if (c >= cap_table.size()) cap_table.resize(c + 1);
    if (!cap_table[c]) {
        cap_table[c].reset(new CapEntry[kCapChunkPages]);
        std::memset(cap_table[c].get(), 0, kCapChunkPages * sizeof(CapEntry));
    }
    cap_table[c][page % kCapChunkPages] = e;
}
// (each kept capture's copy on the GPU, by its generation)
struct CapHash { std::vector<uint64_t> h; std::vector<uint8_t> ok; };   // (a kept capture's slot hashes, see Capture)
struct CapBuf { uint64_t value; VkBuffer buffer; VmaAllocation alloc; uint64_t size; void* map; uint64_t last_use = 0;
                std::shared_ptr<CapHash> hash = {};
                std::shared_ptr<std::vector<uint8_t>> gmask = {};   // (see Capture::gmask; null: every slot whole)
                uint64_t used = 0; };   // (the bytes its capture filled, for RX_VK_RES_LOG)
// ---- (Hookshot) RX_VK_CAPTURE_GRAN=1, with a game built with A64_GRAN=1 (off): the capture by written granules ----
// A capture copied every page the game wrote since the last submit, whole: ~11,900 pages a frame at the Market (46 MB, at the
// memory bus's speed: 1.3-1.9 ms on the frame's loop), of which ~1 MB of words differs from their last copy. The lifted stores
// now mark the 512 bytes they write too (nx2/rtgen.py A64_MARK, host/hos_mem.c hos_gran_take), and a capture copies only the
// granules written, into the page's slot at their place: the rest of the slot is not the page's. What reads a slot follows:
// - a mirror sync copies the slot's granules only (the chunk mask 's diff already has);
// - a granule set holds what was written since the page's last capture THAT A SYNC COPIED: a pool synced later than the next
// capture of the page would miss the first one's granules, so a capture adds the granules of the page's entry while no sync
// has copied that entry (PageX::syn < the entry's generation), read from guest memory, where they are unchanged since (no
// mark came). That is what the whole page gave, for those granules;
// - a read of guest memory as submitted takes the slot's granules over what guest memory holds; a page read that way is
// captured whole from then on (PageX::full), as are the pages of a texture's storage (their shadow is a whole page), a page
// whose mark came without granules (given back by the host), and every page when the chunk diff, the pre-copy or the
// replay's own thread is on;
// - a late page's copies (the take's wait, the late fix) stay whole pages: more than the mask says is the page's, never less.
struct PageX { uint32_t syn; uint8_t gm; uint8_t full; uint16_t pad; };   // (the generation a sync copied it at; its entry's mask)
std::atomic<PageX*> pagex_chunks[(1ull << 27) / 65536];
inline PageX* pagex(uint64_t page, bool make) {
    if (page >= (1ull << 27)) return nullptr;
    std::atomic<PageX*>& c = pagex_chunks[page >> 16];
    PageX* p = c.load(std::memory_order_acquire);
    if (!p) {
        if (!make) return nullptr;
        PageX* n = (PageX*)calloc(65536, sizeof(PageX));
        if (c.compare_exchange_strong(p, n, std::memory_order_acq_rel)) p = n;
        else free(n);
    }
    return p + (page & 65535);
}
typedef void (*GranTakeFn)(const uint32_t*, int64_t, uint8_t*);
GranTakeFn gran_take_fn();
bool capture_gran_on();
std::atomic<int64_t> st_gran_pages{0}, st_gran_bytes{0}, st_gran_full{0}, st_gran_union{0}, st_gran_made_full{0};
inline uint32_t gran_chunk_mask(uint8_t m8) {   // (a granule is two of the diff's 256-byte chunks)
    uint32_t m = 0;
    for (int k = 0; k < 8; k++)
        if (m8 >> k & 1) m |= 3u << (2 * k);
    return m;
}
std::vector<std::shared_ptr<CapHash>> cap_hash_free;   // (pools_mu: retired captures' hash arrays, their capacity kept)
std::map<uint32_t, CapBuf> cap_bufs;
uint64_t cap_bytes = 0, cap_used = 0;   // (pools_mu: the kept buffers' sizes, and what their captures filled of them)
constexpr size_t kKeep = 24;
constexpr uint64_t kCapBytesMax = 768ull << 20;   // (and no more than this in them: loads capture hundreds of MB)
// (retired buffers, for the next captures: making and freeing a 5 MB buffer every submit cost the replay too)
struct FreeBuf { VkBuffer buffer; VmaAllocation alloc; void* map; uint64_t size; uint64_t at = 0; uint64_t ms = 0; };
std::vector<FreeBuf> cap_free;
uint64_t cur_cap_value = 0;   // (the submission being recorded: what its syncs' copies wait for)
uint32_t cur_cap_gen = 0;     // (its capture's generation)
std::atomic<bool> cap_shutting_down{false};     // (set under cap_free_mu when the process starts to exit: after it no worker touches cap_free)
std::mutex cap_free_mu;   // (the submitting threads take buffers from cap_free, the replay gives them back)
uint64_t cap_free_bytes = 0;   // (cap_free_mu)
// (the spare buffers: 12 and, on a card of 6 GB or less, 32 MB of them; RX_VK_CAP_SPARE_MB=<n>: that many, 0: as before.
// After a load the retired ones, 40-60 MB each, stayed as spares for good)
uint64_t cap_spare_max() {
    static const uint64_t v = []() -> uint64_t {
        if (dev.policy.set(Sw::cap_spare_max)) return dev.policy.cap_spare_max;   // (the switch, read by device_policy_env)
        // (the rule is device_policy.h's, asked with the allowance now, at the first call, and not the one at the start)
        uint64_t used = 0, allowed = 0;
        return policy_rule::cap_spare_max(vram_usage(&used, &allowed) ? allowed : 0);
    }();
    return v;
}
// (off by default) RX_VK_CAP_SPARE_AGE=<n>: a spare no capture took for n captures is freed. The 12 spares held 195-385 MB of
// host memory in the Market; with 600 they held 89-411 MB and the submits made 1.5-2x as many buffers (each on the game's thread):
// the spares are taken again, not left over from the load. 0 (default): kept, as before
uint64_t cap_gets = 0;   // (cap_free_mu: buffers asked for)
uint64_t cap_spare_age() {
    static const uint64_t v = getenv("RX_VK_CAP_SPARE_AGE") ? (uint64_t)atoll(getenv("RX_VK_CAP_SPARE_AGE")) : 0;
    return v;
}
uint64_t cap_spares_aged = 0;   // (cap_free_mu)
// (Hookshot, 2026-11-14) RX_VK_CAP_SPARE_SECS=<s> (20 by default; 0: kept, as before): a spare no capture took for that long is
// freed. The spares are what a load's captures left (40-64 MB each): after the scene had loaded, 10 of the 11 were never taken
// again (the fewest spares left after a take, RX_VK_RES_LOG), 276-400 MB of host-cached memory in the three scenes on the RTX
// 5070. Unlike RX_VK_CAP_SPARE_AGE (counted in takes, ~1000 a second: 600 freed spares the next frames took again) it is time:
// a spare left unused for 20 s is not part of the scene's steady need
uint64_t cap_spare_ms() {
    static const uint64_t v = (getenv("RX_VK_CAP_SPARE_SECS") ? (uint64_t)std::max(0, atoi(getenv("RX_VK_CAP_SPARE_SECS"))) : 20) * 1000;
    return v;
}
// (Hookshot, 2026-11-14) RX_VK_CAP_FIT=1 (default; 0: as before): a capture takes a spare at most twice its need's size class
// (cap_buf_size), else a buffer of its size is made; and a spare returned when 12 are kept takes the largest one's place if it is
// smaller. Best fit alone gave the load's 40-64 MB spares to captures of a few MB: 68-886 such takes per 600 frames on the RTX
// 5070, and the 25 kept buffers (each held for 24 captures, for the mirror syncs) were 283-309 MB of which their captures filled
// 150-171 MB
bool cap_fit() {
    static const bool v = env_on("RX_VK_CAP_FIT");   // (off by default here: =1 on)
    return v;
}
uint64_t cap_now_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// (hookshot, RX_VK_CAP_FREE_ASYNC, on; =0: as before) A capture buffer let go past the spares is destroyed on a thread of its
// own, not on the thread that retired it (the replay's take, holding pools_mu): freeing a 64-256 MB host-cached memory object is
// the driver unmapping and releasing every page of it. Only buffers the GPU is done with get here (a retire waits for that), and
// nothing else holds them; the memory goes back a moment later. At the process's exit (cap_shutting_down) what is queued is left
bool cap_free_async_on() {
    static const bool on = [] {
        const bool v = env_on_or("RX_VK_CAP_FREE_ASYNC", 1);
        if (!v) logf("vk: capture buffers past the spares destroyed on the replay's thread, as before (RX_VK_CAP_FREE_ASYNC=0)");
        return v;
    }();
    return on;
}
std::mutex cap_gone_mu;   // (a leaf: the queue alone)
std::condition_variable cap_gone_cv;
std::vector<std::pair<VkBuffer, VmaAllocation>> cap_gone;
bool cap_gone_started = false;
void destroy_buffer_later(VkBuffer buffer, VmaAllocation alloc) {
    {
        std::lock_guard lk(cap_gone_mu);
        cap_gone.push_back({buffer, alloc});
        if (!cap_gone_started) {
            cap_gone_started = true;
            try {
                std::thread([] {
                    std::vector<std::pair<VkBuffer, VmaAllocation>> batch;
                    for (;;) {
                        {
                            std::unique_lock lk(cap_gone_mu);
                            cap_gone_cv.wait(lk, [] { return !cap_gone.empty(); });
                            batch.swap(cap_gone);
                        }
                        for (const auto& [b, a] : batch)
                            if (!cap_shutting_down.load()) destroy_buffer(b, a);
                        batch.clear();
                    }
                }).detach();
            } catch (...) {   // (no thread: destroyed here, as before)
                cap_gone_started = false;
                cap_gone.pop_back();
                destroy_buffer(buffer, alloc);
                return;
            }
        }
    }
    cap_gone_cv.notify_one();
}
void cap_buf_free(VkBuffer buffer, VmaAllocation alloc, void* map, uint64_t size, bool from_worker = false) {
    FreeBuf old[12];
    int n_old = 0;
    const uint64_t now = cap_now_ms();
    {
    std::lock_guard lk(cap_free_mu);
        if (const uint64_t ms = cap_spare_ms())
            for (size_t i = 0; i < cap_free.size();)
                if (now - cap_free[i].ms > ms && n_old < 12) {
                    old[n_old++] = cap_free[i];
                    cap_free_bytes -= cap_free[i].size;
                    cap_free.erase(cap_free.begin() + (ptrdiff_t)i);
                    cap_spares_aged++;
                } else {
                    i++;
                }
        if (const uint64_t age = cap_spare_age())
            for (size_t i = 0; i < cap_free.size();)
                if (cap_gets - cap_free[i].at > age && n_old < 12) {
                    old[n_old++] = cap_free[i];
                    cap_free_bytes -= cap_free[i].size;
                    cap_free.erase(cap_free.begin() + (ptrdiff_t)i);
                    cap_spares_aged++;
                } else {
                    i++;
                }
        // (RX_VK_CAP_FIT: a smaller one takes the place of the largest spare left untaken for 2 s. Not any largest: a castle walk's
        // ~50 MB capture a frame lost its spare to the small ones and made a buffer every frame, ~1000 in a run)
        if (cap_fit() && cap_free.size() >= 12 && size <= (64ull << 20)) {
            size_t big = cap_free.size();
            for (size_t i = 0; i < cap_free.size(); i++)
                if (now - cap_free[i].ms > 2000 && (big == cap_free.size() || cap_free[i].size > cap_free[big].size)) big = i;
            if (big < cap_free.size() && cap_free[big].size > size && n_old < 12) {
                old[n_old++] = cap_free[big];
                cap_free_bytes -= cap_free[big].size;
                cap_free.erase(cap_free.begin() + (ptrdiff_t)big);
            }
        }
    if (from_worker && cap_shutting_down.load()) return;
    // (hookshot, RX 7800 XT: buffers up to 256 MB are kept for the next capture, not only up to 64 MB: a new one costs ~100 ms of
    // vkAllocateMemory on AMD's driver)
    if (cap_free.size() < 12 && size <= (256ull << 20) && cap_free_bytes + size <= std::min<uint64_t>(cap_spare_max(), 768ull << 20)) {
            cap_free.push_back({buffer, alloc, map, size, cap_gets, now});
        cap_free_bytes += size;
            buffer = VK_NULL_HANDLE;
        }
    }
    if (buffer) {
        if (cap_free_async_on() && !cap_shutting_down.load()) destroy_buffer_later(buffer, alloc);
        else destroy_buffer(buffer, alloc);
    }
    for (int i = 0; i < n_old; i++) {
        if (cap_free_async_on() && !cap_shutting_down.load()) destroy_buffer_later(old[i].buffer, old[i].alloc);
        else destroy_buffer(old[i].buffer, old[i].alloc);
    }
}
void cap_buf_retire(std::map<uint32_t, CapBuf>::iterator it) {
    if (it->second.hash && cap_hash_free.size() < 32) cap_hash_free.push_back(std::move(it->second.hash));
    cap_bytes -= it->second.size;
    cap_used -= it->second.used;
    cap_buf_free(it->second.buffer, it->second.alloc, it->second.map, it->second.size);
    cap_bufs.erase(it);
}
void cap_free_trim(uint64_t keep_bytes) {
    std::lock_guard lk(cap_free_mu);
    while (!cap_free.empty() && cap_free_bytes > keep_bytes) {
        const FreeBuf f = cap_free.back();
        cap_free.pop_back();
        cap_free_bytes -= std::min(cap_free_bytes, f.size);
        destroy_buffer(f.buffer, f.alloc);
    }
}
// (a capture's buffer by its size class: 2, 4, 8, then 16 MB and up to the MB. It was never under 16 MB, for captures of ~1 MB:
// 25 kept ones were 413 MB of host-cached system memory, all resident, and three to a 64 MB block left 14 MB of each unused.
// RX_VK_CAP_MIN_MB=<n>: the smallest (2), 16: as before. From 16 MB each has a memory object of its own: RX_VK_CAP_BLOCKS=1: not)
uint64_t cap_buf_size(uint64_t need) {
    static const uint64_t min = (getenv("RX_VK_CAP_MIN_MB") ? (uint64_t)std::max(1, atoi(getenv("RX_VK_CAP_MIN_MB"))) : 2u) << 20;
    uint64_t s = min;
    while (s < need && s < (16ull << 20)) s <<= 1;
    if (s >= need) return s;
    // On a 32 MiB spare budget, rounding a 17 MiB capture up to 32 MiB leaves no room for smaller spares.
    if (need <= (32ull << 20) && cap_spare_max() <= (32ull << 20)) {
        constexpr uint64_t step = 4ull << 20;
        return (need + step - 1) & ~(step - 1);
    }
    // (hookshot: above 16 MB in classes of 32, 64, 128 and 256 MB, so that a buffer given back fits the next capture)
    for (const uint64_t cm : {32, 64, 128, 192, 256})   // (stutter: 192 added, a 130-158 MB capture took the 256 MB class)
        if ((uint64_t)cm << 20 >= need) return (uint64_t)cm << 20;
    return (need + (1u << 20) - 1) & ~((1ull << 20) - 1);
}
// (hookshot) The mirror syncs' pages from the captures go to the mirrors by the write-back diff's compute pass (msync_pass_add,
// after memory_end_submission), not as copy regions. On unless set off (RX_VK_MSYNC_PASS=0, the switches' one rule: hookshot_env.h)
bool msync_pass_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_MSYNC_PASS");
        const bool v = !e || !env_value_off(e);
        logf("vk: mirror syncs' pages from the captures %s (RX_VK_MSYNC_PASS)",
             v ? "by a compute pass" : "as copy regions, as before");
        return v;
    }();
    return on;
}
bool late_fix_on();   // (below)
// (a capture's buffer: read by the mirror syncs' copies, and by their pass, by address, when it is on; storage and an address
// too with the late pages' fix, which writes the write-backs' words into it)
VkBufferUsageFlags cap_buf_usage() {
    static const VkBufferUsageFlags u = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        (msync_pass_on() || late_fix_on() ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0u);
    return u;
}
// (hookshot, on since the stutter fix) The big capture buffers made ahead, on a worker thread. A capture of a scene load holds
// 16 to 256 MB of the pages the game wrote (texture storage), and its buffer was made on the game's thread, inside capture_make:
// AMD's driver commits and clears a host-visible memory object when it is allocated, ~0.8 ms a MB, and the log's isolated spikes
// ("memory sync ... capture 66.8", "capture 107.7", "capture 22.6", "capture 17.4": 5 in a session, each at a load, none in steady
// play) are that, in proportion to the class (128, 64 and 32 MB classes, and 128, 64 MB). The spares only come back from a capture
// retired past the 24 kept ones, of whatever size that one was, so a big class is rarely there when a load asks for it. With this
// a spare of each class of 32, 64 and 128 MB is made in the background once, and again as soon as one is taken.
// (The log at the scene loads without it: "capture ... the buffer of 32/64/256 MB 13-41 ms")
std::atomic<int> cap_prewarm_busy{0};
std::atomic<int> cap_prewarm_class_busy[4]{};   // (32, 64, 128 and 192 MB spares being made)
void cap_exit_note() {
    std::lock_guard lk(cap_free_mu);   // (a worker inside cap_buf_free finishes first; one after this sees the flag under the same mutex)
    cap_shutting_down = true;
}
int cap_prewarm_index(uint64_t size) { return size <= (32ull << 20) ? 0 : size <= (64ull << 20) ? 1 : size <= (128ull << 20) ? 2 : size <= (192ull << 20) ? 3 : -1; }
void cap_spare_make(uint64_t size, int index) {
    VkBuffer b = VK_NULL_HANDLE;
    VmaAllocation a = nullptr;
    void* m = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    if (!cap_shutting_down.load() && create_buffer_cached(size, cap_buf_usage(), b, a, &m, !env_on("RX_VK_CAP_BLOCKS")) && m) {
        static std::atomic<int> told{0};
        if (told++ < 8)
            logf("vk: a %llu MB capture buffer made ahead in %.1f ms, off the game's thread", (unsigned long long)(size >> 20),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        // (a spare, within the same limits as any other: past them it is let go. Not once the process is exiting: the list and the device
        // may be going, and the memory goes with the process)
        cap_buf_free(b, a, m, size, true);
    }
    cap_prewarm_class_busy[index]--;
    cap_prewarm_busy--;
}
/// A spare of at least `size` (32, 64 or 128 MB classes only) is there, or one of its class is being made (cap_free_mu held)
void cap_prewarm_class(uint64_t size) {
    const int index = cap_prewarm_index(size);
    if (index < 0 || size > cap_spare_max()) return;   // (not the 256 MB class, and not past what spares may hold)
    static const uint64_t kCls[4] = {32, 64, 128, 192};
    const uint64_t cls = kCls[index];
    const uint64_t bytes = cls << 20;
    for (const FreeBuf& f : cap_free)
        if (f.size >= bytes) return;
    if (cap_prewarm_class_busy[index].load() > 0 || cap_prewarm_busy.load() >= 3) return;   // (three at a time: each commits its memory)
    static const bool exit_hook = [] { return std::atexit(cap_exit_note) == 0; }();
    (void)exit_hook;
    cap_prewarm_busy++;
    cap_prewarm_class_busy[index]++;
    try {
        std::thread(cap_spare_make, bytes, index).detach();
    } catch (...) {   // (no thread: no spare, the capture goes on)
        cap_prewarm_class_busy[index]--;
        cap_prewarm_busy--;
    }
}
uint64_t cap_bufs_made = 0;   // (capture buffers made, for the log: a spare that was missed)
// (cap_free_mu; RX_VK_RES_LOG, since its last line) the largest need, the takes of a spare over twice the need's size class,
// and the fewest spares left after a take
uint64_t cap_need_max = 0, cap_taken_over = 0, cap_free_low = ~0ull;
bool cap_buf_get(uint64_t need, VkBuffer& buffer, VmaAllocation& alloc, void*& map, uint64_t& size) {
    std::lock_guard lk(cap_free_mu);
    cap_gets++;
    cap_need_max = std::max(cap_need_max, need);
    static bool first = true;
    if (first) {   // (the first capture: the classes a load asks for)
        first = false;
        for (uint64_t c : {32ull << 20, 64ull << 20, 128ull << 20, 192ull << 20}) cap_prewarm_class(c);
    }
    size_t best = cap_free.size();
    for (size_t i = 0; i < cap_free.size(); i++)
        if (cap_free[i].size >= need && (best == cap_free.size() || cap_free[i].size < cap_free[best].size)) best = i;
    if (best < cap_free.size() && cap_fit() && cap_free[best].size > 2 * cap_buf_size(need))   // (RX_VK_CAP_FIT: counted as over)
        best = cap_free.size(), cap_taken_over++;
    if (best < cap_free.size()) {
        buffer = cap_free[best].buffer, alloc = cap_free[best].alloc, map = cap_free[best].map, size = cap_free[best].size;
        cap_free_bytes -= size;
        cap_free.erase(cap_free.begin() + (ptrdiff_t)best);
        if (size > 2 * cap_buf_size(need)) cap_taken_over++;
        cap_free_low = std::min<uint64_t>(cap_free_low, cap_free.size());
        return map != nullptr;
    }
    cap_free_low = 0;
    size = cap_buf_size(need);
    map = nullptr;
    return create_buffer_cached(size, cap_buf_usage(), buffer, alloc, &map,
                                size >= (16ull << 20) && !env_on("RX_VK_CAP_BLOCKS")) && map;
}
std::vector<std::pair<uint64_t, uint64_t>> cap_ranges;   // (the mirrored pools the captures take, as of the last one)
std::mutex cap_ranges_mu;
uint64_t cap_ranges_ver = 1;   // (any of the captures' range lists changed: cap_ranges_mu)
// doc: docs/design.md#4-the-lessons-from-mkw-as-rules-here (rule 2: nothing the replay reads comes from live guest memory)
// (and the game's pools up to RX_VK_CAPTURE_POOL_MB (default 64; 0: none) from when they are made, mirrored or not:
// pools are imported, and mirrored, at the replay's first binding of them, two presents behind the game, and the
// captures taken before that left them out. MK World moves its per-frame data to a new pool every so many frames:
// the new pool's mirror was filled and synced from the game's memory as it was at the replay, its next frames' data
// in it already, until a mirror made the slow way published the list again (seconds): shadows dark below a line,
// the picture cyan below one, in bursts every 30 or 60 frames. The bigger pools hold textures, whose storage the
// capture takes anyway.)
// (RX_VK_CAPTURE_EAGER=<n>: only for their first n captures. Off: pools made at the load are bound by the menus much
// later, and they flashed (30 in the main menu); and the pages the captures copy are almost all the mirrored pools'
// anyway)
std::vector<std::pair<uint64_t, uint64_t>> pool_cap_ranges;   // (merged; cap_ranges_mu)
uint32_t pool_cap_gen = 0;
uint64_t captures_made = 0, pool_cap_built = 0;              // (cap_ranges_mu)
// (hookshot) how many captures have been begun (capture_make, before it collects the written pages), for any thread to
// read: what a thread wrote before it read N is in capture N+1, or in an older one (memory_captures_begun)
std::atomic<uint64_t> captures_begun{0};
// (the pools' starts are 64 KB aligned, and std::hash<uint64_t> is the identity in libc++ and libstdc++. With the list full,
// seen.reserve(8192) asks for a power of two, which libc++ keeps and masks: every start fell in bucket 0, and each find and
// insert walked the whole chain. ~20 ms a refresh in the Market, on the replay thread and on the game's main thread waiting
// for cap_ranges_mu at its submits (the slow submits' "memory sync". tests/test_pool_start_hash.py). Their bits mixed first)
struct PoolStartHash {
    size_t operator()(uint64_t k) const {
        k *= 0x9E3779B97F4A7C15ull;
        return (size_t)(k ^ (k >> 32));
    }
};
std::unordered_map<uint64_t, uint64_t, PoolStartHash> pool_first_seen;     // (a pool's start -> the capture it was first seen at)
uint64_t capture_pool_max() {
    static const uint64_t v = (getenv("RX_VK_CAPTURE_POOL_MB") ? (uint64_t)atoll(getenv("RX_VK_CAPTURE_POOL_MB")) : 64) << 20;
    return v;
}
uint64_t eager_captures() {
    static const uint64_t v = getenv("RX_VK_CAPTURE_EAGER") ? (uint64_t)atoll(getenv("RX_VK_CAPTURE_EAGER")) : 0;
    return v;
}
// ---- (Hookshot, RX_VK_CAP_ONLY_GPU; docs/native-engine/direct-emission.md section 10) ----
// The mirrored pools of cap_ranges are the back end's imports: 16 MB chunks around what was bound (import_chunk), not the
// game's pools. Where the game makes small pools inside its heap (OoT: 8 KB pools of flags 0x24 scattered through the heap at
// 0x1050000000-0x1060000000), a chunk spans heap memory no pool covers, which the game writes every frame and the capture
// copied (85-95% of its pages, section 9). A GPU address below 1 TB is valid only inside a memory pool (NVN), so those pages
// cannot be read by the GPU. 1: the import ranges are clipped to the game's pools before the capture scans them. 2 (a check):
// captured as before, and the pages 1 would leave out counted. With 1 or 2 every replay read and buffer binding that reaches
// into the left-out ranges is counted and the first ones logged (0 expected). 1 by default (2026-11-17): +13% fps on an RTX 3050 near
// the castle, no change on an RTX 4060 Ti in Kokiri (39.2 and 39.0 fps, on and off alike); 0 reads into the left-out ranges, the pictures right. RX_VK_CAP_ONLY_GPU=0 captures as before.
int cap_only_gpu() {
    static const int v = [] {
        const char* e = getenv("RX_VK_CAP_ONLY_GPU");
        const int m = e ? std::clamp(atoi(e), 0, 2) : 1;
        if (m) logf("vk: the capture's imported chunks %s the game's pools (RX_VK_CAP_ONLY_GPU=%d)",
                    m == 1 ? "clipped to" : "checked against", m);
        return m;
    }();
    return v;
}
std::vector<std::pair<uint64_t, uint64_t>> all_pool_ranges;   // (every game pool below 1 TB, merged; cap_ranges_mu)
std::mutex cap_excl_mu;
std::vector<std::pair<uint64_t, uint64_t>> cap_excl;          // (the import ranges outside every pool and texture storage)
std::atomic<int64_t> st_excl_pages{0}, st_excl_reads{0}, st_cap_pages{0}, st_cap_presents{0}, st_cap_takes{0};
std::vector<std::pair<uint64_t, uint64_t>> ranges_merged(std::vector<std::pair<uint64_t, uint64_t>> v) {
    std::sort(v.begin(), v.end());
    std::vector<std::pair<uint64_t, uint64_t>> m;
    for (const auto& r : v) {
        if (!m.empty() && r.first <= m.back().second) m.back().second = std::max(m.back().second, r.second);
        else m.push_back(r);
    }
    return m;
}
/// a and b sorted and merged: a's parts inside b (in = true) or outside it
std::vector<std::pair<uint64_t, uint64_t>> ranges_clip(const std::vector<std::pair<uint64_t, uint64_t>>& a,
                                                       const std::vector<std::pair<uint64_t, uint64_t>>& b, bool in) {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    size_t j = 0;
    for (const auto& r : a) {
        uint64_t at = r.first;
        while (j < b.size() && b[j].second <= at) j++;
        for (size_t k = j; k < b.size() && b[k].first < r.second; k++) {
            const uint64_t lo = std::max(at, b[k].first), hi = std::min(r.second, b[k].second);
            if (!in && lo > at) out.push_back({at, lo});
            if (in && lo < hi) out.push_back({lo, hi});
            at = std::max(at, hi);
        }
        if (!in && at < r.second) out.push_back({at, r.second});
    }
    return out;
}
void refresh_pool_ranges_locked() {   // (cap_ranges_mu held)
    if (!capture_pool_max() && !cap_only_gpu()) return;
    const uint32_t g = host->maps_generation();
    if (g == pool_cap_gen && captures_made - pool_cap_built < 64) return;   // (and every 64: the old ones out)
    pool_cap_gen = g, pool_cap_built = captures_made;
    static std::vector<std::array<uint64_t, 2>> list(8192);
    const int n = host->gpu_pools(reinterpret_cast<uint64_t(*)[2]>(list.data()), (int)list.size());
    std::vector<std::pair<uint64_t, uint64_t>> v, every;
    std::unordered_map<uint64_t, uint64_t, PoolStartHash> seen;
    seen.reserve((size_t)std::max(n, 0));
    for (int i = 0; i < n; i++) {
        if (list[i][1] && list[i][0] < (1ull << 40))
            every.push_back({list[i][0] & ~4095ull, (list[i][0] + list[i][1] + 4095) & ~4095ull});
        if (!list[i][1] || list[i][1] > capture_pool_max() || list[i][0] >= (1ull << 40)) continue;
        auto it = pool_first_seen.find(list[i][0]);
        const uint64_t first = it != pool_first_seen.end() ? it->second : captures_made;
        seen[list[i][0]] = first;
        if (!eager_captures() || captures_made - first < eager_captures())
            v.push_back({list[i][0] & ~4095ull, (list[i][0] + list[i][1] + 4095) & ~4095ull});
    }
    pool_first_seen.swap(seen);
    {   // (RX_VK_CAP_ONLY_GPU: the host's live pools, which hos_gpu_pools' mapping list can lose: see cap_pool_born)
        using LivePools = int (*)(uint64_t (*)[2], int);
        static const LivePools live = (LivePools)(void*)GetProcAddress(GetModuleHandleA(nullptr), "hos_gpu_live_pools");
        if (live && cap_only_gpu()) {
            static std::vector<std::array<uint64_t, 2>> lp(16384);
            const int k = live(reinterpret_cast<uint64_t(*)[2]>(lp.data()), (int)lp.size());
            for (int i = 0; i < k; i++)
                if (lp[i][1] > lp[i][0] && lp[i][0] < (1ull << 40)) every.push_back({lp[i][0] & ~4095ull, (lp[i][1] + 4095) & ~4095ull});
        }
    }
    all_pool_ranges = ranges_merged(std::move(every));
    cap_ranges_ver++;
    std::sort(v.begin(), v.end());
    std::vector<std::pair<uint64_t, uint64_t>> m;
    for (const auto& r : v) {
        if (!m.empty() && r.first <= m.back().second) m.back().second = std::max(m.back().second, r.second);
        else m.push_back(r);
    }
    pool_cap_ranges.swap(m);
}
bool capture_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_CAPTURE");
        const bool v = e ? atoi(e) != 0 : true;
        logf("vk: submit-time capture of written pages %s", v ? "on" : "off");
        return v;
    }();
    return on;
}
std::atomic<bool> captures_seen{false};
std::atomic<uint64_t> cap_view_gen{1};   // (bumped when a capture is published or a page is stamped: cached reads go stale)
void cap_view_bump() { cap_view_gen.fetch_add(1, std::memory_order_release); }
double st_tk_all, st_tk_stamp, st_tk_copy, st_tk_late, st_tk_lock;   // (memory_take_capture's parts, ms)
double st_tk_stamps, st_tk_shadow, st_tk_entries;   // (inside "retires and stamps": the page stamps, the texture shadow, the table entries)
int64_t st_tk_shadow_fast;   // (takes whose shadow block took RX_VK_SHADOW_FAST's path)
std::atomic<int64_t> st_tr_late_ns{0}, st_tr_late_n{0};   // (the same, never reset, for the frame trace)
int64_t st_tk_late_n;   // (the takes that waited for the GPU for their late pages)   // (a capture was ever made: the replay thread's; without it nothing owns pages)

// ---- texture storage in the capture ----
// The textures' storage the replay uploads from joins the capture, and what the capture took of it is kept (the
// shadow) for the uploads to read instead of guest memory: the storage as it was at its submit. MKW writes its terrain's
// height tiles (68 x 68 RGBA16, at new addresses every frame) for a frame and then rewrites them right after the
// submit; uploaded when the replay recorded the submit, from guest memory as it was by then, a frame's tiles could take
// another's heights. Measured at a snowy spot it is rare: of 107,000 pages the uploads read from the copies, 13 had
// been rewritten since their submit. In a Grand Prix it is what flashed: ~100 one-frame flashes a minute (giant
// stretched shapes over most of the picture, black or fog-coloured bands, a 30-frame rhythm), 9-18 rewritten pages of
// 800,000, and none with it on. On by default now that the guarded copy holds (a call, not inlined). (A page the game
// wrote and then freed is still marked: the copy commits it again, zeros, on first touch (commit_on_touch), which is
// what the mirrors must see of it; clearing the marks at the free lost that, and a Grand Prix loaded to a black
// screen.) RX_VK_TEX_CAPTURE=0: off.
bool tex_capture_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_TEX_CAPTURE");
        const bool v = (e ? atoi(e) != 0 : true) && capture_on();
        logf("vk: texture storage in the submit-time capture %s", v ? "on" : "off");
        return v;
    }();
    return on;
}
std::vector<std::pair<uint64_t, uint64_t>> tex_cap_ranges;   // (texture storage the capture takes, merged; cap_ranges_mu)
std::mutex shadow_mu;   // (after pools_mu, before cap_ranges_mu)
struct ShadowPage {
    std::unique_ptr<uint8_t[]> b;
    uint64_t seq = 0;   // (the capture that took it)
};
std::unordered_map<uint32_t, ShadowPage> shadow;   // guest page -> its 4 KB as the last capture taken had it
std::unordered_map<uint32_t, uint32_t> shadow_regions;   // (2 MB region -> its pages in the shadow)
std::atomic<size_t> shadow_n{0};
uint64_t shadow_seq = 0;   // (captures taken)
// (issue 66) the capture taken for the submission being recorded: its copies' seq (0: it took none). Captures are taken on the
// replay thread just before their submission's recording (vk_replay.cpp submit_memory_work), so while one records this is its own
std::atomic<uint64_t> shadow_seq_own{0};
// (256 MB: past it the pages no capture took in the last 64 go, the oldest first: those the game rewrites every frame
// stay; if none can go, a page is read from guest memory as before)
constexpr size_t kShadowMax = 65536;
std::map<uint64_t, std::pair<uint64_t, uint64_t>> tex_spans;   // lo -> (hi, the present a texture was last uploaded from it)
std::vector<std::pair<uint64_t, uint64_t>> tex_merged;          // (tex_spans merged; shadow_mu)
bool tex_spans_changed = false;
uint64_t tex_merged_bytes = 0;

void shadow_erase(uint32_t page) {   // (shadow_mu held)
    if (!shadow.erase(page)) return;
    auto r = shadow_regions.find(page >> 9);
    if (r != shadow_regions.end() && --r->second == 0) shadow_regions.erase(r);
    shadow_n--;
}
void shadow_drop_range(uint64_t lo, uint64_t hi) {   // (shadow_mu held)
    if (hi <= lo) return;
    for (uint64_t r = lo >> 21; r <= (hi - 1) >> 21; r++) {
        if (!shadow_regions.count((uint32_t)r)) continue;
        const uint64_t p0 = std::max(lo >> 12, r << 9), p1 = std::min((hi + 4095) >> 12, (r + 1) << 9);
        for (uint64_t p = p0; p < p1; p++) shadow_erase((uint32_t)p);
    }
}
// ---- (= E2) RX_VK_SHADOW_FAST=1 (off by default): take_capture's texture-shadow block with less work a page ----
// The block (shadow_mu and pools_mu held, on the replay thread or the held flush) did for every page of a capture two binary
// searches of tex_merged (one in the erase pass, one in the insert pass), a hash erase, then for every copied texture page a hash
// insert, a fresh `new uint8_t[4096]` and a 4 KB copy: each page the capture just copied was first erased and then made again.
// Here (shadow_take_fast):
// - whether a page is texture storage comes from one merge walk of the sorted page list against the sorted, disjoint tex_merged
// (the same answer as in_tex_merged); a page list not in ascending order (the  pre-copy's merged list) takes the old code;
// - a copied texture page keeps its entry and its 4 KB block, which the copy overwrites; only texture pages written but not copied
// (a failed copy, a deferred page) are erased, as before;
// - a page new to the shadow takes a block left by an erased one when there is one (shadow_slab, up to 1024 blocks), else new[].
// The shadow after the block is the same: the old code erased every texture page of c->pages and then inserted each copied one,
// so a copied page ends with an entry holding the capture's bytes and seq = shadow_seq, a written but not copied one with none,
// every other entry as it was; shadow_n and shadow_regions (each region's count of entries; a region at 0 leaves the map) follow
// from that membership alone, whichever way the counts moved on the way. The sweeps decide per entry by seq (no early stop), so
// the map's iteration order does not change what they keep. The only path where the order of the old code mattered is the cap
// (kShadowMax reached inside the insert loop, a sweep or a stop part way): the fast path runs only when it cannot be reached,
// shadow_n + (copied texture pages) <= kShadowMax before anything is erased (the old count after its erase pass is at most
// shadow_n, and each insert adds one), else the old code runs as it was. The readers (memory_as_submitted, memory_has_shadow)
// take shadow_mu, held throughout, so none sees a state in between.
bool shadow_fast_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_SHADOW_FAST");
        const bool v = e && atoi(e) != 0;
        logf("vk: a capture's texture shadow: %s (RX_VK_SHADOW_FAST)", v ? "one walk, blocks kept" : "searched, made again, as before");
        return v;
    }();
    return on;
}
std::vector<std::unique_ptr<uint8_t[]>> shadow_slab;   // (4 KB blocks of pages that left the shadow, for new ones; shadow_mu)
constexpr size_t kShadowSlab = 1024;
void shadow_erase_keep(uint32_t page) {   // (shadow_mu held: shadow_erase, the block kept for the next page)
    auto it = shadow.find(page);
    if (it == shadow.end()) return;
    if (it->second.b && shadow_slab.size() < kShadowSlab) shadow_slab.push_back(std::move(it->second.b));
    shadow.erase(it);
    auto r = shadow_regions.find(page >> 9);
    if (r != shadow_regions.end() && --r->second == 0) shadow_regions.erase(r);
    shadow_n--;
}
/// The block for a capture: pages (c->pages), the copied ones and their slots (c->copied, c->slots: slot = index in pages), the
/// capture's mapped copy. False: not taken (the old code runs). shadow_mu held, shadow_seq already advanced
bool shadow_take_fast(const uint32_t* pages, size_t n, const uint32_t* copied, const uint32_t* slots, size_t m, const uint8_t* map) {
    if (!map || !std::is_sorted(pages, pages + n)) return false;
    static thread_local std::vector<uint8_t> flag;   // (bit 0: texture storage, bit 1: copied)
    flag.assign(n, 0);
    size_t r = 0;
    for (size_t i = 0; i < n; i++) {
        const uint64_t g = (uint64_t)pages[i] << 12;
        while (r < tex_merged.size() && tex_merged[r].second <= g) r++;
        if (r < tex_merged.size() && g >= tex_merged[r].first) flag[i] = 1;
    }
    size_t copied_tex = 0;
    for (size_t j = 0; j < m; j++) {
        if (slots[j] >= n || pages[slots[j]] != copied[j]) return false;   // (not the layout described: the old code)
        if (flag[slots[j]] & 1) flag[slots[j]] |= 2, copied_tex++;
    }
    if (shadow_n.load(std::memory_order_relaxed) + copied_tex > kShadowMax) return false;   // (the cap could be reached: the old code)
    for (size_t i = 0; i < n; i++)
        if (flag[i] == 1) shadow_erase_keep(pages[i]);   // (written, not copied: its old copy is stale)
    for (size_t j = 0; j < m; j++) {
        if (flag[slots[j]] != 3) continue;
        const uint32_t p = copied[j];
        auto [it, made] = shadow.try_emplace(p);
        ShadowPage& slot = it->second;
        if (!slot.b) {
            if (!shadow_slab.empty()) slot.b = std::move(shadow_slab.back()), shadow_slab.pop_back();
            else slot.b.reset(new uint8_t[4096]);
        }
        if (made) {
            shadow_regions[p >> 9]++;
            shadow_n++;
        }
        slot.seq = shadow_seq;
        std::memcpy(slot.b.get(), map + (uint64_t)slots[j] * 4096, 4096);
    }
    return true;
}
/// (the spans unused for 1800 presents dropped with their shadow pages; the rest published for the capture)
void publish_tex_spans(uint64_t frame) {
    std::lock_guard lk(shadow_mu);
    // (a span ages by the present: looked over once a present, not at each of its ~20 submissions, which walked
    // the ~300 spans every time, 1 ms a frame on a Steam Deck)
    static uint64_t swept = ~0ull;
    if (frame != swept) {
        swept = frame;
        for (auto it = tex_spans.begin(); it != tex_spans.end();) {
            if (frame > it->second.second + 1800) {
                shadow_drop_range(it->first, it->second.first);
                it = tex_spans.erase(it);
                tex_spans_changed = true;
            } else {
                ++it;
            }
        }
    }
    if (!tex_spans_changed) return;
    tex_spans_changed = false;
    std::vector<std::pair<uint64_t, uint64_t>> v;
    for (const auto& [lo, e] : tex_spans) {
        if (!v.empty() && lo <= v.back().second) v.back().second = std::max(v.back().second, e.first);
        else v.push_back({lo, e.first});
    }
    tex_merged = v;
    tex_merged_bytes = 0;
    for (const auto& r : v) tex_merged_bytes += r.second - r.first;
    std::lock_guard lk2(cap_ranges_mu);
    tex_cap_ranges.swap(v);
    cap_ranges_ver++;
}
/// the captures' ranges: the mirrored imports (clipped to the pools with RX_VK_CAP_ONLY_GPU=1), the texture storage, the small
/// pools; their union, sorted and merged (cap_ranges_mu held). The left-out ranges published for the check
std::vector<std::pair<uint64_t, uint64_t>> capture_union_locked() {
    std::vector<std::pair<uint64_t, uint64_t>> imp = cap_ranges;
    if (cap_only_gpu()) {
        imp = ranges_merged(imp);
        // (2026-11-14, play9) the GPU-visible memory: the host's pools and every pool the back end has a record of since the game
        // made it (until another pool is made over it). The host's counted list lost live pools: one bound for 46 s was in no
        // pool of it (a pool object finalized twice untracked a newer pool's entry: nvn_native.c), and a pool finalized after a
        // submission's capture may still be bound by its replay. A record stays: never left out while it can be bound
        std::vector<std::pair<uint64_t, uint64_t>> gpu = all_pool_ranges;
        nvn_pool_record_ranges(gpu);
        gpu = ranges_merged(std::move(gpu));
        std::vector<std::pair<uint64_t, uint64_t>> keep = gpu;
        keep.insert(keep.end(), tex_cap_ranges.begin(), tex_cap_ranges.end());
        keep = ranges_merged(keep);
        std::vector<std::pair<uint64_t, uint64_t>> excl = ranges_clip(imp, keep, false);
        {
            std::lock_guard lk(cap_excl_mu);
            cap_excl.swap(excl);
        }
        if (cap_only_gpu() == 1) imp = ranges_clip(imp, gpu, true);
    }
    imp.insert(imp.end(), tex_cap_ranges.begin(), tex_cap_ranges.end());
    imp.insert(imp.end(), pool_cap_ranges.begin(), pool_cap_ranges.end());
    return ranges_merged(std::move(imp));
}
/// whether [lo, hi) reaches into one of the ranges of v (sorted, merged)
bool ranges_hit(const std::vector<std::pair<uint64_t, uint64_t>>& v, uint64_t lo, uint64_t hi) {
    auto it = std::upper_bound(v.begin(), v.end(), std::make_pair(lo, ~uint64_t(0)));
    if (it != v.begin() && lo < std::prev(it)->second) return true;
    return it != v.end() && it->first < hi;
}
/// (RX_VK_CAP_ONLY_GPU) whether [lo, hi) of guest memory reaches into the left-out ranges
bool cap_excl_hit(uint64_t lo, uint64_t hi) {
    std::lock_guard lk(cap_excl_mu);
    return ranges_hit(cap_excl, lo, hi);
}
/// (2026-11-14) a pool just made, [lo, hi): one of the pools at once, and out of the left-out ranges at once, so no binding
/// of it can come before (the play session's reads into left-out ranges came after zone changes: pools made since the last
/// pool-list refresh, and pools the host's mapping list had lost (hos_gpu_pools: a pool made over an older one's memory
/// lost its entry at the older one's finalize), were left out while bound)
void cap_pool_born(std::vector<std::pair<uint64_t, uint64_t>>& pools, std::vector<std::pair<uint64_t, uint64_t>>& excl,
                   uint64_t lo, uint64_t hi) {
    pools.push_back({lo & ~4095ull, (hi + 4095) & ~4095ull});
    pools = ranges_merged(std::move(pools));
    excl = ranges_clip(excl, {{lo & ~4095ull, (hi + 4095) & ~4095ull}}, false);
}
std::mutex born_mu;
std::vector<std::pair<uint64_t, uint64_t>> born_stamp;   // (pools made since the replay's last collection: stamped there; born_mu)
/// (RX_VK_CAP_ONLY_GPU) nvnMemoryPoolInitialize, on the game's thread (memory_pool_flags): the pool joins the pools and leaves
/// the left-out ranges before the call returns. With 1 its pages, whose writes before the pool existed were dropped as heap
/// memory's (memory_collect_written), are stamped at the replay's next collection: a mirror over them copies them again
void cap_pool_born_now(uint64_t gpu, uint64_t size) {
    if (!cap_only_gpu() || !size || gpu >= (1ull << 40)) return;
    {
        std::lock_guard lk(cap_ranges_mu);
        std::lock_guard ek(cap_excl_mu);
        cap_pool_born(all_pool_ranges, cap_excl, gpu, gpu + size);
        cap_ranges_ver++;
    }
    if (cap_only_gpu() == 1) {
        std::lock_guard bk(born_mu);
        born_stamp.push_back({gpu, gpu + size});
    }
}
void cap_born_take(std::vector<std::pair<uint64_t, uint64_t>>& out) {
    std::lock_guard bk(born_mu);
    out.swap(born_stamp);
    born_stamp.clear();
}
/// (RX_VK_CAP_ONLY_GPU) a replay read or a buffer binding of [gpu, gpu + n): counted, and the first 20 logged, when it reaches
/// into the left-out ranges
void cap_excl_note(uint64_t gpu, uint64_t n, const char* who, const void* caller = nullptr) {
    if (!n) return;
    const uint64_t g = gpu < (1ull << 40) ? gpu : host->gpu_guest(gpu);
    if (!g || !cap_excl_hit(g, g + n)) return;
    const int64_t k = st_excl_reads.fetch_add(1, std::memory_order_relaxed);
    // (2026-11-14) the first 40 in full, and then one in 1000: the left-out range met, the game's pool there now (the host's live
    // list, the back end's record: flags, when it was made), and whether the binding runs past its pool's end
    if (k >= 40 && k % 1000) return;
    uint64_t elo = 0, ehi = 0;
    {
        std::lock_guard lk(cap_excl_mu);
        auto it = std::upper_bound(cap_excl.begin(), cap_excl.end(), std::make_pair(g, ~uint64_t(0)));
        if (it != cap_excl.begin() && g < std::prev(it)->second) --it;
        if (it != cap_excl.end()) elo = it->first, ehi = it->second;
    }
    uint64_t llo = 0, lhi = 0;
    {
        using LivePools = int (*)(uint64_t (*)[2], int);
        static const LivePools live = (LivePools)(void*)GetProcAddress(GetModuleHandleA(nullptr), "hos_gpu_live_pools");
        static thread_local std::vector<std::array<uint64_t, 2>> lp(16384);
        const int m = live ? live(reinterpret_cast<uint64_t(*)[2]>(lp.data()), (int)lp.size()) : 0;
        for (int i = 0; i < m; i++)
            if (g >= lp[i][0] && g < lp[i][1]) llo = lp[i][0], lhi = lp[i][1];
    }
    uint64_t rlo = 0, rhi = 0, rmade = 0;
    uint32_t rflags = 0;
    nvn_pool_record_at(g, &rlo, &rhi, &rflags, &rmade);
    static const uintptr_t dll = (uintptr_t)GetModuleHandleA("hookshot_vk.dll");
    logf("vk: RX_VK_CAP_ONLY_GPU: %s of %#llx +%#llx (guest %#llx, from v+%#llx) reaches memory the capture leaves out (#%lld): "
         "left-out range %#llx..%#llx; live pool there %#llx..%#llx%s; the back end's pool record %#llx..%#llx flags %#x made %.3f s ago",
         who, (unsigned long long)gpu, (unsigned long long)n, (unsigned long long)g,
         (unsigned long long)(caller && dll ? (uintptr_t)caller - dll : 0), (long long)k + 1, (unsigned long long)elo,
         (unsigned long long)ehi, (unsigned long long)llo, (unsigned long long)lhi,
         llo && g + n > lhi ? " (the binding runs past its end)" : llo ? "" : " (none)", (unsigned long long)rlo,
         (unsigned long long)rhi, rflags, rmade ? (host->now_ns() - (int64_t)rmade) / 1e9 : -1.0);
}
} // namespace

void memory_note_texture_upload(uint64_t guest, uint64_t size) {
    if (cap_only_gpu() && guest < (1ull << 40)) cap_excl_note(guest, size, "a texture upload");
    if (!size || guest >= (1ull << 40) || size > (1ull << 40) || !tex_capture_on()) return;   // (hookshot: no wrap below)
    uint64_t lo, hi;
    {   // (the pool it is in, when that is at most 256 MB; else the 2 MB regions it spans)
        std::lock_guard lk(pools_mu);
        auto it = std::upper_bound(guest_pools.begin(), guest_pools.end(), guest,
                                   [](uint64_t g, const GuestPool& p) { return g < p.lo; });
        if (it != guest_pools.begin() && guest >= std::prev(it)->lo && guest + size <= std::prev(it)->hi &&
            std::prev(it)->hi - std::prev(it)->lo <= (256ull << 20)) {
            lo = std::prev(it)->lo, hi = std::prev(it)->hi;
        } else {
            lo = guest & ~((2ull << 20) - 1), hi = (guest + size + (2ull << 20) - 1) & ~((2ull << 20) - 1);
        }
    }
    const uint64_t frame = texture_frame_no();
    std::lock_guard lk(shadow_mu);
    auto [it, fresh] = tex_spans.try_emplace(lo, hi, frame);
    if (!fresh) {
        it->second.second = frame;
        if (it->second.first >= hi) return;
        it->second.first = hi;
    }
    tex_spans_changed = true;
}

/// A replay read of one page can reuse what the last read of that page decided, while the page's write generation and
/// the capture table are the ones that decision was made from.
bool submitted_page_reuse(uint64_t page, uint32_t page_gen, uint64_t view, uint64_t cached_page, uint32_t cached_gen,
                          uint64_t cached_view) {
    return cached_view != 0 && page == cached_page && page_gen == cached_gen && view == cached_view;
}
struct SubmittedPage {   // (a page no capture holds, as of pgen and view: the read from guest memory stands)
    uint64_t page = ~0ull;
    uint32_t pgen = 0;
    uint64_t view = 0;
};
// (hookshot) 32 of them a thread, by page: a handle program reads a uniform word on one page and a storage buffer word on
// another, and with one page kept nearly every read missed and took pools_mu (program_for's answer handles). A page a
// capture holds is not kept: its words are copied from the capture under pools_mu at each read, not the whole 4 KB page
// into a copy of the thread's own (that copy was a third of the replay's thread in Kokiri Forest on NVIDIA Linux)
constexpr int kSubmittedPages = 32;
thread_local SubmittedPage submitted_pages[kSubmittedPages];
inline SubmittedPage& submitted_page_for(uint64_t page) {   // (a page and its neighbours, and pages 32 apart, in different slots)
    return submitted_pages[(page + (page >> 5) * 13) & (kSubmittedPages - 1)];
}
bool late_fix_slot_pending(uint32_t gen, uint32_t slot, uint64_t* value);   // (below)
bool guarded_copy(void* dst, const void* src, size_t n);   // (below)
std::atomic<int64_t> st_late_def_full{0}, st_late_reads{0}, st_late_syncs{0};   //

void late_fix_cpu_read(uint32_t page, VkBuffer buffer, uint8_t* slot);   // (pools_mu held: the late pages' fix, below)
bool late_defer_any();   // (pools_mu held: RX_VK_LATE_DEFER, below)
void late_defer_cpu_read(uint32_t page, uint32_t gen, uint32_t slot, VmaAllocation alloc, uint8_t* map);
namespace {
/// 0: unreadable, 1: already read, 2: look in the capture under pools_mu. For a validated
/// single physical page, live points to the fallback and the destination has not been touched.
int submitted_read_unlocked(uint64_t gpu, void* dst, uint64_t n, uint64_t& guest, const uint8_t*& live) {
    live = nullptr;
    if (!n || !capture_on() || !captures_seen.load(std::memory_order_relaxed)) return host->gpu_read(gpu, dst, n);
    guest = gpu < (1ull << 40) ? gpu : host->gpu_guest(gpu);
    const uint64_t limit = std::min<uint64_t>(host->guest_size(), kPages << kPageShift);
    if (!guest || guest >= limit || (gpu < (1ull << 40) && n > limit - guest))
        return host->gpu_read(gpu, dst, n);
    const uint64_t page0 = guest >> kPageShift;
    const bool one_page = n <= 4096 - (guest & 4095);
    const uint64_t view = cap_view_gen.load(std::memory_order_acquire);
    const SubmittedPage& submitted_page = submitted_page_for(page0);
    if (one_page && submitted_page_reuse(page0, page_gen(page0), view, submitted_page.page, submitted_page.pgen,
                                               submitted_page.view))
        return host->gpu_read(gpu, dst, n);
    // The host maps these GPU addresses directly into its fixed guest window. Validate the
    // address as gpu_read does. Sparse virtual mappings and partial reads keep its normal path.
    if (one_page && gpu < (1ull << 40) && host->gpu_guest(gpu) == guest) {
        live = host->guest_base() + guest;
        return 2;
    }
    return host->gpu_read(gpu, dst, n) ? 2 : 0;
}
/// (pools_mu held) the pages of that read a capture holds, copied over what guest memory gave: how many
int64_t submitted_read_locked(uint64_t guest, void* dst, uint64_t n) {
    const uint64_t page0 = guest >> kPageShift, page1 = (guest + n - 1) >> kPageShift;
    int64_t from = 0;
    for (uint64_t page = guest >> kPageShift; page < (guest + n + 4095) >> kPageShift; page++) {
        const CapEntry e = cap_entry(page);
        if (!e.gen || e.gen != page_gen(page)) continue;
        auto b = cap_bufs.find(e.gen);
        if (b == cap_bufs.end() || !b->second.map) continue;
        late_fix_cpu_read((uint32_t)page, b->second.buffer, (uint8_t*)b->second.map + (uint64_t)e.slot * 4096);
        if (late_defer_any()) late_defer_cpu_read((uint32_t)page, e.gen, e.slot, b->second.alloc, (uint8_t*)b->second.map);
        const uint64_t a = std::max(page << kPageShift, guest), z = std::min((page + 1) << kPageShift, guest + n);
        std::memcpy((uint8_t*)dst + (a - guest), (const uint8_t*)b->second.map + (uint64_t)e.slot * 4096 + (a - (page << kPageShift)), z - a);
        from++;
    }
    if (page0 == page1 && from == 0) {   // (in no capture: guest memory stands for its next words too)
        SubmittedPage& submitted_page = submitted_page_for(page0);
        submitted_page.page = page0;
        submitted_page.pgen = page_gen(page0);
        submitted_page.view = cap_view_gen.load(std::memory_order_relaxed);
    }
    return from;
}
// Virtual neighbours can map to unrelated physical pages. The full live read has
// already succeeded. Only overlay segments whose physical page has a current capture.
int64_t mapped_submitted_read_locked(uint64_t gpu, uint64_t guest, void* dst, uint64_t n) {
    if (gpu < (1ull << 40) || n <= 4096 - (gpu & 4095))
        return submitted_read_locked(guest, dst, n);
    const uint64_t limit = std::min<uint64_t>(host->guest_size(), kPages << kPageShift);
    int64_t from = 0;
    for (uint64_t at = 0; at < n;) {
        const uint64_t bytes = std::min<uint64_t>(n - at, 4096 - ((gpu + at) & 4095));
        const uint64_t physical = at ? host->gpu_guest(gpu + at) : guest;
        if (physical && physical < limit && bytes <= limit - physical)
            from += submitted_read_locked(physical, static_cast<uint8_t*>(dst) + at, bytes);
        at += bytes;
    }
    return from;
}
void submitted_read_counted(int64_t looked_up, int64_t from) {
    static std::atomic<int64_t> reads{0}, pages_from{0};
    pages_from += from;
    const int64_t before = reads.fetch_add(looked_up);
    if (before / 20000 != (before + looked_up) / 20000)
        logf("vk: replay reads of guest memory as submitted: %lld (%lld pages from the captures)", (long long)(before + looked_up),
             (long long)pages_from.load());
}
}  // namespace
bool memory_read_as_submitted(uint64_t gpu, void* dst, uint64_t n) {
    uint64_t guest = 0;
    const uint8_t* live;
    const int r = submitted_read_unlocked(gpu, dst, n, guest, live);
    if (r != 2) return r != 0;
    int64_t from = 0;
    {
        std::lock_guard lk(pools_mu);
        from = mapped_submitted_read_locked(gpu, guest, dst, n);
    }
    if (!from && live) std::memcpy(dst, live, n);
    submitted_read_counted(1, from);
    return true;
}
/// Several replay reads at once (a draw's uniform ranges, uniform_range): each as memory_read_as_submitted, with pools_mu
/// taken once for all of them that need the captures, not once for each. The lock and its release were two thirds of a
/// uniform snapshot's time on an Arc B580 (about 10,000 a frame in the Market, 4 a draw)
void memory_read_as_submitted_many(const SubmittedRead* reads, size_t count) {
    constexpr size_t kMost = 64;
    uint64_t guests[kMost];
    const uint8_t* live[kMost];
    size_t late[kMost], nlate = 0;
    for (size_t i = 0; i < count; i++) {
        if (i >= kMost) {
            memory_read_as_submitted(reads[i].gpu, reads[i].dst, reads[i].n);
            continue;
        }
        if (submitted_read_unlocked(reads[i].gpu, reads[i].dst, reads[i].n, guests[i], live[i]) == 2) late[nlate++] = i;
    }
    if (!nlate) return;
    size_t missing = 0;
    int64_t from = 0;
    {
        std::lock_guard lk(pools_mu);
        for (size_t k = 0; k < nlate; k++) {
            const size_t i = late[k];
            const auto& read = reads[i];
            const int64_t captured = mapped_submitted_read_locked(read.gpu, guests[i], read.dst, read.n);
            if (!captured && live[i]) late[missing++] = i;
            from += captured;
        }
    }
    for (size_t k = 0; k < missing; k++) {
        const size_t i = late[k];
        std::memcpy(reads[i].dst, live[i], reads[i].n);
    }
    submitted_read_counted((int64_t)nlate, from);
}

uint64_t memory_capture_view() { return cap_view_gen.load(std::memory_order_acquire); }
uint32_t memory_page_stamp(uint64_t gpu) {
    const uint64_t guest = gpu < (1ull << 40) ? gpu : host->gpu_guest(gpu);
    const uint64_t page = guest >> kPageShift;
    if (!guest || page >= kPages || guest >= host->guest_size()) return 0;
    return page_gen(page);
}

bool memory_has_shadow(uint64_t guest, uint64_t n) {
    if (!n || !shadow_n.load(std::memory_order_relaxed)) return false;
    std::lock_guard lk(shadow_mu);
    for (uint64_t r = guest >> 21; r <= (guest + n - 1) >> 21; r++) {
        if (!shadow_regions.count((uint32_t)r)) continue;
        const uint64_t p0 = std::max(guest >> 12, r << 9), p1 = std::min((guest + n + 4095) >> 12, (r + 1) << 9);
        for (uint64_t p = p0; p < p1; p++)
            if (shadow.count((uint32_t)p)) return true;
    }
    return false;
}

/// (RX_VK_TEX_CAPTURE_OFF_FILE=<path> (A/B): while that file exists the uploads read guest memory as before)
bool overlay_off() {
    static const char* path = getenv("RX_VK_TEX_CAPTURE_OFF_FILE");
    if (!path) return false;
    static std::atomic<int64_t> next{0};
    static std::atomic<bool> off{false};
    const int64_t now = GetTickCount64();
    if (now >= next.load(std::memory_order_relaxed)) {
        next.store(now + 200, std::memory_order_relaxed);
        const bool v = GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
        if (v != off.exchange(v)) logf("vk: texture storage as submitted: %s (A/B file)", v ? "OFF" : "on");
    }
    return off.load(std::memory_order_relaxed);
}
std::atomic<int64_t> st_overlaid{0}, st_differed{0};   // (pages read from the shadow; of them, ones guest memory had changed)
std::atomic<int64_t> st_first_live{0};   // (pages a first upload read from guest memory: their copy was an older capture's)
std::atomic<int64_t> st_again_live{0};   // (the same for re-uploads, RX_VK_REUPLOAD_OWN)

// ---- (issue 66) a texture's first upload: only the copies its own submission's capture took ----
// OoT reuses a texture's storage for a new one (the Market's lightmaps, BC6H/BC7 atlases, and its ASTC materials, streamed in
// after arrival) and fills it after submitting the draws that sample it: the console's GPU, running that work later, reads the
// new bytes. The copies the capture kept of those pages were the old texture's, taken 27-267 captures (7-70 frames) earlier
// and never taken again, since the game had not written them before the submit; the first upload read them, and the next
// frame's upload of the pages written since put the right bytes in: one frame of lighting blown out to magenta and white,
// or a roof, a timber frame or a banner in another material's colours (RX 9070 XT, RADV). A first upload now takes a page's
// copy only when the submission being recorded took it (written before that submit and again since: what the copies are for,
// MK World's terrain tiles, which the game rewrites right after the submit that draws them); a copy from an earlier capture
// is the page as it was before the texture existed, and guest memory is read instead. A re-upload: reupload_own_on, below.
// RX_VK_FIRST_UPLOAD_OWN=0: as before
bool first_upload_own_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_FIRST_UPLOAD_OWN");
        const bool v = !e || !env_value_off(e);   // (the switches' rule: unset or empty on; 0, false, off or no off)
        logf("vk: a texture's first upload takes %s (RX_VK_FIRST_UPLOAD_OWN)",
             v ? "only its own submission's copies of its storage" : "every copy of its storage, as before");
        return v;
    }();
    return on;
}

// ---- (issue 66) a texture's re-upload: the same, only the copies its own submission's capture took ----
// With the first uploads fixed, the Market still flashed now and then: bursts of 10-15 one-frame flashes over ~10 s, 20-35 s
// after an arrival (the picture blown out to white and magenta, a bloom, walls and ground in lighter colours, or both blue
// banners white). OoT rewrites the scene's lightmaps (BC6H 1528x920x3 and 1120x1516x3, BC5 2240x3032) and ASTC materials in
// place in that time, every ~9 frames, writing a page more than once. A re-upload in such a frame took 350-7,000 pages from
// earlier captures' copies (none of them the recorded submission's) that differed from guest memory: a capture had taken
// the page between two of those writes, and the next frame's re-upload, from copies equal to guest memory, put the right
// bytes back. Of 28 flashes in two runs, 27 fell within two frames of such a re-upload; the other was a first upload. A
// re-upload now takes a page's copy only when the submission being recorded took it, and guest memory for a page whose copy
// is an earlier capture's, as a first upload does (a copy equal to guest memory gives the same bytes either way).
// RX_VK_REUPLOAD_OWN=0: every copy, as before
bool reupload_own_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_REUPLOAD_OWN");
        const bool v = !e || !env_value_off(e);   // (the switches' rule, as RX_VK_FIRST_UPLOAD_OWN)
        logf("vk: a texture's re-upload takes %s (RX_VK_REUPLOAD_OWN)",
             v ? "only its own submission's copies of its storage" : "every copy of its storage, as before");
        return v;
    }();
    return on;
}

const uint8_t* memory_as_submitted(uint64_t guest, const uint8_t* live, uint64_t n, std::vector<uint8_t>& scratch, bool first,
                                   bool again) {
    if (!n || !shadow_n.load(std::memory_order_relaxed) || overlay_off()) return live;
    const bool own_only = first ? first_upload_own_on() : again && reupload_own_on();
    const uint64_t own = shadow_seq_own.load(std::memory_order_relaxed);
    std::lock_guard lk(shadow_mu);
    bool any = false;
    for (uint64_t r = guest >> 21; r <= (guest + n - 1) >> 21; r++) {
        if (!shadow_regions.count((uint32_t)r)) continue;
        const uint64_t p0 = std::max(guest >> 12, r << 9), p1 = std::min((guest + n + 4095) >> 12, (r + 1) << 9);
        for (uint64_t p = p0; p < p1; p++) {
            auto it = shadow.find((uint32_t)p);
            if (it == shadow.end()) continue;
            if (own_only && (!own || it->second.seq != own)) {   // (an earlier capture's copy: guest memory stands)
                (first ? st_first_live : st_again_live).fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (!any) {
                any = true;
                if (scratch.data() != live) scratch.assign(live, live + n);
            }
            const uint64_t a = std::max(p << 12, guest), b = std::min((p + 1) << 12, guest + n);
            st_overlaid++;
            if (std::memcmp(scratch.data() + (a - guest), it->second.b.get() + (a - (p << 12)), b - a)) st_differed++;
            std::memcpy(scratch.data() + (a - guest), it->second.b.get() + (a - (p << 12)), b - a);
        }
    }
    static std::atomic<int64_t> served{0};
    if (any && served.fetch_add(1) % 2000 == 0)
        logf("vk: texture storage read as submitted (%lld times; %lld pages from the copies, %lld of them rewritten in "
             "guest memory since their submit; %zu pages kept, %zu spans, %llu MB; %lld pages of first uploads read from guest "
             "memory over an earlier capture's copy, %lld of re-uploads)", (long long)served.load(),
             (long long)st_overlaid.load(), (long long)st_differed.load(), shadow.size(), tex_spans.size(),
             (unsigned long long)(tex_merged_bytes >> 20), (long long)st_first_live.load(),
             (long long)st_again_live.load());
    return any ? scratch.data() : live;
}

void memory_drop_shadow(uint64_t guest, uint64_t size) {
    if (!size || !shadow_n.load(std::memory_order_relaxed)) return;
    std::lock_guard lk(shadow_mu);
    shadow_drop_range(guest, guest + size);
}

// ---- (EvilBoyce) GPU results by page: the submission that last wrote a page into guest memory ----
// For invalidate (vk_replay.cpp): a CPU read of GPU results waits until the submission that writes them there is done, as
// the console's coherent memory has them once that work is. Guest page number -> the recording's timeline value (write-backs
// from a mirror, and small bindings written in place).
namespace {
std::mutex wb_pages_mu;
// (each page's value: the timeline value of its last write-back, 0 for never, in chunks of kChunkPages made when a page of
// the chunk is first written back, as the page generations above. Write-backs are of imported pools only, which lie in the
// GPU's windows, so few chunks are ever made. Values are never lowered or dropped: the readers (memory_writeback_value's
// "ever written back" and its value against the GPU's progress, and the capture's wait, the same compare) read one the GPU
// is past as "written back, nothing to wait for", whatever it is. tests/test_wb_pages.py)
struct WbPages {
    std::vector<std::unique_ptr<uint64_t[]>> chunks = std::vector<std::unique_ptr<uint64_t[]>>(kPages / kChunkPages);
    bool any = false;
    void set(uint64_t page, uint64_t value) {
        if (page >= kPages) return;
        auto& c = chunks[page / kChunkPages];
        if (!c) {
            c.reset(new uint64_t[kChunkPages]);
            std::fill(c.get(), c.get() + kChunkPages, 0ull);
        }
        c[page % kChunkPages] = value;
        any = true;
    }
    uint64_t get(uint64_t page) const {
        if (page >= kPages) return 0;
        const auto& c = chunks[page / kChunkPages];
        return c ? c[page % kChunkPages] : 0;
    }
    /// the largest value of pages first..last (0: none), and whether any had one; a chunk never made is skipped whole
    uint64_t max_in(uint64_t first, uint64_t last, bool* ever) const {
        uint64_t v = 0;
        for (uint64_t p = first; p <= last && p < kPages;) {
            const uint64_t end = (p / kChunkPages + 1) * kChunkPages;   // (the first page past p's chunk)
            const auto& c = chunks[p / kChunkPages];
            if (!c) {
                p = end;
                continue;
            }
            for (; p <= last && p < end; p++)
                if (const uint64_t pv = c[p % kChunkPages]) {
                    if (ever) *ever = true;
                    v = std::max(v, pv);
                }
        }
        return v;
    }
    bool empty() const { return !any; }
} wb_pages;
// (hookshot) and the part of that submission the write-back was recorded in (vk_replay.cpp's Recording::part), for a page
// written back from its mirror: a submission is recorded in parts of ~120 draws (maybe_split), each part's end copies back
// what the GPU wrote in it, and the CPU's read needs that part done, not the submission's last one. Hyrule Field: the
// readbacks waited 5.5 ms a frame for last parts, the GPU's whole frame. 0 for a page written in place (note_direct_write:
// no mirror; the later parts of its submission may still read or write it in guest memory, so the read waits for all of it).
// A page's part is replaced with its value, by every note: the two are of the same write
WbPages wb_parts;
// (hookshot) the pages' last write in place (part 0: note_direct_write), for the late pages' fix: no record of which words,
// so a capture with such a page late waits, as before, whatever else writes it back
WbPages wb_direct;
// (hookshot) and whether a page was ever written back, read without wb_pages_mu: wb_pages' "ever" for invalidate_quick
// (vk_replay.cpp), on the game's thread with its virtual core held. Marked under wb_pages_mu before the pages' values, which
// are never dropped either (page_marks.h: the order)
PageMarks<kPages, kChunkPages> wb_ever;
// (RX_VK_LOG_RB_WRITERS, diagnostics: set once it is on) each range noted, to rb_page_note (below, with the writers it keeps)
void (*wb_writers_hook)(uint64_t lo, uint64_t hi, uint64_t value, uint64_t part) = nullptr;
void wb_pages_note(uint64_t guest_lo, uint64_t guest_hi, uint64_t value, uint64_t part) {
    if (!value || guest_hi <= guest_lo) return;
    std::lock_guard lk(wb_pages_mu);
    wb_ever.mark(guest_lo >> kPageShift, (guest_hi - 1) >> kPageShift);
    for (uint64_t p = guest_lo >> kPageShift; p <= (guest_hi - 1) >> kPageShift; p++) {
        wb_pages.set(p, value);
        if (part || !wb_parts.empty()) wb_parts.set(p, part);
        if (!part) wb_direct.set(p, value);
    }
    if (wb_writers_hook) wb_writers_hook(guest_lo, guest_hi, value, part);
}
} // namespace
/// (hookshot) Whether any page of [guest, guest + size) was ever written back: memory_writeback_value's *ever, without its lock
/// (invalidate_quick, on the game's thread with its virtual core held)
bool memory_writeback_ever(uint64_t guest, uint64_t size) {
    return size && wb_ever.any(guest >> kPageShift, (guest + size - 1) >> kPageShift);
}
/// The newest write-back value of the range's pages (0: none; *ever: whether any page was ever written back). With part
/// and whole: *part is the newest part that wrote any of them back from a mirror (0: none), *whole the newest value among
/// the pages with no part (written in place: their whole submission is what a read waits for; 0: none)
uint64_t memory_writeback_value(uint64_t guest, uint64_t size, bool* ever, uint64_t* part, uint64_t* whole) {
    uint64_t v = 0;
    if (ever) *ever = false;
    if (part) *part = 0;
    if (whole) *whole = 0;
    if (!size) return 0;
    std::lock_guard lk(wb_pages_mu);
    if (wb_pages.empty()) return 0;
    const uint64_t first = guest >> kPageShift, last = (guest + size - 1) >> kPageShift;
    v = wb_pages.max_in(first, last, ever);
    if (v && (part || whole)) {
        uint64_t newest = 0, in_place = 0;
        for (uint64_t p = first; p <= last && p < kPages;) {
            if (!wb_pages.chunks[p / kChunkPages]) {   // (a chunk never made: none of its pages, as max_in)
                p = (p / kChunkPages + 1) * kChunkPages;
                continue;
            }
            const uint64_t pv = wb_pages.get(p), pp = pv ? wb_parts.get(p) : 0;
            if (pp) newest = std::max(newest, pp);
            else if (pv) in_place = std::max(in_place, pv);
            p++;
        }
        if (part) *part = newest;
        if (whole) *whole = in_place;
    }
    return v;
}

/// A copy that fails instead of faulting on a page unmapped meanwhile (structured exception handling: the common case
/// costs a memcpy; ReadProcessMemory per page was ~2.3 us each). Not inlined, and the copy a call: without
/// -fasync-exceptions clang catches a fault in a __try block only across a call, and a 4 KB memcpy inlined into the
/// block (the per-page retry) faulted past the handler: the crash that kept the texture storage out of the capture.
/// (hookshot) The capture's copy with non-temporal stores (x86-64; opt-in, RX_VK_CAPTURE_NT=1; memcpy otherwise). A capture's
/// buffer comes from a pool that cycles through some thirty buffers (cap_bufs, cap_free): memory last written several
/// frames ago, in no cache by now. A plain store reads each line before it overwrites it (the read for ownership: the
/// copy asks for twice the memory traffic it needs), and the copy then stays in the caches, where nothing reads it
/// again: the GPU takes the buffer from memory, and ~28 MB a frame (Kokiri Forest: ~5 MB a submit, six submits) went
/// through the L3, pushing the game's own data out of it. A non-temporal store writes the line straight to memory
/// through the write-combining buffers: no read, nothing in the caches. Whole 64-byte lines only: the head up to the
/// destination's next line and the tail are plain copies. A fault on the source (a page unmapped meanwhile) leaves
/// through guarded_copy's handler as memcpy's does: not inlined, a call inside the __try block.
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((noinline)) void* copy_nt(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    if (n < 256) return std::memcpy(dst, src, n);
    if (const size_t head = (size_t)(0 - (uintptr_t)d) & 63) {
        std::memcpy(d, s, head);
        d += head, s += head, n -= head;
    }
    for (size_t lines = n >> 6; lines; lines--, d += 64, s += 64) {
        // (the source: written by the game this frame, read in order: a page ahead of the hardware's prefetch, which
        // stops at a page's end)
        _mm_prefetch((const char*)s + 512, _MM_HINT_T0);
#ifdef __AVX2__
        const __m256i a = _mm256_loadu_si256((const __m256i*)s), b = _mm256_loadu_si256((const __m256i*)(s + 32));
        _mm256_stream_si256((__m256i*)d, a);
        _mm256_stream_si256((__m256i*)(d + 32), b);
#else
        const __m128i a = _mm_loadu_si128((const __m128i*)s), b = _mm_loadu_si128((const __m128i*)(s + 16)),
                      c = _mm_loadu_si128((const __m128i*)(s + 32)), e = _mm_loadu_si128((const __m128i*)(s + 48));
        _mm_stream_si128((__m128i*)d, a);
        _mm_stream_si128((__m128i*)(d + 16), b);
        _mm_stream_si128((__m128i*)(d + 32), c);
        _mm_stream_si128((__m128i*)(d + 48), e);
#endif
    }
    _mm_sfence();   // (the stores in memory before whoever reads the buffer next: the replay's thread, the GPU)
    if (n & 63) std::memcpy(d, s, n & 63);
    return dst;
}
// (opt-in since 2026-11-10: four interleaved field runs on an RTX desktop (main + #1619) showed no lower capture time with
// it, 0.41-0.44 ms a submit either way, and more stutters: 3.43% and 4.22% of frames over 39-41 ms with it, 3.08% and 2.89%
// without; p99 61.9/65.8 ms against 58.5/54.7. RX_VK_CAPTURE_NT=1 turns it on, for CPUs where it helps)
bool capture_nt() {
    static const bool on = getenv("RX_VK_CAPTURE_NT") && atoi(getenv("RX_VK_CAPTURE_NT")) != 0;
    return on;
}
void* (*volatile guarded_memcpy)(void*, const void*, size_t) = capture_nt() ? copy_nt : std::memcpy;
/// (a copy that left at a fault did not reach its fence, and the next reader is another thread)
inline void copy_fence() {
    if (capture_nt()) _mm_sfence();
}
#else
void* (*volatile guarded_memcpy)(void*, const void*, size_t) = std::memcpy;
inline void copy_fence() {}
#endif
#ifdef _WIN32
__declspec(noinline) bool guarded_copy(void* dst, const void* src, size_t n) {
    __try {
        guarded_memcpy(dst, src, n);
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
    return true;
}
#else
// (linux: the guest window is one mapping that stays readable, decommitted pages read as zero: nothing to fault on)
bool guarded_copy(void* dst, const void* src, size_t n) {
    guarded_memcpy(dst, src, n);
    return true;
}
#endif
// The capture's copies with streaming stores. The capture buffers (25 kept and the spares, 5-10 MB each) are never in
// the CPU's cache when a capture fills them, so a plain copy read every line of them from DRAM before writing it (~40 MB a frame of
// it, on the game's main thread, the other game threads mostly idle meanwhile): bandwidth, not threads, held the copy (four helpers
// or eight, the same time). Streaming stores skip that read: ~0.43 -> ~0.31 ms for a 10 MB capture on this machine (5 threads,
// the source just written by 6 others). What the buffers hold is the same; the stores are fenced before the capture is handed on.
// RX_VK_CAPTURE_STREAM=0: memcpy, as before
#if defined(__AVX2__)
__attribute__((noinline)) void* stream_memcpy(void* dst, const void* src, size_t n) {
    if (((uintptr_t)dst & 31) || (n & 127)) return std::memcpy(dst, src, n);
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    for (size_t o = 0; o < n; o += 128) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + o));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + o + 32));
        const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + o + 64));
        const __m256i e = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + o + 96));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(d + o), a);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(d + o + 32), b);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(d + o + 64), c);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(d + o + 96), e);
    }
    return dst;
}
void stream_fence() { _mm_sfence(); }
bool stream_on() {
    static const bool v = env_on("RX_VK_CAPTURE_STREAM");   // (off by default here: =1 on)
    return v;
}
#else
void* stream_memcpy(void* dst, const void* src, size_t n) { return std::memcpy(dst, src, n); }
void stream_fence() {}
bool stream_on() { return false; }
#endif
void* (*volatile guarded_stream_memcpy)(void*, const void*, size_t) = stream_memcpy;
#ifdef _WIN32
__declspec(noinline) bool guarded_copy_stream(void* dst, const void* src, size_t n) {   // (as guarded_copy)
    __try {
        guarded_stream_memcpy(dst, src, n);
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
    return true;
}
#else
bool guarded_copy_stream(void* dst, const void* src, size_t n) {
    guarded_stream_memcpy(dst, src, n);
    return true;
}
#endif
// ---- a capture's copies: on the submitting thread and two helpers, four from 12 CPU threads (RX_VK_CAPTURE_THREADS=<n>, 0-4): ~9 MB of the
// game's dynamic data a submit, four submits a frame, ~1.4 ms on the game's thread each with one ----
// (a page written and then unmapped before the submit faulted a plain copy: ReadProcessMemory fails on it instead.
// VirtualQuery per page took ~1 ms each in this address space: 22 ms a submit for 18 pages) (runs of adjacent pages
// in one copy each, page by page only where a run fails; page i into slot i)
struct CopyPair { uint8_t* dst; const uint8_t* src; };   // (a 4 KB copy between two of our own buffers: copy_pairs)
struct CopyPart {
    const uint32_t* pages;
    int64_t i0, i1;
    uint8_t* out;
    uint8_t* ok;
    const CopyPair* pairs = nullptr;
    int* left = nullptr;   // (the caller's parts not done yet (copy_helpers.mu): a caller waits for its own, not another's too)
    uint8_t* stage = nullptr;       // (guest copies only: the pages with sidx[i] >= 0 are copied into stage + sidx[i] * 4096 first, and out from there)
    const int32_t* sidx = nullptr;
};
void copy_part(const CopyPart& w) {
    if (w.pairs) {
        for (int64_t i = w.i0; i < w.i1; i++) std::memcpy(w.pairs[i].dst, w.pairs[i].src, 4096);
        return;
    }
    uint8_t* const base = (uint8_t*)host->guest_base();   // (guest addresses: no GPU translation per page)
    const uint64_t gsize = host->guest_size();
    for (int64_t i = w.i0; i < w.i1;) {
        const uint64_t ga = (uint64_t)w.pages[i] << kPageShift;
        if (ga + 4096 > gsize) { i++; continue; }
        int64_t j = i + 1;
        while (j < w.i1 && w.pages[j] == w.pages[j - 1] + 1 && ((uint64_t)w.pages[j] << kPageShift) + 4096 <= gsize) j++;
        // (a run in out alone; a run of staged pages: guest memory into the staging first, then out from the staging, so the two
        // hold the same bytes even if the game writes the page meanwhile; a page that fails is ok = 0 in both and not used)
        auto plain = [&](int64_t a, int64_t b) {
            if (guarded_copy(w.out + (uint64_t)a * 4096, base + ((uint64_t)w.pages[a] << kPageShift), (SIZE_T)(b - a) * 4096)) {
                std::memset(w.ok + a, 1, (size_t)(b - a));
            } else {
                for (int64_t q = a; q < b; q++)
                    w.ok[q] = guarded_copy(w.out + (uint64_t)q * 4096, base + ((uint64_t)w.pages[q] << kPageShift), 4096);
                copy_fence();
            }
        };
        auto staged = [&](int64_t a, int64_t b) {
            uint8_t* const st = w.stage + (uint64_t)w.sidx[a] * 4096;
            if (guarded_copy(st, base + ((uint64_t)w.pages[a] << kPageShift), (SIZE_T)(b - a) * 4096)) {
                std::memcpy(w.out + (uint64_t)a * 4096, st, (size_t)(b - a) * 4096);
                std::memset(w.ok + a, 1, (size_t)(b - a));
            } else {
                for (int64_t q = a; q < b; q++) {
                    uint8_t* const sq = w.stage + (uint64_t)w.sidx[q] * 4096;
                    const bool good = guarded_copy(sq, base + ((uint64_t)w.pages[q] << kPageShift), 4096);
                    if (good) std::memcpy(w.out + (uint64_t)q * 4096, sq, 4096);
                    w.ok[q] = good;
                }
                copy_fence();
            }
        };
        if (!w.sidx) {
            plain(i, j);
        } else {
            for (int64_t q = i; q < j;) {   // (split at the staged / not staged edges; staged slots are consecutive for adjacent pages)
                int64_t r = q + 1;
                if (w.sidx[q] >= 0) {
                    while (r < j && w.sidx[r] == w.sidx[r - 1] + 1) r++;
                    staged(q, r);
                } else {
                    while (r < j && w.sidx[r] < 0) r++;
                    plain(q, r);
                }
                q = r;
            }
        }
        i = j;
    }
}
struct CopyHelpers {
    std::mutex mu;
    std::condition_variable cv, done;
    std::deque<CopyPart> parts;
    bool started = false;
} copy_helpers;
// ---- RX_VK_CAPTURE_CLAIM=1 (off by default): the capture's copy in chunks claimed by whoever is free ----
// The copy was cut into k = helpers + 1 equal parts; the submitter copied part 0 and then slept until the slowest helper was
// through. With the game's job group on every core (HOS_NVN_HOLD=2's held flush runs beside it) a helper was often scheduled
// late, and the copy ended with the last part to start (1.36 ms held against 0.6 ms on an idle machine). Here the
// pages are cut into chunks of `chunk` pages, claimed from one atomic counter by the submitter and every helper that wakes in
// time; the submitter claims until none is left, so a helper that never got a core leaves nothing undone.
// Same output, by construction: chunk [i0, i1) is copied by exactly one thread with copy_part, which writes page i into slot i
// (out + i * 4096) and ok[i], as before; a run of adjacent pages cut at a chunk's edge is two copies of the same bytes into the
// same slots (and a failing run falls back page by page within each chunk, so ok[] gets the same per-page answer). The copy reads
// guest memory as before: the order the threads read pages in changes, and the capture always allowed any order (it ran on 5
// threads already). Visibility: copy_part fences its streaming stores; a helper drops its use count under copy_helpers.mu after
// its last chunk, and the submitter waits on that mutex for the count to reach 0, so every store is visible to it when
// copy_pages returns, as with the pending count before.
// Lifetime: the job lives on the submitter's stack. A token not popped yet is removed by the submitter (under the mutex) before
// it waits; a token popped holds `users`, which the submitter waits for. No helper touches the job after copy_pages returns.
struct ClaimJob {
    const uint32_t* pages;
    int64_t n;
    uint8_t* out;
    uint8_t* ok;
    int64_t chunk;
    std::atomic<int64_t> next{0};
    int users = 0;   // (helpers that popped a token and have not finished: copy_helpers.mu)
    // (other work cut the same way: fn(ctx, i0, i1) for each chunk instead of copy_part; null: the copy)
    void (*fn)(void* ctx, int64_t i0, int64_t i1) = nullptr;
    void* ctx = nullptr;
    // (stream R4, RX_VK_CLAIM_HASH: the copy's chunk hashes, as CopyPart's. A claimed copy made none (RX_VK_CAPTURE_CLAIM went on by default
    // after 's hashing copy), and the mirror syncs' chunk diff hashed every page again on the replay thread: 3.7 ms of a
    // frame's 5.4 ms pre-sync on the PC 3050, whose link is slow enough for the diff to be on)
    uint64_t* hash = nullptr;
    uint8_t* hok = nullptr;
};
bool claim_hash_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_CLAIM_HASH");
        const bool v = e && atoi(e) != 0;   // (off by default here; the fork measured PC 3050, castle-move: 31.6 -> 36.1 fps)
        if (v) logf("vk: a claimed capture copy makes the chunk hashes too (RX_VK_CLAIM_HASH)");
        return v;
    }();
    return on;
}
bool capture_claim_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_CAPTURE_CLAIM");
        const bool v = e && atoi(e) != 0;
        logf("vk: the capture's copy: %s (RX_VK_CAPTURE_CLAIM)", v ? "chunks claimed by the submitter and the helpers" : "equal parts, as before");
        return v;
    }();
    return on;
}
int64_t claim_chunk_pages(int64_t n, int participants) {
    // (RX_VK_CAPTURE_CLAIM=<n> >= 8: n pages a chunk; 1: about four chunks a participant, 16 to 256 pages (64 KB to 1 MB))
    static const int64_t fixed = getenv("RX_VK_CAPTURE_CLAIM") ? atoll(getenv("RX_VK_CAPTURE_CLAIM")) : 0;
    if (fixed >= 8) return fixed;
    return std::clamp<int64_t>(n / (4 * std::max(participants, 1)), 16, 256);
}
void copy_helpers_start_locked();
std::atomic<int64_t> st_claim_copies{0}, st_claim_helped{0};   // (copies made with claiming; tokens a helper took in time)
void copy_helper_main() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk capture copy");
    for (;;) {
        CopyPart w;
        {
            std::unique_lock lk(copy_helpers.mu);
            copy_helpers.cv.wait(lk, [] { return !copy_helpers.parts.empty(); });
            w = copy_helpers.parts.front();
            copy_helpers.parts.pop_front();
        }
        copy_part(w);
        std::lock_guard lk(copy_helpers.mu);
        if (--*w.left == 0) copy_helpers.done.notify_all();
    }
}
int capture_copy_helpers() {   // (copy_pages and copy_pairs share one pool of helpers, started by the first call that splits)
    static const int helpers = getenv("RX_VK_CAPTURE_THREADS")
        ? std::clamp(atoi(getenv("RX_VK_CAPTURE_THREADS")), 0, 4)
        : (env_on("RX_VK_LOW_SPEC") ? 0 : std::thread::hardware_concurrency() >= 12 ? 4 : 2);   // (four where the threads are spare)
    return helpers;
}
/// (hookshot) whole's n copies in helpers + 1 parts: the first on the calling thread, the others queued for the helpers.
/// Under 128 copies (512 KB) all on the calling thread: waking the helpers costs more than they save. Its own parts no
/// helper has taken yet the caller runs itself once its first is done, rather than waiting behind what another caller
/// queued before them (copy_pairs runs with pools_mu and shadow_mu held, copy_pages' guest copies can take milliseconds)
void copy_split(CopyPart whole, int64_t n) {
    const int helpers = capture_copy_helpers();
    if (!helpers || n < 128) { whole.i0 = 0, whole.i1 = n; return copy_part(whole); }
    const int k = helpers + 1;
    int left = k - 1;
    auto part = [&](int i) {
        CopyPart w = whole;
        w.i0 = n * i / k, w.i1 = n * (i + 1) / k, w.left = &left;
        return w;
    };
    {
        std::lock_guard lk(copy_helpers.mu);
        if (!copy_helpers.started) {
            copy_helpers.started = true;
            for (int i = 0; i < helpers; i++) std::thread(copy_helper_main).detach();
        }
        for (int i = 1; i < k; i++) copy_helpers.parts.push_back(part(i));
    }
    copy_helpers.cv.notify_all();
    copy_part(part(0));
    std::unique_lock lk(copy_helpers.mu);
    for (;;) {
        auto it = std::find_if(copy_helpers.parts.begin(), copy_helpers.parts.end(), [&](const CopyPart& w) { return w.left == &left; });
        if (it == copy_helpers.parts.end()) break;
        const CopyPart w = *it;
        copy_helpers.parts.erase(it);
        lk.unlock();
        copy_part(w);
        lk.lock();
        --left;
    }
    copy_helpers.done.wait(lk, [&] { return left == 0; });
}
void copy_pages(const uint32_t* pages, int64_t n, uint8_t* out, uint8_t* ok, uint8_t* stage = nullptr, const int32_t* sidx = nullptr) {   // (one capture at a time: submit_order_mu)
    CopyPart w{pages, 0, n, out, ok};
    w.stage = stage, w.sidx = sidx;
    copy_split(w, n);
}
/// (hookshot) n 4 KB copies on the capture's copy helpers too: the texture shadow's (take_capture), thousands of pages a submit
/// for OoT's game thread, which copied them one after another
void copy_pairs(const CopyPair* pairs, int64_t n) {
    CopyPart w{nullptr, 0, n, nullptr, nullptr};
    w.pairs = pairs;
    copy_split(w, n);
}

// (RX_VK_CAPTURE_GRAN) pages [i0, i1) copied by their granule masks: whole pages in runs as copy_part, the others
// run of granules by run, each into its place in the page's slot
struct GranCopy { const uint32_t* pages; uint8_t* masks; uint8_t* out; uint8_t* ok; };
std::unique_ptr<Capture> capture_make();
void shadow_free(Pool& p);
void shadow_forget(Pool& p, uint64_t at, uint64_t len);   // (the write-back's diff, below: a range its shadow no longer knows)
void memory_capture_submit() {
#ifdef TRACY_ENABLE
    HT_SCOPE("memory_capture_submit");
#endif
    std::unique_ptr<Capture> c = capture_make();
    if (!c) return;
    std::lock_guard lk(cap_mu);
    captures.push_back(std::move(c));
    submits_queued.fetch_add(1, std::memory_order_release);
}
// ---- the submit's scan of the write marks on the copy helpers: RX_VK_CAPTURE_SCAN_SPLIT=1 ----
// The scan (collect_in over ~1,500 ranges, ~3.4 GB of guest memory: ~830 KB of marks) is ~0.07 ms a submit on the submitting
// thread, six submits a frame. Split by pages between it and the copy helpers; the parts' pages joined in range order (the
// same list, in the same order, as one scan).
bool scan_split_on() {
    static const bool v = getenv("RX_VK_CAPTURE_SCAN_SPLIT") && atoi(getenv("RX_VK_CAPTURE_SCAN_SPLIT")) != 0;
    return v;
}
int precopy_period_us() {
    static const int v = getenv("RX_VK_PRECOPY_US") ? std::clamp(atoi(getenv("RX_VK_PRECOPY_US")), 50, 100000) : 250;
    return v;
}
struct PreCopy {
    std::mutex mu;                   // (the copy below; the thread holds it while it copies, a capture takes it)
    std::atomic<bool> stop{false};   // (a capture is waiting for mu: the thread stops at its next page)
    std::vector<uint32_t> pages;     // (copied, page i in slot i)
    std::vector<uint64_t> done;      // (what the GPU had done when each one's copy began)
    std::vector<uint8_t> ok;
    std::unordered_map<uint32_t, uint32_t> slot_of;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    void* map = nullptr;
    uint64_t bsize = 0;
    uint64_t want = 4ull << 20;      // (the next buffer's size: the recent captures', and room)
    bool full = false;
    int64_t st_again = 0, st_left = 0, st_runs = 0;   // (pages copied again by the thread, given back as the buffer was full; its rounds)
    std::mutex ranges_mu;            // (the last capture's ranges, for the thread)
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    uint64_t ranges_ver = 0;
    bool started = false;            // (mu)
    uint64_t gen = 0;                // (mu: the captures that took the copies)
    int64_t st_grown = 0;
} pre;
struct PreTaken {
    std::vector<uint32_t> pages;
    std::vector<uint64_t> done;
    std::vector<uint8_t> ok;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    void* map = nullptr;
    uint64_t bsize = 0;
};
/// (mu held, the thread's) The buffer twice as large, what it holds copied over: false when it cannot be, or a capture took the
/// copies while the new one was made (mu let go meanwhile)
bool precopy_grow(std::unique_lock<std::mutex>& lk) {
    const uint64_t gen = pre.gen, want = std::min<uint64_t>(pre.bsize * 2, 64ull << 20);
    if (want <= pre.bsize) return false;
    lk.unlock();
    VkBuffer b = VK_NULL_HANDLE;
    VmaAllocation a = nullptr;
    void* m = nullptr;
    uint64_t sz = 0;
    const bool got = cap_buf_get(want, b, a, m, sz);
    lk.lock();
    if (!got) {
        if (b) destroy_buffer(b, a);
        return false;
    }
    if (pre.gen != gen || !pre.map || sz <= pre.bsize) {
        cap_buf_free(b, a, m, sz);
        return false;
    }
    for (size_t o = 0; o < pre.pages.size(); o += 256) {   // (a capture waiting: given up)
        if (pre.stop.load(std::memory_order_relaxed)) {
            cap_buf_free(b, a, m, sz);
            return false;
        }
        const size_t k = std::min<size_t>(256, pre.pages.size() - o);
        stream_memcpy((uint8_t*)m + o * 4096, (const uint8_t*)pre.map + o * 4096, k * 4096);
    }
    stream_fence();
    cap_buf_free(pre.buffer, pre.alloc, pre.map, pre.bsize);
    pre.buffer = b, pre.alloc = a, pre.map = m, pre.bsize = sz;
    pre.st_grown++;
    return true;
}
// (Hookshot) RX_VK_PRECOPY_KICK=1 (with RX_VK_CAPTURE_PRECOPY=1): the thread copies once per kick from the host
// (mkwvk_precopy_kick, called when the game's main thread leaves its end-of-frame wait: the job threads' writes of the
// frame are final then and the main thread's draw code and submits are still 2-4 ms away), not on the 250 us timer whose
// rounds during the job threads' work cost more than they saved . The copies then run while the job threads idle.
bool precopy_kick_on() {
    static const bool v = getenv("RX_VK_PRECOPY_KICK") && atoi(getenv("RX_VK_PRECOPY_KICK")) != 0;
    return v;
}
// (Hookshot) RX_VK_PRECOPY_KICK=2 (with RX_VK_CAPTURE_PRECOPY=1): one round a frame, kicked by the renderer when a
// present begins (memory_precopy_present), copied by the capture's own helpers (copy_pages: runs of pages, every helper), into a
// buffer sized for what was collected. Why there: with the held flush early (HOS_NVN_HOLD_LINGER_US) the frame's first capture
// (the pages the game wrote during the whole frame: 30-40 MB, 1.6-2 ms) is made on the Presentation Thread at the start of the
// render chain, which the frame waits for; when the previous present begins, the game's job group and module phase of that next
// frame are nearly over, the replay's workers and the copy helpers idle, and the present takes 0.6-1.5 ms. The capture at the
// submit then copies what was written since and what was written again (capture_merge.h: the same pages as at the submit).
// kick 1's single-thread page-by-page rounds copied 100-300 pages ahead of ~7,500 .
int precopy_kick_mode() {
    static const int v = getenv("RX_VK_PRECOPY_KICK") ? atoi(getenv("RX_VK_PRECOPY_KICK")) : 0;
    return v;
}
std::atomic<int64_t> st_pre2_rounds{0}, st_pre2_pages{0}, st_pre2_skipped{0}, st_pre2_ns{0};
HANDLE precopy_kick_event = nullptr;
std::atomic<int64_t> st_precopy_kicks{0};
void precopy_main() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk capture pre-copy");
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x2
#endif
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!timer) timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    uint64_t ver = 0;
    uint8_t* const base = (uint8_t*)host->guest_base();
    const uint64_t gsize = host->guest_size();
    for (;;) {
        LARGE_INTEGER due;
        due.QuadPart = -(LONGLONG)precopy_period_us() * 10;
        if (precopy_kick_on() && precopy_kick_event) WaitForSingleObject(precopy_kick_event, INFINITE);   // (a kick a frame)
        else if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, INFINITE);
        else Sleep(1);
        if (pre.stop.load(std::memory_order_relaxed)) continue;
        {
            std::lock_guard rk(pre.ranges_mu);
            if (pre.ranges_ver != ver) ranges = pre.ranges, ver = pre.ranges_ver;
        }
        if (ranges.empty()) continue;
        std::unique_lock lk(pre.mu);
        if (pre.full || pre.stop.load(std::memory_order_relaxed)) continue;
        if (precopy_kick_mode() >= 2) {   // (one round on the capture's helpers; see precopy_kick_mode)
            if (!pre.pages.empty()) { st_pre2_skipped++; continue; }   // (the last round's copies not taken yet)
            const auto r0 = std::chrono::steady_clock::now();
            const uint64_t done = vk_timeline_completed();   // (before the marks are cleared, as below)
            uint32_t* pages = nullptr;
            const int64_t n = host->collect_written_in(reinterpret_cast<const uint64_t(*)[2]>(ranges.data()), (int)ranges.size(), &pages);
            bool taken = false;
            if (n > 0 && (uint64_t)n * 4096 <= (96ull << 20)) {
                const uint64_t need = (uint64_t)n * 4096 * 5 / 4 + (4ull << 20);   // (room for the pages written after)
                if (pre.map && pre.bsize < need) cap_buf_free(pre.buffer, pre.alloc, pre.map, pre.bsize), pre.buffer = VK_NULL_HANDLE, pre.map = nullptr, pre.bsize = 0;
                if (!pre.map && !cap_buf_get(need, pre.buffer, pre.alloc, pre.map, pre.bsize)) {
                    if (pre.buffer) destroy_buffer(pre.buffer, pre.alloc);
                    pre.buffer = VK_NULL_HANDLE, pre.map = nullptr, pre.bsize = 0;
                }
                if (pre.map) {
                    pre.ok.assign((size_t)n, 0);
                    copy_pages(pages, n, (uint8_t*)pre.map, pre.ok.data(), nullptr, nullptr);
                    pre.pages.assign(pages, pages + n);
                    pre.done.assign((size_t)n, done);
                    taken = true;
                    st_pre2_rounds++, st_pre2_pages += n;
                }
            }
            if (!taken)
                for (int64_t i = 0; i < n; i++) host->mark_written(base + ((uint64_t)pages[i] << kPageShift), 4096), pre.st_left++;
            if (pages) host->free_pages(pages);
            st_pre2_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - r0).count();
            continue;
        }
        if (!pre.map) {   // (the next capture's buffer, made here: not on the submitting thread)
            const uint64_t want = pre.want;
            lk.unlock();
            VkBuffer b = VK_NULL_HANDLE;
            VmaAllocation a = nullptr;
            void* m = nullptr;
            uint64_t sz = 0;
            const bool got = cap_buf_get(want, b, a, m, sz);
            lk.lock();
            if (!got) {
                if (b) destroy_buffer(b, a);
                continue;
            }
            pre.buffer = b, pre.alloc = a, pre.map = m, pre.bsize = sz;
            if (pre.stop.load(std::memory_order_relaxed)) continue;
        }
        const uint64_t done = vk_timeline_completed();   // (before the marks are cleared: a write-back after it is copied again)
        uint32_t* pages = nullptr;
        const int64_t n = host->collect_written_in(reinterpret_cast<const uint64_t(*)[2]>(ranges.data()), (int)ranges.size(), &pages);
        pre.st_runs++;
        uint64_t slots = pre.bsize / 4096 * 3 / 4;   // (a quarter left for the pages written after, at the submit)
        const bool stream = stream_on();
        int64_t i = 0;
        for (; i < n; i++) {
            if (pre.stop.load(std::memory_order_relaxed)) break;
            const uint32_t p = pages[i];
            uint32_t slot;
            if (const auto it = pre.slot_of.find(p); it != pre.slot_of.end()) {
                slot = it->second;
                pre.st_again++;
            } else {
                if (pre.pages.size() >= slots) {
                    const uint64_t gen = pre.gen;
                    if (!precopy_grow(lk)) {   // (the rest given back below, this page with them)
                        if (pre.gen == gen) pre.full = true;
                        break;
                    }
                    slots = pre.bsize / 4096 * 3 / 4;
                }
                slot = (uint32_t)pre.pages.size();
                pre.pages.push_back(p), pre.done.push_back(0), pre.ok.push_back(0);
                pre.slot_of.emplace(p, slot);
            }
            const uint64_t ga = (uint64_t)p << kPageShift;
            uint8_t* dst = (uint8_t*)pre.map + (uint64_t)slot * 4096;
            bool ok = false;
            if (ga + 4096 <= gsize) {
                ok = stream ? guarded_copy_stream(dst, base + ga, 4096) : guarded_copy(dst, base + ga, 4096);
                if (!ok && stream) ok = guarded_copy(dst, base + ga, 4096);
            }
            pre.done[slot] = done, pre.ok[slot] = ok ? 1 : 0;
        }
        for (; i < n; i++) {   // (not copied: given back to the marks, for the submit's scan)
            host->mark_written(base + ((uint64_t)pages[i] << kPageShift), 4096);
            pre.st_left++;
        }
        if (stream) stream_fence();   // (the copies visible before the capture that takes them)
        if (pages) host->free_pages(pages);
    }
}
/// The pages copied so far and their buffer, for the capture being made (the thread stopped at its next page)
void precopy_take(PreTaken& t) {
    pre.stop.store(true, std::memory_order_relaxed);
    std::lock_guard lk(pre.mu);
    if (!pre.started) {
        pre.started = true;
        if (precopy_kick_on()) precopy_kick_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        std::thread(precopy_main).detach();
    }
    t.pages.swap(pre.pages), t.done.swap(pre.done), t.ok.swap(pre.ok);
    pre.pages.clear(), pre.done.clear(), pre.ok.clear(), pre.slot_of.clear();
    t.buffer = pre.buffer, t.alloc = pre.alloc, t.map = pre.map, t.bsize = pre.bsize;
    pre.buffer = VK_NULL_HANDLE, pre.alloc = nullptr, pre.map = nullptr, pre.bsize = 0;
    pre.full = false;
    pre.gen++;
    pre.stop.store(false, std::memory_order_relaxed);
}
/// (the next buffer: 1.5x the mean of the last 16 captures, 2-32 MB. Not their largest: one large capture in 16 made every buffer
/// 40-60 MB, and the 25 kept ones went past the captures' byte budget (821 MB in 17). A capture larger than the room is copied at
/// its submit, as without the thread)
void precopy_size(int64_t n) {
    static int64_t last[16];
    static int at;
    last[at++ & 15] = n;
    int64_t m = 0;
    for (int64_t x : last) m += x;
    m /= 16;
    const uint64_t want = std::min<uint64_t>(std::max<uint64_t>(2ull << 20, (uint64_t)m * 4096 * 3 / 2), 32ull << 20);
    std::lock_guard lk(pre.mu);
    pre.want = want;
}
void precopy_ranges(const std::vector<std::pair<uint64_t, uint64_t>>& r) {
    std::lock_guard lk(pre.ranges_mu);
    pre.ranges = r;
    pre.ranges_ver++;
}
/// (the copied-ahead and the collected pages into one list, each once, in the order of the pages: the capture as without the thread)
void precopy_union(const PreTaken& t, const uint32_t* fresh, int64_t n, std::vector<uint32_t>& out) {
    out.assign(t.pages.begin(), t.pages.end());
    out.insert(out.end(), fresh, fresh + std::max<int64_t>(n, 0));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}
/// Copies pages[i] into slot slots[i] of out, ok[slots[i]] set (runs of adjacent pages and slots in one copy)
void copy_slots(const uint32_t* pages, const uint32_t* slots, size_t n, uint8_t* out, uint8_t* ok) {
    uint8_t* const base = (uint8_t*)host->guest_base();
    const uint64_t gsize = host->guest_size();
    const bool stream = stream_on();
    for (size_t i = 0; i < n;) {
        const uint64_t ga = (uint64_t)pages[i] << kPageShift;
        if (ga + 4096 > gsize) { ok[slots[i]] = 0; i++; continue; }
        size_t j = i + 1;
        while (j < n && pages[j] == pages[j - 1] + 1 && slots[j] == slots[j - 1] + 1 && ((uint64_t)pages[j] << kPageShift) + 4096 <= gsize) j++;
        uint8_t* dst = out + (uint64_t)slots[i] * 4096;
        if (stream ? guarded_copy_stream(dst, base + ga, (SIZE_T)(j - i) * 4096) : guarded_copy(dst, base + ga, (SIZE_T)(j - i) * 4096)) {
            for (size_t q = i; q < j; q++) ok[slots[q]] = 1;
        } else {
            for (size_t q = i; q < j; q++)
                ok[slots[q]] = guarded_copy(out + (uint64_t)slots[q] * 4096, base + ((uint64_t)pages[q] << kPageShift), 4096);
        }
        i = j;
    }
    if (stream) stream_fence();
}
namespace {
/// (capture_make, the game's thread) The pages the CPU wrote in the captured ranges since they were last taken, their
/// marks cleared; not while the replay's collection is marking some of them again (collect_mu)
int64_t capture_scan(const std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint32_t** pages) {
    if (ranges.empty() || !host->collect_written_in) return 0;
    std::lock_guard lk(collect_mu);
    return host->collect_written_in(reinterpret_cast<const uint64_t(*)[2]>(ranges.data()), (int)ranges.size(), pages);
}
}
/// (the pages the CPU wrote since the last capture, copied now: a submit's capture, or memory_capture_now's)
uint64_t memory_captures_begun() { return captures_begun.load(std::memory_order_acquire); }
std::unique_ptr<Capture> capture_make() {
#ifdef TRACY_ENABLE
    HT_SCOPE("capture_make");
#endif
    if (!capture_on()) return nullptr;
    captures_begun.fetch_add(1, std::memory_order_release);   // (before the written pages are collected: memory_captures_begun)
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<Capture> c;
    {
        std::lock_guard lk(cap_mu);
        if (!spare.empty()) c = std::move(spare.back()), spare.pop_back();
    }
    if (!c) c = std::make_unique<Capture>();
    c->pages.clear(), c->copied.clear(), c->slots.clear(), c->nbytes = 0, c->late.clear(), c->late_pv.clear(), c->late_value = 0,
    c->late_done = 0;
    if (c->map) cap_buf_free(c->buffer, c->alloc, c->map, c->bsize), c->buffer = VK_NULL_HANDLE, c->map = nullptr;
    stage_put(*c);
    // (only the pools mirrored in VRAM, the ones the mirrors are filled from, and only their part of the write marks:
    // every written page was taken at first (hundreds of MB a submit where the game streams its world), and the whole
    // collection, over every GPU range, is ~20 ms: 4 of them a frame on the game's thread made it crawl. The replay's
    // collection takes the rest, as before)
    // (the list the replay publishes when it makes mirrors: pools_mu here waited ~40 ms a submit for the replay)
    static thread_local std::vector<std::pair<uint64_t, uint64_t>> mirrored;
    static thread_local std::vector<std::pair<uint64_t, uint64_t>> texr;   // (tex_cap_ranges, sorted and merged: the staged pages)
    {   // (and the texture storage the replay uploads from: see "texture storage in the capture"; and the small pools)
        std::lock_guard lk(cap_ranges_mu);
        captures_made++;
        refresh_pool_ranges_locked();
        static thread_local uint64_t ver = 0;
        if (ver != cap_ranges_ver) {   // (the three lists overlap: their union, sorted, once each time one changes)
            ver = cap_ranges_ver;
            std::vector<std::pair<uint64_t, uint64_t>> all = cap_ranges;
            all.insert(all.end(), tex_cap_ranges.begin(), tex_cap_ranges.end());
            all.insert(all.end(), pool_cap_ranges.begin(), pool_cap_ranges.end());
            std::sort(all.begin(), all.end());
            mirrored.clear();
            for (const auto& r : all) {
                if (!mirrored.empty() && r.first <= mirrored.back().second)
                    mirrored.back().second = std::max(mirrored.back().second, r.second);
                else
                    mirrored.push_back(r);
            }
            texr = tex_cap_ranges;
            std::sort(texr.begin(), texr.end());
            size_t m = 0;
            for (const auto& r : texr) {
                if (m && r.first <= texr[m - 1].second) texr[m - 1].second = std::max(texr[m - 1].second, r.second);
                else texr[m++] = r;
            }
            texr.resize(m);
        }
    }
    captures_seen.store(true, std::memory_order_relaxed);
    const auto t_lock = std::chrono::steady_clock::now();
    uint32_t* pages = nullptr;
    const int64_t n = capture_scan(mirrored, &pages);
    if (n > 0) early_check_cpu_pages(pages, n, false, 0);   // (RX_VK_LOG_EARLY_CHECK: the CPU's writes)
    const auto t_scan = std::chrono::steady_clock::now();
    // (hookshot) A page the CPU wrote that a submission still on the GPU writes back: the copy waits for that write-back.
    // OoT's exposure meter page is written by both: the GPU adds the frame's light into it and the CPU sets the sums back to
    // 0 and fills in the rest. The CPU runs a frame ahead, and a copy taken before the write-back had landed had the GPU's
    // words from the frame before; the mirror sync of the page put them back over the GPU's newer ones, and the exposure
    // stepped for a frame (one-frame flashes, 40-120 a minute in Kokiri Forest). Only the copies of such pages wait, for the
    // last submission writing them back: not every frame's present. That wait held the game's thread (~8% of its time at a
// ~2150-draw scene), so those pages are copied again by the replay, before it uses this capture, once the submission is
// done: the replay waits instead, while the game goes on. RX_VK_CAPTURE_WAIT=1: the game's thread waits, 0: no wait
    if (n > 0) {
        static const int cap_wait = getenv("RX_VK_CAPTURE_WAIT") ? atoi(getenv("RX_VK_CAPTURE_WAIT")) : 2;
        uint64_t v = 0;
        if (cap_wait) {
            const uint64_t done = vk_timeline_completed();
            c->late_done = done;
            std::lock_guard lk(wb_pages_mu);
            if (!wb_pages.empty())
                for (int64_t i = 0; i < n; i++) {
                    const uint64_t pv = wb_pages.get(pages[i]);
                    if (pv > done) {
                        v = std::max(v, pv);
                        if (cap_wait == 2) c->late.push_back((uint32_t)i), c->late_pv.push_back(pv);
                    }
                }
        }
        if (v && cap_wait == 2) c->late_value = v;
        else if (v) vk_timeline_wait(v);
    }
    if (n > 0) shadow_written(pages, n);
    if (n > 0) {
        c->pages.assign(pages, pages + n);
        if ((uint64_t)n * 4096 <= (256ull << 20) &&   // (a load writes more: those are read later, as before)
            cap_buf_get((uint64_t)n * 4096, c->buffer, c->alloc, c->map, c->bsize)) {
            static thread_local std::vector<uint8_t> ok;
            ok.assign((size_t)n, 0);
            // (shadow staging: the pages in the texture ranges are also copied into c->stage, from guest memory first and the buffer
            // from there. Not the late pages: take_capture copies those again into the buffer after the GPU wait, so the buffer is
            // where their newer bytes are, and they stay unstaged (sidx -1: the shadow reads them from the buffer; a few a take))
            // (only where the capture buffer is not cached memory: a cached one the shadow reads as fast as the staging, and the
            // extra copy would be the game thread's for nothing)
            c->buf_cached = host_cached(c->alloc);
            if (!texr.empty() && !c->buf_cached) {
                c->sidx.assign((size_t)n, -1);
                int32_t tn = 0;
                size_t lj = 0;
                RangeWalk in_texr{texr};   // (pages in address order: collect_in over `mirrored`)
                for (int64_t i = 0; i < n; i++) {
                    while (lj < c->late.size() && c->late[lj] < (uint32_t)i) lj++;
                    if (lj < c->late.size() && c->late[lj] == (uint32_t)i) continue;
                    const uint64_t g = (uint64_t)pages[i] << kPageShift;
                    if (in_texr.in(g)) c->sidx[(size_t)i] = tn++;
                }
                if (!tn || !stage_get(*c, (size_t)tn * 4096)) c->sidx.clear();
            }
            copy_pages(pages, n, (uint8_t*)c->map, ok.data(), c->sidx.empty() ? nullptr : c->stage.get(),
                       c->sidx.empty() ? nullptr : c->sidx.data());
            for (int64_t i = 0; i < n; i++)
                if (ok[i]) c->copied.push_back(pages[i]), c->slots.push_back((uint32_t)i);
            c->nbytes = (size_t)n * 4096;
            flush_mapped(c->alloc, c->nbytes);
        }
    }
    if (pages) host->free_pages(pages);
    {   // (what it cost, every 600 submits)
        static int64_t subs, written, copied;
        static double ms, ms_lock, ms_scan;
        static uint64_t span;
        subs++, written += n > 0 ? n : 0, copied += (int64_t)c->copied.size();
        const auto now = std::chrono::steady_clock::now();
        ms += std::chrono::duration<double, std::milli>(now - t0).count();
        ms_lock += std::chrono::duration<double, std::milli>(t_lock - t0).count();
        ms_scan += std::chrono::duration<double, std::milli>(t_scan - t_lock).count();
        if (subs % 600 == 0) {
            span = 0;
            for (const auto& r : mirrored) span += r.second - r.first;
            logf("vk: capture: a submit %.2f ms (lock %.2f, scan %.2f over %zu ranges, %llu MB), %.0f pages written, %.0f "
                 "copied (%.1f MB)", ms / 600, ms_lock / 600, ms_scan / 600, mirrored.size(),
                 (unsigned long long)(span >> 20), written / 600.0, copied / 600.0, copied * 4096.0 / 600 / 1048576);
            ms = ms_lock = ms_scan = 0, written = copied = 0;
        }
    }
    return c;
}

/// The ranges the captures take (the mirrored pools, the texture storage, the small pools): their union, sorted and
/// merged, as of now; empty while no capture was made (memory_collect_written: a page in them written after its
/// submit's capture is left for the next capture). Built again only when one of the three lists changed, as
/// capture_make keeps its own union; the reference is the calling thread's until its next call
const std::vector<std::pair<uint64_t, uint64_t>>& capture_ranges_merged() {
    static const std::vector<std::pair<uint64_t, uint64_t>> none;
    static thread_local std::vector<std::pair<uint64_t, uint64_t>> merged;
    static thread_local uint64_t ver = ~uint64_t(0);
    // (no capture is ever made without the replay thread (RX_VK_ASYNC=0): the mirrored pools' writes were left for one
    // that never came, and the pools kept their first contents: a grey world)
    if (!capture_on() || !captures_seen.load(std::memory_order_relaxed)) return none;
    std::lock_guard lk(cap_ranges_mu);
    refresh_pool_ranges_locked();   // (a pool made since the last capture: the next one takes its pages, not this)
    if (ver != cap_ranges_ver) {
        ver = cap_ranges_ver;
        std::vector<std::pair<uint64_t, uint64_t>> all = cap_ranges;
        all.insert(all.end(), tex_cap_ranges.begin(), tex_cap_ranges.end());
        all.insert(all.end(), pool_cap_ranges.begin(), pool_cap_ranges.end());
        std::sort(all.begin(), all.end());
        merged.clear();
        for (const auto& r : all) {
            if (!merged.empty() && r.first <= merged.back().second)
                merged.back().second = std::max(merged.back().second, r.second);
            else
                merged.push_back(r);
        }
    }
    return merged;
}
// (RX_VK_LOG_INHERIT) 64 bytes at the watched address: the retired mirror, guest memory and the new mirror, read back
VkBuffer iw_buf = VK_NULL_HANDLE;
VmaAllocation iw_alloc = nullptr;
void* iw_map = nullptr;
uint64_t iw_value = 0;   // (the submission whose uploads copied them: printed once it is done)
// (hookshot) see vk_internal.h. Raised where a recording gets a read of guest memory itself, at the GPU's time: a pool
// bound with no mirror (gpu_buffer_bind), a mirror's dirty pages no capture holds (mirror_sync), a mirror's first copy,
// and the pool handed out whole (gpu_buffer: a large texture unswizzled on the GPU, a texel buffer's view). With all
// pools mirrored and captured (the usual case on a card with room) it stays 0. outside: a binding made outside a
// recording's uploads belongs to no part begun yet: counted for the next one, so every part begun so far is older.
// Two reads of guest memory at the GPU's time are not noted:
//  - memory_rb_refresh's copies of the readback pages into their mirrors, in every part's uploads. They are how a write
//    of the CPU's to a page it reads back reaches the mirror at all, so a part after the readback's taking that write is
//    what they are for (and what the console's one memory does); a part that writes such a page again has it noted with
//    its own part, which the read then waits for. Noted, they would be in every part and no read would ever wait for
//    less than its whole submission. What they do carry along is whatever else shares a 4 KB page with a readback range;
//  - the shaders' own pointers into a pool that was never bound and so has no mirror (the page table's entry is the
//    pool's own address): no call of ours is made for those reads.
// Both are as they were before a value's recording became its last part's (v2.0.14 and before)
namespace {
std::atomic<uint64_t> mem_rec_part{0}, mem_in_place_part{0};
void note_in_place_read(bool outside = false) {
    const uint64_t part = mem_rec_part.load() + (outside ? 1 : 0);
    uint64_t seen = mem_in_place_part.load();
    while (seen < part && !mem_in_place_part.compare_exchange_weak(seen, part)) {}
}
} // namespace
void memory_recording_part(uint64_t part) { mem_rec_part.store(part); }
uint64_t memory_in_place_part() { return mem_in_place_part.load(); }

// ---- (hookshot) Who wrote a page the CPU reads back, as the replay records: RX_VK_READBACK_EARLY and RX_VK_LOG_RB_WRITERS ----
// (vk_internal.h) The writer the replay is recording now (its thread's: memory_writer_note) and its draws and dispatches counted.
//  - RX_VK_READBACK_EARLY: the guest ranges whose reads waited for a write-back (memory_rb_waited, at most kEarlyRanges, the
//    oldest let go first), and for each of their pages the last write in place (no mirror: the Steam Deck's every page): its
//    submission and the part it was recorded in (early_in_place_write, from note_direct_write, whatever its size). A read of
//    such a range then waits for that part (memory_in_place_writer_part), and a write to it by a draw, a dispatch, a copy or a
//    clear asks for its part to end right after it (memory_split_wanted).
//  - RX_VK_LOG_RB_WRITERS (diagnostics, through wb_pages_note's hook): each page's first and last writer in its newest
//    submission (memory_rb_writers).
namespace {
thread_local uint32_t wr_what = 0;
thread_local uint64_t wr_key = 0;
std::atomic<uint64_t> wr_work{0};
bool mirror_on();   // (below: the buffer pools' VRAM mirror, which readback_early_on's default asks)
struct PageWriters { RbWriter first, last; };
std::unordered_map<uint64_t, PageWriters> page_writers;
constexpr size_t kEarlyRanges = 64;
struct EarlyPage { uint64_t value = 0, part = 0; };
std::deque<std::pair<uint64_t, uint64_t>> early_ranges;   // (wb_pages_mu)
std::unordered_map<uint64_t, EarlyPage> early_pages;       // (wb_pages_mu)
std::atomic<bool> early_any{false}, split_wanted{false};
void rb_page_note(uint64_t lo, uint64_t hi, uint64_t value, uint64_t part) {   // (wb_pages_mu held)
    if (!rb_writers_on()) return;
    RbWriter w;
    w.value = value, w.part = part ? part : mem_rec_part.load(std::memory_order_relaxed);
    w.work = wr_work.load(std::memory_order_relaxed), w.key = wr_key, w.what = wr_what;
    if (part) w.what = kRbWriterWb, w.key = 0;   // (a mirror's write-back at its part's end: when the CPU can have it)
    if (page_writers.size() > (1u << 20)) page_writers.clear();   // (a diagnostic's memory, bounded)
    for (uint64_t p = lo >> kPageShift; p <= (hi - 1) >> kPageShift; p++) {
        PageWriters& pw = page_writers[p];
        if (pw.last.value != value) pw.first = w;
        pw.last = w;
    }
}
} // namespace
bool rb_writers_on() {
    static const bool on = env_on("RX_VK_LOG_RB_WRITERS");
    return on;
}
/// RX_VK_READBACK_EARLY (vk_replay.cpp's invalidate): a read of pages written in place waits for their writer's part. On by
/// default where the pools have no mirror (readback_early.h), =0 off.
/// Not with a parallel replay (RX_VK_PARALLEL=2 or more): its segments' recordings are numbered as parts but never put on the
/// recordings' timeline (vk_replay.cpp begin_segment_recording), so a read of a page a segment wrote would wait for a part that
/// never comes (250 ms); and the part a command buffer's segments follow had its write-back late there (ballotstomper, the team
/// room). The reads then wait for their whole submission, as without the switch
/// Nor with the render thread (readback_early.h: the rule and its numbers)
bool readback_early_on() {
    static const bool on = [] {
        readback_early::Facts f;
        f.env = getenv("RX_VK_READBACK_EARLY"), f.parallel = getenv("RX_VK_PARALLEL");
        f.render_thread = replay_render_thread_on(), f.mirrors = mirror_on();
        const readback_early::Why w = readback_early::choose(f);
        if (w == readback_early::OFF_PARALLEL)
            logf("vk: RX_VK_READBACK_EARLY left off: RX_VK_PARALLEL=%s records segments whose parts a read cannot wait for", f.parallel);
        else if (w == readback_early::OFF_RENDER_THREAD)
            logf("vk: RX_VK_READBACK_EARLY left off: the render thread is on (RX_VK_ASYNC, Threaded rendering), which it was not "
                 "measured with; reads of GPU results wait for their whole submission");
        else if (w == readback_early::ON_SWITCH)
            logf("vk: reads of GPU results written in place wait for their writer's part (RX_VK_READBACK_EARLY)");
        else if (w == readback_early::ON_DEFAULT)
            logf("vk: reads of GPU results written in place wait for their writer's part (the default without mirrors; "
                 "RX_VK_READBACK_EARLY=0: off)");
        return readback_early::on(w);
    }();
    return on;
}
void memory_writer_note(uint32_t what, uint64_t key) {
    if (!rb_writers_on() && !readback_early_on()) return;
    wr_what = what, wr_key = key;
    static const bool hooked = [] {
        if (!rb_writers_on()) return false;
        std::lock_guard lk(wb_pages_mu);
        wb_writers_hook = rb_page_note;
        return true;
    }();
    (void)hooked;
}
void memory_writer_key(uint64_t key) {
    if (rb_writers_on()) wr_key = key;
}
void memory_count_work() {
    if (rb_writers_on()) wr_work.fetch_add(1, std::memory_order_relaxed);
}
uint64_t memory_work_count() { return wr_work.load(std::memory_order_relaxed); }
/// (RX_VK_READBACK_EARLY) a GPU write in place of [lo, hi) in the submission value, recorded now: each page of it that a range
/// whose read waited reaches keeps it, whichever bytes of the page it writes (a read of the page's other bytes, or a range widened
/// over them later, must not take an earlier part of the same submission: memory_in_place_writer_part), and a write of those
/// ranges' bytes by anything but the uniform updates (written at their part's end already) asks for its part to end after it
thread_local int ec_kind = 0;   // (RX_VK_LOG_EARLY_CHECK: 2 while memory_note_gpu_write notes a bare or wide storage range)
void early_in_place_write(uint64_t lo, uint64_t hi, uint64_t value) {
    if (!readback_early_on() || hi <= lo || !early_any.load(std::memory_order_relaxed)) return;
    const uint64_t part = mem_rec_part.load(std::memory_order_relaxed);
    early_check_gpu_write(lo, hi, value, part, ec_kind ? ec_kind : hi - lo > (1u << 20) ? 1 : 0);   // (RX_VK_LOG_EARLY_CHECK)
    constexpr uint64_t kPageMask = (1ull << kPageShift) - 1;
    bool hit = false;
    {
        std::lock_guard lk(wb_pages_mu);
        for (const auto& [a, b] : early_ranges) {
            const uint64_t x = std::max(a & ~kPageMask, lo), y = std::min(((b - 1) | kPageMask) + 1, hi);
            if (x >= y) continue;
            hit = hit || std::max(a, lo) < std::min(b, hi);
            for (uint64_t pg = x >> kPageShift; pg <= (y - 1) >> kPageShift; pg++) early_pages[pg] = {value, part};
        }
    }
    if (hit && wr_what != kRbWriterUb) split_wanted.store(true, std::memory_order_relaxed);
}
uint64_t memory_in_place_writer_part(uint64_t guest, uint64_t size, uint64_t whole) {
    if (!size || !whole || !readback_early_on()) return 0;
    std::lock_guard lk(wb_pages_mu);
    uint64_t part = 0;
    // (every page of the range that whole's submission wrote in place, as memory_writeback_value found it, must have that write
    // seen here: one written before its range's read waited (a range widened since, or a page tracked later) has no part known,
    // and the read waits for the whole submission, as before. The pages an older submission wrote are done with any part of
    // this one: the recordings' timeline is in order. A mirror's pages: the caller's part)
    // (a write over 1 MiB is noted in early_pages but not in wb_pages (note_direct_write): its page's early entry counts even where
    // wb_pages names an older submission, or a later part of the same submission is missed. zeldaootr's review of
    // https://git.notrexed.dev/Hookshot/Hookshot/pulls/303, finding 1)
    for (uint64_t pg = guest >> kPageShift; pg <= (guest + size - 1) >> kPageShift; pg++) {
        const auto it = early_pages.find(pg);
        if (it != early_pages.end() && it->second.value == whole) { part = std::max(part, it->second.part); continue; }
        if (wb_pages.get(pg) == whole && !wb_parts.get(pg)) return 0;
    }
    return part;
}
void memory_rb_waited(uint64_t guest, uint64_t size) {
    if (!size || !readback_early_on()) return;
    const uint64_t lo = guest, hi = guest + size;
    std::lock_guard lk(wb_pages_mu);
    for (const auto& [a, b] : early_ranges)
        if (a <= lo && hi <= b) return;
    early_ranges.push_back({lo, hi});
    if (early_ranges.size() > kEarlyRanges) {
        const auto [a, b] = early_ranges.front();
        early_ranges.pop_front();
        for (uint64_t pg = a >> kPageShift; pg <= (b - 1) >> kPageShift; pg++) {
            bool kept = false;
            for (const auto& [c, d] : early_ranges) kept = kept || (pg >= (c >> kPageShift) && pg <= ((d - 1) >> kPageShift));
            if (!kept) early_pages.erase(pg);
        }
    }
    early_any.store(true, std::memory_order_relaxed);
}
bool memory_rb_waited_range(uint64_t guest, uint64_t size) {
    if (!size || !readback_early_on() || !early_any.load(std::memory_order_relaxed)) return false;
    std::lock_guard lk(wb_pages_mu);
    for (const auto& [a, b] : early_ranges)
        if (a < guest + size && guest < b) return true;
    return false;
}
bool memory_split_wanted() {   // (asked at every op the replay records: a load, the exchange only when one was asked for)
    return split_wanted.load(std::memory_order_relaxed) && split_wanted.exchange(false, std::memory_order_relaxed);
}
bool memory_readback_early_active() { return early_any.load(std::memory_order_relaxed); }
/// (RX_VK_LOG_RB_WRITERS) the writers of a range's pages in the newest submission any of them was written in: the earliest first
/// writer and the latest last one (by part, then by draw count), and how many of its pages that submission wrote
bool memory_rb_writers(uint64_t guest, uint64_t size, RbWriter& first, RbWriter& last, int* pages) {
    first = last = RbWriter{};
    if (pages) *pages = 0;
    if (!size || !rb_writers_on()) return false;
    std::lock_guard lk(wb_pages_mu);
    const uint64_t lo = guest >> kPageShift, hi = (guest + size - 1) >> kPageShift;
    uint64_t newest = 0;
    for (uint64_t p = lo; p <= hi; p++)
        if (auto it = page_writers.find(p); it != page_writers.end()) newest = std::max(newest, it->second.last.value);
    if (!newest) return false;
    const auto later = [](const RbWriter& a, const RbWriter& b) { return a.part != b.part ? a.part > b.part : a.work > b.work; };
    bool any = false;
    for (uint64_t p = lo; p <= hi; p++) {
        auto it = page_writers.find(p);
        if (it == page_writers.end() || it->second.last.value != newest) continue;
        if (pages) (*pages)++;
        if (!any || later(first, it->second.first)) first = it->second.first;
        if (!any || later(it->second.last, last)) last = it->second.last;
        any = true;
    }
    return any;
}
void memory_recording_value(uint64_t value, uint64_t completed) {
    mem_rec_value = value;
    if (completed > mem_completed) mem_completed = completed;
}
struct LateShadow {
    uint32_t mid = 0;                   // (the mirror whose page holds these bytes; 0: none, void)
    uint64_t used = 0;                  // (gen_now at the last sync: the oldest go past kLateShadowMax)
    std::unique_ptr<uint8_t[]> bytes;   // (4 KB)
};
constexpr size_t kLateShadowMax = 4096;   // (16 MB)
std::unordered_map<uint32_t, LateShadow> late_shadows;   // (guest page -> its shadow; pools_mu)
std::atomic<int64_t> st_late_merged{0}, st_late_merged_bytes{0}, st_late_waited{0}, st_late_no_shadow{0}, st_late_seg{0};
/// Whether the page's shadow holds what its pool's mirror has for the page now (pools_mu held)
std::atomic<int64_t> st_late_why[4];   // (why a late page had no shadow: no entry, bytes not yet taken, void, another mirror)

std::mutex late_fix_mu;
std::atomic<uint64_t> late_fix_top{0};   // (the highest value a pending fix waits for; 0: none pending)
constexpr int kLateFixMax = 4;           // (pending at once; a take past it waits as before)
std::atomic<uint32_t> late_fix_gens[kLateFixMax];   // (the pending fixes' generations, 0: free: the readers' quick test)
std::atomic<int64_t> st_late_deferred{0}, st_late_def_pages{0}, st_late_def_tex{0}, st_late_def_unsent{0};
uint64_t memory_late_fix_pending() { return late_fix_top.load(std::memory_order_acquire); }
std::atomic<int64_t> st_late_def_lower{0};   // (fixes registered with a value below a pending one's)
std::atomic<int64_t> st_late_tex_waits{0};   // (texture storage reads that waited for a pending late page's value)
bool memory_is_rb_page_locked(uint32_t page);
/// (hookshot, RX_VK_FRAME_TRACE) The time the replay has waited for the GPU in take_capture, for the pages a submission still
/// on the GPU writes back: part of the trace's memsync_ms, and the part that is a wait. Frames of 36-201 ms of memory sync in a
/// played session could not be told from a slow copy
std::atomic<int64_t> st_late_wait_ns{0};
int64_t memory_late_wait_ns() { return st_late_wait_ns.load(std::memory_order_relaxed); }
// ---- the late pages' fix (hookshot, 2026-11-13) ----
// A capture's page that a submission still on the GPU writes back is late (capture_make). The replay waited for that
// submission and copied those pages again from guest memory (RX_VK_CAPTURE_WAIT=2): ~4.8 ms of a 17.8 ms frame at Lon Lon
// Ranch by day (an i7-14700K and an RTX 4070 SUPER: ~190 late pages a capture, the wait most of a frame's memory sync), for
// what was almost always nothing: across 2,400 captures one page changed while they waited (~1 word: GPU events), and 32
// others once each. Now the late pages are copied again with no wait, and a small pass (shaders/late_fix.comp), recorded in
// the uploads before anything copies out of the capture's buffer on the GPU, writes into that copy the words the write-backs
// still running wrote to guest memory: the wait's copy, word by word. Which words: the diff pass's stamps (a uint a word, the
// submission's value, in 64 KB blocks made where late pages are found, 64 MB at most, counting for one shadow and for the
// write-backs recorded after the block was filled); their values: the pool's write-back shadow, which holds what each
// write-back wrote. (Not the mirror: the GPU writes some of it through pointers no written binding covers, words no
// write-back ever puts in guest memory: a newer float, or scratch.) Every reader of the buffer on the GPU (mirror_sync,
// mirror_make) then reads what the wait gave. The other readers: the texture shadow forgets a fixed page (textures read
// it from guest memory, as any page with no copy); a read of one on the CPU (memory_read_as_submitted: none measured) waits
// and copies it again, as before, and its fix is dropped. Pages written back whole (a pool with no shadow) take their
// words from a 4 KB snapshot each whole copy also writes (from the second time the page is late); a pool of 16 MB or less
// is given a shadow instead. A capture waits as before when any of its late pages was written in place by the GPU, has no
// stamps or snapshot yet (the first time it is found), or is late for an earlier capture whose fix is not recorded yet.
// RX_VK_LATE_FIX=0: the wait, as before
namespace {
bool late_fix_on() {   // (not with the parallel replay: its segments' workers record copies out of the captures in any order)
    static const bool on = [] {   // (on unless an off value: env_value_off)
        const char* e = getenv("RX_VK_LATE_FIX");
        return (!e || !env_value_off(e)) && !(getenv("RX_VK_PARALLEL") && atoi(getenv("RX_VK_PARALLEL")) > 1);
    }();
    return on;
}
} // namespace
// (hookshot) The late pages' fix (see take_capture) needs to know what wrote a late page back: the diff pass (its stamps say
// which words) or a whole copy (no record of which words: the capture waits). A page late for another write-back (the GPU's
// writes in place, note_direct_write) has neither, and its capture waits, as before. wb_pages_mu
WbPages wb_diff_pages, wb_whole_pages;
enum WbKind { WB_DIFF, WB_WHOLE };
void wb_kind_note(WbKind kind, uint64_t guest_lo, uint64_t guest_hi, uint64_t value) {
    if (!value || guest_hi <= guest_lo || !late_fix_on()) return;
    std::lock_guard lk(wb_pages_mu);
    WbPages& pages = kind == WB_DIFF ? wb_diff_pages : wb_whole_pages;
    for (uint64_t pg = guest_lo >> kPageShift; pg <= (guest_hi - 1) >> kPageShift; pg++) pages.set(pg, value);
}
// (whole copies' page snapshots: a page late for a whole copy gets a 4 KB slot; each whole copy over it from then on also
// copies its part of the page there, in the same command buffer, and its range is kept with the copy's value. pools_mu)
// When the 256 slots are taken, the slots of pages not late for a whole copy in kSnapIdle recordings, with nothing on the GPU
// still writing or reading them, are given back (snap_pages_reclaim); with none such, a new page waits as before
constexpr uint32_t kSnapSlots = 256;
constexpr uint64_t kSnapIdle = 1024;   // (recordings)
// (a small pool, 16 MB or less, whose late page was written back whole: given a shadow at its next write-back, so its pages
// take the diff pass's exact stamps; a whole copy writes words the program never wrote too, older than the CPU's. A pool
// gone leaves the set (late_fix_reap): a new pool at its address starts over. pools_mu)
constexpr uint64_t kLateShadowPoolMax = 16ull << 20;
std::unordered_set<uint64_t> late_shadow_pools;
bool late_shadow_wanted(uint64_t pool_lo) { return !late_shadow_pools.empty() && late_shadow_pools.count(pool_lo); }
struct SnapPage {
    uint32_t slot = 0;
    uint64_t since = 0;   // (the first recording whose whole copies are snapped)
    uint64_t used = 0;    // (the last recording that writes its slot (a whole copy) or reads it (a fix): given back only once done)
    uint64_t late = 0;    // (the recording when a capture last found it late for a whole copy)
    std::vector<std::array<uint64_t, 3>> ranges;   // (lo, hi in the page, the copy's value)
};
std::unordered_map<uint32_t, SnapPage> snap_pages;
std::vector<uint32_t> snap_free;   // (slots given back: snap_pages_reclaim)
VkBuffer snap_buf = VK_NULL_HANDLE;
VmaAllocation snap_alloc = nullptr;
VkDeviceAddress snap_address = 0;
uint32_t snap_used = 0;
void snap_pages_reclaim();
bool snap_page_make(uint32_t pg) {
    if (snap_pages.count(pg)) return false;
    if (snap_used >= kSnapSlots && snap_free.empty()) snap_pages_reclaim();
    if (snap_used >= kSnapSlots && snap_free.empty()) return false;
    if (!snap_buf) {
        if (!create_device_buffer((uint64_t)kSnapSlots * 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, snap_buf, snap_alloc)) {
            snap_buf = VK_NULL_HANDLE, snap_alloc = nullptr;
            return false;
        }
        VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        da.buffer = snap_buf;
        snap_address = vkGetBufferDeviceAddress(dev.device, &da);
    }
    SnapPage s;
    if (!snap_free.empty()) s.slot = snap_free.back(), snap_free.pop_back();   // (no range of it kept: the old page's words are never read)
    else s.slot = snap_used++;
    s.since = mem_rec_value + 1, s.late = mem_rec_value;
    snap_pages.emplace(pg, std::move(s));
    return true;
}
// (pools_mu held) a whole copy of [off, off + size) of the pool, just recorded into cmd: the snapped pages' parts copied too
void late_snap_whole(VkCommandBuffer cmd, const Pool& p, uint64_t off, uint64_t size) {
    if (snap_pages.empty() || !size || !p.mirror || !snap_buf) return;
    const uint64_t lo = p.lo + off, hi = lo + size;
    for (uint64_t pg = lo >> kPageShift; pg <= (hi - 1) >> kPageShift; pg++) {
        const auto it = snap_pages.find((uint32_t)pg);
        if (it == snap_pages.end() || mem_rec_value < it->second.since) continue;
        const uint64_t a = std::max(lo, pg << kPageShift), b = std::min(hi, (pg + 1) << kPageShift);
        const VkBufferCopy c{a - p.lo, (uint64_t)it->second.slot * 4096 + (a - (pg << kPageShift)), b - a};
        vkCmdCopyBuffer(cmd, p.mirror, snap_buf, 1, &c);
        it->second.used = std::max(it->second.used, mem_rec_value);
        std::vector<std::array<uint64_t, 3>>& rs = it->second.ranges;   // (the done ones go: else one a whole copy, for good)
        if (rs.size() >= 32) std::erase_if(rs, [](const std::array<uint64_t, 3>& r) { return r[2] <= mem_completed; });
        rs.push_back({a - (pg << kPageShift), b - (pg << kPageShift), mem_rec_value});
    }
}
// (hookshot) The shadow follows the mirror's sync: a mirror sync of a pool with a shadow copies the same bytes (a capture's
// page, or guest memory as the GPU gets there) into the shadow, and the next write-back counts those pages as known again
// after forgetting the pages the CPU wrote. Before, such a page was filled again from guest memory at the write-back, with
// the CPU's writes made after the capture: where the GPU wrote nothing the mirror still held the capture's words, which
// differed, and the diff wrote them over the CPU's newer ones (the per-character animation program's output: 0x4800-byte
// bindings written whole by their descriptor, padding the program never stores holding 0x6f6f6f6f, put back over floats the
// CPU had written, every frame). Now the diff writes only what the GPU changed since the sync.
// (hookshot) Always-on counts for the late pages' summary line, one for each way a late page's words or a CPU read of them go
// (plain counters, pools_mu held; read and reset with the 600-recording line): reads on the CPU by branch, pages fixed by kind,
// pages a mirror sync copied into a shadow, made known by it, and the ones left unknown because the replay or the collection
// wrote them since (shadow_forgotten_last, set by the write-back's forget: pages the capture did not take)
struct LateStats {
    int64_t rd_fixed = 0, rd_earlier_rec = 0, rd_not_recorded = 0, rd_this_part = 0, fix_diff_pages = 0, fix_whole_pages = 0,
            sync_pages = 0, sync_known = 0, sync_left_unknown = 0;
} st_lfx;
std::vector<uint32_t> shadow_forgotten_last;   // (as the forget's lists gave them, in no order: shadow_synced_known walks it)
bool shadow_forgotten_all = false;   // (the forget's lists overflowed: no page synced since is known to equal guest memory)
struct ShadowSynced { uint64_t lo, page; VkBuffer shadow; };
std::vector<ShadowSynced> shadow_synced;   // (synced since the last write-back's forget, into that shadow; pools_mu)
void shadow_sync_copy(VkCommandBuffer up, VkBuffer src, Pool& p, const std::vector<VkBufferCopy>& rs) {
    if (!p.shadow || rs.empty()) return;
    if (shadow_synced.size() > (1u << 20)) {   // (no write-back for very long: those pages are filled from guest memory, as before)
        shadow_synced.clear();
        return;
    }
    vkCmdCopyBuffer(up, src, p.shadow, (uint32_t)rs.size(), rs.data());
    for (const VkBufferCopy& r : rs)
        for (uint64_t pg = r.dstOffset >> kPageShift; r.size && pg <= (r.dstOffset + r.size - 1) >> kPageShift; pg++)
            shadow_synced.push_back({p.lo, pg, p.shadow}), st_lfx.sync_pages++;
}
// (after the write-back's forget: the pages synced since, known as the sync left them. Not after a forget whose lists overflowed
// (shadow_written_all: the pages written are not known, so none of the synced ones is), and not the ones the forget has just cleared
// because the replay's own writes or the collection reported them since the capture: guest memory has newer words there than the
// shadow, which holds the capture's. Known, they were not read into the shadow again at the write-back, and the diff pass wrote
// the mirror's words of a span whole over guest memory's newer ones, which the late pages' fix then read as stamped words.
// Left unknown they are read from guest memory at the next write-back that writes the page back)
void shadow_synced_known() {
    if (shadow_synced.empty()) return;
    // (which of the synced pages the forget's list holds. That list is in the order its pages were reported, thousands of
    // them at a write-back, and was sorted at every part's end only to be searched here for these few: with the forget's own
    // sorts, 0.36 ms a frame of the render thread in Castle Town's market. Now the synced pages are the ones sorted, and the
    // list is walked once: each page is asked of a filter of the synced pages' low 12 bits, then searched for among them)
    static thread_local std::vector<uint32_t> want;
    static thread_local std::vector<uint8_t> late;
    const auto page_of = [](const ShadowSynced& s) { return (uint32_t)((s.lo >> kPageShift) + s.page); };
    want.clear();
    for (const ShadowSynced& s : shadow_synced) want.push_back(page_of(s));
    std::sort(want.begin(), want.end());
    want.erase(std::unique(want.begin(), want.end()), want.end());
    late.assign(want.size(), 0);
    if (!shadow_forgotten_all && !shadow_forgotten_last.empty()) {
        uint64_t filter[64]{};
        for (const uint32_t w : want) filter[(w & 4095) >> 6] |= 1ull << (w & 63);
        for (const uint32_t f : shadow_forgotten_last) {
            if (!(filter[(f & 4095) >> 6] >> (f & 63) & 1)) continue;
            const auto at = std::lower_bound(want.begin(), want.end(), f);
            if (at != want.end() && *at == f) late[(size_t)(at - want.begin())] = 1;
        }
    }
    for (const ShadowSynced& s : shadow_synced) {   // (a shadow made since has not had the copy: shadow_synced_freed)
        const auto it = imported.find(s.lo);
        if (it == imported.end() || !it->second.shadow || it->second.shadow != s.shadow) continue;
        std::vector<uint64_t>& known = it->second.shadow_known;
        if ((s.page >> 6) >= known.size()) continue;
        if (shadow_forgotten_all || late[(size_t)(std::lower_bound(want.begin(), want.end(), page_of(s)) - want.begin())]) {
            st_lfx.sync_left_unknown++;
            continue;
        }
        known[s.page >> 6] |= 1ull << (s.page & 63);
        st_lfx.sync_known++;
    }
    shadow_synced.clear();
}
constexpr unsigned kStampShift = 16;   // (a block: 64 KB of guest memory, 64 KB of stamps)
constexpr uint64_t kStampBlock = 1ull << kStampShift, kStampMask = kStampBlock - 1, kStampMax = 64ull << 20;
struct StampBlock {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkDeviceAddress address = 0;
    uint64_t pool = 0;                  // (the pool's lo: the stamps say which guest words a write-back of it wrote)
    VkBuffer shadow = VK_NULL_HANDLE;   // (the shadow they are for: the fixes read the words there)
    uint64_t since = 0;                 // (the first recording whose write-backs stamp it; 0 until filled)
};
std::unordered_map<uint64_t, StampBlock> stamp_blocks;   // (guest >> kStampShift; pools_mu)
std::unordered_map<uint64_t, int> stamp_pools;           // (a pool's lo -> its blocks)
std::vector<uint64_t> stamp_unfilled;                    // (made: filled first in the next recording)
struct DeadStamp { VkBuffer buf; VmaAllocation alloc; uint64_t value; };
std::deque<DeadStamp> dead_stamps;
uint64_t stamp_bytes = 0;
bool stamps_in(const Pool& p) {
    const auto it = stamp_pools.find(p.lo);
    return it != stamp_pools.end() && it->second > 0;
}
uint64_t stamp_address(const Pool& p, uint64_t guest) {   // (0: none, or not filled yet)
    const auto it = stamp_blocks.find(guest >> kStampShift);
    if (it == stamp_blocks.end() || !it->second.since || it->second.pool != p.lo || it->second.shadow != p.shadow) return 0;
    return it->second.address + (guest & kStampMask);
}
// (pools_mu held) the blocks of a pool gone, or whose shadow is gone (its write-backs are whole copies then: no stamps),
// let go once the recordings so far are done (a write-back recorded before may still stamp them)
void stamps_drop_unused() {
    for (auto it = stamp_blocks.begin(); it != stamp_blocks.end();) {
        const auto pi = imported.find(it->second.pool);
        if (pi != imported.end() && pi->second.shadow) { ++it; continue; }
        dead_stamps.push_back({it->second.buf, it->second.alloc, mem_rec_value});
        stamp_bytes -= kStampBlock;
        if (--stamp_pools[it->second.pool] <= 0) stamp_pools.erase(it->second.pool);
        it = stamp_blocks.erase(it);
    }
}
void stamp_block_make(Pool& p, uint64_t guest) {
    const uint64_t key = guest >> kStampShift;
    if (!p.shadow || stamp_blocks.count(key) || stamp_bytes + kStampBlock > kStampMax) return;
    StampBlock b;
    if (!create_device_buffer(kStampBlock, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, b.buf, b.alloc))
        return;
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = b.buf;
    b.address = vkGetBufferDeviceAddress(dev.device, &da);
    b.pool = p.lo, b.shadow = p.shadow;
    stamp_blocks.emplace(key, b);
    stamp_pools[p.lo]++;
    stamp_bytes += kStampBlock;
    stamp_unfilled.push_back(key);
    static int logged;
    if (logged++ < 8 || stamp_blocks.size() % 64 == 0)
        logf("vk: late pages' fix: stamps for %#llx (pool %#llx; %zu blocks, %llu KB)", (unsigned long long)(key << kStampShift),
             (unsigned long long)p.lo, stamp_blocks.size(), (unsigned long long)(stamp_bytes >> 10));
}
// (hookshot, RX_VK_LATE_STAMPS_AHEAD, on; =0: as before) A late page with no stamp block yet makes its capture wait (the first
// time), and with it the blocks of the rest of its pool (all of a pool of 16 MB or less, else the 2 MB around the page), not only
// the late pages' own: arriving in an area, the game writes 100-190 pages a capture of a new 4-8 MB pool the GPU writes back too,
// each capture's late pages in other 64 KB blocks, and each new block cost a wait (4 cores: 6-20 waits an arrival, 20-250 ms each
// while the GPU works through the arrival). The blocks are the same ones a later late page would make, sooner; the fixes' rules
// (since, the shadow, the pool) are unchanged. Within kStampMax, as any block. tests/test_late_stamps_ahead.py
bool late_stamps_ahead_on() {
    static const bool on = [] {
        const bool v = env_on_or("RX_VK_LATE_STAMPS_AHEAD", 1);
        if (!v) logf("vk: late pages' fix: a late page's missing stamps made for its own block only, as before (RX_VK_LATE_STAMPS_AHEAD=0)");
        return v;
    }();
    return on;
}
void stamp_blocks_around(Pool& p, uint64_t guest) {   // (pools_mu held)
    constexpr uint64_t kAround = 2ull << 20;   // (a pool past 16 MB: the 2 MB around the page)
    const uint64_t base = guest & ~(kAround - 1);
    uint64_t lo = p.lo, hi = p.hi;
    if (hi - lo > (16ull << 20)) lo = std::max(lo, base), hi = std::min(hi, base + kAround);
    for (uint64_t b = (lo + kStampMask) & ~kStampMask; b + kStampBlock <= hi; b += kStampBlock) stamp_block_make(p, b);
}
struct LateFix { uint64_t dst, src, stamps; uint32_t count, after; };   // (shaders/late_fix.comp's Fix)
static_assert(sizeof(LateFix) == 32, "late_fix.comp's Fix");
std::vector<LateFix> late_fixes;   // (of the captures taken, for the next recording; pools_mu)
struct LatePage {   // (a page with fixes: until they are done, a read of it on the CPU waits as before)
    uint64_t pool = 0, stamp_key = UINT64_MAX;   // (what its fixes read: the pool's shadow, its stamp block)
    VkBuffer shadow = VK_NULL_HANDLE, stamp_buf = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    uint8_t* slot = nullptr;
    uint64_t nbytes = 0, late_value = 0, fix_value = 0;   // (fix_value: the recording that fixes it; 0: not recorded yet)
    uint64_t cap_value = 0;                                 // (its capture's value: the buffer kept under it, cap_bufs)
    size_t first = 0, n = 0;                                // (its fixes in late_fixes, then in fix_map)
    LateFix* fix_map = nullptr;
    VmaAllocation fix_alloc = nullptr;                      // (fix_map's table)
    uint64_t fix_part = 0, fix_end = 0;                     // (fix_part: the recording part it was recorded in; fix_end: the table's bytes up to its last fix)
};
std::unordered_map<uint32_t, LatePage> late_pages;
// (pools_mu held; the slots all taken) the snapshots of pages not late for a whole copy in kSnapIdle recordings given back:
// none whose slot a recording not done yet writes or reads (used: its whole copies, and the fixes late_fix_emit recorded), and
// none with a late page whose fix is not recorded yet or not done (late_fixes still to be recorded read its slot)
void snap_pages_reclaim() {
    for (auto it = snap_pages.begin(); it != snap_pages.end();) {
        const SnapPage& s = it->second;
        const auto lp = late_pages.find(it->first);
        const bool fixing = lp != late_pages.end() && (!lp->second.fix_value || lp->second.fix_value > mem_completed);
        if (fixing || s.used > mem_completed || s.late + kSnapIdle > mem_rec_value) {
            ++it;
            continue;
        }
        snap_free.push_back(s.slot);
        it = snap_pages.erase(it);
    }
}
std::vector<uint32_t> late_built;   // (the pages of the last late_fix_build, until late_fix_kept)
size_t late_built_mark = 0;
struct LateFixTable { VkBuffer buf = VK_NULL_HANDLE; VmaAllocation alloc = nullptr; uint8_t* map = nullptr; VkDeviceAddress address = 0; uint64_t bytes = 0, value = 0; };
LateFixTable late_fix_tables[16];
int late_fix_next = 0;
struct { bool tried = false; VkPipelineLayout layout = VK_NULL_HANDLE; VkPipeline pipe = VK_NULL_HANDLE; } late_fix_pipe;
int64_t st_lf_takes, st_lf_fixed, st_lf_fell[8], st_lf_fixes, st_lf_pages, st_lf_cpu_reads, st_lf_no_table, st_lf_moved, st_lf_part_waits, st_lf_part_fell;
bool late_fix_pipe_ready() {
    if (late_fix_pipe.tried) return late_fix_pipe.pipe != VK_NULL_HANDLE;
    late_fix_pipe.tried = true;
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(dev.device, &li, nullptr, &late_fix_pipe.layout) == VK_SUCCESS) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        const auto spv = BUILTIN_SPV(LATE_FIX_COMP_SPV);
        mi.codeSize = spv.size;
        mi.pCode = spv.code;
        VkShaderModule m = VK_NULL_HANDLE;
        IntelDriverGate gate;   // (as wbd_pipe_ready's: Intel's compiler on Windows, one thread at a time, and the fault guard)
        if (extern_guarded_shader_module(&mi, &m) == VK_SUCCESS) {
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = m;
            ci.stage.pName = "main";
            ci.layout = late_fix_pipe.layout;
            if (extern_guarded_compute(VK_NULL_HANDLE, &ci, &late_fix_pipe.pipe) != VK_SUCCESS)
                late_fix_pipe.pipe = VK_NULL_HANDLE;
            vkDestroyShaderModule(dev.device, m, nullptr);
        }
    }
    const bool ok = late_fix_pipe.pipe != VK_NULL_HANDLE;
    logf("vk: late pages' fix %s", ok ? "on (RX_VK_LATE_FIX=0: the wait, as before)" : "off: its pipeline was not made");
    return ok;
}
LateFixTable* late_fix_table_get(uint64_t bytes) {   // (a ring: one is taken again once the recording that read it is done)
    LateFixTable& t = late_fix_tables[late_fix_next];
    if (t.value > mem_completed) return nullptr;
    if (t.bytes < bytes) {
        if (t.buf) destroy_buffer(t.buf, t.alloc);
        t = {};
        const uint64_t n = std::max<uint64_t>(64u << 10, (bytes + 65535) & ~65535ull);
        void* map = nullptr;
        if (!create_buffer(n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true, t.buf, t.alloc, &map) || !map) {
            if (t.buf) destroy_buffer(t.buf, t.alloc);
            t = {};
            return nullptr;
        }
        VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        da.buffer = t.buf;
        t.address = vkGetBufferDeviceAddress(dev.device, &da);
        t.map = (uint8_t*)map, t.bytes = n;
    }
    late_fix_next = (late_fix_next + 1) % 16;
    t.value = mem_rec_value;
    return &t;
}
// (pools_mu held) the fixes of a capture's late pages, or false: it waits, as before
bool late_fix_build(Capture& c) {
#ifdef TRACY_ENABLE
    HT_SCOPE("late_fix_build");
#endif
    st_lf_takes++;
    if (!late_fix_pipe_ready() || !c.buffer) return false;
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = c.buffer;
    const VkDeviceAddress cap = vkGetBufferDeviceAddress(dev.device, &da);
    // (the GPU's progress now, not at the capture: the late pages are copied again below from guest memory, which has every
    // write-back done by now, as the wait's copy had; the fixes are for the ones not done yet)
    const uint64_t D = std::max(c.late_done, vk_timeline_completed());
    late_built_mark = late_fixes.size();
    late_built.clear();
    static thread_local std::vector<uint8_t> copied;
    copied.assign(c.pages.size(), 0);
    for (uint32_t s : c.slots) copied[s] = 1;
    static thread_local std::vector<std::pair<uint32_t, LatePage>> pages;
    pages.clear();
    int why = cap ? -1 : 0;
    const auto fail = [&](int k) { if (why < 0) why = k; };
    {
        std::lock_guard wl(wb_pages_mu);
        Pool* p = nullptr;
        for (uint32_t i : c.late) {
            if (why >= 0 && why != 2 && why != 3 && why != 4 && why != 5) break;   // (no stamps or shadow yet: the rest looked at, to make them)
            if (!copied[i]) continue;
            const uint64_t pg = c.pages[i], a = pg << kPageShift;
            if (const auto lp = late_pages.find((uint32_t)pg); lp != late_pages.end() && !lp->second.fix_value) {
                fail(2);   // (an earlier capture's fix of it not recorded yet: one entry a page, so this one waits)
                continue;
            }
            if (!p || a < p->lo || a >= p->hi) {
                p = nullptr;
                auto it = imported.upper_bound(a);
                if (it != imported.begin() && a < std::prev(it)->second.hi) p = &std::prev(it)->second;
            }
            if (!p || (p->hi - a) & 3) { fail(1); continue; }
            const uint64_t end = std::min(a + 4096, p->hi);
            const uint64_t dst = cap + (uint64_t)i * 4096, src = p->shadow_address + (a - p->lo);   // (src: the diff fixes')
            const size_t first = late_fixes.size();
            bool any = false;
            uint64_t stamp_key = UINT64_MAX;
            VkBuffer stamp_buf = VK_NULL_HANDLE;
            if (wb_direct.get(pg) > D) { fail(6); continue; }   // (written in place by a submission not done: which words, unknown)
            const bool whole = wb_whole_pages.get(pg) > D, diffed = wb_diff_pages.get(pg) > D;
            if (whole && diffed) { fail(5); continue; }   // (both pending: the fix's workgroups have no order between them)
            if (whole && p->hi - p->lo <= kLateShadowPoolMax) late_shadow_pools.insert(p->lo);   // (its next write-backs: the diff pass)
            if (whole) {   // (from its snapshot: the pending copies' ranges, as they wrote them)
                const auto sit = snap_pages.find((uint32_t)pg);
                if (sit == snap_pages.end()) {
                    snap_page_make((uint32_t)pg);
                    fail(3);
                    continue;
                }
                SnapPage& s = sit->second;
                s.late = mem_rec_value;   // (kept from snap_pages_reclaim for kSnapIdle recordings, and its fixes with it)
                std::erase_if(s.ranges, [](const std::array<uint64_t, 3>& r) { return r[2] <= mem_completed; });
                if (D + 1 < s.since) { fail(4); continue; }
                bool odd = false;
                for (const auto& r : s.ranges) {
                    if (r[2] <= D) continue;
                    if ((r[0] | r[1]) & 3 || a + r[1] > end) { odd = true; break; }
                    late_fixes.push_back({dst + r[0], snap_address + (uint64_t)s.slot * 4096 + r[0], 0, (uint32_t)((r[1] - r[0]) / 4), 0});
                    any = true;
                }
                if (odd || !any) { late_fixes.resize(first); fail(5); continue; }
            }
            if (diffed) {
                if (!p->shadow || !p->shadow_address) { fail(3); continue; }
                const auto it = stamp_blocks.find(a >> kStampShift);
                if (it == stamp_blocks.end()) {
                    stamp_block_make(*p, a);
                    if (late_stamps_ahead_on()) stamp_blocks_around(*p, a);
                    fail(3);
                    continue;
                }
                if (it->second.pool != p->lo || it->second.shadow != p->shadow) {   // (another shadow since: filled again, counts from then)
                    if (it->second.pool != p->lo) {   // (another pool's: its write-backs stamp it from now on, stamps_in)
                        if (--stamp_pools[it->second.pool] <= 0) stamp_pools.erase(it->second.pool);
                        stamp_pools[p->lo]++;
                    }
                    it->second.pool = p->lo, it->second.shadow = p->shadow, it->second.since = 0;
                    stamp_unfilled.push_back(it->first);
                    fail(3);
                    continue;
                }
                if (!it->second.since || D + 1 < it->second.since) { fail(4); continue; }
                late_fixes.push_back({dst, src, it->second.address + (a & kStampMask), (uint32_t)((end - a) / 4), (uint32_t)D});
                stamp_key = it->first, stamp_buf = it->second.buf;
                any = true;
            }
            if (!any && wb_pages.get(pg) <= D) continue;   // (its write-backs done by now: the copy below is all it needs)
            if (!any) {   // (late, with no write-back of these kinds pending: unknown, so as before)
                static int told;
                if (told++ < 12)
                    logf("vk: late pages' fix: page %#llx late with no write-back of a known kind: written back %llu, the capture saw "
                         "%llu done, diff %llu, whole %llu (pool %#llx%s)", (unsigned long long)a,
                         (unsigned long long)wb_pages.get(pg), (unsigned long long)D, (unsigned long long)wb_diff_pages.get(pg),
                         (unsigned long long)wb_whole_pages.get(pg),
                         (unsigned long long)p->lo, p->shadow ? ", shadow" : "");
                fail(6);
                continue;
            }
            LatePage lp;
            lp.buffer = c.buffer, lp.alloc = c.alloc, lp.slot = (uint8_t*)c.map + (uint64_t)i * 4096, lp.nbytes = c.nbytes;
            lp.late_value = c.late_value, lp.first = first, lp.n = late_fixes.size() - first;
            lp.pool = p->lo, lp.shadow = p->shadow, lp.stamp_key = stamp_key, lp.stamp_buf = stamp_buf;
            pages.push_back({(uint32_t)pg, lp});
        }
    }
    if (why >= 0) {
        late_fixes.resize(late_built_mark);
        st_lf_fell[why]++;
        return false;
    }
    {   // (every late page copied again, with no wait: the write-backs done by D are in it, the fixes give the rest)
        uint8_t* const base = (uint8_t*)host->guest_base();
        for (uint32_t i : c.late)
            if (copied[i]) guarded_copy((uint8_t*)c.map + (uint64_t)i * 4096, base + ((uint64_t)c.pages[i] << kPageShift), 4096);
        flush_mapped(c.alloc, c.nbytes);
    }
    for (auto& [pg, lp] : pages) {
        late_pages[pg] = lp, late_built.push_back(pg);
        (lp.stamp_key != UINT64_MAX ? st_lfx.fix_diff_pages : st_lfx.fix_whole_pages)++;
    }
    st_lf_fixed++, st_lf_pages += (int64_t)pages.size(), st_lf_fixes += (int64_t)(late_fixes.size() - late_built_mark);
    return true;
}
void late_fix_reap();
void late_fix_shadows_freed();   // (defined with the write-back's diff, where its freed shadows are known)
bool late_fix_take(Capture& c) {   // (take_capture; pools_mu not held)
    if (!late_fix_on()) return false;
    std::lock_guard lk(pools_mu);
    late_fix_shadows_freed();
    late_fix_reap();
    return late_fix_build(c);
}
// (pools_mu held) the last build's buffer kept with the captures, or not: then nothing reads it and its fixes go
void late_fix_kept(bool kept, uint64_t value) {
    if (kept) {
        for (uint32_t pg : late_built)
            if (const auto it = late_pages.find(pg); it != late_pages.end()) it->second.cap_value = value;
        return;
    }
    late_fixes.resize(late_built_mark);
    for (uint32_t pg : late_built) late_pages.erase(pg);
    late_built.clear();
}
// (pools_mu held) a read on the CPU of a page in a capture's buffer whose late page has a fix: it reads what the GPU's copies
// out of the buffer read. Fixed already: the buffer as the fix left it. Recorded in a submission before this one: that one
// waited for, then the same. Recorded in an earlier part of this submission (parts go to the GPU while the later ones are
// recorded: that part's fix may be running, or run after this read): that part waited for (it is submitted, so the wait
// cannot be for work nobody sends), then the same, and nothing is written to the capture. A fix left to run over a page
// copied again here would put the write-backs' words back over the words the game's thread wrote after they landed
// (late_fix_cpu_read of the PR 248 nightlies did). Not recorded yet, or recorded in this part (not submitted: the GPU has not
// read its table): its fix dropped (the table's counts 0), and the page waits and is copied again, as before. The same for
// an earlier part that cannot be waited for: vk_part_wait says so only when the recordings have no timeline of their own or
// RX_VK_PER_QUEUE_FENCES=0 (no device and no OS decides it), and then the table's counts are zeroed while the GPU may be
// reading it: that is the race this way has, and the log's late-fix line counts those reads ("after the wait of an earlier
// part" and "of them copied again, not waited for"). A wait that ran past kPartWaitGuard (a hang guard, logged) is not
// repeated for that part
constexpr uint64_t kPartWaitGuard = 5000000000ull;   // (ns)
void late_fix_cpu_read(uint32_t page, VkBuffer buffer, uint8_t* slot) {
    if (late_pages.empty()) return;
    const auto it = late_pages.find(page);
    if (it == late_pages.end() || it->second.buffer != buffer || it->second.slot != slot) return;
    LatePage& lp = it->second;
    static uint64_t part_wait_failed = 0;   // (a part waited for past the guard is not waited for again)
    bool fixed = lp.fix_value && lp.fix_value <= mem_completed;   // (the buffer is as the fix left it)
    if (fixed) st_lfx.rd_fixed++;
    if (!fixed && lp.fix_value && lp.fix_value < mem_rec_value) vk_timeline_wait(lp.fix_value), fixed = true, st_lfx.rd_earlier_rec++;
    if (!fixed && lp.fix_value && lp.fix_part && lp.fix_part != mem_rec_part.load()) {   // (an earlier part of this recording)
        const int waited = lp.fix_part == part_wait_failed ? 0 : vk_part_wait(lp.fix_part, kPartWaitGuard);
        if (waited > 0) fixed = true, st_lf_part_waits++;
        else {
            st_lf_part_fell++;
            if (waited == 0 && lp.fix_part != part_wait_failed) {
                part_wait_failed = lp.fix_part;
                logf("vk: late pages' fix: part %llu not done after %llu s: its pages are copied again", (unsigned long long)lp.fix_part,
                     (unsigned long long)(kPartWaitGuard / 1000000000ull));
            }
        }
    }
    if (fixed) {
        invalidate_mapped(lp.alloc, lp.nbytes);
    } else {
        if (!lp.fix_value) {
            st_lfx.rd_not_recorded++;
            for (size_t k = lp.first; k < lp.first + lp.n && k < late_fixes.size(); k++) late_fixes[k].count = 0;
        } else if (lp.fix_map) {
            if (lp.fix_part == mem_rec_part.load()) st_lfx.rd_this_part++;
            for (size_t k = 0; k < lp.n; k++) lp.fix_map[k].count = 0;
            flush_mapped(lp.fix_alloc, lp.fix_end);
        }
        vk_timeline_wait(lp.late_value);
        guarded_copy(slot, (uint8_t*)host->guest_base() + ((uint64_t)page << kPageShift), 4096);
        flush_mapped(lp.alloc, lp.nbytes);
    }
    st_lf_cpu_reads++;
    late_pages.erase(it);
}
// (pools_mu held) the stamps given back and the fixes known done, let go
void late_fix_reap() {
    static uint64_t reaps;
    if (++reaps % 64 == 0) {
        stamps_drop_unused();
        std::erase_if(late_shadow_pools, [](uint64_t lo) { return !imported.count(lo); });   // (pools gone: their shadow no longer wanted)
    }
    while (!dead_stamps.empty() && dead_stamps.front().value <= mem_completed) {
        destroy_buffer(dead_stamps.front().buf, dead_stamps.front().alloc);
        dead_stamps.pop_front();
    }
    for (auto it = late_pages.begin(); it != late_pages.end();)
        it = it->second.fix_value && it->second.fix_value <= mem_completed ? late_pages.erase(it) : std::next(it);
}
// (pools_mu held) the stamps' new blocks filled and the taken captures' fixes recorded, in the recording's uploads: right
// before anything copies out of a capture's buffer on the GPU (mirror_sync, mirror_make) and before its write-back.
// Not at a recording's start: in a burst the captures of its later pieces are taken while its recording goes on, and
// memory_begin_submission gives mirrors back after begin_recording. The upload command buffer runs after every recording
// before it, so after the write-backs the fixes wait for; and before this recording's own work
void late_fix_emit() {
    if (stamp_unfilled.empty() && late_fixes.empty()) return;
    late_built.clear();
    VkCommandBuffer up = upload_command_buffer();   // (its full barrier: after the submissions before, their write-backs)
    for (uint64_t key : stamp_unfilled) {
        const auto it = stamp_blocks.find(key);
        if (it == stamp_blocks.end()) continue;
        vkCmdFillBuffer(up, it->second.buf, 0, VK_WHOLE_SIZE, (uint32_t)(mem_rec_value - 1));
        it->second.since = mem_rec_value + 1;   // (the recording's parts before this one wrote back with no stamps: from the next)
    }
    stamp_unfilled.clear();
    if (!late_fixes.empty()) {
        // (each page's shadow and stamps still what its fixes read: else its fixes go, and it waits and is copied again, as before)
        bool all_gone = true;
        for (auto& [pg, lp] : late_pages) {
            if (lp.fix_value) continue;
            const auto pi = imported.find(lp.pool);
            const Pool* p = pi != imported.end() ? &pi->second : nullptr;
            bool ok = p && p->shadow == lp.shadow;
            if (ok && lp.stamp_key != UINT64_MAX) {
                const auto sb = stamp_blocks.find(lp.stamp_key);
                ok = sb != stamp_blocks.end() && sb->second.buf == lp.stamp_buf && sb->second.shadow == lp.shadow;
            }
            if (ok) { all_gone = false; continue; }
            for (size_t k = lp.first; k < lp.first + lp.n && k < late_fixes.size(); k++) late_fixes[k].count = 0;
            lp.fix_value = UINT64_MAX;   // (marked: waited for below)
            st_lf_moved++;
        }
        const uint64_t bytes = late_fixes.size() * sizeof(LateFix);
        LateFixTable* t = all_gone ? nullptr : late_fix_table_get(bytes);
        if (!t && !all_gone) {   // (no table free: what the fixes do, done by the wait, as before)
            st_lf_no_table++;
            for (auto& [pg, lp] : late_pages)
                if (!lp.fix_value) lp.fix_value = UINT64_MAX;
        }
        if (t) {
            memcpy(t->map, late_fixes.data(), bytes);
            flush_mapped(t->alloc, bytes);
            // (after what this upload command buffer wrote since its last barrier: this emit can run several times in one
            // recording, and between two runs shadow_sync_copy writes a pool's shadow, which the diff fixes read; the stamps'
            // fills above too. Left out when nothing came in between: the dedupe)
            full_barrier(up, "late pages' fix (before)");
            GprofSpan gp(up, "late pages' fix (shadows, snapshots -> capture buffers)");
            gp.items = (int)late_fixes.size();
            vkCmdBindPipeline(up, VK_PIPELINE_BIND_POINT_COMPUTE, late_fix_pipe.pipe);
            const struct { uint64_t table; uint32_t count, pad; } pc{t->address, (uint32_t)late_fixes.size(), 0};
            vkCmdPushConstants(up, late_fix_pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
            const uint32_t n = (uint32_t)late_fixes.size();
            vkCmdDispatch(up, std::min<uint32_t>(n, 65535u), (n + 65534u) / 65535u, 1);
            for (auto& [pg, lp] : late_pages)
                if (!lp.fix_value) {
                    lp.fix_value = mem_rec_value, lp.fix_map = (LateFix*)t->map + lp.first;
                    lp.fix_alloc = t->alloc, lp.fix_part = mem_rec_part.load(), lp.fix_end = (lp.first + lp.n) * sizeof(LateFix);
                    if (!snap_pages.empty())   // (its snapshot's slot read by this recording: not given back before it is done)
                        if (const auto sp = snap_pages.find(pg); sp != snap_pages.end()) sp->second.used = std::max(sp->second.used, mem_rec_value);
                    for (auto& [g, cb] : cap_bufs)   // (its buffer kept until this recording is done: it writes there)
                        if (cb.buffer == lp.buffer && cb.value == lp.cap_value) cb.last_use = std::max(cb.last_use, mem_rec_value);
                }
        }
        uint64_t v = 0;   // (the pages whose fixes went: the wait and the copy, as before)
        for (const auto& [pg, lp] : late_pages)
            if (lp.fix_value == UINT64_MAX) v = std::max(v, lp.late_value);
        if (v) {
            vk_timeline_wait(v);
            for (auto it = late_pages.begin(); it != late_pages.end();) {
                if (it->second.fix_value != UINT64_MAX) { ++it; continue; }
                guarded_copy(it->second.slot, (uint8_t*)host->guest_base() + ((uint64_t)it->first << kPageShift), 4096);
                flush_mapped(it->second.alloc, it->second.nbytes);
                it = late_pages.erase(it);
            }
        }
        late_fixes.clear();
    }
    full_barrier(up);   // (the stamps filled and the capture buffers fixed before anything copies from them)
    static int64_t emits;
    if (++emits % 600 == 0) {
        logf("vk: late pages' fix (600 recordings with one): of %lld captures with late pages %lld fixed (%.1f pages, %.1f fixes "
             "a capture), waited as before: %lld no pool, %lld no stamps or shadow yet, %lld stamps too new, %lld "
             "whole copies, %lld in place or unknown, %lld a page's earlier fix not recorded yet, %lld other; %lld read on the CPU first, %lld with no table, %lld pages whose shadow changed "
             "first; stamps %zu blocks (%llu KB); of the reads on the CPU, %lld waited for an earlier part's fix, %lld copied the page "
             "again instead (that part not waited for); the paths: reads on the CPU with the fix done %lld, of an earlier recording %lld, "
             "not recorded yet %lld, of this part %lld; pages fixed from stamps %lld, from snapshots %lld; the shadows: %lld pages "
             "synced from captures, %lld made known by it, %lld left unknown because the replay or the collection wrote them since",
             (long long)st_lf_takes, (long long)st_lf_fixed, st_lf_fixed ? (double)st_lf_pages / st_lf_fixed : 0.0,
             st_lf_fixed ? (double)st_lf_fixes / st_lf_fixed : 0.0, (long long)st_lf_fell[1],
             (long long)st_lf_fell[3], (long long)st_lf_fell[4], (long long)st_lf_fell[5], (long long)st_lf_fell[6],
             (long long)st_lf_fell[2], (long long)st_lf_fell[0], (long long)st_lf_cpu_reads, (long long)st_lf_no_table, (long long)st_lf_moved, stamp_blocks.size(),
             (unsigned long long)(stamp_bytes >> 10), (long long)st_lf_part_waits, (long long)st_lf_part_fell,
             (long long)st_lfx.rd_fixed, (long long)st_lfx.rd_earlier_rec, (long long)st_lfx.rd_not_recorded, (long long)st_lfx.rd_this_part,
             (long long)st_lfx.fix_diff_pages, (long long)st_lfx.fix_whole_pages, (long long)st_lfx.sync_pages, (long long)st_lfx.sync_known,
             (long long)st_lfx.sync_left_unknown);
        st_lf_takes = st_lf_fixed = st_lf_pages = st_lf_fixes = st_lf_cpu_reads = st_lf_no_table = st_lf_moved = st_lf_part_waits = st_lf_part_fell = 0;
        st_lfx = LateStats{};
        for (int64_t& f : st_lf_fell) f = 0;
    }
}
// ---- the late pages left to their reads (RX_VK_LATE_DEFER, 2026-11-14) ----
// A capture's late page (capture_make: a page the CPU wrote that a submission still on the GPU writes back) is copied again
// once that submission is done, and take_capture waited for it, on the replay's thread, at every capture that had one. Where
// the buffer pools have no mirror in VRAM (an integrated GPU: the Steam Deck), nothing on the GPU reads a capture's copy of a
// page: the draws bind guest memory in place, and the mirrors' syncs, the only GPU copies out of a capture's buffer, have no
// mirror to fill. Its readers are the CPU's: the replay's reads as submitted (memory_read_as_submitted) and the texture shadow.
// So a late page outside the texture storage is left as the capture copied it, noted, and copied again at the first read of
// it, after its own write-back, as the wait would have done; the texture storage's late pages wait as before. Once the game's
// reads of GPU results stop waiting (RX_VK_READBACK_LATE=newest), the game writes those pages again while their submission
// runs, and on the Steam Deck LCD at 1280x720 (Kokiri Forest) the capture's wait was 69-94 of 600 takes, 2.5-3.5 ms a take
// on average, about 17 ms a frame of the replay's "memory sync". RX_VK_LATE_DEFER=0: the wait, as before. Off where pools
// have mirrors (their syncs copy out of the captures) and with the parallel replay (its workers read the captures too)
namespace {
bool mirror_on();   // (below)
struct LateDefer {
    uint32_t gen = 0, slot = 0;   // (the capture's generation and the page's slot in its buffer: the copy that is stale)
    uint64_t value = 0;           // (the write-back the copy missed)
};
std::unordered_map<uint32_t, LateDefer> late_defer;   // (by page; pools_mu)
int64_t st_ld_takes, st_ld_pages, st_ld_kept, st_ld_reads, st_ld_read_waits;   // (pools_mu; logged with a capture's take, every 600)
int64_t st_ld_tex;   // (late pages in a texture's storage, which wait: the replay's thread, logged with the rest)
double st_ld_read_wait_ms;
} // namespace
bool late_defer_on() {   // (readback_early.h's defer: where no pool has a mirror, as RX_VK_READBACK_EARLY unless given)
    static const bool on = [] {
        const char* e = getenv("RX_VK_LATE_DEFER");
        const char* par = getenv("RX_VK_PARALLEL");
        const bool v = readback_early::defer(e, mirror_on(), par, readback_early_on());
        logf("vk: late pages left to their reads %s", v ? (e ? "on (RX_VK_LATE_DEFER)" : "on, with the early reads (RX_VK_LATE_DEFER=0: off)")
             : mirror_on() ? "off: the pools have mirrors" : readback_early::parallel(par) ? "off: the parallel replay"
             : e ? "off (RX_VK_LATE_DEFER)" : "off, as the early reads (RX_VK_LATE_DEFER=1: on)");
        return v;
    }();
    return on;
}
bool late_defer_any() { return !late_defer.empty(); }
/// (pools_mu held) A read on the CPU of a page whose kept copy is late (late_defer_split): after the write-back it missed, the
/// copy taken again from guest memory, as take_capture's wait did, then no longer late
void late_defer_cpu_read(uint32_t page, uint32_t gen, uint32_t slot, VmaAllocation alloc, uint8_t* map) {
    const auto it = late_defer.find(page);
    if (it == late_defer.end()) return;
    const LateDefer d = it->second;
    late_defer.erase(it);
    if (d.gen != gen || d.slot != slot) return;   // (another capture's copy: that one is not late)
    st_ld_reads++;
    if (d.value > vk_timeline_completed()) {
        const auto tw = std::chrono::steady_clock::now();
        vk_timeline_wait(d.value);
        st_ld_read_waits++;
        st_ld_read_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
    }
    guarded_copy(map + (uint64_t)slot * 4096, (uint8_t*)host->guest_base() + ((uint64_t)page << kPageShift), 4096);
    flush_mapped_range(alloc, (uint64_t)slot * 4096, 4096);
}
/// The late pages of c that are left to their reads (indices into c.pages, into `out`), and the write-back the rest wait for (0:
/// none). A page in a texture's own storage waits: the shadow copies it at the take, for that texture's uploads. The shadow's
/// spans are the whole pools (up to 256 MB) textures were uploaded from, and OoT's GPU results (the camera block, the
/// meters) share pools with textures: a page in a span but in no texture's storage is left to its reads, and the shadow lets
/// its stale copy go (take_capture's late_index). One whose write-back that wait covers anyway is copied again with the rest
uint64_t late_defer_split(const Capture& c, std::vector<uint32_t>& out, std::vector<uint64_t>& out_pv) {
    out.clear(), out_pv.clear();
    uint64_t wait = 0;
    {
        std::lock_guard sk(shadow_mu);
        RangeWalk in_tex{tex_merged};   // (c.late in address order: an ordered subset of c.pages)
        for (size_t k = 0; k < c.late.size(); k++) {
            const uint64_t a = (uint64_t)c.pages[c.late[k]] << kPageShift;
            if (in_tex.in(a) && texture_storage_any(a, 4096)) wait = std::max(wait, c.late_pv[k]), st_ld_tex++;
        }
    }
    for (size_t k = 0; k < c.late.size(); k++)
        if (c.late_pv[k] > wait) out.push_back(c.late[k]), out_pv.push_back(c.late_pv[k]);
    return wait;
}
bool take_capture(std::unique_ptr<Capture> c, uint64_t value, uint64_t completed) {
#ifdef TRACY_ENABLE
    HT_SCOPE("take_capture");
#endif
    const auto tk0 = std::chrono::steady_clock::now();
    bool fixed = false;   // (the late pages fixed on the GPU, no wait: late_fix_take)
    static thread_local std::vector<uint32_t> deferred;   // (indices into c->pages: late pages left to their reads)
    static thread_local std::vector<uint64_t> deferred_pv;   // (and the write-back each missed)
    deferred.clear(), deferred_pv.clear();
    if (c->late_value && c->map) {   // (the pages a submission still on the GPU writes back: see capture_make)
      if (!(fixed = late_fix_take(*c))) {
        uint64_t wait = c->late_value;
        if (late_defer_on() && c->late_pv.size() == c->late.size()) wait = late_defer_split(*c, deferred, deferred_pv);
        // (timed apart: the replay thread waits here for the GPU, the more often the fewer pools have a mirror. With
        // RX_VK_MIRROR_MB=1024 on an 8 GB card it was most of a capture's 2.4-3 ms, six submits a frame: Hyrule Field at
        // 35-45 fps instead of 55, issue 995)
        if (wait) {
            const auto tw = std::chrono::steady_clock::now();
            vk_timeline_wait(wait);
            st_tk_late += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
            st_late_wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tw).count(),
                                      std::memory_order_relaxed);   // (hookshot: the frame trace's late_wait_ms)
            st_tk_late_n++;
            uint8_t* const base = (uint8_t*)host->guest_base();
            size_t d = 0;   // (deferred: in c->late's order)
            for (uint32_t i : c->late) {
                if (d < deferred.size() && deferred[d] == i) { d++; continue; }
                guarded_copy((uint8_t*)c->map + (uint64_t)i * 4096, base + ((uint64_t)c->pages[i] << kPageShift), 4096);
            }
            flush_mapped(c->alloc, c->nbytes);
        }
      }
    }
    const auto tl = std::chrono::steady_clock::now();
    std::lock_guard lk(pools_mu);
    const auto tk_locked = std::chrono::steady_clock::now();
    st_tk_lock += std::chrono::duration<double, std::milli>(tk_locked - tl).count();
    cur_cap_value = value;
    cur_cap_gen = 0;
    // (captures past the last kKeep, or past the byte budget, retired once the work that read them is done)
    while (!cap_bufs.empty()) {
        auto oldest = cap_bufs.begin();
        const bool over = cap_bufs.size() > kKeep || cap_bytes > kCapBytesMax;
        if (!over || std::max(oldest->second.value, oldest->second.last_use) > completed) break;
        cap_buf_retire(oldest);
    }
    uint32_t g = 0;
    if (!c->pages.empty()) {
        g = ++gen_now;
        for (uint32_t p : c->pages) stamp_page(p, g);
        pages_collected += c->pages.size();
    }
    static thread_local std::vector<uint8_t> late_index;   // (by index in pages: a late page the GPU fixes)
    if (fixed) {
        late_index.assign(c->pages.size(), 0);
        for (uint32_t i : c->late) late_index[i] = 1;
    } else if (!deferred.empty()) {   // (a late page left to its reads, in the texture storage since the split)
        late_index.assign(c->pages.size(), 0);
        for (uint32_t i : deferred) late_index[i] = 1;
    }
    shadow_seq_own.store(0, std::memory_order_relaxed);   // (no copy is this submission's until its capture's block below)
    if (!c->pages.empty() && tex_capture_on()) {   // (the texture storage it took: the shadow, as of this submit)
        std::lock_guard sk(shadow_mu);
        shadow_seq++;
        shadow_seq_own.store(shadow_seq, std::memory_order_relaxed);
        // (the copies of pages no capture took in the last 256 submits go, over 32 MB of them, every 256 captures: such a page is what guest
        // memory holds (a write drops its copy), and the cap's sweep below came only at 256 MB)
        if (shadow_seq % 256 == 0 && shadow_n.load(std::memory_order_relaxed) > 8192)
            for (auto it = shadow.begin(); it != shadow.end();) {
                if (it->second.seq + 256 >= shadow_seq) { ++it; continue; }
                auto r = shadow_regions.find(it->first >> 9);
                if (r != shadow_regions.end() && --r->second == 0) shadow_regions.erase(r);
                it = shadow.erase(it);
                shadow_n--;
            }
        if (!tex_merged.empty()) {
            // (every page written since the last capture: its old copy is stale; the ones copied now replace it)
            // (hookshot: those copied now are overwritten in place below, their buffers kept: erasing every written page and
            // inserting the copied ones again was a free, an allocation and two hash operations a page, for the pages the
            // game rewrites every frame. c->slots[i] is the index in c->pages of c->copied[i], in order)
            {
                RangeWalk in_tex{tex_merged};
                size_t j = 0;
                for (size_t i = 0; i < c->pages.size(); i++) {
                    if (j < c->slots.size() && c->slots[j] == i) {
                        j++;
                        continue;
                    }
                    const uint32_t p = c->pages[i];
                    if (in_tex.in((uint64_t)p << 12)) shadow_erase(p);
                }
            }
            bool swept = false;
            static int64_t st_takes, st_from_stage, st_from_map, st_cached;   // (the shadow copies by source and the takes whose buffer was cached, logged every 600 takes; shadow_mu)
            int64_t from_stage = 0, from_map = 0;
            static thread_local std::vector<CopyPair> pairs;   // (the copies, made together once the slots are known)
            pairs.clear();
            size_t i = 0;
            RangeWalk in_tex{tex_merged};   // (c->copied in address order too: a subset of c->pages)
            for (; i < c->copied.size(); i++) {
                const uint32_t p = c->copied[i];
                if (!in_tex.in((uint64_t)p << 12)) continue;
                if ((fixed || !deferred.empty()) && late_index[c->slots[i]]) {   // (its bytes are fixed on the GPU, or are to come: read from guest memory instead)
                    shadow_erase(p);
                    continue;
                }
                if (shadow_n.load(std::memory_order_relaxed) >= kShadowMax && !shadow.count(p)) {
                    if (swept) break;
                    swept = true;
                    for (auto it = shadow.begin(); it != shadow.end();) {
                        if (it->second.seq + 64 >= shadow_seq) { ++it; continue; }
                        auto r = shadow_regions.find(it->first >> 9);
                        if (r != shadow_regions.end() && --r->second == 0) shadow_regions.erase(r);
                        it = shadow.erase(it);
                        shadow_n--;
                    }
                    if (shadow_n.load(std::memory_order_relaxed) >= kShadowMax) break;
                }
                ShadowPage& slot = shadow[p];
                if (!slot.b) {
                    slot.b.reset(new uint8_t[4096]);
                    shadow_regions[p >> 9]++;
                    shadow_n++;
                }
                slot.seq = shadow_seq;
                // (the staging, ordinary RAM, when it holds the page; else the capture buffer, maybe write-combined)
                const int32_t sx = c->sidx.empty() ? -1 : c->sidx[c->slots[i]];
                if (sx >= 0) pairs.push_back({slot.b.get(), (const uint8_t*)c->stage.get() + (uint64_t)sx * 4096}), from_stage++;
                else pairs.push_back({slot.b.get(), (const uint8_t*)c->map + (uint64_t)c->slots[i] * 4096}), from_map++;
            }
            // (shadow_mu still held: no slot taken here goes before its copy; the sweep above drops only older ones)
            if (!pairs.empty()) copy_pairs(pairs.data(), (int64_t)pairs.size());
            st_from_stage += from_stage, st_from_map += from_map, st_cached += c->buf_cached ? 1 : 0;
            if (++st_takes % 600 == 0) {
                logf("vk: shadow staging (600 takes): %lld shadow copies from staging, %lld from the capture buffer; capture buffers: %lld "
                     "host-cached, %lld not (write-combined: staged)", (long long)st_from_stage, (long long)st_from_map,
                     (long long)st_cached, (long long)(600 - st_cached));
                st_from_stage = st_from_map = 0, st_cached = 0;
            }
            for (; i < c->copied.size(); i++)   // (the shadow full: the copied pages not replaced lose their old copies)
                if (in_tex.in((uint64_t)c->copied[i] << 12)) shadow_erase(c->copied[i]);
        }
    }
    const auto tk1 = std::chrono::steady_clock::now();
    if (!c->copied.empty() && g && c->map) {   // (its buffer kept with the captures, for the mirror syncs)
        CapBuf b{value, c->buffer, c->alloc, c->bsize, c->map};
        c->buffer = VK_NULL_HANDLE, c->alloc = nullptr, c->map = nullptr, c->bsize = 0;
        cap_bufs.emplace(g, b);
        cap_bytes += b.size;
        cur_cap_gen = g;
        if (!late_defer.empty())   // (a page this capture took again is no longer late in an older one)
            for (uint32_t p : c->copied) late_defer.erase(p);
        for (size_t i = 0; i < c->copied.size(); i++) cap_entry_set(c->copied[i], CapEntry{g, c->slots[i]});
        for (size_t k = 0; k < deferred.size(); k++) {   // (its late pages left to their reads: late_defer_cpu_read)
            const uint32_t p = c->pages[deferred[k]];
            const CapEntry e = cap_entry(p);
            if (e.gen == g && e.slot == deferred[k]) late_defer[p] = {g, deferred[k], deferred_pv[k]}, st_ld_kept++;
        }
    }
    if (!deferred.empty()) st_ld_takes++, st_ld_pages += (int64_t)deferred.size();
    if (fixed) late_fix_kept(c->buffer == VK_NULL_HANDLE, value);   // (the buffer kept with the captures: else nothing reads the pages)
    cap_view_bump();
    stage_put(*c);   // (the shadow's copies from it are done: copy_pairs returned)
    if (c->map) cap_buf_free(c->buffer, c->alloc, c->map, c->bsize), c->buffer = VK_NULL_HANDLE, c->map = nullptr;
    {
        std::lock_guard lk(cap_mu);
        if (spare.size() < 8) spare.push_back(std::move(c));
    }
    {   // (every 600 takes: where a take's time went)
        const auto tk2 = std::chrono::steady_clock::now();
        static int64_t n;
        st_tk_stamp += std::chrono::duration<double, std::milli>(tk1 - tk_locked).count();
        st_tk_all += std::chrono::duration<double, std::milli>(tk2 - tk0).count();
        if (++n % 600 == 0) {
            logf("vk: a capture taken (ms): all %.3f; GPU wait for late pages %.3f (%lld of 600 takes), pools lock %.3f, retires "
                 "and stamps %.3f, bytes copied %.3f; %zu kept (%llu MB), %zu spare buffers, %zu texture pages copied (%llu MB)",
                 st_tk_all / 600, st_tk_late / 600, (long long)st_tk_late_n, st_tk_lock / 600, st_tk_stamp / 600, st_tk_copy / 600,
                 cap_bufs.size(), (unsigned long long)(cap_bytes >> 20), cap_free.size(), shadow_n.load(std::memory_order_relaxed),
                 (unsigned long long)(shadow_n.load(std::memory_order_relaxed) >> 8));
            st_tk_all = st_tk_stamp = st_tk_copy = st_tk_late = st_tk_lock = 0, st_tk_late_n = 0;
            if (late_defer_on()) {   // (and the late pages left to their reads; a copy no read came for, gone with its capture)
                std::erase_if(late_defer, [](const auto& kv) { return !cap_bufs.count(kv.second.gen); });
                logf("vk: late pages left to their reads (600 takes): %lld pages in %lld takes (%lld kept with their capture), %lld "
                     "read on the CPU, %lld of them waited (%.2f ms in all); %zu still late; %lld late pages in a texture's storage "
                     "waited at the take", (long long)st_ld_pages, (long long)st_ld_takes, (long long)st_ld_kept, (long long)st_ld_reads,
                     (long long)st_ld_read_waits, st_ld_read_wait_ms, late_defer.size(), (long long)st_ld_tex);
                st_ld_pages = st_ld_takes = st_ld_kept = st_ld_reads = st_ld_read_waits = st_ld_tex = 0, st_ld_read_wait_ms = 0;
            }
        }
    }
    return true;
}
bool memory_take_capture(uint64_t value, uint64_t completed) {
    if (completed > mem_completed) mem_completed = completed;
    if (!capture_on()) return false;
    std::unique_ptr<Capture> c;
    {
        std::lock_guard lk(cap_mu);
        if (captures.empty()) return false;
        c = std::move(captures.front());
        captures.pop_front();
    }
    return take_capture(std::move(c), value, completed);
}
/// (hookshot) A capture made and taken at once, for a submission going on from a wait the CPU's signal met
/// (vk_replay.cpp resume_queue): the CPU's writes up to that signal, which the console's GPU reads after its wait
bool memory_capture_now(uint64_t value, uint64_t completed) {
#ifdef TRACY_ENABLE
    HT_SCOPE("memory_capture_now");
#endif
    if (completed > mem_completed) mem_completed = completed;
    std::unique_ptr<Capture> c = capture_make();
    return c && take_capture(std::move(c), value, completed);
}


namespace {

/// The imported pool holding [guest, guest + size) (imported now if it was not yet); null outside the pools.
/// pools_mu held.
Pool* pool_cache[4096];   // (by 64 KB page: the pools buffer bindings hit lately)
std::atomic<uint64_t> pools_gen{1};   // (bumped when a pool leaves the table: the recording threads' caches go stale)
struct TlPool {
    Pool* p;
    uint64_t gen;
};
thread_local TlPool tl_pools[1024];   // (a parallel replay thread's: by 64 KB page, read without the lock)
// RX_VK_TL_POOLS_WIDE (on by default since ; =0: off): 16384 entries (1 GB before two pages share one) instead of 1024 (64 MB): the game's
// pools lie further apart than that, and pages 64 MB apart took turns in one entry (a third of a command buffer's buffer binds went
// to the lock, "bind page range")
constexpr uint64_t kTlWide = 16384;
thread_local TlPool* tl_pools_wide = nullptr;
bool tl_wide_on() {
    static const bool on = env_on("RX_VK_TL_POOLS_WIDE");   // (off by default here: =1 on)
    return on;
}

Pool* pool_of(uint64_t guest, uint64_t size) {
    if (!size || size > UINT64_MAX - guest) return nullptr;
    if (last_pool && guest >= last_pool->lo && guest + size <= last_pool->hi) return last_pool;
    Pool*& slot = pool_cache[(guest >> 16) & 4095];
    if (slot && guest >= slot->lo && guest + size <= slot->hi) return last_pool = slot;
    Pool* found = pool_lookup(guest, size);
    if (found) slot = found;
    return found;
}

// ---- pools imported in the background: an import took 5-45 ms (the driver locks the pages; the game's threads keep
// the address space busy committing theirs), and the game makes ~13 pools a second. The host says when the game makes
// one (memory_pool_created), a thread of ours imports it, and a lookup that finds one under way waits for it rather
// than importing it again. Imports made are handed over in a list (the lookup holds pools_mu, the thread never). ----
std::mutex bg_mu;
std::condition_variable bg_cv, bg_done_cv;
std::deque<std::pair<uint64_t, uint64_t>> bg_queue;
std::map<uint64_t, uint64_t> bg_inflight;   // [lo, hi) queued or being imported, under bg_mu
std::vector<std::pair<uint64_t, Pool>> bg_ready;
int64_t st_bg_imports, st_bg_waits, st_prefetch_sent, st_prefetch_used, st_prefetch_dropped;
// (hookshot) pool_lookup's import attempts, counted apart by how they ended: made, refused by the driver (import_pool
// failed after asking it), or held: the range failed within ImportHolds::kGapMs, so import_pool returned at once without
// asking the driver. A held retry is no import: counted as one, a range the driver refused showed as a million imports in
// 0.0 ms. Read at close (import_stats_log) from another thread, so atomic.
std::atomic<int64_t> st_imports_made{0}, st_imports_refused{0}, st_imports_held{0};
void bg_main();
void bg_start() {   // (bg_mu held)
    static bool started;
    if (!started) {
        started = true;
        std::thread(bg_main).detach();
    }
}

void bg_main() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk imports");
    for (;;) {
        std::pair<uint64_t, uint64_t> r;
        {
            std::unique_lock lk(bg_mu);
            bg_cv.wait(lk, [] { return !bg_queue.empty(); });
            r = bg_queue.front();
            bg_queue.pop_front();
        }
        Pool p;
        const bool ok = import_pool(r.first, r.second, p);
        std::lock_guard lk(bg_mu);
        p.prefetched = true;
        if (ok) bg_ready.push_back({r.first, p}), st_bg_imports++;
        bg_inflight.erase(r.first);
        bg_done_cv.notify_all();
    }
}

/// The imports the thread made, into the table (pools_mu held); one already there makes the new one redundant
void bg_drain() {
    std::vector<std::pair<uint64_t, Pool>> ready;
    {
        std::lock_guard lk(bg_mu);
        if (bg_ready.empty()) return;
        ready.swap(bg_ready);
    }
    for (auto& [lo, p] : ready) {
        auto imp = imported.find(lo);
        if (imp != imported.end() && imp->second.hi >= p.hi) {
            destroy_import_buffer(p.buffer);
            vkFreeMemory(dev.device, p.memory, nullptr);
            for (VkDeviceMemory m : p.pieces) vkFreeMemory(dev.device, m, nullptr);
            continue;
        }
        bool overlap = false;   // (amd-fix: a chunk imported ahead that an import made since covers in part: the pools stay disjoint)
        for (auto o = imported.lower_bound(p.hi); o != imported.begin();) {
            --o;
            if (o == imp) continue;
            if (o->second.hi <= p.lo) {
                if (import_reach::beyond(live_reach, o->second.lo, p.lo)) break;   // (none further down reaches p.lo)
                continue;
            }
            overlap = true;
            break;
        }
        if (overlap) {
            st_prefetch_dropped++;
            destroy_import_buffer(p.buffer);
            vkFreeMemory(dev.device, p.memory, nullptr);
            for (VkDeviceMemory m : p.pieces) vkFreeMemory(dev.device, m, nullptr);
            continue;
        }
        if (imp != imported.end()) retire_pool(imp);
        imported_add(p);
        guest_pages_set(p.lo, p.hi, p.address);
    }
}

/// (amd-fix) An import on the replay thread stalled it 20-75 ms on an RX 9060 XT (the driver locks the chunk's pages, the
/// commit of fresh ones), a tiny uniform buffer's binding as long, ~40 times entering Kokiri: hitches. The chunks next to
/// one imported on demand, inside the game's pools, are imported ahead by the background thread; a lookup that needs one
/// still under way waits for it (less than importing it), and a chunk imported ahead and first bound sends for its own
/// neighbours. RX_VK_PREFETCH_CHUNKS: how many on each side (default 0: off, the background imports slowed the replay thread's own to 80-127 ms on AMD). (pools_mu held)
void prefetch_chunks(uint64_t lo, uint64_t hi, uint64_t chunk) {
    static const int n = getenv("RX_VK_PREFETCH_CHUNKS") ? std::max(0, atoi(getenv("RX_VK_PREFETCH_CHUNKS"))) : 0;
    if (n <= 0 || !chunk || !dev.external_memory_host || hi <= lo) return;
    auto in_game_pool = [](uint64_t a, uint64_t b) {
        auto g = std::lower_bound(guest_pools.begin(), guest_pools.end(), b, [](const GuestPool& p, uint64_t v) { return p.lo < v; });
        return g != guest_pools.begin() && std::prev(g)->hi > a;
    };
    auto imported_over = [](uint64_t a, uint64_t b) {
        for (auto o = imported.lower_bound(b); o != imported.begin();) {
            --o;
            if (o->second.hi > a) return true;
            if (import_reach::beyond(live_reach, o->second.lo, a)) break;
        }
        return false;
    };
    std::vector<std::pair<uint64_t, uint64_t>> want;
    for (int k = 0; k < n; k++) {
        const uint64_t up = hi + (uint64_t)k * chunk;
        if (up < UINT64_MAX - chunk) want.push_back({up, up + chunk});
        const uint64_t down = (uint64_t)(k + 1) * chunk;
        if (lo >= down) want.push_back({lo - down, lo - down + chunk});
    }
    std::lock_guard lk(bg_mu);
    for (const auto& [a, b] : want) {
        if (b > (1ull << 40) || !in_game_pool(a, b) || imported_over(a, b)) continue;
        if (bg_inflight.contains(a)) continue;
        if (bg_inflight.size() >= 16) break;
        bg_start();
        bg_inflight.emplace(a, b);
        bg_queue.push_back({a, b});
        st_prefetch_sent++;
    }
    bg_cv.notify_one();
}

/// (imports by chunk (RX_VK_IMPORT_CHUNK_MB), not by pool: each import cost 5-45 ms in the driver, more as their number grew,
/// and the game makes ~13 pools a second; a chunk covers the pools after it at no cost)
/// (hookshot) 4 MB, not 16: a pool's VRAM mirror is its import's whole range, and with 16 MB chunks 28-39% of the mirrors'
/// bytes were outside every game pool (DLAA 2560 x 1440: 459 of 1,666 MB in a Hyrule Field walk, up to 925 MB after Castle
/// Town to the field). At 4 MB the mirrors were 1,236-1,268 MB against 1,645-1,657 in the same walk, at the same frame rate
/// (58.1-58.6 against 58.2-59.0 fps); in Kokiri Forest 843 MB against 1,154; with a 6 GB allowance the VRAM peak stayed under
/// it (5,865-5,887 MB against 6,129-6,135) and the GPU's longest frame was 12.5-13.5 ms against 21.7-24.1 (the frame rate
/// went either way: 53.0 / 52.4 fps against 50.3 / 54.9). It takes ~50% more imports (91-94 against 59-65 in five minutes,
/// 0.8 s of driver time in all against 0.7, each one shorter): span_game_pools still takes a pool across a chunk's end into
/// one import. Only where the pools have mirrors (mirror_on): without them a chunk's padding costs no VRAM, and smaller chunks
/// would only mean more imports. RX_VK_IMPORT_CHUNK_MB=16: as before
bool mirror_on();
uint64_t import_chunk() {
    static const uint64_t chunk =
        (uint64_t)(getenv("RX_VK_IMPORT_CHUNK_MB") ? std::max(0, atoi(getenv("RX_VK_IMPORT_CHUNK_MB"))) : mirror_on() ? 4 : 16) << 20;
    return chunk;
}
/// (hookshot) A game pool across a chunk's end had each chunk imported on its own, at the first binding in it, and a later
/// binding across the end imported the two again as one, on the replay thread: OoT's 16 MB pool at 0x1021745000 bound at
/// 0x1021ffff00+0x160 across 0x1022000000, 130-147 ms, the one hitch of a panning run in Kokiri. Up to this size an import
/// takes the chunks of the game pools across its ends too (RX_VK_IMPORT_SPAN_MB, default 64; 0: not)
uint64_t import_span() {
    static const uint64_t span = (uint64_t)(getenv("RX_VK_IMPORT_SPAN_MB") ? std::max(0, atoi(getenv("RX_VK_IMPORT_SPAN_MB"))) : 64) << 20;
    return span;
}
/// [lo, hi), chunk-aligned, grown over the game pools that run across its ends, while it stays within import_span (pools_mu held)
void span_game_pools(uint64_t& lo, uint64_t& hi, uint64_t chunk) {
    const uint64_t span = import_span();
    if (!chunk || hi - lo >= span) return;
    for (bool grew = true; grew;) {
        grew = false;
        for (const uint64_t e : {lo, hi}) {
            // (a pool across e that fits starts above e - span: the pools from there to e, as the host's may overlap)
            auto g = std::upper_bound(guest_pools.begin(), guest_pools.end(), e, [](uint64_t v, const GuestPool& p) { return v < p.lo; });
            while (g != guest_pools.begin()) {
                --g;
                if (g->lo + span <= e) break;
                if (g->lo >= e || g->hi <= e) continue;
                const uint64_t nlo = std::min(lo, g->lo / chunk * chunk), nhi = std::max(hi, (g->hi + chunk - 1) / chunk * chunk);
                if (nhi - nlo > span || (nlo == lo && nhi == hi)) continue;
                lo = nlo, hi = nhi, grew = true;
            }
        }
    }
}

void noaccess_import_note(uint64_t bound_lo, uint64_t bound_hi, uint64_t lo, uint64_t hi);   // (below)
void clip_noaccess(uint64_t& lo, uint64_t& hi, uint64_t blo, uint64_t bhi);                  // (below)
Pool* pool_lookup(uint64_t guest, uint64_t size) {
    if (guest_pools.empty()) return nullptr;
    bg_drain();
    auto it = std::upper_bound(guest_pools.begin(), guest_pools.end(), guest,
                               [](uint64_t a, const GuestPool& p) { return a < p.lo; });
    if (it == guest_pools.begin()) return nullptr;
    --it;
    if (guest < it->lo || guest >= it->hi || !size || size > it->hi - guest) return nullptr;
    const uint64_t chunk = import_chunk();
    const GuestPool range = chunk ? GuestPool{guest, guest + size} : *it;
    auto covering = [&]() {   // (the import holding the whole range: the last one starting at or below it, or one further down)
        for (auto c = imported.upper_bound(range.lo); c != imported.begin();) {
            --c;
            if (c->second.hi >= range.hi) return c;
            if (import_reach::beyond(live_reach, c->second.lo, range.hi)) break;   // (none further down reaches range.hi)
        }
        return imported.end();
    };
    auto imp = chunk ? covering() : imported.find(range.lo);
    if (chunk && range.hi > UINT64_MAX - (chunk - 1)) return nullptr;
    const uint64_t want_lo = chunk ? range.lo / chunk * chunk : range.lo;
    const uint64_t want_hi = chunk ? (range.hi + chunk - 1) / chunk * chunk : range.hi;
    if (chunk && imp == imported.end()) imp = imported.find(want_lo);   // (a smaller one there: made again, larger)
    if (imp == imported.end() || imp->second.hi < range.hi) {   // (under way in the background: waited for)
        bool waited = false;
        {
            std::unique_lock lk(bg_mu);
            auto under_way = [&] {   // (the pool's own import, or a chunk of the range imported ahead)
                for (const auto& [x, end] : bg_inflight)
                    if (x == range.lo || (chunk && x >= want_lo && x < want_hi)) return true;
                return false;
            };
            if (under_way()) {
                bg_done_cv.wait(lk, [&] { return !under_way(); });
                waited = true;
                st_bg_waits++;
            }
        }
        if (waited) {
            bg_drain();
            imp = chunk ? covering() : imported.find(range.lo);
            if (chunk && imp == imported.end()) imp = imported.find(want_lo);
        }
    }
    if (imp == imported.end() || imp->second.hi < range.hi || imp->second.lo > range.lo) {
        // (the new pool takes in every imported pool its range overlaps, and they are retired: the imported pools stay
        // disjoint. A range crossing a chunk's end made the chunk's pool again, larger, over the next chunk, whose own
        // pool stayed: two mirrors of the same memory. MKW's GPU culling wrote its instance data (a 2.4 MB storage
        // buffer) through one, the draws read it through the other, synced from guest memory before the submission's
        // write-back: the instances of a frame before or older, objects popping in and out, one-frame flashes.)
        uint64_t lo = 0, hi = 0;
        std::vector<uint64_t> overlapped;
        auto plan = [&](bool spanned) {   // (the range, the imported pools it takes in; spanned: span_game_pools's chunks too)
            bool touches_neighbour = false;
            lo = want_lo, hi = std::max(want_hi, imp != imported.end() ? imp->second.hi : 0);
            if (spanned && chunk) span_game_pools(lo, hi, chunk);
            for (bool grew = true; grew;) {
                grew = false;
                overlapped.clear();
                auto it = imported.lower_bound(hi);   // (the pools starting below hi, the last ones first)
                while (it != imported.begin()) {
                    --it;
                    if (it->second.hi <= lo) {
                        if (import_reach::beyond(live_reach, it->second.lo, lo)) break;   // (none further down reaches lo)
                        continue;
                    }
                    // A larger span anticipates future bindings. If only that padding reaches a live
                    // import, keep the neighbour and plan just the chunks this binding needs. A real
                    // crossing still merges the imports and inherits their GPU-written contents.
                    if (spanned && (it->second.hi <= want_lo || it->second.lo >= want_hi)) touches_neighbour = true;
                    overlapped.push_back(it->first);
                    if (it->second.lo < lo) lo = it->second.lo, grew = true;
                    if (it->second.hi > hi) hi = it->second.hi, grew = true;
                }
            }
            if (hi - lo > (1ull << 30)) {   // (no larger than 1 GB: then as before, the chunk's pool alone, overlaps logged)
                static int logged_big;
                if (logged_big++ < 8)
                    logf("vk: pool %#llx..%#llx would take in %zu pools, %llu MB: left overlapping", (unsigned long long)lo,
                         (unsigned long long)hi, overlapped.size(), (unsigned long long)((hi - lo) >> 20));
                lo = want_lo, hi = std::max(want_hi, imp != imported.end() ? imp->second.hi : 0);
                overlapped.clear();
                if (imp != imported.end()) overlapped.push_back(imp->first);
            }
            return touches_neighbour;
        };
        const bool prefer_narrow = plan(true);
        if (prefer_narrow) plan(false);
        const uint64_t first_lo = lo, first_hi = hi;
        Pool p;
        bool ok = false;
        for (int pass = 0; pass < 2 && !ok; pass++) {
            if (pass) {   // Retry the alternate range once if its address or size differs.
                plan(prefer_narrow);
                if (lo == first_lo && hi == first_hi) break;
            }
            const auto t0 = std::chrono::steady_clock::now();
            ok = import_pool(lo, hi, p);
            if (!ok && import_cooling) {   // (held: no driver call, no time spent; the first and every 65536th logged)
                const int64_t held = ++st_imports_held;
                if (held == 1 || held % 65536 == 0)
                    logf("vk: pool %#llx+%#llx: its import as %#llx..%#llx failed less than %lld s ago, not tried again yet "
                         "(%lld lookups waited on a held range)",
                         (unsigned long long)range.lo, (unsigned long long)(range.hi - range.lo), (unsigned long long)lo,
                         (unsigned long long)hi, (long long)(ImportHolds::kGapMs / 1000), (long long)held);
                continue;
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            static double total_ms;   // (the driver attempts' time, made or refused)
            static int n, logged;
            total_ms += ms;
            if (!ok) {   // (refused: its VkResult, the first 16 refusals; held for ImportHolds::kGapMs from now)
                const int64_t refused = ++st_imports_refused;
                if (failures++ < 16) {
                    if (import_result != VK_SUCCESS)
                        logf("vk: pool %#llx+%#llx not imported as %#llx..%#llx in %.1f ms (VkResult %d; %lld refused)",
                             (unsigned long long)range.lo, (unsigned long long)(range.hi - range.lo), (unsigned long long)lo,
                             (unsigned long long)hi, ms, (int)import_result, (long long)refused);
                    else
                        logf("vk: pool %#llx+%#llx not imported as %#llx..%#llx in %.1f ms (%lld refused)",
                             (unsigned long long)range.lo, (unsigned long long)(range.hi - range.lo), (unsigned long long)lo,
                             (unsigned long long)hi, ms, (long long)refused);
                }
                continue;
            }
            n++, st_imports_made++;
            if ((ms >= 2.0 && logged++ < 200) || n % 256 == 0) {
                // (as before when nothing was refused or held; else how many, so the count of imports reads true)
                const int64_t refused = st_imports_refused, held = st_imports_held;
                char also[96] = "";
                if (refused || held)
                    snprintf(also, sizeof also, "; %lld refused, %lld lookups held", (long long)refused, (long long)held);
                logf("vk: pool %#llx+%#llx imported in %.1f ms as %#llx..%#llx (%d imports, %.0f ms in all%s)",
                     (unsigned long long)range.lo, (unsigned long long)(range.hi - range.lo), ms, (unsigned long long)lo,
                     (unsigned long long)hi, n, total_ms, also);
            }
        }
        if (!ok) return nullptr;   // (each attempt counted and logged above)
        for (uint64_t k : overlapped) {   // (the pools it took in)
            auto o = imported.find(k);
            if (o != imported.end())   // (their mirrors' contents come with them; and how often the range was trimmed: mirror_trim.h)
                p.gpu_only |= o->second.gpu_only, p.gpu_written |= o->second.gpu_written, p.trims = std::max(p.trims, o->second.trims),
                    p.trim_epoch = std::max(p.trim_epoch, o->second.trim_epoch), retire_pool(o);
        }
        imp = imported_add(p);
        guest_pages_set(p.lo, p.hi, p.address);
        if (chunk) prefetch_chunks(p.lo, p.hi, chunk);
        if (overlapped.size() > 1) {
            static int logged_merge;
            if (logged_merge++ < 16)
                logf("vk: pool %#llx..%#llx took in %zu pools", (unsigned long long)p.lo, (unsigned long long)p.hi,
                     overlapped.size());
        }
        static int logged;
        if (logged++ < 8 || imported.size() % 256 == 0) logf("vk: %zu pools imported", imported.size());
    }
    if (imp->second.prefetched) {   // (first bound: its neighbours ahead in turn)
        imp->second.prefetched = false;
        st_prefetch_used++;
        prefetch_chunks(imp->second.lo, imp->second.hi, chunk);
        static int64_t logged_at;
        if (st_prefetch_used / 16 != logged_at) {
            logged_at = st_prefetch_used / 16;
            logf("vk: chunks imported ahead: %lld sent, %lld bound, %lld dropped; %lld waits", (long long)st_prefetch_sent,
                 (long long)st_prefetch_used, (long long)st_prefetch_dropped, (long long)st_bg_waits);
        }
    }
    imp->second.range_lo = range.lo, imp->second.range_hi = range.hi;
    return last_pool = &imp->second;
}

// ---- the VRAM mirror (after totk-pc-recomp's vk_memory.c) ----
// An imported pool keeps vertices, indices, constants and storage buffers in system memory, so a discrete GPU reads
// them over PCIe on every draw (TOTK measured ~110 ms of GPU time a frame on a 3090 Ti). A pool bound as a buffer
// gets a device-local copy, made at the start of the next submission, and from then on its bindings use the copy at
// the same offsets:
// - CPU -> mirror: at a submission's first use of a mirror, the pages the CPU wrote since its last sync (the page
//   write generations) are copied in, in the recording's upload command buffer (submitted right before it).
// - GPU -> guest: what the submission's shaders and copies may write into a mirror is copied back into the imported
//   pool at the submission's end, so the CPU (and everything that reads guest memory) sees it.
// RX_VK_MIRROR=0: no mirrors; RX_VK_MIRROR_MB: their budget (default by the card: mirror_budget.h auto_bytes).
bool mirror_on() {
    static const int on = [] {
        // (device_policy.h mirrors: a discrete card, then the switch)
        // (hookshot) on for GCN too (RX 400/500, Vega). k10 turned them off there (an RX 580 ran Kokiri Forest at 20 fps with them
        // and 24.5 without), but on today's renderer an RX 580 runs 28 to 54% faster with them in every area (Kokiri Forest 20.9
        // -> 29.6 fps, Hyrule Field 27.5 -> 42.3), so every discrete card has them; RX_VK_MIRROR=0 still turns them off
        const int v = dev.policy.mirrors;
        logf("vk: VRAM mirror of buffer pools %s", v ? "on" : "off");
        return v;
    }();
    return on != 0;
}

/// (hookshot, smoothness) A read binding in a chunk not imported yet: its import goes to the background thread and the binding is
/// served from a copy in the recording's stream, rather than the replay waiting for the import. Imports on the replay thread were
/// the main hitch left in play (54-61 a run of the verification walks, 5-380 ms each, 2026-11-09). True: an import of the range is
/// under way (sent now or before) and nothing covers it yet; false: the synchronous path as before (covered, outside the game's
/// pools, or next to an import that would have to grow). RX_VK_ASYNC_IMPORT=1 (opt-in: background imports slowed AMD before) (pools_mu held)
int64_t st_async_sent, st_async_copies;
bool import_in_background(uint64_t guest, uint64_t size) {
    if (guest_pools.empty() || !size || !dev.external_memory_host) return false;
    bg_drain();
    auto it = std::upper_bound(guest_pools.begin(), guest_pools.end(), guest, [](uint64_t a, const GuestPool& p) { return a < p.lo; });
    if (it == guest_pools.begin()) return false;
    --it;
    if (guest < it->lo || guest >= it->hi || size > it->hi - guest) return false;
    const uint64_t chunk = import_chunk();
    if (!chunk || guest + size > UINT64_MAX - chunk) return false;
    for (auto c = imported.upper_bound(guest); c != imported.begin();) {   // (covered: by the last one at or below it, or further down)
        --c;
        if (c->second.hi >= guest + size) return false;
        if (import_reach::beyond(live_reach, c->second.lo, guest + size)) break;
    }
    const uint64_t lo = guest / chunk * chunk, hi = (guest + size + chunk - 1) / chunk * chunk;
    for (auto o = imported.lower_bound(hi); o != imported.begin();) {   // (an import there: the synchronous path grows it)
        --o;
        if (o->second.hi <= lo) {
            if (import_reach::beyond(live_reach, o->second.lo, lo)) break;
            continue;
        }
        return false;
    }
    std::lock_guard lk(bg_mu);
    for (const auto& [x, end] : bg_inflight)
        if (x >= lo && x < hi) return true;   // (under way)
    if (bg_inflight.size() >= 16) return false;
    bg_start();
    bg_inflight.emplace(lo, hi);
    bg_queue.push_back({lo, hi});
    bg_cv.notify_one();
    if (st_async_sent++ % 32 == 0)
        logf("vk: imports in the background: %lld sent, %lld bindings served from a copy meanwhile", (long long)st_async_sent,
             (long long)st_async_copies);
    return true;
}
} // namespace
void import_stats_log(const char* when) {
    logf("vk: imports at %s: %lld made, %lld refused by the driver, %lld lookups waited on a held range", when,
         (long long)st_imports_made.load(), (long long)st_imports_refused.load(), (long long)st_imports_held.load());
}
bool pools_mirrored() { return mirror_on(); }
uint64_t test_vram_bytes(uint64_t heap) {
    static const uint64_t cap = getenv("RX_VK_TEST_VRAM_MB") ? (uint64_t)std::max(0, atoi(getenv("RX_VK_TEST_VRAM_MB"))) << 20 : 0;
    return cap && heap > cap ? cap : heap;
}
uint64_t device_vram_bytes() {
    uint64_t vram = 0;
    for (uint32_t h = 0; h < dev.memory.memoryHeapCount; h++)
        if (dev.memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram = std::max(vram, (uint64_t)dev.memory.memoryHeaps[h].size);
    return test_vram_bytes(vram);
}
// (RX_VK_MSYNC_PASS: defined after memory_end_submission)
bool msync_pass_add(Pool& p, VkBuffer buf, const std::vector<VkBufferCopy>& rs);
void msync_pass_flush();
void msync_pass_report();
extern bool msync_batching;
namespace {
// (hookshot) room past the budget for the one hottest pool the full mirrors left out, while VRAM has it (mirrors_by_heat)
uint64_t mirror_heat_extra = 0;
uint64_t mirror_budget() {
    // (a quarter of the card, at most 2 GB under 15 GB, else 3 GB; at most 256 MB on cards of 4.5 GB or less: mirror_budget.h.
    // No menu row or Quality preset sets it; RX_VK_MIRROR_MB=<MB> for a test. Plus mirrors_by_heat's room for its hottest pool)
    static const uint64_t b = [] {
        if (const char* e = getenv("RX_VK_MIRROR_MB")) return (uint64_t)atoi(e) << 20;
        const uint64_t vram = device_vram_bytes();
        // Cards of 6 GB or less: allocate up to 1700 MB to hold open-area peaks without falling back to PCIe reads.
        if (vram && vram <= (6ull << 30) + (512ull << 20)) return std::min<uint64_t>(1700ull << 20, vram / 3);
        return mirror_rule::auto_bytes(vram);
    }();
    return b + mirror_heat_extra;
}
uint64_t mirror_bytes = 0, mirror_epoch = 0;
// (hookshot, mirrors_by_heat) the mirrors, past RX_VK_MIRROR_SMALL_PCT (70) % of the budget, only for a pool the GPU binds much
// (mirror_hot_bytes): the rest stays for a hot pool. First come, first served, a hundred small pools bound at the start filled the
// 512 MB of a 4 GB card, most of them written by the GPU (never given back), and a 132 MB pool the game's indirect dispatches read
// stayed in system memory in most starts: the dispatches 4.6 ms a frame against 2.8 with it in VRAM (55 fps against 59.5, standing,
// Kokiri Forest)
uint64_t mirror_hot_bytes() {   // (RX_VK_MIRROR_HOT_MB, 64: a pool's heat, bound bytes decayed, from which it is hot)
    static const uint64_t hot = (uint64_t)(getenv("RX_VK_MIRROR_HOT_MB") ? std::max(1, atoi(getenv("RX_VK_MIRROR_HOT_MB"))) : 64) << 20;
    return hot;
}
uint64_t mirror_small_cap() {
    static const uint64_t pct = [] {
        const char* e = getenv("RX_VK_MIRROR_SMALL_PCT");
        const int v = e ? atoi(e) : 70;
        return (uint64_t)(v > 0 && v < 100 ? v : 100);
    }();
    return pct >= 100 ? UINT64_MAX : mirror_budget() / 100 * pct;
}
bool mirror_upload = false;   // (a submission is being recorded: its upload command buffer takes the syncs)
std::vector<Pool*> mirror_wanted, mirror_synced;
struct WriteBack {
    Pool* pool;
    uint64_t lo, hi;   // offsets in the pool
};
std::vector<WriteBack> writebacks;

// ---- (hookshot) sparse mirrors (RX_VK_MIRROR_SPARSE, on by default, sparse_mirror.h): memory only in the blocks a game
// pool is in. The mirrors held 243-533 MB of blocks outside every game pool, of 1.1-1.6 GB (DLAA 2560 x 1440, 4 MB import
// chunks; the most after an area change): chunk padding, and pools the game let go. Such a block's pages point at guest
// memory in the page table, so the shaders' pointers reach the game's memory itself there, as in a pool with no mirror; a
// binding that reaches one is served from guest memory too, and the block gets memory at the next submission
// (sparse_grow), filled as mirror_make fills a mirror; unless every such block it reaches is outside every game pool (a
// binding with a guessed size that runs past a pool's end): the mirror serves that one as it is. A block every game pool
// has left is given back (sparse_release) once the GPU is past the work that may have reached it. Needs sparse residency
// for buffers with strict non-resident access (reads 0, writes dropped), which nothing of the game's reaches. On by
// default (the maintainer's call; measured on an RTX 4070 SUPER with 6 and 8 GB allowances and an RTX 3060 Ti 8 GB);
// RX_VK_MIRROR_SPARSE=0 (or off, false, no: hookshot_env.h's rule): whole mirrors, as before. Off on its own, and logged,
// where the device lacks strict sparse residency for buffers ----
bool mirror_sparse_on() {
    static const bool on = [] {
        const char* v = getenv("RX_VK_MIRROR_SPARSE");
        const bool asked = !v || !env_value_off(v);
        // (the mirrors bind on the replay's queue: where that family binds no sparse memory (RADV keeps sparse binding on a
        // family of its own, dev.sparse_queue, which only the dma-buf pools use), a bind there is not allowed and the GPU hangs)
        const bool own_queue = dev.sparse_queue == dev.queue;
        const bool can = dev.sparse_buffers && own_queue && dev.features.sparseResidencyBuffer &&
                         dev.props.sparseProperties.residencyNonResidentStrict;
        if (asked)
            logf("vk: sparse VRAM mirrors (RX_VK_MIRROR_SPARSE): %s",
                 can ? "on"
                 : dev.sparse_buffers && !own_queue ? "not on this device (the replay's queue family binds no sparse memory): whole mirrors"
                                                    : "not on this device (no strict sparse residency for buffers): whole mirrors");
        return asked && can;
    }();
    return on;
}
uint64_t mirror_size(const Pool& p) {
    return p.sparse ? p.resident_bytes : p.hi - p.lo;
}
uint64_t sparse_block(const Pool& p) {
    return p.sparse_block_bytes;
}
VkMemoryRequirements sparse_req_of(const Pool& p) {
    VkMemoryRequirements r{};
    r.size = p.sparse_size, r.alignment = p.sparse_block_bytes, r.memoryTypeBits = p.sparse_types;
    return r;
}
bool block_has(const Pool& p, size_t i) {
    return std::atomic_ref<uint8_t>(const_cast<uint8_t&>(p.has[i])).load(std::memory_order_acquire) != 0;
}
void block_has_set(Pool& p, size_t i, bool v) {
    std::atomic_ref<uint8_t>(p.has[i]).store(v ? 1 : 0, std::memory_order_release);
}
/// Whether every block [off, off + size) reaches has memory (always, for a mirror that is not sparse)
bool sparse_has(const Pool& p, uint64_t off, uint64_t size) {
    if (!p.sparse) return true;
    size_t a = 0, e = 0;
    sparse_mirror::block_span(off, size, sparse_block(p), p.has.size(), &a, &e);
    for (size_t i = a; i < e; i++)
        if (!block_has(p, i)) return false;
    return true;
}
std::vector<Pool*> sparse_wanted;   // (pools a binding found a block with no memory in: given it at the next submission)
std::unordered_set<VkBuffer> sparse_live;   // (the sparse mirrors not destroyed: a released block's unbinding skips the others)
struct DeadBlock { VkBuffer buffer; Pool* pool; size_t index; VmaAllocation alloc; uint64_t value; uint32_t gen = 0; };   // (pool null: its mirror freed)
std::deque<DeadBlock> dead_blocks;
struct DeadSparse {   // (a freed sparse mirror; binds: the binds' value then)
    VkBuffer buffer;
    uint64_t value;
    std::vector<VmaAllocation> blocks;
    uint64_t binds = 0;
    // (sparse_spare.h: what the buffer is kept as once the GPU is past it, made 0: destroyed; and what is bound to it, unbound then)
    sparse_spare::Spare<VkBuffer> spare{};
    std::vector<VkSparseMemoryBind> unbind;
};
std::deque<DeadSparse> dead_sparse;
/// (hookshot, sparse_spare.h) The freed sparse mirrors' buffers kept for the next mirror of their size, and the address
/// space they may hold together, in bytes: RX_VK_SPARSE_SPARE_MB (kDefaultMb; 0: none kept, every buffer destroyed at once,
/// as before). spare_doomed: buffers to destroy once the binds' timeline has reached their value. pools_mu held
sparse_spare::List<VkBuffer> sparse_spares;
std::deque<std::pair<uint64_t, VkBuffer>> spare_doomed;
int64_t st_spare_taken = 0, st_spare_made = 0, st_spare_kept = 0, st_spare_destroyed = 0;
uint64_t sparse_spare_cap() {
    static const uint64_t cap = [] {
        const char* e = getenv("RX_VK_SPARSE_SPARE_MB");
        const uint64_t asked = (uint64_t)(e && *e ? std::max(0, atoi(e)) : (int)sparse_spare::kDefaultMb) << 20;
        // (never more than a quarter of the address space the device has for all its sparse resources together)
        const uint64_t cap = std::min<uint64_t>(asked, dev.props.limits.sparseAddressSpaceSize / 4);
        if (cap)
            logf("vk: sparse mirrors' buffers are kept for the next mirror of their size, up to %llu MB of address space "
                 "(RX_VK_SPARSE_SPARE_MB=0: destroyed at once, as before)", (unsigned long long)(cap >> 20));
        return cap;
    }();
    return cap;
}
/// A sparse buffer for a mirror of n bytes: the spare of that size, else a new one (vk_vma.cpp). after: a spare's value on
/// the binds' timeline (0 for a new one), which goes with it if it is given up again
bool sparse_buffer_for(uint64_t n, VkBufferUsageFlags usage, VkBuffer& buffer, VkMemoryRequirements& req, uint64_t& after) {
    sparse_spare::Spare<VkBuffer> spare;
    after = 0;
    if (sparse_spare_cap() && sparse_spares.take(n, &spare)) {
        buffer = spare.buffer, after = spare.after;
        req = {};
        req.size = spare.size, req.alignment = spare.block, req.memoryTypeBits = spare.types;
        st_spare_taken++;
        return true;
    }
    if (!create_sparse_buffer(n, usage, buffer, req)) return false;
    st_spare_made++;
    return true;
}
/// A sparse buffer no longer wanted, with nothing bound to it from `after` on: destroyed once the GPU has reached that
/// (sparse_reap), at once with no spares (as before)
void sparse_buffer_doom(VkBuffer buffer, uint64_t after) {
    if (!sparse_spare_cap()) {
        destroy_sparse_buffer(buffer);
        return;
    }
    spare_doomed.push_back({after, buffer});
}
/// The same, kept as a spare where one may be: a buffer mirror_make gave up before it bound anything to it
void sparse_buffer_unused(VkBuffer buffer, uint64_t n, const VkMemoryRequirements& req, uint64_t after) {
    std::vector<sparse_spare::Spare<VkBuffer>> gone;
    if (sparse_spare_cap() && sparse_spares.put({buffer, n, req.size, req.alignment, req.memoryTypeBits, after}, sparse_spare_cap(), &gone))
        st_spare_kept++;
    else
        sparse_buffer_doom(buffer, after);
    for (const auto& g : gone) sparse_buffer_doom(g.buffer, g.after);
}
int64_t st_sparse_mixed = 0, st_sparse_past = 0, st_sparse_in_place = 0, st_sparse_grown = 0, st_sparse_released = 0, st_sparse_binds = 0, st_sparse_refused = 0;
double st_sparse_bind_ms = 0;
// (the binds' own timeline: a bind signals the next value, every submission of recordings waits for the latest one on the GPU
// (memory_sparse_wait), and memory an unbind let go is freed once the GPU's count has passed it. Each bind waited for on the
// CPU waited for the whole queue, ~3 ms; ~170-390 of them in a run made the load's longest frame 1.0-1.2 s against 0.6-0.9)
VkSemaphore sparse_tl = VK_NULL_HANDLE;
std::atomic<uint64_t> sparse_tl_value{0};   // (the latest value a bind signals; written under dev.present_mu)
std::deque<std::pair<uint64_t, std::vector<VmaAllocation>>> sparse_frees;   // (unbound memory, freed once the count passes)
uint64_t sparse_tl_done() {
    uint64_t v = 0;
    if (sparse_tl) vkGetSemaphoreCounterValue(dev.device, sparse_tl, &v);
    return v;
}
/// Several sparse buffers' binds (memory or none) in one call, on the queue (pools_mu held, as an import's). *signaled: the
/// value the binds' timeline reaches when they are done. Not waited for on the CPU
bool sparse_bind_infos(std::vector<VkSparseBufferMemoryBindInfo> infos, uint64_t* signaled = nullptr) {
    if (infos.empty()) return true;
    const auto t0 = std::chrono::steady_clock::now();
    // The driver accepts a contiguous range in one VkDeviceMemory. Keep the original
    // per-block lists for the caller's rollback and eventual independent unbinds.
    size_t capacity = 0;
    for (const auto& info : infos)
        if (info.bindCount > 1) capacity += info.bindCount;
    std::vector<VkSparseMemoryBind> ranges;
    ranges.reserve(capacity);   // the pointers installed in earlier infos must stay valid
    for (auto& info : infos) {
        if (info.bindCount < 2) continue;
        const size_t first = ranges.size();
        for (uint32_t i = 0; i < info.bindCount; ++i) {
            const auto& b = info.pBinds[i];
            if (ranges.size() > first) {
                auto& last = ranges.back();
                if (last.resourceOffset + last.size == b.resourceOffset && last.memory == b.memory &&
                    last.flags == b.flags && (!b.memory || last.memoryOffset + last.size == b.memoryOffset)) {
                    last.size += b.size;
                    continue;
                }
            }
            ranges.push_back(b);
        }
        info.bindCount = (uint32_t)(ranges.size() - first);
        info.pBinds = ranges.data() + first;
    }
    VkResult res;
    uint64_t next = 0;
    {   // (as the sparse textures' and the dma-buf pools' binds, on the queue under its lock. The blocks bound are reached by no
        // work submitted before: a block with no memory was left to guest memory in the page table and in bindings, one given
        // back and wanted again takes back the memory it still has, one unbound is past its GPU work; and every submission of
        // recordings after this waits for it)
        std::lock_guard qk(dev.present_mu);
        if (!sparse_tl) {   // (made under dev.present_mu, which memory_sparse_wait reads it under on the replay's submit paths)
            VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            ci.pNext = &ti;
            VkSemaphore made = VK_NULL_HANDLE;
            if (vkCreateSemaphore(dev.device, &ci, nullptr, &made) != VK_SUCCESS) return false;
            sparse_tl = made;
        }
        queue_flush_locked();
        next = sparse_tl_value.load(std::memory_order_relaxed) + 1;
        VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &next;
        VkBindSparseInfo si{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
        si.pNext = &tsi;
        si.bufferBindCount = (uint32_t)infos.size();
        si.pBufferBinds = infos.data();
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &sparse_tl;
        res = vkQueueBindSparse(dev.queue, 1, &si, VK_NULL_HANDLE);
        if (res == VK_SUCCESS) sparse_tl_value.store(next, std::memory_order_relaxed);
    }
    if (signaled) *signaled = res == VK_SUCCESS ? next : 0;
    st_sparse_binds++;
    st_sparse_bind_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (res != VK_SUCCESS) {
        static int logged;
        if (logged++ < 8) logf("vk: sparse mirrors: %zu buffers' blocks not bound (VkResult %d)", infos.size(), (int)res);
    }
    return res == VK_SUCCESS;
}
} // namespace
/// (hookshot, RX_VK_MIRROR_SPARSE) The wait a submission of recordings takes for the sparse mirrors' binds before it, on the
/// GPU; false: none yet (dev.present_mu held, as the binds are made)
bool memory_sparse_wait(VkSemaphoreSubmitInfo* w) {
    const uint64_t v = sparse_tl_value.load(std::memory_order_relaxed);
    if (!sparse_tl || !v) return false;
    *w = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    w->semaphore = sparse_tl, w->value = v, w->stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    return true;
}
namespace {
bool sparse_bind(VkBuffer buffer, std::vector<VkSparseMemoryBind>& binds) {
    if (binds.empty()) return true;
    std::vector<VkSparseBufferMemoryBindInfo> infos{{buffer, (uint32_t)binds.size(), binds.data()}};
    return sparse_bind_infos(std::move(infos));
}
/// A new sparse mirror's memory, in the blocks live marks; false when VMA has none for one or the binding failed (nothing
/// kept then: the buffer is the caller's)
bool sparse_fill(Pool& p, VkBuffer buffer, const VkMemoryRequirements& req, const std::vector<uint8_t>& live) {
    const uint64_t b = req.alignment;
    const size_t nb = (size_t)((req.size + b - 1) / b);
    std::vector<VmaAllocation> blocks(nb, nullptr);
    std::vector<VkSparseMemoryBind> binds;
    bool ok = true;
    for (size_t i = 0; i < nb && i < live.size(); i++) {
        if (!live[i]) continue;
        VkDeviceMemory m = VK_NULL_HANDLE;
        uint64_t off = 0;
        if (!alloc_sparse_block(req, blocks[i], m, off)) {
            ok = false;
            break;
        }
        const uint64_t at = (uint64_t)i * b;
        binds.push_back({at, std::min(b, req.size - at), m, off, 0});
    }
    if (ok) ok = sparse_bind(buffer, binds);
    if (!ok) {
        for (VmaAllocation a : blocks) free_sparse_block(a);
        return false;
    }
    p.sparse = true;
    p.sparse_block_bytes = req.alignment, p.sparse_size = req.size, p.sparse_types = req.memoryTypeBits;
    p.has.assign(nb, 0);
    p.wanted.assign(nb, 0);
    p.block_dead_since.assign(nb, 0);
    p.released.assign(nb, nullptr);
    p.release_gen.assign(nb, 0);
    p.last_bound.assign(nb, 0);
    p.any_wanted = false;
    p.resident_bytes = 0;
    for (size_t i = 0; i < nb; i++)
        if (blocks[i]) p.has[i] = 1, p.resident_bytes += b;
    p.blocks = std::move(blocks);
    p.mirror = buffer;
    p.mirror_alloc = nullptr;
    sparse_live.insert(buffer);
    return true;
}
/// The page table over a sparse mirror's blocks [first, end): its own pages where a block has memory, guest memory's where
/// it has none (deferred into the recording's uploads, as every change of the table)
void sparse_table(const Pool& p, VkDeviceAddress mirror_address, size_t first, size_t end) {
    const uint64_t b = sparse_block(p);
    for (size_t i = first; i < end;) {
        const bool has = block_has(p, i);
        size_t j = i + 1;
        while (j < end && block_has(p, j) == has) j++;
        const uint64_t lo = p.lo + (uint64_t)i * b, hi = std::min(p.hi, p.lo + (uint64_t)j * b);
        if (hi > lo) guest_pages_set(lo, hi, has ? mirror_address + (lo - p.lo) : p.address + (lo - p.lo));
        i = j;
    }
}
VkDeviceAddress buffer_address(VkBuffer buffer) {
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = buffer;
    return vkGetBufferDeviceAddress(dev.device, &da);
}
/// A binding reached blocks [off, off + size): they are in use, whatever game pool is there (a binding with a guessed size
/// runs past a pool's end), and are not given back while bindings keep reaching them (sparse_release). Without the lock too
void sparse_touch(Pool& p, uint64_t off, uint64_t size) {
    if (!p.sparse) return;
    size_t a = 0, e = 0;
    sparse_mirror::block_span(off, size, sparse_block(p), p.last_bound.size(), &a, &e);
    for (size_t i = a; i < e; i++) std::atomic_ref<uint64_t>(p.last_bound[i]).store(mirror_epoch, std::memory_order_relaxed);
}
uint64_t last_bound_of(const Pool& p, size_t i) {
    return std::atomic_ref<uint64_t>(const_cast<uint64_t&>(p.last_bound[i])).load(std::memory_order_relaxed);
}
/// Whether no block [off, off + size) reaches has memory (one that does too is counted)
bool sparse_none(const Pool& p, uint64_t off, uint64_t size) {
    size_t a = 0, e = 0;
    sparse_mirror::block_span(off, size, sparse_block(p), p.has.size(), &a, &e);
    for (size_t i = a; i < e; i++)
        if (block_has(p, i)) {
            st_sparse_mixed++;
            return false;
        }
    return true;
}
/// Whether every block [off, off + size) reaches that has no memory lies outside every game pool (pools_mu held): the usual
/// mixed binding, one with a guessed size that runs past a pool's end. The mirror serves it as it is then: those blocks read 0
/// and drop writes (residencyNonResidentStrict), and nothing of the game's is there. Within a pool, a block with none holds
/// the game's bytes in guest memory only, and the binding is served from there (sparse_want)
bool sparse_gaps_outside_pools(const Pool& p, uint64_t off, uint64_t size) {
    const uint64_t b = sparse_block(p);
    size_t a = 0, e = 0;
    sparse_mirror::block_span(off, size, b, p.has.size(), &a, &e);
    if (e <= a) return true;
    // (the blocks [a, e) within the pool, block-aligned from p.lo as the mirror's: live[i - a] is block i's)
    const uint64_t lo = p.lo + (uint64_t)a * b, hi = std::min(p.hi, p.lo + (uint64_t)e * b);
    if (hi <= lo) return true;
    thread_local std::vector<uint8_t> live;   // (kept: this runs at each such binding, and at write-backs that reach one)
    sparse_mirror::live_blocks(lo, hi, b, guest_pools.begin(), guest_pools.end(), live);
    for (size_t i = a; i < e; i++)
        if (!block_has(p, i) && i - a < live.size() && live[i - a]) return false;
    return true;
}
/// A binding reached blocks with no memory: they get it at the next submission (pools_mu held)
void sparse_want(Pool& p, uint64_t off, uint64_t size) {
    size_t a = 0, e = 0;
    sparse_mirror::block_span(off, size, sparse_block(p), p.wanted.size(), &a, &e);
    for (size_t i = a; i < e; i++)
        if (!p.blocks[i]) p.wanted[i] = 1;
    if (!p.any_wanted) p.any_wanted = true, sparse_wanted.push_back(&p);
}
void cap_pages_over(Pool& p, uint64_t lo, uint64_t hi, std::map<VkBuffer, std::vector<VkBufferCopy>>& over);   // (below, mirror_make's)
void sparse_grow_all(const std::vector<Pool*>& pools);   // (below)
void sparse_want_live();
void sparse_release();
void sparse_reap();

// ---- Write-backs left out: the game's pools the CPU does not read. Every range a submission's shaders write is
// copied from the mirror back to guest memory (69 MB a frame in Hyrule Field, 1.6 ms of GPU time), almost all of it into
// CPU_CACHED pools (NVN flags 0x24) the CPU never reads back: on the console a CPU read of GPU results in a cached pool needs
// nvnMemoryPoolInvalidateMappedRange first, and CPU_NO_ACCESS pools cannot be read at all. So a game pool's write-backs are
// left in the mirror once it has been written back for kLazyWarm submissions with neither an invalidate of it (the CPU reads
// it: memory_cpu_reads) nor a CPU write to it (a mirror sync's dirty page: a stale page from guest memory would go over the
// GPU's words). Either makes the pool eager for good. Not left out: CPU_UNCACHED pools (read without an invalidate), ranges
// outside any game pool, ranges where a live texture's storage lies (re-uploaded from guest memory). A mirror holding such
// writes is kept (not given back: the GPU would read stale guest memory). The counters (every 600 submissions) say how often
// a read or a write came after a range was left out (once each per pool at most: then eager). RX_VK_WB_EAGER=1: all, as before
struct NvnPool {
    uint64_t hi = 0;
    uint32_t flags = 0;
    uint64_t first_wb = 0;   // (mirror_epoch + 1 of its first write-back)
    uint64_t made_ns = 0;    // (host->now_nswhen the game made it)
    bool eager = false, skipped = false;
    uint8_t why = 0;   // (eager: 1 read by the CPU, 2 written by it)
};
std::mutex nvn_pools_mu;   // (after pools_mu when both)
std::map<uint64_t, NvnPool> nvn_pools;   // by guest lo
std::unordered_set<uint32_t> cpu_pages, skipped_pages;   // (guest pages the CPU writes, and the ones whose GPU writes were left out; nvn_pools_mu held)
constexpr uint64_t kLazyWarm = 600;
std::atomic<uint64_t> nvn_pools_gen{1};   // (bumped when a pool is made: lazy_cpu_page's cache)
int64_t st_lazy_bytes = 0, st_lazy_late = 0, st_lazy_cpu = 0, st_lazy_pools = 0;
std::atomic<int64_t> st_lazy_direct{0};
std::atomic<int64_t> st_direct_fast{0}, st_direct_fast_bytes{0};   // (guest-memory binds on the lock-free path, not in direct_by_pool)
int64_t st_kept_by[6];   // (bytes written back, by why: 0 no game pool, 1 read, 2 written by the CPU, 3 uncached, 4 warming, 5 texture)   // (bindings without the lock too)
bool wb_lazy_on() {
    static const bool v = getenv("RX_VK_WB_EAGER") && getenv("RX_VK_WB_EAGER")[0] == '0';   // (lazy only with RX_VK_WB_EAGER=0: off by default here)
    return v;
}
NvnPool* nvn_pool_at(uint64_t g, uint64_t* lo) {   // (nvn_pools_mu held)
    auto it = nvn_pools.upper_bound(g);
    if (it == nvn_pools.begin()) return nullptr;
    --it;
    if (g >= it->second.hi) return nullptr;
    if (lo) *lo = it->first;
    return &it->second;
}
// (a mirror sync copies a page the CPU wrote: that page is written back from now on; if its GPU writes were left out, the
// copy from guest memory goes over them (counted). pools_mu held)

// (RX_VK_MSYNC_FAST) lazy_cpu_page for a mirror sync's dirty pages at once, in their order: the lock taken once, and the
// game pool looked up only when a page leaves the last one found (a sync's pages are in address order). The same sets result.
// A page already put in cpu_pages here is marked in a bitmap (by chunks of 64K pages) and skipped: cpu_pages never loses a page,
// and skipped_pages takes only pages cpu_pages does not hold (lazy_filter), so for such a page both calls below change nothing
// (the castle-move pre-sync's ~8,600 dirty pages a frame were each a hash insert and erase)
std::vector<std::unique_ptr<uint64_t[]>> cpu_page_bits;   // (nvn_pools_mu held)
// (2026-11-13) What the release of a mirror holding left-out write-backs does with one left-out page : copy the mirror's
// bytes back into guest memory only while they are still the newest bytes of a live pool's page.
enum { kRelCopy, kRelDead, kRelWritten, kRelCpuPage, kRelTexture, kRelNoStage, kRelKinds };
int64_t st_release_by[kRelKinds];   // (pages, by verdict at the release; nvn_pools_mu held)
int64_t st_apply_by[kRelKinds];     // (page pieces, by verdict at the copy into guest memory; pools_mu held)
/// (2026-11-14) a release's staged bytes: the host buffer the GPU copies the mirror's left-out pages into, and where each
/// piece (one page at most) goes in guest memory; written there by the CPU once the submission value is done (lazy_apply)
struct LazyStage {
    struct Piece { uint64_t guest, off; uint32_t n; };
    uint64_t value = 0;
    uint32_t synced = 0;   // (the mirror's synced_gen: a page written since is newer in guest memory)
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    void* map = nullptr;
    std::vector<Piece> pieces;
};
std::vector<LazyStage> lazy_stages;   // (pools_mu held)
int64_t st_lazy_staged = 0;
/// the last check of a staged page before the CPU writes it (the host then checks its pool is live and the page not written
/// since the last collection, under the lock pool Finalize takes: hos_gpu_write_live): gen and synced as at the release;
/// cpu_page: a mirror sync saw the CPU write it since
int lazy_apply_verdict(uint32_t gen, uint32_t synced, bool cpu_page) {
    if (gen > synced) return kRelWritten;
    if (cpu_page) return kRelCpuPage;
    return kRelCopy;
}
/// the host's answer for a page it was asked to write (hos_gpu_write_live: 1 written, 2 written by the CPU since, else no live
/// pool), as a verdict; no host function: nothing is written
int lazy_host_verdict(int host_result) {
    return host_result == 1 ? kRelCopy : host_result == 2 ? kRelWritten : kRelDead;
}
/// live: the page's pool record is a pool the game still has (not finalized: its memory may be the game's heap again);
/// gen: the page's write generation, synced: its mirror's (a newer one is a write since the mirror's last sync: guest
/// memory is newer than the GPU's bytes); cpu_page: a mirror sync saw the CPU write it; texture: a live texture's storage
int lazy_release_verdict(bool live, uint32_t gen, uint32_t synced, bool cpu_page, bool texture) {
    if (!live) return kRelDead;
    if (gen > synced) return kRelWritten;
    if (cpu_page) return kRelCpuPage;
    if (texture) return kRelTexture;
    return kRelCopy;
}
// (hookshot, 11-07) The counter reports the replay writes into guest memory on the CPU (vk_replay.cpp NOP_REPORT_COUNTER), by
// guest address: a mirrored pool's write-back copied the GPU's mirror over them at the end of the submission, the mirror's
// stale words over the replay's newer ones, and the next render condition read those: a lighting pass left out, the whole lit
// scene black for a frame (the one-frame dark flashes, every 15-30 frames; 17 in 40 s of Kokiri, 0 with mirrors off). They are
// left out of the write-back now.
std::mutex cpu_words_mu;
// (two sorted sets with no word in both, and the words noted since the last merge: those go into the recent set, which
// goes into the big one only once it is an eighth of its size. A std::set took a node a word and freed 65536 of them at
// once when it filled, ~4% of the replay thread in a Kokiri profile; one sorted vector with every write-back's new words
// merged in moved the whole of it each time, ~2.4% on a 3900X there (11-08): report words come at new addresses all the time)
std::vector<uint64_t> cpu_words, cpu_words_recent, cpu_words_new;
void sorted_merge(std::vector<uint64_t>& into, const uint64_t* b, const uint64_t* e) {
    const size_t mid = into.size();
    into.insert(into.end(), b, e);
    std::inplace_merge(into.begin(), into.begin() + (ptrdiff_t)mid, into.end());
}
void cpu_words_merge() {   // (holding cpu_words_mu)
    if (cpu_words_new.empty()) return;
    std::sort(cpu_words_new.begin(), cpu_words_new.end());
    cpu_words_new.erase(std::unique(cpu_words_new.begin(), cpu_words_new.end()), cpu_words_new.end());
    // (only the words not kept already: the same report words come every frame)
    size_t fresh = 0;
    for (const uint64_t w : cpu_words_new)
        if (!std::binary_search(cpu_words.begin(), cpu_words.end(), w) &&
            !std::binary_search(cpu_words_recent.begin(), cpu_words_recent.end(), w))
            cpu_words_new[fresh++] = w;
    if (fresh) sorted_merge(cpu_words_recent, cpu_words_new.data(), cpu_words_new.data() + fresh);
    if (cpu_words_recent.size() > std::max<size_t>(4096, cpu_words.size() / 8)) {
        sorted_merge(cpu_words, cpu_words_recent.data(), cpu_words_recent.data() + cpu_words_recent.size());
        cpu_words_recent.clear();
    }
    cpu_words_new.clear();
}
} // namespace
void memory_cpu_owned(uint64_t guest, uint64_t size) {
    // (no mirrors: no write-back will ever cut these words out, and they piled up to 65536 to be sorted, merged and dropped)
    if (!guest || !mirror_on()) return;
    std::lock_guard lk(cpu_words_mu);
    if (cpu_words.size() + cpu_words_recent.size() + cpu_words_new.size() > 65536) {   // (as before: forgotten when it gets this big)
        cpu_words_merge();   // (which leaves cpu_words_new empty)
        if (cpu_words.size() + cpu_words_recent.size() > 65536) cpu_words.clear(), cpu_words_recent.clear();
    }
    for (uint64_t a = guest & ~7ull; a < guest + size; a += 8) cpu_words_new.push_back(a);
    // (the write itself, guest_cpu_write, has told the write-back diff's shadows)
}
// (Hookshot, 2026-11-13) HOS_NOACCESS_RAM (off by default here; =1 on; host/hos_memsplit.c has the whole of it): the pools the game made with
// NVN's CPU_NO_ACCESS flag keep no zero pages in RAM. The back end's part: no import of such a pool's pages for another pool's
// binding (clip_noaccess below), and no upload of a texture whose storage lies in such a pool and is all reserved
// (texture_touch: the image is made cleared to zero, which is what the storage reads as)
void nvn_pool_record_ranges(std::vector<std::pair<uint64_t, uint64_t>>& out) {
    std::lock_guard lk(nvn_pools_mu);
    for (const auto& [lo, np] : nvn_pools) out.push_back({lo & ~4095ull, (np.hi + 4095) & ~4095ull});
}
bool nvn_pool_record_at(uint64_t g, uint64_t* lo, uint64_t* hi, uint32_t* flags, uint64_t* made_ns) {
    std::lock_guard lk(nvn_pools_mu);
    const NvnPool* np = nvn_pool_at(g, lo);
    if (!np) return false;
    *hi = np->hi, *flags = np->flags, *made_ns = np->made_ns;
    return true;
}
bool memory_nvn_pool_of(uint64_t guest, uint64_t* lo, uint64_t* hi, uint32_t* flags) {
    std::lock_guard lk(nvn_pools_mu);
    const NvnPool* np = nvn_pool_at(guest, lo);
    if (!np) return false;
    *hi = np->hi, *flags = np->flags;
    return true;
}
bool noaccess_ram_on() {
    static const bool on = [] {
        const char* e = getenv("HOS_NOACCESS_RAM");
        const char* lazy = getenv("HOS_LAZY_COMMIT");
        return e && e[0] == '1' && !(lazy && lazy[0] == '0') && !env_on("HOS_LARGE_PAGES");   // (off by default here: =1 on)
    }();
    return on;
}
bool memory_in_noaccess_pool(uint64_t guest, uint64_t size) {
    if (!size || guest >= (1ull << 40)) return false;
    std::lock_guard lk(nvn_pools_mu);
    uint64_t lo = 0;
    const NvnPool* np = nvn_pool_at(guest, &lo);
    return np && (np->flags & 0x1) && guest + size <= np->hi;
}
namespace {
// (a write-back's runs with the replay's own words cut out of them: runs of p, offsets in the pool)
void cut_cpu_words(const Pool& p, std::vector<VkBufferCopy>& runs) {
    std::lock_guard lk(cpu_words_mu);
    cpu_words_merge();
    if (cpu_words.empty() && cpu_words_recent.empty()) return;
    std::vector<VkBufferCopy> out;
    for (const VkBufferCopy& r : runs) {
        uint64_t at = r.srcOffset;
        const uint64_t end = r.srcOffset + r.size;
        // A run can start inside an owned word: include that word when cutting the overlap.
        const uint64_t first = (p.lo + at) & ~7ull, last = p.lo + end;
        auto a = std::lower_bound(cpu_words.begin(), cpu_words.end(), first);
        auto b = std::lower_bound(cpu_words_recent.begin(), cpu_words_recent.end(), first);
        for (;;) {   // (both sets' words in the run, in order)
            const bool in_a = a != cpu_words.end() && *a < last, in_b = b != cpu_words_recent.end() && *b < last;
            if (!in_a && !in_b) break;
            const uint64_t w = (in_a && (!in_b || *a < *b) ? *a++ : *b++) - p.lo;
            if (w > at) out.push_back({at, at, w - at});
            at = std::max(at, w + 8);
        }
        if (end > at) out.push_back({at, at, end - at});
    }
    runs.swap(out);
}

// ---- a pool remade larger (a 16 MB chunk the game's next pool runs past): the old one leaves the table but stays
// alive, as this submission's bindings, its mirror syncs and its write-backs still use it. Erasing it left those
// lists pointing into a freed map node: the write-back copied with garbage handles and the device was lost (the
// daytime title scene, whose pools cross chunk ends). Retired pools are kept (a few, never reused). ----
// (Extracted from the table, not erased: the node, and the Pool in it, stay where they were, so the mirror's lists and
// a parallel replay thread's pool cache still point at a live one.)
std::deque<std::map<uint64_t, Pool>::node_type> retired_pools;
// (hookshot, 0.7.9.2) The retired pools nothing needs any more: their import given back, no mirror, no shadow, nothing pending,
// the GPU past them. Each area change retires ~30 pools (798 in a 10-minute tour of six places, lv01_base, every one given
// back) and each stayed in retired_pools for the whole session, walked at every submission (retired_imports_give_back), for
// every range the game frees (release_uncovered: up to 512 a submission while an area loads, under pools_mu), at each of the
// game threads' frees (memory_guest_imported, under pools_mu) and at every mirror made: the longer the game ran, the longer
// each of those walks. Such a node moves here, alive at the same address (pointers to it may be kept elsewhere), and its
// per-page vectors are let go. RX_VK_RETIRED_COMPACT=0: as before (every node stays in retired_pools)
std::deque<std::map<uint64_t, Pool>::node_type> retired_done;
size_t retired_pools_count() { return retired_pools.size() + retired_done.size(); }
std::atomic<int64_t> st_retired_walk_ns{0};   // (time in the walks over retired_pools, for the imports line)
uint64_t st_retired_vec_bytes = 0;            // (per-page vectors let go with the nodes moved to retired_done)
bool retired_idle(const Pool& p) {
    return !p.memory && p.pieces.empty() && !p.buffer && p.mstate == Pool::MIRROR_NONE && !p.mirror && !p.shadow &&
           !p.inherit_pending && !p.trimmed && p.retired_value <= mem_completed && p.last_value <= mem_completed;   // (hookshot: the write-backs are
    // kept by page, not by pool, wb_pages: a pool the GPU is past has none of its own pending)
}
template <class T> uint64_t vec_let_go(std::vector<T>& v) {
    const uint64_t b = v.capacity() * sizeof(T);
    std::vector<T>().swap(v);
    return b;
}
void retired_compact() {   // (pools_mu held)
    static const bool on = !getenv("RX_VK_RETIRED_COMPACT") || atoi(getenv("RX_VK_RETIRED_COMPACT")) != 0;
    if (!on) return;
    {   // (nothing should bring a node back to use: none is reachable from the pools' table or the caches emptied at its
        // retirement. Checked anyway, on the newest 256 moved, every 600 submissions: one in use goes back to the walked list)
        static uint64_t checked_at;
        if (mirror_epoch - checked_at >= 600 && !retired_done.empty()) {
            checked_at = mirror_epoch;
            const size_t from = retired_done.size() > 256 ? retired_done.size() - 256 : 0;
            for (size_t i = from; i < retired_done.size();) {
                if (retired_idle(retired_done[i].mapped())) { i++; continue; }
                static int told;
                if (told++ < 4) logf("vk: a retired pool %#llx in use again: walked again", (unsigned long long)retired_done[i].mapped().lo);
                retired_pools.push_back(std::move(retired_done[i]));
                retired_done.erase(retired_done.begin() + (ptrdiff_t)i);
            }
        }
    }
    bool any = false;
    for (const auto& nh : retired_pools)
        if (retired_idle(nh.mapped())) { any = true; break; }
    if (!any) return;
    std::deque<std::map<uint64_t, Pool>::node_type> keep;
    for (auto& nh : retired_pools) {
        Pool& p = nh.mapped();
        if (!retired_idle(p)) { keep.push_back(std::move(nh)); continue; }
        st_retired_vec_bytes += vec_let_go(p.shadow_known) + vec_let_go(p.has) + vec_let_go(p.wanted) +
                                vec_let_go(p.block_dead_since) + vec_let_go(p.released) + vec_let_go(p.release_gen) +
                                vec_let_go(p.last_bound) + vec_let_go(p.refill) + vec_let_go(p.pieces);
        retired_done.push_back(std::move(nh));
    }
    retired_pools.swap(keep);
}
// (hookshot, test) RX_VK_TEST_RETIRED_PAD=<n>: n idle retired pools at the first submission, scattered among other allocations as a
// session's are, to see now what a long session's list costs (with RX_VK_RETIRED_COMPACT=0) and that the compaction takes it away
void retired_test_pad() {   // (pools_mu held)
    static bool done;
    if (done) return;
    done = true;
    const char* e = getenv("RX_VK_TEST_RETIRED_PAD");
    const int n = e ? atoi(e) : 0;
    if (n <= 0) return;
    std::map<uint64_t, Pool> tmp;
    std::vector<void*> junk;
    junk.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        Pool& p = tmp.emplace((uint64_t)i, Pool{}).first->second;
        p.lo = p.hi = 0;
        junk.push_back(malloc(256 + (size_t)(i % 13) * 192));
        retired_pools.push_back(tmp.extract((uint64_t)i));
    }
    for (void* j : junk) free(j);
    logf("vk: test: %d idle retired pools added (RX_VK_TEST_RETIRED_PAD)", n);
}

bool retiring_finalized = false;   // (RX_VK_POOL_FINALIZE: the retirement is finalize_retire's, for the log)
void retire_pool(std::map<uint64_t, Pool>::iterator it) {
    Pool* old = &it->second;
    shadow_free(*old);   // (its write-backs in this submission, if any, copied whole)
    guest_pages_set(old->lo, old->hi, 0);
    old->retired_value = std::max(old->last_value, mem_rec_value);
    old->retired_ms = import_now_ms();   // (the wait before its import is given back counts from here)
    std::erase(mirror_wanted, old);   // (the new pool gets the mirror)
    if (old->any_wanted) std::erase(sparse_wanted, old), old->any_wanted = false;   // (and the blocks a binding wanted)
    if (old->mstate == Pool::MIRROR_WANTED) old->mstate = Pool::MIRROR_NONE;
    // (EvilOOT, 11-07) what the GPU wrote into its mirror and nowhere else (the culling's bare table entries past their first
    // 64 bytes, stores through the page table) was lost here: the new pool's mirror is filled from guest memory. Kokiri's
    // foliage records, written by the GPU while the scene loaded and its pools were remade larger, came out zero: every
    // count 0, the trees' leaf clusters never drawn. So the new mirror inherits them (mirror_make)
    if (old->mstate == Pool::MIRROR_ON) old->inherit_pending = true, old->retired_epoch = mirror_epoch;
    pools_gen++;
    retired_pools.push_back(imported_take(it));
    last_pool = nullptr;
    std::fill(std::begin(pool_cache), std::end(pool_cache), nullptr);
    static int logged;
    if (logged++ < 8 || retired_pools.size() % 64 == 0)
        logf("vk: pool %#llx remade larger (%zu retired)", (unsigned long long)old->lo, retired_pools.size());
}
int64_t st_sync_bytes, st_sync_copies, st_wb_bytes, st_mirrors;
int64_t st_inherit_kept, st_inherit_newer, st_inherit_unfilled;   // (bytes a new mirror took from retired mirrors / from guest memory instead)
int64_t st_sync_live_lost;   // (RX_VK_NO_HOST_IMPORT: live pages of a pool not imported again: left out of a sync)
int64_t st_mirrors_freed = 0;   // (with st_mirrors: equal while no pool has a mirror)
int64_t st_sync_live_pages, st_sync_cap_pages, st_sync_old_pages;   // (mirror syncs' dirty pages: live / this submit's capture / an earlier one's)
std::unordered_map<uint64_t, int64_t> live_by_pool;
std::unordered_map<uint64_t, std::pair<int64_t, int>> direct_by_pool;   // (bindings the GPU reads in guest memory: count, state)
std::unordered_map<uint64_t, uint64_t> direct_bytes_by_pool;           // (and the bytes of those bindings: what the GPU may read over PCIe, a binding's whole range)
int64_t live_seen = 0;   // (pool -> pages its syncs read live, since the last report)

void transfer_barrier(VkCommandBuffer cmd) {
    full_barrier(cmd, "barrier (write-back or sync)");   // (the same barrier; one right after another is left out)
}

// ---- the mirror syncs' scatter copy ----
// The syncs copied ~1570 regions a frame of 4-8 KB each (dirty pages, scattered) with vkCmdCopyBuffer: 1.3 ms of GPU time for
// ~11 MB, ~0.8 us a region whatever its size. Here the regions are cut into entries of at most 4 KB, listed in a host-visible
// ring, and one compute dispatch (shaders/msync_scatter.comp, buffer device addresses) copies them all: a workgroup an entry.
// The presync (memory_sync_mirrors_before_segments) batches every pool's into one dispatch; a sync at a binding is one of its
// own. A full barrier before and after (the upload command buffer's copies, the mirrors' readers). Regions not 16-byte aligned
// go to vkCmdCopyBuffer as before. RX_VK_MSYNC_SCATTER=0: all of them as before
struct ScatterEntry { uint64_t src, dst; uint32_t size, pad[3]; };
static_assert(sizeof(ScatterEntry) == 32);
struct Scatter {
    bool tried = false, ok = false;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkBuffer ring = VK_NULL_HANDLE;
    VmaAllocation ring_alloc = nullptr;
    ScatterEntry* map = nullptr;
    VkDeviceAddress ring_addr = 0;
    size_t pos = 0;
    uint64_t lap_value = 0, prev_lap_value = 0;   // (the newest recording that listed entries in this lap, and in the last one)
} scatter;
constexpr size_t kScatterEntries = 1u << 17;   // (4 MB of entries: ~45 frames of Hyrule Field's)
std::mutex scatter_mu;
thread_local std::vector<ScatterEntry> scatter_pending;
thread_local bool scatter_batch = false;
std::atomic<int64_t> st_scatter_entries{0}, st_scatter_dispatches{0}, st_scatter_bytes{0};
bool scatter_on() {
    static const bool v = env_on("RX_VK_MSYNC_SCATTER");   // (off by default here: =1 on)
    return v;
}
// (hookshot) Upload staging arenas for coalescing mirror_sync capture runs into single contiguous copies:
// ANV / Intel Arc processes hundreds of individual 4 KB copies with heavy command processor latency (~5.5 ms).
// Coalescing adjacent destination pages across capture buffers into a linear host staging buffer reduces
// 1000+ tiny VkBufferCopy commands to < 50 contiguous DMA transfers.
struct MsyncArena {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    uint8_t* mapped = nullptr;
    uint64_t size = 0, used = 0, value = 0;
};
std::vector<MsyncArena> msync_arenas;
constexpr uint64_t kMsyncArenaDefaultSize = 16ull << 20;   // 16 MB arena

struct MsyncSlice {
    VkBuffer buffer = VK_NULL_HANDLE;
    uint64_t offset = 0;
    uint8_t* ptr = nullptr;
    VmaAllocation allocation = nullptr;
};

MsyncSlice msync_stage_alloc(uint64_t bytes, uint64_t align = 256) {
    if (!bytes) return {};
    align = std::max<uint64_t>(align, dev.props.limits.nonCoherentAtomSize);
    const uint64_t done = vk_timeline_completed();
    const uint64_t cur_val = mem_rec_value ? mem_rec_value : 1;
    for (MsyncArena& a : msync_arenas) {
        if (a.value == cur_val) {
            const uint64_t at = (a.used + align - 1) & ~(align - 1);
            if (at + bytes <= a.size) {
                a.used = at + bytes;
                return {a.buffer, at, a.mapped + at, a.allocation};
            }
        }
    }
    for (MsyncArena& a : msync_arenas) {
        if (a.value != cur_val && a.value <= done && a.size >= bytes) {
            a.value = cur_val;
            a.used = bytes;
            return {a.buffer, 0, a.mapped, a.allocation};
        }
    }
    MsyncArena a;
    a.size = std::max(bytes, kMsyncArenaDefaultSize);
    void* m = nullptr;
    if (!create_buffer(a.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, a.buffer, a.allocation, &m) || !m) {
        if (a.buffer) destroy_buffer(a.buffer, a.allocation);
        return {};
    }
    a.mapped = (uint8_t*)m;
    a.used = bytes;
    a.value = cur_val;
    msync_arenas.push_back(a);
    return {a.buffer, 0, a.mapped, a.allocation};
}

// Captured pages in a run are copied from one buffer to one mirror. Combining only contiguous source and
// destination bytes leaves gaps, overlaps and the order of the remaining regions unchanged.
void append_capture_copy(std::vector<VkBufferCopy>& runs, const VkBufferCopy& next) {
    if (!runs.empty()) {
        VkBufferCopy& last = runs.back();
        if (last.size && next.size && last.size <= UINT64_MAX - last.srcOffset &&
            last.size <= UINT64_MAX - last.dstOffset && next.size <= UINT64_MAX - last.size &&
            next.size <= UINT64_MAX - next.srcOffset && next.size <= UINT64_MAX - next.dstOffset &&
            last.srcOffset + last.size == next.srcOffset && last.dstOffset + last.size == next.dstOffset) {
            last.size += next.size;
            return;
        }
    }
    runs.push_back(next);
}

// ---- the mirror syncs' diff: chunks of a dirty page that its mirror already holds, left out of the upload ----
// The syncs copy whole 4 KB pages: every page the CPU wrote since the pool's last sync. On a PCIe 3.0 x4 card (RTX 3050 on an
// i5-10400F board) that was 22-34 MB a frame and 8-12 ms of GPU time (the scatter reads host memory at ~2.8 GB/s). The game
// rewrites many pages with the bytes they held: its per-frame data comes round in rings, and a static object's constants land
// where they were. So each page a sync copies from a capture is hashed by 256-byte chunk (msync_diff.h) and the hashes kept by
// guest page; at the page's next sync only the chunks whose hash changed are copied.
// When the mirror holds the hashed bytes: the hashes are recorded only by a sync outside a parallel replay's segments (those
// syncs' uploads run in the queue's segment order, not in the order the workers record them, so a later record could describe
// an earlier copy); they name the mirror (mirror_id, new at each mirror_make: a new mirror is filled from guest memory and
// retired mirrors) and the submission (rec, the mirror epoch) whose uploads copied them. Anything else that may write the mirror
// there notes the page's gw (the epoch it was recorded in): a written binding (the write-backs' list), a stamp of GPU writes
// (storage writes, copies, transfers), a bare entry's or a wide binding's guessed range, a readback refresh. A submission's
// uploads run before its work, and its GPU writes after them, so the hashes hold while rec > gw. Pages a sync copies another
// way (from guest memory as it is at the replay, a segment's sync, kept for the GPU by RX_VK_PROTECT_BARE) lose theirs.
// Not seen: shader stores through the guest page table past what a binding or a bare entry's guess covers (the syncs already
// copy the CPU's whole page over those today). RX_VK_MSYNC_DIFF=0: off (whole pages, as before), 1: hashed and counted but
// whole pages copied (the measure), 2: changed chunks only (the default on a PCIe link of <= 8 GB/s, see pcie_link_gbps) The cost: hashing the synced pages on the replay
// thread, ~0.13 ms a submission part on an i5-10400F (~30 parts a frame at the Market). RX_VK_MSYNC_DIFF_GROUP=<chunks> (1, 2, 4, 8): copy granularity
// (default 1, 256 bytes). RX_VK_MSYNC_STATS=1: the syncs' bytes by pool and kind every 600 submits, without the diff
struct DiffPage { uint32_t rec, gw, mid, pad; uint64_t h[msync_diff::kChunks]; };
constexpr uint64_t kDiffChunkPages = 1u << 13;   // (32 MB of guest memory a table chunk, 1.1 MB of entries)
std::mutex diff_mu;   // (after pools_mu when both)
std::vector<std::unique_ptr<DiffPage[]>> diff_table;
uint32_t diff_gw_all = 0;   // (a GPU write too large to note by page: every hash older than this epoch is void)
uint32_t diff_mirror_ids = 0;
uint64_t diff_table_bytes = 0;
#ifdef __linux__
// (Linux, any vendor) The card's PCI address from VK_EXT_pci_bus_info, its link and every bridge's above it from sysfs: the
// narrowest one (pcie_link.h). The same maximum-speed, current-width reading as NVML's.
double pcie_link_gbps_sysfs() {
    if (!dev.pci_bus_info || dev.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return -1;
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &pci;
    vkGetPhysicalDeviceProperties2(dev.physical, &p2);
    const auto l = mkwvk::pcie_link::sysfs_link("/sys", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
    if (l.gbps > 0)
        logf("vk: mirror syncs: PCIe link Gen%d x%u (%.1f GB/s; sysfs, the narrowest of %d links, at %s)", l.gen, l.width, l.gbps,
             l.hops, l.limited_by.c_str());
    else logf("vk: mirror syncs: no PCIe link in sysfs for %04x:%02x:%02x.%x", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
    return l.gbps;
}
#endif

unsigned diff_group() {
    static const unsigned v = getenv("RX_VK_MSYNC_DIFF_GROUP") ? std::clamp(atoi(getenv("RX_VK_MSYNC_DIFF_GROUP")), 1, 16) : 1;
    return v;
}
DiffPage* diff_page(uint64_t page, bool make) {   // (diff_mu held)
    const uint64_t c = page / kDiffChunkPages;
    if (c >= diff_table.size()) {
        if (!make) return nullptr;
        diff_table.resize(c + 1);
    }
    if (!diff_table[c]) {
        if (!make) return nullptr;
        diff_table[c].reset(new DiffPage[kDiffChunkPages]);
        std::memset(diff_table[c].get(), 0, kDiffChunkPages * sizeof(DiffPage));
        diff_table_bytes += kDiffChunkPages * sizeof(DiffPage);
    }
    return &diff_table[c][page % kDiffChunkPages];
}
// (the counters: bytes a sync copied, by where from and when; and the diff's)
struct SyncStats {
    int64_t pre = 0, bind = 0, seg = 0;          // (bytes copied: the presync, a binding outside segments, a segment's)
    int64_t cap = 0, old = 0, live = 0;          // (from this submit's capture, an earlier one's, guest memory)
    int64_t pre_unbound = 0;                     // (the presync's bytes of pools no binding of their submission used)
    int64_t d_pages = 0, d_valid = 0;            // (pages hashed; of them with hashes that held)
    int64_t d_same[5] = {};                      // (bytes unchanged at 256, 512, 1K, 2K, 4K granularity)
    int64_t d_skipped = 0, d_inval = 0;          // (bytes left out; pages whose hashes were voided by another copy)
    int64_t d_ns = 0;                            // (time hashing and comparing)
    int64_t d_gw = 0, d_mid = 0, d_new = 0;      // (pages with hashes not holding: written by the GPU, another mirror, none yet)
} sst;
struct PoolSync { int64_t bytes = 0, same = 0, skipped = 0, pages = 0; uint64_t size = 0; };
std::unordered_map<uint64_t, PoolSync> sync_by_pool;
struct PresyncNote { Pool* p; uint64_t key, bound_before; int64_t bytes; };
std::vector<PresyncNote> presync_notes;   // (this submission's presync, checked against its bindings at its end)
thread_local bool in_presync = false;
// (RX_VK_PRESYNC_STATS=1, diagnosis) the pre-syncs' work: pages looked at and dirty ones, pools synced, time, logged
// every 600 pre-syncs that copied anything
struct PresyncScan { int64_t pages = 0, dirty = 0, loop_ns = 0, lazy_ns = 0, cap_pages = 0, live_pages = 0, scan_ns = 0, diff_ns = 0, gpu_ns = 0; };
thread_local PresyncScan presync_scan;
bool presync_stats_on() {
    static const bool v = getenv("RX_VK_PRESYNC_STATS") && atoi(getenv("RX_VK_PRESYNC_STATS")) != 0;
    return v;
}



/// The pages of the pool the CPU wrote since its mirror's sync, copied in (the upload command buffer)
void mirror_sync(Pool& p) {
#ifdef TRACY_ENABLE
    HT_SCOPE("mirror_sync");
#endif
    late_fix_emit();   // (the late pages' fix, before this copies out of a capture's buffer)
    p.synced_epoch = mirror_epoch;
    p.sync_seg = texture_segment();
    mirror_synced.push_back(&p);
    std::vector<VkBufferCopy> runs;   // (from guest memory)
    std::map<VkBuffer, std::vector<VkBufferCopy>> cap_runs;   // (from the captures' copies)
    uint64_t run_lo = 0, run_hi = 0;
    const uint64_t end_page = (p.hi + 4095) >> kPageShift;
    const bool note_writes = memory_wb_followed(p.lo, p.hi);   // (wb_learn.h: once a pool, not a page)
    for (uint64_t page = p.lo >> kPageShift; page < end_page;) {
        if (region_gen[page / kRegionPages] <= p.synced_gen) {   // (a clean 2 MB region, or its rest when the pool starts in it)
            page = std::min(end_page, (page / kRegionPages + 1) * kRegionPages);
            continue;
        }
        const uint32_t pg = page_gen(page);
        const bool dirty = pg > p.synced_gen;
        if (dirty && gpu_page(page)) {   // (diagnostic: see memory_stamp_gpu)
            st_sync_gpu_pages++;
            const CapEntry e0 = cap_entry(page);
            const bool live = !(e0.gen && e0.gen == pg && cap_bufs.count(e0.gen));
            st_sync_gpu_live += live;
            static int logged;
            if (logged++ < 40)
                logf("vk: a mirror sync copies the CPU's version of page %#llx over the GPU's (pool %#llx, %s, page gen %u, "
                     "pool synced at %u, gen now %u)", (unsigned long long)(page << kPageShift), (unsigned long long)p.lo,
                     live ? "live" : "from a capture", pg, (unsigned)p.synced_gen, gen_now);
        }
        // (hookshot) the CPU wrote this page: in a large binding whose write-back may be left out, the copy below would bring
        // guest memory's stale rest of the page over the GPU's: that binding is written back from now on (wb_learn.h)
        if (dirty && note_writes) memory_note_cpu_access(page << kPageShift, 1ull << kPageShift, true);
        if (dirty && protect_bare_on() && prot_page_since(page, p.synced_gen)) {   // (the GPU's records kept: see prot_pages)
            st_prot_kept++;
            page++;
            continue;
        }
        if (dirty) {   // (as a capture took it, if the page was not written since)
            const CapEntry e = cap_entry(page);
            if (e.gen && e.gen == pg) {
                auto b = cap_bufs.find(e.gen);
                if (b != cap_bufs.end()) {
                    const uint64_t ga = std::max(page << kPageShift, p.lo), gb = std::min((page + 1) << kPageShift, p.hi);
                    append_capture_copy(cap_runs[b->second.buffer],
                        {(uint64_t)e.slot * 4096 + (ga - (page << kPageShift)), ga - p.lo, gb - ga});
                    b->second.last_use = std::max(b->second.last_use, cur_cap_value);
                    (e.gen == cur_cap_gen ? st_sync_cap_pages : st_sync_old_pages)++;
                    page++;
                    continue;
                }
            }
        }
        if (dirty) {
            st_sync_live_pages++;   // (dirty, not in this submit's capture: guest memory as it is now)
            live_by_pool[p.lo]++;
            {   // (RX_VK_LOG_LIVE=1: why, 300 lines: the page's generation, its entry's, the newest capture's)
                static const bool log_live = env_on("RX_VK_LOG_LIVE");
                static int logged;
                if (log_live && logged < 300 && (++live_seen % 97) == 0) {
                    logged++;
                    const CapEntry e = cap_entry(page);
                    logf("vk: live page %#llx: gen %u, entry gen %u (buffer %s), current capture %u, pool synced at %u, gen now %u",
                         (unsigned long long)(page << kPageShift), pg, e.gen,
                         e.gen ? (cap_bufs.count(e.gen) ? "kept" : "gone") : "-", cur_cap_gen, (unsigned)p.synced_gen, gen_now);
                }
            }
            const uint64_t a = std::max(page << kPageShift, p.lo) - p.lo, b = std::min((page + 1) << kPageShift, p.hi) - p.lo;
            if (run_hi == a && run_hi) run_hi = b;
            else {
                if (run_hi) runs.push_back({run_lo, run_lo, run_hi - run_lo});
                run_lo = a, run_hi = b;
            }
        }
        page++;
    }
    if (run_hi) runs.push_back({run_lo, run_lo, run_hi - run_lo});
    bool coalesced = false;
    static const bool coalesce_msync = !env_on("RX_VK_NO_COALESCE_MSYNC");
    if (!msync_pass_on() && coalesce_msync && !cap_runs.empty()) {
        size_t total_items = 0;
        uint64_t total_bytes = 0;
        for (const auto& [buf, rs] : cap_runs) {
            total_items += rs.size();
            for (const VkBufferCopy& r : rs) total_bytes += r.size;
        }
        if (total_items > 1 && total_bytes > 0) {
            struct FlatCopy {
                const uint8_t* src;
                uint64_t dst_offset;
                uint64_t size;
            };
            std::vector<FlatCopy> flat;
            flat.reserve(total_items);
            bool all_mapped = true;
            for (const auto& [buf, rs] : cap_runs) {
                const uint8_t* src_base = nullptr;
                for (const auto& [gen, cb] : cap_bufs) {
                    if (cb.buffer == buf && cb.map) {
                        src_base = (const uint8_t*)cb.map;
                        break;
                    }
                }
                if (!src_base) {
                    all_mapped = false;
                    break;
                }
                for (const VkBufferCopy& r : rs) {
                    flat.push_back({src_base + r.srcOffset, r.dstOffset, r.size});
                }
            }
            if (all_mapped) {
                std::sort(flat.begin(), flat.end(), [](const FlatCopy& a, const FlatCopy& b) {
                    return a.dst_offset < b.dst_offset;
                });
                MsyncSlice slice = msync_stage_alloc(total_bytes, 256);
                if (slice.ptr && slice.buffer != VK_NULL_HANDLE) {
                    std::vector<VkBufferCopy> coalesced_runs;
                    uint64_t cur_stage_offset = slice.offset;
                    uint8_t* cur_stage_ptr = slice.ptr;
                    size_t i = 0;
                    while (i < flat.size()) {
                        size_t j = i + 1;
                        uint64_t span_dst = flat[i].dst_offset;
                        uint64_t span_size = flat[i].size;
                        while (j < flat.size() && flat[j - 1].dst_offset + flat[j - 1].size == flat[j].dst_offset) {
                            span_size += flat[j].size;
                            j++;
                        }
                        uint64_t chunk_pos = 0;
                        for (size_t k = i; k < j; k++) {
                            std::memcpy(cur_stage_ptr + chunk_pos, flat[k].src, (size_t)flat[k].size);
                            chunk_pos += flat[k].size;
                        }
                        coalesced_runs.push_back({cur_stage_offset, span_dst, span_size});
                        cur_stage_ptr += chunk_pos;
                        cur_stage_offset += chunk_pos;
                        i = j;
                    }
                    flush_mapped_range(slice.allocation, slice.offset, total_bytes);
                    GprofSpan gp(upload_command_buffer(), "msync copies (CPU pages from a capture -> mirror, coalesced)");
                    gp.items = (int)coalesced_runs.size();
                    gp.bytes = total_bytes;
                    vkCmdCopyBuffer(upload_command_buffer(), slice.buffer, p.mirror, (uint32_t)coalesced_runs.size(), coalesced_runs.data());
                    shadow_sync_copy(upload_command_buffer(), slice.buffer, p, coalesced_runs);
                    st_sync_bytes += (int64_t)total_bytes;
                    st_sync_copies += (int64_t)coalesced_runs.size();
                    coalesced = true;
                }
            }
        }
    }
    if (!coalesced) {
        for (const auto& [buf, rs] : cap_runs) {
            if (msync_pass_on() && msync_pass_add(p, buf, rs)) {   // (as the pass's chunks: dispatched below, or after the presync's loop)
                if (p.shadow) {   // (RX_VK_PASS_TIMING: a span of their own; the copies' span below times both)
                    GprofSpan gp(upload_command_buffer(), "msync shadow copies (CPU pages from a capture -> shadow)");
                    gp.items = (int)rs.size();
                    for (const VkBufferCopy& r : rs) gp.bytes += r.size;
                    shadow_sync_copy(upload_command_buffer(), buf, p, rs);   // (the shadow as the mirror, as below: the pass writes the mirror only)
                }
                for (const VkBufferCopy& r : rs) st_sync_bytes += (int64_t)r.size;
                st_sync_copies += (int64_t)rs.size();
                continue;
            }
            GprofSpan gp(upload_command_buffer(), "msync copies (CPU pages from a capture -> mirror)");
            gp.items = (int)rs.size();
            vkCmdCopyBuffer(upload_command_buffer(), buf, p.mirror, (uint32_t)rs.size(), rs.data());
            shadow_sync_copy(upload_command_buffer(), buf, p, rs);
            for (const VkBufferCopy& r : rs) st_sync_bytes += (int64_t)r.size, gp.bytes += r.size;
            st_sync_copies += (int64_t)rs.size();
        }
    }
    if (!msync_batching) msync_pass_flush();
    if (runs.empty()) return;
    checkpoint_mark(upload_command_buffer(), p.lo, 'm');
    GprofSpan gp(upload_command_buffer(), "msync copies (CPU pages live from guest memory -> mirror)");
    gp.items = (int)runs.size();
    note_in_place_read();   // (these pages as guest memory has them when the GPU gets here)
    vkCmdCopyBuffer(upload_command_buffer(), p.buffer, p.mirror, (uint32_t)runs.size(), runs.data());
    shadow_sync_copy(upload_command_buffer(), p.buffer, p, runs);
    for (const VkBufferCopy& r : runs) st_sync_bytes += (int64_t)r.size, gp.bytes += r.size;
    st_sync_copies += (int64_t)runs.size();
}

} // namespace
std::atomic<int64_t> st_import_texel_views{0}, st_import_texel_bytes{0}, st_import_unswizzle_bytes{0};

// ---- (hookshot) Large written bindings the CPU does not read: no write-back (RX_VK_WB_MAX_KB, wb_learn.h) ----
// The table is fed from the bindings (wb_leave_out, the replay's threads, pools_mu held), the CPU's reads (memory_note_cpu_read,
// invalidate on the game's threads) and the CPU's writes that a mirror sync copies (mirror_sync); its own mutex, taken last.
// RX_VK_WB_MAX_KB=<KB>: the size from which a binding may be left out (unset or 0: every written binding written back, the
// default); RX_VK_WB_PROBATION=<submissions>: how many submissions write it whole before it is (8)
std::mutex wb_learn_mu;
wb_learn::Table wb_table;
int64_t st_wb_left, st_wb_left_bytes, st_wb_stale;
uint64_t wb_cap() {
    static const uint64_t v = [] {
        // (a fixed 1024 KB cap by default was tried on a 6 GB laptop, against ~68 MB a frame of culling write-backs it judged
        // unread there: not taken, a write-back cut short is data the game reads wrong; RX_VK_WB_MAX_KB=1024 sets it)
        const uint64_t b = wb_learn::max_bytes(getenv("RX_VK_WB_MAX_KB"));
        wb_table.probation = (uint32_t)std::max(1, env_int("RX_VK_WB_PROBATION", (int)wb_table.probation));
        if (b) logf("vk: written bindings over %llu KB are written back only where the CPU reads or writes them, after %u submissions "
                    "whole (RX_VK_WB_MAX_KB)", (unsigned long long)(b >> 10), wb_table.probation);
        return b;
    }();
    return v;
}
// (a written binding [lo, hi): false, write it all back; true, only the guest ranges in keep, the parts the CPU has touched)
bool wb_leave_out(uint64_t lo, uint64_t hi, std::vector<std::pair<uint64_t, uint64_t>>& keep) {
    if (!wb_cap()) return false;
    std::lock_guard lk(wb_learn_mu);
    if (!wb_table.plan(lo, hi, wb_cap(), mirror_epoch, keep)) return false;   // (mirror_epoch: the submission)
    uint64_t kept = 0;
    for (const auto& k : keep) kept += k.second - k.first;
    st_wb_left++, st_wb_left_bytes += (int64_t)(hi - lo - kept);
    return true;
}
void memory_note_cpu_access(uint64_t guest, uint64_t size, bool write) {
    if (!size || !wb_cap()) return;
    bool stale;
    {
        std::lock_guard lk(wb_learn_mu);
        // (only in a large binding followed: the CPU's other reads and writes, all over memory, would only fill the table)
        if (!wb_table.followed(guest, guest + size)) return;
        stale = wb_table.read(guest, guest + size);
        if (stale) st_wb_stale++;
    }
    static std::atomic<int> told{0};
    if (stale && told++ < 20)
        logf("vk: the CPU %s %#llx +%#llx, in a binding whose write-back was left out: written back from now on (RX_VK_WB_MAX_KB=0: "
             "every one)", write ? "wrote" : "read", (unsigned long long)guest, (unsigned long long)size);
}
void memory_note_cpu_read(uint64_t guest, uint64_t size) { memory_note_cpu_access(guest, size, false); }
bool memory_wb_followed(uint64_t lo, uint64_t hi) {
    if (!wb_cap()) return false;
    std::lock_guard lk(wb_learn_mu);
    return wb_table.followed(lo, hi);
}
static void wb_learn_report() {   // (memory_end_submission, every 600 submits)
    if (!wb_cap()) return;
    std::lock_guard lk(wb_learn_mu);
    logf("vk: write-backs left out (600 submits): %lld, %.1f MB; %zu bindings followed, %zu ranges the CPU read, %llu resets; CPU reads "
         "or writes of one left out: %lld so far", (long long)st_wb_left, st_wb_left_bytes / 1048576.0, wb_table.seen(), wb_table.reads(),
         (unsigned long long)wb_table.resets(), (long long)st_wb_stale);
    st_wb_left = st_wb_left_bytes = 0;
}
// (hookshot, diagnostics) RX_VK_LOG_WB_SKIPS=1: the written bindings left out, by guest range, every 600 submits
struct WbSkip { uint64_t hi, n, bytes; };
std::mutex wb_skips_mu;
std::map<uint64_t, WbSkip> wb_skips;   // (by guest start)
static bool wb_skips_log() {
    static const bool v = env_on("RX_VK_LOG_WB_SKIPS");
    return v;
}
static void wb_skip_note(uint64_t lo, uint64_t hi) {
    std::lock_guard lk(wb_skips_mu);
    WbSkip& s = wb_skips[lo];
    s.hi = std::max(s.hi, hi), s.n++, s.bytes += hi - lo;
}
static void wb_skips_report() {
    std::vector<std::pair<uint64_t, uint64_t>> top;   // (bytes, lo)
    uint64_t all = 0;
    std::lock_guard lk(wb_skips_mu);
    for (const auto& [lo, s] : wb_skips) top.push_back({s.bytes, lo}), all += s.bytes;
    std::sort(top.rbegin(), top.rend());
    std::string t;
    for (size_t i = 0; i < top.size() && i < 12; i++) {
        const WbSkip& s = wb_skips[top[i].second];
        char b[112];
        snprintf(b, sizeof b, " %#llx+%#llx: %llu, %.1f MB;", (unsigned long long)top[i].second,
                 (unsigned long long)(s.hi - top[i].second), (unsigned long long)s.n, s.bytes / 1048576.0);
        t += b;
    }
    logf("vk: write-backs skipped (RX_VK_WB_MAX_KB, 600 submits): %zu ranges, %.1f MB; the largest:%s", wb_skips.size(),
         all / 1048576.0, t.c_str());
    for (auto& kv : wb_skips) kv.second.bytes = 0, kv.second.n = 0;
}

// ---- (from PR 430, YunaraDebugs) Readback pages: GPU results the CPU reads (invalidate) and then writes, as OoT's exposure meter (a
// sum the GPU adds into each frame; the CPU reads it and stores 0 back). On the console the GPU adds into that memory as it is when
// its work runs, the CPU's 0 included. Here it adds into the pool's mirror, which takes the CPU's 0 only at a mirror sync of a submit
// made after it: one made just before (the game's threads overlap) added onto the old sum, the CPU read 2x the next frame and 0 the
// one after, and the exposure stepped (the light dimming or flaring for a frame in Kokiri). So each submission first copies those
// pages from guest memory into the mirror, on the GPU as its work starts, last of the uploads: what the CPU wrote by then is what the
// GPU adds to. ----
// (hookshot, 11-09) A parallel replay's recordings share one submission (mirror_epoch) and its one write-back at its end: the part
// before the segments, each segment and the recording after them (vk_replay.cpp replay_handle). A refresh in a later one copied the
// guest's page, which has none of the submission's GPU writes, over what the recordings ahead of it had written in the mirror: OoT's
// exposure sum came back ~0 and every frame was too bright (RX_VK_PARALLEL=4). So a recording that joined an open submission
// refreshes a page only when every recording ahead of it in the queue is made and none of them may have written the page: the copy
// then brings the CPU's writes and wipes nothing (and a page first read in the middle of a submission is refreshed ahead of its first
// use, unless the recordings up to that use were all made before it was read). What they may have written are the GPU write
// hazards: every range bound for writing in the mirror, and the ranges a program may write that are not copied back (a bare table
// entry's guessed range past its first 64 bytes, the pointer table's, a wide binding: vk_draw.cpp, memory_note_gpu_write). A
// recording that opens its submission, or comes after its write-back (frame generation's), refreshes every page as before: serial
// replay records the same copies. With RX_VK_FOLLOW_STORES (stores through the page table, bound nowhere) a joining one refreshes
// none.
namespace {
std::unordered_map<uint32_t, uint64_t> rb_pages;   // (guest page -> mirror_epoch it was last read at)
std::atomic<int64_t> st_rb_refresh{0}, st_rb_held{0};
// RX_VK_RB_RANGES=1: a readback page's refresh copies only the bytes the CPU's invalidates of it covered (their union,
// 16-byte aligned outwards), not the whole 4 KB page. The results are small (meters, counters: 0x4000 at most), and the refresh is
// there for the CPU's writes into them (OoT's exposure meter: read, then 0 stored). On a PCIe 3.0 x4 card the ~540 whole pages a
// frame were ~2 MB and ~2 ms of GPU time. Off by default
std::unordered_map<uint32_t, uint32_t> rb_spans;   // (guest page -> lo | hi << 16, bytes in the page)
std::atomic<int64_t> st_rb_bytes{0};
bool rb_ranges_on() {
    static const bool v = getenv("RX_VK_RB_RANGES") && atoi(getenv("RX_VK_RB_RANGES")) != 0;
    return v;
}
// (the open submission the recording this thread is making joined; 0: none: it opens one, or comes after one's end)
thread_local uint64_t rb_joined = 0;
// (the GPU write hazards of submission rb_hazards_epoch, by guest address, with their recording's place in the queue
// (texture_segment); kept past its write-back, for the recording that ends it)
struct RbHazard {
    uint64_t lo, hi, seg;
};
std::vector<RbHazard> rb_hazards;
uint64_t rb_hazards_epoch = 0;
// (a parallel replay's segments: the submission, the first one's order (texture_segment) and how many; the ones made, by order)
struct RbSegments {
    uint64_t epoch = 0, base = 0, n = 0, made = 0;
} rb_segs;
void rb_hazard_locked(uint64_t lo, uint64_t hi) {   // (pools_mu held)
    if (rb_hazards_epoch != mirror_epoch) rb_hazards.clear(), rb_hazards_epoch = mirror_epoch;
    rb_hazards.push_back({lo, hi, texture_segment()});
}
}
void memory_note_readback(uint64_t guest, uint64_t size) {
    if (!size || size > 0x4000) return;   // (small results only: meters and counters)
    std::lock_guard lk(pools_mu);
    for (uint64_t pg = guest >> kPageShift; pg <= (guest + size - 1) >> kPageShift; pg++) rb_pages[(uint32_t)pg] = mirror_epoch;
}
bool memory_is_rb_page_locked(uint32_t page) {   // ('s log; pools_mu held)
    return rb_pages.count(page) != 0;
}
void memory_note_gpu_write(uint64_t gpu, uint64_t size) {
    if (!size) return;
    uint64_t guest = gpu;
    if (gpu >= (1ull << 40)) {   // (as gpu_buffer_bind)
        uint64_t avail = 0;
        if (!host->gpu_ptr(gpu, &avail) || avail < size) return;
        guest = host->gpu_guest(gpu);
    }
    memory_gpu_writes_note(guest, size);   // (RX_VK_UB_SNAPSHOT's guard: a uniform range among them is bound in place)
    std::lock_guard lk(pools_mu);
    rb_hazard_locked(guest, guest + size);
    // and RX_VK_READBACK_EARLY's writers: a bare storage pointer's guessed range or a wide binding is written in
    // place where pools have no mirror, and no page record held it, so a read let go at an earlier writer's part could come
    // back before it (zeldaootr's review of https://git.notrexed.dev/Hookshot/Hookshot/pulls/303, finding 4: on the Steam
    // Deck, Kokiri Forest to the Market, Kakariko and Hyrule Field, 3 of 637 waits for the 0x3980 camera block had 4-21 bytes
    // changed after their writer's part). Only ranges a read waited for are kept (early_in_place_write)
    ec_kind = 2;
    early_in_place_write(guest, guest + size, mem_rec_value);
    ec_kind = 0;
    // (and the pools it reaches keep their mirrors: down the table until none further down can reach guest, import_reach.h)
    for (auto it = imported.upper_bound(guest + size - 1); it != imported.begin();) {
        --it;
        if (it->second.hi > guest) it->second.gpu_only = true;
        else if (import_reach::beyond(live_reach, it->second.lo, guest)) break;
    }
}
void memory_rb_recording_begun() {
    std::lock_guard lk(pools_mu);
    rb_joined = mirror_upload ? mirror_epoch : 0;
}
void memory_rb_segments_begin(uint64_t base, uint64_t n) {
    std::lock_guard lk(pools_mu);
    rb_segs = {mirror_epoch, base, n, 0};
}
/// Records the refresh copies into the submission's upload command buffer; true when it recorded any
bool memory_rb_refresh(VkCommandBuffer up) {
    std::lock_guard lk(pools_mu);
    static const bool follow_stores = env_on("RX_VK_FOLLOW_STORES");
    const bool joined = rb_joined && rb_joined == mirror_epoch;
    const uint64_t seg = texture_segment();
    bool ahead_made = !follow_stores;   // (every recording ahead of this one in the queue made, and what they may write known)
    if (joined && rb_segs.epoch == mirror_epoch && seg >= rb_segs.base && seg - rb_segs.base < rb_segs.n) {
        const uint64_t k = seg - rb_segs.base;   // (a segment's: the ones before it made, and this one now)
        const uint64_t below = k < 64 ? (1ull << k) - 1 : ~0ull;
        ahead_made = ahead_made && k < 64 && (rb_segs.made & below) == below;
        if (k < 64) rb_segs.made |= 1ull << k;
    }
    if (rb_pages.empty() || !up) return false;
    // (no pool has a mirror, as on an integrated GPU: no page has one to be copied into, and each was still looked up among the
    // pools at every recording. The pages not read for a while still go, every 64th time)
    static uint32_t unmirrored = 0;
    if (st_mirrors == st_mirrors_freed && (++unmirrored & 63)) return false;   // (made and freed: a sparse mirror may hold no byte yet)
    auto written_ahead = [&](uint64_t lo, uint64_t hi) {   // (may be, by a recording ahead of this one in its submission)
        if (rb_hazards_epoch != mirror_epoch) return false;
        for (const RbHazard& h : rb_hazards)
            if (h.seg < seg && h.lo < hi && lo < h.hi) return true;
        return false;
    };
    std::map<Pool*, std::vector<VkBufferCopy>> by_pool;
    for (auto it = rb_pages.begin(); it != rb_pages.end();) {
        if (it->second + 600 < mirror_epoch) { it = rb_pages.erase(it); continue; }   // (not read for a while)
        const uint64_t g = (uint64_t)it->first << kPageShift;
        ++it;
        auto pit = imported.upper_bound(g);
        if (pit == imported.begin()) continue;
        --pit;
        Pool& p = pit->second;
        if (g < p.lo || g >= p.hi || p.mstate != Pool::MIRROR_ON || !p.mirror) continue;
        const uint64_t a = g - p.lo, b = std::min(g + 4096, p.hi) - p.lo;
        if (joined && (!ahead_made || written_ahead(p.lo + a, p.lo + b))) {   // (not here: see above)
            st_rb_held++;
            continue;
        }
        by_pool[&p].push_back({a, a, b - a});
    }
    if (by_pool.empty()) return false;
    full_barrier(up);   // (after the older submissions' write-backs into guest memory and this one's other uploads)
    GprofSpan gp(up, "rbrefresh copies (readback pages guest -> mirror)");
    gp.items = 0;
    for (auto& [pp, rs] : by_pool) {
        st_rb_refresh += (int64_t)rs.size();
        // (hookshot) pages next to each other as one copy (~530 a frame in Hyrule Field, a copy a page)
        std::sort(rs.begin(), rs.end(), [](const VkBufferCopy& x, const VkBufferCopy& y) { return x.srcOffset < y.srcOffset; });
        size_t k = 0;
        for (size_t i = 1; i < rs.size(); i++)
            if (rs[k].srcOffset + rs[k].size == rs[i].srcOffset) rs[k].size += rs[i].size;
            else rs[++k] = rs[i];
        rs.resize(k + 1);
        // (guest memory at the GPU's time, and not noted for the readback's part wait: see note_in_place_read)
        vkCmdCopyBuffer(up, pp->buffer, pp->mirror, (uint32_t)rs.size(), rs.data());
        gp.items += (int)rs.size();
        for (const VkBufferCopy& r : rs) gp.bytes += r.size;
    }
    static std::atomic<int64_t> n{0};
    if (++n % 3000 == 1)
        logf("vk: readback pages refreshed from guest memory: %lld pages so far (%lld held: may be written ahead in their submission)",
             (long long)st_rb_refresh.load(), (long long)st_rb_held.load());
    return true;
}

// (RX_VK_POOL_MIRROR: the fork's mirrors of pools never imported were built on its own mirror code, which main's replaced; not
// here, so its callers keep main's paths)
bool memory_pool_mirror_on() { return false; }
BufferRef gpu_buffer(uint64_t gpu, uint64_t size) {
    uint64_t avail = 0;
    if (!host->gpu_ptr(gpu, &avail) || avail < size) return {};
    const uint64_t guest = host->gpu_guest(gpu);
    std::lock_guard lk(pools_mu);
    Pool* p = pool_of(guest, size);
    if (!p) return {};
    if (p->last_value < mem_rec_value) p->last_value = mem_rec_value;   // (in use: not given back under it, imports_give_back)
    note_in_place_read();   // (the pool itself, read by the GPU where the caller's work runs)
    return {p->buffer, guest - p->lo, size, p->address + (guest - p->lo)};
}

void memory_pool_created(uint64_t gpu, uint64_t size) {
    // (RX_VK_BG_IMPORT=1 only: imported eagerly, every pool the game made, most never used by the GPU, the thread's
    // page locking stalled the game's threads (3-7 fps); chunked imports at first use are the default)
    static const bool on = env_on("RX_VK_BG_IMPORT");
    if (!on || !size || gpu >= (1ull << 40) || !dev.external_memory_host) return;   // (virtual pools are not imported)
    const uint64_t lo = gpu & ~(align - 1), hi = (gpu + size + align - 1) & ~(align - 1);
    std::lock_guard lk(bg_mu);
    bg_start();
    if (bg_inflight.contains(lo)) return;
    bg_inflight.emplace(lo, hi);
    bg_queue.push_back({lo, hi});
    bg_cv.notify_one();
}

// (hookshot) Whether one imported pool holds all of [lo, hi): a lookup there imports nothing (vk_texture.cpp
// upload_unswizzle_gpu asks before it reaches for one)
int memory_guest_contained(uint64_t lo, uint64_t hi) {
    std::lock_guard lk(pools_mu);
    auto it = imported.upper_bound(lo);   // (pools by lo: the last one starting at or below lo)
    if (it == imported.begin()) return 0;
    --it;
    return it->second.lo <= lo && it->second.hi >= hi ? 1 : 0;
}
struct DeadImport { VkBuffer buffer; VkDeviceMemory memory; uint64_t value; uint64_t lo, bytes; std::vector<VkDeviceMemory> pieces; };
std::deque<DeadImport> dead_imports;

extern std::deque<std::map<uint64_t, Pool>::node_type> evicted_pools;   // (imports_give_back's, below)
int memory_guest_imported(uint64_t lo, uint64_t hi) {
    if (hi <= lo) return 0;
    std::lock_guard lk(pools_mu);
    auto it = imported.lower_bound(hi);   // (pools by lo: those starting below hi, the last ones checked first)
    while (it != imported.begin()) {
        --it;
        if (it->second.hi > lo) return 1;
        if (import_reach::beyond(live_reach, it->second.lo, lo)) break;   // (none further down reaches lo: import_reach.h)
    }
    for (const auto& nh : retired_pools)   // (those still holding memory: retired_compact sets the others aside)
        if (nh.mapped().memory && nh.mapped().lo < hi && lo < nh.mapped().hi) return 1;
    // (hookshot, 11-13) and the idle pools given back (imports_give_back) whose memory is not freed yet: waiting for the GPU, or
    // for their wait on AMD and Intel. Not counted, the game's freed pages under them were decommitted while the driver still
    // had them imported, which faulted the device (hos_mem.c FreeMemoryPages)
    for (const auto& nh : evicted_pools)
        if (nh.mapped().memory && nh.mapped().lo < hi && lo < nh.mapped().hi) return 1;
    for (const auto& pending : dead_imports)
        if (pending.lo < hi && lo < pending.lo + pending.bytes) return 1;

    // The worker imports outside pools_mu. It moves a range from inflight to ready under bg_mu;
    // bg_drain moves it from ready to the active map while holding pools_mu.
    std::lock_guard bg_lock(bg_mu);
    for (const auto& [start, end] : bg_inflight)
        if (start < hi && lo < end) return 1;
    for (const auto& [start, p] : bg_ready)
        if (start < hi && lo < p.hi) return 1;
    return 0;
}

bool mirror_make(Pool& pool);   // (below)
/// The mirrored pools, for the submit-time capture (memory_capture_submit) (pools_mu held)
void publish_cap_ranges() {
    std::vector<std::pair<uint64_t, uint64_t>> v;
    for (const auto& [lo, pool] : imported)
        if (pool.mstate == Pool::MIRROR_ON || pool.was_mirrored) v.push_back({pool.lo, pool.hi});
    std::lock_guard lk(cap_ranges_mu);
    cap_ranges.swap(v);
    cap_ranges_ver++;
}
// (EvilOOT, 11-07) the game's own memory pool holding a GPU address (below 1 TB, where the address is the guest's), not the
// import that happens to cover it: imports are 16 MB chunks around what was bound, and since upstream 1c7c9c7 only around
// the bound range, so the pool's end was a chunk's end: the pointer tables of the task and mesh stages (whole pools) and the
// bare entries stopped there, the grass's programs read their records 12-18 MB further on as 0 and culled every blade
static bool game_pool_of(uint64_t gpu, uint64_t& lo, uint64_t& hi) {   // (pools_mu held)
    if (gpu >= (1ull << 40) || guest_pools.empty()) return false;
    auto it = std::upper_bound(guest_pools.begin(), guest_pools.end(), gpu, [](uint64_t a, const GuestPool& p) { return a < p.lo; });
    if (it == guest_pools.begin()) return false;
    --it;
    if (gpu < it->lo || gpu >= it->hi) return false;
    lo = it->lo, hi = it->hi;
    return true;
}
/// True when gpu sits in [lo, hi) and that span was cached from the guest-pool table this generation still is.
bool guest_span_covers(uint64_t gpu, uint64_t lo, uint64_t hi, uint64_t cached_gen, uint64_t gen) {
    return cached_gen == gen && lo < hi && gpu >= lo && gpu < hi;
}
struct GuestSpan { uint64_t lo = 1, hi = 0, gen = 0; };
thread_local GuestSpan guest_span;
bool guest_span_cached(uint64_t gpu, uint64_t* lo, uint64_t* hi) {
    if (gpu >= (1ull << 40)) return false;
    const uint64_t gen = guest_pools_gen.load(std::memory_order_relaxed);
    if (!guest_span_covers(gpu, guest_span.lo, guest_span.hi, guest_span.gen, gen)) return false;
    *lo = guest_span.lo;
    *hi = guest_span.hi;
    return true;
}
void guest_span_store(uint64_t lo, uint64_t hi) {   // (pools_mu held: the generation matches the table just read)
    guest_span = {lo, hi, guest_pools_gen.load(std::memory_order_relaxed)};
}
uint64_t gpu_pool_end(uint64_t gpu) {
    uint64_t lo, hi;
    // (storage-buffer tables ask this per entry, and the entries of one table sit in one pool: the locked lookup of a
    // pool was ~8% of replay time before buffer binds cached it)
    if (guest_span_cached(gpu, &lo, &hi)) return hi;
    ParLock plk;   // (before pools_mu, always)
    std::lock_guard lk(pools_mu);
    if (game_pool_of(gpu, lo, hi)) {
        guest_span_store(lo, hi);
        return hi;
    }
    const Pool* p = pool_of(gpu, 1);
    return p ? p->hi : 0;
}
uint64_t gpu_pool_start(uint64_t gpu) {   // (EvilOOT) where it begins
    uint64_t lo, hi;
    if (guest_span_cached(gpu, &lo, &hi)) return lo;
    ParLock plk;
    std::lock_guard lk(pools_mu);
    if (game_pool_of(gpu, lo, hi)) {
        guest_span_store(lo, hi);
        return lo;
    }
    const Pool* p = pool_of(gpu, 1);
    return p ? p->lo : 0;
}
/// (hookshot) How much of [gpu, gpu + n) from gpu on one import can hold: up to the end of the game pool holding gpu, since
/// pool_lookup takes a range inside one pool only (and for a virtual pool's address, from 1 TB, only as far as the host maps it
/// on). in_pool false: no pool holds gpu, so no import does, up to the next pool's start. Never 0 for n > 0 (vk_replay.cpp
/// gpu_piece_bind: the replay's own writes go on a piece at a time)
uint64_t gpu_pool_piece(uint64_t gpu, uint64_t n, bool& in_pool) {
    in_pool = false;
    uint64_t guest = gpu;
    if (gpu >= (1ull << 40)) {
        uint64_t avail = 0;
        if (!host->gpu_ptr(gpu, &avail) || !avail) return n;
        n = std::min(n, avail);
        guest = host->gpu_guest(gpu);
    }
    if (!n) return 0;
    ParLock plk;   // (before pools_mu, always)
    std::lock_guard lk(pools_mu);
    // (pool_lookup's pool: the last one starting at or below guest)
    const auto it = std::upper_bound(guest_pools.begin(), guest_pools.end(), guest, [](uint64_t a, const GuestPool& p) { return a < p.lo; });
    if (it != guest_pools.begin() && guest < (it - 1)->hi) {
        in_pool = true;
        return std::min(n, (it - 1)->hi - guest);
    }
    return it == guest_pools.end() ? n : std::min(n, it->lo - guest);
}

// (pools leaving, a mirror made, or the next submission: a remembered read binding is looked up again)
uint64_t gpu_bind_stamp() {
    return pools_gen.load(std::memory_order_relaxed) ^ (mirror_epoch * 0x9E3779B97F4A7C15ull);
}

uint32_t bind_slot(uint64_t guest, uint64_t size) {
    // (a page-aligned range's low bits are zero: fold the rest in, or every one of them lands in one slot)
    uint64_t h = guest ^ (size * 0x9E3779B97F4A7C15ull);
    h *= 0x9E3779B97F4A7C15ull;
    h ^= h >> 32;
    h ^= h >> 16;
    return (uint32_t)(h & 255u);
}

bool bind_memo_hit(uint64_t guest, uint64_t size, uint64_t stamp, uint64_t cached_guest, uint64_t cached_size,
                   uint64_t cached_stamp, int has_buffer) {
    return has_buffer && size && guest == cached_guest && size == cached_size && stamp == cached_stamp;
}

struct BindMemo {
    uint64_t guest = 0, size = 0, stamp = 0;
    BufferRef ref{};
};
thread_local BindMemo bind_memo[256];

void note_direct_write(Pool& p, uint64_t off, uint64_t size) {   // (pools_mu held)
    p.direct_write_epoch = mirror_epoch;
    early_in_place_write(p.lo + off, p.lo + std::min(off + size, p.hi - p.lo), mem_rec_value);   // (RX_VK_READBACK_EARLY)
    if (size <= (1u << 20))   // (EvilBoyce: GPU results written in place; small ones: the readbacks are)
        wb_pages_note(p.lo + off, p.lo + std::min(off + size, p.hi - p.lo), mem_rec_value, 0);
}

BufferRef gpu_buffer_bind(uint64_t gpu, uint64_t size, bool written) {
    if (written) texture_gpu_wrote(gpu, size);
    // (below 1 TB a GPU address is the guest address itself, and a pool holds the whole range or nothing: the host's
    // two translations are for the virtual pools above)
    uint64_t guest = gpu;
    if (gpu >= (1ull << 40)) {
        uint64_t avail = 0;
        if (!host->gpu_ptr(gpu, &avail) || avail < size) return {};
        guest = host->gpu_guest(gpu);
    }
    // (the same read, while that stamp still stands: the buffer and the offset are the ones found before. a miss
    // runs the lookup below, including a mirror sync, and remembers what it returned. a write is not remembered)
    auto remember = [&](BufferRef ref) {
        if (!written && ref.buffer != VK_NULL_HANDLE && size)
            bind_memo[bind_slot(guest, size)] = BindMemo{guest, size, gpu_bind_stamp(), ref};
        return ref;
    };
    if (!written) {
        const BindMemo& m = bind_memo[bind_slot(guest, size)];
        if (bind_memo_hit(guest, size, gpu_bind_stamp(), m.guest, m.size, m.stamp, m.ref.buffer != VK_NULL_HANDLE))
            return m.ref;
    }
    // (a read binding of a pool this thread found before and whose mirror is in use this submission or that has none,
    // without the lock: every thread's, the replay thread's too (each draw binds several uniform buffers: the pool's
    // lookup under the lock was ~8% of the replay's time in a race); the rest under it, as below)
    const bool par = par_active.load(std::memory_order_relaxed);
    TlPool& tl = tl_pools[(guest >> 16) & 1023];
    if (!written && tl.p && tl.gen == pools_gen.load(std::memory_order_relaxed) && guest >= tl.p->lo &&
        guest <= tl.p->hi && size <= tl.p->hi - guest) {   // (hookshot: no wrap-around for a huge size, as pool_of)
        Pool* p = tl.p;
        const uint64_t off = guest - p->lo;
        if (p->last_value < mem_rec_value) p->last_value = mem_rec_value;
        __atomic_fetch_add(&p->heat, Pool::heat_of(size), __ATOMIC_RELAXED);   // (hookshot: mirrors_by_heat)
        if (!mirror_on() || !mirror_upload || p->mstate == Pool::MIRROR_WANTED ||
            (p->mstate == Pool::MIRROR_REFUSED && (!p->mirror_retry_epoch || mirror_epoch < p->mirror_retry_epoch))) {
            note_in_place_read(!mirror_upload);
            return remember({p->buffer, off, size, 0});
        }
        if (p->mstate == Pool::MIRROR_ON && p->synced_epoch == mirror_epoch && p->sync_seg <= texture_segment() &&
            sparse_has(*p, off, size)) {   // (a sparse mirror's block with no memory: under the lock, below)
            sparse_touch(*p, off, size);
            return remember({p->mirror, off, size, 0});
        }
    }
    ParLock plk;   // (before pools_mu, always: a mirror sync records into the shared upload command buffer)
    std::lock_guard lk(pools_mu);
    {   // (hookshot, smoothness: import_in_background above. Reads up to 4 MB; not in a parallel replay's segment)
        static const bool async_import = [] { const char* e = getenv("RX_VK_ASYNC_IMPORT"); return e && atoi(e) != 0; }();
        if (async_import && !written && !par && size <= (4u << 20) && guest < (1ull << 40) &&
            !(last_pool && guest >= last_pool->lo && guest + size <= last_pool->hi) && import_in_background(guest, size)) {
            uint64_t avail = 0;
            const uint8_t* src = host->gpu_ptr(gpu, &avail);
            if (src && avail >= size)
                if (const BufferRef r = stream_copy(src, size)) {
                    st_async_copies++;
                    return r;
                }
        }
    }
    Pool* p = pool_of(guest, size);
    if (!p) return {};
    tl = {p, pools_gen.load(std::memory_order_relaxed)};
    if (p->last_value < mem_rec_value) p->last_value = mem_rec_value;
    __atomic_fetch_add(&p->heat, Pool::heat_of(size), __ATOMIC_RELAXED);   // (hookshot: mirrors_by_heat)
    const uint64_t off = guest - p->lo;
    if (p->mstate == Pool::MIRROR_REFUSED && p->mirror_retry_epoch && mirror_epoch >= p->mirror_retry_epoch) {
        if (mirror_bytes + (p->hi - p->lo) <= mirror_budget()) {
            p->mstate = Pool::MIRROR_NONE;
            p->mirror_retry_epoch = 0;
        } else {
            p->mirror_retry_epoch = mirror_epoch + 120;
        }
    }
    if (!mirror_on() || !mirror_upload || p->mstate == Pool::MIRROR_REFUSED) {
        if (written) note_direct_write(*p, off, size);
        if (mirror_on()) { auto& e = direct_by_pool[p->lo]; e.first++, e.second = mirror_upload ? (int)p->mstate : 9; direct_bytes_by_pool[p->lo] += size; }
        // (hookshot) written by the GPU in guest memory itself, past the mirror (bound outside a recording's uploads): the
        // write-back diff's shadow no longer knows those pages, and reads them again at its next write-back, after this write
        if (written) shadow_forget(*p, off, std::min(size, p->hi - p->lo - off));
        note_in_place_read(!mirror_upload);
        return remember({p->buffer, off, size, 0});
    }
    // (a pool's mirror is made at its first binding, in this part's uploads: bound in guest memory until the next
    // submission, the GPU read it there while the game's threads wrote it for their next frames, and a pass that drew
    // over the moment a write landed came out in two (shadows dark below a line for a frame, the line moving; a new
    // pool every so many frames as the per-frame data moves on). Not when this part already wrote it in guest memory:
    // the copy at the part's start would miss that. Not in a parallel replay's segment either: its copy goes into the
    // uploads of the segment whose worker bound the pool first, which need not be the first in the queue. A segment
    // before it found the mirror in the page table and read it unfilled (houses, rocks and bushes missing, geometry
    // torn, the more the segments the worse; #994 with RX_VK_PARALLEL))
    if (p->mstate == Pool::MIRROR_NONE && !par && p->direct_write_epoch != mirror_epoch && mirror_make(*p)) {
        publish_cap_ranges();
        static int64_t made_now;
        if (++made_now % 200 == 1)
            logf("vk: pool %#llx (%llu MB): mirror made at its first binding (%lld so)", (unsigned long long)p->lo,
                 (unsigned long long)((p->hi - p->lo) >> 20), (long long)made_now);
    }
    if (p->mstate == Pool::MIRROR_NONE) {   // (made at the next submission's start)
        p->mstate = Pool::MIRROR_WANTED;
        mirror_wanted.push_back(p);
    }
    if (p->mstate != Pool::MIRROR_ON) {
        auto& e = direct_by_pool[p->lo];
        e.first++, e.second = (int)p->mstate;
        direct_bytes_by_pool[p->lo] += size;
        if (written) note_direct_write(*p, off, size);
        note_in_place_read();
        return remember({p->buffer, off, size, 0});
    }
    if (p->sparse && !sparse_has(*p, off, size) && sparse_gaps_outside_pools(*p, off, size)) {
        // (hookshot, RX_VK_MIRROR_SPARSE) its blocks with no memory are all outside every game pool: a binding with a guessed
        // size that runs past a pool's end. Served from the mirror below, as a mirror with memory everywhere would serve it:
        // what the GPU wrote only into the mirror is read, and a write lands there, written back where a block has memory. Not
        // wanted: no game pool is there (sparse_want_live gives them memory at the submission after one comes)
        st_sparse_past++;
    } else if (p->sparse && !sparse_has(*p, off, size)) {
        // (hookshot, RX_VK_MIRROR_SPARSE) blocks of its mirror with no memory where a game pool is (one there since the mirror
        // was made, or let go and back): served from guest memory for this submission, where the page table points those pages
        // too, as a pool with no mirror yet is; they get memory at the next one (sparse_grow). One that reaches blocks with
        // memory too is counted (st_sparse_mixed): GPU-only data in those is missed by it
        sparse_want(*p, off, size);
        sparse_touch(*p, off, size);
        // (counted, read or written; written: the blocks with memory it reached are copied in again from guest memory)
        if (!sparse_none(*p, off, size) && written)
            p->refill.push_back({off, std::min(size, p->hi - p->lo - off)});
        auto& e = direct_by_pool[p->lo];
        e.first++, e.second = 10;   // (state 10 in the log: a sparse mirror's block with no memory yet)
        direct_bytes_by_pool[p->lo] += size;
        if (written) note_direct_write(*p, off, size), shadow_forget(*p, off, std::min(size, p->hi - p->lo - off));
        note_in_place_read();
        st_sparse_in_place++;
        return remember({p->buffer, off, size, 0});
    }
    sparse_touch(*p, off, size);   // (its mirror's blocks are in use: all with memory, or those without past every game pool)
    if (p->synced_epoch != mirror_epoch || p->sync_seg > texture_segment()) mirror_sync(*p);
    if (written) {
        p->gpu_written = true;   // (its mirror kept: mirrors_give_back)
        // (hookshot) a binding over RX_VK_WB_MAX_KB (off unless set; 1024 is the tested size) that the CPU has not read through its
        // probation queues no write-back: OoT writes tens of MB of culling lists and per-frame buffers through the GPU that
        // the CPU never reads, and copying them back was most of a submit's memory sync (issue 384: 68 MB a frame; Hyrule
        // Field on an RTX 3060 Ti, 68 -> 18 MB written back a frame). Learned, not a plain size cap: the CPU does read
        // some large ones (wb_learn.h)
        const uint64_t wlo = p->lo + off, whi = p->lo + std::min(off + size, p->hi - p->lo);
        static thread_local std::vector<std::pair<uint64_t, uint64_t>> keep;
        if (!wb_leave_out(wlo, whi, keep)) writebacks.push_back({p, off, whi - p->lo});
        else {
            for (const auto& [klo, khi] : keep) writebacks.push_back({p, klo - p->lo, khi - p->lo});
            if (wb_skips_log()) wb_skip_note(wlo, whi);
        }
        rb_hazard_locked(p->lo + off, p->lo + std::min(off + size, p->hi - p->lo));   // (memory_rb_refresh)
    }
    return remember({p->mirror, off, size, 0});
}

// ---- mirrors given back ----
// A mirror was never freed: the retired pools' stayed (every pool remade larger, or taken in by a larger one, kept
// its VRAM), and pools the game no longer used kept theirs until the mirrors' budget was full; the pools after that
// stayed in system memory for good (the GPU read them over PCIe). Now a retired pool's mirror goes once the GPU is done
// with the submissions that bound it; a mirror not bound for kIdleEpochs submissions goes when a wanted one does not
// fit, or after kLongIdle; and a pool refused for the budget is tried again when it is next bound.
constexpr uint64_t kIdleEpochs = 600, kLongIdle = 6000;
// the idle mirrors given back by why, the ones made again soon after, and RX_VK_MIRROR_STICKY (mirrors_give_back)
int64_t st_freed_tight = 0, st_freed_budget = 0, st_freed_long = 0, st_remade = 0, st_sticky_kept = 0;
uint64_t st_remade_bytes = 0;
uint64_t st_mirror_bytes_freed = 0;
// (hookshot) A freed mirror's buffer is destroyed only once the GPU is past everything recorded until then. Its pool's
// last_value counts the draws and dispatches that bound it, not the shaders that reach it through the guest page table:
// a culling or fog compute pass already running read a mirror destroyed under it (device lost in 53caac5f and ee7764a6,
// issues 132 and 195, AMD after a scene change). The table goes back to guest memory at once, as before
struct DeadMirror { VkBuffer buffer; VmaAllocation alloc; uint64_t value; };
std::atomic<int> mirror_reads{0};   // (HL_MEM_DROP_POOLS: copies out of a mirror under way, memory_mirror_read)
std::deque<DeadMirror> dead_mirrors;
// A retired pool's import, given back the same way. Each import is a userptr buffer object to amdgpu, and RADV lists every
// buffer object in every submission (buffer device addresses): the kernel checks every page of every userptr object at each
// DRM_IOCTL_AMDGPU_CS. Retired pools were never given back, so the pages it checked only grew with each pool merge: in Kokiri
// Forest ~1.3 GB of imports, 15-45 ms a submission (3-4 fps on an RX 7800 XT with 11 ms of GPU time; issues 312, 413). Its
// table entries went at its retirement, and its mirror goes before (mstate), so once the GPU is past retired_value nothing
// reaches it. The node stays (pointers to it are kept elsewhere); only its buffer and memory go. RX_VK_KEEP_RETIRED=1: kept
uint64_t st_retired_imports_freed = 0, st_retired_import_bytes_freed = 0;
void import_unpinned(uint64_t lo, uint64_t hi);   // (below)
uint64_t dead_import_bytes = 0;   // (queued in dead_imports: still in the heap's use until the GPU is past them)
void reap_mirrors() {   // (pools_mu held)
    while (!dead_mirrors.empty() && dead_mirrors.front().value <= mem_completed) {
        if (env_on("RX_VK_LOG_ADDR")) {
            VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            da.buffer = dead_mirrors.front().buffer;
            logf("vk: addr mirror freed at gpu %#llx (value %llu, completed %llu)", (unsigned long long)vkGetBufferDeviceAddress(dev.device, &da),
                 (unsigned long long)dead_mirrors.front().value, (unsigned long long)mem_completed);
        }
        release_texel_buffer_views(dead_mirrors.front().buffer);   // (a texel view made over the mirror goes with it)
        destroy_buffer(dead_mirrors.front().buffer, dead_mirrors.front().alloc);
        dead_mirrors.pop_front();
    }
    while (!dead_imports.empty() && dead_imports.front().value <= mem_completed) {
        destroy_import_buffer(dead_imports.front().buffer);
        vkFreeMemory(dev.device, dead_imports.front().memory, nullptr);
        for (VkDeviceMemory m : dead_imports.front().pieces) vkFreeMemory(dev.device, m, nullptr);
        st_retired_imports_freed++, st_retired_import_bytes_freed += dead_imports.front().bytes;
        dead_import_bytes -= std::min(dead_import_bytes, dead_imports.front().bytes);
        dead_imports.pop_front();
    }
    while (msync_arenas.size() > 4 && msync_arenas.front().value <= mem_completed) {
        destroy_buffer(msync_arenas.front().buffer, msync_arenas.front().allocation);
        msync_arenas.erase(msync_arenas.begin());
    }
}
// ---- (Hookshot, 2026-11-14) RX_VK_POOL_MIRROR=1 (default 0): a mirror per pool, the pool never imported ----
// The import (VK_EXT_external_memory_host) pins the game's pages: in the owner's play10 session 3.85 GB of imports held the live
// pools in RAM (2.62 GB) and 1.13 GB of memory no pool covers (the 16 MB chunks' reach), which neither cold pages nor the drop can
// take (the substrate refuses a granule ever imported). Here a pool the GPU reads as a buffer gets a table entry with no import, cut
// to the game's pool; its mirror is made at its first binding and filled from guest memory through a host-visible staging buffer
// (a CPU copy on the replay's thread, then a GPU copy into the mirror, in the uploads of the recording); the pages the CPU writes
// later come from the captures as before, and the "live" ones (no kept capture holds them) through staging too. What still
// imports (counted by why, the pool then marked imported for good): a binding while the pool has no mirror and none can be made
// now (the mirrors' budget, a parallel segment, RX_VK_MIRROR_DEFER), a binding for writing (its write-backs go through the import;
// the staged write-back is the next step), readback refreshes and texel views over the pool. Never-imported mirrors are not given
// back as idle (the GPU would have nothing to read there). RX_VK_POOL_MIRROR_CHECK=1 compares mirrors with guest memory (below).
void pool_mirror_check_start();   // (RX_VK_POOL_MIRROR_CHECK: at the end of this file)
extern int64_t st_sw_runs, st_sw_bytes, st_sw_pages_written, st_sw_pages_cpu, st_sw_pages_dead, st_sw_pages_same;   // (the staged write-backs: below)
namespace {   // (with their declarations at the top)
}  // namespace

bool system_heap_usage(uint64_t* used, uint64_t* budget);   // (vk_vma.cpp; said here too: the pressure probe below runs before imports_give_back's own forward of it)
#ifdef __linux__
// A replacement can need both imports and both mirrors until the inheritance copy completes.
// Near RADV's system-heap budget, make room before importing. Only completed capture spares
// can be destroyed here. The budget and spare sizes are estimates, so leave the ordinary
// import and fallback in charge when there is too little reclaimable memory.
bool allocation_in_vram(VmaAllocation allocation);   // (vk_vma.cpp: in a device-local heap)
namespace {   // (declared in the anonymous namespace above import_dma_buf, its one caller)
void radv_import_headroom(uint64_t bytes) {
    constexpr uint64_t large_import = 64ull << 20, submit_reserve = 64ull << 20;
    if (dev.driver_id != VK_DRIVER_ID_MESA_RADV || bytes < large_import) return;
    std::lock_guard lk(cap_free_mu);
    if (cap_free.empty()) return;
    uint64_t used = 0, budget = 0;
    if (!system_heap_usage(&used, &budget)) return;
    const uint64_t keep = budget - std::min(budget, bytes + submit_reserve);
    const uint64_t need = used - std::min(used, keep);
    uint64_t released = 0;
    for (size_t i = cap_free.size(); i > 0 && released < need;) {
        const FreeBuf spare = cap_free[--i];
        if (allocation_in_vram(spare.alloc)) continue;
        cap_free.erase(cap_free.begin() + i);
        cap_free_bytes -= spare.size;
        destroy_buffer(spare.buffer, spare.alloc);
        released += spare.size;
    }
    if (need) {
        static unsigned logged = 0;   // (cap_free_mu)
        if (logged++ < 16)
            logf("vk: RADV system heap %llu of %llu MB before %llu MB import: released %llu MB of capture spares, requested %llu MB",
                 (unsigned long long)(used >> 20), (unsigned long long)(budget >> 20), (unsigned long long)(bytes >> 20),
                 (unsigned long long)(released >> 20), (unsigned long long)(need >> 20));
    }
}
}  // namespace
#endif

/// Whether retired imports are kept, given back at once, or given back after a wait (retired_give_back.h)
retired_give_back::Mode retired_mode() {
    // (hookshot) RX_VK_KEEP_RETIRED=1 keeps them for the session and 0, off, false or no gives them back as soon as the
    // GPU is past them, on every OS and vendor. Unset or empty, the defaults are main's but for one: off Windows at once
    // (the amdgpu submit above, issues 312 and 413), on Windows NVIDIA at once too (kept, ~20 areas of play left 6.8 GB
    // of retired pools imported on an RTX 3070 (8 GB) and the next area lost the device, twice; issue 1057), and on
    // Windows AMD and Intel after a wait, where they were kept for the session: an Intel Arc B390 lost the device
    // entering Kokiri every time with the give-back and 1 run in 5 without (issue 17, fuxau), and the AMD device losses
    // on the Ally X (309) and RX 6600 (571) came right after retired pools went back. Kept, those two vendors' retired
    // pools only grew, multi-GB in a long session, until the system heap's budget was passed and new imports failed
#ifdef _WIN32
    constexpr bool windows = true;
#else
    constexpr bool windows = false;
#endif
    static const retired_give_back::Mode m = [] {
        const char* v = getenv("RX_VK_KEEP_RETIRED");
        return retired_give_back::mode(v && *v, env_on("RX_VK_KEEP_RETIRED"), windows, dev.props.vendorID == 0x10DE);
    }();
    return m;
}
void retired_imports_give_back() {   // (pools_mu held)
    using retired_give_back::Mode;
    const Mode mode = retired_mode();
    // (hookshot) The wait (Windows AMD and Intel, retired_mode): a retired pool's import is given back once the GPU is
    // past it and it has been retired for RX_VK_RETIRE_SECS seconds (default 30), oldest first; while the system heap
    // is past RX_VK_IMPORT_HIGH percent of its budget (default 85; 0: off) the wait is RX_VK_RETIRE_MIN_SECS seconds
    // (default 2), until what brings it down to RX_VK_IMPORT_LOW percent (75) is queued. The free itself still waits
    // for the GPU (dead_imports). Elsewhere those switches are not read and nothing waits. Unlike imports_give_back's
    // valve this one is on by default on Windows AMD and Intel and on any budget, not only under 12 GB: it frees pools nothing
    // uses any more, where that one takes pools still live but idle
    static const uint64_t idle_ms = (uint64_t)std::max(1, env_int("RX_VK_RETIRE_SECS", 30)) * 1000;
    static const uint64_t floor_ms = (uint64_t)std::max(0, env_int("RX_VK_RETIRE_MIN_SECS", 2)) * 1000;
    static const int high = std::clamp(env_int("RX_VK_IMPORT_HIGH", 85), 0, 100);   // (0: off, as in imports_give_back)
    static const int low = std::clamp(env_int("RX_VK_IMPORT_LOW", 75), 10, 95);
    // (the driver asked every 16th submission, as imports_give_back does. What is already queued counts as given back:
    // the heap's use does not go down until the GPU is past it, and the next probe would queue for it again)
    uint64_t want = 0;
    if (mode == Mode::GRACE && high > 0) {
        static uint64_t calls = 0;
        uint64_t used = 0, budget = 0;
        if (++calls % 16 == 0 && system_heap_usage(&used, &budget))
            want = retired_give_back::want(retired_give_back::settled(used, dead_import_bytes), budget, high, low);
    }
    const int64_t now = mode == Mode::GRACE ? import_now_ms() : 0;
    uint64_t retired_bytes = 0;
    size_t retired_alive = 0;
    // (this runs at every part of a submission: the retired pools nothing needs any more are set aside, retired_compact below,
    // so the walk is over those still holding something)
    retired_test_pad();
    const auto walk_t0 = std::chrono::steady_clock::now();
    for (auto& nh : retired_pools) {   // (in the order they were retired: the oldest first)
        Pool& p = nh.mapped();
        if (!p.memory) continue;
        const uint64_t idle = (uint64_t)std::max<int64_t>(0, now - p.retired_ms);
        // (hookshot) a pool whose mirror was refused (over 1 GB, or no room in VRAM when it was asked for) has no mirror either:
        // its import goes as a MIRROR_NONE pool's does, as imports_give_back already lets it. Left out here, every such retired
        // pool kept its import, pinned system memory the driver counts as the GPU's shared memory, for the rest of the session;
        // the more of them the fuller VRAM was when they were refused
        if ((p.mstate == Pool::MIRROR_NONE || p.mstate == Pool::MIRROR_REFUSED) && !p.inherit_pending &&
            p.retired_value <= mem_completed &&
            p.last_value <= mem_completed && retired_give_back::due(mode, idle, idle_ms, floor_ms, want > 0)) {
            dead_imports.push_back({p.buffer, p.memory, std::max(mem_rec_value, p.last_value), p.lo, p.hi - p.lo, std::move(p.pieces)});
            dead_import_bytes += p.hi - p.lo;
            p.pieces.clear();
            p.buffer = VK_NULL_HANDLE, p.memory = VK_NULL_HANDLE, p.address = 0;
            pools_gen++;
            want -= std::min(want, p.hi - p.lo);   // (the short wait ends once the low watermark's worth is queued)
            continue;
        }
        retired_alive++, retired_bytes += p.hi - p.lo;
    }
    st_retired_walk_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - walk_t0).count();
    retired_compact();
    static uint64_t logged_epoch;
    if (mirror_epoch - logged_epoch >= 1200) {
        logged_epoch = mirror_epoch;
        uint64_t live_bytes = 0;   // (only this line reads it: every live pool walked for it at every part, before)
        for (auto& [lo, p] : imported) live_bytes += p.hi - p.lo;
        logf("vk: imports: %zu live (%llu MB), %zu retired still imported (%llu MB), %llu retired given back (%llu MB)",
             imported.size(), (unsigned long long)(live_bytes >> 20), retired_alive, (unsigned long long)(retired_bytes >> 20),
             (unsigned long long)st_retired_imports_freed, (unsigned long long)(st_retired_import_bytes_freed >> 20));
        // (hookshot) the retired pools walked and set aside (retired_compact), and the time their walks took since the last line
        logf("vk: retired pools: %zu walked, %zu set aside (%llu KB of their vectors let go); %.2f ms in their walks over the "
             "last 1200 submissions; page generation %u", retired_pools.size(), retired_done.size(),
             (unsigned long long)(st_retired_vec_bytes >> 10), st_retired_walk_ns.exchange(0) / 1e6, gen_now);
    }
}
// ---- (Hookshot) imports given back once the mirror holds the pool: RX_VK_NO_HOST_IMPORT=1 (default 0) ----
// The driver locks every page of an import (VK_EXT_external_memory_host): 1.4-1.8 GB of the game's memory pinned in RAM, which
// cold pages cannot take out of the working set and the guest-pages substrate cannot give back (memory-boundary.md 11.3, 14).
// Once a pool has its VRAM mirror the import is read only for: the mirror's fill (done), syncs of pages the CPU wrote that no
// kept capture holds ("live" pages), the write-backs of what the GPU wrote (pools bound for writing), readback pages (written
// by the GPU too), and bindings while the pool has no usable mirror (made later, given back as idle, a parallel segment). The
// GPU's texture paths read no import here (the counters "GPU reads of imported guest memory": texel views and unswizzles 0 MB).
// So a mirrored pool never bound for writing, whose mirror's fill is done, gives its import back (freed once the GPU is past
// everything that used it; the host is told then, so its pages are no longer counted pinned), and any of the uses above
// imports it again first (import_again: counted by why). A pool imported again is kept 600 submissions before its import may
// go again, and for good after its third time: a pool whose live pages come every frame keeps its import, as before.
namespace {   // (with import_again's declaration above)
constexpr uint64_t kAgainHoldEpochs = 600;
constexpr uint32_t kAgainGiveUp = 3;
int64_t st_released = 0, st_import_released_bytes = 0, st_again[kAgainKinds] = {}, st_again_failed = 0, st_unpinned_bytes = 0;
double st_again_ms = 0, st_again_max_ms = 0;


}   // namespace


void mirror_free(Pool& p) {
    p.was_mirrored = true;
    {   // (the table back to guest memory, for a pool still in use only: a retired one's range is another pool's now, and its
        // address there sent the foliage culling into freed memory, a device lost)
        const auto live = imported.find(p.lo);
        if (live != imported.end() && &live->second == &p) guest_pages_set(p.lo, p.hi, p.address);
    }
    const uint64_t size = mirror_size(p);
    if (p.sparse) {   // (its buffer, then its blocks' memory: sparse_reap. A released block's is in dead_blocks already)
        dead_sparse.push_back({p.mirror, std::max(mem_rec_value, p.last_value), {}, sparse_tl_value.load(std::memory_order_relaxed)});
        if (sparse_spare_cap() && p.sparse_made) {   // (sparse_spare.h: kept once the GPU is past it, what is bound to it now unbound then)
            DeadSparse& d = dead_sparse.back();
            d.spare = {p.mirror, p.sparse_made, p.sparse_size, p.sparse_block_bytes, p.sparse_types, 0};
            const uint64_t b = p.sparse_block_bytes;
            sparse_mirror::runs(
                p.blocks.size(), [&](size_t i) { return p.blocks[i] || (i < p.released.size() && p.released[i]); }, true,
                [&](size_t first, size_t count) {
                    const uint64_t at = (uint64_t)first * b, end = std::min(p.sparse_size, (uint64_t)(first + count) * b);
                    if (end > at) d.unbind.push_back({at, end - at, VK_NULL_HANDLE, 0, 0});
                });
        }
        for (VmaAllocation& a : p.blocks)
            if (a) dead_sparse.back().blocks.push_back(a), a = nullptr;
        for (VmaAllocation& a : p.released)   // (given back, still bound: freed with the buffer, after it)
            if (a) dead_sparse.back().blocks.push_back(a), a = nullptr;
        for (DeadBlock& d : dead_blocks)   // (and their entries never read this Pool again: a pool with no mirror can be evicted)
            if (d.pool == &p) d.pool = nullptr;
        std::erase(sparse_wanted, &p);
        p.sparse = false, p.any_wanted = false;
        p.blocks.clear(), p.has.clear(), p.wanted.clear(), p.refill.clear(), p.block_dead_since.clear();
        p.released.clear(), p.release_gen.clear(), p.last_bound.clear();
        p.resident_bytes = 0;
        p.sparse_made = 0;
    } else {
        dead_mirrors.push_back({p.mirror, p.mirror_alloc, std::max(mem_rec_value, p.last_value)});
    }
    shadow_free(p);
    p.mirror = VK_NULL_HANDLE, p.mirror_alloc = nullptr;
    mirror_bytes -= std::min(mirror_bytes, size);
    st_mirror_bytes_freed += size;
    st_mirrors_freed++;
    p.mstate = Pool::MIRROR_NONE;
    pools_gen++;   // (a parallel replay thread's pool cache: looked up again)
}
// A live pool's mirror holding GPU writes left out of the write-back was never given back. One idle as
// long as the others that go (kLongIdle, or kIdleEpochs when the mirrors' budget is short) is listed here; the next submission's
// end copies its left-out pages (the CPU's pages and words, and texture storage, cut out, as the write-backs do) from the mirror
// into guest memory and then frees the mirror: the buffer is destroyed once the GPU is past that copy (mirror_free's value), so
// a pool bound later is synced from guest memory holding the GPU's words. RX_VK_KEEP_LAZY_MIRRORS=1: kept, as before
std::vector<Pool*> lazy_release;
int64_t st_lazy_released = 0;
uint64_t st_lazy_released_bytes = 0, st_lazy_written_back = 0;
void mirrors_give_back(uint64_t want_bytes) {   // (pools_mu held)
    // (the mirrors still given back, why, and what the GPU did to them: the first 200, to see which went before a picture fault)
    static int logged_frees = 0;
    const auto told = [](const Pool& p, const char* why) {
        if (logged_frees++ < 200)
            logf("vk: mirror of pool %#llx (%llu MB) given back: %s, %llu submissions since its sync%s%s", (unsigned long long)p.lo,
                 (unsigned long long)((p.hi - p.lo) >> 20), why, (unsigned long long)(mirror_epoch - p.synced_epoch),
                 p.gpu_only ? ", GPU-only data" : "", p.gpu_written ? ", GPU-written" : "");
    };
    for (auto& nh : retired_pools) {
        Pool& p = nh.mapped();
        // (EvilOOT: kept while the pool that took it in has not made its mirror, for 600 submissions at most)
        if (p.inherit_pending && mirror_epoch - p.retired_epoch > kIdleEpochs) p.inherit_pending = false, p.inherit_lost = true;
        if (p.mstate == Pool::MIRROR_ON && p.retired_value <= mem_completed && !p.inherit_pending) {
            told(p, p.inherit_lost ? "retired, not inherited in time" : "retired");
            mirror_free(p);
        }
    }
    // (hookshot) not a pool the GPU wrote past what is copied back (gpu_only), nor one it wrote through a binding (gpu_written): made
    // again, its mirror came from guest memory without that, and the GPU went on from stale data. Paused 1-2 minutes in the background, the idle pools' mirrors all went,
    // and the game came back with cyan for a few seconds (RX 5500M; gone with every mirror kept). A pool refused a mirror for
    // the budget meanwhile is read in system memory, slower, not wrong
    std::vector<Pool*> idle;
    uint64_t kept_bytes = 0;
    for (auto& [lo, p] : imported)
        if (p.mstate == Pool::MIRROR_ON && mirror_epoch - p.synced_epoch > kIdleEpochs && p.last_value <= mem_completed) {
            if (p.gpu_only || p.gpu_written) kept_bytes += p.hi - p.lo;
            else idle.push_back(&p);
        }
    std::sort(idle.begin(), idle.end(), [](const Pool* a, const Pool* b) { return a->synced_epoch < b->synced_epoch; });
    for (Pool* p : idle) {
        const bool long_idle = mirror_epoch - p->synced_epoch > kLongIdle;
        if (long_idle || mirror_bytes + want_bytes > mirror_budget()) told(*p, long_idle ? "idle" : "idle, for room"), mirror_free(*p);
    }
    static int64_t logged_at;
    if (st_mirrors_freed / 16 != logged_at) {
        logged_at = st_mirrors_freed / 16;
        logf("vk: %lld mirrors given back (%llu MB in all); %llu MB of mirrors now, %llu MB of them idle and kept for what the GPU "
             "wrote there alone", (long long)st_mirrors_freed, (unsigned long long)(st_mirror_bytes_freed >> 20),
             (unsigned long long)(mirror_bytes >> 20), (unsigned long long)(kept_bytes >> 20));
    }
}

// ---- imports given back under pressure. The driver pins every imported pool and counts it in the system heap
// the GPU reaches, whose budget is about half the PC's RAM (7.5 GB of 16, 15.4 GB of 32): imports were kept for the session,
// the heap went past its budget (7758 of 7503 MB on an RTX 3060 with 16 GB) and new imports failed with
// VK_ERROR_OUT_OF_DEVICE_MEMORY: terrain left out, load zones stuck; on 32 GB+ Linux PCs (where the budget is 12 GB or more
// and this valve was off) 200+ idle unmirrored pools piled up over a two-hour session (7.4 GB live), bloating every amdgpu
// submit's buffer list and the submit-time capture scan. Past RX_VK_IMPORT_HIGH percent (85) of the effective budget the pools
// bound least lately give their imports back down to RX_VK_IMPORT_LOW percent (75): only pools without a mirror, not bound for
// kImportIdle submissions and done on the GPU. Under a 12 GB heap budget (16 GB PCs and the Steam Deck) the effective budget is
// the heap's own, as before; at 12 GB and above it is capped at 4500 MB by default (RX_VK_IMPORT_CAP_MB=<MB> sets another cap,
// and RX_VK_IMPORT_CAP_MB=0 turns the >= 12 GB cap off; setting RX_VK_IMPORT_HIGH or RX_VK_IMPORT_LOW explicitly without
// RX_VK_IMPORT_CAP_MB uses the full heap budget instead of the 4500 MB cap). They leave the table as a retired pool does (the
// page table's entries 0, the caches emptied) and are imported again at their next binding; their memory is freed once
// everything recorded until now is done. On by default on Linux and on Windows NVIDIA; off by default on Windows AMD and Intel.
// RX_VK_IMPORT_HIGH=0: off, =85: on anywhere against the full budget. (Caveat: last_value counts bindings, not shaders
// reaching a pool through the page table; a pool only reached that way would read zeros until it is bound again. Not seen in
// play: the pools given back were long unbound.) ----
bool system_heap_usage(uint64_t* used, uint64_t* budget);   // (vk_vma.cpp)
std::deque<std::map<uint64_t, Pool>::node_type> evicted_pools;
uint64_t evicted_import_bytes = 0;
int64_t st_imports_evicted = 0;
uint64_t st_import_bytes_evicted = 0;
constexpr uint64_t kImportIdle = 600;



// (hookshot) The game frees an address region without unmapping its pages first, and a freed region's pages
// were only discarded: the regions' commit stood at their whole span (8 GB for the ~2.9 GB mapped at a time). What no import of
// the GPU holds of a freed region is decommitted now (the host's region_release_gaps): the game's thread only queues the range
// (memory_guest_release), and the next submission, which holds pools_mu, gives back the parts no import, retired import, dead
// import or evicted pool stands over; an import let go later gives back its own range then. RX_VK_KEEP_FREED=1: kept, as before
int64_t st_released_calls = 0;
uint64_t st_released_bytes = 0;
void release_uncovered(uint64_t lo, uint64_t hi) {   // (pools_mu held)
#ifdef _WIN32
    static const bool off = env_on("RX_VK_KEEP_FREED");
    if (off || !host->region_release_gaps || hi <= lo) return;
    std::vector<std::pair<uint64_t, uint64_t>> held;
    auto note = [&](uint64_t a, uint64_t b) {
        if (a < hi && b > lo) held.push_back({std::max(a, lo), std::min(b, hi)});
    };
    for (auto it = imported.lower_bound(hi); it != imported.begin();) {   // (the pools starting below hi, the last ones first)
        --it;
        if (it->second.hi + (1ull << 30) < lo) break;   // (imports are at most 1 GB: none further down reaches lo)
        note(it->second.lo, it->second.hi);
    }
    {
        const auto walk_t0 = std::chrono::steady_clock::now();
        for (const auto& nh : retired_pools)
            if (nh.mapped().memory) note(nh.mapped().lo, nh.mapped().hi);
        st_retired_walk_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - walk_t0).count();
    }
    for (const DeadImport& d : dead_imports) {
        if (!d.lo) return;   // (one whose place is not known: nothing given back this time)
        note(d.lo, d.lo + d.bytes);
    }
    for (const auto& nh : evicted_pools) note(nh.mapped().lo, nh.mapped().hi);
    std::sort(held.begin(), held.end());
    uint64_t at = lo, freed = 0;
    for (const auto& [a, b] : held) {
        if (a > at) freed += host->region_release_gaps(at, a);
        at = std::max(at, b);
    }
    if (hi > at) freed += host->region_release_gaps(at, hi);
    st_released_calls++, st_released_bytes += freed;
    static uint64_t logged_at;
    if (st_released_bytes >> 30 != logged_at) {   // (said at every GB)
        logged_at = st_released_bytes >> 30;
        logf("vk: the game's freed memory given back to Windows: %llu MB in all, over %lld freed regions and imports",
             (unsigned long long)(st_released_bytes >> 20), (long long)st_released_calls);
    }
#else
    (void)lo, (void)hi;
#endif
}
// (the game's thread only leaves the range: the replay holds pools_mu for much of a submission, and the game frees regions
// by the thousand while it loads. Taken up at the next submission; a range handed out again meanwhile is no gap by then)
std::mutex released_mu;
std::vector<std::pair<uint64_t, uint64_t>> released_queue;
void released_drain() {   // (pools_mu held)
    static std::vector<std::pair<uint64_t, uint64_t>> take;
    {
        std::lock_guard lk(released_mu);
        if (released_queue.empty()) return;
        const size_t n = std::min<size_t>(released_queue.size(), 512);   // (512 a submission at most)
        take.assign(released_queue.begin(), released_queue.begin() + n);
        released_queue.erase(released_queue.begin(), released_queue.begin() + n);
    }
    for (const auto& [lo, hi] : take) release_uncovered(lo, hi);
}

void imports_give_back() {   // (pools_mu held)
    while (!evicted_pools.empty() && evicted_pools.front().mapped().retired_value <= mem_completed) {
        Pool& p = evicted_pools.front().mapped();
        evicted_import_bytes -= std::min(evicted_import_bytes, p.hi - p.lo);
        destroy_import_buffer(p.buffer);
        vkFreeMemory(dev.device, p.memory, nullptr);
        for (VkDeviceMemory m : p.pieces) vkFreeMemory(dev.device, m, nullptr);   // (a pool imported as several dma-bufs)
        const uint64_t lo = p.lo, hi = p.hi;
        evicted_pools.pop_front();
        release_uncovered(lo, hi);   // (what the game freed under it goes back to Windows)
    }
    // (hookshot) Default by OS and vendor: on off Windows, and on Windows NVIDIA (below); off on Windows AMD and Intel (device
    // losses on Intel Arc and AMD right after imports went back, issues 17, 309 and 571): RX_VK_IMPORT_HIGH=85 turns it on
    // there. Retired imports have their own valve at the same watermarks (retired_imports_give_back), on by default on Windows
    // AMD and Intel whatever the budget: that one frees pools nothing uses any more, this one pools still live that have been idle
    static const char* env_high = getenv("RX_VK_IMPORT_HIGH");
    static const char* env_low = getenv("RX_VK_IMPORT_LOW");
    static const int cap_mb = env_int("RX_VK_IMPORT_CAP_MB",
        ::auto_settings::red_tunic_apu(dev.props.vendorID,
            dev.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 4096 : -1);
    // (hookshot, 11-13) On on Windows NVIDIA too now. Off there, the live pools only grew: 8 to 150 imports, 0.1 to 2.3 GB, over
    // four areas on an RTX card (none given back), the process's working set with them. On Windows AMD and Intel it stays off by
    // default until it has been run there. The device losses it was turned off for came, as the retired pools' did, right after
    // the memory went back on AMD's and Intel's Windows drivers; those pools are given back after a wait there (retired_mode
    // GRACE) without them, so where RX_VK_IMPORT_HIGH turns this valve on there, the pools it takes wait the same way: held, the
    // table already off them, until RX_VK_RETIRE_SECS (30) have passed since, and only then queued for the GPU (retired_value).
    // What waits is counted as given back, so the valve does not take more for the same bytes. RX_VK_IMPORT_HIGH=0: off
#ifdef _WIN32
    // (hookshot, 11-17: and on Windows AMD. Off, the live imports of a session had no bound there: with the retired ones,
    // 3.8 GB live and 3.1 GB retired on an RX 6600 XT (16 GB) after Kokiri, Kakariko and the Market, and past ~7 GB the driver
    // refused every new import (VK_ERROR_INVALID_EXTERNAL_HANDLE), the terrain's index buffers could not be bound and ~1,100
    // draws a frame were left out: Hyrule Field's ground gone. On, the same trip refused no import and lost no device)
    static const int high_default = dev.props.vendorID == 0x10DE || dev.props.vendorID == 0x1002 ? 85 : 0;
#else
    static const int high_default = 85;
#endif
    static const int high = std::clamp(env_int("RX_VK_IMPORT_HIGH", high_default), 0, 100);
    static const int low = std::clamp(env_int("RX_VK_IMPORT_LOW", 70), 10, 95);
    const bool grace = retired_mode() == retired_give_back::Mode::GRACE;
    uint64_t evicted_waiting = 0;
    if (grace) {
        static const int64_t wait_ms = (int64_t)std::max(1, env_int("RX_VK_RETIRE_SECS", 30)) * 1000;
        const int64_t now = import_now_ms();
        for (auto& nh : evicted_pools) {
            Pool& p = nh.mapped();
            if (p.retired_value == ~0ull && now - p.retired_ms >= wait_ms) p.retired_value = mem_rec_value;   // (its wait is over)
            if (p.retired_value == ~0ull) evicted_waiting += p.hi - p.lo;
        }
    }
    static uint64_t calls = 0;
    if (high <= 0 || ++calls % 16) return;   // (the driver asked every 16 submissions)
    uint64_t used = 0, budget = 0;
    if (!system_heap_usage(&used, &budget)) return;
    const uint64_t cap = retired_give_back::live_budget(budget, (env_high && *env_high) || (env_low && *env_low), cap_mb);
    static const bool logged_cfg = [&] {
        if (!cap)
            logf("vk: imports given back past %d%% of a %llu MB system heap budget: off (RX_VK_IMPORT_CAP_MB=0)", high,
                 (unsigned long long)(budget >> 20));
        else if (cap < budget)
            logf("vk: imports given back past %d%% of a %llu MB cap (%llu MB system heap budget): on", high,
                 (unsigned long long)(cap >> 20), (unsigned long long)(budget >> 20));
        else
            logf("vk: imports given back past %d%% of a %llu MB system heap budget: on", high, (unsigned long long)(budget >> 20));
        return true;
    }();
    (void)logged_cfg;
    // (with both valves on, the other ran just before, on this same submission (the two count alike): what it queued is
    // still in the driver's figure until the GPU is past it, and live pools would go for the same bytes. Only where
    // retired imports wait; elsewhere they go back at every submission and this valve reads what it always read)
    if (retired_mode() == retired_give_back::Mode::GRACE) used = retired_give_back::settled(used, dead_import_bytes);
    if (grace) used = retired_give_back::settled(used, evicted_waiting);   // (held for their wait: given back already)
    const uint64_t want = retired_give_back::want(used, cap, high, low);
    // (and under Linux memory pressure, once per /proc/meminfo sample (vk_texture.cpp texture_frame): the shortfall less what is
    // on its way back already, and over one episode of pressure no more in all than its deepest shortfall: linux_pressure.h
    // episode_want. Said in lines of its own, not as the system heap's)
    uint64_t linux_want = 0, over_seq = 0;
    static uint64_t last_over_seq = 0, episode_deepest = 0, episode_given = 0;
    if (const uint64_t over = linux_pressure_shortfall(&over_seq)) {
        if (over_seq != last_over_seq) {
            last_over_seq = over_seq;
            cap_free_trim(32ull << 20);
            episode_deepest = std::max(episode_deepest, over);
            linux_want = linux_pressure::episode_want(linux_pressure::import_want(over, dead_import_bytes + evicted_import_bytes),
                                                      episode_deepest, episode_given);
        }
    } else {
        episode_deepest = episode_given = 0;   // (out of pressure: the next episode starts afresh)
    }
    const bool for_linux = linux_want > want;
    const uint64_t want_bytes = std::max(want, linux_want);
    if (!want_bytes) return;
    std::vector<std::map<uint64_t, Pool>::iterator> idle;
    for (auto it = imported.begin(); it != imported.end(); ++it) {
        const Pool& p = it->second;
        if ((p.mstate == Pool::MIRROR_NONE || p.mstate == Pool::MIRROR_REFUSED) && p.last_value <= mem_completed &&
            mem_rec_value > p.last_value + kImportIdle)
            idle.push_back(it);
    }
    std::sort(idle.begin(), idle.end(), [](auto a, auto b) { return a->second.last_value < b->second.last_value; });
    uint64_t freed = 0;
    int n = 0;
    for (auto it : idle) {
        if (freed >= want_bytes) break;
        Pool& p = it->second;
        guest_pages_set(p.lo, p.hi, 0);
        p.retired_value = mem_rec_value;   // (this submission's recording is being made: freed once it is done)
        if (grace) p.retired_value = ~0ull, p.retired_ms = import_now_ms();   // (AMD, Intel on Windows: after the wait, above)
        freed += p.hi - p.lo;
        evicted_import_bytes += p.hi - p.lo;
        n++;
        evicted_pools.push_back(imported_take(it));
    }
    if (for_linux) episode_given += freed - std::min(freed, want);
    if (!n) {
        static int logged_none;
        if (logged_none++ < 8) {
            if (for_linux)
                logf("vk: memory pressure on Linux: no idle import to give back for %llu MB (%zu imported)",
                     (unsigned long long)(linux_want >> 20), imported.size());
            else
                logf("vk: system heap %llu of %llu MB: no idle import to give back (%zu imported)", (unsigned long long)(used >> 20),
                     (unsigned long long)(budget >> 20), imported.size());
        }
        return;
    }
    pools_gen++;
    last_pool = nullptr;
    std::fill(std::begin(pool_cache), std::end(pool_cache), nullptr);
    if (mirror_on()) publish_cap_ranges();
    st_imports_evicted += n, st_import_bytes_evicted += freed;
    if (for_linux) {
        static int logged_linux;
        if (logged_linux++ < 8 || logged_linux % 16 == 0)
            logf("vk: memory pressure on Linux: %d idle imports given back (%llu MB of %llu MB asked; %llu MB in this episode, "
                 "deepest shortfall %llu MB); %lld (%llu MB) in all (RX_VK_AVAIL_RESERVE_MB, RX_VK_COMMIT_RESERVE_MB)",
                 n, (unsigned long long)(freed >> 20), (unsigned long long)(linux_want >> 20),
                 (unsigned long long)(episode_given >> 20), (unsigned long long)(episode_deepest >> 20),
                 (long long)st_imports_evicted, (unsigned long long)(st_import_bytes_evicted >> 20));
    } else {
        static int logged;
        if (logged++ < 40 || logged % 64 == 0)
            logf("vk: system heap %llu of %llu MB: %d idle imports given back (%llu MB); %lld (%llu MB) in all",
                 (unsigned long long)(used >> 20), (unsigned long long)(budget >> 20), n, (unsigned long long)(freed >> 20),
                 (long long)st_imports_evicted, (unsigned long long)(st_import_bytes_evicted >> 20));
    }
}

// ---- (hookshot) an import trimmed back to the game's live pools (mirror_trim.h) ----
// Every 64 submissions, while VRAM is short: the mirrored import that has been mostly outside every game pool the longest
// (mirror_trim::wait) is retired, and each of its live parts imported again on its own, with a mirror wanted at once: made in
// this same submission's uploads (memory_begin_submission, right after), each takes what the GPU wrote from the old mirror
// (mirror_make's inherit, as for a pool made again larger), and the page table points at the new mirrors before anything is
// recorded. One import a call: each new part is an import in the driver, on the replay thread. Nothing is done unless the new
// mirrors fit the budget beside the old one and the mirrors already wanted, which come first. The old one goes once every part
// has its own mirror and the GPU is done with it (trims_settle).
int64_t st_trims = 0;
uint64_t st_trim_bytes = 0;
int trims_waiting = 0;   // (retired by a trim, their mirror still wanted by a part without its own: trims_settle)
bool mirrors_trim() {   // (pools_mu held; true when one was trimmed: its parts are in mirror_wanted)
    const uint64_t chunk = import_chunk();
    if (!chunk || mirror_epoch % 64 || par_active.load(std::memory_order_relaxed)) return false;
    uint64_t used = 0, allowed = 0;
    const bool pressed = vram_usage(&used, &allowed) && mirror_trim::pressed(used, allowed);
    Pool* pick = nullptr;
    std::vector<mirror_trim::Range> parts, pick_parts;
    for (auto& [lo, p] : imported) {
        const unsigned trims = mirror_trim::counted(p.trims, mirror_epoch - p.trim_epoch);
        if (!pressed || p.mstate != Pool::MIRROR_ON || !mirror_trim::wait(trims)) {
            p.dead_since = 0;
            continue;
        }
        parts = mirror_trim::live(p.lo, p.hi, chunk, mirror_trim::kJoin, guest_pools);
        if (!mirror_trim::worth(p.hi - p.lo, parts)) {
            p.dead_since = 0;
            continue;
        }
        if (!p.dead_since) p.dead_since = mirror_epoch;
        if (mirror_epoch - p.dead_since < mirror_trim::wait(trims)) continue;
        if (!pick || p.dead_since < pick->dead_since) pick = &p, pick_parts = parts;
    }
    if (!pick) return false;
    const uint64_t kept = mirror_trim::bytes(pick_parts), size = pick->hi - pick->lo;
    // (no room for both yet, beside the mirrors already wanted, which mirror_make takes first: looked at again in 64 submissions.
    // Nor while RX_VK_MIRROR_TIGHT's rule refuses every mirror: the parts would get none)
    uint64_t wanted = 0;
    for (const Pool* w : mirror_wanted) wanted += w->hi - w->lo;
    static const bool tight_rule = env_on("RX_VK_MIRROR_TIGHT");
    if (mirror_bytes + wanted + kept > mirror_budget() || (tight_rule && used > allowed / 100 * 90)) return false;
    std::vector<Pool> made;
    for (const mirror_trim::Range& r : pick_parts) {   // (all of them first: one that fails leaves everything as it was)
        Pool np;
        if (!import_pool(r.lo, r.hi, np)) {
            for (Pool& m : made) {
                destroy_import_buffer(m.buffer);
                vkFreeMemory(dev.device, m.memory, nullptr);
                for (VkDeviceMemory piece : m.pieces) vkFreeMemory(dev.device, piece, nullptr);
            }
            pick->dead_since = mirror_epoch;   // (not again for a while)
            return false;
        }
        made.push_back(std::move(np));
    }
    const uint64_t olo = pick->lo, ohi = pick->hi, waited = mirror_epoch - pick->dead_since;
    const bool gpu_only = pick->gpu_only, gpu_written = pick->gpu_written;
    const uint8_t trims = (uint8_t)(mirror_trim::counted(pick->trims, mirror_epoch - pick->trim_epoch) + 1);
    retire_pool(imported.find(olo));   // (the page table's entries of its range 0, its shadow freed, its mirror kept to inherit)
    retired_pools.back().mapped().trimmed = true;
    trims_waiting++;
    for (Pool& m : made) {
        m.gpu_only = gpu_only, m.gpu_written = gpu_written, m.trims = trims, m.trim_epoch = mirror_epoch;
        auto it = imported_add(m);
        guest_pages_set(it->second.lo, it->second.hi, it->second.address);
        it->second.mstate = Pool::MIRROR_WANTED;
        mirror_wanted.push_back(&it->second);
    }
    st_trims++, st_trim_bytes += size - kept;
    static int logged;
    if (logged++ < 40 || st_trims % 64 == 0)
        logf("vk: pool %#llx..%#llx (%llu MB) trimmed to its %zu live part(s), %llu MB: the rest outside every game pool for %llu "
             "submissions (%lld trims, %llu MB in all)", (unsigned long long)olo, (unsigned long long)ohi,
             (unsigned long long)(size >> 20), made.size(), (unsigned long long)(kept >> 20), (unsigned long long)waited,
             (long long)st_trims, (unsigned long long)(st_trim_bytes >> 20));
    return true;
}

// (pools_mu held, after the submission's mirrors are made) A trimmed import's old mirror waits to be inherited until every pool
// over its range has a mirror of its own. A part refused one (the budget, RX_VK_MIRROR_TIGHT, an allocation that failed) points
// at its import meanwhile; when its mirror is made later, mirror_make still copies in what the GPU wrote there from the old
// mirror, as for a pool made again larger. Let go at once, what the GPU wrote in that part and nowhere else was lost. A part
// refused for a failed allocation is tried again at its next binding while the old mirror waits. kIdleEpochs still bounds the
// wait (mirrors_give_back: inherit_lost)
void trims_settle() {
    if (!trims_waiting) return;
    for (auto& nh : retired_pools) {
        Pool& r = nh.mapped();
        if (!r.trimmed) continue;
        bool all = true;
        if (r.inherit_pending && r.mstate == Pool::MIRROR_ON) {
            auto it = imported.lower_bound(r.lo);
            if (it != imported.begin() && std::prev(it)->second.hi > r.lo) --it;
            for (; it != imported.end() && it->first < r.hi; ++it) {
                Pool& q = it->second;
                if (q.hi <= r.lo || q.mstate == Pool::MIRROR_ON) continue;
                all = false;
                if (q.mstate == Pool::MIRROR_REFUSED && q.hi - q.lo <= (1ull << 30)) q.mstate = Pool::MIRROR_NONE;
            }
        }
        if (!all) {
            static int logged;
            if (logged++ < 16)
                logf("vk: trimmed pool %#llx..%#llx: its mirror kept, a live part has none of its own yet", (unsigned long long)r.lo,
                     (unsigned long long)r.hi);
            continue;
        }
        r.trimmed = false, r.inherit_pending = false;   // (every part has its share, or the wait is over: it goes as any retired one)
        trims_waiting--;
    }
}

struct FinNote { uint64_t lo, hi, after; };
std::mutex fin_mu;
std::vector<FinNote> fin_notes;   // (fin_mu)
int64_t st_fin_notes = 0, st_fin_retired = 0, st_fin_kept_live = 0, st_fin_waited_bound = 0;
uint64_t st_fin_bytes = 0, st_fin_mirror_bytes = 0, st_fin_import_bytes = 0;

// (hookshot, cards of 4-6 GB) The mirrors by heat. A pool gets its mirror at its first binding while the budget has room, so the
// pools bound first took it all: on a 4 GB RTX 3050 Ti (256-512 MB of mirrors) 23 small pools held the mirrors, read ~0.4 GB
// over 600 submissions, while two pools of 160 and 132 MB, 96% of the 21.9 GB the GPU read over PCIe in those submissions,
// stayed WANTED for good: mirrors_give_back only frees mirrors idle for kIdleEpochs, and the small ones are bound every frame.
// Here a wanted pool that does not fit takes the room of mirrored pools bound at least four times less (heat), freed as the
// idle ones are: done on the GPU, not GPU-only or GPU-written data, no inheritance pending (read in system memory after, slower,
// not wrong). Every 64 submissions, the hottest wanted pool first. On by default on cards of 6.5 GB or less (mirror_budget's rule
// for them), where the budget is what keeps the hot pools out; measured only on a 4 GB card, so larger cards keep first come, first
// served: their budget and the last 30 % mirror_small_cap keeps are not measured with it. RX_VK_MIRROR_HEAT=0 / 1: off / on on any card
bool mirror_heat_on() {
    static const bool on = [] {
        if (const char* e = getenv("RX_VK_MIRROR_HEAT")) return !env_value_off(e);
        const uint64_t vram = device_vram_bytes();
        return vram && vram <= (6ull << 30) + (512ull << 20);
    }();
    return on;
}
void mirrors_by_heat() {   // (pools_mu held)
    std::vector<Pool*> wanted(mirror_wanted.begin(), mirror_wanted.end());
    std::sort(wanted.begin(), wanted.end(), [](const Pool* a, const Pool* b) { return a->heat > b->heat; });
    static int64_t logged, freed, made_room;
    for (Pool* w : wanted) {
        const uint64_t n = w->hi - w->lo, budget = mirror_budget(), wh = __atomic_load_n(&w->heat, __ATOMIC_RELAXED);
        if (!wh || n > budget || n > (1ull << 30) || mirror_bytes + n <= budget) continue;   // (fits already: mirror_make makes it)
        // (not hot: room made up to the budget is past the small cap, which mirror_make keeps from it, and the mirrors freed for
        // it would be made again at their next binding)
        if (wh < mirror_hot_bytes() && mirror_small_cap() < budget) continue;
        // (the room it needs from the mirrored pools that give the most room for the least heat, colder ones first by heat a
        // MB: a few small pools bound every frame can each be hotter than it, and the 23 of them held the room it needs. They
        // go only if all of their heat together is at most half of its own: what the GPU then reads over PCIe is less, and
        // they cannot take the room back from it, which would need twice its heat)
        std::vector<Pool*> cold;
        for (auto& [lo, p] : imported)
            if (p.mstate == Pool::MIRROR_ON && !p.gpu_only && !p.gpu_written && !p.inherit_pending && p.last_value <= mem_completed &&
                __atomic_load_n(&p.heat, __ATOMIC_RELAXED) < wh / 2)
                cold.push_back(&p);
        std::sort(cold.begin(), cold.end(), [](const Pool* a, const Pool* b) {   // (heat per MB of mirror given back)
            return (double)a->heat / (double)std::max<uint64_t>(1, mirror_size(*a)) <
                   (double)b->heat / (double)std::max<uint64_t>(1, mirror_size(*b));
        });
        uint64_t room = 0, cost = 0;
        size_t take = 0;
        while (take < cold.size() && mirror_bytes - std::min(room, mirror_bytes) + n > budget)
            room += mirror_size(*cold[take]), cost += __atomic_load_n(&cold[take]->heat, __ATOMIC_RELAXED), take++;
        if (mirror_bytes - std::min(room, mirror_bytes) + n > budget || cost > wh / 2) {   // (not enough colder room: as before)
            // (hookshot) The hottest pool left out, the mirrors full of ones the GPU writes (never given back: what it wrote is only
            // there): it gets room past the budget, once, while the card has it. On a 4 GB RTX 3050 Ti at 512 MB, a 132 MB pool
            // bound 7 times a submission was in VRAM in some starts and not in others, first come first served: the GPU 12.4-13.6 ms
            // a frame against 15.0 (standing, Kokiri Forest), and with a budget of 640 MB for all pools the textures ran short.
            // RX_VK_MIRROR_HEAT_EXTRA=1 (a test, off: measured on one card only): on
            static const bool extra_on = env_on("RX_VK_MIRROR_HEAT_EXTRA");
            uint64_t used = 0, allowed = 0;
            if (extra_on && !mirror_heat_extra && w == wanted.front() && n <= (256ull << 20) && wh >= (16ull << 20) &&
                vram_usage(&used, &allowed) && allowed && used + n + (64ull << 20) <= allowed / 100 * 92) {
                mirror_heat_extra = n;
                logf("vk: mirrors by heat: pool %#llx (%llu MB, heat %llu KB) given room past the budget (%llu MB more; VRAM %llu of "
                     "%llu MB used) (RX_VK_MIRROR_HEAT_EXTRA)", (unsigned long long)w->lo, (unsigned long long)(n >> 20),
                     (unsigned long long)(wh >> 10), (unsigned long long)(n >> 20), (unsigned long long)(used >> 20),
                     (unsigned long long)(allowed >> 20));
                continue;
            }
            static int told;
            if (n >= (32ull << 20) && told++ < 24)
                logf("vk: mirrors by heat: pool %#llx (%llu MB, heat %llu KB) given no room: %zu colder mirrors, %llu MB of them taken "
                     "(heat %llu KB), the mirrors at %llu MB of %llu", (unsigned long long)w->lo, (unsigned long long)(n >> 20),
                     (unsigned long long)(wh >> 10), cold.size(), (unsigned long long)(room >> 20), (unsigned long long)(cost >> 10),
                     (unsigned long long)(mirror_bytes >> 20), (unsigned long long)(budget >> 20));
            continue;
        }
        for (size_t i = 0; i < take; i++) {
            Pool* p = cold[i];
            if (logged++ < 40)
                logf("vk: mirror of pool %#llx (%llu MB, heat %llu KB) given back for pool %#llx (%llu MB, heat %llu KB): the GPU reads "
                     "that one more (RX_VK_MIRROR_HEAT=0: off)", (unsigned long long)p->lo, (unsigned long long)((p->hi - p->lo) >> 20),
                     (unsigned long long)(p->heat >> 10), (unsigned long long)w->lo, (unsigned long long)(n >> 20),
                     (unsigned long long)(wh >> 10));
            mirror_free(*p);
            freed++;
        }
        made_room++;
    }
    if (made_room && made_room % 16 == 1)
        logf("vk: mirrors by heat: room made %lld times, %lld colder mirrors given back", (long long)made_room, (long long)freed);
}

void memory_begin_submission() {
    std::lock_guard lk(pools_mu);
    reap_mirrors();
    sparse_reap();   // (hookshot, RX_VK_MIRROR_SPARSE: released blocks and freed sparse mirrors the GPU is past)
    retired_imports_give_back();
    mirror_epoch++;
    mirror_upload = true;
    mirror_synced.clear();
    writebacks.clear();
    imports_give_back();
    released_drain();   // (hookshot: the game's freed regions, given back where no import stands)
    if (tex_capture_on()) publish_tex_spans(texture_frame_no());
    if (!mirror_on()) return;
    mirrors_trim();
    if (mirror_heat_on()) {   // (hookshot: mirrors_by_heat; the heat decays by a quarter every 16 submissions)
        if (mirror_epoch % 16 == 0)
            for (auto& [lo, p] : imported) __atomic_store_n(&p.heat, __atomic_load_n(&p.heat, __ATOMIC_RELAXED) / 4 * 3, __ATOMIC_RELAXED);
        if (mirror_epoch % 64 == 0 && !mirror_wanted.empty()) mirrors_by_heat();
    }
    {
        uint64_t want = 0;
        for (Pool* p : mirror_wanted) want += p->hi - p->lo;
        if (mirror_epoch % 64 == 0 || mirror_bytes + want > mirror_budget()) mirrors_give_back(want);
    }
    if (mirror_sparse_on()) {   // (hookshot, RX_VK_MIRROR_SPARSE: blocks the game's pools left go, blocks bindings wanted come)
        if (mirror_epoch % 64 == 0) sparse_release();
        static uint64_t pools_seen = 0;   // (the game's pools changed: blocks a new one is in get memory before any work reaches them)
        if (const uint64_t g = guest_pools_gen.load(std::memory_order_relaxed); g != pools_seen) pools_seen = g, sparse_want_live();
        if (!sparse_wanted.empty()) {
            std::vector<Pool*> grow;
            grow.swap(sparse_wanted);
            std::erase_if(grow, [](Pool* p) {
                const auto live = imported.find(p->lo);
                if (live == imported.end() || &live->second != p || !p->sparse || p->mstate != Pool::MIRROR_ON) {
                    p->any_wanted = false;
                    if (p->sparse) std::fill(p->wanted.begin(), p->wanted.end(), 0), p->refill.clear();
                    return true;
                }
                return false;
            });
            sparse_grow_all(grow);
        }
        static uint64_t logged_epoch;
        if (mirror_epoch - logged_epoch >= 1200) {
            logged_epoch = mirror_epoch;
            uint64_t whole = 0, held = 0;
            size_t n = 0;
            for (auto& [lo, p] : imported)
                if (p.sparse && p.mstate == Pool::MIRROR_ON) n++, whole += p.hi - p.lo, held += p.resident_bytes;
            logf("vk: sparse mirrors: %zu, %llu MB with memory of %llu MB (%llu MB spared); blocks given memory later %lld, given back "
                 "%lld; bindings served from guest memory meanwhile %lld (%lld reaching blocks with memory too), from the mirror past every game pool %lld; "
                 "%lld binds (%.1f ms); %lld made whole for want of memory",
                 n, (unsigned long long)(held >> 20), (unsigned long long)(whole >> 20),
                 (unsigned long long)((whole - std::min(whole, held)) >> 20), (long long)st_sparse_grown, (long long)st_sparse_released,
                 (long long)st_sparse_in_place, (long long)st_sparse_mixed, (long long)st_sparse_past, (long long)st_sparse_binds, st_sparse_bind_ms, (long long)st_sparse_refused);
            if (sparse_spare_cap())
                logf("vk: sparse mirrors' buffers: %lld taken from the spares, %lld made new; %lld kept as spares, %lld destroyed; %zu "
                     "spares now, %llu MB of address space (RX_VK_SPARSE_SPARE_MB=%llu)",
                     (long long)st_spare_taken, (long long)st_spare_made, (long long)st_spare_kept, (long long)st_spare_destroyed,
                     sparse_spares.items.size(), (unsigned long long)(sparse_spares.held >> 20), (unsigned long long)(sparse_spare_cap() >> 20));
        }
    }
    if (mirror_wanted.empty()) {
        trims_settle();   // (a part's mirror made at its binding since)
        return;
    }
    // (hookshot: the hottest first. Unsorted, the small pools mirrors_by_heat had just given back were made again first, in the room
    // it made for the hot one, and a 132 MB pool bound 7 times a submission stayed in system memory beside 353 small mirrors)
    if (mirror_heat_on())
        std::stable_sort(mirror_wanted.begin(), mirror_wanted.end(), [](const Pool* a, const Pool* b) {
            return __atomic_load_n(&a->heat, __ATOMIC_RELAXED) > __atomic_load_n(&b->heat, __ATOMIC_RELAXED);
        });
    for (Pool* p : mirror_wanted) mirror_make(*p);
    trims_settle();   // (the old mirror waits for no more only once every part has its own)
    mirror_wanted.clear();
    publish_cap_ranges();
    static uint64_t logged_at;
    if (mirror_bytes >> 28 != logged_at) {
        logged_at = mirror_bytes >> 28;
        logf("vk: %lld pools mirrored in VRAM (%llu MB)", (long long)st_mirrors, (unsigned long long)(mirror_bytes >> 20));
    }
}

/// (hookshot) Before a command buffer's parallel segments: every mirrored pool synced once, into the replay thread's
/// recording, ahead of all of them. Workers record out of order, so a pool a later segment bound first was synced in
/// that segment's upload, which runs after the earlier segments' work, and put the CPU's pages back over what the GPU
/// had just written in them (one frame of wrong lighting or exposure: 5x the flashes of serial replay). A segment
/// finds its pools synced now (synced_epoch, sync_seg 0).
void memory_sync_mirrors_before_segments() {
    if (!mirror_on() || !mirror_upload) return;
    std::lock_guard lk(pools_mu);
    msync_batching = true;   // (RX_VK_MSYNC_PASS: every pool's chunks in one dispatch, after the loop)
    for (auto& kv : imported) {
        Pool& p = kv.second;
        if (p.mstate != Pool::MIRROR_ON || p.synced_epoch == mirror_epoch) continue;
        mirror_sync(p);
        p.sync_seg = 0;
    }
    msync_batching = false;
    msync_pass_flush();
}

/// A pool's mirror made, filled in the recording's upload command buffer: all of it, then the pages a capture holds
/// as they were at their submit (pools_mu held). False over the budget (tried again when next bound, as mirrors are
/// given back) or over 1 GB (never)
bool mirror_make(Pool& pool) {
#ifdef TRACY_ENABLE
    HT_SCOPE("mirror_make");
#endif
    Pool* p = &pool;
    if (mirror_epoch < p->retry_epoch) {   // (hookshot: its VRAM refused not long ago; wanted again at a later binding)
        if (p->mstate == Pool::MIRROR_WANTED) p->mstate = Pool::MIRROR_NONE;
        return false;
    }
    {
        const uint64_t n = p->hi - p->lo;
        // (hookshot: RX_VK_MIRROR_TIGHT=1, none while VRAM is 90% full. Off: 8 GB cards sit there all the time, and without their mirrors
        // the GPU read game memory over PCIe)
        static const bool tight_rule = env_on("RX_VK_MIRROR_TIGHT");
        uint64_t used = 0, allowed = 0;
        const bool tight = tight_rule && vram_usage(&used, &allowed) && allowed && used > allowed / 100 * 90;
        // (hookshot, RX_VK_MIRROR_SPARSE) memory only for the blocks a game pool is in: what it takes is theirs
        VkBuffer sparse_buf = VK_NULL_HANDLE;
        VkMemoryRequirements sparse_req{};
        std::vector<uint8_t> live;
        uint64_t take = n;
        uint64_t spare_after = 0;   // (a spare's value on the binds' timeline: goes back with it if it is given up, sparse_spare.h)
        if (mirror_sparse_on() && n <= (1ull << 30) && sparse_buffer_for(n, kUsage, sparse_buf, sparse_req, spare_after)) {
            live = sparse_req.alignment ? sparse_mirror::live_blocks(p->lo, p->hi, sparse_req.alignment, guest_pools.begin(),
                                                                     guest_pools.end())
                                        : std::vector<uint8_t>{};
            if (live.empty()) {
                sparse_buffer_unused(sparse_buf, n, sparse_req, spare_after);
                sparse_buf = VK_NULL_HANDLE;
            } else {
                take = (uint64_t)std::count(live.begin(), live.end(), (uint8_t)1) * sparse_req.alignment;
            }
        }
        // (hookshot) the last RX_VK_MIRROR_SMALL_PCT (70) % of the budget only for a pool the GPU binds much (its heat, at least
        // RX_VK_MIRROR_HOT_MB (64) MB of bound bytes, decayed: mirror_hot_bytes). Holding only the pools under 32 MB to 70 %, pools of
        // 44 to 68 MB took the rest before the 132 MB one
        const bool small_full = mirror_heat_on() && __atomic_load_n(&p->heat, __ATOMIC_RELAXED) < mirror_hot_bytes() &&
                                mirror_bytes + take > mirror_small_cap();
        if (mirror_bytes + take > mirror_budget() || n > (1ull << 30) || tight || small_full) {
            if (sparse_buf) sparse_buffer_unused(sparse_buf, n, sparse_req, spare_after);
            p->mstate = n > (1ull << 30) ? Pool::MIRROR_REFUSED : Pool::MIRROR_NONE;
            if (n > (1ull << 30)) p->mirror_retry_epoch = 0;
            static int logged;
            if (logged++ < 8)
                logf("vk: pool %#llx (%llu MB, its mirror %llu MB) stays in system memory: the mirrors have %llu MB of %llu",
                     (unsigned long long)p->lo, (unsigned long long)(n >> 20), (unsigned long long)(take >> 20),
                     (unsigned long long)(mirror_bytes >> 20),
                     (unsigned long long)(mirror_budget() >> 20));
            return false;
        }
        if (sparse_buf && !sparse_fill(*p, sparse_buf, sparse_req, live)) {
            sparse_buffer_doom(sparse_buf, spare_after);   // (a whole one instead, as before, if the budget has room for it)
            sparse_buf = VK_NULL_HANDLE;
            st_sparse_refused++;
            if (mirror_bytes + n > mirror_budget()) {
                p->mstate = Pool::MIRROR_NONE;
                return false;
            }
        }
        if (sparse_buf) p->sparse_made = n;   // (what the next mirror of this size may take its buffer by, mirror_free)
        if (!sparse_buf && !create_device_buffer(n, kUsage, p->mirror, p->mirror_alloc)) {
            // (hookshot: where the mirrors go by heat, again after 256, 512, ... submissions, eight times at most; elsewhere
            // refused, and tried again after 120 submissions when the budget has room for it)
            const bool again = mirror_heat_on() && p->refusals < 8;
            p->mstate = again ? Pool::MIRROR_NONE : Pool::MIRROR_REFUSED;
            if (again) p->retry_epoch = mirror_epoch + (256ull << p->refusals), p->refusals++;
            else p->mirror_retry_epoch = mirror_epoch + 120;
            static int logged_refused;
            if (logged_refused++ < 16)
                logf("vk: pool %#llx (%llu MB): its mirror's VRAM refused by the driver (%s)", (unsigned long long)p->lo,
                     (unsigned long long)(n >> 20), again ? "tried again later" : "tried again after 120 submissions");
            return false;
        }
        p->mirror_retry_epoch = 0;
        {   // (hookshot) the page table points at the mirror: what the GPU wrote is there, and guest memory has it only after the write-back
            const uint64_t ma = buffer_address(p->mirror);
            if (p->sparse)
                sparse_table(*p, ma, 0, p->has.size());   // (and at guest memory where a block has no memory)
            else
                guest_pages_set(p->lo, p->hi, ma);
            if (env_on("RX_VK_LOG_ADDR"))
                logf("vk: addr mirror %#llx..%#llx at gpu %#llx..%#llx%s", (unsigned long long)p->lo, (unsigned long long)p->hi,
                     (unsigned long long)ma, (unsigned long long)(ma + n), p->sparse ? " (sparse)" : "");
        }
        {
            GprofSpan gp(upload_command_buffer(), "mmake copy (a whole pool -> its new mirror)");
            if (p->sparse) {   // (the blocks with memory)
                std::vector<VkBufferCopy> parts;
                const uint64_t b = sparse_block(*p);
                sparse_mirror::runs(p->has.size(), [&](size_t i) { return p->has[i]; }, true, [&](size_t first, size_t count) {
                    const uint64_t at = (uint64_t)first * b, end = std::min(n, (uint64_t)(first + count) * b);
                    if (end > at) parts.push_back({at, at, end - at}), gp.bytes += end - at;
                });
                note_in_place_read();
                if (!parts.empty())
                    vkCmdCopyBuffer(upload_command_buffer(), p->buffer, p->mirror, (uint32_t)parts.size(), parts.data());
            } else {
                const VkBufferCopy all{0, 0, n};
                gp.bytes = n;
                note_in_place_read();
                vkCmdCopyBuffer(upload_command_buffer(), p->buffer, p->mirror, 1, &all);   // (all of it, at this generation)
            }
        }
        {   // (EvilOOT) then the mirrors of the pools it took in, over their part: what the GPU wrote there and only there
            // (retire_pool). The captures' pages go over it after, as before: the CPU's newer writes win
            bool barrier_done = false;
            int inherited = 0;
            for (auto& nh : retired_pools) {
                Pool& r = nh.mapped();
                if (!r.inherit_pending || r.mstate != Pool::MIRROR_ON || !r.mirror || r.hi <= p->lo || r.lo >= p->hi) continue;
                if (!barrier_done) {   // (after the full copy, and after the shaders of earlier submissions wrote the old mirror)
                    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
                    mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
                    vkCmdPipelineBarrier2(upload_command_buffer(), &di);
                    barrier_done = true;
                }
                const uint64_t a = std::max(r.lo, p->lo), b = std::min(r.hi, p->hi);
                // (only the pages not stamped since its last sync: one the CPU wrote after it has the game's newer data in the
                // copy of guest memory above, and the old mirror's over it counted as synced, stale for good (mirror_inherit.h).
                // As mirror_sync, a page a shader wrote through a bare entry keeps the mirror's copy (RX_VK_PROTECT_BARE=1).
                // Synced in this submission: the whole part. Its synced_gen is raised only at the submission's end
                // (memory_end_submission), so what the GPU wrote in this submission's earlier segments (RX_VK_PARALLEL) is
                // stamped above it, and guest memory has that only after the write-back; the CPU's writes were collected
                // before the sync)
                std::vector<VkBufferCopy> parts;
                uint64_t from_guest = 0;
                const uint32_t synced = r.synced_gen;
                if (r.synced_epoch == mirror_epoch)
                    parts.push_back({a - r.lo, a - p->lo, b - a});
                else
                    from_guest = mirror_inherit::runs(a, b, r.lo, p->lo, kPageShift, kRegionPages,
                                                      [&](uint64_t region) { return region_gen[region] <= synced; },
                                                      [&](uint64_t page) {
                                                          return page_gen(page) <= synced ||
                                                                 (protect_bare_on() && prot_page_since(page, synced));
                                                      },
                                                      [&](uint64_t src, uint64_t dst, uint64_t size) { parts.push_back({src, dst, size}); });
                if (!parts.empty() && (p->sparse || r.sparse)) {
                    // (sparse: only between blocks with memory on both sides. One with none in the retired mirror read 0, over
                    // what guest memory gave the new one: its truth is guest memory, where its page table pointed. One with none
                    // in the new mirror was lost anyway)
                    std::vector<sparse_mirror::Copy> in, out;
                    for (const VkBufferCopy& c : parts) in.push_back({c.srcOffset, c.dstOffset, c.size});
                    const uint64_t b = p->sparse ? sparse_block(*p) : sparse_block(r);
                    out = sparse_mirror::clip(
                        in, b,
                        [&](uint64_t s) { return !r.sparse || (s / sparse_block(r) < r.has.size() && block_has(r, (size_t)(s / sparse_block(r)))); },
                        [&](uint64_t d) { return !p->sparse || (d / sparse_block(*p) < p->has.size() && block_has(*p, (size_t)(d / sparse_block(*p)))); });
                    if (!r.refill.empty()) {   // (a binding wrote there in guest memory and the refill never came: guest memory is newer)
                        std::vector<std::pair<uint64_t, uint64_t>> pending;
                        for (const auto& [o, l] : r.refill) pending.push_back({o, o + l});
                        out = sparse_mirror::cut_src(out, pending);
                    }
                    parts.clear();
                    for (const sparse_mirror::Copy& c : out) parts.push_back({c.src, c.dst, c.size});
                }
                if (!parts.empty())
                    vkCmdCopyBuffer(upload_command_buffer(), r.mirror, p->mirror, (uint32_t)parts.size(), parts.data());
                if (from_guest) {
                    static int told;
                    if (told++ < 40)
                        logf("vk: pool %#llx..%#llx: %llu KB the CPU wrote since retired pool %#llx's last sync kept from guest memory, "
                             "not its mirror", (unsigned long long)p->lo, (unsigned long long)p->hi, (unsigned long long)(from_guest >> 10),
                             (unsigned long long)r.lo);
                }
                if (!r.trimmed) r.inherit_pending = false;   // (a trimmed one: each of its live parts' pools takes its own, mirrors_trim)
                r.retired_value = std::max(r.retired_value, mem_rec_value);   // (in use by this upload until it is done)
                inherited++;
            }
            if (inherited) {
                static int logged;
                if (logged++ < 16)
                    logf("vk: pool %#llx..%#llx: mirror made with the GPU's contents of %d retired pool(s)", (unsigned long long)p->lo,
                         (unsigned long long)p->hi, inherited);
            }
        }
        {   // (then the pages a capture holds as they were at their submit: guest memory is the game's frame ahead)
            std::map<VkBuffer, std::vector<VkBufferCopy>> over;
            cap_pages_over(*p, p->lo, p->hi, over);
            if (!over.empty()) {
                late_fix_emit();   // (the late pages' fix, before the copies out of a capture's buffer below)
                VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};   // (after the full copy)
                mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.memoryBarrierCount = 1;
                di.pMemoryBarriers = &mb;
                vkCmdPipelineBarrier2(upload_command_buffer(), &di);
                for (const auto& [buf, rs] : over)
                    vkCmdCopyBuffer(upload_command_buffer(), buf, p->mirror, (uint32_t)rs.size(), rs.data());
            }
        }
        p->mstate = Pool::MIRROR_ON;
        pools_gen.fetch_add(1, std::memory_order_relaxed);   // (the binding's buffer changes: guest memory, then this mirror)
        p->synced_gen = gen_now;
        p->synced_epoch = mirror_epoch;
        p->sync_seg = texture_segment();
        mirror_synced.push_back(p);
        mirror_bytes += mirror_size(*p);
        st_mirrors++;
        st_sync_bytes += (int64_t)mirror_size(*p);
    }
    return true;
}

namespace {
/// The pages of guest [lo, hi) (inside p) a capture holds as they were at their submit, as copies into p's mirror by the
/// capture's buffer; a sparse mirror's blocks with no memory left out
void cap_pages_over(Pool& p, uint64_t lo, uint64_t hi, std::map<VkBuffer, std::vector<VkBufferCopy>>& over) {
    for (uint64_t page = lo >> kPageShift; page < (hi + 4095) >> kPageShift; page++) {
        const CapEntry e = cap_entry(page);
        if (!e.gen || e.gen != page_gen(page)) continue;
        auto b = cap_bufs.find(e.gen);
        if (b == cap_bufs.end()) continue;
        const uint64_t ga = std::max(page << kPageShift, lo), gb = std::min((page + 1) << kPageShift, hi);
        if (gb <= ga) continue;
        if (p.sparse) {
            const uint64_t blk = (ga - p.lo) / sparse_block(p);
            if (blk >= p.has.size() || !block_has(p, (size_t)blk)) continue;
        }
        over[b->second.buffer].push_back({(uint64_t)e.slot * 4096 + (ga - (page << kPageShift)), ga - p.lo, gb - ga});
        b->second.last_use = std::max(b->second.last_use, mem_rec_value);
    }
}

/// The blocks wanted in these sparse mirrors get memory, filled as mirror_make fills a mirror: guest memory, then the
/// captured pages over, and the page table points at them; and the ranges a binding wrote in guest memory where the blocks
/// had memory (refill) are copied in again. At a submission's start (pools_mu held): the recordings before it are done
/// with guest memory by the fill (full_barrier), and none of this one's work has run. Every mirror's binds in one call: a
/// sparse bind waits for the queue (~5 ms). A block given back whose memory is not freed yet takes it back, still bound
void sparse_grow_all(const std::vector<Pool*>& pools) {
    struct Got { Pool* p; std::vector<size_t> blocks; std::vector<VkSparseMemoryBind> binds; };
    std::vector<Got> gots;
    uint64_t taking = 0;
    bool short_of_memory = false;
    for (Pool* pp : pools) {
        Pool& p = *pp;
        p.any_wanted = false;
        Got g{&p, {}, {}};
        const uint64_t b = sparse_block(p);
        for (size_t i = 0; i < p.wanted.size(); i++) {
            if (!p.wanted[i]) continue;
            p.wanted[i] = 0;
            if (p.blocks[i]) continue;
            if (p.released[i]) {   // (given back since, its memory not freed yet: still bound, taken back as it is, with no bind)
                p.blocks[i] = p.released[i];
                p.released[i] = nullptr;
                ++p.release_gen[i];   // (its dead_blocks entry no longer acts)
                g.blocks.push_back(i);
                taking += b;
                continue;
            }
            VkDeviceMemory m = VK_NULL_HANDLE;
            uint64_t off = 0;
            if (short_of_memory || mirror_bytes + taking + b > mirror_budget() ||
                !alloc_sparse_block(sparse_req_of(p), p.blocks[i], m, off)) {
                short_of_memory = true;   // (the rest stay in guest memory, read there, and are wanted again when bound)
                p.blocks[i] = nullptr;
                continue;
            }
            const uint64_t at = (uint64_t)i * b;
            g.binds.push_back({at, std::min(b, p.sparse_size - at), m, off, 0});
            g.blocks.push_back(i);
            taking += b;
        }
        if (!g.blocks.empty() || !p.refill.empty()) gots.push_back(std::move(g));
    }
    if (short_of_memory) {
        static int logged;
        if (logged++ < 8)
            logf("vk: sparse mirrors: no memory for more blocks (the mirrors have %llu MB of %llu)", (unsigned long long)(mirror_bytes >> 20),
                 (unsigned long long)(mirror_budget() >> 20));
    }
    if (gots.empty()) return;
    {   // (every mirror's new blocks bound at once)
        std::vector<VkSparseBufferMemoryBindInfo> infos;
        for (Got& g : gots)
            if (!g.binds.empty()) infos.push_back({g.p->mirror, (uint32_t)g.binds.size(), g.binds.data()});
        if (!infos.empty() && !sparse_bind_infos(std::move(infos))) {   // (none bound: their new memory back, those blocks left in guest memory)
            for (Got& g : gots)
                for (const VkSparseMemoryBind& bd : g.binds) {
                    const size_t i = (size_t)(bd.resourceOffset / sparse_block(*g.p));
                    free_sparse_block(g.p->blocks[i]);
                    g.p->blocks[i] = nullptr;
                    std::erase(g.blocks, i);
                }
        }
    }
    VkCommandBuffer up = upload_command_buffer();
    full_barrier(up, "barrier (sparse fill: after the work before it wrote guest memory)");
    bool filled = false;
    for (Got& g : gots) {
        Pool& p = *g.p;
        const uint64_t b = sparse_block(p), n = p.hi - p.lo;
        std::sort(g.blocks.begin(), g.blocks.end());
        for (size_t i : g.blocks) block_has_set(p, i, true);
        std::vector<std::pair<uint64_t, uint64_t>> ranges;   // (offsets: the new blocks whole, and the refills)
        for (size_t i : g.blocks) ranges.push_back({(uint64_t)i * b, (uint64_t)i * b + b});
        for (const auto& [off, len] : p.refill) ranges.push_back({off, off + len});
        p.refill.clear();
        std::sort(ranges.begin(), ranges.end());
        std::vector<VkBufferCopy> fill;
        for (const auto& [a, e] : ranges) {   // (only where the mirror has memory: a refill may run into a block with none)
            for (uint64_t at = a; at < std::min(e, n);) {
                const uint64_t blk = at / b, end = std::min({e, n, (blk + 1) * b});
                if (blk < p.has.size() && block_has(p, (size_t)blk)) {
                    if (!fill.empty() && fill.back().srcOffset + fill.back().size >= at)
                        fill.back().size = std::max(fill.back().srcOffset + fill.back().size, end) - fill.back().srcOffset;
                    else
                        fill.push_back({at, at, end - at});
                }
                at = end;
            }
        }
        if (!fill.empty()) {
            filled = true;
            {
                GprofSpan gp(up, "sparse fill (guest memory -> a mirror's new blocks)");
                for (const VkBufferCopy& c : fill) gp.bytes += c.size;
                note_in_place_read();
                vkCmdCopyBuffer(up, p.buffer, p.mirror, (uint32_t)fill.size(), fill.data());
            }
            // (guest memory had those pages while the GPU's pointers wrote there: the write-back diff's shadow knew none of that)
            for (const VkBufferCopy& c : fill) shadow_forget(p, c.dstOffset, c.size);
            std::map<VkBuffer, std::vector<VkBufferCopy>> over;
            for (const VkBufferCopy& c : fill) cap_pages_over(p, p.lo + c.dstOffset, p.lo + c.dstOffset + c.size, over);
            if (!over.empty()) {
                late_fix_emit();   // (the late pages' fix, before the copies out of a capture's buffer below: as mirror_make)
                VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};   // (after the fill)
                mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.memoryBarrierCount = 1;
                di.pMemoryBarriers = &mb;
                vkCmdPipelineBarrier2(up, &di);
                for (const auto& [buf, rs] : over) vkCmdCopyBuffer(up, buf, p.mirror, (uint32_t)rs.size(), rs.data());
            }
        }
        const VkDeviceAddress ma = buffer_address(p.mirror);
        for (size_t i : g.blocks) sparse_table(p, ma, i, i + 1);
        p.resident_bytes += g.blocks.size() * b;
        mirror_bytes += g.blocks.size() * b;
        st_sparse_grown += (int64_t)g.blocks.size();
    }
    // (RX_VK_MSYNC_PASS: a mirror sync after this in the uploads reads and writes these bytes in the pass's dispatch, which
    // the copies' barrier above does not order)
    if (filled && msync_pass_on()) full_barrier(up, "barrier (sparse fill: before the mirror syncs' pass)");
    pools_gen.fetch_add(1, std::memory_order_relaxed);
}

/// Every live sparse mirror's blocks that one of the game's pools reaches and that have no memory: wanted, so that they get
/// it at this submission's start, before any work reaches them (the game made a pool there since; pools_mu held)
void sparse_want_live() {
    for (auto& [lo, p] : imported) {
        if (!p.sparse || p.mstate != Pool::MIRROR_ON || !p.mirror) continue;
        const std::vector<uint8_t> live = sparse_mirror::live_blocks(p.lo, p.hi, sparse_block(p), guest_pools.begin(), guest_pools.end());
        bool any = false;
        for (size_t i = 0; i < live.size() && i < p.blocks.size(); i++)
            if (live[i] && !p.blocks[i]) p.wanted[i] = 1, any = true;
        if (any && !p.any_wanted) p.any_wanted = true, sparse_wanted.push_back(&p);
    }
}

/// Blocks of live sparse mirrors that every game pool has left, and none has come back to for kSparseIdleEpochs submissions:
/// back to guest memory in the page table now, their memory given back once the GPU is past what may have reached them
/// (sparse_reap). The game frees short-lived pools and makes them again at the same addresses: given back the first time
/// none was seen there, ~28,000 blocks a Castle Town to field run were wanted again at once, ~900 binds and 1.8 s of the
/// replay's time in them. pools_mu held
constexpr uint64_t kSparseIdleEpochs = 600;
void sparse_release() {
    for (auto& [lo, p] : imported) {
        if (!p.sparse || p.mstate != Pool::MIRROR_ON || !p.mirror) continue;
        const uint64_t b = sparse_block(p);
        const std::vector<uint8_t> live = sparse_mirror::live_blocks(p.lo, p.hi, b, guest_pools.begin(), guest_pools.end());
        size_t released = 0;
        for (size_t i = 0; i < p.blocks.size(); i++) {
            if (!p.blocks[i] || (i < live.size() && live[i]) || p.wanted[i] || mirror_epoch - last_bound_of(p, i) < kSparseIdleEpochs) {
                p.block_dead_since[i] = 0;
                continue;
            }
            if (!p.block_dead_since[i]) p.block_dead_since[i] = mirror_epoch;
            if (mirror_epoch - p.block_dead_since[i] < kSparseIdleEpochs) continue;
            p.block_dead_since[i] = 0;
            block_has_set(p, i, false);
            const uint64_t a = p.lo + (uint64_t)i * b, e = std::min(p.hi, a + b);
            if (e > a) guest_pages_set(a, e, p.address + (a - p.lo));
            p.released[i] = p.blocks[i];
            dead_blocks.push_back({p.mirror, &p, i, p.blocks[i], std::max(mem_rec_value, p.last_value), ++p.release_gen[i]});
            p.blocks[i] = nullptr;
            p.resident_bytes -= std::min(p.resident_bytes, b);
            mirror_bytes -= std::min(mirror_bytes, b);
            released++;
        }
        if (released) {
            st_sparse_released += (int64_t)released;
            pools_gen.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

/// Released blocks the GPU is past: unbound from their mirror (if it is still alive and the block got no new memory since)
/// and their memory freed. pools_mu held
void sparse_reap() {
    std::map<VkBuffer, std::vector<VkSparseMemoryBind>> unbind;
    std::vector<VmaAllocation> frees;
    while (!dead_blocks.empty() && dead_blocks.front().value <= mem_completed) {
        const DeadBlock d = dead_blocks.front();
        dead_blocks.pop_front();
        // (no pool: its mirror was freed and took the memory along; taken back or given back again since: not this entry's)
        if (!d.pool || d.index >= d.pool->released.size() || d.pool->released[d.index] != d.alloc || d.pool->release_gen[d.index] != d.gen)
            continue;
        d.pool->released[d.index] = nullptr;
        if (sparse_live.count(d.buffer) && d.pool->mirror == d.buffer) {
            const uint64_t b = sparse_block(*d.pool), at = (uint64_t)d.index * b;
            unbind[d.buffer].push_back({at, std::min(b, d.pool->sparse_size - at), VK_NULL_HANDLE, 0, 0});
        }
        frees.push_back(d.alloc);
    }
    uint64_t unbound = 0;
    if (!unbind.empty()) {   // (every mirror's in one call)
        std::vector<VkSparseBufferMemoryBindInfo> infos;
        for (auto& [buf, binds] : unbind) infos.push_back({buf, (uint32_t)binds.size(), binds.data()});
        sparse_bind_infos(std::move(infos), &unbound);
    }
    // (the memory goes once the unbind is done on the GPU; the latest bind's value covers memory a destroyed buffer let go too)
    if (!frees.empty()) sparse_frees.push_back({unbound ? unbound : sparse_tl_value.load(std::memory_order_relaxed), std::move(frees)});
    const uint64_t done = sparse_tl_done();
    while (!sparse_frees.empty() && sparse_frees.front().first <= done) {
        for (VmaAllocation a : sparse_frees.front().second) free_sparse_block(a);
        sparse_frees.pop_front();
    }
    for (auto it = spare_doomed.begin(); it != spare_doomed.end();) {   // (spares given up, once nothing is bound to them on the GPU)
        if (it->first > done) {
            ++it;
            continue;
        }
        destroy_sparse_buffer(it->second);
        st_spare_destroyed++;
        it = spare_doomed.erase(it);
    }
    // (freed mirrors, as reap_mirrors, once their binds are done. sparse_spare.h: the buffer is kept for the next mirror of
    // its size: what is bound to it is unbound, every kept buffer's in one bind, and its blocks' memory goes once that is done
    // on the GPU, as a released block's above. With no spares, or when the bind fails: destroyed and its memory freed at
    // once, as before)
    std::vector<DeadSparse> keep;
    while (!dead_sparse.empty() && dead_sparse.front().value <= mem_completed && dead_sparse.front().binds <= done) {
        DeadSparse d = std::move(dead_sparse.front());
        dead_sparse.pop_front();
        sparse_live.erase(d.buffer);
        if (d.spare.made && sparse_spare_cap()) {
            keep.push_back(std::move(d));
            continue;
        }
        destroy_sparse_buffer(d.buffer);
        st_spare_destroyed++;
        for (VmaAllocation a : d.blocks) free_sparse_block(a);
    }
    if (keep.empty()) return;
    std::vector<VkSparseBufferMemoryBindInfo> infos;
    for (DeadSparse& d : keep)
        if (!d.unbind.empty()) infos.push_back({d.buffer, (uint32_t)d.unbind.size(), d.unbind.data()});
    uint64_t after = 0;
    if (!sparse_bind_infos(std::move(infos), &after)) {   // (still bound: nothing to keep them as)
        for (DeadSparse& d : keep) {
            destroy_sparse_buffer(d.buffer);
            st_spare_destroyed++;
            for (VmaAllocation a : d.blocks) free_sparse_block(a);
        }
        return;
    }
    if (!after) after = sparse_tl_value.load(std::memory_order_relaxed);   // (nothing was bound to any of them: the latest bind's)
    std::vector<VmaAllocation> blocks;
    std::vector<sparse_spare::Spare<VkBuffer>> gone;
    for (DeadSparse& d : keep) {
        blocks.insert(blocks.end(), d.blocks.begin(), d.blocks.end());
        d.spare.after = after;
        if (sparse_spares.put(d.spare, sparse_spare_cap(), &gone))
            st_spare_kept++;
        else
            gone.push_back(d.spare);   // (larger than the cap by itself)
    }
    for (const auto& g : gone) sparse_buffer_doom(g.buffer, g.after);
    if (!blocks.empty()) sparse_frees.push_back({after, std::move(blocks)});
}
} // namespace

// ---- the write-back's diff (hookshot, 2026-11-09) ----
// A mirrored pool's write-back copied every written run whole from the mirror to guest memory: ~69 MB a frame in Hyrule Field
// (an i7-14700K and an RTX 5070 Ti), 4.4 ms of the GPU's ~13, at the end of the frame's last submission, which the pages of the
// next capture wait for (the capture's write-back wait above). The runs are the written bindings' whole ranges: ~82% of what
// they copied was zeros, and of the rest little changed from one frame to the next. Now a pool written back keeps a shadow in
// VRAM, what guest memory holds there as far as it is known, page by page. A compute pass (shaders/wb_diff.comp) copies the
// words of a run that differ from the shadow, into guest memory and the shadow both; a page the CPU wrote since (the captures'
// pages and the replay's own words, shadow_written) is read from guest memory into the shadow first. Guest memory comes out as
// with the whole copy, or, for words the GPU left as they were and the CPU wrote since, with the CPU's words, as on the console:
// the pass only ever writes fewer words than the copy did. The pass takes ~0.7 ms there, the GPU's frame went from ~13.1 to
// ~9.2 ms, and running there the game went from ~53.9 to ~57.9 fps with 4x frame interpolation (from ~55.2 to ~58.7 without).
// (Word by word it was far slower than the copies: the GPU's writes to system memory went at ~1 GB/s; a subgroup's whole span
// written at once goes at the copies' speed or better.)
// RX_VK_WB_DIFF=0: the whole copies, as before. RX_VK_WB_SHADOW_MB: the shadows' VRAM at most (default 1/8 of it)
// What it rests on: a shadow page it knows equals guest memory. So every CPU write of guest memory reaches shadow_written before
// the next write-back reads the list: the captures' pages (capture_make), every page the collection finds (memory_collect_written:
// all of the game's writes with RX_VK_CAPTURE=0), and the replay's own writes, through guest_cpu_write / guest_cpu_wrote: the
// in-stream clears of the counters the game's conditions read, the copies, clears and updates done on the CPU, the textures
// written back into their memory, WriteTexels, the report words (memory_cpu_owned). One the shadows never heard of left the
// CPU's value in guest memory where the whole copy wrote the GPU's, whenever the GPU's result equalled the shadow's old word:
// a counter cleared in the stream and counted to last frame's value again stayed cleared, and what it switched stayed off
// (PR 1109's review). The 600-submit line counts the pages each kind of report made the shadows read again.
// The GPU's own writes of guest memory past the mirror do the same: a written binding given the pool's guest buffer
// (gpu_buffer_bind outside a recording's uploads), and a whole copy in place of the pass, forget the shadow's pages (shadow_forget).
std::mutex shadow_written_mu;
std::vector<uint32_t> shadow_written_pages;   // (guest pages the CPU wrote: unknown to the shadows at the next write-back)
std::vector<uint32_t> shadow_written_collected_list;   // (the same, of the collection: memory_collect_written)
std::vector<uint32_t> shadow_written_by_replay;   // (the same, of the replay's own CPU writes: guest_cpu_wrote)
bool shadow_written_all = false;              // (more than those lists hold: every page unknown at the next write-back)
int64_t st_wbd_known_game = 0, st_wbd_known_collected = 0, st_wbd_known_replay = 0;   // (pages the shadows knew when reported)
struct DeadShadow {
    VkBuffer buffer;
    VmaAllocation alloc;
    uint64_t value;
};
std::deque<DeadShadow> dead_shadows;
uint64_t shadow_bytes = 0;
int64_t st_wbd_runs = 0, st_wbd_bytes = 0, st_wbd_fill_pages = 0;
bool wb_diff_on() {
    // (hookshot) On by default again (v2.0.11). v2.0.9 turned it off with the presents' own queue after the device lost on an RTX 4090
    // with NVIDIA's 621 driver (issue 1161); the reporter's A/B then put that on the present queue (black screen with it alone), and the
    // write-back diff alone ran there and on drivers 615 and 621 for others: ~20% fps (an RTX 3080 laptop went 40 -> 25 fps without
    // it). RX_VK_WB_DIFF=0: off
    // (integrated GPUs too: an Ally X's hard reboots stopped with v2.0.8 and came back with v2.0.9's off, issue 997: the whole copies
    // are long submissions there. Its shadows are the system's RAM on them, so at most 512 MB: shadow_room)
    static const bool on = !getenv("RX_VK_WB_DIFF") || atoi(getenv("RX_VK_WB_DIFF")) != 0;
    return on;
}
void shadow_free(Pool& p) {   // (pools_mu held: destroyed once the submissions so far are done)
    if (!p.shadow) return;
    dead_shadows.push_back({p.shadow, p.shadow_alloc, std::max(mem_rec_value, p.last_value)});
    shadow_bytes -= p.hi - p.lo;
    p.shadow = VK_NULL_HANDLE, p.shadow_alloc = nullptr, p.shadow_address = 0;
    p.shadow_known.clear();
}
void shadows_reap() {
    while (!dead_shadows.empty() && dead_shadows.front().value <= mem_completed) {
        destroy_buffer(dead_shadows.front().buffer, dead_shadows.front().alloc);
        dead_shadows.pop_front();
    }
}
// (hookshot) The shadows' two rules read what is really allocated (the driver's count less the free space inside VMA's blocks,
// as the texture budget's 90% rule does: vram_pressure.h), not the count alone. By the count alone they went at 95% and
// came back only under 85%. On a 12 GB card the texture cache fills VRAM to 93-98% of what the driver allows within minutes
// of play, gigabytes of it free space inside the blocks that evicting cannot give back, so the shadows went (272-448 MB) and
// never returned: the write-backs were whole copies again for the rest of the session. In a player's log (RTX 4070 SUPER,
// 2560 x 1440, 10,462 of 10,732 MB used, 3.5 GB of it free inside the blocks) "the shadows given back" is where 58 fps
// became 26. Here, on v2.0.11's code with 9.8 GB allowed, five minutes across Hyrule Field, two runs each: 56.6 and 58.8 fps
// with 32 and 8 of 120 two-second windows under 55 fps before, 59.8 and 59.6 fps with 1 and 2 after; the GPU's frame
// 11.7-12.0 -> 9.8-10.1 ms, the capture's wait in a frame 2.1-2.4 -> 1.1 ms. (The free space is of no use to the shadows
// themselves, which are allocations of their own: they are made only under the textures' lines. vram_pressure.h)
bool shadow_room(uint64_t n) {   // (the shadows' budget, and room in VRAM: the textures and the mirrors come first)
    static const uint64_t cap = [] {
        if (const char* e = getenv("RX_VK_WB_SHADOW_MB")) return (uint64_t)atoll(e) << 20;
        uint64_t used = 0, allowed = 0;
        const uint64_t eighth = vram_usage(&used, &allowed) && allowed ? allowed / 8 : (uint64_t)512 << 20;   // (unsigned long on Linux)
        // (an integrated GPU's VRAM is the system's RAM: 512 MB at most there)
        return dev.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? std::min<uint64_t>(eighth, (uint64_t)512 << 20) : eighth;
    }();
    if (shadow_bytes + n > cap) return false;
    uint64_t used = 0, allowed = 0, block_free = 0;
    if (!vram_usage(&used, &allowed, &block_free) || !allowed) return true;
    return vram_pressure::shadow_fits(n, used, allowed, block_free);
}
bool shadow_make(Pool& p) {
    const uint64_t n = p.hi - p.lo;
    if (!p.mirror || !shadow_room(n)) return false;
    if (!create_device_buffer(n, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, p.shadow, p.shadow_alloc))
        return false;
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = p.shadow;
    p.shadow_address = vkGetBufferDeviceAddress(dev.device, &da);
    p.shadow_known.assign((size_t)((n >> kPageShift) + 63) / 64, 0);
    shadow_bytes += n;
    static int logged;
    if (logged++ < 16)
        logf("vk: write-back diff: a shadow for pool %#llx (%llu MB; %llu MB of shadows)", (unsigned long long)p.lo,
             (unsigned long long)(n >> 20), (unsigned long long)(shadow_bytes >> 20));
    return true;
}
// (pools_mu held) every 120 write-back submissions: every shadow given back while VRAM is really 95% full, or the driver's count
// at 99%, and the shadows of the pools that had one made again once there is room. A pool's own write-backs may each stay under
// the 256 KB a shadow is first made at, for good after a scene change, so waiting for one never gave it its shadow back: in a
// player's log (issue 2238) 48 of 352 MB stayed whole copies for the rest of the session while the others were made again.
void shadows_check_room() {
    static int64_t n;
    if (++n % 120) return;
    bool wanted = false;
    for (const auto& [lo, p] : imported) wanted = wanted || (p.shadow_wanted && !p.shadow && p.mirror);
    if (!shadow_bytes && !wanted) return;
    uint64_t used = 0, allowed = 0, block_free = 0;
    if (!vram_usage(&used, &allowed, &block_free) || !allowed) return;
    if (!vram_pressure::shadows_go(used, allowed, block_free)) {
        int made = 0, tries = 0;
        uint64_t made_bytes = 0;   // (what this check made, not what all the shadows are)
        for (auto& [lo, p] : imported) {
            if (!p.mirror || p.shadow || !p.shadow_wanted) continue;   // (mirror_free'd: cannot be made again, not a try)
            const uint64_t n = p.hi - p.lo;
            if (!shadow_room(n)) continue;   // (too big to fit now, or past the budget: no attempt, a later check has room)
            if (tries++ == 8) break;         // (hard cap on attempts per check: each is an allocation of the pool's whole size)
            if (shadow_make(p)) made++, made_bytes += n;
        }
        static int logged;
        if (made && logged++ < 8)
            logf("vk: write-back diff: %d shadows made again (%llu MB)", made, (unsigned long long)(made_bytes >> 20));
        return;
    }
    if (!shadow_bytes) return;
    const uint64_t had = shadow_bytes;
    for (auto& [lo, p] : imported) shadow_free(p);
    for (auto& nh : retired_pools) shadow_free(nh.mapped());
    static int logged;
    if (logged++ < 8)
        logf("vk: write-back diff: VRAM %llu of %llu MB: the shadows given back (%llu MB)", (unsigned long long)(used >> 20),
             (unsigned long long)(allowed >> 20), (unsigned long long)(had >> 20));
}
// (pools_mu held) a range of the pool (offsets) unknown to its shadow again
void shadow_forget(Pool& p, uint64_t at, uint64_t len) {
    if (!p.shadow || !len) return;
    for (uint64_t pg = at >> kPageShift; pg <= (at + len - 1) >> kPageShift; pg++) p.shadow_known[pg >> 6] &= ~(1ull << (pg & 63));
}
// (pools_mu held) the pages written since: unknown to their pools' shadows
void shadows_forget_written() {
    static thread_local std::vector<uint32_t> game, collected, replay;
    bool all;
    {
        std::lock_guard lk(shadow_written_mu);
        game.clear(), collected.clear(), replay.clear();
        game.swap(shadow_written_pages);
        collected.swap(shadow_written_collected_list);
        replay.swap(shadow_written_by_replay);
        all = shadow_written_all, shadow_written_all = false;
    }
    if (all)   // (written for a long while with no write-back to read the list: nothing of guest memory is known)
        for (auto& [lo, pool] : imported)
            if (pool.shadow) std::fill(pool.shadow_known.begin(), pool.shadow_known.end(), 0ull);
    shadow_forgotten_last.clear(), shadow_forgotten_all = all;
    if (!shadow_bytes) return;
    // (in this order, each page counted under the first that reports it: the collection's and the replay's counts are what the
    // captures alone would have left known)
    for (int which = 0; which < 3; which++) {
        std::vector<uint32_t>& pages = which == 0 ? game : which == 1 ? collected : replay;
        int64_t& known = which == 0 ? st_wbd_known_game : which == 1 ? st_wbd_known_collected : st_wbd_known_replay;
        if (pages.empty()) continue;
        // (as the lists were built, not sorted: a page's bit is cleared and counted the same wherever it stands in its list. The
        // pool found for a page is kept for the pages after it, and a capture or a collection reports its pages in ascending
        // order, so the pools are looked up once a run of pages, not once a page. Sorting each list at every part's end for
        // that alone was 0.36 ms a frame of the render thread in Castle Town's market)
        Pool* p = nullptr;
        for (uint32_t pg : pages) {
            const uint64_t a = (uint64_t)pg << kPageShift;
            if (!p || a < p->lo || a >= p->hi) {
                p = nullptr;
                auto it = imported.upper_bound(a);
                if (it == imported.begin()) continue;
                --it;
                if (a < it->second.hi) p = &it->second;
                if (!p) continue;
            }
            if (!p->shadow) continue;
            const uint64_t i = (a - p->lo) >> kPageShift;
            uint64_t& bits = p->shadow_known[i >> 6];
            if (bits & (1ull << (i & 63))) known++;
            bits &= ~(1ull << (i & 63));
        }
    }
    // (the pages the capture did not take that were written since: the replay's own writes and the collection's, for the late
    // pages' summary line: a page synced from a capture and also in these has newer words in guest memory than the shadow)
    // (in the lists' order, not sorted: shadow_synced_known walks it once for the pages it asks about)
    shadow_forgotten_last.clear();
    shadow_forgotten_last.insert(shadow_forgotten_last.end(), collected.begin(), collected.end());
    shadow_forgotten_last.insert(shadow_forgotten_last.end(), replay.begin(), replay.end());
}
// (the pass's pipeline: 24 bytes of push constants and no descriptors, its buffers by address; made once)
struct WbdPipe {
    bool tried = false;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
} wbd_pipe;
bool wbd_pipe_ready() {
    if (wbd_pipe.tried) return wbd_pipe.pipe != VK_NULL_HANDLE;
    wbd_pipe.tried = true;
    // (the pass votes across a subgroup (shaders/wb_diff.comp): a GPU without votes in compute keeps the whole copies)
    VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(dev.physical, &p2);
    if (!(sg.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) || !(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        logf("vk: write-back diff off: the GPU has no subgroup votes in compute shaders");
        return false;
    }
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pr;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(dev.device, &li, nullptr, &wbd_pipe.layout) == VK_SUCCESS) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        const auto spv = BUILTIN_SPV(WB_DIFF_COMP_SPV);
        mi.codeSize = spv.size;
        mi.pCode = spv.code;
        IntelDriverGate gate;
        if (extern_guarded_shader_module(&mi, &m) == VK_SUCCESS) {
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = m;
            ci.stage.pName = "main";
            ci.layout = wbd_pipe.layout;
            if (extern_guarded_compute(VK_NULL_HANDLE, &ci, &wbd_pipe.pipe) != VK_SUCCESS)
                wbd_pipe.pipe = VK_NULL_HANDLE;
            vkDestroyShaderModule(dev.device, m, nullptr);
        }
    }
    const bool ok = wbd_pipe.pipe != VK_NULL_HANDLE;
    // (hookshot: the mirror syncs' pass makes it too, RX_VK_MSYNC_PASS, with the diff itself off or on)
    logf("vk: write-back diff %s", !ok           ? "off: its pipeline was not made"
                                   : wb_diff_on() ? "on (RX_VK_WB_DIFF=0: off)"
                                                  : "off (RX_VK_WB_DIFF=0); its pipeline made for the mirror syncs' pass");
    return ok;
}
void shadow_written_to(std::vector<uint32_t>& list, const uint32_t* pages, int64_t n) {
    // (no mirrors: the lists are read by a write-back's diff only, shadows_forget_written, and none is ever made. They grew
    // to 8 million pages, 32 MB, before being dropped, again and again: about 1 % of the replay thread on a Steam Deck)
    if (n <= 0 || !wb_diff_on() || !mirror_on()) return;
    std::lock_guard lk(shadow_written_mu);
    if (shadow_written_pages.size() + shadow_written_collected_list.size() + shadow_written_by_replay.size() > (8u << 20)) {
        shadow_written_all = true;   // (no write-back has read them for a long while: all of them, then)
        shadow_written_pages.clear(), shadow_written_collected_list.clear(), shadow_written_by_replay.clear();
        return;
    }
    list.insert(list.end(), pages, pages + n);
}
void shadow_written(const uint32_t* pages, int64_t n) {
    shadow_written_to(shadow_written_pages, pages, n);
}
void shadow_written_collected(const uint32_t* pages, int64_t n) {
    shadow_written_to(shadow_written_collected_list, pages, n);
}
int guest_cpu_write(uint64_t gpu, const void* src, uint64_t n) {
    const int ok = host->gpu_write(gpu, src, n);
    guest_cpu_wrote(gpu, n);
    return ok;
}
void guest_cpu_wrote(uint64_t gpu, uint64_t n) {
    if (!n || !wb_diff_on() || !mirror_on()) return;   // (as shadow_written_to: nor are the pages looked up)
    uint32_t pages[64];
    int k = 0;
    for (uint64_t a = gpu & ~4095ull; a < gpu + n; a += 4096) {   // (GPU pages are mapped to guest memory page by page)
        const uint64_t g = host->gpu_guest(a);
        if (!g) continue;
        pages[k++] = (uint32_t)(g >> kPageShift);
        if (k == 64) shadow_written_to(shadow_written_by_replay, pages, k), k = 0;
    }
    if (k) shadow_written_to(shadow_written_by_replay, pages, k);
}
// A run of a pool (offsets, in whole words) as the pass's chunks, in order: the words before its first 16-byte vector, the
// vectors 64 KB at most at a time, the words after the last; emit(offset, bytes, vectors). (tests/test_wb_diff_split.py)
template <class F> void wbd_split(uint64_t at, uint64_t size, F&& emit) {
    const uint64_t end = at + size;
    const uint64_t body = std::min<uint64_t>(end, (at + 15) & ~(uint64_t)15), body_end = std::max<uint64_t>(body, end & ~(uint64_t)15);
    if (body > at) emit(at, body - at, false);
    for (uint64_t a = body; a < body_end;) {
        const uint64_t len = std::min<uint64_t>(body_end - a, 64u << 10);
        emit(a, len, true);
        a += len;
    }
    if (end > body_end) emit(body_end, end - body_end, false);
}
// (pools_mu held) the shadow-follows-sync entries of a freed shadow dropped, before shadows_reap destroys it (a shadow made
// after may have its handle: the entries would mark that one's pages known, which never had the copy)
void shadow_synced_freed() {
    if (dead_shadows.empty() || shadow_synced.empty()) return;
    std::erase_if(shadow_synced, [](const ShadowSynced& s) {
        for (const DeadShadow& d : dead_shadows)
            if (d.buffer == s.shadow) return true;
        return false;
    });
}
// (pools_mu held) the late pages' fix: a stamp block whose shadow was freed counts for none: filled again and tied to its
// pool's shadow when a late page is next found there (a new shadow may have the freed one's handle and memory, and the
// stamps were the old one's)
void late_fix_shadows_freed() {
    if (dead_shadows.empty() || stamp_blocks.empty()) return;
    for (auto& [key, b] : stamp_blocks) {
        if (!b.shadow) continue;
        for (const DeadShadow& d : dead_shadows)
            if (d.buffer == b.shadow) {
                b.shadow = VK_NULL_HANDLE, b.since = 0;
                break;
            }
    }
}
struct WbdChunk {   // (shaders/wb_diff.comp's Chunk)
    uint64_t mirror, shadow, guest;
    uint32_t count, vecs;   // (16-byte vectors when bit 0 is set, else words; bit 1: it has stamps, the late pages' fix:
                            // their address in a uint64 a chunk right after the table's chunks, which then has no observations)
};
bool wbd_observe_on() {
    static const bool on = env_on("RX_VK_WB_OBSERVE");
    return on;
}
constexpr uint64_t kWbdObserveChunks = 256u << 10; // At most 1 MiB of optional flags per table.
struct WbdObserved {
    uint64_t batches = 0, chunks = 0, changed = 0, bytes = 0, changed_bytes = 0, missing = 0, last_value = 0;
} wbd_observed;
void wbd_observe_missed(uint64_t chunks, const char* reason) {
    wbd_observed.missing += chunks;
    static unsigned said;
    if (said++ < 8) logf("vk: write-back observations unavailable: %s; ordinary write-back kept", reason);
}
// (the pass's tables, mapped, a ring: one is taken again once the submission that read it is done)
struct WbdTable {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    uint8_t* map = nullptr;
    VkDeviceAddress address = 0;
    uint64_t bytes = 0, value = 0;
    uint64_t observed_offset = 0;
    uint32_t observed_count = 0;
};
WbdTable wbd_tables[64];
int wbd_table_next = 0;
void wbd_observe_collect(WbdTable& t) {
    if (!t.observed_count || t.value > mem_completed) return;
    const uint64_t count = t.observed_count, offset = t.observed_offset;
    t.observed_count = 0; // Consume once, including a failed read. Never turn missing data into clean chunks.
    if (!t.value || !t.map || count > kWbdObserveChunks || offset != count * sizeof(WbdChunk) ||
        offset > t.bytes || count * sizeof(uint32_t) > t.bytes - offset ||
        !invalidate_mapped_checked(t.alloc, offset + count * sizeof(uint32_t))) {
        wbd_observe_missed(count, "result bounds or host visibility");
        return;
    }
    const auto* chunks = reinterpret_cast<const WbdChunk*>(t.map);
    const auto* flags = reinterpret_cast<const uint32_t*>(t.map + offset);
    uint64_t changed = 0, bytes = 0, changed_bytes = 0;
    for (uint64_t i = 0; i < count; ++i) {
        // (vecs 0 or 1 only: a table with the late pages' fix's stamps has no observations)
        const uint64_t size = (uint64_t)chunks[i].count * (chunks[i].vecs ? 16u : 4u);
        if (flags[i] > 1 || chunks[i].vecs > 1 || !size || size > (64u << 10)) {
            wbd_observe_missed(count, "incomplete or invalid GPU result");
            return;
        }
        bytes += size;
        if (flags[i]) ++changed, changed_bytes += size;
    }
    ++wbd_observed.batches;
    wbd_observed.chunks += count;
    wbd_observed.changed += changed;
    wbd_observed.bytes += bytes;
    wbd_observed.changed_bytes += changed_bytes; // Bytes covered by changed chunks, not exact changed-byte counts.
    wbd_observed.last_value = std::max(wbd_observed.last_value, t.value);
}
bool wbd_table_make(uint64_t bytes, WbdTable& t) {
    if (!bytes || bytes > UINT64_MAX - 65535) return false;
    const uint64_t n = std::max<uint64_t>(128u << 10, (bytes + 65535) & ~65535ull);
    void* map = nullptr;
    if (create_buffer(n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                      true, t.buf, t.alloc, &map) && map) {
        VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        da.buffer = t.buf;
        t.address = vkGetBufferDeviceAddress(dev.device, &da);
        if (t.address) {
            t.map = (uint8_t*)map;
            t.bytes = n;
            return true;
        }
    }
    if (t.buf) destroy_buffer(t.buf, t.alloc);
    t = {};
    return false;
}
// (stamps: the bytes after the chunks that the late pages' fix's stamp addresses take; a table with them has no observations)
WbdTable* wbd_table_get(uint64_t chunk_bytes, uint64_t stamps = 0) {
    WbdTable& t = wbd_tables[wbd_table_next];
    if (t.value > mem_completed) return nullptr;   // (still read by a submission the GPU has not done)
    wbd_observe_collect(t); // Before its input table or its trailing flags can be overwritten or freed.
    if (stamps > UINT64_MAX - chunk_bytes) return nullptr;
    const uint64_t bytes = chunk_bytes + stamps;
    const uint64_t count = chunk_bytes / sizeof(WbdChunk);
    const uint64_t extra = !stamps && wbd_observe_on() && mem_rec_value && chunk_bytes % sizeof(WbdChunk) == 0 &&
                           count <= kWbdObserveChunks ? count * sizeof(uint32_t) : 0;
    const uint64_t wanted = bytes <= UINT64_MAX - extra ? bytes + extra : bytes;
    if (t.bytes < wanted) {
        WbdTable next;
        if (wbd_table_make(wanted, next)) {
            if (t.buf) destroy_buffer(t.buf, t.alloc);
            t = next;
        } else if (t.bytes < bytes) {
            // Optional growth failed. Keep an existing input table when it fits; otherwise retry the ordinary
            // size after releasing the retired old buffer, as the old growth path did.
            const bool retry = t.buf || wanted != bytes;
            if (t.buf) destroy_buffer(t.buf, t.alloc);
            t = {};
            if (!retry || !wbd_table_make(bytes, t)) return nullptr;
        }
    }
    wbd_table_next = (wbd_table_next + 1) % 64;
    t.value = mem_rec_value;
    return &t;
}
VkDeviceAddress wbd_observe_begin(WbdTable& t, uint64_t count) {
    if (!wbd_observe_on()) return 0;
    const uint64_t offset = count * sizeof(WbdChunk);
    if (!count || count > kWbdObserveChunks || !t.value || !t.map || !t.address ||
        offset > t.bytes || count * sizeof(uint32_t) > t.bytes - offset || t.address > UINT64_MAX - offset) {
        wbd_observe_missed(count, "result capacity");
        return 0;
    }
    // Sentinel detects missing stores too. Input and results share this slot's existing allocation and lifetime.
    memset(t.map + offset, 0xff, (size_t)count * sizeof(uint32_t));
    if (!flush_mapped_checked(t.alloc, offset + count * sizeof(uint32_t))) {
        wbd_observe_missed(count, "host flush");
        return 0;
    }
    t.observed_offset = offset;
    t.observed_count = (uint32_t)count;
    return t.address + offset;
}


void memory_end_submission(VkCommandBuffer cmd, uint64_t part) {
    std::lock_guard lk(pools_mu);
    late_fix_shadows_freed();
    late_fix_reap();
    late_fix_emit();   // (the late pages' fix: the stamps' new blocks filled before this write-back stamps them)
    {   // (every 600 submissions: how the mirror syncs' dirty pages were read)
        static int64_t n, l_live, l_cap;
        if (++n % 600 == 0) {
            wb_learn_report();
            if (wb_skips_log()) wb_skips_report();
            static int64_t l_old;
            logf("vk: mirror syncs (600 submits): %lld dirty pages from their submits' captures, %lld from earlier ones', "
                 "%lld from guest memory as it was at the replay; %zu captures kept; %lld pages the GPU wrote copied over "
                 "(%lld live); %lld kept for the GPU (RX_VK_PROTECT_BARE)", (long long)(st_sync_cap_pages - l_cap),
                 (long long)(st_sync_old_pages - l_old), (long long)(st_sync_live_pages - l_live), cap_bufs.size(),
                 (long long)st_sync_gpu_pages, (long long)st_sync_gpu_live, (long long)st_prot_kept.load());
            l_cap = st_sync_cap_pages, l_live = st_sync_live_pages, l_old = st_sync_old_pages;
            msync_pass_report();   // (RX_VK_MSYNC_PASS: how its dispatches went)
            if (shadow_bytes || st_wbd_runs) {
                // (hookshot) how much of the shadows holds anything: the pages they know, and the 1 MB parts with any page known,
                // which is what shadows made of parts would take (each is the whole pool now)
                uint64_t known_pages = 0, parts = 0;
                for (const auto& [lo, sp] : imported) {
                    if (!sp.shadow) continue;
                    uint64_t part_bits = 0;   // (256 pages a 1 MB part: 4 words of the bitmap)
                    for (size_t w = 0; w < sp.shadow_known.size(); w++) {
                        known_pages += (uint64_t)__builtin_popcountll(sp.shadow_known[w]);
                        part_bits |= sp.shadow_known[w];
                        if ((w & 3) == 3 || w + 1 == sp.shadow_known.size()) parts += part_bits != 0, part_bits = 0;
                    }
                }
                logf("vk: write-back diff (600 submits): %lld runs, %.1f MB compared; %lld pages read into the shadows; %llu MB of shadows "
                     "(%llu MB of pages known, in %llu MB of 1 MB parts); pages the shadows knew that the CPU wrote: %lld in captures, %lld "
                     "only in the collection, %lld only by the replay itself",
                     (long long)st_wbd_runs, st_wbd_bytes / 1048576.0, (long long)st_wbd_fill_pages, (unsigned long long)(shadow_bytes >> 20),
                     (unsigned long long)(known_pages * 4096 >> 20), (unsigned long long)parts,
                     (long long)st_wbd_known_game, (long long)st_wbd_known_collected, (long long)st_wbd_known_replay);
            }
            st_wbd_runs = st_wbd_bytes = st_wbd_fill_pages = 0;
            st_wbd_known_game = st_wbd_known_collected = st_wbd_known_replay = 0;
            if (wbd_observe_on()) {
                logf("vk: write-back observations (completed since last report): %llu batches, %llu chunks, %llu changed; "
                     "%.1f MB compared, %.1f MB covered by changed chunks; %llu unavailable chunks; last completed value %llu",
                     (unsigned long long)wbd_observed.batches, (unsigned long long)wbd_observed.chunks,
                     (unsigned long long)wbd_observed.changed, wbd_observed.bytes / 1048576.0,
                     wbd_observed.changed_bytes / 1048576.0, (unsigned long long)wbd_observed.missing,
                     (unsigned long long)wbd_observed.last_value);
                wbd_observed = {};
            }
            std::vector<std::pair<int64_t, uint64_t>> top;
            for (const auto& [lo, k] : live_by_pool) top.push_back({k, lo});
            std::sort(top.rbegin(), top.rend());
            std::string s;
            for (size_t i = 0; i < top.size() && i < 8; i++) {
                const auto it = imported.find(top[i].second);
                char b[160];
                bool captured = false;
                {
                    std::lock_guard ck(cap_ranges_mu);
                    for (const auto& r : cap_ranges) captured |= top[i].second >= r.first && top[i].second < r.second;
                }
                snprintf(b, sizeof b, " %#llx (%llu MB%s, state %d): %lld;", (unsigned long long)top[i].second,
                         it != imported.end() ? (unsigned long long)((it->second.hi - it->second.lo) >> 20) : 0ull,
                         captured ? ", captured" : ", NOT captured", it != imported.end() ? (int)it->second.mstate : -1,
                         (long long)top[i].first);
                s += b;
            }
            logf("vk:   read live, by pool:%s", s.c_str());
            live_by_pool.clear();
            // (and the bindings the GPU read in guest memory itself: pools with no mirror yet, or none (state 9: bound
            // outside a recording))
            std::vector<std::pair<int64_t, uint64_t>> dtop;
            for (const auto& [lo, e] : direct_by_pool) dtop.push_back({e.first, lo});
            std::sort(dtop.rbegin(), dtop.rend());
            std::string ds;
            for (size_t i = 0; i < dtop.size() && i < 8; i++) {
                const auto it = imported.find(dtop[i].second);
                char b[160];
                snprintf(b, sizeof b, " %#llx (%llu MB, state %d): %lld binds, %.0f MB;", (unsigned long long)dtop[i].second,
                         it != imported.end() ? (unsigned long long)((it->second.hi - it->second.lo) >> 20) : 0ull,
                         direct_by_pool[dtop[i].second].second, (long long)dtop[i].first,
                         direct_bytes_by_pool[dtop[i].second] / 1048576.0);
                ds += b;
            }
            if (!dtop.empty()) {
                double all = 0;
                for (const auto& [lo, bytes] : direct_bytes_by_pool) all += (double)bytes;
                logf("vk:   bound in guest memory, by pool (the GPU reads these over PCIe; binds and the bytes of their ranges over these 600 submits, "
                     "%.1f MB in all, a range counted once a submit): %s", all / 1048576.0, ds.c_str());
            }
            direct_by_pool.clear(), direct_bytes_by_pool.clear();
        }
    }
    static const bool no_writeback = env_on("RX_VK_NO_WRITEBACK");   // (an experiment: none)
    if (no_writeback) writebacks.clear();
    if (!writebacks.empty()) {
        // per pool, the ranges merged, one copy each (mirror -> the guest's pool)
        std::sort(writebacks.begin(), writebacks.end(), [](const WriteBack& a, const WriteBack& b) {
            return a.pool != b.pool ? a.pool < b.pool : a.lo < b.lo;
        });
        GprofSpan gp(cmd, "wb copies (mirror -> guest memory, with its two barriers)");   // (RX_VK_PASS_TIMING)
        gp.items = 0;
        transfer_barrier(cmd);
        std::vector<VkBufferCopy> runs;
        // (the write-back's diff: the runs of pools with a shadow, as chunks of 64 KB at most, for one pass after the loop)
        const bool diff = wb_diff_on() && wbd_pipe_ready();
        if (diff) shadows_forget_written(), shadow_synced_freed(), shadows_reap(), shadows_check_room(), late_fix_shadows_freed(), shadow_synced_known();
        static thread_local std::vector<WbdChunk> chunks;
        struct ChunkSrc { Pool* p; uint64_t at, len; };
        static thread_local std::vector<ChunkSrc> chunk_src;   // (for the whole copy, when no table is free)
        static thread_local std::vector<uint64_t> chunk_stamps;   // (the late pages' fix: a chunk's stamps, or 0)
        bool any_stamps = false;
        static thread_local std::vector<VkBufferCopy> fills, odd;
        chunks.clear(), chunk_src.clear(), chunk_stamps.clear();
        bool filled = false;
        for (size_t i = 0; i < writebacks.size();) {
            Pool* p = writebacks[i].pool;
            runs.clear();
            for (; i < writebacks.size() && writebacks[i].pool == p; i++) {
                const WriteBack& w = writebacks[i];
                if (!runs.empty() && w.lo <= runs.back().srcOffset + runs.back().size)
                    runs.back().size = std::max(runs.back().size, w.hi - runs.back().srcOffset);
                else
                    runs.push_back({w.lo, w.lo, w.hi - w.lo});
            }
            cut_cpu_words(*p, runs);
            if (p->sparse) {   // (hookshot: never from a sparse mirror's block with no memory, which reads 0. A written binding
                               // that ran past every game pool into one, served from the mirror, has its part there dropped)
                std::vector<sparse_mirror::Copy> in;
                for (const VkBufferCopy& r : runs) in.push_back({r.srcOffset, r.dstOffset, r.size});
                const auto has = [&](uint64_t o) { return o / sparse_block(*p) < p->has.size() && block_has(*p, (size_t)(o / sparse_block(*p))); };
                const std::vector<sparse_mirror::Copy> out = sparse_mirror::clip(in, sparse_block(*p), has, has);
                uint64_t before = 0, after = 0;
                for (const auto& c : in) before += c.size;
                for (const auto& c : out) after += c.size;
                if (after != before) {
                    // (expected where those blocks are all outside every game pool, a binding served from the mirror past a
                    // pool's end: nothing of the game's there. Told only where a game pool is: not expected)
                    bool in_pool = false;
                    for (const auto& c : in)
                        if (!sparse_gaps_outside_pools(*p, c.src, c.size)) {
                            in_pool = true;
                            break;
                        }
                    static int logged;
                    if (in_pool && logged++ < 8)
                        logf("vk: sparse mirror of pool %#llx: %llu bytes of a write-back reached blocks with no memory where a game "
                             "pool is: left out", (unsigned long long)p->lo, (unsigned long long)(before - after));
                    runs.clear();
                    for (const auto& c : out) runs.push_back({c.src, c.dst, c.size});
                }
            }
            if (runs.empty()) continue;
            checkpoint_mark(cmd, p->lo, 'b');
            uint64_t run_bytes = 0;
            for (const VkBufferCopy& r : runs) run_bytes += r.size;
            if (diff && !p->shadow && (run_bytes >= (256u << 10) || late_shadow_wanted(p->lo))) {   // (or one the late pages' fix wants)
                // (a pool whose write-back is worth one, as VRAM allows; wanted stays after VRAM takes it, for shadows_check_room)
                p->shadow_wanted = true;
                shadow_make(*p);
            }
            if (diff && p->shadow && p->address && p->mirror) {
                VkBufferDeviceAddressInfo mda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                mda.buffer = p->mirror;
                const VkDeviceAddress mirror_address = vkGetBufferDeviceAddress(dev.device, &mda);
                // (a run not in whole words, as a copy of the game's may be: copied whole, its pages then unknown to the shadow)
                odd.clear();
                for (const VkBufferCopy& r : runs)
                    if ((r.dstOffset | r.size) & 3) odd.push_back(r);
                // the pages of the runs the shadow does not know: from guest memory, whole, first
                fills.clear();
                const uint64_t pool_n = p->hi - p->lo;
                for (const VkBufferCopy& r : runs)
                    for (uint64_t pg = r.dstOffset >> kPageShift; !((r.dstOffset | r.size) & 3) && pg <= (r.dstOffset + r.size - 1) >> kPageShift; pg++) {
                        uint64_t& bits = p->shadow_known[pg >> 6];
                        if (bits & (1ull << (pg & 63))) continue;
                        bits |= 1ull << (pg & 63);
                        const uint64_t at = pg << kPageShift, len = std::min<uint64_t>(4096, pool_n - at);
                        if (!fills.empty() && fills.back().srcOffset + fills.back().size == at)
                            fills.back().size += len;
                        else
                            fills.push_back({at, at, len});
                        st_wbd_fill_pages++;
                    }
                if (!fills.empty()) {
                    vkCmdCopyBuffer(cmd, p->buffer, p->shadow, (uint32_t)fills.size(), fills.data());
                    filled = true;
                }
                if (!odd.empty()) {
                    vkCmdCopyBuffer(cmd, p->mirror, p->buffer, (uint32_t)odd.size(), odd.data());
                    for (const VkBufferCopy& r : odd) shadow_forget(*p, r.dstOffset, r.size);
                }
                const bool stamped = stamps_in(*p);   // (the late pages' fix: chunks cut at its 64 KB stamp blocks)
                for (const VkBufferCopy& r : runs) {
                    if ((r.dstOffset | r.size) & 3) {
                        wb_kind_note(WB_WHOLE, p->lo + r.dstOffset, p->lo + r.dstOffset + r.size, mem_rec_value);
                        late_snap_whole(cmd, *p, r.dstOffset, r.size);   // (copied whole above, with the odd ones)
                        continue;
                    }
                    wb_kind_note(WB_DIFF, p->lo + r.dstOffset, p->lo + r.dstOffset + r.size, mem_rec_value);
                    wbd_split(r.dstOffset, r.size, [&](uint64_t at, uint64_t len, bool vecs) {
                        while (len) {
                            uint64_t piece = len, st = 0;
                            if (stamped) {
                                const uint64_t g = p->lo + at;
                                piece = std::min<uint64_t>(len, (((g >> kStampShift) + 1) << kStampShift) - g);
                                st = stamp_address(*p, g);
                            }
                            chunks.push_back({mirror_address + at, p->shadow_address + at, p->address + at,
                                              (uint32_t)(piece / (vecs ? 16 : 4)), (vecs ? 1u : 0u) | (st ? 2u : 0u)});
                            chunk_src.push_back({p, at, piece});
                            chunk_stamps.push_back(st);
                            any_stamps |= st != 0;
                            at += piece, len -= piece;
                        }
                    });
                }
                st_wbd_runs += (int64_t)runs.size(), st_wbd_bytes += (int64_t)run_bytes;
            } else {
                vkCmdCopyBuffer(cmd, p->mirror, p->buffer, (uint32_t)runs.size(), runs.data());
                for (const VkBufferCopy& r : runs) {
                    wb_kind_note(WB_WHOLE, p->lo + r.dstOffset, p->lo + r.dstOffset + r.size, mem_rec_value);
                    late_snap_whole(cmd, *p, r.dstOffset, r.size);
                }
            }
            for (const VkBufferCopy& r : runs) st_wb_bytes += (int64_t)r.size, gp.bytes += r.size;
            gp.items += (int)runs.size();
            for (const VkBufferCopy& r : runs) wb_pages_note(p->lo + r.dstOffset, p->lo + r.dstOffset + r.size, mem_rec_value, part);   // (EvilBoyce)
            {   // (hookshot) RX_VK_LOG_WRITEBACK=1: the pools written back the most, every 600 write-back submissions
                static const bool log_wb = env_on("RX_VK_LOG_WRITEBACK");
                if (log_wb) {
                    static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> by_pool;   // lo -> bytes, runs
                    static int64_t subs;
                    auto& e = by_pool[p->lo];
                    for (const VkBufferCopy& r : runs) e.first += r.size;
                    e.second += runs.size();
                    if (i == writebacks.size() && ++subs % 600 == 0) {
                        std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> v(by_pool.begin(), by_pool.end());
                        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
                        std::string s;
                        for (size_t k = 0; k < v.size() && k < 8; k++) {
                            char b[120];
                            const auto it = imported.find(v[k].first);
                            snprintf(b, sizeof b, " %#llx (%llu MB pool): %.1f MB in %.0f runs;", (unsigned long long)v[k].first,
                                     it != imported.end() ? (unsigned long long)((it->second.hi - it->second.lo) >> 20) : 0ull,
                                     v[k].second.first / 600.0 / 1048576.0, v[k].second.second / 600.0);
                            s += b;
                        }
                        logf("vk: write-back a submission, by pool:%s", s.c_str());
                        by_pool.clear();
                    }
                }
            }
        }
        if (!chunks.empty()) {
            WbdTable* t = wbd_table_get(chunks.size() * sizeof(WbdChunk), any_stamps ? chunks.size() * sizeof(uint64_t) : 0);
            if (!t) {   // (no table free: the whole copies, the shadows forgotten for those pages)
                if (wbd_observe_on()) wbd_observe_missed(chunks.size(), "chunk table busy or unavailable");
                for (const ChunkSrc& c : chunk_src) {
                    const VkBufferCopy r{c.at, c.at, c.len};
                    vkCmdCopyBuffer(cmd, c.p->mirror, c.p->buffer, 1, &r);
                    shadow_forget(*c.p, c.at, c.len);
                    wb_kind_note(WB_WHOLE, c.p->lo + c.at, c.p->lo + c.at + c.len, mem_rec_value);
                    late_snap_whole(cmd, *c.p, c.at, c.len);
                }
                static int logged;
                if (logged++ < 8) logf("vk: write-back diff: no chunk table free (%zu chunks): this one copied whole", chunks.size());
            }
            if (filled) transfer_barrier(cmd);   // (the shadows' pages from guest memory before the pass reads them)
            if (t) {
                memcpy(t->map, chunks.data(), chunks.size() * sizeof(WbdChunk));
                if (any_stamps) {   // (the stamps' addresses after the chunks: shaders/wb_diff.comp reads them there)
                    if (wbd_observe_on()) wbd_observe_missed(chunks.size(), "a table with stamps");
                    memcpy(t->map + chunks.size() * sizeof(WbdChunk), chunk_stamps.data(), chunks.size() * sizeof(uint64_t));
                }
                const VkDeviceAddress observations = any_stamps ? 0 : wbd_observe_begin(*t, chunks.size());
                if (!observations) flush_mapped(t->alloc, chunks.size() * (sizeof(WbdChunk) + (any_stamps ? sizeof(uint64_t) : 0)));
                GprofSpan gd(cmd, "wb diff (mirror -> guest memory where it differs from the shadow)");
                gd.items = (int)chunks.size();
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, wbd_pipe.pipe);
                const struct { uint64_t table; uint32_t count, pad; uint64_t observations; }
                    pc{t->address, (uint32_t)chunks.size(), (uint32_t)mem_rec_value, observations};
                vkCmdPushConstants(cmd, wbd_pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
                const uint32_t n = (uint32_t)chunks.size();
                // (its own marker: without one a fault in it read as the last pool's copy, "write-back of pool", in the
                // device-lost logs of an RTX 5060 and a GTX 1660 Ti)
                checkpoint_mark(cmd, n, 'w');
                vkCmdDispatch(cmd, std::min<uint32_t>(n, 65535u), (n + 65534u) / 65535u, 1);
                if (observations) {
                    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
                    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
                    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    dependency.memoryBarrierCount = 1;
                    dependency.pMemoryBarriers = &barrier;
                    vkCmdPipelineBarrier2(cmd, &dependency);
                }
            }
        }
        transfer_barrier(cmd);
        writebacks.clear();
    }
    // (what this submission synced is current as of now: the GPU's own writes, stamped while it was recorded, are
    // in the mirror and written back; CPU writes collected at the next submission get a newer generation)
    for (Pool* p : mirror_synced) p->synced_gen = gen_now;
    mirror_synced.clear();
    mirror_upload = false;
}

/// (RX_VK_RES_LOG=1) one line of the process's memory: commit, working set, the C heaps, and what this file keeps in
/// system memory (the captures' buffers, the texture-storage shadow, the capture table)
void memory_res_log() {
    if (!res_log_on()) return;
    struct Counters {
        DWORD cb, PageFaultCount;
        SIZE_T PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage, QuotaPeakNonPagedPoolUsage,
            QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage, PrivateUsage;
    };
    using MemInfo = BOOL(WINAPI*)(HANDLE, Counters*, DWORD);
    struct Summary { DWORD cb; SIZE_T allocated, committed, reserved, max_reserve; };
    using HeapSum = BOOL(WINAPI*)(HANDLE, DWORD, Summary*);
    static const MemInfo mem_info = (MemInfo)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32GetProcessMemoryInfo");
    static const HeapSum heap_sum = (HeapSum)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "HeapSummary");
    Counters m{};
    m.cb = sizeof m;
    if (mem_info) mem_info(GetCurrentProcess(), &m, sizeof m);
    uint64_t h_alloc = 0, h_commit = 0;
    HANDLE heaps[64];
    const DWORD nh = GetProcessHeaps(64, heaps);
    for (DWORD i = 0; i < nh && i < 64 && heap_sum; i++) {
        Summary su{};
        su.cb = sizeof su;
        if (heap_sum(heaps[i], 0, &su)) h_alloc += su.allocated, h_commit += su.committed;
    }
    size_t kept = 0, spares = 0;
    {
        std::lock_guard lk(cap_mu);
        kept = captures.size(), spares = spare.size();
    }
    size_t chunks = 0;   // (the capture table's chunks made: the replay's, read racily, for the count only)
    for (const auto& c : cap_table) chunks += c != nullptr;
    uint64_t free_n = 0, free_b = 0, made = 0, aged = 0, gets = 0, need_max = 0, taken_over = 0, free_low = 0, used_b = 0;
    char spare_sizes[160] = "";
    {
        std::lock_guard lk(cap_free_mu);
        free_n = cap_free.size(), free_b = cap_free_bytes, made = cap_bufs_made, aged = cap_spares_aged, gets = cap_gets;
        need_max = cap_need_max, taken_over = cap_taken_over, free_low = cap_free_low == ~0ull ? cap_free.size() : cap_free_low;
        cap_need_max = 0, cap_taken_over = 0, cap_free_low = ~0ull;
        int n = 0;
        for (const FreeBuf& f : cap_free)
            if (n < 150) n += snprintf(spare_sizes + n, sizeof spare_sizes - n, " %llu", (unsigned long long)(f.size >> 20));
    }
    used_b = cap_used;   // (the replay's, read racily, for the count only)
    logf("vk: capture buffers since the last line: largest need %llu KB, %llu spares over twice the need's size class (refused with "
         "RX_VK_CAP_FIT, else taken), fewest "
         "spares left after a take %llu; kept buffers filled %llu MB of %llu; spare sizes (MB):%s",
         (unsigned long long)(need_max >> 10), (unsigned long long)taken_over, (unsigned long long)free_low,
         (unsigned long long)(used_b >> 20), (unsigned long long)(cap_bytes >> 20), spare_sizes);
    logf("vk: process memory: private %llu MB, working set %llu MB (peak %llu); C heaps (%lu) %llu MB in use of %llu committed; "
         "captures: %zu queued, %zu buffers kept %llu MB, %llu spare buffers %llu MB (%llu made, %llu spares aged out of %llu asked), "
         "%zu spare records; texture shadow %zu pages "
         "(%llu MB); capture table %zu chunks (%llu MB)",
         (unsigned long long)(m.PrivateUsage >> 20), (unsigned long long)(m.WorkingSetSize >> 20),
         (unsigned long long)(m.PeakWorkingSetSize >> 20), (unsigned long)nh, (unsigned long long)(h_alloc >> 20),
         (unsigned long long)(h_commit >> 20), kept, cap_bufs.size(), (unsigned long long)(cap_bytes >> 20),
         (unsigned long long)free_n, (unsigned long long)(free_b >> 20), (unsigned long long)made,
         (unsigned long long)aged, (unsigned long long)gets, spares, shadow_n.load(),
         (unsigned long long)((uint64_t)shadow_n.load() * 4096 >> 20), chunks,
         (unsigned long long)((uint64_t)chunks * kCapChunkPages * sizeof(CapEntry) >> 20));
    size_t g = 0, used = 0;
    int64_t dropped = 0;
    res_pipeline_counts(&g, &used, &dropped);
    logf("vk: pipelines: %zu graphics made, %zu pipelines bound so far; %lld compute shader modules let go", g, used,
         (long long)dropped);
    mem_kinds_log();
}
/// (RX_VK_FRAME_TRACE) running totals for the frame trace: the captures' GPU waits for their late pages, and the
/// mirror syncs' scatter bytes
void memory_trace_counts(int64_t* late_ns, int64_t* late_n, int64_t* scatter_bytes) {
    *late_ns = st_tr_late_ns.load(std::memory_order_relaxed), *late_n = st_tr_late_n.load(std::memory_order_relaxed);
    *scatter_bytes = st_scatter_bytes.load(std::memory_order_relaxed);
}
/// (RX_VK_LEDGER) the pools' part of a ledger snapshot: the game's pool flags, the live mirrors and imports, the retired mirrors
void memory_ledger_snapshot(LedgerSnap& s) {
    {
        std::lock_guard lk(nvn_pools_mu);
        s.pool_flags.reserve(nvn_pools.size());
        for (const auto& [lo, np] : nvn_pools) s.pool_flags.push_back({lo, np.hi, np.flags});
    }
    std::lock_guard lk(pools_mu);
    for (const auto& [lo, p] : imported) {
        const bool imp = p.memory != VK_NULL_HANDLE, mir = p.mstate == Pool::MIRROR_ON && p.mirror != VK_NULL_HANDLE;
        if (imp) s.imports.push_back({p.lo, p.hi});
        if (mir) s.mirrors.push_back({p.lo, p.hi});
        LedgerChunk c;
        c.lo = p.lo, c.hi = p.hi, c.import = imp, c.mirror = mir, c.gpu_written = p.gpu_written;
        c.bound_ago = mem_rec_value > p.last_value ? mem_rec_value - p.last_value : 0;
        c.synced_ago = mirror_epoch > p.synced_epoch ? mirror_epoch - p.synced_epoch : 0;
        s.chunks.push_back(c);
    }
    for (const auto& nh : retired_pools) s.retired_chunks++, s.retired_chunk_bytes += nh.mapped().hi - nh.mapped().lo;
    s.evicted_chunks = evicted_pools.size();
    for (const DeadMirror& d : dead_mirrors) s.retired_mirror_bytes += allocation_size(d.alloc);
}
// ---- (hookshot) The mirror syncs' pages from the captures by a compute pass (RX_VK_MSYNC_PASS, on unless set off) ----
// A capture's buffer is host-cached system memory (create_buffer_cached), and a mirror sync copied its pages into the mirror as
// copy regions, one or a few pages each (append_capture_copy joins only the contiguous ones): the GPU reads each region over
// PCIe after the one before. The write-back diff's pass (shaders/wb_diff.comp) is a copy when its shadow is the destination
// itself: it reads the source, compares it with the destination, and writes it there where they differ (twice, the same
// address). So the regions go as its chunks, one workgroup a chunk, and a sync's chunks in one dispatch where their copies
// were: the presync's (memory_sync_mirrors_before_segments) every pool's at once. No barrier between the chunks: the copies had
// none either (each page of a mirror comes from one place in a sync), and the full barrier that ends the uploads
// (texture_uploads_recorded) waits for all commands, the dispatch's too. The tables are blocks of host memory, a block taken
// again once the submission that last read it is done; a sync whose table finds no room goes as the copies, as before. (Not
// the write-back diff's own ring, wbd_table_get: one table of 128 KB or more a dispatch there, and a serial replay makes about
// 25 of these dispatches a frame, so its 64 tables could all be in flight; a write-back that finds none copies whole and
// forgets its shadows.)
std::vector<WbdChunk> msync_chunks;   // (pools_mu)
struct MsyncCopy {
    VkBuffer src, dst;
    std::vector<VkBufferCopy> rs;
};
std::vector<MsyncCopy> msync_copies;   // (the same, as copies: for a sync whose table finds no room)
bool msync_batching = false;
int64_t st_msync_dispatches = 0, st_msync_chunks = 0, st_msync_fallbacks = 0;
struct MsyncBlock {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    uint8_t* map = nullptr;
    VkDeviceAddress address = 0;
    uint64_t value = 0, fill = 0;
};
constexpr int kMsyncBlocks = 16;
constexpr uint64_t kMsyncBlockBytes = 256u << 10;   // (8192 chunks: a frame's presync took ~1600 at the Hyrule Field spot)
MsyncBlock msync_blocks[kMsyncBlocks];
int msync_block_cur = -1;

/// (pools_mu held) Room for a table of `bytes`: in the current block, or in the next one once the GPU is done with it
MsyncBlock* msync_block_get(uint64_t bytes) {
    if (bytes > kMsyncBlockBytes) return nullptr;
    if (msync_block_cur >= 0) {
        MsyncBlock& b = msync_blocks[msync_block_cur];
        if (b.map && b.fill + bytes <= kMsyncBlockBytes) {
            b.value = std::max(b.value, mem_rec_value);
            return &b;
        }
    }
    const int next = (msync_block_cur + 1) % kMsyncBlocks;
    MsyncBlock& b = msync_blocks[next];
    if (b.map && b.value > mem_completed) return nullptr;   // (still read by a submission the GPU has not done)
    if (!b.map) {
        void* map = nullptr;
        if (!create_buffer(kMsyncBlockBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, true,
                           b.buf, b.alloc, &map) || !map) {
            if (b.buf) destroy_buffer(b.buf, b.alloc);
            b = {};
            return nullptr;
        }
        VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        da.buffer = b.buf;
        b.address = vkGetBufferDeviceAddress(dev.device, &da);
        if (!b.address) {
            destroy_buffer(b.buf, b.alloc);
            b = {};
            return nullptr;
        }
        b.map = (uint8_t*)map;
    }
    msync_block_cur = next;
    b.fill = 0;
    b.value = mem_rec_value;
    return &b;
}

/// (pools_mu held) A capture buffer's runs into p's mirror (offsets: the buffer's, the pool's) as the pass's chunks, for
/// msync_pass_flush; false: none taken, the caller copies them (no pass, or a run not in whole words)
bool msync_pass_add(Pool& p, VkBuffer buf, const std::vector<VkBufferCopy>& rs) {
    if (rs.empty() || !p.mirror || !buf || !wbd_pipe_ready()) return false;
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = buf;
    const VkDeviceAddress src = vkGetBufferDeviceAddress(dev.device, &da);
    da.buffer = p.mirror;
    const VkDeviceAddress dst = vkGetBufferDeviceAddress(dev.device, &da);
    if (!src || !dst) return false;
    for (const VkBufferCopy& r : rs)
        if (!r.size || ((src + r.srcOffset) | (dst + r.dstOffset) | r.size) & 3) return false;
    for (const VkBufferCopy& r : rs) {
        const uint64_t s = src + r.srcOffset, d = dst + r.dstOffset;
        if ((s ^ d) & 15) {   // (not alike within 16 bytes: words, 64 KB a chunk)
            for (uint64_t at = 0; at < r.size; at += 64u << 10) {
                const uint64_t len = std::min<uint64_t>(r.size - at, 64u << 10);
                msync_chunks.push_back({s + at, d + at, d + at, (uint32_t)(len / 4), 0u});
            }
            continue;
        }
        wbd_split(d, r.size, [&](uint64_t a, uint64_t len, bool vecs) {   // (source, shadow, destination: the shadow is the destination)
            msync_chunks.push_back({s + (a - d), a, a, (uint32_t)(len / (vecs ? 16 : 4)), vecs ? 1u : 0u});
        });
    }
    msync_copies.push_back({buf, p.mirror, rs});
    return true;
}

/// (pools_mu held) The chunks taken since as one dispatch in the upload command buffer, or as their copies when no table has room
void msync_pass_flush() {
#ifdef TRACY_ENABLE
    HT_SCOPE("msync_pass_flush");
#endif
    if (msync_chunks.empty()) {
        msync_copies.clear();
        return;
    }
    VkCommandBuffer cmd = upload_command_buffer();
    const uint64_t bytes = msync_chunks.size() * sizeof(WbdChunk);
    MsyncBlock* b = msync_block_get(bytes);
    if (!b) {
        for (const MsyncCopy& c : msync_copies) {
            GprofSpan gp(cmd, "msync copies (CPU pages from a capture -> mirror)");
            gp.items = (int)c.rs.size();
            vkCmdCopyBuffer(cmd, c.src, c.dst, (uint32_t)c.rs.size(), c.rs.data());
            for (const VkBufferCopy& r : c.rs) gp.bytes += r.size;
        }
        st_msync_fallbacks++;
        static int logged;
        if (logged++ < 8)
            logf("vk: mirror syncs' pass: no table room for %zu chunks: this sync as copies", msync_chunks.size());
    } else {
        memcpy(b->map + b->fill, msync_chunks.data(), bytes);
        flush_mapped_range(b->alloc, b->fill, bytes);
        const uint32_t n = (uint32_t)msync_chunks.size();
        GprofSpan gp(cmd, "msync pass (CPU pages from a capture -> mirror, one dispatch)");
        gp.items = (int)n;
        for (const WbdChunk& c : msync_chunks) gp.bytes += (uint64_t)c.count * (c.vecs ? 16 : 4);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, wbd_pipe.pipe);
        const struct { uint64_t table; uint32_t count, pad; uint64_t observations; } pc{b->address + b->fill, n, 0, 0};   // (no observations)
        vkCmdPushConstants(cmd, wbd_pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(cmd, std::min<uint32_t>(n, 65535u), (n + 65534u) / 65535u, 1);
        b->fill += bytes;
        st_msync_dispatches++, st_msync_chunks += (int64_t)n;
    }
    msync_chunks.clear();
    msync_copies.clear();
}

/// (pools_mu held, every 600 submissions: memory_end_submission) How the pass's dispatches went
void msync_pass_report() {
    if (!msync_pass_on()) return;
    logf("vk: mirror syncs' pass (600 submits): %lld dispatches, %lld chunks; %lld syncs as copies (no table room)",
         (long long)st_msync_dispatches, (long long)st_msync_chunks, (long long)st_msync_fallbacks);
    st_msync_dispatches = st_msync_chunks = st_msync_fallbacks = 0;
}

void memory_stats(int64_t* mirrors, uint64_t* mirror_mb, int64_t* sync_bytes, int64_t* sync_copies, int64_t* wb_bytes) {
    *mirrors = st_mirrors, *mirror_mb = mirror_bytes >> 20, *sync_bytes = st_sync_bytes, *sync_copies = st_sync_copies;
    *wb_bytes = st_wb_bytes;
}
// (hookshot) the game freed the address region [lo, hi): queued for the next submission (released_drain)
void memory_guest_release(uint64_t lo, uint64_t hi) {
    if (hi <= lo) return;
    std::lock_guard lk(released_mu);
    if (!released_queue.empty() && released_queue.back().second == lo) released_queue.back().second = hi;   // (the next one up)
    else if (!released_queue.empty() && released_queue.back().first == hi) released_queue.back().first = lo;   // (or down)
    else if (released_queue.size() < (1u << 16)) released_queue.push_back({lo, hi});
}
/// (RX_VK_LAG_LOG, vk_lag.cpp) The sampled pages of one capture against guest memory now: the pages the game has written other
/// bytes into since the capture copied them, by what the GPU has bound the page for. Kinds: 0 uniform, 1 vertex or index, 2
/// storage, indirect, GPU-written or another binding, 3 never bound by a record. A buffer already retired: all skipped.
void memory_lag_compare(uint32_t gen, const uint32_t* pages, const uint32_t* slots, size_t n, LagCmp& out) {
    const auto t0 = std::chrono::steady_clock::now();
    static thread_local uint8_t cur[4096];
    std::lock_guard lk(pools_mu);
    auto it = cap_bufs.find(gen);
    if (it == cap_bufs.end() || !it->second.map) {
        out.skipped += (int64_t)n;
        return;
    }
    const uint8_t* const map = static_cast<const uint8_t*>(it->second.map);
    const uint64_t used = it->second.used;
    const uint8_t* const base = host->guest_base();
    for (size_t i = 0; i < n; i++) {
        if ((uint64_t)slots[i] * 4096 + 4096 > used) { out.skipped++; continue; }
        if (!guarded_copy(cur, base + ((uint64_t)pages[i] << kPageShift), 4096)) { out.skipped++; continue; }
        out.sampled++;
        if (!std::memcmp(cur, map + (uint64_t)slots[i] * 4096, 4096)) continue;
        const uint16_t u = ledger_use_of(pages[i]);
        out.differ[u & kUseUniform ? 0 : u & (kUseVertex | kUseIndex) ? 1 : u ? 2 : 3]++;
    }
    out.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
void memory_pool_flags(uint64_t gpu, uint64_t size, uint32_t flags) {   // (the game's pools and their flags, for the write-backs left out)
    if (!size || gpu >= (1ull << 40)) return;
    cap_pool_born_now(gpu, size);   // (RX_VK_CAP_ONLY_GPU: a pool from the moment it is made)
    if (ledger_on()) ledger_pool_made(gpu, size);
    std::lock_guard lk(nvn_pools_mu);
    auto it = nvn_pools.upper_bound(gpu);
    if (it != nvn_pools.begin() && std::prev(it)->second.hi > gpu) --it;
    while (it != nvn_pools.end() && it->first < gpu + size) it = nvn_pools.erase(it);   // (memory reused)
    // (a new pool over memory an older one left out write-backs of: those GPU bytes are no one's now)
    if (!skipped_pages.empty())
        for (uint64_t q = gpu >> kPageShift; q < (gpu + size + 4095) >> kPageShift; q++) skipped_pages.erase((uint32_t)q);
    NvnPool np;
    np.hi = gpu + size, np.flags = flags;
    np.made_ns = (uint64_t)host->now_ns();
    nvn_pools[gpu] = np;
    nvn_pools_gen.fetch_add(1, std::memory_order_relaxed);
}
// ---- (RX_VK_POOL_MIRROR) the write-backs of never-imported pools, through staging ----
// The GPU's writes into a mirror reached guest memory by a GPU copy into the pool's import, done when the submission's timeline
// value is. Without an import the submission copies the written runs into a host-cached staging buffer instead, and the first
// thread to see that value done (memory_staged_apply: vk_replay.cpp's completedand its waits, before they report it) writes them
// into guest memory, page by page through the host's hos_gpu_write_live (only inside a live NVN pool, and not over a page the CPU
// wrote since its last collection: the staged lazy release's rules), so whoever then reads guest memory finds them there, as it did
// through the import.
struct StagedWb {
    uint64_t value = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    void* map = nullptr;
    uint64_t bytes = 0, cap = 0;   // (what it holds, and the buffer's size)
    std::vector<std::array<uint64_t, 3>> runs;   // (guest address, offset in the staging buffer, size)
};
std::mutex staged_mu;
std::deque<StagedWb> staged_wbs;   // (by value; staged_mu)
// (applied: kept for the next submissions' write-backs, up to 16 buffers and 256 MB; staged_mu. A buffer made and freed every
// submission, ~200 a second, cost the replay more than the copies)
std::vector<StagedWb> staged_free;
std::atomic<uint64_t> staged_min{~0ull};
int64_t st_sw_runs = 0, st_sw_bytes = 0, st_sw_pages_written = 0, st_sw_pages_cpu = 0, st_sw_pages_dead = 0, st_sw_pages_same = 0;
/// The host's checked write of a GPU result into guest memory (host/nvdrv.c hos_gpu_write_live: only inside a live
/// NVN pool, not over a page the CPU wrote since its last collection), found by name in the exe. The Linux exe exports it since
/// nx2/build.py's HOST_EXPORTS lists it; before, it had none there, and the never-imported pools' write-backs were written
/// unchecked (host->gpu_write). RX_VK_WRITE_LIVE=0: not looked up, as then (said once in the log either way)
using WriteLive = int (*)(uint64_t, const void*, uint64_t);
WriteLive write_live_fn() {
    static const WriteLive f = [] {
        const char* e = getenv("RX_VK_WRITE_LIVE");
        const bool want = !e || !env_value_off(e);   // (the switches' rule)
        const WriteLive fn = want ? (WriteLive)(void*)GetProcAddress(GetModuleHandleA(nullptr), "hos_gpu_write_live") : nullptr;
        logf("vk: the host's checked write of left-out write-backs (hos_gpu_write_live): %s",
             fn ? "found" : !want ? "not looked up (RX_VK_WRITE_LIVE=0)" : "not in the exe");
        return fn;
    }();
    return f;
}
void memory_staged_apply(uint64_t done) {
    if (staged_min.load(std::memory_order_acquire) > done) return;
    const WriteLive write_live = write_live_fn();
    static uint8_t* const base = host->guest_base ? host->guest_base() : nullptr;
    std::lock_guard lk(staged_mu);
    while (!staged_wbs.empty() && staged_wbs.front().value <= done) {
        StagedWb& s = staged_wbs.front();
        invalidate_mapped(s.alloc, s.bytes);
        for (const auto& [guest, off, size] : s.runs)
            for (uint64_t a = guest; a < guest + size;) {
                const uint64_t e = std::min(guest + size, ((a >> 12) + 1) << 12);
                const uint8_t* src = static_cast<const uint8_t*>(s.map) + off + (a - guest);
                // (most of what a write-back copies is what guest memory already holds: the same bytes left alone, no host call.
                // The import's GPU copy wrote them all, for free on the CPU; here each page written costs a call under the host's lock)
                if (base && !std::memcmp(base + a, src, e - a)) {
                    st_sw_pages_same++;
                    a = e;
                    continue;
                }
                const int r = write_live ? write_live(a, src, e - a) : (guest_cpu_write(a, src, e - a) ? 1 : 0);
                (r == 1 ? st_sw_pages_written : r == 2 ? st_sw_pages_cpu : st_sw_pages_dead)++;
                a = e;
            }
        s.runs.clear(), s.bytes = 0;
        staged_free.push_back(std::move(s));
        staged_wbs.pop_front();
    }
    staged_min.store(staged_wbs.empty() ? ~0ull : staged_wbs.front().value, std::memory_order_release);
}
/// (RX_VK_CAP_ONLY_GPU) a present: every 10 s, the capture's pages a frame (and with 2 those it would leave out), the reads
/// into the left-out ranges, and the largest of those ranges
void memory_cap_present() {
    if (!cap_only_gpu()) return;
    st_cap_presents.fetch_add(1, std::memory_order_relaxed);
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(10)) return;
    last = now;
    const double f = (double)std::max<int64_t>(st_cap_presents.exchange(0), 1);
    const int64_t pages = st_cap_pages.exchange(0), excl = st_excl_pages.exchange(0), takes = st_cap_takes.exchange(0);
    std::vector<std::pair<uint64_t, uint64_t>> ex;
    {
        std::lock_guard lk(cap_excl_mu);
        ex = cap_excl;
    }
    uint64_t total = 0;
    for (const auto& r : ex) total += r.second - r.first;
    std::sort(ex.begin(), ex.end(), [](const auto& a, const auto& b) { return a.second - a.first > b.second - b.first; });
    std::string top;
    for (size_t k = 0; k < ex.size() && k < 6; k++) {
        char t[64];
        snprintf(t, sizeof t, " %#llx+%lluK", (unsigned long long)ex[k].first, (unsigned long long)((ex[k].second - ex[k].first) >> 10));
        top += t;
    }
    logf("vk: capture (RX_VK_CAP_ONLY_GPU=%d), a frame over the last 10 s (%.0f frames, %.1f captures a frame): %.0f pages taken "
         "(%.1f MB), of them in the left-out ranges %.0f (counted with 2 only); reads and bindings into the left-out ranges %lld in "
         "all; left-out ranges %zu, %.0f MB, the largest:%s",
         cap_only_gpu(), f, takes / f, pages / f, pages * 4096.0 / 1048576.0 / f, excl / f, (long long)st_excl_reads.load(),
         ex.size(), total / 1048576.0, top.c_str());
}
bool memory_all_reserved(uint64_t guest, uint64_t size) {
#ifdef _WIN32
    if (!size || guest >= (1ull << 40) || guest + size > host->guest_size()) return false;
    const uint8_t* p = host->guest_base() + guest;
    const uint8_t* const end = p + size;
    // (HOS_GUEST_PAGES, host/hos_guestpages.c: a dropped granule is reserved to VirtualQuery but holds bytes, given back on
    // the next touch: not "all reserved". An exe without the export: nothing)
    using Dropped = int (*)(const void*, size_t);
    static const Dropped dropped = (Dropped)(void*)GetProcAddress(GetModuleHandleA(nullptr), "hos_gp_dropped_in");
    if (dropped && dropped(p, size)) return false;
    while (p < end) {
        MEMORY_BASIC_INFORMATION mi{};
        if (VirtualQuery(p, &mi, sizeof mi) != sizeof mi || mi.State != MEM_RESERVE) return false;
        p = (const uint8_t*)mi.BaseAddress + mi.RegionSize;
    }
    return true;
#else
    (void)guest, (void)size;
    return false;   // (linux: the window is one mapping, decommitted pages read as zero but are not told apart)
#endif
}
/// (HL_UBO_CHECK) the texture storage the captures take, as of now (sorted, merged)
std::vector<std::pair<uint64_t, uint64_t>> memory_tex_capture_ranges() {
    std::lock_guard lk(cap_ranges_mu);
    return tex_cap_ranges;
}
/// (frame overlap, vk_replay.cpp: counters) the mirror syncs' dirty pages so far that came from the capture taken last, and
/// those copied from guest memory as it was at the replay (mirror_sync)
void memory_sync_sources(int64_t* capture, int64_t* live) { *capture = st_sync_cap_pages, *live = st_sync_live_pages; }
/// (the same) The capture the next memory_take_capture takes, looked at and left: its pages (written since the capture before
/// it) and how many pools they lie in (pages in no imported pool count as one more)
void memory_capture_peek(int64_t* pages, int64_t* pools) {
    static thread_local std::vector<uint32_t> pg;
    pg.clear();
    {
        std::lock_guard lk(cap_mu);
        if (!captures.empty()) pg = captures.front()->pages;
    }
    *pages = (int64_t)pg.size(), *pools = 0;
    if (pg.empty()) return;
    std::lock_guard lk(pools_mu);
    static thread_local std::vector<uint64_t> seen;
    seen.clear();
    uint64_t lo = 1, hi = 0;
    for (uint32_t page : pg) {
        const uint64_t g = (uint64_t)page << kPageShift;
        if (g >= lo && g < hi) continue;
        uint64_t key = 0;
        lo = 1, hi = 0;
        auto it = imported.upper_bound(g);
        if (it != imported.begin() && (--it, g >= it->second.lo && g < it->second.hi)) key = lo = it->second.lo, hi = it->second.hi;
        if (std::find(seen.begin(), seen.end(), key) == seen.end()) seen.push_back(key);
    }
    *pools = (int64_t)seen.size();
}
} // namespace mkwvk

// (a second export, resolved by host/rx_backend.c if present: a game pool made, with its NVNmemoryPoolFlags (outside
// MkwVkRenderer, whose header (vk_api.h) is in the shader caches' stamp)
#ifdef _WIN32
extern "C" __declspec(dllexport) void mkwvk_pool_flags(uint64_t gpu, uint64_t size, uint32_t flags) {
#else
extern "C" __attribute__((visibility("default"))) void mkwvk_pool_flags(uint64_t gpu, uint64_t size, uint32_t flags) {
#endif
    mkwvk::memory_pool_flags(gpu, size, flags);
}
