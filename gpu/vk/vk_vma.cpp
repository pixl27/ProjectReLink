// Vulkan Memory Allocator (v3.3.0, from Eglantine's CPM cache: nx2/build.py adds its include path): images and
// device buffers are sub-allocated from large blocks (NVIDIA allows ~4096 allocations, and the imported pools
// already take about 2400 of them). Its Vulkan functions come from vkGetInstanceProcAddr / vkGetDeviceProcAddr.
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vk_internal.h"
#include <vk_mem_alloc.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mkwvk {

VmaAllocator vma;

bool vma_init(PFN_vkGetInstanceProcAddr gipa) {
    VmaVulkanFunctions f{};
    f.vkGetInstanceProcAddr = gipa;
    f.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo ci{};
    ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
               (dev.memory_budget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : 0);
    ci.physicalDevice = dev.physical;
    ci.device = dev.device;
    ci.instance = dev.instance;
    ci.vulkanApiVersion = VK_API_VERSION_1_3;
    ci.pVulkanFunctions = &f;
    ci.preferredLargeHeapBlockSize = 64ull << 20;   // (64 MB blocks: Eglantine found 256 MB ones wasted VRAM)
    return vmaCreateAllocator(&ci, &vma) == VK_SUCCESS;
}

// Images come from a pool of 128 MB blocks: on their own, VMA gave most of them dedicated memory (the driver prefers it
// for render targets, and anything over half a default block), a vkAllocateMemory and a kernel call for each, and
// the dynamic resolution makes a new set of render targets every few frames. Images too large for a block, or of a
// memory type the pool's is not, are allocated as before.
VmaPool image_pool = VK_NULL_HANDLE;
bool image_pool_tried = false;
// (RX_VK_IMAGE_BLOCK_MB=<n>, 16 and up: the block's size. VMA keeps one block empty for the next image, 128 MB of VRAM that a 64 MB block
// would halve, at twice the memory objects: the driver allows ~4096, the imported pools take ~2400)
const uint64_t kImageBlock = (getenv("RX_VK_IMAGE_BLOCK_MB") ? (uint64_t)std::max(16, atoi(getenv("RX_VK_IMAGE_BLOCK_MB"))) : 128u) << 20;
thread_local int image_pool_hint = 0;
VmaPool asset_pool = VK_NULL_HANDLE;   // (relink, RX_VK_ASSET_POOL=1: the game's compressed textures, apart)

VmaPool images_pool() {
    if (image_pool_tried) return image_pool;
    image_pool_tried = true;
    VkImageCreateInfo rep{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};   // (a representative image: the pool's memory type)
    rep.imageType = VK_IMAGE_TYPE_2D;
    rep.format = VK_FORMAT_R8G8B8A8_UNORM;
    rep.extent = {256, 256, 1};
    rep.mipLevels = rep.arrayLayers = 1;
    rep.samples = VK_SAMPLE_COUNT_1_BIT;
    rep.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t type = 0;
    if (vmaFindMemoryTypeIndexForImageInfo(vma, &rep, &ai, &type) != VK_SUCCESS) return VK_NULL_HANDLE;
    VmaPoolCreateInfo pi{};
    pi.memoryTypeIndex = type;
    // (relink, RX_VK_IMAGE_BLOCK_MB=<n>, experiment: smaller blocks empty out whole more often, and VMA gives an empty block
    // back to the driver; in 128 MB blocks a block half used by a few small textures kept its whole size counted)
    const uint64_t block = getenv("RX_VK_IMAGE_BLOCK_MB") ? (uint64_t)std::max(8, atoi(getenv("RX_VK_IMAGE_BLOCK_MB"))) << 20 : kImageBlock;
    pi.blockSize = block;
    if (vmaCreatePool(vma, &pi, &image_pool) != VK_SUCCESS) image_pool = VK_NULL_HANDLE;
    if (image_pool && getenv("RX_VK_ASSET_POOL") && atoi(getenv("RX_VK_ASSET_POOL")) != 0 &&
        vmaCreatePool(vma, &pi, &asset_pool) != VK_SUCCESS)
        asset_pool = VK_NULL_HANDLE;
    if (asset_pool) logf("vk: the game's compressed textures from a pool of their own (RX_VK_ASSET_POOL)");
    logf("vk: images from a pool of %llu MB blocks (memory type %u)%s", (unsigned long long)(block >> 20), type,
         image_pool ? "" : ": not made");
    return image_pool;
}

// (relink, diagnostic) the images' bytes by kind: render targets (attachments) and the rest, for the heaps' log line
std::mutex img_kind_mu;
std::unordered_map<VmaAllocation, std::pair<int, uint64_t>> img_kind;
uint64_t img_kind_bytes[2];
void note_image(VmaAllocation a, const VkImageCreateInfo& ci) {
    if (!a) return;
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(vma, a, &info);
    const int k = (ci.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) ? 1 : 0;
    std::lock_guard lk(img_kind_mu);
    img_kind[a] = {k, info.size};
    img_kind_bytes[k] += info.size;
}
bool create_image_raw(const VkImageCreateInfo& ci, VkImage& image, VmaAllocation& allocation, bool dedicated);
bool create_image(const VkImageCreateInfo& ci, VkImage& image, VmaAllocation& allocation, bool dedicated) {
    if (!create_image_raw(ci, image, allocation, dedicated)) return false;
    note_image(allocation, ci);
    return true;
}
bool create_image_raw(const VkImageCreateInfo& ci, VkImage& image, VmaAllocation& allocation, bool dedicated) {
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    // (in VRAM or not at all: an image VMA puts in system memory when VRAM is full is read over PCIe every frame,
    // and takes the RAM the game and everything else on the machine need)
    ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (dev.memory_budget) ai.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
    if (dedicated) {   // (a memory object of its own, which freeing gives back whole: a pool's block is kept by VMA when it empties)
        VmaAllocationCreateInfo dai = ai;
        dai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        if (vmaCreateImage(vma, &ci, &dai, &image, &allocation, nullptr) == VK_SUCCESS) return true;
    }
    static const bool no_pool = getenv("RX_VK_NO_IMAGE_POOL") != nullptr;
    // (relink, RX_VK_DEDICATED_MB=<n>, experiment, off by default) images of n MB and more in memory of their own: in the 128 MB
    // blocks a 47-64 MB texture left a hole when it went, which the driver still counted (hs179 outdoors: 7.5 GB used of 7.26
    // allowed, 2 GB of it free inside the blocks), so the texture budget shrank and textures were evicted and made again
    static const uint64_t dedicated_mb = (uint64_t)(getenv("RX_VK_DEDICATED_MB") ? std::max(0, atoi(getenv("RX_VK_DEDICATED_MB"))) : 0) << 20;
    if (dedicated_mb) {
        VkMemoryRequirements req{};
        VkImage probe = VK_NULL_HANDLE;
        if (vkCreateImage(dev.device, &ci, nullptr, &probe) == VK_SUCCESS) {
            vkGetImageMemoryRequirements(dev.device, probe, &req);
            vkDestroyImage(dev.device, probe, nullptr);
        }
        if (req.size >= dedicated_mb) {
            VmaAllocationCreateInfo dai = ai;
            dai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
            if (vmaCreateImage(vma, &ci, &dai, &image, &allocation, nullptr) == VK_SUCCESS) return true;
        }
    }
    VmaPool chosen = no_pool ? VK_NULL_HANDLE : images_pool();
    if (chosen && asset_pool && image_pool_hint == 1) chosen = asset_pool;
    if (VmaPool pool = chosen) {
        VmaAllocationCreateInfo pai = ai;
        pai.pool = pool;
        pai.usage = VMA_MEMORY_USAGE_UNKNOWN;
        if (vmaCreateImage(vma, &ci, &pai, &image, &allocation, nullptr) == VK_SUCCESS) return true;
    }
    if (vmaCreateImage(vma, &ci, &ai, &image, &allocation, nullptr) == VK_SUCCESS) return true;
    // (relink) VRAM is full (a 4 GB card at a high resolution, #96): an image left unmade is a black screen, so it goes
    // to system memory instead, slow but drawn
    if (getenv("RX_VK_IMAGES_VRAM_ONLY")) return false;
    VmaAllocationCreateInfo sai{};
    sai.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(vma, &ci, &sai, &image, &allocation, nullptr) != VK_SUCCESS) return false;
    static std::atomic<int> told{0};
    if (told++ < 4) logf("vk: VRAM full: a %ux%u image (format %d) in system memory", ci.extent.width, ci.extent.height, (int)ci.format);
    return true;
}
// (relink) The image pool's defragmentation. Its 128 MB blocks kept ~1.2-1.9 GB free between long-lived images scattered by
// the load peaks and the dynamic resolution's targets (hs195/hs196), and the driver counts the blocks: 7.5 GB used of 7.25
// allowed outdoors, so the texture budget squeezed and ~10-17k textures were evicted and made again in 300 s. A pass moves a
// few images into the free space of other blocks (vk_texture.cpp makes each again there and copies it on the GPU) and VMA
// frees the blocks that emptied. While a pass is open its sources are not destroyed (VMA's rule): destroy_image defers them
VmaDefragmentationContext defrag_ctx = nullptr;
VmaDefragmentationPassMoveInfo defrag_info{};
bool defrag_open = false;
std::mutex defrag_mu;
std::vector<VmaAllocation> defrag_busy;   // (the open pass's sources, sorted)
std::vector<std::pair<VkImage, VmaAllocation>> defrag_deferred;
void destroy_image_now(VkImage image, VmaAllocation allocation);
bool image_pool_stats(uint64_t* block_bytes, uint64_t* alloc_bytes, uint32_t* blocks) {
    if (!image_pool) return false;
    VmaStatistics ps{};
    vmaGetPoolStatistics(vma, image_pool, &ps);
    *block_bytes = ps.blockBytes, *alloc_bytes = ps.allocationBytes, *blocks = ps.blockCount;
    return true;
}
bool image_defrag_begin(uint64_t max_bytes, uint32_t max_moves, uint32_t flags) {
    if (defrag_ctx || !image_pool) return false;
    VmaDefragmentationInfo di{};
    di.flags = flags;
    di.pool = image_pool;
    di.maxBytesPerPass = max_bytes;
    di.maxAllocationsPerPass = max_moves;
    return vmaBeginDefragmentation(vma, &di, &defrag_ctx) == VK_SUCCESS;
}
bool image_defrag_pass(std::vector<ImageMove>& moves) {
    moves.clear();
    if (!defrag_ctx || defrag_open) return false;
    defrag_info = {};
    // (defrag_mu over the pass's start too: the reaper below destroys under it, so an image it destroys is either gone before
    // VMA picks the pass's sources or seen as a source and deferred)
    std::lock_guard lk(defrag_mu);
    if (vmaBeginDefragmentationPass(vma, defrag_ctx, &defrag_info) != VK_INCOMPLETE) return false;   // (VK_SUCCESS: none left)
    defrag_busy.clear();
    for (uint32_t i = 0; i < defrag_info.moveCount; i++) {
        moves.push_back({defrag_info.pMoves[i].srcAllocation, defrag_info.pMoves[i].dstTmpAllocation, 0});
        defrag_busy.push_back(defrag_info.pMoves[i].srcAllocation);
    }
    std::sort(defrag_busy.begin(), defrag_busy.end());
    defrag_open = true;
    return true;
}
bool image_defrag_bind(VmaAllocation dst, VkImage image) {
    return vmaBindImageMemory(vma, dst, image) == VK_SUCCESS;
}
uint64_t defrag_moved_bytes = 0;
uint32_t defrag_moved = 0;
bool image_defrag_end_pass(const std::vector<ImageMove>& moves) {
    if (!defrag_open) return false;
    for (uint32_t i = 0; i < defrag_info.moveCount && i < moves.size(); i++) {
        defrag_info.pMoves[i].operation = (VmaDefragmentationMoveOperation)moves[i].op;
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(vma, moves[i].src, &ai);
        if (moves[i].op == 0) defrag_moved_bytes += ai.size, defrag_moved++;
        if (moves[i].op == VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY) {   // (its image destroyed already: VMA frees it)
            std::lock_guard lk(img_kind_mu);
            auto it = img_kind.find(moves[i].src);
            if (it != img_kind.end()) img_kind_bytes[it->second.first] -= it->second.second, img_kind.erase(it);
        }
    }
    VmaStatistics before{};
    vmaGetPoolStatistics(vma, image_pool, &before);
    const auto t0 = std::chrono::steady_clock::now();
    const VkResult r = vmaEndDefragmentationPass(vma, defrag_ctx, &defrag_info);
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<std::pair<VkImage, VmaAllocation>> later;
    {
        std::lock_guard lk(defrag_mu);
        defrag_open = false;
        defrag_busy.clear();
        later.swap(defrag_deferred);
    }
    for (auto& [img, a] : later) destroy_image(img, a);   // (what went while the pass was open: their allocations moved)
    const auto t2 = std::chrono::steady_clock::now();
    const double vma_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(), later_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    if (vma_ms + later_ms > 4.0) {   // (a slow end: VMA's (blocks given back to the driver) or the deferred destructions)
        VmaStatistics after{};
        vmaGetPoolStatistics(vma, image_pool, &after);
        static int logged;
        if (logged++ < 100)
            logf("vk: defrag end pass slow: VMA %.2f ms (blocks %u -> %u), %zu deferred destructions %.2f ms", vma_ms, before.blockCount,
                 after.blockCount, later.size(), later_ms);
    }
    return r == VK_INCOMPLETE;
}
void image_defrag_end(uint64_t* bytes_moved, uint64_t* bytes_freed, uint32_t* moves, uint32_t* blocks_freed) {
    VmaDefragmentationStats st{};
    if (defrag_ctx) vmaEndDefragmentation(vma, defrag_ctx, &st);
    defrag_ctx = nullptr;
    *bytes_moved = defrag_moved_bytes, *moves = defrag_moved;
    *bytes_freed = st.bytesFreed, *blocks_freed = st.deviceMemoryBlocksFreed;
    defrag_moved_bytes = 0, defrag_moved = 0;
}
// (relink) The images are destroyed on a thread of their own (the reaper). On NVIDIA's Windows driver a vmaDestroyImage is a
// kernel call that takes a millisecond or more, and the render thread destroyed the dead textures one after the other as it
// presented: while the camera turned in the Market (hs2591), 2.4 ms of each frame over 25 ms against 0.1 ms of a normal
// one, the largest thing those frames did more of. Every caller destroys an image the GPU is done with, so a later
// destruction changes nothing but when its memory goes back. RX_VK_REAP=0: on the caller's thread, as before
std::mutex reap_mu;
std::condition_variable reap_cv;
std::vector<std::pair<VkImage, VmaAllocation>> reap_queue;
std::atomic<int64_t> reap_pending{0};
void destroy_or_defer(VkImage image, VmaAllocation allocation) {
    std::lock_guard lk(defrag_mu);   // (held through the destruction: no defragmentation pass starts meanwhile)
    if (allocation && defrag_open && std::binary_search(defrag_busy.begin(), defrag_busy.end(), allocation)) {
        defrag_deferred.push_back({image, allocation});   // (a source of the open pass: once it ends)
        return;
    }
    destroy_image_now(image, allocation);
}
bool reap_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_REAP");
        if (e && atoi(e) == 0) return false;
        std::thread([] {
            std::vector<std::pair<VkImage, VmaAllocation>> batch;
            for (;;) {
                {
                    std::unique_lock lk(reap_mu);
                    reap_cv.wait(lk, [] { return !reap_queue.empty(); });
                    batch.swap(reap_queue);
                }
                for (auto& [img, a] : batch) destroy_or_defer(img, a), reap_pending--;
                batch.clear();
            }
        }).detach();
        logf("vk: images destroyed on a thread of their own (RX_VK_REAP=0: on the caller's)");
        return true;
    }();
    return on;
}
void destroy_image(VkImage image, VmaAllocation allocation) {
    if (!image && !allocation) return;
    if (reap_on()) {
        {
            std::lock_guard lk(reap_mu);
            reap_queue.push_back({image, allocation});
        }
        reap_pending++;
        reap_cv.notify_one();
        return;
    }
    destroy_or_defer(image, allocation);
}
void destroy_image_now(VkImage image, VmaAllocation allocation) {
    if (allocation) {
        std::lock_guard lk(img_kind_mu);
        auto it = img_kind.find(allocation);
        if (it != img_kind.end()) img_kind_bytes[it->second.first] -= it->second.second, img_kind.erase(it);
    }
    vmaDestroyImage(vma, image, allocation);
}
uint64_t allocation_size(VmaAllocation allocation) {
    VmaAllocationInfo info{};
    if (allocation) vmaGetAllocationInfo(vma, allocation, &info);
    return info.size;
}
bool create_buffer(uint64_t size, VkBufferUsageFlags usage, bool host_visible, VkBuffer& buffer,
                   VmaAllocation& allocation, void** mapped, bool dedicated) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    if (host_visible) ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    // (a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT gets VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT from VMA:
    // vma_init sets VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT)
    static const bool blocks = getenv("RX_VK_BLOCK_BUFFERS") != nullptr;
    if (dedicated && !blocks) ai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(vma, &bi, &ai, &buffer, &allocation, &info) != VK_SUCCESS) return false;
    if (mapped) *mapped = info.pMappedData;
    return true;
}
/// A buffer in cached host memory, mapped: written on one thread, read back on the CPU too (the captures: the game's
/// threads copy pages into it, the replay reads texture pages out of it and the GPU copies the rest from it)
bool create_buffer_cached(uint64_t size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation,
                          void** mapped, bool dedicated) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (dedicated) ai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;   // (its own memory object: no part of a 64 MB block left unused)
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(vma, &bi, &ai, &buffer, &allocation, &info) != VK_SUCCESS) return false;
    if (mapped) *mapped = info.pMappedData;
    return true;
}
void flush_mapped(VmaAllocation allocation, uint64_t size) {
    if (size) vmaFlushAllocation(vma, allocation, 0, size);
}
void invalidate_mapped(VmaAllocation allocation, uint64_t size) {
    if (size) vmaInvalidateAllocation(vma, allocation, 0, size);
}
void destroy_buffer(VkBuffer buffer, VmaAllocation allocation) {
    vmaDestroyBuffer(vma, buffer, allocation);
}
bool create_device_buffer(uint64_t size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, bool dedicated) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (dev.memory_budget) ai.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
    if (dedicated) ai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    return vmaCreateBuffer(vma, &bi, &ai, &buffer, &allocation, nullptr) == VK_SUCCESS;
}
/// (every heap: what VMA has in blocks and allocations there, what the process uses of it, its budget; one line)
void vma_log_heaps() {
    VmaBudget b[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(vma, b);
    char line[2048];
    int n = snprintf(line, sizeof line, "vk: memory heaps:");
    for (uint32_t h = 0; h < dev.memory.memoryHeapCount && n < 900; h++)
        n += snprintf(line + n, sizeof line - n, " heap %u%s: %u blocks %llu MB, %u allocations %llu MB; used %llu of %llu MB;", h,
                      (dev.memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " (device)" : " (system)",
                      b[h].statistics.blockCount, (unsigned long long)(b[h].statistics.blockBytes >> 20),
                      b[h].statistics.allocationCount, (unsigned long long)(b[h].statistics.allocationBytes >> 20),
                      (unsigned long long)(b[h].usage >> 20), (unsigned long long)(b[h].budget >> 20));
    for (int k = 0; k < 2; k++) {   // (relink: where the free space inside the blocks is: the images' pools, or the rest)
        VmaPool pl = k ? asset_pool : image_pool;
        if (!pl || n >= 1700) continue;
        VmaStatistics ps{};
        vmaGetPoolStatistics(vma, pl, &ps);
        n += snprintf(line + n, sizeof line - n, " %s pool: %u blocks %llu MB, %u allocations %llu MB (%llu MB free in it)", k ? "asset" : "image",
                      ps.blockCount, (unsigned long long)(ps.blockBytes >> 20), ps.allocationCount, (unsigned long long)(ps.allocationBytes >> 20),
                      (unsigned long long)((ps.blockBytes - std::min(ps.blockBytes, ps.allocationBytes)) >> 20));
    }
    if (n < 1900) {
        std::lock_guard lk(img_kind_mu);
        n += snprintf(line + n, sizeof line - n, " images: render targets %llu MB, others %llu MB", (unsigned long long)(img_kind_bytes[1] >> 20),
                      (unsigned long long)(img_kind_bytes[0] >> 20));
    }
    logf("%s", line);
}
bool vram_usage(uint64_t* used, uint64_t* budget, uint64_t* block_free) {
    VmaBudget b[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(vma, b);
    uint64_t u = 0, t = 0, f = 0;
    for (uint32_t h = 0; h < dev.memory.memoryHeapCount; h++)
        if (dev.memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            u += b[h].usage, t += b[h].budget;
            // (the free space in VMA's blocks: the driver counts it as the process's, new allocations take it first)
            f += b[h].statistics.blockBytes - std::min(b[h].statistics.blockBytes, b[h].statistics.allocationBytes);
        }
    *used = u, *budget = t;
    if (block_free) *block_free = f;
    return t != 0;
}

/// The system-memory heaps the GPU reaches: what the process uses there (the imported guest memory too: pinned by the
/// driver) and what the driver allows it, asked now (a new frame index makes VMA ask the driver again); false when unknown
namespace {
std::atomic<uint32_t> budget_frame{0};   // (VMA's frame index: a new one makes it ask the driver for the budgets again)
}
bool vram_usage_now(uint64_t* used, uint64_t* budget) {
    vmaSetCurrentFrameIndex(vma, ++budget_frame);
    return vram_usage(used, budget) && dev.memory_budget;
}
bool system_heap_usage(uint64_t* used, uint64_t* budget) {
    vmaSetCurrentFrameIndex(vma, ++budget_frame);
    VmaBudget b[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(vma, b);
    uint64_t u = 0, t = 0;
    for (uint32_t h = 0; h < dev.memory.memoryHeapCount; h++)
        if (!(dev.memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) u += b[h].usage, t += b[h].budget;
    *used = u, *budget = t;
    return dev.memory_budget && t != 0;
}

} // namespace mkwvk
