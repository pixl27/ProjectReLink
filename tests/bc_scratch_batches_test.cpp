// The two production loops that fill the BC3 scratch, run with a spy in place of Vulkan (tests/test_bc_scratch_layers.py
// cuts them out of gpu/vk/vk_texture.cpp into the marks below): texture_copy_into_bc() whole, and the lines of
// upload_astc_gpu() that form a run. The scratch has bc_array_layers() layers: no run may be larger, each must start at the
// scratch's first layer, and a request larger than the scratch must come out whole, in order, as several runs.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

typedef void* VkCommandBuffer;
typedef void* VkImage;
enum { VK_IMAGE_ASPECT_COLOR_BIT = 1, VK_IMAGE_LAYOUT_GENERAL = 1 };
struct VkImageSubresourceLayers { uint32_t aspectMask, mipLevel, baseArrayLayer, layerCount; };
struct VkOffset3D { int32_t x, y, z; };
struct VkExtent3D { uint32_t width, height, depth; };
struct VkImageCopy {
    VkImageSubresourceLayers srcSubresource;
    VkOffset3D srcOffset;
    VkImageSubresourceLayers dstSubresource;
    VkOffset3D dstOffset;
    VkExtent3D extent;
};
// (main's texture_copy_into_bc asks, for a BC source, whether its format can be blitted: every one can here)
typedef int VkFormat;
enum { VK_FORMAT_FEATURE_BLIT_SRC_BIT = 0x400 };
struct VkFormatProperties { uint32_t linearTilingFeatures, optimalTilingFeatures, bufferFeatures; };
static struct { void* physical = nullptr; } dev;
static void vkGetPhysicalDeviceFormatProperties(void*, VkFormat, VkFormatProperties* p) { p->optimalTilingFeatures = VK_FORMAT_FEATURE_BLIT_SRC_BIT; }
// (and blits a BC source into the scratch, decompressed: no BC source in these runs)
enum { VK_FILTER_NEAREST = 0 };
struct VkImageBlit {
    VkImageSubresourceLayers srcSubresource;
    VkOffset3D srcOffsets[2];
    VkImageSubresourceLayers dstSubresource;
    VkOffset3D dstOffsets[2];
};
static void vkCmdBlitImage(VkCommandBuffer, VkImage, int, VkImage, int, uint32_t, const VkImageBlit*, int) {
    std::fputs("FAIL a blit: no BC source here", stderr);
    std::exit(1);
}
struct Texture {
    VkFormat format = 0;
    bool bc = false, astc = false, bc_resized = false;
    uint32_t width = 0, height = 0, skip = 0;
    uint64_t gpu = 0;
    VkImage image = nullptr;
    struct { uint32_t width = 0, height = 0; } d;
};

constexpr uint32_t kBcDim = 2048;
static uint32_t capacity = 1;
uint32_t bc_array_layers() { return capacity; }
static int dummy;
static VkImage bc_scratch = &dummy;
bool bc_ready() { return true; }
bool bc_scratch_layout(VkCommandBuffer, bool) { return true; }
static unsigned barriers;
void full_barrier(VkCommandBuffer) { ++barriers; }
template <class... A> void logf(const char*, A...) {}

struct Run { uint32_t src_layer, n, dst_layer, scratch_first; bool to_scratch, resized, encoded; };
static std::vector<Run> runs;
void vkCmdCopyImage(VkCommandBuffer, VkImage, int, VkImage dst, int, uint32_t, const VkImageCopy* cp) {
    runs.push_back({cp->srcSubresource.baseArrayLayer, cp->srcSubresource.layerCount, 0, cp->dstSubresource.baseArrayLayer,
                    dst == bc_scratch && cp->dstSubresource.layerCount == cp->srcSubresource.layerCount, false, false});
}
static void encoded(uint32_t n, uint32_t layer, bool resized) {
    if (runs.empty() || runs.back().encoded || runs.back().n != n) {
        std::fprintf(stderr, "FAIL an encode of %u layers with no copy of as many before it\n", n);
        std::exit(1);
    }
    runs.back().dst_layer = layer, runs.back().resized = resized, runs.back().encoded = true;
}
void bc_encode_into(VkCommandBuffer, uint32_t, uint32_t, uint32_t n, Texture&, uint32_t, uint32_t layer, int, int) { encoded(n, layer, false); }
void bc_encode_resized(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t n, Texture&, uint32_t, uint32_t layer) {
    encoded(n, layer, true);
}

/* COPY */

/// upload_astc_gpu()'s runs over the layers it stages (dirty ones, in order): (index of the run's first, its count)
static std::vector<std::pair<size_t, uint32_t>> upload_runs(const std::vector<uint32_t>& layers) {
    std::vector<std::pair<size_t, uint32_t>> out;
    for (size_t k0 = 0; k0 < layers.size();) {
        /* RUN */
        out.push_back({k0, n});
        k0 = k1;
    }
    return out;
}

static int failures;
static void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

/// A copy of `layers` layers from source layer 3 into destination layer 10: the runs it made, checked against the scratch
static bool copy_ok(uint32_t cap, uint32_t layers, bool resized, std::vector<uint32_t> want) {
    capacity = cap;
    runs.clear();
    Texture src, dst;
    src.astc = true, src.image = &dummy;
    dst.bc = true, dst.bc_resized = resized, dst.width = dst.d.width = 512, dst.height = dst.d.height = 256;
    if (!texture_copy_into_bc(nullptr, &src, 0, 3, 0, 0, &dst, 0, 10, 0, 0, 512, 256, layers)) return false;
    uint32_t done = 0;
    size_t i = 0;
    for (const Run& r : runs) {
        if (!r.to_scratch || !r.encoded || r.scratch_first != 0 || r.n == 0 || r.n > cap || r.resized != resized) return false;
        if (r.src_layer != 3 + done || r.dst_layer != 10 + done) return false;   // (whole and in order: no layer left out or twice)
        if (i >= want.size() || want[i++] != r.n) return false;
        done += r.n;
    }
    return done == layers && i == want.size();
}

// The actual mip batch with a spy for its decoder, encoder and copy commands.
struct AstcPush {};
struct AstcPiece { uint32_t level, iw, ih; uint64_t from; AstcPush pc; uint32_t gx, gy; };
struct Staging { int buffer; uint64_t offset; };
struct VkDescriptorBufferInfo { int handle; uint64_t offset, size; };
struct VkDescriptorImageInfo { void* sampler; int view, layout; };
struct VkWriteDescriptorSet {
    int kind, dstBinding = 0, descriptorCount = 0, descriptorType = 0;
    const VkDescriptorImageInfo* pImageInfo = nullptr;
    const VkDescriptorBufferInfo* pBufferInfo = nullptr;
};
struct VkBufferImageCopy {
    uint64_t bufferOffset;
    uint32_t rowLength, imageHeight;
    VkImageSubresourceLayers imageSubresource;
    VkOffset3D imageOffset;
    VkExtent3D imageExtent;
};
struct BcPush { uint32_t blocks_x, blocks_y, width, height, bc3, out_offset; };
enum {
    astc_pipe = 1, bc_pipe, bc7_pipe, astc_layout, bc_layout, bc_out,
    VK_PIPELINE_BIND_POINT_COMPUTE, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
};
constexpr auto VK_NULL_HANDLE = nullptr;
constexpr uint64_t VK_WHOLE_SIZE = ~uint64_t{};
constexpr uint32_t kBcBatchLayers = 2;
constexpr int bc_layer_views[] = {11, 12};
static std::vector<int> pipelines;
static std::vector<BcPush> encodes;
static std::vector<std::vector<VkBufferImageCopy>> copies;
uint32_t bc_mode(VkFormat format) { return (uint32_t)format; }
void vkCmdBindPipeline(VkCommandBuffer, int, int pipeline) { pipelines.push_back(pipeline); }
void vkCmdPushDescriptorSetKHR(VkCommandBuffer, int, int, int, int, const VkWriteDescriptorSet*) {}
void vkCmdPushConstants(VkCommandBuffer, int layout, int, uint32_t, size_t, const void* data) {
    if (layout == bc_layout) encodes.push_back(*static_cast<const BcPush*>(data));
}
void vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t) {}
void descriptors_disturbed() {}
void vkCmdCopyBufferToImage(VkCommandBuffer, int, VkImage, int, uint32_t n, const VkBufferImageCopy* regions) {
    copies.emplace_back(regions, regions + n);
}

/* MIPS */

static bool mips_ok(uint32_t mode) {
    Texture tex;
    tex.format = (VkFormat)mode;
    const std::vector<AstcPiece> pieces{
        {0, 9, 7, 0, {}, 1, 1}, {1, 4, 3, 512, {}, 1, 1}, {2, 2, 1, 1024, {}, 1, 1},
    };
    const uint32_t second = 6u * (mode == 1 ? 8u : 16u);
    const int encoder = mode == 7 ? bc7_pipe : bc_pipe;
    barriers = 0;
    pipelines.clear();
    encodes.clear();
    copies.clear();
    astc_bc_batched(nullptr, tex, pieces, 1536, {42, 0});
    return barriers == 6 && pipelines == std::vector<int>{astc_pipe, encoder, astc_pipe, encoder} &&
           encodes.size() == 3 && encodes[0].bc3 == (mode != 1) && encodes[0].out_offset == 0 &&
           encodes[1].out_offset == (mode == 7 ? second : second / 8) && encodes[2].out_offset == 0 &&
           copies.size() == 2 && copies[0].size() == 2 && copies[1].size() == 1 &&
           copies[0][0].bufferOffset == 0 && copies[0][1].bufferOffset == second && copies[1][0].bufferOffset == 0 &&
           copies[0][0].imageSubresource.mipLevel == 0 && copies[0][1].imageSubresource.mipLevel == 1 &&
           copies[1][0].imageSubresource.mipLevel == 2;
}

int main() {
    check(copy_ok(1, 9, false, {1, 1, 1, 1, 1, 1, 1, 1, 1}), "a copy of 9 layers into a one-layer scratch: 9 runs of one, in order");
    check(copy_ok(4, 9, false, {4, 4, 1}), "into a 4-layer scratch: 4, 4 and a last run of 1");
    check(copy_ok(8, 9, false, {8, 1}), "into an 8-layer scratch: 8 and 1, as before the change");
    check(copy_ok(8, 8, false, {8}), "a copy the scratch holds whole is one run");
    check(copy_ok(1, 1, false, {1}), "a single layer is one run");
    check(copy_ok(1, 3, true, {1, 1, 1}) && copy_ok(4, 6, true, {4, 2}), "a resized BC3 texture: the same runs, through the second scratch");
    {
        capacity = 1;
        runs.clear();
        Texture src, dst;
        src.astc = true, dst.bc = true, dst.width = dst.d.width = 512, dst.height = dst.d.height = 256;
        const bool odd = texture_copy_into_bc(nullptr, &src, 0, 0, 0, 0, &dst, 0, 0, 2, 0, 512, 256, 4);
        const bool wide = texture_copy_into_bc(nullptr, &src, 0, 0, 0, 0, &dst, 0, 0, 0, 0, kBcDim + 4, 256, 4);
        check(!odd && !wide && runs.empty(), "a copy the scratch cannot take (off the block grid, wider than it) writes nothing into it");
    }
    using Runs = std::vector<std::pair<size_t, uint32_t>>;
    const std::vector<uint32_t> dirty{0, 1, 2, 5, 6, 9};   // (an array's dirty layers: runs end at the gaps)
    capacity = 1;
    check(upload_runs(dirty) == Runs{{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}, "an upload into a one-layer scratch: one layer a run");
    capacity = 2;
    check(upload_runs(dirty) == Runs{{0, 2}, {2, 1}, {3, 2}, {5, 1}}, "into a 2-layer scratch: runs of at most 2, ended by a gap");
    capacity = 8;
    check(upload_runs(dirty) == Runs{{0, 3}, {3, 2}, {5, 1}}, "into an 8-layer scratch: the gaps alone end the runs");
    capacity = 4;
    check(upload_runs({7, 8, 9, 10, 11, 12, 13, 14, 15}) == Runs{{0, 4}, {4, 4}, {8, 1}}, "9 layers in a row into a 4-layer scratch: 4, 4 and 1");
    check(mips_ok(1), "BC1: pairs of mips go to distinct output ranges with BC1 offsets");
    check(mips_ok(3), "BC3: pairs of mips go to distinct output ranges with BC3 offsets");
    check(mips_ok(7), "BC7: pairs of mips go to distinct output ranges with BC7 offsets");
    if (failures) return 1;
    std::printf("PASS: no run is larger than the scratch, and a larger request comes out whole\n");
    return 0;
}
