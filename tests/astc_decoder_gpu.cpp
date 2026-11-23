// SPDX-License-Identifier: GPL-3.0-or-later
// Game-free Vulkan compute fixture for gpu/vk/host_shaders/astc_decoder.comp.
// Compares the real compiled ASTC_DECODER_COMP_SPV output against Hookshot's CPU ASTC
// reference (Tegra::Texture::ASTC::DecompressSerial) across all 14 ASTC block sizes,
// odd image dimensions, linear and Switch 2 native tiling, 1-4 partitions, dual-plane
// (all 4 secondary channels), void-extent LDR, and error blocks.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "host_shaders/astc_decoder_comp_spv.h"
#include "video_core/textures/astc.h"

#define CHECK(x)                                                                                                      \
    do {                                                                                                              \
        if (!(x)) {                                                                                                   \
            std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x);                                                       \
            std::exit(1);                                                                                             \
        }                                                                                                             \
    } while (0)

namespace {

struct BlockDim {
    uint32_t w;
    uint32_t h;
};

constexpr std::array<BlockDim, 14> kAstcBlockDims = {{
    {4, 4},   {5, 4},   {5, 5},   {6, 5},   {6, 6},   {8, 5},   {8, 6},
    {8, 8},   {10, 5},  {10, 6},  {10, 8},  {10, 10}, {12, 10}, {12, 12},
}};

struct AstcPush {
    uint32_t block_dims[2];
    uint32_t layer_stride;
    uint32_t block_size;
    uint32_t x_shift;
    uint32_t block_height;
    uint32_t block_height_mask;
    uint32_t native_tiling;
    uint32_t blocks_x;
    uint32_t blocks_y;
};

struct Block128 {
    uint8_t b[16]{};
};

void set_bits(Block128& blk, uint32_t bit_pos, uint32_t count, uint32_t val) {
    for (uint32_t i = 0; i < count; ++i) {
        if ((val >> i) & 1u) {
            const uint32_t p = bit_pos + i;
            if (p < 128u) blk.b[p >> 3u] |= static_cast<uint8_t>(1u << (p & 7u));
        }
    }
}

void set_weight_bits_reversed(Block128& blk, uint32_t weight_bits, uint64_t seed) {
    uint64_t s = seed ^ 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0; i < weight_bits; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const uint32_t bit = static_cast<uint32_t>((s >> 63u) & 1u);
        const uint32_t p = 127u - i;
        if (bit) blk.b[p >> 3u] |= static_cast<uint8_t>(1u << (p & 7u));
    }
}

// Construct a valid LDR void-extent block with constant RGBA16 color.
Block128 make_void_extent(uint16_t r, uint16_t g, uint16_t b, uint16_t a) {
    Block128 blk{};
    // Bits [0..8] = 0x1FC, bit 9 = 0 (LDR), bits [10..11] = 3 (reserved 1s).
    set_bits(blk, 0, 12, 0xDFCu);
    // All-ones void-extent coordinates [12..63] = no extent bounds.
    for (uint32_t p = 12; p < 64; ++p) set_bits(blk, p, 1, 1u);
    blk.b[8] = static_cast<uint8_t>(r & 0xFFu);
    blk.b[9] = static_cast<uint8_t>(r >> 8u);
    blk.b[10] = static_cast<uint8_t>(g & 0xFFu);
    blk.b[11] = static_cast<uint8_t>(g >> 8u);
    blk.b[12] = static_cast<uint8_t>(b & 0xFFu);
    blk.b[13] = static_cast<uint8_t>(b >> 8u);
    blk.b[14] = static_cast<uint8_t>(a & 0xFFu);
    blk.b[15] = static_cast<uint8_t>(a >> 8u);
    return blk;
}

// Construct an invalid block mode (error block -> FillError zeros).
Block128 make_error_block(uint32_t variant) {
    Block128 blk{};
    if (variant == 0u) {
        // Low 4 bits == 0 -> reserved error block mode.
        blk.b[0] = 0x00u;
    } else {
        // 4 partitions + dual-plane -> invalid per ASTC spec.
        // Layout with D=1 (bit 10), valid 2x2 grid (mode bits: A=0, B=0, R=2 -> 0x424),
        // partitions=3 (4 partitions) at bits [11..12].
        set_bits(blk, 0, 11, 0x424u);
        set_bits(blk, 11, 2, 3u);
    }
    return blk;
}

// Encode a valid ASTC block mode using Table C.2.8 layout 0:
// bits [1:0]=00, [3:2]=B (weight_h-2), [4]=R0, [6:5]=A (weight_w-4), [8:7]=R2:R1, [9]=H, [10]=D.
// Or layout 2: bits [1:0]=01, [3:2]=B (weight_h-2), [4]=R0, [6:5]=A (weight_w-4)?
// Let's use layout 4 (bits [1:0]=10, [3:2]=A -> w=A+2 in [2..5], [6:5]=B -> h=B+2 in [2..5],
// [4]=R0, [8:7]=R2:R1, [9]=H, [10]=D): valid for EVERY block size >= 4x4!
uint32_t encode_block_mode(uint32_t wg_w, uint32_t wg_h, uint32_t range_code, bool high_prec, bool dual_plane) {
    // range_code in [2..7] (R0=bit0, R1=bit1, R2=bit2)
    // Layout with bits[1:0] != 00 and bits[3:2] == 00:
    // Actually let's use bits[1:0] = 01 (R0 in bit 4, R2:R1 in bits 3:2? Let's match Table C.2.8 directly):
    // In astc.cpp / astc_decoder.comp:
    //   when (mode & 3) == 1 (layout 2):
    //     B = (mode >> 7) & 3 -> h = B + 2 (2..5)
    //     A = (mode >> 5) & 3 -> w = A + 4 (4..7) -- wait, h can be 4, w can be 4!
    //   when (mode & 3) == 2 (layout 3):
    //     A = (mode >> 5) & 3 -> w = A + 2 (2..5)
    //     B = (mode >> 7) & 3 -> h = B + 2 (2..5)  (actually h = B + 2, w = A + 2? Let's check exact bits!)
    // Safer: use bits[1:0] = 00, bits[3:2] = 01 (R=bits[4],bits[3:2] -> R in 2..7):
    // Wait: let's use exact layout 0 from ASTC spec Table C.2.8:
    //   mode bits: [10]=D, [9]=H, [8:7]=B, [6:5]=A, [4]=R0, [3:2]=00, [1:0]=R2:R1 (with R2:R1 != 00):
    //   w = B + 4, h = A + 2. For 4x4: B=0 -> w=4, A=2 -> h=4! For 2x2..4x4, let's check:
    //   Layout 1 (bits[3:2] == 01, bits[1:0] = R2:R1 != 00): w = B + 8, h = A + 2.
    //   Layout 2 (bits[3:2] == 10, bits[1:0] = R2:R1 != 00): w = A + 2, h = B + 8.
    //   Layout 3 (bits[3:2] == 11, bits[8]==0, bits[1:0] = R2:R1 != 00): w = A + 2, h = B + 6 (B = bit 7).
    //   Layout 4 (bits[3:2] == 11, bits[8]==1, bits[1:0] = R2:R1 != 00): w = B + 2 (B = bit 7), h = A + 2!
    // Layout 4 gives w in {2, 3} and h in {2, 3, 4, 5}, which fits EVERY ASTC block size (4x4..12x12)!
    const uint32_t r0 = range_code & 1u;
    const uint32_t r21 = (range_code >> 1u) & 3u; // must be 1, 2, or 3
    if (wg_w >= 4u && wg_w <= 7u && wg_h >= 2u && wg_h <= 5u) {
        // Layout 0: bits[3:2]=00, bits[1:0]=R2:R1, [4]=R0, [6:5]=A(h-2), [8:7]=B(w-4), [9]=H, [10]=D
        const uint32_t A = wg_h - 2u;
        const uint32_t B = wg_w - 4u;
        return r21 | (0u << 2u) | (r0 << 4u) | (A << 5u) | (B << 7u) | (high_prec ? (1u << 9u) : 0u) |
               (dual_plane ? (1u << 10u) : 0u);
    }
    // Layout 4: bits[3:2]=11, bit[8]=1, bits[1:0]=R2:R1, [4]=R0, [6:5]=A(h-2), [7]=B(w-2), [9]=H, [10]=D
    const uint32_t A = std::min(wg_h - 2u, 3u);
    const uint32_t B = std::min(wg_w - 2u, 1u);
    return r21 | (3u << 2u) | (r0 << 4u) | (A << 5u) | (B << 7u) | (1u << 8u) |
           (high_prec ? (1u << 9u) : 0u) | (dual_plane ? (1u << 10u) : 0u);
}

// Construct a valid LDR ASTC block exercising partitions (1..4), dual-plane (with ccs 0..3),
// CEM modes (0, 4, 6, 8, 10, 12, 13), and both direct-grid and scaled-grid weight grids.
Block128 make_ldr_block(uint32_t bw, uint32_t bh, uint32_t num_parts, uint32_t part_seed, bool dual_plane,
                        uint32_t sec_chan, uint32_t cem, bool direct_grid, uint64_t seed) {
    Block128 blk{};
    // Choose a weight grid that keeps weight bits comfortably in [24..64] so plenty of bits remain for color data.
    uint32_t wg_w = 3u;
    uint32_t wg_h = 3u;
    uint32_t range_code = 4u; // max_weight = 5 (trit + 1 bit) or range_code=3 (max_weight=3, 2 bits)
    bool high_prec = false;
    if (direct_grid && bw <= 7u && bh <= 5u && !dual_plane && num_parts == 1u) {
        wg_w = bw;
        wg_h = bh;
        range_code = 3u; // max_weight=2 (trit encoding: ceil(N*8/5) bits -> 4x4=26, 5x5=40, 6x5=48 bits)
    } else if (dual_plane) {
        wg_w = 3u;
        wg_h = 3u;
        range_code = 4u; // max_weight=3 (2 bits/weight): 2*9*2 = 36 weight bits
    } else if (num_parts >= 3u) {
        wg_w = 3u;
        wg_h = 3u;
        range_code = 6u; // max_weight=5 (trit+1 bit): ceil(9*8/5)+9 = 24 weight bits (ASTC min is 24)
    } else {
        wg_w = (bw >= 4u) ? 4u : 3u;
        wg_h = (bh >= 4u) ? 4u : 3u;
        range_code = (seed & 1u) ? 5u : 6u; // quints / trits
    }

    const uint32_t mode = encode_block_mode(wg_w, wg_h, range_code, high_prec, dual_plane);
    set_bits(blk, 0, 11, mode);
    set_bits(blk, 11, 2, num_parts - 1u);

    uint32_t pos = 13u;
    if (num_parts == 1u) {
        set_bits(blk, pos, 4, cem & 0xFu);
        pos += 4u;
    } else {
        set_bits(blk, pos, 10, part_seed & 0x3FFu);
        pos += 10u;
        // Multi-partition uniform CEM: bits[1:0] = 00, bits[5:2] = cem (all partitions share cem).
        set_bits(blk, pos, 2, 0u);
        set_bits(blk, pos + 2u, 4, cem & 0xFu);
        pos += 6u;
    }

    // Fill color endpoint data bits [pos .. 127-weight_bits-2] deterministically.
    uint64_t s = seed * 0xBF58476D1CE4E5B9ull + 1u;
    for (uint32_t p = pos; p < 80u; ++p) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        set_bits(blk, p, 1, static_cast<uint32_t>((s >> 63u) & 1u));
    }

    // Set weight bits from MSB downward (40 bits is safe; extra bits past weight_bits are color/CCS bits).
    set_weight_bits_reversed(blk, 44u, seed ^ 0xA5A5A5A5u);

    if (dual_plane) {
        // For dual-plane with 3x3 grid, range_code=4 (max_weight=3, 2 bits/weight): weight_bits = 36.
        // CCS is the 2 bits immediately below the weight bits: bits [128 - 36 - 2 .. 128 - 36 - 1] = [90..91].
        blk.b[90 >> 3] &= static_cast<uint8_t>(~(3u << (90 & 7)));
        set_bits(blk, 90, 2, sec_chan & 3u);
    }
    return blk;
}

uint32_t swizzle_offset(uint32_t bx_bytes, uint32_t by, bool native_tiling) {
    uint32_t swizzle = ((bx_bytes & 32u) << 3u) |
                       ((by & 6u) << 5u) |
                       ((bx_bytes & 16u) << 1u) |
                       ((by & 1u) << 4u) |
                       (bx_bytes & 15u);
    if (native_tiling) {
        const uint32_t diff = ((swizzle >> 5u) ^ (swizzle >> 6u)) & 1u;
        swizzle ^= (diff << 5u) | (diff << 6u);
    }
    return swizzle;
}

struct TestImage {
    uint32_t bw, bh;
    uint32_t width, height;
    uint32_t blocks_x, blocks_y;
    bool native_tiling;
    std::vector<uint8_t> linear_astc;
    std::vector<uint8_t> swizzled_astc;
    AstcPush push{};
};

TestImage build_test_image(uint32_t bw, uint32_t bh, bool native_tiling, uint32_t case_idx) {
    TestImage img{};
    img.bw = bw;
    img.bh = bh;
    img.native_tiling = native_tiling;
    // Odd, non-block-multiple dimensions to exercise partial edge blocks on right & bottom.
    img.width = bw * 4u + 3u;
    img.height = bh * 3u + 1u;
    img.blocks_x = (img.width + bw - 1u) / bw;   // 5
    img.blocks_y = (img.height + bh - 1u) / bh;  // 4

    constexpr std::array<uint32_t, 7> kLdrCems = {0u, 4u, 6u, 8u, 10u, 12u, 13u};
    img.linear_astc.resize(static_cast<size_t>(img.blocks_x) * img.blocks_y * 16u);

    for (uint32_t by = 0; by < img.blocks_y; ++by) {
        for (uint32_t bx = 0; bx < img.blocks_x; ++bx) {
            const uint32_t idx = by * img.blocks_x + bx;
            const uint64_t seed = (static_cast<uint64_t>(case_idx) << 16u) | (idx * 131u + 17u);
            Block128 blk{};
            switch (idx) {
                case 0:
                    // Replicated-byte void-extent: both (v >> 8) and round(v / 257) equal exact byte.
                    blk = make_void_extent(0x1212u, 0x5656u, 0x9A9Au, 0xDEDEu);
                    break;
                case 1:
                    blk = make_void_extent(0xFFFFu, 0x0000u, 0x8080u, 0xFFFFu);
                    break;
                case 2:
                    blk = make_error_block(0u);
                    break;
                case 3:
                    blk = make_error_block(1u);
                    break;
                case 4: // 1-part direct weight grid (when bw<=7 && bh<=5), CEM 12 (direct RGBA)
                    blk = make_ldr_block(bw, bh, 1u, 0u, false, 0u, 12u, true, seed);
                    break;
                case 5: // 1-part scaled grid, CEM 8 (direct RGB)
                    blk = make_ldr_block(bw, bh, 1u, 0u, false, 0u, 8u, false, seed);
                    break;
                case 6: // 1-part dual-plane, ccs=0 (R)
                    blk = make_ldr_block(bw, bh, 1u, 0u, true, 0u, 12u, false, seed);
                    break;
                case 7: // 1-part dual-plane, ccs=1 (G)
                    blk = make_ldr_block(bw, bh, 1u, 0u, true, 1u, 12u, false, seed);
                    break;
                case 8: // 1-part dual-plane, ccs=2 (B)
                    blk = make_ldr_block(bw, bh, 1u, 0u, true, 2u, 8u, false, seed);
                    break;
                case 9: // 1-part dual-plane, ccs=3 (A)
                    blk = make_ldr_block(bw, bh, 1u, 0u, true, 3u, 12u, false, seed);
                    break;
                case 10: // 2-partition, CEM 8
                    blk = make_ldr_block(bw, bh, 2u, 137u + case_idx, false, 0u, 8u, false, seed);
                    break;
                case 11: // 2-partition dual-plane, ccs=3
                    blk = make_ldr_block(bw, bh, 2u, 421u + case_idx, true, 3u, 12u, false, seed);
                    break;
                case 12: // 3-partition, CEM 6 (RGB base+scale)
                    blk = make_ldr_block(bw, bh, 3u, 733u + case_idx, false, 0u, 6u, false, seed);
                    break;
                case 13: // 3-partition dual-plane, ccs=1
                    blk = make_ldr_block(bw, bh, 3u, 511u + case_idx, true, 1u, 8u, false, seed);
                    break;
                case 14: // 4-partition, CEM 0 (luminance)
                    blk = make_ldr_block(bw, bh, 4u, 299u + case_idx, false, 0u, 0u, false, seed);
                    break;
                case 15: // 4-partition, CEM 4 (luminance+alpha)
                    blk = make_ldr_block(bw, bh, 4u, 887u + case_idx, false, 0u, 4u, false, seed);
                    break;
                default: {
                    const uint32_t cem = kLdrCems[idx % kLdrCems.size()];
                    const uint32_t parts = 1u + (idx % 3u);
                    blk = make_ldr_block(bw, bh, parts, (idx * 97u + case_idx * 13u) & 0x3FFu, false, 0u, cem,
                                         false, seed);
                    break;
                }
            }
            std::memcpy(img.linear_astc.data() + static_cast<size_t>(idx) * 16u, blk.b, 16u);
        }
    }

    // Swizzle linear blocks into Tegra block-linear layout matching astc_decoder.comp's main().
    const uint32_t block_height = 0u; // 1 GOB tall (8 block rows per gob-row)
    const uint32_t block_height_mask = (1u << block_height) - 1u;
    const uint32_t x_shift = 9u + block_height; // GOB_SIZE_SHIFT (9) + block_height
    const uint32_t gobs_x = ((img.blocks_x * 16u) + 63u) / 64u;
    const uint32_t gobs_y = (img.blocks_y + 7u) / 8u;
    const uint32_t block_size = gobs_x * (512u << block_height);
    const uint32_t layer_stride = gobs_y * block_size;

    img.swizzled_astc.assign(layer_stride, 0u);
    for (uint32_t by = 0; by < img.blocks_y; ++by) {
        const uint32_t gob_y = by >> 3u;
        for (uint32_t bx = 0; bx < img.blocks_x; ++bx) {
            const uint32_t bx_bytes = bx << 4u;
            const uint32_t sw = swizzle_offset(bx_bytes, by, native_tiling);
            const uint32_t off = (gob_y >> block_height) * block_size +
                                 ((gob_y & block_height_mask) << 9u) +
                                 ((bx_bytes >> 6u) << x_shift) + sw;
            CHECK(off + 16u <= img.swizzled_astc.size());
            std::memcpy(img.swizzled_astc.data() + off,
                        img.linear_astc.data() + static_cast<size_t>(by * img.blocks_x + bx) * 16u, 16u);
        }
    }

    img.push.block_dims[0] = bw;
    img.push.block_dims[1] = bh;
    img.push.layer_stride = layer_stride;
    img.push.block_size = block_size;
    img.push.x_shift = x_shift;
    img.push.block_height = block_height;
    img.push.block_height_mask = block_height_mask;
    img.push.native_tiling = native_tiling ? 1u : 0u;
    img.push.blocks_x = img.blocks_x;
    img.push.blocks_y = img.blocks_y;
    return img;
}

int run_cpu_only() {
    // Sanity-check compiled SPIR-V header embedded in gpu/vk/host_shaders/astc_decoder_comp_spv.h.
    CHECK(sizeof(ASTC_DECODER_COMP_SPV) >= 20u);
    CHECK((sizeof(ASTC_DECODER_COMP_SPV) % 4u) == 0u);
    CHECK(ASTC_DECODER_COMP_SPV[0] == 0x07230203u);

    uint64_t total_pixels = 0;
    uint64_t non_zero_pixels = 0;
    for (uint32_t i = 0; i < kAstcBlockDims.size(); ++i) {
        const auto [bw, bh] = kAstcBlockDims[i];
        TestImage img = build_test_image(bw, bh, (i & 1u) != 0u, i);
        std::vector<uint8_t> cpu_rgba(static_cast<size_t>(img.width) * img.height * 4u, 0u);
        Tegra::Texture::ASTC::DecompressSerial(img.linear_astc, img.width, img.height, 1u, bw, bh, cpu_rgba);
        // Block (0,0) is void-extent (0x12, 0x56, 0x9A, 0xDE); verify CPU reference decoded it.
        CHECK(cpu_rgba[0] == 0x12u && cpu_rgba[1] == 0x56u && cpu_rgba[2] == 0x9Au && cpu_rgba[3] == 0xDEu);
        // Verify every block decoded as expected: error blocks (idx 2, 3) are all zeros; all valid blocks (idx >= 4)
        // produce non-zero output so no synthetic test case can silently collapse into FillError.
        for (uint32_t by = 0; by < img.blocks_y; ++by) {
            for (uint32_t bx = 0; bx < img.blocks_x; ++bx) {
                const uint32_t bidx = by * img.blocks_x + bx;
                uint32_t nz = 0;
                for (uint32_t dy = 0; dy < bh && by * bh + dy < img.height; ++dy) {
                    for (uint32_t dx = 0; dx < bw && bx * bw + dx < img.width; ++dx) {
                        const size_t o = (static_cast<size_t>(by * bh + dy) * img.width + (bx * bw + dx)) * 4u;
                        if (cpu_rgba[o] | cpu_rgba[o + 1] | cpu_rgba[o + 2] | cpu_rgba[o + 3]) ++nz;
                    }
                }
                if (bidx == 2u || bidx == 3u) CHECK(nz == 0u);
                if (bidx >= 4u) CHECK(nz > 0u);
            }
        }
        total_pixels += static_cast<uint64_t>(img.width) * img.height;
        for (size_t p = 0; p < cpu_rgba.size(); p += 4u) {
            if (cpu_rgba[p] | cpu_rgba[p + 1] | cpu_rgba[p + 2] | cpu_rgba[p + 3]) ++non_zero_pixels;
        }
    }
    CHECK(non_zero_pixels > total_pixels / 2u);
    std::printf("PASS: ASTC fixture compiles, SPIR-V valid (%zu bytes), CPU reference decoded %llu pixels across 14 block sizes (GPU execution not requested)\n",
                sizeof(ASTC_DECODER_COMP_SPV), static_cast<unsigned long long>(total_pixels));
    return 0;
}

#define VK_APIS(X)                                                                                                    \
    X(vkCreateInstance)                                                                                               \
    X(vkDestroyInstance)                                                                                              \
    X(vkEnumerateInstanceLayerProperties)                                                                             \
    X(vkEnumerateInstanceExtensionProperties)                                                                         \
    X(vkEnumeratePhysicalDevices)                                                                                     \
    X(vkGetPhysicalDeviceProperties)                                                                                  \
    X(vkGetPhysicalDeviceQueueFamilyProperties)                                                                       \
    X(vkGetPhysicalDeviceMemoryProperties)                                                                            \
    X(vkCreateDevice)                                                                                                 \
    X(vkDestroyDevice)                                                                                                \
    X(vkGetDeviceQueue)                                                                                               \
    X(vkCreateShaderModule)                                                                                           \
    X(vkDestroyShaderModule)                                                                                          \
    X(vkCreateDescriptorSetLayout)                                                                                    \
    X(vkDestroyDescriptorSetLayout)                                                                                   \
    X(vkCreatePipelineLayout)                                                                                         \
    X(vkDestroyPipelineLayout)                                                                                        \
    X(vkCreateComputePipelines)                                                                                       \
    X(vkDestroyPipeline)                                                                                              \
    X(vkCreateDescriptorPool)                                                                                         \
    X(vkDestroyDescriptorPool)                                                                                        \
    X(vkAllocateDescriptorSets)                                                                                       \
    X(vkUpdateDescriptorSets)                                                                                         \
    X(vkCreateBuffer)                                                                                                 \
    X(vkDestroyBuffer)                                                                                                \
    X(vkGetBufferMemoryRequirements)                                                                                  \
    X(vkBindBufferMemory)                                                                                             \
    X(vkCreateImage)                                                                                                  \
    X(vkDestroyImage)                                                                                                 \
    X(vkGetImageMemoryRequirements)                                                                                   \
    X(vkBindImageMemory)                                                                                              \
    X(vkCreateImageView)                                                                                              \
    X(vkDestroyImageView)                                                                                             \
    X(vkAllocateMemory)                                                                                               \
    X(vkFreeMemory)                                                                                                   \
    X(vkMapMemory)                                                                                                    \
    X(vkUnmapMemory)                                                                                                  \
    X(vkCreateCommandPool)                                                                                            \
    X(vkDestroyCommandPool)                                                                                           \
    X(vkAllocateCommandBuffers)                                                                                       \
    X(vkResetCommandBuffer)                                                                                           \
    X(vkBeginCommandBuffer)                                                                                           \
    X(vkEndCommandBuffer)                                                                                             \
    X(vkCmdBindPipeline)                                                                                              \
    X(vkCmdBindDescriptorSets)                                                                                        \
    X(vkCmdPushConstants)                                                                                             \
    X(vkCmdDispatch)                                                                                                  \
    X(vkCmdPipelineBarrier)                                                                                           \
    X(vkCmdCopyImageToBuffer)                                                                                         \
    X(vkCreateFence)                                                                                                  \
    X(vkDestroyFence)                                                                                                 \
    X(vkResetFences)                                                                                                  \
    X(vkQueueSubmit)                                                                                                  \
    X(vkWaitForFences)

#define DECLARE_VK(n) PFN_##n n = nullptr;
VK_APIS(DECLARE_VK)
#undef DECLARE_VK

std::atomic<int> validation_errors{0};

std::string lower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties& props, uint32_t bits,
                          VkMemoryPropertyFlags required) {
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & required) == required) return i;
    }
    return UINT32_MAX;
}

int skip_gpu(const char* reason) {
    std::printf("SKIP: %s\n", reason);
    return 0;
}

int run_gpu(const char* requested_device) {
#ifdef _WIN32
    HMODULE loader = LoadLibraryW(L"vulkan-1.dll");
    auto symbol = [&](const char* name) { return loader ? reinterpret_cast<void*>(GetProcAddress(loader, name)) : nullptr; };
    auto close_loader = [&]() { if (loader) FreeLibrary(loader); };
#else
    void* loader = dlopen("libvulkan.so.1", RTLD_NOW);
    auto symbol = [&](const char* name) { return loader ? dlsym(loader, name) : nullptr; };
    auto close_loader = [&]() { if (loader) dlclose(loader); };
#endif
    if (!loader) return skip_gpu("Vulkan loader unavailable");

    bool api_ready = true;
#define LOAD_VK(n) n = reinterpret_cast<PFN_##n>(symbol(#n)); api_ready = api_ready && (n != nullptr);
    VK_APIS(LOAD_VK)
#undef LOAD_VK
    auto gip = reinterpret_cast<PFN_vkGetInstanceProcAddr>(symbol("vkGetInstanceProcAddr"));
    if (!api_ready || !gip) {
        close_loader();
        return skip_gpu("required Vulkan entry points unavailable");
    }

    uint32_t nl = 0;
    CHECK(vkEnumerateInstanceLayerProperties(&nl, nullptr) == VK_SUCCESS);
    std::vector<VkLayerProperties> layers_list(nl);
    CHECK(vkEnumerateInstanceLayerProperties(&nl, layers_list.data()) == VK_SUCCESS);

    uint32_t ne = 0;
    CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr) == VK_SUCCESS);
    std::vector<VkExtensionProperties> extensions(ne);
    CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &ne, extensions.data()) == VK_SUCCESS);

    bool validation = false;
    bool debug_utils = false;
    for (const auto& l : layers_list) {
        if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) validation = true;
    }
    for (const auto& e : extensions) {
        if (std::strcmp(e.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) debug_utils = true;
    }

    const char* layer_name = "VK_LAYER_KHRONOS_validation";
    const char* ext_name = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkDebugUtilsMessengerCreateInfoEXT db{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    db.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    db.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    db.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                            const VkDebugUtilsMessengerCallbackDataEXT* data, void*) -> VkBool32 {
        if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            validation_errors.fetch_add(1, std::memory_order_relaxed);
            std::fprintf(stderr, "validation: %s\n", data->pMessage);
        }
        return VK_FALSE;
    };

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (validation) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer_name;
    }
    if (validation && debug_utils) {
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &ext_name;
        ici.pNext = &db;
    }

    VkInstance instance{};
    const VkResult ir = vkCreateInstance(&ici, nullptr, &instance);
    if (ir == VK_ERROR_INCOMPATIBLE_DRIVER || ir == VK_ERROR_INITIALIZATION_FAILED) {
        close_loader();
        return skip_gpu("Vulkan instance or compatible driver unavailable");
    }
    CHECK(ir == VK_SUCCESS);

    VkDebugUtilsMessengerEXT messenger{};
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger = nullptr;
    if (validation && debug_utils) {
        auto create_messenger =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(gip(instance, "vkCreateDebugUtilsMessengerEXT"));
        destroy_messenger =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(gip(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (create_messenger && destroy_messenger) {
            CHECK(create_messenger(instance, &db, nullptr, &messenger) == VK_SUCCESS);
        }
    }

    uint32_t nd = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &nd, nullptr) == VK_SUCCESS);
    std::vector<VkPhysicalDevice> gpus(nd);
    if (nd) CHECK(vkEnumeratePhysicalDevices(instance, &nd, gpus.data()) == VK_SUCCESS);

    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties chosen_props{};
    uint32_t queue_family = UINT32_MAX;
    const std::string want = requested_device ? lower_str(requested_device) : std::string{};

    for (VkPhysicalDevice cand : gpus) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(cand, &props);
        if (!want.empty() && lower_str(props.deviceName).find(want) == std::string::npos) continue;
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(cand, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(cand, &nq, qf.data());
        uint32_t chosen_q = UINT32_MAX;
        for (uint32_t q = 0; q < nq; ++q) {
            if ((qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qf[q].queueFlags & VK_QUEUE_TRANSFER_BIT)) {
                chosen_q = q;
                break;
            }
        }
        if (chosen_q == UINT32_MAX) continue;
        if (physical == VK_NULL_HANDLE ||
            (want.empty() && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
             chosen_props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
            physical = cand;
            chosen_props = props;
            queue_family = chosen_q;
        }
    }

    if (physical == VK_NULL_HANDLE) {
        if (messenger && destroy_messenger) destroy_messenger(instance, messenger, nullptr);
        vkDestroyInstance(instance, nullptr);
        close_loader();
        if (requested_device) {
            std::fprintf(stderr, "FAIL: requested Vulkan device '%s' not found\n", requested_device);
            return 1;
        }
        return skip_gpu("no Vulkan compute physical device found");
    }

    std::printf("Device: %s\n", chosen_props.deviceName);
    std::printf("validation: %s\n",
                validation ? (messenger ? "enabled" : "enabled; debug utils unavailable") : "unavailable");

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkDevice device{};
    CHECK(vkCreateDevice(physical, &dci, nullptr, &device) == VK_SUCCESS);
    VkQueue queue{};
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &mem_props);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = sizeof(ASTC_DECODER_COMP_SPV);
    smci.pCode = ASTC_DECODER_COMP_SPV;
    VkShaderModule shader{};
    CHECK(vkCreateShaderModule(device, &smci, nullptr, &shader) == VK_SUCCESS);

    const VkDescriptorSetLayoutBinding bindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2;
    dslci.pBindings = bindings;
    VkDescriptorSetLayout set_layout{};
    CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &set_layout) == VK_SUCCESS);

    const VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AstcPush)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pipe_layout{};
    CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipe_layout) == VK_SUCCESS);

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = "main";
    cpci.layout = pipe_layout;
    VkPipeline pipeline{};
    CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) == VK_SUCCESS);
    vkDestroyShaderModule(device, shader, nullptr);

    const VkDescriptorPoolSize pool_sizes[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = pool_sizes;
    VkDescriptorPool desc_pool{};
    CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool) == VK_SUCCESS);

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout;
    VkDescriptorSet desc_set{};
    CHECK(vkAllocateDescriptorSets(device, &dsai, &desc_set) == VK_SUCCESS);

    VkCommandPoolCreateInfo cmdpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cmdpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmdpci.queueFamilyIndex = queue_family;
    VkCommandPool cmd_pool{};
    CHECK(vkCreateCommandPool(device, &cmdpci, nullptr, &cmd_pool) == VK_SUCCESS);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd{};
    CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd) == VK_SUCCESS);

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence{};
    CHECK(vkCreateFence(device, &fci, nullptr, &fence) == VK_SUCCESS);

    uint64_t total_cases = 0;
    uint64_t total_pixels = 0;
    uint64_t total_mismatches = 0;
    uint32_t overall_max_ch_diff = 0;

    for (uint32_t d = 0; d < kAstcBlockDims.size(); ++d) {
        const auto [bw, bh] = kAstcBlockDims[d];
        for (int nt = 0; nt <= 1; ++nt) {
            TestImage img = build_test_image(bw, bh, nt != 0, d * 2u + static_cast<uint32_t>(nt));
            std::vector<uint8_t> cpu_rgba(static_cast<size_t>(img.width) * img.height * 4u, 0u);
            Tegra::Texture::ASTC::DecompressSerial(img.linear_astc, img.width, img.height, 1u, bw, bh, cpu_rgba);

            // Input SSBO (host-visible coherent).
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = img.swizzled_astc.size();
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer in_buf{};
            CHECK(vkCreateBuffer(device, &bci, nullptr, &in_buf) == VK_SUCCESS);
            VkMemoryRequirements in_req{};
            vkGetBufferMemoryRequirements(device, in_buf, &in_req);
            const uint32_t in_mt = find_memory_type(
                mem_props, in_req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            CHECK(in_mt != UINT32_MAX);
            VkMemoryAllocateInfo in_mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            in_mai.allocationSize = in_req.size;
            in_mai.memoryTypeIndex = in_mt;
            VkDeviceMemory in_mem{};
            CHECK(vkAllocateMemory(device, &in_mai, nullptr, &in_mem) == VK_SUCCESS);
            CHECK(vkBindBufferMemory(device, in_buf, in_mem, 0) == VK_SUCCESS);
            void* in_ptr = nullptr;
            CHECK(vkMapMemory(device, in_mem, 0, bci.size, 0, &in_ptr) == VK_SUCCESS);
            std::memcpy(in_ptr, img.swizzled_astc.data(), img.swizzled_astc.size());
            vkUnmapMemory(device, in_mem);

            // Output storage image (RGBA8_UNORM, 2D array with 1 layer).
            VkImageCreateInfo imci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imci.imageType = VK_IMAGE_TYPE_2D;
            imci.format = VK_FORMAT_R8G8B8A8_UNORM;
            imci.extent = {img.width, img.height, 1u};
            imci.mipLevels = 1u;
            imci.arrayLayers = 1u;
            imci.samples = VK_SAMPLE_COUNT_1_BIT;
            imci.tiling = VK_IMAGE_TILING_OPTIMAL;
            imci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            imci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImage out_img{};
            CHECK(vkCreateImage(device, &imci, nullptr, &out_img) == VK_SUCCESS);
            VkMemoryRequirements img_req{};
            vkGetImageMemoryRequirements(device, out_img, &img_req);
            uint32_t img_mt = find_memory_type(mem_props, img_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (img_mt == UINT32_MAX) img_mt = find_memory_type(mem_props, img_req.memoryTypeBits, 0);
            CHECK(img_mt != UINT32_MAX);
            VkMemoryAllocateInfo img_mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            img_mai.allocationSize = img_req.size;
            img_mai.memoryTypeIndex = img_mt;
            VkDeviceMemory img_mem{};
            CHECK(vkAllocateMemory(device, &img_mai, nullptr, &img_mem) == VK_SUCCESS);
            CHECK(vkBindImageMemory(device, out_img, img_mem, 0) == VK_SUCCESS);

            VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            ivci.image = out_img;
            ivci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
            ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView out_view{};
            CHECK(vkCreateImageView(device, &ivci, nullptr, &out_view) == VK_SUCCESS);

            // Readback staging buffer.
            VkBufferCreateInfo rbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            rbci.size = cpu_rgba.size();
            rbci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            rbci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer rb_buf{};
            CHECK(vkCreateBuffer(device, &rbci, nullptr, &rb_buf) == VK_SUCCESS);
            VkMemoryRequirements rb_req{};
            vkGetBufferMemoryRequirements(device, rb_buf, &rb_req);
            const uint32_t rb_mt = find_memory_type(
                mem_props, rb_req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            CHECK(rb_mt != UINT32_MAX);
            VkMemoryAllocateInfo rb_mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            rb_mai.allocationSize = rb_req.size;
            rb_mai.memoryTypeIndex = rb_mt;
            VkDeviceMemory rb_mem{};
            CHECK(vkAllocateMemory(device, &rb_mai, nullptr, &rb_mem) == VK_SUCCESS);
            CHECK(vkBindBufferMemory(device, rb_buf, rb_mem, 0) == VK_SUCCESS);

            VkDescriptorBufferInfo dbi{in_buf, 0, bci.size};
            VkDescriptorImageInfo dii{VK_NULL_HANDLE, out_view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet writes[2]{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = desc_set;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &dbi;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = desc_set;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].pImageInfo = &dii;
            vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

            CHECK(vkResetCommandBuffer(cmd, 0) == VK_SUCCESS);
            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK(vkBeginCommandBuffer(cmd, &cbbi) == VK_SUCCESS);

            VkImageMemoryBarrier to_gen{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_gen.srcAccessMask = 0;
            to_gen.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            to_gen.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_gen.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            to_gen.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_gen.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_gen.image = out_img;
            to_gen.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_gen);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &desc_set, 0, nullptr);
            vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AstcPush), &img.push);
            vkCmdDispatch(cmd, (img.blocks_x + 7u) / 8u, (img.blocks_y + 7u) / 8u, 1u);

            VkImageMemoryBarrier to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_src.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_src.image = out_img;
            to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_src);

            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {img.width, img.height, 1u};
            vkCmdCopyImageToBuffer(cmd, out_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb_buf, 1, &region);
            CHECK(vkEndCommandBuffer(cmd) == VK_SUCCESS);

            CHECK(vkResetFences(device, 1, &fence) == VK_SUCCESS);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            CHECK(vkQueueSubmit(queue, 1, &si, fence) == VK_SUCCESS);
            CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull) == VK_SUCCESS);

            void* rb_ptr = nullptr;
            CHECK(vkMapMemory(device, rb_mem, 0, rbci.size, 0, &rb_ptr) == VK_SUCCESS);
            const auto* gpu_rgba = static_cast<const uint8_t*>(rb_ptr);
            uint64_t case_mismatches = 0;
            for (uint32_t y = 0; y < img.height; ++y) {
                for (uint32_t x = 0; x < img.width; ++x) {
                    const size_t off = (static_cast<size_t>(y) * img.width + x) * 4u;
                    const uint32_t bx = x / bw;
                    const uint32_t by = y / bh;
                    const uint32_t bidx = by * img.blocks_x + bx;
                    // Void-extent (replicated-byte) and error blocks must match bit-exact (tolerance 0).
                    // Interpolated LDR blocks allow 1 LSB because suyu's CPU astc.cpp rounds
                    // (255.0 * C / 65536.0 + 0.5) whereas Vulkan R8G8B8A8_UNORM rounds (C / 65535.0f).
                    const uint32_t tol = (bidx <= 3u) ? 0u : 1u;
                    bool bad = false;
                    for (int c = 0; c < 4; ++c) {
                        const uint32_t diff = static_cast<uint32_t>(
                            std::abs(static_cast<int>(gpu_rgba[off + c]) - static_cast<int>(cpu_rgba[off + c])));
                        if (diff > overall_max_ch_diff) overall_max_ch_diff = diff;
                        if (diff > tol) bad = true;
                    }
                    if (bad) {
                        if (case_mismatches < 4u) {
                            std::fprintf(stderr,
                                         "MISMATCH %ux%u nt=%d (%u,%u) blk=(%u,%u) tol=%u: gpu=(%u,%u,%u,%u) cpu=(%u,%u,%u,%u)\n",
                                         bw, bh, nt, x, y, bx, by, tol, gpu_rgba[off], gpu_rgba[off + 1],
                                         gpu_rgba[off + 2], gpu_rgba[off + 3], cpu_rgba[off], cpu_rgba[off + 1],
                                         cpu_rgba[off + 2], cpu_rgba[off + 3]);
                        }
                        ++case_mismatches;
                    }
                }
            }
            vkUnmapMemory(device, rb_mem);

            vkDestroyBuffer(device, rb_buf, nullptr);
            vkFreeMemory(device, rb_mem, nullptr);
            vkDestroyImageView(device, out_view, nullptr);
            vkDestroyImage(device, out_img, nullptr);
            vkFreeMemory(device, img_mem, nullptr);
            vkDestroyBuffer(device, in_buf, nullptr);
            vkFreeMemory(device, in_mem, nullptr);

            ++total_cases;
            total_pixels += static_cast<uint64_t>(img.width) * img.height;
            total_mismatches += case_mismatches;
        }
    }

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyDescriptorPool(device, desc_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipe_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    vkDestroyDevice(device, nullptr);
    if (messenger && destroy_messenger) destroy_messenger(instance, messenger, nullptr);
    vkDestroyInstance(instance, nullptr);
    close_loader();

    if (validation_errors.load() != 0) {
        std::fprintf(stderr, "FAIL: %d Vulkan validation error(s)\n", validation_errors.load());
        return 1;
    }
    if (total_mismatches != 0) {
        std::fprintf(stderr, "FAIL: %llu pixel mismatch(es) across %llu cases (%llu pixels)\n",
                     static_cast<unsigned long long>(total_mismatches),
                     static_cast<unsigned long long>(total_cases),
                     static_cast<unsigned long long>(total_pixels));
        return 1;
    }
    std::printf("PASS: %llu GPU readback cases (%llu pixels across 14 ASTC block sizes, linear+native tiling) matched CPU reference (0 tol void-extent/error, <=1 LSB UNORM8 rounding, max_diff=%u)\n",
                static_cast<unsigned long long>(total_cases), static_cast<unsigned long long>(total_pixels),
                overall_max_ch_diff);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    bool gpu = false;
    const char* requested_device = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu") == 0) {
            gpu = true;
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            gpu = true;
            requested_device = argv[++i];
        } else if (std::strcmp(argv[i], "--cpu-only") == 0) {
            gpu = false;
        } else {
            std::fprintf(stderr, "usage: %s [--cpu-only | --gpu [--device NAME_SUBSTRING]]\n", argv[0]);
            return 2;
        }
    }
    return gpu ? run_gpu(requested_device) : run_cpu_only();
}
