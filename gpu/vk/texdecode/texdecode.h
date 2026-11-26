// CPU texture decoders for the Vulkan back end: block-linear (GOB) swizzle and unswizzle, and the ASTC LDR decoder. This
// header, the argument and span checks and size functions in block_linear.cpp and astc.cpp, and the tests are this repository's
// own; the decoding core of astc.cpp and block_linear.cpp is ported (MIT): PROVENANCE.md and LICENSE-MIT beside this file say
// where it came from, what was changed and under what licence.
//
// No dependency on suyu, Eden or boost, none on the host or the renderer: spans in, spans out, no threads and no allocation.
// A caller that wants threads splits the work itself (block-linear: at multiples of block_row_pixels rows, which is how
// vk_texture.cpp already splits an unswizzle; ASTC: at any block row, passing the strip's height), as the calls today do.
#pragma once
#include <cstdint>
#include <span>

namespace mkwvk::texdecode {

// ---- Block-linear (the Switch 2 GOB layout; see block_linear.cpp) ----
// A level is `width` blocks by `height` blocks by `depth` slices of `bpb` bytes (bpb 1 to 16, the widest any NVN format has; a
// compressed format's width and height are in blocks), stored as 64-byte by 8-row GOBs in blocks of 2^tile_h GOBs high by
// 2^tile_d deep (both <= 5): the arguments of suyu's UnswizzleTexture / SwizzleTexture with stride_alignment 0. The linear side is tight:
// rows of width * bpb bytes, slice after slice. Width, height and depth have no limit but their u32 type: the sizes are
// computed in 64 bits with checked products, and a level whose storage cannot be counted in 64 bits is refused like any other
// bad argument (what the spans can hold is what limits a real level).

/// Bytes the whole level takes in swizzled storage, padded to whole blocks (0 for a bad argument): what level_geo calls
/// `swizzled`
uint64_t block_linear_size(uint32_t bpb, uint32_t width, uint32_t height, uint32_t depth, uint32_t tile_h, uint32_t tile_d);
/// Bytes of a row of blocks (8 << tile_h rows of texels, one slice): where the next strip of rows starts in swizzled storage
uint64_t block_linear_row_bytes(uint32_t bpb, uint32_t width, uint32_t tile_h, uint32_t tile_d);

/// swizzled -> linear. false, and nothing written, when an argument is out of range, `linear` is shorter than
/// width * height * depth * bpb, or `swizzled` ends before the last GOB a texel is in (it may end before the padding after
/// that: the storage's end need not be padded). Reads only the GOBs the texels are in.
bool deswizzle(std::span<uint8_t> linear, std::span<const uint8_t> swizzled, uint32_t bpb, uint32_t width, uint32_t height,
               uint32_t depth, uint32_t tile_h, uint32_t tile_d);
/// linear -> swizzled, the same arguments. Writes only the bytes of texels: the padding between them is left as it was.
bool swizzle(std::span<uint8_t> swizzled, std::span<const uint8_t> linear, uint32_t bpb, uint32_t width, uint32_t height,
             uint32_t depth, uint32_t tile_h, uint32_t tile_d);

// ---- ASTC LDR (see astc.cpp) ----
// 8-bit RGBA output equal to ARM astcenc's 8-bit decode (the linear and the sRGB mode: `srgb` only changes how endpoints
// widen to 16 bits; the output is not converted). Footprints 4x4 .. 12x12 (the 14 ASTC 2D ones; any bw, bh in 4..12 decodes).
// An illegal block, and an HDR void extent, decode to the caller's error colour (default: opaque magenta, which is what astcenc
// writes); an HDR endpoint mode to magenta for its partition, whatever the error colour. All three are what astcenc reports as an
// error block (ASTCENC_ERR_BAD_BLOCK), and `false` / the count below say so. The error colour is the caller's choice because an
// all-zero block is an illegal one: a caller that decodes storage the game has not written yet may want transparent black.

/// One RGBA texel, in memory order
struct Rgba8 {
    uint8_t r, g, b, a;
};
inline constexpr Rgba8 kErrorMagenta{0xFF, 0x00, 0xFF, 0xFF};

/// One 16-byte block into bw * bh RGBA texels, rows of bw texels. false: an error block as above (out holds what astcenc
/// gives, with `error_color` for the whole block when it is an illegal one) or a footprint outside 4..12 (nothing written).
bool astc_decode_block(const uint8_t block[16], uint32_t bw, uint32_t bh, bool srgb, uint8_t* out,
                       Rgba8 error_color = kErrorMagenta);

/// A level: `blocks` is ceil(width / bw) * ceil(height / bh) * depth 16-byte blocks, row-major, slice after slice (the
/// depth slices are separate 2D images, as in suyu's DecompressSerial); `rgba` is width * height * depth * 4 bytes, the
/// edge blocks clipped. Returns the number of error blocks, or -1 and nothing written for a bad argument (footprint, span too
/// short, a level too large to count in 64 bits).
int64_t astc_decode(std::span<const uint8_t> blocks, uint32_t width, uint32_t height, uint32_t depth, uint32_t bw,
                    uint32_t bh, bool srgb, std::span<uint8_t> rgba, Rgba8 error_color = kErrorMagenta);

} // namespace mkwvk::texdecode
