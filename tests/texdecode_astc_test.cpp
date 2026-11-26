// Decodes every block of tests/texdecode_astc_vectors.vec with gpu/vk/texdecode/astc.cpp, in both modes, and compares with what
// ARM's astcenc gave (scripts/gen_texdecode_astc_vectors.py): the checksum of the 8-bit RGBA output, and whether it called the
// block an error block. Also checks astc_decode over a whole level against the blocks one at a time (edge clipping, depth).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "texdecode/texdecode.h"

using namespace mkwvk::texdecode;
enum { KINDS = 14 };   // (scripts/texdecode_astc_gen.cpp's Kind: the random kinds 0-8, then the five built by hand)
static const char* kind_name[KINDS] = {"random", "void extent", "illegal void extent", "illegal block", "HDR endpoints", "dual plane",
                                       "4 partitions", "2-3 partitions", "legal", "void extent low == high", "void extent reserved bits",
                                       "void extent all ones", "6x10 / 10x6 weight grid", "weights over 96 bits"};
static uint64_t fnv(const uint8_t* p, size_t n) { uint64_t h = 0xCBF29CE484222325ull; for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001B3ull; return h; }
static uint64_t le64(const uint8_t* p) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = v << 8 | p[i]; return v; }

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: test vectors.bin\n"); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    std::vector<uint8_t> v;
    uint8_t buf[65536];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) v.insert(v.end(), buf, buf + n);
    fclose(f);
    if (v.size() < 12 || memcmp(v.data(), "ASTCVEC1", 8)) { fprintf(stderr, "not a vector file\n"); return 2; }
    uint32_t count = v[8] | v[9] << 8 | v[10] << 16 | (uint32_t)v[11] << 24;
    if (v.size() != 12 + (size_t)count * 36) { fprintf(stderr, "vector file is %zu bytes, %u blocks need %zu\n", v.size(), count, 12 + (size_t)count * 36); return 2; }
    long compared = 0, bad = 0, illegal_expected = 0, by_kind[KINDS] = {}, kind_illegal[KINDS] = {}, footprints[13][13] = {}, grids[2] = {};
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t* r = &v[12 + (size_t)i * 36];
        const unsigned bw = r[0], bh = r[1], kind = r[3];
        if (kind >= KINDS) { fprintf(stderr, "block %u: bad kind\n", i); return 2; }
        by_kind[kind]++;
        footprints[bw][bh]++;
        if (kind == 12) grids[r[4] >> 5 & 1]++;   // (block mode bit 5: 0 is the 6x10 grid, 1 the 10x6 one)
        for (int mode = 0; mode < 2; mode++) {
            uint8_t out[12 * 12 * 4];
            std::memset(out, 0xA5, sizeof out);
            const bool ok = astc_decode_block(r + 4, bw, bh, mode == 1, out);
            const bool want_bad = r[2] >> mode & 1;
            const uint64_t sum = fnv(out, bw * bh * 4), want = le64(r + 20 + 8 * mode);
            compared++;
            illegal_expected += want_bad;
            kind_illegal[kind] += want_bad;
            if (ok == want_bad || sum != want) {
                if (bad++ < 20) printf("MISMATCH block %u (%ux%u, %s, %s): legal %d (astcenc %d), sum %016llx (astcenc %016llx)\n", i, bw, bh,
                                       kind_name[kind], mode ? "sRGB" : "linear", ok, !want_bad, (unsigned long long)sum, (unsigned long long)want);
            }
        }
    }
    // (the vectors have to cover what they claim to: every footprint, and every kind with legal and illegal blocks where it can be)
    int nfp = 0;
    for (int w = 4; w <= 12; w++) for (int h = 4; h <= 12; h++) nfp += footprints[w][h] > 0;
    long problems = bad;
    if (nfp != 14) { printf("FAIL: %d footprints in the vectors, not 14\n", nfp); problems++; }
    for (int k = 1; k < KINDS; k++)
        if (by_kind[k] < (k == 12 ? 48 : 14)) { printf("FAIL: only %ld %s blocks\n", by_kind[k], kind_name[k]); problems++; }
    if (!grids[0] || !grids[1]) { printf("FAIL: the weight grid blocks are not both 6x10 and 10x6 (%ld, %ld)\n", grids[0], grids[1]); problems++; }
    for (int k = 1; k < KINDS; k++) {   // (what astcenc said about each kind: the generator classed them by it)
        const bool must_err = k == 2 || k == 3 || k == 4 || k == 9 || k == 10 || k == 13;
        if (kind_illegal[k] != (must_err ? 2 * by_kind[k] : 0)) { printf("FAIL: %s blocks are not all %s in astcenc's answer\n", kind_name[k], must_err ? "errors" : "legal"); problems++; }
    }
    // A whole level at once: 3 slices of 29 x 13 texels (edge blocks clipped on both axes) in each footprint, from the vector
    // blocks, equals the blocks decoded singly and clipped by hand; the error count is the sum.
    long levels = 0;
    for (unsigned bw = 4; bw <= 12; bw++) for (unsigned bh = 4; bh <= 12; bh++) {
        if (!footprints[bw][bh]) continue;
        const unsigned W = 29, H = 13, D = 3, bx = (W + bw - 1) / bw, by = (H + bh - 1) / bh;
        for (int mode = 0; mode < 2; mode++) {
            std::vector<uint8_t> blocks, want((size_t)W * H * D * 4), got((size_t)W * H * D * 4, 0x5A);
            int64_t want_bad = 0;
            uint32_t at = 0;
            for (unsigned n = 0; n < bx * by * D; n++) {
                while (v[12 + (size_t)at * 36] != bw || v[13 + (size_t)at * 36] != bh) at = (at + 1) % count;
                const uint8_t* r = &v[12 + (size_t)at * 36];
                at = (at + 7) % count;
                blocks.insert(blocks.end(), r + 4, r + 20);
            }
            for (unsigned z = 0; z < D; z++) for (unsigned j = 0; j < by; j++) for (unsigned i = 0; i < bx; i++) {
                uint8_t px[12 * 12 * 4];
                want_bad += !astc_decode_block(&blocks[16 * ((z * by + j) * bx + i)], bw, bh, mode == 1, px);
                for (unsigned y = 0; y < bh && j * bh + y < H; y++) for (unsigned x = 0; x < bw && i * bw + x < W; x++)
                    std::memcpy(&want[(((size_t)z * H + j * bh + y) * W + i * bw + x) * 4], px + (y * bw + x) * 4, 4);
            }
            const int64_t n = astc_decode(blocks, W, H, D, bw, bh, mode == 1, got);
            levels++;
            if (n != want_bad || want != got) { printf("MISMATCH level %dx%d %s: %lld errors (want %lld)\n", bw, bh, mode ? "sRGB" : "linear", (long long)n, (long long)want_bad); problems++; }
            if (astc_decode(std::span<const uint8_t>(blocks).first(blocks.size() - 1), W, H, D, bw, bh, mode == 1, got) != -1 ||
                astc_decode(blocks, W, H, D, 3, 3, false, got) != -1 || astc_decode(blocks, W, H, D, 13, 12, false, got) != -1) {
                printf("MISMATCH: a short span or a bad footprint was not refused\n"); problems++;
            }
        }
    }
    {   // A level whose block or texel count wraps 64 bits is refused (-1), nothing written, whatever the spans hold (the first
        // call is the review's reproduction: bx * by * depth = 2^29 * 2^29 * 64 wrapped to 0, and the old code then decoded 2^58 blocks)
        std::vector<uint8_t> in(4096, 0x11), out(4096, 0x5A);
        const std::vector<uint8_t> out0 = out;
        const uint32_t big = 1u << 31;
        if (astc_decode(in, big, big, 64, 4, 4, false, out) != -1 || astc_decode(in, big, big, 64, 12, 12, true, out) != -1 ||
            astc_decode(in, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 4, 4, false, out) != -1 || astc_decode(in, 0xFFFFFFFFu, 2, 1, 4, 4, false, out) != -1 ||
            astc_decode(in, 0xFFFFFFF9u, 1, 1, 12, 12, false, out) != -1 || astc_decode(in, 1, 1, 0xFFFFFFFFu, 4, 4, false, out) != -1 ||
            out != out0) {
            printf("MISMATCH: a level whose size wraps was not refused, or a refused call wrote\n"); problems++;
        }
        // (a width within bw of 2^32 is refused. This call does not isolate the 64-bit block count: it is the output span, 4 GB of texels
        // for a 4 KB buffer, that refuses it, and a span big enough to reach the count's wrap (16 GB) cannot be made here)
        std::vector<uint8_t> in2(16, 0x22);
        if (astc_decode(in2, 0xFFFFFFFFu, 1, 1, 4, 4, false, out) != -1 || out != out0) { printf("MISMATCH: width 2^32-1 was not refused\n"); problems++; }
    }
    {   // The error colour is the caller's: an all-zero block is an illegal one (reserved block mode), magenta by default and what the
        // caller says otherwise, in a block and in a level (edge blocks clipped), the count the same either way. HDR endpoint modes keep magenta.
        const uint8_t zero[16] = {};
        for (uint32_t bw : {4u, 6u, 12u}) {
            std::vector<uint8_t> px(bw * 5 * 4, 0x5A);
            if (astc_decode_block(zero, bw, 5, false, px.data()) || px[0] != 0xFF || px[1] != 0 || px[2] != 0xFF || px[3] != 0xFF) {
                printf("MISMATCH: an all-zero block is not opaque magenta by default\n"); problems++;
            }
            if (astc_decode_block(zero, bw, 5, true, px.data(), Rgba8{1, 2, 3, 4}) || px[0] != 1 || px[1] != 2 || px[2] != 3 || px[3] != 4 ||
                px[px.size() - 4] != 1 || px[px.size() - 1] != 4) {
                printf("MISMATCH: an all-zero block is not the caller's colour\n"); problems++;
            }
        }
        std::vector<uint8_t> level(16 * 6, 0), texels(9 * 7 * 4, 0x5A);   // (9 x 7 texels of 4 x 4 blocks: 3 x 2 blocks, all zero)
        if (astc_decode(level, 9, 7, 1, 4, 4, false, texels, Rgba8{0, 0, 0, 0}) != 6) { printf("MISMATCH: the error count of an all-zero level\n"); problems++; }
        for (uint8_t value : texels) if (value) { printf("MISMATCH: an all-zero level is not 00000000\n"); problems++; break; }
        if (astc_decode(level, 9, 7, 1, 4, 4, false, texels) != 6 || texels[0] != 0xFF || texels[2] != 0xFF || texels[3] != 0xFF || texels[1] != 0) {
            printf("MISMATCH: an all-zero level is not magenta by default\n"); problems++;
        }
    }
    printf("%ld block decodes compared with astcenc (%u blocks x linear and sRGB; %ld of them astcenc error blocks), %ld mismatches; "
           "%ld whole-level decodes\n", compared, count, illegal_expected, bad, levels);
    printf(problems ? "FAIL\n" : "PASS\n");
    return problems ? 1 : 0;
}
