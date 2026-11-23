// (hookshot) The item wheel's letters off Windows (gpu/vk/vk_wheel.cpp text_size, text_mask): stb_truetype, as Dear ImGui has
// it, from a font in memory (the bundled Libertinus Sans Bold: nx2/ui_fonts.py). Windows draws them with GDI; Linux had a GDI
// stand-in for this (host/posix/gdi_posix.c) until 70391859 removed it with the old menu, and its measures are kept here: the
// size in pixels is the em's (GDI's negative font height), a line is the ascent to the descent, the baseline sits the ascent
// below the top. UTF-8 in, every letter the font has (the stand-in drew Latin-1 only); a letter it lacks draws nothing.
// No Vulkan or ImGui here: tests/test_wheel_text.py checks it with the bundled font.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

// (stb_truetype, as Dear ImGui has it: its own static copy, as vk_game_font.h and imgui_draw.cpp have)
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif
#ifndef STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#define STBRP_STATIC
#define STB_RECT_PACK_IMPLEMENTATION
#endif
#include "imstb_rectpack.h"   // (first, as ImGui does: stb_truetype then uses its rectangle packer, not a stand-in of its own)
#include "imstb_truetype.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace mkwvk::wheel_text {

struct Face {
    stbtt_fontinfo info{};
    bool ok = false;
};
/// The font in data (the bytes stay the caller's, for the face's life); ok false when it is not one stb_truetype reads
inline Face face(const unsigned char* data, unsigned size) {
    Face f;
    if (!data || size < 12) return f;
    const int at = stbtt_GetFontOffsetForIndex(data, 0);
    f.ok = at >= 0 && stbtt_InitFont(&f.info, data, at) != 0;
    return f;
}
/// UTF-8 to code points ('?' for a broken sequence)
inline std::vector<int> codepoints(const char* s) {
    std::vector<int> out;
    for (const unsigned char* p = (const unsigned char*)s; *p;) {
        const unsigned c = *p;
        const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        int cp = n == 1 ? (int)c : n == 2 ? (int)(c & 31) : n == 3 ? (int)(c & 15) : (int)(c & 7);
        int k = 1;
        for (; k < n && (p[k] & 0xC0) == 0x80; k++) cp = cp << 6 | (p[k] & 63);
        if (!n || k < n) cp = '?', k = 1;
        out.push_back(cp);
        p += k;
    }
    return out;
}
inline float scale(const Face& f, int px) {
    return stbtt_ScaleForMappingEmToPixels(&f.info, (float)px);
}
/// The width and height of s at px pixels (0 by 0 when the face is not ok)
inline void measure(const Face& f, const char* s, int px, int* w, int* h) {
    *w = *h = 0;
    if (!f.ok) return;
    const float sc = scale(f, px);
    const std::vector<int> cps = codepoints(s);
    float x = 0;
    for (size_t i = 0; i < cps.size(); i++) {
        int adv, lsb;
        stbtt_GetCodepointHMetrics(&f.info, cps[i], &adv, &lsb);
        x += adv * sc;
        if (i + 1 < cps.size()) x += sc * stbtt_GetCodepointKernAdvance(&f.info, cps[i], cps[i + 1]);
    }
    int asc, desc, gap;
    stbtt_GetFontVMetrics(&f.info, &asc, &desc, &gap);
    *w = (int)std::ceil(x), *h = (int)std::ceil((asc - desc) * sc);
}
/// s at px pixels into an 8-bit coverage image (w by h, a byte a pixel), its top-left corner at (x, y): each pixel the larger
/// of what it had and the letters' coverage
inline void draw(const Face& f, const char* s, int px, int x, int y, uint8_t* cov, int w, int h) {
    if (!f.ok) return;
    const float sc = scale(f, px);
    int asc, desc, gap;
    stbtt_GetFontVMetrics(&f.info, &asc, &desc, &gap);
    const int baseline = y + (int)std::lround(asc * sc);
    const std::vector<int> cps = codepoints(s);
    float pen = (float)x;
    std::vector<uint8_t> g;
    for (size_t i = 0; i < cps.size(); i++) {
        int adv, lsb, x0, y0, x1, y1;
        stbtt_GetCodepointHMetrics(&f.info, cps[i], &adv, &lsb);
        const float sub = pen - std::floor(pen);
        stbtt_GetCodepointBitmapBoxSubpixel(&f.info, cps[i], sc, sc, sub, 0, &x0, &y0, &x1, &y1);
        const int gw = x1 - x0, gh = y1 - y0;
        if (gw > 0 && gh > 0) {
            g.assign((size_t)gw * gh, 0);
            stbtt_MakeCodepointBitmapSubpixel(&f.info, g.data(), gw, gh, gw, sc, sc, sub, 0, cps[i]);
            const int ox = (int)std::floor(pen) + x0, oy = baseline + y0;
            for (int yy = 0; yy < gh; yy++) {
                const int py = oy + yy;
                if (py < 0 || py >= h) continue;
                for (int xx = 0; xx < gw; xx++) {
                    const int qx = ox + xx;
                    if (qx < 0 || qx >= w) continue;
                    uint8_t& d = cov[(size_t)py * w + qx];
                    if (g[(size_t)yy * gw + xx] > d) d = g[(size_t)yy * gw + xx];
                }
            }
        }
        pen += adv * sc;
        if (i + 1 < cps.size()) pen += sc * stbtt_GetCodepointKernAdvance(&f.info, cps[i], cps[i + 1]);
    }
}

} // namespace mkwvk::wheel_text
