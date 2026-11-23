// Hookshot native engine: the narrow-phase leaves of hl/math/Support.h. Each line is one guest operation, in the guest's
// order; the comments name the guest function the sequence is read from (by address, nothing of its code).
#include "hl/math/Support.h"

#include <cmath>
#include <cstring>

#include "hl/math/Exact.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace hl::math {

namespace {

struct Dots {
    Guard& g;
    const uint8_t* v;
    float d[3];

    float coord(int64_t i, int c) {
        float f;
        std::memcpy(&f, v + i * 12 + c * 4, 4);
        return g.in(f);
    }
    // (x dx + y dy) + z dz, each product rounded: the guest's lane arithmetic (the sum of the first two, then the third)
    float dot(int64_t i) {
        const float x = coord(i, 0), y = coord(i, 1), z = coord(i, 2);
        const float a = g.mul(x, d[0]), b = g.mul(y, d[1]), c = g.mul(z, d[2]);
        return g.add(g.add(a, b), c);
    }
};

}   // namespace

// 0x817472e0
bool support_point(const uint8_t* vertices, int32_t n, const float dir[4], const SupportTables& t, uint8_t out[16]) {
    if (!vertices || n < 1) return false;
    Guard g;
    Dots w{g, vertices, {g.in(dir[0]), g.in(dir[1]), g.in(dir[2])}};
    float best[4];
    uint32_t idx[4];
    int32_t start;
    uint32_t chosen;
    const uint8_t jump = t.jump[n & 3];
    // The guest enters by a jump on n & 3: one vertex, two, three (when n is exactly 3; another n of 3 mod 4 takes the
    // four-vertex start), or the first four. The cases start the lane maxima; the rounds then take four vertices each from
    // `start`, which is congruent to n mod 4, so the rounds end exactly at n and read nothing past it.
    if (jump == 88 || (jump == 40 && n != 3)) {   // the first four vertices, then the rounds from the fifth (a count of 0 mod 4)
        for (int k = 0; k < 4; k++) {                 // or the fourth (3 mod 4: the guest's second entry, which sets 3)
            best[k] = w.dot(k);
            idx[k] = t.index[k];
        }
        start = jump == 88 ? 4 : 3;
    } else if (jump == 0) {   // the first vertex in all lanes, the rounds from the second
        const float d = w.dot(0);
        for (int k = 0; k < 4; k++) {
            best[k] = d;
            idx[k] = 0;
        }
        start = 1;
    } else if (jump == 14) {   // two vertices, the other two lanes padded with a large negative, the rounds from the third
        best[0] = w.dot(0);
        best[1] = w.dot(1);
        // (the module's padding is minus infinity in the game: an infinity compares the same on x86 and Arm, so only a NaN
        // or a subnormal padding is given up on)
        if (std::isnan(t.pad) || (!is_zero(t.pad) && (bits(t.pad) & 0x7f800000u) == 0)) return false;
        best[2] = best[3] = t.pad;
        for (int k = 0; k < 4; k++) idx[k] = t.index[k];
        start = 2;
    } else if (jump == 40) {   // exactly three: the lanes at the maximum are told by a mask, the table gives the index
        const float d0 = w.dot(0), d1 = w.dot(1), d2 = w.dot(2);
        if (!g.ok) return false;
        const float m01 = d0 > d1 ? d0 : d1;
        const float m = d2 > m01 ? d2 : m01;
        const int mask = ((d0 >= m) ? 1 : 0) + ((d1 >= m) ? 2 : 0);
        chosen = (uint32_t)(int32_t)t.select[mask | ((d2 >= m) ? 4 : 0)];
        goto done;
    } else {
        return false;
    }
    if (start < n) {
        uint32_t lane[4];
        for (int k = 0; k < 4; k++) lane[k] = (uint32_t)start + t.index[k];
        int64_t s = start;
        do {
            for (int k = 0; k < 4; k++) {
                const float d = w.dot(s + k);
                if (d > best[k]) {
                    best[k] = d;
                    idx[k] = lane[k];
                }
            }
            for (int k = 0; k < 4; k++) lane[k] += t.step[k];
            s += 4;
        } while (s < n);   // (the guest counts from `start` by four against n)
    }
    {
        // the lanes' pairs, then the better pair (the first of a tie each time)
        const uint32_t s01 = best[0] >= best[1] ? idx[0] : idx[1];
        const uint32_t s23 = best[2] >= best[3] ? idx[2] : idx[3];
        const float m01 = best[0] > best[1] ? best[0] : best[1];
        const float m23 = best[2] > best[3] ? best[2] : best[3];
        chosen = m01 >= m23 ? s01 : s23;
    }
done:
    if (!g.ok) return false;
    std::memcpy(out, vertices + (int64_t)(int32_t)chosen * 12, 12);
    const uint32_t id = chosen | 0x3f000000u;
    std::memcpy(out + 12, &id, 4);
    return true;
}

// 0x817036a0
bool relative_transform(const PreciseTransform& a, const PreciseTransform& b, RelativeTransform& out) {
    Guard g;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            g.in(a.row[r][c]);
            g.in(b.row[r][c]);
        }
    g.in(a.row[2][3]);
    for (int k = 0; k < 3; k++) {
        g.in(a.hi[k]);
        g.in(a.lo[k]);
        g.in(b.hi[k]);
        g.in(b.lo[k]);
    }
    if (!g.ok) return false;
    // the translation difference, in binary64: (b high + b low) - (a high + a low)
    double dv[3];
    for (int k = 0; k < 3; k++) {
        const double pb = g.add((double)b.hi[k], (double)b.lo[k]);
        const double pa = g.add((double)a.hi[k], (double)a.lo[k]);
        dv[k] = g.sub(pb, pa);
    }
    // rotated by a's rows, each as (x + y) + z, split into a float and what the float left
    float hi[3], lo[3];
    for (int k = 0; k < 3; k++) {
        const double x = g.mul(dv[0], (double)a.row[k][0]);
        const double y = g.mul(dv[1], (double)a.row[k][1]);
        const double z = g.mul(dv[2], (double)a.row[k][2]);
        const double tk = g.add(z, g.add(x, y));
        hi[k] = g.narrow(tk);
        lo[k] = g.narrow(g.sub(tk, (double)hi[k]));
    }
    // a's columns (the guest builds them by lane shuffles: the fourth lanes are the second and last of the third row)
    const float col[3][4] = {
        {a.row[0][0], a.row[1][0], a.row[2][0], a.row[2][1]},
        {a.row[0][1], a.row[1][1], a.row[2][1], a.row[2][3]},
        {a.row[0][2], a.row[1][2], a.row[2][2], a.row[2][3]},
    };
    RelativeTransform r;
    for (int row = 0; row < 3; row++)
        for (int k = 0; k < 4; k++) {
            const float p0 = g.mul(col[0][k], b.row[row][0]);
            const float p1 = g.mul(col[1][k], b.row[row][1]);
            const float p2 = g.mul(col[2][k], b.row[row][2]);
            r.row[row][k] = g.add(p2, g.add(p0, p1));
        }
    r.hi[0] = hi[0];
    r.hi[1] = hi[1];
    r.hi[2] = r.hi[3] = hi[2];
    for (int k = 0; k < 3; k++) r.lo[k] = lo[k];
    r.zero = 0;
    if (!g.ok) return false;
    out = r;
    return true;
}

}   // namespace hl::math
