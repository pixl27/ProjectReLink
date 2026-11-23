// Hookshot native engine: the narrow-phase leaves of hl/math/Support.h with no game (tests/test_native_engine.py): the
// support point against a brute-force argmax over vertex sets of every size (the guest's four jump cases and the rounds),
// its ties, its give-ups and its C surface; the relative transform against the identity, a pure translation and a plain
// formula, and its give-ups.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "hl/bridge/MathBridge.h"
#include "hl/c/hl_math.h"
#include "hl/math/Exact.h"
#include "hl/math/Support.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

using namespace hl;
using math::bits;

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

namespace {

uint32_t rng_state = 12345;
uint32_t rnd() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}
float frand(float lo, float hi) { return lo + (hi - lo) * (float)(rnd() & 0xffff) / 65536.0f; }

math::SupportTables tables() {
    math::SupportTables t = {};
    t.jump[0] = 88;
    t.jump[1] = 0;
    t.jump[2] = 14;
    t.jump[3] = 40;
    t.pad = -std::numeric_limits<float>::infinity();
    for (uint32_t k = 0; k < 4; k++) {
        t.index[k] = k;
        t.step[k] = 4;
    }
    const int8_t sel[8] = {0, 0, 1, 1, 2, 2, 2, 2};
    std::memcpy(t.select, sel, 8);
    return t;
}

float dot_of(const float v[3], const float d[4]) { return (v[0] * d[0] + v[1] * d[1]) + v[2] * d[2]; }

// n vertices then three vertices of garbage (the guest's rounds end exactly at n: nothing past it is read)
std::vector<float> make_set(int n) {
    std::vector<float> v((size_t)(n + 3) * 3);
    for (int i = 0; i < n * 3; i++) v[(size_t)i] = frand(-4.0f, 4.0f);
    for (int k = 0; k < 3; k++)
        for (int c = 0; c < 3; c++) v[(size_t)(n + k) * 3 + c] = 100.0f + (float)(k * 7 + c);
    return v;
}

void support_matches_brute_force() {
    const math::SupportTables t = tables();
    for (int n = 1; n <= 40; n++) {
        for (int rep = 0; rep < 60; rep++) {
            const std::vector<float> v = make_set(n);
            const float d[4] = {frand(-2, 2), frand(-2, 2), frand(-2, 2), frand(-2, 2)};
            int want = 0;
            for (int i = 1; i < n; i++)
                if (dot_of(&v[(size_t)i * 3], d) > dot_of(&v[(size_t)want * 3], d)) want = i;
            uint8_t out[16];
            CHECK(math::support_point(reinterpret_cast<const uint8_t*>(v.data()), n, d, t, out));
            CHECK(!std::memcmp(out, &v[(size_t)want * 3], 12));
            uint32_t id;
            std::memcpy(&id, out + 12, 4);
            CHECK(id == ((uint32_t)want | 0x3f000000u));
        }
    }
}

void support_ties_take_the_first() {
    const math::SupportTables t = tables();
    // four vertices, the second and third equal and the largest: the second
    float v[(4 + 3) * 3] = {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float d[4] = {1, 0, 0, 0};
    uint8_t out[16];
    CHECK(math::support_point(reinterpret_cast<const uint8_t*>(v), 4, d, t, out));
    uint32_t id;
    std::memcpy(&id, out + 12, 4);
    CHECK(id == (1u | 0x3f000000u));
    // three vertices, the first and the last equal and the largest: the last (the three-vertex case's table takes the
    // highest lane of the mask)
    float w[(3 + 3) * 3] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    CHECK(math::support_point(reinterpret_cast<const uint8_t*>(w), 3, d, t, out));
    std::memcpy(&id, out + 12, 4);
    CHECK(id == (2u | 0x3f000000u));
    // one vertex
    CHECK(math::support_point(reinterpret_cast<const uint8_t*>(w), 1, d, t, out));
    std::memcpy(&id, out + 12, 4);
    CHECK(id == 0x3f000000u);
}

void support_gives_up() {
    const math::SupportTables t = tables();
    std::vector<float> v = make_set(5);
    float d[4] = {1, 2, 3, 4};
    uint8_t out[16] = {};
    const uint8_t* p = reinterpret_cast<const uint8_t*>(v.data());
    CHECK(!math::support_point(p, 0, d, t, out));
    CHECK(!math::support_point(nullptr, 5, d, t, out));
    math::SupportTables bad = t;
    bad.jump[1] = 7;   // an unknown case
    CHECK(!math::support_point(p, 5, d, bad, out));
    float nand[4] = {std::nanf(""), 0, 0, 0};
    CHECK(!math::support_point(p, 5, nand, t, out));
    d[3] = std::nanf("");   // (w is not used)
    CHECK(math::support_point(p, 5, d, t, out));
    v[7] = std::numeric_limits<float>::infinity();   // a vertex coordinate the rounds read
    CHECK(!math::support_point(p, 5, d, t, out));
}

void relative_identity_and_translation() {
    math::PreciseTransform a = {}, b = {};
    for (int r = 0; r < 3; r++) a.row[r][r] = 1.0f;
    const float rot[3][3] = {{0.0f, 1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) b.row[r][c] = rot[r][c];
    a.hi[0] = 10.0f;
    a.lo[0] = 0.25f;
    b.hi[0] = 12.0f;
    b.hi[1] = 3.0f;
    b.lo[1] = 0.125f;
    b.lo[2] = 1.5f;
    math::RelativeTransform out;
    CHECK(math::relative_transform(a, b, out));
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) CHECK(out.row[r][c] == rot[r][c]);
    // the translation is b - a = (1.75, 3.125, 1.5), rotated by the identity: exactly representable, so the low part is 0
    CHECK(out.hi[0] == 1.75f && out.hi[1] == 3.125f && out.hi[2] == 1.5f && out.hi[3] == 1.5f);
    CHECK(out.lo[0] == 0.0f && out.lo[1] == 0.0f && out.lo[2] == 0.0f);
    CHECK(out.zero == 0);
    // a translation that does not fit a float: the low part carries the rest
    math::PreciseTransform c2 = a;
    c2.lo[1] = 0.0f;
    c2.hi[1] = 0.0f;
    b.hi[1] = 1.0f;
    b.lo[1] = 1.0e-9f;
    CHECK(math::relative_transform(c2, b, out));
    const double dv = (double)b.hi[1] + (double)b.lo[1];
    CHECK(out.hi[1] == (float)dv);
    CHECK(out.lo[1] == (float)(dv - (double)(float)dv));
    CHECK(out.lo[1] != 0.0f);
}

void relative_matches_a_plain_formula() {
    for (int rep = 0; rep < 500; rep++) {
        math::PreciseTransform a = {}, b = {};
        float* pa = &a.row[0][0];
        float* pb = &b.row[0][0];
        for (int i = 0; i < 19; i++) {
            pa[i] = frand(-3, 3);
            pb[i] = frand(-3, 3);
        }
        math::RelativeTransform out;
        CHECK(math::relative_transform(a, b, out));
        // lane k of the result is column k's entry: (a.row[0][k'] ...) read as the specification says, lane by lane
        for (int r = 0; r < 3; r++) {
            const float col0[4] = {a.row[0][0], a.row[1][0], a.row[2][0], a.row[2][1]};
            const float col1[4] = {a.row[0][1], a.row[1][1], a.row[2][1], a.row[2][3]};
            const float col2[4] = {a.row[0][2], a.row[1][2], a.row[2][2], a.row[2][3]};
            for (int k = 0; k < 4; k++) {
                const float want = col2[k] * b.row[r][2] + (col0[k] * b.row[r][0] + col1[k] * b.row[r][1]);
                CHECK(bits(out.row[r][k]) == bits(want));
            }
        }
        double dv[3];
        for (int k = 0; k < 3; k++) dv[k] = ((double)b.hi[k] + (double)b.lo[k]) - ((double)a.hi[k] + (double)a.lo[k]);
        for (int k = 0; k < 3; k++) {
            // (z + (x + y), as the guest adds)
            const double tz = dv[2] * (double)a.row[k][2], tx = dv[0] * (double)a.row[k][0], ty = dv[1] * (double)a.row[k][1];
            const double tt = tz + (tx + ty);
            CHECK(out.hi[k] == (float)tt);
            CHECK(out.lo[k] == (float)(tt - (double)(float)tt));
        }
        CHECK(out.hi[3] == out.hi[2]);
    }
}

void relative_gives_up() {
    math::PreciseTransform a = {}, b = {};
    for (int r = 0; r < 3; r++) a.row[r][r] = b.row[r][r] = 1.0f;
    math::RelativeTransform out;
    CHECK(math::relative_transform(a, b, out));
    math::PreciseTransform n = b;
    n.row[1][2] = std::nanf("");
    CHECK(!math::relative_transform(a, n, out));
    n = a;
    n.hi[2] = std::numeric_limits<float>::infinity();
    CHECK(!math::relative_transform(n, b, out));
    n = a;
    n.row[2][3] = 1.0e-40f;   // (a lane the columns use)
    CHECK(!math::relative_transform(n, b, out));
    n = a;
    n.row[0][3] = std::nanf("");   // (a lane nothing reads)
    n.hi[3] = std::nanf("");
    CHECK(math::relative_transform(n, b, out));
    // a rotation product that underflows in binary32
    n = a;
    n.row[0][0] = 1.0e-30f;
    b.row[0][0] = 1.0e-30f;
    CHECK(!math::relative_transform(n, b, out));
}

void the_c_surface() {
    alignas(16) uint8_t blob[8 + 16 + 12 * 8] = {};
    // header at 0: offset 24 to the vertices, count 3; vertices at 24
    const int32_t off = 24, n = 3;
    std::memcpy(blob, &off, 4);
    std::memcpy(blob + 4, &n, 4);
    const float vs[9 + 9] = {0, 0, 1, 5, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    std::memcpy(blob + 24, vs, sizeof vs);
    const float dir[4] = {1, 0, 0, 0};
    hl_math_support_tables t;
    const math::SupportTables mt = tables();
    std::memcpy(&t, &mt, sizeof t);
    uint8_t out[16] = {};
    CHECK(hl_math_support_point(blob, dir, &t, out));
    CHECK(!std::memcmp(out, &vs[3], 12));
    uint32_t id;
    std::memcpy(&id, out + 12, 4);
    CHECK(id == (1u | 0x3f000000u));
    // no array: declined
    const int32_t zero = 0;
    std::memcpy(blob, &zero, 4);
    CHECK(!hl_math_support_point(blob, dir, &t, out));
    // the relative transform through the C surface, the output over an input
    math::PreciseTransform a = {}, b = {};
    for (int r = 0; r < 3; r++) a.row[r][r] = b.row[r][r] = 1.0f;
    b.hi[0] = 2.0f;
    uint8_t buf[80];
    std::memcpy(buf, &b, 76);
    CHECK(hl_math_relative_transform(&a, buf, buf));
    float hi0;
    std::memcpy(&hi0, buf + 48, 4);
    CHECK(hi0 == 2.0f);
}

}   // namespace

int main() {
    support_matches_brute_force();
    support_ties_take_the_first();
    support_gives_up();
    relative_identity_and_translation();
    relative_matches_a_plain_formula();
    relative_gives_up();
    the_c_surface();
    std::puts("support: ok");
    return 0;
}
