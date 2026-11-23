/* tests/test_fpxkeep.py's driver: nx2/fpxkeep.py's rules (N: NaN operands, I: infinite operands, Z: exact-zero
   multiply-adds) in the generated wrappers a64_fpx_{add,sub}{32,64}_*, a64_fpx_{mul,fma}{32,64}_fma* and in the kit's
   lane routes (a64_vfpx_add32 / sub32 / mul32 / mul32_elem when a lane fails the vector check), against the exact bodies
   recomp_fpe_* the wrappers fall back to: result bits and FPSR, under random FPCR / FPSR states (the gate open and
   closed). Built against the real a64.h with the a64_suyu.h / a64_suyu.c / a64_fpx.h the generators write, at
   -march=x86-64-v3 (-DEXPECT_FMA=1) and -march=x86-64-v2 (-DEXPECT_FMA=0: rule Z for binary64 needs the host's FMA and
   must keep nothing there).

   fpxkeep_test random <n> <seed>   n random cases of each scalar op (operand classes weighted to NaN, infinity, zero,
                                    subnormal, the range edges and exact or near cancellations)
   fpxkeep_test sweep               every pair (sums) and triple (multiply-adds) of a list of special values, under
                                    FPCR 0 / FZ and four FPSR states; and the Z rule's exponent edge: products whose
                                    exponents sum to the threshold - 2 .. + 2 with addends that make the exact sum 0,
                                    +-2^-1075 / 2^-150 (rounds to zero: underflow), +-2^-1074 / 2^-149
   fpxkeep_test zero <n> <seed>     n multiply-adds of each width whose addend is the negated rounded product or a few
                                    ulps off it (exact cancellations, and remainders the size of the product's rounding
                                    error), half of the binary64 ones with exponents summing to Z's edge - 8 .. + 8
   fpxkeep_test lanes <n> <seed>    n random 2- and 4-lane FADD / FSUB, 4-lane FMUL and FMUL by element, lanes weighted
                                    to quiet NaNs, other NaNs, infinities, subnormals and zeros,
                                    against each lane's exact body (2 lanes: the top half zero)
   The rules' results are counted inside the generated code (A64_FPXK_PROBE, empty in the real build): every rule must
   keep cases (Z for binary64 only when EXPECT_FMA), and none may be kept under a closed gate.
   Exit 0: pass. 1: a mismatch. 4: the counts are wrong. */
static unsigned long long g_rule[4];   /* N, I, Z, lane routes kept */
static int g_count_on = 1;
#define A64_FPXK_PROBE(k) ((void)(g_count_on && g_rule[k]++))
#include "a64.h"
#include <stdio.h>
#include <stdlib.h>
#if !defined(EXPECT_FMA) || RECOMP_FPX_HOST != 2 || RECOMP_FPX_FMA64 != EXPECT_FMA
#error "build with -DEXPECT_FMA=1 at x86-64-v3 or -DEXPECT_FMA=0 at x86-64-v2 (RECOMP_FPX_FMA64 must agree)"
#endif
#define GATE_FPCR (UINT64_C(0x06C8FF07) | (UINT64_C(1) << 32))

static uint64_t rs;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static uint32_t r32(void) { return (uint32_t)(rnd() >> 16); }
static uint32_t rbelow(uint32_t n) { return (uint32_t)(((rnd() >> 32) * n) >> 32); }
static uint32_t mkf(uint32_t s, int e, uint32_t frac) { return (s << 31) | ((uint32_t)e << 23) | (frac & 0x7fffff); }
static uint64_t mkd(uint64_t s, int e, uint64_t frac) { return (s << 63) | ((uint64_t)e << 52) | (frac & UINT64_C(0xfffffffffffff)); }
static float F(uint32_t x) { float f; memcpy(&f, &x, 4); return f; }
static uint32_t B(float f) { uint32_t x; memcpy(&x, &f, 4); return x; }
static double D(uint64_t x) { double f; memcpy(&f, &x, 8); return f; }
static uint64_t BD(double f) { uint64_t x; memcpy(&x, &f, 8); return x; }

/* ---- operands ---- */
static uint32_t sparse32(void) {
    switch (rbelow(4)) { case 0: return 0; case 1: return (uint32_t)1 << rbelow(23); case 2: return r32() & 0x7ff000; default: return r32(); }
}
static uint64_t sparse64(void) {
    switch (rbelow(4)) { case 0: return 0; case 1: return (uint64_t)1 << rbelow(52); case 2: return rnd() & UINT64_C(0xfffffff000000); default: return rnd(); }
}
static uint32_t op32(void) {
    const uint32_t s = r32() & 1;
    switch (rbelow(20)) {
    case 0: return s << 31;
    case 1: return mkf(s, 0, r32() | 1);
    case 2: return mkf(s, 0, rbelow(4) + 1);
    case 3: return mkf(s, 1 + rbelow(3), sparse32());
    case 4: return mkf(s, 254 - rbelow(3), sparse32());
    case 5: case 6: return mkf(s, 255, 0);                                /* inf */
    case 7: return 0x7fc12345u;                                           /* quiet NaN with payload */
    case 8: return mkf(s, 255, 0x400000 | r32());                         /* quiet NaN */
    case 9: return mkf(s, 255, (r32() & 0x3fffff) | (1u << rbelow(22)));  /* signalling NaN */
    case 10: case 11: return mkf(s, 120 + rbelow(15), sparse32());
    default: return mkf(s, 1 + rbelow(254), r32());
    }
}
static uint64_t op64(void) {
    const uint64_t s = r32() & 1;
    switch (rbelow(20)) {
    case 0: return s << 63;
    case 1: return mkd(s, 0, rnd() | 1);
    case 2: return mkd(s, 0, rbelow(4) + 1);
    case 3: return mkd(s, 1 + rbelow(3), sparse64());
    case 4: return mkd(s, 2046 - rbelow(3), sparse64());
    case 5: case 6: return mkd(s, 2047, 0);
    case 7: return UINT64_C(0x7ff8000000000000);
    case 8: return mkd(s, 2047, UINT64_C(0x8000000000000) | rnd());
    case 9: return mkd(s, 2047, (rnd() & UINT64_C(0x7ffffffffffff)) | ((uint64_t)1 << rbelow(51)));
    case 10: case 11: return mkd(s, 1010 + rbelow(25), sparse64());
    default: return mkd(s, 1 + rbelow(2046), rnd());
    }
}
/* a multiply-add's addend: often the (near) negated product, so that it cancels; or a product whose exponents sit at
   the Z rule's edge */
static void fma_case32(uint32_t *a, uint32_t *b, uint32_t *z) {
    *a = op32(); *b = op32();
    if (rbelow(3) == 0) {   /* exponents near the edge ea + eb = 151 */
        const int sum = 148 + (int)rbelow(7), ea = 1 + (int)rbelow((uint32_t)(sum - 1 < 254 ? sum - 1 : 254));
        if (sum - ea >= 1 && sum - ea <= 254) { *a = mkf(r32() & 1, ea, sparse32()); *b = mkf(r32() & 1, sum - ea, sparse32()); }
    }
    const double p = (double)F(*a) * (double)F(*b);
    switch (rbelow(8)) {
    case 0: case 1: *z = B((float)-p); break;
    case 2: *z = B((float)-p) + (uint32_t)((int)rbelow(5) - 2); break;
    case 3: *z = r32() & 0x80000000u; break;
    default: *z = op32();
    }
}
static void fma_case64(uint64_t *a, uint64_t *b, uint64_t *z) {
    *a = op64(); *b = op64();
    if (rbelow(3) == 0) {
        const int sum = 1073 + (int)rbelow(7), ea = 1 + (int)rbelow((uint32_t)(sum - 1));
        if (sum - ea >= 1 && sum - ea <= 2046) { *a = mkd(r32() & 1, ea, sparse64()); *b = mkd(r32() & 1, sum - ea, sparse64()); }
    }
    const double p = D(*a) * D(*b);
    switch (rbelow(8)) {
    case 0: case 1: *z = BD(-p); break;
    case 2: *z = BD(-p) + (uint64_t)((int64_t)rbelow(5) - 2); break;
    case 3: *z = (uint64_t)(r32() & 1) << 63; break;
    default: *z = op64();
    }
}
static void gen_state(uint64_t *fpcr, uint64_t *fpsr) {
    uint64_t cr = 0, sr = 0;
    if (rbelow(2)) cr |= 1u << 24;                        /* FZ */
    if (rbelow(12) == 0) cr |= 1u << 25;                  /* DN (gate closed) */
    if (rbelow(12) == 0) cr |= (uint64_t)rbelow(4) << 22; /* RMode */
    if (rbelow(40) == 0) cr |= r32() & 0x07ff9f07u;
    if (rbelow(200) == 0) cr |= RECOMP_FPX_INHIBIT;
    static const uint32_t bits[6] = {1, 2, 4, 8, 16, 128};
    if (rbelow(3) == 0) sr = 0x91;                        /* IXC, IOC and IDC set */
    else for (int k = 0; k < 6; k++) if (k == 4 ? rbelow(10) != 0 : rbelow(3) != 0) sr |= bits[k];
    *fpcr = cr;
    *fpsr = sr;
}

/* ---- checks: a wrapper against its exact body ---- */
static unsigned long long checks, mismatches, closed_kept;
static int report_left = 20;
static int closed(uint64_t cr, uint64_t sr) { return (cr & GATE_FPCR) || !(sr & 16u); }
static void mismatch(const char *what, uint64_t cr, uint64_t sr, uint64_t a, uint64_t b, uint64_t z, uint64_t want,
                     uint64_t wfpsr, uint64_t got, uint64_t gfpsr) {
    mismatches++;
    if (report_left-- > 0)
        printf("MISMATCH %s fpcr %#llx fpsr %#llx a %#llx b %#llx z %#llx: body %#llx fpsr %#llx, wrapper %#llx fpsr %#llx\n",
               what, (unsigned long long)cr, (unsigned long long)sr, (unsigned long long)a, (unsigned long long)b,
               (unsigned long long)z, (unsigned long long)want, (unsigned long long)wfpsr, (unsigned long long)got,
               (unsigned long long)gfpsr);
}
static unsigned long long kept_total(void) { return g_rule[0] + g_rule[1] + g_rule[2] + g_rule[3]; }
#define CHECK(what, call_ref, call_got, a, b, z) do { \
    Ctx ref, got; \
    memset(&ref, 0, sizeof ref); ref.fpcr = cr; ref.fpsr = sr; got = ref; \
    const uint64_t r_ = (call_ref); \
    const unsigned long long k0_ = kept_total(); \
    const uint64_t v_ = (call_got); \
    if (closed(cr, sr) && kept_total() != k0_) closed_kept++; \
    checks++; \
    if (r_ != v_ || ref.fpsr != got.fpsr) mismatch(what, cr, sr, a, b, z, r_, ref.fpsr, v_, got.fpsr); \
} while (0)

static void check_sum32(uint64_t cr, uint64_t sr, uint32_t a, uint32_t b) {
    CHECK("add32", recomp_fpe_add32(&ref, a, b), a64_fpx_add32_add32(&got, a, b), a, b, 0);
    CHECK("sub32", recomp_fpe_sub32(&ref, a, b), a64_fpx_sub32_sub32(&got, a, b), a, b, 0);
}
static void check_sum64(uint64_t cr, uint64_t sr, uint64_t a, uint64_t b) {
    CHECK("add64", recomp_fpe_add64(&ref, a, b), a64_fpx_add64_add64(&got, a, b), a, b, 0);
    CHECK("sub64", recomp_fpe_sub64(&ref, a, b), a64_fpx_sub64_sub64(&got, a, b), a, b, 0);
}
static void check_fma32(uint64_t cr, uint64_t sr, uint32_t a, uint32_t b, uint32_t z) {
    const uint32_t zm = (a ^ b) & 0x80000000u;   /* (FMUL: the lift passes the product's signed zero) */
    CHECK("mul32", recomp_fpe_fma32(&ref, a, b, zm), a64_fpx_mul32_fma32(&got, a, b, zm), a, b, zm);
    CHECK("fma32", recomp_fpe_fma32(&ref, a, b, z), a64_fpx_fma32_fma32(&got, a, b, z), a, b, z);
}
static unsigned long long z64;   /* Z's binary64 results (none without a host FMA) */
static void check_fma64(uint64_t cr, uint64_t sr, uint64_t a, uint64_t b, uint64_t z) {
    const uint64_t zm = (a ^ b) & UINT64_C(0x8000000000000000);
    const unsigned long long z0 = g_rule[2];
    CHECK("mul64", recomp_fpe_fma64(&ref, a, b, zm), a64_fpx_mul64_fma64(&got, a, b, zm), a, b, zm);
    CHECK("fma64", recomp_fpe_fma64(&ref, a, b, z), a64_fpx_fma64_fma64(&got, a, b, z), a, b, z);
    z64 += g_rule[2] - z0;
}

static void run_random(long n) {
    for (long i = 0; i < n; i++) {
        uint64_t cr, sr;
        gen_state(&cr, &sr);
        check_sum32(cr, sr, op32(), op32());
        check_sum64(cr, sr, op64(), op64());
        uint32_t a, b, z;
        fma_case32(&a, &b, &z);
        check_fma32(cr, sr, a, b, z);
        uint64_t A, Bv, Z;
        fma_case64(&A, &Bv, &Z);
        check_fma64(cr, sr, A, Bv, Z);
    }
}

/* Z alone: multiply-adds whose addend is the negated rounded product (exactly cancelling when the product is
   representable: mantissas of at most 26 / 12 bits) or a few ulps off it, the multiplicands' exponents anywhere and,
   half the time, summing to the binary64 edge 1076 - 8 .. + 8 */
static uint64_t short64(void) { return rnd() & ~((UINT64_C(1) << (26 + rbelow(27))) - 1) & UINT64_C(0xfffffffffffff); }
static void run_zero(long n) {
    for (long i = 0; i < n; i++) {
        uint64_t cr, sr;
        gen_state(&cr, &sr);
        int ea, eb;
        if (rbelow(2)) { const int sum = 1068 + (int)rbelow(17); ea = 1 + (int)rbelow((uint32_t)(sum - 1)); eb = sum - ea; }
        else { ea = 1 + (int)rbelow(2046); eb = 1 + (int)rbelow(2046); }
        if (eb < 1 || eb > 2046) continue;
        const uint64_t a = mkd(r32() & 1, ea, rbelow(3) ? short64() : rnd()), b = mkd(r32() & 1, eb, rbelow(3) ? short64() : rnd());
        const uint64_t z = BD(-(D(a) * D(b))) + (uint64_t)(rbelow(4) ? 0 : (int64_t)rbelow(5) - 2);
        check_fma64(cr, sr, a, b, z);
        const int fa = 1 + (int)rbelow(254), fb = 1 + (int)rbelow(254);
        const uint32_t a32 = mkf(r32() & 1, fa, rbelow(3) ? r32() & 0x7ff800 : r32());
        const uint32_t b32 = mkf(r32() & 1, fb, rbelow(3) ? r32() & 0x7ff800 : r32());
        const uint32_t z32 = B((float)-((double)F(a32) * (double)F(b32))) + (uint32_t)(rbelow(4) ? 0 : (int)rbelow(5) - 2);
        check_fma32(cr, sr, a32, b32, z32);
    }
}

static const uint32_t SPECIAL32[] = {
    0, 0x80000000u, 1, 0x80000001u, 0x007fffffu, 0x00400000u, 0x00800000u, 0x80800000u, 0x3f800000u, 0xbf800000u,
    0x40000000u, 0x3f800001u, 0x2b800000u, 0x33800000u, 0x5f800000u, 0x1f800000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u,
    0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7fc00001u, 0x7fffffffu, 0xffffffffu, 0x7f800001u, 0xff800001u, 0x7fbfffffu,
    0x7fa00000u, 0xffa12345u};
static const uint64_t SPECIAL64[] = {
    0, UINT64_C(0x8000000000000000), 1, UINT64_C(0x8000000000000001), UINT64_C(0x000fffffffffffff),
    UINT64_C(0x0010000000000000), UINT64_C(0x8010000000000000), UINT64_C(0x3ff0000000000000), UINT64_C(0xbff0000000000000),
    UINT64_C(0x4000000000000000), UINT64_C(0x3ff0000000000001), UINT64_C(0x2000000000000000), UINT64_C(0x5ff0000000000000),
    UINT64_C(0x7fefffffffffffff), UINT64_C(0xffefffffffffffff), UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
    UINT64_C(0x7ff8000000000000), UINT64_C(0xfff8000000000000), UINT64_C(0x7ff8000000000001), UINT64_C(0x7fffffffffffffff),
    UINT64_C(0x7ff0000000000001), UINT64_C(0xfff0000000000001), UINT64_C(0x7ff7ffffffffffff), UINT64_C(0x7ff4000000000000),
    UINT64_C(0xfff4123456789abc)};
#define NS32 (sizeof SPECIAL32 / sizeof SPECIAL32[0])
#define NS64 (sizeof SPECIAL64 / sizeof SPECIAL64[0])
static const uint64_t SW_CR[2] = {0, 1u << 24}, SW_SR[4] = {0x10, 0x95, 0x11, 0x90};

static void run_sweep(void) {
    for (int ci = 0; ci < 2; ci++)
        for (int si = 0; si < 4; si++) {
            const uint64_t cr = SW_CR[ci], sr = SW_SR[si];
            for (unsigned i = 0; i < NS32; i++)
                for (unsigned j = 0; j < NS32; j++) {
                    check_sum32(cr, sr, SPECIAL32[i], SPECIAL32[j]);
                    for (unsigned k = 0; k < NS32; k++) check_fma32(cr, sr, SPECIAL32[i], SPECIAL32[j], SPECIAL32[k]);
                }
            for (unsigned i = 0; i < NS64; i++)
                for (unsigned j = 0; j < NS64; j++) {
                    check_sum64(cr, sr, SPECIAL64[i], SPECIAL64[j]);
                    for (unsigned k = 0; k < NS64; k++) check_fma64(cr, sr, SPECIAL64[i], SPECIAL64[j], SPECIAL64[k]);
                }
            /* Z's edge: a = (2^p + 1) 2^.., b likewise, z = -(a*b - 2^E) with E = ea + eb - bias2: the exact sum is
               2^E (and 0, -2^E, 2^(E+1) by the addend's last bits); ea + eb from the threshold - 2 to + 2 */
            for (int sum = 149; sum <= 153; sum++)
                for (int ea = 1; ea <= 254; ea++) {
                    const int eb = sum - ea;
                    if (eb < 1 || eb > 254 || sum - 127 < 1) continue;
                    for (int s = 0; s < 2; s++)
                        for (int d = -2; d <= 2; d++) {
                            const uint32_t a = mkf((uint32_t)s, ea, 1), b = mkf(0, eb, 1);
                            const uint32_t z = mkf((uint32_t)!s, sum - 127, 2) + (uint32_t)d;
                            check_fma32(cr, sr, a, b, z);
                        }
                }
            for (int sum = 1074; sum <= 1078; sum++)
                for (int ea = 1; ea <= 2046; ea += (ea < 40 || ea > 2000 || (ea > 500 && ea < 560)) ? 1 : 7) {
                    const int eb = sum - ea;
                    if (eb < 1 || eb > 2046) continue;
                    for (int s = 0; s < 2; s++)
                        for (int d = -2; d <= 2; d++) {
                            const uint64_t a = mkd((uint64_t)s, ea, 1), b = mkd(0, eb, 1);
                            const uint64_t z = mkd((uint64_t)!s, sum - 1023, 2) + (uint64_t)(int64_t)d;
                            check_fma64(cr, sr, a, b, z);
                        }
                }
        }
}

/* ---- the lane routes: a64_vfpx_* against each lane's exact body ---- */
static uint32_t lane32(void) {
    switch (rbelow(10)) {
    case 0: case 1: return 0x7fc12345u;
    case 2: return op32();
    default: return mkf(r32() & 1, 100 + rbelow(56), r32());   /* ordinary values */
    }
}
static void run_lanes(long n) {
    for (long i = 0; i < n; i++) {
        uint64_t cr, sr;
        gen_state(&cr, &sr);
        uint32_t a[4], b[4];
        for (int j = 0; j < 4; j++) { a[j] = lane32(); b[j] = lane32(); }
        if (rbelow(4) == 0) for (int j = 0; j < 4; j++) b[j] = b[0];   /* (by element) */
        for (int op = 0; op < 4; op++) {
            const int k = (op < 2 && rbelow(2)) ? 2 : 4;
            Ctx ref, got;
            memset(&ref, 0, sizeof ref);
            ref.fpcr = cr; ref.fpsr = sr;
            uint32_t want[4] = {0, 0, 0, 0};
            for (int j = 0; j < k; j++)
                want[j] = (uint32_t)(op == 0 ? recomp_fpe_add32(&ref, a[j], b[j]) : op == 1 ? recomp_fpe_sub32(&ref, a[j], b[j])
                                     : recomp_fpe_fma32(&ref, a[j], op == 3 ? b[0] : b[j], (a[j] ^ (op == 3 ? b[0] : b[j])) & 0x80000000u));
            memset(&got, 0, sizeof got);
            got.fpcr = cr; got.fpsr = sr;
            memcpy(got.vreg[1], a, 16);
            memcpy(got.vreg[2], b, 16);
            got.vreg[0][0] = rnd(); got.vreg[0][1] = rnd();   /* (junk in the destination) */
            const unsigned long long k0 = kept_total();
            if (op == 0) a64_vfpx_add32(&got, got.vreg[0], got.vreg[1], got.vreg[2], k);
            else if (op == 1) a64_vfpx_sub32(&got, got.vreg[0], got.vreg[1], got.vreg[2], k);
            else if (op == 2) a64_vfpx_mul32(&got, got.vreg[0], got.vreg[1], got.vreg[2], 4);
            else a64_vfpx_mul32_elem(&got, got.vreg[0], got.vreg[1], b[0]);
            /* (FPCR only: with IXC clear the first inexact lane's body sets it, and the lanes after it, in the old lane
               route as here, take the fast path and the rescues under the now open gate) */
            if ((cr & GATE_FPCR) && kept_total() != k0) closed_kept++;
            checks++;
            if (memcmp(got.vreg[0], want, 16) || got.fpsr != ref.fpsr) {
                mismatches++;
                if (report_left-- > 0) {
                    uint32_t g[4];
                    memcpy(g, got.vreg[0], 16);
                    printf("MISMATCH lanes op %d k %d fpcr %#llx fpsr %#llx a %08x %08x %08x %08x b %08x %08x %08x %08x: "
                           "bodies %08x %08x %08x %08x fpsr %#llx, got %08x %08x %08x %08x fpsr %#llx\n",
                           op, k, (unsigned long long)cr, (unsigned long long)sr, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
                           want[0], want[1], want[2], want[3], (unsigned long long)ref.fpsr, g[0], g[1], g[2], g[3],
                           (unsigned long long)got.fpsr);
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    const char *mode = argv[1];
    rs = argc > 3 ? strtoull(argv[3], 0, 10) * 2654435761u + 1 : 88172645463325252ull;
    const long n = argc > 2 ? strtol(argv[2], 0, 10) : 0;
    if (!strcmp(mode, "random")) run_random(n);
    else if (!strcmp(mode, "sweep")) run_sweep();
    else if (!strcmp(mode, "lanes")) run_lanes(n);
    else if (!strcmp(mode, "zero")) run_zero(n);
    else return 2;
    printf("fpxkeep %s (%s): %llu checks, %llu mismatches; kept by N %llu, I %llu, Z %llu (binary64 %llu), lane routes "
           "%llu; kept under a closed gate %llu\n", mode, EXPECT_FMA ? "x86-64-v3" : "x86-64-v2", checks, mismatches,
           g_rule[0], g_rule[1], g_rule[2], z64, g_rule[3], closed_kept);
    if (mismatches) return 1;
    if (closed_kept) return 4;
    if (!strcmp(mode, "lanes") ? !g_rule[3] : !strcmp(mode, "zero") ? (!g_rule[2] || !z64 != !EXPECT_FMA)
                                : (!g_rule[0] || !g_rule[1] || !g_rule[2] || !z64 != !EXPECT_FMA)) return 4;
    return 0;
}
