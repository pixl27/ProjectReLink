/* tests/fma_soft_test.c: driver of tests/test_fma_soft.py. Runs the generated runtime's recomp_fpe_fma32 and
   recomp_fpe_fma64 twice on the same operands and the same context, once with nx2/fma_soft.h's early exit off
   (recomp_fma_soft_mode = 0: suyu's exact body, what main does) and once with it on, and requires the result and the
   whole GuestContext (FPSR, FPCR and everything else) to be identical.

     fma_soft_test random <trials> <seed>   random operands of every class, random FPCR and FPSR
     fma_soft_test edge                     every exponent pair around the early exit's range edges, several fractions

   Exit 0 pass, 1 mismatch, 2 the early exit was taken too rarely to prove anything, 3 not this CPU's level (build only).
   Where the early exit is compiled out (an FMA3 build: recomp_fma_soft_mode is not defined) the two runs are the same
   code and none may be taken. */
#include "a64.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int recomp_fma_soft_mode __attribute__((weak));
uint64_t g_fma_soft_taken[2];   /* (RECOMP_FMA_SOFT_PROBE(k) of the runtime: 0 = binary32, 1 = binary64) */

#ifndef EXPECT_SOFT
#error "EXPECT_SOFT: whether the early exit is compiled in (1) or out (0) at this -march"
#endif

static uint64_t rs = 1;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double d_of(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t u_of(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t uf_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

typedef uint64_t (*fma_fn)(GuestContext*, uint64_t, uint64_t, uint64_t);
static long bad[2], tot[2];
static const char* const NAME[2] = {"fma32", "fma64"};

static void check(int dbl, uint64_t fpcr, uint64_t fpsr, uint64_t a, uint64_t b, uint64_t z) {
    static GuestContext init, c0, c1;
    const fma_fn f = dbl ? recomp_fpe_fma64 : recomp_fpe_fma32;
    memset(&init, 0, sizeof init);
    init.fpcr = fpcr;
    init.fpsr = fpsr;
    memcpy(&c0, &init, sizeof init);
    memcpy(&c1, &init, sizeof init);
    recomp_fma_soft_mode = 0;
    const uint64_t r0 = f(&c0, a, b, z);
    recomp_fma_soft_mode = 1;
    const uint64_t r1 = f(&c1, a, b, z);
    tot[dbl]++;
    if (r0 != r1 || memcmp(&c0, &c1, sizeof c0)) {
        if (bad[dbl]++ < 8)
            printf("MISMATCH %s a=%016llx b=%016llx z=%016llx fpcr=%llx fpsr=%llx: exact %016llx fpsr %llx, early %016llx fpsr %llx\n",
                   NAME[dbl], (unsigned long long)a, (unsigned long long)b, (unsigned long long)z, (unsigned long long)fpcr,
                   (unsigned long long)fpsr, (unsigned long long)r0, (unsigned long long)c0.fpsr, (unsigned long long)r1,
                   (unsigned long long)c1.fpsr);
    }
}

/* ---- operands ---- */
static uint64_t frac(int dbl) {
    const int bits = dbl ? 52 : 23;
    const uint64_t all = (UINT64_C(1) << bits) - 1;
    switch (rnd() % 7) {
    case 0: return 0;
    case 1: return all;
    case 2: return UINT64_C(1) << (rnd() % bits);
    case 3: return (rnd() & all) & ~((UINT64_C(1) << (rnd() % (bits - 2))) - 1);   /* few significant bits: exact products */
    case 4: return all - (rnd() % 4);
    default: return rnd() & all;
    }
}
static int pick_exp(int dbl, int near_product) {
    const int top = dbl ? 0x7ff : 0xff, bias = dbl ? 1023 : 127;
    static const int E64[] = {400, 1600, 160, 1880, 100, 1900};   /* (the early exit's own limits) */
    static const int E32[] = {40, 210, 2, 253};
    switch (rnd() % 7) {
    case 0: { static const int S[] = {0, 1, 2, -1, -2}; int s = S[rnd() % 5]; return s < 0 ? top + s + 1 : s; }
    case 1: { const int* E = dbl ? E64 : E32; int n = dbl ? 6 : 4; return E[rnd() % n] + (int)(rnd() % 7) - 3; }
    case 2: return (int)(rnd() % (top + 1));
    case 3: return bias + (int)(rnd() % 49) - 24;
    case 4: return near_product + (int)(rnd() % 9) - 4;
    default: return bias + (int)(rnd() % (dbl ? 801 : 101)) - (dbl ? 400 : 50);
    }
}
static uint64_t make(int dbl, int e) {
    const int top = dbl ? 0x7ff : 0xff;
    if (e < 0) e = 0;
    if (e > top) e = top;
    const uint64_t sign = (rnd() & 1) << (dbl ? 63 : 31);
    return sign | ((uint64_t)e << (dbl ? 52 : 23)) | frac(dbl);
}
static int expo(int dbl, uint64_t x) { return dbl ? (int)((x >> 52) & 0x7ff) : (int)((x >> 23) & 0xff); }

static uint64_t addend(int dbl, uint64_t a, uint64_t b) {
    const int bias = dbl ? 1023 : 127, ep = expo(dbl, a) + expo(dbl, b) - bias;
    const uint64_t signbit = UINT64_C(1) << (dbl ? 63 : 31);
    switch (rnd() % 9) {
    case 0: return (rnd() & 1) ? 0 : signbit;                                    /* +-0 */
    case 1: {                                                                       /* cancels the product to a few ulps */
        uint64_t z;
        if (dbl) z = u_of(-(d_of(a) * d_of(b))) + (rnd() % 7) - 3;
        else z = uf_of((float)(-(double)f_of((uint32_t)a) * (double)f_of((uint32_t)b))) + (uint32_t)(rnd() % 7) - 3;
        return z;
    }
    case 2: return make(dbl, ep + (int)(rnd() % 9) - 4);                           /* about the product's size */
    case 3: return make(dbl, ep - (dbl ? 53 : 24) + (int)(rnd() % 7) - 3);         /* around the product's last place: the sticky cases */
    case 4: return make(dbl, ep + (dbl ? 53 : 24) + (int)(rnd() % 7) - 3);         /* the product is the rounding's sticky part */
    case 8: {   /* z + product is a power of two (or a few ulps off it), with the product's low part left over */
        const int m = ep + (int)(rnd() % 5) - 2;
        uint64_t z;
        if (m < 1 || m > (dbl ? 2046 : 254)) return make(dbl, ep);
        if (dbl) z = u_of(d_of((uint64_t)m << 52) - d_of(a) * d_of(b)) + (rnd() % 5) - 2;
        else z = uf_of((float)((double)f_of((uint32_t)m << 23) - (double)f_of((uint32_t)a) * (double)f_of((uint32_t)b))) + (uint32_t)(rnd() % 5) - 2;
        return (rnd() & 1) ? z ^ signbit : z;
    }
    default: return make(dbl, pick_exp(dbl, ep));
    }
}

static uint64_t pick_fpcr(void) {
    switch (rnd() % 30) {
    case 0: case 1: case 2: case 3: return UINT64_C(1) << 24;                       /* FZ */
    case 4: case 5: return UINT64_C(1) << 25;                                       /* DN */
    case 6: case 7: return (uint64_t)(1 + rnd() % 3) << 22;                         /* RMode */
    case 8: return rnd() % 8;                                                       /* FIZ, AH, NEP */
    case 9: return (rnd() % 256) << 8;                                              /* trap enables */
    case 10: return UINT64_C(1) << 32;                                              /* FPX1's kill switch */
    case 11: return rnd();
    default: return 0;
    }
}
static uint64_t pick_fpsr(void) {
    static const uint64_t S[] = {0, 16, 0x1f, 0x9f, 0x80, 0x08};
    return (rnd() % 3 == 0) ? (rnd() & 0x9f) : S[rnd() % 6];
}

static void random_run(long n) {
    for (long i = 0; i < n; i++) {
        const int dbl = (int)(rnd() & 1);
        const uint64_t a = make(dbl, pick_exp(dbl, 0)), b = make(dbl, pick_exp(dbl, 0));
        const uint64_t z = addend(dbl, a, b);
        const uint64_t fpcr = (rnd() % 4 == 0) ? pick_fpcr() : 0, fpsr = pick_fpsr();
        check(dbl, fpcr, fpsr, a, b, z);
        if (rnd() % 4 == 0) check(dbl, fpcr, fpsr, a, z, b);   /* (an addend as a factor) */
    }
}

/* every exponent pair around the early exit's edges (and the middle), a few fractions each, all sign patterns */
static void edge_run(void) {
    for (int dbl = 0; dbl < 2; dbl++) {
        static const int A64[] = {397, 398, 399, 400, 401, 402, 700, 1023, 1500, 1598, 1599, 1600, 1601, 1602, 1603};
        static const int A32[] = {37, 38, 39, 40, 41, 42, 100, 127, 180, 208, 209, 210, 211, 212, 213};
        static const int Z64[] = {0, 1, 157, 158, 159, 160, 161, 162, 1000, 1878, 1879, 1880, 1881, 1882, 2046, 2047};
        static const int Z32[] = {0, 1, 37, 38, 39, 40, 41, 42, 100, 208, 209, 210, 211, 212, 254, 255};
        const int *A = dbl ? A64 : A32, *Zs = dbl ? Z64 : Z32;
        const int top = dbl ? 0x7ff : 0xff, bias = dbl ? 1023 : 127, bits = dbl ? 52 : 23;
        const uint64_t all = (UINT64_C(1) << bits) - 1;
        const uint64_t F[] = {0, 1, all, all - 1, UINT64_C(1) << (bits - 1), 0x5555555555555ULL & all};
        const uint64_t sbit = UINT64_C(1) << (dbl ? 63 : 31);
        for (int ia = 0; ia < 15; ia++)
            for (int ib = 0; ib < 15; ib++) {
                const int ep = A[ia] + A[ib] - bias;
                for (int iz = 0; iz < 20; iz++) {
                    const int ez = iz < 16 ? Zs[iz] : (iz == 16 ? ep : (iz == 17 ? ep - (bits + 1) : (iz == 18 ? ep + bits + 1 : ep - 1)));
                    if (ez < 0 || ez > top) continue;
                    for (int fa = 0; fa < 6; fa++)
                        for (int fb = 0; fb < 6; fb++)
                            for (int fz = 0; fz < 6; fz += (iz < 16 ? 2 : 1)) {
                                for (int sg = 0; sg < 8; sg++) {
                                    const uint64_t a = ((sg & 1) ? sbit : 0) | ((uint64_t)A[ia] << bits) | F[fa];
                                    const uint64_t b = ((sg & 2) ? sbit : 0) | ((uint64_t)A[ib] << bits) | F[fb];
                                    const uint64_t z = ((sg & 4) ? sbit : 0) | ((uint64_t)ez << bits) | F[fz];
                                    check(dbl, 0, (sg & 1) ? 16 : 0, a, b, z);
                                }
                            }
                }
            }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) return 64;
    const int compiled = &recomp_fma_soft_mode != NULL;
    if (compiled != EXPECT_SOFT) {
        printf("fma soft: the early exit is %s, expected %s\n", compiled ? "compiled in" : "compiled out", EXPECT_SOFT ? "in" : "out");
        return 1;
    }
    if (!compiled) {   /* an FMA3 build: nothing to compare, and nothing may be taken */
        printf("fma soft: compiled out at this -march (as expected)\n");
        return 0;
    }
    if (!strcmp(argv[1], "random")) {
        rs = argc > 3 ? strtoull(argv[3], 0, 10) : 1;
        random_run(argc > 2 ? atol(argv[2]) : 1000000);
    } else {
        rs = 99;
        edge_run();
    }
    printf("fma soft %s: fma32 %ld checks, %ld mismatches, early exit taken %llu; fma64 %ld checks, %ld mismatches, taken %llu\n", argv[1],
           tot[0], bad[0], (unsigned long long)g_fma_soft_taken[0], tot[1], bad[1], (unsigned long long)g_fma_soft_taken[1]);
    if (bad[0] || bad[1]) return 1;
    /* (the early exit runs in the second pass only: some share of the checks must take it, or this proves nothing; the edge sweep
       aims at the range's limits, so most of its checks are outside it) */
    const uint64_t share = strcmp(argv[1], "random") ? 40 : 8;
    if (g_fma_soft_taken[0] < (uint64_t)tot[0] / share || g_fma_soft_taken[1] < (uint64_t)tot[1] / share) {
        printf("fma soft: the early exit was taken too rarely\n");
        return 2;
    }
    return 0;
}
