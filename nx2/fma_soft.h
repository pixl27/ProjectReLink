/* nx2/fma_soft.h: a correctly rounded fused multiply-add for x86-64 hosts without FMA3, in front of suyu's exact bodies.

   nx2/suyu_rt_fixes.py puts this text into the generated runtime (a64_suyu.c, after its #include "a64.h") and an early
   exit at the top of recomp_fpe_fma32 and recomp_fpe_fma64. Without a hardware FMA (RECOMP_FPX_FMA64 is 0: an x86-64-v2
   build, which host_march() picks for a CPU without AVX2) FPX1 keeps no double-precision FMA fast path, and the
   single-precision one only when the FPSR flags it would raise are set already, so most FMADD/FMSUB/FNMADD/FNMSUB/FMLA
   reach the exact bodies: multi-word integer arithmetic that zeroes two arrays and loops over 67 words (binary64) or 9
   (binary32), about 180 ns a call on an i7-3770. This takes the common case in a few ns and leaves everything else to
   the exact body, so no result and no FPSR bit can differ from the exact body's where the early exit is taken.

   binary64: Boldo and Melquiond's fma(a, b, c) = RN(th + RO(tl + ul)): (uh, ul) = ExactMult(a, b) by Veltkamp's split
   and Dekker's product, (th, tl) = TwoSum(c, uh), RO the rounding to odd (the neighbour of RN(x) with an odd last bit
   when the sum was inexact), RN the last rounding to nearest even. Rounding to odd keeps the sticky information, so the
   one final rounding is the correctly rounded result (rounding the intermediate sum to nearest instead rounds twice).
   binary32: the product of two binary32 values is exact in binary64, so one binary64 sum, rounded to odd, then a
   rounding to binary32 (binary64 has more than 2 x 24 + 2 bits, which makes the rounding to odd safe) is the result.
   The inexact flag (FPSR.IXC, bit 4, the only flag a normal-range FMA can raise) is whether the result differs from the
   exact value, read off the two roundings' errors.

   What is taken (anything else returns 0 and the exact body runs): FPCR with no control bit but FZ set (round to
   nearest even, default NaN off, no traps, no FIZ/AH/NEP, no kill switch: RECOMP_FPX_FPCR_MASK, as FPX1's own gate);
   normal operands whose exponents keep the products, their error terms and Veltkamp's split in range (binary64:
   operand exponents 400..1600, product 160..1880, addend 160..1880 or zero; binary32: 40..210, addend likewise or
   zero); a normal, nonzero, finite result; and, as a margin, not an inexact sum whose leading part is a power of two
   (the neighbours are unevenly spaced there; the argument that the flag is read off the errors holds there too and no
   input has been found that needs the exclusion, which costs under 1 % of the calls). Zero, subnormal, infinite and NaN
   operands, results near the range limits and the other rounding modes are all the exact body's.

   It relies on what the rest of FPX1 relies on: strict IEEE binary64 on SSE2 with the default MXCSR (RECOMP_FPX_HOST 2
   is only chosen without FLT_EVAL_METHOD, fast-math and the like) and no contraction of these expressions, which the
   build's -ffp-contract=off gives (and which cannot happen without an FMA instruction in the target anyway).

   HOS_FMA_FAST=0 in the environment turns it off at run time (default on); tests/test_fma_soft.py compares both modes
   (recomp_fma_soft_mode: -1 reads the environment on first use, 0 off, 1 on). */
#if defined(RECOMP_FPX_FMA64) && defined(RECOMP_FPX_HOST) && RECOMP_FPX_HOST == 2 && !RECOMP_FPX_FMA64 && \
    (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#define RECOMP_FMA_SOFT 1
#include <stdlib.h>
#include <string.h>
#ifndef RECOMP_FMA_SOFT_PROBE
#define RECOMP_FMA_SOFT_PROBE(k) ((void)0) /* test builds count here: k 0 = binary32 taken, 1 = binary64 taken */
#endif
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

int recomp_fma_soft_mode = -1;
static int recomp_fma_soft_on(void) {
    int m = recomp_fma_soft_mode;
    if (m < 0) {
        const char* e = getenv("HOS_FMA_FAST");
        recomp_fma_soft_mode = m = (e && e[0] == '0') ? 0 : 1;
    }
    return m;
}

static inline double recomp_fma_soft_u2d(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static inline uint64_t recomp_fma_soft_d2u(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

/* the neighbour of s with an odd last bit, when s = RN(x) and e = x - s (exact, nonzero) says x was not s */
static inline double recomp_fma_soft_odd(double s, double e) {
    if (e != 0.0) {
        uint64_t sb = recomp_fma_soft_d2u(s);
        if (!(sb & 1)) sb += ((e > 0.0) == (s > 0.0)) ? 1 : (uint64_t)-1;
        return recomp_fma_soft_u2d(sb);
    }
    return s;
}

/* a*b + z of raw binary64 bits: 1 with the result bits in *v and whether it is inexact in *ix, or 0 for the exact body */
static inline int recomp_fma_soft64(uint64_t ab, uint64_t bb, uint64_t zb, uint64_t* v, int* ix) {
    const int ea = (int)((ab >> 52) & 0x7ff), eb = (int)((bb >> 52) & 0x7ff), ez = (int)((zb >> 52) & 0x7ff);
    const int ep = ea + eb - 1023;   /* the product's biased exponent, within one */
    if (ea < 400 || ea > 1600 || eb < 400 || eb > 1600) return 0;   /* Veltkamp's split of each cannot overflow */
    if (ep < 160 || ep > 1880) return 0;                            /* the product and its error term are normal */
    const int zzero = !(zb & UINT64_C(0x7fffffffffffffff));
    if (!zzero && (ez < 160 || ez > 1880)) return 0;
    const double a = recomp_fma_soft_u2d(ab), b = recomp_fma_soft_u2d(bb), z = recomp_fma_soft_u2d(zb);
    const double ph = a * b;
    const double S = 134217729.0;   /* 2^27 + 1 */
    const double ca = S * a, ah = ca - (ca - a), al = a - ah;
    const double cb = S * b, bh = cb - (cb - b), bl = b - bh;
    const double pl = ((ah * bh - ph) + ah * bl + al * bh) + al * bl;   /* a*b = ph + pl exactly */
    const double th = z + ph;
    const double t1 = th - z;
    const double tl = (z - (th - t1)) + (ph - t1);                      /* z + ph = th + tl exactly */
    const double s = pl + tl;
    const double t2 = s - pl;
    const double e = (pl - (s - t2)) + (tl - t2);                       /* pl + tl = s + e exactly */
    const double so = recomp_fma_soft_odd(s, e);
    const double r = th + so;
    const uint64_t rb = recomp_fma_soft_d2u(r);
    const int er = (int)((rb >> 52) & 0x7ff);
    if (er < 100 || er > 1900) return 0;                                /* zero, tiny or huge */
    const double t3 = r - th;
    const double err = (th - (r - t3)) + (so - t3);                     /* th + so = r + err exactly */
    if (e != 0.0 && !(recomp_fma_soft_d2u(th) & UINT64_C(0xfffffffffffff))) return 0;   /* (margin) a power of two with a rest */
    *ix = (err != 0.0) || (e != 0.0);
    *v = rb;
    return 1;
}

/* the same for raw binary32 bits in the low 32 */
static inline int recomp_fma_soft32(uint64_t ab, uint64_t bb, uint64_t zb, uint64_t* v, int* ix) {
    const uint32_t a32 = (uint32_t)ab, b32 = (uint32_t)bb, z32 = (uint32_t)zb;
    const int ea = (int)((a32 >> 23) & 0xff), eb = (int)((b32 >> 23) & 0xff), ez = (int)((z32 >> 23) & 0xff);
    if (ea < 40 || ea > 210 || eb < 40 || eb > 210) return 0;
    if ((z32 & 0x7fffffffu) && (ez < 40 || ez > 210)) return 0;
    float fa, fb, fz;
    memcpy(&fa, &a32, 4); memcpy(&fb, &b32, 4); memcpy(&fz, &z32, 4);
    const double p = (double)fa * (double)fb;   /* exact: 24 x 24 bits */
    const double z = (double)fz;
    const double s = p + z;
    const double t1 = s - p;
    const double e = (p - (s - t1)) + (z - t1);   /* p + z = s + e exactly */
    const double so = recomp_fma_soft_odd(s, e);
    const float r = (float)so;
    uint32_t rb;
    memcpy(&rb, &r, 4);
    const int er = (int)((rb >> 23) & 0xff);
    if (er < 2 || er > 253) return 0;
    *ix = (e != 0.0) || ((double)r != so);
    *v = rb;
    return 1;
}
#else
#define RECOMP_FMA_SOFT 0
#endif
