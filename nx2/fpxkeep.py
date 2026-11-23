"""(hookshot) More float results kept off the software bodies (A64_FPXKEEP, lift time): three rules, each proven exact.

FPX1 (a64_suyu.h's comment, third_party/a64-kit/lift/suyu/arm64_to_c.h) keeps a native result only when it is finite and
large, or a sum, or a product of a zero; a64_lift.py's wrappers add rescues when the flags an op could raise are already
set (a64_fpx_sticky32, a64_fpx_fz_add32). Everything else runs the exact body (recomp_fpe_*), multi-word integer code of
~180 ns a call (a few ns for a NaN). In a Market profile the bodies took 1.9 % of game code and the lane route of the
vector ops that end in them another 1.3 %; the cases measured there are the first and the third below:

  N  a NaN operand (FADD, FSUB, FMUL, FNMUL, FMADD family, FMLA lanes; S and D). Arm's result is a function of the
     operand bits alone (FPProcessNaNs, DN = 0 under the gate): the first signalling NaN in operand order, quieted, else
     the first quiet NaN; a multiply-add takes the addend first, and a quiet-NaN addend with an inf * 0 product gives the
     default NaN. Flags: IOC for a signalling NaN (or that inf * 0), IDC when FZ flushes a subnormal operand; nothing else
     (no rounding happens). Computed here with integer operations, the flags or-ed into FPSR. The Havok cloth's particle
     positions carry a quiet NaN in their unused w lane: vector ops on them went to the lanes and the bodies.
  I  an infinite operand and no NaN: the result is an exact infinity of the IEEE sign, with no flag, unless the
     operation is invalid (inf - inf; inf * 0, where under FZ a flushed subnormal is a zero), which is left to the body.
     Under FZ a subnormal operand raises IDC (it is flushed; beside an infinity that changes nothing else).
  Z  a multiply-add of nonzero finite multiplicands that cancels exactly gives +0 with no flag (IEEE and Arm, round to
     nearest). binary32: tested exactly in binary64 (the product exact, one rounding, and no nonzero exact sum is small
     enough to round to zero there). binary64: the host FMA's result +0 with normal multiplicands of biased exponents
     ea + eb >= 1076: a*b and z are then multiples of 2^-1074, so a nonzero exact sum rounds to a nonzero value and a
     native zero is the exact zero (one exponent less fails: a*b + z = 2^-1075 rounds to 0 with UFC); no host FMA, not
     taken. Under FZ no subnormal operand. The game's powf hits this on every call with x a power of two
     (r = z * invc - 1 = 0).

Where: in the outlined scalar wrappers (a64_fpx_add32_add32 ... a64_fpx_fma64_fma64, written by a64_lift.fpx_header),
past FPX1's fast path and the kit's rescues, before the body; and at the head of the kit's lane routes
(a64_vfpx_add32_lanes / _sub32_lanes / a64_vfpx_mul32_lanes, which fastvec's by-element FMUL also takes): the lanes N
or I keep get their result on SSE from the bits, the vector op runs again with zeros in those lanes, and the two are
merged (a64_fpxk_vspecial / a64_fpxk_vdone); with no such lane the lanes run one by one as before. The inline fast paths
(recomp_fpx_*, the hot copies, the vector fast checks) are not changed: the common case costs what it did, and the
lifted functions' text is the same. Inside a fast float region (nx2/fastfp.py) a NaN still sends the run to its exact
copy, which then takes these paths.

A64_FPXKEEP (lift time, an A64_ variable, so in the lift key): unset or 1 on; 0 off, a64_fpx.h as main writes it, to the
byte. Game-free tests: tests/test_fpxkeep.py (every rule against the bodies, with deliberately broken rules that must
fail). install() runs after nx2/fastvec.py's (it edits the lane function fastvec keeps) in nx2/lift.py; the lift's emit
processes run that module level again.
"""
import os

import a64_lift

# (each edit below must match exactly once: a refreshed a64-kit or fastvec that moved these lines stops the lift here)
WRAPPED = {("add32", "add32"): "a64_fpxk_sum32(c, a0, a1, 0, &v)", ("sub32", "sub32"): "a64_fpxk_sum32(c, a0, a1, 1, &v)",
           ("add64", "add64"): "a64_fpxk_sum64(c, a0, a1, 0, &v)", ("sub64", "sub64"): "a64_fpxk_sum64(c, a0, a1, 1, &v)",
           ("mul32", "fma32"): "a64_fpxk_fma32(c, a0, a1, a2, 1, &v)", ("fma32", "fma32"): "a64_fpxk_fma32(c, a0, a1, a2, 0, &v)",
           ("mul64", "fma64"): "a64_fpxk_fma64(c, a0, a1, a2, 1, &v)", ("fma64", "fma64"): "a64_fpxk_fma64(c, a0, a1, a2, 0, &v)"}
SUM_LANES = ("static __attribute__((noinline, cold)) void a64_vfpx_##name##_lanes(Ctx *c, uint64_t *d, const uint64_t *n, "
             "const uint64_t *m, int k) { \\\n")
SUM_LANES_NEW = ("static inline __attribute__((always_inline)) void a64_vfpx_##name(Ctx *c, uint64_t *d, const uint64_t *n, "
                 "const uint64_t *m, int k); \\\n" + SUM_LANES +
                 "    { uint64_t _ka[2], _kb[2], _kt[2]; a64_fpxk_v128 _ks, _kr; uint32_t _kf; \\\n"
                 "      if (a64_fpxk_vspecial(c, n, m, k, #name[0] == 's', _ka, _kb, &_ks, &_kr, &_kf)) { \\\n"
                 "          a64_vfpx_##name(c, _kt, _ka, _kb, k); a64_fpxk_vdone(c, d, _kt, _ks, _kr, _kf); return; } } \\\n")
MUL_LANES = ("static __attribute__((noinline, cold)) void a64_vfpx_mul32_lanes(Ctx *c, uint64_t *d, const uint64_t *n, "
             "const uint64_t *m) {\n")
MUL_LANES_NEW = ("static inline __attribute__((always_inline)) void a64_vfpx_mul32(Ctx *c, uint64_t *d, const uint64_t *n, "
                 "const uint64_t *m, int k);\n" + MUL_LANES +
                 "    { uint64_t _ka[2], _kb[2], _kt[2]; a64_fpxk_v128 _ks, _kr; uint32_t _kf;\n"
                 "      if (a64_fpxk_vspecial(c, n, m, 4, 2, _ka, _kb, &_ks, &_kr, &_kf)) {\n"
                 "          a64_vfpx_mul32(c, _kt, _ka, _kb, 4); a64_fpxk_vdone(c, d, _kt, _ks, _kr, _kf); return; } }\n")

HEADER = r"""
/* ---- (hookshot) results kept off the software bodies: NaN operands (N), infinite operands (I), exact-zero
   multiply-adds (Z); nx2/fpxkeep.py, A64_FPXKEEP. Each a64_fpxk_* is called inside FPX1's gate (RECOMP_FPX_OPEN: round
   to nearest, DN = 0, no traps, FPSR.IXC set) past its fast path; 1 with the Arm result in *v and its flags or-ed into
   FPSR, else 0 (nothing written: the exact body runs). ---- */
#ifndef A64_FPXK_PROBE
#define A64_FPXK_PROBE(k) ((void)0)   /* (tests/test_fpxkeep.py counts the results kept: 0 N, 1 I, 2 Z, 3 a lane route) */
#endif
#if RECOMP_FPX_HOST
#define A64_FPXK static inline __attribute__((always_inline))
/* binary32 / binary64 classes on the bits */
A64_FPXK int a64_fpxk_nan32(uint32_t x) { return (x & 0x7fffffffu) > 0x7f800000u; }
A64_FPXK int a64_fpxk_snan32(uint32_t x) { return a64_fpxk_nan32(x) && !(x & 0x00400000u); }
A64_FPXK int a64_fpxk_inf32(uint32_t x) { return (x & 0x7fffffffu) == 0x7f800000u; }
A64_FPXK int a64_fpxk_sub32(uint32_t x) { return (uint32_t)((x << 1) - 1u) < 0x00ffffffu; }
A64_FPXK int a64_fpxk_nan64(uint64_t x) { return (x & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000); }
A64_FPXK int a64_fpxk_snan64(uint64_t x) { return a64_fpxk_nan64(x) && !(x & UINT64_C(0x0008000000000000)); }
A64_FPXK int a64_fpxk_inf64(uint64_t x) { return (x & UINT64_C(0x7fffffffffffffff)) == UINT64_C(0x7ff0000000000000); }
A64_FPXK int a64_fpxk_sub64(uint64_t x) { return ((x << 1) - 1u) < UINT64_C(0x001fffffffffffff); }
/* N for two operands (FPProcessNaNs: op1 then op2, signalling first); *f gets IOC for a signalling one */
#define A64_FPXK_NAN2(W, Q) \
A64_FPXK uint64_t a64_fpxk_pick##W(uint64_t a, uint64_t b, uint32_t *f) { \
    if (a64_fpxk_snan##W(a)) { *f |= 1u; return a | (Q); } \
    if (a64_fpxk_snan##W(b)) { *f |= 1u; return b | (Q); } \
    return a64_fpxk_nan##W(a) ? a : b; }
A64_FPXK_NAN2(32, 0x00400000u)
A64_FPXK_NAN2(64, UINT64_C(0x0008000000000000))
/* FADD / FSUB (op 1: a - b; the NaN is chosen from a and b as they are, Arm does not negate a NaN in FSUB) */
#define A64_FPXK_SUM(W, T, SIGN, INF) \
A64_FPXK int a64_fpxk_sum##W(Ctx *c, uint64_t a0, uint64_t a1, int op, uint64_t *v) { \
    const T a = (T)a0, b = (T)a1; \
    const int fz = RECOMP_FPX_FZ(c) != 0; \
    uint32_t f = (fz && (a64_fpxk_sub##W(a) || a64_fpxk_sub##W(b))) ? 0x80u : 0u; \
    T r; \
    int rule = 0; \
    if (a64_fpxk_nan##W(a) || a64_fpxk_nan##W(b)) { \
        r = (T)a64_fpxk_pick##W(a, b, &f); \
    } else if (a64_fpxk_inf##W(a) || a64_fpxk_inf##W(b)) { \
        const T sb = op ? b ^ (SIGN) : b;   /* (the sign b enters the sum with) */ \
        if (a64_fpxk_inf##W(a) && a64_fpxk_inf##W(b) && ((a ^ sb) & (SIGN))) return 0;   /* inf - inf: invalid */ \
        r = a64_fpxk_inf##W(a) ? a : ((sb & (SIGN)) | (INF)); \
        rule = 1; \
    } else { \
        return 0; \
    } \
    A64_FPXK_PROBE(rule); \
    c->fpsr |= f; \
    *v = r; \
    return 1; }
A64_FPXK_SUM(32, uint32_t, 0x80000000u, 0x7f800000u)
A64_FPXK_SUM(64, uint64_t, UINT64_C(0x8000000000000000), UINT64_C(0x7ff0000000000000))
/* FMUL / FNMUL (z the product's signed zero) and the FMADD family / FMLA lanes (a*b + z, the signs the op negates
   already flipped by the lifted code): N, I, then Z. zero_: a zero after FZ's flush. */
#define A64_FPXK_FMA(W, T, SIGN, INF, QN, DNAN, ZERO) \
A64_FPXK int a64_fpxk_fma##W(Ctx *c, uint64_t a0, uint64_t a1, uint64_t a2, int mul, uint64_t *v) { \
    const T a = (T)a0, b = (T)a1, z = (T)a2; \
    const int fz = RECOMP_FPX_FZ(c) != 0; \
    const int sa = a64_fpxk_sub##W(a), sb = a64_fpxk_sub##W(b), sz = a64_fpxk_sub##W(z); \
    const int za = !(a & ~(SIGN)) || (fz && sa), zb = !(b & ~(SIGN)) || (fz && sb); \
    uint32_t f = (fz && (sa || sb || sz)) ? 0x80u : 0u; \
    T r; \
    int rule = 0; \
    if (a64_fpxk_nan##W(a) || a64_fpxk_nan##W(b) || a64_fpxk_nan##W(z)) { \
        if (a64_fpxk_nan##W(z) && !a64_fpxk_snan##W(z) && !a64_fpxk_nan##W(a) && !a64_fpxk_nan##W(b) && \
            ((a64_fpxk_inf##W(a) && zb) || (za && a64_fpxk_inf##W(b)))) { \
            f |= 1u; \
            r = (DNAN);   /* a quiet-NaN addend and inf * 0: the default NaN, IOC */ \
        } else if (a64_fpxk_snan##W(z)) { \
            f |= 1u; \
            r = z | (QN); \
        } else if (a64_fpxk_snan##W(a) || a64_fpxk_snan##W(b) || !a64_fpxk_nan##W(z)) { \
            r = (T)a64_fpxk_pick##W(a, b, &f); \
        } else { \
            r = z; \
        } \
    } else if (a64_fpxk_inf##W(a) || a64_fpxk_inf##W(b) || a64_fpxk_inf##W(z)) { \
        if (a64_fpxk_inf##W(a) || a64_fpxk_inf##W(b)) { \
            if (za || zb) return 0;   /* inf * 0: invalid */ \
            const T p = ((a ^ b) & (SIGN)) | (INF); \
            if (a64_fpxk_inf##W(z) && ((p ^ z) & (SIGN))) return 0;   /* inf - inf: invalid */ \
            r = p; \
        } else { \
            r = z; \
        } \
        rule = 1; \
    } else { \
        if (mul || !(a & ~(SIGN)) || !(b & ~(SIGN)) || (fz && (sa || sb || sz))) return 0; \
        if (!(ZERO)) return 0; \
        f = 0; \
        r = 0; \
        rule = 2; \
    } \
    A64_FPXK_PROBE(rule); \
    c->fpsr |= f; \
    *v = r; \
    return 1; }
/* Z: whether a*b + z is exactly zero (a, b nonzero finite, z finite). binary32 in binary64: the product of two binary32
   values is exact there and the sum is rounded once; a nonzero exact sum is a multiple of 2^-298 and far above binary64's
   smallest subnormal, so it cannot round to zero: the test is exact for every operand. binary64: the host's FMA rounds
   the exact sum once, and with normal multiplicands of biased exponents ea + eb >= 1076 the exact sum is a multiple of
   2^-1074 (a*b of 2^(ea+eb-2150), z of 2^-1074 or more), so a nonzero one rounds to a nonzero value; below that edge a
   sum of 2^-1075 rounds to zero with UFC, so it is not taken. Without a host FMA (RECOMP_FPX_FMA64 0) not taken. */
A64_FPXK int a64_fpxk_zero32(uint32_t a, uint32_t b, uint32_t z) {
    return (double)recomp_fpx_f(a) * (double)recomp_fpx_f(b) + (double)recomp_fpx_f(z) == 0.0;
}
A64_FPXK int a64_fpxk_zero64(uint64_t a, uint64_t b, uint64_t z) {
#if RECOMP_FPX_FMA64
    const int ea = (int)((a >> 52) & 0x7ff), eb = (int)((b >> 52) & 0x7ff);
    return ea && eb && ea + eb >= 1076 &&
           recomp_fpx_db(__builtin_fma(recomp_fpx_d(a), recomp_fpx_d(b), recomp_fpx_d(z))) == 0;
#else
    (void)a; (void)b; (void)z; return 0;
#endif
}
A64_FPXK_FMA(32, uint32_t, 0x80000000u, 0x7f800000u, 0x00400000u, 0x7fc00000u, a64_fpxk_zero32(a, b, z))
A64_FPXK_FMA(64, uint64_t, UINT64_C(0x8000000000000000), UINT64_C(0x7ff0000000000000), UINT64_C(0x0008000000000000),
             UINT64_C(0x7ff8000000000000), a64_fpxk_zero64(a, b, z))
#else
#define a64_fpxk_sum32(c, a, b, op, v) 0
#define a64_fpxk_sum64(c, a, b, op, v) 0
#define a64_fpxk_fma32(c, a, b, z, mul, v) 0
#define a64_fpxk_fma64(c, a, b, z, mul, v) 0
#endif
/* The lane routes (a64_vfpx_add32_lanes / _sub32_lanes / a64_vfpx_mul32_lanes, entered when a lane failed the vector
   check), at their head: the lanes N or I keep (op 0 FADD, 1 FSUB, 2 FMUL) get their Arm result from the bits, every
   other lane its operands as they were and the special lanes' a zero (0 + 0, 0 - 0, 0 * 0: exact, no flag), and the
   vector op runs again on those (inline: its fast check, or, where another lane fails, this lane route once more, which
   then finds no special lane and takes the lanes one by one as before). The results merged, the special lanes' flags
   (IOC for a signalling NaN, IDC for an operand FZ flushes) or-ed in after. 0 when no lane is special: the lanes run as
   before. 2 lanes: the top lanes are never special and come zero from the op. */
#if RECOMP_FPX_HOST == 2
typedef __m128i a64_fpxk_v128;
A64_FPXK int a64_fpxk_vspecial(const Ctx *c, const uint64_t *n, const uint64_t *m, int k, int op, uint64_t *a2, uint64_t *b2,
                               __m128i *sp, __m128i *res, uint32_t *flags) {
    if (!RECOMP_FPX_OPEN(c)) return 0;
    const __m128i a = _mm_loadu_si128((const __m128i *)n), b = _mm_loadu_si128((const __m128i *)m);
    const __m128i sgn = _mm_set1_epi32((int)0x80000000u), absm = _mm_set1_epi32(0x7fffffff), inf = _mm_set1_epi32(0x7f800000);
    const __m128i q = _mm_set1_epi32(0x00400000), zero = _mm_setzero_si128();
    const __m128i kmask = k == 4 ? _mm_set1_epi32(-1) : _mm_set_epi32(0, 0, -1, -1);
    const __m128i fzm = _mm_set1_epi32(-(int)(RECOMP_FPX_FZ(c) != 0));
    const __m128i aa = _mm_and_si128(a, absm), ab = _mm_and_si128(b, absm);   /* (0 .. 0x7fffffff: signed compares) */
    const __m128i nana = _mm_cmpgt_epi32(aa, inf), nanb = _mm_cmpgt_epi32(ab, inf);
    const __m128i infa = _mm_cmpeq_epi32(aa, inf), infb = _mm_cmpeq_epi32(ab, inf);
    const __m128i z0a = _mm_cmpeq_epi32(aa, zero), z0b = _mm_cmpeq_epi32(ab, zero);
    const __m128i minn = _mm_set1_epi32(0x00800000);
    const __m128i suba = _mm_and_si128(fzm, _mm_andnot_si128(z0a, _mm_cmplt_epi32(aa, minn)));   /* flushed by FZ */
    const __m128i subb = _mm_and_si128(fzm, _mm_andnot_si128(z0b, _mm_cmplt_epi32(ab, minn)));
    const __m128i nanl = _mm_or_si128(nana, nanb);
    __m128i infl, rinf;
    if (op == 2) {   /* inf * x: inf of the product's sign, unless x is a zero (or flushed to one): invalid */
        const __m128i zl = _mm_or_si128(_mm_or_si128(z0a, suba), _mm_or_si128(z0b, subb));
        infl = _mm_andnot_si128(_mm_or_si128(nanl, zl), _mm_or_si128(infa, infb));
        rinf = _mm_or_si128(_mm_and_si128(_mm_xor_si128(a, b), sgn), inf);
    } else {   /* inf +- x: that inf (b with the sign it enters with), unless both are infinities that cancel */
        const __m128i be = op ? _mm_xor_si128(b, sgn) : b;
        const __m128i cancel = _mm_and_si128(_mm_and_si128(infa, infb), _mm_srai_epi32(_mm_xor_si128(a, be), 31));
        infl = _mm_andnot_si128(_mm_or_si128(nanl, cancel), _mm_or_si128(infa, infb));
        rinf = _mm_or_si128(_mm_and_si128(infa, a), _mm_andnot_si128(infa, _mm_or_si128(_mm_and_si128(be, sgn), inf)));
    }
    const __m128i special = _mm_and_si128(_mm_or_si128(nanl, infl), kmask);
    if (!_mm_movemask_epi8(special)) return 0;
    /* FPProcessNaNs: a signalling a, else a signalling b, else a quiet a, else b; quieted */
    const __m128i sna = _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(a, q), q), nana);
    const __m128i snb = _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(b, q), q), nanb);
    const __m128i picka = _mm_or_si128(sna, _mm_andnot_si128(snb, nana));
    const __m128i nres = _mm_or_si128(_mm_or_si128(_mm_and_si128(picka, a), _mm_andnot_si128(picka, b)), q);
    *res = _mm_or_si128(_mm_and_si128(nanl, nres), _mm_andnot_si128(nanl, rinf));
    *sp = special;
    *flags = (_mm_movemask_epi8(_mm_and_si128(_mm_or_si128(sna, snb), special)) ? 1u : 0u) |
             (_mm_movemask_epi8(_mm_and_si128(_mm_or_si128(suba, subb), special)) ? 0x80u : 0u);
    _mm_storeu_si128((__m128i *)a2, _mm_andnot_si128(special, a));
    _mm_storeu_si128((__m128i *)b2, _mm_andnot_si128(special, b));
    return 1;
}
A64_FPXK void a64_fpxk_vdone(Ctx *c, uint64_t *d, const uint64_t *t, __m128i sp, __m128i res, uint32_t f) {
    const __m128i r = _mm_loadu_si128((const __m128i *)t);
    _mm_storeu_si128((__m128i *)d, _mm_or_si128(_mm_and_si128(sp, res), _mm_andnot_si128(sp, r)));
    c->fpsr |= f;
    A64_FPXK_PROBE(3);
}
#else
typedef struct { uint64_t x[2]; } a64_fpxk_v128;
#define a64_fpxk_vspecial(c, n, m, k, op, a2, b2, sp, res, flags) 0
#define a64_fpxk_vdone(c, d, t, sp, res, f) ((void)0)
#endif
"""


def on():
    """A64_FPXKEEP: on unless 0 (also off, false, no)"""
    v = os.environ.get("A64_FPXKEEP", "1").strip().lower()
    if v in ("", "1", "on", "true", "yes"):
        return True
    if v in ("0", "off", "false", "no"):
        return False
    raise SystemExit("A64_FPXKEEP=%s: 1 or 0" % os.environ.get("A64_FPXKEEP"))


def _once(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit("fpxkeep: %s found %d times (expected once): the kit or fastvec changed; update nx2/fpxkeep.py"
                         % (what, text.count(old)))
    return text.replace(old, new)


def keep_wrappers(text, used):
    """a64_fpx.h's text with HEADER after the kit's FPX_FZ_HEADER and the rescues in the wrappers of WRAPPED"""
    text = _once(text, a64_lift.FPX_FZ_HEADER, a64_lift.FPX_FZ_HEADER + HEADER, "the kit's FPX_FZ_HEADER")
    expected = {shape[:2] for shape in used} & WRAPPED.keys()
    counts = {shape: 0 for shape in expected}
    out = []
    for line in text.split("\n"):
        if line.startswith("A64_FPX_FN uint64_t a64_fpx_"):
            fast, exact = line[len("A64_FPX_FN uint64_t a64_fpx_"):].split("(", 1)[0].split("_")[:2]
            mine = WRAPPED.get((fast, exact))
            if mine:
                if (fast, exact) not in counts:
                    raise SystemExit("fpxkeep: unexpected wrapper a64_fpx_%s_%s" % (fast, exact))
                line = _once(line, "} return recomp_fpe_", "if (%s) return v; } return recomp_fpe_" % mine,
                             "the wrapper a64_fpx_%s_%s's end" % (fast, exact))
                counts[fast, exact] += 1
        out.append(line)
    for (fast, exact), count in sorted(counts.items()):
        if count != 1:
            raise SystemExit("fpxkeep: wrapper a64_fpx_%s_%s edited %d times (expected once)" % (fast, exact, count))
    return "\n".join(out)


_installed = False


def install():
    """the kit's lane routes and fpx_header edited (when A64_FPXKEEP is on); after nx2/fastvec.py's install. Once"""
    global _installed
    if _installed:
        return
    _installed = True
    if not on():
        return
    a64_lift.VFPX_HEADER = _once(a64_lift.VFPX_HEADER, SUM_LANES, SUM_LANES_NEW, "the kit's a64_vfpx_##name##_lanes")
    a64_lift.VFPX_MUL_HEADER = _once(a64_lift.VFPX_MUL_HEADER, MUL_LANES, MUL_LANES_NEW, "the kit's a64_vfpx_mul32_lanes")
    base_header = a64_lift.fpx_header

    def fpx_header(used):
        return keep_wrappers(base_header(used), used)

    a64_lift.fpx_header = fpx_header
