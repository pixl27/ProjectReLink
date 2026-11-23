"""Steady sun shadows near Link (OoT 1.1.1): the nearest sun cascade fitted so that it no longer moves with the camera.

The game refits its sun cascades every frame (main+0x713120, once per cascade): it builds a light basis whose sideways
axis follows the camera (main+0x713620), then the tightest box around the cascade's slice of the view in that basis
(main+0x713be0, called right after and again from main+0x712b9c, the call that makes the frame's box while playing).
Turning the camera turns the shadow map under the scene and walking slides and resizes the box by fractions of a texel:
thin shadows near Link shimmer and crawl. Two sites (host/oot_shadow_fit.c says what they do):

- after the basis is built (the instruction after the call, x19 the cascade): the sun's cascade 0 gets a basis that
  doesn't follow the camera, its rows carried from fit to fit as the sun turns;
- in the box fit, once the slice's bounds are found and checked (s5/s4 min/max x, s2/s3 min/max y; x20 the cascade),
  before the projection is made from them, where the box fit's two modes meet: those bounds become a square of steady
  size on whole texels.

Every value the game makes from the bounds afterwards (the projection, the texture matrix, the constant buffers) follows
them. HOS_SHADOW_STABLE=0 turns it off.
"""

MAIN = 0x80020000
BUILD_ID = "e0f3ccfaf50065ae4ea01eb5aeb13b1a45da3348"

# The call to the basis builder, then fmov s0, s9; fmov s1, s8; add x0, x19, #0x80 (the slice's arguments).
BASIS = MAIN + 0x71318c
BASIS_WORDS = (MAIN + 0x713188, (0x94000126, 0x1e204120, 0x1e204101, 0x91020260))
# cmp w24, #2; b.eq (the other mode); mov w9, wzr; ldp s7, s6, [x21]; fsub s18, s4, s5; fsub s17, s3, s2; fsub s16, s1, s0.
# The site is the ldp, where both modes meet: the other mode (w24 == 2) loads w9 and comes back to it, past the mov. The
# sun's cascade 0 was traced in the first (w24 == 1); the site serves both.
BOX = MAIN + 0x713efc
BOX_WORDS = (MAIN + 0x713ef0, (0x71000b1f, 0x54007840, 0x2a1f03e9, 0x2d401aa7, 0x1e253892, 0x1e223871, 0x1e203830))
# The other mode, b.eq's target: ldrb w9, [x20, #0x614]; b (the ldp)
REJOIN_WORDS = (MAIN + 0x714dfc, (0x39585289, 0x17fffc3f))

BASIS_HOOK = ("{ extern void hos_shadow_fit_basis(Ctx *c, uint64_t cascade); hos_shadow_fit_basis(c, c->x[19]); }"
              "   /* (nx2/oot_shadow_fit.py: the sun's nearest cascade, a world-fixed basis) */")
_REGS = ((0, 5), (1, 4), (2, 2), (3, 3))   # (b[i], s register): min x, max x, min y, max y
BOX_HOOK = ("{ extern int hos_shadow_fit_box(Ctx *c, uint64_t cascade, float b[4]); float _b[4]; "
            + "".join("memcpy(&_b[%d], c->vreg[%d], 4); " % r for r in _REGS)
            + "if (hos_shadow_fit_box(c, c->x[20], _b)) { "
            + "".join("{ uint32_t _u; memcpy(&_u, &_b[%d], 4); a64_vset64(c->vreg[%d], _u); } " % r for r in _REGS)
            + "} }   /* (nx2/oot_shadow_fit.py: the sun's nearest cascade, a steady square on whole texels) */")


def install(lifter, img, build_id):
    def same(start, words):
        try:
            return all(img.word(start + 4 * i) == w for i, w in enumerate(words))
        except Exception:
            return False
    if build_id != BUILD_ID or not all(same(*w) for w in (BASIS_WORDS, BOX_WORDS, REJOIN_WORDS)):
        print("steady sun shadows: unsupported cascade fit, left out")
        return 0
    for site, text in ((BASIS, BASIS_HOOK), (BOX, BOX_HOOK)):
        lifter.after[site] = (lifter.after[site] + "\n" if site in lifter.after else "") + text
    print("steady sun shadows: the nearest cascade's fit hooked")
    return 2
