/* Steady sun shadows near Link: the nearest of the sun's cascades fitted so that it no longer moves with the camera.
   nx2/oot_shadow_fit.py calls these two after the game's light basis for a cascade (main+0x713620) and in its box fit
   (main+0x713be0).

   Every frame the game refits each cascade to the view: a light basis whose sideways axis follows the camera's
   direction, then the tightest box around the cascade's slice of the view, in that basis. Turning the camera turns the
   shadow map under the scene, and walking slides and resizes the box by fractions of a texel, so the edges of thin
   shadows near Link (fences, chains, ivy, grass) shimmer and crawl. For the nearest cascade (the slice from 0.1 m to
   about 10 m) the hooks make:
   - the basis fixed in the world, whichever way the camera looks: row 2 is the sun's direction as the game made it
     (its origin too, so the depth range is the game's), and rows 0 and 1 are carried from one fit to the next by the
     smallest turn that follows the sun. The sun moves (about 0.45 degrees a second in Hyrule Field when the day runs),
     and rows rebuilt from the world's up each fit would also spin about the sun as its bearing changes, sliding the map
     under the scene twice as fast as the game's own fit while Link stands still. Rows 0 and 1 measure from an anchor,
     a point in the world kept at the middle of the slice's depth along the view, which the map turns about as the sun
     moves: from the world's origin, a hundred metres and more away, the map would slide most of a texel a frame. The
     anchor follows by whole texels of the map along rows 0 and 1, and freely along the sun's direction, so its moves
     don't move the map's texels. When the sun jumps (more than a degree between two fits), rows 0 and 1 start again
     from the world's up;
   - the box a square of one size, which holds the slice's bounding sphere and so the slice whichever way the camera
     turns. The size grows at once when the slice needs more, and shrinks only once it has needed a fifth less for a
     second and a half (90 fits): the slice's depth wanders by about a tenth while the camera turns and walks (9.8 to
     10.8 m in Hyrule Field), and that keeps one size;
   - its centre on whole texels of the map, so the map's texels stay on the same places in the world while the camera
     moves.
   The game fits cascade 0's box from two places, on whichever worker thread: right after the basis (main+0x713230), and
   from main+0x712b9c, the one that makes the frame's box while playing (traced in Hyrule Field). So the box hook needs
   nothing from the basis hook's call: it acts on a cascade whose basis rows 0 and 1 are exactly the ones the basis hook
   last stored. The box holds the game's own, so every caster and receiver the game's fit covered is still
   covered. The other cascades and every other light keep the game's fit.

   Only on Shadow resolution Highest (a map of 3072 texels or more, host/oot_shadow.c): the square holds the slice
   whichever way the camera turns, so it is larger than the game's box standing still, and on the game's 1600 texels
   thin shadows such as chains are visibly softer standing still at 32:9 (2.4 cm a texel against about 2.0 x 1.1 cm);
   at 3072 they are about as sharp as the game's own fit at 1600. Below Highest the game's fit is kept. Changing the
   size starts the fit afresh. HOS_SHADOW_STABLE=0: the game's fit at every size, as shipped; HOS_SHADOW_STABLE=1: the
   steady fit at every size. */
#include "hos.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>   /* SRWLOCK (on Linux host/posix's) */

#define FIT_MAIN 0x80020000ULL
#define FIT_RESET 0.99985   /* the cosine of a degree: a bigger turn of the sun between two fits starts the rows afresh */
#define FIT_MIN_SIZE 3072   /* Shadow resolution Highest's map: the smallest the steady fit runs at by default */

/* Kept between cascade 0's fits, under fit_lock. The game fits cascade 0's basis and box on whichever worker thread; a
   trace in Hyrule Field showed one fit after another, never two at once, but nothing in the game promises it (a second
   sun view, another threading of the fit would tear the carried rows and the held size). So each hook holds fit_lock
   while it reads or writes these and the mode below. The walk that finds the sun's cascade 0 reads guest memory alone
   and comes before it: the other cascades and lights never take the lock, and cascade 0's hooks take it once each,
   with nothing to wait for while the fits come one after another. */
static SRWLOCK fit_lock = SRWLOCK_INIT;
static double held;             /* the box's half size kept, in the light's units (metres); 0 before the first fit */
static int under;               /* fits in a row that could have used a box a fifth smaller */
static double anchor[3];        /* the point rows 0 and 1 measure from (metres, world) */
static double axes[2][3];       /* rows 0 and 1, carried from fit to fit */
static double sun_was[3];       /* row 2 at the last fit */
static int carried;             /* axes, sun_was and anchor hold a fit; 0 before the first and after the sun overhead */
static float stored[2][4];      /* rows 0 and 1 with their origins, as the basis hook last stored them */
static int stored_any;
static uint32_t size_was;       /* the map's size at cascade 0's last fit, steady or not; 0 before the first */

/* HOS_SHADOW_STABLE: 0 the game's fit at every size, 2 (set to 1) the steady fit at every size, 1 (unset) the steady fit
   on Shadow resolution Highest only. Under fit_lock. */
static int fit_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = getenv("HOS_SHADOW_STABLE");
        if (e && (!strcmp(e, "0") || !strcmp(e, "false") || !strcmp(e, "off") || !strcmp(e, "no"))) mode = 0;
        else if (e && (!strcmp(e, "1") || !strcmp(e, "true") || !strcmp(e, "on") || !strcmp(e, "yes"))) mode = 2;
        else mode = 1;
        if (mode != 1)
            hos_log("shadow: the sun's nearest cascade %s", mode ? "is kept steady at every size (HOS_SHADOW_STABLE=1)"
                                                                 : "keeps the game's fit (HOS_SHADOW_STABLE=0)");
    }
    return mode;
}

static uint64_t linked(Ctx *c, uint64_t at, uint64_t vtable) {
    const uint64_t o = recomp_load64(c, at);
    return o && recomp_load64(c, o) == vtable ? o : 0;
}

static float load_f(Ctx *c, uint64_t a) {
    const uint32_t u = (uint32_t)recomp_load32(c, a);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void store_f(Ctx *c, uint64_t a, float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    recomp_store32(c, a, u);
}

/* The map's size (texels a side) when cascade is the sun's cascade 0 on the plain orthographic path, else 0. Guest
   memory alone, none of the state above: before fit_lock. */
static uint32_t sun_cascade0(Ctx *c, uint64_t cascade) {
    if (!cascade) return 0;
    const uint64_t scene = linked(c, FIT_MAIN + 0x4e3e258, FIT_MAIN + 0x4b895b0);
    const uint64_t renderer = scene ? linked(c, scene + 0x1e8, FIT_MAIN + 0x4b8bc00) : 0;
    const uint64_t sun = renderer ? linked(c, renderer + 0xb30, FIT_MAIN + 0x4b8a0a8) : 0;   /* the sun's shadow object */
    const uint64_t state = sun ? recomp_load64(c, sun + 8) : 0;
    const uint64_t view = state ? recomp_load64(c, state + 0xd60) : 0;
    if (!view || recomp_load64(c, view + 0x390) != cascade) return 0;   /* the sun's view, its cascade 0 */
    /* the warp after the basis (cascade+0x188): none but the depth axis's sign, as the nearest cascade has */
    static const float plain[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1};
    for (int i = 0; i < 16; i++)
        if (load_f(c, cascade + 0x188 + 4 * i) != plain[i]) return 0;
    const uint32_t size = (uint32_t)recomp_load32(c, sun + 0xe0);
    if (!(size >= 64 && size <= 16384 && size % 2 == 0)) return 0;
    return size;
}

/* Under fit_lock, the size of the sun's cascade 0's map: that size when the steady fit runs at it, else 0 (the game's
   fit). */
static uint32_t fit_size(uint32_t size) {
    if (!fit_mode()) return 0;
    if (size != size_was) {   /* a new size (Shadow resolution changed): the fit starts afresh, the game's below Highest */
        held = 0.0, under = 0, carried = 0, stored_any = 0;
        if (fit_mode() == 1)
            hos_log("shadow: the sun's nearest cascade %s (%u texels)",
                    size >= FIT_MIN_SIZE ? "is kept steady" : "keeps the game's fit below Shadow resolution Highest", size);
        size_was = size;
    }
    return fit_mode() == 2 || size >= FIT_MIN_SIZE ? size : 0;
}

/* The slice's bounding sphere, from the cascade's slice (+0x144 near, +0x154 far) and the view's projection (+0x318):
   its radius, and how far along the view axis its centre is (on that axis, at the far plane when that is where its
   middle falls). 0 when they don't make a slice. */
static int slice_sphere(Ctx *c, uint64_t cascade, double *r, double *along) {
    const double nr = load_f(c, cascade + 0x144), fr = load_f(c, cascade + 0x154);
    const double p00 = load_f(c, cascade + 0x318), p11 = load_f(c, cascade + 0x318 + 0x14);
    if (!(fr > nr && nr >= 0.0 && fr < 1e4 && p00 > 1e-3 && p11 > 1e-3)) return 0;
    const double k2 = 1.0 / (p00 * p00) + 1.0 / (p11 * p11);
    const double mid = 0.5 * (nr + fr) * (1.0 + k2);
    *along = mid >= fr ? fr : mid;
    *r = mid >= fr ? fr * sqrt(k2) : sqrt((fr - mid) * (fr - mid) + fr * fr * k2);
    return 1;
}

/* Under fit_lock: rows 0 and 1 of cascade 0's basis (cascade+0x158, three rows of four) become the carried rows, measured
   from the anchor. */
static void fit_basis(Ctx *c, uint64_t cascade, uint32_t size) {
    double z[3];
    for (int j = 0; j < 3; j++) z[j] = load_f(c, cascade + 0x178 + 4 * j);
    const double zl = z[0] * z[0] + z[1] * z[1] + z[2] * z[2];
    if (!(fabs(zl - 1.0) <= 1e-3)) return;   /* row 2 isn't the sun's direction */
    for (int j = 0; j < 3; j++) z[j] /= sqrt(zl);
    /* the middle of the slice's depth (+0x144 near, +0x154 far) along the view, in the world: the camera's eye and its
       forward (-Z) from its view (cascade+0x2e8, three rows of four) */
    const double nr = load_f(c, cascade + 0x144), fr = load_f(c, cascade + 0x154);
    if (!(fr > nr && nr >= 0.0 && fr < 1e4)) return;
    double v[12], target[3];
    for (int i = 0; i < 12; i++) v[i] = load_f(c, cascade + 0x2e8 + 4 * i);
    for (int j = 0; j < 3; j++) {
        target[j] = -(v[j] * v[3] + v[4 + j] * v[7] + v[8 + j] * v[11]) - v[8 + j] * 0.5 * (nr + fr);
        if (!(fabs(target[j]) < 1e6)) return;
    }
    double x[3];
    const double turn = sun_was[0] * z[0] + sun_was[1] * z[1] + sun_was[2] * z[2];
    const int fresh = !carried || !(turn >= FIT_RESET);
    if (fresh) {   /* level: the world's up (Y) x the sun */
        const double h = z[0] * z[0] + z[2] * z[2];
        if (!(h > 1e-4)) { carried = 0; return; }   /* the sun straight overhead: the game's basis */
        x[0] = z[2] / sqrt(h), x[1] = 0.0, x[2] = -z[0] / sqrt(h);
    } else {       /* row 0 turned as the sun turned, about the axis square to both (Rodrigues), then kept square to it */
        double k[3] = {sun_was[1] * z[2] - sun_was[2] * z[1], sun_was[2] * z[0] - sun_was[0] * z[2],
                       sun_was[0] * z[1] - sun_was[1] * z[0]};
        const double sn = sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
        memcpy(x, axes[0], sizeof x);
        if (sn > 1e-12) {
            for (int j = 0; j < 3; j++) k[j] /= sn;
            const double kx[3] = {k[1] * x[2] - k[2] * x[1], k[2] * x[0] - k[0] * x[2], k[0] * x[1] - k[1] * x[0]};
            const double kd = k[0] * x[0] + k[1] * x[1] + k[2] * x[2];
            for (int j = 0; j < 3; j++) x[j] = x[j] * turn + kx[j] * sn + k[j] * kd * (1.0 - turn);
        }
        const double xz = x[0] * z[0] + x[1] * z[1] + x[2] * z[2];
        for (int j = 0; j < 3; j++) x[j] -= xz * z[j];
        const double xl = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        if (!(xl > 0.5)) { carried = 0; return; }
        for (int j = 0; j < 3; j++) x[j] /= xl;
    }
    const double y[3] = {z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]};
    float rows[2][3];
    for (int j = 0; j < 3; j++) {
        axes[0][j] = x[j], axes[1][j] = y[j], sun_was[j] = z[j];
        rows[0][j] = (float)x[j], rows[1][j] = (float)y[j];
    }
    /* the anchor follows the slice's middle: along rows 0 and 1 by whole texels of the map, so the map's texels stay
       where they were, and along the sun's row (2) all the way, which moves no texel */
    if (fresh || held <= 0.0) {
        memcpy(anchor, target, sizeof anchor);
    } else {
        const double d[3] = {target[0] - anchor[0], target[1] - anchor[1], target[2] - anchor[2]};
        const double texel = 2.0 * held / size;
        for (int k = 0; k < 2; k++) {
            const double s = floor((rows[k][0] * d[0] + rows[k][1] * d[1] + rows[k][2] * d[2]) / texel + 0.5) * texel;
            for (int j = 0; j < 3; j++) anchor[j] += s * rows[k][j];
        }
        const double s = z[0] * d[0] + z[1] * d[1] + z[2] * d[2];
        for (int j = 0; j < 3; j++) anchor[j] += s * z[j];
    }
    carried = 1;
    const uint64_t at = cascade + 0x158;
    for (int k = 0; k < 2; k++) {
        double t = 0.0;
        for (int j = 0; j < 3; j++) {
            store_f(c, at + 16 * k + 4 * j, rows[k][j]);
            stored[k][j] = rows[k][j];
            t -= rows[k][j] * anchor[j];
        }
        store_f(c, at + 16 * k + 12, (float)t);
        stored[k][3] = (float)t;
    }
    stored_any = 1;
}

/* After the game's light basis for a cascade (main+0x713620 returned, x19 the cascade): fit_basis when it is the sun's
   cascade 0 and the steady fit runs at its size. */
void hos_shadow_fit_basis(Ctx *c, uint64_t cascade) {
    uint32_t size = sun_cascade0(c, cascade);
    if (!size) return;
    AcquireSRWLockExclusive(&fit_lock);
    if ((size = fit_size(size))) fit_basis(c, cascade, size);
    ReleaseSRWLockExclusive(&fit_lock);
}

/* Under fit_lock: b made the steady square when cascade 0 has the basis hook's rows. 1 when b changed. */
static int fit_box(Ctx *c, uint64_t cascade, uint32_t size, float b[4]) {
    if (!stored_any) return 0;
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 4; j++)
            if (load_f(c, cascade + 0x158 + 16 * k + 4 * j) != stored[k][j]) return 0;   /* the game's basis: its box */
    double r, along;
    if (!slice_sphere(c, cascade, &r, &along)) return 0;
    if (!(b[1] >= b[0] && b[3] >= b[2])) return 0;
    double need = r;
    if ((b[1] - b[0]) * 0.5 > need) need = (b[1] - b[0]) * 0.5;
    if ((b[3] - b[2]) * 0.5 > need) need = (b[3] - b[2]) * 0.5;
    const double q = ceil(need * 1.01 / 0.5) * 0.5;   /* half-metre steps, a little room */
    if (!(q > 0.0 && q < 1e4)) return 0;
    if (q > held) {
        held = q, under = 0;
        hos_log("shadow: the sun's nearest cascade is %.1f m a side (%u texels)", 2.0 * held, size);
    } else if (q <= held * 0.8) {
        if (++under >= 90) {
            held = q, under = 0;
            hos_log("shadow: the sun's nearest cascade is %.1f m a side (%u texels)", 2.0 * held, size);
        }
    } else {
        under = 0;
    }
    const double texel = 2.0 * held / size;
    const double cx = floor(0.5 * (b[0] + b[1]) / texel + 0.5) * texel;
    const double cy = floor(0.5 * (b[2] + b[3]) / texel + 0.5) * texel;
    b[0] = (float)(cx - held), b[1] = (float)(cx + held);
    b[2] = (float)(cy - held), b[3] = (float)(cy + held);
    return 1;
}

/* In the box fit, after the slice's bounds in the light's basis are found, where the box fit's two modes meet
   (main+0x713efc, x20 the cascade): b = {min x, max x, min y, max y}, made the steady square when the cascade is the
   sun's cascade 0 with the basis hook's rows. 1 when b changed. */
int hos_shadow_fit_box(Ctx *c, uint64_t cascade, float b[4]) {
    uint32_t size = sun_cascade0(c, cascade);
    if (!size) return 0;
    AcquireSRWLockExclusive(&fit_lock);
    size = fit_size(size);
    const int changed = size && fit_box(c, cascade, size, b);
    ReleaseSRWLockExclusive(&fit_lock);
    return changed;
}
