/* (hookshot) The neighbouring map the OptimizeChangeBank thread preloads, kept while the game's choice only flickers.
   nx2/oot_patches.py hooks the place; host/oot_ocb_hold.c holds the state and reads the switch.

   OptimizeChangeBancMgr (vtable 0x84bc1a28) preloads one neighbouring map at a time: its layouts, then every actor pack
   and model in them, set up on its own thread (f_8002f7a0). Its per-frame update (f_832b9880) picks that map among up to 16
   units the current map lists (+0x408, 0x128 bytes each: +0 in use, +1 asked for, +4 position, +0x10 the layout's path,
   +0x120 LoadRadius, +0x124 priority), from Link's position (+0x1ce8) and velocity (+0x1cf4): of the units whose radius he is
   in, the one he moves towards (largest dot product) while he moves faster than 4 or for a while after (+0x1d38), else
   the nearest. Then it asks for that unit (f_832ba540) and lets go of every other one asked for before (f_832ba400).
   Asking for another map than the one loading or loaded (+0x70; the next one at +0x1e8) cancels it (state +0x2c = 5): the
   thread lets go of everything it set up, and then loads the new one from the start. When the choice flips back, the first
   map is read, decompressed and set up again. So a choice that flickers between two neighbours, or in and out of a radius,
   reloads them over and over with Link standing or circling on one spot: the loaders' ~0.8 of a core in the 4-core fountain
   basin.

   With a hold, a choice that names another map than the one held (the one asked for: the next one, else the one loading
   or loaded) is taken only once it has stayed the choice that long; until then the update sees the held map chosen
   (asked for again: nothing happens; the others let go as before). Another unit naming the held map is the same preload
   and taken at once, and a choice held back goes on with the unit that asked for the held map (oot_ocb_held). A first
   choice (nothing held), a direct request (+0x28) and the game's other mode (+0x1d42, two units in turn) are the game's
   own.

   What "stayed the choice" means: the same manager, the same unit AND the same map in it (its layout's path), seen by
   every update without a gap longer than OCB_GAP_NS. A unit whose slot now holds another map (the map changed and the
   slot was filled again), another manager, or a pause in the updates (a loading screen, a freeze) starts its time again:
   a choice never inherits the time of the slot's earlier occupant.

   Every string read here is the manager's own: the constructor (f_80d32c00) points each path at a buffer inside the
   manager (the current one +0x80, the next one +0x1f4, each unit's +0x1c, 0x100 bytes each) and nothing repoints them.
   Before reading, oot_ocb_layout checks that this is so (with the vtable, which names the build and the class); the
   reads then stay inside the manager the update is running on, and follow no pointer the game could have freed. Any
   mismatch: the game's own choice at that update.

   hold_ns: 0 or less, the game's own choice at every frame. now_ns: any clock that does not go back. */
#ifndef OOT_OCB_HOLD_H
#define OOT_OCB_HOLD_H
#include <stdint.h>

/* The manager's layout, as f_80d32c00 builds it and f_832b9880 / f_832ba540 / f_832ba400 read it (build e0f3ccfa) */
enum {
    OCB_DIRECT = 0x28, OCB_STATE = 0x2c, OCB_CURRENT = 0x70, OCB_CURRENT_LEN = 0x78, OCB_CURRENT_CAP = 0x7c,
    OCB_CURRENT_BUF = 0x80, OCB_PENDING = 0x1e8, OCB_PENDING_CAP = 0x1f0, OCB_PENDING_BUF = 0x1f4, OCB_SLOTS = 0x408,
    OCB_SLOT_SIZE = 0x128, OCB_SLOT_ASKED = 0x1, OCB_SLOT_PATH = 0x10, OCB_SLOT_CAP = 0x18, OCB_SLOT_BUF = 0x1c,
    OCB_UNITS = 16, OCB_PATH_MAX = 0x100, OCB_STATE_MAX = 7
};
#define OCB_VTABLE 0x84bc1a28ULL
#define OCB_GAP_NS 500000000LL   /* (updates further apart than this: a break in the choice) */
#define OCB_HOLD_MAX_MS 100000000LL

typedef struct {
    int cand;            /* (the choice that differs from the held map, -2: none) */
    uint64_t cand_key;   /* (its identity: the manager, the unit and the map in it) */
    int64_t since_ns;    /* (since when it has been the choice) */
    int64_t seen_ns;     /* (the last update that saw it) */
    long long held_back, taken, refused;   /* (other choices held back, taken after the hold; updates refused) */
} OotOcbHold;
#define OOT_OCB_HOLD_INIT {-2, 0, 0, 0, 0, 0, 0}

/* The unit the update goes on with: chosen (-1: none in range; key its identity, oot_ocb_key) or held (-1: nothing held) */
static inline int oot_ocb_pick(OotOcbHold *s, int chosen, uint64_t key, int held, int64_t hold_ns, int64_t now_ns) {
    if (hold_ns <= 0 || held < 0 || chosen == held) {
        s->cand = -2;
        return chosen;
    }
    if (s->cand != chosen || s->cand_key != key || now_ns - s->seen_ns > OCB_GAP_NS)
        s->cand = chosen, s->cand_key = key, s->since_ns = now_ns, s->held_back++;
    s->seen_ns = now_ns;
    if (now_ns - s->since_ns >= hold_ns) {
        s->cand = -2, s->taken++;
        return chosen;
    }
    return held;
}

/* The switch's value in ms: not set or empty, dflt_ms; a whole number of ms (at most OCB_HOLD_MAX_MS); 0, a negative
   whole number or "off": 0, the game's own choice; anything else, dflt_ms. Never below 0. */
static inline int64_t oot_ocb_hold_ms(const char *value, int64_t dflt_ms) {
    if (value && *value) {
        if (value[0] == 'o' && value[1] == 'f' && value[2] == 'f' && !value[3]) return 0;
        const char *p = value + (value[0] == '-');
        int64_t v = 0;
        int digits = 0;
        for (; *p >= '0' && *p <= '9'; p++, digits++)
            if (v < OCB_HOLD_MAX_MS) v = v * 10 + (*p - '0');
        if (digits && !*p) return value[0] == '-' ? 0 : v < OCB_HOLD_MAX_MS ? v : OCB_HOLD_MAX_MS;
    }
    return dflt_ms < 0 ? 0 : dflt_ms;
}

/* Guest reads (size 1, 4 or 8 bytes) for the functions below: the lifted code's loads in the game, a buffer in the test */
typedef uint64_t (*OotOcbLoad)(void *ctx, uint64_t addr, int size);

/* 1 when mgr is the manager as this file knows it: its vtable, each path pointing at the manager's own buffer of 0x100
   bytes, a state the game has, in-use flags of 0 or 1. 0: the game's own choice */
static inline int oot_ocb_layout(OotOcbLoad ld, void *ctx, uint64_t mgr) {
    if (!mgr || (mgr & 7) || ld(ctx, mgr, 8) != OCB_VTABLE) return 0;
    if (ld(ctx, mgr + OCB_CURRENT, 8) != mgr + OCB_CURRENT_BUF || ld(ctx, mgr + OCB_CURRENT_CAP, 4) != OCB_PATH_MAX ||
        ld(ctx, mgr + OCB_CURRENT_LEN, 4) >= OCB_PATH_MAX)
        return 0;
    if (ld(ctx, mgr + OCB_PENDING, 8) != mgr + OCB_PENDING_BUF || ld(ctx, mgr + OCB_PENDING_CAP, 4) != OCB_PATH_MAX)
        return 0;
    if (ld(ctx, mgr + OCB_STATE, 4) > OCB_STATE_MAX) return 0;
    for (int k = 0; k < OCB_UNITS; k++) {
        const uint64_t slot = mgr + OCB_SLOTS + (uint64_t)OCB_SLOT_SIZE * k;
        const uint64_t used = ld(ctx, slot, 1);
        if (used > 1) return 0;
        if (used && (ld(ctx, slot + OCB_SLOT_PATH, 8) != slot + OCB_SLOT_BUF ||
                     ld(ctx, slot + OCB_SLOT_CAP, 4) != OCB_PATH_MAX))
            return 0;
    }
    return 1;
}

static inline int oot_ocb_same(OotOcbLoad ld, void *ctx, uint64_t a, uint64_t b) {
    for (int i = 0; i < OCB_PATH_MAX; i++) {
        const uint64_t x = ld(ctx, a + i, 1);
        if (x != ld(ctx, b + i, 1)) return 0;
        if (!x) return 1;
    }
    return 1;
}

/* The unit (0-15) whose map the manager mgr goes on with: the one asked for next (+0x1e8), else, while it loads or is
   loaded (state 1-4), the current one (+0x70); -1 when none, or a direct request (+0x28) is in. Read from the manager's
   own buffers: oot_ocb_layout first.
   A map can be listed by more than one unit (Hyrule Field lists Gerudo Valley twice: the Lake Hylia side and the valley's
   gate), and the update lets go of every unit it did not choose whose +1 is set (f_832ba400, by the unit's path): letting
   go of any unit naming the held map cancels its load. So of the units in use naming it: chosen (the update's unit, any
   negative value none) when it is one, the same preload; else the one asked for (+1), whose request the load is; else
   the first. Taking the first one where the choice names the map held that choice back for another unit of the same map,
   and the update then cancelled the very load it had asked for, at every other update (Gerudo Valley's ~250 files read
   again every ~76 ms at Lon Lon Ranch) */
static inline int oot_ocb_held(OotOcbLoad ld, void *ctx, uint64_t mgr, int chosen) {
    if (ld(ctx, mgr + OCB_DIRECT, 1) & 1) return -1;
    const uint32_t state = (uint32_t)ld(ctx, mgr + OCB_STATE, 4);
    uint64_t want = 0;
    if (ld(ctx, mgr + OCB_PENDING_BUF, 1)) want = mgr + OCB_PENDING_BUF;
    else if (state >= 1 && state <= 4 && ld(ctx, mgr + OCB_CURRENT_BUF, 1)) want = mgr + OCB_CURRENT_BUF;
    if (!want) return -1;
    int first = -1, asked = -1;
    for (int k = 0; k < OCB_UNITS; k++) {
        const uint64_t slot = mgr + OCB_SLOTS + (uint64_t)OCB_SLOT_SIZE * k;
        if (ld(ctx, slot, 1) != 1 || !oot_ocb_same(ld, ctx, slot + OCB_SLOT_BUF, want)) continue;
        if (k == chosen) return k;
        if (first < 0) first = k;
        if (asked < 0 && ld(ctx, slot + OCB_SLOT_ASKED, 1) == 1) asked = k;
    }
    return asked >= 0 ? asked : first;
}

/* The identity of the update's choice (0-15, or negative: none): the manager, the unit and the map in it (FNV-1a of
   its path) */
static inline uint64_t oot_ocb_key(OotOcbLoad ld, void *ctx, uint64_t mgr, int chosen) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h = (h ^ mgr) * 0x100000001b3ULL;
    h = (h ^ (uint64_t)(uint32_t)chosen) * 0x100000001b3ULL;
    if (chosen < 0) return h;
    const uint64_t path = mgr + OCB_SLOTS + (uint64_t)OCB_SLOT_SIZE * chosen + OCB_SLOT_BUF;
    for (int i = 0; i < OCB_PATH_MAX; i++) {
        const uint64_t x = ld(ctx, path + i, 1);
        if (!x) break;
        h = (h ^ x) * 0x100000001b3ULL;
    }
    return h;
}

/* One update of the manager mgr: the unit it goes on with. chosen: the update's own unit (0-15), any negative value none.
   A manager this file does not know (oot_ocb_layout), or a choice that is not an in-use unit: the game's own, counted in
   s->refused. The game's own value comes back as it came */
static inline int oot_ocb_step(OotOcbHold *s, OotOcbLoad ld, void *ctx, uint64_t mgr, int chosen, int64_t hold_ns,
                               int64_t now_ns) {
    if (hold_ns <= 0) {
        s->cand = -2;
        return chosen;
    }
    if (chosen >= OCB_UNITS || !oot_ocb_layout(ld, ctx, mgr) ||
        (chosen >= 0 && ld(ctx, mgr + OCB_SLOTS + (uint64_t)OCB_SLOT_SIZE * chosen, 1) != 1)) {
        s->cand = -2, s->refused++;
        return chosen;
    }
    return oot_ocb_pick(s, chosen, oot_ocb_key(ld, ctx, mgr, chosen), oot_ocb_held(ld, ctx, mgr, chosen), hold_ns,
                        now_ns);
}
#endif
