/* (hookshot) The neighbouring map the OptimizeChangeBank thread preloads, kept while the game's choice only flickers: the
   lifted f_832b9880 hands its choice to hos_oot_ocb_pick before it asks for the unit and lets the others go
   (nx2/oot_patches.py patch_ocb_hold); host/oot_ocb_hold.h has why and the rule.
   HOS_OCB_HOLD_MS=<ms>: a new choice is taken once it has stayed the choice that long (OCB_HOLD_DEFAULT_MS with no switch);
   0, a negative number or off: the game's own choice at every frame. Another build than the one the manager's layout
   comes from (OCB_BUILD): the game's own too. */
#include "hos.h"
#include "oot_ocb_hold.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <string.h>

#define OCB_HOLD_DEFAULT_MS 2000
#define OCB_BUILD "e0f3ccfaf50065ae4ea01eb5aeb13b1a45da3348"   /* (main's build id: nx2/oot_patches.py LIFE_BUILD_ID) */

static OotOcbHold hold = OOT_OCB_HOLD_INIT;
static SRWLOCK hold_lock = SRWLOCK_INIT;   /* (one manager, updated once a frame; this does not rely on it) */
static int hold_read;                      /* (the switch and the build read, under hold_lock) */
static int64_t hold_ns;                    /* (0: the game's own choice; never below 0) */
static long long calls;                    /* (under hold_lock: updates seen, for the log's counts) */

static uint64_t guest_load(void *ctx, uint64_t addr, int size) {
    Ctx *c = (Ctx *)ctx;
    return size == 1 ? recomp_load8(c, addr) : size == 4 ? recomp_load32(c, addr) : recomp_load64(c, addr);
}

/* chosen: the update's unit (x21, its low 32 bits; bit 31: none); mgr: the manager (x19). The unit it goes on with */
uint64_t hos_oot_ocb_pick(Ctx *c, uint64_t mgr, uint64_t chosen) {
    const int want = (int)(uint32_t)chosen;
    AcquireSRWLockExclusive(&hold_lock);
    if (!hold_read) {
        hold_read = 1;
        hold_ns = oot_ocb_hold_ms(getenv("HOS_OCB_HOLD_MS"), OCB_HOLD_DEFAULT_MS) * 1000000;
        if (hold_ns > 0 && strcmp(hos_main_build_id, OCB_BUILD) != 0) {
            hos_log("ocb: build %.8s is not the one the preload manager's layout comes from (%.8s): the game's own choice",
                    hos_main_build_id, OCB_BUILD);
            hold_ns = 0;
        } else if (hold_ns > 0)
            hos_log("ocb: a new neighbour to preload is taken once it stays the choice %lld ms (HOS_OCB_HOLD_MS; 0: the "
                    "game's own at every frame)", (long long)(hold_ns / 1000000));
        else
            hos_log("ocb: off (HOS_OCB_HOLD_MS): the game's own choice of a neighbour to preload at every frame");
    }
    int out = want;
    if (hold_ns > 0) {
        const long long refused = hold.refused;
        out = oot_ocb_step(&hold, guest_load, c, mgr, want, hold_ns, hos_now_ns());
        if (hold.refused != refused && !refused)
            hos_log("ocb: the preload manager at 0x%llx is not laid out as host/oot_ocb_hold.h knows it: the game's own "
                    "choice at each such update", (unsigned long long)mgr);
        if (++calls % 18000 == 0)
            hos_log("ocb: %lld choices of another neighbour held back, %lld taken, %lld updates refused, in %lld updates",
                    hold.held_back, hold.taken, hold.refused, calls);
    }
    ReleaseSRWLockExclusive(&hold_lock);
    return out == want ? chosen : (uint64_t)(uint32_t)out;
}
