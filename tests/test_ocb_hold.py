"""The neighbouring map the OptimizeChangeBank thread preloads, kept while the game's choice only flickers
(host/oot_ocb_hold.h, nx2/oot_patches.py patch_ocb_hold).

The preload manager's update (f_832b9880) picks one of the map's neighbours every frame, and a change cancels the one loading
or loaded (the thread lets go of all of it and loads the other from the start). A choice flickering between two neighbours,
or in and out of a radius, reloaded them over and over with Link on one spot. The update now goes on with the held map until
another choice has stayed the choice HOS_OCB_HOLD_MS (0: the game's own at every frame).

The rule (the header, compiled alone): no hold, nothing held or the same choice: the game's choice; another choice: the held
map until that choice has lasted the hold without a break, then the choice. A choice is the manager, the unit AND the map in
it: a slot filled again with another map within the hold starts its time again (it never inherits the earlier occupant's
time), and so do another manager and a pause in the updates longer than OCB_GAP_NS.

Which map is held, read from a manager laid out as the game's constructor (f_80d32c00) lays it out: the next one asked for,
else the current one while it loads or is loaded (states 1-4), found among the units in use by its path; none with a direct
request in. A map that two units name (Hyrule Field lists Gerudo Valley twice; a player's report): the
update's choice when it is one of them, else the one that asked for it (+1), else the first; with the rest of the update
modelled (it asks for the unit it goes on with and lets go of the other units asked for, by path), standing in the second
unit's radius, or on the edge of another neighbour's, never cancels that map's load (it did at every other update). A manager laid out otherwise (another vtable: another build or class; a path that points away from its own
buffer, as a stale or foreign pointer would; a buffer size, state or in-use flag the game does not have), or a choice that is
not an in-use unit: the game's own choice at that update, counted, and nothing outside the manager read.

The switch's value: not set or empty is the default, a whole number its milliseconds, 0, a negative whole number or "off" 0
(the game's own), anything else the default; the host reads it once.

The patch: the choice goes through the host once, in the update only, where the update tests it (with the registers as
locals, only where the manager's register has its local: else not patched); only in the supported build's lift; a file
patched before, or whose code differs (another game version), is left as it is; the host's function is declared.

python -I -B tests/test_ocb_hold.py
"""
import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import oot_patches  # noqa: E402

FIXTURE = r"""
#include "oot_ocb_hold.h"
#include <stdio.h>
#include <string.h>
static int fails;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)
#define MS 1000000LL

/* (a guest memory: two managers laid out as f_80d32c00 makes them, their strings in their own buffers) */
#define BASE 0x10000000ULL
#define MGR (BASE + 0x100)
#define MGR2 (BASE + 0x2000)
#define MGR_SIZE 0x1d60
#define ELSEWHERE (BASE + 0x3e00)   /* (memory outside both managers: where a stale pointer could point) */
static unsigned char mem[0x4000];
static int outside;   /* (reads outside the managers: none may happen) */
static uint64_t ld(void *ctx, uint64_t a, int size) {
    (void)ctx;
    uint64_t v = 0;
    if (a < BASE || a - BASE + size > sizeof mem) return 0;
    if (!((a >= MGR && a + size <= MGR + MGR_SIZE) || (a >= MGR2 && a + size <= MGR2 + MGR_SIZE))) outside++;
    memcpy(&v, mem + (a - BASE), size);
    return v;
}
static void put(uint64_t a, uint64_t v, int size) { memcpy(mem + (a - BASE), &v, size); }
static void str(uint64_t at, const char *s) { memcpy(mem + (at - BASE), s, strlen(s) + 1); }
static uint64_t slot_of(uint64_t m, int k) { return m + OCB_SLOTS + (uint64_t)OCB_SLOT_SIZE * k; }
static uint64_t slot(int k) { return slot_of(MGR, k); }
static void build(uint64_t m) {   /* (as the constructor: the vtable, each path at its own buffer of 0x100 bytes) */
    memset(mem + (m - BASE), 0, MGR_SIZE);
    put(m, OCB_VTABLE, 8);
    put(m + OCB_CURRENT, m + OCB_CURRENT_BUF, 8), put(m + OCB_CURRENT_CAP, 0x100, 4);
    put(m + OCB_PENDING, m + OCB_PENDING_BUF, 8), put(m + OCB_PENDING_CAP, 0x100, 4);
    for (int k = 0; k < OCB_UNITS; k++)
        put(slot_of(m, k) + OCB_SLOT_PATH, slot_of(m, k) + OCB_SLOT_BUF, 8), put(slot_of(m, k) + OCB_SLOT_CAP, 0x100, 4);
}
static void unit_of(uint64_t m, int k, const char *path) {   /* (in use, its path in its own buffer at +0x1c) */
    put(slot_of(m, k), 1, 1);
    str(slot_of(m, k) + OCB_SLOT_BUF, path);
}
static void unit(int k, const char *path) { unit_of(MGR, k, path); }
static void manager_of(uint64_t m, int state, const char *cur, const char *pend, int direct) {
    put(m + OCB_STATE, (uint64_t)state, 4);
    put(m + OCB_DIRECT, (uint64_t)direct, 1);
    str(m + OCB_CURRENT_BUF, cur), put(m + OCB_CURRENT_LEN, strlen(cur), 4);
    str(m + OCB_PENDING_BUF, pend);
}
static void manager(int state, const char *cur, const char *pend, int direct) { manager_of(MGR, state, cur, pend, direct); }

/* (the rest of f_832b9880 after the choice, simplified: it asks for the unit it goes on with (f_832ba540: another map
   than the one loading or loaded starts loading, cancelling that one; on a new request the unit's +1 is set), then lets
   go of every other unit in use whose +1 is set (f_832ba400, by the unit's path: the map loading or loaded is cancelled
   when it is that path; +1 cleared). loads: maps asked for from the start; cancels: loads cancelled) */
static int loads, cancels;
static const char *path_at(uint64_t a) { return (const char *)mem + (a - BASE); }
static int live(void) { const uint64_t st = ld(0, MGR + OCB_STATE, 4); return st >= 1 && st <= 4; }
static void game_update(int u) {
    if (u >= 0 && u < OCB_UNITS) {
        const char *p = path_at(slot(u) + OCB_SLOT_BUF);
        if (!live() || strcmp(path_at(MGR + OCB_CURRENT_BUF), p)) {
            cancels += live();
            manager(2, p, "", 0), loads++;
            put(slot(u) + OCB_SLOT_ASKED, 1, 1);
        }
    }
    for (int k = 0; k < OCB_UNITS; k++) {
        if (k == u || ld(0, slot(k), 1) != 1 || ld(0, slot(k) + OCB_SLOT_ASKED, 1) != 1) continue;
        put(slot(k) + OCB_SLOT_ASKED, 0, 1);
        if (live() && !strcmp(path_at(MGR + OCB_CURRENT_BUF), path_at(slot(k) + OCB_SLOT_BUF)))
            cancels++, manager(0, "", "", 0);
    }
}

int main(void) {
    const char *A = "Work/Banc/Scene/MarketEntrance_Child/MarketEntrance_Child.bcett.json";
    const char *B = "Work/Banc/Scene/TempleOfTime_Child/TempleOfTime_Child.bcett.json";
    const char *C = "Work/Banc/Scene/HyruleCastle_Child/HyruleCastle_Child.bcett.json";
    const char *D = "Work/Banc/Scene/HyruleField_Child/HyruleField_Child.bcett.json";
    {   /* (no hold, nothing held, the same choice: the game's) */
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        CHECK(oot_ocb_pick(&s, 3, 33, 1, 0, 0) == 3);
        CHECK(oot_ocb_pick(&s, 3, 33, 1, -5 * MS, 0) == 3);
        CHECK(oot_ocb_pick(&s, 3, 33, -1, 1000 * MS, 0) == 3);
        CHECK(oot_ocb_pick(&s, -1, 11, -1, 1000 * MS, 0) == -1);
        CHECK(oot_ocb_pick(&s, 1, 11, 1, 1000 * MS, 0) == 1);
    }
    {   /* (a flicker between the held unit 1 and unit 2, every other frame for 10 s: never taken) */
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        int other = 0;
        for (long long t = 0; t < 10000 * MS; t += 16 * MS) {
            const int c = (t / (16 * MS)) % 2 ? 2 : 1;
            other += oot_ocb_pick(&s, c, (uint64_t)c, 1, 1000 * MS, t) != 1;
        }
        CHECK(other == 0);
    }
    {   /* (in and out of the held unit's radius, 900 ms out each time: never let go) */
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        int other = 0;
        for (long long t = 0; t < 10000 * MS; t += 16 * MS) {
            const int c = (t % (1000 * MS)) < 900 * MS ? -1 : 1;
            other += oot_ocb_pick(&s, c, (uint64_t)(int64_t)c, 1, 1000 * MS, t) != 1;
        }
        CHECK(other == 0);
    }
    {   /* (a real change: unit 2 the choice from t = 5 s on, taken at the first update 1 s later) */
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        long long taken = -1;
        for (long long t = 0; t < 10000 * MS && taken < 0; t += 16 * MS) {
            const int c = t >= 5000 * MS ? 2 : 1;
            if (oot_ocb_pick(&s, c, (uint64_t)c, 1, 1000 * MS, t) == 2) taken = t;
        }
        CHECK(taken >= 6008 * MS && taken < 6024 * MS);   /* (the first update at 5,008 ms, 16 ms apart) */
        CHECK(oot_ocb_pick(&s, 2, 2, 2, 1000 * MS, taken + 16 * MS) == 2);   /* (held now) */
        CHECK(s.taken == 1 && s.held_back == 1);
    }
    {   /* (leaving every radius: let go once out for the hold) */
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        CHECK(oot_ocb_pick(&s, -1, 7, 4, 500 * MS, 0) == 4);
        CHECK(oot_ocb_pick(&s, -1, 7, 4, 500 * MS, 300 * MS) == 4);
        CHECK(oot_ocb_pick(&s, -1, 7, 4, 500 * MS, 499 * MS) == 4);
        CHECK(oot_ocb_pick(&s, -1, 7, 4, 500 * MS, 500 * MS) == -1);
    }
    {   /* (a break in the other choice starts its time again) */
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        CHECK(oot_ocb_pick(&s, 2, 2, 1, 1000 * MS, 0) == 1);
        CHECK(oot_ocb_pick(&s, 2, 2, 1, 1000 * MS, 400 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 2, 1, 1000 * MS, 800 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 3, 3, 1, 1000 * MS, 900 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 2, 1, 1000 * MS, 1300 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 2, 1, 1000 * MS, 1700 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 2, 1, 1000 * MS, 2100 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 2, 1, 1000 * MS, 2299 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 2, 1, 1000 * MS, 2300 * MS) == 2);
    }
    {   /* (the same unit with another identity starts its time again) */
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        CHECK(oot_ocb_pick(&s, 2, 20, 1, 1000 * MS, 0) == 1);
        CHECK(oot_ocb_pick(&s, 2, 20, 1, 1000 * MS, 400 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 21, 1, 1000 * MS, 800 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 21, 1, 1000 * MS, 1200 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 21, 1, 1000 * MS, 1600 * MS) == 1);
        CHECK(oot_ocb_pick(&s, 2, 21, 1, 1000 * MS, 1800 * MS) == 2);
    }

    /* (which unit is held, from a manager laid out as the game's) */
    build(MGR);
    unit(0, A), unit(1, B), unit(5, C);
    manager(4, B, "", 0);
    CHECK(oot_ocb_layout(ld, 0, MGR) && oot_ocb_held(ld, 0, MGR, -1) == 1);   /* (loaded) */
    manager(2, C, "", 0);
    CHECK(oot_ocb_held(ld, 0, MGR, -1) == 5);   /* (loading) */
    manager(5, B, A, 0);
    CHECK(oot_ocb_held(ld, 0, MGR, -1) == 0);   /* (cancelling B for A: A, the next one) */
    manager(5, B, "", 0);
    CHECK(oot_ocb_held(ld, 0, MGR, -1) == -1);  /* (cancelling, nothing next) */
    manager(0, "", "", 0);
    CHECK(oot_ocb_held(ld, 0, MGR, -1) == -1);  /* (idle) */
    manager(4, B, "", 1);
    CHECK(oot_ocb_held(ld, 0, MGR, -1) == -1);  /* (a direct request in) */
    manager(4, "Work/Banc/Scene/Other/Other.bcett.json", "", 0);
    CHECK(oot_ocb_held(ld, 0, MGR, -1) == -1);  /* (a path no unit has) */
    manager(4, "Work/Banc/Scene/TempleOfTime_Child/TempleOfTime_Child.bcett.jso", "", 0);
    CHECK(oot_ocb_held(ld, 0, MGR, -1) == -1);  /* (a prefix is not the path) */
    put(slot(1), 0, 1), manager(4, B, "", 0);
    CHECK(oot_ocb_held(ld, 0, MGR, -1) == -1);  /* (the unit no longer in use: the map changed) */
    CHECK(outside == 0);

    {   /* (TWO UNITS NAMING ONE MAP, as Hyrule Field lists Gerudo Valley: unit 1, the Lake Hylia side, and unit 4, the
           valley's gate; a player's report. Inside unit 4's radius the update chooses 4 and asks for G, so G
           is held; were the held unit 1, the first naming G, choice 4 would be held back for unit 1 and the update's
           release of unit 4, by path, would cancel G's load at every other update) */
        const char *G = "Work/Banc/Scene/GerudoValley_Child/GerudoValley_Child.bcett.json";
        build(MGR);
        unit(0, A), unit(1, G), unit(4, G);
        manager(2, G, "", 0);   /* (G loading) */
        CHECK(oot_ocb_held(ld, 0, MGR, 4) == 4);
        CHECK(oot_ocb_held(ld, 0, MGR, -1) == 1);   /* (no choice: the first naming it) */
        {
            OotOcbHold s = OOT_OCB_HOLD_INIT;
            CHECK(oot_ocb_step(&s, ld, 0, MGR, 4, 2000 * MS, 0) == 4);
            CHECK(oot_ocb_step(&s, ld, 0, MGR, 4, 2000 * MS, 16 * MS) == 4);
            CHECK(s.held_back == 0);
        }
        CHECK(oot_ocb_held(ld, 0, MGR, 1) == 1);
        CHECK(oot_ocb_held(ld, 0, MGR, 16) == 1);   /* (no such unit) */
        CHECK(oot_ocb_held(ld, 0, MGR, (int)0x80000004u) == 1);   /* (none chosen: bit 31 set) */
        manager(4, G, "", 0);
        CHECK(oot_ocb_held(ld, 0, MGR, 4) == 4);    /* (loaded) */
        manager(5, B, G, 0);
        CHECK(oot_ocb_held(ld, 0, MGR, 4) == 4);    /* (G the next one asked for) */
        manager(2, G, "", 0);
        /* (unit 4 asked for G (+1), as the update leaves it: a choice that is not a unit naming G, or none, goes on with
           unit 4, so the update's release of the others does not reach G; a choice naming G is still that choice) */
        put(slot(4) + OCB_SLOT_ASKED, 1, 1);
        CHECK(oot_ocb_held(ld, 0, MGR, -1) == 4);
        CHECK(oot_ocb_held(ld, 0, MGR, 0) == 4);
        CHECK(oot_ocb_held(ld, 0, MGR, 1) == 1);
        CHECK(oot_ocb_held(ld, 0, MGR, 4) == 4);
        put(slot(0) + OCB_SLOT_ASKED, 1, 1), put(slot(4) + OCB_SLOT_ASKED, 0, 1);
        CHECK(oot_ocb_held(ld, 0, MGR, -1) == 1);   /* (the unit asked for names another map: the first naming G) */
        put(slot(4), 0, 1), put(slot(4) + OCB_SLOT_ASKED, 1, 1);
        CHECK(oot_ocb_held(ld, 0, MGR, 4) == 1);    /* (unit 4 no longer in use, its buffer still naming G) */
        CHECK(outside == 0);
    }
    {   /* (the walk, the rest of the update as the game runs it (game_update): Link stays inside unit 4's radius for
           10 s, the update choosing 4 every 16 ms; then he walks the edge of unit 0's radius, the choice flickering
           between 4 and 0 every 400 ms for 10 s, and 0 never lasts the hold. G asked for once, never cancelled) */
        const char *G = "Work/Banc/Scene/GerudoValley_Child/GerudoValley_Child.bcett.json";
        build(MGR);
        unit(0, A), unit(1, G), unit(4, G);
        manager(0, "", "", 0);
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        cancels = loads = 0;
        long long t = 0;
        for (; t < 10000 * MS; t += 16 * MS) game_update(oot_ocb_step(&s, ld, 0, MGR, 4, 2000 * MS, t));
        CHECK(loads == 1 && cancels == 0 && s.held_back == 0);
        if (cancels) printf("  (inside unit 4's radius: G's load cancelled %d times, asked for %d times)\n", cancels, loads);
        for (; t < 20000 * MS; t += 16 * MS)
            game_update(oot_ocb_step(&s, ld, 0, MGR, (t / (400 * MS)) % 2 ? 0 : 4, 2000 * MS, t));
        CHECK(loads == 1 && cancels == 0);
        if (cancels) printf("  (on the edge of unit 0's radius: G's load cancelled %d times)\n", cancels);
        CHECK(!strcmp((const char *)mem + (MGR + OCB_CURRENT_BUF - BASE), G));
        CHECK(s.refused == 0 && outside == 0);
        /* (and choice 0 lasting the hold: A taken, G let go once) */
        for (long long t0 = t; t < t0 + 2100 * MS; t += 16 * MS) game_update(oot_ocb_step(&s, ld, 0, MGR, 0, 2000 * MS, t));
        CHECK(loads == 2 && cancels == 1);
        CHECK(!strcmp((const char *)mem + (MGR + OCB_CURRENT_BUF - BASE), A));
    }
    {   /* (from unit 1's radius into unit 4's: G held through unit 1, the choice 4 is the same map, not held back) */
        const char *G = "Work/Banc/Scene/GerudoValley_Child/GerudoValley_Child.bcett.json";
        build(MGR);
        unit(0, A), unit(1, G), unit(4, G);
        manager(4, G, "", 0);
        put(slot(1) + OCB_SLOT_ASKED, 1, 1);
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 2000 * MS, 0) == 1);
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 4, 2000 * MS, 16 * MS) == 4);
        CHECK(s.held_back == 0);
    }

    {   /* (THE REVIEW'S CASE: unit 1 held back for 1.5 s with B in it; then the map changes and its slot is filled again
           with D, which the update chooses at once. D has been the choice from 1.6 s, not from 0: held until 3.6 s) */
        build(MGR);
        unit(0, A), unit(1, B);
        manager(4, A, "", 0);
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        long long taken = -1;
        for (long long t = 0; t < 1500 * MS; t += 16 * MS) CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 2000 * MS, t) == 0);
        unit(1, D);   /* (the same slot, another map: f_810a44e0 cleared the slots and registered the new map's units) */
        for (long long t = 1600 * MS; t < 6000 * MS && taken < 0; t += 16 * MS)
            if (oot_ocb_step(&s, ld, 0, MGR, 1, 2000 * MS, t) == 1) taken = t;
        CHECK(taken >= 3600 * MS && taken < 3616 * MS);
        if (taken < 3600 * MS) printf("  (the new occupant was taken at %lld ms, on the old one's time)\n", taken / MS);
        CHECK(s.refused == 0 && outside == 0);
    }
    {   /* (the same slot and map, after a pause in the updates longer than OCB_GAP_NS: its time starts again) */
        build(MGR);
        unit(0, A), unit(1, B);
        manager(4, A, "", 0);
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 0) == 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 400 * MS) == 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 1000 * MS) == 0);   /* (600 ms without an update) */
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 1400 * MS) == 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 1800 * MS) == 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 2000 * MS) == 1);
    }
    {   /* (the same unit and map in another manager: its time starts again) */
        build(MGR), build(MGR2);
        unit(0, A), unit(1, B), unit_of(MGR2, 0, A), unit_of(MGR2, 1, B);
        manager(4, A, "", 0), manager_of(MGR2, 4, A, "", 0);
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 0) == 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 400 * MS) == 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR2, 1, 1000 * MS, 800 * MS) == 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR2, 1, 1000 * MS, 1200 * MS) == 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR2, 1, 1000 * MS, 1600 * MS) == 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR2, 1, 1000 * MS, 1800 * MS) == 1);
        CHECK(outside == 0);
    }

    /* (a manager laid out otherwise, or a stale reference: the game's own choice at that update, counted; the held map
       is never looked for through a pointer that left the manager) */
    {
        OotOcbHold s = OOT_OCB_HOLD_INIT;
        long long refused = 0;
#define FRESH() (build(MGR), unit(0, A), unit(1, B), manager(4, A, "", 0), s.cand = -2)
#define GAMES_OWN(why) do { refused = s.refused; CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 0) == 1); \
                            CHECK(s.refused == refused + 1); if (s.refused != refused + 1) printf("  (%s)\n", why); } while (0)
        FRESH();
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 0) == 0);   /* (as the game lays it out: held) */
        CHECK(s.refused == 0);
        FRESH(), put(MGR, 0x84bc1a30ULL, 8);
        GAMES_OWN("another vtable");
        /* (the current path pointing away from its buffer, at memory holding the held path: the earlier code followed it) */
        FRESH(), str(ELSEWHERE, A), put(MGR + OCB_CURRENT, ELSEWHERE, 8), str(MGR + OCB_CURRENT_BUF, "");
        GAMES_OWN("current path elsewhere");
        FRESH(), str(ELSEWHERE, A), put(MGR + OCB_PENDING, ELSEWHERE, 8);
        GAMES_OWN("next path elsewhere");
        FRESH(), str(ELSEWHERE, A), put(slot(0) + OCB_SLOT_PATH, ELSEWHERE, 8);
        GAMES_OWN("a unit's path elsewhere");
        FRESH(), put(slot(0) + OCB_SLOT_PATH, 0, 8);
        GAMES_OWN("a unit's path null");
        FRESH(), put(MGR + OCB_CURRENT_CAP, 0x40, 4);
        GAMES_OWN("current buffer size");
        FRESH(), put(MGR + OCB_PENDING_CAP, 0x1000, 4);
        GAMES_OWN("next buffer size");
        FRESH(), put(slot(1) + OCB_SLOT_CAP, 0, 4);
        GAMES_OWN("a unit's buffer size");
        FRESH(), put(MGR + OCB_CURRENT_LEN, 0x100, 4);
        GAMES_OWN("current length");
        FRESH(), put(MGR + OCB_STATE, 9, 4);
        GAMES_OWN("state");
        FRESH(), put(slot(3), 2, 1);
        GAMES_OWN("in-use flag");
        FRESH(), put(slot(1), 0, 1);
        GAMES_OWN("the choice not in use");
        FRESH();
        refused = s.refused;
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 16, 1000 * MS, 0) == 16);   /* (no such unit) */
        CHECK(oot_ocb_step(&s, ld, 0, MGR + 4, 1, 1000 * MS, 0) == 1);   /* (not aligned) */
        CHECK(oot_ocb_step(&s, ld, 0, 0, 1, 1000 * MS, 0) == 1);   /* (none) */
        CHECK(s.refused == refused + 3);
        CHECK(outside == 0);
        FRESH();   /* (and back to the game's layout: held again) */
        CHECK(oot_ocb_step(&s, ld, 0, MGR, 1, 1000 * MS, 0) == 0);
        /* (none chosen: bit 31 set, any value; the game's own value back when not held) */
        manager(0, "", "", 0);
        CHECK(oot_ocb_step(&s, ld, 0, MGR, (int)0x80000005u, 1000 * MS, 0) == (int)0x80000005u);
    }

    /* (the switch's value: unset, empty, 0, negative, positive, off, malformed) */
    CHECK(oot_ocb_hold_ms(0, 2000) == 2000);
    CHECK(oot_ocb_hold_ms("", 2000) == 2000);
    CHECK(oot_ocb_hold_ms("0", 2000) == 0);
    CHECK(oot_ocb_hold_ms("-0", 2000) == 0);
    CHECK(oot_ocb_hold_ms("-1", 2000) == 0);
    CHECK(oot_ocb_hold_ms("-5", 2000) == 0);
    CHECK(oot_ocb_hold_ms("-2000", 2000) == 0);
    CHECK(oot_ocb_hold_ms("-99999999999999", 2000) == 0);
    CHECK(oot_ocb_hold_ms("off", 2000) == 0);
    CHECK(oot_ocb_hold_ms("1", 2000) == 1);
    CHECK(oot_ocb_hold_ms("2500", 2000) == 2500);
    CHECK(oot_ocb_hold_ms("99999999999999", 2000) == OCB_HOLD_MAX_MS);
    CHECK(oot_ocb_hold_ms("-", 2000) == 2000);
    CHECK(oot_ocb_hold_ms("-x", 2000) == 2000);
    CHECK(oot_ocb_hold_ms("--5", 2000) == 2000);
    CHECK(oot_ocb_hold_ms("12x", 2000) == 2000);
    CHECK(oot_ocb_hold_ms(" 5", 2000) == 2000);
    CHECK(oot_ocb_hold_ms("offf", 2000) == 2000);
    CHECK(oot_ocb_hold_ms(0, -3) == 0);
    {   /* (each value as the update sees it: unset and positive hold B back, 0 and negative are the game's own) */
        const char *vals[] = {0, "0", "-5", "off", "1500"};
        const int want[] = {0, 1, 1, 1, 0};
        for (int i = 0; i < 5; i++) {
            build(MGR), unit(0, A), unit(1, B), manager(4, A, "", 0);
            OotOcbHold s = OOT_OCB_HOLD_INIT;
            const int64_t hold = oot_ocb_hold_ms(vals[i], 2000) * MS;
            CHECK(hold >= 0);
            const int got = oot_ocb_step(&s, ld, 0, MGR, 1, hold, 0);
            CHECK(got == want[i]);
            if (got != want[i]) printf("  (HOS_OCB_HOLD_MS=%s)\n", vals[i] ? vals[i] : "(unset)");
        }
    }
    printf(fails ? "ocb_hold: %d FAILED\n" : "ocb_hold: all passed\n", fails);
    return fails != 0;
}
"""

# (the update as the lift has it around its choice, and the next function, which has the same label text)
LIFTED = '''#include "a64.h"
void f_832b9880(Ctx *__restrict A64_NOESC c) {
    uint64_t x0_ = c->x[0], x8_ = c->x[8], x9_ = c->x[9], x19_ = c->x[19], x20_ = c->x[20], x21_ = c->x[21], x31_ = c->x[31];
c->x[30] = 0x832b9dc8ULL; c->x[0] = x0_; c->x[8] = x8_; c->x[19] = x19_; c->x[20] = x20_; c->x[21] = x21_; c->x[31] = x31_; f_803e99b0(c);x0_ = c->x[0]; x8_ = c->x[8]; x9_ = c->x[9];
L_832b9dc8:;
if (((x21_ >> 31) & 1)) { goto L_832b9e0c; }
    x8_ = 0x128ULL;
L_832b9e0c:;
}

void f_832bb980(Ctx *__restrict A64_NOESC c) {
L_832b9dc8:;
if (((x21_ >> 31) & 1)) { goto L_832b9e0c; }
}
'''
CALL = "    x21_ = hos_oot_ocb_pick(c, x19_, x21_);   " + oot_patches.OCBH_MARK + "\n"


class OcbHoldTests(unittest.TestCase):
    def test_rule(self):
        compiler = shutil.which("gcc") or shutil.which("clang")
        if not compiler:
            self.skipTest("no C compiler")
        with tempfile.TemporaryDirectory(prefix="hookshot-ocb-hold-") as tmp:
            tmp = Path(tmp)
            (tmp / "rule.c").write_text(FIXTURE, encoding="utf-8")
            binary = tmp / "rule.exe"
            build = subprocess.run([compiler, "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-O1", "-I", str(ROOT / "host"),
                                    str(tmp / "rule.c"), "-o", str(binary)], text=True, capture_output=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(binary)], text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("all passed", run.stdout)

    def test_patch(self):
        out, n = oot_patches.patch_ocb_hold(LIFTED)
        self.assertEqual(n, 1)
        update = out[out.index(oot_patches.OCBH_HEAD):out.index("void f_832bb980")]
        self.assertIn("L_832b9dc8:;\n" + CALL + "if (((x21_ >> 31) & 1)) { goto L_832b9e0c; }\n", update)
        self.assertEqual(out.count("hos_oot_ocb_pick(c,"), 1)   # (the one call; the next function's text as it was)
        self.assertIn('#include "a64.h"\n' + oot_patches.OCBH_DECL, out)
        self.assertEqual(oot_patches.patch_ocb_hold(out), (out, 1))   # (patched before: as it is)
        other = LIFTED.replace("{ goto L_832b9e0c; }\n    x8_ = 0x128ULL;", "{ goto L_832b9e10; }\n    x8_ = 0x128ULL;")
        self.assertNotEqual(other, LIFTED)
        self.assertEqual(oot_patches.patch_ocb_hold(other), (other, 0))   # (another game version's code)
        none = LIFTED.replace("void f_832b9880", "void f_932b9880")
        self.assertEqual(oot_patches.patch_ocb_hold(none), (none, 0))

    def test_patch_without_locals(self):
        plain = LIFTED.replace("    uint64_t x0_ = c->x[0], x8_ = c->x[8], x9_ = c->x[9], x19_ = c->x[19], x20_ = c->x[20], "
                               "x21_ = c->x[21], x31_ = c->x[31];\n", "").replace(
            "L_832b9dc8:;\nif (((x21_ >> 31)", "L_832b9dc8:;\nif (((c->x[21] >> 31)", 1)
        out, n = oot_patches.patch_ocb_hold(plain)
        self.assertEqual(n, 1)
        self.assertIn("    c->x[21] = hos_oot_ocb_pick(c, c->x[19], c->x[21]);", out)

    def test_patch_locals_without_the_managers_local(self):
        # (the choice in x21's local but no local for x19, the manager: c->x[19] may be stale there, so not patched)
        no_x19 = LIFTED.replace(" x19_ = c->x[19],", "", 1)
        self.assertNotEqual(no_x19, LIFTED)
        self.assertEqual(oot_patches.patch_ocb_hold(no_x19), (no_x19, 0))
        out, counts, lines = oot_patches.patch_file(no_x19, "code_037.c", True)
        self.assertNotIn("hos_oot_ocb_pick", out)

    def test_patch_file_counts_it(self):
        out, counts, lines = oot_patches.patch_file(LIFTED, "code_037.c", True)
        self.assertEqual(counts["preload choice"], 1)
        self.assertIn(CALL, out)
        self.assertTrue(any("choice of a neighbouring map" in line for line in lines), lines)
        again, counts, lines = oot_patches.patch_file(out, "code_037.c", True)
        self.assertEqual(again, out)
        self.assertEqual(counts["preload choice"], 1)

    def test_patch_file_supported_build_only(self):
        # (another build's lift, or one not standalone: the host's reading of the manager is that build's, so not patched,
        # and the report does not warn about it)
        out, counts, lines = oot_patches.patch_file(LIFTED, "code_037.c", False)
        self.assertNotIn("hos_oot_ocb_pick", out)
        self.assertEqual(counts["preload choice"], 0)
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            oot_patches.report(["code_037.c"], [counts], False)
        self.assertNotIn("preload manager's choice", said.getvalue())
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            oot_patches.report(["code_037.c"], [counts], True)
        self.assertIn("preload manager's choice was found 0 times", said.getvalue())

    def test_host_calls_the_rule(self):
        host = (ROOT / "host" / "oot_ocb_hold.c").read_text(encoding="utf-8")
        for needle in ('getenv("HOS_OCB_HOLD_MS")', "oot_ocb_step(&hold, guest_load, c, mgr, want, hold_ns, hos_now_ns())",
                       "uint64_t hos_oot_ocb_pick(Ctx *c, uint64_t mgr, uint64_t chosen) {",
                       "strcmp(hos_main_build_id, OCB_BUILD) != 0", "if (!hold_read) {"):
            self.assertIn(needle, host)
        # (the build the layout comes from is the one the patches are made for)
        self.assertIn('#define OCB_BUILD "%s"' % oot_patches.LIFE_BUILD_ID, host)
        # (the switch read once, whatever its value: no "not read yet" value it could also take)
        self.assertNotIn("hold_ns < 0", host)


if __name__ == "__main__":
    unittest.main()
