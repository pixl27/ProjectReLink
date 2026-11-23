/* (hookshot) Every cutscene can be skipped. The game's own skip screen (UIEventSkipScreen) offers to skip any event whose setting
   list does not hold the component Event/EventSetting/EventSettingComponent/SkipSetting/DisableSkip.engine__event__SkipSetting.bgyml,
   a BYML of one entry {IsDisableSkip: true}. Three events list it: the opening in the forest, the staff credits and the title
   scene, so those alone cannot be skipped. That file is served with the value false (the same 56 bytes, so the resource size
   table keeps its entry), and those three skip like every other cutscene, through the game's own skip edge (the opening's is
   the one the menu's Skip opening takes too: host/hos_main.c hos_oot_opening). The idea and the BYML's layout are Rekarina's
   (tools/make_oot_prompts.py skip_setting, GPL-3.0); the code is Hookshot's own.

   The copy is made from the player's own file, at its first lookup, into the game folder's cache/ (cache/event_skip/); a file
   whose layout is not the one expected (a BYML whose root dictionary holds the bool IsDisableSkip), or one already false, is
   left as the game's. A mod's own file for the path wins. HOS_SKIP_ANY_EVENT=0: the three keep the game's setting. */
#include "hos.h"
#include "hos_paths.h"
#include "../gpu/vk/compat/hookshot_env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define SKIP_PATH "Event/EventSetting/EventSettingComponent/SkipSetting/DisableSkip.engine__event__SkipSetting.bgyml"
#define SKIP_KEY "event/eventsetting/eventsettingcomponent/skipsetting/disableskip.engine__event__skipsetting.bgyml"

extern const char *hos_romfs_dir;   /* nn_fs.c: the game's RomFS folder */

static uint32_t rd24(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16; }
static uint32_t rd32(const uint8_t *p) { return rd24(p) | (uint32_t)p[3] << 24; }

/* The BYML's IsDisableSkip set false in place: 1 when it was true and is now false, 0 when the layout is not the one
   expected or it was false already (the file then stays the game's) */
int hos_event_skip_byml(uint8_t *d, size_t n) {
    if (n < 16 || d[0] != 'Y' || d[1] != 'B') return 0;
    const uint32_t keys_at = rd32(d + 4), root_at = rd32(d + 12);
    if (keys_at + 4 > n || root_at + 4 > n || d[keys_at] != 0xC2 || d[root_at] != 0xC1) return 0;
    const uint32_t nkeys = rd24(d + keys_at + 1), count = rd24(d + root_at + 1);
    if (keys_at + 4 + 4ull * nkeys > n || root_at + 4 + 8ull * count > n) return 0;
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t *e = d + root_at + 4 + 8 * i;
        const uint32_t key = rd24(e);
        if (key >= nkeys || e[3] != 0xD0) continue;   /* (0xD0: a bool) */
        const uint32_t at = keys_at + rd32(d + keys_at + 4 + 4 * key);
        static const char name[] = "IsDisableSkip";
        if (at + sizeof name > n || memcmp(d + at, name, sizeof name)) continue;   /* (its terminator too) */
        if (!rd32(e + 4)) return 0;   /* (skippable already) */
        memset((uint8_t *)e + 4, 0, 4);
        return 1;
    }
    return 0;
}

static int skip_off(void) {
    static int off = -1;
    if (off < 0) {
        const char *const e = getenv("HOS_SKIP_ANY_EVENT");
        off = e && *e && env_value_off(e);
        if (off) hos_log("events: HOS_SKIP_ANY_EVENT=0: the opening, the credits and the title scene keep the game's no-skip setting");
    }
    return off;
}

/* mods.c: the file for a RomFS path in place of the game's (key: lower case, '/'), 1 with one. mod: a mod's file for it, or
   NULL: the mod's own wins */
int hos_event_skip_romfs(const char *key, const char *mod, char *out, size_t cap) {
    if (strcmp(key, SKIP_KEY) || mod || !hos_romfs_dir || skip_off()) return 0;
    static volatile LONG state = 0;   /* (0 not made yet, 1 made, -1 left as the game's) */
    static char made[MAX_PATH];
    if (state < 0) return 0;
    if (state == 0) {
        char path[MAX_PATH], src[MAX_PATH];
        const DWORD len = hos_game_path_a(path, MAX_PATH);
        char *const slash = len ? (strrchr(path, 92) ? strrchr(path, 92) : strrchr(path, 47)) : NULL;   /* (linux: '/') */
        snprintf(src, sizeof src, "%s/%s", hos_romfs_dir, SKIP_PATH);
        uint8_t data[4096];
        size_t n = 0;
        FILE *f = fopen(src, "rb");
        if (f) n = fread(data, 1, sizeof data, f), fclose(f);
        if (!slash || !n || n == sizeof data || !hos_event_skip_byml(data, n)) {
            hos_log("events: %s left as the game's (not read, or not the layout expected, or skippable already)", SKIP_PATH);
            InterlockedExchange(&state, -1);
            return 0;
        }
        slash[1] = 0;
        char dir[MAX_PATH];
        snprintf(dir, sizeof dir, "%scache", path);
        CreateDirectoryA(dir, NULL);
        snprintf(dir, sizeof dir, "%scache/event_skip", path);
        CreateDirectoryA(dir, NULL);
        snprintf(made, sizeof made, "%s/disableskip.bgyml", dir);
        FILE *w = fopen(made, "wb");
        const int ok = w && fwrite(data, 1, n, w) == n;
        if (w) fclose(w);
        if (!ok) {
            hos_log("events: %s could not be written: the three unskippable cutscenes stay so", made);
            InterlockedExchange(&state, -1);
            return 0;
        }
        hos_log("events: every cutscene can be skipped (%s <- %s; HOS_SKIP_ANY_EVENT=0: as the game has it)", SKIP_PATH, made);
        InterlockedExchange(&state, 1);
    }
    snprintf(out, cap, "%s", made);
    return 1;
}
