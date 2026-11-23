/* (hookshot) Ocarina of Time: the textures an incomplete dump damages, read as a whole texture of their kind.
   The 23.3 GB dumps (a complete one is about 23.8 GB) hold three .zs textures whose start is zeros, not a zstd frame.
   The game fails each model that uses one and asks for it again: the Great Fairy Fountain's floor is in the Hyrule
   Field preload, which then never finishes, and its 10.8 MiB model is decoded about 15 times a second while Link is
   in Kokiri or the Market (~1.3 cores on the Steam Deck). Hookshot Builder copies a stand-in over each, but only in a
   romfs it unpacked itself (nx2/make_game.py stand_in_damaged_files); a romfs another tool unpacked, or one copied
   by hand, keeps them. So the host reads the same stand-in in their place, whoever unpacked the romfs, and writes
   nothing to the player's files. */
#include "hos.h"
#include <stdio.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

HANDLE hos_fs_open_read(const char *path);   /* nn_fs.c */
int hos_fs_resolve_read(const char *guest, char *path, size_t cap);

/* Each damaged texture and the stand-in nx2/nup.py stand_in_for picks for it in the game's Tex folder (the same kind
   of texture by the end of its name, then the longest common start): the builder's choice, so a game folder looks
   the same whichever fixed it. tests/test_damaged_rom_files.py holds the two to each other */
static const struct { const char *damaged, *stand_in; } stand_ins[] = {
    {"Tex/CmnTex_Map_Soil_FairyFountain_Floor_D_Nrm.bntx.zs", "Tex/CmnTex_Map_Sand_FairyFountain_Sand_A_Nrm.bntx.zs"},
    {"Tex/Rock_WaterTemple_Gate_M_01_Nrm.bntx.zs", "Tex/Rock_WaterTemple_Gate_P_01_Nrm.bntx.zs"},
    {"Tex/RBake_Scene_GanonStage_G_2_1_d0_t0.bntx.zs", "Tex/RBake_Scene_GanonStage_G_2_2_d0_t0.bntx.zs"},
};
#define STAND_INS (sizeof stand_ins / sizeof *stand_ins)

/* 1: the file starts as zstd data (a frame, or a skippable frame: nup.py zstd_start); 0: it does not; -1: it could not
   be read, and is left to the open that follows */
static int starts_as_zstd(const char *path) {
    HANDLE h = hos_fs_open_read(path);
    if (h == INVALID_HANDLE_VALUE) return -1;
    unsigned char b[4];
    DWORD got = 0;
    const BOOL ok = ReadFile(h, b, sizeof b, &got, NULL);
    CloseHandle(h);
    if (!ok) return -1;
    return got == 4 && ((b[0] == 0x28 && b[1] == 0xb5 && b[2] == 0x2f && b[3] == 0xfd) ||
                        ((b[0] & 0xf0) == 0x50 && b[1] == 0x2a && b[2] == 0x4d && b[3] == 0x18));
}

/* OpenFile's host path for a game file (guest "content:/Tex/..."): one of the damaged textures, when it is damaged and
   its stand-in is whole, becomes the stand-in's path. 1 when it did. A file that is whole (a complete dump, or a romfs
   the builder fixed) costs a name compare and, for these three names only, a read of its first four bytes */
int hos_stand_in_damaged(const char *guest, char *path, size_t cap) {
    static int said[STAND_INS];
    const char *rel = strstr(guest, ":/");
    if (!rel) return 0;
    for (rel += 2; *rel == '/'; rel++) {}
    if (strncmp(rel, "Tex/", 4)) return 0;
    for (size_t i = 0; i < STAND_INS; i++) {
        if (strcmp(rel, stand_ins[i].damaged)) continue;
        if (starts_as_zstd(path) != 0) return 0;
        const char *stand_in = stand_ins[i].stand_in + 4;
        char alt[1024], alt_guest[1024];
        if ((size_t)snprintf(alt_guest, sizeof alt_guest, "%.*s%s", (int)(rel - guest), guest, stand_ins[i].stand_in) >= sizeof alt_guest ||
            !hos_fs_resolve_read(alt_guest, alt, sizeof alt) ||
            strlen(alt) >= cap || starts_as_zstd(alt) != 1) {
            if (!__atomic_fetch_add(&said[i], 1, __ATOMIC_RELAXED))
                hos_log_at(HOS_LOG_WARN, "fs: %s is damaged in this copy of the game (zeros where its data starts) and its stand-in %s "
                           "is missing or damaged too: the game goes without it and asks for it again and again. Your dump is "
                           "incomplete (about 23.3 GB instead of 23.8 GB): dump the game again", guest, stand_in);
            return 0;
        }
        if (!__atomic_fetch_add(&said[i], 1, __ATOMIC_RELAXED))
            hos_log_at(HOS_LOG_WARN, "fs: %s is damaged in this copy of the game (zeros where its data starts): %s, a similar "
                       "texture, is read in its place. Your dump is incomplete (about 23.3 GB instead of 23.8 GB): dump the game "
                       "again for the original", guest, stand_in);
        memcpy(path, alt, strlen(alt) + 1);
        return 1;
    }
    return 0;
}
