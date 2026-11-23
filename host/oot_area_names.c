/* (hookshot) The place a map of the game is, in English, for the Discord status (oot_area_names.h).

   The game names its maps in its data (Banc/GameBancParam/<map>.game__banc__GameBancParam.bgyml: 80 of them on the
   supported build), with the age, a changed state or Master Quest as a suffix: _Child, _Adult, _Alternative (Kakariko
   after the fire, Zora's Domain frozen), _MQ. The suffixes are taken off and the rest looked up; a few places are named
   by the age (the market's ruins). The rooms of the bosses and of the fights inside a dungeon are maps of their own
   (GohmaStage): they get the fight's name. The names are the places' usual English ones; the game's own, translated,
   are in its message files and not read here. */
#include "oot_area_names.h"
#include <stdio.h>
#include <string.h>

static const struct { const char *map, *name; } names[] = {
    /* (exact names first: a place named by the age or state) */
    {"Market_Adult", "Market Ruins"},
    {"KakarikoVillage_Adult_Alternative", "Kakariko Village"},
    {"ZoraDomain_Adult_Alternative", "Zora's Domain (frozen)"},
    /* places */
    {"KokiriForest", "Kokiri Forest"},
    {"LostWood", "Lost Woods"},
    {"SacredForestMeadow", "Sacred Forest Meadow"},
    {"HyruleField", "Hyrule Field"},
    {"Market", "Hyrule Market"},
    {"HyruleCastle", "Hyrule Castle"},
    {"GanonCastle", "Ganon's Castle"},
    {"KakarikoVillage", "Kakariko Village"},
    {"Graveyard", "Kakariko Graveyard"},
    {"DeathMountainTrail", "Death Mountain Trail"},
    {"DeathMountainCrater", "Death Mountain Crater"},
    {"GoronCity", "Goron City"},
    {"ZoraRiver", "Zora's River"},
    {"ZoraDomain", "Zora's Domain"},
    {"ZoraFountain", "Zora's Fountain"},
    {"GerudoValley", "Gerudo Valley"},
    {"GerudoFortress", "Gerudo's Fortress"},
    {"HountedWasteland", "Haunted Wasteland"},   /* (the game's spelling) */
    {"DesertColossus", "Desert Colossus"},
    {"SageRoom", "Chamber of Sages"},
    /* dungeons */
    {"InsideTheDekuTree", "Inside the Deku Tree"},
    {"DodongoCavern", "Dodongo's Cavern"},
    {"InsideJabuJabuBelly", "Inside Jabu-Jabu's Belly"},
    {"ForestTemple", "Forest Temple"},
    {"FireTemple", "Fire Temple"},
    {"WaterTemple", "Water Temple"},
    {"ShadowTemple", "Shadow Temple"},
    {"SpiritTemple", "Spirit Temple"},
    {"BottomOfTheWell", "Bottom of the Well"},
    {"IceCavern", "Ice Cavern"},
    {"GerudoTrainingGround", "Gerudo Training Ground"},
    {"InsideGanonCastle", "Inside Ganon's Castle"},
    {"CrumblingGanonCastle", "Escaping Ganon's Castle"},
    /* the bosses' rooms */
    {"GohmaStage", "Fighting Queen Gohma"},
    {"DodongoStage", "Fighting King Dodongo"},
    {"BarinadeStage", "Fighting Barinade"},
    {"PhantomGanonStage", "Fighting Phantom Ganon"},
    {"VolvagiaStage", "Fighting Volvagia"},
    {"MorphaStage", "Fighting Morpha"},
    {"BongoBongoStage", "Fighting Bongo Bongo"},
    {"TwinrovaStage", "Fighting Twinrova"},
    {"GanondorfStage", "Fighting Ganondorf"},
    {"GanonStage", "Fighting Ganon"},
    {"FinalBattleField", "The Final Battle"},
    /* the fights inside a dungeon */
    {"BigOctoStage", "Fighting Big Octo"},
    {"DarkLinkStage", "Fighting Dark Link"},
    {"DeadHandStage", "Fighting Dead Hand"},
    {"DinolfosStage", "Fighting Dinolfos"},
    {"FlareDancerStage", "Fighting Flare Dancer"},
    {"GerudoThiefStage", "Fighting the Gerudo"},
    {"IronKnuckleBlackStage", "Fighting an Iron Knuckle"},
    {"IronKnuckleWhiteStage", "Fighting an Iron Knuckle"},
    {"LizalfosStage", "Fighting Lizalfos"},
    {"MegStage", "Fighting Meg"},
    {"NaboruStage", "Fighting Nabooru"},
    {"StalfosStage", "Fighting Stalfos"},
    {"StalfosWeakStage", "Fighting Stalfos"},
    {"WolfosStage", "Fighting Wolfos"},
    {"WolfosWhiteStage", "Fighting White Wolfos"},
    /* the rest */
    {"BossRushIntervalRoom", "Boss Rush"},
    {"EnemyMuseum", "Enemy Museum"},
    {"Demo_ShootingLocation_Tapestry_ED", "The Ending"},
};

static int strip(char *s, const char *suffix) {
    const size_t n = strlen(s), m = strlen(suffix);
    if (n <= m || strcmp(s + n - m, suffix)) return 0;
    s[n - m] = 0;
    return 1;
}

static const char *lookup(const char *key) {
    for (size_t i = 0; i < sizeof names / sizeof *names; i++)
        if (!strcmp(names[i].map, key)) return names[i].name;
    return NULL;
}

int hos_oot_area_name(const char *map, char *out, size_t cap) {
    if (!cap) return 0;
    out[0] = 0;
    if (!map || !*map) return 0;
    char key[96];
    snprintf(key, sizeof key, "%s", map);
    const int mq = strip(key, "_MQ");   /* (Master Quest's: the same place) */
    const char *name = lookup(key);
    if (!name) {
        strip(key, "_Alternative");
        if (!(name = lookup(key)) && (strip(key, "_Child") || strip(key, "_Adult"))) name = lookup(key);
    }
    if (name) {
        snprintf(out, cap, "%s%s", name, mq ? " (Master Quest)" : "");
        return 1;
    }
    /* (not in the table: its name spelt out, a space before each capital that follows a small letter, _ as a space) */
    size_t o = 0;
    for (const char *p = key; *p && o + 2 < cap; p++) {
        const char c = *p == '_' ? ' ' : *p;
        if (p > key && c >= 'A' && c <= 'Z' && p[-1] >= 'a' && p[-1] <= 'z' && o + 3 < cap) out[o++] = ' ';
        if (!(c == ' ' && o && out[o - 1] == ' ')) out[o++] = c;
    }
    while (o && out[o - 1] == ' ') o--;
    out[o] = 0;
    return 1;
}
