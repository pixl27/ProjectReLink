/* (hookshot) The place a map of the game is, in English, for the Discord status (oot_area_names.c). No game state. */
#pragma once
#include <stddef.h>

/* The place map (the game's own map name, Sequence_CurrentBanc: "KakarikoVillage_Adult") is, in English ("Kakariko
   Village"), into out. A map the table does not know is spelt out from its name ("SomeNewPlace_Child": "Some New
   Place"). 0 for an empty map (out empty), 1 otherwise. */
int hos_oot_area_name(const char *map, char *out, size_t cap);
