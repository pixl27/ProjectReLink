/* (hookshot) The boss heart containers Link has taken, for the Discord status's pieces of heart (host/oot_heart_containers.c) */
#pragma once
#include <stddef.h>
#include <stdint.h>

#define HEART_CONTAINERS 8   /* (one in each of the eight dungeons with a boss: HEART_DUNGEONS in the .c) */

/* Where name is in the SARC archive d (n bytes): its first byte at *start, the end at *end. 1 found, 0 not (or not a SARC) */
int heart_sarc_find(const uint8_t *d, size_t n, const char *name, size_t *start, size_t *end);

/* The 64-bit placement ids of the actors whose Gyaml is gyaml in the map file d (n bytes of BYML, either byte order: a root
   dictionary with Actors, a list of dictionaries with Gyaml and Hash), at most cap into ids. How many, -1 not a map file */
int heart_byml_actor_ids(const uint8_t *d, size_t n, const char *gyaml, uint64_t *ids, int cap);

/* Pieces of heart found in all, from Link's most life in quarter hearts, the boss containers he has taken and the pieces he
   holds (0-3): 4 * (hearts - 3 - containers) + held. -1 when one is not known or they do not add up */
int heart_pieces_total(int max_life, int containers, int held);

/* Reads the containers' ids from the RomFS once (a few map files: on the Discord status's thread, not the game's) */
void hos_oot_heart_containers_load(void);

/* (on the game's thread, host/hos_main.c) how many of the containers Link has taken, -1 not known (not loaded, or a flag
   that cannot be read) */
int hos_oot_heart_containers_taken(void);
