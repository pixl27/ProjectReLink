/* (hookshot) What the Discord status says (discord_activity.c): the player's settings, the template filled in, the
   activity's JSON. No I/O and no game state: host/discord_rpc.c reads the settings and the game's place and sends it. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* The status's text, from hookshot_options.ini (discord_text=, discord_image=, discord_image_text=; see
   discord_activity.c). The texts are templates:
     {location}  the place Link is (Kakariko Village), empty when he is not in a map (the title)
     {age}       Child or Adult, {map} the game's own name for the map: both empty when not in a map
     {hearts} 2.75, {max_hearts} 3: Link's hearts (quarters as .25 .5 .75); {rupees} 120; {skulltulas} the Gold Skulltula
                 tokens; {heart_pieces} the pieces of heart he has found (host/oot_heart_containers.c): all empty when not in a map
     {version} v3.0.0, {build} RC7, {commit} 25-g7a029028, {release} v3.0.0-rc.7-25-g7a029028: Hookshot's version
     {break}     the end of the status's first line: what follows is its second, smaller line
     {name|text} text (itself a template) when name is not empty, else nothing
     {!name|text} text when name is empty (the title: {!location|...})
     {{ and }}   a brace */
typedef struct {
    char status[512];
    char image[260], image_text[256];   /* the large picture (an art asset's key or an https:// link) and its hover text */
} DiscordSettings;

/* What the templates are filled with (NULL or "" when not known). */
typedef struct {
    const char *location, *map, *age, *version, *build, *commit, *release;
    const char *hearts, *max_hearts, *rupees, *skulltulas, *heart_pieces;
} DiscordFields;

/* Life in quarter hearts as hearts, in plain digits: 12 "3", 11 "2.75", 2 "0.5"; "" for a negative (not known). */
void discord_hearts_text(int quarters, char *out, size_t cap);

/* Hookshot's version ("v3.0.0-rc.7-25-g7a029028-dirty (source ...)", git describe's) in parts: version "v3.0.0", build
   "RC7" (rc.7 and rc7 alike; "beta.2" "Beta 2"; none on a release), commit "25-g7a029028-dirty" (none on the release's own
   commit), release the whole before " (source". A name that is not v<numbers>: all of it the version (a nightly,
   "nightly-486-09a20cd6": build "N486", commit "09a20cd6"), or a bare hash the commit. Each out holds cap bytes. */
void discord_version_parts(const char *raw, char *version, char *build, char *commit, char *release, size_t cap);

void discord_settings_default(DiscordSettings *s);
/* One line of the options file: a discord_text=, discord_image= or discord_image_text= line sets it (an empty status:
   none shown but the picture); any other line is left alone. 1 when the line was one of these. */
int discord_settings_line(DiscordSettings *s, const char *line);

/* tmpl filled in as the status's two lines (first: before {break}; second: after it, "" when there is none). Each line
   cleaned of what empty parts leave ("[]", "()", runs of spaces, a dash at either end), its outer spaces off, cut to
   Discord's 128 bytes at a whole UTF-8 letter. */
void discord_lines(const char *tmpl, const DiscordFields *f, char *first, size_t first_cap, char *second, size_t second_cap);

/* The SET_ACTIVITY frame's JSON for these settings and fields: the first line as Discord's details, the second as its
   state, each left out when shorter than Discord's 2 letters. Its length, 0 when it does not fit. */
size_t discord_activity_json(const DiscordSettings *s, const DiscordFields *f, unsigned nonce, unsigned long pid,
                             long long started, char *out, size_t cap);

/* Whether the player is in a map: one known, and Link's update ran within the last DISCORD_IN_MAP_NS (the pause menu, a
   cutscene and a load keep him out of it for a while; the title, for good). */
#define DISCORD_IN_MAP_NS 90000000000LL
int discord_in_map(int map_known, int64_t link_seen_ns, int64_t now_ns);
