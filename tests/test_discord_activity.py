"""The Discord status's text: the place Link is and his age, over the version; the version alone on the title; the
player's template.

host/oot_area_names.c names each of the game's maps (the 80 of the supported build, by their names alone: no game file)
in English, and spells out one it does not know. host/discord_activity.c reads the discord_text=, discord_image= and
discord_image_text= lines of the options file and fills the status's template: its fields ({location}, {age}, {map},
{version}, {build}, {commit}, {release}), {break} between its two lines, {name|text} only when name is not empty
(nesting), {{ }} braces; what an empty part leaves ("[]", spaces, a dash) taken off; a line cut to Discord's 128 bytes at
a whole letter and left out under 2. Hookshot's version in parts, from git describe's forms. The activity's JSON is
parsed here (quotes, backslashes and control characters in the player's text). The in-map rule: a known map and Link
seen within 90 s. host/discord_rpc.c is checked in its source for using them and for sending a change within 15 s.

    python -I -B tests/test_discord_activity.py [--cc gcc]
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cc", "gcc"]

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "host"

# (the supported build's maps, Banc/GameBancParam: names only)
MAPS = """BarinadeStage BigOctoStage BongoBongoStage BossRushIntervalRoom BottomOfTheWell CrumblingGanonCastle DarkLinkStage
DeadHandStage DeathMountainCrater_Adult DeathMountainCrater_Child DeathMountainTrail_Adult DeathMountainTrail_Child
Demo_ShootingLocation_Tapestry_ED DesertColossus_Adult DesertColossus_Child DinolfosStage DodongoCavern DodongoStage
EnemyMuseum FinalBattleField FireTemple FlareDancerStage ForestTemple GanonCastle GanonStage GanondorfStage
GerudoFortress_Adult GerudoThiefStage GerudoTrainingGround GerudoValley_Adult GerudoValley_Child GohmaStage GoronCity_Adult
GoronCity_Child Graveyard_Adult Graveyard_Child HountedWasteland_Adult HountedWasteland_Child HyruleCastle HyruleField_Adult
HyruleField_Child IceCavern InsideGanonCastle InsideJabuJabuBelly InsideTheDekuTree IronKnuckleBlackStage
IronKnuckleWhiteStage KakarikoVillage_Adult KakarikoVillage_Adult_Alternative KakarikoVillage_Child KokiriForest_Adult
KokiriForest_Child LizalfosStage LostWood_Adult LostWood_Child Market_Adult Market_Child MegStage MorphaStage NaboruStage
PhantomGanonStage SacredForestMeadow_Adult SacredForestMeadow_Child SageRoom ShadowTemple SpiritTemple StalfosStage
StalfosWeakStage TwinrovaStage VolvagiaStage WaterTemple WolfosStage WolfosWhiteStage ZoraDomain_Adult
ZoraDomain_Adult_Alternative ZoraDomain_Child ZoraFountain_Adult ZoraFountain_Child ZoraRiver_Adult ZoraRiver_Child""".split()

HARNESS = r"""
#include <stdio.h>
#include <string.h>
#include "oot_area_names.h"
#include "discord_activity.h"

static int failures;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); failures++; } } while (0)

static int named(const char *map, const char *want) {
    char out[128];
    hos_oot_area_name(map, out, sizeof out);
    if (strcmp(out, want)) { fprintf(stderr, "  %s: \"%s\", not \"%s\"\n", map, out, want); return 0; }
    return 1;
}

/* the template's two lines against what is wanted */
static int lines(const char *tmpl, const DiscordFields *f, const char *one, const char *two) {
    char a[200], b[200];
    discord_lines(tmpl, f, a, sizeof a, b, sizeof b);
    if (strcmp(a, one) || strcmp(b, two)) { fprintf(stderr, "  %s: [%s] [%s], not [%s] [%s]\n", tmpl ? tmpl : "(null)", a, b, one, two); return 0; }
    return 1;
}

int main(int argc, char **argv) {
    char out[1600];
    /* every map of the game: a name from the table (no _, no "Stage"), one line */
    for (int i = 1; i < argc; i++) {
        hos_oot_area_name(argv[i], out, sizeof out);
        if (!out[0] || strchr(out, '_') || strstr(out, "Stage")) { CHECK(0); fprintf(stderr, "  %s -> \"%s\"\n", argv[i], out); }
        printf("AREA %s\t%s\n", argv[i], out);
    }
    CHECK(named("KakarikoVillage_Adult", "Kakariko Village") && named("KakarikoVillage_Child", "Kakariko Village"));
    CHECK(named("KakarikoVillage_Adult_Alternative", "Kakariko Village"));
    CHECK(named("WaterTemple", "Water Temple") && named("WaterTemple_MQ", "Water Temple (Master Quest)"));
    CHECK(named("Market_Adult", "Market Ruins") && named("Market_Child", "Hyrule Market"));
    CHECK(named("HountedWasteland_Adult", "Haunted Wasteland") && named("GohmaStage", "Fighting Queen Gohma"));
    CHECK(named("SomeNewPlace_Child", "Some New Place") && named("New_Room", "New Room") && named("ABCRoom", "ABCRoom"));
    CHECK(!hos_oot_area_name("", out, sizeof out) && !out[0] && !hos_oot_area_name(NULL, out, sizeof out));
    CHECK(hos_oot_area_name("KokiriForest_Child", out, 6) && !strcmp(out, "Kokir"));   /* (cut to the room there is) */

    /* the template */
    const DiscordFields play = {"Kakariko Village", "KakarikoVillage_Adult", "Adult", "v3.0.0", "RC7", "25-g7a029028", "v3.0.0-rc.7-25-g7a029028", "", "", "", "", ""};
    const DiscordFields title = {"", "", "", "v3.0.0", "RC7", "25-g7a029028", "v3.0.0-rc.7-25-g7a029028", "", "", "", "", ""};
    const DiscordFields release = {"Kakariko Village", "KakarikoVillage_Adult", "Adult", "v3.0.0", "", "", "v3.0.0", "", "", "", "", ""};
    const DiscordFields none = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    const DiscordFields stats = {"Kakariko Village", "KakarikoVillage_Adult", "Adult", "v3.0.0", "RC7", "25-g7a029028", "v3.0.0-rc.7-25-g7a029028", "2.75", "3", "120", "12", "2"};
    const DiscordFields no_tokens = {"Kokiri Forest", "KokiriForest_Child", "Child", "v3.0.0", "", "", "v3.0.0", "3", "3", "0", "", ""};
    DiscordSettings s;
    discord_settings_default(&s);
    CHECK(lines(s.status, &play, "Kakariko Village - Adult", ""));   /* (no stats known: the second line empty) */
    CHECK(lines(s.status, &title, "v3.0.0 RC7", ""));
    {   /* (a nightly has no version number: its commit is shown, on the line and in the hover text) */
        const DiscordFields nightly = {"", "", "", "", "N486", "09a20cd6", "nightly-486-09a20cd6", "", "", "", "", ""};
        const DiscordFields nightly_play = {"Kokiri Forest", "KokiriForest_Child", "Child", "", "N486", "09a20cd6", "nightly-486-09a20cd6", "3", "3", "0", "", ""};
        CHECK(lines(s.status, &nightly, "N486 09a20cd6", "") && lines(s.image_text, &nightly, "Hookshot N486 09a20cd6", ""));
        CHECK(lines(s.status, &nightly_play, "Kokiri Forest - Child", "\xe2\x9d\xa4\xef\xb8\x8f 3/3"));
    }   /* (no location: the first part left out) */
    CHECK(lines(s.status, &release, "Kakariko Village - Adult", ""));
    CHECK(lines("{!location|{version} {build}}{location|{location}}", &title, "v3.0.0 RC7", "") && lines("{!location|t}{location|{location}}", &play, "Kakariko Village", ""));
    CHECK(lines("{!unknown|x}", &play, "{!unknown|x}", "") && lines("{!location}", &play, "{!location}", ""));   /* (not a negation: text) */
    /* (the stats: hearts and tokens with their emoji, a part that is not known left out with its middle dot) */
    CHECK(lines(s.status, &stats, "Kakariko Village - Adult",
                "\xe2\x9d\xa4\xef\xb8\x8f 2.75/3 \xc2\xb7 \xf0\x9f\x95\xb7\xef\xb8\x8f 12 \xc2\xb7 \xf0\x9f\x92\x97 2"));
    CHECK(lines(s.status, &no_tokens, "Kokiri Forest - Child", "\xe2\x9d\xa4\xef\xb8\x8f 3/3"));
    CHECK(lines("{rupees} \xc2\xb7 {skulltulas}", &no_tokens, "0", ""));   /* (a dot left at the end goes) */
    CHECK(lines("\xc2\xb7 {skulltulas} \xc2\xb7", &no_tokens, "", ""));
    {
        char h[16];
        discord_hearts_text(12, h, sizeof h); CHECK(!strcmp(h, "3"));
        discord_hearts_text(11, h, sizeof h); CHECK(!strcmp(h, "2.75"));
        discord_hearts_text(2, h, sizeof h); CHECK(!strcmp(h, "0.5"));
        discord_hearts_text(1, h, sizeof h); CHECK(!strcmp(h, "0.25"));
        discord_hearts_text(0, h, sizeof h); CHECK(!strcmp(h, "0"));
        discord_hearts_text(80, h, sizeof h); CHECK(!strcmp(h, "20"));
        discord_hearts_text(-1, h, sizeof h); CHECK(!h[0]);
    }
    CHECK(lines(s.image_text, &release, "Hookshot v3.0.0", "") && lines(s.image_text, &play, "Hookshot v3.0.0 RC7", ""));
    CHECK(lines("{location} - {age}{break}Hookshot {version} {build} [{commit}]", &title, "", "Hookshot v3.0.0 RC7 [25-g7a029028]"));
    CHECK(lines("{location|At {location}{age| as {age} Link}}", &play, "At Kakariko Village as Adult Link", ""));   /* (nesting) */
    CHECK(lines("{location|At {location}{age| as {age} Link}}", &title, "", ""));
    CHECK(lines("{{location}} {map} {release}", &play, "{location} KakarikoVillage_Adult v3.0.0-rc.7-25-g7a029028", ""));
    CHECK(lines("{unknown} {unknown|x} {location", &play, "{unknown} {unknown|x} {location", ""));   /* (not fields: text) */
    CHECK(lines("{location|{{a|b}", &play, "{a|b", "") && lines("}} {{", &play, "} {", ""));
    CHECK(lines("one{break}two{break}three", &play, "one", "two three"));   /* (two lines: a later break is a space) */
    CHECK(lines("Hookshot {version} ({build})  x", &release, "Hookshot v3.0.0 x", ""));
    CHECK(lines("[x] (y) [ ]", &play, "[x] (y)", ""));
    CHECK(lines("{age} - {location}", &title, "", "") && lines("Spirit-Temple - x-y", &play, "Spirit-Temple - x-y", ""));
    CHECK(lines("   {location}   ", &none, "", "") && lines(NULL, &play, "", ""));
    {   /* (Discord's 128 bytes, at a whole letter) */
        char many[256] = "", a[200], b[8];
        for (int i = 0; i < 100; i++) strcat(many, "\xc3\xa9");
        discord_lines(many, &play, a, sizeof a, b, sizeof b);
        CHECK(strlen(a) == 128 && (a[127] & 0xc0) != 0xc0);
        strcpy(many, "x");
        for (int i = 0; i < 100; i++) strcat(many, "\xc3\xa9");
        discord_lines(many, &play, a, sizeof a, b, sizeof b);
        CHECK(strlen(a) == 127);   /* (128 would split a letter) */
        char deep[256] = "";   /* (nesting past 8: left out, not followed) */
        for (int i = 0; i < 12; i++) strcat(deep, "{location|");
        strcat(deep, "x");
        for (int i = 0; i < 12; i++) strcat(deep, "}");
        discord_lines(deep, &play, a, sizeof a, b, sizeof b);
        CHECK(!a[0]);
    }

    /* the version in parts */
    {
        char v[64], b[64], c[64], r[64];
#define PARTS(raw, ev, eb, ec, er) do { discord_version_parts(raw, v, b, c, r, sizeof v); \
        if (strcmp(v, ev) || strcmp(b, eb) || strcmp(c, ec) || strcmp(r, er)) { CHECK(0); \
            fprintf(stderr, "  %s: [%s] [%s] [%s] [%s]\n", raw, v, b, c, r); } } while (0)
        PARTS("v3.0.0-rc.7-25-g7a029028 (source 1234abcd)", "v3.0.0", "RC7", "25-g7a029028", "v3.0.0-rc.7-25-g7a029028");
        PARTS("v3.0.0-rc7-25-g7a029028", "v3.0.0", "RC7", "25-g7a029028", "v3.0.0-rc7-25-g7a029028");
        PARTS("v3.0.0 (source 1234abcd)", "v3.0.0", "", "", "v3.0.0");
        PARTS("v3.0.0-rc.7", "v3.0.0", "RC7", "", "v3.0.0-rc.7");
        PARTS("v3.1.0-beta.2-3-gabcdef1-dirty", "v3.1.0", "Beta 2", "3-gabcdef1-dirty", "v3.1.0-beta.2-3-gabcdef1-dirty");
        PARTS("v3.0.0-dirty", "v3.0.0", "", "dirty", "v3.0.0-dirty");
        PARTS("2.0.12+abc", "v2.0.12", "", "", "2.0.12");
        PARTS("7a029028", "", "", "7a029028", "7a029028");
        PARTS("nightly-70", "", "N70", "", "nightly-70");
        PARTS("nightly-486-09a20cd6", "", "N486", "09a20cd6", "nightly-486-09a20cd6");   /* (a nightly's own name) */
        PARTS("nightly-486-09a2", "nightly-486-09a2", "", "", "nightly-486-09a2");   /* (too short for a hash) */
        PARTS("unknown", "unknown", "", "", "unknown");
        PARTS("", "", "", "", "");
    }

    /* the settings */
    discord_settings_default(&s);
    CHECK(discord_settings_line(&s, "discord_text={location}{break}x\r\n") && !strcmp(s.status, "{location}{break}x"));
    CHECK(discord_settings_line(&s, "discord_image=\n") && !s.image[0]);
    CHECK(!discord_settings_line(&s, "discord_app_id=123\n") && !discord_settings_line(&s, "game_frame_rate_cap=2\n"));
    /* (the Insert menu's own row, Discord status on or off, is discord_status=1: never the text) */
    CHECK(!discord_settings_line(&s, "discord_status=1\n") && !discord_settings_line(&s, "discord_status=0\n"));
    CHECK(!strcmp(s.status, "{location}{break}x"));
    CHECK(!discord_settings_line(&s, " discord_text=y\n") && !strcmp(s.status, "{location}{break}x"));
    CHECK(discord_settings_line(&s, "discord_text=a\x1e" "b") && !strcmp(s.status, "a b"));   /* (no stray break) */

    /* the in-map rule */
    CHECK(discord_in_map(1, 1000, 1000 + DISCORD_IN_MAP_NS - 1) && !discord_in_map(1, 1000, 1000 + DISCORD_IN_MAP_NS));
    CHECK(!discord_in_map(0, 1000, 1001) && !discord_in_map(1, 0, 5));

    /* the activity's JSON, parsed by the check */
    discord_settings_default(&s);
    printf("JSON play %s\n", discord_activity_json(&s, &play, 7, 42, 1700000000, out, sizeof out) ? out : "-");
    printf("JSON title %s\n", discord_activity_json(&s, &title, 8, 42, 1700000000, out, sizeof out) ? out : "-");
    discord_settings_line(&s, "discord_text=\"quoted\" \\ back\tslash {location}{break}x");   /* (a second line under 2: left out) */
    discord_settings_line(&s, "discord_image=https://example.org/pic.png");
    discord_settings_line(&s, "discord_image_text=");
    printf("JSON custom %s\n", discord_activity_json(&s, &play, 9, 42, 1700000000, out, sizeof out) ? out : "-");
    discord_settings_line(&s, "discord_image=not a key!");
    discord_settings_line(&s, "discord_text=");
    printf("JSON bare %s\n", discord_activity_json(&s, &play, 10, 42, 1700000000, out, sizeof out) ? out : "-");
    CHECK(!discord_activity_json(&s, &play, 10, 42, 1700000000, out, 40) && !out[0]);   /* (too small: nothing) */
    return failures ? 1 : 0;
}
"""


def check_source():
    rpc = (HOST / "discord_rpc.c").read_text(encoding="utf-8")
    for token, what in (("discord_activity_json(", "the activity built by discord_activity.c"),
                        ("discord_in_map(", "the in-map rule"), ("hos_oot_player_seen_ns()", "when Link was last seen"),
                        ("hos_oot_area_name(", "the place's name"), ("discord_settings_line(", "the options file's lines"),
                        ("discord_version_parts(", "the version in parts"),
                        ("in_map ? area : \"\"", "no location when Link is not in a map"),
                        ("hos_oot_stats(", "Link's hearts, rupees and tokens"),
                        ("sent_at + 15000", "a change sent within 15 s, not more often")):
        if token not in rpc:
            raise SystemExit(f"host/discord_rpc.c: {what} is gone ({token})")
    main = (HOST / "hos_main.c").read_text(encoding="utf-8")
    if "Item_Token_Gold" not in main or "int hos_oot_stats(" not in main:
        raise SystemExit("host/hos_main.c: the Gold Skulltula tokens or hos_oot_stats are gone")
    if "int64_t hos_oot_player_seen_ns(void)" not in main:
        raise SystemExit("host/hos_main.c: hos_oot_player_seen_ns is gone")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    a = p.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="discord-activity-") as tmp:
        root = Path(tmp).resolve()
        src, exe = root / "discord_test.c", root / "discord_test"
        src.write_text(HARNESS, encoding="utf-8")
        windows = os.name == "nt"
        extra = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS} if windows else {}
        build = subprocess.run([a.cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(HOST), str(src),
                                str(HOST / "oot_area_names.c"), str(HOST / "discord_activity.c"), "-o", str(exe)], **extra)
        if build.returncode:
            raise SystemExit("the host did not compile: see the compiler's output above")
        run = subprocess.run([str(exe), *MAPS], capture_output=True, text=True, encoding="utf-8", **extra)
        print(run.stderr, end="")
        if run.returncode:
            raise SystemExit("the Discord activity check failed")
    areas = dict(line[5:].split("\t", 1) for line in run.stdout.splitlines() if line.startswith("AREA "))
    if len(areas) != len(MAPS):
        raise SystemExit("not every map was named")
    js = {}
    for line in run.stdout.splitlines():
        if line.startswith("JSON "):
            _, name, body = line.split(" ", 2)
            try:
                js[name] = json.loads(body)
            except ValueError as e:
                raise SystemExit(f"the {name} activity is not JSON ({e}): {body}")
    act = {k: v["args"]["activity"] for k, v in js.items()}
    stamp = {"start": 1700000000}
    want = {
        "play": {"details": "Kakariko Village - Adult", "timestamps": stamp,
                 "assets": {"large_image": "hookshot", "large_text": "Hookshot v3.0.0 RC7"}},
        "title": {"details": "v3.0.0 RC7", "timestamps": stamp,
                  "assets": {"large_image": "hookshot", "large_text": "Hookshot v3.0.0 RC7"}},
        "custom": {"details": "\"quoted\" \\ back slash Kakariko Village", "timestamps": stamp,
                   "assets": {"large_image": "https://example.org/pic.png"}},
        "bare": {"timestamps": stamp, "assets": {"large_image": "hookshot"}},
    }
    for name, expect in want.items():
        if act.get(name) != expect:
            raise SystemExit(f"the {name} activity: {act.get(name)!r}, not {expect!r}")
    if js["play"]["cmd"] != "SET_ACTIVITY" or js["play"]["args"]["pid"] != 42 or js["play"]["nonce"] != "7":
        raise SystemExit("the frame's command, pid or nonce is wrong")
    print("PASS: the Discord status names Link's place and age over his stats, the version where there is no place, "
          "and takes the player's template")


if __name__ == "__main__":
    main()
