"""The debug teleport sends a dropped request once more (host/oot_teleport_check.h, host/hos_main.c oot_teleport_followup).

A request sent in the first frames after a save has loaded was answered 0 and then dropped by the game: nothing reserved, no
event, Link where he was; the control bridge's `teleport --arrive` timed out, in about one scripted start in four. The
follow-up 1.5 s later read exactly that and only logged it. Game-free: the decision compiled with its cases, and the follow-up
run on its own: teleport's state, the start of oot_teleport (its refusals, the 8 s guard among them), the bookkeeping of a
request sent and oot_teleport_followup, all cut from host/hos_main.c, with the game's reads stubbed. It sends a dropped
request again, once, to the same destination, also when a call in between was refused (which clears teleport.banc, the
bridge's reply), and leaves alone one the game took (answered other than 0) or one with an event playing or a map loading
at the check.

    python -I -B tests/test_teleport_check.py [--cc gcc]   (default: $CC, else gcc)
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import srccut   # noqa: E402

FIXTURE = r"""
#include <stdio.h>
#include "oot_teleport_check.h"
static int fails;
static void check(int got, int want, const char *what) {
    if (got != want) { printf("FAIL: %s: %d, wanted %d\n", what, got, want); fails++; }
}
int main(void) {
    check(oot_teleport_dropped("", "HyruleField_Adult", "Market_Adult", 0, 0, 0, 0), 1, "nothing reserved, no event, elsewhere: dropped");
    check(oot_teleport_dropped("Market_Adult", "HyruleField_Adult", "Market_Adult", 0, 0, 0, 0), 0, "reserved: under way");
    check(oot_teleport_dropped("", "HyruleField_Adult", "Market_Adult", 1, 0, 0, 0), 0, "an event playing: under way");
    check(oot_teleport_dropped("", "HyruleField_Adult", "Market_Adult", 0, 1, 0, 0), 0, "the map loading: under way");
    check(oot_teleport_dropped("", "Market_Adult", "Market_Adult", 0, 0, 0, 0), 0, "arrived");
    check(oot_teleport_dropped("", "HyruleField_Adult", "Market_Adult", 0, 0, 0, 1), 0, "sent again already: not a third time");
    check(oot_teleport_dropped("", "HyruleField_Adult", NULL, 0, 0, 0, 0), 0, "nothing was sent");
    check(oot_teleport_dropped("", "", "Market_Adult", 0, 0, 0, 0), 1, "no map read: still not there");
    check(oot_teleport_dropped("", "HyruleField_Adult", "Market_Adult", 0, 0, 1, 0), 0, "the game took it (answered other than 0): only slow, not dropped");
    if (!fails) printf("OK: a dropped teleport is sent once more; one under way or arrived is left alone\n");
    return fails != 0;
}
"""

# The follow-up run on its own. @STATE@, @PRELUDE@, @SENT@ and @FOLLOWUP@ are cut from host/hos_main.c: oot_teleport is its
# own start (every refusal before the game is asked) and its own bookkeeping of a request sent, around a stand-in for the
# request the game takes.
SEQUENCE = r"""
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "oot_teleport_check.h"
typedef struct Ctx { int unused; } Ctx;
#define OOT_EVENTMGR_SLOT 0
static int64_t now_ns;
static char reserved[80], current[80];
static int sends;
static unsigned sent_index;
static int64_t hos_now_ns(void) { return now_ns; }
static void hos_log(const char *fmt, ...) { (void)fmt; }
static int event_playing, loading_now;
static uint64_t answer;
#define EVENTS 0x1000u
static uint64_t oot_singleton(Ctx *c, int slot) { (void)c; return slot == OOT_EVENTMGR_SLOT ? EVENTS : 0; }
static uint64_t recomp_load64(Ctx *c, uint64_t at) { (void)c; return at == EVENTS + 104 ? (uint64_t)event_playing : 0; }
static void oot_gamedata_string(Ctx *c, uint32_t hash, char *out, size_t size) {
    (void)c;
    snprintf(out, size, "%s", hash == 0xee27bf71u ? reserved : current);
}
static int oot_gamedata_bool(Ctx *c, uint32_t hash, int *out) { (void)c; *out = hash == 0xef2b60c2u ? loading_now : 0; return 1; }
typedef struct { const char *adult_banc; } OotDestination;
static const OotDestination oot_destinations[] = {{"Kokiri_Adult"}, {"Kakariko_Adult"}, {"Lake_Adult"}, {"Market_Adult"}};
@STATE@
@PRELUDE@
    const char *banc = oot_destinations[index].adult_banc, *start = "Start", *boot = "Boot", *event = "Event";
    const int ok = 1, child = 0, with_starter = 1;
    const uint64_t result = answer;   /* (the event manager's answer: 0, as every dropped request was answered) */
    sends++, sent_index = index;
    (void)c, (void)actor;
@SENT@
    return ok;
}
@FOLLOWUP@
static int fails;
static void check(long got, long want, const char *what) {
    if (got != want) { printf("FAIL: %s: %ld, wanted %ld\n", what, got, want); fails++; }
}
static void reset(void) {
    memset(&teleport, 0, sizeof teleport);
    sends = 0, sent_index = 99, now_ns = 1000000000000LL;
    event_playing = 0, loading_now = 0, answer = 0;
    strcpy(reserved, ""), strcpy(current, "HyruleField_Adult");
}
#define AT(ms) (now_ns = 1000000000000LL + (ms) * 1000000LL)
int main(void) {
    Ctx c = {0};
    reset();   /* A sent and dropped: sent again once, never a third time */
    check(oot_teleport(&c, 1, 3), 1, "A is sent");
    AT(1600); oot_teleport_followup(&c, 1);
    check(sends, 2, "the follow-up sends a dropped A again");
    check(sent_index, 3, "to A's destination");
    AT(3200); oot_teleport_followup(&c, 1);
    AT(5000); oot_teleport_followup(&c, 1);
    check(sends, 2, "still dropped: not sent a third time");

    reset();   /* A sent and dropped, B refused under the 8 s guard before the follow-up: A is still sent again */
    oot_teleport(&c, 1, 3);
    AT(500);
    check(oot_teleport(&c, 1, 1), 0, "B, 0.5 s after A, is refused");
    check(teleport.banc == NULL, 1, "the refusal clears the bridge's reply");
    AT(1600); oot_teleport_followup(&c, 1);
    check(sends, 2, "the follow-up still sends the dropped A again");
    check(sent_index, 3, "to A's destination, not B's");
    AT(3200); oot_teleport_followup(&c, 1);
    check(sends, 2, "and once only");

    reset();   /* A arrived: left alone */
    oot_teleport(&c, 1, 3);
    strcpy(current, "Market_Adult");
    AT(1600); oot_teleport_followup(&c, 1);
    check(sends, 1, "an arrived request is not sent again");

    reset();   /* A dropped, but an event playing at the check: left alone */
    oot_teleport(&c, 1, 3);
    event_playing = 1;
    AT(1600); oot_teleport_followup(&c, 1);
    check(sends, 1, "an event playing at the check: not sent again");

    reset();   /* A dropped, but a map loading at the check: left alone */
    oot_teleport(&c, 1, 3);
    loading_now = 1;
    AT(1600); oot_teleport_followup(&c, 1);
    check(sends, 1, "a map loading at the check: not sent again");

    reset();   /* A taken by the event manager (answered other than 0), nothing reserved yet: only slow, left alone */
    answer = 1;
    oot_teleport(&c, 1, 3);
    AT(1600); oot_teleport_followup(&c, 1);
    check(sends, 1, "a request the game took is not sent again");

    reset();   /* nothing sent, only a refusal: nothing to send again */
    teleport.requested_at = now_ns, teleport.check_at = now_ns;
    check(oot_teleport(&c, 1, 3), 0, "refused under the guard");
    AT(1600); oot_teleport_followup(&c, 1);
    check(sends, 0, "no request was sent: the follow-up sends none");
    if (!fails) printf("OK: the follow-up sends a dropped request again once, also after a refused call in between\n");
    return fails != 0;
}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    a = p.parse_args()
    src = (ROOT / "host" / "hos_main.c").read_text(encoding="utf-8")
    where = "host/hos_main.c"
    state = srccut.between(src, "static struct {\n    int64_t requested_at, check_at;", "\nstatic int oot_teleport(", where=where)
    prelude = srccut.between(src, "static int oot_teleport(Ctx *c, uint64_t actor, unsigned index) {",
                             "    const uint64_t events = oot_singleton(c, OOT_EVENTMGR_SLOT);\n    A64Fn make_arg", where=where)
    sent = srccut.between(src, "    teleport.requested_at = now;\n    teleport.check_at", "    return ok;   /* (sent;", where=where)
    followup = srccut.block(src, "static void oot_teleport_followup(", where=where)
    sequence = (SEQUENCE.replace("@STATE@", state).replace("@PRELUDE@", prelude).replace("@SENT@", sent)
                .replace("@FOLLOWUP@", followup))
    ok = True
    if "oot_teleport_followup(c, actor);" not in src:
        print("FAIL: the player's update gives the follow-up Link's actor, to send the request with")
        ok = False
    if not (ROOT / "host" / "oot_teleport_check.h").is_file():
        print("FAIL: no host/oot_teleport_check.h")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in (("decision", FIXTURE), ("sequence", sequence)):
            c, exe = Path(tmp) / (name + ".c"), Path(tmp) / (name + ".exe")
            c.write_text(text, encoding="utf-8")
            subprocess.run([a.cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wno-unused-variable", "-I", str(ROOT / "host"), str(c), "-o", str(exe)],
                           check=True)
            r = subprocess.run([str(exe)], capture_output=True, text=True)
            print(r.stdout, end="")
            ok = ok and r.returncode == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
