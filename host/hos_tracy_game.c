/* (hookshot) HOOKSHOT_TRACY: the text of the zones in the lifted game code (nx2/tracy_funcs.py, docs/development/tracy.md):
   a listed function's zone says who called it and what object it ran on; a dispatch point's zone (an indirect call of a
   function listed with `calls`) says which function it called, f_<target>, and on what object.

   What an object is, as far as the game's memory says, looked up once per object and thread and kept:
     - its class's name, where its vtable has run-time type information (the SDK's and the middleware's classes have; the
       game's own were built without);
     - else a name it carries: the first of its first twelve fields that points at a short printable string. The game's
       named objects (its jobs, modules, threads, heaps) keep a pointer to their name near their start. A guess: a field
       that points at some other string is shown as the name;
     - else its vtable's address, which is the same for every object of one class.
   These names are read from the game: they stay in the capture, on this PC.

   Only this build compiles the file (nx2/build.py leaves it out without HOOKSHOT_TRACY), and nothing here runs unless a
   capture is being taken (the callers test the zone). Read only: guest memory is looked at, never written, and only where
   hos_mem_region says it is mapped. */
#ifdef TRACY_ENABLE
#include "hos.h"
#include "hos_tracy.h"
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* the profiler's client, started by the game's main once it knows it is the game (not the crash window's helper) */
int hos_tracy_started, hos_tracy_game_on;
void hos_tracy_start(void) {
    const char *e = getenv("HOS_TRACY_GAME");
    /* (1: the listed functions' zones, 2: their indirect calls'. Both unless HOS_TRACY_GAME says: =0 none, =funcs or
       =calls that kind alone) */
    hos_tracy_game_on = !e || !*e ? 3 : e[0] == '0' ? 0 : !strcmp(e, "funcs") ? 1 : !strcmp(e, "calls") ? 2 : 3;
    ___tracy_startup_profiler();
    hos_tracy_started = 1;
}

enum { HT_NAMES = 512, HT_NAME_LEN = 48 };
typedef struct { uint64_t obj, vt; char name[HT_NAME_LEN]; } HtName;
static _Thread_local HtName ht_names[HT_NAMES];

static uint64_t rd64(uint64_t a) {
    uint64_t v;
    memcpy(&v, GP(a), 8);
    return v;
}
/* the guest string at p when it reads as a name: 4 to 47 characters of letters, digits and a little punctuation */
static int name_at(uint64_t p, char *out) {
    if (p < 0x10000 || !hos_mem_region(p, HT_NAME_LEN)) return 0;
    const unsigned char *s = (const unsigned char *)GP(p);
    int n = 0, letters = 0;
    for (; n < HT_NAME_LEN - 1 && s[n]; n++) {
        const unsigned char ch = s[n];
        const int letter = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
        if (!letter && !(ch >= '0' && ch <= '9') && !strchr("_:.<>-/ ", ch)) return 0;
        letters += letter;
    }
    if (n < 4 || s[n] || letters < 3) return 0;
    memcpy(out, s, (size_t)n + 1);
    return 1;
}
static const char *object_name(uint64_t o) {
    HtName *e = &ht_names[(o >> 4) & (HT_NAMES - 1)];
    if (!o || !hos_mem_region(o, 0x68)) return "";
    const uint64_t vt = rd64(o);
    if (e->obj == o && e->vt == vt) return e->name;
    e->obj = o, e->vt = vt, e->name[0] = 0;
    if (hos_mem_region(vt - 8, 16) == 1) {   /* (a vtable in the image: its type information's name, if it has one) */
        const uint64_t ti = rd64(vt - 8);
        if (ti && hos_mem_region(ti, 16) == 1 && name_at(rd64(ti + 8), e->name)) return e->name;
    }
    for (int i = 1; i <= 12; i++)
        if (name_at(rd64(o + 8 * (uint64_t)i), e->name)) return e->name;
    snprintf(e->name, sizeof e->name, "vt %#llx", (unsigned long long)vt);
    return e->name;
}

/* The game's zones in the order they were begun, per thread, for a check: Tracy stops a capture at the first zone that
   ends out of turn ("Invalid order of zone begin and end events") and says nothing of which. A zone of ours that ends while
   it is not the newest is logged with what it was (the first 8) */
enum { HT_OPEN = 128 };
static _Thread_local struct { uint32_t id; uint64_t what, lr; } ht_open[HT_OPEN];
_Thread_local int hos_tracy_calls_open;   /* (how many: hos_thread.c's fiber switch looks at it) */
static void opened(TracyCZoneCtx z, uint64_t what, uint64_t lr) {
    if (hos_tracy_calls_open < HT_OPEN) {
        ht_open[hos_tracy_calls_open].id = z.id;
        ht_open[hos_tracy_calls_open].what = what;
        ht_open[hos_tracy_calls_open].lr = lr;
    }
    hos_tracy_calls_open++;
}
static void closed(TracyCZoneCtx z, const char *kind) {
    const int top = --hos_tracy_calls_open;
    if (top >= 0 && top < HT_OPEN && ht_open[top].id == z.id) return;
    static volatile LONG told;
    if (InterlockedIncrement(&told) > 8) return;
    int at = -1;
    for (int i = 0; i < HT_OPEN && i < top + 8; i++)
        if (i >= 0 && ht_open[i].id == z.id) at = i;
    hos_log("tracy: a game zone (%s, id %u) ended out of turn on %s: %d open, it was number %d (f_%llx from %#llx); the newest is "
            "f_%llx from %#llx", kind, z.id, hos_self() ? hos_thread_name(hos_self()) : "?", top + 1, at,
            (unsigned long long)(at >= 0 ? ht_open[at].what : 0), (unsigned long long)(at >= 0 ? ht_open[at].lr : 0),
            (unsigned long long)(top >= 0 && top < HT_OPEN ? ht_open[top].what : 0),
            (unsigned long long)(top >= 0 && top < HT_OPEN ? ht_open[top].lr : 0));
}

/* a listed function's zone (the wrappers nx2/tracy_funcs.py generates) */
TracyCZoneCtx hos_tracy_fn_begin(const struct ___tracy_source_location_data *loc, Ctx *c, uint64_t addr) {
    TracyCZoneCtx z = ___tracy_emit_zone_begin(loc, hos_tracy_started && (hos_tracy_game_on & 1) ? 1 : 0);
    if (z.active) {
        opened(z, addr, c->x[30]);
        HT_ZONE_TEXTF(z, "%s, lr %#llx", object_name(c->x[0]), (unsigned long long)c->x[30]);
    }
    return z;
}
void hos_tracy_fn_end(TracyCZoneCtx *z) {
    if (z->active) closed(*z, "function");
    ___tracy_emit_zone_end(*z);
}
/* a dispatch point's indirect call (the header nx2/tracy_funcs.py compiles its chunk with): the zone around it */
/* (the lifted chunk sees nothing of Tracy, so the zone goes to it and back as 16 bytes of its own, HtCallZone in the
   generated header: Tracy's context has another layout with TRACY_ON_DEMAND than without, and a chunk that held it
   as the smaller one lost each zone's end, which is what stopped the first captures: "Invalid order of zone begin and
   end events") */
typedef struct { uint64_t w[2]; } HtCallZone;
_Static_assert(sizeof(TracyCZoneCtx) <= sizeof(HtCallZone), "Tracy's zone context fits the lifted code's 16 bytes");
HtCallZone hos_tracy_call_begin(Ctx *c, uint64_t target) {
    HT_SRCLOC struct ___tracy_source_location_data loc = { "game call", __func__, __FILE__, (uint32_t)__LINE__, HT_COLOR_GAME };
    HtCallZone held = {{0, 0}};
    TracyCZoneCtx z = ___tracy_emit_zone_begin(&loc, hos_tracy_started && (hos_tracy_game_on & 2) ? 1 : 0);
    if (z.active) {
        opened(z, target, c->x[30]);
        HT_ZONE_TEXTF(z, "f_%llx %s, from %#llx", (unsigned long long)target, object_name(c->x[0]), (unsigned long long)c->x[30]);
    }
    memcpy(&held, &z, sizeof z);
    return held;
}
void hos_tracy_call_end(HtCallZone *held) {
    TracyCZoneCtx z;
    memcpy(&z, held, sizeof z);
    if (z.active) closed(z, "call");
    ___tracy_emit_zone_end(z);
}
#endif
