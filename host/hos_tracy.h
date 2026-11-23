/* (hookshot) The Tracy profiler's zones, for the host (C) and the Vulkan back end (C++): docs/development/tracy.md.

   A build made with HOOKSHOT_TRACY=1 (nx2/build.py) defines TRACY_ENABLE: Tracy's client (third_party/tracy) is compiled
   into the game's executable alone, which exports its C API; the back end's module calls the same client through it, so
   the process has one. Every other build, a player's among them, gets the empty macros below: nothing of Tracy is
   compiled, and a call site costs nothing. Call sites use these macros only and never include Tracy themselves.

   The client listens on 127.0.0.1 only and announces nothing on the network (TRACY_ONLY_LOCALHOST, TRACY_NO_BROADCAST),
   and records only while a viewer or tracy-capture is connected (TRACY_ON_DEMAND). A capture holds the game's function
   names and timings: it is made from the game and never leaves the PC (CONTRIBUTING.md, the rule with no exceptions).

     HT_SCOPE("name");                     a zone to the end of the enclosing block
     HT_ZONE(z, "name");                   the same, named, for the macros below
     HT_ZONE_IF(z, cond, "name");          only when cond
     HT_WAIT(z, "name"); HT_WAIT_IF(...)   the same in the waits' colour: a thread that is blocked, not working
     HT_ZONE_END(z);                       ends it before its block does
     HT_ZONE_TEXTF(z, "lr %#llx", lr);     its text (formatted only while a capture is being taken)
     HT_ZONE_VALUE(z, n);
     HT_SCOPE_DRAW("name");                a zone made once per draw: off with RX_VK_TRACY_DRAWS=0
     HT_FRAME();  HT_FRAME_NAMED("name");  a frame's end
     HT_PLOT("name", value);
     HT_THREAD_NAME(name);                 the calling thread's name
   and, where HT_WRAP_LOCKS is defined before this is included (after <windows.h>): every AcquireSRWLockExclusive,
   AcquireSRWLockShared and EnterCriticalSection of the file that has to wait is a "lock wait" zone naming the lock. */
#ifndef HOS_TRACY_H
#define HOS_TRACY_H

#ifdef TRACY_ENABLE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tracy/TracyC.h>

#define HT_CAT2(a, b) a##b
#define HT_CAT(a, b) HT_CAT2(a, b)
#ifdef __cplusplus
#define HT_SRCLOC static constexpr
#else
#define HT_SRCLOC static const
#endif
#define HT_COLOR_WAIT 0x8B3A3Au
#define HT_COLOR_LOCK 0xD04040u
#define HT_COLOR_GAME 0x2E6B8Au   /* (the lifted game code's own zones: nx2/tracy_funcs.py) */

/* (the client is started by the game's main alone, hos_tracy_start in host/hos_tracy_game.c: the crash window's helper,
   which is the same executable, has none, so it takes no port and a capture never connects to it. Until then, and in the
   helper for good, a zone is off. The back end's module is loaded after the start) */
#ifdef TRACY_IMPORTS
#define HT_ON 1
#else
#ifdef __cplusplus
extern "C" {
#endif
extern int hos_tracy_started;
void hos_tracy_start(void);
#ifdef __cplusplus
}
#endif
#define HT_ON hos_tracy_started
#endif
static inline void ht_zone_end_(TracyCZoneCtx *z) { ___tracy_emit_zone_end(*z); }
#define HT_ZONE_AS(z, name, color, on) \
    HT_SRCLOC struct ___tracy_source_location_data HT_CAT(ht_sl_, __LINE__) = { name, __func__, __FILE__, (uint32_t)__LINE__, color }; \
    TracyCZoneCtx z __attribute__((cleanup(ht_zone_end_))) = ___tracy_emit_zone_begin(&HT_CAT(ht_sl_, __LINE__), HT_ON && (on) ? 1 : 0)
#define HT_ZONE(z, name) HT_ZONE_AS(z, name, 0, 1)
#define HT_ZONE_IF(z, cond, name) HT_ZONE_AS(z, name, 0, cond)
#define HT_WAIT(z, name) HT_ZONE_AS(z, name, HT_COLOR_WAIT, 1)
#define HT_WAIT_IF(z, cond, name) HT_ZONE_AS(z, name, HT_COLOR_WAIT, cond)
#define HT_SCOPE(name) HT_ZONE_AS(HT_CAT(ht_z_, __LINE__), name, 0, 1)
#define HT_ZONE_END(z) do { ___tracy_emit_zone_end(z); (z).active = 0; } while (0)
#define HT_ZONE_TEXTF(z, ...) do { \
        if ((z).active) { \
            char ht_b_[192]; \
            const int ht_n_ = snprintf(ht_b_, sizeof ht_b_, __VA_ARGS__); \
            if (ht_n_ > 0) ___tracy_emit_zone_text(z, ht_b_, ht_n_ < (int)sizeof ht_b_ ? (size_t)ht_n_ : sizeof ht_b_ - 1); \
        } \
    } while (0)
#define HT_ZONE_VALUE(z, v) do { if ((z).active) ___tracy_emit_zone_value(z, (uint64_t)(v)); } while (0)
#define HT_FRAME() do { if (HT_ON) ___tracy_emit_frame_mark(0); } while (0)
#define HT_FRAME_NAMED(name) do { if (HT_ON) ___tracy_emit_frame_mark(name); } while (0)
#define HT_PLOT(name, v) do { if (HT_ON) ___tracy_emit_plot(name, (double)(v)); } while (0)
#define HT_THREAD_NAME(name) do { if (HT_ON) ___tracy_set_thread_name(name); } while (0)
/* (the game's fibers, nn::os::SwitchToFiber: the zones after an enter are that fiber's, after a leave the thread's own) */
#ifdef TRACY_FIBERS   /* (a build with HOOKSHOT_TRACY_FIBERS=1 too: nx2/build.py) */
#define HT_FIBER_ENTER(name) do { if (HT_ON) ___tracy_fiber_enter(name); } while (0)
#define HT_FIBER_LEAVE() do { if (HT_ON) ___tracy_fiber_leave(); } while (0)
#else
#define HT_FIBER_ENTER(name) do {} while (0)
#define HT_FIBER_LEAVE() do {} while (0)
#endif
/* (the lifted game code's zones, nx2/tracy_funcs.py: off with HOS_TRACY_GAME=0, read at the start) */
#ifndef TRACY_IMPORTS
#ifdef __cplusplus
extern "C" int hos_tracy_game_on;
#else
extern int hos_tracy_game_on;
#endif
#endif

/* (the zones made once per draw, thousands a frame: on unless RX_VK_TRACY_DRAWS=0, read once per source file) */
static inline int ht_draws_on_(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RX_VK_TRACY_DRAWS");
        on = !(e && e[0] == '0');
    }
    return on;
}
#define HT_SCOPE_DRAW(name) HT_ZONE_AS(HT_CAT(ht_z_, __LINE__), name, 0, ht_draws_on_())

#ifdef HT_WRAP_LOCKS
/* (a lock taken at once makes no zone: the try costs what the lock does. The macros name themselves, which the
   preprocessor leaves alone inside their own expansion: the call in them is the real one) */
#define HT_LOCK_WAIT(l, tried, take) do { \
        if (!(tried)) { \
            HT_ZONE_AS(ht_lk_, "lock wait", HT_COLOR_LOCK, 1); \
            if (ht_lk_.active) ___tracy_emit_zone_text(ht_lk_, #l, sizeof(#l) - 1); \
            take; \
        } \
    } while (0)
#define AcquireSRWLockExclusive(l) HT_LOCK_WAIT(l, TryAcquireSRWLockExclusive(l), AcquireSRWLockExclusive(l))
#define AcquireSRWLockShared(l) HT_LOCK_WAIT(l, TryAcquireSRWLockShared(l), AcquireSRWLockShared(l))
#define EnterCriticalSection(l) HT_LOCK_WAIT(l, TryEnterCriticalSection(l), EnterCriticalSection(l))
#endif

#else   /* no Tracy: nothing */

#define HT_ZONE(z, name) do {} while (0)
#define HT_ZONE_IF(z, cond, name) do {} while (0)
#define HT_WAIT(z, name) do {} while (0)
#define HT_WAIT_IF(z, cond, name) do {} while (0)
#define HT_SCOPE(name) do {} while (0)
#define HT_SCOPE_DRAW(name) do {} while (0)
#define HT_ZONE_END(z) do {} while (0)
#define HT_ZONE_TEXTF(z, ...) do {} while (0)
#define HT_ZONE_VALUE(z, v) do {} while (0)
#define HT_FIBER_ENTER(name) do {} while (0)
#define HT_FIBER_LEAVE() do {} while (0)
#define HT_FRAME() do {} while (0)
#define HT_FRAME_NAMED(name) do {} while (0)
#define HT_PLOT(name, v) do {} while (0)
#define HT_THREAD_NAME(name) do {} while (0)

#endif
#endif
