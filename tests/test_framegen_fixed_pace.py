"""A fixed frame generation multiplier's frames at their times on the game's own timeline, whatever Frame pacing says and
whatever the V-Sync row; under FIFO one a refresh at most.

With Frame pacing Off (its default) the display thread presented a set's frames as they came: within a millisecond of
each other at V-Sync Off, so the screen showed the game's frame and hardly the made ones, and DLSS Frame Gen 3x looked like
the game's own frame rate. The real fixed_dues, pace_lead, pace_delay and fixed_on_timeline of gpu/vk/vk_framegen.cpp:

- each shown frame of a 3x set is shown as far after the one before as its picture is after the one before's, across
  sets too, with the game's frames 15.5 to 22 ms apart (as at ~55 fps);
- the display loop, modelled: sets made at the game's times, their frames done 12 ms after their game frame (DLAA at
  5120x2160), shown one after the other at their times, each set's need fed back when it is shown. The GPU's delay settles
  without running away, every frame is then shown on time and 4 to 9 ms after the one before (never two together), a
  long game frame (60 ms) does not raise it, and it comes back down once the GPU has room;
- at 4x with an NVIDIA surface's 6 spare images (two sets need 8): the next set's copies as images free, its first at
  once, and only sets whose first copy went as they came fed back: the delay settles and every frame is on time. As
  before (a set's copies only once all fit, every set fed back) it ran to a whole game frame;
- the delay never passes a game frame, and a hitch leaves it;
- every set with its times is shown at them, FIFO too, and RX_VK_FG_TIMELINE=0 turns it off;
- under FIFO, modelled with a VRR display that shows each present when it comes, one a refresh at most, the game's
  frames held by the three sets, the display thread waiting while two presents wait to be shown and waking a little
  late (fifo_slot): 3x at ~60 fps on 240 Hz shows every frame when it is presented, as far after the one before as its
  picture (with the set's frames together, as before, they were 4.2, 4.2 and 8.3 ms apart); a set whose frames came late
  goes out a refresh apart, never two presents waiting; game frames 10 and 20 ms apart in turn at 3x on 165 Hz (faster
  than the display in the short ones) are shown each when it is presented, FIFO's catch-up leaving out the made frames
  the refresh has no room for (fifo_catch_up, spread over the set by fifo_pick), so the game keeps its own 66.7 fps
  and each game frame is shown within two refreshes of its time; 4x at 60 fps on a 239.5 Hz display taken as 240 Hz is
  shown evenly at the display's rate; 3x, 4x and 6x at 33 to 60 game frames a second on 165 Hz: past the refresh 160 to
  165 shown a second, each game frame within a refresh of its time, none waiting, under it none left out.

- present_image's frame slots: 3, and a slot per frame of two sets where the swapchain is made for two (RX_VK_FG_TIMELINE
  on, any present mode) while frame generation runs: a set's copies all at once, none waiting in its own first copy's
  slot or in the slot of the shown set's copy; RX_VK_FG_TIMELINE=0 keeps 3.

The text checks: every fixed set with a frame before (blend) and a made frame gets its times, whatever Frame pacing says;
display_main shows it at them whatever the swapchain's mode, under FIFO no sooner than fifo_slot allows (the display's
refresh, the mode read at each present), puts only a set presented ahead on the timeline, takes the next one early
(looked for every 2 ms of a wait) with its copies for the images free, the rest as presents give images back, and feeds
back only a set whose first copy it waited for, that went as it came and that did not follow a long game frame.
vk_main.cpp asks for two sets' images, under FIFO too, one set's with RX_VK_FG_TIMELINE=0.
python -I -B tests/test_framegen_fixed_pace.py [--cxx g++]
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

PREFIX = r'''#include <algorithm>
#include <initializer_list>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)
'''

MAIN = r'''
/* The display loop as display_main runs it for sets on the timeline, presented ahead and taken early: set i's game frame
   comes at now; its copy k goes to the GPU once the swapchain has an image for it beside those the set before still holds
   (its frames not yet presented: o.spare images in all), at once where it has, and is done gpu ns after it went (behind
   the game's work submitted meanwhile). o.whole: a set's copies only once all of them fit, as before. The display thread
   shows the sets in order, each frame at its time or once its copy is done, and feeds a set's need back once its first
   copy is done, unless it followed a long game frame or its first copy did not go as it came (o.feed_all: every set's, as
   before). r.extra: the GPU's delay at the end; r.shown and r.due: every present's time and its due time (the first set
   left out) */
struct Opt { int spare = 64; bool whole = false, feed_all = false; };
struct Run { int64_t extra; int n; int64_t shown[4096]; int64_t due[4096]; };
static void run(const int64_t* gaps, int sets, int64_t gpu, int made, Run& r, int64_t start_extra = 0, Opt o = {}) {
    int64_t last = 0, game_ns = 0, extra = start_extra, display = 0;
    int64_t before[8] = {};   /* when the set before's frames were shown */
    int before_n = 0;
    r.n = 0;
    for (int i = 0; i < sets; i++) {
        const int64_t now = last + gaps[i];
        if (last && gaps[i] <= 100000000) game_ns = game_ns ? (game_ns * 7 + gaps[i]) / 8 : gaps[i];   /* framegen_record */
        int64_t due[8];
        const int64_t set_extra = extra;
        fixed_dues(last, now, made, pace_lead(game_ns, made) + set_extra, due);
        int64_t went[8];
        for (int k = 0; k <= made; k++) {
            const int images = o.whole ? made + 1 : k + 1;           /* this set's, with copy k */
            const int presented = before_n + images - o.spare;        /* of the set before's, first */
            went[k] = presented > 0 ? std::max(now, before[presented - 1]) : now;
        }
        const int64_t copied = went[0] + gpu;                         /* its first copy done */
        const bool as_came = went[0] - now <= 3000000;
        if (i > 0 && gaps[i] * 2 <= game_ns * 3 && (as_came || o.feed_all))
            extra = pace_delay(extra, set_extra + (copied - due[0]) + 1000000, game_ns);
        int64_t t = display;                                          /* shown once the set before was */
        for (int k = 0; k <= made; k++) {
            t = std::max({t, due[k], went[k] + gpu});
            before[k] = t;
            if (i > 0 && r.n < 4096) r.due[r.n] = due[k], r.shown[r.n++] = t;
        }
        before_n = made + 1;
        display = t;
        last = now;
    }
    r.extra = extra;
}
/* frames shown more than 1 ms late, and closer than `close` to the one before, in the second half */
static void count(const Run& r, int64_t close, int& late, int& together) {
    late = together = 0;
    for (int i = r.n / 2; i < r.n; i++) {
        late += r.shown[i] > r.due[i] + 1000000;
        together += r.shown[i] - r.shown[i - 1] < close;
    }
}

/* Under FIFO, on a VRR display: the display loop above, the copies taken early (two sets' images), each frame given its
   time no sooner than fifo_slot allows (refresh: the display's refresh as Windows gives it) and presented wake ns after
   it; the screen showing each present at its present or a refresh of its own (screen) after the one shown before,
   whichever is later. The game's frame i comes gaps[i] after the one before, but not before set i - 3 was shown (three
   sets); while two presents wait to be shown the display thread waits for the screen (FIFO's images). gpu[i]: set i's
   frames done that long after its game frame; a set whose first copy was done before the thread came to it (the
   display held the thread) feeds back its copy's end (ready_at, seen while the set before was shown). A present late
   past its slot by over
   1 ms (the thread held) is the next one's slot. r.present, r.flip, r.content: each frame's present, when the screen
   showed it, where its picture is on the game's timeline; r.game: the game's frames; r.waiting: the most presents waiting
   to be shown at a present; r.extra_max: the GPU's delay at its most over the second half. FIFO's catch-up as take has it
   (fifo_catch_up, the set taken as it comes): a set whose game frame would be shown over a refresh late leaves made
   frames out, spread over it (fifo_pick); r.caught: the frames so left out, r.due: each shown frame's own time,
   r.is_game: whether it is a set's game frame */
struct Fifo {
    int n, waiting, games, caught;
    int64_t extra_max;
    int64_t present[8192], flip[8192], content[8192], due[8192], game[2048];
    bool is_game[8192];
};
static void run_fifo(const int64_t* gaps, int sets, const int64_t* gpu, int made, int64_t refresh, int64_t screen,
                     int64_t wake, Fifo& r, bool feed = true) {
    int64_t last = 0, game_ns = 0, extra = 0, slot = 0, t = 0, done[3] = {0, 0, 0};
    r.n = 0, r.waiting = 0, r.games = 0, r.caught = 0, r.extra_max = 0;
    const auto waiting_at = [&](int64_t at) {
        int w = 0;
        for (int q = r.n - 1; q >= 0 && r.flip[q] > at; q--) w++;
        return w;
    };
    for (int i = 0; i < sets; i++) {
        const int64_t now = std::max(last + gaps[i], done[i % 3]);
        const int64_t span = now - last;
        if (last && span <= 100000000) game_ns = game_ns ? (game_ns * 7 + span) / 8 : span;
        if (r.games < 2048) r.game[r.games++] = now;
        int64_t due[8];
        const int64_t set_extra = extra;
        fixed_dues(last, now, made, pace_lead(game_ns, made) + set_extra, due);
        const int first = i > 0 ? fifo_catch_up(made, due, slot, 0, refresh, now) : 0;
        r.caught += first;
        const bool held = t >= now + gpu[i];                               /* its copy done before the thread came */
        t = std::max(t, now + gpu[i]);                                     /* its first copy waited for */
        const int64_t at = held ? now + gpu[i] : t;
        if (feed && i > 0 && span * 2 <= game_ns * 3)
            extra = pace_delay(extra, set_extra + (at - due[0]) + 1000000, game_ns);
        if (i >= sets / 2) r.extra_max = std::max(r.extra_max, extra);
        const int shown = made + 1 - first;
        for (int q = 0; q < shown && i > 0; q++) {
            const int k = first ? fifo_pick(made + 1, shown, q) : q;      /* the set's frame, as display_main takes it */
            int64_t d = std::max(fifo_slot(due[k], slot, refresh, t), t);
            slot = d;
            t = d + wake;
            for (int w = waiting_at(t); w >= 2; w = waiting_at(t)) t = r.flip[r.n - w];   /* FIFO's images */
            if (t > slot + 1000000) slot = t;
            r.waiting = std::max(r.waiting, waiting_at(t));
            const int64_t flip = r.n ? std::max(t, r.flip[r.n - 1] + screen) : t;
            if (r.n < 8192)
                r.is_game[r.n] = k == made, r.present[r.n] = t, r.flip[r.n] = flip, r.due[r.n] = due[k],
                r.content[r.n++] = last + span * (k + 1) / (made + 1);
        }
        done[i % 3] = t;
        last = now;
    }
}

/* the second half's numbers of a FIFO run: the game's rate, the frames shown a second, the least time between two flips,
   the most a present waited to be shown, and the game frames' time on the screen past their own (most and mean) */
struct Report { double fps, shown_fps; int64_t min_gap, max_wait, lag_max, lag_mean; int caught_per_100; };
static Report report(const Fifo& f) {
    Report o{};
    o.fps = (f.games / 2 - 1) * 1e9 / (double)(f.game[f.games - 1] - f.game[f.games / 2]);
    o.shown_fps = (f.n - f.n / 2 - 1) * 1e9 / (double)(f.flip[f.n - 1] - f.flip[f.n / 2]);
    o.min_gap = INT64_MAX;
    int64_t sum = 0;
    int games = 0;
    for (int i = f.n / 2; i < f.n; i++) {
        o.min_gap = std::min(o.min_gap, f.flip[i] - f.flip[i - 1]);
        o.max_wait = std::max(o.max_wait, f.flip[i] - f.present[i]);
        if (f.is_game[i]) o.lag_max = std::max(o.lag_max, f.flip[i] - f.due[i]), sum += f.flip[i] - f.due[i], games++;
    }
    o.lag_mean = games ? sum / games : 0;
    o.caught_per_100 = f.games ? f.caught * 100 / f.games : 0;
    if (std::getenv("FIFO_REPORT"))
        std::fprintf(stderr, "fifo: game %.1f fps, shown %.1f a second, flips >= %.2f ms apart, a present waited <= %.2f ms, "
                     "game frames late <= %.2f ms (mean %.2f), %d frames left out per 100 sets, waiting %d, delay <= %.2f ms\n", o.fps,
                     o.shown_fps, o.min_gap / 1e6, o.max_wait / 1e6, o.lag_max / 1e6, o.lag_mean / 1e6, o.caught_per_100,
                     f.waiting, f.extra_max / 1e6);
    return o;
}

int main() {
    const int64_t ms = 1000000;
    /* the game's frames 15.5 to 22 ms apart, 3x: two made frames and the game's a set */
    const int64_t gaps[] = {16 * ms, 22 * ms, 15500000, 19 * ms, 21 * ms, 17 * ms, 22 * ms, 16 * ms};
    int64_t last = 1000 * ms, prev_due = 0, prev_content = 0;
    const int64_t delay = 13 * ms;
    for (int64_t g : gaps) {
        const int64_t now = last + g;
        int64_t due[8];
        fixed_dues(last, now, 2, delay, due);
        CHECK(due[2] == now + delay);                                  /* the game's frame one delay after it came */
        for (int k = 0; k < 3; k++) {
            const int64_t content = last + g * (k + 1) / 3;            /* where frame k's picture is between the two */
            CHECK(due[k] - content == delay);                          /* shown a fixed time after its picture */
            if (prev_due) {
                const int64_t shown = due[k] - prev_due, moved = content - prev_content;
                CHECK(shown == moved);                                 /* each shown frame moves as far as its time */
                CHECK(shown >= 5 * ms && shown <= 8 * ms);             /* a third of 15.5 to 22 ms: never together */
            }
            prev_due = due[k], prev_content = content;
        }
        last = now;
    }
    int64_t due[8];
    fixed_dues(0, 30 * ms, 1, 0, due);                                 /* 2x: the midpoint, then the game's frame */
    CHECK(due[0] == 15 * ms && due[1] == 30 * ms);
    fixed_dues(0, 40 * ms, 3, 5 * ms, due);                            /* 4x: quarters */
    CHECK(due[0] == 15 * ms && due[1] == 25 * ms && due[2] == 35 * ms && due[3] == 45 * ms);

    /* the lead: at the game's own frame time the set's first frame is due when its game frame came, no earlier */
    CHECK(pace_lead(18 * ms, 2) == 12 * ms && pace_lead(18 * ms, 1) == 9 * ms && pace_lead(20 * ms, 3) == 15 * ms);
    CHECK(pace_lead(18 * ms, 0) == 0);
    for (int made = 1; made <= 3; made++) {
        const int64_t frame = 18 * ms, now = 500 * ms;
        fixed_dues(now - frame, now, made, pace_lead(frame, made), due);
        CHECK(due[0] == now);
        int64_t next[8];                                                /* the next set, a game frame later */
        fixed_dues(now, now + frame, made, pace_lead(frame, made), next);
        CHECK(next[0] - due[made] == frame / (made + 1));               /* its first frame one step after this game frame */
    }

    /* the display loop: 600 sets at 15.5 to 22 ms (~55 fps), 3x, frames done 12 ms after their game frame */
    static int64_t g600[600];
    for (int i = 0; i < 600; i++) g600[i] = gaps[i % 8];
    static Run r;
    run(g600, 600, 12 * ms, 2, r);
    CHECK(r.extra >= 13 * ms && r.extra <= 16 * ms);   /* settled: the GPU's 12 ms, 1 to spare, the frames' unevenness */
    int late = 0, together = 0, apart = 0;
    for (int i = 300; i < r.n; i++) {                                    /* the second half, settled */
        late += r.shown[i] > r.due[i] + 1000000;
        const int64_t d = r.shown[i] - r.shown[i - 1];
        together += d < 4 * ms;
        apart += d > 9 * ms;
    }
    CHECK(late == 0);                                                    /* every frame on time */
    CHECK(together == 0 && apart == 0);                                  /* 4 to 9 ms apart: never two together */
    const int64_t settled = r.extra;
    run(g600, 600, 30 * ms, 2, r);                                       /* a GPU slower than a game frame */
    CHECK(r.extra <= 22 * ms);                                           /* held to a game frame (the smoothed ~18.5 ms) */
    /* a long game frame (60 ms) among them: the sets after it do not raise the delay */
    static int64_t hitch[600];
    for (int i = 0; i < 600; i++) hitch[i] = i == 400 ? 60 * ms : g600[i];
    run(hitch, 600, 12 * ms, 2, r);
    CHECK(r.extra <= settled + ms);
    /* the GPU with room again: the delay comes back down */
    run(g600, 600, 2 * ms, 2, r, settled);
    CHECK(r.extra <= 5 * ms);                                            /* 2 ms, 1 to spare, the unevenness */

    /* 4x on an NVIDIA surface: 6 spare images, two sets need 8. As before (a set's copies once all fit, every set fed back)
       the next set waited for the images of the one before once the delay passed 3/4 of a game frame, its copies went later,
       the delay rose with them: up to a whole game frame. Now its first copies go at once and a set that waited is left out */
    run(g600, 600, 12 * ms, 3, r);                                       /* 4x with images to spare */
    const int64_t free4 = r.extra;
    CHECK(free4 >= 13 * ms && free4 <= 16 * ms);
    run(g600, 600, 12 * ms, 3, r, 0, Opt{6, true, true});                /* as before */
    CHECK(r.extra >= 17 * ms);                                           /* nearly a game frame: it fed itself */
    const Opt nvidia{6, false, false};
    for (int64_t gpu : {12 * ms, 14 * ms}) {
        run(g600, 600, gpu, 3, r, 0, nvidia);
        int late4 = 0, together4 = 0;
        count(r, 3 * ms, late4, together4);                              /* a quarter of 15.5 to 22 ms: 3.9 to 5.5 */
        CHECK(late4 == 0 && together4 == 0);
        if (gpu == 12 * ms) CHECK(r.extra <= free4 + ms);                /* as if the images were there */
        CHECK(r.extra <= gpu + 3 * ms);                                  /* the GPU's, 1 to spare, the unevenness */
    }

    /* present_image's frame slots for a swapchain's sets: 3; where it is made for two sets of a fixed multiplier, a slot per
       frame of both (no copy waits in the slot of its own set's first copy or of the shown set's copy), at most
       kPresentSlotsMax; one set (FIFO, RX_VK_FG_TIMELINE=0): 3, as before */
    CHECK(present_slots_for(0, true) == 3 && present_slots_for(1, true) == 3);    /* none; Auto */
    CHECK(present_slots_for(2, true) == 4 && present_slots_for(3, true) == 6 && present_slots_for(4, true) == 8);
    CHECK(present_slots_for(2, false) == 3 && present_slots_for(3, false) == 3 && present_slots_for(4, false) == 3);
    CHECK(present_slots_for(99, true) == kPresentSlotsMax);
    for (int m = 2; m <= 4; m++) CHECK(present_slots_for(m, true) >= 2 * m);    /* both sets' frames, up to 4x (8 images) */

    /* the delay's limits: never past a game frame, a hitch (a need past four frames) leaves it, no frame time yet leaves it */
    const int64_t frame = 18 * ms;
    int64_t big = 0;
    for (int i = 0; i < 40; i++) big = pace_delay(big, 3 * frame, frame);
    CHECK(big == frame);
    CHECK(pace_delay(5 * ms, 100 * ms, frame) == 5 * ms);
    CHECK(pace_delay(5 * ms, 9 * ms, 0) == 5 * ms);
    CHECK(pace_delay(0, -10 * ms, frame) == 0);                          /* early with none: none */
    CHECK(pace_delay(8 * ms, 12 * ms, frame) == 10 * ms);                /* up by half of what is missing */
    CHECK(pace_delay(8 * ms, 0, frame) == 7500000);                      /* down by a sixteenth of what is over */

    /* where a set is shown at its times: with them, whatever the swapchain, unless RX_VK_FG_TIMELINE=0 */
    CHECK(fixed_on_timeline(true, false));
    CHECK(!fixed_on_timeline(true, true));                               /* RX_VK_FG_TIMELINE=0 */
    CHECK(!fixed_on_timeline(false, false));                             /* no times: the first set, a hitch, no made frame */

    /* the slots under FIFO: none before the first frame or without the refresh, a due time past the next slot kept, a
       sooner one held to a refresh after the slot before, a slot left far behind started again at now */
    const int64_t hz240 = 4166666, hz165 = 6060606;                      /* as display_refresh_mhz gives them */
    CHECK(fifo_slot(5 * ms, 0, hz240, 0) == 5 * ms);
    CHECK(fifo_slot(5 * ms, 4 * ms, 0, 0) == 5 * ms);
    CHECK(fifo_slot(20 * ms, 10 * ms, hz240, 10 * ms) == 20 * ms);
    CHECK(fifo_slot(10 * ms, 10 * ms, hz240, 10 * ms) == 10 * ms + hz240);
    CHECK(fifo_slot(10 * ms, 10 * ms, hz240, 12 * ms) == 10 * ms + hz240);       /* woken late: from the slot, not now */
    CHECK(fifo_slot(10 * ms, 10 * ms, hz240, 30 * ms) == 30 * ms);               /* held up: again from now */
    CHECK(fifo_slot(40 * ms, 10 * ms, hz240, 30 * ms) == 40 * ms);

    /* FIFO on a 240 Hz VRR display, 3x at ~60 fps (180 shown a second), frames done 5 ms after their game frame, the
       thread 0.3 ms late: each frame shown when it is presented (none waits) and as far after the one before as its
       picture, 4.2 ms at least */
    static int64_t g60[600], gpu5[600];
    const int64_t jit[] = {16 * ms, 17300000, 16500000, 16900000, 16200000, 17100000};
    for (int i = 0; i < 600; i++) g60[i] = jit[i % 6], gpu5[i] = 5 * ms;
    static Fifo f;
    run_fifo(g60, 600, gpu5, 2, hz240, hz240, 300000, f);
    report(f);
    CHECK(f.waiting == 0);
    for (int i = f.n / 2; i < f.n; i++) {
        CHECK(f.flip[i] == f.present[i]);
        const int64_t shown = f.flip[i] - f.flip[i - 1], moved = f.content[i] - f.content[i - 1];
        CHECK(shown >= hz240 && shown <= 6 * ms);
        CHECK(shown - moved <= 100000 && moved - shown <= 100000);       /* within 0.1 ms of its picture's time */
    }
    /* one set's frames done 35 ms after its game frame (a long GPU frame): the frames due together go out a refresh apart,
       none waiting to be shown, and the sets after it on time again */
    static int64_t gpu_late[600];
    for (int i = 0; i < 600; i++) gpu_late[i] = i == 300 ? 35 * ms : 5 * ms;
    run_fifo(g60, 600, gpu_late, 2, hz240, hz240, 300000, f);
    report(f);
    CHECK(f.waiting == 0);
    for (int i = 1; i < f.n; i++) CHECK(f.present[i] - f.present[i - 1] >= hz240);
    for (int i = f.n - 300; i < f.n; i++) CHECK(f.flip[i] == f.present[i]);
    /* 3x on a 165 Hz VRR display, the game's frames 10 and 20 ms apart in turn (66.7 a second, 200 frames asked of a 165
       Hz display; three in 3.3 ms in the short ones): the slots hold them a refresh apart, so each is shown when it is
       presented and none waits; FIFO's catch-up leaves out the made frames the refresh has no room for, so the game keeps
       its own rate (without it the three sets held it at the display's 55) and each game frame is shown within two
       refreshes of its own time (8.5 ms at most fed back, 9.8 not) */
    static int64_t g1020[600];
    for (int i = 0; i < 600; i++) g1020[i] = i % 2 ? 20 * ms : 10 * ms;
    /* each run twice: every set fed back, and none (display_main leaves a set out when its first copy waited, as_came:
       at the refresh limit under FIFO it waits for FIFO's images, so every set is left out and the delay keeps its
       value; the slots, later than any due time there, show the frames the same) */
    for (const bool feed : {true, false}) {
        run_fifo(g1020, 600, gpu5, 2, hz165, hz165, 300000, f, feed);
        const Report o = report(f);
        CHECK(f.waiting == 0);
        for (int i = f.n / 2; i < f.n; i++) CHECK(f.flip[i] == f.present[i] && f.flip[i] - f.flip[i - 1] >= hz165);
        CHECK(o.fps > 66.0 && o.fps < 67.5);
        CHECK(o.shown_fps > (feed ? 160.0 : 145.0) && o.shown_fps <= 165.1);
        CHECK(o.lag_max <= 2 * hz165);
        /* and fed back, the GPU's delay stays under 10 ms: the copies' own 5 ms and 1 to spare, and up to 2.75 ms more
           for the long game frames' sets, as at V-Sync Off: with the game no longer held to 55 a second the 10 and 20 ms
           gaps reach the pacing (8.75 ms at most; it rose to a whole game frame, 18.3 ms, when the time the thread came to
           a set was fed back) */
        CHECK(f.extra_max <= 10 * ms);
    }
    /* 4x at 60 fps on a 239.5 Hz display taken as 240: the slots come a little faster than the screen, FIFO's images
       hold the thread back, and the screen shows them evenly at its own rate */
    static int64_t g16[600];
    for (int i = 0; i < 600; i++) g16[i] = 16666667;
    const int64_t hz2395 = (int64_t)(1e9 / 239.5);
    run_fifo(g16, 600, gpu5, 3, hz240, hz2395, 300000, f);
    report(f);
    CHECK(f.waiting <= 2);
    for (int i = f.n / 2; i < f.n; i++) CHECK(f.flip[i] - f.flip[i - 1] == hz2395);
    /* FIFO's catch-up on a 165 Hz VRR display, the game's frames 16.7 to 30 ms apart (0.4 ms either way), at 3x, 4x and
       6x: where the frames asked pass the refresh, the made frames it has no room for are left out, spread over each set
       (fifo_pick), so the screen shows 160 to 165 frames a second and each game frame within a refresh (and the thread's
       0.3 ms) of its own time, none waiting, the game at its own rate; under the refresh none is left out. 6x at 43 a
       second: 163 shown a second, 2.2 of 5 made frames a set left out (without the catch-up the game was held to 27.5 a
       second and its frames shown 138 ms late; left out at the sets' front, 129 shown a second, moving unevenly) */
    static int64_t gx[600];
    for (const int64_t gap : {16666667LL, 18000000LL, 23255814LL, 30000000LL})
        for (const int m : {2, 3, 5}) {
            for (int i = 0; i < 600; i++) gx[i] = gap + (i % 3 - 1) * 400000;
            run_fifo(gx, 600, gpu5, m, hz165, hz165, 300000, f);
            const Report o = report(f);
            const double asked = 1e9 / gap * (m + 1);
            CHECK(f.waiting == 0 && o.min_gap >= hz165);
            CHECK(o.fps > 1e9 / gap - 0.5);                                   /* the game at its own rate */
            CHECK(o.shown_fps >= std::min(asked, 165.0) - 5.0 && o.shown_fps <= 165.1);
            CHECK(o.lag_max <= hz165 + 500000);
            CHECK((asked < 165.0) == (f.caught == 0));                        /* frames left out only past the refresh */
        }
    std::printf("PASS: a fixed multiplier's frames go out at their own times, under FIFO a refresh apart at least, and wait "
                "only as the GPU asks\n");
}
'''


def check_sources(source, main_cpp):
    record = block(source, "bool framegen_record(", where="gpu/vk/vk_framegen.cpp")
    display = block(source, "void display_main()", where="gpu/vk/vk_framegen.cpp")
    assert re.search(r"set->shown = -1;\s*set->by_time = false;", record), \
        "framegen_record: a reused set starts without times"
    m = re.search(r"if \(blend && set->made > 0 && fg\.game_ns > 0\) \{(.*?)\n    \}", record, re.S)
    assert m, "framegen_record: every fixed set with a frame before and a made frame gets its times"
    body = m.group(1)
    for part in ("fixed_dues(before, now, set->made, pace_lead(fg.game_ns, set->made) + extra, set->due);",
                 "set->span_ns = now - before;", "set->extra_ns = extra;", "set->by_time = true;",
                 "set->game[k] = k == set->made;"):
        assert part in body, "framegen_record's times: " + part
    assert "set->shown" not in body, "framegen_record: a fixed set keeps shown < 0; display_main decides on its times"
    assert "opts.pace" not in body, "framegen_record: the times do not depend on Frame pacing"
    assert record.index("if (auto_rate)") < record.index("fixed_dues("), "framegen_record: Auto keeps its own slots"
    assert record.index("note_maker(maker);") < record.index("fixed_dues("), \
        "framegen_record: the set's made frames are known before its times"

    assert 'getenv("RX_VK_FG_TIMELINE")' in display, "display_main: RX_VK_FG_TIMELINE=0 turns the times off"
    assert "set->shown < 0 && fixed_on_timeline(set->by_time, timeline_off)" in display, \
        "display_main: a set with its times is shown at them, whatever Frame pacing and the swapchain's mode"
    assert "const int mhz = s.line ? display_refresh_mhz() : 0;" in display and \
        "s.refresh = mhz > 0 ? (int64_t)(1e12 / mhz) : 0;" in display, \
        "display_main: a set on the timeline takes the display's refresh for fifo_slot"
    slot = block(display, "if (s.refresh && dev.swap_mode == VK_PRESENT_MODE_FIFO_KHR) {",
                 where="gpu/vk/vk_framegen.cpp display_main")
    assert "fifo_slot(due, slot, s.refresh, now)" in slot, "display_main: under FIFO, fifo_slot after the slot before"
    assert "if (t > slot + 1000000) slot = t;" in display and "slot = slot_was;" in display, \
        "display_main: a late present is the next one's slot; a present not shown gives its slot back"
    at = display.index("if (s.refresh && dev.swap_mode == VK_PRESENT_MODE_FIFO_KHR) {")
    assert display.index("int64_t due = timed ? j.set->due[f]") < at < display.index("if (due < now) due = now;") < \
        display.index("slot = due;") < display.index("take_early(s.shown - k, due);"), \
        "display_main: the slot on the frame's own due time, before a late one goes out at once; the slot it was given kept"
    assert "s.timed = s.auto_timed || s.line;" in display, "display_main: such a set is shown at its times"
    assert re.search(r"int64_t due = timed \? j\.set->due\[f\]", display), "display_main: each frame at its own due time"
    take = block(display, "const auto take = [&](Taken& s, const Job& j, int room, int left)", where="gpu/vk/vk_framegen.cpp")
    # (FIFO's catch-up: a set whose game frame would be shown over a refresh after its own time leaves made frames out,
    # spread over it, and their copies; the frames shown are the ones fifo_pick picks)
    assert "(s.first = fifo_catch_up(j.set->made, j.set->due, slot, left, fifo_refresh, now_ns())))" in take and \
        "s.shown -= s.first, stat.caught += s.first;" in take and "dev.swap_mode == VK_PRESENT_MODE_FIFO_KHR" in take and \
        "present_ahead_wanted(true, j.set->made + 1, dev.swap_spare.load(std::memory_order_relaxed)) &&" in take and \
        "fifo_refresh = s.line ? s.refresh : 0;" in take and \
        "if (!s.auto_timed && dev.swap_mode == VK_PRESENT_MODE_FIFO_KHR && on_line(j.set) &&" in take, \
        "take: under FIFO a set whose game frame would be over a refresh late leaves made frames out (fifo_catch_up)"
    assert take.index("s.shown -= s.first") < take.index("s.ahead = present_ahead_wanted("), \
        "take: the frame left out before the set's copies"
    assert display.count("const int f = s.first ? fifo_pick(j.set->made + 1, s.shown, k) : k;") == 2 and \
        "j.set->frame[k]" not in display and "j.set->game[k]" not in display, \
        "display_main: the copies and the presents take the set's frame the catch-up picked (fifo_pick)"
    assert "take(taken[1], j, spare - left, left);" in display, "take_early: the set before's frames still to show"
    order = [take.index(t) for t in ("s.ahead = present_ahead_wanted(pace || line, s.shown,",
                                     "s.line = line && s.ahead;", "s.timed = s.auto_timed || s.line;",
                                     "copy_up_to(s, room);")]
    assert order == sorted(order), \
        "take: a set on the timeline is presented ahead with Frame pacing Off too, and on it only presented ahead"
    assert "const bool line = on_line(j.set);" in take, "take: the swapchain's mode decides, not Frame pacing"
    copy = block(display, "const auto copy = [&](Taken& s, int k)", where="gpu/vk/vk_framegen.cpp")
    assert "if (k == 0) s.as_came = t0 - j.queued <= 3000000 && now_ns() - t0 <= 3000000;" in copy and \
        "const int64_t t0 = now_ns();" in copy and "as_came" not in take, \
        "copy: the set taken within 3 ms of its coming, its first copy not held over 3 ms in present_image"
    assert "fg.jobs.push_back({fg.recorded, fg.recorded_at, value, now_ns()});" in source, \
        "framegen_queue: the time a set is handed to the display thread"
    m = re.search(r"if \(s\.line\) \{(.*?)\n        \}\n", display, re.S)
    assert m, "display_main: the feedback of a set on the timeline"
    feed = m.group(1)
    assert re.search(r"if \(first && s\.as_came\) \{.*pace_delay\(", feed, re.S) and \
        "const bool first = s.ahead && s.pending[0].used;" in display, \
        "display_main: only a set whose first copy was waited for and went as it came feeds the GPU's delay back"
    assert "const int64_t at = s.ready_at ? s.ready_at : now_ns();" in feed, \
        "display_main: the copy's end as seen while the set before was shown, else now"
    assert "j.set->span_ns * 2 <= j.set->frame_ns * 3" in feed, \
        "display_main: a set after a long game frame does not feed it back"
    # (against the set's first frame's time also when the catch-up leaves frames out: a copy takes as long whichever frame
    # it is, and the delay is the set's; against the first frame shown, 6x on 165 Hz ran the delay down to 0, the frames
    # came due before their copies and the catch-up left out a frame more a set than the refresh needed)
    assert "j.set->extra_ns + (at - j.set->due[0]) + 1000000" in feed, \
        "display_main: the need is the set's own delay and its lateness against its first frame, a millisecond to spare"
    loop = display[display.index("    for (;;) {"):]
    assert loop.index("copy_up_to(s, (int)dev.swap_spare.load(std::memory_order_relaxed));") < \
        loop.index("const bool first = ") < loop.index("vkWaitForFences") < loop.index("pace_delay("), \
        "display_main: a set taken early has its last copies sent, then the lateness once its first copy is done"
    present_loop = loop.index("for (int k = 0; k < s.shown; k++) {")
    assert loop.index("pace_delay(") < present_loop, "display_main: the lateness is taken before the set's first present"
    assert present_loop < loop.index("take_early(s.shown - k, due);") < loop.index("wait_until_near("), \
        "display_main: the next set is taken while this one waits for its frames' times"
    assert re.search(r"for \(;;\) \{\s*take_early\(s\.shown - k, due\);.*?std::min<int64_t>\(due - now - 1000000, 2000000\)",
                     loop, re.S), \
        "display_main: the wait looks for the next set every 2 ms at most (at 2x it comes during the game frame's wait)"
    early = block(display, "const auto take_early = [&](int left, int64_t until)", where="gpu/vk/vk_framegen.cpp")
    assert "shown > spare || spare - left < 1" in early and "take(taken[1], j, spare - left, left);" in early, \
        "take_early: a set the swapchain can hold, its copies for the images this set does not hold"
    assert "!on_line(j.set)" in early and "!taken[0].line" in early, \
        "take_early: only a set on the timeline after one on it (presented ahead)"
    assert "taken[1].ready_at = now_ns();" in loop, "display_main: a set taken early has its first copy's end seen"
    assert "copy_up_to(taken[1], (int)dev.swap_spare.load(std::memory_order_relaxed) - (s.shown - k));" in loop, \
        "display_main: the next set's other copies as this set's presents give images back"
    assert "(t - (s.line ? own : due) > 2000000 ? stat.late : stat.on_time)++;" in loop, \
        "display_main: a frame on the timeline is late against its own time"

    chain = block(main_cpp, "bool create_swapchain(", where="gpu/vk/vk_main.cpp")
    assert "const uint32_t fg_sets = timeline && !set_too_big && !fifo_unpaced ? 2 : 1;" in chain and \
        "const bool fifo_unpaced = mode == VK_PRESENT_MODE_FIFO_KHR && display_refresh_mhz() <= 0;" in chain and \
        "const bool set_too_big = mode == VK_PRESENT_MODE_FIFO_KHR && caps.maxImageCount &&" in chain and \
        'getenv("RX_VK_FG_TIMELINE")' in chain and \
        "caps.minImageCount + std::min<uint32_t>(fg_sets * (uint32_t)swapchain_framegen()," in chain and \
        "(uint32_t)kPresentSlotsMax));" in chain, \
        "create_swapchain: two sets' images while the sets are on the timeline, under FIFO too (take_early); one set's " \
        "with RX_VK_FG_TIMELINE=0"
    assert "const uint32_t slots_made = (uint32_t)present_slots_for(swapchain_framegen(), fg_sets == 2);" in chain and \
        "dev.swap_slots = slots_made;" in chain and \
        chain.index("dev.swap_slots = slots_made;") > chain.index("retire_swapchain();"), \
        "create_swapchain: present_image's frame slots for the sets the chain is made for, stored after the old chain's " \
        "retire_swapchain (which sets 3)"


def check_slots(replay, internal, imgui):
    m = re.search(r"constexpr int kPresentSlotsMax = (\d+);", internal)
    assert m, "vk_internal.h: kPresentSlotsMax"
    assert "Frame frames[kPresentSlotsMax];" in replay and "LogSlot log_slots[kPresentSlotsMax];" in replay, \
        "vk_replay.cpp: the frame slots and their frame-log slots, kPresentSlotsMax of each"
    image = block(replay, "void present_image(VkImage src,", where="gpu/vk/vk_replay.cpp")
    for part in ("const uint32_t slots = framegen_on() ? std::clamp<uint32_t>(dev.swap_slots.load(std::memory_order_relaxed), 3u,",
                 "const uint32_t slot = frame_index % slots;", "Frame& f = frames[slot];",
                 "frame_index = (slot + 1) % slots;", "for (uint32_t k = slots; k < slots_was; k++)"):
        assert part in image, "present_image: " + part
    assert "std::atomic<uint32_t> swap_slots{3};" in internal, "vk_internal.h: the swapchain's frame slots, 3 to start"
    assert "ii.ImageCount = kPresentSlotsMax + 1;" in imgui, \
        "vk_imgui.cpp: ImGui's vertex buffers, one more than the presents in flight"
    return int(m.group(1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    vk = ROOT / "gpu" / "vk"
    source = (vk / "vk_framegen.cpp").read_text(encoding="utf-8")
    main_cpp = (vk / "vk_main.cpp").read_text(encoding="utf-8")
    replay = (vk / "vk_replay.cpp").read_text(encoding="utf-8")
    internal = (vk / "vk_internal.h").read_text(encoding="utf-8")
    imgui = (vk / "vk_imgui.cpp").read_text(encoding="utf-8")
    slots_max = check_slots(replay, internal, imgui)
    check_sources(source, main_cpp)
    fixture = (PREFIX + f"constexpr int kPresentSlotsMax = {slots_max};\n" +
               "\n".join(block(source, m, where="gpu/vk/vk_framegen.cpp") for m in (
                   "void fixed_dues(", "int64_t pace_lead(", "int64_t pace_delay(", "bool fixed_on_timeline(",
                   "int64_t fifo_slot(", "int fifo_pick(", "int fifo_catch_up(")) + "\n" +
               block(replay, "int present_slots_for(", where="gpu/vk/vk_replay.cpp") + MAIN)
    with tempfile.TemporaryDirectory(prefix="framegen-fixed-pace-") as tmp:
        cpp = Path(tmp) / "fixture.cpp"
        exe = Path(tmp) / "fixture"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
