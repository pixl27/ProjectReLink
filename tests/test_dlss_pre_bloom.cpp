#include "../gpu/vk/pre_bloom.h"
#include "../gpu/vk/upscaler_rows.h"

#include <cstdio>
#include <vector>

#define CHECK(name, condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %s at line %d\n", name, __LINE__); return 1; } } while (0)

namespace pb = mkwvk::pre_bloom;

// One pass of a frame, as vk_temporal.cpp temporal_pre_bloom sees it: a draw into the scene ('D') or one that reads it ('R',
// with its program)
struct Pass {
    char kind;
    uint64_t key;
};

// The Market's frame (DLAA at 5120x2160; RX_VK_DUMP of main 7887c497, steps 240 to 295): scene draws, the game's copies of
// the scene at a quarter and at half the size, more scene draws (the shadowed transparents, the particles), a second half-size
// copy, a quarter-size one, the last scene draws, then the bloom's first draw (to an eighth of the size), and the scene
// mapping draw (the scene with the bloom through the tone curve)
constexpr uint64_t kQuarterCopy = 0xb7914ae65792d0c8ull, kHalfCopy = 0xed4819446f84dcb4ull, kQuarter2 = 0x1b464cb9d32aaae5ull;
constexpr uint64_t kBloom = 0x95937683d9f74e95ull, kMapping = 0xab3681946f93594aull;
static const std::vector<Pass> kMarket = {
    {'D', 0}, {'D', 0}, {'D', 0},
    {'R', kQuarterCopy}, {'R', kHalfCopy},
    {'D', 0}, {'D', 0},
    {'R', kHalfCopy},
    {'D', 0},
    {'R', kQuarter2},
    {'D', 0}, {'D', 0}, {'D', 0},
    {'R', kBloom},
    {'R', kMapping},
};

// Draws one frame's passes with the trigger t: where DLAA ran (-1: nowhere), and the frame for learn()
static int play(const std::vector<Pass>& passes, const pb::Trigger& t, pb::Frame& f) {
    f = {};
    int ran = -1;
    for (size_t i = 0; i < passes.size(); i++) {
        if (passes[i].kind == 'D') f.scene_draw();
        else if (f.sampler(passes[i].key, t)) ran = (int)i;
    }
    return ran;
}

int main() {
    pb::Frame f;
    // ---- the first frame: no place known, DLAA at the network; it teaches the bloom's first draw ----
    CHECK("first frame: nowhere", play(kMarket, {}, f) == -1);
    const pb::Trigger t = pb::learn(f);
    CHECK("learned", t.valid);
    CHECK("the bloom's first draw", t.key == kBloom);
    CHECK("its place: the fifth reader", t.place == 4);
    CHECK("six readers", f.samplers == 6);
    // ---- the next frame: DLAA before the bloom's first draw, after the last scene draw, and it teaches the same ----
    CHECK("next frame: before the bloom", play(kMarket, t, f) == 13);
    CHECK("no late draw", !f.late);
    CHECK("the same place again", pb::learn(f) == t);
    // ---- never at the first reader (a copy the scene is drawn into after) ----
    {
        const pb::Trigger first{kQuarterCopy, 0, true};
        CHECK("a copy's place: DLAA runs there", play(kMarket, first, f) == 3);
        CHECK("the draws after it: late", f.late);
        CHECK("late teaches nothing", !pb::learn(f).valid);
    }
    // ---- the same program at another place does not fire (the copies' program at the bloom's place) ----
    {
        std::vector<Pass> other = kMarket;
        other[13].key = kHalfCopy;
        CHECK("another program at the place: nowhere", play(other, t, f) == -1);
        CHECK("it teaches that one", pb::learn(f).key == kHalfCopy && pb::learn(f).place == 4);
    }
    // ---- the bloom's program, but one reader more before it (another area's passes): nowhere, then it learns ----
    {
        std::vector<Pass> more = kMarket;
        more.insert(more.begin() + 9, Pass{'R', kHalfCopy});
        CHECK("one reader more: nowhere", play(more, t, f) == -1);
        const pb::Trigger m = pb::learn(f);
        CHECK("learned there", m.valid && m.key == kBloom && m.place == 5);
        CHECK("then before the bloom", play(more, m, f) == 14);
    }
    // ---- no reader after the last scene draw (a menu over the scene, no bloom): nothing learned ----
    {
        const std::vector<Pass> none = {{'D', 0}, {'R', kQuarterCopy}, {'D', 0}};
        CHECK("none: nowhere", play(none, t, f) == -1);
        CHECK("none: nothing learned", !pb::learn(f).valid);
        const std::vector<Pass> empty = {};
        CHECK("an empty frame: nothing learned", play(empty, t, f) == -1 && !pb::learn(f).valid);
    }
    // ---- it runs once a frame: the same program at the same place cannot come twice, and later readers do not fire ----
    {
        pb::Frame g;
        const pb::Trigger at0{kBloom, 0, true};
        g.scene_draw();
        CHECK("fires", g.sampler(kBloom, at0));
        CHECK("then not again", !g.sampler(kBloom, at0) && !g.sampler(kBloom, at0));
        CHECK("ran at 0", g.ran && g.ran_at == 0);
    }
    // ---- more readers than the list holds: the place past it is not learned, one inside it is ----
    {
        pb::Frame g;
        for (uint32_t i = 0; i < pb::kSamplers + 4; i++) (void)g.sampler(i + 1, {});
        g.scene_draw();
        (void)g.sampler(0x77, {});
        CHECK("past the list: nothing learned", !pb::learn(g).valid);
        pb::Frame h;
        for (uint32_t i = 0; i < pb::kSamplers - 1; i++) (void)h.sampler(i + 1, {});
        h.scene_draw();
        (void)h.sampler(0x77, {});
        const pb::Trigger last = pb::learn(h);
        CHECK("the list's last place: learned", last.valid && last.key == 0x77 && last.place == pb::kSamplers - 1);
    }
    // ---- the menu's DLAA before the bloom row (upscaler_rows.h): shown with NVIDIA DLSS where it runs, changeable at DLAA
    // alone (DLSS's other modes draw the scene smaller than DLSS's picture), fixed while RX_VK_DLSS_PRE_BLOOM sets it ----
    {
        namespace ur = mkwvk::upscaler_rows;
        CHECK("shown: DLAA", ur::pre_bloom_shown(true, ur::DLAA));
        CHECK("shown: DLSS", ur::pre_bloom_shown(true, ur::DLSS));
        CHECK("hidden: no DLSS here", !ur::pre_bloom_shown(false, ur::DLAA));
        for (int u : {ur::GAME, ur::FSR3, ur::FSR4, ur::FSR4_AA, ur::OFF, ur::XESS, ur::XESS_AA})
            CHECK("hidden: another upscaler", !ur::pre_bloom_shown(true, u));
        CHECK("changeable: DLAA", !ur::pre_bloom_fixed(ur::DLAA, false));
        CHECK("fixed: DLSS below Native", ur::pre_bloom_fixed(ur::DLSS, false));
        CHECK("fixed: the switch set", ur::pre_bloom_fixed(ur::DLAA, true));
    }
    std::puts("pre_bloom: ok");
    return 0;
}
