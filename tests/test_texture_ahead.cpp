// Textures prepared ahead of their first draw (gpu/vk/texture_ahead.h): tests/test_texture_ahead.py compiles and runs this.
#include "../gpu/vk/texture_ahead.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::printf("FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    using namespace mkwvk::texture_ahead;
    // ---- the signature: a texture's shape and a sample of its storage ----
    std::vector<uint8_t> a(4096), b;
    for (size_t i = 0; i < a.size(); i++) a[i] = (uint8_t)(i * 7 + 3);
    b = a;
    const uint64_t sa = signature(0x4b, 256, 256, 1, 9, a.size(), a.data());
    CHECK(sa != 0);
    CHECK(sa == signature(0x4b, 256, 256, 1, 9, b.size(), b.data()));       // (the same texture in another run)
    CHECK(sa != signature(0x4c, 256, 256, 1, 9, a.size(), a.data()));       // (another format)
    CHECK(sa != signature(0x4b, 128, 256, 1, 9, a.size(), a.data()));       // (another shape)
    CHECK(sa != signature(0x4b, 256, 256, 1, 8, a.size(), a.data()));
    CHECK(sa != signature(0x4b, 256, 256, 1, 9, a.size() - 64, a.data()));  // (another size)
    b[0] ^= 1;                                                              // (its first bytes are in the sample)
    CHECK(sa != signature(0x4b, 256, 256, 1, 9, b.size(), b.data()));
    b = a, b[b.size() - 1] ^= 1;                                            // (and its last)
    CHECK(sa != signature(0x4b, 256, 256, 1, 9, b.size(), b.data()));
    CHECK(signature(0x4b, 256, 256, 1, 9, a.size(), nullptr) == 0);         // (storage not readable: no signature)
    // (a storage under 64 bytes: its shape alone, nothing read)
    CHECK(signature(0x25, 4, 4, 1, 1, 16, a.data()) == signature(0x25, 4, 4, 1, 1, 16, a.data() + 100));
    CHECK(signature(0x25, 4, 4, 1, 1, 16, a.data()) != 0);
    // (exactly 64 bytes: read once from end to end, never past them)
    CHECK(signature(0x25, 4, 4, 1, 1, 64, a.data()) != signature(0x25, 4, 4, 1, 1, 64, a.data() + 1));

    // ---- the allowance: after a light frame only, told by its draws (the same on a slow PC and a fast one) ----
    const int64_t ms = 1000000;
    CHECK(allowance_ns(5, 12 * ms) == 12 * ms);                 // (a loading screen)
    CHECK(allowance_ns(81, 12 * ms) == 12 * ms);                // (the title, a menu)
    CHECK(allowance_ns(kLightDraws - 1, 12 * ms) == 12 * ms);
    CHECK(allowance_ns(kLightDraws, 12 * ms) == 0);
    CHECK(allowance_ns(1188, 12 * ms) == 0);                    // (play)
    CHECK(kLightDraws >= 100 && kLightDraws <= 500);

    // ---- room in the cache: a reserve under its budget stays free ----
    const uint64_t MB = 1ull << 20;
    CHECK(kReserveBytes == 256 * MB);
    CHECK(room(1000 * MB, 2400 * MB));
    CHECK(room(2144 * MB, 2400 * MB));
    CHECK(!room(2145 * MB, 2400 * MB));
    CHECK(!room(3000 * MB, 2400 * MB));
    CHECK(!room(0, 0));                                         // (the budget not known yet)
    const uint64_t vega_reserve = kVegaReserveBytes + vega_upload_estimate(1024, 1024);
    CHECK(vega_reserve == 41 * MB);
    CHECK(room(131 * MB, 218 * MB, vega_reserve));
    CHECK(!room(178 * MB, 218 * MB, vega_reserve));
    CHECK(!room(UINT64_MAX, 218 * MB, vega_reserve));
    CHECK(!vega_upload_estimate(1025, 1024));
    CHECK(!vega_upload_estimate(0, 1024));

    // ---- the queue: oldest first, a second after it was made ----
    Queue<int> q;
    int got = 0;
    q.made(0x1000, 0, 1);
    q.made(0x2000, 500, 2);
    CHECK(q.waiting() == 2);
    CHECK(!q.take(999, &got));                    // (not settled yet)
    CHECK(q.take(1000, &got) && got == 1);
    CHECK(!q.take(1000, &got));                   // (the next one was made half a second later)
    CHECK(q.take(1500, &got) && got == 2);
    CHECK(q.waiting() == 0 && !q.take(99999, &got));
    // (released, or drawn meanwhile: not handed out)
    q.made(0x1000, 2000, 3);
    q.made(0x2000, 2000, 4);
    q.gone(0x1000);
    CHECK(q.take(3000, &got) && got == 4);
    CHECK(!q.take(3000, &got));
    // (made again at the same place: the new one alone, at its own time)
    q.made(0x3000, 4000, 5);
    q.made(0x3000, 4600, 6);
    CHECK(q.waiting() == 1);
    CHECK(!q.take(5000, &got));
    CHECK(q.take(5600, &got) && got == 6);
    CHECK(!q.take(9000, &got));
    // (gone, then made again at the same place: the new one is there)
    q.made(0x4000, 10000, 7);
    q.gone(0x4000);
    q.made(0x4000, 10100, 8);
    CHECK(q.take(11100, &got) && got == 8);
    // (one nobody could prepare for two minutes is let go: made in play, where no frame is light)
    q.made(0x5000, 20000, 9);
    q.made(0x6000, 139000, 10);
    CHECK(q.trim(140000) == 0 && q.waiting() == 2);
    CHECK(q.trim(140001) == 1 && q.waiting() == 1);
    CHECK(q.take(140001, &got) && got == 10);
    q.made(0x7000, 200000, 11);
    CHECK(!q.take(200000 + kStaleMs + 1, &got) && q.waiting() == 0);   // (take lets them go too)

    // ---- when a texture was made, kept until it is released: what a draw's first upload of it is aged by ----
    Queue<int> m;
    int64_t at = -1;
    CHECK(!m.made_at(0x1000, &at));
    m.made(0x1000, 100, 1);
    CHECK(m.made_at(0x1000, &at) && at == 100);
    CHECK(m.take(1100, &got) && got == 1);
    CHECK(m.made_at(0x1000, &at) && at == 100);     // (looked at already: still known, its first draw may come later)
    m.made(0x1000, 5000, 2);                        // (made again at the same place)
    CHECK(m.made_at(0x1000, &at) && at == 5000);
    m.gone(0x1000);
    CHECK(!m.made_at(0x1000, &at));
    m.made(0x2000, 10000, 3);                       // (one let go for its age is still the game's texture)
    CHECK(m.trim(10000 + kStaleMs + 1) == 1 && m.made_at(0x2000, &at) && at == 10000);

    // ---- what is remembered: a texture first drawn soon after it was made (a load's), not one met later in play ----
    // (a session of an hour and a half tripled the list with textures drawn minutes after their load, and each load then
    // prepared them all: 0.5 to 1 GB more video memory on a Steam Deck)
    CHECK(learnable(1000, 1000));
    CHECK(learnable(1000, 1000 + kLearnMs));
    CHECK(!learnable(1000, 1000 + kLearnMs + 1));
    CHECK(kLearnMs >= 60000);   // (an arrival's textures were made up to a minute before their first draw)

    // ---- what the list holds: a texture where the game put it ----
    // (the game makes the same textures in several zones, at other places, and each zone's first frames draw only some of
    // them: kept by their signature alone, every zone prepared what any zone had drawn. Hyrule Field: 1910 textures
    // prepared for the 770 its first frames draw, 0.8 GB more video memory)
    const uint64_t at_a = place(sa, 0x10a9cca400ull), at_b = place(sa, 0x10aa5e0e00ull);
    CHECK(at_a != 0 && at_a != at_b && at_a != sa);
    CHECK(at_a == place(sa, 0x10a9cca400ull));   // (the same zone loaded again puts it there again)
    CHECK(place(sa + 1, 0x10a9cca400ull) != at_a);
    CHECK(place(0, 0x10a9cca400ull) == 0);       // (no signature: no entry)

    // ---- the list is bounded: full, it starts again with the next new texture ----
    // (a save loaded again puts some of its textures elsewhere, each a new entry: it would grow without end)
    std::unordered_set<uint64_t> list;
    CHECK(remember(&list, 5) && list.size() == 1);
    CHECK(!remember(&list, 5) && list.size() == 1);                       // (one it has)
    for (uint64_t i = 1; list.size() < kDrawnMax; i++) CHECK(remember(&list, i << 8));
    CHECK(!remember(&list, 5) && list.size() == kDrawnMax);               // (full, and one it has: nothing changes)
    CHECK(remember(&list, 7) && list.size() == 1 && list.count(7) == 1);  // (full, and a new one: it starts again)
    CHECK(kDrawnMax >= 8192);

    // ---- the signatures drawn before, kept in a file between runs ----
    if (argc > 1) {
        const std::string path = std::string(argv[1]) + "/drawn.bin";
        std::unordered_set<uint64_t> set{1, 2, 0xfeedfacecafebeefull}, back;
        CHECK(!load_drawn(path.c_str(), &back) && back.empty());   // (no file yet)
        CHECK(save_drawn(path.c_str(), set));
        CHECK(load_drawn(path.c_str(), &back) && back == set);
        set.insert(77);
        CHECK(save_drawn(path.c_str(), set));                      // (over the one there)
        back.clear();
        CHECK(load_drawn(path.c_str(), &back) && back == set);
        // (a file cut short, or not one of these: nothing taken from it)
        std::FILE* f = std::fopen(path.c_str(), "rb+");
        CHECK(f);
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f);
        std::fclose(f);
        f = std::fopen(path.c_str(), "wb");
        std::vector<char> junk((size_t)size - 3, 'x');
        std::fwrite(junk.data(), 1, junk.size(), f);
        std::fclose(f);
        back.clear();
        CHECK(!load_drawn(path.c_str(), &back) && back.empty());
        // (a list from before: version 1 held textures drawn at any time, version 2 signatures without their place: not taken)
        f = std::fopen(path.c_str(), "wb");
        const uint32_t old_head[2] = {kDrawnMagic, 2};
        const uint64_t none = 0;
        std::fwrite(old_head, sizeof old_head, 1, f);
        std::fwrite(&none, sizeof none, 1, f);
        std::fclose(f);
        CHECK(kDrawnVersion == 3 && !load_drawn(path.c_str(), &back) && back.empty());
        // (one that says it holds more than a list ever does: not taken)
        f = std::fopen(path.c_str(), "wb");
        const uint32_t head[2] = {kDrawnMagic, kDrawnVersion};
        const std::vector<uint64_t> many(kDrawnMax + 1, 9);
        const uint64_t count = many.size();
        std::fwrite(head, sizeof head, 1, f);
        std::fwrite(&count, sizeof count, 1, f);
        std::fwrite(many.data(), sizeof(uint64_t), many.size(), f);
        std::fclose(f);
        CHECK(!load_drawn(path.c_str(), &back) && back.empty());
    }
    std::printf("PASS: textures prepared ahead\n");
    return 0;
}
