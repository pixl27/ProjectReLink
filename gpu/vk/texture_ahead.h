// (hookshot, smoothness) Textures prepared ahead of their first draw: the rules (vk_texture.cpp uses them; no Vulkan here;
// tests/test_texture_ahead.py).
//
// A texture is made and uploaded at the first draw that samples it, so after a save is loaded its area's textures all come
// in the first frames after the loading screen: 700 to 1200 of them, two frames of about a second each on a Steam Deck. The
// host announces a texture when the game makes it, and those were made 5 to 60 seconds before with their storage already
// written. But the game makes three to six times more textures than the first frames sample (4500 waiting after a minute in
// Kokiri Forest, 1.4 GB of storage), and nothing in the announcement tells which: so the ones a draw has sampled before are
// remembered by a signature and where the game put them, kept between runs, and only those are prepared, a few milliseconds
// a frame while the frames are light (a loading screen).
// It does nothing for an area change in play: there the game keeps its textures (none is announced again), and what a
// return uploads is what the cache's budget let go.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mkwvk::texture_ahead {

/// How long after it was made a texture is looked at: the game has written its storage by then
constexpr int64_t kSettleMs = 1000;
/// And how long one waits to be prepared before it is let go (made in play, where no frame is light)
constexpr int64_t kStaleMs = 120000;
/// A texture is remembered only when its first draw comes this soon after the game made it: a load's textures (made 5 to
/// 60 s before the first frames that draw them). One met minutes later in play is not: an hour and a half of play tripled
/// the list with those, and every load then prepared them all, 0.5 to 1 GB more video memory on a Steam Deck
constexpr int64_t kLearnMs = 90000;
inline bool learnable(int64_t made_ms, int64_t now_ms) { return now_ms - made_ms <= kLearnMs; }

/// What tells a texture from another, in this run and the next: its shape and a sample of its storage (eight runs of 64
/// bytes spread over it; under 64 bytes, its shape alone). storage: `size` bytes; null (not readable): 0, no signature
inline uint64_t signature(uint32_t format, uint32_t width, uint32_t height, uint32_t depth, uint32_t levels, uint64_t size,
                          const uint8_t* storage) {
    if (!storage) return 0;
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    mix(format), mix(width), mix(height), mix(depth), mix(levels), mix(size);
    if (size >= 64)
        for (int k = 0; k < 8; k++) {
            const uint8_t* q = storage + (size - 64) / 7 * k;
            for (int i = 0; i < 64; i += 8) {
                uint64_t w;
                std::memcpy(&w, q + i, 8);
                mix(w);
            }
        }
    return h ? h : 1;
}

/// What the list of textures drawn holds: a texture where the game put it (its signature and where its storage starts).
/// The game makes the same textures in several areas, at other places, and each area's first frames draw only some of
/// them: kept by their signature alone, every area prepared what any area had drawn (Hyrule Field on a Steam Deck: 1910
/// textures prepared for the 770 its first frames draw, 0.8 GB more video memory). A save loaded again puts most of its
/// textures where they were: two thirds to nine tenths of those an arrival draws are found, the others are uploaded at
/// their first draw as without this. 0 for no signature
inline uint64_t place(uint64_t signature, uint64_t storage_start) {
    if (!signature) return 0;
    uint64_t h = (signature ^ (storage_start * 0x9E3779B97F4A7C15ull)) * 1099511628211ull;
    h ^= h >> 29;
    return h ? h : 1;
}

/// A frame of fewer draws than this is light: a loading screen has 5, the title and the menus under 100, play over 1000.
/// Told by its draws, not by its time: a fast PC replays a frame of play in the time a slow one takes for a loading screen's
constexpr int64_t kLightDraws = 200;
/// The time a frame may spend preparing textures: `budget_ns` after a light frame, none in play
inline int64_t allowance_ns(int64_t last_draws, int64_t budget_ns) { return last_draws < kLightDraws ? budget_ns : 0; }

/// What stays free under the texture cache's budget
constexpr uint64_t kReserveBytes = 256ull << 20;
/// Whether the texture cache has room for one more: kReserveBytes under its budget stay free, so that a texture prepared
/// never pushes another out (a budget of 0: not known yet, no room)
inline bool room(uint64_t cache_bytes, uint64_t budget_bytes, uint64_t reserve = kReserveBytes) {
    return cache_bytes <= budget_bytes && reserve <= budget_bytes - cache_bytes;
}

// Bound speculative uploads to single-sample 2D textures no larger than 1024 square.
// Eight bytes per base texel cover RGBA8 and its mip chain, plus room for image padding.
constexpr uint64_t kVegaReserveBytes = 32ull << 20;
inline uint64_t vega_upload_estimate(uint32_t width, uint32_t height) {
    if (!width || !height || width > 1024 || height > 1024) return 0;
    return (uint64_t)width * height * 8 + (1ull << 20);
}

/// The textures the game has made and that wait to be looked at, oldest first, by where their storage starts
template <class Item>
class Queue {
public:
    void made(uint64_t key, int64_t now_ms, const Item& item) {
        order_.push_back({key, ++seq_, now_ms, item});
        live_[key] = seq_;
        made_at_[key] = now_ms;
    }
    /// Released: it waits no longer, and is no longer the game's
    void gone(uint64_t key) {
        live_.erase(key);
        made_at_.erase(key);
    }
    /// When the game made the texture whose storage starts there (looked at since or not); false: none it has now
    bool made_at(uint64_t key, int64_t* ms) const {
        const auto it = made_at_.find(key);
        if (it == made_at_.end()) return false;
        *ms = it->second;
        return true;
    }
    /// The oldest one made kSettleMs ago or more, taken out; false when none is due yet
    bool take(int64_t now_ms, Item* item) {
        trim(now_ms);
        if (order_.empty() || now_ms - order_.front().made_ms < kSettleMs) return false;
        *item = order_.front().item;
        live_.erase(order_.front().key);
        order_.pop_front();
        return true;
    }
    /// The ones at the front that are gone, made again since, or waiting for over kStaleMs, dropped: how many were let go
    /// for their age. After it the front, if any, is one that waits
    int trim(int64_t now_ms) {
        int stale = 0;
        while (!order_.empty()) {
            const Entry& e = order_.front();
            const auto it = live_.find(e.key);
            if (it != live_.end() && it->second == e.seq) {
                if (now_ms - e.made_ms <= kStaleMs) break;
                live_.erase(it);
                stale++;
            }
            order_.pop_front();
        }
        return stale;
    }
    size_t waiting() const { return live_.size(); }

private:
    struct Entry {
        uint64_t key, seq;
        int64_t made_ms;
        Item item;
    };
    std::deque<Entry> order_;
    std::unordered_map<uint64_t, uint64_t> live_;     // (where a storage starts -> the entry that is its texture now)
    std::unordered_map<uint64_t, int64_t> made_at_;   // (and when the game made that texture, until it releases it)
    uint64_t seq_ = 0;
};

// ---- the signatures of the textures a draw has sampled, between runs: "LSTD", a version, a count, the signatures ----
// (version 3: a texture where the game put it, place(); version 2 held signatures alone, version 1 any texture drawn)
constexpr uint32_t kDrawnMagic = 0x4454534c, kDrawnVersion = 3;
/// The most the list holds (256 KB in its file). Each load of a save adds the textures it draws that the list has not (at
/// another place than the last time, or not drawn then): a tenth to a third of an arrival's, 80 to 260 a load on a Steam
/// Deck. Full, the list starts again with the next new one and the loads after it fill it again
constexpr size_t kDrawnMax = 32768;

/// `entry` put in the list; false when it was there
inline bool remember(std::unordered_set<uint64_t>* set, uint64_t entry) {
    if (set->count(entry)) return false;
    if (set->size() >= kDrawnMax) set->clear();
    set->insert(entry);
    return true;
}

/// The file's signatures added to `set`; false (and nothing added) when it is not there, cut short or not one of these
inline bool load_drawn(const char* path, std::unordered_set<uint64_t>* set) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    uint32_t head[2] = {0, 0};
    uint64_t count = 0;
    std::vector<uint64_t> all;
    bool ok = std::fread(head, sizeof head, 1, f) == 1 && head[0] == kDrawnMagic && head[1] == kDrawnVersion &&
              std::fread(&count, sizeof count, 1, f) == 1 && count <= kDrawnMax;
    if (ok) {
        all.resize((size_t)count);
        ok = !count || std::fread(all.data(), sizeof(uint64_t), all.size(), f) == all.size();
        ok = ok && std::fgetc(f) == EOF;
    }
    std::fclose(f);
    if (!ok) return false;
    set->insert(all.begin(), all.end());
    return true;
}

/// `set` written to the file, whole, through a file beside it: a run cut off while it writes leaves the old one
inline bool save_drawn(const char* path, const std::unordered_set<uint64_t>& set) {
    const std::string tmp = std::string(path) + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const uint32_t head[2] = {kDrawnMagic, kDrawnVersion};
    const uint64_t count = set.size();
    const std::vector<uint64_t> all(set.begin(), set.end());
    bool ok = std::fwrite(head, sizeof head, 1, f) == 1 && std::fwrite(&count, sizeof count, 1, f) == 1 &&
              (!count || std::fwrite(all.data(), sizeof(uint64_t), all.size(), f) == all.size());
    ok = std::fclose(f) == 0 && ok;
    if (ok) {
        std::remove(path);   // (Windows' rename does not replace)
        ok = std::rename(tmp.c_str(), path) == 0;
    }
    if (!ok) std::remove(tmp.c_str());
    return ok;
}

}  // namespace mkwvk::texture_ahead
