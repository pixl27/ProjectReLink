// Effective texture-memory allowance for policies that choose image formats and staging retention.
// nx2/test_texture_memory_policy.py
#pragma once
#include <algorithm>
#include <cstdint>

namespace texture_memory_policy {

constexpr uint64_t MiB = 1ull << 20;
constexpr uint64_t IntegratedDefault = 2560ull * MiB;

// A discrete heap is dedicated VRAM.  An integrated heap is shared system RAM, so the driver's heap allowance is not
// what textures may consume beside the game.  An explicit texture budget is a still tighter statement of that limit.
inline uint64_t effective_allowance(uint64_t driver_allowed, bool integrated, uint64_t configured_budget = 0) {
    const uint64_t cap = configured_budget ? configured_budget : integrated ? IntegratedDefault : 0;
    return cap ? (driver_allowed ? std::min(driver_allowed, cap) : cap) : driver_allowed;
}

inline uint64_t staging_keep(uint64_t effective_allowed) {
    return effective_allowed && effective_allowed <= 6656ull * MiB ? 64ull * MiB : 256ull * MiB;
}

} // namespace texture_memory_policy
