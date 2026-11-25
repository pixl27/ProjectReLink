// Pure bounds/checks for the diagnostic mesh capture and the one-result recording cache.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace mesh_capture {
constexpr uint32_t max_words = 128, sample_vertices = 64, sample_slots = 64;
constexpr uint32_t header_words = 8;
constexpr size_t staging_words = header_words + max_words * sample_vertices + sample_slots;
constexpr size_t max_uniform_bytes = 16u << 10;
constexpr uint32_t task = 1, indirect = 2, storage = 4, image = 8, global = 16, write = 32, soa = 64;
constexpr uint32_t candidate = 128;

inline uint32_t stored_vertices(uint32_t count, uint32_t capacity) {
    return count <= capacity ? std::min({count, capacity, sample_vertices}) : 0;
}
inline uint32_t stored_slots(uint32_t count, uint32_t capacity) {
    return count <= capacity ? std::min(count, sample_slots) : 0;
}

struct Inputs {
    std::array<uint8_t, max_uniform_bytes> bytes;
    size_t size = 0;
    bool append(std::span<const uint8_t> data) {
        // Include each descriptor's extent, so different splits cannot compare equal.
        const uint64_t n = data.size();
        if (size > bytes.size() - sizeof(n) || n > bytes.size() - size - sizeof(n)) return false;
        std::memcpy(bytes.data() + size, &n, sizeof(n));
        size += sizeof(n);
        if (n) std::memcpy(bytes.data() + size, data.data(), n);
        size += n;
        return true;
    }
    bool same(const Inputs& other) const {
        return size == other.size && std::memcmp(bytes.data(), other.bytes.data(), size) == 0;
    }
};
} // namespace mesh_capture
