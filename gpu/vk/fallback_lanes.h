// The lane count a mesh fallback compute stage runs at (vk_draw.cpp fallback_compute_pipelines). tests/test_fallback_lanes.py
//
// (hookshot) NVN's programs are written for warps of 32 lanes, so a stage with a subgroup instruction is held to 32 lanes. On
// Intel's GPUs that is the widest a thread runs, with the fewest registers a lane: Hyrule Castle's colour mesh stages have about
// 280 values live at once against the 32 a lane has at 32 lanes, and spent most of their time moving them to and from scratch
// memory. By Mesa's count (Iris Xe, Mesa 26.2.4) a workgroup of d8e59341cf4f629c takes 843,112 cycles at 32 lanes, 2 x 285,094
// at 16 and 4 x 70,492 at 8; 739acd678ae08655 591,068, 2 x 232,487 and 4 x 75,760.
//
// The translator can keep such a stage's warp in workgroup memory (mma_layout.h mesh_fallback_shared_warp): its SPIR-V then has
// no subgroup instruction and reads no subgroup input, and gives the same result on any number of lanes. spirv_lane_free tells
// that from the SPIR-V itself, where it is written to or read from the program cache; a stage that is not, or whose code was
// never read, stays at 32.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fallback_lanes {

// (the lanes a lane-free stage runs at on Mesa's Intel driver: where the spilling is nearly gone)
constexpr uint32_t kIntelLanes = 8;

// Whether a SPIR-V module's result cannot depend on the subgroup it runs in: no OpGroupNonUniform* instruction, no cooperative
// matrix, no load of a subgroup built-in input (through an access chain or not), no barrier of Subgroup scope. false for a
// module that is not SPIR-V or is cut short.
inline bool spirv_lane_free(const uint32_t* words, size_t count) {
    if (count < 5 || words[0] != 0x07230203u) return false;
    std::vector<uint32_t> inputs;      // (ids of subgroup built-in variables, and of access chains into them)
    std::vector<uint32_t> subgroups;   // (ids of 32-bit constants of value 3, Scope Subgroup)
    const auto has = [](const std::vector<uint32_t>& ids, uint32_t id) {
        for (const uint32_t i : ids)
            if (i == id) return true;
        return false;
    };
    for (size_t at = 5; at < count;) {
        const uint32_t op = words[at] & 0xFFFFu, length = words[at] >> 16;
        if (length == 0 || length > count - at) return false;
        const uint32_t* a = words + at + 1;
        if (op >= 333u && op <= 366u) return false;     // (OpGroupNonUniformElect .. OpGroupNonUniformQuadSwap)
        if (op >= 4456u && op <= 4460u) return false;   // (OpTypeCooperativeMatrixKHR .. OpCooperativeMatrixLengthKHR)
        if (op == 71u && length >= 4 && a[1] == 11u) {  // (OpDecorate id BuiltIn which)
            const uint32_t which = a[2];
            if ((which >= 36u && which <= 41u) || (which >= 4416u && which <= 4420u)) inputs.push_back(a[0]);
        } else if (op == 43u && length == 4 && a[2] == 3u) {   // (OpConstant type id 3)
            subgroups.push_back(a[1]);
        } else if ((op == 65u || op == 66u) && length >= 4 && has(inputs, a[2])) {   // (Op[InBounds]AccessChain type id base)
            inputs.push_back(a[1]);
        } else if (op == 61u && length >= 4 && has(inputs, a[2])) {   // (OpLoad type id pointer)
            return false;
        } else if (op == 224u && length >= 4 && has(subgroups, a[0])) {   // (OpControlBarrier execution memory semantics)
            return false;
        }
        at += length;
    }
    return true;
}

// RX_VK_MESH_LANES's value: 8, 16 or 32 (32: every stage as before). 0 for anything else, which the caller ignores.
inline uint32_t parse_lanes(const char* text) {
    if (!text) return 0;
    uint32_t v = 0;
    for (const char* c = text; *c; c++) {   // (digits only: no sign, no space, nothing after)
        if (*c < '0' || *c > '9' || v > 32) return 0;
        v = v * 10 + (uint32_t)(*c - '0');
    }
    return (v == 8 || v == 16 || v == 32) ? v : 0;
}

// The lanes lane-free fallback stages are asked to run at on this device: RX_VK_MESH_LANES where it is a size the device can
// be held to, else 8 on Mesa's Intel driver where it can, else 32 (as every other stage).
inline uint32_t wanted(bool mesa_intel, uint32_t env_lanes, uint32_t device_min, uint32_t device_max) {
    const auto fits = [&](uint32_t lanes) { return lanes >= device_min && lanes <= device_max; };
    if (env_lanes) return fits(env_lanes) ? env_lanes : 32;
    return mesa_intel && fits(kIntelLanes) ? kIntelLanes : 32;
}

// The lanes one fallback compute stage runs at
inline uint32_t lanes_of(bool lane_free, uint32_t wanted_lanes) { return lane_free ? wanted_lanes : 32; }

}   // namespace fallback_lanes
