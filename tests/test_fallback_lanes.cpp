// fallback_lanes (gpu/vk/fallback_lanes.h): the lanes a mesh fallback compute stage runs at. tests/test_fallback_lanes.py
#include "../gpu/vk/fallback_lanes.h"

#include <cstdint>
#include <cstdio>
#include <vector>

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
}

// A module: the header, then the instructions as (opcode, operand words)
static std::vector<uint32_t> module(const std::vector<std::vector<uint32_t>>& insts) {
    std::vector<uint32_t> w{0x07230203u, 0x00010300u, 0, 100, 0};
    for (const auto& i : insts) {
        w.push_back((uint32_t)(i.size() << 16) | i[0]);
        w.insert(w.end(), i.begin() + 1, i.end());
    }
    return w;
}

static bool lane_free(const std::vector<uint32_t>& m) { return fallback_lanes::spirv_lane_free(m.data(), m.size()); }

int main() {
    using namespace fallback_lanes;
    // OpDecorate (71) id BuiltIn (11) which; OpConstant (43) type id value; OpVariable (59); OpAccessChain (65) type id base index;
    // OpLoad (61) type id pointer; OpControlBarrier (224) execution memory semantics; OpGroupNonUniformBallot (339);
    // OpGroupNonUniformShuffle (345); OpAtomicOr (241); OpLabel (248); OpReturn (253)
    const std::vector<uint32_t> plain = module({{248, 1}, {128, 7, 16, 10, 13}, {253}});
    expect(lane_free(plain), "no subgroup instruction: lane-free");
    // (the shared warp's own instructions: atomics on workgroup words, a Workgroup barrier, the local invocation id (27) read)
    const std::vector<uint32_t> shared = module({{71, 20, 11, 27}, {43, 7, 30, 2}, {43, 7, 31, 0}, {59, 8, 20, 1}, {248, 1}, {61, 9, 21, 20},
                                                 {241, 7, 22, 40, 30, 31, 23}, {224, 30, 30, 31}, {253}});
    expect(lane_free(shared), "atomics, a workgroup barrier and the local invocation id: lane-free");
    expect(!lane_free(module({{248, 1}, {339, 7, 16, 30, 13}, {253}})), "a ballot: not lane-free");
    expect(!lane_free(module({{248, 1}, {345, 7, 16, 30, 13, 14}, {253}})), "a shuffle: not lane-free");
    expect(!lane_free(module({{248, 1}, {333, 7, 16, 30}, {253}})), "the first subgroup instruction (elect): not lane-free");
    expect(!lane_free(module({{248, 1}, {366, 7, 16, 30, 13, 14}, {253}})), "the last (quad swap): not lane-free");
    expect(lane_free(module({{248, 1}, {332, 7, 16, 13}, {367, 7, 17, 13}, {253}})), "the instructions beside them: lane-free");
    expect(!lane_free(module({{4456, 7, 8, 30, 9, 10, 11}, {248, 1}, {253}})), "a cooperative matrix type: not lane-free");
    // (the subgroup's lane id, built-in 41: declared and not read, then read)
    expect(lane_free(module({{71, 20, 11, 41}, {59, 8, 20, 1}, {248, 1}, {253}})), "a subgroup input declared, never read: lane-free");
    expect(!lane_free(module({{71, 20, 11, 41}, {59, 8, 20, 1}, {248, 1}, {61, 9, 21, 20}, {253}})), "the subgroup's lane id read: not lane-free");
    for (const uint32_t which : {36u, 37u, 38u, 39u, 40u, 4416u, 4417u, 4418u, 4419u, 4420u})
        expect(!lane_free(module({{71, 20, 11, which}, {59, 8, 20, 1}, {248, 1}, {61, 9, 21, 20}, {253}})), "a subgroup built-in read: not lane-free");
    for (const uint32_t which : {0u, 26u, 27u, 28u, 29u, 35u, 42u, 4415u, 4421u})
        expect(lane_free(module({{71, 20, 11, which}, {59, 8, 20, 1}, {248, 1}, {61, 9, 21, 20}, {253}})), "another built-in read: lane-free");
    // (a mask's component: an access chain into the input, then the load)
    expect(!lane_free(module({{71, 20, 11, 4420}, {59, 8, 20, 1}, {248, 1}, {65, 9, 21, 20, 31}, {61, 7, 22, 21}, {253}})),
           "a subgroup mask read through an access chain: not lane-free");
    expect(!lane_free(module({{71, 20, 11, 4420}, {59, 8, 20, 1}, {248, 1}, {66, 9, 21, 20, 31}, {61, 7, 22, 21}, {253}})),
           "and through an in-bounds access chain");
    // (a barrier of Subgroup scope, 3; of Workgroup scope, 2)
    expect(!lane_free(module({{43, 7, 30, 3}, {43, 7, 31, 0}, {248, 1}, {224, 30, 30, 31}, {253}})), "a subgroup barrier: not lane-free");
    expect(lane_free(module({{43, 7, 30, 2}, {43, 7, 31, 0}, {248, 1}, {224, 30, 30, 31}, {253}})), "a workgroup barrier: lane-free");
    // (not SPIR-V, cut short, a zero-length instruction: never lane-free, so such a stage stays at 32 lanes)
    std::vector<uint32_t> bad = plain;
    bad[0] = 0x12345678u;
    expect(!lane_free(bad), "not SPIR-V: not lane-free");
    std::vector<uint32_t> cut = plain;
    cut.resize(cut.size() - 2);
    expect(!lane_free(cut), "a module cut short: not lane-free");
    std::vector<uint32_t> zero = plain;
    zero[5] = 0;
    expect(!lane_free(zero), "a zero-length instruction: not lane-free, not a loop");
    expect(!spirv_lane_free(plain.data(), 3), "shorter than a header: not lane-free");

    expect(parse_lanes("8") == 8 && parse_lanes("16") == 16 && parse_lanes("32") == 32, "8, 16 and 32");
    for (const char* text : {"", "0", "4", "12", "64", "8x", " 8", "-8", "eight", "0x8"})
        expect(parse_lanes(text) == 0, "anything else: 0");
    expect(parse_lanes(nullptr) == 0, "not set: 0");

    expect(wanted(true, 0, 8, 32) == 8, "Mesa's Intel driver: 8");
    expect(wanted(false, 0, 8, 32) == 32, "another driver: 32");
    expect(wanted(false, 0, 4, 64) == 32, "another driver, whatever it can be held to: 32");
    expect(wanted(true, 16, 8, 32) == 16, "RX_VK_MESH_LANES=16 over the default");
    expect(wanted(true, 32, 8, 32) == 32, "RX_VK_MESH_LANES=32: as before");
    expect(wanted(false, 8, 8, 32) == 8, "RX_VK_MESH_LANES=8 on another driver");
    expect(wanted(false, 8, 16, 64) == 32, "a size the device cannot be held to: 32");
    expect(wanted(true, 0, 16, 32) == 32, "Mesa's Intel driver without 8 lanes: 32");

    expect(lanes_of(true, 8) == 8 && lanes_of(true, 16) == 16, "a lane-free stage: the lanes wanted");
    expect(lanes_of(false, 8) == 32 && lanes_of(false, 16) == 32, "a stage with a subgroup instruction: 32");
    expect(lanes_of(true, 32) == 32, "32 wanted: 32");

    if (failures) return 1;
    std::puts("fallback_lanes: ok");
    return 0;
}
