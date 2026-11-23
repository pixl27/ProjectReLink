// (hookshot) The indexed descriptor arrays the programs declare: a texture, texture buffer, storage image or image buffer
// descriptor of a stage whose count is above 1, which the shader indexes (a handle read at a dynamic offset of a constant
// buffer, or from a handle table). host_info() gives the translator no device limits, so it sizes a dynamically indexed array
// with HostTranslateInfo's default limit, up to 1024 elements (texture_pass.cpp DynamicDescriptorCount, DESCRIPTOR_MAX_COUNT),
// on every GPU. An array that long can pass a GPU's per-stage limits: Apple's M1 and M2 reach 996 samplers a stage through an
// argument buffer (Metal feature set tables, Apple7 and Apple8). Whether the game declares such arrays, and image arrays at all,
// was never counted. The census counts them by stage, kind and size, and names the ones near a limit (the first 30): 996
// elements or more, a stage whose samplers pass the M1's and M2's 996, or a stage whose descriptors of that kind pass this GPU's
// per-stage limit.
// Each stage of a program is counted once, whatever its variants, where the program cache sees it: translated (saved) or read.
// Rules only, no Vulkan: vk_programs.cpp keeps the census (descriptor_arrays_note, descriptor_arrays_log).
// tests/test_descriptor_array_census.py runs them.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mkwvk {

struct ArrayCensus {
    enum Kind : int { kTexture, kTextureBuffer, kImage, kImageBuffer, kKinds };
    static constexpr const char* kKindNames[kKinds] = {"texture", "texture buffer", "storage image", "image buffer"};
    // (the stages as the GPU runs them: a mesh fallback's task and mesh stages are compute stages)
    static constexpr int kStages = 5;
    static constexpr const char* kStageNames[kStages] = {"vertex", "fragment", "compute", "task", "mesh"};
    static constexpr uint32_t kStageBits[kStages] = {0x1, 0x10, 0x20, 0x40, 0x80};   // (VkShaderStageFlagBits: vk_programs.cpp checks)
    static int stage_of(uint32_t vk_stage) {
        for (int s = 0; s < kStages; s++)
            if (vk_stage == kStageBits[s]) return s;
        return -1;
    }
    static constexpr int kSizes = 6;
    static constexpr uint32_t kSizeTop[kSizes - 1] = {16, 64, 256, 995, 1023};
    static constexpr const char* kSizeNames[kSizes] = {"2-16", "17-64", "65-256", "257-995", "996-1023", "1024 and more"};
    static constexpr uint32_t kCap = 1024;            // (the translator's: HostTranslateInfo::DEFAULT_DESCRIPTOR_LIMIT, DESCRIPTOR_MAX_COUNT)
    static constexpr uint32_t kNear = 996;            // (an array this long is near that cap)
    static constexpr uint64_t kAppleSamplers = 996;   // (Apple M1 and M2, Apple7 and Apple8: the samplers a stage reaches through an argument buffer)
    static constexpr int kNamed = 30;                 // (arrays near a limit named one by one; the rest are counted)
    struct Limits {   // (this GPU's, a stage: maxPerStageDescriptorSamplers, ...SampledImages, ...StorageImages)
        uint64_t samplers = UINT64_MAX, sampled_images = UINT64_MAX, storage_images = UINT64_MAX;
    };
    Limits limits;
    std::set<std::pair<uint64_t, uint32_t>> seen;   // (program key, stage index)
    uint64_t stages = 0, with_arrays = 0, arrays = 0, near_count = 0;   // (near_count, not near: windef.h defines near and far as empty macros)
    uint64_t sizes[kStages][kKinds][kSizes]{};
    uint32_t longest[kStages]{};
    int named = 0;

    static int size_of(uint32_t count) {
        int k = 0;
        while (k < kSizes - 1 && count > kSizeTop[k]) k++;
        return k;
    }

    /// A program's stage (index: its place in the program; vk_stage: its VkShaderStageFlagBits; info: its Shader::Info, or
    /// anything with the same four descriptor lists), counted once: its arrays counted, and a line for each one near a limit
    /// added to lines (the first kNamed of all)
    template <class Info>
    void note(uint64_t program, uint32_t index, uint32_t vk_stage, const Info& info, std::vector<std::string>& lines) {
        const int stage = stage_of(vk_stage);
        if (stage < 0 || !seen.insert({program, index}).second) return;
        stages++;
        uint64_t samplers = 0, sampled = 0, storage = 0;   // (the stage's, its single descriptors too)
        for (const auto& d : info.texture_descriptors) samplers += d.count, sampled += d.count;   // (combined image samplers)
        for (const auto& d : info.texture_buffer_descriptors) sampled += d.count;                 // (uniform texel buffers)
        for (const auto& d : info.image_descriptors) storage += d.count;
        for (const auto& d : info.image_buffer_descriptors) storage += d.count;                   // (storage texel buffers)
        bool any = false;
        const auto each = [&](Kind kind, const auto& list) {
            for (const auto& d : list) {
                const uint32_t count = d.count;
                if (count <= 1) continue;
                any = true;
                arrays++;
                sizes[stage][kind][size_of(count)]++;
                longest[stage] = std::max(longest[stage], count);
                std::string why;
                const auto add = [&why](const std::string& text) { why += (why.empty() ? ": " : "; ") + text; };
                char b[256];
                if (count >= kCap) add("at the translator's cap of 1024 elements, its default limit (the game's may be longer)");
                else if (count >= kNear) add("near the translator's cap of 1024 elements");
                if (kind == kTexture && samplers > kAppleSamplers) {
                    std::snprintf(b, sizeof b, "the stage's %llu samplers pass the %llu that an Apple M1 or M2 stage reaches through an "
                                  "argument buffer", (unsigned long long)samplers, (unsigned long long)kAppleSamplers);
                    add(b);
                }
                if (kind == kTexture && samplers > limits.samplers) {
                    std::snprintf(b, sizeof b, "the stage's %llu samplers pass this GPU's maxPerStageDescriptorSamplers (%llu)",
                                  (unsigned long long)samplers, (unsigned long long)limits.samplers);
                    add(b);
                }
                if ((kind == kTexture || kind == kTextureBuffer) && sampled > limits.sampled_images) {
                    std::snprintf(b, sizeof b, "the stage's %llu sampled images pass this GPU's maxPerStageDescriptorSampledImages (%llu)",
                                  (unsigned long long)sampled, (unsigned long long)limits.sampled_images);
                    add(b);
                }
                if ((kind == kImage || kind == kImageBuffer) && storage > limits.storage_images) {
                    std::snprintf(b, sizeof b, "the stage's %llu storage images pass this GPU's maxPerStageDescriptorStorageImages (%llu)",
                                  (unsigned long long)storage, (unsigned long long)limits.storage_images);
                    add(b);
                }
                if (why.empty()) continue;
                near_count++;
                if (named >= kNamed) continue;
                named++;
                std::snprintf(b, sizeof b, "vk: indexed descriptor array, program %016llx's %s stage: %s[%u]", (unsigned long long)program,
                              kStageNames[stage], kKindNames[kind], count);
                lines.push_back(b + why);
            }
        };
        each(kTexture, info.texture_descriptors);
        each(kTextureBuffer, info.texture_buffer_descriptors);
        each(kImage, info.image_descriptors);
        each(kImageBuffer, info.image_buffer_descriptors);
        with_arrays += any;
    }

    /// The census in lines (when: "the preparation", "close"): the counts, then one for each stage with arrays, by kind and size
    std::vector<std::string> summary(const char* when) const {
        std::vector<std::string> lines;
        char b[512];
        if (!arrays) {
            std::snprintf(b, sizeof b, "vk: indexed descriptor arrays (%s): none in %llu program stages", when, (unsigned long long)stages);
            lines.push_back(b);
            return lines;
        }
        char named_text[48] = "";
        if (near_count > (uint64_t)named) std::snprintf(named_text, sizeof named_text, " (the first %d named above)", named);
        else if (near_count) std::snprintf(named_text, sizeof named_text, " (named above)");
        std::snprintf(b, sizeof b, "vk: indexed descriptor arrays (%s): %llu in %llu of %llu program stages, %llu of them near a "
                      "limit%s", when, (unsigned long long)arrays, (unsigned long long)with_arrays, (unsigned long long)stages,
                      (unsigned long long)near_count, named_text);
        lines.push_back(b);
        for (int s = 0; s < kStages; s++) {
            if (!longest[s]) continue;
            std::string line = std::string("vk: indexed descriptor arrays (") + when + "), " + kStageNames[s] + " stages:";
            for (int k = 0; k < kKinds; k++) {
                uint64_t n = 0;
                for (int z = 0; z < kSizes; z++) n += sizes[s][k][z];
                if (!n) continue;
                std::snprintf(b, sizeof b, "%s %llu %s (", line.back() == ':' ? "" : ",", (unsigned long long)n, kKindNames[k]);
                line += b;
                bool first = true;
                for (int z = 0; z < kSizes; z++) {
                    if (!sizes[s][k][z]) continue;
                    std::snprintf(b, sizeof b, "%s%s: %llu", first ? "" : ", ", kSizeNames[z], (unsigned long long)sizes[s][k][z]);
                    line += b;
                    first = false;
                }
                line += ")";
            }
            std::snprintf(b, sizeof b, "; the longest %u", longest[s]);
            lines.push_back(line + b);
        }
        return lines;
    }
};

} // namespace mkwvk
