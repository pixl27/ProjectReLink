// Frame interpolation (RX_VK_FRAMEGEN=1; off by default): frames made between the game's, from its last two, shown in
// between by a display thread, so a game at 60 fps is shown at 120 (RX_VK_FRAMEGEN_MULT=3: 180). The WebGPU
// renderer's (gpu/render/rx_framegen.cpp, whose comments explain it), on Vulkan:
//   1. each frame is copied (it may be drawn over next frame) and reduced to quarter-size luma;
//   2. block matching: a vector per 16x16 pixels within +-32 pixels (RX_VK_FRAMEGEN_RADIUS quarter texels, default
//      8), then a 3x3 median of them;
//   3. each in-between frame at time t blends the last frame moved back along t and the new one forward along 1 - t.
// Recorded into the present's own recording, after the game's work; the display thread shows the made frames and
// then the game's, evenly over the game's recent frame time, about one frame behind the game. Loading hitches
// (frames over 100 ms apart) are not interpolated.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

#include "vk_framegen.h"
#include "vk_menu.h"
#include "vk_texture.h"
#include "vk_wheel.h"   // (relink: the item wheel's picture, laid over its backdrop)
#include <sirit/sirit.h>   // (relink: the item wheel's passes, built at run time)

#include "shaders/fg_copy_comp_spv.h"
#include "shaders/fg_flow_comp_spv.h"
#include "shaders/fg_luma_comp_spv.h"
#include "shaders/fg_median_comp_spv.h"
#include "shaders/fg_scale_comp_spv.h"
#include "shaders/fg_synth_comp_spv.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION   // (older Windows SDKs lack it: Windows 10 1803 and later take it)
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace mkwvk {
extern std::atomic<uint32_t> present_kind;   // (vk_replay.cpp: the frame log's)
namespace {

constexpr int kMaxMult = 8;   // (up to 8x: players asked; the made frames are allocated for the multiplier chosen only)

struct Img {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView view = VK_NULL_HANDLE;
};
struct OutSet {
    Img frame[kMaxMult];
    int made = 0;   // (frame[0..made) are made frames; frame[made] is the game's)
    bool busy = false;
};
struct Job {
    OutSet* set;
    int64_t at;
    uint64_t value;   // (the recording that makes its frames)
};
struct Pipe {
    VkDescriptorSetLayout set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
};

struct State {
    bool init = false, failed = false;
    uint32_t w = 0, h = 0;
    Pipe copy, luma, flow, median, synth;
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    Img frames[2], lumas[2], flows[2];
    OutSet sets[3];   // (one shown, one waiting its place on the line, one being made: the pacer never holds the game back)
    int cur = 0, next_set = 0;
    bool have_prev = false;
    int64_t last_at = 0;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> jobs;
    bool thread_started = false;
    OutSet* recorded = nullptr;   // (the set the last framegen_record filled, for framegen_queue)
    int64_t recorded_at = 0;
    int mult = 0;   // (the multiplier the sets' frames were made for)
} fg;

// Used only by present_image, whose lock serializes direct and interpolated presents.
struct ScaleState {
    bool init = false, failed = false;
    Pipe pipe;
    VkSampler linear = VK_NULL_HANDLE;
    Img cropped, scaled;
    uint32_t cropped_w = 0, cropped_h = 0, scaled_w = 0, scaled_h = 0;
} scale;

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int multiplier() { return std::clamp(opts.framegen, 1, kMaxMult); }   // (the Insert menu; 1: the game's frames alone)

/// A compute pipeline over bindings of these types (push descriptors), push constants of push bytes
bool make_pipe(Pipe& p, const uint32_t* code, size_t bytes, std::initializer_list<VkDescriptorType> types, uint32_t push,
               const VkSpecializationInfo* spec = nullptr) {
    std::vector<VkDescriptorSetLayoutBinding> b;
    for (VkDescriptorType t : types) b.push_back({(uint32_t)b.size(), t, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    li.bindingCount = (uint32_t)b.size();
    li.pBindings = b.data();
    if (vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &p.set) != VK_SUCCESS) return false;
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, std::max(push, 4u)};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &p.set;
    pi.pushConstantRangeCount = push ? 1 : 0;
    pi.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(dev.device, &pi, nullptr, &p.layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = bytes;
    mi.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev.device, &mi, nullptr, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = m;
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = spec;
    ci.layout = p.layout;
    const VkResult r = vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe);
    vkDestroyShaderModule(dev.device, m, nullptr);
    return r == VK_SUCCESS;
}

bool init() {
    if (fg.init) return !fg.failed;
    fg.init = true;
    fg.failed = true;
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const int radius = getenv("RX_VK_FRAMEGEN_RADIUS") ? std::clamp(atoi(getenv("RX_VK_FRAMEGEN_RADIUS")), 1, 16) : 8;
    const VkSpecializationMapEntry me{0, 0, 4};
    const VkSpecializationInfo spec{1, &me, 4, &radius};
    if (!make_pipe(fg.copy, FG_COPY_COMP_SPV, sizeof FG_COPY_COMP_SPV, {S, I}, 16) ||
        !make_pipe(fg.luma, FG_LUMA_COMP_SPV, sizeof FG_LUMA_COMP_SPV, {S, I}, 0) ||
        !make_pipe(fg.flow, FG_FLOW_COMP_SPV, sizeof FG_FLOW_COMP_SPV, {S, S, I}, 0, &spec) ||
        !make_pipe(fg.median, FG_MEDIAN_COMP_SPV, sizeof FG_MEDIAN_COMP_SPV, {S, I}, 0) ||
        !make_pipe(fg.synth, FG_SYNTH_COMP_SPV, sizeof FG_SYNTH_COMP_SPV, {S, S, S, I}, 4))
        return false;
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.25f;
    vkCreateSampler(dev.device, &si, nullptr, &fg.nearest);
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(dev.device, &si, nullptr, &fg.linear);
    fg.failed = false;
    logf("vk: frame interpolation ready: %dx the game's frames (made from its last two, shown evenly)", multiplier());
    return true;
}

bool init_scale() {
    if (scale.init) return !scale.failed;
    scale.init = scale.failed = true;
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    if (make_pipe(scale.pipe, FG_SCALE_COMP_SPV, sizeof FG_SCALE_COMP_SPV, {S, I}, 8)) {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.25f;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        if (vkCreateSampler(dev.device, &si, nullptr, &scale.linear) == VK_SUCCESS) {
            scale.failed = false;
            return true;
        }
    }
    if (scale.pipe.pipe) vkDestroyPipeline(dev.device, scale.pipe.pipe, nullptr);
    if (scale.pipe.layout) vkDestroyPipelineLayout(dev.device, scale.pipe.layout, nullptr);
    if (scale.pipe.set) vkDestroyDescriptorSetLayout(dev.device, scale.pipe.set, nullptr);
    scale.pipe = {};
    return false;
}

bool make_img(Img& i, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = f;
    ci.extent = {w, h, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = usage;
    if (!create_image(ci, i.image, i.alloc)) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = i.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = f;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(dev.device, &vi, nullptr, &i.view) == VK_SUCCESS;
}
void drop_img(Img& i) {
    if (i.view) vkDestroyImageView(dev.device, i.view, nullptr);
    if (i.image) destroy_image(i.image, i.alloc);
    i = {};
}

void to_general(VkCommandBuffer cmd, VkImage image) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &d);
}

bool ensure_size(VkCommandBuffer cmd, uint32_t w, uint32_t h) {
    if (fg.w == w && fg.h == h && fg.mult == multiplier()) return true;
    {   // (the display thread is done with the old sets first)
        std::unique_lock lk(fg.mu);
        fg.cv.wait(lk, [] { return fg.jobs.empty() && !fg.sets[0].busy && !fg.sets[1].busy && !fg.sets[2].busy; });
    }
    {
        std::lock_guard qk(dev.present_mu);   // (every queue idle: the present queue's too, device_wait_idle_locked)
        device_wait_idle_locked();
    }
    for (int i = 0; i < 2; i++) {
        drop_img(fg.frames[i]), drop_img(fg.lumas[i]), drop_img(fg.flows[i]);
    }
    for (OutSet& set : fg.sets)
        for (Img& t : set.frame) drop_img(t);
    fg.w = w, fg.h = h;
    const uint32_t qw = (w + 3) / 4, qh = (h + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    const VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    bool ok = true;
    for (int i = 0; i < 2; i++) {
        ok = ok && make_img(fg.frames[i], w, h, VK_FORMAT_R8G8B8A8_UNORM, rw | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        ok = ok && make_img(fg.lumas[i], qw, qh, VK_FORMAT_R16_SFLOAT, rw);
        ok = ok && make_img(fg.flows[i], cw, ch, VK_FORMAT_R16G16B16A16_SFLOAT, rw);
    }
    fg.mult = multiplier();
    for (OutSet& set : fg.sets)
        for (int k = 0; k < fg.mult; k++)
            ok = ok && make_img(set.frame[k], w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                rw | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!ok) {
        fg.failed = true;
        logf("vk: frame interpolation: its images (%ux%u) were not made: off", w, h);
        return false;
    }
    for (int i = 0; i < 2; i++) {
        to_general(cmd, fg.frames[i].image), to_general(cmd, fg.lumas[i].image), to_general(cmd, fg.flows[i].image);
    }
    for (OutSet& set : fg.sets)
        for (Img& t : set.frame)
            if (t.image) to_general(cmd, t.image);   // (only the multiplier's frames are made: a barrier on no image crashed the driver)
    fg.have_prev = false;
    return true;
}

/// One pass: pipe over w x h in 8x8 groups, the bindings in order (a sampler when one is given: a sampled image)
void pass(VkCommandBuffer cmd, const Pipe& p, std::initializer_list<std::pair<VkImageView, VkSampler>> bindings,
          uint32_t w, uint32_t h, const void* push = nullptr, uint32_t push_bytes = 0) {
    VkDescriptorImageInfo ii[8];
    VkWriteDescriptorSet ws[8];
    uint32_t n = 0;
    for (const auto& [view, sampler] : bindings) {
        ii[n] = {sampler, view, VK_IMAGE_LAYOUT_GENERAL};
        ws[n] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        ws[n].dstBinding = n;
        ws[n].descriptorCount = 1;
        ws[n].descriptorType = sampler ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ws[n].pImageInfo = &ii[n];
        n++;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, n, ws);
    if (push_bytes) vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cmd);
}

// (relink) The display thread's wait for a frame's time, on a high-resolution timer. sleep_for rests on the system timer,
// whose 1 ms (timeBeginPeriod) Windows 11 does not keep for a process whose window it takes as unseen: with the presents
// no longer blocking (a present queue of their own) the thread slept before most frames and woke up to 20 ms after their time.
void wait_until_near(int64_t ns) {
    static HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    LARGE_INTEGER d;
    d.QuadPart = -std::max<int64_t>(1, ns / 100);   // (relative, 100 ns units)
    if (timer && SetWaitableTimerEx(timer, &d, 0, nullptr, nullptr, nullptr, 0)) {
        WaitForSingleObject(timer, INFINITE);
        return;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
}

void display_main() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk display");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    int64_t interval = 16666667, last_at = 0;
    struct { int on_time, late, skipped; int64_t since; } stat{};
    int64_t next_show = 0, shown_at = 0;
    int hist[6] = {};
    for (;;) {
        Job j;
        {
            std::unique_lock lk(fg.mu);
            fg.cv.wait(lk, [] { return !fg.jobs.empty(); });
            j = fg.jobs.front();
            fg.jobs.pop_front();
        }
        if (last_at) {
            const int64_t d = j.at - last_at;
            if (d > 0 && d < 100000000) interval = (interval * 7 + std::clamp<int64_t>(d, 4000000, 50000000)) / 8;
        }
        last_at = j.at;
        if (!stat.since) stat.since = j.at;
        if (j.at - stat.since > 5000000000LL) {
            logf("vk: frame interpolation: %d frames shown on time, %d late (over 2 ms); shown apart: <7 ms %d, 7-9 %d, 9-12 %d, "
                 "12-17 %d, 17-25 %d, over 25 %d", stat.on_time, stat.late, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
            memset(hist, 0, sizeof hist);
            stat = {0, 0, 0, j.at};
        }
        // (relink) The pacer. The game hands over frames 12 to 28 ms apart around a mean of 19, and shown as they came that
        // was the stutter, whatever the frame rate. Every frame now has its place on an even line, the recent frame time
        // apart, a frame early waiting for it: never more than one game frame behind what the game handed over, and a line
        // that has fallen behind what came starts again from now. RX_VK_PACE=0: as they come.
        const bool pace = opts.pace != 0;
        const int shown = j.set->made + 1;
        const int64_t step = interval / shown;
        for (int k = 0; k <= j.set->made; k++) {
            int64_t due = pace ? next_show : 0;
            const int64_t now = now_ns();
            if (due < now) due = now;
            if (due > j.at + interval + step) due = j.at + interval + step;
            if (due - now > 1000000) wait_until_near(due - now - 1000000);
            while (now_ns() < due) SwitchToThread();
            const int64_t t = now_ns();
            present_kind.store(k < j.set->made ? 1 : 0);
            present_image(j.set->frame[k].image, fg.w, fg.h, nullptr, j.set->frame[k].view);
            menu_note_frame(k == j.set->made);
            if (shown_at) {
                const int64_t d = t - shown_at;
                hist[d < 7000000 ? 0 : d < 9000000 ? 1 : d < 12000000 ? 2 : d < 17000000 ? 3 : d < 25000000 ? 4 : 5]++;
            }
            shown_at = t;
            next_show = t + step;
            (t - due > 2000000 ? stat.late : stat.on_time)++;
        }
        std::lock_guard lk(fg.mu);
        j.set->busy = false;
        fg.cv.notify_all();
    }
}


// ---- (relink) the item wheel's backdrop and picture ----
// The game draws its HUD (hearts, the item row, prompts) onto the gameplay upscale's output, then copies that picture to the
// window texture one to one. With the Item wheel row on, the upscale's output is copied each frame as the upscale leaves it
// (the scene before the HUD): the game stops drawing its scene while its row is shown, so the last frame before the row opens
// is the one there is. While the wheel is open, the window's picture is covered with it, blurred and dimmed, and the wheel
// (vk_wheel.cpp, drawn on the CPU) laid over that. With the row off (the default) nothing here is made or recorded.
struct WheelState {
    std::mutex mu;   // (framegen_wheel_scene on the frame's recording, framegen_wheel_backdrop on the present's)
    bool init = false, failed = false;
    Pipe blur, backdrop, over;
    VkSampler sampler = VK_NULL_HANDLE;
    Img scene;    // (the upscale's output, copied: its format, w x h)
    Img blurred;  // (the backdrop's first half, blurred across: rgba8, the window crop's size)
    Img dimmed;   // (and blurred down and dimmed, then the wheel over it: rgba8, the same size)
    Img picture;  // (the wheel's picture, uploaded: rgba8, premultiplied)
    uint32_t dimmed_w = 0, dimmed_h = 0, picture_w = 0, picture_h = 0;
    uint64_t picture_version = 0;
    struct Up {
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation alloc = nullptr;
        void* map = nullptr;
        uint64_t bytes = 0;
    } ups[4];
    int up_next = 0;
    uint32_t w = 0, h = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::atomic<bool> copied{false};   // (a scene was copied since the row was turned on: an older one is never shown)
    int logged = 0;
} wheel;

/// Pictures of another size: made again, with every queue idle first (only when the window or the scene changes size)
void wheel_idle() {
    std::lock_guard qk(dev.present_mu);
    device_wait_idle_locked();
}

/// The passes' SPIR-V, built here (the build has no shader compiler; vk_coopmat.cpp builds its probes the same way). 8x8
/// groups, push constants ivec2 offset, ivec2 size. kind 0, 1: in (sampled), out (rgba8 storage), a Gaussian of
/// 2 * kBlurTaps + 1 taps offset.x pixels apart: 0 across from the scene, encoded first (pow 1/2.2: the copy to the window's);
/// 1 down from that, then dimmed by offset.y / 1000. kind 2: the wheel's picture (sampled, premultiplied) over the image
/// (storage) at offset
constexpr int kBlurTaps = 12;   // (each side of a pixel)
std::vector<uint32_t> wheel_shader(int kind) {
    using namespace Sirit;
    const bool across = kind == 0;
    Module m{0x00010000};
    m.AddCapability(spv::Capability::Shader);
    m.SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    const Id void_t = m.TypeVoid(), b1 = m.TypeBool(), f32 = m.TypeFloat(32), i32 = m.TypeSInt(32), u32 = m.TypeUInt(32);
    const Id vec3 = m.TypeVector(f32, 3), vec4 = m.TypeVector(f32, 4), ivec2 = m.TypeVector(i32, 2);
    const Id uvec3 = m.TypeVector(u32, 3), bvec2 = m.TypeVector(b1, 2);
    const Id sampled = m.TypeImage(f32, spv::Dim::Dim2D, 0, false, false, 1, spv::ImageFormat::Unknown);
    const Id sampled_t = m.TypeSampledImage(sampled);
    const Id storage = m.TypeImage(f32, spv::Dim::Dim2D, 0, false, false, 2, spv::ImageFormat::Rgba8);
    uint32_t binding = 0;
    const auto var = [&](Id type) {
        const Id v = m.AddGlobalVariable(m.TypePointer(spv::StorageClass::UniformConstant, type), spv::StorageClass::UniformConstant);
        m.Decorate(v, spv::Decoration::DescriptorSet, 0U);
        m.Decorate(v, spv::Decoration::Binding, binding++);
        return v;
    };
    const Id in = var(sampled_t), out = var(storage);
    const Id block = m.TypeStruct(ivec2, ivec2);
    m.Decorate(block, spv::Decoration::Block);
    m.MemberDecorate(block, 0U, spv::Decoration::Offset, 0U);
    m.MemberDecorate(block, 1U, spv::Decoration::Offset, 8U);
    const Id pc = m.AddGlobalVariable(m.TypePointer(spv::StorageClass::PushConstant, block), spv::StorageClass::PushConstant);
    const Id gid = m.AddGlobalVariable(m.TypePointer(spv::StorageClass::Input, uvec3), spv::StorageClass::Input);
    m.Decorate(gid, spv::Decoration::BuiltIn, spv::BuiltIn::GlobalInvocationId);
    const Id ivec2_pc = m.TypePointer(spv::StorageClass::PushConstant, ivec2);
    const Id c0 = m.Constant(i32, 0), c1 = m.Constant(i32, 1);
    const Id f0 = m.Constant(f32, 0.0f), f1 = m.Constant(f32, 1.0f);
    const Id zero3 = m.ConstantComposite(vec3, f0, f0, f0), one3 = m.ConstantComposite(vec3, f1, f1, f1);
    const Id gamma = m.Constant(f32, 1.0f / 2.2f);
    const Id gamma3 = m.ConstantComposite(vec3, gamma, gamma, gamma);

    const Id main = m.OpFunction(void_t, spv::FunctionControlMask::MaskNone, m.TypeFunction(void_t));
    m.AddLabel();
    const Id id = m.OpLoad(uvec3, gid);
    const Id g = m.OpCompositeConstruct(ivec2, m.OpBitcast(i32, m.OpCompositeExtract(u32, id, 0U)),
                                        m.OpBitcast(i32, m.OpCompositeExtract(u32, id, 1U)));
    const Id offset = m.OpLoad(ivec2, m.OpAccessChain(ivec2_pc, pc, c0));
    const Id size = m.OpLoad(ivec2, m.OpAccessChain(ivec2_pc, pc, c1));
    const Id outside = m.OpAny(b1, m.OpSGreaterThanEqual(bvec2, g, size));
    const Id leave = m.OpLabel(), body = m.OpLabel();
    m.OpSelectionMerge(body, spv::SelectionControlMask::MaskNone);
    m.OpBranchConditional(outside, leave, body);
    m.AddLabel(leave);
    m.OpReturn();
    m.AddLabel(body);
    if (kind == 2) {   // (the picture over the image: premultiplied, src + dst * (1 - src.a))
        const Id src = m.OpImageFetch(vec4, m.OpImage(sampled, m.OpLoad(sampled_t, in)), g, spv::ImageOperandsMask::Lod, c0);
        const Id at = m.OpIAdd(ivec2, g, offset);
        const Id img = m.OpLoad(storage, out);
        const Id dst = m.OpImageRead(vec4, img, at);
        const Id keep = m.OpFSub(f32, f1, m.OpCompositeExtract(f32, src, 3U));
        const Id rgb = m.OpFAdd(vec3, m.OpVectorShuffle(vec3, src, src, 0U, 1U, 2U),
                                m.OpVectorTimesScalar(vec3, m.OpVectorShuffle(vec3, dst, dst, 0U, 1U, 2U), keep));
        m.OpImageWrite(img, at, m.OpCompositeConstruct(vec4, rgb, f1));
    } else {
        const Id stride = m.OpCompositeExtract(i32, offset, 0U);
        const Id dir = across ? m.ConstantComposite(ivec2, c1, c0) : m.ConstantComposite(ivec2, c0, c1);
        const Id step = m.OpIMul(ivec2, dir, m.OpCompositeConstruct(ivec2, stride, stride));
        const Id last = m.OpISub(ivec2, size, m.ConstantComposite(ivec2, c1, c1));
        const Id origin = m.ConstantComposite(ivec2, c0, c0);
        double wsum = 0;
        for (int k = -kBlurTaps; k <= kBlurTaps; k++) wsum += std::exp(-2.0 * k * k / (kBlurTaps * kBlurTaps));
        Id sum = zero3;
        for (int k = -kBlurTaps; k <= kBlurTaps; k++) {
            const Id kk = m.Constant(i32, k);
            const Id at = m.OpSClamp(ivec2, m.OpIAdd(ivec2, g, m.OpIMul(ivec2, step, m.OpCompositeConstruct(ivec2, kk, kk))), origin, last);
            const Id v4 = m.OpImageFetch(vec4, m.OpImage(sampled, m.OpLoad(sampled_t, in)), at, spv::ImageOperandsMask::Lod, c0);
            Id c = m.OpVectorShuffle(vec3, v4, v4, 0U, 1U, 2U);
            if (across) c = m.OpPow(vec3, m.OpFClamp(vec3, c, zero3, one3), gamma3);
            const float wk = (float)(std::exp(-2.0 * k * k / (kBlurTaps * kBlurTaps)) / wsum);
            sum = m.OpFAdd(vec3, sum, m.OpVectorTimesScalar(vec3, c, m.Constant(f32, wk)));
        }
        if (!across) {
            const Id dim = m.OpFMul(f32, m.OpConvertSToF(f32, m.OpCompositeExtract(i32, offset, 1U)), m.Constant(f32, 0.001f));
            sum = m.OpVectorTimesScalar(vec3, sum, dim);
        }
        m.OpImageWrite(m.OpLoad(storage, out), g, m.OpCompositeConstruct(vec4, sum, f1));
    }
    m.OpReturn();
    m.OpFunctionEnd();
    m.AddEntryPoint(spv::ExecutionModel::GLCompute, main, "main", gid);
    m.AddExecutionMode(main, spv::ExecutionMode::LocalSize, 8U, 8U, 1U);
    return m.Assemble();
}

void drop_wheel_pipe(Pipe& p) {
    if (p.pipe) vkDestroyPipeline(dev.device, p.pipe, nullptr);
    if (p.layout) vkDestroyPipelineLayout(dev.device, p.layout, nullptr);
    if (p.set) vkDestroyDescriptorSetLayout(dev.device, p.set, nullptr);
    p = {};
}

/// The passes, at the wheel's first opening (wheel.mu held)
bool init_wheel() {
    if (wheel.init) return !wheel.failed;
    wheel.init = wheel.failed = true;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    std::vector<uint32_t> blur, backdrop, over;
    try {
        blur = wheel_shader(0), backdrop = wheel_shader(1), over = wheel_shader(2);
    } catch (const std::exception& e) {
        logf("vk: item wheel: its passes were not built (%s): the game's row shows instead", e.what());
        return false;
    }
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(dev.device, &si, nullptr, &wheel.sampler) != VK_SUCCESS) wheel.sampler = VK_NULL_HANDLE;
    if (!wheel.sampler || !make_pipe(wheel.blur, blur.data(), blur.size() * 4, {S, I}, 16) ||
        !make_pipe(wheel.backdrop, backdrop.data(), backdrop.size() * 4, {S, I}, 16) ||
        !make_pipe(wheel.over, over.data(), over.size() * 4, {S, I}, 16)) {
        drop_wheel_pipe(wheel.blur), drop_wheel_pipe(wheel.backdrop), drop_wheel_pipe(wheel.over);
        if (wheel.sampler) vkDestroySampler(dev.device, wheel.sampler, nullptr);
        wheel.sampler = VK_NULL_HANDLE;
        logf("vk: item wheel: its passes were not made: the game's row shows instead");
        return false;
    }
    wheel.failed = false;
    return true;
}

/// The wheel's picture into wheel.picture (wheel.mu held): uploaded when it changed. False when it could not be
bool wheel_upload(VkCommandBuffer cmd, const WheelPicture& pic) {
    if (wheel.picture_w != pic.w || wheel.picture_h != pic.h) {
        if (wheel.picture.image) wheel_idle(), drop_img(wheel.picture);
        wheel.picture_w = wheel.picture_h = 0, wheel.picture_version = 0;
        if (!make_img(wheel.picture, pic.w, pic.h, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
            drop_img(wheel.picture);
            return false;
        }
        to_general(cmd, wheel.picture.image);
        wheel.picture_w = pic.w, wheel.picture_h = pic.h;
    }
    if (wheel.picture_version == pic.version) return true;
    // (four upload buffers in turn: one is not written again before the three frames after it were recorded and shown)
    WheelState::Up& u = wheel.ups[wheel.up_next];
    wheel.up_next = (wheel.up_next + 1) % 4;
    const uint64_t bytes = (uint64_t)pic.w * pic.h * 4;
    if (u.bytes < bytes) {
        if (u.buf) wheel_idle(), destroy_buffer(u.buf, u.alloc);
        u = {};
        if (!create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, u.buf, u.alloc, &u.map) || !u.map) {
            if (u.buf) destroy_buffer(u.buf, u.alloc);
            u = {};
            return false;
        }
        u.bytes = bytes;
    }
    std::memcpy(u.map, pic.px, bytes);
    full_barrier(cmd);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {pic.w, pic.h, 1};
    vkCmdCopyBufferToImage(cmd, u.buf, wheel.picture.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    full_barrier(cmd);
    wheel.picture_version = pic.version;
    return true;
}

} // namespace

// The Insert menu's Frame generation is off at 0. While it was ignored, every present still copied the frame and
// reduced it to luma (framegen_record), then a second thread showed it. Open areas paid for that on a full GPU and
// gained no made frames. 2x and up is the interpolation; RX_VK_FRAMEGEN=0 keeps it off entirely.
bool framegen_wanted(int menu_value, int allow) {
    return allow && menu_value >= 2;
}

bool framegen_on() {
    static const int allow = !getenv("RX_VK_FRAMEGEN") || atoi(getenv("RX_VK_FRAMEGEN")) != 0;
    return framegen_wanted(opts.framegen, allow) && !dev.offscreen && !fg.failed;
}

VkImage framegen_scale(VkCommandBuffer cmd, VkImageView src, VkImage image, uint32_t sw, uint32_t sh, const int crop[4], uint32_t dw, uint32_t dh,
                       bool force) {
    if (!opts.upscale && !opts.sharpness && !force) return VK_NULL_HANDLE;
    if (!src || !image || !sw || !sh || !dw || !dh) return VK_NULL_HANDLE;
    if (crop && crop[2] > 0 && crop[3] > 0 &&
        (crop[0] < 0 || crop[1] < 0 || (uint32_t)crop[0] >= sw || (uint32_t)crop[1] >= sh)) return VK_NULL_HANDLE;
    if (!init_scale()) return VK_NULL_HANDLE;
    // (the game draws into part of its window textures, 2048 x 1152 of 2560 x 1440 here: the filter was given the whole texture, and the
    // picture came out in the top left corner of the window with black around it. The crop first, then the filter)
    if (crop && crop[2] > 0 && crop[3] > 0 && (crop[0] || crop[1] || (uint32_t)crop[2] < sw || (uint32_t)crop[3] < sh)) {
        const uint32_t cw = std::min((uint32_t)crop[2], sw - (uint32_t)crop[0]), ch = std::min((uint32_t)crop[3], sh - (uint32_t)crop[1]);
        if (scale.cropped_w != cw || scale.cropped_h != ch) {
            {
                std::lock_guard qk(dev.present_mu);
                device_wait_idle_locked();
            }
            drop_img(scale.cropped);
            scale.cropped_w = scale.cropped_h = 0;
            if (!make_img(scale.cropped, cw, ch, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
                drop_img(scale.cropped);
                return VK_NULL_HANDLE;
            }
            scale.cropped_w = cw, scale.cropped_h = ch;
            to_general(cmd, scale.cropped.image);
        }
        VkImageBlit b{};
        b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[0] = {crop[0], crop[1], 0};
        b.srcOffsets[1] = {crop[0] + (int)cw, crop[1] + (int)ch, 1};
        b.dstOffsets[1] = {(int)cw, (int)ch, 1};
        vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, scale.cropped.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
        full_barrier(cmd);
        src = scale.cropped.view;
    }
    if (scale.scaled_w != dw || scale.scaled_h != dh) {
        {
            std::lock_guard qk(dev.present_mu);   // (the queue is idle only while nothing else submits to it)
            device_wait_idle_locked();
        }
        drop_img(scale.scaled);
        scale.scaled_w = scale.scaled_h = 0;
        if (!make_img(scale.scaled, dw, dh, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            drop_img(scale.scaled);
            return VK_NULL_HANDLE;
        }
        scale.scaled_w = dw, scale.scaled_h = dh;
        to_general(cmd, scale.scaled.image);
    }
    const struct { int mode; float sharp; } pc{opts.upscale, opts.sharpness / 100.0f};
    pass(cmd, scale.pipe, {{src, scale.linear}, {scale.scaled.view, VK_NULL_HANDLE}}, dw, dh, &pc, sizeof pc);
    static const bool report = getenv("RX_VK_UPSCALE_LOG") != nullptr;
    static bool reported = false;
    if (report && !reported) {
        reported = true;
        logf("vk: upscale dispatched: mode %d, sharpness %d, source %ux%u, crop request %d,%d %dx%d, destination %ux%u",
             opts.upscale, opts.sharpness, sw, sh, crop ? crop[0] : 0, crop ? crop[1] : 0,
             crop ? crop[2] : (int)sw, crop ? crop[3] : (int)sh, dw, dh);
    }
    return scale.scaled.image;
}

VkImage framegen_made_image(uint32_t* w, uint32_t* h) {
    if (!fg.recorded || !fg.recorded->made) return VK_NULL_HANDLE;
    *w = fg.w, *h = fg.h;
    return fg.recorded->frame[0].image;
}

bool framegen_record(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, const int crop[4]) {
    int x0 = 0, y0 = 0, w = (int)sw, h = (int)sh;
    if (crop && crop[2] > 0 && crop[3] > 0) x0 = crop[0], y0 = crop[1], w = crop[2], h = crop[3];
    w = std::min(w, (int)sw - x0), h = std::min(h, (int)sh - y0);
    if (!src || w <= 0 || h <= 0 || !init() || !ensure_size(cmd, (uint32_t)w, (uint32_t)h)) return false;
    if (!fg.thread_started) {
        fg.thread_started = true;
        std::thread(display_main).detach();
    }
    const int64_t now = now_ns();
    const bool gap = fg.last_at && now - fg.last_at > 100000000;   // (a loading hitch: nothing to blend)
    fg.last_at = now;
    OutSet* set;
    {
        std::unique_lock lk(fg.mu);
        fg.cv.wait(lk, [] { return !fg.sets[fg.next_set].busy; });
        set = &fg.sets[fg.next_set];
        set->busy = true;
        fg.next_set = (fg.next_set + 1) % 3;
    }
    const int cur = fg.cur, prev = cur ^ 1, m = multiplier();
    const bool blend = fg.have_prev && !gap;
    const uint32_t uw = (uint32_t)w, uh = (uint32_t)h;
    const uint32_t qw = (uw + 3) / 4, qh = (uh + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    full_barrier(cmd);
    const int32_t pc[4] = {x0, y0, w, h};
    pass(cmd, fg.copy, {{src, fg.nearest}, {fg.frames[cur].view, VK_NULL_HANDLE}}, uw, uh, pc, sizeof pc);
    pass(cmd, fg.luma, {{fg.frames[cur].view, fg.nearest}, {fg.lumas[cur].view, VK_NULL_HANDLE}}, qw, qh);
    set->made = 0;
    if (blend && m > 1) {
        pass(cmd, fg.flow, {{fg.lumas[prev].view, fg.nearest}, {fg.lumas[cur].view, fg.nearest},
                            {fg.flows[0].view, VK_NULL_HANDLE}}, cw, ch);
        pass(cmd, fg.median, {{fg.flows[0].view, fg.nearest}, {fg.flows[1].view, VK_NULL_HANDLE}}, cw, ch);
        for (int k = 1; k < m; k++) {
            const float t = (float)k / (float)m;
            pass(cmd, fg.synth, {{fg.frames[prev].view, fg.linear}, {fg.frames[cur].view, fg.linear},
                                 {fg.flows[1].view, fg.nearest}, {set->frame[k - 1].view, VK_NULL_HANDLE}},
                 uw, uh, &t, 4);
        }
        set->made = m - 1;
    }
    const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                        {uw, uh, 1}};
    vkCmdCopyImage(cmd, fg.frames[cur].image, VK_IMAGE_LAYOUT_GENERAL, set->frame[set->made].image,
                   VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    full_barrier(cmd);
    fg.have_prev = true;
    fg.cur = prev;
    fg.recorded = set;
    fg.recorded_at = now;
    return true;
}

void framegen_queue(uint64_t value) {
    if (!fg.recorded) return;
    std::lock_guard lk(fg.mu);
    fg.jobs.push_back({fg.recorded, fg.recorded_at, value});
    fg.recorded = nullptr;
    fg.cv.notify_all();
}

/// (relink) The Gameplay tab's Item wheel row is on (HOS_ITEM_WHEEL=0 or 1 over it, as host/oot_wheel.c reads
/// it). Off: nothing below records or makes anything
bool framegen_wheel_on() {
    static const int forced = getenv("HOS_ITEM_WHEEL") ? atoi(getenv("HOS_ITEM_WHEEL")) : -1;
    const bool on = forced >= 0 ? forced != 0 : opts.item_wheel != 0;
    if (!on && wheel.copied.load(std::memory_order_relaxed))   // (turned on again later: the frames before are not the scene)
        wheel.copied.store(false, std::memory_order_relaxed);
    return on;
}

void framegen_wheel_scene(VkCommandBuffer cmd, VkImage image, VkFormat format, uint32_t w, uint32_t h, uint32_t level) {
    if (!cmd || !image || !w || !h || !framegen_wheel_on()) return;
    std::lock_guard lk(wheel.mu);
    if (!init_wheel()) return;
    if (wheel.w != w || wheel.h != h || wheel.format != format) {   // (another scene size: a new picture)
        if (wheel.scene.image) wheel_idle(), drop_img(wheel.scene);
        wheel.w = wheel.h = 0;
        if (!make_img(wheel.scene, w, h, format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
            drop_img(wheel.scene);
            if (wheel.logged++ < 4) logf("vk: item wheel: the scene's copy (%ux%u, format %d) was not made", w, h, (int)format);
            return;
        }
        to_general(cmd, wheel.scene.image);
        wheel.w = w, wheel.h = h, wheel.format = format;
    }
    full_barrier(cmd);
    const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, wheel.scene.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    full_barrier(cmd);
    wheel.copied.store(true, std::memory_order_relaxed);
}

// the backdrop's two images for a window picture of w x h (wheel.mu held): made while the wheel is closed too, so its first
// opening does not wait for them
static bool wheel_backdrop_images(VkCommandBuffer cmd, int w, int h) {
    if (wheel.dimmed_w == (uint32_t)w && wheel.dimmed_h == (uint32_t)h) return true;
    if (wheel.dimmed.image || wheel.blurred.image) wheel_idle(), drop_img(wheel.dimmed), drop_img(wheel.blurred);
    wheel.dimmed_w = wheel.dimmed_h = 0;
    if (!make_img(wheel.dimmed, w, h, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
        !make_img(wheel.blurred, w, h, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
        drop_img(wheel.dimmed), drop_img(wheel.blurred);
        if (wheel.logged++ < 4) logf("vk: item wheel: its backdrop (%dx%d) was not made", w, h);
        return false;
    }
    to_general(cmd, wheel.dimmed.image), to_general(cmd, wheel.blurred.image);
    wheel.dimmed_w = w, wheel.dimmed_h = h;
    logf("vk: item wheel: the backdrop is the scene before the HUD, blurred and dimmed (%dx%d)", w, h);
    return true;
}

bool framegen_wheel_backdrop(VkCommandBuffer cmd, VkImage dst, VkFormat format, uint32_t sw, uint32_t sh, const int crop[4]) {
    if (!cmd || !dst || !sw || !sh || !framegen_wheel_on()) return false;
    int x0 = 0, y0 = 0, w = (int)sw, h = (int)sh;
    if (crop && crop[2] > 0 && crop[3] > 0) x0 = crop[0], y0 = crop[1], w = crop[2], h = crop[3];
    w = std::min(w, (int)sw - x0), h = std::min(h, (int)sh - y0);
    if (w <= 0 || h <= 0) return false;
    WheelPicture pic;
    if (!wheel_picture((uint32_t)w, (uint32_t)h, &pic)) {   // (closed: its backdrop's images made ahead, once)
        std::lock_guard lk(wheel.mu);
        if (!wheel.failed && wheel.init && wheel.copied.load(std::memory_order_relaxed) && format == VK_FORMAT_R8G8B8A8_UNORM &&
            wheel.w == (uint32_t)w && wheel.h == (uint32_t)h)
            wheel_backdrop_images(cmd, w, h);
        return false;
    }
    std::lock_guard lk(wheel.mu);
    if (wheel.failed || !wheel.init || !wheel.copied.load(std::memory_order_relaxed)) {
        if (wheel.logged++ < 4) logf("vk: item wheel: no scene copied yet: not drawn");
        return false;
    }
    // (only the window picture the game copies its scene into one to one: RGBA8 of the scene's size)
    if (format != VK_FORMAT_R8G8B8A8_UNORM || !wheel.scene.image || wheel.w != (uint32_t)w || wheel.h != (uint32_t)h) {
        if (wheel.logged++ < 4)
            logf("vk: item wheel: not drawn (window picture %dx%d format %d, scene %ux%u)", w, h, (int)format, wheel.w, wheel.h);
        return false;
    }
    if (!wheel_backdrop_images(cmd, w, h)) return false;
    if (!wheel_upload(cmd, pic)) {
        if (wheel.logged++ < 4) logf("vk: item wheel: its picture (%ux%u) was not uploaded", pic.w, pic.h);
        return false;
    }
    full_barrier(cmd);
    // (the blur's taps a 400th of the height apart: 12 each side reach about 3% of the height at any size. The opening eases it
    // in over 160 ms, the wheel's own spread: the taps from 0 apart (no blur) and the dimming from none to 0.55)
    static int64_t last_at = 0, opened_at = 0;
    const int64_t t_now = now_ns();
    if (!last_at || t_now - last_at > 150000000) opened_at = t_now;
    last_at = t_now;
    const float t = std::min(1.0f, (float)(t_now - opened_at) / 160e6f), e = 1 - (1 - t) * (1 - t) * (1 - t);
    const int32_t pc[4] = {(int32_t)std::lround(std::max(1, h / 400) * e), (int32_t)std::lround(1000 - 450 * e), w, h};
    pass(cmd, wheel.blur, {{wheel.scene.view, wheel.sampler}, {wheel.blurred.view, VK_NULL_HANDLE}}, w, h, pc, 16);
    full_barrier(cmd);
    pass(cmd, wheel.backdrop, {{wheel.blurred.view, wheel.sampler}, {wheel.dimmed.view, VK_NULL_HANDLE}}, w, h, pc, 16);
    full_barrier(cmd);
    const int32_t po[4] = {pic.x, pic.y, (int32_t)pic.w, (int32_t)pic.h};
    pass(cmd, wheel.over, {{wheel.picture.view, wheel.sampler}, {wheel.dimmed.view, VK_NULL_HANDLE}}, pic.w, pic.h, po, 16);
    full_barrier(cmd);
    const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {x0, y0, 0},
                        {(uint32_t)w, (uint32_t)h, 1}};
    vkCmdCopyImage(cmd, wheel.dimmed.image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    full_barrier(cmd);
    return true;
}

} // namespace mkwvk
