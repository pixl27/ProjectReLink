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
#include <utility>   // (std::exchange: framegen_idle)

#include <cmath>

#include "vk_framegen.h"
#ifdef TRACY_ENABLE
#include "../../host/hos_tracy.h"   // (HOOKSHOT_TRACY: the profiler's zones; nothing in any other build)
#endif
#include "framegen_idle.h"   // (hookshot: when its images are let go, framegen_idle)
#include "vk_antilag.h"   // (AMD Anti-Lag 2: which game frame a set shows, antilag_frame / antilag_present_frame)
#include "vk_dlss.h"   // (NVIDIA DLSS Frame Generation through NGX: dlss_fg_frames)
#include "vk_reflex.h"   // (NVIDIA Reflex: which game frame a set shows, reflex_frame / reflex_present_frame)
#include "vk_fsr3_inputs.h"   // (AMD FSR 3.1 frame interpolation's depth and motion)
#include "vk_fsr3fi.h"
#include "vk_menu.h"
#include "vk_present_rect.h"
#include "crt_policy.h"
#include "vk_gprof.h"
#include "vk_texture.h"
#include "menu_fade.h"
#include "vk_wheel.h"   // (hookshot: the item wheel's picture, laid over its backdrop)
#include <sirit/sirit.h>   // (hookshot: the item wheel's passes, built at run time)

#include "vk_builtin_spv.h"   // (RX_VK_SHADER_DIR)
#include "shaders/fg_copy_comp_spv.h"
#include "shaders/fg_flow_comp_spv.h"
#include "shaders/fg_hud_comp_spv.h"   // (AMD FSR 3.1's interpolation source with the HUD marked: hud_source)
#include "shaders/fg_hud_composite_comp_spv.h"   // (the HUD over FSR 3.1's made frames: fsr3fi_frames)
#include "shaders/fg_hud_layer_comp_spv.h"   // (how see-through the HUD is: hud_source)
#include "shaders/fg_luma_comp_spv.h"
#include "shaders/fg_median_comp_spv.h"
#include "shaders/fg_scale_comp_spv.h"
#include "shaders/fg_scale_fp16_comp_spv.h"
#include "shaders/fg_scale_pq_comp_spv.h"   // (fg_scale.comp with FG_SCALE_PQ: an RGB10A2 destination)
#include "shaders/fg_synth_comp_spv.h"
#include "shaders/hdr_encode_comp_spv.h"
#include "shaders/crt_downsample_comp_spv.h"
#include "shaders/crt_lottes_comp_spv.h"
#include "shaders/crt_hyllian_comp_spv.h"
#include "shaders/crt_zfast_comp_spv.h"
#ifndef HOOKSHOT_ENV_H   // (env_on: the switches' one rule)
#include "hookshot_env.h"
#endif
#include "vk_draw.h"   // (extern_guarded_shader_module, extern_guarded_compute)
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION   // (older Windows SDKs lack it: Windows 10 1803 and later take it)
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace mkwvk {
int display_refresh_mhz();   // (vk_main.cpp: the window's display, mHz; outside the unnamed namespace, where it is defined)
extern std::atomic<uint32_t> present_kind;   // (vk_replay.cpp: the frame log's)
extern std::mutex present_image_mu;   // (vk_replay.cpp: one present, or remake of the chain from it, at a time)
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
    // (Auto: frame[0..shown) each at its own time, due[k] (steady clock), game[k] the game's own frame; shown < 0: the
    // fixed multipliers' frames, made + 1 of them. A fixed multiplier's set with made frames has its times on the game's
    // own timeline too, by_time (framegen_record, fixed_dues); display_main shows it at them (fixed_on_timeline; under
    // FIFO one a refresh at most, fifo_slot))
    int shown = -1;
    int64_t due[kMaxMult] = {};
    bool game[kMaxMult] = {};
    bool by_time = false;
    // (by_time: the game's frame time it was made at (smoothed), the time since the game frame before (span), and the GPU's
    // delay its times have (fg.late_ns then): for pace_delay on the display thread)
    int64_t frame_ns = 0, span_ns = 0, extra_ns = 0;
    uint64_t reflex_id = 0;   // (NVIDIA Reflex: the game frame it was made at, 0 none: its game frame is presented with it)
    uint64_t antilag_id = 0;  // (AMD Anti-Lag 2: the same, its game frame's present stage)
};
struct Job {
    OutSet* set;
    int64_t at;
    uint64_t value;   // (the recording that makes its frames)
    int64_t queued;   // (when framegen_queue handed it to the display thread)
};
struct Pipe {
    VkDescriptorSetLayout set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
};

struct State {
    bool init = false;
    std::atomic<bool> failed{false};   // also read by Reflex's game-thread eligibility check
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
    int64_t game_ns = 0;     // (Auto: the game's frame time, smoothed: the delay its frames are shown at)
    int64_t slot_next = 0;   // (Auto: the display's next refresh to be given a frame, on the game's timeline)
    // (a fixed multiplier's sets on the game's timeline: the delay past their lead (pace_lead) that the GPU asks for,
    // pace_delay. The display thread writes it, framegen_record reads it)
    std::atomic<int64_t> late_ns{0};
} fg;

// Used only by present_image, whose lock serializes direct and interpolated presents.
struct ScaleState {
    bool init = false, failed = false;
    Pipe pipe;
    Pipe pipe_pq;           // (fg_scale_pq, for the game's own HDR10 picture: framegen_scale's pq; made at its first use)
    bool pq_tried = false;
    VkSampler linear = VK_NULL_HANDLE;
    Img cropped, scaled, small;   // (small: the picture at 1/8, for ultrawide's blurred menu sides)
    uint32_t cropped_w = 0, cropped_h = 0, scaled_w = 0, scaled_h = 0, small_w = 0, small_h = 0;
    bool pq = false;        // (the images above are RGB10A2, for pq, else RGBA8)
    VkImageView output_view = VK_NULL_HANDLE;   // the final scale/CRT image, also read by HDR expansion
} scale;

struct HdrState {   // (hookshot) the HDR10 pass (framegen_hdr)
    bool init = false, failed = false;
    Pipe pipe;
    Img out;
    uint32_t w = 0, h = 0;
} hdr;

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// (hookshot) Frame generation's Auto (the menu's 1): a frame for each refresh of the display, no more. Each game frame
// is given the display's refreshes that fall between it and the game's frame before (on the game's own timeline, a
// refresh period apart), and each of them a frame made at its place between the two: 45 fps on a 60 Hz display makes one
// frame for every three of the game's (3 of 4 shown are made, at 3/4, 1/2, 1/4), on 120 Hz two or three a frame; at
// 60 fps on 60 Hz none (a refresh near a game frame shows it), and faster than the display a game frame may be given no refresh and is not shown. Shown one
// game frame time later, each at its refresh's time. The sets' frames are made once for the most a game frame may be
// given (the refresh over 30 fps, 2 to 4). An unknown refresh counts as 60 Hz.
int auto_cap() {
    const int hz = (display_refresh_mhz() + 500) / 1000;
    return std::clamp(hz > 0 ? (hz + 29) / 30 : 2, 2, 4);
}
int multiplier() {   // (the Insert menu's, as it runs: framegen_now; 1 Auto (its most); the game's frames alone at 0)
    const int framegen = framegen_now();
    return framegen == 1 ? auto_cap() : std::clamp(framegen, 1, kMaxMult);
}
// (hookshot) A fixed multiplier makes as many frames a game frame as the display can show, m at most, so the game keeps its
// own rate. Each made frame is a full-size pass on the game's queue and a present of its own, and a frame past the display's
// refresh never reaches the screen: 8x at 60 fps on a 180 Hz display made 7 frames a game frame for 3 refreshes, the display
// thread got ~220 presents a second out, the game waited on its sets and fell from 60 to ~28 fps (3440x1440, 2026-11-11).
// Now: the display's refreshes in the game's frame time (60 fps on 180 Hz: 3, two of them made), 1 (the game's frame alone)
// at the refresh. An unknown refresh or frame time: m. RX_VK_FRAMEGEN_UNCAPPED=1: m always. (tests/test_framegen_auto.py)
int fixed_frames(int m, int hz, int64_t game_ns) {
    if (m <= 1 || hz <= 0 || game_ns <= 0) return std::max(m, 1);
    const int fit = (int)((double)hz * (double)game_ns / 1e9 + 0.2);   // (+0.2: a game a little over 60 fps on 180 Hz keeps its 3)
    return std::clamp(fit, 1, m);
}
/// Auto: the display's refreshes in the game frame's interval (last, now], next the first not given a frame yet (moved
/// past them). Each one's place between the two game frames in t (1: this one) and its time in at; most of them at the
/// most, those past it skipped. A timeline outside the interval (the first frame, a hitch) starts again at now.
int auto_slots(int64_t last, int64_t now, int64_t period, int64_t& next, float* t, int64_t* at, int most) {
    if (now <= last || period <= 0) return 0;
    if (next <= last || next > now + period) next = now;
    int n = 0;
    for (; next <= now; next += period)
        if (n < most) t[n] = (float)(next - last) / (float)(now - last), at[n++] = next;
    // A refresh within a quarter period of the game's frame shows that frame itself: at the game's rate the frames come
    // a little early or late (jitter), and without this nearly every refresh got a made frame at t near 1 and some game
    // frames none (their neighbour two). The timeline is pulled an eighth of the way to the frame, so that a game at the
    // display's rate (or a whole fraction of it) stays on it while the two clocks drift apart, and other rates hardly
    // change how many frames the display is given
    const int64_t snap = period / 4;   // (not "near": windows.h defines it empty)
    if (next - now < snap) {
        n = std::min(n + 1, most);
        t[n - 1] = 1.0f, at[n - 1] = now, next += period + (now - next) / 8;
    } else if (n > 0 && now - (next - period) < snap) {
        t[n - 1] = 1.0f, at[n - 1] = now, next += (now - (next - period)) / 8;
    }
    return n;
}

/// (hookshot) A fixed multiplier's set on the game's own timeline: of its made + 1 frames, frame k is shown at its place
/// between the game's frame before (last) and this one (now), (k + 1) / (made + 1) of the way, delay later (pace_lead and
/// pace_delay's). Shown evenly apart, the smoothed frame time over made + 1, the motion sped up and slowed down by as much
/// as the game's frame times vary: 15.5 to 22 ms at ~55 fps (RTX 5090, 5120x2160, DLSS Frame Gen 3x, V-Sync Off) gave
/// every set the same ~6.3 ms a frame, a stutter with the frames evenly apart. On the game's timeline each frame moves as
/// far as its time.
void fixed_dues(int64_t last, int64_t now, int made, int64_t delay, int64_t* due) {
    for (int k = 0; k <= made; k++) due[k] = last + (now - last) * (k + 1) / (made + 1) + delay;
}

/// (hookshot) A fixed multiplier's set's delay before the GPU's (pace_delay): its made frames come from the game frame that
/// came at now, so none of them can be shown before it. Delayed by made / (made + 1) of the game's frame time (frame), the
/// set's first frame is due at now when the game frame came a frame time after the one before (earlier after a longer
/// one), and its game frame that much later, where the next set's first one follows
int64_t pace_lead(int64_t frame, int made) { return made > 0 ? frame * made / (made + 1) : 0; }

// doc: docs/design.md#13-frame-generations-presents-as-rules (rule 3: the GPU's delay from sets taken as they came)
/// (hookshot) The GPU's delay that a fixed multiplier's sets are shown at past their lead (fg.late_ns), from what the last
/// set needed: its own delay (extra_ns) and how late (negative: early) its first frame could be shown against its time,
/// and a millisecond to spare. The set's frames are done with the game frame's GPU work, so shown at times that did not
/// wait for it, they came past them and went out at once. Up by half of what is missing, down by a sixteenth of what is
/// left over; at most a game frame (the three sets: one shown, the next taken, one being made). A hitch (a need past four
/// game frames) is not the GPU's pace and leaves it as it was
int64_t pace_delay(int64_t extra, int64_t need, int64_t frame) {
    if (frame <= 0 || need > 4 * frame) return extra;
    extra += need > extra ? (need - extra) / 2 : (need - extra) / 16;
    return std::clamp<int64_t>(extra, 0, frame);
}

// doc: docs/design.md#13-frame-generations-presents-as-rules (rule 2: sets on the game's timeline, any mode)
/// (hookshot) Whether display_main shows a fixed multiplier's set at its times on the game's timeline (by_time): whatever
/// Frame pacing says and whatever the swapchain's mode. Where it does not wait for a refresh (immediate: V-Sync Off;
/// mailbox), shown as they came, a set's frames went out within a millisecond, the made frames hardly on the screen, and
/// DLSS Frame Gen 3x at V-Sync Off looked like the game's own frame rate (RTX 5090, 165 Hz with G-Sync). Under FIFO (V-Sync
/// On, and Auto with DLSS Frame Gen) a VRR display shows each present when it comes, one a refresh at most: shown as they
/// came, a set's frames were a refresh apart and its last one stayed until the next set came (3x at 60 fps on a 240 Hz
/// G-Sync display: 4.2, 4.2 and 8.3 ms), and with Frame pacing On a set that came late went out at once and the frames
/// after it a step apart from there (RTX 5070 Ti: 3x uneven, Auto even). RX_VK_FG_TIMELINE=0 (env_off): never, as before
bool fixed_on_timeline(bool by_time, bool env_off) { return by_time && !env_off; }

/// (hookshot) Under FIFO, when a frame of a set on the game's timeline is presented: at its due time, but not sooner than a
/// refresh (refresh; 0: not known) after the time the frame before was given (slot; 0: none yet). The screen shows one a
/// refresh at most, and frames presented sooner (the game's rate times the multiplier at or past the display's; a set's
/// frames due together after it came late) waited in the swapchain, the next set's copies waiting for its images. The
/// times a refresh apart from the last one given, as Auto's slots, not from when the thread presented it: a wake up to a
/// millisecond late is not added to every frame (display_main makes a present later than that the next one's slot). A
/// slot past by more than half a refresh (the thread held up) gives now: the frame goes at once, as display_main does with
/// any time already past
int64_t fifo_slot(int64_t due, int64_t slot, int64_t refresh, int64_t now) {
    if (slot <= 0 || refresh <= 0) return due;
    const int64_t next = slot + refresh;
    return std::max(due, next < now - refresh / 2 ? now : next);
}

/// (hookshot) The set's frame shown k-th (0-based) when `shown` of its `frames` are shown: the frames at the even steps
/// k + 1 of shown across the set, rounded, so the frames left out are spread over it and the picture moves in even steps
/// (6x with 3 shown: the 2nd, 4th and 6th; left out at the front the picture moved 4/6, 1/6 and 1/6 of a game frame). The
/// last is always the set's last frame, the game's; all shown: frame k
int fifo_pick(int frames, int shown, int k) {
    return ((k + 1) * frames * 2 + shown) / (2 * shown) - 1;
}

// doc: docs/design.md#13-frame-generations-presents-as-rules (rule 6: under FIFO the made frames do not queue)
/// (hookshot) FIFO's catch-up (display_main's take): how many made frames a fixed multiplier's set on the game's timeline
/// leaves out, spread over it (fifo_pick). The frames shown go a refresh apart at the least (fifo_slot), the first a
/// refresh after each of the set before's frames still to show (left, the first of them at slot; 0: all shown, slot the
/// last one's): the most kept so that the set's game frame, its last (due[made]), is shown no more than a refresh after
/// its own time. None while the game's frames times the multiplier stay under the refresh; past it as many as the
/// refresh has no room for (3x at 60 game frames a second on 165 Hz: one every few sets), unless something holds the
/// game back (RX_VK_FG_FIFO_CAP: vk_reflex.cpp fifo_min_interval_us). At most made - 1: a made frame and the game's
/// stay, as one frame alone is not presented ahead and would leave the timeline (present_ahead_wanted); so 2x (one made
/// frame) keeps it. No slot or refresh yet: 0. Without it the frames queue: 6x at 165 Hz held the game to 27.5 frames a
/// second and showed each game frame 138 ms after its hand-over
int fifo_catch_up(int made, const int64_t* due, int64_t slot, int left, int64_t refresh, int64_t now) {
    if (made < 2 || slot <= 0 || refresh <= 0) return 0;
    int64_t next = slot + std::max(left, 1) * refresh;
    if (next < now - refresh / 2) next = now;   // (as fifo_slot: a line fallen behind starts again from now)
    for (int shown = made + 1; shown > 2; shown--) {
        // (the frames picked, each at its slot as fifo_slot gives them: the first at next or its own time)
        int64_t at = std::max(due[fifo_pick(made + 1, shown, 0)], next);
        for (int k = 1; k < shown; k++) at = std::max(due[fifo_pick(made + 1, shown, k)], at + refresh);
        if (at - due[made] <= refresh) return made + 1 - shown;
    }
    return made - 1;
}

void drop_pipe(Pipe& p) {
    if (p.pipe) vkDestroyPipeline(dev.device, p.pipe, nullptr);
    if (p.layout) vkDestroyPipelineLayout(dev.device, p.layout, nullptr);
    if (p.set) vkDestroyDescriptorSetLayout(dev.device, p.set, nullptr);
    p = {};
}

/// A compute pipeline over bindings of these types (push descriptors), push constants of push bytes
bool make_pipe(Pipe& p, const uint32_t* code, size_t bytes, std::initializer_list<VkDescriptorType> types, uint32_t push,
               const VkSpecializationInfo* spec = nullptr) {
    Pipe made;
    const auto undo = [&] { drop_pipe(made); return false; };
    std::vector<VkDescriptorSetLayoutBinding> b;
    for (VkDescriptorType t : types) b.push_back({(uint32_t)b.size(), t, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    li.bindingCount = (uint32_t)b.size();
    li.pBindings = b.data();
    if (vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &made.set) != VK_SUCCESS) return false;
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, std::max(push, 4u)};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &made.set;
    pi.pushConstantRangeCount = push ? 1 : 0;
    pi.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(dev.device, &pi, nullptr, &made.layout) != VK_SUCCESS) {
        made.layout = VK_NULL_HANDLE;
        return undo();
    }
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = bytes;
    mi.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    VkResult r;
    {
        IntelDriverGate gate;
        if (extern_guarded_shader_module(&mi, &m) != VK_SUCCESS) return undo();
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = m;
        ci.stage.pName = "main";
        ci.stage.pSpecializationInfo = spec;
        ci.layout = made.layout;
        r = extern_guarded_compute(VK_NULL_HANDLE, &ci, &made.pipe);
    }
    vkDestroyShaderModule(dev.device, m, nullptr);
    if (r != VK_SUCCESS) return undo();
    p = made;
    return true;
}

// The optional interpolation is disabled after setup fails: retain none of its unused setup objects.
void drop_setup() {
    for (Pipe* p : {&fg.copy, &fg.luma, &fg.flow, &fg.median, &fg.synth}) drop_pipe(*p);
    if (fg.nearest) vkDestroySampler(dev.device, fg.nearest, nullptr);
    if (fg.linear) vkDestroySampler(dev.device, fg.linear, nullptr);
    fg.nearest = fg.linear = VK_NULL_HANDLE;
}

bool init() {
    if (fg.init) return !fg.failed;
    fg.init = true;
    fg.failed = true;
    const auto undo = [] { drop_setup(); return false; };
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const int radius = getenv("RX_VK_FRAMEGEN_RADIUS") ? std::clamp(atoi(getenv("RX_VK_FRAMEGEN_RADIUS")), 1, 16) : 8;
    const VkSpecializationMapEntry me{0, 0, 4};
    const VkSpecializationInfo spec{1, &me, 4, &radius};
    if (!make_pipe(fg.copy, BUILTIN_SPV_ARGS(FG_COPY_COMP_SPV), {S, I}, 28) ||
        !make_pipe(fg.luma, BUILTIN_SPV_ARGS(FG_LUMA_COMP_SPV), {S, I}, 0) ||
        !make_pipe(fg.flow, BUILTIN_SPV_ARGS(FG_FLOW_COMP_SPV), {S, S, I}, 0, &spec) ||
        !make_pipe(fg.median, BUILTIN_SPV_ARGS(FG_MEDIAN_COMP_SPV), {S, I}, 0) ||
        !make_pipe(fg.synth, BUILTIN_SPV_ARGS(FG_SYNTH_COMP_SPV), {S, S, S, I}, 4))
        return undo();
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.25f;
    if (vkCreateSampler(dev.device, &si, nullptr, &fg.nearest) != VK_SUCCESS) {
        fg.nearest = VK_NULL_HANDLE;
        return undo();
    }
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    if (vkCreateSampler(dev.device, &si, nullptr, &fg.linear) != VK_SUCCESS) {
        fg.linear = VK_NULL_HANDLE;
        return undo();
    }
    fg.failed = false;
    if (opts.framegen == 1)
        logf("vk: frame interpolation ready: Auto, a frame for each of the display's refreshes (%d Hz; up to %d a game frame)",
             (display_refresh_mhz() + 500) / 1000, multiplier());
    else
        logf("vk: frame interpolation ready: %dx the game's frames (made from its last two, shown evenly)", multiplier());
    return true;
}

bool init_scale() {
    if (scale.init) return !scale.failed;
    scale.init = scale.failed = true;
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const bool fp16 = dev.props.vendorID == 0x1002 && dev.vk12.shaderFloat16 &&
        !env_on("RX_VK_NO_F16") && env_on("RX_VK_SHARPEN_FP16") &&
        make_pipe(scale.pipe, BUILTIN_SPV_ARGS(FG_SCALE_FP16_COMP_SPV), {S, I, S}, 24);
    if (fp16 || make_pipe(scale.pipe, BUILTIN_SPV_ARGS(FG_SCALE_COMP_SPV), {S, I, S}, 24)) {
        if (fp16) logf("vk: SDR window sharpening uses FP16 color taps");
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.25f;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        if (vkCreateSampler(dev.device, &si, nullptr, &scale.linear) == VK_SUCCESS) {
            scale.failed = false;
            return true;
        }
        scale.linear = VK_NULL_HANDLE;
    }
    drop_pipe(scale.pipe);
    return false;
}

bool make_img(Img& i, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage) {
    Img made;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = f;
    ci.extent = {w, h, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = usage;
    if (!create_image(ci, made.image, made.alloc)) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = made.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = f;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(dev.device, &vi, nullptr, &made.view) != VK_SUCCESS) {
        destroy_image(made.image, made.alloc);
        return false;
    }
    i = made;
    return true;
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

// Only after the display jobs and device have drained (ensure_size), or before any frame used these images.
void drop_size() {
    for (int i = 0; i < 2; i++) {
        drop_img(fg.frames[i]), drop_img(fg.lumas[i]), drop_img(fg.flows[i]);
    }
    for (OutSet& set : fg.sets)
        for (Img& t : set.frame) drop_img(t);
    fg.w = fg.h = 0;
    fg.mult = 0;
    fg.have_prev = false;
    fg.recorded = nullptr;
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
    drop_size();
    const uint32_t qw = (w + 3) / 4, qh = (h + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    const VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    bool ok = true;
    for (int i = 0; i < 2; i++) {
        ok = ok && make_img(fg.frames[i], w, h, VK_FORMAT_R8G8B8A8_UNORM, rw | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        ok = ok && make_img(fg.lumas[i], qw, qh, VK_FORMAT_R16_SFLOAT, rw);
        ok = ok && make_img(fg.flows[i], cw, ch, VK_FORMAT_R16G16B16A16_SFLOAT, rw);
    }
    const int mult = multiplier();
    for (OutSet& set : fg.sets)
        for (int k = 0; k < mult; k++)
            ok = ok && make_img(set.frame[k], w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                rw | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!ok) {
        fg.failed = true;
        drop_size();
        // Older submissions were drained above; ensure_size precedes every pass recorded by framegen_record.
        drop_setup();
        logf("vk: frame interpolation: its images (%ux%u) were not made: off", w, h);
        return false;
    }
    fg.w = w, fg.h = h, fg.mult = mult;
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

// The fixed CRT profiles need no history or LUTs. Only the selected pipeline is made, and no CRT
// resource is touched while Off (including the normal direct-present path).
struct CrtState {
    bool init = false, failed = false;
    bool tried[3] = {};
    Pipe downsample, filter[3];
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    Img raster, output;
    uint32_t rw = 0, rh = 0, w = 0, h = 0;
} crt;

bool init_crt(int filter) {
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    if (!crt.init) {
        crt.init = crt.failed = true;
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        si.maxLod = 0.0f;
        if (dev.push_descriptor && vkCmdPushDescriptorSetKHR &&
            make_pipe(crt.downsample, BUILTIN_SPV_ARGS(CRT_DOWNSAMPLE_COMP_SPV), {S, I}, 0)) {
            if (vkCreateSampler(dev.device, &si, nullptr, &crt.nearest) != VK_SUCCESS) {
                crt.nearest = VK_NULL_HANDLE;
            } else {
                si.magFilter = si.minFilter = VK_FILTER_LINEAR;
                if (vkCreateSampler(dev.device, &si, nullptr, &crt.linear) == VK_SUCCESS) crt.failed = false;
                else crt.linear = VK_NULL_HANDLE;
            }
        }
        if (crt.failed) {
            drop_pipe(crt.downsample);
            if (crt.nearest) vkDestroySampler(dev.device, crt.nearest, nullptr);
            if (crt.linear) vkDestroySampler(dev.device, crt.linear, nullptr);
            crt.nearest = crt.linear = VK_NULL_HANDLE;
            logf("vk: CRT filter unavailable: its pipeline or samplers could not be made; the unfiltered picture is kept");
        }
    }
    if (crt.failed) return false;
    if (!crt.tried[filter]) {
        crt.tried[filter] = true;
        // (each filter's shader through builtin_spv, RX_VK_SHADER_DIR: only the one asked for is looked up)
        const bool made = filter == 0   ? make_pipe(crt.filter[filter], BUILTIN_SPV_ARGS(CRT_LOTTES_COMP_SPV), {S, I}, 36)
                          : filter == 1 ? make_pipe(crt.filter[filter], BUILTIN_SPV_ARGS(CRT_HYLLIAN_COMP_SPV), {S, I}, 36)
                                        : make_pipe(crt.filter[filter], BUILTIN_SPV_ARGS(CRT_ZFAST_COMP_SPV), {S, I}, 36);
        if (!made)
            logf("vk: CRT filter unavailable: shader %d could not be made; the unfiltered picture is kept", filter);
    }
    return crt.filter[filter].pipe != VK_NULL_HANDLE;
}

VkImage record_crt(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh,
                   VkImageView& output_view) {
    const int mode = opts.crt_filter;
    const auto raster = crt_policy::raster(dw, dh, opts.crt_lines);
    const int filter = mode <= 2 ? 0 : mode - 2;
    if (!src || !sw || !sh || !raster.w || !raster.h || !init_crt(filter)) return VK_NULL_HANDLE;
    if (crt.rw != raster.w || crt.rh != raster.h || crt.w != dw || crt.h != dh) {
        {
            std::lock_guard qk(dev.present_mu);
            device_wait_idle_locked();
        }
        drop_img(crt.raster);
        drop_img(crt.output);
        crt.rw = crt.rh = crt.w = crt.h = 0;
        const VkImageUsageFlags sampled = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (!make_img(crt.raster, raster.w, raster.h, VK_FORMAT_R8G8B8A8_UNORM, sampled) ||
            !make_img(crt.output, dw, dh, VK_FORMAT_R8G8B8A8_UNORM, sampled | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            drop_img(crt.raster);
            drop_img(crt.output);
            crt.failed = true;
            logf("vk: CRT filter unavailable: its images could not be made; the unfiltered picture is kept");
            return VK_NULL_HANDLE;
        }
        crt.rw = raster.w, crt.rh = raster.h, crt.w = dw, crt.h = dh;
        to_general(cmd, crt.raster.image);
        to_general(cmd, crt.output.image);
    }
    const char* labels[] = {"", "crt-lottes-aperture", "crt-lottes-tv", "crt-hyllian", "crt-zfast"};
    GprofSpan timing(cmd, labels[mode]);   // existing opt-in GPU counters; no timestamp queries unless requested
    // Reduce the finished picture, not the 3D before its temporal upscaler. Area weights avoid aliasing
    // high-resolution detail into the deliberately low-resolution tube raster.
    pass(cmd, crt.downsample, {{src, crt.nearest}, {crt.raster.view, VK_NULL_HANDLE}}, raster.w, raster.h);
    const struct { float source[4], output[4]; uint32_t style; } pc{
        {(float)raster.w, (float)raster.h, 1.0f / raster.w, 1.0f / raster.h},
        {(float)dw, (float)dh, 1.0f / dw, 1.0f / dh}, (uint32_t)mode};
    static_assert(sizeof pc == 36);
    pass(cmd, crt.filter[filter], {{crt.raster.view, mode == 4 ? crt.linear : crt.nearest},
                                  {crt.output.view, VK_NULL_HANDLE}}, dw, dh, &pc, sizeof pc);
    static int last_mode = 0;
    static uint32_t last_w = 0, last_h = 0, last_lines = 0;
    if (last_mode != mode || last_w != dw || last_h != dh || last_lines != raster.h) {
        logf("vk: CRT %s: %ux%u virtual raster, %ux%u output; after scaling, before Hookshot overlays",
             labels[mode], raster.w, raster.h, dw, dh);
        last_mode = mode, last_w = dw, last_h = dh, last_lines = raster.h;
    }
    output_view = crt.output.view;
    return crt.output.image;
}

// (hookshot) The display thread's wait for a frame's time, on a high-resolution timer. sleep_for rests on the system timer,
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

// (hookshot) RX_VK_FG_LATENCY_LOG=<file>: one CSV line for each present of the display thread, its times on the PC's
// performance counter (QueryPerformanceCounter, the clock of PresentMon's --qpc_time): its set's game frame handed over
// (framegen_record, the set's `at`), the present called and returned, and whether it shows the game's frame. A present's
// call and return hold PresentMon's start of that present, so its row gives the game frame's time from its hand-over to the
// screen; PresentMon's own latency starts at the display thread's present before, after the set's wait for its time.
// Under FIFO that join fails: NVIDIA's driver starts each present 0.5 to 100 ms after the call, on a thread of its own, so
// <file>.shown.csv has each game frame's time on screen as the driver tells it: its present carries NVIDIA Reflex's present
// id (vk_reflex.cpp reflex_present_id), and a thread of the log's own waits for that id (VK_KHR_present_wait, as the
// presenter thread does under FIFO) and notes when it is shown. The display thread only queues the id, with the swapchain
// and generation its present went to (a set presented ahead: its PendingPresent's). The thread looks every 0.25 ms with no
// wait, holding present_image_mu and the presenter thread's dev.presenter_queue_mu only when both are free (a swapchain
// is externally synchronized: never beside a present call of the display thread or of the presenter thread, which a made
// frame is handed to (RX_VK_PRESENT_THREAD), nor a remake of the chain, and never in their way; the presenter thread's own
// present wait, outside its lock, can still meet it), and lat_sc_mu, which fg_latency_swapchain_gone takes before the chain is
// made again or destroyed. No present wait (VK_KHR_present_wait): no <file>.shown.csv; with NVIDIA Reflex off it has its
// first line only (no present ids)
struct LatShown {
    VkSwapchainKHR sc;
    uint64_t generation, id;
    int64_t at, called;
};
std::mutex lat_q_mu, lat_sc_mu;
std::condition_variable lat_cv;
std::deque<LatShown> lat_q;   // (lat_q_mu)
uint64_t lat_gone = 0;        // (lat_sc_mu: the swapchains of the generations under it are gone)
std::atomic<bool> lat_waits{false};

void lat_shown_main(FILE* fl) {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk latency log");
    const auto wait_for_present = (PFN_vkWaitForPresentKHR)vkGetDeviceProcAddr(dev.device, "vkWaitForPresentKHR");
    // (a timer of its own: wait_until_near's is the display thread's)
    const HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    const auto nap = [&] {
        LARGE_INTEGER d;
        d.QuadPart = -2500;   // (0.25 ms, 100 ns units)
        if (timer && SetWaitableTimerEx(timer, &d, 0, nullptr, nullptr, nullptr, 0)) WaitForSingleObject(timer, INFINITE);
        else std::this_thread::sleep_for(std::chrono::microseconds(250));
    };
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    std::fputs("present_id,handed_qpc,call_qpc,shown_qpc\n", fl);
    for (uint64_t lines = 1;;) {
        LatShown s;
        {
            std::unique_lock lk(lat_q_mu);
            lat_cv.wait(lk, [] { return !lat_q.empty(); });
            s = lat_q.front();
            lat_q.pop_front();
        }
        VkResult r = VK_TIMEOUT;
        for (int step = 0; step < 800 && r == VK_TIMEOUT; step++) {   // (200 ms at most)
            if (step) nap();
            std::lock_guard g(lat_sc_mu);
            if (s.generation < lat_gone) break;   // (made again since: the swapchain is gone)
            std::unique_lock busy(present_image_mu, std::try_to_lock);
            std::unique_lock handed(dev.presenter_queue_mu, std::defer_lock);
            if (busy.owns_lock() && handed.try_lock()) r = wait_for_present(dev.device, s.sc, s.id, 0);
        }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) continue;   // (suboptimal: shown too, the chain to be made again)
        LARGE_INTEGER qpc;
        QueryPerformanceCounter(&qpc);
        const int64_t now = now_ns();
        const auto ticks = [&](int64_t ns) { return qpc.QuadPart - (long long)((double)(now - ns) * freq.QuadPart / 1e9); };
        std::fprintf(fl, "%llu,%lld,%lld,%lld\n", (unsigned long long)s.id, (long long)ticks(s.at), (long long)ticks(s.called),
                     (long long)qpc.QuadPart);
        if (++lines % 64 == 0) std::fflush(fl);
    }
}

void fg_latency_note(int64_t at, int64_t called, bool game, uint64_t present_id, VkSwapchainKHR sc, uint64_t generation) {
    static FILE* fl = getenv("RX_VK_FG_LATENCY_LOG") ? std::fopen(getenv("RX_VK_FG_LATENCY_LOG"), "w") : nullptr;
    static uint64_t lines;
    if (!fl) return;
    if (!lines && dev.present_wait) {
        const std::string shown = std::string(getenv("RX_VK_FG_LATENCY_LOG")) + ".shown.csv";
        if (FILE* fs = std::fopen(shown.c_str(), "w")) {
            lat_waits.store(true);
            std::thread(lat_shown_main, fs).detach();
        }
    }
    LARGE_INTEGER freq, qpc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&qpc);
    const int64_t now = now_ns();
    const auto ticks = [&](int64_t ns) { return qpc.QuadPart - (long long)((double)(now - ns) * freq.QuadPart / 1e9); };
    if (!lines) std::fputs("game,handed_qpc,call_qpc,return_qpc\n", fl);
    std::fprintf(fl, "%d,%lld,%lld,%lld\n", game ? 1 : 0, (long long)ticks(at), (long long)ticks(called), (long long)qpc.QuadPart);
    if (++lines % 64 == 0) std::fflush(fl);
    if (game && present_id && sc && lat_waits.load(std::memory_order_relaxed)) {
        std::lock_guard g(lat_q_mu);
        if (lat_q.size() < 256) lat_q.push_back({sc, generation, present_id, at, called});
        lat_cv.notify_one();
    }
}

// doc: docs/design.md#13-frame-generations-presents-as-rules (rule 4: the spare images a set presented ahead holds)
/// Present ahead (display_main): a set's copies go to the GPU together, ahead of their presents, which holds as many
/// swapchain images acquired at once as the set has frames, so only where the swapchain has that many to spare (an acquire
/// past them has to time out, the Vulkan spec); a set it cannot hold is shown one frame at a time. pace: Frame pacing On,
/// or the set on the game's timeline (fixed_on_timeline)
static bool present_ahead_wanted(bool pace, int shown, uint32_t spare) {
    // (2x was the only one at first; at 3x and 4x the made frames' copies queued behind the game's next frame the same way,
    // so their presents reached the screen unevenly. vk_main.cpp asks for the images the sets need: 6 spare on an NVIDIA
    // surface, sets of up to 6 frames)
    return pace && shown >= 2 && spare >= (uint32_t)shown;
}

// (the display thread's: the frames it shows have ultrawide's loading cover in their sides already, framegen_record's copy)
thread_local bool wide_cover_in_frame = false;
void display_main() {
    wide_cover_in_frame = true;
    SetThreadDescription(GetCurrentThread(), L"mkw_vk display");
#ifdef TRACY_ENABLE
    HT_THREAD_NAME("mkw_vk display");
#endif
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    int64_t interval = 16666667, last_at = 0;
    // (by_time: sets on the game's timeline; early: those of them taken while the set before was shown; waited: those whose
    // copies did not go to the GPU as they came; ready: the others, ready_ns their first copies' ends past their game frames,
    // summed; slotted: their frames held back under FIFO to a refresh after the one before, fifo_slot; caught: made frames
    // left out by FIFO's catch-up, fifo_catch_up; games, game_ns: the sets' game frames presented and their time from the
    // game's handing over (framegen_record) to their present call, summed)
    struct { int on_time, late, skipped; int64_t since; int by_time, early, waited, ready; int64_t ready_ns; int slotted, caught,
             games; int64_t game_ns; } stat{};
    int64_t next_show = 0, shown_at = 0, slot = 0;   // (slot: the time the last frame was given, for fifo_slot)
    int64_t fifo_refresh = 0;   // (the last set on the line's refresh, ns: take's catch-up under FIFO)
    int hist[7] = {};   // (<2 ms apart: two presents the screen hardly tells apart)
    static const char* timeline_env = getenv("RX_VK_FG_TIMELINE");
    const bool timeline_off = timeline_env && env_value_off(timeline_env);
    // (a set the display thread has taken: how it is shown; presented ahead, its copies, how many went to the GPU so far
    // (copied), whether its first one went as the set came (as_came: within 3 ms of framegen_queue), and when that one was
    // seen done (ready_at, while the set before was shown))
    struct Taken {
        Job j{};
        bool auto_timed = false, line = false, timed = false, ahead = false, early = false, as_came = false;
        int shown = 0, copied = 0;
        int first = 0;   // (made frames left out, spread over the set: FIFO's catch-up, take; frame k shown is fifo_pick's)
        int64_t step = 0, ready_at = 0;
        int64_t refresh = 0;   // (on the game's timeline: the display's refresh, ns, for fifo_slot; 0 not known)
        PendingPresent pending[kMaxMult];
    };
    Taken taken[2];   // (the set being shown, and the next one, taken early: take_early)
    int held = 0;
    // A fixed multiplier's set at its times on the game's timeline (by_time): fixed_on_timeline. Its game[] marks its
    // game frame, as Auto's
    const auto on_line = [&](const OutSet* set) { return set->shown < 0 && fixed_on_timeline(set->by_time, timeline_off); };
    // Frame k of a set presented ahead: its copy into the window to the GPU now, its present at its time (present_commit)
    const auto copy = [&](Taken& s, int k) {
        const Job& j = s.j;
        const bool timed = s.timed;
        const int f = s.first ? fifo_pick(j.set->made + 1, s.shown, k) : k;   // (the set's frame: the catch-up's picks)
        present_kind.store(timed ? (j.set->game[f] ? 0 : 1) : f < j.set->made ? 1 : 0);
        reflex_present_frame = (timed ? j.set->game[f] : f == j.set->made) ? j.set->reflex_id : 0;
        antilag_present_frame = (timed ? j.set->game[f] : f == j.set->made) ? j.set->antilag_id : 0;
        present_defer_into = &s.pending[k];
        const int64_t t0 = now_ns();
        present_image(j.set->frame[f].image, fg.w, fg.h, nullptr, j.set->frame[f].view);
        present_defer_into = nullptr;
        // (as it came: taken within 3 ms of its coming, and its first copy not held in present_image over 3 ms, waiting for a
        // frame slot, a swapchain image or a new swapchain: the review, 2026-11-17. A wait there grows with the delay; the
        // copy's own time to submit (the game's held submits, the queue's lock) is the pipeline's and counts in the delay:
        // timed from after it, a CPU that takes 3 ms to submit left every set out, and the delay stayed at 0)
        if (k == 0) s.as_came = t0 - j.queued <= 3000000 && now_ns() - t0 <= 3000000;
    };
    // The copies of a set presented ahead still to go to the GPU, in order, for as many of its frames as `room` (the spare
    // swapchain images not held by the set before)
    const auto copy_up_to = [&](Taken& s, int room) {
        if (!s.ahead) return;
        for (const int n = std::min(s.shown, room); s.copied < n; s.copied++) copy(s, s.copied);
    };
    // The set taken: its stats, how it is shown, and, presented ahead, its copies to the GPU for `room` of its frames
    const auto take = [&](Taken& s, const Job& j, int room, int left) {
        s = Taken{};
        s.j = j;
        if (last_at) {
            const int64_t d = j.at - last_at;
            if (d > 0 && d < 100000000) interval = (interval * 7 + std::clamp<int64_t>(d, 4000000, 50000000)) / 8;
        }
        last_at = j.at;
        if (!stat.since) stat.since = j.at;
        if (j.at - stat.since > 5000000000LL) {
            // (the game frame's latency here: from its handing over to its present call, the same with RX_VK_FG_TIMELINE=0, so
            // one build gives both sides. A present call, not the screen: under FIFO a present can still wait in the swapchain's
            // queue, which main's path fills more, so this reads main's side low, never the timeline's high)
            logf("vk: frame interpolation: %d frames shown on time, %d late (over 2 ms), %d made ones left out (the game's next "
                 "frame waiting); shown apart: <2 ms %d, 2-7 %d, 7-9 %d, 9-12 %d, 12-17 %d, 17-25 %d, over 25 %d; the game's "
                 "frames presented %.1f ms after the game handed them over", stat.on_time, stat.late, stat.skipped, hist[0],
                 hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], stat.games ? stat.game_ns / 1e6 / stat.games : 0.0);
            if (stat.by_time)
                logf("vk: frame interpolation: %d sets on the game's timeline, %d taken early, %d with copies later than "
                     "they came (left out of the GPU's delay), %d frames held to a refresh after the one before (FIFO), %d made "
                     "frames left out to catch up (FIFO); first "
                     "copies done %.1f ms after their game frame (seen within 2 ms), %.1f ms past their lead for the GPU "
                     "(RX_VK_FG_TIMELINE=0: off)",
                     stat.by_time, stat.early, stat.waited, stat.slotted, stat.caught,
                     stat.ready ? stat.ready_ns / 1e6 / stat.ready : 0.0,
                     fg.late_ns.load(std::memory_order_relaxed) / 1e6);
            memset(hist, 0, sizeof hist);
            stat = {0, 0, 0, j.at, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        // (hookshot) The pacer. The game hands over frames 12 to 28 ms apart around a mean of 19, and shown as they came that
        // was the stutter, whatever the frame rate. Every frame now has its place on an even line, the recent frame time
        // apart, a frame early waiting for it: never more than one game frame behind what the game handed over, and a line
        // that has fallen behind what came starts again from now. RX_VK_PACE=0: as they come.
        const bool pace = opts.pace != 0;
        s.auto_timed = j.set->shown >= 0;   // (Auto: each frame at its own refresh's time)
        s.shown = s.auto_timed ? j.set->shown : j.set->made + 1;
        // (hookshot) FIFO's catch-up: a set whose game frame would come over a refresh late leaves made frames out, spread
        // over it, and their copies (fifo_catch_up, fifo_pick). 3x at 165 Hz and 60 a second without it: the game frame 82
        // ms from its hand-over to the screen, against 51 before the timeline (its latency A/B); 6x: 138 ms. left: the set
        // before's frames still to show. Only a set on the timeline, which the swapchain holds whole (present_ahead_wanted):
        // one shown a frame at a time (7x and 8x under FIFO on an NVIDIA surface by the renderer's own or FSR 3.1: DLSS
        // FG runs them at 6x) goes out as before, all its frames
        if (!s.auto_timed && dev.swap_mode == VK_PRESENT_MODE_FIFO_KHR && on_line(j.set) &&
            present_ahead_wanted(true, j.set->made + 1, dev.swap_spare.load(std::memory_order_relaxed)) &&
            (s.first = fifo_catch_up(j.set->made, j.set->due, slot, left, fifo_refresh, now_ns())))
            s.shown -= s.first, stat.caught += s.first;
        s.step = interval / std::max(s.shown, 1);
        if (s.auto_timed) menu_note_frame(true, 0);   // (the overlay's game rate: Auto counts each game frame, given a refresh or not)
        // (hookshot) Present ahead (#1702). Each frame's copy into the window was submitted at its
        // time, behind the game's next frame on the same queue, and shown only once that frame's rendering was done: at 2x the
        // game's own frames reached the screen ~3 ms after the made ones, 5.7 and 11 ms apart instead of 8.3 (PresentMon, RTX
        // 5080, 60 fps), and a VRR display's refresh changed every frame (flicker on OLED). Now a set's copies go to the GPU
        // together, the first one's end is waited for, and only the presents keep their times on the line.
        const bool line = on_line(j.set);
        s.ahead = present_ahead_wanted(pace || line, s.shown, dev.swap_spare.load(std::memory_order_relaxed));
        // (on the game's timeline only presented ahead: a set shown one frame at a time has no first copy that tells the GPU's
        // delay (pace_delay), and keeps the even line or none, as before)
        s.line = line && s.ahead;
        s.timed = s.auto_timed || s.line;
        if (s.ahead) {
            static bool said;
            if (!said) said = true, logf("vk: present ahead: a set's copies submitted at once, its presents on time");
            copy_up_to(s, room);
        }
        // (fifo_slot's refresh; whether the swapchain is FIFO is read at each present: it is made again for a new mode. Asked
        // after the copies: once a second it asks Windows, which is not the first copy's time, taken in copy)
        const int mhz = s.line ? display_refresh_mhz() : 0;
        s.refresh = mhz > 0 ? (int64_t)(1e12 / mhz) : 0;
        fifo_refresh = s.line ? s.refresh : 0;   // (the catch-up's: the last set's on the timeline; 0 off it or unknown)
    };
    // (hookshot) While a set on the game's timeline waits for its frames' times, the next one is taken as soon as it comes,
    // its copies to the GPU then for as many of its frames as the swapchain has images beside the ones this set still holds
    // (left: its frames still to be presented; vk_main.cpp asks for two sets' images, but an NVIDIA surface has 6 spare,
    // under two sets of 4); the rest as this set's presents give images back. Taken once this set's frames were shown, the
    // copies queued behind the parts of the game's next frame submitted meanwhile: they were done that much later, the set
    // past its times, and the GPU's delay (pace_delay) grew with its own lateness. Not within 3 ms of this set's next frame
    // (its copies take that long to submit)
    const auto take_early = [&](int left, int64_t until) {
        if (held != 1 || !taken[0].line || until - now_ns() < 3000000) return;
        const int spare = (int)dev.swap_spare.load(std::memory_order_relaxed);
        Job j;
        {
            std::lock_guard lk(fg.mu);
            if (fg.jobs.empty()) return;
            j = fg.jobs.front();
            const int shown = j.set->made + 1;
            if (!on_line(j.set) || shown < 2 || shown > spare || spare - left < 1) return;
            fg.jobs.pop_front();
        }
        take(taken[1], j, spare - left, left);
        taken[1].early = true;
        held = 2;
    };
    for (;;) {
        if (!held) {
            Job j;
            {
                std::unique_lock lk(fg.mu);
                fg.cv.wait(lk, [] { return !fg.jobs.empty(); });
                j = fg.jobs.front();
                fg.jobs.pop_front();
            }
            take(taken[0], j, (int)dev.swap_spare.load(std::memory_order_relaxed), 0);
            held = 1;
        }
        Taken& s = taken[0];
        const Job& j = s.j;
        const bool pace = opts.pace != 0;
        // (taken early: its copies that the set before left no images for)
        copy_up_to(s, (int)dev.swap_spare.load(std::memory_order_relaxed));
        const bool first = s.ahead && s.pending[0].used;   // (its first copy made ahead)
        if (first) vkWaitForFences(dev.device, 1, &s.pending[0].done, VK_TRUE, 100000000ull);   // (100 ms at most)
        // (hookshot) A fixed multiplier's set on the game's timeline, presented ahead (its first copy's end waited for): its own
        // delay and how late its first frame can be shown against its time, a millisecond to spare, set the delay of the sets
        // after it (pace_delay). The copy's end as seen while the set before was shown (ready_at, taken early), else now: a set
        // taken early was often done well before the set before it ended. Only from a set whose first copy went to the GPU
        // as it came (as_came): one that waited, for images or for the display thread, had its copies behind more of the
        // game's work the later the set before was shown, so the delay grew with its own lateness (at 4x a set waited for
        // the images of the one before once the delay passed 3/4 of a game frame, and the delay went up to a whole one).
        // Not from a set after a long game frame either (over 1.5 smoothed ones: its first frame's time came early by the
        // difference, which is not the GPU's)
        if (s.line) {
            stat.by_time++;
            stat.early += s.early;
            stat.waited += !s.as_came;
            if (first && s.as_came) {
                const int64_t at = s.ready_at ? s.ready_at : now_ns();
                stat.ready++;
                stat.ready_ns += at - j.at;
                if (j.set->span_ns * 2 <= j.set->frame_ns * 3)
                    fg.late_ns.store(pace_delay(fg.late_ns.load(std::memory_order_relaxed),
                                                j.set->extra_ns + (at - j.set->due[0]) + 1000000, j.set->frame_ns),
                                     std::memory_order_relaxed);
            }
        }
        const bool timed = s.timed;
        for (int k = 0; k < s.shown; k++) {
            const int f = s.first ? fifo_pick(j.set->made + 1, s.shown, k) : k;   // (the set's frame: the catch-up's picks)
            // (hookshot) A made frame is left out when the game's next frame is already waiting: the display thread behind
            // the game held its sets, and the game waited for them (the game's rate fell, frame generation on). The set's
            // game frame is always shown. Not with present ahead (its copies are already on their way), nor for a set on the
            // game's timeline: each of its frames has its own time, and the next set is taken early on purpose (FIFO's
            // catch-up, fifo_catch_up, leaves its made frames out instead)
            if (!s.ahead && !s.line && (timed ? !j.set->game[f] : f < j.set->made)) {
                bool newer = held == 2;
                if (!newer) {
                    std::lock_guard lk(fg.mu);
                    newer = !fg.jobs.empty();
                }
                if (newer) {
                    stat.skipped++;
                    continue;
                }
            }
            int64_t due = timed ? j.set->due[f] : pace ? next_show : 0;
            // (a frame on the game's timeline is late against its own time, below; under FIFO against its refresh slot, a
            // refresh after the slot before, also where fifo_slot starts again from now: so a set late from the GPU or a
            // held thread is counted, the review, 2026-11-17)
            int64_t own = due;
            int64_t now = now_ns();
            if (s.refresh && dev.swap_mode == VK_PRESENT_MODE_FIFO_KHR) {   // (one a refresh at most)
                if (slot > 0) own = std::max(due, slot + s.refresh);
                const int64_t at = fifo_slot(due, slot, s.refresh, now);
                stat.slotted += at > due && at > now;
                due = at;
            }
            if (due < now) due = now;
            if (!timed && due > j.at + interval + s.step) due = j.at + interval + s.step;
            if (timed && due > now + 100000000) due = now;   // (a game frame time gone wrong: not held back)
            const int64_t slot_was = slot;
            slot = due;
            // (the wait in steps of 2 ms at most while the next set may still be taken early: at 2x it comes while this set's
            // game frame waits, which a look at the wait's start alone missed; and while a set taken early has its first copy's
            // end still to be seen, ready_at. Its copies that had no images go to the GPU as this set's presents give them back)
            for (;;) {
                take_early(s.shown - k, due);
                const bool may_see = held == 2 && !taken[1].ready_at && taken[1].pending[0].used;
                if (may_see && vkWaitForFences(dev.device, 1, &taken[1].pending[0].done, VK_TRUE, 0) == VK_SUCCESS)
                    taken[1].ready_at = now_ns();
                if (held == 2 && taken[1].copied < taken[1].shown && due - now_ns() >= 3000000)
                    copy_up_to(taken[1], (int)dev.swap_spare.load(std::memory_order_relaxed) - (s.shown - k));
                now = now_ns();
                if (due - now <= 1000000) break;
                const bool may_take = held == 1 && s.line;
                wait_until_near(may_take || may_see ? std::min<int64_t>(due - now - 1000000, 2000000) : due - now - 1000000);
            }
            while (now_ns() < due) SwitchToThread();
            const int64_t t = now_ns();
            if (t > slot + 1000000) slot = t;   // (presented late, the next set taken meanwhile: the next a refresh after it)
            if (s.ahead) {
                if (!present_commit(s.pending[k])) {   // (not shown: the line goes on, and the overlay and the counts leave it out)
                    next_show = t + s.step;
                    slot = slot_was;   // (nothing shown: its slot is not taken)
                    continue;
                }
            } else {
                present_kind.store(timed ? (j.set->game[f] ? 0 : 1) : f < j.set->made ? 1 : 0);
                // (NVIDIA Reflex: the set's game frame is presented as the game frame it was made at, made frames as none)
                reflex_present_frame = (timed ? j.set->game[f] : f == j.set->made) ? j.set->reflex_id : 0;
                antilag_present_frame = (timed ? j.set->game[f] : f == j.set->made) ? j.set->antilag_id : 0;
                present_image(j.set->frame[f].image, fg.w, fg.h, nullptr, j.set->frame[f].view);
            }
            fg_latency_note(j.at, t, timed ? j.set->game[f] : f == j.set->made, j.set->reflex_id,
                            s.ahead ? s.pending[k].swapchain : VK_NULL_HANDLE, s.ahead ? s.pending[k].swapchain_generation : 0);   // (RX_VK_FG_LATENCY_LOG)
            menu_note_frame(!s.auto_timed && f == j.set->made);
            if (timed ? j.set->game[f] : f == j.set->made) stat.games++, stat.game_ns += t - j.at;   // (the set's game frame)
            if (shown_at) {
                const int64_t d = t - shown_at;
                hist[d < 2000000 ? 0 : d < 7000000 ? 1 : d < 9000000 ? 2 : d < 12000000 ? 3 : d < 17000000 ? 4
                     : d < 25000000 ? 5 : 6]++;
            }
            shown_at = t;
            next_show = t + s.step;
            (t - (s.line ? own : due) > 2000000 ? stat.late : stat.on_time)++;
        }
        {
            std::lock_guard lk(fg.mu);
            j.set->busy = false;
            fg.cv.notify_all();
        }
        if (held == 2) taken[0] = taken[1];
        held--;
    }
}

// (hookshot) The last sets' makers (framegen_maker): each framegen_record notes its set's on the replay thread; the menu's
// thread reads the counts. A ring of kMakerSets, the counts of each maker in it
constexpr int kMakerSets = 64;
std::atomic<int> maker_count[5] = {kMakerSets, 0, 0, 0, 0};   // (by FgMaker)
uint8_t maker_ring[kMakerSets] = {};
int maker_pos = 0;
std::atomic<int64_t> maker_at{0};

void note_maker(FgMaker m) {
    uint8_t& slot = maker_ring[maker_pos];
    maker_count[slot].fetch_sub(1, std::memory_order_relaxed);
    slot = (uint8_t)m;
    maker_count[slot].fetch_add(1, std::memory_order_relaxed);
    maker_pos = (maker_pos + 1) % kMakerSets;
    maker_at.store(now_ns(), std::memory_order_relaxed);
}


// ---- (hookshot) the item wheel's backdrop and picture ----
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
    if (kind == 3) {
        const Id src = m.OpImageFetch(vec4, m.OpImage(sampled, m.OpLoad(sampled_t, in)), g, spv::ImageOperandsMask::Lod, c0);
        const Id img = m.OpLoad(storage, out);
        const Id dst = m.OpImageRead(vec4, img, g);
        const Id alpha = m.OpFMul(f32, m.OpConvertSToF(f32, m.OpCompositeExtract(i32, offset, 0U)), m.Constant(f32, 0.001f));
        const Id factor = m.OpCompositeConstruct(vec4, alpha, alpha, alpha, alpha);
        m.OpImageWrite(img, g, m.OpFMix(vec4, dst, src, factor));
    } else if (kind == 2) {   // (the picture over the image: premultiplied, src + dst * (1 - src.a))
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

// (hookshot) This present gave FSR 3.1 a frame (fsr3fi_frames, right before fsr3fi::generate): its own images were in use, whatever
// came of it and whichever made frames were shown. framegen_idle takes it, and clears it for the next present: both on the replay
// thread, framegen_record before framegen_idle in every present
bool fsr_given = false;

} // namespace

int framegen_capped(int framegen, int dlss_most) {
    return framegen >= 2 && dlss_most > 0 ? std::min(framegen, dlss_most + 1) : framegen;
}
int framegen_now() { return framegen_capped(opts.framegen, dlss_fg_most()); }

FgMaker framegen_maker() {
    // (none recorded for half a second: Frame generation off, or the window's presents without it)
    if (now_ns() - maker_at.load(std::memory_order_relaxed) > 500000000) return FgMaker::None;
    const int b = maker_count[(int)FgMaker::BuiltIn].load(std::memory_order_relaxed);
    const int d = maker_count[(int)FgMaker::Dlss].load(std::memory_order_relaxed);
    const int f = maker_count[(int)FgMaker::Fsr3].load(std::memory_order_relaxed);
    const int p = maker_count[(int)FgMaker::DlssPaused].load(std::memory_order_relaxed);
    if (!b && !d && !f && !p) return FgMaker::None;
    if (p > b && p > d && p > f) return FgMaker::DlssPaused;
    return d >= b && d >= f ? FgMaker::Dlss : f >= b ? FgMaker::Fsr3 : FgMaker::BuiltIn;
}

// The Insert menu's Frame generation is off at 0. While it was ignored, every present still copied the frame and
// reduced it to luma (framegen_record), then a second thread showed it. Open areas paid for that on a full GPU and
// gained no made frames. Auto (1) and 2x and up are the interpolation; RX_VK_FRAMEGEN=0 keeps it off entirely.
bool framegen_wanted(int menu_value, int allow) {
    return allow && menu_value >= 1;   // (1: Auto)
}

bool framegen_on() {
    static const int allow = !getenv("RX_VK_FRAMEGEN") || atoi(getenv("RX_VK_FRAMEGEN")) != 0;
    return framegen_wanted(opts.framegen, allow) && !dev.offscreen && !fg.failed;
}

void framegen_idle(bool recorded) {
    using namespace framegen_idle_rule;
    static Counts counts;
    const bool on = framegen_on();
    // (FSR's own images by this present's own use of them, not by `recorded`: a set the built-in or DLSS-G made leaves them unused)
    const bool fsr_used = std::exchange(fsr_given, false);
    const Release due = step(counts, {on, on && fsr3fi::frames_wanted(), recorded, fsr_used, fg.w != 0});
    if (due.fsr) fsr3fi::release();
    if (!due.shared) return;
    {   // (as ensure_size: the display thread done with the sets, then every queue idle)
        std::unique_lock lk(fg.mu);
        fg.cv.wait(lk, [] { return fg.jobs.empty() && !fg.sets[0].busy && !fg.sets[1].busy && !fg.sets[2].busy; });
    }
    {
        std::lock_guard qk(dev.present_mu);
        device_wait_idle_locked();
    }
    const uint64_t mb = ((uint64_t)fg.w * fg.h * 4 * (2 + 3 * (uint64_t)fg.mult)) >> 20;   // (the RGBA8 frames; lumas and flows are small)
    logf("vk: frame interpolation off: its images (%ux%u, %dx) let go, about %llu MB", fg.w, fg.h, fg.mult, (unsigned long long)mb);
    drop_size();
}

// Ultrawide's menus (the Insert menu's Ultrawide menus): which of nx2/oot_wide.py's keep-list screens keep the whole picture, told
// to the game's side each present, and, while a menu shows, the sides of the middle 16:9 to show out of focus (1)
// or black (2), as fractions of the width. A menu's layout is set up once, when it is first opened: what tells one is
// that the frame draws no world (vk_draw.cpp world_drawn_at), as a pause, the map, a loading screen or a movie
// does not (3 presents' grace for a frame the world skips). The middle is centred: its right edge is 1 - x0. One 8-byte atomic: the
// frame generation's display thread reads it (framegen_scale) while the replay thread writes it (wide_sides_tick)
struct WideSides { float x0 = 0.0f; uint16_t side = 0, cover = 0; };   // (cover: of 65535)
// (KEEP_VTABLES 1 and 3: the pause and map menus. 0, the item wheel's background, and 2, the fades, keep covering the whole picture)
constexpr uint32_t kMenus = 0xa;
std::atomic<bool> wide_possible{false};   // (a margin and the setting on: vk_replay.cpp takes the scale pass's path for the sides)
WideSides wide_sides(float cover) {
    WideSides w;
    if (!host || !host->wide_menus) return w;
    // (KEEP_VTABLES 1 and 3: the pause / system menus and the loading fade, whose layout host/hos_wide_ui.c then places as the
    // HUD's. 0, the item wheel's background, and 2, the event fade, keep covering the whole picture)
    constexpr uint32_t kMenus = 0xa;
    const float margin = host->wide_menus(opts.wide_menus ? 0xfu & ~kMenus : 0xfu);
    extern std::atomic<int64_t> world_drawn_at;
    extern int64_t presents_now;
    extern std::atomic<int64_t> wheel_drawn_at;
    // (no world: a menu. But not the item wheel, whose background is the last frame of play over the whole picture, as on the
    // console: its sides are left as the game draws them)
    const bool no_world = presents_now - world_drawn_at.load(std::memory_order_relaxed) > 3 &&
                          presents_now - wheel_drawn_at.load(std::memory_order_relaxed) > 3;
    wide_possible.store(opts.wide_menus && margin > 0.0f, std::memory_order_relaxed);
    if (opts.wide_menus && margin > 0.0f && no_world) w.x0 = margin, w.side = (uint16_t)opts.wide_menus;
    // (the loading cover between two areas is drawn in the middle like the menus, and the world is drawn again under it before
    // it opens: the sides showed the new area beside a black middle, issue 2220. While it shows, the sides are closed by as
    // much as the cover is, fg_scale.comp's cover, with the world drawn or not)
    if (opts.wide_menus && margin > 0.0f && cover > 0.0f) w.x0 = margin, w.cover = (uint16_t)std::lround(std::min(cover, 1.0f) * 65535.0f);
    static bool covered = false;
    if ((w.cover != 0) != covered) {
        covered = w.cover != 0;
        logf("vk: ultrawide loading cover %s at present %lld (the world last drawn at present %lld)", covered ? "shows" : "gone",
             (long long)presents_now, (long long)world_drawn_at.load(std::memory_order_relaxed));
    }
    static int told = 0;
    if (w.side != told) {   // (each change logged: a report of blurred sides in play says when they came and went)
        logf("vk: ultrawide menu sides %s at present %lld (the world last drawn at present %lld)", w.side ? "on" : "off",
             (long long)presents_now, (long long)world_drawn_at.load(std::memory_order_relaxed));
        told = w.side;
    }
    return w;
}
std::atomic<WideSides> wide_now;   // (wide_sides_tick's, each present)
// (the middle 16:9's left edge while Ultrawide menus has the menus there and the picture is wider, as wide_sides has it; else 0)
float wide_margin() { return opts.wide_menus && host && host->wide_menus ? host->wide_menus(0xfu & ~kMenus) : 0.0f; }
bool wide_sides_possible() { return wide_possible.load(std::memory_order_relaxed); }

VkImage framegen_scale(VkCommandBuffer cmd, VkImageView src, VkImage image, uint32_t sw, uint32_t sh, const int crop[4], uint32_t dw, uint32_t dh,
                       bool pq) {
    WideSides wide = wide_now.load(std::memory_order_relaxed);   // (wide_sides_tick, each present)
    if (wide_cover_in_frame) wide.cover = 0;
    if (!src || !image || !sw || !sh || !dw || !dh) return VK_NULL_HANDLE;
    const PresentSourceRect source = present_source_rect(sw, sh, crop);
    if (!source.w || !source.h) return VK_NULL_HANDLE;
    // (a picture larger than the window takes the pass even with the filter Off: the blit's linear filter reads 2 x 2 of the
    // picture whatever the shrink, and fine detail shimmered)
    const bool shrink = source.w > dw || source.h > dh;
    const bool scaling = opts.upscale || opts.sharpness || wide.side || wide.cover || shrink;
    const bool tube = crt_policy::enabled(opts.crt_filter, pq);
    if (!scaling && !tube) return VK_NULL_HANDLE;
    if (scaling && !init_scale()) return VK_NULL_HANDLE;
    // (pq: the game's own HDR10 picture (Native HDR), PQ codes in RGB10A2. Scaled into RGB10A2 images by fg_scale_pq: through
    // the RGBA8 ones its codes lost two bits, banding in the dark. Without that pass it is shown unscaled. The images made for
    // the other format are made again)
    if (pq && !scale.pq_tried) {
        scale.pq_tried = true;
        const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (!make_pipe(scale.pipe_pq, BUILTIN_SPV_ARGS(FG_SCALE_PQ_COMP_SPV), {S, I, S}, 24))
            logf("vk: the 10-bit window scaling pass could not be made: Native HDR's picture is shown unscaled");
    }
    if (pq && !scale.pipe_pq.pipe) return VK_NULL_HANDLE;
    if (scale.pq != pq) {
        scale.pq = pq;
        scale.cropped_w = scale.cropped_h = scale.scaled_w = scale.scaled_h = scale.small_w = scale.small_h = 0;
    }
    const VkFormat format = pq ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_R8G8B8A8_UNORM;
    const Pipe& pipe = pq ? scale.pipe_pq : scale.pipe;
    // (the game draws into part of its window textures, 2048 x 1152 of 2560 x 1440 here: the filter was given the whole texture, and the
    // picture came out in the top left corner of the window with black around it. The crop first, then the filter)
    if (source.x || source.y || source.w != sw || source.h != sh) {
        const uint32_t cw = source.w, ch = source.h;
        if (scale.cropped_w != cw || scale.cropped_h != ch) {
            {
                std::lock_guard qk(dev.present_mu);
                device_wait_idle_locked();
            }
            drop_img(scale.cropped);
            scale.cropped_w = scale.cropped_h = 0;
            if (!make_img(scale.cropped, cw, ch, format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
                drop_img(scale.cropped);
                return VK_NULL_HANDLE;
            }
            scale.cropped_w = cw, scale.cropped_h = ch;
            to_general(cmd, scale.cropped.image);
        }
        if (pq) {   // (the bits as they are: a blit converts from the image's own format, which is not always the picture's
                    // (vk_texture.cpp format_class shares images by class); a copy keeps the game's 10-bit codes either way)
            VkImageCopy c{};
            c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.srcOffset = {(int)source.x, (int)source.y, 0};
            c.extent = {cw, ch, 1};
            vkCmdCopyImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, scale.cropped.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        } else {
            VkImageBlit b{};
            b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            b.srcOffsets[0] = {(int)source.x, (int)source.y, 0};
            b.srcOffsets[1] = {(int)(source.x + cw), (int)(source.y + ch), 1};
            b.dstOffsets[1] = {(int)cw, (int)ch, 1};
            vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, scale.cropped.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
        }
        full_barrier(cmd);
        src = scale.cropped.view;
    }
    // With no window effect, the CRT itself maps the finished game image to the output pixel grid.
    // Avoid an otherwise redundant full-size scale image and dispatch.
    if (!scaling) return record_crt(cmd, src, source.w, source.h, dw, dh, scale.output_view);
    if (scale.scaled_w != dw || scale.scaled_h != dh) {
        {
            std::lock_guard qk(dev.present_mu);   // (the queue is idle only while nothing else submits to it)
            device_wait_idle_locked();
        }
        drop_img(scale.scaled);
        scale.scaled_w = scale.scaled_h = 0;
        // (sampled too: with HDR on, framegen_hdr reads the scaled picture through a sampler. Without the usage that read is
        // undefined, and an upscaler or a Sharpness above 0 (every preset) gave a black, grey or purple HDR picture where 0 did not)
        if (!make_img(scale.scaled, dw, dh, format,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
            drop_img(scale.scaled);
            return VK_NULL_HANDLE;
        }
        scale.scaled_w = dw, scale.scaled_h = dh;
        to_general(cmd, scale.scaled.image);
    }
    struct { int mode; float sharp; float x0, x1; int side; float cover; } pc{opts.upscale, opts.sharpness / 100.0f, wide.x0, 1.0f - wide.x0, wide.side,
                                                                              wide.cover / 65535.0f};
    if (wide.side == 1) {   // (the blurred sides: first the picture at 1/8, each pixel the mean of its 8 x 8, which they blur)
        // (from #1448's checked source rectangle, as the crop above: the picture src now holds)
        const uint32_t smw = std::max(1u, source.w / 8), smh = std::max(1u, source.h / 8);
        if (scale.small_w != smw || scale.small_h != smh) {
            {
                std::lock_guard qk(dev.present_mu);
                device_wait_idle_locked();
            }
            drop_img(scale.small);
            scale.small_w = scale.small_h = 0;
            if (!make_img(scale.small, smw, smh, format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
                drop_img(scale.small);
                return VK_NULL_HANDLE;
            }
            scale.small_w = smw, scale.small_h = smh;
            to_general(cmd, scale.small.image);
        }
        pc.side = 3;   // (fg_scale.comp: make the small copy)
        pass(cmd, pipe, {{src, scale.linear}, {scale.small.view, VK_NULL_HANDLE}, {src, scale.linear}}, smw, smh, &pc, sizeof pc);
        full_barrier(cmd);
        pc.side = 1;
    }
    pass(cmd, pipe, {{src, scale.linear}, {scale.scaled.view, VK_NULL_HANDLE}, {wide.side == 1 ? scale.small.view : src, scale.linear}},
         dw, dh, &pc, sizeof pc);
    static const bool report = env_on("RX_VK_UPSCALE_LOG");
    static bool reported = false;
    if (report && !reported) {
        reported = true;
        logf("vk: upscale dispatched: mode %d, sharpness %d%% (of RCAS's strongest, neighbours one source pixel apart), source %ux%u, "
             "crop request %d,%d %dx%d, destination %ux%u",
             opts.upscale, opts.sharpness, sw, sh, crop ? crop[0] : 0, crop ? crop[1] : 0,
             crop ? crop[2] : (int)sw, crop ? crop[3] : (int)sh, dw, dh);
    }
    if (tube)
        if (const VkImage filtered = record_crt(cmd, scale.scaled.view, dw, dh, dw, dh, scale.output_view)) return filtered;
    scale.output_view = scale.scaled.view;
    return scale.scaled.image;
}

VkImageView framegen_scaled_view() { return scale.output_view; }

/// (hookshot) The Insert menu's HDR: src's crop to a dw x dh HDR10 image (PQ, Rec. 2020, 10 bits), to be copied into an HDR10
/// swapchain's image as the SDR picture is (shaders/hdr_encode.comp); null when it cannot be made (then the SDR picture;
/// Native's scaled picture, else black).
/// src is the SDR picture, expanded, or with pq the game's own HDR10 picture (Native); either graded by the HDR contrast row
VkImage framegen_hdr(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, const int crop[4], uint32_t dw, uint32_t dh,
                     bool pq) {
    if (!src || !sw || !sh || !dw || !dh) return VK_NULL_HANDLE;
    const PresentSourceRect source = present_source_rect(sw, sh, crop);
    if (!source.w || !source.h) return VK_NULL_HANDLE;
    if (!hdr.init) {
        hdr.init = hdr.failed = true;
        const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (dev.push_descriptor && vkCmdPushDescriptorSetKHR && init_scale() &&
            make_pipe(hdr.pipe, BUILTIN_SPV_ARGS(HDR_ENCODE_COMP_SPV), {S, I}, 48))
            hdr.failed = false;
        else
            logf("vk: HDR: the HDR pass could not be made: SDR");
    }
    if (hdr.failed) return VK_NULL_HANDLE;
    if (hdr.w != dw || hdr.h != dh) {
        {
            std::lock_guard qk(dev.present_mu);
            device_wait_idle_locked();
        }
        drop_img(hdr.out);
        hdr.w = hdr.h = 0;
        if (!make_img(hdr.out, dw, dh, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            drop_img(hdr.out);
            static bool told;
            if (!told) told = true, logf("vk: HDR: no 10-bit storage image on this GPU: SDR");
            return VK_NULL_HANDLE;
        }
        hdr.w = dw, hdr.h = dh;
        to_general(cmd, hdr.out.image);
    }
    const float x0 = source.x / (float)sw, y0 = source.y / (float)sh;
    const float x1 = (source.x + source.w) / (float)sw, y1 = (source.y + source.h) / (float)sh;
    // (hdr_encode.comp's push constants: the crop, paper white, peak, knee, mode, contrast, exposure (1: none), two pads)
    const struct { float uv[4]; float white, peak, knee, mode, contrast, exposure, pad0, pad1; } pc{
        {x0, y0, x1, y1}, (float)std::clamp(opts.hdr_white, 80, 500), (float)std::clamp(opts.hdr_peak, 400, 10000), 0.6f,
        pq ? 1.0f : 0.0f, std::clamp(opts.hdr_contrast, 50, 200) / 100.0f, 1.0f, 0.0f, 0.0f};
    pass(cmd, hdr.pipe, {{src, scale.linear}, {hdr.out.view, VK_NULL_HANDLE}}, dw, dh, &pc, sizeof pc);
    return hdr.out.image;
}

VkImage framegen_made_image(uint32_t* w, uint32_t* h, int* index) {
    if (!fg.recorded || !fg.recorded->made) return VK_NULL_HANDLE;
    *w = fg.w, *h = fg.h;
    // (RX_VK_SHOT_MADE=all: each shot the next of the set's made frames in turn, so a run of shots has every place)
    static const bool all = [] { const char* e = getenv("RX_VK_SHOT_MADE"); return e && !strcmp(e, "all"); }();
    static int turn = 0;
    const int k = all ? turn++ % fg.recorded->made : 0;
    if (index) *index = all ? k : -1;
    return fg.recorded->frame[k].image;
}

// (hookshot) The HUD for AMD FSR 3.1's frame interpolation (#1310, #1420). The game draws its HUD (hearts, minimap, button
// prompts) into the same picture as the scene, after its gameplay upscale, and has no HUD-less colour. The patched
// interpolation used to guess the HUD as the pixels the same in both frames while the scene moved: the see-through parts
// (the minimap, the prompts' backing), the soft edges and anything animated failed the guess and were moved with the
// scene, a HUD that smeared and shook with every turn of the camera. So the picture is copied just after the upscale wrote
// it (framegen_hud_before, from vk_draw.cpp's dispatch of the upscale, whoever drew it); at the present what differs from
// that copy is the HUD, marked in the interpolation source's alpha (shaders/fg_hud.comp), and the interpolation keeps
// those pixels where they are (vk_fsr3fi.h Inputs::hud_mask). A frame without the copy (no gameplay upscale: a menu, a
// movie) keeps the old guess.
// The mark alone still let the HUD smear (#1310 after #1464): the frame before's HUD stayed in the interpolation's previous
// source and was dragged with the scene, and the inpainting spread the HUD's colours into the holes a turn opens beside it.
// So the copy itself, cropped as the frame (hud.hudless), is what FSR 3.1 interpolates: its own HUD-less colour, the HUD in
// neither frame, and this frame's HUD laid over the made frame after, where the presented frame differs from it
// (vk_fsr3fi.h Inputs::present; RX_VK_FSR_FG_HUDLESS=0: the mark as before). The copy only where the HUD is: elsewhere the
// presented frame's own pixels, or FSR 3.1 finds the two different everywhere (the copy is from before the game's last
// pass) and every made frame is the presented one (shaders/fg_hud.comp).
// Laid over whole, the presented pixel still brought this frame's scene with it wherever the HUD lets the scene through
// (the hearts' glow, the item box's shadow, the faint icons, the minimap): a halo of the real frame's background around
// every HUD element of every made frame (#1565). So how see-through each HUD pixel is (T) is learned from frame to frame,
// the HUD staying where it is while the scene behind it moves (shaders/fg_hud_layer.comp: hud.layer, the HUD's own colour
// and T); FSR 3.1 is given the HUD-less colour as its present too, so it lays nothing over its made frames, and the HUD goes
// over each made frame's own scene (shaders/fg_hud_composite.comp); where T is not known, the presented pixel as before
// (RX_VK_FSR_FG_HUD_LAYER=0: everywhere, for an A/B).
namespace {
struct HudState {
    std::mutex mu;   // (framegen_hud_before is called from the replay's workers)
    bool init = false, failed = false;
    Pipe mask;
    // (the picture after the upscale, and as presented: its format and size; two of each, this frame's at k and the frame
    // before's at k ^ 1, for the HUD layer's fit: prev says the frame before's are there and its pair)
    Img before[2], after[2];
    int k = 0;
    bool prev = false;
    Pipe fit_pipe, over_pipe;
    bool layer_failed = false;
    Img fit;     // (the HUD layer's fit, kept from frame to frame: the crop's size, RGBA16F)
    Img layer;   // (the HUD's colour and how see-through it is, the crop's size, RGBA16F: fg_hud_layer.comp)
    bool layered = false;   // (this present's layer made: fsr3fi_frames lays the HUD over the made frames with it)
    uint64_t layers = 0;    // (presents with it: the 600-present line)
    Img source;          // (the interpolation source: the frame's crop, RGBA8, the HUD's alpha 0)
    Img hudless;         // (the same crop of the picture before the HUD, RGBA8: FSR 3.1's HUD-less colour)
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t pw = 0, ph = 0, sw = 0, sh = 0;
    VkImage picture = VK_NULL_HANDLE;   // (the picture copied: framegen_hud_before's, its image)
    const Texture* tex = nullptr;       // (and its texture, at gpu: not read unless texture_alive finds it still there)
    uint64_t gpu = 0;
    bool fresh = false;                 // (copied since the last present)
    bool ready = false;                 // (this present's source made: fsr3fi_frames gives it)
    struct Dead {
        uint64_t frame, value;
        Img img;
    };
    std::vector<Dead> dead;   // (images replaced, freed once the GPU is past them)
    bool told = false;
    uint64_t marked = 0;                // (presents given to FSR 3.1 with the HUD marked: its 600-present line)
} hud;

void hud_retire(Img& i) {
    if (i.image) hud.dead.push_back({texture_frame_no(), vk_retirement_value(), i});
    i = {};
}
void hud_free_dead() {
    const uint64_t now = texture_frame_no(), done = vk_timeline_completed();
    for (size_t i = 0; i < hud.dead.size();) {
        if (now - hud.dead[i].frame > 4 && hud.dead[i].value && done >= hud.dead[i].value) {
            drop_img(hud.dead[i].img);
            hud.dead[i] = hud.dead.back();
            hud.dead.pop_back();
        } else {
            i++;
        }
    }
}
bool hud_image(VkCommandBuffer cmd, Img& i, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage) {
    if (i.image) return true;
    if (!make_img(i, w, h, f, usage)) return false;
    to_general(cmd, i.image);
    return true;
}

/// Whether the HUD goes over FSR 3.1's made frames by its layer (RX_VK_FSR_FG_HUD_LAYER, read once): with the HUD-less
/// colour only (RX_VK_FSR_FG_HUDLESS)
bool hud_layer_wanted() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_FSR_FG_HUD_LAYER");
        return !(e && *e && atoi(e) == 0);
    }();
    return on;
}
bool hudless_wanted(bool native_pq) {
    // The game captures the HUD's linear target before its HDR encode. Treating that colour as PQ
    // corrupts both the HUD-less input and the learned layer. Use the mask with the game's PQ present.
    if (native_pq) return false;
    static const bool on = [] {
        const char* e = getenv("RX_VK_FSR_FG_HUDLESS");
        return !(e && *e && atoi(e) == 0);
    }();
    return on;
}

/// The HUD layer of this present (hud.layer, shaders/fg_hud_layer.comp), from this frame's pictures and the frame before's:
/// how see-through the HUD is where it is known. False: none (the HUD laid over the made frames whole, as presented)
bool hud_layer(VkCommandBuffer cmd, int x0, int y0, uint32_t w, uint32_t h) {
    if (hud.layer_failed || !hud_layer_wanted()) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    if (!hud.fit_pipe.pipe && (!make_pipe(hud.fit_pipe, BUILTIN_SPV_ARGS(FG_HUD_LAYER_COMP_SPV), {S, S, S, S, I, I}, 20) ||
                               !make_pipe(hud.over_pipe, BUILTIN_SPV_ARGS(FG_HUD_COMPOSITE_COMP_SPV), {I, S, S}, 12))) {
        hud.layer_failed = true;
        logf("vk: fsr3fi: the HUD layer's pipelines were not made: the HUD laid over the made frames as presented");
        return false;
    }
    const VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!hud_image(cmd, hud.fit, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, rw) ||
        !hud_image(cmd, hud.layer, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, rw)) {
        hud.layer_failed = true;
        logf("vk: fsr3fi: the HUD layer's images (%ux%u) were not made: the HUD laid over the made frames as presented", w, h);
        return false;
    }
    const int k = hud.k, j = hud.prev ? k ^ 1 : k;   // (no frame before: this frame's bound in its place, not read)
    const int32_t pc[5] = {x0, y0, (int32_t)w, (int32_t)h, hud.prev ? 1 : 0};
    pass(cmd, hud.fit_pipe,
         {{hud.before[k].view, fg.nearest},
          {hud.after[k].view, fg.nearest},
          {hud.before[j].view, fg.nearest},
          {hud.after[j].view, fg.nearest},
          {hud.fit.view, VK_NULL_HANDLE},
          {hud.layer.view, VK_NULL_HANDLE}},
         w, h, pc, sizeof pc);
    if (hud.layers++ == 0)
        logf("vk: fsr3fi: the HUD laid over the made frames by how see-through it is, learned from frame to frame (%ux%u)", w,
             h);
    return true;
}

/// This present's interpolation source with the HUD marked (hud.source), from the game's frame cropped into fg.frames[cur]
/// (at x0, y0 of the presented picture, sw x sh): when the picture was copied after its upscale this frame and is the
/// presented picture's size. False: the frame as it is, the HUD guessed. And its HUD layer (hud_layer: hud.layered)
bool hud_source(VkCommandBuffer cmd, int cur, uint32_t sw, uint32_t sh, int x0, int y0, uint32_t w, uint32_t h, bool pq) {
    std::lock_guard lk(hud.mu);
    hud_free_dead();
    hud.ready = hud.layered = false;
    const bool fresh = hud.fresh;
    hud.fresh = false;
    const int k = hud.k;
    if (!fresh || hud.failed || !hud.picture || hud.pw != sw || hud.ph != sh || !fsr3fi::frames_wanted() ||
        !texture_alive(hud.gpu, hud.tex, hud.picture))   // (the picture destroyed since, or moved: nothing to copy from)
        return hud.prev = false;
    if (!hud.init) {
        hud.init = true;
        const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (!make_pipe(hud.mask, BUILTIN_SPV_ARGS(FG_HUD_COMP_SPV), {S, S, S, I, I}, 20)) {
            hud.failed = true;
            logf("vk: fsr3fi: the HUD mask's pipeline was not made: the HUD guessed");
            return hud.prev = false;
        }
    }
    if (hud.sw != w || hud.sh != h) {   // (the layer's fit is the crop's: started again)
        hud_retire(hud.source), hud_retire(hud.hudless), hud_retire(hud.fit), hud_retire(hud.layer);
        hud.sw = w, hud.sh = h, hud.prev = false;
    }
    const VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!hud_image(cmd, hud.after[k], sw, sh, hud.format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
        !hud_image(cmd, hud.source, w, h, VK_FORMAT_R8G8B8A8_UNORM, rw | VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
        !hud_image(cmd, hud.hudless, w, h, VK_FORMAT_R8G8B8A8_UNORM, rw | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        hud.failed = true;
        logf("vk: fsr3fi: the HUD mask's images (%ux%u) were not made: the HUD guessed", w, h);
        return hud.prev = false;
    }
    full_barrier(cmd);   // (the picture's last writes and reads, and the last readers of this source)
    const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                        {sw, sh, 1}};
    vkCmdCopyImage(cmd, hud.picture, VK_IMAGE_LAYOUT_GENERAL, hud.after[k].image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    full_barrier(cmd);
    // (pq: the native HDR picture, already encoded as the frame is; else linear, sRGB-encoded under the HUD: fg_hud.comp)
    const int32_t pc[5] = {x0, y0, (int32_t)w, (int32_t)h, pq ? 1 : 0};
    pass(cmd, hud.mask,
         {{fg.frames[cur].view, fg.nearest},
          {hud.before[k].view, fg.nearest},
          {hud.after[k].view, fg.nearest},
          {hud.source.view, VK_NULL_HANDLE},
          {hud.hudless.view, VK_NULL_HANDLE}},
         w, h, pc, sizeof pc);
    if (!hud.told) {
        hud.told = true;
        logf("vk: fsr3fi: the HUD marked for it: what the game drew over its picture after the upscale (%ux%u, format %d)",
             sw, sh, (int)hud.format);
    }
    hud.layered = hudless_wanted(pq) && hud_layer(cmd, x0, y0, w, h);
    // (a fit uses the preceding pair only if its layer ran; a Native HDR capture must not feed the next sRGB fit)
    hud.prev = hud.layered;
    hud.k ^= 1;
    hud.marked++;
    return hud.ready = true;
}
} // namespace

void framegen_hud_before(VkCommandBuffer cmd, Texture* picture, uint32_t level) {
    if (!picture || level || !picture->image || picture->samples != VK_SAMPLE_COUNT_1_BIT || picture->levels != 1 ||
        picture->layers != 1 || picture->type != VK_IMAGE_TYPE_2D ||
        (picture->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && picture->format != VK_FORMAT_R16G16B16A16_SFLOAT) ||
        !framegen_on() || !fsr3fi::frames_wanted())
        return;
    std::lock_guard lk(hud.mu);
    if (hud.failed) return;
    if (hud.format != picture->format || hud.pw != picture->width || hud.ph != picture->height) {
        for (int i = 0; i < 2; i++) hud_retire(hud.before[i]), hud_retire(hud.after[i]);
        hud.format = picture->format, hud.pw = picture->width, hud.ph = picture->height, hud.prev = false;
    }
    Img& before = hud.before[hud.k];   // (this frame's: the frame before's, at k ^ 1, is the HUD layer's)
    if (!hud_image(cmd, before, hud.pw, hud.ph, hud.format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        hud.failed = true;
        logf("vk: fsr3fi: the HUD mask's copy (%ux%u) was not made: the HUD guessed", hud.pw, hud.ph);
        return;
    }
    full_barrier(cmd);   // (the upscale's writes; and the last presents' mask and layer passes, which read this copy)
    const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                        {hud.pw, hud.ph, 1}};
    vkCmdCopyImage(cmd, picture->image, VK_IMAGE_LAYOUT_GENERAL, before.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    full_barrier(cmd);   // (before the game draws its HUD into the picture)
    hud.picture = picture->image, hud.tex = picture, hud.gpu = picture->gpu;
    hud.fresh = true;
}

/// (hookshot) A set's made frames by AMD FSR 3.1's frame interpolation (vk_fsr3fi.h), from this frame's colour and the
/// scene's depth and motion (vk_fsr3_inputs.h): made frame i at t[i] between the frame before (0) and this one (1), into
/// set.frame[slot[i]]. Returns count when it made them all, 0 when it did not (no inputs this present, a reset, anything not
/// made), which leaves the set to the interpolation below. Called every present while FSR is asked (count 0 too: a set with
/// nothing to make, or nothing to blend from), so its history is the frame before whatever the set asks for next.
/// sequence: framegen_record's count of its presents (one it did not hand over, a gap among them, makes the next one a
/// reset: vk_fsr3fi.h Inputs::sequence)
int fsr3fi_frames(VkCommandBuffer cmd, OutSet& set, int cur, uint32_t uw, uint32_t uh, uint64_t sequence, bool native_pq,
                  const float* t, const int* slot, int count) {
    // (every 600 presents asked: how many sets FSR made, and how many it left to the interpolation below and why)
    static uint64_t asked = 0, made = 0, no_inputs = 0, taken_before = 0, age[2] = {}, marked_before = 0, layers_before = 0;
    if (++asked % 600 == 0) {
        const uint64_t taken = fsr3_inputs::motions_taken();
        logf("vk: fsr3fi: of the last 600 presents %llu made by FSR 3.1, %llu without its inputs (the renderer's own made "
             "them); %llu frames with the chain's motion (the skinned draws' own); inputs this frame's %llu, the frame before's "
             "%llu; %llu with the HUD marked, %llu with its HUD layer", (unsigned long long)made,
             (unsigned long long)no_inputs, (unsigned long long)(taken - taken_before), (unsigned long long)age[0],
             (unsigned long long)age[1], (unsigned long long)(hud.marked - marked_before),
             (unsigned long long)(hud.layers - layers_before));
        made = no_inputs = 0, taken_before = taken, age[0] = age[1] = 0, marked_before = hud.marked, layers_before = hud.layers;
    }
    fsr3_inputs::Frame f;
    if (!fsr3_inputs::take(f)) return no_inputs++, 0;
    age[texture_frame_no() - f.frame ? 1 : 0]++;   // (how old the inputs are: take gives this frame's or the one before's)
    fsr3fi::Inputs in;
    in.color = fg.frames[cur].view;
    in.color_image = fg.frames[cur].image;
    const bool hudless = hudless_wanted(native_pq);
    // (the HUD layer: FSR 3.1 given the HUD-less colour as its present too, so it lays nothing over the made frames, and the
    // HUD goes over them below by how see-through it is)
    const bool layered = hud.ready && hudless && hud.layered;
    if (hud.ready && hudless) {   // (the picture before the HUD, interpolated; this frame, HUD and all, laid over it)
        in.color = hud.hudless.view, in.color_image = hud.hudless.image;
        in.present = layered ? hud.hudless.view : fg.frames[cur].view;
    } else if (hud.ready) {   // (the same frame, the HUD marked in its alpha: hud_source)
        in.color = hud.source.view, in.color_image = hud.source.image;
        in.hud_mask = true;
    }
    in.color_format = VK_FORMAT_R8G8B8A8_UNORM;   // (ensure_size, for fg.frames)
    in.native_pq = native_pq;
    in.max_luminance = native_pq ? (float)std::clamp(opts.hdr_peak, 400, 10000) : 1.0f;
    in.width = uw, in.height = uh;
    in.depth = f.depth, in.motion = f.motion;
    in.render_width = f.width, in.render_height = f.height;
    in.jitter[0] = f.jitter[0], in.jitter[1] = f.jitter[1];
    in.camera = f.camera;
    in.frame_time_ms = f.frame_ms;
    in.reset = f.reset;
    in.sequence = sequence;
    // (its images in use this present from here, whatever generate makes of it: a reset, its history alone, none made)
    fsr_given = true;
    if (count <= 0) return (void)fsr3fi::generate(cmd, in, t, 0), 0;   // (its history alone)
    if ((int)fsr3fi::generate(cmd, in, t, (uint32_t)count) != count) return 0;
    const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                        {uw, uh, 1}};
    for (int i = 0; i < count; i++) {
        const VkImage image = fsr3fi::made_image((uint32_t)i);
        if (!image) return 0;
        vkCmdCopyImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, set.frame[slot[i]].image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    if (layered) {   // (this frame's HUD over each made frame's own scene: shaders/fg_hud_composite.comp)
        full_barrier(cmd);
        const int32_t pc[3] = {(int32_t)uw, (int32_t)uh, native_pq ? 1 : 0};
        for (int i = 0; i < count; i++)
            pass(cmd, hud.over_pipe,
                 {{set.frame[slot[i]].view, VK_NULL_HANDLE}, {fg.frames[cur].view, fg.nearest}, {hud.layer.view, fg.nearest}},
                 uw, uh, pc, sizeof pc);
    }
    made++;
    static int told = 0;
    if (told < count) {   // (once, and again for a set of more made frames than before)
        told = count;
        logf("vk: fsr3fi: AMD FSR 3.1 makes %d frame(s) between each two of the game's (%ux%u, its depth and motion %ux%u)",
             count, uw, uh, f.width, f.height);
    }
    return count;
}

bool framegen_record(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, const int crop[4], float cover, bool native_pq) {
    const PresentSourceRect source = present_source_rect(sw, sh, crop);
    const int x0 = (int)source.x, y0 = (int)source.y, w = (int)source.w, h = (int)source.h;
    if (!src || w <= 0 || h <= 0 || !init() || !ensure_size(cmd, (uint32_t)w, (uint32_t)h)) return false;
    if (!fg.thread_started) {
        fg.thread_started = true;
        std::thread(display_main).detach();
    }
    const int64_t now = now_ns(), before = fg.last_at;
    const bool gap = fg.last_at && now - fg.last_at > 100000000;   // (a loading hitch: nothing to blend)
    if (fg.last_at && !gap)
        fg.game_ns = fg.game_ns ? (fg.game_ns * 7 + (now - fg.last_at)) / 8 : now - fg.last_at;
    fg.last_at = now;
    OutSet* set;
    {
        std::unique_lock lk(fg.mu);
        fg.cv.wait(lk, [] { return !fg.sets[fg.next_set].busy; });
        set = &fg.sets[fg.next_set];
        set->busy = true;
        fg.next_set = (fg.next_set + 1) % 3;
    }
    // (the multiplier the sets were made for, ensure_size above: the row read again here could be one step up by now, the
    // menu runs on the display thread, and the set's last image would be one that was never made)
    const int cur = fg.cur, prev = cur ^ 1, m = fg.mult;
    const bool auto_rate = opts.framegen == 1;   // (read once, for the same reason)
    const bool blend = fg.have_prev && !gap;
    set->reflex_id = reflex_frame();   // (NVIDIA Reflex: the game frame this set shows, vk_reflex.cpp)
    set->antilag_id = antilag_frame();   // (AMD Anti-Lag 2: the same, vk_antilag.cpp)
    static uint64_t sequence = 0;
    sequence++;
    const uint32_t uw = (uint32_t)w, uh = (uint32_t)h;
    const uint32_t qw = (uw + 3) / 4, qh = (uh + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    full_barrier(cmd);
    // (ultrawide's loading cover, wide_sides: the sides closed in the copy, before the frames between are made from it, so
    // that whichever interpolation makes them treats the sides as it does the middle; the display thread's scale pass then
    // leaves them as they are, wide_cover_in_frame)
    const float edge = cover > 0.0f ? wide_margin() : 0.0f;
    extern std::atomic<bool> present_native_pq;   // (vk_replay.cpp do_present: the game presents its own HDR10 picture, Native)
    const struct { int32_t rect[4]; float x0, cover; int32_t pq; } pc = {{x0, y0, w, h}, edge, edge > 0.0f ? std::min(cover, 1.0f) : 0.0f,
                                                                         present_native_pq.load(std::memory_order_relaxed)};
    pass(cmd, fg.copy, {{src, fg.nearest}, {fg.frames[cur].view, VK_NULL_HANDLE}}, uw, uh, &pc, sizeof pc);
    pass(cmd, fg.luma, {{fg.frames[cur].view, fg.nearest}, {fg.lumas[cur].view, VK_NULL_HANDLE}}, qw, qh);
    // (AMD FSR 3.1's source with the HUD marked, when it is asked: read by fsr3fi_frames below)
    hud_source(cmd, cur, sw, sh, x0, y0, uw, uh, native_pq);
    set->made = 0;
    set->shown = -1;
    set->by_time = false;   // (a reused set: only a fixed multiplier's below sets it)
    if (auto_rate) {   // (Auto: a frame for each of the display's refreshes, at its own time)
        const int mhz = display_refresh_mhz();
        const int64_t period = (int64_t)(1e12 / (mhz > 0 ? mhz : 60000));
        float t[kMaxMult];
        int64_t at[kMaxMult];
        int n;
        if (blend) {
            n = auto_slots(before, now, period, fg.slot_next, t, at, std::min(fg.mult, kMaxMult));
        } else {   // (nothing to blend from: the game's frame, and the timeline starts at it)
            n = 1, t[0] = 1.0f, at[0] = now, fg.slot_next = now + period;
        }
        // (hookshot) AMD FSR 3.1's frame interpolation where it is asked (vk_fsr3fi.h): each made frame at its refresh's
        // place, from the scene's depth and motion. Given every present, with none to make too, so the next set has its
        // history; a set it does not make goes to the renderer's own below. DLSS-G is not asked: it makes a fixed 2x and up
        float made_t[kMaxMult];
        int made_slot[kMaxMult], made_n = 0;
        for (int k = 0; k < n; k++)
            if (t[k] <= 0.999f) made_t[made_n] = t[k], made_slot[made_n++] = k;
        const int fsr_n = blend ? made_n : 0;
        const bool fsr = fsr3fi::frames_wanted() &&
                         fsr3fi_frames(cmd, *set, cur, uw, uh, sequence, native_pq, made_t, made_slot, fsr_n) == fsr_n &&
                         fsr_n > 0;
        bool flow = false;
        for (int k = 0; k < n; k++) {
            set->game[k] = t[k] > 0.999f;
            set->due[k] = at[k] + fg.game_ns;
            if (set->game[k]) {
                const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                    {0, 0, 0}, {uw, uh, 1}};
                vkCmdCopyImage(cmd, fg.frames[cur].image, VK_IMAGE_LAYOUT_GENERAL, set->frame[k].image,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &c);
                continue;
            }
            if (fsr) continue;   // (made above, into this slot)
            if (!flow) {
                flow = true;
                pass(cmd, fg.flow, {{fg.lumas[prev].view, fg.nearest}, {fg.lumas[cur].view, fg.nearest},
                                    {fg.flows[0].view, VK_NULL_HANDLE}}, cw, ch);
                pass(cmd, fg.median, {{fg.flows[0].view, fg.nearest}, {fg.flows[1].view, VK_NULL_HANDLE}}, cw, ch);
            }
            pass(cmd, fg.synth, {{fg.frames[prev].view, fg.linear}, {fg.frames[cur].view, fg.linear},
                                 {fg.flows[1].view, fg.nearest}, {set->frame[k].view, VK_NULL_HANDLE}},
                 uw, uh, &t[k], 4);
        }
        set->shown = n;
        set->made = n > 0 && !set->game[n - 1] ? n : std::max(0, n - 1);   // (for framegen_made_image: any made)
        note_maker(fsr ? FgMaker::Fsr3 : flow ? FgMaker::BuiltIn : FgMaker::None);   // (Auto: DLSS-G is not asked)
        full_barrier(cmd);
        fg.have_prev = true;
        fg.cur = prev;
        fg.recorded = set;
        fg.recorded_at = now;
        return true;
    }
    // (hookshot) NVIDIA DLSS Frame Generation through NGX where it runs and the Frame generation by row asks for it (vk_dlss.cpp):
    // every made frame of the set, from the scene's own depth, motion and camera. A set it does not make goes on below as before
    // (asked every present: turned off, it lets its feature go)
    if (m > 1) {
        VkImage outs[kMaxMult];
        VkImageView views[kMaxMult];
        for (int k = 0; k < m - 1; k++) outs[k] = set->frame[k].image, views[k] = set->frame[k].view;
        set->made = dlss_fg_frames(cmd, fg.frames[cur].image, fg.frames[cur].view, uw, uh, m - 1, outs, views, blend, sequence);
    }
    FgMaker maker = set->made ? FgMaker::Dlss : FgMaker::None;
    // (hookshot) DLSS FG chosen, on a temporal Upscaler, and no picture from it for this set (the game's own menus, movies and
    // loading screens draw no 3D: dlss_fg_holds): the game's frame alone, neither FSR 3.1's frames nor the built-in one's
    const bool held = set->made == 0 && blend && m > 1 && dlss_fg_holds();
    if (held) maker = FgMaker::DlssPaused;
    // (hookshot) the frames this game frame gets: as many as the display shows (fixed_frames), DLSS's own above excepted
    static const bool uncapped = getenv("RX_VK_FRAMEGEN_UNCAPPED") && atoi(getenv("RX_VK_FRAMEGEN_UNCAPPED")) != 0;
    const int mk = uncapped ? m : fixed_frames(m, (display_refresh_mhz() + 500) / 1000, fg.game_ns);
    {
        static int said = -1;
        static int64_t said_at = 0;
        if (mk != said && (said < 0 || now - said_at > 5000000000LL)) {
            if (mk < m)
                logf("vk: frame interpolation: %dx asked, %dx made: the display's %d Hz over the game's %.0f fps (the game keeps its "
                     "rate; RX_VK_FRAMEGEN_UNCAPPED=1: all)", m, mk, (display_refresh_mhz() + 500) / 1000,
                     fg.game_ns > 0 ? 1e9 / (double)fg.game_ns : 0.0);
            else if (said >= 0)
                logf("vk: frame interpolation: %dx made", mk);
            said = mk, said_at = now;
        }
    }
    // (hookshot) Where DLSS-G did not make them, AMD FSR 3.1's frame interpolation where it is asked (vk_fsr3fi.h): the set's
    // made frames at k/m, 1/2 at 2x, 1/3 and 2/3 at 3x. Given every present it is asked, with none to make too (nothing to
    // blend from), so the next set has its history; any it does not make stays with the interpolation below
    if (!held && set->made == 0 && mk > 1 && fsr3fi::frames_wanted()) {
        float made_t[kMaxMult];
        int made_slot[kMaxMult];
        for (int k = 1; k < mk; k++) made_t[k - 1] = (float)k / (float)mk, made_slot[k - 1] = k - 1;
        const int fsr_n = blend ? mk - 1 : 0;
        if (fsr3fi_frames(cmd, *set, cur, uw, uh, sequence, native_pq, made_t, made_slot, fsr_n) == fsr_n && fsr_n > 0)
            set->made = fsr_n, maker = FgMaker::Fsr3;
    }
    if (!held && set->made == 0 && blend && mk > 1) {
        pass(cmd, fg.flow, {{fg.lumas[prev].view, fg.nearest}, {fg.lumas[cur].view, fg.nearest},
                            {fg.flows[0].view, VK_NULL_HANDLE}}, cw, ch);
        pass(cmd, fg.median, {{fg.flows[0].view, fg.nearest}, {fg.flows[1].view, VK_NULL_HANDLE}}, cw, ch);
        for (int k = 1; k < mk; k++) {
            const float t = (float)k / (float)mk;
            pass(cmd, fg.synth, {{fg.frames[prev].view, fg.linear}, {fg.frames[cur].view, fg.linear},
                                 {fg.flows[1].view, fg.nearest}, {set->frame[k - 1].view, VK_NULL_HANDLE}},
                 uw, uh, &t, 4);
        }
        set->made = mk - 1;
        maker = FgMaker::BuiltIn;
    }
    note_maker(maker);
    // (hookshot) The set's times on the game's own timeline (fixed_dues): each frame at its place between the two game
    // frames, its lead (pace_lead) and the GPU's delay (pace_delay) later. display_main shows the set at them
    // (fixed_on_timeline; under FIFO one a refresh at most, fifo_slot). Without a frame before it (the first, a hitch) or a
    // made frame, none
    if (blend && set->made > 0 && fg.game_ns > 0) {
        const int64_t extra = fg.late_ns.load(std::memory_order_relaxed);
        fixed_dues(before, now, set->made, pace_lead(fg.game_ns, set->made) + extra, set->due);
        for (int k = 0; k <= set->made; k++) set->game[k] = k == set->made;
        set->frame_ns = fg.game_ns;
        set->span_ns = now - before;
        set->extra_ns = extra;
        set->by_time = true;
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
    fg.jobs.push_back({fg.recorded, fg.recorded_at, value, now_ns()});
    fg.recorded = nullptr;
    fg.cv.notify_all();
}

/// Each present (vk_replay.cpp): ultrawide's menu setting to the game's side, and the sides to treat now
void wide_sides_tick(float cover) { wide_now = wide_sides(cover); }
// (the reading is of the frame the game works on, which is the presented one or the one after it as its threads fall (measured:
// an opening cover read a present ahead of the picture through one area's entry, and by none or one through another's). So a
// frame's sides are closed by the larger of its reading and the one before: never more open than the middle, at most a frame
// behind it)
float wide_cover_held(float now) {
    static float before = 0.0f;   // (the game's presenting thread's alone)
    const float cover = std::max(now, before);
    before = now;
    return cover;
}
float wide_cover_sample() {
    return wide_cover_held(host && host->wide_cover && wide_possible.load(std::memory_order_relaxed) ? host->wide_cover() : 0.0f);
}

/// (hookshot) The Gameplay tab's Item wheel row is on (HOS_ITEM_WHEEL=0 or 1 over it, as host/oot_wheel.c reads
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


bool menu_fade_wanted() {
    static const bool on = env_on("RX_VK_MENU_FADE");
    return on;
}

void menu_fade_present(VkCommandBuffer cmd, VkImage image, VkImageView storage, uint32_t w, uint32_t h,
                       bool menu, bool world, bool enabled) {
    static menu_fade::Transition transition;
    static Img snapshot;
    static Pipe blend;
    static VkSampler sampler = VK_NULL_HANDLE;
    static uint32_t width = 0, height = 0;
    static bool attempted = false;
    if (!enabled || !image || !storage || !w || !h || uint64_t(w) * h > 2560ull * 1440) {
        transition = {};
        return;
    }
    // Only the replay queue reads the snapshot. Keep one allocation and
    // invalidate it after a size change until the original size returns.
    if (snapshot.image && (w != width || h != height)) {
        transition = {};
        return;
    }
    const int64_t now = now_ns();
    const bool was_armed = transition.armed;
    const auto choice = transition.step(now, menu, world);
    if (transition.armed && !was_armed) logf("vk: menu fade: held the last gameplay frame");
    if (choice.action == menu_fade::Action::Native) return;
    if (!attempted && choice.action == menu_fade::Action::Capture) {
        attempted = true;
        if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return;
        std::vector<uint32_t> code;
        try {
            code = wheel_shader(3);
        } catch (const std::exception& e) {
            logf("vk: menu fade: shader generation failed: %s", e.what());
            return;
        }
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(dev.device, &si, nullptr, &sampler) != VK_SUCCESS) return;
        if (!make_pipe(blend, code.data(), code.size() * sizeof(uint32_t),
                       {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}, 16) ||
            !make_img(snapshot, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            drop_pipe(blend);
            vkDestroySampler(dev.device, sampler, nullptr);
            sampler = VK_NULL_HANDLE;
            return;
        }
        to_general(cmd, snapshot.image);
        width = w, height = h;
        logf("vk: menu fade: GPU snapshot %ux%u", w, h);
    }
    if (!snapshot.image) return;
    full_barrier(cmd);
    if (choice.action == menu_fade::Action::Blend && choice.opacity < 1000) {
        const uint32_t args[4] = {choice.opacity, 0, w, h};
        pass(cmd, blend, {{snapshot.view, sampler}, {storage, VK_NULL_HANDLE}}, w, h, args, sizeof args);
        descriptors_disturbed();
        return;
    }
    if (choice.action == menu_fade::Action::Blend) logf("vk: menu fade: blending into gameplay");
    const bool capture = choice.action == menu_fade::Action::Capture;
    if (capture) transition.captured_at = now;
    const VkImageSubresourceLayers layer{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    const VkImageCopy region{layer, {}, layer, {}, {w, h, 1}};
    vkCmdCopyImage(cmd, capture ? image : snapshot.image, VK_IMAGE_LAYOUT_GENERAL,
                   capture ? snapshot.image : image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    full_barrier(cmd);
}

/// (RX_VK_FG_LATENCY_LOG) Before the swapchain is destroyed: the log's waits for its presents end first. Every time, also
/// before the log's thread has started: its first present's id may be queued for a chain retired meanwhile
void fg_latency_swapchain_gone() {
    std::lock_guard g(lat_sc_mu);
    lat_gone = dev.swapchain_generation + 1;
}

} // namespace mkwvk
