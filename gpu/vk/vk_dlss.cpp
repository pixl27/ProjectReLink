// (relink) NVIDIA DLSS Super Resolution, or AMD FSR 3.1 (vk_fsr.cpp), in place of OoT's gameplay upscale network (vk_dlss.h).
//
// The machinery below is the same for both: the scene's camera, the prep pass, the actor motion, the output pass, the
// jitter's size, the idle release and the status line. Only the step between the prep pass and the output pass differs:
// NVIDIA's NGX evaluates DLSS (RELINK_NGX builds only), or AMD's FidelityFX runtime dispatches FSR's upscaler (every
// build, where amd_fidelityfx_vk.dll loads: vk_fsr.cpp). The comments below say DLSS for either unless they say otherwise.
//
// OoT has no DLSS of its own: its scene is drawn at a lower size and a network of its own (the program
// fd029a317a9a4b94 at 3/4, another for each other share; one dispatch, cooperative matrices) scales it up, from the
// square root of the scene's tonemapped colour, into the frame's B10G11R11 picture (before the HUD). Here that dispatch is replaced:
//  - the scene's camera is taken from the draws into the scene's colour target: their vertex stage's uniform buffer 3
//    holds the target's size at 0x120, this frame's view-projection at 0x30 (row major, clip = M * position; the projection's w row ends -1, 0 at 0xa8). The largest such target in a frame is
//    the scene (reflections and half-size passes have the same layout). The scene's depth is that draw's depth target.
//  - shaders/dlss_prep.comp makes DLSS's inputs at the render size: the colour as the network samples it (LDR), the
//    depth, and motion vectors from the depth reprojected through the two cameras. The game has no motion vectors of its
//    own (no velocity target: the scene pass draws into one B10G11R11 target, and its upscale network samples the colour alone),
//    so that is the camera's motion, and what moves by itself would ghost. The scene camera's sub-pixel jitter
//    (nx2/oot_jitter.py, while DLSS draws) is read back from its projection, given to DLSS and taken off the motion.
//  - actor motion (actor_motion, shaders/dlss_actor.vert): the skinned draws into the scene (vk_draw.cpp's dlss_skin_check:
//    Link, the other characters, the animals, what the game animates with bones) are drawn again over that motion, each
//    vertex skinned with this frame's bones and the frame before's, so their pixels get their own motion. Their poses are
//    copied on the GPU from where the game has them into two buffers that take turns; a pixel is the draw's where its depth is
//    the scene's (within RX_VK_DLSS_ACTOR_TOL steps, default 64). What moves without bones (particles, water, foliage moved in
//    its vertex shader, mesh-shader draws) keeps the camera's motion. RX_VK_DLSS_ACTORS=0: camera motion alone.
//  - DLSS (NGX, Vulkan) scales them to the output's size; shaders/dlss_out.comp squares that back to linear colour
//    into the network's output.
// Anything unexpected (no camera this frame, other sizes, DLSS not there or failing) leaves the network to run, the
// reason in the log when it changes.
//
// RX_VK_DLSS=0 / 1 / 2 / 3 / 4: the game's own, DLSS, DLAA, FSR, FSR native AA over the Insert menu's Upscaler row (0
// also leaves NGX's extensions off). DLSS's runtime (Windows: nvngx_dlss.dll; Linux: libnvidia-ngx-dlss.so.*) is looked for in RX_VK_DLSS_DIR=<folder>
// when set, the game folder's ngx folder, and %LOCALAPPDATA%\ReLink\ngx (Linux: $XDG_DATA_HOME/relink/ngx): NGX takes
// the newest one it finds in them. Windows' paths are wide all the way (GetModuleFileNameW, _wgetenv), so any folder name
// works. The Insert menu's DLSS model row (opts.dlss_model, upscaler_rows.h) chooses DLSS's preset: K by default, M, L, or
// NVIDIA's default (0). RX_VK_DLSS_PRESET=<letter> (or 0) over it.
// RX_VK_DLSS_DEBUG=motion: the motion vectors shown instead of the picture (|x|, |y| over 16 px as red and green, the actors'
// in it); =alternate: every other frame the game's own upscale (a test's reference beside DLSS's). RX_VK_DLSS_LOG=1: NGX's
// own log in ours.
// RX_VK_DLSS_SQUARE=0: the output not squared.
//
// NVIDIA's NGX SDK is not part of ReLink (NVIDIA's license): a local build fetches it (scripts/ngx_sdk.py) and builds with
// RELINK_NGX_SDK, which compiles this file with RELINK_NGX. Without it the NGX parts (under RELINK_NGX) are left out and
// DLSS and DLAA are not offered; FSR still is.
// AMD FSR (RX_VK_DLSS=3, or 4 for its native anti-aliasing, the 3D at full size as with DLAA): the same inputs, given to FFX's
// upscaler with the camera's planes and field of view read from its projection. RX_VK_FSR_SRGB=0: its colour said to be
// linear (by default FFX is told it is gamma-encoded, as it is). RX_VK_FSR_JITTER=flip: the jitter given with the other sign
// (a test of the sign).
// NGX's core is the driver's (Windows: _nvngx.dll, which NVIDIA's loader finds in the driver's own folder; Linux:
// libnvidia-ngx.so). It is never put beside the game: a copy there would be loaded in place of the driver's.
#include "vk_dlss.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlss_actor.h"
#include "dlss_math.h"
#include "upscaler_rows.h"
#include "vk_draw.h"
#include "vk_fsr.h"
#include "vk_menu.h"
#include "vk_paths.h"
#include "vk_state.h"
#include "vk_texture.h"

#ifdef RELINK_NGX
#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_helpers_dlssg_vk.h"   // (relink, 0.7.9.3: DLSS Frame Generation)
#endif

#include "shaders/dlss_actor_frag_spv.h"
#include "shaders/dlss_actor_vert_spv.h"
#include "shaders/dlss_mask_comp_spv.h"
#include "shaders/fg_copy_comp_spv.h"
#include "shaders/dlss_hudless_comp_spv.h"
#include "shaders/dlss_post_comp_spv.h"
#include "shaders/dlss_out_comp_spv.h"
#include "shaders/dlss_out16_comp_spv.h"
#include "shaders/dlss_prep_comp_spv.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

namespace mkwvk {

namespace {
int env_switch(const char* name) {   // (-1: unset)
    const char* e = getenv(name);
    return e && *e ? atoi(e) : -1;
}
// (relink, 0.7.9.3) The Sharpness row with DLSS, DLAA or FSR: on their 3D picture, before the HUD (dlss_out.comp), not on the
// finished picture as it goes to the window (RX_VK_UPSCALER_SHARPEN=0: there, as before). kSharpenGain and kSharpenOver: the
// row's 100 %. MEASURED (19/11, RTX 3060 Ti, the Market's house fronts from the arrival's still camera, pictures at the
// output's own 2048x1152, the mean luma step between neighbours; relink_release sharp_sc6.txt and the ledger): the game's own
// network at 75 % 9.6 to 10.1; DLSS from the same 1536x864 5.9 unsharpened, 8.5 at the row's 50 %, 10.8 to 11.7 at 100 %; FSR 3.1 5.6, 8.1
// and 11.0. So about 80 % looks like the game's own, which is what the row's hint has advised. With the pixel held
// between its neighbours (no overshoot) the measure stopped at 7.2 whatever the gain
bool upscaler_sharpen_on() {
    static const bool on = env_switch("RX_VK_UPSCALER_SHARPEN") != 0;
    return on;
}
static const float kSharpenGain = getenv("RX_VK_UPSCALER_SHARPEN_GAIN") ? (float)atof(getenv("RX_VK_UPSCALER_SHARPEN_GAIN")) : 2.0f;
// (how far past its neighbours a sharpened pixel may go, as a share of their spread, at the row's 100 %: dlss_out.comp)
static const float kSharpenOver = getenv("RX_VK_UPSCALER_SHARPEN_OVER") ? (float)atof(getenv("RX_VK_UPSCALER_SHARPEN_OVER")) : 0.5f;
/// (relink) The Upscaler's values (upscaler_rows.h): which backend, and whether the 3D is drawn at full size
bool is_fsr(int mode) { return mode == upscaler_rows::FSR || mode == upscaler_rows::FSR_AA; }
bool native_aa(int mode) { return mode == upscaler_rows::DLAA || mode == upscaler_rows::FSR_AA; }
/// (relink, 0.7.9.3) No upscaling: no backend, the scene shown as it is (dlss_upscale)
bool is_plain(int mode) { return mode == upscaler_rows::NONE; }
const char* backend_name(int mode) { return is_plain(mode) ? "No upscaling" : is_fsr(mode) ? "AMD FSR" : "NVIDIA DLSS"; }
/// (relink) FSR's view-space units in metres (FFX's viewSpaceToMetersFactor, which scales its depth tolerances): 1, a guess
/// from the scene camera's planes as read in the Market (0.1 to 25041: a near plane of 10 cm, as a game in metres has)
constexpr float kFsrMeters = 1.0f;
}  // namespace

// (what is known of DLSS on this device: -1 not yet, else a DlssWhy. Set once the device is made and when NGX answers)
std::atomic<int> g_why{-1};

DlssWhy dlss_why() {
    if (!dlss_built()) return DLSS_NOT_BUILT;
    static const bool env_off = env_switch("RX_VK_DLSS") == 0;
    if (env_off) return DLSS_ENV_OFF;
    const int w = g_why.load(std::memory_order_relaxed);
    return w < 0 ? DLSS_OK : (DlssWhy)w;
}

int dlss_asked() {
    static const int env = env_switch("RX_VK_DLSS");
    return std::clamp(env >= 0 ? env : opts.dlss, 0, (int)upscaler_rows::kUpscalers - 1);
}

int dlss_mode() {
    // (what is asked for, when it can run here: DLSS and DLAA where dlss_why() lets them, FSR where fsr_why() does)
    const int want = dlss_asked();
    if (is_plain(want)) return want;   // (no backend to ask)
    if (is_fsr(want)) return fsr_why() == FSR_OK ? want : 0;
    if (dlss_why() != DLSS_OK) return 0;
    return want;
}
bool dlss_wanted() { return dlss_mode() != 0; }
bool dlss_full_scale() {
    const int mode = dlss_mode();
    return native_aa(mode) || is_plain(mode);
}

namespace {

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
};

struct Camera {
    float current[16]{};      // (its view-projection)
    float proj[16]{};         // (relink, 0.7.9.3: its projection, the jitter in it; for DLSS Frame Generation's camera)
    float jitter[2]{};        // (the projection's offset of the picture's centre, in clip space: nx2/oot_jitter.py)
    // (relink, for FSR: the near and far planes in the camera's units, read from its projection (far infinite: none), and its
    // vertical field of view in radians)
    float near_z = 0, far_z = 0, fov_y = 0;
    bool infinite = false;
    uint32_t width = 0, height = 0;
    Texture* depth = nullptr;
    uint64_t depth_gen = 0;   // (texture_destroy_generation when taken: a depth texture destroyed since is not used)
    uint64_t frame = 0;       // (texture_frame_no when taken)
    bool valid = false;
};

struct State {
    std::mutex mu;
    bool ngx_tried = false, ngx_ok = false;
    // (DLSS failed, FSR failed: that one's upscale is left to the network from then on, the other can still be chosen. Atomic:
    // dlss_jitter_size reads them on the game's thread, without the lock)
    std::atomic<bool> failed{false}, fsr_failed{false};
    // (relink) The preset and output a feature creation failed for (a key; 0: none): not tried again each frame (a try is tens of ms),
    // but once the preset or output changes, so a failure, out of video memory after quick menu switches say, does not turn DLSS off
    // until a restart. RX_VK_DLSS_RETRY=0: as before, a failed creation fails DLSS for the run
    uint64_t refused = 0;
#ifdef RELINK_NGX
    NVSDK_NGX_Parameter* caps = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
#endif
    uint32_t feature_rw = 0, feature_rh = 0, feature_ow = 0, feature_oh = 0;   // (its largest render size, its output)
    uint32_t min_rw = 0, min_rh = 0;
    unsigned feature_preset = ~0u;   // (the preset it was made with: another one from the menu makes it again)
    uint64_t features_made = 0;      // (how many times: a feature made again in play costs a stall, so each says why)
    // (relink) FSR's upscaler (vk_fsr.cpp): made for the largest render size it can be given and its output, with its flags;
    // the render size its inputs were made for (they grow, never shrink, while it lives), and its jitter's length
    FsrContext* fsr = nullptr;
    uint32_t fsr_rw = 0, fsr_rh = 0, fsr_ow = 0, fsr_oh = 0, fsr_iw = 0, fsr_ih = 0;
    unsigned fsr_flags = 0;
    bool feature_hdr = false;   // (relink, 0.7.9.3: DLSS's feature made for HDR colour: RX_VK_UPSCALE_INPUT=hdr)
    int fsr_phases = 0;
    uint64_t fsr_made = 0;
    std::chrono::steady_clock::time_point fsr_last{};   // (its last dispatch: the frame time it is given)
    int drawn_mode = 0;   // (the Upscaler that last drew: one changed in the menu lets the other's go)
    Image color, depth, motion, output;
    VkDescriptorSetLayout prep_set = VK_NULL_HANDLE, out_set = VK_NULL_HANDLE, out16_set = VK_NULL_HANDLE;
    VkPipelineLayout prep_layout = VK_NULL_HANDLE, out_layout = VK_NULL_HANDLE, out16_layout = VK_NULL_HANDLE;
    VkPipeline prep = VK_NULL_HANDLE, out = VK_NULL_HANDLE, out16 = VK_NULL_HANDLE;
    VkSampler point = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    // (relink, 0.7.9.3) the see-through mask (dlss_mask.comp): the solid picture, the mask, the GPU's flags; its pipeline
    Image opaque, mask, mask_flags, mask_dummy;
    Image delta[2];   // (what the blended draws added, this frame's and the frame before's: by the frame's parity)
    VkDescriptorSetLayout mask_set = VK_NULL_HANDLE;
    VkPipelineLayout mask_layout = VK_NULL_HANDLE;
    VkPipeline mask_pipe = VK_NULL_HANDLE;
    bool mask_failed = false;
    uint64_t mask_frame = 0;          // (the last frame a piece took the solid picture: the upscale's mask is this frame's)
    // (the image the upscale was last given as the scene: other targets of its size and format (the characters' skin, drawn
    // apart) are not it)
    std::atomic<VkImage> scene_image{VK_NULL_HANDLE};
    // (the game's composite of this frame: scene, bloom, exposure and ACES, before its SMAA: dlss_note_post_draw)
    Texture* composite = nullptr;
    uint64_t composite_frame = 0;
    uint64_t composite_gen = 0;
    bool told_composite = false;
    // (relink, 0.7.9.3, the post chain moved after the upscale: dlss_post.comp) the scene's texture (the network's input), its
    // HDR picture kept at the composite's draw, the game's bloom of this frame, the composite's constants
    Texture* scene_tex = nullptr;
    Image hdr, post_out, post_final;
    Texture* bloom = nullptr;
    uint64_t bloom_frame = 0, hdr_frame = 0;
    float post_consts[16]{};   // (the composite's constant buffer: rows 0..11, then 12 13 14 15: -, power, exposure, -)
    VkDescriptorSetLayout post_set = VK_NULL_HANDLE;
    VkPipelineLayout post_layout = VK_NULL_HANDLE;
    VkPipeline post_pipe = VK_NULL_HANDLE;
    bool post_failed = false, post_logged = false;
    uint64_t input_composite = 0, input_network = 0, input_hdr = 0;   // (upscales from each, for the log every 600)
    uint64_t mask_snaps = 0, mask_frames = 0, mask_given = 0, mask_ends = 0, mask_logged = 0;   // (for the status line: solid pictures taken, frames with one)
    Camera cam;               // (this frame's, since the last upscale)
    std::atomic<uint64_t> last_frame{0};   // (the frame DLSS last drew: read on the game's thread too)
    float last_jitter[2]{};   // (its camera's jitter: the frame before's, for the next one)
    float last_vp[16]{};      // (its camera's view-projection: the frame before's)
    std::atomic<uint32_t> jitter_w{0}, jitter_h{0};   // (the scene's size while DLSS draws: the host's jitter's scale)
    std::atomic<float> lod_bias{0.0f};                 // (relink: the textures' mip bias while it draws, upscaler_lod_bias)
    std::atomic<uint32_t> jitter_phases{8};           // (its sequence's length, for DLSS's scale: dlss_jitter_phases)
    std::unordered_map<VkImage, VkImageView> depth_views;
    uint64_t depth_views_gen = 0;
    // (relink, 0.7.9.3: the scene colour's plain 2D views, for the see-through mask: the pass's attachment view reads nothing)
    std::unordered_map<VkImage, VkImageView> color_views;
    uint64_t color_views_gen = 0;
    const char* reason = nullptr;
    int reason_repeats = 0;
    // (actor motion: the skinned draws into the scene since the last upscale; the poses DLSS last drew, by draw: the
    // frame before's, where they were copied to in one of the two pose buffers (each frame's poses go to the other one))
    struct Skin {
        DlssSkin k;
        uint64_t key = 0;
        uint32_t nbones = 0;
    };
    std::vector<Skin> skins;
    struct Pose {
        uint32_t at = 0, nbones = 0;   // (bytes into its buffer: the origin, then the bones from at + 16)
        uint64_t frame = 0;
    };
    std::unordered_map<uint64_t, Pose> poses;
    std::unordered_map<uint64_t, uint32_t> model_bones;   // (by model: the bones its vertices use, 0 when unreadable)
    struct PoseBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation memory = nullptr;
        uint64_t address = 0;
    } pose_buffers[2];
    int pose_at = 0;            // (this frame's)
    VkPipelineLayout actor_layout = VK_NULL_HANDLE;
    VkPipeline actor = VK_NULL_HANDLE;
    bool actor_tried = false;
    struct Upload {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation memory = nullptr;
        void* map = nullptr;
    } uploads[6];               // (the cameras: a ring, one in use stays so for a few frames, as what later() frees)
    int upload_at = 0;
    uint64_t actor_draws = 0, actor_frames = 0, actor_skipped = 0;
    uint64_t actor_one = 0, status_one = 0;   // (the rigidly skinned draws among actor_draws, and at the last status line)
    // (relink, 0.7.9.3: rigid objects, dlss_note_rigid_draw) This frame's rigid draws into the scene with their world matrix;
    // the frame before's matrices by model (12 floats each), with the frame they were taken in
    struct Rigid {
        DlssSkin k;
        uint64_t model = 0;
        float m[12];
    };
    std::vector<Rigid> rigids;
    std::unordered_map<uint64_t, std::vector<float>> rigid_before, rigid_old;   // (kept, their vectors emptied: no allocation a frame)
    std::unordered_set<uint64_t> rigid_same_before, rigid_same_old;            // (model and matrix hashed: an unchanged one found at once)
    uint64_t rigid_frame = 0;
    std::unordered_map<uint64_t, uint64_t> rigid_models;   // (a model's last frame with a rigid draw that moved)
    uint64_t rigid_seen = 0, rigid_moved = 0, status_rigid_seen = 0, status_rigid_moved = 0;
    int64_t actor_ns = 0;       // (the CPU's time in it, since the last status line)
    int64_t rigid_ns = 0;       // (of it, noting the rigid draws: dlss_note_rigid_draw)
    uint64_t status_draws = 0, status_frames = 0;   // (actor_draws and _frames at the last status line: dlss_status)
    uint64_t status_drawn = 0, status_passed = 0;   // (since the last status line: frames DLSS drew, and frames a network
                                                    // of the upscale's shape ran without it (its scene not this frame's))
    float status_jitter[2]{};   // (the last evaluation's jitter, in render pixels, and whether it had motion)
    bool status_motion = false;
    struct Dead { uint64_t frame; std::function<void()> free; };
    std::vector<Dead> dead;   // (what work in flight may still use: freed a few frames on)
    uint64_t asked = 0;       // (the frame an upscale last asked for DLSS: dlss_idle)
    std::atomic<bool> holding{false};   // (a feature, images or frees still to do: dlss_idle has something to let go)
    // (relink, 0.7.9.3) NVIDIA DLSS Frame Generation (dlssg_generate). What NGX said of it (-1 not asked yet, 0 not here, 1
    // here), the most frames it makes between two of the game's, and why not when it is not here (for the menu)
    std::atomic<int> fg_available{-1};
    std::atomic<int> fg_max{1};
    std::atomic<int> fg_why{0};
    bool fg_failed = false;
    uint64_t fg_refused = 0;   // (the size a creation failed for: not tried again each frame)
#ifdef RELINK_NGX
    NVSDK_NGX_Handle* fg_feature = nullptr;
    NVSDK_NGX_Parameter* fg_params = nullptr;
#endif
    uint32_t fg_w = 0, fg_h = 0;
    uint64_t fg_eval_frame = 0;   // (the frame whose inputs it was last given)
    uint64_t fg_made = 0, fg_frames = 0, fg_status_frames = 0, fg_last = 0, fg_missed = 0, fg_status_missed = 0;
    // (each upscale's inputs for it, two frames of them in turn: the depth and motion as DLSS was given them, the camera.
    // The frame generator runs in the present's recording; the next frame's upscale may be recorded first)
    struct FgSlot {
        Image depth, motion;
        Image hudless;          // (relink, 0.7.9.3: the picture before the HUD, as the backbuffer holds it: RX_VK_DLSSG_HUDLESS)
        bool has_hudless = false;
        uint64_t frame = 0;
        uint32_t rw = 0, rh = 0;
        float vp[16], before[16], proj[16];   // (no jitter in any)
        float jitter[2];                      // (in clip space, y up: the game's)
        float near_z = 0, far_z = 0, fov_y = 0;
        bool reset = true, used = true;
    } fg_slots[2];
    int fg_next = 0;
    VkDescriptorSetLayout copy_set = VK_NULL_HANDLE;   // (shaders/fg_copy.comp: a picture into an RGBA8 image)
    VkPipelineLayout copy_layout = VK_NULL_HANDLE;
    VkPipeline copy_pipe = VK_NULL_HANDLE;
    VkDescriptorSetLayout hud_set = VK_NULL_HANDLE;    // (shaders/dlss_hudless.comp: the same, sRGB-encoded)
    VkPipelineLayout hud_layout = VK_NULL_HANDLE;
    VkPipeline hud_pipe = VK_NULL_HANDLE;
    bool copy_failed = false;
};
State& st() {
    static State s;
    return s;
}

// (relink) The textures' mip bias while DLSS or FSR scales the 3D up from rw to ow pixels wide: log2(rw / ow) - 1, as AMD's FSR 2+
// guide gives it (NVIDIA's DLSS guide asks for the same kind of negative bias). Without it the textures were sampled at the mip
// levels of the render size, so the upscaled picture had the render size's texture detail: soft, "muddy", even with DLAA. In
// quarter steps (a sampler per value: the dynamic resolution moves the render size every few frames), between -2.5 and 0.
// RX_VK_UPSCALE_LOD_OFFSET=<float>: another offset than -1; RX_VK_UPSCALE_LOD_BIAS=0: none, as before
float lod_bias_for(uint32_t rw, uint32_t ow) {
    static const bool off = getenv("RX_VK_UPSCALE_LOD_BIAS") && atoi(getenv("RX_VK_UPSCALE_LOD_BIAS")) == 0;
    static const float offset = getenv("RX_VK_UPSCALE_LOD_OFFSET") ? (float)atof(getenv("RX_VK_UPSCALE_LOD_OFFSET")) : -1.0f;
    if (off || !rw || !ow) return 0.0f;
    float bias = std::log2((float)rw / (float)ow) + offset;
    bias = std::clamp(std::round(bias * 4.0f) / 4.0f, -2.5f, 0.0f);
    static int told = 1000;
    const int q = (int)std::lround(bias * 4.0f);
    if (q != told) {
        told = q;
        logf("vk: texture mip bias %.2f while the upscaler draws (%u to %u pixels wide)", (double)bias, rw, ow);
    }
    return bias;
}

bool fail(const char* why) {   // (the network runs: why, when it changes and every 600th time)
    State& s = st();
    if (why != s.reason || ++s.reason_repeats % 600 == 0)
        logf("vk: %s: the game's upscale runs (%s)", backend_name(dlss_mode()), why);
    if (why != s.reason) s.reason = why, s.reason_repeats = 0;
    return false;
}

void later(std::function<void()> f) { st().dead.push_back({texture_frame_no(), std::move(f)}); }
void free_dead(bool all) {
    State& s = st();
    const uint64_t now = texture_frame_no();
    for (size_t i = 0; i < s.dead.size();) {
        if (all || now - s.dead[i].frame > 4) {
            s.dead[i].free();
            s.dead[i] = std::move(s.dead.back());
            s.dead.pop_back();
        } else {
            i++;
        }
    }
}

void drop_image(Image& im) {
    if (!im.image) return;
    const Image old = im;
    later([old] {
        vkDestroyImageView(dev.device, old.view, nullptr);
        destroy_image(old.image, old.memory);
    });
    im = {};
}

bool make_image(VkCommandBuffer cmd, Image& im, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags extra = 0) {
    if (im.image && im.format == format && im.width == w && im.height == h) return true;
    drop_image(im);
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_2D, ic.format = format, ic.extent = {w, h, 1}, ic.mipLevels = 1, ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT, ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | extra;
    if (!create_image(ic, im.image, im.memory)) return im = {}, false;
    VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vc.image = im.image, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = format;
    vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(dev.device, &vc, nullptr, &im.view) != VK_SUCCESS) {
        destroy_image(im.image, im.memory);
        return im = {}, false;
    }
    im.format = format, im.width = w, im.height = h;
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};   // (UNDEFINED -> GENERAL, once)
    b.srcStageMask = VK_PIPELINE_STAGE_2_NONE, b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image, b.subresourceRange = vc.subresourceRange;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
    return true;
}

/// (relink) Images from one layout to another, all of their work before and after (FSR's inputs: vk_fsr.h)
void layouts(VkCommandBuffer cmd, const Image* const* ims, int n, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 b[4]{};
    for (int i = 0; i < n && i < 4; i++) {
        b[i] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b[i].srcStageMask = b[i].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b[i].srcAccessMask = b[i].dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        b[i].oldLayout = from, b[i].newLayout = to;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].image = ims[i]->image, b[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = (uint32_t)std::min(n, 4), di.pImageMemoryBarriers = b;
    vkCmdPipelineBarrier2(cmd, &di);
}

#ifdef RELINK_NGX
NVSDK_NGX_Resource_VK resource(const Image& im, bool write) {
    return NVSDK_NGX_Create_ImageView_Resource_VK(im.view, im.image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, im.format,
                                                  im.width, im.height, write);
}

void NVSDK_CONV ngx_log(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    std::string m(message ? message : "");
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r')) m.pop_back();
    logf("ngx: %s", m.c_str());
}

namespace fs = std::filesystem;
#ifdef _WIN32
/// The game folder (game_module_path's rule: bin\relink-game.exe beside image.bin is the folder above bin), from
/// GetModuleFileNameW: a folder with characters outside the ANSI code page is still the right one
fs::path game_dir() {
    std::wstring exe(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, exe.data(), (DWORD)exe.size());
    if (!n || n >= exe.size()) return {};
    exe.resize(n);
    fs::path dir = fs::path(exe).parent_path();
    if (!_wcsicmp(dir.filename().c_str(), L"bin") && GetFileAttributesW((dir.parent_path() / L"image.bin").c_str()) != INVALID_FILE_ATTRIBUTES)
        dir = dir.parent_path();
    return dir;
}
/// An environment variable as a path (_wgetenv: not through the ANSI code page); empty when unset
fs::path env_path(const wchar_t* name) {
    const wchar_t* e = _wgetenv(name);
    return e && *e ? fs::path(e) : fs::path();
}
/// A path for our log (UTF-8)
std::string shown(const fs::path& p) {
    const std::wstring& w = p.native();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n : 0, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
/// A path for NGX, which takes wide strings: Windows' own
std::wstring wide(const fs::path& p) { return p.native(); }
#else
fs::path game_dir() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    return fs::path(std::string(exe, n)).parent_path();
}
fs::path env_path(const char* name) {
    const char* e = getenv(name);
    return e && *e ? fs::path(e) : fs::path();
}
std::string shown(const fs::path& p) { return p.native(); }
std::wstring wide(const fs::path& p) {
    const std::string& s = p.native();
    return std::wstring(s.begin(), s.end());
}
#endif

#ifdef _WIN32
constexpr const char* kRuntime = "nvngx_dlss.dll";   // (DLSS's runtime, as scripts/ngx_sdk.py fetches it)
constexpr const fs::path::value_type* kRuntimeName = L"nvngx_dlss.dll";
constexpr const fs::path::value_type* kFgRuntimeName = L"nvngx_dlssg.dll";   // (DLSS Frame Generation's, Windows only)
#else
constexpr const fs::path::value_type* kFgRuntimeName = "libnvidia-ngx-dlssg.so";
constexpr const char* kRuntime = "libnvidia-ngx-dlss.so";
constexpr const fs::path::value_type* kRuntimeName = "libnvidia-ngx-dlss.so";
#endif

/// (relink, 0.7.9.3) What NGX says of DLSS Frame Generation on this device, once, from its capabilities: s.fg_available,
/// fg_max (made frames between two of the game's: 1 on the RTX 40 series, up to 3 on the RTX 50), fg_why, logged
void fg_caps(NVSDK_NGX_Parameter* caps, const std::vector<fs::path>& dirs) {
    State& s = st();
    int available = 0, needs_driver = 0, init = 0;
    unsigned major = 0, minor = 0, most = 0;
    NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_Available, &available);
    NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &needs_driver);
    NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, &init);
    NVSDK_NGX_Parameter_GetUI(caps, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor, &major);
    NVSDK_NGX_Parameter_GetUI(caps, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor, &minor);
    NVSDK_NGX_Parameter_GetUI(caps, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &most);
    bool runtime = false;
    std::error_code ec;
    for (const auto& d : dirs)
        for (const auto& f : fs::directory_iterator(d, ec)) runtime |= f.path().filename().native().rfind(kFgRuntimeName, 0) == 0;
    s.fg_max = (int)std::clamp(most, 1u, 7u);
    s.fg_why = available ? DLSSG_OK : needs_driver ? DLSSG_NEEDS_DRIVER : !runtime ? DLSSG_NO_RUNTIME : DLSSG_NEEDS_RTX40;
    s.fg_available = available ? 1 : 0;
    if (available)
        logf("vk: NVIDIA DLSS Frame Generation: available, up to %u made frames between two of the game's", std::max(most, 1u));
    else if (needs_driver)
        logf("vk: NVIDIA DLSS Frame Generation: needs NVIDIA's driver %u.%u or newer", major, minor);
    else
        logf("vk: NVIDIA DLSS Frame Generation: not available here (%#x)%s", (unsigned)init,
             runtime ? ": it needs an RTX 40 series GPU or later" : ": no " "nvngx_dlssg.dll beside DLSS's runtime");
}

/// NGX on this device, once: false when DLSS cannot run here (the reason logged)
// (a project id of ReLink's own: NGX wants one with the custom engine type, any GUID)
constexpr const char* kProjectId = "7b2f5d0c-6a1e-4c39-9f84-3d5e2a1b0c96";

/// Where DLSS's runtime (kRuntime) is looked for (NGX takes the newest it finds); its Frame Generation's (kFgRuntimeName) too
std::vector<fs::path> ngx_dirs() {
    std::vector<fs::path> dirs;
#ifdef _WIN32
    if (fs::path e = env_path(L"RX_VK_DLSS_DIR"); !e.empty()) dirs.push_back(e);
    dirs.push_back(game_dir() / L"ngx");
    if (fs::path l = env_path(L"LOCALAPPDATA"); !l.empty()) dirs.push_back(l / L"ReLink" / L"ngx");
#else
    if (fs::path e = env_path("RX_VK_DLSS_DIR"); !e.empty()) dirs.push_back(e);
    dirs.push_back(game_dir() / "ngx");
    if (fs::path x = env_path("XDG_DATA_HOME"); !x.empty()) dirs.push_back(x / "relink" / "ngx");
    else if (fs::path h = env_path("HOME"); !h.empty()) dirs.push_back(h / ".local" / "share" / "relink" / "ngx");
#endif
    return dirs;
}

bool ngx_init() {
    State& s = st();
    if (s.ngx_tried) return s.ngx_ok;
    s.ngx_tried = true;
    // where DLSS's runtime (kRuntime) is looked for (NGX takes the newest it finds), and its own files go
    const std::vector<fs::path> dirs = ngx_dirs();
    std::vector<std::wstring> wdirs;
    for (const auto& d : dirs) wdirs.push_back(wide(d));
    std::vector<const wchar_t*> paths;
    for (const auto& d : wdirs) paths.push_back(d.c_str());
    const fs::path data = game_dir() / "ngx";
    std::error_code made;   // (not the throwing overload: a folder that cannot be made, read-only or so, must not end the game)
    fs::create_directories(data, made);   // (NGX's logs and its own files: the game folder's, not /usr/share)
    const std::wstring wdata = wide(data);
    NVSDK_NGX_FeatureCommonInfo info{};
    info.PathListInfo.Path = paths.data();
    info.PathListInfo.Length = (unsigned)paths.size();
    static const bool verbose = env_switch("RX_VK_DLSS_LOG") > 0;
    info.LoggingInfo.LoggingCallback = verbose ? ngx_log : nullptr;
    info.LoggingInfo.MinimumLoggingLevel = verbose ? NVSDK_NGX_LOGGING_LEVEL_ON : NVSDK_NGX_LOGGING_LEVEL_OFF;
    // (NGX also writes its own log files into the data folder; with RX_VK_DLSS_LOG=1 its lines go to ours instead)
    info.LoggingInfo.DisableOtherLoggingSinks = verbose;
    // (a project id of ReLink's own: NGX wants one with the custom engine type, any GUID)
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                                                              "1.0", wdata.c_str(), dev.instance, dev.physical, dev.device,
                                                              instance_proc(), vkGetDeviceProcAddr, &info);
    if (NVSDK_NGX_FAILED(r)) {
        logf("vk: NVIDIA DLSS: NGX did not start (%#x): not an NVIDIA GPU, or its driver has no NGX", (unsigned)r);
        g_why = DLSS_NEEDS_RTX;
        return false;
    }
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&s.caps)) || !s.caps) {
        logf("vk: NVIDIA DLSS: NGX has no capabilities for this device");
        g_why = DLSS_NEEDS_RTX;
        return false;
    }
    // (NGX's C functions, not NVSDK_NGX_Parameter's virtual ones: its object comes from NVIDIA's MSVC-built library, whose
    // vtable order is not the Itanium ABI's that llvm-mingw calls with. The overloads only lined up by luck)
    fg_caps(s.caps, dirs);
    int available = 0, needs_driver = 0;
    unsigned major = 0, minor = 0;
    NVSDK_NGX_Parameter_GetI(s.caps, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    NVSDK_NGX_Parameter_GetI(s.caps, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
    NVSDK_NGX_Parameter_GetUI(s.caps, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &major);
    NVSDK_NGX_Parameter_GetUI(s.caps, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minor);
    std::string where;
    for (const auto& d : dirs) where += (where.empty() ? "" : ", ") + shown(d);
    if (!available) {
        // (for the menu: a runtime found in none of the folders is the likely reason; with one there, the GPU (a GTX))
        int init = 0;
        NVSDK_NGX_Parameter_GetI(s.caps, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &init);
        bool runtime = false;
        std::error_code ec;
        for (const auto& d : dirs)
            for (const auto& f : fs::directory_iterator(d, ec))
                runtime |= f.path().filename().native().rfind(kRuntimeName, 0) == 0;
        if (needs_driver) logf("vk: NVIDIA DLSS: needs NVIDIA's driver %u.%u or newer", major, minor);
        else if (!runtime) logf("vk: NVIDIA DLSS: not available: no %s in %s (scripts/ngx_sdk.py fetches it)", kRuntime, where.c_str());
        else logf("vk: NVIDIA DLSS: not available on this GPU (%#x): DLSS needs an NVIDIA RTX GPU", (unsigned)init);
        g_why = needs_driver ? DLSS_NEEDS_DRIVER : !runtime ? DLSS_NO_RUNTIME : DLSS_NEEDS_RTX;
        return false;
    }
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&s.params)) || !s.params) return false;
    logf("vk: NVIDIA DLSS: NGX started (DLSS's runtime: the newest %s in %s)", kRuntime, where.c_str());
    s.ngx_ok = true;
    g_why = DLSS_OK;
    return true;
}
#else
bool ngx_init() { return false; }   // (no NGX in this build: dlss_mode never asks for DLSS, dlss_why says it is not built)
#endif

/// One compute pipeline with push descriptors; what it made is destroyed again when a step fails
bool make_pipeline(const uint32_t* code, size_t size, const VkDescriptorType* types, uint32_t n, uint32_t push,
                   VkDescriptorSetLayout& set, VkPipelineLayout& layout, VkPipeline& pipe) {
    const auto undo = [&] {
        if (layout) vkDestroyPipelineLayout(dev.device, layout, nullptr);
        if (set) vkDestroyDescriptorSetLayout(dev.device, set, nullptr);
        set = VK_NULL_HANDLE, layout = VK_NULL_HANDLE, pipe = VK_NULL_HANDLE;
        return false;
    };
    VkDescriptorSetLayoutBinding b[8]{};
    for (uint32_t i = 0; i < n; i++) b[i] = {i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    li.bindingCount = n, li.pBindings = b;
    if (vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &set) != VK_SUCCESS) return set = VK_NULL_HANDLE, undo();
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1, pi.pSetLayouts = &set, pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(dev.device, &pi, nullptr, &layout) != VK_SUCCESS) return layout = VK_NULL_HANDLE, undo();
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = size, mi.pCode = code;
    VkShaderModule m{};
    if (vkCreateShaderModule(dev.device, &mi, nullptr, &m) != VK_SUCCESS) return undo();
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    ci.layout = layout;
    const bool ok = vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe) == VK_SUCCESS;
    vkDestroyShaderModule(dev.device, m, nullptr);
    return ok || undo();
}

/// The two passes' pipelines and the samplers, once (dlss_upscale stops trying when this fails)
bool make_pipelines() {
    State& s = st();
    if (s.prep && s.out && s.out16 && s.point && s.linear) return true;
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const VkDescriptorType prep_types[5] = {S, S, I, I, I}, out_types[2] = {S, I};
    if (!s.prep && !make_pipeline(DLSS_PREP_COMP_SPV, sizeof DLSS_PREP_COMP_SPV, prep_types, 5, 84, s.prep_set, s.prep_layout, s.prep))
        return false;
    if (!s.out && !make_pipeline(DLSS_OUT_COMP_SPV, sizeof DLSS_OUT_COMP_SPV, out_types, 2, 20, s.out_set, s.out_layout, s.out))
        return false;
    // (relink, Part B) dlss_out.comp's RGBA16F twin, for the PsychoV35 picture: made beside it, chosen by the picture's format
    if (!s.out16 &&
        !make_pipeline(DLSS_OUT16_COMP_SPV, sizeof DLSS_OUT16_COMP_SPV, out_types, 2, 20, s.out16_set, s.out16_layout, s.out16))
        return false;
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (!s.point && vkCreateSampler(dev.device, &si, nullptr, &s.point) != VK_SUCCESS) return s.point = VK_NULL_HANDLE, false;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    if (!s.linear && vkCreateSampler(dev.device, &si, nullptr, &s.linear) != VK_SUCCESS) return s.linear = VK_NULL_HANDLE, false;
    return true;
}

}  // namespace

// ---- (relink, 0.7.9.3) the see-through mask (shaders/dlss_mask.comp) ----
// DLSS and FSR were told nothing of the scene's blended draws (fire, smoke, particles, water, glass): they took this frame's
// colour there as the surface under them, with that surface's motion, and the history trailed them. The game draws them after
// its solid draws (a traced Market frame: ~1500 solid ones, then the blended ones mixed with some that write no depth): the
// solid picture is taken before the first blended draw into the scene (the image the upscale is given, not the characters'
// skin drawn apart at the same size), what the blended draws added is taken at the end of each piece of the frame that drew
// them (the GPU keeps the first picture and the last difference: dlss_mask.comp), and at the upscale the mask is where that
// changed since the frame before, followed by the motion: Navi, flames, sparks, flowing water; not fog or haze over still
// houses, which the history holds well (a mask of everything they add covered the Market's far castle). DLSS gets it as its
// bias towards the current colour, FSR as its reactive mask. RX_VK_DLSS_MASK=0: off (on by default: no frame
// rate cost measured at the Market, 83.6 / 83.8 fps with it, 83.8 without); RX_VK_DLSS_MASK_LO / _HI: the change (as the eye weighs it, 0 to 1) from which the mask starts and where it is
// full (0.04, 0.2); _MOST: its top (0.9); RX_VK_DLSS_DEBUG=mask: the mask shown in the picture (magenta)
namespace {
thread_local uint64_t tl_mask_frame = 0;   // (the frame this recording thread took the solid picture in)
bool mask_on() {
    static const bool on = env_switch("RX_VK_DLSS_MASK") != 0;
    return on;
}
float env_float(const char* name, float fallback) {
    const char* e = getenv(name);
    return e && *e ? (float)atof(e) : fallback;
}
bool make_mask_pipeline(State& s) {
    if (s.mask_pipe) return true;
    if (s.mask_failed || !dev.push_descriptor || !vkCmdPushDescriptorSetKHR || !make_pipelines()) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const VkDescriptorType types[6] = {S, I, I, I, I, I};
    if (make_pipeline(DLSS_MASK_COMP_SPV, sizeof DLSS_MASK_COMP_SPV, types, 6, 32, s.mask_set, s.mask_layout, s.mask_pipe))
        return true;
    s.mask_failed = true;
    logf("vk: the upscalers' see-through mask: its shader was not made: none");
    return false;
}
/// A plain 2D view of a colour texture's first level and layer, for sampling (made once per image; under s.mu)
VkImageView scene_color_view(State& s, Texture* t) {
    const uint64_t gen = texture_destroy_generation();
    if (gen != s.color_views_gen) {   // (a texture was destroyed: its image's view may be stale)
        auto old = std::move(s.color_views);
        s.color_views.clear();
        later([old] {
            for (const auto& [image, view] : old) vkDestroyImageView(dev.device, view, nullptr);
        });
        s.color_views_gen = gen;
    }
    if (auto it = s.color_views.find(t->image); it != s.color_views.end()) return it->second;
    VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vc.image = t->image, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = t->format;
    vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(dev.device, &vc, nullptr, &v) != VK_SUCCESS) v = VK_NULL_HANDLE;
    s.color_views[t->image] = v;
    static std::atomic<int> told{0};
    if (told++ < 4) logf("vk: the upscalers: a plain view of a picture %ux%u, format %d, %u levels, %u layers: its view %s", t->width,
                         t->height, (int)t->format, t->levels, t->layers, v ? "made" : "NOT made");
    return v;
}
/// The flags image (4 x 1), made and cleared once; and a 1 x 1 stand-in for the motion where a step reads none
bool make_mask_flags(VkCommandBuffer cmd, State& s, bool clear) {
    if (!s.mask_dummy.image && !make_image(cmd, s.mask_dummy, VK_FORMAT_R16G16_SFLOAT, 1, 1)) return false;
    if (s.mask_flags.image && !clear) return true;
    // (cleared too when the images were made again at another size, the 3D's (Render resolution Auto): what the flags say of
    // the frame before is then of images that are gone)
    if (!s.mask_flags.image && !make_image(cmd, s.mask_flags, VK_FORMAT_R32_UINT, 4, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return false;
    VkClearColorValue v{};
    v.uint32[0] = v.uint32[1] = v.uint32[2] = v.uint32[3] = 0xffffffffu;
    const VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    full_barrier(cmd);
    vkCmdClearColorImage(cmd, s.mask_flags.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
    full_barrier(cmd);
    return true;
}
/// One step of dlss_mask.comp (its modes there): scene, the sampled view (the scene's colour, or the frame before's difference);
/// opaque, delta and motion its images (none: the solid picture, this frame's difference, the stand-in)
void mask_pass(VkCommandBuffer cmd, State& s, int mode, int which, VkImageView scene, VkImageView opaque, uint32_t w, uint32_t h,
               VkImageView delta = VK_NULL_HANDLE, VkImageView motion = VK_NULL_HANDLE) {
    static const float lo = env_float("RX_VK_DLSS_MASK_LO", 0.04f), hi = env_float("RX_VK_DLSS_MASK_HI", 0.2f),
                       most = env_float("RX_VK_DLSS_MASK_MOST", 0.9f);
    const struct {
        int32_t size[2];
        uint32_t frame;
        int32_t mode, which;
        float lo, hi, most;
    } pc{{(int32_t)w, (int32_t)h}, (uint32_t)texture_frame_no(), mode, which, lo, hi, most};
    const VkImageView dv = delta ? delta : s.delta[texture_frame_no() & 1].view;
    const VkDescriptorImageInfo info[6] = {{s.linear, scene ? scene : s.opaque.view, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, opaque ? opaque : s.opaque.view, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, s.mask.view, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, s.mask_flags.view, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, dv, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, motion ? motion : s.mask_dummy.view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[6]{};
    for (uint32_t i = 0; i < 6; i++) {
        wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wr[i].dstBinding = i, wr[i].descriptorCount = 1, wr[i].pImageInfo = &info[i];
        wr[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.mask_pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.mask_layout, 0, 6, wr);
    vkCmdPushConstants(cmd, s.mask_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    if (mode == 3) vkCmdDispatch(cmd, 1, 1, 1);
    else vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
}
/// The scene's mask is wanted now: DLSS or FSR drew lately, and this target is the scene's size
bool mask_wanted(State& s, Texture* t, uint32_t w, uint32_t h) {
    if (!mask_on() || !t || t->image != s.scene_image.load(std::memory_order_relaxed)) return false;
    const int mode = dlss_mode();
    if (!mode || is_plain(mode)) return false;
    const uint64_t last = s.last_frame.load(std::memory_order_relaxed);
    if (!last || texture_frame_no() - last > 30) return false;
    std::lock_guard lk(s.mu);
    return s.cam.width == w && s.cam.height == h;
}
}  // namespace

void dlss_scene_blended(VkCommandBuffer cmd, Texture* color0, VkImageView view, uint32_t w, uint32_t h) {
    const uint64_t now = texture_frame_no();
    if (tl_mask_frame == now || !color0 || !view) return;
    State& s = st();
    if (!mask_wanted(s, color0, w, h)) return;
    end_pass(cmd);   // (vk_draw.cpp's: the draw opens it again)
    std::lock_guard lk(s.mu);
    if (!make_mask_pipeline(s)) return;
    const bool resized = s.delta[0].width != w || s.delta[0].height != h;
    if (!make_image(cmd, s.opaque, VK_FORMAT_R16G16B16A16_SFLOAT, w, h) || !make_image(cmd, s.mask, VK_FORMAT_R32_SFLOAT, w, h) ||
        !make_image(cmd, s.delta[0], VK_FORMAT_R16G16B16A16_SFLOAT, w, h) ||
        !make_image(cmd, s.delta[1], VK_FORMAT_R16G16B16A16_SFLOAT, w, h) || !make_mask_flags(cmd, s, resized))
        return;
    const VkImageView sv = scene_color_view(s, color0);
    if (!sv || color0->width != w || color0->height != h) return;
    tl_mask_frame = now;
    full_barrier(cmd);   // (the solid draws, written)
    mask_pass(cmd, s, 0, 0, sv, s.opaque.view, w, h);
    full_barrier(cmd);
    mask_pass(cmd, s, 3, 0, VK_NULL_HANDLE, s.opaque.view, w, h);
    full_barrier(cmd);
    descriptors_disturbed();
    if (s.mask_frame != now) s.mask_frames++;
    s.mask_frame = now, s.mask_snaps++;
}

// ---- (relink, 0.7.9.3) the game's post chain (relink_release upscaler_best_TASKS.md, its map) ----
// After the scene and its bloom, one full-screen draw (program ab3681946f93594a) makes the composite: the scene through a
// colour matrix, plus the bloom, times the exposure, through ACES, in linear colour. The game's SMAA (26fe0c70ab9fc547 edges,
// 685adb41edc19de7 weights, 6637ac1d052d22d4 blending) then blends it and writes its square root into the scene's image, the
// network's input, which the upscalers were given. SMAA before a temporal upscaler softens every edge it finds, and DLSS and
// FSR do their own anti-aliasing: they are given the composite instead (its square root, as the network's input is encoded).
// Measured in the Market at 1920x1080, the mean luma step between neighbours (the 3D drawn at 3840x2160 and scaled down: 7.3):
// DLSS unsharpened 5.8 from the network's input, 6.9 from the composite; FSR 5.8 and 6.4; and the colours nearer the
// reference's (mean 106.2 85.2 88.9 against 105.9 84.9 88.8; 103.3 82.0 85.7 after SMAA). RX_VK_UPSCALE_INPUT=network: as
// before. RX_VK_POST_COMPOSITE=<key>: another composite program
namespace {
uint64_t composite_key() {
    static const uint64_t k = getenv("RX_VK_POST_COMPOSITE") ? strtoull(getenv("RX_VK_POST_COMPOSITE"), nullptr, 16)
                                                              : 0xab3681946f93594aull;
    return k;
}
int upscale_input() {   // (1: the composite before SMAA, by default; 0: the network's input, as before; 2: the HDR scene)
    static const int v = [] {
        const char* e = getenv("RX_VK_UPSCALE_INPUT");
        return e && !strcmp(e, "network") ? 0 : e && !strcmp(e, "hdr") ? 2 : 1;
    }();
    return v;
}
}  // namespace

namespace {
uint64_t bloom_key() {   // (the bloom chain's upsampling draw: its last of a frame is the bloom the composite adds)
    static const uint64_t k = getenv("RX_VK_POST_BLOOM") ? strtoull(getenv("RX_VK_POST_BLOOM"), nullptr, 16) : 0x527f93dcd5945f3full;
    return k;
}
/// RX_VK_POST_TEST=ours / diff / swap: the composite made by dlss_post.comp at the render size from the HDR scene kept at the
/// game's composite, given to the upscaler instead of the game's (ours), or the difference with the game's times 16 (diff;
/// swap: with the scene and bloom swapped). A test that dlss_post.comp is the game's composite: diff is black where it is
int post_test() {
    static const int v = [] {
        const char* e = getenv("RX_VK_POST_TEST");
        return !e ? -1 : !strcmp(e, "ours") ? 0 : !strcmp(e, "diff") ? 1 : !strcmp(e, "swap") ? 2 : -1;
    }();
    return v;
}
bool make_post_pipeline(State& s) {
    if (s.post_pipe) return true;
    if (s.post_failed || !dev.push_descriptor || !vkCmdPushDescriptorSetKHR || !make_pipelines()) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const VkDescriptorType types[4] = {S, S, S, I};
    if (make_pipeline(DLSS_POST_COMP_SPV, sizeof DLSS_POST_COMP_SPV, types, 4, 80, s.post_set, s.post_layout, s.post_pipe)) return true;
    s.post_failed = true;
    logf("vk: the upscalers' composite: its shader was not made");
    return false;
}
/// One dlss_post.comp pass (its modes there) at w x h into dst
void post_pass(VkCommandBuffer cmd, State& s, int mode, VkImageView scene, VkImageView bloom, VkImageView game, VkImageView dst,
               uint32_t w, uint32_t h) {
    struct {
        float row[12];
        float power, exposure;
        int32_t size[2];
        int32_t mode;
    } pc{};
    std::memcpy(pc.row, s.post_consts, sizeof pc.row);
    pc.power = s.post_consts[13], pc.exposure = s.post_consts[14];
    pc.size[0] = (int32_t)w, pc.size[1] = (int32_t)h, pc.mode = mode;
    const VkDescriptorImageInfo info[4] = {{s.point, scene, VK_IMAGE_LAYOUT_GENERAL},
                                           {s.linear, bloom ? bloom : scene, VK_IMAGE_LAYOUT_GENERAL},
                                           {s.point, game ? game : scene, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, dst, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[4]{};
    for (uint32_t i = 0; i < 4; i++) {
        wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wr[i].dstBinding = i, wr[i].descriptorCount = 1, wr[i].pImageInfo = &info[i];
        wr[i].descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.post_pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.post_layout, 0, 4, wr);
    vkCmdPushConstants(cmd, s.post_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
}
}  // namespace

void dlss_note_post_draw(VkCommandBuffer cmd, uint64_t key, Texture* color0, uint32_t w, uint32_t h, const NvnState& ns) {
    // (relink, Part B) the composite and bloom are RGBA16F while PsychoV35 is on, B10G11R11 otherwise (hos_scene_rgba16f)
    const bool fmt_ok = color0->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
                        (scene_tonemap_on() && color0->format == VK_FORMAT_R16G16B16A16_SFLOAT);
    if ((key != composite_key() && key != bloom_key()) || !color0 || !fmt_ok) return;
    State& s = st();
    const uint64_t now = texture_frame_no();
    if (key == bloom_key()) {
        std::lock_guard lk(s.mu);
        s.bloom = color0, s.bloom_frame = now;
        return;
    }
    {
        std::lock_guard lk(s.mu);
        s.composite = color0, s.composite_frame = now, s.composite_gen = texture_destroy_generation();
        // (its constants: the fragment stage's buffer 0, which its program reads as c3)
        float c[16]{};
        const nvn_p_bind_buf& b = ns.ubo[1][0];
        if (b.addr && b.size >= sizeof c && memory_read_as_submitted(b.addr, c, sizeof c)) std::memcpy(s.post_consts, c, sizeof c);
        if (!s.post_logged && post_test() >= 0) {
            s.post_logged = true;
            for (int stage = 0; stage < 2; stage++)
                for (int u = 0; u < 4; u++) {
                    const nvn_p_bind_buf& q = ns.ubo[stage][u];
                    float f[16]{};
                    if (!q.addr || !memory_read_as_submitted(q.addr, f, std::min<uint64_t>(sizeof f, q.size))) continue;
                    logf("vk: the composite's buffer %d of stage %d (%llu bytes): %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f | %.4f %.4f "
                         "%.4f %.4f | %.4f %.4f %.4f %.4f", u, stage, (unsigned long long)q.size, f[0], f[1], f[2], f[3], f[4], f[5],
                         f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
                }
        }
    }
    if (post_test() < 0 && upscale_input() != 2) return;
    // (the HDR scene, kept: the game's SMAA writes its picture over the scene's image before the upscale)
    Texture* scene = nullptr;
    {
        std::lock_guard lk(s.mu);
        scene = s.scene_tex;
        if (!scene || scene->width != w || scene->height != h || !make_post_pipeline(s)) return;
    }
    end_pass(cmd);   // (vk_draw.cpp's: the draw opens it again)
    std::lock_guard lk(s.mu);
    const VkImageView sv = scene_color_view(s, scene);
    if (!sv || !make_image(cmd, s.hdr, VK_FORMAT_R16G16B16A16_SFLOAT, w, h)) return;
    full_barrier(cmd);
    post_pass(cmd, s, 3, sv, VK_NULL_HANDLE, VK_NULL_HANDLE, s.hdr.view, w, h);
    full_barrier(cmd);
    descriptors_disturbed();
    s.hdr_frame = now;
}

void dlss_scene_pass_end(VkCommandBuffer cmd, Texture* color0, VkImageView view, uint32_t w, uint32_t h) {
    if (tl_mask_frame != texture_frame_no() || !color0 || !view) return;   // (this thread took no solid picture this frame)
    State& s = st();
    if (color0->image != s.scene_image.load(std::memory_order_relaxed)) return;
    std::lock_guard lk(s.mu);
    if (!s.mask_pipe || s.mask.width != w || s.mask.height != h || !s.mask_flags.image) return;
    const VkImageView sv = scene_color_view(s, color0);
    if (!sv || color0->width != w || color0->height != h) return;
    full_barrier(cmd);   // (the blended draws, written)
    const uint64_t now = texture_frame_no();
    if (s.delta[now & 1].width != w || s.delta[now & 1].height != h) return;
    mask_pass(cmd, s, 1, 0, sv, s.opaque.view, w, h, s.delta[now & 1].view);
    full_barrier(cmd);
    mask_pass(cmd, s, 3, 1, VK_NULL_HANDLE, s.opaque.view, w, h);
    mask_pass(cmd, s, 3, 2 + (int)(now & 1), VK_NULL_HANDLE, s.opaque.view, w, h);
    full_barrier(cmd);
    descriptors_disturbed();
    s.mask_ends++;
}

namespace {
/// A view of a depth texture's depth alone, for sampling (made once per image)
VkImageView depth_view(Texture* t) {
    State& s = st();
    const uint64_t gen = texture_destroy_generation();
    if (gen != s.depth_views_gen) {   // (a texture was destroyed: its image's view may be stale)
        auto old = std::move(s.depth_views);
        s.depth_views.clear();
        later([old] {
            for (const auto& [image, view] : old) vkDestroyImageView(dev.device, view, nullptr);
        });
        s.depth_views_gen = gen;
    }
    if (auto it = s.depth_views.find(t->image); it != s.depth_views.end()) return it->second;
    VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vc.image = t->image, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = t->format;
    vc.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(dev.device, &vc, nullptr, &v) != VK_SUCCESS) v = VK_NULL_HANDLE;
    s.depth_views[t->image] = v;
    return v;
}

// ---- actor motion: the skinned draws drawn again into the motion, each pixel they show moved by its own bones ----

using ActorPush = dlss_actor::Push;

/// Its pipeline, once (false: none, the motion stays the camera's)
bool make_actor_pipeline() {
    State& s = st();
    if (s.actor) return true;
    if (s.actor_tried) return false;
    s.actor_tried = true;
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    vkGetPhysicalDeviceFeatures2(dev.physical, &f);
    if (!dev.vk12.bufferDeviceAddress || !f.features.shaderInt64) {   // (its shaders read the game's buffers by their addresses)
        logf("vk: NVIDIA DLSS: no buffer device addresses or 64-bit integers in shaders: camera motion alone");
        return false;
    }
    VkDescriptorSetLayoutBinding b[2]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    li.bindingCount = 2, li.pBindings = b;
    VkDescriptorSetLayout set{};
    if (vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &set) != VK_SUCCESS) return false;
    const VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ActorPush)};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1, pi.pSetLayouts = &set, pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(dev.device, &pi, nullptr, &s.actor_layout) != VK_SUCCESS) return false;
    VkShaderModule m[2]{};
    const uint32_t* code[2] = {DLSS_ACTOR_VERT_SPV, DLSS_ACTOR_FRAG_SPV};
    const size_t size[2] = {sizeof DLSS_ACTOR_VERT_SPV, sizeof DLSS_ACTOR_FRAG_SPV};
    for (int i = 0; i < 2; i++) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = size[i], mi.pCode = code[i];
        if (vkCreateShaderModule(dev.device, &mi, nullptr, &m[i]) != VK_SUCCESS) return false;
    }
    VkPipelineShaderStageCreateInfo stages[2]{};
    for (int i = 0; i < 2; i++) {
        stages[i] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[i].stage = i ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT;
        stages[i].module = m[i], stages[i].pName = "main";
    }
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL, rs.cullMode = VK_CULL_MODE_NONE, rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1, cb.pAttachments = &ba;
    const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2, ds.pDynamicStates = dyn;
    const VkFormat format = VK_FORMAT_R16G16_SFLOAT;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = 1, ri.pColorAttachmentFormats = &format;
    VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gi.pNext = &ri;
    gi.stageCount = 2, gi.pStages = stages;
    gi.pVertexInputState = &vi, gi.pInputAssemblyState = &ia, gi.pViewportState = &vp, gi.pRasterizationState = &rs;
    gi.pMultisampleState = &ms, gi.pColorBlendState = &cb, gi.pDynamicState = &ds, gi.layout = s.actor_layout;
    const bool ok = vkCreateGraphicsPipelines(dev.device, VK_NULL_HANDLE, 1, &gi, nullptr, &s.actor) == VK_SUCCESS;
    for (VkShaderModule x : m) vkDestroyShaderModule(dev.device, x, nullptr);
    if (!ok) s.actor = VK_NULL_HANDLE, logf("vk: NVIDIA DLSS: the actor motion's pipeline was not made: camera motion alone");
    return ok;
}

/// A range of guest memory as this recording binds it, by its device address (the pools and their mirrors have them)
uint64_t device_address(const BufferRef& r) {
    if (!r) return 0;
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = r.buffer;
    const VkDeviceAddress base = vkGetBufferDeviceAddress(dev.device, &da);
    return base ? base + r.offset : 0;
}

constexpr uint64_t kPoseBytes = 4 << 20;   // (a pose buffer: 280 bones of 48 bytes are 13 KB a draw, 300 of them)

/// The skinned draws into the scene (cam's depth) with a pose of the frame before (DLSS's last), drawn into the motion
/// image over the camera's motion. Each one's bones and origin are copied on the GPU, from where the game has them, into
/// this frame's pose buffer, for this frame and the next. before: the frame before's camera without its jitter. history:
/// DLSS has the frame before. Returns the draws drawn
int actor_motion(VkCommandBuffer cmd, const Camera& cam, const float* before, uint32_t rw, uint32_t rh, VkImageView dview,
                 bool history) {
    State& s = st();
    struct Timed {
        State& s;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timed() { s.actor_ns += (std::chrono::steady_clock::now() - t0).count(); }
    } timed{s};
    std::vector<State::Skin> skins = std::move(s.skins);
    s.skins.clear();
    std::unordered_map<uint64_t, State::Pose> old = std::move(s.poses);
    s.poses.clear();
    std::vector<State::Rigid> rigids;
    rigids.swap(s.rigids);
    // (the rigid draws' matrices of this frame become the frame before's for the next, whatever is drawn below)
    std::swap(s.rigid_old, s.rigid_before);
    std::swap(s.rigid_same_old, s.rigid_same_before);
    if (s.rigid_before.size() > 4096) s.rigid_before.clear();   // (models no longer drawn: let go now and then)
    for (auto& [model, v] : s.rigid_before) v.clear();
    s.rigid_same_before.clear();
    const auto& rigid_old = s.rigid_old;
    const bool rigid_history = history && s.rigid_frame == s.last_frame && s.rigid_frame;
    const auto key_of = [](uint64_t model, const float* m) {
        uint64_t h = model;
        for (int i = 0; i < 12; i++) {
            uint32_t w;
            std::memcpy(&w, &m[i], 4);
            h = (h ^ w) * 0x100000001b3ull;
        }
        return h;
    };
    const auto same_key = [&](const State::Rigid& r) { return key_of(r.model, r.m); };
    for (const auto& r : rigids)
        if (r.k.depth == cam.depth) {
            auto& v = s.rigid_before[r.model];
            v.insert(v.end(), r.m, r.m + 12);
            s.rigid_same_before.insert(same_key(r));
        }
    s.rigid_frame = texture_frame_no();
    static const bool off = env_switch("RX_VK_DLSS_ACTORS") == 0;
    if (off || (skins.empty() && rigids.empty()) || !make_actor_pipeline()) return 0;
    for (auto& b : s.pose_buffers)
        if (!b.buffer) {
            if (!create_device_buffer(kPoseBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                      b.buffer, b.memory))
                return b = {}, 0;
            b.address = device_address({b.buffer, 0, kPoseBytes, 0});
        }
    const uint64_t now = texture_frame_no();
    const State::PoseBuffer& here = s.pose_buffers[s.pose_at];
    const State::PoseBuffer& there = s.pose_buffers[s.pose_at ^ 1];
    static const uint32_t tolerance = env_switch("RX_VK_DLSS_ACTOR_TOL") >= 0 ? (uint32_t)env_switch("RX_VK_DLSS_ACTOR_TOL") : 64;
    std::vector<ActorPush> draws;
    std::vector<uint32_t> counts;
    uint32_t at = 0;
    for (const auto& k : skins) {
        if (k.k.depth != cam.depth) continue;
        const uint32_t bytes = 16 + k.nbones * 48;
        if (at + bytes > kPoseBytes) break;
        // (this frame's pose, from where the game has it: its bones, then its origin at the object's 0x40)
        const BufferRef bones = gpu_buffer_bind(k.k.bones, (uint64_t)k.nbones * 48, false);
        const BufferRef origin = gpu_buffer_bind(k.k.object + 0x40, 12, false);
        if (!bones || !origin) {
            s.actor_skipped++;
            continue;
        }
        const VkBufferCopy c0{origin.offset, at, 12}, c1{bones.offset, at + 16, (uint64_t)k.nbones * 48};
        vkCmdCopyBuffer(cmd, origin.buffer, here.buffer, 1, &c0);
        vkCmdCopyBuffer(cmd, bones.buffer, here.buffer, 1, &c1);
        s.poses[k.key] = {at, k.nbones, now};
        const uint32_t mine = at;
        at += (bytes + 15) & ~15u;
        const auto it = old.find(k.key);
        if (!history || it == old.end() || it->second.frame != s.last_frame || it->second.nbones != k.nbones) {
            s.actor_skipped++;
            continue;
        }
        // (the vertices and indices where the game has them: its buffers as this recording binds them)
        const uint64_t va = device_address(gpu_buffer_bind(k.k.vbuf, k.k.vsize, false));
        const uint64_t ia = device_address(gpu_buffer_bind(k.k.ibuf, (uint64_t)k.k.count << k.k.itype, false));
        if (!va || !ia || va % 4 || k.k.count <= 0 || k.k.itype < 0 || k.k.itype > 2) {
            s.actor_skipped++;
            continue;
        }
        ActorPush a{};
        a.vertices = va, a.count = (uint32_t)std::min<uint64_t>(k.k.vsize / k.k.stride, UINT32_MAX);
        dlss_actor::index_start(ia, k.k.itype, &a.indices, &a.first);
        a.origins[0] = here.address + mine, a.origins[1] = there.address + it->second.at;
        a.bones[0] = a.origins[0] + 16, a.bones[1] = a.origins[1] + 16;
        a.nbones = k.nbones;
        a.itype = (uint32_t)k.k.itype, a.base_vertex = k.k.base_vertex;
        a.stride = k.k.stride / 4, a.pos = k.k.pos, a.sets = (uint32_t)k.k.sets;
        for (int i = 0; i < 2; i++) a.joints[i] = k.k.joints[i], a.weights[i] = k.k.weights[i];
        a.formats = k.k.formats;
        // (RX_VK_DLSS_ACTOR_MARK=1: every actor draw marked, 2: the rigidly skinned ones only; see dlss_actor::kMark)
        static const int mark = env_switch("RX_VK_DLSS_ACTOR_MARK");
        if (mark == 1 || (mark == 2 && (a.formats & dlss_actor::kOneJoint))) a.formats |= dlss_actor::kMark;
        s.actor_one += (a.formats & dlss_actor::kOneJoint) != 0;
        a.tolerance = tolerance;
        draws.push_back(a);
        counts.push_back((uint32_t)k.k.count);
    }
    // (relink, 0.7.9.3) The rigid objects that moved since the frame before: each matched with the frame before's matrix of the
    // same model nearest it (a model drawn many times, pots or crates, comes in another order each frame, so the n-th draw is
    // not the same object: 19/11's object trace), within RX_VK_DLSS_RIGID_REACH units (default 48: a thrown pot or a rolling
    // boulder moves a few units a frame, the same model's next copy is further). An unchanged matrix needs no draw: the camera's
    // motion is already right there. Both matrices go into this frame's pose buffer (origin none, then the 3 x 4 matrix)
    static const float reach = getenv("RX_VK_DLSS_RIGID_REACH") ? (float)atof(getenv("RX_VK_DLSS_RIGID_REACH")) : 48.0f;
    // (RX_VK_DLSS_RIGID_ALL=1, a test of the transform: every rigid draw drawn again, the still ones with their own matrix as the
    // frame before's. A draw writes only where its depth is the scene's, so with RX_VK_DLSS_ACTOR_MARK=3 the motion view shows
    // which pixels the rigid draws were found to cover: the scenery's shapes when world = matrix x position is right)
    static const bool all = env_switch("RX_VK_DLSS_RIGID_ALL") > 0;
    for (const auto& r : rigids) {
        if (r.k.depth != cam.depth || !rigid_history) continue;
        s.rigid_seen++;
        const float* was = r.m;
        if (!all) {
            if (s.rigid_same_old.count(same_key(r))) continue;   // (the same matrix as one of the model's the frame before: still)
            const auto it = rigid_old.find(r.model);
            if (it == rigid_old.end()) continue;
            const int i = dlss_actor::nearest(r.m, it->second.data(), it->second.size() / 12, reach);
            if (i < 0) continue;
            was = it->second.data() + (size_t)i * 12;
            if (dlss_actor::same_3x4(r.m, was)) continue;
            // (a matrix of the frame before that a draw of this frame still holds is that object, still in place: not this
            // one's. A copy of a repeated model newly in view (a fence post, a step) would otherwise take its neighbour's)
            if (s.rigid_same_before.count(key_of(r.model, was))) continue;
        }
        s.rigid_models[r.model] = texture_frame_no();
        if (at + 128 > kPoseBytes) break;
        const uint64_t va = device_address(gpu_buffer_bind(r.k.vbuf, r.k.vsize, false));
        const uint64_t ia = device_address(gpu_buffer_bind(r.k.ibuf, (uint64_t)r.k.count << r.k.itype, false));
        if (!va || !ia || va % 4 || r.k.count <= 0 || r.k.itype < 0 || r.k.itype > 2) {
            s.actor_skipped++;
            continue;
        }
        float pose[32] = {};   // (this frame's: origin 0, matrix; the frame before's: the same at +64 bytes)
        std::memcpy(pose + 4, r.m, 48);
        std::memcpy(pose + 20, was, 48);
        vkCmdUpdateBuffer(cmd, here.buffer, at, sizeof pose, pose);
        ActorPush a{};
        a.vertices = va, a.count = (uint32_t)std::min<uint64_t>(r.k.vsize / r.k.stride, UINT32_MAX);
        dlss_actor::index_start(ia, r.k.itype, &a.indices, &a.first);
        a.origins[0] = here.address + at, a.origins[1] = here.address + at + 64;
        a.bones[0] = a.origins[0] + 16, a.bones[1] = a.origins[1] + 16;
        a.nbones = 1;
        a.itype = (uint32_t)r.k.itype, a.base_vertex = r.k.base_vertex;
        a.stride = r.k.stride / 4, a.pos = r.k.pos, a.sets = 1;
        a.formats = r.k.formats | dlss_actor::kRigid;
        // (RX_VK_DLSS_ACTOR_MARK=3: the rigid draws only marked)
        static const int mark = env_switch("RX_VK_DLSS_ACTOR_MARK");
        if (mark == 1 || mark == 3) a.formats |= dlss_actor::kMark;
        a.tolerance = tolerance;
        at += 128;
        s.rigid_moved++;
        draws.push_back(a);
        counts.push_back((uint32_t)r.k.count);
    }
    s.pose_at ^= 1;
    if (draws.empty()) return 0;
    // (the cameras: this frame's as drawn, the frame before's without its jitter, the jitter, the render size)
    State::Upload& u = s.uploads[s.upload_at];
    s.upload_at = (s.upload_at + 1) % (int)(sizeof s.uploads / sizeof s.uploads[0]);
    if (!u.buffer && !create_buffer_cached(dlss_actor::kHeadWords * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, u.buffer, u.memory, &u.map))
        return u = {}, 0;
    float* head = (float*)u.map;
    std::memcpy(head, cam.current, 64);
    std::memcpy(head + 16, before, 64);
    head[32] = cam.jitter[0], head[33] = cam.jitter[1], head[34] = (float)rw, head[35] = (float)rh;
    flush_mapped(u.memory, dlss_actor::kHeadWords * 4);
    full_barrier(cmd);   // (the poses copied)
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = s.motion.view, ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {rw, rh}}, ri.layerCount = 1, ri.colorAttachmentCount = 1, ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.actor);
    const VkViewport viewport{0, 0, (float)rw, (float)rh, 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, {rw, rh}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    const VkDescriptorBufferInfo bi{u.buffer, 0, dlss_actor::kHeadWords * 4};
    const VkDescriptorImageInfo ii{s.point, dview, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    w[0].dstBinding = 0, w[0].descriptorCount = 1, w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, w[0].pBufferInfo = &bi;
    w[1].dstBinding = 1, w[1].descriptorCount = 1, w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[1].pImageInfo = &ii;
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.actor_layout, 0, 2, w);
    for (size_t i = 0; i < draws.size(); i++) {   // (vertex i of a draw is its i-th index)
        vkCmdPushConstants(cmd, s.actor_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof draws[i], &draws[i]);
        vkCmdDraw(cmd, counts[i], 1, 0, 0);
    }
    vkCmdEndRendering(cmd);
    s.actor_draws += draws.size(), s.actor_frames++;
    return (int)draws.size();
}

#ifdef RELINK_NGX
/// DLSS's preset: the DLSS model row's, RX_VK_DLSS_PRESET over it (upscaler_rows.h dlss_preset)
unsigned wanted_preset() {
    static const char* const env = getenv("RX_VK_DLSS_PRESET");
    return upscaler_rows::dlss_preset(opts.dlss_model, env && *env ? env : nullptr);
}

/// DLSS's feature for this output size and preset, made when either changes (cmd: its creation's work)
bool make_feature(VkCommandBuffer cmd, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh, bool hdr = false) {
    State& s = st();
    const unsigned preset = wanted_preset();
    if (s.feature && s.feature_ow == ow && s.feature_oh == oh && rw <= s.feature_rw && rh <= s.feature_rh &&
        rw >= s.min_rw && rh >= s.min_rh && s.feature_preset == preset && s.feature_hdr == hdr)
        return true;
    const uint64_t refused_key = (uint64_t)ow << 40 ^ (uint64_t)oh << 16 ^ (uint64_t)preset ^ 1ull << 63;
    if (s.refused == refused_key) return fail("its feature was not made");
    // (why it is made: in the log with its time, so a stall in play can be matched with one)
    char why[160];
    if (!s.feature && !s.features_made) snprintf(why, sizeof why, "the first");
    else if (!s.feature) snprintf(why, sizeof why, "again after it was let go");
    else if (s.feature_ow != ow || s.feature_oh != oh)
        snprintf(why, sizeof why, "the output went from %ux%u to %ux%u", s.feature_ow, s.feature_oh, ow, oh);
    else if (s.feature_hdr != hdr)
        snprintf(why, sizeof why, "%s colour (RX_VK_UPSCALE_INPUT)", hdr ? "HDR" : "tonemapped");
    else if (s.feature_preset != preset)
        snprintf(why, sizeof why, "preset %c, was %c (the DLSS model row)", upscaler_rows::preset_letter(preset),
                 upscaler_rows::preset_letter(s.feature_preset));
    else
        snprintf(why, sizeof why, "the render size %ux%u left its range, %ux%u to %ux%u", rw, rh, s.min_rw, s.min_rh,
                 s.feature_rw, s.feature_rh);
    // (the quality mode whose range of render sizes holds this one: the network's 3/4 is DLSS's MaxQuality range)
    // (DLAA's when the render size is the output's)
    static const NVSDK_NGX_PerfQuality_Value scaled[] = {NVSDK_NGX_PerfQuality_Value_MaxQuality, NVSDK_NGX_PerfQuality_Value_DLAA,
                                                        NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf,
                                                        NVSDK_NGX_PerfQuality_Value_UltraPerformance};
    static const NVSDK_NGX_PerfQuality_Value same[] = {NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality};
    const bool aa = rw == ow && rh == oh;
    const NVSDK_NGX_PerfQuality_Value* modes = aa ? same : scaled;
    const size_t nmodes = aa ? 2 : 5;
    NVSDK_NGX_PerfQuality_Value mode = modes[0];
    unsigned max_w = 0, max_h = 0, min_w = 0, min_h = 0, rec_w = 0, rec_h = 0;
    bool found = aa;
    if (aa) max_w = min_w = ow, max_h = min_h = oh;   // (DLAA: the render size is the output's, one size)
    struct Range { unsigned w = 0, h = 0, max_w = 0, max_h = 0, min_w = 0, min_h = 0; bool ok = false; } ranges[5];
    for (size_t i = 0; i < nmodes && !found; i++) {
        const NVSDK_NGX_PerfQuality_Value m = modes[i];
        unsigned w = 0, h = 0, xw = 0, xh = 0, nw = 0, nh = 0;
        float sharp = 0;
        if (NVSDK_NGX_FAILED(NGX_DLSS_GET_OPTIMAL_SETTINGS(s.caps, ow, oh, m, &w, &h, &xw, &xh, &nw, &nh, &sharp)) || !w) continue;
        ranges[i] = {w, h, xw, xh, nw, nh, true};
        if (rw >= nw && rh >= nh && rw <= xw && rh <= xh) {
            mode = m, max_w = xw, max_h = xh, min_w = nw, min_h = nh, rec_w = w, rec_h = h, found = true;
            break;
        }
    }
    // (relink) The render resolution's sizes are rounded, and a mode's range can be a single size: at an output not divisible by
    // 3 the game's third came out a pixel off Ultra Performance's, no mode took it and the game's upscale ran in DLSS's place.
    // A size within a pixel or two (1/64 of the range's top, 2 at least) of a mode's range is that mode's, its range widened to
    // the size, if DLSS can scale it at all (the output 1 to 3 times it). RX_VK_DLSS_NEAR_RANGE=0: no such leeway
    static const bool near_range = !(getenv("RX_VK_DLSS_NEAR_RANGE") && atoi(getenv("RX_VK_DLSS_NEAR_RANGE")) == 0);
    bool widened = false;
    if (!found && near_range && rw && rh && rw <= ow && rh <= oh && (uint64_t)rw * 3 >= ow && (uint64_t)rh * 3 >= oh)
        for (size_t i = 0; i < nmodes && !found; i++) {
            const Range& g = ranges[i];
            if (!g.ok || !g.max_w || !g.max_h) continue;
            const unsigned sw = std::max(2u, g.max_w / 64), sh = std::max(2u, g.max_h / 64);
            if (rw + sw >= g.min_w && rh + sh >= g.min_h && rw <= g.max_w + sw && rh <= g.max_h + sh) {
                mode = modes[i], rec_w = g.w, rec_h = g.h, found = widened = true;
                min_w = std::min(rw, g.min_w), min_h = std::min(rh, g.min_h);
                max_w = std::max(rw, g.max_w), max_h = std::max(rh, g.max_h);
            }
        }
    if (!found) return fail("no DLSS mode takes this render size");   // (the feature it has stays: let go only once a mode takes a size)
    if (widened) {
        static std::atomic<int> told{0};
        if (told++ < 4)
            logf("vk: NVIDIA DLSS: %ux%u is a little outside quality mode %d's range (%ux%u to %ux%u): taken by it", rw, rh, (int)mode,
                 min_w, min_h, max_w, max_h);
    }
    if (s.feature) {
        NVSDK_NGX_Handle* old = s.feature;
        later([old] { NVSDK_NGX_VULKAN_ReleaseFeature(old); });
        s.feature = nullptr;
    }
    // (the DLSS model row's preset for every mode, K by default: NVIDIA's transformer model. 0: NVIDIA's own per mode)
    static_assert(NVSDK_NGX_DLSS_Hint_Render_Preset_K == 11 && NVSDK_NGX_DLSS_Hint_Render_Preset_L == 12 &&
                  NVSDK_NGX_DLSS_Hint_Render_Preset_M == 13 && NVSDK_NGX_DLSS_Hint_Render_Preset_Default == 0);
    for (const char* k : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                          NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                          NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
                          NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality})
        NVSDK_NGX_Parameter_SetUI(s.params, k, preset);
    NVSDK_NGX_DLSS_Create_Params cp{};
    // (relink) A scaled mode's feature made at NGX's recommended render size for that mode, not the top of its range: models L
    // and M cost less GPU time made so (K about the same; not measured here yet). The input images still cover the whole range
    // (feature_rw/rh) and each upscale names its own render size, as
    // NGX's dynamic resolution allows. DLAA keeps its one size. RX_VK_DLSS_CREATE_MAX=1: made at the range's top, as before
    static const bool create_max = getenv("RX_VK_DLSS_CREATE_MAX") != nullptr;
    const bool at_rec = !aa && !create_max && rec_w && rec_h && rec_w <= max_w && rec_h <= max_h;
    cp.Feature.InWidth = at_rec ? rec_w : max_w, cp.Feature.InHeight = at_rec ? rec_h : max_h;
    cp.Feature.InTargetWidth = ow, cp.Feature.InTargetHeight = oh;
    cp.Feature.InPerfQualityValue = mode;
    // (LDR colour, already tonemapped: no exposure; motion at the render size, not jittered; depth not reversed)
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    // (relink, 0.7.9.3: RX_VK_UPSCALE_INPUT=hdr: the HDR scene, its exposure applied already (dlss_post.comp mode 4))
    if (hdr) cp.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    const auto t0 = std::chrono::steady_clock::now();
    const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT(cmd, 1, 1, &s.feature, s.params, &cp);
    const double made_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    note_work();   // (NGX's own commands, which vk_work.h does not see: the next full_barrier is not left out)
    if (NVSDK_NGX_FAILED(r) || !s.feature) {
        logf("vk: NVIDIA DLSS: its feature was not made (%#x) for %ux%u -> %ux%u", (unsigned)r, max_w, max_h, ow, oh);
        s.feature = nullptr;
        static const bool retry = !(getenv("RX_VK_DLSS_RETRY") && atoi(getenv("RX_VK_DLSS_RETRY")) == 0);
        if (!retry) {
            s.failed = true;
            return false;
        }
        s.refused = refused_key;
        return fail("its feature was not made");   // (the game's upscale until the preset or output changes)
    }
    s.refused = 0;
    s.feature_rw = max_w, s.feature_rh = max_h, s.feature_ow = ow, s.feature_oh = oh, s.min_rw = min_w, s.min_rh = min_h;
    s.feature_preset = preset, s.feature_hdr = hdr;
    s.features_made++;
    logf("vk: NVIDIA DLSS: feature %llu made (%s; frame %llu, %.1f ms of CPU): %ux%u (%ux%u to %ux%u, made at %ux%u) -> %ux%u, "
         "mode %d, preset %c%s", (unsigned long long)s.features_made, why, (unsigned long long)texture_frame_no(), made_ms, rw, rh,
         min_w, min_h, max_w, max_h, cp.Feature.InWidth, cp.Feature.InHeight, ow, oh, (int)mode,
         upscaler_rows::preset_letter(preset), preset ? "" : " (NVIDIA's default)");
    s.last_frame = 0;   // (a new history)
    return true;
}

/// NGX's feature let go (when the GPU is done with it)
void drop_feature() {
    State& s = st();
    if (!s.feature) return;
    NVSDK_NGX_Handle* old = s.feature;
    later([old] { NVSDK_NGX_VULKAN_ReleaseFeature(old); });
    s.feature = nullptr;
    s.feature_rw = s.feature_rh = s.feature_ow = s.feature_oh = s.min_rw = s.min_rh = 0;
}
#else
void drop_feature() {}
#endif

// ---- (relink) AMD FSR 3.1's upscaler (vk_fsr.cpp): made, let go, and given the frame ----

/// FSR's upscaler let go (when the GPU is done with it)
void drop_fsr() {
    State& s = st();
    if (!s.fsr) return;
    FsrContext* old = s.fsr;
    later([old] { fsr_destroy(old); });
    s.fsr = nullptr;
    s.fsr_rw = s.fsr_rh = s.fsr_ow = s.fsr_oh = s.fsr_iw = s.fsr_ih = 0;
}

/// FSR's upscaler for a render size rw x rh into dw x dh, made when the output, its flags or the largest render size it
/// takes change. With the game's dynamic size (the Render resolution row's Auto) it is made for render sizes up to the
/// output's, so the size can move without a new one (and a new history); with a fixed share for that share's size (FFX's
/// memory follows it: into 2560x1440, 221 MB made for the output's size, 134 MB for 67%'s), again only when a larger one
/// comes. Its inputs grow
/// with the largest render size given
bool make_fsr(uint32_t rw, uint32_t rh, uint32_t dw, uint32_t dh, const Camera& cam, bool hdr = false) {
    State& s = st();
    const bool dynamic = opts.resolution == 0 && dw != rw;
    // (relink: the FSR version row, opts.fsr_version 1: FSR 3.1 over the runtime's default (FSR 4 where it gives it). A change
    // of row is a change of flags: a new upscaler. The colour given is gamma-encoded unless RX_VK_FSR_SRGB=0: FSR 4 is told at
    // its creation)
    static const bool perceptual = env_switch("RX_VK_FSR_SRGB") != 0;
    const unsigned flags = (cam.infinite ? FSR_DEPTH_INFINITE : 0u) | (dynamic ? FSR_DYNAMIC : 0u) |
                           (opts.fsr_version == 1 && fsr31_available() ? FSR_V31 : 0u) |
                           (perceptual && !hdr ? FSR_PERCEPTUAL : 0u) | (hdr ? FSR_HDR : 0u);
    if (s.fsr && s.fsr_ow == dw && s.fsr_oh == dh && s.fsr_flags == flags && rw <= s.fsr_rw && rh <= s.fsr_rh) {
        s.fsr_iw = std::max(s.fsr_iw, rw), s.fsr_ih = std::max(s.fsr_ih, rh);
        return true;
    }
    const uint32_t mw = dynamic ? dw : rw, mh = dynamic ? dh : rh;   // (the largest render size it takes)
    char why[160];
    if (!s.fsr && !s.fsr_made) snprintf(why, sizeof why, "the first");
    else if (!s.fsr) snprintf(why, sizeof why, "again after it was let go");
    else if (s.fsr_ow != dw || s.fsr_oh != dh)
        snprintf(why, sizeof why, "the output went from %ux%u to %ux%u", s.fsr_ow, s.fsr_oh, dw, dh);
    else if (s.fsr_flags != flags)
        snprintf(why, sizeof why, "its flags went from %#x to %#x", s.fsr_flags, flags);
    else
        snprintf(why, sizeof why, "the render size %ux%u is over its %ux%u", rw, rh, s.fsr_rw, s.fsr_rh);
    drop_fsr();
    const auto t0 = std::chrono::steady_clock::now();
    s.fsr = fsr_create(mw, mh, dw, dh, flags);
    const double made_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!s.fsr) {
        s.fsr_failed = true;
        return fail("its upscaler was not made");
    }
    s.fsr_rw = mw, s.fsr_rh = mh, s.fsr_ow = dw, s.fsr_oh = dh, s.fsr_flags = flags, s.fsr_iw = rw, s.fsr_ih = rh;
    s.fsr_made++;
    // (the jitter's length: FFX's own for this ratio; the host's sequence is the one FFX describes, Halton (2, 3) from its
    // second value (dlss_math.h dlss_jitter_pixels), so only its length is taken)
    s.fsr_phases = fsr_jitter_phases(s.fsr, rw, dw);
    logf("vk: AMD FSR: upscaler %llu made (%s; frame %llu, %.1f ms of CPU): %ux%u (up to %ux%u) -> %ux%u, flags %#x, jitter "
         "of %d phases, FSR %s; camera planes %.4g to %.6g%s, vertical field of view %.1f degrees",
         (unsigned long long)s.fsr_made, why, (unsigned long long)texture_frame_no(), made_ms, rw, rh, mw, mh, dw, dh, flags,
         s.fsr_phases, fsr_context_version(s.fsr), cam.near_z, cam.far_z, cam.infinite ? " (no far plane)" : "", cam.fov_y * 57.29578f);
    s.last_frame = 0;   // (a new history)
    return true;
}

}  // namespace

#ifdef RELINK_NGX
bool dlss_built() { return true; }
#else
bool dlss_built() { return false; }
#endif

void dlss_instance_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& all) {
#ifdef RELINK_NGX
    if (env_switch("RX_VK_DLSS") == 0) return;
    for (const auto& e : all)
        if (!strcmp(e.extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            want.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
            return;
        }
#else
    (void)want, (void)all;
#endif
}

void dlss_device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& all) {
#ifdef RELINK_NGX
    // (NGX's: VK_NVX_binary_import and VK_NVX_image_view_handle, with push descriptors and buffer device addresses,
    // which the replay enables already: core 1.2's bufferDeviceAddress stands for VK_EXT_buffer_device_address. The NVX ones
    // are on whenever the device has them, so the Insert menu's row needs no restart)
    if (env_switch("RX_VK_DLSS") == 0) return;
    bool both = true;
    for (const char* name : {VK_NVX_BINARY_IMPORT_EXTENSION_NAME, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME}) {
        bool have = false, listed = false;
        for (const auto& e : all) have |= !strcmp(e.extensionName, name);
        for (const char* w : want) listed |= !strcmp(w, name);
        if (have && !listed) want.push_back(name);
        both &= have;
    }
    // (NVIDIA's own extensions: without them this is not an NVIDIA GPU, or not one NGX runs on. With them, NGX says, in
    // dlss_probe or at the first upscale: a GTX has them too)
    if (!both) g_why = DLSS_NEEDS_RTX;
    // (relink, 0.7.9.3) What NGX asks of the device for DLSS Frame Generation, before the device is made (its guide: NGX's
    // feature requirements), on an NVIDIA device only: each one the device has is turned on, and said with what is missing.
    // Only when the Frame generator row asks for it (the options are read before the device is made): the question took
    // 1.7 s of the start on the bench's 3060 Ti (21/11). What it asked there (VK_NVX_binary_import, VK_NVX_image_view_handle,
    // VK_KHR_buffer_device_address, VK_KHR_push_descriptor) the device has on anyway, so the row chosen in play needs no
    // restart where NGX asks no more than that (an RTX 40's answer: not seen)
    if (both && dev.physical && dev.instance && dev.props.vendorID == 0x10DE && dlssg_asked()) {
        const std::vector<fs::path> dirs = ngx_dirs();
        std::vector<std::wstring> wdirs;
        for (const auto& d : dirs) wdirs.push_back(wide(d));
        std::vector<const wchar_t*> paths;
        for (const auto& d : wdirs) paths.push_back(d.c_str());
        NVSDK_NGX_FeatureCommonInfo common{};
        common.PathListInfo.Path = paths.data(), common.PathListInfo.Length = (unsigned)paths.size();
        const std::wstring wdata = wide(game_dir() / "ngx");
        NVSDK_NGX_FeatureDiscoveryInfo d{};
        d.SDKVersion = NVSDK_NGX_Version_API;
        d.FeatureID = NVSDK_NGX_Feature_FrameGeneration;
        d.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
        d.Identifier.v.ProjectDesc = {kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0"};
        d.ApplicationDataPath = wdata.c_str();
        d.FeatureInfo = &common;
        const auto t0 = std::chrono::steady_clock::now();
        uint32_t count = 0;
        VkExtensionProperties* props = nullptr;
        const NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(dev.instance, dev.physical, &d, &count, &props);
        std::string added, missing;
        for (uint32_t i = 0; NVSDK_NGX_SUCCEED(r) && props && i < count; i++) {
            const char* name = props[i].extensionName;
            bool have = false, listed = false;
            for (const auto& e : all) have |= !strcmp(e.extensionName, name);
            for (const char* w : want) listed |= !strcmp(w, name);
            if (have && !listed) {
                // (the name kept: NGX's array is its own, and the list is read when the device is made)
                static std::vector<std::string> kept;
                kept.reserve(32);
                if (kept.size() < 32) kept.push_back(name), want.push_back(kept.back().c_str());
            }
            (have ? added : missing) += (have ? (added.empty() ? "" : ", ") : (missing.empty() ? "" : ", ")) + std::string(name);
        }
        NVSDK_NGX_FeatureRequirement req{};
        const NVSDK_NGX_Result q = NVSDK_NGX_VULKAN_GetFeatureRequirements(dev.instance, dev.physical, &d, &req);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        logf("vk: NVIDIA DLSS Frame Generation's device extensions (%#x, asked in %.1f ms): %u asked%s%s%s%s; its requirements (%#x): %s "
             "(flags %#x, architecture %#x or later)", (unsigned)r, ms, count, added.empty() ? "" : ", on: ", added.c_str(),
             missing.empty() ? "" : ", not on this device: ", missing.c_str(), (unsigned)q,
             NVSDK_NGX_SUCCEED(q) && req.FeatureSupported == 0 ? "supported" : "not supported here", (unsigned)req.FeatureSupported,
             req.MinHWArchitecture);
    }
#else
    (void)want, (void)all;   // (FSR needs no extension of its own: vk_fsr.cpp)
#endif
}

void dlss_probe() {
    if (dlss_why() != DLSS_OK || g_why.load(std::memory_order_relaxed) >= 0 || !dev.device) return;
    State& s = st();
    std::lock_guard lk(s.mu);
    ngx_init();
}

void dlss_note_scene_draw(const Texture* color0, uint32_t width, uint32_t height, Texture* depth, uint64_t camera,
                          uint64_t camera_size) {
    if (!color0 || !depth || !camera || camera_size < 0x230 || width < 640 || height < 360) return;
    // (relink, Part B) the linear scene buffer is B10G11R11, but RGBA16F while the PsychoV35 tone mapper is on
    // (hos_scene_rgba16f, host/nn_vi.c): the scene's camera and depth are taken from it either way
    const bool scene_fmt = color0->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
                           (scene_tonemap_on() && color0->format == VK_FORMAT_R16G16B16A16_SFLOAT);
    if (!scene_fmt || !dlss_wanted() || is_plain(dlss_mode())) return;
    State& s = st();
    std::lock_guard lk(s.mu);
    const uint64_t now = texture_frame_no();
    if (s.cam.valid && s.cam.frame == now && (uint64_t)width * height <= (uint64_t)s.cam.width * s.cam.height)
        return;   // (the largest one of the frame wins)
    float c[0x230 / 4];
    if (!memory_read_as_submitted(camera, c, sizeof c)) return;
    // (the target's size at 0x120; the projection at 0x70, its w row (0, 0, -1, 0) at 0xa0)
    if (c[0x120 / 4] != (float)width || c[0x124 / 4] != (float)height || c[0xa0 / 4] != 0.0f || c[0xa4 / 4] != 0.0f ||
        c[0xa8 / 4] != -1.0f || c[0xac / 4] != 0.0f)
        return;
    Camera& k = s.cam;
    std::memcpy(k.current, c + 0x30 / 4, sizeof k.current);
    std::memcpy(k.proj, c + 0x70 / 4, sizeof k.proj);
    // (the offset sead's projection gave the frustum, in its third column: 2 * offset, which moves the picture by -that)
    k.jitter[0] = -c[0x78 / 4], k.jitter[1] = -c[0x88 / 4];
    // (relink, for FSR) the camera's planes and field of view, from the projection (rows at 0x70, 0x80, 0x90). Its depth is
    // OpenGL's, -1 to 1 (dlss_prep.comp's clip space): row 2 is (0, 0, a, b) with a = -(f + n) / (f - n) and
    // b = -2 f n / (f - n), so n = b / (a - 1) and f = b / (a + 1), none when a is -1. The depth buffer then holds
    // (z / w + 1) / 2, which is what a 0-to-1 projection with the same planes writes: the depth FSR expects. Row 1's second
    // value is the cotangent of half the vertical field of view. Values that do not make a camera keep the last ones
    const float p11 = c[0x84 / 4], a = c[0x98 / 4], bb = c[0x9c / 4];
    const float n_z = bb / (a - 1.0f);
    const bool infinite = std::fabs(a + 1.0f) < 1e-6f;
    const float f_z = infinite ? 0.0f : bb / (a + 1.0f);
    if (p11 > 0.0f && std::isfinite(n_z) && n_z > 0.0f && (infinite || (std::isfinite(f_z) && f_z > n_z))) {
        k.near_z = n_z, k.far_z = infinite ? 1e30f : f_z, k.infinite = infinite;
        k.fov_y = 2.0f * std::atan(1.0f / p11);
    }
    k.width = width, k.height = height, k.depth = depth, k.depth_gen = texture_destroy_generation(), k.frame = now;
    k.valid = true;
}

namespace {
/// The upscaler that runs has failed (its upscale left to the network): DLSS's or FSR's
bool backend_failed(int mode) {
    const State& s = st();
    if (is_plain(mode)) return false;
    return (is_fsr(mode) ? s.fsr_failed : s.failed).load(std::memory_order_relaxed);
}
}  // namespace

uint32_t dlss_jitter_size(int axis) {
    // (RX_VK_DLSS_JITTER=0: none. 0 too once DLSS has not drawn for a while: the camera as the game has it)
    static const bool off = env_switch("RX_VK_DLSS_JITTER") == 0;
    State& s = st();
    const uint64_t last = s.last_frame.load(std::memory_order_relaxed);
    const int mode = dlss_mode();
    if (off || !mode || is_plain(mode) || backend_failed(mode) || !last || texture_frame_no() - last > 30) return 0;
    if (axis == 2) return s.jitter_phases.load(std::memory_order_relaxed);
    return axis ? s.jitter_h.load(std::memory_order_relaxed) : s.jitter_w.load(std::memory_order_relaxed);
}

/// (relink, 0.7.9.3) The window's pass leaves its sharpening out: the upscaler's picture was sharpened under the HUD in the last frames
bool upscaler_sharpens() {
    if (!upscaler_sharpen_on()) return false;
    const uint64_t last = st().last_frame.load(std::memory_order_relaxed);
    return last && texture_frame_no() - last <= 30 && dlss_mode() != 0;
}

float upscaler_lod_bias() {
    // (relink: worked out once a frame where DLSS or FSR draws, lod_bias_for; read here by every sampler lookup, ~48000 a frame)
    State& s = st();
    const uint64_t last = s.last_frame.load(std::memory_order_relaxed);
    if (!last || texture_frame_no() - last > 30) return 0.0f;
    return s.lod_bias.load(std::memory_order_relaxed);
}

int dlss_status(char* b, size_t n) {
    State& s = st();
    const uint64_t last = s.last_frame.load(std::memory_order_relaxed);
    const int mode = dlss_mode();
    if (!mode || backend_failed(mode) || !last || texture_frame_no() - last > 30) return 0;
    std::lock_guard lk(s.mu);
    if (is_plain(mode)) {
        snprintf(b, n, "; no upscaling: %llu frames drawn", (unsigned long long)s.status_drawn);
        s.status_drawn = s.status_passed = 0;
        return mode;
    }
    // (its own video memory: the images and buffers it made here, and what NGX says its feature holds (FFX its upscaler),
    // when it says)
    uint64_t own = 0;
    for (const Image* im : {&s.color, &s.depth, &s.motion, &s.output, &s.opaque, &s.mask, &s.delta[0], &s.delta[1]})
        if (im->image) own += allocation_size(im->memory);   // (relink, 0.7.9.3: the see-through mask's too)
    for (const auto& pb : s.pose_buffers)
        if (pb.buffer) own += allocation_size(pb.memory);
    for (const auto& u : s.uploads)
        if (u.buffer) own += allocation_size(u.memory);
    unsigned long long held = 0;
    bool held_known = false;
    if (is_fsr(mode)) {
        held = s.fsr ? fsr_memory(s.fsr) : 0;
        held_known = held != 0;
    }
#ifdef RELINK_NGX
    else held_known = s.caps && NVSDK_NGX_SUCCEED(NGX_DLSS_GET_STATS(s.caps, &held));
#endif
    const uint64_t draws = s.actor_draws - s.status_draws, frames = s.actor_frames - s.status_frames;
    char feature[48] = "";
    if (held_known) snprintf(feature, sizeof feature, ", its %s %llu MB", is_fsr(mode) ? "upscaler" : "feature", held >> 20);
    char passed[64] = "";
    if (s.status_passed) snprintf(passed, sizeof passed, ", %llu left to the game's network", (unsigned long long)s.status_passed);
    static const char* const names[upscaler_rows::kUpscalers] = {"", "DLSS", "DLAA", "FSR", "FSR native AA", "none"};
    snprintf(b, n, "; %s: %llu frames drawn%s, %llu %s made, jitter %+.3f %+.3f px%s, actor motion %llu draws in %llu "
             "frames (%.3f ms CPU a frame, %.3f of it noting rigid draws, %llu of the draws rigidly skinned, %llu rigid objects moved of %llu seen), its images %llu MB%s",
             names[mode], (unsigned long long)s.status_drawn, passed,
             (unsigned long long)(is_fsr(mode) ? s.fsr_made : s.features_made), is_fsr(mode) ? "upscalers" : "features",
             s.status_jitter[0], s.status_jitter[1], s.status_motion ? "" : " (no motion)",
             (unsigned long long)draws, (unsigned long long)frames, frames ? s.actor_ns / 1e6 / (double)frames : 0.0, frames ? s.rigid_ns / 1e6 / (double)frames : 0.0,
             (unsigned long long)(s.actor_one - s.status_one), (unsigned long long)(s.rigid_moved - s.status_rigid_moved),
             (unsigned long long)(s.rigid_seen - s.status_rigid_seen), (unsigned long long)(own >> 20), feature);
    s.status_one = s.actor_one;
    s.status_rigid_moved = s.rigid_moved, s.status_rigid_seen = s.rigid_seen;
    s.status_draws = s.actor_draws, s.status_frames = s.actor_frames, s.actor_ns = 0, s.rigid_ns = 0, s.status_drawn = s.status_passed = 0;
    return mode;
}

void dlss_idle() {
    State& s = st();
    if (!s.holding.load(std::memory_order_relaxed)) return;
    std::lock_guard lk(s.mu);
    const uint64_t now = texture_frame_no();
    const uint64_t idle = 120;
    if (now - s.asked <= idle) return;   // (DLSS in use, or off for under 120 frames (2 s): kept, a row stepped through comes back at once)
    // (relink) Nor while the Upscaler row still names the backend it holds: a loading screen or a cutscene skips the upscale for
    // a few seconds, and letting go there made the feature again at every area entry, 26 ms of the render thread each time
    // (players' logs: "not used for 120 frames" then "feature N made (again after it was let go)" at each load). Let go when
    // the row changes, as before. RX_VK_UPSCALER_IDLE_FREE=1: let go after 120 frames whatever the row says, as before
    static const bool idle_free = getenv("RX_VK_UPSCALER_IDLE_FREE") != nullptr;
    const int mode = dlss_mode();
    if (!idle_free && mode != 0 && !is_plain(mode) && (s.fsr ? is_fsr(mode) : s.feature_ow && !is_fsr(mode))) return;
    // (pull request 138 kept FSR 4 for 30 s for the same reason, its network taking 200 to 900 ms of CPU to make again: the
    // rule above keeps it, as any backend, for as long as the row names it)
    if (s.fsr || s.feature_ow) {
        const char* name = s.fsr ? "AMD FSR" : "NVIDIA DLSS";
        drop_fsr();
        drop_feature();
        logf("vk: %s: not used for %llu frames (the last at frame %llu, now %llu): its %s and images let go; "
             "the next upscale with it makes them again", name, (unsigned long long)idle, (unsigned long long)s.asked,
             (unsigned long long)now, !strcmp(name, "AMD FSR") ? "upscaler" : "feature");
    }
    for (Image* im : {&s.color, &s.depth, &s.motion, &s.output, &s.opaque, &s.mask, &s.delta[0], &s.delta[1], &s.hdr, &s.post_out, &s.post_final}) drop_image(*im);
    {   // (relink, 0.7.9.3: the scene colour views the mask made)
        auto old = std::move(s.color_views);
        s.color_views.clear();
        later([old] {
            for (const auto& [image, view] : old) vkDestroyImageView(dev.device, view, nullptr);
        });
    }
    for (auto& f : s.fg_slots) drop_image(f.depth), drop_image(f.motion), drop_image(f.hudless), f.has_hudless = false, f.used = true;
#ifdef RELINK_NGX
    if (s.fg_feature) {   // (DLSS Frame Generation's, with the inputs it takes)
        NVSDK_NGX_Handle* old = s.fg_feature;
        later([old] { NVSDK_NGX_VULKAN_ReleaseFeature(old); });
        s.fg_feature = nullptr, s.fg_w = s.fg_h = 0;
        logf("vk: NVIDIA DLSS Frame Generation: let go with the upscaler's images");
    }
#endif
    for (auto& pb : s.pose_buffers)   // (the actor motion's poses and cameras)
        if (pb.buffer) {
            const State::PoseBuffer gone = pb;
            later([gone] { destroy_buffer(gone.buffer, gone.memory); });
            pb = {};
        }
    for (auto& u : s.uploads)
        if (u.buffer) {
            const State::Upload gone = u;
            later([gone] { destroy_buffer(gone.buffer, gone.memory); });
            u = {};
        }
    s.poses.clear(), s.skins.clear(), s.model_bones.clear();
    if (!s.depth_views.empty()) {
        auto old = std::move(s.depth_views);
        s.depth_views.clear();
        later([old] {
            for (const auto& [image, view] : old) vkDestroyImageView(dev.device, view, nullptr);
        });
    }
    s.last_frame = 0;   // (a new history when it comes back)
    free_dead(false);
    s.holding = !s.dead.empty();
}

namespace {
/// The bones a skinned draw's vertices use (the largest joint with a weight, plus one), read once a model; 0 when its
/// buffers cannot be read
uint32_t bones_used(const DlssSkin& k) {
    if (k.count <= 0 || k.count > (1 << 20) || k.itype < 0 || k.itype > 2 || !k.stride) return 0;
    std::vector<uint8_t> idx((size_t)k.count << k.itype);
    if (!memory_read_as_submitted(k.ibuf, idx.data(), idx.size())) return 0;
    uint32_t lo = UINT32_MAX, hi = 0;
    for (int i = 0; i < k.count; i++) {
        uint32_t v = 0;
        std::memcpy(&v, idx.data() + ((size_t)i << k.itype), (size_t)1 << k.itype);
        lo = std::min(lo, v), hi = std::max(hi, v);
    }
    const int64_t first = (int64_t)k.base_vertex + lo, n = (int64_t)hi - lo + 1;
    if (first < 0 || n > (1 << 20) || (uint64_t)(first + n) * k.stride > k.vsize) return 0;
    std::vector<uint8_t> v((size_t)n * k.stride);
    if (!memory_read_as_submitted(k.vbuf + (uint64_t)first * k.stride, v.data(), v.size())) return 0;
    uint32_t most = 0;
    for (int64_t i = 0; i < n; i++) {
        const uint8_t* q = v.data() + (size_t)i * k.stride;
        if (k.formats & dlss_actor::kOneJoint) {   // (one joint a vertex, no weights)
            most = std::max(most, (uint32_t)q[k.joints[0]] + 1);
            continue;
        }
        for (int set = 0; set < k.sets; set++)
            for (int j = 0; j < 4; j++) {
                if (!q[k.weights[set] + j]) continue;
                uint32_t joint = 0;
                if (k.formats & (dlss_actor::kByteJoints << set)) joint = q[k.joints[set] + j];
                else std::memcpy(&joint, q + k.joints[set] + 2 * j, 2);
                most = std::max(most, joint + 1);
            }
    }
    return most;
}
}  // namespace

void dlss_note_skinned_draw(const DlssSkin& k) {
    static const bool off = env_switch("RX_VK_DLSS_ACTORS") == 0;
    if (off || !dlss_wanted() || is_plain(dlss_mode()) || !k.depth || !k.bones || k.bones_size < 48 || !k.object ||
        k.object_size < 0x4c)
        return;
    State& s = st();
    std::lock_guard lk(s.mu);
    if (s.skins.size() >= 512) s.skins.clear();   // (no upscale for a while: what was kept is stale)
    struct Timed {
        State& s;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timed() { s.actor_ns += (std::chrono::steady_clock::now() - t0).count(); }
    } timed{s};
    State::Skin d;
    d.k = k;
    const uint64_t model = dlss_actor::key(k.vbuf, k.ibuf, k.count, k.base_vertex, k.program, 0);
    // (only the bones its vertices use are taken: a palette holds up to 280, a model uses a few dozen)
    auto mb = s.model_bones.find(model);
    if (mb == s.model_bones.end()) {
        if (s.model_bones.size() > 4096) s.model_bones.clear();
        mb = s.model_bones.emplace(model, bones_used(k)).first;
    }
    d.nbones = (uint32_t)std::min<uint64_t>(mb->second, std::min<uint64_t>(k.bones_size / 48, 280));
    if (!d.nbones) return;
    int nth = 0;   // (the same model drawn twice in a frame: the n-th draw of it is the n-th the frame before)
    for (const auto& o : s.skins) nth += dlss_actor::same_model(o.key, model);
    d.key = dlss_actor::key(k.vbuf, k.ibuf, k.count, k.base_vertex, k.program, nth);
    s.skins.push_back(std::move(d));
}

void dlss_note_rigid_draw(const DlssSkin& k) {
    static const bool off = env_switch("RX_VK_DLSS_ACTORS") == 0 || env_switch("RX_VK_DLSS_RIGID") == 0;
    if (off || !dlss_wanted() || is_plain(dlss_mode()) || !k.depth || !k.object || k.object_size < 48 || k.count <= 0 ||
        !k.stride)
        return;
    State& s = st();
    std::lock_guard lk(s.mu);
    struct Timed {
        State& s;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timed() { const int64_t d = (std::chrono::steady_clock::now() - t0).count(); s.actor_ns += d, s.rigid_ns += d; }
    } timed{s};
    if (s.rigids.size() >= 8192) s.rigids.clear();   // (no upscale for a while: what was kept is stale)
    // (relink) Reading every rigid draw's matrix every frame cost 0.3 to 0.8 ms of the render thread in Kakariko (650 draws a
    // frame, 30 of them moving). A program none of whose draws moved in the last 600 frames (the scenery's) is read two frames
    // in eight only, a pair, enough to see one start moving; a model seen moving is read every frame.
    // RX_VK_DLSS_RIGID_EVERY=1: every draw every frame
    static const bool every = env_switch("RX_VK_DLSS_RIGID_EVERY") > 0;
    const uint64_t frame = texture_frame_no();
    const uint64_t model = dlss_actor::key(k.vbuf, k.ibuf, k.count, k.base_vertex, k.program, 0);
    if (!every && (frame & 7) > 1) {   // (by model: a program is shared by most of the scenery, one moving draw kept all of it read)
        const auto it = s.rigid_models.find(model);
        if (it == s.rigid_models.end() || frame - it->second > 600) return;
    }
    if (s.rigid_models.size() > 4096) s.rigid_models.clear();
    State::Rigid r;
    // (the matrix as the GPU will read it: the object buffer's first 48 bytes as submitted, and its words at 0x40: zero for a
    // model drawn by its world matrix alone (19/11's trace, 91 % of them); the others (20, 110, 0 and the like, with matrices
    // that are not a world transform: effects, billboards, the torches' flames) are not this kind of draw.
    // RX_VK_DLSS_RIGID_ANY=1: taken whatever is there)
    float o[19];
    if (k.object_size < sizeof o || !memory_read_as_submitted(k.object, o, sizeof o)) return;
    static const bool any = env_switch("RX_VK_DLSS_RIGID_ANY") > 0;
    if (!any && (o[16] != 0.0f || o[17] != 0.0f || o[18] != 0.0f)) return;
    std::memcpy(r.m, o, sizeof r.m);
    for (float v : r.m)
        if (!std::isfinite(v)) return;
    if (dlss_actor::identity_3x4(r.m)) return;   // (fixed geometry in world space, or a skinned model's: not a rigid mover)
    r.k = k;
    r.model = model;
    s.rigids.push_back(r);
}

namespace {
void fg_capture(VkCommandBuffer cmd, State& s, const Camera& cam, const float* before, uint32_t rw, uint32_t rh, bool reset,
                VkImageView out_view, uint32_t ow, uint32_t oh);
/// dlss_out.comp: a perceptual picture (an upscaler's output, or the scene itself) squared into the game's picture, sharpened
/// first when the Sharpness row asks. same_size: read texel for texel; else through the linear filter (DLAA's and FSR's
/// native AA's picture is at the render size)
// (relink, 0.7.9.3: the Sharpness row on the 3D alone, before the HUD is drawn over it. The window's pass leaves its own
// sharpening out meanwhile, upscaler_sharpens. RX_VK_UPSCALER_SHARPEN=0: as before, after the HUD)
void out_pass(VkCommandBuffer cmd, State& s, VkImageView src, bool same_size, VkImageView out_view, uint32_t ow, uint32_t oh,
              bool linear = false, bool out16 = false) {
    // (relink, 0.7.9.3: DLSS's, DLAA's and FSR's own Sharpness value, 40 % by default; None's is the game's own row's)
    const int row = is_plain(dlss_mode()) ? opts.sharpness : opts.up_sharpness;
    const float amount = upscaler_sharpen_on() ? std::clamp(row, 0, 100) / 100.0f : 0.0f;
    const struct {
        int32_t size[2];
        float sharp, over;
        int32_t linear;   // (the picture is linear colour: its square root first)
    } opc{{(int32_t)ow, (int32_t)oh}, amount * kSharpenGain, amount * kSharpenOver, linear ? 1 : 0};
    const VkDescriptorImageInfo info[2] = {{same_size ? s.point : s.linear, src, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, out_view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    w[0].dstBinding = 0, w[0].descriptorCount = 1, w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &info[0];
    w[1].dstBinding = 1, w[1].descriptorCount = 1, w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[1].pImageInfo = &info[1];
    // (relink, Part B) the PsychoV35 picture is RGBA16F: the out shader's storage format must match it (dlss_out16.comp)
    VkPipeline pipe = out16 ? s.out16 : s.out;
    VkPipelineLayout layout = out16 ? s.out16_layout : s.out_layout;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, w);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof opc, &opc);
    vkCmdDispatch(cmd, (ow + 7) / 8, (oh + 7) / 8, 1);
}
}  // namespace

bool dlss_upscale(VkCommandBuffer cmd, uint64_t key, bool known, Texture* in, VkImageView in_view, Texture* out,
                  VkImageView out_view, uint32_t out_level) {
    const int mode = dlss_mode();
    if (!mode) return false;
    const bool fsr = is_fsr(mode);
    State& s = st();
    std::lock_guard lk(s.mu);
    if (!known) {   // (another network of the upscale's shape: only the scene into the picture, the rest untouched and unsaid)
        const Camera& c = s.cam;
        // (the gameplay networks' own formats: they sample the scene's B10G11R11 colour and write the B10G11R11 picture.
        // Other cooperative-matrix networks of the same shape (one RGBA8 into A2B10G10R10, RGB9E5 into B10G11R11) are not it.
        // Under PsychoV35 both the scene input and the picture output are RGBA16F (hos_scene_rgba16f))
        const bool scene_in = in->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
                              (scene_tonemap_on() && in->format == VK_FORMAT_R16G16B16A16_SFLOAT);
        const bool picture_out = out->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
                                 (scene_tonemap_on() && out->format == VK_FORMAT_R16G16B16A16_SFLOAT);
        if (!in || !out || !scene_in || !picture_out ||
            (out->width >> out_level) <= in->width)
            return false;
        // (one in the gameplay networks' formats whose input is not this frame's scene: counted for the status line)
        if (!c.valid || texture_frame_no() - c.frame > 1 || in->width != c.width || in->height != c.height)
            return s.status_passed++, false;
        static std::atomic<uint64_t> told{0};
        if (told.exchange(key) != key) logf("vk: %s: network %016llx is the gameplay upscale (%ux%u -> %ux%u)",
                                            backend_name(mode), (unsigned long long)key, in->width, in->height,
                                            out->width >> out_level, out->height >> out_level);
    }
    free_dead(false);
    if (is_plain(mode)) {
        // (relink, 0.7.9.3) No upscaling: the scene, drawn at the output's size (dlss_full_scale), goes into the picture as the
        // network's output would hold it, the square of its perceptual encoding, and nothing else is done to it: no history,
        // no camera shift, no smoothing of edges. Only the full-size pass, known by its key: a frame the game still draws
        // smaller (the row just changed) keeps the game's network. The Sharpness row applies, as with the others
        if (!known || !in || !in_view || !out || !out_view) return false;
        const uint32_t rw = in->width, rh = in->height;
        const uint32_t ow = std::max(1u, out->width >> out_level), oh = std::max(1u, out->height >> out_level);
        const bool out16 = out->format == VK_FORMAT_R16G16B16A16_SFLOAT;
        if ((out->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && !(scene_tonemap_on() && out16)) || ow < rw || oh < rh ||
            ow > rw * 3 || oh > rh * 3)
            return fail("unexpected output");
        if (!make_pipelines()) return fail("its shader was not made");
        struct Disturbed {
            ~Disturbed() { descriptors_disturbed(); }
        } disturbed;
        full_barrier(cmd);   // (the scene's colour, written before)
        out_pass(cmd, s, in_view, rw == ow && rh == oh, out_view, ow, oh, false, out16);
        // (s.asked is left alone: a backend still held from the row's last choice is let go by dlss_idle)
        s.last_frame = texture_frame_no();
        s.lod_bias = 0.0f;
        s.status_drawn++;
        if (s.drawn_mode != mode) {
            s.drawn_mode = mode;
            logf("vk: no upscaling: the scene is shown as it is drawn, %ux%u -> %ux%u", rw, rh, ow, oh);
        }
        return true;
    }
    const Camera cam = s.cam;
    s.cam.valid = false;   // (each frame's own camera)
    if (in) s.scene_image.store(in->image, std::memory_order_relaxed);   // (relink, 0.7.9.3: the see-through mask's target)
    if (in) s.scene_tex = in;   // (relink, 0.7.9.3: the HDR scene's texture, kept at the composite: dlss_note_post_draw)
    struct SkinsDone {     // (the skinned draws, each frame's own too: actor_motion takes them, or they go here)
        State& s;
        ~SkinsDone() { s.skins.clear(); }
    } skins_done{s};

    if (backend_failed(mode)) return fail(fsr ? "FSR failed earlier" : "DLSS failed earlier");
    s.asked = texture_frame_no(), s.holding = true;   // (not after a failure: what it holds is let go, dlss_idle)
    // (relink) the Upscaler row changed between DLSS and FSR: the other one's feature or upscaler let go at once (its images
    // are the same and stay)
    if (fsr) drop_feature();
    else drop_fsr();
    if (!in || !in_view || !out || !out_view) return fail("no input or output view");
    if (!cam.valid || texture_frame_no() - cam.frame > 1) return fail("no scene camera this frame");
    if (cam.depth_gen != texture_destroy_generation()) return fail("the scene depth may be gone");
    const uint32_t rw = in->width, rh = in->height;
    const uint32_t ow = std::max(1u, out->width >> out_level), oh = std::max(1u, out->height >> out_level);
    if (rw != cam.width || rh != cam.height) return fail("the input is not the scene's size");
    if (cam.depth->width != rw || cam.depth->height != rh || !(cam.depth->usage & VK_IMAGE_USAGE_SAMPLED_BIT) ||
        cam.depth->samples != VK_SAMPLE_COUNT_1_BIT || !(cam.depth->aspect & VK_IMAGE_ASPECT_DEPTH_BIT))
        return fail("unexpected scene depth");
    // (relink, Part B) the picture output is RGBA16F while PsychoV35 is on (hos_scene_rgba16f), B10G11R11 otherwise
    const bool out16 = out->format == VK_FORMAT_R16G16B16A16_SFLOAT;
    if ((out->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && !(scene_tonemap_on() && out16)) || ow < rw || oh < rh ||
        ow > rw * 3 || oh > rh * 3)
        return fail("unexpected output");
    if (!fsr && !ngx_init()) {
        s.failed = true;
        return fail("DLSS not available");
    }
    if (!make_pipelines()) {
        (fsr ? s.fsr_failed : s.failed) = true;
        return fail("its shaders were not made");
    }
    const VkImageView dview = depth_view(cam.depth);
    if (!dview) return fail("no view of the scene depth");
    // (DLAA: DLSS at the render size, anti-aliasing alone; its picture then scaled to the output's by dlss_out.comp. With
    // it the 3D is drawn at full scale, dlss_full_scale. FSR's native anti-aliasing the same)
    const bool native = native_aa(mode);
    const uint32_t dw = native ? rw : ow, dh = native ? rh : oh;
    // (from here commands are recorded, NGX's or FFX's own too: whichever way this returns, the program's descriptor pushes
    // are gone from the bind point and are pushed again)
    struct Disturbed {
        ~Disturbed() { descriptors_disturbed(); }
    } disturbed;
    uint32_t iw = 0, ih = 0;   // (the inputs' size: the largest render size the feature or upscaler takes)
    // (relink, 0.7.9.3: RX_VK_UPSCALE_INPUT=hdr, DLSS and FSR (made for HDR colour): the HDR scene kept at the game's composite, through its colour
    // matrix and exposure; after the upscale the bloom is added and ACES applied at the output's size (dlss_post.comp modes 4
    // and 5))
    const uint64_t now0 = texture_frame_no();
    const bool hdr_path = upscale_input() == 2 && s.hdr.image && s.hdr.width == rw && s.hdr.height == rh &&
                          now0 - s.hdr_frame <= 1 && s.bloom && now0 - s.bloom_frame <= 1 && make_post_pipeline(s);
    // (a frame without them is left to the game's network: DLSS's feature or FSR's upscaler made again for the tonemapped
    // colour and back cost up to a second each (FSR: 988 ms measured), and the first frame would make it so)
    if (upscale_input() == 2 && !hdr_path) return fail("no HDR scene this frame");
    if (fsr) {
        if (!make_fsr(rw, rh, dw, dh, cam, hdr_path)) return false;
        iw = s.fsr_iw, ih = s.fsr_ih;
    } else {
#ifdef RELINK_NGX
        if (!make_feature(cmd, rw, rh, dw, dh, hdr_path)) return false;
        iw = s.feature_rw, ih = s.feature_rh;
#endif
    }
    if (!make_image(cmd, s.color, VK_FORMAT_R16G16B16A16_SFLOAT, iw, ih) ||
        !make_image(cmd, s.depth, VK_FORMAT_R32_SFLOAT, iw, ih) ||
        !make_image(cmd, s.motion, VK_FORMAT_R16G16_SFLOAT, iw, ih, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
        !make_image(cmd, s.output, VK_FORMAT_R16G16B16A16_SFLOAT, dw, dh))
        return fail("its images were not made");

    const uint64_t now = texture_frame_no();
    const uint64_t last = s.last_frame;
    bool reset = !last || now - last > 2;
    struct {
        float reproject[16];
        float jitter[2];
        int32_t size[2];
        int32_t flags;
    } pc{};
    static const bool show_motion = getenv("RX_VK_DLSS_DEBUG") && !strcmp(getenv("RX_VK_DLSS_DEBUG"), "motion");
    // (the motion without the jitter: from this frame's jittered clip space to the frame before's without its jitter.
    // A jitter moves clip x by jitter * w: taken off the previous matrix's first two rows)
    // (the frame before's camera as DLSS last drew it: the buffer's own slot for it, 0x180, holds zeros in this version)
    float previous[16];
    std::memcpy(previous, s.last_vp, sizeof previous);
    dlss_math::unjitter(previous, s.last_jitter);
    bool motion = !reset && dlss_math::reprojection(cam.current, previous, pc.reproject);
    // (relink, 0.7.9.3) A camera cut (a cutscene's, a door, a warp: dlss_math::camera_cut), or a frame whose reprojection
    // cannot be made: the history is of another view, and DLSS and FSR both ask to be told (their reset). It was kept, with
    // a motion of the wrong view or of zero, and the view before showed through the new one for a few frames.
    // RX_VK_UPSCALER_CUT_RESET=0: kept, as before
    static const bool cut_reset = env_switch("RX_VK_UPSCALER_CUT_RESET") != 0;
    if (!reset && cut_reset && (!motion || dlss_math::camera_cut(pc.reproject))) {
        static std::atomic<int> told{0};
        if (told++ < 6)
            logf("vk: %s: a camera cut at frame %llu (%s): its history is reset", backend_name(mode), (unsigned long long)now,
                 motion ? "the view before is not this one" : "no reprojection from the frame before");
        reset = true, motion = false;
    }
    pc.jitter[0] = cam.jitter[0], pc.jitter[1] = cam.jitter[1];
    // (relink, 0.7.9.3) FSR 3.1 given linear colour: its provider ignores the dispatch's sRGB flag (AMD's FSR 3.1 source reads
    // no such flag), so the square root of the colour was accumulated and weighed as if it were linear. dlss_prep.comp squares
    // it (bit 3) and dlss_out.comp takes FSR's linear picture back. FSR 4 is told the colour is gamma-encoded at its creation:
    // it keeps it. RX_VK_FSR_LINEAR=0 / 1 over the default
    static const int fsr_linear_env = env_switch("RX_VK_FSR_LINEAR");
    const bool fsr4_used = fsr && !(s.fsr_flags & FSR_V31) && fsr4_available();
    const bool fsr_linear = fsr && !fsr4_used && fsr_linear_env > 0 && !hdr_path;
    pc.size[0] = (int32_t)rw, pc.size[1] = (int32_t)rh;
    // (relink, 0.7.9.3: RX_VK_UPSCALE_INPUT=composite: the game's composite before its SMAA, this frame's, the scene's size)
    VkImageView color_in = in_view;
    bool from_composite = false;
    // (noted this frame or the one before: a frame's pieces are recorded out of order on several threads, and the composite's
    // draw runs before the upscale on the GPU whichever piece was recorded first)
    if (upscale_input() == 1 && s.composite && now - s.composite_frame <= 1 && s.composite_gen == texture_destroy_generation() &&
        s.composite->width == rw && s.composite->height == rh) {
        if (const VkImageView v = scene_color_view(s, s.composite)) color_in = v, from_composite = true;
    }
    // (RX_VK_POST_TEST: dlss_post.comp's composite at the render size, or its difference with the game's, instead)
    if (post_test() >= 0 && from_composite && now - s.hdr_frame <= 1 && s.hdr.width == rw && s.hdr.height == rh && s.bloom &&
        now - s.bloom_frame <= 1 && make_post_pipeline(s) && make_image(cmd, s.post_out, VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh)) {
        full_barrier(cmd);
        post_pass(cmd, s, post_test(), s.hdr.view, scene_color_view(s, s.bloom), color_in, s.post_out.view, rw, rh);
        full_barrier(cmd);
        color_in = s.post_out.view;
        static int told = 0;
        if (told++ < 1) logf("vk: the upscalers' composite test: mode %d, constants %.4f %.4f %.4f %.4f / power %.4f exposure %.4f",
                             post_test(), s.post_consts[0], s.post_consts[1], s.post_consts[2], s.post_consts[3], s.post_consts[13],
                             s.post_consts[14]);
    }
    if (hdr_path && make_image(cmd, s.post_out, VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh)) {
        full_barrier(cmd);
        post_pass(cmd, s, 4, s.hdr.view, VK_NULL_HANDLE, VK_NULL_HANDLE, s.post_out.view, rw, rh);
        full_barrier(cmd);
        color_in = s.post_out.view, from_composite = false;
    }
    pc.flags = (motion ? 1 : 0) | (show_motion ? 2 : 0) | (fsr_linear ? 8 : 0) | (from_composite ? 16 : 0) | (hdr_path ? 32 : 0);
    if (from_composite != s.told_composite) {
        s.told_composite = from_composite;
        logf("vk: %s: given %s", backend_name(mode), from_composite ? "the game's composite before its SMAA" : "the network's input");
    }
    (hdr_path ? s.input_hdr : from_composite ? s.input_composite : s.input_network)++;
    if (s.input_composite + s.input_network + s.input_hdr >= 600) {
        logf("vk: %s: of the last %llu upscales %llu took the composite before SMAA, %llu the HDR scene, %llu the network's input",
             backend_name(mode), (unsigned long long)(s.input_composite + s.input_network + s.input_hdr),
             (unsigned long long)s.input_composite, (unsigned long long)s.input_hdr, (unsigned long long)s.input_network);
        s.input_composite = s.input_network = s.input_hdr = 0;
    }

    full_barrier(cmd);   // (the scene's colour and depth, written before)
    // (relink, 0.7.9.3: this frame's see-through mask, when the scene's pieces made one: dlss_mask.comp mode 2 clears one of
    // an older frame, so a frame without blended draws gives none)
    static const bool show_mask = getenv("RX_VK_DLSS_DEBUG") && !strcmp(getenv("RX_VK_DLSS_DEBUG"), "mask");
    const bool use_mask = mask_on() && s.mask_pipe && s.mask.image && s.mask.width == rw && s.mask.height == rh &&
                          s.delta[0].width == rw && s.delta[0].height == rh && s.delta[1].width == rw &&
                          s.delta[1].height == rh && s.mask_flags.image && now - s.mask_frame <= 1;
    if (use_mask) s.mask_given++;
    if (mask_on() && ++s.mask_logged % 600 == 0) {
        logf("vk: see-through mask: of the last 600 upscales %llu given it; %llu frames took the solid picture (%llu times), %llu "
             "masks made", (unsigned long long)s.mask_given, (unsigned long long)s.mask_frames, (unsigned long long)s.mask_snaps,
             (unsigned long long)s.mask_ends);
        s.mask_frames = s.mask_snaps = s.mask_given = s.mask_ends = 0;
    }
    {
        const VkDescriptorImageInfo info[5] = {{s.point, color_in, VK_IMAGE_LAYOUT_GENERAL},
                                               {s.point, dview, VK_IMAGE_LAYOUT_GENERAL},
                                               {VK_NULL_HANDLE, s.color.view, VK_IMAGE_LAYOUT_GENERAL},
                                               {VK_NULL_HANDLE, s.depth.view, VK_IMAGE_LAYOUT_GENERAL},
                                               {VK_NULL_HANDLE, s.motion.view, VK_IMAGE_LAYOUT_GENERAL}};
        VkWriteDescriptorSet w[5]{};
        for (uint32_t i = 0; i < 5; i++) {
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstBinding = i, w[i].descriptorCount = 1, w[i].pImageInfo = &info[i];
            w[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.prep);
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.prep_layout, 0, 5, w);
        vkCmdPushConstants(cmd, s.prep_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(cmd, (rw + 7) / 8, (rh + 7) / 8, 1);
    }
    full_barrier(cmd);
    // (what moves by itself: the skinned draws drawn again over the camera's motion, with their own)
    const int actors = actor_motion(cmd, cam, previous, rw, rh, dview, motion);
    if (actors) full_barrier(cmd);
    if (use_mask) {   // (the mask: this frame's difference against the frame before's, where the motion says it was)
        mask_pass(cmd, s, 6, 0, s.delta[(now - 1) & 1].view, s.opaque.view, rw, rh, s.delta[now & 1].view, s.motion.view);
        full_barrier(cmd);
    }
    if (show_mask && use_mask) {   // (RX_VK_DLSS_DEBUG=mask: the mask over the colour the upscaler is given)
        mask_pass(cmd, s, 4, 0, VK_NULL_HANDLE, s.color.view, rw, rh);
        full_barrier(cmd);
    }
    static const bool show_opaque = getenv("RX_VK_DLSS_DEBUG") && !strcmp(getenv("RX_VK_DLSS_DEBUG"), "opaque");
    if (show_opaque && use_mask) {   // (RX_VK_DLSS_DEBUG=opaque: the solid picture taken, instead of the colour)
        mask_pass(cmd, s, 5, 0, s.opaque.view, s.color.view, rw, rh);
        full_barrier(cmd);
    }
    if (show_motion && actors) {   // (RX_VK_DLSS_DEBUG=motion: the motion as it is now, the actors' in it)
        pc.flags = 4;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.prep);
        const VkDescriptorImageInfo info[5] = {{s.point, color_in, VK_IMAGE_LAYOUT_GENERAL},
                                               {s.point, dview, VK_IMAGE_LAYOUT_GENERAL},
                                               {VK_NULL_HANDLE, s.color.view, VK_IMAGE_LAYOUT_GENERAL},
                                               {VK_NULL_HANDLE, s.depth.view, VK_IMAGE_LAYOUT_GENERAL},
                                               {VK_NULL_HANDLE, s.motion.view, VK_IMAGE_LAYOUT_GENERAL}};
        VkWriteDescriptorSet w[5]{};
        for (uint32_t i = 0; i < 5; i++) {
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstBinding = i, w[i].descriptorCount = 1, w[i].pImageInfo = &info[i];
            w[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        }
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.prep_layout, 0, 5, w);
        vkCmdPushConstants(cmd, s.prep_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(cmd, (rw + 7) / 8, (rh + 7) / 8, 1);
        full_barrier(cmd);
    }
    // (the jitter in render pixels, x right and y down: DLSS's convention, and FFX's (its jitter offset is the camera's
    // sub-pixel offset in render pixels, y down, where a projection offset of (2 x / width, -2 y / height) in clip space is
    // (x, y)). RX_VK_FSR_JITTER=flip gives FSR the other sign, a test that it is this one: shaking and a softer picture)
    float jitter_px[2];
    dlss_math::jitter_to_pixels(cam.jitter, rw, rh, jitter_px);
    {
        // (a test that the sign is this one: RX_VK_DLSS_JITTER_TEST=flip gives DLSS or FSR the other sign (a shaking, softer
        // picture if ours is right); RX_VK_FSR_JITTER=flip as before for FSR alone)
        static const char* const test = getenv("RX_VK_DLSS_JITTER_TEST");
        static const bool flip = (test && !strcmp(test, "flip")) ||
                                 (fsr && getenv("RX_VK_FSR_JITTER") && !strcmp(getenv("RX_VK_FSR_JITTER"), "flip"));
        if (flip) jitter_px[0] = -jitter_px[0], jitter_px[1] = -jitter_px[1];
        // (RX_VK_DLSS_JITTER_LOG=1: how well the frames cover a pixel. The game takes its jitter from the frames shown
        // (host/rx_backend.c presents): a frame that got the same offset as the one before, or a sequence that skips, covers
        // the pixel with fewer points and the picture is softer. Every 600 frames: how many repeated the one before, and how
        // many distinct offsets the last ones (as many as the sequence's length) had)
        static const bool jlog = env_switch("RX_VK_DLSS_JITTER_LOG") > 0;
        if (jlog) {
            static float prev[2] = {1e9f, 1e9f};
            static uint64_t frames = 0, repeats = 0, zero = 0;
            static std::vector<std::pair<float, float>> ring;
            const uint32_t phases = s.jitter_phases.load(std::memory_order_relaxed);
            frames++;
            if (jitter_px[0] == prev[0] && jitter_px[1] == prev[1]) repeats++;
            if (jitter_px[0] == 0.0f && jitter_px[1] == 0.0f) zero++;
            prev[0] = jitter_px[0], prev[1] = jitter_px[1];
            ring.push_back({jitter_px[0], jitter_px[1]});
            if (phases && ring.size() > phases) ring.erase(ring.begin(), ring.end() - phases);
            if (frames % 600 == 0) {
                auto v = ring;
                std::sort(v.begin(), v.end());
                const size_t distinct = std::unique(v.begin(), v.end()) - v.begin();
                logf("vk: %s jitter: %llu frames, %llu the same as the frame before, %llu none; the last %zu: %zu distinct "
                     "(sequence %u); now %.3f %.3f px", backend_name(mode), (unsigned long long)frames,
                     (unsigned long long)repeats, (unsigned long long)zero, ring.size(), distinct, phases, jitter_px[0],
                     jitter_px[1]);
                frames = repeats = zero = 0;
            }
        }
    }
    if (fsr) {
        // (FFX takes its inputs in its compute-read state, SHADER_READ_ONLY_OPTIMAL, and leaves them so: they go back to
        // GENERAL after it, for the next frame's prep pass and actor motion. The output stays in GENERAL)
        const Image* inputs[4] = {&s.color, &s.depth, &s.motion, &s.mask};
        const int n_inputs = use_mask ? 4 : 3;
        layouts(cmd, inputs, n_inputs, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        FsrFrame f;
        f.color = {s.color.image, s.color.format, s.color.width, s.color.height};
        f.depth = {s.depth.image, s.depth.format, s.depth.width, s.depth.height};
        f.motion = {s.motion.image, s.motion.format, s.motion.width, s.motion.height};
        f.output = {s.output.image, s.output.format, s.output.width, s.output.height};
        if (use_mask) f.reactive = {s.mask.image, s.mask.format, s.mask.width, s.mask.height};
        f.jitter[0] = jitter_px[0], f.jitter[1] = jitter_px[1];
        // (dlss_prep.comp's motion is where the pixel was in the frame before minus where it is, in render pixels, without
        // the jitter: FFX's convention with a scale of 1, which it divides by the render size itself)
        f.motion_scale[0] = f.motion_scale[1] = 1.0f;
        f.render_w = rw, f.render_h = rh, f.out_w = dw, f.out_h = dh;
        f.reset = reset;
        const auto t = std::chrono::steady_clock::now();
        const double since = std::chrono::duration<double, std::milli>(t - s.fsr_last).count();
        f.frame_ms = reset || !(since > 0.0) ? 16.7f : (float)std::min(since, 200.0);
        s.fsr_last = t;
        // (the camera's planes and field of view from its projection (dlss_note_scene_draw); none read yet: OoT's usual ones)
        f.near_z = cam.near_z > 0 ? cam.near_z : 1.0f;
        f.far_z = cam.near_z > 0 ? cam.far_z : 100000.0f;
        f.fov_y = cam.fov_y > 0 ? cam.fov_y : 0.8f;
        f.meters = kFsrMeters;
        static const bool linear = env_switch("RX_VK_FSR_SRGB") == 0;
        f.perceptual = !linear && !fsr_linear && !hdr_path;   // (dlss_prep.comp's colour is the network's input: the square root of the tonemapped colour)
        const uint32_t r = fsr_dispatch(s.fsr, cmd, f);
        note_work();   // (FFX's own commands, which vk_work.h does not see)
        layouts(cmd, inputs, n_inputs, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        if (r) {
            logf("vk: AMD FSR: its dispatch failed (FFX's code %u): the game's upscale from now on", r);
            s.fsr_failed = true;
            return false;   // (the network writes its output over whatever the dispatch did)
        }
        if (const int phases = fsr_jitter_phases(s.fsr, rw, dw)) s.fsr_phases = phases;
    } else {
#ifdef RELINK_NGX
        NVSDK_NGX_Resource_VK color = resource(s.color, false), depth = resource(s.depth, false), mv = resource(s.motion, false),
                              output = resource(s.output, true);
        NVSDK_NGX_VK_DLSS_Eval_Params ep{};
        ep.Feature.pInColor = &color, ep.Feature.pInOutput = &output;
        ep.pInDepth = &depth, ep.pInMotionVectors = &mv;
        ep.InJitterOffsetX = jitter_px[0], ep.InJitterOffsetY = jitter_px[1];
        NVSDK_NGX_Resource_VK bias = use_mask ? resource(s.mask, false) : NVSDK_NGX_Resource_VK{};
        if (use_mask) ep.pInBiasCurrentColorMask = &bias, ep.InBiasCurrentColorSubrectBase = {0, 0};   // (relink: the see-through mask)
        ep.InRenderSubrectDimensions = {rw, rh};
        ep.InReset = reset ? 1 : 0;
        ep.InMVScaleX = ep.InMVScaleY = 1.0f;
        const NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, s.feature, s.params, &ep);
        note_work();   // (as at its creation: the barrier after it, and the game's next one, are recorded)
        if (NVSDK_NGX_FAILED(r)) {
            logf("vk: NVIDIA DLSS: its evaluation failed (%#x): the game's upscale from now on", (unsigned)r);
            s.failed = true;
            return false;   // (the network writes its output over whatever the evaluation did)
        }
#endif
    }
    full_barrier(cmd);
    VkImageView picture = s.output.view;
    if (hdr_path && make_image(cmd, s.post_final, VK_FORMAT_R16G16B16A16_SFLOAT, dw, dh)) {
        post_pass(cmd, s, 5, s.output.view, scene_color_view(s, s.bloom), VK_NULL_HANDLE, s.post_final.view, dw, dh);
        full_barrier(cmd);
        picture = s.post_final.view;
    }
    out_pass(cmd, s, picture, !native, out_view, ow, oh, fsr_linear, out16);
    s.last_frame = now;
    s.last_jitter[0] = cam.jitter[0], s.last_jitter[1] = cam.jitter[1];
    // (for the status line, dlss_status: the jitter DLSS (or FSR) was given, in render pixels)
    s.status_jitter[0] = jitter_px[0], s.status_jitter[1] = jitter_px[1], s.status_motion = motion;
    s.status_drawn++;

    std::memcpy(s.last_vp, cam.current, sizeof s.last_vp);
    // (the jitter's length: FFX's for FSR, NVIDIA's guide for DLSS: dlss_math.h)
    s.jitter_w = rw, s.jitter_h = rh;
    s.lod_bias = lod_bias_for(rw, dw);   // (relink: upscaler_lod_bias)
    s.jitter_phases = (uint32_t)(fsr && s.fsr_phases > 0 ? s.fsr_phases : dlss_jitter_phases(rw, rh, dw, dh));
    if (s.reason) logf("vk: %s draws the upscale again (frame %llu)", backend_name(mode), (unsigned long long)now);
    s.reason = nullptr;
    // (relink, 0.7.9.3: DLSS Frame Generation's inputs, when it is asked for; the picture as out_pass left it, before the HUD)
    // (a plain 2D view of the picture's image to read it: the view the network writes through reads as nothing)
    const VkImageView out_read = out && out_level == 0 && out->width == ow && out->height == oh ? scene_color_view(s, out) : VK_NULL_HANDLE;
    fg_capture(cmd, s, cam, previous, rw, rh, reset, out_read, ow, oh);
    // (RX_VK_DLSS_DEBUG=alternate, a test's reference: every other frame the game's own upscale over DLSS's, which still
    // draws and keeps its history. A frame of DLSS then has the game's picture of the frames either side to be measured against)
    static const bool alternate = getenv("RX_VK_DLSS_DEBUG") && !strcmp(getenv("RX_VK_DLSS_DEBUG"), "alternate");
    if (alternate && (now & 1)) return false;
    if (s.drawn_mode != mode) {   // (once for each Upscaler that draws)
        s.drawn_mode = mode;
        static const char* const names[upscaler_rows::kUpscalers] = {"", "NVIDIA DLSS", "NVIDIA DLAA", "AMD FSR",
                                                                     "AMD FSR native AA", "No upscaling"};
        logf("vk: %s draws OoT's upscale: %ux%u -> %ux%u%s", names[mode], rw, rh, ow, oh,
             show_motion ? " (its motion shown)" : "");
    }
    return true;
}

// ---- (relink, 0.7.9.3) NVIDIA DLSS Frame Generation ----
// NVIDIA's frame generator through NGX, as its DLSS-FG guide (310.7.0) gives it without Streamline: a feature made for the
// picture's size, evaluated once for each frame to make, in the present's recording, from the picture (with the HUD), the depth
// and motion DLSS (or DLAA, or FSR) was given for that frame, and the camera. vk_framegen.cpp's display thread shows the made
// frames and the game's, evenly, as it does its own; NGX neither times nor shows them. NVIDIA's runtime nvngx_dlssg.dll is looked
// for where DLSS's is (scripts/ngx_sdk.py fetches both). It runs on the RTX 40 series and later only (NGX says,
// FrameGeneration.Available): wherever it does not, or no upscaler of ours drew the frame, ReLink's own frames are made.
// Switches: RX_VK_DLSSG=0 / 1 over the Frame generator row; RX_VK_DLSSG_STUB=1 runs all but NGX's own calls on any GPU (the
// inputs found, the camera worked out and logged, ReLink's frames made: a test on a GPU without it); RX_VK_DLSSG_MATRIX=plain:
// the matrices in our layout instead of NGX's (transposed); RX_VK_DLSSG_MVSCALE=norm: the motion said to be in UV units (1/w,
// 1/h) instead of pixels; RX_VK_DLSSG_LOG=1: an evaluation's camera, the first 4 and then one every 600.
namespace {
bool fg_stub() {
    static const bool on = env_switch("RX_VK_DLSSG_STUB") > 0;
    return on;
}

/// (relink, 0.7.9.3) RX_VK_DLSSG_HUDLESS=1: DLSS Frame Generation is given the picture before the HUD (pHudless), so the HUD
/// is not moved with the 3D in the frames it makes. The game draws its HUD into the picture the upscale wrote (B10G11R11,
/// linear) and one draw then copies that into the window's sRGB texture, which the backbuffer is read from (decoded, into
/// RGBA8: vk_framegen.cpp fg_copy, which keeps the sRGB encoding). The upscale's picture sRGB-encoded into RGBA8 before the
/// HUD is drawn is that backbuffer without it, if that last draw changes nothing but the encoding (measured at the Market
/// outside the HUD: 1.25 / 255 off on average sRGB-encoded, 37.5 linear): RX_VK_DLSSG_HUDLESS_SHOW=1 (with the stub)
/// shows it as the made frame, so a shot's _made.png can be compared with the shot itself outside the HUD
bool hudless_on() {
    static const bool on = env_switch("RX_VK_DLSSG_HUDLESS") > 0 || env_switch("RX_VK_DLSSG_HUDLESS_SHOW") > 0;
    return on;
}
/// shaders/fg_copy.comp's pipeline, once: a sampled picture into an RGBA8 storage image (offset, size)
bool make_copy_pipeline(State& s) {
    if (s.copy_pipe && s.hud_pipe) return true;
    if (s.copy_failed || !dev.push_descriptor || !vkCmdPushDescriptorSetKHR || !make_pipelines()) return false;
    const VkDescriptorType types[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    if ((s.copy_pipe ||
         make_pipeline(FG_COPY_COMP_SPV, sizeof FG_COPY_COMP_SPV, types, 2, 16, s.copy_set, s.copy_layout, s.copy_pipe)) &&
        make_pipeline(DLSS_HUDLESS_COMP_SPV, sizeof DLSS_HUDLESS_COMP_SPV, types, 2, 16, s.hud_set, s.hud_layout, s.hud_pipe))
        return true;
    s.copy_failed = true;
    logf("vk: DLSS Frame Generation: the HUD-less picture's shader was not made: none given");
    return false;
}
/// One copy: encode, sRGB-encoded (dlss_hudless.comp), else as it is (fg_copy.comp)
void copy_pass(VkCommandBuffer cmd, State& s, VkImageView src, VkImageView dst, uint32_t w, uint32_t h, bool encode = false) {
    const VkPipeline pipe = encode ? s.hud_pipe : s.copy_pipe;
    const VkPipelineLayout layout = encode ? s.hud_layout : s.copy_layout;
    const int32_t pc[4] = {0, 0, (int32_t)w, (int32_t)h};
    const VkDescriptorImageInfo info[2] = {{s.point, src, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, dst, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    wr[0].dstBinding = 0, wr[0].descriptorCount = 1, wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[0].pImageInfo = &info[0];
    wr[1].dstBinding = 1, wr[1].descriptorCount = 1, wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[1].pImageInfo = &info[1];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, wr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
}
void fg_capture(VkCommandBuffer cmd, State& s, const Camera& cam, const float* before, uint32_t rw, uint32_t rh, bool reset,
                VkImageView out_view, uint32_t ow, uint32_t oh) {
    if (!dlssg_asked() || opts.framegen < 2) return;
    if (!fg_stub() && s.fg_available.load() == 0) return;
    State::FgSlot& f = s.fg_slots[s.fg_next];
    if (!make_image(cmd, f.depth, VK_FORMAT_R32_SFLOAT, s.depth.width, s.depth.height) ||
        !make_image(cmd, f.motion, VK_FORMAT_R16G16_SFLOAT, s.motion.width, s.motion.height)) {
        f.used = true;
        return;
    }
    full_barrier(cmd);   // (DLSS's inputs, read by it and written before)
    const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {rw, rh, 1}};
    vkCmdCopyImage(cmd, s.depth.image, VK_IMAGE_LAYOUT_GENERAL, f.depth.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    vkCmdCopyImage(cmd, s.motion.image, VK_IMAGE_LAYOUT_GENERAL, f.motion.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    f.has_hudless = false;
    if (hudless_on() && out_view && make_copy_pipeline(s) &&
        make_image(cmd, f.hudless, VK_FORMAT_R8G8B8A8_UNORM, ow, oh, VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        copy_pass(cmd, s, out_view, f.hudless.view, ow, oh, true);   // (out_pass's picture, written before the barrier above)
        f.has_hudless = true;
    }
    full_barrier(cmd);
    f.frame = texture_frame_no(), f.rw = rw, f.rh = rh;
    std::memcpy(f.vp, cam.current, sizeof f.vp);
    dlss_math::unjitter(f.vp, cam.jitter);
    std::memcpy(f.before, before, sizeof f.before);
    std::memcpy(f.proj, cam.proj, sizeof f.proj);
    dlss_math::unjitter(f.proj, cam.jitter);
    f.jitter[0] = cam.jitter[0], f.jitter[1] = cam.jitter[1];
    f.near_z = cam.near_z, f.far_z = cam.infinite ? 1e30f : cam.far_z, f.fov_y = cam.fov_y;
    f.reset = reset, f.used = false;
    s.fg_next ^= 1;
}
}  // namespace

bool dlssg_asked() {
    static const int env = env_switch("RX_VK_DLSSG");
    return env >= 0 ? env != 0 : opts.fg_kind == 1;
}

DlssgWhy dlssg_why() {
    if (fg_stub()) return DLSSG_OK;
#ifndef RELINK_NGX
    return DLSSG_NOT_BUILT;
#else
    const State& s = st();
    if (s.fg_failed) return DLSSG_FAILED;
    const int a = s.fg_available.load();
    return a < 0 ? DLSSG_UNKNOWN : (DlssgWhy)s.fg_why.load();
#endif
}

int dlssg_max_frames() {
    static const int stub_max = env_switch("RX_VK_DLSSG_STUB_MAX") > 0 ? env_switch("RX_VK_DLSSG_STUB_MAX") : 1;
    return fg_stub() ? stub_max : std::max(1, st().fg_max.load());
}

int dlssg_generate(VkCommandBuffer cmd, VkImage back, VkImageView back_view, const VkImage* out, const VkImageView* out_views,
                   int n, uint32_t w, uint32_t h, bool reset) {
    if (n <= 0 || !dlssg_asked() || !back || !back_view) return 0;
    State& s = st();
    std::lock_guard lk(s.mu);
#ifdef RELINK_NGX
    if (!s.ngx_tried && !fg_stub() && dlss_why() == DLSS_OK) ngx_init();   // (asks NGX of the frame generator too: fg_caps)
#endif
    if (s.fg_failed || (!fg_stub() && s.fg_available.load() != 1)) return 0;
    // (this frame's inputs: the slot an upscale of this frame or the one before filled, not used yet)
    const uint64_t now = texture_frame_no();
    State::FgSlot* f = nullptr;
    for (auto& k : s.fg_slots)
        if (!k.used && k.frame + 2 >= now && (!f || k.frame > f->frame)) f = &k;
    if (!f) {
        s.fg_missed++;
        return 0;
    }
    for (auto& k : s.fg_slots)
        if (k.frame <= f->frame) k.used = true;   // (an older one is not taken after this one)
    static const bool ours = getenv("RX_VK_DLSSG_MATRIX") && !strcmp(getenv("RX_VK_DLSSG_MATRIX"), "plain");
    dlss_math::FgCamera cam{};
    const bool camera = dlss_math::fg_camera(f->vp, f->before, f->proj, ours, &cam);
    // (NGX blends this picture with the last it was given: a frame without its inputs in between (ReLink's made) leaves that
    // one two frames old, so the history is reset then too)
    const bool cut = f->reset || reset || !camera || s.fg_eval_frame + 1 != f->frame;
    s.fg_eval_frame = f->frame;
    const int frames = std::min(n, dlssg_max_frames());
    static const bool trace = env_switch("RX_VK_DLSSG_LOG") > 0;
    static uint64_t traced = 0;   // (the first 4, then one every 600: a later one is the play's camera, not the title's)
    if ((trace || fg_stub()) && (traced++ < 4 || traced % 600 == 0)) {
        logf("vk: DLSS Frame Generation%s: frame %llu, picture %ux%u, depth and motion %ux%u of %ux%u, %d frame%s, reset %d; "
             "camera at %.1f %.1f %.1f, forward %.3f %.3f %.3f, up %.3f %.3f %.3f, near %.2f far %.0f fov %.3f; clip to the frame "
             "before's: %.4f %.4f %.4f %.4f / %.4f %.4f %.4f %.4f", fg_stub() ? " (stub)" : "", (unsigned long long)f->frame, w, h,
             f->rw, f->rh, f->depth.width, f->depth.height, frames, frames > 1 ? "s" : "", cut ? 1 : 0, cam.pos[0], cam.pos[1],
             cam.pos[2], cam.fwd[0], cam.fwd[1], cam.fwd[2], cam.up[0], cam.up[1], cam.up[2], f->near_z, f->far_z, f->fov_y,
             cam.clip_to_prev[0], cam.clip_to_prev[1], cam.clip_to_prev[2], cam.clip_to_prev[3], cam.clip_to_prev[4],
             cam.clip_to_prev[5], cam.clip_to_prev[6], cam.clip_to_prev[7]);
    }
    if (fg_stub()) {
        s.fg_frames++;
        // (RX_VK_DLSSG_HUDLESS_SHOW=1: the HUD-less picture as the made frame, a test of it against the shot of the backbuffer)
        static const bool show = env_switch("RX_VK_DLSSG_HUDLESS_SHOW") > 0;
        if (show && f->has_hudless && f->hudless.width == w && f->hudless.height == h && make_copy_pipeline(s)) {
            full_barrier(cmd);
            copy_pass(cmd, s, f->hudless.view, out_views[0], w, h);
            full_barrier(cmd);
            descriptors_disturbed();
            return 1;
        }
        return 0;   // (ReLink's own frames: the stub ends before NGX)
    }
#ifdef RELINK_NGX
    // (its feature for the picture's size; made again when the window's size changes)
    if (!s.fg_feature || s.fg_w != w || s.fg_h != h) {
        const uint64_t key = (uint64_t)w << 32 | h;
        if (s.fg_refused == key) return 0;
        if (s.fg_feature) {
            NVSDK_NGX_Handle* old = s.fg_feature;
            later([old] { NVSDK_NGX_VULKAN_ReleaseFeature(old); });
            s.fg_feature = nullptr;
        }
        if (!s.fg_params && (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&s.fg_params)) || !s.fg_params)) {
            s.fg_failed = true;
            logf("vk: NVIDIA DLSS Frame Generation: no parameters from NGX: ReLink's frames from now on");
            return 0;
        }
        NVSDK_NGX_DLSSG_Create_Params cp{};
        cp.Width = w, cp.Height = h, cp.NativeBackbufferFormat = VK_FORMAT_R8G8B8A8_UNORM;
        cp.RenderWidth = f->rw, cp.RenderHeight = f->rh, cp.DynamicResolutionScaling = true;
        // (the helper of this SDK sets the size and format only: the render size and the dynamic resolution by their names.
        // The game's Render resolution row moves the render size, and Auto moves it every few frames)
        NVSDK_NGX_Parameter_SetUI(s.fg_params, NVSDK_NGX_DLSSG_Parameter_InternalWidth, cp.RenderWidth);
        NVSDK_NGX_Parameter_SetUI(s.fg_params, NVSDK_NGX_DLSSG_Parameter_InternalHeight, cp.RenderHeight);
        NVSDK_NGX_Parameter_SetUI(s.fg_params, NVSDK_NGX_DLSSG_Parameter_DynamicResolution, 1);
        const auto t0 = std::chrono::steady_clock::now();
        const NVSDK_NGX_Result r = NGX_VK_CREATE_DLSSG(cmd, 1, 1, &s.fg_feature, s.fg_params, &cp);
        note_work();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (NVSDK_NGX_FAILED(r) || !s.fg_feature) {
            s.fg_feature = nullptr, s.fg_refused = key;
            logf("vk: NVIDIA DLSS Frame Generation: its feature was not made (%#x) for %ux%u: ReLink's frames until the window's "
                 "size changes", (unsigned)r, w, h);
            return 0;
        }
        s.fg_refused = 0, s.fg_w = w, s.fg_h = h, s.fg_made++;
        logf("vk: NVIDIA DLSS Frame Generation: feature %llu made for %ux%u (render %ux%u, dynamic), %.1f ms of CPU; up to %d "
             "frame%s between two of the game's", (unsigned long long)s.fg_made, w, h, f->rw, f->rh, ms, dlssg_max_frames(),
             dlssg_max_frames() > 1 ? "s" : "");
    }
    struct Disturbed {
        ~Disturbed() { descriptors_disturbed(); }
    } disturbed;
    const auto view_res = [](VkImageView v, VkImage i, VkFormat fmt, uint32_t rw, uint32_t rh, bool write) {
        return NVSDK_NGX_Create_ImageView_Resource_VK(v, i, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, fmt, rw, rh, write);
    };
    NVSDK_NGX_Resource_VK color = view_res(back_view, back, VK_FORMAT_R8G8B8A8_UNORM, w, h, false);
    NVSDK_NGX_Resource_VK depth = resource(f->depth, false), mv = resource(f->motion, false);
    // (RX_VK_DLSSG_HUDLESS=1: the picture before the HUD, when it is the backbuffer's size: fg_capture)
    const bool hudless = hudless_on() && f->has_hudless && f->hudless.width == w && f->hudless.height == h;
    NVSDK_NGX_Resource_VK hud = hudless ? resource(f->hudless, false) : NVSDK_NGX_Resource_VK{};
    static const bool norm = getenv("RX_VK_DLSSG_MVSCALE") && !strcmp(getenv("RX_VK_DLSSG_MVSCALE"), "norm");
    full_barrier(cmd);   // (the picture copied, the made frames free)
    for (int i = 1; i <= frames; i++) {
        NVSDK_NGX_Resource_VK made = view_res(out_views[i - 1], out[i - 1], VK_FORMAT_R8G8B8A8_UNORM, w, h, true);
        NVSDK_NGX_VK_DLSSG_Eval_Params ep{};
        ep.pBackbuffer = &color, ep.pDepth = &depth, ep.pMVecs = &mv, ep.pOutputInterpFrame = &made;
        if (hudless) ep.pHudless = &hud;
        NVSDK_NGX_DLSSG_Opt_Eval_Params op{};
        op.multiFrameCount = (unsigned)frames, op.multiFrameIndex = (unsigned)i;
        std::memcpy(op.cameraViewToClip, cam.view_to_clip, 64), std::memcpy(op.clipToCameraView, cam.clip_to_view, 64);
        std::memcpy(op.clipToPrevClip, cam.clip_to_prev, 64), std::memcpy(op.prevClipToClip, cam.prev_to_clip, 64);
        static const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        std::memcpy(op.clipToLensClip, identity, 64);
        op.jitterOffset[0] = f->jitter[0], op.jitterOffset[1] = -f->jitter[1];   // (clip space, Vulkan's y down)
        // (dlss_prep.comp's motion: where the pixel was minus where it is, in render pixels, y down, no jitter: NVIDIA's
        // convention with a scale of 1 as its parameter reference gives it)
        op.mvecScale[0] = norm ? 1.0f / (float)f->rw : 1.0f, op.mvecScale[1] = norm ? 1.0f / (float)f->rh : 1.0f;
        for (int k = 0; k < 3; k++)
            op.cameraPos[k] = cam.pos[k], op.cameraUp[k] = cam.up[k], op.cameraRight[k] = cam.right[k], op.cameraFwd[k] = cam.fwd[k];
        op.cameraNear = f->near_z > 0 ? f->near_z : 1.0f;
        op.cameraFar = f->far_z > 0 ? std::min(f->far_z, 3.0e38f) : 100000.0f;
        op.cameraFOV = f->fov_y > 0 ? f->fov_y : 0.8f;
        op.cameraAspectRatio = (float)f->rw / (float)std::max(1u, f->rh);
        op.colorBuffersHDR = false, op.depthInverted = false, op.cameraMotionIncluded = true;
        op.reset = cut, op.notRenderingGameFrames = false, op.orthoProjection = false;
        op.motionVectorsInvalidValue = 0.0f, op.motionVectorsDilated = false, op.menuDetectionEnabled = false;
        op.mvecsSubrectBase = {0, 0}, op.mvecsSubrectSize = {f->rw, f->rh};
        op.depthSubrectBase = {0, 0}, op.depthSubrectSize = {f->rw, f->rh};
        op.backbufferSubrectBase = {0, 0}, op.backbufferSubrectSize = {w, h};
        op.outputInterpSubrectBase = {0, 0}, op.outputInterpSubrectSize = {w, h};
        const NVSDK_NGX_Result r = NGX_VK_EVALUATE_DLSSG(cmd, s.fg_feature, s.fg_params, &ep, &op);
        note_work();
        if (NVSDK_NGX_FAILED(r)) {
            logf("vk: NVIDIA DLSS Frame Generation: its evaluation failed (%#x): ReLink's frames from now on", (unsigned)r);
            s.fg_failed = true;
            return 0;   // (the caller makes its own over whatever was written)
        }
    }
    full_barrier(cmd);
    s.fg_last = now, s.fg_frames++, s.holding = true, s.asked = now;
    return frames;
#else
    (void)cmd, (void)out, (void)out_views, (void)w, (void)h, (void)frames, (void)cut;
    return 0;
#endif
}

void dlssg_status(char* b, size_t n) {
    if (!n) return;
    b[0] = 0;
    State& s = st();
    std::lock_guard lk(s.mu);
    const uint64_t made = s.fg_frames - s.fg_status_frames, missed = s.fg_missed - s.fg_status_missed;
    if (!made && !missed) return;
    snprintf(b, n, "; DLSS Frame Generation%s: %llu frames given to it, %llu without this frame's inputs (ReLink's made)",
             fg_stub() ? " (stub)" : "", (unsigned long long)made, (unsigned long long)missed);
    s.fg_status_frames = s.fg_frames, s.fg_status_missed = s.fg_missed;
}

} // namespace mkwvk
