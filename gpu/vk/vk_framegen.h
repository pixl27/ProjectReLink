// Frame interpolation (vk_framegen.cpp): made frames shown between the game's (in a window; RX_VK_FRAMEGEN=0: off).
#pragma once

#include "vk_internal.h"

namespace mkwvk {

bool framegen_on();
/// (RX_VK_FG_LATENCY_LOG) Before the swapchain is destroyed (retire_swapchain): the log's waits for its presents end first
void fg_latency_swapchain_gone();
/// (hookshot) Frame generation's setting as it runs: the Insert menu's (opts.framegen: 0 Off, 1 Auto, 2 to 8 a fixed
/// multiplier), a fixed one at most NVIDIA DLSS Frame Generation's runtime's most made frames + 1 while DLSS-G is to make the
/// frames (vk_dlss.h dlss_fg_most: 6x on an RTX 50's runtime, 2x on an RTX 40's; that most is 0 otherwise: no cap). Past it
/// NGX was not asked for the sets, and the renderer's own or AMD FSR 3.1 made every frame with DLSS FG chosen. A saved value
/// past it is kept (another engine runs it), and is shown and run as that top. Off and Auto are as they are
int framegen_capped(int framegen, int dlss_most);
int framegen_now();
/// (hookshot) Each present: frame generation off for 120 presents lets its images go (the built-in interpolation's two frames, its
/// sets of made frames and their lumas and flows: about 112 MB at 2560x1440 2x, 823 MB at 4K 8x; FSR 3.1's frame interpolation's
/// images too, also while another one is chosen). They were let go only when the window's size changed, so a player who tried it
/// and turned it off kept them for the run. They are made again when it is turned on. recorded: this present recorded made frames
/// (counted as on: framegen_idle.h); FSR 3.1's images are counted by this present's own use of FSR, not by recorded. Called after
/// framegen_queue, so a set this present filled is the display thread's by then
void framegen_idle(bool recorded);
/// Records the frame's processing into the present's recording (src: the window texture's view, sw x sh, the crop
/// x, y, w, h; cover: wide_sides_tick's, this frame's; native_pq: the encoding of that captured present); false when
/// interpolation cannot run (the present goes to the window as without it)
bool framegen_record(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, const int crop[4], float cover, bool native_pq);
/// (hookshot) Who made the made frames of the sets shown lately (the last 64, about a second at 60): what the Insert menu's
/// Frame generation by row and the FPS counter name. None: none made (Off, nothing to blend from, none for half a second).
/// DlssPaused: DLSS FG chosen, and the sets were the game's frames alone, with no Upscaler picture to make frames from (the
/// game's own menus, movies and loading screens: vk_dlss.cpp dlss_fg_holds)
enum class FgMaker { None, BuiltIn, Dlss, Fsr3, DlssPaused };
FgMaker framegen_maker();
/// The first frame the last framegen_record made (null when it made none), w x h, GENERAL; with RX_VK_SHOT_MADE=all, each
/// call the next of its made frames in turn, its place in the set in index
VkImage framegen_made_image(uint32_t* w, uint32_t* h, int* index = nullptr);
/// (hookshot) The gameplay upscale's picture (its output, at level), recorded right after the upscale wrote it, before the
/// game draws its HUD into it: copied while AMD FSR 3.1's frame interpolation is asked for, so that framegen_record can mark
/// the HUD for it (what the game drew over the picture since). Anything else (null, another format or shape) is left alone
struct Texture;
void framegen_hud_before(VkCommandBuffer cmd, Texture* picture, uint32_t level);
/// After that recording's submit (value): the display thread shows its frames
void framegen_queue(uint64_t value);
/// The window's present of an image (GENERAL, w x h; crop as above or null), vk_replay.cpp: the display thread's
void present_image(VkImage src, uint32_t w, uint32_t h, const int crop[4], VkImageView view = VK_NULL_HANDLE);
/// (hookshot) Present ahead (vk_framegen.cpp display_main, #1702): while present_defer_into is set on a thread, present_image
/// submits the frame's copy into its swapchain image and stops there, with what present_commit needs to show it later. The
/// copy is then on the queue before the game's next frame, so its present at its time does not wait for that frame's rendering
struct PendingPresent {
    bool used = false;
    uint32_t index = 0;
    VkFence done = VK_NULL_HANDLE;           // (the copy's fence)
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    uint64_t reflex_frame = ~0ull;           // (NVIDIA Reflex: the frame it shows, vk_reflex.h reflex_present_frame)
    uint64_t antilag_frame = ~0ull;          // (AMD Anti-Lag 2: the frame it shows, vk_antilag.h antilag_present_frame)
    uint64_t swapchain_generation = 0;
};
extern thread_local PendingPresent* present_defer_into;
bool present_commit(const PendingPresent& p);   // (false: not shown, nothing deferred or a swapchain made again since)
/// The Insert menu's output effects at dw x dh: window scaling/sharpness/sides, then the optional SDR CRT.
/// Returns null when Off or unavailable, leaving the existing direct-copy path. CRT runs after frame generation.
/// pq: src is Native HDR10, scaled in RGB10A2; the SDR CRT is bypassed, never applied to PQ codes.
VkImage framegen_scale(VkCommandBuffer cmd, VkImageView src, VkImage image, uint32_t sw, uint32_t sh, const int crop[4], uint32_t dw, uint32_t dh,
                       bool pq = false);
VkImageView framegen_scaled_view();   // final scale/CRT view for the existing HDR expansion
/// (hookshot) The HDR10 picture of src's crop at dw x dh (the Insert menu's HDR), GENERAL; null: none (the SDR picture then;
/// Native's scaled picture, else black: present_image).
/// pq: src is the game's own HDR10 picture (Native), its PQ codes as they are at 100% contrast, else graded and written again;
/// else the SDR picture, expanded
VkImage framegen_hdr(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, const int crop[4], uint32_t dw, uint32_t dh,
                     bool pq = false);
/// Each present: ultrawide's menus (the Insert menu's setting to the game's side, the sides blurred or black while a menu shows);
/// cover: how far the loading cover was closed when the game presented this frame (wide_cover_sample), the sides closed by as much
void wide_sides_tick(float cover);
/// How far the loading cover between two areas is closed now, 0 to 1 (0 unless ultrawide's menus are in the middle): asked as the
/// game presents a frame, on its thread, and carried with that frame to wide_sides_tick, since the game goes on to the next one.
/// The larger of this reading and the last (wide_cover_held): the game may already have moved the cover on for its next frame
float wide_cover_sample();
/// Whether ultrawide's menu sides can show at all: Ultrawide with a margin and the Insert menu's setting on
bool wide_sides_possible();
/// (hookshot) The Gameplay tab's Item wheel row is on (off by default; HOS_ITEM_WHEEL over it). While it is
/// false the two below record nothing and make nothing
bool framegen_wheel_on();
/// With the Item wheel row on: the gameplay upscale's output (image, its view format, w x h at level, GENERAL) as the upscale
/// left it, the scene before the game draws its HUD on it, copied in the frame's own work (the game stops drawing its scene
/// while its item row is shown: the frame before the row opens is the backdrop)
void framegen_wheel_scene(VkCommandBuffer cmd, VkImage image, VkFormat format, uint32_t w, uint32_t h, uint32_t level);
/// While the item wheel is open: the window texture's crop (dst, GENERAL, the game's RGBA8 picture) covered with the scene
/// framegen_wheel_scene last copied, blurred and dimmed, and the wheel (vk_wheel.cpp) over it. False (nothing recorded) when
/// the wheel is closed, no scene was copied since the row was turned on, or the picture is not that scene's
bool framegen_wheel_backdrop(VkCommandBuffer cmd, VkImage dst, VkFormat format, uint32_t sw, uint32_t sh, const int crop[4]);

} // namespace mkwvk
