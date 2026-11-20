// Frame interpolation (vk_framegen.cpp): made frames shown between the game's (in a window; RX_VK_FRAMEGEN=0: off).
#pragma once

#include "vk_internal.h"

namespace mkwvk {

bool framegen_on();
/// Records the frame's processing into the present's recording (src: the window texture's view, sw x sh, the crop
/// x, y, w, h); false when interpolation cannot run (the present goes to the window as without it)
bool framegen_record(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, const int crop[4]);
/// The first frame the last framegen_record made (null when it made none), w x h, GENERAL
VkImage framegen_made_image(uint32_t* w, uint32_t* h);
/// After that recording's submit (value): the display thread shows its frames
void framegen_queue(uint64_t value);
/// The window's present of an image (GENERAL, w x h; crop as above or null), vk_replay.cpp: the display thread's.
/// must_sample: the image is to be read through view (its format is not the picture's), never blitted as it is
void present_image(VkImage src, uint32_t w, uint32_t h, const int crop[4], VkImageView view = VK_NULL_HANDLE,
                   bool must_sample = false);
/// The Insert menu's upscaling of a frame to the window (dw x dh), recorded: the image to show, or null when it is off
/// (force: made even with the upscaling and the sharpness off, as a plain copy through the view)
VkImage framegen_scale(VkCommandBuffer cmd, VkImageView src, VkImage image, uint32_t sw, uint32_t sh, const int crop[4], uint32_t dw, uint32_t dh,
                       bool force = false);
/// (relink) The Gameplay tab's Item wheel row is on (off by default; HOS_ITEM_WHEEL over it). While it is
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
