// (relink, from Longshot) The item wheel's picture (vk_wheel.cpp): host/oot_wheel.c's state (MkwVkHost wheel) drawn on the CPU
// in the game's picture's size, and laid over the scene by vk_framegen.cpp framegen_wheel_backdrop, blurred and dimmed under it.
#pragma once

#include <cstdint>

namespace mkwvk {

/// What wheel_picture drew: an area w x h at (x, y) of the game's picture, premultiplied RGBA8 rows, top down. version
/// changes whenever the pixels do (the backdrop uploads them again only then)
struct WheelPicture {
    const uint8_t* px = nullptr;
    uint32_t w = 0, h = 0;
    int x = 0, y = 0;
    uint64_t version = 0;
};

/// The item wheel for a game picture of pw x ph, this frame: false while it is closed (or the Item wheel row is off).
/// The present's thread only
bool wheel_picture(uint32_t pw, uint32_t ph, WheelPicture* out);

}  // namespace mkwvk
