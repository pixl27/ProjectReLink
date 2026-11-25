// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>

namespace mkwvk::menu_fade {

enum class Action { Native, Capture, Hold, Blend };
struct Present {
    Action action = Action::Native;
    uint32_t opacity = 0;  // thousandths of the saved frame
};

// Updated only by the game's present recording. Loading without a preceding
// menu never uses the saved image. The game's own menu draws its backdrop.
struct Transition {
    int64_t captured_at = -1;
    int64_t fade_at = -1;
    int64_t closed_at = -1;
    bool armed = false;

    Present step(int64_t now, bool menu, bool world) {
        constexpr int64_t capture_interval = 100'000'000;
        constexpr int64_t max_capture_age = 2'000'000'000;
        constexpr int64_t max_hold = 1'500'000'000;
        constexpr int64_t fade_duration = 240'000'000;
        if (menu) {
            if (closed_at >= 0 || fade_at >= 0) armed = false;
            if (!armed && captured_at >= 0 && now - captured_at <= max_capture_age) armed = true;
            closed_at = fade_at = -1;
            return {};
        }
        if (armed) {
            if (closed_at < 0) closed_at = now;
            if (!world && fade_at < 0 && now - closed_at < max_hold)
                return {Action::Hold, 1000};
            if (world) {
                if (fade_at < 0) fade_at = now;
                const int64_t elapsed = now - fade_at;
                if (elapsed < fade_duration)
                    return {Action::Blend, uint32_t((fade_duration - elapsed) * 1000 / fade_duration)};
            }
            armed = false;
            closed_at = fade_at = -1;
            captured_at = -1;
        }
        if (world && (captured_at < 0 || now - captured_at >= capture_interval))
            return {Action::Capture, 0};
        return {};
    }
};

}  // namespace mkwvk::menu_fade
