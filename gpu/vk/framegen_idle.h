// SPDX-License-Identifier: GPL-3.0-or-later
// (hookshot) When frame generation's images are let go (vk_framegen.cpp framegen_idle): after kIdlePresents presents in a row with
// it off. A present that recorded made frames counts as on whatever the row says by then: its set is busy until the display thread
// shows it, and a drain then would wait for a job that is only queued after (the hang a review found: off for 119 presents, a
// present that records with it on and reads Off after). Nothing of Vulkan here: tests/test_framegen_idle.py runs it.
#pragma once
#include <cstdint>

namespace framegen_idle_rule {

constexpr uint32_t kIdlePresents = 120;   // (about 2 s at 60: a row stepped through on the way to another keeps them)

/// The presents in a row off, after this one: on now, or made frames recorded this present, starts again
inline uint32_t next_off(uint32_t off, bool on_now, bool recorded) { return on_now || recorded ? 0 : off + 1; }

/// Whether this present lets them go: there are some, and it has been off long enough
inline bool release_due(uint32_t off, bool has_images) { return has_images && off >= kIdlePresents; }

/// One present, as framegen_idle sees it after the present's queue
struct Present {
    bool on;          // frame generation on by the check (framegen_on: the row, read again)
    bool fsr_wanted;  // FSR 3.1's frame interpolation asked by the check (fsr3fi::frames_wanted)
    bool recorded;    // this present recorded made frames, by any of them (framegen_record's true)
    bool fsr_used;    // this present gave FSR 3.1 a frame: fsr3fi::generate called (its history alone or a reset too), whatever
                      // came of it and whichever made frames were shown
    bool has_images;  // the shared images are there (fg.w)
};

/// framegen_idle's two counts, and what one present lets go. The shared images (the built-in interpolation's frames and sets,
/// which every one of them records into) count any recorded present as on; FSR 3.1's own images only a present that used FSR. A
/// set recorded by DLSS-G or the built-in interpolation leaves them unused: fed `recorded`, the FSR count stayed 0 while another
/// one ran and its images were kept for the run (a second review)
struct Counts {
    uint32_t off = 0, fsr_off = 0;
};
struct Release {
    bool shared, fsr;
};
inline Release step(Counts& c, const Present& p) {
    c.fsr_off = next_off(c.fsr_off, p.on && p.fsr_wanted, p.fsr_used);
    c.off = next_off(c.off, p.on, p.recorded);
    return {release_due(c.off, p.has_images), c.fsr_off >= kIdlePresents};
}

}  // namespace framegen_idle_rule
