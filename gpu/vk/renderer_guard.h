// (hookshot) Every function the renderer gives the host (MkwVkRenderer, vk_api.h) behind cxx_escape's Entry: a C++
// exception out of one is said, with the function's name, and stops the game (cxx_escape.h escaped), where it went on into the
// host's C and ended the game with exception 0x20474343. One more call a function, which cannot be inlined across the table anyway.
// tests/test_cxx_escape.py checks that this list is every member of MkwVkRenderer but version.
#pragma once
#include "cxx_escape.h"
#include "vk_api.h"

namespace mkwvk {

inline void guard_renderer(MkwVkRenderer& t) {
#define MKWVK_GUARD(m) cxx_escape::Entry<&MkwVkRenderer::m>::wrap(t, #m)
    MKWVK_GUARD(start);
    MKWVK_GUARD(resize);
    MKWVK_GUARD(describe);
    MKWVK_GUARD(stats);
    MKWVK_GUARD(nvn_submit);
    MKWVK_GUARD(nvn_fence);
    MKWVK_GUARD(nvn_pushed);
    MKWVK_GUARD(nvn_replayed);
    MKWVK_GUARD(nvn_wait);
    MKWVK_GUARD(wait_writebacks_reading);
    MKWVK_GUARD(nvn_finish);
    MKWVK_GUARD(nvn_present);
    MKWVK_GUARD(nvn_register_texture);
    MKWVK_GUARD(nvn_register_sampler);
    MKWVK_GUARD(nvn_set_texture_pool_size);
    MKWVK_GUARD(nvn_texture_released);
    MKWVK_GUARD(nvn_program_released);
    MKWVK_GUARD(nvn_read_texels);
    MKWVK_GUARD(nvn_write_texels);
    MKWVK_GUARD(nvn_invalidate);
    MKWVK_GUARD(nvn_pool_created);
    MKWVK_GUARD(guest_imported);
    MKWVK_GUARD(nvn_vmap_changing);
    MKWVK_GUARD(nvn_texture_created);
    MKWVK_GUARD(nvn_event_signal);
    MKWVK_GUARD(option);
    MKWVK_GUARD(nvn_submit_q);
    MKWVK_GUARD(nvn_fence_q);
    MKWVK_GUARD(option_set);
    MKWVK_GUARD(jitter_sample);
    MKWVK_GUARD(frame_info);
    MKWVK_GUARD(shot);
    MKWVK_GUARD(setting);
    MKWVK_GUARD(setting_set);
    MKWVK_GUARD(nvn_program_created);
    MKWVK_GUARD(guest_release);
    MKWVK_GUARD(nvn_invalidate_quick);
#undef MKWVK_GUARD
}

}   // namespace mkwvk
