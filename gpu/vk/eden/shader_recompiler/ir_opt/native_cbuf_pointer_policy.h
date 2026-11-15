// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace native_cbuf_pointer_policy {

/// A geometry-stage global-memory access tracked to a non-driver constant bank must stay
/// on the run-time address search when the candidate is not a mapped GPU pointer. Native
/// argument tables mix pointers with handles, sizes, strides, and floating-point values.
///
/// The memory-origin case predates the native validity check: rebasing an address that is
/// partly loaded through memory onto one constant-buffer candidate loses the loaded path.
inline bool leave_global(bool native_program, bool graphics_stage, bool geometry_stage,
                         uint32_t cbuf_index, bool address_loads_memory,
                         bool mapped_pointer) noexcept {
    if (cbuf_index == 0) {
        return false;
    }
    // The older memory-origin workaround is deliberately geometry-only: applying it to
    // fragment shaders changed valid lighting paths. Rejecting an unmapped native scalar is
    // a narrower invariant and applies to every graphics stage, including fragment shaders.
    return (geometry_stage && address_loads_memory) ||
           (graphics_stage && native_program && !mapped_pointer);
}

} // namespace native_cbuf_pointer_policy
