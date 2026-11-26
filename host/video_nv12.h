/* Small, device-independent helpers for decoder frames that can be kept as
   NV12 until a caller actually asks for RGBA. */
#pragma once

#include <stdint.h>
#include <string.h>

static inline void video_yuv420p_to_nv12(uint8_t *dst, int width, int height,
                                         const uint8_t *y, int y_stride,
                                         const uint8_t *u, int u_stride,
                                         const uint8_t *v, int v_stride) {
    for (int row = 0; row < height; row++)
        memcpy(dst + (uint64_t)row * width, y + (uint64_t)row * y_stride, (size_t)width);

    uint8_t *uv = dst + (uint64_t)width * height;
    for (int row = 0; row < height / 2; row++)
        for (int column = 0; column < width / 2; column++) {
            uv[(uint64_t)row * width + 2 * column] = u[(uint64_t)row * u_stride + column];
            uv[(uint64_t)row * width + 2 * column + 1] = v[(uint64_t)row * v_stride + column];
        }
}
