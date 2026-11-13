/* Weak stand-ins for the WebGPU renderer (gpu/render, renamed wgr_* by rx_wgpu_names.h) and its debug hooks, for a build
   without wgpu-native: build.py then skips gpu/render, and host/rx_backend.c and host/nn_hid.c still reference these.
   The Vulkan back end (RX_BACKEND=vulkan, relink_vk.dll) does the rendering; wgr_start fails, so a machine where the
   Vulkan back end cannot start reports it instead of falling back. With wgpu-native present the real definitions win. */
#include <stdint.h>

#define WEAK __attribute__((weak))

typedef struct nvn_tex nvn_tex;
typedef struct nvn_view nvn_view;
typedef struct nvn_samp_par nvn_samp_par;

WEAK int wgr_start(void *hwnd) { (void)hwnd; return 0; }
WEAK void wgr_resize(int width, int height) { (void)width; (void)height; }
WEAK const char *wgr_describe(void) { return "no WebGPU renderer in this build"; }
WEAK void wgr_stats(int64_t *draws, int64_t *skipped, int64_t *pipelines) {
    if (draws) *draws = 0;
    if (skipped) *skipped = 0;
    if (pipelines) *pipelines = 0;
}
WEAK void wgr_nvn_submit(const uint64_t *handles, int n) { (void)handles; (void)n; }
WEAK uint64_t wgr_nvn_fence(void) { return 0; }
WEAK uint64_t wgr_nvn_pushed(void) { return 0; }
WEAK uint64_t wgr_nvn_replayed(void) { return 0; }
WEAK int wgr_nvn_wait(uint64_t fence, uint64_t timeout_ns) { (void)fence; (void)timeout_ns; return 1; }
WEAK void wgr_wait_writebacks_reading(const void *p, uint64_t n) { (void)p; (void)n; }
WEAK void wgr_nvn_finish(void) {}
WEAK void wgr_nvn_present(const nvn_tex *t, const int crop[4]) { (void)t; (void)crop; }
WEAK void wgr_nvn_register_texture(int id, const nvn_tex *t, const nvn_view *v) { (void)id; (void)t; (void)v; }
WEAK void wgr_nvn_register_sampler(int id, const nvn_samp_par *p) { (void)id; (void)p; }
WEAK void wgr_nvn_set_texture_pool_size(int n) { (void)n; }
WEAK void wgr_nvn_texture_released(const nvn_tex *t) { (void)t; }
typedef struct nvn_prog nvn_prog;
typedef struct NVNcopyRegion NVNcopyRegion;
WEAK void wgr_nvn_program_released(const nvn_prog *p) { (void)p; }
WEAK void wgr_nvn_read_texels(const nvn_tex *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst,
                              uint64_t row, uint64_t img) {
    (void)t; (void)level; (void)layer0; (void)r; (void)dst; (void)row; (void)img;
}
WEAK void wgr_nvn_write_texels(const nvn_tex *t, unsigned level, unsigned layer0, const NVNcopyRegion *r,
                               const uint8_t *src, uint64_t row, uint64_t img) {
    (void)t; (void)level; (void)layer0; (void)r; (void)src; (void)row; (void)img;
}
WEAK void wgr_nvn_invalidate(uint64_t gpu, uint64_t size) { (void)gpu; (void)size; }

WEAK void rx_flash_dark_arm(void) {}
WEAK void rx_request_frame_dump(void) {}
WEAK void rx_request_rdoc_capture(int frames) { (void)frames; }
WEAK void rx_set_screenshot_ms(int64_t ms) { (void)ms; }
