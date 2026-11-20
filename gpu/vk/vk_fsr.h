// (relink) AMD FidelityFX Super Resolution 3.1's upscaler (vk_fsr.cpp), for vk_dlss.cpp: AMD's own runtime,
// amd_fidelityfx_vk.dll, loaded at run time through AMD's FidelityFX API (its MIT headers in third_party/ffx_api). In every
// build: nothing of it is linked, so a game folder without the DLL (or a GPU it cannot run on) only loses the choice.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vk_internal.h"

namespace mkwvk {

/// Why FSR cannot be chosen here (the Insert menu's Upscaler row says it); FSR_OK when it can. Known once fsr_start ran
enum FsrWhy {
    FSR_OK,
    FSR_NOT_STARTED,      // (the device is not made yet)
    FSR_NOT_WINDOWS,      // (AMD's runtime is a Windows DLL)
    FSR_ENV_OFF,          // (RX_VK_FSR=0)
    FSR_NO_RUNTIME,       // (no amd_fidelityfx_vk.dll in the game folder nor beside the exe)
    FSR_RUNTIME_FAILED,   // (it is there but did not load, or is not AMD's FidelityFX API)
    FSR_NO_UPSCALER,      // (it has no upscaler for Vulkan)
    FSR_DEVICE,           // (it could not make an upscaler on this device)
    FSR_WHY_COUNT
};
FsrWhy fsr_why();
/// The device's extensions AMD's runtime takes by name (vkGetDeviceProcAddr gives nothing for one not enabled), added to
/// `want` when the device has them (all: what it has)
void fsr_device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& all);
/// Once the device is made (vk_main.cpp): loads AMD's runtime and makes a small upscaler on this device and lets it go, so
/// the Upscaler row knows before the first frame. Logs one line: available, or why not
void fsr_start();
/// The runtime's upscaler, as it names its version ("" before fsr_start or without one)
const char* fsr_version();
/// FFX's opt-in RCAS strength from RX_VK_FSR_SHARPNESS, clamped to [0, 1]. The final-picture scaler uses this to avoid
/// applying the menu's separate sharpness pass over the HUD as well while FSR's RCAS is active.
float fsr_rcas_sharpness();

/// An upscaler for these sizes (FFX's context). Its creation flags: FSR_DEPTH_INFINITE (the projection has no far plane),
/// FSR_DYNAMIC (the render size changes from frame to frame: the Render resolution row's Auto)
struct FsrContext;
enum { FSR_DEPTH_INFINITE = 1, FSR_DYNAMIC = 2 };
/// nullptr when it was not made (why in the log). Its creation records nothing: no command buffer is needed
FsrContext* fsr_create(uint32_t max_render_w, uint32_t max_render_h, uint32_t out_w, uint32_t out_h, unsigned flags);
/// Once the GPU no longer runs what it recorded (vk_dlss.cpp defers it a few frames)
void fsr_destroy(FsrContext* c);
/// The jitter sequence's length FFX asks for at this ratio (its query); 0 when it does not say
int fsr_jitter_phases(FsrContext* c, uint32_t render_w, uint32_t out_w);
/// The video memory the context holds, in bytes; 0 when it does not say
uint64_t fsr_memory(FsrContext* c);

/// One frame's upscale. The images are vk_dlss.cpp's own (made with storage and sampled use). The inputs must be in
/// VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (FFX's compute-read state) and stay so; the output in VK_IMAGE_LAYOUT_GENERAL
/// (its unordered-access state) and stays so
struct FsrImage {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
};
struct FsrFrame {
    FsrImage color, depth, motion, output;
    float jitter[2]{};          // (the camera's sub-pixel offset, in render pixels: x right, y down)
    float motion_scale[2]{};    // (what the motion vectors are multiplied by to be in render pixels)
    uint32_t render_w = 0, render_h = 0, out_w = 0, out_h = 0;
    bool reset = false;         // (no history: the first frame, or the camera cut)
    float frame_ms = 16.7f;
    float near_z = 0, far_z = 0, fov_y = 0;   // (the camera's planes, in its units, and its vertical field of view in radians)
    float meters = 1.0f;        // (a view-space unit in metres)
};
/// Records the upscale into cmd: 0 when done, else FFX's return code (FfxApiReturnCodes)
uint32_t fsr_dispatch(FsrContext* c, VkCommandBuffer cmd, const FsrFrame& f);

} // namespace mkwvk
