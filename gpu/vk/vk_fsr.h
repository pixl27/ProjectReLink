// (relink) AMD FSR's upscaler (vk_fsr.cpp), for vk_dlss.cpp: AMD's own runtime, loaded at run time through AMD's FidelityFX
// API (its MIT headers in third_party/ffx_api). On Windows the FSR SDK 2.x loader, amd_fidelityfx_loader_vk.dll, with its
// upscaler beside it (amd_fidelityfx_upscaler_vk.dll: FSR 4 where AMD's runtime offers it on this GPU, else FSR 3.1); an older
// amd_fidelityfx_vk.dll (FSR 3.1 only) when the loader is not there. On Linux libamd_fidelityfx_vk.so (FSR 3.1). In every
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
    FSR_NO_RUNTIME,       // (no amd_fidelityfx_loader_vk.dll nor amd_fidelityfx_vk.dll in the game folder nor beside the exe)
    FSR_RUNTIME_FAILED,   // (it is there but did not load, or is not AMD's FidelityFX API)
    FSR_NO_UPSCALER,      // (it has no upscaler for Vulkan)
    FSR_DEVICE,           // (it could not make an upscaler on this device)
    FSR_WHY_COUNT
};
FsrWhy fsr_why();
/// The device's extensions AMD's runtime takes by name (vkGetDeviceProcAddr gives nothing for one not enabled), and those
/// FSR 4's programs use where the GPU has them (VK_EXT_shader_float8 on RDNA 4, VK_KHR_compute_shader_derivatives), added to
/// `want` when the device has them (all: what it has)
void fsr_device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& all);
/// The feature structures of those extensions, chained at *tail (and tail moved past them) before the device's features
/// are read: every feature the GPU has of them is then turned on with the device (vk_main.cpp)
void fsr_device_features(void**& tail);
/// Once the device is made (vk_main.cpp): loads AMD's runtime and makes a small upscaler on this device and lets it go, so
/// the Upscaler row knows before the first frame. Logs one line: available, or why not
void fsr_start();
/// The runtime's upscaler, as it names its version ("" before fsr_start or without one): the one FSR runs when nothing
/// else is asked (FSR 4 where AMD's runtime offers it on this GPU)
const char* fsr_version();
/// AMD's runtime makes FSR 4 on this GPU (its test upscaler was one): the FSR version row can choose between it and 3.1
bool fsr4_available();
/// An FSR 3.1 the runtime can make beside its default (the FSR version row's other choice)
bool fsr31_available();

/// An upscaler for these sizes (FFX's context). Its creation flags: FSR_DEPTH_INFINITE (the projection has no far plane),
/// FSR_DYNAMIC (the render size changes from frame to frame: the Render resolution row's Auto), FSR_V31 (FSR 3.1 over the
/// runtime's default: the FSR version row), FSR_PERCEPTUAL (the colour given is gamma-encoded: FSR 4 is told at its
/// creation, FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE; 3.1 only by the dispatch's flag, as before)
struct FsrContext;
enum { FSR_DEPTH_INFINITE = 1, FSR_DYNAMIC = 2, FSR_V31 = 4, FSR_PERCEPTUAL = 8 };
/// nullptr when it was not made (why in the log). Its creation records nothing: no command buffer is needed. An FSR 4 that
/// cannot be made at these sizes falls back to FSR 3.1 (said in the log)
FsrContext* fsr_create(uint32_t max_render_w, uint32_t max_render_h, uint32_t out_w, uint32_t out_h, unsigned flags);
/// The version a context runs, as the runtime names it ("" when it does not say)
const char* fsr_context_version(FsrContext* c);
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
    bool perceptual = false;    // (the colour is gamma-encoded, not linear: FFX's sRGB input flag)
};
/// Records the upscale into cmd: 0 when done, else FFX's return code (FfxApiReturnCodes)
uint32_t fsr_dispatch(FsrContext* c, VkCommandBuffer cmd, const FsrFrame& f);

} // namespace mkwvk
