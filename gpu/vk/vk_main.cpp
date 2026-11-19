// The Vulkan back end's start: vulkan-1.dll, the instance, the device (Vulkan 1.3 and the extensions the NVN replay
// wants where the GPU has them: mesh shaders, host memory import, graphics pipeline libraries), the window's
// swapchain, and the rx_* table (vk_api.h). The NVN replay itself is vk_replay.cpp.
#include "vk_paths.h"
#include "vk_loader.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "vk_api.h"
#include "vk_draw.h"
#include "vk_framegen.h"
#include "vk_internal.h"
#include "gpu_choice.h"
#include "radv_submit_thread.h"
#include "vk_menu.h"
#include "vk_dlss.h"
#include "vk_fsr.h"
#include "vk_reflex.h"
#include "vk_progress.h"
#ifdef _WIN32
#include <dxgi1_6.h>   // (relink: HDR output, hdr_display)
#endif

namespace vkx {
#define VKX_DEFINE(f) PFN_##f f;
VKX_GLOBAL(VKX_DEFINE)
VKX_INSTANCE(VKX_DEFINE)
VKX_DEVICE(VKX_DEFINE)
VKX_DEVICE2(VKX_DEFINE)
VKX_DEVICE3(VKX_DEFINE)
#undef VKX_DEFINE

static PFN_vkGetInstanceProcAddr get_instance_proc;

bool load_global() {
    HMODULE m = LoadLibraryA("vulkan-1.dll");   // (linux: libvulkan.so.1, host/posix/win32_posix.c)
    if (!m) return false;
    get_instance_proc = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(m, "vkGetInstanceProcAddr");
    if (!get_instance_proc) return false;
#define VKX_LOAD(f) f = (PFN_##f)get_instance_proc(nullptr, #f);
    VKX_GLOBAL(VKX_LOAD)
#undef VKX_LOAD
    return vkCreateInstance != nullptr;
}
PFN_vkGetInstanceProcAddr instance_proc_addr() { return get_instance_proc; }
void load_instance(VkInstance instance) {
#define VKX_LOAD(f) f = (PFN_##f)get_instance_proc(instance, #f);
    VKX_INSTANCE(VKX_LOAD)
#undef VKX_LOAD
}
void load_device(VkDevice device) {
#define VKX_LOAD(f) f = (PFN_##f)vkGetDeviceProcAddr(device, #f);
    VKX_DEVICE(VKX_LOAD)
    VKX_DEVICE2(VKX_LOAD)
    VKX_DEVICE3(VKX_LOAD)
#undef VKX_LOAD
}
} // namespace vkx

namespace mkwvk {

std::vector<std::string> gpu_device_names;
const MkwVkHost* host;
int64_t prof_ns[PROF_COUNT];
Device dev;
PFN_vkGetInstanceProcAddr instance_proc() { return vkx::instance_proc_addr(); }

void logf(const char* fmt, ...) {
    char b[1024];
    std::va_list a;
    va_start(a, fmt);
    std::vsnprintf(b, sizeof b, fmt, a);
    va_end(a);
    if (host) host->log(b);
}

// (relink) HDR output: VK_EXT_swapchain_colorspace enabled on the instance (the player asked for HDR and the loader has it), and
// whether the swapchain now is HDR10 (A2B10G10R10, ST 2084) with the display's peak in nits (hdr_output, hdr_peak_nits)
bool hdr_ext = false;
std::atomic<int> hdr_state{0}, hdr_peak{0};
bool hdr_display(float* peak);

namespace {

// (relink, issue 43) A driver that faults in vkCreateShaderModule. On an RX 570 (Polaris, AMD's driver 2.0.220 and 2.0.279) the
// game closed with no message during its first shader preparation: six "CRASH ... at amdvlk64.dll" lines at once, one per
// preparation worker, after the step that builds pipelines had been left out ("the last start did not finish the driver's part"),
// so the only driver call left there is vkCreateShaderModule (vk_shader.cpp make_module), which no guard covered (pipelines have
// guarded_graphics / guarded_compute). Its pointer goes through this guard now: a fault is that shader failing (the translator
// takes a null module for a failed program), logged with the SPIR-V kept beside the game for us (the first two), and said once in
// a box, with what it likely means on a GPU below the minimum. RX_VK_NO_MODULE_GUARD=1: the driver's pointer as before
bool amd_gcn = false;   // (an AMD GPU with only 64-lane subgroups and no mesh shaders: GCN, RX 400 / 500 / Vega; create_device)
#ifdef _WIN32
PFN_vkCreateShaderModule driver_create_shader_module = nullptr;
std::atomic<int> module_faults{0};
__declspec(noinline) VkResult module_call(VkDevice d, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks* a,
                                          VkShaderModule* m, int* faulted) {
    __try {
        return driver_create_shader_module(d, ci, a, m);
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION || GetExceptionCode() == EXCEPTION_ILLEGAL_INSTRUCTION ||
                        GetExceptionCode() == EXCEPTION_INT_DIVIDE_BY_ZERO ? 1 : 0) {
        *faulted = 1;
        return VK_ERROR_UNKNOWN;
    }
}
void module_fault(const VkShaderModuleCreateInfo* ci) {
    const int n = ++module_faults;
    char saved[MAX_PATH + 48] = "";
    if (n <= 2 && ci && ci->pCode && ci->codeSize) {   // (the SPIR-V the driver could not take: spirv-val / spirv-dis it here)
        char path[MAX_PATH];
        std::string dir(path, game_module_path(path, MAX_PATH));
        dir = dir.substr(0, dir.find_last_of("\\/") + 1);
        snprintf(saved, sizeof saved, "%srelink_driver_fault_%d.spv", dir.c_str(), n);
        if (FILE* f = fopen(saved, "wb")) fwrite(ci->pCode, 1, ci->codeSize, f), fclose(f);
        else saved[0] = 0;
    }
    if (n <= 8 || (n & (n - 1)) == 0)
        logf("vk: THE GRAPHICS DRIVER CRASHED reading a shader (vkCreateShaderModule, %zu bytes of SPIR-V%s%s): it is left out "
             "(%d so far)", ci ? (size_t)ci->codeSize : 0, saved[0] ? ", kept as " : "", saved, n);
    if (n != 1 || dev.offscreen || getenv("HOS_HEADLESS")) return;
    std::thread([name = std::string(dev.props.deviceName), old = !dev.mesh_shader] {
        std::string text = "The graphics driver of your " + name + " crashed while ReLink handed it one of the game's shaders. "
            "ReLink leaves such shaders out and goes on, but parts of the picture may be missing, and the driver may fail again.";
        if (old)
            text += "\n\nThis card has no mesh shaders and is below what ReLink supports (NVIDIA GeForce RTX 20 series or newer, "
                    "AMD Radeon RX 6000 series or newer, or Intel Arc). Updating its driver is worth a try, but it may not run "
                    "the game.";
        text += "\n\nIf you report it on ReLink's GitHub page, please attach relink_log.txt and the relink_driver_fault_1.spv file "
                "from the game folder.";
        MessageBoxA(nullptr, text.c_str(), "Project ReLink: graphics driver", MB_OK | MB_ICONWARNING | MB_TOPMOST);
    }).detach();
}
VKAPI_ATTR VkResult VKAPI_CALL guarded_create_shader_module(VkDevice d, const VkShaderModuleCreateInfo* ci,
                                                            const VkAllocationCallbacks* a, VkShaderModule* m) {
    int faulted = 0;
    const VkResult r = module_call(d, ci, a, m, &faulted);
    if (faulted) {
        if (m) *m = VK_NULL_HANDLE;
        module_fault(ci);
    }
    return r;
}
void guard_shader_modules() {
    if (getenv("RX_VK_NO_MODULE_GUARD") || !vkx::vkCreateShaderModule) return;
    driver_create_shader_module = vkx::vkCreateShaderModule;
    vkx::vkCreateShaderModule = guarded_create_shader_module;
}
#else
void guard_shader_modules() {}
#endif

bool has_extension(const std::vector<VkExtensionProperties>& all, const char* name) {
    for (const auto& e : all)
        if (!std::strcmp(e.extensionName, name)) return true;
    return false;
}

bool create_instance() {
    reflex_enable_implicit_layer();
    uint32_t version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&version);
    if (version < VK_API_VERSION_1_3) {
        logf("vk: the Vulkan loader is %u.%u: 1.3 is needed", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version));
        return false;
    }
#ifdef _WIN32
    const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    const uint32_t n_exts = 2;
#else
    // (linux: every window system's surface the loader has, for whichever one SDL's window is on: X11 or Wayland)
    const char* exts[4] = {VK_KHR_SURFACE_EXTENSION_NAME};
    uint32_t n_exts = 1;
    {
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> all(n);
        vkEnumerateInstanceExtensionProperties(nullptr, &n, all.data());
        for (const char* want : {"VK_KHR_xlib_surface", "VK_KHR_xcb_surface", "VK_KHR_wayland_surface"})
            if (has_extension(all, want)) exts[n_exts++] = want;
    }
#endif
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "mkw-recomp";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    std::vector<const char*> ext_list(exts, exts + n_exts);
    {   // (relink) what NVIDIA DLSS needs (vk_dlss.cpp; nothing in a build without it)
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> all(n);
        vkEnumerateInstanceExtensionProperties(nullptr, &n, all.data());
        dlss_instance_extensions(ext_list, all);
        // (relink) HDR: the colour spaces of the ST 2084 swapchain, only when the HDR row is on (nothing changes otherwise)
        if (hdr_wanted()) {
            hdr_ext = has_extension(all, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
            if (hdr_ext) ext_list.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
            else logf("vk: HDR asked for, but the Vulkan loader has no VK_EXT_swapchain_colorspace: the picture stays SDR");
        }
    }
    ci.enabledExtensionCount = (uint32_t)ext_list.size();
    ci.ppEnabledExtensionNames = ext_list.data();
    // RX_VK_VALIDATION=1: the Khronos validation layer (when installed)
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (getenv("RX_VK_VALIDATION")) ci.enabledLayerCount = 1, ci.ppEnabledLayerNames = &layer;
    if (vkCreateInstance(&ci, nullptr, &dev.instance) != VK_SUCCESS) {
        logf("vk: vkCreateInstance failed");
        return false;
    }
    load_instance(dev.instance);
    return true;
}

bool pick_device() {
    menu_init();
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(dev.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> all(n);
    vkEnumeratePhysicalDevices(dev.instance, &n, all.data());

    std::vector<uint32_t> valid_gpus;
    gpu_device_names.clear();
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(all[i], &p);
        logf("vk: device %u: %s, Vulkan %u.%u, driver %#x, type %d", i, p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
             VK_API_VERSION_MINOR(p.apiVersion), p.driverVersion, (int)p.deviceType);
        if (p.apiVersion >= VK_API_VERSION_1_3) {
            valid_gpus.push_back(i);
            gpu_device_names.push_back(p.deviceName);
        }
    }
    menu_set_gpu_count((int)valid_gpus.size());

    // (AMD laptops) With AMD graphics in the processor and an AMD card, Windows has two AMD drivers installed, one an
    // adapter, and each one's Vulkan driver lists every AMD GPU: the card shows up twice. It came first through the
    // processor graphics' driver, which the card's own update leaves as it was: on an RX 7700S that older driver
    // (amdvlk64.dll 32.0.11036.4002) crashed in its shader compiler (0xc0000409) on the game's compute pipelines while
    // the card's 32.0.32015.7029 built them. The same kind of GPU twice: the one with the newer driver wins.
    // RX_VK_DEVICE=<n> or opts.gpu_device: that device, as listed in the log
    int best = -1, best_score = -1;
    uint32_t best_driver = 0;
    int target_raw_idx = -1;
    if (const char* want = getenv("RX_VK_DEVICE")) {
        if (*want) target_raw_idx = atoi(want);
    } else if (const int k = gpu_choice::menu_device(opts.gpu_device, (int)valid_gpus.size()); k >= 0) {
        target_raw_idx = (int)valid_gpus[k];
    }

    for (uint32_t i : valid_gpus) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(all[i], &p);
        if (target_raw_idx >= 0) {
            if ((int)i == target_raw_idx) best = (int)i, best_score = 3;
            continue;
        }
        const int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 :
                          p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
        if (score > best_score || (score == best_score && p.driverVersion > best_driver))
            best = (int)i, best_score = score, best_driver = p.driverVersion;
    }
    if (best < 0 && target_raw_idx >= 0) {
        logf("vk: preferred device %d not available or lacks Vulkan 1.3, falling back to auto selection", target_raw_idx);
        for (uint32_t i : valid_gpus) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(all[i], &p);
            const int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 :
                              p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
            if (score > best_score || (score == best_score && p.driverVersion > best_driver))
                best = (int)i, best_score = score, best_driver = p.driverVersion;
        }
    }
    if (best < 0) {
        logf("vk: no Vulkan 1.3 GPU");
        return false;
    }
    dev.physical = all[best];
    vkGetPhysicalDeviceProperties(dev.physical, &dev.props);
    vkGetPhysicalDeviceMemoryProperties(dev.physical, &dev.memory);
    return true;
}

// (relink, issue 640) What the translated shaders take for granted (vk_shader.cpp make_profile and host_info: no device asked) and
// this GPU lacks, said once at start. A shader that uses one of these anyway is invalid for the driver, which may refuse its pipeline
// or fault in its compiler. The GTX 10 cards' logs ended on a fallback compute pipeline that "could not be made", on drivers whose
// features nobody here could read: this line names those missing, in the log a player already sends, to compare with ours
void log_shader_needs(const VkPhysicalDeviceFeatures& f, const VkPhysicalDeviceVulkan11Features& f11,
                      const VkPhysicalDeviceVulkan12Features& f12, const VkPhysicalDeviceVulkan13Features& f13) {
    VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    p11.pNext = &p12;
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &p11;
    vkGetPhysicalDeviceProperties2(dev.physical, &p2);
    const VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    const VkSubgroupFeatureFlags ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                       VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
                                       VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT;
    const struct { bool have; const char* name; } needs[] = {
        {f12.shaderFloat16 != VK_FALSE, "shaderFloat16"},
        {f12.shaderInt8 != VK_FALSE, "shaderInt8"},
        {f.shaderInt16 != VK_FALSE, "shaderInt16"},
        {f.shaderInt64 != VK_FALSE, "shaderInt64"},
        {f.shaderFloat64 != VK_FALSE, "shaderFloat64"},
        {f12.storageBuffer8BitAccess && f12.uniformAndStorageBuffer8BitAccess, "8-bit storage"},
        {f11.storageBuffer16BitAccess && f11.uniformAndStorageBuffer16BitAccess, "16-bit storage"},
        {f12.shaderBufferInt64Atomics != VK_FALSE, "shaderBufferInt64Atomics"},
        {f12.shaderSharedInt64Atomics != VK_FALSE, "shaderSharedInt64Atomics"},
        {f13.shaderDemoteToHelperInvocation != VK_FALSE, "shaderDemoteToHelperInvocation"},
        {f.shaderStorageImageReadWithoutFormat && f.shaderStorageImageWriteWithoutFormat, "storage images without format"},
        {f.shaderImageGatherExtended != VK_FALSE, "shaderImageGatherExtended"},
        {f.multiViewport != VK_FALSE, "multiViewport"},
        {f12.shaderOutputViewportIndex && f12.shaderOutputLayer, "shaderOutputViewportIndex/Layer"},
        {f12.shaderSampledImageArrayNonUniformIndexing != VK_FALSE, "shaderSampledImageArrayNonUniformIndexing"},
        {f.fragmentStoresAndAtomics && f.vertexPipelineStoresAndAtomics, "stores and atomics in vertex/fragment stages"},
        {p12.shaderDenormPreserveFloat32 != VK_FALSE, "shaderDenormPreserveFloat32"},
        {p12.shaderDenormFlushToZeroFloat32 != VK_FALSE, "shaderDenormFlushToZeroFloat32"},
        {p12.shaderSignedZeroInfNanPreserveFloat16 && p12.shaderSignedZeroInfNanPreserveFloat32, "shaderSignedZeroInfNanPreserve 16/32"},
        {p12.denormBehaviorIndependence != VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE, "denormBehaviorIndependence"},
        {p12.roundingModeIndependence != VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE, "roundingModeIndependence"},
        {(p11.subgroupSupportedStages & stages) == stages, "subgroups in vertex/fragment/compute"},
        {(p11.subgroupSupportedOperations & ops) == ops, "subgroup vote/ballot/shuffle/arithmetic/quad"},
    };
    std::string missing;
    for (const auto& n : needs)
        if (!n.have) missing += missing.empty() ? n.name : std::string(", ") + n.name;
    // (an RTX 3060 Ti, driver 0x9a170000, reports neither of the two fp32 denormal modes and runs every program: a name here is
    // where to look, not a fault by itself)
    logf("vk: shader features the translation assumes and this GPU does not report: %s (subgroup ops %#x on stages %#x; shared "
         "memory %u bytes)", missing.empty() ? "none" : missing.c_str(), (unsigned)p11.subgroupSupportedOperations,
         (unsigned)p11.subgroupSupportedStages, dev.props.limits.maxComputeSharedMemorySize);
}

bool create_device() {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(dev.physical, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(dev.physical, nullptr, &n, exts.data());
    std::vector<const char*> want = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    auto optional = [&](const char* name, bool& flag) {
        flag = has_extension(exts, name);
        if (flag) want.push_back(name);
    };
    // (RX_VK_FORCE_MESH_FALLBACK=1, a test: mesh shaders not used where they are, so the no-mesh fallback the GTX 10xx and RX 5000
    // cards run (issue 640) can be tried on any GPU)
    if (getenv("RX_VK_FORCE_MESH_FALLBACK") && atoi(getenv("RX_VK_FORCE_MESH_FALLBACK")) != 0) dev.mesh_shader = false;
    else optional(VK_EXT_MESH_SHADER_EXTENSION_NAME, dev.mesh_shader);
    optional(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME, dev.external_memory_host);
    optional(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, dev.external_memory_fd);   // (guest memory as udmabufs: vk_memory.cpp import_dma_buf)
    if (dev.external_memory_fd) optional(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME, dev.dma_buf_import);
    optional(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, dev.pipeline_library);
    if (dev.pipeline_library) optional(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME, dev.graphics_pipeline_library);
    optional(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, dev.custom_border_color);
    optional(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, dev.dynamic_state3);
    optional(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME, dev.depth_clip_control);
    optional(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, dev.null_descriptor);
    optional(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME, dev.index_uint8);
    optional(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, dev.push_descriptor);
    optional(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME, dev.conditional_rendering);
    optional(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME, dev.provoking_vertex);
    optional(VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME, dev.vertex_divisor);
    optional(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME, dev.stencil_export);
    optional(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, dev.memory_budget);
    optional(VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME, dev.view_2d_of_3d);
    // (relink) on by default: with a marker before each draw and dispatch the device has not been lost (five
    // runs); without them it was, every run, shortly after OoT's first gameplay frames (nvlddmkm event 153; the
    // extension alone, RX_VK_CHECKPOINTS=ext, did not keep it). The race they hide is not found yet.
    // RX_VK_CHECKPOINTS=0: off. RX_VK_CHECKPOINTS=marker: VK_AMD_buffer_marker's markers instead, where the driver has them
    // (NVIDIA's does): written into system memory as the GPU passes them, they are still there after a reset, where an RTX
    // 5070 Ti that hung gave back no NV checkpoint at all (issue 7: "0 checkpoints (17925724 markers set)")
    const bool want_markers = getenv("RX_VK_CHECKPOINTS") && !strcmp(getenv("RX_VK_CHECKPOINTS"), "marker");
    if (!want_markers && (!getenv("RX_VK_CHECKPOINTS") || strcmp(getenv("RX_VK_CHECKPOINTS"), "0") != 0))
        optional(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, dev.checkpoints);
    // (asked for only: a marker written twice a draw, where the NV ones are not there to hide a race)
    if (const char* c = getenv("RX_VK_CHECKPOINTS"); !dev.checkpoints && c && strcmp(c, "0") != 0 && strcmp(c, "ext") != 0)
        optional(VK_AMD_BUFFER_MARKER_EXTENSION_NAME, dev.buffer_marker);
    // (relink, issues 62 and 66) RX_VK_CHECKPOINTS=marker on a driver without VK_AMD_buffer_marker left no marker of any kind,
    // where the default would have had the NV checkpoints (an Intel Arc B580 has them, issue 62's log): those instead
    if (want_markers && !dev.buffer_marker) {
        optional(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, dev.checkpoints);
        logf("vk: RX_VK_CHECKPOINTS=marker: this driver has no VK_AMD_buffer_marker; %s", dev.checkpoints
             ? "the NV checkpoints instead" : "no NV checkpoints either: no markers");
    }
    optional(VK_EXT_DEVICE_FAULT_EXTENSION_NAME, dev.device_fault);
    // (relink, issues 45, 62 and 66) what this driver offers to name the work of a lost device, said at every start, so a player's
    // first log tells which of RX_VK_CHECKPOINTS=marker and RX_VK_FAULT_INFO=1 can work there before they are asked for
    logf("vk: device lost diagnostics on this driver: buffer markers %s, NV checkpoints %s, fault info %s",
         has_extension(exts, VK_AMD_BUFFER_MARKER_EXTENSION_NAME) ? "yes (RX_VK_CHECKPOINTS=marker)" : "no",
         has_extension(exts, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME) ? (dev.checkpoints ? "yes, on" : "yes, off") : "no",
         dev.device_fault ? "yes (RX_VK_FAULT_INFO=1)" : "no");
    optional(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, dev.coop_matrix);
    dlss_device_extensions(want, exts);   // (relink: NVIDIA DLSS's, vk_dlss.cpp)
    fsr_device_extensions(want, exts);    // (relink: what AMD FSR's runtime takes by name, vk_fsr.cpp)
    reflex_device_extensions(want, exts);   // (relink: NVIDIA Reflex's, vk_reflex.cpp)
    optional(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, dev.float_atomic_add);   // (relink: native float adds, emit_spirv_atomic.cpp)
    // (relink) EDEN_NX2_MAXRECONV=1: maximal reconvergence for programs with subgroup operations (emit_spirv.cpp). It needs
    // VK_KHR_shader_subgroup_uniform_control_flow. Off by default: a test
    bool uniform_cf = false;
    if (const char* m = getenv("EDEN_NX2_MAXRECONV"); m && strcmp(m, "0") != 0) {
        optional(VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME, uniform_cf);
        if (uniform_cf) optional(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME, dev.max_reconvergence);
    }

    // the queue: one family for graphics, compute and transfer
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &n, families.data());
    dev.queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < n && dev.queue_family == UINT32_MAX; i++)
        if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            dev.queue_family = i;
    if (dev.queue_family == UINT32_MAX) return false;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = dev.queue_family;
    q.queueCount = 1;
    q.pQueuePriorities = &priority;
    // (relink) The presents on a queue of a compute family of their own, beside the replay's (RX_VK_PRESENT_QUEUE=1 / 0: on /
    // off; on by default with NVIDIA's driver on Windows, where it was measured). On the replay's queue vkQueuePresentKHR blocked
    // with dev.present_mu held, ~0.8 ms a present with frame interpolation at 4x and up to 8, and the game's main and presentation
    // threads waited there to submit. A second queue of the graphics family took that lock off the presents, but its presents
    // blocked longer (~2-4 ms: the graphics work of the two queues takes turns on the GPU) and the made frames were shown late
    // four times as often. A compute family's queue runs beside the graphics work: in Hyrule Field (a 14700K and an RTX 5070 Ti,
    // 4x, 175 Hz) the game went from ~52 to ~58 fps standing and ~46.5 to ~53 running, its frames over 22 ms from ~13% to ~1%
    // standing and over 40 ms from ~4% to ~0.2% running, the made frames as evenly shown standing. The swapchain's images are
    // shared by both families (vk_main.cpp create_swapchain), and a family that cannot present to the window leaves it off.
    uint32_t compute_family = UINT32_MAX;
    {
        bool want = false;
#ifdef _WIN32
        VkPhysicalDeviceDriverProperties dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &dp;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        want = false;   // (v2.0.9: off by default with the write-back diff until issue 1161's device lost is placed; was:
        (void)dp;       //  NVIDIA's driver on Windows. RX_VK_PRESENT_QUEUE=1: on)
#endif
        if (const char* e = getenv("RX_VK_PRESENT_QUEUE")) want = atoi(e) != 0;
        // (relink: ReShade, and RenoDX through it, draws its effects and its HDR pass on the queue the presents are made on: a
        // compute family's has no graphics, and ReShade turns itself off there. reshade_compat)
        if (want && reshade_compat()) {
            want = false;
            logf("vk: ReShade / RenoDX: the presents stay on the graphics queue (RX_VK_PRESENT_QUEUE left off)");
        }
        for (uint32_t i = 0; want && i < n && compute_family == UINT32_MAX; i++)
            if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                compute_family = i;
    }
    VkDeviceQueueCreateInfo qs[2] = {q, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    qs[1].queueFamilyIndex = compute_family;
    qs[1].queueCount = 1;
    qs[1].pQueuePriorities = &priority;

    // features: what the hardware has of what the replay uses (all of them on the GPUs MKW-class hardware maps to)
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceRobustness2FeaturesEXT robust2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT u8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f.pNext = &f11, f11.pNext = &f12, f12.pNext = &f13;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT border{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceDepthClipControlFeaturesEXT clip{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cond{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    VkPhysicalDeviceProvokingVertexFeaturesEXT pv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT};
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    void** tail = &f13.pNext;
    if (dev.mesh_shader) *tail = &mesh, tail = &mesh.pNext;
    if (dev.null_descriptor) *tail = &robust2, tail = &robust2.pNext;
    if (dev.index_uint8) *tail = &u8, tail = &u8.pNext;
    if (dev.dynamic_state3) *tail = &eds3, tail = &eds3.pNext;
    if (dev.custom_border_color) *tail = &border, tail = &border.pNext;
    if (dev.depth_clip_control) *tail = &clip, tail = &clip.pNext;
    if (dev.conditional_rendering) *tail = &cond, tail = &cond.pNext;
    if (dev.provoking_vertex) *tail = &pv, tail = &pv.pNext;
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT vdiv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT};
    if (dev.vertex_divisor) *tail = &vdiv, tail = &vdiv.pNext;
    if (dev.graphics_pipeline_library) *tail = &gpl, tail = &gpl.pNext;
    VkPhysicalDeviceFaultFeaturesEXT fault{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    if (dev.device_fault) *tail = &fault, tail = &fault.pNext;
    VkPhysicalDeviceImage2DViewOf3DFeaturesEXT v2d3d{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT};
    if (dev.view_2d_of_3d) *tail = &v2d3d, tail = &v2d3d.pNext;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    if (dev.coop_matrix) *tail = &coop, tail = &coop.pNext;
    reflex_device_features(tail);   // (present ids, when NVIDIA Reflex's extensions are there)
    fsr_device_features(tail);      // (relink: FSR 4's FP8 and compute-derivative features, where the GPU has them: vk_fsr.cpp)
    VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR ucf{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR};
    VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR maxr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR};
    if (dev.max_reconvergence) *tail = &ucf, ucf.pNext = &maxr, tail = &maxr.pNext;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT afloat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    if (dev.float_atomic_add) *tail = &afloat, tail = &afloat.pNext;
    vkGetPhysicalDeviceFeatures2(dev.physical, &f);
    log_shader_needs(f.features, f11, f12, f13);
    dev.float_atomic_add = dev.float_atomic_add && afloat.shaderBufferFloat32AtomicAdd;
    if (dev.float_atomic_add) logf("vk: native float32 atomic add on storage buffers (VK_EXT_shader_atomic_float)");
    dev.max_reconvergence = dev.max_reconvergence && ucf.shaderSubgroupUniformControlFlow && maxr.shaderMaximalReconvergence;
    if (uniform_cf || dev.max_reconvergence) logf("vk: maximal reconvergence %s (EDEN_NX2_MAXRECONV)", dev.max_reconvergence ? "on" : "not supported");
    dev.view_2d_of_3d = dev.view_2d_of_3d && v2d3d.sampler2DViewOf3D;
    coop.cooperativeMatrixRobustBufferAccess = VK_FALSE;
    dev.coop_matrix = dev.coop_matrix && coop.cooperativeMatrix &&
                      f12.vulkanMemoryModel && f12.vulkanMemoryModelDeviceScope;
    // (bounds-checked buffer access: an out-of-range access dropped instead of faulting the device. On by default: the GPU
    // culling pass 53caac5f faulted the device while a save loaded, about one start in three, and the scene's lighting read garbage
    // for single frames (the black flashes); with it, neither in A/B runs at the same GPU time. Lichii's OoT branch keeps it on too.
    // The 'Robust buffers' row of relink_options.ini chooses (0 Auto: on, on every vendor; 1 Off; 2 On); RX_VK_ROBUST=0 or 1
    // over it. Off is null descriptors only: some NVIDIA cards ran slower with it on, issues 394, 418, 419)
    static const bool robust = []() -> bool {
        if (const char* e = getenv("RX_VK_ROBUST")) return atoi(e) != 0;
        char exe[MAX_PATH];
        const DWORD n = game_module_path(exe, MAX_PATH);   // (the game folder, not bin beside the launcher: vk_paths.h)
        std::string ini(exe, n);
        ini = ini.substr(0, ini.find_last_of("\\/") + 1) + "relink_options.ini";
        int v = 0;   // (Auto)
        if (FILE* f = fopen(ini.c_str(), "r")) {
            char line[80];
            while (fgets(line, sizeof line, f))
                if (!strncmp(line, "robust_buffers=", 15)) {
                    v = atoi(line + 15);
                    break;
                }
            fclose(f);   // (once: the PR closed it twice when the row said Auto)
        }
        return v != 1;   // (Auto and On: on)
    }();
    const bool robust_ok = robust && robust2.robustBufferAccess2 && f.features.robustBufferAccess;
    robust2.robustBufferAccess2 = robust_ok ? VK_TRUE : VK_FALSE;
    robust2.robustImageAccess2 = VK_FALSE;
    dev.null_descriptor = dev.null_descriptor && robust2.nullDescriptor;
    pv.transformFeedbackPreservesProvokingVertex = VK_FALSE;
    dev.provoking_vertex = dev.provoking_vertex && pv.provokingVertexLast;
    dev.conditional_rendering = dev.conditional_rendering && cond.conditionalRendering;
    // the occlusion queries' real counts (vk_replay.cpp, RX_VK_OCC_QUERIES): a sample count, not only "any", and
    // their reset from the CPU once read (both on when the device has them: the queried chain is what it is created with)
    dev.occlusion_precise = f.features.occlusionQueryPrecise == VK_TRUE;
    dev.host_query_reset = f12.hostQueryReset == VK_TRUE;
    dev.custom_border_color = dev.custom_border_color && border.customBorderColors && border.customBorderColorWithoutFormat;
    dev.depth_clip_control = dev.depth_clip_control && clip.depthClipControl;
    dev.graphics_pipeline_library = dev.graphics_pipeline_library && gpl.graphicsPipelineLibrary;
    if (dev.graphics_pipeline_library) {
        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT lp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &lp;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        dev.graphics_pipeline_library = lp.graphicsPipelineLibraryFastLinking != VK_FALSE;
    }
    dev.eds3_polygon = dev.dynamic_state3 && eds3.extendedDynamicState3PolygonMode;
    dev.eds3_clamp = dev.dynamic_state3 && eds3.extendedDynamicState3DepthClampEnable;
    dev.eds3_blend = dev.dynamic_state3 && eds3.extendedDynamicState3ColorBlendEnable &&
                     eds3.extendedDynamicState3ColorBlendEquation && eds3.extendedDynamicState3ColorWriteMask;
    dev.eds3_a2c = dev.dynamic_state3 && eds3.extendedDynamicState3AlphaToCoverageEnable;
    dev.vk12 = f12;
    dev.vk13 = f13;
    {   // (relink, after TotK Recomp's vk_caps.c) NVN's programs are written for 32 lanes: the translator is told 32, and the lane
        // shuffles of OoT's networks and the lane masks of its mesh programs depend on it. NVIDIA's GPUs have no other size; AMD's
        // default to 64 where they like, and the networks' output came out as coloured noise
        VkPhysicalDeviceSubgroupSizeControlProperties sc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        sg.pNext = &sc;
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        const bool pin = f13.subgroupSizeControl && sc.minSubgroupSize <= 32 && sc.maxSubgroupSize >= 32 && !getenv("RX_VK_NO_SUBGROUP_PIN");
        dev.sg32_stages = pin ? sc.requiredSubgroupSizeStages : 0;
        amd_gcn = dev.props.vendorID == 0x1002 && sc.minSubgroupSize == 64 && sc.maxSubgroupSize == 64;   // (issue 43)
        // (relink, Discord: Artspear, the dogs drawn without textures on an RX 6600) AMD's RDNA cards hold only compute work to 32
        // lanes (requiredSubgroupSizeStages 0x20): their drawing stages run 64 wide where the driver likes, and a program's lane
        // votes, ballots and shuffles, written for 32, mixed two groups. Such GPUs translate with the wide-warp form
        const VkShaderStageFlags drawing = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        dev.big_warp = std::max(sg.subgroupSize, sc.maxSubgroupSize) > 32 && (dev.sg32_stages & drawing) != drawing &&
                       !getenv("RX_VK_NO_BIG_WARP");
        logf("vk: subgroups: default %u lanes, %u-%u, pinned to 32 on stages %#x%s", sg.subgroupSize, sc.minSubgroupSize, sc.maxSubgroupSize,
             (unsigned)dev.sg32_stages, dev.big_warp ? "; the drawing stages may run wider: translated for it" : "");
    }
    if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore) {
        logf("vk: the GPU lacks dynamic rendering, synchronization2 or timeline semaphores");
        return false;
    }
    mesh.multiviewMeshShader = mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;   // (not used)
    dev.mesh_shader = dev.mesh_shader && mesh.meshShader && mesh.taskShader;
    // (relink, issue 640) RDNA1 and older GPUs: no mesh shaders, but task and mesh programs run as compute shaders
    // that write their vertices to buffers, drawn through an ordinary graphics pipeline. It needs buffer device addresses.
    dev.mesh_fallback = !dev.mesh_shader && f12.bufferDeviceAddress && !getenv("RX_VK_NO_MESH_FALLBACK");
    if (dev.mesh_fallback) logf("vk: no mesh shaders: task and mesh programs run as compute shaders (issue 640 fallback)");
    if (amd_gcn && !dev.mesh_shader)   // (relink, issue 43)
        logf("vk: an AMD GCN GPU (Radeon RX 400 / 500 series or Vega: 64-lane subgroups only, no mesh shaders): below what ReLink "
             "supports (an RX 570's driver closed the game during its shader preparation, issue 43)");
    if (robust) logf("vk: robust buffer access %s", robust_ok ? "on" : "not supported");
    f.features.robustBufferAccess = robust_ok ? VK_TRUE : VK_FALSE;   // (off by default: the game's accesses stay in their buffers)
    dev.features = f.features;
    dev.sparse_images = (families[dev.queue_family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) && f.features.sparseBinding &&
                        f.features.sparseResidencyImage2D && !(getenv("RX_VK_SPARSE") && !atoi(getenv("RX_VK_SPARSE")));
    dev.sparse_buffers = (families[dev.queue_family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) && f.features.sparseBinding;
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.pNext = &f;
    ci.queueCreateInfoCount = compute_family != UINT32_MAX ? 2 : 1;
    ci.pQueueCreateInfos = qs;
    ci.enabledExtensionCount = (uint32_t)want.size();
    ci.ppEnabledExtensionNames = want.data();
#ifndef _WIN32
    // RADV can move the kernel submit off the replaying thread. Mesa reads this
    // variable inside vkCreateDevice, so set it only for that call and restore
    // the process environment immediately afterwards. A user's Mesa setting
    // always wins; RX_VK_RADV_SUBMIT_THREAD=0 keeps Mesa's default behaviour.
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &driver;
    vkGetPhysicalDeviceProperties2(dev.physical, &properties);
    const auto submit_thread = radv_submit_thread::decide(driver.driverID == VK_DRIVER_ID_MESA_RADV,
                                                          getenv("MESA_VK_ENABLE_SUBMIT_THREAD"),
                                                          getenv("RX_VK_RADV_SUBMIT_THREAD"));
    bool submit_thread_set = false;
    switch (submit_thread) {
    case radv_submit_thread::Choice::set_on:
        submit_thread_set = setenv("MESA_VK_ENABLE_SUBMIT_THREAD", "1", 0) == 0;
        logf("vk: RADV: Mesa submit thread %s (RX_VK_RADV_SUBMIT_THREAD=0: off)",
             submit_thread_set ? "on" : "could not be enabled");
        break;
    case radv_submit_thread::Choice::user_set:
        logf("vk: RADV: Mesa submit thread left as set by MESA_VK_ENABLE_SUBMIT_THREAD=%s",
             getenv("MESA_VK_ENABLE_SUBMIT_THREAD"));
        break;
    case radv_submit_thread::Choice::switch_off:
        logf("vk: RADV: Mesa submit thread left at its default (RX_VK_RADV_SUBMIT_THREAD=0)");
        break;
    case radv_submit_thread::Choice::not_radv:
        break;
    }
#endif
    const VkResult created = vkCreateDevice(dev.physical, &ci, nullptr, &dev.device);
#ifndef _WIN32
    if (submit_thread_set) unsetenv("MESA_VK_ENABLE_SUBMIT_THREAD");
#endif
    if (created != VK_SUCCESS) {
        logf("vk: vkCreateDevice failed");
        return false;
    }


    load_device(dev.device);
    guard_shader_modules();   // (relink, issue 43)
    vkGetDeviceQueue(dev.device, dev.queue_family, 0, &dev.queue);
    dev.present_queue = VK_NULL_HANDLE;
    dev.present_family = dev.queue_family;
    if (compute_family != UINT32_MAX) vkGetDeviceQueue(dev.device, compute_family, 0, &dev.present_queue), dev.present_family = compute_family;
    if (dev.buffer_marker) buffer_markers_init();
    coopmat_init();
    reflex_device_created();
    fsr_start();   // (relink: whether AMD FSR runs here, in the log at once, vk_fsr.cpp)
    if (dev.push_descriptor) {
        VkPhysicalDevicePushDescriptorPropertiesKHR pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &pp;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        dev.max_push_descriptors = pp.maxPushDescriptors;
    }
    return true;
}

} // namespace

// (relink) "30, smooth" on the start-up screen: FIFO waits for until it is over. With FIFO and three images, AMD's present held
// the queue for a refresh each time the screen was drawn (every 33 ms), and the shaders' preparation went from 3 minutes to hours
// (an RX 6600 XT, 11-17: 1600 of 38809 in 5 minutes). The screen is drawn in the usual mode; the game's first frame makes the
// swapchain again in FIFO (prepare_with_screen)
static std::atomic<bool> fifo_held_for_startup{false};

bool create_swapchain(uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR caps;
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev.physical, dev.surface, &caps);
    if (result != VK_SUCCESS) {
        logf("vk: querying surface capabilities failed (VkResult %d)", (int)result);
        return false;
    }
    if (caps.currentExtent.width != UINT32_MAX) width = caps.currentExtent.width, height = caps.currentExtent.height;
    logf("vk: surface caps images %u..%u extent %ux%u..%ux%u usage %#x alpha %#x transform %#x",
         caps.minImageCount, caps.maxImageCount, caps.minImageExtent.width, caps.minImageExtent.height,
         caps.maxImageExtent.width, caps.maxImageExtent.height, caps.supportedUsageFlags,
         caps.supportedCompositeAlpha, caps.supportedTransforms);
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        logf("vk: surface does not support transfer-destination swapchain images");
        return false;
    }
    if (!width || !height) return false;
    uint32_t n = 0;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical, dev.surface, &n, nullptr);
    if (result != VK_SUCCESS || !n) {
        logf("vk: querying surface formats failed (VkResult %d, count %u)", (int)result, n);
        return false;
    }
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical, dev.surface, &n, formats.data());
    VkSurfaceFormatKHR fmt = formats[0];
    for (const auto& f : formats)   // (an UNORM target: the game's window textures are presented as they are)
        // (in sRGB: RADV on Wayland lists every format again with PASS_THROUGH_EXT and the compositor's colour spaces,
        // and the last UNORM entry left the compositor an untagged surface, a grey picture. Still the last sRGB one, so
        // a driver that lists only sRGB (NVIDIA, X11, Windows) keeps the format it had.)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            fmt = f;
    {   // (relink) HDR output: HDR10 words (the game's own HDR window textures are RGB10A2 holding PQ BT.2020: host/nn_vi.c tells
        // it the display is HDR from hdr_output) when the row is on, the surface lists them and Windows' HDR is on for the display
        bool hdr_now = false;
        if (hdr_ext && hdr_wanted()) {
            float peak = 0;
            // (RX_VK_HDR_FORCE=1, tests: the display's answer left out, the surface's formats alone decide)
            static const bool force = getenv("RX_VK_HDR_FORCE") && atoi(getenv("RX_VK_HDR_FORCE")) != 0;
            const bool display_hdr = hdr_display(&peak) || force;
            static bool listed = false;
            if (!listed) {
                listed = true;
                std::string all;
                for (const auto& f : formats) all += " " + std::to_string((int)f.format) + "/" + std::to_string((int)f.colorSpace);
                logf("vk: HDR: the surface's formats (format/colour space):%s", all.c_str());
            }
            const VkSurfaceFormatKHR* found = nullptr;
            for (const auto& f : formats)
                if (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 && f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) found = &f;
            static int told = -1;
            const int tell = !found ? 1 : !display_hdr ? 2 : 3;
            if (tell != told) {
                told = tell;
                logf(tell == 1 ? "vk: HDR asked for, but this display's surface lists no HDR10 format: the picture stays SDR"
                     : tell == 2 ? "vk: HDR asked for, but Windows' HDR is off for this display: the picture stays SDR"
                                 : "vk: HDR output: HDR10 (ST 2084, BT.2020), the display's peak %.0f nits, paper white %d nits",
                     peak > 0 ? peak : 1000.0f, hdr_paper_white_nits());
            }
            if (found && display_hdr) {
                fmt = *found, hdr_now = true;
                hdr_peak.store(peak > 0 ? (int)peak : 1000);
            }
        }
        hdr_state.store(hdr_now ? 1 : 0);
    }
    // mailbox when the driver has it (latest frame, no tear). The game paces itself (nn_vi.c).
    n = 0;
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(dev.physical, dev.surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes;
    if (result == VK_SUCCESS && n) {
        modes.resize(n);
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(dev.physical, dev.surface, &n, modes.data());
    }
    if (result != VK_SUCCESS || modes.empty())
        logf("vk: querying present modes failed (VkResult %d, count %u); trying mandatory FIFO mode", (int)result, n);
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    // (as on a card or driver with vsync forced. relink: and the "30, smooth" frame rate, whose frames are each held for the
    // same number of refreshes by presenting them again: only FIFO shows every present, one a refresh)
    // (relink: and the Insert menu's V-Sync, for a card whose driver has no mailbox (AMD's on Windows): immediate tears)
    const bool force_fifo = getenv("RX_VK_FIFO") != nullptr || opts.vsync != 0 || (fps_low_cap() > 0 && !fifo_held_for_startup.load());
    if (!force_fifo) {
        for (auto m : modes)
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) mode = m;
        // (Windows, no mailbox: AMD's driver has none, and FIFO held anything under 60 fps at 30.
        //  Linux: immediate on NVIDIA's X11 path turns vblank off for every output, so the other
        //  monitor and other programs tear. RX_VK_IMMEDIATE=1 still asks for it.)
        const bool want_immediate = getenv("RX_VK_IMMEDIATE") != nullptr;
#ifdef _WIN32
        if (want_immediate || mode == VK_PRESENT_MODE_FIFO_KHR)
#else
        if (want_immediate)
#endif
            for (auto m : modes)
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = m;
    }
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = dev.surface;
    ci.minImageCount = std::max(caps.minImageCount, 4u);   // (relink: mailbox with frame interpolation: an image always free)
    // (relink: "30, smooth": three. Each frame takes two of the queue's places, and a fourth image let a third frame queue: 50 ms
    // more of delay for nothing, the display's pace being the frames' anyway)
    if (fps_low_cap() && mode == VK_PRESENT_MODE_FIFO_KHR) ci.minImageCount = std::max(caps.minImageCount, 3u);
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = {width, height};
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // The renderer copies into swapchain images. Readback is optional and only available on surfaces that allow it.
    if (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    // (relink) ReShade / RenoDX: their effects and RenoDX's HDR output pass draw into the swapchain's images (reshade_compat)
    if (reshade_compat()) {
        // (not storage: an sRGB swapchain format has no storage use, whatever the surface lists, and the swapchain would fail)
        for (VkImageUsageFlags u : {VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_SAMPLED_BIT})
            if (caps.supportedUsageFlags & u) ci.imageUsage |= u;
    }
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & ci.compositeAlpha)) {
        for (VkCompositeAlphaFlagBitsKHR alpha : {VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                                                  VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                                                  VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
            if (caps.supportedCompositeAlpha & alpha) { ci.compositeAlpha = alpha; break; }
    }
    ci.presentMode = mode;
    const uint32_t sharing[2] = {dev.queue_family, dev.present_family};
    if (dev.present_queue && dev.present_family != dev.queue_family) {   // (presented from another family: no ownership moves)
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = sharing;
    }
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = dev.swapchain;
    reflex_swapchain_info(ci);   // (NVIDIA Reflex: latency mode on)
    VkSwapchainKHR sc;
    result = vkCreateSwapchainKHR(dev.device, &ci, nullptr, &sc);
    if (result != VK_SUCCESS) {
        logf("vk: vkCreateSwapchainKHR failed (VkResult %d, extent %ux%u, format %d, usage %#x, mode %d, images %u)",
             (int)result, width, height, (int)fmt.format, ci.imageUsage, (int)mode, ci.minImageCount);
        return false;
    }
    if (vkGetSwapchainImagesKHR(dev.device, sc, &n, nullptr) != VK_SUCCESS || !n) {
        vkDestroySwapchainKHR(dev.device, sc, nullptr);
        return false;
    }
    std::vector<VkSemaphore> rendered(n, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& semaphore : rendered) {
        if (vkCreateSemaphore(dev.device, &si, nullptr, &semaphore) != VK_SUCCESS) {
            for (auto made : rendered)
                if (made) vkDestroySemaphore(dev.device, made, nullptr);
            vkDestroySwapchainKHR(dev.device, sc, nullptr);
            return false;
        }
    }
    if (dev.swapchain) {
        device_wait_idle_locked();   // (the present queue's presents of the old swapchain done too)
        for (auto semaphore : dev.swap_rendered) vkDestroySemaphore(dev.device, semaphore, nullptr);
        vkDestroySwapchainKHR(dev.device, dev.swapchain, nullptr);
    }
    dev.swapchain = sc;
    dev.swap_rendered = std::move(rendered);
    dev.swap_format = fmt.format;
    dev.swap_mode = mode;
    dev.swap_extent = {width, height};
    vkGetSwapchainImagesKHR(dev.device, sc, &n, nullptr);
    dev.swap_images.resize(n);
    vkGetSwapchainImagesKHR(dev.device, sc, &n, dev.swap_images.data());
    logf("vk: swapchain %ux%u, %u images, %s", width, height, n,
         mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo");
    reflex_swapchain_created(sc);   // (NVIDIA Reflex: its sleep mode on the new swapchain)
    return true;
}

namespace {
HWND main_window = nullptr;
} // namespace

bool hdr_wanted() {
    static const int env = getenv("RX_VK_HDR") && *getenv("RX_VK_HDR") ? (atoi(getenv("RX_VK_HDR")) != 0 ? 1 : 0) : -1;
    if (env >= 0) return env == 1;
    menu_init();   // (the options file, once: create_instance asks before pick_device does)
    return opts.hdr != 0;
}
bool hdr_output() { return hdr_state.load(std::memory_order_relaxed) == 1; }
int hdr_peak_nits() {
    const int p = hdr_peak.load(std::memory_order_relaxed);
    return p > 0 ? p : 1000;
}
/// (relink) Whether Windows' HDR is on for the display the window is on (DXGI 1.6: its colour space is ST 2084 / BT.2020), and that
/// display's peak brightness. True when it cannot be told (no DXGI 1.6, the display not found): the surface's formats decide then
bool hdr_display(float* peak) {
    *peak = 0;
#ifdef _WIN32
    HMONITOR mon = main_window ? MonitorFromWindow(main_window, MONITOR_DEFAULTTOPRIMARY) : nullptr;
    static HMODULE dxgi = LoadLibraryA("dxgi.dll");
    using Create = HRESULT(WINAPI*)(REFIID, void**);
    const Create create = dxgi ? (Create)(void*)GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory1* factory = nullptr;
    if (!mon || !create || FAILED(create(__uuidof(IDXGIFactory1), (void**)&factory)) || !factory) return true;
    int on = -1;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT a = 0; on < 0 && factory->EnumAdapters1(a, &adapter) == S_OK; a++) {
        IDXGIOutput* out = nullptr;
        for (UINT o = 0; on < 0 && adapter->EnumOutputs(o, &out) == S_OK; o++) {
            IDXGIOutput6* out6 = nullptr;
            DXGI_OUTPUT_DESC1 d{};
            if (SUCCEEDED(out->QueryInterface(__uuidof(IDXGIOutput6), (void**)&out6)) && out6 && SUCCEEDED(out6->GetDesc1(&d)) &&
                d.Monitor == mon) {
                on = d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 ? 1 : 0;
                *peak = d.MaxLuminance;
            }
            if (out6) out6->Release();
            out->Release();
        }
        adapter->Release();
    }
    factory->Release();
    return on != 0;
#else
    return true;
#endif
}

std::atomic<int> presents_shown{0};
std::atomic<int> low_cap_mfps{0};
bool reshade_compat() {
    // (asked when the device is made, after the instance: ReShade's Vulkan layer, an implicit layer the ReShade installer registers
    // for relink-game.exe, has loaded its DLL by then)
    static int on = -1;
    if (on >= 0) return on != 0;
    const char* e = getenv("RX_VK_RESHADE");
    if (e && *e) {
        on = atoi(e) != 0;
    } else {
#ifdef _WIN32
        on = GetModuleHandleA("ReShade64.dll") != nullptr || GetModuleHandleA("ReShade.dll") != nullptr;
#else
        on = 0;
#endif
    }
    if (on) logf("vk: ReShade / RenoDX compatibility on (%s): presents on the graphics queue, the swapchain's images drawable",
                 e && *e ? "RX_VK_RESHADE" : "ReShade's layer is loaded");
    return on != 0;
}
int fps_low_cap() {
    static const int cap = [] {
        const char* e = getenv("HOS_FPS_CAP");
        const char* u = getenv("HOS_FPS_UNLOCKED");
        int v = e && *e ? atoi(e) : 0;
        if (u && atoi(u) != 0) v = 0;   // (Unlocked wins, as host/nn_vi.c has it)
        return v <= 0 || v >= 60 ? 0 : v < 20 ? 20 : v;
    }();
    return cap;
}
int display_refresh_hz() {
#ifdef _WIN32
    // (read again every 120 calls: the window may go to another display)
    static int hz = 0, calls = 0;
    if (hz && ++calls % 120) return hz;
    HMONITOR m = main_window ? MonitorFromWindow(main_window, MONITOR_DEFAULTTOPRIMARY) : nullptr;
    MONITORINFOEXA mi{};
    mi.cbSize = sizeof mi;
    const char* device = m && GetMonitorInfoA(m, (MONITORINFO*)&mi) ? mi.szDevice : nullptr;
    DEVMODEA dm{};
    dm.dmSize = sizeof dm;
    const int reported = EnumDisplaySettingsA(device, ENUM_CURRENT_SETTINGS, &dm) ? (int)dm.dmDisplayFrequency : 0;
    hz = reported > 1 ? reported : 60;
    return hz;
#else
    return 60;
#endif
}
int low_cap_repeats() {
    const int cap = fps_low_cap();
    if (!cap || dev.offscreen || dev.swap_mode != VK_PRESENT_MODE_FIFO_KHR || framegen_on()) return 1;
    // (the refreshes nearest the cap's period: 60 and 120 Hz hold 30 fps exactly; 144 Hz holds each frame 5 refreshes, 28.8 fps)
    const int hz = display_refresh_hz();
    const int repeats = (hz + cap / 2) / cap;
    low_cap_mfps.store(hz * 1000 / std::max(repeats, 1), std::memory_order_relaxed);   // (the rate the display shows: host/nn_vi.c)
    static int told;
    if (repeats != told) {
        told = repeats;
        logf("vk: %d fps cap: each frame held for %d refresh%s of the %d Hz display (%.1f fps)", cap, std::max(repeats, 1),
             repeats > 1 ? "es" : "", hz, (double)hz / std::max(repeats, 1));
    }
    return std::min(std::max(repeats, 1), 8);
}

namespace {
// (relink) relink --prepare-shaders (a builder, after a build): the same work with no game, its progress as
// "@@progress" lines on stdout. A skip there leaves relink_prepare_skipped beside the game: its own starts then
// leave the work out too (what was not prepared is built when first drawn), until a --prepare-shaders run ends it
bool prepare_only() { return getenv("RELINK_PREPARE_ONLY") != nullptr; }
std::string skipped_path() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string path(exe, n);
    return path.substr(0, path.find_last_of("\\/") + 1) + "relink_prepare_skipped";
}
// (relink) The first start after an update prepares every shader (prepare_run) and the pipelines the last sessions noted
// (precompile_run): a minute and more, before the game's first frame. They run on a thread of their own while this one, the
// renderer's start, shows the window at once with what they are doing on it, and keeps it shown and sized; on Windows it is also
// the window's thread, and handles its messages meanwhile (it would not answer otherwise). Closing the window stops them
// (mkwvk_close_asked): what was made is kept, the next start goes on from there. RX_VK_STARTUP_SCREEN=0: as before, all
// of it before the window shows anything (and a console with the count on Windows)
std::atomic<int> prep_stage{0};   // 0 the programs, 1 the pipelines the notes predict (a first start), 2 the noted pipelines
std::atomic<size_t> prep_done{0}, prep_total{0};
std::atomic<bool> prep_running{false}, prep_finished{false};
HANDLE prep_ended = nullptr;
#ifndef _WIN32
const MkwVkWindow* prep_window = nullptr;   // (its size)
#endif
void prep_progress(size_t done, size_t total) {
    prep_done.store(done, std::memory_order_relaxed);
    prep_total.store(total, std::memory_order_relaxed);
}
bool screen_wanted() {
    const char* e = getenv("RX_VK_STARTUP_SCREEN");
    return !e || atoi(e) != 0;
}
/// The window's size now (0x0: minimized, or unknown)
void window_size(uint32_t* w, uint32_t* h) {
    *w = *h = 0;
#ifdef _WIN32
    RECT r{};
    if (main_window && GetClientRect(main_window, &r)) *w = (uint32_t)(r.right - r.left), *h = (uint32_t)(r.bottom - r.top);
#else
    int width = 0, height = 0;
    if (prep_window) prep_window->drawable_size(prep_window->window, &width, &height);
    if (width > 0 && height > 0) *w = (uint32_t)width, *h = (uint32_t)height;
#endif
}
void prepare_with_screen() {
    prep_ended = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    prep_running.store(true);
    if (fps_low_cap() && !getenv("RX_VK_FIFO") && dev.swap_extent.width && dev.swap_extent.height) {
        // (the start-up screen's presents in the usual mode: the swapchain made again at its next present, present_image)
        fifo_held_for_startup.store(true);
        dev.swap_resize.store((uint64_t)dev.swap_extent.width << 32 | dev.swap_extent.height, std::memory_order_release);
    }
    std::thread worker([] {
        const bool update = prepare_is_update();   // (an update, RX_VK_PREPARE_BACKGROUND=1: what the notes name now, the rest in play)
        const bool prepared = !update && prepare_run(prep_progress, false);
        if (prepared && !prepare_stopped()) {   // (a first start only: the pipelines the notes predict, into the driver's cache)
            prep_done.store(0), prep_total.store(0);
            prep_stage.store(1);
            predict_run(prep_progress);
        }
        if (!prepare_stopped()) {
            prep_done.store(0), prep_total.store(0);
            prep_stage.store(2);
            precompile_run(prep_progress);
        }
        if (prepare_stopped()) pipeline_cache_save(true);   // (the driver's part built so far, for the next start)
        else if (update) prepare_background();
        prep_finished.store(true, std::memory_order_release);
        SetEvent(prep_ended);
    });
    // (the screen is drawn and presented on a thread of its own: a present can wait for as long as the window cannot be seen (FIFO
    // on Wayland, a locked screen), and that must hold neither the window's messages nor the game's start)
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    static std::atomic<bool> paint_stop{false};
    HANDLE painted = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    std::thread painter([t0, painted] {
        ProgressClock eta;
        int eta_stage = -1;
        double next_log = 10;
        bool prepared = false;   // (this start prepares the programs: a first start after an update, two steps)
        double finished_at = -1;   // (when the work ended: the staff's ending is played for kStartupOutro seconds after it)
        bool staff_shown = false;   // (the screen showed the staff open: a start that lasted past its first second)
        uint32_t last_w = dev.swap_extent.width, last_h = dev.swap_extent.height;
        for (;;) {
            const bool finished = prep_finished.load(std::memory_order_acquire);
            const double secs = std::chrono::duration<double>(clock::now() - t0).count();
            const int stage = prep_stage.load();
            const size_t done = prep_done.load(std::memory_order_relaxed), total = prep_total.load(std::memory_order_relaxed);
            if (stage != eta_stage) eta.reset(), eta_stage = stage;
            if (total) eta.note(secs, done, total);
            const double left = total ? eta.left() : -1;
            if (secs >= next_log && total) {
                logf("vk: start-up: %s %zu of %zu%s%s", stage == 2 ? "pipelines" : stage ? "predicted pipelines" : "shaders", done, total, left >= 0 ? ", " : "",
                     left_text(left).c_str());
                next_log = secs + 10;
            }
            if (stage == 0 && total) prepared = true;
            StartupState st;
            st.stage = stage, st.done = done, st.total = total, st.left = left, st.secs = secs;
            st.prepared = prepared, st.finished = finished, st.stopping = prepare_stopped();
            if (finished && finished_at < 0) finished_at = secs;
            if (!finished && secs >= 1.0) staff_shown = true;
            st.staff_shown = staff_shown;
            st.since_finished = finished ? secs - finished_at : -1;
            StartupView view = startup_view(st);
            view.gpu = dev.props.deviceName;
            startup_screen(&view);
            uint32_t w, h;
            window_size(&w, &h);
            if (w && h && (w != last_w || h != last_h)) {   // (resized: the window's thread leaves it to the renderer, not started yet)
                dev.swap_resize.store((uint64_t)w << 32 | h, std::memory_order_release);
                last_w = w, last_h = h;
            }
            if (w && h) present_image(nullptr, dev.swap_extent.width, dev.swap_extent.height, nullptr);
            // (the staff's ending, played in full before the game takes the window, however early the work ended; a stop
            // asked by closing does not wait. Drawn at about 60 frames a second meanwhile: it is a moving picture)
            const bool closing = finished && staff_shown && !st.stopping && secs - finished_at < kStartupOutro;
            if ((finished || paint_stop.load()) && !closing) break;
            if (!finished) Sleep(33);
            else {   // (the ending: a frame every sixtieth of a second, counted from its start, whatever the present waited)
                const double next = finished_at + (std::floor((secs - finished_at) * 60) + 1) / 60.0;
                const double now = std::chrono::duration<double>(clock::now() - t0).count();
                if (next > now) Sleep((DWORD)std::max(1.0, (next - now) * 1000));
            }
        }
        SetEvent(painted);
    });
    // (this thread: the window's messages on Windows, until the work is done)
    for (;;) {
#ifdef _WIN32
        MSG m;
        while (PeekMessageA(&m, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&m);
            DispatchMessageA(&m);
        }
#endif
        if (WaitForSingleObject(prep_ended, 33) == WAIT_OBJECT_0) break;
    }
    worker.join();
    paint_stop.store(true);
    if (WaitForSingleObject(painted, 1000 + (DWORD)(kStartupOutro * 1000)) == WAIT_OBJECT_0) painter.join();
    else {   // (its present waits for the window to be seen: the game's own presents will, too)
        logf("vk: start-up: the window is not being shown; the game starts without waiting for it");
        painter.detach();
    }
    startup_screen(nullptr);
    prep_running.store(false);
    if (fifo_held_for_startup.exchange(false)) {   // ("30, smooth": FIFO from the game's first frame on)
        const uint64_t want = dev.swap_resize.load(std::memory_order_acquire);
        if (!want && dev.swap_extent.width && dev.swap_extent.height)
            dev.swap_resize.store((uint64_t)dev.swap_extent.width << 32 | dev.swap_extent.height, std::memory_order_release);
    }
    logf("vk: start-up: the shaders took %.1f s", std::chrono::duration<double>(clock::now() - t0).count());
}
bool started_up() {
    if (const char* e = getenv("RX_VK_START_DELAY")) Sleep((DWORD)atoi(e) * 1000);   // (diagnosis: a long start, as the first one is)
    if (!replay_start()) return false;
    if (prepare_only()) {
        // (three parts: the programs, the pipelines the notes predict, then those they name; most of the time is the first)
        static int part;
        const auto progress = [](size_t done, size_t total) {
            static size_t last = ~(size_t)0;
            if (!total) return;
            static const size_t from[3] = {0, 700, 900}, span[3] = {700, 200, 100};
            const size_t at = from[part] + done * span[part] / total;
            if (at == last) return;
            last = at;
            printf("@@progress %.3f\n", at / 1000.0);
            fflush(stdout);
        };
        printf("@@stage shaders Getting the game ready\n");
        fflush(stdout);
        part = 0;
        const bool prepared = prepare_run(progress, false);
        part = 1;
        if (prepared) predict_run(progress);
        part = 2;
        precompile_run(progress);
        pipeline_cache_save(true);
        remove(skipped_path().c_str());
        return true;
    }
    if (FILE* f = fopen(skipped_path().c_str(), "rb")) {
        fclose(f);
        logf("vk: the shaders' preparation was skipped at the build: left to when each is first drawn");
        return true;
    }
    if (precompile_wanted()) {   // (a release build, before the game: nothing to build during play)
        if (!dev.offscreen && screen_wanted()) {
            prepare_with_screen();
            return true;
        }
        // (a console while it lasts: there is no window yet, and a long wait with nothing on screen looks like a hang)
        static HANDLE console = nullptr;
        const auto progress = [](size_t done, size_t total) {
            static bool tried = false;
            if (!total) return;
            if (!tried) {
                tried = true;
                if (AllocConsole()) {
#ifdef _WIN32
                    // This console is ours. Selection must not block preparation's progress writes.
                    const HANDLE input = CreateFileA("CONIN$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                                     nullptr, OPEN_EXISTING, 0, nullptr);
                    if (input != INVALID_HANDLE_VALUE) {
                        DWORD mode;
                        if (GetConsoleMode(input, &mode))
                            SetConsoleMode(input, (mode & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS);
                        CloseHandle(input);
                    }
#endif
                    SetConsoleTitleA("Project ReLink");
                    console = CreateFileA("CONOUT$", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
                    if (console == INVALID_HANDLE_VALUE) console = nullptr;
                }
            }
            {   // (in the log too, every 10 s: a run with no window has nothing else to show it)
                static DWORD next_log = 0;
                const DWORD now = GetTickCount();
                if (!next_log) next_log = now + 10000;
                if ((int32_t)(now - next_log) >= 0) {
                    logf("vk: start-up: %zu of %zu", done, total);
                    next_log = now + 10000;
                }
            }
            if (!console) return;
            char line[80];
            const int n = snprintf(line, sizeof line, "%cGetting the game ready: %zu of %zu (%zu%%)   ", 13, done, total, done * 100 / total);
            DWORD wrote = 0;
            WriteFile(console, line, (DWORD)n, &wrote, nullptr);
        };
        const bool update = prepare_is_update();   // (an update: what the notes name now, the rest in play)
        if (!update && prepare_run(progress, false)) predict_run(progress);
        if (update) state_list_run();
        precompile_run(progress);
        if (update) prepare_background();
        if (console) CloseHandle(console), console = nullptr, FreeConsole();
    }
    return true;
}
// (the window's surface made: the presents' own queue kept where its family can present to it, create_device)
static void choose_present_queue() {
    if (!dev.present_queue) return;
    VkBool32 ok = VK_FALSE;
    if (vkGetPhysicalDeviceSurfaceSupportKHR(dev.physical, dev.present_family, dev.surface, &ok) != VK_SUCCESS || !ok) {
        logf("vk: queue family %u cannot present to the window: presents on the replay's queue", dev.present_family);
        dev.present_queue = VK_NULL_HANDLE, dev.present_family = dev.queue_family;
        return;
    }
    logf("vk: presents on a queue of their own, of compute family %u (RX_VK_PRESENT_QUEUE=0: off)", dev.present_family);
}

int start(void* hwnd) {
    main_window = (HWND)hwnd;
    if (!hwnd) {   // (HOS_HEADLESS: no window; the frames are rendered all the same, RX_VK_SHOTS writes them out)
        dev.offscreen = true;
        dev.swap_extent = {1920, 1080};
        dev.swap_format = VK_FORMAT_R8G8B8A8_UNORM;
        logf("vk: no window: presents render offscreen (%ux%u)", dev.swap_extent.width, dev.swap_extent.height);
        return started_up() ? 1 : 0;
    }
#ifdef _WIN32
    VkWin32SurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    si.hinstance = GetModuleHandleA(nullptr);
    si.hwnd = (HWND)hwnd;
    if (vkCreateWin32SurfaceKHR(dev.instance, &si, nullptr, &dev.surface) != VK_SUCCESS) return 0;
    VkBool32 ok = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(dev.physical, dev.queue_family, dev.surface, &ok);
    if (!ok) return 0;
    choose_present_queue();
    RECT r{};
    GetClientRect((HWND)hwnd, &r);
    if (!create_swapchain((uint32_t)(r.right - r.left), (uint32_t)(r.bottom - r.top))) return 0;
#else
    const MkwVkWindow* w = (const MkwVkWindow*)hwnd;
    prep_window = w;
    if (w->version != MKWVK_WINDOW_VERSION) {
        logf("vk: unsupported SDL window bridge version %u (expected %u)", w->version, MKWVK_WINDOW_VERSION);
        return 0;
    }
    if (!w->create_surface(w->window, (void*)dev.instance, (void*)&dev.surface)) {
        logf("vk: SDL_Vulkan_CreateSurface failed; check SDL_VIDEODRIVER and the Vulkan window-system extensions");
        return 0;
    }
    VkBool32 ok = VK_FALSE;
    VkResult present = vkGetPhysicalDeviceSurfaceSupportKHR(dev.physical, dev.queue_family, dev.surface, &ok);
    if (present != VK_SUCCESS || !ok) {
        logf("vk: queue-family present support failed (VkResult %d, supported %u)", (int)present, ok);
        return 0;
    }
    choose_present_queue();
    int width = 0, height = 0;
    w->drawable_size(w->window, &width, &height);
    if (width <= 0 || height <= 0) {
        logf("vk: SDL returned zero drawable size %dx%d", width, height);
        return 0;
    }
    if (!create_swapchain((uint32_t)width, (uint32_t)height)) {
        logf("vk: swapchain creation failed for drawable %dx%d", width, height);
        return 0;
    }
#endif
    if (!started_up()) {
        logf("vk: renderer startup failed after window and swapchain initialization");
        return 0;
    }
    return 1;
}

// (the window's thread: only noted. The swapchain is remade by whoever presents next, in present_image: made here, it went from under a
// frame between its acquire and its submit, and the frame's commands named images of the swapchain just destroyed: a lost device)
void resize(int width, int height) {
    if (dev.offscreen || width <= 0 || height <= 0) return;
    dev.swap_resize.store((uint64_t)(uint32_t)width << 32 | (uint32_t)height, std::memory_order_release);
}

const char* describe() {
    static std::string d;
    d = std::string(dev.props.deviceName) + " (Vulkan, native)";
    return d.c_str();
}

MkwVkRenderer table;

} // namespace
} // namespace mkwvk

// (relink) The window is being closed (host/display.c, host/posix/display_sdl.c, before they end the process): while the
// shaders are prepared, their workers are stopped and what they made is kept first (10 s at most: a driver can take longer on one
// pipeline). Looked up by name, so an executable and a back end of different versions still close
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
void mkwvk_close_asked(void) {
    using namespace mkwvk;
    if (!prep_running.load() || prepare_stopped()) return;
    logf("vk: the window was closed while the shaders were prepared: stopping, what is done is kept");
    prepare_stop();
    if (WaitForSingleObject(prep_ended, 10000) != WAIT_OBJECT_0) logf("vk: the preparation did not stop within 10 s: closing anyway");
}

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
const MkwVkRenderer* mkwvk_init(const MkwVkHost* h) {
    using namespace mkwvk;
    if (!h || h->version != MKWVK_API_VERSION) return nullptr;
    host = h;
    if (!load_global()) {
        logf("vk: vulkan-1.dll not found");
        return nullptr;
    }
    if (!create_instance() || !pick_device() || !create_device()) return nullptr;
    logf("vk: %s, Vulkan %u.%u, driver %#x; mesh shaders %s, host memory import %s, pipeline libraries %s",
         dev.props.deviceName, VK_VERSION_MAJOR(dev.props.apiVersion), VK_VERSION_MINOR(dev.props.apiVersion),
         dev.props.driverVersion, dev.mesh_shader ? "yes" : "no", dev.external_memory_host ? "yes" : "no",
         dev.graphics_pipeline_library ? "yes" : "no");
    if (!dev.mesh_shader && !dev.mesh_fallback && !getenv("RX_VK_NO_MESH_WARNING")) {
        // (the game draws its foliage, grass and many effects with mesh shaders: without them those draws are left out and the
        // picture is broken (an RX 580's "static"-like scenes, 2026-11-08). Said once, on its own thread, and the game goes on.)
        // (a GPU with no buffer device addresses has no fallback to run either)
        logf("vk: this GPU has no mesh shaders: the game's foliage, grass and many effects cannot be drawn");
#ifdef _WIN32
        std::thread([name = std::string(dev.props.deviceName)] {
            const std::string text = "Your graphics card (" + name + ") or its driver has no mesh shader support, which Ocarina of Time "
                "uses for foliage, grass and many effects, so parts of the picture will be missing or broken.\n\nReLink needs a GPU "
                "with mesh shaders: NVIDIA GeForce RTX 20 series or newer, AMD Radeon RX 6000 series or newer, or Intel Arc. If your "
                "card is one of those, update its driver.";
            MessageBoxA(nullptr, text.c_str(), "Project ReLink: graphics card", MB_OK | MB_ICONWARNING | MB_TOPMOST);
        }).detach();
#endif
    }
    table.version = MKWVK_API_VERSION;
    table.start = start;
    table.resize = resize;
    table.describe = describe;
    fill_replay_table(table);
    return &table;
}

namespace mkwvk {
HWND app_window() { return main_window; }   // (relink: the Insert menu's Fullscreen row)

// (relink, issues 21 and 23) The device is lost: nothing is drawn again in this run, while the game itself goes on (its sound,
// the frame counter), and players took it for a load that never ends: "black screen after the intro, waited a long time".
// Said once, in the log in plain words and in a box on its own thread (the replay goes on to write its report); the game is
// left running so the log keeps what follows
void device_lost_notice() {
    logf("vk: THE GRAPHICS CARD STOPPED (device lost): the picture stays black until the game is started again. "
         "Please send relink_log.txt with a report");
#ifdef _WIN32
    if (dev.offscreen || getenv("HOS_HEADLESS")) return;   // (no window: shader preparation, tests)
    std::thread([name = std::string(dev.props.deviceName)] {
        const std::string text = "The graphics card (" + name + ") stopped responding to ReLink (Vulkan: device lost), so the "
            "picture will stay black. The game cannot recover from this: close it and start it again.\n\nIf it happens again at "
            "the same place, please open an issue on ReLink's GitHub page and attach relink_log.txt from the game folder.";
        MessageBoxA(nullptr, text.c_str(), "Project ReLink: graphics card stopped", MB_OK | MB_ICONERROR | MB_TOPMOST);
    }).detach();
#endif
}
} // namespace mkwvk
