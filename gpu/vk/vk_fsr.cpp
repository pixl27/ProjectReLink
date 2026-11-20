// (relink) AMD FidelityFX Super Resolution 3.1's upscaler for vk_dlss.cpp, through AMD's FidelityFX API (vk_fsr.h).
//
// AMD's runtime, amd_fidelityfx_vk.dll on Windows or libamd_fidelityfx_vk.so on Linux (the FidelityFX SDK's Vulkan backend,
// MIT), is not linked: fsr_start loads it once the device is made, from RX_VK_FSR_LIB when set, else from the game folder
// (the folder above bin\, vk_paths.h's rule), then from the exe's own folder, and takes its five entry points by name.
// Only AMD's API headers are built in (third_party/ffx_api, with AMD's license text). It then makes a small upscaler on this
// device and lets it go: one that cannot be made means the GPU or its driver lacks what FSR needs, and FSR is not offered.
// The log's start says which, in one line.
// RX_VK_FSR=0: the runtime not loaded, FSR not offered. RX_VK_FSR_DEBUG=1: the runtime checks what it is given (FFX's debug
// checking), its messages in our log.
// RX_VK_FSR_SHARPNESS=<0..1>: FFX's RCAS after reconstruction, before the HUD; off when unset or 0. The menu's Sharpness
// is a separate final-picture pass which also sharpens the HUD.
#include "vk_fsr.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_WIN32) || defined(__linux__)
#include "third_party/ffx_api/ffx_api.h"
#include "third_party/ffx_api/ffx_upscale.h"
#include "third_party/ffx_api/vk/ffx_api_vk.h"
#endif

namespace mkwvk {

namespace {
std::atomic<int> g_why{FSR_NOT_STARTED};
std::string g_version;
}  // namespace

float fsr_rcas_sharpness() {
    static const float value = [] {
        const char* e = getenv("RX_VK_FSR_SHARPNESS");
        if (!e || !*e) return 0.0f;
        char* end = nullptr;
        const float v = strtof(e, &end);
        return end != e && std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : 0.0f;
    }();
    return value;
}

// (acquire: fsr_start's thread writes the entry points and g_version before it stores FSR_OK)
FsrWhy fsr_why() { return (FsrWhy)g_why.load(std::memory_order_acquire); }
const char* fsr_version() { return g_version.c_str(); }

void fsr_device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& all) {
#if defined(_WIN32) || defined(__linux__)
    // (the runtime's Vulkan backend takes vkGetBufferMemoryRequirements2KHR by that name through vkGetDeviceProcAddr, the
    // only function it takes by a KHR name; without VK_KHR_get_memory_requirements2 enabled the device gives a null pointer
    // for it, and the test upscaler's creation crashed on a call to address 0. The extension is core since Vulkan 1.1, so
    // every device here has it)
    const char* off = getenv("RX_VK_FSR");
    if (off && !strcmp(off, "0")) return;
    for (const char* name : {VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME}) {
        bool have = false, listed = false;
        for (const auto& e : all) have |= !strcmp(e.extensionName, name);
        for (const char* w : want) listed |= !strcmp(w, name);
        if (have && !listed) want.push_back(name);
    }
#else
    (void)want, (void)all;
#endif
}

#if defined(_WIN32) || defined(__linux__)

namespace {

struct Api {
#if defined(_WIN32)
    HMODULE dll = nullptr;
#else
    void* dll = nullptr;
#endif
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
} api;

bool env_on(const char* name) {
    const char* e = getenv(name);
    return e && *e && strcmp(e, "0") != 0;
}

#if defined(_WIN32)
/// A wide string as UTF-8, for the log
std::string utf8(const wchar_t* w, size_t n) {
    const int k = WideCharToMultiByte(CP_UTF8, 0, w, (int)n, nullptr, 0, nullptr, nullptr);
    std::string s(k > 0 ? k : 0, '\0');
    if (k > 0) WideCharToMultiByte(CP_UTF8, 0, w, (int)n, s.data(), k, nullptr, nullptr);
    return s;
}
std::string utf8(const std::wstring& w) { return utf8(w.data(), w.size()); }

/// The exe's folder, and the game folder (the folder above bin\ when image.bin is there: vk_paths.h's rule), from
/// GetModuleFileNameW: a folder with characters outside the ANSI code page is still found
void folders(std::wstring& game, std::wstring& exe_dir) {
    std::wstring exe(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, exe.data(), (DWORD)exe.size());
    if (!n || n >= exe.size()) return;
    exe.resize(n);
    const size_t slash = exe.find_last_of(L"\\/");
    exe_dir = slash == std::wstring::npos ? std::wstring() : exe.substr(0, slash);
    game = exe_dir;
    const size_t up = exe_dir.find_last_of(L"\\/");
    if (up != std::wstring::npos && !_wcsicmp(exe_dir.c_str() + up + 1, L"bin")) {
        const std::wstring above = exe_dir.substr(0, up);
        if (GetFileAttributesW((above + L"\\image.bin").c_str()) != INVALID_FILE_ATTRIBUTES) game = above;
    }
}

const char* load_error(DWORD e) {
    switch (e) {
    case ERROR_MOD_NOT_FOUND: return "a DLL it needs is missing: Microsoft's Visual C++ 2015-2022 runtime (x64)?";
    case ERROR_BAD_EXE_FORMAT: return "not a 64-bit DLL";
    case ERROR_INVALID_IMAGE_HASH: return "Windows refused its signature";
    case ERROR_ACCESS_DENIED: return "access denied";
    default: return "";
    }
}
#else
/// A UTF-32 Linux wide string as UTF-8, for the log
std::string utf8(const wchar_t* w, size_t n) {
    std::string s;
    s.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const uint32_t cp = (uint32_t)w[i];
        if (cp < 0x80u) {
            s.push_back((char)cp);
        } else if (cp < 0x800u) {
            s.push_back((char)(0xC0u | (cp >> 6)));
            s.push_back((char)(0x80u | (cp & 0x3Fu)));
        } else if (cp < 0x10000u) {
            s.push_back((char)(0xE0u | (cp >> 12)));
            s.push_back((char)(0x80u | ((cp >> 6) & 0x3Fu)));
            s.push_back((char)(0x80u | (cp & 0x3Fu)));
        } else if (cp <= 0x10FFFFu) {
            s.push_back((char)(0xF0u | (cp >> 18)));
            s.push_back((char)(0x80u | ((cp >> 12) & 0x3Fu)));
            s.push_back((char)(0x80u | ((cp >> 6) & 0x3Fu)));
            s.push_back((char)(0x80u | (cp & 0x3Fu)));
        } else {
            s.push_back('?');
        }
    }
    return s;
}

bool file_exists_linux(const std::string& p) {
    struct stat st{};
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

void folders_linux(std::string& game, std::string& exe_dir) {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return;
    buf[n] = '\0';
    std::string exe(buf, (size_t)n);
    const size_t slash = exe.find_last_of('/');
    exe_dir = slash == std::string::npos ? std::string() : exe.substr(0, slash);
    game = exe_dir;
    const size_t up = exe_dir.find_last_of('/');
    if (up != std::string::npos && exe_dir.compare(up + 1, std::string::npos, "bin") == 0) {
        const std::string above = exe_dir.substr(0, up);
        if (file_exists_linux(above + "/image.bin")) game = above;
    }
}
#endif

/// FFX's messages (its debug checking, its errors), into our log: the first 40, then every 1000th
void message(uint32_t type, const wchar_t* text) {
    static std::atomic<uint64_t> n{0};
    const uint64_t k = n++;
    if (k >= 40 && k % 1000) return;
    std::string m = text ? utf8(text, wcslen(text)) : std::string();
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r')) m.pop_back();
    logf("vk: AMD FSR says (%s, message %llu): %s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning",
         (unsigned long long)k + 1, m.c_str());
}

/// The upscaler versions the runtime has for Vulkan, its first (the one used) into g_version; false: none
bool versions() {
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    uint64_t count = 0;
    q.outputCount = &count;
    if (api.query(nullptr, &q.header) != FFX_API_RETURN_OK || !count) return false;
    std::vector<uint64_t> ids(count);
    std::vector<const char*> names(count, nullptr);
    q.versionIds = ids.data(), q.versionNames = names.data();
    if (api.query(nullptr, &q.header) != FFX_API_RETURN_OK || !count) return false;
    g_version = names[0] ? names[0] : "?";
    return true;
}

}  // namespace

struct FsrContext {
    ffxContext ctx = nullptr;
    // (FFX keeps what its creation was given until the context is destroyed: these live as long as it)
    ffxCreateBackendVKDesc backend{};
    ffxCreateContextDescUpscale desc{};
};

FsrContext* fsr_create(uint32_t max_render_w, uint32_t max_render_h, uint32_t out_w, uint32_t out_h, unsigned flags) {
    if (!api.create) return nullptr;
    FsrContext* c = new FsrContext;
    c->backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    c->backend.vkDevice = dev.device, c->backend.vkPhysicalDevice = dev.physical;
    c->backend.vkDeviceProcAddr = vkGetDeviceProcAddr;
    c->desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    c->desc.header.pNext = &c->backend.header;
    // (the colour vk_dlss.cpp gives is the game's tonemapped linear LDR colour: no HDR, no auto-exposure (FFX then takes
    // an exposure of 1). Its depth is not reversed (0 near, 1 far: dlss_prep.comp writes the depth buffer as it is). Its
    // motion vectors are at the render size and without the jitter (dlss_prep.comp takes it off): neither flag)
    static const bool debug = env_on("RX_VK_FSR_DEBUG");
    c->desc.flags = (flags & FSR_DEPTH_INFINITE ? FFX_UPSCALE_ENABLE_DEPTH_INFINITE : 0u) |
                    (flags & FSR_DYNAMIC ? FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION : 0u) |
                    (debug ? FFX_UPSCALE_ENABLE_DEBUG_CHECKING : 0u);
    c->desc.maxRenderSize = {max_render_w, max_render_h};
    c->desc.maxUpscaleSize = {out_w, out_h};
    c->desc.fpMessage = message;
    const ffxReturnCode_t r = api.create(&c->ctx, &c->desc.header, nullptr);
    if (r != FFX_API_RETURN_OK || !c->ctx) {
        logf("vk: AMD FSR: its upscaler was not made (FFX's code %u) for %ux%u -> %ux%u", (unsigned)r, max_render_w,
             max_render_h, out_w, out_h);
        if (c->ctx) api.destroy(&c->ctx, nullptr);
        delete c;
        return nullptr;
    }
    return c;
}

void fsr_destroy(FsrContext* c) {
    if (!c) return;
    if (c->ctx && api.destroy) api.destroy(&c->ctx, nullptr);
    delete c;
}

int fsr_jitter_phases(FsrContext* c, uint32_t render_w, uint32_t out_w) {
    if (!c || !api.query) return 0;
    int32_t n = 0;
    ffxQueryDescUpscaleGetJitterPhaseCount q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
    q.renderWidth = render_w, q.displayWidth = out_w, q.pOutPhaseCount = &n;
    return api.query(&c->ctx, &q.header) == FFX_API_RETURN_OK && n > 0 ? n : 0;
}

uint64_t fsr_memory(FsrContext* c) {
    if (!c || !api.query) return 0;
    FfxApiEffectMemoryUsage u{};
    ffxQueryDescUpscaleGetGPUMemoryUsage q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE;
    q.gpuMemoryUsageUpscaler = &u;
    return api.query(&c->ctx, &q.header) == FFX_API_RETURN_OK ? u.totalUsageInBytes : 0;
}

namespace {
FfxApiResource resource(const FsrImage& im, uint32_t state) {
    FfxApiResource r{};
    r.resource = (void*)im.image;
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = ffxApiGetSurfaceFormatVK(im.format);
    r.description.width = im.width, r.description.height = im.height, r.description.depth = 1;
    r.description.mipCount = 1;
    r.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    r.description.usage = FFX_API_RESOURCE_USAGE_UAV;   // (as ffxApiGetImageResourceDescriptionVK says of an image with storage use)
    r.state = state;
    return r;
}
}  // namespace

uint32_t fsr_dispatch(FsrContext* c, VkCommandBuffer cmd, const FsrFrame& f) {
    if (!c || !api.dispatch) return FFX_API_RETURN_ERROR;
    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = (void*)cmd;
    // (the inputs in FFX's compute-read state, its SHADER_READ_ONLY_OPTIMAL; the output in its unordered-access state, GENERAL.
    // FFX moves them to what each of its passes needs and back to these by the end of the dispatch)
    d.color = resource(f.color, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = resource(f.depth, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = resource(f.motion, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.output = resource(f.output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    // (no exposure, reactive or transparency mask: empty resources)
    d.jitterOffset = {f.jitter[0], f.jitter[1]};
    d.motionVectorScale = {f.motion_scale[0], f.motion_scale[1]};
    d.renderSize = {f.render_w, f.render_h};
    d.upscaleSize = {f.out_w, f.out_h};
    // (normally the game's Sharpness row acts on the finished picture, including the HUD.  An explicit FSR value uses
    // FFX's own RCAS here instead, while this image is still the 3D alone)
    d.sharpness = fsr_rcas_sharpness();
    d.enableSharpening = d.sharpness > 0.0f;
    static std::atomic<bool> told_sharpness{false};
    if (d.enableSharpening && !told_sharpness.exchange(true))
        logf("vk: AMD FSR: RCAS sharpens the 3D before the HUD at %.0f%% (RX_VK_FSR_SHARPNESS)", d.sharpness * 100.0f);
    d.frameTimeDelta = f.frame_ms;
    d.preExposure = 1.0f;
    d.reset = f.reset;
    d.cameraNear = f.near_z, d.cameraFar = f.far_z, d.cameraFovAngleVertical = f.fov_y;
    d.viewSpaceToMetersFactor = f.meters;
    d.flags = 0;   // (vk_dlss.cpp converts the game's perceptual network input to the linear colour FSR 3.1 requires)
    return api.dispatch(&c->ctx, &d.header);
}

void fsr_start() {
    static bool started = false;
    if (started || !dev.device) return;
    started = true;
    const char* off = getenv("RX_VK_FSR");
    if (off && !strcmp(off, "0")) {
        g_why = FSR_ENV_OFF;
        logf("vk: AMD FSR: off for this run (RX_VK_FSR=0)");
        return;
    }
    std::string from_utf8, looked, failed;
#if defined(_WIN32)
    std::wstring game, exe_dir;
    folders(game, exe_dir);
    std::vector<std::wstring> where;
    if (!game.empty()) where.push_back(game);
    if (!exe_dir.empty() && _wcsicmp(exe_dir.c_str(), game.c_str())) where.push_back(exe_dir);
    for (const auto& d : where) {
        looked += (looked.empty() ? "" : " and ") + utf8(d);
        const std::wstring path = d + L"\\amd_fidelityfx_vk.dll";
        if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) continue;
        // (its own folder first for what it imports, then the system's: not the current directory nor PATH)
        api.dll = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        DWORD e = api.dll ? 0 : GetLastError();
        // (relink) a module it imports is missing (126): Microsoft's Visual C++ runtime, on a PC that has none installed. The
        // release ships its three files in vcrt beside it; that folder is searched only then, so a PC with the runtime keeps
        // its own (newer) one for every other module of the game
        if (!api.dll && e == ERROR_MOD_NOT_FOUND) {
            const std::wstring vcrt = d + L"\\vcrt";
            if (GetFileAttributesW(vcrt.c_str()) != INVALID_FILE_ATTRIBUTES && AddDllDirectory(vcrt.c_str())) {
                api.dll = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                e = api.dll ? 0 : GetLastError();
                if (api.dll) logf("vk: AMD FSR: Microsoft's Visual C++ runtime taken from %s (none installed on this PC)", utf8(vcrt).c_str());
            }
        }
        if (api.dll) {
            from_utf8 = utf8(path);
            break;
        }
        char b[320];
        snprintf(b, sizeof b, "%s%s did not load (error %lu%s%s)", failed.empty() ? "" : "; ", utf8(path).c_str(),
                 (unsigned long)e, *load_error(e) ? ": " : "", load_error(e));
        failed += b;
    }
    if (!api.dll) {
        g_why = failed.empty() ? FSR_NO_RUNTIME : FSR_RUNTIME_FAILED;
        if (failed.empty()) logf("vk: AMD FSR: not available: no amd_fidelityfx_vk.dll in %s", looked.c_str());
        else logf("vk: AMD FSR: not available: %s", failed.c_str());
        return;
    }
    api.create = (PfnFfxCreateContext)(void*)GetProcAddress(api.dll, "ffxCreateContext");
    api.destroy = (PfnFfxDestroyContext)(void*)GetProcAddress(api.dll, "ffxDestroyContext");
    api.configure = (PfnFfxConfigure)(void*)GetProcAddress(api.dll, "ffxConfigure");
    api.query = (PfnFfxQuery)(void*)GetProcAddress(api.dll, "ffxQuery");
    api.dispatch = (PfnFfxDispatch)(void*)GetProcAddress(api.dll, "ffxDispatch");
#else
    std::vector<std::string> candidates;
    if (const char* env_lib = getenv("RX_VK_FSR_LIB"); env_lib && *env_lib) {
        candidates.emplace_back(env_lib);
    } else {
        std::string game, exe_dir;
        folders_linux(game, exe_dir);
        if (!game.empty()) candidates.push_back(game + "/libamd_fidelityfx_vk.so");
        if (!exe_dir.empty() && exe_dir != game) candidates.push_back(exe_dir + "/libamd_fidelityfx_vk.so");
    }
    for (const auto& path : candidates) {
        looked += (looked.empty() ? "" : " and ") + path;
        if (!file_exists_linux(path)) continue;
        dlerror();
        api.dll = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (api.dll) {
            from_utf8 = path;
            break;
        }
        const char* err = dlerror();
        char b[384];
        snprintf(b, sizeof b, "%s%s did not load (%s)", failed.empty() ? "" : "; ", path.c_str(),
                 err && *err ? err : "dlopen failed");
        failed += b;
    }
    if (!api.dll) {
        g_why = failed.empty() ? FSR_NO_RUNTIME : FSR_RUNTIME_FAILED;
        if (failed.empty()) logf("vk: AMD FSR: not available: no libamd_fidelityfx_vk.so at %s", looked.c_str());
        else logf("vk: AMD FSR: not available: %s", failed.c_str());
        return;
    }
    api.create = (PfnFfxCreateContext)dlsym(api.dll, "ffxCreateContext");
    api.destroy = (PfnFfxDestroyContext)dlsym(api.dll, "ffxDestroyContext");
    api.configure = (PfnFfxConfigure)dlsym(api.dll, "ffxConfigure");
    api.query = (PfnFfxQuery)dlsym(api.dll, "ffxQuery");
    api.dispatch = (PfnFfxDispatch)dlsym(api.dll, "ffxDispatch");
    if (!api.create || !api.destroy || !api.configure || !api.query || !api.dispatch) {
        dlclose(api.dll);
        api = {};
        g_why = FSR_RUNTIME_FAILED;
        logf("vk: AMD FSR: not available: %s is not AMD's FidelityFX API (an entry point is missing)", from_utf8.c_str());
        return;
    }
#endif
    if (!api.create || !api.destroy || !api.configure || !api.query || !api.dispatch) {
        api = {};
        g_why = FSR_RUNTIME_FAILED;
        logf("vk: AMD FSR: not available: %s is not AMD's FidelityFX API (an entry point is missing)", from_utf8.c_str());
        return;
    }
    if (!versions()) {
        api.create = nullptr;
        g_why = FSR_NO_UPSCALER;
        logf("vk: AMD FSR: not available: %s has no upscaler for Vulkan", from_utf8.c_str());
        return;
    }
    // (a small upscaler made on this device and let go at once, before anything is recorded with it: its creation makes
    // FSR's pipelines, so a device or driver without what they need fails here, not in play. On a thread of its own: the
    // first time on a PC that takes most of a second (864 ms on an RTX 3060 Ti), which the game's start need not wait for,
    // and the driver keeps the pipelines, so the upscaler made in play takes some 15 ms. Until it ends FSR is not offered
    // (FSR_NOT_STARTED). The device lives until the process ends: nothing destroys it under the thread)
    std::thread([path = from_utf8] {
        const auto t0 = std::chrono::steady_clock::now();
        FsrContext* test = fsr_create(256, 144, 512, 288, 0);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!test) {
            g_why = FSR_DEVICE;
            logf("vk: AMD FSR: not available: %s (%s) could not make an upscaler on %s", path.c_str(), g_version.c_str(),
                 dev.props.deviceName);
            return;
        }
        fsr_destroy(test);
        g_why.store(FSR_OK, std::memory_order_release);
        logf("vk: AMD FSR: available (%s, from %s; a test upscaler made on %s in %.0f ms)", g_version.c_str(), path.c_str(),
             dev.props.deviceName, ms);
    }).detach();
}

#else  // (platforms without Windows/Linux dynamic loader support)

struct FsrContext {};
FsrContext* fsr_create(uint32_t, uint32_t, uint32_t, uint32_t, unsigned) { return nullptr; }
void fsr_destroy(FsrContext*) {}
int fsr_jitter_phases(FsrContext*, uint32_t, uint32_t) { return 0; }
uint64_t fsr_memory(FsrContext*) { return 0; }
uint32_t fsr_dispatch(FsrContext*, VkCommandBuffer, const FsrFrame&) { return 1; }
void fsr_start() {
    static bool started = false;
    if (started) return;
    started = true;
    g_why = FSR_NOT_WINDOWS;
    logf("vk: AMD FSR: not available: unsupported host platform");
}

#endif

} // namespace mkwvk
