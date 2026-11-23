// (hookshot) The size NVIDIA's own driver lets its shader cache grow to, for the pipeline cache file's rule (pcache_load.h) and
// the start's log. A start builds about 2 GB of pipelines into that cache, the shader preparation's predicted ones into it alone
// (vk_draw.cpp predict_run), and the driver cleans its cache down to the limit as it is written: under kRoomMb, the pipelines it
// dropped are compiled again in play, the "everything compiles again" players see with no driver update (players' logs: caches
// held at 1025 MB, the 1 GB limit).
//  - Linux: __GL_SHADER_DISK_CACHE_SIZE, in bytes; host/hos_main.c driver_cache_room sets 10 GiB before Vulkan loads unless the
//    player set a value, so only a player's smaller value reads as small;
//  - Windows: the NVIDIA Control Panel's Shader Cache Size, for every program: the current global profile's
//    PS_SHADERDISKCACHE_MAX_SIZE (MB; 0xFFFFFFFF Unlimited) through nvapi64.dll's driver settings (NvAPI_DRS_*), read only.
//    Function ids, setting ids and NVDRS_SETTING_V1's layout: NVIDIA's open nvapi headers (nvapi_interface.h,
//    NvApiDriverSettings.h, nvapi.h). A launcher can raise a small one through those driver settings.
// Standard library (and Windows) only: tests/test_pcache_load.py builds the Linux part.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <vector>
#include <windows.h>
#endif

namespace nvidia_cache_limit {

/// What the cache must hold to keep a start's pipelines (with room for the predicted ones, which a full preparation adds)
constexpr uint64_t kRoomMb = 4096;
/// 0xFFFFFFFF: Unlimited
constexpr uint64_t kUnlimited = 0xFFFFFFFFull;

struct Limit {
    uint64_t mb = 0;     // (0: not known, or the cache off)
    bool off = false;    // (the driver's shader cache turned off: Disabled, or PS_SHADERDISKCACHE off)
    const char* where = "";
};

/// (Linux) __GL_SHADER_DISK_CACHE_SIZE's value (bytes) as a limit: not known when unset or not a number
inline Limit linux_limit(const char* size) {
    Limit l;
    l.where = "__GL_SHADER_DISK_CACHE_SIZE";
    if (!size || !*size) return l;
    char* end = nullptr;
    const unsigned long long bytes = strtoull(size, &end, 10);
    if (end == size || *end) return l;
    l.mb = bytes >> 20;
    if (!l.mb) l.off = true;
    return l;
}

#if defined(_WIN32)
/// (Windows) The NVIDIA Control Panel's Shader Cache Size, from the driver's settings: not known without NVIDIA's driver
/// (no nvapi64.dll) or where it refuses
inline Limit windows_limit() {
    Limit l;
    l.where = "Shader Cache Size in the NVIDIA Control Panel";
    const HMODULE nvapi = LoadLibraryW(L"nvapi64.dll");   // (kept loaded: the driver's own Vulkan driver has it loaded too)
    if (!nvapi) return l;
    using Query = void*(__cdecl*)(uint32_t);
    const Query query = (Query)(void*)GetProcAddress(nvapi, "nvapi_QueryInterface");
    if (!query) return l;
    using NoArgs = int(__cdecl*)();
    using SessionOut = int(__cdecl*)(void**);
    using SessionIn = int(__cdecl*)(void*);
    using ProfileOut = int(__cdecl*)(void*, void**);
    using GetSetting = int(__cdecl*)(void*, void*, uint32_t, void*);
    const auto initialize = (NoArgs)query(0x0150E828);              // (NvAPI_Initialize)
    const auto create = (SessionOut)query(0x0694D52E);              // (NvAPI_DRS_CreateSession)
    const auto load = (SessionIn)query(0x375DBD6B);                 // (NvAPI_DRS_LoadSettings)
    const auto global = (ProfileOut)query(0x617BFF9F);              // (NvAPI_DRS_GetCurrentGlobalProfile)
    const auto get = (GetSetting)query(0x73BF8338);                 // (NvAPI_DRS_GetSetting)
    const auto destroy = (SessionIn)query(0xDAD9CFF8);              // (NvAPI_DRS_DestroySession)
    if (!initialize || !create || !load || !global || !get || !destroy || initialize() != 0) return l;
    void* session = nullptr;
    if (create(&session) != 0 || !session) return l;
    // (NVDRS_SETTING_V1, pack 4: 12320 bytes; the DWORD current value at 8220)
    constexpr size_t kBytes = 12320, kCurrentAt = 8220;
    void* profile = nullptr;
    auto dword = [&](uint32_t id, uint32_t* v) {
        std::vector<uint8_t> s(kBytes);
        const uint32_t version = (uint32_t)kBytes | (1u << 16);
        memcpy(s.data(), &version, 4);
        if (get(session, profile, id, s.data()) != 0) return false;
        memcpy(v, s.data() + kCurrentAt, 4);
        return true;
    };
    uint32_t on = 1, mb = 0;
    if (load(session) == 0 && global(session, &profile) == 0) {
        dword(0x00198FFF, &on);   // (PS_SHADERDISKCACHE: 0 off)
        if (dword(0x00AC8497, &mb)) {   // (PS_SHADERDISKCACHE_MAX_SIZE, MB)
            l.mb = mb;
            l.off = !on || !mb;
        }
    }
    destroy(session);
    return l;
}
#endif

/// Whether the limit is known and under kRoomMb: the pipelines a start builds do not all stay
constexpr bool small(uint64_t mb) { return mb && mb < kRoomMb; }

}  // namespace nvidia_cache_limit
