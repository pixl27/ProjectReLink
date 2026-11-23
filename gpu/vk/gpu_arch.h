// SPDX-License-Identifier: GPL-3.0-or-later
// (hookshot) GPU architecture classification in one place (gpu_arch.h), with no Vulkan calls or environment reads.
// device_policy.h classifies DeviceFacts once from vendor ID, PCI device ID, device name and (for RADV GCN) 16-bit
// stage-input support so architecture-specific rules and driver workarounds share one table instead of scattered PCI ID
// lists and string checks. Verified by tests/test_gpu_arch.py.
#pragma once

#include <cstdint>
#include <cstring>

namespace mkwvk {

enum class GpuArch : uint8_t {
    Unknown = 0,
    // AMD
    AmdGcn,      // GFX6-8 (GCN 1-4: HD 7000/8000, Rx 200/300, RX 400/500)
    AmdVega,     // GFX9 (Vega 10/12/20, Radeon VII, Raven/Picasso/Renoir/Cezanne/Barcelo APUs)
    AmdRdna1,    // GFX10.1 (Navi 10/12/14: RX 5000 series)
    AmdRdna2,    // GFX10.3 (Navi 21-24: RX 6000 series, Van Gogh / Steam Deck, Rembrandt, Raphael/Granite Ridge)
    AmdRdna3,    // GFX11 / GFX11.5 (Navi 31-33: RX 7000 series, Phoenix/Hawk/Strix/Krackan APUs)
    AmdRdna4,    // GFX12 (Navi 44/48: RX 9060/9070, Radeon AI PRO R9700)
    // Intel
    IntelGen12,       // Xe-LP (Tiger/Rocket/Alder/Raptor Lake iGPUs, DG1)
    IntelAlchemist,   // Arc A-series (DG2: 0x56xx)
    IntelMeteorArrow, // Meteor Lake / Arrow Lake iGPUs (0x7Dxx)
    IntelLunar,       // Lunar Lake iGPUs (Xe2: 0x64xx)
    IntelBattlemage,  // Arc B-series discrete (Xe2: 0xE2xx)
    IntelPanther,     // Panther Lake iGPUs (Xe3: 0xB0xx)
    // NVIDIA
    NvidiaPascal,     // GTX 10 series
    NvidiaTuring,     // GTX 16 / RTX 20 series
    NvidiaAmpere,     // RTX 30 series
    NvidiaAda,        // RTX 40 series
    NvidiaBlackwell,  // RTX 50 series
};

inline const char* gpu_arch_name(GpuArch a) {
    switch (a) {
    case GpuArch::AmdGcn: return "amd_gcn";
    case GpuArch::AmdVega: return "amd_vega";
    case GpuArch::AmdRdna1: return "amd_rdna1";
    case GpuArch::AmdRdna2: return "amd_rdna2";
    case GpuArch::AmdRdna3: return "amd_rdna3";
    case GpuArch::AmdRdna4: return "amd_rdna4";
    case GpuArch::IntelGen12: return "intel_gen12";
    case GpuArch::IntelAlchemist: return "intel_alchemist";
    case GpuArch::IntelMeteorArrow: return "intel_meteor_arrow";
    case GpuArch::IntelLunar: return "intel_lunar";
    case GpuArch::IntelBattlemage: return "intel_battlemage";
    case GpuArch::IntelPanther: return "intel_panther";
    case GpuArch::NvidiaPascal: return "nvidia_pascal";
    case GpuArch::NvidiaTuring: return "nvidia_turing";
    case GpuArch::NvidiaAmpere: return "nvidia_ampere";
    case GpuArch::NvidiaAda: return "nvidia_ada";
    case GpuArch::NvidiaBlackwell: return "nvidia_blackwell";
    default: return "unknown";
    }
}

// AMD RDNA 4 (GFX12): Navi 48 (0x7550 RX 9070 series, 0x7551 Radeon AI PRO R9700) and Navi 44 (0x7590 RX 9060 series),
// with name matching as a fallback for future or uncatalogued GFX12 cards.
// Kept self-contained: tests/test_mma_software.py extracts and compiles is_rdna4() on its own.
inline bool is_rdna4(uint32_t vendor_id, uint32_t device_id, const char* device_name) {
    if (vendor_id != 0x1002) return false;
    if (device_id == 0x7550 || device_id == 0x7551 || device_id == 0x7590) return true;
    return device_name && (std::strstr(device_name, "RX 9") || std::strstr(device_name, "GFX12"));
}

// Match 4-digit "RX <first_digit>NNN" model names (e.g. RX 6750) without matching 3-digit OEM GCN names (e.g. RX 640).
inline bool has_rx_4digit_series(const char* name, char first_digit) {
    if (!name) return false;
    for (const char* p = std::strstr(name, "RX "); p; p = std::strstr(p + 3, "RX ")) {
        if (p[3] == first_digit && p[4] >= '0' && p[4] <= '9' && p[5] >= '0' && p[5] <= '9' &&
            p[6] >= '0' && p[6] <= '9')
            return true;
    }
    return false;
}

// Pure PCI device ID classification for AMD (0x1002), resolved before any device-name fallback so known PCI IDs
// (e.g. Polaris 12 Radeon RX 640 0x6987/0x699F or Polaris 22 RX Vega M 0x694C-0x694F) never misclassify by name.
inline GpuArch classify_amd_pci_id(uint32_t device_id) {
    // RDNA 4 (GFX12): Navi 48 / Navi 44
    if (device_id == 0x7550u || device_id == 0x7551u || device_id == 0x7590u) return GpuArch::AmdRdna4;
    switch (device_id) {
    // RDNA 3 (GFX11.0) and RDNA 3.5 (GFX11.5): Navi 31-33 + Phoenix/Hawk/Strix/Krackan APUs
    case 0x7448: case 0x7449: case 0x744A: case 0x744B: case 0x744C: case 0x745E:
    case 0x7460: case 0x7461: case 0x7470: case 0x747E:
    case 0x7480: case 0x7481: case 0x7483: case 0x7487: case 0x7489: case 0x748B: case 0x7499: case 0x749F:
    case 0x15BF: case 0x15C8: case 0x1900: case 0x1901:
    case 0x150E: case 0x1586: case 0x1114: case 0x1902:
        return GpuArch::AmdRdna3;
    // RDNA 2 (GFX10.3): Navi 21-24 discrete GPUs + Van Gogh/Rembrandt/Raphael/Granite Ridge/Mendocino/Sephiroth APUs
    case 0x73A1: case 0x73A2: case 0x73A3: case 0x73A5: case 0x73AB: case 0x73AE: case 0x73AF: case 0x73BF:
    case 0x73DF:
    case 0x73E0: case 0x73E1: case 0x73E3: case 0x73EF: case 0x73FF:
    case 0x7421: case 0x7422: case 0x7423: case 0x7424: case 0x743F:
    case 0x163F: case 0x1681: case 0x164E: case 0x13C0: case 0x1506: case 0x1435:
        return GpuArch::AmdRdna2;
    // RDNA 1 (GFX10.1): Navi 10 / 12 / 14
    case 0x7310: case 0x7312: case 0x7318: case 0x7319: case 0x731A: case 0x731B: case 0x731E: case 0x731F:
    case 0x7360: case 0x7362:
    case 0x7340: case 0x7341: case 0x7347: case 0x734F:
        return GpuArch::AmdRdna1;
    // Vega (GFX9) APUs: Raven/Picasso/Renoir/Cezanne/Barcelo
    case 0x15D8: case 0x15DD: case 0x1636: case 0x1638: case 0x164C: case 0x15E7:
        return GpuArch::AmdVega;
    default:
        break;
    }
    // Vega (GFX9) discrete: Vega 10 / 12 / 20
    if ((device_id >= 0x6860u && device_id <= 0x687Fu) || (device_id >= 0x69A0u && device_id <= 0x69AFu) ||
        (device_id >= 0x66A0u && device_id <= 0x66AFu))
        return GpuArch::AmdVega;
    // GCN 1-4 (GFX6-8): Oland/Bonaire/Hainan (0x6600-0x667F), Tahiti/Hawaii/Polaris 10/11 (0x6780-0x67FF),
    // Pitcairn/Cape Verde (0x6800-0x683F), Topaz/Tonga (0x6900-0x693F), Polaris 22/Vega M (0x694C-0x694F),
    // Polaris 12 (0x6980-0x699F), Polaris 10 RX 580 2048SP / 590 GME (0x6FDF), Fiji (0x7300).
    // Excludes TeraScale (0x6700-0x677F, 0x6840-0x685F, 0x6880-0x68FF).
    if ((device_id >= 0x6600u && device_id <= 0x667Fu) ||
        (device_id >= 0x6780u && device_id <= 0x67FFu) ||
        (device_id >= 0x6800u && device_id <= 0x683Fu) ||
        (device_id >= 0x6900u && device_id <= 0x693Fu) ||
        (device_id >= 0x694Cu && device_id <= 0x694Fu) ||
        (device_id >= 0x6980u && device_id <= 0x699Fu) ||
        device_id == 0x6FDFu || device_id == 0x7300u)
        return GpuArch::AmdGcn;
    return GpuArch::Unknown;
}

inline GpuArch classify_gpu_arch(uint32_t vendor_id, uint32_t device_id, const char* device_name = "",
                                 bool radv_gcn_hint = false) {
    if (vendor_id == 0x1002) {
        if (const GpuArch by_id = classify_amd_pci_id(device_id); by_id != GpuArch::Unknown)
            return by_id;
        if (device_name) {
            if (std::strstr(device_name, "RX 9") || std::strstr(device_name, "GFX12"))
                return GpuArch::AmdRdna4;
            if (std::strstr(device_name, "GFX11") || std::strstr(device_name, "RX 7") ||
                std::strstr(device_name, "PRO W7") || std::strstr(device_name, "Pro W7") ||
                std::strstr(device_name, "PRO V710") || std::strstr(device_name, "Pro V710") ||
                std::strstr(device_name, "740M") || std::strstr(device_name, "760M") ||
                std::strstr(device_name, "780M") || std::strstr(device_name, "820M") ||
                std::strstr(device_name, "840M") || std::strstr(device_name, "860M") ||
                std::strstr(device_name, "880M") || std::strstr(device_name, "890M") ||
                std::strstr(device_name, "8040S") || std::strstr(device_name, "8050S") ||
                std::strstr(device_name, "8060S") || std::strstr(device_name, "8065S"))
                return GpuArch::AmdRdna3;
            if (std::strstr(device_name, "GFX103") || std::strstr(device_name, "NAVI2") ||
                has_rx_4digit_series(device_name, '6') ||
                std::strstr(device_name, "PRO W6") || std::strstr(device_name, "Pro W6") ||
                std::strstr(device_name, "680M") || std::strstr(device_name, "660M") ||
                std::strstr(device_name, "610M"))
                return GpuArch::AmdRdna2;
            if (std::strstr(device_name, "GFX101") || std::strstr(device_name, "NAVI1") ||
                std::strstr(device_name, "RX 5300") || std::strstr(device_name, "RX 5500") ||
                std::strstr(device_name, "RX 5600") || std::strstr(device_name, "RX 5700") ||
                std::strstr(device_name, "W5500") || std::strstr(device_name, "W5700"))
                return GpuArch::AmdRdna1;
            if (std::strstr(device_name, "Vega") || std::strstr(device_name, "VEGA") ||
                std::strstr(device_name, "GFX9"))
                return GpuArch::AmdVega;
        }
        if (radv_gcn_hint) return GpuArch::AmdGcn;
        return GpuArch::Unknown;
    }
    if (vendor_id == 0x8086) {
        switch (device_id & 0xFF00u) {
        case 0xE200u: return GpuArch::IntelBattlemage;
        case 0xB000u: return GpuArch::IntelPanther;
        case 0x6400u: return GpuArch::IntelLunar;
        case 0x7D00u: return GpuArch::IntelMeteorArrow;
        case 0x5600u: return GpuArch::IntelAlchemist;
        case 0x9A00u: case 0x4600u: case 0xA700u: case 0x4C00u: case 0x4900u: return GpuArch::IntelGen12;
        default: return GpuArch::Unknown;
        }
    }
    if (vendor_id == 0x10DE) {
        const uint32_t hi = device_id & 0xFF00u;
        if ((hi >= 0x2B00u && hi <= 0x2D00u) || hi == 0x2F00u) return GpuArch::NvidiaBlackwell;
        if (hi >= 0x2600u && hi <= 0x2800u) return GpuArch::NvidiaAda;
        if (hi == 0x2200u || hi == 0x2400u || hi == 0x2500u) return GpuArch::NvidiaAmpere;  // (0x23xx is Hopper GH100)
        if (hi == 0x1E00u || hi == 0x1F00u || hi == 0x2100u) return GpuArch::NvidiaTuring;
        // Pascal: GP102/104/106/107 (0x1Bxx-0x1Cxx) and GP108 (0x1D00-0x1D7F); 0x1D80-0x1DFF is Volta (GV100).
        if (hi == 0x1B00u || hi == 0x1C00u || (device_id >= 0x1D00u && device_id <= 0x1D7Fu)) return GpuArch::NvidiaPascal;
        if (device_name) {
            if (std::strstr(device_name, "RTX 50")) return GpuArch::NvidiaBlackwell;
            if (std::strstr(device_name, "RTX 40")) return GpuArch::NvidiaAda;
            if (std::strstr(device_name, "RTX 30")) return GpuArch::NvidiaAmpere;
            if (std::strstr(device_name, "RTX 20") || std::strstr(device_name, "GTX 16")) return GpuArch::NvidiaTuring;
            if (std::strstr(device_name, "GTX 10")) return GpuArch::NvidiaPascal;
        }
        return GpuArch::Unknown;
    }
    return GpuArch::Unknown;
}

inline bool is_rdna3(uint32_t vendor_id, uint32_t device_id, const char* device_name) {
    return vendor_id == 0x1002 && classify_gpu_arch(vendor_id, device_id, device_name) == GpuArch::AmdRdna3;
}

inline bool is_rdna2(uint32_t vendor_id, uint32_t device_id, const char* device_name) {
    return vendor_id == 0x1002 && classify_gpu_arch(vendor_id, device_id, device_name) == GpuArch::AmdRdna2;
}

inline bool is_rdna1(uint32_t vendor_id, uint32_t device_id, const char* device_name) {
    return vendor_id == 0x1002 && classify_gpu_arch(vendor_id, device_id, device_name) == GpuArch::AmdRdna1;
}

}  // namespace mkwvk
