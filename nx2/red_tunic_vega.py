"""Builder defaults for the Windows Radeon Vega 1002:15E7 profile."""
import os
from pathlib import Path
import subprocess
import tempfile

ENV = {
    "RX_VK_READBACK_LATE": "40,2000,3980",
    "RX_VK_TEX_AHEAD": "1", "RX_VK_TEX_AHEAD_MS": "4", "RX_VK_QUICK_MS": "4",
    "HOS_AV1_ASYNC": "1", "HOS_DAV1D_THREADS": "4", "HOS_RESERVE_CORE": "1",
    "RX_VK_SHARPEN_FP16": "1", "RX_VK_ASTC_CPU": "1", "RX_VK_ASYNC_LOAD": "0",
    "RX_VK_HASH_MEMO": "1", "RX_VK_ASYNC_IMPORT": "1",
    "RX_VK_VEGA_ASTC_GPU": "0", "RX_VK_VEGA_ASTC_RETAIN": "1",
    "RX_VK_MENU_FADE": "1", "RX_VK_MIP_STREAM": "0",
    "RX_VK_EXPOSURE_METER_PART": "1", "RX_VK_MESH_CONVERT_SCOPED": "1",
    "RX_VK_CS_MODULE_DROP": "1",
}

GRAPHICS = {
    "quality_preset": "0", "ultrawide": "1", "output_res": "1280x720",
    "handheld_profile": "1", "dlss": "6", "dlss_model": "0", "dlss_pre_bloom": "0",
    "resolution": "3", "sharpness": "20", "frame_generation": "0",
    "frame_generation_by": "0", "frame_rate_cap": "0",
    "nvidia_reflex": "0", "amd_anti-lag_2": "0", "fullscreen": "0", "vsync": "0",
    "upscaling": "0", "crt_filter": "0", "crt_lines": "0", "ultrawide_menus": "1",
    "hud_layout": "0", "hdr": "0", "hdr_paper_white": "200",
    "hdr_peak_brightness": "1000", "hdr_contrast": "100", "hdr_tone_mapper": "0",
    "texture_detail": "3", "anisotropic_filtering": "4", "shadow_resolution": "1",
    "foliage_distance": "25", "object_detail": "100", "grass_density": "25",
    "shadow_distance": "25", "contact_shadows": "1", "base_quality": "1",
    "frame_pacing": "1", "threaded_rendering": "2", "descriptor_reuse": "1",
    "sampler_cache": "0", "robust_buffers": "0", "present_queue": "0",
    "astc_as_rgba8": "0", "memory_budget": "1",
}


def detected():
    """Use the preset automatically only with one unambiguous Windows adapter."""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_VideoController | ForEach-Object { $_.PNPDeviceID }"],
            capture_output=True, text=True, errors="replace", check=True, timeout=12,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    adapters = result.stdout.strip().upper().splitlines()
    return len(adapters) == 1 and adapters[0].startswith("PCI\\VEN_1002&DEV_15E7&")


def add_defaults(path, defaults):
    try:
        before = path.read_bytes()
    except FileNotFoundError:
        before = b""
    keys = {line.partition(b"=")[0].strip() for line in before.splitlines() if b"=" in line}
    missing = [f"{key}={value}".encode("ascii") for key, value in defaults.items()
               if key.encode("ascii") not in keys]
    if not missing:
        return
    newline = b"\r\n" if b"\r\n" in before else b"\n"
    if before and not before.endswith(b"\n"):
        before += newline
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
    try:
        pending.write_bytes(before + newline.join(missing) + newline)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def apply(out):
    folder = Path(out)
    add_defaults(folder / "hookshot_env.txt", ENV)
    add_defaults(folder / "hookshot_options.ini", GRAPHICS)
