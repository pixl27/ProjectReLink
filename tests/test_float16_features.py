"""The translator keeps 16-bit floats only where the device has them (gpu/vk/vk_shader.cpp float16_shaders, host_info), and the
program cache's stamp keeps programs made without them apart, changed only on such a device (disk_stamp).

A module that declares the Float16 capability on a device without shaderFloat16 is invalid
(VUID-VkShaderModuleCreateInfo-pCode-08740): RX_VK_NO_F16 was the only way to the lowering, and the device's feature never fed
it (GTX 10xx, Polaris RX 400/500). Without it the translator lowers the f16 math to 32 bits and the cooperative-matrix networks
are left out; the gameplay TAAU network's stand-in then draws its upscale (tests/test_upscale_fallback.py). RX_VK_NO_F16=1 still
forces the lowering, RX_VK_F16=1 forces f16 on. float16_shaders is compiled here with stubs; the rest are source checks. No game
or GPU required.

python -I -B tests/test_float16_features.py [--cxx clang++]
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]


def block(text, marker):
    """From marker to the brace closing its first one"""
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


PREFIX = r'''
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
using VkBool32 = unsigned;
constexpr VkBool32 VK_FALSE = 0, VK_TRUE = 1;
static struct { struct { VkBool32 shaderFloat16 = VK_TRUE; } vk12; } dev;
static const char *no_f16, *force_f16;
static const char* fixture_getenv(const char* name) {
    if (!strcmp(name, "RX_VK_NO_F16")) return no_f16;
    if (!strcmp(name, "RX_VK_F16")) return force_f16;
    return nullptr;
}
#define getenv fixture_getenv
// (env_on, the switches' rule (hookshot_env.h, included first), over the fixture's environment)
static bool fixture_env_on(const char* name) { const char* v = fixture_getenv(name); return v && !env_value_off(v); }
#define env_on fixture_env_on
static std::string logged;
static void logf(const char* f, ...) {
    char b[512];
    va_list a;
    va_start(a, f);
    vsnprintf(b, sizeof b, f, a);
    va_end(a);
    logged += b;
    logged += "\n";
}
'''

SUFFIX = r'''
int main(int argc, char** argv) {
    if (argc != 5) return 90;
    dev.vk12.shaderFloat16 = !strcmp(argv[1], "has") ? VK_TRUE : VK_FALSE;
    no_f16 = strcmp(argv[2], "-") ? argv[2] : nullptr;
    force_f16 = strcmp(argv[3], "-") ? argv[3] : nullptr;
    const bool want = !strcmp(argv[4], "on");
    const bool first = float16_shaders();
    dev.vk12.shaderFloat16 = !dev.vk12.shaderFloat16;   // (decided once: a second call gives the same answer, logs nothing)
    const bool second = float16_shaders();
    int lines = 0;
    for (char c : logged) lines += c == '\n';
    if (first != want || second != want || lines != 1 || logged.find("vk: shaders: 16-bit floats") != 0) {
        fprintf(stderr, "FAIL %s %s %s: %d %d, %d log lines: %s", argv[1], argv[2], argv[3], first, second, lines, logged.c_str());
        return 1;
    }
    printf("%s RX_VK_NO_F16=%s RX_VK_F16=%s: %s", argv[1], argv[2], argv[3], logged.c_str());
    return 0;
}
'''

CASES = [
    ("has", "-", "-", "on"),      # (the default on a card with f16 shaders: as before)
    ("lacks", "-", "-", "off"),   # (GTX 10xx, Polaris: lowered, no Float16 capability)
    ("has", "1", "-", "off"),     # (RX_VK_NO_F16=1: the lowering on any card, as before)
    ("lacks", "1", "-", "off"),
    ("has", "0", "-", "on"),      # (=0: not forced)
    ("lacks", "-", "1", "on"),    # (RX_VK_F16=1: f16 kept on a card without it, a test)
    ("lacks", "-", "0", "off"),
    ("lacks", "1", "1", "off"),   # (both: off wins)
]


def source_checks(src, draw):
    info = block(src, "Shader::HostTranslateInfo host_info()")
    if "h.support_float16 = float16_shaders();" not in info:
        raise SystemExit("FAIL: host_info's 16-bit floats do not follow float16_shaders")
    fn = block(src, "bool float16_shaders()")
    if "dev.vk12.shaderFloat16" not in fn:
        raise SystemExit("FAIL: float16_shaders does not read the device's shaderFloat16")
    for text, name in ((src, "vk_shader.cpp"), (draw, "vk_draw.cpp")):
        rest = text.replace(fn, "") if name == "vk_shader.cpp" else text
        if 'getenv("RX_VK_NO_F16")' in rest:
            raise SystemExit("FAIL: %s reads RX_VK_NO_F16 itself, apart from float16_shaders" % name)
    if "|| !float16_shaders())" not in src:
        raise SystemExit("FAIL: the networks are not left out without f16 shaders (their modules would declare Float16)")
    if "!float16_shaders()" not in block(draw, "bool taau_key(uint64_t key) {"):
        raise SystemExit("FAIL: the TAAU stand-in's key does not follow float16_shaders")
    print("host_info, the networks and the TAAU key follow float16_shaders")
    stamp = block(src, "uint64_t translator_stamp()")
    marker = "if (!dev.vk12.shaderFloat16) {"
    if marker not in stamp:
        raise SystemExit("FAIL: the program cache's stamp does not tell programs made without 16-bit floats apart")
    added = block(stamp[stamp.index(marker):], marker)
    if "add(&no_f16, sizeof no_f16);" not in added or "float16_shaders()" not in added or "add(&half, sizeof half);" not in added:
        raise SystemExit("FAIL: the stamp's 16-bit float word is not added inside its condition, with the choice made")
    if "half" in stamp.replace(added, ""):
        raise SystemExit("FAIL: the 16-bit float word is added outside its condition: every GPU's stamp would change")
    print("stamp: changed only on a device without shaderFloat16")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="clang++")
    args = p.parse_args()
    src = (ROOT / "gpu" / "vk" / "vk_shader.cpp").read_text(encoding="utf-8")
    draw = (ROOT / "gpu" / "vk" / "vk_draw.cpp").read_text(encoding="utf-8")
    source_checks(src, draw)
    cpp = PREFIX + block(src, "bool float16_shaders()") + "\n" + SUFFIX
    with tempfile.TemporaryDirectory(prefix="float16-") as tmp:
        tmp = Path(tmp)
        (tmp / "check.cpp").write_text(cpp, encoding="utf-8")
        exe = tmp / ("check.exe" if os.name == "nt" else "check")
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Werror", "-include", str(ROOT / "gpu" / "vk" / "compat" / "hookshot_env.h"),
                        str(tmp / "check.cpp"), "-o", str(exe)], check=True)
        for case in CASES:
            subprocess.run([str(exe), *case], check=True)
    print("PASS")


if __name__ == "__main__":
    main()
