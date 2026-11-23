"""AMD FSR 3.1 frame generation's HUD-less colour (gpu/vk/shaders/fg_hud.comp hudless_colour, vk_framegen.cpp hud_source):
the presented frame's own pixel, to the bit, wherever nothing was drawn over the picture after the upscale, and the copy of
the picture before the HUD only under the HUD, sRGB-encoded as the frame is (native HDR: as it is, clamped).

FSR 3.1's HUD-less mode lays the presented frame over each made frame wherever the two differ at all. The copy is in linear
light and the frame is presented sRGB-encoded, so given the copy whole (as main did) they differed everywhere and every made
frame came out as the game's frame: at 4x each game frame was shown four times. Left linear under the HUD, it put a dark ring
around the HUD's see-through parts. The shader's own hudless_colour, compiled as C++, is checked for both; the inpainting
pass takes the presented pixel whole where they differ (scripts/fsr3_shaders.py FI_PATCHES, its own check:
test_fsr3_shaders.py); hud_source tells the shader whether the picture is the native HDR one.

fg_hud_comp_spv.h names the hash of the fg_hud.comp it was made from (its first line), checked with or without glslang; with
glslangValidator the rest of the header is its output byte for byte, but for glslang's own version line. --write makes it.

python -I -B tests/test_fsr3fi_hudless_colour.py [--cxx g++] [--glslang glslangValidator] [--write]
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--glslang", "{glslang}"]

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / "gpu" / "vk" / "shaders"
COMP = SHADERS / "fg_hud.comp"
HEADER = SHADERS / "fg_hud_comp_spv.h"
VAR = "FG_HUD_COMP_SPV"


def source_line():
    """the header's first line: the shader's hash (line ends as LF, as the checkout has them) and how the header is made"""
    digest = hashlib.sha256(COMP.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return ("// fg_hud.comp sha256 %s (tests/test_fsr3fi_hudless_colour.py --write: glslangValidator -V --target-env "
            "vulkan1.3 --vn %s)\n" % (digest, VAR)).encode()


def compile_header(glslang, out):
    r = subprocess.run([glslang, "-V", "--target-env", "vulkan1.3", "--vn", VAR, "-o", str(out), str(COMP)],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("fg_hud.comp does not compile:\n" + r.stdout + r.stderr)
    return out.read_bytes()


def without_version(header):
    """glslang's output without its first line (its own version: two glslang builds differ there alone)"""
    return header.partition(b"\n")[2]


# (just enough of GLSL for hudless_colour)
SHIM = r'''
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct vec3 {
    float r, g, b;
    vec3() : r(0), g(0), b(0) {}
    vec3(float x, float y, float z) : r(x), g(y), b(z) {}
};
static bool operator==(vec3 a, vec3 b) { return a.r == b.r && a.g == b.g && a.b == b.b; }
static bool operator!=(vec3 a, vec3 b) { return !(a == b); }
static float clamp(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
static vec3 clamp(vec3 v, float lo, float hi) { return {clamp(v.r, lo, hi), clamp(v.g, lo, hi), clamp(v.b, lo, hi)}; }
using std::pow;
'''

FIXTURE = r'''
static int bad = 0;
static void need(bool ok, const char* what, double a, double b) {
    if (!ok) bad++, std::printf("FAIL: %s (%.9g against %.9g)\n", what, a, b);
}
static double ref(double v) {   // (the sRGB encode, IEC 61966-2-1)
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    return v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
}
static bool same_bits(vec3 a, vec3 b) { return std::memcmp(&a, &b, sizeof a) == 0; }
int main() {
    // (values over the B10G11R11 picture's range and the RGBA8 frame's)
    const float vals[] = {0.0f, 0.001f, 0.0031308f, 0.004f, 0.018f, 0.1f, 0.2140f, 0.5f, 0.7354f, 0.99f, 1.0f, 1.5f, 8.0f};
    for (float s : vals)
        for (float f : vals)
            for (int pq = 0; pq < 2; pq++) {
                const vec3 scene(s, s * 0.5f, s * 0.25f), shown(f, f * 0.75f, 1.0f - (f > 1 ? 1 : f));
                // nothing drawn over the picture here: the frame's own pixel, to the bit (main gave the linear copy, which
                // FSR 3.1 found different from the frame almost everywhere: every made frame the game's)
                const vec3 out = hudless_colour(scene, scene, shown, pq != 0);
                need(same_bits(out, shown), "no HUD: the presented frame's own pixel", out.r, shown.r);
                // the HUD over it: the copy, sRGB-encoded (native HDR: as it is), clamped as the frame is
                const vec3 drawn(s + 0.25f, s, s);
                const vec3 hud = hudless_colour(scene, drawn, shown, pq != 0);
                const float in[3] = {scene.r, scene.g, scene.b}, got[3] = {hud.r, hud.g, hud.b};
                for (int c = 0; c < 3; c++) {
                    const double want = pq ? clamp(in[c], 0.0f, 1.0f) : ref(in[c]);
                    need(std::fabs(got[c] - want) <= 2e-6, pq ? "under the HUD, native HDR: the copy, clamped"
                                                              : "under the HUD: the copy sRGB-encoded", got[c], want);
                }
            }
    // (the curve's own points: linear 0.5 is sRGB 0.7354, 0.2140 is about 0.5; its two pieces meet)
    need(std::fabs(srgb1(0.5f) - 0.735357f) < 1e-5, "sRGB of linear 0.5", srgb1(0.5f), 0.735357);
    need(std::fabs(srgb1(0.214041f) - 0.5f) < 1e-5, "sRGB of linear 0.214", srgb1(0.214041f), 0.5);
    need(std::fabs(srgb1(0.0031307f) - srgb1(0.0031309f)) < 1e-5, "the two pieces meet", srgb1(0.0031307f),
         srgb1(0.0031309f));
    need(srgb1(-1.0f) == 0.0f && srgb1(4.0f) == 1.0f, "clamped to the frame's 0..1", srgb1(-1.0f), srgb1(4.0f));
    if (bad) return 1;
    std::printf("hudless_colour: the presented frame's pixel where no HUD is, the copy sRGB-encoded under it\n");
    return 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    p.add_argument("--glslang", default="glslangValidator")
    p.add_argument("--write", action="store_true", help="make fg_hud_comp_spv.h from fg_hud.comp")
    a = p.parse_args()
    glslang = shutil.which(a.glslang)
    if a.write:
        if not glslang:
            raise SystemExit("--write needs %s" % a.glslang)
        with tempfile.TemporaryDirectory(prefix="fg-hud-") as tmp:
            HEADER.write_bytes(source_line() + compile_header(glslang, Path(tmp) / "out.h"))
        print("wrote %s" % HEADER)
        return
    first, _, rest = HEADER.read_bytes().partition(b"\n")
    if first + b"\n" != source_line():
        raise SystemExit("fg_hud_comp_spv.h was not made from this fg_hud.comp (its first line names another hash or "
                         "command): python -I -B tests/test_fsr3fi_hudless_colour.py --write")
    text = COMP.read_text(encoding="utf-8")
    body = block(text, "void main()")
    if "imageStore(hudless, g, vec4(hudless_colour(scene, drawn, shown, pc.pq != 0), 1.0));" not in body:
        raise SystemExit("fg_hud.comp: the HUD-less colour is not hudless_colour's (scene, drawn, shown, pc.pq)")
    if "imageStore(dst, g, vec4(shown, scene != drawn ? 0.0 : 1.0));" not in body:
        raise SystemExit("fg_hud.comp: the HUD mark is not where the picture before and after the HUD differ")
    if "layout(push_constant) uniform P { ivec2 offset; ivec2 size; int pq; } pc;" not in text:
        raise SystemExit("fg_hud.comp: the push constants are not the crop and pq (vk_framegen.cpp hud_source's 20 bytes)")
    framegen = (ROOT / "gpu" / "vk" / "vk_framegen.cpp").read_text(encoding="utf-8")
    source = block(framegen, "bool hud_source(")
    if "BUILTIN_SPV_ARGS(FG_HUD_COMP_SPV), {S, S, S, I, I}, 20)" not in source or \
            "const int32_t pc[5] = {x0, y0, (int32_t)w, (int32_t)h, pq ? 1 : 0};" not in source:
        raise SystemExit("vk_framegen.cpp hud_source: the HUD pass is not given the crop and pq (20 bytes)")
    fixture = SHIM + "".join("\nstatic " + block(text, f) for f in ("float srgb1(", "vec3 srgb(", "vec3 hudless_colour(")) \
        + FIXTURE
    with tempfile.TemporaryDirectory(prefix="fg-hud-") as tmp:
        cpp, exe = Path(tmp) / "fixture.cpp", Path(tmp) / "fixture"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Wno-unused-function", str(cpp), "-o", str(exe)],
                       check=True)
        subprocess.run([str(exe)], check=True)
        if glslang:
            made = compile_header(glslang, Path(tmp) / "out.h")
            if without_version(made) != without_version(rest):
                raise SystemExit("fg_hud_comp_spv.h is not glslang's output for fg_hud.comp: "
                                 "python -I -B tests/test_fsr3fi_hudless_colour.py --write")
            print("fg_hud_comp_spv.h: glslang's output for this fg_hud.comp")
        else:
            print("fg_hud_comp_spv.h: names this fg_hud.comp (no %s here to compile it)" % a.glslang)


if __name__ == "__main__":
    main()
