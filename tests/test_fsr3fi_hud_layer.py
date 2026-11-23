"""AMD FSR 3.1 frame generation lays the HUD over its made frames by how see-through it is (#1565): no halo of the real
frame's background around the HUD.

The HUD's glows, shadows, faint icons and the minimap let the scene through: the picture as presented is D = B + T * S, the
HUD's colour B over the scene S by its see-through amount T. FSR 3.1's HUD-less mode laid the presented pixel over each made
frame whole wherever the HUD is, so the real frame's scene came with it, a halo around every HUD element against the made
frame's moved scene. Now shaders/fg_hud_layer.comp learns T from frame to frame (hud_fit: the HUD stays where it is while
the scene behind it moves, D1 - D0 = T * (S1 - S0)), and shaders/fg_hud_composite.comp lays B + T * the made frame's own
scene (hud_over); where T is unknown (the HUD just appeared or changed, no movement behind it yet), the presented pixel as
before. Both functions, compiled as C++, are run over pictures of a HUD pixel through frames:
  - a see-through HUD over a moving scene: T and B learned; the made pixel is the HUD over the made frame's scene, where the
    presented pixel (the old way) is off by the halo;
  - an opaque HUD, an additive glow: T 0 and 1, the made pixel as the truth;
  - the picture's precision (B10G11R11's 6- and 5-bit mantissas): no false "the HUD changed";
  - the HUD changing (a count, the hearts beating): unknown at once, the presented pixel;
  - a HUD that just appeared, no frame before, nothing moving behind it yet: unknown;
  - the camera still: what was learned is kept;
  - no HUD: the made frame untouched; the native HDR picture (pq): the HUD over it as it is;
  - a slow pan (changes behind the HUD near the picture's rounding) and then a turn: no T from the rounding;
  - a scene brighter than the frame's 1 behind the HUD: the presented pixel (the frame clamps it, the made frame too);
  - a fit that is not one (the image's memory before it was written, NaN): started again.
And the headers: fg_hud_layer_comp_spv.h and fg_hud_composite_comp_spv.h name the hash of the shader each was made from
(first line), and with glslangValidator are its output byte for byte but glslang's version line. --write makes them.

python -I -B tests/test_fsr3fi_hud_layer.py [--cxx g++] [--glslang glslangValidator] [--write]
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
PAIRS = [(SHADERS / "fg_hud_layer.comp", SHADERS / "fg_hud_layer_comp_spv.h", "FG_HUD_LAYER_COMP_SPV"),
         (SHADERS / "fg_hud_composite.comp", SHADERS / "fg_hud_composite_comp_spv.h", "FG_HUD_COMPOSITE_COMP_SPV")]


def source_line(comp, var):
    """the header's first line: the shader's hash (line ends as LF, as the checkout has them) and how the header is made"""
    digest = hashlib.sha256(comp.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return ("// %s sha256 %s (tests/test_fsr3fi_hud_layer.py --write: glslangValidator -V --target-env vulkan1.3 --vn %s)\n"
            % (comp.name, digest, var)).encode()


def compile_header(glslang, comp, var, out):
    r = subprocess.run([glslang, "-V", "--target-env", "vulkan1.3", "--vn", var, "-o", str(out), str(comp)],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("%s does not compile:\n%s%s" % (comp.name, r.stdout, r.stderr))
    return out.read_bytes()


def without_version(header):
    """glslang's output without its first line (its own version: two glslang builds differ there alone)"""
    return header.partition(b"\n")[2]


# (just enough of GLSL for hud_fit and hud_over)
SHIM = r'''
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
struct vec3 {
    float r, g, b;
    vec3() : r(0), g(0), b(0) {}
    explicit vec3(float v) : r(v), g(v), b(v) {}
    vec3(float x, float y, float z) : r(x), g(y), b(z) {}
};
struct vec4 {
    float x, y, z, w;
    vec4() : x(0), y(0), z(0), w(0) {}
    explicit vec4(float v) : x(v), y(v), z(v), w(v) {}
    vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    vec4(vec3 v, float d) : x(v.r), y(v.g), z(v.b), w(d) {}
    vec3 rgb() const { return {x, y, z}; }
};
struct bvec3 { bool r, g, b; };
static vec3 operator+(vec3 a, vec3 b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }
static vec3 operator-(vec3 a, vec3 b) { return {a.r - b.r, a.g - b.g, a.b - b.b}; }
static vec3 operator*(float s, vec3 a) { return {s * a.r, s * a.g, s * a.b}; }
static vec3 operator+(vec3 a, float s) { return {a.r + s, a.g + s, a.b + s}; }
static bool operator==(vec3 a, vec3 b) { return a.r == b.r && a.g == b.g && a.b == b.b; }
static bool operator==(vec4 a, vec4 b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
static float dot(vec3 a, vec3 b) { return a.r * b.r + a.g * b.g + a.b * b.b; }
static vec3 abs(vec3 a) { return {std::fabs(a.r), std::fabs(a.g), std::fabs(a.b)}; }
static float clamp(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
static vec3 clamp(vec3 v, float lo, float hi) { return {clamp(v.r, lo, hi), clamp(v.g, lo, hi), clamp(v.b, lo, hi)}; }
static float min(float a, float b) { return a < b ? a : b; }
static bvec3 lessThanEqual(vec3 a, vec3 b) { return {a.r <= b.r, a.g <= b.g, a.b <= b.b}; }
static bool all(bvec3 v) { return v.r && v.g && v.b; }
static bool any(bvec3 v) { return v.r || v.g || v.b; }
static bvec3 greaterThan(vec3 a, vec3 b) { return {a.r > b.r, a.g > b.g, a.b > b.b}; }
static vec3 max(vec3 a, vec3 b) { return {a.r > b.r ? a.r : b.r, a.g > b.g ? a.g : b.g, a.b > b.b ? a.b : b.b}; }
using std::pow;
'''

FIXTURE = r'''
static int bad = 0;
static void need(bool ok, const char* what, double a, double b) {
    if (!ok) bad++, std::printf("FAIL: %s (%.6g against %.6g)\n", what, a, b);
}
// (a value as the B10G11R11 picture keeps it: its mantissa's bits, 6 for red and green, 5 for blue)
static float packed(float v, int bits) {
    if (v <= 0) return 0;
    int e;
    const float m = std::frexp(v, &e);
    return std::ldexp(std::round(m * (1 << (bits + 1))) / (1 << (bits + 1)), e);
}
static vec3 picture(vec3 v) { return {packed(v.r, 6), packed(v.g, 6), packed(v.b, 5)}; }
static float byte(float v) { return std::round(clamp(v, 0, 1) * 255.0f) / 255.0f; }
static vec3 frame(vec3 linear_value) { vec3 e = srgb(linear_value); return {byte(e.r), byte(e.g), byte(e.b)}; }
static float far(vec3 a, vec3 b) { return std::fmax(std::fabs(a.r - b.r), std::fmax(std::fabs(a.g - b.g), std::fabs(a.b - b.b))); }

// The scene behind the pixel at frame i, moving (a camera turn: a different texel each frame)
static vec3 scene_at(int i) {
    return {0.15f + 0.6f * (0.5f + 0.5f * std::sin(i * 1.3f)), 0.2f + 0.5f * (0.5f + 0.5f * std::sin(i * 0.7f + 1.0f)),
            0.3f + 0.4f * (0.5f + 0.5f * std::cos(i * 1.1f))};
}

struct Run {
    vec4 fit, layer;
    bool have = false;
    vec3 s0, d0;
    // (one frame: the picture before the HUD and as presented, as the game blends B over S by T)
    void frame_(vec3 s, vec3 b, float t, bool hud = true) {
        const vec3 s1 = picture(s), d1 = hud ? picture(b + t * s) : s1;
        fit = hud_fit(s1, d1, s0, d0, have, have ? fit : vec4(0.0f), layer);
        s0 = s1, d0 = d1, have = true;
    }
};

// What the made frame shows at a HUD pixel: its own scene m (the made frame's, moved) with this frame's HUD over it
static vec3 made_pixel(const Run& r, vec3 m, vec3 shown) { return hud_over(frame(m), shown, r.layer, false); }

int main() {
    // A see-through HUD (the hearts' glow, the item box's shadow): B over S by T, the scene moving behind it
    const vec3 glow(0.30f, 0.28f, 0.22f);
    const float glow_t = 0.55f;
    Run r;
    for (int i = 0; i < 6; i++) r.frame_(scene_at(i), glow, glow_t);
    need(r.layer.w >= 0, "a see-through HUD over a moving scene: T known after a few frames", r.layer.w, glow_t);
    need(std::fabs(r.layer.w - glow_t) < 0.03f, "T as the game blended it", r.layer.w, glow_t);
    need(far(r.layer.rgb(), glow) < 0.03f, "B, the HUD's own colour", r.layer.x, glow.r);
    // The made frame between this frame and the last: its scene is neither's (the camera between them)
    const vec3 m(0.05f, 0.65f, 0.12f), s_now = scene_at(5);
    const vec3 truth = frame(glow + glow_t * m), shown = frame(glow + glow_t * s_now);
    const vec3 got = made_pixel(r, m, shown);
    need(far(got, truth) <= 3.0f / 255.0f, "the made pixel: the HUD over the made frame's own scene", far(got, truth), 0);
    // (the old way, the presented pixel whole: the real frame's scene through the HUD, the halo)
    need(far(shown, truth) > 40.0f / 255.0f, "(the fixture's halo is there to see: the presented pixel is off)",
         far(shown, truth), 0);

    // An opaque HUD (an icon's body): T 0, the made pixel the HUD's
    Run o;
    const vec3 icon(0.8f, 0.4f, 0.1f);
    for (int i = 0; i < 6; i++) o.frame_(scene_at(i), icon, 0.0f);
    need(o.layer.w >= 0 && o.layer.w < 0.02f, "an opaque HUD: T 0", o.layer.w, 0);
    need(far(made_pixel(o, m, frame(icon)), frame(icon)) <= 2.0f / 255.0f, "an opaque HUD: its own colour", 0, 0);
    // An additive glow (the fairy's): T 1
    Run a;
    for (int i = 0; i < 6; i++) a.frame_(scene_at(i), vec3(0.1f, 0.1f, 0.15f), 1.0f);
    need(a.layer.w > 0.97f, "an additive glow: T 1", a.layer.w, 1);
    need(far(made_pixel(a, m, shown), frame(vec3(0.1f, 0.1f, 0.15f) + m)) <= 3.0f / 255.0f, "an additive glow over the made "
         "frame's scene", 0, 0);

    // Through many frames at the picture's precision: never a false "the HUD changed". (Now and then the faint HUD over a
    // scene comes out as the scene's own value in the picture: no HUD that frame, and the next frame has no pair to check;
    // what was learned is kept through it)
    Run p;
    int unknown = 0, restarts = 0;
    for (int i = 0; i < 400; i++) {
        const float before = p.fit.z;
        p.frame_(scene_at(i), glow, glow_t);
        if (i >= 3 && p.layer.w < 0) unknown++;
        if (i >= 3 && p.fit.z < before) restarts++;
    }
    need(restarts == 0, "400 frames of a still HUD at B10G11R11's precision: never started again", restarts, 0);
    need(unknown <= 4, "and unknown only where the picture cannot tell it from the scene (a frame or two)", unknown, 4);

    // The HUD changes (a count goes up, the hearts beat): unknown at once, the presented pixel
    p.frame_(scene_at(400), vec3(0.9f, 0.1f, 0.1f), 0.2f);
    need(p.layer.w < 0, "the HUD changed: unknown", p.layer.w, -1);
    need(far(hud_over(frame(m), shown, p.layer, false), shown) == 0, "unknown: the presented pixel, as before", 0, 0);
    for (int i = 401; i < 406; i++) p.frame_(scene_at(i), vec3(0.9f, 0.1f, 0.1f), 0.2f);
    need(std::fabs(p.layer.w - 0.2f) < 0.03f, "and learned again", p.layer.w, 0.2);

    // A HUD that just appeared (no HUD there the frame before), or no frame before: unknown
    Run n;
    n.frame_(scene_at(0), glow, glow_t, false);
    n.frame_(scene_at(1), glow, glow_t);
    need(n.layer.w < 0, "a HUD that just appeared: unknown", n.layer.w, -1);
    Run first;
    first.frame_(scene_at(0), glow, glow_t);
    need(first.layer.w < 0, "the first frame: unknown", first.layer.w, -1);
    // Nothing moving behind it yet: unknown
    Run still;
    for (int i = 0; i < 20; i++) still.frame_(scene_at(0), glow, glow_t);
    need(still.layer.w < 0, "a scene that never moved: unknown", still.layer.w, -1);
    // The camera stops: what was learned is kept, however long
    for (int i = 0; i < 300; i++) r.frame_(scene_at(5), glow, glow_t);
    need(std::fabs(r.layer.w - glow_t) < 0.03f, "the camera still: T kept", r.layer.w, glow_t);

    // No HUD: no layer to lay (the made frame untouched: fg_hud_composite.comp returns at (0, 0, 0, 1))
    Run none;
    for (int i = 0; i < 4; i++) none.frame_(scene_at(i), glow, 0.0f, false);
    need(none.layer == vec4(0.0f, 0.0f, 0.0f, 1.0f) && none.fit == vec4(0.0f), "no HUD: nothing laid, nothing kept", 0, 0);

    // A slow pan: the scene behind the HUD moves by about the picture's rounding a frame, for a long while, then the camera
    // turns. Whenever T is used, it is the HUD's to within the turn's needs (the rounding is not taken for T)
    Run slow;
    int wrong = 0;
    for (int i = 0; i < 120; i++) {
        const float k = 0.0015f * i;
        slow.frame_(vec3(0.30f + k, 0.32f + k * 0.5f, 0.35f - k * 0.3f), glow, glow_t);
        if (slow.layer.w >= 0 && std::fabs(slow.layer.w - glow_t) > 0.04f) wrong++;
    }
    for (int i = 0; i < 6; i++) {
        slow.frame_(scene_at(i), glow, glow_t);
        if (slow.layer.w >= 0 && std::fabs(slow.layer.w - glow_t) > 0.04f) wrong++;
    }
    need(wrong == 0, "a slow pan, then a turn: T, when used, the HUD's", wrong, 0);
    need(slow.layer.w >= 0, "and known after the turn", slow.layer.w, glow_t);

    // A scene brighter than the frame's 1 behind the HUD (the sky, a bloom): the presented pixel, the made frame's scene
    // being clamped at 1 as the frame is
    Run bright;
    for (int i = 0; i < 6; i++) bright.frame_(vec3(1.2f + 0.3f * (i % 2), 0.5f + 0.1f * i, 0.3f), glow, glow_t);
    need(bright.layer.w < 0, "a scene over 1 behind the HUD: the presented pixel", bright.layer.w, -1);

    // A fit that is not one (NaN, the image's memory never written): started again, learned as from nothing
    Run junk;
    junk.frame_(scene_at(0), glow, glow_t);
    junk.fit = vec4(std::nanf(""), std::nanf(""), 7.0f, 0.0f);
    for (int i = 1; i < 6; i++) junk.frame_(scene_at(i), glow, glow_t);
    need(std::fabs(junk.layer.w - glow_t) < 0.03f, "a fit that was NaN: learned again", junk.layer.w, glow_t);

    // The native HDR picture (pq): encoded already, the HUD over it as it is
    const vec4 l(0.2f, 0.1f, 0.0f, 0.5f);
    const vec3 hdr = hud_over(vec3(0.6f, 0.4f, 0.2f), vec3(0, 0, 0), l, true);
    need(far(hdr, vec3(0.5f, 0.3f, 0.1f)) < 1e-6f, "native HDR: B + T * the made pixel as it is", hdr.r, 0.5);
    // (the sRGB pair round-trips)
    for (float v : {0.0f, 0.002f, 0.04045f, 0.2f, 0.5f, 0.9f, 1.0f})
        need(std::fabs(srgb1(linear1(v)) - v) < 1e-5f, "sRGB decode and encode undo each other", srgb1(linear1(v)), v);
    if (bad) return 1;
    std::printf("hud layer: T and B learned from the frames, the HUD over the made frame's own scene, unknown -> presented\n");
    return 0;
}
'''


def cut(text, sig):
    """a function cut from a shader, as C++: out parameters by reference, .rgb and .a of a vec4 as the shim has them"""
    return "\nstatic " + block(text, sig).replace("out vec4 ", "vec4& ").replace(".rgb", ".rgb()").replace("l.a", "l.w")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    p.add_argument("--glslang", default="glslangValidator")
    p.add_argument("--write", action="store_true", help="make the two headers from their shaders")
    a = p.parse_args()
    glslang = shutil.which(a.glslang)
    if a.write:
        if not glslang:
            raise SystemExit("--write needs %s" % a.glslang)
        with tempfile.TemporaryDirectory(prefix="fg-hud-layer-") as tmp:
            for comp, header, var in PAIRS:
                header.write_bytes(source_line(comp, var) + compile_header(glslang, comp, var, Path(tmp) / "out.h"))
                print("wrote %s" % header)
        return
    for comp, header, var in PAIRS:
        if not comp.exists() or not header.exists():
            raise SystemExit("%s or its header is missing: the HUD is laid over the made frames whole, halo and all "
                             "(#1565)" % comp.name)
        first = header.read_bytes().partition(b"\n")[0]
        if first + b"\n" != source_line(comp, var):
            raise SystemExit("%s was not made from this %s (its first line names another hash or command): "
                             "python -I -B tests/test_fsr3fi_hud_layer.py --write" % (header.name, comp.name))
    layer = PAIRS[0][0].read_text(encoding="utf-8")
    over = PAIRS[1][0].read_text(encoding="utf-8")
    if "hud_fit(s1, d1, s0, d0, prev, prev ? imageLoad(fit, g) : vec4(0.0), l);" not in block(layer, "void main()"):
        raise SystemExit("fg_hud_layer.comp: main does not run hud_fit on the two frames' pictures")
    # (main's shortcut for the pixels with no HUD, where hud_fit is not run: with no frame before their fit starts again
    # too, or a pixel the HUD reaches later reads the image's memory as it was made)
    if "if (pc.prev == 0) imageStore(fit, g, vec4(0.0));" not in block(layer, "void main()"):
        raise SystemExit("fg_hud_layer.comp: main leaves the fit of a pixel with no HUD as it was when there is no frame "
                         "before")
    if "hud_over(m, texelFetch(shown, g, 0).rgb, l, pc.pq != 0)" not in block(over, "void main()"):
        raise SystemExit("fg_hud_composite.comp: main does not lay the HUD with hud_over")
    fixture = SHIM + "".join(cut(layer, f) for f in ("float see_through(", "bool known(", "bool explains(",
                                                      "vec4 hud_fit(")) \
        + "".join(cut(over, f) for f in ("float srgb1(", "vec3 srgb(", "float linear1(", "vec3 linear(", "vec3 hud_over(")) \
        + FIXTURE
    consts = "".join(line + "\n" for line in layer.splitlines() if line.startswith("const float k"))
    fixture = fixture.replace("using std::pow;\n", "using std::pow;\n" + "static " + consts.replace("\nconst", "\nstatic const"))
    with tempfile.TemporaryDirectory(prefix="fg-hud-layer-") as tmp:
        cpp, exe = Path(tmp) / "fixture.cpp", Path(tmp) / "fixture"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function", str(cpp), "-o",
                        str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
        for comp, header, var in PAIRS:
            if glslang:
                made = compile_header(glslang, comp, var, Path(tmp) / "out.h")
                if without_version(made) != without_version(header.read_bytes().partition(b"\n")[2]):
                    raise SystemExit("%s is not glslang's output for %s: python -I -B tests/test_fsr3fi_hud_layer.py --write"
                                     % (header.name, comp.name))
                print("%s: glslang's output for this %s" % (header.name, comp.name))
            else:
                print("%s: names this %s (no %s here to compile it)" % (header.name, comp.name, a.glslang))


if __name__ == "__main__":
    main()
