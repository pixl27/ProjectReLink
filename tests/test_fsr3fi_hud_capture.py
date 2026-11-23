"""Exercise FSR 3.1 frame generation's gameplay HUD capture selector without Vulkan.

Run python -I -B tests/test_fsr3fi_hud_capture.py --cxx clang++.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srccut import block  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]

PREFIX = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using VkFormat = unsigned;
constexpr VkFormat VK_FORMAT_UNDEFINED = 0, SCENE = 1, MOVIE = 2;
constexpr unsigned VK_IMAGE_VIEW_TYPE_2D = 1;
enum class ViewUse { Sample, Storage };
struct Texture { uint32_t width, height; VkFormat format; };
struct Descriptor { unsigned handle; VkFormat format; };
struct StageInfo {
    bool uses_cooperative_matrix;
    std::vector<Descriptor> texture_descriptors, image_descriptors;
};
struct Stage { StageInfo info; };
struct Program { uint64_t key; std::vector<Stage> stages; };
struct NvnState {};
namespace fsr3fi {
bool enabled = true;
bool frames_wanted() { return enabled; }
}

Texture in_tex{1920, 1080, SCENE}, out_tex{1920, 1080, SCENE};
uint32_t output_level = 0;

uint32_t handle_of(const NvnState&, const Stage&, const Descriptor& descriptor, int) { return descriptor.handle; }
VkFormat image_format(unsigned) { return SCENE; }
bool temporal_colour_pair(VkFormat in, VkFormat out) { return in == SCENE && out == SCENE; }

void texture_view_of(uint32_t handle, unsigned, ViewUse use, VkFormat, Texture** texture, uint32_t* level = nullptr) {
    if (use == ViewUse::Sample && handle == 1) *texture = &in_tex;
    if (use == ViewUse::Storage && handle == 2) {
        *texture = &out_tex;
        *level = output_level;
    }
}
'''

CASES = r'''
void expect(const char* what, const Program& program, Texture* wanted) {
    uint32_t level = 0;
    if (upscale_picture(NvnState{}, program, &level) != wanted || (wanted && level != output_level)) {
        std::fprintf(stderr, "%s\n", what);
        std::exit(1);
    }
}

int main() {
    const Program full{0x9dc98871eb7fc101ull, {{StageInfo{true, {{1, SCENE}}, {{2, SCENE}}}}}};
    const Program inferred{0x123ull, {{StageInfo{true, {{1, SCENE}}, {{2, SCENE}}}}}};
    expect("The known 100% gameplay dispatch did not supply a pre-HUD picture", full, &out_tex);
    expect("An inferred equal-size dispatch supplied a pre-HUD picture", inferred, nullptr);
    out_tex.height = 1079;
    expect("Equal widths with different heights were treated as native resolution", full, nullptr);
    out_tex = {1919, 1080, SCENE};
    expect("A smaller output supplied a pre-HUD picture", full, nullptr);
    out_tex = {3840, 2160, SCENE};
    output_level = 1;
    expect("The selected output mip's equal dimensions were ignored", full, &out_tex);
    output_level = 0;
    in_tex = {1440, 810, SCENE};
    out_tex = {1920, 1080, SCENE};
    expect("A scaled gameplay dispatch lost its pre-HUD picture", inferred, &out_tex);
    out_tex.format = MOVIE;
    expect("A non-scene image supplied a pre-HUD picture", full, nullptr);
    out_tex.format = SCENE;
    fsr3fi::enabled = false;
    expect("HUD capture ran with FSR frame generation inactive", full, nullptr);
    std::puts("PASS: FSR 3.1 HUD capture selects the native gameplay picture and keeps its other guards");
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    text = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    functions = [block(text, marker) for marker in ("bool gameplay_network(", "static Texture* upscale_picture(")]
    with tempfile.TemporaryDirectory(prefix="fsr3fi-hud-capture-") as tmp:
        src = Path(tmp) / "test.cpp"
        exe = Path(tmp) / "test.exe"
        src.write_text("\n".join([PREFIX, *functions, CASES]), encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(src), "-o", str(exe)], check=True)
        return subprocess.run([str(exe)]).returncode


if __name__ == "__main__":
    sys.exit(main())
