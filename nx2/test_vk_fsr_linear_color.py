"""Keep FSR 3.1's colour linear without changing DLSS's perceptual path.

Run with: python3 -I -B nx2/test_vk_fsr_linear_color.py
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def main() -> None:
    dlss = (ROOT / "gpu" / "vk" / "vk_dlss.cpp").read_text(encoding="utf-8")
    pipelines = function(dlss, "bool make_pipelines()")
    upscale = function(dlss, "bool dlss_upscale(")
    assert "static_assert(sizeof(PrepPush) == 84)" in dlss
    assert "static_assert(sizeof(OutPush) == 12)" in dlss
    assert "sizeof(PrepPush)" in pipelines and "sizeof(OutPush)" in pipelines
    assert "fsr ? kPrepLinearColor : 0" in upscale
    assert "const OutPush opc{{(int32_t)ow, (int32_t)oh}, fsr ? 1 : 0}" in upscale
    assert "RX_VK_FSR_SRGB" not in dlss

    prep = (ROOT / "gpu" / "vk" / "shaders" / "dlss_prep.comp").read_text(encoding="utf-8")
    output = (ROOT / "gpu" / "vk" / "shaders" / "dlss_out.comp").read_text(encoding="utf-8")
    assert "const int LINEAR_COLOR = 8" in prep
    assert "if ((pc.flags & LINEAR_COLOR) != 0) c *= c" in prep
    assert "int linear_color" in output
    assert "if (pc.linear_color == 0) c *= c" in output

    fsr = (ROOT / "gpu" / "vk" / "vk_fsr.cpp").read_text(encoding="utf-8")
    dispatch = function(fsr, "uint32_t fsr_dispatch(")
    assert "d.flags = 0" in dispatch
    assert 'getenv("RX_VK_FSR_SHARPNESS")' in fsr
    assert "d.sharpness = fsr_rcas_sharpness()" in dispatch
    assert "d.enableSharpening = d.sharpness > 0.0f" in dispatch
    assert "FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB" not in dispatch
    assert "perceptual" not in (ROOT / "gpu" / "vk" / "vk_fsr.h").read_text(encoding="utf-8")

    # FSR RCAS runs before the game draws its HUD. The later final-picture
    # sharpening pass must not stack on top and sharpen the HUD a second time.
    framegen = (ROOT / "gpu" / "vk" / "vk_framegen.cpp").read_text(encoding="utf-8")
    scale = function(framegen, "VkImage framegen_scale(")
    assert "const bool rcas = fsr_rcas_active()" in scale
    assert "rcas ? 0.0f : opts.sharpness / 100.0f" in scale
    assert 'rcas ? " (FSR RCAS owns the 3D)" : ""' in scale

    # Both paths must still produce the game's linear output: DLSS returns its
    # perceptual value and the output pass squares it; FSR receives and returns
    # the squared linear value, which the output pass copies.
    def prep(sample: float, fsr_path: bool) -> float:
        return sample * sample if fsr_path else sample

    def output_pass(sample: float, fsr_path: bool) -> float:
        return sample if fsr_path else sample * sample

    for sample in (0.0, 0.1, 0.5, 0.9, 1.0):
        expected = sample * sample
        assert output_pass(prep(sample, False), False) == expected
        assert output_pass(prep(sample, True), True) == expected

    print("vk_fsr_linear_color: FSR linear and DLSS perceptual contracts ok")


if __name__ == "__main__":
    main()
