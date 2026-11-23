"""A colour pass that tests depth EQUAL gets a margin towards the camera on Intel's Windows driver.

The game draws a depth prepass, then colour passes that test EQUAL against it with another program. The compiler of
Intel's Windows driver does not give the two programs the same position to the last bit, so most ground pixels failed the
test and the sky showed through (issue 17, Arc B390). The test becomes LESS_OR_EQUAL with a depth bias towards the camera,
only where depth is tested, not written, and the game asked for EQUAL. Not on Mesa's Intel driver, where EQUAL holds and
the LESS_OR_EQUAL drew the terrain over the rocks in front of it (Death Mountain Trail). RX_VK_EQ_BIAS=<units> chooses it
on any GPU, 0 turns it off. A draw that blends keeps EQUAL: LESS_OR_EQUAL drew a haze sheet the game keeps to its own
depth over Death Mountain Trail's mountain.
python -I -B tests/test_equal_depth_margin.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gpu" / "vk" / "vk_draw.cpp"


def main():
    source = SOURCE.read_text(encoding="utf-8")
    state = block(source, "void dynamic_state(VkCommandBuffer cmd")
    # (the margin's size is device_policy.h's equal_depth_bias: 64 on Intel's Windows driver but on Battlemage, then RX_VK_EQ_BIAS, read by
    # device_policy_env; tests/test_device_policy.py has it on each GPU and the switch's values)
    if "static const float eq_bias = dev.policy.equal_depth_bias;" not in state:
        raise SystemExit("the margin is not device_policy.h's equal_depth_bias")
    policy = (ROOT / "gpu" / "vk" / "device_policy.h").read_text(encoding="utf-8")
    if 'get("RX_VK_EQ_BIAS")' not in policy:
        raise SystemExit("no RX_VK_EQ_BIAS switch")
    if "p.equal_depth_bias = policy_rule::equal_depth_bias(f.driver_id, f.device_id);" not in policy:
        raise SystemExit("the margin is not computed by policy_rule::equal_depth_bias")
    eq_rule = block(policy, "inline float equal_depth_bias(VkDriverId driver, uint32_t device)")
    for snippet in (
        "driver != VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS",
        "(device & 0xFF00u) == 0xE200u",
        "return 64.0f;",
    ):
        if snippet not in eq_rule:
            raise SystemExit("policy_rule::equal_depth_bias lacks %r" % snippet)
    cond = "if (eq_bias > 0 && d.dt && !d.dw && d.dop == VK_COMPARE_OP_EQUAL && !blends)"
    if cond not in state:
        raise SystemExit("the margin is not limited to EQUAL tests that do not write depth and do not blend")
    before = state[:state.index(cond)]
    if "(s.color.blend_en >> i & 1) && at.color_format[i]" not in before:
        raise SystemExit("the margin does not look at the targets' blending")
    # (blending on with ONE, ZERO is no blending: such colour passes keep the margin, or the ground has holes)
    if "blend_factor(b.src_rgb) != VK_BLEND_FACTOR_ONE || blend_factor(b.dst_rgb) != VK_BLEND_FACTOR_ZERO" not in before:
        raise SystemExit("blending with ONE, ZERO counts as blending")
    margin = block(state, cond)
    for line in ("d.dop = VK_COMPARE_OP_LESS_OR_EQUAL;", "d.bias_en = VK_TRUE;", "d.bias[0] = -eq_bias"):
        if line not in margin:
            raise SystemExit("the margin lacks %r" % line)
    if not state.index("d.dop = compare_op(s.ds.df);") < state.index(cond):
        raise SystemExit("the margin must come after the game's compare op is read")
    print("EQUAL depth margin ok")


if __name__ == "__main__":
    main()
