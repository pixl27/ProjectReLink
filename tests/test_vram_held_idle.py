"""VRAM held for nothing: frame generation's images are let go when it is off, and a recording's stream and descriptor pool
start smaller and grow.

Frame interpolation's images (vk_framegen.cpp) and FSR 3.1's (vk_fsr3fi.cpp) were let go only when the window's size changed:
about 150 MB at 2560x1440 2x, 800 MB at 4K 8x, kept for the run by a player who tried it and turned it off. (The BC3 scratch's
size is an earlier fix's.)

python -I -B tests/test_vram_held_idle.py
"""
from pathlib import Path
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"


class FramegenIdleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fg = (VK / "vk_framegen.cpp").read_text(encoding="utf-8")
        cls.fi = (VK / "vk_fsr3fi.cpp").read_text(encoding="utf-8")
        cls.replay = (VK / "vk_replay.cpp").read_text(encoding="utf-8")

    def test_each_present_asks(self):
        self.assertIn("framegen_idle(interpolated);", self.replay)

    def test_off_for_a_while_drains_then_drops(self):
        idle = block(self.fg, "void framegen_idle(bool recorded)")
        self.assertIn("if (!due.shared) return;", idle)   # (the rule and its sequences: tests/test_framegen_idle.py)
        # (the display thread and the GPU done with the sets before they go, as ensure_size does)
        for part in ("fg.jobs.empty()", "device_wait_idle_locked();", "drop_size();"):
            self.assertIn(part, idle, part)
        self.assertLess(idle.index("device_wait_idle_locked();"), idle.index("drop_size();"))

    def test_fsr3_interpolation_goes_too_and_its_dead_images_are_freed(self):
        idle = block(self.fg, "void framegen_idle(bool recorded)")
        self.assertIn("fsr3fi::frames_wanted()", idle)
        self.assertIn("fsr3fi::release();", idle)
        release = block(self.fi, "void release()")
        for part in ("drop_image(", "g.ready = 0;", "free_dead();"):
            self.assertIn(part, release, part)


class RecordingMemoryTests(unittest.TestCase):
    """A recording slot's stream (1 MB, grown on demand) and descriptor pool (half, made whole after one ran out)."""
    @classmethod
    def setUpClass(cls):
        cls.draw = (VK / "vk_draw.cpp").read_text(encoding="utf-8")

    def test_the_stream_starts_at_1_mb_and_still_grows(self):
        # (main's streams always grow: RX_VK_STREAM_GROW_OFF is gone)
        self.assertIn('atoi(getenv("RX_VK_STREAM_MB"))) : 1u) << 20;', self.draw)

    def test_a_pool_starts_at_half_and_is_made_whole_after_running_out(self):
        first = block(self.draw, "uint32_t pool_shift_first()")
        self.assertIn('getenv("RX_VK_DESC_POOL_GROW")', first)
        make = block(self.draw, "VkDescriptorPool make_pool(uint32_t shift = 0)")
        self.assertIn("div0 << std::min(shift, 4u)", make)
        begin = block(self.draw, "void draw_begin_recording(int slot, VkCommandBuffer cmd, uint64_t value)")
        self.assertIn("r.pool = make_pool(r.pool_shift);", begin)
        grow = begin[begin.index("vkResetDescriptorPool"):begin.index("r.extra.clear();")]
        self.assertIn("!r.extra.empty() && r.pool_shift > 0", grow)
        self.assertIn("make_pool(0)", grow)
        # (the pools added when one ran out stay whole)
        self.assertIn("const VkDescriptorPool pool = make_pool();", self.draw)


if __name__ == "__main__":
    unittest.main()
