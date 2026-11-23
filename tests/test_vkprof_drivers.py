"""Synthetic renderer samples exercise vkprof without a GPU or game."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vkprof.py"


class DriverAccountingTests(unittest.TestCase):
    def output(self, module):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hookshot_vk.map").write_text(
                " 0001:00000000 replay 0000000180001000 f replay.obj\n", encoding="utf-8")
            samples = root / "all.txt"
            samples.write_text(
                f"2 mkw_vk_replay|{module}+100;hookshot_vk.dll+1010\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(SCRIPT), str(samples)],
                capture_output=True, text=True, timeout=10,
                env=dict(os.environ, RX_MAPS=str(root)))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return proc.stdout

    def test_windows_amd_intel_driver_work_is_running(self):
        for module in ("amdvlk64.dll", "amdxc64.dll", "igvk64.dll", "igxelpicd64.dll"):
            with self.subTest(module=module):
                output = self.output(module)
                self.assertIn("100.0%  mkw_vk_replay (1 samples)", output)
                self.assertIn("share of 1 running samples", output)

    def test_other_amd_user_mode_driver_work_is_running(self):
        output = self.output("amdxx64.dll")
        self.assertIn("100.0%  mkw_vk_replay (1 samples)", output)
        self.assertIn("share of 1 running samples", output)

    def test_nvidia_driver_stays_running(self):
        self.assertIn("100.0%  mkw_vk_replay (1 samples)", self.output("nvoglv64.dll"))

    def test_renderer_leaf_stays_running(self):
        self.assertIn("100.0%  mkw_vk_replay (1 samples)", self.output("hookshot_vk.dll"))

    def test_system_wait_is_not_running(self):
        output = self.output("ntdll.dll")
        self.assertIn("0.0%  mkw_vk_replay (1 samples)", output)
        self.assertIn("share of 0 running samples", output)

    def test_versioned_linux_drivers_stay_running(self):
        for module in ("libnvidia-glcore.so.580.1", "libvulkan_radeon.so", "libvulkan_intel.so.1"):
            with self.subTest(module=module):
                self.assertIn("100.0%  mkw_vk_replay (1 samples)", self.output(module))

    def test_windows_driver_name_is_not_a_prefix_match(self):
        self.assertIn("share of 0 running samples", self.output("amdvlk64.dll.other"))


if __name__ == "__main__":
    unittest.main()
