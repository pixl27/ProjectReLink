"""Game-free checks of recomplauncher.json: it parses, and every settings key is a row of the in-game menu.

Run with: python -I tests/test_recomplauncher_json.py
"""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RecompLauncherJsonTests(unittest.TestCase):
    def test_settings_are_menu_rows(self):
        manifest = json.loads((ROOT / "recomplauncher.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["format"], 1)
        menu = (ROOT / "gpu" / "vk" / "vk_menu.cpp").read_text(encoding="utf-8")
        rows = {re.sub(" ", "_", name.lower()) for name in re.findall(r'\{TAB_[A-Z]+, "([^"]+)"', menu)}
        # (vk_menu.cpp ini_key: the rows whose key is not their name, renamed rows keeping the key an options file has:
        # output_res, resolution, upscaling, frame_rate_cap, dlss)
        ini_key = menu.split("const char* ini_key(const Row& r) {", 1)[1].split("\n}", 1)[0]
        rows |= set(re.findall(r'return "([a-z_0-9]+)";', ini_key))
        keys = {key for setting in manifest["settings"] for choice in setting["choices"] for key in choice["set"]}
        self.assertTrue(keys)
        self.assertEqual(keys - rows, set())

    def test_builds_into_the_launchers_game_folder(self):
        # (RecompLauncher issue 215: without --out the build went to ~/Games/Hookshot, not the drive the player
        # installed to. The folder it builds into is the one it plays, and the settings and saves live in it.)
        manifest = json.loads((ROOT / "recomplauncher.json").read_text(encoding="utf-8"))
        prepare = manifest["prepare"]["args"]
        out = prepare[prepare.index("--out") + 1]
        self.assertEqual(out, "{data}/game")
        self.assertEqual(manifest["start"]["args"], ["--play", out])
        self.assertEqual(manifest["folders"]["game"][0]["path"], out)
        self.assertEqual(manifest["config"][0]["file"][0]["path"], out + "/hookshot_options.ini")
        self.assertIn(out + "/user", manifest["saves"])
        self.assertNotIn("{home}/Games", json.dumps(manifest))
        self.assertNotIn("game", manifest)
        self.assertEqual(manifest["repo"], "spacefarergames/RedTunic")
        self.assertEqual(manifest["inputs"][0]["id"], "game")
        self.assertIn("--game={input:game}", prepare)
        self.assertEqual(prepare[prepare.index("--work") + 1], "{data}/cache/build")
        self.assertIn(out + "/image.bin", manifest["prepare"]["done"])

    def test_appimage_carries_it(self):
        self.assertIn('cp recomplauncher.json "$APPDIR/recomplauncher.json"',
                      (ROOT / "scripts" / "linux" / "appimage.sh").read_text(encoding="utf-8"))

    def test_windows_zip_carries_it(self):
        workflow = (ROOT / ".github" / "workflows" / "windows-builder.yml").read_text(encoding="utf-8")
        self.assertIn('scripts/package_recomplauncher.py dist/hookshot-setup.txt', workflow)
        self.assertIn("dist/RedTunic-*-Windows-x64.zip", workflow)

    def test_linux_tarball_is_published_from_the_appdir(self):
        package = (ROOT / "scripts/linux/appimage.sh").read_text(encoding="utf-8")
        self.assertIn('tar -C "$APPDIR" -czf "$B/RedTunic-$VERSION-Linux-x86_64.tar.gz" .', package)
        workflow = (ROOT / ".github/workflows/linux-appimage.yml").read_text(encoding="utf-8")
        self.assertGreaterEqual(workflow.count('*-Linux-x86_64.tar.gz'), 3)


if __name__ == "__main__":
    unittest.main()
