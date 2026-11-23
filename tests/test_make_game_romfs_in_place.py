"""Game-free checks of make_game's --romfs-in-place (scripts/dev_packs.py's packs and bases): which builds read the romfs where
it is, and which refuse it.

Run with: python -I tests/test_make_game_romfs_in_place.py
"""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "nx2" / "make_game.py"
spec = importlib.util.spec_from_file_location("make_game_under_test", SOURCE)
make_game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_game)


class RomfsInPlaceTests(unittest.TestCase):
    def parse(self, *argv, env=None):
        clean = {k: v for k, v in os.environ.items() if k != make_game.ROMFS_IN_PLACE_ENV}
        with mock.patch.object(sys, "argv", ["make_game.py", "--out", "out", *argv]), \
             mock.patch.dict(os.environ, dict(clean, **(env or {})), clear=True):
            return make_game.parse_args()

    def test_a_folder_build_reads_its_romfs_in_place_when_asked(self):
        self.assertTrue(self.parse("--romfs", "r", "--exefs", "e", "--romfs-in-place").romfs_in_place)
        self.assertTrue(self.parse("--romfs", "r", "--exefs", "e", env={"HOOKSHOT_ROMFS_IN_PLACE": "1"}).romfs_in_place)

    def test_a_build_not_asked_links_as_before(self):
        self.assertFalse(self.parse("--romfs", "r", "--exefs", "e").romfs_in_place)
        self.assertFalse(self.parse("--romfs", "r", "--exefs", "e", env={"HOOKSHOT_ROMFS_IN_PLACE": "0"}).romfs_in_place)

    def test_the_variable_is_passed_over_by_a_nup_build(self):
        """(a player's build with the variable left set in the shell: its game folder still gets its romfs)"""
        self.assertFalse(self.parse("--nup", "game.nup", env={"HOOKSHOT_ROMFS_IN_PLACE": "1"}).romfs_in_place)

    def test_the_switch_with_a_nup_is_refused(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("--nup", "game.nup", "--romfs-in-place")
        self.assertIn("--romfs-in-place", out.getvalue())

    def test_the_read_stage_links_only_without_it(self):
        """main's folder branch: the link (a copy where it is refused) is made only when the romfs is not read in place, and the
        romfs the rest of the build reads (OOT_ROMFS, the shader list, the texts) is then the given folder"""
        src = SOURCE.read_text(encoding="utf-8")
        branch = src[src.index("in_place = bool(getattr(a, \"romfs_in_place\", False))"):]
        branch = branch[:branch.index("shutil.rmtree(kept, ignore_errors=True)")]   # (hookshot: the link after the build check)
        self.assertIn("romfs = os.path.abspath(a.romfs)", branch)
        self.assertIn("if not in_place:\n            link_or_copy(a.romfs, romfs)", branch)


if __name__ == "__main__":
    unittest.main()
