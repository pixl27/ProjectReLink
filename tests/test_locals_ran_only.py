"""Guest registers and flags become C locals only in the functions play runs (nx2/guest_locals.py keep_not_ran).

Locals cost compile time in every function they are given to (the Test Farm: the game's compile +27% with them in all
~295,000) and pay only where the code runs. nx2/ran_funcs.txt names the 77,725 functions a 100-minute play session ran
(scripts/coverage/cover.py): ~42% of the lifted C's estimated compile. Every other function keeps its registers
and flags in the context, as on a lift without locals, so a function the list misses is as fast as before, never slower.

Checked here, without the game:
  - the rule: a function not on the list joins no_local; one on it, and a PLT stub, do not; what was there stays;
  - A64_LOCAL_FUNCS=all, or no list, gives every function its locals (no_local unchanged);
  - the committed list reads: f_ names, sorted, one a line, nothing else (addresses only: no game code);
  - the lift asks it under the locals' own condition, after the functions are known and before they are written;
  - the list is in the lift key: a .txt in nx2/, which make_game.py's lift_sources hashes.

python -I -B tests/test_locals_ran_only.py
"""
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
sys.path.insert(0, str(ROOT / "third_party" / "a64-kit" / "lift"))


class _Lifter:
    def __init__(self, funcs, plt=()):
        self.funcs = {e: ({e: (("op",), 0)}, set()) for e in funcs}
        self.plt = {e: (0, 0) for e in plt}
        self.no_local = set()


class RanOnly(unittest.TestCase):
    def setUp(self):
        try:
            import guest_locals as guest
        except ImportError as e:   # (the kit's Python packages are not there: the lift cannot be loaded)
            self.skipTest("nx2/guest_locals.py does not import here: %s" % e)
        self.guest = guest
        self.env = os.environ.pop("A64_LOCAL_FUNCS", None)

    def tearDown(self):
        os.environ.pop("A64_LOCAL_FUNCS", None)
        if self.env is not None:
            os.environ["A64_LOCAL_FUNCS"] = self.env

    def test_rule(self):
        lifter = _Lifter([0x1000, 0x2000, 0x3000, 0x4000], plt=(0x4000,))
        lifter.no_local.add(0x1000)   # (a spliced function, say: stays)
        self.assertEqual(self.guest.keep_not_ran(lifter, {0x1000, 0x2000}), 1)
        self.assertEqual(lifter.no_local, {0x1000, 0x3000})

    def test_all(self):
        os.environ["A64_LOCAL_FUNCS"] = "all"
        self.assertIsNone(self.guest.ran_functions())
        lifter = _Lifter([0x1000, 0x2000])
        self.assertIsNone(self.guest.keep_not_ran(lifter))
        self.assertEqual(lifter.no_local, set())

    def test_no_list(self):
        keep = self.guest.RAN
        self.guest.RAN = str(ROOT / "nx2" / "no_such_list.txt")
        try:
            self.assertIsNone(self.guest.ran_functions())
        finally:
            self.guest.RAN = keep

    def test_the_committed_list(self):
        text = (ROOT / "nx2" / "ran_funcs.txt").read_text(encoding="ascii")
        lines = text.split("\n")
        self.assertEqual(lines[-1], "", "ends with a newline")
        names = lines[:-1]
        self.assertTrue(all(re.fullmatch(r"f_8[0-9a-f]{7}", n) for n in names), "f_ names only, one a line")
        self.assertEqual(names, sorted(set(names)), "sorted, each once")
        self.assertGreater(len(names), 50000)
        ran = self.guest.ran_functions()
        self.assertEqual(len(ran), len(names))
        self.assertIn(int(names[0][2:], 16), ran)

    def test_the_lift_asks_before_it_writes(self):
        lift = (ROOT / "nx2" / "lift.py").read_text(encoding="utf-8")
        ask = lift.index("local_regs.keep_not_ran(L)")
        guard = lift.rindex("if a64_lift.local_regs() or a64_lift.local_flags():", 0, ask)
        self.assertLess(ask - guard, 900, "keep_not_ran is asked under the locals' own condition")
        self.assertLess(lift.index("discover.discover(L"), ask, "after the functions are known")
        self.assertLess(ask, lift.index("chunks.write(L"), "before they are written")

    def test_in_the_lift_key(self):
        make_game = (ROOT / "nx2" / "make_game.py").read_text(encoding="utf-8")
        self.assertRegex(make_game, r'LIFT_DATA = \([^)]*"\.txt"')
        self.assertEqual(Path(self.guest.RAN).parent.resolve(), (ROOT / "nx2").resolve())


if __name__ == "__main__":
    unittest.main()
