"""Vector-only camera hooks retain integer and NZCV locals in emitted C, game-free."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
sys.path.insert(0, str(ROOT / "third_party" / "a64-kit" / "lift"))


class SplicedLocals(unittest.TestCase):
    def test_emission(self):
        import a64_lift
        import guest_locals
        import oot_wide
        entry = 0x1000
        image = SimpleNamespace(entries={entry}, hooks={})
        lifter = a64_lift.Lifter(image)
        lifter.funcs[entry] = ({entry: (("op",), 1), entry + 4: (("ret",), 0)}, set())
        lifter.after[entry] = oot_wide._clearance(10)
        texts = {1: ("c->x[19] = c->x[19] + 1; c->z = c->x[19] == 0;", 0)}
        with patch.dict(os.environ, {"A64_LOCAL_REGS": "6", "A64_LOCAL_FLAGS": "1", "A64_KEEP_FLAGS": "1"}, clear=False):
            with patch.object(a64_lift, "local_regs", lambda: 6), patch.object(a64_lift, "local_flags", lambda: True):
                self.assertEqual(guest_locals.keep_spliced(lifter), 0)
                emitted, _ = lifter.emit_function(entry, texts)
                self.assertIn("x19_ = c->x[19]", emitted)
                self.assertIn("fz_ = c->z", emitted)
                self.assertIn(str(lifter.after[entry]), emitted)
                lifter.after[entry] = str(lifter.after[entry]) + " c->x[19] = 0;"
                self.assertEqual(guest_locals.keep_spliced(lifter), 1)
                self.assertIn(entry, lifter.no_local)


if __name__ == "__main__":
    unittest.main()
