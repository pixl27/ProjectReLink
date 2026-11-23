"""A function whose code starts below its entry gets no local registers (nx2/guest_locals.py keep_entered_below).

nx2/lift_fixes.py writes `goto L_<entry>;` right after the head of such a function. With A64_LOCAL_REGS or
A64_LOCAL_FLAGS the kit's localize_regs puts the locals' declarations, which load them from the context, at that same
place: the jump passed over the loads and the function ran on whatever the stack held (26 functions in a lift of OoT,
found in review). Such a function now keeps its registers and flags in the context, as on a lift without locals.

Checked here, without the game:
  - the rule: an entry above its first block joins no_local; an entry at its first block, a PLT stub and an empty body do not;
  - the lift asks it, before the functions are written and only when locals are on;
  - the shape it avoids: a head, the goto, then a declaration that loads a local is what must never be written, and a
    function in no_local is written without such a declaration (the kit's localize_regs is not called for it).

python -I -B tests/test_entry_below_locals.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
sys.path.insert(0, str(ROOT / "third_party" / "a64-kit" / "lift"))


class _Lifter:
    def __init__(self, funcs, plt=()):
        self.funcs = funcs
        self.plt = {e: (0, 0) for e in plt}
        self.no_local = set()


def _body(*pcs):
    return ({pc: (("op",), 0) for pc in pcs}, set())


class EnteredBelow(unittest.TestCase):
    def setUp(self):
        try:
            import guest_locals as guest
        except ImportError as e:   # (the kit's Python packages are not there: the lift cannot be loaded)
            self.skipTest("nx2/guest_locals.py does not import here: %s" % e)
        self.guest = guest

    def test_rule(self):
        lifter = _Lifter({
            0x1000: _body(0x1000, 0x1004, 0x1008),          # (entered at its first block: gets locals)
            0x2010: _body(0x2000, 0x2004, 0x2010, 0x2014),  # (code below the entry: lift_fixes jumps to L_2010)
            0x3008: _body(0x3000, 0x3008),                  # (the same)
            0x4004: _body(0x4000, 0x4004),                  # (a PLT stub is written without its body: no goto, not one)
            0x5000: ({}, set()),                            # (nothing walked: left alone)
        }, plt=(0x4004,))
        self.assertEqual(self.guest.keep_entered_below(lifter), 2)
        self.assertEqual(lifter.no_local, {0x2010, 0x3008})

    def test_keeps_what_was_there(self):
        lifter = _Lifter({0x1000: _body(0x1000), 0x2010: _body(0x2000, 0x2010)})
        lifter.no_local.add(0x1000)   # (a spliced function, say)
        self.guest.keep_entered_below(lifter)
        self.assertEqual(lifter.no_local, {0x1000, 0x2010})

    def test_same_rule_as_the_goto(self):
        """lift_fixes writes its goto for `e not in self.plt and min(body) != e`: the same test, read from its source"""
        fixes = (ROOT / "nx2" / "lift_fixes.py").read_text(encoding="utf-8")
        self.assertIn("if e in self.plt or min(body) == e:", fixes)
        self.assertIn('"goto L_%08x;\\n" % e', fixes)
        source = (ROOT / "nx2" / "guest_locals.py").read_text(encoding="utf-8")
        self.assertIn("e not in lifter.plt and body and min(body) != e", source)

    def test_the_lift_asks_before_it_writes(self):
        lift = (ROOT / "nx2" / "lift.py").read_text(encoding="utf-8")
        ask = lift.index("local_regs.keep_entered_below(L)")
        guard = lift.rindex("if a64_lift.local_regs() or a64_lift.local_flags():", 0, ask)
        self.assertLess(ask - guard, 600, "keep_entered_below is asked under the locals' own condition")
        self.assertLess(ask, lift.index("local_regs.check_patched(gen)"), "and before the written files are checked")
        self.assertLess(lift.index("discover.discover(L"), ask, "after the functions are known")


if __name__ == "__main__":
    unittest.main()
