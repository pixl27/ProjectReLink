"""The flags stay in the context when the kit's A64_PACKED_FLAGS experiment is set (nx2/guest_locals.py local_flags).

A64_PACKED_FLAGS=1 makes the kit's pack_flags fold a flag-setting instruction's four stores (c->n=...; c->z=...;
c->c=...; c->v=...;) into one a64_set_nzcv(c, ...). That call writes the context where localize_regs' FLAG_REF does not
see it: with the flags as locals (on by default since
https://git.notrexed.dev/Hookshot/Hookshot/pulls/1576) a CMP's result went to the context and the following B.EQ
read the local Z from before it. So local_flags() answers no while A64_PACKED_FLAGS is set, as the kit reads it.

Checked here, without the game:
  - local_flags() is on by default, off with A64_PACKED_FLAGS set, and A64_LOCAL_FLAGS=1 does not turn it back on;
  - the kit's pack_flags does make the call this guards against, out of the four stores, and leaves no c->n ... that
    localize_regs would turn into a local.

python -I -B tests/test_packed_flags_locals.py
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
sys.path.insert(0, str(ROOT / "third_party" / "a64-kit" / "lift"))

NAMES = ("A64_PACKED_FLAGS", "A64_LOCAL_FLAGS")
# (a CMP's four flag stores, as the emitter writes them: pack_flags packs them only when no value reads the context)
CMP_FLAGS = "c->n=(uint8_t)(r>>63); c->z=(uint8_t)(r==0); c->c=(uint8_t)(a>=b); c->v=(uint8_t)0;"


class PackedFlags(unittest.TestCase):
    def setUp(self):
        try:
            import a64_lift
            import guest_locals as guest
        except ImportError as e:   # (the kit's Python packages are not there: the lift cannot be loaded)
            self.skipTest("nx2/guest_locals.py does not import here: %s" % e)
        self.a64_lift, self.guest = a64_lift, guest
        self.env = {k: os.environ.pop(k, None) for k in NAMES}

    def tearDown(self):
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_local_flags(self):
        self.assertTrue(self.guest.local_flags(), "on by default")
        os.environ["A64_PACKED_FLAGS"] = "1"
        self.assertFalse(self.guest.local_flags(), "off with the packed flags")
        os.environ["A64_LOCAL_FLAGS"] = "1"
        self.assertFalse(self.guest.local_flags(), "A64_LOCAL_FLAGS=1 does not win over them")

    def test_the_kit_packs_past_the_locals(self):
        self.assertEqual(self.a64_lift.pack_flags(CMP_FLAGS), CMP_FLAGS, "unset: left as it is")
        os.environ["A64_PACKED_FLAGS"] = "1"
        packed = self.a64_lift.pack_flags(CMP_FLAGS)
        self.assertIn("a64_set_nzcv(c,", packed)
        self.assertIsNone(self.a64_lift.FLAG_REF.search(packed), "no c->n ... left for localize_regs to see")


if __name__ == "__main__":
    unittest.main()
