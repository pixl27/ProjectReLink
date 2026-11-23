"""The hot list every build uses (assets/profiles/hot_funcs.txt, nx2/make_game.py hot_list_for), game-free.

  - the committed list: one f_<8 hex digits> a line, none twice, at most scripts/hotlist.py's cap, its note in
    assets/profiles/README.md; and no listed function is one whose text edit in nx2/oot_patches.py finds its place by a
    float helper call (a64_fpx_<op>(c, : a hot function's are rewritten to their inline copies, a64_lift.hot_text, so that
    edit would be lost with a WARNING line only: the step divides' hook, which keeps the game's speed above 60 fps, is one);
  - which list a build takes: none with HOOKSHOT_HOT_LIST=0, else an explicit A64_HOT_FUNCS, the checkout's own list,
    or the committed list. A work folder's copy from an earlier build
    is replaced or removed, never used for a build that has no list (issue 505), and the checkout's own file is never
    touched; with --work in the checkout's own work folder and no list of its own, the committed one is read where it is
    (no copy there, which would count as the checkout's own list from then on);
  - the lift key: another committed list translates the game again, the same one does not, and a build with no list never
    takes the key of one with the list, also when the checkout's own list sits in its own work folder;
  (Hookshot's own packaging carries assets/ for the Builder: upstream's builder_package.py check is left out.)

    python -I -B tests/test_shipped_hot_list.py
"""
import argparse
import ast
import contextlib
import io
import importlib.util
import os
import re
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "assets" / "profiles" / "hot_funcs.txt"
HOT_FPX = re.compile(r"\ba64_fpx_(?!hot_|fz_)(\w+)\(c,")   # (third_party/a64-kit/lift/a64_lift.py HOT_FPX: what hot_text rewrites)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_game = load("make_game_under_test", ROOT / "nx2" / "make_game.py")


def clean_env(**extra):
    """os.environ without the switches these tests set (a developer's shell may hold A64_HOT_FUNCS or HOOKSHOT_HOT_LIST)"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("A64_", "OOT_", "RX_", "HOOKSHOT_HOT_LIST"))}
    env.update(extra)
    return env


class TheCommittedList(unittest.TestCase):
    def test_its_lines(self):
        data = LIST.read_bytes()
        self.assertNotIn(b"\r", data, "LF line ends")
        self.assertTrue(data.endswith(b"\n"), "a newline at the end")
        names = data.decode("ascii").split("\n")[:-1]
        bad = [n for n in names if not re.fullmatch(r"f_[0-9a-f]{8}", n)]
        self.assertEqual(bad, [], "every line is f_<8 hex digits>")
        self.assertEqual(len(names), len(set(names)), "no function twice")
        self.assertTrue(0 < len(names) <= 1500, "1 to 1,500 functions (scripts/hotlist.py's cap): %d" % len(names))
        # (what the lift reads of it: the first word of each line that starts with f_, a64_lift.hot_functions)
        self.assertEqual({w for line in data.decode().splitlines() for w in line.split()[:1] if w.startswith("f_")}, set(names))

    def test_its_note(self):
        note = (ROOT / "assets" / "profiles" / "README.md").read_text(encoding="utf-8")
        self.assertIn("## hot_funcs.txt", note)
        section = note[note.index("## hot_funcs.txt"):]
        self.assertIn("Recorded:", section, "the note says where and how it was recorded")
        n = len(LIST.read_text(encoding="ascii").split())
        self.assertTrue("%d functions" % n in section, "the note gives the list's length (%d functions)" % n)

    def test_no_listed_function_loses_an_edit(self):
        # (nx2/oot_patches.py read with ast, never imported: its constants that hold a call hot_text rewrites, and the function
        # heads of the same edit, which shares the constant's stem: STEP_OLD -> STEP_SITES, TURN_RATE_OLD -> TURN_RATE_HEAD)
        tree = ast.parse((ROOT / "nx2" / "oot_patches.py").read_text(encoding="utf-8"))
        consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    consts[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    pass

        def strings(v):
            if isinstance(v, str):
                yield v
            elif isinstance(v, (tuple, list)):
                for x in v:
                    yield from strings(x)

        lift = (ROOT / "third_party" / "a64-kit" / "lift" / "a64_lift.py").read_text(encoding="utf-8")
        self.assertIn('HOT_FPX = re.compile(r"%s")' % HOT_FPX.pattern, lift, "this check's pattern is the lift's hot_text's")
        anchored = sorted(k for k, v in consts.items() if any(HOT_FPX.search(s) for s in strings(v)))
        heads = set()
        for k in anchored:
            stem = k.rsplit("_", 1)[0]
            found = {h for c, v in consts.items() if c == stem or c.startswith(stem + "_")
                     for s in strings(v) for h in re.findall(r"void (f_[0-9a-f]{8})\(Ctx", s)}
            self.assertTrue(found, "%s finds its place by a float helper call, and no function head shares its stem: name "
                                   "its function here so that this check covers it" % k)
            heads |= found
        self.assertTrue({"f_80706be0", "f_83b6ea90"} <= heads, "the step divides' hook is among them (%s)" % sorted(heads))
        listed = set(LIST.read_text(encoding="ascii").split())
        self.assertEqual(sorted(listed & heads), [], "listed functions whose edit would be lost (their calls become "
                                                     "a64_fpx_hot_*): take them out of the list, or anchor the edit otherwise")


class WhichListABuildTakes(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory(prefix="hookshot-hot-list-")
        self.addCleanup(d.cleanup)
        self.d = Path(d.name)
        self.root = self.d / "tree"
        (self.root / "assets" / "profiles").mkdir(parents=True)
        (self.root / "work").mkdir()
        self.shipped = self.root / "assets" / "profiles" / "hot_funcs.txt"
        self.shipped.write_text("f_80001000\nf_80002000\n", newline="\n")
        self.work = self.d / "build-work"
        self.work.mkdir()

    def take(self, **environ):
        return make_game.hot_list_for(str(self.work), environ, root=str(self.root))

    def test_the_committed_one(self):
        got = self.take()
        self.assertEqual(got, str(self.work / "hot_funcs.txt"))
        self.assertEqual((self.work / "hot_funcs.txt").read_bytes(), self.shipped.read_bytes())
        self.assertIn("the committed assets/profiles/hot_funcs.txt", make_game.hot_list_line(got, {}, root=str(self.root)))

    def test_an_old_copy_is_replaced(self):
        # (issue 505: the work folder's copy of an earlier list was used whenever it was there)
        (self.work / "hot_funcs.txt").write_text("f_8000dead\n")
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            self.take()
        self.assertEqual((self.work / "hot_funcs.txt").read_bytes(), self.shipped.read_bytes())
        self.assertIn("replaced by the committed one", said.getvalue(), "the log says a list was replaced")

    def test_an_unchanged_copy_is_not_written_again(self):
        self.take()
        before = os.stat(self.work / "hot_funcs.txt").st_mtime_ns
        os.utime(self.work / "hot_funcs.txt", ns=(before - 10**9, before - 10**9))
        self.take()
        self.assertEqual(os.stat(self.work / "hot_funcs.txt").st_mtime_ns, before - 10**9)

    def test_the_checkouts_own_goes_first(self):
        own = self.root / "work" / "hot_funcs.txt"
        own.write_text("f_80003000\n")
        got = self.take()
        self.assertEqual(got, str(self.work / "hot_funcs.txt"))
        self.assertEqual((self.work / "hot_funcs.txt").read_text(), "f_80003000\n")
        self.assertIn("work/hot_funcs.txt", make_game.hot_list_line(got, {}, root=str(self.root)))

    def test_a_list_named_in_the_shell(self):
        mine = self.d / "mine.txt"
        (self.root / "work" / "hot_funcs.txt").write_text("f_80003000\n")
        mine.write_text("f_80004000\n")
        (self.work / "hot_funcs.txt").write_text("f_8000dead\n")   # (a copy from an earlier build)
        self.assertIsNone(self.take(A64_HOT_FUNCS=str(mine)))
        self.assertFalse((self.work / "hot_funcs.txt").exists(), "the earlier copy is not left for the key to hash")
        self.assertEqual(mine.read_text(), "f_80004000\n")
        # (the shell's list at the work folder's own path stays)
        shutil.copyfile(mine, self.work / "hot_funcs.txt")
        relative = os.path.relpath(self.work / "hot_funcs.txt", self.root)
        self.assertIsNone(self.take(A64_HOT_FUNCS=relative))
        self.assertTrue((self.work / "hot_funcs.txt").exists())

    def test_none(self):
        (self.work / "hot_funcs.txt").write_text("f_80001000\n")
        got = self.take(HOOKSHOT_HOT_LIST="0")
        self.assertEqual(got, "")
        self.assertFalse((self.work / "hot_funcs.txt").exists())
        self.assertIn("none (HOOKSHOT_HOT_LIST=0)", make_game.hot_list_line(got, {}))
        # (over a list in the shell and the checkout's own too)
        (self.root / "work" / "hot_funcs.txt").write_text("f_80003000\n")
        self.assertEqual(self.take(HOOKSHOT_HOT_LIST="0", A64_HOT_FUNCS=str(self.shipped)), "")
        # (any other value is the default)
        self.assertEqual(self.take(HOOKSHOT_HOT_LIST="1"), str(self.work / "hot_funcs.txt"))

    def test_none_in_the_checkouts_own_work_folder_keeps_its_list(self):
        own = self.root / "work" / "hot_funcs.txt"
        own.write_text("f_80003000\n")
        self.assertEqual(make_game.hot_list_for(str(self.root / "work"), {"HOOKSHOT_HOT_LIST": "0"}, root=str(self.root)), "")
        self.assertEqual(own.read_text(), "f_80003000\n", "the checkout's own list is never removed")

    def test_the_committed_one_in_the_checkouts_own_work_folder(self):
        # (--work in the checkout's own work folder with no list of its own: the committed list is read where it is. A copy
        # there would be the checkout's own list from then on, and a later change of the committed one would never reach it)
        work = self.root / "work"
        got = make_game.hot_list_for(str(work), {}, root=str(self.root))
        self.assertEqual(got, str(self.shipped))
        self.assertFalse((work / "hot_funcs.txt").exists(), "no copy in the checkout's work folder")
        self.assertIn("the committed assets/profiles/hot_funcs.txt", make_game.hot_list_line(got, {}, root=str(self.root)))
        self.shipped.write_text("f_80005000\n", newline="\n")
        self.assertEqual(make_game.hot_list_for(str(work), {}, root=str(self.root)), str(self.shipped))
        self.assertFalse((work / "hot_funcs.txt").exists())

    def test_a_tree_without_one(self):
        self.shipped.unlink()
        (self.work / "hot_funcs.txt").write_text("f_8000dead\n")
        self.assertIsNone(self.take())
        self.assertFalse((self.work / "hot_funcs.txt").exists())

    def test_main_takes_it(self):
        # (the build's own lines: the env the lift gets, from hot_list_for, in place of "<work>/hot_funcs.txt when it is there")
        src = (ROOT / "nx2" / "make_game.py").read_text(encoding="utf-8")
        self.assertIn("hot = hot_list_for(work, os.environ)", src)
        self.assertIn('env["A64_HOT_FUNCS"] = hot', src)
        self.assertNotIn('if os.path.exists(os.path.join(work, "hot_funcs.txt")):\n        env["A64_HOT_FUNCS"]', src)


class TheLiftKey(unittest.TestCase):
    def test_the_list_and_none(self):
        with tempfile.TemporaryDirectory(prefix="hookshot-hot-key-") as d:
            d = Path(d)
            root, exefs = d / "tree", d / "exefs"
            (root / "assets" / "profiles").mkdir(parents=True)
            (root / "work").mkdir()
            exefs.mkdir()
            (exefs / "main").write_bytes(b"NSO0 one")
            shipped = root / "assets" / "profiles" / "hot_funcs.txt"
            shipped.write_text("f_80001000\n")
            work = d / "work"
            (work / "image").mkdir(parents=True)

            def key(**environ):
                hot = make_game.hot_list_for(str(work), environ, root=str(root))
                env = {"A64_HOT_CHUNK": "1", "OOT_WORK": str(work)}
                if hot is not None:
                    env["A64_HOT_FUNCS"] = hot
                return make_game.lift_key(str(exefs), str(work), env)

            with mock.patch.dict(os.environ, clean_env(), clear=True), warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)   # (nx2/standalone.py's reads of host/, which lift_key hashes)
                with_list = key()
                self.assertEqual(key(), with_list, "the same list: the lift is kept")
                none = key(HOOKSHOT_HOT_LIST="0")
                self.assertNotEqual(none, with_list, "no list is not the key of the list")
                self.assertEqual(key(HOOKSHOT_HOT_LIST="0"), none)
                self.assertEqual(key(), with_list, "back to the list: its key again")
                shipped.write_text("f_80002000\n")
                self.assertNotEqual(key(), with_list, "another committed list translates the game again")
                # (--work in the checkout's own work folder with no list of its own: the committed list, read where it is, is
                # hashed by its contents)
                work = root / "work"
                (work / "image").mkdir()
                in_place = key()
                self.assertFalse((work / "hot_funcs.txt").exists(), "no copy in the checkout's work folder")
                self.assertEqual(key(), in_place)
                self.assertNotEqual(key(HOOKSHOT_HOT_LIST="0"), in_place)
                shipped.write_text("f_80005000\n")
                self.assertNotEqual(key(), in_place, "a changed committed list translates the game again")
                # (--work in the checkout's own work folder, its own list there: none must still not take the list's key)
                (work / "hot_funcs.txt").write_text("f_80003000\n")
                own = key()
                self.assertNotEqual(key(HOOKSHOT_HOT_LIST="0"), own)
                self.assertTrue((work / "hot_funcs.txt").exists())
                # (no list from the switch over a shell's list: the shell's is not what the lift reads, nor what the key holds)
                mine = d / "mine.txt"
                mine.write_text("f_80004000\n")
                none_own = key(HOOKSHOT_HOT_LIST="0")
                with mock.patch.dict(os.environ, {"A64_HOT_FUNCS": str(mine)}):
                    self.assertEqual(key(HOOKSHOT_HOT_LIST="0"), none_own)


if __name__ == "__main__":
    unittest.main()
