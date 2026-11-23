"""Exercise build.main's real job planning and object keys, without a compiler or game.

python -I -B tests/test_a64_helper_build_flags.py
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("HOOKSHOT_BUILD_TEST_SOURCE", ROOT / "nx2" / "build.py"))
spec = importlib.util.spec_from_file_location("a64_helper_build", SOURCE)
build = importlib.util.module_from_spec(spec)
# Avoid probing the machine's native CPU with clang while importing the consumer.
with mock.patch.dict(os.environ, {"RX_MARCH": "x86-64-v3"}):
    spec.loader.exec_module(build)


class Planned(Exception):
    pass


class HelperBuildFlagsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hookshot-helper-flags-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.gen = self.work / "gen"
        self.gen.mkdir()
        for name in ("code_000.c", "code_hot_0.c", "a64_table.c", "a64_hooks.c", "a64_suyu.c"):
            (self.gen / name).write_text("/* authored planning fixture, not lifted game code */\n", encoding="utf-8")
        image = self.work / "image"
        image.mkdir()
        (image / "image.json").write_text(json.dumps({
            "base": 0, "modules": [{"name": "main", "build_id": "authored-fixture"}],
        }), encoding="utf-8")

    def plan(self, value=None, hot="O2", pgo="", dll=False, light_passes=True):
        jobs, keys = {}, {}
        actual_plan = build.plan_jobs

        def capture(planned):
            # Use the actual cache planner, not just the flag-list construction.
            _, planned_keys = actual_plan(planned)
            jobs.update({Path(job[1]).name: job for job in planned})
            keys.update({Path(src).name: key for (src, obj), key in planned_keys.items()})
            raise Planned()

        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("RX_", "OOT_", "HOOKSHOT_"))}
        env.update(RX_HOT_OPT=hot, HOOKSHOT_TIMES="authored-build-times.json", HOOKSHOT_MEM_GATE="0")
        if not light_passes:
            env["HOOKSHOT_LIGHT_PASSES"] = "0"
        if value is not None:
            env["RX_A64_RT_O2"] = value
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
            stack.enter_context(mock.patch.object(build.kit, "WORK", str(self.work)))
            stack.enter_context(mock.patch.object(build.kit, "wait_for_quiet"))
            stack.enter_context(mock.patch.object(build, "OBJ_CACHE", str(self.work / "objcache")))
            stack.enter_context(mock.patch.object(build, "OBJ_STORE", None))
            stack.enter_context(mock.patch.object(build, "KEPT", {}))
            stack.enter_context(mock.patch.object(build, "PLANNED", {}))
            stack.enter_context(mock.patch.object(build, "PGO", pgo))
            stack.enter_context(mock.patch.object(build, "target", return_value=["--target=x86_64-fixture"]))
            stack.enter_context(mock.patch.object(build.toolchain, "find", return_value="authored-clang"))
            stack.enter_context(mock.patch.object(build.toolchain, "need", side_effect=lambda path: path))
            stack.enter_context(mock.patch.object(build.third_party_objects, "stamp", return_value="authored-tool-stamp"))
            stack.enter_context(mock.patch.object(build.third_party_objects, "folder", return_value=None))
            stack.enter_context(mock.patch.object(build, "ffmpeg_version_key", return_value=b""))
            stack.enter_context(mock.patch.object(build.version, "define", return_value="-DHOOKSHOT_VERSION=fixture"))
            stack.enter_context(mock.patch.object(build.version, "describe", return_value="authored-fixture"))
            if hasattr(build, "light_flags_work"):
                stack.enter_context(mock.patch.object(build, "light_flags_work", return_value=True))
            stack.enter_context(mock.patch.object(build, "plan_jobs", side_effect=capture))
            stack.enter_context(mock.patch.object(build.subprocess, "Popen", side_effect=AssertionError("unexpected child")))
            stack.enter_context(mock.patch.object(build.subprocess, "run", side_effect=AssertionError("unexpected child")))
            stack.enter_context(mock.patch.object(sys, "argv", ["build.py"] + (["--dll"] if dll else [])))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaises(Planned):
                build.main()
            base = build.flags(str(self.gen), str(self.work / "build" / "rt64"), "O1", True)
            if dll:
                base += ["-DA64_POLL_CTX"]
            split = build.split_giants_flags()
            light = list(build.LIGHT_FLAGS) if light_passes and hasattr(build, "LIGHT_FLAGS") else []
        self.assertEqual(set(jobs), {"code_000.c", "code_hot_0.c", "a64_table.c", "a64_hooks.c",
                                     "a64_suyu.c", "a64_rt.c", "eden_shim.c", "mkw_info.c"})
        for flag in ("-fno-strict-aliasing", "-ffp-contract=off", "-DA64_POLLING", "-DA64_WRITE_TRACK",
                     "-mcx16", "-rotation-max-header-size=0", "-capture-tracking-max-uses-to-explore=5000"):
            self.assertIn(flag, base)
        self.assertEqual(jobs["code_000.c"][3], base + split + light)
        for name in ("a64_table.c", "a64_hooks.c", "eden_shim.c", "mkw_info.c"):
            self.assertEqual(jobs[name][3], base, name)
        self.split = split
        return jobs, keys, base

    def hot_split(self):
        # (main's hot_flags: the hot pieces carry the giants' split when build.py has it, and no hoisting to hotter blocks)
        split = self.split if hasattr(build, "hot_flags") else []
        hoist = ["-mllvm", "-disable-hoisting-to-hotter-blocks=all"]
        return split + hoist if '"-disable-hoisting-to-hotter-blocks=all"' in open(build.__file__, encoding="utf-8").read() else split

    def test_default_changes_only_the_two_helpers_and_their_object_keys(self):
        for dll in (False, True):
            for light_passes in (True, False):
                with self.subTest(dll=dll, light_passes=light_passes):
                    before, before_keys, base = self.plan("0", dll=dll, light_passes=light_passes)
                    after, after_keys, _ = self.plan(dll=dll, light_passes=light_passes)
                    for name in before:
                        helper = name in ("a64_rt.c", "a64_suyu.c")
                        self.assertEqual(before[name][:3], after[name][:3])
                        self.assertEqual(before[name][4], after[name][4])
                        if helper:
                            self.assertEqual(before[name][3], base)
                            self.assertEqual(after[name][3], ["-O2" if f == "-O1" else f for f in base])
                        else:
                            self.assertEqual(before[name][3], after[name][3], name)
                        self.assertEqual(before_keys[name] != after_keys[name], helper, name)

    def test_false_and_enabled_values_route_exact_flags(self):
        for value in ("", "0", "false", " OFF ", "No", "1", "true"):
            with self.subTest(value=value):
                jobs, _, base = self.plan(value)
                expected = base if value.strip().lower() in ("", "0", "false", "off", "no") else [
                    "-O2" if f == "-O1" else f for f in base]
                for name in ("a64_rt.c", "a64_suyu.c"):
                    self.assertEqual(jobs[name][3], expected)
                self.assertEqual(jobs["code_hot_0.c"][3], ["-O2" if f == "-O1" else f for f in base] + self.hot_split())

    def test_existing_hot_optimization_and_profile_flags_do_not_reach_helpers(self):
        profile = self.work / "authored.profdata"
        profile.write_bytes(b"authored profile bytes: only hashed by planning, never compiled")
        for pgo in ("", "gen", str(profile)):
            with self.subTest(pgo=pgo):
                jobs, _, base = self.plan(hot="O3", pgo=pgo)
                expected_hot = ["-O3" if f == "-O1" else f for f in base] + self.hot_split()
                if pgo == "gen":
                    expected_hot += ["-fprofile-instr-generate", "-mllvm", "-disable-vp=true"]
                elif pgo:
                    digest = hashlib.sha1(profile.read_bytes(), usedforsecurity=False).hexdigest()[:16]
                    expected_hot += ["-fprofile-instr-use=" + pgo, "-DHOOKSHOT_PROFILE_SHA1=" + digest]
                self.assertEqual(jobs["code_hot_0.c"][3], expected_hot)
                for name in ("a64_rt.c", "a64_suyu.c"):
                    self.assertEqual(jobs[name][3], ["-O2" if f == "-O1" else f for f in base])


if __name__ == "__main__":
    unittest.main()
