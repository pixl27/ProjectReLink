"""The workers a build is asked for reach every step that runs work side by side, not only the compile (nx2/build_jobs.py).

A launcher's Build workers (an Automatic that leaves a quarter of the CPU threads free) gives make_game.py --jobs on Linux and
HOOKSHOT_JOBS on Windows. Only nx2/build.py took it: the lift's processes, the prompts' and the shader preparation's sized
themselves from the CPU count, so an update filled every thread whatever the player chose.
Checked here, with no game:
  - build_jobs.requested and cap: 1 or more is a ceiling, None, "" and 0 are automatic, anything else is refused;
  - make_game.main puts a count it was given (--jobs, or HOOKSHOT_JOBS already set) in HOOKSHOT_JOBS for its steps, and none
    when it was given none (each step keeps its own count, as before);
  - the lift's per-file pool (nx2/chunks.py _each, sized by file_jobs) runs on at most that many processes, counted by their ids;
  - children_on_cpus starts a process on that many CPUs and gives this one its CPUs back; make_game.prepare_shaders starts
    the game in it (a stand-in that prints the CPUs it may use: the game's pools follow them, gpu/vk/vk_progcache.cpp
    usable_cpus), on Linux, where the affinity mask can be read from Python.

python -I -B tests/test_build_jobs.py
"""
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import kit  # noqa: E402,F401  (a64-kit on the path)
import build_jobs  # noqa: E402
import chunks  # noqa: E402
import make_game  # noqa: E402

AFFINITY = hasattr(os, "sched_setaffinity")


def worker_id(_):
    time.sleep(0.02)   # (long enough that the pool starts every worker it may)
    return os.getpid()


class Clean(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for k in ("HOOKSHOT_JOBS", "HOOKSHOT_LIFT_JOBS", "HOOKSHOT_LIFT_NO_FORK"):
            os.environ.pop(k, None)


class Count(Clean):
    def test_requested(self):
        self.assertEqual(build_jobs.requested("4"), 4)
        self.assertEqual(build_jobs.requested(4), 4)
        for automatic in (None, "", " ", "0", 0):
            self.assertIsNone(build_jobs.requested(automatic), repr(automatic))
        for bad in ("-1", "four", "1.5"):
            with self.assertRaises(ValueError):
                build_jobs.requested(bad)

    def test_cap(self):
        self.assertEqual(build_jobs.cap(12), 12)   # (nothing asked: the step's own count)
        self.assertEqual(build_jobs.cap(12, {"HOOKSHOT_JOBS": "4"}), 4)
        self.assertEqual(build_jobs.cap(3, {"HOOKSHOT_JOBS": "4"}), 3)   # (a ceiling, never a floor)
        self.assertEqual(build_jobs.cap(12, {"HOOKSHOT_JOBS": "0"}), 12)
        self.assertEqual(build_jobs.cap(12, {"HOOKSHOT_JOBS": "x"}), 12)   # (make_game.py refuses it at its start)
        os.environ["HOOKSHOT_JOBS"] = "2"
        self.assertEqual(build_jobs.cap(12), 2)

    def test_make_game_keeps_its_parsing(self):
        self.assertEqual(make_game.requested_jobs("6"), 6)
        self.assertIsNone(make_game.requested_jobs("0"))
        with self.assertRaises(ValueError):
            make_game.requested_jobs("-2")


class MainExports(Clean):
    def main(self, jobs):
        with tempfile.TemporaryDirectory() as work, mock.patch.object(make_game, "check", return_value=0), \
                redirect_stdout(io.StringIO()):
            make_game.main(types.SimpleNamespace(jobs=jobs, work=work, check=True))
        return os.environ.get("HOOKSHOT_JOBS")

    def test_a_given_count_reaches_the_steps(self):
        self.assertEqual(self.main(3), "3")

    def test_one_in_the_environment_stays(self):
        os.environ["HOOKSHOT_JOBS"] = "5"
        self.assertEqual(self.main(None), "5")
        os.environ["HOOKSHOT_JOBS"] = "5"
        self.assertEqual(self.main(2), "2")   # (--jobs over the environment, as chosen_jobs takes it)

    def test_none_given_sets_none(self):
        self.assertIsNone(self.main(None))
        self.assertIsNone(self.main(0))   # (0: automatic)


@unittest.skipUnless("fork" in chunks.multiprocessing.get_all_start_methods(), "the forked pool: Linux")
class LiftPool(Clean):
    def pids(self):
        with mock.patch.object(chunks, "_available_gb", return_value=64.0), \
                mock.patch.object(chunks.os, "sched_getaffinity", create=True, return_value=set(range(16))):
            jobs = chunks.file_jobs()
        return jobs, set(chunks._each(worker_id, list(range(64)), jobs)) - {os.getpid()}

    def test_at_most_the_count(self):
        os.environ["HOOKSHOT_JOBS"] = "4"
        jobs, pids = self.pids()
        self.assertEqual(jobs, 4)
        self.assertTrue(1 <= len(pids) <= 4, pids)

    def test_nothing_asked_the_cores(self):
        jobs, pids = self.pids()
        self.assertEqual(jobs, 16)
        self.assertGreater(len(pids), 4)


CPUS_SCRIPT = "import os; print('cpus %d' % len(os.sched_getaffinity(0)), flush=True)"


@unittest.skipUnless(AFFINITY and len(os.sched_getaffinity(0)) >= 3,
                     "the affinity mask from Python, on 3 CPUs or more: Linux")
class PrepareCpus(Clean):
    def child_cpus(self, n):
        import subprocess
        with build_jobs.children_on_cpus(n):
            out = subprocess.run([sys.executable, "-I", "-c", CPUS_SCRIPT], capture_output=True, text=True, check=True).stdout
        return int(out.split()[1])

    def test_the_child_gets_n_and_this_one_its_own_back(self):
        mine = os.sched_getaffinity(0)
        self.assertEqual(self.child_cpus(2), 2)
        self.assertEqual(os.sched_getaffinity(0), mine)
        self.assertEqual(self.child_cpus(None), len(mine))
        self.assertEqual(self.child_cpus(len(mine) + 4), len(mine))

    def prepare(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = make_game.prepare_shaders([sys.executable, "-I", "-c", CPUS_SCRIPT], str(ROOT))
        self.assertEqual(code, 0, out.getvalue())
        return int(next(l for l in out.getvalue().splitlines() if l.startswith("cpus ")).split()[1])

    def test_the_preparation_on_the_count(self):
        mine = len(os.sched_getaffinity(0))
        os.environ["HOOKSHOT_JOBS"] = "2"
        self.assertEqual(self.prepare(), 2)
        os.environ.pop("HOOKSHOT_JOBS")
        self.assertEqual(self.prepare(), mine)   # (nothing asked: every CPU, as before)


if __name__ == "__main__":
    unittest.main()
