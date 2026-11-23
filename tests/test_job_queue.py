"""The job queue's rules (scripts/job_queue.py): what may run beside what, the order, and the environment a job gets.

- A timed run and a build never run at the same time, whichever came first; a job never starts ahead of a waiting one it
  would hold up (a build behind a queued timed run waits; a shared game run behind a queued exclusive one waits).
- A timed run that uses no GPU lets shared game runs go beside it; timed runs of one group go one at a time.
- Priorities and aging: a low ticket that waited an hour is at normal's level, never above high.
- A job's environment never gains an OOT_, RX_ or A64_ variable (make_game.py lift_key hashes them all: the older queue's
  OOT_STATUS_LOCK=<job name> gave every job its own lift key, so no build took another's translation), and the older
  queue's OOT_STATUS_* is dropped.
- The RAM file system check takes the longest mount holding a path.
- Two exclusive jobs run one after the other, two shared ones together (a real queue in a temporary folder).
- Windows: a cancelled job's command and its children end, so do those of a runner killed outright (the job object), and a
  command that ends by itself leaves what it started running.
- Windows protects a player as Linux does: a hookshot.exe with no HOOKSHOT_JOB is someone playing (a timed or --when-idle
  job waits, status names it), one under a job or preparing shaders is not, one whose memory can't be read counts as nobody's;
  a build outside the queue is a python running nx2/build.py or make_game.py, backslashes or not. Checked on a made-up
  process table on any OS (the Windows code, its process reads stood in for), and on Windows with real processes.

python -I -B tests/test_job_queue.py
"""
import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "job_queue.py")
spec = importlib.util.spec_from_file_location("job_queue", SCRIPT)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)

SLOTS = {"shared": 3, "build": 2, "light": 4}


def T(pid, name, mode, state="waiting", t=0, prio="normal", **k):
    gpu, cpu = q.MODES[mode]
    return dict(pid=pid, name=name, mode=mode, gpu=gpu, cpu=cpu, state=state, t=t, prio=prio, **k)


class Rules(unittest.TestCase):
    def test_timing_and_builds_never_together(self):
        ts = [T(1, "b0", "build", "active"), T(2, "fps", "timing", t=1)]
        self.assertIn("builds", q.blocker(ts[1], ts, 10, SLOTS))
        ts = [T(1, "fps", "timing", "active"), T(2, "b1", "build", t=1)]
        self.assertIn("timed run", q.blocker(ts[1], ts, 10, SLOTS))
        self.assertIsNone(q.blocker(ts[1], ts, 10, SLOTS, quiet=False))   # (HOOKSHOT_QUEUE_QUIET=0)

    def test_no_jumping_ahead(self):
        ts = [T(1, "fps", "timing", t=1), T(2, "b1", "build", t=2)]
        self.assertIn("queued ahead", q.blocker(ts[1], ts, 10, SLOTS))
        ts = [T(1, "sh0", "shared", "active"), T(2, "excl", "gpu", t=1), T(3, "sh1", "shared", t=2)]
        self.assertIn("GPU", q.blocker(ts[1], ts, 10, SLOTS))
        self.assertIn("queued ahead", q.blocker(ts[2], ts, 10, SLOTS))
        ts[1]["ready"] = False   # (--when-idle while you play: it keeps its place but holds nobody up)
        self.assertIsNone(q.blocker(ts[2], ts, 10, SLOTS))

    def test_slots(self):
        ts = [T(i, "b%d" % i, "build", "active") for i in (1, 2)] + [T(3, "b3", "build", t=1)]
        self.assertIn("build slots", q.blocker(ts[2], ts, 10, SLOTS))
        self.assertIsNone(q.blocker(ts[2], ts, 10, dict(SLOTS, build=3)))
        ts = [T(i, "s%d" % i, "shared", "active") for i in (1, 2, 3)] + [T(4, "s4", "shared", t=1)]
        self.assertIn("shared GPU slots", q.blocker(ts[3], ts, 10, SLOTS))
        ts = [T(1, "g", "gpu", "active"), T(2, "checks", "light", t=1), T(3, "b", "build", t=2)]
        self.assertIsNone(q.blocker(ts[1], ts, 10, SLOTS))
        self.assertIsNone(q.blocker(ts[2], ts, 10, SLOTS))

    def test_timing_without_gpu_and_groups(self):
        ts = [T(1, "build-time", "timing-cpu", "active"), T(2, "shots", "shared", t=1), T(3, "fps", "timing", t=2)]
        self.assertIsNone(q.blocker(ts[1], ts, 10, SLOTS))
        self.assertIsNotNone(q.blocker(ts[2], ts, 10, SLOTS))   # (the GPU the shot run claimed, and the measuring one)
        ts = [T(1, "area-x-1", "timing", "active"), T(2, "area-x-2", "timing", t=1), T(3, "other-1", "timing", t=2)]
        self.assertIn("one timed run per group", q.blocker(ts[1], ts, 10, SLOTS))
        self.assertNotIn("per group", q.blocker(ts[2], ts, 10, SLOTS))   # (another group's: waits for the GPU only)
        self.assertEqual(q.group(T(9, "area-x-3", "timing")), "area-x")
        self.assertEqual(q.group(T(9, "fps-kokiri", "timing", group="mine")), "mine")

    def test_priorities_and_aging(self):
        low_hour = q.rank(T(1, "x", "build", t=0, prio="low"), 3700)
        self.assertEqual(low_hour[0], 0)
        self.assertLess(q.rank(T(2, "y", "build", t=3000, prio="high"), 3700), low_hour)
        self.assertEqual(q.rank(T(3, "z", "build", t=0, prio="normal"), 99999)[0], 0)


class Environment(unittest.TestCase):
    def test_no_lift_key_variables(self):
        with tempfile.TemporaryDirectory() as d:
            s = dict(q.DEFAULTS, HOOKSHOT_QUEUE_STATE=d)
            base = {"PATH": "/usr/bin", "TMPDIR": d, "OOT_STATUS_LOCK": "build-x", "OOT_STATUS_MODE": "build"}
            for mode in q.MODES:
                env = q.job_env(base, "a-job", mode, s, 42)
                added = {k for k in env if k not in base and k.startswith(q.LIFT_KEY_PREFIXES)}
                self.assertEqual(added, set(), mode)
                self.assertNotIn("OOT_STATUS_LOCK", env)
                self.assertEqual(env["HOOKSHOT_JOB_NAME"], "a-job")
            self.assertIn("__GL_SHADER_DISK_CACHE_PATH", q.job_env(base, "g", "shared", s, 42))
            self.assertNotIn("__GL_SHADER_DISK_CACHE_PATH", q.job_env(base, "b", "build", s, 42))

    def test_ram_file_system(self):
        table = [("/", "btrfs"), ("/tmp", "tmpfs"), ("/tmp/disk", "ext4")]
        self.assertTrue(q.on_ram("/tmp/x", table))
        self.assertFalse(q.on_ram("/tmp/disk/x", table))
        self.assertFalse(q.on_ram("/tmpx", table))

    def test_cpu_sets(self):
        s = dict(q.DEFAULTS)
        self.assertEqual(q.cpu_list("0-2,5"), {0, 1, 2, 5})
        self.assertEqual(q.cpu_text({0, 1, 2, 5}), "0-2,5")
        self.assertEqual(q.cpu_sets(s, domains=[frozenset({0})]), (None, None))   # (one L3 domain: no pinning)


@unittest.skipIf(os.name == "nt", "the shell commands below")
class RealQueue(unittest.TestCase):
    def run_two(self, mode):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, HOOKSHOT_QUEUE_DIR=os.path.join(d, "q"), HOOKSHOT_QUEUE_STATE=os.path.join(d, "s"),
                       HOOKSHOT_QUEUE_SCOPE="0", HOOKSHOT_QUEUE_MIN_FREE_GB="0", HOOKSHOT_QUEUE_MAX_PRESSURE="100",
                       HOOKSHOT_QUEUE_MAX_SWAP_GB="100000", HOOKSHOT_QUEUE_LEGACY=os.path.join(d, "none"))
            for k in ("HOOKSHOT_JOB", "OOT_STATUS_LOCK"):
                env.pop(k, None)
            marks = os.path.join(d, "marks")
            cmd = lambda n: [sys.executable, "-I", SCRIPT, "run", n, "--mode", mode, "--",
                             sys.executable, "-c", "import time; f=open(%r, 'a'); f.write('%s+ %%f\\n' %% time.time()); "
                             "f.flush(); time.sleep(1.5); f.write('%s- %%f\\n' %% time.time())" % (marks, n, n)]
            a = subprocess.Popen(cmd("a"), env=env, stderr=subprocess.DEVNULL)
            time.sleep(0.4)
            b = subprocess.Popen(cmd("b"), env=env, stderr=subprocess.DEVNULL)
            self.assertEqual((a.wait(30), b.wait(30)), (0, 0))
            with open(marks) as f:
                t = {line.split()[0]: float(line.split()[1]) for line in f}
            return t["b+"] < t["a-"]   # (b started while a still ran)

    def test_exclusive_one_at_a_time_shared_together(self):
        self.assertFalse(self.run_two("gpu"))
        self.assertTrue(self.run_two("shared"))


class WindowsPlayers(unittest.TestCase):
    """The Windows code over a made-up process table: Toolhelp's (pid: parent, program) and each process's command line and
    environment as its PEB holds them. Main returned "nobody" on Windows whatever ran"""
    PROCS = {   # (pid: (parent, program, command line, environment block; None: its memory can't be read))
        4: (0, "System", None, None),
        40: (4, "hookshot.exe", '"C:\\Games\\Hookshot\\hookshot.exe"', "=C:=C:\\Games\0PATH=C:\\x\0"),
        41: (4, "Hookshot.exe", "hookshot.exe --x", "HOOKSHOT_JOB=fps\0PATH=C:\\x\0"),
        42: (4, "hookshot.exe", "hookshot.exe --prepare-shaders", "PATH=C:\\x\0"),
        43: (4, "hookshot.exe", None, None),
        50: (4, "python.exe", 'python.exe -I C:\\src\\nx2\\build.py --jobs 4', "PATH=C:\\x\0"),
        51: (4, "python.exe", "python.exe -I nx2/make_game.py", "HOOKSHOT_JOB=b1\0"),
        52: (4, "python.exe", "python.exe -c pass", "PATH=C:\\x\0"),
        53: (60, "python.exe", "python.exe nx2/build.py", "PATH=C:\\x\0"),
        60: (4, "cmd.exe", "cmd /c x", "PATH=C:\\x\0"),
        61: (60, "notepad.exe", "notepad", "PATH=C:\\x\0"),
        62: (61, "conhost.exe", "conhost", "PATH=C:\\x\0"),   # (61 is a pid used again: started before its "parent")
    }
    BORN = {60: 10, 61: 15, 53: 20, 62: 1}

    def setUp(self):
        self.read = []
        self.PROCS = dict(self.PROCS)

        def params(pid):
            self.read.append(pid)
            p = self.PROCS.get(pid)
            return None if p is None or p[2] is None else (p[2], p[3] + "\0")
        split = lambda cmd: [a.strip('"') for a in cmd.split()]   # (CommandLineToArgvW enough for these)
        self.patches = [mock.patch.object(q, "WINDOWS", True),
                        mock.patch.object(q, "_win_snapshot", lambda: {k: v[:2] for k, v in self.PROCS.items()}, create=True),
                        mock.patch.object(q, "_win_params", params, create=True),
                        mock.patch.object(q, "_win_split", split, create=True),
                        mock.patch.object(q, "_win_start", lambda pid: self.BORN.get(pid, 0) if pid in self.PROCS else None),
                        mock.patch.dict(os.environ, {"HOOKSHOT_QUEUE_PLAYING_FILE": ""})]
        for m in self.patches:
            m.start()
        self.s = dict(q.DEFAULTS)

    def tearDown(self):
        for m in reversed(self.patches):
            m.stop()

    def test_games(self):
        games = {g["pid"]: g["job"] for g in q.game_procs(self.s)}
        self.assertEqual(games, {40: "", 41: "fps", 42: "prepare", 43: None})
        self.assertFalse({4, 50, 60, 61} & set(self.read), "only the games' and pythons' memory is read")
        self.assertEqual(q.env_of(40), {"PATH": "C:\\x"}, "the block's =C: entries are no variables")

    def test_playing(self):
        with mock.patch.object(q.time, "sleep"):
            self.assertTrue(q.user_playing(self.s), "a hookshot.exe of nobody's job: someone is playing")
            del self.PROCS[40]
            self.assertFalse(q.user_playing(self.s), "a job's game, shader preparation, an unreadable one: nobody")

    def test_status(self):
        out = io.StringIO()
        with mock.patch.object(q, "read_tickets", lambda *a, **k: []), mock.patch.object(q, "legacy_tickets", lambda s: []), \
                contextlib.redirect_stdout(out):
            q.status(self.s)
        self.assertIn("playing: hookshot.exe (pid 40)", out.getvalue())
        self.assertIn("game: Hookshot.exe pid 41 under job fps", out.getvalue())
        self.assertIn("a build outside the queue: pid 50", out.getvalue())

    def test_builds_outside(self):
        self.assertEqual(q.heavy_procs(self.s), [50, 53])
        self.assertEqual(q.heavy_procs(self.s, q.tree(60)), [50], "a job's tree is its own")

    def test_tree(self):
        self.assertEqual(sorted(q.tree(60)), [53, 60, 61], "61's 62 is older than 61: a pid used again")


@unittest.skipUnless(os.name == "nt", "Windows' processes")
class WindowsPlayersReal(unittest.TestCase):
    """Real processes: a copy of ping.exe named hookshot.exe, with and without HOOKSHOT_JOB, read through Toolhelp and PEB"""
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.exe = os.path.join(self.d, "hookshot.exe")
        shutil.copy(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "PING.EXE"), self.exe)
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            p.kill()
            p.wait()
        shutil.rmtree(self.d, ignore_errors=True)

    def start(self, job=None, args=("-n", "60", "127.0.0.1")):
        env = {k: v for k, v in os.environ.items() if k not in ("HOOKSHOT_JOB", "OOT_STATUS_LOCK")}
        if job:
            env["HOOKSHOT_JOB"] = job
        p = subprocess.Popen([self.exe, *args], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p)
        return p.pid

    def test_a_game_and_a_jobs_game(self):
        mine, jobs = self.start(), self.start(job="probe")
        time.sleep(0.5)
        s = dict(q.DEFAULTS)
        games = {g["pid"]: g for g in q.game_procs(s)}
        self.assertEqual((games[mine]["job"], games[jobs]["job"]), ("", "probe"))
        self.assertEqual(q.argv_of(jobs)[1:], ["-n", "60", "127.0.0.1"])
        with mock.patch.dict(os.environ, {"HOOKSHOT_QUEUE_PLAYING_FILE": ""}):
            self.assertTrue(q.user_playing(s))
        self.assertIn(jobs, q.tree(os.getpid()), "the test's own children are in its tree")

    def test_a_build_outside_the_queue(self):
        script = os.path.join(self.d, "nx2", "build.py")
        os.makedirs(os.path.dirname(script))
        with open(script, "w") as f:
            f.write("import time\ntime.sleep(60)\n")
        p = subprocess.Popen([sys.executable, "-I", script])
        self.procs.append(p)
        time.sleep(0.5)
        self.assertIn(p.pid, q.heavy_procs(dict(q.DEFAULTS)))
        self.assertNotIn(p.pid, q.heavy_procs(dict(q.DEFAULTS), q.tree(os.getpid())))


@unittest.skipUnless(os.name == "nt", "Windows' job object")
class WindowsTree(unittest.TestCase):
    """On Windows a job's command and what it starts end with the job (a real queue in a temporary folder): cancel ends the
    command's own children too, so does a runner killed outright (its job object), and a command that ends by itself leaves
    what it started running. Before the job object a cancelled job's children ran on: builds and games beside the next job"""
    GRAND = ("import os, sys, time; open(sys.argv[1] + '.tmp', 'w').write(str(os.getpid())); "
             "os.replace(sys.argv[1] + '.tmp', sys.argv[1]); time.sleep(120)")

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = dict(os.environ, HOOKSHOT_QUEUE_DIR=os.path.join(self.d, "q"), HOOKSHOT_QUEUE_STATE=os.path.join(self.d, "s"),
                        HOOKSHOT_QUEUE_SCOPE="0", HOOKSHOT_QUEUE_MIN_FREE_GB="0", HOOKSHOT_QUEUE_MAX_PRESSURE="100",
                        HOOKSHOT_QUEUE_MAX_SWAP_GB="100000", HOOKSHOT_QUEUE_LEGACY=os.path.join(self.d, "none"))
        for k in ("HOOKSHOT_JOB", "OOT_STATUS_LOCK"):
            self.env.pop(k, None)
        self.pidfile = os.path.join(self.d, "grandchild")
        self.grand = None

    def tearDown(self):
        if self.grand and q._win_start(self.grand) is not None:
            subprocess.run(["taskkill", "/F", "/PID", str(self.grand)], capture_output=True)
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", self.d], capture_output=True)

    def start(self, waits):
        """A runner whose command starts a grandchild, then waits for it (waits) or ends at once; the grandchild's pid"""
        inner = "import subprocess, sys; p = subprocess.Popen([sys.executable, '-c', %r, %r])%s" % (
            self.GRAND, self.pidfile, "; p.wait()" if waits else "")
        runner = subprocess.Popen([sys.executable, "-I", SCRIPT, "run", "tree", "--mode", "light", "--",
                                   sys.executable, "-c", inner], env=self.env, stderr=subprocess.DEVNULL)
        for _ in range(200):
            if os.path.exists(self.pidfile):
                break
            time.sleep(0.1)
        with open(self.pidfile) as f:
            self.grand = int(f.read())
        return runner

    def gone(self, pid):
        for _ in range(50):
            if q._win_start(pid) is None:
                return True
            time.sleep(0.1)
        return False

    def test_cancel_ends_the_commands_children(self):
        runner = self.start(waits=True)
        subprocess.run([sys.executable, "-I", SCRIPT, "cancel", "tree"], env=self.env, capture_output=True, check=True)
        runner.wait(20)
        self.assertTrue(self.gone(self.grand), "the cancelled job's grandchild still runs")

    def test_a_runner_killed_outright_ends_them(self):
        runner = self.start(waits=True)
        runner.kill()   # (TerminateProcess: no finally runs, as cancel's os.kill did; only the job object is left to act)
        runner.wait(20)
        self.assertTrue(self.gone(self.grand), "a killed runner's grandchild still runs")

    def test_a_command_that_ends_leaves_what_it_started(self):
        runner = self.start(waits=False)
        self.assertEqual(runner.wait(30), 0)
        time.sleep(1)
        self.assertIsNotNone(q._win_start(self.grand), "the job's end took what its command left running")


if __name__ == "__main__":
    unittest.main()
