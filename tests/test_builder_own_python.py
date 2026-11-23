"""A builder package with a Python of its own never runs another one.

With tools/python/python.exe gone (an antivirus took it, most often) both Windows builders started the tools with the
first "python" Windows found (issue 2045, item 10): another version, without the modules the tools need, or not one the
player chose to trust with the game files. Now the start fails and says how to get the package's Python back. Tools with no
python folder at all still use the PC's.

The Windows builder's Runner.cs is read for the rule.

    python -I -B tests/test_builder_own_python.py
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

RUNNER = (ROOT / "builder" / "HookshotBuilder" / "Runner.cs").read_text(encoding="utf-8")


class WindowsBuilderTests(unittest.TestCase):
    def test_the_package_names_its_own_python_whenever_it_has_the_folder(self):
        find = RUNNER[RUNNER.index("public static Tools? Find()"):RUNNER.index("static (string Exe, string[]? Args) PythonOnPath()")]
        self.assertIn('return new Tools(bundled, File.Exists(py) || Directory.Exists(Path.Combine(bundled, "python")) ? py : "python");',
                      find)
        self.assertNotIn('File.Exists(py) ? py : "python"', RUNNER)

    def test_missing_is_the_file_gone_with_its_folder_there(self):
        self.assertIn('public bool OwnPythonMissing => !File.Exists(Python) && Directory.Exists(Path.Combine(Root, "python"));',
                      RUNNER)
        self.assertNotIn('tools.Python == "python"', RUNNER)   # (the old test: never true once the package names its own)

    def test_the_build_says_how_to_get_it_back(self):
        start = RUNNER[RUNNER.index("public void Start()"):]
        self.assertIn("Finished?.Invoke(tools.OwnPythonMissing ? Tools.OwnPythonMissingText :", start)
        self.assertEqual(RUNNER.count("restore it"), 1)

    def test_the_build_runs_at_normal_priority(self):
        # (below normal, a hybrid Intel CPU kept the read and the lift on its efficiency cores for an hour and more; the old
        # priority only when HOOKSHOT_LOW_PRIORITY=1 asks)
        start = RUNNER[RUNNER.index("public void Start()"):]
        below = start.index("ProcessPriorityClass.BelowNormal")
        self.assertEqual(RUNNER.count("ProcessPriorityClass.BelowNormal"), 1)
        self.assertIn('if (Environment.GetEnvironmentVariable("HOOKSHOT_LOW_PRIORITY") == "1")', start[max(0, below - 200):below])

if __name__ == "__main__":
    unittest.main()
