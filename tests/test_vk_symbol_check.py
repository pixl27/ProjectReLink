"""The renderer import check prefers nm output that preserves ELF symbol versions."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("HOOKSHOT_BUILD_SOURCE", ROOT / "nx2" / "build.py"))


def load_check():
    source = BUILD.read_text(encoding="utf-8")
    tree = ast.parse(source, filename="nx2/build.py")
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "check_vk_symbols")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"os": os, "shutil": shutil, "subprocess": subprocess, "TRACY": False}
    exec(compile(module, "nx2/build.py", "exec"), namespace)
    return namespace["check_vk_symbols"]


class CheckVkSymbolsTests(unittest.TestCase):
    def test_prefers_llvm_nm_when_gnu_nm_drops_symbol_versions(self):
        check = load_check()
        tools = {"nm": "/usr/bin/nm", "llvm-nm": "/usr/bin/llvm-nm"}

        def run(command, **kwargs):
            if command[0] == tools["nm"]:
                # GNU nm on the affected system prints the base name without @GLIBC.
                return subprocess.CompletedProcess(command, 0, "                 U malloc\n", "")
            if command[0] == tools["llvm-nm"]:
                return subprocess.CompletedProcess(command, 0, "                 U malloc@GLIBC_2.2.5\n", "")
            raise AssertionError("unexpected nm: %r" % command)

        with mock.patch.object(shutil, "which", side_effect=tools.get), \
                mock.patch.object(subprocess, "run", side_effect=run) as run_mock:
            self.assertTrue(check("hookshot_vk.so"))

        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args.args[0][0], tools["llvm-nm"])

    def test_falls_back_to_nm_when_llvm_nm_is_unavailable(self):
        check = load_check()
        tools = {"nm": "/usr/bin/nm", "llvm-nm": None}

        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "                 U malloc@GLIBC_2.2.5\n", "")

        with mock.patch.object(shutil, "which", side_effect=tools.get), \
                mock.patch.object(subprocess, "run", side_effect=run) as run_mock:
            self.assertTrue(check("hookshot_vk.so"))

        self.assertEqual(run_mock.call_args.args[0][0], tools["nm"])


if __name__ == "__main__":
    unittest.main()
