"""A source with a zero byte in it is refused before its compiler runs, named by its line and column, with what to do.

The translation never writes a zero byte into C. One there was changed after it was written (a drive or the memory): a
player's code_009.c of 26 MB had one where a "(" belonged, and clang's "extraneous ')' before ';'" was all the build said.
Runs nx2/build.py's own compile_one, cut out with ast, with a stand-in compiler: no game, no native compiler.

    python tests/test_compile_zero_byte.py
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'nx2/build.py'
WANTED = ('compile_one', 'cache_put')


class ZeroByteTests(unittest.TestCase):
    def setUp(self):
        source = ast.parse(BUILD.read_text(encoding='utf-8'))
        body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in WANTED]
        body[:0] = [node for node in source.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id in ('DEPFILE_DEPS', 'DEPFILE_PENDING') for t in node.targets)]
        self.temporary = tempfile.TemporaryDirectory(prefix='hookshot-zero-byte-')
        self.folder = Path(self.temporary.name)
        self.ran = self.folder / 'compiler-ran'
        self.compiler = self.folder / 'compiler.py'
        self.compiler.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text("ran")\n'
                                 'Path(sys.argv[-1]).write_bytes(b"authored object fixture")\n', encoding='utf-8')
        self.env = dict(os=os, subprocess=subprocess, time=time, GATE=None, PLANNED={},
                        OBJ_CACHE=str(self.folder / 'cache'), no_throttle=lambda handle: None, below_normal=lambda handle: None,
                        object_key=lambda src, flags, deps: hashlib.sha1(repr((src, flags, deps)).encode()).hexdigest())
        exec(compile(ast.Module(body=body, type_ignores=[]), str(BUILD), 'exec'), self.env)

    def tearDown(self):
        self.temporary.cleanup()

    def compile(self, text):
        src = self.folder / 'code_009.c'
        src.write_bytes(text)
        obj = self.folder / 'code_009.o'
        return src, obj, self.env['compile_one']((sys.executable, str(src), str(obj),
                                                  [str(self.compiler), str(self.ran)], b'authored deps'))   # (main's deps are bytes: DEPFILE_DEPS)

    def test_a_zero_byte_is_refused_by_its_place(self):
        src, obj, (name, _, error) = self.compile(b'void f(void) {\n    g(c,\0_b+_o),x8_);\n}\n')
        self.assertEqual(name, str(src))
        self.assertTrue(error, 'a source with a zero byte was compiled as if whole')
        last = error.strip().splitlines()[-1]   # (make_game's failure_reason names a stopped build by its last "error:" line)
        self.assertIn('%s:2:9: error:' % src, last)
        self.assertIn('zero byte', last)
        self.assertIn('Build again', last)
        self.assertFalse(self.ran.exists(), 'the compiler ran on a damaged source')
        self.assertFalse(obj.exists())
        self.assertFalse(Path(str(obj) + '.sha1').exists())
        self.assertFalse(Path(self.env['OBJ_CACHE']).exists())

    def test_a_whole_source_still_compiles(self):
        _, obj, (_, _, error) = self.compile(b'void f(void) {\n    g(c,(_b+_o),x8_);\n}\n')
        self.assertIsNone(error)
        self.assertTrue(self.ran.exists())
        self.assertEqual(obj.read_bytes(), b'authored object fixture')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=BUILD)
    args, rest = parser.parse_known_args()
    BUILD = args.build
    unittest.main(argv=[sys.argv[0], *rest])
