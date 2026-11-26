"""Build a launcher zip from authored parts: no network, game or SDK downloads."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launcher_package', ROOT / 'scripts/package_recomplauncher.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PortablePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='launcher-package-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.rows = []
        self.output = self.folder / 'RedTunic-Windows-x64.zip'

    def part(self, name, files, destination='.', strip='-', rename='-'):
        path = self.folder / (name + '.zip')
        with zipfile.ZipFile(path, 'w') as archive:
            for member, content in files.items():
                archive.writestr(member, content)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.rows.append(f'part {sha} {path.stat().st_size} {destination} {strip} {rename} https://example.invalid/{path.name}')

    def build(self):
        listing = self.folder / 'hookshot-setup.txt'
        listing.write_text('\n'.join(self.rows), encoding='utf8')
        with mock.patch.object(package.release_parts, 'fetch_part', side_effect=AssertionError('Unexpected network')):
            package.package(listing, self.output)

    def complete(self):
        self.part('app', {'HookshotBuilder/HookshotBuilder.exe': b'authored apphost',
                          'HookshotBuilder/HookshotBuilder.dll': b'authored managed app',
                          'HookshotBuilder/tools/nx2/make_game.py': b'authored build tool'})
        self.part('python', {'HookshotBuilder/tools/python/python.exe': b'authored python'})
        self.part('llvm', {'llvm-release/bin/clang.exe': b'authored compiler'},
                  'HookshotBuilder/tools', 'llvm-release', 'llvm-mingw')

    def test_complete_zip_runs_the_published_program_with_manifest_at_root(self):
        self.complete()
        self.build()
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read('RedTunic.exe'), b'authored apphost')
            self.assertIn('tools/llvm-mingw/bin/clang.exe', archive.namelist())
            self.assertEqual(archive.read('recomplauncher.json'), (ROOT / 'recomplauncher.json').read_bytes())
            self.assertNotIn('HookshotBuilder.exe', archive.namelist())
            self.assertFalse(any(n.startswith('recompdata/') for n in archive.namelist()))

    def test_incomplete_parts_do_not_replace_an_existing_release(self):
        self.output.write_bytes(b'previous release')
        self.part('app', {'HookshotBuilder/HookshotBuilder.exe': b'app'})
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.build()
        self.assertEqual(self.output.read_bytes(), b'previous release')

    def test_corrupt_part_is_refused(self):
        self.complete()
        with (self.folder / 'app.zip').open('ab') as file:
            file.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_windows_case_collisions_are_refused(self):
        self.complete()
        self.part('collision', {'HookshotBuilder/REDTUNIC.exe': b'other'})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.build()

    def test_traversal_is_refused(self):
        self.complete()
        self.part('bad', {'HookshotBuilder/../../outside': b'outside'})
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.build()


if __name__ == '__main__':
    unittest.main()
