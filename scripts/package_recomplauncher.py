"""Make a complete portable Windows release from the existing checked setup parts.

python -I scripts/package_recomplauncher.py dist/hookshot-setup.txt dist/RedTunic-vX-Windows-x64.zip
The list's freshly made parts are beside it; reused toolchain parts use the existing verified downloader.
Only release tools and source are packed, never a locally built game or work directory.
"""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_parts

ROOT = Path(__file__).resolve().parents[1]


def member_path(row, member):
    """Apply the setup list's destination/strip/rename rules without extracting it."""
    path = PurePosixPath(member)
    if path.is_absolute() or any(p in ('..', '.') or ':' in p or '\\' in p for p in path.parts):
        raise ValueError('Unsafe release member: ' + member)
    if row[3] == '.':
        mapped = path
    else:
        prefix = row[4] + '/'
        if not member.startswith(prefix):
            raise ValueError('Unexpected toolchain prefix: ' + member)
        mapped = PurePosixPath(row[3], row[5], member[len(prefix):])
    if not mapped.parts or mapped.parts[0] != 'HookshotBuilder':
        raise ValueError('Member outside the release folder: ' + member)
    if any(p == '..' or ':' in p or '\\' in p for p in mapped.parts):
        raise ValueError('Unsafe release destination: ' + str(mapped))
    rel = mapped.relative_to('HookshotBuilder').as_posix()
    if rel == 'HookshotBuilder.exe':
        rel = 'RedTunic.exe'
    return rel


def package(list_path, output):
    seen = set()
    output = Path(output)
    staged = output.with_suffix(output.suffix + '.new')
    try:
        with zipfile.ZipFile(staged, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as target:
            for row in release_parts.part_lines(str(list_path)):
                if len(row) != 7:
                    raise ValueError('Malformed release part')
                local = Path(list_path).parent / row[6].rsplit('/', 1)[-1]
                if not local.is_file():
                    local = Path(release_parts.fetch_part(row[1], int(row[2]), row[6]))
                if local.stat().st_size != int(row[2]) or release_parts.sha256(local) != row[1]:
                    raise ValueError('Release part checksum mismatch: ' + str(local))
                with zipfile.ZipFile(local) as source:
                    for item in source.infolist():
                        if item.is_dir():
                            continue
                        rel = member_path(row, item.filename)
                        if rel == 'recomplauncher.json':
                            continue
                        if rel.casefold() in seen or (item.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError('Duplicate or linked release member: ' + rel)
                        seen.add(rel.casefold())
                        with source.open(item) as src, target.open(rel, 'w', force_zip64=True) as dst:
                            shutil.copyfileobj(src, dst)
            required = {'RedTunic.exe', 'HookshotBuilder.dll', 'tools/python/python.exe',
                        'tools/nx2/make_game.py', 'tools/llvm-mingw/bin/clang.exe'}
            if missing := {p.casefold() for p in required} - seen:
                raise ValueError('Incomplete portable release: ' + ', '.join(sorted(missing)))
            target.write(ROOT / 'recomplauncher.json', 'recomplauncher.json')
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parts', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    package(args.parts, args.output)
