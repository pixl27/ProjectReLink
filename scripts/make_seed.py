"""The seed of a release, made on the maintainer's PC: the parts CI does not make and no earlier release holds yet.

    python -I scripts/make_seed.py <out folder> --python <folder> --cpm <folder> --work <folder> [--base <url>]

Writes <out>/relink-seed.txt and three parts, each a deterministic zip named by its SHA-256 as scripts/make_setup_parts.py's:

    ReLink-python-<sha>.zip    tools/python: python.org's embeddable package for Windows with the build's modules in
                               Lib/site-packages (lz4, numpy, capstone, unicorn 2.1, zstandard) and its python312._pth
                               naming that folder
    ReLink-cpm-<sha>.zip       tools/cpm: the CPM cache's vulkanheaders, spirv-headers, vulkanmemoryallocator and ffmpeg
                               folders (headers, and FFmpeg's static libraries)
    ReLink-profile-<sha>.zip   tools/work: the play profile (hot_funcs.txt, relink_pieces.prof, image/chunks.txt: addresses
                               and counts made on the development PC, no game code)

Upload the three zips and the list to the builder-parts release and set the repository variable SEED_LIST to the list's
name there: the Windows workflow then starts from it instead of the latest release's list (docs/releasing.md). No game data
is involved.
"""
import argparse
import hashlib
import importlib.util
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def setup_parts():
    spec = importlib.util.spec_from_file_location("make_setup_parts", os.path.join(HERE, "make_setup_parts.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def part(m, out, base, name, folder, under):
    """<folder>'s files as ReLink/tools/<under>/... in <out>/ReLink-<name>-<sha>.zip; the line the list gives it"""
    folder = os.path.abspath(folder)
    names = []
    for d, ds, fs in os.walk(folder):
        ds[:] = sorted(x for x in ds if x not in m.SKIP_DIRS)
        names += [os.path.relpath(os.path.join(d, f), folder).replace(os.sep, "/") for f in fs
                  if os.path.splitext(f)[1].lower() not in m.SKIP_EXT]
    if not names:
        sys.exit("%s: no files" % folder)
    tmp = os.path.join(out, "ReLink-%s.zip" % name)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in sorted(names):
            info = zipfile.ZipInfo("ReLink/tools/%s/%s" % (under, n), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(os.path.join(folder, n.replace("/", os.sep)), "rb") as f:
                z.writestr(info, f.read(), compresslevel=6)
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    sha, size = h.hexdigest(), os.path.getsize(tmp)
    named = os.path.join(out, "ReLink-%s-%s.zip" % (name, sha[:12]))
    os.replace(tmp, named)
    print("%-10s %5d files %5d MB  %s" % (name, len(names), size >> 20, sha[:16]), flush=True)
    return "part %s %d . - - %s/%s" % (sha, size, base, os.path.basename(named))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--python", required=True)
    p.add_argument("--cpm", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--base", default="https://github.com/pixl27/ProjectReLink/releases/download/builder-parts")
    a = p.parse_args()
    if not os.path.isfile(os.path.join(a.python, "python.exe")):
        sys.exit("%s: no python.exe" % a.python)
    for f in ("hot_funcs.txt", "relink_pieces.prof"):
        if not os.path.isfile(os.path.join(a.work, f)):
            sys.exit("%s: no %s" % (a.work, f))
    m = setup_parts()
    base = a.base.rstrip("/")
    os.makedirs(a.out, exist_ok=True)
    lines = ["tag seed",
             part(m, a.out, base, "python", a.python, "python"),
             part(m, a.out, base, "cpm", a.cpm, "cpm"),
             part(m, a.out, base, "profile", a.work, "work")]
    with open(os.path.join(a.out, "relink-seed.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
