"""What the release workflows need from earlier releases, and the ReLink package folder made in CI
(.github/workflows/windows.yml). No game data is involved: the parts hold the builder, its tools
and the play profile (addresses and counts made on the development PC).

    python -I scripts/release_parts.py previous <out list> --repo <owner/name> [--repo ...] [--skip-tag <tag>] [--seed <name>]
        the newest stable release's relink-setup.txt (of the first repository that has one; not --skip-tag's, the release
        being made); prints that release's tag. Exits 1 when no repository has one. --seed: the list of that name in the
        first repository's builder-parts release wins when it is there (scripts/make_seed.py: the maintainer's own parts)
    python -I scripts/release_parts.py adopt <setup list> <new list> <out folder> --base <url>
        the list's Python, emsdk, CPM and wgpu-native parts repacked under this project's name into <out folder>, and the
        list written again with them (a first release has no parts of its own: it starts from another release's)
    python -I scripts/release_parts.py profile <setup list> <work folder>
        the play profile (tools/work of the list's profile part, or of its app part) into <work folder>
    python -I scripts/release_parts.py llvm-mingw <folder> [--name]
        llvm-mingw as scripts/make_setup_parts.py names it (its official release, SHA-256 checked) unpacked into <folder>;
        prints its folder. --name: only prints its name (a cache key)
    python -I scripts/release_parts.py package <out folder> --publish <dotnet publish folder> --setup <setup exe>
                                       [--list <setup list>]
        the package folder make_setup_parts.py splits: the published launcher, the setup, LICENSE, the tracked assets, gpu, host, nx2
        and scripts under tools/, tools/work (the profile) and tools/third_party: the tracked files and, from the list's
        third_party part, the prebuilt files the repository does not carry (a64emit.dll, lld-link.exe, SDL2.dll, SDL3.dll,
        zbic.dll; rebuilt on the player's PC when their sources' stamps differ). Python, emsdk, CPM and wgpu-native are not
        in it: make_setup_parts.py --reuse takes their parts from the list.

GitHub's API is read with $GH_TOKEN when it is set. Downloads are retried (GitHub answers 5xx now and then). Progress goes to
stderr: what a command prints on stdout is its answer.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir(), "release-parts")
# (the prebuilt files of the third_party part: what it holds and git does not)
PREBUILT_EXT = (".dll", ".lib", ".exe", ".sha1")


def get(url, api=False, tries=5):
    headers = {"User-Agent": "relink-ci"}
    if api:
        headers["Accept"] = "application/vnd.github+json"
        if os.environ.get("GH_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
        print("  %s: retrying in 10 s" % url, file=sys.stderr, flush=True)
        time.sleep(10)


def previous(a):
    if a.seed:
        try:
            parts = json.loads(get("https://api.github.com/repos/%s/releases/tags/builder-parts" % a.repo[0], api=True))
            url = next((x["browser_download_url"] for x in parts["assets"] if x["name"] == a.seed), None)
        except urllib.error.HTTPError as e:
            print("%s: builder-parts: %s" % (a.repo[0], e), file=sys.stderr)
            url = None
        if url:
            with open(a.out, "wb") as f:
                f.write(get(url))
            print("seed")
            return 0
        print("no %s in %s's builder-parts release: the releases' lists instead" % (a.seed, a.repo[0]), file=sys.stderr)
    for repo in a.repo:
        try:
            releases = json.loads(get("https://api.github.com/repos/%s/releases?per_page=30" % repo, api=True))
        except urllib.error.HTTPError as e:
            print("%s: %s" % (repo, e), file=sys.stderr)
            continue
        for r in releases:
            if r["draft"] or r["prerelease"] or r["tag_name"] == a.skip_tag:
                continue
            url = next((x["browser_download_url"] for x in r["assets"] if x["name"] in a.list_name), None)
            if url:
                with open(a.out, "wb") as f:
                    f.write(get(url))
                print(r["tag_name"])
                return 0
    print("no release with a %s in %s" % (" or ".join(a.list_name), ", ".join(a.repo)), file=sys.stderr)
    return 1


def part_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.split() for l in f.read().splitlines() if l.startswith("part ")]


def fetch_part(sha, size, url):
    """A part downloaded once into the cache, its size and SHA-256 checked"""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, url.rsplit("/", 1)[-1])
    if not (os.path.isfile(path) and os.path.getsize(path) == size and sha256(path) == sha):
        print("downloading %s (%d MB)" % (url, size >> 20), file=sys.stderr, flush=True)
        data = get(url)
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            sys.exit("%s: not the size and SHA-256 the list gives" % url)
        with open(path + ".new", "wb") as f:
            f.write(data)
        os.replace(path + ".new", path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def part_name(url):
    """python, emsdk, app ...: the middle of <prefix>-<name>-<start of its SHA-256>.zip (the prefix is whatever the release
    that made the part called itself)"""
    base = url.rsplit("/", 1)[-1]
    return base.split("-", 1)[1].rsplit("-", 1)[0] if base.count("-") >= 2 else ""


def named_part(list_path, name):
    for w in part_lines(list_path):
        if len(w) == 7 and part_name(w[6]) == name:
            return fetch_part(w[1], int(w[2]), w[6])
    return None


def top_folder(zip_path):
    """The one folder a part's files are under"""
    with zipfile.ZipFile(zip_path) as z:
        return z.namelist()[0].split("/", 1)[0]


def repack(src, dst, top):
    """src's files under another top folder, as deterministic as make_setup_parts.py's zips; (sha256, size)"""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst + ".new", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for m in sorted(zin.infolist(), key=lambda m: m.filename):
            if m.is_dir() or "/" not in m.filename:
                continue
            info = zipfile.ZipInfo(top + "/" + m.filename.split("/", 1)[1], date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zout.writestr(info, zin.read(m), compresslevel=6)
    os.replace(dst + ".new", dst)
    return sha256(dst), os.path.getsize(dst)


# (the tools a package made in CI does not make itself: make_setup_parts.py --reuse takes their parts from the list)
MIRRORED = ("python", "emsdk", "cpm", "wgpu-native")


def adopt(a):
    """A list whose tool parts were made under another name: each of them downloaded, repacked under ReLink/ and written to
    <out folder> as ReLink-<name>-<sha>.zip (to upload to this repository's builder-parts release), and the list written again
    with those. The other lines stay: the app and third_party parts are only read here (the play profile, the prebuilt files)."""
    os.makedirs(a.out, exist_ok=True)
    with open(a.list, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out_lines, n = [], 0
    for line in lines:
        w = line.split()
        if len(w) == 7 and w[0] == "part":
            base, name = w[6].rsplit("/", 1)[-1], part_name(w[6])
            if name in MIRRORED and not base.startswith("ReLink-"):
                src = fetch_part(w[1], int(w[2]), w[6])
                tmp = os.path.join(a.out, "ReLink-%s.zip" % name)
                sha, size = repack(src, tmp, "ReLink")
                named = os.path.join(a.out, "ReLink-%s-%s.zip" % (name, sha[:12]))
                os.replace(tmp, named)
                line = "part %s %d %s %s %s %s/%s" % (sha, size, w[3], w[4], w[5], a.base.rstrip("/"), os.path.basename(named))
                print("%-12s %5d MB  %s" % (name, size >> 20, sha[:16]), file=sys.stderr, flush=True)
                n += 1
        out_lines.append(line)
    with open(a.new_list, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out_lines) + "\n")
    print("%d parts adopted" % n)
    return 0


def extract(zip_path, prefix, dest, keep=lambda rel: True):
    """The zip's members under prefix (ReLink/...), into dest without that prefix; how many"""
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            if m.is_dir() or not m.filename.startswith(prefix):
                continue
            rel = m.filename[len(prefix):]
            if ".." in rel.split("/") or not keep(rel):
                continue
            out = os.path.join(dest, *rel.split("/"))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f)
            n += 1
    return n


def profile(a):
    # (a seed has the profile as a part of its own; a release's list has it in its app part)
    app = named_part(a.list, "profile") or named_part(a.list, "app")
    if not app:
        sys.exit("no profile or app part in %s" % a.list)
    n = extract(app, top_folder(app) + "/tools/work/", a.work)
    print("%d profile files into %s" % (n, a.work))
    return 0


def llvm_spec():
    spec = importlib.util.spec_from_file_location("make_setup_parts", os.path.join(ROOT, "scripts", "make_setup_parts.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.LLVM   # (sha256, size, top folder in the zip, url)


def llvm_mingw(a):
    sha, size, name, url = llvm_spec()
    if a.name:
        print(name)
        return 0
    folder = os.path.join(a.folder, name)
    if not os.path.isfile(os.path.join(folder, "bin", "clang.exe")):
        z = fetch_part(sha, size, url)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(a.folder)
    print(folder)
    return 0


# SDL, for the controllers that are not Xbox pads (host/pad_sdl.c: Switch Pro, DualSense, DualShock 4 ...) and the
# microphones: SDL2.dll (sdl2-compat, the SDL2 calls on SDL3) and the SDL3.dll it loads, beside the game (nx2/build.py copies
# them from third_party/sdl2). The repository carries no binaries and the parts took them from the release before; the seed
# this project's releases started from had none, so v0.6.1 shipped without them and a Switch Pro Controller did nothing in
# the game (issue 3; the game's log said "no SDL beside the exe or on the system: only XInput controllers (Xbox pads) work").
# Taken here from SDL's own releases, the versions third_party/sdl2/VERSION.txt names, by the size and SHA-256 GitHub
# publishes for each archive: (file in the archive, SHA-256 of the archive, its size, its address)
SDL = (
    ("SDL3.dll", "368bed8b24e44ad9ea372990814f2343fcfb9e468abb7f8f4a3bc146c7304752", 1166032,
     "https://github.com/libsdl-org/SDL/releases/download/release-3.4.20/SDL3-3.4.20-win32-x64.zip"),
    ("SDL2.dll", "94a0184bc2d0670151a023656fb58e8f722fc001bda3a6a424234c88c5501cc0", 1273703,
     "https://github.com/libsdl-org/sdl2-compat/releases/download/release-2.32.74/sdl2-compat-2.32.74-win32-x64.zip"),
)


def sdl_libraries(dest):
    """SDL2.dll and SDL3.dll into dest (tools/third_party/sdl2) where they are not there already; how many were added"""
    n = 0
    for name, sha, size, url in SDL:
        out = os.path.join(dest, name)
        if os.path.isfile(out):   # (the release before had it: the third_party part's prebuilt files)
            continue
        with zipfile.ZipFile(fetch_part(sha, size, url)) as z:
            member = next((m for m in z.namelist() if m.rsplit("/", 1)[-1] == name), None)
            if not member:
                sys.exit("%s: no %s in it" % (url, name))
            data = z.read(member)
        os.makedirs(dest, exist_ok=True)
        with open(out, "wb") as f:
            f.write(data)
        print("%s (%d KB) from %s" % (name, len(data) >> 10, url.rsplit("/", 1)[-1]))
        n += 1
    return n


def tracked(*paths):
    r = subprocess.run(["git", "ls-files", "-z", "--"] + list(paths), cwd=ROOT, check=True, stdout=subprocess.PIPE)
    return [f for f in r.stdout.decode("utf-8").split("\0") if f]


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def package(a):
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(a.publish, out)
    copy(a.setup, os.path.join(out, "ReLink-Setup.exe"))   # (the builder's Repair runs it)
    copy(os.path.join(ROOT, "LICENSE"), os.path.join(out, "LICENSE"))
    tools = os.path.join(out, "tools")
    for f in tracked("assets", "gpu", "host", "nx2", "scripts", "third_party"):
        copy(os.path.join(ROOT, f), os.path.join(tools, *f.split("/")))
    if a.list:
        if not profile(argparse.Namespace(list=a.list, work=os.path.join(tools, "work"))) == 0:
            return 1
        tp = named_part(a.list, "third_party")
        if tp:
            known = set(tracked("third_party"))
            prebuilt = lambda rel: rel.startswith("third_party/") and rel not in known and rel.lower().endswith(PREBUILT_EXT)
            n = extract(tp, top_folder(tp) + "/tools/", tools, keep=prebuilt)
            print("%d prebuilt third_party files from %s" % (n, os.path.basename(tp)))
    else:
        print("no setup list: the package has no play profile and no prebuilt third_party files")
    sdl_libraries(os.path.join(tools, "third_party", "sdl2"))
    print("package folder: %s" % out)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("previous")
    s.add_argument("out")
    s.add_argument("--repo", action="append", required=True)
    s.add_argument("--skip-tag", default="")
    s.add_argument("--list-name", action="append", help="the list's file name in a release (default relink-setup.txt; may be given more than once)")
    s.add_argument("--seed", default="", help="a list's file name in the first repository's builder-parts release: it wins when it is there")
    s = sub.add_parser("adopt")
    s.add_argument("list")
    s.add_argument("new_list")
    s.add_argument("out")
    s.add_argument("--base", required=True, help="where this repository serves its parts from (its builder-parts release)")
    s = sub.add_parser("profile")
    s.add_argument("list")
    s.add_argument("work")
    s = sub.add_parser("llvm-mingw")
    s.add_argument("folder", nargs="?", default="")
    s.add_argument("--name", action="store_true")
    s = sub.add_parser("package")
    s.add_argument("out")
    s.add_argument("--publish", required=True)
    s.add_argument("--setup", required=True)
    s.add_argument("--list")
    a = p.parse_args()
    if a.cmd == "previous":
        a.list_name = a.list_name or ["relink-setup.txt"]
    return {"previous": previous, "adopt": adopt, "profile": profile, "llvm-mingw": llvm_mingw, "package": package}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
