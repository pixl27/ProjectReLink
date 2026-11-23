"""The Builders' Version menu (2026-11-15, the owner: "a version chooser that dynamically gets absolutely every release from the
git"): every release on git.notrexed.dev, read page by page from Forgejo's API when the menu opens, newest first; a release chosen
there is what the builder installs and builds the game with (as a switch of track installs that track's newest), saved, until
Newest or another is chosen.

Checked: the Linux Builder's list (linux_update.releases_in, all_releases: drafts, the parts' and rolling releases and repeated
tags left out, every page read, a page failing part way is no list), the Windows Builder's ReleaseList.cs under .NET 8 on the same
pages (when dotnet is there; which can be installed: its setup and list, from v2.0.6), and both windows' wiring by source.

    python -I -B tests/test_builder_version_menu.py [--dotnet dotnet]
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--dotnet", "{dotnet}"]

ROOT = Path(__file__).resolve().parents[1]


def release(tag, assets=("HookshotBuilder.exe", "hookshot-setup.txt", "hookshot-linux.txt"), pre=False, draft=False,
            published="2026-11-15T06:15:00Z"):
    return {"tag_name": tag, "prerelease": pre, "draft": draft, "published_at": published,
            "assets": [{"name": n} for n in assets]}


# (page 1 full, 50 releases: the 4 real ones, the file-holding ones, a draft, then 41 old versions; page 2 the rest)
PAGE1 = [release("v2.3.1"), release("v2.3.0"), release("v2.2.0-rc.1", pre=True), release("rc", ("Hookshot-x86_64.AppImage",), pre=True),
         release("v2.1.0"), release("linux-parts", ("a.zip",), pre=True), release("builder-parts", ("b.zip",), pre=True),
         release("v2.4.0", draft=True), release("v2.3.1")]   # (a draft, and a tag twice)
PAGE1 += [release("v1.%d.0" % i, ("HookshotBuilder.exe", "hookshot-setup.txt")) for i in range(50 - len(PAGE1))]
PAGE2 = [release("v2.0.6"), release("v2.0.5"), release("v2.2.0-rc.2", ("hookshot-linux.txt",), pre=True)]
WANT_FIRST = ["v2.3.1", "v2.3.0", "v2.2.0-rc.2", "v2.2.0-rc.1", "v2.1.0", "v2.0.6", "v2.0.5"]


def linux_update():
    import sys
    sys.path.insert(0, str(ROOT / "builder/linux"))   # (its part_links, beside it)
    spec = importlib.util.spec_from_file_location("linux_update_under_test", ROOT / "builder/linux/linux_update.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_linux():
    lu = linux_update()
    asked = []

    def get(url):
        asked.append(url)
        page = int(url.rsplit("page=", 1)[1])
        return json.dumps(PAGE1 if page == 1 else PAGE2 if page == 2 else []).encode()
    got = lu.all_releases(get)
    assert asked == [lu.RELEASES_PAGE % 1, lu.RELEASES_PAGE % 2], asked   # (page 2 short: no page 3)
    tags = [r["tag"] for r in got]
    assert tags[:len(WANT_FIRST)] == WANT_FIRST, tags[:10]
    assert len(tags) == len(set(tags)) == len(WANT_FIRST) + (50 - 9), len(tags)
    for gone in ("rc", "linux-parts", "builder-parts", "v2.4.0"):
        assert gone not in tags, gone
    by = {r["tag"]: r for r in got}
    assert by["v2.2.0-rc.1"]["prerelease"] and not by["v2.3.1"]["prerelease"]
    assert by["v2.3.1"]["installable"] and not by["v1.0.0"]["installable"]   # (no hookshot-linux.txt)
    assert by["v2.3.1"]["published"] == "2026-11-15"
    assert lu.release_label(by["v2.2.0-rc.1"]) == "v2.2.0-rc.1 · 2026-11-15 · release candidate"
    assert lu.release_label(by["v1.0.0"]).endswith("can't be installed from here")

    def broken(url):
        if url.endswith("page=2"):
            raise OSError("reset")
        return json.dumps(PAGE1).encode()
    assert lu.all_releases(broken) is None, "a list cut short is no list"
    assert lu.all_releases(lambda url: (_ for _ in ()).throw(OSError("down"))) is None

    tk = (ROOT / "builder/linux/hookshot_builder.py").read_text(encoding="utf-8")
    for need in ('self.pinned = self.cfg.get("version") or None', "postcommand=self.load_versions",
                 'self.lines.put(("versions", linux_update.all_releases()))', 'if self.pinned:   # (a version chosen',
                 'return bool(getattr(self, "pinned", None)) or self.track_id() != linux_update.track_of(self.tools)',
                 'self.cfg["version"] = tag', 'self.pinned = self.cfg["version"] = None', 'elif kind == "versions":'):
        assert need in tk, "hookshot_builder.py: " + need


def check_windows_source():
    cs = (ROOT / "builder/HookshotBuilder/MainWindow.xaml.cs").read_text(encoding="utf-8")
    for need in ("bool Switching => Tools.Find()?.RedTunicRelease != true && (pinned != null || Track != Updater.TrackOf(Tools.Find()?.Version));",
                 "async Task<string?> NewestTag() =>", "pinned ?? (Environment.GetEnvironmentVariable(\"HOOKSHOT_LATEST_TAG\")",
                 "releases = await ReleaseList.Fetch();", "VersionBox.DropDownOpened += VersionsOpened;",
                 "updateTarget = Tools.Find()?.RedTunicRelease == true ? builder : pinned ?? Updater.Target(latestTag, builder, Track);",
                 "updateTarget = tools?.RedTunicRelease == true ? tools.Version : pinned ?? Updater.Target(latestTag, tools?.Version, Track);",
                 "Version = pinned };", "pinned = null;   // (a track chosen"):
        assert need in cs, "MainWindow.xaml.cs: " + need
    assert "await Updater.LatestTag(Track)" in cs and cs.count("await Updater.LatestTag(Track)") == 1, \
        "every lookup of the newest goes through NewestTag (the chosen version first)"
    xaml = (ROOT / "builder/HookshotBuilder/MainWindow.xaml").read_text(encoding="utf-8")
    assert 'x:Name="VersionBox"' in xaml and 'Text="VERSION"' in xaml
    settings = (ROOT / "builder/HookshotBuilder/BuilderSettings.cs").read_text(encoding="utf-8")
    assert "public string? Version { get; set; }" in settings


PROGRAM = r'''
using HookshotBuilder;
static void Check(bool ok, string what) { if (!ok) { Console.Error.WriteLine("FAIL: " + what); Environment.Exit(1); } }
var pages = System.IO.File.ReadAllLines("pages.txt");
var list = ReleaseList.From(pages);
var tags = list.Select(r => r.Tag).ToList();
var want = new[] { %WANT% };
Check(tags.Take(want.Length).SequenceEqual(want), "order: " + string.Join(",", tags.Take(10)));
Check(tags.Count == tags.Distinct().Count() && tags.Count == want.Length + 41, "each once, the rest kept: " + tags.Count);
Check(!tags.Contains("rc") && !tags.Contains("linux-parts") && !tags.Contains("v2.4.0"), "files' releases and drafts left out");
var by = list.ToDictionary(r => r.Tag);
Check(by["v2.3.1"].Installable && by["v2.0.6"].Installable, "installable");
Check(!by["v2.0.5"].Installable, "v2.0.5's setup takes no --tag");
Check(!by["v2.2.0-rc.2"].Installable, "no setup, no install");
Check(by["v1.0.0"].Installable == false, "older than v2.0.6");
Check(by["v2.2.0-rc.1"].Label == "v2.2.0-rc.1 · 15 Nov 2026 · release candidate", "label: " + by["v2.2.0-rc.1"].Label);
Check(by["v2.0.5"].Label.EndsWith("can't be installed from here"), "label of one not installable");
Console.WriteLine("PASS: ReleaseList.From");
'''


def check_dotnet(dotnet):
    with tempfile.TemporaryDirectory(prefix="builder-version-menu-") as folder:
        root = Path(folder)
        (root / "pages.txt").write_text(json.dumps(PAGE1) + "\n" + json.dumps(PAGE2) + "\n", encoding="utf-8")
        (root / "Program.cs").write_text(PROGRAM.replace("%WANT%", ", ".join('"%s"' % t for t in WANT_FIRST)), encoding="utf-8")
        shutil.copyfile(ROOT / "builder/HookshotBuilder/ReleaseList.cs", root / "ReleaseList.cs")
        (root / "NuGet.Config").write_text("<configuration><packageSources><clear/></packageSources></configuration>", encoding="utf-8")
        (root / "Fixture.csproj").write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>
<OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings>
<Nullable>enable</Nullable><TreatWarningsAsErrors>true</TreatWarningsAsErrors><NuGetAudit>false</NuGetAudit>
</PropertyGroup><ItemGroup><None Include="pages.txt" CopyToOutputDirectory="PreserveNewest"/></ItemGroup></Project>''',
                                                 encoding="utf-8")
        subprocess.run([dotnet, "run", "--project", str(root / "Fixture.csproj"), "--configuration", "Release"], check=True, cwd=root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dotnet", default="dotnet")
    a = p.parse_args()
    check_linux()
    check_windows_source()
    dotnet = shutil.which(a.dotnet)
    if dotnet:
        check_dotnet(dotnet)
    else:
        print("(no dotnet: the Windows Builder's ReleaseList was checked by source only)")
    print("PASS: the Version menu lists every release and installs the one chosen")


if __name__ == "__main__":
    main()
