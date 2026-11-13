# Publishing a release

A release is what players install from: `ReLink-Setup.exe` and the list it reads, `relink-setup.txt`. Both are made by the
**Windows** workflow (`.github/workflows/windows.yml`) and attached to the tag's release; the parts the list names go to one
release of their own, `builder-parts`. No game data is involved at any step.

## Each release

1. Set the version in `launcher/ReLink.csproj` (`<Version>`), commit, and push to `main`. The workflow runs on every push and
   leaves its files as an artifact (`ReLink-windows`): a failed run here is found before the tag.
2. Tag the commit `vX.Y.Z` and push the tag.
3. The workflow builds the package again for the tag, uploads the new parts to `builder-parts`, and creates the release with
   `ReLink-Setup.exe` and `relink-setup.txt`. A tag with `-rc` is a prerelease and never "latest".

Players' launchers then offer the update: they compare the game folder's version with the latest release's tag.

## The seed: the first release, new tools, a new play profile

The setup installs six parts. Three are made from this repository at every release: the launcher with the sources and the
play profile (`app`), .NET's runtime (`dotnet`) and `third_party`. llvm-mingw, the compiler, comes from its own official
release. The other two are tools that change rarely and are not rebuilt in CI: Python with the build's modules (`python`)
and the Vulkan headers with FFmpeg's libraries (`cpm`). Each release takes those two, and the play profile, from the
release before it.

A repository with no release yet has nothing to take them from, and a new play profile or a new Python has to come from
somewhere too. That is the seed, made on the maintainer's PC by `scripts/make_seed.py` from three folders:

- `--python`: python.org's embeddable package for Windows (3.12), unpacked, with `Lib\site-packages` and `import site` added
  to its `python312._pth`, and `pip install --target <folder>\Lib\site-packages lz4 numpy capstone "unicorn>=2.1,<2.2"
  zstandard` (delete the `bin` folder pip adds: its launchers hold the path of the Python that ran pip).
- `--cpm`: from a CPM cache, `vulkanheaders/<version>` and `spirv-headers/<version>` (their `include` and license files),
  `vulkanmemoryallocator/<version>` (`include`, license) and `ffmpeg/<version>` (`include`, and `avcodec`, `avutil`,
  `swresample` and `swscale` of `lib`; the build used so far is Takoroka/ffmpeg-av1's 9.0.2-av1-1 for Windows).
- `--work`: the play profile, `hot_funcs.txt`, `relink_pieces.prof` and `image/chunks.txt` of the build folder the profile
  was made in. Addresses and counts; no game code. And `pipeline_states.bin`, a copy of `relink_pipeline_states.bin` from a
  game folder that was played through many places: the pipelines the game asked for there (hashes, formats and draw state).
  A player's first start builds them before the title, from the player's own shaders; without it every place stutters on
  its first visit while the driver compiles.

It writes three zips and `relink-seed.txt`. Upload the four files to the `builder-parts` release (a prerelease, made by hand
the first time), then set the repository variable `SEED_LIST` (Settings, Secrets and variables, Actions, Variables) to
`relink-seed.txt`. While the variable is set, the workflow starts from the seed instead of the latest release's list.
Remove it once a release made from the seed is out: the next releases then chain from that one.

The repository must be public for players to download a release's files, and for the workflow to read the seed.

## Checking a release before it is the latest

`ReLink-Setup.exe` reads two environment variables, for tests:

- `RELINK_SETUP_LIST=<url>`: another list than the latest release's.
- `RELINK_SETUP_BASE=<folder>`: install there instead of `%LOCALAPPDATA%\ReLink`.

The launcher has its own: `RELINK_LATEST_TAG=<tag>` (or `none`) stands in for GitHub's answer, `RELINK_NO_SEARCH=1` skips the
search for `.nup` files, `RELINK_LANG=fr|en` chooses its language, and
`ReLink.exe --snapshot out.png --view install|build|update|ready|problem|home|home-update|smooth|game|controls|playing`
writes a picture of one page without running anything.
