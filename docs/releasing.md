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

## The first release

The setup installs seven parts. Two are made from this repository at every release (the launcher with the sources, and .NET's
runtime) and one from the files listed here (third_party). The other four, Python, emsdk, the CPM packages and wgpu-native,
are tools that change rarely and are not rebuilt in CI: each release takes them from the release before it.

A repository with no release yet has nothing to take them from. Set two repository variables (Settings, Secrets and variables,
Actions, Variables) before the first tag:

- `BOOTSTRAP_REPO`: `owner/name` of a repository whose latest stable release holds a setup list with those parts.
- `BOOTSTRAP_LIST`: that list's file name in its releases.

The workflow then downloads those four parts, repacks them under this project's name (`scripts/release_parts.py adopt`) and
uploads them to this repository's `builder-parts` release, so players only ever download from here. The play profile and the
prebuilt third_party files come from that same release. Once a first release exists, the variables can be removed.

The repository must be public for players to download a release's files.

## Checking a release before it is the latest

`ReLink-Setup.exe` reads two environment variables, for tests:

- `HOOKSHOT_SETUP_LIST=<url>`: another list than the latest release's.
- `HOOKSHOT_SETUP_BASE=<folder>`: install there instead of `%LOCALAPPDATA%\ReLink`.

The launcher has its own: `RELINK_LATEST_TAG=<tag>` (or `none`) stands in for GitHub's answer, `RELINK_NO_SEARCH=1` skips the
search for `.nup` files, `RELINK_LANG=fr|en` chooses its language, and
`ReLink.exe --snapshot out.png --view install|build|update|ready|problem|home|home-update|smooth|game|controls|playing`
writes a picture of one page without running anything.
