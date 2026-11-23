# RenHDrum

RenHDrum is the HD Rumble and DualSense haptics core the game's fine haptics come from: it turns the console's own
vibration values (a low and a high band, an amplitude and a frequency each, per side) into the four-channel 48 kHz audio
a DualSense's voice coils play on a USB cable. Copyright (c) 2026 Three-Dola, MIT OR Apache-2.0 (`LICENSE-MIT`,
`LICENSE-APACHE`, `NOTICE`).

These are the files of <https://github.com/trzz1214/RenHDrum> at the **v0.4.0** tag (commit
`806d1ed4dc92079d5a72fc63f83d422645ef562c`) that the library is built from, unmodified: `Cargo.toml`, `Cargo.lock` (no
dependencies), `src/`, `include/renhdrum.h` (the C ABI `host/rumble_renhdrum.c` binds at run time; `RENHDRUM_VERSION 0x000400`,
and 0x000401 keeps it), the licences and the notice. Its tests, examples, documentation and integrations are left out,
and so are the repository's GPL-3.0-or-later parts (`integrations/brontide`, `integrations/recomp-c/reference`).

upstream: https://github.com/trzz1214/RenHDrum/archive/refs/tags/v0.4.0.tar.gz
upstream-sha256: 072be092d8080465a9000237a8385e1a9de95697209fc37ef3dcc38e58a7dfb0
upstream-root: RenHDrum-0.4.0

**How it reaches the game.** `python -I scripts/renhdrum.py build` compiles it into `renhdrum.dll` (ignored by git) for
x86_64-pc-windows-gnullvm, with a pinned Rust and the llvm-mingw the game is built with; the same source gives the same
bytes. The Builder's release does that in CI (`.forgejo/workflows/build-windows.yml`) and ships the DLL in its package as
`tools/third_party/renhdrum/renhdrum.dll`, and the AppImage's (`.forgejo/workflows/build-linux.yml`) builds `renhdrum.so`
and carries it in `third_party/renhdrum/`, so a player's build puts it beside the game (`nx2/build.py`,
`nx2/make_game.py`) and needs no Rust. `host/rumble_renhdrum.c` loads it (`renhdrum.dll`, `renhdrum.so` on Linux) from the game's own folder by full path; without it
every controller keeps its motors or its own HD rumble, and Test vibration says why.

**A newer RenHDrum.** Replace these files with the new tag's, change the commit and the three `upstream` lines, run
`python -I scripts/check_third_party.py --update renhdrum` and `--upstream renhdrum`, and check `host/rumble_renhdrum.c`'s
`RH_ABI` against the new `include/renhdrum.h`.

tree-sha256: 6162b90e0178015a718ae96a5e7d63ac15b8f14454178eca9b442da309a213d7
