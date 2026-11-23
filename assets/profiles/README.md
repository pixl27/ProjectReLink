# Play profiles

Profiles the build compiles with, committed so every build of a checkout uses the same ones: the release's builders,
a player's own build (with DLSS or FSR 4 too), and a developer's. Names of Hookshot's own functions, a hash of each
one's shape and how often its branches ran, and the guest addresses of the game's hottest functions: no game code or
data.

## hot_funcs.txt: the hot list

The lifted game functions the game spends its time in, one a line, `f_<guest address>`, hottest first (the lift's
`A64_HOT_FUNCS`). The lift puts them in `code_hot.c`, their float helpers inline, and the build compiles them as the eight
hot pieces at `-O2` with their hot callees inlined, where every other lifted file is built at `-O1` (docs/development.md,
"The hot list"). `nx2/make_game.py` (`hot_list_for`) builds with it unless the checkout's own `work/hot_funcs.txt`, a list
`A64_HOT_FUNCS` names in the shell, or `HOOKSHOT_HOT_LIST=0` (no list) says otherwise: copied into the build's work
folder, or read where it is when that folder is the checkout's own `work/`. The AppImage and the Windows builder's tools
folder carry it, so a player's build uses it too. No play profile of the hot pieces (`work/hookshot_pieces.prof`,
`RX_PGO`) comes with it: they are built at `-O2` without one.

It is a list of guest addresses, so it changes mostly with the game's version. The translator can still move it: the
lift decides where a function starts and which giant functions are split into pieces, and a function's time can move to
the host (an entry hook). The lift's line `hot functions (A64_HOT_FUNCS): N of 680 listed found` shows that drift (a name
it does not find is passed over). Make it again when the game or the translator changes where the game spends its time.

Recorded: 680 functions, for the game's version 1.1.1 (build `e0f3ccfa`), from Hookshot's own lift of main 544229a2
(v3.0.0-rc.7; the test builds carried unmerged pull requests too, their switches off) on Windows 11, a Ryzen 9 5900X (12
cores, 24 threads) and an RTX 3090, at 3840x2160, frame rate uncapped, with the all-thread sampler (`HOS_PROFILE=1`: every
game thread's stack, a sweep about every 4 ms):

1. 2026-11-17, a build with no hot list (every lifted function at `-O1`, so the list does not pick itself): adult Link
   played by hand through Kakariko Village with three of its houses, then two legs up Death Mountain Trail, about 3
   minutes; the samples of each area from 3 s after its load to the next load (75,280 samples of lifted code in
   Kakariko, 33,015 on the trail).
2. 2026-11-17, builds of the same days with a checkout's earlier list of 849 functions (which runs those faster, so it
   leans away from them, not towards them): child Link standing in the Market from its save, and child Link standing in
   Hyrule Field by day, two sessions each (92,652 and 68,111 samples).

The rule is `scripts/hotlist.py`'s, applied by a script outside this repository, because the four areas come from three
builds and two of them from one recording: a sample charges the innermost lifted function (a sample in a piece of a split
giant function, `s_<addr>_<n>`, charges the giant, `f_<addr>`, which `scripts/hotlist.py` reaches through the giant's own
frame); the four areas are weighted equally (each area's samples count as much as another's, however long it was
recorded); the hottest first until 85 % of that weight, which came to 680 functions. It covers 85.0 % of Kakariko's
samples of lifted code, 86.7 % of the trail's, 84.6 % of the Market's and 83.8 % of the field's; a list made without one
of the areas still covers 79 to 84 % of that area. The game's Zstd decoder, which `scripts/hotlist.py` adds whatever the
samples say, is not added: 4 of its 22 functions are in the list by their samples.

- Measured (the same PC and route, one run a build, frames in play): against no list, Death Mountain Trail 45.3 -> 56.4 fps
  (+25 %), the game's time a frame 20.8 -> 16.7 ms (-20 %); Kakariko 51.3 -> 55.7 fps (+9 %), 18.2 -> 16.2 ms (-11 %),
  but more frames over 33 ms there (50 against 25 in about 100 s), not explained. The build: 6:17 -> 6:42 (+25 s) in an
  incremental pair (11 of the 92 lifted files came from the cache); about +55 s from scratch, estimated. The eight pieces
  took 1,452 s of compile CPU where the same functions take about 330 s at `-O1`: about a fifth more lifted compile CPU,
  which is what a PC with few threads pays. Issue 505 has these runs and two other PCs' (a Steam Deck: +10 to +22 % on
  the trail with a list of 319 or 773 functions).
- Not yet: other areas (dungeons, the overworld's other places, fights), child Link in play, a play profile of the
  pieces, a build from scratch, and the build time and memory on fewer cores.

## hookshot_vk.profdata.xz: the renderer's

The Vulkan back end's play profile (nx2/build.py `RX_VK_PGO`; docs/development.md, "The renderer's play profile"),
compressed with xz. `nx2/make_game.py` unpacks it into the work folder and builds the renderer with it, unless the
checkout's `work/hookshot_vk.profdata` or `RX_VK_PGO` says otherwise; the Linux AppImage carries it for the player's
build, which compiles the renderer (the AppImage has no built one). A compiler that can't read it builds the renderer
without it.

One profile for every build for now. It ages rather than breaks: a function changed since the recording is compiled as
if there were no profile, the rest keep theirs. Record it again before a release, and when the renderer has changed much.

Recorded: three sessions of an `RX_VK_PGO=gen` renderer, merged with `llvm-profdata merge` (LLVM 23.1.5); 1,744 of the
renderer's functions ran in at least one of them:

1. 2026-11-14, main 0caf0618, NVIDIA GeForce RTX 3090 (mesh shaders), Linux, Ryzen 9 7900X: one session, counters
   cleared once Kokiri Forest had loaded and written at the end of the Market walk (Kokiri Forest standing, the
   teleport, the Market walk). 1,660 functions ran.
2. 2026-11-14, main 5d3ecc78, AMD Radeon RX 580 (RADV POLARIS10, no mesh shaders: the compute fallback), Linux,
   i7-9700K: child Link from the opening through the debug teleport's overworld destinations, each loaded, a few
   seconds standing, the camera turned, a short run (532 s; Hyrule Castle and Gerudo Fortress, adult only, did not
   teleport). 1,573 functions ran.
3. The same PC and build: the dungeons and Desert Colossus the same way (265 s; the Spirit Temple did not teleport).
   1,581 functions ran.

Recordings 2 and 3 add 76 functions recording 1 never ran, among them the compute fallback's draw path
(`fallback_record`, `fallback_buffers`, `fallback_task_groups`, `fallback_pipelines_ready` in vk_draw.cpp).

- Compiler: the Linux builder's clang (LLVM 23.1.5), so C++ standard library names are libstdc++'s: Windows builds
  (llvm-mingw, libc++) match the renderer's own functions and leave the library's out
- Not yet: a Windows recording, adult Link's areas, the pipeline tour's full route
- Measured with recording 1 alone (the 3090, from the Market save, 3 sessions each, median frame): standing 13.53 /
  13.49 / 13.70 ms without, 12.96 / 13.37 / 12.94 ms with; running in the square 15.89 / 16.13 / 15.73 ms without,
  15.13 / 15.67 / 15.52 ms with: about -4 to -5 % standing and -3 % running. The merged profile is not measured yet.
