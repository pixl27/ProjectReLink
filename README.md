# Project ReLink

**Zelda OOT Remake Recomp.** An unofficial static recompilation of *The Legend of Zelda: Ocarina of Time* (the Switch 2
remake) into a native Windows game: the game's code is translated to PC code and compiled on your machine, with no emulator.

[Français plus bas](#français)

![The Project ReLink launcher](docs/launcher.png)

**No game files are in this repository.** ReLink builds the game on your PC from your own copy.

## The game version you need

ReLink works with **one version of the game**: the Switch 2 remake with its update installed.

| | |
|---|---|
| Program | `0400486024DEC000` |
| Update | v131072 |
| Build | `e0f3ccfa` |

Other versions do not work: the launcher checks your copy and says so before it installs anything. Your copy is either a
`.nup` file, or the game's `romfs` and `exefs` folders.

## Install and play

1. Download `ReLink-Setup.exe` from the [latest release](https://github.com/pixl27/ProjectReLink/releases/latest).
2. Run it. It downloads ReLink and its tools (about 500 MB), checks them and opens the launcher. Nothing else to install:
   ReLink brings its own compiler and Python.
3. Pick your copy of the game. The launcher looks for `.nup` files in Downloads, Desktop and Documents; you can also choose
   one, drop it on the window, or give the `romfs` and `exefs` folders.
4. Press **Install**. It takes 10 to 45 minutes depending on your CPU; you can keep using the PC.
5. Press **Play**. The desktop and Start menu shortcuts open the launcher from then on.

![Installing](docs/install.png)

Until a release shows on the releases page, the installer is not published yet: see [Build from source](#build-from-source).

### What your PC needs

- 64-bit Windows 10 or 11, and a CPU with AVX2.
- A graphics card with Vulkan 1.3 and mesh shaders, with an up-to-date driver: NVIDIA GeForce RTX 20 series or newer, AMD
  Radeon RX 6000 series or newer, Intel Arc. We only tested on NVIDIA.
- 8 GB of video memory recommended, 16 GB of RAM or more.
- About 60 GB free while it installs, and an internet connection for the download.

### Updating

The launcher says when a new release is out. **Update** downloads only what changed and builds the game again in the same
folder, from the game files the folder keeps: your `.nup` is not needed again. Saves (`user`), settings and mods are kept.

### Where the files go

- The game: the folder you choose (`Games\ReLink` in your user folder by default). Your copy is never changed.
- ReLink itself: `%LOCALAPPDATA%\ReLink`. The build files (about 4 GB) are deleted once the game is ready.
- To uninstall: delete the game folder, `%LOCALAPPDATA%\ReLink` and the shortcuts.

## Settings

The launcher sets the display (full screen, 3D render resolution, upscaling, sharpness, ultrawide), the smoothness (60
frames a second or unlocked, frame generation), the game's language and the controls. It can put the display at 60 Hz while
you play, so that every frame lasts the same time on a 100 or 144 Hz display, and puts the display's own rate back
afterwards. In the game, **Insert** or **F1** opens the same menu, and **F11** switches full screen.

The launcher is in English; French is one click away on its Game page. An Xbox controller or any XInput controller works,
and so do keyboard and mouse (the keys are listed on the Controls page).

## Measured frame rates

On our PC: **Ryzen 5 5500, GeForce RTX 3060 Ti (8 GB), 16 GB of RAM, Windows 11**. Game capped at 60 frames a second, 3D
render resolution at 75 %, version 0.6.1. Averages over windows of 30 to 60 seconds.

| Place | Frames a second |
|---|---|
| Hyrule Field, walking | 59.8 |
| Kakariko Village, walking | 59.0 |
| Hyrule Castle, walking | 58.8 |
| Lost Woods, walking | 59.6 |
| Sacred Forest Meadow, walking | 60.0 |
| Lake Hylia, walking | 59.2 |
| Zora's Fountain, walking | 58.3 |
| Market, walking | 54.1 |
| Inside the Deku Tree, Dodongo's Cavern, Bottom of the Well (child) | 60 |
| Inside Jabu-Jabu's Belly (child) | 58 to 60 |
| Kakariko Village, Fire Temple, Gerudo's Fortress (adult, standing) | 60 |
| Water Temple (adult, standing) | 58.0 |

At native render resolution (100 %): 51 to 53 in the heaviest views of Kokiri Forest's village, 53.5 to 56.7 in the forest,
52.8 at Zora's Fountain. At Zora's Fountain the render resolution decides: automatic 54.5, 75 % 58.3, 67 % 58.6, 50 % 60.0.

Not reached or not measured yet: a steady 60 everywhere (the Market and native resolution stay below), boss fights, sessions
longer than half an hour, and AMD and Intel cards.

## If the game slows down with adult Link: check your dump

Some dumps hold a few damaged textures. The game then loads the model that needs one again and again for ever, which keeps
more than one CPU core busy as soon as Link is an adult. Two tools in this repository deal with it, from your own files
(nothing of the game is in them):

```powershell
python -I tools/dump_check.py "C:\Games\ReLink\romfs\Tex"
python -I tools/dump_repair.py "C:\Games\ReLink\romfs" "C:\Games\ReLink\mods\00-dump-repair"
```

The first counts the damaged files; the second writes a small mod that stands in for the ones it knows with an intact
texture of the same family. They need the `zstandard` package, which ReLink's own Python has
(`%LOCALAPPDATA%\ReLink\versions\<version>\ReLink\tools\python\python.exe`).

## Mods

RomFS mods go in `mods\<name>\romfs\...` in the game folder, with the same paths as the game's. The log
(`hookshot_log.txt`) names each mod it reads. Patches of the game's code (`exefs`) are not supported: the code is compiled
ahead of time.

## Build from source

For developers. The launcher runs the same script, `nx2/make_game.py`.

```powershell
python -I nx2/make_game.py --nup "C:\path\your-game.nup" --out "C:\Games\ReLink" --language en-US
python -I nx2/make_game.py --romfs "C:\path\romfs" --exefs "C:\path\exefs" --out "C:\Games\ReLink" --language en-US
```

`--check` with your `.nup` only checks the game's version. `--log file` keeps the output. `HOOKSHOT_JOBS=<n>` limits how many
files compile at once.

You need:

- Python 3 from python.org, with `pip install lz4 numpy capstone "unicorn>=2.1,<2.2" zstandard`.
- [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (the `ucrt-x86_64` build), unpacked as `llvm-mingw` in the clone, or
  `HOOKSHOT_MINGW` set to its folder.
- The Vulkan headers and FFmpeg in the CPM cache's layout, with `RX_CPM` set to their folder. Without them the Vulkan
  renderer is not built.
- The prebuilt files of `third_party` and the play profile of `work`. An installed ReLink holds them, in
  `%LOCALAPPDATA%\ReLink\versions\<version>\ReLink\tools`.

The launcher is in `launcher/` (`dotnet build`, .NET 9), the small setup in `launcher/setup/`. Publishing a release is
described in [docs/releasing.md](docs/releasing.md).

### How it works

1. `nx2/lift.py` translates the game's ARM64 code to C.
2. `nx2/build.py` compiles that C together with `host/`, which plays the part of the console's system.
3. `host/nvn_native.c` records the game's graphics calls, and `gpu/vk/` replays them on Vulkan.

## License

Project ReLink is free software under the [GNU General Public License, version 3 or later](LICENSE) (GPL-3.0-or-later). It
reuses and modifies code published under the GPL by other projects (modified in November 2026); each part keeps its
copyright notices and licenses in its files and folders.

Project ReLink is an unofficial fan project, not affiliated with or endorsed by Nintendo. The Legend of Zelda and Ocarina
of Time are trademarks of Nintendo.

---

## Français

**Project ReLink** est une recompilation statique, non officielle, de *The Legend of Zelda: Ocarina of Time* (le remake
Switch 2) en jeu Windows natif : le code du jeu est traduit en code PC et compilé sur votre machine, sans émulateur.
**Ce dépôt ne contient aucun fichier du jeu** : ReLink fabrique le jeu sur votre PC à partir de votre propre copie.

### La version du jeu qu'il vous faut

Une seule version fonctionne : le remake Switch 2 avec sa mise à jour installée (programme `0400486024DEC000`, mise à jour
v131072, build `e0f3ccfa`), en fichier `.nup` ou en dossiers `romfs` et `exefs`. Le lanceur vérifie votre copie avant
d'installer quoi que ce soit.

### Installer et jouer

1. Téléchargez `ReLink-Setup.exe` dans la [dernière version](https://github.com/pixl27/ProjectReLink/releases/latest).
2. Lancez-le. Il télécharge ReLink et ses outils (environ 500 Mo), les vérifie et ouvre le lanceur. Rien d'autre à installer.
3. Choisissez votre copie du jeu : le lanceur cherche les `.nup` dans Téléchargements, Bureau et Documents ; vous pouvez aussi
   en choisir un, le déposer sur la fenêtre, ou donner les dossiers `romfs` et `exefs`.
4. Appuyez sur **Install**. Comptez 10 à 45 minutes selon le processeur ; le PC reste utilisable.
5. Appuyez sur **Play**. Le lanceur est en anglais par défaut ; le français se choisit dans sa page Game.

Tant qu'aucune version n'apparaît sur la page des versions, l'installateur n'est pas encore publié : voir « Build from
source » plus haut.

Il faut Windows 10 ou 11 en 64 bits, un processeur avec AVX2, une carte graphique avec Vulkan 1.3 et les mesh shaders
(GeForce RTX 20 ou plus récente, Radeon RX 6000 ou plus récente, Intel Arc ; nous n'avons testé que sur NVIDIA), 16 Go de RAM
et environ 60 Go libres pendant l'installation.

Le lanceur dit quand une nouvelle version est sortie ; **Update** ne télécharge que ce qui a changé et reconstruit le jeu sur
place. Les sauvegardes, les réglages et les mods sont conservés.

### Images par seconde mesurées

Sur un Ryzen 5 5500, une GeForce RTX 3060 Ti (8 Go), 16 Go de RAM et Windows 11, jeu limité à 60 images par seconde et rendu
3D à 75 % : 58 à 60 dans la plupart des lieux (plaine d'Hyrule 59,8, village Cocorico 59,0, lac Hylia 59,2, fontaine Zora
58,3, donjons 58 à 60) et 54,1 en traversant la place du marché. Le 60 constant partout n'est pas encore atteint, et les
combats de boss, les longues sessions et les cartes AMD et Intel n'ont pas été mesurés. Le tableau complet est dans la partie
anglaise.

### Licence

GPL-3.0-or-later ([LICENSE](LICENSE)). Le projet reprend et modifie du code publié sous GPL par d'autres projets
(modifications de novembre 2026) ; chaque partie garde ses mentions de droits d'auteur et ses licences dans ses fichiers.
Projet de fans non officiel, sans lien avec Nintendo ; The Legend of Zelda et Ocarina of Time sont des marques de Nintendo.
