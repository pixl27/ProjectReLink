# Mods

Red Tunic loads mods that replace or add the game's files (textures, models, texts, sounds and other data), mods that
patch the game's code (ExeFS patches), and Atmosphere cheats. This page covers installing them, turning them off,
their order and checking that they loaded.

## Before you start

- **Back up your saves.** A mod can change what gets saved, and removing it later doesn't undo progress made with it.
  Red Tunic Builder's **Mods + Editor** page keeps a backup each time its save editor saves.
- **Stop the game.** Mods are found when the game starts: add, remove, rename or edit them with the game closed, then
  start it again. The Builder's Mods page sees a new mod by itself (or press **Reload**).

## Install a mod

1. **Open the game folder**, the one with `hookshot.exe`.
2. **Make a `mods` folder** there if there isn't one yet.
3. **Put each mod in a folder of its own** inside `mods`, with the mod's `romfs` (files), `exefs` (code patches) or
   `cheats` folder in it:

   ```text
   mods/
     01-hd-textures/
       romfs/
         ...
     02-auto-jump/
       exefs/
         auto-jump.pchtxt
   ```

   Most mods made for emulators already come in this layout: unpack them into `mods` as they are. Their deeper
   layouts work too, such as `mods/contents/<title id>/<name>/romfs`.
4. **Start the game.** A mod with code patches needs a build first (below).

An update of the game keeps your `mods` folder. A mod made for an older version of the game files may need an update
of its own.

## Code patches (ExeFS)

Red Tunic compiles the game's code ahead of time, so a mod's `exefs` patches (`.pchtxt`, `.ips`) are applied when the
game is built. After adding a mod with patches, changing one, or turning such a mod on or off, build the same folder
again: Red Tunic Builder's **Mods + Editor** page shows a **Rebuild** button then (a mix of patches built before is kept,
and **Apply** brings it back without building). `exefs_patches.txt` in the game folder lists what the build applied.
Only patches for this game's code build are used.

## Turn a mod off

In the Builder's **Mods + Editor** page, untick it; in the game, the Insert menu's **Mods** tab switches it Off. Or
rename its folder so that it starts with `_` or `-`, for example `_02-auto-jump`. Delete the folder to remove the mod for
good.

## When two mods change the same file

The one lower in the list wins. Out of the box the list is in the folders' names' order, so numbering them
(`01-...`, `02-...`) sets it; the Mods tab and the Builder's Mods page move mods up and down, and keep the order and the
switches in `hookshot_mods.txt` in the game folder. Files that only one mod has are used either way.

## Mods that make files bigger

The game sets aside a fixed amount of memory for each of its files, in a resource size table. Red Tunic makes that
table at each start with mods: the game's own, with every mod's table merged in and every file a mod serves raised to
fit, so mods that each make files bigger (HD textures, longer parameter files) load together. The button prompts'
entries are kept in it, so your controller's icons keep working.

## Check that a mod loaded

Open `hookshot_log.txt` in the game folder after a start: its `mods:` lines name each mod found, how many files it
has, and the files the game read from it. No `mods:` lines at all means there is no `mods` folder in this game folder.

## For mod authors

The [readme's Mods section](../readme.md#mods) has the details (the size table, the patches' build id, the cheats), and
the [file mod contract](mods.md) covers discovery, conflicting files and disabling, with an example checked without
the game. To keep mods somewhere else, set `HOS_MODS=<folder>` in `hookshot_env.txt`: it names the folder that holds
the mods, not one mod's `romfs`.
