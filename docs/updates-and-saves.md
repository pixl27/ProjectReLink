# Updates and saves

## Update an installation

Follow the README's [updating instructions](../readme.md#updating) for the current launcher behavior and update options.
Back up your saves before updating. The [save guide](saves.md#where-the-saves-are) lists the locations and overrides;
the README explains [where other files go](../readme.md#where-the-files-go).

## Stable and development builds

The [release page](https://github.com/spacefarergames/RedTunic/releases) identifies stable and prerelease builds.
Read [the versioning guide](versioning.md) and [development-build instructions](../readme.md#latest-development-build)
before switching tracks. Record the exact version when reporting a problem.

## Transfer saves

Use the [save transfer guide](saves.md) for supported sources, backups, and transfer commands.

## If the game stops while it writes a setting

Red Tunic writes `language.txt` and `mouse_sens.txt` beside themselves and moves each into place only when
it is whole, so a crash, a kill or a power cut while one is written leaves it as it was or as the new one, never half of
each. The one-time copy of an older `hookshot_options.ini` from the game folder into the user folder, the note
`saves-moved.txt` and the files the Linux move of the saves copies between drives are written the same way. A stop in
the middle can leave a file ending in `.hosnew` beside the setting. It is what the game was writing, and the next start
removes the ones beside `language.txt`, `mouse_sens.txt` and `hookshot_options.ini`. Any other `.hosnew` file can be
deleted by hand.

## Mods

The [mods guide](using-mods.md) covers installing them, turning them off, their order and checking that they loaded; the
[mods reference](../readme.md#mods) has the details.
