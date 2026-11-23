# Red Tunic 4.0.3

- Windows Vega graphics with PCI ID `1002:15E7` now decode ASTC textures on the CPU and load cached graphics programs synchronously. The AMD 720p preset and shader caches remain enabled.
- On a Ryzen 5 5625U running Windows 11 with AMD driver 26.5.2, the equivalent settings restored room surfaces, the minimap, hearts, magic and equipped items in Water Temple 3F. CPU decoding and synchronous loading can increase loading time. Performance in other areas has yet to be measured.
- The Windows Builder displays the Red Tunic icon in its window and executable metadata, and uses Red Tunic branding in its build prompts.
- If bundled Python is missing, the Windows ZIP prompts for package repair instead of using a system interpreter. Repair links from the standalone Builder open Red Tunic Releases.
- The portable ZIP retains the direct-launch fix from 4.0.2. Extract it into a separate folder, open the Builder, and choose your existing game output folder to rebuild from your local game files.

[Source](https://github.com/spacefarergames/RedTunic) · [Downloads](https://github.com/spacefarergames/RedTunic/releases)
