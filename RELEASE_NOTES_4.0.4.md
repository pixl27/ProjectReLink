# Red Tunic 4.0.4

- Large ASTC textures use parallel CPU decoding on Windows Vega PCI 1002:15E7. Serial and parallel decoding produced identical RGBA8 bytes in the decoder test.
- On that adapter, a deferred texture can bypass the per-frame upload budget at its next eligible lookup after 250 ms. Area loading can still delay that lookup.
- CPU ASTC decoding and synchronous reads of cached graphics programs remain enabled on this Vega adapter. The AMD 720p internal rendering preset remains available.
- In Water Temple on a Ryzen 5 5625U with AMD driver 26.5.2, the largest logged 1152×1152 texture upload took 69 ms in the parallel run and 288 ms in the serial run. One cold run per configuration. No overall frame-rate improvement or shader startup speedup was established.
- The self-contained Windows ZIP includes the Red Tunic Builder and requires your local game files. Extract it into a fresh folder and choose your existing game output directory to update.

[Water Temple test notes](https://github.com/spacefarergames/RedTunic/blob/v4.0.4/docs/tested/2026-11-23-vega-texture-deferral-codex-run01.md)
