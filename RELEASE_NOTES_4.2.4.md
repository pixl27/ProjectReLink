# Red Tunic 4.2.4

The Windows x64 download includes the portable Red Tunic Builder and its dependencies. Extract the whole ZIP, open `HookshotBuilder.exe`, and select your own supported game dump. The Builder applies the Vega settings automatically when Windows reports a single Radeon adapter with PCI ID `1002:15E7`. Updates keep existing choices and local caches.

- The Vega profile uses the reference system's 1280 × 720 output, two-thirds internal render scale, and reduced grass, foliage and shadow distances. Exact CPU ASTC decoding stays enabled for texture compatibility.
- Texture uploads reuse scratch allocations and keep more expensive textures resident when memory allows. The experimental coarse mip streaming path is disabled in this profile after it increased long frames in testing.
- Scoped mesh conversion barriers, GPU clears for known exposure counters and earlier compute shader module cleanup ship enabled in the Vega profile.
- Returning from the inventory or item wheel briefly holds the last complete gameplay frame and fades into the live scene.
- The Builder checks an archive's identity before reusing extracted files and detects an interrupted replacement before updating. Older archive stamps can require one fresh extraction when you select the original archive.

In one paired, 90-present Hyrule Field camera turn on the Ryzen 5 5625U with 16 GB RAM and Radeon driver 26.5.2, the combined compute settings reduced p95 frame time from 294.1 ms to 218.4 ms and frames over 250 ms from six to four. A separate functional check covered Hyrule Field, the inventory, the item wheel and a settled Castle Town arrival. [Measured test runs](docs/tested-configurations.md)

The Linux AppImage is available from [4.2.2](https://github.com/spacefarergames/RedTunic/releases/tag/v4.2.2).
