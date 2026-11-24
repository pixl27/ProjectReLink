# Red Tunic 4.2.0

- The Windows Radeon Vega preset uses CPU ASTC decoding to protect materials on characters, enemies and buildings. GPU decoding of eligible mipmapped textures remains experimental and off by default.
- During ordinary budget eviction, Vega prioritizes recently used textures with costly CPU uploads. Existing memory limits and GPU-written texture protections still apply.
- The Windows Builder seeds the tested Vega graphics and performance defaults when it detects a single matching adapter. It fills missing options and keeps existing values. The preset uses a near-720p internal resolution at 1080p output, low grass, foliage and shadow distance, and synchronous graphics-program loading for HUD compatibility. It also includes the tested core reservation, texture lookahead, hash memo, background imports and AV1 settings. Manual builds can use `--vega-profile` if detection misses the adapter.
- Small staging allocations can retry with dedicated memory. Slow-frame reports now read the full counters for each renderer phase.

In an experimental run with GPU ASTC enabled, on a Ryzen 5 5625U with Radeon Vega and AMD 26.5.2, two paired Hyrule Field camera sweeps improved from 6.8–7.6 to 8.7–9.4 average FPS. The 99th-percentile frame time fell from 639–660 ms to 408–461 ms. These are headless test runs at 67% scale. See the [test records](docs/tested/) for individual results.

Download the Windows x64 Builder ZIP and verify its SHA-256 checksum. The Builder creates or updates a local game from your own game files.
