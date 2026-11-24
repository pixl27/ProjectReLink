# Red Tunic 4.2.2

- The Windows x64 ZIP and Linux x86-64 AppImage build or update the game locally from your own supported dump. The Linux package targets glibc 2.35 and newer.
- The renderer scans existing graphics pipeline notes sequentially. This uses the existing file format and shader translator signature.
- On Windows Vega with CPU ASTC, `RX_VK_ASTC_PREFETCH=1` can prepare up to three learned texture levels on one background worker. A demand upload accepts a result only when the encoded blocks match exactly. `RX_VK_TEX_AHEAD_PRESSURE=1` pauses speculative texture work under RAM or commit pressure.
- `RX_VK_BINDING_REUSE=1` can keep unaffected texture views and samplers across unrelated changes. `RX_VK_MESH_CONVERT_SCOPED=1` and `RX_VK_COMPACT_ON_POOL_OOM=1` are additional experiments for mesh barriers and texture-pool recovery.
- The renderer experiments are off by default. In two windowed Hyrule Field camera turns, 42–44 frames took over 100 ms with the three camera experiments off and 44–45 with them on. The CPU texture decoder remains the Vega default.
- The packages leave shader caches, saves, and profiles generated from gameplay on your PC. A new local build can take longer to prepare its shaders.

To update a Linux installation, download the newer AppImage and open your existing game folder in its Builder. Verify downloads with `SHA256SUMS`.
