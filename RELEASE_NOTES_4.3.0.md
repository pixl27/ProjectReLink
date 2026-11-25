# Red Tunic 4.3.0

- **Graphics → Detail → Material array limit** sets a mip-size target for eligible large ASTC texture arrays, including terrain materials. It can follow Texture Detail or target 1024, 512, or 256 px. Restart to apply it to arrays already loaded.
- **Graphics → GPU and memory → Texture cache ceiling** offers Auto, 768, 1024, 1536, or 2048 MiB. Auto preserves the renderer's hardware-based policy. Lower ceilings can increase texture reloads, and live resources can temporarily exceed the ceiling.
- The two settings leave the shader translator and cache format intact. A matching Windows llvm-mingw build loaded 47,798 existing shader programs and opened the original pipeline cache on Radeon Vega. Cache reuse still depends on the translator, GPU configuration, and toolchain matching.
- The source includes opt-in Vega mesh capture and reuse with GPU-generated count validation. The release keeps it off by default.

Extract the Windows x64 Builder ZIP and run `HookshotBuilder.exe` to build or update from your own supported game dump. Keep the existing game output directory to reuse settings, saves, and compatible caches. The download contains the Builder and tools. The Linux AppImage remains available in the 4.2.4 release.
