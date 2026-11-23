# Red Tunic on Radeon Vega

Red Tunic tunes its native Vulkan renderer for Windows PCs with integrated Radeon graphics. The reference system is a Ryzen 5 5625U with 16 GB of shared RAM and Radeon Vega 1002:15E7.

## Rendering and texture streaming

- With **Auto (this PC)** and a 1920 × 1080 output, the two-thirds render scale gives a 1280 × 720 scene before the game's upscaler.
- The renderer uses the compute fallback for task and mesh shaders on Vega. Cached graphics programs load synchronously on the tested Windows adapter to preserve rendering correctness.
- The CPU decodes ASTC to RGBA8. Both serial and parallel paths use the same block decoder. A cache hit requires the same encoded bytes and decode dimensions.
- The Vega texture retention rule considers available physical memory, commit headroom and driver allocation pressure. The decode cache holds up to 64 MiB of pixel data plus compressed keys.

## Compare performance

Use the same save, graphics settings, driver and camera movement. Separate area-load frames from settled play. Record average FPS, p95 frame time and the counts over 250 and 500 ms to see changes in stalls.

See [tested configurations](tested-configurations.md) for measured runs and the [development guide](development.md) for benchmark tools.
