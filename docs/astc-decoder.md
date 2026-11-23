# ASTC decoder and its checks

On GPUs without hardware ASTC decoding, ASTC textures are decoded on the GPU by
`gpu/vk/host_shaders/astc_decoder.comp` (its SPIR-V in `astc_decoder_comp_spv.h`). The decoder keeps each
invocation's decoded values in on-chip shared memory (`s_buf[16][65]`: one row per thread, so no barriers), hoists
the block-invariant work (partition selection, weight-grid setup, dimension scaling) out of the per-texel loop,
with a direct path when the weight grid matches the block, finds the ISE encoding range by binary search, and
zeroes the colour endpoint data before reading it, as the ASTC specification asks for bits past the colour data.

It replaced Hookshot's earlier hoisting of the same shader (which followed Lichii's decoder change, with
justr4mzi's and SmileyTatsu's work): the new decoder covers that work and more, and only one is kept. The
Eden/yuzu GPL notices remain in the shader.

## Checks

`tests/test_astc_decoder_gpu.py` (with `tests/astc_decoder_gpu.cpp`) compiles the shader, checks the SPIR-V
header against it, and decodes a CPU reference over all 14 block sizes; with `--gpu` it runs the shader on the
GPU and compares its output with `Tegra::Texture::ASTC::DecompressSerial` (odd sizes, linear and native tiling,
1 to 4 partitions, dual plane, void extent, error blocks).

## Native measurement scope

The RTX 5070 Ti runs of 2026-11-12 in docs/tested-configurations.md measured the earlier hoisting: matched ASTC
upload GPU spans in a Kokiri Forest checkpoint load, taken with private GPU timestamps on identical builds. They are
upload times for that decoder, not FPS or start-up claims, and they do not describe the decoder above.
