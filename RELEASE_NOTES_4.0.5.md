# Red Tunic 4.0.5

- On Windows Radeon Vega 1002:15E7, more streamed ASTC texture levels use the Ryzen CPU's parallel decoder. Both decode paths produce the same RGBA8 output.
- On the same adapter, the renderer can retain up to 768 MiB of already resident textures when the driver reports a small local heap and Windows has enough physical and commit headroom. Driver allocation refusals and the commit guard can still trigger eviction. Explicit texture budgets take precedence.
- The Windows download contains the Red Tunic Builder. Build or update your local game with the Builder. Existing graphics settings and the shader-cache format are compatible.
