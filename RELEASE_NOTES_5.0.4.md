# Red Tunic 5.0.4

- Windows Radeon Vega 1002:15E7 uses GPU ASTC decoding by default. The CPU decoder remains available with `RX_VK_ASTC_CPU=1`.
- New Windows builds on the matching Vega adapter use the tested 720p profile, including 50% distant model detail and overlap-aware write-back waits. Existing settings take precedence.
- Choosing **All** for shader preparation now builds outstanding predicted pipelines at startup even on an integrated GPU or after a previous **Most** preparation.
