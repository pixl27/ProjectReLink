# Red Tunic 5.0.0

Red Tunic 5.0.0 is a major update focused on texture decoding and a more reliable start and presentation on compact PCs. It keeps the existing shader translation and cache format so compatible shader work can be reused.

## Performance and loading

- The optimized CPU ASTC decoder is enabled by default, and large textures can use bounded parallel decoding. The camera survey recorded 3.39 seconds of CPU upload work in the serial run and 0.91 seconds in the parallel run across the captured session. In the order-reversed pair, mean frame time improved by 1.2% and p99 by 38%; the first pair favored serial decoding. This is evidence of less CPU texture work, not a repeatable general FPS gain. All measured camera frames in that test exceeded 33.3 ms. See [the full measurement notes](docs/recomp-performance-survey-2026-11-26.md).
- The integrated-GPU in-play shader path now displays the animated Red Tunic loading screen with an indeterminate progress bar and a warm fade-in/fade-out. It does not claim shader compilation percentage when compilation is deferred to first use.
- The renderer fits the fixed title-screen crop inside the selected output and scales the full image to the window. The UI includes a 1280 × 720 choice and a setting to unlock higher output sizes.
- Distant object detail can be reduced to 50%, making the game switch to simpler models sooner.
- The game window title includes the Red Tunic version and, once known, the current area and Link's age.

## Builder, UI and site

- The Windows Builder reports Red Tunic 5.0.0 metadata and uses matching branding. The README and project site present the major changes, supported hardware focus and measured limitations.
- The Builder's shader-preparation option explains that the game displays the Red Tunic loading screen while the first frame appears when preparation is deferred.

## Compatibility and validation

The Windows game was built as an optimized Release build. Its game-free Windows test run finished with 826 passed, 43 failed and 132 skipped; this is not a passing release gate. The failures need to be classified and cleared by CI before the release can be published.

The local runtime smoke test used a private copy of the writable game folder and a directory link to the local ROMFS. It exercised startup and play on Windows with a Ryzen 5 5625U / Radeon Vega iGPU. The cold start peaked at about 5.6 GB of working set and logged a 1.3-second slow submit while the scene was loading. A later warm run reached gameplay and used existing pipeline caches, but neither run was a controlled before/after timing comparison. See the [camera and texture measurements](docs/recomp-performance-survey-2026-11-26.md) for the controlled comparisons.

The installed shader cache, shader seed and user settings were preserved during the local update. Steam Deck and Linux runtime testing are not claimed by the Windows Radeon reference run.

No game files, keys, built game executable, save or shader cache are included. Every game build uses the player's own supported copy.
