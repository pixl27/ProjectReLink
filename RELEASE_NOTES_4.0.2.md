# Red Tunic 4.0.2

Released 23 November 2026.

## Installation and portable builds

- Fixed the Windows portable ZIP failing with `A portable game needs the packaged Builder setup launcher`.
- The packaged Red Tunic Builder validates its portable home, then builds a directly launchable `hookshot.exe` when the setup stub is absent.
- Existing setup-based portable installs retain their relative Builder association and volume checks.
- The release archive includes the self-contained Windows Builder, Python, LLVM-MinGW, source and build dependencies. Supply your own supported game dump locally.
- To update, extract the new Builder ZIP into a separate folder and select the existing game output directory. Keep that output folder for saves and settings.

## Upstream integration

Merged Hookshot through `38b90dfe`, fetched on 23 November 2026. This incorporates 547 upstream commits since the Red Tunic 4.0.0 base.

The integration includes faster ASTC shader decoding and AMD BC mip-tail copy fixes, limits on texture handle tables to prevent repeated uploads, improved loading of cached shader programs when startup preparation is disabled, shared AArch64 runtime helpers compiled at O2, the committed hot-function list, GPU architecture detection, Windows Vulkan-loader compatibility, and additional renderer diagnostics and regression checks.

Upstream performance figures describe the upstream test hardware and scenes. Red Tunic 4.0.2 requires its own scene-by-scene comparison for frame-rate claims.

## Red Tunic features retained

- Original red tunic and Radeon-inspired logo, red settings palette, animated procedural loading fire and embers, and eased preparation progress.
- AMD APU Auto preset with a fixed two-thirds render scale. At 1920 × 1080 output the internal scene is 1280 × 720.
- Ryzen x86-64-v3 CPU target with tuning for the detected Zen generation.
- AMD APU texture-cache ceiling, 4096 MB idle-import budget, BC1 for eligible opaque ASTC textures, and compute fallback optimizations.
- Batched visibility-query copies on Windows AMD integrated graphics, with the explicit override available for comparison.
- Optional FP16 SDR sharpening and render-target compression experiment.
- Sequential reads for compressed assets, time-sliced optional cold-memory scans, and a targeted AMD compute-optimizer override.
- Manual Red Tunic update channel through the project's GitHub Releases page.

## Benchmark context

The earlier user report describes 30–60 FPS in Hyrule Field on an AMD Ryzen 5 system with low grass, foliage and shadow density, and about 50% less stutter. That report lacked a comparison build, route or frame-time capture. It is preliminary and predates the 4.0.2 integration.

## Verification

The merged Windows Vulkan renderer compiled. The portable Builder archive and direct-launch regression checks passed, along with 13 documentation-site checks.

The Windows integration test run reported 801 passes, 55 failures and 132 skips. Several targeted failures were addressed after that run started. The complete suite has yet to pass.

Local game executables, dump data, saves and compiled caches remain on the player's machine.

[Source](https://github.com/spacefarergames/RedTunic) · [Downloads](https://github.com/spacefarergames/RedTunic/releases)
