# Red Tunic 5.0.3

## Revised release

- Windows Radeon Vega 1002:15E7 uses GPU ASTC decoding by default. The CPU decoder remains available with `RX_VK_ASTC_CPU=1`.
- New Windows builds on the matching Vega adapter use the reference PC’s 720p graphics profile, including 50% distant model detail and overlap-aware write-back waits. Existing player settings take precedence.
- Choosing **All** for shader preparation builds outstanding predicted pipelines at startup on integrated GPUs and after a previous **Most** preparation.
- The Windows Builder reads the installed Red Tunic version from the game executable, so stale Hookshot metadata no longer makes an updated game appear to need the same update.
- If you already installed 5.0.3, download the revised Builder and rebuild into your existing game folder to receive this revision.

## Original release

- Fixed hot-code optimization ([#12](https://github.com/spacefarergames/RedTunic/issues/12)). Windows and Linux packages now ship the committed hot-function address list, so the Builder compiles those functions at `-O2` again. The first rebuild may need to translate the game again.
- The settings overlay fades in and out, and navigation plays the game’s interface sounds extracted from your local copy during setup.
- The Windows Builder checks GitHub for stable updates at startup. Choose Update to download and verify the new Builder and rebuild from the game files already in your installation.
- Fixed missing GPU ASTC decoder bounds that could make textures black or garbled, and fixed deferred texture uploads that could get stuck showing a placeholder. ([#7](https://github.com/spacefarergames/RedTunic/pull/7), [#8](https://github.com/spacefarergames/RedTunic/pull/8))
- Steam Deck LCD and OLED use the shared texture budget and full pass barriers by default. Windows Vega keeps its current hardware defaults. ([#9](https://github.com/spacefarergames/RedTunic/pull/9))
