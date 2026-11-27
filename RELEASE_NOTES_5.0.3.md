# Red Tunic 5.0.3

- Fixed hot-code optimization ([#12](https://github.com/spacefarergames/RedTunic/issues/12)). Windows and Linux packages now ship the committed hot-function address list, so the Builder compiles those functions at `-O2` again. The first rebuild may need to translate the game again.
- The settings overlay fades in and out, and navigation plays the game’s interface sounds extracted from your local copy during setup.
- The Windows Builder checks GitHub for stable updates at startup. Choose Update to download and verify the new Builder and rebuild from the game files already in your installation.
- Fixed missing GPU ASTC decoder bounds that could make textures black or garbled, and fixed deferred texture uploads that could get stuck showing a placeholder. ([#7](https://github.com/spacefarergames/RedTunic/pull/7), [#8](https://github.com/spacefarergames/RedTunic/pull/8))
- Steam Deck LCD and OLED use the shared texture budget and full pass barriers by default. Windows Vega keeps its current hardware defaults. ([#9](https://github.com/spacefarergames/RedTunic/pull/9))
