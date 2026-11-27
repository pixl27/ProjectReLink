# Red Tunic 5.0.3

- Fixed hot-code optimization ([#12](https://github.com/spacefarergames/RedTunic/issues/12)). Windows and Linux packages now ship the committed hot-function address list, so the Builder compiles those functions at `-O2` again. The first rebuild may need to translate the game again.
- The settings overlay fades in and out, and navigation plays the game’s interface sounds extracted from your local copy during setup.
- The Windows Builder checks GitHub for stable updates at startup. Choose Update to download and verify the new Builder and rebuild from the game files already in your installation.
