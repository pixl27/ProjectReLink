# Red Tunic 4.4.0

Red Tunic can now be set up, configured and started through RecompLauncher using the current named-input standard. Windows players can also extract the portable ZIP and open `RedTunic.exe` normally. The executable embeds the original Red Tunic icon and reports Red Tunic 4.4.0 in Windows file properties.

- Easy Setup accepts the supported decrypted NUP or extracted game folder and checks build `e0f3ccfa`. It reuses the existing windowless build and progress reporting. It does not download game files or keys.
- Launcher-managed builds, saves, settings and intermediate files stay under `recompdata`; existing standalone installations remain separate. Play waits for the game to close, returns its exit status and suppresses the game's update question.
- The manifest exposes the existing graphics/settings options, including 720p and 1080p output. It ships beside the Windows executable and inside the Linux AppImage and tarball.
- The Windows archive includes the published program, its runtime and build tools. The Linux tarball and AppImage package the same portable folder, usable without FUSE when unpacked.
- A redesigned README and [project website](https://spacefarergames.github.io/RedTunic/) introduce the Radeon and Steam Deck/Linux focus with original section graphics, a real Red Tunic capture and credited IGN video links. Steam Deck-specific performance measurements remain pending; no public-domain license is claimed for IGN or Nintendo imagery.
- XInput controller presence is published atomically, and a failed polling-worker start can be retried. Integrated GPUs with a device-local shared heap now report its memory budget to the existing pressure logic.
- Serial mesh-fallback scratch sharing is available as `RX_VK_MFB_SHARED=1`. It stays off by default. Four camera-turn runs on Ryzen 5 5625U/Radeon Vega showed the combined experimental options increasing mean frame time by 10.9%, so those options are not enabled and no large speedup is claimed.

The shader translator, cache format and generated game shaders are unchanged by this batch. Validation loaded 47,798 cached programs and reported zero pipeline compilations using an independent cache copy. Shader reuse still requires a matching GPU, toolchain and translator configuration. Updates preserve the player's original cache and settings.

The release retains the 4.3.0 texture-memory controls and its opt-in mesh capture/reuse path. It does not add encrypted NSP/XCI readers, player-key support, achievements or macOS support. See [launcher support and remaining standards](docs/recomplauncher.md), [cache-safe measurements](docs/cache-safe-performance.md) and [tested runs](docs/tested-configurations.md).

The downloads contain the Builder, tools and source only. Every game is built locally from the player's own supported copy; no game executable, game assets, save or shader cache is distributed.

Schema, launcher/package checks, normal Builder startup, Windows icon/version resources, Linux packaging and responsive website checks passed. The full Windows source suite remains non-green: 64 remaining failures also reproduce in 4.3.0; the affected fixture and shader-reader timeout passed on retry. Actual RecompLauncher UI/Flatpak integration and Steam Deck hardware have not been tested.
