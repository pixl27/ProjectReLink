# Red Tunic 4.1.0

- The Windows Radeon Vega compatibility path limits texture prediction under memory pressure and keeps live GPU-written depth images when eviction cannot preserve their contents. This targets camera-turn stalls and disappearing shadows while keeping the existing render resolution and quality settings.
- Optional renderer flags in `hookshot_env.txt` include `RX_VK_HASH_MEMO=1` for unchanged tracked texture storage and `RX_VK_ASYNC_IMPORT=1` for moving eligible imports off the replay thread.
- `RX_VK_GONE_RING=1` can now reuse texture-handle views after unrelated evictions. Missing views, changed slots and incomplete eviction history still use the normal resolution path.
- `RX_VK_DEPTH_ZERO=1` clears eligible depth images directly on the GPU after validating the submitted bytes. `RX_VK_WB_SPANS=1` remains an experimental writeback wait mode.
- The Windows ZIP contains the 4.1.0 Red Tunic Builder. It builds or updates a local installation from your own game files.

See [tested configurations](docs/tested-configurations.md) for hardware-specific runs. Individual preview results do not establish a release-wide frame-rate gain.
