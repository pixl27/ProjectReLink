# BC-250 guarded launcher

These scripts run an existing Linux Project ReLink build with an explicitly
selected RADV ICD. They do not download or contain title data.

The launcher is intentionally conservative:

- it asks Bazzite's installed `tuned-ppd` for the performance profile only for
  the lifetime of the run, and verifies every CPU policy changed;
- the D-Bus hold is tied to its caller, so TuneD releases it when the holder is
  killed as well as on an orderly exit;
- a systemd scope bounds memory, swap and elapsed time;
- a second guard stops before host free memory falls below 2 GiB or the AMD GPU
  or Zen CPU reaches 90 C;
- it explicitly joins the existing Wayland session and aborts if ReLink does
  not create a visible mailbox swapchain within 60 seconds;
- it returns the display to the active Wayland virtual terminal on exit;
- serial replay (`RX_VK_PARALLEL=1`) remains the correctness baseline.

On Bazzite/Fedora, install the small runtime dependencies and prime sudo once:

```sh
sudo dnf install python3-dbus tuned-ppd mangohud
sudo -v
```

Do **not** replace `tuned-ppd` with `power-profiles-daemon`; the Fedora packages
conflict, and Bazzite maps the same D-Bus API to its TuneD profiles.

Copy the example profile somewhere writable, then give the launcher the build,
profile and development ICD paths:

```sh
cp scripts/bc250/profile.env.example "$HOME/relink-bc250.env"

scripts/bc250/run-relink-bc250.sh \
  "$HOME/games/relink-build" \
  "$HOME/relink-bc250.env" \
  "$HOME/mesa-directmesh/share/vulkan/icd.d/radeon_icd.x86_64.json"
```

Useful opt-ins are environment variables, not edits to the script:

```sh
# Fullscreen, with MangoHud enabled by default.
BC250_FULLSCREEN=1 scripts/bc250/run-relink-bc250.sh BUILD PROFILE ICD

# Five-minute validation route using an input script already in PROFILE.
BC250_RUNTIME_MAX_SEC=300 BC250_KEEP_INPUT_SCRIPT=1 \
  scripts/bc250/run-relink-bc250.sh BUILD PROFILE ICD
```

Each run writes a timestamped metrics file below `BUILD/logs/`. The defaults
are a 7.031-GiB soft limit, 8.30-GiB hard limit, 2-GiB cgroup swap limit and a
two-hour time limit. They can be changed with the documented `BC250_*_BYTES`
and `BC250_RUNTIME_MAX_SEC` environment variables, but a higher limit is not a
substitute for fixing allocation or reclaim growth.

`BC250_REPLAY_THREADS=4` enables experimental parallel command recording. It
is deliberately not the default until the sparse-bind and mesh-program locking
paths are validated together on the guarded route.
