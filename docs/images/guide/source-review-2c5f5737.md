# Graphics guide image: source review against 2c5f5737

Retain the UI-only `settings-graphics.png` captured from fullscreen-monitor
topic `54d25c79ead4d07e22cdcf5f07e481e19f90f78c` (on main `e335bd5c`) after
reviewing the rebased `feat/dualsense-haptics` tree (based on main `47430e81`)
with content review digest
`sha256:2c5f5737139f10ea8c837d24b1cc7bb0330e09e112d09d19a1a74bd9d9214a91` across
all 77 tracked UI, font and ImGui source inputs. The PNG was inspected at full
size during this review. Its caption, original capture date, PNG hash,
`captured_source` and capture conditions remain unchanged.

Running `scripts/check_guide_screenshots.py --review-sources` against
`sha256:006bc251...` (`415a894`) on the rebased tree flagged 23 changed inputs,
one removed upstream header (`gpu/vk/vk_vendor.h`, deleted by `#1390` `03788b3`
in favor of `device_policy.h`), one new transitive header (`host/hos_tracy.h`,
included via `vk_internal.h` from `#1543` `ef6038a`), and two stub additions in
`scripts/screenshots/menu.cpp` (`dlss_fg_most()` and `framegen_capped()` from
`#1486` `ec76f69` and `#1496` `02a2e4f`):

| Source | Change and retention decision |
|---|---|
| `gpu/vk/device_policy.h`, `fallback_size_policy.h`, `mirror_budget.h`, `texture_budget.h` | Upstream main only (`#1390`, `#1650`, `#1559`, `#1296`, `#1388`): GPU device policy, fallback buffer sizing, and VRAM/texture budget rules. None alters pictured Graphics labels or defaults. |
| `gpu/vk/threaded_rendering.h` | Upstream main only (`#1664`): physical-core Auto tier rule for off-screen **Threaded rendering**. Not shown in the first unscrolled Graphics view. |
| `gpu/vk/upscaler_rows.h`, `vk_dlss.h`, `vk_temporal.h` | Upstream main only (`#1652`, `#1675`, `#1678`, `#1486`): DLAA pre-bloom helpers and `dlss_fg_most()`. With `dlss_built() == false` in the UI-only fixture, the **DLAA before the bloom** row is hidden and pictured **Upscaler** / **Render resolution** rows remain at their defaults. |
| `gpu/vk/vk_draw.h`, `vk_fps.h`, `vk_framegen.h`, `vk_fsr3fi.h`, `vk_internal.h`, `vk_progress.h`, `vk_recorder.h`, `vk_texture.h` | Upstream main only (`#1691`, `#1538`, `#1487`, `#1496`, `#1618`, `#1543`, `#1339`, `#1162`): renderer, HUD-less framegen capture, shader-prep progress bar, and Tracy declarations. None changes the pictured Graphics controls. |
| `gpu/vk/vk_menu_i18n.cpp`, `vk_menu_i18n.h`, `host/posix/windows.h`, `host/hos_tracy.h` | Upstream main only (`#1318`, `#1562`, `#1543`): CJK/Cyrillic font fallback ranges for language names, table-length checks, POSIX shim and Tracy no-op macros. The English fixture rendering is unaffected. |
| `gpu/vk/menu_i18n.h` | Upstream main (`#280`, `#1318`, `#1486`, `#1496`, `#1497`, `#1554`, `#1678`) + PR: upstream updates translations for NFC, DLAA pre-bloom, framegen hints and Debug rows; this PR adds only 13-language entries for the four new **Controls** haptic rows (`Haptic feedback`, `Haptic strength`, `Haptic status`, `Test haptics`). The capture fixture runs in English (`opts.menu_lang = 1`). |
| `gpu/vk/vk_api.h` | Upstream main (`#280`, `#1530`, `#1663`) + PR: upstream adds NFC callbacks, `HOS_OPT_HUD_LAYOUT = 71` and v2 perf summaries; this PR adds `HOS_OPT_HAPTIC_MODE`, `HOS_OPT_HAPTIC_STATUS`, `HOS_OPT_HAPTIC_STRENGTH`, and `HOS_OPT_HAPTIC_TEST` (`63-66`). No pictured Graphics label or fixture interface behavior changes. |
| `gpu/vk/vk_menu.cpp` | Upstream main (`#280`, `#424`, `#1486`, `#1496`, `#1497`, `#1554`, `#1629`, `#1678`) + PR: upstream adds the **NFC** tab, banner logo/title layout, **DLAA before the bloom** (hidden when DLSS is not built), and Debug/hint updates; this PR adds **only** four rows on `TAB_CONTROLS` (**Haptic feedback**, **Haptic strength**, **Haptic status**, **Test haptics**), their text/hint callbacks, `menu_haptic_*` accessors, and `TAB_CONTROLS` input handling. Every pictured Graphics row (`Quality preset`, `Resolution and upscaling`, `Aspect ratio`, `Output resolution`, `Custom output size`, `Profile`, `Upscaler`, `Render resolution`, `Sharpness`, `Frame generation and latency`, `Frame generation`, `Frame generation by`, `Frame pacing`), its default value and its help text remain applicable to the guide's instructions. |
| `gpu/vk/vk_menu.h` | Upstream main (`#1678`, `#1554`) + PR: upstream adds `dlss_pre_bloom` and `present_queue`; this PR adds `Options::haptic = 0`, `Options::haptic_strength = 0`, `Options::haptic_status = 0`, and `menu_haptic_status_option()` / `menu_haptic_test_take()` declarations. Defaults of all pictured Graphics settings are unchanged. |
| `scripts/screenshots/menu.cpp` | Upstream main fixture update (`#1486`, `#1496`): adds `dlss_fg_most() { return 0; }` and `framegen_capped(int framegen, int) { return framegen; }` stand-ins so the fixture links cleanly with unchanged Graphics defaults. |

All remaining 53 inputs in the 77-file `reviewed_source.files` set (including
Libertinus fonts, Dear ImGui sources, `display_modes.h`, `monitor_choice.h`,
`gpu_choice.h`, `intel_compiles.h`, `vk_xess.h`, `host/audio_output_devices.h`)
and `scripts/screenshots/capture_menu.py` are byte-for-byte unchanged relative
to `manifest.json` at `415a894`.

Retention means the image still teaches the labelled Graphics controls and
navigation; it does not mean that a new capture would be pixel-identical
(upstream `#280` adds the **NFC** tab in the top bar and `#1629` updates the top
banner layout, while this PR touches only `TAB_CONTROLS`). No fresh rendering or
pixel-equivalence check was performed.

The review revision in `manifest.json` (`sha256:2c5f5737...`) is the SHA-256
digest of the sorted `path=hash` lines (UTF-8, trailing newline) for all 77
entries in `reviewed_source.files`, matching the repository's content-addressed
review convention without depending on temporary branch commit SHAs.

This is a source-and-existing-image review, not a fresh fixture capture or a live
game comparison. No game, capture build, save transfer or runtime experiment was
performed.
