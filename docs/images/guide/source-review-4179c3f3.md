# Graphics guide image: source review against 4179c3f3

Retain `settings-graphics.png` after reviewing the vibration rework on `feat/dualsense-haptics` (PR #405, `7b5394d8`
and the review commit after it) with content review digest
`sha256:4179c3f3d6f121635148401ca32b40133749c113f3ceb4032215aae45e54316b` across the same 77 tracked UI, font and
ImGui source inputs as `source-review-2c5f5737.md` (the digest is the SHA-256 of the sorted `<path> <sha256>` lines of
`reviewed_source.files`). The PNG, its caption, capture date, `captured_source` and capture conditions are unchanged.

`scripts/check_guide_screenshots.py --review-sources` flagged four changed inputs, all of them the Controls tab's rows:

| Source | Change and retention decision |
|---|---|
| `gpu/vk/vk_menu.cpp` | The Controls tab's Rumble strength, Haptic feedback, Haptic strength, Haptic status and Test haptics rows become **Vibration** (Auto, Basic, Off), **Vibration strength** (25 to 200 %) and **Test vibration**, with their texts, `ini_key` lines, options-file migration and the Test vibration press. No Graphics row, value, default or help text changes. |
| `gpu/vk/vk_menu.h` | `Options::haptic`, `haptic_strength` and `haptic_status` go; `Options::vibration = 0` comes, and `rumble`'s comment. No Graphics default changes. |
| `gpu/vk/vk_api.h` | `HOS_OPT_HAPTIC_*` (63 to 66) become `HOS_OPT_VIBRATION` (63) and `HOS_OPT_VIBRATION_STATUS` (64). No member of `MkwVkHost` or `MkwVkRenderer` moves. |
| `gpu/vk/menu_i18n.h` | The haptic rows' texts leave all 13 tables and the Vibration rows' come in. The capture fixture runs in English. |

The pictured Graphics rows (`Quality preset`, `Resolution and upscaling`, `Aspect ratio`, `Output resolution`,
`Custom output size`, `Profile`, `Upscaler`, `Render resolution`, `Sharpness`, `Frame generation and latency`,
`Frame generation`, `Frame generation by`, `Frame pacing`), their defaults and their help remain what the guide shows.
No fresh rendering or pixel equivalence is claimed.
