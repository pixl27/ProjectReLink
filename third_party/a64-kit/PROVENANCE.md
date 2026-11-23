# a64-kit (vendored subset)

From NotRexed's a64-kit (Rexcomp's AArch64 kit, `Porting Tools\Rexcomp\a64-kit`, a link to
`Ports\TOTK\TOTKPort\a64-kit`) at TOTKPort commit `60de030` (2026-10-30), **with changes**: Hookshot's, most of them
marked `(hookshot)`, and Hookshot's, marked `(hookshot)` (listed below). Only the parts the lifter pipeline needs:
`lift/` (lifter, suyu's translator and its ctypes binding), `rt/` (guest model), `tools/nso.py`, `tools/toolchain.py`,
`tools/survey.py`, `hos/nx_image.py`, `tests/insn_test.py`.

It is vendored, not referenced in place, so that building `a64emit.dll` and `lld-link.exe` (both written next to
their sources) never writes into the TOTKPort checkout, and so that this project is pinned to one version.
Most of the project's changes live outside this folder (`nx2/`): they wrap these modules instead of editing them.

Changes made here after the copy (against Hookshot's first import of it, commit `04407384`):

- Hookshot (NotRexed/Hookshot, before the fork): `lift/a64_lift.py` (game patches after an instruction, setjmp owners,
  page marks grouped per straight-line run of stores), `lift/build_emit.py`, `lift/suyu/arm64_to_c.h` (the FMUL/FMLA-family
  sticky-flag fast path, a64_fpx), `rt/a64.h` and `rt/a64_rt.c` (setjmp in lifted code, the BR hook, an exclusive store's
  page marked after the store), `tools/toolchain.py` (Windows builds with llvm-mingw, cross builds with
  RX_TARGET=windows), `hos/nx_image.py` (the addend of GLOB_DAT and JUMP_SLOT relocations), `README.a64-kit.md`. The
  history is Hookshot's: `git log 04407384.. -- third_party/a64-kit`.
- Hookshot: `tools/toolchain.py` reads `HOOKSHOT_MINGW`, as the rest of Hookshot's build sets it, where Hookshot read
  `HOOKSHOT_MINGW` (marked `(hookshot)`). The `(hookshot)` markers and comments stay as Hookshot wrote them.
- Hookshot, 2026-11-14, from a GPL fork of Hookshot: `lift/a64_lift.py`'s vector float sums (A64_VFPX_SUM)
  fix only the lanes the fast test gave up on, and the operations the fast paths and rescues give up on are done
  with the host's own arithmetic where Arm's result is known (FPX_EASY_HEADER), not the software body.
- Hookshot: `lift/a64_lift.py`'s header helpers are all `always_inline` (marked `(hookshot)`), and the fast float regions' own: without the module inliner
  (nx2/build.py `HOOKSHOT_LIGHT_PASSES`) VFPX_HEADER's 7 `a64_v_*` fell out of line, as real calls.
- Hookshot: `lift/a64_lift.py`'s fast float checks are kept by the instruction's text (`ff_joins`, `ff_convs`,
  `fp_is_frame_kept`, marked `(hookshot)`): the same C, the lift a third quicker on Windows.
- Hookshot: `ff_region` takes `ff_min()` from its caller, read once a function (`least`, marked `(hookshot)`).

So the copy can no longer be refreshed by copying the files again: a refresh keeps these changes, and the tree-sha256
below is this folder's, not upstream's (python -I scripts/check_third_party.py).

Licensing: `lift/suyu/arm64_to_c.h` is suyu's (GPL-2.0-or-later, texts in `lift/suyu/`), so `a64emit.dll`, the
generated runtime and all lifted C are GPL. See `NOTICE.TOTKPort.md`.



Red Tunic 4.0.2 refreshes this copy through Hookshot commit 38b90dfe on 23 November 2026, including the upstream AArch64 lift changes.

tree-sha256: a9f65e695d570892e9f58ec8b98a270a9e0639c5e5f9b808b9e75732facb37c6
