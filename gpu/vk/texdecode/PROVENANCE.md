# texdecode: ported code

`astc.cpp` and `block_linear.cpp` are **ported from other projects, not written here**: the decoding itself (the ASTC tables,
integer-sequence, block-mode, endpoint and weight code; the GOB address function and the segment copy) comes from them. What is
this repository's own: `texdecode.h` (the API and its documentation), the code around the ported core in the two `.cpp` files
(argument, span and overflow checks, the size functions, depth slices, the error-block count; "Changes made here" lists it,
the error colour as a parameter among it: the original's own colour, magenta, is the default), the tests (`tests/test_texdecode_*.py`, `tests/texdecode_*_test.cpp`) and the vector generator
(`scripts/gen_texdecode_astc_vectors.py`, `scripts/texdecode_astc_gen.cpp`). Licence of the ported code: MIT, text and holders
in [LICENSE-MIT](LICENSE-MIT); the files keep their notices in their headers. MIT is compatible with this repository's
GPL-3.0-or-later; the combined work is distributed under the GPL. What ocarina-recomp's own licence is, and what that does and
does not mean for this repository, is under "ocarina-recomp's licence" below.

## Origins

| File | From | Licence |
|---|---|---|
| `astc.cpp` | ocarina-recomp `runtime/tex_astc.c` (https://github.com/sostenesfreitas/ocarina-recomp @ `406b3ae0321eff8dd083a607a8186aa56b52b873`, file sha256 `c0a6e3f7936f0a0140904cb7d82a721f7592e15ea2de6b738045bcc950f5b138`), whose header says it was ported from **LAS Recomp** `runtime/nvn/astc.c` @ `da35b83a` (https://github.com/peiikai/LASRecomp @ `da35b83a9611d046a9a681aae668d7085954d37f`), with "only the include changed" | MIT, Copyright (c) 2026 Peiikai, as ocarina-recomp's `THIRD_PARTY_NOTICES.md` lists it (the "LAS Recomp host runtime" row names `runtime/tex_astc.c`). LAS Recomp's own `LICENSE` at that commit was not read here |
| `block_linear.cpp` | ocarina-recomp `runtime/tex_deswizzle.c` (same commit, file sha256 `1e599fc2e32db5eec967d72b8902437db8eadfeff66cb13259d055a5d27ce079`): the address function `nvn_bl_offset` and the copy loop `tex_bl_copy`. Its header says it was ported from **MacNativeGames** `games/totk/third_party/d2r_gfx/gfx/tex/nvn_deswizzle.c` @ `b75be8cf` (https://github.com/ly05010419/MacNativeGames @ `b75be8cfa704df3701e86130c8ced268e79d6b11`), "comments translated, the copy loop split out as tex_bl_copy" | MIT, Copyright (c) 2026 jhaizhou, as ocarina-recomp's `THIRD_PARTY_NOTICES.md` lists it (the "MacNativeGames block-linear deswizzler" row names `runtime/tex_deswizzle.c`). The upstream `LICENSE` at that commit was not read here |

Who changed what between the upstream files and ocarina-recomp's copies is **not known**. The header of `tex_deswizzle.c`
says the copy loop was split out as `tex_bl_copy` and cites Eglantine's `NativeGobOffset` for the Switch 2 bit order (the
Switch 1 GOB with offset bits 5 and 6 swapped), in the passive voice; its notices row repeats the split and names no author.
Neither says it was ocarina-recomp's own work, and nothing here claims so: the split and the bit order may have been made by
ocarina-recomp, by TotK Recomp or by anyone in between. (An earlier version of this note said they were made in a repository
that is itself GPL-3.0; the files do not say that.) No Eglantine code is listed under either file in the notices, and none is
used here; the bit order is a fact about the hardware that suyu's `SWIZZLE_X_BITS` / `SWIZZLE_Y_BITS` (already in the tree)
state alike.

Not taken: `tex_astc_mt.c` (no licence line in its header, no row in the notices file; it needs that project's host and
platform headers; threading is the caller's here), `tex.h` (a mixed header of that project's, with the NVN descriptor
types), `nvn_tex.c` (deko3d's layout maths, zlib: a level's geometry stays with `vk_texture.cpp`'s `level_geo`).

## The way through ocarina-recomp: TotK Recomp (not settled)

ocarina-recomp's `THIRD_PARTY_NOTICES.md` has the row "TotK Recomp, the base of this repository (fork at 0.2.8)",
https://github.com/valentderah/totk-pc-recomp @ `f469c5a`, **MIT (`LICENSE-MIT`), Copyright (c) 2026 Valiantsin Dzerakh**,
used in "every file that came from it, with the rows below for what TotK Recomp itself took from others". Its README says the
base of the project is dkb-pc-recomp and TotK Recomp, "whose history is kept here". The rows below that one include the two
this file's sources are listed under (LAS Recomp, MacNativeGames), so both files **may** have come with the TotK Recomp base and
not from ocarina-recomp's own additions. The files cannot settle it: neither header names TotK Recomp, the notices rows say
only where the code was first taken from, and the checkout read (a shallow clone: `.git/shallow`, one commit) has no history. So the chain is
upstream -> (possibly TotK Recomp) -> ocarina-recomp -> here, and Valiantsin Dzerakh's MIT notice is carried in
[LICENSE-MIT](LICENSE-MIT) in case it is owed (an extra MIT notice costs nothing; a missing one breaches the licence). To
settle it: read ocarina-recomp's full history, or `totk-pc-recomp @ f469c5a`, for `tex_astc.c` and `tex_deswizzle.c`.

## ocarina-recomp's licence

As its own files state it (read at `406b3ae`):

- `LICENSE` is the unmodified GNU GPL version 3 text. (Its "or any later version" lines are the licence's own appendix, not a
  statement about this project.)
- `README.md` and `README.en.md` (line 164): "GPL-3.0 (LICENSE), like the original project". No "or later".
- `THIRD_PARTY_NOTICES.md` and `runtime/third_party/eglantine/README.md` say the Eglantine files are GPL-3.0-or-later and
  that this is "the reason DKB Recomp is GPL-3.0-or-later" / DKB Recomp "as a whole" is. That names **DKB Recomp**, ocarina-recomp's
  upstream, not ocarina-recomp itself; the notices' Hookshot rows say GPL-3.0.
- `LICENSE-MIT` (the root one) is MIT, Copyright (c) 2026 Valiantsin Dzerakh: TotK Recomp's notice.
- `tex_astc.c`, `tex_deswizzle.c` and `tex.h` have no licence or SPDX line.

So the README says GPL-3.0 and the notices say GPL-3.0-or-later for the project it descends from, and **the files do not settle the
version ocarina-recomp distributes under**. It was settled outside them: the project owner reports (2026-11-21) that
ocarina-recomp's author said "GPL 3 or later" and that Longshot may use its code (the README's own wording is "GPL-3.0"). So whatever
ocarina-recomp itself added on top of the MIT originals (at most the copy-loop split and the Switch 2 bit order in the deswizzle, if
they are its own) is GPL-3.0-or-later, the same as this repository's licence. Our right to use the two files' MIT-licensed parts
still rests on their MIT licence (LAS Recomp, MacNativeGames). Whether any such addition is in the files is what the previous
section leaves open (they may have come through TotK Recomp); `block_linear.cpp` is rewritten around the copy loop and the bit
order. The confirmation is the owner's report of the author's word: it is not in writing in this repository.

## Not checked

The upstream `LICENSE` texts of LAS Recomp @ `da35b83a` and MacNativeGames @ `b75be8cf` (LICENSE-MIT's holders are taken from
ocarina-recomp's notices); the upstream files themselves (only ocarina-recomp's copies and their headers were read); TotK
Recomp; ocarina-recomp's history.

## Changes made here

- `astc.cpp`: C to C++ (`namespace mkwvk::texdecode`, `static_assert`, `<cstring>`); the decoder body is otherwise the
  original. Its `goto error` paths became `return false` out of an inner `decode_block`, with the error colour written by the
  caller (C++ will not jump over initialisations; the original wrote magenta, which stays the default, and the colour is now a
  parameter, `Rgba8 error_color`, because an all-zero block is an illegal one and `vk_texture.cpp` passes transparent black, as
  main's decoder and the GPU shader write an illegal block, so storage the game has not written yet is not magenta). An HDR endpoint mode, which the original decoded to magenta endpoints and
  reported as success, is now also reported as `false` (astcenc counts it as an error block; the pixels are unchanged). The
  original's `astc_decode_ldr` became `astc_decode`: spans, bounds checks, depth slices, a count of error blocks, `-1` for a bad
  argument, and every size product checked against 64-bit wrap (a level too large to count is refused). Not carried over:
  `astc_decode_ldr_mt`.
- `block_linear.cpp`: rewritten around the original's address function and copy loop. The NVN texture descriptor, the level
  and layer lookups, linear-storage copies and the box copy (`nvn_texels_box`) are not here: the caller passes one level's
  width, height and depth in blocks, bytes per block and block height and depth. Added: argument and span checks (the source
  may end after the last GOB a texel is in; bytes per block up to 16; every size product checked against 64-bit wrap), `block_linear_size`, `block_linear_row_bytes`, depth blocks of any 2^n.
- Tests, vectors and the generator are new. The vectors are 3,976 random blocks (fixed seed) and 718 built by hand, 4,694 in all, checked against ARM's astcenc 6.0.0
  (`22ba8a01`); astcenc is not part of this repository.
