"""Game-free: the System page's tinted controller picture (nx2/prompt_assets.py picture_layers). A face's picture is drawn
in grey like the game's Pro Controller, and its tinted copy (Controller_Pro1) is made from the game's own grey/tinted pair.
The game's tint is not a straight line of the grey: the black outlines turn a brown lighter than the #333333 body, which
turns almost black, and the #646464 buttons a light brown. A line fitted through that pair crushed the outlines to black,
lit the body and dimmed the buttons (every drawn face's page, the PlayStation, Xbox and Steam Deck pictures). The copy
follows the pair's own colour for each grey, blended between two greys, and is not thrown by a few blended edge pixels, a
stray pixel or a pixel the tinted picture leaves clear.

python -I -B tests/test_prompt_picture_tint.py
"""
import sys
from pathlib import Path

RUN_ALL_MODULES = ["PIL", "numpy"]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import prompt_assets as P  # noqa: E402

# the game's pair, as measured on its Pro Controller: grey -> tinted (RGB)
PAIR = {0x00: (26, 19, 10), 0x33: (13, 10, 9), 0x64: (84, 73, 53), 0xCD: (148, 124, 99)}
W, H = 1160, 680


def check(cond, what):
    if not cond:
        raise SystemExit("FAIL: " + what)


def pair():
    """the game's plain and tinted pictures: a band of each grey (most of it the body, as the game's), the rest clear; with a
    few stray pixels in the body (the median's), 10 blended edge pixels of a grey of their own (fewer than a level takes), and
    a strip the tinted copy leaves clear (not a level either)"""
    from PIL import Image, ImageDraw
    plain, tinted = Image.new("RGBA", (W, H), (0, 0, 0, 0)), Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for grey, y0, y1 in ((0x33, 0, 400), (0x64, 400, 520), (0x00, 520, 580), (0xCD, 580, 600)):
        ImageDraw.Draw(plain).rectangle((100, y0, 1000, y1 - 1), fill=(grey, grey, grey, 255))
        ImageDraw.Draw(tinted).rectangle((100, y0, 1000, y1 - 1), fill=PAIR[grey] + (255,))
    ImageDraw.Draw(tinted).rectangle((100, 0, 1000, 19), fill=(200, 0, 0, 255))   # (stray: 5% of the body)
    ImageDraw.Draw(plain).rectangle((1010, 0, 1019, 0), fill=(0x4B, 0x4B, 0x4B, 255))   # (an edge: 10 pixels)
    ImageDraw.Draw(tinted).rectangle((1010, 0, 1019, 0), fill=(255, 255, 255, 255))
    ImageDraw.Draw(plain).rectangle((100, 610, 1000, 629), fill=(0x80, 0x80, 0x80, 255))   # (clear in the tinted copy)
    ImageDraw.Draw(tinted).rectangle((100, 610, 1000, 629), fill=(255, 0, 0, 0))
    return plain, tinted


def main():
    import numpy as np
    from PIL import Image
    plain, tinted = pair()
    images = {"plain": plain, "tinted": tinted}
    tex = {P.PICTURES[0]: "plain", P.PICTURE_TINT: "tinted", P.PICTURE_SHADOW: (None, None, 440, 580)}
    P.decode = lambda b, t: images[t]   # (the game's textures: the pair above)

    greys = (0x00, 0x33, 0x64, 0xCD, 0x4B, 0x80)
    picture = Image.new("RGBA", (W, H), (0, 0, 0, 0))   # (a face's picture: each grey, and greys between two of them)
    for i, grey in enumerate(greys):
        picture.paste((grey, grey, grey, 255), (20 + 180 * i, 100, 160 + 180 * i, 300))
    layers = {layer.texture: layer for layer in P.picture_layers(b"", tex, picture)}
    copy = np.asarray(layers[P.PICTURE_TINT].im.convert("RGBA")).astype(int)

    def at(grey):
        return tuple(int(v) for v in copy[200, 90 + 180 * greys.index(grey), :3])

    for grey, want in PAIR.items():
        got = at(grey)
        check(max(abs(g - w) for g, w in zip(got, want)) <= 1, f"grey #{grey:02x} is tinted {got}, the game's is {want}")
    # between two greys: blended, rounded (#4b is 24/49 of the way from the body's #33 to the buttons' #64; #80 from #64 to #cd)
    for grey, lo, hi in ((0x4B, 0x33, 0x64), (0x80, 0x64, 0xCD)):
        t = (grey - lo) / (hi - lo)
        want = tuple(int(np.rint(a + t * (b - a))) for a, b in zip(PAIR[lo], PAIR[hi]))
        check(at(grey) == want, f"grey #{grey:02x} is tinted {at(grey)}, the blend of #{lo:02x} and #{hi:02x} is {want}")
    check(copy[200, 90, 3] == 255 and copy[10, 10, 3] == 0, "the copy's clear and solid parts are not the picture's")
    check(layers[P.PICTURES[0]].im is picture, "the picture itself is not written as it was drawn")
    print("ok: the tinted copy takes the game's colour for each grey, and the blend between two (outlines, body, buttons, "
          "symbols; a stray pixel, a blended edge and a clear part of the game's copy left out)")


if __name__ == "__main__":
    main()
