"""Game-free: the System page's leader-line textures a face gives by hand (nx2/prompt_assets.py guide_layers_of and
hand_layers). ControllerGuide_Pro_02 to 04 (ZR, the d-pad's frame and line, L and R) are the game's own, drawn for the Switch
Pro Controller; a face whose picture differs gives them in nx2/prompt_faces/<face>/, lines only, so nothing of the game's is
in git. The build writes each over the game's whole: the game's labels' bars kept under the lines given, the game's own lines
covered. Taking the hand-made images in place of the made ones afterwards (hand_layers) must leave those three as they are:
put back raw, the bars of the labels on that texture were gone.

python -I -B tests/test_prompt_hand_guides.py
"""
import sys
from pathlib import Path

RUN_ALL_MODULES = ["PIL", "numpy"]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import prompt_assets as P  # noqa: E402

SIZE = (1555, 1511)
FACES = ("ps4", "ps5", "xbox", "steamdeck")   # (each gives 02 to 04 by hand)


def check(cond, what):
    if not cond:
        raise SystemExit("FAIL: " + what)


def game_texture():
    """a guide texture as the game draws one: a label's bar on each side, and a line of its own across the page"""
    from PIL import Image, ImageDraw
    im = Image.new("L", SIZE, 0)
    d = ImageDraw.Draw(im)
    d.rectangle((P.BARS[0] - 7, 280, P.BARS[0] - 1, 1400), fill=255)
    d.rectangle((P.BARS[1], 280, P.BARS[1] + 6, 1400), fill=255)
    d.rectangle((300, 1300, 1200, 1305), fill=255)
    return im


def main():
    import numpy as np
    from PIL import Image
    names = [f"ControllerGuide_Pro_0{i}" for i in range(5)]
    images = {n: game_texture() for n in names}
    P.decode = lambda b, t: images[t]
    for face in FACES:
        hand = {n: P.override(face, n + ".png") for n in names}
        check(all(hand[n] for n in names[2:]) and not any(hand[n] for n in P.GUIDES),
              f"nx2/prompt_faces/{face}/ no longer gives 02 to 04 (and not 00 and 01) by hand: this checks those")
        layers = P.hand_layers(face, P.guide_layers_of(b"", {n: n for n in names}, face))
        for n in names[2:]:
            mine = [layer for layer in layers if layer.texture == n]
            check(len(mine) == 1 and mine[0].sw == 0xFF, f"{face} {n}: not one layer written whatever the controls")
            layer = mine[0]
            got = np.asarray(layer.im) > 0
            given = np.asarray(Image.open(hand[n]).convert("L")) > 0
            check(got[700, P.BARS[0] - 4] and got[700, P.BARS[1] + 3],
                  f"{face} {n}: the game's labels' bars are not under the lines given")
            check((got | ~given).all(), f"{face} {n}: a line given by hand is missing")
            check(not got[1302, 1000] or given[1302, 1000], f"{face} {n}: the game's own line is still drawn")
            x0, y0, x1, y1 = layer.box
            check(x0 <= 300 and x1 >= 1201 and y0 <= 1300 and y1 >= 1306,
                  f"{face} {n}: the box written does not cover the game's own line")
    print(f"ok: {', '.join(FACES)}'s ControllerGuide_Pro_02 to 04 go over the game's, its bars kept and its own lines covered")


if __name__ == "__main__":
    main()
