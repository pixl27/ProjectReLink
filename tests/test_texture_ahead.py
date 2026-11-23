"""Textures prepared ahead of their first draw (RX_VK_TEX_AHEAD; no game/GPU required).

The renderer makes and uploads a texture at the first draw that samples it, so the textures of the area a save is in all
come in the first frames after its loading screen (700 to 1200 of them, two frames of about a second each on a Steam
Deck). The host announces each texture when the game makes it, 5 to 60 seconds before: with the switch on, the ones a
draw has sampled before (kept in the game folder's cache/ between runs) are made and uploaded during the loading screen
instead, a few milliseconds a light frame.

This compiles and runs the rules (gpu/vk/texture_ahead.h: the signature, the place, the allowance, the room, the queue,
the list and its file) and checks that the renderer uses them where it should.

python -I -B tests/test_texture_ahead.py [--cxx g++]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu/vk"


def check_source():
    fail = []
    tex = (VK / "vk_texture.cpp").read_text(encoding="utf-8")
    replay = (VK / "vk_replay.cpp").read_text(encoding="utf-8")
    header = (VK / "vk_texture.h").read_text(encoding="utf-8")
    if '#include "texture_ahead.h"' not in tex:
        fail.append("vk_texture.cpp does not include texture_ahead.h")
    for needle in ("void texture_ahead_made(const nvn_tex* t);", "void texture_ahead_gone(const nvn_tex* t);",
                   "void texture_ahead_frame(int64_t draws);", "void texture_ahead_run();"):
        if needle not in header:
            fail.append("vk_texture.h does not declare: " + needle)
    # the host's announcements reach the queue, and a texture released leaves it
    if "texture_ahead_made(t);" not in block(replay, "void texture_created_of(const nvn_tex* t)"):
        fail.append("vk_replay.cpp: texture_created_of does not hand the texture to texture_ahead_made")
    if "texture_ahead_gone(t);" not in block(replay, "void texture_released_of(const nvn_tex* t)"):
        fail.append("vk_replay.cpp: texture_released_of does not take the texture out (texture_ahead_gone)")
    # prepared in a submission's recording, once its uploads have a command buffer, before its commands are replayed
    submit = block(replay, "bool run_submit(const uint64_t* handles, int n, uint64_t value, uint64_t queue, bool memory_work, int cb = 0,")
    if "texture_ahead_run();" not in submit:
        fail.append("vk_replay.cpp: run_submit does not call texture_ahead_run")
    elif not submit.index("begin_recording(tag);") < submit.index("texture_ahead_run();") < submit.index("replay_handle(cmd,"):
        fail.append("vk_replay.cpp: texture_ahead_run is not between the recording's start and its commands")
    # the allowance comes from the draws of the frame before, once a game frame: the same on a slow PC and a fast one
    tick = block(replay, "void status_tick()")
    if "texture_ahead_frame(" not in tick:
        fail.append("vk_replay.cpp: status_tick does not call texture_ahead_frame")
    elif "draw_stats(" not in tick[:tick.index("texture_ahead_frame(")]:
        fail.append("vk_replay.cpp: status_tick does not give texture_ahead_frame the frame's draws (draw_stats)")
    # off unless asked for, block formats in the game's own memory only, and through the same path as a draw's first use
    on = block(tex, "bool ahead_on()")
    if 'getenv("RX_VK_TEX_AHEAD")' not in on or "atoi(e) != 0" not in on:
        fail.append("ahead_on: RX_VK_TEX_AHEAD no longer turns it on, or it is no longer off without it")
    wanted = block(tex, "bool ahead_wanted(const nvn_texdesc& d, uint64_t gpu, uint64_t size)")
    for needle in ("nvn_block_size(", "1ull << 39", "bw > 1 || bh > 1"):
        if needle not in wanted:
            fail.append("ahead_wanted no longer has: " + needle)
    # (the list holds a texture where the game put it: what one area drew is not prepared in another)
    if "texture_ahead::place(texture_ahead::signature(" not in block(tex, "uint64_t ahead_signature(const nvn_texdesc& d, const nvn_texlayout& L, uint64_t gpu)"):
        fail.append("ahead_signature no longer ties a texture's signature to where its storage starts (texture_ahead::place)")
    made = block(tex, "void texture_ahead_made(const nvn_tex* t)")
    for needle in ("ahead_on()", "ahead_wanted(t->d, t->gpu, t->L.size)", "ahead_queue.made("):
        if needle not in made:
            fail.append("texture_ahead_made no longer has: " + needle)
    learn = block(tex, "void ahead_learn(const Texture& tex) {")
    # (only a texture first drawn soon after the game made it: a load's, not one met minutes later in play; and the
    # list bounded)
    for needle in ("ahead_on()", "ahead_wanted(tex.d, tex.gpu, tex.L.size)", "texture_ahead::remember(&ahead_drawn, ",
                   "ahead_queue.made_at(tex.gpu, &made)", "texture_ahead::learnable(made, "):
        if needle not in learn:
            fail.append("ahead_learn no longer has: " + needle)
    run = block(tex, "void texture_ahead_run()")
    for needle in ("ahead_on()", "texture_ahead::room(", "ahead_drawn.count(", "texture_get(&t, true)", "ahead_allow_ns"):
        if needle not in run:
            fail.append("texture_ahead_run no longer has: " + needle)
    frame = block(tex, "void texture_ahead_frame(int64_t draws)")
    gate = frame.index("draws < texture_ahead::kLightDraws")
    write = frame.index("texture_ahead::save_drawn(")
    if not gate < frame.index("lk.unlock()") < write < frame.index("lk.lock()"):
        fail.append("texture ahead must save only during light frames and outside the registration lock")
    for needle in ("texture_ahead::allowance_ns(draws, ", "ahead_queue.trim(", "texture_ahead::save_drawn("):
        if needle not in frame:
            fail.append("texture_ahead_frame no longer has: " + needle)
    # what a draw samples for the first time is remembered; what was prepared is not learnt from itself
    up = block(tex, "bool upload(Texture& tex, const Ranges* dirty)")
    if "ahead_learn(tex);" not in up or "!ahead_running" not in up:
        fail.append("upload() does not remember a draw's first upload (ahead_learn, outside the preparing)")
    # the switch is documented beside the others
    doc = (ROOT / "docs/development.md").read_text(encoding="utf-8")
    for needle in ("`RX_VK_TEX_AHEAD=1`", "`RX_VK_TEX_AHEAD_MS=<n>`", "hookshot_textures_drawn.bin"):
        if needle not in doc:
            fail.append("docs/development.md does not document: " + needle)
    if fail:
        raise SystemExit("\n".join(fail))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="texture-ahead-") as tmp:
        exe = Path(tmp) / "texture-ahead"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(ROOT / "tests/test_texture_ahead.cpp"), "-o",
                        str(exe)], check=True)
        subprocess.run([str(exe), tmp], check=True)
    check_source()


if __name__ == "__main__":
    main()
