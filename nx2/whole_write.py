"""(issue 1197) A lifted C file rewritten in place (the patches, the chunk split) is written beside it and moved over it: a run
stopped mid-write (the builder closed, a second build stopping the first, a full disk) leaves the old file whole, never the first
megabytes of the new one, which the compile then stopped on at every try ("g_module_b" where g_module_base was).

The move only covers a process that stops: a PC that goes down (power, a crash) can keep the move and lose the data that was still
in memory, and Windows then shows the file at its size full of null bytes. sync=True writes the data to the disk before the move,
for a small file that later runs read back (chunks.txt). The lifted C is not synced (a gigabyte, every lift):
make_game.py lifted_cut_short reads each file's end, and a file of null bytes has no closing brace there."""
import os
import time


def write_whole(path, text, newline="\n", sync=False):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)
        if sync:
            f.flush()
            os.fsync(f.fileno())
    for attempt in range(20):   # (an antivirus reading the file holds it a moment: Windows refuses the move until it lets go)
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)
