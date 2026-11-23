"""A damaged chunks.txt is ignored, and chunks.save replaces it whole (nx2/chunks.py).

<work>/image/chunks.txt (the lift's chunk boundaries) outlives the lift, and the launcher's completely cold build keeps image/.
On Windows a PC that went down while the file was being written left it full of null bytes, and every later lift stopped in
chunks.write with `ValueError: invalid literal for int() with base 16: '\\x00\\x00...'` until the player deleted the work
folder. load() now answers no boundaries for a file it cannot read (the lift then takes them from the sizes, as without the
file), and save() writes beside the file and moves it over, so a stopped write leaves the old file whole.

python -I -B tests/test_chunks_corrupt.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
import kit  # noqa: E402,F401  (a64-kit on the path)
import chunks  # noqa: E402


def main():
    with tempfile.TemporaryDirectory(prefix="hookshot-chunks-corrupt-") as temp:
        image = Path(temp) / "image"
        image.mkdir()
        path = image / chunks.FILE
        # (the reported file: 688 null bytes, no newline, what NTFS shows of a file whose data never reached the disk)
        path.write_bytes(b"\x00" * 688)
        got = chunks.load(str(image))
        assert got == [], "a chunks.txt of null bytes gave boundaries: %r" % got[:4]
        # (bytes that are not text at all, and a line that is not an address)
        for bad in (b"\xff\xfe\x80\x81\n", b"# first guest address\n80001000\nnot hex\n"):
            path.write_bytes(bad)
            got = chunks.load(str(image))
            assert got == [], "a damaged chunks.txt (%r) gave boundaries: %r" % (bad, got)
        # (the lift saves new boundaries over the damaged file: they are read back, and no staged file is left beside them)
        chunks.save(str(image), [0x80002000, 0x80001000])
        assert chunks.load(str(image)) == [0x80001000, 0x80002000], chunks.load(str(image))
        left = sorted(p.name for p in image.iterdir() if p.name != chunks.FILE)
        assert not left, "chunks.save left files beside chunks.txt: %s" % left
    print("a damaged chunks.txt is ignored and replaced whole: ok")


if __name__ == "__main__":
    main()
