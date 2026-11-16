"""(relink) The item wheel's assets, made from the player's own RomFS at the end of every build (nx2/make_game.py), into
<game folder>/wheel/: the game's item icons (icons.bin), the items' names in each of the game's languages (names.bin,
nx2/wheel_names.py) and the item row's sounds (sounds.bin, nx2/wheel_sounds.py). Nothing of the game is in this repository.
With only the Python the builder ships (zstandard): the icons keep the game's own ASTC blocks, decoded by the renderer
(gpu/vk/vk_wheel.cpp, suyu's ASTC decoder), so the build needs no image library.

    python -I nx2/wheel_assets.py --romfs <dir> --icons <game folder>/wheel/icons.bin

icons.bin: "HWIA", u32 version 1, u32 count; then for each icon: char name[48] (NUL-padded game name, "Item_Arrow_Ice"),
u32 format (the BNTX's: 0x2Dxx ASTC 4x4), u32 width, u32 height, u32 bytes, then the blocks in rows, top down.
"""
import argparse
import os
import struct
import sys


def zs_dictionary(romfs):
    """zs.zsdic, out of Pack/ZsDic.pack.zs (a SARC, itself zstd without a dictionary)"""
    import zstandard
    with open(os.path.join(romfs, "Pack", "ZsDic.pack.zs"), "rb") as f:
        b = zstandard.ZstdDecompressor().decompressobj().decompress(f.read())
    data, = struct.unpack_from("<I", b, 12)
    hl, = struct.unpack_from("<H", b, 4)
    n, = struct.unpack_from("<H", b, hl + 6)
    names = hl + 12 + 16 * n + 8
    for i in range(n):
        attr, start, end = struct.unpack_from("<III", b, hl + 12 + 16 * i + 4)
        o = names + (attr & 0xFFFFFF) * 4
        if b[o:b.index(b"\0", o)] == b"zs.zsdic":
            return b[data + start:data + end]
    raise SystemExit("wheel: no zs.zsdic in Pack/ZsDic.pack.zs")


def read_zs(romfs, rel, dictionary):
    """a .zs file of the RomFS, decompressed (one call into a buffer of the frame's stated size: fast on Windows too)"""
    import zstandard
    d = zstandard.ZstdDecompressor(dict_data=zstandard.ZstdCompressionDict(dictionary))
    with open(os.path.join(romfs, rel), "rb") as f:
        data = f.read()
    if zstandard.frame_content_size(data) > 0:
        return d.decompress(data)
    return d.stream_reader(data).read()


def bntx_textures(b):
    """{name: (BRTI offset, format, width, height, GOBs per block, data offset)} of a Switch 2 BNTX (platform "Ounc")"""
    count, = struct.unpack_from("<I", b, 0x24)
    info, = struct.unpack_from("<Q", b, 0x28)
    out = {}
    for i in range(count):
        p, = struct.unpack_from("<Q", b, info + 8 * i)
        no, = struct.unpack_from("<Q", b, p + 0x60)
        nl, = struct.unpack_from("<H", b, no)
        fmt, = struct.unpack_from("<I", b, p + 0x1C)
        w, h = struct.unpack_from("<II", b, p + 0x24)
        bh, = struct.unpack_from("<I", b, p + 0x34)
        mips, = struct.unpack_from("<Q", b, p + 0x70)
        data, = struct.unpack_from("<Q", b, mips)
        out[b[no + 2:no + 2 + nl].decode()] = (p, fmt, w, h, 1 << bh, data)
    return out


def gob_offset(x, y, gobs, pitch):
    """a byte's place in a block-linear texture (x in bytes, y in rows of blocks): the Switch 2's GOB, the Tegra X1's with
    offset bits 5 and 6 swapped (gpu/render/rx_nvn.cpp bl_offset)"""
    a = (y // (8 * gobs)) * 512 * gobs * pitch + (x // 64) * 512 * gobs + (y % (8 * gobs) // 8) * 512
    return a + (((x & 63) >> 5) << 8 | ((y & 7) >> 2) << 7 | ((x & 31) >> 4) << 6 | ((y >> 1) & 1) << 5 | (y & 1) << 4 | (x & 15))


def linear_blocks(b, t):
    """the texture's first level as its compressed blocks in rows, top down (16 bytes a block: ASTC 4x4)"""
    _, fmt, w, h, gobs, data = t
    bpe = 16
    bw, bh = (w + 3) // 4, (h + 3) // 4
    pitch = (bw * bpe + 63) // 64
    return b"".join(b[data + o:data + o + bpe] for o in (gob_offset(x * bpe, y, gobs, pitch) for y in range(bh) for x in range(bw)))


def write_icons(romfs, dictionary, out):
    """The game's item icons (UI/Tex/Icon/<item name>.bntx.zs), their ASTC 4x4 blocks as they are. Returns the count"""
    folder = os.path.join(romfs, "UI", "Tex", "Icon")
    icons = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".bntx.zs"):
            continue
        name = f[:-len(".bntx.zs")]
        b = read_zs(romfs, os.path.join("UI", "Tex", "Icon", f), dictionary)
        tex = bntx_textures(b)
        if not tex:
            print(f"wheel icons: {name} has no texture", file=sys.stderr)
            continue
        t = next(iter(tex.values()))
        if t[1] >> 8 != 0x2D:   # (not ASTC 4x4: the renderer decodes that one only)
            print(f"wheel icons: {name}: format {t[1]:#x} left out", file=sys.stderr)
            continue
        icons.append((name, t[1], t[2], t[3], linear_blocks(b, t)))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"HWIA" + struct.pack("<II", 1, len(icons)))
        for name, fmt, w, h, blocks in icons:
            f.write(name.encode()[:47].ljust(48, b"\0") + struct.pack("<IIII", fmt, w, h, len(blocks)) + blocks)
    print(f"wheel icons: {len(icons)} icons in {out}", file=sys.stderr)
    return len(icons)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--romfs", required=True)
    ap.add_argument("--icons", required=True)
    a = ap.parse_args()
    try:
        write_icons(a.romfs, zs_dictionary(a.romfs), a.icons)
    except Exception as e:   # (an icon folder not the one known: the wheel shows the items' names)
        print(f"wheel icons: not made ({e}): the item wheel shows the items' names", file=sys.stderr)
        if os.path.exists(a.icons):
            os.remove(a.icons)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
