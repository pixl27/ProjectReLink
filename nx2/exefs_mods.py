"""Exefs mods: code patches from the mods folder, applied to a copy of the game's modules before the translation.

The game's code is translated and compiled ahead of time (nx2/lift.py, nx2/build.py), so a patch of the code takes effect
only when the game is built: make_game.py calls prepare() once the exefs has been read and checked, and the lift reads
the patched copy. The patches are found as host/mods.c finds mods, in the layout emulators use:

    <mods>/<name>/exefs/<build id>.ips     IPS32 or IPS: records written into the module whose build id the name is
    <mods>/<name>/exefs/<any>.pchtxt       text patches: the module is named inside (@nsobid-<build id>)

<mods> is the folder's "mods" (or HOS_MODS); any folder at any depth (up to four) holding a romfs or exefs folder is a mod,
and one whose name starts with '-' or '_' is off. Mods are applied in the order of their paths (case aside): when two write
different bytes at the same place, the later one wins, with a line saying so.

Offsets. Both formats give positions in the module's NSO file as it is once decompressed: the 0x100-byte header, then the
segments at their memory offsets. A module image offset is therefore the patch's offset minus 0x100 (a pchtxt line's offset
first gets its file's offset_shift added: "@flag offset_shift 0x100" makes offsets written as image offsets file positions).
A patch writing outside the three segments (into the header, a gap between segments, or past the data into the bss), or
across a segment's end, is left out whole, with a line: none of its records are applied. One gap is not: the rest of the
.text's last page (Discord, 19/11: Always-On Rupee Counter, Faster Dash, HUD of Time). The console maps the code a page at a
time, so those bytes are there and executable, and mods put their own code in them (a "code cave"), as emulators let them:
the .text is made that much longer (zeros up to the patch) and its code is translated with the rest.

The patched modules are written, uncompressed, into <work>/exefs_mods (every other file of the exefs copied as it is), and
<work>/exefs_mods.json says which bytes each patch changed: the lift (nx2/lift.py) reads it to say which of ReLink's own
changes a patch is in the way of. Without any patch that changes a byte, prepare() writes nothing and the build reads the
game's own exefs, exactly as without this file.

relink_exefs_mods.txt in the game folder (write_report, at the end of a build): its first line is a hash of the patch files
the build saw (fnv1a64 below; "none" for none), then one line per patch applied or left out. The game (host/mods_exefs.c)
and the launcher (launcher/ExefsMods.cs) compute the same hash of the patch files in mods, and say a build is needed when
it differs.
"""
import json
import os
import shutil
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = 0x100                       # (the NSO header before the first segment in a patch's offsets)
PAGE = 0x1000                        # (the console maps a module's segments a page at a time)
EXTENSIONS = (".ips", ".pchtxt")
REPORT = "relink_exefs_mods.txt"
DESCRIPTION = "exefs_mods.json"      # (in the work folder, beside the patched copy)
COPY = "exefs_mods"
LIFT_NOTES = "relink_exefs_lift.txt"   # (in <work>/sa/gen: what the lift found, nx2/lift.py)
LIFT_STOP = "relink_exefs_stop.txt"    # (in <work>/sa/gen: why the lift stopped, said by make_game.py as it is)
MAX_DEPTH = 4                        # (host/mods.c find_mods)
FNV_OFFSET, FNV_PRIME = 0xcbf29ce484222325, 0x100000001b3
SEGMENTS = (".text", ".rodata", ".data")


def page_align(v):
    return (v + PAGE - 1) // PAGE * PAGE


class PatchError(ValueError):
    """A patch file that cannot be read: it is left out whole"""


# ---------------------------------------------------------------- the mods and their patch files

def mods_root(out):
    """Where the game looks for mods (host/mods.c): HOS_MODS, else the game folder's mods"""
    return os.environ.get("HOS_MODS") or os.path.join(out, "mods")


def _fold(text):
    """A path's sort key, and its bytes: its UTF-8 bytes (the file system's on Linux) with ASCII letters in lower case, then
    as they are (the game and the launcher sort the same way)"""
    b = os.fsencode(text)
    return bytes(c + 32 if 65 <= c <= 90 else c for c in b), b


def find_mods(root):
    """[(name, path)] of the mods under root, as host/mods.c finds them, sorted by name (case aside); and the folders that are
    off. A mod: a folder below root holding a romfs or exefs folder (it is not looked into further)"""
    found, off = [], []

    def walk(path, name, depth):
        if depth > 0 and (os.path.exists(os.path.join(path, "romfs")) or os.path.exists(os.path.join(path, "exefs"))):
            found.append((name, path))
            return
        if depth >= MAX_DEPTH:
            return
        try:
            entries = list(os.scandir(path))
        except OSError:
            return
        for e in entries:
            try:
                is_dir = e.is_dir()
            except OSError:
                is_dir = False
            if not is_dir or e.name.startswith("."):
                continue
            if e.name.startswith(("-", "_")):
                off.append(name + "/" + e.name if name else e.name)
                continue
            walk(e.path, name + "/" + e.name if name else e.name, depth + 1)

    if os.path.isdir(root):
        walk(root, "", 0)
    found.sort(key=lambda m: _fold(m[0]))
    return found, sorted(off, key=_fold)


def patch_files(root):
    """[(rel, path, mod name)] of every exefs patch file of the mods under root, rel being "<mod>/exefs/<file>" ('/'), in the
    order they are applied: by mod, then by file name (case aside)"""
    files = []
    for name, path in find_mods(root)[0]:
        exefs = os.path.join(path, "exefs")
        try:
            entries = sorted(os.scandir(exefs), key=lambda e: _fold(e.name))
        except OSError:
            continue
        for e in entries:
            try:
                is_file = e.is_file()
            except OSError:
                is_file = False
            if is_file and e.name.lower().endswith(EXTENSIONS):
                files.append((name + "/exefs/" + e.name, e.path, name))
    return files


def fnv1a64(data, h=FNV_OFFSET):
    for c in data:
        h = ((h ^ c) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def files_hash(files):
    """The hash of a set of patch files, as the game and the launcher compute it: FNV-1a 64 over, for each file in the order
    of its relative path's sort key (_fold), the path's UTF-8 bytes, a zero byte, the file's size as 8 bytes (little-endian)
    and its bytes. '%016x', or "none" without a file. contents: {rel: bytes} (what was read; a file that cannot be read
    counts as empty)"""
    if not files:
        return "none"
    h = FNV_OFFSET
    for rel, data in sorted(files.items(), key=lambda kv: _fold(kv[0])):
        h = fnv1a64(_fold(rel)[1] + b"\0" + struct.pack("<Q", len(data)), h)
        h = fnv1a64(data, h)
    return "%016x" % h


# ---------------------------------------------------------------- the two formats

def parse_ips(data):
    """[(file offset, bytes)] of an IPS32 ("IPS32" ... "EEOF": 4-byte offsets) or IPS ("PATCH" ... "EOF": 3-byte offsets) patch.
    Each record: offset, a 2-byte size, that many bytes; size 0 is a run (2-byte count, one byte). All numbers big-endian.
    Bytes after the end marker (the old format's truncation size) are ignored"""
    if data.startswith(b"IPS32"):
        at, width, end = 5, 4, b"EEOF"
    elif data.startswith(b"PATCH"):
        at, width, end = 5, 3, b"EOF"
    else:
        raise PatchError("not an IPS file (it starts %r)" % data[:5])
    writes = []
    while True:
        if data[at:at + len(end)] == end:
            return writes
        if at + width + 2 > len(data):
            raise PatchError("cut short at byte %d (no end marker)" % at)
        offset = int.from_bytes(data[at:at + width], "big")
        size = int.from_bytes(data[at + width:at + width + 2], "big")
        at += width + 2
        if size:
            if at + size > len(data):
                raise PatchError("the record at %#x is cut short" % offset)
            writes.append((offset, bytes(data[at:at + size])))
            at += size
        else:
            if at + 3 > len(data):
                raise PatchError("the run at %#x is cut short" % offset)
            count = int.from_bytes(data[at:at + 2], "big")
            writes.append((offset, bytes([data[at + 2]]) * count))
            at += 3


def _code_part(line):
    """The line without its // comment (one inside a quoted string stays)"""
    quoted, i = False, 0
    while i < len(line):
        c = line[i]
        if c == "\\" and quoted:
            i += 2
            continue
        if c == '"':
            quoted = not quoted
        elif not quoted and line.startswith("//", i):
            return line[:i]
        i += 1
    return line


ESCAPES = {"n": b"\n", "t": b"\t", "r": b"\r", "0": b"\0", "\\": b"\\", '"': b'"', "'": b"'"}


def _string(text, lineno):
    """A quoted value's bytes: UTF-8, with \\n \\t \\r \\0 \\\\ \\" and \\xHH. No terminating zero is added (write \\0)"""
    if len(text) < 2 or not text.endswith('"'):
        raise PatchError("line %d: the string has no closing quote" % lineno)
    body, out, i = text[1:-1], bytearray(), 0
    while i < len(body):
        c = body[i]
        if c == '"':
            raise PatchError("line %d: text after the string's closing quote" % lineno)
        if c != "\\":
            out += c.encode("utf-8")
            i += 1
            continue
        if i + 1 >= len(body):
            raise PatchError("line %d: the string ends in a lone backslash" % lineno)
        e = body[i + 1]
        if e == "x":
            try:
                out.append(int(body[i + 2:i + 4], 16))
            except ValueError:
                raise PatchError("line %d: \\x needs two hex digits" % lineno) from None
            if len(body[i + 2:i + 4]) != 2:
                raise PatchError("line %d: \\x needs two hex digits" % lineno)
            i += 4
        elif e in ESCAPES:
            out += ESCAPES[e]
            i += 2
        else:
            raise PatchError("line %d: unknown escape \\%s" % (lineno, e))
    return bytes(out)


def _hex(text):
    t = text[2:] if text[:2].lower() == "0x" else text
    if not t or any(c not in "0123456789abcdefABCDEF" for c in t):
        raise ValueError(text)
    return int(t, 16)


def parse_pchtxt(text):
    """{module build id (as written): [(file offset, bytes)]} of a text patch, and notes (lines ignored, said in the report).

    Its rules, line by line: "@nsobid-<build id>" names the module the lines after it patch; "@flag offset_shift <n>" adds n
    (0x..., decimal, or else hex) to the offsets after it; "@flag be" makes each hex value's bytes go in reverse order (the value written
    as a big-endian number), "@flag le" back to the bytes in the order written; "@enabled" starts lines that are applied,
    "@disabled" lines that are not (nothing before the first "@enabled" is applied); "@stop" ends the file. "//" starts a
    comment, and so does "#" at a line's start. A patch line is "<hex offset> <hex bytes>" or "<hex offset> "<string>"".
    A line that is not understood in an enabled part leaves the whole file out (PatchError)"""
    out, notes = {}, []
    module, shift, reverse, enabled = None, 0, False, False
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _code_part(raw).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            words = line.split()
            d = words[0].lower()
            if d.startswith("@nsobid-"):
                bid = words[0][8:]
                if not bid or any(c not in "0123456789abcdefABCDEF" for c in bid):
                    raise PatchError("line %d: %s is not a build id" % (lineno, words[0]))
                module = bid
                out.setdefault(module, [])
            elif d == "@flag" and len(words) >= 2:
                flag = words[1].lower()
                if flag == "offset_shift" and len(words) >= 3:
                    try:
                        shift = int(words[2], 0)
                    except ValueError:
                        try:   # (hex without 0x, as the offsets are written: "0100")
                            shift = int(words[2], 16)
                        except ValueError:
                            raise PatchError("line %d: offset_shift %s is not a number" % (lineno, words[2])) from None
                elif flag == "be":
                    reverse = True
                elif flag == "le":
                    reverse = False
                elif flag in ("print_values", "debug_info"):
                    pass
                else:
                    notes.append("line %d: @flag %s is not known: ignored" % (lineno, " ".join(words[1:])))
            elif d == "@enabled":
                enabled = True
            elif d == "@disabled":
                enabled = False
            elif d == "@stop":
                break
            else:
                notes.append("line %d: %s is not known: ignored" % (lineno, words[0]))
            continue
        if not enabled:
            continue
        parts = line.split(None, 1)
        head, rest = parts[0], parts[1].strip() if len(parts) > 1 else ""
        try:
            offset = _hex(head)
        except ValueError:
            raise PatchError("line %d: %r is not a patch line (<hex offset> <hex bytes or \"string\">)" % (lineno, line)) from None
        if not rest:
            raise PatchError("line %d: the offset %s has no value" % (lineno, head))
        if rest.startswith('"'):
            value = _string(rest, lineno)
        else:
            if len(rest) % 2 or any(c not in "0123456789abcdefABCDEF" for c in rest):
                raise PatchError("line %d: %r is not hex bytes (an even number of hex digits)" % (lineno, rest))
            value = bytes.fromhex(rest)
            if reverse:
                value = value[::-1]
        if module is None:
            raise PatchError("line %d: a patch line before any @nsobid- line" % lineno)
        if offset + shift < 0:
            raise PatchError("line %d: the offset %s plus offset_shift %#x is below 0" % (lineno, head, shift))
        out[module].append((offset + shift, value))
    return out, notes


def same_build(a, b):
    """Two build ids as patches write them: hex, case and trailing zeros aside"""
    return a.lower().rstrip("0") == b.lower().rstrip("0")


# ---------------------------------------------------------------- the modules

class Module:
    """An NSO's header and its three segments decompressed (none, LZ4, or ZBIC)"""
    def __init__(self, path):
        self.path, self.name = path, os.path.basename(path)
        with open(path, "rb") as f:
            raw = f.read()
        if raw[:4] != b"NSO0" or len(raw) < HEADER:
            raise ValueError("%s is not an NSO" % path)
        self.raw = raw
        self.flags = struct.unpack_from("<I", raw, 0xC)[0]
        self.build_id = raw[0x40:0x60].hex()
        self.segments = []   # (file offset, memory offset, size)
        self.data = []
        csize = struct.unpack_from("<3I", raw, 0x60)
        for i in range(3):
            foff, moff, size = struct.unpack_from("<3I", raw, 0x10 + 0x10 * i)
            d = raw[foff:foff + csize[i]]
            if self.flags & (1 << i):
                if self.flags & 0x80:
                    sys.path.insert(0, HERE)
                    import zbic
                    d = zbic.decompress(d, size)
                else:
                    import lz4.block
                    d = lz4.block.decompress(d, uncompressed_size=size)
            if len(d) != size:
                raise ValueError("%s segment %d: %d bytes, %d expected" % (path, i, len(d), size))
            self.segments.append((foff, moff, size))
            self.data.append(bytearray(d))
        self.original = [bytearray(d) for d in self.data]
        # (the .text's last page, up to the .rodata: there on the console and executable, where code caves go)
        text_off, text_size = self.segments[0][1], self.segments[0][2]
        self.cave_end = max(text_off + text_size, min(page_align(text_off + text_size), self.segments[1][1]))

    def segment_of(self, offset, length):
        """The segment index holding image bytes [offset, offset + length), or None (the .text's: up to the end of its last
        page, cave_end)"""
        for i, (_, moff, size) in enumerate(self.segments):
            end = self.cave_end if i == 0 else moff + size
            if moff <= offset and offset + length <= end:
                return i
        return None

    def grow(self, seg, end):
        """The segment made long enough for image bytes up to end (the .text into its last page: zeros, the game's and the
        patched copy alike, so only the patch's bytes count as changed)"""
        foff, moff, size = self.segments[seg]
        if end - moff > size:
            more = end - moff - size
            self.data[seg].extend(bytes(more))
            self.original[seg].extend(bytes(more))
            self.segments[seg] = (foff, moff, end - moff)

    def where(self, offset):
        """Words for an image offset outside the segments: which part of the module it falls in"""
        if offset < 0:
            return "the NSO header, before the code"
        for i, (_, moff, size) in enumerate(self.segments):
            if moff <= offset < moff + size:
                return "%s, and goes past its end (%#x)" % (SEGMENTS[i], self.cave_end if i == 0 else moff + size)
        if self.segments[0][1] + self.segments[0][2] <= offset < self.cave_end:
            return ".text's last page, and goes past its end (%#x)" % self.cave_end
        last = max(moff + size for _, moff, size in self.segments)
        if offset >= last:
            return "past the module's data (%#x: the bss, which the game clears at its start)" % last
        return "the gap between two segments"

    def write(self, path):
        """An uncompressed NSO of the patched segments: the header and what lies between it and the first segment (the module
        name) as they were, each segment's file offset and size set anew, the compression flags cleared and the segments'
        SHA-256 (0xa0, checked when flags bits 3-5 say so) made again from the patched bytes"""
        import hashlib
        hdr = bytearray(self.raw[:HEADER])
        body = bytearray(self.raw[HEADER:min(foff for foff, _, _ in self.segments)])
        for i, d in enumerate(self.data):
            struct.pack_into("<I", hdr, 0x10 + 0x10 * i, HEADER + len(body))
            struct.pack_into("<I", hdr, 0x18 + 0x10 * i, len(d))   # (the .text longer with a code cave)
            struct.pack_into("<I", hdr, 0x60 + 4 * i, len(d))
            hdr[0xA0 + 0x20 * i:0xC0 + 0x20 * i] = hashlib.sha256(d).digest()
            body += d
        struct.pack_into("<I", hdr, 0xC, self.flags & ~(0x80 | 7))
        with open(path + ".new", "wb") as f:
            f.write(hdr)
            f.write(body)
        os.replace(path + ".new", path)


def modules_of(exefs):
    """{file name: Module} of the exefs's NSOs"""
    mods = {}
    for name in sorted(os.listdir(exefs)):
        p = os.path.join(exefs, name)
        if name == "main.npdm" or not os.path.isfile(p):
            continue
        with open(p, "rb") as f:
            if f.read(4) != b"NSO0":
                continue
        mods[name] = Module(p)
    return mods


# ---------------------------------------------------------------- applying them

class Result:
    """What prepare() did: exefs (the folder the build reads), description (the json the lift reads, or None), hash, lines"""
    def __init__(self, exefs, description, digest, lines, applied):
        self.exefs, self.description, self.hash, self.lines, self.applied = exefs, description, digest, lines, applied


def _ranges(offsets):
    """Sorted offsets -> [[start, length]]"""
    out = []
    for o in offsets:
        if out and out[-1][0] + out[-1][1] == o:
            out[-1][1] += 1
        else:
            out.append([o, 1])
    return out


def prepare(exefs, root, work, say=print):
    """The patches of the mods under root applied to a copy of exefs (the game's own, decompressed and checked) in
    <work>/exefs_mods. Returns a Result; its exefs is the player's own when no patch changes a byte (nothing is written then)"""
    copy, description = os.path.join(work, COPY), os.path.join(work, DESCRIPTION)
    for old in (copy, description):   # (an earlier build's: never read by this one)
        if os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)
        elif os.path.exists(old):
            os.remove(old)
    files = patch_files(root)
    contents, lines, unread = {}, [], set()
    for rel, path, _ in files:
        try:
            with open(path, "rb") as f:
                contents[rel] = f.read()
        except OSError as e:
            contents[rel] = b""   # (counted as empty in the hash, as the game and the launcher do)
            unread.add(rel)
            lines.append("left out  %s: could not be read (%s)" % (rel, e.strerror or e))
    digest = files_hash(contents)
    if not files:
        return Result(exefs, None, digest, [], [])
    say("exefs mods: %d patch file%s in %s" % (len(files), "" if len(files) == 1 else "s", root))
    modules = modules_of(exefs)
    by_build = {m.build_id: name for name, m in modules.items()}
    owner = {}        # (module, image offset) -> index into applied
    applied = []      # [{"mod", "file", "module", "writes"}]
    conflicts = {}    # (earlier index, later index, module) -> [count, first offset]
    for rel, path, mod in files:
        if rel in unread:
            continue
        data = contents[rel]
        label = rel
        try:
            if rel.lower().endswith(".ips"):
                stem = os.path.basename(rel)[:-4]
                if not stem or any(c not in "0123456789abcdefABCDEF" for c in stem):
                    raise PatchError("its name is not a build id (an .ips file is named <build id>.ips)")
                parts, notes = {stem: parse_ips(data)}, []
            else:
                parts, notes = parse_pchtxt(data.decode("utf-8-sig", "replace"))
        except PatchError as e:
            lines.append("left out  %s: %s" % (label, e))
            continue
        lines += ["note      %s: %s" % (label, n) for n in notes]
        if not parts:
            lines.append("left out  %s: it names no module (no @nsobid- line)" % label)
            continue
        for bid, writes in parts.items():
            name = next((n for b, n in by_build.items() if same_build(b, bid)), None)
            if name is None:
                lines.append("left out  %s: made for the module %s, which is not one of this game's (a mod for another "
                             "version of the game?)" % (label, bid[:16].lower()))
                continue
            if name == "rtld":
                lines.append("left out  %s: it patches rtld, the loader, which ReLink does not run (it loads the modules "
                             "itself)" % label)
                continue
            if not writes:
                lines.append("left out  %s: no patch line for %s (none in an @enabled part)" % (label, name))
                continue
            m = modules[name]
            bad = None
            for off, value in writes:
                img = off - HEADER
                if not value:
                    continue
                if img < 0 or m.segment_of(img, len(value)) is None:
                    bad = "it writes %d bytes at %#x (image %s%#x), in %s" % (
                        len(value), off, "-" if img < 0 else "", abs(img), m.where(img))
                    break
            if bad:
                lines.append("left out  %s: %s. Nothing of it is applied" % (label, bad))
                continue
            index = len(applied)
            applied.append({"mod": mod, "file": rel, "module": name, "changed": 0})
            for off, value in writes:
                img = off - HEADER
                seg = m.segment_of(img, len(value)) if value else None
                if seg is None:
                    continue
                base = m.segments[seg][1]
                m.grow(seg, img + len(value))
                for k, byte in enumerate(value):
                    o = img + k
                    was = owner.get((name, o))
                    if was is not None and was != index and m.data[seg][o - base] != byte:
                        c = conflicts.setdefault((was, index, name), [0, o])
                        c[0] += 1
                    m.data[seg][o - base] = byte
                    owner[(name, o)] = index
    # (the bytes that differ from the game's in the end, each with the patch that wrote it last)
    changed = {}   # index -> [offsets]
    touched = set()
    for (name, o), index in owner.items():
        m = modules[name]
        seg = m.segment_of(o, 1)
        base = m.segments[seg][1]
        if m.data[seg][o - base] != m.original[seg][o - base]:
            changed.setdefault(index, []).append(o)
            touched.add(name)
    for index, p in enumerate(applied):
        n = len(changed.get(index, ()))
        p["changed"] = n
        lines.append("applied   %s: %s, %d byte%s changed" % (p["file"], p["module"], n, "" if n == 1 else "s"))
    for (a, b, name), (count, first) in sorted(conflicts.items()):
        lines.append("conflict  %s+%#x (%d byte%s): %s wins over %s" % (name, first, count, "" if count == 1 else "s",
                                                                       applied[b]["file"], applied[a]["file"]))
    for line in lines:
        say("exefs mods: " + line)
    if not touched:
        if applied:
            say("exefs mods: no byte of the game's code differs: the game's own modules are built")
        return Result(exefs, None, digest, lines, applied)
    os.makedirs(copy)
    for name in sorted(os.listdir(exefs)):
        src, dst = os.path.join(exefs, name), os.path.join(copy, name)
        if name in touched:
            modules[name].write(dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
    desc = {"format": 1, "patches": []}
    for index, p in enumerate(applied):
        if changed.get(index):
            desc["patches"].append({"mod": p["mod"], "file": p["file"], "module": p["module"],
                                    "ranges": _ranges(sorted(changed[index]))})
    with open(description, "w", encoding="utf-8", newline="\n") as f:
        json.dump(desc, f, indent=1, sort_keys=True)
        f.write("\n")
    say("exefs mods: %s patched (%d byte%s): the game is built from %s" % (
        ", ".join(sorted(touched)), sum(len(v) for v in changed.values()),
        "" if sum(len(v) for v in changed.values()) == 1 else "s", copy))
    return Result(copy, description, digest, lines, applied)


def write_report(out, result, extra=()):
    """relink_exefs_mods.txt in the game folder: the hash first (what the game and the launcher compare), then the lines.
    Written at the end of a build, with the game it describes"""
    with open(os.path.join(out, REPORT + ".new"), "w", encoding="utf-8", newline="\n") as f:
        f.write(result.hash + "\n")
        f.write("# The code patches (exefs) of the mods folder when this game was built: the first line is their hash, which the "
                "game and the launcher compare with the patch files there now. Add, change or remove one, then build the game "
                "again from the launcher.\n")
        for line in list(result.lines) + list(extra):
            f.write(line + "\n")
    os.replace(os.path.join(out, REPORT + ".new"), os.path.join(out, REPORT))


def read_report_hash(out):
    try:
        with open(os.path.join(out, REPORT), encoding="utf-8") as f:
            return f.readline().strip()
    except OSError:
        return None


# ---------------------------------------------------------------- what the lift sees (nx2/lift.py)

class Patched:
    """The description prepare() wrote, placed in the lift's image: words[pc] = the patch label that changed the word at pc
    (code), and data[(addr, n)] the rest"""
    def __init__(self, path, img):
        with open(path, encoding="utf-8") as f:
            desc = json.load(f)
        base = {m.name: m.base for m in img.modules}
        self.words, self.labels = {}, []
        for p in desc["patches"]:
            label = p["file"]
            self.labels.append(label)
            b = base.get(p["module"])
            if b is None:
                continue
            for start, n in p["ranges"]:
                for pc in range((b + start) & ~3, b + start + n, 4):
                    if img.in_code(pc):
                        self.words[pc] = label

    def at(self, pc):
        return self.words.get(pc)


def lift_check(L, img, patched):
    """After the lift's discovery: {function name: [patch labels]} of the translated functions a patch changes, and the
    report's lines for the changed code no function holds: in a function ReLink runs natively (host/: the patch has no
    effect), or nowhere the translator reached (code added in unused space and reached only through a pointer: it is not
    translated, and the game stops if it gets there)"""
    import bisect
    touched, covered = {}, set()
    words = patched.words
    if words:
        for e, (body, _) in L.funcs.items():
            hit = words.keys() & body.keys()
            if hit:
                covered |= hit
                touched["f_%08x" % e] = sorted({words[w] for w in hit})
    lines = []
    starts = sorted(set(L.eh_starts) | set(img.hooks))
    native, lost = {}, {}
    for pc in sorted(set(words) - covered):
        i = bisect.bisect_right(starts, pc) - 1
        s = starts[i] if i >= 0 else None
        if s is not None and s in img.hooks:
            native.setdefault((words[pc], img.hooks[s]), []).append(pc)
        else:
            lost.setdefault(words[pc], []).append(pc)
    name_of = lambda pc: next(("%s+%#x" % (m.name, pc - m.base) for m in img.modules
                               if m.base <= pc < m.base + len(m.image)), "%#x" % pc)
    for (label, hook), pcs in sorted(native.items()):
        sym = getattr(img, "hook_syms", {}).get(hook, hook)
        lines.append("no effect %s: it changes %s (%d word%s from %s), which ReLink runs with its own code: the patch "
                     "there does nothing" % (label, sym, len(pcs), "" if len(pcs) == 1 else "s", name_of(pcs[0])))
    for label, pcs in sorted(lost.items()):
        lines.append("not translated %s: %d instruction word%s it writes (from %s) are in no function the translator "
                     "reaches by a branch: code reached only through a pointer is not translated, and the game stops "
                     "there if it is ever run" % (label, len(pcs), "" if len(pcs) == 1 else "s", name_of(pcs[0])))
    return touched, lines


def site_lines(feature, pcs, patched, img):
    """Lines for the sites of one of ReLink's lift-time changes (nx2/oot_wide.py and the others) a patch changed: each left
    itself out (its instruction is not the one it was made for)"""
    out = []
    for pc in pcs:
        label = patched.at(pc)
        if label:
            m = next((m for m in img.modules if m.base <= pc < m.base + len(m.image)), None)
            out.append("warning   %s: it changes %s+%#x, where ReLink's %s goes: left out there" % (
                label, m.name if m else "?", pc - (m.base if m else 0), feature))
    return out
