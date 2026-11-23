"""Where the lifted functions begin in the lift's C (code_*.c): each definition is one line, `void f_<address>(Ctx *__restrict
A64_NOESC c) {`, its name in group 1. Small on purpose: nx2/build.py reads it for every compile, without the lifter.

The lift writes LF or CRLF. A definition starts at column zero. When cutting chunks, the single comment line
the lifter sometimes writes immediately before a definition belongs to the same function.
"""
import re

_DEF = r"^void (f_[0-9a-f]+)\(Ctx \*__restrict A64_NOESC c\) \{"
HEAD = re.compile(_DEF, re.M)
HEAD_BYTES = re.compile(_DEF.encode(), re.M)
_NOTE = re.compile(r"/\* .* \*/\r?\n")


def definitions(text):
    """Yield definition matches. The literal newline and prefix let find skip the body between definitions."""
    pattern, prefix = (HEAD, "\nvoid f_") if isinstance(text, str) else (HEAD_BYTES, b"\nvoid f_")
    at = 0
    while True:
        m = pattern.match(text, at)
        if m:
            yield m
        newline = text.find(prefix, at)
        if newline < 0:
            return
        at = newline + 1


def noted_starts(text):
    """Chunk boundaries, including the preceding note when the entire line is a lifter comment."""
    for m in definitions(text):
        start = m.start()
        if text.endswith((" */\n", " */\r\n"), 0, start):
            previous = text.rfind("\n", 0, start - 1) + 1
            if _NOTE.fullmatch(text, previous, start):
                start = previous
        yield start
