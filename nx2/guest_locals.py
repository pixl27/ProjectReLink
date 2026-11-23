"""(hookshot) Guest registers and flags as C locals: the kit's A64_LOCAL_REGS / A64_LOCAL_FLAGS, on by default and made
to fit the game's patches. A function's most used guest registers (6) and the four flags live in C locals, written back
to the context only where something else may read them (the kit's context marks), where the lifted code otherwise
loads and stores the context around every instruction.

The kit's files are not edited (third_party/: CONTRIBUTING.md). install(), called from nx2/lift.py after fastvec's, and
the keep_* functions lift.py asks before the functions are written change these from outside:

  - the defaults: a64_lift.local_regs / local_flags (which the kit's fast float regions read too) answer 6 registers and flags on
    with the variables unset; A64_LOCAL_REGS=0 and A64_LOCAL_FLAGS=0 turn each off (the kit took any A64_LOCAL_FLAGS
    value, "0" too, as on). Lift-time switches: the lift key holds them.
  - a function that keeps its registers in the context (Lifter.no_local) keeps its flags there too. The kit keeps the
    registers of such a function but makes its flags locals all the same; the game's patches find their places by the
    context's flags (c->n, c->z ...) and the host code they call reads them. localize_regs is wrapped for that, and
    emit_function tells it which function is being written.
  - keep_entered_below(): a function whose code starts below its entry joins no_local: nx2/lift_fixes.py jumps to
    the entry from the function's head, over the place where the locals are loaded.
  - keep_not_ran(): only the functions nx2/ran_funcs.txt names (play was seen to run them) may have locals; the rest
    join no_local and are written as without this change (A64_LOCAL_FUNCS=all: every function may).
  - keep_spliced(): a function with C spliced after one of its instructions (Lifter.after: widescreen, the field of
    view, the shadow cascades, DLSS jitter) joins no_local when it may access integer registers or flags. Explicitly
    marked vector-only hooks leave those locals intact.
  - a function the game's patches would change keeps the context as well: install(touched) takes a function that
    answers that for one function's text as it reads with the context kept (nx2/oot_patches.py's touches), and
    localize_regs asks it of every function before giving it locals. The patches find their places by the lifted text
    (c->x[8], c->n ...) and the host code they call reads the context, so such a function must not have locals; asking
    the patches themselves means no list of them is kept by hand. An edit may want more than its own function in view
    before it acts, and then answers no for a function it would change in a whole file; so, on the safe side,
    keep_mentioned() first keeps every function that holds an address the patch file names (its edits are pinned to
    those). check_patched() looks at the written files afterwards and stops the lift if a function carries both a
    patch's mark and locals."""
import glob
import os
import re

import a64_lift

REGS = 6   # (measured 2026-11-15: 4 and 8 are slower than 6; 8 spills on x86)


def local_regs():
    """A64_LOCAL_REGS: how many of a function's guest registers become locals (0: none; unset: REGS)"""
    v = os.environ.get("A64_LOCAL_REGS", "")
    return int(v) if v.isdigit() else REGS


def local_flags():
    """the four flags as locals, unless A64_LOCAL_FLAGS=0, or the kit's A64_PACKED_FLAGS experiment is set (any value,
    as the kit's pack_flags reads it): it folds a flag-setting instruction's four stores into a64_set_nzcv(c, ...),
    which writes the context where localize_regs does not see it, so a later B.EQ would read a stale local Z"""
    if os.environ.get("A64_PACKED_FLAGS"):
        return False
    return os.environ.get("A64_LOCAL_FLAGS", "1")[:1] != "0"


def keep_spliced(lifter):
    """the functions with C spliced after an instruction keep the context: into lifter.no_local. Returns how many"""
    if not lifter.after:
        return 0
    from oot_wide import VectorOnly
    spliced = {e for e, (body, _) in lifter.funcs.items()
               if any(pc in body and not isinstance(text, VectorOnly) for pc, text in lifter.after.items())}
    lifter.no_local |= spliced
    return len(spliced)


def keep_entered_below(lifter):
    """the functions whose code starts below their entry keep the context: into lifter.no_local. Returns how many.
    nx2/lift_fixes.py writes `goto L_<entry>;` right after such a function's head, which is where localize_regs
    declares the locals and loads them from the context: the jump would pass over the loads, and the function would
    run on whatever the stack holds (26 functions in OoT). A PLT stub is written without its body and is not one"""
    below = {e for e, (body, _) in lifter.funcs.items() if e not in lifter.plt and body and min(body) != e}
    lifter.no_local |= below
    return len(below)


RAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ran_funcs.txt")


def ran_functions():
    """the functions that may get locals (A64_LOCAL_FUNCS): unset, the ones nx2/ran_funcs.txt names (the functions a
    100-minute play session ran, recorded with scripts/coverage/cover.py: 77,725 of the game's ~295,000);
    `all`, every function (None). No such file: every function too. The file's contents are in the lift key (a .txt in
    nx2/: make_game.py's lift_sources)"""
    if os.environ.get("A64_LOCAL_FUNCS", "").strip().lower() == "all" or not os.path.isfile(RAN):
        return None
    with open(RAN, encoding="ascii") as f:
        return {int(w[2:], 16) for line in f for w in line.split()[:1] if w.startswith("f_")}


def keep_not_ran(lifter, ran=None):
    """the functions no play session was seen to run keep the context: into lifter.no_local. Returns how many, or None
    when every function may have locals. Locals cost compile time in every function and pay only where the code runs;
    ~58% of the lifted C's estimated compile is for functions the list does not hold. Such a function is written as
    main writes it (registers and flags in the context), so one the list misses that does run is as fast as before
    this change, never slower. ran: a set of entries (ran_functions() when not given)"""
    ran = ran_functions() if ran is None else ran
    if ran is None:
        return None
    cold = {e for e in lifter.funcs if e not in ran and e not in lifter.plt}
    lifter.no_local |= cold
    return len(cold)


def keep_mentioned(lifter, addresses):
    """the functions that hold one of `addresses` (an entry, or an instruction's address) keep the context: into
    lifter.no_local. Returns how many (nx2/oot_patches.py's mentioned_addresses: the places its edits are pinned to)"""
    import bisect
    entries = sorted(lifter.funcs)
    found = set()
    for a in addresses:
        i = bisect.bisect_right(entries, a) - 1
        e = entries[i] if i >= 0 else None
        if e is not None and (e == a or a in lifter.funcs[e][0]):
            found.add(e)
        else:   # (not in the function that starts before it: code shared with another, or no code at all)
            found.update(x for x, (body, _) in lifter.funcs.items() if a in body)
    lifter.no_local |= found
    return len(found)


PATCH_MARK = "(nx2/oot_patches.py"   # (what a patch's inserted line carries)
LOCAL_DECL = re.compile(r"^    uint(?:64_t x\d+_ = c->x\[|8_t fn_ = c->n)", re.M)   # (localize_regs' declarations)
HEAD = re.compile(r"^void f_([0-9a-f]+)\(Ctx \*__restrict A64_NOESC c\) \{$", re.M)


def check_patched(gen):
    """the written C: no function with a patch's mark has locals (the patch's host code would read a stale context).
    Stops the lift naming them; returns how many patched functions it looked at"""
    bad, seen = [], 0
    for path in sorted(glob.glob(os.path.join(glob.escape(gen), "code_*.c"))):
        with open(path, encoding="utf-8", newline="") as f:
            s = f.read()
        if PATCH_MARK not in s:
            continue
        heads = list(HEAD.finditer(s))
        for k, m in enumerate(heads):
            fn = s[m.start():heads[k + 1].start() if k + 1 < len(heads) else len(s)]
            if PATCH_MARK in fn:
                seen += 1
                if LOCAL_DECL.search(fn):
                    bad.append(m.group(1))
    if bad:
        raise SystemExit("locals: %d patched functions were given locals (%s): the patch's text was not found by "
                         "nx2/oot_patches.py's touches() when the function was written" % (len(bad), " ".join(bad[:12])))
    return seen


_keep = [False]   # (the function being emitted keeps its context: emit_function below sets it for localize_regs)
_kept = [0]       # (functions the patches would change, kept in the context by localize_regs since emit_function last read it)
_installed = False


def install(touched=None):
    """the defaults, and localize_regs and emit_function wrapped. Once; around the kit's emit_function (its fast float regions' too), which this
    wraps. touched(text): whether the game's patches would change this function, given with the context kept"""
    global _installed
    if _installed:
        return
    _installed = True
    a64_lift.local_regs = local_regs   # (the kit's fast float regions read it there too: a64_lift.py's ff_region)
    a64_lift.local_flags = local_flags
    base_localize = a64_lift.localize_regs
    base_emit = a64_lift.Lifter.emit_function

    def kept(text):
        """the function with its registers and flags in the context"""
        a64_lift.local_flags = lambda: False
        try:
            return base_localize(text, False, None)
        finally:
            a64_lift.local_flags = local_flags

    def localize_regs(text, local, dirty):
        if _keep[0]:
            return kept(text)
        if touched is not None and (local or local_flags()):
            plain = kept(text)
            if touched(plain):   # (a patch finds its place here: it stays as the patch expects it)
                _kept[0] += 1
                return plain
        return base_localize(text, local, dirty)

    def emit_function(self, e, texts):
        _keep[0] = e in self.no_local
        _kept[0] = 0
        try:
            return base_emit(self, e, texts)
        finally:
            _keep[0] = False
            if _kept[0]:
                self.stats["locals_kept_for_patches"] += 1

    a64_lift.localize_regs = localize_regs
    a64_lift.Lifter.emit_function = emit_function


class _Both(str):
    """an instruction's C as the lift writes it; .raw: the translator's text it was made from, for the same instruction
    with the flags in the context (.plain, made the first time a function asks for it)"""
    raw = None
    plain = None


class _Plain:
    """a function's view of the instructions' texts as written with the flags in the context (emit_function reads
    texts[key] only); plain_of makes such a text when it is first asked for"""

    def __init__(self, texts, plain_of):
        self.texts = texts
        self.plain_of = plain_of

    def __getitem__(self, key):
        text, flags = self.texts[key]
        return (self.plain_of(text) if isinstance(text, _Both) else text), flags


def install_plain():
    """op_text and emit_function wrapped, after install() (lift.py): a function that keeps its registers in the context
    because a game patch names it (Lifter.no_local: nx2/oot_patches.py's addresses, the spliced and hooked ones) is written
    with the kit's context-flag text for every instruction too. The kit decides once for the whole lift how an instruction
    reads and writes the flags (a scalar compare is `{ if (!a64_fcmp32(c,a,b)) ... }` with the flags in the context and
    `_fk = a64_fcmp32k(...)` with local flags), whatever function the text ends up in; a patch finds its place by the
    lifted text (the horse's jump blend and the moving floors' carry look for `a64_fcmp32(c,...)`).

    So an instruction's text keeps the translator's text it came from, and only when a function in no_local is written
    are its instructions made once more with the flags in the context, once per distinct text (cached on it). With
    nx2/ran_funcs.txt most functions are in no_local (those play did not run), so most texts are made twice: a few
    seconds of a lift, against the compile time the list saves.
    (The FPX shapes an instruction needs are noted at its first writing, from the same translator text; the header that
    lists them is written before any function, so the second writing notes them into a set of its own.)"""
    base_op_text = a64_lift.op_text
    base_emit = a64_lift.Lifter.emit_function

    def op_text(t, fpx):
        r = base_op_text(t, fpx)
        if not a64_lift.local_flags():
            return r
        r = _Both(r)
        r.raw = t
        return r

    def plain_of(text):
        if text.plain is None:
            keep = a64_lift.local_flags
            a64_lift.local_flags = lambda: False
            try:
                text.plain = base_op_text(text.raw, set())
            finally:
                a64_lift.local_flags = keep
        return text.plain

    def emit_function(self, e, texts):
        return base_emit(self, e, _Plain(texts, plain_of) if e in self.no_local else texts)

    a64_lift.op_text = op_text
    a64_lift.Lifter.emit_function = emit_function
