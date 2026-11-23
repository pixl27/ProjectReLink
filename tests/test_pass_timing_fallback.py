"""RX_VK_PASS_TIMING names the mesh fallback's compute and an indirect mesh draw's argument conversion.

On a GPU that runs task and mesh programs as compute shaders (no mesh shaders, or RADV by default, the Steam Deck
among them), each fallback draw ends the pass and runs fills, full barriers and one or two compute dispatches before its
draw; an indirect mesh draw first copies its arguments with three barriers. Untimed, that work showed in the profile as
GPU time between the spans (~9 ms a frame on a Steam Deck in Hyrule Field) and its barriers as "barrier (other)". Each
is a span of its own: "fallback prog <key>" (kind "fallback") and "meshconv". A span opens after end_pass, which closes
the pass's own span (one span at a time a command buffer), and before the first command it should cover.

python -I -B tests/test_pass_timing_fallback.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAW = ROOT / "gpu" / "vk" / "vk_draw.cpp"
GPROF = ROOT / "gpu" / "vk" / "vk_gprof.h"
DEV = ROOT / "docs" / "development.md"
FALLBACK_SPAN = "GprofSpan span(cmd, label);"


def block(text, marker):
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def opened_between(body, name, span, first, what):
    """the span opens after the body's end_pass and before its first `first`"""
    if span not in body:
        raise SystemExit("%s: no RX_VK_PASS_TIMING span of its own (%s)" % (name, what))
    at = body.index(span)
    end_pass = body.find("end_pass(cmd);")
    if end_pass < 0 or end_pass > at:
        raise SystemExit("%s: the span opens before end_pass, inside the pass's own span, and is never timed" % name)
    if body.index(first) < at:
        raise SystemExit("%s: %s runs before the span opens: GPU time between the spans again" % (name, first))


def main():
    draw = DRAW.read_text(encoding="utf-8")

    record = block(draw, "bool fallback_record(VkCommandBuffer cmd, Program& p, const DrawArgs& d) {")
    if '"fallback prog %016llx"' not in record:
        raise SystemExit("fallback_record: its span is not labelled \"fallback prog <key>\" (kind \"fallback\")")
    opened_between(record, "fallback_record", FALLBACK_SPAN, "full_barrier(cmd);", "its fills, barriers and dispatches")
    for command in ("vkCmdFillBuffer(cmd, f->records", "vkCmdBindPipeline", "vkCmdDispatch"):
        if record.index(command) < record.index(FALLBACK_SPAN):
            raise SystemExit("fallback_record: %s runs before its span opens" % command)
    if "begin_pass(" in record:
        raise SystemExit("fallback_record begins a pass inside its span: the pass would go untimed")

    convert = block(draw, "bool mesh_indirect_convert(VkCommandBuffer cmd, const DrawArgs& d) {")
    opened_between(convert, "mesh_indirect_convert", 'GprofSpan gp(cmd, "meshconv ', "vkCmdPipelineBarrier2(",
                   "its copies and barriers")
    if "begin_pass(" in convert:
        raise SystemExit("mesh_indirect_convert begins a pass inside its span: the pass would go untimed")

    header = GPROF.read_text(encoding="utf-8")
    if '"fallback"' not in header or '"meshconv"' not in header:
        raise SystemExit("vk_gprof.h does not say which work the fallback and meshconv kinds hold")
    # (the profiling mode's list of what its report shows: its GPU item is the pass timing block, and it says what the two
    # kinds hold. Not RX_VK_PASS_TIMING's row in the switch table, whose neighbours change too often to edit it alongside)
    lines = DEV.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("- **GPU:** the log's last pass timing block")), None)
    if start is None:
        raise SystemExit("docs/development.md: the profiling mode's GPU item (the pass timing block) is missing")
    item = [lines[start]]
    for line in lines[start + 1:]:
        if not line.startswith("  "):
            break
        item.append(line)
    item = " ".join(part.strip() for part in item)
    if "`fallback prog <key>`" not in item or "`meshconv`" not in item:
        raise SystemExit("docs/development.md: the profiling mode's GPU item does not name the fallback and meshconv kinds")
    print("pass timing: the mesh fallback's compute and the indirect mesh argument conversion are spans of their own ok")


if __name__ == "__main__":
    main()
