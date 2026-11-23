"""A pipeline belongs to the program object it was built from (gpu/vk/vk_draw.cpp graphics_pipeline). The key names a program by
its code and variant hash, and two program objects can share both: a pipeline the precompile or a program's create call built
from one was drawn with the other, its shaders under the other's layout and constants, and the GPU never finished the draw (the
device lost near water on NVIDIA cards). Checked by source, 2026-11-15:

- PipeEntry keeps its program object; every place that makes one stores it (graphics_pipeline, the precompile);
- kept-ready pipelines carry the object they were made from (pipeline_keep_ready, both callers), and only it takes them;
- an entry made for another object is not returned: this one gets its own, under a key of its own, not noted;
- RX_VK_PIPE_BY_PROGRAM=0 turns it off, read with the switches' one rule (env_value_off).

python -I -B tests/test_pipe_by_program.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    draw = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    entry = block(draw, "struct PipeEntry {")
    assert "std::atomic<const Program*> prog{nullptr};" in entry
    assert "std::unordered_map<uint64_t, std::pair<VkPipeline, const Program*>> kept_ready;" in draw
    gp = block(draw, "PipeEntry* graphics_pipeline(")
    for need in ('const char* e = getenv("RX_VK_PIPE_BY_PROGRAM"); return !(e && env_value_off(e));',
                 "const uint64_t own = mix(key, (uint64_t)(uintptr_t)&p);",
                 "if (made_for && made_for != &p) {", "key = own, own_key = true;",
                 "e->prog.store(&p, std::memory_order_release);",
                 "k->second.second == &p", "mine.find(mkey)"):
        assert need in gp, "graphics_pipeline: " + need
    assert gp.index("if (made_for && made_for != &p) {") < gp.index("pipelines.emplace(key")
    assert "e->prog.store(p, std::memory_order_release);   // (whose pipeline it is: graphics_pipeline)" in draw
    assert "kept_ready.try_emplace(key, std::make_pair(pl, made_from))" in draw
    assert "pipeline_keep_ready(r.key, pl, &p)" in draw   # (the load at create calls' pass: precompile_at_create)
    progs = (ROOT / "gpu/vk/vk_programs.cpp").read_text(encoding="utf-8")
    assert "pipeline_keep_ready(n.key, pl, p.get())" in progs
    print("PASS: a pipeline is drawn only with the program object it was built from")


if __name__ == "__main__":
    main()
