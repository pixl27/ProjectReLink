"""Check that opt-in late-page diagnostics report timeout truthfully.

Run with: python3 -I -B nx2/test_vk_capture_wait_diag.py
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def main() -> None:
    source = (ROOT / "gpu" / "vk" / "vk_memory.cpp").read_text(encoding="utf-8")
    take = function(source, "bool take_capture(")

    assert "uint32_t wait_cap_ms = 250" in take
    assert "wait_done = vk_timeline_wait_for(c->late_value, wait_cap_ms)" in take
    assert "wait_done = vk_timeline_completed() >= c->late_value" in take
    assert 'wait_done ? "completed" : "CUT"' in take
    assert "with %u ms cap" in take

    print("vk_capture_wait_diag: ok")


if __name__ == "__main__":
    main()
