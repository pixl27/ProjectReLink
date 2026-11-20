"""Build the movie policy test and verify eager compilation obeys it.

Run with: python3 -I -B nx2/test_movie_upscale_policy.py [--cxx clang++]
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()

    draw = (ROOT / "gpu" / "vk" / "vk_draw.cpp").read_text(encoding="utf-8")
    eager = function(draw, "void eager_compute(")
    decision = eager.index("movie_upscale_policy::use_linear(")
    build = eager.index("make_compute_pipeline(")
    assert decision < build
    assert "movie_upscale_mode(), false, movie_keep_all_networks()" in eager
    assert "uses_cooperative_matrix" in eager
    assert "gameplay_network(cp->key)" in eager

    source = ROOT / "gpu" / "vk" / "tests" / "movie_upscale_policy_test.cpp"
    with tempfile.TemporaryDirectory(prefix="movie-upscale-policy-") as temporary:
        executable = Path(temporary) / "movie_upscale_policy_test"
        subprocess.run(
            [args.cxx, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)

    print("movie_upscale_policy: linear mode skips only unused movie pipelines")


if __name__ == "__main__":
    main()
