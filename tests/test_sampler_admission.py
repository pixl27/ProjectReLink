"""Execute sampler creation and every handle-cache layer with authored failures, and a driver with a low sampler limit."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block   # noqa: E402

# (env_on, the switches' rule, before the code cut out of the sources: gpu/vk/compat/hookshot_env.h)
ENV_ON_H = str(Path(__file__).resolve().parents[1] / "gpu" / "vk" / "compat" / "hookshot_env.h")

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT)
    p.add_argument("--headers", type=Path, required=True)
    p.add_argument("--cxx", default="g++")
    p.add_argument("--sanitize", action="store_true")
    a = p.parse_args()
    source = (a.source / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    nvn = (a.source / "gpu/render/rx_nvn.h").read_text(encoding="utf-8")
    # The samplers' section up to its first function: what owns the samplers made (the book, sampler_room and the rest)
    head = source[source.index("// ---- samplers (by NVN sampler parameters) ----"):source.index("VkSamplerAddressMode wrap(")]
    # (compare_op and the pushed words' two functions are gpu/vk/draw_key.h's, which vk_draw.cpp includes; a --source from
    # before the header still has them in vk_draw.cpp, and they are cut from it as they were)
    key_h = a.source / "gpu/vk/draw_key.h"
    moved = ("VkCompareOp compare_op(", "void words_append(", "int words_same_store(")
    if key_h.is_file():
        key = key_h.read_text(encoding="utf-8")
        assert '#include "draw_key.h"' in source, "vk_draw.cpp does not include draw_key.h"
        for marker in moved:
            assert marker in key and marker not in source, "draw_key.h is not the one place that defines " + marker
    here = lambda markers: [m for m in markers if m not in moved or m in source]
    actual = head + "\n".join(block(source, marker) for marker in here([
        "VkSamplerAddressMode wrap(", "uint32_t aniso_override() {", "VkCompareOp compare_op(",
        "VkSampler sampler_make(int id, bool compare, bool one_level) {",
        "VkSampler sampler_for(int id, bool compare, bool one_level) {"]))
    # Keep standard-container behavior, replacing only the allocator to target an individual cache.
    owner = [
        ("std::unordered_map<SamplerInfo, Made, SamplerInfoHash> made", "FaultMap<SamplerInfo, Made, Owner, SamplerInfoHash> made"),
        ("std::unordered_map<uint64_t, Made*> keys", "FaultMap<uint64_t, Made*, Alias> keys"),
        ("std::vector<Retired> retired", "FaultVector<Retired, Retire> retired"),
    ]
    if "struct SamplerBook {" not in head:   # (a tree before sampler reuse, --source: one map by key owned them)
        owner = [("std::unordered_map<uint64_t, VkSampler> samplers", "FaultMap<uint64_t, VkSampler, Owner> samplers")]
    for old, new in owner + [
        ("std::unordered_map<int, std::pair<uint32_t, nvn_samp_par>> seen", "FaultMap<int, std::pair<uint32_t, nvn_samp_par>, Seen> seen"),
        ("std::unordered_map<uint64_t, VkSampler> mine", "FaultMap<uint64_t, VkSampler, Mine> mine"),
        ("std::unordered_set<uint64_t> failed", "FaultSet<uint64_t, Failure> failed"),
    ]:
        assert actual.count(old) == 1, old
        actual = actual.replace(old, new)
    start = nvn.rfind("typedef struct {", 0, nvn.index("} nvn_samp_par;"))
    params = nvn[start:nvn.index("} nvn_samp_par;", start) + len("} nvn_samp_par;")]
    symbols = sorted(set(re.findall(r"\bNVN_[A-Z0-9_]+", actual)))
    enums = "enum { " + ", ".join(name + "=" + str(i + 1) for i, name in enumerate(symbols)) + " };"
    fixture = (ROOT / "tests/sampler_admission_test.cpp").read_text(encoding="utf-8")
    descriptors = block(source, "struct Pushed {") + ";\n" + "\n".join(block(source, marker) for marker in here([
        "void words_append(", "int words_same_store(", "bool bind_descriptors("]))
    fixture = fixture.replace("/* DESCRIPTORS */", descriptors)
    fixture = fixture.replace("/* DRAW_KEY */", '#include "draw_key.h"' if key_h.is_file() else "namespace mkwvk {}")
    fixture = fixture.replace("/* PARAMS */", params).replace("/* ENUMS */", enums).replace("/* ACTUAL */", actual)
    cases = ["healthy", "host-oom", "device-oom", "unknown", "error-output", "generation-retry", "thread-retry",
             "independent", "valid-after-failure", "owner-commit", "seen-commit", "mine-commit", "failure-commit",
             "registration", "shared-thread", "concurrent", "nearest", "linear", "compare", "compare-explicit",
             "custom-border", "standard-border", "reduction", "compare-reduction", "anisotropy",
             "anisotropy-row", "anisotropy-game", "anisotropy-env-off", "anisotropy-change",
             "anisotropy-no-mips", "anisotropy-nearest-mip", "anisotropy-one-level",
             "upscale-bias", "upscale-bias-plain", "upscale-bias-change",
             "unknown-id", "force-linear", "wrap", "lod-clamp", "custom-reduction",
             "descriptor-push", "descriptor-pool", "descriptor-compute", "descriptor-whole",
             "limit", "limit-full", "limit-reuse", "limit-alias-commit", "limit-retire-commit"]
    failed = []
    has_reuse = "RX_VK_BINDING_REUSE" in actual
    count = len(cases) * 3 + int(has_reuse)
    with tempfile.TemporaryDirectory(prefix="sampler-admission-") as temp:
        cpp, exe = Path(temp) / "fixture.cpp", Path(temp) / "fixture.exe"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-include", ENV_ON_H, "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
                        "-pthread", "-I" + str(a.headers), "-I" + str(a.source / "gpu/vk"),
                        *(["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-pie", "-no-pie"] if a.sanitize else []),
                        str(cpp), "-o", str(exe)], check=True)
        for route in ["id", "make", "cache-off"]:
            for case in cases:
                r = subprocess.run([str(exe), route, case], timeout=20, capture_output=True, text=True)
                if r.returncode:
                    last = (r.stderr.strip().splitlines() or ["exit %d" % r.returncode])[-1]
                    failed.append(f"{route} {case}: {last}")
        if has_reuse:
            r = subprocess.run([str(exe), "reuse", "unrelated-registrations"], timeout=20, capture_output=True, text=True)
            if r.returncode:
                failed.append(f"binding reuse: {r.stderr.strip() or r.returncode}")
    for f in failed:
        print("FAIL: " + f)
    if failed:
        print(f"{len(failed)} of {count} sampler cache/admission/policy/limit cases failed")
        return 1
    print(f"{count} sampler cache/admission/policy/limit cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
