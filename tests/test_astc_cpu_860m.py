"""Windows AMD iGPU fallbacks for ASTC decoding and cached graphics programs.

PCI 1002:1114 is the 860M in #1040. The same Kokiri Forest arrays lose the device on the GPU decoder. The desktop
iGPUs 1002:164E and 1002:13C0 already default to the CPU. Another AMD id, and the same id on another vendor, stay on the
GPU, and off Windows the default stays the GPU. The rule's one line is cut out of device_policy.h and run against the
facts it reads; the switch over it (RX_VK_ASTC_CPU=0 and =1, device_policy_env) is tests/test_device_policy.py's.
On the Vega iGPU 1002:15E7, also check the synchronous program-cache default and its explicit overrides.

python -I -B tests/test_astc_cpu_860m.py [--cxx g++]
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
AMD = 0x1002
NVIDIA = 0x10DE
MARKER = "    p.astc_on_cpu = "


def rule():
    policy = (ROOT / "gpu/vk/device_policy.h").read_text(encoding="utf-8")
    if policy.count(MARKER) != 1:
        raise SystemExit("gpu/vk/device_policy.h: %r appears %d times, not once" % (MARKER.strip(), policy.count(MARKER)))
    at = policy.index(MARKER)
    return policy[at:policy.index(";", at) + 1]


def shader_rule():
    source = (ROOT / "gpu/vk/vk_progcache.cpp").read_text(encoding="utf-8")
    at = source.index("bool async_loads_on() {")
    policy = (ROOT / "gpu/vk/device_policy.h").read_text(encoding="utf-8")
    start = policy.index("inline bool windows_vega_compat(")
    helper = policy[start:policy.index("\n}", start) + 2]
    return "namespace policy_rule {\n" + helper + "\n}\n" + source[at:source.index("\n}", at) + 2]


SOURCE = """
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct DeviceFacts {
    uint32_t vendor_id = 0, device_id = 0;
    bool windows = false;
    bool amd() const { return vendor_id == 0x1002; }
};
struct Policy { bool astc_on_cpu = false; };
struct Device { DeviceFacts facts; } dev;
%s

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    DeviceFacts& f = dev.facts;
    f.windows = atoi(argv[1]) != 0;
    f.vendor_id = (uint32_t)strtoul(argv[2], 0, 0);
    f.device_id = (uint32_t)strtoul(argv[3], 0, 0);
    Policy p;
%s
    printf("%%d %%d\\n", p.astc_on_cpu ? 1 : 0, async_loads_on() ? 1 : 0);
    return 0;
}
"""

CASES = [
    (1, AMD, 0x1114, 1, "860M (PCI 1002:1114) still decodes ASTC on the GPU by default"),
    (1, AMD, 0x164E, 1, "Ryzen 7000 desktop iGPU"),
    (1, AMD, 0x13C0, 1, "Ryzen 9000 desktop iGPU"),
    (1, AMD, 0x15E7, 1, "Vega on Ryzen 5 5625U: Water Temple walls and minimap"),
    (0, AMD, 0x15E7, 0, "Vega off Windows stays on the GPU decoder"),
    (1, AMD, 0x164F, 0, "another AMD id stays on the GPU decoder"),
    (1, NVIDIA, 0x1114, 0, "the 860M's id on another vendor"),
    (0, AMD, 0x1114, 0, "off Windows the default stays the GPU"),
]


def main():
    cxx = "g++"
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--cxx":
        cxx = args[1]
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / "astc.cpp", Path(d) / "astc"
        src.write_text(SOURCE % (shader_rule(), rule()), encoding="utf-8")
        r = subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-o", str(exe), str(src)],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr or r.stdout)
        for windows, vendor, device, want, why in CASES:
            default_async = int(not (windows and vendor == AMD and device == 0x15E7))
            for override, async_expected in ((None, default_async), ("0", 0), ("1", 1)):
                env = dict(os.environ)
                env.pop("RX_VK_ASYNC_LOAD", None)
                if override is not None:
                    env["RX_VK_ASYNC_LOAD"] = override
                r = subprocess.run([str(exe), str(windows), hex(vendor), hex(device)], env=env,
                                   capture_output=True, text=True)
                if r.returncode:
                    raise SystemExit(r.stderr or "the probe exited %d" % r.returncode)
                expected = f"{want} {async_expected}"
                if r.stdout.strip() != expected:
                    raise SystemExit(f"{why}: ASTC and async load got {r.stdout.strip()}, want {expected} (override {override})")
    print("ok")


if __name__ == "__main__":
    main()
