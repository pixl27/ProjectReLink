# SPDX-License-Identifier: GPL-3.0-or-later
"""The load at create lets a loaded graphics program's shader modules go once every noted pipeline of its variant is built
(gpu/vk/vk_programs.cpp load_work, gpu/vk/vk_progcache.cpp program_cache_drop_loaded), and a new pipeline state in play has
them remade from hookshot_programs.bin on its worker (program_cache_complete_graphics). RX_VK_DROP_LOADED_MODULES=0 keeps them.

On Windows NVIDIA's driver keeps each module's SPIR-V in the game's private memory for as long as the module lives: a Market
stand held ~0.9 GB of modules made by the load at create, all of them for programs whose pipelines were
already built. Checked here without a GPU or game:
- vk_programs.cpp's load_work lets the modules go only where every noted pipeline of the variant was built (none deferred to
  a draw, none cold, none whose note did not read) and the switch is on, and counts them for the census line;
- program_cache_drop_loaded, cut out of vk_progcache.cpp with the module table and program_cache_complete_graphics, run on a
  real program cache file with tests/linux_pressure_test.cpp's stand-ins: a graphics program's two modules go and are remade
  with the same SPIR-V (and counted); a mesh fallback program, a compute program, a stage that cannot be remade and a second
  program of the same key and variant keep theirs;
- the switch: on unset, off for 0/off/false/no.

python -I -B tests/test_loaded_modules_drop.py [--cxx clang++]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def block(text, marker):
    if text.count(marker) != 1:
        raise SystemExit("FAIL: no single " + marker)
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def check_load_work():
    programs = (ROOT / "gpu/vk/vk_programs.cpp").read_text(encoding="utf-8")
    work = block(programs, "void load_work(LoadJob& job) {")
    for needle, why in (
            ("if (built && !left && drop_loaded_modules_on()) census.load_modules_dropped += (int64_t)program_cache_drop_loaded(*p);",
             "load_work does not let a loaded program's modules go once its noted pipelines are built"),
            ("census.load_pipes += pipeline_keep_ready(n.key, pl, p.get()), built++;", "a built noted pipeline is not counted"),
            ("else census.load_pipes_cold++, left++;", "a cold noted pipeline does not keep the modules"),
            ("census.load_pipes_deferred++, left++;", "a noted pipeline deferred to its draw does not keep the modules"),
            ("if (!pipe_note_at(notes_in, n.at, note.get())) { left++; continue; }", "a note that did not read does not keep them")):
        if needle not in work:
            raise SystemExit("FAIL: " + why + " (looked for: " + needle + ")")
    if "shader modules let go once their programs' noted pipelines were built" not in programs:
        raise SystemExit("FAIL: the census line does not say how many modules the load let go")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++") or "g++")
    args = parser.parse_args()
    check_load_work()

    shader = (ROOT / "gpu/vk/vk_shader.cpp").read_text(encoding="utf-8")
    cache = (ROOT / "gpu/vk/vk_progcache.cpp").read_text(encoding="utf-8")
    cache_h = (ROOT / "gpu/vk/vk_progcache.h").read_text(encoding="utf-8")
    header = (ROOT / "gpu/vk/vk_shader.h").read_text(encoding="utf-8")

    structs = "\n".join(block(header, m) + ";" for m in ("struct Stage {", "struct Program {"))
    structs += "\n" + block(cache_h, "struct DiskRecord {") + ";"
    functions = "\n".join(block(shader, m) for m in ("uint64_t answers_hash(", "bool answers_fit("))
    functions += "\n" + block(cache, "void idx_put(")
    functions += "\n" + block(cache_h, "struct Reader {") + ";\n"
    functions += cache[cache.index("void put_bytes("):cache.index("void program_cache_save(")]
    functions += block(cache, "void program_cache_save(") + "\n" + block(cache, "bool program_cache_payload(") + "\n"
    functions += block(cache, "void program_cache_read(") + "\n"
    functions += "namespace {\nstd::mutex light_mu;\n}\n"
    functions += block(cache, "namespace module_remake {") + "\n"
    functions += block(cache, "bool program_modules_dropped(") + "\n"
    functions += block(cache, "bool program_cache_complete_graphics(") + "\n"
    for m in ("bool drop_loaded_modules_on(", "int64_t program_cache_remade(", "size_t program_cache_drop_loaded("):
        functions += block(cache, m) + "\n"

    flags = ["-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-D_CRT_SECURE_NO_WARNINGS"]
    with tempfile.TemporaryDirectory(prefix="loaded-modules-") as tmp:
        folder = Path(tmp).resolve()
        (folder / "cache_types.inc").write_text(structs, encoding="utf-8")
        (folder / "cache_functions.inc").write_text(functions, encoding="utf-8")
        exe = folder / "loaded-modules-test.exe"
        subprocess.run([args.cxx, *flags, "-I" + str(folder), "-I" + str(ROOT / "tests"),
                        str(ROOT / "tests/loaded_modules_drop_test.cpp"), "-o", str(exe)], check=True, cwd=folder)
        subprocess.run([str(exe), str(folder)], check=True, timeout=30)
        keep = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TMP", "TEMP")}
        failed = []
        for value, want in ((None, "1"), ("1", "1"), ("0", "0"), ("off", "0"), ("false", "0"), ("no", "0")):
            env = dict(keep)
            if value is not None:
                env["RX_VK_DROP_LOADED_MODULES"] = value
            got = subprocess.run([str(exe), "--switch"], capture_output=True, text=True, timeout=30, env=env).stdout.strip()
            print(("ok   " if got == want else "FAIL ") + "RX_VK_DROP_LOADED_MODULES=%s: %s" % (value, got))
            if got != want:
                failed.append(str(value))
    if failed:
        print("FAIL: the switch")
        return 1
    print("loaded programs' shader modules: let go once their noted pipelines are built, remade for a new state; "
          "RX_VK_DROP_LOADED_MODULES=0 keeps them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
