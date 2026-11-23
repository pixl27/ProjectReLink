# SPDX-License-Identifier: GPL-3.0-or-later
"""A handle table's elements behind the game's own count of its records name no texture.

A shader that reads a texture handle from a table at an index of its own gets a descriptor array of HANDLE_TABLE_SIZE (64)
elements, and stage_handles computes element k's handle from the game's table (OoT's light table: a 16-byte header, then
96-byte records, the shadow map's handle in each). The game's table holds as many records as the frame needs (3 in Kokiri
Forest, about 20 in Castle Town by day); the records behind them are leftovers of earlier frames, and their handles name
slots the game has released and given to other textures since. The draw fetched all 64, and a fetch uploads a texture
from guest memory when a GPU buffer write has marked it: a 3840x2160 target or a 4096x4096 depth on every present, 4 to 8 GB
in the 15 seconds after the pause menu closes. No pixel asks for those elements: the lists a pixel's light index comes
from hold only numbers under the table's count.

stage_handles now gives 0 (no texture) for the elements at and behind the table's count, without computing them, where the
table says it is one: word 0 the number of records, word 2 the capacity, which must be the storage buffer's size in whole
records past the header, rounded down, records of one size from byte 16 with element k's handle inside record k (elements 0,
1, 2 and the last on one line, one storage read each). Any other table (a masked index, a nested read, a header that cannot
be read): every element, as before. RX_VK_HANDLE_TABLE_COUNT=0: every element.

Built from the sources' own text (vk_shader.cpp's stage_handles and what it calls, shader_info.h's HandleOp and
EvaluateHandleProgram), with stubs for the banks and memory, and run both ways.
    python -I -B tests/test_handle_table_count.py [--cxx clang++]
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

PRELUDE = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>
using u32 = uint32_t; using s32 = int32_t; using u64 = uint64_t;
namespace Shader {
struct StorageBufferDescriptor { u32 cbuf_index, cbuf_offset, count = 1; };
struct Info {
 static constexpr size_t MAX_SSBOS{64};
 std::vector<StorageBufferDescriptor> storage_buffers_descriptors;
 std::vector<std::vector<u32>> handle_programs;
};
}
struct NvnState {};
struct Stage { Shader::Info info; };
// a constant bank of 64 words (c3); guest memory of 4096 bytes at 0x10000
u32 bank[64]{}; u32 memory[1024]{}; int reads = 0;   // (reads: the reads of guest memory)
bool word0_fails = false;   // (a read that covers the table's word 0 fails: ub_read's false, its bytes zeros)
uint32_t stage_cbuf_word(const NvnState&, const Stage&, uint32_t index, uint32_t offset) {
 return index == 3 && offset / 4 < 64 ? bank[offset / 4] : 0;
}
bool ub_read(uint64_t addr, void* out, size_t n) {
 ++reads;
 std::memset(out, 0, n);
 if ((word0_fails && addr <= 0x10000 && addr + n > 0x10000) || addr < 0x10000 || addr + n > 0x10000 + 4096) return false;
 std::memcpy(out, reinterpret_cast<const char*>(memory) + (addr - 0x10000), n);
 return true;
}
void logf(const char*, ...) {}
'''

CHECKS = r'''
// The light table's layout: storage buffer 0's address in c3[0x10..0x18], its size in c3[0x18]; at 0x10000 a header (word 0 the
// number of records, word 2 the capacity) and 96-byte records from byte 16, the handle at +0x50 of each: element k reads 0x60 + 96k
static void table(u32 records, u32 capacity, u32 size, u32 filled) {
 std::memset(memory, 0, sizeof memory);
 bank[4] = 0x10000, bank[5] = 0, bank[6] = size;
 memory[0] = records, memory[2] = capacity;
 for (u32 k = 0; k < filled; k++) memory[(16 + 96 * k + 0x50) / 4] = 0x200 + k;   // (the frame's records and the leftovers behind them)
}
int main(int argc, char** argv) {
 const bool on = argc > 1 && std::string(argv[1]) == "on";   // (what RX_VK_HANDLE_TABLE_COUNT this process was started with)
 int failures = 0;
 auto eq = [&](u64 actual, u64 expected, const std::string& label) {
  if (actual != expected) { std::cerr << label << ": " << actual << " != " << expected << '\n'; ++failures; }
 };
 using Op = Shader::HandleOp;
 NvnState s; Stage st;
 st.info.storage_buffers_descriptors = {{3, 0x10}};
 st.info.handle_programs = {
  {u32(Op::Index), u32(Op::Immediate), 96, u32(Op::Mul), u32(Op::Immediate), 0x60, u32(Op::Add), u32(Op::Storage), 0},   // the light table's
  {u32(Op::Index), u32(Op::Immediate), 96, u32(Op::Mul), u32(Op::Immediate), 4, u32(Op::Add), u32(Op::Storage), 0},      // a table without that header
  // a masked index, (k & 3) * 96 + 16: elements 0-2 on the light table's line, 4 and up naming live records 0-3 again
  {u32(Op::Index), u32(Op::Immediate), 3, u32(Op::And), u32(Op::Immediate), 96, u32(Op::Mul), u32(Op::Immediate), 16, u32(Op::Add),
   u32(Op::Storage), 0},
  // a nested read: the record's number read from an index list at 0x800 of the same buffer first
  {u32(Op::Index), u32(Op::Immediate), 4, u32(Op::Mul), u32(Op::Immediate), 0x800, u32(Op::Add), u32(Op::Storage), 0,
   u32(Op::Immediate), 96, u32(Op::Mul), u32(Op::Immediate), 0x60, u32(Op::Add), u32(Op::Storage), 0},
 };
 u32 out[8];
 auto handles = [&](u32 program, u32 count) { std::fill(out, out + 8, 0xdeadu); stage_handles(s, st, program, count, out); };

 // 3 records of a capacity of 8, five leftovers behind them: the leftovers name nothing
 table(3, 8, 16 + 8 * 96 + 48, 8);
 reads = 0;
 handles(0, 8);
 eq(reads, on ? 4 : 8, "only the game's records are read (the header in one read and 3 handles)");
 for (u32 k = 0; k < 3; k++) eq(out[k], 0x200 + k, "a record of the frame keeps its handle (element " + std::to_string(k) + ")");
 for (u32 k = 3; k < 8; k++)
  eq(out[k], on ? 0 : 0x200 + k, "a leftover behind the count names " + std::string(on ? "nothing" : "its old slot with the switch off") +
                                 " (element " + std::to_string(k) + ")");
 // the table holds more records than the array has elements: all of them, as before
 table(8, 8, 16 + 8 * 96, 8);
 handles(0, 4);
 for (u32 k = 0; k < 4; k++) eq(out[k], 0x200 + k, "a count over the array's size cuts nothing");
 // no records this frame: nothing is named
 table(0, 8, 16 + 8 * 96, 8);
 handles(0, 8);
 for (u32 k = 0; k < 8; k++) eq(out[k], on ? 0 : 0x200 + k, "an empty table names nothing");
 // not that table: the capacity is not what the buffer holds, the count is over the capacity, no size, another layout
 table(3, 9, 16 + 8 * 96, 8);
 handles(0, 8);
 for (u32 k = 0; k < 8; k++) eq(out[k], 0x200 + k, "a capacity the buffer's size does not hold: every element");
 table(9, 8, 16 + 8 * 96, 8);
 handles(0, 8);
 for (u32 k = 0; k < 8; k++) eq(out[k], 0x200 + k, "a count over the capacity: every element");
 table(3, 8, 0, 8);
 handles(0, 8);
 for (u32 k = 0; k < 8; k++) eq(out[k], 0x200 + k, "a buffer without a size: every element");
 table(3, 8, 16 + 8 * 96, 0);
 for (u32 k = 0; k < 8; k++) memory[(96 * k + 4) / 4] = 0x300 + k;   // (handles at 96k + 4: the word at byte 4 is in no record from 16)
 handles(1, 8);
 for (u32 k = 0; k < 8; k++) eq(out[k], 0x300 + k, "a table whose handles are not inside records from byte 16: every element");
 // the count's word can't be read while the capacity's can: not taken for an empty table, every element as before
 table(3, 8, 16 + 8 * 96 + 48, 8);
 word0_fails = true;
 handles(0, 8);
 word0_fails = false;
 for (u32 k = 0; k < 8; k++) eq(out[k], 0x200 + k, "an unread count names every element (element " + std::to_string(k) + ")");
 // a masked index under a header that would pass: every element, as the program gives it (a pixel's element 5 is record 1)
 table(3, 8, 16 + 8 * 96 + 48, 0);
 for (u32 k = 0; k < 8; k++) memory[(16 + 96 * k) / 4] = 0x500 + k;
 handles(2, 8);
 for (u32 k = 0; k < 8; k++) eq(out[k], 0x500 + (k & 3), "a masked index: every element (element " + std::to_string(k) + ")");
 // a nested read under a header that would pass: every element
 table(3, 8, 16 + 8 * 96 + 48, 8);
 for (u32 k = 0; k < 8; k++) memory[(0x800 + 4 * k) / 4] = k;
 handles(3, 8);
 for (u32 k = 0; k < 8; k++) eq(out[k], 0x200 + k, "a nested read: every element (element " + std::to_string(k) + ")");
 // a single handle (no array): untouched
 table(0, 8, 16 + 8 * 96, 8);
 handles(0, 1);
 eq(out[0], 0x200, "one element is not a table");
 if (!failures) std::cout << "handle table count checks passed\n";
 return failures ? 1 : 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--cxx', default=os.environ.get('CXX') or 'c++')
    a = p.parse_args()
    root = a.source_root
    info_h = (root / 'gpu/vk/eden/shader_recompiler/shader_info.h').read_text(encoding='utf-8')
    shader = (root / 'gpu/vk/vk_shader.cpp').read_text(encoding='utf-8')
    # (the count's two functions are cut where the source has them: without them stage_handles is main's, and the check
    # fails on the leftovers, which is the bug)
    optional = [block(shader, sig) for sig in ('uint32_t handle_table_live(', 'bool handle_table_count_on()') if sig in shader]
    extracted = '\n'.join([
        'namespace Shader {',
        block(info_h, 'enum class HandleOp : u32 {') + ';',
        block(info_h, 'template <typename ReadCbuf, typename ReadStorage>\nstd::optional<u32> EvaluateHandleProgram('),
        # (the simulated turf's storage remap, where the source has it: Hookshot's has none, its descriptors' offsets are their own)
        *([info_h[info_h.index('inline constexpr u32 kSimTurfStorageBit'):info_h.index('[[nodiscard]] constexpr u32 SimTurfStorageOrig(')],
           block(info_h, '[[nodiscard]] constexpr u32 SimTurfStorageOrig(')] if 'SimTurfStorageOrig(' in info_h else []),
        '}',
        block(shader, 'uint64_t stage_storage_base('),
        block(shader, 'uint32_t stage_storage_word('),
        *optional,
        # (vk_shader.h's default for the program key, where the source's stage_handles takes one; main's has five arguments)
        *(['void stage_handles(const NvnState& s, const Stage& st, uint32_t program, uint32_t count, uint32_t* out, uint64_t key = 0);']
          if 'uint64_t key)' in block(shader, 'void stage_handles(').split('{', 1)[0] else []),
        block(shader, 'void stage_handles('),
    ])
    with tempfile.TemporaryDirectory(prefix='hookshot-handle-table-count-') as tmp:
        src = Path(tmp) / 'check.cpp'
        exe = Path(tmp) / ('check.exe' if os.name == 'nt' else 'check')
        src.write_text(PRELUDE + extracted + CHECKS, encoding='utf-8')
        subprocess.run([a.cxx, '-std=c++20', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', str(src), '-o', str(exe)],
                       check=True, timeout=180)
        for value, expect in ((None, 'on'), ('1', 'on'), ('0', 'off')):   # (unset is on; =0 is main's way)
            env = dict(os.environ)
            env.pop('RX_VK_HANDLE_TABLE_COUNT', None)
            if value is not None:
                env['RX_VK_HANDLE_TABLE_COUNT'] = value
            r = subprocess.run([str(exe), expect], capture_output=True, text=True, timeout=20, env=env)
            if r.returncode != 0 or r.stdout.strip() != 'handle table count checks passed':
                raise SystemExit('FAIL (RX_VK_HANDLE_TABLE_COUNT %s):\n%s%s' % ('unset' if value is None else '=' + value, r.stdout, r.stderr))
    print('handle table count checks passed (the switch unset, =1 and =0)')


if __name__ == '__main__':
    main()
