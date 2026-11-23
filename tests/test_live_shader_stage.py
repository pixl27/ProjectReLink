"""Exercise the renderer's first-use stage reads with synthetic guest pages and a counting lowerer."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from srccut import between, block

FIXTURE = r'''
#include <array>
#include <atomic>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

constexpr uint32_t kControlSize = 216;
struct Code { bool ok = false; uint64_t key = 0; uint8_t source_byte = 0; };
struct nvn_stage_data { uint64_t control, data; };
struct GuestProgram { uint64_t serial; uint32_t nstage; nvn_stage_data sd[6]; };
struct NvnState { GuestProgram prog; };
struct Program { bool failed = false; std::atomic<bool> ready{false}; };
struct { bool native_mma = false; } dev;
struct Guest {
    std::vector<uint8_t> ctl = std::vector<uint8_t>(kControlSize);
    std::vector<uint8_t> data = std::vector<uint8_t>(128);
    bool readable = true;
    bool gpu_read(uint64_t p, void* out, size_t n) const {
        const auto& bytes = (p & 1) ? ctl : data;
        if (!readable || !p || n > bytes.size()) return false;
        std::memcpy(out, bytes.data(), n);
        return true;
    }
} guest;
auto* host = &guest;
bool control_end(const uint8_t* ctl, uint32_t* end) {
    *end = uint32_t(guest.data.size());
    return ctl[0] != 0 && *end != 0;
}
// Force different packages into the same key and lookup slot.
uint64_t code_key(const uint8_t*, const uint8_t*, uint32_t) { return 17; }
static unsigned calls;
Code fake_parse(const uint8_t* ctl, const std::vector<uint8_t>& area) {
    ++calls;
    return {ctl[1] != 255 && area.back() != 255, 17, area.back()};
}
@READERS@
@LOWERER@
@CODES@
@CODES_OF@
std::function<void()> pending;
void jobs_push(std::function<void()> work, bool, std::atomic<bool>*, std::function<void()>) {
    pending = std::move(work);
}
int translated = -1;
void translate_into(Program&, const NvnState&, const Code* vertex, const Code*, const Code*, const uint8_t*) {
    translated = vertex->source_byte;
}
void defer_graphics(const NvnState& s, Program* raw) {
    uint8_t color_types[8]{};
    @STAGE_ARRAY@
    stage_code[0] = @VERTEX@;
    @QUEUE_JOB@
}
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr,"line %d: %s (lowerings %u)\n",__LINE__,#c,calls); return 1; } } while(0)
int main(int argc, char**) {
    guest.ctl[0]=1;
    auto draw=[](uint64_t serial) {
        NvnState s{}; s.prog={serial,1,{{serial*2+1,serial*2+2}}};
        return codes_of(s).size();
    };
    if (argc==2) {
        CHECK(draw(1)==1 && draw(2)==1 && calls==2);
        return 0;
    }
    CHECK(draw(1)==1 && calls==1);
    CHECK(draw(2)==1 && calls==1); // new serial and different GPU addresses, identical bytes
    codes.erase(1);
    CHECK(draw(3)==1 && calls==1); // second serial still owns the lowered stage
    guest.data.back()=1;
    CHECK(draw(4)==1 && calls==2);
    guest.ctl.back()=2;
    CHECK(draw(5)==1 && calls==3);
    dev.native_mma=true;
    CHECK(draw(6)==1 && calls==4);
    guest.readable=false;
    CHECK(draw(7)==0 && calls==4);
    guest.readable=true;
    CHECK(draw(7)==1 && calls==4);
    guest.data.back()=255;
    CHECK(draw(8)==0 && draw(8)==0 && calls==6);
    guest.data.back()=7;
    CHECK(draw(8)==1 && calls==7);
    codes.clear();
    CHECK(draw(9)==1 && calls==8); // all owners released
    guest.data.resize(128*1024);
    CHECK(draw(10)==1 && draw(11)==1 && calls==10);
    guest.data.resize(128);
    guest.data.back()=42;
    NvnState first{}; first.prog={12,1,{{25,26}}};
    Program result;
    defer_graphics(first, &result);
    codes.erase(12);
    guest.data.back()=99;
    CHECK(draw(13)==1 && translated==-1); // replace the colliding slot before translation
    pending();
    CHECK(translated==42);
    pending = {};
}
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='c++')
    args = parser.parse_args()
    shader = (ROOT / 'gpu/vk/vk_shader.cpp').read_text()
    readers = between(shader, '\nCode parse_code(', '/// A control section and its code', unique=False)
    signature = next(line for line in shader.splitlines() if ' codes_of(const NvnState& s) {' in line)
    registry = next(line for line in shader.splitlines() if ' codes;' in line and 'by program serial' in line)
    lowerer = ('Code parse_code(const uint8_t* ctl, const std::vector<uint8_t>& area, uint64_t) '
               if 'uint64_t key);' in readers else 'Code parse_code(const uint8_t* ctl, const std::vector<uint8_t>& area) ')
    pick = block(shader, 'const Program* program_pick(const NvnState& s, ProgramKind kind, const uint8_t color_types[8]) {')
    array = next(line.strip() for line in pick.splitlines() if 'stage_code[8]' in line)
    job = between(pick, '    struct Job {', '    return raw;', unique=False)
    vertex = 'codes_of(s).front()' if 'SharedCode' in array else '&codes_of(s).front()'
    text = (FIXTURE.replace('@STAGE_ARRAY@',array).replace('@QUEUE_JOB@',job).replace('@VERTEX@',vertex)
            .replace('@READERS@', readers).replace('@LOWERER@', lowerer + '{return fake_parse(ctl,area);}')
            .replace('@CODES@', registry).replace('@CODES_OF@', block(shader, signature)))
    with tempfile.TemporaryDirectory() as tmp:
        cpp, exe = Path(tmp)/'test.cpp', Path(tmp)/'test.exe'
        cpp.write_text(text)
        subprocess.run([args.cxx, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', cpp, '-o', exe], check=True)
        env = dict(os.environ)
        env.pop('RX_VK_DUMP_NATIVE', None)
        subprocess.run([exe], env=env, check=True)
        subprocess.run([exe, 'native-dump'], env=dict(env, RX_VK_DUMP_NATIVE='11'), check=True)
        return 0

if __name__ == '__main__':
    raise SystemExit(main())
