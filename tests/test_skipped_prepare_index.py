"""Skipped startup compilation still initializes cached load-at-create metadata.

Compile the real index-only function with call spies; verify both startup skip
paths reach it. No GPU, dump, or shader compiler is needed.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[1]
RUN_ALL_ARGS=['--cxx','{cxx}']
def block(src, marker):
    start=src.index(marker); end=src.index('{',start)+1; depth=1
    while depth:
        depth+=(src[end]=='{')-(src[end]=='}'); end+=1
    return src[start:end]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cxx',default='g++'); a=ap.parse_args()
    draw=(ROOT/'gpu/vk/vk_draw.cpp').read_text()
    source=(ROOT/'gpu/vk/vk_main.cpp').read_text()
    startup=block(source,'bool started_up()')
    skip=block(startup,'if (FILE* f = prepare_all ? nullptr : fopen(skipped_path().c_str(), "rb"))')
    assert 'precompile_index_only();' in skip and skip.index('precompile_index_only();')<skip.index('return true;')
    prepare=block(startup,'if (precompile_wanted())')
    tail=startup[startup.index(prepare)+len(prepare):]
    assert tail.lstrip().startswith('else {\n        precompile_index_only();\n    }')
    assert 'precompile_index_only' not in block(startup,'if (prepare_only())')
    cpp=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <unordered_set>
#include <vector>
struct NoteRef { int key; };
struct ComputePlan { bool none=false; std::unordered_set<uint64_t> keys={11,22}; };
bool create_mode=true, omit_computes=false, omit_states=false;
std::vector<int> calls;
std::unordered_set<uint64_t> received;
bool programs_load_at_create() { return create_mode; }
void pcache_init() { calls.push_back(1); }
bool env_on(const char*) { return omit_states; }
const char* precompile_computes_which() { return nullptr; }   // (RX_VK_PRECOMPILE_COMPUTES, vk_draw.cpp)
ComputePlan computes_plan(const char*,bool states) {
 assert(states==!omit_states); calls.push_back(2); ComputePlan p; p.none=omit_computes; return p;
}
std::vector<NoteRef> pipe_notes_refs() { calls.push_back(3); return {{42}}; }
void programs_index(std::vector<NoteRef> notes,const std::unordered_set<uint64_t>& keys) {
 assert(calls==std::vector<int>({1,2,3})); assert(notes.size()==1 && notes[0].key==42);
 calls.push_back(4); received=keys;
}
void logf(const char*,...) { calls.push_back(5); }
''' + block(draw,'void precompile_index_only()') + r'''
int main() {
 precompile_index_only(); assert(calls==std::vector<int>({1,2,3,4,5}));
 assert(received==std::unordered_set<uint64_t>({11,22}));
 calls.clear(); create_mode=false; precompile_index_only(); assert(calls.empty());
 create_mode=true; omit_computes=true; omit_states=true; precompile_index_only();
 assert(calls==std::vector<int>({1,2,3,4,5}) && received.empty());
 puts("PASS: skipped compilation indexes cached resources, only in create mode");
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp); (p/'test.cpp').write_text(cpp)
        subprocess.run([a.cxx,'-std=c++20','-Wall','-Wextra','-Werror',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
        subprocess.run([str(p/'test')],check=True)
if __name__=='__main__': main()
