"""Exercise the renderer's CPU decoder selector, safe refusal and legacy fallback.
python -I -B tests/test_astc_fast_cpu_call.py --cxx c++
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cxx', default='c++')
    args = p.parse_args()
    text = (ROOT / 'gpu/vk/vk_texture.cpp').read_text(encoding='utf8')
    cut = text[text.index('using AstcCpuDecoder ='):text.index('void decode_astc_vega_cached(')]
    source = r'''
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>
#include "texdecode/texdecode.h"
static int serial=0, parallel=0;
namespace Tegra::Texture::ASTC {
void DecompressSerial(std::span<const uint8_t>,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,std::span<uint8_t>){++serial;}
void Decompress(std::span<const uint8_t>,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,std::span<uint8_t>){++parallel;}
}
bool env_on(const char* name){const char* p=std::getenv(name);return p&&std::strcmp(p,"1")==0;}
void logf(const char*,...){}
using namespace mkwvk;
''' + cut + r'''
int main(int argc,char**) {
 bool fast=argc==2;
 std::vector<uint8_t> blocks(16),out(64,0xA5);
 astc_cpu_decoder()(blocks,4,4,1,4,4,out);
 astc_cpu_decoder(true)(blocks,4,4,1,4,4,out);
 assert(serial==!fast && parallel==!fast);
 if(!fast)return 0;
 assert(std::all_of(out.begin(),out.end(),[](auto c){return c==0;}));
 // A truncated source must not leave uninitialised staging bytes for the GPU.
 blocks.resize(15);std::fill(out.begin(),out.end(),0xA5);
 astc_cpu_decoder()(blocks,4,4,1,4,4,out);
 assert(std::all_of(out.begin(),out.end(),[](auto c){return c==0;}));
 // Keep the legacy linear profile, including for data later viewed as sRGB.
 blocks={0xFC,0xFD,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0x22,0x22,0x44,0x44,0x66,0x66,0x88,0x88};
 astc_cpu_decoder()(blocks,4,4,1,4,4,out);
 for(unsigned i=0;i<64;++i)assert(out[i]==uint8_t(0x22*(i%4+1)));
 // Synthetic astcenc vector 49 distinguishes the two profiles. This adapter
 // deliberately preserves the existing linear-profile/cache identity contract.
 blocks={0x1d,0xef,0x4f,0xea,0x11,0xbb,0xc7,0x30,0x8e,0x29,0x77,0xbb,0x60,0x99,0x26,0x97};
 astc_cpu_decoder()(blocks,4,4,1,4,4,out);
 uint64_t hash=0xCBF29CE484222325ull;
 for(auto c:out)hash=(hash^c)*0x100000001B3ull;
 assert(hash==0xad55491b6761db12ull);
}
'''
    with tempfile.TemporaryDirectory(prefix='redtunic-astc-call-') as folder:
        src=Path(folder)/'call.cpp';exe=Path(folder)/('call.exe' if os.name=='nt' else 'call')
        src.write_text(source,encoding='utf8')
        subprocess.run([args.cxx,'-std=c++20','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'gpu/vk'),str(src),str(ROOT/'gpu/vk/texdecode/astc.cpp'),'-o',str(exe)],check=True)
        for fast in (False, True):
            env=dict(os.environ);env['RX_VK_ASTC_FAST_CPU']='1' if fast else '0'
            subprocess.run([str(exe)]+(['fast'] if fast else []),env=env,check=True)
    print('CPU ASTC renderer call: legacy serial/parallel fallback, reference decoder, transparent error and safe refusal passed')


if __name__=='__main__':
    main()
