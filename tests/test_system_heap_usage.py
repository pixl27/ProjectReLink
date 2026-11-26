"""Exercise the production system heap budget query on discrete and unified-memory GPUs.

No Vulkan driver required. Run python -I -B tests/test_system_heap_usage.py.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from srccut import block

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    source = (ROOT / "gpu/vk/vk_vma.cpp").read_text(encoding="utf-8")
    code = r'''
#include <cstdint>
#include <cassert>
#include <cstdio>
constexpr unsigned VK_MAX_MEMORY_HEAPS=4, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT=1;
[[maybe_unused]] constexpr int VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU=1;
struct VmaBudget { uint64_t usage=0,budget=0; } budgets[4];
struct Heap { unsigned flags=0; };
struct { struct { unsigned memoryHeapCount=0; Heap memoryHeaps[4]; } memory;
    struct { int deviceType=0; } props; bool memory_budget=true; } dev;
unsigned budget_frame=0; int vma=1;
void vmaSetCurrentFrameIndex(int,unsigned) {}
void vmaGetHeapBudgets(int,VmaBudget* b) { for (int i=0;i<4;i++) b[i]=budgets[i]; }
@FUNCTION@
int main() {
    uint64_t used=999,budget=999;
    dev.memory.memoryHeapCount=2;
    dev.memory.memoryHeaps[0].flags=1; budgets[0]={100,200}; budgets[1]={50,150};
    assert(system_heap_usage(&used,&budget)); assert(used==50 && budget==150);
    dev.memory.memoryHeapCount=1; dev.props.deviceType=1;
    assert(system_heap_usage(&used,&budget)); assert(used==100 && budget==200);
    dev.props.deviceType=0;
    assert(!system_heap_usage(&used,&budget)); assert(used==0 && budget==0);
    dev.props.deviceType=1; dev.memory_budget=false;
    assert(!system_heap_usage(&used,&budget)); assert(used==100 && budget==200);
    dev.memory_budget=true; budgets[0].budget=0; assert(!system_heap_usage(&used,&budget));
    puts("PASS: UMA uses its shared heap; discrete GPUs retain separate system-heap accounting");
}
'''.replace("@FUNCTION@", block(source, "bool system_heap_usage("))
    with tempfile.TemporaryDirectory(prefix="system-heap-") as tmp:
        cpp, exe = Path(tmp) / "test.cpp", Path(tmp) / "test.exe"
        cpp.write_text(code, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)

if __name__ == "__main__":
    main()
