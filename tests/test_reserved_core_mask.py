"""The reserved core is one the process may run on, on every path, and the replay thread's log says where it really runs.

hos_reserved_cpus (host/hos_thread.c) keeps a core for the replay. An earlier fix kept its fallback to the cores whose logical
processors are all in the process's affinity mask (a launcher's, `start /affinity`, Process Lasso's, taskset's), but the
Windows choice of the last core of the fastest kind (last_fastest_core, its legacy twin) and Linux's hybrid one returned
before that check: under 0x0FFF of 16 the last core of all, 0xC000, was kept. The replay thread's SetThreadAffinityMask to it
was then refused (Windows refuses a processor outside the process's mask), the thread ran beside the game's, and its log
line said it was pinned.

This check cuts the real code (hos_thread.c from the reserved core's comment to hos_thread_pin: pick_reserved_core,
hos_cache_cpus, both last_fastest_core, the legacy one, hos_reserved_cpus; vk_replay.cpp: the replay thread's placement and
ReplayPin) and runs it over fake topologies and process masks, the Windows branches and the Linux ones, against stubbed
Win32 calls whose SetThreadAffinityMask refuses a processor outside the process's mask as Windows does:
- the reserved core is wholly inside the process's mask (or none), and with no mask it is the core kept before;
- when the pin is refused the log says so and never "runs on"; when it holds, the thread's mask is what the log says.
Three broken variants of the source must fail it: pick_reserved_core without the mask, Windows' last_fastest_core not
passing it, and the replay thread's pin not checked.

python -I -B tests/test_reserved_core_mask.py [--cxx clang++] [--root <checkout>]
(--root: the sources of another checkout, e.g. the commit before the fix, where it must fail)
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import between, block  # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CHECK(ok) do { if (!(ok)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #ok); std::exit(1); } } while (0)

// ---- the Win32 calls the cut code makes, over a fake machine ----
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned char BYTE;
typedef int BOOL;
typedef uint64_t DWORD_PTR;
typedef void* HANDLE;
enum { RelationProcessorCore = 0, RelationCache = 2, ERROR_INSUFFICIENT_BUFFER = 122, ERROR_INVALID_PARAMETER = 87 };
typedef struct { uint64_t Mask; WORD Group; WORD Reserved[3]; } GROUP_AFFINITY;
typedef struct { BYTE Flags, EfficiencyClass; BYTE Reserved[20]; WORD GroupCount; GROUP_AFFINITY GroupMask[1]; } PROCESSOR_RELATIONSHIP;
typedef struct { BYTE Level, Associativity; WORD LineSize; DWORD CacheSize; int Type; BYTE Reserved[20]; GROUP_AFFINITY GroupMask; } CACHE_RELATIONSHIP;
typedef struct { int Relationship; DWORD Size; union { PROCESSOR_RELATIONSHIP Processor; CACHE_RELATIONSHIP Cache; }; }
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
typedef struct { uint64_t ProcessorMask; int Relationship; uint64_t Reserved[2]; } SYSTEM_LOGICAL_PROCESSOR_INFORMATION;

struct Machine {
    int ncores = 0;
    uint64_t core[64] = {};
    unsigned char cls[64] = {};
    int nl3 = 0;
    uint64_t l3[8] = {}, l3size[8] = {};
    bool proc_ok = true;
    uint64_t proc = 0;           // (the process's mask; 0: the call gives 0, as Windows for a process in several groups)
    bool ex_fails = false;       // (GetLogicalProcessorInformationEx fails: the legacy path)
    const char* cpu_core = nullptr;   // (Linux: /sys/devices/cpu_core/cpus; null: no such file)
    const char* reserve = nullptr;    // (HOS_RESERVE_CORE)
    uint64_t thread = 0;         // (the calling thread's affinity, as SetThreadAffinityMask left it)
} M;
static DWORD last_error;
DWORD GetLastError(void) { return last_error; }
HANDLE GetCurrentThread(void) { return (HANDLE)(intptr_t)-2; }
HANDLE GetCurrentProcess(void) { return (HANDLE)(intptr_t)-1; }
static uint64_t all_cpus() { uint64_t a = 0; for (int i = 0; i < M.ncores; i++) a |= M.core[i]; return a; }
BOOL GetProcessAffinityMask(HANDLE, DWORD_PTR* proc, DWORD_PTR* sys) {
    if (!M.proc_ok) { last_error = 5; return 0; }
    *proc = M.proc, *sys = M.proc ? all_cpus() : 0;
    return 1;
}
BOOL GetThreadGroupAffinity(HANDLE, GROUP_AFFINITY* ga) { std::memset(ga, 0, sizeof *ga); ga->Mask = M.thread; return 1; }
BOOL GetLogicalProcessorInformationEx(int rel, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* buf, DWORD* len) {
    if (M.ex_fails) { last_error = 1; return 0; }
    const int n = rel == RelationProcessorCore ? M.ncores : rel == RelationCache ? M.nl3 : 0;
    const DWORD need = (DWORD)(n * sizeof *buf);
    if (!buf || *len < need) { *len = need; last_error = ERROR_INSUFFICIENT_BUFFER; return 0; }
    std::memset(buf, 0, need);
    for (int i = 0; i < n; i++) {
        buf[i].Relationship = rel;
        buf[i].Size = sizeof *buf;
        if (rel == RelationProcessorCore) {
            buf[i].Processor.EfficiencyClass = M.cls[i];
            buf[i].Processor.GroupCount = 1;
            buf[i].Processor.GroupMask[0].Mask = M.core[i];
        } else {
            buf[i].Cache.Level = 3;
            buf[i].Cache.CacheSize = (DWORD)M.l3size[i];
            buf[i].Cache.GroupMask.Mask = M.l3[i];
        }
    }
    *len = need;
    return 1;
}
BOOL GetLogicalProcessorInformation(SYSTEM_LOGICAL_PROCESSOR_INFORMATION* info, DWORD* len) {
    const DWORD need = (DWORD)(M.ncores * sizeof *info);
    if (*len < need) { *len = need; last_error = ERROR_INSUFFICIENT_BUFFER; return 0; }
    std::memset(info, 0, need);
    for (int i = 0; i < M.ncores; i++) info[i].ProcessorMask = M.core[i], info[i].Relationship = RelationProcessorCore;
    *len = need;
    return 1;
}
// (Windows: a thread's mask must lie within the process's; refused, the thread keeps its mask and the call returns 0)
DWORD_PTR SetThreadAffinityMask(HANDLE, DWORD_PTR mask) {
    const uint64_t proc = M.proc ? M.proc : all_cpus();
    if (!mask || (mask & ~proc)) { last_error = ERROR_INVALID_PARAMETER; return 0; }
    const uint64_t prev = M.thread;
    M.thread = mask;
    return prev;
}
DWORD SetThreadIdealProcessor(HANDLE, DWORD) { return 0; }
static const char* fake_getenv(const char* name) {
    if (!std::strcmp(name, "HOS_RESERVE_CORE")) return M.reserve;
    return nullptr;
}
static int env_on(const char* n) { const char* e = fake_getenv(n); return e && std::atoi(e) != 0; }
static int env_on_or(const char* n, int d) { const char* e = fake_getenv(n); return e ? std::atoi(e) != 0 : d; }
#define getenv(n) fake_getenv(n)
static char fake_file;
#define fopen(p, m) (std::strcmp((p), "/sys/devices/cpu_core/cpus") || !M.cpu_core ? (FILE*)nullptr : (FILE*)&fake_file)
#define fgets(s, n, f) (std::snprintf((s), (n), "%s", M.cpu_core), (s))
#define fclose(f) ((void)(f))
static std::vector<std::string> log_lines;
static void hos_log(const char* fmt, ...) {
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    log_lines.push_back(b);
}

// ---- host/hos_thread.c, the Windows branches ----
#undef _WIN32
#define _WIN32 1
namespace win {
@HOS@
}
// ---- the same, the Linux branches ----
#undef _WIN32
namespace lin {
@HOS@
}
#define _WIN32 1

// ---- gpu/vk/vk_replay.cpp: the replay thread's placement, ReplayPin ----
namespace vk {
static std::vector<std::string> lines;
static void logf(const char* fmt, ...) {
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    lines.push_back(b);
}
struct Host { uint64_t (*reserved_cpus)(void); };
static uint64_t reserved_now;
static uint64_t reserved_fake() { return reserved_now; }
static Host host_v = {reserved_fake};
static Host* host = &host_v;
static uint64_t fastest;
static uint64_t fastest_kind_cpus() { return fastest; }
static bool async_on() { return false; }
static uint64_t replay_pin_mask() { return reserved_now; }
thread_local int replay_pin_depth = 0;
thread_local DWORD_PTR replay_pin_prev = 0;
@PIN@;
static void place() {
@PLACE@
}
static bool said(const char* part) {
    for (auto& l : lines) if (l.find(part) != std::string::npos) return true;
    return false;
}
static int count(const char* part) {
    int n = 0;
    for (auto& l : lines) n += l.find(part) != std::string::npos;
    return n;
}
}

// 8 cores with SMT, one kind (a Ryzen 7 5700X3D: core k is logical processors 2k, 2k+1)
static void eight_smt() {
    M = Machine{};
    for (int k = 0; k < 8; k++) M.core[k] = 3ull << (2 * k);
    M.ncores = 8;
}
// an i7-12700K: 8 performance cores with SMT (0-15, class 1), 4 efficiency cores (16-19, class 0)
static void i7_12700k() {
    M = Machine{};
    for (int k = 0; k < 8; k++) M.core[k] = 3ull << (2 * k), M.cls[k] = 1;
    for (int k = 0; k < 4; k++) M.core[8 + k] = 1ull << (16 + k), M.cls[8 + k] = 0;
    M.ncores = 12;
}
// a 7950X3D with SMT: the 96 MB L3 on cores 0-7 (0-15), 32 MB on 8-15 (16-31)
static void r9_7950x3d() {
    M = Machine{};
    for (int k = 0; k < 16; k++) M.core[k] = 3ull << (2 * k);
    M.ncores = 16;
    M.nl3 = 2, M.l3[0] = 0xFFFF, M.l3size[0] = 96u << 20, M.l3[1] = 0xFFFF0000, M.l3size[1] = 32u << 20;
}
// a Core Ultra 7 265KF on Linux: 20 cores without SMT, the performance ones 0-7 in cpu_core's list
static void ultra_265kf() {
    M = Machine{};
    for (int k = 0; k < 20; k++) M.core[k] = 1ull << k;
    M.ncores = 20;
    M.cpu_core = "0-7\n";
}
static uint64_t won(uint64_t proc) { M.proc = proc; M.thread = proc ? proc : all_cpus(); return win::hos_reserved_cpus(); }
static uint64_t lon(uint64_t proc) { M.proc = proc; M.thread = proc ? proc : all_cpus(); return lin::hos_reserved_cpus(); }
static int cases;
#define RESERVED(call, proc, want) do { \
    const uint64_t got_ = (call)(proc); cases++; \
    if (got_ != (uint64_t)(want)) std::fprintf(stderr, "%s: process mask %#llx: reserved %#llx, want %#llx\n", #call, \
                                               (unsigned long long)(proc), (unsigned long long)got_, (unsigned long long)(want)); \
    CHECK(got_ == (uint64_t)(want)); \
    CHECK(!(proc) || !(got_ & ~(uint64_t)(proc))); } while (0)

int main() {
    // Windows, the enhanced path (GetLogicalProcessorInformationEx)
    eight_smt();
    RESERVED(won, 0xFFFF, 0xC000);   // (no mask: the last core, as before)
    RESERVED(won, 0, 0xC000);        // (the mask not known: as with none)
    M.proc_ok = false; RESERVED(won, 0xFFFF, 0xC000); M.proc_ok = true;
    RESERVED(won, 0x0FFF, 0x0C00);   // (6 cores: the last of them, not 0xC000 outside the mask)
    RESERVED(won, 0x00FF, 0x00C0);   // (4 cores, 8 logical processors)
    RESERVED(won, 0x003F, 0);        // (3 cores, 6 logical processors: none, as the fallback)
    RESERVED(won, 0x5555, 0);        // (one logical processor of each core: no core wholly the process's)
    RESERVED(won, 0xFF00, 0xC000);   // (the last 4 cores)
    i7_12700k();
    RESERVED(won, 0xFFFFF, 0xC000);  // (the last performance core, as before)
    RESERVED(won, 0xF0FFF, 0x0C00);  // (6 performance cores and the efficiency ones)
    RESERVED(won, 0xF00FF, 0x00C0);  // (4 performance cores: still 4 of the fastest kind)
    RESERVED(won, 0xF003F, 0);       // (3 performance cores: too few of the fastest kind, as on an i5-1235U)
    RESERVED(won, 0xF0000, 0);       // (the efficiency cores alone: 4 logical processors)
    r9_7950x3d();
    RESERVED(won, 0xFFFFFFFF, 0);    // (the cache cores split: none kept, as before)
    M.reserve = "1";
    RESERVED(won, 0xFFFFFFFF, 0xC0000000);   // (HOS_RESERVE_CORE=1: the last core, as before)
    RESERVED(won, 0x0000FFFF, 0x0000C000);   // (kept to the cache CCD, as 7950X3D owners do: a core of that CCD)
    // Windows, the legacy path (GetLogicalProcessorInformation; hybrid by SMT)
    eight_smt(); M.ex_fails = true;
    RESERVED(won, 0xFFFF, 0xC000);
    RESERVED(won, 0x0FFF, 0x0C00);
    RESERVED(won, 0x5555, 0);
    i7_12700k(); M.ex_fails = true;
    RESERVED(won, 0xFFFFF, 0xC000);
    RESERVED(won, 0xF0FFF, 0x0C00);
    // Linux: the hybrid path (/sys/devices/cpu_core/cpus), and the fallback without that file
    ultra_265kf();
    RESERVED(lon, 0xFFFFF, 0x80);    // (the last performance core, as before)
    RESERVED(lon, 0xFFF3F, 0x20);    // (taskset to 6 performance cores and the efficiency ones)
    RESERVED(lon, 0xFFF00, 0x80000); // (the efficiency cores alone: the fastest kind the process has)
    eight_smt();
    RESERVED(lon, 0xFFFF, 0xC000);
    RESERVED(lon, 0x0FFF, 0x0C00);   // (the earlier fallback, as before)

    // (checker) no mask, a mask that cannot be read (0: Windows, a process with threads in several groups; Linux, a process
    // on CPUs >= 64 only) and a failing call: exactly the core the base kept, on every path
    i7_12700k();
    RESERVED(won, 0, 0xC000);
    M.proc_ok = false; RESERVED(won, 0xFFFFF, 0xC000); M.proc_ok = true;
    i7_12700k(); M.ex_fails = true;
    RESERVED(won, 0, 0xC000);
    M.proc_ok = false; RESERVED(won, 0xFFFFF, 0xC000); M.proc_ok = true;
    RESERVED(won, 0x0FFFF, 0xC000);  // (the performance cores alone: one kind left, the legacy last-core rule)
    eight_smt(); M.ex_fails = true;
    RESERVED(won, 0, 0xC000);
    M.proc_ok = false; RESERVED(won, 0xFFFF, 0xC000); M.proc_ok = true;
    r9_7950x3d(); M.reserve = "1";
    RESERVED(won, 0, 0xC0000000);
    M.proc_ok = false; RESERVED(won, 0xFFFFFFFF, 0xC0000000); M.proc_ok = true;
    M.reserve = nullptr;
    RESERVED(won, 0, 0);             // (the cache split: none, as before)
    ultra_265kf();
    RESERVED(lon, 0, 0x80);
    eight_smt();
    RESERVED(lon, 0, 0xC000);

    // the replay thread's placement: the log says where it really runs
    {   // a reserved core outside the process's mask (HOS_RESERVE_CORE_MASK, or the old choice): refused, and said so
        eight_smt(); M.proc = 0x0FFF; M.thread = 0x0FFF;
        vk::lines.clear(); vk::reserved_now = 0xC000; vk::fastest = 0xFFFF;
        vk::place();
        CHECK(M.thread == 0x0FFF);
        CHECK(!vk::said("runs on logical processors"));
        CHECK(!vk::said("may run on logical processors"));
        CHECK(vk::said("NOT kept"));
    }
    {   // inside it: pinned, and the log's masks are the thread's
        eight_smt(); M.proc = 0xFFFF; M.thread = 0xFFFF;
        vk::lines.clear(); vk::reserved_now = 0xC000; vk::fastest = 0xFFFF;
        vk::place();
        CHECK(M.thread == 0xFFFF);
        CHECK(vk::said("runs on logical processors 0xc000"));
        CHECK(vk::said("may run on logical processors 0xffff"));
        CHECK(!vk::said("NOT kept"));
    }
    {   // end to end: hos_reserved_cpus under 0x0FFF, then the placement: the thread on what the log says
        eight_smt();
        vk::reserved_now = won(0x0FFF);
        vk::lines.clear(); vk::fastest = 0xFFFF;
        vk::place();
        CHECK(vk::reserved_now == 0x0C00);
        CHECK(M.thread == 0x0FFF);
        CHECK(vk::said("runs on logical processors 0xc00,"));
        CHECK(vk::said("may run on logical processors 0xfff"));
    }
    {   // ReplayPin (the synchronous replay's pin): refused, said once and nothing to undo; held, undone after
        eight_smt(); M.proc = 0x0FFF; M.thread = 0x0FFF;
        vk::lines.clear(); vk::reserved_now = 0xC000;
        { vk::ReplayPin a; CHECK(M.thread == 0x0FFF); }
        { vk::ReplayPin b; CHECK(M.thread == 0x0FFF); }
        CHECK(M.thread == 0x0FFF);
        CHECK(vk::count("NOT kept") == 1);
        vk::lines.clear(); vk::reserved_now = 0x0C00;
        { vk::ReplayPin c; CHECK(M.thread == 0x0C00); }
        CHECK(M.thread == 0x0FFF);
        CHECK(!vk::said("NOT kept"));
    }
    std::printf("PASS: %d reserved cores, each one the process may run on wholly (or none), the same as before without a "
                "mask; the replay thread's log says where it really runs\n", cases);
}
'''


def sources(root):
    hos = (root / 'host/hos_thread.c').read_text(encoding='utf-8')
    replay = (root / 'gpu/vk/vk_replay.cpp').read_text(encoding='utf-8')
    return hos, replay


def fixture(hos, replay):
    region = between(hos, "/* The last physical core (its logical processors' mask), kept for the Vulkan back end",
                     'void hos_thread_pin(void *h, const char *name) {')
    # (each case asks again: the two choices' once-only caches off)
    cached = 'static uint64_t mask = ~0ull;\n    if (mask != ~0ull) return mask;'
    if region.count(cached) != 2:
        raise SystemExit('test_reserved_core_mask: hos_cache_cpus and hos_reserved_cpus no longer cache their choice as %r: '
                         'update the check with the source' % cached)
    region = region.replace(cached, 'uint64_t mask = ~0ull;')
    place = between(replay, "    // (on the core the host keeps free of the game's threads, hos_reserved_cpus: its last physical one)",
                    '    // (hookshot, the render thread) A present that comes')
    pin = block(replay, 'struct ReplayPin {')
    return FIXTURE.replace('@HOS@', region).replace('@PLACE@', place).replace('@PIN@', pin)


def run(cxx, text, tmp, name):
    """0 when the fixture built from text passes; the failure's text otherwise"""
    cpp, exe = Path(tmp) / (name + '.cpp'), Path(tmp) / (name + '.exe')
    cpp.write_text(text, encoding='utf-8')
    built = subprocess.run([cxx, '-std=c++20', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-Wno-unused-variable',
                            '-Wno-unused-parameter', str(cpp), '-o', str(exe)], capture_output=True, text=True)
    if built.returncode:
        return 'build failed:\n' + built.stderr[-3000:]
    ran = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    return 0 if ran.returncode == 0 else (ran.stderr + ran.stdout)[-2000:] or 'exit %d' % ran.returncode


def mutate(text, old, new, what, count=1):
    if old not in text:
        raise SystemExit('test_reserved_core_mask: the broken variant "%s" found no %r: update the check with the source'
                         % (what, old))
    return text.replace(old, new, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    hos, replay = sources(args.root)
    with tempfile.TemporaryDirectory(prefix='reserved-core-mask-') as tmp:
        failed = run(args.cxx, fixture(hos, replay), tmp, 'fixture')
        if failed:
            print(failed, file=sys.stderr)
            print('FAIL: test_reserved_core_mask', file=sys.stderr)
            return 1
        broken = {
            "pick_reserved_core without the process's mask":
                fixture(mutate(hos, 'if (!masks[i] || (masks[i] & ~allowed)) continue;', 'if (!masks[i]) continue;',
                               'no mask in pick_reserved_core'), replay),
            "Windows' last_fastest_core not passing the mask (the first of the two calls is Windows')":
                fixture(mutate(hos, 'return pick_reserved_core(masks, classes, n, process_cpus());',
                               'return pick_reserved_core(masks, classes, n, ~0ull);', 'mask not passed'), replay),
            "Linux's last_fastest_core not passing the mask (the second of the two calls)":
                fixture(mutate(hos, 'return pick_reserved_core(masks, classes, n, process_cpus());\n}\n#endif\nuint64_t',
                               'return pick_reserved_core(masks, classes, n, ~0ull);\n}\n#endif\nuint64_t',
                               'Linux mask not passed'), replay),
            "the legacy path without the process's mask":
                fixture(mutate(hos, '            if (mask & ~allowed) continue;\n', '', 'no mask in the legacy path'), replay),
            "an unreadable process mask taken as no processor (the no-mask choice must be the base's)":
                fixture(mutate(hos, '|| !proc) return ~0ull;', '|| !proc) return 0;', 'unreadable mask as none'), replay),
            "the replay thread's pin not checked":
                fixture(hos, mutate(replay, 'if (!SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)use)) {',
                                    'if ((SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)use), false)) {',
                                    'pin not checked')),
        }
        for i, (what, text) in enumerate(broken.items()):
            result = run(args.cxx, text, tmp, 'broken%d' % i)
            if not result:
                print('FAIL: test_reserved_core_mask: the broken variant "%s" passed' % what, file=sys.stderr)
                return 1
            if result.startswith('build failed'):
                print(result, file=sys.stderr)
                print('FAIL: test_reserved_core_mask: the broken variant "%s" did not build' % what, file=sys.stderr)
                return 1
            print('ok: the broken variant "%s" fails: %s' % (what, result.strip().splitlines()[-1]))
    print('test_reserved_core_mask: ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
