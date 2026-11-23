/* Guest memory map: the system region (our own objects), the heap, address regions.
   The whole 4 GB window is committed up front (hos_main.c), so "mapping" here is only bookkeeping and zeroing. */
#include "hos.h"
#include "hos_guestpages.h"
#ifndef HOOKSHOT_ENV_H   /* (env_on: the switches' one rule) */
#include "../gpu/vk/compat/hookshot_env.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef TRACY_ENABLE
#define HT_WRAP_LOCKS   /* (HOOKSHOT_TRACY: this file's locks that have to wait are zones) */
#include "hos_tracy.h"
#endif
static SRWLOCK lock = SRWLOCK_INIT;
#define LOCK() AcquireSRWLockExclusive(&lock)
#define UNLOCK() ReleaseSRWLockExclusive(&lock)
#else
#include <pthread.h>
#include <sys/mman.h>
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
#define LOCK() pthread_mutex_lock(&lock)
#define UNLOCK() pthread_mutex_unlock(&lock)
#endif

uint64_t hos_heap_size, hos_heap_used, hos_total_memory, hos_program_size;
static uint64_t sys_next = HOS_SYSTEM_BASE;

/* Address regions are handed out top-down from HOS_SPACE_END; the heap grows up from HOS_HEAP_BASE. A region being freed
   stays listed, and so taken, until its pages are discarded (freeing: hos_region_free). */
typedef struct { uint64_t addr, size; int freeing; } Range;
static Range *regions;
static int nregions, capregions;
static uint64_t region_floor = ~0ULL;   /* the lowest region (none yet: the window's end) */

/* Free address space below the heap that regions may also use (web: the window is tight): above the null guard
   to the system region, and from the image's end to the heap. */
static uint64_t low_hole[2][2];

void hos_mem_init(uint64_t image_end) {
    hos_program_size = image_end - hos_image_base;
    hos_commit(HOS_SYSTEM_BASE, HOS_SYSTEM_END - HOS_SYSTEM_BASE);
    low_hole[0][0] = 0x00100000ULL, low_hole[0][1] = HOS_SYSTEM_BASE;
    low_hole[1][0] = (image_end + 0xFFFFFULL) & ~0xFFFFFULL, low_hole[1][1] = HOS_HEAP_BASE;
    /* Retail Switch gives an application 3285 MB. HOS_MEMORY_MB overrides it: TOTK sizes its heap from this
       (heap = total - used - 1080 MB), which is the lever for fitting the web build into wasm32. */
    const char *mb = getenv("HOS_MEMORY_MB");
#ifdef A64_MEM_ZERO
    /* (web: TOTK keeps 1080 MB of the total for address regions (its resource memory grows into them) and its heap
       must stay 2104 MB (a 1902 MB heap: an indirect call into nothing 4 s in), so the window must hold heap +
       regions; the holes below the heap help) */
    hos_total_memory = (mb ? strtoull(mb, NULL, 0) : 3285ULL) << 20;
    hos_log("memory: %llu MB for the game (window %llu MB)", (unsigned long long)(hos_total_memory >> 20),
            (unsigned long long)(HOS_SPACE_END >> 20));
#else
    /* mkw-recomp: a Switch 2 application gets 9809 MB (Eglantine's KSystemControl, 12 GB arrangement) */
    hos_total_memory = (mb ? strtoull(mb, NULL, 0) : 9809ULL) << 20;
#endif
}

uint64_t hos_region_floor(void) { return region_floor; }

/* (hookshot, the control bridge: host/control.c) Whether [addr, addr + size) is wholly inside memory the runtime has
   mapped for the game: the image, the system region in use (our own objects), the heap, or an address region. A read or
   write outside it is refused, never faulted. A region being freed (hos_region_free) counts as mapped until its pages are
   handed back: a bridge read there meanwhile may return zeros, or old and new bytes mixed (harmless: the guest has freed
   it). The classes it may fall in (0 none): */
enum { HOS_MEM_IMAGE = 1, HOS_MEM_SYSTEM, HOS_MEM_HEAP, HOS_MEM_REGION };
int hos_mem_region(uint64_t addr, uint64_t size) {
    uint64_t end;
    if (!size || __builtin_add_overflow(addr, size, &end) || !hos_guest_range(addr, size)) return 0;   /* (the window first) */
    if (addr >= hos_image_base && end <= hos_image_end) return HOS_MEM_IMAGE;
    int cls = 0;
    LOCK();
    if (addr >= HOS_SYSTEM_BASE && end <= sys_next) cls = HOS_MEM_SYSTEM;
    else if (addr >= HOS_HEAP_BASE && end <= HOS_HEAP_BASE + hos_heap_size) cls = HOS_MEM_HEAP;
    else
        for (int i = 0; i < nregions && !cls; i++)
            if (addr >= regions[i].addr && end <= regions[i].addr + regions[i].size) cls = HOS_MEM_REGION;
    UNLOCK();
    return cls;
}

/* (Hookshot, hl_mem.c's heap pass) the address regions at or above `from`, [addr, end) pairs in ascending order, at most
   max of them (the lowest); returns how many */
int hos_regions_above(uint64_t from, uint64_t (*out)[2], int max) {
    int n = 0;
    LOCK();
    for (int i = nregions - 1; i >= 0 && n < max; i--)   /* (the list is kept descending) */
        if (regions[i].addr >= from) out[n][0] = regions[i].addr, out[n][1] = regions[i].addr + regions[i].size, n++;
    UNLOCK();
    return n;
}

uint64_t hos_sys_alloc(uint64_t size, uint64_t align) {
    LOCK();
    uint64_t a = (sys_next + align - 1) & ~(align - 1);
    if (a + size > HOS_SYSTEM_END) {
        UNLOCK();
        hos_fatal(NULL, "system region exhausted (%#llx bytes asked)", (unsigned long long)size);
    }
    sys_next = a + size;
    UNLOCK();
    memset(GP(a), 0, size);
    HOS_WRITTEN(GP(a), size);
    return a;
}

static uint64_t region_alloc(uint64_t size, uint64_t align, int commit);
uint64_t hos_region_alloc(uint64_t size, uint64_t align) { return region_alloc(size, align, 1); }

/* (hookshot, night 10/11) What the game maps in its address regions (AllocateMemoryPages / FreeMemoryPages above 4 GB), said
   every 30 s at most: HOS_REGION_LOG=1 */
static int64_t region_mapped, region_allocated, region_freed, region_kept;
static void region_count(int64_t *what, int64_t by) { __atomic_fetch_add(what, by, __ATOMIC_RELAXED); }
static void region_note(void) {
    static int on = -1;
    static int64_t last;
    if (on < 0) on = env_on("HOS_REGION_LOG");
    if (!on) return;
    int64_t before = __atomic_load_n(&last, __ATOMIC_RELAXED);
    const int64_t now = hos_now_ns();
    if (now - before < 30000000000LL || !__atomic_compare_exchange_n(&last, &before, now, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
    hos_log("mem: address regions: %lld MB mapped by the game now (%lld MB mapped in all, %lld MB freed: %lld MB of it kept under GPU imports)",
            (long long)(__atomic_load_n(&region_mapped, __ATOMIC_RELAXED) >> 20), (long long)(__atomic_load_n(&region_allocated, __ATOMIC_RELAXED) >> 20),
            (long long)(__atomic_load_n(&region_freed, __ATOMIC_RELAXED) >> 20), (long long)(__atomic_load_n(&region_kept, __ATOMIC_RELAXED) >> 20));
}

static uint64_t region_alloc(uint64_t size, uint64_t align, int commit) {
    if (align < 0x1000) align = 0x1000;
    size = (size + 0xFFF) & ~0xFFFULL;
    LOCK();
    /* first fit, top-down, between existing regions (kept sorted by address, descending) */
    uint64_t top = HOS_SPACE_END, found = 0;
    int at = nregions;
    for (int i = 0; i <= nregions; i++) {
        uint64_t bottom = i < nregions ? regions[i].addr + regions[i].size : HOS_HEAP_BASE + hos_heap_size;
        if (top >= size) {
            uint64_t a = (top - size) & ~(align - 1);
            if (a >= bottom && a >= HOS_HEAP_BASE + hos_heap_size) { found = a; at = i; break; }
        }
        if (i < nregions) top = regions[i].addr;
    }
    if (!found)   /* the holes below the heap, top-down, between the regions there (the list is descending) */
        for (int h = 1; h >= 0 && !found; h--) {
            uint64_t hi = low_hole[h][1];
            int i = 0;
            while (i < nregions && regions[i].addr >= hi) i++;
            for (;; i++) {
                const uint64_t lo = i < nregions && regions[i].addr + regions[i].size > low_hole[h][0]
                                        ? regions[i].addr + regions[i].size : low_hole[h][0];
                if (hi >= size) {
                    const uint64_t a = (hi - size) & ~(align - 1);
                    if (a >= lo) { found = a; at = i; break; }
                }
                if (i >= nregions || regions[i].addr < low_hole[h][0]) break;
                hi = regions[i].addr;
            }
        }
    if (found) {
        if (nregions == capregions) {
            capregions = capregions ? capregions * 2 : 64;
            regions = (Range *)realloc(regions, capregions * sizeof(Range));
        }
        memmove(&regions[at + 1], &regions[at], (nregions - at) * sizeof(Range));
        regions[at].addr = found;
        regions[at].size = size;
        regions[at].freeing = 0;   /* (the slot still holds the entry moved up from it, which may be being freed) */
        nregions++;
        if (found < region_floor && found >= HOS_HEAP_BASE) region_floor = found;   /* (not the holes below) */
    }
    UNLOCK();
    if (found && commit) hos_commit(found, size);
    if (!found) hos_fatal(NULL, "address space exhausted (%#llx bytes asked, heap %#llx)", (unsigned long long)size,
                          (unsigned long long)hos_heap_size);
    return found;
}

static uint64_t pages_used[(1ULL << 32) / 0x1000 / 64];

static void pages_mark_used_locked(uint64_t addr, uint64_t size) {
    for (uint64_t p = addr >> 12; p < (addr + size + 0xFFF) >> 12 && p < (1ULL << 20); p++) pages_used[p >> 6] |= 1ULL << (p & 63);
}

static void discard_pages(uint64_t addr, uint64_t size) {
    if (!hos_guest_range(addr, size)) return;   /* (hookshot: guest pages only; Linux's discard zeroes what it is given) */
    const uint64_t lo = (addr + 0xFFF) & ~0xFFFULL, hi = (addr + size) & ~0xFFFULL;
    static int keep = -1;
    if (keep < 0) keep = env_on("HOS_KEEP_FREED_PAGES");
    if (hi <= lo || keep) return;
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
    extern int hos_large_pages;   /* (hos_main.c: large pages cannot be discarded) */
    if (hos_large_pages) return;
    typedef DWORD(WINAPI * DiscardFn)(PVOID, SIZE_T);
    static DiscardFn discard;
    static int looked;
    if (!looked) {
        looked = 1;
        discard = (DiscardFn)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "DiscardVirtualMemory");
    }
    /* (HOS_GUEST_PAGES: over dropped or restored granules, piece by piece; dropped ones wholly inside are forgotten) */
    if (hos_gp_ready() && hos_gp_touches(GP(lo), (size_t)(hi - lo))) hos_gp_reset(GP(lo), (size_t)(hi - lo));
    else {
        hos_gp_hold(1);   /* (not while a drop splits the reservation: nothing when off) */
    if (!discard || discard(GP(lo), (SIZE_T)(hi - lo)) != ERROR_SUCCESS)
        VirtualAlloc(GP(lo), (SIZE_T)(hi - lo), MEM_RESET, PAGE_READWRITE);
        hos_gp_hold(0);
    }
#elif !defined(__EMSCRIPTEN__)
    extern int hos_posix_guest_discard(void *p, size_t n);   /* (hos_main.c: MADV_REMOVE on the memfd window) */
    extern int rx_guest_imported(uint64_t lo, uint64_t hi);
    /* (pages a dma-buf import holds stay where they are, zeroed: a hole punched under one would leave the GPU on
       the old pages and the game on new ones) */
    if (rx_guest_imported(lo, hi)) memset(GP(lo), 0, (size_t)(hi - lo));
    else hos_posix_guest_discard(GP(lo), (size_t)(hi - lo));
#endif
}

/* (hookshot) the address regions (hos_region_alloc's), at most cap of them into out ({start, end}); how many there are. For
   host/prompts.c: where the game keeps its resources */
int hos_regions(uint64_t (*out)[2], int cap) {
    LOCK();
    const int n = nregions;
    for (int i = 0; i < n && i < cap; i++) out[i][0] = regions[i].addr, out[i][1] = regions[i].addr + regions[i].size;
    UNLOCK();
    return n;
}

/* (hookshot, night 10/11) The parts of [lo, hi) that no region holds now, in the address regions' part of the window,
   decommitted: what the game mapped in regions it has freed since. The game frees a region without unmapping its pages first
   (20.9 GB mapped and 52 MB unmapped over six areas, hs493), and a freed region's pages were only discarded: the regions'
   commit stood at their whole span, 8 GB, for the ~2.9 GB mapped at a time. The Vulkan back end calls this under its pool
   lock, for ranges none of its imports holds (vk_memory.cpp release_uncovered), so no page the GPU maps goes and no import is
   made meanwhile; a region made since is not a gap, so a range freed and handed out again keeps its pages. Returns the bytes
   given back */
uint64_t hos_region_release_gaps(uint64_t lo, uint64_t hi) {
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
    uint64_t done = 0;
    LOCK();
    const uint64_t floor = region_floor < HOS_SPACE_END ? region_floor : HOS_SPACE_END;
    if (lo < floor) lo = floor;
    if (hi > HOS_SPACE_END) hi = HOS_SPACE_END;
    lo = (lo + 0xFFF) & ~0xFFFULL, hi &= ~0xFFFULL;
    /* (the regions are sorted by address, descending: the gaps of [lo, hi) from its top down) */
    uint64_t top = hi;
    for (int i = 0; i < nregions && top > lo; i++) {
        const uint64_t a = regions[i].addr, b = (a + regions[i].size + 0xFFF) & ~0xFFFULL;
        if (a >= top) continue;   /* (above what is left of the range) */
        if (b < top) {
            const uint64_t g = b > lo ? b : lo;
            if (top > g && VirtualFree(GP(g), (SIZE_T)(top - g), MEM_DECOMMIT)) done += top - g;
        }
        top = a & ~0xFFFULL;
    }
    if (top > lo && VirtualFree(GP(lo), (SIZE_T)(top - lo), MEM_DECOMMIT)) done += top - lo;
    UNLOCK();
    return done;
#else
    (void)lo, (void)hi;
    return 0;
#endif
}

/* nn::os::FreeAddressRegion's work. The discard hands the pages back to the system, milliseconds for a large region: the
   thread gives its virtual core up for it, as for a file read, so the threads that share the core (on the console, often
   of a higher priority than the game's resource memory thread that frees) are not held up for all of it. The region stays
   listed until its pages are discarded, marked as being freed, so that no hos_region_alloc meanwhile is handed pages the
   discard would wipe. A second free of the address meanwhile (the guest freeing it twice) finds it marked and does
   nothing: were it to unlist the region, the address could be handed out again while the first discard still runs, and
   the first free would then unlist the new region. The region leaves the list before the thread waits for its core
   again. The renderer's question in discard_pages (rx_guest_imported) gives a core up and takes one back on its own;
   here the thread holds none by then, so its hos_vcore_away answers 0 (this away has cleared any escape) and only the
   hos_vcore_back below takes a core back: once for each region freed, as before this function gave its core up */
void hos_region_free(uint64_t addr) {
    uint64_t size = 0;
    int found = 0;
    LOCK();
    for (int i = 0; i < nregions; i++) {
        if (regions[i].addr == addr && !regions[i].freeing) {
            regions[i].freeing = 1;
            pages_mark_used_locked(regions[i].addr, regions[i].size);
            size = regions[i].size;
            found = 1;
            break;
        }
    }
    UNLOCK();
    if (!found) return;
    /* (hookshot) HOS_DISCARD_HOLDS_CORE=1: the core kept through the discard, as before (OoT's Resource_Memory thread held a
       core for 14% of the 11.6 s after arriving on Death Mountain Trail, and ModuleSystemWorker1 waited for it) */
    static int holds = -1;
    if (holds < 0) holds = env_on("HOS_DISCARD_HOLDS_CORE");
    const int away = holds ? 0 : hos_vcore_away();
    discard_pages(addr, size);   /* (nothing for a region of no size) */
    LOCK();
    for (int i = 0; i < nregions; i++) {
        if (regions[i].addr == addr && regions[i].freeing) {
            memmove(&regions[i], &regions[i + 1], (nregions - i - 1) * sizeof(Range));
            nregions--;
            break;
        }
    }
    UNLOCK();
    if (away) hos_vcore_back();
    /* (hookshot) and its pages given back, but for those a GPU import holds: those go when the import does */
    extern void rx_guest_release(uint64_t lo, uint64_t hi);
    if (size) rx_guest_release(addr, addr + size);
}

/* (hookshot) whether [addr, addr + n) is guest memory the game was given: the image, the heap so far, or one address region
   (host/cheats.c checks its reads and writes with it) */
int hos_mem_mapped(uint64_t addr, uint64_t n) {
    if (addr + n < addr) return 0;
    if (addr >= hos_image_base && addr + n <= hos_image_end) return 1;
    if (addr >= HOS_HEAP_BASE && addr + n <= HOS_HEAP_BASE + hos_heap_size) return 1;
    int ok = 0;
    LOCK();
    for (int i = 0; i < nregions && !ok; i++) ok = addr >= regions[i].addr && addr + n <= regions[i].addr + regions[i].size;
    UNLOCK();
    return ok;
}

/* ---- nn::os memory -------------------------------------------------------------------------------------- */

#if defined(A64_WRITE_TRACK) && !defined(__EMSCRIPTEN__)
/* HOS_WRITTEN natively (the soft write watch, rt/a64.h A64_MARK): host code wrote [p, p + n) of guest memory */
void hos_mark_written(const void *p, size_t n) {
    const uintptr_t a = (uintptr_t)p - (uintptr_t)A64_MEM;
    if (!n || (uintptr_t)p < (uintptr_t)A64_MEM || a >= HOS_GUEST_SPACE) return;
    for (uintptr_t q = a >> 12; q <= (a + n - 1) >> 12 && q < A64_DIRTY_PAGES; q++)
        __atomic_store_n(&a64_dirty[q], 1, __ATOMIC_RELEASE);
}
#endif

/* (hookshot) a heap of size bytes ends below the address regions and the guest window's end (no wrap-around: the size
   comes from the game). Also svc SetHeapSize's check (hos_svc.c) */
int hos_heap_size_fits(uint64_t size) {
    const uint64_t limit = region_floor < HOS_SPACE_END ? region_floor : HOS_SPACE_END;
    return limit >= HOS_HEAP_BASE && size <= limit - HOS_HEAP_BASE;
}
/* nn::os::SetMemoryHeapSize(unsigned long) -> Result */
HLE(_ZN2nn2os17SetMemoryHeapSizeEm) {
    uint64_t size = ARG(0);
    HOS_TRACE_CALL("SetMemoryHeapSize");
    if (!hos_heap_size_fits(size)) {
        hos_log("SetMemoryHeapSize(%#llx): collides with address regions at %#llx", (unsigned long long)size,
                (unsigned long long)region_floor);
        RET(0xCA01);   /* svc::ResultOutOfMemory */
        return;
    }
    if (size < hos_heap_size) {
        memset(GP(HOS_HEAP_BASE + size), 0, hos_heap_size - size);
        HOS_WRITTEN(GP(HOS_HEAP_BASE + size), hos_heap_size - size);
    }
    /* HOS_PREFAULT=1 (experiment): the new heap pages touched now (a zero written to fresh, zero memory), so that the
       game's first touches while streaming are not page faults (which may hold up the renderer's write-watch scans) */
    if (size > hos_heap_size && env_on("HOS_PREFAULT")) {
        const int64_t t0 = hos_now_ns();
        for (uint64_t a = HOS_HEAP_BASE + hos_heap_size; a < HOS_HEAP_BASE + size; a += 4096) *(volatile uint8_t *)GP(a) = 0;
        hos_log("heap: %llu MB prefaulted in %.0f ms", (unsigned long long)((size - hos_heap_size) >> 20), (hos_now_ns() - t0) / 1e6);
    }
    /* (hookshot) Windows: the heap is left reserved and committed as it is touched (mkw_glue.c commit_on_touch), not all at
       once. OoT asks for ~6.9 GB and touches a fraction of it; committed, all of it counted against RAM plus the page file
       (a 12 GB commit charge at the title screen against a 5 GB working set), and players hit "out of memory" with GBs of
       RAM free. Host calls that hand guest memory to the kernel (nn_fs.c ReadFile/WriteFile) commit their range first.
       HOS_HEAP_COMMIT=1: all of it committed here, as before. Linux's window is MAP_NORESERVE already */
#if defined(_WIN32)
    static int commit_all = -1;
    if (commit_all < 0) commit_all = env_on("HOS_HEAP_COMMIT");
    if (size > hos_heap_size && commit_all) hos_commit(HOS_HEAP_BASE + hos_heap_size, size - hos_heap_size);
#else
    if (size > hos_heap_size) hos_commit(HOS_HEAP_BASE + hos_heap_size, size - hos_heap_size);
#endif
    hos_heap_size = size;
    hos_log("heap: %llu MB at %#llx", (unsigned long long)(size >> 20), (unsigned long long)HOS_HEAP_BASE);
    RET(0);
}

/* nn::os::AllocateMemoryBlock(unsigned long*, unsigned long) -> Result: 2 MB-aligned blocks from the heap */
HLE(_ZN2nn2os19AllocateMemoryBlockEPmm) {
    uint64_t out = ARG(0), size = (ARG(1) + 0x1FFFFF) & ~0x1FFFFFULL;
    HOS_TRACE_CALL("AllocateMemoryBlock");
    LOCK();
    uint64_t a = HOS_HEAP_BASE + hos_heap_used;
    int ok = hos_heap_used + size <= hos_heap_size;
    if (ok) hos_heap_used += size;
    UNLOCK();
    if (!ok) {
        hos_log("AllocateMemoryBlock(%#llx): heap exhausted", (unsigned long long)size);
        RET(0x1A01);   /* os::ResultOutOfMemory */
        return;
    }
    G64(out) = a;
    RET(0);
}

/* nn::os::AllocateAddressRegion(unsigned long*, unsigned long) -> Result */
HLE(_ZN2nn2os21AllocateAddressRegionEPmm) {
    HOS_TRACE_CALL("AllocateAddressRegion");
    /* (hookshot, night 10/11) A region the game asks for is address space only, as on the console: the game maps its pages
       before it uses them (AllocateMemoryPages commits them), and a page touched without that is committed by the first-touch
       handler (mkw_glue.c). OoT's "Resource Memory" thread asks for a region per resource (5078 in the first 100 s, 2194 of
       them of 11 MB in a 3 minute walk) and maps what the console leaves it: 2923 MB once Kokiri Forest is loaded, the 9809 MB
       of an application less its 6894 MB heap (HOS_REGION_LOG=1, hs492). Committed whole at their reservation, the regions
       stood at 8 GB of commit (0x7e00000000 up, vmmap.ps1), the game's memory at 15.1 GB and the process at 29-30 GB after a
       few areas: past what a 16 GB PC with a 24 GB page file holds without growing the file in play (hs483: pool imports
       refused, then the device lost). A freed region's pages are given back (hos_region_release_gaps).
       HOS_REGION_COMMIT=1: committed whole, as before */
    static int whole = -1;
    if (whole < 0) {
        whole = env_on("HOS_REGION_COMMIT");
        hos_log("mem: the game's address regions are committed %s", whole ? "whole (HOS_REGION_COMMIT)" : "as the game maps their pages");
    }
    G64(ARG(0)) = region_alloc(ARG(1), 0x10000, whole);
    RET(0);
}

/* nn::os::FreeAddressRegion(unsigned long) -> Result */
HLE(_ZN2nn2os17FreeAddressRegionEm) {
    HOS_TRACE_CALL("FreeAddressRegion");
    hos_region_free(ARG(0));
    RET(0);
}

/* (hookshot, 11-13) The pages the game holds through AllocateMemoryPages, counted. On the console they come out of the same
   memory as the heap: the application's total (hos_total_memory, 9809 MB) less the heap is all it can map, and past that
   AllocateMemoryPages fails. Here it never did, and OoT held 8.4 GB of mapped pages beside a 6.9 GB heap (13 GB of guest
   memory resident, a working set of 19.6 GB, against the console's 12 GB in all). One bit a 4 KB page of the guest window,
   so that pages mapped twice or freed twice count once. HOS_PAGES_LIMIT=1: past the console's total (heap + pages),
   AllocateMemoryPages fails as it would there; unset, only counted and logged.
   The bits are kept a GB of the window at a time (32 KB each), allocated when a page in that GB is first mapped: one array for
   the whole 512 GB window would be 16 MB of static data, which Windows charges to the commit this change is about */
#define PAGES_LEAF_SHIFT 30   /* (a GB of the window: 2^18 pages, 4096 words of bits) */
#define PAGES_LEAF_PAGES (1ULL << (PAGES_LEAF_SHIFT - 12))
static uint64_t *mapped_bits[(HOS_GUEST_SPACE + (1ULL << PAGES_LEAF_SHIFT) - 1) >> PAGES_LEAF_SHIFT];
static uint64_t mapped_bytes, mapped_peak, mapped_logged;
/* map 1: marked mapped, 0: marked free, 2: nothing changed. The bytes that change (or would, for 2: those not mapped yet) */
static int64_t pages_count_locked(uint64_t a, uint64_t n, int map) {
    int64_t changed = 0;
    for (uint64_t p = a >> 12; p < (a + n + 0xFFF) >> 12 && p < (HOS_GUEST_SPACE >> 12); p++) {
        uint64_t **leaf = &mapped_bits[p / PAGES_LEAF_PAGES];
        if (!*leaf) {   /* (nothing in this GB mapped yet) */
            if (map == 2) { changed += 0x1000; continue; }
            if (map == 0) continue;
            *leaf = (uint64_t *)calloc(PAGES_LEAF_PAGES / 64, sizeof(uint64_t));
            if (!*leaf) continue;   /* (not counted: the report errs low, as before this count) */
        }
        const uint64_t i = p % PAGES_LEAF_PAGES, bit = 1ULL << (i & 63);
        uint64_t *word = &(*leaf)[i >> 6];
        const int was = (*word & bit) != 0;
        if (map == 2) changed += was ? 0 : 0x1000;
        else if (map == 1 && !was) *word |= bit, changed += 0x1000;
        else if (map == 0 && was) *word &= ~bit, changed += 0x1000;
    }
    return changed;
}
static void pages_report_locked(void) {
    if (mapped_bytes > mapped_peak) mapped_peak = mapped_bytes;
    const uint64_t step = 256ULL << 20;
    if (mapped_bytes / step != mapped_logged / step) {
        mapped_logged = mapped_bytes;
        hos_log("mem: the game's mapped pages %llu MB (peak %llu); heap %llu MB; the console's total %llu MB",
                (unsigned long long)(mapped_bytes >> 20), (unsigned long long)(mapped_peak >> 20),
                (unsigned long long)(hos_heap_size >> 20), (unsigned long long)(hos_total_memory >> 20));
    }
}

/* (hookshot) what the game is told it uses: base (its program, and the system's share) plus the heap and its mapped pages;
   base alone with HOS_MEMINFO_OLD=1. QueryMemoryInfo below, and svc GetInfo's used memory (hos_svc.c). Never more than the
   console's total: nothing refuses AllocateMemoryPages by default (HOS_PAGES_LIMIT), and past the total the game's
   free = total - used would wrap to a huge figure and it would map without limit again */
uint64_t hos_used_memory(uint64_t base) {
    static int old = -1;
    if (old < 0) old = env_on("HOS_MEMINFO_OLD");
    if (old) return base;
    LOCK();
    const uint64_t mapped = mapped_bytes;
    UNLOCK();
    const uint64_t used = base + hos_heap_size + mapped;
    return used < hos_total_memory ? used : hos_total_memory;
}

/* nn::os::AllocateMemoryPages(unsigned long, unsigned long) -> Result: new pages read as zero */
HLE(_ZN2nn2os19AllocateMemoryPagesEmm) {
    HOS_TRACE_CALL("AllocateMemoryPages");
    const uint64_t a = ARG(0), n = ARG(1), end = a + n;
    if (!hos_guest_range(a, n)) {   /* (hookshot: a range that wrapped around was zeroed whole, past the guest window) */
        hos_log("AllocateMemoryPages(%#llx, %#llx): outside the guest window", (unsigned long long)a, (unsigned long long)n);
        RET(0xCA01);   /* svc::ResultOutOfMemory */
        return;
    }
    {
        static int limit = -1;
        if (limit < 0) limit = env_on("HOS_PAGES_LIMIT");
        LOCK();
        if (limit && hos_heap_size + mapped_bytes + (uint64_t)pages_count_locked(a, n, 2) > hos_total_memory) {
            UNLOCK();
            static int told;
            if (told++ < 20)
                hos_log("mem: AllocateMemoryPages(%#llx, %llu MB) refused: %llu MB mapped + %llu MB heap would pass the console's %llu MB",
                        (unsigned long long)a, (unsigned long long)(n >> 20), (unsigned long long)(mapped_bytes >> 20),
                        (unsigned long long)(hos_heap_size >> 20), (unsigned long long)(hos_total_memory >> 20));
            RET(0xCA01);   /* svc::ResultOutOfMemory */
            return;
        }
        mapped_bytes += (uint64_t)pages_count_locked(a, n, 1);
        pages_report_locked();
        UNLOCK();
    }
    static int zero_all = -1;
    if (zero_all < 0) zero_all = env_on("HOS_ZERO_ALL_PAGES");
    if (end > (1ULL << 32) && !zero_all) {
        /* (mkw-recomp) above 4 GB the used-page bitmap does not reach, and zeroing the whole range wiped live data
           when the game mapped pages over memory it already used (a heap's list lost its links). Horizon's rule
           instead: only newly mapped pages read as zero. A commit leaves committed pages as they are and gives fresh
           ones zeroed; FreeMemoryPages decommits. */
        hos_commit(a, n);
        region_count(&region_mapped, (int64_t)n), region_count(&region_allocated, (int64_t)n);
        region_note();
    } else if (zero_all || end > (1ULL << 32) || end < a) {
        memset(GP(a), 0, n);
    } else {
        uint64_t runs[64][2];
        int nruns = 0, overflow = 0, in_run = 0;
        LOCK();
        for (uint64_t p = a & ~0xFFFULL; p < end; p += 0x1000) {
            const uint64_t pg = p >> 12, bit = 1ULL << (pg & 63);
            const int used = (pages_used[pg >> 6] & bit) != 0;
            pages_used[pg >> 6] |= bit;
            if (used && !in_run) {
                if (nruns == 64) overflow = 1;
                else runs[nruns][0] = p < a ? a : p;
                in_run = 1;
            } else if (!used && in_run) {
                if (nruns < 64) runs[nruns++][1] = p;
                in_run = 0;
            }
        }
        if (in_run && nruns < 64) runs[nruns++][1] = end;
        UNLOCK();
        if (overflow) memset(GP(a), 0, n);
        else
            for (int i = 0; i < nruns; i++) memset(GP(runs[i][0]), 0, runs[i][1] - runs[i][0]);
    }
    HOS_WRITTEN(GP(a), n);
    RET(0);
}

/* nn::os::FreeMemoryPages(unsigned long, unsigned long) -> Result */
HLE(_ZN2nn2os15FreeMemoryPagesEmm) {
    HOS_TRACE_CALL("FreeMemoryPages");
    if (!hos_guest_range(ARG(0), ARG(1))) {   /* (hookshot: decommitted or discarded: guest pages only) */
        hos_log("FreeMemoryPages(%#llx, %#llx): outside the guest window", (unsigned long long)ARG(0), (unsigned long long)ARG(1));
        RET(0xCA01);
        return;
    }
    LOCK();   /* (hookshot: no longer counted against the console's total, AllocateMemoryPages above) */
    mapped_bytes -= (uint64_t)pages_count_locked(ARG(0), ARG(1), 0);
    pages_report_locked();
    UNLOCK();
    if (ARG(0) + ARG(1) > (1ULL << 32)) {   /* (mkw-recomp: the pair of AllocateMemoryPages' commit above) */
        const uint64_t lo = (ARG(0) + 0xFFF) & ~0xFFFULL, hi = (ARG(0) + ARG(1)) & ~0xFFFULL;
        /* (pages the Vulkan back end imported, as part of a pool's 16 MB chunk, stay committed and are zeroed: the GPU
           still maps them, and decommitted pages under an import faulted the device. Read as zero, as freed pages) */
        extern int rx_guest_imported(uint64_t lo, uint64_t hi);
        region_count(&region_mapped, -(int64_t)ARG(1)), region_count(&region_freed, (int64_t)ARG(1));
        region_note();
        if (hi > lo && rx_guest_imported(lo, hi)) {
            memset(GP(lo), 0, (size_t)(hi - lo));
            region_count(&region_kept, (int64_t)(hi - lo));
        }
#if defined(_WIN32)
        else if (hi > lo && hos_gp_ready() && hos_gp_touches(GP(lo), (size_t)(hi - lo))) hos_gp_decommit(GP(lo), (size_t)(hi - lo));
        else if (hi > lo) hos_gp_hold(1), VirtualFree(GP(lo), (SIZE_T)(hi - lo), MEM_DECOMMIT), hos_gp_hold(0);
#else
        else if (hi > lo) {   /* (linux: kept mapped, the pages read as zero; MADV_REMOVE on the memfd window) */
            extern int hos_posix_guest_discard(void *p, size_t n);
            hos_posix_guest_discard(GP(lo), (size_t)(hi - lo));
        }
#endif
        RET(0);
        return;
    }
    discard_pages(ARG(0), ARG(1));
    RET(0);
}

/* nn::os::QueryMemoryInfo(nn::os::MemoryInfo*). TOTK reads +0 (total available) and +8 (total used):
   heap = align_down(total - used, 2 MB) - 0x43800000 (main+0x1131004). */
HLE(_ZN2nn2os15QueryMemoryInfoEPNS0_10MemoryInfoE) {
    uint64_t info = ARG(0);
    HOS_TRACE_CALL("QueryMemoryInfo");
    G64(info + 0x00) = hos_total_memory;                       /* total available */
    /* total used: program + system, and (hookshot, 11-13) the heap and the pages the game has mapped, as Horizon counts it.
       OoT's resource memory (0x8080f010) takes total - used as what it may still map, before each AllocateMemoryPages: with
       program + system alone it read ~9.6 GB free whatever it held, and mapped 7.9 GB of pages beside its 6.9 GB heap in 15
       minutes of play (2.9 GB is all the console leaves it), never giving any back. HOS_MEMINFO_OLD=1: program + system, as before */
    G64(info + 0x08) = hos_used_memory(hos_program_size + (16ULL << 20));
    G64(info + 0x10) = hos_heap_size;                          /* total heap */
    G64(info + 0x18) = hos_heap_used;                          /* allocated heap */
    G64(info + 0x20) = hos_program_size;                       /* program */
    G64(info + 0x28) = 0;                                      /* thread stacks */
    G32(info + 0x30) = 1;                                      /* threads */
}

/* nn::os::SetMemoryAllocatorForThreadLocal runs as the SDK's own code: it stores the game's allocator where
   nnosAllocateMemoryForThreadLocal finds it, which nnmusl uses for each thread's TLS (hos_thread.c thread_main). */

/* nn::os::SetDyingMessageRegion(unsigned long, unsigned long) -> Result */
HLE(_ZN2nn2os21SetDyingMessageRegionEmm) { RET(0); }
