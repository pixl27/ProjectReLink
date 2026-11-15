#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct HosRegionGap {
    uint64_t addr, size;
} HosRegionGap;

typedef uint64_t (*HosRegionGapRelease)(uint64_t lo, uint64_t hi, void *opaque);

/* Visit the parts of [lo, hi) outside every active region. Regions are sorted by descending address. */
static inline uint64_t hos_region_gaps_release(const HosRegionGap *regions, size_t count, uint64_t lo, uint64_t hi,
                                               HosRegionGapRelease release, void *opaque) {
    if (!release || hi <= lo) return 0;
    uint64_t done = 0, top = hi;
    for (size_t i = 0; i < count && top > lo; i++) {
        const uint64_t a = regions[i].addr;
        const uint64_t b = a + regions[i].size;
        if (a >= top) continue;
        if (b < top) {
            const uint64_t gap = b > lo ? b : lo;
            if (top > gap) done += release(gap, top, opaque);
        }
        top = a;
    }
    if (top > lo) done += release(lo, top, opaque);
    return done;
}
