#pragma once

#if defined(__linux__)
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// Whether [lo, hi) contains an allocated extent in a sparse file.  A hole
// reads as zero, but reading it through a shared mapping instantiates pages;
// callers that only need to recognize an all-zero hole can avoid that fault.
// Returns 1 for data, 0 for holes only, and -1 when the allocation map cannot
// be queried (the caller must then use its ordinary byte-wise path).
static inline int hos_sparse_file_has_data(int fd, uint64_t lo, uint64_t hi) {
    if (fd < 0 || hi < lo || hi > (uint64_t)INT64_MAX) return -1;
    if (hi == lo) return 0;
    char path[64];
    const int path_n = snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    if (path_n <= 0 || (size_t)path_n >= sizeof path) return -1;
    const int scan = open(path, O_RDONLY | O_CLOEXEC);
    if (scan < 0) return -1;
    const off_t data = lseek(scan, (off_t)lo, SEEK_DATA);
    const int saved = errno;
    close(scan);
    if (data < 0) return saved == ENXIO ? 0 : -1;
    return (uint64_t)data < hi;
}

// Zero only allocated extents of [lo, hi) in a sparse file's shared mapping. Holes already read as zero; writing
// across them would instantiate pages just as a guest region is being freed. Reopening through proc gives this scan
// an independent file offset, unlike dup(), whose offset is shared with concurrent SEEK_DATA users.
static inline int hos_sparse_file_zero_data(int fd, void *mapping, uint64_t lo, uint64_t hi) {
    if (fd < 0 || !mapping || hi < lo || hi > (uint64_t)INT64_MAX) return -1;
    if (hi == lo) return 0;
    char path[64];
    const int path_n = snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    if (path_n <= 0 || (size_t)path_n >= sizeof path) return -1;
    const int scan = open(path, O_RDONLY | O_CLOEXEC);
    if (scan < 0) return -1;
    uint64_t at = lo;
    while (at < hi) {
        const off_t data = lseek(scan, (off_t)at, SEEK_DATA);
        if (data < 0) {
            if (errno == ENXIO) break;
            close(scan);
            return -1;
        }
        if ((uint64_t)data >= hi) break;
        const off_t hole = lseek(scan, data, SEEK_HOLE);
        if (hole <= data) {
            close(scan);
            return -1;
        }
        const uint64_t end = (uint64_t)hole < hi ? (uint64_t)hole : hi;
        memset((uint8_t *)mapping + (uint64_t)data, 0, (size_t)(end - (uint64_t)data));
        at = end;
    }
    close(scan);
    return 0;
}
#endif
