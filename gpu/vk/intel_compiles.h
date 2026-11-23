// (hookshot) The Intel shader compiles row's Auto: where Intel's Windows compiler is entered side by side without being asked.
//
// IntelDriverGate (vk_internal.h) lets one thread at a time into igvk64's compiler because it faulted with two in it on an
// Arc A750 and a UHD. On the Arc B-series (Battlemage, PCI device 0xE2xx) with driver 32.0.101.9154 that cost minutes at
// every area's first visit and its reason did not show: run on a B580 (0xE20B, a Core i5-13600K, six background workers),
// a first visit froze the picture for 248 s in Hyrule Field (398 programs) and 409 s in the Market (322) with one thread
// in the compiler; side by side, over three launches and ten other areas, 1,111 programs were first compiled with no
// frame under 5 fps, each area steady in 20 to 80 s, and no fault, device loss or exit. So Auto is side by side on that
// family from the driver that was run, and one at a time on every other Intel GPU, as before: nothing was measured there.
#pragma once

#include <cstdint>

/// Whether the row's Auto means side by side on this Intel GPU and Windows driver (deviceID, driverVersion as Vulkan
/// reports them: Intel's Windows driverVersion is major << 14 | minor, 101 << 14 | 9154 for 32.0.101.9154)
inline bool intel_side_by_side_measured(uint32_t device, uint32_t version) {
    return (device & 0xFF00u) == 0xE200u && version >= (101u << 14 | 9154u);
}

/// The row's value (0 Auto, 1 Side by side, 2 One at a time) as the choice for this GPU
inline bool intel_side_by_side_chosen(int row, uint32_t device, uint32_t version) {
    return row == 1 || (row == 0 && intel_side_by_side_measured(device, version));
}

// (hookshot) The shader preparation on the other Intel GPUs, with the row on Auto. With one thread in the
// compiler it took hours: an Arc iGPU (Core Ultra 7 155H, driver 101.9175) at 77% after 7 h 35 min, its CPU and GPU both
// at 8%; an Arc A730M at 34% of its pipelines with "about 1187 min left". The same A730M on the same driver with the row
// on Side by side built 25,005 of 32,623 noted pipelines in 2,753 s on 16 workers (5 to 35 a second) with no driver
// fault: what ended that run was memory (a 16 GB PC at its commit limit, pipelines refused with -3).
// The fault the gate was put in for (igvk64 read 0x170 on a UHD, driver 101.7088) was the pipeline cache's data
// fetched while other threads built into it, which the fetch lock now keeps apart on every Intel GPU on Windows
// (device_policy.h pcache_fetch_lock). So from driver 101.9175, where side by side was run, the preparation's
// graphics pipelines and shader modules go side by side on Auto (vk_internal.h IntelPrepareLane), as many at once as
// the free memory allows (intel_prepare_workers); play keeps the row's choice. Compute pipelines stay one at a time
// (the gate, and compute_builds_serial: four at once froze a PC). Before 101.9175 (the Iris Xe's and UHD's legacy
// drivers among them) nothing changes, and the row's Side by side and One at a time hold in the preparation too.
constexpr uint32_t kIntelPrepareSideBySideFrom = 101u << 14 | 9175u;
constexpr uint64_t kIntelPrepareReserveMB = 4096;   // (memory left to the rest of the PC, as prepare_workers.h)
constexpr uint64_t kIntelPrepareWorkerMB = 768;     // (a worker, as prepare_workers.h's; Intel's own peak not measured yet)

/// Whether the preparation's graphics builds go side by side: the row (0 Auto, 1 Side by side, 2 One at a time) on this
/// Intel GPU and Windows driver. Wherever play's are; on Auto also from kIntelPrepareSideBySideFrom
inline bool intel_prepare_side_by_side_chosen(int row, uint32_t device, uint32_t version) {
    return intel_side_by_side_chosen(row, device, version) || (row == 0 && version >= kIntelPrepareSideBySideFrom);
}

/// The preparation's pipeline workers while Intel's compiler takes them side by side: want (the threads it would run) held
/// to what the free memory allows past kIntelPrepareReserveMB at kIntelPrepareWorkerMB each, at least 1. avail_mb: the
/// physical memory free now, UINT64_MAX when the system does not say (want, as before)
inline unsigned intel_prepare_workers(unsigned want, uint64_t avail_mb) {
    if (want <= 1 || avail_mb == UINT64_MAX) return want ? want : 1;
    const uint64_t room = avail_mb > kIntelPrepareReserveMB ? (avail_mb - kIntelPrepareReserveMB) / kIntelPrepareWorkerMB : 0;
    return (unsigned)(room < 1 ? 1 : room < want ? room : want);
}
