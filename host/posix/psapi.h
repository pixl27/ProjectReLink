/* (linux) psapi: the process's memory counters, from /proc/self/status (the working set is the resident set, private
   memory the anonymous part of it, the page faults those of getrusage: minor and major, as Windows counts soft and hard
   ones together), for the back end's memory log lines */
#ifndef HOOKSHOT_POSIX_PSAPI_H
#define HOOKSHOT_POSIX_PSAPI_H
#include <windows.h>
#include <sys/resource.h>
typedef struct {
    DWORD cb, PageFaultCount;
    SIZE_T PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage, QuotaPeakNonPagedPoolUsage,
        QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage;
} PROCESS_MEMORY_COUNTERS;
typedef struct {
    DWORD cb, PageFaultCount;
    SIZE_T PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage, QuotaPeakNonPagedPoolUsage,
        QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage, PrivateUsage;
} PROCESS_MEMORY_COUNTERS_EX;
static inline BOOL K32GetProcessMemoryInfo(HANDLE p, PROCESS_MEMORY_COUNTERS *out, DWORD cb) {
    (void)p;
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return FALSE;
    SIZE_T rss = 0, hwm = 0, anon = 0;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        unsigned long long kb;
        if (sscanf(line, "VmRSS: %llu", &kb) == 1) rss = (SIZE_T)kb << 10;
        else if (sscanf(line, "VmHWM: %llu", &kb) == 1) hwm = (SIZE_T)kb << 10;
        else if (sscanf(line, "RssAnon: %llu", &kb) == 1) anon = (SIZE_T)kb << 10;
    }
    fclose(f);
    memset(out, 0, cb);
    out->cb = cb, out->WorkingSetSize = rss, out->PeakWorkingSetSize = hwm, out->PagefileUsage = anon;
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) out->PageFaultCount = (DWORD)(ru.ru_minflt + ru.ru_majflt);
    if (cb >= sizeof(PROCESS_MEMORY_COUNTERS_EX)) ((PROCESS_MEMORY_COUNTERS_EX *)out)->PrivateUsage = anon;
    return TRUE;
}
#define GetProcessMemoryInfo K32GetProcessMemoryInfo
#endif
