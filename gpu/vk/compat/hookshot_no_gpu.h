// (hookshot) Why the Vulkan loader listed no GPU at all, where the PC says: for the renderer's log line (gpu/vk/vk_main.cpp
// pick_device) and the window after the start (host/hos_crash_dialog.h reads that line back), one implementation for the
// host (C) and the renderer (C++).
//
// A Linux player on a real-time kernel (PREEMPT_RT: CachyOS's rt-bore) had NVIDIA's driver not loaded.
// Linux says enough to tell that apart: the graphics cards (/sys/bus/pci/devices/*, class 0x03xxxx), whether NVIDIA's
// kernel driver is loaded (/sys/module/nvidia) and whether the kernel is a real-time one (/sys/kernel/realtime, or
// PREEMPT_RT in /proc/sys/kernel/version). Windows says nothing here (a NULL root): the cause stays not known.
// HOOKSHOT_NO_GPU_KEYS_ONLY: the causes and their names alone, no file reads (the crash window's plain C)
#ifndef HOOKSHOT_NO_GPU_H
#define HOOKSHOT_NO_GPU_H

#include <stddef.h>
#include <string.h>

enum {
    HOOKSHOT_NO_GPU_UNKNOWN,           // nothing more known (no NVIDIA card, or not Linux)
    HOOKSHOT_NO_GPU_NVIDIA_RT,         // an NVIDIA card, NVIDIA's driver not loaded, a real-time kernel
    HOOKSHOT_NO_GPU_NVIDIA_UNLOADED,   // an NVIDIA card, NVIDIA's driver not loaded
    HOOKSHOT_NO_GPU_NVIDIA_VULKAN,     // an NVIDIA card and NVIDIA's driver loaded, but no Vulkan device listed
    HOOKSHOT_NO_GPU_COUNT
};
typedef struct HookshotNoGpu {
    int cause, realtime, nvidia_card, nvidia_loaded;
    char cards[160];   // the graphics cards' PCI vendor:device, "10de:2d05 1002:164e" ("" none found)
} HookshotNoGpu;

// The cause's name in the log line, and back (the crash window: -1 for a name it does not know)
static inline const char *hookshot_no_gpu_key(int cause) {
    static const char *const keys[HOOKSHOT_NO_GPU_COUNT] = {"unknown", "nvidia-rt", "nvidia-unloaded", "nvidia-vulkan"};
    return cause >= 0 && cause < HOOKSHOT_NO_GPU_COUNT ? keys[cause] : keys[0];
}
static inline int hookshot_no_gpu_cause_of(const char *key, size_t n) {
    for (int c = 0; c < HOOKSHOT_NO_GPU_COUNT; c++)
        if (strlen(hookshot_no_gpu_key(c)) == n && !strncmp(hookshot_no_gpu_key(c), key, n)) return c;
    return -1;
}

#ifndef HOOKSHOT_NO_GPU_KEYS_ONLY
#include <stdio.h>
#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>

static inline int hookshot_no_gpu_read(const char *root, const char *rel, char *out, size_t cap) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", root, rel);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    const size_t n = fread(out, 1, cap - 1, f);
    fclose(f);
    out[n] = 0;
    out[strcspn(out, "\r\n")] = 0;
    return 1;
}
static inline int hookshot_no_gpu_is_dir(const char *root, const char *rel) {
    char path[1024];
    struct stat s;
    snprintf(path, sizeof path, "%s/%s", root, rel);
    return !stat(path, &s) && S_ISDIR(s.st_mode);
}
// What the PC says: root "" for its own /sys and /proc, a folder that holds sys/ and proc/ for a test, NULL for nothing
static inline void hookshot_no_gpu_probe(const char *root, HookshotNoGpu *g) {
    memset(g, 0, sizeof *g);
    if (!root) return;
    char text[256];
    g->realtime = (hookshot_no_gpu_read(root, "sys/kernel/realtime", text, sizeof text) && text[0] == '1') ||
                  (hookshot_no_gpu_read(root, "proc/sys/kernel/version", text, sizeof text) && strstr(text, "PREEMPT_RT"));
    g->nvidia_loaded = hookshot_no_gpu_is_dir(root, "sys/module/nvidia");
    int other_card = 0, incomplete = 0;
    char folder[1024];
    snprintf(folder, sizeof folder, "%s/sys/bus/pci/devices", root);
    DIR *d = opendir(folder);
    for (struct dirent *e; d && (e = readdir(d));) {
        if (e->d_name[0] == '.') continue;
        char rel[512], cls[32], vendor[32], device[32];
        snprintf(rel, sizeof rel, "sys/bus/pci/devices/%s/class", e->d_name);
        if (!hookshot_no_gpu_read(root, rel, cls, sizeof cls)) { incomplete = 1; continue; }
        if (strncmp(cls, "0x03", 4)) continue;   // (a display controller)
        snprintf(rel, sizeof rel, "sys/bus/pci/devices/%s/vendor", e->d_name);
        if (!hookshot_no_gpu_read(root, rel, vendor, sizeof vendor) || strncmp(vendor, "0x", 2)) { incomplete = 1; continue; }
        snprintf(rel, sizeof rel, "sys/bus/pci/devices/%s/device", e->d_name);
        if (!hookshot_no_gpu_read(root, rel, device, sizeof device) || strncmp(device, "0x", 2)) { incomplete = 1; continue; }
        if (!strcmp(vendor, "0x10de")) g->nvidia_card = 1;
        else other_card = 1;
        const size_t used = strlen(g->cards);
        snprintf(g->cards + used, sizeof g->cards - used, "%s%.4s:%.4s", used ? " " : "", vendor + 2, device + 2);
    }
    if (d) closedir(d);
    if (g->nvidia_card && !other_card && !incomplete)
        g->cause = g->nvidia_loaded ? HOOKSHOT_NO_GPU_NVIDIA_VULKAN : g->realtime ? HOOKSHOT_NO_GPU_NVIDIA_RT
                                                                               : HOOKSHOT_NO_GPU_NVIDIA_UNLOADED;
}
#else
static inline void hookshot_no_gpu_probe(const char *root, HookshotNoGpu *g) {
    (void)root;
    memset(g, 0, sizeof *g);
}
#endif
// The renderer's line (after "vk: "): "no Vulkan device (cause nvidia-rt): graphics cards 10de:2d05; ...". The length written
static inline size_t hookshot_no_gpu_line(char *out, size_t cap, const HookshotNoGpu *g) {
    const char *why = g->cause == HOOKSHOT_NO_GPU_NVIDIA_RT ? "NVIDIA's module is not loaded on a real-time kernel (PREEMPT_RT); "
                                                              "a driver compatibility or loading issue is possible"
                    : g->cause == HOOKSHOT_NO_GPU_NVIDIA_UNLOADED ? "NVIDIA's kernel module is not loaded"
                    : g->cause == HOOKSHOT_NO_GPU_NVIDIA_VULKAN ? "NVIDIA's module is loaded, but no Vulkan driver lists a GPU"
                                                                : "no Vulkan driver lists a GPU";
    const int n = snprintf(out, cap, "no Vulkan device (cause %s): %s%s%s", hookshot_no_gpu_key(g->cause), why,
                           g->cards[0] ? "; graphics cards " : "", g->cards);
    return n < 0 ? 0 : (size_t)n < cap ? (size_t)n : cap ? cap - 1 : 0;
}
#endif   // (HOOKSHOT_NO_GPU_KEYS_ONLY)

#endif
