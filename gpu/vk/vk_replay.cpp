// The NVN replay on Vulkan: the game's recorded command buffers (host/nvn_native.c, records in rx_nvn.h) at each
// submit, recorded into a Vulkan command buffer and submitted with the next value of a timeline semaphore (the NVN
// fences); and the window presents.
//
// Each recording has two command buffers: the replay's, and one submitted right before it that texture uploads and
// new images' layout changes go to (vk_texture.cpp), so no pass has to end for them. The stream's memory writes
// happen on the GPU timeline: guest memory is shared with the GPU, so a CPU write at replay time would reach draws
// recorded before it. Uniform updates (UPDATE_UB) are written at the end of their recording, and the draws after
// one bind a copy of the range with it in (ub_overlay), so a pass need not end for them either.
//
// Without a window (HOS_HEADLESS) presents render into an image of their own; RX_VK_SHOTS=<dir> writes every
// RX_VK_SHOT_EVERY-th one (default 300) there as a PNG.
#include <algorithm>
#include <sys/stat.h>
#ifndef _WIN32
#include <dlfcn.h>   // (jt_mark: hos_jt_mark_ext, looked up in the exe)
#endif
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <exception>
#include <thread>
#include <string>
#include <map>
#include <set>
#include <shared_mutex>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "vk_temporal.h"
#include "vk_fsr3_inputs.h"   // (the camera jitter while AMD FSR 3.1 upscales)
#include "vk_fsr3.h"          // (retire old FSR 3.1 images after leaving its upscaler)
#include "status_line.h"
#include "vk_reflex.h"
#ifdef TRACY_ENABLE
#include "../../host/hos_tracy.h"   // (HOOKSHOT_TRACY: the profiler's zones; nothing in any other build)
#endif
#include "vk_antilag.h"
#include "vk_draw.h"
#include "vk_dump.h"
#include "vk_paths.h"   // (log_path: the frame trace in the game folder's logs/)
#include "vk_gprof.h"
#include "vk_framegen.h"
#include "vk_dlss.h"   // (hookshot: dlss_fg_idle)
#include "vk_fps.h"
#include "vk_menu.h"
#include "segment_plan.h"   // (the parallel replay's segment count)
#include "oot_cheat_items.h"   // (hookshot: OOT_CHEAT_ITEM_BASE, OOT_CHEAT_AGE_ID: the item wheel's counts and pages)
#include "oot_wheel_state.h"   // (hookshot: the Item wheel and Cycle items rows' option ids)
#include "oot_teleport_capture.h"   // (hookshot: HOS_OPT_TELEPORT_PICTURES)
#include "vk_jobs.h"
#include "vk_shader.h"
#include "vk_state.h"
#include "vk_texture.h"
#include "vk_xfer.h"
#include "vk_present_rect.h"
#include "vk_present_trace.h"

#include "vk_internal.h"
#include "vk_tl_clock.h"   // (RX_VK_TL: the timeline's host time domain)
#include "vk_lag.h"   // (RX_VK_LAG_LOG)
#include "readback_late.h"   // (RX_VK_READBACK_LATE)
#include "exe_base.h"   // (RX_VK_RB_PROF's call sites: offsets in the exe)
#ifndef HOOKSHOT_ENV_H   // (env_on: the switches' one rule)
#include "hookshot_env.h"
#endif
#include "vk_programs.h"
#include "readback_late.h"
#include "cxx_escape.h"   // (the replay's threads: their exceptions said, a start that fails at the commit limit stops as out of memory)
#include "hold_gpu.h"   // (hookshot: sync_watch's rule for queues stopped past hold_ms)
#include "frame_overlap.h"   // (hookshot: when the game's frame wait is answered, and where the presents are made)
#include "submit_retry.h"    // (hookshot: a submit refused for want of memory made again)
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

namespace mkwvk {
// (Hookshot) the host's frame-timeline marks (HOS_JOBTRACE, tools/bench/frameline.py): hos_jt_mark_ext, exported by
// the game's exe and looked up there (the host's headers are not included here; vk_api.h is a shader-stamp input). A call that
// returns at once while the trace window is closed. Kinds: 9/10 the replay thread waiting for its segments, 11/12 a submit's
// capture made, 13/14 a parallel replay's dry pass, 15/16 a submission's memory work (its capture taken), 17 the segments handed
// out to the workers, 18 the replay thread's own segment recorded (RX_VK_PARALLEL_SELF);
// 19, 20, 21 (points, diagnosis): the mirrors synced before the segments, what came before submitted, their recordings begun
void jt_mark(int kind, uint64_t v) {
#ifdef _WIN32
    static void (*const fn)(int, uint64_t) =
        reinterpret_cast<void (*)(int, uint64_t)>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(nullptr), "hos_jt_mark_ext")));
    if (fn) fn(kind, v);
#else
    // (Linux: the exe exports it by name, nx2/build.py's --export-dynamic-symbol list; GetModuleHandle(nullptr) is no handle there)
    static void (*const fn)(int, uint64_t) = reinterpret_cast<void (*)(int, uint64_t)>(dlsym(RTLD_DEFAULT, "hos_jt_mark_ext"));
    if (fn) fn(kind, v);
#endif
}
extern std::atomic<bool> movie_networks_slow;   // (vk_draw.cpp: the movie networks left out, issue 109)
extern std::atomic<int64_t> menu_fade_world_at, menu_fade_pause_at, wheel_drawn_at;  // (vk_draw.cpp)
int64_t presents_now;   // (the presents so far, for SlowWatch)
namespace {
const char* rb_thread_label() {
    thread_local char name[40];
    thread_local bool have = false;
    if (!have) {
        have = true;
        snprintf(name, sizeof name, "tid%lu", (unsigned long)GetCurrentThreadId());
        typedef HRESULT(WINAPI * GetDesc)(HANDLE, PWSTR*);
        const GetDesc gd = (GetDesc)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription");
        PWSTR w = nullptr;
        if (gd && SUCCEEDED(gd(GetCurrentThread(), &w)) && w) {
            char u[40]{};
            WideCharToMultiByte(CP_UTF8, 0, w, -1, u, sizeof u - 1, nullptr, nullptr);
            if (u[0]) snprintf(name, sizeof name, "%s", u);
            LocalFree(w);
        }
    }
    return name;
}
// ---- (stream R2) RX_VK_TL=<present>: a timeline of what the renderer's threads do over 3 frames from that present on (threads by
// name, times in ms from the window's start): the submits and their replays, the CPU's event signals, the stops at event
// waits, the presents and their drains, the fence waits, the readbacks with their waits. Logging only ----
// (RX_VK_RB_PROF: what wait_recorded spent, by this thread since the readback began: the flush's wait (the pending late fix and the
// queue's lock), the recordings' timeline, and the readbacks whose recording was not known yet)
thread_local int64_t rb_wr_ns[2] = {0, 0}, rb_wr_miss = 0, rb_fl_ns[3] = {0, 0, 0};   // (queue_flush_for: the late fix's wait, the queue's lock, the flush)
struct TlEv {
    int64_t t;
    const char* what;
    uint64_t a, b, c;
    char thread[24];
};
constexpr uint32_t kTlMax = 1 << 15;
TlEv tl_ring[kTlMax];
std::atomic<uint32_t> tl_n{0};
std::atomic<int> tl_active{0};
int64_t tl_from = -2;
int64_t tl_t0 = 0;
inline int64_t tl_now() { return std::chrono::steady_clock::now().time_since_epoch().count(); }
// RX_VK_TL=<present>, or RX_VK_TL_S=<seconds after the first present> (the window opens at the first present after that)
int64_t tl_start_ns = 0;
bool tl_wanted() {
    static const int64_t from = getenv("RX_VK_TL") ? atoll(getenv("RX_VK_TL")) : -1;
    tl_from = from;
    if (getenv("RX_VK_TL_S")) tl_start_ns = tl_now() + (int64_t)(atof(getenv("RX_VK_TL_S")) * 1e9);
    return from >= 0 || tl_start_ns;
}
void tl_ev(const char* what, uint64_t a, uint64_t b, uint64_t c) {
    if (!tl_active.load(std::memory_order_relaxed)) return;
    const uint32_t i = tl_n.fetch_add(1);
    if (i >= kTlMax) return;
    TlEv& e = tl_ring[i];
    e.t = tl_now(), e.what = what, e.a = a, e.b = b, e.c = c;
    snprintf(e.thread, sizeof e.thread, "%s", rb_thread_label());
}
/// at the top of a present: opens the window at present `from`, writes it out 3 presents later
inline void tl_ev(const char* what, uint64_t a = 0, uint64_t b = 0) { tl_ev(what, a, b, 0); }
void tl_gpu_open();                  // (stream R2: the GPU side of the window, below with the recordings' timestamps)
void tl_gpu_close();
void tl_gpu_dump(int64_t t0_ns);
void tl_at_present(int64_t presents) {
    static const bool want = tl_wanted();
    if (!want) return;
    if (tl_start_ns && tl_now() >= tl_start_ns) tl_from = presents, tl_start_ns = 0;
    if (presents == tl_from && !tl_active.load()) tl_t0 = tl_now(), tl_n = 0, tl_active = 1, tl_gpu_open();
    static const int64_t frames = getenv("RX_VK_TL_FRAMES") ? std::max(1, atoi(getenv("RX_VK_TL_FRAMES"))) : 3;   // (the window's length in presents)
    if (presents == tl_from + frames + 11 && tl_from >= 0) tl_gpu_dump(tl_t0);   // (the recordings' times are read once their slots come round)
    if (presents == tl_from + frames && tl_active.load()) {
        tl_active = 0;
        tl_gpu_close();
        const uint32_t n = std::min<uint32_t>(tl_n.load(), kTlMax);
        std::vector<uint32_t> ord(n);
        for (uint32_t i = 0; i < n; i++) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [](uint32_t a, uint32_t b) { return tl_ring[a].t < tl_ring[b].t; });
        logf("vk: tl begin (%u events from present %lld)", n, (long long)tl_from);
        for (uint32_t i : ord) {
            const TlEv& e = tl_ring[i];
            logf("vk: tl %9.3f %-22s %-14s %#llx %#llx %#llx", (e.t - tl_t0) / 1e6, e.thread, e.what, (unsigned long long)e.a,
                 (unsigned long long)e.b, (unsigned long long)e.c);
        }
        logf("vk: tl end");
    }
}
} // namespace
// (vk_draw.cpp: real occlusion counts, RX_VK_OCC_REAL) each draw and dispatch of the stream numbered in stream order: on
// the replay thread (serial and dry passes) and on a segment's thread from the dry pass's number at its start
extern thread_local int64_t occ_draw_slot;
int occ_mode();
void occ_advance(uint64_t serial, uint64_t value);
uint64_t occ_report(uint64_t addr, uint64_t lo, uint64_t hi, uint64_t value, uint64_t fixed, uint64_t prog);
void occ_cond(uint64_t addr, int mode, uint64_t x, uint64_t y, bool on, uint64_t serial);
void occ_present(int64_t presents, uint64_t serial, uint64_t done);
bool occ_cond_pending(uint64_t addr);
void occ_resolve_all();
void occ_give_up();
const uint8_t* occ_stop_at = nullptr;   // (RX_VK_OCC_REAL=4: where a dry pass stopped for a condition on a report not read yet)
uint64_t occ_serial = 0;              // (the replay thread's)
thread_local uint64_t occ_wserial = 0;   // (a segment's thread)
uint64_t occ_win_lo = 0;              // (the serial at the last samples-passed reset)
struct OccSlot {   // (the draw's serial for draw's query)
    explicit OccSlot(uint64_t serial) { if (occ_mode()) occ_draw_slot = (int64_t)serial; }
    ~OccSlot() { occ_draw_slot = -1; }
};
void hang_watch_start();   // (EvilOOT, vk_hangdump.cpp: RX_VK_HANG_DUMP)
void gpu_hang_watch_start();   // (below: a GPU that stops finishing work, reported and ended)
struct CondNote { int64_t present; uint64_t addr, x, y; int mode; bool on; };   // (the last render conditions, for a dark frame)
CondNote cond_ring[1 << 15];
uint32_t cond_ring_at = 0;
namespace {

// ---- submissions: a timeline semaphore; value n = the n-th submit done ----
VkSemaphore timeline = VK_NULL_HANDLE;
std::atomic<uint64_t> timeline_sent{0};   // (the highest value a submission already in the queue signals: gpu_hang_watch_start)
void note_timeline_sent(uint64_t v) {
    uint64_t was = timeline_sent.load(std::memory_order_relaxed);
    while (v > was && !timeline_sent.compare_exchange_weak(was, v, std::memory_order_relaxed)) {}
}
// ---- (hookshot) a submit the driver refused for want of memory: memory given back and the submit made again
// (submit_retry.h). The replay's vkQueueSubmit2 calls go through submit_or_retry, each under dev.present_mu ----
bool submit_retry_on() {
    static const bool on = [] {
        const bool v = env_on_or("RX_VK_SUBMIT_RETRY", 1);   // (hookshot_env.h: unset is on)
        if (!v) logf("vk: a submit refused for want of memory is dropped, as before (RX_VK_SUBMIT_RETRY=0)");
        return v;
    }();
    return on;
}
/// (a test: RX_VK_TEST_SUBMIT_REFUSE=<n> answers the first try of every n-th submit with VK_ERROR_OUT_OF_DEVICE_MEMORY without
/// making it, as AMD's Windows driver answered a Legion Go's, so any card plays the refusal and what follows it. Unset: never)
bool test_submit_refused() {
    static const uint64_t every = getenv("RX_VK_TEST_SUBMIT_REFUSE") ? (uint64_t)std::max(0, atoi(getenv("RX_VK_TEST_SUBMIT_REFUSE"))) : 0;
    static uint64_t submits = 0;   // (dev.present_mu)
    return every && ++submits % every == 0;
}
/// What a refused submit does before it is made again (dev.present_mu held, so the queue is this thread's): it waits for the
/// GPU's work before it, so nothing in flight holds memory resident, and tells the next present (texture_memory_refused) to
/// free the idle staging arenas and the reaper's queue. Not here: under the queue's lock nothing may be destroyed (a destroy's
/// rec::boundary can wait for the recorder, whose part's submit waits for this lock), and the arenas are the replay thread's
void submit_give_back(int tried, VkResult res) {
    const auto t0 = std::chrono::steady_clock::now();
    vkQueueWaitIdle(dev.queue);
    texture_memory_refused();
    static std::atomic<int> told{0};
    if (told.fetch_add(1, std::memory_order_relaxed) < 16)
        logf("vk: a submit refused for want of memory (%d): the GPU's work before it waited for (%.1f ms), the idle staging and "
             "the reaper's queue freed at the next present; made again (try %d of %d)", (int)res,
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), tried + 1, submit_retry::kTries);
}
/// vkQueueSubmit2 of one submit (dev.present_mu held), made again after submit_give_back while the driver refuses it for want
/// of memory (submit_retry.h)
VkResult submit_or_retry(const VkSubmitInfo2& si, VkFence fence) {
    int tries = 0;
    VkResult last = VK_SUCCESS;
    const VkResult res = submit_retry::submit(
        [&] {
            const bool refuse = !tries++ && test_submit_refused();
            return last = refuse ? VK_ERROR_OUT_OF_DEVICE_MEMORY : vkQueueSubmit2(dev.queue, 1, &si, fence);
        },
        [&](int tried) { submit_give_back(tried, last); }, [] { return submit_retry_on(); });
    if (tries > 1) {
        static std::atomic<int> told{0};
        if (told.fetch_add(1, std::memory_order_relaxed) < 16)
            logf("vk: the refused submit %s on try %d", res == VK_SUCCESS ? "was taken" : "was refused again, dropped", tries);
    }
    return res;
}
// ---- (hookshot) recordings by number: a second timeline, signalled at the end of every recording with its number, and the
// recording each submission was finished in. The timeline above is signalled only as far as no older submission is stopped
// at a wait, and every NVN queue's submissions share its numbers: a fence taken on one queue waited for the other queues'
// stopped work too, and a wait for it forced that work past the waits it stopped at (OoT's presentation work run ahead of
// the scene work it waits for: a frame of the fog, haze and sun shafts of the frame before). NVN's fences are per queue: a
// wait for one is a wait for that queue's work, here for the recordings it was finished in. RX_VK_PER_QUEUE_FENCES=0: as
// before ----
VkSemaphore rec_timeline = VK_NULL_HANDLE;
uint64_t rec_seq = 0;                       // (the recordings submitted so far; replay_mu)
std::vector<uint64_t> rec_open_values;      // (the submissions finished in the recording being made; replay_mu)
std::vector<uint64_t> rec_open_qvalues;     // (those of them submitted on a queue, value_queue's; replay_mu)
std::mutex rec_map_mu;
// (Hookshot) RX_VK_FENCE_POLL_FAST=1: the submissions finished whole (not stopped at a wait) on a queue, by value
// (slot = value & 16383 holds the value), marked once value_rec has their recording: a fence poll (a wait of timeout 0) on one of
// them answers from the recordings' timeline without sync_order_mu, which submit_q and present hold for whole replays (the main
// thread's two nvnSyncWait looks a frame, f_8042cd20, waited 1.4 ms a frame for it behind the Presentation Thread's replay).
// See wait_inner and invalidate. Off: with the wait moved to the readbacks that need it, the readbacks (~12 a frame on the main
// thread, 2-12 on each module worker) waited 0.5-1.9 ms a frame each thread .
constexpr uint64_t kWholeSlots = 16384;
std::atomic<uint64_t> whole_val[kWholeSlots];
// (Hookshot) RX_VK_READBACK_TICKET=1: the fast poll above, and the readbacks' order it skipped kept by a ticket: the
// poll notes the sync_order_mu section in progress, and a CPU readback (invalidate) waits until that section has left, the order
// the poll's lock gave it, instead of the poll waiting (see invalidate). =2 (a diagnostic, not the locked path's order): only
// the readbacks of ranges with write-backs wait. Off: RX_VK_FENCE_POLL_FAST=1 alone is 's v3
int readback_ticket() {
    static const int v = getenv("RX_VK_READBACK_TICKET") ? atoi(getenv("RX_VK_READBACK_TICKET")) : 0;
    return v;
}
bool fence_poll_fast_on() {
    static const bool v = (getenv("RX_VK_FENCE_POLL_FAST") && atoi(getenv("RX_VK_FENCE_POLL_FAST")) != 0) || readback_ticket();
    return v;
}
std::map<uint64_t, uint64_t> value_rec;     // submission value (or a recording's own) -> the recording that finished it
// (Hookshot, RX_VK_SYNC_HONEST=1, default off) value_rec keeps the FIRST recording a value was put in (emplace): a submission stopped at an
// event wait is mapped to the recording of its piece before the wait, and a game fence for it was answered as soon as that one was
// done, while the rest of its work (after the wait) was still on the GPU (RX_VK_LAG_LOG: 96% of the answers the game gets, 8 ms early).
// With the switch a fence waits for the recording its last piece went into (value_end), or for the timeline while none is known.
std::map<uint64_t, uint64_t> value_end;
bool sync_honest_on() {
    static const bool v = [] {
        const bool on = getenv("RX_VK_SYNC_HONEST") && atoi(getenv("RX_VK_SYNC_HONEST")) != 0;
        if (on) logf("vk: fence waits honest (RX_VK_SYNC_HONEST): a fence waits for the recording its work ended in");
        return on;
    }();
    return v;
}
// (hookshot) A submission is recorded in parts (maybe_split), and each part writes back, at its own end, the pages the GPU
// wrote in it (memory_end_submission). A CPU read of GPU results needs the part that writes its pages back, not the
// submission's last one: part (Recording::part) -> its number on the recordings' timeline, once it is submitted
uint64_t rec_part_next = 0;                 // (the recordings begun so far; replay_mu)
std::map<uint64_t, uint64_t> part_rec;      // (rec_map_mu)
std::condition_variable rec_map_cv;         // (a part has been submitted: wait_recorded_part)
std::unordered_map<uint64_t, uint64_t> q_last;   // queue -> its last submission's value (sync_order_mu)
std::map<uint64_t, uint64_t> value_queue;   // a submission's value -> its queue (sync_order_mu)
std::atomic<uint64_t> pushed{0};
std::atomic<uint64_t> q_values{0};   // the fence values handed out (a submit or a present takes one each)
::mkwvk::rec::DrainingMutex replay_mu;   // one replay at a time (the game submits from several threads; RX_VK_RECORDER=defer: what
                                         // a thread recorded is drained before another may take it)
// (RX_VK_RECORDER=thread: the recorder thread named, and above normal priority as par_worker_main's threads)
// RX_VK_RECORDER_PIN (experiment): next = the physical core below the reserved one (the replay thread's: hos_reserved_cpus),
// sibling = the reserved core itself (shares it with the replay thread), or a hex mask; unset: not pinned
const bool rec_thread_hook = (::mkwvk::rec::g_thread_start_hook = [] {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk recorder");
#ifdef TRACY_ENABLE
    HT_THREAD_NAME("mkw_vk recorder");
#endif
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    const char* pin = getenv("RX_VK_RECORDER_PIN");
    if (!pin || !*pin) return;
    const uint64_t reserved = host && host->reserved_cpus ? host->reserved_cpus() : 0;
    uint64_t m = 0;
    if (!strcmp(pin, "next")) m = reserved ? reserved >> __builtin_popcountll(reserved) : 0;
    else if (!strcmp(pin, "sibling")) m = reserved;
    else m = strtoull(pin, nullptr, 16);
    const bool ok = m && SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)m) != 0;
    ::mkwvk::logf("vk: RX_VK_RECORDER_PIN=%s: the recorder %s %#llx (reserved %#llx)", pin, ok ? "pinned to" : "NOT pinned, mask",
                  (unsigned long long)m, (unsigned long long)reserved);
}, true);
// (RX_VK_RECORDER_IDLE=wait: the recorder's idle waits on its ring counter. Windows: from kernelbase.dll, without linking
// Synchronization.lib; Linux: host/posix's, a futex)
const bool rec_wait_hooks = [] {
#ifdef _WIN32
    using WaitFn = BOOL(WINAPI*)(volatile VOID*, PVOID, SIZE_T, DWORD);
    using WakeFn = VOID(WINAPI*)(PVOID);
    static WaitFn wait = nullptr;
    static WakeFn wake = nullptr;
    if (const HMODULE k = GetModuleHandleA("kernelbase.dll")) {
        wait = (WaitFn)(void*)GetProcAddress(k, "WaitOnAddress");
        wake = (WakeFn)(void*)GetProcAddress(k, "WakeByAddressSingle");
    }
    if (!wait || !wake) return false;
    ::mkwvk::rec::g_wait_on_address = [](volatile void* a, void* c, size_t n, uint32_t ms) { return wait(a, c, n, ms) != 0; };
    ::mkwvk::rec::g_wake_by_address = [](void* a) { wake(a); };
#else
    ::mkwvk::rec::g_wait_on_address = [](volatile void* a, void* c, size_t n, uint32_t ms) { return WaitOnAddress(a, c, n, ms) != 0; };
    ::mkwvk::rec::g_wake_by_address = [](void* a) { WakeByAddressSingle(a); };
#endif
    return true;
}();
// (RX_VK_BATCH: see queue_flush_locked)
VkSemaphore batch_tl = VK_NULL_HANDLE;
uint64_t batch_next = 0, batch_flushed = 0;
struct { std::vector<VkCommandBufferSubmitInfo> cbs; uint64_t signal = 0; bool any_signal = false; uint64_t rec = 0; uint64_t reflex = 0; } pend;   // (rec: the recordings' timeline value the held ones reach; reflex: the game frame of the first, vk_reflex.h)

int64_t st_flushes, st_batched_cbs;
enum { FR_OTHER, FR_POLL, FR_WAIT, FR_SLOT, FR_PRESENT, FR_CAP, FR_VALUE, FR_READBACK, FR_COUNT };
int64_t st_reason[FR_COUNT];
int flush_reason = FR_OTHER;
std::chrono::steady_clock::time_point pend_since;
// (hookshot) whether any submit is held (pend.cbs not empty; written under dev.present_mu only), for the flushes that come
// from the game's threads to look at without the mutex: a present holds dev.present_mu for as long as it takes (4 ms a
// frame in Hyrule Field), and a flush with nothing to send waited for it. The host's copy bookkeeping asks how far the
// replay is under its own lock (rx_nvn_replayed -> queue_flush_poll), so one job thread waiting there held up every other
// one recording a command buffer: 0.5-1.1 threads blocked at any instant
std::atomic<bool> pend_held{false};

struct Recording {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPool upload_pool = VK_NULL_HANDLE;   // (its own: a parallel replay's threads record uploads into it while the
                                                  // replay thread records cmd, and a pool is one thread's at a time)
    VkCommandBuffer cmd = VK_NULL_HANDLE, upload = VK_NULL_HANDLE;
    // (a parallel replay's segment: new images' first layout changes, submitted before its uploads. Its own pool: any
    // segment's worker records into the earliest open segment's, while that segment's worker records cmd and upload)
    VkCommandPool init_pool = VK_NULL_HANDLE;
    VkCommandBuffer init = VK_NULL_HANDLE;
    uint64_t value = 0;   // the timeline value its submit signals (reusable once reached)
    int slot = 0;         // its index: the draw path keeps a descriptor pool and stream per recording
    // (its slot comes round again once its own submit is done: a fence of its own. The timeline is signalled as far
    // as no older submission is unfinished (submit_job), and a slot waiting on its value could wait on the replay)
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    uint64_t batch = 0;   // (RX_VK_BATCH: the batch it went out in, waited for on batch_tl; 0: its own fence)
    uint64_t part = 0;    // (its number among the recordings begun: the pages it writes back are noted with it, part_rec)
};
// (24: the event waits cut a frame's command buffers into pieces, each with recordings of its own, and with 16 the
// replay waited ~1.5 ms a frame for a slot the GPU still had)
constexpr int kRecordings = 48;   // (test: 24 before; with RX_VK_BATCH the ring came round to held submits once a frame)
Recording recordings[kRecordings];
uint64_t rec_end_pos[kRecordings];   // (RX_VK_RECORDER_OWN: the recorder ring's position after each slot's last end, vk_recorder.h)
int recording_index = 0;

/// (a recording slot's last submit done: before it is used again)
void recording_wait(Recording& r) {
#ifdef TRACY_ENABLE
    HT_WAIT(ht_w, "recording_wait");
#endif
    if (!r.submitted) return;
    FpsRtWait rt_wait(fps_stats::kRtGpu);   // (the FPS counter's render lane)
    if (r.batch) {
        {
            std::lock_guard lk(dev.present_mu);
            if (batch_flushed < r.batch) flush_reason = FR_SLOT, queue_flush_locked();
        }
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &batch_tl;
        wi.pValues = &r.batch;
        vkWaitSemaphores(dev.device, &wi, UINT64_MAX);
        r.batch = 0;
        r.submitted = false;
        return;
    }
    vkWaitForFences(dev.device, 1, &r.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(dev.device, 1, &r.fence);
    r.submitted = false;
}

uint64_t completed() {
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(dev.device, timeline, &v);
    memory_staged_apply(v);   // (RX_VK_POOL_MIRROR: the write-backs staged up to v in guest memory before anyone is told v is done)
    return v;
}
// ---- (test) RX_VK_BATCH=1: the replay's submits held and sent as one vkQueueSubmit2 ----
// On amdgpu every command submission pays for all imported host memory (userptr) again; the replay made ~6 a frame.
// Held submits go out at the next point that needs the GPU's results or the queue's order: a wait, a readback, a
// present, a sparse bind, another submit (queue_flush_locked under dev.present_mu).
// (hookshot) on by default with Mesa's RADV, where it was measured (issue 413: a Radeon 8060S on Bazzite, ~10 -> 30-50 fps, the
// replay's queue submit ~70 -> ~6 ms a frame); elsewhere as before. RX_VK_BATCH=0 / 1: off / on on any driver
/// (hookshot, #1993 by mrdev023) The GPU has done every batch sent so far: the batch being held goes out at once. A held
/// batch waited for the next wait or present while the GPU sat idle, and with serial replay the game waits for it mid-frame
/// (take_capture's late pages), so the GPU started cold just as it was waited for. RX 9070 XT, RADV, Kokiri Forest: 33-41
/// fps -> 46-61, no flashes in 5230 frames; ~10 flushes a frame of ~3.7 command buffers each, so batching still groups the
/// submits while the GPU is busy. RX_VK_BATCH_IDLE=0: held until a wait, a present or the cap, as before.
/// (Not by default on the Steam Deck's APU (Van Gogh: 0x163f LCD, 0x1435 OLED), where each submit costs ~1 ms: Deck LCD,
/// Kokiri Forest, main with default switches: queue submit 11.7 ms a frame with it, 2.0 without, replay 40.5 -> 31.4 ms; the
/// fps comes with the readback send below. Other APUs (780M, 890M, 8060S) are unmeasured and keep it. RX_VK_BATCH_IDLE=1: on)
/// (hookshot) While a CPU read of GPU results waits (invalidate, after #2125's catch-up), what is held goes out when the wait
/// begins and then each time RX_VK_READBACK_SEND_CBS command buffers are held (default 8): the GPU works through the catch-up's
/// work meanwhile instead of starting it only at the wait's own flush. Without the idle send (the Steam Deck's APU) that wait
/// doubled: Deck LCD, Kokiri, main: 13.8 / 13.5 fps, no idle send 12.8 / 12.8, with this 14.6 / 14.3. RX_VK_READBACK_SEND=0: off
std::atomic<int> readback_waiting{0};
bool readback_send_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_READBACK_SEND");
        const bool v = !e || atoi(e) != 0;
        logf("vk: held submits go out while a readback waits: %s (RX_VK_READBACK_SEND)", v ? "on" : "off");
        return v;
    }();
    return on;
}
bool readback_send_now() {   // (dev.present_mu held)
    static const size_t cbs = getenv("RX_VK_READBACK_SEND_CBS") ? (size_t)std::max(1, atoi(getenv("RX_VK_READBACK_SEND_CBS"))) : 8;
    return readback_waiting.load(std::memory_order_relaxed) && pend.cbs.size() >= cbs;   // (by the work held: the same split on a slower CPU)
}
bool batch_gpu_idle() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_BATCH_IDLE");
        const bool deck = dev.props.vendorID == 0x1002 && (dev.props.deviceID == 0x163f || dev.props.deviceID == 0x1435);
        const bool v = e ? atoi(e) != 0 : !deck;
        logf("vk: idle send %s (RX_VK_BATCH_IDLE%s)", v ? "on" : "off", !e && deck ? ", off on the Steam Deck" : "");
        return v;
    }();
    uint64_t done = 0;
    return vkGetSemaphoreCounterValue(dev.device, batch_tl, &done) == VK_SUCCESS && done >= batch_flushed;
}
bool batch_on() {
    static const bool on = [] {
        // (device_policy.h batch_submits: RADV, and (k10) not on GCN 1-4, where an RX 580 ran Kokiri Forest at 24 fps without
        // and 20 with; then the switch)
        const bool v = dev.policy.batch_submits;
        logf("vk: batched submits %s (RX_VK_BATCH%s)", v ? "on" : "off",
             !dev.policy.set(Sw::batch_submits) && dev.facts.radv() ? dev.facts.radv_gcn() ? ", off on GCN" : ", on with RADV" : "");
        return v;
    }();
    return on && batch_tl != VK_NULL_HANDLE;
}
// RX_VK_READBACK_STRICT: a readback of results with write-backs waits for the write-back's own value, not only the
// recordings' timeline. On RADV that timeline is signalled before the write-back at the end of the submission has landed,
// and a read in the gap took the light meter's sums from an earlier submission: OoT's auto-exposure jumped, the whole picture
// 8-15% brighter for ~16 frames (an RX 9070 XT in Kokiri: up to 99 pulses in 2 minutes with the CPU busy, 0 with the wait,
// at no measurable cost). On NVIDIA's driver the pulses were not seen and the wait cost ~18% (an RTX 5070 in Kokiri, pull
// request 6). So on by default with RADV only (for the meter's sizes, below); RX_VK_READBACK_STRICT=0 / 1: off / all on any driver.
// RX_VK_READBACK_STRICT=<hex size>[,<hex size>...] (any value but 0 and 1): only the ranges of exactly those sizes wait, the
// others as with 0. The light meter's are 1040 and c0 (stream R2: 0x1040 x3 and 0xc0 a frame, from the ring helper f_80357060);
// a stale meter is the pulse, the other rings (0x2000, 0x40) read a frame late gave a right picture (RX_VK_READBACK_LATE).
// Unset on RADV the default is that list, not all: 1040,c0 stopped the pulses as 1 did (stream R2), and under 1 nearly all the
// strict wait was on two 0x1b580 ranges of the slot scan f_80364080 (RX 9070 XT, Kokiri: 80.8 fps with 1040,c0, 69.1 with 1)
bool readback_strict_wanted(uint64_t size) {
    struct Strict { bool all = false; readback_late::Rules sizes; };
    static const Strict s = [] {
        bool radv = false;
        VkPhysicalDeviceDriverProperties dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &dp;
        if (dev.physical) vkGetPhysicalDeviceProperties2(dev.physical, &p2), radv = dp.driverID == VK_DRIVER_ID_MESA_RADV;
        const char* e = getenv("RX_VK_READBACK_STRICT");
        Strict r;
        if (!e && radv && false) {   // (the fork's RADV default, off here: RX_VK_READBACK_STRICT=1040,c0 asks for it)
            r.sizes = readback_late::parse("1040,c0");
            logf("vk: readback waits for the write-back for the light meter's sizes 1040,c0 (RX_VK_READBACK_STRICT unset, RADV)");
        } else if (e && strcmp(e, "0") && strcmp(e, "1")) {
            r.sizes = readback_late::parse(e);
            r.sizes.newest = false;   // (sizes only)
            logf("vk: readback waits for the write-back for %d sizes (RX_VK_READBACK_STRICT=%s)", (int)r.sizes.sizes.size(), e);
        } else {
            r.all = e && atoi(e) != 0;
            logf("vk: readback waits for the write-back %s (RX_VK_READBACK_STRICT%s)", r.all ? "on" : "off", e ? "" : " unset");
        }
        return r;
    }();
    return s.all || readback_late::wanted(s.sizes, size, false);
}
// (RX_VK_LATE_DEFER) the highest timeline value a submit sent to the queue signals: a capture's late value at or below it
// is reached with nothing more submitted, so its wait can move to the queue (vk_memory.cpp take_capture)
std::atomic<uint64_t> tl_signal_sent{0};
void note_signal_sent(uint64_t v) {
    uint64_t cur = tl_signal_sent.load(std::memory_order_relaxed);
    while (v > cur && !tl_signal_sent.compare_exchange_weak(cur, v, std::memory_order_release, std::memory_order_relaxed)) {}
}
int64_t st_lf_thread, st_lf_blocked, st_lf_held_cbs;   // (dev.present_mu: fixes applied by the fix thread, by a flush that waited)
double st_lf_blocked_ms;
std::atomic<int64_t> st_lf_wait_ns{0};   // (the frame trace's late_fix_ms: any thread's waits for a pending fix, the fix thread's not)
} // namespace
void vk_tl_note(const char* what, uint64_t a, uint64_t b) { tl_ev(what, a, b); }
uint64_t vk_timeline_signal_sent() { return tl_signal_sent.load(std::memory_order_acquire); }
void vk_timeline_wait_sent(uint64_t v) {
    if (!timeline || completed() >= v) return;
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &v;
    for (int i = 0; i < 8 && vkWaitSemaphores(dev.device, &wi, 250000000ull) == VK_TIMEOUT; i++) {}
    (void)completed();   // (RX_VK_POOL_MIRROR: its staged write-backs applied)
}
// Before taking the queue's lock to send something: a pending late fix's value waited for here, outside the lock, so the
// flush under it does not wait there and hold up the threads that only hold their recordings back meanwhile (the Presentation
// Thread's submits; a poll of the main thread's waited 7.5 ms under it in the first try). A poll does not wait at all
thread_local bool tl_poll = false;
void queue_flush_locked() {   // (dev.present_mu held)
    if (pend.cbs.empty()) return;
    VkSemaphoreSubmitInfo sig[3]{{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO},
                                 {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}};
    uint32_t n = 0;
    if (pend.any_signal) {
        sig[n].semaphore = timeline, sig[n].value = pend.signal, sig[n].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        n++;
    }
    if (pend.rec) {   // (the recordings' timeline too: wait_recorded waits on it, and held submits carried no signal of it)
        sig[n].semaphore = rec_timeline, sig[n].value = pend.rec, sig[n].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        n++;
    }
    sig[n].semaphore = batch_tl, sig[n].value = batch_next, sig[n].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    n++;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = (uint32_t)pend.cbs.size();
    si.pCommandBufferInfos = pend.cbs.data();
    si.signalSemaphoreInfoCount = n;
    si.pSignalSemaphoreInfos = sig;
    PROF(PROF_QSUBMIT);
    si.pNext = reflex_submit_id(nullptr, pend.reflex);   // (NVIDIA Reflex: the game frame of the batch's first command buffer)
    VkSemaphoreSubmitInfo sparse_wait;   // (hookshot: after the sparse mirrors' binds, RX_VK_MIRROR_SPARSE)
    if (memory_sparse_wait(&sparse_wait)) si.waitSemaphoreInfoCount = 1, si.pWaitSemaphoreInfos = &sparse_wait;
    const VkResult res = submit_or_retry(si, VK_NULL_HANDLE);
    static int failures;
    if (res != VK_SUCCESS && failures++ < 8) logf("vk: batched submit %llu failed (%d)", (unsigned long long)batch_next, (int)res);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    if (res == VK_SUCCESS && pend.any_signal) note_timeline_sent(pend.signal);
    batch_flushed = batch_next;
    st_flushes++, st_batched_cbs += (int64_t)pend.cbs.size();
    st_reason[flush_reason]++;
    flush_reason = FR_OTHER;
    pend.cbs.clear(), pend.any_signal = false, pend.signal = 0, pend.rec = 0, pend.reflex = 0;
    pend_held.store(false, std::memory_order_release);
}
void queue_flush() {
#ifdef TRACY_ENABLE
    HT_SCOPE("queue_flush");
#endif
    std::lock_guard lk(dev.present_mu);
    queue_flush_locked();
}
// (RX_VK_LATE_DEFER) The fix thread: waits for a pending late fix's value, then applies the fixes reached and sends
// the recordings held behind them, so the GPU has the new frame's work as soon as it is done with the old one (a held recording
// otherwise went out at the next submit that could not be held: a parallel segment's, or a present)
namespace {
std::mutex lf_mu;
std::condition_variable lf_cv;
uint64_t lf_want = 0;   // (lf_mu: the highest value a fix was noted for; 0: nothing to wait for)
} // namespace
void device_wait_idle_locked() {   // (dev.present_mu held)
    queue_flush_locked();
    std::lock_guard pq(dev.present_queue_mu);
    std::lock_guard pr(dev.presenter_queue_mu);   // (the presenter thread's queue too: every queue is idle-waited)
    vkDeviceWaitIdle(dev.device);
}
namespace {
void queue_flush_for(int why) {
    if (!pend_held.load(std::memory_order_acquire)) return;   // (nothing held: nothing to send, the mutex not taken)
    std::lock_guard lk(dev.present_mu);
    if (pend.cbs.empty()) return;
    flush_reason = why;
    queue_flush_locked();
}
// (RX_VK_BATCH_POLL_US=<us>: a progress poll sends held submits only once the oldest has waited that long; 0: always.
// Default 2000: the game polls many times a frame, and each poll that flushed was a whole command submission, the cost
// batching is there to avoid. RX 7800 XT, RADV, Kokiri: 525 -> 67 poll flushes per 120 presents, 38 -> 42.5 fps; issues
// 413, 528)
void queue_flush_poll() {
    static const int64_t us = getenv("RX_VK_BATCH_POLL_US") ? atoll(getenv("RX_VK_BATCH_POLL_US")) : 2000;
    if (!pend_held.load(std::memory_order_acquire)) return;   // (nothing held: nothing to send, the mutex not taken)
    std::lock_guard lk(dev.present_mu);
    if (pend.cbs.empty()) return;
    if (const uint64_t lv = memory_late_fix_pending(); lv && completed() < lv) return;   // (the fix thread sends them)
    if (us > 0 && std::chrono::steady_clock::now() - pend_since < std::chrono::microseconds(us)) return;
    flush_reason = FR_POLL;
    queue_flush_locked();
}
} // namespace
namespace {
void wait_value(uint64_t v) {
    queue_flush_for(FR_VALUE);
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &v;
    vkWaitSemaphores(dev.device, &wi, UINT64_MAX);
    (void)completed();   // (RX_VK_POOL_MIRROR: its staged write-backs applied)
}
bool wait_recorded(uint64_t f, uint64_t timeout_ns, int* ok);
/// (hookshot) What the calling thread's last wait on the recordings' timeline came back with (wait_rec_seq): one that came back
/// VK_ERROR_DEVICE_LOST is said there, and not again by vk_timeline_wait as a capture's wait that timed out
thread_local VkResult wait_rec_result = VK_SUCCESS;
/// (hookshot) A timeline wait that came back VK_ERROR_DEVICE_LOST, said as a lost device (8 times at most). Run 21 (macOS,
/// MoltenVK 1.4.2) logged one as a capture's wait that "timed out (done 0)": the driver marks the device lost, signals every
/// timeline waited on, returns VK_ERROR_DEVICE_LOST from the wait and leaves the counter's value unwritten, so completed() read
/// 0, and the log showed a timeout where the GPU was gone. Only the line: the next submit or the GPU hang watch ends the game,
/// as before
void wait_lost(const char* what, uint64_t v) {
    static std::atomic<int> logged{0};
    if (logged++ >= 8) return;
    logf("vk: %s %llu came back VK_ERROR_DEVICE_LOST: the GPU device is lost", what, (unsigned long long)v);
}
} // namespace
uint64_t vk_timeline_completed() { return timeline ? completed() : 0; }
int64_t vk_value_submitted_ns(uint64_t v);
void vk_timeline_wait(uint64_t v) {   // (at most 250 ms: a submission held back behind the caller's thread goes on without it)
    if (!timeline || completed() >= v) return;
    // (a burst recording whose write-back stamped wb_pages with burst_value does not advance `timeline` while an older queue
    // submission is stopped at an event wait; wait on `rec_timeline` where that recording actually signalled completion)
    int ok = 0;
    if (wait_recorded(v, 250000000ull, &ok)) {
        const bool lost = wait_rec_result == VK_ERROR_DEVICE_LOST;   // (said by wait_rec_seq)
        if (lost) return;
        if (!ok) {
            static std::atomic<int> logged{0};
            if (logged++ < 8)
                logf("vk: a capture's wait for write-back %llu timed out (done %llu)", (unsigned long long)v,
                     (unsigned long long)completed());
        }
        return;
    }
    queue_flush_for(FR_VALUE);
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &v;
    const uint64_t done0 = completed();
    const int64_t t0 = std::chrono::steady_clock::now().time_since_epoch().count(), sub = vk_value_submitted_ns(v);
    const VkResult res = vkWaitSemaphores(dev.device, &wi, 250000000ull);
    if (res == VK_TIMEOUT) {
        static std::atomic<int> logged{0};
        if (logged++ < 8) logf("vk: a capture's wait for write-back %llu timed out (done %llu)", (unsigned long long)v, (unsigned long long)completed());
    }
    (void)completed();   // (RX_VK_POOL_MIRROR: its staged write-backs applied)
    {   // what the capture waits for: how long, how far ahead of the GPU, and how long the GPU had it by then
        static std::mutex mu;
        static int64_t n, unsub, wait_ns, max_ns, gpu_had_ns, ahead, n_traced;
        static const bool trace = env_on("RX_VK_CAPWAIT_TRACE");   // (diagnostic: each wait, 400 of them after 3000)
        const int64_t t1 = std::chrono::steady_clock::now().time_since_epoch().count();
        const uint64_t q = q_values.load();
        std::lock_guard lk(mu);
        n++, wait_ns += t1 - t0, max_ns = std::max(max_ns, t1 - t0), ahead += (int64_t)(v - done0);
        if (trace && n > 3000 && n_traced < 400)
            n_traced++, logf("vk: capture wait for %llu: done %llu, handed out %llu; %.2f ms, submitted %.2f ms before it",
                             (unsigned long long)v, (unsigned long long)done0, (unsigned long long)q, (t1 - t0) / 1e6,
                             sub ? (t0 - sub) / 1e6 : -1.0);
        if (sub) gpu_had_ns += t1 - sub; else unsub++;
        if (n % 600 == 0) {
            logf("vk: capture waits: %lld, %.2f ms each (max %.2f), %.1f values ahead of the GPU, %lld not yet submitted, the GPU "
                 "had the value %.2f ms when it came; v - handed out now %lld", (long long)n, wait_ns / 600 / 1e6, max_ns / 1e6,
                 ahead / 600.0, (long long)unsub, gpu_had_ns / std::max<int64_t>(1, 600 - unsub) / 1e6, (long long)(v - q));
            unsub = wait_ns = max_ns = gpu_had_ns = ahead = 0;
        }
    }
    const bool lost = res == VK_ERROR_DEVICE_LOST;
    if (lost) wait_lost("a capture's wait for write-back", v);
}
namespace {

int64_t st_wait_ns;   // (time the replay waited for a recording slot: the GPU behind)

// GPU time: a timestamp at the start and the end of each recording's command buffer, read when its slot comes
// round again (the work done by then)
VkQueryPool ts_pool = VK_NULL_HANDLE;
bool ts_pending[kRecordings], ts_up_pending[kRecordings];
int64_t st_gpu_ns, st_gpu_max_ns;
// the upload command buffer's time too, and the gaps on the queue between one recording's command buffers and the
// next's (the GPU idle, or held in something of the driver's: a semaphore, paging): absolute timestamps, 4 a slot (the
// upload's start and end, the recording's). The frame trace has them as gpu_up_ms and gpu_gap_ms
int64_t st_gpu_up_ns, st_gpu_gap_ns, st_gpu_gap_max_ns;
uint64_t ts_last_end;   // (the last collected command buffer's end, GPU ticks)
// the recordings' starts and ends by submission order, and the blits' places among them (blit_ts_mark at each
// present's submit): ts_last_end_seen gives, for the oldest blit not yet matched, the last recording end before it and the
// first start after it
struct TsRec { uint64_t seq, start, end, value = 0; };
std::mutex ts_mu;
std::deque<TsRec> ts_recs;
// (stream R2, RX_VK_TL) each collected recording's timestamps in GPU ticks (the upload's start and end, the command buffer's), with its
// number on the recordings' timeline: tl_gpu_dump puts them on the host clock beside the timeline's events
struct TlGpuRec { uint64_t recseq, value, us, ue, cs, ce; };
std::deque<TlGpuRec> tl_gpu_recs;   // (ts_mu)
uint64_t ts_recseq[kRecordings];    // (the recordings' timeline value each slot's last submit signals)
uint64_t tl_rec_lo = 0, tl_rec_hi = 0;
// the GPU's busy tail before each late-page wait ended: the timeline value the capture waited for (gpu_tail_note_late),
// and once its recordings are collected, the GPU time it ran without a gap up to that value's end. GPU work cut inside that tail
// shortens the wait (the frame on PC 3050 is serial: the CPU waits there); work cut before a gap does not. Logged every 120
// presents with the 'GPU time' line; with RX_VK_PASS_TIMING its spans by label (gprof_tail)
uint64_t ts_sig[kRecordings];   // (the value each slot's submission signals)
std::deque<uint64_t> late_values;
int64_t st_tail_ns, st_tail_n, st_tail_recs;
void gpu_tail_compute(uint64_t v, double period) {   // (ts_mu held)
    std::vector<TsRec> s(ts_recs.begin(), ts_recs.end());
    std::sort(s.begin(), s.end(), [](const TsRec& a, const TsRec& b) { return a.start < b.start; });
    int at = -1;
    uint64_t end = 0;
    for (int i = 0; i < (int)s.size(); i++)
        if (s[i].value == v && s[i].end >= end) end = s[i].end, at = i;
    if (at < 0) return;
    const uint64_t thr = (uint64_t)(20000.0 / std::max(period, 1e-3));   // (a gap of 20 us or more: the GPU waited for the CPU)
    uint64_t lo = s[at].start;
    int n = 1;
    for (int j = at - 1; j >= 0; j--) {
        if (s[j].end + thr < lo) break;
        lo = std::min(lo, s[j].start), n++;
    }
    if (end <= lo) return;
    st_tail_ns += (int64_t)((end - lo) * period), st_tail_n++, st_tail_recs += n;
    if (gprof_on()) gprof_tail(lo, end);
}
std::deque<uint64_t> ts_blit_seqs;   // (the submission count at each blit)
uint64_t ts_seq_submitted, ts_seq_collected;
// (frame overlap, below with the render thread: what each recording did, and the GPU's time between recordings)
void fo_recording_begin(const Recording& r);
void fo_recording_end(const Recording& r);
void fo_collect(const Recording& r, const uint64_t* t);
void gpu_time_collect(Recording& r) {
    if (!ts_pool || !ts_pending[r.slot]) return fo_collect(r, nullptr);
    ts_pending[r.slot] = false;
    const bool up = ts_up_pending[r.slot];
    ts_up_pending[r.slot] = false;
    uint64_t t[4] = {0, 0, 0, 0};
    if (vkGetQueryPoolResults(dev.device, ts_pool, 4 * r.slot + 2, 2, sizeof t[0] * 2, t + 2, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return fo_collect(r, nullptr);
    if (up && vkGetQueryPoolResults(dev.device, ts_pool, 4 * r.slot, 2, sizeof t[0] * 2, t, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        t[0] = t[1] = 0;
    if (t[3] <= t[2]) return fo_collect(r, nullptr);
    const double period = (double)dev.props.limits.timestampPeriod;
    const int64_t ns = (int64_t)((t[3] - t[2]) * period);
    st_gpu_ns += ns;
    st_gpu_max_ns = std::max(st_gpu_max_ns, ns);
    fo_collect(r, t + 2);   // (frame overlap: the recording's command buffer, t[2] and t[3] here)
    if (t[1] > t[0]) st_gpu_up_ns += (int64_t)((t[1] - t[0]) * period);
    const uint64_t start = t[0] && t[0] < t[2] ? t[0] : t[2], end = std::max(t[1], t[3]);
    if (ts_last_end && start > ts_last_end && (start - ts_last_end) * period < 2e9) {   // (in order on the one queue; a gap)
        const int64_t gap = (int64_t)((start - ts_last_end) * period);
        st_gpu_gap_ns += gap;
        st_gpu_gap_max_ns = std::max(st_gpu_gap_max_ns, gap);
    }
    if (end > ts_last_end) ts_last_end = end;
    std::lock_guard lk(ts_mu);
    if (dev.calibrated_ts) {   // (RX_VK_TL: the recording's own times, for the timeline's GPU side)
        tl_gpu_recs.push_back({ts_recseq[r.slot], ts_sig[r.slot], up && t[1] > t[0] ? t[0] : 0, up && t[1] > t[0] ? t[1] : 0, t[2], t[3]});
        while (tl_gpu_recs.size() > 512) tl_gpu_recs.pop_front();
    }
    ts_recs.push_back({++ts_seq_collected, start, end, ts_sig[r.slot]});
    while (ts_recs.size() > 256) ts_recs.pop_front();
    while (!late_values.empty() && late_values.front() < ts_sig[r.slot]) {   // (its recordings collected: the tail before it)
        gpu_tail_compute(late_values.front(), period);
        late_values.pop_front();
    }
}
} // namespace
void gpu_tail_note_late(uint64_t v) {
    std::lock_guard lk(ts_mu);
    if (!late_values.empty() && late_values.back() >= v) return;
    late_values.push_back(v);
    while (late_values.size() > 64) late_values.pop_front();
}
namespace {
void blit_ts_mark() {
    std::lock_guard lk(ts_mu);
    ts_blit_seqs.push_back(ts_seq_submitted);
    while (ts_blit_seqs.size() > 8) ts_blit_seqs.pop_front();
}
uint64_t ts_last_end_seen(uint64_t* after_start) {
    std::lock_guard lk(ts_mu);
    *after_start = 0;
    if (ts_blit_seqs.empty()) return 0;
    const uint64_t at = ts_blit_seqs.front();
    ts_blit_seqs.pop_front();
    uint64_t before = 0;
    for (const TsRec& r : ts_recs) {
        if (r.seq <= at) before = std::max(before, r.end);
        else if (!*after_start) *after_start = r.start;
    }
    return before;
}
// ---- (stream R2, RX_VK_TL_PROBE=1, with RX_VK_TL / RX_VK_TL_S) is the GPU's late start of a piece the queue's or the piece's? Before each
// recording's queue submit (and, with RX_VK_PRESENT_QUEUE=1, right after it on the presents' own compute queue) a one-timestamp command
// buffer is submitted; where each one's timestamp falls against the piece's own start says whether a new submit reaches the GPU at once
// while the piece waits (the piece's own content or dependencies) or not (the queue, the driver or the OS). Logging only ----
constexpr uint32_t kProbes = 512;
struct ProbeRec { uint64_t seq; uint32_t slot; int kind; int64_t host_ns; };   // (kind: 0 the replay's queue, 1 the presents' queue, 2 a second queue of the replay's family, 3 the replay's queue from the monitor thread)
struct {
    bool tried = false, ok = false;
    VkQueryPool qp = VK_NULL_HANDLE;
    VkCommandPool pool[4] = {};
    VkCommandBuffer cb[4][kProbes];
    std::mutex mu;   // (probe.n, recs and the buffers of kind 3, which the monitor thread submits)
    int job_every = 32;   // (RX_VK_PROBE_JOB_EVERY: monitor iterations of 0.2 ms between jobs)
    int job_fills = 0;                // (RX_VK_PROBE_JOB=<n>: a synthetic job of n fills of a 256 MB buffer, ~0.4 ms each, every 6.4 ms)
    VkBuffer job_buf = VK_NULL_HANDLE;
    VmaAllocation job_alloc = nullptr;
    VkCommandBuffer job_cb[64];
    int64_t job_host[64];
    uint32_t job_n = 0;
    uint32_t n = 0;
    std::deque<ProbeRec> recs;
} probe;
bool probe_wanted() {
    static const bool v = getenv("RX_VK_TL_PROBE") && atoi(getenv("RX_VK_TL_PROBE")) != 0;
    return v && dev.calibrated_ts;
}
void probe_init() {
    probe.tried = true;
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 4 * kProbes + 128;
    if (vkCreateQueryPool(dev.device, &qi, nullptr, &probe.qp) != VK_SUCCESS) return;
    for (int k = 0; k < 4; k++) {
        if ((k == 1 && !dev.present_queue) || (k == 2 && !dev.probe_queue)) continue;
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.queueFamilyIndex = k == 1 ? dev.present_family : dev.queue_family;
        if (vkCreateCommandPool(dev.device, &ci, nullptr, &probe.pool[k]) != VK_SUCCESS) return;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = probe.pool[k];
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = kProbes;
        if (vkAllocateCommandBuffers(dev.device, &ai, probe.cb[k]) != VK_SUCCESS) return;
        for (uint32_t i = 0; i < kProbes; i++) {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(probe.cb[k][i], &bi);
            vkCmdWriteTimestamp2(probe.cb[k][i], VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, probe.qp, k * kProbes + i);
            vkEndCommandBuffer(probe.cb[k][i]);
        }
    }
    if (getenv("RX_VK_PROBE_JOB")) {
        probe.job_fills = std::max(1, atoi(getenv("RX_VK_PROBE_JOB")));
        if (getenv("RX_VK_PROBE_JOB_EVERY")) probe.job_every = std::max(1, atoi(getenv("RX_VK_PROBE_JOB_EVERY")));
        if (!create_device_buffer(256ull << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT, probe.job_buf, probe.job_alloc)) probe.job_fills = 0;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = probe.pool[0];
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 64;
        if (probe.job_fills && vkAllocateCommandBuffers(dev.device, &ai, probe.job_cb) != VK_SUCCESS) probe.job_fills = 0;
        for (uint32_t i = 0; probe.job_fills && i < 64; i++) {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(probe.job_cb[i], &bi);
            vkCmdWriteTimestamp2(probe.job_cb[i], VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, probe.qp, 4 * kProbes + 2 * i);
            for (int k = 0; k < probe.job_fills; k++) vkCmdFillBuffer(probe.job_cb[i], probe.job_buf, 0, VK_WHOLE_SIZE, 0x01010101u * (k + 1));
            vkCmdWriteTimestamp2(probe.job_cb[i], VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, probe.qp, 4 * kProbes + 2 * i + 1);
            vkEndCommandBuffer(probe.job_cb[i]);
        }
    }
    probe.ok = true;
}
/// one probe on the replay's queue (dev.present_mu held) or on the presents' queue; seq: the recording it belongs to
void probe_monitor_start();
void probe_job_submit();
void probe_submit(uint64_t seq, int kind) {
    if (!tl_active.load(std::memory_order_relaxed) || !probe_wanted()) return;
    if (!probe.tried) probe_init();
    if (kind == 0) {
        static const bool mon = getenv("RX_VK_TL_PROBE") && atoi(getenv("RX_VK_TL_PROBE")) >= 3;
        if (mon) probe_monitor_start();
    }
    if (!probe.ok || (kind == 1 && !dev.present_queue) || (kind == 2 && !dev.probe_queue)) return;
    std::lock_guard pl(probe.mu);
    const uint32_t slot = probe.n++ % kProbes;
    vkResetQueryPool(dev.device, probe.qp, kind * kProbes + slot, 1);
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = probe.cb[kind][slot];
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    probe.recs.push_back({seq, slot, kind, tl_now()});
    while (probe.recs.size() > 8 * kProbes) probe.recs.pop_front();
    if (kind == 1) {
        std::lock_guard pq(dev.present_queue_mu);
        vkQueueSubmit2(dev.present_queue, 1, &si, VK_NULL_HANDLE);
    } else {
        vkQueueSubmit2(kind == 2 ? dev.probe_queue : dev.queue, 1, &si, VK_NULL_HANDLE);
    }
}
/// RX_VK_TL_PROBE=3: a probe on the replay's queue every ~0.2 ms while the window is open, whatever the replay is doing: where in the frame
/// does a new submit wait, and until when
void probe_job_submit() {   // (dev.present_mu held by the monitor)
    std::lock_guard pl(probe.mu);
    const uint32_t i = probe.job_n++ % 64;
    vkResetQueryPool(dev.device, probe.qp, 4 * kProbes + 2 * i, 2);
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = probe.job_cb[i];
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    probe.job_host[i] = tl_now();
    tl_ev("job-sub", i);
    const VkQueue replay_queue = dev.queue;   // (a probe, not a replay submit: nothing held for a late fix is involved)
    vkQueueSubmit2(replay_queue, 1, &si, VK_NULL_HANDLE);
}
void probe_monitor_start() {
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            SetThreadDescription(GetCurrentThread(), L"mkw_vk probe monitor");
            for (uint64_t n = 1;; n++) {
                if (!tl_active.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                {
                    std::unique_lock lk(dev.present_mu, std::try_to_lock);
                    if (lk.owns_lock()) {
                        tl_ev("probe-sub", n);
                        probe_submit(n | (1ull << 40), 3);
                        if (probe.job_fills && n % (uint64_t)probe.job_every == 0) probe_job_submit();
                    }
                }
                const int64_t until = tl_now() + 200000;
                while (tl_now() < until) std::this_thread::yield();
            }
        }).detach();
    });
}
/// the probes' times on the host clock, as timeline lines (tl_gpu_dump's own mapping: host_ms)
template <class F> void probe_dump(F host_ms, uint64_t lo, uint64_t hi) {
    if (!probe.ok) return;
    std::lock_guard pl(probe.mu);
    for (uint32_t i = 0; probe.job_fills && i < 64; i++) {
        uint64_t t[2] = {0, 0};
        if (vkGetQueryPoolResults(dev.device, probe.qp, 4 * kProbes + 2 * i, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS || !t[0] || !t[1]) continue;
        logf("vk: tl %9.3f GPU                    %-14s %#llx %#llx %#llx", host_ms(t[0]), "gpu-job+", (unsigned long long)i, 0ull, 0ull);
        logf("vk: tl %9.3f GPU                    %-14s %#llx %#llx %#llx", host_ms(t[1]), "gpu-job-", (unsigned long long)i, 0ull, 0ull);
    }
    for (const ProbeRec& p : probe.recs) {
        if (p.kind != 3 && (p.seq + 8 < lo || p.seq > hi)) continue;
        uint64_t t = 0;
        if (vkGetQueryPoolResults(dev.device, probe.qp, p.kind * kProbes + p.slot, 1, sizeof t, &t, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS || !t) continue;
        logf("vk: tl %9.3f GPU                    %-14s %#llx %#llx %#llx", host_ms(t), p.kind == 1 ? "gpu-probe2" : p.kind == 2 ? "gpu-probe3" : p.kind == 3 ? "gpu-probe4" : "gpu-probe1", (unsigned long long)p.seq,
             (unsigned long long)p.slot, 0ull);
    }
}

}   // namespace
/// (the FPS counter's Breakdown, vk_fps.cpp) the GPU's time in the timed recordings so far: read when their slots come round
int64_t gpu_time_ns() { return st_gpu_ns; }
/// (and the replay's waits for a recording slot so far: the GPU behind)
int64_t slot_wait_ns() { return st_wait_ns; }
namespace {
void gpu_time_begin(Recording& r) {
    fo_recording_begin(r);   // (at every recording's start: begin_recording)
    if (!ts_pool) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 4 * kRecordings;
        if (vkCreateQueryPool(dev.device, &qi, nullptr, &ts_pool) != VK_SUCCESS) ts_pool = VK_NULL_HANDLE;
        if (!ts_pool) return;
    }
    ts_sig[r.slot] = r.value;   // (a segment's, which signals nothing; submit_recording sets the value signalled)
    vkCmdResetQueryPool(r.upload, ts_pool, 4 * r.slot, 2);   // (each command buffer resets its own: the upload one is not always sent)
    vkCmdWriteTimestamp2(r.upload, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ts_pool, 4 * r.slot);
    vkCmdResetQueryPool(r.cmd, ts_pool, 4 * r.slot + 2, 2);
    vkCmdWriteTimestamp2(r.cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ts_pool, 4 * r.slot + 2);
}
void tl_gpu_open() { tl_rec_lo = rec_seq; }
void tl_gpu_close() { tl_rec_hi = rec_seq; }
/// (stream R2, RX_VK_TL) the GPU's own side of the window, ~11 presents after it closed (a recording's times are read when its slot comes
/// round): per recording of the window, its upload's and its command buffer's start and end on the host's clock (the same as the timeline's
/// events), as "gpu-up" / "gpu-cmd" lines of the form tlshow.py reads (what, recording number, value, duration in us)
void tl_gpu_dump(int64_t t0_ns) {
    static PFN_vkGetCalibratedTimestampsEXT get = (PFN_vkGetCalibratedTimestampsEXT)vkGetDeviceProcAddr(dev.device, "vkGetCalibratedTimestampsEXT");
    if (!dev.calibrated_ts || !get) return;
    VkCalibratedTimestampInfoEXT info[2]{{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_DEVICE_EXT},
                                         {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, tl_host_time_domain()}};   // (QPC; Linux: CLOCK_MONOTONIC, vk_tl_clock.h)
    uint64_t stamp[2] = {0, 0}, dev_dev = 0;
    LARGE_INTEGER freq, qpc;
    QueryPerformanceFrequency(&freq);
    const int64_t steady0 = tl_now();
    if (get(dev.device, 2, info, stamp, &dev_dev) != VK_SUCCESS) return;
    QueryPerformanceCounter(&qpc);
    const int64_t steady1 = tl_now();
    // (the host stamp was taken inside the call: its steady-clock time is between the two reads, taken as their middle)
    const double qpc_ns_now = (double)qpc.QuadPart * 1e9 / (double)freq.QuadPart, stamp_ns = (double)stamp[1] * 1e9 / (double)freq.QuadPart;
    const double host_at_stamp = (double)(steady0 + steady1) * 0.5 - (qpc_ns_now - stamp_ns);
    const double period = (double)dev.props.limits.timestampPeriod;
    auto host_ms = [&](uint64_t ticks) { return (host_at_stamp + ((double)ticks - (double)stamp[0]) * period - (double)t0_ns) / 1e6; };
    std::vector<TlGpuRec> v;
    {
        std::lock_guard lk(ts_mu);
        for (const TlGpuRec& g : tl_gpu_recs)
            if (g.recseq + 8 >= tl_rec_lo && g.recseq <= tl_rec_hi) v.push_back(g);
    }
    logf("vk: tl gpu begin (%zu recordings %llu..%llu)", v.size(), (unsigned long long)tl_rec_lo, (unsigned long long)tl_rec_hi);
    for (const TlGpuRec& g : v) {
        if (g.ue > g.us && g.us) {
            logf("vk: tl %9.3f GPU                    gpu-up+        %#llx %#llx %#llx", host_ms(g.us), (unsigned long long)g.recseq, (unsigned long long)g.value, 0ull);
            logf("vk: tl %9.3f GPU                    gpu-up-        %#llx %#llx %#llx", host_ms(g.ue), (unsigned long long)g.recseq, (unsigned long long)g.value,
                 (unsigned long long)((g.ue - g.us) * period / 1000));
        }
        logf("vk: tl %9.3f GPU                    gpu-cmd+       %#llx %#llx %#llx", host_ms(g.cs), (unsigned long long)g.recseq, (unsigned long long)g.value, 0ull);
        logf("vk: tl %9.3f GPU                    gpu-cmd-       %#llx %#llx %#llx", host_ms(g.ce), (unsigned long long)g.recseq, (unsigned long long)g.value,
             (unsigned long long)((g.ce - g.cs) * period / 1000));
    }
    probe_dump(host_ms, tl_rec_lo, tl_rec_hi);
    logf("vk: tl gpu end");
}
void gpu_time_end(Recording& r, bool uploads = false) {
    fo_recording_end(r);   // (at every recording's end, right before its submit: submit_recording)
    if (!ts_pool) return;
    vkCmdWriteTimestamp2(r.upload, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, ts_pool, 4 * r.slot + 1);
    vkCmdWriteTimestamp2(r.cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, ts_pool, 4 * r.slot + 3);
    ts_pending[r.slot] = true;
    ts_up_pending[r.slot] = uploads;
}
/// A segment's submission (its own vkQueueSubmit2, dev.present_mu held), counted with submit_recording's when it carries timestamps:
/// gpu_time_collect numbers every timed slot it collects, segments too, and a blit's place among them (blit_ts_mark,
/// ts_last_end_seen) is this count. Without it the count fell behind by every segment, and the blit's gaps read 0 after a few frames
void gpu_time_segment_submitted(const Recording& r) {
    if (!ts_pool || !ts_pending[r.slot]) return;
    std::lock_guard lk(ts_mu);
    ts_seq_submitted++;
}

// (hookshot) Real occlusion counts (on unless RX_VK_OCC_QUERIES=0; from Hookshot's draft by jordanella). Ocarina of Time
// reads its z-prepass samples-passed reports back on the CPU a frame or two later and leaves out of its draw lists the
// objects whose count was 0 (docs/development.md); with every report reading 4096 nothing was ever left out. Each draw from
// a samples-passed reset to its report gets an occlusion query, numbered from one counter for all the game's queues
// (occ_next_all; each command buffer's replay works on its state's copy, NvnState::occ_next, so a parallel replay's dry pass
// and its workers agree on them). The report is still written at replay as before ("visible"); once its queries have all
// ended on the GPU, the replay thread writes their sum over it, unless the slot has been reported into again since.
// The rule: a count that is not known for sure stays "visible". A wrong 0 hides an object the player can see; a late or
// unknown count only costs culling. So a report stays "visible" when its run took more than kOccRun draws, had a draw
// that writes colour, was split by another queue's queries, when one of its draws did not really draw (OccGuard in
// vk_draw.cpp: its query is never begun), or when its slot was reported into again first.
constexpr uint64_t kOccPool = 1 << 16;   // (the queries' ring)
constexpr uint64_t kOccRun = 256;        // (the most draws one report counts)
constexpr uint64_t kOccBuffer = 8192;    // (the most queries one command buffer takes)
constexpr uint64_t kOccVisible = 4096;   // (what a report reads while its count is not known)
VkQueryPool occ_pool = VK_NULL_HANDLE;
struct OccReport {
    uint64_t addr, ts;       // (the report's slot and the timestamp written with it; addr 0: a run with no count, freed)
    uint64_t first, n;       // (its queries)
    uint64_t value, part;    // (the submission they go out in (rec_cur_value at the report), and the recording count then)
    bool color = false;      // (a draw of the run writes colour)
    uint64_t seq = 0;        // (the slot's report sequence then: occ_slots)
};
std::deque<OccReport> occ_pending;   // (in report order; the replay thread's)
// (queries read, waiting to be reset until the GPU has finished the submission they went out in: its timeline value. The
// availability bit says the query has ended, not that its command buffer has: a reset then ran into submitted work, in
// the validation layer's view (VUID-vkResetQueryPool-firstQuery-02741, then -vkCmdBeginQuery-None-00807 at the next use))
struct OccFree {
    uint64_t first, n, value;
};
std::deque<OccFree> occ_freeing;
uint64_t occ_next_all = 0;           // (the next query's number, for every queue)
// (each report slot's latest report: a count is written only over the report it belongs to. The timestamp word alone
// was not enough: two reports into one slot within a tick (~0.5 us at RX_VK_TS_SCALE 0.1) look the same)
// Kept in a direct-mapped table (no allocation, one cache line a lookup): a slot that shares an entry with another evicts
// it, and the evicted slot's waiting count is then not written (its report stays "visible": the safe side)
struct OccSlot {
    uint64_t addr = 0, seq = 0;
};
constexpr size_t kOccSlots = 1 << 14;
std::vector<OccSlot> occ_slots(kOccSlots);
uint64_t occ_seq = 0;
OccSlot& occ_slot(uint64_t addr) { return occ_slots[(size_t)((addr >> 4) * 0x9E3779B97F4A7C15ull >> 50)]; }
struct {
    uint64_t real, zero, unknown, stale, lag, calls, polls, ns, sum, small, creal, czero, split, ahead, walks, walk_ns, copies, copied, fell_back;
} occ_st;

/// (hookshot) RX_VK_OCC_COPY: the queries' results copied by the GPU at the end of the recording that ran them
/// (vkCmdCopyQueryPoolResults, after the recording's end barrier) into a buffer in host memory, which occ_collect reads as plain
/// memory. Read from the pool (vkGetQueryPoolResults), each query cost ~3.4 us in the driver on an Arc B580 (101.9154; the time
/// grew with the queries read, not with the calls): 7.4 ms of the render thread's 28.8 ms a frame in Castle Town's market,
/// 34.8 fps there against 45.8 with the copy. A query the copy found not done (never begun: its draw did not draw), or one
/// whose recording ended without the copy, is read from the pool once its submission is done, as before. A count reads
/// kOccUnread until its copy has written it, whichever of its two words the GPU writes first. The copies cost GPU time (~2 ms
/// a frame in the market on the B580, ~1,400 queries). Red Tunic also enables copies for AMD integrated graphics on Windows
/// to avoid the driver cost of frequent CPU query reads. Off with parallel replay and without coherent host memory.
/// RX_VK_OCC_COPY=0 reads from the pool.
int par_threads();
bool occ_copy_wanted() {
    static const bool on = [] {
        if (par_threads() != 0) return false;
        return dev.policy.occ_copy;
    }();
    return on;
}
VkBuffer occ_host_buf = VK_NULL_HANDLE;
VmaAllocation occ_host_alloc = nullptr;
volatile uint64_t* occ_host = nullptr;   // ((count, availability) for each query of the ring, at its place in the pool)
constexpr uint64_t kOccUnread = ~0ull;   // (a count no copy has written since the query's reset)
uint64_t occ_part_lo = UINT64_MAX, occ_part_hi = 0;   // (the queries the recording being made took: occ_copy_part)
void occ_host_clear(uint64_t at, uint64_t k) {
    for (uint64_t i = at; i < at + k; i++) {
        occ_host[2 * i] = kOccUnread;
        occ_host[2 * i + 1] = 0;
    }
}

bool occ_queries_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_OCC_QUERIES");
        if (e && atoi(e) == 0) return false;
        if (getenv("RX_VK_OCC_FIXED")) {   // (a fixed value asked for: that value, as before, whatever RX_VK_OCC_QUERIES says)
            if (e) logf("vk: RX_VK_OCC_QUERIES: off, RX_VK_OCC_FIXED is set");
            return false;
        }
        // (the replay on the submitting thread, the default, writes "visible" at the submit, or at the stop for work held at
        // an event wait, occ_visible_ahead)
        // (RX_VK_ASYNC=1, the render thread: on too. A report reads "visible" from its replay until its queries are done, as
        // here; the game, a frame ahead at most, reads a count that much older, never an older report's: occ_collect)
        if (!dev.host_query_reset) {
            logf("vk: RX_VK_OCC_QUERIES: this device cannot reset queries from the CPU (hostQueryReset): off");
            return false;
        }
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_OCCLUSION;
        qi.queryCount = (uint32_t)kOccPool;
        if (vkCreateQueryPool(dev.device, &qi, nullptr, &occ_pool) != VK_SUCCESS) {
            logf("vk: RX_VK_OCC_QUERIES: the query pool could not be made: off");
            occ_pool = VK_NULL_HANDLE;
            return false;
        }
        vkResetQueryPool(dev.device, occ_pool, 0, (uint32_t)kOccPool);
        if (occ_copy_wanted()) {
            void* m = nullptr;
            if (create_buffer_cached(kOccPool * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, occ_host_buf, occ_host_alloc, &m, true) && m &&
                host_coherent(occ_host_alloc)) {
                occ_host = (volatile uint64_t*)m;
                occ_host_clear(0, kOccPool);
            } else {
                if (occ_host_buf) destroy_buffer(occ_host_buf, occ_host_alloc);
                occ_host_buf = VK_NULL_HANDLE, occ_host_alloc = nullptr;
            }
        }
        logf("vk: occlusion queries on (RX_VK_OCC_QUERIES): the game's samples-passed reports get the GPU's counts%s; %s",
             dev.occlusion_precise ? "" : " (not precise: a count says only none or some)",
             occ_host ? "copied into host memory at each recording's end (RX_VK_OCC_COPY)"
                      : occ_copy_wanted() ? "read from the pool (RX_VK_OCC_COPY: no coherent host memory for the copies)"
                                          : "read from the pool");
        return true;
    }();
    return on;
}
/// The ring's queries [first, first + n) back to unused (each one done on the GPU, or never begun)
void occ_reset(uint64_t first, uint64_t n) {
    while (n) {
        const uint64_t at = first % kOccPool, k = std::min(n, kOccPool - at);
        vkResetQueryPool(dev.device, occ_pool, (uint32_t)at, (uint32_t)k);
        if (occ_host) occ_host_clear(at, k);   // (the GPU is done with them, and with their copies: the same submissions)
        first += k, n -= k;
    }
}
/// Room in the ring for another command buffer's queries. oldest_open: the first query of the oldest run still open in any
/// queue's state (taken, not in occ_pending yet; UINT64_MAX: none)
bool occ_room(uint64_t oldest_open) {
    uint64_t tail = std::min(oldest_open, occ_next_all);
    for (const OccReport& o : occ_pending) tail = std::min(tail, o.first);   // (two queues' reports can come out of order)
    for (const OccFree& f : occ_freeing) tail = std::min(tail, f.first);
    return occ_queries_on() && occ_next_all - tail + kOccBuffer <= kOccPool;
}
/// A command buffer's replay begins on state s: its numbering from the counter every queue shares. When another queue's
/// command buffers took queries since s's last one, a run s has open is split: what it took is freed and its report stays
/// "visible". (Each queue had its own count before: two queues handed out the same queries.) value: the submission it goes
/// out in. Then the limit, from the ring's room (oldest_open: the other states' oldest open run, occ_room)
void occ_begin_buffer(NvnState& s, uint64_t value, uint64_t oldest_open) {
    if (s.occ_next != occ_next_all) {
        if (s.occ_on) {
            if (s.occ_next > s.occ_first)
                occ_pending.push_back({0, 0, s.occ_first, s.occ_next - s.occ_first, value, rec_part_next});
            if (s.occ_next > s.occ_first || !s.occ_overflow) occ_st.split++;
            s.occ_overflow = true;
            s.occ_first = occ_next_all;
        }
        s.occ_next = occ_next_all;
    }
    if (s.occ_on) oldest_open = std::min(oldest_open, s.occ_first);
    s.occ_limit = occ_room(oldest_open) ? s.occ_next + kOccBuffer : 0;
}
/// ... and ends: the queries it took (its dry pass's, in a parallel replay) are the counter's
void occ_end_buffer(const NvnState& s) { occ_next_all = s.occ_next; }
/// A queue's state made as a copy of another's: a run the other has open is not this queue's
void occ_state_copied(NvnState& s) {
    s.occ_on = false, s.occ_overflow = false, s.occ_color = false;
    s.occ_first = s.occ_next;
}
/// A samples-passed reset (NOP_RESET_COUNTER): a run begins; one still open had no report: its queries are freed. A worker
/// numbers as the dry pass did and frees nothing
void occ_counter_reset(NvnState& s, bool worker, uint64_t value) {
    if (s.occ_on && s.occ_next > s.occ_first && !worker)
        occ_pending.push_back({0, 0, s.occ_first, s.occ_next - s.occ_first, value, rec_part_next});
    s.occ_on = s.occ_next < s.occ_limit;
    s.occ_first = s.occ_next;
    s.occ_overflow = false;
    s.occ_color = false;
}
/// The run a report counts, closed (NOP_REPORT_COUNTER, every mode: a worker numbers its queries as the dry pass did)
struct OccRun {
    uint64_t first = 0, n = 0;
    bool counted = false, color = false;
};
OccRun occ_close(NvnState& s, bool samples_passed) {
    OccRun r;
    if (!samples_passed || !s.occ_on) return r;
    r.first = s.occ_first, r.n = s.occ_next - s.occ_first;
    r.counted = !s.occ_overflow, r.color = s.occ_color;
    s.occ_on = false, s.occ_overflow = false;
    return r;
}
/// A slot's report sequence moved on: a count still waiting for an older report there is not written
uint64_t occ_slot_next(uint64_t addr) {
    OccSlot& e = occ_slot(addr);
    e.addr = addr;
    return e.seq = ++occ_seq;
}
/// A report written by the replay (any counter: the slot's sequence moves on), and the count of its run to come, at
/// value (rec_cur_value: the submission its queries go out in. Not the last begin_recording's value: a present's and
/// frame generation's recordings begin with their own)
void occ_reported(const OccRun& run, uint64_t addr, uint64_t ts, bool flag_written, uint64_t value) {
    if (!occ_pool) return;
    const uint64_t seq = occ_slot_next(addr);
    if (run.n)
        occ_pending.push_back({run.counted && !flag_written ? addr : 0, ts, run.first, run.n, value, rec_part_next, run.color, seq});
}
/// A samples-passed report the replay has not reached yet (its queue stopped at an event wait): "visible" now, so the game
/// never reads an older count there in the meantime; an older count still waiting for the slot is not written
void occ_slot_visible(uint64_t addr) {
    if (!occ_pool) return;
    occ_slot_next(addr);
    uint64_t now = 0;
    host->gpu_read(addr, &now, 8);
    if (now == kOccVisible) return;
    // (no memory_cpu_owned: the slot was made the CPU's when it was last reported, and will be again when this report is
    // replayed; ~1,500 more words a frame made every write-back's cut dearer)
    guest_cpu_write(addr, &kOccVisible, 8);
    occ_st.ahead++;
}
/// The reports whose queries are done, written into the game's slots (the replay thread, as a recording begins). The
/// queries of every waiting report are asked for at once, as one run of the ring (two calls where it wraps), without a wait;
/// the reports from the oldest on whose queries have all ended (or whose submission the GPU has finished: a query never begun
/// is unknown then) are written, and their queries reset with one call. (Hookshot's draft asked and reset each report's
/// queries on their own, ~1,500 of each a frame in the Market; waiting for a whole submission instead wrote the counts 3-13
/// recordings late, about two frames, often after the game had read the slot.)
void occ_collect() {
    if (!occ_pool) return;
    if (!occ_freeing.empty()) {   // (the queries read before, whose submissions the GPU has finished: reset)
        const uint64_t done = completed();
        while (!occ_freeing.empty() && done >= occ_freeing.front().value) {
            occ_reset(occ_freeing.front().first, occ_freeing.front().n);
            occ_freeing.pop_front();
        }
    }
    if (occ_pending.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t done = completed();
    // (the waiting reports' queries follow on from each other: each run starts where the last one ended; a gap (two
    // queues' runs reported out of order) ends the batch there and the rest wait for the next call)
    const uint64_t first = occ_pending.front().first;
    uint64_t end = first;
    size_t span = 0;
    for (const OccReport& o : occ_pending) {
        if (o.first != end) break;
        end += o.n, span++;
    }
    static std::vector<uint64_t> got;   // (count, availability) for each query of [first, end)
    got.assign(2 * (end - first), 0);
    const auto pool_read = [&](uint64_t from, uint64_t to) {   // (the queries [first + from, first + to), from the pool)
        for (uint64_t i = from; i < to;) {
            const uint64_t at = (first + i) % kOccPool, k = std::min(to - i, kOccPool - at);
            vkGetQueryPoolResults(dev.device, occ_pool, (uint32_t)at, (uint32_t)k, k * 16, got.data() + 2 * i, 16,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);   // (VK_NOT_READY: some not done)
            i += k;
        }
    };
    if (occ_host) {   // (RX_VK_OCC_COPY: their copies. One is there once its availability word is set and its count written)
        for (uint64_t i = 0; i < end - first; i++) {
            const uint64_t at = (first + i) % kOccPool;
            const uint64_t a = occ_host[2 * at + 1], c = occ_host[2 * at];
            got[2 * i] = c, got[2 * i + 1] = a && c != kOccUnread;
        }
    } else {
        pool_read(0, end - first);
    }
    size_t ready = 0;
    for (size_t r = 0; r < span; r++) {
        const OccReport& o = occ_pending[r];
        const uint64_t from = o.first - first, to = from + o.n;
        bool all = true;
        for (uint64_t i = from; i < to && all; i++) all = got[2 * i + 1] != 0;
        if (!all && occ_host && o.value && done >= o.value) {   // (its submission done: what its copy did not have, from the pool)
            std::fill(got.begin() + 2 * from, got.begin() + 2 * to, 0);
            pool_read(from, to);
            occ_st.fell_back += o.n;
        }
        if (!all && !(o.value && done >= o.value)) break;   // (still on the GPU: it and those after it wait)
        ready++;
    }
    occ_st.polls++;
    if (!ready) {
        const OccReport& o = occ_pending.front();
        static int logged;
        if (rec_part_next - o.part > 200 && logged++ < 4)
            logf("vk: occlusion queries: a report has waited %llu recordings (value %llu, done %llu, %zu waiting)",
                 (unsigned long long)(rec_part_next - o.part), (unsigned long long)o.value, (unsigned long long)done,
                 occ_pending.size());
        occ_st.ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        return;
    }
    end = occ_pending[ready - 1].first + occ_pending[ready - 1].n;
    uint64_t latest = 0;   // (the last submission these queries went out in)
    for (size_t r = 0; r < ready; r++) {
        const OccReport& o = occ_pending.front();
        latest = std::max(latest, o.value);
        bool all = true;
        uint64_t sum = 0;
        for (uint64_t i = o.first - first; i < o.first - first + o.n; i++) {
            if (!got[2 * i + 1]) all = false;   // (its draw did not draw: its query never begun, unknown)
            else sum += got[2 * i];
        }
        if (o.addr && all) {
            uint64_t ts = 0;
            host->gpu_read(o.addr + 8, &ts, 8);
            const OccSlot& e = occ_slot(o.addr);
            static const bool dry = env_on("RX_VK_OCC_DRY");   // (a test: the queries made and read, the reports left at 4096)
            if (ts == o.ts && e.addr == o.addr && e.seq == o.seq) {
                if (!dry) guest_cpu_write(o.addr, &sum, 8);
                (sum ? occ_st.real : occ_st.zero)++;
                occ_st.sum += sum, occ_st.small += sum && sum < 64;
                if (o.color) (sum ? occ_st.creal : occ_st.czero)++;
                occ_st.lag += rec_part_next - o.part;
            } else {
                occ_st.stale++;   // (reported again before this one was done: the newer one stands)
            }
        } else if (o.addr) {
            occ_st.unknown++;
        }
        occ_pending.pop_front();
    }
    // (reset once the GPU has finished their submissions: at once when it has, else at a later call. In order of value:
    // a later batch waits behind an earlier one)
    if (done >= latest && occ_freeing.empty()) occ_reset(first, end - first);
    else occ_freeing.push_back({first, end - first, std::max(latest, occ_freeing.empty() ? 0 : occ_freeing.back().value)});
    occ_st.calls++;
    occ_st.ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    static const bool log_on = env_on("RX_VK_OCC_LOG");
    const uint64_t n = occ_st.real + occ_st.zero + occ_st.unknown + occ_st.stale;
    if (log_on && n >= 5000) {
        const uint64_t w = occ_st.real + occ_st.zero;
        uint64_t nd[3] = {0, 0, 0};   // (vk_draw.cpp: queries left unbegun, by why)
        occ_draw_counts(nd);
        logf("vk: occlusion queries: %llu reports written (%llu of them 0, %.1f%%), %llu left visible (a draw did not draw), "
             "%llu reported again first; written %.1f recordings after the report; %zu waiting; %llu reads (%llu wrote), %.1f us each; "
             "the others: %.0f samples on average, %llu under 64; runs with colour draws %llu (%llu of them 0); "
             "queries not begun: %llu draws returned before drawing, %llu with vertex input not all bound, %llu task batches; "
             "%llu runs split by another queue's queries; %llu slots made visible ahead of a stopped queue (%llu walks, %.1f us each); "
             "%llu copies, %.1f queries each, %llu queries read from the pool after their copy (RX_VK_OCC_COPY)",
             (unsigned long long)w, (unsigned long long)occ_st.zero, w ? 100.0 * occ_st.zero / w : 0.0,
             (unsigned long long)occ_st.unknown, (unsigned long long)occ_st.stale, w ? (double)occ_st.lag / w : 0.0,
             occ_pending.size(), (unsigned long long)occ_st.polls, (unsigned long long)occ_st.calls,
             occ_st.polls ? occ_st.ns / 1e3 / occ_st.polls : 0.0, occ_st.real ? (double)occ_st.sum / occ_st.real : 0.0,
             (unsigned long long)occ_st.small, (unsigned long long)(occ_st.creal + occ_st.czero), (unsigned long long)occ_st.czero,
             (unsigned long long)nd[0], (unsigned long long)nd[1], (unsigned long long)nd[2], (unsigned long long)occ_st.split,
             (unsigned long long)occ_st.ahead, (unsigned long long)occ_st.walks,
             occ_st.walks ? occ_st.walk_ns / 1e3 / occ_st.walks : 0.0, (unsigned long long)occ_st.copies,
             occ_st.copies ? (double)occ_st.copied / occ_st.copies : 0.0, (unsigned long long)occ_st.fell_back);
        occ_st = {};
    }
}

/// (RX_VK_OCC_COPY) The queries the recording took, copied into the host buffer after its last draws (the recording's end
/// barrier is before this: a query not done by then is copied as not available), then made the host's. Every recording's end
/// forgets them (submit_recording): the queries of one that ended without this are read from the pool
void occ_copy_part(VkCommandBuffer cmd) {
    if (occ_host && occ_part_hi > occ_part_lo) {
        for (uint64_t q = occ_part_lo; q < occ_part_hi;) {
            const uint64_t at = q % kOccPool, k = std::min(occ_part_hi - q, kOccPool - at);
            vkCmdCopyQueryPoolResults(cmd, occ_pool, (uint32_t)at, (uint32_t)k, occ_host_buf, at * 16, 16,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            q += k;
        }
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        occ_st.copies++, occ_st.copied += occ_part_hi - occ_part_lo;
    }
    occ_part_lo = UINT64_MAX, occ_part_hi = 0;
}

/// RX_VK_OCC_QUERIES=2: the runs with colour draws counted too (occ_take)
bool occ_all() {
    static const bool on = getenv("RX_VK_OCC_QUERIES") && atoi(getenv("RX_VK_OCC_QUERIES")) == 2;
    return on;
}
/// A draw's query while a run counts: numbered in every mode (the stream's numbering), attached only where it is recorded
void occ_take(NvnState& s, DrawArgs& da, bool dry) {
    if (!s.occ_on || !s.render_enabled || s.raster_discard) return;
    // (only runs of depth-only draws are counted: the game's bounding boxes. A run with a draw that writes colour stays
    // "visible": a query's end waits for its draw to finish, and around the scene's real draws that took ~0.9 ms of GPU time a
    // frame in the Market (RTX 3090) for a few more objects left out. RX_VK_OCC_QUERIES=2: every run counted)
    const bool color = s.chmask.mask != 0;
    s.occ_color |= color;
    if (s.occ_overflow) return;   // (its count unknown already: no more queries for it)
    if (s.occ_next - s.occ_first >= kOccRun || s.occ_next >= s.occ_limit || (color && !occ_all())) {
        s.occ_overflow = true;
        return;
    }
    const uint64_t q = s.occ_next++;
    if (!dry) {
        da.occ_pool = occ_pool, da.occ_query = (uint32_t)(q % kOccPool);
        if (occ_host) occ_part_lo = std::min(occ_part_lo, q), occ_part_hi = std::max(occ_part_hi, q + 1);
    }
}

/// The next recording slot, begun, for the submission that will signal value
Recording& begin_recording(uint64_t value) {
    occ_collect();
    Recording& r = recordings[recording_index];
    recording_index = (recording_index + 1) % kRecordings;
    if (r.value) {
        PROF(PROF_SLOTWAIT);
        const auto t0 = std::chrono::steady_clock::now();
        recording_wait(r);
        st_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        gpu_time_collect(r);
        gprof_collect(r.slot);
    }
    rec::slot_begin(rec_end_pos[r.slot]);   // (RX_VK_RECORDER_OWN: the slot's own resets below and in draw_begin_recording)
    {
        rec::OwnSlot own;
        vkResetCommandPool(dev.device, r.pool, 0);
        vkResetCommandPool(dev.device, r.upload_pool, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(r.cmd, &bi);
        vkBeginCommandBuffer(r.upload, &bi);
    }
    gpu_time_begin(r);
    gprof_begin_recording(r.slot, r.cmd, r.upload);
    r.value = value;
    r.part = ++rec_part_next;
    memory_recording_part(r.part);
    memory_recording_value(value, completed());
    texture_begin_recording(r.upload, value);
    draw_begin_recording(r.slot, r.cmd, value);
    rec::slot_done();
    return r;
}
// when each timeline value's recording went to the queue (host steady clock, ns): a capture's wait reports how
// long the GPU had it (vk_timeline_wait)
std::atomic<int64_t> value_submit_ns[1024];
std::atomic<uint64_t> value_submit_v[1024];
/// The game reads what the GPU wrote into its memory once the timeline reaches the submission; waiting on the host makes
/// it visible to the device only (a fence's or a semaphore's access scope is device access), so each recording ends with
/// a memory dependency into the host's reads
void host_read_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, b.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.memoryBarrierCount = 1, d.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &d);
}
void submit_recording(Recording& r, bool signal_value = true, uint64_t signal_at = 0) {
#ifdef TRACY_ENABLE
    HT_SCOPE("submit_recording");
#endif
    occ_part_lo = UINT64_MAX, occ_part_hi = 0;   // (RX_VK_OCC_COPY: copied only if occ_copy_part did, else read from the pool)
    draw_flush_recording(r.slot);
    const bool uploads = texture_uploads_recorded();
    gprof_submitted(r.slot, uploads);
    gpu_time_end(r);
    host_read_barrier(r.cmd);
    vkEndCommandBuffer(r.upload);
    vkEndCommandBuffer(r.cmd);
    rec_end_pos[r.slot] = rec::produced();   // (the ends waited for the recorder: it is past this)
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = timeline;
    signal.value = signal_at ? signal_at : r.value;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cb[2]{{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}};
    uint32_t n = 0;
    if (uploads) cb[n++].commandBuffer = r.upload;
    cb[n++].commandBuffer = r.cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = n;
    si.pCommandBufferInfos = cb;
    VkSemaphoreSubmitInfo signals[2] = {signal, {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}};
    uint32_t nsig = signal_value ? 1 : 0;
    uint64_t seq = 0;
    // (hookshot) Where this recording is to be found by the threads that wait for it (value_rec, part_rec). A held submit
    // is put there only once it is in the list of held ones, under dev.present_mu: a thread that finds a recording and
    // waits for it flushes first (wait_rec_seq), and one found before it was in the list had nothing to flush and waited
    // for a signal still held, until something else sent the batch (also before the flushes looked at pend_held: found
    // by mooki000 there)
    const auto map_recording = [&] {
        std::lock_guard mk(rec_map_mu);
        // (a submission is recorded in parts (maybe_split, a parallel replay's segments), each part's recording with the
        // submission's value: the value's recording is its last part's, the latest seq. emplace kept the first part's,
        // and a wait for the submission's fence (wait, the readback wait) came back once its first ~120 draws were done:
        // the game wrote the next frame's uniforms, vertices and its exposure meter over memory the later parts still
        // read (with no VRAM mirror, an iGPU: guest memory read in place). A frame of the HUD stretched across the
        // screen, grass and objects glowing white, whole-screen flashes: issue 997)
        const auto finished_in = [&](uint64_t v) {
            auto it = value_rec.find(v);
            if (it == value_rec.end()) value_rec.emplace(v, seq);
            else if (it->second < seq) it->second = seq;
        };
        for (uint64_t v : rec_open_values) finished_in(v);
        if (r.value) finished_in(r.value);
        rec_open_values.clear();
        while (value_rec.size() > 8192) value_rec.erase(value_rec.begin());
        if (r.part) part_rec.emplace_hint(part_rec.end(), r.part, seq);
        while (part_rec.size() > 8192) part_rec.erase(part_rec.begin());
        rec_map_cv.notify_all();
    };
    if (rec_timeline) {   // (and its number on the recordings' timeline; every caller holds replay_mu: in order)
        seq = ++rec_seq;
        signals[nsig].semaphore = rec_timeline;
        signals[nsig].value = seq;
        signals[nsig].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        nsig++;
        if (!batch_on()) map_recording();
    }
    si.signalSemaphoreInfoCount = nsig;
    si.pSignalSemaphoreInfos = nsig ? signals : nullptr;
    std::lock_guard lk(dev.present_mu);   // (one queue: submits and presents take turns)
    if (batch_on()) {
        // (NVIDIA Reflex: the batch is tagged with the game frame its first command buffer belongs to, read here. A batch that
        // spans two game frames counts as the first's; one begun before the frame's sleep returned (reflex_frame() 0) goes
        // untagged. Batching is on by default only with RADV, so on NVIDIA this matters only with RX_VK_BATCH=1)
        if (pend.cbs.empty()) batch_next++, pend_since = std::chrono::steady_clock::now(), pend.reflex = reflex_frame();
        for (uint32_t i = 0; i < n; i++) pend.cbs.push_back(cb[i]);
        pend_held.store(true, std::memory_order_release);
        if (seq) map_recording();   // (held, and said so: now it may be found and waited for)
        if (signal_value) pend.signal = std::max(pend.signal, signal.value), pend.any_signal = true;
        if (seq) pend.rec = std::max(pend.rec, seq);
        r.batch = batch_next;
        r.submitted = true;
        // (RX_VK_BATCH_MAX=<command buffers>: held at most this many, default 32)
        static const size_t cap = getenv("RX_VK_BATCH_MAX") ? (size_t)std::max(1, atoi(getenv("RX_VK_BATCH_MAX"))) : 32;
        if (pend.cbs.size() >= cap) flush_reason = FR_CAP, queue_flush_locked();
        // (hookshot, #1993 by mrdev023) Sent at once while the GPU has done every batch sent so far: a held batch waited for
        // the next wait or present and the GPU sat idle meanwhile, and with serial replay the game waits for it mid-frame
        // (take_capture's late pages), so the GPU started cold just as it was waited for. RX 9070 XT, RADV, Kokiri Forest:
        // 33-41 fps -> 46-61, no flashes in 5230 frames; ~10 flushes a frame of ~3.7 command buffers each, so batching still
        // groups the submits while the GPU is busy. RX_VK_BATCH_IDLE=0: held until a wait, a present or the cap, as before
        static const bool idle_flush = !getenv("RX_VK_BATCH_IDLE") || atoi(getenv("RX_VK_BATCH_IDLE")) != 0;
        if (idle_flush && !pend.cbs.empty()) {
            uint64_t done = 0;
            if (vkGetSemaphoreCounterValue(dev.device, batch_tl, &done) == VK_SUCCESS && done >= batch_flushed)
                flush_reason = FR_OTHER, queue_flush_locked();
        }
        if (!pend.cbs.empty() && batch_gpu_idle()) flush_reason = FR_OTHER, queue_flush_locked();   // (#1993)
        if (!pend.cbs.empty() && readback_send_now()) flush_reason = FR_READBACK, queue_flush_locked();
        return;
    }
    PROF(PROF_QSUBMIT);
    si.pNext = reflex_submit_id(nullptr);   // (NVIDIA Reflex: the frame being made)
    VkSemaphoreSubmitInfo sparse_wait;   // (hookshot: after the sparse mirrors' binds, RX_VK_MIRROR_SPARSE)
    if (memory_sparse_wait(&sparse_wait)) si.waitSemaphoreInfoCount = 1, si.pWaitSemaphoreInfos = &sparse_wait;
    const VkResult res = submit_or_retry(si, r.fence);
    r.submitted = res == VK_SUCCESS;
    static int failures;
    if (res != VK_SUCCESS && failures++ < 8) logf("vk: submit %llu failed (%d)", (unsigned long long)r.value, (int)res);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    if (res == VK_SUCCESS && signal_value) note_timeline_sent(signal.value);
}

// ---- NVN state (the draw path's, vk_state.h) ----
NvnState state;
// (each of the game's queues has one of its own, as the console's channels do: their command buffers are replayed in
// turns at their event waits (submit_job), and one's draws went on with what another's had set. A queue seen first
// starts from the state the replay was at, as all of them did before. The queue-less replays (RX_VK_ASYNC=0, the
// bench) keep the one above.)
std::unordered_map<uint64_t, NvnState> qstates;
NvnState* cur_state = &state;
/// (hookshot) The first query of the oldest occlusion run still open in a queue's state other than except (UINT64_MAX: none)
uint64_t occ_open_oldest(const NvnState* except) {
    uint64_t m = UINT64_MAX;
    if (&state != except && state.occ_on) m = state.occ_first;
    for (const auto& [q, t] : qstates)
        if (&t != except && t.occ_on) m = std::min(m, t.occ_first);
    return m;
}
/// The state the replay goes on with: its queue's. (The draw path's caches made afresh: they are keyed on the state's
/// counters, which the queues' states share no order of.)
void use_queue_state(uint64_t queue) {
    NvnState* s = &state;
    if (queue) {
        auto it = qstates.find(queue);
        if (it == qstates.end()) {
            it = qstates.emplace(queue, *cur_state).first;
            occ_state_copied(it->second);   // (hookshot: the other queue's open occlusion run is not this one's)
        }
        s = &it->second;
    }
    if (s == cur_state) return;
    cur_state = s;
    static uint64_t gen_base = 0;
    gen_base += 1ull << 32;
    s->bind_gen = gen_base, s->rt_serial = gen_base;
    for (int k = 0; k < kNvnStages; k++) s->tex_gen[k] = gen_base, s->ssbo_gen[k] = gen_base;
}

/// payload's copy of a payload shorter than its type, rare: a function of its own, so that the compiler can't merge it with the
/// copy at the type's size into one call of memcpy with a chosen size (clang 23 does, at -O2)
[[gnu::noinline, gnu::cold]] void payload_short(void* v, const uint8_t* a, uint32_t bytes) { std::memcpy(v, a, bytes); }

template <class T>
T payload(const uint8_t* a, uint32_t bytes) {
    // (a payload at least as long as T, nearly every one, is copied at T's size: a copy the compiler makes inline. A variable size
    // was a call of the C library's memcpy for every op replayed, a few percent of the Steam Deck's render thread, where it shows
    // as __memmove_avx_unaligned_erms)
    T v{};
    if (bytes >= sizeof v) [[likely]]
        std::memcpy(&v, a, sizeof v);
    else
        payload_short(&v, a, bytes);
    return v;
}

int64_t st_barriers, st_ub_updates, st_ub_overlays, st_submits;
/* (hookshot) what makes a draw rebuild its descriptors: which NVN binding moved the state's generation. A draw's
   descriptor set is rebuilt when bind_gen moved, and 0.9 of the draws rebuilt theirs (3.2 ms a frame). */
int64_t st_bg_prog, st_bg_bufs, st_bg_ubo, st_bg_ssbo, st_bg_tex, st_bg_samp, st_bg_img;
std::map<int32_t, int64_t> st_barrier_flags;   // (the game's nvnCommandBufferBarrier bits: how many of each combination, the frame log)
std::mutex st_barrier_flags_mu;   // (a parallel replay's threads count their barriers too)
// ---- events in replay order: the values the stream's signals and the CPU's set, for the GPU's waits on them. The
// game's three queues order their work with events (a command buffer waits for another queue's to reach a stage);
// replayed on one queue in the order they were submitted, a waiter submitted before its signaller ran first ----
std::unordered_map<uint64_t, uint32_t> ev_values;
// (hookshot) The CPU's signals, while a replay holds replay_mu: the main thread stopped for it in nvnEventSignal, ~1.7 ms
// a frame of the critical path. They wait here for the replay's next look at the events (ev_cpu_apply).
std::mutex ev_cpu_mu;
std::vector<std::pair<uint64_t, uint32_t>> ev_cpu_pending;
std::atomic<bool> ev_cpu_any{false};
// (stream R2) RX_VK_WATCH_KICK=1: the CPU's event signal wakes the order thread (sync_watch) at once, instead of its Sleep(1)
// finding the signal up to ~1-2 ms later (timer resolution 1 ms). On by default after the second session's validation (castle-move 10
// runs against 10, 97.5 -> 104.5 fps, +7.1%; the three scenes ran clean, shots and frame-log luma right; docs/native-engine/readback.md).
// RX_VK_WATCH_KICK=0: as before
HANDLE watch_kick_event() {
    static HANDLE h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return h;
}
bool watch_kick_on() {
    static const bool v = env_on("RX_VK_WATCH_KICK");   // (off by default here: =1 on)
    return v;
}
// (hookshot) how many signals the CPU has made, and how many of them a look at the stopped queues has been taken since
// (schedule, whole): equal, no queue is stopped at a wait the CPU has met and nobody has looked at (sync_settle_met)
std::atomic<uint64_t> ev_cpu_signals{0}, ev_cpu_settled{0};
// (and whether a stopped queue waits on an event neither the stream nor nvnEventSignal has ever set: its value is read
// from guest memory (event_value_in), where a store of the CPU's or a write-back can meet the wait with no signal counted;
// update_blocked)
std::atomic<bool> ev_wait_in_memory{false};
// (a look's count, kept until the burst it was taken in has finished its recording: burst_end. sync_order_mu)
uint64_t ev_looked = 0;
bool ev_looked_any = false;
std::unordered_set<uint64_t> cpu_events;   // (events the CPU has signalled, nvnEventSignal; ev_cpu_mu)
void ev_cpu_apply() {
    if (!ev_cpu_any.load(std::memory_order_acquire)) return;
    std::lock_guard lk(ev_cpu_mu);
    for (const auto& e : ev_cpu_pending) ev_values[e.first] = e.second, cpu_events.insert(e.first);
    ev_cpu_pending.clear();
    ev_cpu_any.store(false, std::memory_order_release);
}
int64_t st_waits_met, st_waits_unmet;
bool event_wait_met(uint32_t have, uint32_t mode, uint32_t want) {
    return mode == 0 ? have == want : (int32_t)(have - want) >= 0;   // (NVN: EQUAL, GEQUAL_WRAP)
}
using EvMap = std::unordered_map<uint64_t, uint32_t>;
uint32_t event_value_in(const EvMap& ev, uint64_t gpu) {   // (one the stream never set: as the CPU has it)
    auto it = ev.find(gpu);
    if (it != ev.end()) return it->second;
    uint32_t v = 0;
    host->gpu_read(gpu, &v, 4);
    return v;
}
thread_local bool suspend_ok = false;               // (a serial replay that stops at a wait not met: submit_job)
thread_local const uint8_t* suspend_at = nullptr;   // (the wait it stopped at)
// ---- (hookshot) a wait for an event the CPU signals that is met already when the replay reaches it ----
// The game writes part of a frame's data after it has submitted the frame's work, then signals the event that work waits
// for (OoT: the scene buffer's light grid, among others). The console's GPU reads memory after its wait. Here a queue that
// stopped at the wait gets a capture when it goes on (resume_queue). One that found the wait met went on with its submit's
// capture, and when the signal had come after that capture was begun, the frame was drawn with that data as it was before
// the game wrote it: the lit scene and Link dark for a frame (issue 366; 0 to 8 frames in 15,000 at Castle Town's gate,
// the frames where the signal came within ~60 us of the submit). Such a wait now stops there too, once, and schedule()
// lets it on at once through resume_queue and its capture. A signal made before the newest capture was begun needs
// none: what the game wrote up to it is in that capture. On the submitting thread only: the render thread's capture is
// taken at the signal itself (event_signal_queued).
std::unordered_map<uint64_t, uint64_t> ev_cpu_captures;   // (an event: memory_captures_begun at the CPU's last signal of it; ev_cpu_mu)
thread_local const uint8_t* met_resumed_at = nullptr;     // (the wait a resume starts at: not stopped at again)
int64_t st_met_stops;
bool async_on();
bool met_wait_capture_on() {
    static const bool v = !async_on();
    return v;
}
/// Whether the CPU's last signal of this event came after the newest capture was begun: what the game wrote before that
/// signal may be in no capture yet
bool cpu_signal_uncaptured(uint64_t gpu) {
    const uint64_t begun = memory_captures_begun();   // (first: a signal made meanwhile counts as after it)
    std::lock_guard ek(ev_cpu_mu);
    const auto it = ev_cpu_captures.find(gpu);
    return it != ev_cpu_captures.end() && it->second >= begun;
}
uint64_t ts_anchor_ns = 0;   // (the last present's time: counter timestamps are scaled from it)

/// (hookshot) The binding of up to n bytes of guest memory from gpu that one import holds, n cut to its length: all of it when an
/// import can hold it; else the part in the game pool holding gpu (a range across two pools has no import: pool_lookup), or,
/// outside every pool, none for the bytes up to the next one. None too when the range's import failed (logged there)
BufferRef gpu_piece_bind(uint64_t gpu, uint64_t& n) {
    if (const BufferRef b = gpu_buffer_bind(gpu, n, true)) return b;
    bool in_pool = false;
    const uint64_t k = gpu_pool_piece(gpu, n, in_pool);
    if (k == n) return {};   // (in one pool, whose import failed, or in none)
    n = k;
    return in_pool ? gpu_buffer_bind(gpu, n, true) : BufferRef{};
}

/// (hookshot) k bytes into b on the GPU timeline, after a barrier: an update when they are 4-byte aligned (vkCmdUpdateBuffer's
/// rule, up to 64 KB), else put in the recording's stream and copied into place, a copy having no alignment rule. false: no
/// room in the stream, and nothing recorded
bool gpu_put(VkCommandBuffer cmd, const BufferRef& b, const uint8_t* d, uint64_t k) {
    const bool update = !(b.offset & 3) && !(k & 3) && k <= 65536;
    const BufferRef s = update ? BufferRef{} : stream_copy(d, k);
    if (!update && !s) return false;
    full_barrier(cmd);   // overlapping inline writes must reach memory in stream order
    if (update) {
        vkCmdUpdateBuffer(cmd, b.buffer, b.offset, k, d);
    } else {
        const VkBufferCopy r{s.offset, b.offset, k};
        vkCmdCopyBuffer(cmd, s.buffer, b.buffer, 1, &r);
    }
    return true;
}

/// A GPU-timeline write of guest memory (the stream's own writes: they reach memory in command order). (hookshot) Every part an
/// import holds goes on the GPU, an unaligned one too, one import at a time; only bytes no import holds are written on the CPU,
/// at record time, which is exact outside every pool: no import there, so the GPU neither reads nor writes them. An unaligned
/// write, or one across two game pools, went on the CPU whole: out of command order, ahead of the draws recorded before it
void gpu_write(VkCommandBuffer cmd, uint64_t gpu, const void* data, uint64_t n) {
    const uint8_t* d = (const uint8_t*)data;
    while (n) {
        uint64_t k = std::min<uint64_t>(n, 65536);
        const BufferRef b = gpu_piece_bind(gpu, k);
        if (!b || !gpu_put(cmd, b, d, k)) guest_cpu_write(gpu, d, k);   // (no import, a failed one, or a full stream: on the CPU)
        gpu += k, d += k, n -= k;
    }
}

/// (hookshot) n bytes of guest memory from gpu cleared to value on the GPU timeline (NOP_CLEAR_BUF, between its barriers): each
/// part an import holds as a fill of whole words, the up to 3 bytes on either side copied from the recording's stream (a fill
/// takes aligned words only), stamped as the GPU's. The rest on the CPU (an unaligned clear, or one across two game pools, went
/// there whole, out of command order), unless in_stream: the replay cleared those bytes on the CPU in stream order already, and
/// the GPU's parts are not stamped. True when every byte went on the GPU
bool gpu_fill(VkCommandBuffer cmd, uint64_t gpu, uint64_t n, uint32_t value, bool in_stream) {
    uint8_t pattern[8];   // (byte i of the clear is the value's byte i % 4: pattern[i % 4] on)
    std::memcpy(pattern, &value, 4), std::memcpy(pattern + 4, &value, 4);
    bool all = true;
    for (uint64_t done = 0; done < n;) {
        const uint64_t at = gpu + done;
        uint64_t k = n - done;
        const BufferRef b = gpu_piece_bind(at, k);
        const uint64_t head = std::min<uint64_t>(k, (0 - b.offset) & 3), mid = (k - head) & ~3ull, tail = k - head - mid;
        uint8_t ends[6];   // (the head's bytes, then the tail's)
        std::memcpy(ends, pattern + (done & 3), head);
        std::memcpy(ends + head, pattern + ((done + head + mid) & 3), tail);
        const BufferRef s = b && head + tail ? stream_copy(ends, head + tail) : BufferRef{};
        const bool put = b && (s || !(head + tail));
        if (put) {
            uint32_t w;
            std::memcpy(&w, pattern + ((done + head) & 3), 4);
            if (mid) vkCmdFillBuffer(cmd, b.buffer, b.offset + head, mid, w);
            VkBufferCopy r[2];
            uint32_t nr = 0;
            if (head) r[nr++] = {s.offset, b.offset, head};
            if (tail) r[nr++] = {s.offset + head, b.offset + head + mid, tail};
            if (nr) vkCmdCopyBuffer(cmd, s.buffer, b.buffer, nr, r);
            const uint64_t guest = host->gpu_guest(at);
            if (guest && !in_stream) memory_stamp_gpu(guest, k);
        } else if (!in_stream) {   // (no import, a failed one, or a full stream: on the CPU)
            uint32_t buf[1025];
            for (auto& x : buf) x = value;
            for (uint64_t off = 0; off < k;) {
                const uint64_t m = std::min<uint64_t>(4096, k - off);
                if (!guest_cpu_write(at + off, (const uint8_t*)buf + (done & 3), m)) break;
                off += m;
            }
        }
        all = all && put;
        done += k;
    }
    return all;
}

// ---- uniform updates (UPDATE_UB): written at the end of their recording; ub_overlay applies them to copies ----
struct UbUpdate {
    uint64_t value;   // the submission that writes it
    uint64_t dst;
    std::vector<uint8_t> bytes;
};
std::deque<UbUpdate> ub_updates;   // in stream order, oldest first
std::atomic<uint64_t> ub_list_epoch{1};   // (bumped when that deque gains or loses an update)
thread_local uint64_t ub_gen = 0;   // (each recording thread's: its draws' overlays change with what it passed)

// ---- parallel replay (RX_VK_PARALLEL=<threads>): MKW hands over nearly a frame's draws in one command buffer with
// its present, and waits for the last frame's GPU work before it submits the next: the replay of that command buffer
// (6-10 ms) was on the frame's critical path. A pass over its ops on the replay thread keeps only the state, and
// what has to happen in stream order (counter reports and the render conditions that read them, uniform updates);
// segments of it are then recorded on several threads from the state there, each into a command buffer of its own,
// submitted in order. ----
enum class Mode { Serial, Dry, Worker };
thread_local Mode rmode = Mode::Serial;
thread_local size_t ub_limit = SIZE_MAX;   // (a worker: the uniform updates before its place in the stream)
thread_local size_t cond_at = 0;           // (a worker: the next render condition's result)
std::vector<uint8_t> par_conds;            // (the render conditions' results, in stream order: the dry pass's)
struct ParPoint {
    const uint8_t* p;
    NvnState s;
    size_t ub, cond;
    int64_t work;
};
std::vector<ParPoint> par_points;   // (every 64 draws and dispatches of the dry pass)
int64_t dry_work = 0;
// (D4, issue 3: the one-frame black world in Zora's Domain) The dry pass marked the targets a draw writes (dry_mark_written) but not
// those a clear writes. A pass that only clears its targets (two colour targets cleared and sampled by a later segment: the
// 1280 x 720 scene targets) was marked as the GPU's only when the segment that holds the clear recorded it, and a later segment
// that recorded first found the targets never written and uploaded stale guest memory over them, an upload that runs after the
// clear: the targets black for the frame. The log's "written (a pass's colour target) in segment N, uploaded by the later segment M"
// lines counted them (40 and more in a 4-minute run; none with RX_VK_PARALLEL=0). RX_VK_NO_DRY_CLEAR_MARKS=1: as before
bool dry_clear_marks() {
    static const bool on = !env_on("RX_VK_NO_DRY_CLEAR_MARKS");
    return on;
}
// (RX_VK_DRY_LEAN=1) the points whose state the dry pass keeps: only those the boundaries can be at (see
// lean_candidates in replay_handle); null: every point, as before
struct LeanPoints {
    int n = 0;
    int64_t work[2 * 8] = {};
    bool has(int64_t w) const {
        for (int i = 0; i < n; i++)
            if (work[i] == w) return true;
        return false;
    }
};
const LeanPoints* lean_points = nullptr;
int64_t st_lean_kept = 0, st_lean_left = 0;   // (the dry passes' states kept and not, RX_VK_DRY_LEAN)
// (RX_VK_PARALLEL_POINT=<n>: a point every n instead of 64, for finer boundaries; : 32 by default, =0: 64)
int64_t par_point_every() {
    static const int64_t v = getenv("RX_VK_PARALLEL_POINT")
        ? (atoi(getenv("RX_VK_PARALLEL_POINT")) <= 0 ? 64 : std::clamp(atoi(getenv("RX_VK_PARALLEL_POINT")), 8, 256)) : 32;
    return v;
}
// (RX_VK_SEG_EARLY) segments handed out while the dry pass still runs: the dry pass appends to ub_updates and par_conds
// while the workers read them (a deque's map and a vector's buffer may move as they grow). While ub_growing, the replay thread's
// appends hold ub_grow_mu exclusively and the workers' reads (ub_overlay, ub_read, a render condition) hold it shared; the
// workers read only entries appended before their segment was handed out (its end was passed), the lock is for the containers
uint64_t par_seq = 0;   // (the segments' order, across submissions: a recording's uploads precede it; texture_set_segment)
std::shared_mutex ub_grow_mu;
std::atomic<bool> ub_growing{false};
thread_local int ub_read_depth = 0;   // (a read inside a read holds the lock once)
struct UbReadGuard {
    std::shared_lock<std::shared_mutex> lk;
    UbReadGuard() {
        if (ub_read_depth++ == 0 && rmode == Mode::Worker && ub_growing.load(std::memory_order_acquire))
            lk = std::shared_lock<std::shared_mutex>(ub_grow_mu);
    }
    ~UbReadGuard() { ub_read_depth--; }
    UbReadGuard(const UbReadGuard&) = delete;
    UbReadGuard& operator=(const UbReadGuard&) = delete;
};
struct UbGrowGuard {
    std::unique_lock<std::shared_mutex> lk;
    UbGrowGuard() {
        if (ub_growing.load(std::memory_order_relaxed)) lk = std::unique_lock<std::shared_mutex>(ub_grow_mu);
    }
};
// (the dry pass calls on_point at each point it keeps while an early replay_handle runs; the replay thread's only)
struct SegEarlyHook {
    std::function<void(const ParPoint&)> on_point;
    /// whether on_point would take the point at this work count and place: the point's state (NvnState, ~27 KB) is copied
    /// only for those (RX_VK_DRY_WANTED, default on, =0: every point's, as before)
    std::function<bool(int64_t, const uint8_t*)> wants;
};
bool dry_wanted_only() {
    static const bool on = !getenv("RX_VK_DRY_WANTED") || atoi(getenv("RX_VK_DRY_WANTED")) != 0;
    return on;
}
SegEarlyHook* seg_early_hook = nullptr;
size_t ub_first_of_recording = 0;  // (the current recording's first entry)
uint64_t rec_value_now = 0;

void ub_prune(uint64_t done) {
    if (ub_updates.empty() || ub_updates.front().value > done) return;
    while (!ub_updates.empty() && ub_updates.front().value <= done) {
        ub_updates.pop_front();
        if (ub_first_of_recording) ub_first_of_recording--;
    }
    ub_list_epoch.fetch_add(1, std::memory_order_relaxed);
}

} // namespace
int64_t vk_value_submitted_ns(uint64_t v) {   // (0: not submitted yet, or the ring came round)
    return value_submit_v[v & 1023].load(std::memory_order_relaxed) == v ? value_submit_ns[v & 1023].load(std::memory_order_relaxed) : 0;
}

uint64_t ub_generation() {
    return ub_gen;
}
uint64_t ub_view(uint64_t* limit) {
    // (every append and prune moves the epoch, so the epoch and this thread's limit fix the list as ub_overlay reads it: its size and, with
    // RX_VK_UB_CUR_ONLY, which the memo leaves off, where it begins)
    *limit = ub_limit;
    return ub_list_epoch.load(std::memory_order_relaxed);
}
void ub_thread_off() {
    ub_limit = 0;
}
/// The open gap around [addr, addr+bytes) that no update covers. False when one does. [lo, hi) is that gap.
bool ub_gap(std::deque<UbUpdate>::const_iterator first, std::deque<UbUpdate>::const_iterator last,
            uint64_t addr, uint64_t bytes, uint64_t* lo, uint64_t* hi) {
    if (!bytes || addr + bytes < addr) return false;
    uint64_t left = 0, right = ~uint64_t{0};
    const uint64_t stop = addr + bytes;
    for (auto it = first; it != last; ++it) {
        const uint64_t start = it->dst, end = start + it->bytes.size();
        if (start < stop && addr < end) return false;
        if (end <= addr && end > left) left = end;
        if (start >= stop && start < right) right = start;
    }
    *lo = left;
    *hi = right;
    return true;
}
struct UbGapCache {
    uint64_t epoch = 0, lo = 0, hi = 0;
    size_t first = 0, lim = 0;
    bool on = false;
};
thread_local UbGapCache ub_gap_cache;
bool ub_gap_holds(uint64_t addr, uint64_t n, size_t first, size_t lim) {
    return n && ub_gap_cache.on && ub_gap_cache.epoch == ub_list_epoch.load(std::memory_order_relaxed) &&
           ub_gap_cache.first == first && ub_gap_cache.lim == lim && addr >= ub_gap_cache.lo &&
           addr + n >= addr && addr + n <= ub_gap_cache.hi;
}
bool ub_try_gap(size_t first, size_t lim, uint64_t addr, uint64_t n) {
    if (first > lim) return false;
    const uint64_t epoch = ub_list_epoch.load(std::memory_order_relaxed);
    // (a background thread's range is empty: it reads nothing of the deque, not even its start, while the replay
    // thread changes it)
    using It = std::deque<UbUpdate>::const_iterator;
    const It from = first == lim ? It{} : ub_updates.cbegin() + first, to = first == lim ? It{} : from + (lim - first);
    uint64_t lo = 0, hi = 0;
    if (!ub_gap(from, to, addr, n, &lo, &hi)) return false;
    if (ub_list_epoch.load(std::memory_order_relaxed) != epoch) return false;   // (the list changed: do not keep the gap)
    ub_gap_cache = {epoch, lo, hi, first, lim, true};
    return true;
}
// (a draw reads the same uniform words as the draw before it. The gap is what says later updates missed the word:
// it is kept while that gap covers it and its page is unstamped)
struct UbWord {
    uint64_t addr = 0, rec = 0;
    uint32_t page = 0, v = 0;
    bool on = false;
};
thread_local UbWord ub_words[128];
uint32_t ub_word_index(uint64_t addr) {
    // Words 512 bytes or a page apart share the low seven word bits. Add higher bits to separate those strides
    // while keeping runs of up to 64 words collision-free. Hits still compare the full address in ub_word_hit.
    const uint64_t word = addr >> 2;
    return (uint32_t)((word + (word >> 7) + (word >> 14)) & 127);
}
bool ub_word_hit(uint64_t addr, size_t lim, uint32_t* out) {
    UbWord& w = ub_words[ub_word_index(addr)];
    if (!w.on || w.addr != addr || w.rec != rec_value_now || w.page != memory_page_stamp(addr)) return false;
    if (!ub_gap_holds(addr, 4, 0, lim)) return false;
    *out = w.v;
    return true;
}
// (a range an update does overlap: the stream copy stays while the gap's update list is unchanged and the copy is
// still in the recording's stream. A range in the gap has no copy)
struct UbCopy {
    const uint8_t* base = nullptr;
    uint64_t addr = 0, n = 0, at = 0, rec = 0, view = 0, epoch = 0;
    size_t lim = 0, first = 0;
    bool on = false;
};
thread_local UbCopy ub_copies[8];
bool ub_copy_hit(uint64_t addr, uint64_t n, const uint8_t* base, size_t first, size_t lim, uint64_t* at) {
    UbCopy& c = ub_copies[((addr >> 4) ^ n) & 7];
    if (!c.on || c.base != base || c.addr != addr || c.n != n || c.first != first || c.lim != lim ||
        c.rec != rec_value_now || c.epoch != ub_list_epoch.load(std::memory_order_relaxed) ||
        c.view != memory_capture_view())
        return false;
    if (!stream_holds(base, c.at, n)) return false;
    *at = c.at;
    return true;
}
void ub_copy_keep(uint64_t addr, uint64_t n, const uint8_t* base, size_t first, size_t lim, uint64_t at,
                  uint64_t view, uint64_t epoch) {
    if (view != memory_capture_view()) return;
    if (epoch != ub_list_epoch.load(std::memory_order_relaxed)) return;
    UbCopy& c = ub_copies[((addr >> 4) ^ n) & 7];
    c = {};
    c.on = true, c.base = base, c.addr = addr, c.n = n, c.at = at, c.first = first, c.lim = lim;
    c.rec = rec_value_now, c.view = view, c.epoch = epoch;
}
uint64_t ub_overlay(uint64_t addr, uint64_t n, uint8_t* base, uint64_t (*alloc)(uint64_t n, uint64_t align), uint64_t align) {
    UbReadGuard ug;   // (the list may grow meanwhile, RX_VK_SEG_EARLY)
    // (a worker: those before its place in the stream; a background thread: none, not even the list's size)
    const size_t lim = ub_limit ? std::min(ub_limit, ub_updates.size()) : 0;
    static const bool cur_only = env_on("RX_VK_UB_CUR_ONLY");   // (A/B: this submission's updates only)
    const size_t first = cur_only ? std::min(ub_first_of_recording, lim) : 0;
    if (ub_gap_holds(addr, n, first, lim)) return UINT64_MAX;
    const uint64_t view = memory_capture_view();
    uint64_t kept = 0;
    if (ub_copy_hit(addr, n, base, first, lim, &kept)) return kept;
    if (ub_try_gap(first, lim, addr, n)) return UINT64_MAX;
    const uint64_t epoch = ub_list_epoch.load(std::memory_order_relaxed);
    bool any = false;
    for (size_t i = first; i < lim; i++) {
        const UbUpdate& u = ub_updates[i];
        if (u.dst < addr + n && addr < u.dst + u.bytes.size()) {
            any = true;
            break;
        }
    }
    if (!any) return UINT64_MAX;
    const uint64_t at = alloc(n, align);
    if (at == UINT64_MAX) return UINT64_MAX;
    uint8_t* p = base + at;
    if (!memory_read_as_submitted(addr, p, n)) std::memset(p, 0, n);
    for (size_t i = first; i < lim; i++) {
        const UbUpdate& u = ub_updates[i];
        const uint64_t lo = std::max(addr, u.dst), hi = std::min(addr + n, u.dst + u.bytes.size());
        if (lo < hi) std::memcpy(p + (lo - addr), u.bytes.data() + (lo - u.dst), hi - lo);
    }
    st_ub_overlays++;
    ub_copy_keep(addr, n, base, first, lim, at, view, epoch);
    return at;
}

/// (HL_UBO_INLINE) the stream's uniform updates before this place over [addr, addr + n), copied onto p (an inline block's
/// bytes): what ub_overlay puts over the bytes as submitted
void ub_overlay_onto(uint64_t addr, uint64_t n, uint8_t* p) {
    UbReadGuard ug;
    const size_t lim = ub_limit ? std::min(ub_limit, ub_updates.size()) : 0;
    static const bool cur_only = env_on("RX_VK_UB_CUR_ONLY");
    const size_t first = cur_only ? std::min(ub_first_of_recording, lim) : 0;
    if (ub_gap_holds(addr, n, first, lim) || ub_try_gap(first, lim, addr, n)) return;
    for (size_t i = first; i < lim; i++) {
        const UbUpdate& u = ub_updates[i];
        const uint64_t lo = std::max(addr, u.dst), hi = std::min(addr + n, u.dst + u.bytes.size());
        if (lo < hi) std::memcpy(p + (lo - addr), u.bytes.data() + (lo - u.dst), hi - lo);
    }
}

/// Guest memory as the stream's place sees it: as it was at the submission (the capture), with the uniform updates
/// the stream made before this place (a worker: before its segment) over it. For what the replay reads on the CPU for
/// the GPU's sake (handles in uniform and storage buffers): the game's memory now may be its next frames'.
bool ub_read(uint64_t addr, void* dst, uint64_t n) {
    const size_t lim = ub_limit ? std::min(ub_limit, ub_updates.size()) : 0;
    if (n == 4) {
        uint32_t v = 0;
        if (ub_word_hit(addr, lim, &v)) {
            std::memcpy(dst, &v, 4);
            return true;
        }
    }
    const uint32_t page = n == 4 ? memory_page_stamp(addr) : 0;
    if (!memory_read_as_submitted(addr, dst, n)) return false;
    // (most words are not inside an inline update: the scan that finds that is kept for the gap, and the next words
    // in the gap do not walk the list. A word kept from that gap is the capture's own bytes)
    if (ub_gap_holds(addr, n, 0, lim) || ub_try_gap(0, lim, addr, n)) {
        if (n == 4 && page == memory_page_stamp(addr)) {
            UbWord& w = ub_words[ub_word_index(addr)];
            uint32_t v = 0;
            std::memcpy(&v, dst, 4);
            w = {};
            w.on = true, w.addr = addr, w.v = v, w.rec = rec_value_now, w.page = page;
        }
        return true;
    }
    for (size_t i = 0; i < lim; i++) {
        const UbUpdate& u = ub_updates[i];
        const uint64_t lo = std::max(addr, u.dst), hi = std::min(addr + n, u.dst + u.bytes.size());
        if (lo < hi) std::memcpy((uint8_t*)dst + (lo - addr), u.bytes.data() + (lo - u.dst), hi - lo);
    }
    return true;
}

namespace {

/// (hookshot) Uniform updates on their way into one import's buffer as ONE copy from the recording's stream: its regions, and
/// the destination's bytes they hold (sorted, none overlapping another: a copy's regions may not overlap in the destination,
/// and two updates of the same bytes have to land in stream order)
struct UbBatch {
    VkBuffer src = VK_NULL_HANDLE, dst = VK_NULL_HANDLE;
    std::vector<VkBufferCopy> regions;
    std::vector<std::pair<uint64_t, uint64_t>> held;   // ([lo, hi) of dst, by lo)
};
/// Whether k bytes at `at` of dst, staged at `from` of src, can join the batch: the same two buffers (another import may hold
/// the same guest memory: its writes stay behind a barrier), and no byte an earlier region of the batch writes. Joined when true
bool ub_batch_add(UbBatch& b, VkBuffer src, VkBuffer dst, uint64_t from, uint64_t at, uint64_t k) {
    if (!b.regions.empty() && (b.src != src || b.dst != dst)) return false;
    const auto next = std::lower_bound(b.held.begin(), b.held.end(), std::pair<uint64_t, uint64_t>{at, 0});
    if (next != b.held.end() && next->first < at + k) return false;
    if (next != b.held.begin() && std::prev(next)->second > at) return false;
    b.held.insert(next, {at, at + k});
    b.src = src, b.dst = dst;
    b.regions.push_back({from, at, k});
    return true;
}
/// The batch recorded: one barrier, then its regions as one copy (what came before it in the stream has landed)
void ub_batch_emit(VkCommandBuffer cmd, UbBatch& b) {
    if (b.regions.empty()) return;
    full_barrier(cmd);   // overlapping inline writes must reach memory in stream order
    vkCmdCopyBuffer(cmd, b.src, b.dst, (uint32_t)b.regions.size(), b.regions.data());
    b.regions.clear(), b.held.clear();
}

/// The recording's uniform updates into guest memory (after its draws: the ones that came after an update read
/// their overlay copies). (hookshot) Batched: each update's bytes go into the recording's stream, and the updates that
/// write different bytes of one import land as one copy of many regions behind one barrier. A batch ends, and the next
/// starts behind a barrier of its own, where an update writes bytes an earlier one of the batch wrote, or goes to another
/// buffer: stream order, as when each update had its own barrier and vkCmdUpdateBuffer (80 a frame: 0.33 ms of GPU time).
/// One with no room in the stream is written as before (gpu_put), after the batch before it; bytes no import holds on the CPU
void flush_ub_updates(VkCommandBuffer cmd) {
    if (ub_first_of_recording >= ub_updates.size()) return;
    memory_writer_note(kRbWriterUb, 0);   // (RX_VK_LOG_RB_WRITERS: what these write is the uniform updates')
    GprofSpan gp(cmd, "ubupd (the stream's uniform updates -> guest memory, with its barriers)");
    gp.items = (int)(ub_updates.size() - ub_first_of_recording);
    UbBatch batch;
    for (size_t i = ub_first_of_recording; i < ub_updates.size(); i++) {
        uint64_t gpu = ub_updates[i].dst, n = ub_updates[i].bytes.size();
        const uint8_t* d = ub_updates[i].bytes.data();
        gp.bytes += n;
        while (n) {
            uint64_t k = std::min<uint64_t>(n, 65536);
            const BufferRef b = gpu_piece_bind(gpu, k);
            const BufferRef s = b ? stream_copy(d, k) : BufferRef{};
            if (s) {
                if (!ub_batch_add(batch, s.buffer, b.buffer, s.offset, b.offset, k)) {
                    ub_batch_emit(cmd, batch);
                    ub_batch_add(batch, s.buffer, b.buffer, s.offset, b.offset, k);
                }
            } else if (b) {   // (a full stream: after what the batch holds, as an update where it can be one)
                ub_batch_emit(cmd, batch);
                if (!gpu_put(cmd, b, d, k)) guest_cpu_write(gpu, d, k);
            } else {
                guest_cpu_write(gpu, d, k);   // (no import, or a failed one: on the CPU, as gpu_write)
            }
            gpu += k, d += k, n -= k;
        }
    }
    ub_batch_emit(cmd, batch);
    full_barrier(cmd);
}

/// (mesh shading: the stages its resources are bound to, beyond NVN's six)
void log_stage_bind(uint32_t op, int stage) {
    static int logged[64];
    if (stage >= 0 && stage < 64 && logged[stage]++ < 3) logf("vk: record %u binds to stage %d", op, stage);
}

void maybe_split(VkCommandBuffer& cmd);
bool occ_sync(VkCommandBuffer& cmd);
// (RX_VK_PRESYNC_EARLY) the dry pass's draw-and-dispatch count at which the mirrors' pre-sync goes to a worker (0: none)
int64_t presync_kick_at = 0;
void presync_early_start();
// (hookshot) A part's end that waits for the next plain draw (maybe_split, split_part): see part_ends_before_draw
void split_part(VkCommandBuffer& cmd);
bool part_end_wanted = false;
void early_split(VkCommandBuffer& cmd);   // (RX_VK_READBACK_EARLY: a part's end after a readback's writer)
bool early_split_pending = false;   // (RX_VK_READBACK_EARLY: a writer of a waited-for range is in the part being recorded)
extern int64_t part_work;
extern uint64_t rec_cur_value;   // (the submission being recorded: the occlusion queries' value)

/// The handles already stored, so a rebind of the same ones does not rebuild descriptors
bool handles_unchanged(const uint64_t* have, const uint64_t* next, int n) {
    if (n < 0) return false;
    for (int i = 0; i < n; i++)
        if (have[i] != next[i]) return false;
    return true;
}
/// A uniform or storage binding already stored
bool buf_unchanged(int32_t stage, int32_t index, uint64_t addr, uint64_t size, int32_t nstage, int32_t nindex,
                   uint64_t naddr, uint64_t nsize) {
    return stage == nstage && index == nindex && addr == naddr && size == nsize;
}

// ---- (stream R4) RX_VK_PROF_FINE=1: what the replay thread's own time inside submissions is made of ----
// "a frame (fine)" (vk_draw.cpp) sums the draw's sections over every recording thread; the parallel replay spreads them over up to five.
// These are the wall clock of the thread that holds the replay lock, per frame (every 120): the submissions (and those with 300 draws
// or more, the frame's heavy one in its pieces), the command buffers' replay_handle (its dry pass, its segments' wait, the command
// buffers recorded whole on this thread), the memory work before a submission, the recording's end
struct ReplayWall {
    int64_t submit_ns = 0, submits = 0, draws = 0;
    int64_t heavy_ns = 0, heavy_n = 0, heavy_draws = 0;
    int64_t handle_ns = 0, handles = 0;
    int64_t dry_ns = 0, seg_ns = 0, big = 0, whole_ns = 0, whole = 0;
    int64_t mem_ns = 0, finish_ns = 0;
    int64_t early_ns[10] = {};   // (replay_handle_early's steps, EarlyWall)
};
ReplayWall rwall;
bool wall_fine_on() {
    static const bool v = env_on("RX_VK_PROF_FINE");
    return v;
}
inline int64_t wall_now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct WallAdd {   // (adds its scope's wall time to a counter; nothing when the switch is off)
    int64_t& acc;
    int64_t t0;
    explicit WallAdd(int64_t& a) : acc(a), t0(wall_fine_on() ? wall_now() : 0) {}
    ~WallAdd() { if (t0) acc += wall_now() - t0; }
    WallAdd(const WallAdd&) = delete;
    WallAdd& operator=(const WallAdd&) = delete;
};
/// replay_handle_early's steps, each timed to the mark at its end (marks 1 to 8 in the order they come; the destructor, mark 9, adds
/// each gap to the step of the mark that closes it, the marks a path does not reach left out)
struct EarlyWall {
    int64_t at[10] = {};
    EarlyWall() { if (wall_fine_on()) at[0] = wall_now(); }
    void mark(int i) { if (at[0]) at[i] = wall_now(); }
    ~EarlyWall() {
        if (!at[0]) return;
        at[9] = wall_now();
        int64_t prev = at[0];
        for (int i = 1; i < 10; i++)
            if (at[i]) rwall.early_ns[i] += at[i] - prev, prev = at[i];
    }
};// (and the ops of the stream by kind: how many and how long each took in the dry pass, in a segment's recording and in a command buffer
// recorded whole on one thread; an op's time runs to the start of the next, so a draw's includes the loop's own turn)
constexpr int kOpKinds = 64;
std::atomic<int64_t> op_cost_ns[3][kOpKinds], op_cost_n[3][kOpKinds];   // ([0] the dry pass, [1] a segment's worker, [2] the whole thread)
const char* const kOpNames[kOpKinds] = {
    "end", "jump", "begin", "call", "signal event", "report counter", "copy b2b", "update ub", "set rt", "clear color", "clear ds", "viewports",
    "scissor", "depth range", "bind program", "bind state", "bind vattr", "bind vstream", "bind vbuf", "bind ubo", "bind ssbo", "bind tex",
    "bind samp", "stencil", "poly offset", "draw", "draw idx", "dispatch", "barrier", "set tex pool", "set samp pool", "copy b2t",
    "copy stride", "blend color", "copy t2b", "bind img", "copy t2t", "reset counter", "clear tex", "depth clamp", "render enable",
    "raster discard", "render cond", "primitive restart", "bind bufs", "draw indirect", "dispatch indirect", "draw mesh", "wait event",
    "push const", "clear buf", "ubo check", "ubo inline"};
struct OpTimer {   // (one replaycall's: the sums go to the shared counters when it returns)
    bool on;
    int mode;
    int64_t t_prev = 0;
    uint32_t prev = 0;
    int64_t ns[kOpKinds] = {}, n[kOpKinds] = {};
    OpTimer(bool dry, bool worker) : on(wall_fine_on()), mode(dry ? 0 : worker ? 1 : 2) {}
    void at(uint32_t op) {   // (the next op begins)
        const int64_t now = prof_now();
        if (t_prev && prev < kOpKinds) ns[prev] += now - t_prev, n[prev]++;
        t_prev = now, prev = op;
    }
    ~OpTimer() {
        if (!on) return;
        at(0);
        for (int i = 0; i < kOpKinds; i++)
            if (n[i]) op_cost_ns[mode][i].fetch_add(ns[i], std::memory_order_relaxed), op_cost_n[mode][i].fetch_add(n[i], std::memory_order_relaxed);
    }
};
void op_cost_report(double frames) {
    static int64_t last_ns[3][kOpKinds], last_n[3][kOpKinds];
    static const char* const where[3] = {"dry pass", "segment workers", "whole on one thread"};
    for (int m = 0; m < 3; m++) {
        struct Row { int op; double ms, n, us; };
        std::vector<Row> rows;
        double all = 0;
        for (int i = 0; i < kOpKinds; i++) {
            const int64_t ns = op_cost_ns[m][i].load(std::memory_order_relaxed), n = op_cost_n[m][i].load(std::memory_order_relaxed);
            const int64_t dn = ns - last_ns[m][i], dc = n - last_n[m][i];
            last_ns[m][i] = ns, last_n[m][i] = n;
            if (dc <= 0) continue;
            rows.push_back({i, dn / frames / 1e6, dc / frames, dn / 1e3 / (double)dc});
            all += dn / frames / 1e6;
        }
        if (rows.empty()) continue;
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ms > b.ms; });
        char line[900];
        int k = snprintf(line, sizeof line, "vk: a frame (fine, ops of the %s): %.2f ms in all:", where[m], all);
        for (size_t i = 0; i < rows.size() && i < 9 && k < (int)sizeof line - 90; i++)
            k += snprintf(line + k, sizeof line - k, " %s %.2f ms (%.0f, %.2f us)%s", kOpNames[rows[i].op], rows[i].ms, rows[i].n, rows[i].us,
                          i + 1 < rows.size() && i < 8 ? "," : "");
        logf("%s", line);
    }
}void replay_wall_report(double frames) {
    if (!wall_fine_on()) return;
    static int64_t last_early[10];
    {
        static const char* const step[10] = {"", "memory sync before", "readback refresh", "segment recordings begun", "init buffers begun",
                                             "dry loop", "what came before submitted", "its own segment", "waits and submits", "the end"};
        char line[500];
        int k = snprintf(line, sizeof line, "vk: a frame (fine, replay_handle_early's steps, ms):");
        for (int i = 1; i < 10; i++)
            k += snprintf(line + k, sizeof line - k, " %s %.2f%s", step[i], (rwall.early_ns[i] - last_early[i]) / frames / 1e6, i < 9 ? "," : "");
        for (int i = 0; i < 10; i++) last_early[i] = rwall.early_ns[i];
        logf("%s", line);
    }
    static ReplayWall last;
    const ReplayWall& n = rwall;
    const auto ms = [&](int64_t a, int64_t b) { return (a - b) / frames / 1e6; };
    const auto per = [&](int64_t a, int64_t b) { return (a - b) / frames; };
    logf("vk: a frame (fine, replay thread, wall): submissions %.2f ms (%.1f, %.0f draws); of them with 300 draws or more %.2f ms (%.2f, "
         "%.0f draws); replay_handle %.2f ms (%.1f): dry pass %.2f, segments' recording and wait %.2f (%.2f big), recorded whole here %.2f "
         "(%.1f); memory work %.2f, recordings ended %.2f",
         ms(n.submit_ns, last.submit_ns), per(n.submits, last.submits), per(n.draws, last.draws), ms(n.heavy_ns, last.heavy_ns),
         per(n.heavy_n, last.heavy_n), per(n.heavy_draws, last.heavy_draws), ms(n.handle_ns, last.handle_ns), per(n.handles, last.handles),
         ms(n.dry_ns, last.dry_ns), ms(n.seg_ns, last.seg_ns), per(n.big, last.big), ms(n.whole_ns, last.whole_ns), per(n.whole, last.whole),
         ms(n.mem_ns, last.mem_ns), ms(n.finish_ns, last.finish_ns));
    last = n;
    op_cost_report(frames);
}

void ubo_check_refs(uint32_t op, const uint8_t* a, uint32_t avail);   // (HL_UBO_CHECK: below)

/// A draw or a dispatch: what a segment is a count of (dry_work, RX_VK_PARALLEL_SEG)
bool op_is_work(uint32_t op) {
    return op == NOP_DRAW || op == NOP_DRAW_IDX || op == NOP_DRAW_INDIRECT || op == NOP_DRAW_MESH || op == NOP_DISPATCH ||
           op == NOP_DISPATCH_INDIRECT;
}

void replay(VkCommandBuffer& cmd, NvnState& s, const uint8_t* p, const uint8_t* end = nullptr) {
    const Mode mode = rmode;
    const bool dry = mode == Mode::Dry, worker = mode == Mode::Worker;
    OpTimer op_timer(dry, worker);
    // (RX_VK_LOG_RB_WRITERS, RX_VK_READBACK_EARLY: each op said, as the writer of what it writes)
    static const bool rb_writers = rb_writers_on(), early = readback_early_on(), note_ops = rb_writers || early;
    for (;;) {
        if (p == end) return;
        nvn_op_hdr h;
        std::memcpy(&h, p, sizeof h);
        if (op_timer.on) op_timer.at(h.op);
        if (note_ops && !dry) memory_writer_note(h.op, 0);
        bool split_after = false;   // (RX_VK_READBACK_EARLY: an update of uniforms the CPU reads back: its part ends after it)
        const uint8_t* a = p + 8;
        const uint32_t avail = h.size - 8;
        if (!worker && ubo_check_on()) ubo_check_refs(h.op, a, avail);
        // (the part's end maybe_split asked for: here, right before a draw with a vertex stage that will be recorded, so that
        // the next part begins with it and not with a mesh draw or a dispatch)
        if (!dry && !worker && part_end_wanted && (h.op == NOP_DRAW || h.op == NOP_DRAW_IDX || h.op == NOP_DRAW_INDIRECT) &&
            s.render_enabled && !s.raster_discard)
            split_part(cmd);
        switch (h.op) {
        case NOP_END: return;
        case NOP_BEGIN:
            if (!worker && ubo_check_on()) ubo_check_begin_cmd();
            // (HL_UBO_INLINE: an inline slot's bytes live in the recording that bound them, which may be freed once its
            // submission is replayed; a later recording that draws without binding the slot again reads it by address,
            // as before. Every mode: a worker's segment may cross into the next recording)
            for (int st = 0; st < kNvnStages; st++)
                for (int i = 0; i < 16; i++)
                    if (s.ubo_bytes[st][i]) s.ubo_bytes[st][i] = nullptr, s.ubo_bytes_n[st][i] = 0, s.bind_gen++;
            break;
        case NOP_JUMP: case NOP_CALL: break;
        case NOP_UBO_CHECK: {   // (HL_UBO_CHECK: vk_ubo_check.cpp; the replay thread's pass only, each record once)
            if (worker || !ubo_check_on()) break;
            const auto k = payload<nvn_p_ubo_check>(a, avail);
            bool updated = false;
            for (const UbUpdate& u : ub_updates)
                if (u.dst < k.addr + k.size && k.addr < u.dst + u.bytes.size()) { updated = true; break; }
            ubo_check_bind(k, updated);
            break;
        }
        // ---- memory the stream writes: on the GPU timeline ----
        case NOP_SIGNAL_EVENT: {
            if (!worker) {   // (in stream order: the replay thread's pass)
                const auto e = payload<nvn_p_signal>(a, avail);
                ev_cpu_apply();
                auto it = ev_values.find(e.gpu);
                uint32_t old = it != ev_values.end() ? it->second : 0;
                if (it == ev_values.end()) host->gpu_read(e.gpu, &old, 4);
                ev_values[e.gpu] = e.mode == NVN_EVENT_SIGNAL_MODE_ADD ? old + e.value : e.value;
            }
            if (dry) break;
            const auto e = payload<nvn_p_signal>(a, avail);
            uint32_t v = e.value;
            if (e.mode == NVN_EVENT_SIGNAL_MODE_ADD) {   // (the add on the CPU's view: events are few and in order)
                uint32_t old = 0;
                host->gpu_read(e.gpu, &old, 4);
                v = old + e.value;
            }
            end_pass(cmd);
            GprofSpan gp(cmd, "event signal (barrier, write, barrier)");
            full_barrier(cmd);
            gpu_write(cmd, e.gpu, &v, 4);
            break;
        }
        case NOP_WAIT_EVENT: {   // (a wait for an event another queue's work signals: submit_job orders them)
            if (worker) break;
            const auto w = payload<nvn_p_wait_event>(a, avail);
            ev_cpu_apply();
            const uint32_t have = event_value_in(ev_values, w.gpu);
            if (event_wait_met(have, w.mode, w.value)) {
                // (hookshot) met by a signal of the CPU's that no capture follows: stopped here once, for resume_queue's
                if (suspend_ok && p != met_resumed_at && met_wait_capture_on() && cpu_signal_uncaptured(w.gpu)) {
                    st_met_stops++;
                    suspend_at = p;
                    return;
                }
                st_waits_met++;
            } else if (suspend_ok) {   // (on once that work is replayed; a dry pass: its part up to here)
                suspend_at = p;
                return;
            } else {
                st_waits_unmet++;
                static int logged;
                if (logged++ < 20 || st_waits_unmet % 5000 == 0)
                    logf("vk: a wait not met in replay order: event %#llx wants %s %u, has %u (submit %llu; %lld met, %lld "
                         "not)", (unsigned long long)w.gpu, w.mode == 0 ? "==" : ">=", w.value, have,
                         (unsigned long long)st_submits, (long long)st_waits_met, (long long)st_waits_unmet);
            }
            break;
        }
        case NOP_REPORT_COUNTER: {
            const auto c = payload<nvn_p_counter>(a, avail);
            // (hookshot) the run of occlusion queries this report counts (RX_VK_OCC_QUERIES), closed in every mode: a worker
            // numbers its queries as the dry pass did
            const OccRun occ_run = occ_close(s, (uint32_t)c.type == NVN_COUNTER_TYPE_SAMPLES_PASSED);
            if (worker) break;   // (written by the dry pass, in stream order)
            const bool flag_written = (c.type & NVN_COUNTER_WRITTEN) != 0;   // (a clear flag: nvn_native.c wrote it as it was recorded)
            const bool samples = (uint32_t)c.type == NVN_COUNTER_TYPE_SAMPLES_PASSED;
            // (timestamps on the guest tick clock in counter ticks (19.2 MHz, nvnDeviceGetTimestampInNanoseconds
            // converts), the replay's time; occlusion counts grow with every draw, as in the WebGPU renderer)
            // doc: docs/design.md#4-the-lessons-from-mkw-as-rules-here (rule 4: GPU timestamps drive dynamic resolution)
            // (the GPU's time as the game measures it: from the frame's present on, at RX_VK_TS_SCALE (default 0.1)
            // times the real clock. The reports are written as the replay records them, so the real clock gave the
            // replay's time and its lag, not the GPU's: MKW's dynamic resolution moved every frame (each new size a
            // new set of ~20 render targets, re-registered textures, 20-25 ms of replay a frame) and kept low. The
            // GPU takes 6-7 ms a frame at the resolution it then holds. RX_VK_TS_SCALE=1: the real clock;
            // RX_VK_TS_GLOBAL=1: scaled from the start, not from each present.)
            static const double ts_scale = getenv("RX_VK_TS_SCALE") ? atof(getenv("RX_VK_TS_SCALE")) : 0.1;
            static const bool ts_global = env_on("RX_VK_TS_GLOBAL");
            static uint64_t ts_base = 0;
            uint64_t ns = host->ticks() * 32;
            if (ts_scale != 1.0) {
                const uint64_t base = ts_global ? (ts_base ? ts_base : (ts_base = ns)) : std::min(ns, ts_anchor_ns ? ts_anchor_ns : ns);
                ns = base + (uint64_t)((double)(ns - base) * ts_scale);
            }
            // (hookshot) Every samples-passed report reads 4096: everything visible. The count of 16 a draw since the reset went
            // up and down from frame to frame, and OoT's sun visibility followed it: for one frame in ~70 the sunlight left the
            // ground and the haze brightened over the whole picture (one-frame flicker, most of it at load-in; 30-52 per 2,700
            // frames in Kokiri, 4-6 with this, RX 7900 GRE). RX_VK_OCC_FIXED=<n>: that value; -1: the per-draw count as before
            static const int64_t occ_fixed = getenv("RX_VK_OCC_FIXED") ? atoll(getenv("RX_VK_OCC_FIXED")) : 4096;
            uint64_t count = samples ? (occ_fixed >= 0 ? (uint64_t)occ_fixed : s.samples_passed) : 0;
            // (RX_VK_OCC_REAL: measured beside it, and with modes 2 and 3 the last measured count of this address)
            if (samples && !flag_written && occ_mode()) count = occ_report(c.addr, occ_win_lo, occ_serial, rec_value_now, count, s.has_prog ? s.prog.serial : 0);
            const uint64_t rec[2] = {count, ns * 12 / 625};
            if (dump_on())
                logf("vk: dump-op %03d REPORT_COUNTER type %d at %#llx: %llu", dump_seq(), (int)c.type,
                     (unsigned long long)c.addr, (unsigned long long)rec[0]);
            // (written now, in stream order, not on the GPU timeline: SetRenderEnableConditional reads these words at
            // replay time, and MKW gates clears with them as TOTK does (the CPU writes 1 to ask for a clear, the report
            // after the clear resets it). A GPU-timeline write left the 1 in place for the stream's later conditions on
            // the same words, depending on how far the GPU was: extra clears of the HDR target, black frames at random;
            // and its late write could undo the game's next request. No pass ends for it either.)
            if (flag_written) guest_cpu_write(c.addr + 8, rec + 1, 8);
            else guest_cpu_write(c.addr, rec, sizeof rec);
            memory_cpu_owned(host->gpu_guest(c.addr), sizeof rec);   // (vk_memory.cpp: kept out of the mirrors' write-backs)
            // (hookshot) the GPU's count over it once its queries are done (occ_collect); a run whose count is not known is only
            // freed. The slot's report sequence moves on with every report
            occ_reported(occ_run, c.addr, rec[1], flag_written, rec_cur_value);
            break;
        }
        case NOP_RESET_COUNTER: {
            const auto c = payload<nvn_p_counter>(a, avail);
            if (c.type == NVN_COUNTER_TYPE_SAMPLES_PASSED) {
                s.samples_passed = 0;
                // (hookshot) a run of occlusion queries begins (RX_VK_OCC_QUERIES); one still open had no report: its queries are freed
                occ_counter_reset(s, worker, rec_cur_value);
            }
            break;
        }
        case NOP_COPY_B2B: {
            if (dry) break;
            ParLock lk;
            const auto c = payload<nvn_p_copy_bb>(a, avail);
            BufferRef src = gpu_buffer_bind(c.src, c.size, false), dst = gpu_buffer_bind(c.dst, c.size, true);
            {   // (RX_VK_WATCH_RANGE=<lo>:<hi>: buffer copies into it or out of it, 400 lines)
                static const std::pair<uint64_t, uint64_t> range = [] {
                    unsigned long long lo = 0, hi = 0;
                    if (const char* e = getenv("RX_VK_WATCH_RANGE")) sscanf(e, "%llx:%llx", &lo, &hi);
                    return std::pair<uint64_t, uint64_t>{lo, hi};
                }();
                static int logged;
                const bool hit = range.second && ((c.dst < range.second && c.dst + c.size > range.first) ||
                                                  (c.src < range.second && c.src + c.size > range.first));
                if (hit && logged++ < 400)
                    logf("vk: watch range buffer copy %#llx -> %#llx (%#llx bytes): source %s, destination %s", (unsigned long long)c.src,
                         (unsigned long long)c.dst, (unsigned long long)c.size, src ? "bound" : "NOT bound",
                         dst ? "bound" : "NOT bound (on the CPU)");
            }
            end_pass(cmd);
            GprofSpan gp(cmd, "copy buffer->buffer (with its barriers)");
            gp.bytes = c.size;
            full_barrier(cmd);
            if (src && dst) {
                const VkBufferCopy r{src.offset, dst.offset, c.size};
                vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &r);
                const uint64_t guest = host->gpu_guest(c.dst);
                if (guest) memory_stamp_gpu(guest, c.size);
            } else {   // (outside the pools: on the CPU)
                for (uint64_t done = 0; done < c.size;) {
                    uint8_t buf[65536];
                    const uint64_t k = std::min<uint64_t>(sizeof buf, c.size - done);
                    if (!host->gpu_read(c.src + done, buf, k) || !guest_cpu_write(c.dst + done, buf, k)) break;
                    done += k;
                }
            }
            full_barrier(cmd);
            break;
        }
        case NOP_CLEAR_BUF: {   // (hookshot) nvnCommandBufferClearBuffer: on the GPU in order, as a fill (gpu_fill)
            const auto c0 = payload<nvn_p_clear_buf>(a, avail);
            // (the small ones are the counters the reports and conditions use, which happen here on the CPU in stream order. Left
            // to the GPU, a clear earlier in the stream reached guest memory after a later report and wiped it, as far as the GPU
            // was behind: lights and whole rooms switched off by their conditions for a frame. RX_VK_CLEAR_CPU_MAX=0: on the GPU)
            static const uint64_t cpu_max = getenv("RX_VK_CLEAR_CPU_MAX") ? (uint64_t)atoll(getenv("RX_VK_CLEAR_CPU_MAX")) : 256;
            const bool in_stream = c0.size && c0.size <= cpu_max;
            if (in_stream && !worker) {
                uint32_t words[64];
                for (auto& w : words) w = c0.value;
                guest_cpu_write(c0.dst, words, std::min<uint64_t>(c0.size, sizeof words));   // (hookshot: not past words)
            }
            if (dry) break;
            ParLock lk;
            const auto c = c0;
            if (!c.size) break;
            end_pass(cmd);
            GprofSpan gp(cmd, "clear buffer (with its barriers)");
            gp.bytes = c.size;
            full_barrier(cmd);
            const bool on_gpu = gpu_fill(cmd, c.dst, c.size, c.value, in_stream);
            full_barrier(cmd);
            {   // (EvilOOT: RX_VK_WATCH_RANGE, the clears into it too)
                unsigned long long lo = 0, hi = 0;
                static const char* e = getenv("RX_VK_WATCH_RANGE");
                if (e) sscanf(e, "%llx:%llx", &lo, &hi);
                static int logged;
                if (hi && c.dst < hi && c.dst + c.size > lo && logged++ < 400)
                    logf("vk: watch range buffer clear %#llx (%#llx bytes) to %#x: %s", (unsigned long long)c.dst,
                         (unsigned long long)c.size, (unsigned)c.value, on_gpu ? "on the GPU" : "NOT all on the GPU");
            }
            break;
        }
        case NOP_COPY_B2T: if (!dry) { ParLock lk; copy_b2t(cmd, s, payload<nvn_p_copy_bt>(a, avail), true); } break;
        case NOP_COPY_T2B: if (!dry) { ParLock lk; copy_b2t(cmd, s, payload<nvn_p_copy_bt>(a, avail), false); } break;
        case NOP_COPY_T2T: {
            ParLock lk;
            if (dry) dry_mark_copy_target(payload<nvn_p_copy_tt>(a, avail));
            else copy_t2t(cmd, s, payload<nvn_p_copy_tt>(a, avail));
            break;
        }
        case NOP_CLEAR_TEX: if (!dry) { ParLock lk; clear_tex(cmd, s, payload<nvn_p_clear_tex>(a, avail)); } break;
        case NOP_UPDATE_UB: {
            const auto u = payload<nvn_p_update_ub>(a, avail);
            if (worker) {   // (appended by the dry pass: the draws after it read it from here on)
                // (hookshot) counted only when the dry pass appended it (the same test as below): an update of no bytes, or of more
                // than the stream holds, is not in ub_updates, and counting it moved every later draw of the segment one update ahead
                if (u.size && u.size <= avail - sizeof u) ub_limit++, ub_gen++;
                break;
            }
            if (dump_on() && u.size && u.size <= avail - sizeof u) {   // (in a dump: each update, its first words)
                float f[12] = {};
                std::memcpy(f, a + sizeof u, std::min<size_t>(sizeof f, u.size));
                logf("vk: dump-op %03d UPDATE_UB %#llx +%#x: %.5g %.5g %.5g %.5g | %.5g %.5g %.5g %.5g | %.5g %.5g %.5g %.5g", dump_seq(),
                     (unsigned long long)u.dst, (unsigned)u.size, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11]);
            }
            if (u.size && u.size <= avail - sizeof u) {
                UbUpdate up{rec_value_now, u.dst, std::vector<uint8_t>(a + sizeof u, a + sizeof u + u.size)};
                UbGrowGuard ug;   // (segments read the list meanwhile, RX_VK_SEG_EARLY)
                ub_updates.push_back(std::move(up));
                ub_list_epoch.fetch_add(1, std::memory_order_relaxed);
                ub_gen++;
                st_ub_updates++;
                if (ubo_check_on()) {   // (HL_UBO_CHECK: an update into a range bound already, which a copy at the bind misses)
                    bool hit = false;
                    for (int st = 0; st < kNvnStages && !hit; st++)
                        for (int i = 0; i < 16 && !hit; i++)
                            hit = s.ubo[st][i].addr && u.dst < s.ubo[st][i].addr + s.ubo[st][i].size && s.ubo[st][i].addr < u.dst + u.size;
                    if (hit) ubo_check_update_into_bound();
                }
                if (early && !dry && memory_rb_waited_range(host->gpu_guest(u.dst), u.size)) split_after = true;
            }
            break;
        }
        // ---- state ----
        case NOP_SET_RT: s.rt = payload<nvn_p_set_rt>(a, avail), s.rt_serial++; break;
        case NOP_VIEWPORTS: {
            const auto v = payload<nvn_p_viewports>(a, avail);
            for (int i = 0; i < v.n && v.first + i < 16; i++) std::memcpy(s.viewport[v.first + i], v.v[i], 16);
            break;
        }
        case NOP_SCISSOR: {
            const auto c = payload<nvn_p_scissor>(a, avail);
            s.scissor[0] = c.x, s.scissor[1] = c.y, s.scissor[2] = c.w, s.scissor[3] = c.h;
            break;
        }
        case NOP_DEPTH_RANGE: {
            const auto d = payload<nvn_p_depth_range>(a, avail);
            s.depth_near = d.n, s.depth_far = d.f;
            break;
        }
        case NOP_BIND_PROGRAM: {
            const auto b = payload<nvn_p_bind_program>(a, avail);
            // (the same program bound again: its descriptors already stand)
            const bool same = b.prog && s.has_prog ? std::memcmp(&s.prog, b.prog, sizeof s.prog) == 0 : !b.prog && !s.has_prog;
            if (!same) s.bind_gen++, st_bg_prog++;
            if (b.prog) s.prog = *b.prog;
            s.has_prog = b.prog != nullptr && b.prog->nstage > 0;
            break;
        }
        case NOP_BIND_STATE: {
            const auto b = payload<nvn_p_bind_state>(a, avail);
            switch (b.kind) {
            case 0: {
                nvn_blend_st st;
                std::memcpy(&st, b.bytes, sizeof st);
                s.blend[st.target & 7] = st;
                break;
            }
            case 1: std::memcpy(&s.color, b.bytes, sizeof s.color); break;
            case 2: std::memcpy(&s.chmask, b.bytes, sizeof s.chmask); break;
            case 3: std::memcpy(&s.ms, b.bytes, sizeof s.ms); break;
            case 4: std::memcpy(&s.poly, b.bytes, sizeof s.poly); break;
            case 5: std::memcpy(&s.ds, b.bytes, sizeof s.ds); break;
            default: break;
            }
            break;
        }
        case NOP_BIND_VATTR: {
            const auto v = payload<nvn_p_bind_vattr>(a, avail);
            std::memset(s.vattr, 0, sizeof s.vattr);
            for (int i = 0; i < v.n && i < 16; i++) s.vattr[i] = v.st[i];
            s.nvattr = std::min(v.n, 16);
            break;
        }
        case NOP_BIND_VSTREAM: {
            const auto v = payload<nvn_p_bind_vstream>(a, avail);
            for (int i = 0; i < v.n && i < 16; i++) s.vstream[i] = v.st[i];
            break;
        }
        case NOP_BIND_VBUF: {
            const auto v = payload<nvn_p_bind_vbuf>(a, avail);
            for (int i = 0; i < v.n && v.first + i < 16; i++) s.vbuf[v.first + i] = v.r[i];
            break;
        }
        case NOP_BIND_BUFS: {
            auto b = payload<nvn_p_bind_bufs>(a, avail);
            // (deferred binds: the ranges as the game's array held them at the submit, which the host copied into
            // the recording then (nvn_native.c resolve_deferred). Read again here, at the replay, the array was
            // often the next frame's already (MKW refills it as soon as the submit returns): the terrain patches
            // took other patches' placements (holes in the far terrain, the minimap's triangles).)
            {   // (EvilOOT, RX_VK_LOG_DEFERRED=1: compute's deferred binds, as at the submit and as the array holds them now)
                static const bool on = env_on("RX_VK_LOG_DEFERRED");
                static std::atomic<int> told{0};
                if (on && b.src && b.stage == 5 && told < 400) {
                    NVNbufferRange now[16]{};
                    const int n = std::clamp(b.n, 0, 16);
                    host->gpu_read(b.src, now, (uint64_t)n * sizeof(NVNbufferRange));
                    for (int i = 0; i < n && told < 400; i++, told++)
                        logf("vk: deferred %s stage 5 slot %d (array %#llx): submit %#llx+%#llx, now %#llx+%#llx%s", b.ssbo ? "ssbo" : "ubo",
                             b.first + i, (unsigned long long)b.src, (unsigned long long)b.r[i].address, (unsigned long long)b.r[i].size,
                             (unsigned long long)now[i].address, (unsigned long long)now[i].size,
                             now[i].address != b.r[i].address || now[i].size != b.r[i].size ? "  DIFFERS" : "");
                }
            }
            for (int i = 0; i < b.n && i < 16; i++) {
                const int index = b.first + i;
                if (b.stage < 0 || b.stage >= kNvnStages || index < 0 || index >= 16) continue;
                {
                    static const bool on = env_on("RX_VK_LOG_MESH_BINDS");
                    static std::atomic<int> told{0};
                    if (on && b.stage >= 6 && b.r[i].size != 0x100 && told++ < 300)
                        logf("vk: bind (many) %s stage %d index %d addr %#llx size %#llx", b.ssbo ? "ssbo" : "ubo", b.stage, index,
                             (unsigned long long)b.r[i].address, (unsigned long long)b.r[i].size);
                }
                nvn_p_bind_buf& u = (b.ssbo ? s.ssbo : s.ubo)[b.stage][index];
                if (!buf_unchanged(u.stage, u.index, u.addr, u.size, b.stage, index, b.r[i].address, b.r[i].size)) {
                    s.bind_gen++, st_bg_bufs++;
                    if (b.ssbo) s.ssbo_gen[b.stage] = s.bind_gen;
                }
                u.stage = b.stage, u.index = index, u.addr = b.r[i].address, u.size = b.r[i].size;
                if (!b.ssbo && s.ubo_bytes[b.stage][index])   // (by address again: HL_UBO_INLINE's bytes go)
                    s.ubo_bytes[b.stage][index] = nullptr, s.ubo_bytes_n[b.stage][index] = 0, s.bind_gen++;
            }
            break;
        }
        case NOP_BIND_UBO: case NOP_BIND_SSBO: {
            const auto b = payload<nvn_p_bind_buf>(a, avail);
            if (b.stage >= kNvnStages) log_stage_bind(h.op, b.stage);
            {   // (hookshot) RX_VK_LOG_MESH_BINDS=1: what the game binds to the task and mesh stages, 80 lines
                static const bool on = env_on("RX_VK_LOG_MESH_BINDS");
                static std::atomic<int> told{0};
                if (on && b.stage >= 6 && b.size != 0x100 && told++ < 300)
                    logf("vk: bind %s stage %d index %d addr %#llx size %#llx", h.op == NOP_BIND_UBO ? "ubo" : "ssbo", b.stage, b.index,
                         (unsigned long long)b.addr, (unsigned long long)b.size);
            }
            if (b.stage >= 0 && b.stage < kNvnStages && b.index >= 0 && b.index < 16) {
                nvn_p_bind_buf& u = (h.op == NOP_BIND_UBO ? s.ubo : s.ssbo)[b.stage][b.index];
                if (!buf_unchanged(u.stage, u.index, u.addr, u.size, b.stage, b.index, b.addr, b.size)) {
                    s.bind_gen++;
                    (h.op == NOP_BIND_UBO ? st_bg_ubo : st_bg_ssbo)++;
                    if (h.op == NOP_BIND_SSBO) s.ssbo_gen[b.stage] = s.bind_gen;
                }
                u = b;
                if (h.op == NOP_BIND_UBO && s.ubo_bytes[b.stage][b.index]) {   // (by address again: HL_UBO_INLINE's bytes go)
                    s.ubo_bytes[b.stage][b.index] = nullptr, s.ubo_bytes_n[b.stage][b.index] = 0;
                    s.bind_gen++;
                }
            }
            break;
        }
        case NOP_UBO_INLINE: {   // (HL_UBO_INLINE: the bind, and its block's bytes for the descriptor; every mode, as a state record)
            const auto b = payload<nvn_p_ubo_inline>(a, avail);
            if (b.stage < 0 || b.stage >= kNvnStages || b.index < 0 || b.index >= 16 || b.n > avail - sizeof b) break;
            const uint8_t* bytes = a + sizeof b;
            nvn_p_bind_buf& u = s.ubo[b.stage][b.index];
            const uint8_t* had = s.ubo_bytes[b.stage][b.index];
            const bool same = buf_unchanged(u.stage, u.index, u.addr, u.size, b.stage, b.index, b.addr, b.size) && had &&
                              s.ubo_bytes_n[b.stage][b.index] == b.n && (had == bytes || std::memcmp(had, bytes, b.n) == 0);
            if (!same) s.bind_gen++;   // (other bytes: the descriptors are built again)
            u.stage = b.stage, u.index = b.index, u.addr = b.addr, u.size = b.size;
            s.ubo_bytes[b.stage][b.index] = bytes, s.ubo_bytes_n[b.stage][b.index] = b.n;
            break;
        }
        case NOP_BIND_TEX: case NOP_BIND_SAMP: case NOP_BIND_IMG: {
            nvn_p_bind_handles b;
            std::memset(&b, 0, sizeof b);
            std::memcpy(&b, a, std::min<size_t>(sizeof b, h.size - 8));
            // (deferred: the handles as of the submit, as the ranges above)
            if (b.stage >= kNvnStages) log_stage_bind(h.op, b.stage);
            if (b.stage < 0 || b.stage >= kNvnStages || b.first < 0) break;
            uint64_t* to = h.op == NOP_BIND_TEX ? s.tex[b.stage] : h.op == NOP_BIND_SAMP ? s.samp[b.stage] : s.img[b.stage];
            const int lim = h.op == NOP_BIND_IMG ? 32 : 128;
            int n = 0;
            while (n < b.n && b.first + n < lim) n++;
            if (!handles_unchanged(to + b.first, b.h, n)) {   // (the same handles: the draw keeps its descriptors)
                s.bind_gen++;
                (h.op == NOP_BIND_TEX ? st_bg_tex : h.op == NOP_BIND_SAMP ? st_bg_samp : st_bg_img)++;
                s.tex_gen[b.stage] = s.bind_gen;
            }
            for (int k = 0; k < n; k++) to[b.first + k] = b.h[k];
            break;
        }
        case NOP_STENCIL: {
            const auto st = payload<nvn_p_stencil>(a, avail);
            for (int f = 0; f < 2; f++) {
                if (!(st.face & (1 << f))) continue;
                if (st.which == 0) s.stencil_value_mask[f] = (uint32_t)st.value_mask;
                else if (st.which == 1) s.stencil_mask[f] = (uint32_t)st.mask;
                else s.stencil_ref[f] = (uint32_t)st.ref;
            }
            break;
        }
        case NOP_POLY_OFFSET: {
            const auto o = payload<nvn_p_poly_offset>(a, avail);
            s.poly_offset[0] = o.factor, s.poly_offset[1] = o.units, s.poly_offset[2] = o.clamp;
            break;
        }
        case NOP_BLEND_COLOR: std::memcpy(s.blend_color, payload<nvn_p_blend_color>(a, avail).rgba, 16); break;
        case NOP_PRIMITIVE_RESTART: {
            const auto r = payload<nvn_p_restart>(a, avail);
            s.restart = r.enable != 0, s.restart_index = (uint32_t)r.index;
            break;
        }
        case NOP_DEPTH_CLAMP: s.depth_clamp = payload<nvn_p_flag>(a, avail).value != 0; break;
        case NOP_RASTER_DISCARD: s.raster_discard = payload<nvn_p_flag>(a, avail).value != 0; break;
        case NOP_RENDER_ENABLE:
            s.render_enabled = payload<nvn_p_flag>(a, avail).value != 0;
            if (dump_on()) logf("vk: dump-op %03d RENDER_ENABLE %d", dump_seq(), (int)s.render_enabled);
            break;
        case NOP_RENDER_COND: {   // (read on the CPU for now; VK_EXT_conditional_rendering later)
            if (worker) {   // (as the dry pass found it, with the reports before it written and none after)
                UbReadGuard ug;   // (par_conds may grow meanwhile, RX_VK_SEG_EARLY)
                s.render_enabled = cond_at < par_conds.size() ? par_conds[cond_at] != 0 : true;
                cond_at++;
                break;
            }
            const auto r = payload<nvn_p_render_cond>(a, avail);
            // (RX_VK_OCC_REAL=4: a report of this stream not read yet: what is recorded goes to the GPU and is waited for)
            if (occ_cond_pending(r.addr)) {
                if (dry) {   // (a dry pass ends here: replay_handle records and submits its segments, waits, and goes on from here)
                    occ_stop_at = p;
                    return;
                }
                if (occ_sync(cmd)) occ_resolve_all();
                else occ_give_up();
            }
            uint64_t x = r.x, y = r.y;
            if (!r.snapped) {
                host->gpu_read(r.addr, &x, 8);
                host->gpu_read(r.addr + 16, &y, 8);
            }
            s.render_enabled = r.mode == 0 ? x == y : x != y;
            {   // (RX_VK_COND_ALWAYS=1 (test): every condition holds. The frame log's dark frames dump the last ones)
                static const bool always = env_on("RX_VK_COND_ALWAYS");
                const uint32_t at = cond_ring_at++ % (1 << 15);
                cond_ring[at] = {presents_now, r.addr, x, y, (int)r.mode, s.render_enabled};
                if (always) s.render_enabled = true;
            }
            if (dry) {
                UbGrowGuard ug;   // (segments read it meanwhile, RX_VK_SEG_EARLY)
                par_conds.push_back(s.render_enabled ? 1 : 0);
            }
            if (occ_mode()) occ_cond(r.addr, (int)r.mode, x, y, s.render_enabled, occ_serial);
            if (dump_on())
                logf("vk: dump-op %03d RENDER_COND mode %d at %#llx: %llu vs %llu: %s", dump_seq(), r.mode,
                     (unsigned long long)r.addr, (unsigned long long)x, (unsigned long long)y,
                     s.render_enabled ? "on" : "off");
            break;
        }
        case NOP_SET_TEX_POOL: case NOP_SET_SAMP_POOL:
            s.bind_gen++;
            for (int k = 0; k < kNvnStages; k++) s.tex_gen[k] = s.bind_gen;
            break;
        case NOP_COPY_STRIDE: s.copy_stride = payload<nvn_p_copy_stride>(a, avail).stride; break;
        case NOP_PUSH_CONST: {   // (hookshot) compute push constants, kept in the state (vk_bank.cpp serves them from c0)
            const auto u = payload<nvn_p_push_const>(a, avail);
            if (u.size && u.size <= avail - sizeof u && (uint64_t)u.offset + u.size <= sizeof s.cs_push) {
                const uint8_t* src = a + sizeof u;
                const bool same = (uint64_t)u.offset + u.size <= s.cs_push_n && std::memcmp(s.cs_push + u.offset, src, u.size) == 0;
                std::memcpy(s.cs_push + u.offset, src, u.size);
                s.cs_push_n = std::max<uint32_t>(s.cs_push_n, u.offset + u.size);
                if (!same) s.bind_gen++;
            }
            break;
        }
        case NOP_BARRIER:
            if (dry) break;
            {
                const int32_t bits = payload<int32_t>(a, avail);
                std::lock_guard lk(st_barrier_flags_mu);
                st_barrier_flags[bits]++;
            }
            end_pass(cmd);
            {
                const int32_t bits = payload<int32_t>(a, avail);   // (ORDER_FRAGMENTS(_TILED) and INVALIDATE_TEXTURE only)
                if (game_barrier_narrow_on() && (bits & 0x6) && !(bits & ~0x16)) game_fragment_barrier(cmd);
                else full_barrier(cmd, "barrier (the game's nvnCommandBufferBarrier)");
            }
            st_barriers++;
            break;
        // ---- work ----
        case NOP_CLEAR_COLOR:
            // (hookshot) in the dry pass a clear marks its pass's targets as the GPU's, as a draw does (dry_mark_written): the clear
            // opens a pass on the worker, and that marks them (begin_pass), but only when the worker gets there. A target the game
            // clears in one segment and samples in a later one was still "never written" for the later segment when it was
            // recorded first: it uploaded what the target's memory held, an upload that runs after the clear. 13 of them in 4,300
            // presents of Castle Town's market with the camera turning, each an 853 x 480 target made that frame ("written in
            // segment N, uploaded by the later segment N+2; written by: a pass's colour target"), among 38 one-frame flashes
            // against 9 on one thread.
            if (s.render_enabled && dry) dry_mark_written(s, false, true);
            if (s.render_enabled && !dry) {
                const auto c = payload<nvn_p_clear_color>(a, avail);
                clear_color_target(cmd, s, c.index, c.rgba, (uint32_t)c.mask);
            } else if (s.render_enabled && dry_clear_marks()) {
                dry_mark_written(s, false, dry_work == 0);   // (D4: a clear writes its targets too: see dry_clear_marks)
            }
            break;
        case NOP_CLEAR_DS:
            if (s.render_enabled && dry) dry_mark_written(s, false, true);   // (as NOP_CLEAR_COLOR above)
            if (s.render_enabled && !dry) {
                const auto c = payload<nvn_p_clear_ds>(a, avail);
                clear_depth_stencil(cmd, s, c.depth, c.depth_mask != 0, (uint32_t)c.stencil, (uint32_t)c.stencil_mask);
            } else if (s.render_enabled && dry_clear_marks()) {
                dry_mark_written(s, false, dry_work == 0);
            }
            break;
        case NOP_DRAW: {
            const auto d = payload<nvn_p_draw>(a, avail);
            DrawArgs da;
            occ_take(s, da, dry);   // (hookshot: RX_VK_OCC_QUERIES, numbered in the dry pass too)
            if (s.render_enabled && !s.raster_discard && !dry) {
                da.prim = d.prim, da.first = d.first, da.count = d.count;
                da.base_instance = d.base_instance, da.instances = d.instances;
                OccSlot os(worker ? occ_wserial : occ_serial);
                draw(cmd, s, da);
            }
            s.samples_passed += 16;
            break;
        }
        case NOP_DRAW_IDX: {
            const auto d = payload<nvn_p_draw_idx>(a, avail);
            DrawArgs da;
            occ_take(s, da, dry);   // (hookshot: RX_VK_OCC_QUERIES, numbered in the dry pass too)
            if (s.render_enabled && !s.raster_discard && !dry) {
                da.prim = d.prim, da.indexed = true, da.itype = d.itype, da.count = d.count;
                da.base_vertex = d.base_vertex, da.base_instance = d.base_instance, da.instances = d.instances;
                da.ibuf = d.ibuf;
                OccSlot os(worker ? occ_wserial : occ_serial);
                draw(cmd, s, da);
            }
            s.samples_passed += 16;
            break;
        }
        case NOP_DRAW_INDIRECT: {
            const auto d = payload<nvn_p_draw_indirect>(a, avail);
            DrawArgs da;
            occ_take(s, da, dry);   // (hookshot: RX_VK_OCC_QUERIES, numbered in the dry pass too)
            if (s.render_enabled && !s.raster_discard && !dry) {
                da.prim = d.prim, da.indexed = d.indexed != 0, da.itype = d.itype;
                da.indirect = d.indirect, da.count_addr = d.count_addr, da.max = d.max, da.stride = d.stride;
                da.ibuf = d.ibuf;
                OccSlot os(worker ? occ_wserial : occ_serial);
                draw(cmd, s, da);
            }
            s.samples_passed += 16;
            break;
        }
        case NOP_DRAW_MESH: {
            const auto d = payload<nvn_p_draw_mesh>(a, avail);
            static int logged_mesh;
            if (logged_mesh++ < 8)
                logf("vk: mesh draw kind %d: %d %d %d, indirect %#llx count %#llx max %d stride %d", d.kind, d.x, d.y,
                     d.z, (unsigned long long)d.indirect, (unsigned long long)d.count_addr, d.max, d.stride);
            DrawArgs da;
            occ_take(s, da, dry);   // (hookshot: RX_VK_OCC_QUERIES, numbered in the dry pass too)
            if (s.render_enabled && !s.raster_discard && !dry) {
                da.mesh = true;
                // (DrawMeshTasks(first, count): the count of task workgroups, as Eden draws NVN2's mesh draws; a
                // first task other than 0 has no Vulkan counterpart)
                da.groups[0] = d.y, da.groups[1] = 1, da.groups[2] = 1;
                if (d.kind == 0) da.first_task = d.x;   // (hookshot: the task programs add it, vk_draw.cpp's mesh_args)
                da.indirect = d.indirect, da.count_addr = d.count_addr, da.max = d.max, da.stride = d.stride;
                OccSlot os(worker ? occ_wserial : occ_serial);
                draw(cmd, s, da);
            }
            s.samples_passed += 16;
            break;
        }
        case NOP_DISPATCH: {
            const auto d = payload<nvn_p_dispatch>(a, avail);
            if (s.render_enabled && !dry) dispatch(cmd, s, (uint32_t)d.x, (uint32_t)d.y, (uint32_t)d.z, 0);
            break;
        }
        case NOP_DISPATCH_INDIRECT: {
            const auto d = payload<nvn_p_dispatch_indirect>(a, avail);
            if (s.render_enabled && !dry) dispatch(cmd, s, 1, 1, 1, d.addr);
            break;
        }
        default: break;
        }
        if (op_is_work(h.op)) {
            if (worker) occ_wserial++;
            else if (occ_mode()) occ_advance(++occ_serial, rec_value_now);
            else ++occ_serial;
            if (dry) {
                const bool compute = h.op == NOP_DISPATCH || h.op == NOP_DISPATCH_INDIRECT;
                if (s.render_enabled && (compute || !s.raster_discard)) dry_mark_written(s, compute, dry_work == 0);
                static const int64_t point_every = par_point_every();
                if (++dry_work % point_every == 0 && lean_points && !lean_points->has(dry_work)) {
                    st_lean_left++;   // (a point no boundary can be at, RX_VK_DRY_LEAN: its state not copied)
                } else if (dry_work % point_every == 0) {
                    if (lean_points) st_lean_kept++;
                    // (the early path takes one point a boundary, the first at or after its planned place: a point none takes was a
                    // copy of the whole state, 58 of them a big command buffer, the vector growing by moves of the same)
                    if (seg_early_hook && dry_wanted_only() && !seg_early_hook->wants(dry_work, p + h.size)) {
                        // (not a boundary)
                    } else {
                        par_points.push_back({p + h.size, s, ub_updates.size(), par_conds.size(), dry_work});
                        if (seg_early_hook) seg_early_hook->on_point(par_points.back());   // (RX_VK_SEG_EARLY)
                    }
                }
                if (dry_work == presync_kick_at) presync_early_start();   // (a command buffer big enough to be cut, likely)
            } else if (!worker) {
                if (rb_writers) memory_count_work();
                maybe_split(cmd);
            }
        }
        // (RX_VK_READBACK_EARLY: this op wrote a page the CPU reads back and has waited for: its part ends here, so the read
        // waits for the GPU's work up to its writer, not for the draws a part runs on to. At least early_gap draws into
        // a part: writers in a row (four dispatches within ten draws) end one part, not four, each end a submit and barriers)
        if (early && !dry && !worker) {
            static const int64_t early_gap = std::max<int64_t>(0, env_int("RX_VK_READBACK_EARLY_GAP", 16));
            if (split_after || memory_split_wanted()) early_split_pending = true;
            if (early_split_pending && part_work >= early_gap) early_split(cmd);
        }
        p += h.size;
    }
}

/// (HL_UBO_CHECK) the guest ranges a record reads through an address, by kind (uniform binds: ubo_check_bind). An indexed
/// indirect draw's index count is in GPU memory: its index buffer's first 64 KB stand for it
void ubo_check_refs(uint32_t op, const uint8_t* a, uint32_t avail) {
    switch (op) {
    case NOP_BIND_SSBO: {
        const auto b = payload<nvn_p_bind_buf>(a, avail);
        ubo_check_ref(b.addr, b.size, kRefStorage);
        break;
    }
    case NOP_BIND_BUFS: {
        const auto b = payload<nvn_p_bind_bufs>(a, avail);
        for (int i = 0; i < b.n && i < 16; i++) ubo_check_ref(b.r[i].address, b.r[i].size, b.ssbo ? kRefStorage : kRefUboOther);
        break;
    }
    case NOP_BIND_VBUF: {
        const auto v = payload<nvn_p_bind_vbuf>(a, avail);
        for (int i = 0; i < v.n && i < 16; i++) ubo_check_ref(v.r[i].address, v.r[i].size, kRefVertex);
        break;
    }
    case NOP_DRAW_IDX: {
        const auto d = payload<nvn_p_draw_idx>(a, avail);
        ubo_check_ref(d.ibuf, (uint64_t)std::max(d.count, 0) * (d.itype == 0 ? 1 : d.itype == 1 ? 2 : 4), kRefIndex);
        break;
    }
    case NOP_DRAW_INDIRECT: {
        const auto d = payload<nvn_p_draw_indirect>(a, avail);
        ubo_check_ref(d.indirect, (uint64_t)std::max(d.max, 1) * (uint64_t)std::max(d.stride, 20), kRefIndirect);
        ubo_check_ref(d.count_addr, 4, kRefIndirect);
        if (d.indexed) ubo_check_ref(d.ibuf, 65536, kRefIndex);
        break;
    }
    case NOP_DRAW_MESH: {
        const auto d = payload<nvn_p_draw_mesh>(a, avail);
        ubo_check_ref(d.indirect, (uint64_t)std::max(d.max, 1) * (uint64_t)std::max(d.stride, 12), kRefIndirect);
        ubo_check_ref(d.count_addr, 4, kRefIndirect);
        break;
    }
    case NOP_DISPATCH_INDIRECT: ubo_check_ref(payload<nvn_p_dispatch_indirect>(a, avail).addr, 12, kRefIndirect); break;
    case NOP_COPY_B2B: {
        const auto c = payload<nvn_p_copy_bb>(a, avail);
        ubo_check_ref(c.src, c.size, kRefOther);
        break;
    }
    case NOP_RENDER_COND: ubo_check_ref(payload<nvn_p_render_cond>(a, avail).addr, 24, kRefOther); break;
    default: break;
    }
}

// ---- chunks: a long submission goes to the GPU in parts (RX_VK_CHUNK=<draws and dispatches a part>, default 240;
// 0: whole), so the GPU starts on its first part while the rest is recorded. The game waits for its fences within
// the frame: with the whole submission recorded before the GPU saw any of it, its wait was the replay's time plus
// the GPU's. Only the last part signals the submission's timeline value; the parts' slots free with it.
// 240 a part: each part end costs a command buffer, a queue submit and its barriers, and the first part still reaches
// the GPU within a millisecond or two of the recording's start. Kokiri Forest 63.0 -> 67.9 fps against 120 a part, the
// Market 68.6 -> 70.8, Hyrule Field the same (RTX 3090, 2560x1440, six interleaved sessions each); 480 was no better,
// 30 and sending the submission whole were worse (whole: 46 fps, the replay's time and the GPU's added up). ----
Recording* rec_cur = nullptr;
uint64_t rec_cur_value = 0;
int rec_parts = 0;
int64_t part_work = 0;
// (RX_VK_READBACK_EARLY) the value the recording being made writes back with (rec_cur_value while rec_cur is set, 0 between
// recordings), for the other threads: a read whose writer's submission is still being recorded has not seen all of its
// writers yet (early_recorded_wait). Set where a recording opens (rec_open) and cleared where it is finished (rec_close)
std::atomic<uint64_t> rec_open_tag{0};
bool rec_tags_on() { return readback_early_on() || rb_writers_on(); }   // (their only readers: unset, nothing is kept)
void rec_open(uint64_t tag) {
    if (rec_tags_on()) rec_open_tag.store(tag, std::memory_order_release);
}
void rec_close();   // (below, with the recordings' timeline it wakes the waiters of)
// (a slot is used again once its own submit is done (recording_wait): a submission's parts may outnumber the slots,
// which a wait on the timeline value it signals at its end kept them from. A parallel replay's segments are recorded
// at once, kMaxBatch of them at most: fewer than the slots)
constexpr int kMaxParts = 32;
constexpr int kMaxBatch = 12;   // (the arrays' size; seg_cap() is the number a command buffer is cut into at most)
/// (RX_VK_FINE_SEGMENTS=1, off by default) a big command buffer cut into more segments than there
/// are recorders: the last ones are short, and a recorder that is done takes the next, so the replay thread's own (last) segment, which
/// starts when the dry pass is over, is a smaller share (PC 3050, castle-move: 4.7 -> 3.6 ms, the frame 36.0 -> 37.7 fps). Up to 10
/// segments of ~128 draws and dispatches, not one round of at most one a recorder (as set by RX_VK_SEG_MAX, RX_VK_PARALLEL_SEG,
/// each alone). Off: at most 6 segments of ~160, as before
bool fine_segments() {
    static const bool v = [] {
        const char* e = getenv("RX_VK_FINE_SEGMENTS");
        const bool on = e && atoi(e) != 0;   // (measured only under 16 hardware threads: off until run here)
        logf("vk: a big command buffer's segments: %s (RX_VK_FINE_SEGMENTS)", on ? "up to 10 of ~128 draws, more than recorders" : "up to 6 of ~160 draws");
        return on;
    }();
    return v;
}
/// (RX_VK_SEG_MAX=<n>) at most that many segments a command buffer: 10 with fine_segments, else 6 (three recording threads, more
/// segments only took slots and submits)
int seg_cap() {
    static const int v = [] {
        const char* e = getenv("RX_VK_SEG_MAX");
        return e ? std::clamp(atoi(e), 2, kMaxBatch) : fine_segments() ? 10 : 6;
    }();
    return v;
}
// (a short command buffer a parallel replay records on this thread: one segment's worth, and a segment is not split)
thread_local bool split_off = false;

/// (hookshot) Whether a part ends only right before a plain draw (one with a vertex stage), so that no cut leaves a part
/// beginning with a mesh draw or a dispatch (a submission's own first recording begins as the game began it). On an Intel Arc B580 (Windows, driver 32.0.101.9154, native mesh shaders) a command buffer
/// whose first draws are mesh draws drew every plain draw after them as exploded triangles: white shards over the scene
/// in Kokiri Forest while moving, in about 2% of the frames (whenever the 120th draw or dispatch fell among the depth
/// pre-pass's mesh draws or on the culling dispatch before them). A cut forced after that pass's first mesh draw showed it
/// in 333 of 335 frames; with only the new part's leading mesh draws left out, 0 of 324. Not a race (the GPU waited for
/// after each part: 317 of 319), not a state the replay keeps (all of a draw's state sent again before every draw: 319 of
/// 328), nothing the validation layer names, and the uniform data the draws were given was the submission's own. With
/// this the forced cut shows none (0 of 324), and over six 30 s walks alternated with it off, 0 of 1,101 frames against
/// 40 of 1,121, at the same frame rate (a run's median 52 and 48 fps with it off, 55 and 50 with it on).
/// A run of mesh draws or dispatches with no plain draw in it is no longer cut: the part after it is shorter by what the
/// run went over. A plain draw that draw() then leaves out (no targets, a program not ready) still begins the part.
/// On by default on Intel, where it was measured; RX_VK_SPLIT_BEFORE_DRAW=1 turns it on elsewhere, =0 off (a part ends
/// after its RX_VK_CHUNK-th draw or dispatch, whatever follows, as before)
bool part_ends_before_draw() {
    return dev.policy.split_before_draw;   // (device_policy.h: Intel, then the switch)
}

void maybe_split(VkCommandBuffer& cmd) {
    static const int64_t every = getenv("RX_VK_CHUNK") ? atoll(getenv("RX_VK_CHUNK")) : 240;
    if (split_off || !rec_cur || every <= 0 || ++part_work < every || rec_parts + 1 >= kMaxParts || dump_on()) return;
    if (part_ends_before_draw()) {   // (at the next plain draw: replay(). The submission's end comes first when none follows)
        part_end_wanted = true;
        return;
    }
    split_part(cmd);
}

std::atomic<int64_t> st_early_splits{0};   // (RX_VK_READBACK_EARLY: parts ended after a readback's writer)
bool early_end_asked = false;   // (early_split asked for the part's end: counted in st_early_splits once it is made)

/// The recording's part ends here and the submission goes on in a new one (maybe_split says when)
void split_part(VkCommandBuffer& cmd) {
#ifdef TRACY_ENABLE
    HT_SCOPE("split_part");
#endif
    part_end_wanted = false;
    // (asked for in another command buffer of the submission, or before a parallel replay's segments took their parts:
    // what maybe_split refuses is refused here too, and it asks again at the next draw it would cut after)
    if (split_off || !rec_cur || rec_parts + 1 >= kMaxParts || dump_on()) return;
    end_pass(cmd);
    // (a part's end does not write the stream's uniform updates recorded so far into guest memory, nor start a new window
    // of them: the submission's last part writes them all, as an unsplit submission does (finish_recording) and as the
    // parallel replay does. A uniform binding is not what changes: one overlapping an update gets an overlay copy built
    // from the whole list (ub_overlay), the same bytes either way. What changes is everything else that reads guest memory
    // after a part's end: a later part's storage, vertex, index or texture reads of an updated range, and, on a GPU whose
    // buffer pools have no VRAM mirror, the game's own CPU reads, since the GPU then writes the memory the game sees partway
    // through its frame. Written at each part's end, on Intel without mirrors: white blocks and flashes at Kokiri, and
    // a whole-screen white flash every 8 frames on the Arc 140V (issue 999).
    // Measured on an Intel iGPU only (no mirrors), so the default there; with mirrors (Arc discrete) the write goes to the
    // mirror, and it was not tested there: left as before. RX_VK_SPLIT_UBFLUSH=1: written at each part's end, as before,
    // =0: at the last part's on any GPU)
    // (device_policy.h split_ub_flush: not on Intel without mirrors, the mirrors as the run has them; then the switch)
    const bool part_flush = dev.policy.split_ub_flush;
    if (part_flush) flush_ub_updates(cmd);
    memory_end_submission(cmd, rec_cur->part);
    full_barrier(cmd, "barrier (recording end)");
    occ_copy_part(cmd);
    submit_recording(*rec_cur, false);
    if (part_flush) ub_first_of_recording = ub_updates.size();
    Recording& r = begin_recording(rec_cur_value);
    memory_begin_submission();   // (not memory_gpu_writes_begin: the submission's GPU writes so far stay noted, vk_memory.cpp)
    full_barrier(r.cmd, "barrier (recording start)");
    rec_cur = &r;
    cmd = r.cmd;
    part_work = 0;
    rec_parts++;
    if (early_end_asked) st_early_splits++;
    early_split_pending = early_end_asked = false;   // (the writer's part has ended, by its count or after it)
    if (readback_early_on()) (void)memory_split_wanted();
}

/// (RX_VK_READBACK_EARLY) The part ends after a writer of a range a read has waited for (replay(), at least
/// RX_VK_READBACK_EARLY_GAP draws into the part), so that the read waits for the GPU's work up to its writer. Where a part
/// ends only right before a plain draw (part_ends_before_draw: Intel by default, RX_VK_SPLIT_BEFORE_DRAW), this end waits
/// for that draw too, as maybe_split's does: the writer's part runs on over the mesh draws and dispatches after it, so a
/// read waits for a little more than its writer, never less, and no part begins with a mesh draw or a dispatch. With no
/// plain draw left in the submission its end comes first, and the read waits for the whole submission, as without the switch
void early_split(VkCommandBuffer& cmd) {
#ifdef TRACY_ENABLE
    HT_SCOPE("early_split");
#endif
    early_end_asked = true;
    if (part_ends_before_draw()) {   // (at the next plain draw: replay())
        part_end_wanted = true;
        return;
    }
    split_part(cmd);
}

/// (RX_VK_OCC_REAL=4) what is recorded so far submitted and waited for; recording goes on in a new part
bool occ_sync(VkCommandBuffer& cmd) {
    if (!rec_cur || rec_parts + 1 >= kMaxParts || dump_on()) return false;
    end_pass(cmd);
    // (a part's end does not write the stream's uniform updates recorded so far into guest memory, nor start a new window
    // of them: the submission's last part writes them all, as an unsplit submission does (finish_recording) and as the
    // parallel replay does. A uniform binding is not what changes: one overlapping an update gets an overlay copy built
    // from the whole list (ub_overlay), the same bytes either way. What changes is everything else that reads guest memory
    // after a part's end: a later part's storage, vertex, index or texture reads of an updated range, and, on a GPU whose
    // buffer pools have no VRAM mirror, the game's own CPU reads, since the GPU then writes the memory the game sees partway
    // through its frame. Written at each part's end, on Intel without mirrors: white blocks and flashes at Kokiri, and
    // a whole-screen white flash every 8 frames on the Arc 140V (issue 999).
    // Measured on an Intel iGPU only (no mirrors), so the default there; with mirrors (Arc discrete) the write goes to the
    // mirror, and it was not tested there: left as before. RX_VK_SPLIT_UBFLUSH=1: written at each part's end, as before,
    // =0: at the last part's on any GPU)
    // (device_policy.h split_ub_flush, as split_part's: not on Intel without mirrors; then the switch)
    const bool part_flush = dev.policy.split_ub_flush;
    if (part_flush) flush_ub_updates(cmd);
    memory_end_submission(cmd, rec_cur->part);
    full_barrier(cmd, "barrier (recording end)");
    Recording* old = rec_cur;
    occ_copy_part(cmd);
    submit_recording(*rec_cur, false);
    if (part_flush) ub_first_of_recording = ub_updates.size();
    Recording& r = begin_recording(rec_cur_value);
    memory_begin_submission();
    full_barrier(r.cmd, "barrier (recording start)");
    rec_cur = &r;
    cmd = r.cmd;
    part_work = 0;
    rec_parts++;
    recording_wait(*old);
    return true;
}

// ---- parallel replay: the threads and a command buffer's segments ----
// (RX_VK_PARALLEL=<threads recording a command buffer, the replay thread one of them>; 0, 1: off, the default. Segments
// recorded out of order left one-frame black scenes in play, 48-93 in a walk through Kakariko against none on the
// replay thread alone, at about the same frame rate here: 36-40 fps against 38-41)
// (2026-11-08, the fork: with the game's occlusion queries turned off (a switch since removed) the flashes of out-of-order segments did not come back in six Kokiri walks (five in one
// walk without it), and the replay fell from 12.6 to 8.5 ms (54 -> 58 fps, the game's 60 Hz pacing then the limit). But the picture is wrong
// with segments of 160 or 320 draws: the fog is washed out and the ground has no shadows, from the first frame, with or without
// Occlusion Off. Segments of 640 and one big one are right, and so are two threads on segments of 1000; no boundary placement alone does it
// (RX_VK_PARALLEL_BOUNDARY=<k>, RX_VK_PARALLEL_MINWORK=<n> tried). : not a race (the segments recorded in order on one thread
// were as wrong): Kokiri's command buffers are ~750 draws, so 640 and 1000 never split them. Each segment's recording refreshed the
// exposure meter's readback page from guest memory (memory_rb_refresh), wiping what the segments before it had added: now once a part)
// (hookshot) One cause of those is gone: the dry pass did not mark the targets of a clear (NOP_CLEAR_COLOR in replay()). With
// that, Castle Town's market and eight other places showed no black frame on 3 to 8 threads; the Kakariko walk was not run
// again, so it stays off. On a 9950X3D the replay's wall time in the market goes from 10.9 ms a frame to 7.75 on 3 threads.
int par_threads() {
    // Off by default, as before the renderer merge: RX_VK_PARALLEL=<n> (2 to 8) records a big command buffer's segments on n
    // threads. The merge had brought another tree's default, on with 8 or more hardware threads (4 threads; 5 from 16), measured
    // there as a gain (Kokiri 54 -> 65 fps on its PC). Here it cost a quarter of the frame rate: Kokiri 30.5 fps against 38.9 with
    // it off (RTX 4060 Ti, Ryzen 5 7600X, 12 threads). The exposure meter the game reads back every frame (0x3980 bytes, two
    // buffers in turn) is written back only in a parallel submission's last part, so the job thread reading it waited for the
    // whole submission, 16 ms median (RX_VK_LOG_READBACK_WAITS), where the serial replay's earlier part was done (0.01 ms)
    static const int n = std::clamp(getenv("RX_VK_PARALLEL") ? atoi(getenv("RX_VK_PARALLEL")) : 0, 0, 8);
    return n;
}
struct ParPool {
    std::mutex mu;
    std::condition_variable cv, done;
    std::deque<std::function<void()>> jobs;
    std::deque<std::function<void()>> own[8];   // (RX_VK_PARALLEL_AFFINITY: a worker's own jobs)
    int pending = 0;
    bool started = false;
};
ParPool par_pool;
/// RX_VK_PARALLEL_IDEAL=1: each worker's ideal processor is a physical core of its own (its first logical processor),
/// from the second core on: two recorders on one core's two hardware threads share its caches and units
void par_worker_ideal(int index) {
    static const bool on = getenv("RX_VK_PARALLEL_IDEAL") && atoi(getenv("RX_VK_PARALLEL_IDEAL")) != 0;
    if (!on) return;
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) + 1);
    if (!GetLogicalProcessorInformation(info.data(), &len)) return;
    std::vector<int> firsts;   // (each physical core's first logical processor)
    for (size_t i = 0; i < len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); i++)
        if (info[i].Relationship == RelationProcessorCore && info[i].ProcessorMask)
            firsts.push_back(__builtin_ctzll((unsigned long long)info[i].ProcessorMask));
    std::sort(firsts.begin(), firsts.end());
    if (firsts.size() < 2) return;
    const int cpu = firsts[1 + index % (int)(firsts.size() - 1)];
    const DWORD prev = SetThreadIdealProcessor(GetCurrentThread(), (DWORD)cpu);
    logf("vk: parallel replay worker %d: ideal processor %d (was %d; %d cores) (RX_VK_PARALLEL_IDEAL)", index, cpu, (int)prev,
         (int)firsts.size());
}
void par_worker_main(int index) {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk replay worker");
#ifdef TRACY_ENABLE
    HT_THREAD_NAME("mkw_vk replay worker");
#endif
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    par_worker_ideal(index);
    std::deque<std::function<void()>>& own = par_pool.own[index & 7];
    for (;;) {
        std::function<void()> fn;
        {
            std::unique_lock lk(par_pool.mu);
            par_pool.cv.wait(lk, [&own] { return !own.empty() || !par_pool.jobs.empty(); });
            std::deque<std::function<void()>>& q = !own.empty() ? own : par_pool.jobs;
            fn = std::move(q.front());
            q.pop_front();
        }
#ifdef TRACY_ENABLE
        HT_SCOPE("replay worker job");
#endif
        fn();
        std::lock_guard lk(par_pool.mu);
        if (--par_pool.pending == 0) par_pool.done.notify_all();
    }
}

/// A recording slot for a segment a worker records (its command buffer only: uploads go to the shared one)
Recording& begin_segment_recording(uint64_t value) {
    Recording& r = recordings[recording_index];
    recording_index = (recording_index + 1) % kRecordings;
    if (r.value) {
        PROF(PROF_SLOTWAIT);
        recording_wait(r);
        gpu_time_collect(r);
        gprof_collect(r.slot);
    }
    vkResetCommandPool(dev.device, r.pool, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(r.cmd, &bi);
    r.value = value;
    return r;
}
// ---- RX_VK_SEG_SETUP_PAR=1 (off by default): a segment's command buffers begun by the thread that records it ----
// Before releasing any segment, the replay thread took each segment's slot, reset its two command pools, began its command buffer
// and its upload one and wrote their start timestamps (gpu_time_begin): 5-6 segments, two big command buffers a frame, part of
// the ~0.2 ms "setup" on the frame's loop . Here it only takes the slots, in order (begin_segment_slot: the wait for the
// slot's last use, its GPU times collected, its value); the resets, the begins and the timestamps move to the start of record_job,
// on the recording thread, before anything is recorded into them.
// Same commands in the same command buffers: each pool and command buffer belongs to one slot, and only the job's thread touches
// them from its start to its end (Vulkan's external synchronisation holds); the job is submitted only after its end (seg_mu), so
// the queue sees the same command buffers. The timestamp writes and resets use the slot's own four queries, as before; the query
// pool exists by then (begin_recording made it before any segment; without one the old path runs). The init command buffers stay
// on the replay thread: texture_segments_begin hands them out before any job starts. A segment whose init could not be begun
// (the fallback below) leaves its slot unsubmitted and its command buffers not begun: begin_recording resets the pools when the
// slot comes round, as for a begun one.
bool seg_setup_par_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_SEG_SETUP_PAR");
        const bool v = e && atoi(e) != 0;
        logf("vk: a segment's command buffers: %s (RX_VK_SEG_SETUP_PAR)", v ? "begun by the thread recording it" : "begun by the replay thread, as before");
        return v;
    }();
    return on;
}
Recording& begin_segment_slot(uint64_t value) {   // (begin_segment_recording without the pool reset and the begin)
    Recording& r = recordings[recording_index];
    recording_index = (recording_index + 1) % kRecordings;
    if (r.value) {
        PROF(PROF_SLOTWAIT);
        recording_wait(r);
        gpu_time_collect(r);
        gprof_collect(r.slot);
    }
    r.value = value;
    return r;
}
void segment_begin_here(Recording& r) {   // (the rest of it and of the upload's setup, on the recording thread)
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandPool(dev.device, r.pool, 0);
    vkBeginCommandBuffer(r.cmd, &bi);
    vkResetCommandPool(dev.device, r.upload_pool, 0);
    vkBeginCommandBuffer(r.upload, &bi);
    gpu_time_begin(r);
}

struct Segment {
    const uint8_t *p, *end;
    const NvnState* s;
    size_t ub, cond;
    uint64_t serial = 0;   // (the stream number of its first draw or dispatch)
};
/// One segment recorded into cmd from the state at its start (any thread: the replay thread's own included)

void record_segment(VkCommandBuffer& cmd, const Segment& g) {
#ifdef TRACY_ENABLE
    HT_SCOPE("record_segment");
#endif
    rmode = Mode::Worker;
    par_stats_segment(true);
    NvnState s = *g.s;
    ub_limit = g.ub, cond_at = g.cond;
    occ_wserial = g.serial;
    ub_gen++;
    full_barrier(cmd);   // (after what the command buffers before it did)
    replay(cmd, s, g.p, g.end);
    end_pass(cmd);
    full_barrier(cmd);
    par_stats_segment(false);
    rmode = Mode::Serial;
    ub_limit = SIZE_MAX;
}


// ---- RX_VK_PRESYNC_EARLY=1: the mirrors' pre-sync off the replay thread's path ----
// The pre-sync (memory_sync_mirrors_before_segments: every mirrored pool not synced in this submission part, its CPU pages copied
// in) ran on the replay thread between the dry pass and the segments, ~0.39 ms a big command buffer (0.78 ms of the
// frame's serial loop). Here a replay worker runs it while the replay thread runs the dry pass, into a command buffer of its own
// (a recording slot's cmd: the replay thread's upload command buffer is its own while it runs the dry pass, which may make
// textures), submitted right after the dry pass and before the recording that holds what came before (rec_cur), so its copies
// run where they ran in rec_cur's uploads: after the work submitted before, before rec_cur's work and the segments. Each pool
// is synced once, under pools_mu, whichever thread gets there first: the dry pass binds no buffer (its copies and clears are
// left to the segments), and memory_begin_submission (mirror_epoch) runs only between recordings. The job starts at the dry
// pass's start when the replay thread's command buffer of the same order was cut into segments in the last frame, else when the
// dry pass reaches its RX_VK_PRESYNC_EARLY-th draw or dispatch (1: 64); a dry pass that stops sooner (a wait not met, a small
// command buffer: the held submits' are) is not cut and wants no pre-sync. When the dry pass ends: segments follow and no worker
// has started it: the replay thread runs it; no segments: dropped if not started. The pre-sync on the replay thread stays after
// it, finding nothing left to sync (a safety net).
struct PresyncEarly {
    std::mutex mu;
    std::condition_variable cv;
    uint64_t gen = 0;   // (the job's number: a job the replay thread took back is left a no-op in the pool's queue)
    bool taken = false, done = false, used = false;
    Recording* r = nullptr;
} presync_early;
int64_t presync_early_at() {
    static const int64_t v = getenv("RX_VK_PRESYNC_EARLY") ? std::max(0, atoi(getenv("RX_VK_PRESYNC_EARLY"))) : 0;
    return v == 1 ? 64 : v;
}
/// The pre-sync into e.r's command buffer on this thread (marks 22/23)
void presync_early_run(PresyncEarly& e) {
    jt_mark(22, 0);
    const UploadTarget was = texture_upload_swap({e.r->cmd, false});
    memory_sync_mirrors_before_segments();
    const UploadTarget mine = texture_upload_swap(was);
    if (mine.used) full_barrier(e.r->cmd);   // (before what reads the mirrors: rec_cur's work and the segments)
    e.used = mine.used;
    jt_mark(23, 0);
}
/// Before the dry pass (the replay thread): a slot for the pre-sync's commands, and the job to a worker
void presync_early_start() {
    PresyncEarly& e = presync_early;
    Recording& r = begin_segment_recording(rec_cur_value);
    uint64_t gen;
    {
        std::lock_guard l(e.mu);
        gen = ++e.gen;
        e.taken = e.done = e.used = false;
        e.r = &r;
    }
    {
        std::lock_guard lk(par_pool.mu);
        par_pool.jobs.push_back([gen] {
            PresyncEarly& e = presync_early;
            {
                std::lock_guard l(e.mu);
                if (e.gen != gen || e.taken) return;
                e.taken = true;
            }
            presync_early_run(e);
            {
                std::lock_guard l(e.mu);
                e.done = true;
            }
            e.cv.notify_all();
        });
        par_pool.pending++;
    }
    par_pool.cv.notify_one();
}

/// A command buffer of the submission from h (its start, or where it stopped at a wait): on the replay thread alone
/// (RX_VK_PARALLEL off, a small one, dumps and profiles), or its dry pass there and its segments on the threads.
/// may_stop: up to a wait not met, which it returns (null: the command buffer done)
const uint8_t* occ_go_on(VkCommandBuffer& cmd, const uint8_t* at, bool may_stop, Recording* last);

// ---- RX_VK_SEG_EARLY=1: a big command buffer's segments handed out as the dry pass passes their boundaries ----
// 's loop: the Presentation Thread's dry pass (~0.25-0.5 ms a big command buffer) ran before any segment started. Here
// the boundaries are planned from the command buffer of the same order in the last frame (its draw count and time profile,
// RX_VK_PARALLEL_BALANCE's), and segment k goes to a worker as soon as the dry pass passes the point after its planned end (the
// first kept point at or past it): what a segment takes from the dry pass (its starting state, the uniform updates and render
// conditions before its end, the written-image marks of the draws before its end) is there by then. The replay thread records
// the last segment itself once the dry pass ends (its end is where the dry pass stopped). Before the dry pass, in rec_cur:
// its pass ended, the mirrors synced (the pre-sync, ahead of every segment as before) and the part's readback refresh made
// (memory_rb_refresh: a worker ending its recording first would have made it in a segment's uploads, after rec_cur's work);
// the recording slots, the segments' order (par_seq) and their init command buffers for the planned count, par_active on.
// While the dry pass runs with segments out: its appends to ub_updates and par_conds hold ub_grow_mu (UbGrowGuard), its marks
// ParLock (dry_mark_written), and the texture paths it shares with the draws take ParLock as in a segment. Points it never
// reaches (a shorter command buffer) leave their slots unused (their init command buffers ended). No boundary reached: the
// whole command buffer is recorded here, into rec_cur, as a one-segment one is.
bool seg_early_on() {
    static const bool v = getenv("RX_VK_SEG_EARLY") && atoi(getenv("RX_VK_SEG_EARLY")) != 0;
    return v;
}
// ---- RX_VK_EARLY_GPU: the GPU given a big command buffer's work as it is recorded, not once its dry pass is over ----
// At the bench's settings on the PC 3050 (17 zones, 2560x1440, 3D at 100%) the GPU works 18.8 ms of a 24.4 ms frame and sits idle
// 5.2: the game's readback of the previous frame's light meter waits for the GPU's whole frame, the CPU's signal follows it by
// ~0.05 ms, and from the signal on nothing reaches the GPU before the pre-sync, the first big command buffer's whole dry pass and
// the replay thread's own last segment are done (~5 ms, R6's trace). =1: rec_cur (the work before it, the pre-sync's copies, the
// part's readback refresh) is submitted before the dry pass, the dry pass's own uploads go to a recording of their own, and each
// segment is submitted in order by the thread that finishes it. =2: also the first segment cut short (RX_VK_EARLY_GPU_FIRST draws
// and dispatches, 48 by default), so the GPU starts sooner. Off by default (the fork, on: PC 3050, interleaved A/B, same
// pictures: kokiri-forest 36.45 -> 38.05 fps, market 42.05 -> 42.85); unset or =0: exactly as before.
int early_gpu_mode() {
    static const int v = [] {
        const char* e = getenv("RX_VK_EARLY_GPU");
        const int m = e ? std::clamp(atoi(e), 0, 2) : 0;   // (off unless asked: Hookshot's replay does not take this path)
        logf("vk: a big command buffer's work to the GPU: %s (RX_VK_EARLY_GPU=%d)",
             m == 0 ? "after its dry pass and the replay thread's own segment, as before"
                    : m == 1 ? "as recorded: the work before it first, each segment in order as it is done"
                             : "as recorded, the first segment short",
             m);
        return m;
    }();
    return v;
}
int64_t early_gpu_first() {
    static const int64_t v = getenv("RX_VK_EARLY_GPU_FIRST") ? std::max<int64_t>(8, atoll(getenv("RX_VK_EARLY_GPU_FIRST"))) : 48;
    return v;
}
/// (the plans: 0 used (segments cut early), 1 used but no boundary reached (recorded whole), 2 rejected before the dry pass (the last
/// count too small: the old path). Logged every 2400 big command buffers predicted, and the first time each kind happens)
void seg_early_count(int kind) {
    static int64_t n[3], all;
    const bool first = n[kind]++ == 0;
    if (first || ++all % 2400 == 0)
        logf("vk: early segments (RX_VK_SEG_EARLY): plans used %lld, used but no boundary reached (recorded whole) %lld, rejected "
             "(the old path) %lld", (long long)n[0], (long long)n[1], (long long)n[2]);
}
/// The replay cannot go on: logged, the host's queued log written out (as hos_fatal does: a device lost's report is many
/// lines, and the queue went with the process), written to stderr and the process ended (exit code 3). For a parallel
/// replay's init command buffer that could not be ended: images already made and published lost their first layout
/// change, and any later submission may use them
[[noreturn]] void replay_fatal(const char* why) {
    // The host's crash/freeze records read this prefix (host/hos_report_prefixes.h); keep it stable.
    logf("vk: FATAL: %s", why);
    if (host && host->log_flush) host->log_flush();
    std::fprintf(stderr, "vk: FATAL: %s\n", why);
    std::fflush(stderr);
    std::_Exit(3);
}
const uint8_t* replay_handle_early(VkCommandBuffer& cmd, const uint8_t* h, bool may_stop, int nplan, const int64_t* plan,
                                   bool self_help, mkwvk::segment_plan::Profile& profile, int64_t* total_out, bool* cut) {
    NvnState& state = *cur_state;
    presync_kick_at = 0;   // (the pre-sync is made here, before the dry pass)
    occ_stop_at = nullptr;
    EarlyWall early_wall;
    const auto t_dry0 = std::chrono::steady_clock::now();
    const size_t ub0 = ub_updates.size();
    par_points.clear(), par_conds.clear(), dry_work = 0;
    const uint64_t serial0 = occ_serial;
    Recording& r = *rec_cur;
    end_pass(r.cmd);
    full_barrier(r.cmd);
    memory_sync_mirrors_before_segments();   // (rec_cur's uploads, before the dry pass: ahead of every segment)
    early_wall.mark(1);
    texture_rb_refresh_now();                // (the part's readback refresh, in r: before any segment's)
    early_wall.mark(2);
    jt_mark(19, 0);
    const uint64_t base = ++par_seq;
    par_seq += (uint64_t)nplan;
    memory_rb_segments_begin(base, (uint64_t)nplan);
    struct Job {
        Recording* r;
        Segment g;
        NvnState s;   // (its starting state: the dry pass's points move as it goes on)
        uint64_t order;
        int index = 0;
        std::atomic<bool> done{false};
        bool uploads = false;
        int inits = 0;
        int64_t t0 = 0, t1 = 0;
    };
    std::vector<std::unique_ptr<Job>> jobs;
    for (int k = 0; k < nplan; k++) {
        auto j = std::make_unique<Job>();
        j->r = &begin_segment_recording(rec_cur_value);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandPool(dev.device, j->r->upload_pool, 0);
        vkBeginCommandBuffer(j->r->upload, &bi);
        // (the segments' GPU time too, as replay_handle's . Without it these segments were in no GPU-time count: the log's
        // "GPU time" ~3.8 ms a frame instead of ~6.3 on an RX 9070 XT in Kokiri, and its gaps and RX_VK_TL's GPU side showed the GPU
        // idle 3-5 ms before each present while it ran them. A slot never used (a boundary not reached) is never ended: not collected)
        gpu_time_begin(*j->r);
        j->order = base + (uint64_t)k;
        j->index = k;
        jobs.push_back(std::move(j));
    }
    early_wall.mark(3);
    std::vector<VkCommandBuffer> inits;
    {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        for (auto& jp : jobs)
            if (jp->r->init && vkResetCommandPool(dev.device, jp->r->init_pool, 0) == VK_SUCCESS &&
                vkBeginCommandBuffer(jp->r->init, &bi) == VK_SUCCESS)
                inits.push_back(jp->r->init);
            else
                break;
    }
    const bool early = inits.size() == jobs.size();   // (else: no segment, the dry pass then the whole here, below)
    if (early) texture_segments_begin(inits.data(), inits.size());
    early_wall.mark(4);
    jt_mark(21, 0);
    // (RX_VK_EARLY_GPU) rec_cur goes to the GPU now, with the pre-sync's copies: the dry pass records nothing into it
    // but uploads (images it makes or marks), which go to rd, a recording of its own submitted where rec_cur was (after the dry
    // pass, ahead of every segment not submitted by then). A segment is submitted early only while the dry pass has recorded no
    // upload (dry_uploaded): an image the dry pass makes is made under ParLock, as the segments' lookups are, and the watch is set
    // before its first command is recorded, so no segment that could see it goes ahead of rd
    const int egpu = early ? early_gpu_mode() : 0;
    std::atomic<bool> dry_uploaded{false};
    Recording* rd = nullptr;
    if (egpu) {
        submit_recording(r, false);   // (what came before, its uploads with the pre-sync: ahead of every segment in the queue)
        for (auto& jp : jobs) ts_recseq[jp->r->slot] = rec_seq;   // (RX_VK_TL: shown with r, the recording they follow)
        rd = &begin_recording(rec_cur_value);
        full_barrier(rd->cmd);
        rec_cur = rd;
        cmd = rd->cmd;   // (no boundary reached: the whole command buffer is recorded into rd, below)
        texture_upload_watch(&dry_uploaded);
    }
    bool init_lost = false;
    const auto submit_job = [&init_lost](Job* j) {   // (a segment, in the queue's order: the caller's)
        if (j->inits < 0) init_lost = true;   // (its init command buffer could not be ended: the replay stops, below)
        if (init_lost) return;                // (nothing from it on is submitted)
        VkCommandBufferSubmitInfo cb[3]{{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO},
                                        {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}};
        uint32_t n = 0;
        if (j->inits > 0) cb[n++].commandBuffer = j->r->init;
        if (j->uploads) cb[n++].commandBuffer = j->r->upload;
        cb[n++].commandBuffer = j->r->cmd;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.commandBufferInfoCount = n;
        si.pCommandBufferInfos = cb;
        late_fix_wait_unlocked();   //
        std::lock_guard lk(dev.present_mu);
        queue_flush_locked();
        gpu_time_segment_submitted(*j->r);   // (the blits' places among the timed recordings)
        const VkResult res = submit_or_retry(si, j->r->fence);   // (hookshot: a refusal for want of memory made again)
        j->r->submitted = res == VK_SUCCESS;
        if (res != VK_SUCCESS) logf("vk: segment submit %llu failed (%d)", (unsigned long long)j->r->value, (int)res);
        if (res == VK_ERROR_DEVICE_LOST) device_lost();
    };
    // (RX_VK_EARLY_GPU) the segments that are done go out in order, by whichever thread finishes one; held while the dry pass runs
    // once it has recorded an upload (rd must go first), until the replay thread submits rd at the dry pass's end (sub_after_rd)
    std::mutex sub_mu;
    size_t next_sub = 0;
    bool sub_after_rd = false;
    const auto submit_ready = [&] {   // (sub_mu held)
        while (next_sub < jobs.size() && jobs[next_sub]->done.load()) {
            if (!sub_after_rd && dry_uploaded.load(std::memory_order_acquire)) return;
            submit_job(jobs[next_sub++].get());
        }
    };
    std::condition_variable seg_cv;
    std::mutex seg_mu;
    const auto now_ns = [] { return (int64_t)std::chrono::steady_clock::now().time_since_epoch().count(); };
    const auto record_job = [&seg_cv, &seg_mu, &now_ns, &sub_mu, &submit_ready, egpu](Job* j) {   // (as replay_handle's)
        j->t0 = now_ns();
        work_seg_begin();   // (par_active is on while these segments record)
        texture_set_segment(j->order);
        texture_segment_worker(j->index);
        texture_begin_recording(j->r->upload, j->r->value);
        draw_begin_recording(j->r->slot, j->r->cmd, j->r->value);
        VkCommandBuffer c = j->r->cmd;
        const int64_t ta = par_stats_on ? now_ns() : 0;
        record_segment(c, j->g);
        const int64_t tc = par_stats_on ? now_ns() : 0;
        j->uploads = texture_uploads_recorded();
        j->inits = texture_segment_end(j->index);
        texture_segment_worker(-1);
        gpu_time_end(*j->r, j->uploads);   // (as replay_handle's)
        host_read_barrier(c);   // (as the other segments': what the GPU wrote, seen by the host's reads)
        vkEndCommandBuffer(j->r->upload);
        vkEndCommandBuffer(j->r->cmd);
        work_seg_end();   //
        j->t1 = now_ns();
        par_stats_segment_parts(j->t0, ta, tc, j->t1);
        {
            std::lock_guard l(seg_mu);
            j->done = true;
        }
        seg_cv.notify_all();
        if (egpu) {
            std::lock_guard sl(sub_mu);
            submit_ready();
        }
    };
    // the dry pass, handing segments out as it passes their ends
    size_t open = 0;   // (the segment whose end is looked for)
    jobs[0]->s = state;
    jobs[0]->g = {h, nullptr, &jobs[0]->s, ub0, 0, serial0};
    SegEarlyHook hook;
    hook.wants = [&](int64_t work, const uint8_t* at) { return (int)open + 1 < nplan && work >= plan[open] && at != jobs[open]->g.p; };
    hook.on_point = [&](const ParPoint& q) {
        if (!hook.wants(q.work, q.p)) return;
        Job* j = jobs[open].get();
        j->g.end = q.p;
        {
            std::lock_guard lk(par_pool.mu);
            par_pool.jobs.push_back([j, &record_job] { record_job(j); });
            par_pool.pending++;
        }
        par_pool.cv.notify_one();
        jt_mark(26, open);
        Job* n = jobs[++open].get();
        n->s = q.s;
        n->g = {q.p, nullptr, &n->s, q.ub, q.cond, serial0 + (uint64_t)q.work};
    };
    if (early) {
        par_active = true;   // (the dry pass's texture paths and marks take ParLock from here on)
        ub_growing.store(true, std::memory_order_relaxed);
        seg_early_hook = &hook;
    }
    rmode = Mode::Dry;
    VkCommandBuffer none = cmd;
    suspend_ok = may_stop;
    jt_mark(13, 0);
    replay(none, state, h);
    jt_mark(14, 0);
    if (egpu) texture_upload_watch(nullptr);
    suspend_ok = false;
    rmode = Mode::Serial;
    seg_early_hook = nullptr;
    early_wall.mark(5);
    ub_growing.store(false, std::memory_order_release);   // (nothing is appended from here on)
    const uint8_t* const occ_stop = suspend_at ? nullptr : occ_stop_at;
    const uint8_t* const stop = suspend_at ? suspend_at : occ_stop;
    const int64_t total = dry_work;
    *total_out = total;
    static double dry_ms, seg_ms, segs_n, slow_ms, wait_ms, self_ms, early_n;
    static int64_t timed, whole_n;
    dry_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dry0).count();
    if (wall_fine_on()) rwall.dry_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_dry0).count();
    const size_t used = open + 1;
    if (early)
        for (size_t k = used; k < jobs.size(); k++) texture_segment_end((int)k);   // (never reached: their init command buffers ended)
    if (used < 2 || !early) {   // (no boundary reached, or no init command buffer: the whole command buffer here, into rec_cur)
        if (early) {
            texture_segment_end(0);
            par_active = false;
            texture_segments_begin(nullptr, 0);
        }
        whole_n++;
        Segment whole = jobs[0]->g;
        whole.end = stop;
        {
            WallAdd wall_whole(rwall.whole_ns);
            rwall.whole++;
            record_segment(cmd, whole);
        }
        *cut = false;
        return occ_stop ? occ_go_on(cmd, occ_stop, may_stop, nullptr) : stop;
    }
    *cut = true;
    Job* last = jobs[open].get();
    last->g.end = stop;
    if (egpu) {   // (rd, the dry pass's uploads, ahead of every segment not submitted yet; those submitted before it never saw them)
        std::lock_guard sl(sub_mu);
        submit_recording(*rd, false);
        sub_after_rd = true;
        submit_ready();
        rec_parts++;
    } else {
        submit_recording(r, false);   // (what came before, its uploads with the pre-sync: ahead of every segment in the queue)
        for (size_t k = 0; k < used; k++) ts_recseq[jobs[k]->r->slot] = rec_seq;   // (RX_VK_TL: shown with r, the recording they follow)
    }
    jt_mark(20, 0);
    early_wall.mark(6);
    const auto t_seg0 = std::chrono::steady_clock::now();
    jt_mark(17, used);
    record_job(last);   // (the replay thread's: the last one)
    jt_mark(18, 0);
    early_wall.mark(7);
    int64_t pt_wait_ns = 0;
    for (size_t k = 0; k < used; k++) {   // (in order, each as soon as it is done)
        Job* j = jobs[k].get();
        while (self_help && !j->done.load()) {   // (RX_VK_PARALLEL_SELF=2: one no worker has started yet)
            std::function<void()> fn;
            {
                std::lock_guard lk(par_pool.mu);
                if (par_pool.jobs.empty()) break;
                fn = std::move(par_pool.jobs.front());
                par_pool.jobs.pop_front();
            }
            fn();
            std::lock_guard lk(par_pool.mu);
            if (--par_pool.pending == 0) par_pool.done.notify_all();
        }
        {
            const int64_t w0 = now_ns();
            jt_mark(9, (uint64_t)j->index);
            std::unique_lock l(seg_mu);
            seg_cv.wait(l, [j] { return j->done.load(); });
            pt_wait_ns += now_ns() - w0;
            jt_mark(10, (uint64_t)j->index);
        }
        if (egpu) {   // (this one and those before it, when no other thread has: all of them are done)
            std::lock_guard sl(sub_mu);
            submit_ready();
        } else {
            submit_job(j);
        }
    }
    {
        std::unique_lock lk(par_pool.mu);
        par_pool.done.wait(lk, [] { return par_pool.pending == 0; });
    }
    if (init_lost)
        replay_fatal("a parallel replay segment's init command buffer could not be ended: new images lost their first layout "
                     "change, and the work after it may use them");
    early_wall.mark(8);
    par_active = false;
    texture_segments_begin(nullptr, 0);
    seg_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_seg0).count();
    if (wall_fine_on()) rwall.seg_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_seg0).count(), rwall.big++;
    segs_n += (double)used, early_n += (double)(used - 1);
    {
        int64_t hi = 0;
        for (size_t k = 0; k < used; k++) hi = std::max(hi, jobs[k]->t1 - jobs[k]->t0);
        slow_ms += hi / 1e6, wait_ms += pt_wait_ns / 1e6, self_ms += (last->t1 - last->t0) / 1e6;
    }
    {   // (the profile, as replay_handle keeps it)
        double from[kMaxBatch], to[kMaxBatch], ms[kMaxBatch];
        size_t m = 0;
        for (size_t k = 0; k < used && k < (size_t)kMaxBatch && total > 0; k++, m++) {
            const uint64_t end = k + 1 < used ? jobs[k + 1]->g.serial : serial0 + (uint64_t)total;
            from[m] = (double)(jobs[k]->g.serial - serial0) / (double)total, to[m] = (double)(end - serial0) / (double)total;
            ms[m] = (jobs[k]->t1 - jobs[k]->t0) / 1e6;
        }
        mkwvk::segment_plan::profile_add(profile, from, to, ms, m, 0.25);
    }
    if (++timed % 240 == 0) {
        logf("vk: parallel replay, early segments (RX_VK_SEG_EARLY), a command buffer: dry pass %.2f ms (segments out during it "
             "%.1f), segments after it %.2f ms (%.1f of them), slowest %.2f ms, the replay thread recorded %.2f ms and waited %.2f ms; "
             "%lld recorded whole (no boundary reached)", dry_ms / 240, early_n / 240, seg_ms / 240, segs_n / 240, slow_ms / 240,
             self_ms / 240, wait_ms / 240, (long long)whole_n);
        dry_ms = seg_ms = segs_n = slow_ms = wait_ms = self_ms = early_n = 0;
        par_stats_log();
    }
    texture_set_segment(++par_seq);
    Recording& nr = begin_recording(rec_cur_value);
    full_barrier(nr.cmd);
    rec_cur = &nr;
    cmd = nr.cmd;
    rec_parts += (int)used + 1;
    return occ_stop ? occ_go_on(cmd, occ_stop, may_stop, jobs[used - 1]->r) : stop;
}
// RX_VK_DRY_LEAN: see replay_handle
bool dry_lean_on() {
    static const bool on = [] {
        const char* e = getenv("RX_VK_DRY_LEAN");
        const bool v = e && atoi(e) != 0;
        logf("vk: the dry pass's states: %s (RX_VK_DRY_LEAN)", v ? "kept only where a boundary can be" : "every point, as before");
        return v;
    }();
    return on;
}

std::recursive_mutex par_mu;

/// The draws and dispatches from p to its stream's end, counted without applying anything (the dry pass's dry_work);
/// stops at `most`, when how many more there are is not asked.
int64_t stream_work(const uint8_t* p, int64_t most) {
    int64_t n = 0;
    for (;;) {
        nvn_op_hdr h;
        std::memcpy(&h, p, sizeof h);
        if (h.op == NOP_END) return n;
        if (op_is_work(h.op) && ++n >= most) return n;
        p += h.size;
    }
}

/// (hookshot) RX_VK_PARALLEL with the uniform snapshot on (RX_VK_UB_SNAPSHOT: every Intel GPU by default): each command
/// buffer recorded on the replay thread, as with RX_VK_PARALLEL off (logged once). A segment is recorded on its thread
/// while those before it still are, so the snapshot's guard there (uniform_range: a uniform range the GPU writes earlier
/// in the submission is bound in place, vk_memory.cpp's memory_gpu_written) could not know their GPU writes, nor the ones
/// before the segments, and the segment's uniform copy would not have the GPU's bytes
static bool par_snapshot_serial(int threads) {
    static const bool on = [threads] {
        if (threads < 2 || !uniform_snapshot_on()) return false;
        logf("vk: parallel replay (RX_VK_PARALLEL=%d): command buffers recorded on the replay thread, the uniform snapshot "
             "is on (RX_VK_UB_SNAPSHOT): a segment's uniform copies could not see the GPU writes before them",
             threads);
        return true;
    }();
    return on;
}

/// A command buffer of the submission from h (its start, or where it stopped at a wait): on the replay thread alone
/// (RX_VK_PARALLEL off, one too short for two segments, dumps and profiles, the uniform snapshot on), or its dry pass
/// there and its segments on the threads. may_stop: up to a wait not met, which it returns (null: the command buffer done)
const uint8_t* replay_handle(VkCommandBuffer& cmd, const uint8_t* h, bool may_stop = false) {
    const int threads = par_threads();
    NvnState& state = *cur_state;
    suspend_at = nullptr;
    // (hookshot) the occlusion queries this command buffer may take (RX_VK_OCC_QUERIES), numbered from the counter all the
    // queues share: set before its dry pass's start state is copied, so the workers number theirs as the dry pass does; the
    // counter goes on from where the dry pass (or the replay here) left this state, however this returns
    occ_begin_buffer(state, rec_cur_value, occ_open_oldest(&state));
    struct OccEnd {
        const NvnState& s;
        ~OccEnd() { occ_end_buffer(s); }
    } occ_end{state};
    if (threads < 2 || !rec_cur || dump_on() || gprof_on() || probe_serial() || par_snapshot_serial(threads)) {
        suspend_ok = may_stop;
        replay(cmd, state, h);
        suspend_ok = false;
        return suspend_at;
    }
    // (segments of ~RX_VK_PARALLEL_SEG draws and dispatches (default 160), taken in order by the threads and submitted
    // as each is done with those before it: the GPU starts on the first while the rest are recorded)
    static const int64_t per = getenv("RX_VK_PARALLEL_SEG") ? std::max(32, atoi(getenv("RX_VK_PARALLEL_SEG"))) : fine_segments() ? 128 : 160;
    // (too little work for two segments: recorded here, as with parallel replay off. The dry pass would apply the stream's
    // state for nothing, and its one segment would walk the stream again here on this thread.)
    static int64_t shorts;
    if (stream_work(h, 2 * per) < 2 * per) {
        shorts++;
        const bool was_split = split_off;
        split_off = true;   // (the segment the dry pass would have recorded is one, and a segment is not split)
        suspend_ok = may_stop;
        replay(cmd, state, h);
        suspend_ok = false;
        split_off = was_split;
        return suspend_at;
    }
    // the dry pass: the state at every 64 draws and dispatches, the render conditions, the uniform updates
    const auto t_dry0 = std::chrono::steady_clock::now();
    const NvnState start = state;
    const size_t ub0 = ub_updates.size();
    par_points.clear(), par_conds.clear(), dry_work = 0;
    dry_marks_begin();
    rmode = Mode::Dry;
    VkCommandBuffer none = cmd;
    suspend_ok = may_stop;
    replay(none, state, h);
    suspend_ok = false;
    rmode = Mode::Serial;
    const uint8_t* const stop = suspend_at;   // (the segments end there)
    const int64_t total = dry_work;
    static double dry_ms, seg_ms, segs_n;
    static int64_t timed;
    dry_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dry0).count();
    const int nseg = (int)std::clamp<int64_t>(total / per, 1, seg_cap());
    std::vector<Segment> seg;
    seg.push_back({h, nullptr, &start, ub0, 0});
    for (int k = 1; k < nseg; k++) {
        const int64_t want = total * k / nseg;
        const ParPoint* at = nullptr;
        for (const ParPoint& q : par_points)
            if (q.work >= want) { at = &q; break; }
        if (!at || at->p == seg.back().p) continue;
        seg.back().end = at->p;
        seg.push_back({at->p, nullptr, &at->s, at->ub, at->cond});
    }
    seg.back().end = stop;
    if (seg.size() == 1) {   // (one segment: here, as the dry pass left the reports and conditions)
        record_segment(cmd, seg[0]);
        return stop;
    }
    if (!par_pool.started) {
        par_pool.started = true;
        for (int i = 0; i < threads; i++) std::thread(par_worker_main, i).detach();
        logf("vk: parallel replay: big command buffers recorded on %d threads in segments of ~%lld draws and dispatches",
             threads, (long long)per);
    }
    // what came before, to the GPU now (its uploads included); then each segment into a recording of its own (its
    // command buffer and uploads), recorded in order by the threads and submitted in order as they are done
    Recording& r = *rec_cur;
    end_pass(r.cmd);
    full_barrier(r.cmd);
    memory_sync_mirrors_before_segments();   // (in r's uploads, ahead of every segment)
    submit_recording(r, false);
    static uint64_t seq = 0;   // (the segments' order, across submissions: a recording's uploads precede it)
    const uint64_t base = ++seq;
    seq += seg.size();
    memory_rb_segments_begin(base, seg.size());   // (their order in the queue: memory_rb_refresh)
    struct Job {
        Recording* r;
        Segment g;
        uint64_t order;
        int index = 0;
        std::atomic<bool> done{false};
        bool uploads = false;
        int inits = 0;   // (texture_segment_end's)
    };
    std::vector<std::unique_ptr<Job>> jobs;
    for (size_t k = 0; k < seg.size(); k++) {
        auto j = std::make_unique<Job>();
        j->r = &begin_segment_recording(rec_cur_value);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandPool(dev.device, j->r->upload_pool, 0);
        vkBeginCommandBuffer(j->r->upload, &bi);
        j->g = seg[k];
        j->order = base + k;
        j->index = (int)k;
        jobs.push_back(std::move(j));
    }
    std::vector<VkCommandBuffer> inits;   // (each one's init command buffer, begun)
    {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        for (auto& jp : jobs)
            if (jp->r->init && vkResetCommandPool(dev.device, jp->r->init_pool, 0) == VK_SUCCESS &&
                vkBeginCommandBuffer(jp->r->init, &bi) == VK_SUCCESS)
                inits.push_back(jp->r->init);
            else
                break;
    }
    if (inits.size() != jobs.size()) {
        // (one could not be begun: no worker has started, and the segments are not recorded. The whole command buffer is
        // recorded here as one segment, as a short one is, into what follows (the post-part as below): its new images are
        // made in queue order. The segments' recordings go unsubmitted; their slots come round again as unused ones)
        static int told;
        if (told++ < 8) logf("vk: parallel replay: a segment's init command buffer could not be begun: recorded on the replay thread");
        texture_set_segment(++seq);
        Recording& rest = begin_recording(rec_cur_value);
        full_barrier(rest.cmd);
        rec_cur = &rest;
        cmd = rest.cmd;
        rec_parts++;
        Segment whole = seg[0];
        whole.end = stop;
        record_segment(cmd, whole);
        return stop;
    }
    texture_segments_begin(inits.data(), inits.size());
    const auto t_seg0 = std::chrono::steady_clock::now();
    const int64_t wait0 = par_lock_wait_ns.load(), locks0 = par_locks.load();
    par_active = true;
    std::condition_variable seg_cv;
    std::mutex seg_mu;
    {
        std::lock_guard lk(par_pool.mu);
        for (auto& jp : jobs) {
            Job* j = jp.get();
            par_pool.jobs.push_back([j, &seg_cv, &seg_mu] {
                texture_set_segment(j->order);
                texture_segment_worker(j->index);
                texture_begin_recording(j->r->upload, j->r->value);
                draw_begin_recording(j->r->slot, j->r->cmd, j->r->value);
                VkCommandBuffer c = j->r->cmd;
                record_segment(c, j->g);
                j->uploads = texture_uploads_recorded();
                j->inits = texture_segment_end(j->index);   // (after its last new image)
                texture_segment_worker(-1);
                host_read_barrier(c);
                vkEndCommandBuffer(j->r->upload);
                vkEndCommandBuffer(j->r->cmd);
                {
                    std::lock_guard l(seg_mu);
                    j->done = true;
                }
                seg_cv.notify_all();
            });
            par_pool.pending++;
        }
    }
    par_pool.cv.notify_all();
    bool init_lost = false;
    for (auto& jp : jobs) {   // (in order, each as soon as it is done)
        Job* j = jp.get();
        {
            FpsRtWait rt_wait(fps_stats::kRtWorkers);   // (the FPS counter's render lane)
            std::unique_lock l(seg_mu);
            seg_cv.wait(l, [j] { return j->done.load(); });
        }
        if (j->inits < 0) init_lost = true;   // (its init command buffer could not be ended: the replay stops, below)
        if (init_lost) continue;              // (nothing from it on is submitted)
        draw_flush_recording(j->r->slot);
        VkCommandBufferSubmitInfo cb[3]{{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO},
                                        {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}};
        uint32_t n = 0;
        if (j->inits > 0) cb[n++].commandBuffer = j->r->init;   // (new images made in it or after it: first in the queue)
        if (j->uploads) cb[n++].commandBuffer = j->r->upload;
        cb[n++].commandBuffer = j->r->cmd;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.commandBufferInfoCount = n;
        si.pCommandBufferInfos = cb;
        std::lock_guard lk(dev.present_mu);
        queue_flush_locked();
        si.pNext = reflex_submit_id(nullptr);   // (NVIDIA Reflex: the frame being made; after the flush, which tags its own)
        VkSemaphoreSubmitInfo sparse_wait;   // (hookshot: after the sparse mirrors' binds, RX_VK_MIRROR_SPARSE)
        if (memory_sparse_wait(&sparse_wait)) si.waitSemaphoreInfoCount = 1, si.pWaitSemaphoreInfos = &sparse_wait;
        const VkResult res = submit_or_retry(si, j->r->fence);   // (hookshot: a refusal for want of memory made again)
        j->r->submitted = res == VK_SUCCESS;
        static std::atomic<int> failures{0};   // (hookshot: once lost, every segment failed: thousands of lines in a log)
        if (res != VK_SUCCESS && failures++ < 8)
            logf("vk: segment submit %llu failed (%d)", (unsigned long long)j->r->value, (int)res);
        if (res == VK_ERROR_DEVICE_LOST) device_lost();
    }
    {
        FpsRtWait rt_wait(fps_stats::kRtWorkers);
        std::unique_lock lk(par_pool.mu);
        par_pool.done.wait(lk, [] { return par_pool.pending == 0; });
    }
    if (init_lost)   // (every worker done: none is left with a job)
        replay_fatal("a parallel replay segment's init command buffer could not be ended: new images lost their first layout "
                     "change, and the work after it may use them");
    par_active = false;
    texture_segments_begin(nullptr, 0);
    seg_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_seg0).count();
    segs_n += (double)seg.size();
    static double wait_ms, locks_n;
    wait_ms += (par_lock_wait_ns.load() - wait0) / 1e6, locks_n += (double)(par_locks.load() - locks0);
    if (++timed % 240 == 0) {
        logf("vk: parallel replay, a command buffer: dry pass %.2f ms, segments %.2f ms (%.1f of them), lock taken %.0f "
             "times, waited for %.2f ms (all threads); %lld short ones recorded here", dry_ms / 240, seg_ms / 240,
             segs_n / 240, locks_n / 240, wait_ms / 240, (long long)shorts);
        dry_ms = seg_ms = segs_n = wait_ms = locks_n = 0;
        shorts = 0;
    }
    // what follows, after them all (the uniform updates are written to guest memory at the submission's end)
    texture_set_segment(++seq);
    Recording& nr = begin_recording(rec_cur_value);
    full_barrier(nr.cmd);
    rec_cur = &nr;
    cmd = nr.cmd;
    rec_parts += (int)seg.size() + 1;
    return stop;
}

/// (RX_VK_OCC_REAL=4) a dry pass stopped at a condition on a report not read yet: what it covered is recorded (and
/// submitted when last is its last segment's recording, else submitted here); the GPU waited for, the reports read, and the
/// command buffer replayed on from the condition
const uint8_t* occ_go_on(VkCommandBuffer& cmd, const uint8_t* at, bool may_stop, Recording* last) {
    if (last) recording_wait(*last);
    if (last || occ_sync(cmd)) occ_resolve_all();
    else occ_give_up();
    return replay_handle(cmd, at, may_stop);
}

// ---- the rx_* functions ----

// ---- a benchmark (RX_VK_BENCH=<present>[,<times>], default 50 times): the frame before that present replayed again and
// again, the game held (its next present waits) and the recordings kept (the replayed value the host frees them by
// is held at the frame's start); the replay's CPU time each time, logged. The same frame every time: what a change to
// the replay costs or saves, without the title scenes' differences. ----
std::vector<std::vector<uint64_t>> frame_submits;   // (this frame's submits' handles)
uint64_t frame_first_value = 0;
std::atomic<uint64_t> replayed_hold{0};             // (0: none)

void bench_replay(const std::vector<uint64_t>& handles) {
    const uint64_t value = pushed.load();   // (reached already: no slot waits on it)
    rec_value_now = value;
    ub_first_of_recording = ub_updates.size();
    Recording& first = begin_recording(value);
    memory_begin_submission();
    memory_gpu_writes_begin();   // (RX_VK_UB_SNAPSHOT's guard: a submission's start, not a part's: split_part)
    full_barrier(first.cmd);
    rec_cur = &first, rec_cur_value = value, rec_parts = 0, part_work = 0, part_end_wanted = false;
    rec_open(value);
    VkCommandBuffer cmd = first.cmd;
    for (uint64_t h : handles)
        if (h) {
            occ_begin_buffer(*cur_state, value, occ_open_oldest(cur_state));   // (hookshot: as replay_handle)
            replay(cmd, *cur_state, reinterpret_cast<const uint8_t*>(h));
            occ_end_buffer(*cur_state);
        }
    Recording& r = *rec_cur;
    rec_cur = nullptr;
    end_pass(r.cmd);
    flush_ub_updates(r.cmd);
    memory_end_submission(r.cmd, r.part);
    full_barrier(r.cmd);
    submit_recording(r, false);
    rec_close();
}

void bench_frame(int64_t presents) {
    static int64_t at = -2;
    static int times = 50;
    if (at == -2) {
        at = -1;
        if (const char* e = getenv("RX_VK_BENCH")) {
            at = atoll(e);
            if (const char* c = strchr(e, ',')) times = std::max(1, atoi(c + 1));
        }
    }
    if (at < 0 || presents + 1 != at || frame_submits.empty()) return;
    replayed_hold = std::max<uint64_t>(1, frame_first_value - 1);
    int64_t draws0 = 0, sk = 0, di = 0, pl = 0;
    draw_stats(&draws0, &sk, &di, &pl);
    std::vector<double> ms;
    int64_t prof0[PROF_COUNT];
    prof_full++;   // (every phase timed, not one draw in eight: the A/B differences are small)
    for (int i = 0; i < PROF_COUNT; i++) prof0[i] = prof_ns[i];
    const size_t ub_before = ub_updates.size();
    queue_flush();
    for (int i = 0; i < times; i++) {
        vkDeviceWaitIdle(dev.device);
        ub_updates.resize(ub_before);   // (the last time's uniform updates are written: as if pruned)
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto& h : frame_submits) bench_replay(h);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    vkDeviceWaitIdle(dev.device);
    prof_full--;
    int64_t draws1 = 0;
    draw_stats(&draws1, &sk, &di, &pl);
    replayed_hold = 0;
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0;
    for (double v : ms) sum += v;
    logf("vk: bench: the frame before present %lld (%zu submits, %lld draws) replayed %d times: median %.2f ms, min %.2f, "
         "mean %.2f, max %.2f", (long long)at, frame_submits.size(), (long long)((draws1 - draws0) / times), times,
         sorted[sorted.size() / 2], sorted.front(), sum / times, sorted.back());
    static const char* names[PROF_COUNT] = {"submit", "present", "draw", "descriptors", "texture lookups", "uploads",
                                            "pipelines", "programs", "commands", "dispatch", "memory sync",
                                            "queue submit", "slot wait"};
    char line[600];
    int n = snprintf(line, sizeof line, "vk: bench: a replay's time (ms):");
    for (int i = 2; i < PROF_COUNT; i++)
        n += snprintf(line + n, sizeof line - n, " %s %.2f%s", names[i], (prof_ns[i] - prof0[i]) / (double)times / 1e6,
                      i + 1 < PROF_COUNT ? "," : "");
    logf("%s", line);
}

/// (hookshot) Slow submits and presents after the load, ranked by the replay phase that took their time.
/// The per-event lines stop at 80; this keeps counting and prints a rollup on the first and every 10, so a session
/// still says whether the hitches were uploads, pipelines or memory sync. Measure only.
void hitch_note(const char* what, double ms, const double phase[PROF_COUNT]) {
    struct Acc {
        int n = 0, submits = 0, presents = 0;
        double ms = 0;
        double phase[PROF_COUNT]{};
    };
    static Acc hitch;
    static std::mutex mu;
    int n = 0, submits = 0, presents = 0;
    double total = 0, top = 0, second = 0;
    int top_i = -1, second_i = -1;
    double phases[PROF_COUNT]{};
    {
        std::lock_guard<std::mutex> lk(mu);
        hitch.n++;
        hitch.ms += ms;
        if (what && !strcmp(what, "present")) hitch.presents++;
        else hitch.submits++;
        for (int i = 2; i < PROF_COUNT; i++) hitch.phase[i] += phase[i];
        if (hitch.n != 1 && hitch.n % 10 != 0) return;
        n = hitch.n, submits = hitch.submits, presents = hitch.presents, total = hitch.ms;
        std::memcpy(phases, hitch.phase, sizeof phases);
    }
    static const char* names[PROF_COUNT] = {"submit", "present", "draw", "descriptors", "texture lookups",
                                            "uploads", "pipelines", "programs", "commands", "dispatch",
                                            "memory sync", "queue submit", "slot wait"};
    for (int i = 2; i < PROF_COUNT; i++) {
        if (phases[i] > top) second = top, second_i = top_i, top = phases[i], top_i = i;
        else if (phases[i] > second) second = phases[i], second_i = i;
    }
    if (top_i < 0 || top < 1.0)
        logf("vk: hitch rollup: %d slow (%d submits, %d presents), %.0f ms, no phase over 1 ms", n, submits, presents, total);
    else if (second_i >= 0 && second >= 1.0)
        logf("vk: hitch rollup: %d slow (%d submits, %d presents), %.0f ms, mostly %s (%.0f ms), then %s (%.0f ms)",
             n, submits, presents, total, names[top_i], top, names[second_i], second);
    else
        logf("vk: hitch rollup: %d slow (%d submits, %d presents), %.0f ms, mostly %s (%.0f ms)",
             n, submits, presents, total, names[top_i], top);
}

/// A slow submit (over 15 ms): what took its time, logged (80 of them). The stalls in it are exact (PROF, PROF_EXACT); the
/// per-draw phases' cheap part is sampled, one draw in eight (PROF_HOT): an estimate, exact with RX_VK_PROF_ALL=1
struct SlowWatch {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    int64_t prof0[PROF_COUNT];
    int64_t ms0[4];   // (hookshot: memory sync's parts and the lock wait, at the start)
    const char* what;
    const bool active = presents_now >= 600;
    explicit SlowWatch(const char* w) : what(w) {
        if (!active) return;
        for (int i = 2; i < PROF_COUNT; i++) prof0[i] = prof_ns[i].get();
        for (int i = 0; i < 3; i++) ms0[i] = memsync_part_ns[i].load(std::memory_order_relaxed);
        ms0[3] = memsync_lock_wait_ns.load(std::memory_order_relaxed);
    }
    ~SlowWatch() {
        if (!active) return;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < 15.0 || presents_now < 600) return;   // (after the first load)
        double phase[PROF_COUNT]{};
        for (int i = 2; i < PROF_COUNT; i++) phase[i] = (prof_ns[i] - prof0[i]) / 1e6;
        hitch_note(what, ms, phase);
        static int logged;
        if (logged >= 80) return;
        logged++;
        static const char* names[PROF_COUNT] = {"submit", "present", "draw", "descriptors", "texture lookups",
                                                "uploads", "pipelines", "programs", "commands", "dispatch",
                                                "memory sync", "queue submit", "slot wait"};
        char line[600];
        int k = snprintf(line, sizeof line, "vk: slow %s: %.1f ms:", what, ms);
        for (int i = 2; i < PROF_COUNT; i++) {
            if (phase[i] >= 1.0) k += snprintf(line + k, sizeof line - k, " %s %.1f", names[i], phase[i]);
        }
        {   // (hookshot: which part of "memory sync" it was: the pools' refresh [and its lock waits], the capture, the written pages collected.
            // Counted across threads, so a part another thread spent meanwhile shows too)
            const double a = (memsync_part_ns[0].load(std::memory_order_relaxed) - ms0[0]) / 1e6,
                         b = (memsync_part_ns[1].load(std::memory_order_relaxed) - ms0[1]) / 1e6,
                         c = (memsync_part_ns[2].load(std::memory_order_relaxed) - ms0[2]) / 1e6,
                         w = (memsync_lock_wait_ns.load(std::memory_order_relaxed) - ms0[3]) / 1e6;
            if (a + b + c >= 1.0 || w >= 1.0)
                k += snprintf(line + k, sizeof line - k, " [memory sync: maps %.1f (lock waits %.1f), capture %.1f, collect %.1f]", a, w, b, c);
        }
        logf("%s", line);
    }
};

/// The memory work before a submission's recording: the pools mapped, its capture taken, the other writes collected
/// (replay_mu held). In the order of the submits: one behind its queue's stopped one (submit_job) has it done as it comes.
void submit_memory_work(uint64_t value) {
    WallAdd wall_mem(rwall.mem_ns);
#ifdef TRACY_ENABLE
    HT_SCOPE("submit_memory_work");
#endif
    PROF(PROF_MEMSYNC);
    jt_mark(15, value);
    struct JtEnd { ~JtEnd() { jt_mark(16, 0); } } jt_end;
    const auto m0 = std::chrono::steady_clock::now();
    memory_sync();
    const auto m1a = std::chrono::steady_clock::now();
    memsync_part_ns[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(m1a - m0).count();
    // (the capture takes only the mirrored pools' pages, the collection the rest: skipping it whenever a capture was
    // taken, which with the replay thread is at every submit, left every other CPU write uncollected. Textures never
    // saw them: MK World's font cache kept its first glyphs (the menus lost letters), CPU-streamed terrain went stale)
    const auto m1 = std::chrono::steady_clock::now();
    memory_take_capture(value, completed());
    const auto m2 = std::chrono::steady_clock::now();
    memory_collect_written();
    const auto m3 = std::chrono::steady_clock::now();
    memsync_part_ns[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(m2 - m1).count();
    memsync_part_ns[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(m3 - m2).count();
    {   // (every 600 submits: what the memory work before a submit's recording cost)
        static double a, b, c;
        static int64_t k;
        a += std::chrono::duration<double, std::milli>(m1 - m0).count();
        b += std::chrono::duration<double, std::milli>(m2 - m1).count();
        c += std::chrono::duration<double, std::milli>(m3 - m2).count();
        if (++k % 600 == 0) {
            logf("vk: a submit's memory work (ms): maps %.2f, capture taken %.2f, writes collected %.2f", a / 600, b / 600, c / 600);
            a = b = c = 0;
        }
    }
}

// ---- the game's queues ordered by their events ----
// MK World's two worker queues pass stages back and forth with events: a command buffer waits (nvnCommandBufferWaitEvent)
// for the other queue's to reach a stage, several times a frame, and the two are submitted one after the other.
// Replayed one after the other on the one Vulkan queue, the first went past waits for work the second had not done
// yet: what that work writes was read before it was written (55% of the waits). Now a submission is replayed as it
// comes as far as its waits allow, and what it recorded goes to the GPU; at a wait not met it stops, and its queue's
// later submits wait behind it (their memory work done in turn). Once another queue's work has signalled the event
// it goes on from there, as on the console, where it would have waited on the GPU. A present, a released texture or
// program, a wait from another thread for the replay, or RX_VK_HOLD_MS (default 3) with one stopped, and the stopped
// ones go on as their waits allow, the oldest past its wait whenever all of them wait. The timeline is signalled as
// far as no older submission is unfinished. (Holding a whole submission until the other queue's came, as this did
// first, had the registrations made meanwhile reach its draws before it: the menus' scenes in rectangles of fog.)
struct LaterSubmit {
    std::vector<uint64_t> handles;
    uint64_t value = 0;
};
struct QueueRun {   // (a queue stopped at a wait)
    bool waiting = false;
    std::vector<uint64_t> handles;   // (the submission it stopped in)
    int cb = 0;                      // (its command buffer, and the wait there)
    const uint8_t* p = nullptr;
    uint64_t value = 0;
    std::deque<LaterSubmit> later;   // (its queue's submits since)
    uint64_t ahead_upto = 0;         // (hookshot: its submissions up to this value walked by occ_visible_ahead already)
    bool blocked() const { return waiting || !later.empty(); }
    uint64_t oldest() const { return waiting ? value : later.front().value; }
};
std::unordered_map<uint64_t, QueueRun> qruns;
std::atomic<int> blocked_n{0};                      // (queues stopped, for the other threads)
std::chrono::steady_clock::time_point blocked_since;
uint64_t done_max = 0, signaled = 0;                // (the newest value recorded whole; the timeline's last signal)
int64_t st_stops, st_resumed, st_forced_waits, st_forced[4], st_resume_captures;   // (made to go by: presents, releases, waits, time)
double st_stopped_max_ms;
int hold_ms() {
    // (50: with the replay caught up, the worker queue's submission came up to 16 ms before the one that lets it on;
    // at 3 ms it went past its waits ~80 times a race. The game never waits for it before that submit)
    static const int v = getenv("RX_VK_HOLD_MS") ? std::max(0, atoi(getenv("RX_VK_HOLD_MS"))) : 50;
    return v;
}
const uint8_t* op_next(const uint8_t* p) {
    nvn_op_hdr h;
    std::memcpy(&h, p, sizeof h);
    return p + h.size;
}
/// (hookshot) Work the replay holds back (its queue stopped at an event wait): handles[cb] from op p (null: its start) and the
/// command buffers after it. Its samples-passed reports read "visible" until it is replayed: the slot may hold an older count,
/// and the game may read it before the replay gets there (occ_slot_visible)
void occ_visible_ahead(const uint64_t* handles, int n, int cb, const uint8_t* p) {
    if (!occ_pool) return;
    const auto t0 = std::chrono::steady_clock::now();
    for (; cb < n; cb++, p = nullptr) {
        if (!handles[cb]) continue;
        for (const uint8_t* o = p ? p : reinterpret_cast<const uint8_t*>(handles[cb]);; o = op_next(o)) {
            nvn_op_hdr h;
            std::memcpy(&h, o, sizeof h);
            if (h.op == NOP_END) break;
            if (h.op != NOP_REPORT_COUNTER) continue;
            const auto c = payload<nvn_p_counter>(o + 8, h.size - 8);
            if (c.type == NVN_COUNTER_TYPE_SAMPLES_PASSED) occ_slot_visible(c.addr);   // (not a flag: NVN_COUNTER_WRITTEN)
        }
    }
    occ_st.walks++;
    occ_st.walk_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
nvn_p_wait_event wait_at(const uint8_t* p) {   // (p: a wait op)
    nvn_op_hdr h;
    std::memcpy(&h, p, sizeof h);
    return payload<nvn_p_wait_event>(p + 8, h.size - 8);
}
bool wait_met_at(const uint8_t* p) {
    const nvn_p_wait_event w = wait_at(p);
    ev_cpu_apply();
    return event_wait_met(event_value_in(ev_values, w.gpu), w.mode, w.value);
}
uint64_t oldest_unfinished() {
    uint64_t m = UINT64_MAX;
    for (const auto& [queue, q] : qruns)
        if (q.blocked()) m = std::min(m, q.oldest());
    return m;
}
// ---- (from PR 430, YunaraDebugs) Releases while queues are stopped: kept until the stopped work has gone on, instead of making it
// go at once (drain 1). Made to go early, a queue's work ran before the work it waits for, and about every 13th frame in Kokiri a
// pass read what was not there yet: the ground fog thinner for that frame, the light dimmed. The stopped work runs as before, at
// its wait, a present or hold_ms; the release follows it. ----
void do_texture_released(const nvn_tex* t);
void do_program_released(const nvn_prog* p);
std::mutex held_mu;   // (its own: sync_order_mu is held while whole submissions replay, and every release waited for them)
std::vector<nvn_tex> held_tex_releases;    // (held_mu)
std::vector<nvn_prog> held_prog_releases;  // (held_mu)
std::atomic<int64_t> st_held_releases{0};
/// Whether a stopped queue waits on an event whose value is guest memory's (never set by the stream or nvnEventSignal)
bool waits_on_memory_event() {
    for (const auto& [queue, q] : qruns)
        if (q.waiting && !ev_values.count(wait_at(q.p).gpu)) return true;
    return false;
}
void update_blocked() {
    int n = 0;
    for (const auto& [queue, q] : qruns) n += q.blocked();
    ev_wait_in_memory.store(n && waits_on_memory_event(), std::memory_order_release);
    if (n && !blocked_n.load()) blocked_since = std::chrono::steady_clock::now();
    blocked_n = n;
    if (!n) {   // (the stopped work has gone on: its releases now)
        std::vector<nvn_tex> ts;
        std::vector<nvn_prog> ps;
        {
            std::lock_guard hl(held_mu);
            ts.swap(held_tex_releases);
            ps.swap(held_prog_releases);
        }
        for (const nvn_tex& t : ts) do_texture_released(&t);
        for (const nvn_prog& p : ps) do_program_released(&p);
    }
}
void log_order_stats() {
    logf("vk: queues ordered by their events: %lld submissions stopped at a wait, %lld went on once it was reached, %lld "
         "waits gone past; made to go by presents %lld, releases %lld, waits for the replay %lld, %d ms %lld (stopped "
         "%.2f ms at most); waits met %lld, not met %lld; captures at a CPU event's wait %lld, %lld of them at a wait met "
         "on arrival", (long long)st_stops, (long long)st_resumed,
         (long long)st_forced_waits, (long long)st_forced[0], (long long)st_forced[1], (long long)st_forced[2], hold_ms(),
         (long long)st_forced[3], st_stopped_max_ms, (long long)st_waits_met, (long long)st_waits_unmet,
         (long long)st_resume_captures, (long long)st_met_stops);
}

// ---- a burst: the pieces replayed one after the other with no memory work between them (a submit's job: its own
// piece and those of the stopped queues it lets on, then theirs again: MK World's two queues pass five or six stages
// back and forth a frame) go into one recording. Each piece had its own, and its memory sync, write-backs and submit:
// ~8 a frame. The burst's recording is tagged with the newest value among them (what is used in it is kept until
// that one is done; the timeline is signalled at its end as far as allowed, as for any). ----
bool burst_open = false;
uint64_t burst_value = 0;
void burst_begin(uint64_t value) {
    burst_open = !dump_on();
    burst_value = value;
    for (const auto& [queue, q] : qruns) {
        if (q.waiting) burst_value = std::max(burst_value, q.value);
        if (!q.later.empty()) burst_value = std::max(burst_value, q.later.back().value);
    }
}
void finish_recording(uint64_t value);
void rb_span_close();   // (RX_VK_LOG_RB_WRITERS: below, with the readback's)
void burst_end() {
    if (!burst_open) return;
    burst_open = false;
    {
        std::lock_guard lk(replay_mu);
        if (rec_cur) rb_span_close(), finish_recording(burst_value);
    }
    // (the looks taken in this burst count only now: what they let on is recorded, its write-backs are in the page table
    // and its recording is submitted. Said at the look's end, a readback that came before this point found the counts
    // equal, took no look, and read the page table before the burst's recording had noted its pages: found by mooki000)
    if (ev_looked_any) ev_cpu_settled.store(ev_looked, std::memory_order_release), ev_looked_any = false;
}

void rb_span_begin(uint64_t value, uint64_t part);   // (RX_VK_LOG_RB_WRITERS: below, with the readback's)
void rb_span_end(uint64_t value, uint64_t part);
/// The recording being made: ended and submitted, the timeline signalled as far as no older submission is unfinished
void finish_recording(uint64_t value) {
#ifdef TRACY_ENABLE
    HT_SCOPE("finish_recording");
#endif
    (void)value;
    Recording& r = *rec_cur;
    rec_cur = nullptr;
    end_pass(r.cmd);
    flush_ub_updates(r.cmd);
    memory_end_submission(r.cmd, r.part);
    full_barrier(r.cmd, "barrier (recording end)");
    occ_copy_part(r.cmd);
    const uint64_t oldest = oldest_unfinished();
    const uint64_t at = oldest == UINT64_MAX ? done_max : std::min(done_max, oldest - 1);
    const bool signal = at > signaled;
    if (signal) signaled = at;
    submit_recording(r, signal, at);
    rec_close();   // (its last part is on the recordings' timeline now: an early read's writers are all known)
}

// ---- (hookshot) the synchronous replay on the core kept for the replay ----
// With RX_VK_ASYNC off (the default) a submission is replayed on the thread that submits it, or on whichever thread goes on
// with a queue stopped at an event wait, and the core the host keeps free of the game's threads for the replay thread
// (host->reserved_cpus: hos_reserved_cpus, the last core of the fastest kind) had nothing on it. Now a thread runs there
// while it replays and takes a submission's capture, and goes back to its own processors after: every thread that replays,
// none by name; with RX_VK_ASYNC=1 the render thread is there already. Ryzen 9 3900X, Kokiri Forest: see the PR.
// (Only preferring the core, as the thread's ideal processor with every processor allowed, gained
// nothing there: the pin is what helps)
bool async_on();
// (On Linux pthread_create inherits the calling thread's affinity, so win32_posix.c's thread_self and thread_entry
// apply process_affinity when a new thread registers via CreateThread or GetCurrentThread in SetThreadDescription,
// keeping helper threads made on first use inside a replay off the reserved core, as Windows does)
uint64_t replay_pin_mask() {
    static const uint64_t m = [] {
        const uint64_t r = host->reserved_cpus ? host->reserved_cpus() : 0;
        if (r) logf("vk: replays run on logical processors %#llx while they replay", (unsigned long long)r);
        return r;
    }();
    return m;
}
thread_local int replay_pin_depth = 0;
thread_local DWORD_PTR replay_pin_prev = 0;
struct ReplayPin {
    bool counted = false;
    ReplayPin() {
        if (async_on()) return;
        counted = true;
        if (replay_pin_depth++) return;   // (the outermost one: a submit's replay inside its capture's)
        if (const uint64_t m = replay_pin_mask()) {
            replay_pin_prev = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)m);
            static std::atomic<bool> told{false};   // (refused: a mask outside the process's. Said once, not every replay)
            if (!replay_pin_prev && !told.exchange(true, std::memory_order_relaxed))
                logf("vk: a replay was NOT kept to logical processors %#llx (error %lu): replays run where their threads may",
                     (unsigned long long)m, (unsigned long)GetLastError());
        }
    }
    ~ReplayPin() {
        if (!counted || --replay_pin_depth) return;
        if (replay_pin_prev) SetThreadAffinityMask(GetCurrentThread(), replay_pin_prev), replay_pin_prev = 0;
    }
    ReplayPin(const ReplayPin&) = delete;
    ReplayPin& operator=(const ReplayPin&) = delete;
};

/// A submission's command buffers from command buffer cb (op p; null: its start) on, into recordings of its own,
/// submitted: true when it finished; false when it stopped at a wait not met (may_stop), its place kept in qruns.
/// memory_work: its memory work first (else done as it came)
bool run_submit(const uint64_t* handles, int n, uint64_t value, uint64_t queue, bool memory_work, int cb = 0,
                const uint8_t* p = nullptr, bool may_stop = false) {
    tl_ev("run+", value, queue, cb);
    std::lock_guard lk(replay_mu);
    tl_ev("run-locked", value);
    struct TlRun { uint64_t v; ~TlRun() { tl_ev("run-", v); } } tl_run{value};
    ReplayPin pin;
    SlowWatch slow("submit");
    PROF(PROF_SUBMIT);
    struct WallSubmit {   // (RX_VK_PROF_FINE: this submission's wall time and draws, in the frame's counters)
        int64_t t0 = 0, d0 = 0;
        WallSubmit() {
            if (!wall_fine_on()) return;
            int64_t sk, di, pl;
            draw_stats(&d0, &sk, &di, &pl);
            t0 = wall_now();
        }
        ~WallSubmit() {
            if (!t0) return;
            const int64_t dt = wall_now() - t0;
            int64_t d1, sk, di, pl;
            draw_stats(&d1, &sk, &di, &pl);
            rwall.submit_ns += dt, rwall.submits++, rwall.draws += d1 - d0;
            if (d1 - d0 >= 300) rwall.heavy_ns += dt, rwall.heavy_n++, rwall.heavy_draws += d1 - d0;
        }
    } wall_submit;
    if (lag_on()) lag_poll_late(), lag_run_start(value);   // (RX_VK_LAG_LOG)
    if (readback_early_on()) (void)memory_split_wanted(), early_split_pending = early_end_asked = false;   // (the last op's: its part has ended since)
    if (memory_work) {
        if (frame_submits.empty()) frame_first_value = value;
        frame_submits.emplace_back(handles, handles + std::max(n, 0));
        submit_memory_work(value);
    }
    if (burst_open && rec_cur) {   // (the burst's recording goes on: what the piece before did, done first)
        end_pass(rec_cur->cmd);
        full_barrier(rec_cur->cmd, "barrier (a burst's next submit start)");
        use_queue_state(queue);
    } else {
        const uint64_t tag = burst_open ? std::max(burst_value, value) : value;
        use_queue_state(queue);
        ub_prune(completed());
        rec_value_now = tag;
        ub_first_of_recording = ub_updates.size();
        Recording& first = begin_recording(tag);
        memory_begin_submission();
        memory_gpu_writes_begin();   // (RX_VK_UB_SNAPSHOT's guard: a submission's start, not a part's: split_part)
        full_barrier(first.cmd, "barrier (recording start)");
        rec_cur = &first, rec_cur_value = tag, rec_parts = 0, part_work = 0, part_end_wanted = false;
        rec_open(tag);
        if (rb_writers_on()) rb_span_begin(tag, first.part);
        texture_mip_stream_run();
        texture_ahead_run();   // (hookshot, RX_VK_TEX_AHEAD: into this recording's uploads, while the frames are light)
        temporal_prewarm(first.cmd);   // (hookshot, RX_VK_DLSS_PREWARM: DLSS's feature, in a light frame before the first 3D one)
    }
    VkCommandBuffer cmd = rec_cur->cmd;
    const uint8_t* stop = nullptr;
    for (; cb < n; cb++, p = nullptr) {
        if (!handles[cb]) continue;
        {
            WallAdd wall_handle(rwall.handle_ns);
            rwall.handles++;
            stop = replay_handle(cmd, p ? p : reinterpret_cast<const uint8_t*>(handles[cb]), may_stop);
        }
        if (stop) break;
    }
    if (stop) {
        tl_ev("stop", value, queue, wait_at(stop).gpu);
        QueueRun& q = qruns[queue];
        q.waiting = true;
        q.handles.assign(handles, handles + n);
        q.cb = cb, q.p = stop, q.value = value;
        lag_note_stop(value, 1);   // (RX_VK_LAG_LOG)
        if (value > q.ahead_upto) {   // (hookshot: its reports "visible" until they are replayed; once a submission: a later
            occ_visible_ahead(handles, n, cb, stop);   // stop in it holds back less of it)
            q.ahead_upto = value;
        }
        if (st_stops++ < 12) {
            const nvn_p_wait_event w = wait_at(stop);
            logf("vk: submission %llu (queue %#llx) stopped at command buffer %d's wait for event %#llx %s %u (has %u)",
                 (unsigned long long)value, (unsigned long long)queue, cb, (unsigned long long)w.gpu,
                 w.mode == 0 ? "==" : ">=", w.value, event_value_in(ev_values, w.gpu));
        }
        if (st_stops % 6000 == 0) log_order_stats();
    } else {
        done_max = std::max(done_max, value);
        rec_open_values.push_back(value);
        if (queue && rec_timeline) rec_open_qvalues.push_back(value);   // (value_queue has it: submit_q)
    }
    if (!burst_open) {   // (a burst's: at its end)
        WallAdd wall_finish(rwall.finish_ns);
        rb_span_close(), finish_recording(value);   // (RX_VK_LOG_RB_WRITERS: this submission's last part)
    }
    if (stop) return false;
    if (pushed.load() < value) pushed = value;
    st_submits++;
    if (!burst_open && dump_pending()) wait_value(value), dump_write();
    lag_run_end(value, rec_seq, burst_open);   // (RX_VK_LAG_LOG: the recording is whole)
    return true;
}
void do_submit(const uint64_t* handles, int n) {   // (RX_VK_ASYNC=0: on the submitting thread, in its order)
    run_submit(handles, n, ++q_values, 0, true);
}

/// A stopped queue on (forced: past the wait it stopped at), then its later submits, as far as their waits allow
void resume_queue(uint64_t queue, bool forced) {
    QueueRun& q = qruns[queue];
    for (;;) {
        if (q.waiting) {
            // (hookshot) a wait for an event the CPU signals, met: the game wrote what this work reads after it submitted it,
            // up to the signal (OoT's presentation queue waits for its frame's data this way, 1 frame in 12 here). The
            // console's GPU reads memory after its wait; the capture taken at the submit had the frame before's: the fog and
            // exposure of the frame before, for a frame. A capture now, for what follows the wait.
            bool cpu_event = false;
            if (!forced && !async_on()) {   // (the render thread: taken at the signal, event_signal_queued)
                std::lock_guard ek(ev_cpu_mu);
                cpu_event = cpu_events.count(wait_at(q.p).gpu) != 0;
            }
            if (cpu_event) {
                std::lock_guard lk(replay_mu);
                if (rec_cur) rb_span_close(), finish_recording(burst_open ? burst_value : q.value);   // (its mirrors synced again after)
                memory_capture_now(q.value, completed());
                st_resume_captures++;
            }
            q.waiting = false;
            std::vector<uint64_t> handles = std::move(q.handles);
            q.handles.clear();
            const uint8_t* from = forced ? op_next(q.p) : q.p;
            (forced ? st_forced_waits : st_resumed)++;
            met_resumed_at = forced ? nullptr : from;   // (its wait: met, and its capture taken or not wanted)
            const bool went = run_submit(handles.data(), (int)handles.size(), q.value, queue, false, q.cb, from, true);
            met_resumed_at = nullptr;
            if (!went) return;
            continue;
        }
        if (q.later.empty()) return;
        LaterSubmit s = std::move(q.later.front());
        q.later.pop_front();
        if (!run_submit(s.handles.data(), (int)s.handles.size(), s.value, queue, false, 0, nullptr, true)) return;
    }
}
/// (hookshot, the render thread) Whether a queue is stopped at a wait for an event the CPU signals (nvnEventSignal has set
/// it before) that is not met: its frame's data is not all written yet (present held, replay_thread_main)
bool cpu_wait_unmet() {
    for (const auto& [queue, q] : qruns) {
        if (!q.waiting) continue;
        {
            std::lock_guard ek(ev_cpu_mu);
            if (!cpu_events.count(wait_at(q.p).gpu)) continue;
        }
        if (!wait_met_at(q.p)) return true;
    }
    return false;
}
/// The stopped queues whose events are reached go on, until none can
void schedule() {
    // (the CPU's signals made before this look: counted before the first wait is tested, so one made meanwhile, which a
    // test may or may not have seen, is still to be looked at; said once every queue those let on is recorded)
    const uint64_t signals = ev_cpu_signals.load(std::memory_order_acquire);
    for (bool moved = true; moved;) {
        moved = false;
        for (auto& [queue, q] : qruns)
            if (q.waiting && wait_met_at(q.p)) {
                resume_queue(queue, false);
                moved = true;
            }
    }
    // (in a burst the recording of what this look let on is finished at the burst's end, and the count is said there)
    if (burst_open) ev_looked = signals, ev_looked_any = true;
    else ev_cpu_settled.store(signals, std::memory_order_release);
}
/// Every stopped queue on now (why: an index of st_forced): in turn as their waits allow; when all of them wait, the
/// oldest past its wait
void drain(int why) {
    if (oldest_unfinished() == UINT64_MAX) return;
    st_forced[why]++;
    st_stopped_max_ms = std::max(st_stopped_max_ms,
                                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - blocked_since).count());
    burst_begin(0);
    for (;;) {
        schedule();
        uint64_t best = UINT64_MAX, bq = 0;
        for (const auto& [queue, q] : qruns)
            if (q.blocked() && q.oldest() < best) best = q.oldest(), bq = queue;
        if (best == UINT64_MAX) break;
        resume_queue(bq, qruns[bq].waiting);
    }
    burst_end();
    update_blocked();
}

// ---- (hookshot) the same on the submitting thread, the default. do_submit replayed each queue's submission as it came and went
// past every wait: 2 of 3 were not met. A lit pass waiting for another queue's work that had not been submitted yet read what
// that work makes before it was made, and for a frame lit surfaces came out black (a race with the game's threads: 3-9% of
// frames with frame generation off, less the less they overlapped). The stopped work goes on from whichever thread's submit or
// signal lets it, or a present, a wait for it, or hold_ms (sync_watch: nothing else may come while the game polls). ----
bool async_on();
/// RX_VK_PER_QUEUE_FENCES=0: a fence waits for every queue's work up to it, stopped work forced on, as before
bool per_queue_fences_off() {
    static const bool v = getenv("RX_VK_PER_QUEUE_FENCES") && atoi(getenv("RX_VK_PER_QUEUE_FENCES")) == 0;
    return v;
}
// (Hookshot) sync_order_mu counts its sections: `entered` once a holder has it, `left` as it lets it go (both only
// ever changed by the holder, so entered - left is 1 while one is held, else 0). Every recording of the sync-waits replay (and so
// every write-back registration: wb_pages_note, at a binding written in place and at a submission's write-back) is made inside
// such a section: submit_q, present, sync_drain / sync_settle / sync_watch, wait_queue_empty. A fast fence poll keeps the number
// of the section in progress as a ticket (order_ticket_take); a readback waits only until that section has left
// (order_ticket_wait), not for whichever replay holds the lock when it reads. See invalidate for why that is the order it needs.
struct OrderMutex {
    std::mutex mu;
    std::atomic<uint64_t> entered{0}, left{0};
    void lock() {
        mu.lock();
        entered.fetch_add(1, std::memory_order_seq_cst);
    }
    bool try_lock() {
        if (!mu.try_lock()) return false;
        entered.fetch_add(1, std::memory_order_seq_cst);
        return true;
    }
    void unlock();
};
OrderMutex sync_order_mu;   // (before replay_mu, always; all of the order's state: qruns, burst_*)
std::atomic<uint64_t> order_ticket{0};   // (the newest section a fast poll answered during; see order_ticket_take)
std::atomic<int> order_waiters{0};
std::mutex order_wait_mu;
std::condition_variable order_wait_cv;
void OrderMutex::unlock() {
    left.fetch_add(1, std::memory_order_seq_cst);   // (after the section's registrations: release)
    mu.unlock();
    // (a readback waiting for this section: woken. Its waiters++ and this load are both seq_cst, as are this left++ and its
    // check, so either it sees left or this sees it waiting and the notify comes after it is in the wait: no lost wake-up)
    if (order_waiters.load(std::memory_order_seq_cst)) {
        { std::lock_guard lk(order_wait_mu); }
        order_wait_cv.notify_all();
    }
}
/// a fast poll: the section in progress now (0 or the one past the last that left: none to wait for), kept as the
/// newest ticket (a maximum: tickets of polls on other threads only make a readback wait longer, never shorter)
void order_ticket_take() {
    const uint64_t t = sync_order_mu.entered.load(std::memory_order_seq_cst);
    uint64_t cur = order_ticket.load(std::memory_order_relaxed);
    while (cur < t && !order_ticket.compare_exchange_weak(cur, t, std::memory_order_seq_cst, std::memory_order_relaxed)) {}
}
/// until the section of the newest ticket has left (at once when it has: two loads)
void order_ticket_wait() {
    const uint64_t t = order_ticket.load(std::memory_order_seq_cst);
    if (sync_order_mu.left.load(std::memory_order_acquire) >= t) return;
    jt_mark(36, 1);   // (36/37 with 1: the readback waiting for its ticket's section)
    {
        std::unique_lock lk(order_wait_mu);
        order_waiters.fetch_add(1, std::memory_order_seq_cst);
        order_wait_cv.wait(lk, [t] { return sync_order_mu.left.load(std::memory_order_seq_cst) >= t; });
        order_waiters.fetch_sub(1, std::memory_order_seq_cst);
    }
    jt_mark(37, 1);
}
bool sync_waits_on() {
    return !async_on() && !dump_on();
}
void sync_drain(int why) {
#ifdef TRACY_ENABLE
    HT_SCOPE("sync_drain");
#endif
    if (!blocked_n.load()) return;
    std::lock_guard lk(sync_order_mu);
    drain(why);
    update_blocked();
}
/// (hookshot) The stopped queues whose waits are met go on; none is forced past its wait
void sync_settle() {
#ifdef TRACY_ENABLE
    HT_SCOPE("sync_settle");
#endif
    if (!blocked_n.load()) return;
    std::lock_guard lk(sync_order_mu);
    burst_begin(0);
    schedule();
    burst_end();
    update_blocked();
}
/// (hookshot) The same for a CPU read of GPU results (invalidate): only when the CPU has signalled an event since the
/// stopped queues were last looked at, which is when one of them can be stopped at a wait that is met with nobody to let it
/// on (the hole the readback's settle closes: its work not recorded, its write-back unknown). Every other met wait was met by
/// a signal in the stream, replayed under sync_order_mu by a thread that looks at the queues before it lets the mutex go.
/// Without this test the readback took the mutex whenever anything was stopped, and waited for whoever held it: the
/// presenting thread, for the whole replay of its submission and for its present (Hyrule Field: a job thread blocked there
/// at 24 % of all instants, 3.9 ms a frame, for a look that let nothing on). An event the stream and nvnEventSignal never
/// set is the exception: its value is read from guest memory, where a plain store can meet a wait; while a queue is stopped
/// at such a wait the readback looks every time, as before (ev_wait_in_memory). True when it looked.
bool sync_settle_met() {
    if (!blocked_n.load()) return false;
    if (!ev_wait_in_memory.load(std::memory_order_acquire) &&
        ev_cpu_settled.load(std::memory_order_acquire) == ev_cpu_signals.load(std::memory_order_acquire))
        return false;
    sync_settle();
    return true;
}
std::atomic<int> gpu_waiters{0};   // (the game's threads in a wait for the GPU on recorded work of their own queue: GpuWaiter, at wait())
// (how often the stopped queues were held on past hold_ms for such a wait, and how often they were then made to go at the bound)
std::atomic<int64_t> st_hold_gpu{0}, st_hold_gpu_forced{0};
void sync_watch() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk order");
    for (;;) {
        if (watch_kick_on()) WaitForSingleObject(watch_kick_event(), 1);
        else Sleep(1);
        if (!blocked_n.load()) continue;
        std::unique_lock lk(sync_order_mu, std::try_to_lock);
        if (!lk) continue;
        tl_ev("watch+");
        burst_begin(0);   // (a signal from the CPU the thread that made it could not act on)
        schedule();
        burst_end();
        update_blocked();
        tl_ev("watch-");
        // (hookshot) held on while a thread of the game's waits for the GPU on older work, and for hold_ms after: hold_gpu.h
        static const double most_ms = getenv("RX_VK_HOLD_GPU_MS") ? std::max(atof(getenv("RX_VK_HOLD_GPU_MS")), (double)hold_ms()) : 1000.0;
        static bool holding = false;
        static std::chrono::steady_clock::time_point gpu_wait_seen{};
        const auto now = std::chrono::steady_clock::now();
        const bool gpu_wait = gpu_waiters.load(std::memory_order_acquire) > 0;
        if (gpu_wait) gpu_wait_seen = now;
        if (!blocked_n.load()) {
            holding = false;
            continue;
        }
        const auto ms_since = [&](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(now - t).count(); };
        switch (hold_gpu::choose(gpu_wait, ms_since(blocked_since), ms_since(std::max(blocked_since, gpu_wait_seen)), hold_ms(),
                                 most_ms)) {
        case hold_gpu::Choice::Wait: break;
        case hold_gpu::Choice::Hold:
            if (!holding) {
                holding = true;
                const int64_t n = ++st_hold_gpu;
                if (n <= 8 || n % 50 == 0)
                    logf("vk: stopped queues held past %d ms while a thread waits for the GPU: %lld times so far (%lld of them "
                         "made to go at %.0f ms)", hold_ms(), (long long)n, (long long)st_hold_gpu_forced.load(), most_ms);
            }
            break;
        case hold_gpu::Choice::Force:
            if (holding) st_hold_gpu_forced++;
            holding = false;
            drain(3);
            update_blocked();
            break;
        }
    }
}

// doc: docs/design.md#4-the-lessons-from-mkw-as-rules-here (rule 1: cross-queue event waits are honoured)
/// A submit on the replay thread: replayed as far as its waits allow, or (its queue stopped) after its queue's
void submit_job(std::vector<uint64_t>& handles, uint64_t value, uint64_t queue) {
#ifdef TRACY_ENABLE
    HT_SCOPE("submit_job");
#endif
    const bool may_stop = !dump_on();
    if (!may_stop) drain(2);
    auto it = qruns.find(queue);
    if (it != qruns.end() && it->second.blocked()) {   // (its memory work now, in the order of the submits)
        std::lock_guard lk(replay_mu);
        if (frame_submits.empty()) frame_first_value = value;
        frame_submits.emplace_back(handles);
        submit_memory_work(value);
        lag_note_stop(value, 2);   // (RX_VK_LAG_LOG: behind a stopped submission of its queue)
        occ_visible_ahead(handles.data(), (int)handles.size(), 0, nullptr);   // (hookshot: as run_submit's stop)
        it->second.ahead_upto = std::max(it->second.ahead_upto, value);
        it->second.later.push_back({std::move(handles), value});
        return;
    }
    if (may_stop) burst_begin(value);   // (its piece and what it lets on: one recording)
    run_submit(handles.data(), (int)handles.size(), value, queue, true, 0, nullptr, may_stop);
    schedule();   // (what it signalled may let the stopped ones on)
    burst_end();
}
uint64_t get_replayed() {
    queue_flush_poll();
    const uint64_t hold = replayed_hold.load();
    return hold ? std::min(hold, completed()) : completed();
}
/// (hookshot) Waits for the recording submission value f (or the recording of value f) was finished in, on the recordings'
/// timeline: true when that recording is known (the result in *ok), false when it is not (f not finished yet)
static void wait_rec_seq(uint64_t seq, uint64_t timeout_ns, int* ok);
bool wait_recorded(uint64_t f, uint64_t timeout_ns, int* ok) {
#ifdef TRACY_ENABLE
    HT_WAIT(ht_w, "wait_recorded");
#endif
    if (!rec_timeline) return false;
    uint64_t seq = 0;
    {
        std::lock_guard mk(rec_map_mu);
        auto it = value_rec.find(f);
        if (it == value_rec.end()) return false;
        seq = it->second;
    }
    wait_rec_seq(seq, timeout_ns, ok);
    return true;
}
/// (hookshot) The same for one part of a submission (Recording::part, as memory_end_submission noted it with the pages that
/// part writes back): true when that part is submitted (the result in *ok), false when it is not known (older than the map
/// keeps, or never submitted on the recordings' timeline). A part's pages are noted at its end, just before its submit, on
/// the thread that replays it: a read from another thread in between waits for that submit here (250 ms at most, then
/// false: the caller waits as it did before parts were known)
bool wait_recorded_part(uint64_t part, uint64_t timeout_ns, int* ok) {
#ifdef TRACY_ENABLE
    HT_WAIT(ht_w, "wait_recorded_part");
#endif
    if (!rec_timeline || per_queue_fences_off() || !part) return false;
    uint64_t seq = 0;
    {
        std::unique_lock mk(rec_map_mu);
        const auto known = [&] {
            auto it = part_rec.find(part);
            if (it == part_rec.end()) return false;
            seq = it->second;
            return true;
        };
        if (!known()) {
            if (!part_rec.empty() && part_rec.rbegin()->first > part) return false;   // (a newer part is in: this one will not come)
            if (!rec_map_cv.wait_for(mk, std::chrono::milliseconds(250), known)) return false;
        }
    }
    wait_rec_seq(seq, timeout_ns, ok);
    return true;
}
/// (RX_VK_READBACK_EARLY) The recording being made is finished: no more of its writers will be noted (rec_open_tag)
void rb_release_close(uint64_t value);   // (RX_VK_LOG_RB_WRITERS: below, with the readback's)
void rec_close() {
    if (!rec_tags_on()) return;
    if (rb_writers_on()) rb_release_close(rec_open_tag.load(std::memory_order_relaxed));
    rec_open_tag.store(0, std::memory_order_release);
    { std::lock_guard mk(rec_map_mu); }   // (a waiter in early_recorded_wait looks under rec_map_mu: it is asleep or sees 0)
    rec_map_cv.notify_all();
}
/// (RX_VK_READBACK_EARLY) Waits, at most 250 ms, until the submission whose value a read's pages were last written back with
/// (whole) is no longer being recorded; true when it is not. A read let go at its writer's part while the rest of that
/// submission was still being recorded took the newest writer recorded so far, not the range's last: on the Steam Deck in
/// Kokiri Forest, the Market, Kakariko and Hyrule Field, 269 of 745 reads of the culling camera block (+0x3980) were let go at
/// dispatch 7984161db3e053b5's part (or another of the first writers), and dispatch fd958faab99f20a2, recorded after the read,
/// wrote ~1170 of its bytes later in the same submission. Once the recording is finished every write of that submission is
/// noted (memory_in_place_writer_part), the uniform updates of its last part's flush too. *ms: the time waited.
bool early_recorded_wait(uint64_t whole, double* ms) {
#ifdef TRACY_ENABLE
    HT_WAIT(ht_w, "early_recorded_wait");
#endif
    *ms = 0;
    if (rec_open_tag.load(std::memory_order_acquire) != whole) return true;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock mk(rec_map_mu);
    const bool done = rec_map_cv.wait_for(mk, std::chrono::milliseconds(250),
                                          [whole] { return rec_open_tag.load(std::memory_order_acquire) != whole; });
    *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return done;
}
static void wait_rec_seq(uint64_t seq, uint64_t timeout_ns, int* ok) {
    // (hookshot) reached already: nothing to wait for, and nothing to flush for it. The flush below took dev.present_mu,
    // which a present holds for as long as it takes: a readback whose recording was long done waited there for the
    // presenting thread's present, 4-5 ms (Hyrule Field: a job thread at 9 % of all instants; queue_flush_for now looks
    // at pend_held first)
    uint64_t reached = 0;
    if (vkGetSemaphoreCounterValue(dev.device, rec_timeline, &reached) == VK_SUCCESS && reached >= seq) {
        *ok = 1;
        wait_rec_result = VK_SUCCESS;
        return;
    }
    // (a batched submit holds its recordings' signal until the batch goes out: the readback wait in invalidate came here
    // without a flush, and the job that waited held up the main thread, so nothing flushed it: the load hang on RADV)
    queue_flush_for(FR_WAIT);
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &rec_timeline;
    wi.pValues = &seq;
    VkResult res;
    {
        FpsWait fw(fps_gpu_wait_kind());   // (the FPS counter's Breakdown; in the game's fence wait, its kind)
        res = vkWaitSemaphores(dev.device, &wi, timeout_ns);
    }
    *ok = res == VK_SUCCESS;
    wait_rec_result = res;
    const bool lost = res == VK_ERROR_DEVICE_LOST;   // (said as a lost device, not by vk_timeline_wait as a wait that timed out)
    if (lost) wait_lost("a wait for recording", seq);
}
uint64_t fence();
/// (hookshot) A fence of one queue (nvnQueueFenceSync): the value of that queue's last submission
uint64_t fence_q(uint64_t queue) {
    if (async_on() || !sync_waits_on() || per_queue_fences_off() || !queue) return fence();
    std::lock_guard lk(sync_order_mu);
    auto it = q_last.find(queue);
    return it != q_last.end() ? it->second : fence();
}
thread_local int wait_path;   // (how a fence wait ended: 1 recorded whole, 2 stopped work held, 3 the timeline)
int wait_inner(uint64_t f, uint64_t timeout_ns);
uint64_t fence();
/// the game's fence waits that block (nvnSyncWait, nvnQueueFinish): how long, how far ahead of the GPU, by path.
/// RX_VK_FENCE_WAIT_LOG=1: also the first 60 over 5 ms, one by one
int wait_core(uint64_t f, uint64_t timeout_ns) {
    if (completed() >= f) return 1;
    const uint64_t done0 = completed(), latest = fence();
    bool recorded;
    {
        std::lock_guard mk(rec_map_mu);
        recorded = value_rec.count(f) != 0;
    }
    const int blocked0 = (int)blocked_n.load();
    const auto t0 = std::chrono::steady_clock::now();
    wait_path = 0;
    tl_ev("fence-wait+", f, timeout_ns);
    const int r = wait_inner(f, timeout_ns);
    tl_ev("fence-wait-", f, (uint64_t)wait_path);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static std::mutex mu;
    static int64_t n, n_long, by_path[4], unrec, logged;
    static double sum_ms, max_ms, ahead;
    static const bool each = env_on("RX_VK_FENCE_WAIT_LOG");
    std::lock_guard lk(mu);
    if (!timeout_ns) return r;
    n++, sum_ms += ms, max_ms = std::max(max_ms, ms), ahead += (double)(f - done0), by_path[wait_path & 3]++, unrec += !recorded;
    if (ms > 5) n_long++;
    if (each && ms > 5 && logged < 60)
        logged++, logf("vk: fence wait %llu: %.2f ms, path %d, done %llu, latest %llu, %srecorded, %d queues stopped",
                       (unsigned long long)f, ms, wait_path, (unsigned long long)done0, (unsigned long long)latest,
                       recorded ? "" : "not ", blocked0);
    if (n % 300 == 0) {
        logf("vk: fence waits: %lld, %.2f ms each (max %.2f, %lld over 5 ms), %.1f values ahead of the GPU, %lld not recorded yet; "
             "path recorded %lld, stopped %lld, timeline %lld", (long long)n, sum_ms / 300, max_ms, (long long)n_long, ahead / 300,
             (long long)unrec, (long long)by_path[1], (long long)by_path[2], (long long)by_path[3]);
        sum_ms = max_ms = ahead = 0, n_long = unrec = 0, by_path[0] = by_path[1] = by_path[2] = by_path[3] = 0;
    }
    return r;
}
/// (RX_VK_LAG_LOG: each answer the game gets to a fence wait or poll, with the clock and the GPU's state at the answer)
// (hookshot, RX_VK_LOG_READBACK_RANGES: gf_note_value) this thread's last wait for the GPU
thread_local int64_t gf_waited_ns = 0;
int64_t gf_now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void gate_need_value(uint64_t f);   // (below, with the render thread)
// (hookshot) A thread of the game's waiting, with a timeout that is not a poll's, for the GPU to finish recorded work of a
// queue none of whose work up to it is stopped (wait's part wait): counted in gpu_waiters while it waits, for sync_watch's
// hold (hold_gpu.h)
struct GpuWaiter {
    const bool counted;
    explicit GpuWaiter(bool count) : counted(count) {
        if (counted) gpu_waiters.fetch_add(1, std::memory_order_acq_rel);
    }
    ~GpuWaiter() {
        if (counted) gpu_waiters.fetch_sub(1, std::memory_order_acq_rel);
    }
};
int wait(uint64_t f, uint64_t timeout_ns) {
    gf_waited_ns = gf_now();   // (RX_VK_LOG_READBACK_RANGES: this thread waited for the GPU, or asked whether it was done)
    if (completed() >= f) return 1;
    // (the FPS counter's Breakdown: this wait's kind, by whether its work was recorded when it began: the renderer behind, or
    // the GPU)
    bool recorded = true;
    if (timeout_ns && fps_breakdown_on()) {
        std::lock_guard mk(rec_map_mu);
        recorded = value_rec.count(f) != 0;
    }
    FpsFenceWait fence_kind(recorded);
    if (timeout_ns) gate_need_value(f);   // (the render thread's whole frames: this one's submit goes on without its present)
    FpsRender fr;   // (the FPS counter's Breakdown: the replay of stopped work in here is the renderer's, the waits the GPU's)
    queue_flush_for(FR_WAIT);
    // (hookshot) a fence of one queue whose work up to it is recorded whole (none of it stopped): that work, not the other
    // queues' stopped work the timeline above waits for (which a wait for it forced past its waits after RX_VK_WAIT_HOLD_US)
    if (!async_on() && sync_waits_on() && !per_queue_fences_off()) {
        bool mine = false;
        {
            std::lock_guard lk(sync_order_mu);
            auto it = value_queue.find(f);
            if (it != value_queue.end()) {
                auto q = qruns.find(it->second);
                mine = q == qruns.end() || !q->second.blocked() || q->second.oldest() > f;
            }
        }
        int ok = 0;
        if (mine) {
            GpuWaiter waiting(timeout_ns > 1000000);   // (a wait for the GPU, not a poll: sync_watch holds the stopped queues meanwhile)
            if (wait_recorded(f, timeout_ns, &ok)) return ok;
        }
    }
    // (a wait that needs stopped work: another queue's submit usually lets that on in a moment. Forced at once, as the main thread
    // waits ~15 times a second, 69000 waits went past in five minutes, and each one that mattered was a black frame.
    // RX_VK_WAIT_HOLD_US: how long the waiter gives it. A wait for older work than the stopped ones needs nothing of them)
    if (timeout_ns && !async_on() && blocked_n.load()) {
        static const int64_t hold_ns = (getenv("RX_VK_WAIT_HOLD_US") ? atoll(getenv("RX_VK_WAIT_HOLD_US")) : 3000) * 1000;
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            queue_flush_for(FR_WAIT);
            if (completed() >= f) return 1;
            bool needs;
            {
                std::lock_guard lk(sync_order_mu);
                needs = oldest_unfinished() <= f;
            }
            if (!needs) break;
            const int64_t waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            if ((uint64_t)waited >= timeout_ns) break;   // (sync_watch's, past hold_ms)
            if (waited >= hold_ns) {
                sync_drain(2);
                break;
            }
            std::this_thread::yield();
        }
    }
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &f;
    queue_flush_for(FR_WAIT);
    VkResult res;
    {
        FpsWait fw(fps_gpu_wait_kind());
        res = vkWaitSemaphores(dev.device, &wi, timeout_ns);
    }
    static int logged;
    if (res != VK_SUCCESS && timeout_ns > 1000000000ull && logged++ < 8)
        logf("vk: wait for fence %llu (done %llu, pushed %llu): %d", (unsigned long long)f,
             (unsigned long long)completed(), (unsigned long long)pushed.load(), (int)res);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    return res == VK_SUCCESS;
}
int wait_inner(uint64_t f, uint64_t timeout_ns) {
    gf_waited_ns = gf_now();   // (RX_VK_LOG_READBACK_RANGES: this thread waited for the GPU, or asked whether it was done)
    if (completed() >= f) return 1;
    struct PollFlag {   // (a poll does not wait for a pending late fix in its flushes)
        bool prev;
        explicit PollFlag(bool poll) : prev(tl_poll) { tl_poll = poll; }
        ~PollFlag() { tl_poll = prev; }
    } poll_flag(timeout_ns == 0);
    // (the FPS counter's Breakdown: this wait's kind, by whether its work was recorded when it began: the renderer behind, or
    // the GPU)
    bool recorded = true;
    if (timeout_ns && fps_breakdown_on()) {
        std::lock_guard mk(rec_map_mu);
        recorded = value_rec.count(f) != 0;
    }
    FpsFenceWait fence_kind(recorded);
    if (timeout_ns) gate_need_value(f);   // (the render thread's whole frames: this one's submit goes on without its present)
    FpsRender fr;   // (the FPS counter's Breakdown: the replay of stopped work in here is the renderer's, the waits the GPU's)
    queue_flush_for(FR_WAIT);
    // (Hookshot) RX_VK_FENCE_POLL_FAST=1: a poll (timeout 0) of a fence whose submission was finished whole on its
    // queue skips sync_order_mu below, held by submit_q / present for whole replays (1.4 ms a frame on the main thread). The
    // answer is the locked path's: (1) "mine" is true for such a value: a queue's submissions run in order, so one finished
    // whole has no unfinished one of its queue at or below it (qruns' oldest> f or not blocked), and value_queue has it (it
    // was submitted on a queue, and fewer than 8000 values came since: value_queue keeps the last 8192); being finished never
    // reverts, so the lock would only have made the same decision later; (2) then wait_recorded(f) as there, under rec_map_mu,
    // which the mark is set under (after value_rec has f). Only a poll: a wait with a timeout keeps the locked path (with
    // batched submits, a wait right after the mark could miss the flush of its recording: RADV's load hang). Any other case
    // (not marked, an older value, a slot reused by a newer one) takes the locked path. The lock also ordered the poll after the
    // replay in progress, which the CPU's readbacks of GPU results need (their write-back registrations): invalidate waits for it
    // there instead, with this switch on (without that wait the Market's exposure blew the scene out to white).
    if (!timeout_ns && fence_poll_fast_on() && !async_on() && sync_waits_on() && !per_queue_fences_off() &&
        whole_val[f & (kWholeSlots - 1)].load(std::memory_order_acquire) == f && f + 8000 > q_values.load()) {
        int ok = 0;
        wait_path = 1;
        if (readback_ticket()) order_ticket_take();   // (the section the lock would have waited for; see invalidate)
        if (wait_recorded(f, 0, &ok)) return ok;
    }
    // (hookshot) a fence of one queue whose work up to it is recorded whole (none of it stopped): that work, not the other
    // queues' stopped work the timeline above waits for (which a wait for it forced past its waits after RX_VK_WAIT_HOLD_US)
    if (!async_on() && sync_waits_on() && !per_queue_fences_off()) {
        bool mine = false;
        {
            jt_mark(32, 0);   // (32/33 the wait taking sync_order_mu)
            std::lock_guard lk(sync_order_mu);
            jt_mark(33, 0);
            auto it = value_queue.find(f);
            if (it != value_queue.end()) {
                auto q = qruns.find(it->second);
                mine = q == qruns.end() || !q->second.blocked() || q->second.oldest() > f;
            }
        }
        int ok = 0;
        wait_path = 1;
        if (mine) {
            GpuWaiter waiting(timeout_ns > 1000000);   // (a wait for the GPU, not a poll: sync_watch holds the stopped queues meanwhile)
            if (wait_recorded(f, timeout_ns, &ok)) return ok;
        }
    }
    // (a wait that needs stopped work: another queue's submit usually lets that on in a moment. Forced at once, as the main thread
    // waits ~15 times a second, 69000 waits went past in five minutes, and each one that mattered was a black frame.
    // RX_VK_WAIT_HOLD_US: how long the waiter gives it. A wait for older work than the stopped ones needs nothing of them)
    wait_path = 3;
    if (timeout_ns && !async_on() && blocked_n.load()) {
        wait_path = 2;
        static const int64_t hold_ns = (getenv("RX_VK_WAIT_HOLD_US") ? atoll(getenv("RX_VK_WAIT_HOLD_US")) : 3000) * 1000;
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            queue_flush_for(FR_WAIT);
            if (completed() >= f) return 1;
            bool needs;
            {
                std::lock_guard lk(sync_order_mu);
                needs = oldest_unfinished() <= f;
            }
            if (!needs) break;
            const int64_t waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            if ((uint64_t)waited >= timeout_ns) break;   // (sync_watch's, past hold_ms)
            if (waited >= hold_ns) {
                sync_drain(2);
                break;
            }
            std::this_thread::yield();
        }
    }
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &f;
    queue_flush_for(FR_WAIT);
    VkResult res;
    {
        FpsWait fw(fps_gpu_wait_kind());
        res = vkWaitSemaphores(dev.device, &wi, timeout_ns);
    }
    if (res == VK_SUCCESS) (void)completed();   // (RX_VK_POOL_MIRROR: its staged write-backs applied)
    static int logged;
    if (res != VK_SUCCESS && timeout_ns > 1000000000ull && logged++ < 8)
        logf("vk: wait for fence %llu (done %llu, pushed %llu): %d", (unsigned long long)f,
             (unsigned long long)completed(), (unsigned long long)pushed.load(), (int)res);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    return res == VK_SUCCESS;
}
void wait_queue_empty();
bool replay_behind();
void finish() {
    gf_waited_ns = gf_now();
    FpsWait fw(fps_stats::kGpuWait);   // (the FPS counter's Breakdown)
    wait_queue_empty();
    wait_value(pushed.load());
}
/// (the game's nvnQueueFinish; RX_VK_LAG_LOG: what the GPU and the order's stopped queues have left when it returns)
void finish_game() {
    const int64_t t0 = lag_on() ? host->now_ns() : 0;
    finish();
    if (lag_on()) lag_finish(t0, host->now_ns(), q_values.load(), completed(), pushed.load(), blocked_n.load());
}
void wait_writebacks_reading(const void*, uint64_t) {}   // (nothing to write back: the GPU writes guest memory)
/// (host: the CPU is about to read [gpu, gpu + size), nvnMemoryPoolInvalidateMappedRange / nvnBufferInvalidateMappedRange)
/// RX_VK_LOG_INVALIDATE=1: the ranges, 400 lines, and the textures the GPU wrote there (their texels are not in memory)
// ---- live textures' storage, by 4 KB page (as the host made and finalized them, ahead of the replay) ----
// For invalidate: the CPU reads a range back; texels the GPU wrote into a texture there are not in memory until a
// write-back, and a texture the replay has not listed yet (it runs behind) needs it to catch up first. Ranges no live
// texture's storage touches hold buffers only, which the submissions write back themselves: no catch-up for them.
// (Counts per page: storage shared by several textures; two levels, chunks of 1M pages made when first touched.)
constexpr uint64_t kTexPageChunk = 1u << 20;
std::unique_ptr<std::atomic<uint16_t>[]> tex_pages[(1ull << 39 >> 12) / kTexPageChunk];
std::mutex tex_pages_mu;
// (hookshot, the render thread) and by byte: a readback next to a texture's storage in its page, not in it, needs no catch-up
// (the Market: 2.4 such readbacks a frame, each held 3.3-11 ms for the jobs handed over). tex_live_mu
std::mutex tex_live_mu;
std::map<uint64_t, std::pair<uint64_t, int>> tex_live;   // (storage start -> its largest size, how many live textures start there)
uint64_t tex_live_max = 0;
void tex_live_add(const nvn_tex* t, int d) {
    std::lock_guard lk(tex_live_mu);
    if (d > 0) {
        auto& e = tex_live[t->gpu];
        e.first = std::max<uint64_t>(e.first, t->L.size), e.second++;
        tex_live_max = std::max<uint64_t>(tex_live_max, t->L.size);
    } else if (auto it = tex_live.find(t->gpu); it != tex_live.end() && --it->second.second <= 0) {
        tex_live.erase(it);
    }
}
bool tex_live_any(uint64_t gpu, uint64_t size) {
    std::lock_guard lk(tex_live_mu);
    for (auto it = tex_live.lower_bound(gpu > tex_live_max ? gpu - tex_live_max : 0); it != tex_live.end() && it->first < gpu + size; ++it)
        if (it->first + it->second.first > gpu) return true;
    return false;
}
void tex_pages_add(const nvn_tex* t, int d) {
    if (!t || !t->gpu || t->gpu >= (1ull << 39) || !t->L.size) return;   // (the guest window: virtual pools are not read)
    if (async_on()) tex_live_add(t, d);   // (the render thread's: invalidate)
    const uint64_t p0 = t->gpu >> 12, p1 = std::min((t->gpu + t->L.size - 1) >> 12, uint64_t((1ull << 39 >> 12) - 1));
    {   // (RX_VK_TEX_PAGE_WATCH=<hex address>: the textures whose storage has that page)
        static const uint64_t watch = getenv("RX_VK_TEX_PAGE_WATCH") ? strtoull(getenv("RX_VK_TEX_PAGE_WATCH"), nullptr, 16) >> 12 : 0;
        static std::atomic<int> logged{0};
        if (watch && watch >= p0 && watch <= p1 && logged++ < 60)
            logf("vk: tex page watch: %+d texture %#llx +%#llx target %d format %d flags %#x %ux%ux%u levels %u serial %llu", d,
                 (unsigned long long)t->gpu, (unsigned long long)t->L.size, t->d.target, t->d.format, (unsigned)t->d.flags, t->d.width,
                 t->d.height, t->d.depth, t->d.levels, (unsigned long long)t->serial);
    }
    for (uint64_t p = p0; p <= p1; p++) {
        auto& c = tex_pages[p / kTexPageChunk];
        if (!c) {
            std::lock_guard lk(tex_pages_mu);
            if (!c) {
                auto n = std::make_unique<std::atomic<uint16_t>[]>(kTexPageChunk);
                for (uint64_t i = 0; i < kTexPageChunk; i++) n[i].store(0, std::memory_order_relaxed);
                c = std::move(n);
            }
        }
        std::atomic<uint16_t>& v = c[p % kTexPageChunk];
        if (d > 0) v.fetch_add(1, std::memory_order_relaxed);
        else if (v.load(std::memory_order_relaxed)) v.fetch_sub(1, std::memory_order_relaxed);
    }
}
bool tex_pages_any(uint64_t gpu, uint64_t size) {
    if (gpu >= (1ull << 39)) return true;   // (not tracked: as before)
    const uint64_t p0 = gpu >> 12, p1 = std::min((gpu + size - 1) >> 12, uint64_t((1ull << 39 >> 12) - 1));
    for (uint64_t p = p0; p <= p1; p++) {
        const auto& c = tex_pages[p / kTexPageChunk];
        if (!c) { p = (p / kTexPageChunk + 1) * kTexPageChunk - 1; continue; }
        if (c[p % kTexPageChunk].load(std::memory_order_relaxed)) return true;
    }
    return false;
}
void texture_created_of(const nvn_tex* t) {
    tex_pages_add(t, +1);
    texture_ahead_made(t);   // (hookshot, RX_VK_TEX_AHEAD: it may be made and uploaded before its first draw)
}
uint64_t fence();
struct RbScope {
    bool on;
    uint64_t gpu, size;
    int64_t t0 = 0, ph[6] = {0, 0, 0, 0, 0, 0}, pt = 0;
    bool ever = false, pending = false, newest = false, late = false;
    uint64_t value = 0, fence_at = 0;
    char site[80] = "";   // (the exe's frames above the renderer: the guest function that asked, by its address in the exe's map)
    explicit RbScope(bool o, uint64_t g, uint64_t s) : on(o), gpu(g), size(s) {
        if (!on) return;
        void* bt[14];
        const USHORT n = CaptureStackBackTrace(1, 14, bt, nullptr);
        static const uintptr_t base = exe_base();
        int w = 0, seen = 0;
        for (USHORT i = 0; i < n && seen < 5; i++) {
            const uintptr_t a = (uintptr_t)bt[i];
            if (a < base || a >= base + 0x30000000) continue;
            if (seen++ >= 2) w += snprintf(site + w, sizeof site - w, "%s%x", w ? "," : "", (unsigned)(a - base));
        }
        rb_wr_ns[0] = rb_wr_ns[1] = 0, rb_wr_miss = 0, rb_fl_ns[0] = rb_fl_ns[1] = rb_fl_ns[2] = 0;
        t0 = pt = now_ns();
    }
    static int64_t now_ns() { return std::chrono::steady_clock::now().time_since_epoch().count(); }
    void mark(int p) { if (!on) return; const int64_t t = now_ns(); ph[p] += t - pt; pt = t; }
    void skip() { if (on) pt = now_ns(); }   // (time not charged to a phase)
    ~RbScope();
};
struct RbAgg {
    int64_t calls = 0, ever = 0, pending = 0, newest = 0, late = 0, ns_total = 0, ph[6] = {0, 0, 0, 0, 0, 0}, ns_max = 0, wr[2] = {0, 0}, miss = 0, fl[3] = {0, 0, 0};
    void add(const RbScope& s, int64_t total) {
        wr[0] += rb_wr_ns[0], wr[1] += rb_wr_ns[1], miss += rb_wr_miss, fl[0] += rb_fl_ns[0], fl[1] += rb_fl_ns[1], fl[2] += rb_fl_ns[2];
        calls++, ever += s.ever, pending += s.pending, newest += s.newest, late += s.late, ns_total += total;
        for (int i = 0; i < 6; i++) ph[i] += s.ph[i];
        ns_max = std::max(ns_max, total);
    }
};
std::mutex rb_prof_mu;
std::map<std::pair<uint64_t, uint64_t>, RbAgg> rb_prof_ranges;
std::map<std::string, RbAgg> rb_prof_threads, rb_prof_sites;
RbAgg rb_prof_all;
int64_t rb_prof_next = 300, rb_prof_from = 0;
RbScope::~RbScope() {
    if (!on) return;
    const int64_t total = now_ns() - t0;
    if (tl_active.load(std::memory_order_relaxed)) {
        if (pending) tl_ev("rb-val", value, fence_at, completed());
        tl_ev("rb-end", gpu, size, (uint64_t)(total / 1000) | (uint64_t)(ph[2] / 1000) << 24 | (uint64_t)(ph[3] / 1000) << 40 | (uint64_t)late << 63);
    }
    static const bool aggregate = env_on("RX_VK_RB_PROF");
    if (!aggregate) return;
    std::lock_guard lk(rb_prof_mu);
    if (!rb_prof_from) rb_prof_from = presents_now, rb_prof_next = presents_now + 300;   // (the first window starts here)
    rb_prof_all.add(*this, total);
    rb_prof_threads[rb_thread_label()].add(*this, total);
    if (rb_prof_sites.size() < 256 || rb_prof_sites.count(site)) {
        char k[120];
        snprintf(k, sizeof k, "%s %#llx+%#llx", site, (unsigned long long)gpu, (unsigned long long)size);
        rb_prof_sites[k].add(*this, total);
    }
    if (rb_prof_ranges.size() < 512 || rb_prof_ranges.count({gpu, size})) rb_prof_ranges[{gpu, size}].add(*this, total);
    if (presents_now < rb_prof_next) return;
    const int64_t frames = std::max<int64_t>(1, presents_now - rb_prof_from);
    static int logged = 0;
    if (logged++ < 200) {
        auto line = [&](const char* what, const RbAgg& a) {
            logf("vk: rbprof %lld presents %s: %.1f calls/frame (%.1f with write-backs, %.1f pending, %.1f newest, %.1f late), %.2f ms/frame "
                 "[ticket %.2f, catchup %.2f, recorded %.2f (flush %.2f = late fix %.2f + lock %.2f + sending %.2f, timeline %.2f, unknown %.1f/frame), gpu %.2f, strict %.2f, textures %.2f], max %.2f ms",
                 (long long)frames, what, a.calls / (double)frames, a.ever / (double)frames, a.pending / (double)frames,
                 a.newest / (double)frames, a.late / (double)frames, a.ns_total / 1e6 / frames, a.ph[0] / 1e6 / frames, a.ph[1] / 1e6 / frames,
                 a.ph[2] / 1e6 / frames, a.wr[0] / 1e6 / frames, a.fl[0] / 1e6 / frames, a.fl[1] / 1e6 / frames, a.fl[2] / 1e6 / frames, a.wr[1] / 1e6 / frames, a.miss / (double)frames, a.ph[3] / 1e6 / frames,
                 a.ph[4] / 1e6 / frames, a.ph[5] / 1e6 / frames, a.ns_max / 1e6);
        };
        line("all", rb_prof_all);
        for (const auto& t : rb_prof_threads) line(t.first.c_str(), t.second);
        {
            std::vector<std::pair<std::string, RbAgg>> sv(rb_prof_sites.begin(), rb_prof_sites.end());
            std::sort(sv.begin(), sv.end(), [](const auto& a, const auto& b) { return a.second.ns_total > b.second.ns_total; });
            for (size_t i = 0; i < sv.size() && i < 10; i++) line(("site " + sv[i].first).c_str(), sv[i].second);
        }
        std::vector<std::pair<std::pair<uint64_t, uint64_t>, RbAgg>> v(rb_prof_ranges.begin(), rb_prof_ranges.end());
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.ns_total > b.second.ns_total; });
        for (size_t i = 0; i < v.size() && i < 12; i++) {
            char nm[64];
            snprintf(nm, sizeof nm, "range %#llx +%#llx", (unsigned long long)v[i].first.first, (unsigned long long)v[i].first.second);
            line(nm, v[i].second);
        }
    }
    rb_prof_all = RbAgg();
    rb_prof_threads.clear();
    rb_prof_sites.clear();
    rb_prof_ranges.clear();
    rb_prof_from = presents_now;
    rb_prof_next = presents_now + 300;
}
// ---- (hookshot, diagnostics) RX_VK_LOG_RB_WRITERS=1 (with RX_VK_LOG_READBACK_WAITS=1 for a line a wait) ----
// What wrote the pages a readback waits for (memory_rb_writers: the op, the program, the part, the draw), where that is in its
// submission, and how long the read would have waited for the part its last writer was recorded in rather than for its whole
// submission: the readback waits for that part first, timed, then as before (the same wait in two steps). Every 3000
// readbacks, by the size read: the waits, the part's share of them, and the writers
struct SubSpan { uint64_t first_part = 0, last_part = 0, w0 = 0, w1 = 0; };
std::mutex rb_span_mu;
std::map<uint64_t, SubSpan> rb_spans;   // (submission value -> its first and last parts, its draws' and dispatches' count then)
void rb_span_begin(uint64_t value, uint64_t part) {
    std::lock_guard lk(rb_span_mu);
    SubSpan& s = rb_spans[value];
    if (!s.first_part) s.first_part = part, s.w0 = memory_work_count();
    while (rb_spans.size() > 4096) rb_spans.erase(rb_spans.begin());
}
void rb_span_end(uint64_t value, uint64_t part) {
    std::lock_guard lk(rb_span_mu);
    SubSpan& s = rb_spans[value];
    if (!s.first_part) s.first_part = part, s.w0 = memory_work_count();
    s.last_part = std::max(s.last_part, part), s.w1 = memory_work_count();
}
/// The recording about to be finished closes its submission's span: on every path that finishes one (run_submit's, a burst's
/// end, a resumed queue's capture). Closed only outside a burst before, a burst's span kept no last part or total work: the
/// log took the last writer's count for the total and printed a last part of 0
void rb_span_close() {
    if (rb_writers_on() && rec_cur) rb_span_end(rec_cur->value, rec_cur->part);
}
const char* rb_what_name(uint32_t what) {
    switch (what) {
    case kRbWriterUb: return "uniform-updates";
    case kRbWriterWb: return "write-back";
    case NOP_SIGNAL_EVENT: return "event";
    case NOP_COPY_B2B: return "copy";
    case NOP_CLEAR_BUF: return "clear";
    case NOP_COPY_T2B: return "texture->buffer";
    case NOP_DRAW: case NOP_DRAW_IDX: case NOP_DRAW_INDIRECT: case NOP_DRAW_MESH: return "draw";
    case NOP_DISPATCH: case NOP_DISPATCH_INDIRECT: return "dispatch";
    default: return "other";
    }
}
struct RbDiag {
    bool on = false;
    RbWriter first, last;
    int pages = 0;
    SubSpan span;
    double part_ms = -1;   // (when the last writer's part was done, into the wait; -1: not waited for apart)
    // (the bytes read, up to 1 MB: before the wait, once the last writer's part was done, after the whole wait. What changed
    // after the part says whether waiting for the part alone would have given the CPU the same bytes)
    bool snap = false;
    int64_t changed_before_part = 0, changed_after_part = 0;
};
thread_local std::vector<uint8_t> rb_snap[3];
uint64_t rb_diag_early_part(uint64_t early, uint64_t part, bool allowed) {
    return allowed && early ? std::max(early, part) : 0;
}
void rb_diag_wait(uint64_t gpu, uint64_t size, RbDiag& d, uint64_t whole, uint64_t part, bool early_allowed) {
    d.on = memory_rb_writers(host->gpu_guest(gpu), size, d.first, d.last, &d.pages);
    if (!d.on) return;
    // with RX_VK_READBACK_EARLY on, the part the early wait lets the read go at (its own writers' record, which holds
    // the writes over 1 MiB and the bare and wide bindings the diagnostics' record has not), or no check at all where the early
    // wait would take the whole submission: "bytes changed after the part" is then what an early read would have missed. (Any
    // CPU write to the range in the wait is counted too: the game's other threads run)
    if (readback_early_on() && whole) {
        double rec_ms = 0;   // (as invalidate: its writer's submission recorded to its end first, then its last writer's part)
        const bool recorded = !early_allowed || early_recorded_wait(whole, &rec_ms);
        const uint64_t early = recorded ? memory_in_place_writer_part(host->gpu_guest(gpu), size, whole) : 0;
        d.last.part = rb_diag_early_part(early, part, early_allowed);   // (exactly invalidate's target, or 0 for the whole)
    }
    if (size <= (1u << 20)) {
        rb_snap[0].resize(size);
        d.snap = host->gpu_read(gpu, rb_snap[0].data(), size);
    }
    {
        std::lock_guard lk(rb_span_mu);
        if (auto it = rb_spans.find(d.last.value); it != rb_spans.end()) d.span = it->second;
    }
    const auto t0 = std::chrono::steady_clock::now();
    int ok = 0;
    if (d.last.part && wait_recorded_part(d.last.part, UINT64_MAX, &ok))
        d.part_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (d.snap && d.part_ms >= 0) {
        rb_snap[1].resize(size);
        d.snap = host->gpu_read(gpu, rb_snap[1].data(), size);
    } else {
        d.snap = false;   // (no part waited for apart: nothing to tell)
    }
}
int64_t rb_bytes_differ(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    int64_t n = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) n += a[i] != b[i];
    return n;
}
struct RbWriterAgg {
    int64_t n = 0, parted = 0;
    double ms = 0, part_ms = 0, whole_ms = 0;   // (part_ms, whole_ms: the waits whose part was waited for apart)
    int64_t snaps = 0, changed = 0, changed_after = 0, after_waits = 0;   // (bytes changed in the wait, after the part; waits with any after)
    std::map<std::pair<uint32_t, uint64_t>, int> writers;
};
std::mutex rb_agg_mu;
std::map<uint64_t, RbWriterAgg> rb_agg;   // (by the size read)
void rb_diag_done(uint64_t gpu, uint64_t size, RbDiag& d, double ms, bool log_line) {
    if (!d.on) return;
    if (d.snap) {
        rb_snap[2].resize(size);
        if ((d.snap = host->gpu_read(gpu, rb_snap[2].data(), size))) {
            d.changed_before_part = rb_bytes_differ(rb_snap[0], rb_snap[1]);
            d.changed_after_part = rb_bytes_differ(rb_snap[1], rb_snap[2]);
        }
    }
    {
        std::lock_guard lk(rb_agg_mu);
        RbWriterAgg& a = rb_agg[size];
        a.n++, a.ms += ms;
        if (d.part_ms >= 0) a.parted++, a.part_ms += d.part_ms, a.whole_ms += ms;
        a.writers[{d.last.what, d.last.key}]++;
        if (d.snap) {
            a.snaps++, a.changed += d.changed_before_part + d.changed_after_part, a.changed_after += d.changed_after_part;
            a.after_waits += d.changed_after_part != 0;
        }
    }
    if (!log_line) return;
    const uint64_t w0 = d.span.w0, w1 = std::max(d.span.w1, d.last.work);
    logf("vk: readback writers %#llx +%#llx: last %s %016llx (part %llu, draw %llu of %llu in submission %llu), first %s %016llx "
         "(part %llu, draw %llu), %d pages; the submission's parts %llu..%llu; its last writer's part done %.2f ms into the wait "
         "of %.2f; bytes changed %lld by then, %lld after%s", (unsigned long long)gpu, (unsigned long long)size, rb_what_name(d.last.what), (unsigned long long)d.last.key,
         (unsigned long long)d.last.part, (unsigned long long)(d.last.work >= w0 ? d.last.work - w0 : 0),
         (unsigned long long)(w1 - w0), (unsigned long long)d.last.value, rb_what_name(d.first.what),
         (unsigned long long)d.first.key, (unsigned long long)d.first.part,
         (unsigned long long)(d.first.work >= w0 ? d.first.work - w0 : 0), d.pages, (unsigned long long)d.span.first_part,
         (unsigned long long)d.span.last_part, d.part_ms, ms, (long long)d.changed_before_part, (long long)d.changed_after_part,
         d.snap ? "" : " (not read)");
}
void rb_diag_report() {
    std::lock_guard lk(rb_agg_mu);
    for (auto& [size, a] : rb_agg) {
        if (a.n < 3) continue;
        std::vector<std::pair<int, std::pair<uint32_t, uint64_t>>> top;
        for (const auto& [k, c] : a.writers) top.push_back({c, k});
        std::sort(top.rbegin(), top.rend());
        std::string who;
        for (size_t i = 0; i < top.size() && i < 4; i++) {
            char one[96];
            snprintf(one, sizeof one, "%s%s %016llx x%d", i ? ", " : "", rb_what_name(top[i].second.first),
                     (unsigned long long)top[i].second.second, top[i].first);
            who += one;
        }
        logf("vk: readback writers, size %#llx: %lld waits, %.2f ms each; %lld with the last writer's part known: done %.2f ms into "
             "their %.2f ms (%.2f ms of each wait after it); bytes the wait changed: %.1f each, %.1f of them after the part (%lld of "
             "%lld waits); last writers: %s", (unsigned long long)size, (long long)a.n, a.ms / a.n,
             (long long)a.parted, a.parted ? a.part_ms / a.parted : 0.0, a.parted ? a.whole_ms / a.parted : 0.0,
             a.parted ? (a.whole_ms - a.part_ms) / a.parted : 0.0, a.snaps ? (double)a.changed / a.snaps : 0.0,
             a.snaps ? (double)a.changed_after / a.snaps : 0.0, (long long)a.after_waits, (long long)a.snaps, who.c_str());
    }
    rb_agg.clear();
}
// ---- (diagnostics) RX_VK_LOG_RB_WRITERS: what a read was let go with, at the game's own timing ----
// Each read whose range's newest writer's submission was not done keeps that writer's draw count and part as the diagnostics'
// record had them (memory_rb_writers), and whether that submission was still being recorded; once its recording is finished
// (rec_close, after its last part's uniform updates) the record says whether a writer of the range recorded after the read
// followed in the same submission: bytes the read was let go without, the game's to wait for. And whether the same thread read
// the range again for that submission, as a poll does. Every 3000 readbacks, by the size read. No wait and no copy here: with
// RX_VK_LOG_RB_WRITERS=2 this is all it does, and the replay's and the game's timing are their own (=1 adds the timed waits
// and the snapshots above). Not seen: the writes the record does not hold (bindings over 1 MiB, bare and wide bindings)
struct RbRelease {
    uint64_t gpu = 0, size = 0, value = 0, part = 0, work = 0;
    uint32_t tid = 0;
    bool open = false, early = false, closed = false, later = false, again = false, seen = false;
};
std::mutex rb_rel_mu;
std::vector<RbRelease> rb_rels;   // (rb_rel_mu)
bool rb_probe_only() {
    static const bool v = getenv("RX_VK_LOG_RB_WRITERS") && atoi(getenv("RX_VK_LOG_RB_WRITERS")) == 2;
    return v;
}
uint32_t rb_tid() {
    static std::atomic<uint32_t> next{0};
    thread_local const uint32_t id = ++next;
    return id;
}
/// A read of [gpu, gpu + size) whose newest write-back is value, let go now; pending: that submission was not done
void rb_release_note(uint64_t gpu, uint64_t size, uint64_t value, bool pending, bool early) {
    if (!value) return;
    const uint32_t tid = rb_tid();
    RbWriter first, last;
    int pages = 0;
    const bool known = pending && memory_rb_writers(host->gpu_guest(gpu), size, first, last, &pages) && last.value == value;
    const bool open = known && rec_open_tag.load(std::memory_order_acquire) == value;
    std::lock_guard lk(rb_rel_mu);
    for (RbRelease& r : rb_rels)
        if (r.gpu == gpu && r.size == size && r.value == value && r.tid == tid) r.again = true;
    if (!known) return;
    if (rb_rels.size() >= 4096) rb_rels.erase(rb_rels.begin());   // (bounded: a diagnostic's memory)
    RbRelease r;
    r.gpu = gpu, r.size = size, r.value = value, r.part = last.part, r.work = last.work, r.tid = tid, r.open = open, r.early = early;
    r.closed = !open;   // (its recording finished already: every writer of the range in it was in the record at the read)
    rb_rels.push_back(r);
}
/// The recording of value finished (rec_close): the reads let go before it, against the range's last writer in it
void rb_release_close(uint64_t value) {
    std::lock_guard lk(rb_rel_mu);
    for (RbRelease& r : rb_rels) {
        if (r.closed || r.value != value) continue;
        r.closed = true;
        RbWriter first, last;
        int pages = 0;
        if (memory_rb_writers(host->gpu_guest(r.gpu), r.size, first, last, &pages) && last.value == value)
            r.later = last.part > r.part || last.work > r.work;
    }
}
void rb_release_report() {
    struct Agg { int64_t n = 0, open = 0, early = 0, later = 0, later_early = 0, later_last = 0; };
    std::map<uint64_t, Agg> agg;
    {
        std::lock_guard lk(rb_rel_mu);
        std::erase_if(rb_rels, [&](RbRelease& r) {
            if (!r.closed) return false;
            if (!r.seen) return !(r.seen = true);   // (one more interval: a read again of it comes after its recording too)
            Agg& a = agg[r.size];
            a.n++, a.open += r.open, a.early += r.early, a.later += r.later, a.later_early += r.later && r.early;
            a.later_last += r.later && !r.again;
            return true;
        });
    }
    for (const auto& [size, a] : agg)
        if (a.n >= 3)
            logf("vk: readback releases, size %#llx: %lld let go before their writer's submission was done (%lld by the early wait), "
                 "%lld of them while it was still recorded; %lld before a writer of the range recorded later in it (%lld by the early "
                 "wait), %lld of those the thread's last read of the range for it", (unsigned long long)size, (long long)a.n,
                 (long long)a.early, (long long)a.open, (long long)a.later, (long long)a.later_early, (long long)a.later_last);
}
/// (hookshot) invalidate's wait for a write-back's value, at most RX_VK_READBACK_WAIT_MS (default 250; 0: no limit, as before).
/// On the Steam Deck (RADV, no buffer mirrors, queues replayed in submit order) the game's main thread and one of its jobs
/// waited here, right after Kokiri Forest's save loaded, for a value the timeline never reached while they waited (it had been
/// submitted; the GPU stayed one value short), and the game stopped for good: 2 of 8 starts, 4 of 4 with
/// RX_VK_CAPTURE_WAIT=0. Past the limit the CPU reads what memory holds, as the capture's wait does (vk_timeline_wait), and
/// the first times are logged with how far the timeline and the submits had got: 6-8 a start, all in the save's load, and no
/// start stopped (0 of 4). True when reached
bool readback_wait_value(uint64_t v) {
    static const int64_t limit_ms = getenv("RX_VK_READBACK_WAIT_MS") ? atoll(getenv("RX_VK_READBACK_WAIT_MS")) : 250;
    if (limit_ms <= 0) {
        wait_value(v);
        return true;
    }
    queue_flush_for(FR_VALUE);
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &v;
    if (vkWaitSemaphores(dev.device, &wi, (uint64_t)limit_ms * 1000000ull) != VK_TIMEOUT) return true;
    static std::atomic<int> logged{0};
    if (logged++ < 8)
        logf("vk: a readback's wait for write-back %llu timed out after %lld ms (done %llu, submitted %llu, handed out %llu)",
             (unsigned long long)v, (long long)limit_ms, (unsigned long long)completed(), (unsigned long long)pushed.load(),
             (unsigned long long)q_values.load());
    return false;
}
// The diagnostic's second snapshot must be after the whole submission even if the normal path succeeded at an early part.
// A failed wait makes the comparison invalid; it must not be reported as an unchanged range.
void rb_diag_finish(RbDiag& d, uint64_t value) {
    if (!d.snap) return;
    // value_rec can still name an earlier part while the rest is being recorded. The submission timeline advances only
    // when the whole value is finished; a timeout leaves no valid second snapshot.
    d.snap = readback_wait_value(value);
}
/// (hookshot) RX_VK_READBACK_LATE (readback_late.h): whether a readback of that size, written by the newest submission or
/// not, is read without its wait. Off unless the switch is set
bool readback_late_wanted(uint64_t size, bool newest) {
    static const readback_late::Rules rules = [] {
        const char* e = getenv("RX_VK_READBACK_LATE");
        const readback_late::Rules r = readback_late::parse(e);
        if (r.any()) logf("vk: readbacks read without their wait: RX_VK_READBACK_LATE=%s (newest %d, %d sizes)", e, (int)r.newest, (int)r.sizes.size());
        return r;
    }();
    return readback_late::wanted(rules, size, newest);
}
/// (hookshot) RX_VK_LOG_READBACK_RANGES=1: the readbacks by range, in the log every 3000 calls: how many waited and for how
/// long, how many had the newest submission as their writer, how many were read without their wait (RX_VK_READBACK_LATE).
/// How the ranges worth a rule are found on a given PC
struct RbRange {
    uint64_t gpu, size;
    int64_t calls, waits, ns, newest, late;
    int64_t age = 0, aged = 0, at_ns = 0, sub = 0, after_wait = 0;   // (hookshot: see gf_note_value)
};
std::mutex rb_ranges_mu;
std::vector<RbRange> rb_ranges;
// (hookshot, RX_VK_LOG_READBACK_RANGES) How old what a readback gets is, in the game's frames: each submission's value with
// the game's frame (its presents handed over so far) and its place among that frame's submissions, kept for the last 4096
// values; a read's age is the frames handed over since the newest write-back of its range (0: written by work submitted
// since the last present). And when in the game's frame the read comes (ms after its last present's hand-over), and whether
// the reading thread waited for the GPU (nvnSyncWait, nvnQueueFinish) in the 2 ms before it
struct GfValue {
    uint32_t frame = 0, index = 0;
};
GfValue gf_values[4096];
std::atomic<uint32_t> gf_frame{0}, gf_index{0};
std::atomic<int64_t> gf_present_ns{0};
void gf_note_value(uint64_t value) { gf_values[value % 4096] = {gf_frame.load(), gf_index.fetch_add(1)}; }
void gf_note_present() { gf_frame++, gf_index = 0, gf_present_ns = gf_now(); }
void rb_range_age(uint64_t gpu, uint64_t size, uint64_t value) {
    const GfValue e = gf_values[value % 4096];
    const int64_t now = gf_now();
    std::lock_guard lk(rb_ranges_mu);
    for (auto& r : rb_ranges)
        if (r.gpu == gpu && r.size == size) {
            r.age += (int64_t)gf_frame.load() - e.frame, r.aged++, r.at_ns += now - gf_present_ns.load(), r.sub += e.index;
            r.after_wait += gf_waited_ns && now - gf_waited_ns < 2000000;
            return;
        }
}
void rb_range_note(uint64_t gpu, uint64_t size, bool waited, int64_t ns, bool newest, bool late) {
    std::lock_guard lk(rb_ranges_mu);
    for (auto& r : rb_ranges)
        if (r.gpu == gpu && r.size == size) {
            r.calls++;
            r.waits += waited;
            r.ns += ns;
            r.newest += newest;
            r.late += late;
            return;
        }
    if (rb_ranges.size() < 256) rb_ranges.push_back({gpu, size, 1, waited, ns, newest, late});
}
void rb_ranges_log() {
    static std::atomic<int> logged{0};
    std::vector<RbRange> top;
    {
        std::lock_guard lk(rb_ranges_mu);
        top = rb_ranges;
        rb_ranges.clear();
    }
    if (logged++ >= 400) return;
    static const bool by_size = getenv("RX_VK_LOG_READBACK_RANGES") && atoi(getenv("RX_VK_LOG_READBACK_RANGES")) == 2;   // (the largest first)
    std::sort(top.begin(), top.end(), [](const RbRange& a, const RbRange& b) {
        if (by_size && a.size != b.size) return a.size > b.size;
        return a.ns + a.late * 5000000 > b.ns + b.late * 5000000;
    });
    int64_t late = 0;
    for (const auto& r : top) late += r.late;
    std::string s;
    for (size_t i = 0; i < top.size() && i < 8; i++) {
        char b[360];
        const double n = top[i].aged ? (double)top[i].aged : 1.0;
        snprintf(b, sizeof b, " | %#llx +%#llx: %lld calls, %lld waited %.1f ms, %lld newest, %lld late; got work %.2f frames old "
                 "(its frame's submit %.1f), read %.1f ms into the frame, %lld after a GPU wait of its thread",
                 (unsigned long long)top[i].gpu, (unsigned long long)top[i].size, (long long)top[i].calls,
                 (long long)top[i].waits, top[i].ns / 1e6, (long long)top[i].newest, (long long)top[i].late, top[i].age / n,
                 top[i].sub / n, top[i].at_ns / 1e6 / n, (long long)top[i].after_wait);
        s += b;
    }
    logf("vk: readback ranges at present %lld: %d ranges, %lld read late%s", (long long)presents_now, (int)top.size(),
         (long long)late, s.c_str());
}
void readback_fell_back(int what = 0);   // (the render thread's log)
// ---- (hookshot) RX_VK_LOG_EARLY_CHECK: what a read RX_VK_READBACK_EARLY let go would have missed against main's wait ----
// sheep's probe (Bunny Hood #2: the reader held at the early part, then for the whole submission) saw the 0x3980 camera block change
// after the early boundary in 269 of 745 compares, and could not tell the game's own writes from the GPU's; and the whole submission
// is more than main waits for. Here, for one read let go early in RX_VK_LOG_EARLY_CHECK (1: every one; 4: one in four):
//  - s1: the range as the early wait returns it;
//  - sm: the range right after the wait main makes for the same read (invalidate's usual choice for pages written in place: the
//    newest part of its submission submitted so far, wait_recorded, else the whole submission), made by this read now (the sampled
//    reads come back as main's would: a check run is not a measurement). s1 -> sm is what the early wait let the game miss;
//  - s2: the range once the read's submission is done on the main timeline, copied by a thread of its own after it waits (later,
//    informative only: a copy after a host wait is not at the GPU's boundary). sm -> s2 is the parts recorded after the read, which
//    main misses too.
// The game's own writes: the soft write watch (only CPU stores, and the host's marked writes, mark a page; the GPU's writes in place
// never do). A page the collections report from the read until the second full collection begun after a copy (memory_collect_written:
// a mark published late) is the CPU's in that interval. A change on a page the CPU wrote is ambiguous (the GPU may have written the
// same page): what proves a missed GPU write is a change on a page the CPU did not write, and the clean intervals (no CPU page) are
// counted. A host write into guest memory that is not marked (some nvn getters) would show as the GPU's. The GPU writes the replay
// records after the read in a later part of the read's submission are counted by kind
struct EcDiff {
    int64_t words_gpu = 0, words_cpu = 0;   // (changed 4-byte words on pages the CPU did not write, and on pages it did)
    int64_t first_gpu = -1;                 // (the offset of the first such word on a page the CPU did not write)
};
/// a -> b over [guest, guest + size): changed words, by whether the CPU wrote a page they are on (bit k of cpu: the range's k-th page)
EcDiff ec_compare(const uint8_t* a, const uint8_t* b, uint64_t guest, uint64_t size, uint64_t cpu) {
    EcDiff d;
    const auto touched = [&](uint64_t off) {
        const uint64_t k = ((guest + off) >> 12) - (guest >> 12);
        return k < 64 && ((cpu >> k) & 1);
    };
    for (uint64_t off = 0; off < size; off += 4) {
        const uint64_t n = std::min<uint64_t>(4, size - off);
        if (!std::memcmp(a + off, b + off, n)) continue;
        if (touched(off) || touched(off + n - 1)) {   // (a word across a page's end: either page)
            d.words_cpu++;
        } else {
            if (d.first_gpu < 0) d.first_gpu = (int64_t)off;
            d.words_gpu++;
        }
    }
    return d;
}
/// the pages of [guest, guest + size) among the collected ones (in any order), as bits by the range's page
uint64_t ec_pages_hit(uint64_t guest, uint64_t size, const uint32_t* pages, int64_t n) {
    const uint64_t p0 = guest >> 12, p1 = (guest + size - 1) >> 12;
    uint64_t m = 0;
    for (int64_t i = 0; i < n; i++)
        if (pages[i] >= p0 && pages[i] <= p1 && pages[i] - p0 < 64) m |= 1ull << (pages[i] - p0);
    return m;
}
int early_check_every() {   // (RX_VK_LOG_EARLY_CHECK=<n>: one read in n let go early is checked; 0, unset or RX_VK_READBACK_EARLY off: none)
    static const int n = [] {
        const char* e = getenv("RX_VK_LOG_EARLY_CHECK");
        return readback_early_on() && e ? std::max(0, atoi(e)) : 0;
    }();
    return n;
}
bool early_check_on() { return early_check_every() > 0; }
struct EarlyCheck {
    uint64_t gpu = 0, guest = 0, size = 0, value = 0, part = 0;
    std::vector<uint8_t> s1, sm, s2;
    uint64_t cpu1 = 0, cpu2 = 0;      // (pages the CPU wrote: s1..sm, sm..s2)
    uint32_t gen_m = 0, gen_2 = 0;    // (the full collections begun when sm, s2 had been copied)
    bool closed1 = false, closed2 = false, valid1 = false, valid2 = false;
    int stage = 0;                    // (0: sm being taken by the read, 1: s2 to take, 2: the collections to close the intervals)
    uint64_t wb_2 = 0;                // (the range's newest write-back noted when s2 was taken: a newer submission's?)
    int gpu_noted = 0, kinds = 0;     // (the replay's own GPU writes after the read, later parts: bit 0 binding, 1 over 1 MiB, 2 bare or wide)
};
std::mutex ec_mu;
std::condition_variable ec_cv;
std::deque<std::unique_ptr<EarlyCheck>> ec_pending;   // (ec_mu; in the order of the reads)
std::atomic<uint32_t> ec_full_begun{0};
struct EcStats {
    int64_t n = 0, lost1 = 0, same1 = 0, gpu1 = 0, amb1 = 0, clean1 = 0, clean_gpu1 = 0;
    int64_t lost2 = 0, same2 = 0, gpu2 = 0, amb2 = 0, newer2 = 0, noted = 0;
    int64_t kinds[3] = {};
};
EcStats ec_st;                               // (ec_mu)
std::map<uint64_t, EcStats> ec_by_size;      // (ec_mu)
std::atomic<int64_t> ec_skipped{0};
bool ec_wait(VkSemaphore s, uint64_t v) {
    if (!s || !v) return false;
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &s;
    wi.pValues = &v;
    return vkWaitSemaphores(dev.device, &wi, 3000000000ull) == VK_SUCCESS;
}
void ec_finish(EarlyCheck& e) {   // (ec_mu held) one check's counts
    EcStats* ss[2] = {&ec_st, &ec_by_size[e.size]};
    const EcDiff d1 = e.valid1 ? ec_compare(e.s1.data(), e.sm.data(), e.guest, e.size, e.cpu1) : EcDiff{};
    const EcDiff d2 = e.valid2 ? ec_compare(e.sm.data(), e.s2.data(), e.guest, e.size, e.cpu2) : EcDiff{};
    const bool newer = e.wb_2 > e.value;   // (a later submission was recorded writing the range before s2: its changes may be in it)
    for (EcStats* s : ss) {
        s->n++;
        if (!e.valid1) s->lost1++;
        else if (d1.words_gpu) s->gpu1++;
        else if (d1.words_cpu) s->amb1++;
        else s->same1++;
        if (e.valid1 && !e.cpu1) s->clean1++, s->clean_gpu1 += d1.words_gpu != 0;
        if (!e.valid2) s->lost2++;
        else if (newer && (d2.words_gpu || d2.words_cpu)) s->newer2++;
        else if (d2.words_gpu) s->gpu2++;
        else if (d2.words_cpu) s->amb2++;
        else s->same2++;
        s->noted += e.gpu_noted != 0;
        for (int k = 0; k < 3; k++) s->kinds[k] += (e.kinds >> k) & 1;
    }
    static int told1, told2;
    if (d1.words_gpu && told1++ < 16)
        logf("vk: early-read check: %#llx +%#llx (submission %llu, let go at part %llu): %lld words changed by main's release on pages "
             "the CPU did not write, the first at +%#llx (%lld on pages it wrote); GPU writes recorded after the read: %d (kinds %#x)",
             (unsigned long long)e.gpu, (unsigned long long)e.size, (unsigned long long)e.value, (unsigned long long)e.part,
             (long long)d1.words_gpu, (unsigned long long)d1.first_gpu, (long long)d1.words_cpu, e.gpu_noted, e.kinds);
    if (d2.words_gpu && !newer && told2++ < 16)
        logf("vk: early-read check: %#llx +%#llx (submission %llu, let go at part %llu): %lld words changed after main's release, before "
             "the submission's end, on pages the CPU did not write, the first at +%#llx; GPU writes recorded after the read: %d (kinds %#x)",
             (unsigned long long)e.gpu, (unsigned long long)e.size, (unsigned long long)e.value, (unsigned long long)e.part,
             (long long)d2.words_gpu, (unsigned long long)d2.first_gpu, e.gpu_noted, e.kinds);
    if (ec_st.n >= 100) {
        std::string sizes;
        for (const auto& [sz, s] : ec_by_size) {
            char b[200];
            snprintf(b, sizeof b, " | %#llx: %lld, missed %lld (clean %lld, missed of them %lld), ambiguous %lld, after main's %lld",
                     (unsigned long long)sz, (long long)s.n, (long long)s.gpu1, (long long)s.clean1, (long long)s.clean_gpu1,
                     (long long)s.amb1, (long long)s.gpu2);
            sizes += b;
        }
        logf("vk: early-read check (RX_VK_LOG_EARLY_CHECK=%d), %lld reads (%lld skipped): early -> main's release: %lld unchanged, %lld "
             "changed on a page the CPU did not write (a GPU write main waits for and the early wait missed: should be 0), %lld changed "
             "only on pages the CPU wrote (ambiguous), %lld lost; %lld with no CPU page written (clean), %lld of them missed; main's "
             "release -> the submission's end: %lld unchanged, %lld GPU (parts recorded after the read), %lld ambiguous, %lld a newer "
             "submission's, %lld lost; GPU writes the replay recorded after the read in a later part: %lld reads (bindings %lld, over "
             "1 MiB %lld, bare or wide %lld)%s",
             early_check_every(), (long long)ec_st.n, (long long)ec_skipped.exchange(0), (long long)ec_st.same1, (long long)ec_st.gpu1,
             (long long)ec_st.amb1, (long long)ec_st.lost1, (long long)ec_st.clean1, (long long)ec_st.clean_gpu1, (long long)ec_st.same2,
             (long long)ec_st.gpu2, (long long)ec_st.amb2, (long long)ec_st.newer2, (long long)ec_st.lost2, (long long)ec_st.noted,
             (long long)ec_st.kinds[0], (long long)ec_st.kinds[1], (long long)ec_st.kinds[2], sizes.c_str());
        ec_st = EcStats{};
        ec_by_size.clear();
    }
}
void ec_reap() {   // (ec_mu held) the checks done, in the order of their reads
    while (!ec_pending.empty() && ec_pending.front()->stage == 2 && ec_pending.front()->closed1 && ec_pending.front()->closed2) {
        ec_finish(*ec_pending.front());
        ec_pending.pop_front();
    }
}
void early_check_thread() {   // (s2: once each check's submission is done on the main timeline)
    for (;;) {
        EarlyCheck* e = nullptr;
        {
            std::unique_lock lk(ec_mu);
            ec_cv.wait(lk, [] {
                for (auto& x : ec_pending) if (x->stage == 1) return true;
                return false;
            });
            for (auto& x : ec_pending)
                if (x->stage == 1) { e = x.get(); break; }
        }
        // (the entry stays in ec_pending, and only reaches stage 2, which may be reaped, below: its address holds)
        const bool ok = ec_wait(timeline, e->value);
        std::vector<uint8_t> snap(e->size);
        const bool read = ok && host->gpu_read(e->gpu, snap.data(), e->size);
        const uint64_t wb = read ? memory_writeback_value(e->guest, e->size, nullptr) : 0;
        std::lock_guard lk(ec_mu);
        e->s2 = std::move(snap), e->valid2 = read, e->wb_2 = wb, e->stage = 2;
        e->gen_2 = ec_full_begun.load();   // (after the copy: a collection begun after this saw any CPU write before it)
        e->cpu2 |= e->cpu1;   // (pages collected while sm was copied are in cpu1 only)
        if (!read) e->closed2 = true;
        ec_reap();
    }
}
/// (invalidate) one read in RX_VK_LOG_EARLY_CHECK let go at its writer's part: what it returns, then main's wait for it and what that
/// returns. v: the range's newest write-back; checked only when it is the in-place writer's (no mirror's part in it), up to 64 KiB
void early_check_read(uint64_t gpu, uint64_t size, uint64_t v, uint64_t whole, uint64_t part, bool unforced) {
    static std::atomic<int64_t> reads{0};
    if (reads.fetch_add(1, std::memory_order_relaxed) % early_check_every()) return;
    if (size > 0x10000 || v != whole) {
        ec_skipped++;
        return;
    }
    static std::once_flag started;
    std::call_once(started, [] { std::thread(early_check_thread).detach(); });
    EarlyCheck* e = nullptr;
    bool ok1 = false;
    {
        std::lock_guard lk(ec_mu);
        if (ec_pending.size() >= 32) {
            ec_skipped++;
            return;
        }
        auto x = std::make_unique<EarlyCheck>();
        x->gpu = gpu, x->guest = host->gpu_guest(gpu), x->size = size, x->value = whole, x->part = part;
        // (copied under ec_mu and listed before it is let go: a collection that finds a CPU write made after the copy reports it after
        // it scanned, so after this, and finds the check)
        x->s1.resize(size);
        ok1 = host->gpu_read(gpu, x->s1.data(), size);
        e = x.get();
        ec_pending.push_back(std::move(x));
    }
    // (main's wait for this read: invalidate's usual choice for a page written in place whose submission is pending)
    int ok = 0;
    if (!(unforced && wait_recorded(v, UINT64_MAX, &ok))) {
        if (unforced && replay_behind()) wait_queue_empty();
        readback_wait_value(v);
    }
    std::vector<uint8_t> sm(size);
    const bool okm = host->gpu_read(gpu, sm.data(), size);
    std::lock_guard lk(ec_mu);
    e->sm = std::move(sm), e->valid1 = ok1 && okm, e->stage = 1;
    e->gen_m = ec_full_begun.load();   // (after the copy)
    ec_cv.notify_one();
}
// (hookshot) the invalidates that kept their virtual core (invalidate_quick) and the ones that gave it up, counted on each
// thread and added here every 256 calls (the kept ones come per shape per view: no shared line written at each), on the
// readbacks line's cadence below
std::atomic<int64_t> inv_kept{0}, inv_gave{0};
void inv_count(bool kept) {
    thread_local uint32_t k = 0, g = 0;
    (kept ? k : g)++;
    if (k + g < 256) return;
    inv_kept.fetch_add(k, std::memory_order_relaxed), inv_gave.fetch_add(g, std::memory_order_relaxed);
    k = g = 0;
}
void invalidate(uint64_t gpu, uint64_t size) {
    static const bool log_inv = env_on("RX_VK_LOG_INVALIDATE");
    static std::atomic<int> logged{0};
    if (log_inv && logged++ < 400) {
        std::string hits;
        {
            std::lock_guard lk(replay_mu);
            texture_describe_written(gpu, size, hits);
        }
        logf("vk: invalidate %#llx +%#llx:%s", (unsigned long long)gpu, (unsigned long long)size,
             hits.empty() ? " no texture the GPU wrote" : hits.c_str());
    }
    // (hookshot) a range the CPU reads: a large binding over it keeps its write-back (wb_learn.h, RX_VK_WB_MAX_KB)
    memory_note_cpu_read(host->gpu_guest(gpu), size);
    {   // (EvilBoyce, diagnostics) RX_VK_LOG_READBACK=<present>: from that present on, 300 invalidates with the fence handed
        // out, the GPU's completed value, whether the replay is behind, and the first words the CPU is about to read
        static const int64_t from = getenv("RX_VK_LOG_READBACK") ? atoll(getenv("RX_VK_LOG_READBACK")) : -1;
        static std::atomic<int> n{0};
        if (from >= 0 && presents_now >= from && n++ < 300) {
            uint32_t w[8]{};
            host->gpu_read(gpu, w, sizeof w);
            logf("vk: readback %lld %#llx +%#llx: fence %llu done %llu behind %d | %08x %08x %08x %08x %08x %08x %08x %08x",
                 (long long)presents_now, (unsigned long long)gpu, (unsigned long long)size, (unsigned long long)fence(),
                 (unsigned long long)completed(), (int)replay_behind(), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        }
    }
    // (EvilBoyce) GPU results in buffers: the CPU reads them once the submission that writes them back is done. The game
    // reads them when its own pacing says the GPU has finished that frame, which on the console's coherent memory means the
    // values are there; here they reach guest memory only with the write-back at the end of the submission that the replay
    // may not have recorded or the GPU not finished yet. OoT reads its exposure meter (a sum the GPU adds to each frame and
    // the CPU reads and sets to 0) this way: it read 0 one frame, then the old sum came back over the CPU's 0 and the next
    // frame added to it (2x): the exposure stepped 0.2 EV up and down several times a second (the white flashes).
    if (size) {
        bool ever = false;
        if ((memory_writeback_value(host->gpu_guest(gpu), size, &ever), ever)) {
            memory_note_readback(host->gpu_guest(gpu), size);   // (PR 430: the CPU may write it back: memory_rb_refresh)
            struct Waiting {   // (readback_send_on: what is held goes out now, and while this thread waits)
                bool on;
                explicit Waiting(bool w) : on(w) { if (on) readback_waiting++, queue_flush_for(FR_READBACK); }
                ~Waiting() { if (on) readback_waiting--; }
            } waiting(readback_send_on() && memory_writeback_value(host->gpu_guest(gpu), size, nullptr) > completed());
            static std::atomic<int64_t> calls{0}, waited{0}, ns{0};
            const auto w0 = std::chrono::steady_clock::now();
            bool w = false;
            const uint64_t f0 = fence(), c0 = completed();
            const bool behind0 = replay_behind();
            static const bool no_catchup = env_on("RX_VK_READBACK_NO_CATCHUP");   // (A/B: recorded work only)
            // (hookshot) not on the submitting thread's replay (the default): there what is handed over is recorded at once, and
            // the work not recorded yet is stopped at an event wait. Catching up forced it past that wait, ahead of the work it
            // waits for (OoT's presentation work before the scene's: the fog of the frame before, for a frame, at 1 in 20 of the
            // exposure readbacks). The write-back is waited for on the recordings' timeline instead, which a stopped queue
            // does not hold back. The stopped work whose waits are met goes on first, never forced (sync_settle): it may be what
            // writes this range, and its write-back is known only once it is recorded. Without it the CPU read the range
            // before that work, with nothing to wait for (once d6c5ddcc's catch-up no longer forced anything, it was left
            // out here too). RX_VK_PER_QUEUE_FENCES=0 waits for every queue's work instead: the game's readbacks 19-87 ms
            // each on an RX 5500M, against ~1 ms here
            // (hookshot) RX_VK_ASYNC=1, the render thread: no catch-up either, and nothing forced. The game is a frame ahead
            // of the replay by design there: what it reads is what the GPU has written so far, as on the console, whose GPU
            // runs behind the CPU too. What must hold is that no write-back lands on the range after the CPU has read it and
            // written its answer (the exposure sum the CPU reads and sets to 0): the write-backs of the work recorded so far
            // are waited for below, by part; the work still queued takes the range from guest memory when the GPU gets to it
            // (memory_rb_refresh), after the CPU's answer. Catching up instead (the older replay thread, removed) held the
            // game's job threads for the whole replay of the frame, 7.5-11 ms a readback in the Market, and forced half of
            // the stopped submissions past their waits (the flashes)
            // (not unforced: the submitting thread's replay with a dump being written (dump_on) or RX_VK_PER_QUEUE_FENCES=0, as before)
            const bool behind = async_on();
            const bool unforced = behind || (sync_waits_on() && !per_queue_fences_off());
            static std::atomic<int64_t> settled{0}, held_back{0}, in_place{0};
            // That is the console's answer for a read the game makes without waiting for the GPU: the latest result that has
            // landed. Where the game does wait first (a fence, nvnSyncWait, nvnQueueFinish), that wait is the exact one and
            // the read after it finds its write-back recorded. RX_VK_READBACK_EXACT (a test, off): 1, every range is read
            // after everything handed over before the read is replayed as far as the game has let it go (wait_queue_empty
            // there: nothing forced, and work stopped behind a signal the CPU has not made yet is not waited for); 2, only
            // the ranges over 16 KiB (tables the CPU acts on, not meters: Hyrule Field's 0x8b040 and 0x9b040 bytes)
            static const int exact_env = getenv("RX_VK_READBACK_EXACT") ? atoi(getenv("RX_VK_READBACK_EXACT")) : 0;
            const bool exact = behind && (exact_env == 1 || (exact_env == 2 && size > 0x4000));
            static std::atomic<int64_t> st_early_waits{0}, st_early_ns{0}, st_early_whole{0};   // (RX_VK_READBACK_EARLY)
            static std::atomic<int64_t> st_early_rec{0}, st_early_rec_ns{0}, st_early_rec_late{0};   // (and the waits for the writer's recording first)
            if (behind0 && !no_catchup) {   // (work handed over may write it back too)
                if (!unforced) wait_queue_empty();
                else if (behind) {   // (the render thread: neither, see above; but an exact range)
                    if (exact) wait_queue_empty(), readback_fell_back(3);
                }
                else if (sync_settle_met()) settled++;
                w = !behind || exact;
            }
            const auto w1 = std::chrono::steady_clock::now();
            // (hookshot) what is waited for is the part of the submission that writes the range back, when its pages come
            // from a mirror (memory_end_submission, at each part's end), not the submission's last part: the range is in
            // guest memory once that part's copy is done, and the later parts read their mirrors and the submit's capture,
            // not guest memory. Since a value's recording is its last part's (issue 997) every readback waited for the GPU's
            // whole submission: in Hyrule Field one job a frame for 5.5 ms (0.1 ms before), the game's other job threads
            // polling meanwhile: 79 -> 64-71 fps and 39 -> 64-73 ms of CPU a frame on an RTX 3090. A page written in place (no
            // mirror: `whole`) keeps the wait for all of its submission, whose later parts read and write guest memory
            // itself.
            uint64_t part = 0, whole = 0;
            const uint64_t v = memory_writeback_value(host->gpu_guest(gpu), size, nullptr, &part, &whole);
            static const bool log_each = env_on("RX_VK_LOG_READBACK_WAITS");
            RbDiag rbd;   // (RX_VK_LOG_RB_WRITERS: its writers, and its last writer's part waited for first, timed: below)
            uint64_t l_part = 0, l_last = 0, l_sent = 0, l_done = 0;   // (the log's: recordings, on their timeline)
            if (log_each && v > completed() && rec_timeline) {
                std::lock_guard mk(rec_map_mu);
                if (auto it = part_rec.find(part); it != part_rec.end()) l_part = it->second;
                if (auto it = value_rec.find(v); it != value_rec.end()) l_last = it->second;
                if (!part_rec.empty()) l_sent = part_rec.rbegin()->second;
                vkGetSemaphoreCounterValue(dev.device, rec_timeline, &l_done);
            }
            const bool pending = v > completed(), newest = pending && v >= f0;
            const bool late = pending && readback_late_wanted(size, newest);   // (RX_VK_READBACK_LATE: read without its wait)
            if (rb_writers_on() && pending && !late && !rb_probe_only())
                rb_diag_wait(gpu, size, rbd, whole, part, unforced && whole > completed());
            // (hookshot) RX_VK_READBACK_EARLY (on where no pool has a mirror): pages written in place (the Steam Deck's every page) wait for
            // up to the part their last writer was recorded in, not their whole submission. There the game's job polled ~18 ms a frame
            // for its camera block's read while the GPU finished the rest of a 22-26 ms submission after the writer. The parts after
            // it may still read guest memory in place, which the game, let go, may write: as on the console, whose GPU reaches the
            // writer before the rest of its command buffer too, but the replay's GPU runs later than the console's (issue 997 was
            // reads let go before their writer: value_rec). Pages whose reads waited end their writer's part right after it
            // (memory_split_wanted). Not known (0: written before it was on, or its part not mapped): the whole submission, as before. A
            // range RX_VK_READBACK_LATE reads late is not waited for at all, here either
            bool early_done = false;
            // (only a read with a page written in place pending: with VRAM mirrors (a dGPU) a page is written back by part already,
            // and nothing is noted, split or rebuilt for it: on an RTX 5090 the switch cost ~2 fps doing that and waited early 0 times.
            // A sparse mirror's block with no memory yet is written in place for a submission: those pages take the early wait)
            if (pending && !late && readback_early_on() && unforced && whole > completed()) {
                memory_rb_waited(host->gpu_guest(gpu), size);
                // (its writer's submission recorded to its end first: the part found below is then its last writer's, not the
                // newest recorded so far. Past 250 ms, the whole submission as before)
                double rec_ms = 0;
                const bool recorded = early_recorded_wait(whole, &rec_ms);
                if (rec_ms > 0) {
                    st_early_rec.fetch_add(1, std::memory_order_relaxed);
                    st_early_rec_ns.fetch_add((int64_t)(rec_ms * 1e6), std::memory_order_relaxed);
                }
                if (!recorded) st_early_rec_late.fetch_add(1, std::memory_order_relaxed);
                const uint64_t writer = recorded ? memory_in_place_writer_part(host->gpu_guest(gpu), size, whole) : 0;
                const uint64_t target = std::max(writer, part);   // (a mirror's part as well, when pages of both kinds are read)
                int ok = 0;
                const auto e0 = std::chrono::steady_clock::now();
                if (writer && wait_recorded_part(target, UINT64_MAX, &ok)) {
                    early_done = true, w = true;
                    if (early_check_on()) early_check_read(gpu, size, v, whole, target, unforced);   // (RX_VK_LOG_EARLY_CHECK)
                    st_early_waits.fetch_add(1, std::memory_order_relaxed);
                    st_early_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - e0).count(),
                                          std::memory_order_relaxed);
                } else {
                    st_early_whole.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (early_done) {
            } else if (pending && !late) {
                int ok = 0;
                // (and not when a part after it reads guest memory in place, on a card whose pools are not all mirrored
                // or whose mirrors take pages no capture holds: the game, let go after the first part, would write what
                // those parts still read. Then the last part, as before)
                const bool in_place_after = memory_in_place_part() > part;
                if (part && in_place_after) held_back++;
                const bool parted = unforced && part && whole <= completed() && !in_place_after &&
                                    wait_recorded_part(part, UINT64_MAX, &ok);
                if (!parted && !part) in_place++;   // (pages written in place (no mirror: part 0): the whole submission waited for)
                if (!parted && !(unforced && wait_recorded(v, UINT64_MAX, &ok))) {
                    if (unforced && replay_behind()) wait_queue_empty();   // (not recorded whole: as before)
                    if (behind) readback_fell_back();
                    readback_wait_value(v);
                }
                w = true;
            }
            if (w) {
                rb_diag_finish(rbd, v);
                waited++;
                const int64_t d = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count();
                ns += d;
                static std::atomic<int> logged{0};
                const bool line = log_each && presents_now > 4000 && logged++ < 300;
                if (line)
                    logf("vk: readback wait %lld %#llx +%#llx: %.2f ms (catch-up %.2f) value %llu, fence %llu, done %llu, behind %d; "
                         "recordings: its part's %llu, its submission's last %llu (in place: value %llu), %llu submitted, %llu done",
                         (long long)presents_now, (unsigned long long)gpu, (unsigned long long)size, d / 1e6,
                         std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count() / 1e6, (unsigned long long)v,
                         (unsigned long long)f0, (unsigned long long)c0, (int)behind0, (unsigned long long)l_part,
                         (unsigned long long)l_last, (unsigned long long)whole, (unsigned long long)l_sent, (unsigned long long)l_done);
                if (presents_now > 4000) rb_diag_done(gpu, size, rbd, d / 1e6, line);
            }
            if (rb_writers_on() && presents_now > 4000) rb_release_note(gpu, size, v, pending, early_done);
            static const bool log_ranges = env_on("RX_VK_LOG_READBACK_RANGES");
            if (log_ranges) {
                rb_range_note(gpu, size, w,
                              w ? std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count() : 0,
                              newest, late);
                // (what the CPU is about to read: the range's newest write-back, waited for above unless read late)
                rb_range_age(gpu, size, late ? std::min<uint64_t>(v, completed()) : v);
            }
            if (++calls % 3000 == 0) {
                if (log_ranges) rb_ranges_log();
                const int64_t n = waited.exchange(0);
                logf("vk: readbacks of GPU results: %lld of 3000 waited for their write-back, %.2f ms each; %lld looked at the "
                     "stopped queues first, %lld waited for their submission's last part (guest memory read in place after theirs), "
                     "%lld waited for their whole submission (their pages written in place, no mirror)",
                     (long long)n, n ? ns.exchange(0) / (double)n / 1e6 : 0.0, (long long)settled.exchange(0),
                     (long long)held_back.exchange(0), (long long)in_place.exchange(0));
                if (rb_writers_on()) rb_diag_report(), rb_release_report();
                if (readback_early_on()) {
                    const int64_t e = st_early_waits.exchange(0), r = st_early_rec.exchange(0);
                    logf("vk: readbacks waited for their writer's part (RX_VK_READBACK_EARLY): %lld, %.2f ms each; %lld for their whole "
                         "submission (writer not known); %lld parts ended after a readback's writer; %lld waited for their writer's "
                         "submission to be recorded first, %.2f ms each (%lld of them past 250 ms: their whole submission)", (long long)e,
                         e ? st_early_ns.exchange(0) / (double)e / 1e6 : 0.0, (long long)st_early_whole.exchange(0),
                         (long long)st_early_splits.exchange(0), (long long)r, r ? st_early_rec_ns.exchange(0) / (double)r / 1e6 : 0.0,
                         (long long)st_early_rec_late.exchange(0));
                }
                logf("vk: invalidates since the last readbacks line: %lld kept their virtual core (nothing there to wait for), %lld "
                     "gave it up (counted by 256 a thread; RX_VK_INVALIDATE_QUICK=0: none kept)",
                     (long long)inv_kept.exchange(0), (long long)inv_gave.exchange(0));
            }
        }
    }
    // (textures the GPU wrote there: their texels into memory before the CPU reads it, as vmap_changing does. The game
    // waited for the work that wrote them; the replay is caught up first)
    {   // (RX_VK_READBACK_OFF_FILE=<path> (A/B): while that file exists, not)
        static const char* off_file = getenv("RX_VK_READBACK_OFF_FILE");
        if (off_file && GetFileAttributesA(off_file) != INVALID_FILE_ATTRIBUTES) return;
    }
    // (the replay caught up first only for ranges a texture the GPU wrote may be in: MKW invalidates a few buffer
    // ranges every frame (GPU results it reads back, written back with their submission; one of them a ring of
    // slots at a new address each time), and the catch-up for each held its main thread until the replay had
    // recorded everything queued: ~10% of the main thread's time. A 1 MB region caught up for kProbe times without a
    // texture found holds buffers only from then on; one a texture was ever found in keeps the catch-up.)
    // (decided by the live textures' storage, not by what the replay has listed: a region guessed to hold buffers
    // only, after probes that found no texture, left MKW's Grand Prix waiting at its map for a texture's texels)
    if (!size) return;
    // (hookshot) and on the submitting thread's replay, the catch-up lets on what may go without forcing anything past its wait
    // (once the exposure readback above no longer did, this one forced OoT's presentation work on, every frame or two)
    // (RX_VK_ASYNC=1, the render thread: wait_queue_empty there waits for what was handed over before this call, nothing
    // forced)
    const bool unforced = !async_on() && sync_waits_on() && !per_queue_fences_off();
    if (!texture_gpu_written_in(gpu, size)) {
        if (!replay_behind()) return;
        if (!tex_pages_any(gpu, size)) return;   // (no texture's storage there: buffers only)
        if (async_on()) {
            if (!tex_live_any(gpu, size)) return readback_fell_back(1);
            readback_fell_back(2);
            static std::atomic<int> told{0};
            if (told++ < 24) logf("vk: a readback in a live texture's storage: %#llx +%#llx", (unsigned long long)gpu, (unsigned long long)size);
        }
        unforced ? sync_settle() : wait_queue_empty();
        if (!texture_gpu_written_in(gpu, size)) return;
    }
    {   // (what the catch-ups for a texture cost the game's thread, every 300 calls)
        static std::atomic<int64_t> calls{0}, ns{0};
        const auto w0 = std::chrono::steady_clock::now();
        unforced ? sync_settle() : wait_queue_empty();
        ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count();
        if (++calls % 300 == 0)
            logf("vk: invalidate %#llx +%#llx: textures there; the catch-ups waited %.2f ms each (300 calls)",
                 (unsigned long long)gpu, (unsigned long long)size, ns.exchange(0) / 300.0 / 1e6);
    }
    std::lock_guard lk(replay_mu);
    texture_flush_to_guest(gpu, size);
}
/// (hookshot) invalidate with the game's virtual core kept (host/rx_backend.c rx_nvn_invalidate, nvn_invalidate_quick): 1 when
/// invalidate would find nothing in [gpu, gpu + size) to wait for or to write back, done here as it would be (the CPU's read
/// noted for the write-backs left out, memory_note_cpu_read: the one thing it does then); 0 when it may find something, and
/// the host calls it with the core given up, as before. Read without a lock, from marks set and never taken off
/// (page_marks.h): no page there ever written back (memory_writeback_ever: invalidate's `ever`, none of its waits), no texture
/// the GPU wrote ever listed there (texture_gpu_written_ever: texture_gpu_written_in false, no texels to write back) and no
/// live texture's storage (tex_pages_any, as invalidate reads it: no catch-up, behind or not). OoT's model calc invalidates
/// each shape's 64-byte occlusion-query slot, per view, from the stage slot jobs: the counts are written there on the CPU
/// (REPORT_COUNTER and occ_collect, guest_cpu_write), never written back, and each call gave up the core, let another thread
/// on it and queued to get one back, for nothing to wait for. On the console the call is a cache operation and the thread
/// runs on. It takes no lock but memory_note_cpu_read's (wb_learn_mu, a leaf, only with RX_VK_WB_MAX_KB) and never waits,
/// so holding the core here cannot keep another guest thread from what it waits for. Not with invalidate's diagnostics on
/// (RX_VK_LOG_INVALIDATE, RX_VK_LOG_READBACK); ranges past 512 GB (virtual pools, not followed) always go the slow way.
/// RX_VK_INVALIDATE_QUICK=0: never, as before
int invalidate_quick(uint64_t gpu, uint64_t size) {
    static const bool on = [] {
        const char* q = getenv("RX_VK_INVALIDATE_QUICK");
        const char* rb = getenv("RX_VK_LOG_READBACK");
        const bool v = (!q || !env_value_off(q)) && !env_on("RX_VK_LOG_INVALIDATE") && !(rb && atoll(rb) >= 0);
        logf("vk: invalidates with nothing to wait for keep their virtual core: %s",
             v ? "on" : "off (RX_VK_INVALIDATE_QUICK=0 or invalidate's logs)");
        return v;
    }();
    constexpr uint64_t kFollowed = 1ull << 39;   // (tex_pages' and the marks' window: GPU addresses below 512 GB)
    const bool kept = on && (!size || (gpu < kFollowed && size <= kFollowed - gpu && !tex_pages_any(gpu, size) &&
                                       !texture_gpu_written_ever(gpu, size) &&
                                       !memory_writeback_ever(host->gpu_guest(gpu), size)));
    if (kept) memory_note_cpu_read(host->gpu_guest(gpu), size);
    inv_count(kept);
    return kept;
}
void do_texture_released(const nvn_tex* t) {
    std::lock_guard lk(replay_mu);
    if (t) {
        mkwvk::texture_released(t->serial);   // (vk_bank.cpp: the registrations of it)
        mkwvk::texture_released(t);           // (vk_texture.cpp: its image may take CPU writes again)
    }
}

// ---- presents ----
struct Frame {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence done = VK_NULL_HANDLE;
};
// (Hookshot, RX_VK_ACQUIRE_FENCE, gone: the acquired image is always waited for on the CPU now, image_free below, so the
// queue never stops at the blit's submit; with the game's thread let run ahead of the GPU that wait was 30-50 ms a frame
// on PC 3050, every later submission behind it)
// the blit's own GPU time and the queue's gaps around it: a timestamp at the blit's start and end, read 3 presents
// later (the slot's fence waited for). With the recordings' absolute timestamps (gpu_time_collect) the gap before the blit
// (the last recording's end to the blit's start) and after it (to the next recording's start) tell where the queue stops
VkQueryPool blit_ts_pool = VK_NULL_HANDLE;
bool blit_ts_pending[3];
double st_blit_ms, st_blit_before_ms, st_blit_after_ms, st_blit_before_max_ms, st_blit_after_max_ms;
int64_t st_blit_n;
uint64_t blit_last_end;
Frame frames[kPresentSlotsMax];   // (present_image uses present_slots_for of them)
uint32_t frame_index = 0;
int64_t presents = 0;

// offscreen (no window): the picture, and the readback of a screenshot
VkImage off_image = VK_NULL_HANDLE;
VmaAllocation off_alloc = nullptr;
// (two readbacks, by the present's parity: a shot is read at a later present, after that one's own copy was submitted,
// and with one buffer the CPU read it while the GPU wrote the next frame over it: torn shots, and flashes counted)
VkBuffer shot_bufs[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
VmaAllocation shot_allocs[2] = {nullptr, nullptr};
void* shot_maps[2] = {nullptr, nullptr};
// (RX_VK_SHOT_W / RX_VK_SHOT_H: another size, e.g. the game's output size for its own pixels)
uint32_t shot_dimension(const char* text, uint32_t fallback) {
    if (!text) return fallback;
    if (!*text) return 0;
    uint32_t value = 0;
    for (; *text; text++) {
        if (*text < '0' || *text > '9') return 0;
        const uint32_t digit = (uint32_t)(*text - '0');
        if (value > (UINT32_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    return value;
}
const uint32_t kShotW = shot_dimension(getenv("RX_VK_SHOT_W"), 1280),
               kShotH = shot_dimension(getenv("RX_VK_SHOT_H"), 720);
VkImage shot_image = VK_NULL_HANDLE;
VkImageView shot_view = VK_NULL_HANDLE;   // (hookshot: RX_VK_SHOT_MENU's menu draws through it)
VkBuffer made_buf = VK_NULL_HANDLE;   // (RX_VK_SHOT_MADE: a made frame's readback)
VmaAllocation made_buf_alloc = nullptr;
void* made_map = nullptr;
VkImage made_image = VK_NULL_HANDLE;
VmaAllocation made_image_alloc = nullptr;
VmaAllocation shot_image_alloc = nullptr;

bool shot_dimensions_ready() {
    static int state = 0;
    if (state) return state > 0;
    state = -1;
    const uint32_t limit = dev.props.limits.maxImageDimension2D;
    const uint64_t pixels = (uint64_t)kShotW * kShotH, row = (uint64_t)kShotW * 3 + 1;
    bool valid = false;
    if (kShotW && kShotH && kShotW <= limit && kShotH <= limit && pixels <= SIZE_MAX / 4) {
        const uint64_t raw = row * kShotH;
        // write_png emits one IDAT: zlib header, stored blocks (5 bytes per65535), and Adler32.
        // PNG chunks are limited to2^31-1 bytes. This is an encoder limit, not a memory budget.
        constexpr uint64_t max_chunk = 0x7fffffffu;
        if (raw <= max_chunk) {
            const uint64_t encoded = 2 + raw + 5 * ((raw + 65534) / 65535) + 4;
            valid = encoded <= max_chunk && encoded <= SIZE_MAX - 4;
        }
    }
    if (!valid) {
        logf("vk: screenshots disabled: RX_VK_SHOT_W/H must be positive decimal dimensions within the device limit "
             "(%u) and the PNG/readback size limits (got %ux%u)", limit, kShotW, kShotH);
        return false;
    }
    state = 1;
    return true;
}

bool made_shot_init() {
    if (made_image) return true;
    VkImage next_image = VK_NULL_HANDLE;
    VmaAllocation next_alloc = nullptr, next_buf_alloc = nullptr;
    VkBuffer next_buf = VK_NULL_HANDLE;
    void* next_map = nullptr;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {kShotW, kShotH, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (!create_image(ci, next_image, next_alloc)) return false;
    if (!create_buffer((uint64_t)kShotW * kShotH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                       next_buf, next_buf_alloc, &next_map)) {
        destroy_image(next_image, next_alloc);
        return false;
    }
    made_image = next_image, made_image_alloc = next_alloc;
    made_buf = next_buf, made_buf_alloc = next_buf_alloc, made_map = next_map;
    return true;
}

bool offscreen_init() {
    if (!shot_dimensions_ready()) return false;
    if (off_image) return true;
    VkImage next_off = VK_NULL_HANDLE, next_shot = VK_NULL_HANDLE;
    VkImageView next_view = VK_NULL_HANDLE;
    VmaAllocation next_off_alloc = nullptr, next_shot_alloc = nullptr, next_allocs[2]{};
    VkBuffer next_bufs[2]{};
    void* next_maps[2]{};
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {dev.swap_extent.width, dev.swap_extent.height, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    bool ok = create_image(ci, next_off, next_off_alloc);
    ci.extent = {kShotW, kShotH, 1};
    ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;   // (hookshot: RX_VK_SHOT_MENU draws the menu over it, vk_menu.cpp)
    ok = ok && create_image(ci, next_shot, next_shot_alloc);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = next_shot;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ci.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ok = ok && vkCreateImageView(dev.device, &vi, nullptr, &next_view) == VK_SUCCESS;
    for (int i = 0; i < 2; i++)
        ok = ok && create_buffer((uint64_t)kShotW * kShotH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                                 next_bufs[i], next_allocs[i], &next_maps[i]);
    if (!ok) {
        for (int i = 0; i < 2; i++)
            if (next_bufs[i]) destroy_buffer(next_bufs[i], next_allocs[i]);
        if (next_view) vkDestroyImageView(dev.device, next_view, nullptr);
        if (next_shot) destroy_image(next_shot, next_shot_alloc);
        if (next_off) destroy_image(next_off, next_off_alloc);
        return false;
    }
    // Publish the ready marker only after both images, the menu view and both mapped readbacks exist.
    off_image = next_off, off_alloc = next_off_alloc;
    shot_image = next_shot, shot_image_alloc = next_shot_alloc;
    shot_view = next_view;
    for (int i = 0; i < 2; i++)
        shot_bufs[i] = next_bufs[i], shot_allocs[i] = next_allocs[i], shot_maps[i] = next_maps[i];
    return true;
}

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &d);
}

// A completed capture copy must also be made available in the host memory domain.
void capture_host_read_barrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize bytes) {
    VkBufferMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    b.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buffer;
    b.size = bytes;
    VkDependencyInfo d{};
    d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d.bufferMemoryBarrierCount = 1;
    d.pBufferMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &d);
}

/// The crop of src (x, y, w, h; w 0: all of it) scaled into a rectangle of dst.
/// An empty source clears the whole destination image; current callers give that fallback a full-image rectangle.
void blit_crop_to_rect(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, const int crop[4], VkImage dst,
                       VkImageLayout dst_layout, PresentRect rect) {
    const PresentSourceRect source = present_source_rect(sw, sh, crop);
    if (!rect.w || !rect.h) return;
    if (!source.w || !source.h) {
        const VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, dst, dst_layout, &black, 1, &range);
        return;
    }
    VkImageBlit b{};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[0] = {(int)source.x, (int)source.y, 0};
    b.srcOffsets[1] = {(int)(source.x + source.w), (int)(source.y + source.h), 1};
    b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstOffsets[0] = {(int)rect.x, (int)rect.y, 0};
    b.dstOffsets[1] = {(int)(rect.x + rect.w), (int)(rect.y + rect.h), 1};
    vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, dst_layout, 1, &b, VK_FILTER_LINEAR);
}

/// The crop of src scaled to the whole destination (captures keep their existing dimensions).
void blit_crop(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, const int crop[4], VkImage dst,
               VkImageLayout dst_layout, uint32_t dw, uint32_t dh) {
    blit_crop_to_rect(cmd, src, sw, sh, crop, dst, dst_layout, {0, 0, dw, dh});
}

// ---- PNG (stored deflate blocks: no compression library needed) ----
uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> out{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t k = i;
            for (int j = 0; j < 8; j++) k = k & 1 ? 0xEDB88320u ^ (k >> 1) : k >> 1;
            out[i] = k;
        }
        return out;
    }();
    for (size_t i = 0; i < n; i++) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c;
}
} // namespace
/// Whether the storage of a texture the game made reaches [gpu, gpu + size) (tex_pages_any: true where it is not
/// tracked), for vk_memory.cpp's late pages left to their reads (RX_VK_LATE_DEFER)
bool texture_storage_any(uint64_t gpu, uint64_t size) { return tex_pages_any(gpu, size); }
/// Whether the replay runs on the render thread (async_on: RX_VK_ASYNC, or the Threaded rendering row), for
/// vk_memory.cpp's RX_VK_READBACK_EARLY, which is left off with it
bool replay_render_thread_on() { return async_on(); }

// (vk_internal.h) One part of a submission done on the GPU: the late pages' fix reads a page of the capture whose fix an earlier
// part of this submission recorded (vk_memory.cpp, late_fix_cpu_read), after that part's fix has run
int vk_part_wait(uint64_t part, uint64_t timeout_ns) {
#ifdef TRACY_ENABLE
    HT_WAIT(ht_w, "vk_part_wait");
#endif
    int ok = 0;
    if (!wait_recorded_part(part, timeout_ns, &ok)) return -1;
    return ok ? 1 : 0;
}

// Called while replaying, including workers joined before the next signal is issued.
// The current logical value may already have completed before a stopped queue resumes.
// A new signal instead follows every recording using the images being retired, including
// batched work and the unsignalled post-present interpolation recording. Zero retains forever.
uint64_t vk_retirement_value() {
    return timeline && signaled != UINT64_MAX ? signaled + 1 : 0;
}

/// A submission or a wait came back VK_ERROR_DEVICE_LOST. Nothing sent from here on runs, and the replay kept going: no
/// submission completed again, so no capture or staging arena was let go, and system memory filled until the window was
/// closed (issue 1275: captures kept 25 -> 6527, staging 256 -> 1408 MB). Now the report once (a thread that sees the loss
/// too waits for it, instead of ending the process before it is written), then the process ends as replay_fatal does
[[noreturn]] void device_lost(const char* why) {
    static std::once_flag once;
    std::call_once(once, checkpoints_report);
    // (the presents on a queue of their own: once an RTX 4090 on NVIDIA's 621.14 had a black screen with them: the way back
    // named where a player reads it. A local array: device_lost can run on several threads at once, and replay_fatal uses the
    // text before it ends the process)
    char lost[400];
    std::snprintf(lost, sizeof lost, "%s%s",
                  why ? why
                      : "the GPU device was lost (VK_ERROR_DEVICE_LOST): the driver stopped or reset the GPU, and nothing sent "
                        "to it runs any more (on Linux, dmesg or journalctl -k around this time names the reset)",
                  dev.present_queue_supported ? ". The presents were on a queue of their own: RX_VK_PRESENT_QUEUE=0 in "
                                                "hookshot_env.txt turns that off; please report your GPU and driver" : "");
    replay_fatal(lost);
}

/// (hookshot) A GPU hang the driver never reports as a lost device. RTX 4070 Ti, NVIDIA 595.99.02, Linux, Hyrule Castle: the
/// kernel's Xid 109 (CTX SWITCH TIMEOUT) ended the game's GPU context, the timeline stopped 4 short of what was sent, and
/// from then on every wait came back VK_TIMEOUT or never came back, none VK_ERROR_DEVICE_LOST: device_lost() did not run,
/// the game froze, and the log had no report of what the GPU was running. A thread reads the timeline once a second from the first present: a lost device
/// it sees goes to device_lost(), and work sent to the queue that makes no progress at all for RX_VK_GPU_HANG_SEC
/// seconds (default 30) is taken as a hang and goes there too. A window the system stops presenting can hold the queue
/// that long as well (the present waits for its image): RX_VK_GPU_HANG_SEC=0 logs the stall and lets the game wait.
void gpu_hang_watch_start() {
    static std::atomic<bool> started{false};
    if (!timeline || started.exchange(true)) return;
    const char* e = getenv("RX_VK_GPU_HANG_SEC");
    const int limit = e ? std::max(0, atoi(e)) : 30;
    // (atoi reads a typo as 0, which turns the exit off: the value it settled on is logged)
    if (limit) logf("vk: GPU hang watch: work that finishes nothing for %d s ends the game (RX_VK_GPU_HANG_SEC)", limit);
    else logf("vk: GPU hang watch: a stall is only logged (RX_VK_GPU_HANG_SEC=%s)", e);
    std::thread([limit] {
        SetThreadDescription(GetCurrentThread(), L"mkw_vk gpu hang watch");
        uint64_t last = 0;
        int still = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            uint64_t done = 0;
            if (vkGetSemaphoreCounterValue(dev.device, timeline, &done) == VK_ERROR_DEVICE_LOST) device_lost();
            const uint64_t sent = timeline_sent.load(std::memory_order_relaxed);
            if (done >= sent || done != last) {
                last = done, still = 0;
                continue;
            }
            still++;
            if (still == 10 || still % 30 == 0)
                logf("vk: the GPU has finished nothing for %d s: the timeline at %llu, %llu sent (RX_VK_GPU_HANG_SEC=%d)",
                     still, (unsigned long long)done, (unsigned long long)sent, limit);
            if (limit && still >= limit) {
                static char why[256];
                std::snprintf(why, sizeof why, "the GPU hung: no submission finished in %d s, and the driver never reported the "
                              "device lost (on Linux, dmesg or journalctl -k around this time names the fault, an Xid on NVIDIA)", still);
                logf("vk: GPU hang: the timeline at %llu, %llu sent", (unsigned long long)done, (unsigned long long)sent);
                device_lost(why);
            }
        }
    }).detach();
}

bool write_png(const char* path, const uint8_t* rgba, uint32_t w, uint32_t h) {
    auto failed = [path](const char* phase, unsigned error = 0) {
        static std::atomic<unsigned> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 8)
            logf("vk: PNG %s failed (error %u): %s", phase, error, path ? path : "(null)");
        return false;
    };
    if (!path || !*path || !rgba || !w || !h || w > 0x7fffffffu || h > 0x7fffffffu)
        return failed("dimensions/path");
    const uint64_t pixels = (uint64_t)w * h;
    if (pixels > SIZE_MAX / 4) return failed("size");
    const uint64_t raw_size = ((uint64_t)w * 3 + 1) * h;
    constexpr uint64_t max_chunk = 0x7fffffffu;
    if (raw_size > max_chunk || 6 + raw_size + 5 * ((raw_size + 65534) / 65535) > max_chunk)
        return failed("size");
    try {
        std::vector<uint8_t> raw;
        raw.reserve((size_t)(w * 3 + 1) * h);
        for (uint32_t y = 0; y < h; y++) {
            raw.push_back(0);
            for (uint32_t x = 0; x < w; x++) {
                const uint8_t* q = rgba + ((size_t)y * w + x) * 4;
                raw.insert(raw.end(), q, q + 3);
            }
        }
        std::vector<uint8_t> z = {0x78, 0x01};
        uint32_t a = 1, b = 0;
        for (uint8_t c : raw) a = (a + c) % 65521, b = (b + a) % 65521;
        for (size_t at = 0; at < raw.size();) {
            const size_t n = std::min<size_t>(65535, raw.size() - at);
            z.push_back(at + n == raw.size() ? 1 : 0);
            z.push_back((uint8_t)n), z.push_back((uint8_t)(n >> 8));
            z.push_back((uint8_t)~n), z.push_back((uint8_t)(~n >> 8));
            z.insert(z.end(), raw.begin() + (long)at, raw.begin() + (long)(at + n));
            at += n;
        }
        const uint32_t adler = b << 16 | a;
        for (int i = 3; i >= 0; i--) z.push_back((uint8_t)(adler >> (i * 8)));

        // Encode before opening a file; only a completed private temporary replaces the destination.
        static std::atomic<uint64_t> next{0};
        std::string temporary;
        FILE* f = nullptr;
        for (int attempt = 0; attempt < 128; attempt++) {
            temporary = std::string(path) + ".part." + std::to_string(next.fetch_add(1, std::memory_order_relaxed));
            f = fopen(temporary.c_str(), "wbx");
            if (f || errno != EEXIST) break;
        }
        if (!f) return failed("open", errno);
        bool ok = true;
        unsigned error = 0;
        const char* phase = "write";
        auto put = [&](const void* bytes, size_t n) {
            if (ok && n && fwrite(bytes, 1, n, f) != n) {
                error = errno ? errno : EIO;
                ok = false;
            }
        };
        auto be32 = [&](uint32_t v) {
            const uint8_t q[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v};
            put(q, sizeof q);
        };
        auto chunk = [&](const char* type, const uint8_t* d, size_t n) {
            be32((uint32_t)n);
            put(type, 4);
            put(d, n);
            be32(crc32(d, n, crc32((const uint8_t*)type, 4)) ^ 0xFFFFFFFFu);
        };
        static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
        put(sig, sizeof sig);
        const uint8_t ihdr[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                                 (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 2, 0, 0, 0};
        chunk("IHDR", ihdr, sizeof ihdr);
        chunk("IDAT", z.data(), z.size());
        chunk("IEND", nullptr, 0);
        if (fclose(f) != 0) {
            if (ok) phase = "close", error = errno ? errno : EIO;
            ok = false;
        }
        if (ok) {
            phase = "publish";
#ifdef _WIN32
            ok = MoveFileExA(temporary.c_str(), path, MOVEFILE_REPLACE_EXISTING) != 0;
            if (!ok) error = GetLastError();
#else
            ok = rename(temporary.c_str(), path) == 0;
            if (!ok) error = errno;
#endif
        }
        if (!ok) {
            if (remove(temporary.c_str()) != 0) failed("temporary cleanup", errno);
            return failed(phase, error);
        }
        return true;
    } catch (const std::exception&) {
        return failed("encoding allocation");
    }
}


// (do_present: the game's picture is already HDR10, Native HDR taken. Read by present_image, and by vk_dlss.cpp, whose
// feature is made for HDR colour while it is)
std::atomic<bool> present_native_pq{false};

namespace {

int64_t fps_n = 0;
std::chrono::steady_clock::time_point fps_t = std::chrono::steady_clock::now();

/// RX_VK_SLOW_LOG=<ms>: a frame that long after the last: its replay's times, for what the stutters are
void log_slow_frame() {
    static const int limit = getenv("RX_VK_SLOW_LOG") ? atoi(getenv("RX_VK_SLOW_LOG")) : 0;
    if (!limit) return;
    static auto last_t = std::chrono::steady_clock::now();
    static int64_t last[PROF_COUNT];
    static int64_t last_pl, last_tr, last_d, last_up;
    static int logged;
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - last_t).count();
    last_t = now;
    int64_t d = 0, sk = 0, di = 0, pl = 0, tr = 0, fl = 0, up = 0, astc = 0, upb = 0;
    draw_stats(&d, &sk, &di, &pl);
    shader_stats(&tr, &fl);
    texture_stats(&up, &astc, &upb);
    if (ms >= limit && d - last_d >= 500 && logged < 800) {
        logged++;
        const auto dp = [](int i) { return (prof_ns[i] - last[i]) / 1e6; };
        logf("vk: slow frame at present %lld: %.1f ms; submit %.1f, draw %.1f, descriptors %.1f, pipelines %.1f, programs %.1f, "
             "commands %.1f, memory sync %.1f, queue submit %.1f, slot wait %.1f; %lld draws, %lld new pipelines, %lld new programs, "
             "%lld uploads", (long long)presents, ms, dp(PROF_SUBMIT), dp(PROF_DRAW), dp(PROF_DESC), dp(PROF_PIPE), dp(PROF_PROGRAM),
             dp(PROF_CMDS), dp(PROF_MEMSYNC), dp(PROF_QSUBMIT), dp(PROF_SLOTWAIT), (long long)(d - last_d), (long long)(pl - last_pl),
             (long long)(tr - last_tr), (long long)(up - last_up));
    }
    for (int i = 0; i < PROF_COUNT; i++) last[i] = prof_ns[i];
    last_pl = pl, last_tr = tr, last_d = d, last_up = up;
}

uint64_t made_ring[32];
int64_t made_ring_present = 0;
void log_frame_stats() {
    log_slow_frame();
    made_ring[presents % 32] = texture_made();
    made_ring_present = presents;
    if (presents % 600 == 0) programs_census_log("play");
    if (++fps_n == 120) {
        const auto now = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(now - fps_t).count();
        int64_t d = 0, sk = 0, di = 0, pl = 0, tr = 0, fl = 0, passes = 0, mesh = 0, up = 0, astc = 0, upb = 0;
        int64_t st_clears = 0, masked = 0;
        clear_stats(&st_clears, &masked);
        double pms = 0;
        draw_stats(&d, &sk, &di, &pl);
        draw_stats2(&passes, &mesh, &pms);
        shader_stats(&tr, &fl);
        texture_stats(&up, &astc, &upb);
        int64_t pskip = 0, pwaits = 0;
        double pwait_ms = 0;
        draw_stats3(&pskip, &pwaits, &pwait_ms);
        static int64_t l_pskip, l_pwaits;
        static double l_pwait_ms;
        static int64_t l_reused, l_saves;
        int64_t links = 0, lib_hits = 0, lib_cold = 0, lib_built = 0;
        double link_ms = 0, lib_build_ms = 0;
        static int64_t l_links;
        static double l_link_ms;
        char libs[256] = "";   // (the stage libraries' part of the line, with RX_VK_GPL only: other logs keep the line they had)
        if (draw_stats4(&links, &link_ms, &lib_hits, &lib_cold, &lib_built, &lib_build_ms))
            snprintf(libs, sizeof libs, "; %.2f library links tried (%.2f ms), %lld libraries the driver had and %lld cold in all; "
                     "%lld of the cold built by workers (%.0f ms)", (links - l_links) / 120.0, (link_ms - l_link_ms) / 120.0,
                     (long long)lib_hits, (long long)lib_cold, (long long)lib_built, lib_build_ms);
        {
            static int64_t p0, b0, u0, ss0, t0, sa0, i0;
            logf("vk: a frame: %lld bind_gen moves: program %.1f, many-binds %.1f, ubo %.1f, ssbo %.1f, tex %.1f, samp %.1f, img %.1f",
                 (long long)((st_bg_prog - p0) + (st_bg_bufs - b0) + (st_bg_ubo - u0) + (st_bg_ssbo - ss0) + (st_bg_tex - t0) +
                             (st_bg_samp - sa0) + (st_bg_img - i0)),
                 (st_bg_prog - p0) / 120.0, (st_bg_bufs - b0) / 120.0, (st_bg_ubo - u0) / 120.0, (st_bg_ssbo - ss0) / 120.0,
                 (st_bg_tex - t0) / 120.0, (st_bg_samp - sa0) / 120.0, (st_bg_img - i0) / 120.0);
            p0 = st_bg_prog, b0 = st_bg_bufs, u0 = st_bg_ubo, ss0 = st_bg_ssbo, t0 = st_bg_tex, sa0 = st_bg_samp, i0 = st_bg_img;
        }
        logf("vk: a frame: %.0f draws kept their neighbour's descriptors (%.0f of them through other slots' registrations)",
             (draw_desc_reused() - l_reused) / 120.0, (draw_slotgen_saves() - l_saves) / 120.0);
        l_reused = draw_desc_reused(), l_saves = draw_slotgen_saves();
        programs_frames_note();   // (the load default's safety net: vk_programs.h)
        logf("vk: a frame: %.1f draws left out (not ready), %.2f waits for shaders or pipelines (%.1f ms); "
             "translation %.0f ms in all, %d jobs queued%s",
             (pskip - l_pskip) / 120.0, (pwaits - l_pwaits) / 120.0, (pwait_ms - l_pwait_ms) / 120.0,
             shader_translate_ms(), jobs_pending(), libs);
        l_pskip = pskip, l_pwaits = pwaits, l_pwait_ms = pwait_ms, l_links = links, l_link_ms = link_ms;
        {
            static const char* names[PROF_COUNT] = {"submit", "present", "draw", "descriptors", "texture lookups",
                                                    "uploads", "pipelines", "programs", "commands", "dispatch",
                                                    "memory sync", "queue submit", "slot wait"};
            static int64_t last[PROF_COUNT];
            char line[600];
            int n = snprintf(line, sizeof line, "vk: a frame's replay time (ms):");
            for (int i = 0; i < PROF_COUNT; i++) {
                n += snprintf(line + n, sizeof line - n, " %s %.2f%s", names[i], (prof_ns[i] - last[i]) / 120.0 / 1e6,
                              i + 1 < PROF_COUNT ? "," : "");
                last[i] = prof_ns[i];
            }
            logf("%s", line);
        }
        int64_t mirrors = 0, sync_b = 0, sync_c = 0, wb_b = 0;
        uint64_t mirror_mb = 0;
        memory_stats(&mirrors, &mirror_mb, &sync_b, &sync_c, &wb_b);
        static int64_t l_sync_b, l_sync_c, l_wb_b;
        logf("vk: a frame: mirrors %lld (%llu MB): %.2f MB synced in %.1f copies, %.2f MB written back",
             (long long)mirrors, (unsigned long long)mirror_mb, (sync_b - l_sync_b) / 120.0 / 1048576.0,
             (sync_c - l_sync_c) / 120.0, (wb_b - l_wb_b) / 120.0 / 1048576.0);
        l_sync_b = sync_b, l_sync_c = sync_c, l_wb_b = wb_b;
        static int64_t l_d, l_sk, l_di, l_pass, l_sub, l_up, l_upb, l_bar, l_ubu, l_ubo, l_wait;
        logf("vk: %lld presents, %.1f fps; a frame: %.0f draws (%.0f skipped), %.0f dispatches, %.0f passes, %.0f "
             "barriers, %.0f submits, %.1f uploads (%.1f MB), %.0f ub updates (%.0f overlays), GPU-bound wait %.1f ms (recording slots only); "
             "%lld pipelines (%.0f ms), %lld programs (%lld failed), %zu textures (%llu MB)",
             (long long)presents, 120.0 / secs, (d - l_d) / 120.0, (sk - l_sk) / 120.0, (di - l_di) / 120.0,
             (passes - l_pass) / 120.0, (st_barriers - l_bar) / 120.0, (st_submits - l_sub) / 120.0,
             (up - l_up) / 120.0, (upb - l_upb) / 120.0 / 1048576.0, (st_ub_updates - l_ubu) / 120.0,
             (st_ub_overlays - l_ubo) / 120.0, (st_wait_ns - l_wait) / 120.0 / 1e6, (long long)pl, pms,
             (long long)tr, (long long)fl, texture_count(), (unsigned long long)(texture_bytes() >> 20));
        {
            const auto waits = gpu_wait_stats::counters.take();
            const double scale = 1.0 / (120.0 * 1e6);
            logf("vk: host time in GPU resource waits: %.2f ms a frame (fences %.2f, timelines %.2f, idle %.2f, queries %.2f; "
                 "sum across threads, not frame time; excludes pacing and driver submit/present blocking)",
                 (waits[gpu_wait_stats::kFence] + waits[gpu_wait_stats::kTimeline] + waits[gpu_wait_stats::kIdle] +
                  waits[gpu_wait_stats::kQuery]) * scale,
                 waits[gpu_wait_stats::kFence] * scale, waits[gpu_wait_stats::kTimeline] * scale,
                 waits[gpu_wait_stats::kIdle] * scale, waits[gpu_wait_stats::kQuery] * scale);
        }
        draw_skip_report(120.0);
        {   // (hookshot: the full barriers recorded and the ones left out as the second of two in a row, RX_VK_BARRIER_DEDUPE_OFF=1)
            static int64_t l_made, l_left;
            int64_t made = 0, left = 0;
            barrier_stats(&made, &left);
            static int64_t l_relaxed;
            const int64_t relaxed = pass_barriers_relaxed();
            logf("vk: a frame: %.1f full barriers recorded, %.1f left out (nothing recorded since the one before), %.1f pass starts "
                 "ordered in their fragment stages only (RX_VK_PASS_BARRIER=game)", (made - l_made) / 120.0, (left - l_left) / 120.0,
                 (relaxed - l_relaxed) / 120.0);
            l_made = made, l_left = left, l_relaxed = relaxed;
            std::string fl;   // (NVN's barrier bits: ORDER_PRIMITIVES 1, ORDER_FRAGMENTS 2, ORDER_FRAGMENTS_TILED 4, ORDER_INDIRECT_DATA 8, INVALIDATE_TEXTURE 0x10, _SHADER 0x20, _TEXTURE_DESCRIPTOR 0x40, _ZCULL 0x80)
            std::lock_guard lk(st_barrier_flags_mu);
            for (const auto& [bits, n] : st_barrier_flags) {
                char b[48];
                snprintf(b, sizeof b, " %#x: %.1f;", (unsigned)bits, n / 120.0);
                fl += b;
            }
            logf("vk: a frame: the game's barriers by their NVN flags:%s", fl.c_str());
            st_barrier_flags.clear();
        }
        static int64_t l_gpu;
        static uint64_t l_made;
        logf("vk: GPU time %.1f ms a frame (the longest submission %.1f ms); %.1f textures made a frame",
             (st_gpu_ns - l_gpu) / 120.0 / 1e6, st_gpu_max_ns / 1e6, (texture_made() - l_made) / 120.0);
        {   // (issue 109: the movie networks too slow on this card: left out from now on, vk_draw.cpp movie_networks_slow)
            static const bool keep_all = getenv("RX_VK_NETWORK_KEEP") && !strcmp(getenv("RX_VK_NETWORK_KEEP"), "all");
            const double gpu_ms = (st_gpu_ns - l_gpu) / 120.0 / 1e6, draws = (d - l_d) / 120.0;
            menu_auto_tune(gpu_ms, draws, 120.0 / secs);   // (the Quality preset Auto (this PC): measured in play)
            if (!keep_all && gpu_ms > 40.0 && draws < 40.0 && !movie_networks_slow.exchange(true))
                logf("vk: %.0f ms of GPU time a frame for %.0f draws (a movie): its upscaler networks are left out from now on "
                     "(a linear upscale; RX_VK_NETWORK_KEEP=all keeps them)", gpu_ms, draws);
        }
        l_made = texture_made();
        {
            static int64_t l_ub, l_un;
            int64_t ub = 0, un = 0;
            texture_unswizzle_stats(&ub, &un);
            logf("vk: a frame: %.2f MB unswizzled on the CPU in %.2f ms", (ub - l_ub) / 120.0 / 1048576.0,
                 (un - l_un) / 120.0 / 1e6);
            l_ub = ub, l_un = un;
        }
        l_gpu = st_gpu_ns, st_gpu_max_ns = 0;
        gprof_report(120);
        l_d = d, l_sk = sk, l_di = di, l_pass = passes, l_sub = st_submits, l_up = up, l_upb = upb, l_bar = st_barriers;
        l_ubu = st_ub_updates, l_ubo = st_ub_overlays, l_wait = st_wait_ns;
        fps_t = now, fps_n = 0;
    }
}

/// (hookshot) Once a game frame: every 600 a status line (the video memory, DLSS's details while it draws), and every 60
/// a look at the driver's allowance, warned of once when the use passes it (again only after it was back under)
void status_tick() {
    namespace sl = status_line;
    {   // (hookshot, RX_VK_TEX_AHEAD: a light frame, one of few draws, lets the next one prepare textures)
        static int64_t l_draws = 0;
        int64_t draws = 0, sk = 0, di = 0, pl = 0;
        draw_stats(&draws, &sk, &di, &pl);
        texture_ahead_frame(draws - l_draws);
        temporal_note_frame(draws - l_draws);   // (and RX_VK_DLSS_PREWARM's)
        l_draws = draws;
    }
    static uint64_t frames = 0;
    // (the slowest frame since the last line, present to present: a stall shows here, to be matched with what the log says
    // around it, such as DLSS's feature made again)
    static int64_t last_ns = 0, slowest_ns = 0;
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (last_ns) slowest_ns = std::max(slowest_ns, now_ns - last_ns);
    // (RX_VK_FRAME_TRACE=1: every frame a line of hookshot_frames.csv beside the game, present to present and what the frame did,
    // so that each stutter can be matched to its cause: waits for shaders or pipelines, translation, uploads, the memory sync,
    // pipeline compiles, the GPU behind. RX_VK_FRAME_LOG has the frame times alone)
    static FILE* trace = [] {
        if (!getenv("RX_VK_FRAME_TRACE") || !atoi(getenv("RX_VK_FRAME_TRACE"))) return (FILE*)nullptr;
        // (the game folder's logs/: vk_paths.h; with HOS_PROFILE=1, frames.csv in that run's profile folder, which the host
        // chose: host/profiling.c)
        const char* dir = getenv("HOS_PROFILE_DIR");
        FILE* f = fopen(dir && *dir ? (std::string(dir) + "/frames.csv").c_str() : log_path("hookshot_frames.csv").c_str(), "w");
        if (f) fprintf(f, "frame,ms,pipe_waits,pipe_wait_ms,left_out,translate_ms,uploads,upload_mb,textures,memsync_ms,pipelines_ms,"
                          "uploads_ms,programs_ms,slot_wait_ms,gpu_behind_ms,draw_ms,desc_ms,cmds_ms,tex_lookup_ms,textures_gone,"
                          "late_wait_ms,readback_wait_ms,tex_frame_ms,game_ms,render_ms,gpu_wait_ms,pace_ms,wait_render_ms,"
                          "wait_gpu_ms,wait_main_ms,main_cpu_ms,main_blocked_ms,main_vcore_ms,at_ns,ab_side\n");
        return f;
    }();
    // seconds since the process started, for the bench's 66 s cut: the steady clock's offset from the creation time
    static const int64_t launch_offset_ns = [&] {
        if (!trace) return (int64_t)0;
        FILETIME c, e, k, u, now;
        if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return (int64_t)0;
        GetSystemTimePreciseAsFileTime(&now);
        const auto ft = [](const FILETIME& f) { return (int64_t)(((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime); };
        return (ft(now) - ft(c)) * 100 - now_ns;
    }();
#ifdef TRACY_ENABLE
    if (last_ns) HT_PLOT("frame ms (renderer)", (now_ns - last_ns) / 1e6);
#endif
    if (trace && last_ns) {
        static int64_t l_ps, l_pw, l_up, l_upb, l_wait, l_prof[PROF_COUNT];
        static double l_pwms, l_tr;
        int64_t ps = 0, pw = 0, up = 0, astc = 0, upb = 0;
        double pwms = 0;
        draw_stats3(&ps, &pw, &pwms);
        texture_stats(&up, &astc, &upb);
        const double tr = shader_translate_ms();
        const auto dp = [](int i) { return (prof_ns[i] - l_prof[i]) / 1e6; };
        static uint64_t l_gone;
        const uint64_t gone = texture_destroy_generation();
        // (hookshot: the waits no other column has: the capture's for the GPU (inside memsync_ms), a read-back's for the GPU, and
        // texture_frame as a whole (the present's eviction and sweeps; the read-back's wait is inside it when an eviction asked))
        static int64_t l_late, l_rb, l_tf;
        const int64_t late = memory_late_wait_ns(), rb = texture_readback_wait_ns(), tf = texture_frame_ns();
        // (hookshot) the game's frame as the FPS counter's Breakdown splits it (vk_fps.cpp: measured while the trace is on; the
        // presenting thread's frame, which with the render thread on is the game's present before this one), and the game's main
        // thread's own time over the trace's frame (MkwVkHost main_thread_times): on the CPU, blocked in waits and sleeps, and
        // waiting for a virtual core. Its time running but off the CPU (the system's other work) is the frame less the three
        const fps_stats::Frame ff = fps_last_frame();
        static int64_t l_mt[3];
        int64_t mt[3] = {0, 0, 0};
        if (host->main_thread_times) host->main_thread_times(mt);
        // (at_ns: this present on the host's clock, hos_now_ns: the clock of the profile's samples, presents.csv and events.txt,
        // so a frame is found in them. ab_side (hookshot): the frame's side in an A/B within the run, HOS_AB (host/hos_ab.c):
        // -1 none, 0 or 1, 2 or 3 in a transition; scripts/ab_summary.py)
        static int64_t l_at;
        const int64_t at = host->now_ns ? host->now_ns() : 0;
        const int ab_side = host->version >= 35 && host->ab_frame ? host->ab_frame(l_at, at) : -1;
        l_at = at;
        fprintf(trace, "%llu,%.2f,%lld,%.2f,%lld,%.1f,%lld,%.2f,%zu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%llu,%.2f,%.2f,%.2f,"
                       "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%lld,%d\n",
                (unsigned long long)frames, (now_ns - last_ns) / 1e6, (long long)(pw - l_pw), pwms - l_pwms, (long long)(ps - l_ps),
                tr - l_tr, (long long)(up - l_up), (upb - l_upb) / 1048576.0, texture_count(), dp(PROF_MEMSYNC), dp(PROF_PIPE),
                dp(PROF_UPLOAD), dp(PROF_PROGRAM), dp(PROF_SLOTWAIT), (st_wait_ns - l_wait) / 1e6, dp(PROF_DRAW), dp(PROF_DESC),
                dp(PROF_CMDS), dp(PROF_TEXGET), (unsigned long long)(gone - l_gone), (late - l_late) / 1e6, (rb - l_rb) / 1e6,
                (tf - l_tf) / 1e6, ff.part[fps_stats::kGame], ff.part[fps_stats::kRender], ff.part[fps_stats::kGpuWait],
                ff.part[fps_stats::kPace], ff.part[fps_stats::kWaitRender], ff.part[fps_stats::kWaitGpu], ff.part[fps_stats::kWaitMain],
                (mt[2] - l_mt[2]) / 1e6, (mt[1] - l_mt[1]) / 1e6, (mt[0] - l_mt[0]) / 1e6,
                // (at_ns: this present on the host's clock, hos_now_ns: the clock of the profile's samples, presents.csv and
                // events.txt, so a frame is found in them)
                (long long)(host->now_ns ? host->now_ns() : 0), ab_side);
#ifdef TRACY_ENABLE
        // (HOOKSHOT_TRACY: the trace's main values as plots, beside the zones)
        HT_PLOT("game_ms", ff.part[fps_stats::kGame]);
        HT_PLOT("render_ms", ff.part[fps_stats::kRender]);
        HT_PLOT("wait_gpu_ms", ff.part[fps_stats::kWaitGpu]);
        HT_PLOT("wait_render_ms", ff.part[fps_stats::kWaitRender]);
        HT_PLOT("wait_main_ms", ff.part[fps_stats::kWaitMain]);
        HT_PLOT("memsync_ms", dp(PROF_MEMSYNC));
#endif
        l_gone = gone, l_late = late, l_rb = rb, l_tf = tf;
        std::memcpy(l_mt, mt, sizeof l_mt);
        if (frames % 60 == 0) fflush(trace);
        l_ps = ps, l_pw = pw, l_pwms = pwms, l_up = up, l_upb = upb, l_tr = tr, l_wait = st_wait_ns;
        for (int i = 0; i < PROF_COUNT; i++) l_prof[i] = prof_ns[i];
    }
    last_ns = now_ns;
    if (++frames % sl::kCheckEvery) return;
    uint64_t used = 0, allowed = 0;
    const bool known = vram_usage_now(&used, &allowed);
    static bool told_unknown = false;
    if (!known && !told_unknown) {
        told_unknown = true;
        logf("vk: status: the driver gives no video-memory budget (VK_EXT_memory_budget): only the game's own allocations are known");
    }
    char dlss[320] = "";
    const bool line = frames % sl::kEvery == 0;
    const int mode = line ? temporal_status(dlss, sizeof dlss) : temporal_mode();   // (temporal_status: since the last line, so only for one)
    static bool over = false;
    if (known && sl::crossed(over, used, allowed)) {
        char w[200];
        sl::over_text(w, sizeof w, used, allowed, mode);
        logf("%s", w);
    }
    if (!line) return;
    char v[128];
    sl::vram_text(v, sizeof v, used, allowed, known);
    logf("vk: status: %d frames, the slowest %.1f ms; %s%s", sl::kEvery, slowest_ns / 1e6, v, dlss);
    slowest_ns = 0;
}

// (RX_VK_FRAME_LOG=<file>: one CSV line a present: frame, seconds since the first, frame time in ms. Frame pacing on the
// Steam Deck, where MangoHud's log stays empty under gamescope. Taken once the present is handed to the swapchain
// (present_out), or without a window (HOS_HEADLESS) once the picture's offscreen copy is submitted: a headless run wrote none)
void frame_log_present() {
    static FILE* fl = getenv("RX_VK_FRAME_LOG") ? std::fopen(getenv("RX_VK_FRAME_LOG"), "w") : nullptr;
    static std::chrono::steady_clock::time_point first, prev;
    static uint64_t frames;
    if (!fl) return;
    const auto t = std::chrono::steady_clock::now();
    // (and the QueryPerformanceCounter value of the present, PresentMon's TimeInQPC clock and HOS_LOG_QPC's: a long frame then
    // falls next to the log lines around it; 0 where there is no such counter)
    long long qpc = 0;
#ifdef _WIN32
    LARGE_INTEGER q;
    if (QueryPerformanceCounter(&q)) qpc = (long long)q.QuadPart;
#endif
    if (!frames) first = prev = t, std::fputs("frame,time_s,frametime_ms,qpc\n", fl);
    std::fprintf(fl, "%llu,%.4f,%.3f,%lld\n", (unsigned long long)frames, std::chrono::duration<double>(t - first).count(),
                 std::chrono::duration<double, std::milli>(t - prev).count(), qpc);
    prev = t;
    if (++frames % 60 == 0) std::fflush(fl);
}

// ---- (hookshot) the control bridge's shots (host/control.c, docs/control.md): a PNG of the first present after a given one,
// asked for from another thread. Read back as RX_VK_SHOTS' are (at a later present, once the GPU reached it, and written on a
// thread of its own), with an image and a buffer of their own: the two kinds never share a readback. Never waited for: a
// present looks at the readback once its value is completed() and moves on otherwise ----
struct ShotAsk {
    std::string path;
    int64_t after = -1;
    void (*done)(void* ctx, int64_t present, int ok) = nullptr;
    void* ctx = nullptr;
};
std::mutex shot_ask_mu;
std::deque<ShotAsk> shot_asks;   // (shot_ask_mu)
struct { bool on; uint64_t value; int64_t present; ShotAsk ask; } shot_asked{};   // (replay_mu: the one being read back)
VkImage ask_image = VK_NULL_HANDLE;
VmaAllocation ask_image_alloc = nullptr, ask_buf_alloc = nullptr;
VkBuffer ask_buf = VK_NULL_HANDLE;
void* ask_map = nullptr;

int shot_ask(const char* path, int64_t after, void (*done)(void*, int64_t, int), void* ctx) {
    if (!path || !*path || !done) return 0;
    std::lock_guard lk(shot_ask_mu);
    if (shot_asks.size() >= 16) return 0;
    shot_asks.push_back({path, after, done, ctx});
    return 1;
}
bool shot_ask_init() {
    if (ask_image) return true;
    if (!shot_dimensions_ready()) return false;   // (RX_VK_SHOT_W/H out of bounds: no shots of either kind)
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation image_alloc = nullptr, buf_alloc = nullptr;
    VkBuffer buf = VK_NULL_HANDLE;
    void* map = nullptr;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {kShotW, kShotH, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (!create_image(ci, image, image_alloc)) return false;
    if (!create_buffer((uint64_t)kShotW * kShotH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, buf, buf_alloc, &map)) {
        destroy_image(image, image_alloc);
        return false;
    }
    ask_image = image, ask_image_alloc = image_alloc, ask_buf = buf, ask_buf_alloc = buf_alloc, ask_map = map;
    return true;
}
/// (do_present, replay_mu) the copy of this present for the first shot asked for before it, unless one is being read back
void shot_ask_record(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, const int crop[4], uint64_t value) {
    if (shot_asked.on || !src) return;
    ShotAsk a;
    {
        std::lock_guard lk(shot_ask_mu);
        if (shot_asks.empty() || shot_asks.front().after >= presents) return;
        a = std::move(shot_asks.front());
        shot_asks.pop_front();
    }
    if (!shot_ask_init()) {
        a.done(a.ctx, presents, 0);
        return;
    }
    image_barrier(cmd, ask_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    blit_crop(cmd, src, sw, sh, crop, ask_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kShotW, kShotH);
    image_barrier(cmd, ask_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {kShotW, kShotH, 1}};
    vkCmdCopyImageToBuffer(cmd, ask_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ask_buf, 1, &c);
    capture_host_read_barrier(cmd, ask_buf, (VkDeviceSize)kShotW * kShotH * 4);
    shot_asked.on = true, shot_asked.value = value, shot_asked.present = presents, shot_asked.ask = std::move(a);
}
/// (do_present, replay_mu) the shot being read back, once the GPU is past it: never waited for
void shot_ask_take() {
    if (!shot_asked.on || completed() < shot_asked.value) return;
    invalidate_mapped(ask_buf_alloc, (uint64_t)kShotW * kShotH * 4);   // (non-coherent host memory: what the GPU wrote)
    auto px = std::make_shared<std::vector<uint8_t>>((const uint8_t*)ask_map, (const uint8_t*)ask_map + (size_t)kShotW * kShotH * 4);
    ShotAsk a = std::move(shot_asked.ask);
    const int64_t at = shot_asked.present;
    shot_asked.on = false;
    std::thread([px, a, at] {
        // (through a file of its own and a rename: whoever is told the path never reads half a picture)
        const std::string part = a.path + ".part";
        bool ok = write_png(part.c_str(), px->data(), kShotW, kShotH);
        if (ok) {
            std::remove(a.path.c_str());
            ok = std::rename(part.c_str(), a.path.c_str()) == 0;
        }
        if (!ok) std::remove(part.c_str());
        logf("vk: present %lld %s %s (the control bridge)", (long long)at, ok ? "written to" : "could not be written to",
             a.path.c_str());
        a.done(a.ctx, at, ok ? 1 : 0);
    }).detach();
}
void frame_info(int64_t* replayed, int64_t* draws, int* width, int* height) {
    int64_t v[4] = {0, 0, 0, 0};
    draw_stats(&v[0], &v[1], &v[2], &v[3]);   // (counters the replay's thread writes: read as they are, a sample)
    if (replayed) *replayed = __atomic_load_n(&presents_now, __ATOMIC_RELAXED);
    if (draws) *draws = v[0];
    if (width) *width = (int)dev.swap_extent.width;
    if (height) *height = (int)dev.swap_extent.height;
}

uint64_t present_value = 0;   // (the present job's value, handed out as it came; 0: the next one)
struct PendingShot {
    bool on = false;
    uint64_t value = 0;
    int64_t presents = 0;
    std::shared_ptr<std::vector<uint8_t>> pixels;
};
std::shared_ptr<std::vector<uint8_t>> shot_pixels(uint64_t at_value, int64_t at_present) {
    wait_value(at_value);
    // The completed transfer may target non-coherent host memory.
    invalidate_mapped(shot_allocs[at_present & 1], (uint64_t)kShotW * kShotH * 4);
    const uint8_t* const sm = (const uint8_t*)shot_maps[at_present & 1];
    return std::make_shared<std::vector<uint8_t>>(sm, sm + (size_t)kShotW * kShotH * 4);
}
// (hookshot) the Travel page's own pictures, vk_menu.cpp: when the game side says Link has stayed at a place, the
// middle 60% of the picture shown, away from the HUD's corners, at 480x270; recorded with one present and read at a later one
// once the GPU has finished it (a present never waits for it), handed to the menu's picture worker
struct TeleportShot {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation image_alloc = nullptr;
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation buf_alloc = nullptr;
    void* map = nullptr;
    bool tried = false;
    int pending = -1;
    uint64_t value = 0;
};
TeleportShot tp_shot;
constexpr uint32_t kTpShotW = 480, kTpShotH = 270;
void teleport_capture(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, const int crop[4], uint64_t value) {
    if (tp_shot.pending >= 0) {   // (the last one's pixels, once the GPU is past its present: never waited for here)
        if (completed() < tp_shot.value) return;
        invalidate_mapped(tp_shot.buf_alloc, (uint64_t)kTpShotW * kTpShotH * 4);
        menu_store_capture(tp_shot.pending, (const uint8_t*)tp_shot.map, kTpShotW, kTpShotH);
        tp_shot.pending = -1;
    }
    if (!src) return;
    const int want = menu_take_capture_request();
    if (want < 0) return;
    if (!tp_shot.image) {
        if (tp_shot.tried) return;
        tp_shot.tried = true;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ci.extent = {kTpShotW, kTpShotH, 1};
        ci.mipLevels = ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation image_alloc = nullptr;
        if (!create_image(ci, image, image_alloc)) {
            logf("vk: Travel pictures: no image could be made");
            return;
        }
        if (!create_buffer((uint64_t)kTpShotW * kTpShotH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, tp_shot.buf, tp_shot.buf_alloc,
                           &tp_shot.map)) {
            destroy_image(image, image_alloc);
            logf("vk: Travel pictures: no buffer could be made");
            return;
        }
        tp_shot.image = image, tp_shot.image_alloc = image_alloc;
    }
    int x0 = 0, y0 = 0, w = (int)sw, h = (int)sh;
    if (crop && crop[2] > 0 && crop[3] > 0) x0 = crop[0], y0 = crop[1], w = crop[2], h = crop[3];
    const int middle[4] = {x0 + w / 5, y0 + h / 5, w - 2 * (w / 5), h - 2 * (h / 5)};
    if (middle[2] <= 0 || middle[3] <= 0) return;
    image_barrier(cmd, tp_shot.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    blit_crop(cmd, src, sw, sh, middle, tp_shot.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kTpShotW, kTpShotH);
    image_barrier(cmd, tp_shot.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {kTpShotW, kTpShotH, 1}};
    vkCmdCopyImageToBuffer(cmd, tp_shot.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tp_shot.buf, 1, &c);
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};   // (the copy seen by the host's read at the next present)
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
    tp_shot.pending = want, tp_shot.value = value;
}

// (hookshot) F12's screenshot (vk_screenshot.h): the picture this present shows, whole and at its own size (the game's output
// resolution: before the window's scaling, the Insert menu and the FPS counter), copied with the present's recording and
// handed to the menu's writer at a later present, once the GPU is past it (never waited for). Its image and buffer live
// only that long: a 3840x2160 picture is 33 MB of each
struct ScreenShot {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation image_alloc = nullptr;
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation buf_alloc = nullptr;
    void* map = nullptr;
    uint32_t w = 0, h = 0;
    bool pending = false;
    uint64_t value = 0;
};
ScreenShot screen_shot;
void screenshot_capture(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, const int crop[4], uint64_t value) {
    ScreenShot& s = screen_shot;
    if (s.pending) {
        if (completed() < s.value) return;
        const size_t bytes = (size_t)s.w * s.h * 4;
        invalidate_mapped(s.buf_alloc, bytes);
        std::vector<uint8_t> px((const uint8_t*)s.map, (const uint8_t*)s.map + bytes);
        menu_store_screenshot(std::move(px), s.w, s.h);
        destroy_image(s.image, s.image_alloc);
        destroy_buffer(s.buf, s.buf_alloc);
        s = ScreenShot{};
        return;
    }
    if (!src || !sw || !sh || !menu_take_screenshot_request()) return;
    const bool cropped = crop && crop[2] > 0 && crop[3] > 0;
    s.w = cropped ? (uint32_t)crop[2] : sw, s.h = cropped ? (uint32_t)crop[3] : sh;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {s.w, s.h, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (!create_image(ci, s.image, s.image_alloc)) {
        logf("vk: screenshot: no %ux%u image could be made", s.w, s.h);
        s = ScreenShot{};
        return;
    }
    if (!create_buffer((uint64_t)s.w * s.h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, s.buf, s.buf_alloc, &s.map)) {
        logf("vk: screenshot: no buffer for a %ux%u picture could be made", s.w, s.h);
        destroy_image(s.image, s.image_alloc);
        s = ScreenShot{};
        return;
    }
    image_barrier(cmd, s.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    blit_crop(cmd, src, sw, sh, cropped ? crop : nullptr, s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, s.w, s.h);
    image_barrier(cmd, s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {s.w, s.h, 1}};
    vkCmdCopyImageToBuffer(cmd, s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.buf, 1, &c);
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};   // (the copy seen by the host's read at a later present)
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
    s.pending = true, s.value = value;
}

void do_present(const nvn_tex* t, const int crop[4], float cover) {
#ifdef TRACY_ENABLE
    HT_SCOPE("do_present");
#endif
    static PendingShot pending;
    // (hookshot) NVIDIA Reflex (vk_reflex.cpp): this frame's simulation ended, its submission starts; on the way out (after the
    // present) the driver's sleep until the next frame should begin. AMD Anti-Lag 2 (vk_antilag.cpp) waits there as well: the
    // next frame's input stage
    temporal_idle();   // (DLSS, FSR: their video memory let go once not used for a while)
    status_tick();   // (the status line, and the video-memory warning)
    reflex_present_begins();
    struct ReflexSleep { ~ReflexSleep() { FpsWait fw(fps_stats::kPace); reflex_sleep(); antilag_input(); } } reflex_sleep_after;
    {   // (hookshot, RX_VK_PASS_TRACE_AT=<frames>: the passes of the frames around these frame numbers, for finding what a frame is made of)
        static std::atomic<int64_t> seen{0};
        static const int64_t at = getenv("RX_VK_PASS_TRACE_AT") ? atoll(getenv("RX_VK_PASS_TRACE_AT")) : 0;
        if (at && ++seen % at == 0) pass_traces_log(texture_frame_no() - 1, texture_frame_no());
    }
    static const char* shots = getenv("RX_VK_SHOTS");
    static const int shot_every = getenv("RX_VK_SHOT_EVERY") ? std::max(1, atoi(getenv("RX_VK_SHOT_EVERY"))) : 300;
    VkImage src = VK_NULL_HANDLE;
    VkImageView direct_view = VK_NULL_HANDLE;
    uint32_t sw = 0, sh = 0;
    bool shot = false, interpolated = false;
    VkImage made_src = VK_NULL_HANDLE;
    int made_index = 0;
    VkImage headless_fx = VK_NULL_HANDLE;   // CRT output at the capture pixel grid, reused by both screenshot routes
    {
        std::lock_guard lk(replay_mu);
        fsr3::collect_retired();   // The last FSR 3.1 engine release may have left GPU work in flight.
        bench_frame(presents);
        SlowWatch slow("present");
        frame_submits.clear();
        PROF(PROF_PRESENT);
        const uint64_t value = present_value ? present_value : ++q_values;
        present_value = 0;
        rec_value_now = value;
        ub_first_of_recording = ub_updates.size();
        Recording& r = begin_recording(value);
        memory_begin_submission();
        memory_gpu_writes_begin();   // (RX_VK_UB_SNAPSHOT's guard: a submission's start, not a part's: split_part)
        Texture* tex = t ? texture_for_present(t) : nullptr;
        static int logged_p;
        if (logged_p++ < 4)
            logf("vk: present texture %#llx %ux%u format %#x: %s", (unsigned long long)(t ? t->gpu : 0),
                 t ? t->d.width : 0, t ? t->d.height : 0, t ? t->d.format : 0,
                 tex ? (tex->gpu_written ? "rendered" : "never rendered") : "no image");
        if (tex) src = tex->image, sw = tex->width, sh = tex->height;
        if (t) temporal_note_present(t->d.width, t->d.height);   // (RX_VK_DLSS_PREWARM: the size its feature is made for)
        // (the Insert menu's HDR set to Native: the game, told its display is HDR10, presents PQ codes in RGB10A2 (vk
        // A2B10G10R10); that picture goes to the HDR10 swapchain as PQ, not through the SDR-to-HDR expansion. Told by the format
        // the game presents, the one its view is made in (direct_view below), not by the cached image's: images are shared by
        // format class (vk_texture.cpp format_class), and while RGBA8 and RGB10A2 shared one (#2038) the window's RGB10A2 picture
        // was a view of an RGBA8 image and took the SDR expansion, a flat, grey picture. The image's format is not asked as well:
        // an image first made as RGB10A2 under an RGBA8 picture would take an SDR picture for PQ)
        const VkFormat shown_format = t ? vk_format(t->d.format).format : VK_FORMAT_UNDEFINED;
        present_native_pq.store(tex && dev.hdr && opts.hdr == 2 && shown_format == VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                std::memory_order_relaxed);
        {   // (the presented picture's format, the game's and its image's, and the HDR path it takes, said when any of them
            // changes: what an HDR report needs)
            static int64_t said = -1;
            const int64_t now =
                tex ? ((int64_t)tex->format * 4096 + (int64_t)shown_format) * 8 + (dev.hdr ? 4 : 0) + opts.hdr : -2;
            if (now != said) {
                said = now;
                logf("vk: present: %ux%u, format %d (its image's %d); the Insert menu's HDR %d, HDR10 swapchain %s: %s",
                     tex ? tex->width : 0, tex ? tex->height : 0, (int)shown_format, tex ? (int)tex->format : 0, opts.hdr,
                     dev.hdr ? "yes" : "no",
                     !dev.hdr ? "SDR" : present_native_pq.load(std::memory_order_relaxed) ? "the game's PQ picture (Native)"
                                                                                        : "the SDR picture expanded to HDR10");
            }
        }
        bool wheel = false;
        if (tex && !dev.offscreen && (opts.upscale || opts.sharpness || opts.crt_filter || dev.hdr || wide_sides_possible())) {
            // The Hookshot item wheel composites its own backdrop and wheel before interpolation or screenshots read them.
            wheel = framegen_wheel_backdrop(r.cmd, tex->image, tex->format, sw, sh, crop);
            // Keep the sampled view with the same cached image as the direct present, before advancing its age.
            direct_view = texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample);
            tex->last_frame = texture_frame_no();
            tex->last_value = std::max(tex->last_value, value);
        }
        if (menu_fade_wanted()) {
            const VkImageUsageFlags required = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                               VK_IMAGE_USAGE_STORAGE_BIT;
            const bool compatible = tex && !dev.offscreen && !dev.hdr && !framegen_on() && swapchain_framegen() < 2 &&
                                    tex->format == VK_FORMAT_R8G8B8A8_UNORM && shown_format == tex->format &&
                                    tex->type == VK_IMAGE_TYPE_2D && tex->levels == 1 && tex->layers == 1 &&
                                    tex->samples == VK_SAMPLE_COUNT_1_BIT && (tex->usage & required) == required;
            VkImageView storage = compatible ? texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Storage) : VK_NULL_HANDLE;
            if (storage) {
                tex->last_frame = texture_frame_no();
                tex->last_value = std::max(tex->last_value, value);
            }
            const bool menu = wheel || (presents_now > 0 && wheel_drawn_at.load(std::memory_order_relaxed) == presents_now) ||
                              menu_fade_pause_at.load(std::memory_order_relaxed) == presents_now;
            const bool world = menu_fade_world_at.load(std::memory_order_relaxed) == presents_now;
            menu_fade_present(r.cmd, src, storage, sw, sh, menu, world, compatible);
        }
        // (frame interpolation: made frames and this one shown by the display thread, vk_framegen.cpp. Recorded
        // after the submit that signals the present's value, in a submit of its own: the game waits for that value
        // before it submits its next frame, and the interpolation's GPU work (~1 ms) held it back. The display
        // thread's present follows it on the one queue. RX_VK_SHOT_MADE shots need it in this recording.)
        static const bool fg_inline = env_on("RX_VK_SHOT_MADE");
        const bool fg_after = framegen_on() && tex && !fg_inline;
        interpolated = framegen_on() && tex && fg_inline &&
                       framegen_record(r.cmd, texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample), sw,
                                       sh, crop, cover, present_native_pq.load(std::memory_order_relaxed));
        texture_frame(completed());
        full_barrier(r.cmd);
        presents++;
        presents_now = presents;
        new_refuse_test(presents);
        wide_sides_tick(cover);
        if (presents == 1) hang_watch_start(), gpu_hang_watch_start();
        {   // (EvilBoyce, diagnostics) RX_VK_WATCH_CPU=<present>:<gpu address>[,...] (hex addresses): from that present on, the
            // CPU's writes to those 4 KB pages logged with the writer's stack (host page watch: 12 reports)
            static bool armed = false;
            static const char* e = getenv("RX_VK_WATCH_CPU");
            if (e && !armed && host->watch_guest_writes && presents >= atoll(e)) {
                armed = true;
                for (const char* q = strchr(e, ':'); q && *q;) {
                    char* end;
                    const uint64_t a = strtoull(q + 1, &end, 16);
                    if (end == q + 1) break;
                    if (const uint64_t g = host->gpu_guest(a)) host->watch_guest_writes(g & ~4095ull, 4096);
                    q = *end ? end : nullptr;
                }
            }
        }
        ts_anchor_ns = host->ticks() * 32;
        dump_present(presents);
        if (src && (dev.offscreen || shots) && offscreen_init()) {
            if (dev.offscreen) {
                image_barrier(r.cmd, off_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                // No swapchain exists here. Exercise the same output effects on every frame, at the
                // requested capture size, not only when a screenshot is due. Off keeps the original blit.
                if (opts.crt_filter && tex)
                    headless_fx = framegen_scale(r.cmd, texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample),
                                                 src, sw, sh, crop, kShotW, kShotH,
                                                 present_native_pq.load(std::memory_order_relaxed));
                if (headless_fx)
                    blit_crop(r.cmd, headless_fx, kShotW, kShotH, nullptr, off_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              dev.swap_extent.width, dev.swap_extent.height);
                else
                    blit_crop(r.cmd, src, sw, sh, crop, off_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              dev.swap_extent.width, dev.swap_extent.height);
            }
            // (RX_VK_SHOT_FROM / _TO: every present in that range too)
            static const int64_t shot_from = getenv("RX_VK_SHOT_FROM") ? atoll(getenv("RX_VK_SHOT_FROM")) : -1;
            static const int64_t shot_to = getenv("RX_VK_SHOT_TO") ? atoll(getenv("RX_VK_SHOT_TO")) : -1;
            shot = shots && (presents % shot_every == 0 || (presents >= shot_from && presents <= shot_to));
            // (RX_VK_SHOT_GO=<file>: every present while that file is there too, a test's run of frames)
            static const char* shot_go = getenv("RX_VK_SHOT_GO");
            if (shots && shot_go && !shot) {
                struct stat sb;
                shot = stat(shot_go, &sb) == 0;
            }
            // (RX_VK_SHOT_MENU=1, a debug aid: without a window the Insert menu is neither ticked nor drawn; this ticks it
            // at each present and draws it into the shots, for pictures of it from a headless run)
            static const bool shot_menu = env_on("RX_VK_SHOT_MENU");
            if (shot_menu && dev.offscreen) menu_tick();
            // (RX_VK_SHOT_MADE=1: the frame interpolation made before it too, written beside it as vk_N_made.png)
            static const bool shot_made = env_on("RX_VK_SHOT_MADE");
            uint32_t mw = 0, mh = 0;
            made_src = shot && shot_made && interpolated ? framegen_made_image(&mw, &mh, &made_index) : VK_NULL_HANDLE;
            if (made_src && !made_shot_init()) made_src = VK_NULL_HANDLE;
            if (made_src) {
                image_barrier(r.cmd, made_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                blit_crop(r.cmd, made_src, mw, mh, nullptr, made_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kShotW, kShotH);
                image_barrier(r.cmd, made_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                const VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {kShotW, kShotH, 1}};
                vkCmdCopyImageToBuffer(r.cmd, made_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, made_buf, 1, &c);
                capture_host_read_barrier(r.cmd, made_buf, (VkDeviceSize)kShotW * kShotH * 4);
            }
            if (shot) {
                // A pending shot may be two (or more) presents old. Own its pixels before this slot is written again.
                if (pending.on && (pending.presents & 1) == (presents & 1) && !pending.pixels)
                    pending.pixels = shot_pixels(pending.value, pending.presents);
                image_barrier(r.cmd, shot_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                // (RX_VK_SHOT_FX=1: through the window's scale pass too, ultrawide's menu sides as the window shows them. For headless
                // runs: in a window it resizes the pass's image to the shot's size and the present back again, two device waits a
                // shot, and it is not under present_image_mu, which frame generation's display thread takes for the same pass)
                static const bool shot_fx = env_on("RX_VK_SHOT_FX");
                // (pq as the window's present has it: Native HDR's picture through the 10-bit pass, not a format switch each shot)
                const VkImage fx_img = headless_fx ? headless_fx : shot_fx && tex ? framegen_scale(r.cmd, texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample),
                                                                       src, sw, sh, crop, kShotW, kShotH,
                                                                       present_native_pq.load(std::memory_order_relaxed))
                                                        : VK_NULL_HANDLE;
                if (fx_img) full_barrier(r.cmd), blit_crop(r.cmd, fx_img, kShotW, kShotH, nullptr, shot_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kShotW, kShotH);
                else blit_crop(r.cmd, src, sw, sh, crop, shot_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kShotW, kShotH);
                if (shot_menu && dev.offscreen) menu_record_image(r.cmd, shot_image, shot_view, kShotW, kShotH);   // (headless: the menu over the shot)
                image_barrier(r.cmd, shot_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                const VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {kShotW, kShotH, 1}};
                vkCmdCopyImageToBuffer(r.cmd, shot_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, shot_bufs[presents & 1], 1, &c);
                capture_host_read_barrier(r.cmd, shot_bufs[presents & 1], (VkDeviceSize)kShotW * kShotH * 4);
            }
        }
        shot_ask_record(r.cmd, headless_fx ? headless_fx : src, headless_fx ? kShotW : sw, headless_fx ? kShotH : sh,
                        headless_fx ? nullptr : crop, value);   // the bridge captures the same CRT output, without Hookshot overlays
        teleport_capture(r.cmd, src, sw, sh, crop, value);   // (hookshot: the Travel page's own pictures)
        screenshot_capture(r.cmd, src, sw, sh, crop, value);   // (hookshot: F12)
        memory_end_submission(r.cmd, r.part);
        full_barrier(r.cmd);
        submit_recording(r);   // (nothing is stopped at a present: drain)
        if (pushed.load() < value) pushed = value;
        done_max = std::max(done_max, value), signaled = std::max(signaled, value);
        shot_ask_take();   // (an earlier present's, when the GPU is past it)
        if (fg_after) {
            // (its slot is free again once the next submit's value is reached: that one comes after it)
            Recording& g = begin_recording(value + 1);
            interpolated = framegen_record(g.cmd, texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample),
                                           sw, sh, crop, cover, present_native_pq.load(std::memory_order_relaxed));
            submit_recording(g, false);
        }
        // (frame generation at Off or Auto, where DLSS-G is not asked: its feature, if made, let go, vk_dlss.cpp)
        if (swapchain_framegen() < 2) dlss_fg_idle();
        if (interpolated) {   // (NVIDIA Reflex: the game frame's rendering is submitted; the display thread presents it later)
            reflex_game_submitted();
            framegen_queue(value);
        }
        else menu_note_frame(true);   // (the FPS counter's, as frame generation's thread counts what it shows)
        // (and the built-in interpolation's and FSR 3.1's images, off for a while: after the queue, so a set this present recorded
        // is the display thread's to show and free before any drain waits for it; and counted as on, whatever the row says now)
        framegen_idle(interpolated);
        {   // (RX_VK_WAIT_GPU=<frames>: the replay no further ahead of the GPU than that)
            static uint64_t values[8];
            static const int ahead = [] {
                const char* e = getenv("RX_VK_WAIT_GPU");
                const long count = e ? std::strtol(e, nullptr, 10) : 0;
                constexpr long capacity = sizeof(values) / sizeof(values[0]);
                if (count > capacity) logf("vk: RX_VK_WAIT_GPU=%ld exceeds the ring capacity; using %ld", count, capacity);
                return (int)std::clamp<long>(count, 0, capacity);
            }();
            if (ahead > 0) {
                const uint64_t old = values[presents % ahead];
                values[presents % ahead] = value;
                if (old) wait_value(old);
            }
        }
        if (presents % 300 == 0) pipeline_cache_save(false);
        {   // (RX_VK_DUMP_KEY: F9 dumps the next frame, 8 times at most)
            static const bool dump_key = env_on("RX_VK_DUMP_KEY");
            static bool was_down;
            static int key_left = 8;
            const bool down = dump_key && (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
            if (down && !was_down && key_left > 0) {
                key_left--;
                static const int n = getenv("RX_VK_DUMP_KEY_N") ? std::max(1, atoi(getenv("RX_VK_DUMP_KEY_N"))) : 1;
                logf("vk: F9 at present %lld: the next %d frame(s) are dumped", (long long)presents, n);
                dump_request(presents + 1, n);
            }
            was_down = down;
        }
        {   // (RX_VK_DUMP_FILE=<path>: that file appearing dumps the next frame, and is deleted: a test that steers the
            // game live (HOS_INPUT_LIVE) dumps where it stands)
            // (RX_VK_DUMP_MOD=<m>,<r>[,<n>]: the file arms the next n frames (default 2) whose present is r mod m
            // instead: a glitch that comes back every m frames, dumped in its own frame (the flash detector names it))
            static const char* dump_file = getenv("RX_VK_DUMP_FILE");
            static int mod_m = -1, mod_r = 0, mod_n = 2, mod_left = 0;
            if (mod_m < 0) {
                mod_m = 0;
                if (const char* e = getenv("RX_VK_DUMP_MOD")) sscanf(e, "%d,%d,%d", &mod_m, &mod_r, &mod_n);
            }
            if (dump_file && presents % 10 == 0 && GetFileAttributesA(dump_file) != INVALID_FILE_ATTRIBUTES) {
                DeleteFileA(dump_file);
                if (mod_m > 0) {
                    mod_left = std::max(1, mod_n);
                    logf("vk: dump file at present %lld: the next %d frames at %d mod %d are dumped", (long long)presents,
                         mod_left, mod_r, mod_m);
                } else {
                    logf("vk: dump file at present %lld: the next frame is dumped", (long long)presents);
                    dump_request(presents + 1, 1);
                }
            }
            if (mod_left > 0 && (presents + 2) % mod_m == mod_r) {   // (armed at the present before the frame's own)
                mod_left--;
                dump_request(presents + 1, 1);
            }
        }
        // (a shot is read once the GPU reached its frame, at a later present, and written on a thread of its own:
        // waiting for the GPU and encoding the PNG there took the replay ~570 ms a shot, and frequent shots
        // halved the frame rate test runs measured. RX_VK_SHOT_MADE shots are taken at once, as before.)
        auto take = [&](uint64_t at_value, int64_t at_present) {
            auto px = pending.on && pending.presents == at_present && pending.pixels
                          ? std::move(pending.pixels) : shot_pixels(at_value, at_present);
            // (RX_VK_DUMP_BLACK=<n>: a shot with 12-90% pure black pixels in a lit picture, or a green cast over it,
            // dumps the next frame, n times at most: the glitches some scenes show now and then)
            static int black_left = getenv("RX_VK_DUMP_BLACK") ? atoi(getenv("RX_VK_DUMP_BLACK")) : 0;
            if (black_left > 0) {
                const uint8_t* q0 = px->data();
                uint64_t black = 0, white = 0, sum = 0, ch[3] = {};
                const uint64_t n = (uint64_t)kShotW * kShotH;
                for (uint64_t i = 0; i < n; i++) {
                    const uint8_t* q = q0 + i * 4;
                    black += q[0] < 3 && q[1] < 3 && q[2] < 3;
                    white += q[0] > 235 && q[1] > 235 && q[2] > 235;
                    sum += q[0] + q[1] + q[2];
                    ch[0] += q[0], ch[1] += q[1], ch[2] += q[2];
                }
                // (hookshot: a picture blown out to white, the HUD on it: written as white_<present>.png, 40 of them)
                static int white_left = 40;
                if (at_present > 600 && white * 100 > n * 40 && white_left > 0) {
                    white_left--;
                    logf("vk: present %lld: %.1f%% white pixels, mean %llu %llu %llu", (long long)at_present, 100.0 * white / n,
                         (unsigned long long)(ch[0] / n), (unsigned long long)(ch[1] / n), (unsigned long long)(ch[2] / n));
                    auto w0 = px;
                    const std::string name = std::string(shots) + "/white_" + std::to_string(at_present) + ".png";
                    std::thread([w0, name] { write_png(name.c_str(), w0->data(), kShotW, kShotH); }).detach();
                }
                const bool green = ch[1] * 10 > ch[0] * 12 && ch[1] * 10 > ch[2] * 14 && ch[1] / n > 120;
                if (at_present > 600 && ((black * 8 > n && black * 100 < n * 97 && sum / (n * 3) > 4) || green)) {
                    black_left--;
                    logf("vk: present %lld: %.1f%% black pixels, mean %llu %llu %llu: the next frame is dumped",
                         (long long)at_present, 100.0 * black / n, (unsigned long long)(ch[0] / n),
                         (unsigned long long)(ch[1] / n), (unsigned long long)(ch[2] / n));
                    dump_request(presents + 1);
                }
            }
            // (RX_VK_FLASH=1, with RX_VK_SHOT_EVERY=1: a frame that differs from both its neighbours in a part of the
            // picture where they agree is a one-frame flash: the three are written as flash_<present>_{a,b,c}.png and
            // the next frame is dumped (a few times). The other shots are written every 300th present only.)
            static const bool flash = getenv("RX_VK_FLASH") && atoi(getenv("RX_VK_FLASH")) != 0;
            if (flash) {
                constexpr uint32_t W = 64, H = 36;
                auto small = std::make_shared<std::vector<uint8_t>>(W * H * 3);
                for (uint32_t y = 0; y < H; y++) {
                    const uint32_t y0 = (uint64_t)y * kShotH / H;
                    const uint32_t y1 = std::max(y0 + 1, (uint32_t)((uint64_t)(y + 1) * kShotH / H));
                    for (uint32_t x = 0; x < W; x++) {
                        const uint32_t x0 = (uint64_t)x * kShotW / W;
                        const uint32_t x1 = std::max(x0 + 1, (uint32_t)((uint64_t)(x + 1) * kShotW / W));
                        for (int ch = 0; ch < 3; ch++) {
                            uint64_t sum = 0, samples = 0;
                            // Partition the full image and count the samples actually visited. A partial
                            // stride must not brighten the cell or wrap its mean back through uint8.
                            for (uint32_t yy = y0; yy < y1; yy += 4)
                                for (uint32_t xx = x0; xx < x1; xx += 4) {
                                    sum += (*px)[((size_t)yy * kShotW + xx) * 4 + ch];
                                    samples++;
                                }
                            (*small)[(y * W + x) * 3 + ch] = (uint8_t)(sum / samples);
                        }
                    }
                }
                static std::shared_ptr<std::vector<uint8_t>> sa, sb, pa, pb;
                static int64_t present_b = -1;
                static int flashes_left = 400, dumps_left = getenv("RX_VK_FLASH_DUMPS") ? atoi(getenv("RX_VK_FLASH_DUMPS")) : 6;
                if (sa && sb && present_b == at_present - 1) {
                    // (RX_VK_FLASH_DIFF, RX_VK_FLASH_CELLS: a subtler flash; logged with how many cells went up, and where)
                    static const int fl_diff = getenv("RX_VK_FLASH_DIFF") ? atoi(getenv("RX_VK_FLASH_DIFF")) : 48;
                    static const int fl_cells = getenv("RX_VK_FLASH_CELLS") ? atoi(getenv("RX_VK_FLASH_CELLS")) : 40;
                    int hits = 0, up = 0, minx = W, maxx = -1, miny = H, maxy = -1, peak = 0;
                    uint64_t tab = 0, tac = 0;   // (camera motion: a and c differ as much as either from b; a flash: a ~ c)
                    for (uint32_t i = 0; i < W * H * 3; i++)
                        tab += std::abs((*sa)[i] - (*sb)[i]), tac += std::abs((*sa)[i] - (*small)[i]);
                    for (uint32_t i = 0; i < W * H; i++) {
                        int dab = 0, dbc = 0, dac = 0;
                        for (int ch = 0; ch < 3; ch++) {
                            dab = std::max(dab, std::abs((*sa)[i * 3 + ch] - (*sb)[i * 3 + ch]));
                            dbc = std::max(dbc, std::abs((*sb)[i * 3 + ch] - (*small)[i * 3 + ch]));
                            dac = std::max(dac, std::abs((*sa)[i * 3 + ch] - (*small)[i * 3 + ch]));
                        }
                        if (dab > fl_diff && dbc > fl_diff && dac < fl_diff / 3) {
                            hits++;
                            up += (*sb)[i * 3 + 1] > (*sa)[i * 3 + 1];
                            peak = std::max(peak, dab);
                            const int cx = (int)(i % W), cy = (int)(i / W);
                            minx = std::min(minx, cx), maxx = std::max(maxx, cx);
                            miny = std::min(miny, cy), maxy = std::max(maxy, cy);
                        }
                    }
                    // (RX_VK_FLASH_REGION=<level>: also while the picture moves, when the cells' test cannot hold: a 16x9 block of
                    // the picture whose mean brightness jumps by the level and comes back the next frame)
                    static const int fl_region = getenv("RX_VK_FLASH_REGION") ? atoi(getenv("RX_VK_FLASH_REGION")) : 0;
                    int region_hits = 0, region_peak = 0, region_at = -1;
                    if (fl_region)
                        for (uint32_t ry = 0; ry < 9; ry++)
                            for (uint32_t rx = 0; rx < 16; rx++) {
                                int m[3] = {0, 0, 0};
                                for (uint32_t y = ry * 4; y < ry * 4 + 4; y++)
                                    for (uint32_t x = rx * 4; x < rx * 4 + 4; x++) {
                                        const size_t o = (y * W + x) * 3 + 1;
                                        m[0] += (*sa)[o], m[1] += (*sb)[o], m[2] += (*small)[o];
                                    }
                                const int ab = (m[1] - m[0]) / 16, cb = (m[1] - m[2]) / 16, ac = std::abs(m[0] - m[2]) / 16;
                                if (((ab > fl_region && cb > fl_region) || (ab < -fl_region && cb < -fl_region)) &&
                                    ac * 2 < std::min(std::abs(ab), std::abs(cb))) {
                                    region_hits++;
                                    if (std::abs(ab) > std::abs(region_peak)) region_peak = ab, region_at = (int)(ry * 16 + rx);
                                }
                            }
                    if (region_hits && flashes_left > 0)
                        logf("vk: region flash at present %lld: %d regions, peak %+d at region %d,%d", (long long)present_b, region_hits,
                             region_peak, region_at % 16, region_at / 16);
                    if (((hits >= fl_cells && tac * 2 < tab) || region_hits) && flashes_left > 0) {
                        flashes_left--;
                        logf("vk: flash at present %lld: %d of %u cells differ from both neighbours, %d brighter, peak %d, "
                             "cells %d,%d-%d,%d (now present %lld, texture frame %llu)", (long long)present_b, hits, W * H, up,
                             peak, minx, miny, maxx, maxy, (long long)presents, (unsigned long long)texture_frame_no());
                        const std::string base = std::string(shots) + "/flash_" + std::to_string(present_b);
                        auto a0 = pa, b0 = pb, c0 = px;
                        std::thread([a0, b0, c0, base] {
                            write_png((base + "_a.png").c_str(), a0->data(), kShotW, kShotH);
                            write_png((base + "_b.png").c_str(), b0->data(), kShotW, kShotH);
                            write_png((base + "_c.png").c_str(), c0->data(), kShotW, kShotH);
                        }).detach();
                        if (dumps_left > 0) dumps_left--, dump_request(presents + 1);
                        static int records_left = 12;   // (RX_VK_FLASH_RECORD: the draws around it)
                        const uint64_t tf = (uint64_t)(present_b + (int64_t)texture_frame_no() - presents);
                        probe_log(tf);
                        if (records_left > 0) records_left--, draw_records_log(tf - 1, tf + 1), pass_traces_log(tf - 1, tf + 1);
                    }
                }
                sa = sb, pa = pb, sb = small, pb = px, present_b = at_present;
                if (at_present % 300) return;
            }
            std::string path = std::string(shots) + "/vk_" + std::to_string(1000000 + at_present).substr(1) + ".png";
            std::thread([px, path, at_present] {
                if (write_png(path.c_str(), px->data(), kShotW, kShotH))
                    logf("vk: present %lld written to %s", (long long)at_present, path.c_str());
            }).detach();
        };
        if (pending.on && (shot || completed() >= pending.value)) take(pending.value, pending.presents), pending.on = false;
        if (shot && made_src && made_map) {
            take(value, presents);
            invalidate_mapped(made_buf_alloc, (uint64_t)kShotW * kShotH * 4);
            char path[1024];
            if (made_index < 0) snprintf(path, sizeof path, "%s/vk_%06lld_made.png", shots, (long long)presents);
            else snprintf(path, sizeof path, "%s/vk_%06lld_made%d.png", shots, (long long)presents, made_index);
            write_png(path, (const uint8_t*)made_map, kShotW, kShotH);
        } else if (shot) {
            pending = {true, value, presents, {}};
        }
        log_frame_stats();
    }
    if (dev.offscreen) frame_log_present();   // (no window: the copy into the offscreen image above is its present)
    if (dev.offscreen || interpolated) return;
    present_image(src, sw, sh, crop, direct_view);
}

} // namespace

int shot_ask_next(const char* path, void (*done)(void* ctx, int64_t present, int ok), void* ctx) { return shot_ask(path, -1, done, ctx); }

// (hookshot) RX_VK_FRAMELOG=<file>: every frame shown, the game's and the made ones, as they go to the screen, 320x180, while
// <file>.go exists (scripts/framelog.py finds the flashes). Taken at the last step and written on a thread of its own, so
// that the game keeps its speed: a shot of every frame at 1280x720 held it at 30 fps, where the flashes may not show.
// A record: u64 time (ns), u32 sequence, u32 kind (0 the game's, 1 made), 320x180 RGBA.
void exposure_log_dump(const char* why);   // (vk_draw.cpp)
void probe_dump_all(const char* why);
bool probe_serial();
namespace {
constexpr uint32_t kLogW = 320, kLogH = 180;
const char* framelog_path() {
    static const char* p = getenv("RX_VK_FRAMELOG");
    return p;
}
struct LogSlot {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    void* map = nullptr;
    bool used = false;
    int64_t at = 0;
    uint32_t seq = 0, kind = 0;
};
LogSlot log_slots[kPresentSlotsMax];   // (by present_image's frame slot)
VkImage log_image = VK_NULL_HANDLE;
VmaAllocation log_image_alloc = nullptr;
std::mutex log_mu;
std::condition_variable log_cv;
std::deque<std::vector<uint8_t>> log_queue;
std::atomic<int> log_state{0};   // 0: not tried; 1: complete sink; -1: disabled after a failure
void framelog_resources_drop() {
    for (LogSlot& l : log_slots) {
        if (l.buf) destroy_buffer(l.buf, l.alloc);
        l = LogSlot{};
    }
    if (log_image) destroy_image(log_image, log_image_alloc);
    log_image = VK_NULL_HANDLE;
    log_image_alloc = nullptr;
}
void framelog_write(FILE* f) {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk frame log");
    for (;;) {
        std::vector<uint8_t> rec;
        {
            std::unique_lock lk(log_mu);
            log_cv.wait(lk, [] { return !log_queue.empty(); });
            rec = std::move(log_queue.front());
            log_queue.pop_front();
        }
        const size_t written = fwrite(rec.data(), 1, rec.size(), f);
        if (written != rec.size() || fflush(f) != 0) {
            {
                std::lock_guard lk(log_mu);
                log_state.store(-1, std::memory_order_release);
                log_queue.clear();   // (no worker remains to consume these; don't retain the queued CPU pictures)
            }
            logf("vk: frame log stopped: %s failed (wrote %zu of %zu bytes)", written == rec.size() ? "flush" : "write",
                 written, rec.size());
            fclose(f);
            return;
        }
    }
}
void framelog_enqueue(std::vector<uint8_t> rec) {
    std::lock_guard lk(log_mu);
    if (log_state.load(std::memory_order_acquire) > 0 && log_queue.size() < 512) {
        log_queue.push_back(std::move(rec));
        log_cv.notify_one();
    }
}
bool framelog_ready() {
    const int state = log_state.load(std::memory_order_acquire);
    if (state) return state > 0;
    log_state.store(-1, std::memory_order_release);
    if (!framelog_path()) return false;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {kLogW, kLogH, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (!create_image(ci, log_image, log_image_alloc)) {
        framelog_resources_drop();
        logf("vk: frame log disabled: the readback image could not be made");
        return false;
    }
    for (LogSlot& l : log_slots) {
        if (!create_buffer_cached((uint64_t)kLogW * kLogH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, l.buf, l.alloc, &l.map) || !l.map) {
            framelog_resources_drop();
            logf("vk: frame log disabled: the complete mapped readback set could not be made");
            return false;
        }
    }
    FILE* f = fopen(framelog_path(), "wb");
    if (!f) {
        framelog_resources_drop();
        logf("vk: frame log disabled: cannot open %s", framelog_path());
        return false;
    }
    // Other presenters are serialized by present_image_mu. Publish before the worker can report a failure.
    log_state.store(1, std::memory_order_release);
    try {
        std::thread([f] { framelog_write(f); }).detach();
    } catch (const std::exception& e) {
        log_state.store(-1, std::memory_order_release);
        fclose(f);
        framelog_resources_drop();
        logf("vk: frame log disabled: writer thread could not start (%s)", e.what());
        return false;
    }
    logf("vk: the frame log writes every frame shown to %s", framelog_path());
    return true;
}
bool framelog_on() {
    static bool on = false;
    static int64_t next = 0;
    if (!framelog_path()) return false;
    const int64_t now = prof_now();
    if (now >= next) {
        next = now + 250000000;
        const std::string go = std::string(framelog_path()) + ".go";
        on = GetFileAttributesA(go.c_str()) != INVALID_FILE_ATTRIBUTES;
    }
    return on && framelog_ready();
}
void framelog_collect(int slot) {
    LogSlot& l = log_slots[slot];
    if (!l.used) return;
    l.used = false;
    // present_image waited for this slot's fence; expose its GPU writes before the CPU copy.
    invalidate_mapped(l.alloc, (uint64_t)kLogW * kLogH * 4);
    std::vector<uint8_t> rec(16 + (size_t)kLogW * kLogH * 4);
    std::memcpy(rec.data(), &l.at, 8);
    std::memcpy(rec.data() + 8, &l.seq, 4);
    std::memcpy(rec.data() + 12, &l.kind, 4);
    std::memcpy(rec.data() + 16, l.map, (size_t)kLogW * kLogH * 4);
    {   // (a picture blown out to white: what the exposure was)
        const uint8_t* q = rec.data() + 16;
        uint64_t sum = 0;
        const size_t n = (size_t)kLogW * kLogH;
        for (size_t i = 0; i < n; i += 7) sum += q[i * 4 + 1];
        const uint64_t mean = sum / (n / 7 + 1);
        static uint64_t last_mean;
        if (mean > 225 && last_mean < 215 && l.kind == 0) {
            logf("vk: white frame: frame log sequence %u, present %lld, brightness %llu after %llu", l.seq, (long long)presents,
                 (unsigned long long)mean, (unsigned long long)last_mean);
            char line[700];
            int n2 = snprintf(line, sizeof line, "vk: white detail: textures made by present (now %lld):", (long long)made_ring_present);
            for (int k = 30; k >= 0 && n2 < 660; k--) {
                const int64_t at = made_ring_present - k;
                const uint64_t a = made_ring[(at - 1) & 31], b = made_ring[at & 31];
                if (b > a) n2 += snprintf(line + n2, sizeof line - n2, " -%d:%llu", k, (unsigned long long)(b - a));
            }
            logf("%s", line);
            exposure_log_dump("white frame");
            probe_dump_all("white frame");
        }
        if (l.kind == 0 && mean * 2 < last_mean && last_mean > 40) {   // (a picture gone dark for a frame, the HUD on it)
            static int dark_left = 6;
            if (dark_left > 0) {
                dark_left--;
                logf("vk: dark frame: frame log sequence %u, present %lld, brightness %llu after %llu", l.seq, (long long)presents,
                     (unsigned long long)mean, (unsigned long long)last_mean);
                char name[32];
                snprintf(name, sizeof name, "cond_%d.txt", dark_left);
                if (FILE* o = fopen(name, "w")) {
                    for (uint32_t i = 0; i < (1 << 15); i++) {
                        const CondNote& c = cond_ring[(cond_ring_at + i) % (1 << 15)];
                        if (c.addr)
                            fprintf(o, "%lld %#llx %d %llu %llu %d%c", (long long)c.present, (unsigned long long)c.addr, c.mode,
                                    (unsigned long long)c.x, (unsigned long long)c.y, c.on ? 1 : 0, 10);
                    }
                    fclose(o);
                }
                exposure_log_dump("dark frame");
                probe_dump_all("dark frame");
                pass_traces_log(texture_frame_no() - 7, texture_frame_no());
            }
        }
        if (l.kind == 0) last_mean = mean;
    }
    static const bool drop = env_on("RX_VK_FRAMELOG_DROP");
    if (drop) return;
    framelog_enqueue(std::move(rec));
}
} // namespace
std::atomic<uint32_t> present_kind{0};   // (vk_framegen.cpp: 1 while a made frame goes out)

// (hookshot) One thread in here at a time; the queue's own lock (dev.present_mu, the replay's submits take it too) is held only
// for the submit and the present. Held over vkAcquireNextImageKHR and the fence, it stopped the replay's submits for as long as
// the swapchain had no image free, twice a frame with frame interpolation.
std::mutex present_image_mu;
static bool present_out(uint32_t index, std::unique_lock<std::mutex>& qk);   // (present_image's end, below)
thread_local PendingPresent* present_defer_into = nullptr;
static VkFence present_copy_done = VK_NULL_HANDLE;   // (present_image's copy, while it presents: the presenter thread may take it)
namespace {
bool present_thread_on();
void present_hand_off(uint32_t index, VkFence done);
void present_finish(VkResult res, bool on_presenter);
std::atomic<bool> presenter_surface_lost{false};   // (the presenter thread's present: VK_ERROR_SURFACE_LOST_KHR, for present_image)
}  // namespace
// doc: docs/design.md#13-frame-generations-presents-as-rules (rule 4: a frame slot per frame of both sets)
/// (hookshot) The frame slots present_image uses while frame generation runs, for the sets a swapchain is made for
/// (vk_main.cpp create_swapchain: framegen is swapchain_framegen's multiplier, 0 none; two_sets while RX_VK_FG_TIMELINE is
/// on, in every present mode, but for a set too big for the chain under FIFO). Each present waits for the copy last made in its slot, so the slots are
/// also how many presents can be on the GPU at once: 3, and a slot per frame of two sets where the next set is taken while
/// one is shown (2 x the multiplier, at most kPresentSlotsMax). A set presented ahead (vk_framegen.cpp display_main) submits
/// its copies at once: with fewer slots than its frames its 4th copy waited in its own first copy's slot, and with a slot per
/// frame of one set the next set's copy k waited in the slot of the shown set's copy k (the review, 2026-11-17), each until
/// the GPU had done that copy and so the game's work queued before it, which held the frames of the set still being shown.
/// One set's (RX_VK_FG_TIMELINE=0; under FIFO a set too big for the chain) keeps 3, as before
int present_slots_for(int framegen, bool two_sets) {
    return two_sets && framegen >= 2 ? std::min(2 * framegen, kPresentSlotsMax) : 3;
}

// (hookshot, #1376) The swapchain's image is waited for on the CPU, never on the replay's queue. The copy into it waited on the
// GPU for the acquire's semaphore, on the queue every submission of the game's work goes to, so whatever held the image held
// the replay. AMD's driver on Windows hands an image back from vkAcquireNextImageKHR at once and leaves the wait to the
// semaphore; with the game's fullscreen window behind another one (a browser, OBS: RX 9070 XT, issues 316, 1355, 1376) the
// system gave the image back late or never, the queue finished nothing, and the GPU hang watch ended the game 30 s later
// ("no submission finished in 30 s, and the driver never reported the device lost", no TDR). Now the acquire signals
// acquire_fence; a copy goes to the queue only once the image is free; an image not free in kImageFreeNs is held (frames are
// not shown meanwhile, the game goes on) and looked at again, without waiting, at each present until it is
static VkFence acquire_fence = VK_NULL_HANDLE;
struct HeldImage {
    bool held = false;
    uint32_t index = 0;
    uint64_t generation = 0;   // (dev.swapchain_generation it was acquired from)
};
static HeldImage held_image;
// (250 ms: past a refresh of a 24 Hz display for each image FIFO may hold queued, at frame generation's most; far under the GPU
// hang watch's 30 s)
constexpr uint64_t kImageFreeNs = 250000000ull;
// (hookshot) An image held this long is taken as never coming back, and the swapchain is made again: then twice as
// long, and at most 4 s, between the remakes that brought none back. With frame generation on an RX 9070 XT (Windows, full
// screen) an image held after Alt+Tab was not given back once the game was in front again either, until F11 made the
// swapchain again: the game and its sound went on, its picture stayed as it was. The system lets an image go when another
// is shown, and while one is held nothing else is presented, so nothing ever let it go. A new swapchain's images are all
// free; while the window stays hidden its first image is held again, and the remakes are kept far apart
constexpr int64_t kHeldRemakeNs = 1000000000ll;
/// Whether an image held for held_ns, after `remakes` swapchains made again for holds that brought no image back, is let go
/// with its swapchain
static bool held_remake_due(int64_t held_ns, int remakes) {
    return held_ns >= kHeldRemakeNs << (remakes < 2 ? remakes : 2);
}
/// Whether the acquired image is free (acquire_fence signaled, and reset for the next acquire), waiting up to timeout_ns
static bool image_free(uint64_t timeout_ns) {
    if (!acquire_fence) return false;   // (none could be made again: held_image_drop)
    VkResult r;
    {
        const gpu_wait_stats::Exclude pacing;   // (the display's pace, not the GPU's work: not counted as a GPU wait)
        r = vkWaitForFences(dev.device, 1, &acquire_fence, VK_TRUE, timeout_ns);
    }
    if (r == VK_ERROR_DEVICE_LOST) device_lost();
    if (r != VK_SUCCESS) return false;   // (VK_TIMEOUT: not yet)
    if (vkResetFences(dev.device, 1, &acquire_fence) == VK_ERROR_DEVICE_LOST) device_lost();
    return true;
}
/// A held image of a swapchain made again since: let go. Its fence may never be signaled now; then a new one takes its place
/// (the old one is left as it is, not destroyed: the presentation engine may still signal it). With none made, no acquire
/// until one is (acquire_fence_ready)
static void held_image_drop() {
    held_image.held = false;
    if (image_free(0)) return;
    acquire_fence = VK_NULL_HANDLE;
}
/// Whether there is an unsignaled acquire fence to acquire with, made again here when held_image_drop let the last one go
static bool acquire_fence_ready() {
    if (acquire_fence) return true;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence made = VK_NULL_HANDLE;
    if (vkCreateFence(dev.device, &fi, nullptr, &made) != VK_SUCCESS) return false;
    acquire_fence = made;
    return true;
}

void present_image(VkImage src, uint32_t sw, uint32_t sh, const int crop[4], VkImageView view) {
#ifdef TRACY_ENABLE
    HT_SCOPE("present_image");
#endif
    std::lock_guard lk(present_image_mu);
    static int hdr_was = -1;   // (the Insert menu's HDR as the swapchain was made for it; -1: not yet looked at)
    const bool hdr_changed = hdr_was >= 0 && hdr_was != opts.hdr;   // (turned on or off: the swapchain's format)
    if (hdr_was < 0) hdr_was = opts.hdr;
    static int vsync_was = -1;   // (the V-Sync row as the swapchain was made for it: its present mode)
    // (and frame generation's multiplier, swapchain_framegen: a spare image a frame; and swapchain_dlssg: FIFO under V-Sync
    // Auto with DLSS Frame Generation making the frames. It turns to no once, when NGX's answer at the first present is a no
    // or fewer frames than the multiplier asks; and past 4x the multiplier itself: under FIFO a set too big for the chain's
    // spare images takes one set's images, create_swapchain. The multiplier as it runs, framegen_now: DLSS FG's top with a
    // saved value past it)
    const int framegen = framegen_now();
    const int vsync_now = menu_vsync() * 16 + swapchain_framegen() * 2 + (swapchain_dlssg() ? 1 : 0) +
                          (framegen > 4 ? framegen * 256 : 0);
    const bool vsync_changed = vsync_was >= 0 && vsync_was != vsync_now;
    if (vsync_was < 0) vsync_was = vsync_now;
    if (presenter_surface_lost.exchange(false, std::memory_order_acquire)) {   // (the presenter's present found the surface lost)
        present_thread_drain();   // (before the queue's lock: the presenter takes it)
        std::lock_guard qk(dev.present_mu);
        invalidate_lost_surface(VK_ERROR_SURFACE_LOST_KHR);
        return;
    }
    const uint64_t want = dev.swap_resize.exchange(0, std::memory_order_acquire);
    if (want || !dev.swapchain || reflex_swapchain_stale() || hdr_changed || vsync_changed) {   // (resized, gone, Reflex, HDR or V-Sync changed)
        present_thread_drain();   // (before the queue's lock: the presenter takes it)
        std::lock_guard qk(dev.present_mu);
        const uint32_t width = want ? (uint32_t)(want >> 32) : dev.swap_extent.width;
        const uint32_t height = want ? (uint32_t)want : dev.swap_extent.height;
        if (!create_swapchain(width, height)) {
            static int64_t failed;   // (a frame not shown for want of a swapchain, said as the acquire's are)
            if (++failed <= 4 || failed % 600 == 0)
                logf("vk: a frame not shown: no swapchain of %ux%u (%lld frames so far)", width, height, (long long)failed);
            // Retry without replacing a newer resize published while recreation was in progress.
            uint64_t empty = 0;
            dev.swap_resize.compare_exchange_strong(empty, ((uint64_t)width << 32) | height,
                                                   std::memory_order_release, std::memory_order_relaxed);
            return;   // (the old swapchain may already be retired; do not acquire or reset a frame fence)
        }
        hdr_was = opts.hdr, vsync_was = vsync_now;
    }
    // Resize or surface recovery can replace the chain after the display thread chose present-ahead.
    // A deferred pair needs two spare images before either one can be presented.
    if (present_defer_into && dev.swap_spare.load(std::memory_order_relaxed) < 2) return;
    if (src) {   // (the Window scaling filter's row: whether the window is the picture's size. Only the game's frames and the
                 // made ones: the start-up screen (no src) is drawn at the window's own size, which made the row read "Not
                 // needed" while the game's output was another size; until the first frame the size is not known: needed)
        const bool cropped = crop && crop[2] > 0 && crop[3] > 0;
        menu_note_present_sizes(cropped ? (uint32_t)crop[2] : sw, cropped ? (uint32_t)crop[3] : sh, dev.swap_extent.width,
                                dev.swap_extent.height);
    }
    menu_tick();
    // (the slots in turn: the swapchain's count while frame generation runs (create_swapchain: present_slots_for), else 3; a
    // slot past them waits with its fence signalled. Where the count went down, the frame log's pictures in the slots past it
    // are dropped (RX_VK_FRAMELOG, at most kPresentSlotsMax - 3 of them): read only once the count came back up, they went
    // in late and out of order)
    const uint32_t slots = framegen_on() ? std::clamp<uint32_t>(dev.swap_slots.load(std::memory_order_relaxed), 3u,
                                                                 (uint32_t)kPresentSlotsMax) : 3u;
    static uint32_t slots_was = 3;
    for (uint32_t k = slots; k < slots_was; k++) log_slots[k].used = false;
    slots_was = slots;
    const uint32_t slot = frame_index % slots;
    Frame& f = frames[slot];
    const int log_slot = (int)slot;
    frame_index = (slot + 1) % slots;
    // (the presenter thread, RX_VK_PRESENT_THREAD: the present handed over before this one is made first. The swapchain is one
    // thread's at a time: vkAcquireNextImageKHR below ran beside the presenter's vkQueuePresentKHR and vkWaitForPresentKHR of
    // the frame before, which waited for it only at this frame's hand-over, after the acquire. One present out at most, as
    // before; with the presenter off or idle, nothing)
    present_thread_drain();
    const int64_t fence_t0 = fps_breakdown_on() ? fps_clock() : 0;   // (the FPS counter's Breakdown: this frame slot's last copy)
    if (vkWaitForFences(dev.device, 1, &f.done, VK_TRUE, UINT64_MAX) == VK_ERROR_DEVICE_LOST) device_lost();
    framelog_collect(log_slot);
    uint32_t index = 0;
    const int64_t acquire_t0 = fence_t0 ? fps_clock() : 0;   // (then the swapchain's image: the display's pace)
    if (fence_t0) fps_wait(fps_stats::kGpuWait, acquire_t0 - fence_t0);
#ifdef _WIN32
    const uint64_t acquire_timeout = UINT64_MAX;
#else
    // (linux: a Wayland compositor gives no image back while the window cannot be seen: its screen asleep, another workspace. A wait
    // without end there held the presentation thread, and with it the game, until the screen came back. 100 ms, then the window is
    // taken as not shown: frames are not waited for and not shown, and the game goes on at its speed, as on Windows, until an image
    // comes back)
    static bool unseen = false;
    const uint64_t acquire_timeout = unseen ? 0 : 100000000ull;
#endif
    if (held_image.held && held_image.generation != dev.swapchain_generation) held_image_drop();   // (its swapchain is gone)
    const bool was_held = held_image.held;
    if (!was_held && !acquire_fence_ready()) return;   // (no fence to acquire with: the frame is not shown)
    if (was_held) index = held_image.index;   // (acquired at an earlier present, not free then: no other acquire meanwhile)
    VkResult res = VK_SUCCESS;
    if (!was_held) res = vkAcquireNextImageKHR(dev.device, dev.swapchain, acquire_timeout, VK_NULL_HANDLE, acquire_fence, &index);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    if (acquire_t0) fps_wait(fps_stats::kPace, fps_clock() - acquire_t0);
    if (res == VK_ERROR_SURFACE_LOST_KHR) {
        present_thread_drain();   // (before the queue's lock: the presenter takes it)
        std::lock_guard qk(dev.present_mu);
        invalidate_lost_surface(res);
        return;
    }
#ifndef _WIN32
    if (res == VK_TIMEOUT || res == VK_NOT_READY) {
        if (!unseen) logf("vk: the window is not shown (no image back in 100 ms): frames are not shown until it is");
        unseen = true;
        return;
    }
    if (unseen && (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR)) {
        unseen = false;
        logf("vk: the window is shown again");
    }
#endif
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
        // (a frame not shown, said: in two runs on a PC the game went on at 90 fps for 30 s with
        // nothing presented after a swapchain was made again, and nothing in the log said so
        static int64_t unshown;
        if (++unshown <= 4 || unshown % 600 == 0)
            logf("vk: a frame not shown: the image acquire gave %d (%lld frames so far)", (int)res, (long long)unshown);
    }
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        present_thread_drain();
        std::lock_guard qk(dev.present_mu);
        create_swapchain(dev.swap_extent.width, dev.swap_extent.height);
        return;
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) return;
    {   // (#1376: the image free before any work for it goes to the queue; a held one is looked at without waiting)
        const int64_t free_t0 = fence_t0 ? fps_clock() : 0;
        const bool ready = image_free(was_held ? 0 : kImageFreeNs);
        if (free_t0) fps_wait(fps_stats::kPace, fps_clock() - free_t0);
        static int said;     // (the first 16 holds are said, and their ends)
        static bool told;    // (this hold was said)
        static int64_t held_since;   // (when this hold began, fps_clock)
        static int remakes, said_remakes;   // (swapchains made again for holds since an image last came back; the first 16 said)
        if (!ready) {
            const int64_t now = fps_clock();
            if (!was_held) {
                held_since = now;
                told = said < 16;
                if (told) said++, logf("vk: the window's image was not given back in 250 ms (the window hidden?): frames are not "
                                       "shown until it is");
            }
            held_image = {true, index, dev.swapchain_generation};
            if (held_remake_due(now - held_since, remakes)) {   // (never given back; a new swapchain, as F11 makes)
                if (said_remakes < 16)
                    said_remakes++, logf("vk: the window's image was not given back in %.0f s: the swapchain made again",
                                         (now - held_since) / 1e9);
                if (remakes < 1000) remakes++;
                present_thread_drain();   // (before the queue's lock: the presenter takes it)
                std::lock_guard qk(dev.present_mu);
                create_swapchain(dev.swap_extent.width, dev.swap_extent.height);   // (the held image let go at the next present)
                held_since = now;   // (not made, a window with no size: tried again only after the next wait, not at each present)
            }
            return;
        }
        held_image.held = false;
        remakes = 0;
        if (was_held && told) told = false, logf("vk: the window's image was given back: frames are shown again");
    }
    if (vkResetFences(dev.device, 1, &f.done) == VK_ERROR_DEVICE_LOST) device_lost();
    if (vkResetCommandPool(dev.device, f.pool, 0) == VK_ERROR_DEVICE_LOST) device_lost();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(f.cmd, &bi) == VK_ERROR_DEVICE_LOST) device_lost();
    VkImage dst = dev.swap_images[index];
    full_barrier(f.cmd);   // (after the game's work for the frame: earlier submits on this queue)
    image_barrier(f.cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    const PresentAspect aspect{src && host && host->aspect_ratio ? host->aspect_ratio() : 0.0f};
    const PresentRect rect = present_rect(sw, sh, crop, dev.swap_extent.width, dev.swap_extent.height, aspect);
    [[maybe_unused]] uint32_t trace_route = 0; // direct; 1 scaled, 2 clear, 3 HDR conversion
    const bool has_bars = rect.x || rect.y || rect.w != dev.swap_extent.width || rect.h != dev.swap_extent.height;
    if (has_bars) {
        const VkClearColorValue black{0.0f, 0.0f, 0.0f, 1.0f};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(f.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        // The following blit overlaps the clear in the content rectangle.
        image_barrier(f.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }
    // (Native HDR: scaled in 10 bits (framegen_scale's pq), and read by the HDR pass in every case, through the view in the
    // format the game presents (do_present): at 100% contrast it writes the game's PQ codes as they are. Never blitted from its
    // image: a blit converts from the image's own format, not the picture's)
    const bool native_pq = present_native_pq.load(std::memory_order_relaxed);
    const VkImage scaled = src && view ? framegen_scale(f.cmd, view, src, sw, sh, crop, rect.w, rect.h, native_pq) : VK_NULL_HANDLE;
    // (hookshot) the Insert menu's HDR: the picture (scaled, or the game's own with its crop) to HDR10, copied in as the SDR one is
    const VkImage hdr_image = dev.hdr && src && view
        ? (scaled ? framegen_hdr(f.cmd, framegen_scaled_view(), rect.w, rect.h, nullptr, rect.w, rect.h, native_pq)
                  : framegen_hdr(f.cmd, view, sw, sh, crop, rect.w, rect.h, native_pq))
        : VK_NULL_HANDLE;
    if (hdr_image) {
        trace_route = 3;
        full_barrier(f.cmd);
        blit_crop_to_rect(f.cmd, hdr_image, rect.w, rect.h, nullptr, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, rect);
    } else if (scaled) {
        trace_route = 1;
        blit_crop_to_rect(f.cmd, scaled, rect.w, rect.h, nullptr, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, rect);
    } else if (src && !native_pq) {
        blit_crop_to_rect(f.cmd, src, sw, sh, crop, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, rect);
    } else {
        if (src) {   // (Native, unscaled, and the HDR pass not made: black, said once, rather than the picture converted)
            static bool told;
            if (!told) told = true, logf("vk: HDR: Native's picture could not go through the HDR pass: shown black");
        }
        trace_route = 2;
        const VkClearColorValue black{};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(f.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    }
    if (src && framelog_on()) {
        static uint32_t seq;
        LogSlot& l = log_slots[log_slot];
        image_barrier(f.cmd, log_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        blit_crop(f.cmd, src, sw, sh, crop, log_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kLogW, kLogH);
        image_barrier(f.cmd, log_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        const VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {kLogW, kLogH, 1}};
        vkCmdCopyImageToBuffer(f.cmd, log_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, l.buf, 1, &c);
        capture_host_read_barrier(f.cmd, l.buf, (VkDeviceSize)kLogW * kLogH * 4);
        l.used = true, l.at = host->now_ns(), l.seq = seq++, l.kind = present_kind.load();
    }
    // Geometry trace: after choosing the actual presentation route, without frame-log pixel readback.
    static const bool trace_geometry = getenv("RX_VK_PRESENT_LOG") && atoi(getenv("RX_VK_PRESENT_LOG")) != 0;
    if (trace_geometry) {
        static PresentGeometryTrace trace;
        PresentGeometrySample sample;
        const PresentContentSize content = present_content_size(sw, sh, crop);
        sample.source = {sw, sh};
        sample.content = {content.w, content.h};
        sample.output = {dev.swap_extent.width, dev.swap_extent.height};
        sample.destination = {rect.x, rect.y, rect.w, rect.h};
        sample.crop_supplied = crop != nullptr;
        if (crop) std::copy_n(crop, 4, sample.crop.begin());
        sample.sampled_view = view != VK_NULL_HANDLE;
        sample.host_aspect(src && host && host->aspect_ratio, aspect.ratio);
        sample.kind = src ? std::min(present_kind.load(), 1u) : 2;
        sample.route = trace_route;
        sample.filter = opts.upscale;
        sample.sharpness = opts.sharpness;
        sample.framegen = opts.framegen;
        const PresentTraceDecision decision = trace.note(sample);
        if (decision == PresentTraceDecision::Emit) {
            const char* kinds[] = {"game", "made", "startup"};
            const char* routes[] = {"direct", "scaled", "clear", "hdr"};
            const char* aspects[] = {"unavailable", "invalid", "valid"};
            logf("vk: present geometry: kind=%s source=%ux%u crop=%s:%d,%d %dx%d content=%ux%u output=%ux%u "
                 "destination=%u,%u %ux%u host-aspect=%s:%.9g route=%s view=%d filter-request=%d sharpness-request=%d framegen-request=%d",
                 kinds[sample.kind], sw, sh, crop ? "supplied" : "none", sample.crop[0], sample.crop[1], sample.crop[2], sample.crop[3],
                 content.w, content.h, sample.output[0], sample.output[1], rect.x, rect.y, rect.w, rect.h,
                 aspects[sample.aspect_state], sample.aspect, routes[sample.route], int(sample.sampled_view), sample.filter,
                 sample.sharpness, sample.framegen);
        } else if (decision == PresentTraceDecision::Limit) {
            logf("vk: present geometry: limit of %u changes reached; further geometry messages suppressed", PresentGeometryTrace::kLimit);
        }
    }
    menu_record(f.cmd, index, dev.swap_extent.width, dev.swap_extent.height);   // (hookshot) Dear ImGui's overlays, over all of it
    menu_rumble_tick();   // (hookshot: the Rumble strength row's own rumble, vk_menu.cpp)
    image_barrier(f.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    if (vkEndCommandBuffer(f.cmd) == VK_ERROR_DEVICE_LOST) device_lost();
    // (no wait: the image was free before the copy was recorded, image_free above. Not on the timeline either: one queue runs
    // submits in order, and a wait for a value another thread had counted but not yet submitted blocked the queue for good)
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = dev.swap_rendered[index];
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cb.commandBuffer = f.cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cb;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &signal;
    std::unique_lock qk(dev.present_mu);
    if (!pend.cbs.empty()) flush_reason = FR_PRESENT;
    queue_flush_locked();
    static int64_t presents_counted, st_flushes0, st_cbs0, r0[FR_COUNT];
    if (st_flushes && ++presents_counted % 120 == 0) {
        logf("vk: batched submits: %lld flushes, %.1f command buffers each, over 120 presents; by reason: poll %lld, wait %lld, "
             "slot %lld, present %lld, cap %lld, value %lld, other %lld", (long long)(st_flushes - st_flushes0),
             (double)(st_batched_cbs - st_cbs0) / std::max<int64_t>(1, st_flushes - st_flushes0),
             (long long)(st_reason[FR_POLL] - r0[FR_POLL]), (long long)(st_reason[FR_WAIT] - r0[FR_WAIT]),
             (long long)(st_reason[FR_SLOT] - r0[FR_SLOT]), (long long)(st_reason[FR_PRESENT] - r0[FR_PRESENT]),
             (long long)(st_reason[FR_CAP] - r0[FR_CAP]), (long long)(st_reason[FR_VALUE] - r0[FR_VALUE]),
             (long long)(st_reason[FR_OTHER] - r0[FR_OTHER]));
        st_flushes0 = st_flushes, st_cbs0 = st_batched_cbs;
        std::memcpy(r0, st_reason, sizeof r0);
    }
    reflex_submit_ends();
    // (NVIDIA Reflex: the frame this present shows; with frame generation the set's game frame, none for a made frame)
    si.pNext = reflex_submit_id(nullptr, reflex_present_frame);
    res = submit_or_retry(si, f.done);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    if (res != VK_SUCCESS) {
        // (the present below waits for the semaphore this submit signals: with the submit refused it would wait for good, on the
        // presents' own queue unseen by anything that waits on the replay's queue)
        static int logged;
        if (logged++ < 8) logf("vk: the present's blit submit failed (VkResult %d): this frame is not shown", (int)res);
        return;
    }
    if (present_defer_into) {   // (present ahead: the copy is queued; present_commit shows it at its time)
        present_defer_into->used = true;
        present_defer_into->index = index;
        present_defer_into->done = f.done;
        present_defer_into->swapchain = dev.swapchain;
        present_defer_into->reflex_frame = reflex_present_frame;   // (NVIDIA Reflex: the frame it shows, at its present)
        present_defer_into->antilag_frame = antilag_present_frame;   // (AMD Anti-Lag 2: the same)
        present_defer_into->swapchain_generation = dev.swapchain_generation;
        return;
    }
    present_copy_done = f.done;
    present_out(index, qk);
    present_copy_done = VK_NULL_HANDLE;
}

bool present_commit(const PendingPresent& p) {
    if (!p.used) return false;
    std::lock_guard lk(present_image_mu);
    std::unique_lock qk(dev.present_mu);
    if (p.swapchain != dev.swapchain || p.swapchain_generation != dev.swapchain_generation) return false;   // (made again since the copy)
    const uint64_t was = reflex_present_frame, al_was = antilag_present_frame;
    reflex_present_frame = p.reflex_frame;   // (NVIDIA Reflex: the frame this present shows, as at its copy)
    antilag_present_frame = p.antilag_frame;   // (AMD Anti-Lag 2: the same)
    const bool shown = present_out(p.index, qk);
    reflex_present_frame = was, antilag_present_frame = al_was;
    return shown;
}

// The present of a swapchain image whose copy is submitted, as present_image ended before; qk holds dev.present_mu
static bool present_out(uint32_t index, std::unique_lock<std::mutex>& qk) {
#ifdef TRACY_ENABLE
    HT_SCOPE("present_out");
#endif
    VkResult res;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.pNext = reflex_present_id(nullptr);   // (NVIDIA Reflex: the frame's present id, and its present's markers)
    if (present_copy_done && !pi.pNext && present_thread_on()) {   // (RX_VK_PRESENT_THREAD: below)
        antilag_present();   // (AMD Anti-Lag 2: the frame's present stage)
        qk.unlock();
        present_hand_off(index, present_copy_done);
        return true;   // (taken as shown: only present_commit's presents, never handed over, read it)
    }
    if (present_thread_on()) {   // (one handed over before, made first: the presents in order; the queue's lock let go meanwhile)
        qk.unlock();
        present_thread_drain();
        qk.lock();
    }
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &dev.swap_rendered[index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &dev.swapchain;
    pi.pImageIndices = &index;
    antilag_present();   // (AMD Anti-Lag 2: the frame's present stage)
    // (hookshot) vkQueuePresentKHR blocked a quarter of the display thread's time with frame interpolation at 4x (a 14700K and an
    // RTX 5070 Ti), and with the queue's lock held there the game's main thread waited ~9% of its time and the presentation thread
    // ~15% to submit. With a queue of their own (dev.present_queue, a compute family's: vk_main.cpp create_device) the presents go
    // out with the submits' lock let go: the blit above signals swap_rendered on the replay's queue (submitted already, as a binary
    // semaphore's wait needs). (Waiting for the blit's fence before the present, with the lock let go, was worse.)
    // (the FPS counter's Breakdown: the present's wait is vsync's, the pacing's; unless the frame's copy was not done when it was
    // called and was when it returned. Then the present waited for the GPU: NVIDIA's driver on Linux waits there for the copy, after
    // all of the frame's work, ~2 ms a frame on an RTX 3090. Two reads of its fence, while the Breakdown shows; nothing waits)
    const VkFence copy = fps_breakdown_on() ? present_copy_done : VK_NULL_HANDLE;
    // (a wait of no time: the fence's state. Not through the recorder's boundary: nothing recorded is behind a fence's state)
    const auto copy_done = [&] { return (::vkx::vkWaitForFences)(dev.device, 1, &copy, VK_TRUE, 0) == VK_SUCCESS; };
    const bool copy_was_done = copy && copy_done();
    const int64_t present_t0 = fps_breakdown_on() ? fps_clock() : 0;
    if (dev.present_queue_supported) {
        qk.unlock();
        std::lock_guard pq(dev.present_queue_mu);
        res = vkQueuePresentKHR(dev.present_queue, &pi);
    } else {
        res = vkQueuePresentKHR(dev.queue, &pi);
    }
    if (present_t0) {
        const bool waited_gpu = copy && !copy_was_done && copy_done();
        fps_wait(waited_gpu ? fps_stats::kGpuWait : fps_stats::kPace, fps_clock() - present_t0);
    }
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    reflex_presented();
    if (res == VK_ERROR_DEVICE_LOST) device_lost();   // (on the presents' own queue this is the first place a lost device shows)
    if (res == VK_ERROR_SURFACE_LOST_KHR) {
        if (!qk.owns_lock()) qk.lock();
        invalidate_lost_surface(res);
        return false;
    }
    {   // (every 600 presents: how they ended; a window shot is taken before the present, so a black window shows only here)
        static int64_t n_ok, n_subopt, n_old, n_other;
        (res == VK_SUCCESS ? n_ok : res == VK_SUBOPTIMAL_KHR ? n_subopt : res == VK_ERROR_OUT_OF_DATE_KHR ? n_old : n_other)++;
        if ((n_ok + n_subopt + n_old + n_other) % 600 == 0)
            logf("vk: presents%s: %lld shown, %lld suboptimal, %lld out of date, %lld failed", dev.present_queue_supported ? " (own queue)" : "",
                 (long long)n_ok, (long long)n_subopt, (long long)n_old, (long long)n_other);
    }
    present_finish(res, false);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        logf("vk: present result %d: the swapchain is made again", (int)res);
        if (!qk.owns_lock()) qk.lock();
        create_swapchain(dev.swap_extent.width, dev.swap_extent.height);
    }
    return res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR;
}
namespace {
/// What follows a present, on the thread that made it. The presenter thread asks the next present_image for a new swapchain (its
/// swap_resize) rather than making one itself
void present_finish(VkResult res, bool on_presenter) {
#ifdef TRACY_ENABLE
    HT_SCOPE("present_finish");
#endif
    if (res < 0 && res != VK_ERROR_OUT_OF_DATE_KHR) {   // (a present that failed: said, a few times)
        static int logged;
        if (logged++ < 8) logf("vk: present failed (VkResult %d%s)", (int)res, dev.present_queue_supported ? ", on the presents' own queue" : "");
    }
    frame_log_present();
    if (on_presenter && res == VK_ERROR_SURFACE_LOST_KHR) presenter_surface_lost.store(true, std::memory_order_release);
    if (on_presenter && (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)) {   // (the size it has: made again as it is)
        uint64_t empty = 0;
        dev.swap_resize.compare_exchange_strong(empty, (uint64_t)dev.swap_extent.width << 32 | dev.swap_extent.height,
                                                std::memory_order_release, std::memory_order_relaxed);
    }
}
}  // namespace

// (hookshot) The presenter thread (RX_VK_PRESENT_THREAD=1; off by default: alone it gains nothing, the game's frame wait holds
// the game for the same GPU work, docs/development/frame-pipeline.md). With NVIDIA's driver on Wayland, vkQueuePresentKHR waits
// on the CPU until the frame's copy into the window's image is done on the GPU, after all of the frame's work: 2.4 ms a frame of the
// game's presentation thread on an RTX 3090 in Kokiri Forest (HOS_CHAIN), on the frame's critical path, since the game's main thread
// waits for that thread's present to return before it starts the next frame's commands. present_image hands the present to a thread
// of its own, which waits for the copy's fence, then presents on a queue no other thread uses (dev.presenter_queue, vk_main.cpp
// create_device: it never takes the replay's queue lock), and the game's thread goes on at once.
// - One present out at most: the next frame's acquire waits until it is made (present_image: the swapchain is one thread's at a
//   time); under FIFO (V-Sync On), with VK_KHR_present_wait, until it is shown. No frame is buffered past what the game's own
//   present gave: the game is never further ahead than before.
// - The presents in order; a swapchain is made again or dropped only with the presenter idle (present_thread_drain, before the
//   queue's lock).
// - Only the presents made right after their copy (present_image; not frame interpolation's, made by present_commit), and NVIDIA
//   Reflex off (its markers are the game thread's). A device without a second queue that can present keeps them on the game's thread.
// - No wait without end: the copy's fence and the present wait in steps, a device lost ends them, and a drain gives up after 5 s.
namespace {
struct Presenter {
    std::mutex mu;
    std::condition_variable cv;
    bool busy = false;               // (a present handed over and not yet made)
    uint32_t index = 0;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFence done = VK_NULL_HANDLE;   // (the copy's fence)
    int64_t copy_done_at = 0;        // (fps_clock() when the presenter's wait for it ended: the FPS counter's Breakdown)
    bool started = false;
};
Presenter presenter;
/// The copy's fence, in 100 ms steps: false after 4 s (a GPU that stopped: the present is not made) or a device lost. Less
/// than present_thread_drain's 5 s: a drain that gave up at the moment the presenter did went on to remake the swapchain while
/// the presenter still held it
bool presenter_wait_copy(VkFence done) {
#ifdef TRACY_ENABLE
    HT_WAIT(ht_w, "presenter_wait_copy");
#endif
    for (int step = 0; step < 40; step++) {
        const VkResult r = vkWaitForFences(dev.device, 1, &done, VK_TRUE, 100000000ull);
        if (r == VK_SUCCESS) return true;
        if (r == VK_ERROR_DEVICE_LOST) device_lost();
        if (r != VK_TIMEOUT) return false;
    }
    static int told;
    if (told++ < 4) logf("vk: presenter thread: a frame's copy not done in 4 s: its present not made");
    return false;
}
void presenter_loop() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk presenter");
#ifdef TRACY_ENABLE
    HT_THREAD_NAME("mkw_vk presenter");
#endif
    static const auto wait_for_present = dev.present_wait
        ? (PFN_vkWaitForPresentKHR)vkGetDeviceProcAddr(dev.device, "vkWaitForPresentKHR") : nullptr;
    VkSwapchainKHR ids_of = VK_NULL_HANDLE;   // (the swapchain the present ids count on: they start again on a new one)
    uint64_t next_id = 0;
    std::unique_lock lk(presenter.mu);
    for (;;) {
        presenter.cv.wait(lk, [] { return presenter.busy; });
        const uint32_t index = presenter.index;
        const VkSwapchainKHR sc = presenter.swapchain;
        const VkFence done = presenter.done;
        lk.unlock();
        const auto t0 = std::chrono::steady_clock::now();
        const bool copied = !done || presenter_wait_copy(done);
        const auto t1 = std::chrono::steady_clock::now();
        if (fps_breakdown_on()) {
            const int64_t at = fps_clock();
            std::lock_guard g(presenter.mu);
            presenter.copy_done_at = at;
        }
        VkResult res = VK_NOT_READY;
        double shown_ms = -1;
        if (copied) {
            if (sc != ids_of) ids_of = sc, next_id = 0;
            // (under FIFO the present is waited for until shown: the next frame is not handed over before, as the game's own
            // present held it at the swapchain before. Mailbox and immediate: as before, the present's return)
            const bool wait_shown = wait_for_present && dev.swap_mode == VK_PRESENT_MODE_FIFO_KHR;
            const uint64_t id = ++next_id;
            VkPresentIdKHR pid{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
            pid.swapchainCount = 1;
            pid.pPresentIds = &id;
            VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            pi.pNext = dev.present_wait ? &pid : nullptr;   // (ids on every present of the swapchain, so they go up)
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &dev.swap_rendered[index];
            pi.swapchainCount = 1;
            pi.pSwapchains = &sc;
            pi.pImageIndices = &index;
            {
                std::lock_guard pq(dev.presenter_queue_mu);
                res = vkQueuePresentKHR(dev.presenter_queue, &pi);
            }
            if (res == VK_ERROR_DEVICE_LOST) device_lost();
            if (wait_shown && (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR)) {
                const auto ts = std::chrono::steady_clock::now();
                VkResult w = VK_TIMEOUT;
                for (int step = 0; step < 5 && w == VK_TIMEOUT; step++) w = wait_for_present(dev.device, sc, id, 20000000ull);
                if (w == VK_ERROR_DEVICE_LOST) device_lost();
                if (w == VK_SUCCESS) shown_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
            }
        }
        {   // (what it waited for, every 600 presents)
            static double fence_ms, present_ms, shown_sum;
            static int n, shown_n;
            const auto t2 = std::chrono::steady_clock::now();
            fence_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            present_ms += std::chrono::duration<double, std::milli>(t2 - t1).count() - std::max(shown_ms, 0.0);
            if (shown_ms >= 0) shown_sum += shown_ms, shown_n++;
            if (++n % 600 == 0) {
                if (shown_n)
                    logf("vk: presenter thread: the copy's fence %.2f ms, the present %.2f ms, shown %.2f ms after it (%d of 600)",
                         fence_ms / 600, present_ms / 600, shown_sum / shown_n, shown_n);
                else
                    logf("vk: presenter thread: the copy's fence %.2f ms, the present %.2f ms a present", fence_ms / 600, present_ms / 600);
                fence_ms = present_ms = shown_sum = 0, shown_n = 0;
            }
        }
        if (copied) present_finish(res, true);
        lk.lock();
        presenter.busy = false;
        presenter.cv.notify_all();
    }
}
bool present_thread_on() {
    static const bool wanted = [] {   // (off by default: no gain alone, docs/development/frame-pipeline.md)
        const char* e = getenv("RX_VK_PRESENT_THREAD");
        return e && atoi(e) != 0;
    }();
    // (a queue of its own that presents to this surface: vk_main.cpp; a new surface checks again)
    const bool on = wanted && dev.presenter_queue && dev.presenter_supported;
    static int told = -1;
    if (told != (int)on) {
        told = on;
        logf("vk: presenter thread: %s", on ? "each present made on a thread and queue of its own once its copy is done (RX_VK_PRESENT_THREAD=1)"
                                            : wanted ? "no queue of its own that presents to the window: presents on the game's thread"
                                                     : "off (RX_VK_PRESENT_THREAD=1: on)");
    }
    return on;
}
/// Waits, with presenter.mu held in lk, until the presenter has made the present it was handed (limit_ms: at most that long; false
/// then). For the FPS counter's Breakdown, the wait is the GPU's until the presenter's wait for the copy ended, and the pacing's after it
/// (the present itself, vsync)
bool presenter_wait_idle(std::unique_lock<std::mutex>& lk, int limit_ms = 0) {
    if (!presenter.busy) return true;
    const int64_t t0 = fps_breakdown_on() ? fps_clock() : 0;
    const auto idle = [] { return !presenter.busy; };
    if (limit_ms) {
        if (!presenter.cv.wait_for(lk, std::chrono::milliseconds(limit_ms), idle)) return false;
    } else {
        presenter.cv.wait(lk, idle);
    }
    if (!t0) return true;
    const int64_t t1 = fps_clock();
    const int64_t gpu = presenter.copy_done_at ? std::clamp<int64_t>(presenter.copy_done_at - t0, 0, t1 - t0) : 0;
    fps_wait(fps_stats::kGpuWait, gpu);
    fps_wait(fps_stats::kPace, t1 - t0 - gpu);
    return true;
}
/// The present of image index, whose copy signals done, to the presenter thread (started at the first), once the one before is made
void present_hand_off(uint32_t index, VkFence done) {
#ifdef TRACY_ENABLE
    HT_SCOPE("present_hand_off");
#endif
    std::unique_lock lk(presenter.mu);
    if (!presenter.started) {
        presenter.started = true;
        ::mkwvk::cxx_escape::spawn_needed("presenter", presenter_loop);
    }
    presenter_wait_idle(lk);   // (the frame before's present made: one out at most; the presenter's own waits are bounded)
    presenter.index = index, presenter.swapchain = dev.swapchain, presenter.done = done, presenter.busy = true;
    presenter.copy_done_at = 0;
    presenter.cv.notify_all();
}
}  // namespace
/// Waits until the presenter thread has made the present it was handed (none: at once). Before a swapchain is made again or destroyed,
/// and never with dev.present_mu held (a resize must not wait on a present that waits for the window). Gives up after 5 s, said
void present_thread_drain() {
#ifdef TRACY_ENABLE
    HT_WAIT(ht_w, "present_thread_drain");
#endif
    std::unique_lock lk(presenter.mu);
    if (!presenter_wait_idle(lk, 5000)) logf("vk: presenter thread: its present not made in 5 s; going on without it");
}


void memory_precopy_present();   // (vk_memory.cpp, : RX_VK_PRECOPY_KICK=2)
namespace {

void set_texture_pool_size(int) {}
void read_texels(const nvn_tex* t, unsigned level, unsigned layer0, const NVNcopyRegion* r, uint8_t* dst, uint64_t row,
                 uint64_t img) {
    finish();   // (what the submits before it wrote)
    std::lock_guard lk(replay_mu);
    {   // (RX_VK_LOG_READ_TEXELS=1: the CPU reading textures back, 200 lines: are they ones the GPU wrote?)
        static const bool log_reads = env_on("RX_VK_LOG_READ_TEXELS");
        static int logged;
        if (log_reads && t && logged < 200) {
            logged++;
            Texture* tex = texture_get(t, false);
            logf("vk: ReadTexels %#llx %ux%ux%u NVN %#x level %u layer %u region (%d,%d,%d %dx%dx%d) row %llu: %s", (unsigned long long)t->gpu,
                 t->d.width, t->d.height, t->d.depth, t->d.format, level, layer0, r ? r->xoffset : 0, r ? r->yoffset : 0,
                 r ? r->zoffset : 0, r ? r->width : 0, r ? r->height : 0, r ? r->depth : 0, (unsigned long long)row,
                 !tex ? "no image" : tex->gpu_written ? (tex->released ? "rendered, finalized" : "RENDERED (guest memory is stale)")
                                                     : "uploaded");
        }
    }
    texture_read_texels(t, level, layer0, r, dst, row, img, false);
}
/// (host: virtual pages [gpu, gpu + size) are about to be unmapped or remapped) what the GPU wrote into textures there
/// goes into their memory first, as on the console, where it is there all along (vk_texture.cpp, texture_flush_to_guest)
bool replay_behind();
void vmap_changing(uint64_t gpu, uint64_t size) {
    if (!size) return;
    static const bool log_calls = env_on("RX_VK_LOG_VMAP_CHANGING");   // (each call, 300 lines)
    static std::atomic<int> logged{0};
    // (the list of GPU-written textures is complete only once the replay has caught up: MK World unmaps a tile's pool
    // a frame or two after it submitted the drawing, which the replay may not have recorded yet)
    const bool listed = texture_gpu_written_in(gpu, size);
    bool behind = false, listed_after = listed;
    if (!listed) {
        behind = replay_behind();
        if (behind) {
            wait_queue_empty();
            listed_after = texture_gpu_written_in(gpu, size);
        }
    }
    if (log_calls && logged++ < 300)
        logf("vk: vmap changing %#llx..%#llx: listed %d, replay behind %d, listed after %d", (unsigned long long)gpu,
             (unsigned long long)(gpu + size), (int)listed, (int)behind, (int)listed_after);
    if (!listed_after) return;
    wait_queue_empty();   // (the work the game recorded before: submitted)
    std::lock_guard lk(replay_mu);
    const int n = texture_flush_to_guest(gpu, size);
    if (log_calls && logged < 300) logf("vk: vmap changing %#llx: %d written back", (unsigned long long)gpu, n);
}
void write_texels(const nvn_tex* t, unsigned level, unsigned layer0, const NVNcopyRegion* r, const uint8_t* src,
                  uint64_t row, uint64_t img) {
    finish();   // (after the submits before it read the texture)
    std::lock_guard lk(replay_mu);
    texture_read_texels(t, level, layer0, r, const_cast<uint8_t*>(src), row, img, true);
}
void stats(int64_t* draws, int64_t* skipped, int64_t* pipelines) {
    int64_t v[4] = {0, 0, 0, 0};
    draw_stats(&v[0], &v[1], &v[2], &v[3]);
    if (draws) *draws = v[0];
    if (skipped) *skipped = v[1];
    if (pipelines) *pipelines = v[3];
}
void register_texture_locked(int id, const nvn_tex* t, const nvn_view* v) { register_texture(id, t, v); }
void do_program_released(const nvn_prog* p) {
    std::lock_guard lk(replay_mu);   // (the replay may be translating it)
    program_released(p);
}

// ---- the render thread (RX_VK_ASYNC=1; unset or 0, the default: none, the replay on the submitting thread) ----
// The game's submits and presents are queued and replayed by a thread of the back end's own, so the game's thread
// (MKW's Presentation Thread, 70% busy replaying before) goes on with its frame while the last one is recorded. As on
// the console, where the GPU reads the commands and the memory they name after the submit: the host resolves the
// deferred bindings at the submit and keeps the recordings until rx_nvn_replayed() passes them (nvn_native.c), and
// a fence value is handed out at the submit, signalled when the replay's submission completes. At most one present
// waits (RX_VK_PRESENTS_AHEAD): the game is held at that when the replay falls behind. Finalized textures and programs
// are queued too, in order with the work that may still use them.
enum JobKind { JOB_SUBMIT, JOB_PRESENT, JOB_TEXTURE_RELEASED, JOB_PROGRAM_RELEASED, JOB_REGISTER_TEXTURE, JOB_REGISTER_SAMPLER,
               JOB_EVENT_SIGNAL };
struct Job {
    JobKind kind;
    std::vector<uint64_t> handles;
    nvn_tex tex;
    bool has_tex = false;
    int id = 0;                 // (a registration's slot, its view and sampler)
    nvn_view view{};
    bool has_view = false;
    nvn_samp_par samp{};
    uint64_t ev_gpu = 0;        // (a CPU signal: the event and its value)
    uint32_t ev_value = 0;
    uint64_t value = 0;         // (a submit's or present's timeline value: its fence, handed out as it came)
    uint64_t queue = 0;         // (a submit's queue)
    int crop[4] = {0, 0, 0, 0};
    bool has_crop = false;
    float cover = 0.0f;         // (a present's: ultrawide's loading cover as the game presented the frame, wide_cover_sample)
    nvn_prog prog;
    std::chrono::steady_clock::time_point queued;
};
std::mutex q_mu;
std::condition_variable q_cv, q_space, q_idle;
std::deque<Job> q_jobs;
int q_presents = 0;
bool q_busy = false;
uint64_t q_handed = 0, q_done = 0;   // (jobs handed over and done: q_mu)
// (hookshot, the render thread) Whole frames, a test: RX_VK_FRAME_GATE=1. The render thread then starts a frame's jobs
// when the frame's present has been handed over, not before, and the present is handed over after the main thread's
// signal for the frame (present(): the signal the frame's work waits for): the game has then written the whole frame.
// They were made because replaying a frame's first submits while the game was still writing the rest of that frame (the
// render thread caught up with the game: 0.3-2 % of the presents in the Market) drew the lit scene black for one frame
// at 5-15 % of those moments. Measured in the Market with the flash detector and the meter log (which makes the moments
// frequent): 15 and 15 flashes a session without them and with the gate alone (the present still came before the
// signal), 0 with both. Those moments were also where a submit's capture missed the pages the replay's collection was
// marking again (collect_mu, vk_memory.cpp: a third of the captures with jobs replayed as they come, and Kokiri
// Forest's ground mist gone for a frame in 16 to 89 of 3,600 frames). With that window closed, jobs replayed as they
// come (the default) drew no such frame on an Arc B580 in 23,400 frames (Kokiri Forest and the Market standing, four
// sessions), and a frame no longer waits for its present before its replay starts: 86.95 -> 92.4 fps in Kokiri Forest,
// 53.6 -> 58.0 in the Market (medians of four sessions each, interleaved with four of whole frames). With whole frames,
// a frame behind, the replay still runs beside the game for the whole frame (the larger of the two, not their sum), and
// a frame is shown no later than the replay on the submitting thread shows it. What a game's thread waits for goes on
// at once (gate_upto: a queue finish, a texture read back, a fence of this frame), and so does a job that has waited
// RX_VK_FRAME_GATE_MS (1000) with no present behind it (a load that submits without presenting; at 100 a game frame
// stalled by swapping let its first jobs on, and two such frames flashed in 19,763 presents).
uint64_t gate_upto = 0;   // (jobs up to this hand-over number go on without their frame's present: q_mu)
std::deque<std::pair<uint64_t, uint64_t>> gate_values;   // (the last submits' and presents' fence values and hand-over numbers: q_mu)
struct {
    std::atomic<int64_t> timeouts, demands, fences;
} gate_st;
bool gate_on() {
    static const bool on = async_on() && env_on("RX_VK_FRAME_GATE");
    return on;
}
std::chrono::milliseconds gate_ms() {
    static const int v = getenv("RX_VK_FRAME_GATE_MS") ? std::max(0, atoi(getenv("RX_VK_FRAME_GATE_MS"))) : 1000;
    return std::chrono::milliseconds(v);
}
/// (q_mu) Whether the oldest job handed over may be replayed now
bool gate_ready() {
    if (q_jobs.empty()) return false;
    if (!gate_on() || q_presents > 0 || q_done < gate_upto) return true;
    if (std::chrono::steady_clock::now() - q_jobs.front().queued < gate_ms()) return false;
    gate_upto = q_handed, gate_st.timeouts++;   // (what is queued now goes on: none of it waits again)
    return true;
}
/// (a game's thread, about to wait for fence value f) The submit that value was handed out for goes on without its present
void gate_need_value(uint64_t f) {
    if (!gate_on()) return;
    std::lock_guard lk(q_mu);
    for (const auto& [value, number] : gate_values) {
        if (value < f) continue;
        // (the frame before's, the usual one, has its present handed over: nothing to let on)
        if (number > gate_upto && number > q_done && !q_presents) gate_upto = number, gate_st.fences++, q_cv.notify_all();
        break;
    }
}
// (the render thread's log, every 600 presents: what its jobs took, and what the game's threads waited for it)
struct {
    std::atomic<int64_t> job_ns[3], room_ns, room_n, catchup_ns, catchup_n, depth, rb_fallback, held, held_ns, held_forced,
        sig_caps, walk_ns, tex_page_only, tex_exact, rb_exact, lag_ns, lag_max;
} bh_st;
void bh_shown(const std::chrono::steady_clock::time_point& handed) {   // (a present made: how long after the game handed it over)
    const int64_t d = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - handed).count();
    bh_st.lag_ns += d;
    if (d > bh_st.lag_max.load()) bh_st.lag_max = d;
}
int64_t bh_now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void fo_log(double frames);   // (frame overlap, below: the GPU between recordings and what comes after a frame's last reads)
void bh_log() {
    static int64_t last = bh_now();
    const int64_t now = bh_now();
    const double frames = 600.0, wall = (now - last) / 1e6 / frames;
    last = now;
    const double sub = bh_st.job_ns[0].exchange(0) / 1e6 / frames, pres = bh_st.job_ns[1].exchange(0) / 1e6 / frames,
                 other = bh_st.job_ns[2].exchange(0) / 1e6 / frames;
    const int64_t rn = bh_st.room_n.exchange(0), cn = bh_st.catchup_n.exchange(0);
    if (gate_on())
        logf("vk: the render thread's whole frames (600 presents): %lld jobs went on before their present after %d ms, %lld times a "
             "game's thread waited for the work handed over, %lld times for a fence of the frame in the making",
             (long long)gate_st.timeouts.exchange(0), (int)gate_ms().count(), (long long)gate_st.demands.exchange(0),
             (long long)gate_st.fences.exchange(0));
    logf("vk: the render thread (600 presents, %.2f ms a frame): its jobs %.2f ms a frame (submits %.2f, presents %.2f, others "
         "%.2f), idle %.2f; the game waited for room at %lld presents, %.2f ms each (%.2f a frame; %.2f jobs queued at a "
         "present); %lld catch-ups (a finish, a texture read back, a write-back not recorded yet: %lld), %.2f ms each; "
         "%lld presents held for the CPU's signal, %.2f ms each (%lld let go by RX_VK_HOLD_MS); %lld captures at a CPU signal; "
         "the submits' waits read in %.3f ms a frame; readbacks: %lld next to a texture's storage (no catch-up), %lld in one, "
         "%lld read exactly (after the work handed over: among the catch-ups); a present made %.2f ms after its hand-over (%.2f at most)",
         wall, sub + pres + other, sub, pres, other, wall - sub - pres - other, (long long)rn,
         rn ? bh_st.room_ns.load() / 1e6 / rn : 0.0, bh_st.room_ns.exchange(0) / 1e6 / frames,
         bh_st.depth.exchange(0) / frames, (long long)cn, (long long)bh_st.rb_fallback.exchange(0),
         cn ? bh_st.catchup_ns.exchange(0) / 1e6 / cn : 0.0, (long long)bh_st.held.load(),
         bh_st.held.load() ? bh_st.held_ns.exchange(0) / 1e6 / bh_st.held.load() : 0.0, (long long)bh_st.held_forced.exchange(0),
         (long long)bh_st.sig_caps.exchange(0), bh_st.walk_ns.exchange(0) / 1e6 / frames, (long long)bh_st.tex_page_only.exchange(0),
         (long long)bh_st.tex_exact.exchange(0), (long long)bh_st.rb_exact.exchange(0), bh_st.lag_ns.exchange(0) / 1e6 / frames,
         bh_st.lag_max.exchange(0) / 1e6);
    bh_st.held = 0;
    fo_log(frames);
}
int presents_ahead() {   // (RX_VK_PRESENTS_AHEAD=<n>: how many presents the game may hand over ahead of the replay; 1)
    static const int n = std::clamp(getenv("RX_VK_PRESENTS_AHEAD") ? atoi(getenv("RX_VK_PRESENTS_AHEAD")) : 1, 1, 8);
    return n;
}

// ---- (hookshot) Frame overlap: what each recording does to the game's memory, and the GPU's time between recordings.
// frame_overlap.h has the rules; docs/development/frame-pipeline.md the frame's sequence. ----
// The game waits once a frame, with no timeout, for the fence of the frame before's present (OoT's main thread, at
// 0x8042cde0), and makes the CPU signal its next frame's work waits for only after it: the render thread gets a frame once
// the GPU has finished the one before. Counted always, with the render thread: per recording, whether it reads the game's
// memory at the GPU's time (a dispatch, a draw that follows pointers, a binding in place) and whether its part's end writes
// any back; per frame, how many recordings and how much GPU time come after the last of each kind; and the GPU's time
// between recordings, idle and in upload command buffers, by where in the frame (bh_log's second line).
/// Whether recordings are counted: the render thread, its recordings made in order
bool fo_tracking() {
    static const bool on = async_on() && !par_threads();
    return on && rec_timeline != VK_NULL_HANDLE;
}
struct FoSlot {
    uint64_t seq = 0;   // (its number on the recordings' timeline)
    int column = 0;     // (frame_overlap::gap_column: where in its frame it comes)
};
FoSlot fo_slots[kRecordings];   // (replay_mu, as the rest of fo: recordings are begun, ended and collected under it)
VkQueryPool fo_up_pool = VK_NULL_HANDLE;   // (a timestamp at the start of each recording's upload command buffer, by slot)
struct {
    uint64_t last_read = 0, last_touch = 0;   // (the newest recording that read the game's memory, and that read or wrote it)
    int64_t follows = 0, follows0 = 0, wb0 = 0;   // (reading draws and dispatches so far; at the last recording's end, with the bytes written back)
    uint64_t last_end = 0;                    // (the GPU's clock at the end of the recording collected last; 0: not known)
    int64_t work0 = 0;                        // (draws and dispatches so far, at the last recording's end)
    int frame_working = 0;                    // (this frame's recordings so far that had one)
    frame_overlap::Frames reads, touches;     // (the frames recorded whole, by their last recording of each kind: for the tails' GPU time)
} fo;
std::atomic<uint64_t> fo_present{0};      // (the value of the present being made: its own recording ends its frame)
struct {
    std::atomic<int64_t> read_tail, touch_tail, read_tail_ns, touch_tail_ns, idle_ns[frame_overlap::kGapColumns], uploads_ns[frame_overlap::kGapColumns];
} fo_st;
// ---- (hookshot) Frame ahead: counters only. What the parts recorded in the signal's job took that came with the signal
// (frame_overlap.h PartInputs; docs/development/frame-pipeline.md has what is and is not counted, and what the numbers decide).
struct {
    frame_overlap::AheadTally tally;   // (replay_mu)
    int64_t cap0 = 0, live0 = 0;       // (memory_sync_sources at the last part's end, or at the signal's job's start)
} fa;
std::atomic<bool> fa_in_signal{false};   // (the render thread is in a signal's job that took a capture)
struct {
    std::atomic<int64_t> signals, pages, pools, regs, releases, other_signals, frames_any;
} fa_st;
struct FaWindow {   // (the render thread's: the jobs since the frame's last submit)
    int regs = 0, releases = 0, signals = 0;
} fa_window;
/// (the render thread, at a signal's job that brings a capture, before the capture is taken)
void fa_signal_begin() {
    if (!fo_tracking()) return;
    int64_t pages = 0, pools = 0;
    memory_capture_peek(&pages, &pools);
    fa_st.signals++, fa_st.pages += pages, fa_st.pools += pools;
    fa_st.regs += fa_window.regs, fa_st.releases += fa_window.releases, fa_st.other_signals += fa_window.signals;
    fa_st.frames_any += fa_window.regs || fa_window.releases || fa_window.signals;
    fa_window = {};
    std::lock_guard lk(replay_mu);
    memory_sync_sources(&fa.cap0, &fa.live0);
    fa_in_signal.store(true, std::memory_order_release);
}
void fo_recording_begin(const Recording& r) {
    if (!fo_tracking()) return;
    fo_slots[r.slot] = {};
    if (!fo_up_pool) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kRecordings;
        if (vkCreateQueryPool(dev.device, &qi, nullptr, &fo_up_pool) != VK_SUCCESS) fo_up_pool = VK_NULL_HANDLE;
        if (!fo_up_pool) return;
    }
    // (in the upload command buffer itself: one with no uploads is not submitted, and its slot's query keeps an older use's
    // time, which fo_collect tells by its place)
    vkCmdResetQueryPool(r.upload, fo_up_pool, (uint32_t)r.slot, 1);
    vkCmdWriteTimestamp2(r.upload, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, fo_up_pool, (uint32_t)r.slot);
}
void fo_recording_end(const Recording& r) {
    if (!fo_tracking()) return;
    const uint64_t seq = rec_seq + 1;   // (the number submit_recording gives it next: its callers hold replay_mu)
    int64_t mirrors = 0, sync_bytes = 0, sync_copies = 0, wb = 0;
    uint64_t mirror_mb = 0;
    memory_stats(&mirrors, &mirror_mb, &sync_bytes, &sync_copies, &wb);
    const frame_overlap::RecordingFacts f{fo.follows - fo.follows0, wb - fo.wb0, memory_in_place_part(), r.part};
    fo.follows0 = fo.follows, fo.wb0 = wb;
    if (frame_overlap::reads_game(f)) fo.last_read = seq;
    if (frame_overlap::touches_game(f)) fo.last_touch = seq;
    int64_t draws = 0, skipped = 0, dispatches = 0, pipes = 0;
    draw_stats(&draws, &skipped, &dispatches, &pipes);
    // (the six submits before the signal's job record no draw: they and the first part after them are the frame's lead)
    fo_slots[r.slot] = {seq, frame_overlap::gap_column(fo.frame_working)};
    if (draws + dispatches != fo.work0) {
        fo.frame_working++;
        if (fa_in_signal.load(std::memory_order_acquire)) {   // (frame ahead: a part of the signal's job, and what it took)
            int64_t cap = 0, live = 0;
            memory_sync_sources(&cap, &live);
            fa.tally.part({cap - fa.cap0, live - fa.live0});
            fa.cap0 = cap, fa.live0 = live;
        }
    }
    fo.work0 = draws + dispatches;
    const uint64_t present = fo_present.load(std::memory_order_acquire);
    if (!present || r.value != present) return;
    // (the present's own recording: its frame is recorded whole)
    fo_present.store(0, std::memory_order_release);
    fo.frame_working = 0;
    fa.tally.frame_end();
    fo.reads.note(present, fo.last_read, seq), fo.touches.note(present, fo.last_touch, seq);
    fo_st.read_tail += (int64_t)(seq - std::min(fo.last_read, seq)), fo_st.touch_tail += (int64_t)(seq - std::min(fo.last_touch, seq));
}
/// A recording's GPU timestamps as its slot comes round (t: its start and end; null: none), in the order the recordings were made
void fo_collect(const Recording& r, const uint64_t* t) {
    if (!fo_tracking()) return;
    if (!t) {
        fo.last_end = 0;
        return;
    }
    const double period = (double)dev.props.limits.timestampPeriod;
    const FoSlot& s = fo_slots[r.slot];
    uint64_t up = 0;
    if (!fo_up_pool || vkGetQueryPoolResults(dev.device, fo_up_pool, (uint32_t)r.slot, 1, sizeof up, &up, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        up = 0;
    const frame_overlap::Gap g = frame_overlap::gap(fo.last_end, up, t[0], period);
    fo_st.idle_ns[s.column] += g.idle_ns, fo_st.uploads_ns[s.column] += g.uploads_ns;
    fo.last_end = t[1];
    const int64_t ns = (int64_t)((double)(t[1] - t[0]) * period);
    if (fo.reads.after_release(s.seq)) fo_st.read_tail_ns += ns;
    if (fo.touches.after_release(s.seq)) fo_st.touch_tail_ns += ns;
}
void fo_log(double frames) {
    if (!fo_tracking()) return;
    constexpr int n = frame_overlap::kGapColumns;
    double idle[n], up[n], idle_after = 0, up_after = 0;
    for (int i = 0; i < n; i++) {
        idle[i] = fo_st.idle_ns[i].exchange(0) / 1e6 / frames, up[i] = fo_st.uploads_ns[i].exchange(0) / 1e6 / frames;
        if (i) idle_after += idle[i], up_after += up[i];
    }
    logf("vk: frame overlap (600 presents): the GPU between recordings, ms a frame, idle + in the next one's uploads: up to a "
         "frame's first recording with a draw %.2f + %.2f (the window's copy is in the idle), after it %.2f + %.2f; after a "
         "frame's last recording that reads or writes the game's memory come %.2f recordings (%.2f ms of GPU time), after its "
         "last that reads it %.2f (%.2f ms)",
         idle[0], up[0], idle_after, up_after, fo_st.touch_tail.exchange(0) / frames, fo_st.touch_tail_ns.exchange(0) / 1e6 / frames,
         fo_st.read_tail.exchange(0) / frames, fo_st.read_tail_ns.exchange(0) / 1e6 / frames);
    {   // (frame ahead: counters only)
        frame_overlap::AheadTally t;
        {
            std::lock_guard lk(replay_mu);
            t = fa.tally;
            const int index = t.index;
            const bool three = t.three;
            fa.tally = {};
            fa.tally.index = index, fa.tally.three = three;   // (a frame under way keeps its place)
        }
        const double f = (double)std::max<int64_t>(1, t.frames);
        logf("vk: frame ahead (600 presents): %.2f parts a frame recorded in the signal's job, %.1f%% of them took nothing that "
             "came with the signal; of the others, %.2f a frame took pages of the signal's capture into a mirror (%.1f pages a "
             "frame) and %.2f pages read as guest memory had them (%.1f pages a frame); a frame's first part took nothing in "
             "%lld of %lld frames, its first three in %lld",
             t.parts / f, frame_overlap::final_share(t), t.by_signal / f, t.signal_pages / f, t.by_live / f, t.live_pages / f,
             (long long)t.first_final, (long long)t.frames, (long long)t.first_three_final);
        const int64_t sig = fa_st.signals.exchange(0);
        const double g = (double)std::max<int64_t>(1, sig);
        logf("vk: frame ahead, what came with the signal: %lld signals with a capture, each of %.1f pages in %.1f pools; between a "
             "frame's last submit and that signal %.2f texture and sampler registrations, %.2f releases and %.2f other CPU "
             "signals a signal (%lld of them with any)",
             (long long)sig, fa_st.pages.exchange(0) / g, fa_st.pools.exchange(0) / g, fa_st.regs.exchange(0) / g,
             fa_st.releases.exchange(0) / g, fa_st.other_signals.exchange(0) / g, (long long)fa_st.frames_any.exchange(0));
    }
    // (and where in the frame: before each of its recordings with a draw, the second to the twelfth, then all the later ones)
    char line[512];
    int at = 0;
    for (int i = 1; i < n && at < (int)sizeof line - 24; i++)
        at += snprintf(line + at, sizeof line - at, "%s%.2f+%.2f", i > 1 ? ", " : "", idle[i], up[i]);
    logf("vk: frame overlap, by place: idle + uploads before a frame's 2nd to %dth recording with a draw, then all later ones: %s",
         n - 1, line);
}

/// (hookshot) Whether this run has the render thread, from the one place async_on reads: RX_VK_ASYNC when it is set (the
/// switch, for testing, over the menu: 0 the submitting thread, any other value the render thread), else the Insert
/// menu's Threaded rendering as this run resolved it, its render thread's half (vk_menu.cpp menu_render_thread: On, or Auto
/// from 4 cores, in both of its tiers). Read once: a change of the row applies at the next start
int render_thread_mode() {
    static const int mode = [] {
        const char* e = getenv("RX_VK_ASYNC");
        return e ? atoi(e) : menu_render_thread() ? 1 : 0;
    }();
    return mode;
}
/// (hookshot) RX_VK_ASYNC=1 (any value but 0): the render thread. Off by default: the replay on the submitting thread
/// (TotK Recomp's reason for it: OoT rewrites double-buffered data right after a submit with no fence, which lands
/// inside a replay running behind).
/// The replay on a thread of its own, with the game let run a frame ahead of it, so a frame costs the larger of the
/// game's work and the replay's, not their sum. The older replay thread (RX_VK_ASYNC=1 before, the render thread =2;
/// removed) could not: the game waited for the replay three times a frame anyway. Its presenting thread polls its
/// frame's event, which the GPU signals at the frame's end, before it presents (20 of a 37 ms frame asleep there,
/// profiled; host/nvn_native.c answers the poll here); its job threads' readbacks caught the replay up (invalidate:
/// 7.5-11 ms each, and the stopped queues forced past their waits, 29789 of 60000: whole-screen white or black frames,
/// 28 per 1000 gameplay frames against 1.3 on the submitting thread); and the occlusion queries were off (2006 draws a
/// frame for 1709). Here:
///  - a submit: the capture and the hand-over, on the game's thread;
///  - a present: handed over; the game waits there while the present before it is not replayed yet (RX_VK_PRESENTS_AHEAD,
///    1): one frame ahead at most;
///  - a readback waits for the write-backs of the work recorded so far, never for the queue (invalidate);
///  - nothing forces a stopped queue but its present and RX_VK_HOLD_MS: a catch-up (a queue finish, a texture read back)
///    waits for what was handed over before it; a release waits for the stopped work, as on the submitting thread;
///  - the occlusion queries stay on (occ_queries_on).
bool async_on() {
    static const bool on = render_thread_mode() != 0;   // (the switch, or the Threaded rendering row: render_thread_mode)
    return on;
}
// (the recorder's half of the Threaded rendering row: with RX_VK_RECORDER unset, vk_recorder.h's mode_from_env asks the row
// through this, set before anything runs; the checks that build vk_recorder.h alone leave it null, the recorder off. The
// recorder thread's half (menu_recorder_thread: On, or Auto from 6 cores), not the render thread's: on 4 cores Auto has the
// render thread alone, as the recorder thread beside it was the slowest there, 18 Nov)
[[maybe_unused]] const bool rec_setting_hooked = (rec::g_setting_thread = menu_recorder_thread, true);

void readback_fell_back(int what) {
    (what == 0 ? bh_st.rb_fallback : what == 1 ? bh_st.tex_page_only : what == 2 ? bh_st.tex_exact : bh_st.rb_exact)++;
}
#ifdef _WIN32
/// (hookshot) The logical processors of the cores of the fastest kind in the calling thread's processor group: the highest
/// EfficiencyClass, a hybrid CPU's performance cores; every core on a CPU of one kind (replay_thread_main)
uint64_t fastest_kind_cpus() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (!len) return 0;
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)buf.data(), &len)) return 0;
    GROUP_AFFINITY ga{};
    const WORD group = GetThreadGroupAffinity(GetCurrentThread(), &ga) ? ga.Group : 0;
    uint64_t mask = 0;
    int best = -1;
    for (DWORD at = 0; at + sizeof(DWORD) * 2 <= len;) {
        const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf.data() + at);
        if (!e->Size) break;
        if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount && e->Processor.GroupMask[0].Group == group) {
            const int cls = e->Processor.EfficiencyClass;
            const uint64_t m = (uint64_t)e->Processor.GroupMask[0].Mask;
            if (cls > best) best = cls, mask = m;
            else if (cls == best) mask |= m;
        }
        at += e->Size;
    }
    return mask;
}
#endif
void replay_thread_main() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk replay");
#ifdef TRACY_ENABLE
    HT_THREAD_NAME("mkw_vk replay");
#endif
    fps_render_thread();   // (the FPS counter's render lane is this thread's)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    // (on the core the host keeps free of the game's threads, hos_reserved_cpus: its last physical one)
    // (hookshot) Windows: its ideal processor there, and free to go to the other cores of the fastest kind when another
    // program's thread takes it: the game's threads are kept off the core, other programs' are not. Kept to the core alone, a
    // thread at priority 15 (Intel's PresentMon service, which the Intel Graphics Software overlay runs) took its one usable
    // logical processor for 30-40 ms at a time (Windows had parked the other) and the replay waited, ready to run, until it was
    // done: 14-37 frames over 40 ms in each 40 s run in Castle Town's market on an i5-13600K with an Arc B580, none with the
    // thread free to move. (Elsewhere than on Windows: the reserved core alone)
    if (host->reserved_cpus) {
        if (const uint64_t m = host->reserved_cpus()) {
            uint64_t use = m;
#ifdef _WIN32
            {
                DWORD_PTR proc = 0, sys = 0;
                if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys)) proc = 0;
                use |= fastest_kind_cpus() & (uint64_t)proc;
            }
#endif
            // (hookshot) Windows refuses a mask with a processor outside the process's, and the thread stays where it was:
            // the log says so, not the placement it did not get (hos_reserved_cpus keeps to the process's cores; a
            // HOS_RESERVE_CORE_MASK outside them still gets here)
            if (!SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)use)) {
                logf("vk: the replay thread was NOT kept to logical processors %#llx (error %lu): it runs where the process "
                     "may, beside the game's threads", (unsigned long long)use, (unsigned long)GetLastError());
            } else {
#ifdef _WIN32
                if (use != m) SetThreadIdealProcessor(GetCurrentThread(), (DWORD)(63 - __builtin_clzll(m)));
#endif
                logf("vk: the replay thread runs on logical processors %#llx, kept from the game's threads%s", (unsigned long long)m,
                     use != m ? " (its ideal processor there; free to go to the other fastest cores when another program takes it)" : "");
                if (use != m) logf("vk: the replay thread may run on logical processors %#llx", (unsigned long long)use);
            }
        }
    }
    // (hookshot, the render thread) A present that comes while a queue is stopped at a wait for an event the CPU signals is
    // held until that signal's job, RX_VK_HOLD_MS at most. OoT submits its presentation queue's work before the frame's data
    // is all written, behind such a wait, and signals when it is; its presenting thread then polls the frame's event, which
    // the GPU signals at that work's end, and presents. With the poll answered at once (host/nvn_native.c) the present is
    // handed over before the main thread's signal about every other frame: forced on by it (drain 0), the frame was
    // replayed from the submit's capture, without what the game wrote since, and every one of its waits was gone past
    // (44259 at 3380 presents, measured). The jobs between the present and the signal are replayed meanwhile: none is a
    // submit of the next frame, which the main thread starts after its signal
    Job held_present;
    bool present_held = false;
    int64_t present_held_at = 0;
    auto held_present_run = [&](bool forced) {
#ifdef TRACY_ENABLE
        HT_SCOPE("render job: held present");
#endif
        bh_st.held_ns += bh_now() - present_held_at, bh_st.held_forced += forced;
        present_held = false;
        const int64_t t0 = bh_now();
        drain(0);
        present_value = held_present.value;
        fo_present.store(held_present.value, std::memory_order_release);   // (frame overlap: its recording ends the frame)
        do_present(held_present.has_tex ? &held_present.tex : nullptr, held_present.has_crop ? held_present.crop : nullptr, held_present.cover);
        bh_shown(held_present.queued);
        update_blocked();
        bh_st.job_ns[1] += bh_now() - t0;
        if (presents % 3600 == 0) log_order_stats();
        if (presents % 600 == 0) bh_log();
        std::lock_guard lk(q_mu);
        q_presents--, q_space.notify_all();
    };
    for (;;) {
        Job j;
        bool have = false;
        {
            std::unique_lock lk(q_mu);
            FpsRtWait rt_idle(fps_stats::kRtIdle);   // (the FPS counter's render lane: waiting for a job)
            // (gate_ready: the oldest job, once its frame's present is handed over or a game's thread waits for it)
            if (present_held)   // (the signal's job, or on without it)
                q_cv.wait_for(lk, std::chrono::milliseconds(2), [] { return gate_ready(); });
            else if (blocked_n.load())   // (a queue stopped at a wait: the next job, or on without it after a while)
                q_cv.wait_until(lk, blocked_since + std::chrono::milliseconds(hold_ms()), [] { return gate_ready(); });
            else
                while (!gate_ready()) {
                    if (q_jobs.empty()) q_cv.wait(lk);
                    else q_cv.wait_until(lk, q_jobs.front().queued + gate_ms());
                }
            if (gate_ready()) j = std::move(q_jobs.front()), q_jobs.pop_front(), have = true;
            q_busy = true;
        }
        // (stopped ones: on now when stopped too long. No thread's wait for the replay forces them: wait_queue_empty)
        if (blocked_n.load() && std::chrono::steady_clock::now() >= blocked_since + std::chrono::milliseconds(hold_ms())) drain(3);
        if (!have) {
            if (present_held) {
                const bool late = bh_now() - present_held_at >= (int64_t)hold_ms() * 1000000;
                if (late || !cpu_wait_unmet()) held_present_run(late);
            }
            update_blocked();
            std::lock_guard lk(q_mu);
            q_busy = false;
            continue;   // (no job done: nothing for wait_queue_empty's waiters, which wait for q_done)
        }
        // (RX_VK_TIMELINE: one frame's jobs every 240 presents: when the game handed each over, how long it waited
        // for the replay thread, how long it took, its draws; the present's time is 0)
        static const bool timeline = getenv("RX_VK_TIMELINE") != nullptr;
        static const bool timeline_late = timeline && atoi(getenv("RX_VK_TIMELINE")) >= 2;
        const auto t_start = std::chrono::steady_clock::now();
        int64_t d0 = 0, x;
        if (timeline) draw_stats(&d0, &x, &x, &x);
        bool deferred = false;
        // (frame ahead's counters: what is handed over between a frame's last submit and its signal)
#ifdef TRACY_ENABLE
        HT_ZONE(ht_job, "render job");   // (one job of the render thread, from the game's hand-over queue)
        HT_ZONE_TEXTF(ht_job, "%s, value %llu, waited %.3f ms", j.kind == JOB_SUBMIT ? "submit" : j.kind == JOB_PRESENT ? "present" :
                      j.kind == JOB_EVENT_SIGNAL ? "event signal" : "other", (unsigned long long)j.value,
                      std::chrono::duration<double, std::milli>(t_start - j.queued).count());
#endif
        if (j.kind == JOB_SUBMIT) fa_window = {};
        else if (j.kind == JOB_REGISTER_TEXTURE || j.kind == JOB_REGISTER_SAMPLER) fa_window.regs++;
        else if (j.kind == JOB_TEXTURE_RELEASED || j.kind == JOB_PROGRAM_RELEASED) fa_window.releases++;
        switch (j.kind) {
        case JOB_SUBMIT: submit_job(j.handles, j.value, j.queue); break;
        case JOB_PRESENT:   // (the frame's work whole before it is shown)
            if (present_held) held_present_run(true);   // (two presents ahead: the one before it first)
            if (cpu_wait_unmet()) {
                held_present = std::move(j), present_held = deferred = true, present_held_at = bh_now();
                bh_st.held++;
                break;
            }
            drain(0);
            present_value = j.value;
            fo_present.store(j.value, std::memory_order_release);   // (frame overlap: its recording ends the frame)
            do_present(j.has_tex ? &j.tex : nullptr, j.has_crop ? j.crop : nullptr, j.cover);
            bh_shown(j.queued);
            break;
        case JOB_TEXTURE_RELEASED:   // (after the stopped work that may use it)
            if (blocked_n.load()) {   // (kept until it has gone on, not forced: update_blocked)
                std::lock_guard hl(held_mu);
                held_tex_releases.push_back(j.tex);
                st_held_releases++;
                break;
            }
            drain(1);   // (nothing stopped)
            do_texture_released(&j.tex);
            break;
        case JOB_PROGRAM_RELEASED:
            if (blocked_n.load()) {
                std::lock_guard hl(held_mu);
                held_prog_releases.push_back(j.prog);
                st_held_releases++;
                break;
            }
            drain(1);
            do_program_released(&j.prog);
            break;
        case JOB_REGISTER_TEXTURE: register_texture_copy(j.id, j.tex, j.has_view ? &j.view : nullptr); break;
        case JOB_REGISTER_SAMPLER: register_sampler_copy(j.id, j.samp); break;
        case JOB_EVENT_SIGNAL:   // (the CPU set an event: the stopped ones may go on)
            {
                std::lock_guard ek(ev_cpu_mu);
                cpu_events.insert(j.ev_gpu);
            }
            if (j.id) fa_signal_begin();   // (frame ahead's counters: what this capture brings, before it is taken)
            else fa_window.signals++;
            if (j.id) {   // (its capture: what the game wrote up to the signal, for the work that waited for it)
#ifdef TRACY_ENABLE
                HT_SCOPE("signal job: capture");
#endif
                std::lock_guard lk(replay_mu);
                memory_take_capture(j.queue, completed());
                st_resume_captures++;
            }
            ev_values[j.ev_gpu] = j.ev_value;
            burst_begin(0);
            schedule();
            burst_end();
            fa_in_signal.store(false, std::memory_order_release);
            break;
        }
        update_blocked();
        if (j.kind == JOB_PRESENT && !deferred && presents % 3600 == 0) log_order_stats();
        bh_st.job_ns[j.kind == JOB_SUBMIT ? 0 : j.kind == JOB_PRESENT ? 1 : 2] +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_start).count();
        if (j.kind == JOB_PRESENT && !deferred && presents % 600 == 0) bh_log();
        if (present_held && !deferred && !cpu_wait_unmet()) held_present_run(false);
        if (timeline && !deferred && (j.kind == JOB_SUBMIT || j.kind == JOB_PRESENT)) {
            struct T { char k; double q, s, e; int64_t d; size_t h; };
            static std::vector<T> frame;
            const auto t_end = std::chrono::steady_clock::now();
            int64_t d1 = 0;
            draw_stats(&d1, &x, &x, &x);
            auto ms = [](auto a) { return std::chrono::duration<double, std::milli>(a.time_since_epoch()).count(); };
            frame.push_back({j.kind == JOB_SUBMIT ? 's' : 'p', ms(j.queued), ms(t_start), ms(t_end), d1 - d0, j.handles.size()});
            if (j.kind == JOB_PRESENT) {
                // (RX_VK_TIMELINE=2: also every frame handed over 20 ms or more after the last one, 200 of them)
                static double last_q = 0;
                static int late_logged = 0;
                const double q_now = frame.back().q;
                const bool late = timeline_late && last_q && q_now - last_q >= 20.0 && late_logged < 200;
                if (late) late_logged++;
                last_q = q_now;
                if (presents % 240 == 0 || late) {
                    const double base = frame.back().q;
                    char line[2048];
                    int n = snprintf(line, sizeof line, "vk: timeline%s (ms from the present's hand-over):",
                                     late ? " of a late frame" : "");
                    for (const T& t : frame)
                        if (n < 1900)
                            n += snprintf(line + n, sizeof line - n, " %c q%+.1f s%+.1f e%+.1f %lldd %zuh;", t.k, t.q - base,
                                          t.s - base, t.e - base, (long long)t.d, t.h);
                    logf("%s", line);
                }
                frame.clear();
            }
        }
        std::lock_guard lk(q_mu);
        if (j.kind == JOB_PRESENT && !deferred) q_presents--, q_space.notify_all();
        q_busy = false;
        q_done++;
        q_idle.notify_all();   // (wait_queue_empty: a job done)
    }
}

void enqueue(Job&& j) {
#ifdef TRACY_ENABLE
    HT_SCOPE("enqueue (game thread)");
#endif
    j.queued = std::chrono::steady_clock::now();
    std::unique_lock lk(q_mu);
    static bool started;
    if (!started) {
        started = true;
        ::mkwvk::cxx_escape::spawn_needed("replay", replay_thread_main);
        logf("vk: the replay runs on a thread of its own, the render thread (RX_VK_ASYNC=0: on the submitting thread)");
    }
    if (j.kind == JOB_PRESENT) {   // (the game waits here while the present before it is not replayed: a frame ahead at most)
        bh_st.depth += (int64_t)q_jobs.size() + (q_busy ? 1 : 0);
        if (q_presents >= presents_ahead()) {
            FpsWait room_wait(fps_stats::kWaitRender);   // (the FPS counter's Breakdown: the render thread a frame behind)
            const int64_t t0 = bh_now();
            q_space.wait(lk, [] { return q_presents < presents_ahead(); });
            bh_st.room_ns += bh_now() - t0, bh_st.room_n++;
        }
        q_presents++;
    }
    if (j.kind == JOB_SUBMIT || j.kind == JOB_PRESENT) j.value = ++q_values;
    if (j.kind == JOB_SUBMIT) gf_note_value(j.value);
    q_handed++;
    if ((j.kind == JOB_SUBMIT || j.kind == JOB_PRESENT) && gate_on()) {   // (a fence of it, waited for before its present: gate_need_value)
        gate_values.emplace_back(j.value, q_handed);
        if (gate_values.size() > 64) gate_values.pop_front();
    }
    q_jobs.push_back(std::move(j));
    q_cv.notify_one();
}

void wait_queue_empty() {
    if (!async_on()) {
        // A catch-up (a queue finish, a CPU read back of what the GPU wrote, a pool remapped) used to force every queue
        // stopped at an event wait past it. The work those waits are for is mostly not submitted yet (the waiting thread is the one
        // that submits it), so the stopped submission went on without it and its lit passes read what that work makes before it
        // was made: the scene black, or grey, or blown out for a frame, every other frame for seconds at a time in Kokiri Forest
        // once the game ran at 60 fps (at 4-8 fps the work had always come first). Measured over 70 s of Kokiri gameplay: 10000-
        // 40000 waits gone past and 250-1100 flashing frames before, ~300 and ~65 now, 60 fps either way. A catch-up now lets on the
        // waits that are met; what stays stopped goes on with its own queue's work, a present, or sync_watch's hold_ms as before.
        if (!blocked_n.load()) return;
        std::lock_guard lk(sync_order_mu);
        burst_begin(0);
        schedule();
        burst_end();
        update_blocked();
        return;
    }
    // (the render thread: what was handed over before this call, replayed as far as its waits allow: nothing forced)
    std::unique_lock lk(q_mu);
    const uint64_t upto = q_handed;
    if (q_done >= upto) return;
    if (gate_on() && upto > gate_upto) gate_upto = upto, gate_st.demands++, q_cv.notify_all();   // (on without their present)
    const int64_t t0 = bh_now();
    q_idle.wait(lk, [upto] { return q_done >= upto; });
    bh_st.catchup_ns += bh_now() - t0, bh_st.catchup_n++;
}
/// (work handed to the replay thread and not yet recorded)
bool replay_behind() {
    if (!async_on()) return blocked_n.load() > 0;
    std::lock_guard lk(q_mu);
    return !q_jobs.empty() || q_busy || blocked_n.load() > 0;
}

std::mutex submit_order_mu;   // (captures and submits queued in the same order)
// ---- (hookshot, the render thread) the capture at a CPU signal ----
// A queue stopped at a wait for an event the CPU signals reads what the game wrote between its submit and the signal
// (resume_queue: the fog and exposure of the frame before otherwise). On the submitting thread that capture is taken when the
// queue goes on; behind the game that is too late, the game has gone on writing. Here it is taken on the game's thread,
// at the nvnEventSignal that meets a wait of a submission handed over, and the render thread takes it at the signal's job.
// The waits are read from the command buffers at their submit: the last 256 (~14 a frame). submit_order_mu
struct BhWait {
    uint64_t gpu = 0;
    uint32_t mode = 0, value = 0;
    bool open = false;
};
BhWait bh_waits[256];
size_t bh_waits_n = 0;
std::unordered_map<uint64_t, uint32_t> bh_cpu_value;   // (the events the CPU has signalled, and their last value)
void bh_note_waits(const uint64_t* handles, int n) {
    const int64_t t0 = bh_now();
    for (int cb = 0; cb < n; cb++) {
        if (!handles[cb]) continue;
        for (const uint8_t* o = reinterpret_cast<const uint8_t*>(handles[cb]);; o = op_next(o)) {
            nvn_op_hdr h;
            std::memcpy(&h, o, sizeof h);
            if (h.op == NOP_END) break;
            if (h.op != NOP_WAIT_EVENT) continue;
            const auto w = payload<nvn_p_wait_event>(o + 8, h.size - 8);
            const auto it = bh_cpu_value.find(w.gpu);
            if (it != bh_cpu_value.end() && event_wait_met(it->second, w.mode, w.value)) continue;   // (met already: its submit's capture)
            bh_waits[bh_waits_n++ % 256] = {w.gpu, w.mode, w.value, true};
        }
    }
    bh_st.walk_ns += bh_now() - t0;
}
bool bh_signal_meets(uint64_t gpu, uint32_t value) {
    bh_cpu_value[gpu] = value;
    bool any = false;
    for (BhWait& w : bh_waits)
        if (w.open && w.gpu == gpu && event_wait_met(value, w.mode, w.value)) w.open = false, any = true;
    return any;
}
/// (hookshot, the render thread) Whether a submission handed over waits for an event the CPU signals that it has not
/// signalled yet: the frame's data is not all written. submit_order_mu
std::condition_variable bh_signal_cv;
bool bh_cpu_wait_open() {
    for (const BhWait& w : bh_waits) {
        if (!w.open) continue;
        const auto it = bh_cpu_value.find(w.gpu);
        if (it != bh_cpu_value.end() && !event_wait_met(it->second, w.mode, w.value)) return true;
    }
    return false;
}
void submit_q(const uint64_t* handles, int n, uint64_t queue) {
    tl_ev("submit+", queue, n);
    struct TlSub { ~TlSub() { tl_ev("submit-"); } } tl_sub;
#ifdef TRACY_ENABLE
    HT_SCOPE("rx submit_q (game thread)");
#endif
    FpsRender fr;   // (the FPS counter's Breakdown: the replay, on this thread)
    if (!async_on()) {
        std::lock_guard order(sync_order_mu);
        tl_ev("submit-locked");
        ReplayPin pin;   // (the capture and the replay on the replay's core)
        // (the replay here is on the submitting thread, but the game's other threads go on writing their next frames: a pool
        // synced from guest memory later in the frame read those. The capture is taken here too)
        // (EvilOOT) on by default: without it every mirror sync read guest memory as it was at the replay, the game's later
        // frames. OoT's GPU culling alternates two list sets: every other frame its setup read the next frame's records,
        // counted into the other set, and the foliage, grass, characters and particles of that frame were never drawn.
        // With it the syncs take the submits' captures, as the console's GPU sees its memory.
        jt_mark(11, 0);
        lag_bq_enter();   // (RX_VK_LAG_LOG)
        tl_ev("capture+");
        memory_capture_submit();
        tl_ev("capture-");
        jt_mark(12, 0);
        // (RX_VK_SUBMIT_DELAY_US=<us> (a test): a console's GPU starts later than this replay does. Does what a submission reads
        // arrive just after it, from the game's other threads?)
        static const int delay_us = getenv("RX_VK_SUBMIT_DELAY_US") ? atoi(getenv("RX_VK_SUBMIT_DELAY_US")) : 0;
        if (delay_us) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(delay_us);
            while (std::chrono::steady_clock::now() < until) std::this_thread::yield();
        }
        if (sync_waits_on()) {
            static std::once_flag watch;
            std::call_once(watch, [] { ::mkwvk::cxx_escape::spawn_needed("order", sync_watch); });
            std::vector<uint64_t> h(handles, handles + std::max(n, 0));
            const uint64_t value = ++q_values;
            lag_submit(value);
            gf_note_value(value);
            if (queue) {
                q_last[queue] = value;
                value_queue[value] = queue;
                while (value_queue.size() > 8192) value_queue.erase(value_queue.begin());
            }
            submit_job(h, value, queue);
            update_blocked();
            return;
        }
        return do_submit(handles, n);
    }
    std::lock_guard lk(submit_order_mu);
    memory_capture_submit();   // (the pages as they are now, on the game's thread: vk_memory.cpp)
    bh_note_waits(handles, n);
    Job j{JOB_SUBMIT};
    j.handles.assign(handles, handles + std::max(n, 0));
    j.queue = queue;
    enqueue(std::move(j));
}
void submit(const uint64_t* handles, int n) {
    submit_q(handles, n, 0);
}
void present(const nvn_tex* t, const int crop[4]) {
    fps_presented();   // (the FPS counter: the game's frame ends here, on its presenting thread)
#ifdef TRACY_ENABLE
    HT_SCOPE("rx present (game thread)");
#endif
    gf_note_present();
    programs_presented();   // (the programs' life: whether the game is in a load)
    FpsRender fr;
    if (!async_on()) {
        std::lock_guard order(sync_order_mu);
        ReplayPin pin;   // (the frame's last replays and the present on the replay's core)
        if (sync_waits_on()) {
            drain(0);   // (the frame's work whole before it is shown)
            update_blocked();
            present_value = ++q_values;
            return do_present(t, crop, wide_cover_sample());
        }
        return do_present(t, crop, wide_cover_sample());
    }
    {   // (the render thread) The present is handed over after the CPU's signal its frame's work waits for: the main
        // thread makes that signal when it has written the frame's data, and the presenting thread, its poll answered
        // at once, came before it at most presents. With whole frames (gate_on) the frame is then whole at its present,
        // which is where the render thread starts it (gate_ready); without them it stays as it was with them, which is
        // what jobs replayed as they come were measured with. At most RX_VK_FRAME_GATE_MS. RX_VK_PRESENT_AFTER_SIGNAL=0
        // (a test): handed over at once
        static const bool after = env_on_or("RX_VK_PRESENT_AFTER_SIGNAL", true);
        // (as long as the gate gives a job: a frame the game takes longer over is waited for too, not started half written)
        static const auto after_ms = std::max(gate_ms(), std::chrono::milliseconds(hold_ms()));
        if (after) {
            static std::atomic<int64_t> n{0}, waited{0}, ns{0}, late{0};
            std::unique_lock lk(submit_order_mu);
            if (bh_cpu_wait_open()) {
                FpsWait main_wait(fps_stats::kWaitMain);   // (the FPS counter's Breakdown: the game's main thread, not the renderer)
                const int64_t t0 = bh_now();
                if (!bh_signal_cv.wait_for(lk, after_ms, [] { return !bh_cpu_wait_open(); })) late++;
                ns += bh_now() - t0, waited++;
            }
            if (++n % 600 == 0) {
                const int64_t w = waited.exchange(0);
                logf("vk: presents after the CPU's signal (600): %lld waited for it, %.2f ms each, %lld let go after RX_VK_FRAME_GATE_MS",
                     (long long)w, w ? ns.exchange(0) / 1e6 / w : 0.0, (long long)late.exchange(0));
            }
        }
    }
    Job j{JOB_PRESENT};
    j.cover = wide_cover_sample();
    if (t) j.tex = *t, j.has_tex = true;
    if (crop) std::memcpy(j.crop, crop, sizeof j.crop), j.has_crop = true;
    enqueue(std::move(j));
}
void texture_released_of(const nvn_tex* t) {
    if (!t) return;
    tex_pages_add(t, -1);
    texture_ahead_gone(t);
    if (!async_on()) {
        // (PR 430: after the stopped work that may use it, when it goes on. The lock is held_mu, only while queues are
        // stopped: sync_order_mu, taken on every release, made the thread finalizing textures (OptimizeChangeBank) wait for whole
        // submissions being replayed under it, 45% of its time in Kokiri. A release just after the stopped work went on waits
        // for the next update_blocked: the next submit or present)
        if (sync_waits_on() && blocked_n.load()) {
            {
                std::lock_guard hl(held_mu);
                held_tex_releases.push_back(*t);
            }
            if (++st_held_releases % 2000 == 1) logf("vk: releases kept for stopped work: %lld", (long long)st_held_releases.load());
            return;
        }
        sync_drain(1);   // (after the stopped work that may use it)
        return do_texture_released(t);
    }
    Job j{JOB_TEXTURE_RELEASED};
    j.tex = *t, j.has_tex = true;
    enqueue(std::move(j));
}
void program_released_locked(const nvn_prog* p) {
    if (!p) return;
    programs_freed(p);   // (the program's life: the game's call, now; the replay's part follows in order below)
    if (!async_on()) {
        if (sync_waits_on() && blocked_n.load()) {   // (PR 430: as texture releases, on held_mu)
            {
                std::lock_guard hl(held_mu);
                held_prog_releases.push_back(*p);
            }
            st_held_releases++;
            return;
        }
        sync_drain(1);
        return do_program_released(p);
    }
    Job j{JOB_PROGRAM_RELEASED};
    j.prog = *p;
    enqueue(std::move(j));
}
// ---- texture and sampler registrations: in order with the submits ----
// A registration took effect at once, on the game's thread, while the replay recorded submits up to two presents
// older: the game, which registers the slots of its next frames as it builds them (sets of render targets at each
// dynamic-resolution step), had the frame being recorded sample the next ones' textures, of another size: its
// picture went to pieces below or above a line (edges clamped into stripes, the sky's or fog's colour, the shadows
// dark) for a frame, with the size changes (every few frames when the resolution swung). Now they are jobs, applied
// where the game made them among its submits, as finalized textures already were.
// doc: docs/design.md#4-the-lessons-from-mkw-as-rules-here (rule 3: registrations are queued in order with the submits)
void register_texture_queued(int id, const nvn_tex* t, const nvn_view* v) {
    if (!async_on()) return register_texture_locked(id, t, v);
    if (id < 0 || id >= (1 << 20) || !t) return;
    t = (const nvn_tex*)descriptor_ok(t, sizeof(nvn_tex));
    if (!t) return;
    v = (const nvn_view*)descriptor_ok(v, sizeof(nvn_view));
    Job j{JOB_REGISTER_TEXTURE};
    j.id = id, j.tex = *t;
    if (v) j.view = *v, j.has_view = true;
    enqueue(std::move(j));
}
void event_signal_queued(uint64_t gpu, uint32_t value) {
#ifdef TRACY_ENABLE
    HT_SCOPE("rx event_signal (game thread)");
#endif
    if (!async_on()) {
        std::lock_guard lk(ev_cpu_mu);
        ev_cpu_pending.emplace_back(gpu, value);
        ev_cpu_captures[gpu] = memory_captures_begun();   // (after the game's writes for it: cpu_signal_uncaptured)
        ev_cpu_any.store(true, std::memory_order_release);
        ev_cpu_signals.fetch_add(1, std::memory_order_release);   // (after it is in the list: schedule)
        return;   // (sync_watch or the next submit lets the stopped ones on: not this thread, ~1.7 ms a frame of the critical path)
    }
    Job j{JOB_EVENT_SIGNAL};
    j.ev_gpu = gpu, j.ev_value = value;
    std::lock_guard lk(submit_order_mu);
    if (bh_signal_meets(gpu, value)) {
        memory_capture_submit();
        j.id = 1, j.queue = q_values.load();   // (its capture, and the newest value handed out: the take's)
        bh_st.sig_caps++;
    }
    enqueue(std::move(j));
    bh_signal_cv.notify_all();
}
void register_sampler_queued(int id, const nvn_samp_par* p) {
    if (!async_on()) return register_sampler(id, p);
    if (id < 0 || id >= (1 << 12) || !p) return;
    p = (const nvn_samp_par*)descriptor_ok(p, sizeof(nvn_samp_par));
    if (!p) return;
    Job j{JOB_REGISTER_SAMPLER};
    j.id = id, j.samp = *p;
    enqueue(std::move(j));
}
uint64_t fence() { return async_on() || sync_waits_on() ? q_values.load() : pushed.load(); }
uint64_t get_pushed() { return fence(); }

// (a parallel replay's segments) each recording's init command pool and buffer: made with RX_VK_PARALLEL only (par_threads
// is the process's), last in replay_start, which runs once a process. All or none: a failure gives back what was made and
// the start fails, as with the recordings' other pools. Kept as long as the device, like those
void init_pools_destroy() {
    for (Recording& r : recordings) {
        if (r.init_pool) vkDestroyCommandPool(dev.device, r.init_pool, nullptr);   // (its command buffer with it)
        r.init_pool = VK_NULL_HANDLE, r.init = VK_NULL_HANDLE;
    }
}
bool init_pools_make(const VkCommandPoolCreateInfo& pi) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    for (Recording& r : recordings) {
        VkCommandPool pool = VK_NULL_HANDLE;   // (the outputs are kept only on success)
        VkCommandBuffer cb = VK_NULL_HANDLE;
        if (vkCreateCommandPool(dev.device, &pi, nullptr, &pool) != VK_SUCCESS) {
            init_pools_destroy();
            return false;
        }
        r.init_pool = ai.commandPool = pool;
        if (vkAllocateCommandBuffers(dev.device, &ai, &cb) != VK_SUCCESS) {
            init_pools_destroy();
            return false;
        }
        r.init = cb;
    }
    return true;
}

} // namespace

/// (frame overlap, vk_internal.h; vk_draw.cpp push_constants) A draw or a dispatch is recorded now; follows: a dispatch, or a
/// draw whose program follows pointers into the game's memory: its recording reads that memory at the GPU's time
void frame_overlap_op(bool follows) {
    if (fo_tracking()) fo.follows += follows;
}

/// (RX_VK_LOG_EARLY_CHECK, vk_internal.h) a GPU write the replay records now, in place: counted for the pending checks of reads of its
/// submission let go at an earlier part (kind: 0 a binding, a copy or clear; 1 over 1 MiB; 2 a bare or wide storage range)
void early_check_gpu_write(uint64_t lo, uint64_t hi, uint64_t value, uint64_t part, int kind) {
    if (!early_check_on()) return;
    std::lock_guard lk(ec_mu);
    for (auto& e : ec_pending)
        if (e->value == value && part > e->part && lo < e->guest + e->size && e->guest < hi) e->gpu_noted++, e->kinds |= 1 << kind;
}
/// (RX_VK_LOG_EARLY_CHECK, vk_internal.h) pages the soft write watch reported (the CPU's writes); full: memory_collect_written's,
/// over every range, begun as collection gen
void early_check_cpu_pages(const uint32_t* pages, int64_t n, bool full, uint32_t gen) {
    if (!early_check_on()) return;
    std::lock_guard lk(ec_mu);
    for (auto& e : ec_pending) {
        if (e->stage == 2 && e->closed1 && e->closed2) continue;
        const uint64_t m = n > 0 ? ec_pages_hit(e->guest, e->size, pages, n) : 0;
        if (!e->closed1) e->cpu1 |= m;
        if (e->stage >= 1 && !e->closed2) e->cpu2 |= m;
        // (the second full collection begun after a copy: a mark the game's store published late is in one of them)
        if (full && e->stage >= 1 && gen > e->gen_m + 1) e->closed1 = true;
        if (full && e->stage == 2 && gen > e->gen_2 + 1) e->closed2 = true;
    }
    ec_reap();
}
uint32_t early_check_collect_begin() { return early_check_on() ? ++ec_full_begun : 0; }

bool replay_start() {
    menu_init();
    if (!memory_init() || !vma_init(instance_proc())) return false;
    {   // (hookshot) the staging arenas on an empty device: see texture_staging_prewarm
        extern void texture_staging_prewarm(int n);
        static const int prewarm = getenv("RX_VK_STAGING_PREWARM") ? atoi(getenv("RX_VK_STAGING_PREWARM")) : 4;
        if (prewarm > 0) texture_staging_prewarm(prewarm);
    }
    VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo tsi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    tsi.pNext = &ti;
    if (vkCreateSemaphore(dev.device, &tsi, nullptr, &timeline) != VK_SUCCESS) return false;
    if (vkCreateSemaphore(dev.device, &tsi, nullptr, &batch_tl) != VK_SUCCESS) batch_tl = VK_NULL_HANDLE;
    if (vkCreateSemaphore(dev.device, &tsi, nullptr, &rec_timeline) != VK_SUCCESS) rec_timeline = VK_NULL_HANDLE;
    lag_set_timelines([]() -> uint64_t { return completed(); },   // (RX_VK_LAG_LOG: the thread that stamps the GPU's completions)
                      []() -> uint64_t {
                          uint64_t v = 0;
                          if (rec_timeline) vkGetSemaphoreCounterValue(dev.device, rec_timeline, &v);
                          return v;
                      },
                      [](uint64_t value) -> uint64_t {
                          std::lock_guard mk(rec_map_mu);
                          const auto it = value_rec.find(value);
                          return it != value_rec.end() ? it->second : 0;
                      },
                      [](uint64_t tl_next, uint64_t rec_next, uint64_t timeout_ns) {
                          VkSemaphore s[2] = {timeline, rec_timeline};
                          uint64_t v[2] = {tl_next, rec_next};
                          VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                          wi.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
                          wi.semaphoreCount = rec_timeline ? 2 : 1;
                          wi.pSemaphores = s;
                          wi.pValues = v;
                          vkWaitSemaphores(dev.device, &wi, timeout_ns);
                      });
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = dev.queue_family;
    pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    int slot = 0;
    for (Recording& r : recordings) {
        r.slot = slot++;
        VkFenceCreateInfo rfi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(dev.device, &rfi, nullptr, &r.fence) != VK_SUCCESS) return false;
        if (vkCreateCommandPool(dev.device, &pi, nullptr, &r.pool) != VK_SUCCESS) return false;
        if (vkCreateCommandPool(dev.device, &pi, nullptr, &r.upload_pool) != VK_SUCCESS) return false;
        ai.commandPool = r.pool;
        ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev.device, &ai, &r.cmd);
        ai.commandPool = r.upload_pool;
        vkAllocateCommandBuffers(dev.device, &ai, &r.upload);
    }
    ai.commandBufferCount = 1;
    for (Frame& f : frames) {
        if (vkCreateCommandPool(dev.device, &pi, nullptr, &f.pool) != VK_SUCCESS) return false;
        ai.commandPool = f.pool;
        vkAllocateCommandBuffers(dev.device, &ai, &f.cmd);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(dev.device, &fi, nullptr, &f.done);
    }
    {   // (the swapchain image's own fence: present_image waits for it on the CPU, image_free)
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(dev.device, &fi, nullptr, &acquire_fence) != VK_SUCCESS) return false;
    }
    if (par_threads() >= 2 && !init_pools_make(pi)) return false;   // (last: nothing after it fails and leaves them made)
    logf("vk: replay ready (%s; push descriptors %s (%u), extended dynamic state 3 %s, custom border colours %s)",
         dev.offscreen ? "offscreen" : "window", dev.push_descriptor ? "yes" : "no", dev.max_push_descriptors,
         dev.eds3_blend ? "yes" : "no", dev.custom_border_color ? "yes" : "no");
    return true;
}

/// (hookshot) The host's camera jitter (nx2/oot_jitter.py, host/rx_backend.c): the scene's size while a temporal upscaler
/// draws, NVIDIA DLSS or AMD FSR 4 (vk_temporal.cpp) or AMD FSR 3.1 (vk_fsr3_inputs.cpp), whichever it is (one at a time: the
/// Upscaler row); 0, no jitter, otherwise
void jitter_sample(MkwVkJitter* out) {
    *out = temporal_jitter_sample();
    if (!out->width) *out = fsr3_inputs::jitter_sample();
}

int jitter_option(int id) {
    const int axis = id - HOS_OPT_DLSS_JITTER_W;
    const uint32_t d = temporal_jitter_size(axis);
    return (int)(d ? d : fsr3_inputs::jitter_size(axis));
}

// (hookshot) A C++ exception thrown in the replay goes up into the game's thread, whose C frames have no handler: the
// process died with "exception 0x20474343" (a C++ throw nothing caught, then 0xc00000ff while unwinding) and the log never
// said what was thrown. At the entries the game's threads call most, its type and message are logged first, then it goes on
// up as before (the crash report flushes the log)
template <auto F> struct Guarded;
template <typename R, typename... A, R (*F)(A...)> struct Guarded<F> {
    static R call(A... a) {
        try {
            return F(a...);
        } catch (const std::exception& e) {
            logf("vk: C++ exception in the renderer, not caught: %s", e.what());
            throw;
        } catch (...) {
            logf("vk: C++ exception in the renderer, not caught (not a std::exception)");
            throw;
        }
    }
};

void fill_replay_table(MkwVkRenderer& t) {
    t.stats = stats;
    t.nvn_submit = Guarded<&submit>::call;
    t.nvn_submit_q = Guarded<&submit_q>::call;
    t.nvn_fence_q = fence_q;
    t.nvn_fence = fence;
    t.nvn_pushed = get_pushed;
    t.nvn_replayed = get_replayed;
    t.nvn_wait = Guarded<static_cast<int (*)(uint64_t, uint64_t)>(&wait)>::call;
    t.wait_writebacks_reading = wait_writebacks_reading;
    t.nvn_finish = Guarded<static_cast<void (*)()>(&finish)>::call;
    t.nvn_present = Guarded<&present>::call;
    t.nvn_register_texture = register_texture_queued;
    t.nvn_event_signal = event_signal_queued;
    t.option = [](int id) -> int {   // (3: the fixed path's scale in 1/10000, vk_menu.cpp; DLAA: full scale, vk_dlss.cpp)
        if (id == 11) return opts.stick_dead_zone;
        if (id == HOS_OPT_WALK_MODE) return opts.walk_mode;
        if (id == HOS_OPT_WALK_SPEED) return opts.walk_speed;
        if (id == HOS_OPT_CHEAT_HEARTS) return menu_heart_protection();
        if (id == HOS_OPT_CHEAT_FLIGHT) return menu_player_flight();
        if (id == HOS_OPT_CHEAT_ACTION) return menu_take_cheat_action();
        if (id == HOS_OPT_CHEAT_CAMERA) return menu_free_camera();
        if (id == HOS_OPT_ITEM_WHEEL) return opts.item_wheel != 0;   // (hookshot: host/oot_wheel.c)
        if (id == HOS_OPT_CYCLE_ITEMS) return opts.cycle_items;
        if (id == OOT_CHEAT_AGE_ID) return menu_link_age();
        if (id >= OOT_CHEAT_ITEM_BASE && id < OOT_CHEAT_ITEM_BASE + 4096) return menu_item_count(id - OOT_CHEAT_ITEM_BASE);
        if (id == HOS_OPT_TELEPORT_PICTURES) return menu_teleport_captures_on();   // (hookshot: host/hos_main.c oot_capture_tick)
        if (id == HOS_OPT_CHEAT_SPEED) return menu_link_speed();
        if (id == HOS_OPT_HDR_NATIVE) return dev.hdr && opts.hdr == 2;
        if (id == HOS_OPT_HDR_TONEMAP) return scene_tonemap_on();
        if (id == HOS_OPT_HDR_PEAK) return std::clamp(opts.hdr_peak, 400, 10000);
        if (id == HOS_OPT_HDR_WHITE) return std::clamp(opts.hdr_white, 80, 500);
        if (id == HOS_OPT_DISCORD) return opts.discord != 0;
        if (id == HOS_OPT_VOLUME) return opts.volume;
        if (id == HOS_OPT_AUDIO_OUTPUT) return opts.audio_output;
        if (id == HOS_OPT_MUTE_BACKGROUND) return opts.mute_background;
        if (id == HOS_OPT_DISABLE_CAMERA_CENTERING) return opts.disable_camera_centering;
        if (id == HOS_OPT_AUTO_RUN) return opts.auto_run;   // (host/nn_hid.c and host/auto_run.h)
        if (id == HOS_OPT_TEXT_SIZE) return opts.text_size;
        if (id == HOS_OPT_CAMERA_SPEED) return opts.camera_speed;
        if (id == HOS_OPT_PROMPTS) return opts.prompts;
        if (id == HOS_OPT_FOV) return opts.fov;
        if (id == HOS_OPT_TRANSITION_AUTOSAVE) return menu_transition_autosave();
        if (id == HOS_OPT_SHADOW_RES) return menu_shadow_res();
        if (id == HOS_OPT_DAMAGE_TAKEN) return menu_damage_taken();
        if (id == HOS_OPT_SKIP_OPENING) return opts.skip_opening;
        if (id == HOS_OPT_FPS_CAP) return opts.fps_cap;
        if (id == HOS_OPT_HOLD_TOGGLE) return opts.hold_toggle;
        if (id == HOS_OPT_MOUSE_INVERT_X) return opts.mouse_invert_x;
        if (id == HOS_OPT_HUD_LAYOUT) return opts.hud_layout;
        if (id == HOS_OPT_SKIP_TITLE) return opts.skip_title != 0;
        if (id == HOS_OPT_RENDER_THREAD) return async_on();   // (host/nvn_native.c: the present poll answered at once)
        if (id == HOS_OPT_GFX_FOLIAGE) return opts.gfx_foliage;
        if (id == HOS_OPT_GFX_OBJECTS) return opts.gfx_objects;
        if (id == HOS_OPT_GFX_GRASS) return opts.gfx_grass;
        if (id == HOS_OPT_GFX_SHADOWS) return opts.gfx_shadows;
        if (id == HOS_OPT_GFX_CONTACT_SHADOWS) return opts.gfx_contact_shadows;
        if (id == HOS_OPT_GFX_BASE_QUALITY) return opts.gfx_base_quality;
        if (id == HOS_OPT_GFX_UNCAPPED) return opts.gfx_uncapped;
        if (id == HOS_OPT_NATIVE_ZSTD) return native_zstd_default.load(std::memory_order_acquire);   // (host/hos_zstd.c)
        /* (hookshot) Vibration (host/vibration.h), and what plays the controller used last as the host wrote it back */
        if (id == HOS_OPT_VIBRATION) return opts.vibration;
        if (id == HOS_OPT_VIBRATION_STATUS) return menu_vibration_status_option();
        return id == 0 ? opts.realtime : id == 1 ? menu_is_open() : id == 2 ? resolution_auto() && !temporal_full_scale() : id == 3 ? (temporal_full_scale() ? 10000 : resolution_scale()) : id == 4 ? opts.pad_background : id == 5 ? opts.text_speed : id == 6 ? opts.mouse_native :
               id == 7 ? opts.mouse_sens : id == 8 ? opts.mouse_invert : id == 9 ? opts.rumble : id == 16 ? opts.invert_cam_x : id == 17 ? opts.invert_cam_y : id == HOS_OPT_GYRO ? opts.gyro : id == HOS_OPT_GYRO_SENS_X ? opts.gyro_sens_x : id == HOS_OPT_GYRO_SENS_Y ? opts.gyro_sens_y : id == HOS_OPT_GYRO_INV_X ? opts.gyro_inv_x : id == HOS_OPT_GYRO_INV_Y ? opts.gyro_inv_y : id == HOS_OPT_GYRO_DEADZONE ? opts.gyro_deadzone :
               id == HOS_OPT_DLSS_JITTER_W || id == HOS_OPT_DLSS_JITTER_H || id == HOS_OPT_DLSS_JITTER_PHASES ? jitter_option(id) : 0;
    };
    t.option_set = menu_option_set;
    t.jitter_sample = jitter_sample;
    t.frame_info = frame_info;
    t.shot = shot_ask;
    t.setting = menu_setting;
    t.setting_set = menu_setting_set;
    t.nvn_register_sampler = register_sampler_queued;
    t.nvn_set_texture_pool_size = set_texture_pool_size;
    t.nvn_texture_released = texture_released_of;
    t.nvn_program_released = program_released_locked;
    t.nvn_program_created = programs_created;
    programs_life_on(), programs_load_at_create();
    t.nvn_read_texels = read_texels;
    t.nvn_write_texels = write_texels;
    t.nvn_invalidate = invalidate;
    t.nvn_invalidate_quick = invalidate_quick;
    t.nvn_pool_created = memory_pool_created;
    t.guest_imported = memory_guest_imported;
    t.nvn_vmap_changing = vmap_changing;
    t.nvn_texture_created = texture_created_of;
}

std::atomic<bool> par_active{false};
namespace {
struct alignas(64) ParCounts {   // (one cache line a thread)
    std::atomic<int64_t> locks{0}, wait_ns{0};
};
constexpr int kParCountSlots = 32;
ParCounts par_counts[kParCountSlots];
std::atomic<int> par_count_next{0};
int par_count_slot() {
    static thread_local int slot = par_count_next.fetch_add(1, std::memory_order_relaxed) % kParCountSlots;
    return slot;
}
} // namespace
void par_counts_add(int64_t locks, int64_t wait_ns) {
    ParCounts& c = par_counts[par_count_slot()];
    // (only this thread writes its slot, so a load and a store: no locked add. Two threads sharing a slot past 32 lose a count)
    c.locks.store(c.locks.load(std::memory_order_relaxed) + locks, std::memory_order_relaxed);
    if (wait_ns) c.wait_ns.store(c.wait_ns.load(std::memory_order_relaxed) + wait_ns, std::memory_order_relaxed);
}
void par_counts_sum(int64_t* locks, int64_t* wait_ns) {
    int64_t l = 0, w = 0;
    for (const ParCounts& c : par_counts) l += c.locks.load(std::memory_order_relaxed), w += c.wait_ns.load(std::memory_order_relaxed);
    *locks = l, *wait_ns = w;
}

// ---- RX_VK_PARLOCK_STATS=1: where the recording threads take ParLock, how long they wait for it and hold it, and
// what a segment's draws cost by their place in it (blocks of 32), logged every 240 parallel command buffers. A diagnostic: it
// adds two clock reads to every taking ----
bool par_stats_on = [] { const char* e = getenv("RX_VK_PARLOCK_STATS"); return e && atoi(e) != 0; }();
namespace {
constexpr int kParSites = 128, kParBlocks = 10;
struct ParSite {
    std::atomic<uint64_t> key{0};
    const char* file = nullptr;
    unsigned line = 0;
    std::atomic<int64_t> n{0}, wait_ns{0}, waited{0}, hold_ns{0};
};
ParSite par_sites[kParSites];
struct ParBlock {
    std::atomic<int64_t> draws{0}, ns{0}, locks{0};
};
ParBlock par_blocks[kParBlocks];
thread_local int par_seg_draw = -1;     // (draws begun in the thread's segment; -1: not in one)
thread_local int64_t par_block_t0 = 0, par_block_locks = 0;
thread_local int64_t par_first_draw = 0, par_last_draw = 0;
std::atomic<int64_t> par_parts[6];   // (setup, lead, draws, tail, finish ns; segments)
void par_block_close(int draws) {
    if (par_seg_draw <= 0 || draws <= 0) return;
    const int b = std::min((par_seg_draw - 1) / 32, kParBlocks - 1);
    par_blocks[b].draws.fetch_add(draws, std::memory_order_relaxed);
    par_blocks[b].ns.fetch_add(par_now_ns() - par_block_t0, std::memory_order_relaxed);
    par_blocks[b].locks.fetch_add(par_block_locks, std::memory_order_relaxed);
}
} // namespace
int par_stats_site(const char* file, unsigned line) {
    par_block_locks++;
    const uint64_t key = ((uint64_t)(uintptr_t)file ^ (uint64_t)line << 48) | 1;
    for (uint32_t i = (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 57), n = 0; n < (uint32_t)kParSites; n++, i = (i + 1) % kParSites) {
        uint64_t k = par_sites[i].key.load(std::memory_order_acquire);
        if (k == key) return (int)i;
        if (!k) {
            static std::mutex mu;
            std::lock_guard lk(mu);
            k = par_sites[i].key.load(std::memory_order_acquire);
            if (k && k != key) continue;
            if (!k) {
                par_sites[i].file = file, par_sites[i].line = line;
                par_sites[i].key.store(key, std::memory_order_release);
            }
            return (int)i;
        }
    }
    return -1;
}
void par_stats_note(int site, int64_t wait_ns, int64_t hold_ns, bool released) {
    if (site < 0) return;
    ParSite& s = par_sites[site];
    if (released) s.n.fetch_add(1, std::memory_order_relaxed), s.hold_ns.fetch_add(hold_ns, std::memory_order_relaxed);
    if (wait_ns) s.waited.fetch_add(1, std::memory_order_relaxed), s.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
}
void par_stats_segment(bool begin) {
    if (!par_stats_on) return;
    if (!begin) {
        if (par_seg_draw > 0) par_block_close((par_seg_draw - 1) % 32 + 1);
        par_seg_draw = -1;
        return;
    }
    par_seg_draw = 0;
    par_block_t0 = par_now_ns(), par_block_locks = 0;
}
void par_stats_segment_parts(int64_t t0, int64_t ta, int64_t tc, int64_t t1) {
    if (!par_stats_on) return;
    const int64_t fd = par_first_draw ? par_first_draw : tc, ld = par_last_draw ? par_last_draw : tc;
    par_parts[0].fetch_add(ta - t0), par_parts[1].fetch_add(fd - ta), par_parts[2].fetch_add(ld - fd);
    par_parts[3].fetch_add(tc - ld), par_parts[4].fetch_add(t1 - tc), par_parts[5].fetch_add(1);
    par_first_draw = par_last_draw = 0;
}
void par_stats_draw() {
    if (!par_stats_on || par_seg_draw < 0) return;
    par_last_draw = par_now_ns();
    if (!par_first_draw) par_first_draw = par_last_draw;
    if (par_seg_draw > 0 && par_seg_draw % 32 == 0) {
        par_block_close(32);
        par_block_t0 = par_now_ns(), par_block_locks = 0;
    }
    par_seg_draw++;
}
namespace {
std::atomic<int64_t> par_why_n[PW_COUNT];
}
void par_why(int kind) {
    if (kind >= 0 && kind < PW_COUNT && par_stats_on && par_active.load(std::memory_order_relaxed)) par_why_n[kind].fetch_add(1, std::memory_order_relaxed);
}
void par_stats_log() {
    if (!par_stats_on) return;
    {
        static const char* names[PW_COUNT] = {"bind written", "bind no page", "bind page stale", "bind page range", "bind mirror state",
                                              "bind sync epoch", "bind sync segment", "bind past the pool's end", "view new", "view version", "view destroyed",
                                              "touch by view", "touch by slot", "slot new", "slot other handle", "slot version",
                                              "slot destroyed", "touch: not uploaded", "touch: a later segment's upload",
                                              "touch: not checked this submission"};
        char w[1400];
        int n = 0;
        for (int k = 0; k < PW_COUNT; k++)
            n += snprintf(w + n, sizeof w - n, "%s %s %.1f", k ? "," : "", names[k], par_why_n[k].exchange(0) / 240.0);
        logf("vk: parallel replay, why the lookups went to the lock, a command buffer:%s", w);
    }
    struct Row { int64_t n, waited, wait, hold; const char* file; unsigned line; };
    std::vector<Row> rows;
    for (ParSite& s : par_sites)
        if (s.key.load(std::memory_order_acquire)) {
            Row r{s.n.exchange(0), s.waited.exchange(0), s.wait_ns.exchange(0), s.hold_ns.exchange(0), s.file, s.line};
            if (r.n) rows.push_back(r);
        }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.wait + a.hold > b.wait + b.hold; });
    char line[1600];
    int n = 0;
    for (size_t i = 0; i < rows.size() && i < 14 && n < (int)sizeof line - 140; i++) {
        const char* f = rows[i].file;
        for (const char* q = f; *q; q++)
            if (*q == '/' || *q == 92) f = q + 1;
        n += snprintf(line + n, sizeof line - n, "%s %s:%u %.1f (waited %.1f, %.3f ms; held %.3f ms)", i ? "," : "", f, rows[i].line,
                      rows[i].n / 240.0, rows[i].waited / 240.0, rows[i].wait / 240e6, rows[i].hold / 240e6);
    }
    logf("vk: parallel replay, the lock by place, a command buffer: taken (waited, waiting ms; held ms):%s", n ? line : " none");
    n = 0;
    for (int b = 0; b < kParBlocks && n < (int)sizeof line - 80; b++) {
        const int64_t d = par_blocks[b].draws.exchange(0), ns = par_blocks[b].ns.exchange(0), lk = par_blocks[b].locks.exchange(0);
        if (d)
            n += snprintf(line + n, sizeof line - n, "%s %d-%d: %.2f us (%.2f locks)", n ? "," : "", b * 32,
                          b == kParBlocks - 1 ? 9999 : b * 32 + 31, ns / 1e3 / (double)d, (double)lk / (double)d);
    }
    logf("vk: parallel replay, a draw by its place in its segment:%s", n ? line : " none");
    {
        int64_t v[6];
        for (int k = 0; k < 6; k++) v[k] = par_parts[k].exchange(0);
        const double segs = v[5] ? (double)v[5] : 1.0;
        logf("vk: parallel replay, a segment in parts (ms): set up %.3f, before its first draw %.3f, its draws %.3f, from its last "
             "draw on %.3f, ended %.3f (%.0f segments)", v[0] / segs / 1e6, v[1] / segs / 1e6, v[2] / segs / 1e6, v[3] / segs / 1e6,
             v[4] / segs / 1e6, (double)v[5]);
    }
}
std::atomic<int64_t> par_lock_wait_ns{0}, par_locks{0};
std::recursive_mutex& par_mutex() {
    return par_mu;
}
int64_t par_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool replay_tex_storage_in(uint64_t gpu, uint64_t size) { return tex_pages_any(gpu, size); }

} // namespace mkwvk
