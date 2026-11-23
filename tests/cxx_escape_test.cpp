// A C++ exception that leaves the renderer (gpu/vk/cxx_escape.h, renderer_guard.h), with the real headers and hooks
// of its own: the log printed, out of memory printed and exit 4, the FATAL line printed and exit 3 (in the game neither returns).
// The functions are the real MkwVkRenderer table's members behind guard_renderer, called as the host calls them; the threads are
// real ones. No game, window or GPU. One case a process: tests/test_cxx_escape.py runs each and reads what it printed.
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#if defined(__GLIBC__) || __has_include(<gnu/libc-version.h>)
#include <pthread.h>
#define HAVE_DEFAULT_STACK 1   // (pthread_setattr_default_np: a thread that cannot be started, on demand)
#endif

#ifndef _Static_assert   // (gpu/render/rx_nvn.h is C: clang, the build's compiler, takes its asserts in C++ too; g++ does not)
#define _Static_assert static_assert
#endif
#include "gpu/vk/renderer_guard.h"

using namespace mkwvk;

static void fail(const char* why) {
    std::printf("FAIL %s\n", why);
    std::fflush(stdout);
    std::_Exit(1);
}
#define CHECK(x) \
    do { \
        if (!(x)) fail(#x); \
    } while (0)

static std::atomic<int> logs{0};
static void hook_log(const char* line) {
    logs++;
    std::printf("LOG %s\n", line);
    std::fflush(stdout);
}
[[noreturn]] static void hook_oom(const char* what) {
    std::printf("OOM %s\n", what);
    std::fflush(stdout);
    std::_Exit(4);
}
[[noreturn]] static void hook_fatal(const char* why) {
    std::printf("FATAL %s\n", why);
    std::fflush(stdout);
    std::_Exit(3);
}
[[noreturn]] static void wait_for_the_thread() {   // (a thread's escape ends the process; this one waits for it)
    std::this_thread::sleep_for(std::chrono::seconds(20));
    fail("the thread's exception did not stop the process");
    std::_Exit(1);
}

// (the table's functions as the renderer gives them: each does what the case asks)
static int option_calls = 0;
static int option(int id) {
    option_calls++;
    if (id == 1) throw std::bad_alloc();
    if (id == 2) {
        std::vector<int> v;
        v.reserve(SIZE_MAX / 2);   // (a size past any PC's: std::length_error)
    }
    if (id == 3) return std::vector<int>{1, 2, 3}.at(7);
    if (id == 4) throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again), "thread constructor failed");
    if (id == 5) throw std::system_error(std::make_error_code(std::errc::permission_denied), "a lock");
    return id * 3;
}
static void submit(const uint64_t* handles, int n) {
    if (n == 99) throw std::bad_alloc();
    if (n == 98) throw std::bad_array_new_length();
    (void)handles;
}

#ifdef HAVE_DEFAULT_STACK
static void no_threads() {   // (every thread from here asks for a stack no PC has: pthread_create says EAGAIN, as at the commit limit)
    pthread_attr_t a;
    pthread_attr_init(&a);
    CHECK(pthread_attr_setstacksize(&a, (size_t)1 << 47) == 0);   // (128 TB: past the address space)
    CHECK(pthread_setattr_default_np(&a) == 0);
    pthread_attr_destroy(&a);
    bool refused = false;
    try {
        std::thread([] {}).join();
    } catch (const std::system_error&) {
        refused = true;
    }
    if (!refused) {   // (this system gave it anyway: nothing to see here)
        std::printf("SKIP no thread is refused here\n");
        std::exit(0);
    }
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) fail("no case");
    const std::string c = argv[1];
    cxx_escape::g_hooks = {hook_log, hook_oom, hook_fatal};
    MkwVkRenderer t{};
    t.option = option;
    t.nvn_submit = submit;
    guard_renderer(t);
    if (c == "pass") {   // (a call that throws nothing: its arguments and result through, nothing said; a null left null)
        CHECK(t.option != option && t.nvn_submit != submit);
        CHECK(t.option(7) == 21 && option_calls == 1);
        const uint64_t h[2] = {1, 2};
        t.nvn_submit(h, 2);
        CHECK(!t.nvn_present && !t.start && !t.guest_release);
        guard_renderer(t);   // (again: not wrapped twice)
        CHECK(t.option(8) == 24 && option_calls == 2);
        CHECK(logs == 0);
        std::printf("PASS\n");
        return 0;
    }
    if (c == "entry-bad-alloc") t.option(1);
    if (c == "entry-length") t.option(2);
    if (c == "entry-range") t.option(3);
    if (c == "entry-system-memory") t.option(4);
    if (c == "entry-system-other") t.option(5);
    if (c == "entry-submit") {
        const uint64_t h[1] = {1};
        t.nvn_submit(h, 99);
    }
    if (c == "entry-array-length") t.nvn_submit(nullptr, 98);
    if (c == "thread") {   // (the replay thread's body: its exception said and stopped on, from that thread)
        cxx_escape::spawn_needed("replay", [] { (void)std::vector<int>{1}.at(4); });
        wait_for_the_thread();
    }
    if (c == "thread-memory") {
        cxx_escape::spawn_needed("presenter", [](int n) { if (n == 3) throw std::bad_alloc(); }, 3);
        wait_for_the_thread();
    }
    if (c == "fork-join") {   // (every part made once, the helpers' starts run on them)
        std::atomic<int> parts[6] = {};
        static std::atomic<int> starts{0};
        cxx_escape::fork_join("texture gather", 6, [&](unsigned i) { parts[i]++; }, [] { starts++; });
        for (auto& p : parts) CHECK(p == 1);
        CHECK(starts == 5);
        cxx_escape::fork_join("texture gather", 1, [&](unsigned i) { parts[i]++; });
        CHECK(parts[0] == 2);
        std::printf("PASS\n");
        return 0;
    }
    if (c == "fork-join-throws") {   // (a helper's exception: said and stopped on)
        cxx_escape::fork_join("texture unswizzle", 4, [](unsigned i) {
            if (i == 2) throw std::out_of_range("part 2");
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        });
        fail("fork_join returned after a helper threw");
    }
    if (c == "fork-join-caller-throws") {   // (part(0)'s exception comes out once the helpers are joined, as it would have)
        std::atomic<int> done{0};
        try {
            cxx_escape::fork_join("texture check", 4, [&](unsigned i) {
                if (i == 0) throw std::runtime_error("part 0");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                done++;
            });
        } catch (const std::runtime_error& e) {
            CHECK(!std::strcmp(e.what(), "part 0"));
            CHECK(done == 3);
            std::printf("PASS\n");
            return 0;
        }
        fail("part(0)'s exception did not come out");
    }
#ifdef HAVE_DEFAULT_STACK
    if (c == "fork-join-no-threads") {   // (no helper can be started: the caller makes every part, said once in the log)
        no_threads();
        std::atomic<int> parts[4] = {};
        cxx_escape::fork_join("texture gather", 4, [&](unsigned i) { parts[i]++; });
        for (auto& p : parts) CHECK(p == 1);
        CHECK(logs == 1);
        std::printf("PASS\n");
        return 0;
    }
    if (c == "spawn-no-threads") {   // (spawn: false and said; spawn_needed: out of memory)
        no_threads();
        CHECK(!cxx_escape::spawn("upscaler probe", [] {}));
        CHECK(logs == 1);
        cxx_escape::spawn_needed("replay", [] {});
        fail("spawn_needed returned without its thread");
    }
#else
    if (c == "fork-join-no-threads" || c == "spawn-no-threads") {
        std::printf("SKIP no pthread_setattr_default_np here\n");
        return 0;
    }
#endif
    fail(("the case came back: " + c).c_str());
}
