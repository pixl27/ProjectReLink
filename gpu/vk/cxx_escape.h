// (hookshot) A C++ exception that leaves the renderer. Nothing above hookshot_vk.dll catches one: the host is C, and
// a renderer thread's body is the top of its stack. So a std::length_error from a vector, a std::system_error from a thread that
// could not be started, or a std::bad_alloc ended the game with exception 0x20474343 (llvm-mingw's code for a C++ exception nothing
// caught), "An unexpected error", and no word of what was thrown or where. v3.0.0 still did, with the renderer's new_handler
// (vk_main.cpp new_refused), and several of its reports have no "a C++ allocation refused" line before the crash: what
// was thrown there was not a refused operator new.
// Here the ways out of the renderer catch a std::exception and stop the game with what it was: every function it gives the host
// (renderer_guard.h, and mkwvk_init), and the bodies of the replay's own threads (the replay, the presenter, the order watch:
// spawn_needed), which the host's calls hand most of the work to. A thread that cannot be started (at the commit limit its stack
// has no commit: std::system_error) comes out of std::thread's constructor into one of those, and is out of memory there. The
// texture decode's helper threads, started in play, go through fork_join: one that cannot be started leaves its part to the
// caller (before, the vector of threads was destroyed with the others still running: std::terminate). escaped names the exception
// (its type and what()) and where it came out in the log, then:
// - std::bad_alloc (not bad_array_new_length), or a std::system_error that says no memory or no resources: out of memory, through
//   the host's line and notice as new_refused does (MkwVkHost out_of_memory, version 28);
// - any other: the renderer's FATAL line and exit code 3 (as vk_replay.cpp replay_fatal), which the crash window shows.
// Only std::exception is caught, never catch (...): a catch-all could take a foreign exception (an access violation, a driver's
// MSVC exception 0xe06d7363) away from the host's crash record, which needs its backtrace.
// No game, window or GPU needed: tests/test_cxx_escape.py runs it with hooks of its own.
#pragma once
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <typeinfo>
#include <utility>
#include <vector>

namespace mkwvk::cxx_escape {

enum class Kind { memory, size, other };
struct Seen {
    Kind kind;
    const char* type;   // (the standard type's name; another type's typeid name)
};
/// What an exception that came out is: memory, a size past any PC's memory (a size gone wrong), or anything else
inline Seen classify(const std::exception& e) noexcept {
    if (dynamic_cast<const std::bad_array_new_length*>(&e)) return {Kind::size, "std::bad_array_new_length"};
    if (dynamic_cast<const std::bad_alloc*>(&e)) return {Kind::memory, "std::bad_alloc"};
    if (dynamic_cast<const std::length_error*>(&e)) return {Kind::size, "std::length_error"};
    if (dynamic_cast<const std::out_of_range*>(&e)) return {Kind::other, "std::out_of_range"};
    if (dynamic_cast<const std::invalid_argument*>(&e)) return {Kind::other, "std::invalid_argument"};
    if (const auto* s = dynamic_cast<const std::system_error*>(&e)) {
        // (no memory, or no thread: libstdc++ says EAGAIN for a thread it could not start; libc++ on Windows may give the
        // system's own error number, so its words say it too)
        const std::error_code c = s->code();
        const bool memory = c == std::errc::not_enough_memory || c == std::errc::resource_unavailable_try_again ||
                            std::strstr(e.what(), "thread constructor failed");
        return {memory ? Kind::memory : Kind::other, "std::system_error"};
    }
    if (dynamic_cast<const std::bad_function_call*>(&e)) return {Kind::other, "std::bad_function_call"};
    if (dynamic_cast<const std::logic_error*>(&e)) return {Kind::other, "std::logic_error"};
    if (dynamic_cast<const std::runtime_error*>(&e)) return {Kind::other, "std::runtime_error"};
    return {Kind::other, typeid(e).name()};
}

/// What escaped does (set by mkwvk_init; a test sets its own). out_of_memory and fatal do not return
struct Hooks {
    void (*log)(const char* line) = nullptr;            // (the renderer's log: logf)
    void (*out_of_memory)(const char* what) = nullptr;  // (the host's line and notice, waited for, then the stop; null: an older host)
    void (*fatal)(const char* why) = nullptr;           // (the FATAL line and exit code 3, as vk_replay.cpp replay_fatal)
};
inline Hooks g_hooks;

/// The game stops: memory, through the host's out-of-memory line and notice (what names it there); else, or with an older host,
/// the renderer's FATAL line why
[[noreturn]] inline void stop(bool memory, const char* what, const char* why) {
    if (memory && g_hooks.out_of_memory) g_hooks.out_of_memory(what);
    if (g_hooks.fatal) g_hooks.fatal(why);
    std::fprintf(stderr, "vk: FATAL: %s\n", why);
    std::fflush(stderr);
    std::_Exit(3);
}

/// An exception e came out of where (a function the host calls, a thread's body): said, then the game stops. Fixed buffers only:
/// at the commit limit nothing here asks for memory
[[noreturn]] inline void escaped(const char* where, const std::exception& e) {
    const Seen s = classify(e);
    const char* what = e.what() ? e.what() : "";
    char line[512], oom[256];
    std::snprintf(line, sizeof line, "vk: a C++ exception came out of %s: %s: %s", where, s.type, what);
    if (g_hooks.log) g_hooks.log(line);
    std::snprintf(oom, sizeof oom, "the renderer's C++ allocations (%s out of %s)", s.type, where);
    std::snprintf(line, sizeof line, "%s came out of %s: %s%s", s.type, where, what,
                  s.kind == Kind::size ? " (a size past any PC's memory: one gone wrong)" : "");
    stop(s.kind == Kind::memory, oom, line);
}

/// A function the host calls through MkwVkRenderer, kept with its name: Entry<&MkwVkRenderer::m>::wrap(t, "m") puts call in
/// its place, which calls it and hands an exception out of it to escaped (renderer_guard.h, every member)
template <auto Member> struct Entry;
template <class T, class R, class... A, R (*T::*M)(A...)> struct Entry<M> {
    static inline R (*fn)(A...) = nullptr;
    static inline const char* name = "";
    static R call(A... a) {
        try {
            return fn(a...);
        } catch (const std::exception& e) {
            escaped(name, e);
        }
    }
    static void wrap(T& t, const char* n) {
        if (!(t.*M) || t.*M == &call) return;   // (none given: still none, for the host's null checks)
        fn = t.*M, name = n;
        t.*M = &call;
    }
};

inline void said_not_started(const char* name, const std::system_error& e) {
    if (!g_hooks.log) return;
    char line[256];
    std::snprintf(line, sizeof line, "vk: the %s thread could not be started (%s): %s", name, e.what(),
                  "at the commit limit its stack has no memory");
    g_hooks.log(line);
}

/// A thread's body that runs f(a...), its exceptions handed to escaped(name): for a std::thread the caller keeps (and joins), or
/// starts in a try of its own
template <class F, class... A> auto guarded(const char* name, F&& f, A&&... a) {
    return [name, f = std::forward<F>(f), ... a = std::forward<A>(a)]() mutable {
        try {
            std::invoke(std::move(f), std::move(a)...);
        } catch (const std::exception& e) {
            escaped(name, e);
        }
    };
}

/// A detached thread that runs f(a...), named name in the log, its exceptions handed to escaped. false: it could not be started
/// (said in the log): the caller goes on without it
template <class F, class... A> bool spawn(const char* name, F&& f, A&&... a) {
    try {
        std::thread(guarded(name, std::forward<F>(f), std::forward<A>(a)...)).detach();
        return true;
    } catch (const std::system_error& e) {
        said_not_started(name, e);
        return false;
    }
}

/// spawn, for a thread the game cannot go on without (the replay, the presenter, the workers): one that could not be started
/// stops the game, as out of memory (the commit limit is what refuses a thread its stack)
template <class F, class... A> void spawn_needed(const char* name, F&& f, A&&... a) {
    if (spawn(name, std::forward<F>(f), std::forward<A>(a)...)) return;
    char what[160], why[192];
    std::snprintf(what, sizeof what, "the renderer's threads (the %s thread)", name);
    std::snprintf(why, sizeof why, "the %s thread could not be started", name);
    stop(true, what, why);
}

/// part(0) .. part(n - 1): part(0) on this thread, the others on helper threads (each starting with start(), when given), all
/// joined before it returns. A helper that cannot be started (the commit limit) leaves its part, and the ones after it, to this
/// thread: done, only later. A helper's exception goes to escaped; one of part(0)'s comes out of here once the helpers are joined
template <class F> void fork_join(const char* name, unsigned n, F&& part, void (*start)() = nullptr) {
    std::vector<std::thread> pool;
    struct Join {
        std::vector<std::thread>& pool;
        ~Join() {
            for (std::thread& t : pool)
                if (t.joinable()) t.join();
        }
    } join{pool};
    pool.reserve(n > 1 ? n - 1 : 0);
    unsigned helpers = 1;
    for (; helpers < n; helpers++) {
        try {
            pool.emplace_back([&part, start, name, t = helpers] {
                if (start) start();
                try {
                    part(t);
                } catch (const std::exception& e) {
                    escaped(name, e);
                }
            });
        } catch (const std::system_error& e) {
            static std::atomic<int> said{0};   // (a few: a texture's decode asks for helpers often)
            if (said.fetch_add(1, std::memory_order_relaxed) < 4) said_not_started(name, e);
            break;
        }
    }
    part(0u);
    for (unsigned t = helpers; t < n; t++) part(t);
}

}   // namespace mkwvk::cxx_escape
