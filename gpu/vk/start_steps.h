// (hookshot) The first start after a preparation, under RX_VK_LOAD=create, runs two passes on the start-up bar's one part:
// precompile_warm (every noted pipeline built into the pipeline cache) and then precompile_at_create (each mesh fallback
// program's compute pipelines, then the kept programs' noted pipelines). Each counted its own steps: the warm reached its
// total, the bar never goes back (PrepareBar), and the screen read 100% through the second pass, 91 s on a Steam Deck (#1522).
// One count over both, as #714 did for the predicted pipelines: the warm's steps, then the second pass's, counted before the
// warm (create_steps) and at least one step held for it, so the warm's end is never the bar's. vk_draw.cpp precompile_run;
// tests/test_start_progress_at_create.py.
#pragma once

#include <atomic>
#include <cstddef>

#include "fallback_loaded.h"

namespace start_steps {

using Progress = void (*)(size_t done, size_t total);

inline std::atomic<Progress> shown{nullptr};
inline std::atomic<size_t> warmed{0}, after{0}, created{0};

/// The second pass's steps as precompile_at_create counts them (a step for each program it is given, and one more for each
/// mesh fallback program's compute pipelines), from what program_cache_load_some(..., fallbacks) gives it: every variant of
/// each key whose records it takes (`taken`: key -> its records, kind 4 first; the records read in and the variants in
/// memory already), and the mesh fallback programs in memory under other keys (fallback_loaded::each)
template <class Taken, class Programs>
size_t create_steps(const Taken& taken, const Programs& programs) {
    size_t kept = 0, fallbacks = 0;
    for (const auto& kv : taken) {
        for (const auto& d : kv.second) kept++, fallbacks += (d.kind & 4) != 0;
        const auto in = programs.find(kv.first);
        if (in != programs.end())
            for (const auto& p : in->second) kept++, fallbacks += p->fallback;
    }
    const size_t loaded = fallback_loaded::each(programs, taken, [](const auto&) {});
    return kept + fallbacks + 2 * loaded;
}

/// The passes of this start reported to progress. then: a second pass follows the warm, of `steps` (create_steps)
inline void begin(Progress progress, bool then, size_t steps) {
    warmed = 0, created = 0, after = then ? (steps ? steps : 1) : 0;
    shown = progress;
}
/// The warm's progress: its steps first, the second pass's after them
inline void warm(size_t done, size_t total) {
    warmed = total;
    if (const Progress p = shown.load()) p(done, total + after.load());
}
/// The second pass's progress, after the warm's steps (none where the notes were covered already)
inline void create(size_t done, size_t total) {
    const size_t w = warmed.load();
    created = total;
    if (const Progress p = shown.load()) p(w + done, w + total);
}
/// The second pass over: the count at its end (a pass with nothing to do reports nothing)
inline void end() {
    const size_t n = warmed.load() + created.load();
    if (const Progress p = shown.load(); p && n) p(n, n);
}

}  // namespace start_steps
