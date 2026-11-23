// The load a create call queues and the first draw that needs it (gpu/vk/vk_programs.cpp: LoadJob, load_done, load_publish,
// programs_load_wait, cut from the source by tests/test_program_load_wait.py), on stand-ins for the programs' lists and the job's
// work. A first use before any worker began the job runs it itself; one that comes while a worker runs it waits for it; either
// way what the job read is in the programs' lists once when the wait returns, and a worker that takes the job later does nothing.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../gpu/vk/program_life.h"

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)
using Clock = std::chrono::steady_clock;
struct Program {
    uint64_t key = 0, variant = 1;
};
std::mutex shader_mu, life_mu;
std::unordered_map<uint64_t, std::vector<std::unique_ptr<Program>>> programs;
struct {
    std::atomic<int64_t> load_pipes{0}, load_pipes_cold{0}, load_pipes_deferred{0};
    std::atomic<int64_t> units_loaded{0}, units_run_by_draw{0}, units_waited{0}, unit_wait_ns{0};
} census;
bool programs_life_flag = false;
void logf(const char*, ...) {}
program_life::Table table;
#include "load_types.inc"
std::atomic<int> works{0};
std::atomic<bool> hold{false};
// Run the renderer's graphics-note loop with a controllable pipeline source.
struct NoteAt { unsigned variant, at, key; };
struct PipeNote {};
using VkPipeline = int;
std::atomic<bool> hold_read{false};
std::atomic<int> reads{0};
bool pipe_note_at(FILE*, unsigned, PipeNote*) {
    reads++;
    while (hold_read.load()) std::this_thread::yield();
    return true;
}
std::vector<uint64_t> deferred_keys;
void pipeline_defer(unsigned key) { deferred_keys.push_back(key); }
bool pipeline_keep_ready(unsigned, VkPipeline, const Program*) { return true; }
std::atomic<int> probes{0};
struct {
    VkPipeline graphics(const Program&, const PipeNote&) {
        probes++;
        while (hold.load()) std::this_thread::yield();
        return 1;
    }
} source;
void load_work(LoadJob& job) {
    works++;
    const std::pair<int, std::vector<NoteAt>> records{0, {{1, 0, 1}, {2, 1, 9}, {1, 2, 2}}};
    const auto* notes = &records;
    auto note = std::make_unique<PipeNote>();
    FILE* notes_in = nullptr;
    const int reads_before = reads.load();
    int64_t bytes = 0;
    for (const uint64_t key : job.keys) {
        auto p = std::make_unique<Program>();
        p->key = key;
        size_t built = 0, left = 0;   // (load_work's: whether the program's modules may go once its notes are built)
#include "load_graphics_notes.inc"
        (void)built, (void)left;
        job.made.push_back({key, std::move(p)});
    }
    CHECK(bytes == (reads.load() - reads_before) * (int64_t)sizeof(PipeNote));
}
void load_wait_note(int64_t) {}   // (the safety net of the default, gpu/vk/vk_programs.cpp: not here)
#include "load_functions.inc"

std::shared_ptr<LoadJob> queue_job(uint64_t unit, std::vector<uint64_t> keys) {
    {
        std::lock_guard lk(life_mu);
        CHECK(table.create(unit * 10, unit, true));
        table.loading(unit);
    }
    auto job = std::make_shared<LoadJob>();
    job->unit = unit, job->keys = std::move(keys);
    std::lock_guard lk(jobs_mu);
    for (const uint64_t key : job->keys) job_by_key[key] = job;
    jobs_in_flight.fetch_add(1);
    job_queue.push_back(job);
    return job;
}
// what a worker does with a job it takes from the queue (load_worker's body, without its wait)
void worker_takes(const std::shared_ptr<LoadJob>& job) {
    int queued = 0;
    if (!job->state.compare_exchange_strong(queued, 1)) return;
    load_work(*job);
    load_done(*job);   // (a worker never publishes: it never takes shader_mu)
}

int main() {
    // no job for the key: nothing happens, nothing waits
    {
        std::lock_guard lk(shader_mu);
        programs_load_wait(99);
    }
    CHECK(works == 0 && programs.empty());
    // a first use before any worker began the job: run here, published before the wait returns, counted
    auto a = queue_job(1, {11, 12});
    {
        std::lock_guard lk(shader_mu);
        programs_load_wait(12);
        CHECK(programs[11].size() == 1 && programs[12].size() == 1);
    }
    CHECK(deferred_keys == std::vector<uint64_t>({1, 2, 1, 2}));
    CHECK(probes == 0 && reads == 0 && census.load_pipes_deferred == 4);  // A draw running its own load only needs the program.
    CHECK(works == 1 && census.units_run_by_draw == 1 && census.units_waited == 0 && census.units_loaded == 1);
    CHECK(jobs_in_flight == 0 && job_by_key.empty());
    CHECK(table.find(1)->state == program_life::State::Ready);
    worker_takes(a);   // (the worker reaches it in the queue later: nothing to do, nothing published twice)
    CHECK(works == 1 && programs[11].size() == 1 && programs[12].size() == 1);
    {
        std::lock_guard lk(shader_mu);
        programs_load_wait(11);   // (a later pick of the other key: no job any more)
    }
    CHECK(works == 1 && programs[11].size() == 1);
    // a first use while a worker runs the job: waits for it, then publishes what the worker read (the worker's own publish,
    // behind shader_mu, then finds it done)
    auto b = queue_job(2, {21, 22});
    hold = true;
    std::thread worker([&] { worker_takes(b); });
    while (!probes.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::thread release([&] {
        while (!b->draw_waiting.load()) std::this_thread::yield();
        hold = false;
    });
    {
        std::lock_guard lk(shader_mu);
        programs_load_wait(21);
        CHECK(programs[21].size() == 1 && programs[22].size() == 1);   // (in the list when the wait returns: the pick finds it, and translates nothing)
    }
    worker.join(), release.join();
    CHECK(deferred_keys == std::vector<uint64_t>({1, 2, 1, 2, 2, 1, 2}));
    CHECK(probes == 1 && census.load_pipes_deferred == 7);  // The worker finished its in-flight probe and published both keys.
    CHECK(works == 2 && census.units_waited == 1 && census.units_run_by_draw == 1 && programs[21].size() == 1);
    CHECK(census.units_loaded == 2 && jobs_in_flight == 0 && job_by_key.empty());
    // a job a worker finished before any draw asked: ready, and what it read stays the job's (the worker took no lock of the
    // picks') until a pick asks for the key, which then finds it at once, with no wait counted
    auto c = queue_job(3, {31});
    worker_takes(c);
    CHECK(probes == 3);  // With no draw waiting the worker warmed both pipelines.
    CHECK(works == 3 && census.units_loaded == 3 && table.find(3)->state == program_life::State::Ready);
    CHECK(programs.count(31) == 0 && jobs_in_flight == 1);
    {
        std::lock_guard lk(shader_mu);
        programs_load_wait(31);
        CHECK(programs[31].size() == 1);
    }
    CHECK(works == 3 && census.units_waited == 1 && census.units_run_by_draw == 1 && jobs_in_flight == 0 && job_by_key.empty());
    // a job that read nothing (its records were gone): the unit is not ready, and its first use takes the path there was before
    auto d = queue_job(4, {});
    worker_takes(d);
    CHECK(table.find(4)->state == program_life::State::Created && table.use(4) == program_life::Use::Miss && census.units_loaded == 3);
    // A draw arriving during disk I/O can request that exact pipeline later, without a speculative driver call here.
    auto e = queue_job(5, {51});
    const int before = reads.load();
    hold_read = true;
    std::thread reading([&] { worker_takes(e); });
    while (reads.load() == before) std::this_thread::yield();
    std::thread finish_read([&] {
        while (!e->draw_waiting.load()) std::this_thread::yield();
        hold_read = false;
    });
    {
        std::lock_guard lk(shader_mu);
        programs_load_wait(51);
    }
    reading.join();
    finish_read.join();
    CHECK(reads == before + 1 && probes == 3 && census.load_pipes_deferred == 9);
    CHECK(deferred_keys == std::vector<uint64_t>({1, 2, 1, 2, 2, 1, 2, 1, 2}));
    std::printf("program_load_wait: ok\n");
    return 0;
}
