"""RX_VK_EARLY_GPU: a big command buffer's work goes to the GPU as it is recorded, in the queue's order.

replay_handle_early's submit rule (the submit_ready lambda of gpu/vk/vk_replay.cpp) is cut out and compiled with stub segments
and a stub submit_job that records the order:
- the segments go out in their order only, each once, whatever order they finish in, and none past one not done;
- once the dry pass has recorded an upload (dry_uploaded), no segment goes out until rd has (sub_after_rd); then all that are done;
- random finishing orders and upload moments over 100000 rounds keep this.
And, by the text:
- off by default (early_gpu_mode: RX_VK_EARLY_GPU, 0 when unset: the old path), only on the early path with segments (early;
  Hookshot's own replay_handle does not plan that path, so its short first segment, mode 2, is not planned there);
- on: rec_cur (the pre-sync's copies, the readback refresh) is submitted before the dry pass, rd is begun and made rec_cur and the
  dry pass's command buffer, and the upload watch is set before the dry pass and cleared right after it;
- the watch is set by upload_cmd before its first command is recorded (vk_texture.cpp);
- a worker submits under sub_mu after its segment is marked done; at the dry pass's end rd is submitted under sub_mu before
  sub_after_rd is set, and before the replay thread records the last segment; every segment goes through submit_job;
- off: rec_cur submitted after the dry pass as before, each segment by the replay thread in order.

python -I -B tests/test_early_gpu.py --cxx <compiler>
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def function(source, marker):
    start = source.index(marker)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def need(cond, why):
    if not cond:
        raise SystemExit("FAIL: " + why)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    rp = (ROOT / "gpu/vk/vk_replay.cpp").read_text(encoding="utf-8")
    tex = (ROOT / "gpu/vk/vk_texture.cpp").read_text(encoding="utf-8")

    m = function(rp, "int early_gpu_mode() {")
    need('const int m = e ? std::clamp(atoi(e), 0, 2) : 0;' in m and 'getenv("RX_VK_EARLY_GPU")' in m, "RX_VK_EARLY_GPU is not off by default")
    rh = function(rp, "const uint8_t* replay_handle(VkCommandBuffer& cmd, const uint8_t* h, bool may_stop")
    need("early_gpu_mode()" not in rh, "replay_handle asks for the early GPU path (Hookshot's does not plan it)")

    uc = function(tex, "VkCommandBuffer upload_cmd() {")
    need(uc.index("up_watch->store(true, std::memory_order_release);") < uc.index("full_barrier(up_cmd);"),
         "upload_cmd: the watch is not set before the first command is recorded")

    e = function(rp, "const uint8_t* replay_handle_early(")
    need("const int egpu = early ? early_gpu_mode() : 0;" in e, "early submits outside the path with segments")
    on = e[e.index("if (egpu) {"):e.index("bool init_lost = false;")]
    i_dry = e.index("jt_mark(13, 0);")
    need(e.index("if (egpu) {") < i_dry, "the early submit is not before the dry pass")
    for step in ("submit_recording(r, false);", "rd = &begin_recording(rec_cur_value);", "rec_cur = rd;", "cmd = rd->cmd;",
                 "texture_upload_watch(&dry_uploaded);"):
        need(step in on, "on, before the dry pass: " + step)
    need(on.index("submit_recording(r, false);") < on.index("rd = &begin_recording(rec_cur_value);"), "rd begun before rec_cur went out")
    need(e.index("VkCommandBuffer none = cmd;") > e.index("cmd = rd->cmd;"), "the dry pass does not run on rd's command buffer")
    after = e[i_dry:]
    need(after.index("if (egpu) texture_upload_watch(nullptr);") < after.index("const size_t used = open + 1;"),
         "the upload watch outlives the dry pass")

    rj = e[e.index("const auto record_job = "):e.index("// the dry pass, handing segments out")]
    need(rj.index("j->done = true;") < rj.index("std::lock_guard sl(sub_mu);") < rj.index("submit_ready();"),
         "a worker submits before its segment is done, or without sub_mu")
    cut = e[e.index("*cut = true;"):]
    on_end = cut[cut.index("if (egpu) {"):cut.index("} else {")]
    need(on_end.index("std::lock_guard sl(sub_mu);") < on_end.index("submit_recording(*rd, false);") < on_end.index("sub_after_rd = true;")
         < on_end.index("submit_ready();"), "rd is not submitted under sub_mu before the held segments")
    need(cut.index("submit_recording(*rd, false);") < cut.index("record_job(last);"), "rd goes after the last segment is recorded")
    off_end = cut[cut.index("} else {"):cut.index("jt_mark(20, 0);")]
    need("submit_recording(r, false);" in off_end, "off: rec_cur is not submitted after the dry pass as before")
    loop = cut[cut.index("record_job(last);"):cut.index("par_pool.done.wait(")]
    need("submit_job(j);" in loop and "submit_ready();" in loop, "the replay thread's own submits are not in order")
    sj = function(e, "const auto submit_job = [&init_lost](Job* j)")
    sub = "submit_or_retry(si, j->r->fence)" if "submit_or_retry(si, j->r->fence)" in sj else "vkQueueSubmit2(dev.queue, 1, &si, j->r->fence)"
    need(sub in sj and sj.index("queue_flush_locked();") < sj.index(sub),   # (submit_or_retry: a refusal for want of memory made again)
         "submit_job: the held recordings (rec_cur among them) do not go first")

    sr = e[e.index("const auto submit_ready = [&] {"):]
    sr = sr[:sr.index("};") + 2]
    body = sr[sr.index("{") + 1:sr.rindex("}")]

    fixture = r'''
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)
struct Job { std::atomic<bool> done{false}; int index = 0; };
struct World {
    std::vector<std::unique_ptr<Job>> jobs;
    std::vector<int> order;
    std::atomic<bool> dry_uploaded{false};
    bool sub_after_rd = false;
    size_t next_sub = 0;
    void submit_job(Job* j) { order.push_back(j->index); }
    void submit_ready() {
BODY
    }
};
int main() {
    std::mt19937 rng(7);
    for (int round = 0; round < 100000; round++) {
        World w;
        const int n = 2 + (int)(rng() % 9), used = 1 + (int)(rng() % n);
        for (int k = 0; k < n; k++) w.jobs.push_back(std::make_unique<Job>()), w.jobs.back()->index = k;
        std::vector<int> finish(used);
        for (int k = 0; k < used; k++) finish[k] = k;
        std::shuffle(finish.begin(), finish.end(), rng);
        const int upload_at = (int)(rng() % (used + 2)) - 1;   // (-1: never; else before the k-th finish)
        const int dry_end = (int)(rng() % (used + 1));          // (the dry pass ends after this many finishes)
        for (int f = 0; f <= used; f++) {
            if (f == dry_end) { w.sub_after_rd = true; w.order.push_back(-1); w.submit_ready(); }   // (rd, then the held ones)
            if (f == used) break;
            if (f == upload_at && !w.sub_after_rd) w.dry_uploaded = true;
            w.jobs[finish[f]]->done = true;
            const size_t before = w.order.size();
            w.submit_ready();
            if (w.dry_uploaded && !w.sub_after_rd) CHECK(w.order.size() == before);   // (held while rd has not gone)
            for (size_t i = 0, k = 0; i < w.order.size(); i++) {   // (in order, each once, none not done)
                if (w.order[i] < 0) continue;
                CHECK(w.order[i] == (int)k);
                CHECK(w.jobs[k]->done.load());
                k++;
            }
        }
        if (!w.sub_after_rd) { w.sub_after_rd = true; w.order.push_back(-1); }
        w.submit_ready();
        int k = 0;
        for (int x : w.order) if (x >= 0) CHECK(x == k++);
        CHECK(k == used);   // (every segment out by the end, none past the last used)
    }
    std::puts("ok");
}
'''.replace("BODY", body)
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "t.cpp"), os.path.join(d, "t.exe")
        Path(src).write_text(fixture, encoding="utf-8")
        r = subprocess.run([args.cxx, "-std=c++20", "-O1", src, "-o", exe], capture_output=True, text=True)
        need(r.returncode == 0, "the submit rule does not compile:\n" + r.stdout + r.stderr)
        r = subprocess.run([exe], capture_output=True, text=True)
        need(r.returncode == 0 and "ok" in r.stdout, "the submit rule: " + r.stdout + r.stderr)
    print("PASS early GPU: off by default, rec_cur before the dry pass, its uploads held behind rd, segments in order")


if __name__ == "__main__":
    main()
