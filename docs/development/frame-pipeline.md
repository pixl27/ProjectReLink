# The frame pipeline: who waits for whom in one frame

With the render thread (Threaded rendering, `RX_VK_ASYNC=1`) a frame goes through four stages: the game's threads, the
render thread (`gpu/vk/vk_replay.cpp` `replay_thread_main`), the presenter thread when it is on, and the GPU. This page is
the order they run in, the waits that tie them together and why each exists, what was measured on a GPU-bound PC, and the
switches. The switches' rows are in [development.md](../development.md#time-and-profiles).

Measured on an RTX 3090, NVIDIA 620.70, Linux, 2560x1440 native, vsync off, no cap (2026-11-17): Kokiri Forest 14.4 ms a
frame with 11.0 ms of "GPU time", Hyrule Field 10.9 and 8.5.

## One frame, in order

`P(k)` is present k, `S(k)` the main thread's CPU signal for frame k, `G(k)` the moment the GPU finishes frame k.

| When (Kokiri, ms after `P(k)` is handed over) | Who | What |
|---|---|---|
| about -2.1 to -1.7 | the game's main thread and its Presentation Thread | hand over frame k's six submits (`submit_q`: the capture of the pages written, then a job). Their work stops at a `nvnCommandBufferWaitEvent` on an event the CPU signals: no draw can be recorded yet |
| about -2.2 | Presentation Thread | calls the present; it is held (`present()`, "presents after the CPU's signal": 2.2 ms each) until `S(k)` |
| about -2.2 to -0.1 | main thread | **the frame wait**: `nvnSyncWait` with no timeout (called from `0x8042cde0`) on the fence the Presentation Thread took right after `P(k-1)`: that present's value. It ends at `G(k-1)` (the log's `nvn: GPU waits in 5 s`: one a frame, 2.2 ms each). A second `nvnSyncWait` right after it (`0x8042ce34`) is on the window texture's acquire sync, which `nvnWindowAcquireTexture` hands out signalled (`fenced = 3`, fence 0): it never waits |
| -0.05 | main thread | `S(k)`: `nvnEventSignal`. A capture of what was written since the submits, and a job |
| 0 | Presentation Thread | `P(k)` handed over (a job); it takes a new fence and waits for the main thread's next message |
| +0.1 to +1.1 | render thread | the frame gate opens at `P(k)`: the six submits' jobs (memory work, six nearly empty recordings) |
| +1.1 to +10.6 | render thread | the signal's job: the stopped queues go on, 2,640 draws recorded in parts of 240, each part submitted to the GPU at its end. The GPU's first real work of the frame arrives at about +2.0 |
| +10.6 to +15.4 | render thread | the present's job: its own recording (which signals the present's fence value), the copy into the window, then `vkQueuePresentKHR`, which NVIDIA's Linux driver holds until the copy is done: `G(k)` |
| from 0 | main thread and job threads | the next frame's game logic and command recording, beside all of the above; its readbacks (`nvnBufferInvalidateMappedRange`) each wait for the part that writes their range back (`invalidate`) |

So the frame is a ring: `G(k-1)` releases the frame wait, the signal follows, the render thread records frame k, the GPU
runs it, `G(k)`. From `G(k-1)` to the GPU's first real work of frame k the GPU has nothing to do: about 2.3 ms in this
sample (0.3 until `P(k)`, 1.0 for the six submits, 0.9 for the first part). The rest of the 3.4 ms between the frame time
and "GPU time" is GPU work that line does not count (each recording's upload command buffer, the copy into the window)
and whatever the GPU idles between parts.

## The waits, and what each is for

- **The game's frame wait** (above) is the game's own: on the console it keeps the game one frame ahead of the GPU at
  most, puts its reads of GPU results (the exposure meter, the culling tables) one frame after their frame, and guards the
  per-frame memory it fills two frames ahead in turn.
- **Whole frames** (the frame gate, `gate_ready`; a test now, `RX_VK_FRAME_GATE=1`, off by default since #1121's change): the
  render thread starts a frame's jobs at its present, and the present is handed over after the CPU's signal. Replaying a frame's first jobs before the
  signal drew the lit scene black for a frame (15 flashes a session in the Market with the gate off or with the present
  still handed over before the signal, 0 with both: the comment above `gate_upto`).
- **One present ahead** (`RX_VK_PRESENTS_AHEAD`, 1): present k+1 is not accepted while present k is not replayed.
- **The present's wait** is the driver's: NVIDIA's on Linux returns from `vkQueuePresentKHR` when the frame's GPU work is
  done; elsewhere the call returns at once. The presenter thread (`RX_VK_PRESENT_THREAD=1`, off) takes it off the render
  thread: a thread and a queue of its own, one present out at most, and the next frame's acquire waits for it, so the
  swapchain is one thread's at a time.

## What the experiments say

- `RX_VK_PRESENT_THREAD=1` alone, or `RX_VK_PRESENTS_AHEAD=2`, change nothing: the render thread is free ~5 ms sooner, but
  frame k+1's draws wait for `S(k+1)`, which waits for `G(k)` in the game's frame wait.
- `RX_VK_FRAME_GATE=0` with the presenter thread: 3 to 5 % more fps, the render thread then idle 3.7 ms a frame. The six
  submits' jobs are done before the signal instead of after it (about 0.7 ms sooner to the first part). It is the
  configuration the gate was made against: jobs replayed before the signal, the present handed over before it too.
- A faster recording does not shorten the frame: the GPU is behind the render thread by the frame's end (4.5 ms of the
  copy's fence on the presenter thread), so what the recording saves the present waits.
- Two frames ahead (the closed pull request 916: the frame wait answered for the frame before last) gained 11 to 15 % on
  Windows and made shadows flicker; the cause was not found.

So the GPU's idle time at the frame's start cannot be given back by moving work on the render thread: nothing of frame k
can be recorded before `S(k)`, and `S(k)` follows the frame wait. Only an earlier answer to that wait moves it.

## What is counted, always: `vk: frame overlap`

Two lines follow the render thread's in the log every 600 presents. All of it is per frame, on the GPU's own clock (a
timestamp at the start and the end of each recording's command buffer, and one at the start of its upload command buffer).

- `vk: frame overlap (600 presents): the GPU between recordings, ms a frame, idle + in the next one's uploads`: *up to a
  frame's first recording with a draw* (the lead: from the frame before's present, over the six submits' empty
  recordings; the window's copy is in its idle part, it has no timestamp of ours) and *after it* (all the later
  recordings together). Idle is time with nothing of ours on the GPU: the GPU waiting for the render thread. Uploads are
  the upload command buffers: mirror syncs, texture uploads, the readback pages' refresh, the page table.
- The same line: *after a frame's last recording that reads or writes the game's memory come N recordings (Z ms of GPU
  time), after its last that reads it M (W ms)*: a recording reads the game's memory at the GPU's time with a dispatch, a
  draw whose program follows pointers through the page table, or a binding or copy served in place; it writes it when its
  part's end copies anything back (results, the stream's uniform updates, event words).
- `vk: frame overlap, by place`: the same idle + uploads before the frame's 2nd to 12th recording with a draw, one pair
  each, then all later ones as one pair. A part is `RX_VK_CHUNK` (240) draws and dispatches, so the pairs follow the
  frame's passes in order: where the idle is, the GPU has outrun the recording.

## What was measured (Kokiri Forest, RTX 3090, 2560x1440, 2026-11-17; a frame ~15 ms, GPU time 11.0 ms)

| Build | Lead: idle + uploads | After it: idle + uploads |
|---|---|---|
| parts of 240 throughout (as now) | 2.43 in all | 1.83 in all (not yet split) |
| with three short first parts of 48 draws (removed) | 1.35 + 0.16 | second 0.19 + 0.11, third 0.05 + 0.03, later ones 1.38 + 0.68 |

In both, after a frame's last recording that touches the game's memory comes 1.0 recording (the present's own) and
0.00 ms of GPU time. Fps were the same in three interleaved pairs in three areas.

**The conclusion.** The GPU waits for the recording during the frame's geometry, and the recording waits for the GPU
during the frame's last passes. The GPU is idle about 3.5 ms of a 15 ms frame: about 1.4 ms at the frame boundary and about
1.4 ms among the later parts, both waiting for the render thread (the rest is the window's copy and uploads). Feeding it
sooner at the boundary only moved the idle time to mid-frame: the total, and the frame, stayed.

**What is left:** recording the geometry faster. The GPU outruns the render thread there, so the render thread's speed
through those parts is what the frame waits for. Two CPU changes of the draw path that left fps unchanged when measured
alone (descriptor reuse by section; a program-pick cache) are worth measuring again against the by-place line.

## Tried and removed

- **The game's frame wait answered early** (`RX_VK_FRAME_WAIT_EARLY`, the earlier study's option A: answer it once the
  frame is recorded whole and the GPU is past its last recording that reads the game's memory). No ceiling: W = 0.00 ms,
  answered early 0 times in 600 presents. A frame's last game recording both reads and writes the game's memory, so the
  only earlier answer is two frames ahead (pull request 916: 11 to 15 % on Windows, shadows flickered) by another name.
- **Short first parts** (three parts of 48 draws while the GPU had nothing left): the lead fell from 2.4 to 1.5 ms and the
  idle time moved to mid-frame (the table above). Fps flat.
- **The render thread looking for a job for 300 us before it sleeps** (for its ~0.2 ms wake): no gain seen.
- **Lifting the frame gate with the presenter thread** (`RX_VK_FRAME_GATE=0 RX_VK_PRESENT_THREAD=1`): 3 to 5 %, and the
  gate's reason stands (jobs replayed before the signal drew the lit scene black).
- **Two presents ahead** (`RX_VK_PRESENTS_AHEAD=2`), **the presenter thread alone**, **early and late readbacks**
  (`RX_VK_READBACK_EARLY=1`, `RX_VK_READBACK_LATE=newest`): no change. `RX_VK_PARALLEL=4`: slower.

## Recording ahead of the signal: the measurement (`vk: frame ahead`)

The one project left with a real ceiling (~2.5 ms of 15.4 in Kokiri Forest) is to record a frame's parts before the CPU's
signal into recordings that are not submitted, keep what each part read, and validate and submit at the signal: the
frame's submits are in the render thread's hands 1.7 to 2.1 ms before it. It only works if what a part reads is nearly
always final when the last submit is handed over. Two lines, every 600 presents, counters only (nothing is recorded early
or submitted differently), say how often:

- `vk: frame ahead (600 presents)`: the parts (recordings with a draw) made in the signal's job a frame; the share of them
  that took nothing that came with the signal; of the others, how many a frame took *pages of the signal's capture into a
  mirror* and how many *pages read as guest memory had them*, with the pages; and in how many frames the first part, and
  the first three parts (the ones that would be recorded ahead first), took nothing.
- `vk: frame ahead, what came with the signal`: the signal's capture (the pages written between the last submit's capture
  and the signal's, and the pools they lie in), and the texture and sampler registrations, texture and program releases
  and other CPU signals handed over between a frame's last submit and that signal.

What can arrive with the signal, and whether it is counted:

| What | Counted |
|---|---|
| The signal's capture: pages the game wrote after the last submit's capture | yes: its size, and per part the pages of it copied into a mirror. A pool's dirty pages are copied when a part first binds the pool, so they count for the first part that uses the pool (whichever of its pages it reads) and not again for a later one: exact by pool for the first part, a lower bound for later ones |
| Dirty pages in no capture, read as guest memory has them at the replay | yes, per part |
| Words the replay reads at record time through the captures (handle words, bind arrays, object fields: `submitted_read`) from a page of the signal's capture | no: by page they are in the signal's capture's size, not per part |
| Registrations and releases queued in the window | per frame only, not by the part that uses the slot |
| Other CPU signals in the window; the signal's own event value (always new) | per frame |
| Occlusion results landing during the job (`occ_collect` at each recording's start) | no: they land at the replay's time whatever the signal does, and a count not known reads visible |
| What the render thread itself changes in the window | nothing: under the frame gate it runs none of the frame's jobs before the present |

**The decision rule.** About 90 % of the parts final in all three areas, and the first part final in nearly every frame:
go on to the smallest real step, recording the first part ahead (expect the lead down ~0.9 ms with
`RX_VK_PRESENT_THREAD=1`, and no flash in the Market). Most parts not final, or the first part rarely: nothing can be
recorded before the signal without the game's cooperation, the lead is fixed, and the project is dropped. A signal's
capture of hundreds of pages in many pools with a registration in most windows says the same even where the per-part
share looks good, since those are the kinds not counted per part.

Cost: two loads at each recording's end, and once a frame a walk of the signal's capture's page list (a few hundred
pages): about 0.01 ms a frame.

## Not done: the six empty submits

Before the signal's job the render thread runs the frame's six submits, ~0.9 ms in all. By the timeline four of them take
0.0 to 0.1 ms: their queue is already stopped, so they do their memory work (0.06 ms: the capture taken, the writes
collected) and make no recording at all (`submit_job`). The first and the last take 0.2 to 0.4 ms: each makes a recording
(the slot's reset, the occlusion results collected, the memory's begin and end, a queue submit of ~7 us) and replays up to
its wait. Their fence values are not signalled by those recordings (a stopped submission is not finished), so no waiter
could see them go; but what they hold is not nothing (the memory work's mirror syncs go into their uploads, state and
clears before the wait into the recording), and keeping one recording open across jobs needs memory work inside an open
recording, which the mirrors' bookkeeping (one submission, one epoch, one write-back) has never had. Not attempted
without a measurement of where those 0.2 to 0.4 ms go.

## Recording ahead of the signal: measured, and dropped

The `vk: frame ahead` lines were added to decide whether a frame's parts could be recorded before the game's signal
(its commands arrive about 2 ms earlier) and submitted at it. Measured standing in Kokiri Forest (RTX 3090, 1440p,
about 10 parts a frame):

```
vk: frame ahead (600 presents): 9.91 parts a frame recorded in the signal's job, 69.6% of them took nothing that came with the signal; of the others, 3.01 a frame took pages of the signal's capture into a mirror (56.8 pages a frame) and 0.01 pages read as guest memory had them (0.3 pages a frame); a frame's first part took nothing in 0 of 595 frames, its first three in 0
```

The frame's first part was never final before the signal, so by the rule above the project is dropped. The count
charges a pool's late pages to the first part that binds the pool, so it overstates what the first part reads; but the
game writes about 57 pages just before its signal in every frame, which is what per-frame uniform data read by most
draws would look like, and that defeats the idea whichever part is charged.
