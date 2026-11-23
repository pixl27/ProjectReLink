# The Tracy profiler

A build made with `HOOKSHOT_TRACY=1` carries the [Tracy](https://github.com/wolfpld/tracy) profiler's client
(`third_party/tracy`, version 0.15.0): every game and renderer thread, the host's waits and hand-offs, its locks, the
NVN calls that do work on the game's threads and the renderer's stages on one timeline, with the frames marked where
the game presents. It is a developer's option: off by default, and a build without it compiles nothing of Tracy and
has no zone in it (`host/hos_tracy.h`'s macros are empty), so a player's build and its build time are the same as
before.

## The rule: a capture never leaves your PC

A capture holds the game's function names, its guest addresses and its timings: it is made from the game, like a
base or a runtime ([CONTRIBUTING.md](../../CONTRIBUTING.md#the-rule-with-no-exceptions)). Never upload, paste or
attach a `.tracy` file, a screenshot of the viewer or an export of it; share what you concluded, in words and numbers.

The client is built so that nothing goes out by itself: `TRACY_ONLY_LOCALHOST` (it listens on 127.0.0.1 only: no other
machine can connect), `TRACY_NO_BROADCAST` (it announces nothing on the network) and `TRACY_NO_CODE_TRANSFER` (it sends
no machine code to the viewer). `TRACY_ON_DEMAND`: it records only while a viewer or `tracy-capture` is connected, so
a run nobody captures keeps no data.

## Build

```sh
HOOKSHOT_TRACY=1 python -I scripts/dev_build.py pack     # this checkout over its base, with Tracy
python -I scripts/dev_build.py rt                        # build-linux/rt-<checkout>/, the game folder to run
```

or a whole game folder: `HOOKSHOT_TRACY=1 python -I scripts/dev_build.py build`. `nx2/build.py` reads the variable:
the host's files and the back end's own are compiled with `TRACY_ENABLE` (and their frame pointers kept, for the
sampler's call stacks), Tracy's client (`public/TracyClient.cpp`) goes into the game's executable, and the lifted game
code is not compiled again. The process has one client: the executable exports Tracy's C API (`___tracy_*`) and
`hookshot_vk.so` calls it there, as it calls the host's `hos_posix_*`. On Windows the executable's import library
(`hookshot_tracy.lib` in the work folder) is what `hookshot_vk.dll` links, and the renderer's zones reach the capture
through it. Tracy's client asks for its system libraries by `#pragma comment(lib, ...)` lines that are under
`_MSC_VER`, which llvm-mingw's clang does not define, so `nx2/build.py` names the ones the game does not link
anyway: `ws2_32` for its sockets and `secur32` for the user name it shows the viewer (`GetUserNameExA`); without the
second the executable does not link.

## Capture

The viewer and the capture tool must be Tracy **0.15.0**: the protocol is locked to the version.

```sh
tracy-capture -a 127.0.0.1 -o run.tracy -s 30      # 30 s into a file; without -s, until Ctrl-C or the game ends
tracy -a 127.0.0.1                                 # or the viewer, live; it also opens run.tracy
```

Start either before the game to have the start-up too: it waits for the game. Keep the file in the checkout's
`build-linux/` or another folder that is never committed or sent. `tracy-capture -s <seconds>` ended some captures
at once here (on Windows `-s 60` saved its minute); Ctrl-C (or `timeout -s INT <seconds> tracy-capture ...`) ends one
and saves it. `TRACY_PORT=<n>` in the game's environment and `-p <n>` give a second game a port of its own (8086 is
the default).

`tracy-csvexport run.tracy` gives each zone's count and times; with `-u`, every zone one per line (start, length,
thread), the zone's text in the last column, commas and all. Its threads are numbers, without their names: a
`vcore wait`'s text (who gave the core) is one way to tell them apart. An export is made from the game like the
capture: it stays on your PC.

The client is started by the game's `main`, not before it (`TRACY_MANUAL_LIFETIME`): the crash window's helper is
the same executable, and with a client of its own it took the next port and outlived the game, so a later capture
connected to it and recorded nothing.

The game's fibers (`nn::os::SwitchToFiber`) are not told to Tracy unless the build also has
`HOOKSHOT_TRACY_FIBERS=1` (`TRACY_FIBERS`: a fiber gets a row of its own, and its zones follow it from thread to
thread). With it every zone goes through one queue under a lock: a headless capture ran at 10 to 19 fps with it and
36 without, so it is off. No zone was seen open across a fiber switch (`host/hos_thread.c` logs the first five
that are: `tracy: a fiber switch ... with N game call zones open`); if a capture ever stops with "Invalid order of
zone begin and end events", look for that line and build with the option.

On Linux Tracy's own sampler (every thread's call stack, 10 kHz by default; `TRACY_SAMPLING_HZ=<n>` in the game's
environment changes it) works for the game's own code as an ordinary user where `/proc/sys/kernel/perf_event_paranoid`
is 2 or less. The lifted game code has no zones (hundreds of thousands of functions): the samples are how it shows,
meant to be named `f_<guest address>` from the executable's symbols. They are not yet: Tracy names a sampled address
from debug information or the dynamic symbols, the executable has neither for the lifted code, and its samples show
as `[unknown] + <address>` (exporting the names from a Tracy build's executable is being checked). Context switches, CPU usage of other programs and kernel
frames need more than a user has (`perf_event_paranoid` of -1, or the game run with `CAP_PERFMON`/as root): without
them the capture has none of those, and everything else.

On Windows Tracy samples and records context switches through Event Tracing for Windows, which its manual says
needs an administrator; the captures made here ran as a user with `TRACY_NO_SAMPLING=1`, so neither has been tried.
Nor are there zones in the game's code, whose list is for ELF objects (below): a thread that holds a core outside
every zone runs the game's code, or host code that has no zone, and the capture cannot say which.

## What the zones are

Waits are dark red, locks that had to wait bright red. A wait's text has the object's guest address and the guest
caller (`lr`), which tell the game's waits apart.

| Zone | Where | What it is |
|---|---|---|
| frame marks | `nvnQueuePresentTexture` | the game's present ends a frame |
| `event wait`, `semaphore wait`, `queue receive wait`, `WaitAny`, `cv wait`, `cv timed wait`, `mutex wait` | `host/nn_os_sync.c` | a guest thread's block on an `nn::os` object that was not ready: its spin, then `asleep` inside it; a mutex wait names the holder |
| `event signal`, `semaphore release`, `queue send`, `cv signal`, `mutex unlock: wake` | the same | the other end: the same guest address in the text |
| `kernel wait` | `hos_thread_wait` | the SVC layer's waits (mutexes, condition variables, the address arbiter) |
| `sleep`, `sleep 0 (yield)`, `YieldThread` | `host/hos_thread.c` | guest and host sleeps, with the time asked |
| `vcore wait`, `vcore yield` | `vc_acquire`, `vc_yield` | ready to run and waiting for one of the game's virtual cores; the text says which thread gave it |
| `asleep` | `hos_cv_sleep` | the slept part of any of the waits above |
| `thread join` | `hos_thread_join` | |
| `lock wait` | every SRW lock and critical section in `hos_thread.c`, `nn_os_sync.c`, `nvn_native.c`, `hos_svc.c`, `hos_mem.c` | a host lock that was held by another thread; the text is the lock (`&vc_lock`, `&copy_lock`, `&table_lock`) |
| `nvnQueueSubmitCommands`, `nvnCommandBufferEndRecording`, `nvnQueuePresentTexture`, `nvnWindowAcquireTexture`, `nvnSyncWait`, `nvnQueueFinish`, `nvn...InvalidateMappedRange`, `nvn texels`, `nvn held_flush` | `host/nvn_native.c` | the NVN calls that work or wait on a game thread |
| `rx submit_q`, `rx present`, `rx event_signal`, `enqueue` (game thread) | `gpu/vk/vk_replay.cpp` | the game's hand-over to the render thread |
| `render job` (text: submit, present, event signal; how long it waited in the queue), `submit_job`, `do_present`, `render job: held present`, `signal job: capture` | the render thread's loop | one job of the render thread |
| `record_segment`, `split_part`, `early_split`, `submit_recording`, `finish_recording`, `submit_memory_work`, `queue_flush`, `replay worker job` | `vk_replay.cpp` | the replay's parts and their submits |
| `draw`, `build_descriptors`, `bind_descriptors`, `program_for` | `vk_draw.cpp` | once per draw; `RX_VK_TRACY_DRAWS=0` leaves them out |
| `memory_sync`, `memory_table_flush`, `capture_make`, `take_capture`, `memory_capture_now`, `memory_capture_submit`, `mirror_sync`, `mirror_make`, `msync_pass_flush`, `late_fix_build` | `vk_memory.cpp` | memory sync and captures |
| `texture create`, `texture upload`, `texture upload_astc_gpu`, `texture upload_unswizzle_gpu` | `vk_texture.cpp` | |
| `build_graphics`, `gpl_link`, `cached_graphics`, `cached_compute` (pipeline) | `vk_draw.cpp` | pipeline creation and compiles |
| `worker job`, `serial build` | `vk_jobs.cpp` | the background workers: shader translation, pipeline builds |
| `recorder idle` | `vk_recorder.h` | the recorder thread with nothing to run; between two of them it runs its packets |
| `present_image`, `present_out`, `present_finish`, `present_hand_off`, `presenter_wait_copy`, `present_thread_drain` | `vk_replay.cpp` | the present |
| `recording_wait`, `wait_recorded`, `wait_recorded_part`, `early_recorded_wait`, `vk_part_wait`, `sync_drain`, `sync_settle` | `vk_replay.cpp` | waits for the replay or the GPU |

Plots: `frame ms (renderer)` every present; with `RX_VK_FRAME_TRACE=1` (which `HOS_PROFILE=1` sets) also `game_ms`,
`render_ms`, `wait_gpu_ms`, `wait_render_ms`, `wait_main_ms` and `memsync_ms`, the frame trace's columns.

Not there: Vulkan GPU zones for the passes (`RX_VK_PASS_TIMING` has them, in the log) and allocations.

## The game's own code

The lifted game code is hundreds of thousands of functions, so it gets zones from a list (`nx2/tracy_funcs.py`; Linux,
where the objects are ELF), in blue. `HOS_TRACY_GAME=0` leaves them out of a capture (`funcs` or `calls`: one
kind alone). The list:

```
<guest address> [calls] [the zone's name]
```

- **A listed function** `f_<address>` is a zone each time it runs, whoever calls it: its text is the object it ran on
  and the guest caller. Nothing is translated or compiled again for it: in the object of the chunk that defines the
  function the symbol is made undefined and its code named `__real_f_<address>` (a copy of the object with its symbol
  table rewritten), and a generated file defines `f_<address>` as the zone around it, so the direct calls of every
  chunk and the indirect ones (through `a64_table.c`) reach the zone.
- **`calls`** makes every indirect call the function itself makes a zone too (`game call`), with the function called
  (`f_<target>`) and the object it was called on in its text. This is for the game's dispatch points: a job loop
  running its jobs, the module workers their modules, the actor executors their actors. The function's chunk alone is
  compiled again (about 20 s for one chunk) with a generated header that turns the call into a zone where its return
  address lies in the listed function; the other functions of the chunk compile as before.
- **What an object is called** (`host/hos_tracy_game.c`): its class's name where its vtable has run-time type
  information (the SDK's and the middleware's classes do; the game's own classes were built without), else a name the
  object carries (the first of its first twelve fields that points at a short printable string: the game's named jobs,
  modules, threads and heaps keep one near their start; a guess, so a field pointing at some other string is shown as
  the name), else its vtable's address, the same for every object of a class. These names are read from the game:
  they stay in the capture, on your PC.

`nx2/tracy_funcs.txt` is the list every Tracy build takes: the two job loops (`calls`) and the frame's phases on the
main thread. `HOOKSHOT_TRACY_FUNCS=<file>` adds a list of your own after it (a later line for the same address
wins), for example the hottest functions of a profile. Such a list is made from the game: keep it out of the
checkout. A function the translation does not define as a function of its own (a piece of a split giant, an address
inside a function) is named in the build's log and left out.

The check behind these zones (`host/hos_tracy_game.c`): each of them that ends while it is not the newest on its thread
is logged (`tracy: a game zone ... ended out of turn`, the first eight), because Tracy stops a capture at the first
such zone and does not say which. The lifted chunk sees nothing of Tracy: its zone travels through it as 16 bytes
of the host's own (Tracy's zone context has one layout with `TRACY_ON_DEMAND` and another without, and a chunk
compiled without the define lost every zone's end).

Two limits. A zone ends when its function returns: where the game leaves a function by `longjmp` (its C++ exceptions
do not exist here, `setjmp` does), a zone that was open stays open in the capture. And a function inlined by the
compiler into a caller in its own chunk (the hot chunk at `-O2`) has no call to route through the zone there.

## Adding a zone

`host/hos_tracy.h` is the only header a call site includes (C and C++); its top lists the macros. `HT_SCOPE("name");`
at the start of a block is a zone to its end, and every use sits in `#ifdef TRACY_ENABLE` ... `#endif`, so a build
without Tracy, and the checks that compile one function cut from a file, never see it
(`tests/test_tracy_funcs.py`). Keep them off loops that run more than a few thousand times a frame, or
use `HT_SCOPE_DRAW`. A zone costs tens of nanoseconds while a capture is taken, and one call that returns at once
while none is.

## Cost

Without a viewer connected the zones record nothing. While one is, a capture costs frame rate (in a headless run
here, about 65 fps before the capture and 15 to 19 during it with the per-draw zones on; a played session with
`RX_VK_TRACY_DRAWS=0` ran at about 54 fps against about 70 without Tracy; those were builds with `TRACY_FIBERS`,
see above: without it a headless capture with every zone on ran at 36 fps) and 240,000 to 550,000 zones a second,
about 4 MB of file a second with everything on, a quarter of the zones the four per-draw ones: read the times as shares, not as an unprofiled frame's, and compare with
`RX_VK_TRACY_DRAWS=0` before reading a per-draw number. `TRACY_NO_SAMPLING=1` in the game's environment turns
Tracy's sampler off; `TRACY_SAMPLING_HZ=<n>` slows it. The
profiling mode (`HOS_PROFILE=1`) can run at the same time: its sampler interrupts each thread every 2 ms and its pass
timing makes the GPU finish each pass before the next, both of which then show in the capture as time. Use it with
Tracy for its `frames.csv` plots, not for frame-rate numbers.
