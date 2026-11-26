# Cache-preserving performance update — 2026-11-25

This batch targets camera-turn hitches and compute-fallback memory use without changing the shader translator, serialized program layouts, shader settings included in cache keys, or generated SPIR-V headers. The release checkout is `work/red-tunic-build-fix` (renderer API 36); the older workspace-root renderer is API 34 and was not installed.

The new code can share the mesh fallback's records/payload scratch between serial recording producers with `RX_VK_MFB_SHARED=1`. It is off by default following the combined-option measurements below. Parallel recorders retain private scratch, and the existing barriers still order buffer resets, task work, mesh work and drawing. The system-heap budget query now accounts for an integrated GPU whose only heap is also device-local, allowing the existing import-pressure logic to see its shared-memory budget. XInput's controller-presence flag uses atomic accesses across threads, and failed worker creation can be retried.

| Research item | Release-checkout disposition |
|---|---|
| Attribute-by-attribute mesh records ([Longshot #906](https://git.magicbean.dev/MagicBeanDev/Longshot/pulls/906)) | Already implemented; SoA remains the default. Changing record layout would invalidate shader caches, so it is unchanged. |
| Byte-range readback waits ([#2366](https://git.magicbean.dev/MagicBeanDev/Longshot/pulls/2366)) | Already implemented behind `RX_VK_WB_SPANS`; tested with the other cache-neutral options. |
| ASTC block-hash reuse and zero-depth clears ([#2343](https://git.magicbean.dev/MagicBeanDev/Longshot/pulls/2343)) | Existing conservative implementations retained. `RX_VK_HASH_MEMO` is already enabled in this PC's settings; zero-depth clears remain separately selectable. |
| Exact texture-handle staleness | Already implemented through the destruction ring and `RX_VK_GONE_RING` / `RX_VK_BINDING_REUSE`; no second invalidation mechanism added. |
| Background XInput polling ([#2124](https://git.magicbean.dev/MagicBeanDev/Longshot/pulls/2124)) | Already implemented; this batch fixes presence publication and failed-worker retry. |
| Shared fallback scratch / UMA heap accounting ([#2515](https://git.magicbean.dev/MagicBeanDev/Longshot/pulls/2515)) | Missing host-side mechanisms implemented with regression checks and parallel-recording isolation. |
| Shader loading at creation, module identifiers, asynchronous pipeline preparation and HUD separation | Already present; retained. |
| New mesh batching/culling, global-load lowering, subgroup changes and faster neural/GPU ASTC shaders | Excluded from this batch: they change generated shaders or their inputs and therefore require separate cache preparation. |
| New recorder defaults and other recompilers' presentation/interpolation systems | Architecture-specific changes are not transplanted without their prerequisites and hardware measurements. The release's existing thread/presentation policy is retained. |

## Shader compatibility

The unchanged translator source stamp is `acf893449f50d836`. With this build's ABI (`Shader::Info` 792 bytes, `Program` 312 bytes, vector 24 bytes, libc++ 230105), its translator-build stamp is `f33763fbeb8ea6dc`. Folding in program-cache format 1 gives `2f7a062bf817cf2d`, matching this PC's existing cache set. The per-device program stamp remains `fd47bc59a7a1e538`. No stamp is overridden or forged.

Validation uses an independent copy of the existing cache on the NVMe drive and a development runtime. It never hard-links writable shader caches to the originals. The smoke session loaded all 47,798 cached program keys and the existing 1,385,345 KB driver pipeline cache. At close it reported 13 pipelines obtained from the driver cache and **0 compiled**, with 4,151 additional hits from the startup cache.

## Validation

The Windows renderer linked successfully. The input source compiled as a whole translation unit, and its development executable linked by replacing one object while reusing 570 existing game objects: no lifted game code was recompiled. Linux syntax checks passed for the complete `vk_draw.cpp`, `vk_memory.cpp`, `vk_texture.cpp`, `vk_vma.cpp` and `host/nn_hid.c` translation units. Independent review checked buffer ownership, queue ordering, parallel isolation and cache-key compatibility.

The focused checks cover serial-producer changes, parallel buffer isolation, disabled fallback sharing, unified-memory and discrete heap accounting, failed XInput worker creation, SDL takeover/resumption, disconnection, fallback allocation failures and the existing grass-capacity cases. The full-suite results and comparative camera-route measurements are recorded with the local validation logs and the tested-run entries; a skipped check is not a pass.

The final 4.4.0 source suite reported 874 passed, 66 failed (including one 300-second timeout), and 58 skipped. An affected portable-archive fixture was updated to include the new launcher dependency and passes on retry. The shader-reader timeout also passed on retry (154.7 seconds). The other 64 failures reproduce in the public 4.3.0 source checkout: its comparison batch returned two passes and 64 failures. This is not a green full-suite result. The new sharing and heap tests also pass with Linux GCC; focused Windows checks use Clang. Tested-run fragment validation passes all 25 checks. The new launcher/package checks pass, including the published Windows apphost under its public filename; normal no-argument startup was observed opening the Builder.

The smoke scene was the adult save beside the lake/bridge in the game's `HyruleField_Adult` map. Hardware: Ryzen 5 5625U (6 cores / 12 threads), AMD Radeon integrated graphics (`1002:15e7`), AMD proprietary Vulkan driver 26.5.2, Windows 11 Pro for Workstations build 26300, approximately 16 GB RAM. Output 1920×1080, existing handheld/render settings and frame generation off. The camera-turn screenshot was visually inspected; landscape, water, foliage, Link and the HUD rendered. Timing is measured separately with the job queue's `timing` mode.

Four isolated 240-frame camera turns ran in off/on/on/off order, using the same executable, renderer, save and copied cache. The toggled group was `RX_VK_MFB_SHARED`, `RX_VK_WB_SPANS`, `RX_VK_DEPTH_ZERO`, `RX_VK_GONE_RING` and `RX_VK_BINDING_REUSE`; the already-enabled hash memo stayed on. The job queue recorded no outside builds or CPU pressure in these runs. These are group measurements, not an attribution to an individual switch. The new input and heap-accounting fixes were present in both groups, so their individual performance effect was not measured.

| Run | Options | Mean frame time | FPS | p99 frame time | Frames over 100 ms |
|---|---|---:|---:|---:|---:|
| before1 | off | 104.05 ms | 9.61 | 565.62 ms | 71/240 |
| after1 | on | 106.61 ms | 9.38 | 564.86 ms | 71/240 |
| after2 | on | 115.63 ms | 8.65 | 545.57 ms | 88/240 |
| before2 | off | 96.44 ms | 10.37 | 559.11 ms | 58/240 |

The enabled group averaged 111.12 ms versus 100.24 ms disabled, a 10.9% increase in frame time. It did not improve pacing in this scene, so the existing experimental options remain off and the new scratch-sharing option is opt-in. No substantial compute-fallback speedup is claimed. Shader-changing upstream algorithms need a separate cache-changing evaluation.
