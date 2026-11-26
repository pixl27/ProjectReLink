# Camera stalls and texture work: project comparison, 26 November 2026

Target: Ryzen 5 5625U, Radeon Vega PCI 1002:15e7, 16 GB shared RAM, AMD 26.5.2, Windows 11. Source baseline: RedTunic v4.4.0 (`bc199725b73f`). The objective is fewer camera-turn stalls without invalidating prepared shaders.

## Public sources examined

| Project | Public source and relevant difference | RedTunic decision |
|---|---|---|
| Longshot | [CPU decoder replacement, #2307](https://git.magicbean.dev/MagicBeanDev/Longshot/pulls/2307): ported ASTC LDR decoder; also newer target-memory pooling and integrated-memory policies | Trial the decoder independently. Existing target pooling and texture budgets must be distinguished from newer pooling by allocation size. No blanket merge of renderer or translator changes. |
| NX2navi | [Source](https://github.com/tangerineturtle1/NX2navi), v0.12.0: content comparisons before repeat uploads; longer bounded retention of released textures; automatic lower texture detail on integrated GPUs | Different C renderer. RedTunic already has exact-block ASTC reuse, decoded-level retention, texture detail and shared-memory budgets. These are design comparisons, not interchangeable patches. |
| Triforce | [Current source repository](https://git.triforceproject.com/triforce/source), separate from its download repository: texture warm-ahead, render-target allocation reuse by size, layered ASTC upload splitting and a different GPU ASTC shader | Warm-ahead and bounded uploads already exist in RedTunic, with Vega-specific CPU preparation. Allocation-size reuse is a separate host-side candidate. GPU ASTC changes do not apply while this PC requires CPU ASTC; do not enable them or change built shaders. |
| HeyListen | [Source](https://github.com/Sacrappy/HeyListen), [texture-resolution measurements](https://github.com/Sacrappy/HeyListen/blob/main/docs/research/texres1-texture-resolution.md), [disk-cache investigation](https://github.com/Sacrappy/HeyListen/blob/main/docs/research/bc7c-texture-cache.md) | Storage-fingerprint caching measured inside run-to-run FPS noise. Earlier synchronous disk-cache reads regressed performance; newer read-ahead needs separate integration. BC cache is not directly usable for Vega's RGBA8 CPU path. |
| ocarina-recomp | [Source](https://github.com/sostenesfreitas/ocarina-recomp): separate C renderer, ASTC decoder lineage used by Longshot, bounded GPU submission lead | Use the reviewed standalone decoder adaptation with provenance. Do not transplant its unrelated submission lifecycle into this renderer. |
| Hookshot | [Upstream forge](https://git.notrexed.dev/Hookshot/Hookshot) returned HTTP 521 during this audit | Existing local Hookshot-derived source is available; latest remote differences cannot be verified while the forge is unavailable. |
| Ocarina Reborn | [v1.1.7 source](https://github.com/misterrageguy-cyber/OoT2026-PC-Recomp): byte-range visibility waits, delayed return to recently released neighboring areas, virtual-core fixes | Byte-range waits already exist as `RX_VK_WB_SPANS`; test alone, since the earlier combined switch group regressed. RedTunic also already holds unstable neighboring-area choices (`HOS_OCB_HOLD_MS`, 2 s); Reborn's return-only rule differs. Scheduler changes require their own reproductions. |
| TOoT:R | [Source](https://github.com/Unkn0wnAli4n/TOoT-R): integrated-GPU decoded texture disk cache, lower texture budgets, bounded parallel texture work | RedTunic already has parallel decode and a local experimental disk-cache path in the installed development renderer. No duplicate disk cache; prioritize cheaper first-time decoding. |
| GreenKid-MV | [Source](https://github.com/adi0900/GreenKid-MV): nxnative ARM/Metal runtime; deferred resource release, memory-spike and pipeline-preparation fixes | Metal/Apple ARM implementations cannot be copied into Windows Vulkan. RedTunic already defers resource destruction and prepares pipelines; preserve these contracts. |

## Evidence and candidate

An earlier camera-route log (`before1`, 25 November) recorded approximately 41 ms/frame in texture uploads, 43 ms in texture lookups, 23 ms of host GPU waits and 71 ms of GPU time. These overlap or nest: **do not add them**. There were no graphics-program waits or shader-compilation stalls. That run used older output settings and is diagnostic context, not the baseline for a new performance claim.

The new CPU decoder is enabled by default with `RX_VK_ASTC_FAST_CPU=0` as the rollback. Demand uploads, background read-ahead and streamed mip levels all use the same selector. Existing memory bounds, upload budgets, resource barriers and shader translation are unchanged. The default follows the user's direction; the measurements below remain limited and do not establish an across-hardware gain.

## PC camera-route measurements

Four 240-frame camera turns were recorded in Hyrule Field at 1280x720 output and 67% render resolution, using the same copied save and pinned settings. The renderer DLL was rebuilt from this branch; the installed game executable reported v4.1.0 (`2cd2475d`), so this is a controlled renderer comparison on the PC, not a complete v4.4.0 application test. Input started at the same Hyrule Field coordinates and facing for baseline2 and fast2. The isolated test cache was shared sequentially and grew warmer between runs.

| Arm | Start frame | Mean frame time | Mean FPS | P99 | Worst | Frames >100 ms |
|---|---:|---:|---:|---:|---:|---:|
| Legacy 1 | 3450 | 131.31 ms | 7.62 | 655.1 ms | 751.7 ms | 103/240 |
| Port 1 | 3087 | 125.90 ms | 7.94 | 371.3 ms | 532.8 ms | 136/240 |
| Legacy 2 | 3155 | 136.22 ms | 7.34 | 552.9 ms | 739.7 ms | 115/240 |
| Port 2 | 4948 | 117.04 ms | 8.54 | 423.4 ms | 476.6 ms | 112/240 |

The second turn pair improved mean frame time by 14.1%, P99 by 23.4%, and the worst frame by 35.6%; the first pair improved mean by 4.1% and the worst frame by 29.1%, but had 33 more frames over 100 ms. All 960 measured frames exceeded 33.3 ms. These short runs support a modest decode improvement, not a substantial, repeatable camera-smoothness result. A current v4.4.0 executable and independently reset, matching cache states still need testing. No memory reduction was measured. The legacy logs attribute most cumulative slow-submit time to draw and descriptor work; shader-program waits and pipeline compilations were zero.

The port preserves the existing CPU integration's linear-profile interpretation, including when the image is subsequently viewed as sRGB. It does **not** add format-correct sRGB CPU decoding. Its 8-bit truncation differs from suyu's rounding; results are not byte-identical. Illegal blocks remain transparent black; refused arguments produce zero output rather than stale staging bytes. The library's independent reference test covers both profiles, while the renderer adapter test explicitly checks the legacy linear-profile choice.

The port and its synthetic vectors come from Longshot's `gpu/vk/texdecode` adaptation of LAS Recomp via ocarina-recomp. Notices and the detailed provenance are retained beside the source. No game textures or translated game shaders are included in these tests.

## Vulkan and driver research

- The local AMD 26.5.2 Vulkan driver advertises `VK_EXT_descriptor_buffer` and `descriptorBuffer=true`. Khronos' [Vulkan Samples descriptor-buffer example](https://github.com/KhronosGroup/Vulkan-Samples/blob/main/samples/extensions/descriptor_buffer_basic/README.adoc) shows it replacing descriptor pool/set allocation, updates and binds with application-managed descriptor memory. RedTunic currently requests descriptor buffers only for its Linux FSR4 integration; ordinary replay still builds descriptors per draw. The captured frames spent about 29-35 ms in RedTunic's descriptor phase, making a general descriptor-buffer prototype the strongest next renderer experiment. It is a substantial binding/layout change and must be measured against today's push-descriptor and indexed-array paths before adoption.
- The local driver does not list `VK_EXT_host_image_copy`. Khronos' [extension proposal](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_host_image_copy.adoc) says host copies can avoid staging memory, but may be slower because the CPU must swizzle optimal-layout images; it specifically recommends measuring per implementation. This PC cannot exercise the extension today.
- RedTunic already batches queue submissions for Mesa RADV. Its [switch policy](../gpu/vk/switches.def) keeps batching off on other drivers unless overridden. The local AMD proprietary driver is not RADV, and the run's queue-submit portion was about 1.2 ms within a 65 ms replay frame, so forcing that path is not a strong candidate here.
- Khronos' [Vulkan profiling guide](https://github.com/KhronosGroup/Vulkan-Guide/blob/main/chapters/profiling.adoc) separates command recording, descriptor updates, submission and GPU execution. RedTunic's last complete 600-present pass-timing window measured 43.8 ms/frame of command-buffer GPU work; mesh fallback spans were 7.74 ms (17.7%), general dispatch 9.60 ms (21.9%), and render passes 22.90 ms (52.3%). This makes mesh fallback meaningful, but not the only or largest GPU target. The Windows log also shows large CPU draw/descriptor time. AMD's [Radeon GPU Profiler](https://gpuopen.com/manuals/rgp_manual/) targets RDNA hardware; this Vega iGPU is GCN, so support is not established. RedTunic's built-in `RX_VK_PASS_TIMING=1` and full `HOS_PROFILE=1` report are the available next measurements here; no blind shader or barrier change is justified by the totals.
- The local driver lists `VK_EXT_descriptor_heap` and `VK_EXT_host_image_copy` as unavailable; Vulkan-Samples' [descriptor-heap example](https://github.com/KhronosGroup/Vulkan-Samples/blob/main/samples/extensions/descriptor_heap/README.adoc) therefore does not provide a runnable path on this PC.

## Validation record

- Reference vectors: 4,694 synthetic blocks in linear and sRGB modes, 9,388 comparisons, zero mismatches; 28 whole-level tests.
- Independent cold review: 250,000 random block decodes preserved guard bytes; no bounds or shader-cache blocker found.
- Renderer adapter: switch off/on, original serial/parallel routes, transparent illegal blocks, short-source refusal and explicit linear-profile choice.
- Synthetic decoder timings are exploratory only. In-game measurements are recorded above. Repeated, controlled in-game runs on a current executable and complete-suite results are still needed to validate the new default across hardware tiers.

Prepared program records, shader translation sources and the renderer API are unchanged. The writable benchmark cache and copied saves stayed local; the player's original shader cache and saves were not used as writable benchmark files.
