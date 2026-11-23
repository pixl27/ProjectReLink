// (hookshot) The device's element order of the 16x8x16 f16 cooperative matrices HMMA_nx2 runs on.
//
// A lane's HMMA fragment registers hold its matrix elements in the console's order (PTX mma.m16n8k16's: two f16
// a register, low half first). The host's cooperative matrices hold each lane's share in an order of the
// implementation's own; vk_coopmat.cpp measures it at start, and only when every lane holds the same positions in
// both does the back end turn HMMA_nx2 on. Element e of the console's fragment is then the device's element
// mma_order_x[e].
#pragma once

#include <array>
#include <cstdint>

namespace Shader::Backend::SPIRV {

inline std::array<uint32_t, 8> mma_order_a{0, 1, 2, 3, 4, 5, 6, 7};
inline std::array<uint32_t, 4> mma_order_b{0, 1, 2, 3};
inline std::array<uint32_t, 4> mma_order_c{0, 1, 2, 3};
inline bool mma_software = false;   // (HMMA_nx2 in software: emit_spirv_warp.cpp, RX_VK_MMA_SOFTWARE=1)
// (hookshot) and in which form (mma_software.h SoftwareMmaForm: 0 the loop, 1 unrolled, 2 unrolled with the GPU's f16 rounding),
// chosen by vk_coopmat.cpp's self-test; and whether its quad broadcasts may be used (the device has quad operations in every stage)
inline uint32_t mma_software_form = 0;
inline bool mma_software_quad = false;
// (amd-fix) float colour outputs carry no NaN or infinity to their target (EmitSetFragColor): one NaN texel spreads
// through bloom/blur/downsample averages into glowing blocks (OoT's flowers on AMD, Hookshot issue 606). On for every
// card since v2.0.6 (vk_coopmat.cpp coopmat_init); RX_VK_NAN_GUARD=0 turns it off.
inline bool frag_nan_guard = false;
// (amd-fix) gl_Position decorated Invariant (vertex and mesh outputs): the game draws a depth pre-pass, then the colour
// pass again with depth EQUAL; Maxwell computes the same position in both, AMD's compiler may not without Invariant, and
// the colour pass is then rejected in spots: holes showing the bright sky, blown up by bloom (the glowing flowers).
// On for every card but NVIDIA (vk_coopmat.cpp); RX_VK_NO_INVARIANT=1 off.
inline bool position_invariant = false;
// (hookshot) An out-of-bounds texel fetch answers zero, as NVIDIA's GPUs answer it (measured on an RTX 3090), rather than
// whatever the host's does (EmitImageFetch). Vulkan leaves it undefined without robustImageAccess; on AMD the ground fog's
// off-screen depth reads drew a pale box at the screen's edges. Every card but NVIDIA (gpu/vk/device_policy.h); RX_VK_FETCH_BOUNDS
inline bool fetch_bounds = false;
// (hookshot) and texel buffers' fetches too: only when robustBufferAccess2 is off (the Debug tab's Robust buffers Off, RX_VK_ROBUST=0,
// or a driver without it), which otherwise answers their out-of-bounds reads with zero already (vk_coopmat.cpp)
inline bool fetch_bounds_buffers = false;
// (hookshot) isnan tested on the value's bits (IsNan in emit_spirv_floating_point.cpp), not OpIsNan, for AMD's older Windows
// shader compiler (RX 5000 and RX 6000: gpu/vk/device_policy.h); RX_VK_NAN_BITS=0 or 1
inline bool nan_bits = false;
// (hookshot) In compute-fallback mesh stages, emit NoContraction f32 multiplies as OpFma(a, b, -0.0f) (EmitFPMul32):
// AMD's older Windows proprietary shader compiler (RDNA 1 and 2: gpu/vk/device_policy.h) ignores NoContraction and
// reassociates fma(t, v * rsq, add) into fma(rsq * t, v, add) in depth-only mesh fallback shaders where gl_Position
// cannot be decorated Invariant, causing grass tips to fail the colour pass's EQUAL depth test and flash white.
// OpFma(a, b, -0.0f) is IEEE-754 bit-exact with a * b and blocks the reassociation; RX_VK_MESH_STRICT_FMUL=0 or 1
inline bool mesh_fallback_strict_fmul = false;
// (hookshot) a storage buffer table's entries searched in a loop the driver is asked not to unroll (DefineGlobalMemoryFunctions),
// not one test an entry written out sixteen times. Mesa inlines every function, DontInline or not: OoT's foliage culling
// (53caac5f) has 46 global accesses, each with its ~70 table entries written out, over 100k instructions that Intel's
// compiler did not finish in minutes, and every compute pipeline queued behind it on Intel's one build thread waited:
// mesh-drawn rocks and doors white, the forest grey. In a loop: 12 s. On for Intel's Mesa driver (vk_coopmat.cpp); RX_VK_GM_LOOP=0 or 1
inline bool global_table_loop = false;
// (hookshot) The mesh fallback's task and mesh stages read global memory through the guest page table alone, the way the console's
// GPU reads it, not by searching the program's storage buffers (DefineGlobalMemoryFunctions). The search is the cost there: on an
// Iris Xe the fallback's colour mesh stages are held to 32 lanes at the register file's limit, and each of their global loads,
// inlined with its four buffers and three table entries, filled the program with spill code: 739acd67 (Hyrule Castle's) took
// 12.3 ms a frame, 5.7 ms with its loads through the table, its 243 million loads the same words both ways. A storage buffer in a
// virtual pool (from 1 TB) is not in the table and reads 0. On for Intel's Mesa driver (vk_coopmat.cpp); RX_VK_MESH_PAGES=0 or 1
inline bool mesh_fallback_pages = false;
// (hookshot) The mesh fallback's mesh workgroups keep their attribute region (the stage buffer's vertex and primitive lines, up to
// 32 KB) in a slot of a pool in the records buffer instead of in workgroup memory (emit_spirv.h MESH_FALLBACK_POOL_*). Intel
// gives workgroup memory in powers of two, so Hyrule Castle's colour mesh stages, 9 to 18 KB of it, took 16 or 32 KB each: few
// workgroups fit on a subslice at once, and nothing hid the stages' spill traffic. On an Iris Xe 739acd67 took 5.3-6.9 ms a
// dispatch with its region in workgroup memory, 7.4 ms given 8 KB more of it, and 2.6-2.8 ms with the region in a slot.
// On for Intel's Mesa driver (vk_coopmat.cpp); RX_VK_MESH_GLOBAL_ATTRS=0 or 1
inline bool mesh_fallback_global_attributes = false;
// (hookshot) The mesh fallback's task and mesh stages keep their warp in workgroup memory, not in the host's subgroup: a lane is the
// invocation's place in its 32-thread workgroup, and ballots, votes and shuffles go through words of workgroup memory that every
// lane of the block's region writes and waits for (emit_spirv_warp.cpp, the shared warp). Such a stage has no subgroup
// instruction left, so it gives the same result at any lane count, and the renderer runs it at 8 lanes (vk_draw.cpp): on an
// Iris Xe the colour mesh stages, held to 32 lanes for their subgroup instructions, spent most of their time spilling
// (d8e59341: 843k cycles a workgroup at 32 lanes by Mesa's count, 4 x 70k at 8). Only stages it can be done for: 32-thread
// workgroups, no warp operation in a loop, no read of the stage buffer's outputs (EmitContext::PlanSharedWarp). On for Intel's
// Mesa driver (vk_coopmat.cpp); RX_VK_MESH_SHARED_WARP=0 or 1
inline bool mesh_fallback_shared_warp = false;
// (RX_VK_MESH_SHARED_WARP_CHECK=1, for a developer: each shared ballot, vote and shuffle beside the subgroup's own, those
// compared and those that differ counted past the fallback's records (emit_spirv.h WARP_COUNTER_*) and logged; the stage
// keeps its subgroup instructions, so it stays at 32 lanes)
inline bool mesh_fallback_shared_warp_check = false;

} // namespace Shader::Backend::SPIRV
