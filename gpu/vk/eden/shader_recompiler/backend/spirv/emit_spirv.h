// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/mma_layout.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::SPIRV {

constexpr u32 NUM_TEXTURE_SCALING_WORDS = 4;
constexpr u32 NUM_IMAGE_SCALING_WORDS = 2;
constexpr u32 NUM_TEXTURE_AND_IMAGE_SCALING_WORDS =
    NUM_TEXTURE_SCALING_WORDS + NUM_IMAGE_SCALING_WORDS;

struct RescalingLayout {
    alignas(16) std::array<u32, NUM_TEXTURE_SCALING_WORDS> rescaling_textures;
    alignas(16) std::array<u32, NUM_IMAGE_SCALING_WORDS> rescaling_images;
    u32 down_factor;
};
struct RenderAreaLayout {
    std::array<f32, 4> render_area;
};
constexpr u32 RESCALING_LAYOUT_WORDS_OFFSET = offsetof(RescalingLayout, rescaling_textures);
constexpr u32 RESCALING_LAYOUT_DOWN_FACTOR_OFFSET = offsetof(RescalingLayout, down_factor);
constexpr u32 RENDERAREA_LAYOUT_OFFSET = offsetof(RenderAreaLayout, render_area);
// (hookshot) guest memory's page table, a u64 device address pushed to every compute program that reads global memory, after the blocks
// above. It was a constant in the SPIR-V, and the address changes from run to run: the drivers' pipeline caches never hit
constexpr u32 GUEST_PAGES_PUSH_OFFSET = sizeof(RescalingLayout) + sizeof(RenderAreaLayout);
// (hookshot) a task program's first workgroup: NVN draws tasks [first, first + count), Vulkan from 0. The address of the draw's first
// (a u32, 16 bytes per draw of an indirect draw, read at + DrawIndex * 16) is pushed here
constexpr u32 MESH_FIRST_PUSH_OFFSET = GUEST_PAGES_PUSH_OFFSET + 8;
// (hookshot) the mesh shader fallback's buffers, after it: the task payloads and the vertex records (u64 device addresses).
// Paths without the fallback never push them.
constexpr u32 MESH_FALLBACK_PUSH_OFFSET = MESH_FIRST_PUSH_OFFSET + 8;
constexpr u32 MESH_FALLBACK_PAYLOAD_PUSH_OFFSET = MESH_FALLBACK_PUSH_OFFSET;
constexpr u32 MESH_FALLBACK_RECORDS_PUSH_OFFSET = MESH_FALLBACK_PUSH_OFFSET + 8;
constexpr u32 MESH_FALLBACK_INDIRECT_INPUT_PUSH_OFFSET = MESH_FALLBACK_PUSH_OFFSET + 16;
constexpr u32 MESH_FALLBACK_PUSH_END = MESH_FALLBACK_PUSH_OFFSET + 24;
// (hookshot) the fallback's mesh workgroups a draw can run: each a slot, a word of the payload buffer saying whether its task
// started it; then each task workgroup's payload. It was 1024, a record block each, and Ocarina's floor grass (433 tasks of
// 32 slots) lost 13 of 14 meshlets. At 16384, Hyrule Field's far trees (3cb76827/4f0c0b40: 721 tasks of 32) lost the tasks
// past 512, a different set each frame as the culling's list order changes: they flickered. 65535: Vulkan's least
// maxComputeWorkGroupCount, which the mesh dispatch can reach; the payload buffer is ~6.5 MB a recording thread at 800 words.
constexpr u32 MESH_FALLBACK_CAPACITY = 65535;
// (hookshot) the records buffer: a VkDrawIndirectCommand the vertex stage's draw reads (its vertex count the records written),
// the allocator's word, then the records. A mesh workgroup takes records for its primitives' corners only, so a draw covers
// what was drawn, not every slot's maximum
constexpr u32 MESH_FALLBACK_DRAW_WORD = 0;        // (vertexCount; instanceCount at word 1, set to 1 by the host)
constexpr u32 MESH_FALLBACK_ALLOC_WORD = 4;       // (records taken, past the end too: those that do not fit are not drawn)
// (a VkDispatchIndirectCommand for the mesh workgroups a task program started (y and z 1, set by the host): each started task
// thread takes an index of it and writes its slot there in the payload buffer's map, so the mesh dispatch runs only those. With
// a count from the GPU (DrawMeshTasksIndirect, the floor grass) the host knew none, and dispatched no mesh workgroup)
constexpr u32 MESH_FALLBACK_DISPATCH_WORD = 5;
constexpr u32 MESH_FALLBACK_HEADER_WORDS = 8;
// (128 MB. Every program takes records from it, so a program with longer records fits fewer: MeshFallbackFit. At 16M words
// the floor grass at the Grass distance's 4x lost its blades: their program's records are 22 words and fit 762600,
// its partner's 7 words, and each drew ~1.02M records a frame, so the blades past the fit were left out and drawn white.
// 32M words fit 1.52M of 22 words)
constexpr u32 MESH_FALLBACK_RECORD_WORDS = 32u << 20;
// (emit_spirv_mesh.cpp BufferWordAt makes a word's byte offset in 32 bits)
static_assert((u64{MESH_FALLBACK_HEADER_WORDS} + MESH_FALLBACK_RECORD_WORDS) * 4 < (u64{1} << 32));
// (hookshot, mma_layout.h mesh_fallback_global_attributes) the mesh workgroups' attribute regions out of workgroup memory: a pool
// after the records, in the same buffer. Its header is a ticket counter and a word a slot, the ticket generation that may take
// that slot next (both zeroed before each draw, with the records' header); then the slots, each one workgroup's region
constexpr u64 MESH_FALLBACK_POOL_OFFSET = u64{MESH_FALLBACK_RECORD_WORDS} * 4 + 4096;   // (past the header and the records)
constexpr u32 MESH_FALLBACK_POOL_SLOTS = 1024;            // (a power of two: a ticket's slot is its low bits)
constexpr u32 MESH_FALLBACK_POOL_SLOT_SHIFT = 10;
constexpr u32 MESH_FALLBACK_POOL_HEADER_BYTES = 8192;     // (the counter and a word a slot: 4100 bytes)
constexpr u32 MESH_FALLBACK_POOL_SLOT_BYTES = 32768;      // (the largest region a slot holds; a larger one stays in workgroup memory)
constexpr u64 MESH_FALLBACK_POOL_BYTES = MESH_FALLBACK_POOL_HEADER_BYTES + u64{MESH_FALLBACK_POOL_SLOTS} * MESH_FALLBACK_POOL_SLOT_BYTES;
constexpr u32 MESH_FALLBACK_POOL_WAIT = 1u << 20;         // (a wait for a slot's last holder gives up after this many reads)
static_assert((MESH_FALLBACK_POOL_SLOTS >> MESH_FALLBACK_POOL_SLOT_SHIFT) == 1 && MESH_FALLBACK_POOL_HEADER_BYTES >= 4 + MESH_FALLBACK_POOL_SLOTS * 4);
static_assert(MESH_FALLBACK_POOL_OFFSET >= (u64{MESH_FALLBACK_HEADER_WORDS} + MESH_FALLBACK_RECORD_WORDS) * 4);
// (hookshot, mma_layout.h mesh_fallback_shared_warp) the shared warp's words in workgroup memory: an if with a warp operation
// inside has the lanes that entered it and those that reached it; a ballot or a vote its lanes and those that arrived; a shuffle
// a value a lane and those that arrived, or one value where the source lane is a constant
constexpr u32 WARP_REGION_WORDS = 2;
constexpr u32 WARP_BALLOT_WORDS = 2;
constexpr u32 WARP_SHUFFLE_WORDS = 33;
constexpr u32 WARP_BROADCAST_WORDS = 2;
constexpr u32 WARP_WAIT = 1u << 16;   // (a wait for other lanes gives up after this many reads: a wrong word, not a hung GPU)
// (its counters, four words right past the records buffer's regular end, which vk_draw.cpp fallback_buffers_of adds and
// clears when the shared warp or the attribute pool is on: the waits that gave up, with RX_VK_MESH_SHARED_WARP_CHECK the
// answers compared with the subgroup's and those that differed, and the attribute pool's waits for a slot that gave up
// (mma_layout.h mesh_fallback_global_attributes). They only grow and wrap; fallback_warp_counters logs them)
constexpr u32 WARP_COUNTER_BYTES = 16;
constexpr u32 WARP_COUNTER_WAITS = 0;
constexpr u32 WARP_COUNTER_COMPARED = 4;
constexpr u32 WARP_COUNTER_DIFFERENT = 8;
constexpr u32 WARP_COUNTER_POOL_WAITS = 12;
// (the largest records buffer vk_draw.cpp fallback_buffers_of makes, a recording thread's: past it, every program is left
// without the fallback. The largest layout, the records with the attribute pool and the counters, ~160 MB, fits)
constexpr u64 MESH_FALLBACK_RECORDS_LIMIT = 192ull << 20;
static_assert(MESH_FALLBACK_POOL_OFFSET + MESH_FALLBACK_POOL_BYTES + WARP_COUNTER_BYTES <= MESH_FALLBACK_RECORDS_LIMIT);
/// Whether the records buffer carries those counters past its regular end: the shared warp's and the pool's waits count there
[[nodiscard]] inline bool MeshFallbackCounters() {
    return mesh_fallback_shared_warp || mesh_fallback_global_attributes;
}
/// The fallback's records buffer without those counters: the header and the records, and the attribute regions' pool where
/// mma_layout.h mesh_fallback_global_attributes is on
[[nodiscard]] inline u64 MeshFallbackRecordsBytes() {
    return mesh_fallback_global_attributes ? MESH_FALLBACK_POOL_OFFSET + MESH_FALLBACK_POOL_BYTES
                                           : (u64{MESH_FALLBACK_HEADER_WORDS} + MESH_FALLBACK_RECORD_WORDS) * 4;
}

// (hookshot) how the records lie in their area (RuntimeInfo::mesh_fallback_layout, RX_VK_MESH_FB_LAYOUT):
// - Aos: record after record, a record's words together (word H + record * W + line). Neighbouring threads write words a
//   record apart, so each store of a mesh workgroup touches as many cache lines as it has threads.
// - Soa: line after line, the area cut into W lines of `fit` words (word H + line * fit + record), and the threads take the
//   records in turn, so neighbouring threads write neighbouring words. The same records, the same count, the same draw.
enum class MeshFallbackLayout : u32 { Aos = 0, Soa = 1 };
/// A line's words: the records of `record_words` words the area holds
[[nodiscard]] constexpr u32 MeshFallbackFit(u32 record_words) {
    return MESH_FALLBACK_RECORD_WORDS / (record_words ? record_words : 1);
}

[[nodiscard]] std::vector<u32> EmitSPIRV(const Profile& profile, const RuntimeInfo& runtime_info, IR::Program& program, Bindings& bindings);
[[nodiscard]] inline std::vector<u32> EmitSPIRV(const Profile& profile, IR::Program& program) {
    Bindings binding;
    return EmitSPIRV(profile, {}, program, binding);
}

} // namespace Shader::Backend::SPIRV
