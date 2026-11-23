// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/backend/spirv/mma_layout.h"
#include "shader_recompiler/backend/spirv/mma_software.h"

namespace Shader::Backend::SPIRV {
namespace {
Id SubgroupScope(EmitContext& ctx) {
    return ctx.Const(static_cast<u32>(spv::Scope::Subgroup));
}

bool StageSupportsSubgroups(EmitContext& ctx) {
    return ctx.profile.SupportsSubgroupStage(ctx.stage);
}

Id GetThreadId(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.u32_zero_value;
    }
    return ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id);
}

Id WarpExtract(EmitContext& ctx, Id value) {
    const Id thread_id{GetThreadId(ctx)};
    const Id local_index{ctx.OpShiftRightArithmetic(ctx.U32[1], thread_id, ctx.Const(5U))};
    if (ctx.profile.has_broken_spirv_subgroup_mask_vector_extract_dynamic) {
        const Id c0_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(0U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 0U), ctx.Const(0U))};
        const Id c1_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(1U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 1U), ctx.Const(0U))};
        const Id c2_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(2U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 2U), ctx.Const(0U))};
        const Id c3_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(3U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 3U), ctx.Const(0U))};
        const Id c0_or_c1{ctx.OpBitwiseOr(ctx.U32[1], c0_sel, c1_sel)};
        const Id c2_or_c3{ctx.OpBitwiseOr(ctx.U32[1], c2_sel, c3_sel)};
        const Id c0_or_c1_or_c2_or_c3{ctx.OpBitwiseOr(ctx.U32[1], c0_or_c1, c2_or_c3)};
        return c0_or_c1_or_c2_or_c3;
    } else {
        return ctx.OpVectorExtractDynamic(ctx.U32[1], value, local_index);
    }
}

Id LoadMask(EmitContext& ctx, Id mask) {
    const Id value{ctx.OpLoad(ctx.U32[4], mask)};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpCompositeExtract(ctx.U32[1], value, 0U);
    }
    return WarpExtract(ctx, value);
}

void SetInBoundsFlag(IR::Inst* inst, Id result) {
    IR::Inst* const in_bounds{inst->GetAssociatedPseudoOperation(IR::Opcode::GetInBoundsFromOp)};
    if (!in_bounds) {
        return;
    }
    in_bounds->SetDefinition(result);
    in_bounds->Invalidate();
}

Id ComputeMinThreadId(EmitContext& ctx, Id thread_id, Id segmentation_mask) {
    return ctx.OpBitwiseAnd(ctx.U32[1], thread_id, segmentation_mask);
}

Id ComputeMaxThreadId(EmitContext& ctx, Id min_thread_id, Id clamp, Id not_seg_mask) {
    return ctx.OpBitwiseOr(ctx.U32[1], min_thread_id,
                           ctx.OpBitwiseAnd(ctx.U32[1], clamp, not_seg_mask));
}

Id GetMaxThreadId(EmitContext& ctx, Id thread_id, Id clamp, Id segmentation_mask) {
    const Id not_seg_mask{ctx.OpNot(ctx.U32[1], segmentation_mask)};
    const Id min_thread_id{ComputeMinThreadId(ctx, thread_id, segmentation_mask)};
    return ComputeMaxThreadId(ctx, min_thread_id, clamp, not_seg_mask);
}

Id HostThreadId(EmitContext& ctx, Id thread_id) {
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return thread_id;
    }
    const Id partition_idx{ctx.OpShiftRightLogical(ctx.U32[1], GetThreadId(ctx), ctx.Const(5u))};
    const Id partition_base{ctx.OpShiftLeftLogical(ctx.U32[1], partition_idx, ctx.Const(5u))};
    return ctx.OpIAdd(ctx.U32[1], thread_id, partition_base);
}

Id GuestLane(EmitContext& ctx, Id index) {
    return ctx.OpBitwiseAnd(ctx.U32[1], index, ctx.Const(31U));
}

bool UseShuffleFallback(EmitContext& ctx) {
    return ctx.profile.has_broken_spirv_subgroup_shuffle ||
           (ctx.profile.has_broken_spirv_fragment_shuffle && ctx.stage == Stage::Fragment);
}

Id ShuffleAbsolute(EmitContext& ctx, Id value, Id src_thread_id, bool broadcast) {
    if (!broadcast) {
        return ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, src_thread_id);
    }
    Id result{ctx.u32_zero_value};
    for (u32 lane = 0; lane < ctx.profile.max_subgroup_size; ++lane) {
        const Id read{
            ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, ctx.Const(lane))};
        const Id matches{ctx.OpIEqual(ctx.U1, src_thread_id, ctx.Const(lane))};
        result = ctx.OpSelect(ctx.U32[1], matches, read, result);
    }
    return result;
}

Id ShuffleRelative(EmitContext& ctx, Id value, Id delta, Id src_thread_id, spv::Op op) {
    const bool broadcast{UseShuffleFallback(ctx)};
    if (!ctx.profile.support_shuffle_relative || broadcast) {
        return ShuffleAbsolute(ctx, value, HostThreadId(ctx, src_thread_id), broadcast);
    }
    const Id result{ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, delta)};
    ctx.opcode_rewrites.emplace_back(result.value, op);
    return result;
}

Id BroadcastLane(EmitContext& ctx, Id value, u32 lane) {
    Id result{
        ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, ctx.Const(lane))};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return result;
    }
    const Id partition_idx{ctx.OpShiftRightLogical(ctx.U32[1], GetThreadId(ctx), ctx.Const(5u))};
    for (u32 base = 32; base < ctx.profile.max_subgroup_size; base += 32) {
        const Id read{ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value,
                                                     ctx.Const(base + lane))};
        const Id matches{ctx.OpIEqual(ctx.U1, partition_idx, ctx.Const(base >> 5))};
        result = ctx.OpSelect(ctx.U32[1], matches, read, result);
    }
    return result;
}

// (hookshot, mma_layout.h mesh_fallback_shared_warp) The shared warp: a fallback stage's 32 lanes are its workgroup's invocations,
// on however many hardware threads the driver runs them, and what a warp operation needs of the other lanes goes through words of
// workgroup memory. The lanes that run a block are its region's: the whole warp at the top, and inside an if the lanes that took
// it, which every lane of the region around it says (WarpOpenRegion) before any goes in. A site (a ballot, a vote, a shuffle) is
// reached by all of its region's lanes, since the planner (EmitContext::DefineSharedWarp) takes no stage that leaves an if early
// or has a site in a loop: each lane writes its part and its arrival, then waits for the arrivals it needs. A lane only waits
// for lanes behind it in the program, which wait for none ahead of them, so the waits end.
Id WarpWord(EmitContext& ctx, Id index) {
    return ctx.OpAccessChain(ctx.warp_word_pointer, ctx.warp_words, index);
}

Id WarpWord(EmitContext& ctx, u32 index) {
    return WarpWord(ctx, ctx.Const(index));
}

u32 WarpTake(EmitContext& ctx, u32 words) {
    if (ctx.warp_word_next + words > ctx.warp_word_count) {
        throw LogicError("Shared warp: {} words past the {} planned", ctx.warp_word_next + words, ctx.warp_word_count);
    }
    const u32 first{ctx.warp_word_next};
    ctx.warp_word_next += words;
    return first;
}

Id WarpScope(EmitContext& ctx) {
    return ctx.Const(static_cast<u32>(spv::Scope::Workgroup));
}

void WarpOr(EmitContext& ctx, u32 word, Id bits) {
    ctx.OpAtomicOr(ctx.U32[1], WarpWord(ctx, word), WarpScope(ctx), ctx.u32_zero_value, bits);
}

Id WarpRead(EmitContext& ctx, Id pointer) {
    return ctx.OpAtomicLoad(ctx.U32[1], pointer, WarpScope(ctx), ctx.u32_zero_value);
}

void WarpWait(EmitContext& ctx, u32 word, Id lanes) {
    ctx.OpFunctionCall(ctx.void_id, ctx.warp_wait, ctx.Const(word), lanes);
}

/// (RX_VK_MESH_SHARED_WARP_CHECK) one more answer compared, and one more difference where there is one: the counters past
/// the fallback's records (emit_spirv.h WARP_COUNTER_*)
void WarpCheck(EmitContext& ctx, Id differs) {
    const Id device{ctx.Const(static_cast<u32>(spv::Scope::Device))};
    ctx.OpAtomicIAdd(ctx.U32[1], ctx.MeshFallbackWarpCounter(WARP_COUNTER_COMPARED), device, ctx.u32_zero_value, ctx.Const(1U));
    ctx.OpAtomicIAdd(ctx.U32[1], ctx.MeshFallbackWarpCounter(WARP_COUNTER_DIFFERENT), device, ctx.u32_zero_value,
                     ctx.OpSelect(ctx.U32[1], differs, ctx.Const(1U), ctx.u32_zero_value));
}

/// The lanes of the block's region for which `pred` holds
Id SharedBallot(EmitContext& ctx, Id pred) {
    const u32 word{WarpTake(ctx, WARP_BALLOT_WORDS)};
    if (pred.value == ctx.true_value.value) {
        return ctx.warp_region;
    }
    WarpOr(ctx, word, ctx.OpSelect(ctx.U32[1], pred, ctx.warp_bit, ctx.u32_zero_value));
    WarpOr(ctx, word + 1, ctx.warp_bit);
    WarpWait(ctx, word + 1, ctx.warp_region);
    return WarpRead(ctx, WarpWord(ctx, word));
}

/// `value` of lane `source` (0 where that lane is not in the block's region: the subgroup's shuffle leaves it undefined)
Id SharedShuffle(EmitContext& ctx, Id value, Id source) {
    const u32 word{WarpTake(ctx, WARP_SHUFFLE_WORDS)};
    const Id lane{ctx.OpBitwiseAnd(ctx.U32[1], source, ctx.Const(31U))};
    ctx.OpAtomicExchange(ctx.U32[1], WarpWord(ctx, ctx.OpIAdd(ctx.U32[1], ctx.warp_lane, ctx.Const(word))), WarpScope(ctx),
                         ctx.u32_zero_value, value);
    WarpOr(ctx, word + 32, ctx.warp_bit);
    WarpWait(ctx, word + 32,
             ctx.OpBitwiseAnd(ctx.U32[1], ctx.OpShiftLeftLogical(ctx.U32[1], ctx.Const(1U), lane), ctx.warp_region));
    return WarpRead(ctx, WarpWord(ctx, ctx.OpIAdd(ctx.U32[1], lane, ctx.Const(word))));
}

/// `value` of the constant lane `lane`: one word, which only that lane writes
Id SharedBroadcast(EmitContext& ctx, Id value, u32 lane) {
    const u32 word{WarpTake(ctx, WARP_BROADCAST_WORDS)};
    WarpOr(ctx, word,
           ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, ctx.warp_lane, ctx.Const(lane)), value, ctx.u32_zero_value));
    WarpOr(ctx, word + 1, ctx.warp_bit);
    WarpWait(ctx, word + 1, ctx.OpBitwiseAnd(ctx.U32[1], ctx.Const(1U << lane), ctx.warp_region));
    return WarpRead(ctx, WarpWord(ctx, word));
}

/// A lane mask of the shared warp, from the lane's bit: lower lanes, lower or equal, higher, higher or equal
Id SharedLtMask(EmitContext& ctx) {
    return ctx.OpISub(ctx.U32[1], ctx.warp_bit, ctx.Const(1U));
}

Id SharedLeMask(EmitContext& ctx) {
    return ctx.OpBitwiseOr(ctx.U32[1], SharedLtMask(ctx), ctx.warp_bit);
}

/// (the check: where a shuffle's two answers must agree: its source in range and a lane of the region; the subgroup's answer
/// for any other lane is undefined)
Id ShuffleCounts(EmitContext& ctx, Id in_range, Id source) {
    if (!ctx.warp_shared_check) {
        return Id{};
    }
    const Id bit{ctx.OpShiftLeftLogical(ctx.U32[1], ctx.Const(1U), ctx.OpBitwiseAnd(ctx.U32[1], source, ctx.Const(31U)))};
    return ctx.OpLogicalAnd(ctx.U1, in_range,
                            ctx.OpINotEqual(ctx.U1, ctx.OpBitwiseAnd(ctx.U32[1], bit, ctx.warp_region), ctx.u32_zero_value));
}

/// What the stage uses: the shared answer, or with the check the subgroup's, the two compared (`counts`: where they must agree)
template <typename Native>
Id SharedOr(EmitContext& ctx, Id shared, Native&& native, Id counts = Id{}, bool boolean = false) {
    if (!ctx.warp_shared_check) {
        return shared;
    }
    const Id subgroup{native()};
    Id differs{boolean ? ctx.OpLogicalNotEqual(ctx.U1, shared, subgroup) : ctx.OpINotEqual(ctx.U1, shared, subgroup)};
    if (Sirit::ValidId(counts)) {
        differs = ctx.OpLogicalAnd(ctx.U1, differs, counts);
    }
    WarpCheck(ctx, differs);
    return subgroup;
}
} // Anonymous namespace

void WarpOpenRegion(EmitContext& ctx, const IR::Block* body, Id cond) {
    if (!ctx.warp_shared) {
        return;
    }
    const bool region{ctx.warp_region_ifs.contains(body)};
    ctx.warp_region_opened.push_back(region);
    if (!region) {
        return;
    }
    // (every lane of the region around says whether it goes in, then that it has said so; the lanes that go in are known
    // once all of them have)
    const u32 word{WarpTake(ctx, WARP_REGION_WORDS)};
    WarpOr(ctx, word, ctx.OpSelect(ctx.U32[1], cond, ctx.warp_bit, ctx.u32_zero_value));
    WarpOr(ctx, word + 1, ctx.warp_bit);
    WarpWait(ctx, word + 1, ctx.warp_region);
    ctx.warp_regions.push_back(ctx.warp_region);
    ctx.warp_region = WarpRead(ctx, WarpWord(ctx, word));
}

void WarpCloseRegion(EmitContext& ctx) {
    if (!ctx.warp_shared || ctx.warp_region_opened.empty()) {
        return;
    }
    const bool region{ctx.warp_region_opened.back()};
    ctx.warp_region_opened.pop_back();
    if (region) {
        ctx.warp_region = ctx.warp_regions.back();
        ctx.warp_regions.pop_back();
    }
}

/// The shared warp's start, in the stage's entry block: the invocation's lane and bit, the whole warp as the region, and
/// the words at zero (the caller's barrier follows: every lane has zeroed its words before any writes one)
void WarpPrologue(EmitContext& ctx) {
    if (!ctx.warp_shared) {
        return;
    }
    ctx.warp_lane = ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.local_invocation_id), 0U);
    ctx.warp_bit = ctx.OpShiftLeftLogical(ctx.U32[1], ctx.Const(1U), ctx.warp_lane);
    ctx.warp_region = ctx.Const(0xffffffffU);
    for (u32 first = 0; first < ctx.warp_word_count; first += 32) {
        const Id at{ctx.OpUMin(ctx.U32[1], ctx.OpIAdd(ctx.U32[1], ctx.warp_lane, ctx.Const(first)),
                               ctx.Const(ctx.warp_word_count - 1))};
        ctx.OpStore(WarpWord(ctx, at), ctx.u32_zero_value);
    }
}

Id EmitLaneId(EmitContext& ctx) {
    const auto subgroup = [&] {
        const Id id{GetThreadId(ctx)};
        if (!ctx.profile.warp_size_potentially_larger_than_guest) {
            return id;
        }
        return ctx.OpBitwiseAnd(ctx.U32[1], id, ctx.Const(31U));
    };
    if (ctx.warp_shared) {
        return SharedOr(ctx, ctx.warp_lane, subgroup);
    }
    return subgroup();
}

Id EmitVoteAll(EmitContext& ctx, Id pred) {
    if (!StageSupportsSubgroups(ctx)) {
        return pred;
    }
    const auto subgroup = [&] {
        if (!ctx.profile.warp_size_potentially_larger_than_guest) {
            return ctx.OpGroupNonUniformAll(ctx.U1, SubgroupScope(ctx), pred);
        }
        const Id mask_ballot{
            ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), ctx.true_value)};
        const Id active_mask{WarpExtract(ctx, mask_ballot)};
        const Id ballot{
            WarpExtract(ctx, ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred))};
        const Id lhs{ctx.OpBitwiseAnd(ctx.U32[1], ballot, active_mask)};
        return ctx.OpIEqual(ctx.U1, lhs, active_mask);
    };
    if (ctx.warp_shared) {
        return SharedOr(ctx, ctx.OpIEqual(ctx.U1, SharedBallot(ctx, pred), ctx.warp_region), subgroup, Id{}, true);
    }
    return subgroup();
}

Id EmitVoteAny(EmitContext& ctx, Id pred) {
    if (!StageSupportsSubgroups(ctx)) {
        return pred;
    }
    const auto subgroup = [&] {
        if (!ctx.profile.warp_size_potentially_larger_than_guest) {
            return ctx.OpGroupNonUniformAny(ctx.U1, SubgroupScope(ctx), pred);
        }
        const Id mask_ballot{
            ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), ctx.true_value)};
        const Id active_mask{WarpExtract(ctx, mask_ballot)};
        const Id ballot{
            WarpExtract(ctx, ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred))};
        const Id lhs{ctx.OpBitwiseAnd(ctx.U32[1], ballot, active_mask)};
        return ctx.OpINotEqual(ctx.U1, lhs, ctx.u32_zero_value);
    };
    if (ctx.warp_shared) {
        return SharedOr(ctx, ctx.OpINotEqual(ctx.U1, SharedBallot(ctx, pred), ctx.u32_zero_value), subgroup, Id{}, true);
    }
    return subgroup();
}

Id EmitVoteEqual(EmitContext& ctx, Id pred) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.true_value;
    }
    const auto subgroup = [&] {
        if (!ctx.profile.warp_size_potentially_larger_than_guest) {
            return ctx.OpGroupNonUniformAllEqual(ctx.U1, SubgroupScope(ctx), pred);
        }
        const Id mask_ballot{
            ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), ctx.true_value)};
        const Id active_mask{WarpExtract(ctx, mask_ballot)};
        const Id ballot{
            WarpExtract(ctx, ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred))};
        const Id lhs{ctx.OpBitwiseXor(ctx.U32[1], ballot, active_mask)};
        return ctx.OpLogicalOr(ctx.U1, ctx.OpIEqual(ctx.U1, lhs, ctx.u32_zero_value),
                               ctx.OpIEqual(ctx.U1, lhs, active_mask));
    };
    if (ctx.warp_shared) {
        const Id ballot{SharedBallot(ctx, pred)};
        return SharedOr(ctx,
                        ctx.OpLogicalOr(ctx.U1, ctx.OpIEqual(ctx.U1, ballot, ctx.u32_zero_value),
                                        ctx.OpIEqual(ctx.U1, ballot, ctx.warp_region)),
                        subgroup, Id{}, true);
    }
    return subgroup();
}

Id EmitSubgroupBallot(EmitContext& ctx, Id pred) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.OpSelect(ctx.U32[1], pred, ctx.Const(1u), ctx.u32_zero_value);
    }
    const auto subgroup = [&] {
        const Id ballot{ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred)};
        if (!ctx.profile.warp_size_potentially_larger_than_guest) {
            return ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
        }
        return WarpExtract(ctx, ballot);
    };
    if (ctx.warp_shared) {
        return SharedOr(ctx, SharedBallot(ctx, pred), subgroup);
    }
    return subgroup();
}

Id EmitSubgroupEqMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.Const(1u);
    }
    if (ctx.warp_shared) {
        return SharedOr(ctx, ctx.warp_bit, [&] { return LoadMask(ctx, ctx.subgroup_mask_eq); });
    }
    return LoadMask(ctx, ctx.subgroup_mask_eq);
}

Id EmitSubgroupLtMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.u32_zero_value;
    }
    if (ctx.warp_shared) {
        return SharedOr(ctx, SharedLtMask(ctx), [&] { return LoadMask(ctx, ctx.subgroup_mask_lt); });
    }
    return LoadMask(ctx, ctx.subgroup_mask_lt);
}

Id EmitSubgroupLeMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.Const(1u);
    }
    if (ctx.warp_shared) {
        return SharedOr(ctx, SharedLeMask(ctx), [&] { return LoadMask(ctx, ctx.subgroup_mask_le); });
    }
    return LoadMask(ctx, ctx.subgroup_mask_le);
}

Id EmitSubgroupGtMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.u32_zero_value;
    }
    if (ctx.warp_shared) {
        return SharedOr(ctx, ctx.OpNot(ctx.U32[1], SharedLeMask(ctx)), [&] { return LoadMask(ctx, ctx.subgroup_mask_gt); });
    }
    return LoadMask(ctx, ctx.subgroup_mask_gt);
}

Id EmitSubgroupGeMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.Const(1u);
    }
    if (ctx.warp_shared) {
        return SharedOr(ctx, ctx.OpNot(ctx.U32[1], SharedLtMask(ctx)), [&] { return LoadMask(ctx, ctx.subgroup_mask_ge); });
    }
    return LoadMask(ctx, ctx.subgroup_mask_ge);
}

Id EmitShuffleIndex(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                    Id segmentation_mask) {
    const Id not_seg_mask{ctx.OpNot(ctx.U32[1], segmentation_mask)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id min_thread_id{ComputeMinThreadId(ctx, thread_id, segmentation_mask)};
    const Id max_thread_id{ComputeMaxThreadId(ctx, min_thread_id, clamp, not_seg_mask)};

    const Id lhs{ctx.OpBitwiseAnd(ctx.U32[1], GuestLane(ctx, index), not_seg_mask)};
    const Id src_thread_id{ctx.OpBitwiseOr(ctx.U32[1], lhs, min_thread_id)};
    const Id in_range{ctx.OpSLessThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    if (!StageSupportsSubgroups(ctx)) {
        return value;
    }
    const IR::Value lane{inst->Arg(1).Resolve()};
    const IR::Value segment{inst->Arg(3).Resolve()};
    if (lane.IsImmediate() && segment.IsImmediate() && segment.U32() == 0) {
        const auto subgroup = [&] { return BroadcastLane(ctx, value, lane.U32() & 31); };
        if (ctx.warp_shared) {
            const Id source{ctx.Const(lane.U32() & 31)};
            const Id shared{SharedBroadcast(ctx, value, lane.U32() & 31)};
            return ctx.OpSelect(ctx.U32[1], in_range,
                                SharedOr(ctx, shared, subgroup, ShuffleCounts(ctx, in_range, source)),
                                value);
        }
        return ctx.OpSelect(ctx.U32[1], in_range, subgroup(), value);
    }
    const auto subgroup = [&] {
        return ShuffleAbsolute(ctx, value, HostThreadId(ctx, src_thread_id), UseShuffleFallback(ctx));
    };
    if (ctx.warp_shared) {
        const Id shared{SharedShuffle(ctx, value, src_thread_id)};
        return ctx.OpSelect(ctx.U32[1], in_range,
                            SharedOr(ctx, shared, subgroup, ShuffleCounts(ctx, in_range, src_thread_id)),
                            value);
    }
    return ctx.OpSelect(ctx.U32[1], in_range, subgroup(), value);
}

Id EmitShuffleUp(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                 Id segmentation_mask) {
    if (!StageSupportsSubgroups(ctx)) {
        SetInBoundsFlag(inst, ctx.false_value);
        return value;
    }
    const Id delta{GuestLane(ctx, index)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id max_thread_id{GetMaxThreadId(ctx, thread_id, clamp, segmentation_mask)};
    const Id src_thread_id{ctx.OpISub(ctx.U32[1], thread_id, delta)};
    const Id in_range{ctx.OpSGreaterThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    const auto subgroup = [&] {
        return ShuffleRelative(ctx, value, delta, src_thread_id, spv::Op::OpGroupNonUniformShuffleUp);
    };
    if (ctx.warp_shared) {
        const Id shared{SharedShuffle(ctx, value, src_thread_id)};
        return ctx.OpSelect(ctx.U32[1], in_range,
                            SharedOr(ctx, shared, subgroup, ShuffleCounts(ctx, in_range, src_thread_id)),
                            value);
    }
    return ctx.OpSelect(ctx.U32[1], in_range, subgroup(), value);
}

Id EmitShuffleDown(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                   Id segmentation_mask) {
    if (!StageSupportsSubgroups(ctx)) {
        SetInBoundsFlag(inst, ctx.false_value);
        return value;
    }
    const Id delta{GuestLane(ctx, index)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id max_thread_id{GetMaxThreadId(ctx, thread_id, clamp, segmentation_mask)};
    const Id src_thread_id{ctx.OpIAdd(ctx.U32[1], thread_id, delta)};
    const Id in_range{ctx.OpSLessThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    const auto subgroup = [&] {
        return ShuffleRelative(ctx, value, delta, src_thread_id, spv::Op::OpGroupNonUniformShuffleDown);
    };
    if (ctx.warp_shared) {
        const Id shared{SharedShuffle(ctx, value, src_thread_id)};
        return ctx.OpSelect(ctx.U32[1], in_range,
                            SharedOr(ctx, shared, subgroup, ShuffleCounts(ctx, in_range, src_thread_id)),
                            value);
    }
    return ctx.OpSelect(ctx.U32[1], in_range, subgroup(), value);
}

Id EmitShuffleButterfly(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                        Id segmentation_mask) {
    if (!StageSupportsSubgroups(ctx)) {
        SetInBoundsFlag(inst, ctx.false_value);
        return value;
    }
    const Id mask{GuestLane(ctx, index)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id max_thread_id{GetMaxThreadId(ctx, thread_id, clamp, segmentation_mask)};
    const Id src_thread_id{ctx.OpBitwiseXor(ctx.U32[1], thread_id, mask)};
    const Id in_range{ctx.OpSLessThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    // (hookshot) through the broadcasts where shuffles are broken (UseShuffleFallback: AMD's fragment stages, Lichii #273),
    // as the absolute and relative shuffles are; #273 left SHFL.BFLY native, and some of Kokiri Forest's plants use it
    const auto subgroup = [&] {
        return UseShuffleFallback(ctx)
                   ? ShuffleAbsolute(ctx, value, HostThreadId(ctx, src_thread_id), true)
                   : ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, mask);
    };
    if (ctx.warp_shared) {
        const Id shared{SharedShuffle(ctx, value, src_thread_id)};
        return ctx.OpSelect(ctx.U32[1], in_range,
                            SharedOr(ctx, shared, subgroup, ShuffleCounts(ctx, in_range, src_thread_id)),
                            value);
    }
    return ctx.OpSelect(ctx.U32[1], in_range, subgroup(), value);
}

Id EmitQuadBroadcast(EmitContext& ctx, Id value, Id lane) {
    if (ctx.profile.support_quad_shuffles) {
        return ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, lane);
    }
    const Id base{ctx.OpBitwiseAnd(ctx.U32[1], GetThreadId(ctx), ctx.Const(~3u))};
    const Id local_lane{ctx.OpBitwiseAnd(ctx.U32[1], lane, ctx.Const(3u))};
    const Id src_thread_id{ctx.OpBitwiseOr(ctx.U32[1], base, local_lane)};
    return ShuffleAbsolute(ctx, value, src_thread_id,
                           ctx.profile.has_broken_spirv_subgroup_shuffle);
}

Id EmitQuadSwap(EmitContext& ctx, Id value, Id direction) {
    if (ctx.profile.support_quad_shuffles) {
        const Id result{
            ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, direction)};
        ctx.opcode_rewrites.emplace_back(result.value, spv::Op::OpGroupNonUniformQuadSwap);
        return result;
    }
    const Id xor_mask{ctx.OpIAdd(ctx.U32[1], direction, ctx.Const(1u))};
    // (hookshot) not through UseShuffleFallback's broadcasts: a program with quad shuffles alone (uses_quad_shuffles) has
    // neither the lane id input nor the ballot capability they need, and the SPIR-V made with them crashed the start
    return ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, xor_mask);
}

Id EmitFSwizzleAdd(EmitContext& ctx, Id op_a, Id op_b, Id swizzle) {
    const Id three{ctx.Const(3U)};
    Id mask{GetThreadId(ctx)};
    mask = ctx.OpBitwiseAnd(ctx.U32[1], mask, three);
    mask = ctx.OpShiftLeftLogical(ctx.U32[1], mask, ctx.Const(1U));
    mask = ctx.OpShiftRightLogical(ctx.U32[1], swizzle, mask);
    mask = ctx.OpBitwiseAnd(ctx.U32[1], mask, three);

    const Id modifier_a{ctx.OpVectorExtractDynamic(ctx.F32[1], ctx.fswzadd_lut_a, mask)};
    const Id modifier_b{ctx.OpVectorExtractDynamic(ctx.F32[1], ctx.fswzadd_lut_b, mask)};

    const Id result_a{ctx.OpFMul(ctx.F32[1], op_a, modifier_a)};
    const Id result_b{ctx.OpFMul(ctx.F32[1], op_b, modifier_b)};
    return ctx.OpFAdd(ctx.F32[1], result_a, result_b);
}

Id EmitDPdxFine(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdxFine(ctx.F32[1], op_a);
}

Id EmitDPdyFine(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdyFine(ctx.F32[1], op_a);
}

Id EmitDPdxCoarse(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdxCoarse(ctx.F32[1], op_a);
}

Id EmitDPdyCoarse(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdyCoarse(ctx.F32[1], op_a);
}

// (hookshot) HMMA in software (mma_software.h): the 32 lanes' fragments through subgroup operations in one helper function,
// accumulated with FMA in f32 and rounded to f16 after every 8 columns of K, as HMMA.F16 does. It needs no cooperative matrices
// (RX_VK_MMA_SOFTWARE=1, vk_coopmat.cpp); the form is the one vk_coopmat.cpp's self-test chose for this device (mma_software_form).
void DefineSoftwareMatrixFunction(EmitContext& ctx) {
    ctx.AddCapability(spv::Capability::GroupNonUniform);
    ctx.AddCapability(spv::Capability::GroupNonUniformShuffle);
    if (!ctx.profile.SupportsSubgroupStage(ctx.stage) || ctx.profile.has_broken_spirv_subgroup_shuffle)
        throw NotImplementedException("Software HMMA requires subgroup shuffle support");
    const auto form = static_cast<SoftwareMmaForm>(mma_software_form);
    const bool quad = form != SoftwareMmaForm::Loop && mma_software_quad;
    if (quad) ctx.AddCapability(spv::Capability::GroupNonUniformQuad);
    const SoftwareMmaTypes types{ctx.U32[1], ctx.U32[2], ctx.U32[4], ctx.F32[1], ctx.F32[2], ctx.F16[1], ctx.U1};
    ctx.software_matrix_function = DefineSoftwareMma(ctx, types, ctx.subgroup_local_invocation_id, form, quad);
}


// (hookshot) HMMA_nx2: D(16x8) = A(16x16) * B(16x8) + C on the subgroup's cooperative matrices. Each
// lane's fragment registers hold its elements in the console's order (two f16 a word); they go to and come
// from the device's elements in its own order (mma_layout.h, measured by vk_coopmat.cpp).
Id EmitMatrixMulAddF16(EmitContext& ctx, Id a, Id b, Id c) {
    if (mma_software) return ctx.OpFunctionCall(ctx.U32[2], ctx.software_matrix_function, a, b, c);
    const Id f16{ctx.F16[1]};
    const Id scope{ctx.Const(static_cast<u32>(spv::Scope::Subgroup))};
    const auto type = [&](u32 rows, u32 columns, spv::CooperativeMatrixUse use) {
        return ctx.TypeCooperativeMatrixKHR(f16, scope, ctx.Const(rows), ctx.Const(columns),
                                            ctx.Const(static_cast<u32>(use)));
    };
    const Id a_type{type(16, 16, spv::CooperativeMatrixUse::MatrixAKHR)};
    const Id b_type{type(16, 8, spv::CooperativeMatrixUse::MatrixBKHR)};
    const Id c_type{type(16, 8, spv::CooperativeMatrixUse::MatrixAccumulatorKHR)};
    const Id zero{ctx.Constant(f16, 0U)};
    const auto load = [&](Id matrix_type, Id words, u32 count, const auto& order) {
        Id m{ctx.OpCompositeConstruct(matrix_type, zero)};
        for (u32 w = 0; w < count; ++w) {
            const Id word{count == 1 ? words : ctx.OpCompositeExtract(ctx.U32[1], words, w)};
            const Id pair{ctx.OpBitcast(ctx.F16[2], word)};
            for (u32 h = 0; h < 2; ++h) {
                m = ctx.OpCompositeInsert(matrix_type, ctx.OpCompositeExtract(f16, pair, h), m, order[w * 2 + h]);
            }
        }
        return m;
    };
    const Id d{ctx.OpCooperativeMatrixMulAddKHR(c_type, load(a_type, a, 4, mma_order_a), load(b_type, b, 2, mma_order_b),
                                                    load(c_type, c, 2, mma_order_c))};
    Id out[2];
    for (u32 w = 0; w < 2; ++w) {
        const Id pair{ctx.OpCompositeConstruct(ctx.F16[2], ctx.OpCompositeExtract(f16, d, mma_order_c[w * 2]),
                                               ctx.OpCompositeExtract(f16, d, mma_order_c[w * 2 + 1]))};
        out[w] = ctx.OpBitcast(ctx.U32[1], pair);
    }
    return ctx.OpCompositeConstruct(ctx.U32[2], out[0], out[1]);
}

} // namespace Shader::Backend::SPIRV
