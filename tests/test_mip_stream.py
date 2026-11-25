"""Check Vulkan recording order and publication guards for Vega mip previews."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def function(text, first, following):
    start = text.index(first)
    return text[start:text.index(following, start)]


def ordered(source, *steps):
    start = 0
    for step in steps:
        start = source.index(step, start) + len(step)


def main():
    vk = (ROOT / "gpu/vk/vk_texture.cpp").read_text(encoding="utf-8")
    preview = function(vk, "bool mip_stream_preview(", "bool texture_dirty(")
    ordered(preview, "tex.levels < 2", "for (uint32_t mip = 0; mip < tex.levels; ++mip)",
            "copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1}",
            "vkCmdCopyBufferToImage(", "tex.levels - 1, copies.data()",
            "full_barrier(cmd,", "region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1}",
            "region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}",
            "vkCmdBlitImage(", "tex.mip_stream_ticket = job->ticket")
    promotion = function(vk, "void texture_mip_stream_run_impl()", "bool upload_inner(Texture& tex, const Ranges* dirty) {")
    ordered(promotion, "queue.promoted_frame == frame_no",
            "it->second->id == job->texture_id", "it->second->mip_stream_ticket == job->ticket",
            "texture_dirty(tex, tex.upload_gen, dirty)", "memory_writeback_ever",
            "std::memcmp(job->encoded.data(), guest_bytes(", "tex.uploaded = false",
            "tex.mip_stream_repair = true", "vkCmdCopyBufferToImage(",
            "tex.last_value = std::max(tex.last_value, rec_value)", "queue.promoted_frame = frame_no")
    upload = function(vk, "bool upload(Texture& tex,", "struct AstcCpuCacheEntry")
    ordered(upload, "const uint64_t pending_mip = tex.mip_stream_ticket", "if (pending_mip) dirty = nullptr",
            "upload_inner(tex, dirty)", "if (ok && pending_mip) tex.mip_stream_ticket = 0")
    draw = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    ordered(draw, "mip_stream_draw = s.rt.depth && st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT",
            "texture_view_of(handle", "mip_stream_draw = upload_defer_ok = false")
    print("mip streaming: initialized preview, ordered promotion, stale-source fallback, world draw gate")


if __name__ == "__main__":
    main()
