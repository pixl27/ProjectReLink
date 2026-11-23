"""Sparse buffers are bound on a queue of the driver's sparse family when the replay's family binds no sparse memory.

Mesa's RADV keeps sparse binding on a queue family of its own (queue flags SPARSE_BINDING alone). The renderer asked only
the replay's graphics family, found none, and took a guest pool over udmabuf's size_limit_mb (64 MB by default) as a
host-pointer import, which the memfd guest window's shared pages cannot be: the pool was no import at all, its storage
buffers bound nothing, the foliage culling's compute passes wrote nowhere, and grass and small plants flickered in and out
(RX 7800 XT, Mesa 25.0.7, v2.1.0: "pool ... not imported (VkResult -13)", "storage buffer NOT bound (reads zeros)").

python -I -B tests/test_sparse_queue.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"


def main():
    head = (VK / "vk_internal.h").read_text(encoding="utf-8")
    if "VkQueue sparse_queue = VK_NULL_HANDLE;" not in head:
        raise SystemExit("the device has no sparse queue")

    setup = (VK / "vk_main.cpp").read_text(encoding="utf-8")
    pick = setup[setup.index("uint32_t sparse_family = UINT32_MAX;"):setup.index("// features: what the hardware has")]
    for want, why in (
        ("!(families[dev.queue_family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)", "another family is asked though the replay's binds sparse memory"),
        ("i != dev.queue_family && i != compute_family", "the sparse family may be the replay's or the presents' own"),
        ("families[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT", "the family picked need not bind sparse memory"),
        ('"RX_VK_SPARSE_QUEUE"', "no switch back to the replay's family alone"),
        ("if (sparse_family != UINT32_MAX) queues(sparse_family, 1);", "no queue of the sparse family is asked of the device"),
    ):
        if want not in pick:
            raise SystemExit(why)
    if "ci.queueCreateInfoCount = (uint32_t)qs.size();" not in setup:
        raise SystemExit("the device is not made with every queue asked for")
    if "dev.sparse_buffers = (own_family_sparse || sparse_family != UINT32_MAX) && f.features.sparseBinding;" not in setup:
        raise SystemExit("sparse buffers do not follow the sparse family and the device's sparseBinding")
    if "vkGetDeviceQueue(dev.device, sparse_family, 0, &dev.sparse_queue);" not in setup:
        raise SystemExit("the sparse family's queue is not taken")
    # (the sparse textures bind on the replay's queue: they stay with its family's own flag)
    images = setup[setup.index("dev.sparse_images ="):setup.index("const bool own_family_sparse")]
    if "families[dev.queue_family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT" not in images:
        raise SystemExit("sparse images no longer follow the replay's own family")

    memory = (VK / "vk_memory.cpp").read_text(encoding="utf-8")
    start = memory.index("bool import_dma_buf_pieces(")
    pieces = memory[start:memory.index("\n}\n", start)]
    own = pieces.index("vkQueueBindSparse(dev.sparse_queue, 1, &si, fence);")
    shared = pieces.index("vkQueueBindSparse(dev.queue, 1, &si, fence);")
    if not pieces.index("dev.sparse_queue != dev.queue") < own < shared:
        raise SystemExit("a sparse queue of its own is not the first choice for a pool's dma-bufs")
    if "queue_flush_locked();" in pieces[own - 400:own]:
        raise SystemExit("the sparse family's bind flushes the replay's held submits")
    if "std::lock_guard qk(dev.present_mu);" not in pieces[own:shared]:
        raise SystemExit("a bind on the replay's queue does not hold its lock")
    if pieces.index("std::lock_guard fk(fence_mu);") > own:
        raise SystemExit("the sparse queue's one lock is not held for its bind")

    # (the sparse mirrors bind on the replay's queue only: never on, and never made, where the pools use a queue of their own)
    mirror = memory[memory.index("bool mirror_sparse_on() {"):memory.index("uint64_t mirror_size(")]
    if "dev.sparse_buffers && own_queue &&" not in mirror or "const bool own_queue = dev.sparse_queue == dev.queue;" not in mirror:
        raise SystemExit("the sparse VRAM mirrors may be on where the replay's queue binds no sparse memory")

    start = memory.index("bool import_dma_buf(")
    whole = memory[start:memory.index("\n}\n", start)]
    # (main says it before RADV's import headroom is taken: one test, hi - lo > piece && !dev.sparse_buffers)
    refuse = whole[whole.index("if (hi - lo > piece && !dev.sparse_buffers) {"):whole.index("if (!import_dma_buf_pieces(lo, hi, piece, p)) return false;")]
    if "size_limit_mb" not in refuse or "return false;" not in refuse:
        raise SystemExit("a pool that cannot be imported is not said, with udmabuf's limit to raise")
    print("sparse buffers bound on the sparse family's queue ok")


if __name__ == "__main__":
    main()
