"""Keep the Vega compute switches on the bounded synchronization paths."""
from pathlib import Path

DRAW = (Path(__file__).resolve().parents[1] / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")


def check_order(source, *steps):
    offset = 0
    for step in steps:
        offset = source.index(step, offset) + len(step)


def main():
    bind = DRAW[DRAW.index("void build_descriptors("):DRAW.index("void descriptors_disturbed()")]
    check_order(bind, "p.compute && exposure_meter_program(p.key) && readback_early_on()",
                "addr == base && size == exposure_bytes", "d.count == 1 && d.is_written && !bare && !wide_n",
                "memory_rb_waited(guest, exposure_bytes)", "gpu_buffer_bind(base, n, d.is_written && !bare)",
                "if (meter && b && b.size >= exposure_bytes)", "++exposure_meter_bindings")
    dispatch = DRAW[DRAW.index("void dispatch(VkCommandBuffer"):DRAW.index("void dispatch(VkCommandBuffer") + 45000]
    check_order(dispatch, "exposure_meter_bindings = 0", "build_descriptors(s, p)", "vkCmdDispatchIndirect(",
                "x && y && z && exposure_meter_bindings == 1", 'full_barrier(cmd, "exposure meter before clear")',
                "vkCmdFillBuffer(cmd, exposure_meter_target.buffer, exposure_meter_target.offset, 16, 0)",
                'full_barrier(cmd, "exposure meter before accumulation")', "vkCmdDispatch(cmd, x, y, z)")
    mesh = DRAW[DRAW.index("bool mesh_indirect_convert("):DRAW.index("void mesh_count_note(")]
    check_order(mesh, "!dev.mesh_shader && scoped_on && !trace_counts", "mesh_single_ring, mesh_single_mem, &mapped",
                "std::fill_n(", "flush_mapped_range(mesh_single_mem", "fixed_yz ? mesh_single_ring : mesh_ring",
                "buffers[1].buffer = converted", "if (!fixed_yz) vkCmdFillBuffer(cmd, converted, off + 4, 8, 1u)",
                "vkCmdCopyBuffer(cmd, args.buffer, converted, n, regions)", "VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT",
                "mesh_conv_buf = converted")
    check_order(DRAW, "if (made && quick) p.optimize_owed.store(true", "else if (made) cs_module_release(p);")
    check_order(dispatch, "if (quick) jobs_push_serial(", "else cs_module_release(q);",
                "q.compute_state.store(2, std::memory_order_release);")
    print("Vega compute tuning: exact meter, fixed mesh arguments, and safe compute module cleanup")


if __name__ == "__main__":
    main()
