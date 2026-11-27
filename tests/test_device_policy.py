"""What each PC gets by being what it is (gpu/vk/device_policy.h), on made-up PCs, and where the renderer fills and reads it.

device_policy() is compiled with the Vulkan headers and run on the facts of PCs players have: NVIDIA's driver on Linux
and Windows (4, 6, 8 and 24 GB cards), NVK, AMD's Windows driver in its two compiler branches, on a Navi 22 card and on
the graphics in a Ryzen 7000 desktop CPU, RADV on an RDNA 2 card, on a GCN card and on the Steam Deck, AMDVLK, Intel's
Windows driver on an integrated GPU with 4 threads and on three Arc cards, ANV, llvmpipe. For each PC the check names
every field of DevicePolicy: the yes/no fields that are on (every other one must be off) and every number. A rule that
leaks to a PC it is not for, or a new field no PC here expects, fails by name.

The values are the rules as the renderer has them, which device_policy() holds for the sites that read it and repeats for
those that still decide for themselves. Four rules were corrected when they moved here (the PCs' comments say "corrected":
Compute for mesh programs by RADV and not by "AMD off Windows"; one program's builds serial by AMD's Windows driver and
compute builds serial by Intel's, not by the vendors; batched submits' GCN by the one definition). Where the study of these
rules doubts another one, the PC's comment says so and the value is today's all the same. The sizes are in whole MB.

Then the switches (device_policy_env) through their one rule, on made-up environments: unset or empty is the PC's
default, 0, off, false and no are off, any other value on; a number is digits only, in its range, and anything else is
ignored and named in the notes; the two that can only turn their rule off; what follows the mirrors follows the switch.
Each case's comment says where a value means something else than it did before the one rule.

Then the rules' edges (policy_rule), the log block (six "vk: policy:" lines a PC, each under 900 characters, whatever
the driver calls itself), and the source: create_device fills the facts and makes the policy before the device, the start
logs the block, the sites that moved read the policy, the header is no file of the shader translation's stamp, and each
field the translation reads (vk_shader.cpp, the one stamped file that does) is folded into the program cache's stamp.

    python -I -B tests/test_device_policy.py --headers <Vulkan-Headers/include> [--cxx g++]
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (python -I leaves the check's own folder off the path)
from srccut import block   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"

# The yes/no fields, in DevicePolicy's order
BOOLS = """fetch_bounds nan_bits mesh_fallback_strict_fmul bc_copy_per_layer gather_offset frag_shuffle_fallback position_invariant mesh_no_optimizer
amd_retry_no_optimizer amd_cloth_no_optimizer fault_gate_serial task_mesh_fallback intel_compile_gate compute_builds_serial
pcache_fetch_lock split_before_draw astc_on_cpu inline_deku_web_loads networks_left_out dma_buf_imports nv_checkpoints
allowance_at_count_rule
mirrors ub_snapshot split_ub_flush native_float_add warp64 mesh_shared_warp batch_submits mma_unroll_try global_table_loop occ_copy
intel_side_by_side_auto fsr4_tune gpu_clock_peak presenter_queue pcache_small_pc small_vram_textures astc_bc_all
room_in_blocks_rule squeeze_damped linux_hold_caps_budget dlss_cnn auto_render_third rtx rdna4 threaded_auto_render
threaded_auto_recorder drop_modules_small_ram native_zstd_keep_core""".split()
# The numbers: the field and how the fixture prints it (sizes in whole MB; -1: no limit)
NUMS = [
    ("equal_depth_bias", "(long long)p.equal_depth_bias"),
    ("mesh_mode", "(long long)p.mesh_mode"),
    ("fallback_phi_limit", "(long long)p.fallback_phi_limit"),
    ("retired_mode", "(long long)p.retired_mode"),
    ("idle_import_high_pct", "(long long)p.idle_import_high_pct"),
    ("import_chunk_mb", "(long long)p.import_chunk_mb"),
    ("texture_budget", "mbs(p.texture_budget)"),
    ("mirror_budget", "mbs(p.mirror_budget)"),
    ("target_pool_mb", "(long long)p.target_pool_mb"),
    ("compact_min_free", "mbs(p.compact_min_free)"),
    ("wb_shadow_cap", "mbs(p.wb_shadow_cap)"),
    ("cap_spare_max", "mbs(p.cap_spare_max)"),
    ("staging_keep", "mbs(p.staging_keep)"),
    ("image_blocks_at_start", "(long long)p.image_blocks_at_start"),
    ("vma_block", "mbs(p.vma_block)"),
    ("target_block", "mbs(p.target_block)"),
    ("count_line_pct", "(long long)p.count_line_pct"),
    ("auto_tier", "(long long)p.auto_tier"),
    ("workers", "(long long)p.workers"),
    ("capture_helpers", "(long long)p.capture_helpers"),
    ("texture_threads", "(long long)p.texture_threads"),
    ("texture_gather_threads", "(long long)p.texture_gather_threads"),
    ("program_workers_in_play", "(long long)p.program_workers_in_play"),
]
NUM_NAMES = [n for n, _ in NUMS]

INTEL = 101 << 14   # (Intel's Windows driverVersion: 101 << 14 | the build)
AMD_OLD = "(AMD proprietary shader compiler)"

# The facts' columns, in the fixture's struct D's order:
#   vendor, device, driver id, driver version, device type, driverInfo, the GPU's name, windows, SteamDeck=1,
#   VRAM heap MB, allowed at the start MB, RAM MB, CPUs, usable CPUs, the physical cores among those (SMT siblings once:
#   Threaded rendering's Auto counts by them, the render thread from 4 and the recorder thread too from 6),
#   mesh shaders, buffer device addresses, float add, cooperative matrices, NV checkpoints, 16-bit stage inputs,
#   dynamic indexing of storage buffer arrays, lanes, most lanes, the stages pinned to 32
# num: the numbers in NUM_NAMES' order:
#   equal depth bias, mesh mode, phi limit, retired mode (1 at once, 2 after a wait), idle imports' high %, import chunk MB,
#   textures MB, mirrors MB, target pool MB, compaction from MB, shadows MB, spare captures MB (-1 none), staging kept MB,
#   blocks at the start, VMA block MB, target block MB, count line %, auto tier,
#   workers, capture helpers, texture threads, gather threads, program workers in play
PCS = [
    dict(name="RTX 3090 (24 GB), NVIDIA's Linux driver, 24 threads",
         facts=(0x10DE, 0x2204, "NVIDIA_PROPRIETARY", 0, "DISCRETE_GPU", "580.105.08", "NVIDIA GeForce RTX 3090", 0, 0,
                24576, 24000, 32768, 24, 24, 12, 1, 1, 1, 1, 1, 1, 1, 32, 32, 0),
         on="inline_deku_web_loads nv_checkpoints allowance_at_count_rule mirrors split_ub_flush native_float_add "
            "presenter_queue room_in_blocks_rule rtx threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 1, 85, 4, 11000, 3072, 768, 512, 3000, -1, 256, 24, 64, 32, 97, 5, 6, 4, 8, 6, 4)),
    # (an 8 GB card: every large ASTC texture as BC3, the squeeze damped, 12 blocks at the start and the 32 MB blocks)
    dict(name="RTX 4070 Laptop (8 GB), NVIDIA's Windows driver",
         facts=(0x10DE, 0x2860, "NVIDIA_PROPRIETARY", 0, "DISCRETE_GPU", "581.57", "NVIDIA GeForce RTX 4070 Laptop GPU", 1, 0,
                8188, 7400, 16200, 16, 16, 8, 1, 1, 1, 1, 1, 1, 1, 32, 32, 0),
         on="inline_deku_web_loads nv_checkpoints allowance_at_count_rule mirrors split_ub_flush native_float_add "
            "presenter_queue astc_bc_all room_in_blocks_rule squeeze_damped rtx threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 1, 85, 4, 4776, 2047, 616, 462, 925, -1, 256, 12, 32, 16, 97, 3, 6, 4, 8, 6, 4)),
    # (a 4 GB card. The rules that ask the vendor take NVK for NVIDIA's compiler: no invariant positions, no fetch bounds,
    # the Deku web's loads inlined, the presenter's queue; the study doubts each. And it doubts the mirrors' 1365 MB here,
    # a third of the card, where mirror_budget.h argues for 256)
    dict(name="GTX 1650 (4 GB), NVK",
         facts=(0x10DE, 0x1F82, "MESA_NVK", 0, "DISCRETE_GPU", "Mesa 26.0.1", "NVIDIA GeForce GTX 1650 (NVK TU117)", 0, 0,
                4096, 3900, 16000, 8, 8, 4, 0, 1, 0, 0, 0, 1, 1, 32, 32, 0),
         on="inline_deku_web_loads dma_buf_imports allowance_at_count_rule mirrors split_ub_flush presenter_queue "
            "small_vram_textures astc_bc_all room_in_blocks_rule squeeze_damped dlss_cnn threaded_auto_render "
            "drop_modules_small_ram",
         num=(0, 0, 0, 1, 85, 4, 2200, 1365, 325, 243, 487, 32, 128, 10, 32, 16, 97, 1, 2, 2, 4, 4, 2)),
    # (Navi 22: task + mesh draws on the compute fallback, and with it the retry without the optimizer)
    dict(name="RX 6700 XT, AMD's Windows driver (RDNA 2 compiler)",
         facts=(0x1002, 0x73DF, "AMD_PROPRIETARY", 0x800161, "DISCRETE_GPU", "26.10.1 " + AMD_OLD, "AMD Radeon RX 6700 XT", 1, 0,
                12272, 11500, 32600, 12, 12, 6, 1, 1, 0, 0, 0, 1, 1, 64, 64, 0xFF),
         on="bc_copy_per_layer fetch_bounds networks_left_out nan_bits mesh_fallback_strict_fmul gather_offset frag_shuffle_fallback position_invariant mesh_no_optimizer "
            "amd_retry_no_optimizer amd_cloth_no_optimizer fault_gate_serial task_mesh_fallback allowance_at_count_rule "
            "mirrors split_ub_flush warp64 fsr4_tune room_in_blocks_rule threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 2, 85, 4, 7158, 2048, 768, 512, 1437, -1, 256, 24, 64, 32, 97, 4, 4, 4, 6, 6, 3)),
    # (the card has the float add, but no run on AMD has shown it right: the loop. RDNA 4 tries the software matrix
    # multiplies' unrolled forms on AMD's proprietary driver; the study doubts the fragment shuffles' fallback and the
    # unoptimized mesh programs on LLPC, where neither was measured)
    dict(name="RX 9070 XT, AMD's Windows driver (LLPC)",
         facts=(0x1002, 0x7550, "AMD_PROPRIETARY", 0x800161, "DISCRETE_GPU", "26.10.1 (LLPC)", "AMD Radeon RX 9070 XT", 1, 0,
                16304, 15300, 32600, 16, 16, 8, 1, 1, 1, 1, 0, 1, 1, 64, 64, 0xFF),
         on="bc_copy_per_layer fetch_bounds networks_left_out gather_offset frag_shuffle_fallback position_invariant mesh_no_optimizer amd_cloth_no_optimizer "
            "fault_gate_serial allowance_at_count_rule mirrors split_ub_flush warp64 mma_unroll_try fsr4_tune room_in_blocks_rule rdna4 "
            "threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 2, 85, 4, 11000, 3072, 768, 512, 1912, -1, 256, 24, 64, 32, 97, 5, 6, 4, 8, 6, 4)),
    # (Compute for its mesh programs, with no Mesa version that ends it, which the study doubts.
    # Corrected: fault_gate_serial is off, as issue 996 is AMD's Windows driver's; by the vendor it was on here)
    dict(name="RX 6800 XT, RADV",
         facts=(0x1002, 0x73BF, "MESA_RADV", 0, "DISCRETE_GPU", "Mesa 26.2.5", "AMD Radeon RX 6800 XT (RADV NAVI21)", 0, 0,
                16368, 16000, 32000, 16, 16, 8, 1, 1, 1, 0, 0, 1, 1, 64, 64, 0xFF),
         on="fetch_bounds networks_left_out gather_offset frag_shuffle_fallback position_invariant dma_buf_imports allowance_at_count_rule mirrors "
            "split_ub_flush warp64 batch_submits mma_unroll_try fsr4_tune room_in_blocks_rule "
            "threaded_auto_render threaded_auto_recorder",
         num=(0, 2, 0, 1, 85, 4, 11000, 3072, 768, 512, 2000, -1, 256, 24, 64, 32, 97, 5, 6, 4, 8, 6, 4)),
    # (RDNA 4 on RADV: no batched submits, the gaps between held submits let its clock drop)
    dict(name="RX 9070 XT, RADV",
         facts=(0x1002, 0x7550, "MESA_RADV", 0, "DISCRETE_GPU", "Mesa 26.2.6", "AMD Radeon RX 9070 XT (RADV GFX1201)", 0, 0,
                16304, 15300, 32600, 16, 16, 8, 1, 1, 1, 0, 0, 1, 1, 64, 64, 0xFF),
         on="fetch_bounds networks_left_out gather_offset frag_shuffle_fallback position_invariant dma_buf_imports allowance_at_count_rule mirrors "
            "split_ub_flush warp64 mma_unroll_try fsr4_tune room_in_blocks_rule rdna4 "
            "threaded_auto_render threaded_auto_recorder",
         num=(0, 2, 0, 1, 85, 4, 11000, 3072, 768, 512, 1912, -1, 256, 24, 64, 32, 97, 5, 6, 4, 8, 6, 4)),
    # (GCN: no batched submits, the phi limit. Its heap is 8192 MB exactly: the small blocks (<= 8 GB) and the large
    # card's 15% of blocks at the start (< 8 GB is the other branch), the two comparators the study names.
    # Corrected: fault_gate_serial is off on RADV)
    dict(name="RX 580 (8 GB), RADV on GCN",
         facts=(0x1002, 0x67DF, "MESA_RADV", 0, "DISCRETE_GPU", "Mesa 26.2.5", "AMD Radeon RX 580 Series (RADV POLARIS10)", 0, 0,
                8192, 8000, 16000, 8, 8, 4, 0, 1, 1, 0, 0, 0, 1, 64, 64, 0),
         on="fetch_bounds networks_left_out gather_offset frag_shuffle_fallback position_invariant dma_buf_imports allowance_at_count_rule mirrors "
            "split_ub_flush warp64 mma_unroll_try fsr4_tune astc_bc_all room_in_blocks_rule squeeze_damped "
            "threaded_auto_render drop_modules_small_ram",
         num=(0, 2, 50000, 1, 85, 4, 4778, 2048, 666, 500, 1000, -1, 256, 24, 32, 16, 97, 1, 2, 2, 4, 4, 2)),
    # (corrected: batched submits are on. Vega is GFX9, not GCN 1-4: it has 16-bit stage inputs, which is the one
    # definition of GCN (DeviceFacts::radv_gcn); batch_on's own, no stage pinned to 32 lanes and 64-lane waves, took it
    # for GCN and left them off. Nothing was measured on Vega either way. Its CPU here has 8 cores without SMT: Threaded
    # rendering's Auto gives it both threads, where 8 threads of 4 cores (the RX 580's, the same count) get the render thread)
    dict(name="RX Vega 64 (8 GB), RADV on GFX9",
         facts=(0x1002, 0x687F, "MESA_RADV", 0, "DISCRETE_GPU", "Mesa 26.2.5", "AMD Radeon RX Vega (RADV VEGA10)", 0, 0,
                8176, 7900, 16000, 8, 8, 8, 0, 1, 1, 0, 0, 1, 1, 64, 64, 0),
         on="fetch_bounds networks_left_out gather_offset frag_shuffle_fallback position_invariant dma_buf_imports allowance_at_count_rule mirrors split_ub_flush "
            "warp64 batch_submits mma_unroll_try fsr4_tune astc_bc_all room_in_blocks_rule squeeze_damped "
            "threaded_auto_render threaded_auto_recorder drop_modules_small_ram",
         num=(0, 2, 0, 1, 85, 4, 4769, 2044, 658, 493, 987, -1, 256, 12, 32, 16, 97, 1, 2, 2, 4, 4, 2)),
    # (no mirrors, so 16 MB chunks; the mirrors' budget is worked out all the same, by the heap. Uniform updates are still
    # written at each part's end and no snapshot is taken, which the study doubts for every GPU without mirrors.
    # Corrected: fault_gate_serial is off on RADV. Its 4 cores and 8 threads: Threaded rendering's Auto by the 4 cores is the
    # render thread alone, which the menu then turns off all the same, as the pools have no mirrors)
    dict(name="Steam Deck (LCD), RADV",
         facts=(0x1002, 0x163F, "MESA_RADV", 0, "INTEGRATED_GPU", "Mesa 25.1.0", "AMD Custom GPU 0405 (RADV VANGOGH)", 0, 1,
                7168, 6800, 14800, 8, 8, 4, 1, 1, 1, 0, 0, 1, 1, 64, 64, 0xFF),
         on="fetch_bounds networks_left_out gather_offset frag_shuffle_fallback position_invariant dma_buf_imports allowance_at_count_rule "
            "split_ub_flush warp64 batch_submits mma_unroll_try fsr4_tune gpu_clock_peak small_vram_textures astc_bc_all "
            "squeeze_damped linux_hold_caps_budget threaded_auto_render drop_modules_small_ram",
         num=(0, 2, 0, 1, 85, 16, 2560, 1792, 566, 425, 512, -1, 256, 0, 32, 16, 90, 1, 2, 2, 4, 4, 2)),
    # (corrected, twice: mesh_mode is Automatic (0), so its own mesh shaders, as the deadlock of issue 2245 is RADV's and
    # the rule now asks for RADV; "AMD off Windows" gave AMDVLK Compute. And fault_gate_serial is off, as on RADV.
    # LLPC's task and mesh programs still go unoptimized here, which the study doubts)
    dict(name="RX 6800, AMDVLK",
         facts=(0x1002, 0x73BF, "AMD_OPEN_SOURCE", 0, "DISCRETE_GPU", "2025.Q2.1 (LLPC)", "AMD Radeon RX 6800", 0, 0,
                16368, 16000, 32000, 12, 12, 6, 1, 1, 0, 0, 0, 1, 1, 64, 64, 0xFF),
         on="fetch_bounds networks_left_out gather_offset position_invariant mesh_no_optimizer dma_buf_imports "
            "allowance_at_count_rule mirrors split_ub_flush warp64 fsr4_tune room_in_blocks_rule "
            "threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 1, 85, 4, 11000, 3072, 768, 512, 2000, -1, 256, 24, 64, 32, 97, 5, 4, 4, 6, 6, 3)),
    # (4 threads beside an integrated GPU: Auto's 33%. It reports the float add, but its family is not on the list. The
    # part split before a plain draw is on though it has no mesh shaders, which the study doubts)
    dict(name="UHD Graphics (i3-1115G4, 16 GB, 4 threads), Intel's Windows driver",
         facts=(0x8086, 0x9A78, "INTEL_PROPRIETARY_WINDOWS", INTEL | 7088, "INTEGRATED_GPU", "101.7088", "Intel(R) UHD Graphics", 1, 0,
                8100, 7402, 16100, 4, 4, 2, 0, 1, 1, 0, 0, 1, 1, 32, 32, 0xFF),
         on="fetch_bounds networks_left_out gather_offset position_invariant intel_compile_gate compute_builds_serial pcache_fetch_lock "
            "split_before_draw allowance_at_count_rule ub_snapshot small_vram_textures astc_bc_all squeeze_damped "
            "auto_render_third native_zstd_keep_core",
         num=(64, 0, 0, 2, 0, 16, 2560, 2025, 616, 462, 512, -1, 256, 0, 32, 16, 90, 1, 2, 2, 2, 2, 2)),
    # (Battlemage from 101.9154: side by side, so no gate, and the float add; equal_depth_bias stays 0 so alpha-tested
    # Z-prepass cutouts do not render opaque. Compute builds stay one at a time though the gate is open, which the study
    # doubts and a B580 would have to say)
    dict(name="Arc B580, Intel's Windows driver 101.9154",
         facts=(0x8086, 0xE20B, "INTEL_PROPRIETARY_WINDOWS", INTEL | 9154, "DISCRETE_GPU", "101.9154", "Intel(R) Arc(TM) B580 Graphics",
                1, 0, 12216, 11600, 32500, 20, 20, 14, 1, 1, 1, 0, 0, 1, 1, 32, 32, 0xFF),
         on="fetch_bounds networks_left_out gather_offset position_invariant compute_builds_serial pcache_fetch_lock split_before_draw "
            "allowance_at_count_rule mirrors ub_snapshot split_ub_flush native_float_add occ_copy intel_side_by_side_auto "
            "room_in_blocks_rule threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 2, 0, 4, 7126, 2048, 768, 512, 1450, -1, 256, 24, 64, 32, 97, 4, 6, 4, 8, 6, 4)),
    # (Battlemage Arc B570 on Windows 101.9175: equal_depth_bias stays 0 so Z-prepass alpha cutouts do not render opaque)
    dict(name="Arc B570, Intel's Windows driver 101.9175",
         facts=(0x8086, 0xE20C, "INTEL_PROPRIETARY_WINDOWS", INTEL | 9175, "DISCRETE_GPU", "101.9175", "Intel(R) Arc(TM) B570 Graphics",
                1, 0, 10074, 9305, 16311, 12, 12, 6, 1, 1, 1, 0, 0, 1, 1, 32, 32, 0xFF),
         on="fetch_bounds networks_left_out gather_offset position_invariant compute_builds_serial pcache_fetch_lock split_before_draw "
            "allowance_at_count_rule mirrors ub_snapshot split_ub_flush native_float_add occ_copy intel_side_by_side_auto "
            "room_in_blocks_rule threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 2, 0, 4, 5876, 2048, 768, 512, 1163, -1, 256, 24, 64, 32, 97, 3, 4, 4, 6, 6, 3)),
    # (an Arc A card: the gate, and the loop for float adds, which the study doubts on Intel's Windows driver)
    dict(name="Arc A750, Intel's Windows driver 101.9175",
         facts=(0x8086, 0x56A1, "INTEL_PROPRIETARY_WINDOWS", INTEL | 9175, "DISCRETE_GPU", "101.9175", "Intel(R) Arc(TM) A750 Graphics",
                1, 0, 8128, 7700, 16200, 12, 12, 6, 1, 1, 1, 0, 0, 1, 1, 32, 32, 0xFF),
         on="fetch_bounds networks_left_out gather_offset position_invariant intel_compile_gate compute_builds_serial pcache_fetch_lock "
            "split_before_draw allowance_at_count_rule mirrors ub_snapshot split_ub_flush occ_copy astc_bc_all "
            "room_in_blocks_rule squeeze_damped threaded_auto_render threaded_auto_recorder",
         num=(64, 0, 0, 2, 0, 4, 4741, 2032, 641, 481, 962, -1, 256, 12, 32, 16, 97, 3, 4, 4, 6, 6, 3)),
    # (corrected: compute_builds_serial is off, as issue 496 is the compiler of Intel's Windows driver; by the vendor it
    # was on here. No equal-depth margin either: the game's EQUAL holds on Mesa's compiler (Iris Xe, Mesa 26.2.7:
    # Kakariko, the Death Mountain gate and trail without a hole), and the margin's LESS_OR_EQUAL let the terrain draw
    # in front of the canyon's rocks on Death Mountain Trail, a flat wall where the depth pass had none. The part split
    # still asks the vendor and reaches Mesa's driver: its site names no driver, so a run on ANV has to say, and the
    # study doubts it)
    dict(name="Iris Xe, ANV",
         facts=(0x8086, 0x9A49, "INTEL_OPEN_SOURCE_MESA", 0, "INTEGRATED_GPU", "Mesa 26.0.1", "Mesa Intel(R) Xe Graphics (TGL GT2)", 0, 0,
                11850, 10000, 15800, 8, 8, 4, 0, 1, 1, 0, 0, 1, 1, 32, 32, 0xFF),
         on="fetch_bounds networks_left_out gather_offset position_invariant split_before_draw dma_buf_imports "
            "allowance_at_count_rule ub_snapshot mesh_shared_warp global_table_loop small_vram_textures astc_bc_all linux_hold_caps_budget "
            "threaded_auto_render drop_modules_small_ram native_zstd_keep_core",
         num=(0, 0, 20000, 1, 85, 16, 2560, 2048, 768, 512, 512, -1, 256, 0, 64, 32, 90, 1, 2, 2, 4, 4, 2)),
    # (neither discrete nor integrated: no mirrors, yet a dedicated card's budgets, blocks at the start and 97% line.
    # The study doubts it: the two tests are not a partition)
    dict(name="llvmpipe",
         facts=(0x10005, 0, "MESA_LLVMPIPE", 0, "CPU", "Mesa 26.0.1 (LLVM 20.1.8)", "llvmpipe (LLVM 20.1.8, 256 bits)", 0, 0,
                32000, 25600, 32000, 16, 16, 8, 0, 1, 0, 0, 0, 1, 1, 8, 8, 0),
         on="fetch_bounds networks_left_out position_invariant dma_buf_imports allowance_at_count_rule split_ub_flush room_in_blocks_rule "
            "threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 1, 85, 16, 11000, 3072, 768, 512, 3200, -1, 256, 24, 64, 32, 97, 1, 6, 4, 8, 6, 4)),
    # (a 6 GB card: the small spares and staging, the CNN model. The study doubts its 1700 MB of mirrors, set before
    # mirror_budget.h's rule is reached)
    dict(name="RTX 3060 Laptop (6 GB), NVIDIA's Windows driver",
         facts=(0x10DE, 0x2520, "NVIDIA_PROPRIETARY", 0, "DISCRETE_GPU", "581.57", "NVIDIA GeForce RTX 3060 Laptop GPU", 1, 0,
                6144, 5700, 16200, 16, 16, 8, 1, 1, 1, 1, 1, 1, 1, 32, 32, 0),
         on="inline_deku_web_loads nv_checkpoints allowance_at_count_rule mirrors split_ub_flush native_float_add "
            "presenter_queue astc_bc_all room_in_blocks_rule squeeze_damped dlss_cnn rtx "
            "threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 1, 85, 4, 3584, 1700, 475, 356, 712, 32, 128, 12, 32, 16, 97, 2, 6, 4, 8, 6, 4)),
    dict(name="RTX 4090 (24 GB), NVIDIA's Windows driver",
         facts=(0x10DE, 0x2684, "NVIDIA_PROPRIETARY", 0, "DISCRETE_GPU", "581.57", "NVIDIA GeForce RTX 4090", 1, 0,
                24564, 23500, 65000, 32, 32, 16, 1, 1, 1, 1, 1, 1, 1, 32, 32, 0),
         on="inline_deku_web_loads nv_checkpoints allowance_at_count_rule mirrors split_ub_flush native_float_add "
            "presenter_queue room_in_blocks_rule rtx threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 1, 85, 4, 11000, 3072, 768, 512, 2937, -1, 256, 24, 64, 32, 97, 5, 6, 4, 8, 6, 4)),
    # (no mesh shaders: Minimum, and no task + mesh fallback to choose)
    dict(name="RX 5700 (8 GB), AMD's Windows driver (RDNA 1 compiler)",
         facts=(0x1002, 0x731F, "AMD_PROPRIETARY", 0x800161, "DISCRETE_GPU", "25.9.1 " + AMD_OLD, "AMD Radeon RX 5700", 1, 0,
                8176, 7378, 16300, 12, 12, 6, 0, 1, 0, 0, 0, 1, 1, 64, 64, 0x20),
         on="bc_copy_per_layer fetch_bounds networks_left_out nan_bits mesh_fallback_strict_fmul gather_offset frag_shuffle_fallback position_invariant mesh_no_optimizer "
            "amd_cloth_no_optimizer fault_gate_serial allowance_at_count_rule mirrors split_ub_flush warp64 fsr4_tune "
            "astc_bc_all room_in_blocks_rule squeeze_damped threaded_auto_render threaded_auto_recorder",
         num=(0, 0, 0, 2, 85, 4, 4769, 2044, 614, 461, 922, -1, 256, 12, 32, 16, 97, 1, 4, 4, 6, 6, 3)),
    # (ASTC on the CPU: 70% of 10,232 MB allowed is 7,162 MB, under the 8 GB cap for 32 GB RAM. The process may use 8 of the
    # PC's 24 threads: the workers go by the 24, Threaded rendering's Auto by the 4 cores of those 8 (the render thread
    # alone), which the study names as two ways of counting)
    dict(name="Ryzen 9 7900X's graphics (32 GB), AMD's Windows driver, on 8 of 24 threads",
         facts=(0x1002, 0x164E, "AMD_PROPRIETARY", 0x800161, "INTEGRATED_GPU", "26.10.1 " + AMD_OLD, "AMD Radeon(TM) Graphics", 1, 0,
                16300, 10232, 32600, 24, 8, 4, 1, 1, 0, 0, 0, 1, 1, 64, 64, 0xFF),
         on="bc_copy_per_layer fetch_bounds networks_left_out nan_bits mesh_fallback_strict_fmul gather_offset frag_shuffle_fallback position_invariant mesh_no_optimizer "
            "amd_cloth_no_optimizer fault_gate_serial astc_on_cpu allowance_at_count_rule split_ub_flush warp64 occ_copy fsr4_tune "
            "astc_bc_all threaded_auto_render",
         num=(0, 0, 0, 2, 85, 16, 7162, 3072, 768, 512, 512, -1, 256, 0, 64, 32, 90, 1, 6, 4, 8, 6, 4)),
    dict(name="Radeon 860M (0x1002:0x1114), AMD's Windows driver 26.10.1 (LLPC)",
         facts=(0x1002, 0x1114, "AMD_PROPRIETARY", 0x800161, "INTEGRATED_GPU", "26.10.1 " + AMD_OLD, "AMD Radeon 860M", 1, 0,
                16300, 10232, 32600, 24, 8, 4, 1, 1, 0, 0, 0, 1, 1, 64, 64, 0xFF),
         on="bc_copy_per_layer fetch_bounds networks_left_out nan_bits mesh_fallback_strict_fmul gather_offset frag_shuffle_fallback position_invariant mesh_no_optimizer "
            "amd_cloth_no_optimizer fault_gate_serial astc_on_cpu allowance_at_count_rule split_ub_flush warp64 occ_copy fsr4_tune "
            "astc_bc_all threaded_auto_render",
         num=(0, 0, 0, 2, 85, 16, 7162, 3072, 768, 512, 512, -1, 256, 0, 64, 32, 90, 1, 6, 4, 8, 6, 4)),
]

# The switches (device_policy_env). The fields a switch can set, in the Sw enum's order (its bit in set_by_switch), and what
# else the cases look at: the import chunk (which follows the mirrors) and the texture budget (which follows astc_on_cpu)
SW = ["astc_on_cpu", "batch_submits", "compute_builds_serial", "pcache_fetch_lock", "split_before_draw", "ub_snapshot",
      "split_ub_flush", "occ_copy", "mirrors", "fsr4_tune", "position_invariant", "equal_depth_bias", "cap_spare_max",
      "staging_keep", "workers", "mesh_fallback_strict_fmul", "bc_copy_per_layer"]
ENV_FIELDS = SW + ["import_chunk_mb", "texture_budget"]
NVIDIA, RADV, B580, UHD, AMD_WIN, RAPHAEL, SMALL = (
    "RTX 3090 (24 GB), NVIDIA's Linux driver, 24 threads", "RX 6800 XT, RADV", "Arc B580, Intel's Windows driver 101.9154",
    "UHD Graphics (i3-1115G4, 16 GB, 4 threads), Intel's Windows driver", "RX 6700 XT, AMD's Windows driver (RDNA 2 compiler)",
    "Ryzen 9 7900X's graphics (32 GB), AMD's Windows driver, on 8 of 24 threads", "RTX 3060 Laptop (6 GB), NVIDIA's Windows driver")
# Each: the PC, the environment, the fields that differ from the PC's defaults (PCS), the fields marked as set by a switch,
# and the switches whose value must be named as ignored. "was": what the value did before the one rule
ENV = [
    (NVIDIA, {}, {}, [], []),
    # yes/no: the off words off, anything else on, unset or empty the default
    (NVIDIA, {"RX_VK_BATCH": "1"}, {"batch_submits": 1}, ["batch_submits"], []),
    (NVIDIA, {"RX_VK_BATCH": "on"}, {"batch_submits": 1}, ["batch_submits"], []),      # (was off: read as a number)
    (NVIDIA, {"RX_VK_BATCH": "true"}, {"batch_submits": 1}, ["batch_submits"], []),    # (was off)
    (NVIDIA, {"RX_VK_BATCH": "00"}, {"batch_submits": 1}, ["batch_submits"], []),      # (was off: no off word, so on)
    (RADV, {"RX_VK_BATCH": "0"}, {"batch_submits": 0}, ["batch_submits"], []),
    (RADV, {"RX_VK_BATCH": " OFF "}, {"batch_submits": 0}, ["batch_submits"], []),
    (RADV, {"RX_VK_BATCH": "no"}, {"batch_submits": 0}, ["batch_submits"], []),
    (RADV, {"RX_VK_BATCH": "False"}, {"batch_submits": 0}, ["batch_submits"], []),
    (RADV, {"RX_VK_BATCH": ""}, {}, [], []),                                           # (was off: empty is the default now)
    (NVIDIA, {"RX_VK_COMPUTE_SERIAL": "true"}, {"compute_builds_serial": 1}, ["compute_builds_serial"], []),   # (was off: not a 1)
    (NVIDIA, {"RX_VK_COMPUTE_SERIAL": "2"}, {"compute_builds_serial": 1}, ["compute_builds_serial"], []),      # (was off)
    (AMD_WIN, {"RX_VK_MESH_STRICT_FMUL": "0"}, {"mesh_fallback_strict_fmul": 0}, ["mesh_fallback_strict_fmul"], []),
    (NVIDIA, {"RX_VK_MESH_STRICT_FMUL": "1"}, {"mesh_fallback_strict_fmul": 1}, ["mesh_fallback_strict_fmul"], []),
    (AMD_WIN, {"RX_VK_BC_COPY_PER_LAYER": "0"}, {"bc_copy_per_layer": 0}, ["bc_copy_per_layer"], []),
    (AMD_WIN, {"RX_VK_BC_COPY_PER_LAYER": ""}, {}, [], []),
    (NVIDIA, {"RX_VK_BC_COPY_PER_LAYER": "1"}, {"bc_copy_per_layer": 1}, ["bc_copy_per_layer"], []),
    (UHD, {"RX_VK_COMPUTE_SERIAL": "0"}, {"compute_builds_serial": 0}, ["compute_builds_serial"], []),
    (UHD, {"RX_VK_COMPUTE_SERIAL": "off"}, {"compute_builds_serial": 0}, ["compute_builds_serial"], []),
    (UHD, {"RX_VK_COMPUTE_SERIAL": ""}, {}, [], []),                                   # (was off)
    (RAPHAEL, {"RX_VK_ASTC_CPU": "0"}, {"astc_on_cpu": 0, "texture_budget": 4096}, ["astc_on_cpu"], []),
    (RAPHAEL, {"RX_VK_ASTC_CPU": ""}, {}, [], []),                                     # (was off)
    (UHD, {"RX_VK_ASTC_CPU": "1"}, {"astc_on_cpu": 1}, ["astc_on_cpu"], []),           # (<= 16 GB RAM: budget cap stays 2560 MB)
    (NVIDIA, {"RX_VK_ASTC_CPU": "yes"}, {"astc_on_cpu": 1}, ["astc_on_cpu"], []),
    (NVIDIA, {"RX_VK_PCACHE_FETCH_LOCK": "1", "RX_VK_SPLIT_BEFORE_DRAW": "1", "RX_VK_UB_SNAPSHOT": "1"},
     {"pcache_fetch_lock": 1, "split_before_draw": 1, "ub_snapshot": 1}, ["pcache_fetch_lock", "split_before_draw", "ub_snapshot"], []),
    (B580, {"RX_VK_PCACHE_FETCH_LOCK": "0", "RX_VK_SPLIT_BEFORE_DRAW": "no", "RX_VK_UB_SNAPSHOT": "off", "RX_VK_OCC_COPY": "0"},
     {"pcache_fetch_lock": 0, "split_before_draw": 0, "ub_snapshot": 0, "occ_copy": 0},
     ["pcache_fetch_lock", "split_before_draw", "ub_snapshot", "occ_copy"], []),
    (B580, {"RX_VK_PCACHE_FETCH_LOCK": "", "RX_VK_SPLIT_BEFORE_DRAW": "", "RX_VK_UB_SNAPSHOT": "", "RX_VK_OCC_COPY": ""},
     {}, [], []),                                                                      # (each was off)
    (NVIDIA, {"RX_VK_OCC_COPY": "on"}, {"occ_copy": 1}, ["occ_copy"], []),             # (was off: read as a number)
    (RAPHAEL, {"RX_VK_OCC_COPY": "0"}, {"occ_copy": 0}, ["occ_copy"], []),
    # the mirrors, and what follows them: the import chunk, and on Intel the uniform updates at each part's end
    (NVIDIA, {"RX_VK_MIRROR": "0"}, {"mirrors": 0, "import_chunk_mb": 16}, ["mirrors"], []),
    (NVIDIA, {"RX_VK_MIRROR": "off"}, {"mirrors": 0, "import_chunk_mb": 16}, ["mirrors"], []),
    (B580, {"RX_VK_MIRROR": "0"}, {"mirrors": 0, "import_chunk_mb": 16, "split_ub_flush": 0}, ["mirrors"], []),
    (B580, {"RX_VK_MIRROR": "0", "RX_VK_SPLIT_UBFLUSH": "1"}, {"mirrors": 0, "import_chunk_mb": 16}, ["mirrors", "split_ub_flush"], []),
    (UHD, {"RX_VK_MIRROR": "yes"}, {"mirrors": 1, "import_chunk_mb": 4, "split_ub_flush": 1}, ["mirrors"], []),   # (was off)
    (UHD, {"RX_VK_SPLIT_UBFLUSH": "1"}, {"split_ub_flush": 1}, ["split_ub_flush"], []),
    (NVIDIA, {"RX_VK_SPLIT_UBFLUSH": "0"}, {"split_ub_flush": 0}, ["split_ub_flush"], []),
    (NVIDIA, {"RX_VK_SPLIT_UBFLUSH": ""}, {}, [], []),                                 # (was off)
    # the two that only turn their rule off
    (AMD_WIN, {"RX_VK_FSR4_TUNE": "0"}, {"fsr4_tune": 0}, ["fsr4_tune"], []),
    (AMD_WIN, {"RX_VK_FSR4_TUNE": "off"}, {"fsr4_tune": 0}, ["fsr4_tune"], []),
    (AMD_WIN, {"RX_VK_FSR4_TUNE": "1"}, {}, [], []),
    (AMD_WIN, {"RX_VK_FSR4_TUNE": "abc"}, {}, [], []),                                 # (was off: a word that is no number)
    (NVIDIA, {"RX_VK_FSR4_TUNE": "1"}, {}, [], []),                                    # (never on off AMD's GPUs)
    (AMD_WIN, {"RX_VK_NO_INVARIANT": "1"}, {"position_invariant": 0}, ["position_invariant"], []),
    (AMD_WIN, {"RX_VK_NO_INVARIANT": "0"}, {}, [], []),
    (AMD_WIN, {"RX_VK_NO_INVARIANT": ""}, {}, [], []),
    # numbers: digits only, in the range; anything else ignored and named
    (NVIDIA, {"RX_VK_WORKERS": "1"}, {"workers": 1}, ["workers"], []),
    (NVIDIA, {"RX_VK_WORKERS": "16"}, {"workers": 16}, ["workers"], []),
    (NVIDIA, {"RX_VK_WORKERS": "0"}, {}, [], ["RX_VK_WORKERS"]),                       # (was 1)
    (NVIDIA, {"RX_VK_WORKERS": "17"}, {}, [], ["RX_VK_WORKERS"]),                      # (was 16)
    (NVIDIA, {"RX_VK_WORKERS": "four"}, {}, [], ["RX_VK_WORKERS"]),                    # (was 1)
    (NVIDIA, {"RX_VK_WORKERS": " 4"}, {}, [], ["RX_VK_WORKERS"]),                      # (was 4)
    (NVIDIA, {"RX_VK_WORKERS": ""}, {}, [], []),                                       # (was 1)
    (NVIDIA, {"RX_VK_EQ_BIAS": "32"}, {"equal_depth_bias": "32"}, ["equal_depth_bias"], []),
    (UHD, {"RX_VK_EQ_BIAS": "0"}, {"equal_depth_bias": "0"}, ["equal_depth_bias"], []),
    (UHD, {"RX_VK_EQ_BIAS": "12.5"}, {"equal_depth_bias": "12.5"}, ["equal_depth_bias"], []),
    (UHD, {"RX_VK_EQ_BIAS": "-1"}, {}, [], ["RX_VK_EQ_BIAS"]),                         # (was off, as any value under 0)
    (UHD, {"RX_VK_EQ_BIAS": "abc"}, {}, [], ["RX_VK_EQ_BIAS"]),                        # (was off: read as 0)
    (UHD, {"RX_VK_EQ_BIAS": "1e2"}, {}, [], ["RX_VK_EQ_BIAS"]),                        # (was 100)
    (UHD, {"RX_VK_EQ_BIAS": "64."}, {}, [], ["RX_VK_EQ_BIAS"]),
    (UHD, {"RX_VK_EQ_BIAS": ""}, {}, [], []),                                          # (was off)
    (SMALL, {"RX_VK_CAP_SPARE_MB": "0"}, {"cap_spare_max": -1}, ["cap_spare_max"], []),   # (0: no limit, as before)
    (SMALL, {"RX_VK_CAP_SPARE_MB": "64"}, {"cap_spare_max": 64}, ["cap_spare_max"], []),
    (SMALL, {"RX_VK_CAP_SPARE_MB": ""}, {}, [], []),                                   # (was no limit)
    (SMALL, {"RX_VK_CAP_SPARE_MB": "-5"}, {}, [], ["RX_VK_CAP_SPARE_MB"]),             # (was a huge limit)
    (SMALL, {"RX_VK_CAP_SPARE_MB": "lots"}, {}, [], ["RX_VK_CAP_SPARE_MB"]),           # (was no limit)
    (SMALL, {"RX_VK_STAGING_KEEP_MB": "256"}, {"staging_keep": 256}, ["staging_keep"], []),
    (SMALL, {"RX_VK_STAGING_KEEP_MB": "0"}, {"staging_keep": 0}, ["staging_keep"], []),
    (SMALL, {"RX_VK_STAGING_KEEP_MB": "-1"}, {}, [], ["RX_VK_STAGING_KEEP_MB"]),       # (was a huge size: every arena kept)
    (SMALL, {"RX_VK_STAGING_KEEP_MB": "abc"}, {}, [], ["RX_VK_STAGING_KEEP_MB"]),      # (was 0)
    # a switch of a rule that still decides at its site is not read here
    (NVIDIA, {"RX_VK_MESH_OPT": "1", "RX_VK_NATIVE_FADD": "0", "RX_VK_VRAM_MB": "1"}, {}, [], []),
]

# The rules at their edges: the fixture's expression, and its value (sizes in whole MB; -1: no limit)
MB = "* policy_rule::kMB"
RULES = [
    ("mbs(policy_rule::cap_spare_max(0))", -1),                       # (allowance not known: no limit)
    ("mbs(policy_rule::cap_spare_max(6656 %s))" % MB, 32),
    ("mbs(policy_rule::cap_spare_max(6656 %s + 1))" % MB, -1),
    ("mbs(policy_rule::staging_keep(0))", 256),
    ("mbs(policy_rule::staging_keep(6656 %s))" % MB, 128),
    ("mbs(policy_rule::staging_keep(6656 %s + 1))" % MB, 256),
    ("(long long)policy_rule::small_vram_textures(4608 %s, false, 0)" % MB, 1),
    ("(long long)policy_rule::small_vram_textures(4608 %s + 1, false, 0)" % MB, 0),
    ("(long long)policy_rule::small_vram_textures(0, false, 0)", 0),
    ("(long long)policy_rule::small_vram_textures(0, true, 16384 %s)" % MB, 1),
    ("(long long)policy_rule::small_vram_textures(0, true, 16384 %s + 1)" % MB, 0),
    ("(long long)policy_rule::small_vram_textures(0, true, 0)", 0),   # (RAM not known: not small by it)
    ("mbs(policy_rule::wb_shadow_cap(0, false))", 512),
    ("mbs(policy_rule::wb_shadow_cap(8000 %s, true))" % MB, 512),
    ("mbs(policy_rule::wb_shadow_cap(2048 %s, true))" % MB, 256),
    ("mbs(policy_rule::wb_shadow_cap(16000 %s, false))" % MB, 2000),
    ("mbs(policy_rule::mirror_budget(0))", 3072),
    ("mbs(policy_rule::mirror_budget(6656 %s))" % MB, 1700),
    ("mbs(policy_rule::mirror_budget(6660 %s))" % MB, 1665),          # (past 6.5 GB: mirror_budget.h's quarter)
    ("mbs(policy_rule::integrated_texture_budget(3000 %s, 0))" % MB, 1024),   # (RAM not known: as 8 GB, the 1 GB floor)
    ("mbs(policy_rule::integrated_texture_budget(512 %s, 32768 %s))" % (MB, MB), 512),   # (never more than the share)
    ("mbs(policy_rule::integrated_texture_budget(6144 %s, 65536 %s))" % (MB, MB), 6144),
    ("mbs(call_integrated_texture_budget(8192 %s, 16384 %s, false))" % (MB, MB), 2560),
    ("mbs(call_integrated_texture_budget(8192 %s, 16384 %s, true))" % (MB, MB), 2560),
    ("mbs(call_integrated_texture_budget(8192 %s, 24576 %s, false))" % (MB, MB), 3584),
    ("mbs(call_integrated_texture_budget(8192 %s, 24576 %s, true))" % (MB, MB), 6144),
    ("mbs(call_integrated_texture_budget(8192 %s, 26283 %s, false))" % (MB, MB), 4096),
    ("mbs(call_integrated_texture_budget(8192 %s, 26283 %s, true))" % (MB, MB), 8192),
    ("mbs(policy_rule::vma_block(8192 %s))" % MB, 32),
    ("mbs(policy_rule::vma_block(8192 %s + 1))" % MB, 64),
    ("mbs(policy_rule::target_block(8192 %s))" % MB, 16),
    ("mbs(policy_rule::target_block(8192 %s + 1))" % MB, 32),
    # (the workers' steps: 2 up to 8 threads, 4 at 15 and 6 at 16, which the study doubts)
    ("(long long)policy_rule::workers(0)", 2),
    ("(long long)policy_rule::workers(8)", 2),
    ("(long long)policy_rule::workers(9)", 3),
    ("(long long)policy_rule::workers(15)", 4),
    ("(long long)policy_rule::workers(16)", 6),
    ("(long long)policy_rule::capture_helpers(11)", 2),
    ("(long long)policy_rule::capture_helpers(12)", 4),
    ("(long long)policy_rule::texture_threads(0)", 1),
    ("(long long)policy_rule::texture_threads(64)", 8),
    ("(long long)policy_rule::texture_gather_threads(64)", 6),
    ("(long long)policy_rule::program_workers_in_play(0, 1)", 1),     # (never more than the process may use)
    ("(long long)policy_rule::program_workers_in_play(24, 24)", 4),
    ("(long long)policy_rule::program_workers_in_play(4, 4)", 2),
    # (the float add's list: NVIDIA's driver; Intel's Windows driver by family, from the driver that was run)
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_NVIDIA_PROPRIETARY, 0x2204, 0)", 1),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_MESA_NVK, 0x2204, 0)", 0),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0x7D55, 101u << 14 | 9175u)", 1),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0x7D55, 101u << 14 | 9174u)", 0),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0xE20B, 101u << 14 | 9153u)", 0),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0xB080, 101u << 14 | 9150u)", 1),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0xB080, 101u << 14 | 9149u)", 0),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0x64A0, 101u << 14 | 8991u)", 1),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0x64A0, 101u << 14 | 8990u)", 0),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0x56A1, 101u << 14 | 9999u)", 0),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, 0x7D55, 0xFFFFFFFFu)", 0),
    ("(long long)policy_rule::native_fadd_listed(VK_DRIVER_ID_AMD_PROPRIETARY, 0x7550, 0xFFFFFFFFu)", 0),
    # (equal_depth_bias: 0 on discrete Battlemage 0xE2xx and Mesa ANV; 64 on older/integrated Intel Windows families)
    ("(long long)policy_rule::equal_depth_bias(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0xE20Cu)", 0),
    ("(long long)policy_rule::equal_depth_bias(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0xE20Bu)", 0),
    ("(long long)policy_rule::equal_depth_bias(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0x56A1u)", 64),
    ("(long long)policy_rule::equal_depth_bias(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0x9A78u)", 64),
    ("(long long)policy_rule::equal_depth_bias(VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS, 0xB080u)", 64),
    ("(long long)policy_rule::equal_depth_bias(VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, 0x9A49u)", 0),
    ("(long long)policy_rule::equal_depth_bias(VK_DRIVER_ID_NVIDIA_PROPRIETARY, 0x2204u)", 0),
    # (DeviceFacts::arch() / amd_win() end-to-end, including AMD proprietary Windows GCN without radv_gcn)
    ("(long long)[] { DeviceFacts f{}; f.vendor_id = 0x1002; f.device_id = 0x67B0; f.driver_id = VK_DRIVER_ID_AMD_PROPRIETARY; f.windows = true; return f.amd_win() && !f.radv_gcn() && f.arch() == GpuArch::AmdGcn; }()", 1),
    ("(long long)[] { DeviceFacts f{}; f.vendor_id = 0x1002; f.device_id = 0x73DF; f.driver_id = VK_DRIVER_ID_AMD_PROPRIETARY; f.windows = true; return f.amd_win() && f.rdna2() && !f.rdna3() && !f.rdna4(); }()", 1),
    ("(long long)[] { DeviceFacts f{}; f.vendor_id = 0x1002; f.device_id = 0x73DF; f.driver_id = VK_DRIVER_ID_AMD_PROPRIETARY; f.windows = false; return f.amd_win(); }()", 0),
    ("(long long)[] { DeviceFacts f{}; f.vendor_id = 0x1002; f.device_id = 0x731F; f.driver_id = VK_DRIVER_ID_AMD_PROPRIETARY; f.windows = true; return f.rdna1(); }()", 1),
    ("(long long)[] { DeviceFacts f{}; f.vendor_id = 0x1002; f.device_id = 0x1114; f.driver_id = VK_DRIVER_ID_AMD_PROPRIETARY; f.windows = true; return f.rdna3(); }()", 1),
    ("(long long)[] { DeviceFacts f{}; f.vendor_id = 0x1002; f.device_id = 0x130F; f.driver_id = VK_DRIVER_ID_MESA_RADV; f.storage_io16 = false; return f.radv_gcn() && f.arch() == GpuArch::AmdGcn; }()", 1),
]

FIXTURE = r"""
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include "device_policy.h"
using namespace mkwvk;
struct D {
    const char* name;
    uint32_t vendor, device;
    VkDriverId driver;
    uint32_t version;
    VkPhysicalDeviceType type;
    const char* info;
    const char* gpu;
    int windows, deck;
    uint64_t vram_mb, allowed_mb, ram_mb;
    unsigned cpus, usable, cores;
    int mesh, bda, fadd, coop, checkpoints, io16, dynamic;
    uint32_t lanes, lanes_max, sg32;
};
static DeviceFacts facts_of(const D& d) {
    DeviceFacts f;
    f.vendor_id = d.vendor, f.device_id = d.device, f.driver_version = d.version;
    f.driver_id = d.driver, f.device_type = d.type, f.driver_info = d.info, f.device_name = d.gpu;
    f.windows = d.windows != 0, f.steam_deck_env = d.deck != 0;
    f.vram_heap = d.vram_mb << 20, f.allowed_at_start = d.allowed_mb << 20, f.ram = d.ram_mb << 20;
    f.cpus_all = d.cpus, f.cpus_usable = d.usable, f.cores_usable = d.cores;
    f.mesh_shader = d.mesh != 0, f.mesh_queries = d.mesh != 0, f.buffer_device_address = d.bda != 0;
    f.float32_atomic_add = d.fadd != 0, f.coop_matrix_ext = d.coop != 0, f.nv_checkpoints_ext = d.checkpoints != 0;
    f.storage_io16 = d.io16 != 0, f.storage_buffer_array_dynamic_indexing = d.dynamic != 0, f.robust_image_access = true;
    f.subgroup_size = d.lanes, f.subgroup_max = d.lanes_max, f.sg32_stages = d.sg32;
    return f;
}
static long long mbs(uint64_t bytes) { return bytes == ~0ull ? -1 : (long long)(bytes >> 20); }
template <class B = bool>
static uint64_t call_integrated_texture_budget(uint64_t share, uint64_t ram, B astc_on_cpu) {
    if constexpr (requires(B cpu) { policy_rule::integrated_texture_budget(share, ram, cpu); }) {
        return policy_rule::integrated_texture_budget(share, ram, astc_on_cpu);
    } else {
        (void)astc_on_cpu;
        return policy_rule::integrated_texture_budget(share, ram);
    }
}
static void log_block(int k, const DevicePolicy& p, const DeviceFacts& f) {
    device_policy_log(p, f, [&](const char* line) { std::printf("LOG %d %zu %s\n", k, std::strlen(line), line); });
}
static std::map<std::string, std::string> env;   // (a case's environment, in place of the process's)
static const char* get(const char* name) {
    auto it = env.find(name);
    return it == env.end() ? nullptr : it->second.c_str();
}
// A PC's policy with the case's environment over it: the fields a switch can set, the bits of those one did, and the notes
static void env_case(int i, const D& d) {
    const DeviceFacts f = facts_of(d);
    DevicePolicy p = device_policy(f);
    device_policy_env(p, f, get);
    std::printf("ENV %d astc_on_cpu=%d batch_submits=%d compute_builds_serial=%d pcache_fetch_lock=%d split_before_draw=%d "
                "ub_snapshot=%d split_ub_flush=%d occ_copy=%d mirrors=%d fsr4_tune=%d position_invariant=%d equal_depth_bias=%g "
                "cap_spare_max=%lld staging_keep=%lld workers=%u mesh_fallback_strict_fmul=%d bc_copy_per_layer=%d import_chunk_mb=%u texture_budget=%lld set=%u notes=%s\n",
                i, (int)p.astc_on_cpu, (int)p.batch_submits, (int)p.compute_builds_serial, (int)p.pcache_fetch_lock,
                (int)p.split_before_draw, (int)p.ub_snapshot, (int)p.split_ub_flush, (int)p.occ_copy, (int)p.mirrors, (int)p.fsr4_tune,
                (int)p.position_invariant, (double)p.equal_depth_bias, mbs(p.cap_spare_max), mbs(p.staging_keep), p.workers,
                (int)p.mesh_fallback_strict_fmul, (int)p.bc_copy_per_layer,
                (unsigned)p.import_chunk_mb, mbs(p.texture_budget), (unsigned)p.set_by_switch, p.switch_notes);
    if (i == 0 || p.set_by_switch) log_block(1000 + i, p, f);   // (the block with switches set: still six lines under their length)
}
int main() {
    const D devices[] = {
@@DEVICES@@
    };
    int k = 0;
    for (const D& d : devices) {
        const DeviceFacts f = facts_of(d);
        const DevicePolicy p = device_policy(f);
        std::printf("POLICY %d", k);
@@PRINTS@@
        std::printf("\n");
        log_block(k, p, f);
        k++;
    }
    {   // (a driver that says a lot about itself: the lines stay under their length)
        DeviceFacts f = facts_of(devices[0]);
        const std::string wordy(300, 'x');
        f.driver_info = wordy.c_str();
        log_block(-1, device_policy(f), f);
    }
    {   // (a PC nothing is known of: the defaults do not divide by a count or a size that is 0)
        const DeviceFacts none{};
        const DevicePolicy p = device_policy(none);
        std::printf("NONE %d %d %u %lld %lld\n", (int)p.mirrors, p.auto_tier, p.workers, mbs(p.texture_budget), mbs(p.mirror_budget));
    }
    {   // (every switch set at once, with long values: the notes are cut, not overrun)
        env.clear();
        const std::string wordy(200, '9');
        for (const char* name : {"RX_VK_MIRROR", "RX_VK_SPLIT_UBFLUSH", "RX_VK_UB_SNAPSHOT", "RX_VK_ASTC_CPU", "RX_VK_BATCH",
                                 "RX_VK_COMPUTE_SERIAL", "RX_VK_PCACHE_FETCH_LOCK", "RX_VK_SPLIT_BEFORE_DRAW", "RX_VK_OCC_COPY",
                                 "RX_VK_FSR4_TUNE", "RX_VK_NO_INVARIANT", "RX_VK_EQ_BIAS", "RX_VK_CAP_SPARE_MB",
                                 "RX_VK_STAGING_KEEP_MB", "RX_VK_WORKERS", "RX_VK_MESH_STRICT_FMUL"})
            env[name] = wordy;
        const DeviceFacts f = facts_of(devices[0]);
        DevicePolicy p = device_policy(f);
        device_policy_env(p, f, get);
        std::printf("NOTES %zu\n", std::strlen(p.switch_notes));
        log_block(-2, p, f);
    }
@@ENV@@
@@RULES@@
    {   // (the two parsers alone)
        uint64_t n = 7;
        double v = 7;
        std::string got;
        const auto say = [&](bool ok) { got += ok ? '1' : '0'; };
        say(policy_rule::parse_number("12", 1, 16, n));
        say(n == 12);
        say(policy_rule::parse_number("16", 1, 16, n));
        say(policy_rule::parse_number("17", 1, 16, n));
        say(policy_rule::parse_number("0", 1, 16, n));
        say(policy_rule::parse_number("", 0, 16, n));
        say(policy_rule::parse_number("1 ", 0, 16, n));
        say(policy_rule::parse_number("+1", 0, 16, n));
        say(policy_rule::parse_number("99999999999999999999999", 0, ~0ull, n));   // (past 64 bits: not a wrapped number)
        say(policy_rule::parse_number("18446744073709551615", 0, ~0ull, n));
        got += ' ';
        say(policy_rule::parse_decimal("0.5", 10, v));
        say(v == 0.5);
        say(policy_rule::parse_decimal("10", 10, v));
        say(policy_rule::parse_decimal("10.5", 10, v));
        say(policy_rule::parse_decimal(".5", 10, v));
        say(policy_rule::parse_decimal("1.2.3", 10, v));
        say(policy_rule::parse_decimal("-1", 10, v));
        std::printf("PARSE %s\n", got.c_str());
    }
    return 0;
}
"""


def check_source():
    main = (VK / "vk_main.cpp").read_text(encoding="utf-8")
    create = block(main, "bool create_device() {")
    for text, why in (
        ("DeviceFacts& pf = dev.facts;", "the facts are not filled in dev.facts"),
        ("dev.policy = device_policy(pf);", "the policy is not made from the facts"),
        ("pf.driver_info = dev.driver.driverInfo;", "the facts' driverInfo does not point into the device's own copy"),
    ):
        if text not in create:
            raise SystemExit("FAIL: create_device: %s" % why)
    if "pf.cores_usable = usable_cores();" not in create:
        raise SystemExit("FAIL: create_device: the facts' cores are not usable_cores (Threaded rendering's Auto counts by "
                         "them)")
    if not create.index("pf.sg32_stages = dev.sg32_stages;") < create.index("dev.policy = device_policy(pf);") \
            < create.index("device_policy_env(dev.policy, pf,") < create.index("vkCreateDevice(dev.physical, &ci"):
        raise SystemExit("FAIL: create_device: the policy is not made once every fact is filled, then the switches read, "
                         "before the device")
    # (corrected: Compute for mesh programs by the driver the deadlock is in, asked before the rule; tests/test_mesh_mode_switch.py)
    if "if (mesh_mode_used == 0 && fadd_driver.driverID == VK_DRIVER_ID_MESA_RADV) mesh_mode_used = 2;" not in create:
        raise SystemExit("FAIL: create_device: Automatic's Compute is not RADV's, as device_policy.h's mesh_mode has it")
    if "device_policy_log(dev.policy, dev.facts," not in main:
        raise SystemExit("FAIL: vk_main.cpp does not log the policy block at the start")
    if main.index("device_policy_log(dev.policy, dev.facts,") < main.index('logf("vk: create_device failed");'):
        raise SystemExit("FAIL: the policy block is logged before the device is made")
    # (the sites that moved read the policy: the rule and the switch are device_policy.h's, and no site reads its switch again)
    sources = {}
    for path, marker, needs in (
        ("vk_texture.cpp", "bool astc_on_cpu() {", ("const bool v = dev.policy.astc_on_cpu;",)),
        # (Hookshot's staging_keep() holds the staging rule, which texture_frame and the arenas' spares ask)
        ("vk_texture.cpp", "uint64_t staging_keep() {",
         ("if (dev.policy.set(Sw::staging_keep)) return dev.policy.staging_keep;",
          "return policy_rule::staging_keep(vram_usage(&used, &allowed) ? allowed : 0);")),
        ("vk_texture.cpp", "void texture_frame(uint64_t completed)",
         ("const uint64_t keep = staging_keep();",
          "policy_rule::squeeze_damped(allowed) ? (over / 4) : over;",
          "texture_rule::integrated_bytes(a, device_vram_bytes(), astc_on_cpu())",
          "own = policy_rule::integrated_texture_budget(driver_budget, system_ram, astc_on_cpu());")),
        ("vk_replay.cpp", "bool batch_on() {", ("const bool v = dev.policy.batch_submits;",)),
        ("vk_replay.cpp", "bool occ_copy_wanted() {", ("if (par_threads() != 0) return false;", "return dev.policy.occ_copy;")),
        ("vk_replay.cpp", "bool part_ends_before_draw() {", ("return dev.policy.split_before_draw;",)),
        ("vk_replay.cpp", "void split_part(VkCommandBuffer& cmd) {", ("const bool part_flush = dev.policy.split_ub_flush;",)),
        ("vk_memory.cpp", "uint64_t cap_spare_max() {",
         ("if (dev.policy.set(Sw::cap_spare_max)) return dev.policy.cap_spare_max;",
          "policy_rule::cap_spare_max(vram_usage(&used, &allowed) ? allowed : 0)")),
        ("vk_memory.cpp", "bool mirror_on() {", ("const int v = dev.policy.mirrors;",)),
        ("vk_draw.cpp", "bool pcache_fetch_lock_on() {", ("static const bool on = dev.policy.pcache_fetch_lock;",)),
        ("vk_draw.cpp", "bool serial_builds_on() {", ("static const bool on = dev.policy.compute_builds_serial;",)),
        ("vk_draw.cpp", "bool ub_snapshot_wanted() {", ("return dev.policy.ub_snapshot;",)),
        ("vk_draw.cpp", "bool prepare_run(", ("std::min(dev.policy.program_workers_in_play, cpus)",)),
        ("vk_coopmat.cpp", "void coopmat_init() {", ("Shader::Backend::SPIRV::position_invariant = dev.policy.position_invariant;",)),
        ("vk_jobs.cpp", "void start() {", ("const unsigned n = dev.policy.workers;",)),
    ):
        text = sources.setdefault(path, (VK / path).read_text(encoding="utf-8"))
        body = block(text, marker)
        for need in needs:
            if need not in body:
                raise SystemExit("FAIL: %s: %s no longer has %s" % (path, marker.split("(")[0].split()[-1], need))
    for path, need in (("vk_fsr4.cpp", "static const bool v = dev.policy.fsr4_tune;"),
                       ("vk_draw.cpp", "static const float eq_bias = dev.policy.equal_depth_bias;"),
                       ("vk_draw.cpp", "q.set_serial(driver_faults_mode() == 2 || dev.policy.fault_gate_serial);"),
                       # (the Intel families' list written once: native_fadd_wanted asks policy_rule's)
                       ("vk_main.cpp", "return policy_rule::native_fadd_listed(driver, device, version);")):
        if need not in sources.setdefault(path, (VK / path).read_text(encoding="utf-8")):
            raise SystemExit("FAIL: %s no longer has %s" % (path, need))
    if "0x7D00u" in main or "0xE200u" in main:
        raise SystemExit("FAIL: vk_main.cpp has the float add's families again: device_policy.h's list is the one")
    if "1536ull << 20" in sources["vk_texture.cpp"]:
        raise SystemExit("FAIL: vk_texture.cpp has the integrated GPUs' RAM table again: device_policy.h's is the one")
    # (a switch device_policy_env reads is read nowhere else: one rule for what its values mean)
    header = (VK / "device_policy.h").read_text(encoding="utf-8")
    env = block(header, "template <class Get> void device_policy_env(")
    switches = sorted(set(re.findall(r'"(RX_VK_[A-Z0-9_]+)"', env)))
    if len(switches) != len(SW):
        raise SystemExit("FAIL: device_policy_env reads %d switches and the check knows %d fields: %s" % (len(switches), len(SW), switches))
    for path in sorted(VK.glob("*.cpp")) + sorted(VK.glob("*.h")):
        # (vk_shader.cpp names switches only to hash their text into the translator's stamp: it decides nothing by them)
        if path.name in ("device_policy.h", "vk_shader.cpp"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        code = "\n".join(line.split("//")[0] for line in text.splitlines())
        for name in switches:
            if '"%s"' % name in code:
                raise SystemExit("FAIL: %s reads %s itself: device_policy_env does, by the one rule" % (path.name, name))
    # (the header reads the environment through what it is handed only, and it is no file of the translator's stamp: an
    # edit to it must not have every player translate again. nx2/translator_files.py lists the stamp's files)
    if "getenv(" in header:
        raise SystemExit("FAIL: device_policy.h calls getenv: device_policy_env is handed its reader")
    internal = (VK / "vk_internal.h").read_text(encoding="utf-8")
    if "DeviceFacts facts{};" not in internal or "DevicePolicy policy{};" not in internal:
        raise SystemExit("FAIL: vk_internal.h's Device does not hold the facts and the policy")
    sys.path.insert(0, str(ROOT / "nx2"))
    import translator_files
    # (the translation's own rules are read from the policy by vk_shader.cpp, and by no other file of the stamp. A rule
    # changed in device_policy.h does not change HOOKSHOT_SHADER_STAMP, so it reaches the program cache only through the
    # values translator_stamp folds: each field vk_shader.cpp reads is folded there, itself or as the profile's value
    # made from it and its switch)
    folded = {
        "warp64": ("p.warp_size_potentially_larger_than_guest = dev.policy.warp64;",
                   "profile().warp_size_potentially_larger_than_guest ? 1u : 0u"),
        "frag_shuffle_fallback": ("p.has_broken_spirv_fragment_shuffle = dev.policy.frag_shuffle_fallback;",
                                  "profile().has_broken_spirv_fragment_shuffle ? 1u : 0u"),
        "gather_offset": ("p.need_gather_subpixel_offset = dev.policy.gather_offset;",
                          "profile().need_gather_subpixel_offset ? 64u : 0u"),
        "inline_deku_web_loads": ("vrt.inline_global_128_loads = dev.policy.inline_deku_web_loads && vsc->key == kDekuWebVertexCode;",
                                  "dev.policy.inline_deku_web_loads ? 1u : 0u"),
        "networks_left_out": ("(Shader::Backend::SPIRV::mma_software || dev.policy.networks_left_out)",
                              "dev.policy.networks_left_out ? 1u : 0u"),
    }
    for path in translator_files.files(str(VK)):
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        if Path(path).name == "device_policy.h":
            raise SystemExit("FAIL: device_policy.h is in the shader translation's stamp")
        if Path(path).name != "vk_shader.cpp":
            if "dev.policy" in text or "dev.facts" in text or '"device_policy.h"' in text:
                raise SystemExit("FAIL: %s is in the shader translation's stamp and uses the device policy: vk_shader.cpp "
                                 "reads it for the translation, and folds what it reads into the stamp" % path)
            continue
        if "dev.facts" in text:
            raise SystemExit("FAIL: vk_shader.cpp reads the device's facts: a rule by them is device_policy.h's")
        stamp = block(text, "uint64_t translator_stamp() {")
        read = sorted(set(re.findall(r"dev\.policy\.(\w+)", text)))
        if read != sorted(folded):
            raise SystemExit("FAIL: vk_shader.cpp reads the policy's %s and the check knows %s: a field the translation reads "
                             "is folded into translator_stamp and named here" % (", ".join(read), ", ".join(sorted(folded))))
        for name, (site, fold) in folded.items():
            if site not in text:
                raise SystemExit("FAIL: vk_shader.cpp no longer has %s" % site)
            if fold not in stamp:
                raise SystemExit("FAIL: translator_stamp does not fold %s (%s): programs made by one PC's rule would be kept "
                                 "for another's" % (name, fold))


def c_row(pc):
    (vendor, device, driver, version, kind, info, gpu, windows, deck, vram, allowed, ram, cpus, usable, cores, mesh, bda, fadd,
     coop, checkpoints, io16, dynamic, lanes, lanes_max, sg32) = pc["facts"]
    return ('        {"%s", 0x%X, 0x%X, VK_DRIVER_ID_%s, 0x%Xu, VK_PHYSICAL_DEVICE_TYPE_%s, "%s", "%s", %d, %d, %dull, %dull, %dull, '
            '%du, %du, %du, %d, %d, %d, %d, %d, %d, %d, %du, %du, 0x%Xu},'
            % (pc["name"], vendor, device, driver, version, kind, info, gpu, windows, deck, vram, allowed, ram, cpus, usable,
               cores, mesh, bda, fadd, coop, checkpoints, io16, dynamic, lanes, lanes_max, sg32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    for pc in PCS:   # (the table itself: every number given, every name on the list)
        if len(pc["num"]) != len(NUM_NAMES) or len(pc["facts"]) != 25:
            raise SystemExit("FAIL: the table's row for %s is short or long" % pc["name"])
        unknown = set(pc["on"].split()) - set(BOOLS)
        if unknown:
            raise SystemExit("FAIL: the table's row for %s names %s, no yes/no field" % (pc["name"], ", ".join(sorted(unknown))))
    prints = "\n".join('        std::printf(" %s=%%d", (int)p.%s);' % (n, n) for n in BOOLS) + "\n" + \
             "\n".join('        std::printf(" %s=%%lld", %s);' % (n, e) for n, e in NUMS)
    rules = "\n".join('    std::printf("RULE %d %%lld\\n", %s);' % (i, e) for i, (e, _) in enumerate(RULES))
    names = [pc["name"] for pc in PCS]
    cases = []
    for i, (pc, environment, changed, marked, ignored) in enumerate(ENV):
        unknown = (set(changed) - set(ENV_FIELDS)) | (set(marked) - set(SW))
        if unknown or any('"' in v or "\\" in v for v in environment.values()):
            raise SystemExit("FAIL: the switches' case %d names %s, or a value the fixture cannot hold" % (i, sorted(unknown)))
        sets = "".join(' env["%s"] = "%s";' % kv for kv in environment.items())
        cases.append("    env.clear();%s env_case(%d, devices[%d]);" % (sets, i, names.index(pc)))
    source = FIXTURE.replace("@@DEVICES@@", "\n".join(c_row(pc) for pc in PCS)).replace("@@PRINTS@@", prints) \
                    .replace("@@RULES@@", rules).replace("@@ENV@@", "\n".join(cases))
    with tempfile.TemporaryDirectory(prefix="device-policy-") as tmp:
        tmp = Path(tmp)
        cpp, exe = tmp / "fixture.cpp", tmp / "fixture"
        cpp.write_text(source, encoding="utf-8")
        r = subprocess.run([args.cxx, "-std=c++20", "-O1", "-Wall", "-I" + str(VK), "-I" + str(args.headers), str(cpp), "-o", str(exe)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode:
            print(r.stdout + r.stderr)
            raise SystemExit("the fixture did not build")
        out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30, check=True).stdout.splitlines()

    failures = []
    policies = {}
    logs = {}
    rule_values = {}
    none = None
    envs = {}
    notes_length = parsed = None
    for line in out:
        word, _, rest = line.partition(" ")
        if word == "ENV":
            head, _, notes = rest.partition(" notes=")
            i, _, fields = head.partition(" ")
            envs[int(i)] = (dict(f.split("=") for f in fields.split()), notes)
        elif word == "NOTES":
            notes_length = int(rest)
        elif word == "PARSE":
            parsed = rest
        elif word == "POLICY":
            k, _, fields = rest.partition(" ")
            policies[int(k)] = dict(f.split("=") for f in fields.split())
        elif word == "LOG":
            k, n, text = rest.split(" ", 2)
            logs.setdefault(int(k), []).append((int(n), text))
        elif word == "RULE":
            i, value = rest.split()
            rule_values[int(i)] = int(value)
        elif word == "NONE":
            none = rest.split()
        else:
            failures.append("a line the check does not know: %s" % line[:80])

    # (every field of the struct is one the fixture prints: a field added to DevicePolicy without a row here fails)
    header = (VK / "device_policy.h").read_text(encoding="utf-8")
    struct = block(header, "struct DevicePolicy {")
    for name in BOOLS + NUM_NAMES:
        if not any(("%s%s" % (name, end)) in struct for end in (" =", ",", ";")):
            failures.append("DevicePolicy has no field %s" % name)
    declared = 0
    for line in struct.splitlines():
        code = line.split("//")[0].strip()
        if "set_by_switch" in code or "switch_notes" in code:   # (which switches set a field: no decision of their own)
            continue
        if code.endswith(";") and "=" in code and "(" not in code.split("=")[0]:
            declared += code.count("=")
    if declared != len(BOOLS) + len(NUM_NAMES):
        failures.append("DevicePolicy declares %d fields and the check names %d: add the new one to BOOLS or NUMS, with its "
                        "value on every PC" % (declared, len(BOOLS) + len(NUM_NAMES)))

    for k, pc in enumerate(PCS):
        got = policies.get(k)
        if got is None:
            failures.append("%s: no policy printed" % pc["name"])
            continue
        on = set(pc["on"].split())
        for name in BOOLS:
            if got.get(name) != ("1" if name in on else "0"):
                failures.append("%s: %s is %s, expected %s" % (pc["name"], name, got.get(name), "on" if name in on else "off"))
        for name, want in zip(NUM_NAMES, pc["num"]):
            if got.get(name) != str(want):
                failures.append("%s: %s is %s, expected %d" % (pc["name"], name, got.get(name), want))
    # (the switches: every field a switch can set, on the PC's defaults but where the case says; which were set; what was ignored)
    for i, (pc_name, environment, changed, marked, ignored) in enumerate(ENV):
        what = "%s with %s" % (pc_name, environment or "nothing set")
        if i not in envs:
            failures.append("%s: nothing printed" % what)
            continue
        got, notes = envs[i]
        pc = PCS[names.index(pc_name)]
        on, nums = set(pc["on"].split()), dict(zip(NUM_NAMES, pc["num"]))
        for name in ENV_FIELDS:
            want = str(changed[name]) if name in changed else ("1" if name in on else "0") if name in BOOLS else str(nums[name])
            if got.get(name) != want:
                failures.append("%s: %s is %s, expected %s" % (what, name, got.get(name), want))
        mask = sum(1 << SW.index(name) for name in marked)
        if got.get("set") != str(mask):
            failures.append("%s: set by a switch %s, expected %d (%s)" % (what, got.get("set"), mask, ", ".join(marked) or "none"))
        for name in ignored:
            if "%s=%s ignored (" % (name, environment[name]) not in notes:
                failures.append("%s: the notes do not say %s was ignored: %s" % (what, name, notes))
        if notes.count(" ignored (") != len(ignored):
            failures.append("%s: the notes say more was ignored than was: %s" % (what, notes))
        for name in marked:   # (a switch that set a field is named with its value)
            if not any("%s=%s" % (switch, value) in notes for switch, value in environment.items()):
                failures.append("%s: the notes do not name the switch that set %s: %s" % (what, name, notes))
        if (i == 0 or marked) and len(logs.get(1000 + i, [])) != 6:
            failures.append("%s: its log block is not six lines" % what)
        if marked and "*" not in "".join(text for _, text in logs.get(1000 + i, [])[:5]):
            failures.append("%s: the log block marks no value as set by its switch" % what)
    if "*" in "".join(text for _, text in logs.get(1000, [])[:5]):
        failures.append("the log block marks a value as set by its switch with no switch set")
    if notes_length != 319:
        failures.append("every switch set with a long value: the notes are %s characters, expected 319 (cut at their room)" % notes_length)
    if parsed != "1110000001 1110000":
        failures.append("parse_number and parse_decimal: %s, expected 1110000001 1110000" % parsed)
    for k in list(range(len(PCS))) + [-1, -2] + [1000 + i for i in range(len(ENV)) if 1000 + i in logs]:
        lines = logs.get(k, [])
        if len(lines) != 6:
            failures.append("PC %d: %d log lines, expected 6" % (k, len(lines)))
        for n, text in lines:
            if not text.startswith("vk: policy: ") or n != len(text) or n >= 900:
                failures.append("PC %d: a log line of %d characters that is not a 'vk: policy: ' line under 900: %s" % (k, n, text[:60]))
    # (a driver's fix that is on says the version it was seen on, beside the driver's own: the Navi 22 card's fallback)
    amd = "".join(text for _, text in logs.get(names.index(AMD_WIN), []))
    if "on driver version 0x800161:" not in amd or "task_mesh_fallback=1(seen on 0x800161)" not in amd:
        failures.append("the log block does not say the driver's version and the one task_mesh_fallback was seen on: %s" % amd[:300])
    # (DeviceFacts::arch() / arch_name() end-to-end on the facts log line across representative PCs)
    for pc_name, want_arch in (
        (NVIDIA, "nvidia_ampere"),
        ("RTX 4070 Laptop (8 GB), NVIDIA's Windows driver", "nvidia_ada"),
        ("GTX 1650 (4 GB), NVK", "nvidia_turing"),
        (AMD_WIN, "amd_rdna2"),
        ("RX 9070 XT, AMD's Windows driver (LLPC)", "amd_rdna4"),
        ("RX 580 (8 GB), RADV on GCN", "amd_gcn"),
        ("RX Vega 64 (8 GB), RADV on GFX9", "amd_vega"),
        ("Steam Deck (LCD), RADV", "amd_rdna2"),
        ("RX 5700 (8 GB), AMD's Windows driver (RDNA 1 compiler)", "amd_rdna1"),
        ("Radeon 860M (0x1002:0x1114), AMD's Windows driver 26.10.1 (LLPC)", "amd_rdna3"),
        (UHD, "intel_gen12"),
        (B580, "intel_battlemage"),
        ("Arc A750, Intel's Windows driver 101.9175", "intel_alchemist"),
        ("llvmpipe", "unknown"),
    ):
        facts_line = logs.get(names.index(pc_name), [(0, "")])[0][1]
        if "(arch %s)" % want_arch not in facts_line:
            failures.append("%s: facts line missing (arch %s): %s" % (pc_name, want_arch, facts_line[:200]))
    for i, (expr, want) in enumerate(RULES):
        if rule_values.get(i) != want:
            failures.append("%s is %s, expected %d" % (expr, rule_values.get(i), want))
    # (nothing known: no mirrors, Minimum (no mesh shaders), 2 workers, 4 GB of textures, 3 GB of mirrors, as the rules say)
    if none != ["0", "1", "2", "4096", "3072"]:
        failures.append("a PC nothing is known of: %s, expected 0 1 2 4096 3072" % none)
    if failures:
        raise SystemExit("FAIL:\n  " + "\n  ".join(failures))
    check_source()
    print("ok: %d PCs, each of DevicePolicy's %d fields as the rules have it; %d environments through the switches' one rule; "
          "%d rule edges; the log block under 900 characters a line; the facts filled before the device and the moved sites "
          "reading the policy" % (len(PCS), len(BOOLS) + len(NUM_NAMES), len(ENV), len(RULES)))


if __name__ == "__main__":
    main()
