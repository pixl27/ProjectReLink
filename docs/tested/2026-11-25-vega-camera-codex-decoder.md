| Date | Build | OS / desktop | GPU / driver | CPU / RAM | Settings | Where | Result |
|---|---|---|---|---|---|---|---|
| 2026-11-25 | Red Tunic 4.2.2 renderer, CPU ASTC decoder before and after partition and endpoint setup changes | Windows 11 build 26300 | Radeon Vega 1002:15e7, 26.5.2 | Ryzen 5 5625U, 16 GB | llvm-mingw, x86-64-v3, Zen 3 tuning | Game-free synthetic ASTC with irregular image edges, 1 to 4 partitions, single and dual planes, LDR and HDR endpoint modes, all 14 block sizes | Optimized serial and parallel decoders matched the baseline RGBA8 bytes on every fixture. |
