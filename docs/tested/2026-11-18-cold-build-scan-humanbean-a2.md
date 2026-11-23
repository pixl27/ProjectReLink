| Date | Build | OS / desktop | GPU / driver | CPU / RAM | Settings | Where | Result |
|---|---|---|---|---|---|---|---|
| 2026-11-18 | 430eef4f | CachyOS Linux 7.3.3, headless | RTX 3090, NVIDIA 620.70.06, CPU-only test | Ryzen 9 7900X, 32 GB | Fresh Python process, empty cost and size records, default chunk budget | Cold build scan a2 over 93 generated C files (1,564,790,227 bytes), input reads excluded | Chunk cuts 5.868 s, largest functions 5.630 s, combined 11.498 s. All four runs had identical chunk contents, costs and maximum function sizes. The queue reported CPU pressure. |
