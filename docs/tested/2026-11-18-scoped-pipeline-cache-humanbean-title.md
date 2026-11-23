| Date | Build | OS / desktop | GPU / driver | CPU / RAM | Settings | Where | Result |
|---|---|---|---|---|---|---|---|
| 2026-11-18 | scoped memory policy on 430eef4f | CachyOS Linux, headless | GeForce RTX 3090, NVIDIA 620.70.06 | Ryzen 9 7900X, 32 GiB | 18 GiB hard process scope, warmed driver cache, application cache override unset | Headless startup to rendered title | Renderer reported 18,432 MiB hard capacity and 4,292 MiB of driver cache. It left the application cache unloaded. |
