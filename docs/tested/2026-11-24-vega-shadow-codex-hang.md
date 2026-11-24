| Date | Build | OS / desktop | GPU / driver | CPU / RAM | Settings | Where | Result |
|---|---|---|---|---|---|---|---|
| 2026-11-24 | Local 4.0.2 host with the 4.0.6 Vega renderer backport | Windows 11 Pro for Workstations 26300 | Radeon Vega 1002:15E7 / AMD 26.5.2 | Ryzen 5 5625U / 16 GB | 1920x1080, headless Vulkan, frame generation enabled | Lakeside checkpoint, map ID HyruleField_Adult, area entry | Stopped presenting at frame 1901; watchdog found the renderer inside the AMD Vulkan driver under the actor motion shader module path (shadow baseline trace) |
