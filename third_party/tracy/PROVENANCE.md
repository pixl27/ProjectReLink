Tracy Profiler 0.15.0, from https://github.com/wolfpld/tracy (tag v0.15.0, commit
f2a4f2529e7e71844c9e434b877770a21ecf7587, archive sha256
3d57da54aa67d7875da1e8fecd0bd5957675cef73c27dd52bffb6ab8d290a5e6), unmodified. BSD 3-clause license: LICENSE.

Kept: public/ (the client: TracyClient.cpp, the C and C++ API headers in tracy/, client/, common/ and its libbacktrace)
and LICENSE. Left out: the viewer, the capture and export tools, the manual, the examples and the tests.

A developer's option, never in a player's build: nx2/build.py compiles public/TracyClient.cpp only when HOOKSHOT_TRACY=1
(docs/development/tracy.md), into the game's executable, with TRACY_ONLY_LOCALHOST and TRACY_NO_BROADCAST (it listens on
127.0.0.1 alone and announces nothing on the network), TRACY_ON_DEMAND, TRACY_NO_CRASH_HANDLER and
TRACY_NO_CODE_TRANSFER. The host and the Vulkan back end reach it through host/hos_tracy.h's macros, which are empty in
every other build. Tracy's protocol is locked to its version: captures are taken and read with the tools of this tag.

upstream: https://github.com/wolfpld/tracy/archive/refs/tags/v0.15.0.tar.gz
upstream-sha256: 3d57da54aa67d7875da1e8fecd0bd5957675cef73c27dd52bffb6ab8d290a5e6
upstream-root: tracy-0.15.0

tree-sha256: 91dcb8fdb25f141136da3cb446755370d954b8e771a582c428637422131328aa
