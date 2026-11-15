#!/usr/bin/env bash
# Launch an existing Project ReLink build on BC-250 with the validated guards.
set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 RELINK_BUILD_DIR PROFILE_ENV RADV_ICD_JSON" >&2
    exit 2
fi

BC250_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BC250_BUILD_DIR=$1
BC250_PROFILE_ENV=$2
BC250_ICD_JSON=$3
BC250_GAME_EXE=${BC250_GAME_EXE:-$BC250_BUILD_DIR/relink}

for BC250_REQUIRED_PATH in \
    "$BC250_BUILD_DIR" "$BC250_PROFILE_ENV" "$BC250_ICD_JSON" "$BC250_GAME_EXE"; do
    if [[ ! -e $BC250_REQUIRED_PATH ]]; then
        echo "error: required path does not exist: $BC250_REQUIRED_PATH" >&2
        exit 1
    fi
done
if [[ ! -x $BC250_GAME_EXE ]]; then
    echo "error: game executable is not executable: $BC250_GAME_EXE" >&2
    exit 1
fi
if [[ $(uname -m) != x86_64 ]]; then
    echo "error: the validated BC-250 Project ReLink build requires Linux/x86_64" >&2
    exit 1
fi
if ! command -v systemd-run >/dev/null 2>&1; then
    echo "error: systemd-run is required for the bounded launch scope" >&2
    exit 1
fi

set -a
# This is executable shell input and must be a trusted, locally reviewed file.
# shellcheck disable=SC1090
source "$BC250_PROFILE_ENV"
set +a

# Remove rejected/diagnostic test state inherited from older profiles. The
# caller may select replay workers explicitly through BC250_REPLAY_THREADS.
unset \
    RX_VK_ASTC_STAGE RX_VK_GPU_UNSWIZZLE RX_VK_ASYNC \
    HOS_VCORE_WAITS HOS_PROFILE HOS_THREAD_TIMES HOS_PRESENT_NOWAIT \
    HOS_FRAMES_AHEAD HOS_PIN_RENDER RX_VK_CHUNK
if [[ ${BC250_KEEP_INPUT_SCRIPT:-0} != 1 ]]; then
    unset HOS_INPUT_SCRIPT HOS_INPUT_SCRIPT_BACKGROUND
fi

export RELINK_NO_UPDATE_CHECK=1
export RX_BACKEND=vulkan
export VK_DRIVER_FILES=$BC250_ICD_JSON
# An SSH-launched process does not inherit the graphical session environment.
# Keep SDL on the user's existing Wayland/X11 session; allowing its kmsdrm
# fallback can steal a new VT and leave the television on a blank console.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
export DISPLAY=${DISPLAY:-:0}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-wayland,x11}
export RADV_BC250_MESH_TASK_IDX_PASSES=${RADV_BC250_MESH_TASK_IDX_PASSES:-2}
export RADV_BC250_KEEP_IB_KB=${RADV_BC250_KEEP_IB_KB:-4096}
export HOS_FPS_LOG=${HOS_FPS_LOG:-1}
export HOS_FPS_TARGET=${HOS_FPS_TARGET:-60}
export HOS_FPS_MAX=${HOS_FPS_MAX:-60}
export HOS_MEMORY_MB=${HOS_MEMORY_MB:-9809}
export HOS_FULLSCREEN=${BC250_FULLSCREEN:-0}
export MANGOHUD=${MANGOHUD:-1}
export RX_VK_PARALLEL=${BC250_REPLAY_THREADS:-1}
export RX_VK_OCC_QUERIES=${BC250_OCC_QUERIES:-1}
export HOS_OUTPUT_RES=${HOS_OUTPUT_RES:-3}
export HOS_DYNRES=${HOS_DYNRES:-0}
export HOS_RES_SCALE=${HOS_RES_SCALE:-0.67}
export RX_VK_LEAN_PIPELINES=${RX_VK_LEAN_PIPELINES:-1}
export RX_VK_IMAGE_BLOCK_MB=${RX_VK_IMAGE_BLOCK_MB:-32}
export RX_VK_SPARE_MB=${RX_VK_SPARE_MB:-16}
export RX_VK_STAGING_KEEP_MB=${RX_VK_STAGING_KEEP_MB:-0}
export BC250_MIN_AVAILABLE_MIB=${BC250_MIN_AVAILABLE_MIB:-2048}

[[ $HOS_MEMORY_MB =~ ^[0-9]+$ ]] && (( HOS_MEMORY_MB >= 9809 )) || {
    echo "error: HOS_MEMORY_MB must be at least 9809 for this title" >&2
    exit 2
}
[[ $HOS_FPS_MAX =~ ^[0-9]+$ ]] || {
    echo "error: HOS_FPS_MAX must be an integer" >&2
    exit 2
}
if (( HOS_FPS_MAX > 60 )); then
    export HOS_FPS_UNLOCKED=${HOS_FPS_UNLOCKED:-1}
fi

BC250_LOG_DIR=${BC250_LOG_DIR:-$BC250_BUILD_DIR/logs}
BC250_LOG_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BC250_METRICS_LOG=$BC250_LOG_DIR/relink-bc250-$BC250_LOG_STAMP.metrics
BC250_GAME_LOG=$BC250_LOG_DIR/relink-bc250-$BC250_LOG_STAMP.log
BC250_RUNTIME_MAX_SEC=${BC250_RUNTIME_MAX_SEC:-7200}
BC250_MEMORY_HIGH_BYTES=${BC250_MEMORY_HIGH_BYTES:-7549747200}
BC250_MEMORY_MAX_BYTES=${BC250_MEMORY_MAX_BYTES:-8912896000}
BC250_MEMORY_SWAP_MAX_BYTES=${BC250_MEMORY_SWAP_MAX_BYTES:-2147483648}
BC250_SCOPE=relink-bc250-$BASHPID
BC250_LAUNCH_PID=0
BC250_RESTORE_VT=

[[ $BC250_RUNTIME_MAX_SEC =~ ^[0-9]+$ ]] || {
    echo "error: BC250_RUNTIME_MAX_SEC must be an integer number of seconds" >&2
    exit 2
}
(( BC250_RUNTIME_MAX_SEC >= 60 )) || {
    echo "error: BC250_RUNTIME_MAX_SEC must be at least 60 seconds" >&2
    exit 2
}
for BC250_LIMIT_NAME in \
    BC250_MEMORY_HIGH_BYTES BC250_MEMORY_MAX_BYTES BC250_MEMORY_SWAP_MAX_BYTES; do
    if [[ ! ${!BC250_LIMIT_NAME} =~ ^[0-9]+$ ]]; then
        echo "error: $BC250_LIMIT_NAME must be an integer number of bytes" >&2
        exit 2
    fi
done
if (( BC250_MEMORY_HIGH_BYTES > BC250_MEMORY_MAX_BYTES )); then
    echo "error: BC250_MEMORY_HIGH_BYTES must not exceed BC250_MEMORY_MAX_BYTES" >&2
    exit 2
fi
if [[ ! -S $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY ]]; then
    echo "error: Wayland socket is unavailable: $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" >&2
    exit 1
fi

# ReLink or a display hotplug can leave the kernel on a blank virtual terminal.
# Prefer the seat's Wayland VT and return to it on every normal/signal exit.
if [[ -r /sys/class/tty/tty0/active ]]; then
    BC250_ACTIVE_TTY=$(</sys/class/tty/tty0/active)
    [[ $BC250_ACTIVE_TTY =~ ^tty([0-9]+)$ ]] && BC250_RESTORE_VT=${BASH_REMATCH[1]}
fi
if command -v loginctl >/dev/null 2>&1; then
    while read -r BC250_SESSION_ID _ _ BC250_SESSION_SEAT _; do
        [[ $BC250_SESSION_SEAT == seat0 ]] || continue
        BC250_SESSION_TYPE=$(loginctl show-session "$BC250_SESSION_ID" -p Type --value 2>/dev/null || true)
        BC250_SESSION_VT=$(loginctl show-session "$BC250_SESSION_ID" -p VTNr --value 2>/dev/null || true)
        if [[ $BC250_SESSION_TYPE == wayland && $BC250_SESSION_VT =~ ^[1-9][0-9]*$ ]]; then
            BC250_RESTORE_VT=$BC250_SESSION_VT
            break
        fi
    done < <(loginctl list-sessions --no-legend 2>/dev/null || true)
fi
if [[ -n $BC250_RESTORE_VT ]]; then
    sudo -n chvt "$BC250_RESTORE_VT"
fi

cleanup_launcher() {
    local status=$?
    trap - EXIT TERM INT HUP
    if systemctl --user is-active --quiet "$BC250_SCOPE.scope" 2>/dev/null; then
        systemctl --user stop "$BC250_SCOPE.scope" 2>/dev/null || true
    fi
    if (( BC250_LAUNCH_PID > 0 )) && kill -0 "$BC250_LAUNCH_PID" 2>/dev/null; then
        kill -TERM "$BC250_LAUNCH_PID" 2>/dev/null || true
        wait "$BC250_LAUNCH_PID" 2>/dev/null || true
    fi
    BC250_LAUNCH_PID=0
    if [[ -n $BC250_RESTORE_VT ]]; then
        sudo -n chvt "$BC250_RESTORE_VT" 2>/dev/null || true
    fi
    return "$status"
}

trap cleanup_launcher EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

mkdir -p "$BC250_LOG_DIR"

echo "build:         $BC250_BUILD_DIR"
echo "executable:    $BC250_GAME_EXE"
echo "profile:       $BC250_PROFILE_ENV"
echo "ICD:           $VK_DRIVER_FILES"
echo "replay:        $RX_VK_PARALLEL thread(s; 0/1 is serial)"
echo "resolution:    output $HOS_OUTPUT_RES, scale $HOS_RES_SCALE, dynamic $HOS_DYNRES"
echo "MangoHud:      $MANGOHUD"
echo "guest memory:  ${HOS_MEMORY_MB} MiB"
echo "metrics:       $BC250_METRICS_LOG"
echo "game log:      $BC250_GAME_LOG"
echo "time limit:    ${BC250_RUNTIME_MAX_SEC}s"
echo "memory:        high $BC250_MEMORY_HIGH_BYTES, max $BC250_MEMORY_MAX_BYTES, swap $BC250_MEMORY_SWAP_MAX_BYTES"

cd -- "$BC250_BUILD_DIR"
"$BC250_SCRIPT_DIR/with-bazzite-performance.sh" \
    systemd-run --user --scope --collect --quiet --unit="$BC250_SCOPE" \
        --property=MemoryHigh="$BC250_MEMORY_HIGH_BYTES" \
        --property=MemoryMax="$BC250_MEMORY_MAX_BYTES" \
        --property=MemorySwapMax="$BC250_MEMORY_SWAP_MAX_BYTES" \
        --property=CPUWeight=10000 \
        --property=IOWeight=1000 \
        --property=RuntimeMaxSec="$BC250_RUNTIME_MAX_SEC" \
        -- "$BC250_SCRIPT_DIR/guard-bc250-relink.sh" \
           "$BC250_METRICS_LOG" "$BC250_GAME_LOG" "$BC250_GAME_EXE" &
BC250_LAUNCH_PID=$!
set +e
wait "$BC250_LAUNCH_PID"
BC250_LAUNCH_STATUS=$?
set -e
BC250_LAUNCH_PID=0

cleanup_launcher
trap - EXIT
exit "$BC250_LAUNCH_STATUS"
