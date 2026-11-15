#!/usr/bin/env bash
# Stop a BC-250 Project ReLink run before host OOM or an unsafe GPU temperature.
set -euo pipefail

if (( $# < 3 )); then
    echo "usage: $0 METRICS_LOG GAME_LOG COMMAND [ARG ...]" >&2
    exit 2
fi

BC250_METRICS_LOG=$1
BC250_GAME_LOG=$2
shift 2
BC250_MIN_AVAILABLE_MIB=${BC250_MIN_AVAILABLE_MIB:-2048}
BC250_SAMPLE_INTERVAL_SEC=${BC250_SAMPLE_INTERVAL_SEC:-1}
BC250_MAX_TEMP_MILLIC=${BC250_MAX_TEMP_MILLIC:-90000}
BC250_REQUIRE_SWAPCHAIN=${BC250_REQUIRE_SWAPCHAIN:-1}
BC250_SWAPCHAIN_TIMEOUT_SEC=${BC250_SWAPCHAIN_TIMEOUT_SEC:-60}

[[ $BC250_MIN_AVAILABLE_MIB =~ ^[0-9]+$ ]] || {
    echo "error: BC250_MIN_AVAILABLE_MIB must be an integer" >&2
    exit 2
}
(( BC250_MIN_AVAILABLE_MIB >= 2048 )) || {
    echo "error: BC250_MIN_AVAILABLE_MIB must be at least 2048" >&2
    exit 2
}
[[ $BC250_MAX_TEMP_MILLIC =~ ^[0-9]+$ ]] || {
    echo "error: BC250_MAX_TEMP_MILLIC must be an integer" >&2
    exit 2
}
[[ $BC250_SAMPLE_INTERVAL_SEC =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "error: BC250_SAMPLE_INTERVAL_SEC must be a positive number" >&2
    exit 2
}
[[ $BC250_REQUIRE_SWAPCHAIN =~ ^[01]$ ]] || {
    echo "error: BC250_REQUIRE_SWAPCHAIN must be 0 or 1" >&2
    exit 2
}
[[ $BC250_SWAPCHAIN_TIMEOUT_SEC =~ ^[0-9]+$ ]] || {
    echo "error: BC250_SWAPCHAIN_TIMEOUT_SEC must be an integer" >&2
    exit 2
}
(( BC250_SWAPCHAIN_TIMEOUT_SEC >= 5 )) || {
    echo "error: BC250_SWAPCHAIN_TIMEOUT_SEC must be at least 5 seconds" >&2
    exit 2
}
awk -v value="$BC250_SAMPLE_INTERVAL_SEC" 'BEGIN { exit !(value > 0) }' || {
    echo "error: BC250_SAMPLE_INTERVAL_SEC must be greater than zero" >&2
    exit 2
}

BC250_MIN_AVAILABLE_KIB=$((BC250_MIN_AVAILABLE_MIB * 1024))
BC250_CHILD_PID=0
BC250_STOPPING=0
BC250_SWAPCHAIN_READY=0
BC250_START_SECONDS=$SECONDS

read_number() {
    local path=$1 value
    [[ -r $path ]] || { echo 0; return; }
    value=$(<"$path")
    [[ $value =~ ^[0-9]+$ ]] && echo "$value" || echo 0
}

stop_child() {
    (( BC250_CHILD_PID > 0 )) || return 0
    kill -0 "$BC250_CHILD_PID" 2>/dev/null || return 0
    (( BC250_STOPPING == 0 )) || return 0
    BC250_STOPPING=1
    kill -TERM "$BC250_CHILD_PID" 2>/dev/null || true
    for _ in {1..100}; do
        kill -0 "$BC250_CHILD_PID" 2>/dev/null || return 0
        sleep 0.1
    done
    kill -KILL "$BC250_CHILD_PID" 2>/dev/null || true
}

on_signal() {
    local status=$1
    stop_child
    wait "$BC250_CHILD_PID" 2>/dev/null || true
    exit "$status"
}

trap 'on_signal 143' TERM
trap 'on_signal 130' INT
trap 'on_signal 129' HUP

BC250_MEM_AVAILABLE_KIB=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)
if (( BC250_MEM_AVAILABLE_KIB < BC250_MIN_AVAILABLE_KIB )); then
    echo "GUARD: MemAvailable is below ${BC250_MIN_AVAILABLE_MIB} MiB before launch" >&2
    exit 125
fi

if [[ -n ${BC250_DRM_DEVICE:-} ]]; then
    BC250_GPU_DEVICE=$BC250_DRM_DEVICE
else
    BC250_GPU_DEVICE=
    shopt -s nullglob
    for BC250_VENDOR_PATH in /sys/class/drm/card*/device/vendor; do
        if [[ $(<"$BC250_VENDOR_PATH") == 0x1002 ]]; then
            BC250_GPU_DEVICE=${BC250_VENDOR_PATH%/vendor}
            break
        fi
    done
    shopt -u nullglob
fi
if [[ -z $BC250_GPU_DEVICE || ! -d $BC250_GPU_DEVICE ]]; then
    echo "error: could not find the BC-250 AMD DRM device" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$BC250_METRICS_LOG")"
mkdir -p "$(dirname -- "$BC250_GAME_LOG")"
: >"$BC250_GAME_LOG"
printf '%s\n' \
    'utc mem_available_kib cgroup_bytes cgroup_swap_bytes proc_rss_anon_kib proc_rss_file_kib proc_rss_shmem_kib cgroup_anon_bytes cgroup_file_bytes cgroup_shmem_bytes cgroup_kernel_bytes cgroup_high_events cgroup_psi_some_usec vram_bytes gtt_bytes sclk_mhz max_temp_millic' \
    >"$BC250_METRICS_LOG"

"$@" >>"$BC250_GAME_LOG" 2>&1 &
BC250_CHILD_PID=$!

while kill -0 "$BC250_CHILD_PID" 2>/dev/null; do
    if (( BC250_REQUIRE_SWAPCHAIN == 1 && BC250_SWAPCHAIN_READY == 0 )); then
        if grep -Eq 'vk: swapchain .* mailbox' "$BC250_GAME_LOG"; then
            BC250_SWAPCHAIN_READY=1
            echo "GUARD: verified a visible mailbox swapchain"
        elif grep -Eq 'SDL_Init: .*no window|vk: no window:' "$BC250_GAME_LOG"; then
            echo "GUARD: ReLink fell back to offscreen rendering; stopping game" >&2
            stop_child
            wait "$BC250_CHILD_PID" 2>/dev/null || true
            exit 124
        elif (( SECONDS - BC250_START_SECONDS >= BC250_SWAPCHAIN_TIMEOUT_SEC )); then
            echo "GUARD: no mailbox swapchain after ${BC250_SWAPCHAIN_TIMEOUT_SEC}s; stopping game" >&2
            stop_child
            wait "$BC250_CHILD_PID" 2>/dev/null || true
            exit 124
        fi
    fi
    BC250_MEM_AVAILABLE_KIB=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)
    BC250_CGROUP_REL=$(awk -F: '$1 == "0" { print $3; exit }' \
        "/proc/$BC250_CHILD_PID/cgroup" 2>/dev/null || true)
    BC250_CGROUP_ROOT=/sys/fs/cgroup$BC250_CGROUP_REL
    BC250_CGROUP_BYTES=$(read_number "$BC250_CGROUP_ROOT/memory.current")
    BC250_CGROUP_SWAP_BYTES=$(read_number "$BC250_CGROUP_ROOT/memory.swap.current")
    read -r BC250_RSS_ANON_KIB BC250_RSS_FILE_KIB BC250_RSS_SHMEM_KIB < <(
        awk 'BEGIN { a = f = s = 0 }
             $1 == "RssAnon:" { a = $2 }
             $1 == "RssFile:" { f = $2 }
             $1 == "RssShmem:" { s = $2 }
             END { print a, f, s }' "/proc/$BC250_CHILD_PID/status" 2>/dev/null || echo '0 0 0'
    )
    read -r BC250_CGROUP_ANON BC250_CGROUP_FILE BC250_CGROUP_SHMEM BC250_CGROUP_KERNEL < <(
        awk 'BEGIN { a = f = s = k = 0 }
             $1 == "anon" { a = $2 }
             $1 == "file" { f = $2 }
             $1 == "shmem" { s = $2 }
             $1 == "kernel" { k = $2 }
             END { print a, f, s, k }' "$BC250_CGROUP_ROOT/memory.stat" 2>/dev/null || echo '0 0 0 0'
    )
    BC250_CGROUP_HIGH=$(awk '$1 == "high" { print $2; found = 1; exit }
        END { if (!found) print 0 }' "$BC250_CGROUP_ROOT/memory.events" 2>/dev/null || echo 0)
    BC250_CGROUP_PSI=$(awk '$1 == "some" {
        for (i = 2; i <= NF; i++) if ($i ~ /^total=/) {
            sub(/^total=/, "", $i); print $i; found = 1; exit
        }} END { if (!found) print 0 }' \
        "$BC250_CGROUP_ROOT/memory.pressure" 2>/dev/null || echo 0)
    BC250_VRAM_BYTES=$(read_number "$BC250_GPU_DEVICE/mem_info_vram_used")
    BC250_GTT_BYTES=$(read_number "$BC250_GPU_DEVICE/mem_info_gtt_used")
    BC250_SCLK_MHZ=$(awk '/\*/ { sub(/Mhz/, "", $2); print $2 + 0; exit }' \
        "$BC250_GPU_DEVICE/pp_dpm_sclk" 2>/dev/null || echo 0)
    BC250_MAX_SEEN_TEMP=0
    shopt -s nullglob
    for BC250_SENSOR_PATH in "$BC250_GPU_DEVICE"/hwmon/hwmon*/temp*_input; do
        BC250_SENSOR_VALUE=$(read_number "$BC250_SENSOR_PATH")
        (( BC250_SENSOR_VALUE > BC250_MAX_SEEN_TEMP )) && BC250_MAX_SEEN_TEMP=$BC250_SENSOR_VALUE
    done
    for BC250_HWMON_PATH in /sys/class/hwmon/hwmon*; do
        [[ -r $BC250_HWMON_PATH/name ]] || continue
        [[ $(<"$BC250_HWMON_PATH/name") == k10temp ]] || continue
        for BC250_SENSOR_PATH in "$BC250_HWMON_PATH"/temp*_input; do
            BC250_SENSOR_VALUE=$(read_number "$BC250_SENSOR_PATH")
            (( BC250_SENSOR_VALUE > BC250_MAX_SEEN_TEMP )) && BC250_MAX_SEEN_TEMP=$BC250_SENSOR_VALUE
        done
    done
    shopt -u nullglob

    printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
        "$(date -u +%FT%TZ)" "$BC250_MEM_AVAILABLE_KIB" "$BC250_CGROUP_BYTES" \
        "$BC250_CGROUP_SWAP_BYTES" "$BC250_RSS_ANON_KIB" "$BC250_RSS_FILE_KIB" \
        "$BC250_RSS_SHMEM_KIB" "$BC250_CGROUP_ANON" "$BC250_CGROUP_FILE" \
        "$BC250_CGROUP_SHMEM" "$BC250_CGROUP_KERNEL" "$BC250_CGROUP_HIGH" \
        "$BC250_CGROUP_PSI" "$BC250_VRAM_BYTES" "$BC250_GTT_BYTES" \
        "$BC250_SCLK_MHZ" "$BC250_MAX_SEEN_TEMP" >>"$BC250_METRICS_LOG"

    if (( BC250_MEM_AVAILABLE_KIB < BC250_MIN_AVAILABLE_KIB )); then
        echo "GUARD: MemAvailable fell below ${BC250_MIN_AVAILABLE_MIB} MiB; stopping game" >&2
        stop_child
        wait "$BC250_CHILD_PID" 2>/dev/null || true
        exit 125
    fi
    if (( BC250_MAX_SEEN_TEMP >= BC250_MAX_TEMP_MILLIC )); then
        echo "GUARD: GPU temperature reached ${BC250_MAX_SEEN_TEMP} millidegrees C; stopping game" >&2
        stop_child
        wait "$BC250_CHILD_PID" 2>/dev/null || true
        exit 126
    fi
    sleep "$BC250_SAMPLE_INTERVAL_SEC"
done

set +e
wait "$BC250_CHILD_PID"
BC250_CHILD_STATUS=$?
set -e
if (( BC250_REQUIRE_SWAPCHAIN == 1 && BC250_SWAPCHAIN_READY == 0 )); then
    echo "GUARD: game exited before a mailbox swapchain was verified" >&2
    exit 124
fi
exit "$BC250_CHILD_STATUS"
