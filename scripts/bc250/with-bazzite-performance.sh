#!/usr/bin/env bash
# Hold Bazzite's tuned-ppd performance profile for exactly one command.
set -euo pipefail

if (( $# == 0 )); then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

if [[ ${BC250_PERFORMANCE_HOLD:-1} == 0 ]]; then
    exec "$@"
fi

for required in busctl sudo /usr/bin/python3; do
    if ! command -v "$required" >/dev/null 2>&1; then
        echo "error: performance-profile helper not found: $required" >&2
        exit 1
    fi
done

if ! /usr/bin/python3 -c 'import dbus' >/dev/null 2>&1; then
    echo "error: Python D-Bus bindings are required (Fedora: python3-dbus)" >&2
    exit 1
fi
if ! busctl introspect --system net.hadess.PowerProfiles \
        /net/hadess/PowerProfiles net.hadess.PowerProfiles \
        | grep -q '[.]HoldProfile'; then
    echo "error: tuned-ppd does not expose HoldProfile on the system bus" >&2
    exit 1
fi

# tuned-ppd associates a hold with its D-Bus caller and releases it if that
# caller disappears. Keeping the helper's stdin pipe open therefore gives us
# both an orderly ReleaseProfile and a crash-safe release on parent death.
BC250_HOLDER_CODE=$(printf '%s\n' \
    'import dbus' \
    'import sys' \
    'bus = dbus.SystemBus()' \
    'obj = bus.get_object("net.hadess.PowerProfiles", "/net/hadess/PowerProfiles")' \
    'iface = dbus.Interface(obj, "net.hadess.PowerProfiles")' \
    'cookie = iface.HoldProfile("performance", "Project ReLink", "org.biosed.OcarinaAsahiPort")' \
    'print(int(cookie), flush=True)' \
    'try:' \
    '    sys.stdin.buffer.read()' \
    'finally:' \
    '    try:' \
    '        iface.ReleaseProfile(cookie)' \
    '    except dbus.DBusException:' \
    '        pass')

BC250_HOLDER_PID=0
BC250_HOLDER_IN_FD=-1
BC250_HOLDER_OUT_FD=-1
BC250_CHILD_PID=0
BC250_HOLDER_CLEANED=0
BC250_RELEASE_FAILED=0

release_profile() {
    (( BC250_HOLDER_CLEANED == 0 )) || return 0
    BC250_HOLDER_CLEANED=1

    if (( BC250_HOLDER_IN_FD >= 0 )); then
        exec {BC250_HOLDER_IN_FD}>&-
        BC250_HOLDER_IN_FD=-1
    fi
    if (( BC250_HOLDER_OUT_FD >= 0 )); then
        exec {BC250_HOLDER_OUT_FD}>&-
        BC250_HOLDER_OUT_FD=-1
    fi
    if (( BC250_HOLDER_PID > 0 )); then
        wait "$BC250_HOLDER_PID" 2>/dev/null || true
        BC250_HOLDER_PID=0
    fi

    for _ in {1..30}; do
        BC250_ACTIVE_PROFILE=$(busctl get-property --system net.hadess.PowerProfiles \
            /net/hadess/PowerProfiles net.hadess.PowerProfiles ActiveProfile 2>/dev/null || true)
        if [[ $BC250_ACTIVE_PROFILE != 's "performance"' ]]; then
            return 0
        fi
        sleep 0.1
    done
    echo "error: performance profile remained active after releasing the hold" >&2
    BC250_RELEASE_FAILED=1
}

forward_signal() {
    local status=$1
    if (( BC250_CHILD_PID > 0 )) && kill -0 "$BC250_CHILD_PID" 2>/dev/null; then
        kill -TERM "$BC250_CHILD_PID" 2>/dev/null || true
        wait "$BC250_CHILD_PID" 2>/dev/null || true
    fi
    BC250_CHILD_PID=0
    exit "$status"
}

on_exit() {
    local status=$?
    trap - EXIT
    release_profile
    exit "$status"
}

trap on_exit EXIT
trap 'forward_signal 143' TERM
trap 'forward_signal 130' INT
trap 'forward_signal 129' HUP

coproc BC250_PPD_HOLDER {
    sudo -n /usr/bin/python3 -c "$BC250_HOLDER_CODE"
}
BC250_HOLDER_PID=$BC250_PPD_HOLDER_PID
BC250_HOLDER_OUT_FD=${BC250_PPD_HOLDER[0]}
BC250_HOLDER_IN_FD=${BC250_PPD_HOLDER[1]}

if ! IFS= read -r BC250_HOLD_COOKIE <&"$BC250_HOLDER_OUT_FD"; then
    echo "error: could not acquire Bazzite's performance profile" >&2
    echo "hint: run 'sudo -v' once, then retry" >&2
    exit 1
fi
if [[ ! $BC250_HOLD_COOKIE =~ ^[0-9]+$ ]]; then
    echo "error: tuned-ppd returned an invalid hold cookie" >&2
    exit 1
fi

# The D-Bus method returns before TuneD has necessarily finished applying the
# mapped profile. Do not launch until both the profile and all CPU policies are
# visibly in performance mode.
BC250_PROFILE_READY=0
for _ in {1..50}; do
    BC250_ACTIVE_PROFILE=$(busctl get-property --system net.hadess.PowerProfiles \
        /net/hadess/PowerProfiles net.hadess.PowerProfiles ActiveProfile 2>/dev/null || true)
    BC250_GOVERNORS_OK=1
    BC250_POLICY_COUNT=0
    shopt -s nullglob
    for BC250_GOVERNOR_PATH in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        (( BC250_POLICY_COUNT += 1 ))
        if [[ $(<"$BC250_GOVERNOR_PATH") != performance ]]; then
            BC250_GOVERNORS_OK=0
            break
        fi
    done
    shopt -u nullglob
    if [[ $BC250_ACTIVE_PROFILE == 's "performance"' ]] \
            && (( BC250_POLICY_COUNT > 0 )) && (( BC250_GOVERNORS_OK == 1 )); then
        BC250_PROFILE_READY=1
        break
    fi
    sleep 0.1
done

if (( BC250_PROFILE_READY == 0 )); then
    echo "error: Bazzite's performance profile did not select every CPU policy" >&2
    exit 1
fi

echo "power profile: performance (${BC250_POLICY_COUNT} CPU policies, hold ${BC250_HOLD_COOKIE})"

"$@" &
BC250_CHILD_PID=$!
set +e
wait "$BC250_CHILD_PID"
BC250_CHILD_STATUS=$?
set -e
BC250_CHILD_PID=0

release_profile
trap - EXIT
(( BC250_RELEASE_FAILED == 0 )) || exit 1
exit "$BC250_CHILD_STATUS"
