"""(hookshot) An A/B within one run (HOS_AB, host/hos_ab.c) read from the frame trace: each side's frame times.

    python -I scripts/ab_summary.py <frames.csv>... [--boot 2000] [--seed 1] [--json]

<frames.csv> is the frame trace of a run with HOS_AB and RX_VK_FRAME_TRACE=1 set: logs/hookshot_frames.csv in the game folder
(frames.csv in the profile folder with HOS_PROFILE=1). Its ab_side column gives each frame's side: -1 outside the A/B (none set,
the warm-up), 0 the switch's normal value, 1 the A/B's value, 2 or 3 those sides' frames in a transition (a flip inside the
frame, or less than 0.3 s before its end). Frames outside the A/B and in transitions are left out.

A period is a run of frames of one side. The last one is left out when the run cut it short (under 80 % of the median period's
time), and then one more when needed so that the periods kept start and end on the same side (an odd count): the two sides'
periods are then centred on the same time, and a frame time that drifts steadily over the run (a scene warming up, memory
growing) moves both sides' means alike.

For each side: frames, mean, median and 99th percentile frame time, the share of frames of 16.7 ms or more, and fps (1000 / the
mean). Then side 1 less side 0, with a 95 % interval for the mean's difference from a block bootstrap over periods (frames within
a period are alike, so a period is resampled whole; each side's periods resampled apart). Each file alone, then all of them
together when there are several (their periods pooled). --json prints the numbers as JSON instead.
"""
import argparse
import csv
import json
import random
import statistics
import sys

SLOW_MS = 16.7
STEADY = (0, 1)       # (ab_side: a side's steady frames)
TRANSITION = (2, 3)   # (ab_side: a side's frames in a transition, side + 2)
NAMES = {0: "0 normal", 1: "1 A/B"}


def read_frames(path):
    """[(ms, ab_side)] of the trace, in order; a line cut short (the run ended while writing it) left out"""
    out = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = csv.DictReader(f)
        if not rows.fieldnames or "ab_side" not in rows.fieldnames or "ms" not in rows.fieldnames:
            raise SystemExit("%s: no ms and ab_side columns: not a frame trace of a build with HOS_AB (host/hos_ab.c)" % path)
        for r in rows:
            try:
                out.append((float(r["ms"]), int(r["ab_side"])))
            except (TypeError, ValueError):
                continue
    return out


def periods_of(frames):
    """The A/B's periods in order: {side, ms: [the steady frames' times], total: every frame's time, transitions}"""
    periods, last = [], None
    for ms, v in frames:
        if v not in STEADY and v not in TRANSITION:   # (outside the A/B: its warm-up)
            last = None
            continue
        side = v & 1
        # (a new period: the side changed, or a transition after steady frames, as after a stall that took a whole period)
        if last is None or side != last or (v in TRANSITION and periods[-1]["ms"]):
            periods.append({"side": side, "ms": [], "total": 0.0, "transitions": 0})
        p = periods[-1]
        p["total"] += ms
        if v in TRANSITION:
            p["transitions"] += 1
        else:
            p["ms"].append(ms)
        last = side
    return [p for p in periods if p["ms"]]


def trim(periods):
    """The periods kept: the last left out when cut short, then one more when needed for the same side at both ends"""
    dropped = 0
    if len(periods) > 2:
        median = statistics.median(p["total"] for p in periods[:-1])
        if periods[-1]["total"] < 0.8 * median:
            periods, dropped = periods[:-1], 1
    if len(periods) > 1 and periods[0]["side"] != periods[-1]["side"]:
        periods, dropped = periods[:-1], dropped + 1
    return periods, dropped


def percentile(sorted_ms, q):
    """The q-th percentile (0-100), interpolated between ranks"""
    if not sorted_ms:
        return 0.0
    at = (len(sorted_ms) - 1) * q / 100.0
    lo = int(at)
    hi = min(lo + 1, len(sorted_ms) - 1)
    return sorted_ms[lo] + (sorted_ms[hi] - sorted_ms[lo]) * (at - lo)


def side_stats(periods, side):
    ms = sorted(m for p in periods if p["side"] == side for m in p["ms"])
    if not ms:
        return {"periods": 0, "frames": 0}
    mean = sum(ms) / len(ms)
    return {"periods": sum(1 for p in periods if p["side"] == side), "frames": len(ms), "mean_ms": mean,
            "median_ms": percentile(ms, 50), "p99_ms": percentile(ms, 99),
            "slow_share": sum(1 for m in ms if m >= SLOW_MS) / len(ms), "fps": 1000.0 / mean}


def bootstrap(periods, boot, seed):
    """The 2.5th and 97.5th percentiles of side 1's mean less side 0's, each side's periods resampled whole; None with fewer
    than 2 periods a side"""
    by = {s: [(sum(p["ms"]), len(p["ms"])) for p in periods if p["side"] == s] for s in STEADY}
    if len(by[0]) < 2 or len(by[1]) < 2 or boot < 1:
        return None
    rnd = random.Random(seed)
    diffs = []
    for _ in range(boot):
        means = []
        for s in STEADY:
            pick = [rnd.choice(by[s]) for _ in by[s]]
            means.append(sum(t for t, _ in pick) / sum(n for _, n in pick))
        diffs.append(means[1] - means[0])
    diffs.sort()
    return percentile(diffs, 2.5), percentile(diffs, 97.5)


def summary(name, periods, dropped, left_out, boot, seed):
    s0, s1 = side_stats(periods, 0), side_stats(periods, 1)
    out = {"name": name, "periods": len(periods), "periods_left_out": dropped, "transition_frames": left_out[0],
           "outside_frames": left_out[1], "sides": {"0": s0, "1": s1}}
    if s0["frames"] and s1["frames"]:
        d = s1["mean_ms"] - s0["mean_ms"]
        out["diff"] = {"mean_ms": d, "mean_pct": 100 * d / s0["mean_ms"], "fps_pct": 100 * (s1["fps"] / s0["fps"] - 1),
                       "median_ms": s1["median_ms"] - s0["median_ms"], "p99_ms": s1["p99_ms"] - s0["p99_ms"],
                       "slow_share": s1["slow_share"] - s0["slow_share"]}
        ci = bootstrap(periods, boot, seed)
        if ci:
            out["diff"]["ci_ms"] = list(ci)
            out["diff"]["ci_pct"] = [100 * c / s0["mean_ms"] for c in ci]
    return out


def show(r):
    print("%s: %d periods (%d of side 0, %d of side 1); left out: %d period(s) at the end, %d transition frames, %d frames "
          "outside the A/B" % (r["name"], r["periods"], r["sides"]["0"]["periods"], r["sides"]["1"]["periods"],
                               r["periods_left_out"], r["transition_frames"], r["outside_frames"]))
    print("  %-9s %7s %8s %8s %8s %9s %7s" % ("side", "frames", "mean ms", "median", "p99", ">=16.7ms", "fps"))
    for s in ("0", "1"):
        st = r["sides"][s]
        if not st["frames"]:
            print("  %-9s %7d  (no frames)" % (NAMES[int(s)], 0))
            continue
        print("  %-9s %7d %8.3f %8.3f %8.3f %8.1f%% %7.1f" % (NAMES[int(s)], st["frames"], st["mean_ms"], st["median_ms"],
                                                           st["p99_ms"], 100 * st["slow_share"], st["fps"]))
    d = r.get("diff")
    if not d:
        print("  no difference: a side has no frames")
        return
    line = "  side 1 - side 0: mean %+.3f ms (%+.2f %%, fps %+.2f %%), median %+.3f ms, p99 %+.3f ms, >=16.7 ms %+.1f points" % (
        d["mean_ms"], d["mean_pct"], d["fps_pct"], d["median_ms"], d["p99_ms"], 100 * d["slow_share"])
    print(line)
    if "ci_ms" not in d:
        print("  95 %% interval: too few periods (2 a side at least)")
        return
    lo, hi = d["ci_pct"]
    verdict = ("side 1 is slower" if lo > 0 else "side 1 is faster" if hi < 0 else "no difference shown (the interval holds 0)")
    print("  95 %% interval of the mean's difference: %+.3f to %+.3f ms (%+.2f to %+.2f %%): %s"
          % (d["ci_ms"][0], d["ci_ms"][1], lo, hi, verdict))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("frames", nargs="+", help="frame traces (logs/hookshot_frames.csv) of runs with HOS_AB")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (default 2000)")
    ap.add_argument("--seed", type=int, default=1, help="the bootstrap's seed, for the same interval each time (default 1)")
    ap.add_argument("--json", action="store_true", help="the numbers as JSON")
    a = ap.parse_args()
    results, pooled, pooled_out, pooled_dropped = [], [], [0, 0], 0
    for path in a.frames:
        frames = read_frames(path)
        left_out = [sum(1 for _, v in frames if v in TRANSITION), sum(1 for _, v in frames if v not in STEADY + TRANSITION)]
        periods, dropped = trim(periods_of(frames))
        results.append(summary(path, periods, dropped, left_out, a.boot, a.seed))
        pooled += periods
        pooled_dropped += dropped
        pooled_out = [pooled_out[0] + left_out[0], pooled_out[1] + left_out[1]]
    if len(a.frames) > 1:
        results.append(summary("all %d files" % len(a.frames), pooled, pooled_dropped, pooled_out, a.boot, a.seed))
    if a.json:
        json.dump(results, sys.stdout, indent=1)
        print()
        return
    for r in results:
        show(r)


if __name__ == "__main__":
    main()
