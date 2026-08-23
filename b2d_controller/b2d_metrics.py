r"""Aggregate Bench2Drive's four headline metrics over a set of local closed-loop runs.

This reproduces tools/b2d_finalize.py -- the script the leaderboard runner actually calls -- so
that a set of runs done on this workstation can be scored on the same terms as a dashboard
submission. Every formula below was read out of that file and Bench2Drive's own
tools/efficiency_smoothness_benchmark.py rather than reimplemented from the paper:

  Driving Score  mean of per-route ``score_composed`` = score_route * score_penalty, where the
                 penalty is a product of per-infraction coefficients (collision 0.6, red light
                 0.7, ...) computed by the leaderboard itself.
  Success Rate   fraction of routes with status Completed/Perfect AND no infraction of any kind
                 except ``min_speed_infractions``. A route can finish 100% and still fail this.
  Efficiency     mean of the percentages embedded in the ``min_speed_infractions`` messages
                 ("Average speed is 139.5% of the surrounding traffic's one"). Recorded but NOT
                 penalised -- the leaderboard has that event set to 'unused'.
  Comfortness    per route, metric_info.json is cut into 20-tick segments and a segment counts
                 only if ALL SIX channels stay in bounds; the metric is the ratio of such
                 segments, averaged over routes.

Comfortness is computed by importing Bench2Drive's own seg_compute_comfort_metric, not a copy, so
it cannot drift from the scored version. Note that that function has three known defects (deg/s
values checked against rad/s bounds, an undifferentiated "yaw acceleration" channel, and a 0.1 s
derivative delta on 0.05 s data) -- see the project's submission-hooks note. They are left intact
here on purpose: the point of this script is to report what the leaderboard would report.

Usage:
    python b2d_metrics.py --runs <runs_root> --suffix _mpckf [--suffix _pidviz] [--json out.json]
"""

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# viz_utils.B2D_TOOLS_DIR 와 같은 기본값 (그쪽 주석 참고) -- 두 곳이 같은 파일을 가리키도록 유지.
DEFAULT_TOOLS_DIR = os.path.join(HERE, "comfort_metric")


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def is_success(rec):
    """b2d_finalize.is_success: completed, and clean apart from the min-speed record."""
    if rec.get("status") not in ("Completed", "Perfect"):
        return False
    for key, val in (rec.get("infractions") or {}).items():
        if key == "min_speed_infractions":
            continue
        if isinstance(val, list) and val:
            return False
    return True


def efficiency_of(rec):
    """Mean of the percentages in the min-speed messages; >1000% dropped as the tool does."""
    vals = []
    for entry in (rec.get("infractions") or {}).get("min_speed_infractions") or []:
        m = re.search(r"\d+\.?\d*%", entry)
        if not m:
            continue
        pct = float(m.group().rstrip("%"))
        if pct <= 1000:
            vals.append(pct)
    return _mean(vals)


def comfort_of(run_dir, tools_dir):
    """Bench2Drive's own segment-wise comfort ratio for one route, or None without metric_info.

    On the lab's Ubuntu machine (--tools-dir pointed at the verified checkout), this imports the
    LAB'S SCORING MODULE, not a copy, so the number is the number the dashboard reports. Off that
    machine (the default --tools-dir, see DEFAULT_TOOLS_DIR above), it imports this repo's own
    b2d_controller/comfort_metric/efficiency_smoothness_benchmark.py instead -- a local
    reconstruction of the same fix, not verified byte-for-byte against the real file (see that
    file's own docstring for exactly what that means and how to replace it with the real one).

    The module (either copy) was corrected on 2026-08-18 (md5 0c650615... on the verified lab
    checkout) after this project reported three defects in it: yaw values in deg/s were being
    checked against rad/s bounds, the "yaw acceleration" channel was never differentiated, and
    jerks used a 0.1 s step on 0.05 s data. All three are fixed upstream now, the bounds are
    unchanged, and this project's own b2d_comfort_fixed.py -- written to score against while the
    fix was pending -- was verified to agree with it to 0.0000 on route 27582 before being retired.

    One difference from that interim module is worth recording: the corrected official version
    does NOT phase-unwrap the yaw rate ("a rate is not an angle"), which is right, and which the
    interim one had inherited from the buggy original. It made no numerical difference at the
    rates these runs reach, but the official behaviour is what is used from here on.
    """
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import numpy as np
        from efficiency_smoothness_benchmark import seg_compute_comfort_metric
    except Exception as exc:
        print(f"  ! comfort unavailable ({exc})")
        return None
    hits = glob.glob(os.path.join(run_dir, "frames", "*", "metric_info.json"))
    if not hits:
        return None
    frames = json.load(open(hits[0]))
    keys = ["acceleration", "angular_velocity", "forward_vector",
            "right_vector", "location", "rotation"]
    acc = {k: [] for k in keys}
    for _, frame in frames.items():
        for k in keys:
            acc[k].append(frame[k])
    if len(acc["angular_velocity"]) < 2:
        return None
    return float(seg_compute_comfort_metric(**{k: np.array(v) for k, v in acc.items()}))


def load_record(run_dir):
    path = os.path.join(run_dir, "results.json")
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))["_checkpoint"]["records"][0]
    except Exception:
        return None


def collect(runs_root, suffix, tools_dir):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(runs_root, "route*" + suffix))):
        route = os.path.basename(run_dir)[len("route"):-len(suffix)] if suffix else "?"
        rec = load_record(run_dir)
        if rec is None:
            rows.append(dict(route=route, dir=run_dir, missing=True))
            continue
        s = rec.get("scores", {})
        rows.append(dict(
            route=route, dir=run_dir, missing=False,
            status=rec.get("status"),
            score_route=s.get("score_route"),
            score_penalty=s.get("score_penalty"),
            score_composed=s.get("score_composed"),
            success=is_success(rec),
            efficiency=efficiency_of(rec),
            comfort=comfort_of(run_dir, tools_dir),
            duration=(rec.get("meta") or {}).get("duration_game"),
            infractions={k: len(v) for k, v in (rec.get("infractions") or {}).items() if v},
        ))
    return rows


def summarize(rows):
    done = [r for r in rows if not r["missing"]]
    if not done:
        return None
    return dict(
        n_routes=len(done),
        driving_score=_mean([r["score_composed"] for r in done]),
        route_completion=_mean([r["score_route"] for r in done]),
        success_rate=100.0 * sum(1 for r in done if r["success"]) / len(done),
        success_num=sum(1 for r in done if r["success"]),
        efficiency=_mean([r["efficiency"] for r in done]),
        comfortness=_mean([r["comfort"] for r in done]),
    )


def print_table(label, rows, summary):
    print(f"\n===== {label} =====")
    print(f"{'route':>7} {'status':<26} {'DS':>7} {'RC%':>6} {'pen':>7} "
          f"{'succ':>5} {'eff%':>7} {'comf':>6}  infractions")
    for r in rows:
        if r["missing"]:
            print(f"{r['route']:>7} {'(results.json 없음)':<26}")
            continue
        inf = ", ".join(f"{k}:{v}" for k, v in r["infractions"].items()
                        if k != "min_speed_infractions") or "-"
        print(f"{r['route']:>7} {str(r['status'])[:26]:<26} {r['score_composed']:7.2f} "
              f"{r['score_route']:6.1f} {r['score_penalty']:7.4f} "
              f"{'O' if r['success'] else 'X':>5} "
              f"{(r['efficiency'] if r['efficiency'] is not None else float('nan')):7.2f} "
              f"{(r['comfort'] if r['comfort'] is not None else float('nan')):6.3f}  {inf}")
    if summary:
        print(f"{'':>7} {'':<26} {'-'*7} {'-'*6} {'-'*7} {'-'*5} {'-'*7} {'-'*6}")
        n_txt = "n=%d" % summary["n_routes"]
        print(f"{'MEAN':>7} {n_txt:<26} "
              f"{summary['driving_score']:7.2f} {summary['route_completion']:6.1f} "
              f"{'':>7} {summary['success_rate']:4.0f}% "
              f"{(summary['efficiency'] or float('nan')):7.2f} "
              f"{(summary['comfortness'] or float('nan')):6.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/home/ailab/2026intern/kmlee/vad_demo_video/runs")
    ap.add_argument("--suffix", action="append", default=None,
                    help="run-directory suffix identifying one controller, e.g. _mpckf. "
                         "Repeat for several; default: _mpckf and _pidviz")
    ap.add_argument("--tools-dir", default=DEFAULT_TOOLS_DIR,
                    help="directory containing efficiency_smoothness_benchmark.py -- defaults to "
                         "this repo's own local copy (b2d_controller/comfort_metric/, see its "
                         "docstring); pass the lab machine's verified checkout "
                         "(/home/ailab/2026intern/kmlee/vad_demo_video/Bench2Drive/tools) to use "
                         "that instead")
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args()
    suffixes = args.suffix or ["_mpckf", "_pidviz"]

    out = {}
    for suf in suffixes:
        rows = collect(args.runs, suf, args.tools_dir)
        summary = summarize(rows)
        print_table(suf.lstrip("_"), rows, summary)
        out[suf.lstrip("_")] = dict(routes=rows, summary=summary)

    labels = [k for k in out if out[k]["summary"]]
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        sa, sb = out[a]["summary"], out[b]["summary"]
        print(f"\n===== {a} vs {b} =====")
        print(f"{'metric':<16} {a:>12} {b:>12} {'diff':>12}")
        for key, name in (("driving_score", "Driving Score"), ("success_rate", "Success Rate %"),
                          ("efficiency", "Efficiency %"), ("comfortness", "Comfortness")):
            va, vb = sa.get(key), sb.get(key)
            if va is None or vb is None:
                continue
            print(f"{name:<16} {va:12.4f} {vb:12.4f} {va - vb:+12.4f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
