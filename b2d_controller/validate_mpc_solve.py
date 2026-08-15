r"""Validate MpcKfController end-to-end (build_trajectory_splines() fit + VyKalmanFilter step +
LateralMPC.solve() + SpeedMPC.solve() + speed PID pedal tracker) against real VAD trajectory
samples (collect_offline_samples.py's output) -- checks that BOTH QPs actually SOLVE (OSQP status,
finite/in-range control outputs) across a broad batch. mpc_kf_controller.py's LateralMPC/SpeedMPC
are unchanged copies of carla_control's own mpc_mpc_comparison.py LateralMPC and
longitudinal_mpc.py's SpeedMPC -- see those files' own class docstrings for the QP derivations
(condensed LPV-bicycle-model / scalar-integrator state-space, output tracking, box/rate
constraints) this is exercising.

validate_trajectory_fit.py already checks the spline fit itself (vx_preview/kappa_preview sanity,
e.g. no unrealistic curvature); this file goes one step further and actually calls solve() with
those previews, since a sane-looking preview can still make OSQP choke in ways the fit-only check
never exercises (e.g. a near-stationary sample's near-degenerate condensed QP, or a preview that
makes H numerically ill-conditioned) -- that failure mode only shows up once solve() actually runs.

Each sample gets a FRESH MpcKfController (no KF state or LateralMPC.last_solution/last_status
carried from one sample to the next): the offline dataset's samples are independent frames from
different routes/episodes (not a continuous drive), so persisting state across them would mix
unrelated tracks into one fake "history" -- same per-sample independence validate_trajectory_fit.py's
check_one() already assumes.

Usage (Windows, from this file's own directory -- .venv lives one level up in carla_control/, so
invoke its python.exe directly rather than relying on `python` being on PATH):
    cd C:\Users\mumu2\carla_control\b2d_controller
    ..\.venv\Scripts\python.exe validate_mpc_solve.py --samples vad_trajectory_samples_offline.pkl
"""
import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpc_kf_controller import MpcKfController

# OSQP statuses that count as a genuine solve (result.x is the actual optimum, not a stale
# last_solution fallback) -- see LateralMPC.solve()'s own "result.x is None" fallback branch, which
# silently keeps driving on the previous tick's plan rather than raising; that's the right behavior
# for a live agent (never hand CARLA a NaN steer), but it means a bad status has to be caught HERE,
# from metadata['kf_status'], not by looking for an exception or a NaN control output.
OK_STATUSES = {"solved", "solved inaccurate"}


def check_one(record):
    """Run MpcKfController.control_pid() on one sample; return a dict of pass/fail + diagnostics.
    Never raises -- a call that throws is itself a finding, caught and reported as a failure."""
    out = dict(idx=record["idx"], folder=record.get("folder", ""), speed=record["speed"], ok=True,
              reasons=[], lat_status=None, lon_status=None)
    try:
        controller = MpcKfController()
        r_meas = float(record["angular_velocity"][2])
        ay_meas = float(record["acceleration"][1])
        steer, throttle, brake, metadata = controller.control_pid(
            record["out_truck"], record["speed"], r_meas, ay_meas)
    except Exception as exc:
        out["ok"] = False
        out["reasons"].append(f"exception: {exc}")
        return out

    lat_status, lon_status = metadata["kf_status"], metadata["speed_mpc_status"]
    out["lat_status"], out["lon_status"] = lat_status, lon_status
    if lat_status not in OK_STATUSES:
        out["ok"] = False
        out["reasons"].append(f"LateralMPC OSQP status: {lat_status}")
    if lon_status not in OK_STATUSES:
        out["ok"] = False
        out["reasons"].append(f"SpeedMPC OSQP status: {lon_status}")
    if not (np.isfinite(steer) and np.isfinite(throttle)):
        out["ok"] = False
        out["reasons"].append(f"non-finite control output (steer={steer}, throttle={throttle})")
    if not (-1.0 - 1e-6 <= steer <= 1.0 + 1e-6):
        out["ok"] = False
        out["reasons"].append(f"steer out of [-1, 1]: {steer:.4f}")
    if not (0.0 - 1e-6 <= throttle <= 1.0 + 1e-6):
        out["ok"] = False
        out["reasons"].append(f"throttle out of [0, 1]: {throttle:.4f}")

    out["steer"] = steer
    out["throttle"] = throttle
    out["brake"] = brake
    out["delta_deg"] = np.degrees(metadata["delta_rad"])
    out["a_cmd"] = metadata["a_cmd"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "vad_trajectory_samples_offline.pkl"))
    args = parser.parse_args()

    with open(args.samples, "rb") as f:
        data = pickle.load(f)
    records = data if isinstance(data, list) else [data]

    print(f"Loaded {len(records)} sample(s) from {args.samples}\n")

    results = [check_one(r) for r in records]
    n_ok = sum(r["ok"] for r in results)
    n_fail = len(results) - n_ok

    print(f"=== {n_ok}/{len(results)} solved cleanly, {n_fail} failed ===\n")

    if n_fail:
        print("--- failures ---")
        for r in results:
            if not r["ok"]:
                print(f"  idx={r['idx']:4d}  speed={r['speed']:.2f}  folder={r['folder']}")
                for reason in r["reasons"]:
                    print(f"      {reason}")
        print()

    lat_counts = Counter(r["lat_status"] for r in results)
    lon_counts = Counter(r["lon_status"] for r in results)
    print("--- LateralMPC OSQP status breakdown ---")
    for status, n in lat_counts.most_common():
        print(f"  {status!r}: {n}")
    print("--- SpeedMPC OSQP status breakdown ---")
    for status, n in lon_counts.most_common():
        print(f"  {status!r}: {n}")

    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        steers = np.array([r["steer"] for r in ok_results])
        deltas = np.array([r["delta_deg"] for r in ok_results])
        a_cmds = np.array([r["a_cmd"] for r in ok_results])
        print("\n--- summary over cleanly-solved samples ---")
        print(f"  steer  : min={steers.min():+.3f}  median={np.median(steers):+.3f}  max={steers.max():+.3f}")
        print(f"  delta  : min={deltas.min():+.2f} deg  median={np.median(deltas):+.2f} deg  "
             f"max={deltas.max():+.2f} deg")
        print(f"  a_cmd  : min={a_cmds.min():+.2f} m/s^2  median={np.median(a_cmds):+.2f} m/s^2  "
             f"max={a_cmds.max():+.2f} m/s^2")


if __name__ == "__main__":
    main()
