r"""Validate mpc_kf_controller.py's build_trajectory_splines()/preview_from_splines() against real
VAD trajectory samples (collect_offline_samples.py's output) -- checks the PathSpline + vx-spline
fit behaves sanely across a broad batch, not just the handful of cases hand-picked during design.

Usage (Windows, from this file's own directory -- .venv lives one level up in carla_control/, so
invoke its python.exe directly rather than relying on `python` being on PATH):
    cd C:\Users\mumu2\carla_control\b2d_controller
    ..\.venv\Scripts\python.exe validate_trajectory_fit.py --samples vad_trajectory_samples_offline.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpc_kf_controller import DT, VX_FLOOR, build_trajectory_splines, preview_from_splines


def check_one(record, n_p=20, dt=DT):
    """Run the real fitting pipeline on one sample; return a dict of pass/fail + diagnostics.
    Never raises -- a fit that throws is itself a finding, caught and reported as a failure."""
    out = dict(idx=record["idx"], folder=record.get("folder", ""), speed=record["speed"], ok=True,
              reasons=[])
    try:
        path, vx_spline, s_max_wp = build_trajectory_splines(record["out_truck"], record["speed"])
        vx_preview, kappa_preview = preview_from_splines(path, vx_spline, s_max_wp, n_p, dt)
    except Exception as exc:
        out["ok"] = False
        out["reasons"].append(f"exception: {exc}")
        out["vx_preview"] = None
        out["kappa_preview"] = None
        return out

    if not np.all(np.isfinite(vx_preview)):
        out["ok"] = False
        out["reasons"].append("vx_preview has NaN/Inf")
    if not np.all(np.isfinite(kappa_preview)):
        out["ok"] = False
        out["reasons"].append("kappa_preview has NaN/Inf")
    if np.any(vx_preview < VX_FLOOR - 1e-6):
        out["ok"] = False
        out["reasons"].append(f"vx_preview below VX_FLOOR ({vx_preview.min():.3f})")
    # A B2D car isn't going to be doing 40 m/s in a 6-waypoint/3s VAD horizon; a spline blowing
    # past this is a fit-quality problem (over-fit tail, extrapolation runaway), not real driving.
    if np.any(vx_preview > 40.0):
        out["ok"] = False
        out["reasons"].append(f"vx_preview unrealistically high ({vx_preview.max():.1f} m/s)")
    # Town streets: nothing here corners tighter than ~5 m radius (kappa=0.2/m). A spline recovering
    # more than that from 6 coarse points is almost certainly ringing, not a real corner.
    if np.any(np.abs(kappa_preview) > 0.3):
        out["ok"] = False
        out["reasons"].append(f"kappa_preview unrealistically sharp (max|k|={np.abs(kappa_preview).max():.3f})")

    out["vx_preview"] = vx_preview
    out["kappa_preview"] = kappa_preview
    out["vx_range"] = (float(vx_preview.min()), float(vx_preview.max()))
    out["kappa_range"] = (float(kappa_preview.min()), float(kappa_preview.max()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "vad_trajectory_samples_offline.pkl"))
    args = parser.parse_args()

    with open(args.samples, "rb") as f:
        data = pickle.load(f)
    # both collect_offline_samples.py (a list) and the earlier online out_truck_dump.pkl-style
    # merge (sequential pickle.dump calls) are supported -- normalize to a list either way.
    if isinstance(data, list):
        records = data
    else:
        records = [data]

    print(f"Loaded {len(records)} sample(s) from {args.samples}\n")

    results = [check_one(r) for r in records]
    n_ok = sum(r["ok"] for r in results)
    n_fail = len(results) - n_ok

    print(f"=== {n_ok}/{len(results)} passed, {n_fail} failed ===\n")

    if n_fail:
        print("--- failures ---")
        for r in results:
            if not r["ok"]:
                print(f"  idx={r['idx']:4d}  speed={r['speed']:.2f}  folder={r['folder']}")
                for reason in r["reasons"]:
                    print(f"      {reason}")

    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        vx_mins = np.array([r["vx_range"][0] for r in ok_results])
        vx_maxs = np.array([r["vx_range"][1] for r in ok_results])
        kappa_mins = np.array([r["kappa_range"][0] for r in ok_results])
        kappa_maxs = np.array([r["kappa_range"][1] for r in ok_results])
        print("\n--- summary over passing samples ---")
        print(f"  vx_preview      : min={vx_mins.min():.2f}  median_min={np.median(vx_mins):.2f}  "
             f"median_max={np.median(vx_maxs):.2f}  max={vx_maxs.max():.2f}  (m/s)")
        print(f"  kappa_preview   : min={kappa_mins.min():+.4f}  median_min={np.median(kappa_mins):+.4f}  "
             f"median_max={np.median(kappa_maxs):+.4f}  max={kappa_maxs.max():+.4f}  (1/m)")
        speeds = np.array([r["speed"] for r in ok_results])
        print(f"  input speed     : min={speeds.min():.2f}  max={speeds.max():.2f}  mean={speeds.mean():.2f}  (m/s)")


if __name__ == "__main__":
    main()
