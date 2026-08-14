r"""Deep-dive on ONE real VAD trajectory sample: prints the raw waypoints, the actual fitted spline
(knots + per-segment cubic polynomial coefficients, not just a plot), and the resulting vx/kappa
preview -- validate_trajectory_fit.py only reports pass/fail + aggregate stats across the whole
batch; this is the single-sample "how did this number actually come out" companion to it.

Usage:
    python3 inspect_one_sample.py                  # random sample
    python3 inspect_one_sample.py --idx 196         # a specific one (e.g. the sharpest-curvature
                                                     # sample validate_trajectory_fit.py's summary
                                                     # flagged as the max|kappa| case)
    python3 inspect_one_sample.py --idx 196 --seed 0
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CARLA_CONTROL_ROOT = os.path.dirname(HERE)   # this file lives in carla_control/b2d_controller/ --
# viz_utils.py is one level up, in carla_control/ itself. Only this dev/inspection script needs it
# (and, through it, `carla`, for the CARLA_ROOT sys.path setup below) -- mpc_kf_controller.py itself
# stays carla_control-free since IT ships inside the .sif; this script never does.
sys.path.insert(0, HERE)
sys.path.insert(0, CARLA_CONTROL_ROOT)

CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/carla/CARLA_0.9.15"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

from mpc_kf_controller import (DT, VAD_WP_DT, _chord_length_station, build_trajectory_splines,
                               preview_from_splines)
from viz_utils import plot_trajectory_fit


def print_spline_formula(name, spl, unit=""):
    """Per-segment cubic Taylor expansion x(s) = c0 + c1*u + c2*u^2 + c3*u^3, u = s - s_i, read off
    the spline's own derivative evaluations at each knot -- public API only (spl(x, nu=n)), not the
    private FITPACK tck tuple, so this works regardless of scipy version internals."""
    knots = spl.get_knots()
    print(f"  {name}: {len(knots)} knots (spline pieces: {len(knots) - 1}), "
         f"residual={spl.get_residual():.4g}")
    print(f"    knots (station, m): {np.round(knots, 3).tolist()}")
    eps = 1e-6
    for i in range(len(knots) - 1):
        s0, s1 = knots[i], knots[i + 1]
        s_eval = min(s0 + eps, (s0 + s1) / 2)
        c0 = float(spl(s_eval, 0))
        c1 = float(spl(s_eval, 1))
        c2 = float(spl(s_eval, 2)) / 2.0
        c3 = float(spl(s_eval, 3)) / 6.0
        print(f"    s in [{s0:6.3f}, {s1:6.3f}]:  f(s) = {c0:+.4f} {c1:+.4f}*u {c2:+.5f}*u^2 "
             f"{c3:+.6f}*u^3   (u = s - {s0:.3f}){unit}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "vad_trajectory_samples_offline.pkl"))
    parser.add_argument("--idx", type=int, default=None, help="sample index; default: random")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for random --idx pick")
    parser.add_argument("--n-p", type=int, default=20)
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"),
                        help="directory image outputs go into (kept separate from the .pkl/.py files)")
    parser.add_argument("--out", default=None,
                        help="output filename; default: inspect_idx<N>.png under --plot-dir")
    args = parser.parse_args()

    with open(args.samples, "rb") as f:
        records = pickle.load(f)
    records = records if isinstance(records, list) else [records]

    rng = np.random.default_rng(args.seed)
    idx = args.idx if args.idx is not None else int(rng.integers(len(records)))
    r = records[idx]
    waypoints, target, speed = r["out_truck"], r["target"], r["speed"]

    print(f"=== sample idx={idx}  (dataset entry idx={r.get('idx')}, folder={r.get('folder', '?')}) ===")
    print(f"current speed: {speed:.3f} m/s\n")

    print(f"raw VAD waypoints, [lateral, forward] m, {VAD_WP_DT}s apart (this file's own convention "
         f"-- forward is index 1, confirmed via pid_controller.py's atan2 formula):")
    print(f"  origin (t=0.0s): lateral= 0.000  forward= 0.000   <- ego, always the fit's own origin")
    for i, wp in enumerate(waypoints):
        print(f"  wp{i} (t={(i + 1) * VAD_WP_DT:.1f}s): lateral={wp[0]:+.4f}  forward={wp[1]:+.4f}")
    print(f"  target (t=unknown, extrapolated -- see mpc_kf_controller.py's build_trajectory_splines "
         f"docstring): lateral={target[0]:+.4f}  forward={target[1]:+.4f}\n")

    # Reproduce build_trajectory_splines()'s own intermediate arrays so we can print/inspect them,
    # not just call the function as a black box.
    fwd = [0.0] + [wp[1] for wp in waypoints] + [target[1]]
    lat = [0.0] + [wp[0] for wp in waypoints] + [target[0]]
    s_all = _chord_length_station(fwd, lat)
    print(f"chord-length station s for [origin, wp0..wp{len(waypoints) - 1}, target]:")
    print(f"  s = {np.round(s_all, 3).tolist()}  (m)\n")

    path, vx_spline, s_max_wp = build_trajectory_splines(waypoints, target)
    print(f"PathSpline geometry fit -- x(s) [forward] and y(s) [lateral], smoothing=0.05, k=3:")
    print_spline_formula("x(s) [forward, m]", path._sx)
    print_spline_formula("y(s) [lateral, m]", path._sy)
    print()

    n_wp = len(waypoints)
    s_wp = s_all[:n_wp + 1]
    t_wp = np.arange(n_wp + 1) * VAD_WP_DT
    ds, dt = np.diff(s_wp), np.diff(t_wp)
    v_seg = ds / np.maximum(dt, 1e-3)
    s_mid = s_wp[:-1] + ds / 2.0
    print(f"vx(s) spline input -- per-interval average speed ds/dt, assigned to each interval's own "
         f"arc-length midpoint (s_max_wp={s_max_wp:.3f} m is the last real waypoint's station; "
         f"vx_spline is only data-informed up to here):")
    for sm, v in zip(s_mid, v_seg):
        print(f"    s_mid={sm:6.3f} m   v={v:5.2f} m/s")
    print_spline_formula("vx(s) [m/s]", vx_spline)
    print()

    vx_preview, kappa_preview = preview_from_splines(path, vx_spline, s_max_wp, args.n_p, DT)
    s_cursor = np.concatenate([[0.0], np.cumsum(vx_preview * DT)[:-1]])
    print(f"preview_from_splines() walk (n_p={args.n_p}, dt={DT}s) -- station cursor advances by "
         f"vx(s_cursor)*dt each step, both vx and kappa sampled off the SAME station:")
    print(f"  {'step':>4}  {'s (m)':>7}  {'vx (m/s)':>9}  {'kappa (1/m)':>12}")
    for j in range(args.n_p):
        print(f"  {j:4d}  {s_cursor[j]:7.3f}  {vx_preview[j]:9.3f}  {kappa_preview[j]:+12.5f}")

    # ---- plot (viz_utils.plot_trajectory_fit -- shared COLOR_*/LINEWIDTH/FONTSIZE_* knobs) ----
    # No matplotlib.use() here: viz_utils.py already sets TkAgg at import time (above), which is
    # what makes the plt.show() below actually pop up a window -- Agg (this file's own earlier
    # choice) is non-interactive and silently no-ops on show(), which was why nothing appeared.
    import matplotlib.pyplot as plt

    fig = plot_trajectory_fit(
        fwd, lat, path, vx_spline, s_max_wp, s_mid, v_seg, vx_preview, kappa_preview, DT, speed,
        title=f"idx={idx}  speed={speed:.2f} m/s  folder={r.get('folder', '')}")

    os.makedirs(args.plot_dir, exist_ok=True)
    out_path = args.out or os.path.join(args.plot_dir, f"inspect_idx{idx}.png")
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"\nsaved plot -> {out_path}")
    import matplotlib
    print(f"[debug] backend={matplotlib.get_backend()}  interactive={matplotlib.is_interactive()}  "
         f"open_figs={plt.get_fignums()}  DISPLAY={os.environ.get('DISPLAY')}")
    plt.show(block=True)   # explicit block=True: default (None) auto-detection has been the one
                           # unverified difference from the known-working bare TkAgg repro
    print("[debug] plt.show() returned")


if __name__ == "__main__":
    main()
