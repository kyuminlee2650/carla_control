r"""Deep-dive on ONE real VAD trajectory sample: prints the raw waypoints, the actual fitted spline
(knots + per-segment cubic polynomial coefficients, not just a plot), and the resulting vx/kappa
preview -- validate_trajectory_fit.py only reports pass/fail + aggregate stats across the whole
batch; this is the single-sample "how did this number actually come out" companion to it.

Usage (Windows, from this file's own directory -- .venv lives one level up in carla_control/, so
invoke its python.exe directly rather than relying on `python` being on PATH):
    cd C:\Users\mumu2\carla_control\b2d_controller
    ..\.venv\Scripts\python.exe inspect_one_sample.py                  # random sample
    ..\.venv\Scripts\python.exe inspect_one_sample.py --idx 196         # a specific one (e.g. the
                                                     # sharpest-curvature sample
                                                     # validate_trajectory_fit.py's summary flagged
                                                     # as the max|kappa| case)
    ..\.venv\Scripts\python.exe inspect_one_sample.py --idx 196 --seed 0
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np
from scipy.interpolate import UnivariateSpline

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

from mpc_kf_controller import (DT, VAD_WP_DT, MpcKfController, _chord_length_station,
                               build_trajectory_splines, preview_from_splines)
from viz_utils import plot_trajectory_fit


def print_spline_formula(name, spl, unit="", xvar="s"):
    """Per-segment cubic Taylor expansion f(xvar) = c0 + c1*u + c2*u^2 + c3*u^3, u = xvar - xvar_i,
    read off the spline's own derivative evaluations at each knot -- public API only (spl(x, nu=n)),
    not the private FITPACK tck tuple, so this works regardless of scipy version internals. xvar
    names the independent variable ("s" for the station-parameterized fits, "t" for s(t))."""
    knots = spl.get_knots()
    print(f"  {name}: {len(knots)} knots (spline pieces: {len(knots) - 1}), "
         f"residual={spl.get_residual():.4g}")
    print(f"    knots ({xvar}): {np.round(knots, 3).tolist()}")
    eps = 1e-6
    for i in range(len(knots) - 1):
        s0, s1 = knots[i], knots[i + 1]
        s_eval = min(s0 + eps, (s0 + s1) / 2)
        c0 = float(spl(s_eval, 0))
        c1 = float(spl(s_eval, 1))
        c2 = float(spl(s_eval, 2)) / 2.0
        c3 = float(spl(s_eval, 3)) / 6.0
        print(f"    {xvar} in [{s0:6.3f}, {s1:6.3f}]:  f({xvar}) = {c0:+.4f} {c1:+.4f}*u {c2:+.5f}*u^2 "
             f"{c3:+.6f}*u^3   (u = {xvar} - {s0:.3f}){unit}")


def print_derived_formula(name, formula, s_values, values, unit=""):
    """yaw(s)/kappa(s) aren't independently fit splines -- they're analytic combinations of X(s)'s
    and Y(s)'s own derivatives (see PathSpline.yaw()/kappa() in mpc_kf_controller.py), and X(s)/Y(s)
    are fit as two separate UnivariateSplines whose knots aren't guaranteed to coincide (FITPACK
    places knots per-spline from each one's own residual structure, not just from the shared input
    stations) -- so a single combined per-segment Taylor expansion the way print_spline_formula()
    does for one spline isn't well-defined here. Print the defining formula once instead, then its
    value at each real waypoint's own station."""
    print(f"  {name}: {formula}")
    label = name.split("(")[0].strip()
    for s, v in zip(s_values, values):
        print(f"    s={s:6.3f} m   {label}={v:+9.5f}{unit}")


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
    waypoints, speed = r["out_truck"], r["speed"]
    r_meas = float(r["angular_velocity"][2])   # yaw rate, rad/s -- tick_data['angular_velocity'][2]
    ay_meas = float(r["acceleration"][1])       # lateral accel, m/s^2 -- tick_data['acceleration'][1]
    a_meas = float(r["acceleration"][0])        # longitudinal accel, m/s^2 -- tick_data['acceleration'][0]
    # control_mpc() requires gear now (no no-telemetry fallback, see mpc_kf_controller.py module
    # docstring point 5); this offline dataset has no real gear channel, so this is a speed-only
    # placeholder purely to exercise LookupController.step() -- see validate_mpc_solve.py's own
    # _placeholder_gear() for the same reasoning.
    gear = next((g for top_speed, g in ((3, 1), (7, 2), (12, 3), (18, 4), (25, 5)) if speed < top_speed), 6)

    print(f"=== sample idx={idx}  (dataset entry idx={r.get('idx')}, folder={r.get('folder', '?')}) ===")
    print(f"current speed: {speed:.3f} m/s\n")

    print(f"raw VAD waypoints, [lateral, forward] m, {VAD_WP_DT}s apart (this file's own convention "
         f"-- forward is index 1, confirmed via pid_controller.py's atan2 formula):")
    print(f"  origin (t=0.0s): lateral= 0.000  forward= 0.000   <- ego, always the fit's own origin")
    for i, wp in enumerate(waypoints):
        print(f"  wp{i} (t={(i + 1) * VAD_WP_DT:.1f}s): lateral={wp[0]:+.4f}  forward={wp[1]:+.4f}")
    print()

    # Reproduce build_trajectory_splines()'s own intermediate arrays so we can print/inspect them,
    # not just call the function as a black box.
    fwd = [0.0] + [wp[1] for wp in waypoints]
    lat = [0.0] + [wp[0] for wp in waypoints]
    s_all = _chord_length_station(fwd, lat)
    print(f"chord-length station s for [origin, wp0..wp{len(waypoints) - 1}]:")
    print(f"  s = {np.round(s_all, 3).tolist()}  (m)\n")

    n_wp = len(waypoints)
    s_wp = s_all[:n_wp + 1]
    t_wp = np.arange(n_wp + 1) * VAD_WP_DT

    # s(t): diagnostic-only fit -- build_trajectory_splines()/the real controller never build this;
    # preview_from_splines() instead walks a station cursor forward via vx(s)*dt each step (see its
    # own docstring). Built here purely to show what "predicted arc-length position vs. time" looks
    # like as an explicit formula, off the same (t_wp, s_wp) pairs vx(s)'s own fit is built from.
    s_of_t = UnivariateSpline(t_wp, s_wp, k=min(3, len(t_wp) - 1), s=0.05)
    print_spline_formula("s(t) [m]", s_of_t, xvar="t")
    print()

    path, vx_spline, s_max_wp = build_trajectory_splines(waypoints, speed)
    print(f"PathSpline geometry fit -- x(s) [forward] and y(s) [lateral], smoothing=0.05, k=3:")
    print_spline_formula("x(s) [forward, m]", path._sx)
    print_spline_formula("y(s) [lateral, m]", path._sy)
    print()

    ds, dt = np.diff(s_wp), np.diff(t_wp)
    # Same anchor build_trajectory_splines() adds internally -- see its docstring: without it,
    # vx_spline(0) is only VAD's own implied average speed over the first interval, not the actual
    # measured current speed.
    v_seg = np.concatenate([[float(speed)], ds / np.maximum(dt, 1e-3)])
    s_mid = np.concatenate([[0.0], s_wp[:-1] + ds / 2.0])
    print(f"vx(s) spline input -- measured current speed anchored at s=0, then per-interval average "
         f"speed ds/dt assigned to each interval's own arc-length midpoint (s_max_wp={s_max_wp:.3f} m "
         f"is the last real waypoint's station; vx_spline is only data-informed up to here):")
    for sm, v in zip(s_mid, v_seg):
        print(f"    s_mid={sm:6.3f} m   v={v:5.2f} m/s")
    print_spline_formula("vx(s) [m/s]", vx_spline)
    print()

    print_derived_formula("yaw(s)", "atan2(Y'(s), X'(s))  [rad; printed here in deg]",
                          s_wp, np.degrees(path.yaw(s_wp)), " deg")
    print()
    print_derived_formula("kappa(s)", "(X'(s)*Y''(s) - Y'(s)*X''(s)) / (X'(s)^2 + Y'(s)^2)^1.5",
                          s_wp, path.kappa(s_wp), " 1/m")
    print()

    vx_preview, kappa_preview = preview_from_splines(path, vx_spline, s_max_wp, args.n_p, DT)
    s_cursor = np.concatenate([[0.0], np.cumsum(vx_preview * DT)[:-1]])
    print(f"preview_from_splines() walk (n_p={args.n_p}, dt={DT}s) -- station cursor advances by "
         f"vx(s_cursor)*dt each step, both vx and kappa sampled off the SAME station:")
    print(f"  {'step':>4}  {'s (m)':>7}  {'vx (m/s)':>9}  {'kappa (1/m)':>12}")
    for j in range(args.n_p):
        print(f"  {j:4d}  {s_cursor[j]:7.3f}  {vx_preview[j]:9.3f}  {kappa_preview[j]:+12.5f}")
    print()

    # MpcKfController.control_mpc() end-to-end on this one sample (spline fit + KF step +
    # LateralMPC.solve(), the same QP carla_control/mpc_mpc_comparison.py's own LateralMPC class
    # derives -- mpc_kf_controller.py's copy is unchanged math) -- a FRESH instance, since this one
    # sample is an independent frame, not a continuation of whatever the last-inspected sample was;
    # see validate_mpc_solve.py's own docstring for why the batch check does the same per sample.
    controller = MpcKfController()
    steer, throttle, brake, metadata = controller.control_mpc(
        waypoints, speed, r_meas, ay_meas, gear, a_meas)
    print(f"MpcKfController.control_mpc() -- fresh instance, LateralMPC.solve() + SpeedMPC.solve() result:")
    print(f"  LateralMPC OSQP status: {metadata['kf_status']!r}   "
         f"SpeedMPC OSQP status: {metadata['speed_mpc_status']!r}")
    print(f"  steer={steer:+.4f}  delta={math.degrees(metadata['delta_rad']):+.2f} deg  "
         f"v_y_hat={metadata['v_y_hat']:+.3f} m/s")
    print(f"  throttle={throttle:.4f}  brake={brake:.4f}  a_cmd={metadata['a_cmd']:+.3f} m/s^2  "
         f"gear={metadata['gear']} (placeholder) u_raw={metadata['u_raw']:+.3f} "
         f"u_filtered={metadata['u_filtered']:+.3f} saturated={metadata['pedal_saturated']}")

    # ---- plot (viz_utils.plot_trajectory_fit -- shared COLOR_*/LINEWIDTH/FONTSIZE_* knobs) ----
    # No matplotlib.use() here: viz_utils.py already sets TkAgg at import time (above), which is
    # what makes the plt.show() below actually pop up a window -- Agg (this file's own earlier
    # choice) is non-interactive and silently no-ops on show(), which was why nothing appeared.
    import matplotlib.pyplot as plt

    fig = plot_trajectory_fit(
        fwd, lat, path, vx_spline, s_max_wp, s_mid, v_seg, vx_preview, kappa_preview, DT, speed,
        title=f"idx={idx}  speed={speed:.2f} m/s \n folder={r.get('folder', '')}")

    os.makedirs(args.plot_dir, exist_ok=True)
    out_path = args.out or os.path.join(args.plot_dir, f"inspect_idx{idx}.png")
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"\nsaved plot -> {out_path}")
    import matplotlib
    plt.show(block=True)   



if __name__ == "__main__":
    main()
