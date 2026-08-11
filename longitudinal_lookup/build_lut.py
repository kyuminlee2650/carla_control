"""Longitudinal control-input lookup table -- data cleaning + fitting.

Turns collect_lut_data.py's raw sweep CSV into a per-gear (v_x, a_x) -> u grid, saved as an .npz
for LongitudinalLUT (longitudinal_lut.py) to load at runtime.

Cleaning: a gear that is lugging -- high gear at low speed -- barely responds to throttle, and
what the sweep recorded there is dominated by driveline oscillation rather than by u. Inverting
(v, a) -> u over that data yields nonsense, so trusted_speed_mask() drops those speed bins before
fitting (see its docstring) -- every bin that individually passes the repeatability check is kept,
not just the longest consecutive run, so a noisy patch in the middle of a gear's range doesn't
take clean data elsewhere down with it. Raw samples with an unrealistic |a_x| (single-tick
launch/lockup transients) are dropped first.

Fitting: for each gear, fits a scattered interpolator over the cleaned (v_x, a_x) -> u samples,
then evaluates it on a regular grid so the runtime lookup can use a fast RegularGridInterpolator
instead of re-triangulating scattered data on every call. Points outside the sampled envelope fall
back to the nearest observed sample rather than extrapolating.

A per-gear (v_x, a_x) -> u 3D surface, with the raw samples (including whatever the trust cut
excluded) scattered on top, is saved alongside the table when --save-plot is passed.

Usage (Ubuntu):
    cd ~/carla_control
    python3 longitudinal_lookup/build_lut.py --save-plot

Usage (Windows):
    cd C:\\Users\\mumu2\\carla_control
    .venv\\Scripts\\python.exe longitudinal_lookup\\build_lut.py
    .venv\\Scripts\\python.exe longitudinal_lookup\\build_lut.py --save-plot
"""

import argparse
import math
import os
import sys

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
# viz_utils.py lives one level up, alongside the path-tracking controllers
sys.path.append(os.path.dirname(HERE))


def trusted_speed_mask(v_x, a_x, u, threshold, bin_step, bin_tol, min_group, min_groups):
    """Boolean mask over v_x/a_x/u selecting every speed bin where a gear's recorded acceleration
    is repeatable -- not just the longest run of consecutive good bins.

    Noise is measured as the median, over the control inputs tested at a given speed, of the
    spread of a_x within one input. Low spread means "same input, same acceleration" -- the
    property the inversion depends on. A noisy patch in the middle of a gear's range (e.g. one
    resonant speed) shouldn't disqualify clean data well outside it, so every bin that passes on
    its own is kept, even if it isn't contiguous with the others.
    """
    lo_v, hi_v = float(v_x.min()), float(v_x.max())
    n_bins = max(1, int(math.floor((hi_v - lo_v) / bin_step)) + 1)
    centers = [lo_v + i * bin_step for i in range(n_bins)]
    u_values = np.unique(np.round(u, 2))

    good_centers = []
    for c in centers:
        near = np.abs(v_x - c) <= bin_tol
        spreads = [a_x[near & (np.abs(u - uv) < 0.01)].std()
                   for uv in u_values
                   if (near & (np.abs(u - uv) < 0.01)).sum() >= min_group]
        if len(spreads) >= min_groups and float(np.median(spreads)) <= threshold:
            good_centers.append(c)

    if not good_centers:
        return None

    mask = np.zeros(v_x.shape, dtype=bool)
    for c in good_centers:
        mask |= np.abs(v_x - c) <= bin_tol
    return mask


def build_gear_table(v_x, a_x, u, v_res, a_res):
    points = np.column_stack([v_x, a_x])
    linear = LinearNDInterpolator(points, u)
    nearest = NearestNDInterpolator(points, u)

    v_grid = np.linspace(v_x.min(), v_x.max(), v_res)
    a_grid = np.linspace(a_x.min(), a_x.max(), a_res)
    vv, aa = np.meshgrid(v_grid, a_grid, indexing="ij")
    query = np.column_stack([vv.ravel(), aa.ravel()])

    u_table = linear(query)
    missing = np.isnan(u_table)
    if missing.any():
        u_table[missing] = nearest(query[missing])
    u_table = np.clip(u_table, -1.0, 1.0).reshape(vv.shape)

    return v_grid, a_grid, u_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=os.path.join(HERE, "longitudinal_lut.csv"))
    parser.add_argument("--out", dest="out_path", default=os.path.join(HERE, "longitudinal_lut.npz"))
    parser.add_argument("--v-res", type=int, default=60, help="v_x grid points per gear")
    parser.add_argument("--a-res", type=int, default=60, help="a_x grid points per gear")
    parser.add_argument("--max-abs-accel", type=float, default=12.0,
                         help="drop raw samples with |a_x| above this (m/s^2); filters the "
                              "single-tick launch/lockup transient spikes collect_lut_data.py "
                              "logs at the start of full-throttle and full-brake trials")
    parser.add_argument("--trust-threshold", type=float, default=1.0,
                        help="max acceleration spread within one control input (m/s^2) for a speed "
                             "to be considered repeatable; set to 0 to disable the speed cut")
    parser.add_argument("--trust-bin-step", type=float, default=2.0, help="speed bin spacing for the noise scan (m/s)")
    parser.add_argument("--trust-bin-tol", type=float, default=1.0, help="half-width of each speed bin (m/s)")
    parser.add_argument("--trust-min-group", type=int, default=8, help="samples needed at one control input to measure its spread")
    parser.add_argument("--trust-min-groups", type=int, default=2, help="control inputs needed at a speed to judge it")
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"), help="where the LUT surface figure goes")
    parser.add_argument("--save-plot", action="store_true", help="draw and save a 3D (v_x, a_x) -> u surface per gear")
    parser.add_argument("--no-raw-overlay", action="store_true", help="don't scatter the cleaned samples onto the surface plot")
    parser.add_argument("--elev", type=float, default=25.0, help="surface plot camera elevation (deg)")
    parser.add_argument("--azim", type=float, default=-60.0, help="surface plot camera azimuth (deg)")
    args = parser.parse_args()

    raw_all = np.genfromtxt(args.in_path, delimiter=",", names=True)
    raw = raw_all[np.abs(raw_all["a_x"]) <= args.max_abs_accel]
    gears = sorted(int(g) for g in set(raw["gear"]))

    tables = {}
    gear_tables = {}
    for gear in gears:
        gear_mask = raw["gear"] == gear
        v_x, a_x, u = raw["v_x"][gear_mask], raw["a_x"][gear_mask], raw["u"][gear_mask]
        n_before = int(gear_mask.sum())

        if args.trust_threshold > 0:
            trust_mask = trusted_speed_mask(v_x, a_x, u, args.trust_threshold, args.trust_bin_step,
                                            args.trust_bin_tol, args.trust_min_group, args.trust_min_groups)
            if trust_mask is None:
                print(f"gear {gear}: no trustworthy speed bins found, skipping "
                      f"(raise --trust-threshold to include it)")
                continue
            v_x, a_x, u = v_x[trust_mask], a_x[trust_mask], u[trust_mask]

        v_grid, a_grid, u_table = build_gear_table(v_x, a_x, u, args.v_res, args.a_res)
        tables[f"g{gear}_v"] = v_grid
        tables[f"g{gear}_a"] = a_grid
        tables[f"g{gear}_u"] = u_table
        tables[f"g{gear}_vrange"] = np.array((v_grid[0], v_grid[-1]))
        gear_tables[gear] = (v_grid, a_grid, u_table)
        cut = f", {n_before - v_x.size} dropped as untrustworthy" if args.trust_threshold > 0 else ""
        print(f"gear {gear}: v in [{v_grid[0]:.2f}, {v_grid[-1]:.2f}] m/s, "
              f"a in [{a_grid[0]:.2f}, {a_grid[-1]:.2f}] m/s^2, {v_x.size} raw samples{cut}")

    if not tables:
        raise SystemExit("Every gear was excluded. Try raising --trust-threshold.")

    kept_gears = sorted({int(k[1:].split("_")[0]) for k in tables})
    np.savez(args.out_path, gears=np.array(kept_gears), **tables)
    print(f"Saved {args.out_path}")

    if args.save_plot:
        from viz_utils import plot_lut_surfaces
        plot_lut_surfaces(gear_tables, args.plot_dir, raw=None if args.no_raw_overlay else raw,
                          elev=args.elev, azim=args.azim)


if __name__ == "__main__":
    main()
