"""Invert build_longitudinal_lut.py's raw sweep data into a per-gear
(v_x, a_x) -> u grid, saved as an .npz for LongitudinalLUT (longitudinal_lut.py)
to load at runtime.

For each gear: fits a scattered interpolator over the raw (v_x, a_x) -> u
samples, then evaluates it on a regular grid so the runtime lookup can use
a fast RegularGridInterpolator instead of re-triangulating scattered data
on every MPC step. Points outside the sampled envelope (no raw data nearby,
e.g. an (v, a) combo that gear/throttle combination never reached) fall
back to the nearest observed sample rather than extrapolating.

Paths default to this script's own directory, so it runs from anywhere.

Usage:
    cd ~/carla_control
    .venv/bin/python longitudinal_lookup/build_reverse_lut.py
"""

import argparse
import math
import os

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# defaults resolve next to this script, so it runs from any working directory
HERE = os.path.dirname(os.path.abspath(__file__))


def trusted_speed_range(v_x, a_x, u, threshold, bin_step, bin_tol, min_group, min_groups):
    """Speed window where a gear's recorded acceleration is repeatable.

    A gear that is lugging -- high gear at low speed -- barely responds to
    throttle, and what the sweep recorded there is dominated by driveline
    oscillation rather than by u. Inverting (v, a) -> u over that data yields
    nonsense (validation found the table asking for throttle to decelerate), so
    those speeds are cut before the table is fitted.

    Noise is measured as the median, over the control inputs tested at a given
    speed, of the spread of a_x within one input. Low spread means "same input,
    same acceleration" -- the property the inversion depends on. The returned
    window is the longest run of consecutive speed bins under the threshold.
    """
    lo_v, hi_v = float(v_x.min()), float(v_x.max())
    n_bins = max(1, int(math.floor((hi_v - lo_v) / bin_step)) + 1)
    centers = [lo_v + i * bin_step for i in range(n_bins)]
    u_values = np.unique(np.round(u, 2))

    good = []
    for c in centers:
        near = np.abs(v_x - c) <= bin_tol
        spreads = [a_x[near & (np.abs(u - uv) < 0.01)].std()
                   for uv in u_values
                   if (near & (np.abs(u - uv) < 0.01)).sum() >= min_group]
        if len(spreads) >= min_groups and float(np.median(spreads)) <= threshold:
            good.append(c)

    if not good:
        return None

    runs, current = [], [good[0]]
    for c in good[1:]:
        if c - current[-1] <= bin_step * 1.5:
            current.append(c)
        else:
            runs.append(current)
            current = [c]
    runs.append(current)
    best = max(runs, key=len)
    return best[0] - bin_tol, best[-1] + bin_tol


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
                              "single-tick launch/lockup transient spikes build_longitudinal_lut.py "
                              "logs at the start of full-throttle and full-brake trials")
    parser.add_argument("--trust-threshold", type=float, default=1.0,
                        help="max acceleration spread within one control input (m/s^2) for a speed "
                             "to be considered repeatable; set to 0 to disable the speed cut")
    parser.add_argument("--trust-bin-step", type=float, default=2.0, help="speed bin spacing for the noise scan (m/s)")
    parser.add_argument("--trust-bin-tol", type=float, default=1.0, help="half-width of each speed bin (m/s)")
    parser.add_argument("--trust-min-group", type=int, default=8, help="samples needed at one control input to measure its spread")
    parser.add_argument("--trust-min-groups", type=int, default=2, help="control inputs needed at a speed to judge it")
    args = parser.parse_args()

    raw = np.genfromtxt(args.in_path, delimiter=",", names=True)
    raw = raw[np.abs(raw["a_x"]) <= args.max_abs_accel]
    gears = sorted(int(g) for g in set(raw["gear"]))

    tables = {}
    for gear in gears:
        mask = raw["gear"] == gear
        v_x, a_x, u = raw["v_x"][mask], raw["a_x"][mask], raw["u"][mask]
        n_before = int(mask.sum())

        v_range = None
        if args.trust_threshold > 0:
            v_range = trusted_speed_range(v_x, a_x, u, args.trust_threshold, args.trust_bin_step,
                                          args.trust_bin_tol, args.trust_min_group, args.trust_min_groups)
            if v_range is None:
                print(f"gear {gear}: 신뢰 가능한 속도 구간을 찾지 못해 건너뜀 "
                      f"(--trust-threshold 를 올리면 포함됩니다)")
                continue
            keep = (v_x >= v_range[0]) & (v_x <= v_range[1])
            v_x, a_x, u = v_x[keep], a_x[keep], u[keep]

        v_grid, a_grid, u_table = build_gear_table(v_x, a_x, u, args.v_res, args.a_res)
        tables[f"g{gear}_v"] = v_grid
        tables[f"g{gear}_a"] = a_grid
        tables[f"g{gear}_u"] = u_table
        tables[f"g{gear}_vrange"] = np.array(v_range if v_range else (v_grid[0], v_grid[-1]))
        cut = f", 신뢰 구간 밖 {n_before - v_x.size}개 제외" if args.trust_threshold > 0 else ""
        print(f"gear {gear}: v in [{v_grid[0]:.2f}, {v_grid[-1]:.2f}] m/s, "
              f"a in [{a_grid[0]:.2f}, {a_grid[-1]:.2f}] m/s^2, {v_x.size} raw samples{cut}")

    if not tables:
        raise SystemExit("모든 기어가 제외되었습니다. --trust-threshold 를 올려 보세요.")

    kept_gears = sorted({int(k[1:].split("_")[0]) for k in tables})
    np.savez(args.out_path, gears=np.array(kept_gears), **tables)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()
