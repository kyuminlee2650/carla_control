"""Invert build_longitudinal_lut.py's raw sweep data into a per-gear
(v_x, a_x) -> u grid, saved as an .npz for LongitudinalLUT (longitudinal_lut.py)
to load at runtime.

For each gear: fits a scattered interpolator over the raw (v_x, a_x) -> u
samples, then evaluates it on a regular grid so the runtime lookup can use
a fast RegularGridInterpolator instead of re-triangulating scattered data
on every MPC step. Points outside the sampled envelope (no raw data nearby,
e.g. an (v, a) combo that gear/throttle combination never reached) fall
back to the nearest observed sample rather than extrapolating.

Usage:
    python3 build_reverse_lut.py --in longitudinal_lut.csv --out longitudinal_lut.npz
"""

import argparse

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


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
    parser.add_argument("--in", dest="in_path", default="longitudinal_lut.csv")
    parser.add_argument("--out", dest="out_path", default="longitudinal_lut.npz")
    parser.add_argument("--v-res", type=int, default=60, help="v_x grid points per gear")
    parser.add_argument("--a-res", type=int, default=60, help="a_x grid points per gear")
    parser.add_argument("--max-abs-accel", type=float, default=12.0,
                         help="drop raw samples with |a_x| above this (m/s^2); filters the "
                              "single-tick launch/lockup transient spikes build_longitudinal_lut.py "
                              "logs at the start of full-throttle and full-brake trials")
    args = parser.parse_args()

    raw = np.genfromtxt(args.in_path, delimiter=",", names=True)
    raw = raw[np.abs(raw["a_x"]) <= args.max_abs_accel]
    gears = sorted(int(g) for g in set(raw["gear"]))

    tables = {}
    for gear in gears:
        mask = raw["gear"] == gear
        v_x, a_x, u = raw["v_x"][mask], raw["a_x"][mask], raw["u"][mask]
        v_grid, a_grid, u_table = build_gear_table(v_x, a_x, u, args.v_res, args.a_res)
        tables[f"g{gear}_v"] = v_grid
        tables[f"g{gear}_a"] = a_grid
        tables[f"g{gear}_u"] = u_table
        print(f"gear {gear}: v in [{v_grid[0]:.2f}, {v_grid[-1]:.2f}] m/s, "
              f"a in [{a_grid[0]:.2f}, {a_grid[-1]:.2f}] m/s^2, {mask.sum()} raw samples")

    np.savez(args.out_path, gears=np.array(gears), **tables)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()
