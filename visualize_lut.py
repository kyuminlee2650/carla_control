"""3D surface visualization of the longitudinal control-input lookup table.

One (v_x, a_x) -> u surface per gear, laid out as small multiples with a
shared color scale so gears stay visually comparable. u is a diverging
quantity (negative = brake, positive = throttle, zero = neither), so the
color scale is diverging around zero rather than a plain sequential ramp.

Usage:
    python3 visualize_lut.py --npz longitudinal_lut.npz --out lut_surfaces.png
    python3 visualize_lut.py --npz longitudinal_lut.npz --raw-csv longitudinal_lut.csv --show
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projection

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# diverging: blue (brake) <-> gray (neither) <-> red (throttle)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "brake_throttle", ["#2a78d6", "#f0efec", "#e34948"]
)


def style_3d_axes(ax):
    ax.set_facecolor(SURFACE)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(SURFACE)
        pane.set_edgecolor(GRIDLINE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = GRIDLINE
        axis.label.set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="longitudinal_lut.npz")
    parser.add_argument("--raw-csv", default=None, help="overlay raw sweep samples as scatter points")
    parser.add_argument("--out", default="lut_surfaces.png")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--max-abs-accel", type=float, default=12.0,
                         help="drop |a_x| above this when overlaying raw samples, so the same "
                              "launch/lockup transient spikes build_reverse_lut.py filters out "
                              "don't blow up the axis range here too")
    args = parser.parse_args()

    data = np.load(args.npz)
    gears = [int(g) for g in data["gears"]]

    raw = None
    if args.raw_csv:
        raw = np.genfromtxt(args.raw_csv, delimiter=",", names=True)
        raw = raw[np.abs(raw["a_x"]) <= args.max_abs_accel]

    ncols = min(3, len(gears))
    nrows = math.ceil(len(gears) / ncols)
    fig = plt.figure(figsize=(5.2 * ncols, 4.6 * nrows), facecolor=SURFACE)
    fig.suptitle("Longitudinal control-input lookup table  (u: brake -1 -> throttle +1)",
                 color=INK_PRIMARY, fontsize=13)

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    mappable = None

    for i, gear in enumerate(gears):
        v_grid = data[f"g{gear}_v"]
        a_grid = data[f"g{gear}_a"]
        u_table = data[f"g{gear}_u"]
        vv, aa = np.meshgrid(v_grid, a_grid, indexing="ij")

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        mappable = ax.plot_surface(
            vv, aa, u_table, cmap=DIVERGING_CMAP, norm=norm,
            linewidth=0, antialiased=True, alpha=0.92,
        )

        if raw is not None:
            mask = raw["gear"] == gear
            ax.scatter(raw["v_x"][mask], raw["a_x"][mask], raw["u"][mask],
                       s=4, color=INK_PRIMARY, alpha=0.25, depthshade=False)

        ax.set_title(f"Gear {gear}", color=INK_PRIMARY, fontsize=11)
        ax.set_xlabel("v_x (m/s)")
        ax.set_ylabel("a_x (m/s²)")
        ax.set_zlabel("u")
        ax.set_zlim(-1, 1)
        ax.view_init(elev=args.elev, azim=args.azim)
        style_3d_axes(ax)

    cbar = fig.colorbar(mappable, ax=fig.get_axes(), shrink=0.6, pad=0.02,
                         label="u  (brake ←  0  → throttle)")
    cbar.ax.yaxis.label.set_color(INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)

    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
