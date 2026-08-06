"""Time-series figure for validate_accel_tracking.py.

The grid test's scatter plot cannot show a continuous run: what matters here is
how the response sits against the reference over time -- phase lag, overshoot,
where the input rails, what the gear is doing.
"""

import datetime
import math
import os

import numpy as np

import matplotlib
matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
import matplotlib.pyplot as plt

COLOR_BG = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
COLOR_AQUA = "#1baf7a"
COLOR_RED = "#e34948"


def _style(ax):
    ax.set_facecolor(COLOR_BG)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    ax.title.set_color(COLOR_INK)
    ax.xaxis.label.set_color(COLOR_MUTED)
    ax.yaxis.label.set_color(COLOR_MUTED)


def _smooth(x, n=5):
    return np.convolve(x, np.ones(n) / n, mode="same") if n > 1 else x


def plot_tracking_grid(cases, args, compare=None, show=True):
    """Small multiples: one panel per (gear, initial speed) run.

    Reference and response share an axis so phase lag and amplitude loss are
    both visible; the control input rides on a twin axis because saturation is
    what explains most large errors.
    """
    compare_cases = {}
    if compare is not None:
        for r in compare:
            compare_cases.setdefault((int(r["case_gear"]), float(r["case_v0"])), []).append(r)

    # rows = gears, columns = initial speeds, so a whole gear reads across and a
    # whole speed reads down; combinations the LUT does not cover stay blank
    by_key = dict(cases)
    all_gears = sorted({g for g, _ in by_key})
    per_fig = max(1, getattr(args, "gears_per_figure", 3))
    chunks = [all_gears[i:i + per_fig] for i in range(0, len(all_gears), per_fig)]
    paths = []
    for part, chunk in enumerate(chunks, start=1):
        paths.append(_plot_chunk(by_key, chunk, compare_cases, args, show,
                                 part, len(chunks)))
    return paths


def _plot_chunk(by_key, gears, compare_cases, args, show, part, n_parts):
    speeds = sorted({v for g, v in by_key if g in gears})
    n_rows, n_cols = len(gears), len(speeds)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.0 * n_rows),
                             squeeze=False, sharex=True, sharey="row", constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)

    first_panel = True
    for row, gear in enumerate(gears):
        for col, v0 in enumerate(speeds):
            ax = axes[row][col]
            if (gear, v0) not in by_key:
                ax.axis("off")
                ax.text(0.5, 0.5, "outside\ncalibrated range", transform=ax.transAxes,
                        ha="center", va="center", color=COLOR_MUTED, fontsize=10)
                continue
            case = by_key[(gear, v0)]
            _style(ax)
            t = np.array([r["t"] for r in case])
            a_ref = np.array([r["a_ref"] for r in case])
            a_meas = _smooth(np.array([r["a_fd"] for r in case]))
            u = np.array([r["u"] for r in case])
            err = a_meas - a_ref

            ax_u = ax.twinx()
            ax_u.plot(t, u, color=COLOR_MUTED, linewidth=0.9, alpha=0.5)
            ax_u.set_ylim(-1.1, 1.1)
            ax_u.set_yticks([])
            rail = np.abs(u) >= 0.999
            if rail.any():
                ax_u.scatter(t[rail], u[rail], color=COLOR_RED, s=5, zorder=5)

            ax.plot(t, a_ref, color=COLOR_MUTED, linewidth=2.2, linestyle="--", label="reference")
            if (gear, v0) in compare_cases:
                ct = np.array([r["t"] for r in compare_cases[(gear, v0)]])
                ca = _smooth(np.array([r["a_fd"] for r in compare_cases[(gear, v0)]]))
                ax.plot(ct, ca, color=COLOR_AQUA, linewidth=1.3, alpha=0.9, label="feedforward only")
            ax.plot(t, a_meas, color=COLOR_BLUE, linewidth=1.7, label="measured")
            ax.set_zorder(ax_u.get_zorder() + 1)
            ax.patch.set_visible(False)

            label = f"MAE {np.mean(np.abs(err)):.3f}"
            if rail.mean() > 0.01:
                label += f"   saturated {100*rail.mean():.0f}%"
            ax.text(0.02, 0.96, label, transform=ax.transAxes, va="top", ha="left",
                    fontsize=9, color=COLOR_INK)

            if row == 0:
                ax.set_title(f"$v_0$ = {v0:.0f} m/s", fontsize=11)
            if row == n_rows - 1:
                ax.set_xlabel("t (s)")
            if col == 0:
                ax.set_ylabel(f"gear {gear if gear else 'auto'}\n$a$ (m/s$^2$)")
            if first_panel:
                ax.legend(frameon=False, labelcolor=COLOR_INK, fontsize=8, loc="lower right")
                first_panel = False

    mode = f"feedforward + PI (kp={args.kp}, ki={args.ki})" if args.pi else "feedforward only"
    part_label = f"  —  gears {gears[0]}–{gears[-1]} ({part}/{n_parts})" if n_parts > 1 else ""
    fig.suptitle(f"Acceleration profile tracking — {args.profile}, amplitude {args.amplitude} m/s², "
                 f"jerk limit {args.jerk_max} m/s³ — {mode}{part_label}\n"
                 f"grey trace = control input u (red = saturated)",
                 color=COLOR_INK, fontsize=13)

    os.makedirs(args.plot_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_g{gears[0]}-{gears[-1]}" if n_parts > 1 else ""
    out_path = os.path.join(args.plot_dir, f"accel_tracking_{stamp}{suffix}.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\n그래프 저장: {out_path}")

    if show and matplotlib.get_backend().lower() != "agg":
        print("창을 닫으면 종료됩니다.")
        plt.show()
    plt.close(fig)
    return out_path


def plot_tracking(rows, args, compare=None, show=True):
    t = np.array([r["t"] for r in rows])
    a_ref = np.array([r["a_ref"] for r in rows])
    a_meas = _smooth(np.array([r["a_fd"] for r in rows]))
    v = np.array([r["v"] for r in rows])
    u = np.array([r["u"] for r in rows])
    gear = np.array([r["gear"] for r in rows])
    err = a_meas - a_ref

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    ax_a, ax_e, ax_u, ax_v = axes

    _style(ax_a)
    ax_a.plot(t, a_ref, color=COLOR_MUTED, linewidth=2.4, linestyle="--", label="reference")
    if compare is not None:
        ct = np.array([r["t"] for r in compare])
        ca = _smooth(np.array([r["a_fd"] for r in compare]))
        ax_a.plot(ct, ca, color=COLOR_AQUA, linewidth=1.6, alpha=0.85, label="compare run")
    ax_a.plot(t, a_meas, color=COLOR_BLUE, linewidth=1.8, label="measured")
    ax_a.set_ylabel("acceleration (m/s$^2$)")
    ax_a.set_title("Commanded vs. achieved longitudinal acceleration")
    ax_a.legend(frameon=False, labelcolor=COLOR_INK, fontsize=9, ncol=3)

    _style(ax_e)
    ax_e.axhline(0.0, color=COLOR_MUTED, linewidth=1.2, linestyle="--")
    ax_e.plot(t, err, color=COLOR_RED, linewidth=1.4)
    ax_e.fill_between(t, err, 0, color=COLOR_RED, alpha=0.15)
    ax_e.set_ylabel("error (m/s$^2$)")
    ax_e.set_title(f"Tracking error   (MAE {np.mean(np.abs(err)):.3f}, "
                   f"RMSE {np.sqrt(np.mean(err**2)):.3f}, max {np.abs(err).max():.2f})")

    _style(ax_u)
    ax_u.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_u.plot(t, u, color=COLOR_ORANGE, linewidth=1.4, label="control input $u$")
    rail = np.abs(u) >= 0.999
    if rail.any():
        ax_u.scatter(t[rail], u[rail], color=COLOR_RED, s=10, zorder=5, label="saturated")
    ax_u.set_ylabel("$u$  (throttle $-$ brake)")
    ax_u.set_ylim(-1.15, 1.15)
    ax_u.set_title("Control input")
    ax_u.legend(frameon=False, labelcolor=COLOR_INK, fontsize=9, ncol=2)

    _style(ax_v)
    ax_v.plot(t, v, color=COLOR_BLUE, linewidth=1.8, label="speed")
    ax_v.set_ylabel("speed (m/s)")
    ax_v.set_xlabel("t (s)")
    ax_v.set_title("Speed and engaged gear")
    ax_g = ax_v.twinx()
    ax_g.step(t, gear, color=COLOR_ORANGE, linewidth=1.4, where="post", label="gear")
    ax_g.set_ylabel("gear", color=COLOR_MUTED)
    ax_g.set_ylim(-0.5, 7)
    ax_g.tick_params(colors=COLOR_MUTED, labelsize=9)
    ax_g.spines["top"].set_visible(False)
    lines = ax_v.get_lines() + ax_g.get_lines()
    ax_v.legend(lines, [l.get_label() for l in lines],
                frameon=False, labelcolor=COLOR_INK, fontsize=9, ncol=2)

    mode = f"feedforward + PI (kp={args.kp}, ki={args.ki})" if args.pi else "feedforward only"
    fig.suptitle(f"Acceleration tracking — {args.profile} profile, "
                 f"amplitude {args.amplitude} m/s², jerk limit {args.jerk_max} m/s³ — {mode}",
                 color=COLOR_INK, fontsize=14)

    os.makedirs(args.plot_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.plot_dir, f"accel_tracking_{stamp}.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\n그래프 저장: {out_path}")

    if show and matplotlib.get_backend().lower() != "agg":
        print("창을 닫으면 종료됩니다.")
        plt.show()
    plt.close(fig)
    return out_path
