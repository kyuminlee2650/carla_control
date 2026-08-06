"""Non-control helpers for stanley_pathtracking.py: spectator camera + result plotting.

Split out so the main script only contains the parts relevant to debugging the controller.
"""
import datetime
import math
import os

import carla

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# categorical slots from the house palette (references/palette.md), light mode
COLOR_BG = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_BLUE = "#2a78d6"      # actual / primary series
COLOR_ORANGE = "#eb6834"    # ego trajectory
COLOR_AQUA = "#1baf7a"      # throttle
COLOR_RED = "#e34948"       # brake / error


def follow_with_spectator(world, vehicle, back=8.0, up=4.0, pitch=-15.0):
    """Move the spectator to a 3rd-person chase view behind the vehicle."""
    transform = vehicle.get_transform()
    yaw = transform.rotation.yaw
    offset = carla.Location(
        x=-back * math.cos(math.radians(yaw)),
        y=-back * math.sin(math.radians(yaw)),
        z=up,
    )
    spectator_transform = carla.Transform(
        transform.location + offset,
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )
    world.get_spectator().set_transform(spectator_transform)


def _style_axes(ax):
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


def plot_results(path_x, path_y, hist, target_speed_ms, out_dir):
    """Save a 7-panel figure: trajectory, speed, throttle/brake, steer, cross-track error, heading error, heading."""
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    fig.patch.set_facecolor(COLOR_BG)
    ax_xy, ax_speed, ax_ctrl, ax_steer = axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3]
    ax_true_cte, ax_heading_err, ax_yaw = axes[1, 0], axes[1, 1], axes[1, 2]
    axes[1, 3].axis("off")  # unused since dropping the local-plan cte panel

    ax_xy.plot(path_x, path_y, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired path")
    ax_xy.plot(hist["x"], hist["y"], color=COLOR_ORANGE, linewidth=2, solid_capstyle="round", label="ego trajectory")
    ax_xy.scatter([path_x[0]], [path_y[0]], color=COLOR_BLUE, zorder=5, label="start")
    ax_xy.scatter([path_x[-1]], [path_y[-1]], color=COLOR_RED, marker="*", s=140, zorder=5, label="goal")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("Desired path vs. ego trajectory")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend(frameon=False, labelcolor=COLOR_INK, fontsize=9)

    ax_speed.axhline(target_speed_ms, color=COLOR_MUTED, linewidth=2, linestyle="--", label="target")
    ax_speed.plot(hist["t"], hist["v_x"], color=COLOR_BLUE, linewidth=2, solid_capstyle="round", label="actual")
    ax_speed.set_xlabel("t (s)")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.set_title("Speed")
    ax_speed.legend(frameon=False, labelcolor=COLOR_INK, fontsize=9)

    ax_ctrl.plot(hist["t"], hist["throttle"], color=COLOR_AQUA, linewidth=2, solid_capstyle="round", label="throttle")
    ax_ctrl.plot(hist["t"], hist["brake"], color=COLOR_RED, linewidth=2, solid_capstyle="round", label="brake")
    ax_ctrl.set_xlabel("t (s)")
    ax_ctrl.set_ylabel("control input")
    ax_ctrl.set_title("Control inputs")
    ax_ctrl.legend(frameon=False, labelcolor=COLOR_INK, fontsize=9)

    ax_steer.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_steer.plot(hist["t"], hist["steer_deg"], color=COLOR_BLUE, linewidth=2, solid_capstyle="round")
    ax_steer.set_xlabel("t (s)")
    ax_steer.set_ylabel("steer (deg)")
    ax_steer.set_title("Steering angle")

    ax_true_cte.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_true_cte.plot(hist["t"], hist["e_y"], color=COLOR_ORANGE, linewidth=2, solid_capstyle="round")
    ax_true_cte.set_xlabel("t (s)")
    ax_true_cte.set_ylabel("cross-track error (m)")
    ax_true_cte.set_title("Cross-track error vs. global path (front axle)")

    ax_heading_err.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_heading_err.plot(hist["t"], hist["e_theta"], color=COLOR_RED, linewidth=2, solid_capstyle="round")
    ax_heading_err.set_xlabel("t (s)")
    ax_heading_err.set_ylabel("heading error (deg)")
    ax_heading_err.set_title("Heading error (path - vehicle)")

    ax_yaw.plot(hist["t"], hist["path_yaw"], color=COLOR_MUTED, linewidth=2, linestyle="--", label="road heading")
    ax_yaw.plot(hist["t"], hist["yaw"], color=COLOR_BLUE, linewidth=2, solid_capstyle="round", label="vehicle yaw")
    ax_yaw.set_xlabel("t (s)")
    ax_yaw.set_ylabel("heading (deg)")
    ax_yaw.set_title("Vehicle heading vs. road heading")
    ax_yaw.legend(frameon=False, labelcolor=COLOR_INK, fontsize=9)

    for ax in (ax_xy, ax_speed, ax_ctrl, ax_steer, ax_heading_err, ax_yaw, ax_true_cte):
        _style_axes(ax)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"run_{timestamp}.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.show()  # blocks until the window is closed
    plt.close(fig)
    return out_path
