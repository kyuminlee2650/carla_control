r"""Post-hoc analysis of one saved VAD closed-loop run -- vad_b2d_agent.py's own SAVE_PATH output
(rgb_front/, rgb_front_left/, rgb_front_right/, rgb_back*, bev/, meta/*.json, one entry per saved
frame). No CARLA server, no model forward pass -- pure log replay, so this runs fine on a machine
with nothing but the control-only .venv (matplotlib/Pillow/imageio-ffmpeg, all already installed
here; no carla import needed since nothing in this file talks to a live simulator).

Built to dig into ONE specific problem run (e.g. a signalized-turn scenario where the car braked/
turned wrong) by looking at VAD's own predicted trajectory (`plan`/`all_plan` in meta/*.json) and
the resulting control input (steer/throttle/brake) side by side with the actual camera footage --
not just aggregate metrics.

meta/*.json schema (team_code/pid_controller.py's own PIDController.control_pid() metadata, plus
what vad_b2d_agent.py's run_step() adds on top -- read off both files directly, not guessed):
    speed, steer, throttle, brake          -- final applied control (already clipped/zeroed)
    steer_traj, throttle_traj, brake_traj  -- same triple pre run_step()'s own
                                               `if throttle_traj > brake_traj: brake_traj = 0.0`
                                               (usually identical to steer/throttle/brake; kept
                                               separate here in case a run has them differ)
    wp_1..wp_4                             -- first 4 of `plan`, VAD's own [lateral, forward]
                                               ego-frame convention (confirmed by pid_controller.py's
                                               own `atan2(aim[1], aim[0])` -- forward is index 1),
                                               duplicated into named keys for convenience
    plan                                   -- VAD's chosen-command trajectory, 6 points, same
                                               [lateral, forward] convention as wp_*
    all_plan                               -- VAD's per-command trajectories, 6 commands x 6 points
                                               -- plan == all_plan[command]
    command                                -- which of the 6 candidates was actually used
    aim, target                            -- PIDController's own two candidate steering-aim points
                                               (aim: picked off `plan` itself; target: the global-
                                               route aim passed in from outside)
    angle, angle_last, angle_target, angle_final, delta, desired_speed
                                            -- PIDController.control_pid()'s own intermediates;
                                               angle_final is whichever of angle/angle_target the
                                               use_target_to_aim switch below picked THAT tick

use_target_to_aim: PIDController.control_pid() itself doesn't save this bool, only its two
candidate angles and the one it picked (angle_final) -- recomputed here from angle/angle_last/
angle_target/target[1] with pid_controller.py's own exact formula (angle_thresh=0.3,
dist_thresh=10, both confirmed against team_code/pid_controller.py's PIDController defaults) so
this script can flag ticks where the aim source FLIPS, a classic source of steering jitter right at
an intersection -- angle_final alone doesn't show which of its two inputs is driving it tick to
tick.

Frame index -> saved file basename ("%04d" % (self.step // 10)): run_step() calls save() every
tick, but save()'s own `frame = self.step // 10` only advances every 10 ticks (earlier calls in
that window get overwritten) -- so consecutive saved frames are ~10 sim ticks apart. At a 20 Hz /
0.05 s tick (matches the IMU sensor_tick and Bench2Drive's documented save cadence) that is roughly
0.5 s/frame. Kept as an approximate secondary time axis ("~Ns"), never asserted as exact, since it
is inferred from the save() code rather than read from a logged timestamp.

Usage (Windows, .venv lives one level up in carla_control/):
    cd C:\Users\mumu2\carla_control\b2d_controller
    ..\.venv\Scripts\python.exe analyze_run_log.py --log-dir "<path to the saved run folder>"
    ..\.venv\Scripts\python.exe analyze_run_log.py --log-dir "<...>" --skip-video   # summary PNG only, fast
    ..\.venv\Scripts\python.exe analyze_run_log.py --log-dir "<...>" --start 20 --end 45 --fps 4

Output (into --out-dir, default: the log folder's own parent / "<folder name>_analysis/"):
    summary.png   time-series overview: speed, steer, throttle+brake, aim-source/angle, command
    review.mp4    rgb_front + bev + VAD trajectory (plan/all_plan) + control readout, one
                  composited frame per saved frame (skipped with --skip-video)
Console output: a short list of candidate "problem points" -- brake on/off edges, command
changes, aim-source flips, and the largest tick-to-tick steer jumps -- so you don't have to scrub
the whole video blind before you know where to look.
"""
import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")   # headless: this script only ever renders to file/array, never plt.show()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

# One simulator tick, exactly: CARLA's fixed_delta_seconds and the agent's own sensor_tick both
# run this loop at 20 Hz (mpc_kf_controller.DT). Dumped filenames are RAW TICK INDICES, so a
# frame's timestamp is simply index * TICK_DT_S -- correct whatever cadence the agent dumped at,
# which the old fixed FRAME_DT_S=0.5 was not (it hard-coded the agent's own step//10 cadence, and
# silently mislabelled every axis once that cadence changed).
TICK_DT_S = 0.05

COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
COLOR_AQUA = "#1baf7a"
COLOR_RED = "#e34948"
COLOR_PURPLE = "#8b5cf6"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BG = "#fcfcfb"
COLOR_INK = "#0b0b0b"

ANGLE_THRESH = 0.3   # PIDController defaults (team_code/pid_controller.py), used to recompute
DIST_THRESH = 10     # use_target_to_aim below -- see module docstring

# vad_b2d_agent.py's own lidar2img['CAM_FRONT'], copied verbatim -- used to draw VAD's predicted
# trajectory onto the front camera the same way Bench2Drive's own
# vad_b2d_agent_visualize.draw_traj() does. That function feeds the plan's two columns straight in
# as lidar (x, y) with a fixed ground height and prepends the bonnet pixel; both conventions are
# copied rather than re-derived, so this overlay lines up with B2D's own visualization.
LIDAR2IMG_CAM_FRONT = np.array([
    [1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
    [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
    [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
CAM_CANVAS = (900, 1600)     # (h, w) of the dumped rgb_front images
CAM_GROUND_Z = -1.84         # b2d's own ground height, see vad_demo_video/README.md
CAM_BONNET_PX = (800, 900)   # draw_traj()'s own ego start pixel (lidar origin projects behind the
                             # front camera in b2d's calibration, so it cannot be projected)


# --------------------------------------------------------------------------------------- loading

def load_run(log_dir):
    """Read every meta/*.json into one dict-of-arrays, sorted by frame index. Raises with a clear
    message if meta/ is missing or empty -- a wrong --log-dir is a common mistake, better to fail
    loudly here than three functions deeper."""
    meta_dir = os.path.join(log_dir, "meta")
    paths = sorted(glob.glob(os.path.join(meta_dir, "*.json")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not paths:
        raise FileNotFoundError(f"no meta/*.json under {meta_dir!r} -- wrong --log-dir?")

    idx, records = [], []
    for p in paths:
        with open(p) as f:
            records.append(json.load(f))
        idx.append(int(os.path.splitext(os.path.basename(p))[0]))
    idx = np.asarray(idx)

    get = lambda key, default=np.nan: np.asarray([r.get(key, default) for r in records], dtype=float)
    data = dict(
        idx=idx, records=records,
        speed=get("speed"), desired_speed=get("desired_speed"),
        steer=get("steer"), throttle=get("throttle"), brake=get("brake"),
        steer_traj=get("steer_traj"), throttle_traj=get("throttle_traj"), brake_traj=get("brake_traj"),
        angle=get("angle"), angle_last=get("angle_last"), angle_target=get("angle_target"),
        angle_final=get("angle_final"), delta=get("delta"),
        command=get("command", -1).astype(int),
    )

    # Which controller wrote this run. mpc_kf_controller.MpcKfController's metadata carries
    # delta_rad/a_cmd/kf_status; team_code/pid_controller.py's carries angle/aim/target/
    # desired_speed. The two share only speed/steer/throttle/brake/plan/all_plan/command, so the
    # panels that read anything else have to branch -- and a run must never be silently rendered
    # with the other controller's panels all NaN.
    data["is_mpc"] = bool(records) and "delta_rad" in records[0]

    if data["is_mpc"]:
        data.update(
            delta_rad=get("delta_rad"), a_cmd=get("a_cmd"), v_y_hat=get("v_y_hat"),
            u_raw=get("u_raw"), u_filtered=get("u_filtered"), gear=get("gear", 0),
            lat_status=[r.get("kf_status", "?") for r in records],
            spd_status=[r.get("speed_mpc_status", "?") for r in records],
        )
        # LateralMPC.solve() falls back to a held/decayed command on anything outside these two --
        # see mpc_kf_controller._OK_STATUSES. Tracked so the summary can show WHERE that happened.
        data["lat_ok"] = np.asarray([s in ("solved", "solved inaccurate")
                                     for s in data["lat_status"]])
        data["use_target_to_aim"] = np.zeros(len(idx), dtype=bool)   # PID-only concept
        return data

    # use_target_to_aim: pid_controller.py's own exact formula, recomputed from what got saved --
    # see module docstring for why this isn't just read off a saved key.
    target_y = np.asarray([r.get("target", (np.nan, np.nan))[1] for r in records], dtype=float)
    use_target = (np.abs(data["angle_target"]) < np.abs(data["angle"]))
    use_target = use_target | ((np.abs(data["angle_target"] - data["angle_last"]) > ANGLE_THRESH)
                               & (target_y < DIST_THRESH))
    data["use_target_to_aim"] = use_target
    return data


# --------------------------------------------------------------------------------------- styling

def _style_axes(ax):
    ax.set_facecolor(COLOR_BG)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    ax.xaxis.label.set_color(COLOR_MUTED)
    ax.yaxis.label.set_color(COLOR_MUTED)


def _title(ax, text):
    ax.set_title(text, fontsize=11, fontweight="bold", color=COLOR_INK, loc="left")


def _brake_shading(ax, idx, brake):
    """Vertical bands over every brake==1 run -- shared by every time-series panel so a braking
    episode (e.g. the red light this script was written to look at) lines up visually across all
    of them at once."""
    on = brake > 0.5
    if not on.any():
        return
    edges = np.flatnonzero(np.diff(np.concatenate([[False], on, [False]])))
    for s, e in zip(edges[0::2], edges[1::2]):
        ax.axvspan(idx[s] - 0.5, idx[min(e, len(idx) - 1)] - 0.5, color=COLOR_RED, alpha=0.08, zorder=0)


# --------------------------------------------------------------------------------------- summary figure

def build_summary_figure(data, out_path, title):
    idx = data["idx"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(title, fontsize=15, fontweight="bold", color=COLOR_INK)

    is_mpc = data.get("is_mpc", False)

    ax = axes[0, 0]
    _brake_shading(ax, idx, data["brake"])
    ax.plot(idx, data["speed"], color=COLOR_BLUE, linewidth=1.5, label="speed")
    if is_mpc:
        # MpcKfController has no "desired_speed" -- SpeedMPC emits an ACCELERATION command, and the
        # LUT+PID pedal layer tracks that directly (see mpc_kf_controller module docstring pt 5).
        # a_cmd is plotted on a twin axis rather than dropped, since it is the actual longitudinal
        # setpoint this controller works to.
        ax2 = ax.twinx()
        ax2.plot(idx, data["a_cmd"], color=COLOR_ORANGE, linewidth=1.2, linestyle="--", label="a_cmd")
        ax2.set_ylabel("m/s^2", color=COLOR_ORANGE, fontsize=9)
        ax2.tick_params(axis="y", colors=COLOR_ORANGE, labelsize=8)
        ax2.spines["top"].set_visible(False)
        _title(ax, "speed & SpeedMPC a_cmd  (red band = braking)")
    else:
        ax.plot(idx, data["desired_speed"], color=COLOR_ORANGE, linewidth=1.2, linestyle="--",
                label="desired_speed")
        _title(ax, "speed vs desired_speed  (red band = braking)")
    ax.set_ylabel("m/s"); ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style_axes(ax)

    ax = axes[0, 1]
    _brake_shading(ax, idx, data["brake"])
    ax.axhline(0, color=COLOR_MUTED, linewidth=0.8)
    ax.plot(idx, data["steer"], color=COLOR_BLUE, linewidth=1.5, label="steer")
    dsteer = np.diff(data["steer"], prepend=data["steer"][0])
    jump_idx = idx[np.argsort(-np.abs(dsteer))[:5]]
    for j in jump_idx:
        ax.axvline(j, color=COLOR_PURPLE, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_ylabel("[-1, 1]"); _title(ax, "steer  (dotted = top-5 tick-to-tick jumps)")
    _style_axes(ax)

    ax = axes[1, 0]
    _brake_shading(ax, idx, data["brake"])
    ax.plot(idx, data["throttle"], color=COLOR_AQUA, linewidth=1.5, label="throttle")
    ax.plot(idx, data["brake"], color=COLOR_RED, linewidth=1.5, label="brake")
    ax.set_ylabel("[0, 1]"); ax.set_ylim(-0.05, 1.05); _title(ax, "throttle & brake"); ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)

    ax = axes[1, 1]
    _brake_shading(ax, idx, data["brake"])
    if is_mpc:
        # LateralMPC's own wheel-angle command, plus the ticks where its QP did NOT solve and the
        # controller was therefore running on a held/decayed command instead of a fresh optimum.
        # Those ticks are the ones worth scrubbing to in the video.
        ax.plot(idx, np.degrees(data["delta_rad"]), color=COLOR_BLUE, linewidth=1.5,
                label="delta (LateralMPC)")
        bad = ~data["lat_ok"]
        if bad.any():
            ax.plot(idx[bad], np.degrees(data["delta_rad"])[bad], linestyle="none", marker="x",
                    color=COLOR_RED, markersize=5, label="QP not solved (held/decayed)")
        ax.plot(idx, np.degrees(data["v_y_hat"]) * 0 + np.nan, alpha=0)   # keep legend order stable
        ax.set_ylabel("deg"); _title(ax, "LateralMPC steer angle & QP failures")
        switch = np.flatnonzero(np.diff(bad.astype(int)) != 0) + 1
    else:
        ax.plot(idx, data["angle"], color=COLOR_MUTED, linewidth=1.0, linestyle="--", label="angle (traj aim)")
        ax.plot(idx, data["angle_target"], color=COLOR_AQUA, linewidth=1.0, linestyle="--", label="angle_target (route aim)")
        ax.plot(idx, data["angle_final"], color=COLOR_BLUE, linewidth=1.8, label="angle_final (picked)")
        switch = np.flatnonzero(np.diff(data["use_target_to_aim"].astype(int)) != 0) + 1
        for s in switch:
            ax.axvline(idx[s], color=COLOR_PURPLE, linewidth=0.8, linestyle=":", alpha=0.7)
        ax.set_ylabel("normalized angle"); _title(ax, "steering aim source  (dotted = use_target_to_aim flips)")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)

    ax = axes[2, 0]
    ax.step(idx, data["command"], color=COLOR_ORANGE, linewidth=1.5, where="post")
    cswitch = np.flatnonzero(np.diff(data["command"]) != 0) + 1
    for s in cswitch:
        ax.axvline(idx[s], color=COLOR_PURPLE, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_ylabel("command idx (0-5)"); ax.set_xlabel("frame #"); _title(ax, "VAD route command  (dotted = changes)")
    _style_axes(ax)

    ax = axes[2, 1]
    ax.axis("off")
    n = len(idx)
    n_brake = int((data["brake"] > 0.5).sum())
    lines = [
        f"controller: {'MpcKfController (LateralMPC + SpeedMPC + KF)' if is_mpc else 'PIDController (stock)'}",
        f"frames: {n}  ({idx[-1] * TICK_DT_S:.1f}s of driving, "
        f"1 frame per {int(np.median(np.diff(idx))) if n > 1 else 1} tick(s))",
        f"braking: {n_brake} frames ({100*n_brake/n:.0f}%)",
        f"command switches: {len(cswitch)}  at frames {list(idx[cswitch])[:10]}",
        f"largest steer jumps at frames: {sorted(jump_idx.tolist())}",
        f"max |steer|: {np.nanmax(np.abs(data['steer'])):.3f}   max throttle: {np.nanmax(data['throttle']):.3f}",
    ]
    if is_mpc:
        from collections import Counter
        nbad = int((~data["lat_ok"]).sum())
        lines.insert(4, f"LateralMPC QP not solved: {nbad}/{n} frames ({100*nbad/n:.0f}%)"
                        f"  -> {dict(Counter(data['lat_status']))}")
        lines.append(f"max |delta|: {np.nanmax(np.abs(np.degrees(data['delta_rad']))):.2f} deg"
                     f"   a_cmd range: [{np.nanmin(data['a_cmd']):+.2f}, {np.nanmax(data['a_cmd']):+.2f}] m/s^2")
    else:
        lines.insert(4, f"aim-source flips: {len(switch)}  at frames {list(idx[switch])[:10]}")
    ax.text(0.0, 0.95, "\n\n".join(lines), transform=ax.transAxes, va="top", ha="left",
           fontsize=10.5, color=COLOR_INK, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return dict(n_brake=n_brake, command_switches=idx[cswitch].tolist(),
               aim_switches=idx[switch].tolist(), steer_jumps=sorted(jump_idx.tolist()))


# --------------------------------------------------------------------------------------- per-frame panels (video)

def _fig_to_rgb(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return np.ascontiguousarray(buf[:, :, :3])


def overlay_plan_on_cam(img, plan, cmap_name="winter", width=7):
    """Draw VAD's predicted trajectory onto the front-camera image, following
    Bench2Drive's own vad_b2d_agent_visualize.draw_traj(): project the plan's two columns as lidar
    (x, y) at a fixed ground height, drop points outside the canvas, prepend the bonnet pixel, then
    spline-smooth and stroke with a colour gradient.

    Two deliberate differences from that function, neither of which changes the geometry:
      * drawn with PIL instead of cv2 -- cv2 is not installed in the control-only .venv this script
        is meant to run in (see module docstring), and nothing else here needs it.
      * gradient comes from a matplotlib colormap. Default 'winter' is what VAD's own
        visualization uses for the PLANNING trajectory specifically (see vad_demo_video/README.md);
        draw_traj()'s hue_start/hue_end HSV ramp is the b2d equivalent of the same idea.
    img is a PIL Image (already at CAM_CANVAS size); returns a new PIL Image."""
    from PIL import ImageDraw
    from scipy.interpolate import splprep, splev

    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2 or len(plan) < 2:
        return img

    h, w = CAM_CANVAS
    pts_4d = np.stack([plan[:, 0], plan[:, 1],
                       np.full(len(plan), CAM_GROUND_Z), np.ones(len(plan))])
    pts_2d = (LIDAR2IMG_CAM_FRONT @ pts_4d).T
    depth = pts_2d[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pts_2d[:, 0] /= depth
        pts_2d[:, 1] /= depth
    # depth > 0 keeps points actually IN FRONT of the camera -- without it a point behind the image
    # plane projects to a mirrored pixel that can still land inside the canvas bounds.
    mask = ((depth > 0) & (pts_2d[:, 0] > 0) & (pts_2d[:, 0] < w)
            & (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < h))
    if not mask.any():
        return img
    xy = np.concatenate([np.array([CAM_BONNET_PX], dtype=float), pts_2d[mask, 0:2]], axis=0)

    # splprep needs strictly increasing parameterization; duplicate pixels (a stopped car predicting
    # a near-zero plan) make it raise, so fall back to the polyline in that case.
    try:
        tck, _ = splprep([xy[:, 0], xy[:, 1]], s=0)
        smooth = np.stack(splev(np.linspace(0, 1, 100), tck)).T
    except Exception:
        smooth = xy

    out = img.copy()
    draw = ImageDraw.Draw(out)
    cmap = plt.get_cmap(cmap_name)
    n = len(smooth)
    for i in range(n - 1):
        c = tuple(int(255 * v) for v in cmap(i / max(n - 1, 1))[:3])
        draw.line([tuple(smooth[i]), tuple(smooth[i + 1])], fill=c, width=width)
    return out


def _trajectory_panel(record, w_px, h_px, dpi=100):
    """VAD's own [lateral, forward] convention plotted directly as (x, y) -- lateral left/right on
    the x-axis, forward-ahead on the y-axis, so the picture reads like a bird's-eye 'road going up'
    view without any axis-flip bookkeeping."""
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(COLOR_BG)
    ax = fig.add_axes([0.14, 0.10, 0.83, 0.82])
    ax.set_facecolor(COLOR_BG)

    all_plan = record.get("all_plan")
    command = int(record.get("command", -1))
    if all_plan:
        for c, traj in enumerate(all_plan):
            traj = np.asarray(traj)
            if c == command:
                continue
            ax.plot(traj[:, 0], traj[:, 1], color=COLOR_MUTED, linewidth=1.0, alpha=0.35, zorder=1)

    plan = np.asarray(record.get("plan", []))
    if plan.size:
        ax.plot(plan[:, 0], plan[:, 1], color=COLOR_ORANGE, linewidth=2.2, marker="o", markersize=3,
               zorder=3, label=f"plan (cmd {command})")

    aim = record.get("aim"); target = record.get("target")
    if aim is not None:
        ax.scatter([aim[0]], [aim[1]], color=COLOR_BLUE, s=45, zorder=4, label="aim")
    if target is not None:
        ax.scatter([target[0]], [target[1]], color=COLOR_RED, s=45, marker="^", zorder=4, label="target")
    ax.scatter([0], [0], color=COLOR_INK, s=30, marker="s", zorder=5, label="ego")

    ax.axhline(0, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.axvline(0, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("lateral (m)", fontsize=8, color=COLOR_MUTED)
    ax.set_ylabel("forward (m)", fontsize=8, color=COLOR_MUTED)
    ax.tick_params(labelsize=7, colors=COLOR_MUTED)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    arr = _fig_to_rgb(fig)
    plt.close(fig)
    return arr


def _gauge_panel(record, frame_idx, w_px, h_px, dpi=100):
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(COLOR_BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    speed = record.get("speed", float("nan"))
    desired = record.get("desired_speed", float("nan"))
    steer = record.get("steer", 0.0)
    throttle = record.get("throttle", 0.0)
    brake = record.get("brake", 0.0)
    command = record.get("command", -1)

    header = f"frame {frame_idx:04d}   {frame_idx * TICK_DT_S:5.1f}s   cmd {command}"
    ax.text(0.03, 0.92, header, fontsize=12, fontweight="bold", color=COLOR_INK, family="monospace")
    ax.text(0.03, 0.78, f"speed        {speed:5.2f} m/s", fontsize=11, color=COLOR_INK, family="monospace")

    if "delta_rad" in record:
        # MpcKfController readout -- the PID's desired_speed line has no counterpart here (see
        # build_summary_figure); a_cmd + the wheel angle + the QP status are the equivalents.
        ax.text(0.03, 0.68, f"a_cmd      {record.get('a_cmd', float('nan')):+6.2f} m/s^2",
                fontsize=11, color=COLOR_MUTED, family="monospace")
        ax.text(0.03, 0.60, f"delta      {np.degrees(record['delta_rad']):+6.2f} deg   "
                            f"gear {int(record.get('gear', 0))}",
                fontsize=10, color=COLOR_MUTED, family="monospace")
        st = record.get("kf_status", "?")
        ok = st in ("solved", "solved inaccurate")
        ax.text(0.03, 0.02, f"lat QP: {st}", fontsize=10, family="monospace",
                fontweight="bold" if not ok else "normal",
                color=COLOR_INK if ok else COLOR_RED)
    else:
        ax.text(0.03, 0.68, f"desired      {desired:5.2f} m/s", fontsize=11, color=COLOR_MUTED,
                family="monospace")

    def _bar(y, label, value, lo, hi, color):
        ax.text(0.03, y, f"{label:9s}{value:+.3f}", fontsize=11, color=COLOR_INK, family="monospace")
        bx0, bx1, bw = 0.45, 0.97, 0.97 - 0.45
        ax.add_patch(plt.Rectangle((bx0, y - 0.015), bw, 0.05, facecolor=COLOR_GRID, edgecolor="none"))
        frac = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
        zero_frac = np.clip((0.0 - lo) / (hi - lo), 0.0, 1.0)
        x0, x1 = sorted([bx0 + zero_frac * bw, bx0 + frac * bw])
        ax.add_patch(plt.Rectangle((x0, y - 0.015), max(x1 - x0, 0.002), 0.05, facecolor=color, edgecolor="none"))

    _bar(0.52, "steer", steer, -1.0, 1.0, COLOR_BLUE)
    _bar(0.40, "throttle", throttle, 0.0, 1.0, COLOR_AQUA)
    _bar(0.28, "brake", brake, 0.0, 1.0, COLOR_RED)

    if brake > 0.5:
        ax.text(0.03, 0.10, "BRAKING", fontsize=13, fontweight="bold", color=COLOR_RED, family="monospace")

    arr = _fig_to_rgb(fig)
    plt.close(fig)
    return arr


# --------------------------------------------------------------------------------------- video

def build_video(log_dir, data, out_path, fps=4.0, start=0, end=None,
                cam_w=1280, cam_h=720, bev_px=300, gauge_w=480, overlay_traj=True):
    import imageio_ffmpeg

    idx, records = data["idx"], data["records"]
    end = len(idx) if end is None else min(end, len(idx))
    sel = range(start, end)

    # Three dump layouts are in circulation: upstream's rgb_front/*.png + bev/, the vaddump
    # variant's CAM_FRONT/*.jpg with no bev/ at all, and the current agent's CAM_FRONT/*.jpg +
    # bev/*.jpg (it moved to the CAM_*/ naming so one dump also feeds render_vad_style_ctrl.py).
    # Detect rather than assume, and treat bev as optional -- otherwise a whole run silently
    # renders zero frames (the per-frame existence check just `continue`s), which reads as "the
    # script broke" rather than "this dump has no BEV".
    cam_subdir = next((d for d in ("rgb_front", "CAM_FRONT")
                       if os.path.isdir(os.path.join(log_dir, d))), None)
    if cam_subdir is None:
        raise FileNotFoundError(
            f"no front-camera folder under {log_dir!r} (looked for rgb_front/ and CAM_FRONT/)")

    traj_w = cam_w - bev_px - gauge_w
    if traj_w <= 0:
        raise ValueError("cam_w too small for bev_px + gauge_w")
    canvas_w, canvas_h = cam_w, cam_h + bev_px

    writer = imageio_ffmpeg.write_frames(
        out_path, size=(canvas_w, canvas_h), fps=fps, quality=6,
        macro_block_size=1, ffmpeg_log_level="error",
    )
    writer.send(None)
    n_written = 0
    try:
        for i in sel:
            frame_idx = int(idx[i])
            # stock save() writes .png, the vaddump variant writes .jpg -- try both rather than
            # hard-coding either, or a whole run silently yields zero frames.
            cam_path = next((c for c in (os.path.join(log_dir, cam_subdir, f"{frame_idx:04d}{e}")
                                         for e in (".png", ".jpg")) if os.path.exists(c)), None)
            if cam_path is None:
                continue   # a frame missing its images (e.g. B2D_SAVE_IMAGES was off) is skipped
            bev_path = next((c for c in (os.path.join(log_dir, "bev", f"{frame_idx:04d}{e}")
                                         for e in (".png", ".jpg")) if os.path.exists(c)), None)
            has_bev = bev_path is not None

            canvas = Image.new("RGB", (canvas_w, canvas_h), (252, 252, 251))
            cam = Image.open(cam_path).convert("RGB")
            if overlay_traj:
                # Overlay BEFORE the resize, so the projection runs at the calibration's own
                # CAM_CANVAS resolution that LIDAR2IMG_CAM_FRONT is expressed in.
                cam = overlay_plan_on_cam(cam, records[i].get("plan", []))
            cam = cam.resize((cam_w, cam_h))
            canvas.paste(cam, (0, 0))
            if has_bev:
                bev = Image.open(bev_path).convert("RGB").resize((bev_px, bev_px))
                canvas.paste(bev, (0, cam_h))

            traj_arr = _trajectory_panel(records[i], traj_w, bev_px)
            canvas.paste(Image.fromarray(traj_arr).resize((traj_w, bev_px)), (bev_px, cam_h))

            gauge_arr = _gauge_panel(records[i], frame_idx, gauge_w, bev_px)
            canvas.paste(Image.fromarray(gauge_arr).resize((gauge_w, bev_px)), (bev_px + traj_w, cam_h))

            writer.send(np.asarray(canvas).tobytes())
            n_written += 1
    finally:
        writer.close()
    return n_written


# --------------------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", required=True, help="saved run folder (contains meta/, rgb_front/, bev/, ...)")
    parser.add_argument("--out-dir", default=None, help="default: '<log-dir>_analysis' next to --log-dir")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None,
                        help="default: real time, derived from the dump cadence "
                             "(1 / (ticks-between-frames * 0.05s)) -- pass a number to override")
    parser.add_argument("--skip-video", action="store_true", help="only write summary.png (fast)")
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    out_dir = args.out_dir or (log_dir.rstrip("\\/") + "_analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {log_dir} ...")
    data = load_run(log_dir)
    print(f"  {len(data['idx'])} frames")

    title = os.path.basename(log_dir)
    summary_path = os.path.join(out_dir, "summary.png")
    stats = build_summary_figure(data, summary_path, title)
    print(f"Summary figure -> {summary_path}")
    print(f"  controller: {'MpcKfController' if data['is_mpc'] else 'PIDController (stock)'}")
    print(f"  braking frames: {stats['n_brake']}")
    print(f"  command switches at frames: {stats['command_switches']}")
    if data["is_mpc"]:
        # aim-source flips are a PIDController concept (its two candidate steering aims); the
        # equivalent "go look here" pointer for the MPC is where its QP stopped solving and the
        # controller fell back to a held/decayed command.
        bad = np.flatnonzero(~data["lat_ok"])
        print(f"  LateralMPC QP not solved: {len(bad)}/{len(data['idx'])} frames"
              f"  at {list(data['idx'][bad])[:15]}")
    else:
        print(f"  aim-source flips at frames: {stats['aim_switches']}")
    print(f"  largest steer jumps at frames: {stats['steer_jumps']}")

    if not args.skip_video:
        video_path = os.path.join(out_dir, "review.mp4")
        # Real-time playback by default: the dump cadence is whatever the agent used (every tick
        # now, every 10th before), and the two are not comparable side by side unless the video
        # fps compensates. Reading it off the data means a PID dump and an MPC dump of the same
        # route play back at the same wall-clock speed without anyone passing --fps by hand.
        idx = data["idx"]
        stride = int(np.median(np.diff(idx))) if len(idx) > 1 else 1
        fps = args.fps if args.fps is not None else 1.0 / max(stride, 1) / TICK_DT_S
        print(f"Rendering video (fps={fps:g}{'' if args.fps is not None else ', real time'}) ...")
        n = build_video(log_dir, data, video_path, fps=fps, start=args.start, end=args.end)
        print(f"Video -> {video_path} ({n} frames, {n * max(stride,1) * TICK_DT_S:.1f}s of driving)")


if __name__ == "__main__":
    main()
