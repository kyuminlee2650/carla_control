r"""Controller A/B figures for two Bench2Drive closed-loop dumps of the SAME route.

Built for the MPC-KF vs stock-PID comparison: same VAD checkpoint, same planner, same route --
only the controller differs, so every difference these figures show is the controller's.

Why this is not viz_utils.plot_results(): that function's headline panels are cross-track error,
heading error and "road heading", all of which need a KNOWN reference path. There is no such path
here. VAD re-plans from scratch every tick in its own ego frame (mpc_kf_controller module docstring
point 1: e_y is anchored to 0 every tick precisely because there is no map route), so a "tracking
error" against VAD's own output measures how much VAD moved its mind, not how well the controller
tracked. The four figures below deliberately use only quantities that are well defined without a
reference path -- measured vehicle dynamics, the commands themselves, the Kalman filter against
ground truth, and the driven line against the actual OpenDRIVE lane geometry.

    fig 1  comfort     PID vs MPC on the six channels B2D's Comfortness metric scores,
                       with the B2D limits drawn as red dashed lines
    fig 2  control     PID vs MPC: v_x, yaw, steer, throttle, brake, and u = throttle - brake
    fig 3  kalman      MPC-KF only: v_y estimate vs ground truth (PID has no v_y estimate)
    fig 4  trajectory  both driven lines over the real Town10HD lane boundaries

Usage (from this directory, with carla_control's own .venv -- it has carla, scipy and matplotlib):
    ../.venv/bin/python compare_runs.py \
        --mpc-dir <run>/frames/<dump>  --pid-dir <run>/frames/<dump>  --out-dir <dir>

Each --*-dir is any directory holding `metric_info.json` and `meta/` (a live frames/<dump>/
directory, or one of the trimmed copies under runs/../archive/).
"""

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys

import numpy as np
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# carla_control's own house style + panel helpers, reused so these figures sit next to the ones
# viz_utils already produces. Only the styling/limits come from there; every plotting function
# below is new (see module docstring for why plot_results() itself does not apply).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import viz_utils as V

import carla

TICK_DT_S = 0.05        # CARLA fixed_delta_seconds / the agent's sensor_tick -- the loop is 20 Hz
SG_WINDOW, SG_POLY = 7, 2   # B2D's own savgol settings (efficiency_smoothness_benchmark.py)

DEFAULT_XODR = "/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town10HD.xodr"
B2D_BENCHMARK = ("/home/ailab/2026intern/kmlee/vad_demo_video/Bench2Drive/tools/"
                 "efficiency_smoothness_benchmark.py")


def _colors(runs):
    """Line colours by position. With a baseline present it draws underneath in muted grey and the
    MPC sits on top in blue; a single run is always the blue one, because a lone grey trace reads
    as the thing being compared against rather than the subject."""
    return (V.COLOR_BLUE,) if len(runs) == 1 else (V.COLOR_MUTED, V.COLOR_BLUE)


def infer_xodr(dump_dir):
    """Find the OpenDRIVE map for whichever town this dump was recorded in.

    Bench2Drive names its dump directory after the scenario, e.g.
    bench2drive220_RouteScenario_26950_rep0_Town03_..., so the town is right there. Defaulting to
    a fixed map instead would silently draw route 26950's Town03 trajectory over Town10HD's lane
    geometry -- a picture that looks plausible and is entirely wrong.
    """
    name = os.path.basename(os.path.abspath(dump_dir).rstrip("/"))
    m = re.search(r"_(Town\d+\w*?)_", name + "_")
    if not m:
        print(f"  ! could not read a town from {name!r}; falling back to {DEFAULT_XODR}")
        return DEFAULT_XODR
    town = m.group(1)
    path = os.path.join(os.path.dirname(DEFAULT_XODR), town + ".xodr")
    if not os.path.exists(path):
        print(f"  ! {town}.xodr not installed (large maps ship without a loose .xodr); "
              f"fig 4 will have no lane geometry")
        return path
    return path


# --------------------------------------------------------------------------- B2D comfort limits

def b2d_limits(path=B2D_BENCHMARK):
    """The six Comfortness bounds, imported from Bench2Drive's own benchmark script rather than
    re-typed here, so the red lines in fig 1 cannot drift away from the thresholds that actually
    decide the score. Falls back to viz_utils' copy if that tree isn't present."""
    try:
        spec = importlib.util.spec_from_file_location("b2d_esb", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return {
            "yaw_rate":   (-m.max_abs_yaw_rate,  m.max_abs_yaw_rate),
            "yaw_acc":    (-m.max_abs_yaw_accel, m.max_abs_yaw_accel),
            "a_x":        (m.min_lon_accel,      m.max_lon_accel),
            "a_y":        (-m.max_abs_lat_accel, m.max_abs_lat_accel),
            "lon_jerk":   (-m.max_abs_lon_jerk,  m.max_abs_lon_jerk),
            "jerk_total": (-m.max_abs_mag_jerk,  m.max_abs_mag_jerk),
        }, path
    except Exception as exc:                                   # pragma: no cover - env dependent
        print(f"  ! could not read {path} ({exc}); falling back to viz_utils.B2D_COMFORT_LIMITS")
        L = V.B2D_COMFORT_LIMITS
        return {"yaw_rate": L["yaw_rate"], "yaw_acc": L["yaw_acc"], "a_x": L["a_x"],
                "a_y": L["a_y"], "lon_jerk": L["jerk"],
                "jerk_total": (-L["jerk_total"][1], L["jerk_total"][1])}, "viz_utils"


# --------------------------------------------------------------------------- loading

def load_run(run_dir, label):
    """Read one dump into tick-aligned arrays.

    Two files, two cadences, one time base:
      * metric_info.json -- written every tick, keyed by the agent's own step counter. This is the
        GROUND TRUTH channel (CARLA's own pose/velocity/acceleration for the hero actor), and it is
        what B2D's Comfortness metric reads.
      * meta/NNNN.json   -- one file per dumped frame, holding what the CONTROLLER computed
        (steer/throttle/brake, and for the MPC also v_y_hat). Its filenames are raw tick indices
        with the current agent; older mpckf dumps used step//10, which is detected below rather
        than assumed, since silently misaligning commands against pose by 10x would quietly
        invalidate every panel that puts the two on one time axis.
    """
    mi_path = os.path.join(run_dir, "metric_info.json")
    if not os.path.exists(mi_path):
        raise FileNotFoundError(f"{mi_path} not found -- is {run_dir!r} a dump directory?")
    mi = json.load(open(mi_path))
    ticks = np.asarray(sorted(int(k) for k in mi), dtype=int)
    rec = [mi[str(t)] for t in ticks]

    run = dict(
        label=label, dir=run_dir, tick=ticks, t=ticks * TICK_DT_S,
        loc=np.array([r["location"][:2] for r in rec], dtype=float),
        acc=np.array([r["acceleration"][:2] for r in rec], dtype=float),
        fwd=np.array([r["forward_vector"][:2] for r in rec], dtype=float),
        right=np.array([r["right_vector"][:2] for r in rec], dtype=float),
        # get_angular_velocity() is DEGREES/s (see the unit note in comfort_signals)
        wz_deg=np.array([r["angular_velocity"][2] for r in rec], dtype=float),
    )
    # Heading from the forward vector, NOT from rotation[]. metric_info's rotation is written by
    # leaderboard/autoagents/autonomous_agent.py:150 as [roll, pitch, yaw] -- yaw last, not the
    # [pitch, yaw, roll] order carla.Rotation's own constructor uses. Deriving it from
    # forward_vector sidesteps the ordering entirely and was checked to match rotation[2] exactly.
    run["yaw_deg"] = np.degrees(np.arctan2(run["fwd"][:, 1], run["fwd"][:, 0]))

    meta_paths = sorted(glob.glob(os.path.join(run_dir, "meta", "*.json")),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    meta_idx = np.asarray([int(os.path.splitext(os.path.basename(p))[0]) for p in meta_paths])
    stride = 1
    if len(meta_idx) and meta_idx.max() * 2 < ticks.max():
        # Old `frame = self.step // 10` dump: indices only span a tenth of the run.
        stride = int(round(ticks.max() / max(meta_idx.max(), 1)))
        print(f"  [{label}] meta/ uses the legacy step//{stride} naming -- rescaling to tick index")
    meta_tick = np.clip(meta_idx * stride, 0, ticks.max())

    # One pass over meta/, not one per column -- these dumps are ~800 files.
    keys = ("speed", "steer", "throttle", "brake", "v_y_hat")
    cols = {k: np.full(len(ticks), np.nan) for k in keys}
    for p, tk in zip(meta_paths, meta_tick):
        with open(p) as f:
            d = json.load(f)
        i = int(np.searchsorted(ticks, tk))
        if i >= len(ticks):
            continue
        for k in keys:
            v = d.get(k)
            if v is not None and np.isscalar(v):
                cols[k][i] = float(v)
    run.update(cols)
    run["meta_stride"] = stride
    run["is_mpc"] = np.isfinite(run["v_y_hat"]).any()
    print(f"  [{label}] {len(ticks)} ticks ({ticks.max()*TICK_DT_S:.2f}s), "
          f"{len(meta_paths)} meta frames, controller="
          f"{'MpcKfController' if run['is_mpc'] else 'PIDController (stock)'}")
    return run


# --------------------------------------------------------------------------- derived signals

def _sg(y, deriv=0):
    """savgol with B2D's window/polyorder. delta is the REAL tick period, 0.05 s."""
    win = min(SG_WINDOW, len(y) if len(y) % 2 else len(y) - 1)
    if win <= SG_POLY:
        return np.asarray(y, dtype=float)
    return savgol_filter(np.asarray(y, dtype=float), window_length=win, polyorder=SG_POLY,
                         deriv=deriv, delta=TICK_DT_S)


def comfort_signals(run, raw_b2d=False):
    r"""The six channels B2D's Comfortness metric scores, from the same metric_info.json fields it
    reads, with the same savgol smoothing (window 7, polyorder 2).

    THREE DELIBERATE CORRECTIONS to Bench2Drive's own efficiency_smoothness_benchmark.py, all
    verifiable in that file; pass raw_b2d=True to reproduce it exactly instead.

      1. UNITS. It feeds `angular_velocity[2]` straight into bounds of 0.95 and 1.93 rad/s, but
         metric_info.json is built from hero_actor.get_angular_velocity(), which CARLA returns in
         DEGREES/s (leaderboard/autoagents/autonomous_agent.py). The bound is therefore effectively
         0.95 deg/s = 0.017 rad/s, which essentially no normal driving passes. Converted here.
      2. YAW ACCELERATION IS NOT DIFFERENTIATED. Lines 91-103 of that file compute
         `_z_yaw_acc = savgol_filter(_z_yaw_rate, polyorder, window_length)` and then
         `_z_yaw_rate = savgol_filter(_z_yaw_rate, polyorder, window_length)` -- the same call
         twice, with no `deriv=1` on the first. Its "yaw acceleration" channel is literally the
         same array as its yaw rate, merely checked against a different bound. Differentiated here.
      3. SAMPLE PERIOD. It passes `time_interval = 0.1` as the derivative delta, but metric_info
         is written every tick and the loop runs at 20 Hz, so the true spacing is 0.05 s. Every
         jerk it reports is therefore half the real value. Uses 0.05 here.

    None of this changes the A/B conclusion -- the same transform is applied to both runs -- but
    the absolute numbers here will NOT match a Comfortness score printed by that script, and the
    figures say so.
    """
    dt = 0.1 if raw_b2d else TICK_DT_S
    wz = run["wz_deg"] if raw_b2d else np.radians(run["wz_deg"])

    lon = np.einsum("ij,ij->i", run["acc"], run["fwd"])
    lat = np.einsum("ij,ij->i", run["acc"], run["right"])
    mag = np.hypot(run["acc"][:, 0], run["acc"][:, 1])

    def sg(y, deriv=0):
        win = min(SG_WINDOW, len(y) if len(y) % 2 else len(y) - 1)
        if win <= SG_POLY:
            return np.asarray(y, dtype=float)
        return savgol_filter(np.asarray(y, dtype=float), window_length=win,
                             polyorder=SG_POLY, deriv=deriv, delta=dt)

    yaw_rate = sg(wz)
    yaw_acc = sg(wz) if raw_b2d else sg(wz, deriv=1)
    return dict(
        yaw_rate=yaw_rate, yaw_acc=yaw_acc,
        a_x=sg(lon), a_y=sg(lat),
        lon_jerk=sg(sg(lon), deriv=1), jerk_total=sg(sg(mag), deriv=1),
    )


def v_y_ground_truth(run):
    """Body-frame lateral velocity from CARLA's own pose: differentiate location (savgol, so the
    derivative is smoothed rather than a raw difference of a quantized position) and project onto
    the hero's right_vector. This is the signal the MPC's VyKalmanFilter is estimating -- nothing
    in the dump records v_y directly, which is exactly why the filter exists."""
    vx_w = _sg(run["loc"][:, 0], deriv=1)
    vy_w = _sg(run["loc"][:, 1], deriv=1)
    return np.einsum("ij,ij->i", np.stack([vx_w, vy_w], axis=1), run["right"])


def collisions(run_dir):
    """Collision positions from the run's results.json, if it sits next to the dump. Used to mark
    fig 4 -- a driven line that suddenly bends is much easier to read with the impact annotated."""
    for cand in (os.path.join(run_dir, "..", "..", "results.json"),
                 os.path.join(run_dir, "..", "results.json"),
                 os.path.join(run_dir, "results.json")):
        if os.path.exists(cand):
            try:
                rec = json.load(open(cand))["_checkpoint"]["records"][0]
            except Exception:
                continue
            out = []
            for msg in rec["infractions"].get("collisions_vehicle", []) + \
                       rec["infractions"].get("collisions_layout", []) + \
                       rec["infractions"].get("collisions_pedestrian", []):
                # "... at (x=-58.919, y=131.594, z=-0.005)" -- note the leading "x=" is consumed by
                # the split itself, so only the y/z fields still carry a "name=" prefix. Pulling
                # the numbers with a regex instead of splitting on "=" avoids that asymmetry.
                nums = re.findall(r"[-+]?\d*\.?\d+", msg.split("(x=")[-1]) if "(x=" in msg else []
                if len(nums) >= 2:
                    out.append((float(nums[0]), float(nums[1])))
            return out, rec["scores"]
    return [], None


# --------------------------------------------------------------------------- map geometry

def lane_geometry(xodr_path, bbox, spacing=0.5, margin=25.0):
    """Lane centre lines and lane EDGES around the driven area, straight out of the OpenDRIVE file.

    Built with carla.Map(name, xodr_string), which parses the map offline -- no CARLA server, no
    GPU, nothing to launch. Each waypoint carries its own lane_width, so the two edges are just
    centre +- (width/2) * the waypoint's right vector; that is the actual lane boundary the
    leaderboard's OUTSIDE_ROUTE_LANES_INFRACTION criterion is about, not a drawn approximation.

    generate_waypoints() returns an unordered soup, which plots as dotted noise. Grouping by
    (road_id, lane_id) and sorting each group by its own longitudinal coordinate `s` recovers the
    actual polylines, so the boundaries draw as continuous lines. Groups are additionally split
    wherever consecutive samples jump much further than `spacing` -- one (road, lane) pair can
    reappear on disjoint stretches, and without the split those get joined by a long false chord
    straight across the map.

    Each edge also carries its OpenDRIVE lane-marking class, because "did it leave its lane" reads
    very differently depending on which line was crossed. Town10HD's route corridor uses three:

        SolidSolid / Yellow  -- the centre line, dividing opposing traffic. Crossing this puts the
                                ego into oncoming lanes.
        Solid      / White   -- road edge; beyond it is kerb/sidewalk, not a driving lane.
        Broken     / White   -- an ordinary lane divider between same-direction lanes, legal to
                                cross, and NOT an infraction on its own (see criterion_flags()).

    Returns (edges, centres): centres is a list of (N, 2) polylines; edges is a list of
    (polyline, marking_class) pairs with marking_class one of "centre", "edge", "broken", "none".
    """
    with open(xodr_path) as f:
        cmap = carla.Map(os.path.splitext(os.path.basename(xodr_path))[0], f.read())
    x0, x1, y0, y1 = bbox

    def klass(marking):
        t, c = str(marking.type), str(marking.color)
        if t == "NONE":
            return "none"
        if c == "Yellow" or t in ("SolidSolid", "BrokenSolid", "SolidBroken"):
            return "centre"
        if t == "Broken" or t == "BrokenBroken":
            return "broken"
        return "edge"

    groups = {}
    for wp in cmap.generate_waypoints(spacing):
        loc = wp.transform.location
        if not (x0 - margin <= loc.x <= x1 + margin and y0 - margin <= loc.y <= y1 + margin):
            continue
        yaw = math.radians(wp.transform.rotation.yaw)
        rx, ry = -math.sin(yaw), math.cos(yaw)   # CARLA's right vector for that heading
        h = wp.lane_width / 2.0
        key = (wp.road_id, wp.lane_id)
        groups.setdefault(key, []).append((
            wp.s, loc.x, loc.y,
            loc.x - rx * h, loc.y - ry * h, klass(wp.left_lane_marking),
            loc.x + rx * h, loc.y + ry * h, klass(wp.right_lane_marking)))

    edges, centres = [], []
    for pts in groups.values():
        pts.sort(key=lambda p: p[0])
        xy = np.array([[p[1], p[2]] for p in pts], dtype=float)
        gap = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        cuts = (np.flatnonzero(gap > max(spacing * 4.0, 2.0)) + 1).tolist()
        for a, b in zip([0] + cuts, cuts + [len(pts)]):
            seg = pts[a:b]
            if len(seg) < 2:
                continue
            centres.append(np.array([[p[1], p[2]] for p in seg], dtype=float))
            # A lane's marking can change partway along it, so each side is split again wherever
            # its class changes -- otherwise a run of centre line and a run of broken line would
            # be drawn as one polyline in whichever style happened to come first.
            for xi, yi, ki in ((3, 4, 5), (6, 7, 8)):
                start = 0
                for i in range(1, len(seg) + 1):
                    if i == len(seg) or seg[i][ki] != seg[start][ki]:
                        if i - start >= 2:
                            edges.append((np.array([[p[xi], p[yi]] for p in seg[start:i]], dtype=float),
                                          seg[start][ki]))
                        start = i
    return edges, centres


def criterion_flags(run, xodr_path):
    r"""Per-tick reconstruction of what the leaderboard's OUTSIDE_ROUTE_LANES criterion reacts to.

    That criterion (scenario_runner .../atomic_criteria.py, OutsideRouteLanesTest) accumulates
    distance while EITHER of two conditions holds, and reports the total as a percentage of the
    route driven. They are different things and worth separating on the picture:

      outside : `distance to the nearest Driving-or-Parking lane centre > lane_width/2 + 0.5 m`
                -- the ego is off the drivable surface (kerb, median, sidewalk), whatever the
                markings say.
      wrong   : the ego is on a Driving lane whose direction opposes its own heading -- crossing
                a centre line into oncoming traffic is how this happens. The real criterion is
                stateful (hysteresis, a junction exemption, and a "was it going back in?" flip),
                so what is computed here is the plain instantaneous test with the same junction
                exemption -- a diagnostic overlay, NOT a reproduction of the reported number.

    Crossing a BROKEN white line into an adjacent same-direction lane trips neither. That is the
    short answer to "is a lane departure only about the centre line": no -- but the centre line is
    the one that turns it into the `wrong` case.
    """
    with open(xodr_path) as f:
        cmap = carla.Map(os.path.splitext(os.path.basename(xodr_path))[0], f.read())
    loc, yaw = run["loc"], run["yaw_deg"]
    outside = np.zeros(len(loc), dtype=bool)
    wrong = np.zeros(len(loc), dtype=bool)
    for i, (p, yw) in enumerate(zip(loc, yaw)):
        here = carla.Location(x=float(p[0]), y=float(p[1]), z=0.0)
        dwp = cmap.get_waypoint(here, lane_type=carla.LaneType.Driving)
        pwp = cmap.get_waypoint(here, lane_type=carla.LaneType.Parking)
        best_d, best_w = float("inf"), 3.5
        for wp in (dwp, pwp):
            if wp is None:
                continue
            d = here.distance(wp.transform.location)
            if d < best_d:
                best_d, best_w = d, wp.lane_width
        outside[i] = best_d > best_w / 2.0 + 0.5
        if dwp is not None and not dwp.is_junction:
            wrong[i] = abs((dwp.transform.rotation.yaw - yw + 180.0) % 360.0 - 180.0) > 90.0
    step = np.concatenate([[0.0], np.hypot(np.diff(loc[:, 0]), np.diff(loc[:, 1]))])
    return dict(outside=outside, wrong=wrong,
                dist_outside=float(step[outside].sum()), dist_wrong=float(step[wrong].sum()),
                dist_total=float(step.sum()))


# --------------------------------------------------------------------------- figures

def _line(ax, t, y, color, label=None, linewidth=1.4, linestyle="-"):
    """Plot a series that may be sparse. Channels read from meta/ only exist on dumped frames, so
    a run dumped every 10th tick leaves NaN in between -- matplotlib then draws nothing at all
    between isolated finite samples and the curve silently vanishes. Masking to the finite samples
    keeps the line continuous, and a sparse series additionally gets markers so it is honest about
    being sampled rather than pretending to be a 20 Hz trace."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if not ok.any():
        return
    sparse = ok.sum() < 0.3 * len(y)
    ax.plot(t[ok], y[ok], color=color, linewidth=linewidth, linestyle=linestyle,
            solid_capstyle="round", label=label,
            marker="." if sparse else None, markersize=4)


def _series(ax, runs, key, sig=None, ylabel="", title="", limits=None):
    for run, color in zip(runs, _colors(runs)):
        y = (sig[run["label"]][key] if sig is not None else run[key])
        _line(ax, run["t"], y, color, label=run["label"])
    if limits is not None:
        V._b2d_limit_lines(ax, *limits)
    ax.set_ylabel(ylabel)
    V._title(ax, title)
    V._style_axes(ax)


def fig_comfort(runs, limits, raw_b2d, out_dir):
    """fig 1: the six B2D Comfortness channels, both controllers, with the B2D bounds in red."""
    sig = {r["label"]: comfort_signals(r, raw_b2d=raw_b2d) for r in runs}
    note = ("B2D script reproduced verbatim (deg/s yaw, undifferentiated yaw 'accel', dt=0.1)"
            if raw_b2d else
            "yaw converted deg/s -> rad/s, yaw accel actually differentiated, dt = 0.05 s "
            "(see comfort_signals() docstring)")
    fig, ax = V._panels(f"Ride comfort — B2D Comfortness channels vs their limits\n{note}",
                        n_rows=3, n_cols=2, figsize=(15, 11))

    panels = [
        ("yaw_rate",   "yaw rate (rad/s)",            "Yaw rate"),
        ("yaw_acc",    "yaw accel (rad/s$^2$)",       "Yaw acceleration"),
        ("a_x",        "$a_x$ (m/s$^2$)",             "Longitudinal acceleration"),
        ("a_y",        "$a_y$ (m/s$^2$)",             "Lateral acceleration"),
        ("lon_jerk",   "jerk (m/s$^3$)",              "Longitudinal jerk"),
        ("jerk_total", "|jerk| (m/s$^3$)",            "Total jerk magnitude"),
    ]
    for a, (key, ylab, title) in zip(ax, panels):
        _series(a, runs, key, sig=sig, ylabel=ylab, title=title, limits=limits[key])
    ax[0].legend(frameon=False, fontsize=9, loc="upper right")
    for a in ax[-2:]:
        a.set_xlabel("t (s)")

    # Violation tally per channel, so the comparison is quotable as numbers and not only as shapes.
    print("\n  ticks outside the B2D limit (lower is better)")
    print(f"    {'channel':11s} {'limit':>18s}  " + "  ".join(f"{r['label']:>16s}" for r in runs))
    for key, _ylab, _t in panels:
        lo, hi = limits[key]
        cells = []
        for r in runs:
            y = sig[r["label"]][key]
            n = int(((y < lo) | (y > hi)).sum())
            cells.append(f"{n:5d} / {len(y):4d}  ")
        print(f"    {key:11s} [{lo:+7.2f}, {hi:+6.2f}]  " + "  ".join(c.rjust(16) for c in cells))
    return V._save(fig, out_dir, "fig1_comfort"), fig


def fig_control(runs, out_dir):
    """fig 2: what each controller actually commanded, plus the states those commands produced."""
    fig, ax = V._panels("Control inputs and vehicle state — PID vs MPC-KF",
                        n_rows=3, n_cols=2, figsize=(15, 11))

    _series(ax[0], runs, "speed", ylabel="$v_x$ (m/s)", title="Speed")
    for run, color in zip(runs, _colors(runs)):
        # unwrapped so a heading crossing +-180 deg does not draw a full-scale vertical jump
        _line(ax[1], run["t"], np.degrees(np.unwrap(np.radians(run["yaw_deg"]))),
              color, label=run["label"])
    ax[1].set_ylabel("yaw (deg)"); V._title(ax[1], "Heading (unwrapped)"); V._style_axes(ax[1])

    _series(ax[2], runs, "steer", ylabel="steer [-1, 1]", title="Steering command")
    _series(ax[3], runs, "throttle", ylabel="throttle [0, 1]", title="Throttle command")
    _series(ax[4], runs, "brake", ylabel="brake [0, 1]", title="Brake command")

    ax[5].axhline(0.0, color=V.COLOR_AXIS, linewidth=1)
    for run, color in zip(runs, _colors(runs)):
        _line(ax[5], run["t"], run["throttle"] - run["brake"], color, label=run["label"])
    ax[5].set_ylim(-1.05, 1.05)
    ax[5].set_ylabel("$u$")
    V._title(ax[5], "Pedal command $u$ = throttle - brake")
    V._style_axes(ax[5])

    ax[0].legend(frameon=False, fontsize=9, loc="upper right")
    for a in ax[-2:]:
        a.set_xlabel("t (s)")
    return V._save(fig, out_dir, "fig2_control"), fig


def fig_kalman(run, out_dir):
    """fig 3: MPC-KF only -- does VyKalmanFilter actually estimate v_y?

    Ground truth is differentiated from CARLA's own pose (v_y_ground_truth), which the filter never
    sees; it runs off the IMU's yaw rate and lateral acceleration alone. The stock PID has no v_y
    estimate at all, so there is nothing to overlay for it -- this figure is single-controller by
    construction, not by omission.
    """
    gt = v_y_ground_truth(run)
    est = run["v_y_hat"]
    ok = np.isfinite(est) & np.isfinite(gt)

    fig, ax = V._panels("Lateral velocity — Kalman filter estimate vs ground truth  (MPC-KF)",
                        n_rows=2, n_cols=1, figsize=(15, 8))
    _line(ax[0], run["t"], gt, V.COLOR_MUTED, linewidth=2.0, linestyle="--",
          label="$v_y$ ground truth (differentiated CARLA pose)")
    _line(ax[0], run["t"], est, V.COLOR_BLUE, linewidth=1.5,
          label="$\\hat{v}_y$ Kalman filter")
    ax[0].set_ylabel("$v_y$ (m/s)")
    V._title(ax[0], "Estimate vs truth")
    ax[0].legend(frameon=False, fontsize=9, loc="upper right")
    V._style_axes(ax[0])

    err = est - gt
    ax[1].axhline(0.0, color=V.COLOR_AXIS, linewidth=1)
    _line(ax[1], run["t"], err, V.COLOR_RED, linewidth=1.3)
    ax[1].fill_between(run["t"][ok], err[ok], 0, color=V.COLOR_RED, alpha=0.12)
    ax[1].set_ylabel("error (m/s)")
    ax[1].set_xlabel("t (s)")
    stats = ""
    if ok.sum() > 2:
        rmse = float(np.sqrt(np.mean(err[ok] ** 2)))
        corr = float(np.corrcoef(est[ok], gt[ok])[0, 1])
        stats = f"   RMSE = {rmse:.4f} m/s,  corr = {corr:+.3f},  n = {int(ok.sum())}"
        print(f"\n  Kalman filter v_y: RMSE {rmse:.4f} m/s, corr {corr:+.3f}, "
              f"gt std {np.std(gt[ok]):.4f}, est std {np.std(est[ok]):.4f}")
    V._title(ax[1], "Estimation error $\\hat{v}_y - v_y$" + stats)
    V._style_axes(ax[1])
    return V._save(fig, out_dir, "fig3_kalman_vy"), fig


def fig_trajectory(runs, xodr_path, out_dir):
    """fig 4: both driven lines on the real lane geometry, so a lane departure is visible as one."""
    allxy = np.concatenate([r["loc"] for r in runs], axis=0)
    bbox = (allxy[:, 0].min(), allxy[:, 0].max(), allxy[:, 1].min(), allxy[:, 1].max())
    edges, centres = lane_geometry(xodr_path, bbox)

    # Equal aspect is mandatory here -- a lane departure is a distance, and any x/y stretch would
    # make a 1 m excursion look like anything at all. The figure is sized to the route's own
    # aspect ratio instead, so "equal" does not leave most of the canvas empty.
    # A run that fails early leaves a very short driven line, and padding it by a fixed margin
    # would crop the map down to a few metres of anonymous asphalt with no junction, kerb or
    # branch in view -- exactly the context needed to see WHY it went wrong. Pad by the larger of
    # a fixed margin and the route's own extent.
    span = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    pad = max(12.0, 40.0 - span)
    xlim = (bbox[0] - pad, bbox[1] + pad)
    ylim = (bbox[2] - pad, bbox[3] + pad)
    span_x, span_y = xlim[1] - xlim[0], ylim[1] - ylim[0]
    height = float(np.clip(16.0 * span_y / max(span_x, 1e-6), 3.5, 12.0))

    fig, ax = plt.subplots(figsize=(16, height), constrained_layout=True)
    fig.patch.set_facecolor(V.COLOR_BG)
    for k, seg in enumerate(centres):
        ax.plot(seg[:, 0], seg[:, 1], color=V.COLOR_AXIS, linewidth=0.6, alpha=0.4,
                linestyle=(0, (1, 5)), zorder=1,
                label="lane centre" if k == 0 else None)
    # Marking style follows what the line actually is, so "did it leave its lane" is readable:
    # the yellow centre line is the one that means oncoming traffic, a broken white line is legal
    # to cross, and the solid white edge is where the drivable surface stops.
    STYLE = {"centre": dict(color="#c9a227", linewidth=2.0, linestyle="-",
                            label="centre line (SolidSolid/Yellow) — opposing traffic"),
             "edge":   dict(color=V.COLOR_MUTED, linewidth=1.5, linestyle="-",
                            label="road edge (Solid/White)"),
             "broken": dict(color=V.COLOR_AXIS, linewidth=1.2, linestyle=(0, (6, 6)),
                            label="lane divider (Broken/White) — crossing is legal"),
             "none":   dict(color=V.COLOR_AXIS, linewidth=0.6, linestyle=(0, (2, 6)),
                            label=None)}
    seen = set()
    for seg, cls in edges:
        st = dict(STYLE[cls])
        if cls in seen:
            st["label"] = None
        seen.add(cls)
        ax.plot(seg[:, 0], seg[:, 1], zorder=2, alpha=0.9, **st)

    for run, color in zip(runs, _colors(runs)):
        xy = run["loc"]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.2,
                solid_capstyle="round", zorder=3, label=f"{run['label']} driven line")
        flags = run.get("flags")
        if flags is not None:
            for key, col, lab in (("wrong", V.COLOR_RED, "in an opposing lane"),
                                  ("outside", V.COLOR_ORANGE, "off the drivable surface")):
                mask = flags[key]
                if mask.any():
                    ax.scatter(xy[mask, 0], xy[mask, 1], s=26, color=col, zorder=4, alpha=0.85,
                               label=(lab if lab not in seen else None))
                    seen.add(lab)
        ax.scatter(*xy[0], s=70, color=color, marker="o", zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.scatter(*xy[-1], s=110, color=color, marker="*", zorder=5,
                   edgecolor="white", linewidth=1.0)
        hits, scores = collisions(run["dir"])
        for hx, hy in hits:
            ax.scatter(hx, hy, s=170, marker="X", color=V.COLOR_RED, zorder=6,
                       edgecolor="white", linewidth=1.2)
            ax.annotate(f"{run['label']} collision", (hx, hy), textcoords="offset points",
                        xytext=(8, 10), fontsize=9, color=V.COLOR_RED)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])   # CARLA's y axis points "down" in world coords; flipping it
                                    # makes the plot read like a map, not a mirror image of one
    ax.set_xlabel("x (m, CARLA world)")
    ax.set_ylabel("y (m, CARLA world)")
    town = os.path.splitext(os.path.basename(xodr_path))[0]
    V._title(ax, f"Driven trajectories over {town} lane geometry  "
                 "(o = start, * = end, X = collision)")
    ax.legend(frameon=False, fontsize=10, loc="best")
    V._style_axes(ax)
    return V._save(fig, out_dir, "fig4_trajectory"), fig


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mpc-dir", required=True, help="MPC-KF dump (holds metric_info.json + meta/)")
    ap.add_argument("--pid-dir", default=None,
                    help="stock-PID dump of the SAME route. Optional: with no baseline yet, "
                         "the figures still render for the MPC run alone")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--xodr", default=None,
                    help="OpenDRIVE map for fig 4. Default: inferred from the dump directory "
                         "name, which carries the town (…_RouteScenario_<id>_rep0_<Town>_…)")
    ap.add_argument("--raw-b2d", action="store_true",
                    help="reproduce Bench2Drive's comfort signals verbatim, bugs included "
                         "(see comfort_signals() docstring) instead of the corrected ones")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading runs ...")
    mpc = load_run(os.path.abspath(args.mpc_dir), "MPC-KF")
    # PID first so the baseline draws underneath in muted grey and the MPC sits on top in blue.
    runs = [mpc]
    if args.pid_dir:
        runs = [load_run(os.path.abspath(args.pid_dir), "PID (stock)"), mpc]
    else:
        print("  (--pid-dir 없음: MPC 단독 렌더 -- 비교 패널은 한 줄만 그려집니다)")

    xodr = args.xodr or infer_xodr(args.mpc_dir)
    print(f"  map: {xodr}")
    if not mpc["is_mpc"]:
        print("  ! --mpc-dir has no v_y_hat in meta/ -- is it really the MPC run?")

    limits, src = b2d_limits()
    print(f"  B2D limits from {src}")

    print("\n  lane-departure diagnostic (see criterion_flags() -- NOT the leaderboard's own"
          " stateful number)")
    for r in runs:
        r["flags"] = criterion_flags(r, xodr)
        f = r["flags"]
        print(f"    {r['label']:12s} driven {f['dist_total']:6.1f} m | "
              f"opposing lane {f['dist_wrong']:6.1f} m ({100*f['dist_wrong']/f['dist_total']:4.1f}%) | "
              f"off drivable {f['dist_outside']:5.1f} m ({100*f['dist_outside']/f['dist_total']:4.1f}%)")

    out = []
    out.append(fig_comfort(runs, limits, args.raw_b2d, args.out_dir)[0])
    out.append(fig_control(runs, args.out_dir)[0])
    if mpc["is_mpc"]:
        out.append(fig_kalman(mpc, args.out_dir)[0])
    out.append(fig_trajectory(runs, xodr, args.out_dir)[0])

    print()
    for p in out:
        print(f"Figure saved: {p}")


if __name__ == "__main__":
    main()
