"""Everything the path-tracking scripts do that isn't control: cameras, live view, metrics, plots.

Split out so the main scripts only contain the parts relevant to debugging the controller. The
former bev_view.py and video_recorder.py modules live here too -- one import for all of it.

Layout
    1. run identity         (run_name)
    2. palette + axis styling
    3. spectator camera     (follow_with_spectator)
    4. video recording      (VIEWS, VideoRecorder)
    5. live BEV view        (BevView)
    6. metrics              (rmse/max/mean, print_error_summary)
    7. plotting internals   (panel grids, series lookup, rmse badges)
    8. result figures       (plot_results)

Both VideoRecorder and BevView follow the same lifecycle: construct once after the vehicle is
spawned, call close() from the caller's finally block.

plot_results() draws three figures:
    fig 1  lateral tracking     -- cross-track error, heading error, yaw, yaw rate, yaw accel,
                                   v_y, a_y, steer
    fig 2  longitudinal tracking-- speed error, speed, a_x, jerk, total jerk magnitude,
                                   throttle + brake (combined)
    fig 3  trajectory           -- desired path vs. ego trajectory (equal aspect, so its own figure)

Panels whose series a given stack does not record are drawn as an explicit "not recorded" note
rather than crashing, so older scripts that log a smaller hist dict still plot.
"""
import datetime
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time

import carla

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projection
import numpy as np


# ---------------------------------------------------------------------------
# 1. run identity
# ---------------------------------------------------------------------------

# Stamped once at import, i.e. once per process, so every artifact of a run -- the figures and the
# mp4 -- carries the same <script>_<date>_<time> stem and they pair up in the output directories.
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_name(suffix="", name=None):
    """<script>_<date>_<time>[_suffix], the shared stem for this run's output files.

    name defaults to the script that was run, so stanley_PID.py writes stanley_PID_20260807_181500
    and a future controller names its own files without touching this module.
    """
    if name is None:
        main = sys.modules.get("__main__")
        path = getattr(main, "__file__", None) or sys.argv[0]
        name = os.path.splitext(os.path.basename(path))[0] or "run"
    stem = f"{name}_{RUN_STAMP}"
    return f"{stem}_{suffix}" if suffix else stem


# ---------------------------------------------------------------------------
# 2. palette + axis styling
# ---------------------------------------------------------------------------

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
COLOR_PURPLE = "#8b5cf6"    # jerk / derived series

# cycled across runs when a plot function is handed {label: hist} instead of one hist -- index 0
# is COLOR_BLUE, so a single-run call still gets the same look it always had
COMPARE_COLORS = [COLOR_BLUE, COLOR_ORANGE, COLOR_AQUA, COLOR_RED, COLOR_PURPLE]

# Shared stroke/type weights -- every plot function in this module should read these rather than
# hardcode its own numbers, so retuning one constant retunes every figure at once. Values match
# what was already hardcoded throughout this file before this was pulled out, so introducing the
# knob does not itself change how anything currently looks.
LINEWIDTH = 1.5          # primary data series
LINEWIDTH_THIN = 1.0     # reference lines: axhline/axvline, zero lines, gridlines
MARKERSIZE = 28          # scatter marker area (matplotlib's `s=`)
FONTSIZE_TITLE = 14      # figure suptitle
FONTSIZE_SUBTITLE = 11   # per-axes title
FONTSIZE_LABEL = 10      # axis labels
FONTSIZE_TICK = 9        # tick labels
FONTSIZE_LEGEND = 9      # legend text


def _style_axes(ax):
    ax.set_facecolor(COLOR_BG)
    ax.grid(True, color=COLOR_GRID, linewidth=LINEWIDTH_THIN * 0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=FONTSIZE_TICK)
    ax.title.set_color(COLOR_INK)
    ax.title.set_fontsize(FONTSIZE_SUBTITLE)
    ax.xaxis.label.set_color(COLOR_MUTED)
    ax.yaxis.label.set_color(COLOR_MUTED)
    ax.xaxis.label.set_fontsize(FONTSIZE_LABEL)
    ax.yaxis.label.set_fontsize(FONTSIZE_LABEL)


def _legend(ax, **kwargs):
    ax.legend(frameon=False, labelcolor=COLOR_INK, fontsize=FONTSIZE_LEGEND, **kwargs)


# ---------------------------------------------------------------------------
# 3. spectator camera
# ---------------------------------------------------------------------------

# the framing of the CARLA 3D window, shared with VIEWS["chase"] so the two cannot drift apart
CHASE_BACK, CHASE_UP, CHASE_PITCH = 8.0, 4.0, -15.0


def follow_with_spectator(world, vehicle, back=CHASE_BACK, up=CHASE_UP, pitch=CHASE_PITCH):
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


# ---------------------------------------------------------------------------
# 4. video recording
# ---------------------------------------------------------------------------

# Mounts are body-frame offsets from the actor origin, the same origin follow_with_spectator
# measures from -- so the chase view reproduces the CARLA 3D window and the others are the usual
# demo angles.
VIEWS = {
    "chase": (carla.Location(x=-CHASE_BACK, z=CHASE_UP), carla.Rotation(pitch=CHASE_PITCH)),
    "hood":  (carla.Location(x=1.2, z=1.3), carla.Rotation(pitch=0.0)),
    "front": (carla.Location(x=-5.5, z=2.2), carla.Rotation(pitch=-8.0)),
    "top":   (carla.Location(x=0.0, z=28.0), carla.Rotation(pitch=-90.0)),
}


class VideoRecorder:
    """Record the run to an mp4 through a CARLA RGB camera attached to the ego vehicle.

    Why a sensor and not a screen recorder: the control scripts run the world in synchronous mode,
    so the camera delivers exactly one image per world.tick() -- the video's frame rate is the
    simulation rate (1/dt), independent of how fast the loop actually runs on the wall clock. A run
    sped up with --times-run, or slowed down by a heavy MPC solve, still comes out as a correct
    real-time video.

    Encoding happens on a writer thread so the sensor callback (which runs on CARLA's own listener
    thread and must return fast) only pays for a BGRA->RGB copy. ffmpeg comes from imageio-ffmpeg,
    a static binary inside the venv -- nothing is installed system-wide.

        rec = VideoRecorder(world, vehicle, "drive.mp4", fps=1.0 / args.dt)
        ...
        rec.close()
    """

    def __init__(self, world, vehicle, out_path, fps=20.0, width=1280, height=720,
                 view="chase", fov=90.0, quality=6):
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; pick one of {sorted(VIEWS)}")
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.out_path = out_path
        self.width, self.height = width, height
        self.frames = 0
        self._dropped = 0

        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(width))
        blueprint.set_attribute("image_size_y", str(height))
        blueprint.set_attribute("fov", str(fov))
        # no sensor_tick: leave it at 0 so the camera fires on every tick of the synchronous world

        location, rotation = VIEWS[view]
        # Rigid, not SpringArmGhost: a spring arm treats the offset as an arm and rotates it by the
        # mount's own pitch, so the chase view asking for 4 m up at -15 deg actually sat at 1.8 m
        # (and "top" ended up 28 m *behind* the car at ground level). Rigid places the camera
        # literally at the offset, which is what makes the recording match the spectator window.
        # The cost is that the camera now inherits the body's pitch and roll, so the horizon dips a
        # degree or two under braking where the spectator stays level.
        self.camera = world.spawn_actor(blueprint, carla.Transform(location, rotation),
                                        attach_to=vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)

        # bounded so a slow encoder throttles into dropped frames instead of eating all the RAM
        self._queue = queue.Queue(maxsize=240)
        self._writer_thread = threading.Thread(target=self._write_loop, args=(fps, quality),
                                               daemon=True)
        self._writer_thread.start()
        self.camera.listen(self._on_image)
        print(f"Recording to {out_path} ({width}x{height} @ {fps:.0f}fps, {view} view)")

    def _on_image(self, image):
        """Runs on CARLA's listener thread -- convert and hand off, nothing slow here."""
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        try:
            self._queue.put_nowait(bgra[:, :, [2, 1, 0]].tobytes())  # BGRA -> RGB24
        except queue.Full:
            self._dropped += 1

    def _write_loop(self, fps, quality):
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            self.out_path, size=(self.width, self.height), fps=fps, quality=quality,
            macro_block_size=1, ffmpeg_log_level="error",
        )
        writer.send(None)  # seed the generator; this is what launches ffmpeg
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                writer.send(frame)
                self.frames += 1
        finally:
            writer.close()

    def close(self):
        if self.camera is not None and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()
        self.camera = None
        self._queue.put(None)          # sentinel: drains whatever is still queued, then closes ffmpeg
        self._writer_thread.join(timeout=60.0)
        if self._dropped:
            print(f"  warning: encoder fell behind, {self._dropped} frames dropped")
        print(f"Recording done: {self.out_path} ({self.frames} frames)")
        return self.out_path


# ---------------------------------------------------------------------------
# 5. live BEV view
# ---------------------------------------------------------------------------

def _nearest_index(x, y, path_x, path_y, last_idx, search_window=30):
    """Same forward-only nearest-neighbor search as lateral_error() in the control script, kept as
    an independent copy here so BevView never has to read the control script's own last_idx."""
    lo = last_idx
    hi = min(len(path_x), last_idx + search_window)
    dists = [math.hypot(x - path_x[i], y - path_y[i]) for i in range(lo, hi)]
    return lo + dists.index(min(dists))


def _find_vehicle(world, near_x, near_y, max_dist=5.0, max_ticks=40):
    """world.get_actors() (bulk, unfiltered) only reflects a freshly-spawned actor after at least
    one tick has happened -- and the control script may not have called world.tick() yet by the
    time it constructs BevView. Since the world is already in synchronous mode by then, nudge it
    ourselves (a client other than the one driving the main loop is allowed to tick it too) until
    the actor list catches up."""
    for _ in range(max_ticks):
        best, best_d = None, None
        for actor in world.get_actors().filter("vehicle.*"):
            d = math.hypot(actor.get_location().x - near_x, actor.get_location().y - near_y)
            if best is None or d < best_d:
                best, best_d = actor, d
        if best is not None and best_d <= max_dist:
            return best
        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            time.sleep(0.1)
    raise RuntimeError("BevView: no vehicle found near the path start -- is it spawned yet?")


def _bev_process_main(vehicle_id, path_x, path_y, host, port, view_radius, trail_max_len,
                      redraw_interval_ms, stop_event, ready_event):
    """Entry point for the child process: owns its own CARLA connection and is this process's
    real main thread, so it's safe for matplotlib/Tk to live here for the process's whole life."""
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    vehicle = world.get_actor(vehicle_id)
    if vehicle is None:
        print(f"BevView: vehicle id {vehicle_id} not found in child process; exiting.")
        return

    lock = threading.Lock()
    trail_x, trail_y = [], []
    state = {"ego_xy": None, "last_idx": 0}

    def on_tick(snapshot):
        if stop_event.is_set() or not vehicle.is_alive:
            return
        try:
            loc = vehicle.get_location()
        except RuntimeError:
            return  # actor was destroyed between the is_alive check and this call
        x, y = loc.x, loc.y
        with lock:
            state["last_idx"] = _nearest_index(x, y, path_x, path_y, state["last_idx"])
            state["ego_xy"] = (x, y)
            trail_x.append(x)
            trail_y.append(y)
            if len(trail_x) > trail_max_len:
                del trail_x[: -trail_max_len]
                del trail_y[: -trail_max_len]

    tick_id = world.on_tick(on_tick)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(COLOR_BG)
    _style_axes(ax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.plot(path_x, path_y, color=COLOR_MUTED, linewidth=1.5, linestyle="--", label="desired path")
    (trail_line,) = ax.plot([], [], color=COLOR_ORANGE, linewidth=2, label="ego trajectory")
    (ego_dot,) = ax.plot([], [], color=COLOR_BLUE, marker="o", markersize=9, linestyle="", label="ego")
    _legend(ax, loc="upper right")

    n_path = len(path_x)

    def redraw(_frame):
        if stop_event.is_set():
            plt.close(fig)
            return trail_line, ego_dot
        with lock:
            tx, ty = list(trail_x), list(trail_y)
            ego_xy = state["ego_xy"]
            last_idx = state["last_idx"]
        if ego_xy is None:
            return trail_line, ego_dot
        x, y = ego_xy
        trail_line.set_data(tx, ty)
        ego_dot.set_data([x], [y])
        ax.set_title(f"last_idx = {last_idx}/{n_path - 1}")
        r = view_radius
        ax.set_xlim(x - r, x + r)
        ax.set_ylim(y - r, y + r)
        return trail_line, ego_dot

    anim = FuncAnimation(fig, redraw, interval=redraw_interval_ms, cache_frame_data=False)
    ready_event.set()
    plt.show()  # blocks until redraw() closes fig via stop_event

    try:
        world.remove_on_tick(tick_id)
    except Exception:
        pass


class BevView:
    """Live top-down (BEV) view of the desired path and the ego vehicle -- self-driving.

    Finds the spawned vehicle and refreshes itself every simulation tick via world.on_tick(); the
    control script only has to construct a BevView once and call close() -- no per-tick push needed.
    Draws exactly: desired path (static), ego trajectory (trail), current ego position, and last_idx
    (nearest desired-path index to the ego, recomputed here independently of whatever the control
    script tracks internally -- BevView never reads the control script's state).

    Process model: matplotlib/Tk GUI calls are only safe on a process's real main thread, so the
    whole plot lives in its own child process (started here, killed in close()) rather than a
    background thread of the caller's process. Inside that child process, world.on_tick() callbacks
    still arrive on CARLA's own internal listener thread, so a lock still guards the handoff to the
    redraw timer, which runs on the child process's main thread via plt.show().
    """

    def __init__(self, path_x, path_y, host="localhost", port=2000, view_radius=40.0,
                 trail_max_len=5000, redraw_interval_ms=50):
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
        vehicle = _find_vehicle(world, path_x[0], path_y[0])

        self._stop_event = mp.Event()
        ready_event = mp.Event()
        self._process = mp.Process(
            target=_bev_process_main,
            args=(vehicle.id, list(path_x), list(path_y), host, port, view_radius, trail_max_len,
                  redraw_interval_ms, self._stop_event, ready_event),
            daemon=True,
        )
        self._process.start()
        if not ready_event.wait(timeout=5.0):
            print("BevView: GUI window didn't come up within 5s, continuing anyway.")

    def close(self):
        self._stop_event.set()
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
        plt.ioff()  # the child leaves interactive mode on; plot_results()'s plt.show() must block


# ---------------------------------------------------------------------------
# 6. metrics
# ---------------------------------------------------------------------------

def error_stats(values):
    """(rmse, max|e|, mean) of a series. Returns None for an empty series."""
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    n = len(values)
    rmse = math.sqrt(sum(v * v for v in values) / n)
    return rmse, max(abs(v) for v in values), sum(values) / n


def magnitude_stats(values):
    """(mean|v|, peak|v|) of a series. Returns None for an empty series.

    For signals that oscillate around zero by design (acceleration, jerk, yaw rate) a signed
    mean/RMSE against a zero reference isn't the interesting number -- how big the swings
    typically are, and how big they get, is. error_stats() stays the right tool for anything
    that is itself a tracking error (e_y, e_theta, speed error).
    """
    values = [abs(v) for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return sum(values) / len(values), max(values)


def speed_error_series(hist, target_speed_ms):
    """Reference-minus-measured speed, using a logged per-step reference when there is one.

    The cruise scripts hold a constant target; the MPC stack sweeps v_des, so prefer hist["v_des"]
    whenever it exists -- scoring a swept reference against a constant is meaningless.
    """
    v_des = hist.get("v_des")
    if v_des:
        return [d - v for d, v in zip(v_des, hist["v_x"])]
    return [target_speed_ms - v for v in hist["v_x"]]


def print_error_summary(hist, target_speed_ms):
    """RMSE / max / mean of each tracked error, plus mean/peak magnitude of the raw longitudinal
    and lateral dynamics signals, over the whole run.

    The dynamics rows are printed as sections that appear only when this stack actually recorded
    them -- a steer=0 run (e.g. longitudinal_PID.py) has no yaw_rate/yaw_acc/a_y, and the section
    is skipped rather than printed empty.
    """
    if not hist["t"]:
        return

    print(f"\n=== error summary: {len(hist['t'])} steps, {hist['t'][-1]:.1f} s ===")
    for name, unit, series in (("cross-track", "m", hist.get("e_y", [])),
                               ("heading    ", "deg", hist.get("e_theta", [])),
                               ("speed      ", "m/s", speed_error_series(hist, target_speed_ms))):
        stats = error_stats(series)
        if stats is None:
            continue
        rmse, peak, bias = stats
        print(f"  {name}  RMSE={rmse:7.3f} {unit:<3}  max|e|={peak:7.3f} {unit:<3}  mean={bias:+7.3f} {unit}")

    long_rows = (("accel a_x   ", "m/s^2", hist.get("a_x", [])),
                ("jerk        ", "m/s^3", hist.get("jerk", [])),
                ("|jerk| total", "m/s^3", hist.get("jerk_total", [])))
    lat_rows = (("yaw rate ", "deg/s  ", hist.get("yaw_rate", [])),
               ("yaw accel", "rad/s^2", hist.get("yaw_acc", [])),
               ("accel a_y", "m/s^2  ", hist.get("a_y", [])))

    for section, rows in (("longitudinal dynamics", long_rows), ("lateral dynamics", lat_rows)):
        printed_header = False
        for name, unit, series in rows:
            stats = magnitude_stats(series)
            if stats is None:
                continue
            if not printed_header:
                print(f"  -- {section} --")
                printed_header = True
            mean_abs, peak = stats
            print(f"  {name}  mean|.|={mean_abs:7.3f} {unit:<7}  peak|.|={peak:7.3f} {unit}")


# ---------------------------------------------------------------------------
# 7. plotting internals
# ---------------------------------------------------------------------------

def _panels(title, n_rows=3, n_cols=2, figsize=(15, 10)):
    """A styled grid sharing the time axis, flattened in row-major order."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(title, fontsize=FONTSIZE_TITLE, color=COLOR_INK)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        _style_axes(ax)
    return fig, axes


def _get(hist, key):
    """Series as an array, or None when this stack never recorded it."""
    values = hist.get(key)
    if values is None or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return arr


def _runs(data):
    """Normalize a single hist dict, or {label: hist} for a multi-controller comparison, into the
    latter. A lone hist has a top-level "t" key; a label dict does not (no controller is named
    "t"), so that's what tells the two apart."""
    return {"": data} if "t" in data else data


def _not_recorded(ax, key):
    ax.text(0.5, 0.5, f'"{key}" not recorded by this run', transform=ax.transAxes,
            ha="center", va="center", color=COLOR_MUTED, fontsize=10)


def _rmse_badge(ax, e, unit):
    """Corner box with the panel's own RMSE, so a figure is readable without the console output."""
    stats = error_stats(e)
    if stats is None:
        return
    ax.text(0.985, 0.05, f"RMSE = {stats[0]:.3f} {unit}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=COLOR_INK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=COLOR_AXIS, alpha=0.85))


def _error_multi(ax, runs, colors, multi, ylabel, panel_title, unit, series_fn):
    """One error-vs-time panel, for a single run or several overlaid.

    series_fn(hist) -> the error array for that run (or None if this stack never recorded it).
    Single run keeps the filled-band/corner-badge look; multiple runs switch to plain colored
    lines with each RMSE folded into the legend, since stacked badges stop being readable.
    """
    ax.axhline(0.0, color=COLOR_AXIS, linewidth=1, linestyle="--")
    found = False
    for label, hist in runs.items():
        e = series_fn(hist)
        if e is None:
            continue
        found = True
        t = np.asarray(hist["t"], dtype=float)
        color = colors[label]
        if not multi:
            ax.plot(t, e, color=color, linewidth=1.6, solid_capstyle="round")
            ax.fill_between(t, e, 0, color=color, alpha=0.15)
            _rmse_badge(ax, e, unit)
        else:
            stats = error_stats(e)
            tag = f"{label} (RMSE={stats[0]:.2f})" if stats else label
            ax.plot(t, e, color=color, linewidth=1.4, label=tag)
    if not found:
        _not_recorded(ax, "e")
    elif multi:
        _legend(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title)


def _dynamics_panel(ax, runs, colors, multi, key, ylabel, panel_title, fill_color=None):
    """One plain time-series panel (acceleration, jerk, yaw rate, ...), for a single run or
    several overlaid. Single run gets an optional fill; multiple runs get one colored line each
    plus a legend. "not recorded" if no run in `runs` logged `key` at all."""
    found = False
    for label, hist in runs.items():
        series = _get(hist, key)
        if series is None:
            continue
        found = True
        t = np.asarray(hist["t"], dtype=float)
        color = colors[label]
        ax.plot(t, series, color=color, linewidth=1.3 if multi else 1.5,
               solid_capstyle="round", label=label if multi else None)
        if not multi and fill_color:
            ax.fill_between(t, series, 0, color=fill_color, alpha=0.15)
    if not found:
        _not_recorded(ax, key)
    else:
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=1)
        if multi:
            _legend(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title)


def _save(fig, out_dir, stem):
    out_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    return out_path


# ---------------------------------------------------------------------------
# 8. result figures
# ---------------------------------------------------------------------------

def plot_lateral(data, title="Lateral tracking performance"):
    """fig 1: cross-track error, heading error, yaw pair, yaw rate, yaw acceleration, lateral
    velocity, lateral acceleration, steer -- for one controller, or several overlaid.

    data: a single hist dict, or {label: hist} to compare controllers on one figure (see
    plot_longitudinal, which this mirrors panel-for-panel via the same _error_multi/
    _dynamics_panel helpers). A lone hist keeps the original look; multiple runs switch each
    panel to plain colored lines -- one color per run from COMPARE_COLORS -- with RMSE folded
    into the legend for the two error panels.
    """
    runs = _runs(data)
    multi = len(runs) > 1
    fig, (ax_ey, ax_eth, ax_yaw, ax_r, ax_racc, ax_vy, ax_ay, ax_steer) = _panels(
        title, n_rows=4, figsize=(15, 13))
    colors = dict(zip(runs, COMPARE_COLORS))

    _error_multi(ax_ey, runs, colors, multi, "lateral error (m)", "Lateral error", "m",
                lambda hist: _get(hist, "e_y"))
    _error_multi(ax_eth, runs, colors, multi, "heading error (deg)", "Heading error", "deg",
                lambda hist: _get(hist, "e_theta"))

    first_hist = next(iter(runs.values()))
    ax_yaw.plot(first_hist["t"], first_hist["path_yaw"], color=COLOR_MUTED, linewidth=2,
               linestyle="--", label="path yaw")
    for label, hist in runs.items():
        ax_yaw.plot(hist["t"], hist["yaw"], color=colors[label], linewidth=1.8,
                   solid_capstyle="round", label=(label or "ego yaw"))
    ax_yaw.set_ylabel("heading (deg)")
    ax_yaw.set_title("Vehicle heading vs. road heading")
    _legend(ax_yaw, ncol=min(len(runs) + 1, 4))

    _dynamics_panel(ax_r, runs, colors, multi, "yaw_rate", "yaw rate (deg/s)", "Yaw rate")
    _dynamics_panel(ax_racc, runs, colors, multi, "yaw_acc", "yaw accel (rad/s$^2$)", "Yaw acceleration")
    _dynamics_panel(ax_vy, runs, colors, multi, "v_y", "$v_y$ (m/s)", "Lateral velocity (body frame)")
    _dynamics_panel(ax_ay, runs, colors, multi, "a_y", "$a_y$ (m/s$^2$)", "Lateral acceleration (body frame)")
    ax_ay.set_xlabel("t (s)")

    _dynamics_panel(ax_steer, runs, colors, multi, "steer_deg", "steer (deg)", "Steering angle (front wheel)")
    ax_steer.set_xlabel("t (s)")

    return fig


def plot_longitudinal(data, target_speed_ms, title="Longitudinal tracking performance"):
    """fig 2: speed error, speed pair, longitudinal acceleration, longitudinal jerk, total jerk
    magnitude, control input u -- for one controller, or several overlaid.

    The last panel plots u = throttle - brake, a single signed series in [-1, 1] (throttle and
    brake are mutually exclusive in every hist this repo logs, so the subtraction reconstructs the
    actual command exactly) rather than the two separate [0, 1] series, since that's the pedal
    signal a controller actually computed before it got split into carla.VehicleControl's two
    fields.

    data: a single hist dict, or {label: hist} to compare controllers on one figure (e.g.
    longitudinal_mpc.py's --controller both). A lone hist keeps the original look (filled error
    band, corner RMSE badge, aqua-above/red-below u fill); multiple runs switch each panel to
    plain colored lines -- one color per run from COMPARE_COLORS -- with RMSE folded into the
    legend instead of a badge, since several badges stacked in one corner stop being readable.
    """
    runs = _runs(data)
    multi = len(runs) > 1
    fig, (ax_ev, ax_v, ax_a, ax_j, ax_jtot, ax_cmd) = _panels(title)
    colors = dict(zip(runs, COMPARE_COLORS))

    _error_multi(ax_ev, runs, colors, multi, "speed error (m/s)", "Speed error (reference - measured)",
                "m/s", lambda hist: np.asarray(speed_error_series(hist, target_speed_ms), dtype=float))

    first_hist = next(iter(runs.values()))
    v_des = _get(first_hist, "v_des")
    if v_des is None:
        ax_v.axhline(target_speed_ms, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    else:
        ax_v.plot(first_hist["t"], v_des, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    for label, hist in runs.items():
        ax_v.plot(hist["t"], hist["v_x"], color=colors[label], linewidth=1.8, solid_capstyle="round",
                  label=(label or "ego vel"))
    ax_v.set_ylabel("speed (m/s)")
    ax_v.set_title("Speed")
    _legend(ax_v, ncol=min(len(runs) + 1, 4))

    _dynamics_panel(ax_a, runs, colors, multi, "a_x", "$a_x$ (m/s$^2$)", "Longitudinal acceleration")
    _dynamics_panel(ax_j, runs, colors, multi, "jerk", "jerk (m/s$^3$)", "Longitudinal jerk (ride comfort)")
    _dynamics_panel(ax_jtot, runs, colors, multi, "jerk_total", "|jerk| (m/s$^3$)",
                    "Total jerk magnitude (long. + lat.)", fill_color=COLOR_PURPLE)
    ax_jtot.set_xlabel("t (s)")

    ax_cmd.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    if not multi:
        hist = first_hist
        t = np.asarray(hist["t"], dtype=float)
        u = np.asarray(hist["throttle"], dtype=float) - np.asarray(hist["brake"], dtype=float)
        ax_cmd.plot(t, u, color=COLOR_BLUE, linewidth=1.6, solid_capstyle="round")
        ax_cmd.fill_between(t, u, 0, where=(u >= 0), color=COLOR_AQUA, alpha=0.15, interpolate=True)
        ax_cmd.fill_between(t, u, 0, where=(u <= 0), color=COLOR_RED, alpha=0.15, interpolate=True)
    else:
        for label, hist in runs.items():
            t = np.asarray(hist["t"], dtype=float)
            u = np.asarray(hist["throttle"], dtype=float) - np.asarray(hist["brake"], dtype=float)
            ax_cmd.plot(t, u, color=colors[label], linewidth=1.4, label=label)
        _legend(ax_cmd, ncol=len(runs))
    ax_cmd.set_ylim(-1.05, 1.05)
    ax_cmd.set_ylabel("$u$")
    ax_cmd.set_title("Longitudinal control input $u$  (u > 0: throttle, u < 0: brake)")
    ax_cmd.set_xlabel("t (s)")

    return fig


def plot_trajectory(path_x, path_y, data, title="Desired path vs. ego trajectory"):
    """fig 3: the xy view, for one run or several overlaid. Its own figure because equal aspect
    fights a shared time-series grid.

    data: a single hist dict, or {label: hist} to compare controllers' driven lines against the
    same desired path (see plot_longitudinal). A lone hist keeps the original orange trajectory
    line; multiple runs get one color per run from COMPARE_COLORS instead.
    """
    runs = _runs(data)
    multi = len(runs) > 1
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    _style_axes(ax)
    colors = dict(zip(runs, COMPARE_COLORS))

    ax.plot(path_x, path_y, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired path")
    for label, hist in runs.items():
        color = colors[label] if multi else COLOR_ORANGE
        ax.plot(hist["x"], hist["y"], color=color, linewidth=2, solid_capstyle="round",
               label=(label or "ego trajectory"))
    ax.scatter([path_x[0]], [path_y[0]], color=COLOR_BLUE, zorder=5, label="start")
    ax.scatter([path_x[-1]], [path_y[-1]], color=COLOR_RED, marker="*", s=140, zorder=5, label="goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, color=COLOR_INK)
    ax.set_aspect("equal", adjustable="datalim")
    _legend(ax, ncol=min(len(runs) + 2, 4))
    return fig


def plot_results(path_x, path_y, data, target_speed_ms, out_dir,
                 show=True, summary=True, label="", name=None):
    """Save + show the three result figures and print the error summary.

    data: a single hist dict, or {label: hist} to compare controllers across all three figures
    at once (see plot_longitudinal/plot_lateral/plot_trajectory) -- e.g. stanley_mpc.py running
    more than one --controller. Each run gets its own error-summary block when there's more than
    one.

    Files are named <script>_<date>_<time>_<kind>.png; pass name to override the script part.
    Returns the list of saved paths (lateral, longitudinal, trajectory).
    """
    if summary:
        for run_label, hist in _runs(data).items():
            if run_label:
                print(f"\n### {run_label} ###")
            print_error_summary(hist, target_speed_ms)

    suffix = f" — {label}" if label else ""
    figures = [
        ("lateral", plot_lateral(data, f"Lateral tracking performance{suffix}")),
        ("longitudinal", plot_longitudinal(data, target_speed_ms,
                                           f"Longitudinal tracking performance{suffix}")),
        ("trajectory", plot_trajectory(path_x, path_y, data)),
    ]

    os.makedirs(out_dir, exist_ok=True)
    out_paths = [_save(fig, out_dir, run_name(kind, name)) for kind, fig in figures]
    for path in out_paths:
        print(f"Figure saved: {path}")

    if show:
        plt.show()  # blocks until every window is closed
    for _, fig in figures:
        plt.close(fig)
    return out_paths


def plot_longitudinal_result(data, target_speed_ms, out_dir, show=True, summary=True, label="", name=None):
    """Like plot_results(), but only the longitudinal figure.

    For stacks with no lateral control at all (e.g. longitudinal_PID.py, steer pinned at 0) --
    there's no path or steering to put in the other two figures.

    data: a single hist dict, or {label: hist} to overlay several controllers on one figure (see
    plot_longitudinal) -- e.g. longitudinal_mpc.py's --controller both. Each run gets its own
    error-summary block when there's more than one.
    """
    if summary:
        for run_label, hist in _runs(data).items():
            if run_label:
                print(f"\n### {run_label} ###")
            print_error_summary(hist, target_speed_ms)

    suffix = f" — {label}" if label else ""
    fig = plot_longitudinal(data, target_speed_ms, f"Longitudinal tracking performance{suffix}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = _save(fig, out_dir, run_name("longitudinal", name))
    print(f"Figure saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 8. LUT results
# ---------------------------------------------------------------------------

def plot_lut_validation(hist_ff, hist_pid, mode, args, out_dir, hist_pidonly=None, show=True, name=None):
    """Overlay LUT feedforward-only vs. feedforward+PID vs. (optionally) plain-PID-only tracking,
    for validate_lut.py.

    mode == "accel": reference/response is commanded longitudinal acceleration (a_cmd vs a_x).
    mode == "speed":  reference/response is vehicle speed (v_des vs v_x) -- a_cmd there is the
                      analytic derivative of v_des fed to the LUT as its feedforward target, not
                      scored directly.

    hist_pidonly is optional (validate_lut.py's speed mode doesn't run that trial) so old callers
    and 2-trial modes keep working without passing it.
    """
    if mode == "accel":
        ref_key, resp_key, unit, ylabel = "a_cmd", "a_x", "m/s$^2$", "acceleration (m/s$^2$)"
    else:
        ref_key, resp_key, unit, ylabel = "v_des", "v_x", "m/s", "speed (m/s)"

    series = [("feedforward only", hist_ff, COLOR_AQUA), ("feedforward + PID", hist_pid, COLOR_BLUE)]
    if hist_pidonly is not None:
        series.append(("PID only", hist_pidonly, COLOR_ORANGE))
    ts = {label: np.asarray(hist["t"], dtype=float) for label, hist, _ in series}

    title = "LUT feedforward vs. feedforward+PID" + (" vs. PID only" if hist_pidonly is not None else "")
    fig, (ax_main, ax_err, ax_u) = _panels(
        f"{title} -- {mode} tracking, {args.profile} profile",
        n_rows=3, n_cols=1, figsize=(14, 10))

    ax_main.plot(ts["feedforward only"], hist_ff[ref_key], color=COLOR_MUTED, linewidth=2.2,
                linestyle="--", label="reference")
    for label, hist, color in series:
        ax_main.plot(ts[label], hist[resp_key], color=color, linewidth=1.6, label=label)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title("Tracking")
    _legend(ax_main, ncol=len(series) + 1)

    ax_err.axhline(0.0, color=COLOR_AXIS, linewidth=1, linestyle="--")
    for label, hist, color in series:
        err = np.asarray(hist[ref_key], dtype=float) - np.asarray(hist[resp_key], dtype=float)
        ax_err.plot(ts[label], err, color=color, linewidth=1.4, label=label)
    ax_err.set_ylabel(f"error ({unit})")
    ax_err.set_title("Tracking error (reference - measured)")
    _legend(ax_err, ncol=len(series))

    ax_u.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    for label, hist, color in series:
        ax_u.plot(ts[label], hist["u"], color=color, linewidth=1.3, label=label)
    ax_u.set_ylim(-1.15, 1.15)
    ax_u.set_ylabel("pedal $u$")
    ax_u.set_xlabel("t (s)")
    ax_u.set_title("Control input")
    _legend(ax_u, ncol=len(series))

    os.makedirs(out_dir, exist_ok=True)
    out_path = _save(fig, out_dir, run_name(mode, name))
    print(f"Figure saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# blue (brake) <-> neutral <-> red (throttle), built from the house palette so it matches
# everything else rather than introducing its own hex codes
_UA_CMAP = LinearSegmentedColormap.from_list("brake_throttle", [COLOR_BLUE, "#f0efec", COLOR_RED])


def plot_lut_raw_distribution(raw, out_dir, trusted_ranges=None, show=True, name=None):
    """Per-gear (v_x, a_x) scatter of collect_lut_data.py's raw sweep CSV, colored by control
    input u -- for eyeballing coverage and density right after a sweep.

    raw: structured array from np.genfromtxt(..., names=True) with gear/u/v_x/a_x columns.
    trusted_ranges: optional {gear: (v_lo, v_hi) or None}, from build_lut.py's
    trusted_speed_range() -- when given, shades the speed window each gear's fit actually trusted
    (None for a gear means it had no trustworthy window at all).

    gear 0 (CARLA's mid-shift/clutch-disengaged sentinel, not a real gear) is left out -- the
    runtime controller never queries it (see LookupController.feedforward()), and its samples
    are sparse and noisy enough that trusted_speed_range() rejects them anyway.
    """
    trusted_ranges = trusted_ranges or {}
    gears = sorted(g for g in {int(x) for x in raw["gear"]} if g != 0)
    ncols = min(3, len(gears))
    nrows = math.ceil(len(gears) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False,
                             constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    subtitle = ("\nshaded band = speed range build_lut.py trusted" if trusted_ranges else "")
    fig.suptitle(f"Raw sweep data -- per-gear (v_x, a_x), colored by control input u{subtitle}",
                fontsize=12, color=COLOR_INK)

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    flat_axes = axes.ravel()
    mappable = None
    for ax, gear in zip(flat_axes, gears):
        _style_axes(ax)
        mask = raw["gear"] == gear
        v, a, u = raw["v_x"][mask], raw["a_x"][mask], raw["u"][mask]
        mappable = ax.scatter(v, a, c=u, cmap=_UA_CMAP, norm=norm, s=6, alpha=0.35, linewidths=0)
        v_range = trusted_ranges.get(gear)
        if v_range is not None:
            ax.axvspan(v_range[0], v_range[1], color=COLOR_AQUA, alpha=0.10, zorder=0)
        title = f"gear {gear}"
        if trusted_ranges and v_range is None:
            title += "  [excluded]"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("v_x (m/s)")
        ax.set_ylabel("a_x (m/s$^2$)")
    for ax in flat_axes[len(gears):]:
        ax.axis("off")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=fig.get_axes(), shrink=0.6, pad=0.02,
                            label="u  (brake <- 0 -> throttle)")
        cbar.ax.yaxis.label.set_color(COLOR_MUTED)
        cbar.ax.tick_params(colors=COLOR_MUTED)

    os.makedirs(out_dir, exist_ok=True)
    out_path = _save(fig, out_dir, run_name("lut_raw_distribution", name))
    print(f"Figure saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _style_3d_axes(ax):
    ax.set_facecolor(COLOR_BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(COLOR_BG)
        pane.set_edgecolor(COLOR_GRID)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = COLOR_GRID
        axis.label.set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)


def plot_lut_surfaces(gear_tables, out_dir, raw=None, show=True, elev=25.0, azim=-60.0, name=None):
    """One (v_x, a_x) -> u 3D surface per gear, small multiples on a shared diverging color scale
    (u: brake -1 -> throttle +1) so gears stay visually comparable -- for build_lut.py to sanity
    check a fit right after producing it.

    gear_tables: {gear: (v_grid, a_grid, u_table)}, u_table shaped (len(v_grid), len(a_grid)) --
    exactly what build_lut.py's build_gear_table() returns per gear.
    raw: optional structured array (gear/u/v_x/a_x columns) overlaid as raw sweep samples -- pass
    everything build_lut.py read in (not just what a gear's trust cut kept) to see the surface
    against what got excluded too, not only what it was fit from.
    """
    gears = sorted(gear_tables)
    ncols = min(3, len(gears))
    nrows = math.ceil(len(gears) / ncols)
    fig = plt.figure(figsize=(5.2 * ncols, 4.6 * nrows), facecolor=COLOR_BG)
    fig.suptitle("Longitudinal control-input lookup table  (u: brake -1 -> throttle +1)",
                color=COLOR_INK, fontsize=13)

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    mappable = None
    for i, gear in enumerate(gears):
        v_grid, a_grid, u_table = gear_tables[gear]
        vv, aa = np.meshgrid(v_grid, a_grid, indexing="ij")

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        mappable = ax.plot_surface(vv, aa, u_table, cmap=_UA_CMAP, norm=norm,
                                   linewidth=0, antialiased=True, alpha=0.92)

        if raw is not None:
            mask = raw["gear"] == gear
            ax.scatter(raw["v_x"][mask], raw["a_x"][mask], raw["u"][mask],
                      s=4, color=COLOR_MUTED, alpha=0.12, depthshade=False)

        ax.set_title(f"Gear {gear}", color=COLOR_INK, fontsize=11)
        ax.set_xlabel("v_x (m/s)")
        ax.set_ylabel("a_x (m/s$^2$)")
        ax.set_zlabel("u")
        ax.set_zlim(-1, 1)
        ax.view_init(elev=elev, azim=azim)
        _style_3d_axes(ax)

    cbar = fig.colorbar(mappable, ax=fig.get_axes(), shrink=0.6, pad=0.02,
                        label="u  (brake <- 0 -> throttle)")
    cbar.ax.yaxis.label.set_color(COLOR_MUTED)
    cbar.ax.tick_params(colors=COLOR_MUTED)

    os.makedirs(out_dir, exist_ok=True)
    out_path = _save(fig, out_dir, run_name("lut_surfaces", name))
    print(f"Figure saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 9. lateral parameter identification figures
# ---------------------------------------------------------------------------

def plot_cornering_stiffness_fit(log, window, fit, out_dir=None, show=True, name=None):
    """For estimate_cornering_stiffness.py: run overview (steady window shaded) plus the two
    per-axle Fy-vs-alpha fits.

    The scatter in the two fit panels is usually a tight cloud, not a spread-out line -- one run
    holds one speed and one steer angle, so it is a single operating point measured many times,
    not a sweep. A real Cf/Cr campaign runs the script at several speeds/steer angles and pools
    the (alpha, Fy) pairs before fitting; this figure is per-run.

    Markers are drawn last (highest zorder) with a background-colored edge so they read as
    distinct dots even where they sit almost exactly on the fit line -- a plain small marker at
    low alpha gets visually absorbed by the line and the shaded bracket band under it.
    """
    import matplotlib.pyplot as plt

    start, end = window
    t = np.asarray(log["t"], dtype=float)

    fig, (ax_t, ax_f, ax_r) = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    for ax in (ax_t, ax_f, ax_r):
        _style_axes(ax)

    ax_t.plot(t, log["v_x"], color=COLOR_BLUE, linewidth=LINEWIDTH, label="v_x (m/s)")
    ax_t.plot(t, np.degrees(log["r"]), color=COLOR_ORANGE, linewidth=LINEWIDTH,
             label="psi_dot (deg/s)")
    ax_t.plot(t, log["a_y_imu"], color=COLOR_AQUA, linewidth=LINEWIDTH, label="a_y IMU (m/s^2)")
    ax_t.axvspan(t[start], t[end - 1], color=COLOR_BLUE, alpha=0.12, label="steady window")
    ax_t.set_xlabel("t (s)")
    ax_t.set_title("Run overview")
    _legend(ax_t)

    for ax, alpha, Fy, C, C_lo, C_hi, color, axle in (
            (ax_f, fit["alpha_f"], fit["Fyf"], fit["Cf"], fit["Cf_forward"], fit["Cf_reverse"],
             COLOR_BLUE, "front"),
            (ax_r, fit["alpha_r"], fit["Fyr"], fit["Cr"], fit["Cr_forward"], fit["Cr_reverse"],
             COLOR_ORANGE, "rear")):
        alpha_deg = np.degrees(np.asarray(alpha, dtype=float))
        Fy = np.asarray(Fy, dtype=float)
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)
        ax.axvline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)

        lo, hi = min(0.0, alpha_deg.min()), max(0.0, alpha_deg.max())
        xs = np.linspace(lo, hi, 20)
        ax.fill_between(xs, C_lo * np.radians(xs), C_hi * np.radians(xs), color=color, alpha=0.12,
                        zorder=1, label=f"bracket [{C_lo:,.0f}, {C_hi:,.0f}]")
        ax.plot(xs, C * np.radians(xs), color=color, linewidth=LINEWIDTH, linestyle="--",
               zorder=2, label=f"C={C:,.0f} N/rad")
        ax.scatter(alpha_deg, Fy, s=MARKERSIZE, facecolor=color, edgecolor=COLOR_BG,
                  linewidth=0.6, alpha=0.85, zorder=3, label="measured")

        ax.set_xlabel(f"alpha_{axle[0]} (deg)")
        ax.set_ylabel(f"Fy{axle[0]} (N)")
        ax.set_title(f"{axle.capitalize()} axle: Fy = C * alpha")
        _legend(ax)

    out_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = _save(fig, out_dir, run_name("cornering_stiffness", name))
        print(f"Figure saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_cornering_stiffness_sweep(trials, pooled, out_dir=None, show=True, name=None):
    """For estimate_cornering_stiffness.py's --sweep: the two per-axle Fy-vs-alpha fits pooled
    across every settled trial, plus a third panel scoring the sweep itself -- does each trial's
    own Cf/Cr agree with the pooled number, or does it drift with a_y (the standard tell that a
    trial left the tire's linear region -- see ISO 4138's steady-state circular test, which is
    what this sweep effectively runs).

    trials: list of per-trial dicts with "target_speed", "a_y_est" and "window" set by
    run_sweep(), plus a "fit" dict (from fit_cornering_stiffness) for every trial pool_trials()
    was able to fit. Trials with window is None or no "fit" are skipped here.
    """
    import matplotlib.pyplot as plt

    done = [tr for tr in trials if tr.get("fit") is not None]

    fig, (ax_f, ax_r, ax_c) = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    for ax in (ax_f, ax_r, ax_c):
        _style_axes(ax)

    a_y_vals = [tr["a_y_est"] for tr in done]
    cmap = plt.get_cmap("viridis")
    norm = (plt.Normalize(min(a_y_vals), max(a_y_vals)) if len(set(a_y_vals)) > 1 else None)

    for ax, key_alpha, key_Fy, C, C_lo, C_hi, axle in (
            (ax_f, "alpha_f", "Fyf", pooled["Cf"], pooled["Cf_forward"], pooled["Cf_reverse"],
             "front"),
            (ax_r, "alpha_r", "Fyr", pooled["Cr"], pooled["Cr_forward"], pooled["Cr_reverse"],
             "rear")):
        color = COLOR_BLUE if axle == "front" else COLOR_ORANGE
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)
        ax.axvline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)

        all_alpha_deg = np.degrees(pooled[key_alpha])
        lo, hi = min(0.0, all_alpha_deg.min()), max(0.0, all_alpha_deg.max())
        xs = np.linspace(lo, hi, 20)
        ax.fill_between(xs, C_lo * np.radians(xs), C_hi * np.radians(xs), color=color, alpha=0.12,
                        zorder=1, label=f"bracket [{C_lo:,.0f}, {C_hi:,.0f}]")
        ax.plot(xs, C * np.radians(xs), color=color, linewidth=LINEWIDTH, linestyle="--",
               zorder=2, label=f"pooled C={C:,.0f} N/rad")

        for tr in done:
            alpha_deg = np.degrees(tr["fit"][key_alpha])
            Fy = tr["fit"][key_Fy]
            c = cmap(norm(tr["a_y_est"])) if norm else color
            ax.scatter(alpha_deg, Fy, s=MARKERSIZE, facecolor=c, edgecolor=COLOR_BG,
                      linewidth=0.6, alpha=0.9, zorder=3)

        ax.set_xlabel(f"alpha_{axle[0]} (deg)")
        ax.set_ylabel(f"Fy{axle[0]} (N)")
        ax.set_title(f"{axle.capitalize()} axle: {len(done)} trials pooled (color = a_y)")
        _legend(ax)

    ays = [tr["a_y_est"] for tr in done]
    Cfs = [tr["fit"]["Cf"] for tr in done]
    Crs = [tr["fit"]["Cr"] for tr in done]
    ax_c.axhline(pooled["Cf"], color=COLOR_BLUE, linewidth=LINEWIDTH_THIN, linestyle="--",
                label=f"pooled Cf={pooled['Cf']:,.0f}")
    ax_c.axhline(pooled["Cr"], color=COLOR_ORANGE, linewidth=LINEWIDTH_THIN, linestyle="--",
                label=f"pooled Cr={pooled['Cr']:,.0f}")
    ax_c.scatter(ays, Cfs, s=MARKERSIZE, facecolor=COLOR_BLUE, edgecolor=COLOR_BG, linewidth=0.6,
                zorder=3, label="Cf per trial")
    ax_c.scatter(ays, Crs, s=MARKERSIZE, facecolor=COLOR_ORANGE, edgecolor=COLOR_BG,
                linewidth=0.6, zorder=3, label="Cr per trial")
    ax_c.set_xlabel("a_y (m/s^2, kinematic estimate)")
    ax_c.set_ylabel("C (N/rad)")
    ax_c.set_title("Constancy check: C should not drift with a_y")
    _legend(ax_c)

    out_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = _save(fig, out_dir, run_name("cornering_stiffness_sweep", name))
        print(f"Figure saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path
