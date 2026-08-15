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
FONTSIZE_TITLE = 22      # figure suptitle
FONTSIZE_SUBTITLE = 18   # per-axes title
FONTSIZE_LABEL = 16      # axis labels
FONTSIZE_TICK = 14        # tick labels
FONTSIZE_LEGEND = 16      # legend text


def _style_axes(ax):
    ax.set_facecolor(COLOR_BG)
    ax.grid(True, color=COLOR_GRID, linewidth=LINEWIDTH_THIN * 0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=FONTSIZE_TICK)
    # title styling is NOT set here: Axes.set_title() unconditionally resets fontsize/fontweight/
    # color to matplotlib's rcParams defaults on every call (it applies its own `default` dict
    # before kwargs), so anything set on ax.title before the real set_title() call just gets
    # wiped out -- see _title() below, which every title in this file should go through instead.
    ax.xaxis.label.set_color(COLOR_MUTED)
    ax.yaxis.label.set_color(COLOR_MUTED)
    ax.xaxis.label.set_fontsize(FONTSIZE_LABEL)
    ax.yaxis.label.set_fontsize(FONTSIZE_LABEL)


def _title(ax, text, fontsize=FONTSIZE_SUBTITLE):
    """Every panel title in this file should be set through here, not ax.set_title() directly --
    see the note in _style_axes() for why setting title style any other way doesn't stick."""
    ax.set_title(text, fontsize=fontsize, fontweight="bold", color=COLOR_INK)


def _legend(ax, **kwargs):
    ax.legend(frameon=False, labelcolor=COLOR_INK, fontsize=FONTSIZE_LEGEND, **kwargs)


def _bottom_legend(fig, handles, title=None, max_ncol=6):
    """Shared fig-level legend below every axes, sized to clear the bottom row's own xlabel
    instead of plot_comparison()'s fixed bbox_to_anchor=(0.5, -0.02) -- that offset only clears
    the xlabel when the figure has enough rows above it that one more legend row is a small
    fraction of the total height; a single-row figure (this file's newer per-speed/per-steer
    reports) needs noticeably more room, and more again once enough handles wrap the legend onto a
    second row. ncol is capped at max_ncol so a long handle list wraps instead of running off the
    figure edge or shrinking to unreadable size.
    """
    ncol = max(1, min(len(handles), max_ncol))
    rows = math.ceil(len(handles) / ncol)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
              labelcolor=COLOR_INK, fontsize=FONTSIZE_LEGEND,
              bbox_to_anchor=(0.5, -0.06 - 0.09 * rows), title=title)


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
        _title(ax, f"last_idx = {last_idx}/{n_path - 1}")
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


# B2D "Comfortness" / Driving Smoothness hard limits (2.4 Comfortness table) -- (lo, hi) per
# signal, in the same units print_error_summary already logs them in EXCEPT yaw_rate: hist logs
# that one in degrees (matches the rest of this file's dashboards), but B2D's own 0.95 rad/s limit
# is in radians, so that conversion happens at the comparison site in b2d_comfort_penalty() below,
# not here.
B2D_COMFORT_LIMITS = {
    "a_x":        (-4.05, 2.40),   # m/s^2 -- asymmetric: braking vs accelerating limits differ
    "a_y":        (-4.90, 4.90),   # m/s^2
    "yaw_rate":   (-0.95, 0.95),   # rad/s
    "yaw_acc":    (-1.93, 1.93),   # rad/s^2
    "jerk":       (-4.13, 4.13),   # m/s^3
    "jerk_total": (0.0, 8.37),     # m/s^3 -- a norm (||j||), always >= 0, so the "lo" half of the
}                                   # excess formula below is naturally always 0 for this one

def _band_penalty(values, lo, hi):
    """Mean, band-width-normalized excess-outside-[lo,hi]: 0 if the signal never left the band,
    1 if it sat a full band-width past the limit for the entire run. Continuous rather than B2D's
    own 20-frame binary Smooth/not-Smooth segment scoring, which is fine as the paper's own
    reported number but is a flat, uninformative signal to tune against -- two configurations that
    are both "0% smooth" can still be very differently bad, and a pass/fail score can't tell them
    apart. None if `values` is empty (this hist never recorded the signal)."""
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    width = hi - lo
    excess = [max(0.0, v - hi) + max(0.0, lo - v) for v in values]
    return (sum(excess) / len(excess)) / width


def b2d_comfort_penalty(hist):
    """B2D Comfortness-style penalty terms for one run: {name: P}, P=0 perfect, growing
    unboundedly worse (no cap) the further/longer a signal sits outside its limit. "total" sums
    whatever terms this hist actually recorded (None for any that weren't, e.g. a steer=0 run has
    no a_y/yaw_rate/yaw_acc at all) -- route completion is deliberately not a term here since
    incomplete runs are eyeballed and thrown away rather than scored.

    Comfort/smoothness terms only (a_x/a_y/yaw_rate/yaw_acc/jerk/jerk_total) -- lateral_error and
    lap_time score tracking/route-progress, a different thing, and were dropped from this scoring
    entirely rather than just excluded from "total"."""
    terms = {}
    terms["a_x"] = _band_penalty(hist.get("a_x", []), *B2D_COMFORT_LIMITS["a_x"])
    terms["a_y"] = _band_penalty(hist.get("a_y", []), *B2D_COMFORT_LIMITS["a_y"])
    yaw_rate_rad = [math.radians(v) for v in hist.get("yaw_rate", [])]
    terms["yaw_rate"] = _band_penalty(yaw_rate_rad, *B2D_COMFORT_LIMITS["yaw_rate"])
    terms["yaw_acc"] = _band_penalty(hist.get("yaw_acc", []), *B2D_COMFORT_LIMITS["yaw_acc"])
    terms["jerk"] = _band_penalty(hist.get("jerk", []), *B2D_COMFORT_LIMITS["jerk"])
    terms["jerk_total"] = _band_penalty(hist.get("jerk_total", []), *B2D_COMFORT_LIMITS["jerk_total"])

    available = [v for v in terms.values() if v is not None]
    terms["total"] = sum(available) if available else None
    return terms


def print_error_summary(hist, target_speed_ms):
    """RMSE / max / mean of each tracked error, plus mean/peak magnitude of the raw longitudinal
    and lateral dynamics signals, over the whole run.

    The dynamics rows are printed as sections that appear only when this stack actually recorded
    them -- a steer=0 run (e.g. longitudinal_PID.py) has no yaw_rate/yaw_acc/a_y, and the section
    is skipped rather than printed empty.
    """
    if not hist["t"]:
        return

    # v_y_hat - v_y (a genuine tracking error, unlike a_y/yaw_rate/... below which are raw
    # dynamics signals with no target) only exists for a run that logged a Kalman-filter v_y
    # estimate alongside ground truth (mpc_mpc_KF.py's "mpc-kf" controller) -- [] for every other
    # hist, same "row only appears if recorded" gating the dynamics sections already use.
    v_y_hat = hist.get("v_y_hat", [])
    v_y_est_err = ([hat - true for hat, true in zip(v_y_hat, hist["v_y"])]
                  if len(v_y_hat) and len(v_y_hat) == len(hist.get("v_y", [])) else [])

    print(f"\n=== error summary: {len(hist['t'])} steps, {hist['t'][-1]:.1f} s ===")
    for name, unit, series in (("cross-track", "m", hist.get("e_y", [])),
                               ("heading    ", "deg", hist.get("e_theta", [])),
                               ("speed      ", "m/s", speed_error_series(hist, target_speed_ms)),
                               ("v_y estimate", "m/s", v_y_est_err)):
        stats = error_stats(series)
        if stats is None:
            continue
        rmse, peak, bias = stats
        print(f"  {name}  RMSE={rmse:7.3f} {unit:<3}  max|e|={peak:7.3f} {unit:<3} ")

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

    penalty = b2d_comfort_penalty(hist)
    if penalty["total"] is not None:
        print("  -- B2D comfort penalty (0 = perfect, unbounded above) --")
        for key, label in (("a_x", "long accel  "), ("a_y", "lat accel   "),
                           ("yaw_rate", "yaw rate    "), ("yaw_acc", "yaw accel   "),
                           ("jerk", "long jerk   "), ("jerk_total", "|jerk| total")):
            if penalty[key] is not None:
                print(f"    {label}  P={penalty[key]:.4f}")
        print(f"    {'TOTAL':<12}  P={penalty['total']:.4f}")


# ---------------------------------------------------------------------------
# 7. plotting internals
# ---------------------------------------------------------------------------

def _panels(title, n_rows=3, n_cols=2, figsize=(15, 10)):
    """A styled grid sharing the time axis, flattened in row-major order. title="" (or None) skips
    the suptitle entirely -- for figures (like plot_comparison()'s) where each panel's own title
    already carries the identifying info and a figure-level title would just be redundant."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    if title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE, color=COLOR_INK, fontweight="bold")
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


def _error_multi(ax, runs, colors, multi, ylabel, panel_title, unit, series_fn, legend=True):
    """One error-vs-time panel, for a single run or several overlaid.

    series_fn(hist) -> the error array for that run (or None if this stack never recorded it).
    Single run keeps the filled-band/corner-badge look; multiple runs switch to plain colored
    lines with each RMSE folded into the legend, since stacked badges stop being readable.

    legend=False skips this panel's own legend -- for callers (plot_comparison()) that build one
    shared legend for the whole figure instead of repeating it on every panel.
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
    elif multi and legend:
        _legend(ax)
    ax.set_ylabel(ylabel)
    _title(ax, panel_title)


def _dynamics_panel(ax, runs, colors, multi, key, ylabel, panel_title, fill_color=None, ref_key=None,
                    legend=True):
    """One plain time-series panel (acceleration, jerk, yaw rate, ...), for a single run or
    several overlaid. Single run gets an optional fill; multiple runs get one colored line each
    plus a legend. "not recorded" if no run in `runs` logged `key` at all.

    ref_key: an optional companion series (e.g. "a_cmd") drawn as a dashed line in the same color as
    its run, right on top of the measured one -- for stacks that never computed it (stanley_PID.py
    has no a_cmd concept at all), _get() just returns None and the dashed line is silently skipped,
    so the same panel code works whether or not a given controller has a reference to show.

    legend=False skips this panel's own legend -- for callers (plot_comparison()) that build one
    shared legend for the whole figure instead of repeating it on every panel."""
    found = False
    has_ref = False
    for label, hist in runs.items():
        series = _get(hist, key)
        if series is None:
            continue
        found = True
        t = np.asarray(hist["t"], dtype=float)
        color = colors[label]
        ref = _get(hist, ref_key) if ref_key is not None else None
        series_label = label if multi else None
        if ref is not None:
            has_ref = True
            series_label = label if label else "actual"
        # When there's a ref line to overlay, swap to LINEWIDTH_THIN (series) / LINEWIDTH (ref,
        # thicker) instead of the plain multi/non-multi widths below -- a thin solid + thick dashed
        # pair reads clearly even when the two nearly overlap (v_y_hat tracking v_y closely, a_cmd
        # tracking a_x closely, ...), which same-width same-color-plus-alpha didn't: the dashed line
        # all but disappeared under the solid one. ref is always COLOR_RED regardless of the
        # series' own color -- every current caller draws at most one (series, ref) pair per panel
        # (plot_lateral()'s ref_key uses are all multi=False; plot_kf_series() always hands this a
        # single-entry runs dict), so a fixed contrasting ref color never collides with another
        # run's own ref, and reads as "the estimate/reference" at a glance instead of just a paler
        # copy of whatever color the series happened to get. No ref -> untouched (1.3/1.5 as before).
        series_lw = LINEWIDTH_THIN if ref is not None else (1.3 if multi else 1.5)
        ax.plot(t, series, color=color, linewidth=series_lw,
               solid_capstyle="round", label=series_label)
        if not multi and fill_color:
            ax.fill_between(t, series, 0, color=fill_color, alpha=0.15)
        if ref is not None:
            ref_label = f"{label} ref" if label else "reference"
            ax.plot(t, ref, color=COLOR_RED, linewidth=LINEWIDTH, linestyle="--", alpha=0.9,
                   label=ref_label)
    if not found:
        _not_recorded(ax, key)
    else:
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=1)
        if (multi or has_ref) and legend:
            _legend(ax)
    ax.set_ylabel(ylabel)
    _title(ax, panel_title)


def _b2d_limit_lines(ax, lo, hi):
    """Red dashed line(s) marking a B2D_COMFORT_LIMITS band on a dynamics panel -- both bounds for
    a two-sided range, just the top one when lo==0 (a magnitude/norm signal like |jerk|, which
    never goes negative, so a line at 0 would just sit on the axis). Drawn at zorder=2.5, above the
    data lines (zorder~2 by default) and the axhline(0) reference (zorder~1), so the limit itself
    always reads clearly instead of blending into whatever data line happens to sit on top of it."""
    ax.axhline(hi, color=COLOR_RED, linewidth=1.8, linestyle=(0, (5, 3)), alpha=0.95, zorder=2.5)
    if lo != 0:
        ax.axhline(lo, color=COLOR_RED, linewidth=1.8, linestyle=(0, (5, 3)), alpha=0.95, zorder=2.5)


def _save(fig, out_dir, stem):
    out_path = os.path.join(out_dir, f"{stem}.png")
    # bbox_inches="tight": recrops to whatever the figure actually drew, so a fig-level legend
    # placed just outside the constrained-layout axes area (plot_comparison()'s shared bottom
    # legend) doesn't get clipped off the saved PNG.
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return out_path


# ---------------------------------------------------------------------------
# 8. result figures
# ---------------------------------------------------------------------------

def plot_lateral(hist, title="Lateral tracking performance"):
    """fig 1: cross-track error, heading error, yaw pair, yaw rate, yaw acceleration, lateral
    velocity, lateral acceleration, steer -- for one controller's single run.

    hist: one run's hist dict. For comparing several controllers' runs against each other, see
    plot_comparison() instead -- it reuses the same _error_multi/_dynamics_panel helpers this
    function calls with multi=True, in its own dedicated figures rather than overlaying them here.
    """
    runs, colors = {"": hist}, {"": COLOR_BLUE}
    fig, (ax_ey, ax_eth, ax_yaw, ax_r, ax_racc, ax_vy, ax_ay, ax_steer) = _panels(
        title, n_rows=4, figsize=(15, 13))

    _error_multi(ax_ey, runs, colors, False, "lateral error (m)", "Lateral error", "m",
                lambda h: _get(h, "e_y"))
    _error_multi(ax_eth, runs, colors, False, "heading error (deg)", "Heading error", "deg",
                lambda h: _get(h, "e_theta"))

    ax_yaw.plot(hist["t"], hist["path_yaw"], color=COLOR_MUTED, linewidth=2,
               linestyle="--", label="path yaw")
    ax_yaw.plot(hist["t"], hist["yaw"], color=COLOR_BLUE, linewidth=1.8,
               solid_capstyle="round", label="ego yaw")
    ax_yaw.set_ylabel("heading (deg)")
    _title(ax_yaw, "Vehicle heading vs. road heading")
    _legend(ax_yaw)

    _dynamics_panel(ax_r, runs, colors, False, "yaw_rate", "yaw rate (deg/s)", "Yaw rate")
    _b2d_limit_lines(ax_r, *(math.degrees(v) for v in B2D_COMFORT_LIMITS["yaw_rate"]))
    _dynamics_panel(ax_racc, runs, colors, False, "yaw_acc", "yaw accel (rad/s$^2$)", "Yaw acceleration")
    _b2d_limit_lines(ax_racc, *B2D_COMFORT_LIMITS["yaw_acc"])
    # ref_key="v_y_hat": when a run logged a Kalman-filter v_y estimate alongside ground truth
    # (mpc_mpc_KF.py's "mpc-kf" controller), it's overlaid as a dashed line in the same color --
    # see _dynamics_panel's ref_key doc. Runs that never log it (every other stack, plus this same
    # stack's own ground-truth "mpc" baseline run) just get _get()==None and the overlay is skipped,
    # so this is a no-op for every plot_lateral() caller that existed before mpc_mpc_KF.py.
    _dynamics_panel(ax_vy, runs, colors, False, "v_y", "$v_y$ (m/s)", "Lateral velocity (body frame)",
                    ref_key="v_y_hat")
    _dynamics_panel(ax_ay, runs, colors, False, "a_y", "$a_y$ (m/s$^2$)", "Lateral acceleration (body frame)")
    _b2d_limit_lines(ax_ay, *B2D_COMFORT_LIMITS["a_y"])
    ax_ay.set_xlabel("t (s)")

    _dynamics_panel(ax_steer, runs, colors, False, "steer_deg", "steer (deg)", "Steering angle (front wheel)")
    ax_steer.set_xlabel("t (s)")

    return fig


def plot_kf_series(runs, key, ref_key, ylabel, title):
    """One-panel time-series overlay across several runs: `key` (solid) vs. `ref_key` (dashed, if
    logged), one color per run via COMPARE_COLORS. This is the shared machinery behind plot_kf_vy()
    (v_y_hat vs. ground-truth v_y) and kalman_filter.py's clean-vs-noisy sensor channel plots (dpsi,
    a_y) alike -- reuses _dynamics_panel's own multi-run/ref_key mechanism (same one plot_lateral()'s
    single-run panels use) rather than a bespoke routine, so every one of these figures stays on the
    same LINEWIDTH/FONTSIZE/COLOR_* knobs as everything else in this file.

    runs: {label: hist}, e.g. {"5 m/s": hist_5, "10 m/s": hist_10, "15 m/s": hist_15} -- same shape
    plot_comparison() takes. A run missing `ref_key` just draws `key` alone (ref line skipped, per
    _dynamics_panel's own "not recorded" handling).
    """
    colors = {label: COMPARE_COLORS[i % len(COMPARE_COLORS)] for i, label in enumerate(runs)}
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(title, fontsize=FONTSIZE_TITLE, color=COLOR_INK, fontweight="bold")
    _style_axes(ax)
    _dynamics_panel(ax, runs, colors, True, key, ylabel, "", ref_key=ref_key)
    ax.set_xlabel("t (s)")
    return fig


def plot_kf_vy(runs, title="v_y: Kalman-filter estimate vs. ground truth"):
    """v_y_hat (dashed) vs. ground-truth v_y (solid) -- see plot_kf_series(), which this wraps."""
    return plot_kf_series(runs, "v_y", "v_y_hat", "$v_y$ (m/s)", title)


def plot_kf_run(hist, title=""):
    """kalman_filter.py's whole per-run report as ONE figure, 3 stacked panels sharing a time axis --
    v_y estimate vs. ground truth, dpsi clean vs. noisy sensor, a_y clean vs. noisy sensor -- instead
    of 3 separate plot_kf_series() figures/windows for the same run. hist needs "v_y_hat" (from
    replay()) and "dpsi_noisy"/"ay_noisy" (the noisy measurements replay() actually fed the filter,
    see kalman_filter.py's main()) alongside the usual "v_y"/"yaw_rate"/"a_y" ground truth.

    Single-run (not {label: hist}) on purpose, unlike plot_kf_series/plot_kf_vy -- this is one run's
    full picture, not several runs' v_y overlaid, so it reuses _dynamics_panel with multi=False (same
    convention plot_lateral()'s own single-run panels use) rather than the multi-run color cycle.
    """
    runs, colors = {"": hist}, {"": COLOR_BLUE}
    fig, (ax_vy, ax_dpsi, ax_ay) = plt.subplots(3, 1, figsize=(11, 12), sharex=True,
                                                constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    if title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE, color=COLOR_INK, fontweight="bold")
    for ax in (ax_vy, ax_dpsi, ax_ay):
        _style_axes(ax)

    _dynamics_panel(ax_vy, runs, colors, False, "v_y", "$v_y$ (m/s)",
                    "v_y estimate vs. ground truth", ref_key="v_y_hat")
    _dynamics_panel(ax_dpsi, runs, colors, False, "yaw_rate", "dpsi (deg/s)",
                    "dpsi: clean vs. noisy sensor", ref_key="dpsi_noisy")
    _dynamics_panel(ax_ay, runs, colors, False, "a_y", "$a_y$ (m/s$^2$)",
                    "a_y: clean vs. noisy sensor", ref_key="ay_noisy")
    ax_ay.set_xlabel("t (s)")
    return fig


def plot_trajectory_fit(fwd, lat, path, vx_spline, s_max_wp, s_mid, v_seg, vx_preview, kappa_preview,
                        dt, speed, title=""):
    r"""b2d_controller's single-sample deep dive: how one VAD-waypoint PathSpline+vx-spline fit
    (mpc_kf_controller.py's build_trajectory_splines()/preview_from_splines()) actually looks --
    4 panels sharing the same COLOR_*/LINEWIDTH/FONTSIZE_* knobs as every other figure in this file.
    Built for b2d_controller/inspect_one_sample.py, which computes every array this takes (it needs
    the same intermediates for its own printed formulas, so recomputing them here would just be a
    second, possibly-divergent copy).

    fwd, lat: the fit's own input points in this file's (forward, lateral) convention -- [origin,
    *waypoints], length N (7 for VAD's usual 6 waypoints; no route-command target point mixed in --
    see mpc_kf_controller.py's build_trajectory_splines() docstring for why). path: the fitted
    PathSpline. vx_spline: the fitted speed spline. s_max_wp: the last waypoint's station, now also
    equal to path.s_max (the fit no longer extends past the real waypoints). s_mid, v_seg: the
    per-interval speed samples vx_spline was fit against. vx_preview, kappa_preview:
    preview_from_splines()'s own output (length n_p). dt: control period, used only to reconstruct
    the preview's own station cursor for plotting. speed: current speed (m/s, at t=0) -- NOT what
    vx_preview/vx_spline show, which is VAD's own predicted FUTURE speed along the trajectory; the
    two can differ a lot (e.g. current speed high, predicted speed low -- VAD forecasting a
    slowdown into a turn), by design, not by mistake.

    Note kappa_preview/vx_preview only ever cover station 0 to roughly n_p*dt*vx -- a TIME horizon,
    not path.kappa(s)/vx_spline(s)'s own full spatial domain (0 to path.s_max, now the same as
    s_max_wp). At low speed that's a small fraction of the fitted curve; the dense curves are still
    fit from every input point regardless of how far the preview's own marker series happens to
    reach.
    """
    s_cursor = np.concatenate([[0.0], np.cumsum(np.asarray(vx_preview) * dt)[:-1]])
    s_dense = np.linspace(0.0, path.s_max, 300)
    fx, fy = path.xy(s_dense)
    yaw_dense = np.degrees(path.yaw(s_dense))
    kappa_dense = path.kappa(s_dense)
    s_dense_wp = np.linspace(0.0, s_max_wp, 100)

    # Legend labels below are kept short on purpose (this file's convention everywhere else) --
    # a full-sentence label on the speed panel's axhline once made that legend's rendered bbox
    # bigger than its own panel, and constrained_layout (which sizes each panel around everything
    # drawn in it, legends included) shrank the actual data area down into a corner to make room.
    # The fuller explanation lives in the caption below instead.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    if title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE, color=COLOR_INK, fontweight="bold")
    for ax in axes.ravel():
        _style_axes(ax)

    ax = axes[0, 0]
    ax.plot(fx, fy, "-", color=COLOR_BLUE, linewidth=LINEWIDTH, label="fitted path")
    ax.plot(fwd[0], lat[0], "^", color=COLOR_BLUE, markersize=10, label="ego (origin)")
    ax.plot(fwd[1:], lat[1:], "o", color=COLOR_ORANGE, markersize=8, label="VAD waypoints")
    ax.set_xlabel("forward (m)"); ax.set_ylabel("lateral (m)")
    _title(ax, "path fit"); _legend(ax, loc="best")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[0, 1]
    ax.plot(s_dense, yaw_dense, "-", color=COLOR_PURPLE, linewidth=LINEWIDTH)
    ax.set_xlabel("station s (m)"); ax.set_ylabel("yaw (deg)")
    _title(ax, "path.yaw(s)")

    ax = axes[1, 0]
    ax.plot(s_dense, kappa_dense, "-", color=COLOR_PURPLE, linewidth=LINEWIDTH, label="path.kappa(s)")
    ax.plot(s_cursor, kappa_preview, "o", color=COLOR_RED, markersize=5, label="kappa_preview")
    ax.axhline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)
    ax.set_xlabel("station s (m)"); ax.set_ylabel("kappa (1/m)")
    _title(ax, "curvature"); _legend(ax, loc="best")

    ax = axes[1, 1]
    ax.plot(s_dense_wp, vx_spline(s_dense_wp), "-", color=COLOR_AQUA, linewidth=LINEWIDTH, label="vx_spline(s)")
    ax.plot(s_mid, v_seg, "o", color=COLOR_ORANGE, markersize=7, label="speed samples")
    ax.plot(s_cursor, vx_preview, "x", color=COLOR_RED, markersize=6, label="vx_preview")
    ax.set_xlabel("station s (m)"); ax.set_ylabel("vx (m/s)")
    _title(ax, "speed fit"); _legend(ax, loc="best")

    # supxlabel (not a bare fig.text): constrained_layout reserves real space for it like any other
    # figure-level label, so it stays inside the canvas on both plt.show() and savefig() -- a plain
    # fig.text() placed below the constrained_layout-managed area gets clipped by the display
    # window and only survives savefig's separate bbox_inches="tight" recompute.
    fig.supxlabel(
        "*t=0 speed -- vx_preview/vx_spline show VAD's own predicted FUTURE speed along the path, "
        "which can differ a lot (e.g. slowing into this turn).",
        fontsize=FONTSIZE_TICK, color=COLOR_MUTED, wrap=True)

    return fig


def plot_longitudinal(hist, target_speed_ms, title="Longitudinal tracking performance"):
    """fig 2: speed error, speed pair, longitudinal acceleration, longitudinal jerk, total jerk
    magnitude, control input u -- for one controller's single run.

    The last panel plots u = throttle - brake, a single signed series in [-1, 1] (throttle and
    brake are mutually exclusive in every hist this repo logs, so the subtraction reconstructs the
    actual command exactly) rather than the two separate [0, 1] series, since that's the pedal
    signal a controller actually computed before it got split into carla.VehicleControl's two
    fields.

    hist: one run's hist dict. For comparing several controllers' runs against each other, see
    plot_comparison() instead (reuses _error_multi/_dynamics_panel with multi=True there).
    """
    runs, colors = {"": hist}, {"": COLOR_BLUE}
    fig, (ax_ev, ax_v, ax_a, ax_j, ax_jtot, ax_cmd) = _panels(title)

    _error_multi(ax_ev, runs, colors, False, "speed error (m/s)", "Speed error (reference - measured)",
                "m/s", lambda h: np.asarray(speed_error_series(h, target_speed_ms), dtype=float))

    v_des = _get(hist, "v_des")
    if v_des is None:
        ax_v.axhline(target_speed_ms, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    else:
        ax_v.plot(hist["t"], v_des, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    ax_v.plot(hist["t"], hist["v_x"], color=COLOR_BLUE, linewidth=1.8, solid_capstyle="round",
              label="ego vel")
    ax_v.set_ylabel("speed (m/s)")
    _title(ax_v, "Speed")
    _legend(ax_v)

    _dynamics_panel(ax_a, runs, colors, False, "a_x", "$a_x$ (m/s$^2$)", "Longitudinal acceleration",
                    ref_key="a_cmd")
    _b2d_limit_lines(ax_a, *B2D_COMFORT_LIMITS["a_x"])
    _dynamics_panel(ax_j, runs, colors, False, "jerk", "jerk (m/s$^3$)", "Longitudinal jerk (ride comfort)")
    _b2d_limit_lines(ax_j, *B2D_COMFORT_LIMITS["jerk"])
    _dynamics_panel(ax_jtot, runs, colors, False, "jerk_total", "|jerk| (m/s$^3$)",
                    "Total jerk magnitude (long. + lat.)", fill_color=COLOR_PURPLE)
    _b2d_limit_lines(ax_jtot, *B2D_COMFORT_LIMITS["jerk_total"])
    ax_jtot.set_xlabel("t (s)")

    ax_cmd.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    t = np.asarray(hist["t"], dtype=float)
    u = np.asarray(hist["throttle"], dtype=float) - np.asarray(hist["brake"], dtype=float)
    ax_cmd.plot(t, u, color=COLOR_BLUE, linewidth=1.6, solid_capstyle="round")
    ax_cmd.fill_between(t, u, 0, where=(u >= 0), color=COLOR_AQUA, alpha=0.15, interpolate=True)
    ax_cmd.fill_between(t, u, 0, where=(u <= 0), color=COLOR_RED, alpha=0.15, interpolate=True)
    ax_cmd.set_ylim(-1.05, 1.05)
    ax_cmd.set_ylabel("$u$")
    _title(ax_cmd, "Longitudinal control input $u$  (u > 0: throttle, u < 0: brake)")
    ax_cmd.set_xlabel("t (s)")

    return fig


def plot_trajectory(path_x, path_y, hist, title="Desired path vs. ego trajectory"):
    """fig 3: the xy view for one run's driven line against the desired path. Its own figure
    because equal aspect fights a shared time-series grid.

    hist: one run's hist dict. For overlaying several controllers' driven lines on the same
    desired path, see plot_comparison() instead.
    """
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    _style_axes(ax)

    ax.plot(path_x, path_y, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired path")
    ax.plot(hist["x"], hist["y"], color=COLOR_ORANGE, linewidth=2, solid_capstyle="round",
           label="ego trajectory")
    ax.scatter([path_x[0]], [path_y[0]], color=COLOR_BLUE, zorder=5, label="start")
    ax.scatter([path_x[-1]], [path_y[-1]], color=COLOR_RED, marker="*", s=140, zorder=5, label="goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    _title(ax, title)
    ax.set_aspect("equal", adjustable="datalim")
    _legend(ax)
    return fig


def plot_results(path_x, path_y, hist, target_speed_ms, out_dir,
                 show=True, summary=True, label="", name=None):
    """Save + show the three result figures and print the error summary, for one controller's
    single run. For comparing 2+ controllers' runs against each other, see plot_comparison().

    Files are named <script>_<date>_<time>_<kind>.png; pass name to override the script part.
    Returns the list of saved paths (lateral, longitudinal, trajectory).
    """
    if summary:
        print_error_summary(hist, target_speed_ms)

    suffix = f" — {label}" if label else ""
    figures = [
        ("lateral", plot_lateral(hist, f"Lateral tracking performance{suffix}")),
        ("longitudinal", plot_longitudinal(hist, target_speed_ms,
                                           f"Longitudinal tracking performance{suffix}")),
        ("trajectory", plot_trajectory(path_x, path_y, hist)),
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


def plot_comparison(results, path_x, path_y, out_dir, target_speed_ms, show=True, name=None):
    """Compare 2+ controllers' runs against each other, instead of overlaying them onto
    plot_results()'s own three figures (which those are no longer built to do -- see their
    docstrings). Produces:

      1 trajectory figure   -- desired path + every trial's driven path, one color per trial.
      1 comparison figure   -- lateral error, heading error, yaw rate, yaw acceleration, a_x, a_y,
                               longitudinal jerk, total jerk (4x2 grid), each panel one line per
                               trial via the same _error_multi/_dynamics_panel helpers plot_lateral/
                               plot_longitudinal use internally, called here with multi=True.
      2 figures per trial    -- that trial's own plot_lateral()/plot_longitudinal(), labeled.

    results: {label: hist}, 2+ entries. Total figures for N trials: 1 + 1 + 2N (6 for N=2). Also
    prints print_error_summary() per trial, same as plot_results() does for a single run.
    """
    if len(results) < 2:
        raise ValueError(f"plot_comparison needs 2+ results to compare, got {len(results)}")

    for run_label, hist in results.items():
        print(f"\n### {run_label} ###")
        print_error_summary(hist, target_speed_ms)

    colors = dict(zip(results, COMPARE_COLORS))
    os.makedirs(out_dir, exist_ok=True)
    figures = []   # (stem, fig) pairs, saved+closed together at the end

    # ---- 1. trajectory overlay ---- #
    fig_traj, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    fig_traj.patch.set_facecolor(COLOR_BG)
    _style_axes(ax)
    ax.plot(path_x, path_y, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired path")
    for run_label, hist in results.items():
        ax.plot(hist["x"], hist["y"], color=colors[run_label], linewidth=2,
               solid_capstyle="round", label=run_label)
    ax.scatter([path_x[0]], [path_y[0]], color=COLOR_BLUE, zorder=5, label="start")
    ax.scatter([path_x[-1]], [path_y[-1]], color=COLOR_RED, marker="*", s=140, zorder=5, label="goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    _title(ax, "Desired path vs. driven trajectories")
    ax.set_aspect("equal", adjustable="datalim")
    _legend(ax, ncol=min(len(results) + 2, 4))
    figures.append(("trajectory-compare", fig_traj))

    # ---- 2. 8-panel comparison figure ---- #
    # No per-panel legends -- one shared legend for the whole figure goes on at the end instead.
    fig_cmp, (ax_ey, ax_eth, ax_r, ax_racc, ax_ax, ax_ay, ax_j, ax_jtot) = _panels(
        "Performance Comparison", n_rows=4, figsize=(15, 13))

    def _rmse_suffix(key):
        """'  (A: 0.02 | B: 0.48 m)' -- appended to an error panel's title so the number that used
        to live in the legend reads right next to the panel it belongs to instead."""
        parts = []
        for run_label, hist in results.items():
            e = _get(hist, key)
            stats = error_stats(e) if e is not None else None
            if stats is not None:
                parts.append(f"{run_label}: {stats[0]:.2f}")
        return f"  ({' | '.join(parts)})" if parts else ""

    _error_multi(ax_ey, results, colors, True, "lateral error (m)",
                "Lateral error" + _rmse_suffix("e_y") + " m", "m",
                lambda h: _get(h, "e_y"), legend=False)
    _error_multi(ax_eth, results, colors, True, "heading error (deg)",
                "Heading error" + _rmse_suffix("e_theta") + " deg", "deg",
                lambda h: _get(h, "e_theta"), legend=False)
    _dynamics_panel(ax_r, results, colors, True, "yaw_rate", "yaw rate (deg/s)", "Yaw rate", legend=False)
    _b2d_limit_lines(ax_r, *(math.degrees(v) for v in B2D_COMFORT_LIMITS["yaw_rate"]))
    _dynamics_panel(ax_racc, results, colors, True, "yaw_acc", "yaw accel (rad/s$^2$)", "Yaw acceleration",
                    legend=False)
    _b2d_limit_lines(ax_racc, *B2D_COMFORT_LIMITS["yaw_acc"])
    _dynamics_panel(ax_ax, results, colors, True, "a_x", "$a_x$ (m/s$^2$)", "Longitudinal acceleration",
                    legend=False)
    _b2d_limit_lines(ax_ax, *B2D_COMFORT_LIMITS["a_x"])
    _dynamics_panel(ax_ay, results, colors, True, "a_y", "$a_y$ (m/s$^2$)", "Lateral acceleration",
                    legend=False)
    _b2d_limit_lines(ax_ay, *B2D_COMFORT_LIMITS["a_y"])
    _dynamics_panel(ax_j, results, colors, True, "jerk", "jerk (m/s$^3$)", "Longitudinal jerk", legend=False)
    _b2d_limit_lines(ax_j, *B2D_COMFORT_LIMITS["jerk"])
    _dynamics_panel(ax_jtot, results, colors, True, "jerk_total", "|jerk| (m/s$^3$)", "Total jerk magnitude",
                    legend=False)
    _b2d_limit_lines(ax_jtot, *B2D_COMFORT_LIMITS["jerk_total"])
    ax_j.set_xlabel("t (s)")
    ax_jtot.set_xlabel("t (s)")

    # one shared legend for the whole figure, bottom center -- controller-name -> color only (the
    # RMSE numbers now live on the lateral/heading panels' own titles instead)
    handles = [plt.Line2D([0], [0], color=colors[run_label], linewidth=2, label=run_label)
              for run_label in results]
    fig_cmp.legend(handles=handles, loc="lower center", ncol=len(results), frameon=False,
                  labelcolor=COLOR_INK, fontsize=FONTSIZE_LEGEND, bbox_to_anchor=(0.5, -0.02))
    figures.append(("comparison", fig_cmp))

    # ---- 3. each trial's own lateral/longitudinal pair ---- #
    for run_label, hist in results.items():
        suffix = f" — {run_label}"
        stem = run_label.replace(" ", "-")
        figures.append((f"lateral-{stem}",
                        plot_lateral(hist, f"Lateral tracking performance{suffix}")))
        figures.append((f"longitudinal-{stem}",
                        plot_longitudinal(hist, target_speed_ms,
                                          f"Longitudinal tracking performance{suffix}")))

    out_paths = [_save(fig, out_dir, run_name(stem, name)) for stem, fig in figures]
    for path in out_paths:
        print(f"Figure saved: {path}")

    if show:
        plt.show()  # blocks until every window is closed
    for _, fig in figures:
        plt.close(fig)
    return out_paths


def _plot_longitudinal_multi(runs, target_speed_ms, title):
    """Longitudinal-only overlay of 2+ runs, built directly from the shared _error_multi/
    _dynamics_panel helpers (multi=True) -- plot_longitudinal() itself is single-run only (see its
    own docstring), so plot_longitudinal_result builds this comparison layout itself instead of
    delegating to it, the same way plot_comparison() does for the full lateral+longitudinal case."""
    colors = dict(zip(runs, COMPARE_COLORS))
    fig, (ax_ev, ax_v, ax_a, ax_j, ax_jtot, ax_cmd) = _panels(title)

    _error_multi(ax_ev, runs, colors, True, "speed error (m/s)", "Speed error (reference - measured)",
                "m/s", lambda h: np.asarray(speed_error_series(h, target_speed_ms), dtype=float))

    first_hist = next(iter(runs.values()))
    v_des = _get(first_hist, "v_des")
    if v_des is None:
        ax_v.axhline(target_speed_ms, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    else:
        ax_v.plot(first_hist["t"], v_des, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    for run_label, hist in runs.items():
        ax_v.plot(hist["t"], hist["v_x"], color=colors[run_label], linewidth=1.8,
                 solid_capstyle="round", label=run_label)
    ax_v.set_ylabel("speed (m/s)")
    _title(ax_v, "Speed")
    _legend(ax_v, ncol=min(len(runs) + 1, 4))

    _dynamics_panel(ax_a, runs, colors, True, "a_x", "$a_x$ (m/s$^2$)", "Longitudinal acceleration",
                    ref_key="a_cmd")
    _b2d_limit_lines(ax_a, *B2D_COMFORT_LIMITS["a_x"])
    _dynamics_panel(ax_j, runs, colors, True, "jerk", "jerk (m/s$^3$)", "Longitudinal jerk (ride comfort)")
    _b2d_limit_lines(ax_j, *B2D_COMFORT_LIMITS["jerk"])
    _dynamics_panel(ax_jtot, runs, colors, True, "jerk_total", "|jerk| (m/s$^3$)",
                    "Total jerk magnitude (long. + lat.)")
    _b2d_limit_lines(ax_jtot, *B2D_COMFORT_LIMITS["jerk_total"])
    ax_jtot.set_xlabel("t (s)")

    ax_cmd.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    for run_label, hist in runs.items():
        t = np.asarray(hist["t"], dtype=float)
        u = np.asarray(hist["throttle"], dtype=float) - np.asarray(hist["brake"], dtype=float)
        ax_cmd.plot(t, u, color=colors[run_label], linewidth=1.4, label=run_label)
    _legend(ax_cmd, ncol=len(runs))
    ax_cmd.set_ylim(-1.05, 1.05)
    ax_cmd.set_ylabel("$u$")
    _title(ax_cmd, "Longitudinal control input $u$  (u > 0: throttle, u < 0: brake)")
    ax_cmd.set_xlabel("t (s)")
    return fig


def plot_longitudinal_result(data, target_speed_ms, out_dir, show=True, summary=True, label="", name=None):
    """Like plot_results(), but only the longitudinal figure.

    For stacks with no lateral control at all (e.g. longitudinal_PID.py, steer pinned at 0) --
    there's no path or steering to put in the other two figures.

    data: a single hist dict, or {label: hist} to overlay several controllers on one figure --
    e.g. longitudinal_mpc.py's --controller both. Each run gets its own error-summary block when
    there's more than one.
    """
    runs = _runs(data)
    if summary:
        for run_label, hist in runs.items():
            if run_label:
                print(f"\n### {run_label} ###")
            print_error_summary(hist, target_speed_ms)

    suffix = f" — {label}" if label else ""
    title = f"Longitudinal tracking performance{suffix}"
    if len(runs) > 1:
        fig = _plot_longitudinal_multi(runs, target_speed_ms, title)
    else:
        fig = plot_longitudinal(next(iter(runs.values())), target_speed_ms, title)

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
    _title(ax_main, "Tracking")
    _legend(ax_main, ncol=len(series) + 1)

    ax_err.axhline(0.0, color=COLOR_AXIS, linewidth=1, linestyle="--")
    for label, hist, color in series:
        err = np.asarray(hist[ref_key], dtype=float) - np.asarray(hist[resp_key], dtype=float)
        ax_err.plot(ts[label], err, color=color, linewidth=1.4, label=label)
    ax_err.set_ylabel(f"error ({unit})")
    _title(ax_err, "Tracking error (reference - measured)")
    _legend(ax_err, ncol=len(series))

    ax_u.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    for label, hist, color in series:
        ax_u.plot(ts[label], hist["u"], color=color, linewidth=1.3, label=label)
    ax_u.set_ylim(-1.15, 1.15)
    ax_u.set_ylabel("pedal $u$")
    ax_u.set_xlabel("t (s)")
    _title(ax_u, "Control input")
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
                fontsize=12, color=COLOR_INK, fontweight="bold")

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
        _title(ax, title, fontsize=10)
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
                color=COLOR_INK, fontsize=13, fontweight="bold")

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

        _title(ax, f"Gear {gear}", fontsize=11)
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

def plot_cornering_stiffness_speed_bands(queued, pooled, by_speed, out_dir=None, show=True,
                                         name=None):
    """For estimate_cornering_stiffness.py: alpha-vs-Fy fit for the selected speeds, colored by
    target speed, with ONE band spanning the selected speeds' own point-estimate Cf/Cr values
    (not each speed's internal per-steer spread) drawn behind the final pooled line.

    The point being made is "pooling the selected speeds into one Cf/Cr is justified": if the
    band spanning what each selected speed's own Cf/Cr independently came out to is already
    tight around the pooled (black dashed) line, the data itself says Cf/Cr doesn't drift across
    the speeds that made it through selection -- rather than just asserting that and pooling
    anyway. This is deliberately NOT each speed's own internal uncertainty (that question --
    "is any given speed's own fit noisy" -- is what estimate_cornering_stiffness.py's speed
    SELECTION step already answered before a speed's trials ever reach this plot).

    queued/pooled/by_speed: collect_sweep_trials()'s own return values (or, from estimate_
    cornering_stiffness.py's main(), the selected-speed subset of them), plotted as-is -- the
    estimation itself (fit_cornering_stiffness/pool_trials/pool_by_speed) is untouched by this
    function. by_speed[v]["Cf"/"Cr"] (one point estimate per speed) is what the band's min/max is
    drawn from, not by_speed[v]["Cf_min"/"Cf_max"] (that speed's own internal spread).
    """
    import matplotlib.pyplot as plt

    done = [tr for tr in queued if tr.get("fit") is not None]
    speeds = sorted({tr["target_speed"] for tr in done})
    cmap = plt.get_cmap("plasma")
    # continuous norm over the actual speed values (not evenly spaced by rank), so the colorbar's
    # own axis is a true speed scale -- 4 and 5 m/s sit close together on it, 4 and 15 m/s far
    # apart, matching what the tick labels say rather than just enumerating "speed #3 of 12".
    speed_norm = plt.Normalize(min(speeds), max(speeds)) if len(speeds) > 1 else None
    speed_color = ({v: cmap(speed_norm(v)) for v in speeds} if speed_norm
                   else {speeds[0]: cmap(0.5)})

    fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    for ax in (ax_f, ax_r):
        _style_axes(ax)
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)
        ax.axvline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)

    for ax, key_alpha, key_Fy, c_key, C_pooled, axle in (
            (ax_f, "alpha_f", "Fyf", "Cf", pooled["Cf"], "front"),
            (ax_r, "alpha_r", "Fyr", "Cr", pooled["Cr"], "rear")):
        all_alpha_deg = np.degrees(pooled[key_alpha])
        lo, hi = min(0.0, all_alpha_deg.min()), max(0.0, all_alpha_deg.max())
        xs = np.linspace(lo, hi, 20)

        C_vals = [r[c_key] for r in by_speed.values() if r is not None]
        if C_vals:
            C_lo, C_hi = min(C_vals), max(C_vals)
            ax.fill_between(xs, C_lo * np.radians(xs), C_hi * np.radians(xs), color=COLOR_MUTED,
                            alpha=0.18, zorder=1,
                            label=f"selected speeds' C range [{C_lo:,.0f}, {C_hi:,.0f}]")

        for tr in done:
            alpha_deg = np.degrees(tr["fit"][key_alpha])
            Fy = tr["fit"][key_Fy]
            ax.scatter(alpha_deg, Fy, s=MARKERSIZE, facecolor=speed_color[tr["target_speed"]],
                      edgecolor=COLOR_BG, linewidth=0.6, alpha=0.9, zorder=3)

        ax.plot(xs, C_pooled * np.radians(xs), color=COLOR_INK, linewidth=LINEWIDTH * 1.4,
               linestyle="--", zorder=4, label=f"pooled C={C_pooled:,.0f} N/rad")
        ax.set_xlabel(f"alpha_{axle[0]} (deg)")
        ax.set_ylabel(f"Fy{axle[0]} (N)")
        _title(ax, f"{axle.capitalize()} axle: Fy = C * alpha, selected speeds")
        _legend(ax)

    # vertical colorbar (not a swatch legend) so "what speed is this color" reads off a continuous
    # scale instead of matching dots to a wrapped multi-row legend -- the right call once there
    # are enough speeds that a legend row per speed stops being readable.
    if speed_norm is not None:
        sm = plt.cm.ScalarMappable(norm=speed_norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_f, ax_r], pad=0.02, aspect=30)
        cbar.set_label("target speed (m/s)", fontsize=FONTSIZE_LABEL, color=COLOR_MUTED)
        cbar.ax.tick_params(colors=COLOR_MUTED, labelsize=FONTSIZE_TICK)
        cbar.outline.set_visible(False)

    out_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = _save(fig, out_dir, run_name("cornering_stiffness_speed_bands", name))
        print(f"Figure saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_speed_slip_angle(queued, out_dir=None, show=True, name=None):
    """For estimate_cornering_stiffness.py: every steady-window sample's (v_x, alpha) plotted
    directly, colored by steer angle -- shows how far alpha_f/alpha_r actually swept at each speed
    (not just its fitted Cf), which is the evidence --speeds should be widened or narrowed against:
    a speed whose slip-angle range has collapsed toward the noise floor, or whose steer-angle
    "rays" have started crossing, is a speed no longer worth including in the pooled fit.
    """
    import matplotlib.pyplot as plt

    done = [tr for tr in queued if tr.get("fit") is not None]
    steers = sorted({tr["steer_deg"] for tr in done})
    cmap = plt.get_cmap("viridis")
    steer_color = {d: cmap(i / max(1, len(steers) - 1)) for i, d in enumerate(steers)}

    fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    for ax in (ax_f, ax_r):
        _style_axes(ax)
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)

    for tr in done:
        color = steer_color[tr["steer_deg"]]
        v_x = np.asarray(tr["log"]["v_x"])[slice(*tr["window"])]
        alpha_f_deg = np.degrees(tr["fit"]["alpha_f"])
        alpha_r_deg = np.degrees(tr["fit"]["alpha_r"])
        ax_f.scatter(v_x, alpha_f_deg, s=MARKERSIZE * 0.5, facecolor=color, edgecolor="none",
                    alpha=0.5, zorder=2)
        ax_r.scatter(v_x, alpha_r_deg, s=MARKERSIZE * 0.5, facecolor=color, edgecolor="none",
                    alpha=0.5, zorder=2)

    ax_f.set_xlabel("v_x (m/s)"); ax_f.set_ylabel("alpha_f (deg)")
    ax_r.set_xlabel("v_x (m/s)"); ax_r.set_ylabel("alpha_r (deg)")
    _title(ax_f, "Front slip angle vs speed")
    _title(ax_r, "Rear slip angle vs speed")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=steer_color[d],
                          markeredgecolor="none", markersize=8, label=f"{d:g} deg")
              for d in steers]
    _bottom_legend(fig, handles, title="steer")

    out_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = _save(fig, out_dir, run_name("speed_slip_angle", name))
        print(f"Figure saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def cornering_stiffness_speed_stats(queued, by_speed):
    """Per-speed sample count / Cf & Cr / std(C across that speed's own steer angles) / alpha
    range, front AND rear axle both -- the numbers behind both print_cornering_stiffness_speed_
    table() and estimate_cornering_stiffness.py's own speed-selection step (drop a speed if
    either axle's C_std_pct or alpha_max_deg exceeds a threshold), split out as its own function
    so both read the exact same numbers instead of the selection logic recomputing its own
    version of what the table already printed.

    Returns {v: {"n_samples", "Cf", "Cf_std", "Cf_std_pct", "alpha_f_min_deg", "alpha_f_max_deg",
    "Cr", "Cr_std", "Cr_std_pct", "alpha_r_min_deg", "alpha_r_max_deg"}}, skipping speeds with no
    settled trial (by_speed.get(v) is None). n_samples is shared (front/rear windows are the same
    ticks, just a different axle's slip angle/force).
    """
    done = [tr for tr in queued if tr.get("fit") is not None]
    stats = {}
    for v in sorted({tr["target_speed"] for tr in done}):
        trials_v = [tr for tr in done if tr["target_speed"] == v]
        r = by_speed.get(v)
        if r is None or not trials_v:
            continue
        n_samples = sum(len(tr["fit"]["alpha_f"]) for tr in trials_v)
        Cf_std = float(np.std([tr["fit"]["Cf"] for tr in trials_v]))
        Cr_std = float(np.std([tr["fit"]["Cr"] for tr in trials_v]))
        alpha_f_all = np.degrees(np.concatenate([tr["fit"]["alpha_f"] for tr in trials_v]))
        alpha_r_all = np.degrees(np.concatenate([tr["fit"]["alpha_r"] for tr in trials_v]))
        stats[v] = {
            "n_samples": n_samples,
            "Cf": r["Cf"], "Cf_std": Cf_std,
            "Cf_std_pct": 100.0 * Cf_std / r["Cf"] if r["Cf"] else float("inf"),
            "alpha_f_min_deg": float(alpha_f_all.min()), "alpha_f_max_deg": float(alpha_f_all.max()),
            "Cr": r["Cr"], "Cr_std": Cr_std,
            "Cr_std_pct": 100.0 * Cr_std / r["Cr"] if r["Cr"] else float("inf"),
            "alpha_r_min_deg": float(alpha_r_all.min()), "alpha_r_max_deg": float(alpha_r_all.max()),
        }
    return stats


def print_cornering_stiffness_speed_table(queued, by_speed):
    """Per-speed sample count / Cf & Cr / std(C across that speed's own steer angles) / alpha
    range, front and rear both -- the plain-text companion to
    plot_cornering_stiffness_speed_bands(): n_samples too small or the alpha range collapsing
    toward 0 both say a speed has hit the noise floor; a wide C std says it's drifting instead.
    Together with plot_speed_slip_angle(), this is the evidence for deciding which speeds are
    actually worth pooling into the final Cf/Cr -- on EITHER axle, not just the front, since a
    speed can look fine at the front and still be noise on the rear (lower Fzr/alpha_r means the
    rear fit generally has less signal to work with).
    """
    stats = cornering_stiffness_speed_stats(queued, by_speed)
    print(f"\n{'v (m/s)':>8} {'n samp':>7} "
          f"{'Cf (N/rad)':>12} {'Cf std%':>8} {'alpha_f (deg)':>18} "
          f"{'Cr (N/rad)':>12} {'Cr std%':>8} {'alpha_r (deg)':>18}")
    all_speeds = sorted({tr["target_speed"] for tr in queued})
    for v in all_speeds:
        s = stats.get(v)
        if s is None:
            print(f"{v:8.1f} {'--':>7} {'--':>12} {'--':>8} {'--':>18} "
                  f"{'--':>12} {'--':>8} {'--':>18}")
            continue
        print(f"{v:8.1f} {s['n_samples']:7d} "
              f"{s['Cf']:12,.0f} {s['Cf_std_pct']:7.1f}% "
              f"[{s['alpha_f_min_deg']:+6.2f},{s['alpha_f_max_deg']:+6.2f}] "
              f"{s['Cr']:12,.0f} {s['Cr_std_pct']:7.1f}% "
              f"[{s['alpha_r_min_deg']:+6.2f},{s['alpha_r_max_deg']:+6.2f}]")
