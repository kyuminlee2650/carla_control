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
    fig 1  lateral tracking     -- cross-track error, heading error, yaw, yaw rate, v_y, steer
    fig 2  longitudinal tracking-- speed error, speed, a_x, jerk, throttle, brake
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


def _legend(ax, **kwargs):
    ax.legend(frameon=False, labelcolor=COLOR_INK, fontsize=9, **kwargs)


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
    """RMSE / max / mean of each tracked error over the whole run."""
    if not hist["t"]:
        return

    print(f"\n=== error summary: {len(hist['t'])} steps, {hist['t'][-1]:.1f} s ===")
    for name, unit, series in (("cross-track", "m", hist["e_y"]),
                               ("heading    ", "deg", hist["e_theta"]),
                               ("speed      ", "m/s", speed_error_series(hist, target_speed_ms))):
        stats = error_stats(series)
        if stats is None:
            continue
        rmse, peak, bias = stats
        print(f"  {name}  RMSE={rmse:7.3f} {unit:<3}  max|e|={peak:7.3f} {unit:<3}  mean={bias:+7.3f} {unit}")


# ---------------------------------------------------------------------------
# 7. plotting internals
# ---------------------------------------------------------------------------

def _panels(title, n_rows=3, n_cols=2, figsize=(15, 10)):
    """A styled grid sharing the time axis, flattened in row-major order."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(title, fontsize=14, color=COLOR_INK)
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


def _error_panel(ax, t, e, ylabel, color, unit):
    ax.axhline(0.0, color=COLOR_AXIS, linewidth=1, linestyle="--")
    ax.plot(t, e, color=color, linewidth=1.6, solid_capstyle="round")
    ax.fill_between(t, e, 0, color=color, alpha=0.15)
    ax.set_ylabel(ylabel)
    _rmse_badge(ax, e, unit)


def _save(fig, out_dir, stem):
    out_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    return out_path


# ---------------------------------------------------------------------------
# 8. result figures
# ---------------------------------------------------------------------------

def plot_lateral(hist, title="Lateral tracking performance"):
    """fig 1: cross-track error, heading error, yaw pair, yaw rate, lateral velocity, steer."""
    t = np.asarray(hist["t"], dtype=float)
    fig, (ax_ey, ax_eth, ax_yaw, ax_r, ax_vy, ax_steer) = _panels(title)

    _error_panel(ax_ey, t, _get(hist, "e_y"), "cross-track error (m)", COLOR_ORANGE, "m")
    ax_ey.set_title("Cross-track error vs. global path (front axle)")

    _error_panel(ax_eth, t, _get(hist, "e_theta"), "heading error (deg)", COLOR_RED, "deg")
    ax_eth.set_title("Heading error (path - vehicle)")

    ax_yaw.plot(t, hist["path_yaw"], color=COLOR_MUTED, linewidth=2, linestyle="--", label="path yaw")
    ax_yaw.plot(t, hist["yaw"], color=COLOR_BLUE, linewidth=1.8, solid_capstyle="round", label="ego yaw")
    ax_yaw.set_ylabel("heading (deg)")
    ax_yaw.set_title("Vehicle heading vs. road heading")
    _legend(ax_yaw, ncol=2)

    yaw_rate = _get(hist, "yaw_rate")
    if yaw_rate is None:
        _not_recorded(ax_r, "yaw_rate")
    else:
        ax_r.axhline(0.0, color=COLOR_AXIS, linewidth=1)
        ax_r.plot(t, yaw_rate, color=COLOR_PURPLE, linewidth=1.5, solid_capstyle="round")
    ax_r.set_ylabel("yaw rate (deg/s)")
    ax_r.set_title("Yaw rate")

    v_y = _get(hist, "v_y")
    if v_y is None:
        _not_recorded(ax_vy, "v_y")
    else:
        ax_vy.axhline(0.0, color=COLOR_AXIS, linewidth=1)
        ax_vy.plot(t, v_y, color=COLOR_AQUA, linewidth=1.5, solid_capstyle="round")
    ax_vy.set_ylabel("$v_y$ (m/s)")
    ax_vy.set_title("Lateral velocity (body frame)")
    ax_vy.set_xlabel("t (s)")

    ax_steer.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_steer.plot(t, hist["steer_deg"], color=COLOR_BLUE, linewidth=1.5, solid_capstyle="round")
    ax_steer.set_ylabel("steer (deg)")
    ax_steer.set_title("Steering angle (front wheel)")
    ax_steer.set_xlabel("t (s)")

    return fig


def plot_longitudinal(hist, target_speed_ms, title="Longitudinal tracking performance"):
    """fig 2: speed error, speed pair, longitudinal acceleration, jerk, throttle, brake."""
    t = np.asarray(hist["t"], dtype=float)
    fig, (ax_ev, ax_v, ax_a, ax_j, ax_thr, ax_brk) = _panels(title)

    e_vel = np.asarray(speed_error_series(hist, target_speed_ms), dtype=float)
    _error_panel(ax_ev, t, e_vel, "speed error (m/s)", COLOR_RED, "m/s")
    ax_ev.set_title("Speed error (reference - measured)")

    v_des = _get(hist, "v_des")
    if v_des is None:
        ax_v.axhline(target_speed_ms, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    else:
        ax_v.plot(t, v_des, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    ax_v.plot(t, hist["v_x"], color=COLOR_BLUE, linewidth=1.8, solid_capstyle="round", label="ego vel")
    ax_v.set_ylabel("speed (m/s)")
    ax_v.set_title("Speed")
    _legend(ax_v, ncol=2)

    a_x = _get(hist, "a_x")
    if a_x is None:
        _not_recorded(ax_a, "a_x")
    else:
        ax_a.axhline(0.0, color=COLOR_AXIS, linewidth=1)
        ax_a.plot(t, a_x, color=COLOR_BLUE, linewidth=1.5, solid_capstyle="round")
    ax_a.set_ylabel("$a_x$ (m/s$^2$)")
    ax_a.set_title("Longitudinal acceleration")

    jerk = _get(hist, "jerk")
    if jerk is None:
        _not_recorded(ax_j, "jerk")
    else:
        ax_j.axhline(0.0, color=COLOR_AXIS, linewidth=1)
        ax_j.plot(t, jerk, color=COLOR_PURPLE, linewidth=1.3, solid_capstyle="round")
    ax_j.set_ylabel("jerk (m/s$^3$)")
    ax_j.set_title("Longitudinal jerk (ride comfort)")

    ax_thr.plot(t, hist["throttle"], color=COLOR_AQUA, linewidth=1.6, solid_capstyle="round")
    ax_thr.fill_between(t, hist["throttle"], 0, color=COLOR_AQUA, alpha=0.15)
    ax_thr.set_ylim(-0.05, 1.05)
    ax_thr.set_ylabel("throttle")
    ax_thr.set_title("Throttle")
    ax_thr.set_xlabel("t (s)")

    ax_brk.plot(t, hist["brake"], color=COLOR_RED, linewidth=1.6, solid_capstyle="round")
    ax_brk.fill_between(t, hist["brake"], 0, color=COLOR_RED, alpha=0.15)
    ax_brk.set_ylim(-0.05, 1.05)
    ax_brk.set_ylabel("brake")
    ax_brk.set_title("Brake")
    ax_brk.set_xlabel("t (s)")

    return fig


def plot_trajectory(path_x, path_y, hist, title="Desired path vs. ego trajectory"):
    """fig 3: the xy view. Its own figure because equal aspect fights a shared time-series grid."""
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    _style_axes(ax)

    ax.plot(path_x, path_y, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired path")
    ax.plot(hist["x"], hist["y"], color=COLOR_ORANGE, linewidth=2, solid_capstyle="round", label="ego trajectory")
    ax.scatter([path_x[0]], [path_y[0]], color=COLOR_BLUE, zorder=5, label="start")
    ax.scatter([path_x[-1]], [path_y[-1]], color=COLOR_RED, marker="*", s=140, zorder=5, label="goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, color=COLOR_INK)
    ax.set_aspect("equal", adjustable="datalim")
    _legend(ax)
    return fig


def plot_results(path_x, path_y, hist, target_speed_ms, out_dir,
                 show=True, summary=True, label="", name=None):
    """Save + show the three result figures and print the error summary.

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
