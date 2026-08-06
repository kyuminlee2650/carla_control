"""Live top-down (BEV) view of the desired path and the ego vehicle -- self-driving.

Finds the spawned vehicle and refreshes itself every simulation tick via world.on_tick(); the
control script only has to construct a BevView once and call close(), same as before -- no
per-tick push needed. Draws exactly: desired path (static), ego trajectory (trail), current ego
position, and last_idx (nearest desired-path index to the ego, recomputed here independently of
whatever the control script tracks internally -- BevView never reads the control script's state).

Process model: matplotlib/Tk GUI calls are only safe on a process's real main thread, so the whole
plot lives in its own child process (started here, killed in close()) rather than a background
thread of the caller's process. Inside that child process, world.on_tick() callbacks still arrive
on CARLA's own internal listener thread, so a lock still guards the handoff to the redraw timer,
which runs on the child process's main thread via plt.show().
"""
import math
import multiprocessing as mp
import sys
import threading
import time

sys.path.append("/home/ailab/carla/CARLA_0.9.15/PythonAPI/carla")
import carla


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
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    COLOR_BG = "#fcfcfb"
    COLOR_GRID = "#e1e0d9"
    COLOR_AXIS = "#c3c2b7"
    COLOR_MUTED = "#898781"
    COLOR_ORANGE = "#eb6834"   # ego trajectory
    COLOR_BLUE = "#2a78d6"     # ego position

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
    ax.set_facecolor(COLOR_BG)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.plot(path_x, path_y, color=COLOR_MUTED, linewidth=1.5, linestyle="--", label="desired path")
    (trail_line,) = ax.plot([], [], color=COLOR_ORANGE, linewidth=2, label="ego trajectory")
    (ego_dot,) = ax.plot([], [], color=COLOR_BLUE, marker="o", markersize=9, linestyle="", label="ego")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

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
