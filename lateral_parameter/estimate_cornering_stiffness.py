r"""Steady-state front/rear cornering stiffness (Cf, Cr) estimation.

Drives Town03's central roundabout (a closed loop, so there's no "running out of road" the way a
straight-line sweep would) across a sweep of target speeds, using a light Stanley controller to
hold the car on the roundabout's own lane rather than an open-loop constant steer -- an open-loop
angle that doesn't *exactly* match the road's curvature drifts the car sideways over the seconds
it takes to reach steady state, and on a 2-lane road with a central island that ends in a curb
strike (confirmed live: a fixed steer collided at ~4s in). Stanley just keeps the car following
the known road geometry instead; whatever steer angle it ends up commanding each tick is measured
and used directly, so the physics below doesn't care that the angle isn't open-loop.

At each speed, once the car settles into steady circular motion (r_dot ~ 0, v_x converged), the
2-DOF bicycle model's steady-state force/moment balance no longer involves the yaw inertia at all
-- Iz only shows up once r_dot is nonzero -- so Cf/Cr can be solved for here without knowing Iz.
estimate_yaw_inertia.py (a separate, later script) loads this script's output and uses it to solve
for Iz from a transient step-steer instead; see that file's docstring for why the two are split.

Model (standard 2-DOF bicycle, CG-relative slip angles):

    beta   = v_y / v_x                              (sideslip; CARLA gives v_y directly -- a real
                                                       car can't measure this without extra sensors,
                                                       but this is a sim, so it's free)
    alpha_f = delta - beta - lf * r / v_x            (front slip angle)
    alpha_r =       - beta + lr * r / v_x            (rear slip angle)

At steady state (r_dot = 0), summing forces/moments about the CG gives the tire forces directly
from the motion, with no tire model needed yet:

    Fyf = (lr / L) * m * v_x * r
    Fyr = (lf / L) * m * v_x * r

so Cf = Fyf / alpha_f and Cr = Fyr / alpha_r at each swept speed; averaging (least squares) over
several speeds is just noise reduction, not a requirement for identifiability -- each operating
point already determines Cf and Cr independently.

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python lateral_parameter/estimate_cornering_stiffness.py --save-plot

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe lateral_parameter\estimate_cornering_stiffness.py
    .venv\Scripts\python.exe lateral_parameter\estimate_cornering_stiffness.py --speeds 3,4,5,6,7 --save-plot
"""

import argparse
import json
import math
import os
import queue
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# functions.py, viz_utils.py live one level up, alongside the other controllers
sys.path.append(os.path.dirname(HERE))

CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/2026intern/carla"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

import carla

from functions import PID, CollisionWatch, LowPassFilter, clipping, get_vehicle_geometry, lateral_error, normalize_angle

MAP_NAME = "Town03"

# Town03's central roundabout: a closed ring across road_id 9-14, found by scanning the map's
# waypoints for a long sustained-curvature arc (see the design discussion -- this isn't documented
# anywhere in CARLA itself). Lane -4 is the inner lane, radius ~21 m; -5 is the outer, ~25 m.
RING_ROAD_ID = 9
RING_LANE_ID = -4


def stanley_control(v_x, e_y, e_theta, k_theta=1.0, k=1.2, k_soft=1.0):
    return k_theta * e_theta + math.atan2(k * e_y, k_soft + abs(v_x))


def build_ring_path(carla_map, step=2.0, n_points=90):
    """Walk forward from the roundabout's inner lane, ~n_points*step meters -- comfortably more
    than one full lap (circumference ~130 m) -- so lateral_error() always has road ahead of the
    car during a several-second trial. Loop position wrapping doesn't matter here since only the
    array's forward order (not real-world uniqueness) is used."""
    wp = None
    for w in carla_map.generate_waypoints(2.0):
        if w.road_id == RING_ROAD_ID and w.lane_id == RING_LANE_ID:
            wp = w
            break
    if wp is None:
        raise RuntimeError(f"no waypoint found on road_id={RING_ROAD_ID} lane_id={RING_LANE_ID} -- "
                           f"has the roundabout's road layout changed?")

    path_x, path_y, path_yaw = [], [], []
    for _ in range(n_points):
        path_x.append(wp.transform.location.x)
        path_y.append(wp.transform.location.y)
        path_yaw.append(math.radians(wp.transform.rotation.yaw))
        nxt = wp.next(step)
        if not nxt:
            break
        wp = nxt[0]
    return path_x, path_y, path_yaw


def run_trial(world, path_x, path_y, path_yaw, blueprint, imu_bp, max_steer, target_speed, args):
    """Spawn at rest at the start of the ring path, hold the car on it with Stanley while a PID
    brings v_x up to and holds `target_speed`, wait for steady circular motion, then collect a
    short window of (v_x, v_y, r, delta).

    Returns the window dict {"v_x": [...], "v_y": [...], "r": [...], "delta": [...]}, one entry
    per tick, NOT pre-averaged -- alpha_f/alpha_r/Fyf/Fyr are nonlinear (products/ratios) in these,
    so averaging v_x and r separately and then multiplying is not the same as multiplying per tick
    and then averaging, and Stanley's own steer-correction oscillation (r swings ~+-30% around its
    mean within a window) is large enough that this order-of-operations bias turned out to matter.
    None if the car never settled (timeout) or hit something.
    """
    spawn_transform = carla.Transform(
        carla.Location(path_x[0], path_y[0], 0.3),
        carla.Rotation(yaw=math.degrees(path_yaw[0])),
    )
    vehicle = world.spawn_actor(blueprint, spawn_transform)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    collision = CollisionWatch(world, vehicle)
    collision.arm()

    speed_pid = PID(kp=0.5, ki=0.2, kd=0.0, dt=args.dt)
    r_dot_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)   # raw gyro diff is too noisy to gate on directly
    # The commanded steer is what's sent to the vehicle every tick (must react fast to hold the
    # lane); this filtered copy is a separate estimate of the *actual* front wheel angle for the
    # alpha_f calc only, on the theory that a real steering actuator can't track a fast-toggling
    # command instantaneously -- using the raw command as if it were the true wheel angle would
    # itself be a modeling error whenever delta is moving quickly, which is exactly when Stanley
    # is fighting to hold the lane.
    delta_filter = LowPassFilter(tau=args.delta_tau, dt=args.dt, initial=0.0)
    prev_r = None
    last_idx = 0

    settle_ticks = int(args.settle_time / args.dt)
    window_ticks = int(args.window / args.dt)
    converged_ticks = 0
    window = {"v_x": [], "v_y": [], "r": [], "delta": []}

    imu = None
    try:
        world.tick()
        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)

        steps = int(args.max_duration / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)

            transform = vehicle.get_transform()
            yaw = math.radians(transform.rotation.yaw)
            ego_x, ego_y = transform.location.x, transform.location.y
            vel = vehicle.get_velocity()
            v_x = vel.x * math.cos(yaw) + vel.y * math.sin(yaw)
            v_y = -vel.x * math.sin(yaw) + vel.y * math.cos(yaw)
            r = imu_data.gyroscope.z   # rad/s, body-frame yaw rate straight off the sensor

            last_idx, e_y = lateral_error(ego_x, ego_y, yaw, path_x, path_y, last_idx)
            e_theta = normalize_angle(path_yaw[last_idx] - yaw)
            delta = stanley_control(v_x, e_y, e_theta, k_theta=args.k_theta, k=args.k_stanley)
            steer_cmd = clipping(delta / max_steer, 1.0, -1.0)
            # run every tick (not just once logging starts) so the filter's lag is already
            # warmed up by the time we start using its output
            delta_true = delta_filter.step(steer_cmd * max_steer)

            u = clipping(speed_pid.step(target_speed - v_x), 1.0, -1.0)
            control = carla.VehicleControl()
            control.throttle, control.brake = (u, 0.0) if u >= 0 else (0.0, -u)
            control.steer = steer_cmd
            vehicle.apply_control(control)

            if collision.hit:
                print(f"    collision at t={i*args.dt:.1f}s -- aborting this trial")
                return None

            r_dot = r_dot_filter.step(0.0 if prev_r is None else (r - prev_r) / args.dt)
            prev_r = r

            if not window["v_x"]:  # still waiting to settle
                # Only gate on v_x here -- r_dot never stays inside a tight tolerance because
                # Stanley's own e_y corrections keep nudging the steer angle (confirmed live: r
                # oscillates roughly +-30% around its true mean, e.g. -9 to -18 deg/s for a mean
                # near -14 deg/s, which matches v_x/R for this roundabout). A longer averaging
                # window below cancels that oscillation instead of waiting for it to vanish.
                settled = abs(v_x - target_speed) < args.speed_tol
                converged_ticks = converged_ticks + 1 if settled else 0
                if i % 20 == 0:
                    print(f"    t={i*args.dt:5.1f}s  v_x={v_x:5.2f}/{target_speed:.1f} m/s  "
                          f"r={math.degrees(r):+6.2f} deg/s  r_dot={r_dot:+.3f} rad/s^2  "
                          f"e_y={e_y:+.2f} m  settled_ticks={converged_ticks}/{settle_ticks}")
                if converged_ticks >= settle_ticks:
                    window["v_x"].append(v_x)
                    window["v_y"].append(v_y)
                    window["r"].append(r)
                    window["delta"].append(delta_true)
            else:
                window["v_x"].append(v_x)
                window["v_y"].append(v_y)
                window["r"].append(r)
                window["delta"].append(delta_true)
                if len(window["v_x"]) >= window_ticks:
                    print(f"    window std: v_x={np.std(window['v_x']):.3f} m/s  "
                          f"v_y={np.std(window['v_y']):.3f} m/s  r={math.degrees(np.std(window['r'])):.2f} deg/s  "
                          f"delta={math.degrees(np.std(window['delta'])):.2f} deg")
                    return window   # full per-tick window, not pre-averaged -- see main()'s docstring note

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    finally:
        collision.destroy()
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        vehicle.destroy()

    return None   # never settled within max_duration


def fit_stiffness(alpha, Fy):
    """Least-squares Cf or Cr through the origin: min_C sum((C*alpha - Fy)^2)."""
    alpha = np.asarray(alpha)
    Fy = np.asarray(Fy)
    return float(np.sum(alpha * Fy) / np.sum(alpha * alpha))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--times-run", type=float, default=2.0, help="how times for simulation running?")

    parser.add_argument("--speeds", default="4,5,6,7,8,9",
                        help="comma-separated target speeds to sweep (m/s) -- off the low end "
                             "where slip angles get small enough that noise in v_y/r dominates "
                             "(both alpha_f/alpha_r and beta divide by v_x, so low speed is where "
                             "estimation noise is worst, not best), pushing the high end past the "
                             "usual ~0.3g linear-tire ceiling on purpose to see where Cf/Cr stop "
                             "holding steady -- check the per-point fit, not just the pooled one")
    parser.add_argument("--k-theta", type=float, default=1.0,
                        help="Stanley heading-error gain (stanley_PID.py's own default). Softening "
                             "this was tried as a fix for r oscillating within a settled window "
                             "(+-30%% or more around its mean) -- it made tracking worse (e_y grew "
                             "to 2.7-3.3 m, nearly off the lane) without reducing the oscillation, "
                             "so the oscillation isn't from over-correction; it's something shorter"
                             "-timescale (waypoint spacing discretization, or genuinely not reaching "
                             "steady state in a few seconds). Left at the safe default; fit_stiffness() "
                             "already fits over every raw tick rather than a pre-averaged window, "
                             "which is the correct way to average out this kind of noise")
    parser.add_argument("--k-stanley", type=float, default=1.2, help="Stanley cross-track gain (stanley_PID.py's own default)")
    parser.add_argument("--delta-tau", type=float, default=0.15,
                        help="low-pass time constant (s) modeling steering-actuator lag between "
                             "the commanded steer and the true front wheel angle used in alpha_f -- "
                             "only affects the alpha_f calc, never what's actually sent to the "
                             "vehicle (that stays the raw, fast-reacting command)")
    parser.add_argument("--speed-tol", type=float, default=0.2, help="settled once |v_x - target| is under this (m/s)")
    parser.add_argument("--settle-time", type=float, default=2.0, help="how long v_x must hold within tolerance before logging starts (s)")
    parser.add_argument("--window", type=float, default=2.5,
                        help="duration of the averaged steady-state window (s) -- long enough to "
                             "average out Stanley's own steer-correction oscillation in r")
    parser.add_argument("--max-duration", type=float, default=20.0, help="per-speed safety cutoff (s)")

    parser.add_argument("--out", default=os.path.join(HERE, "cornering_stiffness.json"))

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"))
    parser.add_argument("--save-plot", action="store_true", help="draw and save an alpha-vs-Fy fit figure")
    args = parser.parse_args()

    speeds = [float(v) for v in args.speeds.split(",")]

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    current_map = world.get_map().name.split("/")[-1]
    if current_map != MAP_NAME:
        print(f"Loading {MAP_NAME} (current map: {current_map})...")
        world = client.load_world(MAP_NAME)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    path_x, path_y, path_yaw = build_ring_path(world.get_map())
    print(f"Ring path: {len(path_x)} points, road_id={RING_ROAD_ID} lane_id={RING_LANE_ID}, "
          f"start=({path_x[0]:.1f}, {path_y[0]:.1f})")

    spawn_location = carla.Location(path_x[0], path_y[0], 0.0)
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(spawn_location) < 10.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    probe_transform = carla.Transform(carla.Location(path_x[0], path_y[0], 0.3),
                                      carla.Rotation(yaw=math.degrees(path_yaw[0])))
    probe = world.spawn_actor(blueprint, probe_transform)
    wheelbase, lf, lr, max_steer = get_vehicle_geometry(probe, probe_transform)
    mass = probe.get_physics_control().mass
    probe.destroy()
    print(f"mass={mass:.1f} kg  wheelbase={wheelbase:.2f} m  lf={lf:.2f} m  lr={lr:.2f} m")

    L = lf + lr

    def alpha_Fy(v_x, v_y, r, delta):
        beta = v_y / v_x
        a_f = delta - beta - lf * r / v_x
        a_r = -beta + lr * r / v_x
        return a_f, a_r, (lr / L) * mass * v_x * r, (lf / L) * mass * v_x * r

    alpha_f, alpha_r, Fyf, Fyr = [], [], [], []   # every tick of every window -- see run_trial()'s
                                                   # docstring for why this isn't averaged first
    per_speed = []
    try:
        for target_speed in speeds:
            print(f"\n=== target speed {target_speed:.1f} m/s ===")
            window = run_trial(world, path_x, path_y, path_yaw, blueprint, imu_bp, max_steer,
                              target_speed, args)
            if window is None:
                print(f"  did not settle within {args.max_duration:.0f}s -- skipped")
                continue

            tick_Cf, tick_Cr = [], []
            for v_x, v_y, r, delta in zip(window["v_x"], window["v_y"], window["r"], window["delta"]):
                a_f, a_r, fyf, fyr = alpha_Fy(v_x, v_y, r, delta)
                alpha_f.append(a_f); alpha_r.append(a_r); Fyf.append(fyf); Fyr.append(fyr)
                tick_Cf.append(fyf / a_f); tick_Cr.append(fyr / a_r)

            v_x_mean = float(np.mean(window["v_x"]))
            r_mean = float(np.mean(window["r"]))
            print(f"  settled: v_x={v_x_mean:.2f} m/s  r={math.degrees(r_mean):+.2f} deg/s  "
                  f"(within-window spread: Cf {np.std(tick_Cf):,.0f}, Cr {np.std(tick_Cr):,.0f} N/rad)")
            per_speed.append({"v_x": v_x_mean, "r": r_mean, "n_ticks": len(window["v_x"]),
                             "Cf_mean": float(np.mean(tick_Cf)), "Cf_std": float(np.std(tick_Cf)),
                             "Cr_mean": float(np.mean(tick_Cr)), "Cr_std": float(np.std(tick_Cr))})
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("\nCleaned up: world settings restored.")

    if len(per_speed) < 2:
        raise SystemExit(f"only {len(per_speed)} speed(s) settled -- need at least 2 to fit Cf/Cr")

    print(f"\n{'v_x':>6} {'Cf_mean':>10} {'Cf_std':>9} {'Cr_mean':>10} {'Cr_std':>9}")
    for p in per_speed:
        print(f"{p['v_x']:6.2f} {p['Cf_mean']:10,.0f} {p['Cf_std']:9,.0f} "
              f"{p['Cr_mean']:10,.0f} {p['Cr_std']:9,.0f}")

    Cf = fit_stiffness(alpha_f, Fyf)
    Cr = fit_stiffness(alpha_r, Fyr)
    print(f"\nCf = {Cf:,.0f} N/rad   Cr = {Cr:,.0f} N/rad  (least squares over {len(alpha_f)} ticks total)")

    out = {"Cf": Cf, "Cr": Cr, "mass": mass, "wheelbase": wheelbase, "lf": lf, "lr": lr,
          "per_speed": per_speed}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.out}")

    if args.save_plot:
        from viz_utils import COLOR_AXIS, COLOR_BLUE, COLOR_ORANGE, _legend, _save, _style_axes, COLOR_BG
        import matplotlib.pyplot as plt

        fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        fig.patch.set_facecolor(COLOR_BG)
        for ax, alpha, Fy, C, name, color in (
            (ax_f, alpha_f, Fyf, Cf, "front", COLOR_BLUE),
            (ax_r, alpha_r, Fyr, Cr, "rear", COLOR_ORANGE),
        ):
            _style_axes(ax)
            ax.axhline(0.0, color=COLOR_AXIS, linewidth=1)
            ax.axvline(0.0, color=COLOR_AXIS, linewidth=1)
            ax.scatter(alpha, Fy, color=color, zorder=3, label="measured")
            xs = np.linspace(min(alpha + [0]), max(alpha + [0]), 20)
            ax.plot(xs, C * xs, color=color, linestyle="--", label=f"C={C:,.0f} N/rad")
            ax.set_xlabel("slip angle (rad)")
            ax.set_ylabel("tire lateral force (N)")
            ax.set_title(f"{name} axle")
            _legend(ax)

        os.makedirs(args.plot_dir, exist_ok=True)
        out_path = _save(fig, args.plot_dir, "cornering_stiffness_fit")
        print(f"Figure saved: {out_path}")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
