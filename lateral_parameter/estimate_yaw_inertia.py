r"""Yaw moment of inertia (Iz) estimation via step-steer transient response.

Loads Cf, Cr (and m, lf, lr) from estimate_cornering_stiffness.py's output
(lateral_parameter/cornering_stiffness.json). At steady state Iz drops out of the yaw moment
balance entirely (r_dot = 0) -- exactly why that script could solve for Cf/Cr without knowing Iz.
That also means Iz can only be identified from a transient, where r_dot != 0:

    Iz * r_dot = lf*Fyf - lr*Fyr = lf*Cf*alpha_f - lr*Cr*alpha_r

With Cf, Cr already known, the right-hand side (call it M) is computable at every tick straight
from (v_x, v_y, r, delta), so this is just a 1-parameter least-squares fit:

    Iz = sum(M * r_dot) / sum(r_dot^2)

Maneuver: cruise straight at a target speed under a PID, then apply a constant open-loop step in
steering angle and log the whole transient as r rises from ~0 to its new steady value. The rise
itself is where r_dot is large -- exactly the excitation this fit needs (unlike Cf/Cr, which
needed several different steady speeds, one well-logged transient already spans a wide range of
r_dot on its own). Steer magnitude is kept in the same ~8-10 deg range
estimate_cornering_stiffness.py used, so alpha_f/alpha_r stay inside the region Cf/Cr were actually
calibrated over -- this script isn't re-deriving a tire model, just borrowing the linear one
already fit.

Run on Town06's long straight (spawn index 86, same spot longitudinal_PID.py etc. use) -- a
4-lane-wide highway stretch with plenty of room for the car to drift sideways during the transient
without leaving the pavement, and it starts straight (no ring-path setup needed the way the
cornering-stiffness test did).

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python lateral_parameter/estimate_yaw_inertia.py --save-plot

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py --speeds 5,6,7,8 --steer-deg 9 --save-plot
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

from functions import PID, CollisionWatch, LowPassFilter, clipping, get_vehicle_geometry

MAP_NAME = "Town06"
ORIGIN_INDEX = 86


def run_trial(world, origin_transform, blueprint, imu_bp, max_steer, steer_cmd, target_speed, args):
    """Cruise straight to target_speed, hold it briefly, then step the steering to `steer_cmd` and
    log the transient for args.transient_time seconds.

    Returns the transient window dict {"v_x": [...], "v_y": [...], "r": [...], "r_dot": [...],
    "delta": [...]}, one entry per logged tick, or None if the car never reached cruise speed or
    hit something.
    """
    vehicle = world.spawn_actor(blueprint, origin_transform)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    collision = CollisionWatch(world, vehicle)
    collision.arm()

    speed_pid = PID(kp=0.5, ki=0.2, kd=0.0, dt=args.dt)
    r_dot_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    prev_r = None

    cruise_ticks = int(args.cruise_time / args.dt)
    transient_ticks = int(args.transient_time / args.dt)
    cruise_settled_ticks = 0
    stepped = False
    window = {"v_x": [], "v_y": [], "r": [], "r_dot": [], "delta": []}

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
            vel = vehicle.get_velocity()
            v_x = vel.x * math.cos(yaw) + vel.y * math.sin(yaw)
            v_y = -vel.x * math.sin(yaw) + vel.y * math.cos(yaw)
            r = imu_data.gyroscope.z   # rad/s, body-frame yaw rate straight off the sensor

            r_dot = r_dot_filter.step(0.0 if prev_r is None else (r - prev_r) / args.dt)
            prev_r = r

            u = clipping(speed_pid.step(target_speed - v_x), 1.0, -1.0)
            control = carla.VehicleControl()
            control.throttle, control.brake = (u, 0.0) if u >= 0 else (0.0, -u)
            control.steer = steer_cmd if stepped else 0.0
            vehicle.apply_control(control)

            if collision.hit:
                print(f"    collision at t={i*args.dt:.1f}s -- aborting this trial")
                return None

            if not stepped:
                settled = abs(v_x - target_speed) < args.speed_tol and abs(r_dot) < 0.05
                cruise_settled_ticks = cruise_settled_ticks + 1 if settled else 0
                if i % 20 == 0:
                    print(f"    t={i*args.dt:5.1f}s  v_x={v_x:5.2f}/{target_speed:.1f} m/s  "
                          f"cruise_settled_ticks={cruise_settled_ticks}/{cruise_ticks}")
                if cruise_settled_ticks >= cruise_ticks:
                    stepped = True
                    print(f"    STEP at t={i*args.dt:.1f}s: steer -> {math.degrees(steer_cmd * max_steer):+.1f} deg")
            else:
                window["v_x"].append(v_x)
                window["v_y"].append(v_y)
                window["r"].append(r)
                window["r_dot"].append(r_dot)
                window["delta"].append(steer_cmd * max_steer)
                if len(window["v_x"]) >= transient_ticks:
                    return window

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    finally:
        collision.destroy()
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        vehicle.destroy()

    return None   # never reached cruise speed within max_duration


def fit_inertia(M, r_dot):
    """Least-squares Iz through the origin: min_Iz sum((Iz*r_dot - M)^2)."""
    M = np.asarray(M)
    r_dot = np.asarray(r_dot)
    return float(np.sum(M * r_dot) / np.sum(r_dot * r_dot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--times-run", type=float, default=2.0, help="how times for simulation running?")

    parser.add_argument("--cf-cr-file", default=os.path.join(HERE, "cornering_stiffness.json"),
                        help="output of estimate_cornering_stiffness.py")
    parser.add_argument("--speeds", default="5,6,7,8", help="comma-separated cruise speeds to step from (m/s)")
    parser.add_argument("--steer-deg", type=float, default=9.0,
                        help="step steer magnitude (deg) -- kept in the range Cf/Cr were "
                             "calibrated over, since this script reuses that linear tire fit "
                             "rather than re-deriving one")
    parser.add_argument("--speed-tol", type=float, default=0.2, help="cruise counts as settled once |v_x - target| is under this (m/s)")
    parser.add_argument("--cruise-time", type=float, default=2.0, help="how long cruise must hold settled before the step (s)")
    parser.add_argument("--transient-time", type=float, default=2.5,
                        help="how long to log after the step -- long enough to capture the rise "
                             "and settle into the new steady turn")
    parser.add_argument("--max-duration", type=float, default=20.0, help="per-speed safety cutoff (s)")

    parser.add_argument("--out", default=os.path.join(HERE, "yaw_inertia.json"))

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"))
    parser.add_argument("--save-plot", action="store_true", help="draw and save the r(t) transient and M-vs-r_dot fit")
    args = parser.parse_args()

    with open(args.cf_cr_file) as f:
        tire = json.load(f)
    Cf, Cr, mass, lf, lr = tire["Cf"], tire["Cr"], tire["mass"], tire["lf"], tire["lr"]
    print(f"Loaded from {args.cf_cr_file}: Cf={Cf:,.0f} N/rad  Cr={Cr:,.0f} N/rad  "
          f"mass={mass:.1f} kg  lf={lf:.2f} m  lr={lr:.2f} m")

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

    origin_transform = world.get_map().get_spawn_points()[ORIGIN_INDEX]

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    probe = world.spawn_actor(blueprint, origin_transform)
    wheelbase, lf_probe, lr_probe, max_steer = get_vehicle_geometry(probe, origin_transform)
    probe.destroy()
    if abs(lf_probe - lf) > 0.05 or abs(lr_probe - lr) > 0.05:
        print(f"warning: this vehicle's lf/lr ({lf_probe:.2f}/{lr_probe:.2f}) differ from the "
              f"cornering-stiffness file's ({lf:.2f}/{lr:.2f}) -- Cf/Cr may not transfer cleanly")

    steer_rad = math.radians(args.steer_deg)
    steer_cmd = clipping(steer_rad / max_steer, 1.0, -1.0)

    L = lf + lr
    M_all, r_dot_all = [], []
    per_speed = []
    try:
        for target_speed in speeds:
            print(f"\n=== cruise speed {target_speed:.1f} m/s ===")
            window = run_trial(world, origin_transform, blueprint, imu_bp, max_steer, steer_cmd,
                              target_speed, args)
            if window is None:
                print(f"  did not settle within {args.max_duration:.0f}s -- skipped")
                continue

            trial_M, trial_r_dot = [], []
            for v_x, v_y, r, r_dot, delta in zip(window["v_x"], window["v_y"], window["r"],
                                                 window["r_dot"], window["delta"]):
                beta = v_y / v_x
                alpha_f = delta - beta - lf * r / v_x
                alpha_r = -beta + lr * r / v_x
                Fyf = Cf * alpha_f
                Fyr = Cr * alpha_r
                M = lf * Fyf - lr * Fyr
                trial_M.append(M)
                trial_r_dot.append(r_dot)
            M_all.extend(trial_M)
            r_dot_all.extend(trial_r_dot)

            r_peak = max(window["r"], key=abs)
            print(f"  transient logged: {len(window['v_x'])} ticks, "
                  f"r_dot range [{min(trial_r_dot):+.2f}, {max(trial_r_dot):+.2f}] rad/s^2, "
                  f"peak r={math.degrees(r_peak):+.2f} deg/s")
            per_speed.append({"target_speed": target_speed, "n_ticks": len(window["v_x"]),
                             "Iz_trial": fit_inertia(trial_M, trial_r_dot)})
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("\nCleaned up: world settings restored.")

    if len(M_all) < 10:
        raise SystemExit(f"only {len(M_all)} ticks logged -- need more transient data to fit Iz")

    print(f"\n{'speed':>6} {'n_ticks':>8} {'Iz_trial':>12}")
    for p in per_speed:
        print(f"{p['target_speed']:6.1f} {p['n_ticks']:8d} {p['Iz_trial']:12,.0f}")

    Iz = fit_inertia(M_all, r_dot_all)
    print(f"\nIz = {Iz:,.0f} kg*m^2  (least squares over {len(M_all)} ticks total)")

    out = {"Iz": Iz, "Cf": Cf, "Cr": Cr, "mass": mass, "lf": lf, "lr": lr, "per_speed": per_speed}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.out}")

    if args.save_plot:
        from viz_utils import COLOR_AXIS, COLOR_BLUE, _legend, _save, _style_axes, COLOR_BG
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        fig.patch.set_facecolor(COLOR_BG)
        _style_axes(ax)
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=1)
        ax.axvline(0.0, color=COLOR_AXIS, linewidth=1)
        ax.scatter(r_dot_all, M_all, color=COLOR_BLUE, s=8, alpha=0.4, zorder=3, label="measured")
        xs = np.linspace(min(r_dot_all + [0]), max(r_dot_all + [0]), 20)
        ax.plot(xs, Iz * xs, color=COLOR_BLUE, linestyle="--", label=f"Iz={Iz:,.0f} kg*m^2")
        ax.set_xlabel("yaw acceleration r_dot (rad/s^2)")
        ax.set_ylabel("net yaw moment M = lf*Fyf - lr*Fyr (N*m)")
        ax.set_title("Yaw inertia fit: M = Iz * r_dot")
        _legend(ax)

        os.makedirs(args.plot_dir, exist_ok=True)
        out_path = _save(fig, args.plot_dir, "yaw_inertia_fit")
        print(f"Figure saved: {out_path}")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
