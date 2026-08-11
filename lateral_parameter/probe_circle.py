r"""Look at a constant-steer, constant-speed circle before trusting any fit to it.

This measures nothing. It drives one circle and draws what happened, so the assumptions the
cornering-stiffness fit rests on can be checked by eye rather than assumed:

  - Does a fixed steer command actually produce a fixed-radius circle, and which radius?
  - Is the motion in steady state? A steady circle needs v_x_dot, v_y_dot and psi_ddot all at
    zero; the fit's force split is only valid where they are.
  - How far does the commanded steer sit from the wheels' real angle, and do the left and right
    wheels agree? (They cannot -- Ackermann makes them differ -- so the bicycle model's single
    virtual wheel is their mean, and it is worth seeing how big that spread is.)
  - Does the speed controller hold the target while the car is cornering?

Figures produced:
  1-3. viz_utils.plot_results -- trajectory, lateral tracking, longitudinal tracking
  4.   steering: command vs measured front-left / front-right / mean, and what the steering curve
       predicts the command should have become
  5.   steady state: v_x_dot, v_y_dot, psi_ddot, speed tracking, instantaneous radius, and a_y
       measured three independent ways

The sites are the two flat patches of tarmac found by scanning the maps -- see
bicycle.CIRCLE_SITES.

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python lateral_parameter/probe_circle.py --site town06 --radius 10 --speed 5
    .venv/bin/python lateral_parameter/probe_circle.py --site town03 --radius 20 --speed 5

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe lateral_parameter\probe_circle.py --site town06 --radius 10 --speed 5
    .venv\Scripts\python.exe lateral_parameter\probe_circle.py --site town03 --radius 20 --speed 5
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
sys.path.append(os.path.dirname(HERE))

# functions.py resolves CARLA_ROOT and puts the simulator's PythonAPI on sys.path
from functions import PID, CollisionWatch, clipping, get_vehicle_geometry, lateral_error, normalize_angle

import carla

from viz_utils import follow_with_spectator

from bicycle import (CIRCLE_SITES, circle_path, front_steer_angle, steering_curve_scale,
                     zero_phase_derivative)


def run(world, site, args):
    """Drive one constant-steer circle and return the per-tick history."""
    physics_probe = None
    centre = site["centre"]
    path_x, path_y, path_yaw = circle_path(centre, args.radius, step=2.0)

    lo, hi = site["radius_range"]
    if not lo <= args.radius <= hi:
        print(f"WARNING: R={args.radius} m is outside this site's drivable band {lo}-{hi} m")

    spawn = carla.Transform(
        carla.Location(path_x[0], path_y[0], site["z"] + 0.3),
        carla.Rotation(yaw=math.degrees(path_yaw[0])),
    )
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(spawn.location) < 15.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter(args.vehicle)[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    vehicle = world.spawn_actor(blueprint, spawn)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    collision = CollisionWatch(world, vehicle)
    collision.arm()

    physics = vehicle.get_physics_control()
    wheelbase, lf, lr, max_steer = get_vehicle_geometry(vehicle, spawn)
    mass = physics.mass
    com = physics.center_of_mass

    # Aim the command so the *wheels* end up at the kinematic angle for this radius once the car
    # is at speed. atan(L/R) rather than L/R because at small radii they differ by several
    # percent, and dividing by the steering-curve scale undoes the reduction CARLA will apply.
    target_wheel_angle = math.atan(wheelbase / args.radius)
    scale = steering_curve_scale(physics, args.speed)
    steer_cmd = clipping(target_wheel_angle / (max_steer * scale), 1.0, -1.0)
    if args.steer_deg is not None:
        steer_cmd = clipping(math.radians(args.steer_deg) / max_steer, 1.0, -1.0)
    steer_cmd *= -1.0 if args.direction < 0 else 1.0

    print(f"mass={mass:.1f} kg  L={wheelbase:.3f} m  lf={lf:.3f}  lr={lr:.3f}  "
          f"max_steer={math.degrees(max_steer):.1f} deg")
    print(f"centre_of_mass=({com.x:.2f}, {com.y:.2f}, {com.z:.2f}) m -- IMU is mounted here, not "
          f"at the actor origin, so a_y needs no r_dot*offset correction")
    print(f"target R={args.radius:.1f} m -> kinematic wheel angle atan(L/R)="
          f"{math.degrees(target_wheel_angle):.2f} deg")
    print(f"steering_curve scale at {args.speed:.1f} m/s = {scale:.3f}  ->  steer command "
          f"{steer_cmd:+.4f} (= {math.degrees(steer_cmd*max_steer):+.2f} deg before the curve)")

    speed_pid = PID(kp=0.5, ki=0.2, kd=0.0, dt=args.dt)
    hist = {k: [] for k in (
        "t", "x", "y", "v_x", "v_y", "v_des", "a_x", "jerk", "a_y", "yaw_rate", "yaw_acc",
        "jerk_total", "last_idx", "steer_deg", "throttle", "brake", "e_y", "yaw", "path_yaw",
        "e_theta",
        # extras this script draws itself
        "steer_cmd_deg", "delta_fl", "delta_fr", "delta_mean", "curve_scale",
        "a_y_imu", "a_y_corr", "a_y_kin", "a_x_imu", "radius_inst", "roll", "pitch")}

    # The IMU rides at the centre of mass: an accelerometer offset from the CG by d reads
    # a_y + r_dot*d_x, which at r_dot ~ 0.3 rad/s^2 and d_x = 0.3 m is 0.09 m/s^2 -- over 10% of
    # a_y at the low end. Mounting it correctly is cheaper than correcting for it.
    imu = None
    last_idx = 0
    prev_a_x = prev_a_y_total = None
    try:
        world.tick()
        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(carla.Location(com.x, com.y, com.z)),
                                attach_to=vehicle)
        imu.listen(imu_queue.put)

        for i in range(int(args.duration / args.dt)):
            step_start = time.time()
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)

            transform = vehicle.get_transform()
            yaw = math.radians(transform.rotation.yaw)
            roll = math.radians(transform.rotation.roll)
            pitch = math.radians(transform.rotation.pitch)
            ego_x, ego_y = transform.location.x, transform.location.y
            vel = vehicle.get_velocity()
            v_x = vel.x * math.cos(yaw) + vel.y * math.sin(yaw)
            v_y = -vel.x * math.sin(yaw) + vel.y * math.cos(yaw)
            r = imu_data.gyroscope.z
            a_x_imu = imu_data.accelerometer.x
            a_y_imu = imu_data.accelerometer.y

            fl = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
            fr = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
            delta_mean = front_steer_angle(vehicle)

            last_idx, e_y = lateral_error(ego_x, ego_y, yaw, path_x, path_y, last_idx)
            e_theta = normalize_angle(path_yaw[last_idx] - yaw)

            u = clipping(speed_pid.step(args.speed - v_x), 1.0, -1.0)
            control = carla.VehicleControl()
            control.throttle, control.brake = (u, 0.0) if u >= 0 else (0.0, -u)
            control.steer = 0.0 if i * args.dt < args.straight_time else steer_cmd
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            if collision.hit:
                print(f"  collision at t={i*args.dt:.1f}s -- stopping (the log up to here is kept)")
                break

            jerk = 0.0 if prev_a_x is None else (a_x_imu - prev_a_x) / args.dt
            a_total = math.hypot(a_x_imu, a_y_imu)
            jerk_total = 0.0 if prev_a_y_total is None else (a_total - prev_a_y_total) / args.dt
            prev_a_x, prev_a_y_total = a_x_imu, a_total

            hist["t"].append(i * args.dt)
            hist["x"].append(ego_x); hist["y"].append(ego_y)
            hist["v_x"].append(v_x); hist["v_y"].append(v_y); hist["v_des"].append(args.speed)
            hist["a_x"].append(a_x_imu); hist["jerk"].append(jerk); hist["jerk_total"].append(jerk_total)
            hist["a_y"].append(a_y_imu * math.cos(roll) + 9.81 * math.sin(roll))
            hist["yaw_rate"].append(math.degrees(r))
            hist["yaw_acc"].append(0.0)            # filled in after the run, zero-phase
            hist["last_idx"].append(last_idx)
            hist["steer_deg"].append(math.degrees(delta_mean))
            hist["throttle"].append(control.throttle); hist["brake"].append(control.brake)
            hist["e_y"].append(e_y)
            hist["yaw"].append(math.degrees(yaw))
            hist["path_yaw"].append(math.degrees(path_yaw[last_idx]))
            hist["e_theta"].append(math.degrees(e_theta))

            hist["steer_cmd_deg"].append(math.degrees(control.steer * max_steer))
            hist["delta_fl"].append(fl); hist["delta_fr"].append(fr)
            hist["delta_mean"].append(math.degrees(delta_mean))
            hist["curve_scale"].append(steering_curve_scale(physics, max(v_x, 0.0)))
            hist["a_y_imu"].append(a_y_imu)
            # An accelerometer measures specific force in the BODY frame, and body roll tilts that
            # frame so gravity leaks into its y axis. Verified on this rig: raw IMU and the
            # kinematic v_x*psi_dot disagree by a persistent offset that matches g*sin(roll) in
            # both size and sign, and the corrected value agrees with the kinematic one to 0.2%.
            # Without this the planar force balance picks up a bias proportional to roll, hence
            # roughly proportional to a_y itself -- a multiplicative error on Cf and Cr.
            hist["a_y_corr"].append(a_y_imu * math.cos(roll) + 9.81 * math.sin(roll))
            hist["a_y_kin"].append(v_x * r)
            hist["a_x_imu"].append(a_x_imu)
            hist["radius_inst"].append(v_x / r if abs(r) > 1e-4 else float("nan"))
            hist["roll"].append(roll); hist["pitch"].append(pitch)

            if i % int(1.0 / args.dt) == 0:
                print(f"  t={i*args.dt:5.1f}s  v_x={v_x:5.2f}/{args.speed:.1f}  "
                      f"r={math.degrees(r):+7.2f} deg/s  R={v_x/r if abs(r)>1e-4 else float('nan'):8.2f} m  "
                      f"delta={math.degrees(delta_mean):+6.2f} deg  e_y={e_y:+6.2f} m  "
                      f"a_y={a_y_imu:+6.2f}")

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    finally:
        collision.destroy()
        if imu is not None and imu.is_alive:
            imu.stop(); imu.destroy()
        vehicle.destroy()

    # CARLA's accelerometer emits garbage on the first tick or two after spawn (measured:
    # -378,000 m/s^2 on tick 1, then clean). Two samples are enough to wreck every mean and
    # every plot's y-scale, so the head of the log is dropped before anything reads it.
    if args.skip_ticks and len(hist["t"]) > args.skip_ticks:
        for key in list(hist):
            hist[key] = hist[key][args.skip_ticks:]
        t0 = hist["t"][0]
        hist["t"] = [t - t0 for t in hist["t"]]

    # Filter the accelerometer for display and for instantaneous checks -- but keep the raw
    # series, because the two are for different jobs. Measured on this rig, 97% of the IMU's
    # variance sits above 6 Hz (82% above 12 Hz, near Nyquist at a 50 Hz tick) while the signal
    # itself is DC: raw std is 5.0 m/s^2 against a v_x*psi_dot reference whose std is 0.099. Yet
    # the raw MEAN is already accurate to -0.07% of that reference, because the noise is
    # zero-mean and averages out. So anything that fits over a window should use the raw series;
    # filtering only helps when a single tick's value has to be read or plotted, and filtfilt's
    # edge handling actually shifts the mean slightly (+0.56% at 1 Hz).
    #
    # The filter is zero-phase for the same reason the derivatives are: the whole point of
    # comparing a_y against v_x*psi_dot is that they line up in time, and a causal filter would
    # manufacture a residual out of its own lag.
    if args.imu_lpf and len(hist["t"]) > 20:
        from scipy.signal import butter, filtfilt
        b, a = butter(2, args.imu_lpf / (0.5 / args.dt))
        for key in ("a_y_imu", "a_y_corr", "a_x_imu"):
            hist[key + "_raw"] = list(hist[key])
            hist[key] = filtfilt(b, a, np.array(hist[key])).tolist()
        hist["a_y"] = list(hist["a_y_corr"])

    # Derivatives are computed here, zero-phase, rather than causally during the run: a causal
    # filter would delay them relative to the motion that produced them, which matters when the
    # whole point is to say whether they are zero at the same instants.
    dt = args.dt
    hist["v_x_dot"] = zero_phase_derivative(np.array(hist["v_x"]), dt, window_s=args.deriv_window).tolist()
    hist["v_y_dot"] = zero_phase_derivative(np.array(hist["v_y"]), dt, window_s=args.deriv_window).tolist()
    psi_ddot = zero_phase_derivative(np.radians(np.array(hist["yaw_rate"])), dt,
                                window_s=args.deriv_window)
    hist["psi_ddot"] = psi_ddot.tolist()
    hist["yaw_acc"] = psi_ddot.tolist()
    hist["a_y_dyn"] = (np.array(hist["v_y_dot"]) + np.array(hist["a_y_kin"])).tolist()
    return hist, (path_x, path_y, path_yaw), dict(max_steer=max_steer, wheelbase=wheelbase,
                                                  lf=lf, lr=lr, mass=mass, steer_cmd=steer_cmd)


def plot_steering(hist, args, out_dir):
    """Commanded steer vs what the wheels actually did."""
    from viz_utils import (COLOR_AQUA, COLOR_BLUE, COLOR_MUTED, COLOR_ORANGE, COLOR_PURPLE, COLOR_RED,
                           _legend, _panels, _save)
    import matplotlib.pyplot as plt

    fig, (ax_ang, ax_split) = _panels(
        f"Steering: command vs wheels — R={args.radius:.1f} m, {args.speed:.1f} m/s",
        n_rows=2, n_cols=1, figsize=(13, 8))
    t = hist["t"]

    ax_ang.plot(t, hist["steer_cmd_deg"], color=COLOR_MUTED, linewidth=2, linestyle="--",
                label="command x max_steer (what a naive model assumes)")
    predicted = np.array(hist["steer_cmd_deg"]) * np.array(hist["curve_scale"])
    ax_ang.plot(t, predicted, color=COLOR_RED, linewidth=1.6, linestyle=":",
                label="command x steering_curve(v)")
    ax_ang.plot(t, hist["delta_fl"], color=COLOR_BLUE, linewidth=1.4, label="front-left wheel")
    ax_ang.plot(t, hist["delta_fr"], color=COLOR_AQUA, linewidth=1.4, label="front-right wheel")
    ax_ang.plot(t, hist["delta_mean"], color=COLOR_ORANGE, linewidth=2.2,
                label="mean (the bicycle model's delta)")
    ax_ang.set_ylabel("steer angle (deg)")
    ax_ang.set_title("A constant command is not a constant wheel angle")
    _legend(ax_ang, ncol=2)

    ax_split.plot(t, np.array(hist["delta_fl"]) - np.array(hist["delta_fr"]),
                  color=COLOR_PURPLE, linewidth=1.6, label="left - right (Ackermann spread)")
    residual = np.array(hist["delta_mean"]) - predicted
    ax_split.plot(t, residual, color=COLOR_RED, linewidth=1.6,
                  label="measured mean - curve prediction")
    ax_split.axhline(0.0, color=COLOR_MUTED, linewidth=1)
    ax_split.set_ylabel("angle difference (deg)")
    ax_split.set_xlabel("t (s)")
    ax_split.set_title("Left/right split, and what the steering curve does not explain")
    _legend(ax_split)

    path = _save(fig, out_dir, "probe_circle_steering")
    print(f"Figure saved: {path}")
    return fig


def plot_trajectory_detail(hist, site, args, out_dir):
    """The xy path, but readable for a circle test.

    viz_utils.plot_trajectory draws the driven line against the reference, which is the right
    picture for tracking a route and the wrong one here: several laps of a circle land on top of
    each other, so a single overlapping ring hides the spin-up, hides how round the circle
    actually is, and hides anything that happens at one place on it.

    Two views instead. Left, the path coloured lap by lap so they can be told apart. Right, the
    same data as radial deviation against angle -- which turns "is it a circle" into a flat line
    at zero, and makes a feature at one point on the ground show up as the same bump on every lap.
    """
    from viz_utils import (COLOR_AXIS, COLOR_BLUE, COLOR_INK, COLOR_MUTED, COLOR_RED, COMPARE_COLORS,
                           _legend, _save, _style_axes, COLOR_BG)
    import matplotlib.pyplot as plt

    cx, cy = site["centre"]
    x = np.array(hist["x"]); y = np.array(hist["y"]); t = np.array(hist["t"])
    radius = np.hypot(x - cx, y - cy)
    angle = np.degrees(np.unwrap(np.arctan2(y - cy, x - cx)))
    lap = np.floor((angle - angle[0]) / -360.0).astype(int) if args.direction < 0 else \
        np.floor((angle - angle[0]) / 360.0).astype(int)

    fig, (ax_xy, ax_dev) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(f"Trajectory — {args.site}, target R={args.radius:.1f} m, {args.speed:.1f} m/s",
                 fontsize=14, color=COLOR_INK)
    _style_axes(ax_xy); _style_axes(ax_dev)

    theta = np.linspace(0, 2 * math.pi, 361)
    ax_xy.plot(cx + args.radius * np.cos(theta), cy + args.radius * np.sin(theta),
               color=COLOR_MUTED, linewidth=2, linestyle="--", label=f"target R={args.radius:.1f} m")
    for k in range(lap.max() + 1):
        m = lap == k
        if m.sum() < 2:
            continue
        ax_xy.plot(x[m], y[m], color=COMPARE_COLORS[k % len(COMPARE_COLORS)], linewidth=1.6,
                   label=f"lap {k}")
    ax_xy.scatter([x[0]], [y[0]], color=COLOR_BLUE, zorder=5, s=60, label="start")
    ax_xy.scatter([cx], [cy], color=COLOR_MUTED, marker="+", s=120, zorder=5, label="centre")
    ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("Driven path, one colour per lap")
    ax_xy.set_aspect("equal", adjustable="datalim")

    # Measured against the *reference* circle's centre, an offset circle looks like a large clean
    # sinusoid -- which says nothing about roundness, only that the car settled somewhere else.
    # It has to: a constant steer picks its own radius, so unless that happens to equal the
    # target the two circles cannot be concentric. Fitting a circle to the driven path and
    # measuring against that separates "is it round" from "is it where I aimed".
    settled = t > t[-1] * 0.4
    fx, fy = x[settled], y[settled]
    A = np.c_[2 * fx, 2 * fy, np.ones(fx.size)]
    sol, *_ = np.linalg.lstsq(A, fx ** 2 + fy ** 2, rcond=None)
    fit_cx, fit_cy = sol[0], sol[1]
    fit_R = math.sqrt(sol[2] + fit_cx ** 2 + fit_cy ** 2)
    offset = math.hypot(fit_cx - cx, fit_cy - cy)

    fit_radius = np.hypot(x - fit_cx, y - fit_cy)
    fit_angle = np.degrees(np.arctan2(y - fit_cy, x - fit_cx)) % 360.0
    for k in range(lap.max() + 1):
        m = (lap == k) & settled
        if m.sum() < 2:
            continue
        ax_dev.plot(fit_angle[m], (fit_radius[m] - fit_R) * 100.0, ".", markersize=2,
                    color=COMPARE_COLORS[k % len(COMPARE_COLORS)], label=f"lap {k}")
    ax_dev.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_dev.set_xlabel("angle around the fitted circle (deg)")
    ax_dev.set_ylabel("deviation from the fitted circle (cm)")
    ax_dev.set_title(f"Roundness: fitted R={fit_R:.3f} m, centre {offset:.2f} m from the target's")
    ax_dev.set_xlim(0, 360)
    _legend(ax_dev, ncol=2)

    print(f"  driven circle: R={fit_R:.3f} m (target {args.radius:.1f}, "
          f"{100*(fit_R/args.radius-1):+.1f}%), centre offset {offset:.2f} m, "
          f"roundness +/-{np.abs(fit_radius[settled]-fit_R).max()*100:.1f} cm")

    ax_xy.plot(fit_cx + fit_R * np.cos(theta), fit_cy + fit_R * np.sin(theta),
               color=COLOR_RED, linewidth=1.0, linestyle=":", label=f"fitted R={fit_R:.2f} m")
    _legend(ax_xy, ncol=2)

    path = _save(fig, out_dir, "probe_circle_trajectory")
    print(f"Figure saved: {path}")
    return fig


def _robust_ylim(ax, series, pad=0.2):
    """Scale to the bulk of the data, not to its outliers.

    A single spike -- CARLA's accelerometer emits them, and so does driving over a seam --
    otherwise compresses the entire trace into a flat line at this figure's scale.
    """
    v = np.asarray(series, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    lo, hi = np.percentile(v, [1, 99])
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    margin = (hi - lo) * pad
    ax.set_ylim(lo - margin, hi + margin)


def plot_steady_state(hist, args, out_dir):
    """The three derivatives that must vanish in steady state, plus speed and radius."""
    from viz_utils import (COLOR_AQUA, COLOR_AXIS, COLOR_BLUE, COLOR_MUTED, COLOR_ORANGE,
                           COLOR_RED, _legend, _panels, _save)

    fig, axes = _panels(
        f"Steady-state evidence — R={args.radius:.1f} m, {args.speed:.1f} m/s",
        n_rows=4, n_cols=2, figsize=(15, 14))
    ax_v, ax_vxd, ax_vy, ax_psi, ax_R, ax_roll, ax_ay, ax_res = axes
    t = hist["t"]

    ax_v.plot(t, hist["v_x"], color=COLOR_BLUE, linewidth=1.8, label="$v_x$")
    ax_v.axhline(args.speed, color=COLOR_MUTED, linewidth=2, linestyle="--", label="target")
    ax_v.set_ylabel("$v_x$ (m/s)"); ax_v.set_title("Speed tracking")
    _legend(ax_v)

    ax_vxd.plot(t, hist["v_x_dot"], color=COLOR_BLUE, linewidth=1.6, label=r"$\dot{v}_x$")
    ax_vxd.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_vxd.set_ylabel(r"$\dot{v}_x$ (m/s$^2$)")
    ax_vxd.set_title("Longitudinal acceleration (0 in steady state)")
    _legend(ax_vxd)

    ax_vy.plot(t, hist["v_y"], color=COLOR_ORANGE, linewidth=1.6, label="$v_y$")
    ax_vy.plot(t, hist["v_y_dot"], color=COLOR_RED, linewidth=1.6, label=r"$\dot{v}_y$")
    ax_vy.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_vy.set_ylabel("m/s, m/s$^2$")
    ax_vy.set_title(r"Lateral velocity and its rate ($\dot{v}_y$ = 0 in steady state)")
    _legend(ax_vy)

    ax_psi.plot(t, hist["yaw_rate"], color=COLOR_ORANGE, linewidth=1.6, label=r"$\dot{\psi}$ (deg/s)")
    ax_psi.plot(t, np.degrees(hist["psi_ddot"]), color=COLOR_RED, linewidth=1.6,
                label=r"$\ddot{\psi}$ (deg/s$^2$)")
    ax_psi.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_psi.set_ylabel("deg/s, deg/s$^2$")
    ax_psi.set_title(r"Yaw rate and yaw acceleration ($\ddot{\psi}$ = 0 in steady state)")
    _robust_ylim(ax_psi, np.concatenate([np.array(hist["yaw_rate"]),
                                         np.degrees(hist["psi_ddot"])]), pad=0.3)
    _legend(ax_psi)

    # |v_x/psi_dot|: the sign just says which way the car is going round
    R_inst = np.abs(np.array(hist["radius_inst"], dtype=float))
    ax_R.plot(t, R_inst, color=COLOR_BLUE, linewidth=1.6, label=r"$|v_x/\dot{\psi}|$")
    ax_R.axhline(args.radius, color=COLOR_MUTED, linewidth=2, linestyle="--", label="target R")
    ax_R.set_ylabel("radius (m)"); ax_R.set_xlabel("t (s)")
    ax_R.set_title("Instantaneous turn radius")
    _robust_ylim(ax_R, R_inst, pad=0.3)
    _legend(ax_R)

    roll_deg = np.degrees(hist["roll"])
    ax_roll.plot(t, roll_deg, color=COLOR_RED, linewidth=1.6, label="body roll")
    ax_roll.plot(t, np.degrees(hist["pitch"]), color=COLOR_MUTED, linewidth=1.2, label="pitch")
    ax_roll.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_roll.set_ylabel("angle (deg)"); ax_roll.set_xlabel("t (s)")
    ax_roll.set_title(r"Body attitude — roll tilts the IMU, leaking $g\sin\phi$ into $a_y$")
    _legend(ax_roll)

    if "a_y_corr_raw" in hist:
        ax_ay.plot(t, hist["a_y_corr_raw"], color=COLOR_MUTED, linewidth=0.6, alpha=0.5,
                   label="IMU unfiltered")
    ax_ay.plot(t, hist["a_y_corr"], color=COLOR_BLUE, linewidth=1.8,
               label=r"IMU: roll-corrected + zero-phase LPF")
    ax_ay.plot(t, hist["a_y_kin"], color=COLOR_ORANGE, linewidth=1.6, linestyle="--",
               label=r"$v_x\dot{\psi}$ (steady-state form)")
    ax_ay.plot(t, hist["a_y_dyn"], color=COLOR_AQUA, linewidth=1.4, linestyle=":",
               label=r"$\dot{v}_y + v_x\dot{\psi}$ (full)")
    ax_ay.set_ylabel("$a_y$ (m/s$^2$)"); ax_ay.set_xlabel("t (s)")
    ax_ay.set_title("Lateral acceleration, four ways")
    _robust_ylim(ax_ay, hist["a_y_kin"], pad=0.6)
    _legend(ax_ay, ncol=2)

    # In true steady state v_y_dot = 0, so the corrected IMU and the kinematic term must agree.
    # Their gap is therefore a direct, Iz-free measure of how far from steady state we are.
    residual = np.array(hist["a_y_corr"]) - np.array(hist["a_y_kin"])
    ax_res.plot(t, residual, color=COLOR_RED, linewidth=1.4,
                label=r"$a_y^{corr} - v_x\dot{\psi}$  ($=\dot{v}_y$)")
    ax_res.plot(t, hist["v_y_dot"], color=COLOR_BLUE, linewidth=1.4, linestyle="--",
                label=r"$\dot{v}_y$ (differentiated)")
    ax_res.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_res.set_ylabel("m/s$^2$"); ax_res.set_xlabel("t (s)")
    ax_res.set_title("Steady-state residual (two independent estimates of the same thing)")
    _robust_ylim(ax_res, np.concatenate([residual, np.array(hist["v_y_dot"])]), pad=0.4)
    _legend(ax_res)

    path = _save(fig, out_dir, "probe_circle_steady_state")
    print(f"Figure saved: {path}")
    return fig


def summarise(hist, args, geom):
    """Print the numbers the figures are meant to make obvious."""
    t = np.array(hist["t"])
    tail = t >= (t[-1] - args.summary_window)
    if tail.sum() < 5:
        print("\n(run too short to summarise a settled tail)")
        return

    def stat(key, scale=1.0):
        v = np.array(hist.get(key, hist["a_y_corr"]))[tail] * scale
        return v.mean(), v.std()

    print(f"\n--- last {args.summary_window:.1f} s ---")
    for label, key, unit, scale in (
        ("v_x", "v_x", "m/s", 1.0),
        ("v_x_dot", "v_x_dot", "m/s^2", 1.0),
        ("v_y", "v_y", "m/s", 1.0),
        ("v_y_dot", "v_y_dot", "m/s^2", 1.0),
        ("yaw rate", "yaw_rate", "deg/s", 1.0),
        ("psi_ddot", "psi_ddot", "deg/s^2", 180.0 / math.pi),
        ("radius", "radius_inst", "m", 1.0),
        ("a_y IMU filt", "a_y_corr", "m/s^2", 1.0),
        ("a_y IMU raw", "a_y_corr_raw", "m/s^2", 1.0),
        ("a_y kin", "a_y_kin", "m/s^2", 1.0),
        ("roll", "roll", "deg", 180.0 / math.pi),
        ("delta mean", "delta_mean", "deg", 1.0),
        ("e_y", "e_y", "m", 1.0),
    ):
        m, s = stat(key, scale)
        print(f"  {label:>11}: {m:+9.4f} +/- {s:7.4f} {unit}")

    R = np.array(hist["radius_inst"])[tail]
    print(f"\n  radius spread: {abs(R.std()/R.mean())*100:.2f}% of mean "
          f"(target {args.radius:.1f} m, achieved {abs(R.mean()):.2f} m, "
          f"{100*(abs(R.mean())/args.radius-1):+.1f}%)")
    gap = np.array(hist["a_y_corr"])[tail] - np.array(hist["a_y_kin"])[tail]
    print(f"  |a_y_imu - v_x*psi_dot|: mean {abs(gap.mean()):.4f} m/s^2 "
          f"({100*abs(gap.mean())/abs(np.array(hist['a_y_kin'])[tail].mean()):.1f}% of a_y) "
          f"-- this is the steady-state residual, and equals v_y_dot when the surface is flat")
    fl = np.array(hist["delta_fl"])[tail]; fr = np.array(hist["delta_fr"])[tail]
    print(f"  Ackermann spread (FL-FR): {(fl-fr).mean():+.3f} deg")
    cmd = np.array(hist["steer_cmd_deg"])[tail]; meas = np.array(hist["delta_mean"])[tail]
    print(f"  measured/commanded steer: {meas.mean()/cmd.mean():.4f}  "
          f"(steering_curve alone predicts {np.array(hist['curve_scale'])[tail].mean():.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=tuple(CIRCLE_SITES), default="town06")
    parser.add_argument("--radius", type=float, default=10.0, help="target circle radius (m)")
    parser.add_argument("--speed", type=float, default=5.0, help="target speed (m/s)")
    parser.add_argument("--steer-deg", type=float, default=None,
                        help="override the steer command (deg of wheel angle before the curve); "
                             "default is derived from atan(L/R) and the steering curve")
    parser.add_argument("--direction", type=float, default=-1.0,
                        help="-1 turns the way the reference circle runs, +1 the other way")
    parser.add_argument("--straight-time", type=float, default=0.0,
                        help="seconds to run straight before applying the steer, so the step in "
                             "is visible")
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--summary-window", type=float, default=3.0,
                        help="length of the settled tail summarised at the end (s)")
    parser.add_argument("--imu-lpf", type=float, default=2.0,
                        help="zero-phase low-pass cutoff (Hz) for the accelerometer, for display "
                             "and per-tick checks; the unfiltered series is kept as *_raw. 0 "
                             "disables. Nearly all of the IMU's variance is above 6 Hz")
    parser.add_argument("--skip-ticks", type=int, default=10,
                        help="ticks to discard at the start -- the IMU spikes on spawn")
    parser.add_argument("--deriv-window", type=float, default=0.25,
                        help="zero-phase derivative window (s)")

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--vehicle", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--dt", type=float, default=0.02, help="fixed sim step (s)")
    parser.add_argument("--times-run", type=float, default=2.0, help="how times for simulation running?")

    parser.add_argument("--out-dir", default=os.path.join(HERE, "plots"))
    parser.add_argument("--save-json", default=None, help="also dump the raw per-tick history")
    parser.add_argument("--no-show", action="store_true", help="save figures without displaying")
    args = parser.parse_args()

    site = CIRCLE_SITES[args.site]
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != site["map"]:
        print(f"Loading {site['map']}...")
        world = client.load_world(site["map"])

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    print(f"site '{args.site}' ({site['map']}) centre={site['centre']} "
          f"drivable radii {site['radius_range'][0]}-{site['radius_range'][1]} m")
    try:
        hist, path, geom = run(world, site, args)
    finally:
        world.apply_settings(original_settings)
        print("\nCleaned up: world settings restored.")

    if len(hist["t"]) < 10:
        raise SystemExit("too few ticks logged to plot")

    summarise(hist, args, geom)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(hist, f)
        print(f"History saved: {args.save_json}")

    import matplotlib.pyplot as plt
    from viz_utils import plot_results

    path_x, path_y, _ = path
    os.makedirs(args.out_dir, exist_ok=True)
    plot_results(path_x, path_y, hist, args.speed, args.out_dir, show=False, summary=True,
                 label=f"{args.site} R={args.radius:.0f} m", name="probe_circle")
    plot_trajectory_detail(hist, site, args, args.out_dir)
    plot_steering(hist, args, args.out_dir)
    plot_steady_state(hist, args, args.out_dir)
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
