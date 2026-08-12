r"""Yaw moment of inertia (Iz), measured two independent ways.

    --method step      (default)  transient step-steer response, through the tire model
    --method impulse              airborne angular impulse, with no tire model at all
    --report-only                 no simulator; just re-run the checks on the saved JSON

Why both: the step-steer fit needs Cf/Cr to turn measured motion into a yaw moment, so any scale
error in those lands entirely in Iz, and nothing inside that experiment can tell the two apart.
The impulse method removes the tires from the problem completely, so the pair brackets the truth
in a way neither does alone. Run impulse first -- it is the one that can falsify the other.

--------------------------------------------------------------------------------------------
method = step
--------------------------------------------------------------------------------------------
Loads Cf, Cr (and m, lf, lr) from estimate_cornering_stiffness.py's output. At steady state Iz
drops out of the yaw moment balance entirely (r_dot = 0) -- which is exactly why that script can
solve for Cf/Cr without knowing Iz, and equally why Iz is only identifiable from a transient:

    Iz * r_dot = lf*Fyf - lr*Fyr = lf*Cf*alpha_f - lr*Cr*alpha_r

With Cf, Cr known the right-hand side (call it M) is computable at every tick from
(v_x, v_y, r, delta), leaving one unknown. The fit integrates rather than differentiates --
integral(M dt) = Iz*(r - r0) -- because differentiating r forces a smoothing filter that cannot
win: wide enough to control noise, it clips the peak of r_dot and inflates Iz. See
bicycle.fit_inertia_integral() for the numbers behind that choice.

Maneuver: cruise straight, then a constant open-loop step in steering, logged from just before the
step through the rise into the new steady turn. Steer magnitude stays in the ~8-10 deg range
estimate_cornering_stiffness.py calibrated over, so the borrowed linear tire model still applies.
Run on Town06's long straight (spawn index 86, same spot longitudinal_PID.py uses): four lanes of
room to drift sideways during the transient, and it starts straight.

delta is the *measured* front wheel angle (bicycle.front_steer_angle), not `steer_cmd * max_steer`.
A step in command is not a step at the wheel -- the actuator ramps over roughly the same timescale
as the transient being fitted.

--------------------------------------------------------------------------------------------
method = impulse
--------------------------------------------------------------------------------------------
Drop the vehicle far above the map. With no wheel in contact the only external force is gravity,
which acts at the centre of mass and so exerts no yaw moment whatsoever. Hit it with a known
angular impulse about z and only the rigid-body relation is left:

    Iz = J_z / delta_r

No Cf, no Cr, no slip angles, no bicycle model. CARLA's impulse units are not reliably documented,
so the script calibrates instead of assuming: phase 1 applies a pure linear impulse and recovers
the mass from P/dv, which pins the units to SI if it returns the blueprint's mass. The residual
ambiguity -- whether angular impulse is taken in kg*m^2*rad/s or kg*m^2*deg/s -- is reported both
ways; they differ by 57.3x, so only one can be a plausible car.

Two systematic effects are handled rather than hoped away: UE4's angular damping (the spin decays,
so r(t) = r0*exp(-t/tau) is fitted and r0 extrapolated back to the impulse), and cross-axis
coupling (roll/pitch rates are logged, and a warning fires if a pure z impulse is not producing a
pure z rotation). Sweeping several impulse magnitudes is the linearity check.

Usage (Ubuntu):
    cd ~/carla_control
    python3 lateral_parameter/estimate_yaw_inertia.py --method impulse --save-plot
    python3 lateral_parameter/estimate_yaw_inertia.py --save-plot
    python3 lateral_parameter/estimate_yaw_inertia.py --report-only

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py --method impulse --save-plot
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py --speeds 5,6,7,8 --steer-deg 9 --save-plot
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py --report-only
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

# functions.py resolves CARLA_ROOT and puts the simulator's PythonAPI on sys.path, so it has to
# be imported before carla's navigation helpers are reachable.
from functions import PID, CollisionWatch, LowPassFilter, clipping, get_vehicle_geometry

import carla

from bicycle import (bracket, centered_integral_pair, fit_inertia_derivative, fit_inertia_integral,
                     front_steer_angle, report, steady_yaw_rate, steer_convention_report,
                     transient_span, zero_phase_derivative, yaw_mode)

MAP_NAME = "Town06"
ORIGIN_INDEX = 86


def setup_world(args):
    """Connect, load the map if needed, switch to synchronous mode. Returns (world, old_settings)."""
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
    return world, original_settings


# ============================================================================================
# method = step
# ============================================================================================

def run_step_trial(world, origin_transform, blueprint, imu_bp, max_steer, steer_cmd, target_speed,
                   args):
    """Cruise straight to target_speed, step the steering, log through the transient.

    Logging starts `args.pre_step_time` before the step, not at it: the derivative cross-check is
    centred and needs samples on both sides of every point it differentiates, and the most
    important point of the run is the step instant itself.

    Returns {"t", "v_x", "v_y", "r", "delta", "delta_cmd", "step_idx"} with t measured from the
    step, r left raw. None if the car never reached cruise speed or hit something.
    """
    vehicle = world.spawn_actor(blueprint, origin_transform)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    collision = CollisionWatch(world, vehicle)
    collision.arm()

    speed_pid = PID(kp=0.5, ki=0.2, kd=0.0, dt=args.dt)
    # Causal, and deliberately so: this only drives the "has it settled" gate, where lag is
    # harmless. It is NOT what the fit uses.
    r_dot_gate = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    prev_r = None

    cruise_ticks = int(args.cruise_time / args.dt)
    pre_ticks = int(args.pre_step_time / args.dt)
    transient_ticks = int(args.transient_time / args.dt)
    cruise_settled_ticks = 0
    stepped = False
    step_idx = None
    window = {"t": [], "v_x": [], "v_y": [], "r": [], "delta": [], "delta_cmd": []}

    imu = None
    try:
        world.tick()
        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)

        for i in range(int(args.max_duration / args.dt)):
            step_start = time.time()
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)

            transform = vehicle.get_transform()
            yaw = math.radians(transform.rotation.yaw)
            vel = vehicle.get_velocity()
            v_x = vel.x * math.cos(yaw) + vel.y * math.sin(yaw)
            v_y = -vel.x * math.sin(yaw) + vel.y * math.cos(yaw)
            r = imu_data.gyroscope.z   # rad/s, body-frame yaw rate straight off the sensor
            # Read before the new command is applied: the angle the physics engine actually held
            # over the interval that produced the v_y and r just measured.
            delta_meas = front_steer_angle(vehicle)

            r_dot_gated = r_dot_gate.step(0.0 if prev_r is None else (r - prev_r) / args.dt)
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
                settled = abs(v_x - target_speed) < args.speed_tol and abs(r_dot_gated) < 0.05
                cruise_settled_ticks = cruise_settled_ticks + 1 if settled else 0
                # rolling pre-step buffer, so the step lands with history already behind it
                if settled:
                    window["t"].append(0.0)   # rewritten once step_idx is known
                    window["v_x"].append(v_x); window["v_y"].append(v_y); window["r"].append(r)
                    window["delta"].append(delta_meas)
                    window["delta_cmd"].append(0.0)
                    for key in window:
                        if len(window[key]) > pre_ticks:
                            del window[key][0]
                else:
                    for key in window:
                        window[key] = []
                if i % int(1.0 / args.dt) == 0:
                    print(f"    t={i*args.dt:5.1f}s  v_x={v_x:5.2f}/{target_speed:.1f} m/s  "
                          f"settled={cruise_settled_ticks}/{cruise_ticks}")
                if cruise_settled_ticks >= cruise_ticks:
                    stepped = True
                    # +1 because the stepped command is only applied at the *end* of this
                    # iteration: the sample read at the top of the next one still belongs to the
                    # unsteered tick, and the one after it is the first the step actually moved.
                    step_idx = len(window["t"]) + 1
                    print(f"    STEP at t={i*args.dt:.1f}s: steer -> "
                          f"{math.degrees(steer_cmd * max_steer):+.1f} deg "
                          f"({len(window['t'])} pre-step ticks buffered)")
            else:
                window["t"].append(0.0)
                window["v_x"].append(v_x); window["v_y"].append(v_y); window["r"].append(r)
                window["delta"].append(delta_meas)
                window["delta_cmd"].append(steer_cmd * max_steer)
                if len(window["t"]) - step_idx >= transient_ticks:
                    window["t"] = [(k - step_idx) * args.dt for k in range(len(window["r"]))]
                    window["step_idx"] = step_idx
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


def method_step(world, args):
    """Full step-steer identification. Returns the dict written to --out."""
    with open(args.cf_cr_file) as f:
        tire = json.load(f)
    Cf, Cr, mass, lf, lr = tire["Cf"], tire["Cr"], tire["mass"], tire["lf"], tire["lr"]
    print(f"Loaded {os.path.basename(args.cf_cr_file)}: Cf={Cf:,.0f} N/rad  Cr={Cr:,.0f} N/rad  "
          f"mass={mass:.1f} kg  lf={lf:.2f} m  lr={lr:.2f} m")

    if args.dt > 0.02:
        print(f"\nWARNING: --dt {args.dt} is too coarse. The yaw rise lasts ~0.2-0.3 s, leaving "
              f"roughly {0.25/args.dt:.0f} samples across it. Against a simulated bicycle with a "
              f"known Iz this estimator holds to ~2% at dt=0.01 but drifts 10-28% high at "
              f"dt=0.05 once realistic gyro noise is present.\n")

    origin_transform = world.get_map().get_spawn_points()[ORIGIN_INDEX]
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter(args.vehicle)[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    probe = world.spawn_actor(blueprint, origin_transform)
    _, lf_probe, lr_probe, max_steer = get_vehicle_geometry(probe, origin_transform)
    probe.destroy()
    if abs(lf_probe - lf) > 0.05 or abs(lr_probe - lr) > 0.05:
        print(f"warning: this vehicle's lf/lr ({lf_probe:.2f}/{lr_probe:.2f}) differ from the "
              f"cornering-stiffness file's ({lf:.2f}/{lr:.2f}) -- Cf/Cr may not transfer cleanly")

    steer_cmd = clipping(math.radians(args.steer_deg) / max_steer, 1.0, -1.0)

    S_all, R_all, M_all, r_dot_all = [], [], [], []
    delta_meas_all, delta_cmd_all = [], []
    raw_log, per_speed = [], []
    for target_speed in [float(v) for v in args.speeds.split(",")]:
        print(f"\n=== cruise speed {target_speed:.1f} m/s ===")
        window = run_step_trial(world, origin_transform, blueprint, imu_bp, max_steer, steer_cmd,
                               target_speed, args)
        if window is None:
            print(f"  did not settle within {args.max_duration:.0f}s -- skipped")
            continue

        v_x = np.array(window["v_x"]); v_y = np.array(window["v_y"])
        r = np.array(window["r"]); delta = np.array(window["delta"])
        step_idx = window["step_idx"]

        # Order matters: the fit window comes from r alone, then the derivative window is sized
        # from that rise. A fixed derivative window either smears r_dot's peak or passes noise,
        # and only the rise itself says which.
        start, end = transient_span(r, step_idx, args.dt, settle_frac=args.settle_frac)
        rise_s = (end - start) * args.dt
        r_dot = zero_phase_derivative(r, args.dt, window_s=args.savgol_window, rise_s=rise_s)

        beta = v_y / v_x
        M = np.array(lf * Cf * (delta - beta - lf * r / v_x) - lr * Cr * (-beta + lr * r / v_x))

        M_fit, r_dot_fit, r_fit = M[start:end], r_dot[start:end], r[start:end]
        S_c, R_c = centered_integral_pair(M_fit, r_fit, args.dt)
        S_all.extend(S_c.tolist()); R_all.extend(R_c.tolist())
        M_all.extend(M_fit.tolist()); r_dot_all.extend(r_dot_fit.tolist())
        delta_meas_all.extend(delta.tolist()); delta_cmd_all.extend(window["delta_cmd"])

        tail = max(1, int(0.5 / args.dt))
        r_ss = float(np.mean(r[-tail:])); v_ss = float(np.mean(v_x[-tail:]))
        delta_ss = float(np.mean(delta[-tail:]))
        r_ss_pred = steady_yaw_rate(delta_ss, v_ss, Cf, Cr, mass, lf, lr)

        fwd, rev, geo = fit_inertia_integral(M_fit, r_fit, args.dt)
        d_geo = fit_inertia_derivative(M_fit, r_dot_fit)[2]
        wn, zeta = yaw_mode(Cf, Cr, mass, geo, lf, lr, v_ss) if geo == geo else (float("nan"),) * 2

        print(f"  logged {len(r)} ticks ({step_idx} pre-step), fit window {end-start} ticks "
              f"({rise_s:.2f} s of rise)")
        if end - start < 15:
            print(f"    WARNING: only {end-start} samples across the rise -- resolution-limited. "
                  f"Rerun with a smaller --dt before reading anything into this trial.")
        print(f"  steady-state r: measured {math.degrees(r_ss):+.2f} deg/s  vs  predicted "
              f"{math.degrees(r_ss_pred):+.2f} deg/s from Cf/Cr")
        print(f"  Iz = {geo:,.0f} kg*m^2   bracket [{fwd:,.0f}, {rev:,.0f}]   "
              f"derivative cross-check {d_geo:,.0f}")
        print(f"  implied yaw mode at {v_ss:.1f} m/s: omega_n={wn:.2f} rad/s  zeta={zeta:.2f}"
              + ("  -> r(t) should not overshoot" if zeta >= 1 else "  -> r(t) should overshoot"))

        per_speed.append({"target_speed": target_speed, "n_ticks": len(r),
                         "n_fit_ticks": int(end - start), "Iz_forward": fwd, "Iz_reverse": rev,
                         "Iz_trial": geo, "Iz_derivative": d_geo, "r_ss_measured": r_ss,
                         "r_ss_predicted": r_ss_pred, "v_ss": v_ss, "delta_ss": delta_ss,
                         "omega_n": wn, "zeta": zeta})
        raw_log.append({"target_speed": target_speed, "step_idx": step_idx, "dt": args.dt,
                       "t": window["t"], "v_x": window["v_x"], "v_y": window["v_y"],
                       "r": window["r"], "delta": window["delta"],
                       "delta_cmd": window["delta_cmd"],
                       "fit_start": int(start), "fit_end": int(end)})

    if len(S_all) < 10:
        raise SystemExit(f"only {len(S_all)} ticks logged -- not enough transient data to fit Iz")

    print(f"\n{'speed':>6} {'n_fit':>6} {'Iz_fwd':>11} {'Iz_rev':>11} {'Iz':>11} {'Iz_deriv':>11} {'zeta':>6}")
    for p in per_speed:
        print(f"{p['target_speed']:6.1f} {p['n_fit_ticks']:6d} {p['Iz_forward']:11,.0f} "
              f"{p['Iz_reverse']:11,.0f} {p['Iz_trial']:11,.0f} {p['Iz_derivative']:11,.0f} "
              f"{p['zeta']:6.2f}")

    Iz_fwd, Iz_rev, Iz = bracket(S_all, R_all)
    Iz_deriv = fit_inertia_derivative(M_all, r_dot_all)[2]
    print(f"\nIz = {Iz:,.0f} kg*m^2   (integral form, pooled over {len(S_all)} fit ticks)")
    print(f"   errors-in-variables bracket [{Iz_fwd:,.0f}, {Iz_rev:,.0f}] -- random error only, "
          f"NOT a confidence interval")
    print(f"   derivative-form cross-check: {Iz_deriv:,.0f} kg*m^2")

    # A rigid body's inertia cannot depend on how fast it was going, so spread across trials is
    # pure estimator bias and bounds how far this number can be trusted.
    if len(per_speed) >= 2:
        trials = np.array([p["Iz_trial"] for p in per_speed])
        spread = (trials.max() - trials.min()) / trials.mean()
        print(f"across-speed spread: {spread*100:.0f}% of mean "
              f"({trials.min():,.0f} .. {trials.max():,.0f})")
        if spread > 0.15:
            print(f"  WARNING: Iz is a rigid-body property and cannot vary with speed. A "
                  f"{spread*100:.0f}% spread is systematic error upstream -- most likely Cf/Cr, "
                  f"whose scale error lands entirely here. --method impulse settles it.")

    steer_slope, steer_note = steer_convention_report(delta_meas_all, delta_cmd_all)
    print(f"measured/commanded steer slope: {steer_slope:.3f}")
    if steer_note:
        print(f"  WARNING: {steer_note}")

    return {"method": "step", "Iz": Iz, "Iz_forward": Iz_fwd, "Iz_reverse": Iz_rev,
            "Iz_derivative": Iz_deriv, "Cf": Cf, "Cr": Cr, "mass": mass, "lf": lf, "lr": lr,
            "steer_slope": steer_slope, "per_speed": per_speed, "raw": raw_log,
            "_plot": {"S": S_all, "R": R_all}}


# ============================================================================================
# method = impulse
# ============================================================================================

def spawn_airborne(world, blueprint, base_transform, height, imu_bp, settle_ticks=10):
    """Spawn `height` metres up and let the physics state settle.

    The settle ticks matter: a freshly spawned actor reports zero velocity for a tick or two while
    UE4 registers the body, and an impulse applied inside that window hits a body that is not yet
    integrating.
    """
    transform = carla.Transform(
        carla.Location(base_transform.location.x, base_transform.location.y,
                       base_transform.location.z + height),
        base_transform.rotation,
    )
    vehicle = world.spawn_actor(blueprint, transform)
    vehicle.set_simulate_physics(True)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    # hands off the controls -- a brake command would spin up wheel physics we do not want
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0, hand_brake=False))

    imu_queue = queue.Queue()
    imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    imu.listen(imu_queue.put)
    for _ in range(settle_ticks):
        world.tick()
        imu_queue.get(timeout=2.0)
    return vehicle, imu, imu_queue


def measure_mass(world, blueprint, base_transform, imu_bp, impulse_n_s, args):
    """Phase 1: pure linear impulse at the CoM in world x. Returns (mass CARLA behaved as, dv_x).

    Gravity is along z and cannot touch v_x, so the x channel isolates the impulse without needing
    to model the fall at all.
    """
    vehicle, imu, imu_queue = spawn_airborne(world, blueprint, base_transform, args.height, imu_bp)
    try:
        v_before = vehicle.get_velocity()
        vehicle.add_impulse(carla.Vector3D(impulse_n_s, 0.0, 0.0))
        world.tick()
        imu_queue.get(timeout=2.0)
        dvx = vehicle.get_velocity().x - v_before.x
        return (impulse_n_s / dvx if abs(dvx) > 1e-6 else float("nan")), dvx
    finally:
        imu.stop(); imu.destroy(); vehicle.destroy()


def fit_decay(t, r):
    """Fit r(t) = r0*exp(-t/tau), returning (r0, tau).

    The impulse lands between two ticks, so the first sample is already one step into the decay;
    extrapolating back to t=0 recovers the step the impulse actually produced rather than what
    survived a tick of damping. Falls back to the first sample if the trace does not decay.
    """
    mask = np.abs(r) > 1e-4
    if mask.sum() < 3:
        return (float(r[0]) if len(r) else float("nan")), float("inf")
    sign = 1.0 if np.mean(r[mask]) > 0 else -1.0
    slope, intercept = np.polyfit(t[mask], np.log(np.abs(r[mask])), 1)
    if slope >= 0:
        return float(r[0]), float("inf")
    return float(sign * math.exp(intercept)), float(-1.0 / slope)


def measure_yaw_impulse(world, blueprint, base_transform, imu_bp, J, args):
    """Phase 2: angular impulse about z, then log the spin-down."""
    vehicle, imu, imu_queue = spawn_airborne(world, blueprint, base_transform, args.height, imu_bp)
    try:
        vehicle.add_angular_impulse(carla.Vector3D(0.0, 0.0, J))
        ts, rz, rx, ry = [], [], [], []
        for k in range(int(args.decay_time / args.dt)):
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)
            ts.append((k + 1) * args.dt)
            rz.append(imu_data.gyroscope.z)
            rx.append(imu_data.gyroscope.x)
            ry.append(imu_data.gyroscope.y)
        r0, tau = fit_decay(np.array(ts), np.array(rz))
        return {"J": J, "t": ts, "r_z": rz, "r_x": rx, "r_y": ry, "r_first": rz[0],
               "r0": r0, "tau": tau,
               "cross_axis": float(max(np.max(np.abs(rx)), np.max(np.abs(ry))))}
    finally:
        imu.stop(); imu.destroy(); vehicle.destroy()


def method_impulse(world, args):
    """Tire-free Iz from an airborne angular impulse. Returns the dict written to --out."""
    blueprint = world.get_blueprint_library().filter(args.vehicle)[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")
    base_transform = world.get_map().get_spawn_points()[ORIGIN_INDEX]

    probe = world.spawn_actor(blueprint, base_transform)
    physics = probe.get_physics_control()
    mass, com = physics.mass, physics.center_of_mass
    # `moi` is the ENGINE's rotational inertia, not the chassis yaw inertia -- printed only so it
    # is never mistaken for the latter.
    print(f"blueprint mass={mass:.1f} kg  center_of_mass=({com.x:.2f}, {com.y:.2f}, {com.z:.2f}) m"
          f"  engine moi={physics.moi:.2f} (NOT the chassis Iz)")
    for i, w in enumerate(physics.wheels):
        print(f"  wheel[{i}] lat_stiff_value={w.lat_stiff_value:.2f}  "
              f"lat_stiff_max_load={w.lat_stiff_max_load:.2f}  tire_friction={w.tire_friction:.2f}")
    probe.destroy()

    print("\n=== phase 1: linear impulse (mass / units calibration) ===")
    mass_meas, dvx = measure_mass(world, blueprint, base_transform, imu_bp, args.linear_impulse, args)
    err = 100.0 * (mass_meas / mass - 1.0) if mass_meas == mass_meas else float("nan")
    print(f"  P={args.linear_impulse:,.0f} N*s -> dv_x={dvx:.4f} m/s  =>  mass={mass_meas:,.1f} kg "
          f"vs blueprint {mass:.1f} kg ({err:+.1f}%)")
    units_ok = mass_meas == mass_meas and abs(err) < 5.0
    if not units_ok:
        print("  WARNING: recovered mass does not match the blueprint, so CARLA's impulse units "
              "are not the SI N*s assumed here (or the body was not free). Everything below "
              "inherits that -- fix this first.")

    # Phase 1 proves linear impulses are SI N*s, but that does NOT carry over to the angular API:
    # measured here, add_angular_impulse lands about six orders of magnitude off a kg*m^2*rad/s
    # reading. UE4 works internally in centimetres and exposes both radian and degree variants, so
    # there are four candidate conventions. Rather than pick one, probe the actual scale with a
    # small impulse and size the real sweep from it, so the spin is large enough to measure
    # properly and to fit the damping on.
    print("\n=== phase 2a: probing the angular impulse scale ===")
    probe_t = measure_yaw_impulse(world, blueprint, base_transform, imu_bp, args.probe_impulse, args)
    scale = probe_t["r0"] / args.probe_impulse    # rad/s produced per unit of J
    print(f"  J={args.probe_impulse:,.0f} -> r0={probe_t['r0']:.3e} rad/s   "
          f"=> {scale:.3e} rad/s per unit J")
    if abs(scale) < 1e-18:
        raise SystemExit("angular impulse produced no measurable rotation -- cannot continue")
    targets = [float(x) for x in args.target_rates.split(",")]
    impulses = [math.radians(t) / scale for t in targets]
    print(f"  sizing the sweep for {targets} deg/s  =>  J = "
          + ", ".join(f"{j:,.3g}" for j in impulses))

    print("\n=== phase 2b: angular impulse about z ===")
    trials = []
    for J in impulses:
        t = measure_yaw_impulse(world, blueprint, base_transform, imu_bp, J, args)
        t["Iz_if_rad"] = J / t["r0"] if t["r0"] else float("nan")
        t["Iz_if_deg"] = J / math.degrees(t["r0"]) if t["r0"] else float("nan")
        trials.append(t)
        print(f"  J={J:>8,.0f}: r0={math.degrees(t['r0']):+8.2f} deg/s "
              f"(first sample {math.degrees(t['r_first']):+.2f}, tau={t['tau']:.2f} s)  =>  "
              f"Iz={t['Iz_if_rad']:>10,.0f} [if rad] | {t['Iz_if_deg']:>9,.0f} [if deg]")
        if t["cross_axis"] > args.cross_axis_tol:
            print(f"    WARNING: roll/pitch rate reached {t['cross_axis']:.3f} rad/s -- the z "
                  f"impulse is exciting other axes, so gyro.z is not a clean measurement here.")

    iz_rad_all = np.array([t["Iz_if_rad"] for t in trials])
    spread = (iz_rad_all.max() - iz_rad_all.min()) / iz_rad_all.mean()
    print(f"\nlinearity across impulse magnitudes: spread {spread*100:.2f}% of mean")
    if spread > 0.05:
        print("  WARNING: Iz cannot depend on J. Something is nonlinear -- most likely the car "
              "touched down, or the impulse was clamped.")

    # Four candidate conventions, from UE4's radian/degree variants crossed with its centimetre
    # internals. They are separated by factors of 57.3 and 10,000, and a passenger car's yaw
    # inertia is known to within a factor of ~3, so at most one of the four can be the answer.
    # That is what makes this identifiable at all without documentation.
    raw_rad = float(np.mean(iz_rad_all))                                  # J / r0   [rad]
    candidates = {
        "kg*m^2*rad/s":  raw_rad,
        "kg*m^2*deg/s":  raw_rad / math.degrees(1.0),
        "kg*cm^2*rad/s": raw_rad / 1e4,
        "kg*cm^2*deg/s": raw_rad / math.degrees(1.0) / 1e4,
    }
    print("\nIz under each candidate unit convention for add_angular_impulse:")
    for name, value in candidates.items():
        print(f"  {name:>16} -> {value:>18,.1f} kg*m^2")

    plausible = [(n, v) for n, v in candidates.items() if args.iz_min < v < args.iz_max]
    Iz = units = None
    if len(plausible) == 1:
        units, Iz = plausible[0]
        print(f"\nGROUND TRUTH Iz = {Iz:,.0f} kg*m^2")
        print(f"  Only the '{units}' reading falls in the {args.iz_min:,.0f}-{args.iz_max:,.0f} "
              f"kg*m^2 a {mass:.0f} kg car can possibly have; the others are off by factors of "
              f"57.3 or 10,000. So add_angular_impulse takes {units}.")
        print(f"  This is a *measurement*, not the documented convention -- it is confirmed by the "
              f"linearity above ({spread*100:.2f}% across a {max(1e-9, max(t['J'] for t in trials)/min(t['J'] for t in trials)):.0f}x "
              f"range of J) and should be cross-checked against --method step.")
    else:
        print(f"\n  -> {len(plausible)} candidates fall in the plausible range; cannot resolve the "
              f"convention from magnitude alone.")

    return {"method": "impulse", "Iz": Iz, "units": units, "Iz_raw_rad": raw_rad,
            "candidates": candidates, "linearity_spread": spread, "mass": mass,
            "mass_measured": mass_meas, "linear_impulse": args.linear_impulse,
            "units_ok": units_ok, "trials": trials}


# ============================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("step", "impulse"), default="step",
                        help="step: transient step-steer through the tire model. impulse: airborne "
                             "angular impulse, no tire model at all (run this one first)")
    parser.add_argument("--report-only", action="store_true",
                        help="skip the simulator entirely and re-run the validation checks on the "
                             "saved JSON files")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--vehicle", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="fixed sim step (s) -- finer than the 0.05 the other scripts use, "
                             "because the yaw transient lasts only ~0.2-0.3 s. Trials are a few "
                             "seconds each, so the extra ticks are cheap")
    parser.add_argument("--times-run", type=float, default=25.0, help="how times for simulation running?")

    # ---- method = step ---- #
    parser.add_argument("--cf-cr-file", default=os.path.join(HERE, "cornering_stiffness.json"))
    parser.add_argument("--speeds", default="5,6,7,8", help="cruise speeds to step from (m/s)")
    parser.add_argument("--steer-deg", type=float, default=9.0,
                        help="step steer magnitude (deg) -- kept in the range Cf/Cr were "
                             "calibrated over, since this reuses that linear tire fit")
    parser.add_argument("--speed-tol", type=float, default=0.2)
    parser.add_argument("--cruise-time", type=float, default=2.0)
    parser.add_argument("--pre-step-time", type=float, default=0.5,
                        help="how long to buffer before the step (s), so the centred derivative "
                             "has samples on both sides of the step instant")
    parser.add_argument("--transient-time", type=float, default=2.5,
                        help="how long to log after the step. The fit uses only the rise; the "
                             "settled tail is kept for the steady-state gain check")
    parser.add_argument("--settle-frac", type=float, default=0.98)
    parser.add_argument("--savgol-window", type=float, default=None,
                        help="width (s) of the derivative window for the cross-check fit. Default "
                             "sizes it at a quarter of the measured rise")
    parser.add_argument("--max-duration", type=float, default=20.0)

    # ---- method = impulse ---- #
    parser.add_argument("--height", type=float, default=80.0,
                        help="spawn height above the map (m) -- high enough not to reach the "
                             "ground within --decay-time")
    parser.add_argument("--probe-impulse", type=float, default=1e6,
                        help="small angular impulse used only to measure how much rotation a unit "
                             "of J actually buys, so the real sweep can be sized in physical terms")
    parser.add_argument("--target-rates", default="5,10,20,40",
                        help="yaw rates (deg/s) the sweep aims for; the impulses to reach them are "
                             "computed from the probe. Several, because Iz is a constant and a "
                             "value that drifts with J means the model is wrong")
    parser.add_argument("--iz-min", type=float, default=500.0,
                        help="plausibility window used to pick the impulse unit convention")
    parser.add_argument("--iz-max", type=float, default=8000.0)
    parser.add_argument("--linear-impulse", type=float, default=5000.0,
                        help="linear impulse (N*s) for the phase-1 mass/units calibration")
    parser.add_argument("--decay-time", type=float, default=1.0)
    parser.add_argument("--cross-axis-tol", type=float, default=0.05)

    parser.add_argument("--out", default=None,
                        help="defaults to yaw_inertia.json (step) or yaw_inertia_impulse.json "
                             "(impulse)")
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"))
    parser.add_argument("--save-plot", action="store_true")
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        HERE, "yaw_inertia.json" if args.method == "step" else "yaw_inertia_impulse.json")

    if args.report_only:
        with open(args.cf_cr_file) as f:
            tire = json.load(f)
        yawf = json.load(open(out_path)) if os.path.exists(out_path) else {}
        report(tire["Cf"], tire["Cr"], tire["mass"], tire["lf"], tire["lr"],
               Iz=yawf.get("Iz"), raw_transients=yawf.get("raw"))
        return

    world, original_settings = setup_world(args)
    results = None
    try:
        results = method_step(world, args) if args.method == "step" else method_impulse(world, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("\nCleaned up: world settings restored.")

    if results is None:
        return

    plot_data = results.pop("_plot", None)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    if args.method == "step":
        report(results["Cf"], results["Cr"], results["mass"], results["lf"], results["lr"],
               Iz=results["Iz"], raw_transients=results["raw"])
        impulse_path = os.path.join(HERE, "yaw_inertia_impulse.json")
        if os.path.exists(impulse_path):
            with open(impulse_path) as f:
                truth = json.load(f).get("Iz")
            if truth:
                print(f"\n--- vs tire-free ground truth ---")
                print(f"impulse method: {truth:,.0f} kg*m^2   step method: {results['Iz']:,.0f} "
                      f"({results['Iz']/truth:.2f}x)")
                if abs(results["Iz"] / truth - 1) > 0.2:
                    print(f"  The step fit is off by {100*(results['Iz']/truth-1):+.0f}%. Since M's "
                          f"scale is Cf/Cr's scale, that error most likely belongs to Cf/Cr.")

    if args.save_plot:
        save_plots(results, plot_data, args)


def save_plots(results, plot_data, args):
    from viz_utils import COLOR_AXIS, COLOR_BLUE, COLOR_ORANGE, _legend, _save, _style_axes, COLOR_BG
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    _style_axes(ax_a); _style_axes(ax_b)
    ax_a.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_a.axvline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_b.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    ax_b.axvline(0.0, color=COLOR_AXIS, linewidth=1)

    if results["method"] == "step":
        for trial in results["raw"]:
            ax_a.plot(trial["t"], np.degrees(trial["r"]), linewidth=1.2,
                      label=f"{trial['target_speed']:.0f} m/s")
            fs, fe = trial["fit_start"], trial["fit_end"]
            ax_a.plot(trial["t"][fs:fe], np.degrees(trial["r"][fs:fe]), linewidth=3, alpha=0.25,
                      color=COLOR_ORANGE)
        ax_a.set_xlabel("time from step (s)"); ax_a.set_ylabel("yaw rate r (deg/s)")
        ax_a.set_title("Step response (shaded = window used for the fit)")

        S, R = plot_data["S"], plot_data["R"]
        Iz, lo, hi = results["Iz"], results["Iz_forward"], results["Iz_reverse"]
        ax_b.scatter(R, S, color=COLOR_BLUE, s=8, alpha=0.4, zorder=3, label="measured")
        xs = np.linspace(min(R + [0]), max(R + [0]), 20)
        ax_b.plot(xs, Iz * xs, color=COLOR_BLUE, linestyle="--", label=f"Iz={Iz:,.0f} kg*m^2")
        ax_b.fill_between(xs, lo * xs, hi * xs, color=COLOR_BLUE, alpha=0.12,
                          label=f"bracket [{lo:,.0f}, {hi:,.0f}]")
        ax_b.set_xlabel("yaw rate, centred per trial (rad/s)")
        ax_b.set_ylabel("delivered angular impulse, centred (N*m*s)")
        ax_b.set_title("Yaw inertia fit: integral(M dt) = Iz * (r - r0)")
        stem = "yaw_inertia_step"
    else:
        for t in results["trials"]:
            ax_a.plot(t["t"], np.degrees(t["r_z"]), linewidth=1.4, label=f"J={t['J']:,.0f}")
            ax_a.plot(0.0, math.degrees(t["r0"]), marker="o", markersize=5, color=COLOR_AXIS)
        ax_a.set_xlabel("time after impulse (s)"); ax_a.set_ylabel("yaw rate (deg/s)")
        ax_a.set_title("Airborne spin-down (dots = damping-corrected r0)")

        r0s = [t["r0"] for t in results["trials"]]
        js = [t["J"] for t in results["trials"]]
        Iz_rad = results["Iz_raw_rad"]
        ax_b.scatter(r0s, js, color=COLOR_BLUE, zorder=3, label="measured")
        xs = np.linspace(0, max(r0s) * 1.05, 20)
        ax_b.plot(xs, Iz_rad * xs, color=COLOR_BLUE, linestyle="--", label=f"Iz={Iz_rad:,.0f}")
        ax_b.set_xlabel("yaw rate step r0 (rad/s)"); ax_b.set_ylabel("applied angular impulse J")
        ax_b.set_title("Linearity: J = Iz * r0")
        stem = "yaw_inertia_impulse"

    _legend(ax_a); _legend(ax_b)
    os.makedirs(args.plot_dir, exist_ok=True)
    print(f"Figure saved: {_save(fig, args.plot_dir, stem)}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
