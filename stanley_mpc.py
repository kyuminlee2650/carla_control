r"""Stanley lateral controller + longitudinal controller, PID or MPC picked by --controller.

Same file as stanley_PID.py (path source, Stanley law, warm-up gate, BEV/video, plot_results) --
the lateral half (Stanley) is never touched here; --controller only swaps the longitudinal half,
and takes one or more of two stacks -- given more than one, results overlay on one comparison
figure per plot_results() (lateral, longitudinal, trajectory all three), instead of eyeballing
separate runs:

    stanley+mpc   (default) SpeedMPC (longitudinal_mpc.py) solves for a target acceleration
                  a_cmd each cycle, and the LUT-feedforward + PID pedal layer
                  (longitudinal_lookup/lookup_controller.py) turns that into throttle/brake.
                  SpeedMPC is imported rather than redefined here so there is one QP
                  implementation in the repo, not two -- see longitudinal_mpc.py's own docstring
                  for the QP itself and how it was tuned (--np/--nc/--w-v/--w-a/--w-j/--u-tau
                  below default to those tuned values).
    stanley+pid   plain speed PID straight to the pedal, no MPC involved -- the same controller
                  stanley_PID.py uses (same gains), so this is the baseline that stack is meant
                  to beat, not a reimplementation of it.

Both controllers share one run_trial() (spawn -> warm-up -> log -> teardown) instead of each
having its own copy of that harness; only the per-step longitudinal law differs (PidLongitudinal /
MpcLongitudinal, both a single step(ctx) -> pedal method) -- Stanley itself is computed once per
tick in run_trial() and is identical either way.

Path source: CARLA has no built-in "desired path" object.
agents.navigation.GlobalRoutePlanner runs A* over the map's waypoint topology
between an origin and destination to produce one.

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python stanley_mpc.py --times-run 1 --target-speed 15
    .venv/bin/python stanley_mpc.py --times-run 20 --save-plot --record --target-speed 15
    .venv/bin/python stanley_mpc.py --controller stanley+pid stanley+mpc --times-run 20 --save-plot --target-speed 15
    .venv/bin/python stanley_mpc.py --controller stanley+pid stanley+mpc --profile estop --stop-time 10 --stop-duration 5 --save-plot --target-speed 15

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe stanley_mpc.py --times-run 1 --target-speed 15
    .venv\Scripts\python.exe stanley_mpc.py --times-run 20 --save-plot --record --target-speed 15
    .venv\Scripts\python.exe stanley_mpc.py --times-run 20 --save-plot --record --target-speed 15 --profile sine
    .venv\Scripts\python.exe stanley_mpc.py --controller stanley+pid stanley+mpc --times-run 20 --save-plot --target-speed 15
    .venv\Scripts\python.exe stanley_mpc.py --controller stanley+pid stanley+mpc --profile estop --stop-time 10 --stop-duration 5 --save-plot --target-speed 15
"""

import argparse
import math
import os
import queue
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# see functions.py for why this path is needed alongside the pip-installed carla package
CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/2026intern/carla"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.append(os.path.join(HERE, "longitudinal_lookup"))

import carla

from functions import (PID, AngleUnwrapper, ImuAcceleration, LowPassFilter, build_path,
                       clipping, get_vehicle_geometry, lateral_error, normalize_angle)
from viz_utils import (BevView, VIEWS, VideoRecorder, follow_with_spectator, plot_results,
                       print_error_summary, run_name)

from longitudinal_lut import LongitudinalLUT
from lookup_controller import LookupController
from longitudinal_mpc import SpeedMPC

# Fixed on purpose: the route this controller has been tuned against (build_path()'s default
# origin/dest spawn indices) is a property of this specific map, not something the controller
# should have to rediscover on whatever map the server happens to have loaded.
MAP_NAME = "Town10HD_Opt"


WARM_START_SPEED_TOL = 0.3   # m/s
WARM_START_ACCEL_TOL = 0.5   # m/s^2
WARM_START_TIMEOUT = 15.0    # s -- safety cap in case target_speed is unreachable


def front_axle_offset(vehicle, lf):
    return vehicle.get_physics_control().center_of_mass.x + lf


def stanley_control(v_x, e_y, e_theta, k_theta=1.0, k=1.2, k_soft=1.0):
    return k_theta * e_theta + math.atan2(k * e_y, k_soft + abs(v_x))


def speed_reference(args, t):
    if args.profile == "sine":
        return args.target_speed + args.sine_amplitude * math.sin(2.0 * math.pi * t / args.sine_period)
    if args.profile == "estop":
        if args.stop_time <= t < args.stop_time + args.stop_duration:
            return 0.0
        return args.target_speed
    return args.target_speed


def reference_preview(args, t0, n_p, dt, warmed_up):
    """Length-Np array of v_des at t0, t0+dt, ..., t0+(Np-1)*dt -- the MPC's look-ahead.

    Before warm-up completes the profile hasn't started yet (t is undefined relative to it), so
    preview a flat target_speed instead, same as the t=0 value every profile shares.
    """
    if not warmed_up:
        return np.full(n_p, args.target_speed)
    return np.array([speed_reference(args, t0 + k * dt) for k in range(n_p)])


# ----------------------------------------------------------------------------- longitudinal controllers
# Common interface: step(ctx) -> pedal command u in [-1, 1] (positive throttle, negative brake).
# ctx (a SimpleNamespace, set fresh by run_trial() each cycle) carries t, v_x, v_ref, a_x
# (filtered), a_x_raw (IMU, for the LUT layer), gear, warmed_up. reset(u) is called once, right
# after the warm-up hand-off, so a controller can drop whatever state it accumulated chasing the
# warm-up setpoint before scoring starts. Steer/Stanley is computed once in run_trial() and is
# identical regardless of which of these runs -- only the longitudinal half differs.

class PidLongitudinal:
    """Speed PID straight to the pedal -- stanley_PID.py's own controller, same gains."""
    label = "PID"

    def __init__(self, args):
        self.pid = PID(kp=0.7, ki=0.15, kd=0.05, dt=args.dt)
        self.filter = LowPassFilter(tau=0.2, dt=args.dt, initial=0.0)

    def reset(self, u):
        pass   # stanley_PID.py never resets its PID at hand-off either; match it exactly

    def step(self, ctx):
        return clipping(self.filter.step(self.pid.step(ctx.v_ref - ctx.v_x)), 1, -1)


class MpcLongitudinal:
    """SpeedMPC -> a_cmd -> LUT feedforward + PID pedal layer (see module docstring)."""
    label = "MPC"

    def __init__(self, args):
        self.args = args
        self.mpc = SpeedMPC(dt=args.dt, n_p=args.n_p, n_c=args.n_c,
                            w_v=args.w_v, w_a=args.w_a, w_j=args.w_j,
                            a_min=args.a_min, a_max=args.a_max)
        self.pedal_ctrl = LookupController(LongitudinalLUT(args.lut), kp=args.kp, ki=args.ki,
                                           kd=args.kd, dt=args.dt)
        self.u_filter = LowPassFilter(tau=args.u_tau, dt=args.dt, initial=0.0)

    def reset(self, u):
        self.pedal_ctrl.reset()   # drop the warm-up phase's PID integral before scoring starts
        self.u_filter = LowPassFilter(tau=self.args.u_tau, dt=self.args.dt, initial=u)

    def step(self, ctx):
        preview = reference_preview(self.args, ctx.t, self.args.n_p, self.args.dt, ctx.warmed_up)
        ctx.a_cmd = self.mpc.solve(ctx.v_x, preview)   # stashed for the console print line
        # SpeedMPC's own condensed prediction X = A_bar x0 + B_bar U*, re-derived here (not
        # returned by solve()) so mpc_mpc.py's LPV lateral controller can schedule A(v_x) against
        # the same v_x trajectory this loop is actually planning to drive, instead of assuming
        # constant speed over its horizon. Unused by stanley_mpc.py itself.
        ctx.v_x_preview = (self.mpc.A_bar.ravel() * ctx.v_x + self.mpc.B_bar @ self.mpc.last_solution)
        u_raw = self.pedal_ctrl.step(ctx.gear, ctx.v_x, ctx.a_cmd, a_meas=ctx.a_x_raw)
        return self.u_filter.step(u_raw)


LONGITUDINAL = {"stanley+pid": PidLongitudinal, "stanley+mpc": MpcLongitudinal}


# ----------------------------------------------------------------------------- one trial

def run_trial(world, origin_transform, path_x, path_y, path_yaw, blueprint, imu_bp, controller,
              args, recorder_factory):
    """Spawn one vehicle, drive the whole path under Stanley (lateral) + `controller`
    (longitudinal), tear it down. Returns the run's hist dict.

    Same warm-up gate as stanley_PID.py: launch from rest under the real controller and hold off
    on logging until v_x/a_x have actually settled near the profile's own t=0 value (target_speed)
    instead of faking that starting condition.
    """
    vehicle = world.spawn_actor(blueprint, origin_transform)

    wheelbase, lf, lr, max_steer = get_vehicle_geometry(vehicle, origin_transform)
    front_offset = front_axle_offset(vehicle, lf)
    if not 0.0 < front_offset < wheelbase:
        raise RuntimeError(f"front_offset={front_offset:.2f} m is not inside the wheelbase "
                           f"({wheelbase:.2f} m); the front-axle reference point is wrong.")
    print(f"[{controller.label}] front_axle_offset={front_offset:.2f} m")

    steer_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0)
    yaw_unwrapper = AngleUnwrapper()
    rh_unwrapper = AngleUnwrapper()
    last_idx = 0

    accel = ImuAcceleration(dt=args.dt)
    yaw_acc_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    prev_yaw_rate_rad = None

    bev = None if args.no_live_view else BevView(path_x, path_y)

    hist = {"t": [], "x": [], "y": [], "v_x": [], "v_y": [], "v_des": [], "a_x": [], "jerk": [],
            "a_y": [], "yaw_rate": [], "yaw_acc": [], "jerk_total": [], "last_idx": [],
            "steer_deg": [], "throttle": [], "brake": [], "e_y": [], "yaw": [], "path_yaw": [],
            "e_theta": []}

    warmed_up = False
    log_start_i = 0

    imu = None
    recorder = None
    try:
        world.tick()

        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)
        recorder = recorder_factory(vehicle)

        steps = int(args.max_duration / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)

            # ---- sensor data ---- #
            transform = vehicle.get_transform()
            yaw = yaw_unwrapper.step(math.radians(transform.rotation.yaw))
            ego_x = transform.location.x
            ego_y = transform.location.y
            vel_vec = vehicle.get_velocity()

            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)
            v_y = -vel_vec.x * math.sin(yaw) + vel_vec.y * math.cos(yaw)

            accel.step(imu_data)
            a_x, a_y, a_x_raw = accel.a_x, accel.a_y, accel.a_x_raw
            yaw_rate_rad = imu_data.gyroscope.z
            yaw_rate = math.degrees(yaw_rate_rad)

            jerk, jerk_total = accel.jerk, accel.jerk_total

            yaw_acc = yaw_acc_filter.step(
                0.0 if prev_yaw_rate_rad is None else (yaw_rate_rad - prev_yaw_rate_rad) / args.dt)
            prev_yaw_rate_rad = yaw_rate_rad

            # ---- Stanley (lateral) ---- #
            front_x = ego_x + front_offset * math.cos(yaw)
            front_y = ego_y + front_offset * math.sin(yaw)

            last_idx, e_y = lateral_error(front_x, front_y, yaw, path_x, path_y, last_idx)
            road_heading = rh_unwrapper.step(path_yaw[last_idx])

            e_theta = normalize_angle(path_yaw[last_idx] - yaw)

            delta = stanley_control(v_x, e_y, e_theta)
            steer = clipping(delta / max_steer, 3 / 7, -3 / 7)
            steer_deg = steer * math.degrees(max_steer)

            # ---- longitudinal (PID or MPC, per --controller) ---- #
            t_probe = (i - log_start_i) * args.dt
            v_ref = args.target_speed if not warmed_up else speed_reference(args, t_probe)
            ctx = SimpleNamespace(t=t_probe, v_x=v_x, v_ref=v_ref, a_x=a_x, a_x_raw=a_x_raw,
                                  gear=vehicle.get_control().gear, warmed_up=warmed_up)
            control_value = controller.step(ctx)

            control = carla.VehicleControl()
            if control_value >= 0:
                control.throttle = control_value
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = -control_value
            control.steer = steer_filter.step(steer)
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            if not warmed_up:
                converged = abs(v_x - args.target_speed) < WARM_START_SPEED_TOL and abs(a_x) < WARM_START_ACCEL_TOL
                timed_out = i * args.dt >= WARM_START_TIMEOUT
                if converged or timed_out:
                    warmed_up = True
                    log_start_i = i
                    controller.reset(control_value)
                    status = "converged" if converged else f"timed out after {WARM_START_TIMEOUT:.0f}s"
                    print(f"[{controller.label}] Warm-start {status}: v_x={v_x:.2f} m/s, "
                          f"a_x={a_x:.2f} m/s^2 -- logging starts now.")
                else:
                    elapsed = time.time() - step_start
                    if elapsed < args.dt / args.times_run:
                        time.sleep(args.dt / args.times_run - elapsed)
                    continue

            # recompute now that log_start_i may have just been updated above -- the t used for
            # the controller's step() earlier in this same iteration was based on the pre-handoff
            # value and would log a stale (much larger) timestamp for this first sample otherwise
            t = (i - log_start_i) * args.dt
            hist["t"].append(t)
            hist["x"].append(ego_x)
            hist["y"].append(ego_y)
            hist["v_x"].append(v_x)
            hist["v_y"].append(v_y)
            hist["v_des"].append(v_ref)
            hist["a_x"].append(a_x)
            hist["jerk"].append(jerk)
            hist["a_y"].append(a_y)
            hist["yaw_rate"].append(yaw_rate)
            hist["yaw_acc"].append(yaw_acc)
            hist["jerk_total"].append(jerk_total)
            hist["last_idx"].append(last_idx)
            hist["steer_deg"].append(steer_deg)
            hist["throttle"].append(control.throttle)
            hist["brake"].append(control.brake)
            hist["e_y"].append(e_y)
            hist["yaw"].append(math.degrees(yaw))
            hist["path_yaw"].append(math.degrees(road_heading))
            hist["e_theta"].append(math.degrees(e_theta))

            if i % 5 == 0:
                extra = f"   a_cmd={ctx.a_cmd:+.2f} m/s^2" if hasattr(ctx, "a_cmd") else ""
                print(f"[{controller.label}] t={t:5.1f}s   global_idx={last_idx}/{len(path_x) - 1}   "
                      f"v_x={v_x:5.1f} m/s{extra}   steer={steer_deg:+.2f} deg   e_y={e_y:+.2f} m")

            if last_idx >= len(path_x) - 1:
                print(f"[{controller.label}] Reached end of path (global_idx {last_idx}/{len(path_x) - 1}).")
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    finally:
        if recorder is not None:
            recorder.close()  # before vehicle.destroy(): the camera is attached to it
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        if bev is not None:
            bev.close()
        vehicle.destroy()

    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--controller", nargs="+", default=["stanley+mpc"],
                        choices=("stanley+pid", "stanley+mpc"),
                        help="which longitudinal controller(s) to pair with Stanley and score; "
                             "pass both to overlay them on one comparison figure")
    parser.add_argument("--target-speed", type=float, default=5, help="m/s")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--goal-tolerance", type=float, default=2.0, help="stop within this many meters of goal (m)")
    parser.add_argument("--times-run", type=float,default=2.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=100.0, help="safety cutoff (s)")

    parser.add_argument("--profile", default="constant", choices=("constant", "sine", "estop"),
                        help="speed reference shape: flat target-speed, a sine wave around it, or "
                             "an emergency stop (drops to 0) that resumes target-speed after --stop-duration")
    parser.add_argument("--sine-amplitude", type=float, default=3.0, help="sine profile peak deviation (m/s)")
    parser.add_argument("--sine-period", type=float, default=3.0, help="sine profile period (s)")
    parser.add_argument("--stop-time", type=float, default=10.0,
                        help="estop profile: when the emergency stop begins, seconds into the scored run")
    parser.add_argument("--stop-duration", type=float, default=5.0,
                        help="estop profile: how long the reference stays at 0 before resuming target-speed (s)")

    # ---- MPC ---- #
    mpc = parser.add_argument_group("--controller stanley+mpc")
    mpc.add_argument("--np", dest="n_p", type=int, default=40, help="prediction horizon (steps)")
    mpc.add_argument("--nc", dest="n_c", type=int, default=40, help="control horizon (steps, <= --np)")
    mpc.add_argument("--w-v", type=float, default=10.0, help="speed-tracking weight")
    mpc.add_argument("--w-a", type=float, default=1, help="commanded-acceleration magnitude weight")
    mpc.add_argument("--w-j", type=float, default=10, help="commanded-acceleration rate (jerk) weight")
    mpc.add_argument("--a-min", type=float, default=-4.05, help="hard lower bound on a_cmd (m/s^2)")
    mpc.add_argument("--a-max", type=float, default=2.4, help="hard upper bound on a_cmd (m/s^2)")
    mpc.add_argument("--lut", default=os.path.join(HERE, "longitudinal_lookup", "longitudinal_lut.npz"))
    mpc.add_argument("--kp", type=float, default=0.15, help="accel-tracking PID proportional gain")
    mpc.add_argument("--ki", type=float, default=0.6, help="accel-tracking PID integral gain")
    mpc.add_argument("--kd", type=float, default=0.0, help="accel-tracking PID derivative gain")
    mpc.add_argument("--u-tau", type=float, default=0.02,
                     help="low-pass filter time constant on the pedal command u, before it's applied (s)")

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"),
                         help="directory to save the end-of-run result figures into")
    parser.add_argument("--save-plot", action="store_true",
                        help="draw and save the end-of-run result figures (off by default)")
    parser.add_argument("--no-live-view", action="store_true", help="skip the live BEV plan/vehicle view")

    # ---- video ---- #
    parser.add_argument("--record", nargs="?", const="auto", default="",
                        help="record the drive to an mp4; bare flag auto-names it under --video-dir")
    parser.add_argument("--video-dir", default=os.path.join(HERE, "videos"),
                        help="where auto-named recordings go")
    parser.add_argument("--record-view", default="chase", choices=sorted(VIEWS),
                        help="camera mount for the recording")
    parser.add_argument("--record-res", default="1280x720", help="recording resolution, WxH")
    args = parser.parse_args()

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

    origin_transform, path_x, path_y, path_yaw = build_path(world)
    print(f"Route: {len(path_x)} points, "
          f"start=({path_x[0]:.1f}, {path_y[0]:.1f}) goal=({path_x[-1]:.1f}, {path_y[-1]:.1f})")

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    keys = list(dict.fromkeys(args.controller))   # de-dupe, keep the order given on the CLI

    def recorder_factory(key, n_trials):
        if not args.record:
            return lambda vehicle: None
        suffix = key.replace("+", "-") if n_trials > 1 else ""

        def make(vehicle):
            if args.record == "auto" or n_trials > 1:
                path = os.path.join(args.video_dir, run_name(suffix) + ".mp4")
            else:
                path = args.record
            rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
            return VideoRecorder(world, vehicle, path, fps=1.0 / args.dt, width=rec_w, height=rec_h,
                                 view=args.record_view)
        return make

    results = {}
    try:
        for key in keys:
            controller = LONGITUDINAL[key](args)
            print(f"\n=== running {controller.label} ===")
            results[controller.label] = run_trial(world, origin_transform, path_x, path_y, path_yaw,
                                                   blueprint, imu_bp, controller, args,
                                                   recorder_factory(key, len(keys)))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("Cleaned up: world settings restored.")

    have_data = all(len(hist["t"]) > 1 for hist in results.values())
    if not args.save_plot or not have_data:
        for label, hist in results.items():
            if len(results) > 1:
                print(f"\n### {label} ###")
            print_error_summary(hist, args.target_speed)  # plot_results prints it otherwise
    else:
        data = results if len(results) > 1 else next(iter(results.values()))
        title = " vs ".join(results) if len(results) > 1 else next(iter(results), "")
        try:
            plot_results(path_x, path_y, data, args.target_speed, args.plot_dir,
                         label=f"Stanley + {title}")
        except Exception as exc:
            print(f"Plotting failed: {exc}")
            for label, hist in results.items():
                print_error_summary(hist, args.target_speed)


if __name__ == "__main__":
    main()
