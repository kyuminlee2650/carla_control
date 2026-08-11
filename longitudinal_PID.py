r"""PID longitudinal cruise control, steer forced to 0 -- no Stanley, no desired path.

Isolates the longitudinal loop from stanley_PID.py to check it on its own against a speed
reference: a flat initial-speed, a sine wave around it, or a step away from it partway through the
run. With steer pinned at 0 there is no lateral correction, so the spawn point is fixed to the
longest straight on a fixed map -- on a curve the car would drift off the road and the run would
measure a wall hit instead of the powertrain.

Usage (Ubuntu):
    cd ~/carla_control
    python3 longitudinal_PID.py --profile constant --initial-speed 15
    python3 longitudinal_PID.py --profile sine --initial-speed 15 --sine-amplitude 3 --sine-period 10 --save-plot
    python3 longitudinal_PID.py --profile step --initial-speed 15 --step-size 5 --step-time 10 --save-plot

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe longitudinal_PID.py --profile constant --initial-speed 15
    .venv\Scripts\python.exe longitudinal_PID.py --profile sine --initial-speed 15 --sine-amplitude 3 --sine-period 10
    .venv\Scripts\python.exe longitudinal_PID.py --profile constant --initial-speed 15 --times-run 10 --save-plot --record
    .venv\Scripts\python.exe longitudinal_PID.py --profile sine --initial-speed 15 --times-run 10 --sine-amplitude 5 --sine-period 10 --save-plot --record
    .venv\Scripts\python.exe longitudinal_PID.py --profile step --initial-speed 15 --step-size -5 --step-time 10 --save-plot
"""

import argparse
import math
import os
import queue
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# see functions.py for why this path is needed alongside the pip-installed carla package
CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/carla/CARLA_0.9.15"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

import carla

from functions import PID, ImuAcceleration, LowPassFilter, clipping
from viz_utils import (VIEWS, VideoRecorder, follow_with_spectator, plot_longitudinal_result,
                       print_error_summary, run_name)


MAP_NAME = "Town06"
ORIGIN_INDEX = 86


def speed_reference(args, t):
    if args.profile == "sine":
        return args.initial_speed + args.sine_amplitude * math.sin(2.0 * math.pi * t / args.sine_period)
    if args.profile == "step":
        return args.initial_speed + (args.step_size if t >= args.step_time else 0.0)
    return args.initial_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--initial-speed", type=float, default=15.0,
                        help="m/s; starting speed, also the sine profile's midline and the step "
                             "profile's pre-step level")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--duration", type=float, default=30.0, help="scored run length (s)")
    parser.add_argument("--times-run", type=float, default=2.0, help="how times for simulation running?")

    parser.add_argument("--profile", default="constant", choices=("constant", "sine", "step"),
                        help="speed reference shape: flat initial-speed, a sine wave around it, "
                             "or a single step away from it partway through the run")
    parser.add_argument("--sine-amplitude", type=float, default=3.0, help="sine profile peak deviation (m/s)")
    parser.add_argument("--sine-period", type=float, default=10.0, help="sine profile period (s)")
    parser.add_argument("--step-size", type=float, default=5.0,
                        help="step profile: m/s added to initial-speed after --step-time (negative = "
                             "a deceleration step)")
    parser.add_argument("--step-time", type=float, default=10.0,
                        help="step profile: when the step happens, seconds into the scored run")

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"),
                         help="directory to save the end-of-run result figures into")
    parser.add_argument("--save-plot", action="store_true",
                        help="draw and save the end-of-run result figures (off by default)")

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

    origin_transform = world.get_map().get_spawn_points()[ORIGIN_INDEX]

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    vehicle = world.spawn_actor(blueprint, origin_transform)

    pid = PID(kp=0.7, ki=0.15, kd=0.05, dt=args.dt)
    speed_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0.0)

    # a_x/a_y off the IMU -- see stanley_PID.py for why (true body-frame values straight from the
    # sensor, nothing to derive by hand). a_y only exists here to feed jerk_total; nothing plots it
    # on its own since there's no lateral figure in a steer=0 run.
    accel = ImuAcceleration(dt=args.dt)

    recorder = None
    if args.record:
        video_path = (os.path.join(args.video_dir, run_name() + ".mp4")
                      if args.record == "auto" else args.record)
        rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
        recorder = VideoRecorder(world, vehicle, video_path, fps=1.0 / args.dt,
                                 width=rec_w, height=rec_h, view=args.record_view)

    hist = {"t": [], "v_x": [], "v_des": [], "a_x": [], "jerk": [], "jerk_total": [],
            "throttle": [], "brake": []}

    # Same warm-up gate as stanley_PID.py: launch from rest under the real PID and hold off on
    # logging until v_x/a_x have actually settled near the profile's own t=0 value (initial_speed,
    # since sin(0) = 0 for sine and the step hasn't happened yet at t=0 for step) instead of faking
    # that starting condition.
    warmed_up = False
    log_start_i = 0
    WARM_START_SPEED_TOL = 0.3   # m/s
    WARM_START_ACCEL_TOL = 0.5   # m/s^2
    WARM_START_TIMEOUT = 15.0    # s -- safety cap in case initial_speed is unreachable

    imu = None
    try:
        world.tick()

        # spawned after the priming tick above, so the first world.tick() in the loop below
        # produces this sensor's first queued sample -- one put() per get() keeps them in
        # lockstep for the rest of the run.
        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)

        steps = int((args.duration + WARM_START_TIMEOUT) / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)

            transform = vehicle.get_transform()
            yaw = math.radians(transform.rotation.yaw)
            vel_vec = vehicle.get_velocity()
            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)  # body-frame forward speed

            accel.step(imu_data)
            a_x, a_y = accel.a_x, accel.a_y

            jerk, jerk_total = accel.jerk, accel.jerk_total

            v_ref = args.initial_speed if not warmed_up else speed_reference(args, (i - log_start_i) * args.dt)
            e_vel = v_ref - v_x
            control_value = clipping(speed_filter.step(pid.step(e_vel)), 1, -1)

            control = carla.VehicleControl()
            if control_value >= 0:
                control.throttle = control_value
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = -control_value
            control.steer = 0.0
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            if not warmed_up:
                converged = abs(v_x - args.initial_speed) < WARM_START_SPEED_TOL and abs(a_x) < WARM_START_ACCEL_TOL
                timed_out = i * args.dt >= WARM_START_TIMEOUT
                if converged or timed_out:
                    warmed_up = True
                    log_start_i = i
                    status = "converged" if converged else f"timed out after {WARM_START_TIMEOUT:.0f}s"
                    print(f"Warm-start {status}: v_x={v_x:.2f} m/s, a_x={a_x:.2f} m/s^2 -- logging starts now.")
                else:
                    elapsed = time.time() - step_start
                    if elapsed < args.dt / args.times_run:
                        time.sleep(args.dt / args.times_run - elapsed)
                    continue

            t = (i - log_start_i) * args.dt
            hist["t"].append(t)
            hist["v_x"].append(v_x)
            hist["v_des"].append(v_ref)
            hist["a_x"].append(a_x)
            hist["jerk"].append(jerk)
            hist["jerk_total"].append(jerk_total)
            hist["throttle"].append(control.throttle)
            hist["brake"].append(control.brake)

            if i % 5 == 0:
                print(f"t={t:5.1f}s   v_x={v_x:5.2f} m/s   v_ref={v_ref:5.2f} m/s   e_vel={e_vel:+.2f} m/s")

            if t >= args.duration:
                print(f"Reached duration ({args.duration:.0f}s).")
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if recorder is not None:
            recorder.close()  # before vehicle.destroy(): the camera is attached to it
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        vehicle.destroy()
        world.apply_settings(original_settings)
        print("Cleaned up: vehicle destroyed, world settings restored.")

    if not args.save_plot or len(hist["t"]) <= 1:
        print_error_summary(hist, args.initial_speed)  # plot_longitudinal_result prints it otherwise
    else:
        try:
            plot_longitudinal_result(hist, args.initial_speed, args.plot_dir,
                                     label=f"Longitudinal PID ({args.profile})")
        except Exception as exc:
            print(f"Plotting failed: {exc}")
            print_error_summary(hist, args.initial_speed)


if __name__ == "__main__":
    main()
