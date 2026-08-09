r"""Stanley lateral controller + PID longitudinal controller.

Path source: CARLA has no built-in "desired path" object.
agents.navigation.GlobalRoutePlanner runs A* over the map's waypoint topology
between an origin and destination to produce one.

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python stanley_PID.py --times-run 1 --target-speed 15
    .venv/bin/python stanley_PID.py --times-run 20 --save-plot --record --target-speed 15

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe stanley_PID.py --times-run 1 --target-speed 15
    .venv\Scripts\python.exe stanley_PID.py --times-run 20 --save-plot --record --target-speed 15
    .venv\Scripts\python.exe stanley_PID.py --times-run 20 --save-plot --record --target-speed 15 --profile sine
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
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/2026intern/carla"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

import carla

from functions import (PID, AngleUnwrapper, LowPassFilter, build_path, clipping,
                       get_vehicle_geometry, lateral_error, normalize_angle)
from viz_utils import (BevView, VIEWS, VideoRecorder, follow_with_spectator, plot_results,
                       print_error_summary, run_name)

# Fixed on purpose: the route this controller has been tuned against (build_path()'s default
# origin/dest spawn indices) is a property of this specific map, not something the controller
# should have to rediscover on whatever map the server happens to have loaded.
MAP_NAME = "Town10HD_Opt"


def front_axle_offset(vehicle, lf):
    return vehicle.get_physics_control().center_of_mass.x + lf


def stanley_control(v_x, e_y, e_theta, k_theta=1.0, k=1.2, k_soft=1.0):
    return k_theta * e_theta + math.atan2(k * e_y, k_soft + abs(v_x))


def speed_reference(args, t):
    if args.profile == "sine":
        return args.target_speed + args.sine_amplitude * math.sin(2.0 * math.pi * t / args.sine_period)
    return args.target_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--target-speed", type=float, default=5, help="m/s")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--goal-tolerance", type=float, default=2.0, help="stop within this many meters of goal (m)")
    parser.add_argument("--times-run", type=float,default=2.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=100.0, help="safety cutoff (s)")

    parser.add_argument("--profile", default="constant", choices=("constant", "sine"),
                        help="speed reference shape: flat target-speed, or a sine wave around it")
    parser.add_argument("--sine-amplitude", type=float, default=5.0, help="sine profile peak deviation (m/s)")
    parser.add_argument("--sine-period", type=float, default=10.0, help="sine profile period (s)")

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
    vehicle = world.spawn_actor(blueprint, origin_transform)

    wheelbase, lf, lr, max_steer = get_vehicle_geometry(vehicle, origin_transform)
    front_offset = front_axle_offset(vehicle, lf)
    if not 0.0 < front_offset < wheelbase:
        raise RuntimeError(f"front_offset={front_offset:.2f} m is not inside the wheelbase "
                           f"({wheelbase:.2f} m); the front-axle reference point is wrong.")
    print(f"front_axle_offset={front_offset:.2f} m")

    pid = PID(kp=0.7, ki=0.15, kd=0.05, dt=args.dt)
    speed_filter = LowPassFilter(tau=0.2, dt=args.dt, initial=0.0)
    steer_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0)
    yaw_unwrapper = AngleUnwrapper()
    rh_unwrapper = AngleUnwrapper()
    last_idx = 0


    accel_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    jerk_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    accel_y_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    jerk_y_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    yaw_acc_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    prev_a_x = None
    prev_a_y = None
    prev_yaw_rate_rad = None

    bev = None if args.no_live_view else BevView(path_x, path_y)

    recorder = None
    if args.record:
        video_path = (os.path.join(args.video_dir, run_name() + ".mp4")
                      if args.record == "auto" else args.record)
        rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
        recorder = VideoRecorder(world, vehicle, video_path, fps=1.0 / args.dt,
                                 width=rec_w, height=rec_h, view=args.record_view)

    hist = {"t": [], "x": [], "y": [], "v_x": [], "v_y": [], "v_des": [], "a_x": [], "jerk": [],
            "a_y": [], "yaw_rate": [], "yaw_acc": [], "jerk_total": [], "last_idx": [],
            "steer_deg": [], "throttle": [], "brake": [], "e_y": [], "yaw": [], "path_yaw": [],
            "e_theta": []}

    warmed_up = False
    log_start_i = 0
    WARM_START_SPEED_TOL = 0.3   # m/s
    WARM_START_ACCEL_TOL = 0.5   # m/s^2
    WARM_START_TIMEOUT = 15.0    # s -- safety cap in case target_speed is unreachable

    imu = None
    try:
        world.tick()

        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        print("IMU noise (0 = clean): "
              f"accel_stddev=({imu_bp.get_attribute('noise_accel_stddev_x').as_float()}, "
              f"{imu_bp.get_attribute('noise_accel_stddev_y').as_float()}, "
              f"{imu_bp.get_attribute('noise_accel_stddev_z').as_float()})  "
              f"gyro_stddev=({imu_bp.get_attribute('noise_gyro_stddev_x').as_float()}, "
              f"{imu_bp.get_attribute('noise_gyro_stddev_y').as_float()}, "
              f"{imu_bp.get_attribute('noise_gyro_stddev_z').as_float()})")
        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)

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

            a_x = accel_filter.step(imu_data.accelerometer.x)
            a_y = accel_y_filter.step(imu_data.accelerometer.y)
            yaw_rate_rad = imu_data.gyroscope.z
            yaw_rate = math.degrees(yaw_rate_rad)  

            jerk = jerk_filter.step(0.0 if prev_a_x is None else (a_x - prev_a_x) / args.dt)
            prev_a_x = a_x

            jerk_y = jerk_y_filter.step(0.0 if prev_a_y is None else (a_y - prev_a_y) / args.dt)
            prev_a_y = a_y
            jerk_total = math.hypot(jerk, jerk_y)

            yaw_acc = yaw_acc_filter.step(
                0.0 if prev_yaw_rate_rad is None else (yaw_rate_rad - prev_yaw_rate_rad) / args.dt)
            prev_yaw_rate_rad = yaw_rate_rad


            # ---- Stanley + PID ---- #
            front_x = ego_x + front_offset * math.cos(yaw)
            front_y = ego_y + front_offset * math.sin(yaw)

            last_idx, e_y = lateral_error(front_x, front_y, yaw, path_x, path_y, last_idx)
            road_heading = rh_unwrapper.step(path_yaw[last_idx])

            e_theta = normalize_angle(path_yaw[last_idx] - yaw)
            v_ref = args.target_speed if not warmed_up else speed_reference(args, (i - log_start_i) * args.dt)
            e_vel = v_ref - v_x

            delta = stanley_control(v_x, e_y, e_theta)
            steer = clipping(delta / max_steer, 3 / 7, -3 / 7)
            steer_deg = steer * math.degrees(max_steer)

            control_value = clipping(speed_filter.step(pid.step(e_vel)), 1, -1)

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
                    status = "converged" if converged else f"timed out after {WARM_START_TIMEOUT:.0f}s"
                    print(f"Warm-start {status}: v_x={v_x:.2f} m/s, a_x={a_x:.2f} m/s^2 -- logging starts now.")
                else:
                    elapsed = time.time() - step_start
                    if elapsed < args.dt / args.times_run:
                        time.sleep(args.dt / args.times_run - elapsed)
                    continue

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
                print(f"t={t:5.1f}s   global_idx={last_idx}/{len(path_x) - 1}   v_x={v_x:5.1f} m/s   steer={steer_deg:+.2f} deg   e_y={e_y:+.2f} m")


            if last_idx >= len(path_x) - 1:
                print(f"Reached end of path (global_idx {last_idx}/{len(path_x) - 1}).")
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
        if bev is not None:
            bev.close()
        vehicle.destroy()
        world.apply_settings(original_settings)
        print("Cleaned up: vehicle destroyed, world settings restored.")

    if not args.save_plot or len(hist["t"]) <= 1:
        print_error_summary(hist, args.target_speed)  # plot_results prints it otherwise
    else:
        try:
            plot_results(path_x, path_y, hist, args.target_speed, args.plot_dir,
                         label="Stanley + PID")
        except Exception as exc:
            print(f"Plotting failed: {exc}")
            print_error_summary(hist, args.target_speed)


if __name__ == "__main__":
    main()