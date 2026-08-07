"""Stanley lateral controller + PID longitudinal controller.

Path source: CARLA has no built-in "desired path" object.
agents.navigation.GlobalRoutePlanner runs A* over the map's waypoint topology
between an origin and destination to produce one.

Usage:
    cd ~/carla_control
    .venv/bin/python stanley_PID.py --times-run 1 --target-speed 15
    .venv/bin/python stanley_PID.py --times-run 10 --save-plot --record --target-speed 15
"""

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.append("/home/ailab/2026intern/carla/PythonAPI/carla")

import carla

from functions import (PID, AngleUnwrapper, LowPassFilter, build_path, clipping,
                       get_vehicle_geometry, lateral_error, normalize_angle)
from viz_utils import (BevView, VIEWS, VideoRecorder, follow_with_spectator, plot_results,
                       print_error_summary, run_name)


def front_axle_offset(vehicle, lf):
    """How far ahead of the actor origin the front axle sits, in the body frame (m).

    Stanley is defined at the front axle, but transform.location is the actor origin, which sits
    between the axles -- tracking the origin instead is a constant phantom cross-track error in
    every corner. lf is measured from the centre of mass, so add where the centre of mass sits
    relative to the origin to land on the offset from the origin itself.
    """
    return vehicle.get_physics_control().center_of_mass.x + lf


def stanley_control(v_x, e_y, e_theta, k_theta=1.0, k=1.2, k_soft=1.0):
    """Stanley steering law (Hoffmann et al. 2007): delta = e_theta + atan(k * e_y / v).

    k_theta is 1.0 because the canonical law applies no gain to the heading term at all.

    k_soft is the paper's softening constant. Without it the atan2 denominator reaches zero at
    standstill and the law returns exactly +-90 deg for any nonzero e_y -- a 0.5 m offset pins the
    steering at the clip, which is what locks the wheel over after the vehicle stops.
    """
    return k_theta * e_theta + math.atan2(k * e_y, k_soft + abs(v_x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--target-speed", type=float, default=5, help="m/s")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--goal-tolerance", type=float, default=2.0, help="stop within this many meters of goal (m)")
    parser.add_argument("--times-run", type=float,default=2.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=100.0, help="safety cutoff (s)")
    parser.add_argument("--warm-start", type=float, default=0.5,
                        help="hold the spawn speed for this long (s) while the drivetrain spins up; 0 disables")
    parser.add_argument("--warm-start-gear", type=int, default=3,
                        help="gear to hold during the warm start; 0 leaves shifting to CARLA")

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
    client.set_timeout(30.0)
    world = client.get_world()

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

    initial_yaw = math.radians(origin_transform.rotation.yaw)
    vehicle.set_target_velocity(carla.Vector3D(
        x=args.target_speed * math.cos(initial_yaw),
        y=args.target_speed * math.sin(initial_yaw),
        z=0.0,
    ))

    wheelbase, lf, lr, max_steer = get_vehicle_geometry(vehicle, origin_transform)
    front_offset = front_axle_offset(vehicle, lf)
    if not 0.0 < front_offset < wheelbase:
        raise RuntimeError(f"front_offset={front_offset:.2f} m is not inside the wheelbase "
                           f"({wheelbase:.2f} m); the front-axle reference point is wrong.")
    print(f"front_axle_offset={front_offset:.2f} m")

    pid = PID(kp=0.7, ki=0.15, kd=0.05, dt=args.dt)
    speed_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0.0)
    steer_filter = LowPassFilter(tau=0.01, dt=args.dt, initial=0)
    yaw_unwrapper = AngleUnwrapper()
    rh_unwrapper = AngleUnwrapper()
    last_idx = 0

    accel_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    jerk_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    prev_v_x = None
    prev_a_x = 0.0

    warm_start_steps = int(args.warm_start / args.dt)

    bev = None if args.no_live_view else BevView(path_x, path_y)

    recorder = None
    if args.record:
        # same <script>_<date>_<time> stem the figures get, so the mp4 sits next to its plots
        video_path = (os.path.join(args.video_dir, run_name() + ".mp4")
                      if args.record == "auto" else args.record)
        rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
        recorder = VideoRecorder(world, vehicle, video_path, fps=1.0 / args.dt,
                                 width=rec_w, height=rec_h, view=args.record_view)

    hist = {"t": [], "x": [], "y": [], "v_x": [], "v_y": [], "v_des": [], "a_x": [], "jerk": [],
            "yaw_rate": [], "last_idx": [], "steer_deg": [], "throttle": [], "brake": [],
            "e_y": [], "yaw": [], "path_yaw": [], "e_theta": []}

    try:
        world.tick()
        steps = int(args.max_duration / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()

            transform = vehicle.get_transform()
            yaw = yaw_unwrapper.step(math.radians(transform.rotation.yaw))  # continuous from here on out
            ego_x = transform.location.x
            ego_y = transform.location.y
            vel_vec = vehicle.get_velocity()
            # body-frame velocities (not vel_vec.x/.y, which are world-frame)
            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)
            v_y = -vel_vec.x * math.sin(yaw) + vel_vec.y * math.cos(yaw)
            yaw_rate = vehicle.get_angular_velocity().z  # CARLA reports deg/s

            # first step has no previous sample to difference against
            a_x = accel_filter.step(0.0 if prev_v_x is None else (v_x - prev_v_x) / args.dt)
            jerk = jerk_filter.step(0.0 if prev_v_x is None else (a_x - prev_a_x) / args.dt)
            prev_v_x, prev_a_x = v_x, a_x

            if i < warm_start_steps:
                vehicle.set_target_velocity(carla.Vector3D(x=args.target_speed * math.cos(yaw),
                                                           y=args.target_speed * math.sin(yaw),
                                                           z=0.0))

            # Stanley tracks the front axle, not the actor origin
            front_x = ego_x + front_offset * math.cos(yaw)
            front_y = ego_y + front_offset * math.sin(yaw)

            # index first, then read the path off it -- reading before the update fed the controller
            # the previous step's road heading, a free half-metre of lag at 15 m/s
            last_idx, e_y = lateral_error(front_x, front_y, yaw, path_x, path_y, last_idx)
            road_heading = rh_unwrapper.step(path_yaw[last_idx])
            # both traces are unwrapped, so normalize_angle recovers the true error either way
            e_theta = normalize_angle(path_yaw[last_idx] - yaw)
            e_vel = args.target_speed - v_x

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
            if i < warm_start_steps and args.warm_start_gear:
                control.manual_gear_shift = True
                control.gear = args.warm_start_gear
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            t = i * args.dt
            hist["t"].append(t)
            hist["x"].append(ego_x)
            hist["y"].append(ego_y)
            hist["v_x"].append(v_x)
            hist["v_y"].append(v_y)
            hist["v_des"].append(args.target_speed)
            hist["a_x"].append(a_x)
            hist["jerk"].append(jerk)
            hist["yaw_rate"].append(yaw_rate)
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