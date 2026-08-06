"""Longitudinal control-input calibration sweep.

Builds the raw data for a (gear, v_x, a_x) -> u lookup table, where u in
[-1, 1] is the longitudinal control input (positive = throttle, negative =
brake). For each gear, drives at a constant u from rest up to terminal speed
(throttle) or from a high speed down to rest (brake), logging (v_x, a_x)
every tick. One trial per (gear, u) covers the full speed range for that
pair, so this is much cheaper than gridding over (gear, v, u) directly.

Output is a CSV of raw samples -- inverting it into a gear/v/a -> u table
for MPC lookup is a separate step (e.g. scipy.interpolate over this data).

Usage:
    cd ~/carla_control
    python3 build_longitudinal_lut.py --out longitudinal_lut.csv
"""

import argparse
import csv
import math
import sys

sys.path.append("/home/ailab/carla/CARLA_0.9.15/PythonAPI/carla")

import carla


def body_frame_long_state(vehicle):
    """Project world-frame velocity/acceleration onto the vehicle's forward axis."""
    yaw = math.radians(vehicle.get_transform().rotation.yaw)
    vel = vehicle.get_velocity()
    acc = vehicle.get_acceleration()
    v_x = vel.x * math.cos(yaw) + vel.y * math.sin(yaw)
    a_x = acc.x * math.cos(yaw) + acc.y * math.sin(yaw)
    return v_x, a_x


def apply_long_control(vehicle, gear, u):
    control = carla.VehicleControl()
    control.manual_gear_shift = True
    control.gear = gear
    control.reverse = False
    control.hand_brake = False
    if u >= 0:
        control.throttle = u
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = -u
    vehicle.apply_control(control)


def reset_trial(world, vehicle, origin_transform, start_speed_ms):
    vehicle.set_transform(origin_transform)
    yaw = math.radians(origin_transform.rotation.yaw)
    vehicle.set_target_velocity(carla.Vector3D(
        x=start_speed_ms * math.cos(yaw),
        y=start_speed_ms * math.sin(yaw),
        z=0.0,
    ))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    world.tick()
    world.tick()  # let the teleport/velocity settle before the trial's control kicks in


def run_throttle_trial(world, vehicle, origin_transform, gear, u, dt, args, writer):
    reset_trial(world, vehicle, origin_transform, 0.0)
    start_loc = vehicle.get_transform().location
    plateau_count = 0
    t = 0.0
    while t < args.max_duration:
        apply_long_control(vehicle, gear, u)
        world.tick()
        v_x, a_x = body_frame_long_state(vehicle)
        writer.writerow([gear, u, round(t, 3), round(v_x, 4), round(a_x, 4)])

        if vehicle.get_transform().location.distance(start_loc) > args.max_distance:
            break
        if a_x < args.plateau_eps:
            plateau_count += 1
            if plateau_count > args.plateau_ticks:
                break
        else:
            plateau_count = 0
        t += dt


def run_brake_trial(world, vehicle, origin_transform, gear, u, dt, args, writer):
    reset_trial(world, vehicle, origin_transform, args.brake_start_speed)
    start_loc = vehicle.get_transform().location
    t = 0.0
    while t < args.max_duration:
        apply_long_control(vehicle, gear, u)
        world.tick()
        v_x, a_x = body_frame_long_state(vehicle)
        writer.writerow([gear, u, round(t, 3), round(v_x, 4), round(a_x, 4)])

        if vehicle.get_transform().location.distance(start_loc) > args.max_distance:
            break
        if v_x < args.min_speed:
            break
        t += dt


def frange(start, stop, step):
    n = round((stop - start) / step)
    return [round(start + i * step, 4) for i in range(n + 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--out", default="longitudinal_lut.csv")
    parser.add_argument("--gears", default="", help="comma-separated gear list; default = all forward gears reported by the vehicle physics")
    parser.add_argument("--throttle-min", type=float, default=0.1)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--throttle-step", type=float, default=0.1)
    parser.add_argument("--brake-min", type=float, default=0.1)
    parser.add_argument("--brake-max", type=float, default=1.0)
    parser.add_argument("--brake-step", type=float, default=0.1)
    parser.add_argument("--brake-start-speed", type=float, default=30.0, help="starting speed for every brake trial (m/s)")
    parser.add_argument("--max-duration", type=float, default=15.0, help="per-trial cutoff (s)")
    parser.add_argument("--max-distance", type=float, default=300.0, help="per-trial cutoff (m); keeps a trial on one straight")
    parser.add_argument("--plateau-eps", type=float, default=0.05, help="|a_x| below this counts as 'no longer accelerating' (m/s^2)")
    parser.add_argument("--plateau-ticks", type=int, default=40, help="consecutive plateau ticks before ending a throttle trial early")
    parser.add_argument("--min-speed", type=float, default=0.2, help="brake trial ends once v_x drops below this (m/s)")
    parser.add_argument("--origin-index", type=int, default=0, help="spawn point index; pick one on a long straight")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    spawn_points = world.get_map().get_spawn_points()
    origin_transform = spawn_points[args.origin_index]

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    vehicle = world.spawn_actor(blueprint, origin_transform)
    world.tick()

    if args.gears:
        gears = [int(g) for g in args.gears.split(",")]
    else:
        gears = list(range(1, len(vehicle.get_physics_control().forward_gears) + 1))
    print(f"Gears to sweep: {gears}")

    throttles = frange(args.throttle_min, args.throttle_max, args.throttle_step)
    brakes = [-b for b in frange(args.brake_min, args.brake_max, args.brake_step)]
    print(f"Throttle grid: {throttles}")
    print(f"Brake grid: {brakes}")

    total_trials = len(gears) * (len(throttles) + len(brakes))
    done = 0

    try:
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gear", "u", "t", "v_x", "a_x"])

            for gear in gears:
                for u in throttles:
                    done += 1
                    print(f"[{done}/{total_trials}] gear={gear} u={u:+.2f} (throttle sweep)")
                    run_throttle_trial(world, vehicle, origin_transform, gear, u, args.dt, args, writer)

                for u in brakes:
                    done += 1
                    print(f"[{done}/{total_trials}] gear={gear} u={u:+.2f} (brake sweep)")
                    run_brake_trial(world, vehicle, origin_transform, gear, u, args.dt, args, writer)
    finally:
        vehicle.destroy()
        world.apply_settings(original_settings)
        print(f"Cleaned up. Data saved to {args.out}")


if __name__ == "__main__":
    main()
