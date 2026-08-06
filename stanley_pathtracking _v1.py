"""Stanley lateral controller + PID longitudinal controller.

Path source: CARLA has no built-in "desired path" object.
agents.navigation.GlobalRoutePlanner runs A* over the map's waypoint topology
between an origin and destination to produce one.

Usage:
    cd ~/carla_control
    python3 stanley_pathtracking.py --target-speed 20 --origin-index 0 --dest-index 50
"""
import argparse
import math
import os
import sys
import time
import matplotlib.pyplot as plt

sys.path.append("/home/ailab/carla/CARLA_0.9.15/PythonAPI/carla")

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

from viz_utils import follow_with_spectator, plot_results


class PID:
    def __init__(self, kp, ki, kd, dt, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.out_min, self.out_max = out_min, out_max
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error):
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        out = self.kp * error + self.ki * self._integral + self.kd * derivative

        return out


class LowPassFilter:
    def __init__(self, tau, dt, initial=0.0):
        self.alpha = dt / (tau + dt)
        self.state = initial

    def step(self, x):
        self.state += self.alpha * (x - self.state)
        return self.state


def cliping(value,max_val,min_val):
    return max(min_val,min(value,max_val))

class AngleUnwrapper:
    """Turns consecutive [-pi, pi]-wrapped angle readings into a continuous trace.

    e.g. pi -> pi+0.01 instead of jumping to -pi+0.01, so plots/derivatives don't spike at the wrap.
    """
    def __init__(self):
        self._prev_raw = None
        self._unwrapped = None

    def step(self, raw_angle):
        if self._unwrapped is None:
            self._unwrapped = raw_angle
        else:
            self._unwrapped += normalize_angle(raw_angle - self._prev_raw)
        self._prev_raw = raw_angle
        return self._unwrapped


def speed_ms(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2+v.y**2)


def build_path(world, sampling_resolution=1, origin_index=0, dest_index=70):
    """Trace a route with GlobalRoutePlanner and flatten it into x/y/yaw arrays."""
    spawn_points = world.get_map().get_spawn_points()
    dest_index = min(dest_index, len(spawn_points) - 1)
    if dest_index == origin_index:
        dest_index = (origin_index + 1) % len(spawn_points)

    origin_transform = spawn_points[origin_index]
    destination = spawn_points[dest_index].location

    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)
    route = grp.trace_route(origin_transform.location, destination)
    if len(route) < 2:
        raise RuntimeError("Route too short; pick a farther --dest-index.")

    path_x = [wp.transform.location.x for wp, _ in route]
    path_y = [wp.transform.location.y for wp, _ in route]
    path_yaw = [math.radians(wp.transform.rotation.yaw) for wp, _ in route]
    path_yaw = unwrap_angles(path_yaw)
    return origin_transform, path_x, path_y, path_yaw


def normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def unwrap_angles(angles):
    """Unwrap a sequence of [-pi, pi] angles so consecutive values don't jump across the +-pi boundary."""
    unwrapped = [angles[0]]
    for prev_raw, raw in zip(angles, angles[1:]):
        unwrapped.append(unwrapped[-1] + normalize_angle(raw - prev_raw))
    return unwrapped


def get_vehicle_geometry(vehicle):
    """Wheelbase (m) and max front-wheel steer angle (rad) from the spawned vehicle's physics."""
    physics = vehicle.get_physics_control()
    wheels = physics.wheels  # order: [front_left, front_right, rear_left, rear_right]
    front_mid = carla.Vector3D(
        (wheels[0].position.x + wheels[1].position.x) / 2.0,
        (wheels[0].position.y + wheels[1].position.y) / 2.0,
        (wheels[0].position.z + wheels[1].position.z) / 2.0,
    )
    rear_mid = carla.Vector3D(
        (wheels[2].position.x + wheels[3].position.x) / 2.0,
        (wheels[2].position.y + wheels[3].position.y) / 2.0,
        (wheels[2].position.z + wheels[3].position.z) / 2.0,
    )
    # wheel positions are in cm
    wheelbase = front_mid.distance(rear_mid) / 100.0
    max_steer_deg = (wheels[0].max_steer_angle + wheels[1].max_steer_angle) / 2.0
    return wheelbase, math.radians(max_steer_deg)

def stanley_control(vehicle, wheelbase, path_x, path_y, path_yaw, last_idx, search_window=5,k_theta=0.6,k=1):
    """Front-axle Stanley steering: target index, steer angle (rad), front axle (x, y), cross-track error (m)."""
    transform = vehicle.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    fx = transform.location.x + wheelbase * math.cos(yaw)
    fy = transform.location.y + wheelbase * math.sin(yaw)

    lo = last_idx
    hi = min(len(path_x), last_idx + search_window)
    dists = [math.hypot(fx - path_x[i], fy - path_y[i]) for i in range(lo, hi)]
    target_idx = lo + dists.index(min(dists))

    dx = path_x[target_idx] - fx
    dy = path_y[target_idx] - fy
    error_front_axle = -math.sin(yaw) * dx + math.cos(yaw) * dy

    theta_e = k_theta*normalize_angle(path_yaw[target_idx] - yaw)
    theta_d = math.atan2(k * error_front_axle, speed_ms(vehicle))
    delta = theta_e + theta_d
    return target_idx, delta, (fx, fy), error_front_axle, yaw, theta_e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--target-speed", type=float, default=15*3.6, help="km/h")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--goal-tolerance", type=float, default=2.0, help="stop within this many meters of goal (m)")
    parser.add_argument("--times-run", type=float,default=2.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=30.0, help="safety cutoff (s)")
    parser.add_argument("--plot-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                         help="directory to save the end-of-run result figure into")
    parser.add_argument("--no-plot", action="store_true", help="skip saving the result figure")
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

    blueprint = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(blueprint, origin_transform)

    # start already at target speed so the run tests steering, not the accel ramp
    initial_yaw = math.radians(origin_transform.rotation.yaw)
    initial_speed_ms = args.target_speed / 3.6
    vehicle.set_target_velocity(carla.Vector3D(
        x=initial_speed_ms * math.cos(initial_yaw),
        y=initial_speed_ms * math.sin(initial_yaw),
        z=0.0,
    ))

    wheelbase, max_steer = get_vehicle_geometry(vehicle)
    print(wheelbase,max_steer)
    print(f"wheelbase={wheelbase:.2f} m  max_steer={math.degrees(max_steer):.1f} deg")

    pid = PID(kp=0.35, ki=0.15, kd=0.05, dt=args.dt)
    speed_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0.0)
    steer_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0)
    yaw_unwrapper = AngleUnwrapper()
    last_idx = 0
    hist = {"t": [], "x": [], "y": [], "speed": [], "steer": [], "steer_deg": [], "throttle": [], "brake": [],
            "cte": [], "yaw": [], "path_yaw": [], "heading_error": []}

    try:
        world.tick()
        steps = int(args.max_duration / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()

            target_idx, delta, _, cte, yaw, theta_e = stanley_control(
                vehicle, wheelbase, path_x, path_y, path_yaw, last_idx
            )
            last_idx = target_idx
            steer = cliping(delta / max_steer,3/7,-3/7)
            steer_deg = steer * math.degrees(max_steer)

            current_speed_ms = speed_ms(vehicle)
            current_speed_kmh = current_speed_ms * 3.6
            speed_error = args.target_speed - current_speed_kmh
            control_value = speed_filter.step(pid.step(speed_error))

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

            loc = vehicle.get_transform().location
            t = i * args.dt
            hist["t"].append(t)
            hist["x"].append(loc.x)
            hist["y"].append(loc.y)
            hist["speed"].append(current_speed_ms)
            hist["steer"].append(steer)
            hist["steer_deg"].append(steer_deg)
            hist["throttle"].append(control.throttle)
            hist["brake"].append(control.brake)
            hist["cte"].append(cte)
            hist["yaw"].append(math.degrees(yaw_unwrapper.step(yaw)))
            hist["path_yaw"].append(math.degrees(path_yaw[target_idx]))
            hist["heading_error"].append(math.degrees(theta_e))

            if i % 10 == 0:
                print(f"t={t:5.1f}s  idx={target_idx:4d}/{len(path_x)}  "
                      f"speed={current_speed_kmh:5.1f} km/h  steer={steer:+.2f}  cte={cte:+.2f} m")

            dist_to_goal = math.hypot(loc.x - path_x[-1], loc.y - path_y[-1])
            if target_idx >= len(path_x) - 2 and dist_to_goal < args.goal_tolerance:
                print(f"Reached goal (within {args.goal_tolerance} m).")
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt/args.times_run:
                time.sleep(args.dt/args.times_run - elapsed)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        vehicle.destroy()
        world.apply_settings(original_settings)
        print("Cleaned up: vehicle destroyed, world settings restored.")

    if not args.no_plot and len(hist["t"]) > 1:
        try:
            out_path = plot_results(path_x, path_y, hist, args.target_speed / 3.6, args.plot_dir)
            print(f"Saved result figure: {out_path}")
        except Exception as exc:
            print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
