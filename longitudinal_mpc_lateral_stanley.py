"""Stanley lateral controller + PID longitudinal controller.

Path source: CARLA has no built-in "desired path" object.
agents.navigation.GlobalRoutePlanner runs A* over the map's waypoint topology
between an origin and destination to produce one.

Usage:
    cd ~/carla_control
    python3 stanley_pathtracking_v2.py --times-run 1
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
from mock_planner import MockPlanner, local_to_world
from bev_view import BevView

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
        

def normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


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
        
        

def clipping(value,max_val,min_val):
    return max(min_val,min(value,max_val))



def build_path(world, sampling_resolution=1, origin_index=0, dest_index=100):
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
    yaw_unwrapper = AngleUnwrapper()
    path_yaw = [yaw_unwrapper.step(math.radians(wp.transform.rotation.yaw)) for wp, _ in route]
    return origin_transform, path_x, path_y, path_yaw


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

    print(front_mid/100,rear_mid/100)
    print(f"wheelbase={wheelbase:.2f} m  max_steer={max_steer_deg:.1f} deg")
    return wheelbase, math.radians(max_steer_deg)

def lateral_error(x, y, yaw, path_x, path_y, last_idx, search_window=30):
    """Signed cross-track error of (x, y) against an arbitrary fixed path -- independent of whatever
    trajectory a controller happens to be tracking. Used to score true deviation from the global
    route, since the local plan resets itself to the vehicle's position every replan and so can't
    be used to measure real tracking performance (see the "isn't this cheating" discussion)."""
    lo = last_idx
    hi = min(len(path_x), last_idx + search_window)
    dists = [math.hypot(x - path_x[i], y - path_y[i]) for i in range(lo, hi)]
    idx = lo + dists.index(min(dists))
    dx = path_x[idx] - x
    dy = path_y[idx] - y
    err = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return idx, err


def world_to_local(x, y, yaw, origin_x, origin_y, origin_yaw):
    """Inverse of mock_planner.local_to_world for a single pose: world (x, y, yaw) into the
    ego-relative frame defined by (origin_x, origin_y, origin_yaw)."""
    dx, dy = x - origin_x, y - origin_y
    cos_o, sin_o = math.cos(origin_yaw), math.sin(origin_yaw)
    local_x = dx * cos_o + dy * sin_o
    local_y = -dx * sin_o + dy * cos_o
    return local_x, local_y, normalize_angle(yaw - origin_yaw)



def stanley_control(v_x, e_y, e_theta, k_theta=0.5,k=1):
    delta = k_theta * e_theta + math.atan2(k * e_y, v_x)
    return delta


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
    parser.add_argument("--no-live-view", action="store_true", help="skip the live BEV plan/vehicle view")
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
    target_speed_ms = args.target_speed / 3.6
    vehicle.set_target_velocity(carla.Vector3D(
        x=target_speed_ms * math.cos(initial_yaw),
        y=target_speed_ms * math.sin(initial_yaw),
        z=0.0,
    ))

    wheelbase, max_steer = get_vehicle_geometry(vehicle)

    pid = PID(kp=0.35, ki=0.15, kd=0.05, dt=args.dt)
    speed_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0.0)
    steer_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0)
    yaw_unwrapper = AngleUnwrapper()
    rh_unwrapper = AngleUnwrapper()
    last_idx = 0

    bev = None if args.no_live_view else BevView(path_x, path_y)

    hist = {"t": [], "x": [], "y": [], "v_x": [], "last_idx": [], "steer_deg": [], "throttle": [], "brake": [],
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
            # body-frame longitudinal velocity (not vel_vec.x, which is world-frame)
            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)
            road_heading = rh_unwrapper.step(path_yaw[last_idx])

            last_idx, e_y = lateral_error(ego_x, ego_y, yaw, path_x, path_y, last_idx)
            e_theta = road_heading - yaw
            e_vel = target_speed_ms - v_x

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

            t = i * args.dt
            hist["t"].append(t)
            hist["x"].append(ego_x)
            hist["y"].append(ego_y)
            hist["v_x"].append(v_x)
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
        if bev is not None:
            bev.close()
            plt.ioff()  # BevView leaves interactive mode on; turn it off so plot_results()'s plt.show() blocks again
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