"""Building blocks shared by every controller in this repo -- nothing controller-specific.

What belongs here: signal processing (PID, filters, angle handling), path construction, vehicle
parameters, and error measurement. What does not: the control laws themselves, and any geometry a
single law needs (e.g. Stanley's front-axle reference point, which it derives from lf itself).

    from functions import PID, build_path, get_vehicle_geometry, lateral_error
"""

import math
import sys

sys.path.append("/home/ailab/2026intern/carla/PythonAPI/carla")

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


class PID:
    def __init__(self, kp, ki, kd, dt, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.out_min, self.out_max = out_min, out_max
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error):
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        # Conditional integration: throttle saturates at 1.0 on every uphill/corner recovery, and
        # integrating through that saturation is what made the speed overshoot after each dip.
        # Only accumulate when the integrator isn't pushing further into a limit it already hit.
        trial = self._integral + error * self.dt
        raw = self.kp * error + self.ki * trial + self.kd * derivative
        if not ((raw > self.out_max and error > 0) or (raw < self.out_min and error < 0)):
            self._integral = trial

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.out_min, min(out, self.out_max))


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



def heading_from_points(path_x, path_y, stencil=6):
    """Path heading from a centred finite difference over the polyline.

    Preferred over each waypoint's own rotation.yaw, which is noisy around junctions and lane
    changes -- that jitter fed straight into e_theta and showed up as steering chatter.
    """
    n = len(path_x)
    unwrapper = AngleUnwrapper()
    headings = []
    for i in range(n):
        a = max(0, i - stencil)
        b = min(n - 1, i + stencil)
        headings.append(unwrapper.step(math.atan2(path_y[b] - path_y[a], path_x[b] - path_x[a])))
    return headings


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
    path_yaw = heading_from_points(path_x, path_y)
    return origin_transform, path_x, path_y, path_yaw


def get_vehicle_geometry(vehicle, spawn_transform):
    """Bicycle-model parameters: (wheelbase, lf, lr, max_steer).

    lf / lr are the distances from the centre of mass to the front and rear axle -- the split every
    dynamic bicycle model, slip-angle estimate and lateral tyre-force term needs. max_steer is the
    max front-wheel angle in rad; all distances are metres.

    A controller that references some other point on the body (Stanley's front axle, a pure-pursuit
    rear axle) derives that offset from lf/lr on its own -- it is not vehicle geometry, it is a
    property of the law.

    spawn_transform must be the pose the vehicle was spawned at: in synchronous mode the server has
    not populated the actor yet before the first world.tick(), so vehicle.get_transform() still
    reads (0, 0, 0) here and cannot be used as the body-frame reference.
    """
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

    # Both axles and the centre of mass are measured against the actor origin, which sits between
    # the axles (and not at their midpoint): the wheels come back as world positions, so project
    # them onto the heading to get the body-frame offset, while center_of_mass is already relative
    # to the origin in the body frame.
    yaw = math.radians(spawn_transform.rotation.yaw)

    def offset_from_origin(wheel_mid):
        dx = wheel_mid.x / 100.0 - spawn_transform.location.x
        dy = wheel_mid.y / 100.0 - spawn_transform.location.y
        return dx * math.cos(yaw) + dy * math.sin(yaw)

    com_x = physics.center_of_mass.x
    lf = offset_from_origin(front_mid) - com_x
    lr = com_x - offset_from_origin(rear_mid)
    if not (0.0 < lf < wheelbase and 0.0 < lr < wheelbase):
        # a bad reference pose silently turns into a huge phantom cross-track error, so fail loudly
        raise RuntimeError(f"lf={lf:.2f} m / lr={lr:.2f} m do not straddle the centre of mass "
                           f"inside the wheelbase ({wheelbase:.2f} m); spawn_transform does not "
                           f"match the vehicle pose.")

    print(f"wheelbase={wheelbase:.2f} m  lf={lf:.2f} m  lr={lr:.2f} m  "
          f"max_steer={max_steer_deg:.1f} deg")
    return wheelbase, lf, lr, math.radians(max_steer_deg)




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


