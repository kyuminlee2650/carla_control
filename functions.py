"""Building blocks shared by every controller in this repo -- nothing controller-specific.

What belongs here: signal processing (PID, filters, angle handling), path construction, vehicle
parameters, and error measurement. What does not: the control laws themselves, and any geometry a
single law needs (e.g. Stanley's front-axle reference point, which it derives from lf itself).

    from functions import PID, build_path, get_vehicle_geometry, lateral_error
"""

import math
import os
import sys

import numpy as np

# CARLA's PyPI wheel (`pip install carla`) only ships the compiled client API; the navigation
# helpers under PythonAPI/carla/agents (GlobalRoutePlanner) only exist in the simulator's own
# source tree, so that tree still has to be added to sys.path by hand.

# Per-machine defaults: home Windows box first on Windows, lab Ubuntu box first on Linux.
CARLA_ROOT_CANDIDATES = (
    (r"C:\CARLA_0.9.15\WindowsNoEditor", r"C:\CARLA_0.9.15")
    if os.name == "nt" else
    ("/home/ailab/carla/CARLA_0.9.15", "/home/ailab/2026intern/carla", "/opt/carla-simulator")
)


def resolve_carla_root():
    """First CARLA source tree that actually contains PythonAPI/carla/agents.

    $CARLA_ROOT is tried first but is *verified*, not trusted. On the lab machine .bashrc exports
    CARLA_ROOT=/home/ailab/carla_bench2drive for a different project, and that directory has no
    PythonAPI at all -- taking it on faith made every script here die on `No module named 'agents'`
    in any login shell. Since the variable belongs to that other project, this resolves around it
    rather than asking anyone to change their environment.
    """
    tried = []
    env_root = os.environ.get("CARLA_ROOT")
    for candidate in ([env_root] if env_root else []) + list(CARLA_ROOT_CANDIDATES):
        if os.path.isdir(os.path.join(candidate, "PythonAPI", "carla", "agents")):
            return candidate
        tried.append(candidate)

    raise RuntimeError(
        "no CARLA source tree found -- none of these contain PythonAPI/carla/agents:\n  "
        + "\n  ".join(tried)
        + "\nSet CARLA_ROOT to the simulator's install directory (the one holding PythonAPI/), "
          "or add it to CARLA_ROOT_CANDIDATES in functions.py.")


CARLA_ROOT = resolve_carla_root()
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


class PID:
    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error):
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        trial = self._integral + error * self.dt
        raw = self.kp * error + self.ki * trial + self.kd * derivative
        if not ((raw > 1 and error > 0) or (raw < -1 and error < 0)):
            self._integral = trial

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return out


class LowPassFilter:
    def __init__(self, tau, dt, initial=0.0):
        self.alpha = dt / (tau + dt)
        self.state = initial

    def step(self, x):
        self.state += self.alpha * (x - self.state)
        return self.state
        

# Front-wheel steering limit of vehicle.lincoln.mkz_2020, the only car this repo drives. Read off
# VehiclePhysicsControl.wheels[0].max_steer_angle; named rather than inlined so a different
# blueprint is one grep away from working.
MAX_STEER_ANGLE = math.radians(70.0)


def steering_curve_scale(physics, speed_ms):
    """The factor CARLA applies to a steer command at this speed.

    VehiclePhysicsControl.steering_curve is a lookup whose x axis is km/h, not m/s: on the stock
    vehicles 0 -> 1.0, 20 -> 0.9, 60 -> 0.8, 120 -> 0.7. So the same command is a smaller wheel
    angle the faster you go -- measured on the mkz_2020, 0.93 at 4 m/s down to 0.87 at 9 m/s.
    """
    xs = [point.x for point in physics.steering_curve]
    ys = [point.y for point in physics.steering_curve]
    lo, hi = xs[0], xs[-1]
    x = min(max(speed_ms * 3.6, lo), hi)
    for i in range(1, len(xs)):
        if x <= xs[i]:
            span = xs[i] - xs[i - 1]
            t = 0.0 if span == 0 else (x - xs[i - 1]) / span
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def track_over_wheelbase(physics):
    """track / wheelbase, the only vehicle geometry the Ackermann conversion needs.

    Wheel positions come back in centimetres of world coordinates, but both distances here are
    between wheels, so the vehicle's orientation drops out.
    """
    wheels = physics.wheels
    track = math.hypot(wheels[0].position.x - wheels[1].position.x,
                       wheels[0].position.y - wheels[1].position.y)
    front_x = 0.5 * (wheels[0].position.x + wheels[1].position.x)
    front_y = 0.5 * (wheels[0].position.y + wheels[1].position.y)
    rear_x = 0.5 * (wheels[2].position.x + wheels[3].position.x)
    rear_y = 0.5 * (wheels[2].position.y + wheels[3].position.y)
    wheelbase = math.hypot(front_x - rear_x, front_y - rear_y)
    return track / wheelbase


def control_input(u, steer, v_x, vehicle, physics):
    """Apply one tick of control: a signed pedal command and a bicycle-model steer angle.

    u:      throttle-minus-brake in [-1, 1]. Positive is throttle, negative is brake; they are
            mutually exclusive, which is what every stack in this repo already assumes.
    steer:  the wheel angle you want, in RADIANS, in the sense the *bicycle model* means it -- the
            single virtual wheel on the centre line. That is what every control law in this repo
            computes, so it is what this takes.
    v_x:    current forward speed (m/s), body frame. Not the target speed: see below.

    Two conversions stand between that angle and the number CARLA wants, and both were measured
    on this build rather than assumed:

      Ackermann. CARLA drives the INNER wheel to `command * max_steer * curve` and derives the
      outer one from the geometry -- confirmed by reversing the turn and watching which wheel
      tracked the command. The inner wheel is always the larger angle, so handing it the bicycle
      angle realises something smaller than asked: commanding 16 deg produced an equivalent angle
      of 14.87 deg, 7% short. Inverting the geometry fixes it, and the geometry is exact:

          cot(inner) = cot(bicycle) - track / (2 * wheelbase)

      Steering curve. CARLA multiplies the command by steering_curve(speed) before it reaches the
      wheels, so a fixed command is a shrinking angle as the car speeds up. Dividing by the curve
      cancels it -- but it must be the curve at the speed the car has *now*, not the one it is
      heading for. Freezing it at a target speed leaves the angle ~10% too large through the whole
      spin-up, because the curve is near 1.0 while the car is slow; on a constant-radius test that
      is a 10% error in radius, enough to put a car inside a roundabout's kerb.

    Between them the naive `angle / max_steer` is out by about 15%.

    Returns the VehicleControl that was applied, so callers can log throttle/brake/steer without
    rebuilding it.
    """
    control = carla.VehicleControl()
    if u >= 0:
        control.throttle, control.brake = float(u), 0.0
    else:
        control.throttle, control.brake = 0.0, float(-u)

    if abs(steer) < 1e-6:
        inner = 0.0
    else:
        cot_inner = 1.0 / math.tan(abs(steer)) - 0.5 * track_over_wheelbase(physics)
        # cot <= 0 would mean an inner wheel past 90 deg; the steering limit binds long before
        # that, so clamp rather than let the arithmetic wrap.
        inner = MAX_STEER_ANGLE if cot_inner <= 0.0 else math.atan(1.0 / cot_inner)
        inner = math.copysign(inner, steer)

    scale = steering_curve_scale(physics, max(v_x, 0.0))
    control.steer = clipping(inner / (MAX_STEER_ANGLE * scale), 1.0, -1.0)

    vehicle.apply_control(control)
    return control


class ImuAcceleration:
    """Shared post-processing for the IMU's accelerometer: settle, low-pass, differentiate.

    Every controller in this repo needs the same three things off `imu_data.accelerometer` --
    longitudinal and lateral acceleration, and the jerk derived from them -- and each used to
    build its own four filters and repeat the arithmetic inline. They had drifted apart: the two
    MPC stacks clamped a_x to +/-8 m/s^2 and the two PID stacks clamped nothing, and no stack ever
    clamped a_y.

    Why there is no clamp here at all now. The spikes it existed for are real but they are a
    *spawn* artefact, not a magnitude problem: measured on this build, the accelerometer reports
    around -378,000 m/s^2 on the first tick after spawn and is clean from roughly the third
    onward. A magnitude clamp is a poor tool for that -- 8 m/s^2 is inside the range a real hard
    stop reaches, so the clamp silently flattens genuine braking while only partially taming a
    1e5 spike. Skipping the known-bad opening ticks removes the artefact without touching any
    real sample.

    That matters because the spike is otherwise not local. A single -378,000 sample entering a
    causal low-pass with tau = 0.15 s leaves roughly -94,000 in the filter state, which then needs
    about two seconds to decay back under 1 m/s^2 -- so one bad tick contaminates a second or two
    of output, and in the MPC stacks feeds that straight to the pedal layer.

    The filtering itself is causal, and deliberately so: this output drives controllers, where a
    little lag is harmless and looking into the future is not an option. Do not reuse this for
    parameter identification -- there the lag is a bias, and offline zero-phase filtering (or no
    filtering at all, since the noise is zero-mean and averages out) is the right choice.
    """

    def __init__(self, dt, tau=0.15, jerk_tau=0.15, settle_ticks=3):
        self.dt = dt
        self.settle_ticks = settle_ticks
        self._ticks = 0
        self._accel_x = LowPassFilter(tau=tau, dt=dt, initial=0.0)
        self._accel_y = LowPassFilter(tau=tau, dt=dt, initial=0.0)
        self._jerk_x = LowPassFilter(tau=jerk_tau, dt=dt, initial=0.0)
        self._jerk_y = LowPassFilter(tau=jerk_tau, dt=dt, initial=0.0)
        self._prev_a_x = None
        self._prev_a_y = None
        self.a_x_raw = self.a_y_raw = 0.0
        self.a_x = self.a_y = 0.0
        self.jerk = self.jerk_y = self.jerk_total = 0.0

    def step(self, imu_data):
        """Feed one IMU measurement. Returns self, so attributes can be read straight after."""
        self._ticks += 1
        if self._ticks <= self.settle_ticks:
            # Hold everything at zero and, critically, do not let these samples into the filter
            # state -- that is the whole point of skipping them.
            return self

        self.a_x_raw = imu_data.accelerometer.x
        self.a_y_raw = imu_data.accelerometer.y
        self.a_x = self._accel_x.step(self.a_x_raw)
        self.a_y = self._accel_y.step(self.a_y_raw)

        self.jerk = self._jerk_x.step(
            0.0 if self._prev_a_x is None else (self.a_x - self._prev_a_x) / self.dt)
        self.jerk_y = self._jerk_y.step(
            0.0 if self._prev_a_y is None else (self.a_y - self._prev_a_y) / self.dt)
        self._prev_a_x, self._prev_a_y = self.a_x, self.a_y
        self.jerk_total = math.hypot(self.jerk, self.jerk_y)
        return self


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




class CollisionWatch:
    """Latches the first collision reported for the vehicle.

    A sweep that drives into scenery and keeps logging records the wall, not the powertrain (a
    -863 m/s^2 spike, then a bounce and re-acceleration) -- a trial now stops on impact and the
    caller discards the run-up to it too, since the vehicle is already disturbed before the
    contact is reported.
    """

    def __init__(self, world, vehicle):
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)
        self.hit = False
        self.hit_with = None

    def _on_collision(self, event):
        self.hit = True
        # Knowing what was struck is the difference between "the kerb, so the radius is wrong"
        # and "a prop nobody knew was there" -- worth one attribute.
        other = getattr(event, "other_actor", None)
        self.hit_with = getattr(other, "type_id", None) or "unknown"

    def arm(self):
        self.hit = False

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()




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


def speed_reference(args, t):
    """v_des(t) under args.profile -- shared by every controller script's speed reference, so a
    profile change (e.g. adding one) only has to happen in one place.

    "sine": args.initial_speed + args.sine_amplitude * sin(2*pi*t/args.sine_period)
    "step": args.initial_speed, then + args.step_size once t >= args.step_time
    anything else: flat args.initial_speed
    """
    if args.profile == "sine":
        return args.initial_speed + args.sine_amplitude * math.sin(2.0 * math.pi * t / args.sine_period)
    if args.profile == "step":
        return args.initial_speed + (args.step_size if t >= args.step_time else 0.0)
    return args.initial_speed


def reference_preview(args, t0, n_p, dt, warmed_up):
    """Length-Np array of v_des at t0, t0+dt, ..., t0+(Np-1)*dt -- the MPC's look-ahead.

    Before warm-up completes the profile hasn't started yet (t is undefined relative to it), so
    preview a flat initial_speed instead, same as the t=0 value every profile shares.
    """
    if not warmed_up:
        return np.full(n_p, args.initial_speed)
    return np.array([speed_reference(args, t0 + k * dt) for k in range(n_p)])



