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
from scipy.interpolate import UnivariateSpline

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
    """Shared post-processing for the IMU's accelerometer: settle and low-pass.

    Every controller in this repo needs the same two things off `imu_data.accelerometer` --
    longitudinal and lateral acceleration -- and each used to build its own filters and repeat the
    arithmetic inline. They had drifted apart: the two MPC stacks clamped a_x to +/-8 m/s^2 and the
    two PID stacks clamped nothing, and no stack ever clamped a_y. (It used to differentiate them
    into jerk here too; see the note above __init__ for where that went and why.)

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

    # No jerk here any more. This class used to also differentiate a_x/a_y through a second pair of
    # tau=0.15 low-passes and expose .jerk/.jerk_y/.jerk_total. Every jerk number in this project now
    # comes from the SCORING module's own derivative instead (a non-causal Savitzky-Golay over each
    # 20-tick segment, rebuilt post-run by viz_utils.add_scored_comfort_channels()), because the two
    # disagreed badly: measured on real runs, corr(causal-LPF jerk, scored jerk) was 0.29-0.71 with
    # the scored peak 1.2-1.7x higher, so a figure or a printed summary built on this one could sit
    # comfortably inside the B2D limits on a segment the score had already failed. The accelerations
    # themselves are untouched -- undifferentiated, the two agree to corr 1.000 -- and they are what
    # the controllers actually consume, which is why the causal filtering below stays.
    def __init__(self, dt, tau=0.15, settle_ticks=3):
        self.dt = dt
        self.settle_ticks = settle_ticks
        self._ticks = 0
        self._accel_x = LowPassFilter(tau=tau, dt=dt, initial=0.0)
        self._accel_y = LowPassFilter(tau=tau, dt=dt, initial=0.0)
        self.a_x_raw = self.a_y_raw = 0.0
        self.a_x = self.a_y = 0.0

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



def spawn_at(world, x, y, z_margin=0.5):
    """A carla.Transform for a raw (x, y) map coordinate, snapped onto the nearest road waypoint
    (for correct lane heading/z -- an unsnapped raw Location has no orientation and may sit inside
    the road mesh) with a small z margin, same as CARLA's own spawn points carry, so the vehicle
    doesn't spawn clipped into the ground. Does NOT touch the route itself (build_path()'s
    path_x/path_y stay exactly as traced) -- callers use this only to place the vehicle somewhere
    other than the route's own start, e.g. further back on a straight stretch so it can reach
    --initial-speed before run_trial()'s curvature-based logging gate (see mpc_mpc1.py) lets it
    start scoring."""
    wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0))
    t = wp.transform
    return carla.Transform(carla.Location(t.location.x, t.location.y, t.location.z + z_margin), t.rotation)


def build_path(world, sampling_resolution=1, origin_index=0, dest_index=100):
    """Trace a route with GlobalRoutePlanner and flatten it into x/y arrays. Heading/curvature are
    no longer derived here -- see build_path_spline()/PathSpline, which fits a continuous curve to
    these same points and gets yaw/kappa from its own derivatives instead of a finite-difference
    stencil over this array."""
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
    return origin_transform, path_x, path_y


def _chord_length_station(path_x, path_y):
    """Cumulative Euclidean (chord-length) distance along the raw point sequence -- only an
    approximation of true arc length along the curve PathSpline later fits, but a standard and
    (at GlobalRoutePlanner's ~1m point spacing) accurate enough way to get an initial station
    parameterization to fit x(s)/y(s) against in the first place."""
    s = [0.0]
    for i in range(1, len(path_x)):
        s.append(s[-1] + math.hypot(path_x[i] - path_x[i - 1], path_y[i] - path_y[i - 1]))
    return np.asarray(s, dtype=float)


class PathSpline:
    """Arc-length-parameterized SMOOTHING cubic spline fit of a path: x(s), y(s). yaw(s) and
    kappa(s) are analytic derivatives of that fit (atan2(y', x') and the standard curvature formula
    (x'y''-y'x'')/(x'^2+y'^2)^1.5) -- no finite-difference stencil, no dependence on how densely or
    evenly the input points happen to be sampled. Point-to-path queries (project()) work the same
    way regardless of the input source (dense GlobalRoutePlanner waypoints today, sparser planner
    output like VAD later): a coarse scan followed by Newton refinement on the continuous curve,
    rather than a nearest-neighbor search over a specific discrete array.

    smoothing: UnivariateSpline's own smoothing factor (0 = exact interpolation through every input
    point, which lets input noise show up directly in kappa's second derivative; larger = smoother
    fit, less exact through the points). Tune by eye against a route's known corners, the same way
    Cf/Cr got tuned against measured behavior elsewhere in this project.
    """

    def __init__(self, path_x, path_y, smoothing=1.0, min_spacing=0.01):
        s = _chord_length_station(path_x, path_y)
        px = np.asarray(path_x, dtype=float)
        py = np.asarray(path_y, dtype=float)
        # UnivariateSpline requires strictly increasing x. GlobalRoutePlanner's route can repeat
        # (or nearly repeat) a waypoint at junctions/lane-change points -- 14 such spots on the
        # default Town10HD_Opt route alone -- which silently corrupts the whole fit (not just near
        # the duplicate) if left in, so drop the second of any pair closer than min_spacing rather
        # than assume the input is already well-formed.
        keep = np.concatenate([[True], np.diff(s) > min_spacing])
        s, px, py = s[keep], px[keep], py[keep]

        self.s_min, self.s_max = float(s[0]), float(s[-1])
        self._sx = UnivariateSpline(s, px, k=3, s=smoothing)
        self._sy = UnivariateSpline(s, py, k=3, s=smoothing)

    def xy(self, s):
        return self._sx(s), self._sy(s)

    def yaw(self, s):
        return np.arctan2(self._sy(s, 1), self._sx(s, 1))

    def kappa(self, s):
        dx, dy = self._sx(s, 1), self._sy(s, 1)
        ddx, ddy = self._sx(s, 2), self._sy(s, 2)
        denom = (dx * dx + dy * dy) ** 1.5   # ~v_x^3 in the parameterization's own speed; never
        return (dx * ddy - dy * ddx) / np.maximum(denom, 1e-9)   # near 0 for an arc-length fit

    def project(self, x, y, last_s, window=30.0, coarse_step=0.5, newton_iters=4):
        """Closest point on the spline to (x, y), searched forward from last_s (never backward --
        same forward-only assumption lateral_error()'s old array search made). Coarse scan over
        [last_s, min(s_max, last_s+window)] at coarse_step to land in the right basin, then a few
        Newton steps on the orthogonality condition (C(s)-P)*C'(s)=0 for sub-sample accuracy.

        Returns (s_star, e_y). e_y is (vehicle - path) projected onto the path's own left-normal
        n(yaw) = (-sin(yaw), cos(yaw)) -- the same Frenet, vehicle-minus-path sign convention the
        old array-based lateral_error() used (positive = vehicle to the left of the path's own
        tangent direction; see LateralMPC's docstring for why this is the convention e_y_dot =
        v_y + v_x*e_psi needs)."""
        lo = last_s
        hi = min(self.s_max, last_s + window)
        if hi <= lo:
            lo, hi = self.s_max - 1e-6, self.s_max
        s_grid = np.arange(lo, hi, coarse_step)
        if s_grid.size == 0:
            s_grid = np.array([lo])
        xs, ys = self.xy(s_grid)
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        s_star = float(s_grid[np.argmin(d2)])

        for _ in range(newton_iters):
            cx, cy = self.xy(s_star)
            dx, dy = float(self._sx(s_star, 1)), float(self._sy(s_star, 1))
            ddx, ddy = float(self._sx(s_star, 2)), float(self._sy(s_star, 2))
            f = (cx - x) * dx + (cy - y) * dy
            fp = dx * dx + dy * dy + (cx - x) * ddx + (cy - y) * ddy
            if abs(fp) < 1e-9:
                break
            s_star -= f / fp
            s_star = min(max(s_star, self.s_min), self.s_max)

        cx, cy = self.xy(s_star)
        yaw_s = float(self.yaw(s_star))
        dx_v, dy_v = x - float(cx), y - float(cy)
        e_y = -math.sin(yaw_s) * dx_v + math.cos(yaw_s) * dy_v
        return s_star, e_y


def build_path_spline(path_x, path_y, smoothing=1.0):
    """Fit a PathSpline to build_path()'s raw x/y arrays -- the one-time step every controller
    calls right after build_path(), same way build_path_station()/build_path_curvature() used to
    be called once after it."""
    return PathSpline(path_x, path_y, smoothing=smoothing)


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




def lateral_error(x, y, path, last_s, window=30.0):
    """Signed cross-track error of (x, y) against a PathSpline -- independent of whatever
    trajectory a controller happens to be tracking. Used to score true deviation from the global
    route, since the local plan resets itself to the vehicle's position every replan and so can't
    be used to measure real tracking performance (see the "isn't this cheating" discussion).

    Thin wrapper over PathSpline.project(): kept as its own function since every caller in this
    repo already spells it this way, but the actual projection (coarse scan + Newton refine on the
    continuous curve) and the Frenet vehicle-minus-path sign convention both live on PathSpline now
    -- see its docstring. Returns (s_star, e_y), station replacing the old integer array index."""
    return path.project(x, y, last_s, window=window)


def speed_reference(args, t):
    """v_des(t) under args.profile -- shared by every controller script's speed reference, so a
    profile change (e.g. adding one) only has to happen in one place.

    "sine": args.initial_speed + args.sine_amplitude * sin(2*pi*t/args.sine_period)
    "step": args.initial_speed, then + args.step_size over the window [args.step_time,
            args.step_time + args.step_duration), then back to args.initial_speed
    anything else: flat args.initial_speed

    The step is a WINDOW, not a permanent change: args.step_duration (seconds) says how long the
    stepped value is held before the reference snaps back to args.initial_speed. That one parameter
    is what turns this profile into an emergency-stop-and-restart test -- --step-size -<initial
    speed> (or anything more negative, see the clamp below) drops the reference to 0 for
    step_duration seconds and then demands the original speed again in a single step, so the run
    covers a hard decel and a hard re-accel in one profile instead of only the decel.

    Scripts that expose --step-time/--step-size but not --step-duration (mpc_mpc.py,
    longitudinal_mpc.py, mpc_mpc_comparison.py) keep the old behaviour untouched: a missing
    step_duration -- and an explicit None -- both mean "hold forever", the permanent step this
    profile used to be.

    The stepped value is clamped at 0 rather than allowed to go negative: below a full stop there is
    nothing further to ask for (v_des is a speed, and every consumer -- SpeedMPC's tracking cost,
    refine_speed_preview's forward walk, VAD's waypoint spacing -- reads it as one), so
    --step-size -100 is simply "stop", not "drive backwards at 90 m/s".
    """
    if args.profile == "sine":
        return args.initial_speed + args.sine_amplitude * math.sin(2.0 * math.pi * t / args.sine_period)
    if args.profile == "step":
        duration = getattr(args, "step_duration", None)
        stepping = t >= args.step_time and (duration is None or t < args.step_time + duration)
        if stepping:
            return max(0.0, args.initial_speed + args.step_size)
        return args.initial_speed
    return args.initial_speed


def reference_preview(args, t0, n_p, dt, warmed_up, speed_fn=speed_reference):
    """Length-Np array of v_des at t0, t0+dt, ..., t0+(Np-1)*dt -- the MPC's look-ahead.

    speed_fn(args, t) -> float supplies the actual profile; defaults to this module's own
    speed_reference() but is injectable so a script with a different profile set/args shape (e.g.
    stanley_mpc.py's own speed_reference(), which adds an "estop" profile and uses args.target_speed
    instead of args.initial_speed) can still share this walking/warm-up logic instead of keeping a
    second copy of it.

    Before warm-up completes the profile hasn't started yet (t is undefined relative to it), so
    preview a flat speed_fn(args, 0.0) instead -- every profile's own t=0 value already equals its
    steady-state target (sin(0)=0, no step/stop yet), so this is the same flat value as before
    without hardcoding which attribute name holds it.
    """
    if not warmed_up:
        return np.full(n_p, speed_fn(args, 0.0))
    return np.array([speed_fn(args, t0 + k * dt) for k in range(n_p)])


def refine_speed_preview(v_preview, path, last_s, dt, a_y_max=4.9):
    """Clip a time-domain speed preview (reference_preview()'s output) to what upcoming curvature
    allows: v_target[k] = min(v_preview[k], sqrt(a_y_max/|kappa(s_k)|)), the same v <= sqrt(a_y/kappa)
    relation behind every a_y=v^2/r note elsewhere in this project about corners this route's
    lateral controllers can't out-steer at speed. Without this, the reference profile a longitudinal
    MPC tracks has no idea a curve is coming and only reacts to it laterally, after the fact.

    s_k is found by walking a station cursor forward using the ALREADY-refined v_target at each
    prior step, not the raw v_preview -- self-consistent, since slowing down now means arriving at
    a later station later too, the same forward-walk idea curvature_preview() already uses for the
    lateral MPC's own kappa preview (just threaded through v itself here instead of only read).

    a_y_max: comfortable/grip lateral-acceleration budget (m/s^2) a curve of a given radius is
    allowed to demand. Tune like any other physical limit in this project (Cf/Cr, delta_max, ...).
    """
    v_preview = np.asarray(v_preview, dtype=float)
    s_cursor = last_s
    out = np.empty_like(v_preview)
    for k, v in enumerate(v_preview):
        kappa = float(path.kappa(min(s_cursor, path.s_max)))
        v_curve = math.sqrt(a_y_max / abs(kappa)) if abs(kappa) > 1e-6 else float("inf")
        v_k = min(v, v_curve)
        out[k] = v_k
        s_cursor += max(v_k, 0.0) * dt
    return out



