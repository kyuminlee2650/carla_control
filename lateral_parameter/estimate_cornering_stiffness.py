r"""Flat test pad for constant-radius cornering experiments.

Right now this only builds the world. The identification code that used to live here has been
removed so it can be written from scratch against a clean surface.

The pad is generated at runtime from a hand-written OpenDRIVE string -- one straight road with a
lot of wide lanes -- via client.generate_opendrive_world(). No Unreal Editor, no map files, no
RoadRunner. CARLA turns the road network into a procedural mesh and hands back a world.

Why bother, when Town03 and Town06 both have tarmac big enough to drive a circle on:

  - Radius is continuous. The two real sites only offered R = 8-11.5 m and 18.5-24.5 m with a gap
    between them, because those are the widths the maps happen to have. The pad covers roughly
    5-28 m without a break, which is what a sweep separating lateral load from speed needs.
  - No road defects. Town06's circle crosses something at ~236 deg that recurs every lap, throws
    the body into a 4.5 deg roll transient, and is invisible in the map's own elevation data --
    it was only found by driving the circle and watching. A procedural mesh has nothing on it.
  - No kerbs. Town03's inner edge is the roundabout's central island, so a circle that comes out
    tighter than requested mounts it. Here there is nothing to hit, and wall_height=0 keeps CARLA
    from generating barriers at the road edge either.
  - One map. No reloading between sites partway through a sweep.

Measured, not assumed: elevation range over the whole pad is 0.0000 m, and an identical
constant-steer circle driven here and on Town06 returned the same radius, a_y, slip angles and
stiffnesses to four decimal places (Cf 108,691 vs 108,700). The surfaces are physically
indistinguishable, so results from the pad are directly comparable with anything measured on the
stock maps. That is expected rather than lucky: with tire_friction 3.5 the operating range sits
far below saturation, and in a tire's linear region the lateral force is set by lat_stiff_value
and vertical load, with surface friction only fixing the ceiling.

Usage (Ubuntu):
    cd ~/carla_control
    python3 lateral_parameter/estimate_cornering_stiffness.py
    python3 lateral_parameter/estimate_cornering_stiffness.py --lanes 16 --length 200

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe lateral_parameter\estimate_cornering_stiffness.py
    .venv\Scripts\python.exe lateral_parameter\estimate_cornering_stiffness.py --lanes 16 --length 200
"""

import argparse
import math
import os
import queue
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))

# functions.py resolves CARLA_ROOT and puts the simulator's PythonAPI on sys.path
from functions import get_vehicle_geometry, PID, LowPassFilter, clipping, control_input

import carla

from viz_utils import follow_with_spectator

from bicycle import zero_phase_derivative


LANE_WIDTH = 3.5


def build_xodr(lanes_each_side=12, length=130.0, lane_width=LANE_WIDTH):
    """OpenDRIVE for a single straight road, `lanes_each_side` lanes wide on each side.

    A pad rather than a circle: with the whole rectangle drivable, any radius up to about the
    half-width can be driven anywhere on it, so the geometry does not have to be decided here.

    `level="true"` on every lane and a flat <elevationProfile> are what keep it dead level -- no
    superelevation, no crown, no camber for gravity to leak into an accelerometer through.
    """
    def lane(i):
        return (f'          <lane id="{i}" type="driving" level="true">\n'
                f'            <width sOffset="0" a="{lane_width}" b="0" c="0" d="0"/>\n'
                f'          </lane>\n')

    left = "".join(lane(i) for i in range(lanes_each_side, 0, -1))
    right = "".join(lane(-i) for i in range(1, lanes_each_side + 1))
    return f'''<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="flatpad" version="1" date=""
          north="0" south="0" east="0" west="0"/>
  <road name="pad" length="{length}" id="1" junction="-1">
    <planView>
      <geometry s="0" x="0" y="0" hdg="0" length="{length}"><line/></geometry>
    </planView>
    <elevationProfile>
      <elevation s="0" a="0" b="0" c="0" d="0"/>
    </elevationProfile>
    <lateralProfile/>
    <lanes>
      <laneSection s="0">
        <left>
{left}        </left>
        <center><lane id="0" type="none" level="true"/></center>
        <right>
{right}        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
'''


def load_pad(client, lanes_each_side=12, length=130.0, dt=0.05):
    """Generate the pad world and put it in synchronous mode. Returns (world, centre, max_radius).

    wall_height=0 matters: CARLA's default is 1.0 m, which fences the road edge with an invisible
    barrier -- fine for a route, not for a circle that may drift wide.
    """
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,      # one piece, not chopped into segments
        wall_height=0.0,
        additional_width=0.0,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )
    world = client.generate_opendrive_world(build_xodr(lanes_each_side, length), params)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)

    centre = (length / 2.0, 0.0)
    max_radius = lanes_each_side * LANE_WIDTH
    return world, centre, max_radius




def spawn_vehicle(world, centre, blueprint_filter="vehicle.lincoln.mkz_2020", ride_height=0.3):

    transform = carla.Transform(carla.Location(x=centre[0], y=centre[1], z=ride_height),
                                carla.Rotation(yaw=0.0))
    blueprint = world.get_blueprint_library().filter(blueprint_filter)[0]
    vehicle = world.spawn_actor(blueprint, transform)
    stop(vehicle)

    wheelbase, lf, lr, max_steer = get_vehicle_geometry(vehicle, transform)
    physics = vehicle.get_physics_control()
    return vehicle, (wheelbase, lf, lr, max_steer, physics.mass, physics.center_of_mass)


def stop(vehicle):
    """Zero the velocities a freshly spawned or teleported actor is left holding."""
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))


def attach_imu(world, vehicle, centre_of_mass):
    """Mount an IMU at the centre of mass and return (sensor, queue).

    At the CoM, not the actor origin. An accelerometer a distance d from the CG also reads
    psi_ddot*d, which at this rig's numbers -- d = 0.3 m, psi_ddot around 0.3 rad/s^2 -- is
    0.09 m/s^2, over 10% of a_y at the low end of the sweep. Mounting it in the right place is
    cheaper than correcting for it afterwards.
    """
    blueprint = world.get_blueprint_library().find("sensor.other.imu")
    imu = world.spawn_actor(
        blueprint,
        carla.Transform(carla.Location(centre_of_mass.x, centre_of_mass.y, centre_of_mass.z)),
        attach_to=vehicle,
    )
    queue_ = queue.Queue()
    imu.listen(queue_.put)
    return imu, queue_


def add_derivatives(log, dt, window_s=0.4):
    """Differentiate the logged signals after the run, zero-phase.

    After, not during, and zero-phase rather than a causal filter, for the same reason in both
    cases: a causal filter delays its output, and every use of these derivatives is a comparison
    against something measured at the same instant. Gating on |v_y_dot|, or checking a_y_imu
    against v_x*psi_dot, both become meaningless if one side is shifted in time by the filter that
    was supposed to clean it up. Offline there is no reason to accept that -- the whole trace is
    already in hand, so the derivative can look both ways.

    The window is wide (0.4 s) on purpose. These are used to answer "has this settled", which is a
    low-frequency question; a narrow window would just pass per-tick noise through.
    """
    log["v_x_dot"] = zero_phase_derivative(log["v_x"], dt, window_s=window_s).tolist()
    log["v_y_dot"] = zero_phase_derivative(log["v_y"], dt, window_s=window_s).tolist()
    log["psi_ddot"] = zero_phase_derivative(log["r"], dt, window_s=window_s).tolist()
    return log




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)

    # ---- map ---- #
    parser.add_argument("--lanes", type=int, default=12,
                        help="driving lanes each side of the centre line; the pad's half-width is "
                             "lanes * 3.5 m, which is also the largest circle it can hold")
    parser.add_argument("--length", type=float, default=1000.0, help="pad length (m)")

    # ---- simulation ---- #
    parser.add_argument("--times-run", type=float, default=5.0,
                        help="simulation speed relative to real time: 1 = real time, 2 = twice as "
                             "fast, and so on. The simulator has no clock of its own in "
                             "synchronous mode, so this is purely how long the client waits "
                             "between ticks")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--target-speed", type=float, default=5)
    parser.add_argument("--steer-deg", type=float, default=16.0,
                        help="front wheel angle to hold, in degrees of actual wheel angle -- not "
                             "a fraction of max_steer and not a radius. Positive and negative "
                             "just pick which way round the circle goes")
    parser.add_argument("--settle-ticks", type=int, default=20,
                        help="ticks to let the car drop onto the surface before reporting")
    parser.add_argument("--duration", type=float, default=50)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(180.0)

    print(f"generating a {args.length:.0f} x {2*args.lanes*LANE_WIDTH:.0f} m flat pad "
          f"({args.lanes} lanes each side)...")
    world, centre, max_radius = load_pad(client, args.lanes, args.length, args.dt)
    print(f"  map={world.get_map().name}  centre=({centre[0]:.1f}, {centre[1]:.1f})  "
          f"pad holds a circle up to R={max_radius:.1f} m, or ~{max_radius/2:.1f} m "
          f"starting from the centre")

    pid = PID(kp=0.7, ki=0.15, kd=0.05, dt=args.dt)
    speed_filter = LowPassFilter(tau=0.2, dt=args.dt, initial=0.0)

    vehicle = imu = None
    try:
        vehicle, geometry = spawn_vehicle(world, centre, "vehicle.lincoln.mkz_2020")
        wheelbase, lf, lr, max_steer, mass, com = geometry
        physics = vehicle.get_physics_control()

        delta = math.radians(args.steer_deg)
        if abs(delta) > max_steer:
            raise SystemExit(f"--steer-deg {args.steer_deg} exceeds this vehicle's "
                             f"{math.degrees(max_steer):.1f} deg of steering")
        print(f"\nholding a {args.steer_deg:+.2f} deg wheel angle "
              f"(kinematic radius L/tan(delta) = {wheelbase/math.tan(abs(delta)):.2f} m "
              f"before any tire slip)")

        imu, imu_queue = attach_imu(world, vehicle, com)

        # The settle ticks double as the IMU's warm-up. CARLA derives the accelerometer from
        # velocity differences and reports garbage for the first tick or two after spawn -- around
        # -378,000 m/s^2 was measured -- so those samples get consumed here rather than logged.
        for _ in range(args.settle_ticks):
            world.tick()
            imu_queue.get(timeout=2.0)
            follow_with_spectator(world, vehicle)

        log = {k: [] for k in ("t", "x", "y", "yaw", "v_x", "v_y", "r",
                               "a_x_imu", "a_y_imu", "delta", "steer_cmd", "throttle", "brake")}

        # Wall-clock budget per step. The simulator advances by args.dt of *simulated* time on
        # every tick regardless of how long that took to compute, so pacing is entirely up to the
        # client: sleeping until dt/times_run has passed makes one second of simulation take
        # 1/times_run seconds of real time. times_run = 1 is therefore real time, 2 is twice as
        # fast, and anything the machine cannot keep up with simply runs slower than asked.
        budget = args.dt / args.times_run
        started, behind = time.time(), 0

        for i in range(int(args.duration / args.dt)):
            step_start = time.time()
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)

            transform = vehicle.get_transform()
            yaw = math.radians(transform.rotation.yaw)
            location = transform.location
            velocity = vehicle.get_velocity()
            # Body frame: v_x is what the steering curve is indexed on and what the bicycle model
            # divides by. |v| would be close but not equal -- they differ by cos(beta), and beta
            # reaches several degrees in a tight circle.
            v_x = velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)
            v_y = -velocity.x * math.sin(yaw) + velocity.y * math.cos(yaw)

            e_vel = args.target_speed - v_x
            control_value = clipping(speed_filter.step(pid.step(e_vel)), 1, -1)

            # Takes effect on the next tick, not this one -- the state just read is the result of
            # the previous command.
            control = control_input(control_value, delta, v_x, vehicle, physics)
            follow_with_spectator(world, vehicle)

            r = imu_data.gyroscope.z

            log["t"].append(i * args.dt)
            log["x"].append(location.x); log["y"].append(location.y); log["yaw"].append(yaw)
            log["v_x"].append(v_x); log["v_y"].append(v_y); log["r"].append(r)
            log["a_x_imu"].append(imu_data.accelerometer.x)
            log["a_y_imu"].append(imu_data.accelerometer.y)
            log["delta"].append(delta); log["steer_cmd"].append(control.steer)
            log["throttle"].append(control.throttle); log["brake"].append(control.brake)

            if i % int(1.0 / args.dt) == 0:
                print(f"t={i*args.dt:6.2f}s  v_x={v_x:6.3f}/{args.target_speed:.1f}  "
                      f"v_y={v_y:+6.3f}  psi_dot={math.degrees(r):+7.2f} deg/s  "
                      f"R={v_x/r if abs(r) > 1e-4 else float('nan'):7.2f} m  "
                      f"cmd={control.steer:+.4f}  delta={math.degrees(delta):+6.2f} deg "
                      f"(naive {math.degrees(control.steer*max_steer):+6.2f})")

            elapsed = time.time() - step_start
            if elapsed < budget:
                time.sleep(budget - elapsed)
            else:
                behind += 1

        wall = time.time() - started
        sim = args.duration
        print(f"\n{sim:.1f}s of simulation in {wall:.1f}s of real time "
              f"({sim/wall:.2f}x, asked for {args.times_run:.2f}x)")
        if behind:
            print(f"  {behind} of {int(sim/args.dt)} steps missed the {budget*1000:.0f} ms budget "
                  f"-- the simulator could not keep up, so it ran slower than requested")

        add_derivatives(log, args.dt)
        summarise(log, args)
    finally:
        if imu is not None:
            imu.stop()
            imu.destroy()
        if vehicle is not None:
            vehicle.destroy()
        print("actors destroyed")


def summarise(log, args, tail_s=3.0):
    """Report the settled tail: is it steady, and do the two routes to a_y agree?"""
    t = np.array(log["t"])
    tail = t >= t[-1] - tail_s
    if tail.sum() < 5:
        return

    def stat(key, scale=1.0):
        v = np.array(log[key])[tail] * scale
        return v.mean(), v.std()

    print(f"\n--- last {tail_s:.1f} s ---")
    for label, key, unit, scale in (
            ("v_x", "v_x", "m/s", 1.0),
            ("v_y", "v_y", "m/s", 1.0),
            ("psi_dot", "r", "deg/s", 180.0 / math.pi),
            ("delta", "delta", "deg", 180.0 / math.pi),
            ("v_x_dot", "v_x_dot", "m/s^2", 1.0),
            ("v_y_dot", "v_y_dot", "m/s^2", 1.0),
            ("psi_ddot", "psi_ddot", "deg/s^2", 180.0 / math.pi),
            ("a_y IMU", "a_y_imu", "m/s^2", 1.0)):
        mean, std = stat(key, scale)
        print(f"  {label:>9}: {mean:+9.4f} +/- {std:7.4f} {unit}")

    v_x = np.array(log["v_x"])[tail]
    r = np.array(log["r"])[tail]
    a_y_kin = v_x * r
    a_y_imu = np.array(log["a_y_imu"])[tail]
    print(f"\n  a_y from v_x*psi_dot : {a_y_kin.mean():+.4f} +/- {a_y_kin.std():.4f}")
    print(f"  a_y from the IMU     : {a_y_imu.mean():+.4f} +/- {a_y_imu.std():.4f}")
    print(f"  gap                  : {a_y_imu.mean() - a_y_kin.mean():+.4f} m/s^2")
    print("  In steady state the two must agree -- their difference is v_y_dot. A gap that "
          "persists\n  while v_y_dot is ~0 is the accelerometer reading something else: it "
          "measures specific\n  force in the BODY frame, so body roll tilts g into its y axis.")
    print(f"  IMU noise is {a_y_imu.std()/max(a_y_kin.std(), 1e-9):.0f}x the kinematic route's, "
          f"and scales like 1/dt (dt={args.dt})")


if __name__ == "__main__":
    main()
