"""Longitudinal control-input calibration sweep.

Builds the raw data for a (gear, v_x, a_x) -> u lookup table, where u in
[-1, 1] is the longitudinal control input (positive = throttle, zero =
coasting, negative = brake). For each gear, drives at a constant u from rest
up to the sweep's top speed (throttle) or from that speed down to rest
(coast/brake), logging (v_x, a_x) every tick. One trial per (gear, u) covers
the full speed range for that pair, so this is much cheaper than gridding
over (gear, v, u) directly.

A collision sensor ends a trial on impact and the last --collision-trim
seconds are discarded, so scenery the vehicle drove into never reaches the
table.

Output is a CSV of raw samples -- inverting it into a gear/v/a -> u table
for MPC lookup is a separate step (e.g. scipy.interpolate over this data).

Paths default to this script's own directory, so it runs from anywhere.

Usage:
    cd ~/carla_control
    .venv/bin/python longitudinal_lookup/build_longitudinal_lut.py
"""

import argparse
import csv
import math
import os

import sys

import carla

# defaults resolve next to this script, so it runs from any working directory
HERE = os.path.dirname(os.path.abspath(__file__))
# viz_utils lives one level up, alongside the path-tracking controllers
sys.path.append(os.path.dirname(HERE))


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


class CollisionWatch:
    """Latches the first collision reported for the vehicle.

    An earlier sweep drove into scenery and kept logging: the trial recorded a
    -863 m/s^2 spike, then bounced off and re-accelerated, so ~5% of the data
    described a wall rather than the powertrain. A trial now stops on impact and
    throws away the run-up to it as well, since the vehicle is already being
    disturbed before the contact is reported.
    """

    def __init__(self, world, vehicle):
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)
        self.hit = False

    def _on_collision(self, _event):
        self.hit = True

    def arm(self):
        self.hit = False

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()


def flush_trial(rows, collided, args, writer):
    """Write a trial's rows, dropping the tail if it ended in a collision."""
    if collided:
        drop = int(round(args.collision_trim / args.dt))
        rows = rows[:-drop] if drop < len(rows) else []
    for row in rows:
        writer.writerow(row)
    return len(rows)


def run_throttle_trial(world, vehicle, origin_transform, gear, u, dt, args, writer, collision, spectate):
    reset_trial(world, vehicle, origin_transform, 0.0)
    collision.arm()
    start_loc = vehicle.get_transform().location
    plateau_count = 0
    rows = []
    t = 0.0
    while t < args.max_duration:
        apply_long_control(vehicle, gear, u)
        world.tick()
        if spectate:
            spectate(world, vehicle)
        v_x, a_x = body_frame_long_state(vehicle)
        rows.append([gear, u, round(t, 3), round(v_x, 4), round(a_x, 4)])

        if collision.hit:
            break
        # urban controller: no need to calibrate past city speeds
        if v_x > args.max_speed:
            break
        if vehicle.get_transform().location.distance(start_loc) > args.max_distance:
            break
        if a_x < args.plateau_eps:
            plateau_count += 1
            if plateau_count > args.plateau_ticks:
                break
        else:
            plateau_count = 0
        t += dt
    return flush_trial(rows, collision.hit, args, writer), collision.hit


def run_decel_trial(world, vehicle, origin_transform, gear, u, dt, args, writer, collision, spectate):
    """Coast (u = 0) or brake (u < 0) down from the sweep's top speed."""
    reset_trial(world, vehicle, origin_transform, args.brake_start_speed)
    collision.arm()
    start_loc = vehicle.get_transform().location
    rows = []
    t = 0.0
    while t < args.max_duration:
        apply_long_control(vehicle, gear, u)
        world.tick()
        if spectate:
            spectate(world, vehicle)
        v_x, a_x = body_frame_long_state(vehicle)
        rows.append([gear, u, round(t, 3), round(v_x, 4), round(a_x, 4)])

        if collision.hit:
            break
        if vehicle.get_transform().location.distance(start_loc) > args.max_distance:
            break
        if v_x < args.min_speed:
            break
        t += dt
    return flush_trial(rows, collision.hit, args, writer), collision.hit


def frange(start, stop, step):
    n = round((stop - start) / step)
    return [round(start + i * step, 4) for i in range(n + 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--out", default=os.path.join(HERE, "longitudinal_lut.csv"))
    parser.add_argument("--gears", default="", help="comma-separated gear list; default = all forward gears reported by the vehicle physics")
    parser.add_argument("--throttle-min", type=float, default=0.1)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--throttle-step", type=float, default=0.1)
    parser.add_argument("--brake-min", type=float, default=0.1)
    parser.add_argument("--brake-max", type=float, default=1.0)
    parser.add_argument("--brake-step", type=float, default=0.1)
    parser.add_argument("--brake-start-speed", type=float, default=25.0, help="starting speed for every coast/brake trial (m/s)")
    parser.add_argument("--max-speed", type=float, default=25.0,
                        help="throttle trials stop here (m/s); urban driving needs no calibration above this")
    parser.add_argument("--max-duration", type=float, default=25.0,
                        help="per-trial cutoff (s); high gears accelerate slowly and need the headroom")
    parser.add_argument("--collision-trim", type=float, default=0.5,
                        help="discard this much data before a collision (s); the vehicle is already "
                             "disturbed before the impact is reported")
    parser.add_argument("--max-distance", type=float, default=500.0,
                        help="per-trial cutoff (m); a safety net only -- at --max-speed for "
                             "--max-duration the vehicle cannot exceed this, so speed and time govern")
    parser.add_argument("--plateau-eps", type=float, default=0.05, help="|a_x| below this counts as 'no longer accelerating' (m/s^2)")
    parser.add_argument("--plateau-ticks", type=int, default=40, help="consecutive plateau ticks before ending a throttle trial early")
    parser.add_argument("--min-speed", type=float, default=0.2, help="brake trial ends once v_x drops below this (m/s)")
    parser.add_argument("--map", default="", help="load this map first (e.g. Town06); default = keep the current one")
    parser.add_argument("--origin-index", type=int, default=-1,
                        help="spawn point index; -1 picks the one with the longest straight ahead")
    parser.add_argument("--no-spectator", action="store_true", help="do not move the camera to follow the vehicle")
    args = parser.parse_args()

    spectate = None
    if not args.no_spectator:
        from viz_utils import follow_with_spectator
        spectate = follow_with_spectator

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.map) if args.map else client.get_world()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    spawn_points = world.get_map().get_spawn_points()
    if args.origin_index < 0:
        from find_straight_spawn import straight_length
        carla_map = world.get_map()
        lengths = []
        for i, transform in enumerate(spawn_points):
            wp = carla_map.get_waypoint(transform.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
            lengths.append((straight_length(wp, 2.0, args.max_distance, 3.0) if wp else 0.0, i))
        best_length, origin_index = max(lengths)
        print(f"스폰포인트 자동 선택: index {origin_index} (직선 {best_length:.0f}m 확보)")
        if best_length < args.max_distance:
            print(f"  경고: 시행이 최대 {args.max_distance:.0f}m 까지 갈 수 있는데 직선은 "
                  f"{best_length:.0f}m 입니다. 충돌 시 해당 시행은 중단됩니다.")
    else:
        origin_index = args.origin_index
    origin_transform = spawn_points[origin_index]

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
    # u = 0 is coasting: engine braking and drag with neither pedal applied. It is
    # the input a controller reaches for most often when it wants to shed a little
    # speed, and the previous sweep never measured it -- the table interpolated
    # straight across the gap between the smallest brake and the smallest throttle.
    decels = [0.0] + [-b for b in frange(args.brake_min, args.brake_max, args.brake_step)]
    print(f"Throttle grid: {throttles}")
    print(f"Coast/brake grid: {decels}")

    total_trials = len(gears) * (len(throttles) + len(decels))
    done = 0
    collisions = 0

    collision = CollisionWatch(world, vehicle)
    try:
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gear", "u", "t", "v_x", "a_x"])

            for gear in gears:
                for u in throttles:
                    done += 1
                    n, hit = run_throttle_trial(world, vehicle, origin_transform, gear, u,
                                                args.dt, args, writer, collision, spectate)
                    collisions += hit
                    print(f"[{done}/{total_trials}] gear={gear} u={u:+.2f} (throttle) "
                          f"-> {n} samples{'  [충돌: 마지막 %.1fs 버림]' % args.collision_trim if hit else ''}")

                for u in decels:
                    done += 1
                    label = "coast" if u == 0.0 else "brake"
                    n, hit = run_decel_trial(world, vehicle, origin_transform, gear, u,
                                             args.dt, args, writer, collision, spectate)
                    collisions += hit
                    print(f"[{done}/{total_trials}] gear={gear} u={u:+.2f} ({label}) "
                          f"-> {n} samples{'  [충돌: 마지막 %.1fs 버림]' % args.collision_trim if hit else ''}")
    finally:
        collision.destroy()
        vehicle.destroy()
        world.apply_settings(original_settings)
        print(f"\n충돌로 조기 종료된 시행: {collisions}/{total_trials}")
        print(f"Cleaned up. Data saved to {args.out}")


if __name__ == "__main__":
    main()
