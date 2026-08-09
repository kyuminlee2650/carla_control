r"""Longitudinal control-input calibration sweep -- data collection.

Builds the raw data for a (gear, v_x, a_x) -> u lookup table, where u in [-1, 1] is the
longitudinal control input (positive = throttle, zero = coasting, negative = brake). For each
gear, drives at a constant u from rest up to the sweep's top speed (throttle) or from that speed
down to rest (coast/brake), logging (v_x, a_x) every tick. One trial per (gear, u) covers the
full speed range for that pair, so this is much cheaper than gridding over (gear, v, u) directly.

a_x comes from an IMU (sensor.other.imu), not vehicle.get_acceleration() -- the latter is
simulator ground truth with nothing to match on real hardware, where the only way to know
acceleration is to read it off an actual sensor.

Throttle trials start from rest, which needs no warm-up: v=0 is a state the wheels are already
consistent with. Decel (coast/brake) trials need to start already moving, and teleporting the body
to a speed with set_target_velocity() leaves the wheels' own spin at whatever it was -- the same
mismatch that showed up elsewhere in this repo as a bogus sharp deceleration right after launch.
So instead each decel trial is preceded by a real warm-up: drive from rest under automatic
transmission and a simple proportional pedal until v_x settles near --brake-start-speed with a_x
near zero, then switch into the trial's own gear/input right there, with no teleport in between.

Besides the per-gear sweep (gear forced via manual_gear_shift), one more pass runs every u with
automatic transmission -- gear is whatever CARLA's own shift logic selects at each tick, which is
what a controller sees in practice. Both passes write into the same CSV; build_lut.py buckets rows
by whatever gear ended up in the "gear" column, so the automatic pass's rows simply add more
samples (including at the shift boundaries the manual sweep never visits) to those same buckets.

Spawn point is fixed to Town06 index 86 (the longest straight on that map, ~864 m) -- a trial
holds one input for a while and needs road ahead of it, and picking that dynamically on every run
is unnecessary once the map is fixed.

A collision sensor ends a trial on impact and the last --collision-trim seconds are discarded, so
scenery the vehicle drove into never reaches the table.

Output is a CSV of raw samples -- inverting it into a gear/v/a -> u table for runtime lookup is
build_lut.py, a separate step. Pass --save-plot for a per-gear (v_x, a_x) distribution figure of
what just got collected (no cleaning applied -- see build_lut.py for that), or --record for an mp4
of the whole sweep.

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python longitudinal_lookup/collect_lut_data.py

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe longitudinal_lookup\collect_lut_data.py
    .venv\Scripts\python.exe longitudinal_lookup\collect_lut_data.py --gears 1,2,3 --throttle-step 0.2
    .venv\Scripts\python.exe longitudinal_lookup\collect_lut_data.py --skip-auto
    .venv\Scripts\python.exe longitudinal_lookup\collect_lut_data.py --save-plot --record
"""

import argparse
import csv
import math
import os
import queue
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# functions.py (CollisionWatch, clipping), viz_utils.py (recorder/plotting) live one level up
sys.path.append(os.path.dirname(HERE))

import carla

from functions import CollisionWatch, clipping
from viz_utils import VIEWS, VideoRecorder, follow_with_spectator, plot_lut_raw_distribution, run_name

MAP_NAME = "Town06"
ORIGIN_INDEX = 86


def body_frame_speed(vehicle):
    """Project world-frame velocity onto the vehicle's forward axis."""
    yaw = math.radians(vehicle.get_transform().rotation.yaw)
    vel = vehicle.get_velocity()
    return vel.x * math.cos(yaw) + vel.y * math.sin(yaw)


def tick(world, imu_queue):
    """Advance the sim one step and return this tick's IMU accelerometer.x (m/s^2, body-frame
    already -- no yaw projection needed, unlike get_acceleration()). One put() per get() keeps the
    queue in lockstep with world.tick(), so every tick anywhere in this file must go through here."""
    world.tick()
    return imu_queue.get(timeout=2.0).accelerometer.x


def apply_long_control(vehicle, gear, u):
    """gear=None leaves the transmission automatic; otherwise forces that gear."""
    control = carla.VehicleControl()
    if gear is not None:
        control.manual_gear_shift = True
        control.gear = gear
    control.hand_brake = False
    if u >= 0:
        control.throttle = u
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = -u
    vehicle.apply_control(control)


def reset_at_rest(world, vehicle, origin_transform, imu_queue):
    """Teleport to the origin at rest -- v=0 needs no warm-up, see warm_up_to_speed()."""
    vehicle.set_transform(origin_transform)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    tick(world, imu_queue)
    tick(world, imu_queue)  # let the teleport settle before anything reads the vehicle's state


def warm_up_to_speed(world, vehicle, target_speed, dt, args, imu_queue):
    """Drive up from rest to target_speed under automatic transmission, so a decel trial can
    start from real motion instead of a teleported one. Returns once v_x/a_x settle within
    tolerance of the target, or after --warmup-timeout regardless."""
    start_loc = vehicle.get_transform().location
    for _ in range(int(args.warmup_timeout / dt)):
        v_x = body_frame_speed(vehicle)
        u = clipping(0.15 * (target_speed - v_x), 1.0, -1.0)
        apply_long_control(vehicle, None, u)
        a_x = tick(world, imu_queue)
        follow_with_spectator(world, vehicle)
        if vehicle.get_transform().location.distance(start_loc) > args.max_distance:
            break  # ran out of straight before settling
        v_x = body_frame_speed(vehicle)
        if abs(v_x - target_speed) < args.warmup_speed_tol and abs(a_x) < args.warmup_accel_tol:
            break
    return vehicle.get_transform().location


def flush_trial(rows, collided, args, writer):
    """Write a trial's rows, dropping the tail if it ended in a collision."""
    if collided:
        drop = int(round(args.collision_trim / args.dt))
        rows = rows[:-drop] if drop < len(rows) else []
    for row in rows:
        writer.writerow(row)
    return len(rows)


def run_throttle_trial(world, vehicle, origin_transform, gear, u, dt, args, writer, collision, imu_queue):
    reset_at_rest(world, vehicle, origin_transform, imu_queue)
    collision.arm()
    start_loc = vehicle.get_transform().location
    plateau_count = 0
    rows = []
    t = 0.0
    while t < args.max_duration:
        apply_long_control(vehicle, gear, u)
        a_x = tick(world, imu_queue)
        follow_with_spectator(world, vehicle)
        v_x = body_frame_speed(vehicle)
        engaged = gear if gear is not None else vehicle.get_control().gear
        rows.append([engaged, u, round(t, 3), round(v_x, 4), round(a_x, 4)])

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


def run_decel_trial(world, vehicle, origin_transform, gear, u, dt, args, writer, collision, imu_queue):
    """Coast (u = 0) or brake (u < 0) down from the sweep's top speed."""
    reset_at_rest(world, vehicle, origin_transform, imu_queue)
    start_loc = warm_up_to_speed(world, vehicle, args.brake_start_speed, dt, args, imu_queue)
    collision.arm()
    rows = []
    t = 0.0
    while t < args.max_duration:
        apply_long_control(vehicle, gear, u)
        a_x = tick(world, imu_queue)
        follow_with_spectator(world, vehicle)
        v_x = body_frame_speed(vehicle)
        engaged = gear if gear is not None else vehicle.get_control().gear
        rows.append([engaged, u, round(t, 3), round(v_x, 4), round(a_x, 4)])

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
    parser.add_argument("--skip-auto", action="store_true",
                        help="skip the extra automatic-transmission pass, sweep manual gears only")
    parser.add_argument("--throttle-min", type=float, default=0.1)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--throttle-step", type=float, default=0.1)
    parser.add_argument("--brake-min", type=float, default=0.1)
    parser.add_argument("--brake-max", type=float, default=1.0)
    parser.add_argument("--brake-step", type=float, default=0.1)
    parser.add_argument("--brake-start-speed", type=float, default=18.0, help="target speed a decel trial's warm-up climbs to before the trial itself starts (m/s)")
    parser.add_argument("--warmup-timeout", type=float, default=15.0, help="give up waiting for the warm-up to settle after this long (s)")
    parser.add_argument("--warmup-speed-tol", type=float, default=0.3, help="warm-up counts as settled once |v_x - target| is under this (m/s)")
    parser.add_argument("--warmup-accel-tol", type=float, default=0.5, help="warm-up counts as settled once |a_x| is under this (m/s^2)")
    parser.add_argument("--max-speed", type=float, default=18.0,
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

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"),
                         help="directory to save the raw-data distribution figure into")
    parser.add_argument("--save-plot", action="store_true",
                        help="draw and save a per-gear (v_x, a_x) distribution figure of the sweep (off by default)")

    # ---- video ---- #
    parser.add_argument("--record", nargs="?", const="auto", default="",
                        help="record the whole sweep to an mp4; bare flag auto-names it under --video-dir")
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

    origin_transform = world.get_map().get_spawn_points()[ORIGIN_INDEX]

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    vehicle = world.spawn_actor(blueprint, origin_transform)
    world.tick()

    # spawned after the priming tick above, so the first tick() call below produces this
    # sensor's first queued sample -- one put() per get() keeps them in lockstep for the run
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")
    imu_queue = queue.Queue()
    imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    imu.listen(imu_queue.put)

    recorder = None
    if args.record:
        video_path = (os.path.join(args.video_dir, run_name() + ".mp4")
                      if args.record == "auto" else args.record)
        rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
        recorder = VideoRecorder(world, vehicle, video_path, fps=1.0 / args.dt,
                                 width=rec_w, height=rec_h, view=args.record_view)

    if args.gears:
        gears = [int(g) for g in args.gears.split(",")]
    else:
        gears = list(range(1, len(vehicle.get_physics_control().forward_gears) + 1))
    passes = [(g, f"gear {g}") for g in gears]
    if not args.skip_auto:
        passes.append((None, "auto"))
    print(f"Passes to sweep: {[label for _, label in passes]}")

    throttles = frange(args.throttle_min, args.throttle_max, args.throttle_step)
    # u = 0 is coasting: engine braking and drag with neither pedal applied. It is the input a
    # controller reaches for most often when it wants to shed a little speed, and skipping it
    # would leave the table interpolating straight across the gap between the smallest brake and
    # the smallest throttle.
    decels = [0.0] + [-b for b in frange(args.brake_min, args.brake_max, args.brake_step)]
    print(f"Throttle grid: {throttles}")
    print(f"Coast/brake grid: {decels}")

    total_trials = len(passes) * (len(throttles) + len(decels))
    done = 0
    collisions = 0

    collision = CollisionWatch(world, vehicle)
    try:
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gear", "u", "t", "v_x", "a_x"])

            for gear, label in passes:
                for u in throttles:
                    done += 1
                    n, hit = run_throttle_trial(world, vehicle, origin_transform, gear, u,
                                                args.dt, args, writer, collision, imu_queue)
                    collisions += hit
                    print(f"[{done}/{total_trials}] {label} u={u:+.2f} (throttle) "
                          f"-> {n} samples{'  [collision: last %.1fs dropped]' % args.collision_trim if hit else ''}")

                for u in decels:
                    done += 1
                    kind = "coast" if u == 0.0 else "brake"
                    n, hit = run_decel_trial(world, vehicle, origin_transform, gear, u,
                                             args.dt, args, writer, collision, imu_queue)
                    collisions += hit
                    print(f"[{done}/{total_trials}] {label} u={u:+.2f} ({kind}) "
                          f"-> {n} samples{'  [collision: last %.1fs dropped]' % args.collision_trim if hit else ''}")
    finally:
        collision.destroy()
        if recorder is not None:
            recorder.close()  # before vehicle.destroy(): the camera is attached to it
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        vehicle.destroy()
        world.apply_settings(original_settings)
        print(f"\nTrials ended early by collision: {collisions}/{total_trials}")
        print(f"Cleaned up. Data saved to {args.out}")

    if args.save_plot:
        raw = np.genfromtxt(args.out, delimiter=",", names=True)
        plot_lut_raw_distribution(raw, args.plot_dir)


if __name__ == "__main__":
    main()
