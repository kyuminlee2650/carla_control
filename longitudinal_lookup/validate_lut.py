r"""LUT feedforward vs. feedforward+PID vs. PID-only -- longitudinal acceleration tracking
validation.

The LUT is a feedforward: given (gear, v_x, a_cmd) it looks up the pedal input it believes
produces a_cmd, with no correction for whatever the table gets wrong. This checks three things --
whether the feedforward alone tracks a commanded acceleration profile (constant or sine, around
--target-accel), how much a PID closing the loop on the LUT's own error improves that, and how
the whole LUT+PID stack compares to a plain PID straight to the pedal with no table at all (the
baseline the LUT is meant to beat, not just kp=ki=kd=0 with the table still doing the work). One
continuous drive, steer forced to 0, on Town06's longest straight (index 86); the same profile
runs through all three back to back and they're compared.

The run launches from rest under the controller being tested (a small proportional law gets it up
to --target-speed first, since a_cmd needs some cruising speed to be a meaningful test) and only
starts logging once v_x/a_x have converged near that baseline -- the a_cmd profile itself only
starts once logging does.

a_meas comes from an IMU (sensor.other.imu), not vehicle.get_acceleration() or a differentiated
v_x -- the former is simulator ground truth, the latter assumes clean velocity, and neither has a
real-hardware equivalent as direct as reading an actual accelerometer. It goes to the controller
raw, not low-pass filtered: this IMU's noise_accel_stddev is 0 (checked on the live blueprint), so
there is no actual sensor noise to remove, and collect_lut_data.py fit the LUT on the same raw
signal -- filtering only here would compare the controller's a_meas against a lagged version of
what the table was calibrated against, adding pure delay (worse phase margin, not better) for no
noise-rejection benefit. It is still clamped to MAX_PLAUSIBLE_ACCEL, which is a different concern:
a fresh spawn's suspension settling, or a gear shift completing, can each spike the reading for one
sample regardless of sensor noise (a live run caught +11.2 m/s^2 the instant a 1st-to-2nd upshift
finished, with the feedforward's own lookup smooth through the same tick), and that would wind the
PID's integral up for a long time -- or, with kp alone, just yank u across a full swing for a tick
that doesn't reflect any real tracking error.

The pedal command u, on the other hand, is low-pass filtered (--u-tau) after the controller and
before apply_control() -- that's smoothing the actuator, not the sensor, so it's a different
question from the a_meas one above: even a perfectly clean control law can compute a raw u that
swings between throttle and brake tick to tick, and asking a real actuator to track that is its
own kind of unrealistic.

Usage (Ubuntu):
    cd ~/carla_control
    python3 longitudinal_lookup/validate_lut.py --profile constant
    python3 longitudinal_lookup/validate_lut.py --profile sine --save-plot

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe longitudinal_lookup\validate_lut.py --times-run 20 --profile constant --save-plot
    .venv\Scripts\python.exe longitudinal_lookup\validate_lut.py --times-run 20 --profile sine --save-plot
"""

import argparse
import math
import os
import queue
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# functions.py (clipping, LowPassFilter), viz_utils.py (follow_with_spectator, plotting) live one
# level up
sys.path.append(os.path.dirname(HERE))

import carla

from functions import LowPassFilter, clipping
from viz_utils import error_stats, follow_with_spectator

from longitudinal_lut import LongitudinalLUT
from lookup_controller import LookupController

MAP_NAME = "Town06"
ORIGIN_INDEX = 86

WARM_START_SPEED_TOL = 0.3   # m/s
WARM_START_ACCEL_TOL = 0.5   # m/s^2
WARM_START_TIMEOUT = 15.0    # s -- safety cap in case the target speed is unreachable

MAX_PLAUSIBLE_ACCEL = 8.0  # m/s^2 -- past what this test ever legitimately commands (a_cmd tops
                           # out at target_accel + amplitude, well under this); a spawn settling or
                           # a gear-shift settling tick can otherwise spike the IMU into the teens
                           # for one sample (confirmed live: +11.2 m/s^2 the instant gear 1->2
                           # completed, feedforward's own lookup was smooth through the same tick),
                           # and feeding that raw into the PID's integral swings u hard for no
                           # reason tied to actual tracking error (see lookup_controller.py)


def body_frame_speed(vehicle):
    yaw = math.radians(vehicle.get_transform().rotation.yaw)
    vel = vehicle.get_velocity()
    return vel.x * math.cos(yaw) + vel.y * math.sin(yaw)


def tick(world, imu_queue):
    """Advance the sim one step and return this tick's IMU accelerometer.x (m/s^2, body-frame
    already), clamped to what the vehicle can plausibly produce. One put() per get() keeps the
    queue in lockstep with world.tick()."""
    world.tick()
    ax = imu_queue.get(timeout=2.0).accelerometer.x
    return clipping(ax, MAX_PLAUSIBLE_ACCEL, -MAX_PLAUSIBLE_ACCEL)


def accel_reference(args, t):
    """cos, not sin: sin(0) = 0 matches the warm-up's a_cmd~0 hand-off, but integrating a sine
    that starts at zero drifts speed one-sided by 2*amplitude/omega before it comes back --
    amplitude=1.5, period=8 pushes v 3.8 m/s up from --target-speed within the first half period,
    enough to run past the LUT's calibrated ceiling on its own. cos(0) = amplitude instead, so the
    integral is a symmetric +-amplitude/omega oscillation around --target-speed (a step in a_cmd
    right at the hand-off, but nothing a PID -- or even the raw feedforward -- can't absorb)."""
    if args.profile == "sine":
        return args.target_accel + args.accel_amplitude * math.sin(2.0 * math.pi * t / args.accel_period)
    return args.target_accel


def run_trial(world, vehicle, controller, args, imu_queue):
    warmed_up = False
    log_start_i = 0
    # smooths the pedal command itself, not the measurement -- a real actuator shouldn't be asked
    # to jump between throttle and brake every tick just because the controller's raw output does
    u_filter = LowPassFilter(tau=args.u_tau, dt=args.dt, initial=0.0)
    hist = {"t": [], "v_x": [], "a_x": [], "a_cmd": [], "u": []}

    steps = int((args.duration + WARM_START_TIMEOUT) / args.dt)
    for i in range(steps):
        step_start = time.time()
        a_meas = tick(world, imu_queue)
        v_x = body_frame_speed(vehicle)
        gear = vehicle.get_control().gear

        if not warmed_up:
            a_cmd = clipping(0.8 * (args.target_speed - v_x), 2.5, -2.5)
        else:
            t = (i - log_start_i) * args.dt
            a_cmd = accel_reference(args, t)

        u = u_filter.step(controller.step(gear, v_x, a_cmd, a_meas))
        control = carla.VehicleControl()
        control.throttle, control.brake = (u, 0.0) if u >= 0 else (0.0, -u)
        control.steer = 0.0
        vehicle.apply_control(control)
        follow_with_spectator(world, vehicle)

        if not warmed_up:
            converged = (abs(v_x - args.target_speed) < WARM_START_SPEED_TOL
                        and abs(a_meas) < WARM_START_ACCEL_TOL)
            timed_out = i * args.dt >= WARM_START_TIMEOUT
            if converged or timed_out:
                warmed_up = True
                log_start_i = i
                controller.reset()  # drop the warm-up law's integral state before scoring starts
                status = "converged" if converged else f"timed out after {WARM_START_TIMEOUT:.0f}s"
                print(f"  warm-start {status}: v_x={v_x:.2f} m/s -- logging starts now")
            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
            continue

        hist["t"].append(t)
        hist["v_x"].append(v_x)
        hist["a_x"].append(a_meas)
        hist["a_cmd"].append(a_cmd)
        hist["u"].append(u)

        if i % 10 == 0:
            print(f"  t={t:5.1f}s   v_x={v_x:5.2f} m/s   a_x={a_meas:+5.2f} m/s^2   "
                  f"a_cmd={a_cmd:+5.2f} m/s^2   u={u:+.2f}")

        if t >= args.duration:
            print(f"  reached duration ({args.duration:.0f}s)")
            break

        elapsed = time.time() - step_start
        if elapsed < args.dt / args.times_run:
            time.sleep(args.dt / args.times_run - elapsed)
    return hist


def report(hist, label):
    err = [c - m for c, m in zip(hist["a_cmd"], hist["a_x"])]
    stats = error_stats(err)
    if stats is None:
        print(f"{label:>18}:  no samples logged")
        return
    rmse, peak, bias = stats
    print(f"{label:>18}:  RMSE={rmse:.3f} m/s^2  max|e|={peak:.3f} m/s^2  mean={bias:+.3f} m/s^2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--duration", type=float, default=30.0, help="scored run length per trial (s)")
    parser.add_argument("--times-run", type=float, default=2.0, help="how times for simulation running?")
    parser.add_argument("--lut", default=os.path.join(HERE, "longitudinal_lut.npz"))

    parser.add_argument("--profile", default="constant", choices=("constant", "sine"),
                        help="a_cmd shape: flat, or a sine wave around --target-accel")
    parser.add_argument("--target-speed", type=float, default=5.0,
                        help="m/s; baseline cruise speed the run launches to before a_cmd starts")

    parser.add_argument("--target-accel", type=float, default=0.0, help="m/s^2; sine profile's midline")
    parser.add_argument("--accel-amplitude", type=float, default=1.5, help="sine profile peak deviation (m/s^2)")
    parser.add_argument("--accel-period", type=float, default=8.0, help="sine profile period (s)")

    parser.add_argument("--kp", type=float, default=0.15, help="feedforward+PID trial's proportional gain")
    parser.add_argument("--ki", type=float, default=0.6, help="feedforward+PID trial's integral gain")
    parser.add_argument("--kd", type=float, default=0.0, help="feedforward+PID trial's derivative gain")
    parser.add_argument("--u-tau", type=float, default=0.2, help="low-pass filter time constant on the pedal command u, before it's applied (s)")

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"),
                         help="directory to save the end-of-run comparison figure into")
    parser.add_argument("--save-plot", action="store_true",
                        help="draw and save the end-of-run comparison figure (off by default)")
    args = parser.parse_args()

    lut = LongitudinalLUT(args.lut)

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
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    trials = [
        ("feedforward only", 0.0, 0.0, 0.0, True),
        ("feedforward + PID", args.kp, args.ki, args.kd, True),
        ("PID only", args.kp, args.ki, args.kd, False),
    ]
    results = {}
    try:
        for label, kp, ki, kd, use_ff in trials:
            vehicle = world.spawn_actor(blueprint, origin_transform)
            imu = None
            try:
                world.tick()
                # spawned after the priming tick above, so the first tick() call in run_trial()
                # produces this sensor's first queued sample -- one put() per get() keeps them in
                # lockstep for the whole trial
                imu_queue = queue.Queue()
                imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
                imu.listen(imu_queue.put)

                controller = LookupController(lut, kp=kp, ki=ki, kd=kd, dt=args.dt,
                                               use_feedforward=use_ff)
                print(f"\n=== {label} ===")
                results[label] = run_trial(world, vehicle, controller, args, imu_queue)
            finally:
                if imu is not None and imu.is_alive:
                    imu.stop()
                    imu.destroy()
                vehicle.destroy()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("\nCleaned up: world settings restored.")

    if len(results) < 2:
        return

    print(f"\n=== accel tracking, {args.profile} profile ===")
    for label, _, _, _, _ in trials:
        if label in results:
            report(results[label], label)

    if args.save_plot:
        from viz_utils import plot_lut_validation
        plot_lut_validation(results["feedforward only"], results["feedforward + PID"],
                            "accel", args, args.plot_dir,
                            hist_pidonly=results.get("PID only"))


if __name__ == "__main__":
    main()
