r"""Drive the cornering-stiffness sweep, judge which trials reached steady state, and log ONLY
that steady-state data -- no fitting, no Cf/Cr, nothing Iz-dependent. Split out of what used to be
one estimate_cornering_stiffness.py so the *driving* (expensive, needs CARLA) and the *estimation*
(cheap, pure post-processing on already-steady data) can be run and rerun independently.

Steady-state judgment (add_derivatives/find_steady_windows, --steady-hold/--thresh-*) lives here,
not in estimate_cornering_stiffness.py: whether a trial settles is a fact about the drive itself
(speed, steer angle, PID tuning), not something to decide after the fact. A trial that never
reaches a steady window is not written to --out at all -- "never settled" prints immediately, so a
PID-tuning sweep chasing a resonance-looking failure to settle at some speed (e.g. the
longitudinal speed-hold PID beating against the lateral dynamics at a particular speed) gets an
instant per-trial verdict without a separate estimation pass. What IS saved is trimmed to exactly
the steady window (slice_log()): the array in --out's "log" for a trial is the steady-state data,
full stop -- estimate_cornering_stiffness.py never needs to find or trust a window of its own.

The pad and vehicle spawn are unchanged from before this split -- only the fitting/pooling/
selection logic moved out, into estimate_cornering_stiffness.py, and the steady-state judgment
moved IN, from there.

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
stock maps.

Combinations are filtered before anything drives: R < --min-radius breaks the bicycle model's
small-angle assumption (~9% kinematic error at R=5m, ~4% at R=8m), and a_y > --max-ay is skipped
because higher lateral acceleration measurably increases body roll here -- a live IMU-vs-kinematic
gap check saw ~0.16 m/s^2 of roll-induced bias at just 2.6 m/s^2 of a_y, exactly the contamination
attach_imu()'s CoM mounting is built to avoid.

The full default --speeds x --steers grid is a lot of real time (each trial drives --duration
seconds of simulated time). --out defaults to MERGING this run's trials into whatever is already
there rather than overwriting it (keyed by (target_speed, steer_deg)), so a sweep can be split
across several shorter invocations instead of one long one -- see the --speeds 1,2,3 / 4,5,6
example below. --fresh discards the existing file instead.

Usage (Ubuntu):
    cd ~/carla_control
    python3 lateral_parameter/collect_cornering_data.py
    python3 lateral_parameter/collect_cornering_data.py --speeds 4,5,6,7,8,9,10
    python3 lateral_parameter/collect_cornering_data.py --record
    # split a full sweep across several runs -- each one merges into the same --out:
    python3 lateral_parameter/collect_cornering_data.py --speeds 1,2,3
    python3 lateral_parameter/collect_cornering_data.py --speeds 4,5,6
    python3 lateral_parameter/collect_cornering_data.py --speeds 7,8,9,10

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe lateral_parameter\collect_cornering_data.py
    .venv\Scripts\python.exe lateral_parameter\collect_cornering_data.py --speeds 1,2,3
    .venv\Scripts\python.exe lateral_parameter\collect_cornering_data.py --speeds 4,5,6
"""

import argparse
import json
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

from viz_utils import VIEWS, VideoRecorder, follow_with_spectator, run_name

from bicycle import front_steer_angle, zero_phase_derivative


LANE_WIDTH = 3.5
VEHICLE_BP = "vehicle.lincoln.mkz_2020"


# --------------------------------------------------------------------------------------------
# pad + vehicle
# --------------------------------------------------------------------------------------------

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


def spawn_vehicle(world, centre, blueprint_filter=VEHICLE_BP, ride_height=0.3, need_geometry=True):
    """need_geometry=False skips get_vehicle_geometry() (and the wheelbase/lf/lr/max_steer line it
    prints) -- geometry is identical for every trial (same blueprint, same spawn pose), so only
    the one-off probe spawn in collect_sweep_logs() needs to compute or print it; a fresh vehicle
    spawned per trial does not, and printing it again every trial was pure noise."""
    transform = carla.Transform(carla.Location(x=centre[0], y=centre[1], z=ride_height),
                                carla.Rotation(yaw=0.0))
    blueprint = world.get_blueprint_library().filter(blueprint_filter)[0]
    vehicle = world.spawn_actor(blueprint, transform)
    stop(vehicle)

    if not need_geometry:
        return vehicle, None

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


# --------------------------------------------------------------------------------------------
# drive + log
# --------------------------------------------------------------------------------------------

def drive_and_log(world, vehicle, physics, max_steer, args, target_speed, delta, imu_queue,
                  thresholds=None, verbose=True):
    """Settle onto the surface, then hold `delta` (bicycle-model wheel angle, rad) while a PID
    chases `target_speed`, logging every tick. Returns the log dict (derivatives not added --
    that, and everything downstream of it, is estimate_cornering_stiffness.py's job now).

    A fresh PID/LowPassFilter every call, not one shared across a whole sweep -- otherwise trial
    N's integral windup and filter state would bias trial N+1's transient, which would then bleed
    into how quickly it reaches (and stays inside) the steady window.

    verbose prints one line per simulated second: target vs actual v_x (the number to watch for a
    longitudinal-PID oscillation -- a speed that keeps swinging past target and back, rather than
    settling toward it, means the PID/--speed-filter-tau tuning is the problem, not the lateral
    side), plus v_y/psi_dot/a_y and a live "settled now?" read. That live read is a CAUSAL
    approximation (this printed second vs the last one, not the offline zero-phase derivative over
    a 0.4 s window collect_sweep_logs() uses afterwards to actually decide the steady window) --
    differencing tick-to-tick instead (0.05 s apart) was tried first and was unusable: dividing raw
    IMU noise by dt=0.05 amplifies it 20x, so it read "settled now: no" continuously even with v_x
    dead on target. Differencing across the full printed second is still noisier than the offline
    computation and can disagree in the fine print, but it is enough to see an oscillation never
    dying out versus a transient that is decaying.

    thresholds (if given) also drives an early stop: once the live "settled now?" read has been YES
    for --steady-hold seconds back to back, driving continues --settle-extra seconds longer (to
    bank some steady data beyond the bare minimum the offline window needs) and then stops -- no
    reason to keep driving the full --duration once a trial is clearly settled, and most of a
    50 s --duration was otherwise being spent doing exactly that. A trial that never settles still
    runs the full --duration and comes back with "never settled", unchanged. thresholds=None (the
    default) disables this and always runs the full --duration, same as before this existed.
    """
    pid = PID(kp=args.pid_kp, ki=args.pid_ki, kd=args.pid_kd, dt=args.dt)
    speed_filter = LowPassFilter(tau=args.speed_filter_tau, dt=args.dt, initial=0.0)

    # The settle ticks double as the IMU's warm-up. CARLA derives the accelerometer from
    # velocity differences and reports garbage for the first tick or two after spawn -- around
    # -378,000 m/s^2 was measured -- so those samples get consumed here rather than logged.
    for _ in range(args.settle_ticks):
        world.tick()
        imu_queue.get(timeout=8.0)
        follow_with_spectator(world, vehicle)

    log = {k: [] for k in ("t", "x", "y", "yaw", "v_x", "v_y", "r",
                           "a_x_imu", "a_y_imu", "delta", "delta_measured", "steer_cmd",
                           "throttle", "brake")}

    # Wall-clock budget per step. The simulator advances by args.dt of *simulated* time on every
    # tick regardless of how long that took to compute, so pacing is entirely up to the client:
    # sleeping until dt/times_run has passed makes one second of simulation take 1/times_run
    # seconds of real time.
    budget = args.dt / args.times_run
    started, behind = time.time(), 0
    # (t, v_x, v_y, r, a_y_imu) at the last PRINTED second, for the live "settled now?" read.
    # Differencing across the full ~1 s print interval, not one 0.05 s tick, is what keeps this
    # from drowning in raw IMU noise -- a_y_imu is the unfiltered accelerometer, and dividing a
    # single tick's noise by dt=0.05 amplifies it 20x, which made every printed line read
    # "settled now: no" even while v_x sat dead on target. add_derivatives() avoids exactly this
    # with a wide 0.4 s zero-phase window; this live read is a cruder version of the same idea.
    prev_print = None
    settled_since = None   # sim time (s) the live "settled now?" read first turned YES, back to
                           # back with no "no" in between -- None whenever it's currently reading no
    last_i = 0

    for i in range(int(args.duration / args.dt)):
        last_i = i
        step_start = time.time()
        world.tick()
        imu_data = imu_queue.get(timeout=8.0)

        transform = vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        location = transform.location
        velocity = vehicle.get_velocity()
        # Body frame: v_x is what the steering curve is indexed on and what the bicycle model
        # divides by. |v| would be close but not equal -- they differ by cos(beta), and beta
        # reaches several degrees in a tight circle.
        v_x = velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)
        v_y = -velocity.x * math.sin(yaw) + velocity.y * math.cos(yaw)
        a_y_imu = imu_data.accelerometer.y

        e_vel = target_speed - v_x
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
        log["a_y_imu"].append(a_y_imu)
        log["delta"].append(delta); log["delta_measured"].append(front_steer_angle(vehicle))
        log["steer_cmd"].append(control.steer)
        log["throttle"].append(control.throttle); log["brake"].append(control.brake)

        stop_now = False
        if i % int(1.0 / args.dt) == 0:
            t_now = i * args.dt
            settled_now = None
            if thresholds is not None and prev_print is not None:
                dt_window = t_now - prev_print[0]
                v_x_dot = (v_x - prev_print[1]) / dt_window
                v_y_dot = (v_y - prev_print[2]) / dt_window
                r_dot = (r - prev_print[3]) / dt_window
                a_y_dot = (a_y_imu - prev_print[4]) / dt_window
                settled_now = (abs(v_x_dot) < thresholds["v_x_dot"]
                              and abs(v_y_dot) < thresholds["v_y_dot"]
                              and abs(r_dot) < thresholds["psi_ddot"]
                              and abs(a_y_dot) < thresholds["a_y_dot"])
                if settled_now:
                    if settled_since is None:
                        settled_since = t_now
                    elif t_now - settled_since >= args.steady_hold + args.settle_extra:
                        stop_now = True
                else:
                    settled_since = None

            if verbose:
                settled_str = ("" if settled_now is None
                               else ("  settled now: YES" if settled_now else "  settled now: no"))
                print(f"t={t_now:6.2f}s  v_x={v_x:6.3f}/{target_speed:.1f} (e={e_vel:+.3f})  "
                      f"v_y={v_y:+6.3f}  psi_dot={math.degrees(r):+7.2f} deg/s  "
                      f"a_y={a_y_imu:+6.3f}  throttle={control.throttle:.2f} "
                      f"brake={control.brake:.2f}{settled_str}")
                if stop_now:
                    print(f"  confirmed steady since t={settled_since:.2f}s -- stopping early at "
                          f"t={t_now:.2f}s ({t_now - settled_since:.1f}s of confirmed steady data) "
                          f"instead of driving the full {args.duration:.0f}s")
            prev_print = (t_now, v_x, v_y, r, a_y_imu)

        elapsed = time.time() - step_start
        if elapsed < budget:
            time.sleep(budget - elapsed)
        else:
            behind += 1

        if stop_now:
            break

    if verbose:
        wall = time.time() - started
        sim = (last_i + 1) * args.dt   # actual simulated time driven -- may be < --duration if
                                       # the early-stop above fired
        print(f"\n{sim:.1f}s of simulation in {wall:.1f}s of real time "
              f"({sim/wall:.2f}x, asked for {args.times_run:.2f}x)")
        if behind:
            print(f"  {behind} of {int(sim/args.dt)} steps missed the {budget*1000:.0f} ms budget "
                  f"-- the simulator could not keep up, so it ran slower than requested")

    return log


def make_video_factory(world, args):
    """--record -> a collect_sweep_logs() video_factory(vehicle, trial) that opens one
    VideoRecorder per (speed, steer) trial, named after that trial so a whole sweep's worth of
    clips don't overwrite each other."""
    if not args.record:
        return None

    rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))

    def factory(vehicle, trial):
        suffix = f"v{trial['target_speed']:g}_d{trial['steer_deg']:g}"
        if args.record == "auto":
            video_path = os.path.join(args.video_dir, run_name(suffix) + ".mp4")
        else:
            base, ext = os.path.splitext(args.record)
            video_path = f"{base}_{suffix}{ext}"
        return VideoRecorder(world, vehicle, video_path, fps=1.0 / args.dt,
                             width=rec_w, height=rec_h, view=args.record_view)

    return factory


# --------------------------------------------------------------------------------------------
# steady-state detection
# --------------------------------------------------------------------------------------------
#
# Lives here, not in estimate_cornering_stiffness.py: whether a trial settled is a fact about the
# *drive* (this trial's speed, steer angle, and PID tuning), not a choice the estimator should be
# making after the fact. Judging it here means a trial that never reached steady state is simply
# never saved -- estimate_cornering_stiffness.py can then trust that every logged sample IS
# steady-state data, with no window-finding or thresholds of its own. It also means a PID-tuning
# sweep (chasing a resonance-looking failure to settle at some speed) gets an immediate
# settled/never-settled verdict per trial, without a separate estimation pass.

def add_derivatives(log, dt, window_s=0.4):
    """Differentiate the logged signals, zero-phase.

    Zero-phase rather than causal, because every use of these derivatives is a comparison against
    something measured at the same instant -- a causal filter would manufacture a discrepancy out
    of its own phase shift. Computed on the FULL trace before any trimming (see slice_log()): the
    Savitzky-Golay window needs samples on both sides of every point, so differentiating an
    already-trimmed steady window would smear its edges.

    The window is wide (0.4 s) on purpose: these mostly answer "has this settled", a low-frequency
    question, and a narrow window would just pass per-tick noise through. estimate_cornering_
    stiffness.py's with_iz method reuses psi_ddot/v_y_dot straight from the trimmed log rather
    than re-differentiating, so the steady-window gate and the with_iz force balance always agree
    on what these signals were.
    """
    log["v_x_dot"] = zero_phase_derivative(log["v_x"], dt, window_s=window_s).tolist()
    log["v_y_dot"] = zero_phase_derivative(log["v_y"], dt, window_s=window_s).tolist()
    log["psi_ddot"] = zero_phase_derivative(log["r"], dt, window_s=window_s).tolist()
    log["a_y_dot"] = zero_phase_derivative(log["a_y_imu"], dt, window_s=window_s).tolist()
    return log


def steady_mask(log, thresholds):
    """Per-tick bool: are v_x, v_y, psi_dot and a_y all momentarily unchanging.

    "Unchanging" is judged on the derivatives added by add_derivatives(), not on the raw signals
    -- steady cornering has v_x, v_y, r and a_y sitting at some nonzero constant, not at zero, so
    it is their rate of change that has to be small, not their value. Each gets its own threshold
    in `thresholds` (keys "v_x_dot", "v_y_dot", "psi_ddot", "a_y_dot") because the four live on
    different scales: 1 deg/s^2 of psi_ddot and 1 m/s^2 of v_x_dot are not the same size of
    disturbance.
    """
    return (
        (np.abs(log["v_x_dot"]) < thresholds["v_x_dot"]) &
        (np.abs(log["v_y_dot"]) < thresholds["v_y_dot"]) &
        (np.abs(log["psi_ddot"]) < thresholds["psi_ddot"]) &
        (np.abs(log["a_y_dot"]) < thresholds["a_y_dot"])
    )


def tick_jump_mask(log, max_jump):
    """Per-tick bool: is the raw jump from the previous tick to this one within a physically
    plausible size, for every signal named in `max_jump` ({"v_x": limit, "v_y": limit, "r":
    limit}, units are the signal's own per TICK, not per second). Index 0 is always True (nothing
    to compare it against).

    This exists because add_derivatives()'s 0.4 s smoothing structurally CANNOT see a specific
    failure mode: a small number of ticks where a raw signal (observed: gyroscope r) alternates
    between two distinct values every single tick -- a period-2 oscillation, the highest frequency
    a discrete signal can carry. A low-order polynomial smooth (Savitzky-Golay, same as
    add_derivatives() uses) averages that kind of alternation to within a few thousandths of a
    deg/s^2 REGARDLESS of how large the raw swing is (measured case: r alternating between 0.50
    and 0.61 rad/s, a jump big enough to imply ~123 deg/s^2 raw, read back at ~0.003-0.009 deg/s^2
    once smoothed -- three orders of magnitude below the 0.01 deg/s^2 threshold that was supposed
    to catch exactly this). The result: a window fully corrupted by this glitch on the RAW signal
    still reads as perfectly steady on the SMOOTHED one, and fit_cornering_stiffness_zero_moment/
    with_iz() use the raw v_x/v_y/r directly, so the corruption goes straight into Cf/Cr with no
    warning. Checking raw tick-to-tick jumps directly is the only way to see it: no window can
    smooth it away because smoothing is exactly what hides it in the first place.

    Root cause not confirmed (suspected: an IMU sensor-callback/queue hiccup during a real-time
    pacing stall -- see drive_and_log()'s "steps missed the ms budget" warning), so this is a
    symptom detector, not a fix for whatever produces the symptom.
    """
    n = len(log["t"])
    ok = np.ones(n, dtype=bool)
    for key, limit in max_jump.items():
        arr = np.asarray(log[key], dtype=float)
        ok[1:] &= np.abs(np.diff(arr)) <= limit
    return ok


def runs(mask):
    """Contiguous [start, end) index ranges where `mask` is True, in order, any length."""
    out = []
    run_start = None
    for i, ok in enumerate(mask):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            out.append((run_start, i))
            run_start = None
    if run_start is not None:
        out.append((run_start, len(mask)))
    return out


def find_steady_windows(log, dt, thresholds, hold_s=2.0, max_tick_jump=None):
    """Contiguous [start, end) index ranges where v_x, v_y, psi_dot and a_y have all stopped
    changing for at least `hold_s` seconds, AND (if max_tick_jump is given) no raw tick-to-tick
    jump in that span exceeded a physically plausible size (see tick_jump_mask() -- a smoothed-
    derivative gate alone cannot see that kind of glitch, by construction).

    A single instant under threshold proves nothing -- noise alone crosses back and forth. Only a
    run of at least `hold_s` seconds where every one of the four stayed under its threshold
    counts, which is what turns this into "settled", not just "quiet right now".
    """
    mask = steady_mask(log, thresholds)
    if max_tick_jump is not None:
        mask = mask & tick_jump_mask(log, max_tick_jump)
    hold_ticks = max(1, int(round(hold_s / dt)))
    return [(s, e) for s, e in runs(mask) if e - s >= hold_ticks]


def longest_window(windows):
    """The longest (start, end) run, or None if `windows` is empty."""
    return max(windows, key=lambda w: w[1] - w[0]) if windows else None


def slice_log(log, window):
    """Trim every array-valued key in `log` to the [start, end) window.

    Called once a steady window is found, so what gets saved to disk IS the steady-state data --
    the whole array, nothing to slice further downstream. estimate_cornering_stiffness.py never
    needs to know a `window` existed at all.
    """
    s, e = window
    return {k: v[s:e] for k, v in log.items()}


def collect_sweep_logs(world, centre, args, thresholds, max_tick_jump=None, video_factory=None):
    """Drive every (speed, steer) combination in --speeds x --steers, filtered by
    --min-radius/--max-ay, and log only the settled portion of each -- no Cf/Cr fitting, but the
    steady-state judgment itself (find_steady_windows()) happens here, not in
    estimate_cornering_stiffness.py.

    video_factory(vehicle, trial) -> VideoRecorder|None, called right after a trial's vehicle is
    spawned, before that trial drives. If it returns a recorder, this closes it once the trial's
    drive finishes, before vehicle.destroy() (VideoRecorder's own lifecycle requirement -- the
    camera is attached to the vehicle). None (the default) records nothing.

    Returns (trials, mass, lf, lr, wheelbase, max_steer). trials is every queued combination, each
    with "log": None for ones that failed to drive at all OR never reached a steady window --
    left in (not dropped) so the printed summary accounts for every combination that was asked
    for, but main() drops the None ones before saving, so --out only ever holds steady data.
    """
    speeds = [float(v) for v in args.speeds.split(",")]
    steers = [float(d) for d in args.steers.split(",")]

    probe, geometry = spawn_vehicle(world, centre, VEHICLE_BP)
    wheelbase, lf, lr, max_steer, mass, com = geometry
    probe.destroy()

    trials = []
    for v in speeds:
        for d in steers:
            delta = math.radians(d)
            if abs(delta) > max_steer:
                print(f"  skip v={v:.1f} delta={d:.1f}deg: exceeds max_steer "
                      f"{math.degrees(max_steer):.1f}deg")
                continue
            R = wheelbase / math.tan(abs(delta))
            a_y_est = v * v / R
            if R < args.min_radius:
                print(f"  skip v={v:.1f} delta={d:.1f}deg: R={R:.1f}m < --min-radius "
                      f"{args.min_radius:.1f}m")
                continue
            if a_y_est > args.max_ay:
                print(f"  skip v={v:.1f} delta={d:.1f}deg: a_y~{a_y_est:.2f} m/s^2 > --max-ay "
                      f"{args.max_ay:.1f} m/s^2")
                continue
            trials.append({"target_speed": v, "steer_deg": d, "R": R, "a_y_est": a_y_est})

    print(f"\n{len(trials)} of {len(speeds)*len(steers)} (speed, steer) combinations queued "
          f"(mass={mass:.0f} kg  lf={lf:.2f} m  lr={lr:.2f} m)")
    if not trials:
        print("nothing to run -- widen --speeds/--steers or relax --min-radius/--max-ay")
        return trials, mass, lf, lr, wheelbase, max_steer

    for n, trial in enumerate(trials, 1):
        v, d = trial["target_speed"], trial["steer_deg"]
        print(f"\n[{n}/{len(trials)}] v={v:.1f} m/s  delta={d:.1f}deg  "
              f"R~{trial['R']:.1f}m  a_y~{trial['a_y_est']:.2f} m/s^2")
        trial["log"] = None
        vehicle = imu = log = recorder = None
        try:
            vehicle, _ = spawn_vehicle(world, centre, VEHICLE_BP, need_geometry=False)
            physics = vehicle.get_physics_control()
            imu, imu_queue = attach_imu(world, vehicle, physics.center_of_mass)
            if video_factory is not None:
                recorder = video_factory(vehicle, trial)
            log = drive_and_log(world, vehicle, physics, max_steer, args, v, math.radians(d),
                                imu_queue, thresholds=thresholds, verbose=not args.quiet)
        except Exception as exc:
            # One trial's transient hiccup (a slow tick missing the IMU's 5s window, a spawn
            # collision, ...) should not lose every trial that already logged before it, nor the
            # ones still queued after it -- print, clean up what exists, and move on.
            print(f"  trial failed ({exc!r}) -- skipping")
        finally:
            if recorder is not None:
                recorder.close()   # before vehicle.destroy(): the camera is attached to it
            if imu is not None:
                imu.stop()
                imu.destroy()
            if vehicle is not None:
                vehicle.destroy()

        if log is None:
            trial["log"] = None
            print("  logging failed")
            continue

        add_derivatives(log, args.dt)
        window = longest_window(find_steady_windows(log, args.dt, thresholds,
                                                     hold_s=args.steady_hold,
                                                     max_tick_jump=max_tick_jump))
        if window is None:
            trial["log"] = None
            print(f"  never settled ({len(log['t'])} ticks driven, "
                  f"{args.duration:.0f}s) -- not logging this trial")
            continue

        t = log["t"]
        print(f"  steady t={t[window[0]]:.2f}-{t[window[1]-1]:.2f}s "
              f"({t[window[1]-1]-t[window[0]]:.2f}s) -- logging {window[1]-window[0]} of "
              f"{len(t)} ticks")
        trial["log"] = slice_log(log, window)

    return trials, mass, lf, lr, wheelbase, max_steer


# --------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)

    # ---- map ---- #
    parser.add_argument("--lanes", type=int, default=50,
                        help="driving lanes each side of the centre line; the pad's half-width is "
                             "lanes * 3.5 m, which is also the largest circle it can hold")
    parser.add_argument("--length", type=float, default=1000.0, help="pad length (m)")

    # ---- simulation ---- #
    parser.add_argument("--times-run", type=float, default=20.0,
                        help="simulation speed relative to real time: 1 = real time, 2 = twice as "
                             "fast, and so on. The simulator has no clock of its own in "
                             "synchronous mode, so this is purely how long the client waits "
                             "between ticks")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--settle-ticks", type=int, default=20,
                        help="ticks to let the car drop onto the surface before logging")
    parser.add_argument("--duration", type=float, default=50,
                        help="per-trial drive duration (s), applies to every trial")
    parser.add_argument("--pid-kp", type=float, default=0.4, help="speed-hold PID proportional gain")
    parser.add_argument("--pid-ki", type=float, default=0.08, help="speed-hold PID integral gain")
    parser.add_argument("--pid-kd", type=float, default=0.02, help="speed-hold PID derivative gain")
    parser.add_argument("--speed-filter-tau", type=float, default=0.3,
                        help="time constant (s) of the low-pass filter on the PID's output. "
                             "These four defaults (down from kp=0.8/ki=0.15/kd=0.05/tau=0.2) were "
                             "retuned after the original gains produced a limit cycle at 6 m/s -- "
                             "throttle slamming 0<->0.85 on a ~2-3s period, v_x oscillating "
                             "5.0-6.7 m/s and never settling even in 50s. The new gains settle "
                             "cleanly (no oscillation) at every speed checked, 2-10 m/s")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the per-second v_x/v_y/psi_dot/a_y + live settled-now "
                             "line each trial prints by default -- useful for watching a "
                             "longitudinal-PID oscillation (a speed that keeps swinging past "
                             "target rather than settling) live, but noisy over a big sweep")

    # ---- sweep ---- #
    parser.add_argument("--speeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                        help="comma-separated target speeds (m/s) -- every one that passes "
                             "--min-radius/--max-ay is driven and logged; whether it's USEFUL for "
                             "the final Cf/Cr is decided later, in estimate_cornering_stiffness.py")
    parser.add_argument("--steers", default="-20,-15,-10,-5,-2.5,2.5,5,10,15,20",
                        help="comma-separated wheel angles (deg) -- every speed is driven at all "
                             "of these (subject to --min-radius/--max-ay)")
    parser.add_argument("--min-radius", type=float, default=8.0,
                        help="skip combos whose kinematic radius L/tan(delta) falls below this "
                             "-- the bicycle model's small-angle assumption runs ~9%% off at "
                             "R=5m, ~4%% at R=8m")
    parser.add_argument("--max-ay", type=float, default=6.0,
                        help="skip combos whose kinematic a_y = v^2*tan(delta)/L exceeds this "
                             "(m/s^2) -- keeps clear of tire saturation and of the body-roll IMU "
                             "contamination that grows with lateral acceleration")

    # ---- steady-state detection -- a trial that never satisfies this is not logged at all ---- #
    parser.add_argument("--steady-hold", type=float, default=2.0,
                        help="seconds v_x, v_y, psi_dot and a_y must all stay under their "
                             "threshold, back to back, before a trial counts as settled (and gets "
                             "logged at all -- see --thresh-*)")
    parser.add_argument("--settle-extra", type=float, default=3.0,
                        help="once the live per-second check has read settled for --steady-hold "
                             "seconds straight, keep driving this many seconds longer (to bank "
                             "some margin beyond the bare minimum) and then stop early instead of "
                             "running the full --duration. A trial that never settles is "
                             "unaffected and still runs the full --duration")
    parser.add_argument("--thresh-vx-dot", type=float, default=0.003, help="m/s^2")
    parser.add_argument("--thresh-vy-dot", type=float, default=0.01, help="m/s^2")
    parser.add_argument("--thresh-psi-ddot", type=float, default=0.01, help="deg/s^2")
    parser.add_argument("--thresh-ay-dot", type=float, default=0.1, help="m/s^2")

    # ---- raw tick-jump glitch guard -- catches what the smoothed --thresh-* gate structurally
    # cannot (see tick_jump_mask()'s docstring): a signal alternating between two values every
    # single tick nets to ~0 after 0.4 s smoothing no matter how large the raw swing, so a window
    # can pass every --thresh-* check while still being corrupted tick-for-tick. Defaults sit
    # comfortably above the tick-to-tick noise a genuinely clean window shows (observed: v_x/v_y
    # jumps ~0.0001 m/s, r jumps ~0.0001-0.0006 rad/s) and comfortably below the one glitch caught
    # so far (r alternating with a ~0.108 rad/s tick-to-tick jump) ---- #
    parser.add_argument("--max-jump-vx", type=float, default=0.3,
                        help="largest allowed tick-to-tick |v_x[i]-v_x[i-1]| (m/s) inside a "
                             "candidate steady window")
    parser.add_argument("--max-jump-vy", type=float, default=0.3, help="same, v_y (m/s)")
    parser.add_argument("--max-jump-r-deg", type=float, default=3.0,
                        help="same, yaw rate r (deg/s, i.e. how much the RATE itself is allowed "
                             "to jump in one tick -- not an acceleration)")

    # ---- output ---- #
    parser.add_argument("--out", default=os.path.join(HERE, "cornering_sweep_log.json"),
                        help="raw per-tick logs for every driven trial, plus mass/lf/lr/wheelbase/"
                             "max_steer/dt -- estimate_cornering_stiffness.py's own --log-file. "
                             "By default this run's trials are MERGED into whatever is already at "
                             "--out (keyed by (target_speed, steer_deg), this run's data winning "
                             "on overlap) -- so a full sweep can be split into several shorter "
                             "invocations, e.g. --speeds 1,2,3 today and --speeds 4,5,6 tomorrow, "
                             "without redoing the speeds already logged. --fresh disables this")
    parser.add_argument("--fresh", action="store_true",
                        help="overwrite --out instead of merging into it")

    # ---- video ---- #
    parser.add_argument("--record", nargs="?", const="auto", default="",
                        help="record every trial to its own mp4; bare flag auto-names each clip "
                             "under --video-dir")
    parser.add_argument("--video-dir", default=os.path.join(HERE, "videos"),
                        help="where auto-named recordings go")
    parser.add_argument("--record-view", default="chase", choices=sorted(VIEWS),
                        help="camera mount for the recordings")
    parser.add_argument("--record-res", default="1280x720", help="recording resolution, WxH")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(180.0)

    print(f"generating a {args.length:.0f} x {2*args.lanes*LANE_WIDTH:.0f} m flat pad "
          f"({args.lanes} lanes each side)...")
    world, centre, max_radius = load_pad(client, args.lanes, args.length, args.dt)
    print(f"  map={world.get_map().name}  centre=({centre[0]:.1f}, {centre[1]:.1f})  "
          f"pad holds a circle up to R={max_radius:.1f} m, or ~{max_radius/2:.1f} m "
          f"starting from the centre")

    thresholds = {
        "v_x_dot": args.thresh_vx_dot,
        "v_y_dot": args.thresh_vy_dot,
        "psi_ddot": math.radians(args.thresh_psi_ddot),
        "a_y_dot": args.thresh_ay_dot,
    }
    max_tick_jump = {
        "v_x": args.max_jump_vx,
        "v_y": args.max_jump_vy,
        "r": math.radians(args.max_jump_r_deg),
    }
    video_factory = make_video_factory(world, args)
    trials, mass, lf, lr, wheelbase, max_steer = collect_sweep_logs(
        world, centre, args, thresholds, max_tick_jump=max_tick_jump, video_factory=video_factory)

    if not trials:
        return   # collect_sweep_logs already printed why

    n_settled = sum(1 for tr in trials if tr["log"] is not None)
    # Only settled trials go into --out: a combo that never reached steady state has nothing worth
    # keeping (estimate_cornering_stiffness.py no longer judges steady state at all, so an
    # unsettled entry would just be silently wrong data to it), and dropping it here rather than
    # saving a null placeholder means re-running a bad combo with better PID tuning can't
    # accidentally erase a GOOD entry already on disk for it from a previous batch.
    new_trials = [{"target_speed": tr["target_speed"], "steer_deg": tr["steer_deg"],
                  "R": tr["R"], "a_y_est": tr["a_y_est"], "log": tr["log"]}
                 for tr in trials if tr["log"] is not None]

    all_trials = new_trials
    if not args.fresh and os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        if (abs(existing["mass"] - mass) > 1.0 or abs(existing["lf"] - lf) > 0.01
                or abs(existing["lr"] - lr) > 0.01 or abs(existing["dt"] - args.dt) > 1e-9):
            print(f"\n  WARNING: {args.out} was logged with a different vehicle/dt "
                  f"(mass={existing['mass']:.1f} vs {mass:.1f} kg, dt={existing['dt']} vs "
                  f"{args.dt}) -- merging anyway, but the two batches may not be comparable. "
                  f"Use --fresh to discard the old file instead.")
        # keyed by (speed, steer) so re-logging a combo overwrites its stale entry rather than
        # duplicating it -- this run's data always wins on overlap. Existing entries with a null
        # log (from a file written before this behavior existed) are dropped here too, so an old
        # file self-heals the first time it's merged into.
        merged = {(t["target_speed"], t["steer_deg"]): t
                 for t in existing["trials"] if t.get("log") is not None}
        for t in new_trials:
            merged[(t["target_speed"], t["steer_deg"])] = t
        all_trials = [merged[k] for k in sorted(merged)]
        print(f"\nmerging into existing {args.out} "
              f"({len(existing['trials'])} trials already logged) -> {len(all_trials)} total")

    result = {
        "mass": mass, "lf": lf, "lr": lr, "wheelbase": wheelbase, "max_steer": max_steer,
        "dt": args.dt, "trials": all_trials,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{n_settled} of {len(trials)} queued trials settled and were logged this run.")
    print(f"Saved: {args.out} ({len(all_trials)} trials total)")


if __name__ == "__main__":
    main()
