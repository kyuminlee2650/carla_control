r"""Cf/Cr cornering-stiffness sweep: flat test pad, drive/fit/pool pipeline, and the evidence
report built on top of it -- one file. (Previously split across this script, which only did the
drive/fit/pool sweep and an older per-a_y plot, and a separate estimate_cornering_stiffness.py
that reused it for a different report; merged back into one file under this name since nothing
outside this file ever imported the old script as a module -- only its JSON output, which this
script still produces.)

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

Pipeline, run end to end by main() on every invocation:

  1. collect_sweep_trials() drives every (speed, steer) combination in --speeds x --steers
     (default 1-15 m/s, all --steers), filtered by --min-radius/--max-ay, and fits each settled
     trial -- see its own docstring for why the final Cf/Cr is a pooled through-origin regression,
     not an average of per-trial numbers.
  2. print_cornering_stiffness_speed_table() + plot_speed_slip_angle() report EVERY driven speed,
     unfiltered -- sample count, Cf/Cr, std(C) across that speed's own steer angles, and the alpha
     range actually swept, front AND rear axle both. This is the evidence the next step acts on,
     not just a courtesy log.
  3. Speed selection: a speed is dropped from the final Cf/Cr if EITHER axle's std(C) across its
     own steer angles exceeds --max-cf-std-pct/--max-cr-std-pct of that axle's own C (the low-a_y
     noise-floor tell -- this is what excludes 1-3 m/s by default), or its largest |alpha| sample
     on that axle exceeds --max-alpha-f-deg/--max-alpha-r-deg (the tell that speed is pushing the
     tire response out of the linear region the whole rig assumes). Checked separately per axle
     since the rear fit generally has less signal (lower Fzr/alpha_r) and can fail where the front
     passes. --no-select pools every driven speed instead.
  4. The final Cf/Cr is pool_trials()/pool_by_speed() run again, this time over only the trials at
     selected speeds. plot_cornering_stiffness_speed_bands() (alpha-vs-Fy scatter colored by
     target speed, with a vertical colorbar, each selected speed's own Cf/Cr line + spread band,
     and the final pooled line on top) is drawn from that same selected-speed subset, and the
     selection itself (kept/dropped speeds and why) is written into --out alongside it.
  5. --record: one mp4 per (speed, steer) trial, via collect_sweep_trials()'s video_factory hook
     (VideoRecorder + follow_with_spectator -- follow_with_spectator itself is already called
     every tick inside drive_and_log()).

Usage (Ubuntu):
    cd ~/carla_control
    python3 lateral_parameter/estimate_cornering_stiffness.py --save-plot
    python3 lateral_parameter/estimate_cornering_stiffness.py --speeds 4,5,6,7,8,9,10 --save-plot
    python3 lateral_parameter/estimate_cornering_stiffness.py --record --save-plot
    python3 lateral_parameter/estimate_cornering_stiffness.py --max-cf-std-pct 10 --max-alpha-f-deg 6 --save-plot

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe lateral_parameter\estimate_cornering_stiffness.py --save-plot
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

from viz_utils import (VIEWS, VideoRecorder, cornering_stiffness_speed_stats,
                       follow_with_spectator, plot_cornering_stiffness_speed_bands,
                       plot_speed_slip_angle, print_cornering_stiffness_speed_table, run_name)

from bicycle import bracket, front_steer_angle, report, slip_angles, steady_axle_forces, \
    zero_phase_derivative


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


def spawn_vehicle(world, centre, blueprint_filter=VEHICLE_BP, ride_height=0.3):

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


# --------------------------------------------------------------------------------------------
# drive + fit
# --------------------------------------------------------------------------------------------

def drive_and_log(world, vehicle, physics, max_steer, args, target_speed, delta, imu_queue,
                  verbose=True):
    """Settle onto the surface, then hold `delta` (bicycle-model wheel angle, rad) while a PID
    chases `target_speed`, logging every tick. Returns the log dict (derivatives not added yet).

    A fresh PID/LowPassFilter every call, not one shared across a whole sweep -- otherwise trial
    N's integral windup and filter state would bias trial N+1's transient, which would then bleed
    into how quickly it reaches (and stays inside) the steady window.
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

    for i in range(int(args.duration / args.dt)):
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
        log["a_y_imu"].append(imu_data.accelerometer.y)
        log["delta"].append(delta); log["delta_measured"].append(front_steer_angle(vehicle))
        log["steer_cmd"].append(control.steer)
        log["throttle"].append(control.throttle); log["brake"].append(control.brake)

        if verbose and i % int(1.0 / args.dt) == 0:
            print(f"t={i*args.dt:6.2f}s  v_x={v_x:6.3f}/{target_speed:.1f}  "
                  f"v_y={v_y:+6.3f}  psi_dot={math.degrees(r):+7.2f} deg/s  "
                  f"R={v_x/r if abs(r) > 1e-4 else float('nan'):7.2f} m  "
                  f"cmd={control.steer:+.4f}  delta={math.degrees(delta):+6.2f} deg "
                  f"(naive {math.degrees(control.steer*max_steer):+6.2f})")

        elapsed = time.time() - step_start
        if elapsed < budget:
            time.sleep(budget - elapsed)
        else:
            behind += 1

    if verbose:
        wall = time.time() - started
        sim = args.duration
        print(f"\n{sim:.1f}s of simulation in {wall:.1f}s of real time "
              f"({sim/wall:.2f}x, asked for {args.times_run:.2f}x)")
        if behind:
            print(f"  {behind} of {int(sim/args.dt)} steps missed the {budget*1000:.0f} ms budget "
                  f"-- the simulator could not keep up, so it ran slower than requested")

    return log


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


def find_steady_windows(log, dt, thresholds, hold_s=2.0):
    """Contiguous [start, end) index ranges where v_x, v_y, psi_dot and a_y have all stopped
    changing for at least `hold_s` seconds.

    A single instant under threshold proves nothing -- noise alone crosses back and forth. Only a
    run of at least `hold_s` seconds where every one of the four stayed under its threshold
    counts, which is what turns this into "settled", not just "quiet right now".
    """
    hold_ticks = max(1, int(round(hold_s / dt)))
    return [(s, e) for s, e in runs(steady_mask(log, thresholds)) if e - s >= hold_ticks]


def longest_window(windows):
    """The longest (start, end) run, or None if `windows` is empty."""
    return max(windows, key=lambda w: w[1] - w[0]) if windows else None


def fit_cornering_stiffness(log, steady, mass, lf, lr):
    """Cf, Cr from the settled window.

    bicycle.steady_axle_forces() turns a_y into the Fyf/Fyr the steady-state force balance
    requires (no Iz needed -- r_dot ~ 0 kills it out of the yaw moment balance). bicycle.
    slip_angles() turns the same window's kinematics into alpha_f/alpha_r, using the *measured*
    wheel angle rather than the commanded one (see front_steer_angle). Regressing one against the
    other through the origin is Cf = Fyf/alpha_f, Cr = Fyr/alpha_r -- bracket() gives both
    one-sided slopes plus their geometric mean, which is what gets reported as Cf/Cr.
    """
    v_x = np.array(log["v_x"])[steady]
    v_y = np.array(log["v_y"])[steady]
    r = np.array(log["r"])[steady]
    delta = np.array(log["delta_measured"])[steady]

    a_y = v_x * r
    Fyf, Fyr = steady_axle_forces(a_y, mass, lf, lr)
    alpha_f, alpha_r = slip_angles(delta, v_x, v_y, r, lf, lr)

    Cf_fwd, Cf_rev, Cf = bracket(Fyf, alpha_f)
    Cr_fwd, Cr_rev, Cr = bracket(Fyr, alpha_r)
    return {
        "Cf": Cf, "Cf_forward": Cf_fwd, "Cf_reverse": Cf_rev,
        "Cr": Cr, "Cr_forward": Cr_fwd, "Cr_reverse": Cr_rev,
        "alpha_f": alpha_f, "Fyf": Fyf, "alpha_r": alpha_r, "Fyr": Fyr,
    }


def pool_trials(trials, mass, lf, lr):
    """Fit each settled trial individually, then once more pooled across all of them -- the
    number that actually gets used. Mutates each trial dict in place with its own "fit".

    Fitting every trial on its own first (rather than only the pooled fit) is what makes a
    per-trial Cf/Cr table possible: systematic drift of the individual numbers with a_y or speed
    is the standard tell that some trials have left the tire's linear region (this whole sweep is,
    in effect, an ISO 4138 steady-state circular test). Returns None if no trial ever settled.
    """
    alpha_f_all, Fyf_all, alpha_r_all, Fyr_all = [], [], [], []
    for tr in trials:
        if tr["window"] is None:
            continue
        fit = fit_cornering_stiffness(tr["log"], slice(*tr["window"]), mass, lf, lr)
        tr["fit"] = fit
        alpha_f_all.append(fit["alpha_f"]); Fyf_all.append(fit["Fyf"])
        alpha_r_all.append(fit["alpha_r"]); Fyr_all.append(fit["Fyr"])

    if not alpha_f_all:
        return None

    alpha_f_all = np.concatenate(alpha_f_all); Fyf_all = np.concatenate(Fyf_all)
    alpha_r_all = np.concatenate(alpha_r_all); Fyr_all = np.concatenate(Fyr_all)
    Cf_fwd, Cf_rev, Cf = bracket(Fyf_all, alpha_f_all)
    Cr_fwd, Cr_rev, Cr = bracket(Fyr_all, alpha_r_all)
    return {
        "Cf": Cf, "Cf_forward": Cf_fwd, "Cf_reverse": Cf_rev,
        "Cr": Cr, "Cr_forward": Cr_fwd, "Cr_reverse": Cr_rev,
        "alpha_f": alpha_f_all, "Fyf": Fyf_all, "alpha_r": alpha_r_all, "Fyr": Fyr_all,
        "n_trials": sum(1 for tr in trials if tr["window"] is not None),
    }


def pool_by_speed(trials, mass, lf, lr):
    """Regroup pool_trials()'s per-trial fits by target_speed and pool each speed's (alpha, Fy)
    pairs on its own -- one Cf/Cr per speed (steer angle collapsed out) instead of one number for
    the whole sweep. Must run after pool_trials() has populated tr["fit"] for every settled trial.

    This is the speed-only cut of the same constancy check pool_trials()'s per-trial fits already
    give across the full (speed, steer) grid: if Cf/Cr still drifts once every steer angle at a
    given speed is pooled together, the drift tracks speed itself (actuator lag, steering-curve
    residual, ...) rather than just a_y.

    Also reports, per speed, how much the *individual* per-trial Cf/Cr (one per steer angle)
    disagree with each other -- min/max and the spread as a percentage of their mean. A wide
    spread at a given speed is the same low-a_y noise-floor symptom as drift across speeds, just
    seen within one speed instead of across the sweep: fixing the speed does not fix a tiny alpha.

    Returns {speed: {"Cf":..., "Cr":..., "n_trials":..., "Cf_min"/"Cf_max"/"Cf_spread_pct":...,
    "Cr_min"/"Cr_max"/"Cr_spread_pct":...}}, skipping speeds with no settled trial.
    """
    by_speed = {}
    for tr in trials:
        if tr.get("fit") is None:
            continue
        by_speed.setdefault(tr["target_speed"], []).append(tr["fit"])

    result = {}
    for v, fits in by_speed.items():
        alpha_f = np.concatenate([f["alpha_f"] for f in fits])
        Fyf = np.concatenate([f["Fyf"] for f in fits])
        alpha_r = np.concatenate([f["alpha_r"] for f in fits])
        Fyr = np.concatenate([f["Fyr"] for f in fits])
        Cf_fwd, Cf_rev, Cf = bracket(Fyf, alpha_f)
        Cr_fwd, Cr_rev, Cr = bracket(Fyr, alpha_r)

        Cf_vals = [f["Cf"] for f in fits]
        Cr_vals = [f["Cr"] for f in fits]
        Cf_lo, Cf_hi = min(Cf_vals), max(Cf_vals)
        Cr_lo, Cr_hi = min(Cr_vals), max(Cr_vals)
        result[v] = {
            "Cf": Cf, "Cf_forward": Cf_fwd, "Cf_reverse": Cf_rev,
            "Cr": Cr, "Cr_forward": Cr_fwd, "Cr_reverse": Cr_rev,
            "n_trials": len(fits),
            "Cf_min": Cf_lo, "Cf_max": Cf_hi,
            "Cf_spread_pct": 100.0 * (Cf_hi - Cf_lo) / Cf if len(fits) > 1 else 0.0,
            "Cr_min": Cr_lo, "Cr_max": Cr_hi,
            "Cr_spread_pct": 100.0 * (Cr_hi - Cr_lo) / Cr if len(fits) > 1 else 0.0,
        }
    return result


def collect_sweep_trials(world, centre, args, thresholds, video_factory=None):
    """Drive every (speed, steer) combination in --speeds x --steers, fit each settled trial, and
    pool -- the actual data-collection/estimation pipeline this whole script's report is built on.

    Why pool instead of just averaging Cf across trials: alpha_f/alpha_r and Fyf/Fyr both scale
    with the operating point, so a plain average of per-trial Cf weights a barely-turning 1 m/s
    trial (tiny alpha, tiny Fy, mostly noise) the same as a well-loaded 8 m/s trial. Pooling the
    raw (alpha, Fy) pairs before the single through-origin regression lets bracket() do that
    weighting properly -- points with more signal (bigger alpha^2) dominate the sum the way a
    least-squares fit is supposed to.

    Combinations are filtered before anything drives: R < --min-radius breaks the bicycle model's
    small-angle assumption (~9% kinematic error at R=5m, ~4% at R=8m), and a_y > --max-ay is
    skipped because higher lateral acceleration measurably increases body roll here -- a live
    IMU-vs-kinematic gap check saw ~0.16 m/s^2 of roll-induced bias at just 2.6 m/s^2 of a_y,
    which is exactly the contamination the whole rig is built to avoid (see attach_imu's
    docstring).

    video_factory(vehicle, trial) -> VideoRecorder|None, called right after a trial's vehicle is
    spawned, before that trial drives. If it returns a recorder, this closes it once the trial's
    drive finishes, before vehicle.destroy() (VideoRecorder's own lifecycle requirement -- the
    camera is attached to the vehicle). None (the default) records nothing.

    Returns (queued, pooled, by_speed, mass, lf, lr). pooled/by_speed are None if no trial ever
    settled (queued may still be non-empty in that case); queued is [] only if nothing was queued
    at all (every combination filtered out), in which case pooled/by_speed are also None.
    """
    speeds = [float(v) for v in args.speeds.split(",")]
    steers = [float(d) for d in args.steers.split(",")]

    probe, geometry = spawn_vehicle(world, centre, VEHICLE_BP)
    wheelbase, lf, lr, max_steer, mass, com = geometry
    probe.destroy()

    queued = []
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
            queued.append({"target_speed": v, "steer_deg": d, "R": R, "a_y_est": a_y_est})

    print(f"\n{len(queued)} of {len(speeds)*len(steers)} (speed, steer) combinations queued "
          f"(mass={mass:.0f} kg  lf={lf:.2f} m  lr={lr:.2f} m)")
    if not queued:
        print("nothing to run -- widen --speeds/--steers or relax --min-radius/--max-ay")
        return queued, None, None, mass, lf, lr

    for n, trial in enumerate(queued, 1):
        v, d = trial["target_speed"], trial["steer_deg"]
        print(f"\n[{n}/{len(queued)}] v={v:.1f} m/s  delta={d:.1f}deg  "
              f"R~{trial['R']:.1f}m  a_y~{trial['a_y_est']:.2f} m/s^2")
        trial["window"] = None
        vehicle = imu = log = recorder = None
        try:
            vehicle, _ = spawn_vehicle(world, centre, VEHICLE_BP)
            physics = vehicle.get_physics_control()
            imu, imu_queue = attach_imu(world, vehicle, physics.center_of_mass)
            if video_factory is not None:
                recorder = video_factory(vehicle, trial)
            log = drive_and_log(world, vehicle, physics, max_steer, args, v, math.radians(d),
                                imu_queue, verbose=False)
        except Exception as exc:
            # One trial's transient hiccup (a slow tick missing the IMU's 5s window, a spawn
            # collision, ...) should not lose every trial that already settled before it, nor
            # the ones still queued after it -- print, clean up what exists, and move on.
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
            continue

        add_derivatives(log, args.dt)
        window = longest_window(find_steady_windows(log, args.dt, thresholds,
                                                     hold_s=args.steady_hold))
        trial["log"] = log
        trial["window"] = window
        if window is None:
            print("  never settled")
        else:
            t = log["t"]
            print(f"  steady t={t[window[0]]:.2f}-{t[window[1]-1]:.2f}s "
                  f"({t[window[1]-1]-t[window[0]]:.2f}s)")

    pooled = pool_trials(queued, mass, lf, lr)
    by_speed = pool_by_speed(queued, mass, lf, lr) if pooled is not None else None
    return queued, pooled, by_speed, mass, lf, lr


# --------------------------------------------------------------------------------------------
# video + report
# --------------------------------------------------------------------------------------------

def make_video_factory(world, args):
    """--record -> a collect_sweep_trials() video_factory(vehicle, trial) that opens one
    VideoRecorder per (speed, steer) trial, named after that trial so a whole sweep's worth of
    clips don't overwrite each other -- same auto-naming split mpc_mpc1.py uses for its own
    --controller a b --record case, just keyed on (speed, steer) instead of controller name."""
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
                        help="ticks to let the car drop onto the surface before reporting")
    parser.add_argument("--duration", type=float, default=50,
                        help="per-trial drive duration (s), applies to every trial")
    parser.add_argument("--pid-kp", type=float, default=0.7, help="speed-hold PID proportional gain")
    parser.add_argument("--pid-ki", type=float, default=0.15, help="speed-hold PID integral gain")
    parser.add_argument("--pid-kd", type=float, default=0.05, help="speed-hold PID derivative gain")
    parser.add_argument("--speed-filter-tau", type=float, default=0.2,
                        help="time constant (s) of the low-pass filter on the PID's output")

    # ---- steady-state detection ---- #
    parser.add_argument("--steady-hold", type=float, default=2.0,
                        help="seconds v_x, v_y, psi_dot and a_y must all stay under their "
                             "threshold, back to back, before the run counts as settled")
    parser.add_argument("--thresh-vx-dot", type=float, default=0.01, help="m/s^2")
    parser.add_argument("--thresh-vy-dot", type=float, default=0.01, help="m/s^2")
    parser.add_argument("--thresh-psi-ddot", type=float, default=0.01, help="deg/s^2")
    parser.add_argument("--thresh-ay-dot", type=float, default=0.1, help="m/s^2")

    # ---- sweep ---- #
    parser.add_argument("--speeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                        help="comma-separated target speeds (m/s) -- every speed is driven and "
                             "reported (table + slip-angle plot), then --max-cf-std-pct/"
                             "--max-alpha-f-deg below decide which of them actually go into the "
                             "final pooled Cf/Cr")
    parser.add_argument("--steers", default="6,9,12,16,20",
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

    # ---- speed selection (applied AFTER the full sweep drives, front AND rear both) ---- #
    parser.add_argument("--max-cf-std-pct", type=float, default=10.0,
                        help="drop a speed from the final pooled Cf/Cr if std(Cf) across that "
                             "speed's own steer angles exceeds this %% of its own Cf -- the "
                             "low-a_y noise-floor tell (this is what excluded 1-3 m/s by default "
                             "before --speeds was widened to include them)")
    parser.add_argument("--max-cr-std-pct", type=float, default=10.0,
                        help="same as --max-cf-std-pct, rear axle (std(Cr) as %% of that speed's "
                             "own Cr) -- checked separately since the rear fit generally has less "
                             "signal (lower Fzr/alpha_r) and can fail this where the front passes")
    parser.add_argument("--max-alpha-f-deg", type=float, default=7.0,
                        help="drop a speed from the final pooled Cf/Cr if its largest |alpha_f| "
                             "sample exceeds this many degrees -- the tell that a speed is "
                             "pushing this speed/steer combo's tire response out of the linear "
                             "region the whole rig assumes")
    parser.add_argument("--max-alpha-r-deg", type=float, default=7.0,
                        help="same as --max-alpha-f-deg, rear axle")
    parser.add_argument("--no-select", action="store_true",
                        help="skip speed selection -- pool every driven speed into the final "
                             "Cf/Cr instead of only the ones passing the --max-c*-std-pct/"
                             "--max-alpha-*-deg thresholds on both axles")

    # ---- output ---- #
    parser.add_argument("--out", default=os.path.join(HERE, "cornering_stiffness_speed_report.json"),
                        help="where to write the per-speed/pooled Cf/Cr summary (scalars only, no "
                             "raw per-tick arrays)")
    parser.add_argument("--no-plot", action="store_true", help="skip every figure, print only")
    parser.add_argument("--save-plot", action="store_true", help="also save the figures as PNGs")
    parser.add_argument("--no-show", action="store_true",
                        help="build (and, with --save-plot, save) the figures but never call "
                             "plt.show() -- for headless/background runs")
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"))

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

    video_factory = make_video_factory(world, args)
    queued, pooled, by_speed, mass, lf, lr = collect_sweep_trials(
        world, centre, args, thresholds, video_factory=video_factory)

    if not queued:
        return   # collect_sweep_trials already printed why
    if pooled is None:
        print("\nno trial ever settled -- nothing to report")
        return

    # ---- report basis: every driven speed, unfiltered -- the evidence the selection below acts on
    print_cornering_stiffness_speed_table(queued, by_speed)
    if not args.no_plot:
        plot_speed_slip_angle(
            queued, out_dir=args.plot_dir if args.save_plot else None, show=not args.no_show)

    # ---- select which speeds go into the final pooled Cf/Cr (front AND rear both) ---- #
    def _drop_reasons(s):
        """Every reason this speed's stats fail the thresholds, front axle then rear -- empty
        means it's kept. One function shared by the selected_speeds filter and the "why dropped"
        print below, so the two can never disagree the way they briefly did when the reason
        printer had its own copy of this logic (a NaN Cf_std_pct silently passed `nan >
        threshold`, dropping the speed but printing no reason for it).
        """
        reasons = []
        if not math.isfinite(s["Cf_std_pct"]):
            reasons.append("Cf fit is NaN (alpha_f-Fyf correlation not consistently signed at "
                           "this speed -- unusable, not just noisy)")
        elif s["Cf_std_pct"] > args.max_cf_std_pct:
            reasons.append(f"Cf std {s['Cf_std_pct']:.1f}% > {args.max_cf_std_pct:.1f}%")
        if s["alpha_f_max_deg"] > args.max_alpha_f_deg:
            reasons.append(f"alpha_f {s['alpha_f_max_deg']:.2f} deg > {args.max_alpha_f_deg:.1f} deg")
        if not math.isfinite(s["Cr_std_pct"]):
            reasons.append("Cr fit is NaN (alpha_r-Fyr correlation not consistently signed at "
                           "this speed -- unusable, not just noisy)")
        elif s["Cr_std_pct"] > args.max_cr_std_pct:
            reasons.append(f"Cr std {s['Cr_std_pct']:.1f}% > {args.max_cr_std_pct:.1f}%")
        if s["alpha_r_max_deg"] > args.max_alpha_r_deg:
            reasons.append(f"alpha_r {s['alpha_r_max_deg']:.2f} deg > {args.max_alpha_r_deg:.1f} deg")
        return reasons

    stats = cornering_stiffness_speed_stats(queued, by_speed)
    if args.no_select:
        selected_speeds = sorted(stats)
    else:
        selected_speeds = sorted(v for v, s in stats.items() if not _drop_reasons(s))
    dropped_speeds = sorted(set(stats) - set(selected_speeds))

    print(f"\nSpeed selection (--max-cf-std-pct {args.max_cf_std_pct:.1f}, "
          f"--max-cr-std-pct {args.max_cr_std_pct:.1f}, "
          f"--max-alpha-f-deg {args.max_alpha_f_deg:.1f}, "
          f"--max-alpha-r-deg {args.max_alpha_r_deg:.1f}"
          f"{', DISABLED (--no-select)' if args.no_select else ''}):")
    print(f"  kept:    {', '.join(f'{v:g}' for v in selected_speeds) or '(none)'} m/s")
    for v in dropped_speeds:
        print(f"  dropped {v:g} m/s: {'; '.join(_drop_reasons(stats[v]))}")

    if not selected_speeds:
        print("\nno speed survived selection -- widen --max-cf-std-pct/--max-alpha-f-deg")
        return

    # ---- final Cf/Cr, pooled over only the selected speeds ---- #
    selected_trials = [tr for tr in queued
                       if tr.get("fit") is not None and tr["target_speed"] in selected_speeds]
    final_pooled = pool_trials(selected_trials, mass, lf, lr)
    final_by_speed = pool_by_speed(selected_trials, mass, lf, lr)

    print(f"\nFinal Cf/Cr, pooled over {final_pooled['n_trials']} trials at the "
          f"{len(selected_speeds)} selected speed(s):")
    print(f"Cf = {final_pooled['Cf']:,.0f} N/rad  (bracket [{final_pooled['Cf_forward']:,.0f}, "
          f"{final_pooled['Cf_reverse']:,.0f}])")
    print(f"Cr = {final_pooled['Cr']:,.0f} N/rad  (bracket [{final_pooled['Cr_forward']:,.0f}, "
          f"{final_pooled['Cr_reverse']:,.0f}])")
    report(final_pooled["Cf"], final_pooled["Cr"], mass, lf, lr)

    result = {
        "Cf": final_pooled["Cf"], "Cr": final_pooled["Cr"], "mass": mass, "lf": lf, "lr": lr,
        "n_trials": final_pooled["n_trials"],
        "selected_speeds": selected_speeds,
        "dropped_speeds": dropped_speeds,
        "selection": {"max_cf_std_pct": args.max_cf_std_pct, "max_cr_std_pct": args.max_cr_std_pct,
                     "max_alpha_f_deg": args.max_alpha_f_deg,
                     "max_alpha_r_deg": args.max_alpha_r_deg, "disabled": args.no_select},
        "by_speed": {
            str(v): {"Cf": r["Cf"], "Cr": r["Cr"], "n_trials": r["n_trials"],
                    "Cf_min": r["Cf_min"], "Cf_max": r["Cf_max"],
                    "Cf_spread_pct": r["Cf_spread_pct"],
                    "Cr_min": r["Cr_min"], "Cr_max": r["Cr_max"],
                    "Cr_spread_pct": r["Cr_spread_pct"]}
            for v, r in sorted(final_by_speed.items())
        },
        # every driven speed's own diagnostics, selected or not -- the record of why
        "all_speed_stats": {str(v): s for v, s in sorted(stats.items())},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")

    if not args.no_plot:
        plot_cornering_stiffness_speed_bands(
            selected_trials, final_pooled, final_by_speed,
            out_dir=args.plot_dir if args.save_plot else None, show=not args.no_show)


if __name__ == "__main__":
    main()
