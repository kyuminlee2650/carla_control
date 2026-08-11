r"""Longitudinal MPC (speed-tracking QP) + pedal layer, steer forced to 0.

Same isolation as longitudinal_PID.py -- no Stanley, no desired path, fixed spawn on Town06's
longest straight (index 86) -- so the runs test the exact same speed-profile-tracking task.
--controller takes one or more of three stacks and, given more than one, overlays them on one
comparison figure instead of eyeballing separate runs (viz_utils.plot_longitudinal accepts
{label: hist}):

    mpc+ff+pid   (default) SpeedMPC -> a_cmd -> LUT feedforward + PID pedal layer
    mpc+pid      SpeedMPC -> a_cmd -> PID-only pedal layer, no LUT term at all -- the same
                 use_feedforward=False baseline validate_lut.py's "PID only" trial uses, so this
                 isolates what the LUT is actually buying the MPC stack over closing the
                 acceleration loop with feedback alone
    pid          plain speed PID straight to the pedal, no MPC involved -- functions.PID with
                 longitudinal_PID.py's own gains, so it's the same controller, not a
                 reimplementation of it

    --controller mpc+ff+pid mpc+pid          # does the LUT feedforward help?
    --controller mpc+ff+pid pid              # does the MPC help at all, top to bottom?
    --controller mpc+ff+pid mpc+pid pid      # all three at once

Control stack (two layers, cascaded) for mpc+ff+pid / mpc+pid:

    1. SpeedMPC (this file): a linear MPC over the plant

           x_k = v_{x,k},   x_{k+1} = A x_k + B a_{x,k},   A=1, B=T

       i.e. a first-order integrator from commanded acceleration to speed. The decision variable
       is the acceleration sequence itself (not jerk) -- free for the first Nc steps of the
       horizon, held after that -- so solving the QP each cycle and applying only its first
       element (receding horizon) hands the layer below a target acceleration a_cmd directly, with
       no extra integration/anchoring step needed. See the SpeedMPC docstring for the QP.

    2. LookupController (longitudinal_lookup/lookup_controller.py): the same LUT-feedforward +
       PID acceleration tracker validate_lut.py validates on its own -- turns a_cmd into a pedal
       command u in [-1, 1]. mpc+pid runs the identical class with use_feedforward=False rather
       than reimplementing a bare PID. This script does not reimplement that layer either way; it
       is imported as-is so every run here is testing the exact stack validate_lut.py validated.

a_meas for the pedal layer comes raw off the IMU (functions.ImuAcceleration.a_x_raw), not low-pass
filtered -- same reasoning as validate_lut.py: the LUT was calibrated against the raw signal, and
filtering only on this side would compare the controller's a_meas against a lagged version of what
it was fit on. A separately filtered a_x (tau=0.15, matching longitudinal_PID.py) is kept purely
for the jerk derivative and the result plot, exactly as longitudinal_PID.py does for its own a_x.

Every controller shares one run_trial() (spawn -> warm-up -> log -> teardown) instead of each
having its own copy of that harness; only the per-step control law differs (PidController /
MpcController, both a single step(ctx) -> pedal method).

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python longitudinal_mpc.py --profile constant --initial-speed 15
    .venv/bin/python longitudinal_mpc.py --profile sine --initial-speed 15 --sine-amplitude 3 --sine-period 10 --save-plot
    .venv/bin/python longitudinal_mpc.py --profile step --initial-speed 15 --step-size 5 --step-time 10 --save-plot
    .venv/bin/python longitudinal_mpc.py --controller mpc+ff+pid mpc+pid pid --profile sine --save-plot

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe longitudinal_mpc.py --profile constant --initial-speed 15
    .venv\Scripts\python.exe longitudinal_mpc.py --profile sine --initial-speed 15 --sine-amplitude 3 --sine-period 10
    .venv\Scripts\python.exe longitudinal_mpc.py --profile constant --initial-speed 15 --times-run 10 --save-plot --record
    .venv\Scripts\python.exe longitudinal_mpc.py --controller mpc+ff+pid mpc+pid --profile sine --initial-speed 15 --sine-amplitude 3 --sine-period 3 --save-plot
    .venv\Scripts\python.exe longitudinal_mpc.py --controller mpc+ff+pid mpc+pid pid --profile step --initial-speed 15 --step-size -5 --step-time 5 --save-plot
"""

import argparse
import math
import os
import queue
import sys
import time
from types import SimpleNamespace

import numpy as np
import osqp
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))

# see functions.py for why this path is needed alongside the pip-installed carla package
CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/2026intern/carla"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.append(os.path.join(HERE, "longitudinal_lookup"))

import carla

from functions import PID, ImuAcceleration, LowPassFilter, clipping
from viz_utils import (VIEWS, VideoRecorder, follow_with_spectator, plot_longitudinal_result,
                       print_error_summary, run_name)

from longitudinal_lut import LongitudinalLUT
from lookup_controller import LookupController


MAP_NAME = "Town06"
ORIGIN_INDEX = 86


WARM_START_SPEED_TOL = 0.3   # m/s
WARM_START_ACCEL_TOL = 0.5   # m/s^2
WARM_START_TIMEOUT = 15.0    # s -- safety cap in case initial_speed is unreachable


class SpeedMPC:
    """Speed-tracking MPC over a scalar first-order integrator: x_k = v_{x,k}, x_{k+1} = A x_k +
    B a_{x,k}, A=1, B=T. Decision variable is the acceleration sequence itself -- free for the
    first Nc steps of the horizon Np (Nc <= Np) and held at its last value after that.

    Substituting the forward solution gives the condensed prediction X = A_bar x0 + B_bar U, and
    the tracking cost

        J = (X - Xref)' W1 (X - Xref) + U' W2 U + (Phi U)' W3 (Phi U)

    becomes a box-constrained QP in U alone:

        min 1/2 U' H U + f' U   s.t.   a_min <= U <= a_max
        H = 2 (B_bar' W1 B_bar + W2 + Phi' W3 Phi)   (constant -- built once)
        f = 2 B_bar' W1 (A_bar x0 - Xref)            (rebuilt every cycle)

    W1 = w_v I_Np weights every predicted speed error the same way; W2 = w_a I_Nc penalizes the
    commanded acceleration's own magnitude (comfort/effort); Phi is the first-difference operator,
    (Phi U)_l = a_{x,l+1} - a_{x,l}, and W3 = w_j I_(Nc-1) penalizes that difference -- a jerk-rate
    cost on the commanded acceleration even though jerk itself is not a decision variable here.
    W2 alone already makes H positive definite (B_bar' W1 B_bar is only positive semidefinite), so
    the QP has a unique solution even with w_j = 0.

    Only the first element of U* is applied each cycle (receding horizon). Because U* already
    lives in acceleration units, that element *is* a_cmd -- no separate integration/anchoring step
    is needed the way a jerk-input MPC would need one.
    """

    def __init__(self, dt, n_p, n_c, w_v, w_a, w_j, a_min=-4.05, a_max=2.4):
        if not (0 < n_c <= n_p):
            raise ValueError(f"need 0 < n_c <= n_p, got n_c={n_c}, n_p={n_p}")
        if not (a_min < a_max):
            raise ValueError(f"need a_min < a_max, got a_min={a_min}, a_max={a_max}")

        self.T = dt
        self.n_p = n_p
        self.n_c = n_c
        self.a_min = a_min
        self.a_max = a_max

        self.A = 1.0   # fixed by the problem: x_{k+1} = A x_k + B a_{x,k}
        self.B = dt

        self.A_bar = np.full((n_p, 1), self.A)   # A^i = 1 for every i since A=1
        self.B_bar = self._build_b_bar()
        self.Phi = self._build_phi()

        W1 = w_v * np.eye(n_p)
        W2 = w_a * np.eye(n_c)
        W3 = w_j * np.eye(max(n_c - 1, 0))

        self.H = 2.0 * (self.B_bar.T @ W1 @ self.B_bar + W2 + self.Phi.T @ W3 @ self.Phi)
        self.H = 0.5 * (self.H + self.H.T)      # symmetrize against round-off
        self._B_W1 = 2.0 * self.B_bar.T @ W1    # reused every cycle to form f

        self._solver = osqp.OSQP()
        self._solver.setup(
            P=sparse.csc_matrix(self.H),
            q=np.zeros(n_c),
            A=sparse.csc_matrix(np.eye(n_c)),   # native box constraint on U itself
            l=a_min * np.ones(n_c),
            u=a_max * np.ones(n_c),
            verbose=False,
            polish=False,   # polishing logs to stdout even when verbose is off
        )
        self.last_solution = np.zeros(n_c)
        self.last_status = "unsolved"

    def _build_b_bar(self):
        """Np x Nc. [B_bar]_{i,c} = A^(i-c) B for c < Nc, c <= i; the last column accumulates
        every step the held input still acts over. A=1 so every power of A collapses to 1."""
        B_bar = np.zeros((self.n_p, self.n_c))
        for i in range(1, self.n_p + 1):
            for c in range(1, self.n_c + 1):
                if c < self.n_c:
                    if c <= i:
                        B_bar[i - 1, c - 1] = self.B
                elif i >= self.n_c:
                    B_bar[i - 1, c - 1] = (i - self.n_c + 1) * self.B
        return B_bar

    def _build_phi(self):
        """(Nc-1) x Nc first-difference operator: (Phi U)_l = a_{x,l+1} - a_{x,l}."""
        rows = max(self.n_c - 1, 0)
        Phi = np.zeros((rows, self.n_c))
        for l in range(rows):
            Phi[l, l] = -1.0
            Phi[l, l + 1] = 1.0
        return Phi

    def solve(self, v_x, v_ref_preview):
        """One receding-horizon step. v_ref_preview: length-Np sequence of desired speed at each
        future prediction step (a known/analytic profile is previewed in full; a planner handing
        over a shorter horizon would just repeat its last value to fill the rest). Returns a_cmd,
        the acceleration to hand the pedal layer, clipped to [a_min, a_max]."""
        x_ref = np.asarray(v_ref_preview, dtype=float).reshape(-1, 1)
        f = self._B_W1 @ (self.A_bar * v_x - x_ref)

        self._solver.update(q=f.ravel())
        result = self._solver.solve()
        self.last_status = result.info.status

        if result.x is None or not np.all(np.isfinite(result.x)):
            # keep driving on the previous plan rather than dropping to zero acceleration
            u = self.last_solution
        else:
            u = result.x
            self.last_solution = u.copy()

        return float(np.clip(u[0], self.a_min, self.a_max))


def speed_reference(args, t):
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


# ----------------------------------------------------------------------------- controllers
# Common interface: step(ctx) -> pedal command u in [-1, 1] (positive throttle, negative brake).
# ctx (a SimpleNamespace, set fresh by run_trial() each cycle) carries t, v_x, v_ref, a_x
# (filtered), a_x_raw (IMU, for the LUT layer), gear, warmed_up. reset(u) is called once, right
# after the warm-up hand-off, so a controller can drop whatever state it accumulated chasing the
# warm-up setpoint before scoring starts.

class PidController:
    """Speed PID straight to the pedal -- longitudinal_PID.py's own controller, same defaults."""
    label = "PID"

    def __init__(self, args):
        self.pid = PID(kp=args.pid_kp, ki=args.pid_ki, kd=args.pid_kd, dt=args.dt)
        self.filter = LowPassFilter(tau=args.pid_tau, dt=args.dt, initial=0.0)

    def reset(self, u):
        pass   # longitudinal_PID.py never resets its PID at hand-off either; match it exactly

    def step(self, ctx):
        return clipping(self.filter.step(self.pid.step(ctx.v_ref - ctx.v_x)), 1, -1)


class MpcController:
    """SpeedMPC -> a_cmd -> pedal layer, tracking a_cmd either with the LUT feedforward + PID
    stack (see module docstring) or with the PID-only baseline validate_lut.py's "PID only" trial
    uses (use_ff=False -- LookupController.step() still runs, but its feedforward() call is
    skipped, per its own use_feedforward flag: the same class either way, not a reimplementation).
    """

    def __init__(self, args, use_ff):
        self.args = args
        self.label = "MPC+FF+PID" if use_ff else "MPC+PID"
        self.mpc = SpeedMPC(dt=args.dt, n_p=args.n_p, n_c=args.n_c,
                            w_v=args.w_v, w_a=args.w_a, w_j=args.w_j,
                            a_min=args.a_min, a_max=args.a_max)
        self.pedal_ctrl = LookupController(LongitudinalLUT(args.lut), kp=args.kp, ki=args.ki,
                                           kd=args.kd, dt=args.dt, use_feedforward=use_ff)
        self.u_filter = LowPassFilter(tau=args.u_tau, dt=args.dt, initial=0.0)

    def reset(self, u):
        self.pedal_ctrl.reset()   # drop the warm-up phase's PID integral before scoring starts
        self.u_filter = LowPassFilter(tau=self.args.u_tau, dt=self.args.dt, initial=u)

    def step(self, ctx):
        preview = reference_preview(self.args, ctx.t, self.args.n_p, self.args.dt, ctx.warmed_up)
        ctx.a_cmd = self.mpc.solve(ctx.v_x, preview)   # stashed for the console print line
        u_raw = self.pedal_ctrl.step(ctx.gear, ctx.v_x, ctx.a_cmd, a_meas=ctx.a_x_raw)
        return self.u_filter.step(u_raw)


CONTROLLERS = {
    "mpc+ff+pid": lambda args: MpcController(args, use_ff=True),
    "mpc+pid": lambda args: MpcController(args, use_ff=False),
    "pid": PidController,
}


# ----------------------------------------------------------------------------- one trial

def run_trial(world, origin_transform, blueprint, imu_bp, controller, args, recorder_factory):
    """Spawn one vehicle, drive it under `controller` for the scored profile, tear it down.

    Same warm-up gate as longitudinal_PID.py: launch from rest under the real controller and hold
    off on logging until v_x/a_x have actually settled near the profile's own t=0 value
    (initial_speed, since sin(0) = 0 for sine and the step hasn't happened yet at t=0 for step)
    instead of faking that starting condition.
    """
    vehicle = world.spawn_actor(blueprint, origin_transform)

    # a_x/a_y off the IMU -- see stanley_PID.py for why (true body-frame values straight from the
    # sensor, nothing to derive by hand). a_y only exists here to feed jerk_total; nothing plots it
    # on its own since there's no lateral figure in a steer=0 run.
    accel = ImuAcceleration(dt=args.dt)

    hist = {"t": [], "v_x": [], "v_des": [], "a_x": [], "jerk": [], "jerk_total": [],
            "throttle": [], "brake": []}

    warmed_up = False
    log_start_i = 0

    imu = None
    recorder = None
    try:
        world.tick()

        # spawned after the priming tick above, so the first world.tick() in the loop below
        # produces this sensor's first queued sample -- one put() per get() keeps them in
        # lockstep for the rest of the trial.
        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)
        recorder = recorder_factory(vehicle)

        steps = int((args.duration + WARM_START_TIMEOUT) / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)

            transform = vehicle.get_transform()
            yaw = math.radians(transform.rotation.yaw)
            vel_vec = vehicle.get_velocity()
            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)  # body-frame forward speed

            accel.step(imu_data)
            a_x, a_y, a_x_raw = accel.a_x, accel.a_y, accel.a_x_raw

            jerk, jerk_total = accel.jerk, accel.jerk_total

            t = (i - log_start_i) * args.dt
            v_ref = args.initial_speed if not warmed_up else speed_reference(args, t)
            ctx = SimpleNamespace(t=t, v_x=v_x, v_ref=v_ref, a_x=a_x, a_x_raw=a_x_raw,
                                  gear=vehicle.get_control().gear, warmed_up=warmed_up)
            u = controller.step(ctx)

            control = carla.VehicleControl()
            control.throttle, control.brake = (u, 0.0) if u >= 0 else (0.0, -u)
            control.steer = 0.0
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            if not warmed_up:
                converged = abs(v_x - args.initial_speed) < WARM_START_SPEED_TOL and abs(a_x) < WARM_START_ACCEL_TOL
                timed_out = i * args.dt >= WARM_START_TIMEOUT
                if converged or timed_out:
                    warmed_up = True
                    log_start_i = i
                    controller.reset(u)
                    status = "converged" if converged else f"timed out after {WARM_START_TIMEOUT:.0f}s"
                    print(f"[{controller.label}] Warm-start {status}: v_x={v_x:.2f} m/s, "
                          f"a_x={a_x:.2f} m/s^2 -- logging starts now.")
                else:
                    elapsed = time.time() - step_start
                    if elapsed < args.dt / args.times_run:
                        time.sleep(args.dt / args.times_run - elapsed)
                    continue

            # recompute now that log_start_i may have just been updated above -- the t used for
            # the controller's step() earlier in this same iteration was based on the pre-handoff
            # value and would log a stale (much larger) timestamp for this first sample otherwise
            t = (i - log_start_i) * args.dt
            hist["t"].append(t)
            hist["v_x"].append(v_x)
            hist["v_des"].append(v_ref)
            hist["a_x"].append(a_x)
            hist["jerk"].append(jerk)
            hist["jerk_total"].append(jerk_total)
            hist["throttle"].append(control.throttle)
            hist["brake"].append(control.brake)

            if i % 5 == 0:
                extra = f"   a_cmd={ctx.a_cmd:+.2f} m/s^2" if hasattr(ctx, "a_cmd") else ""
                print(f"[{controller.label}] t={t:5.1f}s   v_x={v_x:5.2f} m/s   v_ref={v_ref:5.2f} m/s   "
                      f"e_vel={v_ref - v_x:+.2f} m/s{extra}   u={u:+.2f}")

            if t >= args.duration:
                print(f"[{controller.label}] Reached duration ({args.duration:.0f}s).")
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    finally:
        if recorder is not None:
            recorder.close()  # before vehicle.destroy(): the camera is attached to it
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        vehicle.destroy()

    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--controller", nargs="+", default=["mpc+ff+pid"],
                        choices=("mpc+ff+pid", "mpc+pid", "pid"),
                        help="which longitudinal controller(s) to run and score -- mpc+ff+pid: "
                             "SpeedMPC -> LUT feedforward + PID pedal layer; mpc+pid: SpeedMPC -> "
                             "PID-only pedal layer, no LUT (validate_lut.py's 'PID only' baseline); "
                             "pid: plain speed PID straight to the pedal, no MPC at all. Pass more "
                             "than one to overlay them on one comparison figure")
    parser.add_argument("--initial-speed", type=float, default=15.0,
                        help="m/s; starting speed, also the sine profile's midline and the step "
                             "profile's pre-step level")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--duration", type=float, default=15.0, help="scored run length (s)")
    parser.add_argument("--times-run", type=float, default=2.0, help="how times for simulation running?")

    parser.add_argument("--profile", default="constant", choices=("constant", "sine", "step"),
                        help="speed reference shape: flat initial-speed, a sine wave around it, "
                             "or a single step away from it partway through the run")
    parser.add_argument("--sine-amplitude", type=float, default=3.0, help="sine profile peak deviation (m/s)")
    parser.add_argument("--sine-period", type=float, default=1.0, help="sine profile period (s)")
    parser.add_argument("--step-size", type=float, default=5.0,
                        help="step profile: m/s added to initial-speed after --step-time (negative = "
                             "a deceleration step)")
    parser.add_argument("--step-time", type=float, default=5.0,
                        help="step profile: when the step happens, seconds into the scored run")

    # ---- MPC ---- #
    mpc = parser.add_argument_group("--controller mpc+ff+pid / mpc+pid")
    mpc.add_argument("--np", dest="n_p", type=int, default=40, help="prediction horizon (steps)")
    mpc.add_argument("--nc", dest="n_c", type=int, default=40, help="control horizon (steps, <= --np)")
    mpc.add_argument("--w-v", type=float, default=10.0, help="speed-tracking weight")
    mpc.add_argument("--w-a", type=float, default=1, help="commanded-acceleration magnitude weight")
    mpc.add_argument("--w-j", type=float, default=10, help="commanded-acceleration rate (jerk) weight")
    mpc.add_argument("--a-min", type=float, default=-4.05, help="hard lower bound on a_cmd (m/s^2)")
    mpc.add_argument("--a-max", type=float, default=2.4, help="hard upper bound on a_cmd (m/s^2)")
    mpc.add_argument("--lut", default=os.path.join(HERE, "longitudinal_lookup", "longitudinal_lut.npz"))
    mpc.add_argument("--kp", type=float, default=0.15, help="accel-tracking PID proportional gain")
    mpc.add_argument("--ki", type=float, default=0.6, help="accel-tracking PID integral gain")
    mpc.add_argument("--kd", type=float, default=0.0, help="accel-tracking PID derivative gain")
    mpc.add_argument("--u-tau", type=float, default=0.02,
                     help="low-pass filter time constant on the pedal command u, before it's applied (s) "
                          "-- kept low: multi-controller sine sweeps showed a slower actuator response "
                          "here lags the LUT+PID inner loop more, producing bigger corrections later and "
                          "*more* jerk, not less (0.2 measured ~1.5x the jerk of 0.02 at equal w_v/w_a/w_j)")

    # ---- PID ---- #
    pid = parser.add_argument_group("--controller pid")
    pid.add_argument("--pid-kp", type=float, default=0.5)
    pid.add_argument("--pid-ki", type=float, default=0.2)
    pid.add_argument("--pid-kd", type=float, default=0.05)
    pid.add_argument("--pid-tau", type=float, default=0.1, help="output low-pass time constant (s)")

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"),
                         help="directory to save the end-of-run result figures into")
    parser.add_argument("--save-plot", action="store_true",
                        help="draw and save the end-of-run result figures (off by default)")

    # ---- video ---- #
    parser.add_argument("--record", nargs="?", const="auto", default="",
                        help="record the drive to an mp4; bare flag auto-names it under --video-dir")
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

    # A map that was just (re)loaded above is still settling server-side for a few frames --
    # ticking synchronously against it too soon can leave a freshly-spawned sensor's first
    # callback missing, which showed up as queue.Empty on run_trial()'s very first IMU read.
    # A handful of throwaway ticks before anything is spawned lets that settle once, here,
    # instead of every trial having to guard against it.
    for _ in range(10):
        world.tick()

    origin_transform = world.get_map().get_spawn_points()[ORIGIN_INDEX]

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    keys = list(dict.fromkeys(args.controller))   # de-dupe, keep the order given on the CLI

    def recorder_factory(key, n_trials):
        if not args.record:
            return lambda vehicle: None
        suffix = key.replace("+", "-") if n_trials > 1 else ""

        def make(vehicle):
            if args.record == "auto" or n_trials > 1:
                path = os.path.join(args.video_dir, run_name(suffix) + ".mp4")
            else:
                path = args.record
            rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
            return VideoRecorder(world, vehicle, path, fps=1.0 / args.dt, width=rec_w, height=rec_h,
                                 view=args.record_view)
        return make

    results = {}
    try:
        for key in keys:
            controller = CONTROLLERS[key](args)
            print(f"\n=== running {controller.label} ===")
            results[controller.label] = run_trial(world, origin_transform, blueprint, imu_bp,
                                                   controller, args,
                                                   recorder_factory(key, len(keys)))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("Cleaned up: world settings restored.")

    have_data = all(len(hist["t"]) > 1 for hist in results.values())
    if not args.save_plot or not have_data:
        for label, hist in results.items():
            if len(results) > 1:
                print(f"\n### {label} ###")
            print_error_summary(hist, args.initial_speed)  # plot_longitudinal_result prints it otherwise
    else:
        data = results if len(results) > 1 else next(iter(results.values()))
        title = " vs ".join(results) if len(results) > 1 else next(iter(results), "")
        try:
            plot_longitudinal_result(data, args.initial_speed, args.plot_dir,
                                     label=f"{title} ({args.profile})")
        except Exception as exc:
            print(f"Plotting failed: {exc}")
            for label, hist in results.items():
                print_error_summary(hist, args.initial_speed)


if __name__ == "__main__":
    main()
