"""Longitudinal controller comparison: PID vs MPC, both with the same Stanley lateral controller.

Pick the longitudinal controller with --long-ctrl:

    pid   speed PID -> low-pass -> pedal          (the controller from stanley_pathtracking_v2.py)
    mpc   MPC -> a_target -> LUT + accel PI -> pedal   (the stack from longitudinal_mpc_lateral_stanley.py)

Everything else -- route, vehicle, Stanley gains, speed reference, logging, scoring -- is shared, so
a pair of runs differs only in the longitudinal controller. Both track the same speed profile
v = v0 + A sin(2*pi*t/T); pass --amplitude 0 for the constant setpoint stanley_pathtracking_v2.py
uses. The PID stack has no acceleration command of its own, so a_target and jerk are logged as NaN
for it and the scoring skips them.

Every run writes its history to runs/<ctrl>_<stamp>.npz. Overlay two of them with --compare, which
needs no CARLA server:

    .venv/bin/python long_comp.py --long-ctrl pid --record
    .venv/bin/python long_comp.py  --long-ctrl mpc --record
    .venv/bin/python long_comp.py  --compare runs/pid_*.npz runs/mpc_*.npz
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "longitudinal_lookup"))

import carla

# non-longitudinal machinery is shared with the MPC script rather than copied, so the two stay in
# step; the PID itself comes from the v2 script so it is literally the controller being compared
from longitudinal_mpc_lateral_stanley import (
    AngleUnwrapper, LowPassFilter, build_path, build_x_ref, clipping, estimate_lag,
    get_vehicle_geometry, lateral_error, speed_profile, stanley_control,
)
from stanley_pathtracking_v2 import PID

from viz_utils import BevView, VIEWS, VideoRecorder, follow_with_spectator

from longitudinal_lut import LongitudinalLUT
from longitudinal_pi import LongitudinalAccelPI
from longitudinal_mpc import LongitudinalMPC
from build_longitudinal_lut import CollisionWatch

NAN = float("nan")


# ----------------------------------------------------------------------------- controllers
# Both expose step(t, v_x, a_meas, gear, args) -> (u, a_target, jerk), where u is the pedal command
# in [-1, 1] (positive throttle, negative brake). a_target/jerk are NaN for stacks that have no
# such internal signal.

class PidLongitudinal:
    """Speed PID straight to the pedal -- stanley_pathtracking_v2.py's longitudinal controller.

    Note this PID does not clamp or anti-wind-up internally (its class doesn't), so the integral
    keeps growing while the pedal is railed; the clip happens only on the way out. That is the
    controller as written, and changing it here would mean comparing something else.
    """
    label = "PID"

    def __init__(self, args):
        self.pid = PID(kp=args.pid_kp, ki=args.pid_ki, kd=args.pid_kd, dt=args.dt)
        self.filter = LowPassFilter(tau=args.pid_tau, dt=args.dt, initial=0.0)

    def step(self, t, v_x, a_meas, gear, args):
        v_des, _ = speed_profile(t, args)
        u = clipping(self.filter.step(self.pid.step(v_des - v_x)), 1.0, -1.0)
        return u, NAN, NAN

    def describe(self, args):
        return (f"PID kp={args.pid_kp} ki={args.pid_ki} kd={args.pid_kd}, "
                f"output low-pass tau={args.pid_tau}s -> pedal directly")


class MpcLongitudinal:
    """MPC picks the jerk; the LUT converts the implied acceleration to a pedal input and an
    acceleration PI closes the remainder -- longitudinal_mpc_lateral_stanley.py's stack."""
    label = "MPC"

    def __init__(self, args):
        lut = LongitudinalLUT(args.lut)
        self.mpc = LongitudinalMPC(dt=args.dt, n_p=args.n_p, n_c=args.n_c,
                                   w_v=args.w_v, w_a=args.w_a, w_j=args.w_j,
                                   jerk_max=args.jerk_max, command_mode=args.command_mode)
        self.mpc.reset(0.0)
        self.accel_pi = LongitudinalAccelPI(lut, kp=args.kp, ki=args.ki, tau=args.accel_tau)
        self.u_prev = 0.0

    def step(self, t, v_x, a_meas, gear, args):
        jerk, a_target = self.mpc.solve(v_x, a_meas, build_x_ref(self.mpc, t, args),
                                        u_prev=self.u_prev)
        u = self.accel_pi.step(gear, v_x, a_target, args.dt, a_meas=a_meas)
        self.u_prev = u
        return u, a_target, jerk

    def describe(self, args):
        return (f"MPC Np={args.n_p} Nc={args.n_c} (horizon {args.n_p*args.dt:.1f}s), "
                f"w_v={args.w_v} w_a={args.w_a} w_j={args.w_j}, |jerk|<={args.jerk_max} "
                f"-> LUT + accel PI(kp={args.kp}, ki={args.ki})")


CONTROLLERS = {"pid": PidLongitudinal, "mpc": MpcLongitudinal}


# ----------------------------------------------------------------------------- scoring & plots

def report(hist, args, label):
    """Score the whole run -- there is no warm-up window, every sample counts."""
    v, v_des = np.array(hist["v_x"]), np.array(hist["v_des"])
    e_v = v - v_des
    lag, corr = estimate_lag(v_des, v, args.dt)

    print(f"\n=== [{label}] speed profile tracking ===")
    print(f"  MAE {np.mean(np.abs(e_v)):.3f} m/s   RMSE {np.sqrt(np.mean(e_v**2)):.3f}   "
          f"max {np.abs(e_v).max():.3f}   bias {np.mean(e_v):+.3f}")
    print(f"  correlation peaks at {lag*1000:.0f} ms lag ({corr:.4f})")

    u = np.array(hist["u"])
    print(f"  pedal saturated {100*np.mean(np.abs(u) >= 0.999):.1f}%   gears {sorted(set(hist['gear']))}")

    a_meas, a_des = np.array(hist["a_meas"]), np.array(hist["a_des"])
    print(f"  acceleration MAE {np.mean(np.abs(a_meas - a_des)):.3f} m/s^2  "
          f"(actual {a_meas.min():+.2f}~{a_meas.max():+.2f}, ref {a_des.min():+.2f}~{a_des.max():+.2f})")

    jerk = np.array(hist["jerk"])
    if not np.all(np.isnan(jerk)):   # MPC only
        on_limit = np.mean(np.abs(jerk) >= args.jerk_max - 1e-3)
        print(f"  jerk {jerk.min():+.2f}~{jerk.max():+.2f} (limit +-{args.jerk_max}), "
              f"on the limit {100*on_limit:.1f}% of the time")

    # lateral is identical between runs by construction -- report it to confirm that held
    e_y = np.abs(np.array(hist["e_y"]))
    print(f"  [lateral] cross-track MAE {e_y.mean():.3f} m  max {e_y.max():.3f} m")


def save_run(hist, args, label, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{args.long_ctrl}_{stamp}.npz")
    np.savez(out_path, label=label, args=json.dumps(vars(args)),
             **{k: np.array(v, dtype=float) for k, v in hist.items()})
    print(f"Run data saved: {out_path}")
    return out_path


def _panels(n_rows, title):
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True,
                             constrained_layout=True)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=13)
    return fig, axes


def plot_run(hist, args, label, out_dir):
    t = np.array(hist["t"])
    fig, (ax_v, ax_e, ax_a, ax_j, ax_u) = _panels(
        5, f"{label} longitudinal control — $v_0$={args.v0} m/s, A={args.amplitude}, T={args.period}s")

    ax_v.plot(t, hist["v_des"], "--", color="0.45", lw=2.2, label="reference")
    ax_v.plot(t, hist["v_x"], color="#2a78d6", lw=1.7, label="measured")
    ax_v.set_ylabel("speed (m/s)")
    ax_v.legend(frameon=False, ncol=2)

    e_v = np.array(hist["v_x"]) - np.array(hist["v_des"])
    ax_e.axhline(0.0, color="0.45", lw=1.2, ls="--")
    ax_e.plot(t, e_v, color="#e34948", lw=1.4)
    ax_e.fill_between(t, e_v, 0, color="#e34948", alpha=0.15)
    ax_e.set_ylabel("speed error (m/s)")

    ax_a.plot(t, hist["a_des"], "--", color="0.45", lw=2.2, label="reference $a_{des}$")
    if not np.all(np.isnan(hist["a_target"])):
        ax_a.plot(t, hist["a_target"], color="#1baf7a", lw=1.2, label="$a_{target}$")
    ax_a.plot(t, hist["a_meas"], color="#2a78d6", lw=1.6, label="measured")
    ax_a.set_ylabel("acceleration (m/s$^2$)")
    ax_a.legend(frameon=False, ncol=3)

    # the MPC's actual decision variable; the limit lines show when it is running railed, which is
    # what a low w_j looks like
    ax_j.set_ylabel("jerk (m/s$^3$)")
    if np.all(np.isnan(hist["jerk"])):
        ax_j.text(0.5, 0.5, "no jerk command — this stack drives the pedal directly",
                  transform=ax_j.transAxes, ha="center", va="center", color="0.5", fontsize=11)
    else:
        ax_j.axhline(0.0, color="0.7", lw=1)
        ax_j.axhline(args.jerk_max, color="#e34948", lw=1.2, ls="--", label="$\\pm j_{max}$")
        ax_j.axhline(-args.jerk_max, color="#e34948", lw=1.2, ls="--")
        ax_j.plot(t, hist["jerk"], color="#8b5cf6", lw=1.3, label="MPC jerk $j$")
        ax_j.legend(frameon=False, ncol=2)

    ax_u.axhline(0.0, color="0.7", lw=1)
    ax_u.plot(t, hist["u"], color="#eb6834", lw=1.4, label="pedal $u$")
    ax_u.set_ylim(-1.15, 1.15)
    ax_u.set_ylabel("$u$")
    ax_u.set_xlabel("t (s)")
    ax_g = ax_u.twinx()
    ax_g.step(t, hist["gear"], color="0.45", lw=1.2, where="post", label="gear")
    ax_g.set_ylabel("gear")
    ax_g.set_ylim(-0.5, 7)
    lines = ax_u.get_lines()[1:] + ax_g.get_lines()
    ax_u.legend(lines, [l.get_label() for l in lines], frameon=False, ncol=2)

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{args.long_ctrl}_{stamp}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved: {out_path}")
    return out_path, fig


def plot_compare(npz_paths, out_dir):
    """Overlay runs saved by save_run(). The reference comes from the first one; a mismatched
    reference means the runs aren't comparable, so say so rather than drawing a misleading chart."""
    runs = [(os.path.basename(p), np.load(p, allow_pickle=True)) for p in npz_paths]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8b5cf6"]

    fig, (ax_v, ax_e, ax_a, ax_j, ax_u) = _panels(5, "Longitudinal controller comparison")
    base = runs[0][1]
    jerk_limits = set()
    ax_v.plot(base["t"], base["v_des"], "--", color="0.45", lw=2.2, label="reference")
    ax_a.plot(base["t"], base["a_des"], "--", color="0.45", lw=2.2, label="reference")

    print("\n=== comparison ===")
    for i, (name, d) in enumerate(runs):
        label = str(d["label"])
        color = colors[i % len(colors)]
        t = d["t"]
        n = min(len(t), len(base["t"]))
        if not np.allclose(d["v_des"][:n], base["v_des"][:n], atol=1e-6):
            print(f"  warning: {name} tracks a different reference than the first run "
                  f"-- these are not comparable")
        e_v = d["v_x"] - d["v_des"]
        ax_v.plot(t, d["v_x"], color=color, lw=1.6, label=label)
        ax_e.plot(t, e_v, color=color, lw=1.4, label=label)
        ax_a.plot(t, d["a_meas"], color=color, lw=1.5, label=label)
        ax_u.plot(t, d["u"], color=color, lw=1.3, label=label)

        print(f"  {label:>4}  MAE {np.mean(np.abs(e_v)):.3f}  "
              f"RMSE {np.sqrt(np.mean(e_v**2)):.3f}  max {np.abs(e_v).max():.3f} m/s  "
              f"saturated {100*np.mean(np.abs(d['u']) >= 0.999):.1f}%  "
              f"mean |e_y| {np.mean(np.abs(d['e_y'])):.3f} m   ({name})")

    ax_e.axhline(0.0, color="0.45", lw=1.2, ls="--")
    ax_u.axhline(0.0, color="0.7", lw=1)
    ax_u.set_ylim(-1.15, 1.15)
    ax_v.set_ylabel("speed (m/s)")
    ax_e.set_ylabel("speed error (m/s)")
    ax_a.set_ylabel("acceleration (m/s$^2$)")
    ax_u.set_ylabel("pedal $u$")
    ax_u.set_xlabel("t (s)")
    for ax in (ax_v, ax_e, ax_a, ax_u):
        ax.legend(frameon=False, ncol=len(runs) + 1)

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"compare_{stamp}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nComparison figure saved: {out_path}")
    return out_path, fig


# ----------------------------------------------------------------------------- args

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--long-ctrl", default="mpc", choices=sorted(CONTROLLERS),
                   help="which longitudinal controller to drive with")
    p.add_argument("--compare", nargs="+", metavar="RUN.npz",
                   help="overlay saved runs and exit; no CARLA server needed")

    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--map", default="", help="load this map first, e.g. Town06")
    p.add_argument("--origin-index", type=int, default=0, help="route start spawn point")
    p.add_argument("--dest-index", type=int, default=100, help="route end spawn point")
    p.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    p.add_argument("--max-duration", type=float, default=30.0, help="safety cutoff (s)")
    p.add_argument("--times-run", type=float, default=2.0, help="wall-clock speed-up factor")
    p.add_argument("--gear", type=int, default=0, help="force this gear; 0 leaves shifting to CARLA")
    p.add_argument("--collision-trim", type=float, default=0.5,
                   help="drop this many seconds of data before a crash (s)")

    ref = p.add_argument_group("speed reference  v = v0 + A sin(2 pi t / T)")
    ref.add_argument("--v0", type=float, default=10.0, help="centre speed (m/s)")
    ref.add_argument("--amplitude", type=float, default=4.0, help="A (m/s); 0 = constant setpoint")
    ref.add_argument("--period", type=float, default=10.0, help="T (s)")

    lat = p.add_argument_group("lateral (Stanley -- identical for both controllers)")
    lat.add_argument("--k-theta", type=float, default=1.0, help="heading-error gain")
    lat.add_argument("--k-e", type=float, default=1.2, help="cross-track gain")

    pid = p.add_argument_group("--long-ctrl pid")
    pid.add_argument("--pid-kp", type=float, default=0.35)
    pid.add_argument("--pid-ki", type=float, default=0.15)
    pid.add_argument("--pid-kd", type=float, default=0.05)
    pid.add_argument("--pid-tau", type=float, default=0.1, help="output low-pass time constant (s)")

    mpc = p.add_argument_group("--long-ctrl mpc")
    mpc.add_argument("--lut", default=os.path.join(HERE, "longitudinal_lookup", "longitudinal_lut.npz"))
    mpc.add_argument("--np", dest="n_p", type=int, default=20, help="prediction horizon (steps)")
    mpc.add_argument("--nc", dest="n_c", type=int, default=10, help="control horizon (steps)")
    mpc.add_argument("--w-v", type=float, default=1.0, help="speed-tracking weight")
    mpc.add_argument("--w-a", type=float, default=0.1, help="acceleration weight")
    mpc.add_argument("--w-j", type=float, default=0.1, help="jerk weight")
    mpc.add_argument("--jerk-max", type=float, default=4.13, help="jerk limit (m/s^3)")
    mpc.add_argument("--command-mode", default="integrate", choices=("integrate", "anchor"))
    mpc.add_argument("--kp", type=float, default=0.10, help="accel-tracking PI proportional gain")
    mpc.add_argument("--ki", type=float, default=0.40, help="accel-tracking PI integral gain")
    mpc.add_argument("--accel-tau", type=float, default=0.10,
                     help="time constant of the measured-accel filter (s)")

    out = p.add_argument_group("output")
    out.add_argument("--record", action="store_true", help="record the drive to an mp4")
    out.add_argument("--record-view", default="chase", choices=sorted(VIEWS))
    out.add_argument("--record-res", default="1280x720", help="recording resolution, WxH")
    out.add_argument("--video-dir", default=os.path.join(HERE, "videos"))
    out.add_argument("--run-dir", default=os.path.join(HERE, "runs"),
                     help="where per-run .npz histories go")
    out.add_argument("--plot-dir", default=os.path.join(HERE, "plots"))
    out.add_argument("--no-plot", action="store_true", help="skip the result figure")
    out.add_argument("--no-show", action="store_true", help="save figures but do not open a window")
    out.add_argument("--no-live-view", action="store_true", help="skip the live BEV view")
    return p


# ----------------------------------------------------------------------------- run

def main():
    args = build_parser().parse_args()

    if args.compare:
        _, fig = plot_compare(args.compare, args.plot_dir)
        if not args.no_show:
            plt.show()
        plt.close(fig)
        return

    controller = CONTROLLERS[args.long_ctrl](args)
    print(f"Longitudinal controller: {controller.describe(args)}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.map) if args.map else client.get_world()
    print(f"Map: {world.get_map().name}")

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    origin_transform, path_x, path_y, path_yaw = build_path(
        world, origin_index=args.origin_index, dest_index=args.dest_index)
    print(f"Route: {len(path_x)} points, start=({path_x[0]:.1f}, {path_y[0]:.1f}) "
          f"goal=({path_x[-1]:.1f}, {path_y[-1]:.1f})")

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    vehicle = world.spawn_actor(blueprint, origin_transform)

    initial_yaw = math.radians(origin_transform.rotation.yaw)
    if args.v0 > 0:
        vehicle.set_target_velocity(carla.Vector3D(
            x=args.v0 * math.cos(initial_yaw),
            y=args.v0 * math.sin(initial_yaw), z=0.0))

    _, max_steer = get_vehicle_geometry(vehicle)

    accel_filter = LowPassFilter(tau=args.accel_tau, dt=args.dt, initial=0.0)
    steer_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0.0)
    yaw_unwrapper, rh_unwrapper = AngleUnwrapper(), AngleUnwrapper()
    prev_v, last_idx = None, 0

    bev = None if args.no_live_view else BevView(path_x, path_y)

    recorder = None
    if args.record:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
        recorder = VideoRecorder(world, vehicle,
                                 os.path.join(args.video_dir, f"{args.long_ctrl}_{stamp}.mp4"),
                                 fps=1.0 / args.dt, width=rec_w, height=rec_h,
                                 view=args.record_view)

    collision = CollisionWatch(world, vehicle)
    collision.arm()

    hist = {k: [] for k in ("t", "x", "y", "v_x", "steer_deg", "throttle", "brake", "e_y", "yaw",
                            "path_yaw", "e_theta", "a_meas", "a_target", "jerk", "gear", "u",
                            "v_des", "a_des")}

    try:
        world.tick()
        for i in range(int(args.max_duration / args.dt)):
            step_start = time.time()
            world.tick()

            transform = vehicle.get_transform()
            yaw = yaw_unwrapper.step(math.radians(transform.rotation.yaw))
            ego_x, ego_y = transform.location.x, transform.location.y
            vel_vec = vehicle.get_velocity()
            # body-frame longitudinal velocity (not vel_vec.x, which is world-frame)
            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)
            road_heading = rh_unwrapper.step(path_yaw[last_idx])

            last_idx, e_y = lateral_error(ego_x, ego_y, yaw, path_x, path_y, last_idx)
            e_theta = road_heading - yaw

            delta = stanley_control(v_x, e_y, e_theta, k_theta=args.k_theta, k=args.k_e)
            steer = clipping(delta / max_steer, 3 / 7, -3 / 7)

            # filtered dv/dt: the MPC state and the scoring both read this one signal
            a_meas = accel_filter.step((v_x - prev_v) / args.dt) if prev_v is not None else 0.0
            prev_v = v_x

            t = i * args.dt
            v_des, a_des = speed_profile(t, args)
            gear = args.gear if args.gear > 0 else vehicle.get_control().gear
            u, a_target, jerk = controller.step(t, v_x, a_meas, gear, args)

            control = carla.VehicleControl()
            if args.gear > 0:
                control.manual_gear_shift = True
                control.gear = args.gear
            control.throttle, control.brake = (u, 0.0) if u >= 0 else (0.0, -u)
            control.steer = steer_filter.step(steer)
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            for key, value in (("t", t), ("x", ego_x), ("y", ego_y), ("v_x", v_x),
                               ("steer_deg", steer * math.degrees(max_steer)),
                               ("throttle", control.throttle), ("brake", control.brake),
                               ("e_y", e_y), ("yaw", math.degrees(yaw)),
                               ("path_yaw", math.degrees(road_heading)),
                               ("e_theta", math.degrees(e_theta)), ("a_meas", a_meas),
                               ("a_target", a_target), ("jerk", jerk), ("gear", gear), ("u", u),
                               ("v_des", v_des), ("a_des", a_des)):
                hist[key].append(value)

            if i % 10 == 0:
                print(f"t={t:5.1f}s  v={v_x:5.2f}/{v_des:5.2f}  e_v={v_x-v_des:+5.2f}  "
                      f"a={a_meas:+5.2f}/{a_des:+5.2f}  u={u:+.2f}  gear={gear}  e_y={e_y:+.2f}")

            if collision.hit:
                # a crash writes a huge negative acceleration into the log; scoring past it would
                # measure the wall, so stop and drop the last moments
                print(f"\nCollision at t={t:.1f}s -- stopping, discarding the last "
                      f"{args.collision_trim}s of data")
                for key in hist:
                    del hist[key][-int(round(args.collision_trim / args.dt)):]
                break

            if last_idx >= len(path_x) - 1:
                print(f"Reached end of path ({last_idx}/{len(path_x) - 1}).")
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        collision.destroy()
        if recorder is not None:
            recorder.close()   # before vehicle.destroy(): the camera is attached to it
        if bev is not None:
            bev.close()
            plt.ioff()   # BevView leaves interactive mode on
        vehicle.destroy()
        world.apply_settings(original_settings)
        print("Cleaned up: vehicle destroyed, world settings restored.")

    if len(hist["t"]) > 1:
        report(hist, args, controller.label)
        save_run(hist, args, controller.label, args.run_dir)
        if not args.no_plot:
            _, fig = plot_run(hist, args, controller.label, args.plot_dir)
            if not args.no_show:
                print("Close the window to exit.")
                plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()
