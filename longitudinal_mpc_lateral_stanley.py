"""Stanley lateral controller + MPC longitudinal controller.

Longitudinal stack, top to bottom:

    MPC          decides jerk over a horizon, giving the acceleration the
                 vehicle should hold one step from now
    lookup table converts (gear, speed, desired accel) into a pedal input
    PI           closes the remaining gap between commanded and measured accel

The MPC plant is a jerk-input double integrator in [speed, acceleration]; it
never sees pedals. Everything below a_target is the calibrated table's job,
which keeps the optimisation linear and the QP small.

Validation drives a speed profile rather than a constant setpoint:

    v_des = v0 + A sin(wt)      a_des = A w cos(wt)

The pair is the derivative of itself, so the reference is something the plant
model can actually follow -- a constant a_des of 0 asks the vehicle to change
speed and hold zero acceleration at the same time, and no controller can score
well against that. Note A w^2 is the jerk the profile demands, so shortening
the period raises it quadratically; keep it under the MPC's jerk limit or the
run measures the constraint rather than the controller.

Lateral is Stanley on a GlobalRoutePlanner route, kept only so the vehicle
stays on the road; the routes used here are gentle enough that steering stays
near zero and does not disturb the longitudinal measurement.

Usage:
    cd ~/carla_control
    .venv/bin/python longitudinal_mpc_lateral_stanley.py
    .venv/bin/python longitudinal_mpc_lateral_stanley.py --period 6 --amplitude 3
"""

import argparse
import datetime
import math
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# GlobalRoutePlanner lives in CARLA's PythonAPI tree, not in the pip package
sys.path.append("/home/ailab/2026intern/carla/PythonAPI/carla")
sys.path.append(os.path.join(HERE, "longitudinal_lookup"))

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

from viz_utils import follow_with_spectator, plot_results
from mock_planner import MockPlanner, local_to_world
from bev_view import BevView

from longitudinal_lut import LongitudinalLUT
from longitudinal_pi import LongitudinalAccelPI
from longitudinal_mpc import LongitudinalMPC
from build_longitudinal_lut import CollisionWatch



class LowPassFilter:
    def __init__(self, tau, dt, initial=0.0):
        self.alpha = dt / (tau + dt)
        self.state = initial

    def step(self, x):
        self.state += self.alpha * (x - self.state)
        return self.state
        

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
    yaw_unwrapper = AngleUnwrapper()
    path_yaw = [yaw_unwrapper.step(math.radians(wp.transform.rotation.yaw)) for wp, _ in route]
    return origin_transform, path_x, path_y, path_yaw


def get_vehicle_geometry(vehicle):
    """Wheelbase (m) and max front-wheel steer angle (rad) from the spawned vehicle's physics."""
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

    print(front_mid/100,rear_mid/100)
    print(f"wheelbase={wheelbase:.2f} m  max_steer={max_steer_deg:.1f} deg")
    return wheelbase, math.radians(max_steer_deg)

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


def world_to_local(x, y, yaw, origin_x, origin_y, origin_yaw):
    """Inverse of mock_planner.local_to_world for a single pose: world (x, y, yaw) into the
    ego-relative frame defined by (origin_x, origin_y, origin_yaw)."""
    dx, dy = x - origin_x, y - origin_y
    cos_o, sin_o = math.cos(origin_yaw), math.sin(origin_yaw)
    local_x = dx * cos_o + dy * sin_o
    local_y = -dx * sin_o + dy * cos_o
    return local_x, local_y, normalize_angle(yaw - origin_yaw)



def stanley_control(v_x, e_y, e_theta, k_theta=0.5,k=1):
    delta = k_theta * e_theta + math.atan2(k * e_y, v_x)
    return delta


def speed_profile(t, args):
    """Reference (v_des, a_des) at time t, with a_des = d v_des / dt.

    The warm-up holds v0 with zero acceleration. CARLA's transmission starts in
    first gear no matter what speed the vehicle is spawned at, so without it the
    profile would begin during a violent downshift transient.
    """
    if t < args.warmup:
        return args.v0, 0.0
    tau = t - args.warmup
    w = 2.0 * math.pi / args.period
    return args.v0 + args.amplitude * math.sin(w * tau), args.amplitude * w * math.cos(w * tau)


def build_x_ref(mpc, t, args):
    """Stack the profile over the prediction horizon, one step per row."""
    v_des, a_des = [], []
    for i in range(1, mpc.n_p + 1):
        v, a = speed_profile(t + i * mpc.T, args)
        v_des.append(v)
        a_des.append(a)
    return mpc.reference(v_des, a_des)


def estimate_lag(ref, meas, dt, max_lag_s=1.5):
    """Time shift that best aligns the response with the reference."""
    ref, meas = np.asarray(ref), np.asarray(meas)
    best_lag, best_corr = 0.0, -2.0
    for k in range(int(max_lag_s / dt) + 1):
        a = ref[:ref.size - k] if k else ref
        b = meas[k:]
        if a.size < 10 or a.std() == 0 or b.std() == 0:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c > best_corr:
            best_corr, best_lag = c, k * dt
    return best_lag, best_corr


def report(hist, args):
    """Score the run once the warm-up is over."""
    t = np.array(hist["t"])
    scored = t >= args.warmup
    v_des = np.array(hist["v_des"])[scored]
    v = np.array(hist["v_x"])[scored]
    a_des = np.array(hist["a_des"])[scored]
    a_meas = np.array(hist["a_meas"])[scored]
    u = np.array(hist["u"])[scored]
    jerk = np.array(hist["jerk"])[scored]

    e_v = v - v_des
    lag, corr = estimate_lag(v_des, v, args.dt)

    print("\n=== 속도 프로파일 추종 ===")
    print(f"  MAE {np.mean(np.abs(e_v)):.3f} m/s   RMSE {np.sqrt(np.mean(e_v**2)):.3f}   "
          f"최대 {np.abs(e_v).max():.3f}   bias {np.mean(e_v):+.3f}")
    print(f"  지연 {lag*1000:.0f} ms 에서 상관 최대 ({corr:.4f})")
    print(f"  속도 범위 {v.min():.2f} ~ {v.max():.2f} m/s  (기준 {v_des.min():.2f} ~ {v_des.max():.2f})")

    e_a = a_meas - a_des
    print("\n=== 가속도 (참고) ===")
    print(f"  MAE {np.mean(np.abs(e_a)):.3f} m/s^2   실제 범위 {a_meas.min():+.2f} ~ {a_meas.max():+.2f}  "
          f"(기준 {a_des.min():+.2f} ~ {a_des.max():+.2f})")
    print(f"  MPC jerk 범위 {jerk.min():+.2f} ~ {jerk.max():+.2f}  (제약 ±{args.jerk_max})")
    on_limit = np.mean(np.abs(jerk) >= args.jerk_max - 1e-3)
    print(f"  jerk 제약에 붙은 비율 {100*on_limit:.1f}%")
    print(f"  제어입력 포화 {100*np.mean(np.abs(u) >= 0.999):.1f}%   기어 {sorted(set(hist['gear']))}")


def plot_speed_tracking(hist, args, out_dir):
    t = np.array(hist["t"])
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    ax_v, ax_e, ax_a, ax_u = axes
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    if args.warmup > 0:
        for ax in axes:
            ax.axvspan(0, args.warmup, color="0.9", zorder=0)

    ax_v.plot(t, hist["v_des"], "--", color="0.45", lw=2.2, label="reference")
    ax_v.plot(t, hist["v_x"], color="#2a78d6", lw=1.7, label="measured")
    ax_v.set_ylabel("speed (m/s)")
    ax_v.set_title("Speed profile tracking   (grey band = warm-up, excluded from scoring)")
    ax_v.legend(frameon=False, ncol=2)

    e_v = np.array(hist["v_x"]) - np.array(hist["v_des"])
    ax_e.axhline(0.0, color="0.45", lw=1.2, ls="--")
    ax_e.plot(t, e_v, color="#e34948", lw=1.4)
    ax_e.fill_between(t, e_v, 0, color="#e34948", alpha=0.15)
    ax_e.set_ylabel("speed error (m/s)")

    ax_a.plot(t, hist["a_des"], "--", color="0.45", lw=2.2, label="reference $a_{des}$")
    ax_a.plot(t, hist["a_target"], color="#1baf7a", lw=1.2, alpha=0.9, label="MPC $a_{target}$")
    ax_a.plot(t, hist["a_meas"], color="#2a78d6", lw=1.6, label="measured")
    ax_a.set_ylabel("acceleration (m/s$^2$)")
    ax_a.legend(frameon=False, ncol=3)

    ax_u.axhline(0.0, color="0.7", lw=1)
    ax_u.plot(t, hist["u"], color="#eb6834", lw=1.4, label="control input $u$")
    ax_u.set_ylabel("$u$")
    ax_u.set_ylim(-1.15, 1.15)
    ax_u.set_xlabel("t (s)")
    ax_g = ax_u.twinx()
    ax_g.step(t, hist["gear"], color="0.45", lw=1.2, where="post", label="gear")
    ax_g.set_ylabel("gear")
    ax_g.set_ylim(-0.5, 7)
    lines = ax_u.get_lines()[1:] + ax_g.get_lines()
    ax_u.legend(lines, [l.get_label() for l in lines], frameon=False, ncol=2)

    fig.suptitle(f"Longitudinal MPC — $v_0$={args.v0} m/s, A={args.amplitude} m/s, T={args.period}s "
                 f"(|a| peak {args.amplitude*2*math.pi/args.period:.2f}, "
                 f"|j| peak {args.amplitude*(2*math.pi/args.period)**2:.2f})\n"
                 f"Np={args.n_p} Nc={args.n_c}, w_v={args.w_v} w_a={args.w_a} w_j={args.w_j}",
                 fontsize=13)

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"mpc_speed_tracking_{stamp}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\n그래프 저장: {out_path}")
    return out_path, fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--v0", type=float, default=15.0, help="profile centre speed (m/s)")
    parser.add_argument("--amplitude", type=float, default=4.0, help="speed profile amplitude A (m/s)")
    parser.add_argument("--period", type=float, default=10.0, help="speed profile period T (s)")
    parser.add_argument("--warmup", type=float, default=15.0,
                        help="hold v0 this long before the profile starts, so the vehicle reaches it and settles (s)")
    parser.add_argument("--start-speed", type=float, default=15,
                        help="inject this speed at spawn; 0 starts from rest. CARLA's transmission always "
                             "starts in first gear, so injecting cruise speed triggers a violent downshift")
    parser.add_argument("--gear", type=int, default=0, help="force this gear; 0 leaves shifting to CARLA")
    parser.add_argument("--map", default="", help="load this map first, e.g. Town06; default keeps the current one")
    parser.add_argument("--origin-index", type=int, default=0, help="route start spawn point")
    parser.add_argument("--dest-index", type=int, default=100, help="route end spawn point")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--goal-tolerance", type=float, default=2.0, help="stop within this many meters of goal (m)")
    parser.add_argument("--times-run", type=float,default=2.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=30.0, help="safety cutoff (s)")
    parser.add_argument("--plot-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                         help="directory to save the end-of-run result figure into")
    parser.add_argument("--no-plot", action="store_true", help="skip the result figure")
    parser.add_argument("--no-show", action="store_true", help="save the figure but do not open a window")
    parser.add_argument("--no-live-view", action="store_true", help="skip the live BEV plan/vehicle view")
    parser.add_argument("--lut", default=os.path.join(HERE, "longitudinal_lookup", "longitudinal_lut.npz"))
    parser.add_argument("--np", dest="n_p", type=int, default=20, help="MPC prediction horizon (steps)")
    parser.add_argument("--nc", dest="n_c", type=int, default=10, help="MPC control horizon (steps); input held after this")
    parser.add_argument("--w-v", type=float, default=1.0, help="MPC speed-tracking weight")
    parser.add_argument("--w-a", type=float, default=0.1, help="MPC acceleration weight")
    parser.add_argument("--w-j", type=float, default=0.01, help="MPC jerk weight")
    parser.add_argument("--jerk-max", type=float, default=4.13, help="jerk limit (m/s^3)")
    parser.add_argument("--collision-trim", type=float, default=0.5,
                        help="discard this much data before a collision (s)")
    parser.add_argument("--command-mode", default="integrate", choices=("integrate", "anchor"),
                        help="how a_cmd is formed: integrated from the previous command "
                             "(with anti-windup) or re-anchored on measured acceleration")
    parser.add_argument("--kp", type=float, default=0.10, help="accel-tracking PI proportional gain")
    parser.add_argument("--ki", type=float, default=0.40, help="accel-tracking PI integral gain")
    parser.add_argument("--accel-tau", type=float, default=0.10, help="time constant of the measured-accel filter (s)")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.map) if args.map else client.get_world()
    print(f"맵: {world.get_map().name}")

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    origin_transform, path_x, path_y, path_yaw = build_path(
        world, origin_index=args.origin_index, dest_index=args.dest_index)
    # a curvy route makes Stanley work hard and pollutes the longitudinal measurement,
    # so report how straight the usable part of it actually is
    turn = [abs(math.degrees(path_yaw[i + 1] - path_yaw[i])) for i in range(len(path_yaw) - 1)]
    straight_m = next((i for i, d in enumerate(turn) if d > 1.0), len(turn))
    print(f"Route: {len(path_x)} points, start=({path_x[0]:.1f}, {path_y[0]:.1f}) "
          f"goal=({path_x[-1]:.1f}, {path_y[-1]:.1f})")
    print(f"  첫 곡선까지 {straight_m} m, 전체 방향전환 {sum(turn):.0f} deg")
    reach = args.v0 * args.max_duration
    if straight_m < reach * 0.5:
        print(f"  경고: {args.max_duration:.0f}초 동안 최대 {reach:.0f}m 를 달리는데 직선은 {straight_m}m 입니다 "
              f"— 곡선에서 이탈·충돌할 수 있습니다")

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    vehicle = world.spawn_actor(blueprint, origin_transform)

    initial_yaw = math.radians(origin_transform.rotation.yaw)
    if args.start_speed > 0:
        vehicle.set_target_velocity(carla.Vector3D(
            x=args.start_speed * math.cos(initial_yaw),
            y=args.start_speed * math.sin(initial_yaw),
            z=0.0,
        ))

    wheelbase, max_steer = get_vehicle_geometry(vehicle)

    lut = LongitudinalLUT(args.lut)
    mpc = LongitudinalMPC(dt=args.dt, n_p=args.n_p, n_c=args.n_c,
                          w_v=args.w_v, w_a=args.w_a, w_j=args.w_j, jerk_max=args.jerk_max,
                          command_mode=args.command_mode)
    mpc.reset(0.0)
    u_prev = 0.0
    accel_ctrl = LongitudinalAccelPI(lut, kp=args.kp, ki=args.ki, tau=args.accel_tau)
    w = 2.0 * math.pi / args.period
    print(f"MPC: Np={args.n_p} Nc={args.n_c} (지평 {args.n_p*args.dt:.1f}s), "
          f"w_v={args.w_v} w_a={args.w_a} w_j={args.w_j}, |jerk| <= {args.jerk_max}")
    print(f"프로파일: v = {args.v0} + {args.amplitude}·sin(2πt/{args.period}), "
          f"가속도 진폭 {args.amplitude*w:.2f} m/s^2, 저크 진폭 {args.amplitude*w*w:.2f} m/s^3")
    if args.amplitude * w * w > args.jerk_max:
        print(f"  경고: 프로파일이 요구하는 저크가 제약 {args.jerk_max}을 넘습니다 — "
              f"참조 자체가 실현 불가능하므로 제약을 측정하게 됩니다")
    print(f"기어: {'자동변속' if args.gear == 0 else f'{args.gear}단 고정'}, "
          f"워밍업 {args.warmup}s 후 채점 시작")

    accel_filter = LowPassFilter(tau=args.accel_tau, dt=args.dt, initial=0.0)
    prev_v = None
    steer_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0)
    yaw_unwrapper = AngleUnwrapper()
    rh_unwrapper = AngleUnwrapper()
    last_idx = 0

    bev = None if args.no_live_view else BevView(path_x, path_y)
    collision = CollisionWatch(world, vehicle)
    collision.arm()

    hist = {"t": [], "x": [], "y": [], "v_x": [], "last_idx": [], "steer_deg": [], "throttle": [], "brake": [],
            "e_y": [], "yaw": [], "path_yaw": [], "e_theta": [],
            "a_meas": [], "a_target": [], "jerk": [], "gear": [], "u": [],
            "v_des": [], "a_des": []}

    try:
        world.tick()
        steps = int(args.max_duration / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()

            transform = vehicle.get_transform()
            yaw = yaw_unwrapper.step(math.radians(transform.rotation.yaw))  # continuous from here on out
            ego_x = transform.location.x
            ego_y = transform.location.y
            vel_vec = vehicle.get_velocity()
            # body-frame longitudinal velocity (not vel_vec.x, which is world-frame)
            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)
            road_heading = rh_unwrapper.step(path_yaw[last_idx])

            last_idx, e_y = lateral_error(ego_x, ego_y, yaw, path_x, path_y, last_idx)
            e_theta = road_heading - yaw

            delta = stanley_control(v_x, e_y, e_theta)
            steer = clipping(delta / max_steer, 3 / 7, -3 / 7)
            steer_deg = steer * math.degrees(max_steer)

            # measured acceleration: filtered dv/dt, shared by the MPC state and
            # the PI feedback so both layers act on the same signal
            a_meas = accel_filter.step((v_x - prev_v) / args.dt) if prev_v is not None else 0.0
            prev_v = v_x

            # MPC picks the jerk; the state it implies one step ahead is the
            # acceleration the lookup table is asked to deliver
            t_now = i * args.dt
            v_des, a_des = speed_profile(t_now, args)
            x_ref = build_x_ref(mpc, t_now, args)
            jerk, a_target = mpc.solve(v_x, a_meas, x_ref, u_prev=u_prev)
            gear = args.gear if args.gear > 0 else vehicle.get_control().gear
            control_value = accel_ctrl.step(gear, v_x, a_target, args.dt, a_meas=a_meas)
            u_prev = control_value

            control = carla.VehicleControl()
            if args.gear > 0:
                control.manual_gear_shift = True
                control.gear = args.gear
            if control_value >= 0:
                control.throttle = control_value
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = -control_value
            control.steer = steer_filter.step(steer)
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            t = i * args.dt
            hist["t"].append(t)
            hist["x"].append(ego_x)
            hist["y"].append(ego_y)
            hist["v_x"].append(v_x)
            hist["last_idx"].append(last_idx)
            hist["steer_deg"].append(steer_deg)
            hist["throttle"].append(control.throttle)
            hist["brake"].append(control.brake)
            hist["e_y"].append(e_y)
            hist["yaw"].append(math.degrees(yaw))  
            hist["path_yaw"].append(math.degrees(road_heading))
            hist["e_theta"].append(math.degrees(e_theta))
            hist["a_meas"].append(a_meas)
            hist["a_target"].append(a_target)
            hist["jerk"].append(jerk)
            hist["gear"].append(gear)
            hist["u"].append(control_value)
            hist["v_des"].append(v_des)
            hist["a_des"].append(a_des)

            if i % 10 == 0:
                phase = "워밍업" if t < args.warmup else "      "
                print(f"t={t:5.1f}s {phase} v={v_x:5.2f}/{v_des:5.2f}  e_v={v_x-v_des:+5.2f}  "
                      f"a={a_meas:+5.2f}/{a_des:+5.2f}  j={jerk:+5.2f}  u={control_value:+.2f}  "
                      f"gear={gear}  e_y={e_y:+.2f}")


            if collision.hit:
                # a crash writes -100 m/s^2 into the log; scoring past it would
                # measure the wall, so stop and drop the last moments
                print(f"\n충돌 발생 (t={t:.1f}s) — 주행 중단, 직전 {args.collision_trim}s 데이터 폐기")
                drop = int(round(args.collision_trim / args.dt))
                for key in hist:
                    del hist[key][-drop:]
                break

            if last_idx >= len(path_x) - 1:
                print(f"Reached end of path (global_idx {last_idx}/{len(path_x) - 1}).")
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        collision.destroy()
        if bev is not None:
            bev.close()
            plt.ioff()  # BevView leaves interactive mode on; turn it off so plot_results()'s plt.show() blocks again
        vehicle.destroy()
        world.apply_settings(original_settings)
        print("Cleaned up: vehicle destroyed, world settings restored.")

    if len(hist["t"]) > 1:
        report(hist, args)
        if not args.no_plot:
            _, fig = plot_speed_tracking(hist, args, args.plot_dir)
            if not args.no_show:
                print("창을 닫으면 종료됩니다.")
                plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()