"""Continuous acceleration-profile tracking test.

The grid test (validate_longitudinal_lut.py) holds one target for a couple of
seconds and scores the settled value. That answers "is the static map right"
but says nothing about how the loop behaves when the target keeps moving --
which is the only thing it will ever do under an MPC that replans every tick.

Here the vehicle drives one continuous run against a time-varying reference
acceleration, jerk-limited so it stays inside what the powertrain can follow.
Nothing is reset between samples, so integrator behaviour, sign reversals and
gear changes all show up the way they would on the road.

Usage:
    cd ~/carla_control
    .venv/bin/python longitudinal_lookup/validate_accel_tracking.py --pi
    .venv/bin/python longitudinal_lookup/validate_accel_tracking.py --profile chirp --duration 60
    .venv/bin/python longitudinal_lookup/validate_accel_tracking.py --pi --compare-csv <이전결과.csv>
"""

import argparse
import csv
import math
import os

import numpy as np

import carla

from longitudinal_lut import LongitudinalLUT
from longitudinal_pi import LongitudinalAccelPI

HERE = os.path.dirname(os.path.abspath(__file__))


def body_frame_long_state(vehicle):
    yaw = math.radians(vehicle.get_transform().rotation.yaw)
    vel = vehicle.get_velocity()
    acc = vehicle.get_acceleration()
    return (vel.x * math.cos(yaw) + vel.y * math.sin(yaw),
            acc.x * math.cos(yaw) + acc.y * math.sin(yaw))


def apply_control(vehicle, u, gear=None):
    """Automatic transmission unless a gear is forced -- that is how the
    controller will actually run, and it exercises the gear-change handling."""
    control = carla.VehicleControl()
    if gear is not None:
        control.manual_gear_shift = True
        control.gear = gear
    control.hand_brake = False
    if u >= 0:
        control.throttle, control.brake = float(u), 0.0
    else:
        control.throttle, control.brake = 0.0, float(-u)
    vehicle.apply_control(control)


def rate_limit(a, jerk_max, dt):
    """Clamp the reference's rate of change so it never demands more jerk than
    the vehicle (and a passenger) will tolerate."""
    out = np.empty_like(a)
    out[0] = a[0]
    step = jerk_max * dt
    for i in range(1, a.size):
        out[i] = out[i - 1] + max(-step, min(step, a[i] - out[i - 1]))
    return out


def build_profile(args):
    """Every profile starts at zero acceleration, so each run begins from the
    same known state: cruising at v0 with a = 0."""
    t = np.arange(0.0, args.duration, args.dt)
    w = 2.0 * math.pi / args.period
    if args.profile == "sine":
        a = args.amplitude * np.sin(w * t)
    elif args.profile == "triangle":
        # 0 -> +A -> 0 -> -A -> 0
        phase = (t / args.period) % 1.0
        a = args.amplitude * np.where(phase < 0.5,
                                      np.where(phase < 0.25, 4 * phase, 2 - 4 * phase),
                                      np.where(phase < 0.75, 2 - 4 * phase, 4 * phase - 4))
    elif args.profile == "steps":
        # holds zero for the first quarter period, then alternates
        a = args.amplitude * np.where(t < args.period / 4.0, 0.0, np.sign(np.sin(w * (t - args.period / 4.0))))
    elif args.profile == "chirp":
        # frequency sweeps up, so the run shows where tracking starts to fall behind
        f0, f1 = 1.0 / args.period, args.chirp_f1
        k = (f1 - f0) / args.duration
        a = args.amplitude * np.sin(2.0 * math.pi * (f0 * t + 0.5 * k * t * t))
    else:
        raise SystemExit(f"알 수 없는 프로파일: {args.profile}")
    a[0] = 0.0
    return t, rate_limit(a, args.jerk_max, args.dt)


def case_infeasible(lut, gear, v0, args):
    """Why this (gear, speed) cannot be scored fairly, or None if it can.

    The profile swings the speed by roughly its own integral. If the vehicle
    cannot produce the commanded acceleration at the ends of that swing, the run
    spends its time pinned on the actuator limit and measures the engine, not
    the controller. Asking the table for the required input and seeing whether
    it comes back railed is the same question the vehicle will face.
    """
    swing = args.amplitude * args.period / (2.0 * math.pi)
    lo, hi = v0 - swing, v0 + swing
    if not (lut.supports(gear, lo) and lut.supports(gear, hi)):
        r = lut.speed_range[gear]
        return f"프로파일 구간 {lo:.1f}~{hi:.1f} m/s 가 유효 범위 {r[0]:.1f}~{r[1]:.1f} 밖"
    if lut.lookup(gear, hi, args.amplitude) >= 0.999:
        return f"{hi:.1f} m/s 에서 +{args.amplitude} m/s^2 요구시 풀스로틀로도 부족 (엔진 한계)"
    if lut.lookup(gear, lo, -args.amplitude) <= -0.999:
        return f"{lo:.1f} m/s 에서 -{args.amplitude} m/s^2 요구시 풀브레이크로도 부족"
    return None


def run_case(world, vehicle, origin, controller, gear, v0, a_ref, args, collision=None):
    """One continuous run: settle at (v0, a = 0), then follow the profile."""
    yaw = math.radians(origin.rotation.yaw)
    vehicle.set_transform(origin)
    vehicle.set_target_velocity(carla.Vector3D(v0 * math.cos(yaw), v0 * math.sin(yaw), 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    world.tick(); world.tick()
    controller.reset()
    if collision is not None:
        collision.arm()

    rows = []
    n_warm = int(round(args.warmup / args.dt))
    for k in range(n_warm + a_ref.size):
        warming = k < n_warm
        # warm-up holds a = 0, so the profile starts from steady cruise at v0
        a_cmd = 0.0 if warming else float(a_ref[k - n_warm])

        v_x, _ = body_frame_long_state(vehicle)
        engaged = gear if gear is not None else vehicle.get_control().gear

        # a runaway reference would leave the calibrated band and measure clamping
        if v_x > args.v_max and a_cmd > 0:
            a_cmd = 0.0
        elif v_x < args.v_min and a_cmd < 0:
            a_cmd = 0.0

        u = controller.step(engaged, v_x, a_cmd, args.dt)
        apply_control(vehicle, u, gear)
        world.tick()

        if collision is not None and collision.hit:
            # scoring a run that hit scenery would measure the wall, not the loop
            print(f"    경고: gear={gear} v0={v0:.0f} 주행 중 충돌 — t={(k-n_warm)*args.dt:.1f}s 에서 중단")
            return rows, True

        v_next, a_imu_next = body_frame_long_state(vehicle)
        if not warming:
            rows.append({
                "case_gear": gear if gear is not None else 0,
                "case_v0": v0,
                "t": round((k - n_warm) * args.dt, 3),
                "a_ref": round(a_cmd, 4),
                "a_fd": round((v_next - v_x) / args.dt, 4),
                "a_imu": round(a_imu_next, 4),
                "v": round(v_next, 4),
                "u": round(float(u), 4),
                "gear": engaged,
            })
    return rows, False


def estimate_lag(a_ref, a_meas, dt, max_lag_s=1.5):
    """Time shift that best aligns the response with the reference."""
    best_lag, best_corr = 0.0, -2.0
    for k in range(int(max_lag_s / dt) + 1):
        ref = a_ref[:a_ref.size - k] if k else a_ref
        meas = a_meas[k:]
        if ref.size < 10 or ref.std() == 0 or meas.std() == 0:
            continue
        c = float(np.corrcoef(ref, meas)[0, 1])
        if c > best_corr:
            best_corr, best_lag = c, k * dt
    return best_lag, best_corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--lut", default=os.path.join(HERE, "longitudinal_lut.npz"))
    parser.add_argument("--out", default=os.path.join(HERE, "accel_tracking.csv"))
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"))
    parser.add_argument("--map", default="", help="load this map first (e.g. Town06)")
    parser.add_argument("--origin-index", type=int, default=-1, help="-1 picks the longest straight")
    parser.add_argument("--vehicle", default="vehicle.lincoln.mkz_2020")

    parser.add_argument("--profile", default="sine", choices=("sine", "triangle", "steps", "chirp"))
    parser.add_argument("--amplitude", type=float, default=2.0, help="peak commanded accel (m/s^2)")
    parser.add_argument("--period", type=float, default=8.0, help="profile period (s)")
    parser.add_argument("--chirp-f1", type=float, default=1.0, help="chirp end frequency (Hz)")
    parser.add_argument("--duration", type=float, default=40.0, help="run length (s)")
    parser.add_argument("--jerk-max", type=float, default=2.0, help="reference jerk limit (m/s^3)")
    parser.add_argument("--warmup", type=float, default=3.0, help="settle at the profile's first value before scoring (s)")
    parser.add_argument("--start-speed", type=float, default=15.0)
    parser.add_argument("--v-min", type=float, default=6.0, help="reference is held non-negative below this (m/s)")
    parser.add_argument("--v-max", type=float, default=24.0, help="reference is held non-positive above this (m/s)")

    parser.add_argument("--pi", action="store_true", help="enable PI feedback (otherwise feedforward only)")
    parser.add_argument("--kp", type=float, default=0.10)
    parser.add_argument("--ki", type=float, default=0.40)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--gears", default="1,2,3,4,5,6", help="gears to run, comma separated; 'auto' leaves shifting to CARLA")
    parser.add_argument("--speeds", default="10,14,18,22", help="initial speeds to run, comma separated (m/s)")
    parser.add_argument("--allow-infeasible", action="store_true",
                        help="run cases whose reference the vehicle cannot reach (normally skipped)")
    parser.add_argument("--gears-per-figure", type=int, default=3,
                        help="split the grid into figures of this many gear rows")

    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--from-csv", default="", help="re-analyse and re-plot a saved run")
    parser.add_argument("--compare-csv", default="", help="overlay a previous run on the figure")
    args = parser.parse_args()

    if args.from_csv:
        rows = load_run(args.from_csv)
        report(rows, args)
        return

    t_ref, a_ref = build_profile(args)
    lut = LongitudinalLUT(args.lut)
    controller = LongitudinalAccelPI(lut, args.kp if args.pi else 0.0,
                                     args.ki if args.pi else 0.0, args.tau)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.map) if args.map else client.get_world()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if args.origin_index < 0:
        from find_straight_spawn import straight_length
        lengths = []
        for i, tf in enumerate(spawn_points):
            wp = carla_map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            lengths.append((straight_length(wp, 2.0, 800.0, 3.0) if wp else 0.0, i))
        best_len, origin_index = max(lengths)
        print(f"맵 {carla_map.name}, 스폰 {origin_index}번 (직선 {best_len:.0f}m)")
    else:
        origin_index = args.origin_index
    origin = spawn_points[origin_index]

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter(args.vehicle)[0]
    vehicle = world.spawn_actor(blueprint, origin)
    world.tick()

    auto_gear = args.gears.strip().lower() == "auto"
    gears = [None] if auto_gear else [int(g) for g in args.gears.split(",")]
    speeds = [float(s) for s in args.speeds.split(",")]

    print(f"프로파일 {args.profile}: 진폭 {args.amplitude} m/s^2, 주기 {args.period}s, "
          f"저크 제한 {args.jerk_max} m/s^3, 조합당 {args.duration}s (0에서 시작)")
    print(f"제어: {'피드포워드 + PI (kp=%.2f, ki=%.2f)' % (args.kp, args.ki) if args.pi else '피드포워드 단독'}")
    print(f"조합: 기어 {'자동' if auto_gear else gears} × 초기속도 {speeds} = "
          f"{len(gears)*len(speeds)}회 주행\n")

    from build_longitudinal_lut import CollisionWatch
    collision = CollisionWatch(world, vehicle)

    rows = []
    crashed = 0
    try:
        for gear in gears:
            for v0 in speeds:
                if gear is not None and not args.allow_infeasible:
                    skip = case_infeasible(lut, gear, v0, args)
                    if skip:
                        print(f"  건너뜀 gear={gear} v0={v0:.0f}: {skip}")
                        continue
                case, hit = run_case(world, vehicle, origin, controller, gear, v0, a_ref, args, collision)
                crashed += hit
                if not case:
                    continue
                rows.extend(case)
                err = np.array([r["a_fd"] for r in case]) - np.array([r["a_ref"] for r in case])
                print(f"  gear={gear if gear else 'auto'} v0={v0:5.1f} m/s → "
                      f"MAE {np.mean(np.abs(err)):.3f}  최대 {np.abs(err).max():.2f} m/s^2"
                      f"{'  [충돌로 중단]' if hit else ''}")
    finally:
        collision.destroy()
        vehicle.destroy()
        world.apply_settings(original_settings)
    if crashed:
        print(f"\n주의: {crashed}개 주행이 충돌로 중단되었습니다")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_gear", "case_v0", "t", "a_ref", "a_fd", "a_imu", "v", "u", "gear"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n결과 저장: {args.out}")

    report(rows, args)


def load_run(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {k: float(v) for k, v in r.items()}
            row["gear"] = int(row["gear"])
            rows.append(row)
    return rows


def smooth(x, n=5):
    """Moving average; the one-tick difference used for scoring is noisy."""
    if n <= 1:
        return x
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


def split_cases(rows):
    """Group rows by (gear, initial speed), preserving run order."""
    cases, order = {}, []
    for r in rows:
        key = (int(r["case_gear"]), float(r["case_v0"]))
        if key not in cases:
            cases[key] = []
            order.append(key)
        cases[key].append(r)
    return [(k, cases[k]) for k in order]


def case_metrics(case, dt):
    a_ref = np.array([r["a_ref"] for r in case])
    a_meas = smooth(np.array([r["a_fd"] for r in case]), 5)
    err = a_meas - a_ref
    lag, corr = estimate_lag(a_ref, a_meas, dt)
    aligned = float("nan")
    if lag > 0:
        shifted = a_meas[int(round(lag / dt)):]
        aligned = float(np.mean(np.abs(shifted - a_ref[:shifted.size])))
    u = np.array([r["u"] for r in case])
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max": float(np.abs(err).max()),
        "bias": float(np.mean(err)),
        "lag_ms": lag * 1000.0,
        "corr": corr,
        "aligned_mae": aligned,
        "rail_pct": 100.0 * float(np.mean(np.abs(u) >= 0.999)),
    }


def report(rows, args):
    cases = split_cases(rows)

    print("\n=== 조합별 추종 성능 ===")
    print(" 기어  초기속도 |  MAE   RMSE   최대   bias  | 지연(ms) 지연보정MAE | 포화%")
    for (gear, v0), case in cases:
        m = case_metrics(case, args.dt)
        aligned = f"{m['aligned_mae']:.3f}" if m["aligned_mae"] == m["aligned_mae"] else "  -  "
        print(f"  {gear if gear else 'auto':>3}  {v0:6.1f}   | {m['mae']:.3f}  {m['rmse']:.3f}  "
              f"{m['max']:5.2f}  {m['bias']:+.3f} |   {m['lag_ms']:4.0f}     {aligned}    | {m['rail_pct']:5.1f}")

    overall = case_metrics(rows, args.dt)
    print(f"\n=== 전체 ===")
    print(f"  MAE {overall['mae']:.3f}  RMSE {overall['rmse']:.3f}  최대 {overall['max']:.2f}  "
          f"bias {overall['bias']:+.3f} m/s^2  포화 {overall['rail_pct']:.1f}%")
    lags = [case_metrics(c, args.dt)["lag_ms"] for _, c in cases]
    print(f"  응답 지연: 중앙값 {np.median(lags):.0f} ms (범위 {min(lags):.0f}~{max(lags):.0f})")

    if not args.no_plot:
        from plot_accel_tracking import plot_tracking_grid
        compare = load_run(args.compare_csv) if args.compare_csv else None
        plot_tracking_grid(cases, args, compare=compare, show=not args.no_show)


if __name__ == "__main__":
    main()
