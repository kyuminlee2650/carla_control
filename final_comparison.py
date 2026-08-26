r"""Final comparison: every controller stack built in this repo -- "stanley", "mpc", "mpc-kf",
"mpc-kin", "vad-pid" -- driven on the same route and scored together, any subset in one command.

Unlike mpc_mpc_KF.py/mpc_mpc_kinematic.py (each a copy-plus-one-addition of the one before it, see
their own docstrings), this file does not hold its own copy of any controller. LateralMPC,
LateralMPCKinematic, MpcLongitudinal, VadPidController, curvature_preview/vx_preview_for_lateral,
and VX_EPS/VEHICLE_DEFAULTS are all IMPORTED from mpc_mpc_kinematic.py (the one existing file that
already builds mpc/mpc-kf/mpc-kin/vad-pid together); stanley_control()/front_axle_offset() are
imported from stanley_mpc.py. This script's whole purpose is comparing the controllers that already
exist elsewhere in the repo, so it drives the SAME class objects those files use rather than a
lookalike copy that could silently drift out of sync with the thing it's supposed to be scoring --
only run_trial()'s per-tick dispatch and main()'s orchestration below are original to this file; see
mpc_mpc_kinematic.py's own docstring for the "mpc"/"mpc-kf"/"mpc-kin"/"vad-pid" design itself (route
spline, longitudinal SpeedMPC, the dynamic and kinematic LateralMPC QPs, the VyKalmanFilter
closed-loop verification, wiring order).

"stanley" (the one controller not already wired together anywhere): the classic Stanley law
(stanley_control(), imported from stanley_mpc.py) paired with the SAME MpcLongitudinal every other
non-"vad-pid" entry here already uses -- i.e. this is stanley_mpc.py's "stanley+mpc" stack, not
"stanley+pid" (that longitudinal PID variant isn't imported at all; nothing here needs it). Unlike
"mpc"/"mpc-kf"/"mpc-kin", Stanley's control law is evaluated at the FRONT AXLE
(front_axle_offset() ahead of the vehicle origin along its own heading), not the ego/CG point --
that's the formulation itself, not a stylistic choice (see stanley_control()'s docstring). Only the
control law's own inputs (e_y_front/e_theta_front) use that front-axle projection; hist["e_y"] and
everything else logged/scored stays the ego-referenced lateral_error() every other controller here
already logs, so "e_y" means the same physical quantity for all five stacks in the comparison
figures -- see run_trial()'s "stanley" branch for exactly where the two projections split. "stanley"
therefore builds neither LateralMPC/LateralMPCKinematic nor VyKalmanFilter -- there is no QP and no
v_y/r state to estimate at all, steering is one closed-form arctan each tick.

"mpc"/"mpc-kf"/"mpc-kin"/"vad-pid" run through the imported classes exactly as mpc_mpc_kinematic.py
itself drives them, so this file can score Stanley against the dynamic-model stack, the
kinematic-model stack, the Kalman-filtered stack, and the real Bench2DriveZoo/VAD baseline all at
once -- any subset of "stanley"/"mpc"/"mpc-kf"/"mpc-kin"/"vad-pid" can be run in one command and land
in the same comparison figures (plot_comparison() already overlays an arbitrary number of runs --
COMPARE_COLORS has exactly 5 entries, one per controller here -- so nothing in viz_utils needed to
change for a 5-way comparison).

Usage (Ubuntu -- verified on this lab machine):
    # 1) the simulator, in its own terminal. This box's CARLA is NOT the ~/carla/CARLA_0.9.15 path
    #    hardcoded above/in $CARLA_ROOT (that one no longer exists here) -- functions.py finds the
    #    real tree by probing, see resolve_carla_root():
    #      /home/ailab/2026intern/carla/CarlaUE4.sh
    # 2) the controller, from this repo's own venv. The venv interpreter is spelled out on every
    #    line below on purpose, so any one of them can be copied and run on its own: the system
    #    python3 has neither carla nor osqp installed, so a bare `python3 final_comparison.py` fails
    #    unless the venv happens to be active. `source .venv/bin/activate` once and then using
    #    plain `python3` is equivalent.
    cd ~/carla_control
    # every controller in this repo, one run each, all five overlaid (the default --controller)
    .venv/bin/python final_comparison.py --profile constant --initial-speed 5 --save-plot
    # just Stanley alone (3 figures, same shape as mpc_mpc.py's single-controller output)
    .venv/bin/python final_comparison.py --profile constant --initial-speed 5 --save-plot --controller stanley
    # Stanley vs. the dynamic-model MPC: does the QP actually beat a closed-form law on this route?
    .venv/bin/python final_comparison.py --profile constant --initial-speed 5 --save-plot --controller stanley mpc
    # any subset, any order, with video
    .venv/bin/python final_comparison.py --profile step --initial-speed 10 --save-plot --record \
        --controller stanley mpc-kin vad-pid
    # emergency stop at t=10s, held 5s, then straight back up to --initial-speed (see --profile step)
    .venv/bin/python final_comparison.py --profile step --initial-speed 10 --step-time 10 --step-size -10 \
        --step-duration 5 --save-plot --controller stanley mpc mpc-kf mpc-kin vad-pid

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    # every controller in this repo, one run each, all five overlaid (the default --controller)
    .venv\Scripts\python.exe final_comparison.py --profile constant --initial-speed 5 --save-plot
    # just Stanley alone (3 figures, same shape as mpc_mpc.py's single-controller output)
    .venv\Scripts\python.exe final_comparison.py --profile constant --initial-speed 5 --save-plot --controller stanley
    # Stanley vs. the dynamic-model MPC: does the QP actually beat a closed-form law on this route?
    .venv\Scripts\python.exe final_comparison.py --profile constant --initial-speed 5 --save-plot --controller stanley mpc
    # any subset, any order, with video
    .venv\Scripts\python.exe final_comparison.py --profile step --initial-speed 10 --save-plot --record --controller stanley mpc-kin vad-pid
"""

import argparse
import math
import os
import shutil
import queue
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/carla/CARLA_0.9.15"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.append(os.path.join(HERE, "longitudinal_lookup"))

import carla

from functions import (AngleUnwrapper, ImuAcceleration, LowPassFilter, build_path,
                       build_path_spline, clipping, control_input, get_vehicle_geometry,
                       lateral_error, normalize_angle, speed_reference, spawn_at)
from kalman_filter.kalman_filter import VyKalmanFilter
from viz_utils import (VIEWS, VideoRecorder, follow_with_spectator, plot_comparison, plot_kf_run,
                       plot_results, print_error_summary, run_name, stack_videos_side_by_side)

# Fixed on purpose, same as stanley_mpc.py: the route (build_path()'s default origin/dest spawn
# indices) is a property of this specific map, not something to rediscover via CLI flags.
MAP_NAME = "Town10HD_Opt"

# --warm-start-speed injection: how many ticks to hold the gear by hand, and which gear to force
# for a given injected speed. set_target_velocity() writes the RIGID BODY's velocity and nothing
# else -- the gearbox stays where a standing start left it (neutral, gear 0) and the engine at idle
# -- so the car flies off at the injected speed with a drivetrain that knows nothing about it, and
# the automatic downshifts into that mismatch and drags it back. Measured at 14 m/s: 14.00 -> 9.34
# in one second regardless of throttle. Forcing a plausible gear for a few ticks removes it
# (14.00 -> 13.65), after which manual_gear_shift is released and the automatic takes over normally.
WARM_START_INJECT_TICKS = 4
WARM_START_INJECT_GEARS = ((6.0, 2), (10.0, 3), (14.0, 4), (1e9, 5))   # (speed below, gear)

# Warm-up is now a FIXED short settle, not a convergence gate. The gate existed to get a car that
# launched from rest up to --initial-speed over a straight run-up, and to wait until it had actually
# reached the route's start; --warm-start-speed does both at spawn instead. What is left is the part
# injection cannot fix -- the first ticks are not physical: ImuAcceleration discards its own first 3
# samples (the accelerometer reports around -378,000 m/s^2 right after a spawn), the suspension is
# still settling, and "mpc-kf"'s Kalman covariance starts at eye(2) and has to converge. Measured
# with injection at the route start, the old gate passed after 0.3 s; 1.0 s is that with margin.
# 0.25 s = 5 ticks. The floor is ImuAcceleration's own 3 discarded samples (0.15 s), during which
# a_x/a_y are pinned at 0 and must not reach the controller; the rest is margin for the suspension.
# Longer is actively worse, not safer: with the car injected at speed onto the route start, the
# curvature cap starts slowing it for the first corner immediately, so the wait does not "settle"
# at --initial-speed, it just eats route. Measured hand-off speed from a 10 m/s injection:
# 0.25 s -> 9.04, 0.5 s -> 8.94, 1.0 s -> 8.17, 2.0 s -> 6.71 m/s.
WARM_START_SETTLE_S = 0.25    # s -- 게이트를 보기 전 무조건 버리는 하한 (아래 (2))
# 고정 대기만으로는 점수 구간 t=0 이 물리적으로 정돈된 상태가 아니라는 것이 측정으로 드러나서,
# 하한 뒤에 수렴 게이트를 둔다. 측정된 두 가지:
#
# (1) 횡오차. 스폰이 경로 시작에서 0.2 m 떨어져 있고 0.25 s 로는 그게 흡수되지 않아, 점수 구간
#     첫 틱의 |e_y| 가 0.346 m 로 찍힌다 (t=0.45 s 면 0.011 m 로 사라지는 과도응답이다). 이 한
#     점이 cross-track 의 max|e| 를 그대로 결정해 버려서, Stanley 대비 비율이 1.16(짐) 으로
#     나오다가 앞 5 틱만 빼면 0.57(크게 이김) 로 뒤집힌다. 제어 품질이 아니라 출발 조건을
#     재고 있었다는 뜻이고, 'peak <= 0.15 m' 같은 절대 목표는 어떤 제어기로도 통과할 수 없었다.
#
# (2) 속도. set_target_velocity() 로 넣은 속도는 정착 구간에서 그냥 유지되지 않는다 -- 10 m/s
#     주입이 t=0 에 9.00, t=1.40 s 에 6.94 m/s 까지 내려갔다가 t=4 s 에 9.72 로 돌아온다.
#     이게 곡률 캡 때문이 아니라는 것도 확인했다: 첫 4 초 내내 v_des_curve 는 10.00 으로,
#     캡이 걸려 있지 않다. 즉 점수 구간의 앞 몇 초가 순수한 주입 회복 과도응답이었다.
#
# 그래서 "속도·가속도·횡오차가 모두 조용해질 때까지" 기다린다. 고정 시간을 늘리는 것과는 다르다
# (0.25 -> 1.0 s 로 늘리면 핸드오프 속도가 9.04 -> 8.17 로 더 나빠진다는 측정이 있다): 게이트는
# 조건이 만족되는 즉시 넘어가므로, 기다림이 '가라앉는 데 필요한 만큼'으로 끝난다.
WARM_START_HOLD_TICKS = 5    # 연속으로 이만큼 만족해야 인정 (dt=0.05 기준 0.25 s)
WARM_START_TIMEOUT = 30.0    # s -- 수렴하지 않을 때의 안전장치. 발동하면 그 사실을 찍는다.
                             # 15 였는데 공통 기점(WARM_START_START_S)까지 가는 데만 실측
                             # 11.95~13.35 s 가 걸려 여유가 없었다 -- 조금만 느린 주행이
                             # 타임아웃으로 빠지면 정확히 이 게이트가 막으려던 상황(과도응답
                             # 한복판에서 채점 시작)이 된다.

# 그리고 게이트만으로는 부족하다. 제어기마다 가라앉는 데 걸리는 시간이 달라서 -- 측정: Stanley
# 5.10 s (s=20.2 m), "mpc-kf" 10.25 s (s=64.0 m) -- 게이트만 쓰면 각자 다른 지점에서 채점을
# 시작한다. 그러면 한쪽은 앞 64 m 의 코너를 통째로 건너뛴 채로 비교되므로, 같은 경로를 달렸다는
# 전제가 깨진다 (측정된 채점 길이도 685 틱 대 590 틱으로 벌어졌다). 그래서 "가라앉았고 AND
# 공통 기점을 지났을 때" 채점을 시작한다 -- 모든 제어기가 정확히 같은 구간을 받는다.
# 값은 관측된 최장 수렴 거리(64 m)에 여유를 얹은 것. 327 m 경로의 앞 21% 를 버리는 셈인데,
# 남는 257 m 에 이 경로의 코너가 전부 들어 있다.
WARM_START_START_S = 5.0     # m -- 모든 제어기가 채점을 시작하는 공통 경로 위치
# 80 이었는데 과했다. 그 값은 이 기점의 목적(주입 과도응답을 채점 밖으로 내보내기)이 아니라,
# 중간에 시도했다가 걷어낸 수렴 게이트의 최장 관측 거리(64 m)에 여유를 얹은 것이라 근거가
# 남아 있지 않았다. 327 m 경로에서 82 m(25%)를 버려 채점이 733틱 36.6 s -> 553틱 27.6 s 로
# 줄어 있었다. 실제로 필요한 거리는 훨씬 짧다 -- 10 m/s 실측으로 t=1.4 s 에 v_x 가 6.94 까지
# 가라앉았다가 t=4.0 s(약 35 m)에 9.72 로 돌아오고, |e_y| 는 그 시점 0.001 m 다.
# 30 m 는 그 회복 구간의 대부분을 지난 지점이고, 10 m/s 기준으로 정했다 (사용자 결정:
# 15 m/s 는 이번 튜닝 대상이 아니다 -- 그 속도에서는 같은 시간에 더 멀리 가므로 30 m 가
# 과도응답 끝보다 이를 수 있다는 점만 알고 있으면 된다).


# Every controller "object" this file drives -- the LateralMPC/LateralMPCKinematic QPs, the
# MpcLongitudinal/LookupController/SpeedMPC longitudinal stack, the VadPidController adapter, and
# the curvature-preview helpers they share -- is IMPORTED from mpc_mpc_kinematic.py, the one file
# that already builds mpc/mpc-kf/mpc-kin/vad-pid, rather than copied a sixth time: this script's
# whole point is comparing the ACTUAL controllers that exist elsewhere in the repo, and a copy can
# silently drift out of sync with the original it's supposed to be scoring (mpc_mpc_kinematic.py
# itself is a copy of mpc_mpc_KF.py's copy of mpc_mpc_comparison.py's originals for exactly this
# reason -- see its own docstring -- but that chain stops here: importing means there is now
# exactly one LateralMPC, one MpcLongitudinal, one VadPidController in the whole repo, not six).
# Same reasoning for Stanley: stanley_control()/front_axle_offset() come from stanley_mpc.py.
from mpc_mpc_kinematic import (VX_EPS, VEHICLE_DEFAULTS, LateralMPC, LateralMPCKinematic,
                               MpcLongitudinal, VadPidController, curvature_preview,
                               vx_preview_for_lateral)
from stanley_mpc import front_axle_offset, stanley_control


# ----------------------------------------------------------------------------- one trial

def run_trial(world, spawn_transform, path_x, path_y, path, blueprint, imu_bp, controller,
             controller_key, args, spawn_to_start_m=0.0, video_suffix=""):
    """Spawn one vehicle, drive it under `controller`, tear it down.

    spawn_transform is where the vehicle actually spawns -- with --spawn-x/-y this sits well before
    path's own s=0 (see functions.spawn_at()), NOT the route's own start; path/path_x/path_y are
    untouched either way. last_s starts at 0.0 below regardless: path.project() clips to the nearest
    in-domain station until the vehicle physically reaches the route's start, so a spawn point behind
    the route just means the first several ticks project onto s=0 (near-zero e_y, since --spawn-x/-y
    is meant to sit on the same straight road) rather than requiring the path itself to reach back to
    where the car spawned.

    spawn_to_start_m: main()'s own straight-line distance from spawn_transform to the route's actual
    start (origin_transform.location). Reported at startup so a spawn point that is not actually on
    the route's own start is visible; with --warm-start-speed the run no longer needs a run-up. Also
    v_x/a_x, to gate when logging starts -- see the warm-up gate note below.

    controller_key: "mpc" and "mpc-kf" both run the identical MpcLongitudinal (longitudinal) +
    LateralMPC (lateral) pair -- they differ in exactly one thing, what feeds x0's v_y slot each
    tick. "mpc" uses CARLA's own ground truth (vehicle.get_velocity(), body frame); "mpc-kf" runs a
    VyKalmanFilter (constructed below, per-trial like lateral_mpc) and uses its v_y_hat instead --
    see the module docstring's "Step 5" for the causality/noise details. "mpc-kin" runs the same
    MpcLongitudinal against LateralMPCKinematic instead. "stanley" also runs the same
    MpcLongitudinal, but pairs it with the closed-form Stanley law (stanley_control()) instead of
    any QP -- no lateral_mpc/kf object is built for it at all, see its own branch below for why its
    control-law inputs are a separate front-axle projection rather than the ego-referenced one every
    other branch logs. "vad-pid" takes none of these: it runs `controller` (a VadPidController) as
    one combined lateral+longitudinal call, so neither LateralMPC/LateralMPCKinematic nor the filter
    nor Stanley is built for it at all.

    video_suffix: appended to the recorded filename (run_name()'s own suffix mechanism) so two
    controllers recorded in the same process (--controller a b --record) don't overwrite each
    other's mp4 -- main() passes the controller key here when 2+ controllers are selected, "" (no
    change) for a single one.

    Same warm-up gate as stanley_mpc.py/longitudinal_mpc.py in spirit -- launch from rest under the
    real longitudinal controller and hold off on logging until v_x/a_x have actually settled near the
    profile's own t=0 value (--initial-speed) -- but ANDed with one more condition: the vehicle must
    also have physically covered spawn_to_start_m, i.e. actually reached the route's own s=0, not
    just gotten close to --initial-speed somewhere on the spawn-to-route-start stretch. Without that,
    logging could start (and the flat --v_des_log profile with it) before the car has rejoined the
    scored route at all. Steering (lateral or combined) runs from tick one regardless -- only the
    *logging* start is gated.
    """
    vehicle = world.spawn_actor(blueprint, spawn_transform)
    physics = vehicle.get_physics_control()
    wheelbase, lf, lr, max_steer = get_vehicle_geometry(vehicle, spawn_transform)

    front_offset = None
    if controller_key == "stanley":
        front_offset = front_axle_offset(vehicle, lf)
        if not 0.0 < front_offset < wheelbase:
            raise RuntimeError(f"front_offset={front_offset:.2f} m is not inside the wheelbase "
                               f"({wheelbase:.2f} m); the front-axle reference point is wrong.")

    # delta_max capped well under max_steer (the wheel's own physical limit, ~70 deg): Cf/Cr were
    # calibrated over an ~8-16 deg range (estimate_cornering_stiffness.py), so the linear tire model
    # this QP is built on stops being valid long before 70 deg -- and separately, max_steer is the
    # INNER wheel's limit (Ackermann), not the bicycle model's single virtual wheel, which needs a
    # smaller angle than the inner wheel for the same turn. --delta-max-deg picks a value inside
    # both limits rather than deriving the (still oversized, relative to the tire model) Ackermann
    # bound.
    lateral_mpc = None
    if controller_key in ("mpc", "mpc-kf"):
        lateral_mpc = LateralMPC(
            dt=args.dt, n_p=args.lat_n_p, n_c=args.lat_n_c,
            mass=args.mass, Iz=args.iz, lf=lf, lr=lr, Cf=args.cf, Cr=args.cr,
            w_ey=args.w_ey, w_epsi=args.w_epsi, w_ay=args.w_ay, w_r=args.w_r, w_rdot=args.w_rdot,
            w_delta=args.w_delta, w_ddelta=args.w_ddelta,
            delta_max=math.radians(args.delta_max_deg), ddelta_max=math.radians(args.ddelta_max_deg) * args.dt)
    elif controller_key == "mpc-kin":
        # L = lf+lr only -- the kinematic model has no Cf/Cr/mass/Iz at all, see LateralMPCKinematic's
        # docstring. delta_max/ddelta_max are shared with the dynamic controllers (actuator limits,
        # not cost weights); the --kin-w-* weights are its own set, NOT --w-ey/--w-epsi/--w-r/
        # --w-delta/--w-ddelta -- this QP has a different structure (no output layer) and w_r means a
        # physically different thing here (see LateralMPCKinematic's docstring and --kin-w-r's help).
        lateral_mpc = LateralMPCKinematic(
            dt=args.dt,
            n_p=args.kin_n_p if args.kin_n_p is not None else args.lat_n_p,
            n_c=args.kin_n_c if args.kin_n_c is not None else args.lat_n_c, L=lf + lr,
            w_ey=args.kin_w_ey, w_epsi=args.kin_w_epsi, w_r=args.kin_w_r,
            w_delta=args.kin_w_delta, w_ddelta=args.kin_w_ddelta,
            delta_max=math.radians(args.delta_max_deg), ddelta_max=math.radians(args.ddelta_max_deg) * args.dt)

    # v_y estimator for "mpc-kf" -- one filter per trial, same reason lateral_mpc above is built
    # fresh per trial rather than shared. prev_delta_for_kf carries the steering angle actually
    # applied last tick into this tick's predict()/update() (see module docstring on causality);
    # kf_rng draws the synthetic gyro/accel noise this filter sees (real CARLA IMU noise is ~0).
    kf, kf_rng, prev_delta_for_kf = None, None, 0.0
    if controller_key == "mpc-kf":
        kf = VyKalmanFilter(
            dt=args.dt, mass=args.mass, Iz=args.iz, lf=lf, lr=lr, Cf=args.cf, Cr=args.cr,
            Q=np.diag([args.kf_q_vy, args.kf_q_r]),
            R=np.diag([args.kf_r_dpsi if args.kf_r_dpsi is not None else math.radians(args.kf_gyro_std) ** 2,
                      args.kf_r_ay if args.kf_r_ay is not None else args.kf_accel_std ** 2]),
            vx_floor=VX_EPS, x0=[0.0, 0.0], P0=np.eye(2))
        kf_rng = np.random.default_rng(args.kf_seed)

    # Stanley's previous applied delta, for the hard steer-rate clamp at the bottom of the loop.
    # NOTE this is a deliberate divergence from stanley_mpc.py, whose own run_trial() low-passes the
    # steer command (tau=0.1s) instead: this script's job is a controller comparison, so every
    # controller here is held to the same --ddelta-max-deg the MPC variants' QPs enforce, rather
    # than each keeping its own native smoothing. Read Stanley's numbers here as "Stanley under the
    # MPCs' actuator limit", not as a reproduction of stanley_mpc.py.
    prev_delta_stanley = 0.0

    accel = ImuAcceleration(dt=args.dt)
    yaw_unwrapper = AngleUnwrapper()
    rh_unwrapper = AngleUnwrapper()
    last_s = 0.0

    # matches plot_results()'s expectations (viz_utils.plot_lateral/plot_longitudinal/plot_trajectory).
    # v_y_hat/dpsi_noisy/ay_noisy only ever get appended to for "mpc-kf" below -- stay empty for
    # "mpc"/"vad-pid", which viz_utils._get() treats the same as "never recorded" (see plot_lateral's
    # ref_key="v_y_hat", print_error_summary's "v_y estimate" row, and main()'s extra
    # plot_kf_run() figure for the "mpc-kf" trial only).
    hist = {"t": [], "x": [], "y": [], "v_x": [], "v_y": [], "v_y_hat": [], "dpsi_noisy": [],
            "ay_noisy": [], "v_des": [], "v_des_curve": [], "a_x": [], "a_x_raw": [], "a_y_raw": [],
            "a_y": [], "yaw_rate": [], "steer_deg": [],
            "throttle": [], "brake": [], "e_y": [], "yaw": [], "path_yaw": [], "e_theta": [],
            "a_cmd": []}
    warmed_up = False
    log_start_i = 0
    hold_ticks = 0      # 웜업 게이트를 연속으로 만족한 틱 수 (WARM_START_HOLD_TICKS 참고)
    imu = None
    recorder = None
    video_meta = None   # 녹화했을 때만 채워진다 (run_trial 의 반환값 2번째)
    try:
        world.tick()

        # Optional rolling start. From rest the car has to accelerate over spawn_to_start_m before
        # the warm-up gate can pass, and the straight is short. Injected AFTER the priming tick so
        # the suspension has taken a step (not mid-drop), with the angular velocity zeroed too --
        # setting only the linear part leaves whatever spin the spawn imparted. The gear is then
        # held by hand for a few ticks; see WARM_START_INJECT_GEARS for why that is not optional.
        if args.warm_start_speed > 0:
            v0 = args.warm_start_speed
            gear = next(g for lim, g in WARM_START_INJECT_GEARS if v0 < lim)
            fwd = spawn_transform.get_forward_vector()
            vehicle.set_target_velocity(carla.Vector3D(fwd.x * v0, fwd.y * v0, 0.0))
            vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            for _ in range(WARM_START_INJECT_TICKS):
                hold = carla.VehicleControl()
                hold.throttle, hold.steer = 0.5, 0.0
                hold.manual_gear_shift, hold.gear = True, gear
                vehicle.apply_control(hold)
                world.tick()
            release = carla.VehicleControl()
            release.throttle, release.steer = 0.5, 0.0
            release.manual_gear_shift = False
            vehicle.apply_control(release)
            world.tick()
            print(f"  rolling start: injected {v0:.1f} m/s in gear {gear}")

        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)

        if args.record:
            if args.record == "auto":
                video_path = os.path.join(args.video_dir, run_name(video_suffix) + ".mp4")
            elif video_suffix:
                base, ext = os.path.splitext(args.record)
                video_path = f"{base}_{video_suffix}{ext}"
            else:
                video_path = args.record
            rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
            recorder = VideoRecorder(world, vehicle, video_path, fps=1.0 / args.dt,
                                     width=rec_w, height=rec_h, view=args.record_view)

        steps = int((args.max_duration + WARM_START_TIMEOUT) / args.dt)
        for i in range(steps):
            step_start = time.time()
            # 녹화 중이면 인코더가 밀린 만큼 여기서 기다린다 (프레임 유실 -> 영상 끊김 방지).
            # 카메라는 tick 당 한 장을 내므로 다음 tick 을 미루는 것이 곧 역압이다.
            if recorder is not None:
                recorder.throttle()
            world.tick()
            # 10s, not 2s: right after a fresh client.load_world(), the server is still streaming
            # map assets/compiling shaders, so the first several ticks can take much longer than a
            # steady-state ~dt-paced tick -- a real hang still raises Empty, just with more margin.
            imu_data = imu_queue.get(timeout=10.0)

            transform = vehicle.get_transform()
            yaw = yaw_unwrapper.step(math.radians(transform.rotation.yaw))
            ego_x, ego_y = transform.location.x, transform.location.y
            vel_vec = vehicle.get_velocity()
            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)
            v_y = -vel_vec.x * math.sin(yaw) + vel_vec.y * math.cos(yaw)
            r = imu_data.gyroscope.z
            yaw_rate_deg = math.degrees(r)

            accel.step(imu_data)
            a_x, a_x_raw, a_y = accel.a_x, accel.a_x_raw, accel.a_y
            a_y_raw = accel.a_y_raw


            t = (i - log_start_i) * args.dt
            v_ref = args.initial_speed if not warmed_up else speed_reference(args, t)
            ctx = SimpleNamespace(t=t, v_x=v_x, v_ref=v_ref, a_x=a_x, a_x_raw=a_x_raw,
                                  gear=vehicle.get_control().gear, warmed_up=warmed_up,
                                  path=path, last_s=last_s, ego_x=ego_x, ego_y=ego_y, yaw=yaw)

            if controller_key in ("mpc", "mpc-kf", "mpc-kin"):
                # v_y source for x0, computed BEFORE x0 itself: "mpc-kf" needs its filter's predict()+
                # update() to have already run this tick (using vx measured just above and
                # prev_delta_for_kf -- last tick's delta, since this tick's doesn't exist until
                # lateral_mpc.solve() below computes it) -- see module docstring's Step 5. x0's r stays
                # the real gyro reading either way, only v_y is swapped. "mpc-kin" needs neither: its
                # model has no v_y/r state at all (see LateralMPCKinematic's docstring), so this whole
                # if/else is skipped for it.
                if controller_key == "mpc-kf":
                    # KNOWN INCONSISTENCY, kept deliberately -- read this before quoting "mpc-kf"
                    # numbers as evidence for the filter.
                    #
                    # The synthetic noise below goes ONLY to the Kalman filter. x0 further down
                    # still takes the clean `r`, not r_meas, even though both stand for the same
                    # gyro channel: the filter is handed a sensor this run is pretending is noisy,
                    # while the controller is handed one that is not. On a real car there is no
                    # second, clean source for r -- it comes off the same gyro -- so this split does
                    # not correspond to any physical setup.
                    #
                    # What it costs: it flatters "mpc-kf". Part of any advantage it shows over a
                    # controller fed a genuinely noisy r is just the clean r, not the estimator. It
                    # also skews the "mpc" (ground-truth v_y) comparison, since that one gets clean
                    # v_y AND clean r while this gets a filtered v_y and clean r -- the two differ
                    # by which signal was noised, not only by how v_y was obtained.
                    #
                    # The two honest alternatives, if this ever needs to be a claim about the
                    # filter rather than a tuning harness: feed x0 r_meas (the raw noisy gyro), or
                    # feed it kf.x[1] (the filter's own yaw-rate estimate -- VyKalmanFilter's state
                    # is [v_y, r], so that value already exists and is currently discarded).
                    r_meas = r + kf_rng.normal(0.0, math.radians(args.kf_gyro_std))
                    ay_meas = a_y + kf_rng.normal(0.0, args.kf_accel_std)
                    kf.step(v_x, prev_delta_for_kf, [r_meas, ay_meas])
                    v_y_for_x0 = kf.v_y
                elif controller_key == "mpc":
                    v_y_for_x0 = v_y   # "mpc" -- ground truth

                # longitudinal (MPC) runs first so ctx.v_x_preview is ready for the lateral schedule
                # below (see the "which speed plan" discussion in mpc_mpc_comparison.py's module
                # docstring)
                u = controller.step(ctx)

                last_s, raw_e_y = lateral_error(ego_x, ego_y, path, last_s)
                yaw_s = float(path.yaw(last_s))
                road_heading = rh_unwrapper.step(yaw_s)
                e_theta = normalize_angle(yaw_s - yaw)
                vx_preview = vx_preview_for_lateral(ctx, lateral_mpc.n_p)
                kappa_preview = curvature_preview(path, last_s, vx_preview, args.dt)
                # `r` here is the CLEAN gyro even for "mpc-kf", whose filter was fed a noised copy
                # of this same channel a few lines up -- see the note there for why that is not a
                # physical setup and what it does to the numbers.
                x0 = [raw_e_y, -e_theta] if controller_key == "mpc-kin" else [v_y_for_x0, r, raw_e_y, -e_theta]
                delta = lateral_mpc.solve(x0, vx_preview, kappa_preview)
                prev_delta_for_kf = delta   # this tick's delta becomes next tick's "already applied"
                                            # (a no-op outside "mpc-kf", harmless either way)

                control = control_input(u, delta, v_x, vehicle, physics)
                steer_deg = math.degrees(delta)
                throttle_log, brake_log = control.throttle, control.brake
                a_cmd_log, v_des_log = ctx.a_cmd, ctx.v_ref
                v_des_curve_log = ctx.v_ref_curve
                reset_arg = u
            elif controller_key == "stanley":
                # longitudinal (MpcLongitudinal, same as "mpc"/"mpc-kf"/"mpc-kin") first, same order
                # as that branch, though nothing here actually depends on it -- kept consistent.
                u = controller.step(ctx)

                # ego-referenced projection, logged/scored exactly like every other controller here
                # (see module docstring) -- NOT what Stanley's own law uses below.
                last_s, raw_e_y = lateral_error(ego_x, ego_y, path, last_s)
                yaw_s = float(path.yaw(last_s))
                road_heading = rh_unwrapper.step(yaw_s)
                e_theta = normalize_angle(yaw_s - yaw)

                # Stanley's own control inputs: a SEPARATE projection at the front axle, seeded from
                # the ego station just computed (project() does a local search either way, so the
                # seed only affects search cost, not the result) -- this is the classic Stanley
                # formulation (see stanley_control()'s docstring), and deliberately does not touch
                # last_s/raw_e_y/e_theta above, which stay the ego-referenced scoring numbers.
                front_x = ego_x + front_offset * math.cos(yaw)
                front_y = ego_y + front_offset * math.sin(yaw)
                s_front, e_y_front = lateral_error(front_x, front_y, path, last_s)
                e_theta_front = normalize_angle(float(path.yaw(s_front)) - yaw)
                # lateral_error()'s e_y is vehicle-minus-path; stanley_control()'s atan2(k*e_y, ...)
                # expects path-minus-vehicle to steer the right way, hence the flip (see
                # stanley_mpc.py's run_trial(), same flip).
                delta = stanley_control(v_x, -e_y_front, e_theta_front)
                # stanley_control()'s raw atan2 term is unbounded (approaches +-90 deg as e_y grows)
                # and stanley_mpc.py never applies it unclipped -- its own run_trial() clips to
                # +-(3/7)*max_steer = +-30 deg before driving the wheel. Reuse --delta-max-deg here
                # (30 deg by default, exactly that same bound) rather than hardcode 3/7 again: it is
                # both the actuator limit and the range Cf/Cr were calibrated over (see the flag's
                # own help), the same reason "mpc"/"mpc-kf"/"mpc-kin" clip their own QP output to it
                # inside LateralMPC/LateralMPCKinematic.solve(). Without this, a large e_y_front
                # commands a wheel angle CARLA's Ackermann conversion just saturates to full lock at
                # (control_input()'s cot_inner <= 0 branch) -- exactly the "stanley used to perform
                # better" regression this fixes.
                delta_max = math.radians(args.delta_max_deg)
                delta = clipping(delta, delta_max, -delta_max)
                # Same hard steer-rate limit every other controller gets, always on -- it used to
                # sit behind a STANLEY_HARD_RATE env var that defaulted OFF, which meant the
                # comparison this script exists to make was run with Stanley on a tau=0.1s low-pass
                # while "mpc"/"mpc-kf"/"mpc-kin" were on --ddelta-max-deg. Two different rate
                # treatments is not a controller comparison, so the clamp is now unconditional and
                # the LPF is gone with it (stacking both would hand Stanley an extra smoothing stage
                # none of the others have).
                #
                # NOT equivalent to what the MPCs get, and worth remembering when reading results:
                # theirs is a CONSTRAINT inside the QP (-ddelta_max <= GU-Uprev <= ddelta_max), so
                # the optimizer plans the whole horizon knowing the limit; this is a post-hoc clamp
                # on an already-computed delta, which just truncates whatever Stanley asked for.
                # Same bound, different authority over the solution.
                ddelta_max = math.radians(args.ddelta_max_deg) * args.dt
                delta = clipping(delta, prev_delta_stanley + ddelta_max,
                                 prev_delta_stanley - ddelta_max)
                prev_delta_stanley = delta

                control = control_input(u, delta, v_x, vehicle, physics)
                steer_deg = math.degrees(delta)
                throttle_log, brake_log = control.throttle, control.brake
                a_cmd_log, v_des_log = ctx.a_cmd, ctx.v_ref
                v_des_curve_log = ctx.v_ref_curve
                reset_arg = u
            else:   # "vad-pid" -- single combined lateral+longitudinal call, see VadPidController
                last_s, raw_e_y = lateral_error(ego_x, ego_y, path, last_s)
                yaw_s = float(path.yaw(last_s))
                road_heading = rh_unwrapper.step(yaw_s)
                e_theta = normalize_angle(yaw_s - yaw)

                steer, throttle, brake = controller.step(ctx)
                vehicle.apply_control(carla.VehicleControl(
                    steer=steer, throttle=throttle, brake=1.0 if brake else 0.0))
                steer_deg = steer * math.degrees(max_steer)
                throttle_log, brake_log = throttle, (1.0 if brake else 0.0)
                a_cmd_log = float("nan")   # no scalar accel command in this controller -- see hist schema
                v_des_log = ctx.v_ref      # flat/un-refined -- the baseline gets no curvature speed cap
                v_des_curve_log = float("nan")   # 곡률 보정 자체가 없는 제어기
                reset_arg = None

            follow_with_spectator(world, vehicle)

            if not warmed_up:
                # 하한 안에서는 게이트를 보지도 않는다: ImuAcceleration 이 앞 3 샘플을 버리는
                # 동안 a_x 를 0 으로 고정해서 내놓기 때문에, 그 구간을 물리면 |a_x| = 0 이
                # 조건을 즉시 통과해 주입 직후 첫 틱에 핸드오프된다.
                past_floor = i * args.dt >= args.warm_start_settle
                # 판정은 경로 위치 하나로 한다. v_x/a_x/e_y 가 "가라앉을 때까지" 기다리는
                # 조건도 달아 봤지만 15 m/s 에서 무너졌다: 그 속도에서는 곡률 캡이 코너마다
                # 목표 속도를 바꿔서 차가 늘 가속 아니면 감속 중이고, |a_x| < 0.3 이 연속으로
                # 만족되는 순간이 사실상 없다. 게이트가 경로 끝에 가서야 열려서 채점 구간이
                # 64 틱(3.2 s)까지 쪼그라들고 Comfortness 가 0.0000 으로 나왔다 -- 가라앉지
                # 않는 대상에게 가라앉기를 요구한 셈이다. 공통 기점은 그런 실패 모드가 없다:
                # 결정론적이고, 모든 제어기와 모든 속도에서 정확히 같은 구간을 준다. 기점까지
                # 80 m 를 달리는 동안 주입 과도응답은 이미 사라져 있다 (10 m/s 에서 t=4 s,
                # 약 35 m 면 v_x 가 9.72 로 회복되고 |e_y| 는 0.001 m 다).
                settled = past_floor and last_s >= WARM_START_START_S
                hold_ticks = hold_ticks + 1 if settled else 0
                converged = hold_ticks >= WARM_START_HOLD_TICKS
                timed_out = i * args.dt >= WARM_START_TIMEOUT
                if not (converged or timed_out):
                    elapsed = time.time() - step_start
                    if elapsed < args.dt / args.times_run:
                        time.sleep(args.dt / args.times_run - elapsed)
                    continue
                warmed_up = True
                log_start_i = i
                controller.reset(reset_arg)
                status = (f"converged after {i * args.dt:.2f}s" if converged
                          else f"TIMED OUT after {WARM_START_TIMEOUT:.0f}s (not settled -- the "
                               f"scored window starts mid-transient)")
                print(f"Warm-start {status}: v_x={v_x:.2f} m/s, a_x={a_x:.2f} m/s^2, "
                      f"e_y={raw_e_y:+.3f} m, s={last_s:.1f} m -- logging starts now.")

            t = (i - log_start_i) * args.dt
            hist["t"].append(t)
            hist["x"].append(ego_x)
            hist["y"].append(ego_y)
            hist["v_x"].append(v_x)
            hist["v_y"].append(v_y)
            if controller_key == "mpc-kf":
                hist["v_y_hat"].append(v_y_for_x0)
                hist["dpsi_noisy"].append(math.degrees(r_meas))   # same unit as hist["yaw_rate"]
                hist["ay_noisy"].append(ay_meas)                  # same unit as hist["a_y"]
            hist["v_des"].append(v_des_log)
            # 곡률 보정된 목표속도는 따로 보관: "v_des"(그래프/속도 RMSE 기준)는 이제 --profile 이
            # 준 목표속도 그 자체이고, refine_speed_preview()가 커브 앞에서 깎아낸 값은 진단용으로만
            # 남는다. "mpc"/"mpc-kf" 만 이 값을 가진다 ("vad-pid" 는 애초에 곡률 보정을 안 받음).
            hist["v_des_curve"].append(v_des_curve_log)
            hist["a_x"].append(a_x)
            # 필터 통과 전 원시 IMU 값 -- 공식 B2D comfortness 가 자기 Savitzky-Golay 를 직접
            # 걸기 때문에, 여기서 이미 저역통과된 a_x/a_y 를 넘기면 이중 평활이 되어 점수가
            # 실제보다 좋게 나온다 (viz_utils.b2d_comfortness 참고).
            hist["a_x_raw"].append(a_x_raw)
            hist["a_y_raw"].append(a_y_raw)
            hist["a_y"].append(a_y)
            hist["yaw_rate"].append(yaw_rate_deg)
            hist["steer_deg"].append(steer_deg)
            hist["throttle"].append(throttle_log)
            hist["brake"].append(brake_log)
            hist["e_y"].append(raw_e_y)
            hist["yaw"].append(math.degrees(yaw))
            hist["path_yaw"].append(math.degrees(road_heading))
            hist["e_theta"].append(math.degrees(e_theta))
            hist["a_cmd"].append(a_cmd_log)

            if i % 20 == 0:
                print(f"t={t:5.1f}s  v_x={v_x:5.2f}/{v_ref:.2f} m/s  steer={steer_deg:+.2f} deg  "
                      f"e_y={raw_e_y:+.2f} m  s={last_s:6.1f}/{path.s_max:.1f} m")

            if last_s >= path.s_max - 0.1:
                print(f"Reached end of path (s={last_s:.1f}/{path.s_max:.1f} m).")
                break
            if t >= args.max_duration:
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    finally:
        if recorder is not None:
            recorder.close()  # before vehicle.destroy(): the camera is attached to it
            # 합치기(viz_utils.stack_videos_side_by_side)에 필요한 정보. frames 는 실제로 쓰인
            # 프레임 수 -- 두 주행의 길이가 다를 때 짧은 쪽을 얼마나 늘릴지 계산하는 데 쓴다.
            video_meta = {"path": recorder.out_path, "frames": recorder.frames}
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        vehicle.destroy()

    return hist, video_meta


def _debug_steer_saturation(label, hist, args):
    """TEMPORARY diagnostic (STEER_SAT_DEBUG=1) -- how often steer_deg sits at/near delta_max, and
    how often its tick-to-tick change sits at/near ddelta_max*dt, to check whether the MPC variants'
    hard actuator constraints are binding more than Stanley's own (softer) clip+filter."""
    steer = np.asarray(hist["steer_deg"], dtype=float)
    delta_max_deg = args.delta_max_deg
    ddelta_max_deg_per_tick = args.ddelta_max_deg * args.dt
    rate = np.diff(steer)
    near_mag = np.sum(np.abs(steer) > 0.9 * delta_max_deg)
    at_mag = np.sum(np.abs(steer) > 0.99 * delta_max_deg)
    near_rate = np.sum(np.abs(rate) > 0.9 * ddelta_max_deg_per_tick)
    at_rate = np.sum(np.abs(rate) > 0.99 * ddelta_max_deg_per_tick)
    print(f"  [sat-debug] {label}: n={len(steer)} steer_deg max|.|={np.max(np.abs(steer)):.2f} "
         f"(limit {delta_max_deg:.1f}) -- >90% of limit: {near_mag} ticks, >99%: {at_mag} ticks | "
         f"rate max|.|={np.max(np.abs(rate)):.2f} deg/tick (limit {ddelta_max_deg_per_tick:.2f}) "
         f"-- >90% of limit: {near_rate} ticks, >99%: {at_rate} ticks")


STANLEY_CACHE_DIR = os.path.join(HERE, "run_cache")


def _stanley_cache_path(args):
    """One file per speed -- a Stanley run is only a fair reference for another run at the same
    --initial-speed, so the speed is part of the name rather than something to check later."""
    return os.path.join(STANLEY_CACHE_DIR, f"Stanley_{args.initial_speed:g}ms.npz")


def load_stanley(args):
    """(hist, video_frames) from the cached Stanley run at this speed, or (None, 0)."""
    path = _stanley_cache_path(args)
    if not os.path.exists(path):
        print(f"  ! --stanley-cached: {path} 없음 -- 새로 주행합니다")
        return None, 0
    d = np.load(path, allow_pickle=False)
    hist = {k: d[k].tolist() for k in d.files if not k.startswith("_")}
    frames = next((int(d[k]) for k in ("_frames", "_video_frames") if k in d.files), 0)
    print(f"  [stanley-cache] loaded {len(hist['t'])} ticks <- {path}")
    return hist, frames


def main():
    parser = argparse.ArgumentParser()

    # ---- simulation setting ---- #
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--times-run", type=float, default=5.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=100.0, help="scored run length (s)")
    parser.add_argument("--spawn-x", type=float, default=-90,
                        help="m -- vehicle spawns at the road waypoint nearest this raw map (x, y) "
                             "(see functions.spawn_at), NOT the route's own start; the scored route "
                             "itself (path_x/path_y, from build_path()'s own origin/dest indices) is "
                             "untouched. Default (-64.8, 24.5) IS this map's route start: with "
                             "--warm-start-speed injecting the speed at spawn, there is nothing to "
                             "ramp up over, so the run no longer starts on a run-up before the "
                             "route. It used to default to (-90, 25) -- the same point backed up "
                             "25 m along the straight -- purely to give a standing start room to "
                             "reach --initial-speed. Back it up again if you launch from rest "
                             "(--warm-start-speed 0).")
    parser.add_argument("--spawn-y", type=float, default=24.5, help="m -- see --spawn-x")

    # ---- speed profile ---- #
    parser.add_argument("--profile", default="constant", choices=("constant", "sine", "step"),
                        help="speed reference shape: flat initial-speed, a sine wave around it, or "
                             "a step change at --step-time held for --step-duration and then "
                             "released back to --initial-speed. The step window is what makes an "
                             "emergency-stop-and-restart run: --step-size -<initial speed> "
                             "--step-duration <seconds> brakes to a standstill and then demands the "
                             "original speed again in one step (see functions.speed_reference)")
    parser.add_argument("--initial-speed", type=float, default=10.0,
                        help="m/s -- kept modest by default: the route's sharpest corners "
                             "(~10-13m radius, idx~90-150) demand a_y=v^2/r that outgrows what "
                             "steering alone can correct for well above this speed (see tuning "
                             "notes above LateralMPC's argparse group)")
    parser.add_argument("--sine-amplitude", type=float, default=3.0, help="sine profile peak deviation (m/s)")
    parser.add_argument("--sine-period", type=float, default=5.0, help="sine profile period (s)")
    parser.add_argument("--step-size", type=float, default=-15,
                        help="step profile speed change (m/s), signed. Negative decelerates; the "
                             "result is clamped at 0, so anything <= -(initial speed) is a full stop")
    parser.add_argument("--step-time", type=float, default=10.0,
                        help="step profile: when it happens (s). Both step edges land here in the "
                             "REFERENCE, not in the response: the longitudinal MPC sees them "
                             "--np*--dt (2s by default) early through its own preview and starts "
                             "braking/accelerating before the edge, while 'vad-pid' -- which has no "
                             "preview at all -- only reacts once the edge has passed. That gap is a "
                             "real difference between the two stacks, not a profile artifact")
    parser.add_argument("--step-duration", type=float, default=5,
                        help="step profile: how long the stepped speed is held (s) before the "
                             "reference returns to --initial-speed. Default (unset) holds it for "
                             "the rest of the run, the permanent step this profile used to be. "
                             "Both edges are steps, so a stop window here is a hard decel followed "
                             "by a hard re-accel -- how hard each one actually gets is bounded by "
                             "--a-min/--a-max (the longitudinal MPC), not by this profile")

    # ---- longitudinal MPC ---- #
    mpc = parser.add_argument_group("longitudinal MPC")
    mpc.add_argument("--np", dest="n_p", type=int, default=40, help="prediction horizon (steps)")
    mpc.add_argument("--nc", dest="n_c", type=int, default=5, help="control horizon (steps, <= --np)")
    mpc.add_argument("--w-v", type=float, default=150.0, help="speed-tracking weight")
    mpc.add_argument("--w-a", type=float, default=15, help="commanded-acceleration magnitude weight")
    mpc.add_argument("--w-j", type=float, default=30,
                     help="commanded-acceleration rate (jerk) weight. Raised from 10 in the same "
                          "10 m/s search that set the lateral comfort weights: the lateral side "
                          "cannot reach a_x/jerk at all, so those channels only move from here")
    mpc.add_argument("--a-min", type=float, default=-4.05, help="hard lower bound on a_cmd (m/s^2)")
    mpc.add_argument("--a-max", type=float, default=2.4, help="hard upper bound on a_cmd (m/s^2)")
    mpc.add_argument("--ay-max", type=float, default=4.9,
                     help="comfortable/grip lateral-accel budget (m/s^2) a curve of a given radius "
                          "is allowed to demand -- caps the speed preview itself via v <= "
                          "sqrt(ay_max/kappa) ahead of the curve, per functions.refine_speed_preview. "
                          "Below B2D's own 4.90 comfort limit on purpose -- an offline B2D-penalty "
                          "search (joint lateral+longitudinal sim) followed by a real CARLA check "
                          "found 4.15 trades a bit of lap time for a much smoother corner entry "
                          "(TOTAL penalty 0.39 -> 0.32, |jerk| and lat-accel terms roughly halved)")
    mpc.add_argument("--lut", default=os.path.join(HERE, "longitudinal_lookup", "longitudinal_lut.npz"))
    # Pedal layer: validate_lut.py's tuned pedal-layer value -- all three scripts run the identical stack (LookupController = LUT feedforward + PID on acceleration error, then the u_tau low-pass), so the gains it was tuned against carry over unchanged.
    mpc.add_argument("--kp", type=float, default=0.6, help="accel-tracking PID proportional gain")
    mpc.add_argument("--ki", type=float, default=0.05, help="accel-tracking PID integral gain")
    mpc.add_argument("--kd", type=float, default=0.25, help="accel-tracking PID derivative gain")
    mpc.add_argument("--u-tau", type=float, default=0.6,
                     help="low-pass filter time constant on the pedal command u, before it's applied (s)")

    # ---- lateral MPC ---- #
    NOTE_LAT_TUNE = (
                     'Retuned 2026-08-26 at 10 AND 15 m/s together, against the cached Stanley baseline in '
                     'run_cache/ (10 m/s cross 0.135/0.303 m, heading 3.31/13.24 deg, Comfortness 0.270; 15 '
                     'm/s 0.160/0.436 m, 3.59/14.88 deg, 0.097). All nine knobs were searched TOGETHER '
                     '(population perturbation -- every weight jittered log-normally at once, never one at a '
                     'time), scored on repeated medians, and the finalists re-verified over 15 runs per speed. '
                     'The requirement was: beat BOTH Stanley and "mpc-kin" on cross-track RMSE and peak at '
                     'both speeds, hold cross-track RMSE <= 0.07 m / peak <= 0.17 m at 10 m/s, and hold the '
                     'BEST Comfortness of the three at both speeds. Measured, 15-run medians -- 10 m/s: cross '
                     '0.041 m RMSE / 0.121 m peak, heading 2.52/12.10 deg, Comfortness 0.371 (mpc-kin 0.333, '
                     'Stanley 0.270); 15 m/s: cross 0.047/0.192 m, heading 2.52/12.36 deg, Comfortness 0.129 '
                     '(mpc-kin 0.097, Stanley 0.097). CAUTION on Comfortness: it is passed-segments/total (35 '
                     'segments at 10 m/s, 31 at 15), so it moves in steps of ~0.03 and single runs flip by one '
                     'segment -- rank configs on repeated medians only. At 15 m/s the margin over Stanley IS '
                     'one segment; it held over 9- and 15-run medians but do not expect a single run to show '
                     'it. The reason it is that tight is structural: most failing segments fail on lat_acc, '
                     'and lat_acc = v^2*kappa is set by the curvature speed cap (--ay-max) in the longitudinal '
                     'stack all three controllers share, not by these weights.')
    lat = parser.add_argument_group("lateral MPC")
    lat.add_argument("--lat-np", dest="lat_n_p", type=int, default=22,
                     help="lateral prediction horizon (steps) -- 1.1s at dt=0.05. Set by the joint "
                          "10+15 m/s search NOTE_LAT_TUNE describes, together with --lat-nc; "
                          "the pair was searched alongside the weights, not fixed first. Longer "
                          "horizons (28-35) were sampled and lost: they smooth the corner entry "
                          "but give back cross-track peak on this route's tighter corners")
    lat.add_argument("--lat-nc", dest="lat_n_c", type=int, default=9, help="lateral control horizon (steps, <= --lat-np)")
    lat.add_argument("--w-ey", type=float, default=637.9681,
                     help="cross-track error weight, 'mpc'/'mpc-kf' only -- see --kin-w-ey for 'mpc-kin'. "
                          + NOTE_LAT_TUNE +
                          "This one sets the tracking/comfort trade directly and the 2026-08-26 "
                          "search pushed it far higher than the old 122: paired with --w-ddelta "
                          "~99 and --w-epsi ~25 it buys cross-track peak 0.121 m at 10 m/s "
                          "WITHOUT losing Comfortness, which is the combination single-weight "
                          "sweeps of this knob never found")
    lat.add_argument("--w-epsi", type=float, default=24.8441,
                     help="heading error weight, 'mpc'/'mpc-kf' only -- see --kin-w-epsi for 'mpc-kin'")
    lat.add_argument("--w-ay", type=float, default=0.2293,
                     help="lateral acceleration tracking weight, 'mpc'/'mpc-kf' only ('mpc-kin' has no "
                          "a_y output at all, see LateralMPCKinematic's docstring) -- default 0 (see "
                          "mpc_mpc.py's LateralMPC docstring: forcing a_y/r/r_dot toward the steady-turn "
                          "feedforward fights e_y/e_psi's own targets in a curve and was measured to "
                          "cost ~2m of steady cross-track offset before this was found)")
    lat.add_argument("--w-r", type=float, default=4.0009,
                     help="yaw rate tracking weight, 'mpc'/'mpc-kf' only (see --w-ay) -- see --kin-w-r "
                          "for 'mpc-kin's own (differently-scaled) steering-feedforward weight")
    lat.add_argument("--w-rdot", type=float, default=4.1942,
                     help="yaw acceleration tracking weight, 'mpc'/'mpc-kf' only -- 'mpc-kin' has no "
                          "r_dot output (see --w-ay and LateralMPCKinematic's docstring). Lowered from "
                          "120: rdot is formed as D*delta with D = lf*Cf/Iz = 33.3, so the effective "
                          "penalty on the input is w_rdot*D^2 -- 120 gave 1.3e5 against w_delta = 1, "
                          "which throttled the steering response enough to cost route completion.")
    lat.add_argument("--w-delta", type=float, default=0.4501,
                     help="steer magnitude weight, 'mpc'/'mpc-kf' only -- see --kin-w-delta for 'mpc-kin'")
    lat.add_argument("--w-ddelta", type=float, default=99.0259,
                     help="steer rate weight, 'mpc'/'mpc-kf' only -- raised from 1 in the same B2D-"
                          "penalty search that set --ay-max: with the curve-speed cap doing most of the "
                          "comfort work, a stiffer rate cost here trims the rest without hurting "
                          "lateral_error. See --kin-w-ddelta for 'mpc-kin'")
    lat.add_argument("--delta-max-deg", type=float, default=30.0,
                     help="hard steer-magnitude limit (deg, bicycle-model wheel angle), shared by "
                          "'mpc'/'mpc-kf'/'mpc-kin' -- kept well under max_steer since Cf/Cr (and, for "
                          "'mpc-kin', the small-angle tan(delta)~=delta linearization) are only valid "
                          "over the range they were calibrated/derived for. Also clips 'stanley''s raw "
                          "atan2 output (see run_trial()'s 'stanley' branch) -- the exact same 30 deg "
                          "default stanley_mpc.py's own +-(3/7)*max_steer clip already produces, so "
                          "this isn't a new limit for Stanley, just this file's own copy of that one")
    lat.add_argument("--ddelta-max-deg", type=float, default=70.0,
                     help="hard steer-rate limit (deg/s), shared by 'mpc'/'mpc-kf'/'mpc-kin'. Raised "
                          "from 70 (mpc_mpc_kinematic.py's own default) in this file specifically: at "
                          "15 m/s on this route 'mpc-kin' was found saturating this constraint ~19%% "
                          "of ticks (STEER_SAT_DEBUG=1), and 70 was simply too tight for the route's "
                          "corners at that speed -- raising it to 100 dropped 'mpc-kin' cross-track "
                          "RMSE 0.218->0.141 m and heading RMSE 4.02->1.98 deg with no oscillation "
                          "(steer magnitude stayed well under --delta-max-deg throughout), and helped "
                          "'mpc-kf' too (cross-track 0.069->0.048 m) with no regression at 10 m/s for "
                          "either. Still well under what stanley_mpc.py's own Stanley law does "
                          "unconstrained on this same route (its raw tick-to-tick rate reaches ~107 "
                          "deg/s with no hard cap at all, see run_trial()'s 'stanley' branch/module "
                          "docstring) -- this isn't loosening the comparison in Stanley's favor, it's "
                          "closing part of the gap the other direction.")
    lat.add_argument("--mass", type=float, default=VEHICLE_DEFAULTS["mass"],
                     help="vehicle mass (kg) -- 'mpc'/'mpc-kf' only, 'mpc-kin' doesn't use it")
    lat.add_argument("--iz", type=float, default=VEHICLE_DEFAULTS["iz"],
                     help="yaw moment of inertia (kg m^2) -- 'mpc'/'mpc-kf' only, 'mpc-kin' doesn't use it")
    lat.add_argument("--cf", type=float, default=VEHICLE_DEFAULTS["cf"],
                     help="front cornering stiffness (N/rad) -- 'mpc'/'mpc-kf' only, 'mpc-kin' doesn't use it")
    lat.add_argument("--cr", type=float, default=VEHICLE_DEFAULTS["cr"],
                     help="rear cornering stiffness (N/rad) -- 'mpc'/'mpc-kf' only, 'mpc-kin' doesn't use it")

    # ---- lateral MPC (kinematic) ---- #
    # Own weight set, deliberately NOT shared with --w-ey/--w-epsi/--w-r/--w-delta/--w-ddelta above:
    # LateralMPCKinematic's QP has a different structure (no output layer, see its docstring) and w_r
    # in particular means something physically different here (a direct steering-vs-Ackermann-
    # feedforward term, not an output-tracking term riding on Cf/Cr-scaled dynamics), so a weight
    # tuned for one model has no reason to transfer to the other. --lat-np/--lat-nc/--delta-max-deg/
    # --ddelta-max-deg above ARE still shared -- those are horizon length and actuator limits, not
    # model-specific cost weights.
    NOTE_KIN_TUNE = (
                     'Retuned 2026-08-26 at 10 m/s, against the cached Stanley baseline in run_cache/ (cross '
                     '0.135/0.303 m, heading 3.31/13.24 deg, Comfortness 0.270). All seven knobs were searched '
                     'TOGETHER (population perturbation -- every weight jittered log-normally at once, never '
                     'one at a time), scored on repeated medians. The hard requirement was that all FOUR error '
                     'metrics beat Stanley with real margin. Measured, 14-run medians at 10 m/s: cross 0.103 m '
                     'RMSE / 0.259 m peak, heading 2.36 / 9.53 deg -- ratios 0.76 / 0.86 / 0.71 / 0.72, plus '
                     'Comfortness 0.333 vs 0.270. The 5 metrics that still lose to Stanley are all in the MEAN '
                     'family (a_y mean/peak, yaw-rate mean, yaw-accel mean, |jerk| total mean): tracking the '
                     'curvature accurately forces a_y = v^2*kappa, so those means cannot be won without giving '
                     'the path back. Note the winning weights are SMALL on --kin-w-delta and LARGE on --kin-w- '
                     'ey relative to the old defaults -- leaning on the steer-magnitude cost was what the old '
                     'tuning did, and it cost cross-track margin.')
    kin = parser.add_argument_group("lateral MPC (kinematic, --controller mpc-kin)")
    # mpc-kin gets its OWN horizon knobs. --lat-np/--lat-nc are shared by "mpc"/"mpc-kf"/"mpc-kin",
    # so tuning the horizon for one of them silently retunes the others -- and "mpc-kf"'s pair is
    # already fixed by its own search. Default None = fall back to the shared value, so leaving
    # these alone reproduces exactly what this script did before they existed.
    kin.add_argument("--kin-np", dest="kin_n_p", type=int, default=14,
                     help="'mpc-kin' prediction horizon (steps). Shorter than --lat-np's 20 on "
                          "purpose: the kinematic model has no tyre slip, so the further ahead it "
                          "predicts the more it is predicting a car that does not exist. None "
                          "falls back to --lat-np")
    kin.add_argument("--kin-nc", dest="kin_n_c", type=int, default=3,
                     help="'mpc-kin' control horizon (steps, <= --kin-np). None falls back to "
                          "--lat-nc")
    kin.add_argument("--kin-w-ey", type=float, default=18.5776,
                     help="cross-track error weight. mpc_mpc_kinematic.py's own default is 3.0: a "
                          "10 m/s closed-loop check on this route found w_ey>=6 (with w_epsi scaled "
                          "alongside it) threw the loop into steer oscillation under THAT file's "
                          "--ddelta-max-deg=70 (yaw-rate peaks ~90-98 deg/s, cross-track RMSE ~0.30 m) "
                          "-- see mpc_mpc_kinematic.py's own copy of this help text for the full "
                          "isolation story (w_ey/w_epsi too high relative to --kin-w-delta demands "
                          "corrections sized for a no-slip vehicle that a real slipping car can't "
                          "deliver in one step). Raised back up here specifically because this file "
                          "also raised --ddelta-max-deg to 100 (see its own help) -- with that extra "
                          "actuator-rate headroom, w_ey=10 was retested at the exact w_ey=6 point that "
                          "used to oscillate and found clean (yaw-rate peak ~59-75 deg/s at 10/15 m/s, "
                          "no oscillation), while measurably improving 'mpc-kin' at every speed tested: "
                          "cross-track RMSE 10 m/s 0.14->0.12 m, 15 m/s 0.14->0.13 m (comfortness "
                          "0.0->0.03), 5 m/s 0.17->0.14 m. That last number is the caveat: at 5 m/s "
                          "'mpc-kin' still trails 'stanley' (cross-track ~0.05 m) even at w_ey=10, and "
                          "mpc_mpc_kinematic.py's own docstring records the OLD (pre-this-file, even "
                          "higher) gains topping out at 0.11 m there before instability set in higher "
                          "up -- this looks like the no-slip kinematic model's own structural floor on "
                          "this route's low-speed corners, not something --kin-w-ey alone closes the "
                          "rest of the way. See --kin-w-epsi (scaled alongside this)")
    kin.add_argument("--kin-w-epsi", type=float, default=2.0164,
                     help="heading error weight -- see --kin-w-ey. " + NOTE_KIN_TUNE)
    kin.add_argument("--kin-w-r", type=float, default=1.14,
                     help="steering-vs-Ackermann-feedforward tracking weight (delta -> L*kappa, see "
                          "LateralMPCKinematic's docstring) -- off by default. Measured to have very "
                          "little effect either way on the oscillation described under --kin-w-ey (it "
                          "is NOT what fixes it), but delta_ff=L*kappa is the no-slip target -- at "
                          "speed the real car needs more delta than that for the same curvature, so "
                          "this term pulls delta toward a value that's measurably too small. Left at 0 "
                          "since --kin-w-ey/--kin-w-epsi's own e_psi feedback already supplies the "
                          "steering demand, correctly sized, without this potentially-biased assist. "
                          "The 2026-08-26 joint search nevertheless settled on ~1.14 rather than "
                          "0: with --kin-w-delta driven down to ~0.11 the steer command needs "
                          "SOME anchor, and a mildly-too-small Ackermann target beat none")
    kin.add_argument("--kin-w-delta", type=float, default=0.1126, help="steer magnitude weight")
    kin.add_argument("--kin-w-ddelta", type=float, default=84.4172,
                     help="steer rate weight -- also measured to have little effect on the --kin-w-ey "
                          "oscillation on its own (see there), but doesn't hurt and gives some extra "
                          "smoothing on top of the w_ey/w_epsi fix")

    # ---- controller selection ---- #
    parser.add_argument("--controller", nargs="+",
                        default=["stanley", "mpc", "mpc-kf", "mpc-kin", "vad-pid"],
                        choices=("stanley", "mpc", "mpc-kf", "mpc-kin", "vad-pid"),
                        help="which controller(s) to run this route with. Default: every controller "
                             "in this repo, one trial each, all overlaid in one comparison. One "
                             "controller: unchanged single-trial output (3 figures). Two or more: "
                             "each runs its own trial and results are compared instead (trajectory "
                             "overlay + 8-metric comparison figure + each trial's own lateral/"
                             "longitudinal pair; plot_comparison() overlays any number of runs, one "
                             "color per entry -- COMPARE_COLORS has exactly 5, one per controller "
                             "here). 'stanley' pairs the closed-form Stanley law (stanley_control(), "
                             "evaluated at the front axle) with the same MpcLongitudinal every other "
                             "entry but 'vad-pid' uses -- no QP, no v_y/r estimation. 'mpc' and "
                             "'mpc-kf' run the identical dynamic-model LateralMPC/MpcLongitudinal "
                             "pair -- 'mpc' feeds x0 CARLA's own ground-truth v_y, 'mpc-kf' feeds it "
                             "VyKalmanFilter's online estimate instead (kalman_filter.py). 'mpc-kin' "
                             "runs LateralMPCKinematic instead of LateralMPC -- a rear-axle kinematic "
                             "bicycle model (state [e_y, e_psi] only, no v_y/r, no Kalman filter "
                             "involved at all) -- against the same MpcLongitudinal; see "
                             "LateralMPCKinematic's own docstring for the state-space derivation. "
                             "'vad-pid' is the real Bench2DriveZoo/VAD baseline (PIDController.control_pid, "
                             "see vad_pid_controller.py) -- combined lateral+longitudinal PID, binary "
                             "brake, flat/un-refined speed target (no curvature-aware speed cap).")

    # ---- Kalman filter (v_y estimation, --controller mpc-kf) ---- #
    kfg = parser.add_argument_group("Kalman filter (--controller mpc-kf)")
    kfg.add_argument("--kf-gyro-std", type=float, default=10,
                     help="synthetic gyro (dpsi) noise stddev fed to the filter, deg/s -- CARLA's "
                          "own sensor.other.imu runs near-noiseless by default, see kalman_filter.py")
    kfg.add_argument("--kf-accel-std", type=float, default=1,
                     help="synthetic accelerometer (a_y) noise stddev fed to the filter, m/s^2")
    kfg.add_argument("--kf-seed", type=int, default=0, help="seed for the filter's synthetic measurement noise")
    kfg.add_argument("--kf-q-vy", type=float, default=1e-3, help="process noise variance on v_y, (m/s)^2/step")
    kfg.add_argument("--kf-q-r", type=float, default=1e-3, help="process noise variance on r, (rad/s)^2/step")
    kfg.add_argument("--kf-r-dpsi", type=float, default=None,
                     help="measurement noise variance on dpsi, (rad/s)^2 -- default: matches "
                          "--kf-gyro-std^2 (the noise actually injected)")
    kfg.add_argument("--kf-r-ay", type=float, default=None,
                     help="measurement noise variance on a_y, (m/s^2)^2 -- default: matches "
                          "--kf-accel-std^2 (the noise actually injected). Both Q/R default to a "
                          "generic starting point -- `python3 kalman_filter.py --tune` finds better "
                          "values against logged data (mpc_mpc.py --log-npz) far cheaper than tuning "
                          "against live CARLA runs here.")

    parser.add_argument("--warm-start-settle", type=float, default=WARM_START_SETTLE_S,
                        help="seconds ALWAYS discarded before the warm-up convergence gate is even "
                             "consulted, to let the un-physical first ticks pass (spawn accelerometer "
                             "spike, suspension, Kalman covariance). A floor, not the whole warm-up: "
                             "see WARM_START_SETTLE_S for the gate that runs after it")
    parser.add_argument("--warm-start-speed", type=float, default=None,
                        help="speed (m/s) injected at spawn so the run starts rolling. Default: "
                             "--initial-speed, i.e. the car begins the scored route already at "
                             "speed; pass 0 to launch from rest instead. The matching gear is "
                             "forced for a few ticks with it (see WARM_START_INJECT_GEARS) -- "
                             "without that the injected speed collapses back to ~9.3 m/s within a "
                             "second no matter the throttle, because set_target_velocity() moves "
                             "the body and leaves the gearbox in neutral")
    parser.add_argument("--stanley-cached", action="store_true",
                        help="replay a saved Stanley baseline (run_cache/Stanley_<speed>ms.npz, "
                             "plus the .mp4 alongside it if present) instead of driving it -- for "
                             "holding the baseline fixed while tuning another controller. Nothing "
                             "here writes that file; it is produced out of band. Falls back to "
                             "driving if there is no cache for this --initial-speed")

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
    if args.warm_start_speed is None:
        args.warm_start_speed = args.initial_speed

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

    origin_transform, path_x, path_y = build_path(world)
    path = build_path_spline(path_x, path_y)
    print(f"Route: {len(path_x)} points, {path.s_max:.1f} m, "
          f"start=({path_x[0]:.1f}, {path_y[0]:.1f}) goal=({path_x[-1]:.1f}, {path_y[-1]:.1f})")

    # Only the vehicle's spawn point moves -- path_x/path_y/path above stay exactly the traced
    # route, so scoring/logging (last_s, e_y, ...) is unaffected by where the car spawns.
    spawn_transform = spawn_at(world, args.spawn_x, args.spawn_y)
    spawn_to_start_m = origin_transform.location.distance(spawn_transform.location)
    print(f"Spawn: ({args.spawn_x:.1f}, {args.spawn_y:.1f}) -> nearest waypoint "
          f"({spawn_transform.location.x:.1f}, {spawn_transform.location.y:.1f}), "
          f"{spawn_to_start_m:.1f} m from route start "
          f"({origin_transform.location.x:.1f}, {origin_transform.location.y:.1f}).")

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(spawn_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    controller_labels = {"stanley": "Stanley", "mpc": "MPC", "mpc-kf": "MPC-KF", "mpc-kin": "MPC-KIN",
                        "vad-pid": "VAD-PID"}
    multi = len(args.controller) > 1
    results = {}
    # Cached baselines go in FIRST so they keep the leftmost colour/dash slot and the first summary
    # block -- the same position they had when they were driven alongside everything else.
    videos = []   # [(label, mp4 경로, 프레임 수)] -- --controller 순서 유지
    try:
        for key in args.controller:
            # "stanley"/"mpc"/"mpc-kf"/"mpc-kin" all drive through the same MpcLongitudinal
            # (longitudinal) -- see run_trial()'s controller_key docstring for what differs between
            # them on the lateral side. "vad-pid" replaces the whole split pair with its own
            # combined controller instead.
            if key == "stanley" and args.stanley_cached:
                chist, cframes = load_stanley(args)
                if chist:
                    results[controller_labels[key]] = chist
                    cmp4 = os.path.splitext(_stanley_cache_path(args))[0] + ".mp4"
                    if args.record and cframes and os.path.exists(cmp4):
                        os.makedirs(args.video_dir, exist_ok=True)
                        work = os.path.join(args.video_dir, run_name("stanley") + ".mp4")
                        shutil.copy2(cmp4, work)     # the stitcher deletes what it consumes
                        videos.append((controller_labels[key], work, cframes))
                        print(f"  [stanley-cache] 영상도 합치기에 포함 ({cframes} frames)")
                    continue

            controller = VadPidController(args) if key == "vad-pid" else MpcLongitudinal(args)
            print(f"\n=== running controller: {controller_labels[key]} ===")
            hist, video_meta = run_trial(world, spawn_transform, path_x, path_y, path, blueprint,
                                         imu_bp, controller, key, args,
                                         spawn_to_start_m=spawn_to_start_m,
                                         video_suffix=key if multi else "")
            if hist and hist["t"]:
                results[controller_labels[key]] = hist
                if os.environ.get("STEER_SAT_DEBUG"):
                    _debug_steer_saturation(controller_labels[key], hist, args)
                if os.environ.get("HIST_DUMP_DIR"):
                    import json
                    dump_path = os.path.join(os.environ["HIST_DUMP_DIR"], f"{key}.json")
                    with open(dump_path, "w") as f:
                        json.dump({k: v for k, v in hist.items() if isinstance(v, list)}, f)
                    print(f"  [hist-dump] wrote {dump_path}")
            # --controller 에 적은 순서 그대로 쌓는다 -- 그 순서가 곧 합친 영상의 좌->우 배치다.
            if video_meta is not None:
                videos.append((controller_labels[key], video_meta["path"], video_meta["frames"]))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("Cleaned up: world settings restored.")

    # 제어기를 둘 이상 녹화했으면 좌우로 합쳐 하나만 남긴다. 순서는 --controller 인자 순서.
    if len(videos) >= 2:
        if args.record == "auto":
            combined = os.path.join(args.video_dir, run_name("combined") + ".mp4")
        else:
            base, ext = os.path.splitext(args.record)
            combined = f"{base}_combined{ext}"
        stack_videos_side_by_side(videos, combined, fps=1.0 / args.dt)

    # One more figure just for the "mpc-kf" trial: v_y estimate vs. ground truth stacked over the
    # dpsi/a_y clean-vs-noisy sensor channels the filter actually ran on, same 3-panel report
    # kalman_filter.py's own offline replay draws (viz_utils.plot_kf_run). Runs whenever "mpc-kf"
    # was one of --controller's picks, alone or against "mpc".
    #
    # BUILT BEFORE plot_results()/plot_comparison() below, and deliberately left OPEN (no
    # plt.close): plt.show() displays every figure open at the time it is called, and those two
    # functions call it internally and block there. Built afterwards, as this used to be, the
    # figure was created only once the others had already been shown and dismissed -- so it was
    # saved to disk but never appeared on screen at all. Creating it first puts it on screen
    # alongside the rest, and viz_utils._show() then closes the whole batch on one keypress.
    if args.save_plot and "MPC-KF" in results:
        try:
            fig = plot_kf_run(results["MPC-KF"],
                              title=r"MPC-KF: $v_y$ estimate + sensor noise")
            os.makedirs(args.plot_dir, exist_ok=True)
            out_path = os.path.join(args.plot_dir, run_name("kf") + ".png")
            fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
            print(f"Figure saved: {out_path}")
        except Exception as exc:
            print(f"MPC-KF report plotting failed: {exc}")

    if len(results) == 1:
        (label, hist), = results.items()
        if args.save_plot:
            try:
                plot_results(path_x, path_y, hist, args.initial_speed, args.plot_dir, label=label)
            except Exception as exc:
                print(f"Plotting failed: {exc}")
                print_error_summary(hist, args.initial_speed)
        else:
            print_error_summary(hist, args.initial_speed)
    elif len(results) >= 2:
        if args.save_plot:
            try:
                plot_comparison(results, path_x, path_y, args.plot_dir, args.initial_speed)
            except Exception as exc:
                print(f"Plotting failed: {exc}")
                for label, hist in results.items():
                    print(f"\n--- {label} ---")
                    print_error_summary(hist, args.initial_speed)
        else:
            for label, hist in results.items():
                print(f"\n--- {label} ---")
                print_error_summary(hist, args.initial_speed)



if __name__ == "__main__":
    main()
