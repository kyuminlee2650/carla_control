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

WARM_START_SPEED_TOL = 0.3    # m/s
WARM_START_ACCEL_TOL = 0.5    # m/s^2
WARM_START_REACH_TOL = 0.5    # m -- how close to spawn_to_start_m counts as "reached" (GPS/tick noise)
WARM_START_TIMEOUT = 15.0     # s -- safety cap in case initial_speed is unreachable


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
    start (origin_transform.location). Used below both to size warm_start_timeout and, together with
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
            dt=args.dt, n_p=args.lat_n_p, n_c=args.lat_n_c, L=lf + lr,
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

    # stanley_mpc.py's own run_trial() low-pass-filters its final steer command (tau=0.1s) before
    # applying it; "mpc"/"mpc-kf"/"mpc-kin" get the equivalent smoothing structurally, from their
    # own QP's w_ddelta cost + hard ddelta_max rate constraint (see LateralMPC/LateralMPCKinematic's
    # docstrings) -- Stanley's closed-form law has no such term, so without this filter its raw
    # per-tick delta would be unsmoothed in a way stanley_mpc.py's own output never is.
    steer_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0.0) if controller_key == "stanley" else None
    # TEMPORARY (STANLEY_HARD_RATE=1): apples-to-apples test of whether "mpc"/"mpc-kf"/"mpc-kin"'s
    # underperformance at higher speed is explained by their hard ddelta_max rate CONSTRAINT (a QP
    # inequality, enforced every tick) rather than stanley_mpc.py's own soft tau=0.1s low-pass
    # (which measurably lets Stanley's delta change faster tick-to-tick than that hard limit would
    # -- see the sat-debug numbers this was built to check). Not stanley_mpc.py's real behavior;
    # off by default.
    prev_delta_stanley = 0.0

    accel = ImuAcceleration(dt=args.dt)
    yaw_unwrapper = AngleUnwrapper()
    rh_unwrapper = AngleUnwrapper()
    yaw_acc_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    prev_yaw_rate_rad = None
    last_s = 0.0

    # matches plot_results()'s expectations (viz_utils.plot_lateral/plot_longitudinal/plot_trajectory).
    # v_y_hat/dpsi_noisy/ay_noisy only ever get appended to for "mpc-kf" below -- stay empty for
    # "mpc"/"vad-pid", which viz_utils._get() treats the same as "never recorded" (see plot_lateral's
    # ref_key="v_y_hat", print_error_summary's "v_y estimate" row, and main()'s extra
    # plot_kf_run() figure for the "mpc-kf" trial only).
    hist = {"t": [], "x": [], "y": [], "v_x": [], "v_y": [], "v_y_hat": [], "dpsi_noisy": [],
            "ay_noisy": [], "v_des": [], "v_des_curve": [], "a_x": [], "a_x_raw": [], "a_y_raw": [],
            "jerk": [], "a_y": [], "yaw_rate": [], "yaw_acc": [], "jerk_total": [], "steer_deg": [],
            "throttle": [], "brake": [], "e_y": [], "yaw": [], "path_yaw": [], "e_theta": [],
            "a_cmd": []}
    warmed_up = False
    log_start_i = 0
    # WARM_START_TIMEOUT (15s) alone assumed warm-up only ever needs to cover a speed/accel
    # transient; the reach-the-route-start gate below can genuinely need longer than that to also
    # cover spawn_to_start_m at a modest --initial-speed -- pad the cap by a generous (1.5x, so the
    # vehicle doesn't need to be at cruise speed for the whole stretch) estimate of that drive time
    # rather than let a legitimate --spawn-x/-y distance get cut off by timed_out.
    warm_start_timeout = max(WARM_START_TIMEOUT,
                             1.5 * spawn_to_start_m / max(args.initial_speed, 0.5)
                             + WARM_START_TIMEOUT)

    imu = None
    recorder = None
    video_meta = None   # 녹화했을 때만 채워진다 (run_trial 의 반환값 2번째)
    try:
        world.tick()

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

        steps = int((args.max_duration + warm_start_timeout) / args.dt)
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
            jerk, jerk_total = accel.jerk, accel.jerk_total

            yaw_acc = yaw_acc_filter.step(
                0.0 if prev_yaw_rate_rad is None else (r - prev_yaw_rate_rad) / args.dt)
            prev_yaw_rate_rad = r

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
                if os.environ.get("STANLEY_HARD_RATE"):
                    # skip the tau=0.1s LPF here -- "mpc"/"mpc-kf"/"mpc-kin" have no equivalent
                    # post-hoc filter of their own (their only rate-limiting is the QP's hard
                    # ddelta_max constraint), so stacking the LPF on top of the same hard clamp
                    # would still leave Stanley with an extra smoothing stage none of the others get.
                    ddelta_max = math.radians(args.ddelta_max_deg) * args.dt
                    delta = clipping(delta, prev_delta_stanley + ddelta_max,
                                     prev_delta_stanley - ddelta_max)
                else:
                    delta = steer_filter.step(delta)
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
                # dist_from_spawn stays < spawn_to_start_m the whole time the car is still short of
                # the route's actual start -- straight-line, not path station, since last_s itself
                # stays pinned at path.s_min (0.0) the whole time the car is behind the route (see
                # the module docstring), so it can't tell "still approaching" from "just arrived".
                dist_from_spawn = math.hypot(ego_x - spawn_transform.location.x,
                                             ego_y - spawn_transform.location.y)
                reached_start = dist_from_spawn >= spawn_to_start_m - WARM_START_REACH_TOL
                converged = (abs(v_x - args.initial_speed) < WARM_START_SPEED_TOL
                            and abs(a_x) < WARM_START_ACCEL_TOL
                            and reached_start)
                timed_out = i * args.dt >= warm_start_timeout
                if converged or timed_out:
                    warmed_up = True
                    log_start_i = i
                    controller.reset(reset_arg)
                    status = "converged" if converged else f"timed out after {warm_start_timeout:.0f}s"
                    print(f"Warm-start {status}: v_x={v_x:.2f} m/s, a_x={a_x:.2f} m/s^2, "
                          f"dist_from_spawn={dist_from_spawn:.1f}/{spawn_to_start_m:.1f} m -- "
                          f"logging starts now.")
                else:
                    elapsed = time.time() - step_start
                    if elapsed < args.dt / args.times_run:
                        time.sleep(args.dt / args.times_run - elapsed)
                    continue

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
            hist["jerk"].append(jerk)
            hist["a_y"].append(a_y)
            hist["yaw_rate"].append(yaw_rate_deg)
            hist["yaw_acc"].append(yaw_acc)
            hist["jerk_total"].append(jerk_total)
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


def main():
    parser = argparse.ArgumentParser()

    # ---- simulation setting ---- #
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--times-run", type=float, default=5.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=100.0, help="scored run length (s)")
    parser.add_argument("--spawn-x", type=float, default=-100.0,
                        help="m -- vehicle spawns at the road waypoint nearest this raw map (x, y) "
                             "(see functions.spawn_at), NOT the route's own start; the scored route "
                             "itself (path_x/path_y, from build_path()'s own origin/dest indices) is "
                             "untouched. Gives the warm-up gate (see run_trial docstring) a straight "
                             "run-up to ramp up to --initial-speed on. Default (-90, 25) is this "
                             "map's own route start (-64.8, 24.5) backed up along the same straight "
                             "road; pick a point on this specific route's own straight lead-up for a "
                             "different route.")
    parser.add_argument("--spawn-y", type=float, default=25.0, help="m -- see --spawn-x")

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
    mpc.add_argument("--nc", dest="n_c", type=int, default=40, help="control horizon (steps, <= --np)")
    mpc.add_argument("--w-v", type=float, default=10.0, help="speed-tracking weight")
    mpc.add_argument("--w-a", type=float, default=1, help="commanded-acceleration magnitude weight")
    mpc.add_argument("--w-j", type=float, default=30, help="commanded-acceleration rate (jerk) weight")
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
    mpc.add_argument("--kp", type=float, default=0.15, help="accel-tracking PID proportional gain")
    mpc.add_argument("--ki", type=float, default=0.6, help="accel-tracking PID integral gain")
    mpc.add_argument("--kd", type=float, default=0.0, help="accel-tracking PID derivative gain")
    mpc.add_argument("--u-tau", type=float, default=0.02,
                     help="low-pass filter time constant on the pedal command u, before it's applied (s)")

    # ---- lateral MPC ---- #
    lat = parser.add_argument_group("lateral MPC")
    lat.add_argument("--lat-np", dest="lat_n_p", type=int, default=25,
                     help="lateral prediction horizon (steps) -- 1.25s at dt=0.05. Narrowed back "
                          "down from 30 in the same B2D-penalty search that set --ay-max: 30 (and "
                          "45) measurably worsened lateral_error, likely too long relative to the "
                          "route's tighter corners for the tuning at hand")
    lat.add_argument("--lat-nc", dest="lat_n_c", type=int, default=25, help="lateral control horizon (steps, <= --lat-np)")
    lat.add_argument("--w-ey", type=float, default=1000.0,
                     help="cross-track error weight, 'mpc'/'mpc-kf' only -- see --kin-w-ey for 'mpc-kin'")
    lat.add_argument("--w-epsi", type=float, default=100.0,
                     help="heading error weight, 'mpc'/'mpc-kf' only -- see --kin-w-epsi for 'mpc-kin'")
    lat.add_argument("--w-ay", type=float, default=1,
                     help="lateral acceleration tracking weight, 'mpc'/'mpc-kf' only ('mpc-kin' has no "
                          "a_y output at all, see LateralMPCKinematic's docstring) -- default 0 (see "
                          "mpc_mpc.py's LateralMPC docstring: forcing a_y/r/r_dot toward the steady-turn "
                          "feedforward fights e_y/e_psi's own targets in a curve and was measured to "
                          "cost ~2m of steady cross-track offset before this was found)")
    lat.add_argument("--w-r", type=float, default=3,
                     help="yaw rate tracking weight, 'mpc'/'mpc-kf' only (see --w-ay) -- see --kin-w-r "
                          "for 'mpc-kin's own (differently-scaled) steering-feedforward weight")
    lat.add_argument("--w-rdot", type=float, default=10,
                     help="yaw acceleration tracking weight, 'mpc'/'mpc-kf' only -- 'mpc-kin' has no "
                          "r_dot output (see --w-ay and LateralMPCKinematic's docstring). Lowered from "
                          "120: rdot is formed as D*delta with D = lf*Cf/Iz = 33.3, so the effective "
                          "penalty on the input is w_rdot*D^2 -- 120 gave 1.3e5 against w_delta = 1, "
                          "which throttled the steering response enough to cost route completion.")
    lat.add_argument("--w-delta", type=float, default=0.1,
                     help="steer magnitude weight, 'mpc'/'mpc-kf' only -- see --kin-w-delta for 'mpc-kin'")
    lat.add_argument("--w-ddelta", type=float, default=0.1,
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
    lat.add_argument("--ddelta-max-deg", type=float, default=100.0,
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
    kin = parser.add_argument_group("lateral MPC (kinematic, --controller mpc-kin)")
    kin.add_argument("--kin-w-ey", type=float, default=10.0,
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
    kin.add_argument("--kin-w-epsi", type=float, default=10.0, help="heading error weight -- see --kin-w-ey")
    kin.add_argument("--kin-w-r", type=float, default=0.0,
                     help="steering-vs-Ackermann-feedforward tracking weight (delta -> L*kappa, see "
                          "LateralMPCKinematic's docstring) -- off by default. Measured to have very "
                          "little effect either way on the oscillation described under --kin-w-ey (it "
                          "is NOT what fixes it), but delta_ff=L*kappa is the no-slip target -- at "
                          "speed the real car needs more delta than that for the same curvature, so "
                          "this term pulls delta toward a value that's measurably too small. Left at 0 "
                          "since --kin-w-ey/--kin-w-epsi's own e_psi feedback already supplies the "
                          "steering demand, correctly sized, without this potentially-biased assist")
    kin.add_argument("--kin-w-delta", type=float, default=1.0, help="steer magnitude weight")
    kin.add_argument("--kin-w-ddelta", type=float, default=60.0,
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
    videos = []   # [(label, mp4 경로, 프레임 수)] -- --controller 순서 유지
    try:
        for key in args.controller:
            # "stanley"/"mpc"/"mpc-kf"/"mpc-kin" all drive through the same MpcLongitudinal
            # (longitudinal) -- see run_trial()'s controller_key docstring for what differs between
            # them on the lateral side. "vad-pid" replaces the whole split pair with its own
            # combined controller instead.
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

    # Extra, on top of whatever plot_results()/plot_comparison() above already drew (same format as
    # mpc_mpc_comparison.py, untouched) -- one more figure just for the "mpc-kf" trial: v_y estimate
    # vs. ground truth stacked over the dpsi/a_y clean-vs-noisy sensor channels the filter actually
    # ran on, same 3-panel report kalman_filter.py's own offline replay draws (viz_utils.plot_kf_run).
    # Runs whenever "mpc-kf" was one of --controller's picks, regardless of whether it ran alone or
    # against "mpc".
    if args.save_plot and "MPC-KF" in results:
        try:
            import matplotlib.pyplot as plt
            fig = plot_kf_run(results["MPC-KF"],
                              title=r"MPC-KF: $v_y$ estimate + sensor noise")
            os.makedirs(args.plot_dir, exist_ok=True)
            out_path = os.path.join(args.plot_dir, run_name("kf") + ".png")
            fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            print(f"Figure saved: {out_path}")
        except Exception as exc:
            print(f"MPC-KF report plotting failed: {exc}")


if __name__ == "__main__":
    main()
