r"""Closed-loop verification of VyKalmanFilter (kalman_filter.py): drive the same route twice under
the identical LateralMPC/MpcLongitudinal control law, once with x0's v_y taken from CARLA's own
ground truth ("mpc") and once with it taken from the Kalman filter's own online estimate ("mpc-kf"),
then compare the two runs (trajectory overlay, 8-metric comparison, each trial's own lateral/
longitudinal pair) via viz_utils.plot_comparison(). Structurally this file IS mpc_mpc_comparison.py
(LateralMPC/MpcLongitudinal/run_trial/main all copied from it) with "mpc-kf" added alongside its
"vad-pid" baseline -- see mpc_mpc_comparison.py's own docstring for the shared Step 1-4 design
(route spline, longitudinal SpeedMPC, the LateralMPC QP itself, wiring order).

Step 5 (the one addition): "mpc-kf" runs one VyKalmanFilter per trial (constructed in run_trial(),
not main() -- OSQP/Kalman state is per-vehicle, same reason a fresh LateralMPC is built per trial).
Each tick, BEFORE x0/delta are computed (v_y_hat has to already exist to build x0), the filter's
predict()+update() run once using vx (measured, trusted directly) and delta_prev (whatever steering
was actually applied last tick -- this tick's own delta doesn't exist yet, see kalman_filter.py's
docstring on causality). Its gyro/accel inputs are the sim IMU's clean r/a_y plus injected synthetic
Gaussian noise (--kf-gyro-std/--kf-accel-std) -- default CARLA IMU noise is ~0, so without this the
filter would just be handed the truth and R would have nothing real to be tuned against, same reason
kalman_filter.py's offline replay injects it. Only x0's v_y slot is replaced by v_y_hat; x0's r stays
the real (clean) gyro reading both runs use, so the comparison isolates the v_y-estimation effect
specifically instead of also degrading yaw-rate feedback. hist["v_y"] (ground truth) is still logged
every tick regardless of controller, so the "mpc-kf" run's own v_y_hat can be checked against it
after a real drive, not just against the offline-replayed one -- see viz_utils.plot_lateral()'s
v_y panel (ref_key="v_y_hat") and print_error_summary()'s "v_y estimate" row.

"vad-pid" is a third pick for --controller, carried over unchanged from mpc_mpc_comparison.py
(_vad_waypoints_and_target + VadPidController, same adapter and same reasoning) so this file can
score the Kalman-filtered stack against the real Bench2DriveZoo/VAD baseline as well as against its
own ground-truth twin -- any two or three of "mpc"/"mpc-kf"/"vad-pid" can be run in one command and
land in the same comparison figures.

Usage (Ubuntu -- verified on this lab machine):
    # 1) the simulator, in its own terminal. This box's CARLA is NOT the ~/carla/CARLA_0.9.15 path
    #    hardcoded above/in $CARLA_ROOT (that one no longer exists here) -- functions.py finds the
    #    real tree by probing, see resolve_carla_root():
    #      /home/ailab/2026intern/carla/CarlaUE4.sh
    # 2) the controller, from this repo's own venv. The venv interpreter is spelled out on every
    #    line below on purpose, so any one of them can be copied and run on its own: the system
    #    python3 has neither carla nor osqp installed, so a bare `python3 mpc_mpc_KF.py` fails
    #    unless the venv happens to be active. `source .venv/bin/activate` once and then using
    #    plain `python3` is equivalent.
    cd ~/carla_control
    # ground-truth baseline alone (3 figures, same as mpc_mpc.py)
    .venv/bin/python mpc_mpc_KF.py --profile constant --initial-speed 5 --save-plot --controller mpc
    # closed-loop KF verification: both runs + comparison figures
    .venv/bin/python mpc_mpc_KF.py --profile constant --initial-speed 5 --save-plot --controller mpc mpc-kf
    # against the real VAD/Bench2Drive PID baseline (any subset, in any order)
    .venv/bin/python mpc_mpc_KF.py --profile step --initial-speed 10 --save-plot --controller mpc-kf vad-pid --record
    .venv/bin/python mpc_mpc_KF.py --profile constant --initial-speed 5 --save-plot --record \
        --controller mpc mpc-kf vad-pid
    # emergency stop at t=10s, held 5s, then straight back up to --initial-speed (see --profile step)
    .venv/bin/python mpc_mpc_KF.py --profile step --initial-speed 10 --step-time 10 --step-size -10 \
        --step-duration 5 --save-plot --controller mpc mpc-kf vad-pid
    # with Q/R found by `.venv/bin/python kalman_filter.py --tune`
    .venv/bin/python mpc_mpc_KF.py --controller mpc mpc-kf --save-plot \
        --kf-q-vy 1e-3 --kf-q-r 1e-3 --kf-r-dpsi 1e-5 --kf-r-ay 4e-2
"""

import argparse
import json
import math
import os
import queue
import sys
import time
from types import SimpleNamespace

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import expm

# vx 하한. forward Euler 였을 때 이 값(0.5)은 안정성 장치였다 -- 이산 A 가 저속에서 단위원을
# 벗어나므로 모델을 그 아래에서 못 쓰게 막아야 했다. 정확한 ZOH 는 전 속도에서 수축하므로 그
# 역할이 사라지고, 남는 것은 1/vx 의 0 나눗셈 방어뿐이다. 그래서 수치가 여전히 움직이는 어떤
# 속도보다도 훨씬 아래로 내린다 (b2d_controller/mpc_kf_controller.py 의 VX_EPS 와 같은 값).
VX_EPS = 1e-3

HERE = os.path.dirname(os.path.abspath(__file__))

CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/carla/CARLA_0.9.15"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.append(os.path.join(HERE, "longitudinal_lookup"))

import carla

from functions import (AngleUnwrapper, ImuAcceleration, LowPassFilter, build_path,
                       build_path_spline, control_input, get_vehicle_geometry, lateral_error,
                       normalize_angle, reference_preview,
                       refine_speed_preview, speed_reference, spawn_at)
from kalman_filter.kalman_filter import VyKalmanFilter
from longitudinal_lut import LongitudinalLUT
from lookup_controller import LookupController
from longitudinal_mpc import SpeedMPC
from vad_pid_controller import PIDController as VadPIDController
from viz_utils import (VIEWS, VideoRecorder, follow_with_spectator, plot_comparison, plot_kf_run,
                       plot_results, print_error_summary, run_name, stack_videos_side_by_side)

# Fixed on purpose, same as stanley_mpc.py: the route (build_path()'s default origin/dest spawn
# indices) is a property of this specific map, not something to rediscover via CLI flags.
MAP_NAME = "Town10HD_Opt"

WARM_START_SPEED_TOL = 0.3    # m/s
WARM_START_ACCEL_TOL = 0.5    # m/s^2
WARM_START_REACH_TOL = 0.5    # m -- how close to spawn_to_start_m counts as "reached" (GPS/tick noise)
WARM_START_TIMEOUT = 15.0     # s -- safety cap in case initial_speed is unreachable


def _load_vehicle_defaults():
    """lateral_parameter/yaw_inertia.json carries Cf/Cr/mass alongside Iz (the cornering-stiffness
    trial's own output, re-saved when the inertia trial ran on top of it) -- one file, not two."""
    path = os.path.join(HERE, "lateral_parameter", "yaw_inertia.json")
    try:
        with open(path) as f:
            data = json.load(f)
        return dict(mass=data["mass"], iz=data["Iz"], cf=data["Cf"], cr=data["Cr"])
    except (FileNotFoundError, KeyError) as exc:
        print(f"Warning: could not load vehicle params from {path} ({exc}); using rough defaults.")
        return dict(mass=1696.0, iz=2916, cf=82941, cr=54425)


VEHICLE_DEFAULTS = _load_vehicle_defaults()


def curvature_preview(path, last_s, vx_preview, dt):
    """Length-Np curvature preview: walk a continuous station cursor forward by v_x*dt each
    predicted step and evaluate PathSpline.kappa() directly there -- no more array index to walk
    (path.kappa() takes any station in [s_min, s_max] directly), so this is just a station-vs-time
    remap, not a search.

    Re-run every cycle, not cached: both last_s (the vehicle keeps moving) and vx_preview (a new
    speed plan every cycle) change, so the space<->time mapping this builds is only valid for the
    one cycle it was built in.

    Index j of the returned array feeds LateralMPC's forward-Euler step x_{j+1} = x_j + dt*(A(vx_j)
    x_j + B delta_j + E(vx_j) kappa_j) (see _condense()'s Ed[j] placement), so kappa_prev[j] has to
    be the curvature at x_j's own position (s_j) -- evaluate BEFORE advancing s_cursor for step j,
    not after, or every entry ends up one step (dt*v_x, ~0.25 m at 5 m/s) further down the path than
    the state it's actually paired with."""
    s_cursor = last_s
    kappa_prev = []
    for vx in vx_preview:
        kappa_prev.append(float(path.kappa(min(s_cursor, path.s_max))))
        s_cursor += max(vx, 0.0) * dt
    return kappa_prev


# ----------------------------------------------------------------------------- lateral MPC

class LateralMPC:
    r"""MPC lateral controller over the 2-DOF bicycle-model error dynamics.

        x = [v_y, r, e_y, e_psi]^T,  input delta (front steer, rad),  disturbance kappa (path
        curvature, 1/m), r = yaw rate:

        x_dot = A(v_x) x + B delta + E(v_x) kappa

            A(v_x) = [[-(Cf+Cr)/(m vx),        (lr Cr - lf Cf)/(m vx) - vx, 0, 0],
                      [(lr Cr - lf Cf)/(Iz vx), -(lf^2 Cf + lr^2 Cr)/(Iz vx), 0, 0],
                      [1,                       0,                            0, vx],
                      [0,                       1,                            0, 0]]
            B = [Cf/m, lf Cf/Iz, 0, 0]^T                E(v_x) = [0, 0, 0, -vx]^T

    v_x is the scheduling parameter (previewed over the horizon, not held fixed), so A/E/C below are
    re-linearized and re-discretized at every predicted step.

    Output y = [e_y, e_psi, a_y, r, r_dot]^T. e_y, e_psi, r come straight off the state; a_y and
    r_dot don't, so they're derived from the state equation rather than being new unknowns: r_dot is
    just the state equation's own second row, and a_y = v_y_dot + v_x*r -- add v_x*r to the first
    row's RHS and its -vx/+vx terms cancel, leaving a_y linear in v_y, r, delta with nothing left
    over:

        C(v_x) = [[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [-(Cf+Cr)/(m vx), (lr Cr - lf Cf)/(m vx), 0, 0],
                  [0, 1, 0, 0],
                  [(lr Cr - lf Cf)/(Iz vx), -(lf^2 Cf + lr^2 Cr)/(Iz vx), 0, 0]]
        D = [0, 0, Cf/m, 0, lf Cf/Iz]^T   (v_x-independent)

    Reference: e_y_ref = e_psi_ref = r_dot_ref = 0, r_ref = v_x*kappa, a_y_ref = v_x^2*kappa
    (steady-turn feedforward built from the previewed curvature).

    Forward-Euler per predicted step k: A_d,k = I + Ts*A(v_x,k), B_d = Ts*B (v_x-independent),
    E_d,k = Ts*E(v_x,k). Condensing over the Np-step horizon with an Nc-step control horizon
    (Nc <= Np, free for Nc steps then held) gives

        X = Abar x0 + Bbar U + Ebar K,   Y = Cbar X + Dbar U

    Bbar/Dbar are condensed as if every one of the Np steps had its own free input, then
    right-multiplied by the Np x Nc hold matrix M_c that folds the held tail onto the last free
    column -- same result as the recursive h_i = A_d,i h_{i-1} + B_d accumulation, without writing
    that recursion out separately. Ebar is condensed the same way but never blocked -- kappa is a
    known preview, not a decision variable, so every one of the Np steps keeps its own free column.

    Cost J = (Y-Yref)' W1 (Y-Yref) + U' W2 U + (GU-Uprev)' W3 (GU-Uprev), where G is now Nc x Nc
    (row 0 = [1,0,...,0], row l>=1 = -1 at col l-1 / +1 at col l) and Uprev = [delta_prev,0,...,0]',
    delta_prev the delta actually applied last tick (self.last_solution[0] going into this solve()).
    GU - Uprev is then the true rate sequence [delta_0-delta_prev, delta_1-delta_0, ...] -- unlike a
    G that only differences within this one solve's own planned U, this anchors step 0's rate to
    what the vehicle is actually doing right now, so -ddelta_max <= GU-Uprev <= ddelta_max is a real
    per-tick actuator-rate limit rather than just a smoothness prior on the internal plan (a QP that
    only constrained consecutive U elements never limited the applied delta's tick-to-tick jump at
    all, since receding-horizon control only ever executes U[0] and resolves from scratch next
    tick). Reduces to the box/rate-constrained QP

        min 1/2 U' H U + f' U   s.t.  -delta_max <= U <= delta_max,
                                       Uprev-ddelta_max <= GU <= Uprev+ddelta_max
        H = 2[(CbarBbar+Dbar)' W1 (CbarBbar+Dbar) + W2 + G' W3 G]
        f = 2[(CbarBbar+Dbar)' W1 (Cbar Abar x0 + Cbar Ebar K - Yref) - G' W3 Uprev]

    (the -G'W3 Uprev term comes from expanding (GU-Uprev)'W3(GU-Uprev) = U'G'W3GU - 2Uprev'W3GU +
    Uprev'W3Uprev -- the cross term is linear in U and has to land in f, not just the U'G'W3GU
    piece in H, or the rate cost silently stops penalizing relative to delta_prev at all.)

    Only the first element of U* is applied each cycle (receding horizon). H depends on the v_x
    preview (changes every cycle), so the QP is rebuilt from scratch on every solve() rather than
    reused with just q/l/u updated the way a fixed-A QP would be. l/u aren't fully static either
    now (Uprev shifts the rate rows), so they're assembled fresh in solve() too.
    """

    N_X = 4   # [v_y, r, e_y, e_psi]
    N_Y = 5   # [e_y, e_psi, a_y, r, r_dot]

    def __init__(self, dt, n_p, n_c, mass, Iz, lf, lr, Cf, Cr,
                w_ey, w_epsi, w_ay, w_r, w_rdot, w_delta, w_ddelta,
                delta_max, ddelta_max, vx_floor=VX_EPS):
        if not (0 < n_c <= n_p):
            raise ValueError(f"need 0 < n_c <= n_p, got n_c={n_c}, n_p={n_p}")
        if not (delta_max > 0 and ddelta_max > 0):
            raise ValueError(f"need delta_max, ddelta_max > 0, got {delta_max}, {ddelta_max}")

        self.dt, self.n_p, self.n_c = dt, n_p, n_c
        self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr = mass, Iz, lf, lr, Cf, Cr
        self.delta_max, self.ddelta_max, self.vx_floor = delta_max, ddelta_max, vx_floor

        self.B_cont = np.array([[Cf / mass], [lf * Cf / Iz], [0.0], [0.0]])
        # Bd is no longer a constant: under exact ZOH it is
        # integral_0^dt expm(A s) ds * B, and A depends on vx, so _discretize() returns one Bd per
        # previewed step. Kept here only because _continuous()/B_cont callers still read B_cont.
        self.D = np.array([[0.0], [0.0], [Cf / mass], [0.0], [lf * Cf / Iz]])

        self.M_c = self._build_hold_matrix(n_p, n_c)
        self.G = self._build_rate_matrix(n_c)

        self.W1 = np.diag(np.tile([w_ey, w_epsi, w_ay, w_r, w_rdot], n_p))
        self.W2 = w_delta * np.eye(n_c)
        self.W3 = w_ddelta * np.eye(n_c)   # G is now n_c x n_c (see _build_rate_matrix), not n_c-1
        self._GtW3G = self.G.T @ self.W3 @ self.G

        A_ineq = np.vstack([np.eye(n_c), self.G])
        self._A_ineq = sparse.csc_matrix(A_ineq)
        # box bounds on U are static; the rate rows are NOT (they shift with delta_prev each tick,
        # see solve()), so only the delta_max half is precomputed here.
        self._delta_l = -delta_max * np.ones(n_c)
        self._delta_u = delta_max * np.ones(n_c)

        self.last_solution = np.zeros(n_c)
        self.last_status = "unsolved"

    @staticmethod
    def _build_hold_matrix(n_p, n_c):
        """Np x Nc: raw free-input step i maps to column min(i, Nc-1) -- identity for the first
        Nc-1 steps, then every step from Nc-1 on shares the held last column."""
        M = np.zeros((n_p, n_c))
        for i in range(n_p):
            M[i, min(i, n_c - 1)] = 1.0
        return M

    @staticmethod
    def _build_rate_matrix(n_c):
        """Nc x Nc first-difference operator: (G U)_0 = delta_0, (G U)_l = delta_l - delta_{l-1}
        for l >= 1. Row 0 is deliberately just [1,0,...,0] -- solve() subtracts Uprev (delta_prev
        in slot 0) from G@U to turn that first row into delta_0 - delta_prev, anchoring the rate
        constraint to the delta actually applied last tick instead of leaving step 0 unconstrained."""
        G = np.eye(n_c)
        for l in range(1, n_c):
            G[l, l - 1] = -1.0
        return G

    def _continuous(self, vx):
        vx = max(vx, self.vx_floor)
        m, Iz, lf, lr, Cf, Cr = self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr
        A = np.array([
            [-(Cf + Cr) / (m * vx),           (lr * Cr - lf * Cf) / (m * vx) - vx, 0.0, 0.0],
            [(lr * Cr - lf * Cf) / (Iz * vx), -(lf**2 * Cf + lr**2 * Cr) / (Iz * vx), 0.0, 0.0],
            [1.0,                              0.0,                                   0.0, vx],
            [0.0,                              1.0,                                   0.0, 0.0],
        ])
        E = np.array([[0.0], [0.0], [0.0], [-vx]])
        C = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-(Cf + Cr) / (m * vx),           (lr * Cr - lf * Cf) / (m * vx), 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [(lr * Cr - lf * Cf) / (Iz * vx), -(lf**2 * Cf + lr**2 * Cr) / (Iz * vx), 0.0, 0.0],
        ])
        return A, E, C

    def _discretize(self, vx_preview):
        """Exact zero-order-hold discretization of (A, B, E) at each previewed speed, via

            expm([[A, B, E],
                  [0, 0, 0],
                  [0, 0, 0]] * dt)  =  [[Ad, Bd, Ed],
                                        [ 0,  1,  0],
                                        [ 0,  0,  1]]

        The augmented-matrix identity is used rather than the usual Bd = A^-1 (Ad - I) B because A
        is SINGULAR here -- the e_y and e_psi rows are pure integrators, so A has two zero
        eigenvalues at every speed and the inverse does not exist.

        This replaces forward Euler (Ad = I + dt*A). Euler is only accurate while dt*|lambda| << 1,
        and the bicycle model's lateral eigenvalues scale as 1/vx, so at low speed the discrete A
        left the unit circle and the prediction diverged. Exact ZOH is contractive at every speed.
        Cost is one expm of a 6x6 per previewed step, ~72 us measured, i.e. ~1.8 ms for a 25-step
        horizon against a 50 ms control period.
        """
        Ad, Bd, Ed, Ck = [], [], [], []
        n = self.N_X
        for vx in vx_preview:
            A, E, C = self._continuous(vx)
            aug = np.zeros((n + 2, n + 2))
            aug[:n, :n] = A
            aug[:n, n:n + 1] = self.B_cont
            aug[:n, n + 1:n + 2] = E
            M = expm(aug * self.dt)
            Ad.append(M[:n, :n])
            Bd.append(M[:n, n:n + 1])
            Ed.append(M[:n, n + 1:n + 2])
            Ck.append(C)
        return Ad, Bd, Ed, Ck

    def _condense(self, Ad, Bd, Ed, Ck):
        n_p, n_x, n_y = self.n_p, self.N_X, self.N_Y
        A_bar = np.zeros((n_p * n_x, n_x))
        B_full = np.zeros((n_p * n_x, n_p))
        E_bar = np.zeros((n_p * n_x, n_p))

        prod = np.eye(n_x)
        for k in range(n_p):
            prod = Ad[k] @ prod
            A_bar[k * n_x:(k + 1) * n_x, :] = prod

        for j in range(n_p):
            col_b = Bd[j].copy()      # per-step now, see _discretize's docstring
            col_e = Ed[j].copy()
            B_full[j * n_x:(j + 1) * n_x, j:j + 1] = col_b
            E_bar[j * n_x:(j + 1) * n_x, j:j + 1] = col_e
            for i in range(j + 1, n_p):
                col_b = Ad[i] @ col_b
                col_e = Ad[i] @ col_e
                B_full[i * n_x:(i + 1) * n_x, j:j + 1] = col_b
                E_bar[i * n_x:(i + 1) * n_x, j:j + 1] = col_e

        C_bar = np.zeros((n_p * n_y, n_p * n_x))
        D_full = np.zeros((n_p * n_y, n_p))
        for k in range(n_p):
            C_bar[k * n_y:(k + 1) * n_y, k * n_x:(k + 1) * n_x] = Ck[k]
            D_full[k * n_y:(k + 1) * n_y, k:k + 1] = self.D

        B_bar = B_full @ self.M_c
        D_bar = D_full @ self.M_c
        return A_bar, B_bar, E_bar, C_bar, D_bar

    def solve(self, x0, vx_preview, kappa_preview):
        """x0 = [v_y, r, e_y, e_psi] (vehicle-minus-path sign convention). vx_preview/kappa_preview:
        length-Np previews. Returns delta_cmd (rad, clipped to +-delta_max)."""
        vx_preview = np.clip(np.asarray(vx_preview, dtype=float), self.vx_floor, None)
        kappa_preview = np.asarray(kappa_preview, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1, 1)

        Ad, Bd, Ed, Ck = self._discretize(vx_preview)
        A_bar, B_bar, E_bar, C_bar, D_bar = self._condense(Ad, Bd, Ed, Ck)

        K = kappa_preview.reshape(-1, 1)
        y_ref = np.zeros((self.n_p * self.N_Y, 1))
        y_ref[2::self.N_Y, 0] = vx_preview**2 * kappa_preview   # a_y_ref
        y_ref[3::self.N_Y, 0] = vx_preview * kappa_preview      # r_ref

        # delta actually applied last tick (this solve's own U[0] once it's computed becomes NEXT
        # tick's delta_prev) -- anchors the rate constraint/cost to reality instead of just to this
        # one solve's own internal plan, see the class docstring.
        U_prev = np.zeros((self.n_c, 1))
        U_prev[0, 0] = self.last_solution[0]

        M = C_bar @ B_bar + D_bar
        H = 2.0 * (M.T @ self.W1 @ M + self.W2 + self._GtW3G)
        H = 0.5 * (H + H.T)
        f = 2.0 * (M.T @ self.W1 @ (C_bar @ A_bar @ x0 + C_bar @ E_bar @ K - y_ref)
                  - self.G.T @ self.W3 @ U_prev)

        rate_l = -self.ddelta_max + U_prev.ravel()
        rate_u = self.ddelta_max + U_prev.ravel()
        l_bound = np.concatenate([self._delta_l, rate_l])
        u_bound = np.concatenate([self._delta_u, rate_u])

        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(H), q=f.ravel(), A=self._A_ineq, l=l_bound, u=u_bound,
                    verbose=False, polish=False)
        result = solver.solve()
        self.last_status = result.info.status

        if result.x is None or not np.all(np.isfinite(result.x)):
            u = self.last_solution   # keep steering per the previous plan rather than snapping to 0
        else:
            u = result.x
            self.last_solution = u.copy()

        return float(np.clip(u[0], -self.delta_max, self.delta_max))


# ----------------------------------------------------------------------------- longitudinal (MPC)

class MpcLongitudinal:
    """SpeedMPC -> a_cmd -> LUT feedforward + PID pedal layer (see module docstring)."""
    label = "MPC"

    def __init__(self, args):
        self.args = args
        self.mpc = SpeedMPC(dt=args.dt, n_p=args.n_p, n_c=args.n_c,
                            w_v=args.w_v, w_a=args.w_a, w_j=args.w_j,
                            a_min=args.a_min, a_max=args.a_max)
        self.pedal_ctrl = LookupController(LongitudinalLUT(args.lut), kp=args.kp, ki=args.ki,
                                           kd=args.kd, dt=args.dt)
        self.u_filter = LowPassFilter(tau=args.u_tau, dt=args.dt, initial=0.0)

    def reset(self, u):
        self.pedal_ctrl.reset()   # drop the warm-up phase's PID integral before scoring starts
        self.u_filter = LowPassFilter(tau=self.args.u_tau, dt=self.args.dt, initial=u)

    def step(self, ctx):
        preview = reference_preview(self.args, ctx.t, self.args.n_p, self.args.dt, ctx.warmed_up)
        # curvature-aware speed cap: reduces the reference itself ahead of a curve, instead of
        # relying on the lateral controller to out-steer whatever speed the profile blindly asked
        # for -- see the module docstring's "no curvature-aware speed reduction" tuning note.
        preview = refine_speed_preview(preview, ctx.path, ctx.last_s, self.args.dt,
                                       a_y_max=self.args.ay_max)
        ctx.v_ref_curve = float(preview[0])   # curvature-clipped target, stashed for hist["v_des"]
        ctx.a_cmd = self.mpc.solve(ctx.v_x, preview)
        # SpeedMPC's own condensed prediction X = A_bar x0 + B_bar U*, re-derived here (not returned
        # by solve()) so LateralMPC's curvature preview can schedule against the same v_x trajectory
        # this loop is actually planning to drive, instead of assuming constant speed over its
        # horizon -- see the "which speed plan" discussion in the module docstring.
        ctx.v_x_preview = (self.mpc.A_bar.ravel() * ctx.v_x + self.mpc.B_bar @ self.mpc.last_solution)
        u_raw = self.pedal_ctrl.step(ctx.gear, ctx.v_x, ctx.a_cmd, a_meas=ctx.a_x_raw)
        return self.u_filter.step(u_raw)


def vx_preview_for_lateral(ctx, n_p_lat):
    """MpcLongitudinal stashes its own condensed speed prediction on ctx.v_x_preview; reuse it
    (truncated/padded to the lateral horizon) instead of assuming constant speed over it."""
    raw = np.asarray(ctx.v_x_preview, dtype=float)
    if len(raw) >= n_p_lat:
        return raw[:n_p_lat]
    return np.concatenate([raw, np.full(n_p_lat - len(raw), raw[-1])])


# ----------------------------------------------------------------------------- VAD/Bench2Drive PID baseline
# Copied verbatim from mpc_mpc_comparison.py (adapter + wrapper), not imported: that file is a
# sibling script rather than a library, and importing it would drag its whole argparse/main and a
# second copy of LateralMPC in with it. Keep the two in sync by hand if either is edited.

def _vad_waypoints_and_target(path, last_s, ego_x, ego_y, yaw, target_speed,
                              n_wp=6, dt_wp=0.5, target_lookahead_m=10.0):
    """Synthesize VAD's two control_pid() inputs from this repo's own reference PathSpline, since
    this repo has neither a learned trajectory predictor nor CARLA's discrete route commands:

      waypoints: n_wp points spaced dt_wp seconds apart AT THE FLAT target_speed (never
                 curvature-refined -- the baseline doesn't get that feature, per instruction),
                 matching VAD's own 6-step/0.5s prediction cadence, each rotated into the ego frame
                 in VAD's own [lateral, forward] axis order (this repo's usual convention elsewhere
                 is [forward, lateral] -- confirmed opposite by reading control_pid()'s own
                 angle = degrees(pi/2 - atan2(aim[1], aim[0])) formula: that only zeroes out on a
                 straight-ahead point if index 0 is lateral and index 1 is forward).
      target:    one further point (target_lookahead_m ahead), standing in for VAD's coarse
                 route-command waypoint -- this repo has no discrete turn commands to draw one from,
                 so a fixed lookahead is the practical substitute.

    Sign of "lateral" here was only reasoned from the source, not confirmed by driving -- flip it in
    the rotation below if a live CARLA check shows the car steering the wrong way.

    Past path.s_max, points are extrapolated straight along the route's own final tangent rather
    than clamped to s_max: clamping made every waypoint within one dt_wp*target_speed of the goal
    collapse onto the exact same point (s_max), so control_pid()'s desired_speed -- the norm of
    consecutive waypoint differences -- read as 0 and its brake<desired_speed<brake_speed check
    slammed the brake on ~7.5 m (at 15 m/s) short of the goal, well outside run_trial()'s own
    "reached end" band (last_s >= path.s_max - 0.1) -- the trial then never finishes early and just
    sits there until --max-duration. Extrapolating keeps the preview's own forward speed nonzero
    all the way through the actual finish line instead.

    target_speed = 0 (the --profile step stop window, see functions.speed_reference) collapses every
    waypoint onto the single point at last_s on purpose: control_pid() then reads desired_speed = 0
    and brakes, which is exactly the commanded behaviour. Steering does NOT degenerate with it --
    target above is a fixed GEOMETRIC lookahead, independent of target_speed, and control_pid()'s
    own use_target_to_aim test (|angle_target| < |angle|) switches to it precisely when the collapsed
    waypoints make its own aim angle meaningless.
    """
    def _ego_frame(s):
        if s <= path.s_max:
            wx, wy = path.xy(s)
        else:
            x_end, y_end = path.xy(path.s_max)
            yaw_end = float(path.yaw(path.s_max))
            overshoot = s - path.s_max
            wx = x_end + overshoot * math.cos(yaw_end)
            wy = y_end + overshoot * math.sin(yaw_end)
        dx, dy = wx - ego_x, wy - ego_y
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return lateral, forward

    waypoints = []
    s = last_s
    for _ in range(n_wp):
        s = s + max(target_speed, 0.0) * dt_wp
        waypoints.append(_ego_frame(s))
    waypoints = np.array(waypoints, dtype=float)

    target = np.array(_ego_frame(last_s + target_lookahead_m), dtype=float)
    return waypoints, target


class VadPidController:
    """Wraps vad_pid_controller.PIDController -- VAD/Bench2Drive's real baseline, doing lateral AND
    longitudinal control in one combined call (unlike this file's split LateralMPC/MpcLongitudinal
    pair). Speed target is ctx.v_ref, the flat/un-refined reference -- never run through
    refine_speed_preview(), per instruction: the curvature-aware speed cap is specific to the stack
    being evaluated, not the baseline it's compared against.

    No Kalman filter is involved on this branch either way: control_pid() never uses v_y, so there
    is nothing for a v_y estimate to feed -- "vad-pid" is a baseline for the whole stack, not a
    third v_y source."""
    label = "VAD-PID"

    def __init__(self, args):
        self.pid = VadPIDController()

    def reset(self, u=None):
        pass   # no warm-up state of its own to drop (unlike MpcLongitudinal's PID pedal layer)

    def step(self, ctx):
        waypoints, target = _vad_waypoints_and_target(
            ctx.path, ctx.last_s, ctx.ego_x, ctx.ego_y, ctx.yaw, ctx.v_ref)
        steer, throttle, brake, _ = self.pid.control_pid(waypoints, ctx.v_x, target)
        return float(steer), float(throttle), bool(brake)


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
    see the module docstring's "Step 5" for the causality/noise details. "vad-pid" takes neither
    branch: it runs `controller` (a VadPidController) as one combined lateral+longitudinal call, so
    neither LateralMPC nor the filter is built for it at all.

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
    _, lf, lr, max_steer = get_vehicle_geometry(vehicle, spawn_transform)

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

            if controller_key in ("mpc", "mpc-kf"):
                # v_y source for x0, computed BEFORE x0 itself: "mpc-kf" needs its filter's predict()+
                # update() to have already run this tick (using vx measured just above and
                # prev_delta_for_kf -- last tick's delta, since this tick's doesn't exist until
                # lateral_mpc.solve() below computes it) -- see module docstring's Step 5. x0's r stays
                # the real gyro reading either way, only v_y is swapped.
                if controller_key == "mpc-kf":
                    r_meas = r + kf_rng.normal(0.0, math.radians(args.kf_gyro_std))
                    ay_meas = a_y + kf_rng.normal(0.0, args.kf_accel_std)
                    kf.step(v_x, prev_delta_for_kf, [r_meas, ay_meas])
                    v_y_for_x0 = kf.v_y
                else:
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
                x0 = [v_y_for_x0, r, raw_e_y, -e_theta]
                delta = lateral_mpc.solve(x0, vx_preview, kappa_preview)
                prev_delta_for_kf = delta   # this tick's delta becomes next tick's "already applied"

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


def main():
    parser = argparse.ArgumentParser()

    # ---- simulation setting ---- #
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--times-run", type=float, default=5.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=100.0, help="scored run length (s)")
    parser.add_argument("--spawn-x", type=float, default=-90.0,
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
    parser.add_argument("--step-size", type=float, default=-10.0,
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
    lat.add_argument("--w-ey", type=float, default=10.0, help="cross-track error weight")
    lat.add_argument("--w-epsi", type=float, default=10.0, help="heading error weight")
    lat.add_argument("--w-ay", type=float, default=1,
                     help="lateral acceleration tracking weight -- default 0 (see mpc_mpc.py's "
                          "LateralMPC docstring: forcing a_y/r/r_dot toward the steady-turn "
                          "feedforward fights e_y/e_psi's own targets in a curve and was measured "
                          "to cost ~2m of steady cross-track offset before this was found)")
    lat.add_argument("--w-r", type=float, default=3, help="yaw rate tracking weight (see --w-ay)")
    lat.add_argument("--w-rdot", type=float, default=30,
                     help="yaw acceleration tracking weight (see --w-ay). Lowered from 120: rdot is "
                          "formed as D*delta with D = lf*Cf/Iz = 33.3, so the effective penalty on "
                          "the input is w_rdot*D^2 -- 120 gave 1.3e5 against w_delta = 1, which "
                          "throttled the steering response enough to cost route completion.")
    lat.add_argument("--w-delta", type=float, default=1, help="steer magnitude weight")
    lat.add_argument("--w-ddelta", type=float, default=30,
                     help="steer rate weight -- raised from 1 in the same B2D-penalty search that "
                          "set --ay-max: with the curve-speed cap doing most of the comfort work, a "
                          "stiffer rate cost here trims the rest without hurting lateral_error")
    lat.add_argument("--delta-max-deg", type=float, default=30.0,
                     help="hard steer-magnitude limit (deg, bicycle-model wheel angle) -- kept "
                          "well under max_steer since Cf/Cr are only valid over the range they "
                          "were calibrated on")
    lat.add_argument("--ddelta-max-deg", type=float, default=70.0, help="hard steer-rate limit (deg/s)")
    lat.add_argument("--mass", type=float, default=VEHICLE_DEFAULTS["mass"], help="vehicle mass (kg)")
    lat.add_argument("--iz", type=float, default=VEHICLE_DEFAULTS["iz"], help="yaw moment of inertia (kg m^2)")
    lat.add_argument("--cf", type=float, default=VEHICLE_DEFAULTS["cf"], help="front cornering stiffness (N/rad)")
    lat.add_argument("--cr", type=float, default=VEHICLE_DEFAULTS["cr"], help="rear cornering stiffness (N/rad)")

    # ---- controller selection ---- #
    parser.add_argument("--controller", nargs="+", default=["mpc"],
                        choices=("mpc", "mpc-kf", "vad-pid"),
                        help="which controller(s) to run this route with. One controller (default): "
                             "unchanged single-trial output (3 figures). Two or more: each runs its "
                             "own trial and results are compared instead (trajectory overlay + "
                             "8-metric comparison figure + each trial's own lateral/longitudinal pair). "
                             "'mpc' and 'mpc-kf' run the identical LateralMPC/MpcLongitudinal pair -- "
                             "'mpc' feeds x0 CARLA's own ground-truth v_y, 'mpc-kf' feeds it "
                             "VyKalmanFilter's online estimate instead (kalman_filter.py); running both "
                             "is the closed-loop verification this file exists for -- see module docstring. "
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

    controller_labels = {"mpc": "MPC", "mpc-kf": "MPC-KF", "vad-pid": "VAD-PID"}
    multi = len(args.controller) > 1
    results = {}
    videos = []   # [(label, mp4 경로, 프레임 수)] -- --controller 순서 유지
    try:
        for key in args.controller:
            # "mpc" and "mpc-kf" both drive through MpcLongitudinal; "vad-pid" replaces the whole
            # split pair with its own combined controller instead.
            controller = VadPidController(args) if key == "vad-pid" else MpcLongitudinal(args)
            print(f"\n=== running controller: {controller_labels[key]} ===")
            hist, video_meta = run_trial(world, spawn_transform, path_x, path_y, path, blueprint,
                                         imu_bp, controller, key, args,
                                         spawn_to_start_m=spawn_to_start_m,
                                         video_suffix=key if multi else "")
            if hist and hist["t"]:
                results[controller_labels[key]] = hist
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
                              title="MPC-KF: v_y estimate + sensor noise")
            os.makedirs(args.plot_dir, exist_ok=True)
            out_path = os.path.join(args.plot_dir, run_name("kf") + ".png")
            fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            print(f"Figure saved: {out_path}")
        except Exception as exc:
            print(f"MPC-KF report plotting failed: {exc}")


if __name__ == "__main__":
    main()
