r"""Lateral LPV-MPC, built from scratch (mpc_mpc.py is left untouched, kept only as a reference).

Step 1: get the route as flat coordinate arrays and precompute station (cumulative arc length) and
signed curvature along it, the way mpc_mpc.py's build_path_curvature() does -- but written here
rather than imported, since this file is not reusing mpc_mpc.py's code, only its design.

Step 2: spawn the vehicle and drive it longitudinally under longitudinal_mpc.SpeedMPC.
MpcLongitudinal below is this file's own copy of stanley_mpc.MpcLongitudinal's wrapper (SpeedMPC ->
a_cmd -> LUT+PID pedal layer), written here rather than imported for the same reason
build_path_station/build_path_curvature are: this file reuses the *design*, not stanley_mpc.py's
code. The one piece actually imported is SpeedMPC itself (the QP), plus the reference-profile
functions that used to live in longitudinal_mpc.py and were moved to functions.py so this file (and
any other) can share them without copying.

Step 3: LateralMPC -- the LPV-MPC QP itself (condensed state-space, output tracking, box/rate
constraints), per the derivation given for this file.

Step 4: wire LateralMPC into run_trial(). Which speed plan feeds its curvature preview matters: it
schedules against MpcLongitudinal's own condensed prediction (ctx.v_x_preview, stashed in Step 2),
not the raw reference profile -- the reference ignores a_min/a_max and so can promise a station the
car physically cannot reach yet, which would desync the previewed curvature from where the vehicle
will actually be. Final control goes through functions.control_input, not a hand-rolled
carla.VehicleControl -- see its own docstring for why (Ackermann inner/outer split + steering-curve
speed scaling).


Tuning notes (from actually driving this route, Town10HD_Opt idx 0->100): w_ey/w_epsi/w_ddelta and
--lat-np were retuned (60/15/4, Np 20->30) after the defaults tracked poorly through two ~10-13m
radius corners around path idx~90-150 -- but weight/horizon changes only moved cross-track RMSE a
little (1.25m -> 1.22m at 5 m/s). The dominant lever turned out to be speed, not the QP: a_y=v^2/r
at that radius is ~2.5 m/s^2 at 5 m/s but ~6.4 m/s^2 at 8 m/s, which is a mismatch no amount of
lateral-only retuning fixes -- 6 and 8 m/s both ran off-road there even with a widened preview.
--initial-speed therefore defaults to 5, not because the QP can't be pushed harder, but because
this decoupled architecture has no curvature-aware speed reduction feeding back from the lateral
side into SpeedMPC's reference -- that would be the real fix, and is future work, not a tuning knob.

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe mpc_mpc1.py --profile constant --initial-speed 5
    .venv\Scripts\python.exe mpc_mpc1.py --profile sine --initial-speed 5 --sine-amplitude 2 --sine-period 5
    .venv\Scripts\python.exe mpc_mpc1.py --profile step --initial-speed 5 --step-size 2 --step-time 10
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

HERE = os.path.dirname(os.path.abspath(__file__))

CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/carla/CARLA_0.9.15"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.append(os.path.join(HERE, "longitudinal_lookup"))

import carla

from functions import (AngleUnwrapper, ImuAcceleration, LowPassFilter, build_path, control_input,
                       get_vehicle_geometry, lateral_error, normalize_angle, reference_preview,
                       speed_reference)
from longitudinal_lut import LongitudinalLUT
from lookup_controller import LookupController
from longitudinal_mpc import SpeedMPC
from viz_utils import follow_with_spectator, plot_results, print_error_summary

# Fixed on purpose, same as stanley_mpc.py: the route (build_path()'s default origin/dest spawn
# indices) is a property of this specific map, not something to rediscover via CLI flags.
MAP_NAME = "Town10HD_Opt"

WARM_START_SPEED_TOL = 0.3   # m/s
WARM_START_ACCEL_TOL = 0.5   # m/s^2
WARM_START_TIMEOUT = 15.0    # s -- safety cap in case initial_speed is unreachable


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


def build_path_station(path_x, path_y):
    """Cumulative arc length (station, m) at each path point: path_s[0] = 0, path_s[i] = path_s[i-1]
    + the Euclidean distance from point i-1 to i. Precomputed once over the whole route so later
    lookups (nearest-point search, curvature preview) only ever walk forward from wherever the
    vehicle already is, instead of re-scanning the path every step."""
    path_s = [0.0]
    for i in range(1, len(path_x)):
        path_s.append(path_s[-1] + math.hypot(path_x[i] - path_x[i - 1], path_y[i] - path_y[i - 1]))
    return path_s


def build_path_curvature(path_yaw, path_s, stencil=5):
    """Signed curvature (1/m) at each path point: kappa_i = d(path_yaw)/ds via a centred finite
    difference over arc length. Reuses the already-unwrapped path_yaw build_path() produces and the
    already-computed path_s, rather than re-deriving heading or arc length from scratch."""
    n = len(path_yaw)
    path_kappa = []
    for i in range(n):
        a = max(0, i - stencil)
        b = min(n - 1, i + stencil)
        ds = path_s[b] - path_s[a]
        path_kappa.append(0.0 if ds < 1e-6 else (path_yaw[b] - path_yaw[a]) / ds)
    return path_kappa


def curvature_preview(path_s, path_kappa, last_idx, vx_preview, dt):
    """Length-Np curvature preview: path_kappa is indexed by path point (space, ~sampling_resolution
    apart), but the QP needs it indexed by prediction step (time, dt apart) -- two different axes
    that only line up once you know how far the vehicle will actually travel each step. Walk forward
    along the path by the previewed speed each step (station s += v_x * dt, starting from last_idx,
    the vehicle's current position) and read off path_kappa at whatever point that lands on.

    Re-run every cycle, not cached: both last_idx (the vehicle keeps moving) and vx_preview (a new
    speed plan every cycle) change, so the space<->time mapping this builds is only valid for the
    one cycle it was built in."""
    n = len(path_s)
    idx = last_idx
    s_cursor = path_s[last_idx]
    kappa_prev = []
    for vx in vx_preview:
        s_cursor += max(vx, 0.0) * dt
        while idx < n - 1 and path_s[idx + 1] < s_cursor:
            idx += 1
        kappa_prev.append(path_kappa[idx])
    return kappa_prev


# ----------------------------------------------------------------------------- lateral MPC (LPV)

class LateralMPC:
    r"""LPV-MPC lateral controller over the 2-DOF bicycle-model error dynamics.

        x = [v_y, r, e_y, e_psi]^T,  input delta (front steer, rad),  disturbance kappa (path
        curvature, 1/m), r = yaw rate:

        x_dot = A(v_x) x + B delta + E(v_x) kappa

            A(v_x) = [[-(Cf+Cr)/(m vx),        (lr Cr - lf Cf)/(m vx) - vx, 0, 0],
                      [(lr Cr - lf Cf)/(Iz vx), -(lf^2 Cf + lr^2 Cr)/(Iz vx), 0, 0],
                      [1,                       0,                            0, vx],
                      [0,                       1,                            0, 0]]
            B = [Cf/m, lf Cf/Iz, 0, 0]^T                E(v_x) = [0, 0, 0, -vx]^T

    v_x is the scheduling parameter (previewed over the horizon, not held fixed), so A/E/C below are
    re-linearized and re-discretized at every predicted step -- the "LPV" in the name.

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

    Cost J = (Y-Yref)' W1 (Y-Yref) + U' W2 U + (GU)' W3 (GU), G the (Nc-1) x Nc first-difference
    operator on U (steer rate). Reduces to the box/rate-constrained QP

        min 1/2 U' H U + f' U   s.t.  -delta_max <= U <= delta_max,  -ddelta_max <= GU <= ddelta_max
        H = 2[(CbarBbar+Dbar)' W1 (CbarBbar+Dbar) + W2 + G' W3 G]
        f = 2(CbarBbar+Dbar)' W1 (Cbar Abar x0 + Cbar Ebar K - Yref)

    Only the first element of U* is applied each cycle (receding horizon). H depends on the v_x
    preview (changes every cycle), so the QP is rebuilt from scratch on every solve() rather than
    reused with just q/l/u updated the way a fixed-A QP would be.
    """

    N_X = 4   # [v_y, r, e_y, e_psi]
    N_Y = 5   # [e_y, e_psi, a_y, r, r_dot]

    def __init__(self, dt, n_p, n_c, mass, Iz, lf, lr, Cf, Cr,
                w_ey, w_epsi, w_ay, w_r, w_rdot, w_delta, w_ddelta,
                delta_max, ddelta_max, vx_floor=0.5):
        if not (0 < n_c <= n_p):
            raise ValueError(f"need 0 < n_c <= n_p, got n_c={n_c}, n_p={n_p}")
        if not (delta_max > 0 and ddelta_max > 0):
            raise ValueError(f"need delta_max, ddelta_max > 0, got {delta_max}, {ddelta_max}")

        self.dt, self.n_p, self.n_c = dt, n_p, n_c
        self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr = mass, Iz, lf, lr, Cf, Cr
        self.delta_max, self.ddelta_max, self.vx_floor = delta_max, ddelta_max, vx_floor

        self.B_cont = np.array([[Cf / mass], [lf * Cf / Iz], [0.0], [0.0]])
        self.D = np.array([[0.0], [0.0], [Cf / mass], [0.0], [lf * Cf / Iz]])

        self.M_c = self._build_hold_matrix(n_p, n_c)
        self.G = self._build_rate_matrix(n_c)

        self.W1 = np.diag(np.tile([w_ey, w_epsi, w_ay, w_r, w_rdot], n_p))
        self.W2 = w_delta * np.eye(n_c)
        self.W3 = w_ddelta * np.eye(max(n_c - 1, 0))
        self._GtW3G = self.G.T @ self.W3 @ self.G

        A_ineq = np.vstack([np.eye(n_c), self.G])
        self._A_ineq = sparse.csc_matrix(A_ineq)
        rows_rate = max(n_c - 1, 0)
        self._l = np.concatenate([-delta_max * np.ones(n_c), -ddelta_max * np.ones(rows_rate)])
        self._u = np.concatenate([delta_max * np.ones(n_c), ddelta_max * np.ones(rows_rate)])

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
        """(Nc-1) x Nc first-difference operator: (G U)_l = delta_{l+1} - delta_l."""
        rows = max(n_c - 1, 0)
        G = np.zeros((rows, n_c))
        for l in range(rows):
            G[l, l] = -1.0
            G[l, l + 1] = 1.0
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
        Ad, Ed, Ck = [], [], []
        eye = np.eye(self.N_X)
        for vx in vx_preview:
            A, E, C = self._continuous(vx)
            Ad.append(eye + self.dt * A)
            Ed.append(self.dt * E)
            Ck.append(C)
        return Ad, Ed, Ck

    def _condense(self, Ad, Ed, Ck):
        n_p, n_x, n_y = self.n_p, self.N_X, self.N_Y
        A_bar = np.zeros((n_p * n_x, n_x))
        B_full = np.zeros((n_p * n_x, n_p))
        E_bar = np.zeros((n_p * n_x, n_p))

        prod = np.eye(n_x)
        for k in range(n_p):
            prod = Ad[k] @ prod
            A_bar[k * n_x:(k + 1) * n_x, :] = prod

        for j in range(n_p):
            col_b = self.B_cont.copy()
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

        Ad, Ed, Ck = self._discretize(vx_preview)
        A_bar, B_bar, E_bar, C_bar, D_bar = self._condense(Ad, Ed, Ck)

        K = kappa_preview.reshape(-1, 1)
        y_ref = np.zeros((self.n_p * self.N_Y, 1))
        y_ref[2::self.N_Y, 0] = vx_preview**2 * kappa_preview   # a_y_ref
        y_ref[3::self.N_Y, 0] = vx_preview * kappa_preview      # r_ref

        M = C_bar @ B_bar + D_bar
        H = 2.0 * (M.T @ self.W1 @ M + self.W2 + self._GtW3G)
        H = 0.5 * (H + H.T)
        f = 2.0 * M.T @ self.W1 @ (C_bar @ A_bar @ x0 + C_bar @ E_bar @ K - y_ref)

        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(H), q=f.ravel(), A=self._A_ineq, l=self._l, u=self._u,
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


# ----------------------------------------------------------------------------- one trial

def run_trial(world, origin_transform, path_x, path_y, path_yaw, path_s, path_kappa, blueprint,
             imu_bp, controller, args):
    """Spawn one vehicle, drive it under `controller` (longitudinal MPC) + LateralMPC (lateral),
    tear it down.

    Same warm-up gate as stanley_mpc.py/longitudinal_mpc.py: launch from rest under the real
    longitudinal controller and hold off on logging until v_x/a_x have actually settled near the
    profile's own t=0 value (initial_speed) instead of faking that starting condition. The lateral
    MPC runs from tick one regardless -- only the *logging* start is gated, same as stanley_mpc.py's
    Stanley half (it has nothing to warm up; steering the whole time is what keeps the car on the
    route while the speed loop settles).
    """
    vehicle = world.spawn_actor(blueprint, origin_transform)
    physics = vehicle.get_physics_control()
    _, lf, lr, _ = get_vehicle_geometry(vehicle, origin_transform)

    # delta_max capped well under max_steer (the wheel's own physical limit, ~70 deg): Cf/Cr were
    # calibrated over an ~8-16 deg range (estimate_cornering_stiffness.py), so the linear tire model
    # this QP is built on stops being valid long before 70 deg -- and separately, max_steer is the
    # INNER wheel's limit (Ackermann), not the bicycle model's single virtual wheel, which needs a
    # smaller angle than the inner wheel for the same turn. --delta-max-deg picks a value inside
    # both limits rather than deriving the (still oversized, relative to the tire model) Ackermann
    # bound.
    lateral_mpc = LateralMPC(
        dt=args.dt, n_p=args.lat_n_p, n_c=args.lat_n_c,
        mass=args.mass, Iz=args.iz, lf=lf, lr=lr, Cf=args.cf, Cr=args.cr,
        w_ey=args.w_ey, w_epsi=args.w_epsi, w_ay=args.w_ay, w_r=args.w_r, w_rdot=args.w_rdot,
        w_delta=args.w_delta, w_ddelta=args.w_ddelta,
        delta_max=math.radians(args.delta_max_deg), ddelta_max=math.radians(args.ddelta_max_deg) * args.dt)

    accel = ImuAcceleration(dt=args.dt)
    yaw_unwrapper = AngleUnwrapper()
    rh_unwrapper = AngleUnwrapper()
    yaw_acc_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    prev_yaw_rate_rad = None
    last_idx = 0

    # matches plot_results()'s expectations (viz_utils.plot_lateral/plot_longitudinal/plot_trajectory)
    hist = {"t": [], "x": [], "y": [], "v_x": [], "v_y": [], "v_des": [], "a_x": [], "jerk": [],
            "a_y": [], "yaw_rate": [], "yaw_acc": [], "jerk_total": [], "steer_deg": [],
            "throttle": [], "brake": [], "e_y": [], "yaw": [], "path_yaw": [], "e_theta": []}
    warmed_up = False
    log_start_i = 0

    imu = None
    try:
        world.tick()

        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)

        steps = int((args.max_duration + WARM_START_TIMEOUT) / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)

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
            jerk, jerk_total = accel.jerk, accel.jerk_total

            yaw_acc = yaw_acc_filter.step(
                0.0 if prev_yaw_rate_rad is None else (r - prev_yaw_rate_rad) / args.dt)
            prev_yaw_rate_rad = r

            # ---- longitudinal (MPC) -- runs first so ctx.v_x_preview is ready for the lateral
            # schedule below (see the "which speed plan" discussion in the module docstring) ---- #
            t = (i - log_start_i) * args.dt
            v_ref = args.initial_speed if not warmed_up else speed_reference(args, t)
            ctx = SimpleNamespace(t=t, v_x=v_x, v_ref=v_ref, a_x=a_x, a_x_raw=a_x_raw,
                                  gear=vehicle.get_control().gear, warmed_up=warmed_up)
            u = controller.step(ctx)

            # ---- lateral (LPV-MPC) ---- #
            last_idx, raw_e_y = lateral_error(ego_x, ego_y, yaw, path_x, path_y, last_idx)
            road_heading = rh_unwrapper.step(path_yaw[last_idx])
            e_theta = normalize_angle(path_yaw[last_idx] - yaw)
            vx_preview = vx_preview_for_lateral(ctx, lateral_mpc.n_p)
            kappa_preview = curvature_preview(path_s, path_kappa, last_idx, vx_preview, args.dt)
            # lateral_error()'s sign is path-minus-vehicle; LateralMPC's error dynamics (see its
            # docstring) were derived vehicle-minus-path, hence the flip.
            x0 = [v_y, r, -raw_e_y, -e_theta]
            delta = lateral_mpc.solve(x0, vx_preview, kappa_preview)

            # Final control: functions.control_input, not a hand-rolled VehicleControl -- it
            # accounts for the Ackermann inner/outer wheel split and the speed-dependent steering
            # curve CARLA applies underneath the command, ~15% more accurate than naive
            # delta/max_steer (see its docstring). No extra low-pass on delta here: the QP already
            # has ddelta_max as a hard constraint, and stacking a filter on top of that was found
            # (in mpc_mpc.py) to add a second, unmodeled lag that self-sustains oscillation.
            control = control_input(u, delta, v_x, vehicle, physics)
            follow_with_spectator(world, vehicle)

            if not warmed_up:
                converged = (abs(v_x - args.initial_speed) < WARM_START_SPEED_TOL
                            and abs(a_x) < WARM_START_ACCEL_TOL)
                timed_out = i * args.dt >= WARM_START_TIMEOUT
                if converged or timed_out:
                    warmed_up = True
                    log_start_i = i
                    controller.reset(u)
                    status = "converged" if converged else f"timed out after {WARM_START_TIMEOUT:.0f}s"
                    print(f"Warm-start {status}: v_x={v_x:.2f} m/s, a_x={a_x:.2f} m/s^2 -- "
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
            hist["v_des"].append(v_ref)
            hist["a_x"].append(a_x)
            hist["jerk"].append(jerk)
            hist["a_y"].append(a_y)
            hist["yaw_rate"].append(yaw_rate_deg)
            hist["yaw_acc"].append(yaw_acc)
            hist["jerk_total"].append(jerk_total)
            hist["steer_deg"].append(math.degrees(delta))
            hist["throttle"].append(control.throttle)
            hist["brake"].append(control.brake)
            hist["e_y"].append(raw_e_y)
            hist["yaw"].append(math.degrees(yaw))
            hist["path_yaw"].append(math.degrees(road_heading))
            hist["e_theta"].append(math.degrees(e_theta))

            if i % 20 == 0:
                print(f"t={t:5.1f}s  v_x={v_x:5.2f}/{v_ref:.2f} m/s  a_cmd={ctx.a_cmd:+.2f} m/s^2  "
                      f"delta={math.degrees(delta):+.2f} deg  e_y={raw_e_y:+.2f} m  "
                      f"idx={last_idx}/{len(path_x) - 1}")

            if last_idx >= len(path_x) - 1:
                print(f"Reached end of path (idx {last_idx}/{len(path_x) - 1}).")
                break
            if t >= args.max_duration:
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    finally:
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        vehicle.destroy()

    return hist


def main():
    parser = argparse.ArgumentParser()

    # ---- simulation setting ---- #
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--times-run", type=float, default=20.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=100.0, help="scored run length (s)")

    # ---- speed profile ---- #
    parser.add_argument("--profile", default="constant", choices=("constant", "sine", "step"),
                        help="speed reference shape: flat initial-speed, a sine wave around it, or "
                             "a step change at --step-time")
    parser.add_argument("--initial-speed", type=float, default=5.0,
                        help="m/s -- kept modest by default: the route's sharpest corners "
                             "(~10-13m radius, idx~90-150) demand a_y=v^2/r that outgrows what "
                             "steering alone can correct for well above this speed (see tuning "
                             "notes above LateralMPC's argparse group)")
    parser.add_argument("--sine-amplitude", type=float, default=3.0, help="sine profile peak deviation (m/s)")
    parser.add_argument("--sine-period", type=float, default=5.0, help="sine profile period (s)")
    parser.add_argument("--step-size", type=float, default=3.0, help="step profile speed change (m/s)")
    parser.add_argument("--step-time", type=float, default=10.0, help="step profile: when it happens (s)")

    # ---- longitudinal MPC ---- #
    mpc = parser.add_argument_group("longitudinal MPC")
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
                     help="low-pass filter time constant on the pedal command u, before it's applied (s)")

    # ---- lateral LPV-MPC ---- #
    lat = parser.add_argument_group("lateral LPV-MPC")
    lat.add_argument("--lat-np", dest="lat_n_p", type=int, default=30,
                     help="lateral prediction horizon (steps) -- 1.5s at dt=0.05, widened from 20 "
                          "so the QP sees the route's ~10-13m-radius corners (Town10HD_Opt, "
                          "idx~90-150 of the default route) before it's on top of them")
    lat.add_argument("--lat-nc", dest="lat_n_c", type=int, default=20, help="lateral control horizon (steps, <= --lat-np)")
    lat.add_argument("--w-ey", type=float, default=60.0, help="cross-track error weight")
    lat.add_argument("--w-epsi", type=float, default=15.0, help="heading error weight")
    lat.add_argument("--w-ay", type=float, default=0,
                     help="lateral acceleration tracking weight -- default 0 (see mpc_mpc.py's "
                          "LateralMPC docstring: forcing a_y/r/r_dot toward the steady-turn "
                          "feedforward fights e_y/e_psi's own targets in a curve and was measured "
                          "to cost ~2m of steady cross-track offset before this was found)")
    lat.add_argument("--w-r", type=float, default=0.0, help="yaw rate tracking weight (see --w-ay)")
    lat.add_argument("--w-rdot", type=float, default=0.0, help="yaw acceleration tracking weight (see --w-ay)")
    lat.add_argument("--w-delta", type=float, default=2, help="steer magnitude weight")
    lat.add_argument("--w-ddelta", type=float, default=4.0,
                     help="steer rate weight -- lowered from a stiffer default so the QP can react "
                          "quickly enough into the sharp corners, at the cost of a slightly less "
                          "smooth command")
    lat.add_argument("--delta-max-deg", type=float, default=30.0,
                     help="hard steer-magnitude limit (deg, bicycle-model wheel angle) -- kept "
                          "well under max_steer since Cf/Cr are only valid over the range they "
                          "were calibrated on")
    lat.add_argument("--ddelta-max-deg", type=float, default=60.0, help="hard steer-rate limit (deg/s)")
    lat.add_argument("--mass", type=float, default=VEHICLE_DEFAULTS["mass"], help="vehicle mass (kg)")
    lat.add_argument("--iz", type=float, default=VEHICLE_DEFAULTS["iz"], help="yaw moment of inertia (kg m^2)")
    lat.add_argument("--cf", type=float, default=VEHICLE_DEFAULTS["cf"], help="front cornering stiffness (N/rad)")
    lat.add_argument("--cr", type=float, default=VEHICLE_DEFAULTS["cr"], help="rear cornering stiffness (N/rad)")

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"),
                        help="directory to save the end-of-run result figures into")
    parser.add_argument("--save-plot", action="store_true",
                        help="draw and save the end-of-run result figures (off by default)")
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

    origin_transform, path_x, path_y, path_yaw = build_path(world)
    path_s = build_path_station(path_x, path_y)
    path_kappa = build_path_curvature(path_yaw, path_s)
    print(f"Route: {len(path_x)} points, "
          f"start=({path_x[0]:.1f}, {path_y[0]:.1f}) goal=({path_x[-1]:.1f}, {path_y[-1]:.1f})")

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    try:
        controller = MpcLongitudinal(args)
        hist = run_trial(world, origin_transform, path_x, path_y, path_yaw, path_s, path_kappa,
                         blueprint, imu_bp, controller, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        hist = None
    finally:
        world.apply_settings(original_settings)
        print("Cleaned up: world settings restored.")

    if hist and hist["t"]:
        if args.save_plot:
            try:
                plot_results(path_x, path_y, hist, args.initial_speed, args.plot_dir, label="LPV-MPC")
            except Exception as exc:
                print(f"Plotting failed: {exc}")
                print_error_summary(hist, args.initial_speed)
        else:
            print_error_summary(hist, args.initial_speed)


if __name__ == "__main__":
    main()
