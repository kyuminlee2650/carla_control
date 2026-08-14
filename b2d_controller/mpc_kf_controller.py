r"""B2D/VAD closed-loop controller: LateralMPC (carla_control's own LPV bicycle-model QP) + a plain
speed-error PID + VyKalmanFilter's v_y estimate, meant to replace team_code/pid_controller.py's
PIDController inside team_code/vad_b2d_agent.py's run_step() (see that file's line ~396:
`self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)`).

Self-contained on purpose (no import of carla_control/ or kalman_filter/): this file has to run
inside the b2d_zoo conda env / the submission .sif, which has no reason to also carry
carla_control's own CARLA_ROOT sys.path setup or its viz/offline-analysis dependencies.
VyKalmanFilter/LateralMPC/PathSpline below are copies of kalman_filter.py's, mpc_mpc_KF.py's, and
functions.py's own math (unchanged) -- see those files for the derivation docstrings; this file's
own docstrings focus on what's DIFFERENT about running them here.

Four things are different from carla_control's own driving scripts, and why:

1. No known map route -- VAD's own ~6-waypoint ego-frame prediction (`out_truck`, already cumsum'd
   from per-step displacements) plus one route-command point (`target`) is all there is each tick.
   build_trajectory_splines() fits a PathSpline (functions.py's own class, copied verbatim -- its
   docstring already anticipated this exact use: "sparser planner output like VAD later") directly
   to those ~8 points instead of a GlobalRoutePlanner route, PLUS a companion vx(s) spline built
   from the waypoints' own known timing (confirmed 0.5s apart -- Bench2DriveZoo/docs/
   CONVERT_GUIDE.md: "Bench2Drive runs at 10Hz... window length... 0.5s"). x0's e_y is 0 and e_psi
   is path.yaw(0) by construction every tick (ego is redefined as the origin of a fresh ego-frame
   plan every cycle, not carried forward against a persistent map route).

2. No get_physics_control(). carla_control's functions.control_input() reads the vehicle's own
   max_steer_angle and speed-dependent steering_curve to convert a delta (rad) into CARLA's
   normalized VehicleControl.steer -- a leaderboard agent can't call get_physics_control() at all.
   steer_from_delta() below approximates it as a flat delta / MAX_STEER_ANGLE scale, ignoring the
   curve's own speed-dependent softening. A known simplification, not a bug -- see its docstring.

3. Vehicle params are hardcoded (mass/Iz/Cf/Cr/lf/lr), not read from get_physics_control() or
   re-measured -- same reason as #2, and per instruction: reuse carla_control's own
   lateral_parameter/yaw_inertia.json values outright, on the assumption B2D's ego vehicle is the
   same vehicle.lincoln.mkz_2020 carla_control was calibrated against.

4. vx_preview now comes from the vx(s) spline (see #1) instead of a flat/ramped guess. This isn't
   just "more accurate" -- an early version that flat-held vx_preview at the CURRENT (often low)
   speed for all n_p steps hit a real numerical failure: LateralMPC's condensed A_bar/B_bar involve
   up to n_p products of Ad, whose eigenvalues scale like 1/vx, so a floored Ad raised to the 20th
   power overflowed into eigenvalues around 1e18-1e36 (measured) and OSQP reported "non convex"/
   "primal infeasible" almost every tick. A real speed profile (even this coarse a fit) keeps most
   of the horizon off the floor as soon as the plan says to actually accelerate, which fixes it.

Coordinate convention: VAD's own waypoints/target are [lateral, forward] (index 0 sideways, index 1
ahead) -- confirmed by pid_controller.py's own `angle = degrees(pi/2 - atan2(aim[1], aim[0]))`
formula, which only zeroes out on a straight-ahead point under that ordering. Every function below
takes/returns that native [lateral, forward] ordering at its public boundary (to keep the
vad_b2d_agent.py call site a near-drop-in replacement for control_pid()) and converts internally to
this file's own [forward, lateral] convention (matching carla_control's C(v_x)/x0/PathSpline sign
convention) wherever the math needs it.
"""
import math
from collections import deque

import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import UnivariateSpline

# vehicle.lincoln.mkz_2020's own front-wheel lock, matches carla_control/functions.py's
# MAX_STEER_ANGLE constant -- see module docstring point 2 for why this can't be read live here.
MAX_STEER_ANGLE_RAD = math.radians(70.0)

# carla_control/lateral_parameter/yaw_inertia.json's own measured values -- hardcoded per
# instruction, see module docstring point 3.
VEHICLE_MASS = 1696.0
VEHICLE_IZ = 2916
VEHICLE_CF = 82941.0
VEHICLE_CR = 54425.0
VEHICLE_LF = 1.1718419429705567
VEHICLE_LR = 1.6886329628368866

DT = 0.05            # control period, matches sensors()'s IMU sensor_tick and CARLA's fixed_delta_seconds
VX_FLOOR = 0.5        # same floor carla_control's LateralMPC/VyKalmanFilter use
VAD_WP_DT = 0.5        # seconds between VAD's own future-trajectory steps -- confirmed, see module docstring


# ----------------------------------------------------------------------------- VyKalmanFilter
# Copy of kalman_filter/kalman_filter.py's VyKalmanFilter -- unchanged math, including the
# vx_floor low-speed gate (see that file's step() docstring for why it's there: below vx_floor the
# model has no vx dependence in B, so a held delta_prev re-injects the same "phantom" lateral
# velocity every tick and the low-vx-weakened measurement update can't cancel it back out --
# asserting v_y=0/r=z[0] there is the physically correct answer, not a fallback).

class VyKalmanFilter:
    N_X = 2   # [v_y, r]
    N_Z = 2   # [dpsi_meas, ay_meas]

    def __init__(self, dt, mass, Iz, lf, lr, Cf, Cr, Q, R, vx_floor=0.5, x0=None, P0=None):
        self.dt = dt
        self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr = mass, Iz, lf, lr, Cf, Cr
        self.vx_floor = vx_floor
        self.B_cont = np.array([[Cf / mass], [lf * Cf / Iz]])
        self.B_disc = self.dt * self.B_cont
        self.D = np.array([[0.0], [Cf / mass]])
        self.Q = np.asarray(Q, dtype=float).reshape(self.N_X, self.N_X)
        self.R = np.asarray(R, dtype=float).reshape(self.N_Z, self.N_Z)
        self.x = np.zeros((self.N_X, 1)) if x0 is None else np.asarray(x0, dtype=float).reshape(self.N_X, 1)
        self.P = np.eye(self.N_X) if P0 is None else np.asarray(P0, dtype=float).reshape(self.N_X, self.N_X)

    def _continuous_A(self, vx):
        vx = max(vx, self.vx_floor)
        m, Iz, lf, lr, Cf, Cr = self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr
        return np.array([
            [-(Cf + Cr) / (m * vx),           (lr * Cr - lf * Cf) / (m * vx) - vx],
            [(lr * Cr - lf * Cf) / (Iz * vx), -(lf**2 * Cf + lr**2 * Cr) / (Iz * vx)],
        ])

    def _H(self, vx):
        vx = max(vx, self.vx_floor)
        m, lf, lr, Cf, Cr = self.mass, self.lf, self.lr, self.Cf, self.Cr
        return np.array([
            [0.0, 1.0],
            [-(Cf + Cr) / (m * vx), (lr * Cr - lf * Cf) / (m * vx)],
        ])

    def predict(self, vx, delta_prev):
        Ad = np.eye(self.N_X) + self.dt * self._continuous_A(vx)
        self.x = Ad @ self.x + self.B_disc * delta_prev
        self.P = Ad @ self.P @ Ad.T + self.Q
        return self.x.ravel()

    def update(self, vx, delta_prev, z):
        H = self._H(vx)
        z = np.asarray(z, dtype=float).reshape(self.N_Z, 1)
        y = z - (H @ self.x + self.D * delta_prev)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.N_X) - K @ H) @ self.P
        return self.x.ravel()

    def step(self, vx, delta_prev, z):
        if vx < self.vx_floor:
            self.x = np.array([[0.0], [z[0]]])
            self.P = np.diag([1e-6, self.R[0, 0]])
            return self.x.ravel()
        self.predict(vx, delta_prev)
        return self.update(vx, delta_prev, z)

    @property
    def v_y(self):
        return float(self.x[0, 0])

    @property
    def r(self):
        return float(self.x[1, 0])


# ----------------------------------------------------------------------------- LateralMPC
# Copy of mpc_mpc_KF.py's LateralMPC -- unchanged math. See that file for the full QP derivation
# docstring; kept terse here since it's not new.

class LateralMPC:
    N_X = 4   # [v_y, r, e_y, e_psi]
    N_Y = 5   # [e_y, e_psi, a_y, r, r_dot]

    def __init__(self, dt, n_p, n_c, mass, Iz, lf, lr, Cf, Cr,
                w_ey, w_epsi, w_ay, w_r, w_rdot, w_delta, w_ddelta,
                delta_max, ddelta_max, vx_floor=0.5):
        self.dt, self.n_p, self.n_c = dt, n_p, n_c
        self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr = mass, Iz, lf, lr, Cf, Cr
        self.delta_max, self.ddelta_max, self.vx_floor = delta_max, ddelta_max, vx_floor

        self.B_cont = np.array([[Cf / mass], [lf * Cf / Iz], [0.0], [0.0]])
        self.B_disc = dt * self.B_cont
        self.D = np.array([[0.0], [0.0], [Cf / mass], [0.0], [lf * Cf / Iz]])

        self.M_c = self._build_hold_matrix(n_p, n_c)
        self.G = self._build_rate_matrix(n_c)
        self.W1 = np.diag(np.tile([w_ey, w_epsi, w_ay, w_r, w_rdot], n_p))
        self.W2 = w_delta * np.eye(n_c)
        self.W3 = w_ddelta * np.eye(n_c)
        self._GtW3G = self.G.T @ self.W3 @ self.G

        A_ineq = np.vstack([np.eye(n_c), self.G])
        self._A_ineq = sparse.csc_matrix(A_ineq)
        self._delta_l = -delta_max * np.ones(n_c)
        self._delta_u = delta_max * np.ones(n_c)

        self.last_solution = np.zeros(n_c)
        self.last_status = "unsolved"

    @staticmethod
    def _build_hold_matrix(n_p, n_c):
        M = np.zeros((n_p, n_c))
        for i in range(n_p):
            M[i, min(i, n_c - 1)] = 1.0
        return M

    @staticmethod
    def _build_rate_matrix(n_c):
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
            col_b = self.B_disc.copy()
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
        vx_preview = np.clip(np.asarray(vx_preview, dtype=float), self.vx_floor, None)
        kappa_preview = np.asarray(kappa_preview, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1, 1)

        Ad, Ed, Ck = self._discretize(vx_preview)
        A_bar, B_bar, E_bar, C_bar, D_bar = self._condense(Ad, Ed, Ck)

        K = kappa_preview.reshape(-1, 1)
        y_ref = np.zeros((self.n_p * self.N_Y, 1))
        y_ref[2::self.N_Y, 0] = vx_preview**2 * kappa_preview
        y_ref[3::self.N_Y, 0] = vx_preview * kappa_preview

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
            u = self.last_solution
        else:
            u = result.x
            self.last_solution = u.copy()

        return float(np.clip(u[0], -self.delta_max, self.delta_max))


# ----------------------------------------------------------------------------- PathSpline
# Copy of carla_control/functions.py's PathSpline + _chord_length_station -- unchanged math. Its
# own docstring already anticipated exactly this use ("sparser planner output like VAD later").

def _chord_length_station(path_x, path_y):
    s = [0.0]
    for i in range(1, len(path_x)):
        s.append(s[-1] + math.hypot(path_x[i] - path_x[i - 1], path_y[i] - path_y[i - 1]))
    return np.asarray(s, dtype=float)


class PathSpline:
    """Arc-length-parameterized SMOOTHING cubic spline fit of a path: x(s), y(s). yaw(s) and
    kappa(s) are analytic derivatives of that fit -- no finite-difference stencil. See
    carla_control/functions.py's own PathSpline for the full docstring; unchanged here."""

    def __init__(self, path_x, path_y, smoothing=1.0, min_spacing=0.01):
        s = _chord_length_station(path_x, path_y)
        px = np.asarray(path_x, dtype=float)
        py = np.asarray(path_y, dtype=float)
        keep = np.concatenate([[True], np.diff(s) > min_spacing])
        s, px, py = s[keep], px[keep], py[keep]

        self.s_min, self.s_max = float(s[0]), float(s[-1])
        k = min(3, len(s) - 1)   # UnivariateSpline needs k < number of points; degrade gracefully
        self._sx = UnivariateSpline(s, px, k=k, s=smoothing)
        self._sy = UnivariateSpline(s, py, k=k, s=smoothing)

    def xy(self, s):
        return self._sx(s), self._sy(s)

    def yaw(self, s):
        return np.arctan2(self._sy(s, 1), self._sx(s, 1))

    def kappa(self, s):
        dx, dy = self._sx(s, 1), self._sy(s, 1)
        ddx, ddy = self._sx(s, 2), self._sy(s, 2)
        denom = (dx * dx + dy * dy) ** 1.5
        return (dx * ddy - dy * ddx) / np.maximum(denom, 1e-9)


def build_trajectory_splines(waypoints, target, dt_wp=VAD_WP_DT, smoothing_xy=0.05, smoothing_v=0.5):
    r"""Fit a PathSpline (x(s)/y(s)/yaw(s)/kappa(s)) + a companion vx(s) spline directly from VAD's
    own ~6 waypoints + 1 route-command target point, in place of carla_control's
    PathSpline-over-a-known-route. See module docstring point 1 for why.

    waypoints: VAD's out_truck, (N, 2) in its own [lateral, forward] ego-frame convention (index 0
    sideways, index 1 ahead), ego at the origin, heading along +forward, each step dt_wp seconds
    after the last. target: local_command_xy, same convention, one further point with an unknown
    timestamp (a route-command aim point, not a VAD-timed prediction) -- included in the geometry
    fit for shape only, excluded from the speed fit.

    Returns (path, vx_spline, s_max_wp): path is a PathSpline in this file's own (forward, lateral)
    convention; vx_spline(s) -> m/s is a plain UnivariateSpline (speed isn't a geometric property of
    the path, so it gets its own fit, not a PathSpline derivative); s_max_wp is the last real
    waypoint's station -- vx_spline is only informed by data up to there, beyond it (out to
    target's station) callers should hold the last sample rather than trust extrapolation.
    """
    fwd = [0.0] + [wp[1] for wp in waypoints] + [target[1]]
    lat = [0.0] + [wp[0] for wp in waypoints] + [target[0]]
    path = PathSpline(fwd, lat, smoothing=smoothing_xy)

    s_all = _chord_length_station(fwd, lat)
    n_wp = len(waypoints)
    s_wp = s_all[:n_wp + 1]              # station at the origin + each real waypoint
    t_wp = np.arange(n_wp + 1) * dt_wp   # 0, dt_wp, 2*dt_wp, ... -- known VAD cadence

    # One speed sample per waypoint interval, assigned to that interval's own arc-length midpoint
    # (an average speed over [s_{i-1}, s_i] describes the midpoint, not either edge).
    ds = np.diff(s_wp)
    dt = np.diff(t_wp)
    v_seg = ds / np.maximum(dt, 1e-3)
    s_mid = s_wp[:-1] + ds / 2.0

    order = np.argsort(s_mid)
    s_mid, v_seg = s_mid[order], v_seg[order]
    keep = np.concatenate([[True], np.diff(s_mid) > 1e-3])
    s_mid, v_seg = s_mid[keep], v_seg[keep]

    k = min(3, len(s_mid) - 1) if len(s_mid) >= 2 else 0
    vx_spline = UnivariateSpline(s_mid, v_seg, k=max(k, 1), s=smoothing_v) if len(s_mid) >= 2 \
        else (lambda s, _v=float(v_seg[0]) if len(v_seg) else 0.0: np.full_like(np.asarray(s, dtype=float), _v))

    return path, vx_spline, float(s_wp[-1])


def preview_from_splines(path, vx_spline, s_max_wp, n_p, dt, vx_floor=VX_FLOOR):
    """Walk a station cursor forward by vx(s_cursor)*dt each step (same technique
    mpc_mpc_KF.py's curvature_preview() uses, extended to also sample vx(s) instead of taking it
    as a given input) -- produces (vx_preview, kappa_preview), both length n_p, off the SAME
    station parameterization so they stay consistent with each other tick to tick."""
    s_cursor = 0.0
    vx_preview = np.zeros(n_p)
    kappa_preview = np.zeros(n_p)
    for j in range(n_p):
        s_kappa = min(s_cursor, path.s_max)
        s_v = min(s_cursor, s_max_wp)
        kappa_preview[j] = float(path.kappa(s_kappa))
        vx_preview[j] = max(float(vx_spline(s_v)), vx_floor)
        s_cursor += vx_preview[j] * dt
    return vx_preview, kappa_preview


def steer_from_delta(delta_rad):
    """delta (rad, LateralMPC's own front-wheel bicycle-model angle) -> CARLA's normalized
    VehicleControl.steer in [-1, 1]. See module docstring point 2 for why this is a flat scale
    (MAX_STEER_ANGLE_RAD) rather than the real speed-dependent steering_curve carla_control's own
    functions.control_input() applies -- no get_physics_control() access here to read that curve."""
    return float(np.clip(delta_rad / MAX_STEER_ANGLE_RAD, -1.0, 1.0))


class SimplePID:
    """Minimal windowed PID -- same shape as team_code/pid_controller.py's own PID class,
    duplicated (not imported) so this file has no dependency on that module and can be dropped
    into the .sif build on its own."""

    def __init__(self, kp, ki, kd, n=20):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._window = deque([0.0] * n, maxlen=n)

    def step(self, error):
        self._window.append(error)
        integral = float(np.mean(self._window))
        derivative = self._window[-1] - self._window[-2] if len(self._window) >= 2 else 0.0
        return self.kp * error + self.ki * integral + self.kd * derivative


# ----------------------------------------------------------------------------- the drop-in controller

class MpcKfController:
    r"""Drop-in replacement for team_code/pid_controller.py's PIDController. Same
    control_pid(waypoints, speed, target) call signature PLUS r_meas/ay_meas appended -- the two
    IMU channels (gyro z, accelerometer y) PIDController never needed that this file's v_y
    estimator does. vad_b2d_agent.py's call site needs exactly two lines changed: instantiate this
    instead of PIDController, and pass tick_data['angular_velocity'][2] /
    tick_data['acceleration'][1] as the two extra args -- already unpacked into local variables for
    the can_bus feature every tick, no new sensor wiring needed.

    No synthetic measurement noise is added anywhere in this file (unlike kalman_filter.py's
    offline replay or mpc_mpc_KF.py's closed-loop comparison harness, which both inject Gaussian
    noise deliberately to make R worth tuning against): this is the real deployed agent, so it
    consumes B2D's own IMU signal as-is. kf_r_dpsi/kf_r_ay below default to the same near-zero
    values carla_control's own `kalman_filter.py --tune` search converged to under a --gyro-std 0
    --accel-std 0 (no synthetic noise) run -- with the true sensor this clean, minimal process
    noise was the RMSE-minimizing setting, and this deployment is exactly that same regime.
    """

    def __init__(self, dt=DT, n_p=20, n_c=20,
                w_ey=4000.0, w_epsi=100.0, w_ay=0.0, w_r=0.0, w_rdot=0.0,
                w_delta=1.0, w_ddelta=30.0, delta_max_deg=30.0, ddelta_max_deg=70.0,
                kf_q_vy=1e-8, kf_q_r=1e-8, kf_r_dpsi=1e-5, kf_r_ay=1e-5,
                speed_kp=1.0, speed_ki=0.3, speed_kd=0.0, max_throttle=0.75,
                brake_speed=0.4, brake_ratio=1.1, clip_delta=0.25):
        self.dt, self.n_p = dt, n_p
        self.lateral_mpc = LateralMPC(
            dt=dt, n_p=n_p, n_c=n_c, mass=VEHICLE_MASS, Iz=VEHICLE_IZ, lf=VEHICLE_LF, lr=VEHICLE_LR,
            Cf=VEHICLE_CF, Cr=VEHICLE_CR, w_ey=w_ey, w_epsi=w_epsi, w_ay=w_ay, w_r=w_r, w_rdot=w_rdot,
            w_delta=w_delta, w_ddelta=w_ddelta,
            delta_max=math.radians(delta_max_deg), ddelta_max=math.radians(ddelta_max_deg) * dt,
            vx_floor=VX_FLOOR)
        self.kf = VyKalmanFilter(dt=dt, mass=VEHICLE_MASS, Iz=VEHICLE_IZ, lf=VEHICLE_LF, lr=VEHICLE_LR,
                                 Cf=VEHICLE_CF, Cr=VEHICLE_CR, Q=np.diag([kf_q_vy, kf_q_r]),
                                 R=np.diag([kf_r_dpsi, kf_r_ay]), vx_floor=VX_FLOOR,
                                 x0=[0.0, 0.0], P0=np.eye(2))
        self.speed_pid = SimplePID(speed_kp, speed_ki, speed_kd)
        self.max_throttle, self.brake_speed, self.brake_ratio, self.clip_delta = (
            max_throttle, brake_speed, brake_ratio, clip_delta)
        self.prev_delta = 0.0

    def control_pid(self, waypoints, speed, target, r_meas, ay_meas):
        """waypoints/target: VAD's own [lateral, forward] convention, unconverted (matches
        PIDController.control_pid()'s own call signature). speed: tick_data['speed'] (m/s).
        r_meas: yaw rate (rad/s, tick_data['angular_velocity'][2]). ay_meas: lateral accel (m/s^2,
        tick_data['acceleration'][1]). Returns (steer, throttle, brake, metadata), same shape
        PIDController.control_pid() returns."""
        speed = float(speed)
        vx = speed

        # v_y estimate: predict()+update() BEFORE this tick's delta exists, using LAST tick's
        # delta (self.prev_delta) -- same causality kalman_filter.py's own docstring lays out.
        self.kf.step(vx, self.prev_delta, [r_meas, ay_meas])
        v_y = self.kf.v_y
        # x0's r is the direct (real) gyro reading, not the filtered kf.r -- only v_y is the
        # estimate here, same split mpc_mpc_KF.py's "mpc-kf" controller uses.
        r = r_meas

        path, vx_spline, s_max_wp = build_trajectory_splines(waypoints, target)
        vx_preview, kappa_preview = preview_from_splines(path, vx_spline, s_max_wp, self.n_p, self.dt)

        x0 = [v_y, r, 0.0, float(path.yaw(0.0))]
        delta = self.lateral_mpc.solve(x0, vx_preview, kappa_preview)
        self.prev_delta = delta
        steer = steer_from_delta(delta)

        # Longitudinal: same brake/throttle shape pid_controller.py's own control_pid() uses (its
        # desired_speed reused here as vx_preview[0], now spline-derived instead of a raw waypoint-
        # spacing average), just handed to our own SimplePID instance -- keeps the comparison to
        # the baseline about the LATERAL controller (this project's actual subject).
        desired_speed = float(vx_preview[0])
        brake = bool(desired_speed < self.brake_speed
                    or (speed / desired_speed if desired_speed > 1e-6 else float("inf")) > self.brake_ratio)
        speed_error = float(np.clip(desired_speed - speed, 0.0, self.clip_delta))
        throttle = float(np.clip(self.speed_pid.step(speed_error), 0.0, self.max_throttle))
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': speed, 'steer': steer, 'throttle': throttle, 'brake': float(brake),
            'v_y_hat': v_y, 'delta_rad': delta, 'desired_speed': desired_speed,
            'kf_status': self.lateral_mpc.last_status,
        }
        return steer, throttle, brake, metadata
