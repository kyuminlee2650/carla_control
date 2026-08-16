r"""B2D/VAD closed-loop controller: LateralMPC + SpeedMPC (carla_control's own LPV-bicycle-model and
scalar-integrator QPs) + a LUT-feedforward+PID accel-tracking pedal layer (carla_control's own
MpcLongitudinal, see point 5 below) + VyKalmanFilter's v_y estimate, meant to
replace team_code/pid_controller.py's
PIDController inside team_code/vad_b2d_agent.py's run_step() (see that file's line ~396:
`self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)` -- this file's own
control_mpc() drops the local_command_xy arg entirely, see point 1 below for why, so the call site
needs that argument removed, not just renamed. Named control_mpc(), not control_pid(), and the
call site's own attribute should be renamed off self.pidcontroller too (e.g. self.controller) --
this class is MPC-based, keeping PID-flavored names around would be actively misleading.)

Self-contained on purpose (no import of carla_control/ or kalman_filter/): this file has to run
inside the b2d_zoo conda env / the submission .sif, which has no reason to also carry
carla_control's own CARLA_ROOT sys.path setup or its viz/offline-analysis dependencies.
VyKalmanFilter/LateralMPC/PathSpline below are copies of kalman_filter.py's, mpc_mpc_KF.py's, and
functions.py's own math (unchanged) -- see those files for the derivation docstrings; this file's
own docstrings focus on what's DIFFERENT about running them here.

Five things are different from carla_control's own driving scripts, and why:

1. No known map route -- VAD's own ~6-waypoint ego-frame prediction (`out_truck`, already cumsum'd
   from per-step displacements) plus the ego origin is all there is each tick (7 points total; no
   route-command point mixed in -- local_command_xy is a global-route-following aim that
   pid_controller.py's own steering logic only uses as a same-tick fallback/override for a single
   angle, never fits into a continuous curve with the waypoints -- see reference_upstream/
   pid_controller.py's control_pid(). Fitting it into this geometry/speed spline risked bending the
   fit toward a direction VAD's own local planning never predicted, for a point this controller
   never actually needs). build_trajectory_splines() fits a PathSpline (functions.py's own class,
   copied verbatim -- its docstring already anticipated this exact use: "sparser planner output
   like VAD later") directly to those 7 points instead of a GlobalRoutePlanner route, PLUS a
   companion vx(s) spline built
   from the waypoints' own known timing (confirmed 0.5s apart -- Bench2DriveZoo/docs/
   CONVERT_GUIDE.md: "Bench2Drive runs at 10Hz... window length... 0.5s"). x0's e_y is 0 and e_psi
   is path.yaw(0) by construction every tick (ego is redefined as the origin of a fresh ego-frame
   plan every cycle, not carried forward against a persistent map route).

2. get_physics_control() isn't reachable through run_step()'s own arguments -- a leaderboard agent
   only gets sensor input_data and returns a VehicleControl, no vehicle actor reference -- but IS
   reachable via srunner.scenariomanager.carla_data_provider.CarlaDataProvider.get_hero_actor(),
   which leaderboard/scenario_runner already register the ego actor into; vad_b2d_agent.py fetches
   it once (physics is static for a fixed blueprint, same "measured once, reused" convention
   carla_control's own run_trial()s use) and passes it into control_mpc()'s physics= arg.
   steer_from_delta_physics() then applies the same Ackermann inner-wheel + speed-dependent
   steering_curve correction functions.control_input() does (~15% combined error otherwise, see
   that function's own docstring) -- unchanged math, minus the vehicle.apply_control() side effect
   (this agent returns VehicleControl, it doesn't apply it itself). steer_from_delta() (the flat
   delta/MAX_STEER_ANGLE_RAD scale) is kept as the physics=None fallback control_mpc() uses when no
   physics is available -- e.g. validate_mpc_solve.py/inspect_one_sample.py's offline samples, which
   have no live vehicle actor to read.

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

5. Longitudinal pedal tracking is now carla_control's own MpcLongitudinal (mpc_mpc_comparison.py/
   longitudinal_mpc.py), ported verbatim (LongitudinalLUT, LookupController, functions.PID,
   functions.LowPassFilter, all below): SpeedMPC's own a_cmd goes through a LUT feedforward
   (longitudinal_lut.npz, calibrated (gear, v_x, a_x) -> pedal-command lookup, accurate to ~0.4
   m/s^2 inside its envelope per that file's own docstring) + PID closing the remainder, then a
   low-pass filter (tau=u_tau) before being split into throttle/brake -- not turned into a one-
   step-ahead desired_speed for a window-PID speed tracker the way an earlier version of this file
   did. That desired_speed+SimplePID path (and its own SimplePID class) has been removed entirely,
   not kept as a fallback -- gear/a_meas/longitudinal_lut.npz are now required control_mpc()
   arguments, on purpose (module docstring point 5's own instruction: match MpcLongitudinal, don't
   quietly degrade to something untested when telemetry is missing). A caller with no live vehicle
   actor (offline scripts -- validate_mpc_solve.py/inspect_one_sample.py) has to supply its own
   gear stand-in; see those files' own comments. gear (vehicle.get_control().gear) and a_meas (a
   real IMU longitudinal-accel reading, tick_data['acceleration'][0] -- lookup_controller.py's own
   docstring requires "a real IMU reading", satisfied here the same way ay_meas already is for the
   Kalman filter) both come from a live vehicle actor, fetched via the same CarlaDataProvider route
   point 2's physics uses -- see vad_b2d_agent.py.

   Weights (SpeedMPC's w_v/w_a/w_j/a_min/a_max, and the pedal loop's kp/ki/kd/u_tau) are
   mpc_mpc_comparison.py's own --w-v/--w-a/--w-j/--a-min/--a-max/--kp/--ki/--kd/--u-tau defaults
   verbatim, not re-tuned for this deployment. SpeedMPC's own n_p/n_c also now match that file's
   --np/--nc default (40) instead of sharing LateralMPC's n_p (still 20, untouched) -- the two
   previews still come from ONE preview_from_splines() call (build_trajectory_splines() only fits
   one spline either way), evaluated at the longer of the two lengths, with LateralMPC taking the
   matching-length prefix of the same array rather than a second, independently-walked preview.

Coordinate convention: VAD's own waypoints are [lateral, forward] (index 0 sideways, index 1
ahead) -- confirmed by pid_controller.py's own `angle = degrees(pi/2 - atan2(aim[1], aim[0]))`
formula, which only zeroes out on a straight-ahead point under that ordering. Every function below
takes/returns that native [lateral, forward] ordering at its public boundary and converts internally
to this file's own [forward, lateral] convention (matching carla_control's C(v_x)/x0/PathSpline sign
convention) wherever the math needs it.
"""
import math
import os

import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline

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


# ----------------------------------------------------------------------------- SpeedMPC
# Copy of carla_control/longitudinal_mpc.py's SpeedMPC -- unchanged math (see that file's own class
# docstring for the QP derivation: a scalar first-order-integrator plant, x_{k+1} = x_k + T*a_k,
# condensed the same way LateralMPC is). Unlike LateralMPC, H/A here don't depend on the vx preview
# (the plant here has no scheduling parameter), so this one is built once in __init__ and reused via
# solver.update() every solve() -- see that docstring for why that's safe here but not for
# LateralMPC. mpc_mpc_comparison.py's own MpcLongitudinal wraps this with a LUT-feedforward+PID
# pedal layer calibrated from real CARLA get_physics_control() data (longitudinal_lookup/);
# MpcKfController.control_mpc() now wraps it the same way -- see module docstring point 5 and the
# LongitudinalLUT/LookupController section below.

class SpeedMPC:
    """Speed-tracking MPC over a scalar first-order integrator: x_k = v_{x,k}, x_{k+1} = A x_k +
    B a_{x,k}, A=1, B=T. Decision variable is the acceleration sequence itself -- free for the
    first Nc steps of the horizon Np (Nc <= Np) and held at its last value after that. See
    carla_control/longitudinal_mpc.py's own SpeedMPC docstring for the full condensed-QP derivation
    (H/A built once here since neither depends on Uprev; only q and the rate half of l/u move each
    solve(), via Uprev) -- unchanged here."""

    def __init__(self, dt, n_p, n_c, w_v, w_a, w_j, a_min=-4.05, a_max=2.4, jerk_max=4.13):
        if not (0 < n_c <= n_p):
            raise ValueError(f"need 0 < n_c <= n_p, got n_c={n_c}, n_p={n_p}")
        if not (a_min < a_max):
            raise ValueError(f"need a_min < a_max, got a_min={a_min}, a_max={a_max}")
        if not (jerk_max > 0):
            raise ValueError(f"need jerk_max > 0, got jerk_max={jerk_max}")

        self.T = dt
        self.n_p = n_p
        self.n_c = n_c
        self.a_min = a_min
        self.a_max = a_max
        self.jerk_max = jerk_max

        self.A = 1.0   # fixed by the problem: x_{k+1} = A x_k + B a_{x,k}
        self.B = dt

        self.A_bar = np.full((n_p, 1), self.A)   # A^i = 1 for every i since A=1
        self.B_bar = self._build_b_bar()
        self.Phi = self._build_phi()

        W1 = w_v * np.eye(n_p)
        W2 = w_a * np.eye(n_c)
        W3 = w_j * np.eye(n_c)   # Phi is now n_c x n_c (see _build_phi), not n_c-1

        self.H = 2.0 * (self.B_bar.T @ W1 @ self.B_bar + W2 + self.Phi.T @ W3 @ self.Phi)
        self.H = 0.5 * (self.H + self.H.T)      # symmetrize against round-off
        self._B_W1 = 2.0 * self.B_bar.T @ W1        # reused every cycle to form f
        self._2PhiT_W3 = 2.0 * self.Phi.T @ W3      # reused every cycle for f's Uprev cross term

        self._A_ineq = sparse.csc_matrix(np.vstack([np.eye(n_c), self.Phi]))
        self._a_l = a_min * np.ones(n_c)
        self._a_u = a_max * np.ones(n_c)

        self._solver = osqp.OSQP()
        self._solver.setup(
            P=sparse.csc_matrix(self.H),
            q=np.zeros(n_c),
            A=self._A_ineq,
            l=np.concatenate([self._a_l, -jerk_max * dt * np.ones(n_c)]),
            u=np.concatenate([self._a_u, jerk_max * dt * np.ones(n_c)]),
            verbose=False,
            polish=False,
        )
        self.last_solution = np.zeros(n_c)
        self.last_status = "unsolved"

    def _build_b_bar(self):
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
        Phi = np.eye(self.n_c)
        for l in range(1, self.n_c):
            Phi[l, l - 1] = -1.0
        return Phi

    def solve(self, v_x, v_ref_preview):
        """One receding-horizon step. v_ref_preview: length-Np sequence of desired speed at each
        future prediction step. Returns a_cmd, the acceleration to hand the pedal layer, clipped to
        [a_min, a_max]."""
        x_ref = np.asarray(v_ref_preview, dtype=float).reshape(-1, 1)
        u_prev = np.zeros((self.n_c, 1))
        u_prev[0, 0] = self.last_solution[0]

        f = self._B_W1 @ (self.A_bar * v_x - x_ref) - self._2PhiT_W3 @ u_prev

        rate_l = -self.jerk_max * self.T + u_prev.ravel()
        rate_u = self.jerk_max * self.T + u_prev.ravel()
        l_full = np.concatenate([self._a_l, rate_l])
        u_full = np.concatenate([self._a_u, rate_u])

        self._solver.update(q=f.ravel(), l=l_full, u=u_full)
        result = self._solver.solve()
        self.last_status = result.info.status

        if result.x is None or not np.all(np.isfinite(result.x)):
            u = self.last_solution
        else:
            u = result.x
            self.last_solution = u.copy()

        return float(np.clip(u[0], self.a_min, self.a_max))


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


def build_trajectory_splines(waypoints, speed, dt_wp=VAD_WP_DT, smoothing_xy=0.05, smoothing_v=0.5):
    r"""Fit a PathSpline (x(s)/y(s)/yaw(s)/kappa(s)) + a companion vx(s) spline directly from VAD's
    own ~6 waypoints plus the ego origin (7 points total), in place of carla_control's
    PathSpline-over-a-known-route. See module docstring point 1 for why no route-command point
    (local_command_xy) is mixed into this fit.

    waypoints: VAD's out_truck, (N, 2) in its own [lateral, forward] ego-frame convention (index 0
    sideways, index 1 ahead), ego at the origin, heading along +forward, each step dt_wp seconds
    after the last. speed: the actual MEASURED current speed (m/s, tick_data['speed']) -- anchors
    the vx(s) fit at s=0. Without it, vx_spline(0) is only VAD's own *implied* average speed over
    the first waypoint interval (a coarse ds/dt over up to dt_wp seconds of prediction), which can
    disagree with the real current speed by several m/s (confirmed: one sample's fit started near
    5.4 m/s against a real 7.04 m/s) -- exactly the value vx_preview[0] hands the MPC for its very
    first control step, where that gap matters most.

    Returns (path, vx_spline, s_max_wp): path is a PathSpline in this file's own (forward, lateral)
    convention; vx_spline(s) -> m/s is a plain UnivariateSpline (speed isn't a geometric property of
    the path, so it gets its own fit, not a PathSpline derivative); s_max_wp is the last waypoint's
    station, which is now also path.s_max (the fit no longer extends past the real waypoints) --
    kept as its own name since it's what vx_spline's own data actually covers.
    """
    fwd = [0.0] + [wp[1] for wp in waypoints]
    lat = [0.0] + [wp[0] for wp in waypoints]
    path = PathSpline(fwd, lat, smoothing=smoothing_xy)

    s_wp = _chord_length_station(fwd, lat)   # station at the origin + each real waypoint
    n_wp = len(waypoints)
    t_wp = np.arange(n_wp + 1) * dt_wp       # 0, dt_wp, 2*dt_wp, ... -- known VAD cadence

    # One speed sample per waypoint interval, assigned to that interval's own arc-length midpoint
    # (an average speed over [s_{i-1}, s_i] describes the midpoint, not either edge), PLUS the
    # measured current speed anchored at s=0 itself -- see docstring above for why that anchor is
    # needed (VAD's own segment averages don't otherwise pin down the fit's value at the origin).
    ds = np.diff(s_wp)
    dt = np.diff(t_wp)
    v_seg = np.concatenate([[float(speed)], ds / np.maximum(dt, 1e-3)])
    s_mid = np.concatenate([[0.0], s_wp[:-1] + ds / 2.0])

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
    VehicleControl.steer in [-1, 1]. Flat delta/MAX_STEER_ANGLE_RAD scale, no Ackermann/
    steering_curve correction -- the physics=None fallback control_mpc() uses when there's no live
    vehicle actor to read (see module docstring point 2). Prefer steer_from_delta_physics() whenever
    physics is available."""
    return float(np.clip(delta_rad / MAX_STEER_ANGLE_RAD, -1.0, 1.0))


def track_over_wheelbase(physics):
    """track / wheelbase, the only vehicle geometry the Ackermann conversion needs. Copy of
    carla_control/functions.py's track_over_wheelbase() -- unchanged math; see that file's own
    docstring. physics.wheels order (CARLA convention): [front_left, front_right, rear_left,
    rear_right]."""
    wheels = physics.wheels
    track = math.hypot(wheels[0].position.x - wheels[1].position.x,
                       wheels[0].position.y - wheels[1].position.y)
    front_x = 0.5 * (wheels[0].position.x + wheels[1].position.x)
    front_y = 0.5 * (wheels[0].position.y + wheels[1].position.y)
    rear_x = 0.5 * (wheels[2].position.x + wheels[3].position.x)
    rear_y = 0.5 * (wheels[2].position.y + wheels[3].position.y)
    wheelbase = math.hypot(front_x - rear_x, front_y - rear_y)
    return track / wheelbase


def steering_curve_scale(physics, speed_ms):
    """The factor CARLA applies to a steer command at this speed. Copy of carla_control/
    functions.py's steering_curve_scale() -- unchanged math; see that file's own docstring
    (VehiclePhysicsControl.steering_curve's x axis is km/h, not m/s)."""
    xs = [point.x for point in physics.steering_curve]
    ys = [point.y for point in physics.steering_curve]
    lo, hi = xs[0], xs[-1]
    x = min(max(speed_ms * 3.6, lo), hi)
    for i in range(1, len(xs)):
        if x <= xs[i]:
            span = xs[i] - xs[i - 1]
            t = 0.0 if span == 0 else (x - xs[i - 1]) / span
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def steer_from_delta_physics(delta_rad, physics, v_x, max_steer_angle_rad=MAX_STEER_ANGLE_RAD):
    """delta (rad, bicycle-model front-wheel angle) -> CARLA's normalized VehicleControl.steer,
    WITH the Ackermann inner-wheel + speed-dependent steering_curve corrections carla_control's own
    functions.control_input() applies (naive delta/max_steer is ~15% off without them, per that
    function's own docstring). Same math as control_input(), minus the Ackermann geometry's
    `cot(inner) = cot(bicycle) - track/(2*wheelbase)` being anything other than a straight port, and
    minus control_input()'s own vehicle.apply_control(control) call -- a leaderboard agent returns
    VehicleControl, it doesn't apply it itself. physics: vehicle.get_physics_control(), fetched once
    by the caller (see module docstring point 2 for why it can't be read from here directly) and
    reused every tick -- steering_curve/wheel geometry are static for a fixed blueprint."""
    if abs(delta_rad) < 1e-6:
        inner = 0.0
    else:
        cot_inner = 1.0 / math.tan(abs(delta_rad)) - 0.5 * track_over_wheelbase(physics)
        # cot <= 0 would mean an inner wheel past 90 deg; the steering limit binds long before
        # that, so clamp rather than let the arithmetic wrap (same as control_input()).
        inner = max_steer_angle_rad if cot_inner <= 0.0 else math.atan(1.0 / cot_inner)
        inner = math.copysign(inner, delta_rad)
    scale = steering_curve_scale(physics, max(v_x, 0.0))
    return float(np.clip(inner / (max_steer_angle_rad * scale), -1.0, 1.0))


# ----------------------------------------------------------------------------- longitudinal pedal layer
# Copies of carla_control/functions.py's PID/LowPassFilter and
# carla_control/longitudinal_lookup/longitudinal_lut.py's LongitudinalLUT and
# lookup_controller.py's LookupController -- unchanged math, see those files' own docstrings for
# the derivations/tuning notes. Ported here (not imported) for the same self-contained-.sif reason
# every other embedded copy in this file exists -- see module docstring point 5.

class PID:
    """Anti-windup-clamped PID -- carla_control/functions.py's own PID class, unchanged. Drives the
    LUT+PID accel-tracking loop LookupController.step() below uses (module docstring point 5)."""

    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error):
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        trial = self._integral + error * self.dt
        raw = self.kp * error + self.ki * trial + self.kd * derivative
        if not ((raw > 1 and error > 0) or (raw < -1 and error < 0)):
            self._integral = trial

        return self.kp * error + self.ki * self._integral + self.kd * derivative


class LowPassFilter:
    """Copy of carla_control/functions.py's LowPassFilter -- unchanged math."""

    def __init__(self, tau, dt, initial=0.0):
        self.alpha = dt / (tau + dt)
        self.state = initial

    def step(self, x):
        self.state += self.alpha * (x - self.state)
        return self.state


class LongitudinalLUT:
    """Runtime (gear, v_x, a_x) -> pedal-command lookup. Copy of carla_control/longitudinal_lookup/
    longitudinal_lut.py's LongitudinalLUT -- unchanged math; see that file's own docstring (loads
    the .npz build_lut.py produced from collect_lut_data.py's real CARLA sweep, accurate to ~0.4
    m/s^2 inside its calibrated envelope). npz_path defaults to a file of the same name sitting
    next to this one -- deploy note in module docstring point 5: copy longitudinal_lut.npz into
    team_code/ alongside this file and vad_b2d_agent.py, all three together."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.gears = [int(g) for g in data["gears"]]
        self._interp = {}
        self._bounds = {}
        self.speed_range = {}
        for gear in self.gears:
            v_grid = data[f"g{gear}_v"]
            a_grid = data[f"g{gear}_a"]
            u_table = data[f"g{gear}_u"]
            self._interp[gear] = RegularGridInterpolator(
                (v_grid, a_grid), u_table, bounds_error=False, fill_value=None
            )
            self._bounds[gear] = (v_grid.min(), v_grid.max(), a_grid.min(), a_grid.max())
            key = f"g{gear}_vrange"
            self.speed_range[gear] = (tuple(float(x) for x in data[key]) if key in data
                                      else (float(v_grid.min()), float(v_grid.max())))

    def lookup(self, gear, v_x, a_x):
        if gear not in self._interp:
            raise ValueError(f"No calibration data for gear {gear}; available: {self.gears}")
        v_min, v_max, a_min, a_max = self._bounds[gear]
        v_q = min(max(v_x, v_min), v_max)
        a_q = min(max(a_x, a_min), a_max)
        u = float(self._interp[gear]([[v_q, a_q]])[0])
        return max(-1.0, min(1.0, u))

    def gears_for_speed(self, v_x):
        lo_hi = lambda g: self.speed_range.get(g)
        return [g for g in self.gears if lo_hi(g) is not None and lo_hi(g)[0] <= v_x <= lo_hi(g)[1]]


class LookupController:
    """LUT feedforward + PID feedback -> pedal command u in [-1, 1]. Copy of carla_control/
    longitudinal_lookup/lookup_controller.py's LookupController -- unchanged math; see that file's
    own docstring for why a_meas must be a real IMU reading (no differentiate-v_x fallback) and why
    kd defaults to 0 (a single kp/ki that stays stable across gears beat a higher-kd tune that only
    damped some operating points)."""

    def __init__(self, lut, kp, ki, kd=0.0, dt=0.05, use_feedforward=True):
        self.lut = lut
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.use_feedforward = use_feedforward
        self.reset()

    def reset(self):
        self.pid = PID(self.kp, self.ki, self.kd, self.dt)
        self.last_gear = None
        self.saturated = False

    def feedforward(self, gear, v_x, a_cmd):
        if gear == 0 or gear not in self.lut.gears:
            if self.last_gear is not None:
                gear = self.last_gear
            else:
                candidates = self.lut.gears_for_speed(v_x)
                gear = candidates[-1] if candidates else self.lut.gears[0]
        else:
            self.last_gear = gear
        return self.lut.lookup(gear, v_x, a_cmd)

    def step(self, gear, v_x, a_cmd, a_meas):
        error = a_cmd - a_meas
        ff = self.feedforward(gear, v_x, a_cmd) if self.use_feedforward else 0.0
        u = ff + self.pid.step(error)
        clipped = max(-1.0, min(1.0, u))
        self.saturated = clipped != u
        return clipped


# ----------------------------------------------------------------------------- the drop-in controller

class MpcKfController:
    r"""Drop-in-ish replacement for team_code/pid_controller.py's PIDController. Unlike
    PIDController.control_pid(waypoints, speed, target), this drops target/local_command_xy
    entirely -- see module docstring point 1 for why -- so vad_b2d_agent.py's call site needs that
    argument removed (not just renamed), in addition to instantiating this class instead of
    PIDController and passing tick_data['angular_velocity'][2] / tick_data['acceleration'][1] as
    the two extra args (r_meas, ay_meas) -- already unpacked into local variables for the can_bus
    feature every tick, no new sensor wiring needed.

    Method is named control_mpc(), deliberately not control_pid(): this controller is MPC-based
    (LateralMPC + SpeedMPC), and only the LUT+PID pedal layer underneath them (module docstring
    point 5) is actually PID -- keeping the upstream method name would misleadingly suggest the
    whole thing is a PID controller. Same reason the call site's own attribute should be
    self.controller, not self.pidcontroller (see vad_b2d_agent.py).

    gear/a_meas/longitudinal_lut.npz are required, not optional -- no no-telemetry fallback pedal
    tracker exists in this class (an earlier version had one; removed on purpose, see module
    docstring point 5). A caller with no live vehicle actor to read gear/a_meas from (e.g. a
    from-scratch offline script) has to supply its own stand-in values; validate_mpc_solve.py/
    inspect_one_sample.py do exactly that -- see their own comments for why a placeholder gear is
    fine there.

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
                max_throttle=0.75,
                n_p_speed=40, n_c_speed=40, w_v=10.0, w_a=1.0, w_j=30.0,
                a_min=-4.05, a_max=2.4, jerk_max=4.13,
                lut_path=None, pedal_kp=0.15, pedal_ki=0.6, pedal_kd=0.0, pedal_u_tau=0.02):
        self.dt, self.n_p, self.n_p_speed = dt, n_p, n_p_speed
        self.lateral_mpc = LateralMPC(
            dt=dt, n_p=n_p, n_c=n_c, mass=VEHICLE_MASS, Iz=VEHICLE_IZ, lf=VEHICLE_LF, lr=VEHICLE_LR,
            Cf=VEHICLE_CF, Cr=VEHICLE_CR, w_ey=w_ey, w_epsi=w_epsi, w_ay=w_ay, w_r=w_r, w_rdot=w_rdot,
            w_delta=w_delta, w_ddelta=w_ddelta,
            delta_max=math.radians(delta_max_deg), ddelta_max=math.radians(ddelta_max_deg) * dt,
            vx_floor=VX_FLOOR)
        # n_p_speed/n_c_speed default to mpc_mpc_comparison.py's own --np/--nc (40), decoupled from
        # LateralMPC's n_p (still 20) -- see module docstring point 5 for how one preview walk still
        # feeds both.
        self.speed_mpc = SpeedMPC(dt=dt, n_p=n_p_speed, n_c=n_c_speed, w_v=w_v, w_a=w_a, w_j=w_j,
                                  a_min=a_min, a_max=a_max, jerk_max=jerk_max)
        self.kf = VyKalmanFilter(dt=dt, mass=VEHICLE_MASS, Iz=VEHICLE_IZ, lf=VEHICLE_LF, lr=VEHICLE_LR,
                                 Cf=VEHICLE_CF, Cr=VEHICLE_CR, Q=np.diag([kf_q_vy, kf_q_r]),
                                 R=np.diag([kf_r_dpsi, kf_r_ay]), vx_floor=VX_FLOOR,
                                 x0=[0.0, 0.0], P0=np.eye(2))
        self.max_throttle = max_throttle

        # Longitudinal pedal layer -- carla_control's own MpcLongitudinal (module docstring point
        # 5), required. Deliberately NOT wrapped in a try/except that degrades to some fallback --
        # a missing/corrupt longitudinal_lut.npz should fail loudly at construction (deploy-time),
        # not silently produce a different (worse, untested-in-this-form) control law mid-drive.
        lut_path = lut_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "longitudinal_lut.npz")
        self.pedal_ctrl = LookupController(LongitudinalLUT(lut_path), kp=pedal_kp, ki=pedal_ki,
                                           kd=pedal_kd, dt=dt)
        self.u_filter = LowPassFilter(tau=pedal_u_tau, dt=dt, initial=0.0)

        self.prev_delta = 0.0

    def control_mpc(self, waypoints, speed, r_meas, ay_meas, gear, a_meas, physics=None):
        """waypoints: VAD's own [lateral, forward] convention, unconverted. speed:
        tick_data['speed'] (m/s). r_meas: yaw rate (rad/s, tick_data['angular_velocity'][2]).
        ay_meas: lateral accel (m/s^2, tick_data['acceleration'][1]). gear: vehicle.get_control().
        gear (int; 0 is CARLA's own "between gears" value -- LookupController.feedforward() already
        falls back to the last engaged gear or a speed-based guess for that case, same as
        carla_control's own LookupController does, see that class's own comment). a_meas: a real
        IMU longitudinal-accel reading (m/s^2, tick_data['acceleration'][0]) -- LookupController
        drives throttle/brake off SpeedMPC's own a_cmd through the LUT+PID pedal layer (module
        docstring point 5); there's no fallback if this or the LUT is unavailable, see class
        docstring. physics: optional vehicle.get_physics_control() (fetched once by the caller via
        CarlaDataProvider, see module docstring point 2) -- when given, steer_from_delta_physics()
        applies the real Ackermann + steering_curve correction; when None (e.g. offline
        validate_mpc_solve.py/inspect_one_sample.py samples with no live vehicle actor), falls back
        to the flat steer_from_delta() scale -- this fallback is steering-only, unaffected by
        gear/a_meas being required now. Returns (steer, throttle, brake, metadata), same shape
        PIDController.control_pid() returns (this one just never takes target/local_command_xy --
        see module docstring point 1). Named control_mpc(), not control_pid(), since this class is
        MPC-based -- see class docstring."""
        speed = float(speed)
        vx = speed

        # v_y estimate: predict()+update() BEFORE this tick's delta exists, using LAST tick's
        # delta (self.prev_delta) -- same causality kalman_filter.py's own docstring lays out.
        self.kf.step(vx, self.prev_delta, [r_meas, ay_meas])
        v_y = self.kf.v_y
        # x0's r is the direct (real) gyro reading, not the filtered kf.r -- only v_y is the
        # estimate here, same split mpc_mpc_KF.py's "mpc-kf" controller uses.
        r = r_meas

        path, vx_spline, s_max_wp = build_trajectory_splines(waypoints, speed)
        # One station-walk feeds both MPCs, evaluated at the longer of the two horizons -- module
        # docstring point 5 -- LateralMPC then takes the matching-length PREFIX of the same array
        # rather than a second, independently-walked preview (n_p_speed=40 > n_p=20 by default, so
        # in practice this walks 40 steps and lateral uses steps 0-19 of it).
        n_p_preview = max(self.n_p, self.n_p_speed)
        vx_preview_full, kappa_preview_full = preview_from_splines(
            path, vx_spline, s_max_wp, n_p_preview, self.dt)
        vx_preview_lat, kappa_preview_lat = vx_preview_full[:self.n_p], kappa_preview_full[:self.n_p]
        vx_preview_speed = vx_preview_full[:self.n_p_speed]

        x0 = [v_y, r, 0.0, float(path.yaw(0.0))]
        delta = self.lateral_mpc.solve(x0, vx_preview_lat, kappa_preview_lat)
        self.prev_delta = delta
        steer = (steer_from_delta_physics(delta, physics, vx) if physics is not None
                else steer_from_delta(delta))

        # Longitudinal: SpeedMPC (same QP carla_control/longitudinal_mpc.py's own SpeedMPC solves)
        # plans a_cmd against its own vx_preview_speed, then carla_control's own MpcLongitudinal
        # pedal layer (LUT feedforward + PID + low-pass filter, module docstring point 5) drives
        # it to throttle/brake -- the QP decides the acceleration plan, this layer is what actually
        # tracks it against the real vehicle's (gear, v_x) -> pedal response.
        a_cmd = self.speed_mpc.solve(speed, vx_preview_speed)
        u_raw = self.pedal_ctrl.step(int(gear), speed, a_cmd, a_meas=float(a_meas))
        u = self.u_filter.step(u_raw)
        throttle = float(np.clip(max(u, 0.0), 0.0, self.max_throttle))
        brake = float(np.clip(max(-u, 0.0), 0.0, 1.0))

        metadata = {
            'speed': speed, 'steer': steer, 'throttle': throttle, 'brake': brake,
            'v_y_hat': v_y, 'delta_rad': delta, 'a_cmd': a_cmd, 'u_raw': u_raw, 'u_filtered': u,
            'pedal_saturated': self.pedal_ctrl.saturated, 'gear': int(gear),
            'steer_physics_corrected': physics is not None,
            'kf_status': self.lateral_mpc.last_status, 'speed_mpc_status': self.speed_mpc.last_status,
        }
        return steer, throttle, brake, metadata
