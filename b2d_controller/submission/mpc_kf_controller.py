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
from scipy.linalg import expm

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

# ----------------------------------------------------------------- low-speed handling
# There is no vx_floor here any more. carla_control (and this file, until now) floored vx at
# 0.5 m/s everywhere the lateral bicycle model is evaluated, because the model's A matrix carries
# 1/vx terms and the model was discretized with FORWARD EULER, Ad = I + dt*A. Forward Euler turns
# a stiff-but-stable A into an EXPANDING map: measured on this vehicle's own parameters,
# rho(I + dt*A) is 8.28 at vx = 0.5 and only drops to 1 at vx = 2.338 m/s. LateralMPC._condense()
# then multiplies 25 of those together, so max|A_bar| reached 8.55e22 and the condensed Hessian
# H = 2(M^T W1 M + W2 + G^T W3 G) -- positive semidefinite in exact arithmetic -- came out
# numerically indefinite (min eig -2.8e33). That is what OSQP reported as "problem non convex":
# 159 of 683 real frames on route 28154. Raising the floor past 2.338 would have hidden it, at the
# price of telling the model the car is doing 2.5 m/s while it sits still.
#
# Both MPCs and the Kalman filter now use EXACT zero-order-hold discretization instead
# (scipy.linalg.expm, see LateralMPC._discretize). The continuous A is stable at every vx
# (eigenvalue real parts <= 0 at 0.5, 1, 2, 3, 8 m/s -- checked, not assumed), so expm(A*dt) has
# spectral radius <= 1 at every speed and nothing amplifies. Measured over the same 683 frames:
# 0 non-convex, max|A_bar| 10.26 -- the value the healthy high-speed frames already had -- and
# that result is unchanged all the way down to vx = 0.001 m/s.
#
# What remains is only a guard against literal division by zero: the dump has ticks at
# speed = 0.000 and -0.02 m/s, and 1/0 is undefined however good the discretization is. VX_EPS is
# that guard and nothing else -- it is not a tuning knob and not a modelling choice, so it is set
# far below any speed at which the numbers still move (the Hessian's min eigenvalue changes by
# 0.4% between vx = 0.05 and vx = 0.001).
#
# SpeedMPC is untouched on purpose: its plant is the scalar integrator x_{k+1} = x_k + dt*a, and
# for xdot = u forward Euler already IS exact ZOH (e^0 = 1, integral of e^0 = dt). Its Hessian is
# constant and has never gone non-convex (707/707 solved).
VX_EPS = 1e-3

# ----------------------------------------------------------------- QP failure handling
# Only these two OSQP statuses mean result.x is the actual optimum -- same set
# validate_mpc_solve.py's own OK_STATUSES uses. Everything else ("primal infeasible",
# "maximum iterations reached", "problem non convex", ...) must be treated as a FAILED solve.
#
# This matters because OSQP does NOT signal failure through result.x: on "primal infeasible" it
# returns a finite garbage iterate (measured: x = 2.14e9), so the old
# `if result.x is None or not np.all(np.isfinite(result.x))` guard never fired, the garbage was
# clipped straight to +-delta_max, AND it was written back into self.last_solution -- poisoning
# U_prev for every following tick. That is exactly how route 28154 ended up pinned at +30.00000
# deg of steer for 50+ consecutive frames after its collision. The status has to be checked.
_OK_STATUSES = ("solved", "solved inaccurate")

# On a failed solve we hold the last GOOD command (the behavior the original code intended), but
# only for a bounded number of consecutive failures -- holding a large steer angle forever is what
# turns a recoverable one-off QP failure into driving in a circle. After that, decay it
# geometrically toward 0 (wheels straighten / acceleration coasts out) so a persistent failure
# degrades into something inert instead of something committed.
QP_FAIL_HOLD_TICKS = 5
QP_FAIL_DECAY = 0.7
VAD_WP_DT = 0.5        # seconds between VAD's own future-trajectory steps -- confirmed, see module docstring


# ----------------------------------------------------------------------------- VyKalmanFilter
# Ported from kalman_filter/kalman_filter.py's VyKalmanFilter, with TWO deviations from that copy,
# both consequences of the exact-ZOH change (see the VX_EPS block at the top of this file):
#   * predict() discretizes with expm instead of Ad = I + dt*A. Under forward Euler the covariance
#     recursion P <- Ad P Ad^T + Q amplified P by rho(Ad)^2 = 68x per tick at vx = 0.5 m/s.
#   * the vx_floor low-speed gate is gone. That gate asserted v_y = 0, r = z[0] below the floor;
#     the reference's own step() docstring justifies it by the model re-injecting a "phantom"
#     lateral velocity that the low-vx-weakened measurement update cannot cancel -- which is a
#     description of a divergent predict step, i.e. of the forward-Euler problem. With a
#     contractive Ad at every speed there is nothing to cancel, so the filter keeps estimating
#     v_y through low-speed stretches instead of pinning it to zero.

class VyKalmanFilter:
    N_X = 2   # [v_y, r]
    N_Z = 2   # [dpsi_meas, ay_meas]

    def __init__(self, dt, mass, Iz, lf, lr, Cf, Cr, Q, R, vx_eps=VX_EPS, x0=None, P0=None):
        self.dt = dt
        self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr = mass, Iz, lf, lr, Cf, Cr
        self.vx_eps = vx_eps
        self.B_cont = np.array([[Cf / mass], [lf * Cf / Iz]])
        self.D = np.array([[0.0], [Cf / mass]])
        self.Q = np.asarray(Q, dtype=float).reshape(self.N_X, self.N_X)
        self.R = np.asarray(R, dtype=float).reshape(self.N_Z, self.N_Z)
        self.x = np.zeros((self.N_X, 1)) if x0 is None else np.asarray(x0, dtype=float).reshape(self.N_X, 1)
        self.P = np.eye(self.N_X) if P0 is None else np.asarray(P0, dtype=float).reshape(self.N_X, self.N_X)

    def _continuous_A(self, vx):
        vx = max(vx, self.vx_eps)
        m, Iz, lf, lr, Cf, Cr = self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr
        return np.array([
            [-(Cf + Cr) / (m * vx),           (lr * Cr - lf * Cf) / (m * vx) - vx],
            [(lr * Cr - lf * Cf) / (Iz * vx), -(lf**2 * Cf + lr**2 * Cr) / (Iz * vx)],
        ])

    def _H(self, vx):
        vx = max(vx, self.vx_eps)
        m, lf, lr, Cf, Cr = self.mass, self.lf, self.lr, self.Cf, self.Cr
        return np.array([
            [0.0, 1.0],
            [-(Cf + Cr) / (m * vx), (lr * Cr - lf * Cf) / (m * vx)],
        ])

    def predict(self, vx, delta_prev):
        # Exact ZOH, same reasoning as LateralMPC._discretize -- and it matters more here, because
        # the covariance recursion squares the transition: P <- Ad P Ad^T + Q grows by rho(Ad)^2
        # per tick, which under forward Euler was 68x per tick at vx = 0.5 and 2.1e3 at vx = 0.1.
        Ad, Bd = self._discrete(vx)
        self.x = Ad @ self.x + Bd * delta_prev
        self.P = Ad @ self.P @ Ad.T + self.Q
        return self.x.ravel()

    def _discrete(self, vx):
        """(Ad, Bd) by exact zero-order hold, via expm([[A, B], [0, 0]] * dt) = [[Ad, Bd], [0, 1]].
        The augmented form is used rather than Bd = A^-1 (Ad - I) B because A is singular at the
        speeds this now has to survive."""
        n = self.N_X
        aug = np.zeros((n + 1, n + 1))
        aug[:n, :n] = self._continuous_A(vx)
        aug[:n, n:n + 1] = self.B_cont
        M = expm(aug * self.dt)
        return M[:n, :n], M[:n, n:n + 1]

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
        # The `if vx < vx_floor: x = [0, z[0]]` bypass this used to open with is gone with the
        # floor. It existed because the forward-Euler predict step was divergent below ~2.3 m/s,
        # so the only safe thing there was to throw the model away and take the measured yaw rate
        # neat. Exact ZOH is contractive at every speed (rho(Ad) <= 1 down to vx = VX_EPS), so the
        # filter can now just run, and v_y keeps being estimated through the low-speed stretches
        # instead of being pinned to 0 -- which is most of a junction approach.
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
                delta_max, ddelta_max, vx_eps=VX_EPS):
        self.dt, self.n_p, self.n_c = dt, n_p, n_c
        self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr = mass, Iz, lf, lr, Cf, Cr
        self.delta_max, self.ddelta_max, self.vx_eps = delta_max, ddelta_max, vx_eps

        self.B_cont = np.array([[Cf / mass], [lf * Cf / Iz], [0.0], [0.0]])
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
        # consecutive failed solves -- drives the hold-then-decay fallback, see _OK_STATUSES
        self.consecutive_failures = 0

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
        vx = max(vx, self.vx_eps)
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

        The augmented-matrix identity is used rather than the usual Bd = A^-1 (Ad - I) B because
        A is SINGULAR here -- the e_y and e_psi rows are pure integrators, so A has two zero
        eigenvalues at every speed and the inverse does not exist. One expm of a 6x6 per preview
        step, measured at ~72 us, so ~1.8 ms for the 25-step lateral horizon against a 50 ms
        control period.

        Bd is returned per step now (it used to be the constant self.B_disc = dt * B_cont). Under
        forward Euler the input matrix genuinely was speed-independent; under ZOH it is not, since
        Bd = integral_0^dt expm(A s) ds * B and A depends on vx. _condense() takes the list.
        """
        n = self.N_X
        Ad, Bd, Ed, Ck = [], [], [], []
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
        vx_preview = np.clip(np.asarray(vx_preview, dtype=float), self.vx_eps, None)
        kappa_preview = np.asarray(kappa_preview, dtype=float)
        x0 = np.asarray(x0, dtype=float).reshape(-1, 1)

        Ad, Bd, Ed, Ck = self._discretize(vx_preview)
        A_bar, B_bar, E_bar, C_bar, D_bar = self._condense(Ad, Bd, Ed, Ck)

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

        # setup() itself raises OSQPException when the condensed H comes out numerically
        # non-convex (confirmed: negative-definite P -> "Error in LDL_factor ... non-convex" at
        # SETUP, not at solve). Unguarded that propagates out of run_step() and kills the agent
        # mid-drive, so the whole solve is wrapped -- a QP failure must degrade to a held command,
        # never to a dead agent. See _OK_STATUSES for why the status (not result.x) is the signal.
        try:
            solver = osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(H), q=f.ravel(), A=self._A_ineq, l=l_bound, u=u_bound,
                        verbose=False, polish=False)
            result = solver.solve()
            status, x = result.info.status, result.x
        except Exception as exc:
            status, x = "raised: %s" % type(exc).__name__, None
        self.last_status = status

        if status in _OK_STATUSES and x is not None and np.all(np.isfinite(x)):
            u = x
            self.last_solution = u.copy()
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures > QP_FAIL_HOLD_TICKS:
                self.last_solution = self.last_solution * QP_FAIL_DECAY
            u = self.last_solution

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
        # consecutive failed solves -- drives the hold-then-decay fallback, see _OK_STATUSES
        self.consecutive_failures = 0

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

        # Same failed-solve handling as LateralMPC.solve() -- see _OK_STATUSES. This QP's H does not
        # depend on the vx preview (so it never goes non-convex, and it solved 65/65 in the route
        # 28154 run), but the infeasible-returns-finite-garbage trap is identical, and decaying
        # a_cmd toward 0 on a persistent failure (coast) beats holding a stale full-throttle
        # acceleration command indefinitely.
        try:
            self._solver.update(q=f.ravel(), l=l_full, u=u_full)
            result = self._solver.solve()
            status, x = result.info.status, result.x
        except Exception as exc:
            status, x = "raised: %s" % type(exc).__name__, None
        self.last_status = status

        if status in _OK_STATUSES and x is not None and np.all(np.isfinite(x)):
            u = x
            self.last_solution = u.copy()
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures > QP_FAIL_HOLD_TICKS:
                self.last_solution = self.last_solution * QP_FAIL_DECAY
            u = self.last_solution

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

        # DEGENERATE PLAN GUARD. min_spacing filtering can leave a SINGLE surviving point when
        # VAD's own plan collapses to (nearly) the ego origin -- which is exactly what VAD does in
        # its known freeze mode (see the stop-sign analysis: 3 s plans shrinking to 0.025-0.060 m).
        # k = min(3, len(s)-1) is then 0, and UnivariateSpline raises "k should be 1 <= k <= 5",
        # which propagates out of run_step() and kills the agent mid-drive. There is no meaningful
        # path to fit from one point, so degrade to "straight ahead, zero curvature" and let the
        # longitudinal side (which sees vx_ref ~ 0 from the same collapsed plan) do the stopping.
        # A short-but-not-single plan is a SECOND degenerate case, handled in kappa() rather than
        # here. k = min(3, len(s) - 1) drops to 1 when only two points survive, and scipy's splev
        # refuses a 2nd derivative above the spline degree ("0<=der=2<=k=1 must hold") instead of
        # returning 0 -- verified against scipy 1.10 in this env. That killed the agent 2.4 s into
        # routes 2416 and 3144 of the ability-10 set (both "Failed - Agent crashed", DS 0). A
        # degree-1 fit is a straight segment, so its curvature IS identically zero; kappa() now
        # returns that instead of asking splev for a derivative it cannot give.
        self.degenerate = len(s) < 2
        if self.degenerate:
            self.s_min = self.s_max = 0.0
            self._sx = self._sy = None
            return

        self.s_min, self.s_max = float(s[0]), float(s[-1])
        k = min(3, len(s) - 1)   # UnivariateSpline needs k < number of points; degrade gracefully
        self._k = k
        self._sx = UnivariateSpline(s, px, k=k, s=smoothing)
        self._sy = UnivariateSpline(s, py, k=k, s=smoothing)

    def xy(self, s):
        if self.degenerate:
            # straight along +forward (this file's own (forward, lateral) convention)
            return np.asarray(s, dtype=float), np.zeros_like(np.asarray(s, dtype=float))
        return self._sx(s), self._sy(s)

    def yaw(self, s):
        if self.degenerate:
            return np.zeros_like(np.asarray(s, dtype=float))
        return np.arctan2(self._sy(s, 1), self._sx(s, 1))

    def kappa(self, s):
        # self._k < 2 -> the fit is a straight segment (see the degenerate-plan guard): curvature
        # is exactly zero, and splev would raise rather than say so.
        if self.degenerate or self._k < 2:
            return np.zeros_like(np.asarray(s, dtype=float))
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


def _wrap_pi(a):
    """Wrap an angle to (-pi, pi]. e_psi is a heading DIFFERENCE, and without this a path heading
    of +179 deg against an ego heading of -179 deg reads as a 358 deg error instead of 2 deg --
    which would saturate the steering the wrong way at exactly the moment the two agree."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def route_override_decision(waypoints, target, aim_dist=4.0, angle_thresh=0.3, dist_thresh=10.0):
    r"""Stock PIDController.control_pid()'s own `use_target_to_aim` test, ported verbatim.

    Reference: Bench2DriveZoo/team_code/pid_controller.py:52-82. Same aim_dist/angle_thresh/
    dist_thresh defaults, same two conditions, same normalized angle units (1.0 == 90 deg):

        use = |angle_target| < |angle_aim|                       # the route point is straighter
           or (|angle_target - angle_last| > angle_thresh        # ...or a sudden turn command
               and target[1] < dist_thresh)                      #    that is already close

    The first condition is the one that fires constantly (measured over the ability-10 set: 34.3%
    of all ticks) and is almost always a no-op -- median disagreement 0.1-0.9 deg, i.e. pure noise
    suppression on straight roads. The second is the one that matters: on route 27582 the pair
    fired on 72.7% of ticks with a MEDIAN disagreement of 10.0 deg, and that is the one route where
    this controller drove off the drivable surface while stock PID completed at DS 100.

    waypoints/target are both in VAD's own [lateral, forward] ego frame (index 0 sideways, index 1
    ahead) -- the convention control_mpc() already receives. Note the useful identity

        degrees(pi/2 - atan2(fwd, lat)) / 90 * 90deg  ==  PathSpline.yaw  (radians, this file's
                                                          own (forward, lateral) fit)

    because pi/2 - atan2(y, x) == atan2(x, y) exactly. So these normalized angles are the SAME
    quantity the lateral plant's e_psi is built from -- no unit conversion and no sign flip is
    needed to move between the two, which is why the port is literal.

    Returns (use_target, angle_aim, angle_target, angle_last), the last three in stock's own
    normalized units so they can be logged and compared against a stock PID dump directly.
    """
    wps = np.asarray(waypoints, dtype=float)
    if wps.ndim != 2 or len(wps) < 2 or target is None:
        return False, 0.0, 0.0, 0.0
    tgt = np.asarray(target, dtype=float).reshape(-1)

    # aim point: the waypoint whose midpoint-norm is closest to aim_dist (stock's own loop).
    best_norm = 1e5
    aim = wps[0]
    for i in range(len(wps) - 1):
        norm = float(np.linalg.norm((wps[i + 1] + wps[i]) / 2.0))
        if abs(aim_dist - best_norm) > abs(aim_dist - norm):
            aim, best_norm = wps[i], norm
    aim_last = wps[-1] - wps[-2]

    def _ang(v):
        return math.degrees(math.pi / 2 - math.atan2(float(v[1]), float(v[0]))) / 90.0

    angle = _ang(aim)
    angle_last = _ang(aim_last)
    angle_target = _ang(tgt)
    use = (abs(angle_target) < abs(angle)) or \
          (abs(angle_target - angle_last) > angle_thresh and float(tgt[1]) < dist_thresh)
    return bool(use), angle, angle_target, angle_last


class GlobalPath:
    r"""The route itself as an arc-length-parameterized reference, in CARLA WORLD coordinates.

    This is the reference the lateral plant was designed for and has never had. VAD's own plan is
    re-anchored to the ego every tick, so x0's e_y is 0 by construction -- one of the four states
    carries no measurement at all and the controller is a heading regulator, not a path follower.
    Projecting onto a fixed world-frame route makes e_y a real signed quantity for the first time.

    Frame: CARLA's world is left-handed with x forward and y to the RIGHT of a yaw-0 heading
    (Transform.get_right_vector() at yaw 0 is (0, 1, 0)), so the right normal at heading psi is
    (-sin psi, cos psi) and e_y is POSITIVE WHEN THE EGO IS RIGHT OF THE PATH. That matches this
    file's existing ego-frame convention: VAD's out_truck index 0 (+lateral) is also right --
    verified on 3317 real ticks, corr(path.yaw(0), steer) = +0.40 with CARLA steer +1 = right.
    e_psi keeps the same definition as the VAD-frame path (psi_ego - psi_path), so the two
    reference sources are interchangeable in x0 without touching the plant.

    Build it ONCE per route: the route is static, so the spline fit is a one-time cost (0.5-8 ms
    for 500-5000 m). Per tick only project() runs, measured at 0.043-0.059 ms -- about 1% of the
    lateral QP's own 5.52 ms.

    smoothing_per_point scales UnivariateSpline's total-residual bound with the point count, since
    that bound is a SUM over points and a route can be any length; 1e-3 m^2/point is an RMS
    residual near 3 cm, enough to take the centimetre-level jitter out of the second derivative
    without cutting corners at junctions.
    """

    def __init__(self, xy_world, smoothing_per_point=1e-3, min_spacing=0.05):
        P = np.asarray(xy_world, dtype=float).reshape(-1, 2)
        if len(P) >= 2:
            keep = np.concatenate([[True], np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])) > min_spacing])
            P = P[keep]
        if len(P) < 4:
            raise ValueError("GlobalPath needs at least 4 distinct points, got %d" % len(P))
        self.P = P
        self.s_nodes = _chord_length_station(P[:, 0], P[:, 1])
        self.path = PathSpline(P[:, 0], P[:, 1],
                               smoothing=smoothing_per_point * len(P), min_spacing=min_spacing)
        self.s_max = float(self.path.s_max)

    def kappa(self, s):
        return self.path.kappa(np.clip(s, 0.0, self.s_max))

    def project(self, ego_xy, s_hint=None, back=6.0, ahead=40.0):
        """(s_star, e_y, psi_path) for the ego's current world position.

        s_hint restricts the search to [s_hint - back, s_hint + ahead]. That window is not an
        optimisation -- it is required for correctness. Routes double back on themselves (a
        U-turn, or two legs of a loop running one lane apart), and a global argmin there snaps to
        whichever leg happens to be nearer, which teleports the station cursor and the whole
        curvature preview to a different part of the map. With a hint the cursor can only advance.
        """
        ego = np.asarray(ego_xy, dtype=float).reshape(2)
        if s_hint is None:
            lo, hi = 0, len(self.P)
        else:
            lo = int(np.searchsorted(self.s_nodes, float(s_hint) - back))
            hi = int(np.searchsorted(self.s_nodes, float(s_hint) + ahead)) + 1
            lo = max(0, min(lo, len(self.P) - 2))
            hi = max(lo + 2, min(hi, len(self.P)))
        d = self.P[lo:hi] - ego
        i = lo + int(np.argmin(np.einsum("ij,ij->i", d, d)))

        # Refine off the node onto whichever adjacent segment the ego actually falls on, so s_star
        # is continuous rather than quantised to the 1 m node spacing (a quantised station makes
        # the curvature preview step instead of slide, and the MPC sees that as a disturbance).
        s_star = float(self.s_nodes[i])
        best = float("inf")
        for j in (i - 1, i):
            if j < 0 or j + 1 >= len(self.P):
                continue
            a, b = self.P[j], self.P[j + 1]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 <= 1e-12:
                continue
            t = float(np.clip((ego - a) @ ab / L2, 0.0, 1.0))
            proj = a + t * ab
            dist = float(np.hypot(*(ego - proj)))
            if dist < best:
                best = dist
                s_star = float(self.s_nodes[j] + t * (self.s_nodes[j + 1] - self.s_nodes[j]))

        s_star = float(np.clip(s_star, 0.0, self.s_max))
        psi = float(self.path.yaw(s_star))
        px, py = self.path.xy(s_star)
        # signed offset onto the path's RIGHT normal (-sin psi, cos psi)
        e_y = float(-(ego[0] - float(px)) * math.sin(psi) + (ego[1] - float(py)) * math.cos(psi))
        return s_star, e_y, psi


def preview_from_splines(path, vx_spline, s_max_wp, n_p, dt, vx_eps=VX_EPS, ay_max=None,
                         kappa_path=None, kappa_s0=0.0):
    """Walk a station cursor forward by vx(s_cursor)*dt each step (same technique
    mpc_mpc_KF.py's curvature_preview() uses, extended to also sample vx(s) instead of taking it
    as a given input) -- produces (vx_preview, kappa_preview), both length n_p, off the SAME
    station parameterization so they stay consistent with each other tick to tick.

    ay_max (m/s^2, None disables): carla_control/functions.py's own refine_speed_preview() cap,
    v_target[k] = min(v[k], sqrt(ay_max/|kappa_k|)) -- lets the LONGITUDINAL side see a curve coming
    and slow for it, instead of only reacting laterally after the fact. Folded into this walk rather
    than applied as a separate post-pass over the returned array (which is how functions.py does it,
    against its own reference_preview() output): refine_speed_preview() re-walks its own station
    cursor using the refined speeds, so bolting it on afterward would leave vx_preview sampled at
    different stations than kappa_preview -- exactly the consistency this function's own docstring
    exists to guarantee. Capping in-loop keeps one cursor and one parameterization, and the
    self-consistency refine_speed_preview()'s docstring asks for (slowing now means arriving at a
    later station later) falls out of the same walk.

    kappa_path/kappa_s0 (None keeps the old behaviour exactly): take CURVATURE from a different
    reference than the one vx comes from, starting at station kappa_s0 on it. This is what the
    route-override branch needs -- geometry from the global route, speed profile still from VAD's
    own plan (VAD is the only thing that knows about the red light, the pedestrian and the lead
    car, and the override test says nothing at all about speed). The single cursor stays valid
    across the two because it is a DISTANCE TRAVELLED and both references are arc-length
    parameterized: after ds metres the ego is at s=ds on VAD's plan and s=kappa_s0+ds on the
    route. The two curves do diverge, so ds along one is not exactly ds along the other, but the
    override only engages when they disagree by a heading, not a length, and over a 25-step
    horizon the difference is far below the spacing of either fit."""
    s_cursor = 0.0
    kpath = kappa_path if kappa_path is not None else path
    ks0 = float(kappa_s0) if kappa_path is not None else 0.0
    vx_preview = np.zeros(n_p)
    kappa_preview = np.zeros(n_p)
    for j in range(n_p):
        s_kappa = min(ks0 + s_cursor, kpath.s_max)
        s_v = min(s_cursor, s_max_wp)
        kappa = float(kpath.kappa(s_kappa))
        kappa_preview[j] = kappa
        v = max(float(vx_spline(s_v)), vx_eps)
        if ay_max is not None and abs(kappa) > 1e-6:
            v = min(v, math.sqrt(ay_max / abs(kappa)))
        # re-floor: the curve cap must not drive the previewed speed to zero, or the station
        # cursor stops advancing and the whole horizon collapses onto s = 0.
        vx_preview[j] = max(v, vx_eps)
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


class ImuAcceleration:
    """Copy of carla_control/functions.py's ImuAcceleration -- unchanged math and unchanged
    defaults (tau=0.15, settle_ticks=3); see that class's own docstring for the full reasoning.
    The short version of why this must be here rather than skipped: on this build the accelerometer
    reports around -378,000 m/s^2 on the first tick after spawn, and a single such sample entering
    a causal tau=0.15 s low-pass leaves roughly -94,000 in the filter state, which then needs about
    two seconds to decay back under 1 m/s^2. So the opening ticks are dropped outright (never
    entering the filter state) instead of clamped -- a magnitude clamp would silently flatten
    genuine hard braking while only partially taming a 1e5 spike.

    Only the pieces this deployment actually consumes are kept: a_y (filtered, the Kalman filter's
    measurement) and a_x_raw (unfiltered, what the LUT+PID pedal layer wants -- see
    longitudinal_mpc.py's own note that filtering only that side would make LookupController
    compare a_cmd against a lagged a_meas). a_x (filtered) is kept too since it costs one filter and
    keeps this a faithful copy; the jerk channels are dropped, nothing here uses them.

    step_xy(ax, ay) is the entry point this deployment uses: a leaderboard agent gets
    input_data['IMU'][1] as a numpy array, not a carla IMU measurement object. step(imu_data) is
    kept for callers that do have the object, and just delegates."""

    def __init__(self, dt, tau=0.15, settle_ticks=3):
        self.dt = dt
        self.settle_ticks = settle_ticks
        self._ticks = 0
        self._accel_x = LowPassFilter(tau=tau, dt=dt, initial=0.0)
        self._accel_y = LowPassFilter(tau=tau, dt=dt, initial=0.0)
        self.a_x_raw = self.a_y_raw = 0.0
        self.a_x = self.a_y = 0.0

    def step_xy(self, ax, ay):
        self._ticks += 1
        if self._ticks <= self.settle_ticks:
            # Hold everything at zero and, critically, do not let these samples into the filter
            # state -- that is the whole point of skipping them.
            return self
        self.a_x_raw = float(ax)
        self.a_y_raw = float(ay)
        self.a_x = self._accel_x.step(self.a_x_raw)
        self.a_y = self._accel_y.step(self.a_y_raw)
        return self

    def step(self, imu_data):
        return self.step_xy(imu_data.accelerometer.x, imu_data.accelerometer.y)


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

    def __init__(self, dt=DT, n_p=25, n_c=25,
                w_ey=10.0, w_epsi=10.0, w_ay=1.0, w_r=3.0, w_rdot=30.0,
                w_delta=1.0, w_ddelta=30.0, delta_max_deg=30.0, ddelta_max_deg=70.0,
                kf_q_vy=1e-8, kf_q_r=1e-8, kf_r_dpsi=1e-5, kf_r_ay=1e-5,
                max_throttle=1.0,
                n_p_speed=40, n_c_speed=40, w_v=10.0, w_a=1.0, w_j=30.0,
                a_min=-4.05, a_max=2.4, jerk_max=4.13, ay_max=4.9,
                lut_path=None, pedal_kp=0.15, pedal_ki=0.6, pedal_kd=0.0, pedal_u_tau=0.02):
        # Lateral weights/horizons are carla_control's own tuned argparse defaults (the
        # --lat-np/--lat-nc/--w-ey/--w-epsi/--w-ay/--w-r/--w-rdot/--w-delta/--w-ddelta group),
        # adopted verbatim rather than the earlier hand-set w_ey=4000/w_epsi=100/w_ay=w_r=w_rdot=0
        # this file shipped with. Two of those defaults contradict their own --help prose (which
        # still describes an older exploration): --w-ay's help says "default 0" but the default is
        # 1, and --ay-max's help says 4.15 was chosen but the default is 4.9. The DEFAULTS are what
        # is used here, since those are what the tuning search actually left in place; flip w_ay to
        # 0.0 / ay_max to 4.15 here if the prose is the intended config instead.
        #
        # w_rdot: 120 -> 30. That default is not the yaw-comfort term it looks like. r_dot is an
        # OUTPUT with a direct feedthrough from delta (D = lf*Cf/Iz = 33.33 rad/s^2 per rad), so
        # weighting it puts an effective w_rdot * 33.33^2 = 133,316 penalty on steering itself --
        # against w_delta = 1 and w_epsi = 10. Measured consequence on route 27582's junction: at
        # the tick needing 21.2 deg of steer the QP commanded 12.3, and its own horizon predicted
        # e_psi diverging to -25.4 deg while it accepted that plan, because shedding heading error
        # (10 * 0.44^2 ~ 2 per step) is far cheaper than steering for it (~3000 per step). At
        # w_rdot = 12 the same solve predicts -16.0 deg instead, and the planned ramp goes from
        # 0.29 to 0.53 deg/tick. 30 is the conservative middle: it keeps real yaw-acceleration
        # damping (the Comfortness channels this weight does also serve) while cutting the
        # steering penalty ~4x. w_r and w_ddelta were measured to have no effect here and are
        # deliberately left alone so this change stays a single variable.
        self.dt, self.n_p, self.n_p_speed = dt, n_p, n_p_speed
        self.ay_max = ay_max
        self.lateral_mpc = LateralMPC(
            dt=dt, n_p=n_p, n_c=n_c, mass=VEHICLE_MASS, Iz=VEHICLE_IZ, lf=VEHICLE_LF, lr=VEHICLE_LR,
            Cf=VEHICLE_CF, Cr=VEHICLE_CR, w_ey=w_ey, w_epsi=w_epsi, w_ay=w_ay, w_r=w_r, w_rdot=w_rdot,
            w_delta=w_delta, w_ddelta=w_ddelta,
            delta_max=math.radians(delta_max_deg), ddelta_max=math.radians(ddelta_max_deg) * dt,
            vx_eps=VX_EPS)
        # n_p_speed/n_c_speed default to mpc_mpc_comparison.py's own --np/--nc (40), decoupled from
        # LateralMPC's n_p (--lat-np, 25) -- see module docstring point 5 for how one preview walk
        # still feeds both.
        self.speed_mpc = SpeedMPC(dt=dt, n_p=n_p_speed, n_c=n_c_speed, w_v=w_v, w_a=w_a, w_j=w_j,
                                  a_min=a_min, a_max=a_max, jerk_max=jerk_max)
        self.kf = VyKalmanFilter(dt=dt, mass=VEHICLE_MASS, Iz=VEHICLE_IZ, lf=VEHICLE_LF, lr=VEHICLE_LR,
                                 Cf=VEHICLE_CF, Cr=VEHICLE_CR, Q=np.diag([kf_q_vy, kf_q_r]),
                                 R=np.diag([kf_r_dpsi, kf_r_ay]), vx_eps=VX_EPS,
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

        # Route-override state (see control_mpc()'s `target`/`global_path` arguments).
        # _route_s carries the station cursor forward so project() can search a window instead of
        # the whole route; _override_hold implements the release delay described below.
        self._route_s = None
        self._override_hold = 0
        self.override_hold_ticks = 5      # 0.25 s at 20 Hz

    def control_mpc(self, waypoints, speed, r_meas, ay_meas, gear, a_meas, physics=None,
                    target=None, ego_xy=None, ego_yaw=None, global_path=None):
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
        gear/a_meas being required now.

        target/ego_xy/ego_yaw/global_path (all optional, all four needed together) enable the
        ROUTE OVERRIDE. target is local_command_xy, the route planner's next command point in
        VAD's own [lateral, forward] ego frame. ego_xy/ego_yaw are the ego's CARLA WORLD pose
        (metres, radians) -- read them straight off the hero actor's transform, not off the
        compass/GPS pair, so they land in the same frame as global_path with no convention to get
        wrong. global_path is a GlobalPath built once per route.

        When route_override_decision() fires, the LATERAL reference (e_y, e_psi, curvature
        preview) switches from VAD's plan to the route; the SPEED preview never does. See that
        function and GlobalPath for why, and preview_from_splines()'s kappa_path argument for how
        the two parameterizations share one station cursor. With any of the four missing this is
        exactly the old VAD-only controller -- which is also the deliberate fallback for the ticks
        before the hero actor exists.

        Returns (steer, throttle, brake, metadata), same shape PIDController.control_pid()
        returns. Named control_mpc(), not control_pid(), since this class is MPC-based -- see
        class docstring."""
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

        # ------------------------------------------------------------------ route override
        # Project onto the route on EVERY tick when one is available, not only while overriding:
        # the station cursor has to keep advancing or its search window goes stale, and having e_y
        # logged even in VAD mode is what makes it possible to tell afterwards whether the ego was
        # drifting off the route before the override ever fired.
        route_s, route_e_y, route_psi = None, None, None
        if global_path is not None and ego_xy is not None:
            try:
                route_s, route_e_y, route_psi = global_path.project(ego_xy, self._route_s)
                self._route_s = route_s
            except Exception:
                route_s, route_e_y, route_psi = None, None, None

        fired, ang_aim, ang_target, ang_last = route_override_decision(waypoints, target)
        # Release delay, not a symmetric hysteresis band: the trigger is a disagreement between two
        # angles that both move every tick, so it chatters on and off around the threshold. Latching
        # for a few ticks after it last fired keeps the reference from alternating at 20 Hz, which
        # the plant would see as a disturbance rather than a command. Engaging is instant -- the
        # case this exists for is a sudden turn command, and delaying that defeats the point.
        if fired:
            self._override_hold = self.override_hold_ticks
        elif self._override_hold > 0:
            self._override_hold -= 1
        use_route = (self._override_hold > 0 and route_s is not None and ego_yaw is not None)

        if use_route:
            e_y0 = route_e_y
            e_psi0 = _wrap_pi(float(ego_yaw) - route_psi)
            kappa_path, kappa_s0, ref_source = global_path, route_s, "global"
        else:
            # VAD reference: the fit passes through the ego by construction (the origin is its
            # first point), so e_y is 0 -- measured, not assumed: the smoothing spline's offset at
            # s=0 is median 1.3 mm / max 5.3 cm over 2729 real ticks.
            e_y0 = 0.0
            e_psi0 = -float(path.yaw(0.0))
            kappa_path, kappa_s0, ref_source = None, 0.0, "vad"

        # One station-walk feeds both MPCs, evaluated at the longer of the two horizons -- module
        # docstring point 5 -- LateralMPC then takes the matching-length PREFIX of the same array
        # rather than a second, independently-walked preview (n_p_speed=40 > n_p=25 by default, so
        # in practice this walks 40 steps and lateral uses steps 0-24 of it). The ay_max curve cap
        # is applied inside that one walk, so the speed profile the LONGITUDINAL MPC tracks already
        # knows about upcoming curvature, and the lateral prefix stays consistent with it.
        n_p_preview = max(self.n_p, self.n_p_speed)
        vx_preview_full, kappa_preview_full = preview_from_splines(
            path, vx_spline, s_max_wp, n_p_preview, self.dt, ay_max=self.ay_max,
            kappa_path=kappa_path, kappa_s0=kappa_s0)
        vx_preview_lat, kappa_preview_lat = vx_preview_full[:self.n_p], kappa_preview_full[:self.n_p]
        vx_preview_speed = vx_preview_full[:self.n_p_speed]

        # e_psi is the EGO's heading error against the path (psi_ego - psi_path), which is the sign
        # convention the [v_y, r, e_y, e_psi] plant is built around (e_psi_dot = r - vx*kappa).
        # The ego frame is re-anchored every tick, so psi_ego == 0 and e_psi == -path.yaw(0) --
        # NEGATED. Passing +path.yaw(0) inverts the dominant feedback term: measured on a straight
        # path aimed 8 deg RIGHT (zero curvature), it commanded 3.50 deg of LEFT steer. The
        # curvature feedforward (y_ref's vx^2*kappa / vx*kappa) is unaffected and was already
        # correct -- a right-curving plan with yaw(0)~0 gives +1.10 deg, the right way -- which is
        # why this only showed up once VAD's plan had a real heading offset at s=0.
        # This is what drove route 28154 off the road: at the frame before impact VAD's plan started
        # +3.7 deg to the right and the controller answered with -7.26 deg of left steer.
        # e_y0/e_psi0 come from whichever reference won above. Both definitions agree by
        # construction: e_psi is psi_ego - psi_path in either frame (the VAD fit is written in an
        # ego frame where psi_ego == 0, hence the bare negation there), and e_y is the offset onto
        # the path's right normal, which is 0 when the path is defined to pass through the ego.
        x0 = [v_y, r, e_y0, e_psi0]
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
        # max_throttle defaults to 1.0, NOT the 0.75 this file shipped with. That 0.75 came from
        # Bench2Drive's stock team_code/pid_controller.py (PIDController's own max_throttle) and
        # its agent's `np.clip(throttle, 0, 0.75)`; carla_control's own functions.control_input()
        # applies u uncapped, and the leaderboard enforces no throttle limit of its own.
        #
        # It mattered because the cap sat OUTSIDE the pedal loop. LookupController.step() clips to
        # +-1 and reports .saturated against +-1, PID.step()'s anti-windup guard is also +-1, and
        # the LUT is calibrated over the full u in [0, 1] in every gear -- so in the (0.75, 1.0]
        # band the feedforward promised an acceleration the pedal never delivered, the tracking
        # error never cleared, and the integrator kept winding with nothing watching. Measured on
        # route 28154: 4 of 71 dumped ticks pinned at the cap, worst case asking 0.996 and getting
        # 0.75 while accelerating out of a stop (a_cmd +2.07, near a_max).
        # At 1.0 the loop's own +-1 bound IS the real limit, so this clip is a no-op and the
        # anti-windup is honest again. Setting it below 1.0 reintroduces the same blind spot --
        # if that is ever wanted, push the bound INTO LookupController.step()/PID.step() instead
        # of clipping here.
        throttle = float(np.clip(max(u, 0.0), 0.0, self.max_throttle))
        brake = float(np.clip(max(-u, 0.0), 0.0, 1.0))

        metadata = {
            'speed': speed, 'steer': steer, 'throttle': throttle, 'brake': brake,
            'v_y_hat': v_y, 'delta_rad': delta, 'a_cmd': a_cmd, 'u_raw': u_raw, 'u_filtered': u,
            'pedal_saturated': self.pedal_ctrl.saturated, 'gear': int(gear),
            'steer_physics_corrected': physics is not None,
            'kf_status': self.lateral_mpc.last_status, 'speed_mpc_status': self.speed_mpc.last_status,
            # Route-override diagnostics. ref_source is the one field that says which geometry the
            # steering actually came from on this tick -- without it a run cannot be attributed
            # after the fact, since the override leaves no other trace in the control signal.
            'ref_source': ref_source, 'e_y': float(e_y0), 'e_psi': float(e_psi0),
            'override_fired': bool(fired), 'override_hold': int(self._override_hold),
            'route_s': (None if route_s is None else float(route_s)),
            'route_e_y': (None if route_e_y is None else float(route_e_y)),
            'angle_aim': float(ang_aim), 'angle_target': float(ang_target),
            'angle_last': float(ang_last),
        }
        return steer, throttle, brake, metadata
