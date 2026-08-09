r"""LPV-MPC lateral controller (bicycle-model error dynamics), scored against stanley_mpc.py's own
Stanley law under an identical longitudinal stack. The point of this file is to validate the
lateral MPC itself, so --controller isolates the lateral half (mpc vs stanley) while longitudinal
defaults to the same stack both ways -- mpc+mpc vs stanley+mpc, not a longitudinal comparison. Code
shape follows stanley_mpc.py (path source, warm-up gate, BEV/video, plot_results,
--controller overlay-comparison); Stanley itself is imported from there, not reimplemented, so the
comparison partner is exactly stanley_mpc.py's own lateral half, not a second copy of it.

    lateral+longitudinal, "+"-joined, any combination of:
        mpc       (default) LateralMPC (this file): a QP solved every cycle over the LPV bicycle
                  model below.
        stanley   stanley_mpc.stanley_control() -- imported, front-axle referenced, same gains.
      x
        mpc       (default) SpeedMPC (longitudinal_mpc.py) -> LUT-feedforward + PID pedal layer.
                  MpcLongitudinal/PidLongitudinal are imported from stanley_mpc.py rather than
                  redefined -- one longitudinal implementation, not two.
        pid       plain speed PID straight to the pedal, no MPC.

    --controller mpc+mpc stanley+mpc   (default) does the lateral MPC beat Stanley?
    --controller mpc+mpc mpc+pid       does the lateral MPC's v_x preview (needs mpc+mpc's own
                                        speed plan) matter, or is mpc+pid's flat-speed fallback fine?

Lateral model (from the design given for this file): 2-DOF bicycle model, state x = [v_y, r, e_y,
e_psi]^T (r = yaw rate), input delta (front steer, rad), disturbance kappa (path curvature, 1/m):

    x_dot = A(v_x) x + B delta + E(v_x) kappa

A(v_x) depends on the (previewed) longitudinal speed, so it is re-linearized and re-discretized at
every predicted step -- the "LPV" in the name, as opposed to SpeedMPC's fixed-A LTI model. See the
LateralMPC docstring for A/B/E/C/D and the condensed QP.

e_y/e_psi sign convention: both are VEHICLE MINUS PATH (positive e_y means the vehicle sits toward
its own body-frame -y axis relative to the path -- the same axis run_trial() already computes v_y
along; positive e_psi means vehicle heading is ahead of path heading), which is the opposite sign to
functions.lateral_error()'s path-minus-vehicle convention that the rest of this repo logs. This
matters here because it is exactly the convention the given error dynamics (e_y_dot = v_y +
v_x*e_psi, e_psi_dot = r - v_x*kappa) were derived in -- get it backwards and the controller steers
away from the path instead of toward it. hist still logs functions.lateral_error()'s raw sign (i.e.
the negative of the QP's internal e_y) so plots stay comparable with stanley_PID.py/stanley_mpc.py;
only the state fed to the QP is flipped.

v_x preview for the LPV schedule: rather than assume constant speed over the horizon, this reuses
whatever SpeedMPC is actually planning to do (its own condensed prediction, exposed by
stanley_mpc.MpcLongitudinal as ctx.v_x_preview) truncated/padded to the lateral horizon. Under
mpc+pid, there is no such plan, so the schedule falls back to holding the current v_x constant.

kappa preview: path curvature is precomputed once per point (build_path_curvature) from the same
path_yaw heading_from_points() already produces, then looked up ahead along predicted arc length
(curvature_preview) using the same v_x preview.

Cf, Cr, Iz, mass default to lateral_parameter/'s estimated values (cornering-stiffness roundabout
test + step-steer transient test), not guesses -- --cf/--cr/--iz/--mass override them if needed.
lf/lr/max_steer are queried live off the spawned vehicle (get_vehicle_geometry), not from that file,
since those are exact CARLA physics properties, not something that needed estimating.

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python mpc_mpc.py --times-run 1 --target-speed 10
    .venv/bin/python mpc_mpc.py --controller mpc+mpc stanley+mpc --times-run 20 --save-plot --target-speed 10

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe mpc_mpc.py --times-run 1 --target-speed 10
    .venv\Scripts\python.exe mpc_mpc.py --controller mpc+mpc stanley+mpc --times-run 20 --save-plot --target-speed 10
    .venv\Scripts\python.exe mpc_mpc.py --controller mpc+mpc stanley+mpc --profile estop --stop-time 10 --stop-duration 5 --save-plot --target-speed 10
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

# see functions.py for why this path is needed alongside the pip-installed carla package
CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/2026intern/carla"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.append(os.path.join(HERE, "longitudinal_lookup"))

import carla

from functions import (AngleUnwrapper, LowPassFilter, build_path, clipping, get_vehicle_geometry,
                       lateral_error, normalize_angle)
from viz_utils import (BevView, VIEWS, VideoRecorder, follow_with_spectator, plot_results,
                       print_error_summary, run_name)

from stanley_mpc import MpcLongitudinal, PidLongitudinal, front_axle_offset, stanley_control

# Fixed on purpose, same reasoning as stanley_mpc.py: the route (build_path()'s default
# origin/dest spawn indices) is a property of this specific map.
MAP_NAME = "Town10HD_Opt"

MAX_PLAUSIBLE_ACCEL = 8.0  # m/s^2 -- see longitudinal_mpc.py/validate_lut.py

WARM_START_SPEED_TOL = 0.3   # m/s
WARM_START_ACCEL_TOL = 0.5   # m/s^2
WARM_START_TIMEOUT = 15.0    # s -- safety cap in case target_speed is unreachable


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
        return dict(mass=1696.0, iz=1500.0, cf=40000.0, cr=50000.0)


VEHICLE_DEFAULTS = _load_vehicle_defaults()


# ----------------------------------------------------------------------------- lateral MPC (LPV)

class LateralMPC:
    r"""LPV-MPC lateral controller over the 2-DOF bicycle-model error dynamics

        x = [v_y, r, e_y, e_psi]^T,  input delta (front steer, rad),  disturbance kappa (1/m)

        x_dot = A(v_x) x + B delta + E(v_x) kappa

            A(v_x) = [[-(Cf+Cr)/(m vx),        (lr Cr - lf Cf)/(m vx) - vx, 0, 0],
                      [(lr Cr - lf Cf)/(Iz vx), -(lf^2 Cf + lr^2 Cr)/(Iz vx), 0, 0],
                      [1,                       0,                            0, vx],
                      [0,                       1,                            0, 0]]
            B = [Cf/m, lf Cf/Iz, 0, 0]^T                E(v_x) = [0, 0, 0, -vx]^T

    v_x is the scheduling parameter -- previewed over the horizon (see vx_preview_for_lateral()),
    not held fixed, so A/E/C below are re-linearized and re-discretized at every predicted step.

    Output y = [e_y, e_psi, a_y, r, r_dot]^T tracks a reference built from the previewed curvature
    (e_y_ref = e_psi_ref = r_dot_ref = 0, r_ref = v_x kappa, a_y_ref = v_x^2 kappa -- steady-turn
    feedforward):

        C(v_x) = [[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [-(Cf+Cr)/(m vx), (lr Cr - lf Cf)/(m vx), 0, 0],
                  [0, 1, 0, 0],
                  [(lr Cr - lf Cf)/(Iz vx), -(lf^2 Cf + lr^2 Cr)/(Iz vx), 0, 0]]
        D = [0, 0, Cf/m, 0, lf Cf/Iz]^T

    (a_y's row/D: a_y = v_y_dot + v_x r, substitute the v_y_dot state row and the -v_x+v_x terms
    cancel. r_dot's row/D is just the r_dot state row directly -- r_dot is not a state itself.)

    Forward-Euler per predicted step k: A_d,k = I + Ts A(v_x,k), B_d = Ts B (v_x-independent),
    E_d,k = Ts E(v_x,k). Condensing over the horizon with a Nc-step control horizon (free for Nc
    steps, held after) gives X = Abar x0 + Bbar U + Ebar K, Y = Cbar X + Dbar U. Bbar/Dbar are built
    by condensing as if every one of the Np steps had its own free input, then right-multiplying by
    a Np x Nc holding matrix Mc that maps the Nc free values plus a held tail onto those Np raw
    columns -- reproduces the recursive hold-column accumulation without writing it out by hand
    twice. Ebar is condensed the same way but never blocked (kappa is a known preview, not a
    decision variable).

    Cost J = (Y-Yref)' W1 (Y-Yref) + U' W2 U + (GU)' W3 (GU), G the (Nc-1) x Nc first-difference
    operator on U. Reduces to the box/rate-constrained QP

        min 1/2 U' H U + f' U   s.t.  -delta_max <= U <= delta_max,  -ddelta_max <= GU <= ddelta_max
        H = 2[(CbarBbar+Dbar)' W1 (CbarBbar+Dbar) + W2 + G' W3 G]
        f = 2(CbarBbar+Dbar)' W1 (Cbar Abar x0 + Cbar Ebar K - Yref)

    Only the first element of U* is applied each cycle (receding horizon). Unlike SpeedMPC, H is
    LPV (depends on the v_x preview, which changes every cycle) so the QP -- osqp problem included
    -- is rebuilt from scratch on every solve() call rather than reused with just q/l/u updated.

    w_ay/w_r/w_rdot default to 0 (verified offline against this same model, not guessed): weighting
    a_y/r/r_dot toward the steady-turn feedforward (v_x^2 kappa, v_x kappa, 0) as independent output
    targets fights e_y/e_psi's own targets in a curve. The true e_y=0 steady state on a curve needs
    a small nonzero e_psi (sideslip-dependent, e_psi_ss = -v_y_ss/v_x, not the 0 this class's e_psi
    reference asks for); forcing r/a_y toward the feedforward pulls the solution toward e_psi=0
    instead, and since e_y accumulates any sustained e_psi bias (e_y_dot = v_y + v_x e_psi), a small
    angular bias compounds into meters of steady cross-track offset (observed: ~2m on a 50m-radius
    turn at these default weights before this was found -- see mpc_mpc.py's own test notes). Cutting
    w_ay/w_r/w_rdot to 0 let e_psi settle wherever e_y=0 actually needs it and the offset dropped to
    a few cm. kappa still reaches the controller either way, through Ebar's open-loop effect on the
    predicted e_y/e_psi trajectory -- these three weights are a knob for extra smoothing on top of
    that, not the thing that makes curves trackable at all.
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
        """x0 = [v_y, r, e_y, e_psi] (vehicle-minus-path sign convention -- see module docstring).
        vx_preview/kappa_preview: length-Np previews. Returns delta_cmd (rad, clipped to
        +-delta_max)."""
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


# ----------------------------------------------------------------------------- curvature preview

def build_path_curvature(path_x, path_y, path_yaw, stencil=5):
    """path_s: cumulative arc length at each path point. path_kappa: signed curvature (1/m) at
    each point, kappa_i = d(path_yaw)/ds via a centred finite difference over arc length -- same
    stencil idea as functions.heading_from_points(), just one derivative further along, and reusing
    the already-unwrapped path_yaw it produced instead of re-deriving heading from scratch."""
    n = len(path_x)
    path_s = [0.0]
    for i in range(1, n):
        path_s.append(path_s[-1] + math.hypot(path_x[i] - path_x[i - 1], path_y[i] - path_y[i - 1]))
    path_kappa = []
    for i in range(n):
        a = max(0, i - stencil)
        b = min(n - 1, i + stencil)
        ds = path_s[b] - path_s[a]
        path_kappa.append(0.0 if ds < 1e-6 else (path_yaw[b] - path_yaw[a]) / ds)
    return path_s, path_kappa


def curvature_preview(path_s, path_kappa, last_idx, vx_preview, dt):
    """Length-Np curvature preview: walk forward along the path by the previewed speed each step
    (arc length s += v_x * dt) and read off path_kappa at whatever point that lands on."""
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


def vx_preview_for_lateral(ctx, n_p_lat):
    """MpcLongitudinal stashes its own condensed speed prediction on ctx.v_x_preview; reuse it
    (truncated/padded to the lateral horizon) instead of assuming constant speed. PidLongitudinal
    sets nothing there, so mpc+pid falls back to holding the current v_x flat over the horizon."""
    raw = getattr(ctx, "v_x_preview", None)
    if raw is None:
        return np.full(n_p_lat, max(ctx.v_x, 0.1))
    raw = np.asarray(raw, dtype=float)
    if len(raw) >= n_p_lat:
        return raw[:n_p_lat]
    return np.concatenate([raw, np.full(n_p_lat - len(raw), raw[-1])])


def speed_reference(args, t):
    if args.profile == "sine":
        return args.target_speed + args.sine_amplitude * math.sin(2.0 * math.pi * t / args.sine_period)
    if args.profile == "estop":
        if args.stop_time <= t < args.stop_time + args.stop_duration:
            return 0.0
        return args.target_speed
    return args.target_speed


# ----------------------------------------------------------------------------- lateral controllers
# Common interface: step(ctx) -> delta (rad). ctx (a SimpleNamespace, set fresh by run_trial() each
# cycle) carries ego_x/ego_y/yaw/v_x/v_y/yaw_rate_rad, path_x/path_y/path_yaw/path_s/path_kappa,
# last_idx (read the incoming value, write the updated one back onto ctx), dt, and -- if the
# longitudinal half is MpcLongitudinal -- v_x_preview. Both stash ctx.raw_e_y/ctx.e_theta (the
# functions.lateral_error()/path-minus-vehicle convention, not the QP's internal sign) for hist
# logging, so the two stacks log directly comparably.

class StanleyLateral:
    """stanley_mpc.py's own front-axle-referenced Stanley law -- imported, not reimplemented, so
    this is exactly the comparison partner it names itself, not a second copy of it."""
    label = "Stanley"
    filter_steer = True   # Stanley has no built-in rate limit, needs run_trial()'s steer_filter

    def __init__(self, vehicle, lf, lr, max_steer, args):
        self.front_offset = front_axle_offset(vehicle, lf)
        if not 0.0 < self.front_offset < lf + lr:
            raise RuntimeError(f"front_offset={self.front_offset:.2f} m is not inside the "
                               f"wheelbase ({lf + lr:.2f} m); the front-axle reference point is wrong.")

    def reset(self):
        pass

    def step(self, ctx):
        front_x = ctx.ego_x + self.front_offset * math.cos(ctx.yaw)
        front_y = ctx.ego_y + self.front_offset * math.sin(ctx.yaw)
        ctx.last_idx, e_y = lateral_error(front_x, front_y, ctx.yaw, ctx.path_x, ctx.path_y, ctx.last_idx)
        e_theta = normalize_angle(ctx.path_yaw[ctx.last_idx] - ctx.yaw)
        ctx.raw_e_y, ctx.e_theta = e_y, e_theta
        return stanley_control(ctx.v_x, e_y, e_theta)


class LpvMpcLateral:
    """LateralMPC (this file) -- CG-referenced, unlike Stanley's front-axle point (see module
    docstring for why e_y/e_psi get sign-flipped going into the QP)."""
    label = "LPV-MPC"
    # The QP already optimizes its own steer-rate limit (--ddelta-max-deg) with G/W3, using U[0]'s
    # continuity with the *previous solve's* planned trajectory. Stacking run_trial()'s steer_filter
    # on top adds a second, unmodeled lag between "what the QP just planned" and "what actually
    # reached the tires" that the QP's own x0/continuity assumption doesn't know about -- found live
    # to be a real source of sustained oscillation (the loop chases its own filtered-away correction
    # every cycle), not just redundant smoothing the way it is for Stanley's unconstrained law.
    filter_steer = False

    def __init__(self, vehicle, lf, lr, max_steer, args):
        self.dt = args.dt
        self.mpc = LateralMPC(
            dt=args.dt, n_p=args.lat_n_p, n_c=args.lat_n_c,
            mass=args.mass, Iz=args.iz, lf=lf, lr=lr, Cf=args.cf, Cr=args.cr,
            w_ey=args.w_ey, w_epsi=args.w_epsi, w_ay=args.w_ay, w_r=args.w_r, w_rdot=args.w_rdot,
            w_delta=args.w_delta, w_ddelta=args.w_ddelta,
            delta_max=max_steer, ddelta_max=math.radians(args.ddelta_max_deg) * args.dt)

    def reset(self):
        pass

    def step(self, ctx):
        ctx.last_idx, raw_e_y = lateral_error(ctx.ego_x, ctx.ego_y, ctx.yaw, ctx.path_x, ctx.path_y,
                                              ctx.last_idx)
        e_theta = normalize_angle(ctx.path_yaw[ctx.last_idx] - ctx.yaw)
        ctx.raw_e_y, ctx.e_theta = raw_e_y, e_theta

        vx_preview = vx_preview_for_lateral(ctx, self.mpc.n_p)
        kappa_preview = curvature_preview(ctx.path_s, ctx.path_kappa, ctx.last_idx, vx_preview, self.dt)
        x0 = [ctx.v_y, ctx.yaw_rate_rad, -raw_e_y, -e_theta]   # vehicle-minus-path (module docstring)
        return self.mpc.solve(x0, vx_preview, kappa_preview)


LATERAL = {"stanley": StanleyLateral, "mpc": LpvMpcLateral}
LONGITUDINAL = {"pid": PidLongitudinal, "mpc": MpcLongitudinal}


# ----------------------------------------------------------------------------- one trial

def run_trial(world, origin_transform, path_x, path_y, path_yaw, path_s, path_kappa, blueprint,
              imu_bp, lateral, longitudinal, args, recorder_factory):
    """Spawn one vehicle, drive the whole path under `lateral` + `longitudinal`, tear it down.
    Returns the run's hist dict.

    Longitudinal runs first each tick (not lateral first, unlike stanley_mpc.py) because
    LpvMpcLateral's schedule needs whatever v_x plan the longitudinal controller just produced;
    StanleyLateral ignores it either way.
    """
    vehicle = world.spawn_actor(blueprint, origin_transform)
    wheelbase, lf, lr, max_steer = get_vehicle_geometry(vehicle, origin_transform)
    lateral_ctrl = lateral(vehicle, lf, lr, max_steer, args)
    label = f"{lateral_ctrl.label}+{longitudinal.label}"

    steer_filter = LowPassFilter(tau=0.1, dt=args.dt, initial=0)
    yaw_unwrapper = AngleUnwrapper()
    rh_unwrapper = AngleUnwrapper()
    last_idx = 0

    accel_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    jerk_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    accel_y_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    jerk_y_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    yaw_acc_filter = LowPassFilter(tau=0.15, dt=args.dt, initial=0.0)
    prev_a_x = None
    prev_a_y = None
    prev_yaw_rate_rad = None

    bev = None if args.no_live_view else BevView(path_x, path_y)

    hist = {"t": [], "x": [], "y": [], "v_x": [], "v_y": [], "v_des": [], "a_x": [], "jerk": [],
            "a_y": [], "yaw_rate": [], "yaw_acc": [], "jerk_total": [], "last_idx": [],
            "steer_deg": [], "throttle": [], "brake": [], "e_y": [], "yaw": [], "path_yaw": [],
            "e_theta": []}

    warmed_up = False
    log_start_i = 0

    imu = None
    recorder = None
    try:
        world.tick()

        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)
        recorder = recorder_factory(vehicle)

        steps = int(args.max_duration / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)

            # ---- sensor data ---- #
            transform = vehicle.get_transform()
            yaw = yaw_unwrapper.step(math.radians(transform.rotation.yaw))
            ego_x = transform.location.x
            ego_y = transform.location.y
            vel_vec = vehicle.get_velocity()

            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)
            v_y = -vel_vec.x * math.sin(yaw) + vel_vec.y * math.cos(yaw)

            a_x_raw = clipping(imu_data.accelerometer.x, MAX_PLAUSIBLE_ACCEL, -MAX_PLAUSIBLE_ACCEL)
            a_x = accel_filter.step(a_x_raw)
            a_y = accel_y_filter.step(imu_data.accelerometer.y)
            yaw_rate_rad = imu_data.gyroscope.z
            yaw_rate = math.degrees(yaw_rate_rad)

            jerk = jerk_filter.step(0.0 if prev_a_x is None else (a_x - prev_a_x) / args.dt)
            prev_a_x = a_x

            jerk_y = jerk_y_filter.step(0.0 if prev_a_y is None else (a_y - prev_a_y) / args.dt)
            prev_a_y = a_y
            jerk_total = math.hypot(jerk, jerk_y)

            yaw_acc = yaw_acc_filter.step(
                0.0 if prev_yaw_rate_rad is None else (yaw_rate_rad - prev_yaw_rate_rad) / args.dt)
            prev_yaw_rate_rad = yaw_rate_rad

            # ---- longitudinal (PID or MPC, per --controller) -- computed first so, under
            # ...+mpc, ctx.v_x_preview is available below for LpvMpcLateral's schedule ---- #
            t_probe = (i - log_start_i) * args.dt
            v_ref = args.target_speed if not warmed_up else speed_reference(args, t_probe)
            ctx = SimpleNamespace(t=t_probe, v_x=v_x, v_ref=v_ref, a_x=a_x, a_x_raw=a_x_raw,
                                  gear=vehicle.get_control().gear, warmed_up=warmed_up,
                                  ego_x=ego_x, ego_y=ego_y, yaw=yaw, v_y=v_y, yaw_rate_rad=yaw_rate_rad,
                                  path_x=path_x, path_y=path_y, path_yaw=path_yaw, path_s=path_s,
                                  path_kappa=path_kappa, last_idx=last_idx, dt=args.dt)
            control_value = longitudinal.step(ctx)

            # ---- lateral (mpc or stanley, per --controller) ---- #
            delta = lateral_ctrl.step(ctx)
            last_idx, raw_e_y, e_theta = ctx.last_idx, ctx.raw_e_y, ctx.e_theta
            road_heading = rh_unwrapper.step(path_yaw[last_idx])
            steer = clipping(delta / max_steer, 1, -1)
            steer_deg = steer * math.degrees(max_steer)

            control = carla.VehicleControl()
            if control_value >= 0:
                control.throttle = control_value
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = -control_value
            control.steer = steer_filter.step(steer) if lateral_ctrl.filter_steer else steer
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            if not warmed_up:
                converged = abs(v_x - args.target_speed) < WARM_START_SPEED_TOL and abs(a_x) < WARM_START_ACCEL_TOL
                timed_out = i * args.dt >= WARM_START_TIMEOUT
                if converged or timed_out:
                    warmed_up = True
                    log_start_i = i
                    longitudinal.reset(control_value)
                    lateral_ctrl.reset()
                    status = "converged" if converged else f"timed out after {WARM_START_TIMEOUT:.0f}s"
                    print(f"[{label}] Warm-start {status}: v_x={v_x:.2f} m/s, "
                          f"a_x={a_x:.2f} m/s^2 -- logging starts now.")
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
            hist["yaw_rate"].append(yaw_rate)
            hist["yaw_acc"].append(yaw_acc)
            hist["jerk_total"].append(jerk_total)
            hist["last_idx"].append(last_idx)
            hist["steer_deg"].append(steer_deg)
            hist["throttle"].append(control.throttle)
            hist["brake"].append(control.brake)
            hist["e_y"].append(raw_e_y)
            hist["yaw"].append(math.degrees(yaw))
            hist["path_yaw"].append(math.degrees(road_heading))
            hist["e_theta"].append(math.degrees(e_theta))

            if i % 5 == 0:
                extra = f"   a_cmd={ctx.a_cmd:+.2f} m/s^2" if hasattr(ctx, "a_cmd") else ""
                print(f"[{label}] t={t:5.1f}s   global_idx={last_idx}/{len(path_x) - 1}   "
                      f"v_x={v_x:5.1f} m/s{extra}   delta={math.degrees(delta):+.2f} deg   "
                      f"e_y={raw_e_y:+.2f} m")

            if last_idx >= len(path_x) - 1:
                print(f"[{label}] Reached end of path (global_idx {last_idx}/{len(path_x) - 1}).")
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    finally:
        if recorder is not None:
            recorder.close()  # before vehicle.destroy(): the camera is attached to it
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        if bev is not None:
            bev.close()
        vehicle.destroy()

    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--controller", nargs="+", default=["mpc+mpc", "stanley+mpc"],
                        choices=("mpc+mpc", "stanley+mpc", "mpc+pid", "stanley+pid"),
                        help="lateral+longitudinal combos to score; default overlays the lateral "
                             "MPC against Stanley under the same (MPC) longitudinal stack, since "
                             "that -- not a longitudinal comparison -- is this file's own point")
    parser.add_argument("--target-speed", type=float, default=5, help="m/s")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--times-run", type=float, default=2.0, help="how times for simulation running?")
    parser.add_argument("--max-duration", type=float, default=100.0, help="safety cutoff (s)")

    parser.add_argument("--profile", default="constant", choices=("constant", "sine", "estop"),
                        help="speed reference shape: flat target-speed, a sine wave around it, or "
                             "an emergency stop (drops to 0) that resumes target-speed after --stop-duration")
    parser.add_argument("--sine-amplitude", type=float, default=3.0, help="sine profile peak deviation (m/s)")
    parser.add_argument("--sine-period", type=float, default=3.0, help="sine profile period (s)")
    parser.add_argument("--stop-time", type=float, default=10.0,
                        help="estop profile: when the emergency stop begins, seconds into the scored run")
    parser.add_argument("--stop-duration", type=float, default=5.0,
                        help="estop profile: how long the reference stays at 0 before resuming target-speed (s)")

    # ---- longitudinal MPC (--controller mpc+mpc) ---- #
    lon = parser.add_argument_group("--controller mpc+mpc (longitudinal)")
    lon.add_argument("--np", dest="n_p", type=int, default=40, help="longitudinal prediction horizon (steps)")
    lon.add_argument("--nc", dest="n_c", type=int, default=40, help="longitudinal control horizon (steps, <= --np)")
    lon.add_argument("--w-v", type=float, default=10.0, help="speed-tracking weight")
    lon.add_argument("--w-a", type=float, default=1, help="commanded-acceleration magnitude weight")
    lon.add_argument("--w-j", type=float, default=10, help="commanded-acceleration rate (jerk) weight")
    lon.add_argument("--a-min", type=float, default=-4.05, help="hard lower bound on a_cmd (m/s^2)")
    lon.add_argument("--a-max", type=float, default=2.4, help="hard upper bound on a_cmd (m/s^2)")
    lon.add_argument("--lut", default=os.path.join(HERE, "longitudinal_lookup", "longitudinal_lut.npz"))
    lon.add_argument("--kp", type=float, default=0.15, help="accel-tracking PID proportional gain")
    lon.add_argument("--ki", type=float, default=0.6, help="accel-tracking PID integral gain")
    lon.add_argument("--kd", type=float, default=0.0, help="accel-tracking PID derivative gain")
    lon.add_argument("--u-tau", type=float, default=0.02,
                     help="low-pass filter time constant on the pedal command u, before it's applied (s)")

    # ---- lateral LPV-MPC ---- #
    lat = parser.add_argument_group("lateral LPV-MPC")
    lat.add_argument("--lat-np", dest="lat_n_p", type=int, default=15, help="lateral prediction horizon (steps)")
    lat.add_argument("--lat-nc", dest="lat_n_c", type=int, default=15, help="lateral control horizon (steps, <= --lat-np)")
    lat.add_argument("--w-ey", type=float, default=25.0, help="cross-track error weight")
    lat.add_argument("--w-epsi", type=float, default=8.0, help="heading error weight")
    lat.add_argument("--w-ay", type=float, default=0.0,
                     help="lateral acceleration tracking weight -- default 0: weighting a_y/r/r_dot "
                          "toward their steady-turn feedforward as hard as e_y/e_psi fights reaching "
                          "e_y=0 in a curve (verified offline -- see LateralMPC docstring), since the "
                          "feedforward's own e_psi companion is nonzero and these force e_psi->0 too")
    lat.add_argument("--w-r", type=float, default=0.0, help="yaw rate tracking weight (see --w-ay)")
    lat.add_argument("--w-rdot", type=float, default=0.0, help="yaw acceleration tracking weight (see --w-ay)")
    lat.add_argument("--w-delta", type=float, default=1.0, help="steer magnitude weight")
    lat.add_argument("--w-ddelta", type=float, default=8.0, help="steer rate weight")
    lat.add_argument("--ddelta-max-deg", type=float, default=120.0, help="hard steer-rate limit (deg/s)")
    lat.add_argument("--mass", type=float, default=VEHICLE_DEFAULTS["mass"], help="vehicle mass (kg)")
    lat.add_argument("--iz", type=float, default=VEHICLE_DEFAULTS["iz"], help="yaw moment of inertia (kg m^2)")
    lat.add_argument("--cf", type=float, default=VEHICLE_DEFAULTS["cf"], help="front cornering stiffness (N/rad)")
    lat.add_argument("--cr", type=float, default=VEHICLE_DEFAULTS["cr"], help="rear cornering stiffness (N/rad)")

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"),
                        help="directory to save the end-of-run result figures into")
    parser.add_argument("--save-plot", action="store_true",
                        help="draw and save the end-of-run result figures (off by default)")
    parser.add_argument("--no-live-view", action="store_true", help="skip the live BEV plan/vehicle view")

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

    origin_transform, path_x, path_y, path_yaw = build_path(world)
    path_s, path_kappa = build_path_curvature(path_x, path_y, path_yaw)
    print(f"Route: {len(path_x)} points, "
          f"start=({path_x[0]:.1f}, {path_y[0]:.1f}) goal=({path_x[-1]:.1f}, {path_y[-1]:.1f})")

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    keys = list(dict.fromkeys(args.controller))   # de-dupe, keep the order given on the CLI

    def recorder_factory(key, n_trials):
        if not args.record:
            return lambda vehicle: None
        suffix = key.replace("+", "-") if n_trials > 1 else ""

        def make(vehicle):
            if args.record == "auto" or n_trials > 1:
                path = os.path.join(args.video_dir, run_name(suffix) + ".mp4")
            else:
                path = args.record
            rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
            return VideoRecorder(world, vehicle, path, fps=1.0 / args.dt, width=rec_w, height=rec_h,
                                 view=args.record_view)
        return make

    results = {}
    try:
        for key in keys:
            lat_key, lon_key = key.split("+")
            lateral_cls, longitudinal_cls = LATERAL[lat_key], LONGITUDINAL[lon_key]
            label = f"{lateral_cls.label}+{longitudinal_cls.label}"
            print(f"\n=== running {label} ===")
            longitudinal = longitudinal_cls(args)
            results[label] = run_trial(world, origin_transform, path_x, path_y, path_yaw,
                                       path_s, path_kappa, blueprint, imu_bp, lateral_cls,
                                       longitudinal, args, recorder_factory(key, len(keys)))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("Cleaned up: world settings restored.")

    have_data = all(len(hist["t"]) > 1 for hist in results.values())
    if not args.save_plot or not have_data:
        for label, hist in results.items():
            if len(results) > 1:
                print(f"\n### {label} ###")
            print_error_summary(hist, args.target_speed)  # plot_results prints it otherwise
    else:
        data = results if len(results) > 1 else next(iter(results.values()))
        title = " vs ".join(results) if len(results) > 1 else next(iter(results), "")
        try:
            plot_results(path_x, path_y, data, args.target_speed, args.plot_dir, label=title)
        except Exception as exc:
            print(f"Plotting failed: {exc}")
            for label, hist in results.items():
                print_error_summary(hist, args.target_speed)


if __name__ == "__main__":
    main()
