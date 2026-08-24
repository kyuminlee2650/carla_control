r"""LPV Kalman filter estimating lateral velocity v_y from yaw-rate and lateral-acceleration
measurements, over the same 2-DOF bicycle-model dynamics mpc_mpc.py's LateralMPC uses for its own
v_y/r rows -- plus the offline tool that replays it against runs mpc_mpc.py logged with
--log-npz, to validate/tune it before mpc_mpc_KF.py ever puts it in the live control loop.

Layout
    1. VyKalmanFilter          the filter itself: state-space + predict/update, no I/O of its own
    2. offline replay + tuning load_run/replay/score/tune() against logged .npz runs
    3. CLI                     `python3 kalman_filter.py [--tune] [--save-plot]`

Why offline replay before closed-loop: mpc_mpc.py's hist already carries ground-truth v_y (read
straight off vehicle.get_velocity() in the vehicle frame), so replaying a logged run through the
filter and comparing v_y_hat against that ground truth is the cheapest way to catch a broken filter
or bad Q/R -- no CARLA server needed, and wrong the first time doesn't cost a drive. "Closed-loop"
in the replay sense: predict() and update() run back to back every tick, each one built on the
filter's own previous corrected estimate, never reset to the logged ground truth mid-run -- not a
series of independent single-step checks. Once this offline check looks right, mpc_mpc_KF.py's
"mpc-kf" controller does the real thing: the filter's own v_y_hat, produced online tick by tick,
actually drives the car (feeds LateralMPC's x0), and ground truth is only logged alongside it for
comparison, not fed back into the filter.

Causality, here and in mpc_mpc_KF.py alike: only delta_prev = delta[k-1] (whatever steering was
already applied last cycle) is available at tick k, since the controller hasn't computed delta[k]
yet -- it needs this filter's v_y_hat to do so. Both predict() and update() are driven by
delta_prev, never by the current tick's own delta -- see VyKalmanFilter's docstring.

Synthetic measurement noise: sensor.other.imu ran with default (near-zero) noise on the logged
runs, so the logged yaw_rate/a_y are close to clean truth. Gaussian noise is added here, offline
(and equivalently in mpc_mpc_KF.py's live loop), to both channels before they reach the filter --
otherwise R has nothing to be tuned against and the filter collapses to "trust the model, ignore
the measurement" for any R.

Usage:
    cd carla_control/kalman_filter
    python3 kalman_filter.py --save-plot                 # replay every kf_data/*.npz, default Q/R
    python3 kalman_filter.py --tune --save-plot           # search Q/R first, then replay + plot

Or as a library, from anywhere under carla_control/ (mpc_mpc_KF.py does this):
    from kalman_filter.kalman_filter import VyKalmanFilter
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
CARLA_CONTROL_ROOT = os.path.dirname(HERE)   # this file lives in carla_control/kalman_filter/ --
# functions.py/viz_utils.py/etc. are one level up, in carla_control/ itself, so that needs to be on
# sys.path too. Only matters for the offline CLI below (`import carla`/`from viz_utils import ...`);
# a caller that only imports VyKalmanFilter itself never touches this path at all.
sys.path.append(CARLA_CONTROL_ROOT)

# Only needed for the offline CLI's plotting/reporting (viz_utils.plot_kf_vy/print_error_summary,
# which import `carla`) -- VyKalmanFilter itself has no CARLA dependency at all. Same CARLA_ROOT
# resolution every standalone script in this repo does, so `python3 kalman_filter.py` works on its
# own instead of relying on some other script having already fixed up sys.path first.
CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/carla/CARLA_0.9.15"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))


class VyKalmanFilter:
    r"""LPV Kalman filter for x = [v_y, r]^T (r = yaw rate), scheduled on v_x (measured directly,
    not estimated -- only v_y is hidden here).

        x_k = A_d(vx_{k-1}) x_{k-1} + B_d delta_{k-1} + w_{k-1},   w ~ N(0, Q)
        z_k = H(vx_k) x_k + D delta_{k-1} + v_k,                    v ~ N(0, R)

    A/B are the top-left 2x2 block / top 2 rows of LateralMPC's own continuous bicycle-model
    matrices (mpc_mpc.py's LateralMPC._continuous()), dropping the e_y/e_psi path-following rows --
    this filter only estimates vehicle-frame dynamics, it has no notion of a path:

        A(vx) = [[-(Cf+Cr)/(m vx),        (lr Cr - lf Cf)/(m vx) - vx],
                 [(lr Cr - lf Cf)/(Iz vx), -(lf^2 Cf + lr^2 Cr)/(Iz vx)]]
        B = [Cf/m, lf Cf/Iz]^T

    z = [dpsi_meas, ay_meas]^T: dpsi is the gyro's own z-axis reading, so it comes straight off
    state 2 (H row 1 = [0, 1]) -- unlike v_y, which is exactly the thing being estimated and so
    cannot appear as a direct measurement. ay is the accelerometer reading, a_y = v_y_dot + vx*r,
    same derivation as LateralMPC's C matrix's a_y row (add vx*r to the state equation's own first
    row and the -vx/+vx terms cancel, leaving a_y linear in v_y/r/delta):

        H(vx) = [[0, 1],
                 [-(Cf+Cr)/(m vx), (lr Cr - lf Cf)/(m vx)]]
        D = [0, Cf/m]^T

    delta_{k-1} appears in both the transition and the measurement model, not delta_k: at the time
    z_k is read, the controller hasn't computed delta_k yet (it needs this filter's own v_y_hat to
    do so -- see mpc_mpc.py's x0 = [v_y, r, e_y, e_psi] going into LateralMPC.solve()), so the only
    steering input available at tick k is whatever was actually applied over the (k-1 -> k)
    interval. step() below takes a single delta_prev for exactly this reason, not two.

    Forward-Euler discretization, same convention as LateralMPC: A_d = I + dt*A, B_d = dt*B. Re-
    linearized every step since vx changes tick to tick (the "LPV" in the name) -- no fixed-gain
    steady-state simplification here on purpose, vx swings too much over a run (5 -> 15 m/s across
    mpc_mpc.py's own --log-npz data) for one Kalman gain to fit everywhere.
    """

    N_X = 2   # [v_y, r]
    N_Z = 2   # [dpsi_meas, ay_meas]

    def __init__(self, dt, mass, Iz, lf, lr, Cf, Cr, Q, R, vx_floor=1e-3, x0=None, P0=None):
        self.dt = dt
        self.mass, self.Iz, self.lf, self.lr, self.Cf, self.Cr = mass, Iz, lf, lr, Cf, Cr
        # 이제 1/vx 의 0 나눗셈 방어일 뿐이다. 저속 우회 게이트를 떠받치던 안정성
        # 역할은 정확한 ZOH 로 옮겨갔다 (step() 참고).
        self.vx_floor = vx_floor

        self.B_cont = np.array([[Cf / mass], [lf * Cf / Iz]])
        # Bd is not a constant under exact ZOH (it is integral_0^dt expm(A s) ds * B and A depends
        # on vx), so it is formed per call in _discrete(). B_cont is kept for that.
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

    def _discrete(self, vx):
        """(Ad, Bd) by exact zero-order hold, via expm([[A, B], [0, 0]] * dt) = [[Ad, Bd], [0, 1]].

        The augmented form is used rather than Bd = A^-1 (Ad - I) B because A is singular at the
        speeds this has to survive. Replaces forward Euler (Ad = I + dt*A), which mattered more
        here than in the MPC: the covariance recursion squares the transition, so P <- Ad P Ad^T + Q
        amplified P by rho(Ad)^2 per tick -- 68x per tick at vx = 0.5 m/s under Euler. Exact ZOH is
        contractive at every speed.
        """
        n = self.N_X
        aug = np.zeros((n + 1, n + 1))
        aug[:n, :n] = self._continuous_A(vx)
        aug[:n, n:n + 1] = self.B_cont
        M = expm(aug * self.dt)
        return M[:n, :n], M[:n, n:n + 1]

    def predict(self, vx, delta_prev):
        """Advance x_{k-1} -> x_k over one dt, linearized at vx (the scheduling speed for this
        interval) and driven by delta_prev (the steering angle actually applied over it)."""
        Ad, Bd = self._discrete(vx)
        self.x = Ad @ self.x + Bd * delta_prev
        self.P = Ad @ self.P @ Ad.T + self.Q
        return self.x.ravel()

    def update(self, vx, delta_prev, z):
        """Correct the predicted x_k with a new z_k = [dpsi_meas, ay_meas]."""
        H = self._H(vx)
        z = np.asarray(z, dtype=float).reshape(self.N_Z, 1)
        y = z - (H @ self.x + self.D * delta_prev)   # innovation
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.N_X) - K @ H) @ self.P
        return self.x.ravel()

    def step(self, vx, delta_prev, z):
        """predict() then update() for one tick -- the form a per-cycle replay/control loop
        actually calls. Returns the corrected [v_y, r].

        The `if vx < vx_floor: x = [0, z[0]]` bypass this used to open with is gone. It existed
        because the forward-Euler predict step was divergent below ~2.3 m/s, so the only safe thing
        there was to throw the model away and take the measured yaw rate neat. The old docstring
        justified it by the model re-injecting a "phantom" lateral velocity that the low-vx-weakened
        measurement update could not cancel -- which is a description of a divergent predict step,
        i.e. of the forward-Euler problem, not of the physics. Exact ZOH is contractive at every
        speed (rho(Ad) <= 1 down to the scheduling floor), so there is nothing to cancel, and v_y
        keeps being estimated through low-speed stretches instead of being pinned to 0 -- which is
        most of a junction approach.

        vx_floor still applies inside _continuous_A()/_H() as a *scheduling* floor, so the 1/vx
        terms stay finite. Only the hard bypass is removed."""
        self.predict(vx, delta_prev)
        return self.update(vx, delta_prev, z)

    @property
    def v_y(self):
        return float(self.x[0, 0])

    @property
    def r(self):
        return float(self.x[1, 0])


# ----------------------------------------------------------------------------- offline replay + tuning

# mpc_mpc.py --log-npz's own metadata keys (see its main()); everything else in the .npz is a
# per-tick hist series in the exact schema plot_lateral()/print_error_summary() already expect, so
# load_run() below can hand that part straight to viz_utils without reshaping it.
META_KEYS = ("dt", "mass", "iz", "cf", "cr", "lf", "lr")


def load_run(npz_path):
    """One logged mpc_mpc.py run: the vehicle params actually used that run (LateralMPC's/this
    filter's A/H depend on lf/lr/mass/Iz/Cf/Cr, not just whatever args defaults to) plus the raw
    hist dict, untouched -- kept in mpc_mpc.py's own units (degrees for yaw_rate/steer_deg) so it
    can be handed straight to viz_utils.plot_lateral()/print_error_summary() later without having
    to convert back."""
    d = np.load(npz_path)
    meta = {k: float(d[k]) for k in META_KEYS}
    # .tolist(), not the raw ndarray: viz_utils' plot/report helpers (built for mpc_mpc.py's own
    # hist, which is a dict of plain lists appended tick by tick) do truthiness checks like
    # `if not hist["t"]` / `if v_des` that raise "ambiguous truth value" on a numpy array.
    hist = {k: d[k].tolist() for k in d.files if k not in META_KEYS}
    return dict(name=os.path.splitext(os.path.basename(npz_path))[0], hist=hist, **meta)


def inject_noise(hist, gyro_std_rad, accel_std, rng):
    """Additive white Gaussian noise on the gyro/accel channels only, converted to the filter's own
    units (rad/s) here -- v_x is treated as directly measured (wheel speed sensors are far cleaner
    than the IMU) and delta as exactly what was commanded, same as the filter's own D matrix
    assumes."""
    r_true = np.radians(hist["yaw_rate"])
    ay_true = np.asarray(hist["a_y"], dtype=float)
    r_meas = r_true + rng.normal(0.0, gyro_std_rad, size=r_true.shape)
    ay_meas = ay_true + rng.normal(0.0, accel_std, size=ay_true.shape)
    return r_meas, ay_meas


def replay(run, Q_diag, R_diag, gyro_std_rad, accel_std, seed, vx_floor=1e-3):
    """Run the filter tick-by-tick over one logged run -- see the module docstring for why this
    counts as "closed-loop": every step builds on the filter's own previous corrected estimate.
    v_y_hat[0] is the filter's initial guess (unfiltered, no measurement processed yet), not a real
    estimate -- callers that care about accuracy should skip a burn-in window (see score()).

    Also returns the noisy r_meas/ay_meas (rad/s, m/s^2) this call fed the filter -- score()/tune()
    ignore them, but main()'s plotting wants them to draw clean-vs-noisy sensor comparisons."""
    hist = run["hist"]
    rng = np.random.default_rng(seed)
    r_meas, ay_meas = inject_noise(hist, gyro_std_rad, accel_std, rng)
    delta_rad = np.radians(hist["steer_deg"])
    v_x = hist["v_x"]

    n = len(hist["t"])
    v_y_hat = np.zeros(n)
    kf = VyKalmanFilter(dt=run["dt"], mass=run["mass"], Iz=run["iz"], lf=run["lf"], lr=run["lr"],
                        Cf=run["cf"], Cr=run["cr"], Q=np.diag(Q_diag), R=np.diag(R_diag),
                        vx_floor=vx_floor, x0=[0.0, r_meas[0]], P0=np.eye(2))
    for k in range(1, n):
        kf.step(v_x[k], delta_rad[k - 1], [r_meas[k], ay_meas[k]])
        v_y_hat[k] = kf.v_y
    return v_y_hat, r_meas, ay_meas


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def score(run, Q_diag, R_diag, gyro_std_rad, accel_std, seeds, vx_floor=1e-3):
    """Mean v_y RMSE over several noise-seed draws (not just one), so tuning doesn't just fit Q/R
    to one particular random noise realization -- burn-in is the first 1s of the run, dropped so
    the filter's startup transient (x0 = [0, r_meas[0]] is a guess) doesn't dominate a run that's
    otherwise only 15-60s long."""
    burn_in = int(round(1.0 / run["dt"]))
    v_y_true = run["hist"]["v_y"][burn_in:]
    errs = []
    for seed in seeds:
        v_y_hat, _, _ = replay(run, Q_diag, R_diag, gyro_std_rad, accel_std, seed, vx_floor)
        errs.append(rmse(v_y_hat[burn_in:], v_y_true))
    return float(np.mean(errs))


# log10(Q diag) search box for tune() -- keeps Nelder-Mead's simplex from wandering to a
# 1e-165-style degenerate corner an unbounded log-space search finds (float underflow, and Q->0
# just means "trust the model completely, ignore the measurement"): -8 is already far below any
# variance this state needs, +3 is far above it.
LOG_BOUNDS = [(-8.0, 3.0)] * 2


def tune(runs, gyro_std_rad, accel_std, seeds, R_diag, x0_q, vx_floor=1e-3):
    """Nelder-Mead over log10([q_vy, q_r]) ONLY -- R is deliberately NOT searched here, and is
    fixed to R_diag (by default the actual injected noise variance, gyro_std_rad**2/accel_std**2 --
    see main()). That's not a simplification, it's the textbook division of labor: R describes the
    sensor and is knowable independent of any particular drive (a spec sheet, or here, literally
    the noise this script injected), while Q describes how much the *model* is trusted and has no
    other source of truth than tuning it against data like this. Letting R float too let an earlier
    version of this search find q->0, r->inf ("ignore every measurement, trust the bicycle model
    outright") as a real RMSE-minimizing optimum -- unsurprising in a sim this clean (the bicycle
    model has near-zero mismatch against CARLA's own linear-ish tire response at these speeds/
    angles), but a filter that has learned to ignore its own sensors isn't demonstrating anything
    about Kalman filtering, and would be the wrong lesson to carry into a real vehicle where model
    mismatch is not this small. Bounded to LOG_BOUNDS, see its comment; minimizes the mean per-run
    RMSE (not the sum, so a longer 5 m/s run doesn't dominate over the shorter 15 m/s one)."""
    def objective(log_q):
        q_vy, q_r = 10.0 ** log_q
        return float(np.mean([score(run, [q_vy, q_r], R_diag, gyro_std_rad, accel_std, seeds, vx_floor)
                              for run in runs]))

    x0_log = np.clip(np.log10(x0_q), [lo for lo, _ in LOG_BOUNDS], [hi for _, hi in LOG_BOUNDS])
    result = minimize(objective, x0_log, method="Nelder-Mead", bounds=LOG_BOUNDS,
                      options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 300, "disp": False})
    return (10.0 ** result.x), result


def main():
    import carla   # noqa: F401 -- resolves sys.path for viz_utils' own `import carla`, see the header
    from viz_utils import plot_kf_run, print_error_summary

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", nargs="+", default=None,
                        help="logged run(s) from mpc_mpc.py --log-npz; default: every .npz under "
                             "--log-dir")
    parser.add_argument("--log-dir", default=os.path.join(HERE, "kf_data"),
                        help="where --npz globs when not given explicitly")
    parser.add_argument("--speed", nargs="+", type=int, default=None,
                        help="only replay the run(s) whose logged initial speed (m/s, rounded) "
                             "matches one of these, e.g. --speed 5 15 -- default: every loaded run. "
                             "Running all of them at once with --save-plot pops up a figure per run, "
                             "which gets noisy fast; this narrows it down.")

    noise = parser.add_argument_group("synthetic measurement noise")
    noise.add_argument("--gyro-std", type=float, default=0,
                       help="synthetic gyro (dpsi) noise stddev, deg/s")
    noise.add_argument("--accel-std", type=float, default=0,
                       help="synthetic accelerometer (a_y) noise stddev, m/s^2")
    noise.add_argument("--seed", type=int, default=0, help="noise seed used for the reported/plotted replay")
    noise.add_argument("--n-seeds", type=int, default=3,
                       help="number of noise-seed draws averaged over during --tune scoring")

    kf_args = parser.add_argument_group("Kalman filter Q/R (diagonal only)")
    kf_args.add_argument("--q-vy", type=float, default=1e-8, help="process noise variance on v_y, (m/s)^2/step")
    kf_args.add_argument("--q-r", type=float, default=1e-8, help="process noise variance on r, (rad/s)^2/step")
    kf_args.add_argument("--r-dpsi", type=float, default=None,
                         help="measurement noise variance on dpsi, (rad/s)^2 -- default: matches "
                              "--gyro-std^2 (the noise actually injected)")
    kf_args.add_argument("--r-ay", type=float, default=None,
                         help="measurement noise variance on a_y, (m/s^2)^2 -- default: matches "
                              "--accel-std^2 (the noise actually injected)")
    kf_args.add_argument("--tune", action="store_true",
                         help="search Q ONLY (Nelder-Mead, log-space, bounded) to minimize mean v_y "
                              "RMSE across all --npz runs before replaying/plotting with the result "
                              "-- R stays fixed at --r-dpsi/--r-ay (default: the actual injected "
                              "noise variance), see tune()'s docstring for why R isn't searched")
    kf_args.add_argument("--vx-floor", type=float, default=0.5, help="same floor LateralMPC uses on vx")

    parser.add_argument("--plot-dir", default=os.path.join(HERE, "kf_plots"))
    parser.add_argument("--save-plot", action="store_true",
                        help="draw+save one 3-panel figure per run (v_y estimate + dpsi/a_y clean-"
                             "vs-noisy, viz_utils.plot_kf_run) and show all of them once every run "
                             "is done -- see --speed to cut down how many pop up at once")
    args = parser.parse_args()

    npz_paths = args.npz or sorted(glob.glob(os.path.join(args.log_dir, "*.npz")))
    if not npz_paths:
        parser.error(f"no --npz given and no .npz files found under {args.log_dir} "
                     f"(run mpc_mpc.py --log-npz first)")
    runs = [load_run(p) for p in npz_paths]
    print(f"Loaded {len(runs)} run(s): " + ", ".join(r["name"] for r in runs))

    if args.speed is not None:
        runs = [r for r in runs if round(r["hist"]["v_des"][0]) in args.speed]
        if not runs:
            parser.error(f"no loaded run's initial speed matches --speed {args.speed}")
        print(f"Filtered to --speed {args.speed}: " + ", ".join(r["name"] for r in runs))

    gyro_std_rad = math.radians(args.gyro_std)
    r_dpsi_default = gyro_std_rad ** 2
    r_ay_default = args.accel_std ** 2

    # R is fixed to the actual injected noise variance by default, never searched -- see tune()'s
    # docstring for why. --r-dpsi/--r-ay still let it be overridden (e.g. to explore a filter that
    # deliberately under/over-trusts a channel relative to its true noise), --tune or not.
    r_dpsi = args.r_dpsi if args.r_dpsi is not None else r_dpsi_default
    r_ay = args.r_ay if args.r_ay is not None else r_ay_default

    if args.tune:
        seeds = list(range(args.n_seeds))
        (q_vy, q_r), result = tune(runs, gyro_std_rad, args.accel_std, seeds, [r_dpsi, r_ay],
                                   [args.q_vy, args.q_r], args.vx_floor)
        print(f"\nTuned (Nelder-Mead, {result.nit} iters, mean RMSE={result.fun:.4f} m/s):")
        print(f"  Q = diag({q_vy:.3e}, {q_r:.3e})   R = diag({r_dpsi:.3e}, {r_ay:.3e}) (fixed)")
    else:
        q_vy, q_r = args.q_vy, args.q_r

    # One figure per run (not the full plot_lateral dashboard -- all that matters here is whether
    # v_y_hat tracks v_y, and whether the sensor channels feeding it look right, see the module
    # docstring): v_y estimate vs. ground truth stacked over the two raw measurement channels the
    # filter actually consumed -- clean (hist["yaw_rate"]/["a_y"]) vs. the noisy dpsi/ay replay() fed
    # it. plot_kf_run() merges what used to be 3 separate figures into these 3 stacked panels, so
    # --speed (above) is the knob for "too many windows at once", not the panel count. Kept open
    # rather than closed so they can all be shown at once after the loop.
    figs = []
    for run in runs:
        v_y_hat, r_meas, ay_meas = replay(run, [q_vy, q_r], [r_dpsi, r_ay], gyro_std_rad,
                                          args.accel_std, args.seed, args.vx_floor)
        hist = dict(run["hist"])
        hist["v_y_hat"] = v_y_hat.tolist()
        hist["dpsi_noisy"] = np.degrees(r_meas).tolist()   # deg/s, same unit as hist["yaw_rate"]
        hist["ay_noisy"] = ay_meas.tolist()                # m/s^2, same unit as hist["a_y"]

        target_speed = int(hist["v_des"][0])
        print(f"\n--- {run['name']} ---")
        print_error_summary(hist, target_speed)

        if args.save_plot:
            fig = plot_kf_run(hist, title=f"{target_speed} m/s driving -- Kalman filter report")
            os.makedirs(args.plot_dir, exist_ok=True)
            out_path = os.path.join(args.plot_dir, f"{run['name']}_kf.png")
            fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
            print(f"  -> {out_path}")
            figs.append(fig)

    if figs:
        import matplotlib.pyplot as plt
        plt.show()   # blocks until every figure window is closed


if __name__ == "__main__":
    main()
