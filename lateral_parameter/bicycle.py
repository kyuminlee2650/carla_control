r"""The 2-DOF bicycle model, the measurements that feed it, and the checks that test it.

Shared by estimate_cornering_stiffness.py and estimate_yaw_inertia.py. Everything here is specific
to lateral parameter identification, which is why it lives beside those two scripts rather than in
the repo-wide functions.py -- nothing else in the codebase uses a slip angle.

Model (CG-relative slip angles, small-angle linear tires):

    beta    = v_y / v_x
    alpha_f = delta - beta - lf*r/v_x           Fyf = Cf * alpha_f
    alpha_r =       - beta + lr*r/v_x           Fyr = Cr * alpha_r

    m*(v_y_dot + v_x*r) = Fyf + Fyr             (lateral force balance)
    Iz*r_dot            = lf*Fyf - lr*Fyr       (yaw moment balance)

Note what the second equation does at steady state: r_dot = 0 kills Iz entirely. That single fact
sets up the whole identification strategy -- Cf/Cr are identifiable from steady cornering without
knowing Iz, and Iz is only identifiable from a transient, using a Cf/Cr already in hand.
"""

import math
import os
import sys

import numpy as np

# functions.py lives one level up and owns CARLA_ROOT resolution for the whole repo; importing it
# is what puts the simulator's PythonAPI on sys.path (see functions.resolve_carla_root).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import CARLA_ROOT  # noqa: F401  (imported for its sys.path side effect)

import carla

G = 9.81


# --------------------------------------------------------------------------------------------
# test sites
# --------------------------------------------------------------------------------------------

# Patches of tarmac big enough to drive a full circle on, found by scanning every map's drivable
# lanes for centres where a whole circle stays inside a driving lane. Both are dead flat --
# elevation range 0.000 m and |roll| 0.000 deg around every feasible radius -- which matters
# because an accelerometer measures specific force: on a banked surface a_y would pick up
# g*sin(bank), and 1 deg of bank is 0.17 m/s^2, over 20% of a_y at the low end of the sweep.
#
# The two together cover R = 8-11.5 m and 18.5-24.5 m. That gap is unavoidable (no map has a
# wider single patch) but the two bands are what decouple lateral acceleration from speed: the
# same a_y is reachable at several speeds, and several a_y at one speed.
#
# The lower radius limit is set by the model, not the tarmac. Small radii need large steer angles
# and the bicycle model's small-angle assumption breaks down: at R = 5 m, L/R is 32.8 deg while
# atan(L/R) is 29.8 deg, a 9% disagreement that lands straight in alpha_f. R >= 8 m keeps it
# under 4%.
# `radius_range` is the measured drivable band. `usable_range` is what a sweep should actually
# ask for, and it is deliberately narrower at BOTH ends:
#
#   - the low end because of the model, not the tarmac (see above), and on Town03 because the
#     inner edge of the band is the roundabout's kerbed central island -- there is nothing
#     forgiving about overshooting inwards there;
#   - the high end because understeer makes the achieved radius about 9% larger than the one
#     asked for, so requesting the outer edge lands the car past it.
CIRCLE_SITES = {
    "town06": {                       # a large multi-lane intersection
        "map": "Town06",
        "centre": (662.2, 141.2),
        "z": 0.0,
        "radius_range": (4.0, 11.5),
        "usable_range": (8.0, 10.4),
    },
    "town03": {                       # the central roundabout's annulus (lanes -4 and -5)
        "map": "Town03",
        "centre": (-0.4996, 0.3835),
        "z": 0.0,
        "radius_range": (18.5, 24.5),
        "usable_range": (19.5, 22.2),
    },
}


def circle_path(centre, radius, start_angle=math.pi / 2.0, direction=-1.0, step=2.0, laps=2.2):
    """Points and headings around a circle -- the reference line for a constant-radius test.

    Generated rather than taken from map waypoints, so the radius is exactly constant (which is
    the condition a steady-cornering test assumes) and any radius inside the site's range can be
    asked for, not just the ones a lane happens to sit on.
    """
    cx, cy = centre
    n_points = max(8, int(laps * 2.0 * math.pi * radius / step))
    path_x, path_y, path_yaw = [], [], []
    for k in range(n_points):
        theta = start_angle + direction * (k * step / radius)
        path_x.append(cx + radius * math.cos(theta))
        path_y.append(cy + radius * math.sin(theta))
        path_yaw.append(theta + direction * math.pi / 2.0)   # tangent, pointing along travel
    return path_x, path_y, path_yaw


def steering_curve_scale(physics, speed_ms):
    """The factor CARLA applies to a steer command at this speed.

    VehiclePhysicsControl.steering_curve is a lookup with x in km/h: 0 -> 1.0, 20 -> 0.9,
    60 -> 0.8, 120 -> 0.7 on the stock vehicles. The same command is therefore a smaller wheel
    angle the faster you go -- measured on the mkz_2020, 0.93 at 4 m/s down to 0.87 at 9 m/s.
    """
    xs = [p.x for p in physics.steering_curve]
    ys = [p.y for p in physics.steering_curve]
    return float(np.interp(speed_ms * 3.6, xs, ys))


# --------------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------------

def front_steer_angle(vehicle):
    """Measured physical front-wheel angle (rad), averaged over the two front wheels.

    Use this rather than `steer_cmd * max_steer_angle` anywhere the wheel angle feeds a fit. That
    product is not the wheel angle, for two separate reasons:

      - CARLA scales the commanded steer by VehiclePhysicsControl.steering_curve, a speed-dependent
        factor, so the same command is a smaller angle the faster you go. That is a speed-dependent
        bias -- exactly the shape of error that shows up as a stiffness drifting with speed.
      - The steering actuator takes several ticks to reach the command, so the two differ whenever
        the command is moving, which is precisely when a lateral fit is reading alpha_f.

    get_wheel_steer_angle() reports the angle the physics engine actually used, so both effects
    drop out and no actuator-lag time constant has to be guessed. Measured on a simulated bicycle
    with a 0.1 s actuator lag, feeding the *commanded* angle instead put Iz out by 2.7x.

    The two front wheels differ by the Ackermann geometry; the bicycle model's single virtual wheel
    sits between them, so their mean is the right reduction.
    """
    fl = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
    fr = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
    return math.radians(0.5 * (fl + fr))


def bicycle_steer_angle(vehicle):
    """The single virtual wheel angle equivalent to the two real front wheels (rad).

    Ackermann steers the inner wheel harder than the outer, so neither is the bicycle model's
    delta and their arithmetic mean is only an approximation. The geometry that actually holds is
    on the cotangents -- cot(delta_outer) - cot(delta_inner) = track/L -- so the equivalent single
    wheel is the one whose cotangent is their average:

        delta = arccot( (cot(delta_L) + cot(delta_R)) / 2 )

    Measured on the mkz_2020 at R = 10 m this differs from the arithmetic mean by 0.07 deg out of
    14.9 (0.5%). Small, but alpha_f is a ~1-2 deg residual of an ~15 deg steer, so a 0.5% error in
    delta is several percent of alpha_f, and it costs nothing to be right.

    Falls back to the mean when either wheel is near zero, where the cotangent blows up.
    """
    fl = math.radians(vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
    fr = math.radians(vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel))
    if abs(fl) < 1e-3 or abs(fr) < 1e-3 or fl * fr <= 0.0:
        return 0.5 * (fl + fr)
    cot_mean = 0.5 * (1.0 / math.tan(fl) + 1.0 / math.tan(fr))
    return math.atan(1.0 / cot_mean)


# There is deliberately no roll-corrected accelerometer helper here. Rotating the measured
# specific force by the roll angle (a*cos(phi) + g*sin(phi)) removes the gravity that leaks into
# the body's y axis, and that part needs only the angle -- but it is only half of the story. A
# sensor sitting a distance h off the roll axis also reads phi_ddot*h and phi_dot^2*h, and CARLA
# exposes no roll-axis location to get h from. Those terms vanish when the roll angle is steady,
# which is why the correction looked exact on a settled circle, and reappear the moment the body
# starts oscillating in roll -- exactly the condition under which the correction was being relied
# on. Everything downstream now uses v_x*psi_dot and a differentiated v_y instead, neither of
# which involves the body's attitude at all.


def steer_convention_report(delta_measured, delta_commanded):
    """Check the measured wheel angle against the commanded one over a whole run.

    front_steer_angle() is only usable if CARLA reports the angle in the same sign convention the
    bicycle model uses. Rather than assume, regress measured on commanded through the origin: the
    slope should land near but below 1 -- below because of the steering curve, near because the
    command is still what drives the wheel. A negative slope means the convention is flipped and
    every alpha_f in the run has the wrong sign.

    Returns (slope, note), note being None when nothing looks wrong.
    """
    m = np.asarray(delta_measured, dtype=float)
    c = np.asarray(delta_commanded, dtype=float)
    denom = float(np.sum(c * c))
    if denom <= 0.0:
        return float("nan"), "commanded steer was identically zero -- cannot check the convention"
    slope = float(np.sum(m * c) / denom)

    if slope < 0.0:
        return slope, (f"measured/commanded steer slope is {slope:+.3f} -- CARLA reports the wheel "
                       f"angle with the OPPOSITE sign to the command. Every alpha_f in this run is "
                       f"wrong; negate front_steer_angle() before trusting anything below.")
    if not 0.5 <= slope <= 1.2:
        return slope, (f"measured/commanded steer slope is {slope:.3f}, outside the ~0.7-1.0 the "
                       f"steering curve alone explains -- check max_steer_angle and steering_curve "
                       f"for this blueprint.")
    return slope, None


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------

def static_axle_loads(mass, lf, lr):
    """Static front/rear axle vertical loads (N): Fzf = m*g*lr/L, Fzr = m*g*lf/L."""
    L = lf + lr
    return mass * G * lr / L, mass * G * lf / L


def understeer_gradient(Cf, Cr, mass, lf, lr):
    """K = (m/L)*(lr/Cf - lf/Cr), rad per m/s^2. Steady-state steer is delta = L/R + K*a_y."""
    L = lf + lr
    return (mass / L) * (lr / Cf - lf / Cr)


def yaw_mode(Cf, Cr, mass, Iz, lf, lr, v):
    """Natural frequency (rad/s) and damping ratio of the yaw/sideslip mode.

        omega_n^2      = [Cf*Cr*L^2 - m*v^2*(lf*Cf - lr*Cr)] / (m*Iz*v^2)
        2*zeta*omega_n = [Iz*(Cf+Cr) + m*(lf^2*Cf + lr^2*Cr)] / (m*Iz*v)

    This is the independent test of Iz. The steady-state yaw gain fixes Cf and Cr but contains no
    Iz at all, while omega_n scales as 1/sqrt(Iz) -- so a measured step response whose overshoot
    contradicts the zeta predicted here is evidence about Iz specifically, not about the tires.
    """
    L = lf + lr
    wn_sq = (Cf * Cr * L ** 2 - mass * v ** 2 * (lf * Cf - lr * Cr)) / (mass * Iz * v ** 2)
    if wn_sq <= 0:
        return float("nan"), float("nan")   # past the critical speed: unstable, no oscillatory mode
    wn = math.sqrt(wn_sq)
    zeta = (Iz * (Cf + Cr) + mass * (lf ** 2 * Cf + lr ** 2 * Cr)) / (mass * Iz * v) / (2 * wn)
    return wn, zeta


def steady_yaw_rate(delta, v, Cf, Cr, mass, lf, lr):
    """Steady-state yaw rate for a given steer and speed: r = v*delta / (L + K*v^2). No Iz in it."""
    L = lf + lr
    return v * delta / (L + understeer_gradient(Cf, Cr, mass, lf, lr) * v ** 2)


def simulate_yaw(delta, v_x, dt, Cf, Cr, mass, Iz, lf, lr, v_y0=0.0, r0=0.0):
    """Integrate the model on a measured delta(t)/v_x(t) and return the predicted r(t)."""
    v_y, r = v_y0, r0
    out = []
    for d, v in zip(delta, v_x):
        out.append(r)
        if v < 0.5:                       # slip angles divide by v_x; undefined near standstill
            continue
        alpha_f = d - (v_y + lf * r) / v
        alpha_r = -(v_y - lr * r) / v
        Fyf, Fyr = Cf * alpha_f, Cr * alpha_r
        v_y += dt * ((Fyf + Fyr) / mass - v * r)
        r += dt * ((lf * Fyf - lr * Fyr) / Iz)
    return np.array(out)


# --------------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------------

def bracket(y, x):
    """Both one-sided least-squares slopes of y = k*x through the origin, and their geometric mean.

    Neither direction is unbiased when both variables carry noise: regressing y on x is biased low
    (noise inflates the denominator -- errors-in-variables attenuation), regressing x on y and
    inverting is biased high, and the truth lies between.

    Read the gap for what it is: it bounds the *random* part only. It is not a confidence interval
    and does not cover systematic error -- on simulated data with a known Iz the truth fell inside
    it in well under half of runs, because at coarse dt the dominant error is resolution bias,
    which both directions share. A narrow bracket means "noise is not the problem", not "correct".
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    denom = float(np.sum(x * x))
    cross = float(np.sum(y * x))
    forward = cross / denom if denom > 0 else float("nan")
    reverse = float(np.sum(y * y)) / cross if abs(cross) > 0 else float("nan")
    geometric = math.sqrt(forward * reverse) if forward > 0 and reverse > 0 else float("nan")
    return forward, reverse, geometric


def fit_stiffness(alpha, Fy):
    """Cf or Cr by least squares through the origin: min_C sum((C*alpha - Fy)^2)."""
    alpha, Fy = np.asarray(alpha), np.asarray(Fy)
    return float(np.sum(alpha * Fy) / np.sum(alpha * alpha))


def moment_impulse(M, dt):
    """Cumulative trapezoid of M -- angular impulse delivered since the window started."""
    M = np.asarray(M, dtype=float)
    if len(M) < 2:
        return np.zeros_like(M)
    return np.concatenate(([0.0], np.cumsum(0.5 * (M[1:] + M[:-1])) * dt))


def centered_integral_pair(M, r, dt):
    """(impulse, yaw rate) for the integral Iz fit, each centred on its own mean.

    The relation is integral(M dt) = Iz*(r - r0), and the tempting move is to subtract the first
    sample as r0. That makes one noisy sample the reference for the whole trace, and a constant
    offset in a through-origin fit is a slope error: at dt=0.05 with 0.02 rad/s of gyro noise it
    cost +46% on a known Iz. Centring both series is algebraically a free intercept and absorbs
    that; the same case then lands within 2%. Centring per trial also makes trials poolable, which
    subtracting r0 does not, since each trial's integral starts from its own arbitrary zero.
    """
    S = moment_impulse(M, dt)
    r = np.asarray(r, dtype=float)
    return S - S.mean(), r - r.mean()


def fit_inertia_integral(M, r, dt):
    """Iz from the integrated moment balance: integral(M dt) = Iz*(r - r0). The preferred form.

    Differentiating r and regressing M on r_dot is the obvious approach and the wrong one.
    Differentiation amplifies noise and forces a smoothing filter that cannot win: wide enough to
    control noise, it clips the peak of r_dot, which inflates Iz directly. Integration averages
    noise down and needs no filter. Against a simulated bicycle with a known Iz (1200-3400
    kg*m^2, dt 0.01-0.05, gyro noise to 0.01 rad/s): derivative form +5% to +26%, integral form
    -5% to +4%.

    Returns (forward, reverse, geometric_mean).
    """
    if len(M) < 3:
        return (float("nan"),) * 3
    return bracket(*centered_integral_pair(M, r, dt))


def fit_inertia_derivative(M, r_dot):
    """Iz from the pointwise balance M = Iz*r_dot. Kept only as an independent cross-check.

    Biased high by however much the derivative filter clipped r_dot's peak, so a disagreement with
    fit_inertia_integral() is a statement about the smoothing, not about Iz.
    """
    return bracket(np.asarray(M, dtype=float), np.asarray(r_dot, dtype=float))


def zero_phase_derivative(signal, dt, window_s=None, rise_s=None, polyorder=2):
    """Differentiate a logged signal without introducing lag.

    Savitzky-Golay fits a local polynomial over a centred window and differentiates it
    analytically, so unlike any causal filter it adds no delay between the derivative and the
    samples it came from. That matters wherever the derivative is compared against something
    measured at the same instants -- a causal filter would manufacture a discrepancy out of its
    own phase shift.

    Window width is the one real choice. It must be short against whatever transient is being
    resolved: differentiating a yaw rate over a 0.24 s rise, a 0.25 s window returned 2556 for a
    true Iz of 1800 where a 0.05 s window returned 1822. Pass `rise_s` to size it from the
    transient (a quarter of it), or `window_s` to set it outright.
    """
    from scipy.signal import savgol_filter

    signal = np.asarray(signal, dtype=float)
    if window_s is None:
        window_s = max(5 * dt, 0.25 * rise_s) if rise_s else 5 * dt
    n = max(polyorder + 2, int(round(window_s / dt)))
    if n % 2 == 0:
        n += 1
    if n >= len(signal):
        return np.gradient(signal, dt)   # too short to filter; still zero-phase, just noisier
    return savgol_filter(signal, n, polyorder, deriv=1, delta=dt, mode="interp")


def transient_span(r, step_idx, dt, settle_frac=0.98, tail_s=0.5, margin_s=0.15):
    """Index range [start, end) covering the rise, selected on r rather than on r_dot.

    Selecting by |r_dot| would condition the sample set on the very quantity whose noise biases the
    fit, stacking a selection bias on the attenuation one. r is the integral of the regressor and
    far cleaner, so thresholding on it is safe.
    """
    r = np.asarray(r, dtype=float)
    tail = max(1, int(tail_s / dt))
    r_ss = float(np.mean(r[-tail:]))
    if abs(r_ss) < 1e-6:
        return step_idx, len(r)

    end = len(r)
    for k in range(step_idx, len(r)):
        if abs(r[k]) >= settle_frac * abs(r_ss):
            end = min(len(r), k + int(margin_s / dt))
            break
    return step_idx, max(end, step_idx + 3)


# --------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------

def vaf(measured, predicted):
    """Variance accounted for (%). 100 is perfect; negative is worse than predicting the mean."""
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    denom = np.var(measured)
    if denom <= 0:
        return float("nan")
    return float(100.0 * (1.0 - np.var(measured - predicted) / denom))


def nrmse(measured, predicted):
    """RMS error as a percentage of the measured signal's peak-to-peak range."""
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    rng = measured.max() - measured.min()
    if rng <= 0:
        return float("nan")
    return float(100.0 * math.sqrt(np.mean((measured - predicted) ** 2)) / rng)


def report(Cf, Cr, mass, lf, lr, Iz=None, raw_transients=None, speeds=(4, 6, 8, 10, 15, 20)):
    """Print the checks the fits were *not* optimised for.

    Reporting how well a least-squares fit matches the data it was fitted to proves very little --
    it is guaranteed to look good by the criterion it minimised. Each check below is something the
    fit had no opportunity to tune:

      1. Load consistency. One tire compound on all four corners means stiffness should scale with
         the load each axle carries, so Cf/Cr ought to track Fzf/Fzr. Never fitted, so a
         disagreement is real evidence about lf/lr or about delta.
      2. Understeer gradient. A single number for the steady-state balance, with well-known ranges
         for real cars (roughly 2-4 deg/g).
      3. Yaw mode. omega_n and zeta depend on Iz; the steady-state gain does not. That asymmetry
         is what makes them a test of Iz rather than a restatement of the fit.
      4. Simulation against logged transients, scored by VAF -- and a genuinely held-out test if
         the transients come from speeds or steer angles outside the identification set.
    """
    L = lf + lr
    print(f"\n{'='*78}\nVALIDATION\n{'='*78}")
    print(f"m={mass:.0f} kg  L={L:.3f} m  lf={lf:.3f}  lr={lr:.3f}")
    print(f"Cf={Cf:,.0f} N/rad  Cr={Cr:,.0f} N/rad" + (f"  Iz={Iz:,.0f} kg*m^2" if Iz else ""))

    print("\n--- 1. axle load consistency ---")
    Fzf, Fzr = static_axle_loads(mass, lf, lr)
    load_ratio, stiff_ratio = Fzf / Fzr, Cf / Cr
    print(f"Fzf={Fzf:,.0f} N  Fzr={Fzr:,.0f} N   Fzf/Fzr={load_ratio:.2f}   "
          f"({100*lr/L:.0f}% of the weight on the front axle)")
    print(f"Cf/Fzf={Cf/Fzf:.2f} /rad   Cr/Fzr={Cr/Fzr:.2f} /rad   Cf/Cr={stiff_ratio:.2f}")
    if (stiff_ratio - 1.0) * (load_ratio - 1.0) < 0:
        print("  FAIL: the heavier axle fits softer. With one compound on all four corners that "
              "cannot happen -- suspect lf/lr or the delta feeding alpha_f. Compare "
              "wheels[i].lat_stiff_value front vs rear.")
    elif not 0.6 <= stiff_ratio / load_ratio <= 1.6:
        print(f"  MARGINAL: Cf/Cr is {stiff_ratio/load_ratio:.2f}x the load ratio. Load saturation "
              f"explains some of that; beyond ~1.6 it usually does not.")
    else:
        print("  OK: the stiffness split matches the load split.")

    print("\n--- 2. steady-state balance ---")
    K = understeer_gradient(Cf, Cr, mass, lf, lr)
    print(f"understeer gradient K = {K:.5f} rad/(m/s^2) = {math.degrees(K)*G:.2f} deg/g")
    if K > 0:
        print(f"characteristic speed = {math.sqrt(L/K):.1f} m/s ({math.sqrt(L/K)*3.6:.0f} km/h)"
              f"   [understeering]")
    else:
        print(f"critical speed = {math.sqrt(-L/K):.1f} m/s   [OVERSTEERING -- unstable above it]")
    if math.degrees(K) * G > 5.0:
        print(f"  NOTE: {math.degrees(K)*G:.1f} deg/g is high for a passenger car (typical 2-4), "
              f"consistent with Cf being low relative to Cr.")

    if Iz:
        print("\n--- 3. yaw mode (this is what actually tests Iz) ---")
        print(f"{'v':>6} {'omega_n':>9} {'f_n':>7} {'zeta':>7}  prediction")
        for v in speeds:
            wn, zeta = yaw_mode(Cf, Cr, mass, Iz, lf, lr, float(v))
            if wn != wn:
                print(f"{v:6.1f} {'--':>9} {'--':>7} {'--':>7}  unstable at this speed")
            elif zeta >= 1.0:
                print(f"{v:6.1f} {wn:9.2f} {wn/(2*math.pi):7.2f} {zeta:7.2f}  no overshoot in r(t)")
            else:
                os_pct = 100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))
                print(f"{v:6.1f} {wn:9.2f} {wn/(2*math.pi):7.2f} {zeta:7.2f}  ~{os_pct:.0f}% overshoot")

    if Iz and raw_transients:
        print("\n--- 4. simulation vs logged transients ---")
        print(f"{'v':>6} {'VAF':>8} {'NRMSE':>8}  {'peak r meas':>12} {'peak r sim':>11}")
        vafs = []
        for trial in raw_transients:
            dt = trial["dt"]
            s = trial["fit_start"]
            r_meas = np.array(trial["r"])
            r_sim = simulate_yaw(np.array(trial["delta"])[s:], np.array(trial["v_x"])[s:], dt,
                                Cf, Cr, mass, Iz, lf, lr, v_y0=trial["v_y"][s], r0=r_meas[s])
            v_here = vaf(r_meas[s:], r_sim)
            vafs.append(v_here)
            print(f"{trial['target_speed']:6.1f} {v_here:7.1f}% {nrmse(r_meas[s:], r_sim):7.1f}% "
                  f"{math.degrees(max(r_meas[s:], key=abs)):11.2f}  "
                  f"{math.degrees(max(r_sim, key=abs)):10.2f}")
        mean_vaf = float(np.mean(vafs))
        print(f"mean VAF = {mean_vaf:.1f}%")
        print("  OK: the model reproduces the recorded yaw response." if mean_vaf > 90 else
              f"  The model explains only {mean_vaf:.0f}% of the yaw-rate variance.")
