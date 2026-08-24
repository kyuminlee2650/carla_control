r"""Cf/Cr from a logged cornering-stiffness sweep -- pure post-processing, no CARLA connection, no
steady-state judgment of its own. Reads the raw per-tick logs collect_cornering_data.py drove,
judged steady, and saved (see that script's docstring -- it decides which trials settled and trims
each one down to exactly its steady window; a trial that never settled is not in the file at all).
Every sample this script sees IS steady-state data already, so it only fits, pools, filters, and
reports -- it never differentiates a signal or asks "has this settled".

Iz is NOT used here. bicycle.axle_forces_general() (the general force+moment balance using the
true Iz) and bicycle.steady_axle_forces() (the r_dot=0 special case used below) were run side by
side on this pipeline's own data and agreed to within 1% every time -- collect_cornering_data.py's
steady-window gate already constrains the data closely enough to true steady state (r_dot really
is ~0) that the more complete formula bought nothing. axle_forces_general() is still in bicycle.py
if a future dataset (looser steady-state thresholds, different maneuver) needs it back.

Nothing here is selected by SPEED, or by target_speed at all. Two trials at the same speed but
different steer angles can disagree substantially (a small-steer trial's alpha is small relative
to whatever noise the fit carries, which biases its own Cf low -- bracket()'s own docstring calls
this errors-in-variables attenuation), so a per-speed statistic averages that bias into a
"representative" number instead of removing it. Every trial's samples go into one pool regardless
of speed, and the ONLY data-cleaning applied is filter_by_quadrant(): a sample whose alpha and Fy
have opposite sign (quadrant 2 or 4 of the alpha-Fy plane) cannot be a genuine C*alpha point for a
positive C, so it is dropped. No alpha-magnitude bounds, no per-speed std%/range gates, no
residual-based outlier rejection -- this one physical sanity check is deliberately the whole filter
(see filter_by_quadrant()'s own docstring for why simpler is the point, not a compromise).

Pipeline, run end to end by main() on every invocation:

  1. load_log() reads --log-file (collect_cornering_data.py's --out) -- every trial in it already
     IS its own steady window.
  2. fit_cornering_stiffness() computed on every trial's full (already-steady) log.
  3. print_cornering_stiffness_speed_table() reports every trial grouped by speed -- purely
     informational; nothing here is gated by it.
  4. pool_trials() concatenates every trial's (alpha, Fy) points into one front-axle and one
     rear-axle pool, applies filter_by_quadrant(), and fits Cf/Cr (bracket()) on the survivors.
  5. plot_cornering_stiffness_quadrant() draws every logged sample -- kept ones in blue, points
     the quadrant filter dropped in red -- so exactly what was and wasn't trusted is visible on
     the plot itself, with no speed color-coding and no per-speed shading.

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python lateral_parameter/collect_cornering_data.py                        # drive + log (once)
    .venv/bin/python lateral_parameter/estimate_cornering_stiffness.py --save-plot      # fit (many times)

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe lateral_parameter\collect_cornering_data.py
    .venv\Scripts\python.exe lateral_parameter\estimate_cornering_stiffness.py --save-plot
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))

# functions.py resolves CARLA_ROOT and puts the simulator's PythonAPI on sys.path as a module-level
# side effect -- needed only because viz_utils.py itself does `import carla` at import time. This
# script never connects to a running simulator; it is pure post-processing on logged JSON.
import functions  # noqa: F401

from viz_utils import plot_cornering_stiffness_quadrant, print_cornering_stiffness_speed_table

from bicycle import bracket, report, slip_angles, steady_axle_forces


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------

def load_log(path):
    """collect_cornering_data.py's --out. Returns (trials, mass, lf, lr, wheelbase, max_steer, dt).

    Every trial in the file already settled (collect_cornering_data.py drops the ones that
    didn't), so "log" here is never None and is always exactly the steady window -- there is
    nothing left for this script to filter or trim.
    """
    with open(path) as f:
        data = json.load(f)
    return (data["trials"], data["mass"], data["lf"], data["lr"], data["wheelbase"],
            data["max_steer"], data["dt"])


# --------------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------------

def fit_cornering_stiffness(log, mass, lf, lr):
    """Cf, Cr from one (already-steady) trial, assuming r_dot=0 exactly.

    bicycle.steady_axle_forces() turns a_y=v_x*r into the Fyf/Fyr the steady-state force balance
    requires (no Iz needed -- r_dot ~ 0 kills it out of the yaw moment balance; see the module
    docstring for why this assumption was checked, not just made, on this pipeline's own data).
    bicycle.slip_angles() turns the same log's kinematics into alpha_f/alpha_r, using the
    *measured* wheel angle rather than the commanded one (see front_steer_angle). Regressing one
    against the other through the origin is Cf = Fyf/alpha_f, Cr = Fyr/alpha_r -- bracket() gives
    both one-sided slopes plus their geometric mean, which is what gets reported as Cf/Cr.
    """
    v_x = np.array(log["v_x"])
    v_y = np.array(log["v_y"])
    r = np.array(log["r"])
    delta = np.array(log["delta_measured"])

    a_y = v_x * r
    Fyf, Fyr = steady_axle_forces(a_y, mass, lf, lr)
    alpha_f, alpha_r = slip_angles(delta, v_x, v_y, r, lf, lr)

    Cf_fwd, Cf_rev, Cf = bracket(Fyf, alpha_f)
    Cr_fwd, Cr_rev, Cr = bracket(Fyr, alpha_r)
    return {
        "Cf": Cf, "Cf_forward": Cf_fwd, "Cf_reverse": Cf_rev,
        "Cr": Cr, "Cr_forward": Cr_fwd, "Cr_reverse": Cr_rev,
        "alpha_f": alpha_f, "Fyf": Fyf, "alpha_r": alpha_r, "Fyr": Fyr,
    }


def analyze_logged_trials(trials, mass, lf, lr):
    """fit_cornering_stiffness() on every logged trial (every one of which is already
    steady-state data -- nothing here decides "has this settled", that already happened in
    collect_cornering_data.py). Mutates each trial dict in place with "fit".
    """
    for tr in trials:
        tr["fit"] = fit_cornering_stiffness(tr["log"], mass, lf, lr)
    return trials


def pool_by_speed(trials):
    """Regroup per-trial fits by target_speed and pool each speed's (alpha, Fy) pairs on its own
    -- one Cf/Cr per speed (steer angle collapsed out), purely for the informational table
    (print_cornering_stiffness_speed_table()). Nothing downstream selects on this; see the module
    docstring for why speed itself isn't a filtering axis here.
    """
    by_speed = {}
    for tr in trials:
        fit = tr.get("fit")
        if fit is None:
            continue
        by_speed.setdefault(tr["target_speed"], []).append(fit)

    result = {}
    for v, fits in by_speed.items():
        alpha_f = np.concatenate([f["alpha_f"] for f in fits])
        Fyf = np.concatenate([f["Fyf"] for f in fits])
        alpha_r = np.concatenate([f["alpha_r"] for f in fits])
        Fyr = np.concatenate([f["Fyr"] for f in fits])
        Cf_fwd, Cf_rev, Cf = bracket(Fyf, alpha_f)
        Cr_fwd, Cr_rev, Cr = bracket(Fyr, alpha_r)

        Cf_vals = [f["Cf"] for f in fits]
        Cr_vals = [f["Cr"] for f in fits]
        Cf_lo, Cf_hi = min(Cf_vals), max(Cf_vals)
        Cr_lo, Cr_hi = min(Cr_vals), max(Cr_vals)
        result[v] = {
            "Cf": Cf, "Cf_forward": Cf_fwd, "Cf_reverse": Cf_rev,
            "Cr": Cr, "Cr_forward": Cr_fwd, "Cr_reverse": Cr_rev,
            "n_trials": len(fits),
            "Cf_min": Cf_lo, "Cf_max": Cf_hi,
            "Cf_spread_pct": 100.0 * (Cf_hi - Cf_lo) / Cf if len(fits) > 1 else 0.0,
            "Cr_min": Cr_lo, "Cr_max": Cr_hi,
            "Cr_spread_pct": 100.0 * (Cr_hi - Cr_lo) / Cr if len(fits) > 1 else 0.0,
        }
    return result


def filter_by_quadrant(alpha, Fy):
    """Bool mask, True for points where sign(alpha) matches sign(Fy) -- quadrants 1 and 3 of the
    alpha-Fy plane.

    Fy = C*alpha with a positive C means a genuine cornering force always pushes in the same
    direction as the slip angle that caused it. A point in quadrant 2 (alpha<0, Fy>0) or 4
    (alpha>0, Fy<0) has the wrong sign relationship and cannot be a real C*alpha sample -- only
    noise or a measurement artifact lands there.

    Deliberately the ONLY data-cleaning step in this pipeline (see the module docstring): no
    alpha-magnitude bounds, no per-speed statistics, no residual-based rejection. Simpler was the
    point, not a fallback -- every other filter tried on this data either needed its own
    justification for a threshold value or (the speed-level ones) averaged a per-trial bias into a
    "representative" number instead of removing it. This check needs no threshold at all: the sign
    relationship is exact, not a matter of degree.
    """
    alpha = np.asarray(alpha, dtype=float)
    Fy = np.asarray(Fy, dtype=float)
    return (alpha * Fy) >= 0


def pool_trials(trials):
    """Pool every trial's (alpha, Fy) pairs, apply filter_by_quadrant(), fit Cf/Cr (bracket())
    on the survivors. Returns the FULL (unfiltered) alpha/Fy arrays alongside a bool mask each,
    so the caller (the plot) can show every logged sample and which ones were actually used.

    Pooling raw pairs rather than averaging each trial's own Cf is what lets bracket() weight
    points with more signal (bigger alpha^2) the way a least-squares fit is supposed to -- a
    barely-turning trial does not get the same vote as a well-loaded one.
    """
    alpha_f_all, Fyf_all, alpha_r_all, Fyr_all = [], [], [], []
    for tr in trials:
        fit = tr.get("fit")
        if fit is None:
            continue
        alpha_f_all.append(fit["alpha_f"]); Fyf_all.append(fit["Fyf"])
        alpha_r_all.append(fit["alpha_r"]); Fyr_all.append(fit["Fyr"])

    if not alpha_f_all:
        return None

    alpha_f_all = np.concatenate(alpha_f_all); Fyf_all = np.concatenate(Fyf_all)
    alpha_r_all = np.concatenate(alpha_r_all); Fyr_all = np.concatenate(Fyr_all)

    mask_f = filter_by_quadrant(alpha_f_all, Fyf_all)
    mask_r = filter_by_quadrant(alpha_r_all, Fyr_all)

    Cf_fwd, Cf_rev, Cf = bracket(Fyf_all[mask_f], alpha_f_all[mask_f])
    Cr_fwd, Cr_rev, Cr = bracket(Fyr_all[mask_r], alpha_r_all[mask_r])
    return {
        "Cf": Cf, "Cf_forward": Cf_fwd, "Cf_reverse": Cf_rev,
        "Cr": Cr, "Cr_forward": Cr_fwd, "Cr_reverse": Cr_rev,
        "alpha_f": alpha_f_all, "Fyf": Fyf_all, "mask_f": mask_f,
        "alpha_r": alpha_r_all, "Fyr": Fyr_all, "mask_r": mask_r,
        "n_trials": sum(1 for tr in trials if tr.get("fit") is not None),
        "n_dropped_f": int((~mask_f).sum()), "n_points_f": int(len(mask_f)),
        "n_dropped_r": int((~mask_r).sum()), "n_points_r": int(len(mask_r)),
    }


# --------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", default=os.path.join(HERE, "cornering_sweep_log.json"),
                        help="collect_cornering_data.py's own --out")
    parser.add_argument("--out", default=os.path.join(HERE, "cornering_stiffness_speed_report.json"))
    parser.add_argument("--no-plot", action="store_true", help="skip the figure, print only")
    parser.add_argument("--save-plot", action="store_true", help="also save the figure as a PNG")
    parser.add_argument("--no-show", action="store_true",
                        help="build (and, with --save-plot, save) the figure but never call "
                             "plt.show() -- for headless/background runs")
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"))
    args = parser.parse_args()

    trials, mass, lf, lr, wheelbase, max_steer, dt = load_log(args.log_file)
    print(f"Loaded {os.path.basename(args.log_file)}: {len(trials)} (already-steady) trials  "
          f"mass={mass:.1f} kg  lf={lf:.2f} m  lr={lr:.2f} m  dt={dt:.3f} s")
    if not trials:
        print("\nno trials in the log file -- nothing to report")
        return

    analyze_logged_trials(trials, mass, lf, lr)

    # ---- informational only: every trial grouped by speed. Nothing here selects anything --
    # see the module docstring for why speed isn't a filtering axis in this pipeline ---- #
    by_speed_all = pool_by_speed(trials)
    print_cornering_stiffness_speed_table(trials, by_speed_all)

    # ---- pool every trial's samples, apply the one quadrant filter, fit ---- #
    final = pool_trials(trials)
    print(f"\nFinal Cf/Cr, pooled over {final['n_trials']} trials "
          f"({final['n_points_f']} front / {final['n_points_r']} rear samples):")
    print(f"  quadrant filter: dropped {final['n_dropped_f']} of {final['n_points_f']} front "
          f"points, {final['n_dropped_r']} of {final['n_points_r']} rear points "
          f"(sign(alpha) != sign(Fy))")
    print(f"Cf = {final['Cf']:,.0f} N/rad  (bracket [{final['Cf_forward']:,.0f}, "
          f"{final['Cf_reverse']:,.0f}])")
    print(f"Cr = {final['Cr']:,.0f} N/rad  (bracket [{final['Cr_forward']:,.0f}, "
          f"{final['Cr_reverse']:,.0f}])")
    report(final["Cf"], final["Cr"], mass, lf, lr)

    result = {
        "Cf": final["Cf"], "Cr": final["Cr"], "mass": mass, "lf": lf, "lr": lr,
        "n_trials": final["n_trials"],
        "n_points_f": final["n_points_f"], "n_dropped_f": final["n_dropped_f"],
        "n_points_r": final["n_points_r"], "n_dropped_r": final["n_dropped_r"],
        "by_speed": {
            str(v): {"Cf": r["Cf"], "Cr": r["Cr"], "n_trials": r["n_trials"],
                    "Cf_min": r["Cf_min"], "Cf_max": r["Cf_max"],
                    "Cf_spread_pct": r["Cf_spread_pct"],
                    "Cr_min": r["Cr_min"], "Cr_max": r["Cr_max"],
                    "Cr_spread_pct": r["Cr_spread_pct"]}
            for v, r in sorted(by_speed_all.items())
        },
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")

    if not args.no_plot:
        plot_cornering_stiffness_quadrant(
            final["alpha_f"], final["Fyf"], final["mask_f"],
            final["alpha_r"], final["Fyr"], final["mask_r"],
            final["Cf"], final["Cr"],
            out_dir=args.plot_dir if args.save_plot else None, show=not args.no_show)


if __name__ == "__main__":
    main()
