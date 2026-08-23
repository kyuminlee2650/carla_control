r"""LOCAL PATCH, NOT A VERBATIM UPSTREAM COPY -- read this before trusting or redistributing it.

Source: Thinklab-SJTU/Bench2Drive, tools/efficiency_smoothness_benchmark.py, tag 0.0.4 (also that
repo's default branch; its only commit touching this file is 2024-08-19, so 0.0.4 is the newest
public version -- confirmed via the GitHub API). That file is licensed CC BY-NC-ND 4.0
(Attribution-NonCommercial-NoDerivatives): copying it verbatim is fine, but this file is NOT
verbatim -- three lines are patched below (see PATCH markers), which makes it a derivative work
the license does not actually grant permission to share. It is kept here only as a stand-in for
internal use on this machine until it can be replaced by the real thing (see next paragraph);
don't push this file to a public/shared remote.

Why patch instead of just using tag 0.0.4 as-is: this project (see b2d_controller/b2d_metrics.py's
comfort_of()) already found and reported three defects in this exact module, and the lab's own
checkout was corrected upstream on 2026-08-18 (md5 0c650615...) -- verified DIFFERENT from tag
0.0.4's md5 (93c9b4b0...), i.e. tag 0.0.4 is still the pre-fix version. That corrected file lives
at /home/ailab/2026intern/kmlee/vad_demo_video/Bench2Drive/tools/efficiency_smoothness_benchmark.py
on the lab's Ubuntu machine, not reachable from here. The three patches below are this project's
own best reconstruction of those three defects from their written description (b2d_metrics.py's
comfort_of() docstring) applied to the tag-0.0.4 source -- NOT verified byte-for-byte against the
real md5 0c650615... file. Replace this file with that real one (same filename, so no other code
needs to change) the next time you're on that machine, and delete this docstring's caveat once
it's confirmed to match.

Patches applied (search PATCH #1/#2/#3 below):
  #1 deg/s vs rad/s: angular_velocity is passed in deg/s everywhere this project calls in from
     (see viz_utils.b2d_comfortness/b2d_controller.b2d_metrics.comfort_of), but max_abs_yaw_rate/
     max_abs_yaw_accel below are documented in rad/s -- the original never converted, so a real
     yaw-rate excursion in deg/s was being compared against a bound ~57x too tight in the wrong
     units. Fixed by converting the raw angular-velocity z-channel to radians right where it's
     extracted, before anything downstream (phase-unwrap, differentiation, bound check) touches it.
  #2 "yaw acceleration" never differentiated: _z_yaw_acc was computed as savgol_filter(_z_yaw_rate,
     ...) with no deriv= argument, i.e. a smoothed COPY of yaw rate, not its derivative -- the
     six-channel comfort check was silently checking yaw rate twice (once correctly as "yaw rate",
     once mislabeled as "yaw acceleration") and never actually checking angular acceleration at
     all. Fixed by adding deriv=deriv_order, delta=dx, matching how magnitude_jerk/lon_jerk already
     differentiate acceleration into jerk a few lines below.
  #3 0.1s vs 0.05s derivative step: compute_comfort_metric()'s time_interval defaults to 0.1s, but
     this project (and Bench2Drive's own CARLA runs) samples at 0.05s -- and seg_compute_comfort_metric()
     never actually forwards its own window_size/poly_order/deriv_order/time_interval parameters
     into its two compute_comfort_metric() calls (see those call sites, unchanged from upstream: a
     separate latent bug, not one of the three documented ones, but it means the fix has to land on
     compute_comfort_metric()'s OWN default, not seg_compute_comfort_metric()'s -- changing the
     latter alone would have no effect). Fixed by changing that default to 0.05.

  Also (mentioned in b2d_metrics.py's docstring as a related but separate correction, not one of
  the "three defects", yet explicitly part of "the official behaviour... used from here on"):
  phase-unwrapping is no longer applied to the yaw RATE ("a rate is not an angle" -- phase-unwrap
  only makes sense for a cyclic angle like heading, not an already-continuous rate signal). The
  original inherited this from copy-pasting the same _phase_unwrap() used elsewhere for headings;
  removed here for #2's sake too, since differentiating an unwrap-mangled rate would be wrong. Per
  that same docstring this made no numerical difference at the rates real runs reach, so it is not
  expected to change scores here either -- it is applied anyway to match the described official
  behaviour as closely as this reconstruction can.

Below this docstring, everything is unchanged from the public tag 0.0.4 source except the four
PATCH-marked spots.
"""

from scipy.signal import savgol_filter
# import numpy.typing as npt
import numpy as np
import json
import re
import argparse
import os

# (1) ego_jerk_metric,
max_abs_mag_jerk = 8.37  # [m/s^3]

# (2) ego_lat_acceleration_metric
max_abs_lat_accel = 4.89  # [m/s^2]

# (3) ego_lon_acceleration_metric
max_lon_accel = 2.40  # [m/s^2]
min_lon_accel = -4.05

# (4) ego_yaw_acceleration_metric
max_abs_yaw_accel = 1.93  # [rad/s^2]

# (5) ego_lon_jerk_metric
max_abs_lon_jerk = 4.13  # [m/s^3]

# (6) ego_yaw_rate_metric
max_abs_yaw_rate = 0.95  # [rad/s]

'''
window size = 8
'''

def chunk_arrays(arrays, m):
    chunks = [chunk_array(arr, m) for arr in arrays]
    return chunks

def chunk_array(arr, m):
    chunks = [arr[i * m:(i + 1) * m] for i in range((len(arr) + m - 1) // m)]
    return chunks

def seg_compute_comfort_metric(acceleration,
    angular_velocity,
    forward_vector,
    right_vector,
    location,
    rotation,
    window_size: int = 7,
    poly_order: int = 2,
    deriv_order: int = 1,
    time_interval: float = 0.1,
    per_step: int = 20):
    episode_len = len(angular_velocity)
    if episode_len <= per_step:
        res = compute_comfort_metric( acceleration, angular_velocity, forward_vector, right_vector, location, rotation)
        return 1. if res else 0.
    seg_data = chunk_arrays([acceleration, angular_velocity, forward_vector, right_vector, location, rotation], per_step)

    res = []
    for index in range(len(seg_data[0])):
        if len(seg_data[0][index]) < per_step:
            continue
        res.append(compute_comfort_metric(seg_data[0][index], seg_data[1][index], seg_data[2][index], seg_data[3][index], seg_data[4][index], seg_data[5][index]))
    return res.count(True) / len(res)
    pass

def compute_comfort_metric(
    acceleration,
    angular_velocity,
    forward_vector,
    right_vector,
    location,
    rotation,
    window_size: int = 7,
    poly_order: int = 2,
    deriv_order: int = 1,
    time_interval: float = 0.05  # PATCH #3: was 0.1 -- actual sample spacing here is 0.05s
):
    window_size = min(window_size, len(acceleration))
    if not (poly_order < window_size):
        raise ValueError(f"{poly_order} < {window_size} does not hold!")

    _2d_acceleration = np.array([[v[0], v[1]] for v in acceleration])
    _2d_forward_vector = np.array([[vec[0], vec[1]] for vec in forward_vector])
    _2d_right_vector = np.array([[vec[0], vec[1]] for vec in right_vector])
    # PATCH #1: np.radians(...) added -- angular_velocity[:, 2] arrives in deg/s (this project's
    # own convention), but max_abs_yaw_rate/max_abs_yaw_accel above are in rad/s; the original left
    # this unconverted.
    _z_angular_rate = np.radians(np.array([a[2] for a in angular_velocity]))
    # PATCH (see docstring): no longer phase-unwrapped -- a rate is not an angle, matching the
    # described official behaviour rather than the original's copy-pasted _phase_unwrap() call.
    _z_yaw_rate = _z_angular_rate

    lon_acc = np.einsum('ij,ij->i', _2d_acceleration, _2d_forward_vector)
    lat_acc = np.einsum('ij,ij->i', _2d_acceleration, _2d_right_vector)
    magnitude_acc = np.hypot(_2d_acceleration[:, 0], _2d_acceleration[:, 1])

    dx = time_interval

    # PATCH #2: deriv=deriv_order, delta=dx added -- without them this was savgol-SMOOTHING
    # _z_yaw_rate (deriv defaults to 0 in savgol_filter), not differentiating it, so "yaw
    # acceleration" was actually just a smoothed copy of yaw rate.
    _z_yaw_acc = savgol_filter(
        _z_yaw_rate,
        polyorder=poly_order,
        window_length=window_size,
        deriv=deriv_order,
        delta=dx,
        axis=-1,
    )

    _z_yaw_rate = savgol_filter(
        _z_yaw_rate,
        polyorder=poly_order,
        window_length=window_size,
        axis=-1,
    )

    lon_acc = savgol_filter(
        lon_acc,
        polyorder=poly_order,
        window_length=window_size,
        axis=-1,
    )

    lat_acc = savgol_filter(
        lat_acc,
        polyorder=poly_order,
        window_length=window_size,
        axis=-1,
    )

    magnitude_acc = savgol_filter(
        magnitude_acc,
        polyorder=poly_order,
        window_length=window_size,
        axis=-1,
    )

    magnitude_jerk = savgol_filter(
        magnitude_acc,
        polyorder=poly_order,
        window_length=window_size,
        deriv=deriv_order,
        delta=dx,
        axis=-1,
    )

    lon_jerk = savgol_filter(
        lon_acc,
        polyorder=poly_order,
        window_length=window_size,
        deriv=deriv_order,
        delta=dx,
        axis=-1,
    )

    res = np.array([
        _within_bound(
        lon_acc, min_bound=min_lon_accel, max_bound=max_lon_accel
    ),
        _within_bound(
        lat_acc, min_bound=-max_abs_lat_accel, max_bound=max_abs_lat_accel
    ),
        _within_bound(
        magnitude_jerk, min_bound=-max_abs_mag_jerk, max_bound=max_abs_mag_jerk
    ),
        _within_bound(
        lon_jerk, min_bound=-max_abs_lon_jerk, max_bound=max_abs_lon_jerk
    ),
        _within_bound(
        _z_yaw_acc, min_bound=-max_abs_yaw_accel, max_bound=max_abs_yaw_accel
    ),
        _within_bound(
        _z_yaw_rate, min_bound=-max_abs_yaw_rate, max_bound=max_abs_yaw_rate
    )
    ])
    res = bool(np.all(res))
    return res

def _approximate_derivatives(
    y,
    x,
    window_size: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
):
    window_size = min(window_size, len(x))

    if not (poly_order < window_size):
        raise ValueError(f"{poly_order} < {window_size} does not hold!")

    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative = savgol_filter(
        y,
        polyorder=poly_order,
        window_size=window_size,
        deriv=deriv_order,
        delta=dx,
        axis=axis,
    )
    return derivative

def _within_bound(
    metric,
    min_bound = None,
    max_bound = None,
):
    """
    Determines wether values in batch-dim are within bounds.
    :param metric: metric values
    :param min_bound: minimum bound, defaults to None
    :param max_bound: maximum bound, defaults to None
    :return: array of booleans wether metric values are within bounds
    """
    min_bound = min_bound if min_bound else float(-np.inf)
    max_bound = max_bound if max_bound else float(np.inf)
    metric_values = np.array(metric)
    metric_within_bound = (metric_values > min_bound) & (metric_values < max_bound)
    return np.all(metric_within_bound, axis=-1)

def _phase_unwrap(headings):
    """
    Returns an array of heading angles equal mod 2 pi to the input heading angles,
    and such that the difference between successive output angles is less than or
    equal to pi radians in absolute value
    :param headings: An array of headings (radians)
    :return The phase-unwrapped equivalent headings.
    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation of yaw to be very large.
    # We want unwrapped[j] = headings[j] - 2*pi*adjustments[j] for some integer-valued adjustments making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= headings[j+1] - headings[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (headings[j+1] - headings[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((headings[j+1] - headings[j]) / (2*pi)).
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(
        np.round(np.diff(headings, axis=-1) / two_pi), axis=-1
    )
    unwrapped = headings - two_pi * adjustments
    return unwrapped


def read_from_json(filepath, metric_dir=None):
    with open(filepath, 'r') as f:
        data = json.load(f)
    all_data = []
    driving_efficiency = []
    for record in data["_checkpoint"]["records"]:
        filepath = os.path.join(metric_dir, record["save_name"], 'metric_info.json')
        temp_dict = {}
        temp_dict["acceleration"] = []
        temp_dict["angular_velocity"] = []
        temp_dict["forward_vector"] = []
        temp_dict["right_vector"] = []
        temp_dict["location"] = []
        temp_dict["rotation"] = []
        with open(filepath, 'r') as file:
            json_data = json.load(file)
            for k, v in json_data.items():
                temp_dict["acceleration"].append(v["acceleration"])
                temp_dict["angular_velocity"].append(v["angular_velocity"])
                temp_dict["forward_vector"].append(v["forward_vector"])
                temp_dict["right_vector"].append(v["right_vector"])
                temp_dict["location"].append(v["location"])
                temp_dict["rotation"].append(v["rotation"])
        for k, v in temp_dict.items():
            temp_dict[k] = np.array(v)
        all_data.append(temp_dict)
        if len(record["infractions"]["min_speed_infractions"]) < 1:
            continue
        else:
            driving_e = []
            for min_speed_infrac in record["infractions"]["min_speed_infractions"]:
                number = re.search(r"\b\d+\.?\d*%", min_speed_infrac)
                if float(number.group().rstrip('%')) > 1000:
                    continue
                driving_e.append(float(number.group().rstrip('%')))
            driving_e = sum(driving_e) / len(driving_e)
            driving_efficiency.append(driving_e)
    return all_data, driving_efficiency

if __name__=='__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-f', '--file', default="uniad_b2d_traj/merged.json", help='route file')
    argparser.add_argument('-m', '--metric_dir', default="eval_bench2drive220_uniad_traj/")
    args = argparser.parse_args()
    all_data, driving_efficiency_list = read_from_json(args.file, args.metric_dir)
    comfort_res = []
    for record in all_data:
        comfort_res.append(seg_compute_comfort_metric(**record))
    print(f'Driving Efficiency={sum(driving_efficiency_list) / len(driving_efficiency_list)}')
    print(f'Driving Smoothness={sum(comfort_res)/len(comfort_res)}')
