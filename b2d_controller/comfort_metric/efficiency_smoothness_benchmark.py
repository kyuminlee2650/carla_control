"""Corrected Bench2Drive comfort metric used only by the leaderboard finalizer."""

from __future__ import print_function

import numpy as np
from scipy.signal import savgol_filter


MAX_ABS_MAG_JERK = 8.37       # m/s^3
MAX_ABS_LAT_ACCEL = 4.89      # m/s^2
MAX_LON_ACCEL = 2.40          # m/s^2
MIN_LON_ACCEL = -4.05         # m/s^2
MAX_ABS_YAW_ACCEL = 1.93      # rad/s^2
MAX_ABS_LON_JERK = 4.13       # m/s^3
MAX_ABS_YAW_RATE = 0.95       # rad/s
CARLA_TICK_SECONDS = 0.05


def _within_bound(values, lower, upper):
    values = np.asarray(values)
    return bool(np.all((values > lower) & (values < upper)))


def _smooth(values, window_size, poly_order, deriv=0, delta=1.0):
    return savgol_filter(values, window_length=window_size,
                         polyorder=poly_order, deriv=deriv, delta=delta,
                         axis=-1)


def compute_comfort_metric(acceleration, angular_velocity, forward_vector,
                           right_vector, location, rotation, window_size=7,
                           poly_order=2, time_interval=CARLA_TICK_SECONDS):
    """Return whether every corrected comfort channel is in range."""
    del location, rotation
    window_size = min(window_size, len(acceleration))
    if poly_order >= window_size:
        raise ValueError("%s < %s does not hold!" % (poly_order, window_size))

    acceleration = np.asarray(acceleration, dtype=float)
    forward_vector = np.asarray(forward_vector, dtype=float)
    right_vector = np.asarray(right_vector, dtype=float)
    lon_acc = np.einsum("ij,ij->i", acceleration[:, :2], forward_vector[:, :2])
    lat_acc = np.einsum("ij,ij->i", acceleration[:, :2], right_vector[:, :2])
    magnitude_acc = np.hypot(acceleration[:, 0], acceleration[:, 1])

    # CARLA reports angular velocity in degrees/s. A rate is not an angle, so
    # convert it once and do not phase-unwrap it before differentiating.
    yaw_rate = np.deg2rad(np.asarray(angular_velocity, dtype=float)[:, 2])
    yaw_acc = _smooth(yaw_rate, window_size, poly_order, deriv=1,
                      delta=time_interval)
    yaw_rate = _smooth(yaw_rate, window_size, poly_order)
    lon_acc = _smooth(lon_acc, window_size, poly_order)
    lat_acc = _smooth(lat_acc, window_size, poly_order)
    magnitude_acc = _smooth(magnitude_acc, window_size, poly_order)
    magnitude_jerk = _smooth(magnitude_acc, window_size, poly_order, deriv=1,
                              delta=time_interval)
    lon_jerk = _smooth(lon_acc, window_size, poly_order, deriv=1,
                       delta=time_interval)

    return all((
        _within_bound(lon_acc, MIN_LON_ACCEL, MAX_LON_ACCEL),
        _within_bound(lat_acc, -MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL),
        _within_bound(magnitude_jerk, -MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK),
        _within_bound(lon_jerk, -MAX_ABS_LON_JERK, MAX_ABS_LON_JERK),
        _within_bound(yaw_acc, -MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL),
        _within_bound(yaw_rate, -MAX_ABS_YAW_RATE, MAX_ABS_YAW_RATE),
    ))


def seg_compute_comfort_metric(acceleration, angular_velocity, forward_vector,
                               right_vector, location, rotation, window_size=7,
                               poly_order=2, time_interval=CARLA_TICK_SECONDS,
                               per_step=20):
    """Return the fraction of 20 Hz fixed-length segments that are comfortable."""
    if len(angular_velocity) <= per_step:
        return float(compute_comfort_metric(
            acceleration, angular_velocity, forward_vector, right_vector,
            location, rotation, window_size, poly_order, time_interval))
    outcomes = []
    for start in range(0, len(angular_velocity), per_step):
        stop = start + per_step
        if stop > len(angular_velocity):
            continue
        outcomes.append(compute_comfort_metric(
            acceleration[start:stop], angular_velocity[start:stop],
            forward_vector[start:stop], right_vector[start:stop],
            location[start:stop], rotation[start:stop], window_size,
            poly_order, time_interval))
    return float(sum(outcomes)) / len(outcomes) if outcomes else 0.0
