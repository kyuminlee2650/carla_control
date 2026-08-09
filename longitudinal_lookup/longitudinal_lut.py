"""Runtime lookup for the (gear, v_x, a_x) -> u longitudinal control-input table.

Loads the .npz produced by build_lut.py (which itself fits the raw sweep data
from collect_lut_data.py) and serves fast bilinear lookups for a longitudinal
MPC/controller.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class LongitudinalLUT:
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
            # speeds where the calibration was repeatable enough to invert;
            # tables built before this existed fall back to the grid extent
            key = f"g{gear}_vrange"
            self.speed_range[gear] = (tuple(float(x) for x in data[key]) if key in data
                                      else (float(v_grid.min()), float(v_grid.max())))

    def lookup(self, gear, v_x, a_x):
        """u in [-1, 1] for the requested gear/speed/desired-accel, clamped to the
        calibrated envelope (nearest calibrated point outside it, not extrapolated)."""
        if gear not in self._interp:
            raise ValueError(f"No calibration data for gear {gear}; available: {self.gears}")
        v_min, v_max, a_min, a_max = self._bounds[gear]
        v_q = min(max(v_x, v_min), v_max)
        a_q = min(max(a_x, a_min), a_max)
        u = float(self._interp[gear]([[v_q, a_q]])[0])
        return max(-1.0, min(1.0, u))

    def supports(self, gear, v_x):
        """Whether this gear was calibrated reliably at this speed. A lookup
        outside the range still returns a number (clamped to the nearest
        calibrated point), but it is not backed by trustworthy data."""
        if gear not in self.speed_range:
            return False
        lo, hi = self.speed_range[gear]
        return lo <= v_x <= hi

    def gears_for_speed(self, v_x):
        """Every gear calibrated at this speed, lowest first (empty if none).

        Deliberately not a shift policy: the calibration says which gears have
        trustworthy data here, not which one the vehicle should be in. Picking
        among them is the controller's decision.
        """
        return [g for g in self.gears if self.supports(g, v_x)]
