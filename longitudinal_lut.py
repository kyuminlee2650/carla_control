"""Runtime lookup for the (gear, v_x, a_x) -> u longitudinal control-input table.

Loads the .npz produced by build_reverse_lut.py (which itself inverts the raw
sweep data from build_longitudinal_lut.py) and serves fast bilinear lookups
for a longitudinal MPC/controller.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class LongitudinalLUT:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.gears = [int(g) for g in data["gears"]]
        self._interp = {}
        self._bounds = {}
        for gear in self.gears:
            v_grid = data[f"g{gear}_v"]
            a_grid = data[f"g{gear}_a"]
            u_table = data[f"g{gear}_u"]
            self._interp[gear] = RegularGridInterpolator(
                (v_grid, a_grid), u_table, bounds_error=False, fill_value=None
            )
            self._bounds[gear] = (v_grid.min(), v_grid.max(), a_grid.min(), a_grid.max())

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
