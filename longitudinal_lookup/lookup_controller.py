"""Longitudinal acceleration controller: LUT feedforward + PID feedback.

The lookup table (LongitudinalLUT, longitudinal_lut.py) answers "what control
input produces this acceleration", which validation showed is right to about
0.4 m/s^2 inside its calibrated envelope. functions.PID closes the remaining
gap -- table error, road grade, anything the sweep did not model -- and the
sum is clipped to the actuator's range. This class only owns the LUT lookup
and the gear fallback; the PID math itself lives in functions.PID, so there is
one implementation of integral/anti-windup/derivative in the repo instead of
two.

kd defaults to 0 (plain PI, the original design). It exists because a single
kp/ki tuned for one operating point is not automatically stable at another: the
LUT's du/da is markedly steeper in low gears at low speed than in a cruising
gear (low gear multiplies torque, so the same pedal delta moves acceleration
much further), so gains that track tightly at highway speed can undamp into a
sustained oscillation idling in 1st -- confirmed live, not assumed. Raising kd
did not resolve that cleanly (it damped some, then saturated harder before it
damped fully), so the working fix is conservative kp/ki that stay stable
everywhere rather than kd -- see validate_lut.py's defaults.

a_meas must come from the caller as a real IMU reading (see validate_lut.py) --
every acceleration measurement in this repo is IMU-based now, on the ground
that it is the one signal an actual car could read off a real sensor. This
class has no fallback that computes it another way (differentiating v_x,
CARLA's get_acceleration()), so a numerical stand-in never quietly substitutes
for the sensor without that being visible at the call site.

Known trade-off: functions.PID's anti-windup only sees its own kp/ki/kd
contribution, not the feedforward term added on top here, so a combined output
that saturates only because ff is already near a rail is not caught by it --
only PID-side saturation is. In practice ff is rarely close enough to +-1 on
its own for this to matter, but it is not the same guarantee a windup check
over the full ff+PID sum would give.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import PID


class LookupController:
    def __init__(self, lut, kp, ki, kd=0.0, dt=0.05, use_feedforward=True):
        self.lut = lut
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.use_feedforward = use_feedforward  # False: plain PID straight to the pedal, no LUT
                                                 # term at all -- the baseline the LUT is meant to
                                                 # beat, not just a kp=ki=0 ablation of it
        self.reset()

    def reset(self):
        self.pid = PID(self.kp, self.ki, self.kd, self.dt)
        self.last_gear = None
        self.saturated = False

    def feedforward(self, gear, v_x, a_cmd):
        """LUT term. During a gear change CARLA reports gear 0, which the table
        has no entry for; the last engaged gear is the closest thing to valid.

        At startup there is no last gear either, and returning 0 there deadlocks:
        no input means the transmission never engages, so the gear stays 0
        forever. Guess from the speed instead -- one tick of throttle gets the
        vehicle moving and the real gear is readable from then on.
        """
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
        """One control cycle. Returns u in [-1, 1]. a_meas: IMU-measured acceleration."""
        error = a_cmd - a_meas
        ff = self.feedforward(gear, v_x, a_cmd) if self.use_feedforward else 0.0
        u = ff + self.pid.step(error)
        clipped = max(-1.0, min(1.0, u))
        self.saturated = clipped != u
        return clipped
