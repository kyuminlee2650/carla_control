"""Longitudinal acceleration controller: LUT feedforward + PI feedback.

The lookup table answers "what control input produces this acceleration", which
validation showed is right to about 0.4 m/s^2 inside its calibrated envelope.
The PI loop closes the remaining gap -- table error, road grade, anything the
sweep did not model -- and the sum is clipped to the actuator's range.

The measured acceleration used for feedback is a filtered dv/dt rather than
CARLA's get_acceleration(): the two agree closely overall (r = 0.997) but the
raw sensor swings by several m/s^2 in the lugging regimes, and differentiating
speed keeps the feedback consistent with how the controller is scored.
"""


class LongitudinalAccelPI:
    def __init__(self, lut, kp, ki, tau=0.1):
        self.lut = lut
        self.kp = kp
        self.ki = ki
        self.tau = tau          # first-order filter time constant for measured accel (s)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.a_filt = 0.0
        self.prev_v = None
        self.last_gear = None
        self.saturated = False

    def measure_accel(self, v_x, dt):
        """Filtered dv/dt. Returns the current acceleration estimate."""
        if self.prev_v is None:
            self.prev_v = v_x
            return self.a_filt
        a_raw = (v_x - self.prev_v) / dt
        self.prev_v = v_x
        alpha = dt / (self.tau + dt)
        self.a_filt += alpha * (a_raw - self.a_filt)
        return self.a_filt

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

    def step(self, gear, v_x, a_cmd, dt, a_meas=None):
        """One control cycle. Returns u in [-1, 1].

        Pass a_meas when the caller already estimates acceleration (an MPC needs
        it for its own state), so both layers act on the same signal instead of
        filtering the same speed twice.
        """
        if a_meas is None:
            a_meas = self.measure_accel(v_x, dt)
        else:
            self.a_filt = a_meas
            self.prev_v = v_x
        error = a_cmd - a_meas

        self.integral += error * dt
        u = self.feedforward(gear, v_x, a_cmd) + self.kp * error + self.ki * self.integral

        clipped = max(-1.0, min(1.0, u))
        self.saturated = clipped != u
        return clipped
