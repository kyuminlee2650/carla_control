"""Longitudinal MPC over a jerk-input double integrator.

State is speed and acceleration, the input is jerk:

    x = [v_x, a_x]^T,   x_{k+1} = A x_k + B j_k
    A = [[1, T], [0, 1]],   B = [T^2/2, T]^T

Over a prediction horizon N_p the input is free for the first N_c steps and
held at j_{N_c-1} afterwards, which is the usual way to keep the decision
vector short without truncating the prediction. Substituting the forward
solution gives the condensed form X = A_bar x0 + B_bar U, and the tracking cost

    J = (X - X_ref)' W1 (X - X_ref) + U' W2 U

becomes a box-constrained QP in U alone:

    min 1/2 U' H U + f' U   s.t.   -j_max <= U <= j_max
    H = 2 (B_bar' W1 B_bar + W2)          (constant -- built once)
    f = 2 B_bar' W1 (A_bar x0 - X_ref)    (rebuilt every cycle)

W2 = w_j I is positive definite, so H is positive definite even where
B_bar' W1 B_bar is rank deficient, and the QP has a unique solution.

Only the first element of U* is applied (receding horizon). The acceleration
command handed to the lookup table is integrated from the previous command
rather than re-based on the measured acceleration each cycle:

    a_cmd <- a_cmd + T j_0*

Anchoring it to the measurement instead (a_cmd = a_meas + T j) deadlocks as
soon as the plant lags: the command can never sit further than one jerk step
(T j_max = 0.2 m/s^2 here) from wherever the vehicle currently is, so a lagging
response drags the command along with it and the optimiser answers max jerk
every cycle. In simulation with a 0.2 s plant lag that pins jerk on its limit
96% of the time.

Integrating instead lets the command lead the plant, at the cost of winding up
when the vehicle cannot deliver -- the command drifted to +-16 m/s^2 against a
+-2.5 capability in the same test. The fix is the usual conditional-integration
rule: when the pedal is already railed and the increment pushes further that
way, drop the increment. That needs no model of the achievable range, only the
saturation flag the PI layer already produces.
"""

import numpy as np
import osqp
from scipy import sparse


class LongitudinalMPC:
    def __init__(self, dt, n_p=20, n_c=10, w_v=8, w_a=0.1, w_j=30, jerk_max=4.13,
                 command_mode="integrate"):
        self.T = dt
        self.n_p = n_p
        self.n_c = n_c
        self.jerk_max = jerk_max
        self.command_mode = command_mode    # "integrate" | "anchor"
        self.a_cmd = None

        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.B = np.array([[0.5 * dt * dt], [dt]])

        self.A_bar = self._build_a_bar()
        self.B_bar = self._build_b_bar()

        # W1 = I_Np (x) Q  weights every predicted step the same way
        Q = np.diag([w_v, w_a])
        self.W1 = np.kron(np.eye(n_p), Q)
        self.W2 = w_j * np.eye(n_c)

        self.H = 2.0 * (self.B_bar.T @ self.W1 @ self.B_bar + self.W2)
        self.H = 0.5 * (self.H + self.H.T)          # symmetrise against round-off
        self._B_W1 = 2.0 * self.B_bar.T @ self.W1   # reused every cycle to form f

        self._solver = osqp.OSQP()
        self._solver.setup(
            P=sparse.csc_matrix(self.H),
            q=np.zeros(n_c),
            A=sparse.csc_matrix(np.eye(n_c)),       # box constraint on U itself
            l=-jerk_max * np.ones(n_c),
            u=jerk_max * np.ones(n_c),
            verbose=False,
            polish=False,   # polishing logs to stdout even when verbose is off
        )
        self.last_solution = np.zeros(n_c)
        self.last_status = "unsolved"

    def _build_a_bar(self):
        """[A; A^2; ...; A^Np], stacked 2Np x 2."""
        blocks = []
        power = np.eye(2)
        for _ in range(self.n_p):
            power = power @ self.A
            blocks.append(power.copy())
        return np.vstack(blocks)

    def _build_b_bar(self):
        """2Np x Nc. Columns before the last hold a single A^(i-c) B term; the
        last column accumulates every step the held input still acts over."""
        B_bar = np.zeros((2 * self.n_p, self.n_c))
        powers = [np.eye(2)]
        for _ in range(self.n_p):
            powers.append(powers[-1] @ self.A)

        for i in range(1, self.n_p + 1):
            rows = slice(2 * (i - 1), 2 * i)
            for c in range(1, self.n_c + 1):
                if c < self.n_c:
                    if c <= i:
                        B_bar[rows, c - 1] = (powers[i - c] @ self.B).ravel()
                else:
                    if i >= self.n_c:
                        acc = sum(powers[m] for m in range(0, i - self.n_c + 1))
                        B_bar[rows, c - 1] = (acc @ self.B).ravel()
        return B_bar

    def reference(self, v_des, a_des=0.0):
        """Stack a constant [v_des, a_des] target over the horizon.

        Accepts scalars or per-step sequences, so a planner that hands over a
        speed profile can pass it straight through.
        """
        v = np.broadcast_to(np.asarray(v_des, dtype=float), (self.n_p,))
        a = np.broadcast_to(np.asarray(a_des, dtype=float), (self.n_p,))
        return np.column_stack([v, a]).ravel()

    def reset(self, a_x=0.0):
        """Start the acceleration command from the vehicle's current state."""
        self.a_cmd = float(a_x)

    def solve(self, v_x, a_x, x_ref, u_prev=None):
        """One receding-horizon step. Returns (jerk, a_cmd).

        u_prev is the pedal input actually applied last cycle; pass it so a
        railed actuator stops the command winding further into saturation.
        """
        x0 = np.array([[v_x], [a_x]])
        f = self._B_W1 @ (self.A_bar @ x0 - x_ref.reshape(-1, 1))

        self._solver.update(q=f.ravel())
        result = self._solver.solve()
        self.last_status = result.info.status

        if result.x is None or not np.all(np.isfinite(result.x)):
            # keep driving on the previous plan rather than dropping to zero jerk
            u = self.last_solution
        else:
            u = result.x
            self.last_solution = u.copy()

        jerk = float(np.clip(u[0], -self.jerk_max, self.jerk_max))

        if self.command_mode == "anchor":
            self.a_cmd = float(a_x + self.T * jerk)
            return jerk, self.a_cmd

        if self.a_cmd is None:
            self.a_cmd = float(a_x)
        proposal = self.a_cmd + self.T * jerk
        if u_prev is not None:
            pushing_up = u_prev >= 0.999 and proposal > self.a_cmd
            pushing_down = u_prev <= -0.999 and proposal < self.a_cmd
            if pushing_up or pushing_down:
                proposal = self.a_cmd     # pedal already railed -- do not wind further
        self.a_cmd = float(proposal)
        return jerk, self.a_cmd
