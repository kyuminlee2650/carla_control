r"""Longitudinal MPC (speed-tracking QP) + pedal layer, steer forced to 0.

Same isolation as longitudinal_PID.py -- no Stanley, no desired path, fixed spawn on Town06's
longest straight (index 86) -- so the runs test the exact same speed-profile-tracking task.
--controller takes one or more of three stacks and, given more than one, overlays them on one
comparison figure instead of eyeballing separate runs (viz_utils.plot_longitudinal accepts
{label: hist}):

    mpc+lut+pid   (default) SpeedMPC -> a_cmd -> LUT feedforward + PID pedal layer
    mpc+pid      SpeedMPC -> a_cmd -> PID-only pedal layer, no LUT term at all -- the same
                 use_feedforward=False baseline validate_lut.py's "PID only" trial uses, so this
                 isolates what the LUT is actually buying the MPC stack over closing the
                 acceleration loop with feedback alone
    pid          plain speed PID straight to the pedal, no MPC involved -- functions.PID with
                 longitudinal_PID.py's own gains, so it's the same controller, not a
                 reimplementation of it

    --controller mpc+lut+pid mpc+pid          # does the LUT feedforward help?
    --controller mpc+lut+pid pid              # does the MPC help at all, top to bottom?
    --controller mpc+lut+pid mpc+pid pid      # all three at once

Control stack (two layers, cascaded) for mpc+lut+pid / mpc+pid:

    1. SpeedMPC (this file): a linear MPC over the plant

           x_k = v_{x,k},   x_{k+1} = A x_k + B a_{x,k},   A=1, B=T

       i.e. a first-order integrator from commanded acceleration to speed. The decision variable
       is the acceleration sequence itself (not jerk) -- free for the first Nc steps of the
       horizon, held after that -- so solving the QP each cycle and applying only its first
       element (receding horizon) hands the layer below a target acceleration a_cmd directly, with
       no extra integration/anchoring step needed. See the SpeedMPC docstring for the QP.

    2. LookupController (longitudinal_lookup/lookup_controller.py): the same LUT-feedforward +
       PID acceleration tracker validate_lut.py validates on its own -- turns a_cmd into a pedal
       command u in [-1, 1]. mpc+pid runs the identical class with use_feedforward=False rather
       than reimplementing a bare PID. This script does not reimplement that layer either way; it
       is imported as-is so every run here is testing the exact stack validate_lut.py validated.

a_meas for the pedal layer comes raw off the IMU (functions.ImuAcceleration.a_x_raw), not low-pass
filtered -- same reasoning as validate_lut.py: the LUT was calibrated against the raw signal, and
filtering only on this side would compare the controller's a_meas against a lagged version of what
it was fit on. A separately filtered a_x (tau=0.15, matching longitudinal_PID.py) is kept purely
for the result plot, exactly as longitudinal_PID.py does for its own a_x. Jerk is not derived from
it at all any more -- every jerk number comes from the scoring module's own derivative, rebuilt
post-run by viz_utils.add_scored_comfort_channels().

Every controller shares one run_trial() (spawn -> warm-up -> log -> teardown) instead of each
having its own copy of that harness; only the per-step control law differs (PidController /
MpcController, both a single step(ctx) -> pedal method).

Usage (Ubuntu):
    cd ~/carla_control
    .venv/bin/python longitudinal_mpc.py --profile constant --initial-speed 15
    .venv/bin/python longitudinal_mpc.py --profile sine --initial-speed 15 --sine-amplitude 3 --sine-period 10 --save-plot
    .venv/bin/python longitudinal_mpc.py --profile step --initial-speed 15 --step-size 5 --step-time 10 --save-plot
    .venv/bin/python longitudinal_mpc.py --controller mpc+lut+pid mpc+pid pid --profile sine --save-plot

    # emergency stop and restart: hold a full stop for 5s from t=5, then demand 15 m/s again
    .venv/bin/python longitudinal_mpc.py --profile step --initial-speed 15 --step-size -15 \
        --step-time 5 --step-duration 5 --duration 20 --save-plot

    # several controllers recorded and stitched left-to-right into one mp4, in --controller order
    .venv/bin/python longitudinal_mpc.py --controller pid mpc+pid mpc+lut+pid --profile sine \
        --save-plot --record

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe longitudinal_mpc.py --profile constant --initial-speed 15
    .venv\Scripts\python.exe longitudinal_mpc.py --profile sine --initial-speed 15 --sine-amplitude 3 --sine-period 10
    .venv\Scripts\python.exe longitudinal_mpc.py --profile constant --initial-speed 15 --times-run 10 --save-plot --record
    .venv\Scripts\python.exe longitudinal_mpc.py --controller mpc+lut+pid mpc+pid --profile sine --initial-speed 15 --sine-amplitude 3 --sine-period 3 --save-plot
    .venv\Scripts\python.exe longitudinal_mpc.py --controller mpc+lut+pid mpc+pid pid --profile step --initial-speed 15 --step-size -5 --step-time 5 --save-plot

    # emergency stop and restart: hold a full stop for 5s from t=5, then demand 15 m/s again
    .venv\Scripts\python.exe longitudinal_mpc.py --profile step --initial-speed 15 --step-size -15 --step-time 5 --step-duration 5 --duration 20 --save-plot

    # several controllers recorded and stitched left-to-right into one mp4, in --controller order
    .venv\Scripts\python.exe longitudinal_mpc.py --controller pid mpc+pid mpc+lut+pid --profile sine --save-plot --record
"""

import argparse
import math
import os
import queue
import sys
import time
from types import SimpleNamespace

import numpy as np
import osqp
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))

# see functions.py for why this path is needed alongside the pip-installed carla package
CARLA_ROOT = os.environ.get("CARLA_ROOT") or (
    r"C:\CARLA_0.9.15\WindowsNoEditor" if os.name == "nt" else "/home/ailab/carla/CARLA_0.9.15"
)
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
sys.path.append(os.path.join(HERE, "longitudinal_lookup"))

import carla

from functions import PID, ImuAcceleration, LowPassFilter, clipping, reference_preview, speed_reference
from viz_utils import (VIEWS, VideoRecorder, follow_with_spectator, plot_longitudinal_result,
                       print_comfort_report, print_error_summary, run_name,
                       stack_videos_side_by_side)

from longitudinal_lut import LongitudinalLUT
from lookup_controller import LookupController


MAP_NAME = "Town06"
ORIGIN_INDEX = 86


CREEP_SPEED = 0.5            # m/s -- below this the pedal layer coasts rather than brakes; see the
                             # branch in run_trial() for what goes wrong without it

# --warm-start-speed injection: how many ticks to hold the gear by hand, and which gear to force
# for a given injected speed. final_comparison.py 와 같은 값, 같은 이유다 (그쪽 주석 참고):
# set_target_velocity() 는 강체 속도만 쓰고 기어박스는 중립(0)에 둔 채라, 자동변속기가 뒤늦게
# 그 불일치로 물리면서 주입한 속도를 도로 끌어내린다 -- 14 m/s 주입이 스로틀과 무관하게 1 초
# 만에 9.34 m/s 로 무너지는 것이 측정됐다. 몇 틱만 기어를 손으로 잡아주면 사라진다.
WARM_START_INJECT_TICKS = 4
WARM_START_INJECT_GEARS = ((6.0, 2), (10.0, 3), (14.0, 4), (1e9, 5))   # (speed below, gear)


# 웜업 = 속도 주입 -> 고정 하한 대기 -> 수렴 게이트. 세 단계 모두 이유가 측정으로 붙어 있다.
#
# (1) 왜 주입인가. 정지 출발로 initial_speed 까지 올라오길 기다리는 예전 방식은 전제가 틀렸다.
#     그 구간의 --controller pid 는 오차가 커서 스로틀이 0.02 <-> 0.95 로 포화를 왕복하고, 그
#     포화가 리밋 사이클을 먹여살린다: v_x 가 9.07 <-> 10.58 을 주기 ~1 s 로 돌며 30 s 까지
#     전혀 잦아들지 않는다 (|a_x| 평균 5-10 s 1.70 -> 20-30 s 1.81). 가라앉지 않는 대상에게
#     "가라앉을 때까지"를 물으니 게이트는 사이클이 0 을 스쳐가는 순간을 잡았을 뿐이었고, 점수
#     구간이 매번 진동의 임의 위상에서 시작했다. 구르는 출발이면 오차가 작아 u 가 선형 영역
#     (0.2~0.7) 에 머물고, PID 도 그냥 수렴한다 -- 그래서 게이트가 비로소 의미를 가진다.
#
# (2) 왜 고정 하한이 먼저인가. 게이트는 a_x 를 보는데 ImuAcceleration 은 자기 앞 3 샘플을
#     버리는 동안 a_x 를 0 으로 고정해서 내놓는다 (스폰 직후 가속도계가 -378,000 m/s^2 를
#     찍기 때문). 그 구간을 게이트에 그냥 물리면 |a_x| = 0 이 조건을 즉시 통과해 버려서,
#     주입 직후 첫 틱에 핸드오프된다. 그래서 게이트는 이 하한 뒤에만 본다.
#
# (3) 왜 a_x 까지 보는가. 속도만 맞추는 걸로는 부족하다는 게 측정으로 나왔다. 주입을
#     initial_speed + 1 로 해서 핸드오프 v_x 를 9.79 (목표 10) 까지 끌어올려 봤더니, 그 순간
#     a_x 가 -1.13 m/s^2 였다 -- 얹어준 1 m/s 를 털어내며 감속하는 중. 오차가 작아 제어기는
#     u ~ 0.01 만 내고, 차는 8.10 까지 가라앉았다가 그 오차로 스로틀을 0.86 까지 포화시켜
#     리밋 사이클을 다시 켰다 (PID: speed RMSE 0.393 -> 1.057, jerk 평균 1.30 -> 14.49,
#     Comfortness 0.833 -> 0.000). 그래서 주입은 initial_speed 그대로 두고, v_x 와 a_x 가
#     둘 다 조용해질 때까지 기다린다.
WARM_START_SETTLE_S = 0.25   # s -- 게이트를 보기 전 무조건 버리는 하한 (위 (2))
WARM_START_SPEED_TOL = 0.3   # m/s
WARM_START_ACCEL_TOL = 0.3   # m/s^2
WARM_START_HOLD_TICKS = 5    # 연속으로 이만큼 만족해야 인정 (dt=0.05 기준 0.25 s) -- 한 틱짜리
                             # 판정은 과도응답이 0 을 지나가는 순간도 통과시킨다
WARM_START_TIMEOUT = 15.0    # s -- 수렴하지 않을 때의 안전장치. 발동하면 그 사실을 찍는다


class SpeedMPC:
    """Speed-tracking MPC over a scalar first-order integrator: x_k = v_{x,k}, x_{k+1} = A x_k +
    B a_{x,k}, A=1, B=T. Decision variable is the acceleration sequence itself -- free for the
    first Nc steps of the horizon Np (Nc <= Np) and held at its last value after that.

    Substituting the forward solution gives the condensed prediction X = A_bar x0 + B_bar U, and
    the tracking cost

        J = (X - Xref)' W1 (X - Xref) + U' W2 U + (Phi U - Uprev)' W3 (Phi U - Uprev)

    where Phi is now Nc x Nc (row 0 = [1,0,...,0], row l>=1 = -1 at col l-1 / +1 at col l) and
    Uprev = [a_cmd_prev, 0, ..., 0]', a_cmd_prev the acceleration actually applied last tick
    (self.last_solution[0] going into this solve()) -- so Phi U - Uprev is the true jerk sequence
    [a_0-a_prev, a_1-a_0, ...], anchored to what the vehicle is actually doing right now rather than
    only differencing within this one solve's own planned U (a Phi that only did that never limited
    the *applied* a_cmd's tick-to-tick jump at all, since receding-horizon control only ever
    executes U[0] and resolves fresh next tick -- same gap the lateral MPC's rate constraint had
    before it got the same Uprev treatment). This becomes a box/rate-constrained QP in U alone:

        min 1/2 U' H U + f' U   s.t.  a_min <= U <= a_max,
                                       Uprev - jerk_max*T <= Phi U <= Uprev + jerk_max*T
        H = 2 (B_bar' W1 B_bar + W2 + Phi' W3 Phi)         (constant -- built once)
        f = 2 B_bar' W1 (A_bar x0 - Xref) - 2 Phi' W3 Uprev   (rebuilt every cycle: Uprev changes)

    (the -2 Phi' W3 Uprev term comes from expanding (Phi U - Uprev)' W3 (Phi U - Uprev) = U' Phi' W3
    Phi U - 2 Uprev' W3 Phi U + Uprev' W3 Uprev -- the cross term is linear in U and has to land in
    f, not just the U' Phi' W3 Phi piece in H, or the jerk cost silently stops penalizing relative
    to a_cmd_prev at all.)

    The hard jerk_max constraint reuses the same Phi/Uprev the soft w_j cost above does, stacked as
    extra rows onto the box constraint (A = [I; Phi]) rather than folded into a bound on U itself --
    OSQP's own A need not be the identity, so there's no need to invert Phi to get there.

    W1 = w_v I_Np weights every predicted speed error the same way; W2 = w_a I_Nc penalizes the
    commanded acceleration's own magnitude (comfort/effort); W3 = w_j I_Nc penalizes the jerk
    sequence above -- a jerk-rate cost on the commanded acceleration even though jerk itself is not
    a decision variable here. W2 alone already makes H positive definite (B_bar' W1 B_bar is only
    positive semidefinite), so the QP has a unique solution even with w_j = 0.

    Only the first element of U* is applied each cycle (receding horizon). Because U* already
    lives in acceleration units, that element *is* a_cmd -- no separate integration/anchoring step
    is needed the way a jerk-input MPC would need one. H and A are both constant (neither depends on
    Uprev), so (unlike the lateral MPC, which rebuilds its whole QP every cycle) this one is
    still built once in __init__ -- only q and the rate half of l/u move each solve(), via Uprev.
    """

    def __init__(self, dt, n_p, n_c, w_v, w_a, w_j, a_min=-4.05, a_max=2.4, jerk_max=4.13):
        if not (0 < n_c <= n_p):
            raise ValueError(f"need 0 < n_c <= n_p, got n_c={n_c}, n_p={n_p}")
        if not (a_min < a_max):
            raise ValueError(f"need a_min < a_max, got a_min={a_min}, a_max={a_max}")
        if not (jerk_max > 0):
            raise ValueError(f"need jerk_max > 0, got jerk_max={jerk_max}")

        self.T = dt
        self.n_p = n_p
        self.n_c = n_c
        self.a_min = a_min
        self.a_max = a_max
        self.jerk_max = jerk_max

        self.A = 1.0   # fixed by the problem: x_{k+1} = A x_k + B a_{x,k}
        self.B = dt

        self.A_bar = np.full((n_p, 1), self.A)   # A^i = 1 for every i since A=1
        self.B_bar = self._build_b_bar()
        self.Phi = self._build_phi()

        W1 = w_v * np.eye(n_p)
        W2 = w_a * np.eye(n_c)
        W3 = w_j * np.eye(n_c)   # Phi is now n_c x n_c (see _build_phi), not n_c-1

        self.H = 2.0 * (self.B_bar.T @ W1 @ self.B_bar + W2 + self.Phi.T @ W3 @ self.Phi)
        self.H = 0.5 * (self.H + self.H.T)      # symmetrize against round-off
        self._B_W1 = 2.0 * self.B_bar.T @ W1        # reused every cycle to form f
        self._2PhiT_W3 = 2.0 * self.Phi.T @ W3      # reused every cycle for f's Uprev cross term

        # box (a_min<=U<=a_max) stacked with rate (Uprev-jerk_max*dt <= Phi U <= Uprev+jerk_max*dt)
        # -- no need to invert Phi to fold the rate bound into a bound on U itself, OSQP's own A
        # need not be the identity; stacking Phi in as extra rows and giving it its own l/u block
        # is the direct way. Only the box half is static -- the rate half's l/u shift with Uprev
        # every solve() (see there), so those rows are filled with placeholders here and replaced
        # each cycle via update(l=..., u=...).
        self._A_ineq = sparse.csc_matrix(np.vstack([np.eye(n_c), self.Phi]))
        self._a_l = a_min * np.ones(n_c)
        self._a_u = a_max * np.ones(n_c)

        self._solver = osqp.OSQP()
        self._solver.setup(
            P=sparse.csc_matrix(self.H),
            q=np.zeros(n_c),
            A=self._A_ineq,
            l=np.concatenate([self._a_l, -jerk_max * dt * np.ones(n_c)]),
            u=np.concatenate([self._a_u, jerk_max * dt * np.ones(n_c)]),
            verbose=False,
            polish=False,   # polishing logs to stdout even when verbose is off
        )
        self.last_solution = np.zeros(n_c)
        self.last_status = "unsolved"

    def _build_b_bar(self):
        """Np x Nc. [B_bar]_{i,c} = A^(i-c) B for c < Nc, c <= i; the last column accumulates
        every step the held input still acts over. A=1 so every power of A collapses to 1."""
        B_bar = np.zeros((self.n_p, self.n_c))
        for i in range(1, self.n_p + 1):
            for c in range(1, self.n_c + 1):
                if c < self.n_c:
                    if c <= i:
                        B_bar[i - 1, c - 1] = self.B
                elif i >= self.n_c:
                    B_bar[i - 1, c - 1] = (i - self.n_c + 1) * self.B
        return B_bar

    def _build_phi(self):
        """Nc x Nc first-difference operator: (Phi U)_0 = a_0, (Phi U)_l = a_l - a_{l-1} for l >= 1.
        Row 0 is deliberately just [1,0,...,0] -- solve() subtracts Uprev (a_cmd_prev in slot 0)
        from Phi@U to turn that first row into a_0 - a_cmd_prev, anchoring the jerk cost to the
        acceleration actually applied last tick instead of leaving step 0 unanchored."""
        Phi = np.eye(self.n_c)
        for l in range(1, self.n_c):
            Phi[l, l - 1] = -1.0
        return Phi

    def solve(self, v_x, v_ref_preview):
        """One receding-horizon step. v_ref_preview: length-Np sequence of desired speed at each
        future prediction step (a known/analytic profile is previewed in full; a planner handing
        over a shorter horizon would just repeat its last value to fill the rest). Returns a_cmd,
        the acceleration to hand the pedal layer, clipped to [a_min, a_max]."""
        x_ref = np.asarray(v_ref_preview, dtype=float).reshape(-1, 1)
        # acceleration actually applied last tick (this solve's own U[0] once computed becomes NEXT
        # tick's a_cmd_prev) -- anchors the jerk cost to reality instead of just to this one solve's
        # own internal plan, see the class docstring.
        u_prev = np.zeros((self.n_c, 1))
        u_prev[0, 0] = self.last_solution[0]

        f = self._B_W1 @ (self.A_bar * v_x - x_ref) - self._2PhiT_W3 @ u_prev

        # hard jerk constraint's rate rows shift with Uprev every cycle, same reason its rows
        # in f do -- box (a_min/a_max) rows are static, only concatenated fresh here since OSQP
        # wants one full l/u vector per update(), not a way to patch a sub-block in place.
        rate_l = -self.jerk_max * self.T + u_prev.ravel()
        rate_u = self.jerk_max * self.T + u_prev.ravel()
        l_full = np.concatenate([self._a_l, rate_l])
        u_full = np.concatenate([self._a_u, rate_u])

        self._solver.update(q=f.ravel(), l=l_full, u=u_full)
        result = self._solver.solve()
        self.last_status = result.info.status

        if result.x is None or not np.all(np.isfinite(result.x)):
            # keep driving on the previous plan rather than dropping to zero acceleration
            u = self.last_solution
        else:
            u = result.x
            self.last_solution = u.copy()

        return float(np.clip(u[0], self.a_min, self.a_max))


# ----------------------------------------------------------------------------- controllers
# Common interface: step(ctx) -> pedal command u in [-1, 1] (positive throttle, negative brake).
# ctx (a SimpleNamespace, set fresh by run_trial() each cycle) carries t, v_x, v_ref, a_x
# (filtered), a_x_raw (IMU, for the LUT layer), gear, warmed_up. reset(u) is called once, right
# after the warm-up hand-off, so a controller can drop whatever state it accumulated chasing the
# warm-up setpoint before scoring starts.

class PidController:
    """Speed PID straight to the pedal -- longitudinal_PID.py's own controller, same defaults."""
    label = "PID"

    def __init__(self, args):
        self.args = args
        self.pid = PID(kp=args.pid_kp, ki=args.pid_ki, kd=args.pid_kd, dt=args.dt)
        self.filter = LowPassFilter(tau=args.pid_tau, dt=args.dt, initial=0.0)

    def reset(self, u):
        """출력 저역통과만 지금 적용된 u 로 다시 시드한다. 적분은 일부러 그대로 둔다.

        적분을 지우면 안 되는 이유: 순수 PID 는 정속 스로틀을 적분항이 들고 있다. MpcController
        는 reset() 에서 페달 PID 적분을 털어도 되는데, 그건 LUT 피드포워드가 정속분을 대신 내주기
        때문이다. 여기서 같은 짓을 하면 점수 구간 t=0 에서 스로틀이 0 으로 떨어져 속도가 꺼진다.

        정작 털어야 하는 건 필터 상태다. step() 에서 필터가 PID 와 clipping *사이*에 있는데,
        PID 의 안티와인드업(functions.PID.step)은 필터 이전의 raw 로 적분만 보호하고 필터 상태는
        아무도 지키지 않는다. 정지 출발 구간에서는 e=10 m/s -> kp*e=7 이라 필터 상태가 1 을 한참
        넘겨 올라가고 (alpha = dt/(tau+dt) = 1/3), PID 가 명령을 낮춘 뒤에도 그게 1 밑으로 내려올
        때까지 스로틀이 1.0 에 붙어 있다. 웜업에서 쌓인 그 잔재가 그대로 점수 구간 t=0 에 얹혀서
        PID 만 초반에 오버슈트했다 -- MpcController 는 출력 필터를 initial=u 로 재생성하므로 같은
        잔재가 없어서, 비교 자체도 PID 에게만 불리했다.

        longitudinal_PID.py 는 이 reset 이 없다 (거기도 같은 웜업 게이트를 쓰므로 같은 잔재를
        안고 t=0 을 출발한다). 예전에는 "그쪽과 정확히 맞춘다"는 이유로 여기도 비워뒀지만, 그건
        같은 결함을 두 곳에서 재현한 것이지 맞춘 것이 아니다.
        """
        self.filter = LowPassFilter(tau=self.args.pid_tau, dt=self.args.dt, initial=u)

    def step(self, ctx):
        return clipping(self.filter.step(self.pid.step(ctx.v_ref - ctx.v_x)), 1, -1)


class MpcController:
    """SpeedMPC -> a_cmd -> pedal layer, tracking a_cmd either with the LUT feedforward + PID
    stack (see module docstring) or with the PID-only baseline validate_lut.py's "PID only" trial
    uses (use_lut=False -- LookupController.step() still runs, but its feedforward() call is
    skipped, per its own use_feedforward flag: the same class either way, not a reimplementation).
    """

    def __init__(self, args, use_lut):
        self.args = args
        # "LUT", not "FF", everywhere: the feedforward term IS the longitudinal lookup table, so
        # the label, the --controller key and the flag all name the thing rather than its role.
        self.label = "MPC+LUT+PID" if use_lut else "MPC+PID"
        self.mpc = SpeedMPC(dt=args.dt, n_p=args.n_p, n_c=args.n_c,
                            w_v=args.w_v, w_a=args.w_a, w_j=args.w_j,
                            a_min=args.a_min, a_max=args.a_max)
        # Two DIFFERENT pedal-layer tunings, matching validate_lut.py's two corresponding trials:
        # with the LUT carrying the bulk of the command the PID only has to trim the table's error,
        # so it is tuned soft; without it the same PID has to produce the whole pedal command on its
        # own, which is a different plant to close a loop around and was tuned separately (that is
        # what validate_lut.py's "feedforward + PID" vs "PID only" trials measure). Sharing one
        # gain set between them, as this used to, meant whichever variant was not tuned for it ran
        # on gains that were never validated.
        kp, ki, kd = ((args.kp, args.ki, args.kd) if use_lut
                      else (args.mpc_pid_kp, args.mpc_pid_ki, args.mpc_pid_kd))
        self.pedal_ctrl = LookupController(LongitudinalLUT(args.lut), kp=kp, ki=ki, kd=kd,
                                           dt=args.dt, use_feedforward=use_lut)
        self.u_filter = LowPassFilter(tau=args.u_tau, dt=args.dt, initial=0.0)

    def reset(self, u):
        self.pedal_ctrl.reset()   # drop the warm-up phase's PID integral before scoring starts
        self.u_filter = LowPassFilter(tau=self.args.u_tau, dt=self.args.dt, initial=u)

    def step(self, ctx):
        preview = reference_preview(self.args, ctx.t, self.args.n_p, self.args.dt, ctx.warmed_up)
        ctx.a_cmd = self.mpc.solve(ctx.v_x, preview)   # stashed for the console print line
        u_raw = self.pedal_ctrl.step(ctx.gear, ctx.v_x, ctx.a_cmd, a_meas=ctx.a_x_raw)
        return self.u_filter.step(u_raw)


CONTROLLERS = {
    "mpc+lut+pid": lambda args: MpcController(args, use_lut=True),
    "mpc+pid": lambda args: MpcController(args, use_lut=False),
    "pid": PidController,
}


# ----------------------------------------------------------------------------- one trial

def run_trial(world, origin_transform, blueprint, imu_bp, controller, args, recorder_factory):
    """Spawn one vehicle, drive it under `controller` for the scored profile, tear it down.

    Warm-up is final_comparison.py's: --warm-start-speed injects the profile's own t=0 value at
    spawn (initial_speed, since sin(0) = 0 for sine and the step hasn't happened yet at t=0 for
    step), then a FIXED --warm-start-settle wait is discarded before logging starts. No convergence
    gate and no timeout -- see WARM_START_SETTLE_S for the measurements that killed the gate.
    longitudinal_PID.py still carries the old gate; it was deliberately left alone.
    """
    vehicle = world.spawn_actor(blueprint, origin_transform)

    # a_x/a_y off the IMU -- see stanley_PID.py for why (true body-frame values straight from the
    # sensor, nothing to derive by hand). a_y feeds jerk_total and the Comfortness score below;
    # nothing plots it on its own, since there's no lateral figure in a steer=0 run.
    accel = ImuAcceleration(dt=args.dt)

    # yaw/yaw_rate/a_y/a_x_raw/a_y_raw are not read by the longitudinal figure -- they are here so
    # viz_utils.b2d_comfortness() can score the run, which needs all six of the channels B2D's own
    # metric_info.json carries and returns None if any is missing (which is why a run of this
    # script used to print no Comfortness line at all). Steering is pinned to 0, so the two yaw
    # channels sit at ~0 and the score is decided by the longitudinal ones -- exactly the point of
    # scoring this stack. The RAW accelerations are logged alongside the filtered a_x because the
    # scoring function runs its own Savitzky-Golay pass and prefers them: handing it the already
    # low-passed channel double-smooths and scores the run more kindly than it deserves.
    hist = {"t": [], "v_x": [], "v_des": [], "a_x": [], "a_x_raw": [], "a_y": [], "a_y_raw": [],
            "yaw": [], "yaw_rate": [], "throttle": [], "brake": []}

    warmed_up = False
    log_start_i = 0
    hold = 0            # 웜업 게이트를 연속으로 만족한 틱 수 (WARM_START_HOLD_TICKS 참고)

    imu = None
    recorder = None
    video_meta = None
    try:
        world.tick()

        # spawned after the priming tick above, so the first world.tick() in the loop below
        # produces this sensor's first queued sample -- one put() per get() keeps them in
        # lockstep for the rest of the trial.
        # 구르는 출발. 위의 priming tick 뒤에 주입한다 (서스펜션이 한 스텝 간 뒤, 낙하 도중이
        # 아니라), 스폰이 남긴 회전이 있을 수 있으므로 각속도도 같이 0 으로 눌러준다. 그 다음
        # 몇 틱은 기어를 손으로 잡는다 -- WARM_START_INJECT_GEARS 참고. steer 는 이 스크립트
        # 전체에서 0 이므로 forward vector 방향으로만 주면 된다.
        if args.warm_start_speed > 0:
            v0 = args.warm_start_speed
            gear = next(g for lim, g in WARM_START_INJECT_GEARS if v0 < lim)
            fwd = origin_transform.get_forward_vector()
            vehicle.set_target_velocity(carla.Vector3D(fwd.x * v0, fwd.y * v0, 0.0))
            vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            for _ in range(WARM_START_INJECT_TICKS):
                hold = carla.VehicleControl()
                hold.throttle, hold.steer = 0.5, 0.0
                hold.manual_gear_shift, hold.gear = True, gear
                vehicle.apply_control(hold)
                world.tick()
            release = carla.VehicleControl()
            release.throttle, release.steer = 0.5, 0.0
            release.manual_gear_shift = False
            vehicle.apply_control(release)
            world.tick()
            print(f"  rolling start: injected {v0:.1f} m/s in gear {gear}")

        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        imu.listen(imu_queue.put)
        recorder = recorder_factory(vehicle)

        steps = int((args.duration + WARM_START_TIMEOUT) / args.dt)
        for i in range(steps):
            step_start = time.time()
            world.tick()
            # 10s, not 2s: right after a fresh client.load_world(), the server is still streaming
            # map assets/compiling shaders, so the first several ticks can take much longer than a
            # steady-state ~dt-paced tick. Same margin matters on the SECOND trial onward here,
            # where the previous trial's vehicle/camera teardown overlaps this one's first ticks --
            # a real hang still raises Empty, just with more margin.
            imu_data = imu_queue.get(timeout=10.0)

            transform = vehicle.get_transform()
            yaw = math.radians(transform.rotation.yaw)
            vel_vec = vehicle.get_velocity()
            v_x = vel_vec.x * math.cos(yaw) + vel_vec.y * math.sin(yaw)  # body-frame forward speed

            # same source and units as the lateral stacks (mpc_mpc.py, mpc_mpc_KF.py): the IMU
            # gyroscope's z channel in rad/s, converted once to the deg/s every hist in this repo
            # stores yaw_rate in
            yaw_rate_deg = math.degrees(imu_data.gyroscope.z)

            accel.step(imu_data)
            a_x, a_y, a_x_raw, a_y_raw = accel.a_x, accel.a_y, accel.a_x_raw, accel.a_y_raw


            t = (i - log_start_i) * args.dt
            v_ref = args.initial_speed if not warmed_up else speed_reference(args, t)
            ctx = SimpleNamespace(t=t, v_x=v_x, v_ref=v_ref, a_x=a_x, a_x_raw=a_x_raw,
                                  gear=vehicle.get_control().gear, warmed_up=warmed_up)
            u = controller.step(ctx)

            control = carla.VehicleControl()
            if u >= 0:
                control.throttle, control.brake = u, 0.0
            elif v_x < CREEP_SPEED:
                # Below creep speed a negative u coasts instead of braking, or the car can never
                # restart after a full stop (--profile step with --step-duration). At a standstill
                # the vehicle still rocks a few cm/s, so a_meas alternates about +-0.4 m/s^2 tick to
                # tick; through the pedal PID's derivative term that is d(error)/dt ~ -16 m/s^3, and
                # at kd=0.25 with dt=0.05 the kd term alone is -4.1 -- enough to flip u from the
                # +1.0 the rest of the loop is asking for (ff +0.17, kp*e +1.19, ki*I +0.40) to a
                # clipped -1.0. Measured: u alternating +1.000/-1.000 every tick, and since a brake
                # at standstill stops CARLA's transmission engaging, every throttle tick's creep is
                # killed by the next brake tick. a_cmd sat pinned at +2.40 for 9 s with v_x at 0.
                control.throttle, control.brake = 0.0, 0.0
            else:
                control.throttle, control.brake = 0.0, -u
            control.steer = 0.0
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            if not warmed_up:
                # 하한(고정 대기) 안에서는 게이트를 보지도 않는다 -- 위 (2)
                past_floor = i * args.dt >= args.warm_start_settle
                settled = (past_floor
                           and abs(v_x - args.initial_speed) < WARM_START_SPEED_TOL
                           and abs(a_x) < WARM_START_ACCEL_TOL)
                hold = hold + 1 if settled else 0
                converged = hold >= WARM_START_HOLD_TICKS
                timed_out = i * args.dt >= WARM_START_TIMEOUT
                if not (converged or timed_out):
                    elapsed = time.time() - step_start
                    if elapsed < args.dt / args.times_run:
                        time.sleep(args.dt / args.times_run - elapsed)
                    continue
                warmed_up = True
                log_start_i = i
                controller.reset(u)
                status = (f"converged after {i * args.dt:.2f}s" if converged
                          else f"TIMED OUT after {WARM_START_TIMEOUT:.0f}s (not settled -- the "
                               f"scored window starts mid-transient)")
                print(f"[{controller.label}] Warm-start {status}: v_x={v_x:.2f} m/s, "
                      f"a_x={a_x:.2f} m/s^2 -- logging starts now.")

            # recompute now that log_start_i may have just been updated above -- the t used for
            # the controller's step() earlier in this same iteration was based on the pre-handoff
            # value and would log a stale (much larger) timestamp for this first sample otherwise
            t = (i - log_start_i) * args.dt
            hist["t"].append(t)
            hist["v_x"].append(v_x)
            hist["v_des"].append(v_ref)
            hist["a_x"].append(a_x)
            hist["a_x_raw"].append(a_x_raw)
            hist["a_y"].append(a_y)
            hist["a_y_raw"].append(a_y_raw)
            hist["yaw"].append(math.degrees(yaw))
            hist["yaw_rate"].append(yaw_rate_deg)
            hist["throttle"].append(control.throttle)
            hist["brake"].append(control.brake)

            if i % 5 == 0:
                extra = f"   a_cmd={ctx.a_cmd:+.2f} m/s^2" if hasattr(ctx, "a_cmd") else ""
                print(f"[{controller.label}] t={t:5.1f}s   v_x={v_x:5.2f} m/s   v_ref={v_ref:5.2f} m/s   "
                      f"e_vel={v_ref - v_x:+.2f} m/s{extra}   u={u:+.2f}")

            if t >= args.duration:
                print(f"[{controller.label}] Reached duration ({args.duration:.0f}s).")
                break

            elapsed = time.time() - step_start
            if elapsed < args.dt / args.times_run:
                time.sleep(args.dt / args.times_run - elapsed)
    finally:
        if recorder is not None:
            recorder.close()  # before vehicle.destroy(): the camera is attached to it
            # 합치기(viz_utils.stack_videos_side_by_side)에 필요한 정보. frames 는 실제로 쓰인
            # 프레임 수 -- 두 주행의 길이가 다를 때 짧은 쪽을 얼마나 늘릴지 계산하는 데 쓴다.
            video_meta = {"path": recorder.out_path, "frames": recorder.frames}
        if imu is not None and imu.is_alive:
            imu.stop()
            imu.destroy()
        vehicle.destroy()

    return hist, video_meta


def _comfort_report(args, hist):
    """The opt-in per-segment Comfortness breakdown, for one run."""
    if args.comfort_report or args.comfort_report_failures_only:
        print_comfort_report(hist, failures_only=args.comfort_report_failures_only)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--controller", nargs="+", default=["mpc+lut+pid"],
                        choices=("mpc+lut+pid", "mpc+pid", "pid"),
                        help="which longitudinal controller(s) to run and score -- mpc+lut+pid: "
                             "SpeedMPC -> LUT feedforward + PID pedal layer; mpc+pid: SpeedMPC -> "
                             "PID-only pedal layer, no LUT (validate_lut.py's 'PID only' baseline); "
                             "pid: plain speed PID straight to the pedal, no MPC at all. Pass more "
                             "than one to overlay them on one comparison figure")
    parser.add_argument("--initial-speed", type=float, default=10.0,
                        help="m/s; starting speed, also the sine profile's midline and the step "
                             "profile's pre-step level")
    parser.add_argument("--warm-start-speed", type=float, default=None,
                        help="speed (m/s) injected at spawn so the run starts rolling. Default: "
                             "--initial-speed. Aiming it HIGH to compensate the sag before "
                             "hand-off is a trap -- see WARM_START_SETTLE_S (3); the convergence "
                             "gate is what puts t=0 on the profile. Pass 0 to launch from rest "
                             "instead. The matching gear is "
                             "forced for a few ticks with it (see WARM_START_INJECT_GEARS) -- "
                             "without that the injected speed collapses back to ~9.3 m/s within a "
                             "second no matter the throttle, because set_target_velocity() moves "
                             "the body and leaves the gearbox in neutral")
    parser.add_argument("--warm-start-settle", type=float, default=WARM_START_SETTLE_S,
                        help="seconds discarded before logging starts, to let the un-physical first "
                             "ticks pass (spawn accelerometer spike, suspension). Not a convergence "
                             "gate -- a fixed wait; see WARM_START_SETTLE_S for why the gate went")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    parser.add_argument("--duration", type=float, default=15.0, help="scored run length (s)")
    parser.add_argument("--times-run", type=float, default=5.0, help="how times for simulation running?")

    parser.add_argument("--profile", default="constant", choices=("constant", "sine", "step"),
                        help="speed reference shape: flat initial-speed, a sine wave around it, "
                             "or a step away from it at --step-time, held for --step-duration and "
                             "then released back to --initial-speed. The step window is what makes "
                             "an emergency-stop-and-restart run: --step-size -<initial speed> "
                             "--step-duration <seconds> brakes to a standstill and then demands the "
                             "original speed again in one step (see functions.speed_reference)")
    parser.add_argument("--sine-amplitude", type=float, default=1, help="sine profile peak deviation (m/s)")
    parser.add_argument("--sine-period", type=float, default=5.0, help="sine profile period (s)")
    parser.add_argument("--step-size", type=float, default=-10.0,
                        help="step profile: m/s added to initial-speed after --step-time (negative = "
                             "a deceleration step). The result is clamped at 0, so anything "
                             "<= -(initial speed) is a full stop rather than a negative reference")
    parser.add_argument("--step-time", type=float, default=3.0,
                        help="step profile: when the step happens, seconds into the scored run")
    parser.add_argument("--step-duration", type=float, default=3,
                        help="step profile: how long the stepped speed is held (s) before the "
                             "reference returns to --initial-speed. Unset (the default) holds it "
                             "for the rest of the run -- the permanent step this script's step "
                             "profile has always been, so leaving it off changes nothing. Set it "
                             "and BOTH edges become steps, i.e. a hard decel followed by a hard "
                             "re-accel; how hard each one gets is bounded by --a-min/--a-max, not "
                             "by this profile. Give --duration room for the second edge: the run "
                             "ends at --duration, so a window that closes at or after "
                             "--step-time + --step-duration never shows the re-accel. "
                             "mpc_mpc_KF.py/mpc_mpc_kinematic.py/final_comparison.py default this "
                             "to 5 instead; match them explicitly if you want the same run here")

    # ---- MPC ---- #
    mpc = parser.add_argument_group("--controller mpc+lut+pid / mpc+pid")
    mpc.add_argument("--np", dest="n_p", type=int, default=40, help="prediction horizon (steps)")
    mpc.add_argument("--nc", dest="n_c", type=int, default=5, help="control horizon (steps, <= --np)")
    mpc.add_argument("--w-v", type=float, default=150, help="speed-tracking weight")
    mpc.add_argument("--w-a", type=float, default=15, help="commanded-acceleration magnitude weight")
    mpc.add_argument("--w-j", type=float, default=30, help="commanded-acceleration rate (jerk) weight")
    mpc.add_argument("--a-min", type=float, default=-4.05, help="hard lower bound on a_cmd (m/s^2)")
    mpc.add_argument("--a-max", type=float, default=2.4, help="hard upper bound on a_cmd (m/s^2)")
    mpc.add_argument("--lut", default=os.path.join(HERE, "longitudinal_lookup", "longitudinal_lut.npz"))
    # Pedal layer: validate_lut.py's tuned pedal-layer value -- all three scripts run the identical stack (LookupController = LUT feedforward + PID on acceleration error, then the u_tau low-pass), so the gains it was tuned against carry over unchanged.
    mpc.add_argument("--kp", type=float, default=0.6, help="accel-tracking PID proportional gain")
    mpc.add_argument("--ki", type=float, default=0.05, help="accel-tracking PID integral gain")
    mpc.add_argument("--kd", type=float, default=0.25, help="accel-tracking PID derivative gain")
    # "mpc+pid" only -- validate_lut.py's "PID only" trial gains (see MpcController.__init__ for
    # why the no-LUT variant does not share the --kp/--ki/--kd above).
    #
    # NOT --pid-kp/--pid-ki/--pid-kd: those already belong to the standalone "--controller pid"
    # group below and feed PidController, whose PID closes on SPEED error. This one closes on
    # ACCELERATION error. Different quantity, different units, gains that are not interchangeable --
    # sharing the flag names would have silently handed each controller the other's tuning.
    mpc.add_argument("--mpc-pid-kp", type=float, default=0.5,
                     help="mpc+pid (no LUT): accel-tracking PID proportional gain")
    mpc.add_argument("--mpc-pid-ki", type=float, default=0.1,
                     help="mpc+pid (no LUT): accel-tracking PID integral gain")
    mpc.add_argument("--mpc-pid-kd", type=float, default=0.1,
                     help="mpc+pid (no LUT): accel-tracking PID derivative gain")
    mpc.add_argument("--u-tau", type=float, default=0.6,
                     help="low-pass filter time constant on the pedal command u, before it's applied (s). "
                          "The earlier 0.1 default came from sine sweeps reading a slower actuator as "
                          "*more* jerk (0.2 measured ~1.5x the jerk of 0.02 at equal w_v/w_a/w_j); that "
                          "was measured against the old kp/kd, and validate_lut.py retuned the pedal "
                          "loop as a whole, so the two are not comparable term by term")

    # ---- PID ---- #
    pid = parser.add_argument_group("--controller pid")
    pid.add_argument("--pid-kp", type=float, default=0.6)
    pid.add_argument("--pid-ki", type=float, default=0.2)
    pid.add_argument("--pid-kd", type=float, default=0.3)
    pid.add_argument("--pid-tau", type=float, default=0.05, help="output low-pass time constant (s)")

    parser.add_argument("--comfort-report", action="store_true",
                        help="after the error summary, print the per-segment breakdown behind the "
                             "Comfortness score: one row per scored 1-second segment, marking which "
                             "of the six channels left its band. Use it when the score disagrees "
                             "with the figure -- the score reads the RAW acceleration channels "
                             "while the figure plots the low-passed ones, and one bad sample fails "
                             "a whole segment no matter how far out it went")
    parser.add_argument("--comfort-report-failures-only", action="store_true",
                        help="--comfort-report, but only the segments that failed")

    # ---- plot ---- #
    parser.add_argument("--plot-dir", default=os.path.join(HERE, "plots"),
                         help="directory to save the end-of-run result figures into")
    parser.add_argument("--save-plot", action="store_true",
                        help="draw and save the end-of-run result figures (off by default)")

    # ---- video ---- #
    parser.add_argument("--record", nargs="?", const="auto", default="",
                        help="record the drive to an mp4; bare flag auto-names it under --video-dir. "
                             "With several --controller entries each trial is recorded separately "
                             "and then stitched left-to-right into one mp4 in --controller order, "
                             "and the per-trial files are deleted. The trials run one after another, "
                             "so the panels are each run's own t=0 played together, not the same "
                             "instant; a run that finishes early holds its last frame")
    parser.add_argument("--video-dir", default=os.path.join(HERE, "videos"),
                        help="where auto-named recordings go")
    parser.add_argument("--record-view", default="chase", choices=sorted(VIEWS),
                        help="camera mount for the recording")
    parser.add_argument("--record-res", default="1280x720", help="recording resolution, WxH")
    args = parser.parse_args()
    if args.warm_start_speed is None:
        args.warm_start_speed = args.initial_speed

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    current_map = world.get_map().name.split("/")[-1]
    if current_map != MAP_NAME:
        print(f"Loading {MAP_NAME} (current map: {current_map})...")
        world = client.load_world(MAP_NAME)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    # A map that was just (re)loaded above is still settling server-side for a few frames --
    # ticking synchronously against it too soon can leave a freshly-spawned sensor's first
    # callback missing, which showed up as queue.Empty on run_trial()'s very first IMU read.
    # A handful of throwaway ticks before anything is spawned lets that settle once, here,
    # instead of every trial having to guard against it.
    for _ in range(10):
        world.tick()

    origin_transform = world.get_map().get_spawn_points()[ORIGIN_INDEX]

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.get_location().distance(origin_transform.location) < 5.0:
            actor.destroy()

    blueprint = world.get_blueprint_library().filter("vehicle.lincoln.mkz_2020")[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")

    keys = list(dict.fromkeys(args.controller))   # de-dupe, keep the order given on the CLI

    def recorder_factory(key, n_trials):
        if not args.record:
            return lambda vehicle: None
        suffix = key.replace("+", "-") if n_trials > 1 else ""

        def make(vehicle):
            if args.record == "auto" or n_trials > 1:
                path = os.path.join(args.video_dir, run_name(suffix) + ".mp4")
            else:
                path = args.record
            rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))
            return VideoRecorder(world, vehicle, path, fps=1.0 / args.dt, width=rec_w, height=rec_h,
                                 view=args.record_view)
        return make

    results = {}
    videos = []
    try:
        for key in keys:
            controller = CONTROLLERS[key](args)
            print(f"\n=== running {controller.label} ===")
            hist, video_meta = run_trial(world, origin_transform, blueprint, imu_bp,
                                         controller, args,
                                         recorder_factory(key, len(keys)))
            results[controller.label] = hist
            # --controller 에 적은 순서 그대로 쌓는다 -- 그 순서가 곧 합친 영상의 좌->우 배치다.
            if video_meta is not None:
                videos.append((controller.label, video_meta["path"], video_meta["frames"]))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("Cleaned up: world settings restored.")

    # 제어기를 둘 이상 녹화했으면 좌우로 합쳐 하나만 남긴다. 순서는 --controller 인자 순서.
    # 제어기들은 순차로 주행하므로 동시 녹화가 아니라 "각자의 t=0 부터"를 나란히 놓은 것이고,
    # 길이가 다르면 짧은 쪽 마지막 프레임을 늘려 끝을 맞춘다 (그쪽 docstring 참고).
    if len(videos) >= 2:
        if args.record == "auto":
            combined = os.path.join(args.video_dir, run_name("combined") + ".mp4")
        else:
            base, ext = os.path.splitext(args.record)
            combined = f"{base}_combined{ext}"
        stack_videos_side_by_side(videos, combined, fps=1.0 / args.dt)

    have_data = all(len(hist["t"]) > 1 for hist in results.values())
    if not args.save_plot or not have_data:
        for label, hist in results.items():
            if len(results) > 1:
                print(f"\n### {label} ###")
            print_error_summary(hist, args.initial_speed, lateral=False)  # plot_longitudinal_result prints it otherwise
            _comfort_report(args, hist)
    else:
        data = results if len(results) > 1 else next(iter(results.values()))
        try:
            # lateral=False: steer is pinned at 0 here, so the yaw/a_y rows are noise floor. They
            # are still LOGGED (b2d_comfortness() needs all six channels) -- just not printed.
            # label is just the profile -- the controller names are in the figure's own bottom
            # legend, so repeating them in the suptitle only made it wrap on a 3-controller run
            plot_longitudinal_result(data, args.initial_speed, args.plot_dir, lateral=False,
                                     label=args.profile)
            # plot_longitudinal_result() already printed each run's error summary; the comfort
            # breakdown is opt-in and goes after it, per run, in the same order
            for label, hist in results.items():
                if args.comfort_report or args.comfort_report_failures_only:
                    if len(results) > 1:
                        print(f"\n### {label} ###")
                    _comfort_report(args, hist)
        except Exception as exc:
            print(f"Plotting failed: {exc}")
            for label, hist in results.items():
                print_error_summary(hist, args.initial_speed, lateral=False)
                _comfort_report(args, hist)


if __name__ == "__main__":
    main()
