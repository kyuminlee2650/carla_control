"""Verbatim port of the PID baseline controller from Thinklab-SJTU/Bench2DriveZoo
(team_code/pid_controller.py, uniad/vad branch) -- the actual controller VAD's own Bench2Drive
agent (team_code/vad_b2d_agent.py) drives with, fetched and confirmed against the real source this
session so the comparison in mpc_mpc1.py is against the real baseline, not an approximation.

Only the control law itself is ported (gains, PID math, control_pid()'s steer/throttle/brake logic)
-- unchanged from upstream, including quirks that look unusual out of context: the derivative term
is *not* divided by dt (the gains below were tuned against that, at whatever tick rate the upstream
agent runs), and the integral term is a windowed *mean*, not an accumulated sum (self-bounding,
no separate anti-windup needed). mpc_mpc1.py supplies the two inputs control_pid() expects
(`waypoints`, `target`) by sampling this repo's own PathSpline instead of a learned trajectory
predictor and CARLA's discrete route commands, neither of which this repo has -- see
mpc_mpc1.VadPidController for that adapter.
"""

from collections import deque

import numpy as np


class PID:
    """K_P*error + K_I*mean(window) + K_D*(window[-1]-window[-2]) -- named PID here (not the
    functions.PID this repo already has elsewhere) since the two are unrelated implementations
    with different constructor signatures; keeping them distinct avoids conflating a controller
    that's meant to reproduce an external baseline exactly with this repo's own PID class."""

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class PIDController:
    """Same defaults as upstream: turn_controller (steering) KP=0.75/KI=0.75/KD=0.3, n=40;
    speed_controller (throttle) KP=5.0/KI=0.5/KD=1.0, n=40."""

    def __init__(self, turn_KP=0.75, turn_KI=0.75, turn_KD=0.3, turn_n=40, speed_KP=5.0,
                speed_KI=0.5, speed_KD=1.0, speed_n=40, max_throttle=0.75, brake_speed=0.4,
                brake_ratio=1.1, clip_delta=0.25, aim_dist=4.0, angle_thresh=0.3, dist_thresh=10):
        self.turn_controller = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh

    def control_pid(self, waypoints, speed, target):
        """Predicts vehicle control with a PID controller.

        waypoints: array of 2D points in the ego/LIDAR frame, **[lateral, forward]** axis order
            (not this repo's usual [forward, lateral]) -- VAD's own predicted future-trajectory
            output, upstream. target: one farther point, same frame/axis order -- VAD's coarse
            route-command waypoint, upstream. speed: current speed (scalar, m/s).
        """
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            desired_speed += np.linalg.norm(
                    waypoints[i + 1] - waypoints[i]) * 2.0 / num_pairs

            norm = np.linalg.norm((waypoints[i + 1] + waypoints[i]) / 2.0)
            if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (np.abs(angle_target - angle_last) > self.angle_thresh
                                                   and target[1] < self.dist_thresh)
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            "speed": float(speed), "steer": float(steer), "throttle": float(throttle),
            "brake": float(brake), "aim": tuple(aim), "target": tuple(target),
            "desired_speed": float(desired_speed), "angle": float(angle),
            "angle_last": float(angle_last), "angle_target": float(angle_target),
            "angle_final": float(angle_final), "delta": float(delta),
        }
        return steer, throttle, brake, metadata
