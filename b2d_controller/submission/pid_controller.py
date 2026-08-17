r"""Drop-in replacement for Bench2DriveZoo/team_code/pid_controller.py that runs MpcKfController.

WHY THIS FILE EXISTS, AND WHY IT IS SHAPED LIKE A PID CONTROLLER
---------------------------------------------------------------
Bench2DriveBoard scores a submitted .sif, but the submission does NOT own the agent. Verified
against the live cluster files, not the repo copy:

  * bench2drive_runner.py bind-mounts the canonical trees READ-ONLY over whatever the image ships:
        --bind <host>/sources/Bench2Drive     : /opt/bench2drive     : ro
        --bind <host>/sources/Bench2DriveZoo  : /opt/Bench2DriveZoo  : ro
        --bind <host>/jobs/b2d_submit_shard.sh: /workspace/b2d_submit.sh : ro
  * that shard script hard-codes  TEAM_AGENT="${B2D_ZOO_ROOT}/team_code/vad_b2d_agent.py"
  * the dashboard's own .env.local overrides only ROUTES_FILE / SPLIT / EXPECTED -- it does not
    point SHARD_SCRIPT or ZOO_ROOT anywhere else.

So team_code/vad_b2d_agent.py is always the stock agent, and dropping our own agent into the image
would simply be shadowed by the read-only mount. What the image DOES own is /opt/vad-deps, which
the shard script puts FIRST on PYTHONPATH. Bench2DriveZoo/ and Bench2DriveZoo/team_code/ have no
__init__.py, i.e. they are namespace packages, so a module placed at

    /opt/vad-deps/Bench2DriveZoo/team_code/pid_controller.py

shadows THAT ONE MODULE while planner.py, adzoo, mmcv and everything else still resolve from the
bound canonical tree (verified by import test: team_code.__path__ merges both directories, our
pid_controller wins, planner still comes from the canonical tree). The stock agent's line 17 is
`from Bench2DriveZoo.team_code.pid_controller import PIDController`, so this is the single
injection point the submission contract leaves open. Hence the class keeps the name PIDController
and the signature control_pid(waypoints, speed, target); only the body is different.

THREE CONSEQUENCES OF NOT OWNING THE AGENT
------------------------------------------
1. IMU. control_pid()'s three arguments do not include the IMU, and MpcKfController needs yaw rate
   and lateral acceleration. They ARE available: the stock agent computes `tick_data` in the same
   frame that calls us (its line 402), and tick_data['angular_velocity']/['acceleration'] are the
   raw sensor.other.imu channels. This file reads them out of the caller's frame rather than
   re-deriving them from hero_actor.get_angular_velocity()/get_acceleration(), because those are
   DIFFERENT SIGNALS -- degrees/s and world-frame respectively -- and the Kalman filter is tuned on
   the IMU ones. Frame introspection is ugly; using a signal the filter was not tuned for would be
   worse, and silently so.

2. THROTTLE CEILING. The stock agent clips `np.clip(throttle_traj, 0, 0.75)` after we return, and
   that line is in the read-only tree. max_throttle is therefore set to 0.75 here, NOT the 1.0 the
   standalone deployment uses: matching the real ceiling means control_mpc()'s own clip is the
   binding one, instead of the pedal layer emitting up to 1.0 and having a quarter of it silently
   removed downstream. (The pedal loop's anti-windup still guards at +-1 -- see the note at
   mpc_kf_controller.control_mpc()'s throttle line. Pushing the bound into LookupController/PID
   would need editing a verbatim copy of carla_control's PID class, which is a separate decision.)

3. BRAKE DEADBAND. The stock agent also applies `if brake_traj < 0.05: brake_traj = 0.0`. Small
   brake commands from the pedal layer will be dropped. Nothing to do about it here; noted so the
   behaviour is not mistaken for a controller bug when reading logs.

FAILURE POLICY: loud. If the IMU cannot be recovered, or the hero actor is missing after the run
has started, this raises. It must never quietly fall back to the stock PID -- that produces a
plausible leaderboard score for a controller that was never actually exercised.
"""

import importlib.util
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name):
    """Import a module from THIS directory by absolute path.

    Deliberately not `from Bench2DriveZoo.team_code.mpc_kf_controller import ...`: that would go
    back through the merged namespace-package path and could bind to a different copy if the
    canonical tree ever grows a file of the same name. The controller must be the one shipped
    beside this shim, and nothing else.
    """
    path = os.path.join(_HERE, name + ".py")
    spec = importlib.util.spec_from_file_location("b2dsub_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_mpc = _load_sibling("mpc_kf_controller")
MpcKfController = _mpc.MpcKfController
ImuAcceleration = _mpc.ImuAcceleration
DT = _mpc.DT

# The ceiling the stock agent will apply to whatever we return (see docstring point 2).
AGENT_THROTTLE_CEILING = 0.75

print("[mpc-kf-shim] active: Bench2DriveZoo.team_code.pid_controller -> MpcKfController "
      "(loaded from %s)" % _HERE, flush=True)


def _imu_from_caller():
    """Pull the stock agent's own `tick_data` out of the calling frame.

    The stock agent's run_step() does

        tick_data = self.tick(input_data)
        ...
        steer, throttle, brake, meta = self.pidcontroller.control_pid(
            out_truck, tick_data['speed'], local_command_xy)

    so one frame up from control_pid() there is a local named `tick_data` holding
    input_data['IMU'][1] split into 'acceleration' (accelerometer x/y/z) and 'angular_velocity'
    (gyroscope x/y/z). Several frames are searched rather than exactly one, so an added wrapper in
    a future agent revision does not silently break this.
    """
    frame = sys._getframe(1)
    depth = 0
    while frame is not None and depth < 6:
        tick = frame.f_locals.get("tick_data")
        if isinstance(tick, dict) and "acceleration" in tick and "angular_velocity" in tick:
            return tick
        frame = frame.f_back
        depth += 1
    raise RuntimeError(
        "[mpc-kf-shim] could not find the caller's tick_data (IMU). The agent calling "
        "control_pid() is not the expected Bench2Drive vad_b2d_agent, or its run_step() no longer "
        "keeps tick_data in scope. Refusing to run on substituted inputs.")


class PIDController(object):
    """Stock PIDController's interface, MpcKfController's behaviour.

    Every stock keyword is accepted so that an agent revision passing them explicitly still
    constructs. They are ignored on purpose -- this controller has its own tuning, and silently
    honouring e.g. speed_KP here would suggest a PID loop that does not exist. max_throttle is the
    one exception: it is genuinely used, clamped to the ceiling the agent will impose anyway.
    """

    def __init__(self, turn_KP=0.75, turn_KI=0.75, turn_KD=0.3, turn_n=40,
                 speed_KP=5.0, speed_KI=0.5, speed_KD=1.0, speed_n=40,
                 max_throttle=0.75, brake_speed=0.4, brake_ratio=1.1, clip_delta=0.25,
                 aim_dist=4.0, angle_thresh=0.3, dist_thresh=10):
        self.max_throttle = min(float(max_throttle), AGENT_THROTTLE_CEILING)
        self.controller = MpcKfController(max_throttle=self.max_throttle)
        # Same preprocessing every carla_control driving script applies before the IMU reaches the
        # filter/pedal layer: drop the first ticks outright (the accelerometer reads about
        # -378,000 m/s^2 on the spawn tick) and low-pass a_y; a_x stays raw so the LUT+PID pedal
        # layer compares a_cmd against an unlagged measurement.
        self.accel = ImuAcceleration(dt=DT)
        self._hero = None
        self._physics = None
        self._logged = False

    # -- hero actor ---------------------------------------------------------------------------
    def _hero_state(self):
        """(gear, physics). Both come from the hero actor, fetched lazily.

        gear is read fresh every tick because it actually changes; physics (steering curve, wheel
        geometry) is static for a fixed blueprint and is cached. Before the actor is registered
        with CarlaDataProvider, control_mpc() falls back to its flat steer scale and treats gear 0
        as "guess from speed", which is the same warm-up path the standalone deployment takes.
        """
        if self._hero is None:
            try:
                from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            except ImportError as exc:                       # pragma: no cover - env dependent
                raise RuntimeError(
                    "[mpc-kf-shim] srunner is not importable, so the hero actor's gear and "
                    "physics cannot be read: %s" % exc)
            self._hero = CarlaDataProvider.get_hero_actor()
            if self._hero is not None:
                self._physics = self._hero.get_physics_control()
        if self._hero is None:
            return 0, None
        return self._hero.get_control().gear, self._physics

    # -- the stock entry point ----------------------------------------------------------------
    def control_pid(self, waypoints, speed, target):
        """waypoints: VAD's out_truck, (N, 2) in its own [lateral, forward] ego frame.
        speed: measured speed (m/s). target: the route-command point -- UNUSED, deliberately.

        MpcKfController fits its path from VAD's own waypoints only; mixing the route-command
        point into that fit is what its module docstring point 1 argues against, so `target` is
        accepted to keep the signature and then dropped.
        """
        tick = _imu_from_caller()
        self.accel.step_xy(float(tick["acceleration"][0]), float(tick["acceleration"][1]))
        r_meas = float(tick["angular_velocity"][2])   # imu gyroscope.z, rad/s, as-is
        ay_meas = self.accel.a_y                      # low-pass filtered (Kalman measurement)
        a_meas = self.accel.a_x_raw                   # raw, settle-gated (pedal layer)

        gear, physics = self._hero_state()
        steer, throttle, brake, metadata = self.controller.control_mpc(
            np.asarray(waypoints, dtype=float), float(speed),
            r_meas, ay_meas, gear, a_meas, physics=physics)

        if not self._logged:
            print("[mpc-kf-shim] first control tick: speed=%.2f gear=%s physics=%s "
                  "steer=%+.4f throttle=%.3f brake=%.3f lat_qp=%s"
                  % (speed, gear, physics is not None, steer, throttle, brake,
                     metadata.get("kf_status")), flush=True)
            self._logged = True

        # json.dump()-safe: the agent copies this dict into pid_metadata, and numpy scalars are
        # not serializable if a run is ever scored with B2D_SAVE_IMAGES=1.
        metadata = {k: (v.item() if isinstance(v, np.generic) else v)
                    for k, v in metadata.items()}
        metadata["controller"] = "MpcKfController"
        return float(steer), float(throttle), float(brake), metadata
