"""Pure CARLA Python API test: PID longitudinal cruise control (no steering yet).

Verifies option-1 control loop (client-side sync tick -> read state -> solve -> apply)
before building the MPC path-tracking controller on top of it.

Usage:
    cd ~/carla_control
    python3 pid_cruise_control.py --target-speed 10 --duration 20
"""
import argparse
import math
import time

import carla


class PID:
    def __init__(self, kp, ki, kd, dt, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.out_min, self.out_max = out_min, out_max
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error):
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        clamped = max(self.out_min, min(self.out_max, out))
        if clamped != out:  # anti-windup: don't accumulate integral while saturated
            self._integral -= error * self.dt
        return clamped


def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def follow_with_spectator(world, vehicle, back=8.0, up=4.0, pitch=-15.0):
    """Move the spectator to a 3rd-person chase view behind the vehicle."""
    transform = vehicle.get_transform()
    yaw = transform.rotation.yaw
    offset = carla.Location(
        x=-back * math.cos(math.radians(yaw)),
        y=-back * math.sin(math.radians(yaw)),
        z=up,
    )
    spectator_transform = carla.Transform(
        transform.location + offset,
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )
    world.get_spectator().set_transform(spectator_transform)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--target-speed", type=float, default=30.0, help="km/h")
    parser.add_argument("--duration", type=float, default=20.0, help="seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="fixed sim step (s)")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    blueprint = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(blueprint, spawn_point)

    pid = PID(kp=0.35, ki=0.15, kd=0.05, dt=args.dt)

    try:
        world.tick()
        steps = int(args.duration / args.dt)
        for i in range(steps):
            world.tick()
            current_speed = speed_kmh(vehicle)
            error = args.target_speed - current_speed
            control_value = pid.step(error)

            control = carla.VehicleControl()
            if control_value >= 0:
                control.throttle = control_value
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = -control_value
            control.steer = math.sin(0.1*i)
            vehicle.apply_control(control)
            follow_with_spectator(world, vehicle)

            if i % max(1, int(0.5 / args.dt)) == 0:
                print(f"t={i*args.dt:5.1f}s  speed={current_speed:5.1f} km/h  "
                      f"target={args.target_speed:5.1f}  error={error:6.2f}  "
                      f"throttle={control.throttle:.2f}  brake={control.brake:.2f}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        vehicle.destroy()
        world.apply_settings(original_settings)
        print("Cleaned up: vehicle destroyed, world settings restored.")


if __name__ == "__main__":
    main()
