r"""Yaw moment of inertia (Iz), measured tire-free by an airborne angular impulse.

Drop the vehicle far above the map. With no wheel in contact the only external force is gravity,
which acts at the centre of mass and so exerts no yaw moment whatsoever. Hit it with a known
angular impulse about z and only the rigid-body relation is left:

    Iz = J_z / delta_r

No Cf, no Cr, no slip angles, no bicycle model, no steering at all -- this measurement never
touches the tires. CARLA's impulse units are not reliably documented, so the script calibrates
instead of assuming: phase 1 applies a pure linear impulse and recovers
the mass from P/dv, which pins the units to SI if it returns the blueprint's mass. That does NOT
carry over to the angular API, though -- add_angular_impulse lands about six orders of magnitude
off a kg*m^2*rad/s reading. Phase 2 applies several arbitrary angular-impulse magnitudes J
(--impulses), one per trial, and simply reads back whatever yaw rate r0 each one happened to
produce -- nothing is chosen or back-solved from a target rate. Each (J, r0) pair gives its own
Iz = J/r0 under all four unit conventions UE4's radian/degree variants crossed with its
centimetre-scaled internals allow; only kg*cm^2*deg/s lands in the plausible range for a passenger
car (the others are off by factors of 57.3 or 10,000): see ANGULAR_IMPULSE_UNITS below. The script
uses that conversion directly for every trial's Iz, while still printing the other three candidates
each run so the assumption stays auditable rather than silently baked in.

Two systematic effects are handled rather than hoped away: UE4's angular damping (the spin decays,
so r(t) = r0*exp(-t/tau) is fitted and r0 extrapolated back to the impulse), and cross-axis
coupling (roll/pitch rates are logged, and a warning fires if a pure z impulse is not producing a
pure z rotation). Sweeping several impulse magnitudes is the linearity check -- Iz is a rigid-body
property and cannot depend on how hard it was hit, so the per-trial Iz column in the printed table
should read flat across the whole sweep.

--record (needs 2+ --impulses) also produces one combined mp4: every trial's clip, played back to
back in the order applied, each labelled on-screen with the J that produced it (phase 3). The
per-trial clips are deleted once the combined one exists.

--report-only skips the simulator entirely and just reprints the table from the saved JSON.

Usage (Ubuntu):
    cd ~/carla_control
    python3 lateral_parameter/estimate_yaw_inertia.py
    python3 lateral_parameter/estimate_yaw_inertia.py --record
    python3 lateral_parameter/estimate_yaw_inertia.py --report-only

Usage (Windows):
    cd C:\Users\mumu2\carla_control
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py --record
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py --impulses 1e8,3e8,6e8,1e9,2e9 --record
    .venv\Scripts\python.exe lateral_parameter\estimate_yaw_inertia.py --report-only
"""

import argparse
import json
import math
import os
import queue
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# functions.py, viz_utils.py live one level up, alongside the other controllers
sys.path.append(os.path.dirname(HERE))

# functions.py resolves CARLA_ROOT and puts the simulator's PythonAPI on sys.path as a module-level
# side effect, so it has to be imported before carla's navigation helpers are reachable -- even
# though nothing here calls into it directly.
import functions  # noqa: F401

import carla

from viz_utils import VIEWS, VideoRecorder, concat_videos_sequential, run_name

MAP_NAME = "Town06"
ORIGIN_INDEX = 86

# Empirically established (see the module docstring): sweeping impulse magnitude and checking all
# four unit conventions UE4's radian/degree variants crossed with its centimetre-scaled internals
# allow, only this one lands in the plausible range for a passenger car's yaw inertia.
# method_impulse() uses it directly for every trial's Iz; the other three candidates are still
# computed and printed each run as an audit trail, not re-derived from scratch.
ANGULAR_IMPULSE_UNITS = "kg*cm^2*deg/s"


def setup_world(args):
    """Connect, load the map if needed, switch to synchronous mode. Returns (world, old_settings)."""
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
    return world, original_settings


def spawn_airborne(world, blueprint, base_transform, height, imu_bp, settle_ticks=10):
    """Spawn `height` metres up and let the physics state settle.

    The settle ticks matter: a freshly spawned actor reports zero velocity for a tick or two while
    UE4 registers the body, and an impulse applied inside that window hits a body that is not yet
    integrating.

    Cleans up its own actors if anything here raises (e.g. the settle loop's IMU wait timing out
    right after a fresh map load, while the server is still streaming assets) -- without this, a
    failure here leaks the vehicle/IMU, and the very next spawn_actor() at the same base_transform
    then fails with "collision at spawn position", turning one transient timeout into every
    subsequent trial failing too.
    """
    transform = carla.Transform(
        carla.Location(base_transform.location.x, base_transform.location.y,
                       base_transform.location.z + height),
        base_transform.rotation,
    )
    vehicle = world.spawn_actor(blueprint, transform)
    try:
        vehicle.set_simulate_physics(True)
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        # hands off the controls -- a brake command would spin up wheel physics we do not want
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0, hand_brake=False))

        imu_queue = queue.Queue()
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    except Exception:
        vehicle.destroy()
        raise
    try:
        imu.listen(imu_queue.put)
        for _ in range(settle_ticks):
            world.tick()
            imu_queue.get(timeout=2.0)
    except Exception:
        imu.stop(); imu.destroy(); vehicle.destroy()
        raise
    return vehicle, imu, imu_queue


def measure_mass(world, blueprint, base_transform, imu_bp, impulse_n_s, args):
    """Phase 1: pure linear impulse at the CoM in world x. Returns (mass CARLA behaved as, dv_x).

    Gravity is along z and cannot touch v_x, so the x channel isolates the impulse without needing
    to model the fall at all.
    """
    vehicle, imu, imu_queue = spawn_airborne(world, blueprint, base_transform, args.height, imu_bp)
    try:
        v_before = vehicle.get_velocity()
        vehicle.add_impulse(carla.Vector3D(impulse_n_s, 0.0, 0.0))
        world.tick()
        imu_queue.get(timeout=2.0)
        dvx = vehicle.get_velocity().x - v_before.x
        return (impulse_n_s / dvx if abs(dvx) > 1e-6 else float("nan")), dvx
    finally:
        imu.stop(); imu.destroy(); vehicle.destroy()


def fit_decay(t, r):
    """Fit r(t) = r0*exp(-t/tau), returning (r0, tau).

    The impulse lands between two ticks, so the first sample is already one step into the decay;
    extrapolating back to t=0 recovers the step the impulse actually produced rather than what
    survived a tick of damping. Falls back to the first sample if the trace does not decay.
    """
    mask = np.abs(r) > 1e-4
    if mask.sum() < 3:
        return (float(r[0]) if len(r) else float("nan")), float("inf")
    sign = 1.0 if np.mean(r[mask]) > 0 else -1.0
    slope, intercept = np.polyfit(t[mask], np.log(np.abs(r[mask])), 1)
    if slope >= 0:
        return float(r[0]), float("inf")
    return float(sign * math.exp(intercept)), float(-1.0 / slope)


def measure_yaw_impulse(world, blueprint, base_transform, imu_bp, J, args, video_factory=None,
                        label=None):
    """Phase 2: angular impulse about z, then log the spin-down.

    video_factory(vehicle, label) -> VideoRecorder|None, called right after the vehicle is spawned
    airborne, before the angular impulse is applied. If it returns a recorder, this closes it
    once the decay window finishes, before vehicle.destroy() (VideoRecorder's own lifecycle
    requirement -- the camera is attached to the vehicle). None (the default) records nothing.

    label names the clip (e.g. "01_J145799561"). Falls back to the raw J if omitted.
    """
    vehicle, imu, imu_queue = spawn_airborne(world, blueprint, base_transform, args.height, imu_bp)
    recorder = (video_factory(vehicle, label if label is not None else f"J{J:,.0f}")
               if video_factory is not None else None)
    result = None
    try:
        vehicle.add_angular_impulse(carla.Vector3D(0.0, 0.0, J))
        ts, rz, rx, ry = [], [], [], []
        for k in range(int(args.decay_time / args.dt)):
            world.tick()
            imu_data = imu_queue.get(timeout=2.0)
            ts.append((k + 1) * args.dt)
            rz.append(imu_data.gyroscope.z)
            rx.append(imu_data.gyroscope.x)
            ry.append(imu_data.gyroscope.y)
        r0, tau = fit_decay(np.array(ts), np.array(rz))
        result = {"J": J, "t": ts, "r_z": rz, "r_x": rx, "r_y": ry, "r_first": rz[0],
                 "r0": r0, "tau": tau,
                 "cross_axis": float(max(np.max(np.abs(rx)), np.max(np.abs(ry))))}
        return result
    finally:
        if recorder is not None:
            video_path = recorder.close()   # before vehicle.destroy(): the camera is attached to it
            # result is the same dict object the caller is about to receive -- mutating it here
            # (rather than returning something new from finally) still reaches the caller.
            if result is not None:
                result["video_path"] = video_path
                result["video_frames"] = recorder.frames
        imu.stop(); imu.destroy(); vehicle.destroy()


def make_video_factory(world, args):
    """--record -> a measure_yaw_impulse() video_factory(vehicle, label) that opens one
    VideoRecorder per airborne trial, named after its label so every trial gets its own clip
    instead of overwriting each other -- same per-trial auto-naming
    estimate_cornering_stiffness.py uses for its own --record, keyed on label here."""
    if not args.record:
        return None

    rec_w, rec_h = (int(v) for v in args.record_res.lower().split("x"))

    def factory(vehicle, label):
        if args.record == "auto":
            video_path = os.path.join(args.video_dir, run_name(label) + ".mp4")
        else:
            base, ext = os.path.splitext(args.record)
            video_path = f"{base}_{label}{ext}"
        return VideoRecorder(world, vehicle, video_path, fps=1.0 / args.dt,
                             width=rec_w, height=rec_h, view=args.record_view)

    return factory


def print_yaw_impulse_table(trials):
    """Per-trial J (given) / measured r0 / decay tau / Iz (kg*m^2, under the ANGULAR_IMPULSE_UNITS
    convention) / cross-axis leakage. This is the table --report-only reprints from the saved JSON.

    Column order follows the causal chain, not alphabetical convenience: J is what was applied,
    everything to its right is what was measured or derived from that single trial.
    """
    print(f"\n{'#':>3} {'J (given, raw impulse)':>24} {'r0 (measured, deg/s)':>22} "
          f"{'tau (s)':>10} {'Iz (kg*m^2)':>12} {'cross-axis (rad/s)':>19}")
    for i, t in enumerate(trials, start=1):
        tau_str = "inf" if not math.isfinite(t["tau"]) or t["tau"] > 1e6 else f"{t['tau']:.2f}"
        print(f"{i:3d} {t['J']:24,.0f} {math.degrees(t['r0']):22.2f} "
              f"{tau_str:>10} {t['Iz']:12,.1f} {t['cross_axis']:19.4f}")


def method_impulse(world, args):
    """Tire-free Iz from an airborne angular impulse. Returns the dict written to --out."""
    blueprint = world.get_blueprint_library().filter(args.vehicle)[0]
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")
    base_transform = world.get_map().get_spawn_points()[ORIGIN_INDEX]

    probe = world.spawn_actor(blueprint, base_transform)
    physics = probe.get_physics_control()
    mass, com = physics.mass, physics.center_of_mass
    # `moi` is the ENGINE's rotational inertia, not the chassis yaw inertia -- printed only so it
    # is never mistaken for the latter.
    print(f"blueprint mass={mass:.1f} kg  center_of_mass=({com.x:.2f}, {com.y:.2f}, {com.z:.2f}) m"
          f"  engine moi={physics.moi:.2f} (NOT the chassis Iz)")
    for i, w in enumerate(physics.wheels):
        print(f"  wheel[{i}] lat_stiff_value={w.lat_stiff_value:.2f}  "
              f"lat_stiff_max_load={w.lat_stiff_max_load:.2f}  tire_friction={w.tire_friction:.2f}")
    probe.destroy()

    print("\n=== phase 1: linear impulse (mass / units calibration) ===")
    mass_meas, dvx = measure_mass(world, blueprint, base_transform, imu_bp, args.linear_impulse, args)
    err = 100.0 * (mass_meas / mass - 1.0) if mass_meas == mass_meas else float("nan")
    print(f"  P={args.linear_impulse:,.0f} N*s -> dv_x={dvx:.4f} m/s  =>  mass={mass_meas:,.1f} kg "
          f"vs blueprint {mass:.1f} kg ({err:+.1f}%)")
    units_ok = mass_meas == mass_meas and abs(err) < 5.0
    if not units_ok:
        print("  WARNING: recovered mass does not match the blueprint, so CARLA's impulse units "
              "are not the SI N*s assumed here (or the body was not free). Everything below "
              "inherits that -- fix this first.")

    # Phase 1 proves linear impulses are SI N*s, but that does NOT carry over to the angular API:
    # measured here, add_angular_impulse lands about six orders of magnitude off a kg*m^2*rad/s
    # reading. The logic below is deliberately the direct one: apply each of several arbitrary,
    # user-given angular impulses J (nothing is back-solved from a desired rate), read back
    # whatever yaw rate r0 that particular J happened to produce, and compute Iz = J/r0 per trial.
    # Cause and effect only run one way -- J in, r0 out -- so this is the loop that actually
    # generates the "same J, different flavours of the same story" the module docstring promises:
    # several independent (J, r0) pairs, each yielding its own Iz, that should all agree.
    video_factory = make_video_factory(world, args)

    impulses = [float(x) for x in args.impulses.split(",")]
    print(f"\n=== phase 2: angular impulse about z -- {len(impulses)} arbitrary magnitudes ===")
    print("  J (given) = " + ", ".join(f"{j:,.3g}" for j in impulses))

    trials = []
    for i, J in enumerate(impulses, start=1):
        label = f"{i:02d}_J{J:,.0f}".replace(",", "")
        t = measure_yaw_impulse(world, blueprint, base_transform, imu_bp, J, args,
                                video_factory=video_factory, label=label)
        # J / r0[deg/s] / 1e4 converts UE4's centimetre-scaled internal units back to m^2 -- see
        # ANGULAR_IMPULSE_UNITS. r0 is measured, not chosen: it is whatever this J produced.
        t["Iz_raw_rad"] = J / t["r0"] if t["r0"] else float("nan")            # J / r0   [rad]
        t["Iz"] = t["Iz_raw_rad"] / math.degrees(1.0) / 1e4 if t["r0"] else float("nan")
        trials.append(t)
        print(f"  [{i}/{len(impulses)}] J={J:>14,.0f} (given)  ->  r0={math.degrees(t['r0']):+8.2f} "
              f"deg/s (measured; first sample {math.degrees(t['r_first']):+.2f}, tau={t['tau']:.2f} s)"
              f"  =>  Iz={t['Iz']:>9,.1f} kg*m^2")
        if abs(t["r0"]) < math.radians(0.5):
            print(f"    WARNING: r0 only {math.degrees(t['r0']):.3f} deg/s -- too close to gyro "
                  f"noise to trust. Rerun with a larger J at this position in --impulses.")
        if t["cross_axis"] > args.cross_axis_tol:
            print(f"    WARNING: roll/pitch rate reached {t['cross_axis']:.3f} rad/s -- the z "
                  f"impulse is exciting other axes, so gyro.z is not a clean measurement here.")

    print_yaw_impulse_table(trials)

    iz_all = np.array([t["Iz"] for t in trials])
    spread = (iz_all.max() - iz_all.min()) / iz_all.mean()
    Iz = float(np.mean(iz_all))
    j_range = max(impulses) / min(impulses)
    print(f"\nlinearity across impulse magnitudes: spread {spread*100:.4f}% of mean over a "
          f"{j_range:.0f}x range of J")
    if spread > 0.05:
        print("  WARNING: Iz cannot depend on J. Something is nonlinear -- most likely the car "
              "touched down, or the impulse was clamped.")

    print(f"\nIz = {Iz:,.1f} kg*m^2  (add_angular_impulse assumed to take "
          f"{ANGULAR_IMPULSE_UNITS}, averaged over {len(trials)} trials)")
    plausible = args.iz_min < Iz < args.iz_max
    if not plausible:
        print(f"  WARNING: {Iz:,.0f} kg*m^2 falls outside the {args.iz_min:,.0f}-"
              f"{args.iz_max:,.0f} kg*m^2 plausible range for a {mass:.0f} kg car. Either this "
              f"vehicle is unusual, or the {ANGULAR_IMPULSE_UNITS} convention no longer holds "
              f"(e.g. a CARLA/UE4 version change) -- check the candidates table below.")

    # Four candidate conventions, from UE4's radian/degree variants crossed with its centimetre
    # internals, kept here purely as an audit trail: they are separated by factors of 57.3 and
    # 10,000, and a passenger car's yaw inertia is known to within a factor of ~3, so at most one
    # of the four can be the answer -- which is how ANGULAR_IMPULSE_UNITS was established in the
    # first place (see the module docstring).
    raw_rad = float(np.mean([t["Iz_raw_rad"] for t in trials]))
    candidates = {
        "kg*m^2*rad/s":  raw_rad,
        "kg*m^2*deg/s":  raw_rad / math.degrees(1.0),
        "kg*cm^2*rad/s": raw_rad / 1e4,
        "kg*cm^2*deg/s": raw_rad / math.degrees(1.0) / 1e4,
    }
    print(f"\nIz under each candidate unit convention (audit -- only '{ANGULAR_IMPULSE_UNITS}' is "
          f"used above):")
    for name, value in candidates.items():
        flag = "  <-- used" if name == ANGULAR_IMPULSE_UNITS else ""
        print(f"  {name:>16} -> {value:>18,.1f} kg*m^2{flag}")

    combined_video = None
    if video_factory is not None and len(trials) >= 2:
        print("\n=== phase 3: combining per-trial recordings ===")
        # One clip per trial, played back to back in the order applied, each labelled with the J
        # that produced it -- so the video makes the same "same physics, different magnitude"
        # point the table does, just watchable instead of read. Unit is spelled out because J on
        # its own is just a number: ANGULAR_IMPULSE_UNITS is what add_angular_impulse actually
        # takes it as (established in phase 2 above), so this is the physically correct label, not
        # a guess.
        entries = [(f"J = {t['J']:.0f} {ANGULAR_IMPULSE_UNITS}", t.get("video_path"),
                   t.get("video_frames", 0)) for t in trials]
        if args.record == "auto":
            combined_path = os.path.join(args.video_dir, run_name("impulses") + ".mp4")
        else:
            base, ext = os.path.splitext(args.record)
            combined_path = f"{base}_impulses{ext}"
        combined_video = concat_videos_sequential(entries, combined_path)
        if combined_video is None:
            print("  ! could not combine per-trial videos into one -- individual clips are still "
                  "on disk")

    return {"method": "impulse", "Iz": Iz, "units": ANGULAR_IMPULSE_UNITS, "Iz_raw_rad": raw_rad,
            "candidates": candidates, "linearity_spread": spread, "mass": mass,
            "mass_measured": mass_meas, "linear_impulse": args.linear_impulse,
            "units_ok": units_ok, "trials": trials, "combined_video": combined_video}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true",
                        help="skip the simulator entirely and re-run the validation checks on the "
                             "saved JSON")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--vehicle", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--dt", type=float, default=0.01, help="fixed sim step (s)")

    parser.add_argument("--height", type=float, default=80.0,
                        help="spawn height above the map (m) -- high enough not to reach the "
                             "ground within --decay-time")
    parser.add_argument("--impulses", default="145799561,291599122,583198243,1166396486",
                        help="comma-separated raw angular-impulse magnitudes (CARLA's native "
                             "add_angular_impulse units) to apply about z, one trial each -- no "
                             "target rate, no back-solving: each J is applied as given and whatever "
                             "yaw rate it produces is measured and used to compute that trial's Iz. "
                             "Several, because Iz is a constant and a value that drifts with J means "
                             "the model is wrong. Defaults are what worked well for "
                             "vehicle.lincoln.mkz_2020 (~5-40 deg/s of resulting spin, under the "
                             "kg*cm^2*deg/s convention) -- for a different vehicle, start with one "
                             "value and watch the printed r0 to size the rest, since too small "
                             "drowns in gyro noise and too large risks clipping/nonlinearity")
    parser.add_argument("--iz-min", type=float, default=500.0,
                        help="plausibility window used to sanity-check the unit convention")
    parser.add_argument("--iz-max", type=float, default=8000.0)
    parser.add_argument("--linear-impulse", type=float, default=5000.0,
                        help="linear impulse (N*s) for the phase-1 mass/units calibration")
    parser.add_argument("--decay-time", type=float, default=1.0)
    parser.add_argument("--cross-axis-tol", type=float, default=0.05)

    # ---- video ---- #
    parser.add_argument("--record", nargs="?", const="auto", default="",
                        help="record each --impulses trial, then (with 2+ trials) concatenate them "
                             "in order into one combined mp4 with each segment's J overlaid at the "
                             "top; bare flag auto-names it under --video-dir")
    parser.add_argument("--video-dir", default=os.path.join(HERE, "videos"),
                        help="where auto-named recordings go")
    parser.add_argument("--record-view", default="top", choices=sorted(VIEWS),
                        help="camera mount for the recordings -- 'top' (default) reads the yaw "
                             "spin most clearly since the body-rigid-attached camera co-rotates "
                             "with the car, so a top-down view shows the ground visibly spinning "
                             "beneath an apparently still car")
    parser.add_argument("--record-res", default="1280x720", help="recording resolution, WxH")

    parser.add_argument("--out", default=os.path.join(HERE, "yaw_inertia_impulse.json"))
    args = parser.parse_args()

    if args.report_only:
        if not os.path.exists(args.out):
            raise SystemExit(f"{args.out} not found -- run without --report-only first")
        with open(args.out) as f:
            saved = json.load(f)
        try:
            print(f"Loaded {os.path.basename(args.out)}: Iz = {saved['Iz']:,.1f} kg*m^2 "
                  f"({saved.get('units', ANGULAR_IMPULSE_UNITS)})")
            print_yaw_impulse_table(saved["trials"])
        except KeyError:
            raise SystemExit(f"{args.out} is from an older version of this script (different "
                              f"trial fields) -- rerun without --report-only to regenerate it")
        iz_all = np.array([t["Iz"] for t in saved["trials"]])
        spread = (iz_all.max() - iz_all.min()) / iz_all.mean()
        print(f"\nlinearity across impulse magnitudes: spread {spread*100:.4f}% of mean")
        return

    world, original_settings = setup_world(args)
    results = None
    try:
        results = method_impulse(world, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        world.apply_settings(original_settings)
        print("\nCleaned up: world settings restored.")

    if results is None:
        return

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
