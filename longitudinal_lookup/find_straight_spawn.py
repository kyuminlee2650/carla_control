"""Rank spawn points by how far a vehicle can drive straight from them.

The calibration sweep holds one control input and drives until the trial ends,
so it needs a spawn point with a long straight ahead. On a short straight the
vehicle reaches a curve, drifts off the road and hits scenery, and the trial
records the wall instead of the powertrain.

Straight length here is how far the lane's centerline stays within --yaw-tol of
its initial heading, walked in --step increments.

Usage:
    cd ~/carla_control
    .venv/bin/python longitudinal_lookup/find_straight_spawn.py                  # current map
    .venv/bin/python longitudinal_lookup/find_straight_spawn.py --map Town06     # loads the map first
    .venv/bin/python longitudinal_lookup/find_straight_spawn.py --compare Town10HD_Opt,Town06,Town04
"""

import argparse

import carla


def angle_diff(a, b):
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def straight_length(waypoint, step, max_len, yaw_tol):
    """Distance the lane stays within yaw_tol of its starting heading."""
    start_yaw = waypoint.transform.rotation.yaw
    current = waypoint
    total = 0.0
    while total < max_len:
        nxt = current.next(step)
        if not nxt:
            break
        # at a junction take the branch that continues straightest
        current = min(nxt, key=lambda w: abs(angle_diff(w.transform.rotation.yaw, start_yaw)))
        if abs(angle_diff(current.transform.rotation.yaw, start_yaw)) > yaw_tol:
            break
        total += step
    return total


def survey(world, args):
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    scored = []
    for i, transform in enumerate(spawn_points):
        wp = carla_map.get_waypoint(transform.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        scored.append((straight_length(wp, args.step, args.max_len, args.yaw_tol), i))
    scored.sort(reverse=True)
    return carla_map.name, len(spawn_points), scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="", help="load this map before surveying")
    parser.add_argument("--compare", default="", help="comma-separated maps to load and compare in turn")
    parser.add_argument("--step", type=float, default=2.0, help="waypoint walk increment (m)")
    parser.add_argument("--max-len", type=float, default=800.0, help="stop measuring past this (m)")
    parser.add_argument("--yaw-tol", type=float, default=3.0, help="heading deviation that ends the straight (deg)")
    parser.add_argument("--top", type=int, default=8, help="spawn points to list per map")
    parser.add_argument("--needed", type=float, default=300.0,
                        help="straight length the sweep needs (m); reported as a pass count")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    maps = [m.strip() for m in args.compare.split(",") if m.strip()] if args.compare else [args.map]
    for name in maps:
        world = client.load_world(name) if name else client.get_world()
        map_name, n_spawn, scored = survey(world, args)
        ok = sum(1 for length, _ in scored if length >= args.needed)
        print(f"\n=== {map_name} ===")
        print(f"  스폰포인트 {n_spawn}개 | {args.needed:.0f}m 이상 직선 확보: {ok}개 "
              f"| 최장 {scored[0][0]:.0f}m")
        print(f"  상위 {args.top}개 (직선거리 / 스폰 인덱스):")
        for length, idx in scored[:args.top]:
            print(f"    {length:6.0f} m   --origin-index {idx}")


if __name__ == "__main__":
    main()
