"""Fakes a VAD-style planning head so the controller can be developed independently of a real E2E model.

Interface contract (matches what VAD/UniAD-style planners hand to a downstream controller):
    plan(ego_x, ego_y, ego_yaw, ego_speed) -> list of (dx, dy)
        ego-relative BEV waypoints, in the ego's current local frame (x-forward, y-left).

Just the desired path's own points, nearest-to-horizon, re-expressed in the ego frame -- no
Hermite correction back toward the path. So index 0 is the nearest path point, not necessarily
(0.0, 0.0); whatever cross-track/heading error the ego currently has shows up directly in the
plan, and the downstream controller (segment-projection tracking) is what corrects for it.
"""
import math


def _normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MockPlanner:
    def __init__(self, path_x, path_y, horizon_s, num_points=6, search_window=50):
        self.path_x = path_x
        self.path_y = path_y
        self.horizon_s = horizon_s
        self.num_points = num_points
        self.search_window = search_window
        self._last_idx = 0

    def _nearest_idx(self, x, y):
        lo = self._last_idx
        hi = min(len(self.path_x), lo + self.search_window)
        dists = [math.hypot(x - self.path_x[i], y - self.path_y[i]) for i in range(lo, hi)]
        idx = lo + dists.index(min(dists))
        self._last_idx = idx
        return idx

    def _horizon_end_idx(self, nearest_idx, ego_speed, min_speed=1.0):
        """Walk forward along the path until we've covered speed * horizon_s meters."""
        lookahead_dist = max(ego_speed, min_speed) * self.horizon_s
        idx = nearest_idx
        traveled = 0.0
        while idx < len(self.path_x) - 1 and traveled < lookahead_dist:
            traveled += math.hypot(self.path_x[idx + 1] - self.path_x[idx],
                                    self.path_y[idx + 1] - self.path_y[idx])
            idx += 1
        return idx

    def plan(self, ego_x, ego_y, ego_yaw, ego_speed, nearest_idx=None):
        """nearest_idx: closest-point index on the global path, if the caller already computed one
        (e.g. against the same fixed path used for the true-deviation metric) -- reused here instead
        of re-searching, so there's a single source of truth for "where on the global path is the ego."
        Falls back to an internal search if not given."""
        if nearest_idx is None:
            nearest_idx = self._nearest_idx(ego_x, ego_y)
        else:
            self._last_idx = nearest_idx
        end_idx = self._horizon_end_idx(nearest_idx, ego_speed)

        start_idx = nearest_idx
        if end_idx == start_idx:
            # Near the goal there's no path left ahead to fill the horizon -- sampling forward from
            # here would repeat the same point num_points+1 times, collapsing every segment to zero
            # length and losing the heading (atan2(0, 0)). Sample the path's tail leading up to the
            # goal instead, so the plan still carries a real heading into the endpoint.
            start_idx = max(0, end_idx - self.num_points)

        # Just the path's own points, evenly spaced from start_idx out to the horizon.
        world_points = []
        for k in range(self.num_points + 1):
            idx = min(start_idx + round((end_idx - start_idx) * k / self.num_points), len(self.path_x) - 1)
            world_points.append((self.path_x[idx], self.path_y[idx]))

        cos_e, sin_e = math.cos(-ego_yaw), math.sin(-ego_yaw)
        local_points = []
        for wx, wy in world_points:
            dx, dy = wx - ego_x, wy - ego_y
            local_points.append((dx * cos_e - dy * sin_e, dx * sin_e + dy * cos_e))
        return local_points


def local_to_world(local_points, ego_x, ego_y, ego_yaw):
    """Inverse of the local-frame transform above -- what the controller does on each planner tick."""
    cos_e, sin_e = math.cos(ego_yaw), math.sin(ego_yaw)
    world_points = []
    for lx, ly in local_points:
        wx = ego_x + lx * cos_e - ly * sin_e
        wy = ego_y + lx * sin_e + ly * cos_e
        world_points.append((wx, wy))
    return world_points


if __name__ == "__main__":
    # No CARLA needed: demo on a synthetic 90-deg-turn path so the planner can be sanity-checked in isolation.
    path_x = [i * 1.0 for i in range(0, 40)]
    path_y = [0.0] * 40
    for i in range(40, 80):
        path_x.append(39.0 + math.sin((i - 40) / 39.0 * math.pi / 2) * 10.0)
        path_y.append(10.0 - math.cos((i - 40) / 39.0 * math.pi / 2) * 10.0)
    planner = MockPlanner(path_x, path_y, num_points=6)

    # ego sitting 1.5m off the path (cross-track error) with a heading error, driving at 10 m/s
    ego_x, ego_y, ego_yaw, ego_speed = 10.0, 1.5, math.radians(10), 10.0
    local_wps = planner.plan(ego_x, ego_y, ego_yaw, ego_speed)
    world_wps = local_to_world(local_wps, ego_x, ego_y, ego_yaw)

    print("local (ego-frame) waypoints -- index 0 is the nearest path point, offset by the ego's "
          "current cross-track/heading error, not (0,0):")
    for lx, ly in local_wps:
        print(f"  ({lx:+.3f}, {ly:+.3f})")
    print("\nworld-frame waypoints (should land exactly on the path, since these are the path's own points):")
    for wx, wy in world_wps:
        print(f"  ({wx:+.2f}, {wy:+.2f})")
