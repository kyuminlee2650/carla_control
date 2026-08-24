import carla, collections
# gen_hdmap 은 Left/Right 를 세그먼트의 '마지막 waypoint' 에서 한 번만 기록한다.
# 같은 (road_id, lane_id) 구간 안에서 이웃의 boundary 여부가 바뀌면 그 샘플은 구간 일부에 대해 틀린다.
for town in ["Town03","Town05"]:
    cm = carla.Map(town, open(f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr").read())
    wps = [w for w in cm.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving and not w.is_junction]
    runs = collections.defaultdict(lambda: {'L': set(), 'R': set()})
    for w in wps:
        k = (w.road_id, w.lane_id)
        for side, nb in (('L', w.get_left_lane()), ('R', w.get_right_lane())):
            runs[k][side].add((nb is None) or (nb.lane_type != carla.LaneType.Driving))
    unstable = sum(1 for v in runs.values() for s in 'LR' if len(v[s]) > 1)
    total    = sum(1 for v in runs.values() for s in 'LR' if len(v[s]) > 0)
    print(f"{town}: (road,lane) side 수 {total} 중 구간 내부에서 boundary 여부가 뒤바뀌는 side = {unstable} ({100*unstable/total:.1f}%)")
