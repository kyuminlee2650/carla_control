import sys, collections

import carla
for town in ["Town03", "Town05", "Town12"]:
    p = f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
    try:
        cm = carla.Map(town, open(p).read())
    except Exception as e:
        print(town, "skip", e); continue
    wps = cm.generate_waypoints(2.0)
    mk = collections.Counter()          # marking type on true drivable edges
    mk_inner = collections.Counter()    # marking type between two driving lanes
    nbr = collections.Counter()         # lane_type of the neighbour on the edge side
    n_edge = n_inner = 0
    for w in wps:
        if w.lane_type != carla.LaneType.Driving: continue
        for side, nb, m in (("L", w.get_left_lane(), w.left_lane_marking),
                            ("R", w.get_right_lane(), w.right_lane_marking)):
            is_edge = (nb is None) or (nb.lane_type != carla.LaneType.Driving)
            nbr[str(nb.lane_type) if nb is not None else "None"] += 1
            if is_edge:
                n_edge += 1; mk[str(m.type) if m else "None"] += 1
            else:
                n_inner += 1; mk_inner[str(m.type) if m else "None"] += 1
    print(f"\n===== {town}  driving wps={sum(1 for w in wps if w.lane_type==carla.LaneType.Driving)} =====")
    print(f"  drivable-edge sides : {n_edge}   inner sides: {n_inner}")
    print("  marking type ON EDGE     :", dict(mk.most_common()))
    print("  marking type BETWEEN lanes:", dict(mk_inner.most_common()))
    print("  neighbour lane_type      :", dict(nbr.most_common(10)))
