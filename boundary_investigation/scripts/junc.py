import collections, carla
for town in ["Town03", "Town05"]:
    cm = carla.Map(town, open(f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr").read())
    wps = [w for w in cm.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving]
    st = collections.Counter()
    for w in wps:
        j = "junction" if w.is_junction else "road"
        for nb in (w.get_left_lane(), w.get_right_lane()):
            edge = (nb is None) or (nb.lane_type != carla.LaneType.Driving)
            st[(j, "edge" if edge else "inner")] += 1
    tot_j = sum(v for k,v in st.items() if k[0]=="junction")
    tot_r = sum(v for k,v in st.items() if k[0]=="road")
    print(f"{town}: junction sides {tot_j} -> edge {st[('junction','edge')]} ({100*st[('junction','edge')]/max(tot_j,1):.0f}%)"
          f" | road sides {tot_r} -> edge {st[('road','edge')]} ({100*st[('road','edge')]/max(tot_r,1):.0f}%)")
