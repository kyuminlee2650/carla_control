import carla, collections
for town in ["Town03","Town05"]:
    cm = carla.Map(town, open(f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr").read())
    wps = [w for w in cm.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving]
    # outermost driving lane per (road_id, section?, sign) -- npz has no section, mimic that
    outer = {}
    for w in wps:
        k = (w.road_id, 1 if w.lane_id > 0 else -1)
        outer[k] = max(outer.get(k, 0), abs(w.lane_id))
    st = collections.Counter()
    for w in wps:
        if w.is_junction: continue
        k = (w.road_id, 1 if w.lane_id > 0 else -1)
        pred_outer = abs(w.lane_id) == outer[k]        # npz-inferable guess: this lane's outer edge is boundary
        nb = w.get_right_lane() if w.lane_id < 0 else w.get_left_lane()   # the outward side
        truth = (nb is None) or (nb.lane_type != carla.LaneType.Driving)
        st[(pred_outer, truth)] += 1
    tp,fp,fn,tn = st[(True,True)],st[(True,False)],st[(False,True)],st[(False,False)]
    print(f"{town}: outer-edge-of-outermost-lane 규칙  TP={tp} FP={fp} FN={fn} TN={tn}"
          f"  -> precision {tp/max(tp+fp,1):.3f}  recall {tp/max(tp+fn,1):.3f}")
