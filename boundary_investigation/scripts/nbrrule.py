import carla, collections
# npz-only 추론 규칙 재현: Center 의 Left/Right 가 가리키는 (road_id, lane_id) 가
# npz 키(=Driving 차선)에 없으면 그 side 를 boundary 로 본다.
for town in ["Town03","Town05"]:
    cm = carla.Map(town, open(f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr").read())
    wps = [w for w in cm.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving]
    driving_keys = {(w.road_id, w.lane_id) for w in wps}     # npz 가 저장하는 키 집합
    st = collections.Counter()
    for w in wps:
        if w.is_junction: continue
        for nb in (w.get_left_lane(), w.get_right_lane()):
            truth = (nb is None) or (nb.lane_type != carla.LaneType.Driving)
            pred  = (nb is None) or ((nb.road_id, nb.lane_id) not in driving_keys)
            st[(pred, truth)] += 1
    tp,fp,fn,tn = st[(True,True)],st[(True,False)],st[(False,True)],st[(False,False)]
    print(f"{town}: TP={tp} FP={fp} FN={fn} TN={tn}"
          f"  -> precision {tp/max(tp+fp,1):.3f}  recall {tp/max(tp+fn,1):.3f}")
