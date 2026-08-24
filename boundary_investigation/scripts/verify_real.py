import numpy as np, collections, carla

NPZ = '/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz'
m = dict(np.load(NPZ, allow_pickle=True)['arr'])

# --- npz 만으로 내리는 판정 ---
driving_keys = {(rid, lid) for rid, road in m.items() for lid in road if lid != 'Trigger_Volumes'}
pred = {}
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type'] == 'Center'), None)
        if ct is None: continue
        if 'Junction' in ct['TopologyType']: continue          # 교차로 제외
        for side in ('Left', 'Right'):
            nb = ct[side]
            pred[(rid, lid, side)] = (nb is None or nb[0] is None or tuple(nb) not in driving_keys)

# --- CARLA 정답 ---
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
truth = {}
for w in cm.generate_waypoints(1.0):
    if w.lane_type != carla.LaneType.Driving or w.is_junction: continue
    for side, nb in (('Left', w.get_left_lane()), ('Right', w.get_right_lane())):
        truth.setdefault((w.road_id, w.lane_id, side), set()).add(
            (nb is None) or (nb.lane_type != carla.LaneType.Driving))

common = set(pred) & {k for k, v in truth.items() if len(v) == 1}
st = collections.Counter((pred[k], next(iter(truth[k]))) for k in common)
tp, fp, fn, tn = st[(1,1)], st[(1,0)], st[(0,1)], st[(0,0)]
print(f"공식 Town03 npz 로만 내린 경계 판정  (비교차로, 대조 가능한 {len(common)} side)")
print(f"   TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"   precision {tp/max(tp+fp,1):.4f}   recall {tp/max(tp+fn,1):.4f}")
print(f"\n   npz 에만 있고 CARLA 대조 불가: {len(set(pred)-set(truth))}"
      f" | CARLA 에만: {len(set(truth)-set(pred))}")
print(f"   경계로 판정된 side: {sum(pred.values())} / {len(pred)}")
