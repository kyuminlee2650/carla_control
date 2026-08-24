import numpy as np, collections, carla, boundary_patch as bp

NPZ = '/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz'
m = dict(np.load(NPZ, allow_pickle=True)['arr'])
driving_keys = {(r, l) for r, road in m.items() for l in road if l != 'Trigger_Volumes'}

# --- 패치 적용 결과 ---
before, after = collections.Counter(), collections.Counter()
tagged = []                      # (road_id, lane_id, side, 폴리라인)
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type'] == 'Center'), None)
        bs = bp.boundary_sides(lane, driving_keys)
        for sl in lane:
            t = bp.resolve_lane_type(sl, ct, bs)
            before[sl['Type']] += 1; after[t] += 1
            if t == 'Boundary':
                tagged.append((rid, lid, bp.marking_side(sl, ct),
                               np.asarray([rp[0] for rp in sl['Points']], float)[:, :2]))

print("타입 분포  변경 전 :", dict(before.most_common()))
print("           변경 후 :", dict(after.most_common()))
print(f"\nBoundary 로 라벨된 폴리라인 {len(tagged)}개")

# --- 검증: 그 폴리라인의 바깥쪽 이웃이 정말 비주행인가 (CARLA 대조) ---
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
ok = bad = skip = 0
for rid, lid, side, P in tagged:
    mid = P[len(P)//2]
    w = cm.get_waypoint(carla.Location(x=float(mid[0]), y=float(mid[1]), z=0.0),
                        lane_type=carla.LaneType.Driving)
    if w is None: skip += 1; continue
    nb = w.get_left_lane() if side == 'left' else w.get_right_lane()
    if (nb is None) or (nb.lane_type != carla.LaneType.Driving): ok += 1
    else: bad += 1
print(f"검증(중점에서 CARLA 조회): 바깥이 비주행 {ok}   주행 {bad}   조회실패 {skip}")
print(f"   -> 정확도 {100*ok/max(ok+bad,1):.2f}%")
