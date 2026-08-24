import numpy as np, collections, carla, boundary_patch as bp
TOWN='Town03'
m = dict(np.load(f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{TOWN}_HD_map.npz', allow_pickle=True)['arr'])
dk = {(r,l) for r,rd in m.items() for l in rd if l!='Trigger_Volumes'}

need, got = set(), set()          # (road, lane, side) 단위
blen = collections.Counter()
for rid, road in m.items():
    for lid, lane in road.items():
        if lid=='Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type']=='Center'), None)
        bs = bp.boundary_sides(lane, dk)
        for side in ('left','right'):
            if bs[side]: need.add((rid,lid,side))
        for sl in lane:
            s = bp.marking_side(sl, ct)
            if s is not None and bs[s]:
                got.add((rid,lid,s))
                P = np.asarray([rp[0] for rp in sl['Points']],float)[:,:2]
                blen['Boundary'] += np.linalg.norm(np.diff(P,axis=0),axis=1).sum()
miss = need - got
print(f"경계여야 할 (road,lane,side) : {len(need)}")
print(f"실제로 폴리라인이 라벨된 것   : {len(got)}    누락 {len(miss)}")
if miss:
    print("  누락 예시:", list(miss)[:8])
    for rid,lid,side in list(miss)[:3]:
        print(f"    road {rid} lane {lid} {side}: 이 lane 의 엔트리 타입 = {[e['Type'] for e in m[rid][lid]]}")

# 커버리지: 비교차로 도로 가장자리 실제 길이 대비
cm = carla.Map(TOWN, open(f'/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{TOWN}.xodr').read())
true_len = 0.0
for w in cm.generate_waypoints(1.0):
    if w.lane_type!=carla.LaneType.Driving or w.is_junction: continue
    for nb in (w.get_left_lane(), w.get_right_lane()):
        if (nb is None) or (nb.lane_type!=carla.LaneType.Driving): true_len += 1.0   # 1 m 간격
print(f"\n비교차로 실제 도로 가장자리 총연장 : {true_len:8.0f} m")
print(f"Boundary 라벨 폴리라인 총연장      : {blen['Boundary']:8.0f} m")
print(f"   커버리지 {100*blen['Boundary']/true_len:.1f}%")
