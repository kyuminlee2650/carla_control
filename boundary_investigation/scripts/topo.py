import numpy as np, collections, carla

# 1) waypoint.next(d) 가 실제로 뭘 돌려주는가
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(1.0)
same = diff = 0
nb = collections.Counter()
for w in wps:
    nxt = [x for x in w.next(0.05) if x.lane_type == carla.LaneType.Driving]
    nb[len(nxt)] += 1
    for x in nxt:
        if (x.road_id, x.lane_id) == (w.road_id, w.lane_id): same += 1
        else: diff += 1
print("next(0.05) 의 Driving 후속 개수 분포:", dict(sorted(nb.items())))
print(f"후속이 '자기 자신과 같은 (road_id, lane_id)' 인 경우: {same}  /  다른 차선: {diff}")
print(f"   -> {100*same/(same+diff):.1f}% 가 자기 자신")

# 2) 그럼 실제 npz 의 Topology 필드에는 뭐가 들어있나
NPZ='/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz'
m = dict(np.load(NPZ, allow_pickle=True)['arr'])
c = collections.Counter(); n = collections.Counter()
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        for e in lane:
            t = e['Topology']
            n[len(t)] += 1
            for item in t:
                c['자기 자신' if tuple(item) == (rid, lid) else '다른 차선'] += 1
print(f"\nnpz Topology 필드: 항목 개수 분포 {dict(sorted(n.items()))}")
print(f"   내용물: {dict(c)}")
