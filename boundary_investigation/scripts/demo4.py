import numpy as np, carla
m = dict(np.load('/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz',
                 allow_pickle=True)['arr'])
cm = carla.Map('Town03', open('/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr').read())
wps = cm.generate_waypoints(2.0)

for rid, road in m.items():
    lids = [l for l in road if l != 'Trigger_Volumes']
    neg = sorted([l for l in lids if l < 0], reverse=True)
    if len(neg) < 2: continue
    cts = {l: next((e for e in road[l] if e['Type']=='Center'), None) for l in neg}
    if any(c is None or c['TopologyType'] != 'Normal' for c in cts.values()): continue
    print(f"=== road {rid} (일반 구간) : 같은 방향 차선 {neg} ===\n")
    for lid in neg:
        ct = cts[lid]
        w  = next((w for w in wps if w.road_id==rid and w.lane_id==lid), None)
        L, R = (w.get_left_lane(), w.get_right_lane()) if w else (None, None)
        f = lambda n: f"({n.road_id},{n.lane_id},{str(n.lane_type)})" if n else "None"
        print(f"  lane {lid:+d}")
        print(f"     Left     = {str(ct['Left']):16s}  <- 옆(기준선 쪽)   CARLA 실제: {f(L)}")
        print(f"     Right    = {str(ct['Right']):16s}  <- 옆(바깥쪽)     CARLA 실제: {f(R)}")
        print(f"     Topology = {str([tuple(x) for x in ct['Topology']]):16s}  <- 앞(이어지는 차선)")
        print()
    break
