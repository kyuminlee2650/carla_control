import numpy as np, collections
# ---- prepare_B2D.gengrate_map() 의 루프를 그대로 재현 ----
map_info = dict(np.load('/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town02_HD_map.npz',
                        allow_pickle=True)['arr'])
lane_points, lane_types, lane_sample_points = [], [], []
trigger_volumes_points, trigger_volumes_types, trigger_volumes_sample_points = [], [], []

for road_id, road in map_info.items():                 # road_id : 순회 변수
    for lane_id, lane in road.items():                 # lane_id : 순회 변수
        if lane_id == 'Trigger_Volumes':
            for stv in lane:
                p = np.array(stv['Points']); p[:,1] *= -1
                trigger_volumes_points.append(p)
                trigger_volumes_sample_points.append(p.mean(axis=0))
                trigger_volumes_types.append(stv['Type'])
        else:
            for single_lane in lane:
                points = np.array([rp[0] for rp in single_lane['Points']])   # location 만
                points[:,1] *= -1
                lane_points.append(points)                                   # 평평한 리스트에 append
                lane_types.append(single_lane['Type'])                       # 문자열 하나
                n = points.shape[0]
                dp = [50*i for i in range(n//50 + (1 if n % 50 else 0))]
                dp.append(n-1)
                lane_sample_points.append(points[dp])
# ---- 여기까지가 함수 전부. road_id / lane_id 는 어디에도 저장되지 않음 ----

print("=== gengrate_map() 출력 (Town02) ===")
print(f"  lane_points        : list, len={len(lane_points)}   원소 예: ndarray{lane_points[0].shape}")
print(f"  lane_types         : list, len={len(lane_types)}   원소 예: {lane_types[:6]}")
print(f"  lane_sample_points : list, len={len(lane_sample_points)}   원소 예: ndarray{lane_sample_points[0].shape}")
print(f"  trigger_volumes_*  : len={len(trigger_volumes_points)}  types={collections.Counter(trigger_volumes_types)}")
print()
print("  입력 npz : road", len(map_info), "개 ->", sum(len(v) for v in map_info.values()), "개 (road,lane) 키")
print("  출력     : 평평한 폴리라인", len(lane_points), "개.  road/lane 경계는 흔적 없음")
print()
print("  살아남은 필드 : Points(위치만), Type")
print("  사라진 필드   : road_id, lane_id, Left, Right, Topology, TopologyType, Color,")
print("                  roll/pitch/yaw, is_junction")
