import carla, collections
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(1.0)
# is_junction 의 정의 확인
agree = sum(1 for w in wps if w.is_junction == (w.junction_id != -1))
print(f"is_junction == (junction_id != -1) 인 waypoint: {agree}/{len(wps)}")
# 한 road 안에서 is_junction 이 섞이는가
byroad = collections.defaultdict(set)
for w in wps: byroad[w.road_id].add(w.is_junction)
mixed = [r for r,v in byroad.items() if len(v)>1]
print(f"한 road 안에서 is_junction 이 섞이는 road: {len(mixed)}/{len(byroad)}  -> road 단위 속성")
# junction 하나가 몇 개의 연결로(road)로 이뤄지는가
j = collections.defaultdict(set)
for w in wps:
    if w.junction_id != -1: j[w.junction_id].add(w.road_id)
print(f"junction 당 연결로 road 수: min {min(map(len,j.values()))}  max {max(map(len,j.values()))}  평균 {sum(map(len,j.values()))/len(j):.1f}")
# Topology = next(0.05) 의 Driving 후속. 분기 개수 분포
nb = collections.Counter()
for w in wps:
    nb[len([x for x in w.next(0.05) if x.lane_type==carla.LaneType.Driving])] += 1
print("next(0.05) Driving 후속 개수 분포:", dict(sorted(nb.items())))
