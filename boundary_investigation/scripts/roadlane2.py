import carla, collections
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(1.0)
print("generate_waypoints 가 돌려주는 lane_type:", dict(collections.Counter(str(w.lane_type) for w in wps)))

roads = collections.defaultdict(set)
for w in wps: roads[w.road_id].add((w.section_id, w.lane_id, str(w.lane_type)))

# 양방향 + 비주행 차선이 섞인 road 예시
for rid in sorted(roads):
    ls = [l for l in roads[rid] if l[0]==0]
    ids = [l[1] for l in ls]
    if len(ls) >= 4 and min(ids) < 0 and max(ids) > 0:
        print(f"\n=== road {rid} section 0 (기준선에서 바깥쪽으로) ===")
        for s, lid, lt in sorted(ls, key=lambda t: -t[1]):
            print(f"    lane_id {lid:+3d}   {lt}")
        break

# lane_id 부호 = 주행방향
print("\n=== lane_id 부호와 주행방향 ===")
for rid in sorted(roads):
    ls = [l for l in roads[rid] if l[0]==0 and l[2]=='Driving']
    if any(l[1]<0 for l in ls) and any(l[1]>0 for l in ls):
        for w in wps:
            if w.road_id==rid and w.section_id==0 and w.lane_type==carla.LaneType.Driving:
                print(f"    road {rid} lane {w.lane_id:+3d}  yaw={w.transform.rotation.yaw:8.2f}  s={w.s:6.1f}")
        break

# 이웃 조회 방향 확인
print("\n=== get_left/right_lane() 은 '주행방향 기준' 인가 ===")
for rid in sorted(roads):
    ls = [l for l in roads[rid] if l[0]==0 and l[2]=='Driving']
    if any(l[1]<0 for l in ls) and any(l[1]>0 for l in ls):
        seen=set()
        for w in wps:
            if w.road_id!=rid or w.section_id!=0 or w.lane_type!=carla.LaneType.Driving: continue
            if w.lane_id in seen: continue
            seen.add(w.lane_id)
            L,R = w.get_left_lane(), w.get_right_lane()
            f=lambda n: f"({n.road_id},{n.lane_id},{str(n.lane_type)})" if n else "None"
            print(f"    lane {w.lane_id:+3d}  Left={f(L):28s} Right={f(R)}")
        break
