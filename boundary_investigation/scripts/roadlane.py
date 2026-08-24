import carla, collections
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(1.0)

roads = collections.defaultdict(set)
for w in wps:
    roads[w.road_id].add((w.section_id, w.lane_id, str(w.lane_type)))
print(f"Town03: road_id 개수 = {len(roads)}   전체 waypoint = {len(wps)}")
print(f"road_id 값 범위 = {min(roads)} ~ {max(roads)}  (연속인가? {sorted(roads)==list(range(min(roads),max(roads)+1))})")

junc = [w for w in wps if w.is_junction]
print(f"junction 내부 waypoint = {len(junc)}, 그 road_id 개수 = {len(set(w.road_id for w in junc))}")
print(f"junction_id 개수 = {len(set(w.junction_id for w in junc))}")

print("\n=== 한 road 의 lane_id 구성 (차선 안쪽->바깥쪽) ===")
for rid in sorted(roads):
    lanes = sorted(roads[rid], key=lambda t: (t[0], t[1]))
    ids = [l[1] for l in lanes if l[0] == lanes[0][0]]
    if len(ids) >= 5 and min(ids) < 0 and max(ids) > 0:
        for s, lid, lt in lanes:
            if s != lanes[0][0]: continue
            print(f"   road {rid} section {s}  lane_id {lid:+3d}   lane_type={lt}")
        break

print("\n=== '차로 번호' 가 아니라는 증거: 가장 안쪽 Driving lane 의 |lane_id| 분포 ===")
inner = collections.Counter()
for rid, ls in roads.items():
    for sign in (-1, 1):
        d = [abs(l[1]) for l in ls if l[2] == 'Driving' and (l[1] < 0) == (sign < 0)]
        if d: inner[min(d)] += 1
print("   ", dict(sorted(inner.items())), "  <- 1 이 아니면 안쪽에 비주행 차선(갓길/연석 등)이 있다는 뜻")

print("\n=== lane section 별로 lane_type 이 바뀌는 (road, lane_id) 가 있는가 ===")
byrl = collections.defaultdict(set)
for rid, ls in roads.items():
    for s, lid, lt in ls: byrl[(rid, lid)].add(lt)
mixed = {k: v for k, v in byrl.items() if len(v) > 1}
print(f"    {len(mixed)} / {len(byrl)} 개 (road,lane_id) 가 section 마다 lane_type 이 다름")
for k, v in list(mixed.items())[:5]: print("      ", k, sorted(v))
