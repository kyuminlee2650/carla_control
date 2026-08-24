import carla, collections, numpy as np, itertools
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(0.5)

# 369 -> 248 재조정 확인
roads = {w.road_id for w in wps}
jroads = {w.road_id for w in wps if w.is_junction}
print(f"driving waypoint 가 생기는 road: {len(roads)}  (그중 교차로 연결로 {len(jroads)})")

# 한 junction 안의 연결로들이 물리적으로 겹치는가
byj = collections.defaultdict(lambda: collections.defaultdict(list))
for w in wps:
    if w.is_junction: byj[w.junction_id][w.road_id].append((w.transform.location.x, w.transform.location.y))
jid = max(byj, key=lambda k: len(byj[k]))
rs = {r: np.array(p) for r, p in byj[jid].items()}
print(f"\njunction {jid}: 연결로 {len(rs)}개")
cross = 0
for a, b in itertools.combinations(rs, 2):
    d = np.linalg.norm(rs[a][:,None,:] - rs[b][None,:,:], axis=-1).min()
    if d < 1.0: cross += 1
print(f"  연결로 쌍 {len(list(itertools.combinations(rs,2)))}개 중 중심선 거리 1 m 이내로 스치는 쌍: {cross}")

# 연결로 waypoint 의 이웃 유무
n_none = n_any = 0
for w in wps:
    if not w.is_junction: continue
    for nb in (w.get_left_lane(), w.get_right_lane()):
        n_any += 1
        if nb is None: n_none += 1
print(f"\n교차로 내부 waypoint 의 좌/우 이웃 조회 {n_any}회 중 None 반환 {n_none}회 ({100*n_none/n_any:.0f}%)")
