"""gen_hdmap.py 의 폴리라인 생성 로직을 코리도 주변에서만 재현하고 Boundary 를 라벨링한다."""
import numpy as np, collections, sys, os, carla

XODR=("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{t}.xodr",
      "/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/{t}/OpenDrive/{t}.xodr")
R = 80.0
STEP = 1.0

town = sys.argv[1]
cor = np.load(f'corridor_{town}.npy', allow_pickle=True).item()
S = np.concatenate([v['snap'][:, :2] for v in cor.values()])
path = next(q.format(t=town) for q in XODR if os.path.exists(q.format(t=town)))
print(f'{town}: xodr 로딩 ...', flush=True)
cm = carla.Map(town, open(path).read())

def near(loc):
    return np.min(np.linalg.norm(S - np.array([loc.x, loc.y]), axis=1)) <= R

# ---- 코리도 주변 Driving waypoint 를 BFS 로 수집 ----
seen, queue, wps = set(), [], []
for x, y, yaw, isj in np.concatenate([v['snap'] for v in cor.values()]):
    w = cm.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0), lane_type=carla.LaneType.Driving)
    if w: queue.append(w)
while queue:
    w = queue.pop()
    k = (w.road_id, w.section_id, w.lane_id, round(w.s / STEP))
    if k in seen: continue
    seen.add(k)
    if not near(w.transform.location): continue
    wps.append(w)
    for nxt in (w.next(STEP) + w.previous(STEP)):
        if nxt.lane_type == carla.LaneType.Driving: queue.append(nxt)
    for nb in (w.get_left_lane(), w.get_right_lane()):
        if nb is not None and nb.lane_type == carla.LaneType.Driving: queue.append(nb)
print(f'{town}: 코리도 내 Driving waypoint {len(wps)}개', flush=True)

# ---- (road, section, lane) 별로 묶어 폴리라인 생성 ----
lanes = collections.defaultdict(list)
for w in wps: lanes[(w.road_id, w.section_id, w.lane_id)].append(w)

polys = []          # (type, Nx2, road_id, lane_id, is_junction, side)
for key, ws in lanes.items():
    ws.sort(key=lambda w: w.s)
    rid, sec, lid = key
    for side in ('left', 'right'):
        cur_t, buf = None, []
        for w in ws:
            mk = w.left_lane_marking if side == 'left' else w.right_lane_marking
            nb = w.get_left_lane()  if side == 'left' else w.get_right_lane()
            is_bd = (not w.is_junction) and ((nb is None) or (nb.lane_type != carla.LaneType.Driving))
            t = 'Boundary' if is_bd else str(mk.type)
            c, rv, hw = w.transform.location, w.transform.get_right_vector(), w.lane_width / 2
            sgn = -1.0 if side == 'left' else 1.0
            p = (c.x + rv.x * hw * sgn, c.y + rv.y * hw * sgn)
            if t != cur_t:
                if len(buf) > 1: polys.append((cur_t, np.array(buf, np.float32), rid, lid, ws[0].is_junction, side))
                cur_t, buf = t, []
            buf.append(p)
        if len(buf) > 1: polys.append((cur_t, np.array(buf, np.float32), rid, lid, ws[0].is_junction, side))
    C = np.array([[w.transform.location.x, w.transform.location.y] for w in ws], np.float32)
    if len(C) > 1: polys.append(('Center', C, rid, lid, ws[0].is_junction, ''))

bands=[]
for w in wps:
    c,rv,hw=w.transform.location,w.transform.get_right_vector(),w.lane_width/2
    bands.append([(c.x-rv.x*hw,c.y-rv.y*hw),(c.x+rv.x*hw,c.y+rv.y*hw)])
print(f'{town}: 폴리라인 {len(polys)}  타입 {dict(collections.Counter(t for t,*_ in polys).most_common())}', flush=True)
np.save(f'carla_{town}.npy', dict(polys=polys, corridor=cor, bands=np.array(bands,np.float32)), allow_pickle=True)
