import numpy as np, collections, carla, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import boundary_patch as bp

TOWN = 'Town03'
NPZ  = f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{TOWN}_HD_map.npz'
XODR = f'/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{TOWN}.xodr'

# ---------- 1. 라벨링 ----------
m = dict(np.load(NPZ, allow_pickle=True)['arr'])
driving_keys = {(r, l) for r, rd in m.items() for l in rd if l != 'Trigger_Volumes'}
polys = []                                     # (type, Nx2, is_junction)
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type'] == 'Center'), None)
        bs = bp.boundary_sides(lane, driving_keys)
        isj = ct is not None and 'Junction' in ct.get('TopologyType', '')
        for sl in lane:
            t = bp.resolve_lane_type(sl, ct, bs)
            P = np.asarray([rp[0] for rp in sl['Points']], float)[:, :2]
            polys.append((t, P, isj))
print('폴리라인', len(polys), '| 타입', dict(collections.Counter(t for t,_,_ in polys).most_common()))

# ---------- 2. 주행가능영역 (배경) ----------
cm = carla.Map(TOWN, open(XODR).read())
segs = []
for w in cm.generate_waypoints(1.0):
    if w.lane_type != carla.LaneType.Driving: continue
    c = w.transform.location; rv = w.transform.get_right_vector(); hw = w.lane_width/2
    segs.append([(c.x-rv.x*hw, c.y-rv.y*hw), (c.x+rv.x*hw, c.y+rv.y*hw)])
print('주행영역 segment', len(segs))

# ---------- 3. 그리기 ----------
OTHER = {'Broken':'#4a90d9', 'Solid':'#4a90d9', 'SolidSolid':'#4a90d9',
         'NONE':'#4a90d9', 'SolidBroken':'#4a90d9', 'BrokenSolid':'#4a90d9'}

def draw(ax, step=5, lw_b=2.2, lw_o=0.7, lw_c=0.4):
    ax.add_collection(LineCollection(segs, colors='#d8d8d8', linewidths=2.4, zorder=0))
    for t, P, isj in polys:
        if t == 'Boundary': continue
        Q = P[::step] if len(P) > step*3 else P
        if t == 'Center':
            ax.plot(Q[:,0], Q[:,1], color='#bbbbbb', lw=lw_c, ls=(0,(4,4)), zorder=1)
        else:
            ax.plot(Q[:,0], Q[:,1], color=OTHER.get(t,'#4a90d9'), lw=lw_o, zorder=2)
    for t, P, isj in polys:                       # 경계는 맨 위에
        if t != 'Boundary': continue
        Q = P[::step] if len(P) > step*3 else P
        ax.plot(Q[:,0], Q[:,1], color='#e02020', lw=lw_b, zorder=3)
    ax.set_aspect('equal'); ax.invert_yaxis()

# 줌 위치: 연결로가 가장 많은 교차로 + 차선 많은 일반 구간
byj = collections.defaultdict(list)
for w in cm.generate_waypoints(2.0):
    if w.is_junction: byj[w.junction_id].append((w.transform.location.x, w.transform.location.y))
jbig = max(byj, key=lambda k: len(set(map(tuple, byj[k]))))
jc = np.array(byj[jbig]).mean(axis=0)

wide = collections.defaultdict(list)
for w in cm.generate_waypoints(2.0):
    if not w.is_junction: wide[w.road_id].append((w.transform.location.x, w.transform.location.y, w.lane_id))
rbig = max(wide, key=lambda r: len({x[2] for x in wide[r]}))
rc = np.array([[x,y] for x,y,_ in wide[rbig]]).mean(axis=0)

allP = np.concatenate([P for _,P,_ in polys])
fig = plt.figure(figsize=(22, 13))
gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])

ax0 = fig.add_subplot(gs[:, 0]); draw(ax0, step=8)
ax0.set_title(f'{TOWN} full map   |   RED = Boundary (auto-labelled)   BLUE = other lane markings   GREY = drivable area', fontsize=11)
ax0.set_xlim(allP[:,0].min()-20, allP[:,0].max()+20); ax0.set_ylim(allP[:,1].max()+20, allP[:,1].min()-20)

for k, (cx, cy, r, ttl) in enumerate([
        (jc[0], jc[1], 60, f'Intersection (junction {jbig}) - no RED should cut across the junction'),
        (rc[0], rc[1], 70, f'Multi-lane road (road {rbig}) - only the two outermost lines should be RED'),
        (allP[:,0].mean(), allP[:,1].mean(), 90, 'Map centre'),
        (jc[0]+120, jc[1]+120, 80, 'Around the intersection')]):
    ax = fig.add_subplot(gs[k//2, 1+k%2]); draw(ax, step=1, lw_b=3.0, lw_o=1.2, lw_c=0.7)
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy+r, cy-r); ax.set_title(ttl, fontsize=10)

plt.tight_layout(); plt.savefig('/tmp/claude-1000/-home-ailab-carla-control/40dfe823-9875-434c-accf-6ce088910cc3/scratchpad/town03_boundary.png', dpi=110, bbox_inches='tight')
print('saved')
