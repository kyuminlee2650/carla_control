import xml.etree.ElementTree as ET, numpy as np, collections, sys, carla, os

XML='/home/ailab/2026intern/jsn/2026-Summer-Internship/pipeline/data/bench2drive10_abilities.xml'
XODR=("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{t}.xodr",
      "/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/{t}/OpenDrive/{t}.xodr")
WANT={'24367','27582','3144','2416','2715','2286','17569','2790','3373','1792'}

root=ET.parse(XML).getroot()
routes={}
for rt in root.findall('.//route'):
    if rt.get('id') not in WANT: continue
    P=np.array([[float(p.get('x')),float(p.get('y')),float(p.get('z'))]
                for p in rt.findall('.//waypoints/position')])
    routes[rt.get('id')]=dict(town=rt.get('town'), xml=P,
                              scenario=[s.get('type') for s in rt.findall('.//scenarios/scenario')][0])

town=sys.argv[1]
path=next(q.format(t=town) for q in XODR if os.path.exists(q.format(t=town)))
print(f'{town}: xodr 로딩 {os.path.getsize(path)/1e6:.0f} MB ...', flush=True)
cm=carla.Map(town, open(path).read())
print(f'{town}: 로딩 완료', flush=True)

out={}
for rid,info in routes.items():
    if info['town']!=town: continue
    P=info['xml']
    # 4 m 간격 선형 보간
    seg=np.linalg.norm(np.diff(P[:,:2],axis=0),axis=1)
    dense=[]
    for i,d in enumerate(seg):
        n=max(int(d//4),1)
        for k in range(n): dense.append(P[i]+(P[i+1]-P[i])*k/n)
    dense.append(P[-1]); dense=np.array(dense)
    # 도로에 스냅
    snap=[]
    for q in dense:
        w=cm.get_waypoint(carla.Location(x=float(q[0]),y=float(q[1]),z=float(q[2])),
                          lane_type=carla.LaneType.Driving)
        if w is None: continue
        L=w.transform.location
        snap.append([L.x,L.y,w.transform.rotation.yaw, float(w.is_junction)])
    snap=np.array(snap)
    out[rid]=dict(town=town, scenario=info['scenario'], xml=P, snap=snap)
    d=np.linalg.norm(np.diff(snap[:,:2],axis=0),axis=1).sum()
    print(f'  route {rid:>6} {info["scenario"]:<42} 샘플 {len(snap):5d}  경로장 {d:8.1f} m', flush=True)
np.save(f'corridor_{town}.npy', out, allow_pickle=True)
