import numpy as np, collections, carla, boundary_patch as bp
town='Town06'
m = dict(np.load(f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{town}_HD_map.npz', allow_pickle=True)['arr'])
dk = {(r,l) for r,rd in m.items() for l in rd if l!='Trigger_Volumes'}
pred={}
for rid,road in m.items():
    for lid,lane in road.items():
        if lid=='Trigger_Volumes': continue
        ct=next((e for e in lane if e['Type']=='Center'),None)
        if ct is None or 'Junction' in ct.get('TopologyType',''): continue
        bs=bp.boundary_sides(lane,dk)
        for s in ('left','right'): pred[(rid,lid,s)]=bs[s]
cm=carla.Map(town,open(f'/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr').read())
truth={}
for w in cm.generate_waypoints(1.0):
    if w.lane_type!=carla.LaneType.Driving or w.is_junction: continue
    for s,nb in (('left',w.get_left_lane()),('right',w.get_right_lane())):
        truth.setdefault((w.road_id,w.lane_id,s),set()).add((nb is None) or (nb.lane_type!=carla.LaneType.Driving))
common=set(pred)&{k for k,v in truth.items() if len(v)==1}
st=collections.Counter((pred[k],next(iter(truth[k]))) for k in common)
tp,fp,fn,tn=st[(1,1)],st[(1,0)],st[(0,1)],st[(0,0)]
print(f"{town} npz-only 규칙 (비교차로 {len(common)} side): TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"   precision {tp/max(tp+fp,1):.4f}   recall {tp/max(tp+fn,1):.4f}")
print(f"   npz 에만 {len(set(pred)-set(truth))} | CARLA 에만 {len(set(truth)-set(pred))}")
