import numpy as np, collections, boundary_patch as bp
m = dict(np.load('/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz', allow_pickle=True)['arr'])
dk = {(r,l) for r,rd in m.items() for l in rd if l!='Trigger_Volumes'}

bpolys = []
for rid, road in m.items():
    for lid, lane in road.items():
        if lid=='Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type']=='Center'), None)
        bs = bp.boundary_sides(lane, dk)
        for sl in lane:
            if bp.resolve_lane_type(sl, ct, bs) == 'Boundary':
                bpolys.append((rid, lid, np.asarray([rp[0] for rp in sl['Points']],float)[:,:2]))
L = lambda P: np.linalg.norm(np.diff(P,axis=0),axis=1).sum()

# (a) 같은 lane 안의 완전 중복
seen, uniq, dup_same = [], [], 0
for rid,lid,P in bpolys:
    if any(rid==r and lid==l and Q.shape==P.shape and np.allclose(Q,P) for r,l,Q in uniq):
        dup_same += 1
    else: uniq.append((rid,lid,P))
print(f"Boundary 폴리라인 {len(bpolys)}개 -> 같은 lane 내 완전중복 {dup_same}개 제거 후 {len(uniq)}개")
print(f"   총연장 {sum(L(P) for _,_,P in bpolys):8.0f} m  ->  {sum(L(P) for _,_,P in uniq):8.0f} m")

# (b) 서로 다른 lane 이 같은 물리적 선을 공유하는가 (인접 차선 공유 선)
dup_cross = 0; final = []
for rid,lid,P in uniq:
    if any(Q.shape==P.shape and np.allclose(Q,P) for _,_,Q in final): dup_cross += 1
    else: final.append((rid,lid,P))
print(f"   서로 다른 lane 간 중복 {dup_cross}개 -> 최종 {len(final)}개, 총연장 {sum(L(P) for _,_,P in final):.0f} m")
print(f"   (비교차로 실제 가장자리 총연장 7908 m 대비 {100*sum(L(P) for _,_,P in final)/7908:.1f}%)")
