import numpy as np, collections, sys, os, resource, boundary_patch as bp
town = sys.argv[1]
R = 140.0                                   # 코리도 반경
NPZ = f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{town}_HD_map.npz'

cor = np.load(f'corridor_{town}.npy', allow_pickle=True).item()
S = np.concatenate([v['snap'][:, :2] for v in cor.values()])          # 이 타운 모든 라우트 샘플
lo, hi = S.min(0) - R, S.max(0) + R

print(f'{town}: npz 로딩 {os.path.getsize(NPZ)/1e6:.0f} MB ...', flush=True)
m = dict(np.load(NPZ, allow_pickle=True)['arr'])
print(f'{town}: road {len(m)}개, RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB', flush=True)

driving_keys = {(r, l) for r, rd in m.items() for l in rd if l != 'Trigger_Volumes'}
kept, gstat = [], collections.Counter()
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type'] == 'Center'), None)
        bs = bp.boundary_sides(lane, driving_keys)
        isj = ct is not None and 'Junction' in ct.get('TopologyType', '')
        for sl in lane:
            t = bp.resolve_lane_type(sl, ct, bs)
            gstat[t] += 1
            P = np.asarray([rp[0] for rp in sl['Points']], np.float32)[:, :2]
            if P[:, 0].max() < lo[0] or P[:, 0].min() > hi[0] or \
               P[:, 1].max() < lo[1] or P[:, 1].min() > hi[1]: continue
            if np.min(np.linalg.norm(P[::20][:, None, :] - S[None, ::3, :], axis=-1)) > R: continue
            side = bp.marking_side(sl, ct)
            kept.append((t, P, int(rid), int(lid), bool(isj), side or ''))
del m
print(f'{town}: 전체 폴리라인 {sum(gstat.values())} -> 코리도 내 {len(kept)}', flush=True)
print(f'{town}: 전체 타입 분포 {dict(gstat.most_common())}', flush=True)
np.save(f'labelled_{town}.npy',
        dict(kept=kept, global_stat=dict(gstat), corridor=cor), allow_pickle=True)
print(f'{town}: 저장 완료  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB', flush=True)
