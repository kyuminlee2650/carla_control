import json, glob, os, numpy as np
ROUTES = {'24367':'Town06','27582':'Town11','3144':'Town12','2416':'Town12','2715':'Town12',
          '2286':'Town12','17569':'Town12','2790':'Town12','3373':'Town13','1792':'Town12'}
RUNS = '/home/ailab/2026intern/kmlee/vad_demo_video/runs'
out = {}
for r, town in ROUTES.items():
    cands = sorted(glob.glob(f'{RUNS}/route{r}_*/frames/*/metric_info.json'))
    if not cands:
        print(f'route {r:>6} ({town}): metric_info 없음'); continue
    p = max(cands, key=os.path.getsize)
    mi = json.load(open(p))
    ticks = sorted(mi, key=int)
    locs = np.array([mi[t]['location'] for t in ticks], float)
    out[r] = dict(town=town, locs=locs, src=p.split('/frames/')[0].split('/')[-1])
    d = np.linalg.norm(np.diff(locs[:,:2],axis=0),axis=1).sum()
    print(f'route {r:>6} ({town}): tick {len(ticks):5d}  주행거리 {d:7.1f} m  <- {out[r]["src"]}')
np.save('traj.npy', out, allow_pickle=True)
print(f'\n수집 완료 {len(out)}/10')
