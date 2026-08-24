import numpy as np, collections, json

TOWNS = ['Town06','Town11','Town12','Town13']
LAT, LON = 15.0, 30.0                      # pc_range: x +-15, y +-30
rows = []
for town in TOWNS:
    d = np.load(f'carla_{town}.npy', allow_pickle=True).item()
    polys, cor = d['polys'], d['corridor']
    for rid, info in cor.items():
        snap = info['snap']
        cnt = collections.defaultdict(list)          # type -> 프레임별 인스턴스 수
        dmin = {'Boundary': [], 'SolidSolid': []}
        for x, y, yaw, isj in snap:
            e = np.array([x, y]); a = np.deg2rad(yaw)
            f = np.array([np.cos(a), np.sin(a)]); r = np.array([-np.sin(a), np.cos(a)])
            per = collections.Counter()
            best = {'Boundary': np.inf, 'SolidSolid': np.inf}
            for t, P, *_ in polys:
                V = P - e
                lat, lon = V @ r, V @ f
                inb = (np.abs(lat) <= LAT) & (np.abs(lon) <= LON)
                if inb.sum() > 1:
                    per[t] += 1
                    if t in best:
                        best[t] = min(best[t], float(np.linalg.norm(V[inb], axis=1).min()))
            for k, v in per.items(): cnt[k].append(v)
            for k in dmin:
                cnt.setdefault(k, [])
                if len(cnt[k]) < len(cnt.get('Center', [])) : pass
                dmin[k].append(best[k])
        n = len(snap)
        def stat(t):
            v = np.array(cnt.get(t, []) + [0]*(n - len(cnt.get(t, []))))
            return 100*float((v > 0).mean()), float(v.mean())
        ss_pct, ss_mean = stat('SolidSolid')
        bd_pct, bd_mean = stat('Boundary')
        db = np.array(dmin['Boundary']); ds = np.array(dmin['SolidSolid'])
        rows.append(dict(route=rid, town=town, scenario=info['scenario'], frames=n,
                         ss_pct=ss_pct, ss_mean=ss_mean, bd_pct=bd_pct, bd_mean=bd_mean,
                         bd_dist=float(np.median(db[np.isfinite(db)])) if np.isfinite(db).any() else float('nan'),
                         ss_dist=float(np.median(ds[np.isfinite(ds)])) if np.isfinite(ds).any() else float('nan')))

order = ['24367','27582','3144','2416','2715','2286','17569','2790','3373','1792']
rows.sort(key=lambda r: order.index(r['route']))
json.dump(rows, open('metrics.json','w'), indent=1)

print(f"{'route':>6} {'town':>7} {'scenario':<42} {'frm':>4} | "
      f"{'SS %frm':>8} {'SS inst':>8} {'SS dist':>8} | {'BD %frm':>8} {'BD inst':>8} {'BD dist':>8}")
print('-'*128)
for r in rows:
    fd = lambda v: f'{v:8.1f}' if np.isfinite(v) else '       -'
    print(f"{r['route']:>6} {r['town']:>7} {r['scenario']:<42} {r['frames']:>4} | "
          f"{r['ss_pct']:8.1f} {r['ss_mean']:8.2f} {fd(r['ss_dist'])} | "
          f"{r['bd_pct']:8.1f} {r['bd_mean']:8.2f} {fd(r['bd_dist'])}")
print('-'*128)
w = np.array([r['frames'] for r in rows], float); w /= w.sum()
print(f"{'가중평균':>6} {'':>7} {'':<42} {int(sum(r['frames'] for r in rows)):>4} | "
      f"{sum(w[i]*rows[i]['ss_pct'] for i in range(len(rows))):8.1f} "
      f"{sum(w[i]*rows[i]['ss_mean'] for i in range(len(rows))):8.2f} {'':>8} | "
      f"{sum(w[i]*rows[i]['bd_pct'] for i in range(len(rows))):8.1f} "
      f"{sum(w[i]*rows[i]['bd_mean'] for i in range(len(rows))):8.2f}")
