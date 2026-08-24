import numpy as np, collections
m = dict(np.load('/home/ailab/2026intern/jsn/maps/Town12_HD_map.npz', allow_pickle=True)['arr'])
c = collections.Counter(); pts = collections.Counter(); col = collections.Counter()
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes':
            for t in lane: c['TV:'+t['Type']] += 1
            continue
        for sl in lane:
            c[sl['Type']] += 1
            pts[sl['Type']] += len(sl['Points'])
            col[(sl['Type'], sl.get('Color'))] += 1
print("=== lane 'Type' histogram (polyline count / total points) ===")
for k, v in c.most_common():
    print(f"{k:22s} {v:7d}   pts={pts.get(k,0):9d}")
print("\n=== (Type, Color) ===")
for k, v in col.most_common(20): print(f"{str(k):40s} {v}")
KEEP = {'Broken','Solid','SolidSolid','Center'}
tot = sum(v for k,v in c.items() if not k.startswith('TV:'))
kept = sum(v for k,v in c.items() if k in KEEP)
print(f"\nb2d dataset keeps {kept}/{tot} = {100*kept/tot:.1f}% of polylines; drops {tot-kept}")
