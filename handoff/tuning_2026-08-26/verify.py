import sys, json, statistics as st
sys.path.insert(0,'/tmp/claude-1000/-home-ailab-carla-control/09773ca3-6589-4edb-a014-3af4978602cf/scratchpad')
from evo import run, T

CTRL = sys.argv[1]; SPEED = int(sys.argv[2]); REPS = int(sys.argv[3]); CFGS = sys.argv[4]
OUT = sys.argv[5] if len(sys.argv)>5 else None
cfgs = [json.loads(l) for l in open(CFGS) if l.strip()]
res = []
for ci, cfg in enumerate(cfgs):
    runs = []
    for i in range(REPS):
        r = run(SPEED, CTRL, **cfg)
        if r is not None: runs.append(r)
    if not runs:
        print('#%d FAILED' % ci); continue
    med = {}
    for k in runs[0]:
        if isinstance(runs[0][k], tuple):
            med[k] = (st.median([x[k][0] for x in runs]), st.median([x[k][1] for x in runs]))
        else:
            med[k] = st.median([x[k] for x in runs])
    q = T.ratios(med, SPEED)
    bad = sorted([x for x,v in q.items() if v>=1.0 and x not in T.WATCH_ONLY])
    print('#%d n=%d cr=%.3f/%.3f hd=%.2f/%.2f comf=%.3f | lose:%s | %s' % (
        ci, len(runs), med['cross_rmse'], med['cross_peak'], med['head_rmse'],
        med['head_peak'], med['comf'], ','.join(bad), json.dumps(cfg)))
    res.append({'cfg':cfg,'med':med,'q':{k:round(v,4) for k,v in q.items()},'n':len(runs)})
if OUT: json.dump(res, open(OUT,'w'), indent=1)
