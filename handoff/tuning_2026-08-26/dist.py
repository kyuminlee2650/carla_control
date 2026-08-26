import sys, json, statistics as st
sys.path.insert(0,'/tmp/claude-1000/-home-ailab-carla-control/09773ca3-6589-4edb-a014-3af4978602cf/scratchpad')
from evo import run
CTRL, SPEED, REPS, CFGS, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
cfgs=[json.loads(l) for l in open(CFGS) if l.strip()]
res=[]
for ci,cfg in enumerate(cfgs):
    rs=[r for r in (run(SPEED,CTRL,**cfg) for _ in range(REPS)) if r is not None]
    c=[r['comf'] for r in rs]; pk=[r['cross_peak'] for r in rs]; rm=[r['cross_rmse'] for r in rs]
    hr=[r['head_rmse'] for r in rs]; hp=[r['head_peak'] for r in rs]
    print('#%d n=%d comf mean=%.4f med=%.4f min=%.3f max=%.3f | rmse med=%.3f max=%.3f | peak med=%.3f max=%.3f | hd %.2f/%.2f | %s'%(
        ci,len(rs),st.mean(c),st.median(c),min(c),max(c),st.median(rm),max(rm),st.median(pk),max(pk),
        st.median(hr),st.median(hp),json.dumps(cfg)))
    res.append({'cfg':cfg,'comf':c,'peak':pk,'rmse':rm,'head_rmse':hr,'head_peak':hp})
json.dump(res,open(OUT,'w'))
