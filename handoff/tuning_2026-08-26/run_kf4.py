"""Comfortness 마진을 두껍게 만드는 국소 탐색. 10 m/s 는 5회, 15 m/s 는 3회 중앙값으로
채점한다 -- 앞선 단계에서 같은 설정의 Comfortness 가 배치마다 한 구간씩 흔들려 순위가
뒤집히는 것을 확인했기 때문."""
import sys, json, random, statistics as st
sys.path.insert(0, '/tmp/claude-1000/-home-ailab-carla-control/09773ca3-6589-4edb-a014-3af4978602cf/scratchpad')
from evo import run, jitter, T

OUT = sys.argv[1]; N = int(sys.argv[2]); random.seed(int(sys.argv[3])); SEEDFILE = sys.argv[4]

SPEC = dict(lat_np=(12,40,'int'), lat_nc=(4,20,'int'),
            w_ey=(50,2000,'f'), w_epsi=(0.5,200,'f'), w_ay=(0.01,2,'f'),
            w_r=(0.3,50,'f'), w_rdot=(0.3,50,'f'),
            w_delta=(0.05,10,'f'), w_ddelta=(5,600,'f'))

def med(speed, cfg, reps):
    rs = [run(speed, 'mpc-kf', **cfg) for _ in range(reps)]
    rs = [r for r in rs if r is not None]
    if len(rs) < max(2, reps-1): return None
    out = {}
    for k in rs[0]:
        if isinstance(rs[0][k], tuple):
            out[k] = (st.median([x[k][0] for x in rs]), st.median([x[k][1] for x in rs]))
        else:
            out[k] = st.median([x[k] for x in rs])
    return out

def score(a, b):
    viol = (max(0.0, a['cross_peak']/0.165-1) + max(0.0, a['cross_rmse']/0.068-1)
            + max(0.0, b['cross_peak']/0.270-1) + max(0.0, b['cross_rmse']/0.098-1))
    return 5*viol + max(0.0,(0.45-a['comf'])/0.45) + 0.6*max(0.0,(0.26-b['comf'])/0.26)

f = open(OUT,'a',buffering=1); pop = []
def ev(cfg):
    cfg = dict(cfg)
    if cfg['lat_nc'] > cfg['lat_np']: cfg['lat_nc'] = cfg['lat_np']
    a = med(10, cfg, 5)
    if a is None: f.write(json.dumps({'kw':cfg,'fail':10})+"\n"); return
    b = med(15, cfg, 3)
    if b is None: f.write(json.dumps({'kw':cfg,'fail':15})+"\n"); return
    s = score(a,b)
    f.write(json.dumps({'kw':cfg,'m10':a,'m15':b,'s':round(s,4)})+"\n")
    pop.append((s,cfg)); pop.sort(key=lambda x:x[0]); del pop[8:]

seeds=[json.loads(l) for l in open(SEEDFILE) if l.strip()]
for c in seeds: ev(c)
for i in range(max(0,N-len(seeds))):
    if not pop: ev(dict(seeds[0])); continue
    ev(jitter(random.choice(pop[:5])[1], SPEC, random.choice([0.12,0.25,0.45])))
f.close()
