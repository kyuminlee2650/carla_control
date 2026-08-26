"""반복 중앙값으로 채점하는 mpc-kf 최종 탐색. 구속조건(횡오차)은 여유를 두고 만족시키고
목적은 두 속도의 Comfortness 최대화 -- 단일 실행 Comfortness 는 구간 1개 차이로 0.03 씩
튀므로 중앙값이 아니면 순위가 노이즈로 뒤집힌다."""
import sys, json, random, math, statistics as st
sys.path.insert(0, '/tmp/claude-1000/-home-ailab-carla-control/09773ca3-6589-4edb-a014-3af4978602cf/scratchpad')
from evo import run, jitter, T

OUT = sys.argv[1]; N = int(sys.argv[2]); random.seed(int(sys.argv[3]))
SEEDFILE = sys.argv[4]; REPS = 3

SPEC = dict(lat_np=(10,40,'int'), lat_nc=(4,20,'int'),
            w_ey=(20,2000,'f'), w_epsi=(0.2,200,'f'), w_ay=(0.005,2,'f'),
            w_r=(0.2,50,'f'), w_rdot=(0.2,50,'f'),
            w_delta=(0.02,10,'f'), w_ddelta=(2,600,'f'))

def med(speed, cfg):
    rs = [run(speed, 'mpc-kf', **cfg) for _ in range(REPS)]
    rs = [r for r in rs if r is not None]
    if len(rs) < 2: return None
    out = {}
    for k in rs[0]:
        if isinstance(rs[0][k], tuple):
            out[k] = (st.median([x[k][0] for x in rs]), st.median([x[k][1] for x in rs]))
        else:
            out[k] = st.median([x[k] for x in rs])
    return out

def score(a, b):
    h10 = max(a['cross_peak']/0.15, a['cross_rmse']/0.065)
    h15 = max(b['cross_peak']/0.25, b['cross_rmse']/0.090)
    s = (4*max(0.0, h10-1) + 4*max(0.0, h15-1)
         + max(0.0, (0.45-a['comf'])/0.45) + max(0.0, (0.25-b['comf'])/0.25))
    return s, h10, h15

f = open(OUT, 'a', buffering=1)
pop = []
def ev(cfg):
    cfg = dict(cfg)
    if cfg['lat_nc'] > cfg['lat_np']: cfg['lat_nc'] = cfg['lat_np']
    a = med(10, cfg)
    if a is None: f.write(json.dumps({'kw':cfg,'fail':10})+"\n"); return
    b = med(15, cfg)
    if b is None: f.write(json.dumps({'kw':cfg,'fail':15})+"\n"); return
    s, h10, h15 = score(a, b)
    f.write(json.dumps({'kw':cfg,'m10':a,'m15':b,'s':round(s,4),
                        'h10':round(h10,3),'h15':round(h15,3)})+"\n")
    pop.append((s, cfg)); pop.sort(key=lambda x: x[0]); del pop[10:]

seeds = [json.loads(l) for l in open(SEEDFILE) if l.strip()]
for cfg in seeds: ev(cfg)
for i in range(max(0, N-len(seeds))):
    if not pop:
        ev(dict(seeds[0])); continue
    ev(jitter(random.choice(pop[:6])[1], SPEC, random.choice([0.15,0.3,0.55])))
f.close()
