import sys, json, random, math
sys.path.insert(0, '/tmp/claude-1000/-home-ailab-carla-control/09773ca3-6589-4edb-a014-3af4978602cf/scratchpad')
from evo import run, jitter, T

OUT = sys.argv[1]; N = int(sys.argv[2]); random.seed(int(sys.argv[3]))
SEEDFILE = sys.argv[4] if len(sys.argv) > 4 else None

SPEC = dict(kin_np=(6,26,'int'), kin_nc=(2,9,'int'),
            kin_w_ey=(0.2,80,'f'), kin_w_epsi=(0.05,80,'f'), kin_w_r=(0.01,80,'f'),
            kin_w_delta=(0.02,40,'f'), kin_w_ddelta=(1,3000,'f'))
BASE = dict(kin_np=15, kin_nc=4, kin_w_ey=3.43, kin_w_epsi=2.0,
            kin_w_r=2.93, kin_w_delta=1.8, kin_w_ddelta=87.0)
ERR = ("cross_rmse","cross_peak","head_rmse","head_peak")

def score(r):
    q = T.ratios(r, 10)
    m4 = max(q[k] for k in ERR)
    pen = sum(max(0.0, q[k]-1.0) for k in q
              if k not in T.WATCH_ONLY and k not in ERR and k != 'comf')
    mean = sum(q[k] for k in q if k not in T.WATCH_ONLY)/13.0
    s = (3.0*max(0.0, m4-0.90) + 0.6*max(0.0, m4-0.75)
         + 2.0*max(0.0, q['comf']-0.95) + 0.5*pen + 0.1*mean)
    return s, q

f = open(OUT, 'a', buffering=1)
pop = []
def ev(cfg):
    cfg = dict(cfg)
    if cfg['kin_nc'] > cfg['kin_np']: cfg['kin_nc'] = cfg['kin_np']
    r = run(10, 'mpc-kin', **cfg)
    if r is None:
        f.write(json.dumps({'kw':cfg,'fail':True})+"\n"); return
    s, q = score(r)
    f.write(json.dumps({'v':2,'kw':cfg,'r':r,'s':round(s,4),
                        'q':{k:round(v,4) for k,v in q.items()}})+"\n")
    pop.append((s, cfg)); pop.sort(key=lambda x: x[0]); del pop[12:]

seeds = [dict(BASE)]
if SEEDFILE:
    seeds += [json.loads(l) for l in open(SEEDFILE) if l.strip()]
for cfg in seeds: ev(cfg)
if not pop: ev(dict(BASE)); ev(dict(BASE))
for i in range(max(0, N-len(seeds))):
    if not pop:
        cfg = {k:(random.randint(*SPEC[k][:2]) if SPEC[k][2]=='int'
                  else round(math.exp(random.uniform(math.log(SPEC[k][0]), math.log(SPEC[k][1]))),4))
               for k in SPEC}
    else:
        cfg = jitter(random.choice(pop[:8])[1], SPEC, random.choice([0.2,0.4,0.7]))
    ev(cfg)
f.close()
