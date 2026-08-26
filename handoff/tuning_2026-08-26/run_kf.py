import sys, os, json, random, math
sys.path.insert(0, '/tmp/claude-1000/-home-ailab-carla-control/09773ca3-6589-4edb-a014-3af4978602cf/scratchpad')
from evo import run, jitter, T

OUT = sys.argv[1]; N = int(sys.argv[2]); random.seed(int(sys.argv[3]))
SEEDFILE = sys.argv[4] if len(sys.argv) > 4 else None
JOINT = os.environ.get('JOINT', '0') == '1'
COMF_T = float(os.environ.get('COMF_T', '0.40'))
PEAK_T = float(os.environ.get('PEAK_T', '0.17'))
RMSE_T = float(os.environ.get('RMSE_T', '0.07'))
# 15 m/s 목표 (stage 3): mpc-kin/stanley 를 이겨야 하는 값. 환경변수로 넣는다.
P15 = float(os.environ.get('PEAK15', '0.30'))
R15 = float(os.environ.get('RMSE15', '0.12'))
C15 = float(os.environ.get('COMF15', '0.12'))

SPEC = dict(lat_np=(8,45,'int'), lat_nc=(2,20,'int'),
            w_ey=(5,4000,'f'), w_epsi=(0.05,400,'f'), w_ay=(0.0005,5,'f'),
            w_r=(0.02,200,'f'), w_rdot=(0.02,200,'f'),
            w_delta=(0.005,20,'f'), w_ddelta=(0.5,2000,'f'))
BASE = dict(lat_np=30, lat_nc=15, w_ey=122.2, w_epsi=2.4, w_ay=0.1,
            w_r=12.85, w_rdot=5.84, w_delta=0.3, w_ddelta=32.4)

def sc10(r):
    q = T.ratios(r, 10)
    hard = max(r['cross_peak']/PEAK_T, r['cross_rmse']/RMSE_T, COMF_T/max(r['comf'],1e-9))
    loss = sum(max(0.0, q[k]-1.0) for k in q if k not in T.WATCH_ONLY)
    blow = sum(max(0.0, q[k]-1.30) for k in T.WATCH_ONLY)
    mean = sum(q[k] for k in q if k not in T.WATCH_ONLY)/13.0
    return max(hard,1.0) + 0.25*loss + 0.10*mean + 0.2*blow, hard, q

def sc15(r):
    q = T.ratios(r, 15)
    hard = max(r['cross_peak']/P15, r['cross_rmse']/R15, C15/max(r['comf'],1e-9))
    loss = sum(max(0.0, q[k]-1.0) for k in q if k not in T.WATCH_ONLY)
    return max(hard,1.0) + 0.25*loss, hard, q

f = open(OUT, 'a', buffering=1)
pop = []

def ev(cfg):
    cfg = dict(cfg)
    if cfg['lat_nc'] > cfg['lat_np']: cfg['lat_nc'] = cfg['lat_np']
    r10 = run(10, 'mpc-kf', **cfg)
    if r10 is None:
        f.write(json.dumps({'kw':cfg,'fail':10})+"\n"); return
    s10, h10, q10 = sc10(r10)
    rec = {'kw':cfg,'r10':r10,'s10':round(s10,4),'h10':round(h10,4)}
    s = s10
    if JOINT:
        r15 = run(15, 'mpc-kf', **cfg)
        if r15 is None:
            rec['fail'] = 15; f.write(json.dumps(rec)+"\n"); return
        s15, h15, q15 = sc15(r15)
        rec.update(r15=r15, s15=round(s15,4), h15=round(h15,4))
        s = s10 + s15
    rec['s'] = round(s,4)
    f.write(json.dumps(rec)+"\n")
    pop.append((s, cfg)); pop.sort(key=lambda x: x[0]); del pop[12:]

seeds = [dict(BASE)]
if SEEDFILE:
    for l in open(SEEDFILE):
        seeds.append(json.loads(l))
for cfg in seeds[:N]:
    ev(cfg)
if not pop:
    ev(dict(BASE)); ev(dict(BASE))
done = len(seeds[:N])
for i in range(N - done):
    if not pop:
        cfg = {k:(random.randint(*SPEC[k][:2]) if SPEC[k][2]=='int'
                  else round(math.exp(random.uniform(math.log(SPEC[k][0]), math.log(SPEC[k][1]))),4))
               for k in SPEC}
    else:
        parent = random.choice(pop[:8])[1]
        cfg = jitter(parent, SPEC, random.choice([0.2, 0.4, 0.7]))
    ev(cfg)
f.close()
