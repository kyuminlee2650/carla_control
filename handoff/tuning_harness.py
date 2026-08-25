import re, subprocess, sys, numpy as np
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_CACHE_DIR = os.path.join(REPO, "run_cache")
sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import viz_utils as V

CH = [("a_x","accel a_x"),("a_y","accel a_y"),("yaw_rate","yaw rate"),
      ("yaw_acc","yaw accel"),("jerk","jerk "),("jerk_tot",r"\|jerk\| total")]
def _stanley(v):
    d = np.load(os.path.join(RUN_CACHE_DIR, f"Stanley_{v}ms.npz"))
    h = {k: d[k].tolist() for k in d.files if not k.startswith("_")}
    m = V.add_scored_comfort_channels(h)
    ey, et = V.error_stats(h["e_y"]), V.error_stats(h["e_theta"])
    r = dict(cross_rmse=ey[0], cross_peak=ey[1], head_rmse=et[0], head_peak=et[1],
             comf=V.b2d_comfortness(h))
    for key, mk in (("a_x","a_x"),("a_y","a_y"),("yaw_rate","yaw_rate"),
                    ("yaw_acc","yaw_acc_scored"),("jerk","jerk_scored"),
                    ("jerk_tot","jerk_total_scored")):
        a = np.abs(np.asarray(m[mk], float))
        r[key] = (float(np.nanmean(a)), float(np.nanmax(a)))
    return r
REF = {v: _stanley(v) for v in (10, 15)}

# 인수인계 문서를 쓴 쪽 컴퓨터에는 REPO/.venv 가 있었지만 이 컴퓨터에는 없다 (carla/numpy/
# scipy 는 시스템 python3.10 에 들어 있고, 이 저장소는 cvxpy 를 안 쓴다). .venv 가 있으면
# 그대로 쓰고, 없으면 지금 이 모듈을 돌리는 인터프리터로 떨어진다 -- 양쪽 다 그대로 동작한다.
_VENV = os.path.join(REPO, ".venv/bin/python")
PYTHON = _VENV if os.path.exists(_VENV) else sys.executable


def run(speed, ctrl, **kw):
    cmd = [PYTHON,"final_comparison.py",
           "--controller",ctrl,"--initial-speed",str(speed),"--times-run","20"]
    for k,v in kw.items(): cmd += ["--"+k.replace("_","-"), str(v)]
    try:
        o = subprocess.run(cmd, cwd=REPO, capture_output=True,
                           text=True, timeout=600).stdout
    except subprocess.TimeoutExpired: return None
    if "Reached end of path" not in o: return None
    # 웜업이 타임아웃된 주행은 측정으로 쓸 수 없다. 그 경우 채점 구간이 공통 기점(s=80 m)이
    # 아니라 과도응답 한복판에서 시작해서, 구간 수부터 달라진다 -- 측정 사례: 정상 553 틱
    # (11.6 s 에 수렴, Comfortness 0.3333) 대 타임아웃 736 틱(v_x 가 3.73 m/s 까지 기어감,
    # |a_x| 3.83, Comfortness 0.3056). 같은 설정인데도 그렇다. 실패로 돌려서 재주행시킨다.
    if "TIMED OUT" in o: return None
    g = lambda p: float(re.search(p,o).group(1)) if re.search(p,o) else float("nan")
    r = dict(cross_rmse=g(r"cross-track\s+RMSE=\s*([\d.]+)"),
             cross_peak=g(r"cross-track[^\n]*max\|e\|=\s*([\d.]+)"),
             head_rmse=g(r"heading\s+RMSE=\s*([\d.]+)"),
             head_peak=g(r"heading[^\n]*max\|e\|=\s*([\d.]+)"),
             comf=g(r"Comfortness\s+([\d.]+)"))
    for key, pat in CH:
        r[key] = (g(pat+r"\s+mean\|\.\|=\s*([\d.]+)"), g(pat+r"[^\n]*peak\|\.\|=\s*([\d.]+)"))
    return r

def ratios(r, speed):
    """<1 이면 Stanley 우세. Comfortness 는 역수."""
    ref = REF[speed]; out = {}
    for k in ("cross_rmse","cross_peak","head_rmse","head_peak"):
        out[k] = r[k]/ref[k]
    out["comf"] = ref["comf"]/max(r["comf"],1e-9)
    for k,_ in CH:
        out[k+"_m"] = r[k][0]/ref[k][0]; out[k+"_p"] = r[k][1]/ref[k][1]
    return out

def summarize(label, cfgs):
    """두 속도 모두에서의 worst ratio"""
    worst = {}
    for sp in (10, 15):
        r = run(sp, cfgs["ctrl"], **cfgs.get("kw", {}))
        if r is None: return None
        q = ratios(r, sp); worst[sp] = (max(q.values()), q, r)
    return worst

# 향상 대상에서 제외하는 지표. a_x / lon jerk 는 세 컨트롤러가 공유하는 MpcLongitudinal 이
# 만들어내는 양이고, 횡방향 가중치로는 곡률 프리뷰를 통한 2차 효과밖에 못 미친다 (측정: 횡
# 가중치를 어떻게 바꿔도 a_x 비율이 1.02~1.16 에서 안 움직임). |jerk| 는 다르다 -- 크기의
# 미분이라 횡 성분이 섞이므로 횡 제어기가 실제로 움직일 수 있고, 최적화 대상으로 남긴다.
WATCH_ONLY = ("a_x_m", "a_x_p", "jerk_m", "jerk_p")
WATCH_BLOWUP = 1.30

def core_worst(q):
    core = max(v for k, v in q.items() if k not in WATCH_ONLY)
    blow = sum(max(0.0, q[k] - WATCH_BLOWUP) for k in WATCH_ONLY)
    return core + blow

def fmt_bad(q):
    bad = {k: v for k, v in q.items() if v >= 1.0 and k not in WATCH_ONLY}
    watch = {k: v for k, v in q.items() if k in WATCH_ONLY and v >= WATCH_BLOWUP}
    s = ", ".join("%s %.2f" % kv for kv in sorted(bad.items(), key=lambda x: -x[1])) or "없음"
    if watch:
        s += "  [참고지표 폭주: %s]" % ", ".join("%s %.2f" % kv for kv in watch.items())
    return s

ALLOW_FAIL = 3   # 횡방향 오차 외에 이만큼은 Stanley 에 져도 용인

def obj_allow(q):
    """13개 core 비율을 내림차순 정렬했을 때 (ALLOW_FAIL+1) 번째 값.
    이 값이 1.0 미만이면 '져도 되는 개수' 안에서 나머지는 전부 이겼다는 뜻."""
    core = sorted((v for k, v in q.items() if k not in WATCH_ONLY), reverse=True)
    blow = sum(max(0.0, q[k] - WATCH_BLOWUP) for k in WATCH_ONLY)
    return core[ALLOW_FAIL] + blow

def fmt_rank(q):
    core = sorted(((v, k) for k, v in q.items() if k not in WATCH_ONLY), reverse=True)
    return "  ".join("%s %.2f" % (k, v) for v, k in core[:5])
