"""Population perturbation search. 부모 후보를 골라 모든 가중치를 동시에 log-normal 로
흔든다 (좌표하강 금지). 느린 설정은 짧은 타임아웃으로 버려 처리량을 지킨다."""
import sys, os, re, json, random, math, subprocess
sys.path.insert(0, '/home/ailab/carla_control/handoff')
import tuning_harness as T

REPO = '/home/ailab/carla_control'
RUN_TIMEOUT = 45

def run(speed, ctrl, **kw):
    cmd = [T.PYTHON, "final_comparison.py", "--controller", ctrl,
           "--initial-speed", str(speed), "--times-run", "20"]
    for k, v in kw.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    try:
        o = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=RUN_TIMEOUT).stdout
    except subprocess.TimeoutExpired:
        return None
    if "Reached end of path" not in o or "TIMED OUT" in o:
        return None
    g = lambda p: float(re.search(p, o).group(1)) if re.search(p, o) else float("nan")
    r = dict(cross_rmse=g(r"cross-track\s+RMSE=\s*([\d.]+)"),
             cross_peak=g(r"cross-track[^\n]*max\|e\|=\s*([\d.]+)"),
             head_rmse=g(r"heading\s+RMSE=\s*([\d.]+)"),
             head_peak=g(r"heading[^\n]*max\|e\|=\s*([\d.]+)"),
             comf=g(r"Comfortness\s+([\d.]+)"))
    for key, pat in T.CH:
        r[key] = (g(pat + r"\s+mean\|\.\|=\s*([\d.]+)"),
                  g(pat + r"[^\n]*peak\|\.\|=\s*([\d.]+)"))
    return r

def jitter(cfg, spec, sigma):
    out = {}
    for k, v in cfg.items():
        lo, hi, kind = spec[k]
        if kind == 'int':
            nv = int(round(v * math.exp(random.gauss(0, sigma * 0.6))))
            nv += random.choice([-1, 0, 0, 1])
        else:
            nv = v * math.exp(random.gauss(0, sigma))
        nv = min(hi, max(lo, nv))
        out[k] = int(nv) if kind == 'int' else round(nv, 4)
    return out
