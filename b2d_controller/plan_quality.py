r"""VAD의 주행 플랜 품질을 폐루프 덤프에서 직접 잰다. 시뮬레이터가 필요 없다.

두 지표를 함께 낸다. 하나만 보면 오해하기 쉽기 때문이다.

  L2 (자기일관성)   틱 t 의 플랜이 예측한 t+1s/2s/3s 위치 vs 그 시각의 실제 ego 위치.
                    VAD 논문의 오픈루프 planning L2 와 같은 형태지만, 여기서는 정답이
                    expert 궤적이 아니라 "이 제어기가 실제로 간 곳"이다. 폐루프에서는
                    차가 플랜을 따라가므로 이 값은 낙관적으로 나온다 -- 플래너가 엉뚱한
                    곳을 가리켜도 제어기가 그리로 가면 L2 는 작다. 그래서 이것만으로는
                    플래너 품질을 판정할 수 없고, 아래 지표와 반드시 같이 봐야 한다.

  |e_y| (외부 정답) 플랜 끝점을 route planner 의 경로에 투영한 횡편차. 경로는 제어기와
                    무관하게 정해져 있으므로 폐루프 되먹임에 오염되지 않는다. "플래너가
                    가려는 곳이 실제 도로 위인가"를 직접 묻는다.

사용:
    python plan_quality.py --run <run_dir> --route <id> --town <name> [--max-tick N]
    python plan_quality.py --compare <runA> <runB> --route <id> --town <name>
"""

import argparse
import glob
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

CARLA_API = "/home/ailab/2026intern/carla/PythonAPI/carla"
LEADERBOARD = "/home/ailab/2026intern/kmlee/vad_demo_video/Bench2Drive/leaderboard"
ROUTES_XML = LEADERBOARD + "/data/bench2drive220.xml"
XODR_DIRS = ("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{t}.xodr",
             "/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/{t}/OpenDrive/{t}.xodr")
TICK_DT = 0.05
WP_DT = 0.5          # VAD 웨이포인트 간격 (초)


def build_route(route_id, town):
    sys.path.insert(0, CARLA_API)
    sys.path.insert(0, LEADERBOARD)
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from leaderboard.utils.route_manipulation import downsample_route
    xodr = next((p.format(t=town) for p in XODR_DIRS if os.path.exists(p.format(t=town))), None)
    if xodr is None:
        raise SystemExit(f"{town}.xodr 없음")
    node = next(r for r in ET.parse(ROUTES_XML).getroot().findall("route")
                if r.get("id") == str(route_id))
    keys = [carla.Location(x=float(p.get("x")), y=float(p.get("y")), z=float(p.get("z")))
            for p in node.find("waypoints").findall("position")]
    grp = GlobalRoutePlanner(carla.Map(town, open(xodr).read()), 1.0)
    pts = []
    for a, b in zip(keys[:-1], keys[1:]):
        pts += grp.trace_route(a, b)
    # 에이전트가 받는 것은 50 m 다운샘플본이므로 그 사이를 다시 이어 1 m 해상도를 복원한다
    tf = [(w.transform, o) for w, o in pts]
    ids = downsample_route(tf, 50)
    q = [tf[i][0].location for i in ids]
    dense = []
    for a, b in zip(q[:-1], q[1:]):
        dense += grp.trace_route(a, b)
    return np.array([[w.transform.location.x, w.transform.location.y] for w, _ in dense])


def project(route, p, hint=None, back=8.0, ahead=60.0):
    """(station, 부호 있는 횡편차). +는 경로 오른쪽. hint 로 탐색을 국소화한다."""
    d0 = route - p
    if hint is None:
        i = int(np.argmin(np.einsum("ij,ij->i", d0, d0)))
    else:
        lo, hi = max(0, hint - int(back)), min(len(route), hint + int(ahead))
        i = lo + int(np.argmin(np.einsum("ij,ij->i", d0[lo:hi], d0[lo:hi])))
    j = min(max(i, 1), len(route) - 1)
    t = route[j] - route[j - 1]
    psi = math.atan2(t[1], t[0])
    dv = p - route[i]
    return i, float(-dv[0] * math.sin(psi) + dv[1] * math.cos(psi))


def analyse(run_dir, route, max_tick=None):
    dumps = sorted(glob.glob(os.path.join(run_dir, "frames", "*", "")))
    if not dumps:
        dumps = [run_dir if run_dir.endswith("/") else run_dir + "/"]
    d = dumps[0]
    mi = json.load(open(d + "metric_info.json"))
    ticks = sorted(mi, key=int)
    loc = np.array([mi[t]["location"][:2] for t in ticks], dtype=float)
    fwd = np.array([mi[t]["forward_vector"][:2] for t in ticks], dtype=float)
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])

    metas = sorted(glob.glob(d + "meta/*.json"))
    n = min(len(metas), len(loc))
    if max_tick:
        n = min(n, max_tick)

    l2 = {1: [], 2: [], 3: []}
    plan_ey, ego_ey = [], []
    hint = None
    for i in range(n):
        m = json.load(open(metas[i]))
        pl = np.array(m.get("plan") or [], dtype=float)
        if pl.ndim != 2 or len(pl) < 6:
            continue
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        # VAD 는 [lateral, forward], +lateral = 우측. 월드로 옮긴다.
        world = np.stack([loc[i][0] + pl[:, 1] * c - pl[:, 0] * s,
                          loc[i][1] + pl[:, 1] * s + pl[:, 0] * c], axis=1)
        for sec in (1, 2, 3):
            k = int(sec / WP_DT) - 1                 # 1s -> idx1, 2s -> idx3, 3s -> idx5
            fut = i + int(sec / TICK_DT)
            if k < len(world) and fut < len(loc):
                l2[sec].append(float(np.hypot(*(world[k] - loc[fut]))))
        hint, e_ego = project(route, loc[i], hint)
        _, e_plan = project(route, world[-1], hint)
        ego_ey.append(abs(e_ego))
        plan_ey.append(abs(e_plan))
    return dict(n=len(plan_ey), l2={k: np.array(v) for k, v in l2.items()},
                plan_ey=np.array(plan_ey), ego_ey=np.array(ego_ey))


def onroute(r, thresh=0.5):
    """ego 가 경로 위(|e_y| < thresh)일 때의 플랜 편차만 추린다.

    플랜은 ego 프레임 기준이므로, 차가 이미 벗어나 있으면 플랜도 자동으로 벗어난 것으로
    집계된다. 그 교란을 빼야 "제어기가 제대로 데려다 놓았는데도 플래너가 엉뚱한 곳을
    가리키는가"를 볼 수 있다. 이 조건에서의 값이 플래너 자체의 품질이다.
    """
    m = r["ego_ey"] < thresh
    return m.sum(), (r["plan_ey"][m] if m.any() else np.array([np.nan]))


def show(tag, r):
    if not r["n"]:
        print(f"{tag:<26} 유효 프레임 없음")
        return
    q = lambda a: (np.median(a), np.percentile(a, 90), a.max())
    l2s = "  ".join(f"{s}s {np.mean(r['l2'][s]):5.2f}" if len(r["l2"][s]) else f"{s}s   -"
                    for s in (1, 2, 3))
    pm, p9, px = q(r["plan_ey"])
    em, e9, ex = q(r["ego_ey"])
    k, sub = onroute(r)
    print(f"{tag:<26} n={r['n']:4d} | L2 {l2s} | "
          f"플랜 |e_y| med {pm:5.2f} p90 {p9:5.2f} max {px:6.2f} | ego |e_y| med {em:5.2f}")
    print(f"{'':<26}   └ ego 가 경로 위(<0.5m)인 {k:4d}프레임만: "
          f"플랜 |e_y| med {np.nanmedian(sub):5.2f}  p90 {np.nanpercentile(sub,90):5.2f}  "
          f"max {np.nanmax(sub):6.2f}   |  >2m 인 비율 {100*np.nanmean(sub>2):4.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="run dir (여러 번 지정 가능)")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--route", required=True)
    ap.add_argument("--town", required=True)
    ap.add_argument("--max-tick", type=int, default=None)
    args = ap.parse_args()
    route = build_route(args.route, args.town)
    print(f"route {args.route} ({args.town}) 복원 {len(route)}점\n")
    print("L2 = 플랜 vs 실제 주행 (폐루프라 낙관적) | |e_y| = 경로 대비 횡편차 (외부 정답)\n")
    for i, run in enumerate(args.run):
        lab = args.label[i] if i < len(args.label) else os.path.basename(run.rstrip("/"))
        show(lab, analyse(run, route, args.max_tick))


if __name__ == "__main__":
    main()
