r"""라우트별 (지도 예측 이상) x (주행 실패) 교차표.

이 표가 답하려는 질문은 하나다 — dedup950 이 정답에 없는 SolidSolid 를 만들어내는 것이
주행 실패의 원인인가, 아니면 동반 증상인가.

  · 환각이 있는 라우트에서만 실패한다  -> 지도 분류 오류가 원인. 파인튜닝 대상이 명확해진다.
  · 환각이 없어도 실패한다             -> 다른 원인이 있고 SolidSolid 는 곁가지다.

route 2286 이 이미 후자를 가리키고 있어(환각 0인데 실패), 표본을 늘려 판정한다.

열 설명
  DS / 상태      results.json 그대로
  SS 예측        신뢰도 0.5 이상 SolidSolid 예측 수. PlanMapBoundLoss 가 경계로 인정하는
                 유일한 클래스이고 그 임계값이 0.5 이므로 같은 기준을 쓴다.
  SS 정답        OpenDRIVE 에서 ego 차선 좌우 표시에 SolidSolid 가 실제로 있는지.
                 예측이 "환각"인지 "정확한 검출"인지는 이것과 대조해야만 말할 수 있다.
  이동거리       속도 적분. 끼여서 멈춘 라우트를 완주와 구분한다.

사용:
    python crosstab.py --root <closedloop dir> --tag d950pid
    python crosstab.py --root <runs dir> --suffix _pidviz_d950 --local
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np

CARLA_API = "/home/ailab/2026intern/carla/PythonAPI/carla"
XODR_DIRS = ("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{t}.xodr",
             "/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/{t}/OpenDrive/{t}.xodr")
MAPC = ["Broken", "Solid", "SolidSolid", "Center", "TrafficLight", "StopSign"]
SS_IDX = 2          # PlanMapBoundLoss 의 lane_bound_cls_idx
MAP_THRESH = 0.5    # 같은 손실의 map_thresh
_MAPS = {}


def carla_map(town):
    if town not in _MAPS:
        sys.path.insert(0, CARLA_API)
        import carla
        p = next((q.format(t=town) for q in XODR_DIRS if os.path.exists(q.format(t=town))), None)
        _MAPS[town] = carla.Map(town, open(p).read()) if p else None
    return _MAPS[town]


def gt_solidsolid(town, locs, every=20):
    """ego 가 지나간 차선의 좌우 표시에 SolidSolid 가 실제로 있었는지."""
    cm = carla_map(town)
    if cm is None:
        return None, {}
    import carla
    c = Counter()
    for L in locs[::every]:
        w = cm.get_waypoint(carla.Location(x=float(L[0]), y=float(L[1]), z=float(L[2])),
                            lane_type=carla.LaneType.Driving)
        if w is None:
            continue
        for m in (w.left_lane_marking, w.right_lane_marking):
            if m is not None:
                c[str(m.type)] += 1
    return c.get("SolidSolid", 0), dict(c)


def scan(dump_dir):
    """한 라우트의 덤프에서 필요한 값만 뽑는다."""
    mi_path = os.path.join(dump_dir, "metric_info.json")
    if not os.path.exists(mi_path):
        return None
    mi = json.load(open(mi_path))
    ticks = sorted(mi, key=int)
    locs = np.array([mi[t]["location"] for t in ticks], dtype=float)

    ss = 0
    total = 0
    cls = Counter()
    for p in sorted(glob.glob(os.path.join(dump_dir, "pred", "*.npz")))[::2]:
        d = np.load(p)
        k = d["map_scores_3d"] >= MAP_THRESH
        total += int(k.sum())
        ss += int(((d["map_labels_3d"] == SS_IDX) & k).sum())
        for l in d["map_labels_3d"][k]:
            cls[MAPC[int(l) % 6]] += 1

    dist = np.nan
    metas = sorted(glob.glob(os.path.join(dump_dir, "meta", "*.json")))
    if metas:
        step = max(1, len(metas) // 400)
        v = [abs(json.load(open(q)).get("speed", 0.0)) for q in metas[::step]]
        dist = float(np.sum(v) * 0.05 * step)
    return dict(n_tick=len(ticks), locs=locs, ss=ss, total=total, cls=dict(cls), dist=dist)


def result_of(path):
    try:
        rec = json.load(open(path))["_checkpoint"]["records"][0]
    except Exception:
        return None
    s = rec["scores"]
    inf = {k: len(v) for k, v in rec["infractions"].items()
           if v and k != "min_speed_infractions"}
    return dict(status=rec["status"], ds=s["score_composed"], rc=s["score_route"],
                pen=s["score_penalty"], inf=inf)


def collect_cluster(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "route_*"))):
        r = os.path.basename(d).replace("route_", "")
        res = result_of(os.path.join(d, "route_results", f"{r}.json"))
        dumps = sorted(glob.glob(os.path.join(d, "frames", "*", "")))
        town = None
        if dumps:
            m = re.search(r"_(Town\d+\w*?)_", os.path.basename(dumps[0].rstrip("/")) + "_")
            town = m.group(1) if m else None
        out[r] = dict(res=res, dump=scan(dumps[0]) if dumps else None, town=town)
    return out


def collect_local(root, suffix):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, f"route*{suffix}"))):
        r = os.path.basename(d)[len("route"):-len(suffix)]
        res = result_of(os.path.join(d, "results.json"))
        dumps = sorted(glob.glob(os.path.join(d, "frames", "*", "")))
        town = None
        if dumps:
            m = re.search(r"_(Town\d+\w*?)_", os.path.basename(dumps[0].rstrip("/")) + "_")
            town = m.group(1) if m else None
        out[r] = dict(res=res, dump=scan(dumps[0]) if dumps else None, town=town)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--suffix", default="_pidviz_d950")
    ap.add_argument("--order", default="24367,27582,3144,2416,2715,2790,3373,1792,17569,2286")
    args = ap.parse_args()

    data = collect_local(args.root, args.suffix) if args.local else collect_cluster(args.root)
    order = [r for r in args.order.split(",") if r in data] + \
            [r for r in sorted(data) if r not in args.order.split(",")]

    print(f"{'route':>7} {'town':>9} {'DS':>7} {'RC%':>6} {'이동':>7} "
          f"{'SS예측':>7} {'SS비율':>7} {'SS정답':>7}  상태 / 위반")
    print("-" * 108)
    rows = []
    for r in order:
        e = data[r]
        res, dp = e["res"], e["dump"]
        if dp is None:
            print(f"{r:>7} {str(e['town']):>9}  (덤프 없음)")
            continue
        ssgt, marks = gt_solidsolid(e["town"], dp["locs"]) if e["town"] else (None, {})
        ratio = 100.0 * dp["ss"] / dp["total"] if dp["total"] else 0.0
        ds = f"{res['ds']:7.2f}" if res else "      -"
        rc = f"{res['rc']:6.1f}" if res else "     -"
        st = (res["status"][:26] + (" | " + ", ".join(f"{k}:{v}" for k, v in res["inf"].items())
                                    if res["inf"] else "")) if res else \
             ("결과없음 (정지로 중단)" if dp["dist"] < 30.0 else "결과없음 (진행중?)")
        gt = "없음" if ssgt == 0 else (f"{ssgt}개" if ssgt else "?")
        print(f"{r:>7} {str(e['town']):>9} {ds} {rc} {dp['dist']:6.0f}m "
              f"{dp['ss']:>7} {ratio:6.1f}% {gt:>7}  {st}")
        rows.append((r, res, dp, ssgt, ratio))

    print("-" * 108)
    # 실패 판정. results.json 이 없는 라우트를 그냥 빼면 안 된다 -- 끼여서 수동/워치독으로
    # 끊긴 실행이 정확히 그 상태이고, 그것이야말로 가장 심한 실패다. 덤프는 있는데 결과가
    # 없고 이동거리까지 짧으면 "정지"로 센다. 아직 도는 중인 것과 구분하려고 이동거리를 쓴다.
    STUCK_M = 30.0
    def verdict(x):
        res, dp = x[1], x[2]
        if res is not None:
            return "실패" if res["ds"] < 99.0 else "정상"
        if dp["dist"] < STUCK_M:
            return "정지"
        return "판정불가"

    hall = [x for x in rows if x[3] == 0 and x[2]["ss"] > 0]     # 정답 없는데 예측함
    clean = [x for x in rows if x[3] == 0 and x[2]["ss"] == 0]
    bad = lambda g: sum(1 for x in g if verdict(x) in ("실패", "정지"))
    und = lambda g: sum(1 for x in g if verdict(x) == "판정불가")
    print(f"SolidSolid 환각 있음 {len(hall):2d}개 중 실패/정지 {bad(hall):2d}개"
          f"  (판정불가 {und(hall)}개)")
    print(f"SolidSolid 환각 없음 {len(clean):2d}개 중 실패/정지 {bad(clean):2d}개"
          f"  (판정불가 {und(clean)}개)")
    print("\n환각 없이도 실패하는 라우트가 있으면 SolidSolid 는 단독 원인이 아니다.")


if __name__ == "__main__":
    main()
