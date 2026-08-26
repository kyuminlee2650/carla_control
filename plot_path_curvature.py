"""이 저장소가 쓰는 경로의 곡률 kappa(s) 그림 하나.

경로는 final_comparison.py 와 똑같이 만든다 -- functions.build_path() 의 기본 origin/dest
스폰 인덱스로 GlobalRoutePlanner 루트를 뽑고, build_path_spline() 의 기본 smoothing=1.0 으로
호를 맞춘다. 따라서 여기 kappa 는 제어기가 실제로 프리뷰로 받아보는 그 kappa 다 (유한차분
스텐실이 아니라 스플라인의 해석적 2계 도함수, PathSpline docstring 참고).

서식(글꼴 크기/굵은 제목/팔레트)은 viz_utils 의 상수를 그대로 읽어 쓴다 -- 숫자를 여기
다시 적으면 viz_utils 를 고쳤을 때 이 그림만 조용히 어긋난다.
"""
import argparse, os
import numpy as np

import matplotlib
import viz_utils as V           # TkAgg 로 잡고 Times New Roman/STIX 를 세팅한다
from functions import build_path, build_path_spline

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_NAME = "Town10HD_Opt"       # final_comparison.py 와 동일하게 고정
SCORE_START_S = 5.0             # final_comparison.WARM_START_START_S -- 채점 시작 기점


def corner_marks(s, kappa, min_abs=0.008, min_gap=15.0):
    """|kappa| 국소 최대점 중 유의미한 것들의 (s, kappa) -- 코너마다 하나씩만 남기려고
    min_gap (m) 안에서는 가장 큰 것 하나만 취한다."""
    a = np.abs(kappa)
    peaks = [i for i in range(1, len(s) - 1)
             if a[i] >= a[i - 1] and a[i] > a[i + 1] and a[i] >= min_abs]
    peaks.sort(key=lambda i: -a[i])
    kept = []
    for i in peaks:
        if all(abs(s[i] - s[j]) >= min_gap for j in kept):
            kept.append(i)
    return sorted(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--ay-max", type=float, default=4.9,
                    help="곡률 속도 캡의 횡가속도 한계 (m/s^2) -- final_comparison.py 기본값과 "
                         "같게 두면 v=sqrt(ay_max/kappa) 가 실제로 걸리는 kappa 문턱선이 그려진다")
    ap.add_argument("--speeds", type=float, nargs="*", default=[10.0, 15.0],
                    help="문턱선을 그릴 주행 속도들 (m/s). 빈 목록이면 문턱선 없음")
    ap.add_argument("--smoothing", type=float, default=1.0,
                    help="PathSpline 평활 계수 -- build_path_spline() 기본값과 같아야 제어기가 "
                         "보는 kappa 와 같은 곡선이 나온다")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "plots"))
    ap.add_argument("--show", action="store_true", help="저장만 하지 말고 창으로도 띄운다")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != MAP_NAME:
        print(f"Loading {MAP_NAME}...")
        world = client.load_world(MAP_NAME)

    _, path_x, path_y = build_path(world)
    path = build_path_spline(path_x, path_y, smoothing=args.smoothing)
    s = np.linspace(0.0, path.s_max, 2000)
    kappa = path.kappa(s)
    print(f"Route: {len(path_x)} points, {path.s_max:.1f} m, "
          f"|kappa| max {np.abs(kappa).max():.4f} 1/m (R={1/np.abs(kappa).max():.1f} m)")

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(V.COLOR_BG)
    V._style_axes(ax)

    # 채점 구간 밖(출발 직후 5 m)은 흐리게 -- 지표는 여기서부터 쌓인다
    ax.axvspan(0.0, SCORE_START_S, color=V.COLOR_GRID, alpha=0.7, linewidth=0)
    ax.axhline(0.0, color=V.COLOR_AXIS, linewidth=V.LINEWIDTH_THIN)

    # 곡률 속도 캡이 걸리기 시작하는 kappa 문턱: v^2*kappa = ay_max
    for v, color, ls in zip(args.speeds,
                            (V.COLOR_AQUA, V.COLOR_RED, V.COLOR_PURPLE),
                            ("--", "-.", ":")):
        k_crit = args.ay_max / (v * v)
        for sign in (+1, -1):
            ax.axhline(sign * k_crit, color=color, linewidth=V.LINEWIDTH_THIN, linestyle=ls,
                       label=(rf"$|\kappa|=a_{{y,\max}}/v^2$, $v$={v:g} m/s" if sign > 0 else None))

    ax.plot(s, kappa, "-", color=V.COLOR_BLUE, linewidth=V.LINEWIDTH, label=r"path.kappa($s$)")

    for i in corner_marks(s, kappa):
        ax.plot(s[i], kappa[i], "o", color=V.COLOR_ORANGE, markersize=6, zorder=5)
        ax.annotate(rf"$R$={1/abs(kappa[i]):.0f} m",
                    xy=(s[i], kappa[i]),
                    xytext=(0, 11 if kappa[i] > 0 else -20), textcoords="offset points",
                    ha="center", fontsize=V.FONTSIZE_TICK - 2, color=V.COLOR_INK)

    ax.set_xlabel("station $s$ (m)")
    ax.set_ylabel(r"$\kappa$ (1/m)")
    ax.set_xlim(0.0, path.s_max)
    V._title(ax, "Path curvature along the route")
    V._legend(ax, loc="best", ncol=2)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, V.run_name("path-curvature") + ".png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Figure saved: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
