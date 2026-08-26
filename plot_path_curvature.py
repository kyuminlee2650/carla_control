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


def corner_marks(s, kappa, min_abs=0.1, min_gap=3.0):
    """|kappa| >= min_abs 인 국소 최대점 전부의 인덱스.

    min_gap (m) 은 "같은 봉우리인가"만 거르는 값이지 코너를 솎아내는 값이 아니다 -- 스플라인
    위에서 한 봉우리의 꼭대기가 수치적으로 여러 개의 국소 최대로 쪼개지면 같은 자리에 라벨이
    겹쳐 찍히므로 그것만 합친다. 서로 다른 코너(이 경로에서는 5~8 m 떨어진 좌우 연속 코너가
    있다)는 남긴다. 큰 것부터 훑으므로 합쳐질 때 살아남는 쪽은 항상 더 급한 봉우리다."""
    a = np.abs(kappa)
    peaks = sorted((i for i in range(1, len(s) - 1)
                    if a[i] >= a[i - 1] and a[i] > a[i + 1] and a[i] >= min_abs),
                   key=lambda i: -a[i])
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
    ap.add_argument("--smoothing", type=float, default=1.0,
                    help="PathSpline 평활 계수 -- build_path_spline() 기본값과 같아야 제어기가 "
                         "보는 kappa 와 같은 곡선이 나온다")
    ap.add_argument("--mark-above", type=float, default=0.1,
                    help="|kappa| 가 이 값 이상인 봉우리마다 반경 R=1/|kappa| 라벨을 붙인다 "
                         "(1/m). 아주 크게 주면 라벨 없음")
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

    ax.axhline(0.0, color=V.COLOR_AXIS, linewidth=V.LINEWIDTH_THIN)

    ax.plot(s, kappa, "-", color=V.COLOR_BLUE, linewidth=V.LINEWIDTH)

    marks = corner_marks(s, kappa, min_abs=args.mark_above)
    print(f"Peaks with |kappa| >= {args.mark_above:g}: {len(marks)} -- "
          + ", ".join(f"s={s[i]:.0f} m R={1/abs(kappa[i]):.1f} m" for i in marks))
    for i in marks:
        ax.plot(s[i], kappa[i], "o", color=V.COLOR_ORANGE, markersize=6, zorder=5)
        ax.annotate(rf"$R$={1/abs(kappa[i]):.1f} m", xy=(s[i], kappa[i]),
                    xytext=(0, 12 if kappa[i] > 0 else -22), textcoords="offset points",
                    ha="center", fontsize=V.FONTSIZE_TICK - 2, color=V.COLOR_INK, zorder=6)

    ax.set_xlabel("station $s$ (m)")
    ax.set_ylabel(r"$\kappa$ (1/m)")
    ax.set_xlim(0.0, path.s_max)
    # 봉우리 라벨이 위아래로 12~22 pt 씩 삐져나오므로 데이터 범위에 여백을 준다
    lim = np.abs(kappa).max() * 1.28
    ax.set_ylim(-lim, lim)
    V._title(ax, "Path curvature along the route")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, V.run_name("path-curvature") + ".png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Figure saved: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
