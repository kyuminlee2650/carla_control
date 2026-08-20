r"""Traffic Token closed-loop 실행을 영상으로 만든다.

기존 render_ab_quad.py 를 쓰지 않는 이유는 이 실험의 관심사가 다르기 때문이다. 거기서
보려던 것은 궤적과 지도 벡터였고, 여기서 보려는 것은 하나다 --

    실제 신호는 무엇이고, 모델은 그것을 무엇이라 보았고, 차는 움직였는가.

TT 에이전트가 rec/*.pkl.gz 에 매 틱 그 세 가지를 다 남긴다(gt_tl_state / p_state / speed).
그래서 카메라·BEV 위에 그 시계열을 겹쳐 놓으면 "초록불인데 안 간다" 가 한 화면에서 보인다.

레이아웃
    상단  카메라 6채널 2x3
    중단  VAD 자체 BEV (좌)  +  신호등 패널 (우)
    하단  시간축 -- 신호 상태 띠, P(green) 곡선, 속도

주의: 카메라 PNG 는 10 step 마다 한 장(frame = step // 10)이고 rec 는 매 step 이다.
프레임 f 에 대응하는 rec 구간은 [10f, 10f+10) 이고, 대표값으로 그 구간의 첫 행을 쓴다.
"""
import argparse
import glob
import gzip
import json
import os
import os.path as osp
import pickle
import subprocess
import tempfile

import cv2
import numpy as np

CAMS = [('rgb_front_left', 'FRONT_LEFT'), ('rgb_front', 'FRONT'), ('rgb_front_right', 'FRONT_RIGHT'),
        ('rgb_back_left', 'BACK_LEFT'), ('rgb_back', 'BACK'), ('rgb_back_right', 'BACK_RIGHT')]
W_OUT = 1600
CAM_W, CAM_H = W_OUT // 3, 180
MID_H, TL_W = 380, 520
TIME_H = 200
FONT = cv2.FONT_HERSHEY_SIMPLEX
# TT 에이전트의 라벨 (vad_b2d_agent_tt.py): 0=STOP 1=GREEN 255=무시
C_STOP, C_GREEN, C_IGN = (60, 60, 220), (80, 200, 90), (110, 110, 110)


def load_rec(run):
    f = sorted(glob.glob(osp.join(run, 'rec', '*.pkl.gz')))
    if not f:
        return []
    d = pickle.load(gzip.open(f[0], 'rb'))
    return d['rows'] if isinstance(d, dict) and 'rows' in d else d


def state_color(s):
    return C_STOP if s == 0 else (C_GREEN if s == 1 else C_IGN)


def state_text(s):
    return 'RED/STOP' if s == 0 else ('GREEN' if s == 1 else 'n/a')


def draw_panel(rows, i, res):
    """신호등 패널: 실제 상태 · 모델 예측 · 제어."""
    p = np.full((MID_H, TL_W, 3), 26, np.uint8)
    r = rows[i] if i < len(rows) else rows[-1]
    y = 34
    cv2.putText(p, 'TRAFFIC LIGHT', (18, y), FONT, 0.62, (200, 200, 200), 1, cv2.LINE_AA)
    y += 34

    st = r.get('gt_tl_state', 255)
    rel = r.get('gt_tl_relevant', 0)
    cv2.rectangle(p, (18, y - 18), (44, y + 4), state_color(st), -1)
    cv2.putText(p, 'ground truth  %s' % state_text(st), (56, y), FONT, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    y += 28
    d = r.get('gt_tl_dist', float('nan'))
    cv2.putText(p, 'relevant %d   dist %.1f m' % (rel, d if d == d else -1),
                (56, y), FONT, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
    y += 40

    ps = r.get('p_state') or []
    pg = float(ps[1]) if len(ps) > 1 else float('nan')
    cv2.putText(p, 'model  P(green)', (18, y), FONT, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    y += 22
    bw = TL_W - 40
    cv2.rectangle(p, (18, y), (18 + bw, y + 22), (55, 55, 55), -1)
    if pg == pg:
        w = int(bw * max(0.0, min(1.0, pg)))
        cv2.rectangle(p, (18, y), (18 + w, y + 22), C_GREEN if pg >= 0.5 else C_STOP, -1)
        cv2.putText(p, '%.3f' % pg, (18 + bw - 74, y + 17), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(p, (18 + bw // 2, y - 3), (18 + bw // 2, y + 25), (200, 200, 200), 1)
    y += 46

    pr = r.get('p_rel') or []
    prv = float(pr[1]) if len(pr) > 1 else float('nan')
    cv2.putText(p, 'model  P(relevant)  %s' % ('%.3f' % prv if prv == prv else '-'),
                (18, y), FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    y += 34

    cv2.putText(p, 'speed %5.2f m/s' % r.get('speed', 0.0), (18, y), FONT, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    y += 26
    for lab, key, col in (('throttle', 'throttle', (90, 200, 90)), ('brake', 'brake', (80, 80, 230))):
        v = float(r.get(key, 0.0))
        cv2.putText(p, lab, (18, y + 12), FONT, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.rectangle(p, (100, y), (100 + 300, y + 14), (55, 55, 55), -1)
        cv2.rectangle(p, (100, y), (100 + int(300 * min(1.0, v)), y + 14), col, -1)
        y += 22

    if res:
        y += 6
        cv2.putText(p, res, (18, y + 10), FONT, 0.46, (200, 200, 120), 1, cv2.LINE_AA)
    return p


def draw_time(rows, i):
    """시간축: 신호 상태 띠 + P(green) 곡선 + 속도."""
    t = np.full((TIME_H, W_OUT, 3), 22, np.uint8)
    n = len(rows)
    if n < 2:
        return t
    x = lambda k: int(20 + (W_OUT - 40) * k / (n - 1))

    cv2.putText(t, 'ground-truth light', (20, 18), FONT, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    for k in range(n - 1):
        cv2.rectangle(t, (x(k), 24), (x(k + 1) + 1, 44), state_color(rows[k].get('gt_tl_state', 255)), -1)

    cv2.putText(t, 'P(green)', (20, 66), FONT, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    y0, y1 = 72, 132
    cv2.line(t, (20, (y0 + y1) // 2), (W_OUT - 20, (y0 + y1) // 2), (70, 70, 70), 1)   # 0.5
    pts = []
    for k, r in enumerate(rows):
        ps = r.get('p_state') or []
        if len(ps) > 1:
            pts.append((x(k), int(y1 - (y1 - y0) * max(0.0, min(1.0, float(ps[1]))))))
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(t, a, b, (120, 210, 255), 1, cv2.LINE_AA)

    cv2.putText(t, 'speed', (20, 154), FONT, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    y0, y1 = 158, 192
    vmax = max(1.0, max(float(r.get('speed', 0.0)) for r in rows))
    pts = [(x(k), int(y1 - (y1 - y0) * float(r.get('speed', 0.0)) / vmax)) for k, r in enumerate(rows)]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(t, a, b, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(t, '%.0f' % vmax, (W_OUT - 46, y0 + 12), FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.line(t, (x(i), 20), (x(i), TIME_H - 4), (255, 220, 80), 1)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='out/<tag>/route_<id>')
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--fps', type=float, default=8.0)
    args = ap.parse_args()

    dumps = sorted(glob.glob(osp.join(args.run, 'frames', '*', '')))
    if not dumps:
        raise SystemExit('frames/ 없음: %s' % args.run)
    d = dumps[0]
    rows = load_rec(args.run)
    if not rows:
        print('[tt] rec 없음 -- 신호등 패널 없이 진행')

    res = ''
    rj = glob.glob(osp.join(args.run, 'route_results', '*.json'))
    if rj:
        try:
            x = json.load(open(rj[0]))['_checkpoint']['records'][0]
            inf = {k: len(v) for k, v in x['infractions'].items() if v and k != 'min_speed_infractions'}
            res = '%s  DS %.2f  %s' % (x['status'], x['scores']['score_composed'], inf or '')
        except Exception:
            pass

    n = len(sorted(glob.glob(osp.join(d, 'rgb_front', '*.png'))))
    if n == 0:
        raise SystemExit('카메라 프레임 없음')
    H = 20 + CAM_H * 2 + MID_H + TIME_H
    tmp = tempfile.mkdtemp(prefix='tt_')
    print('[tt] %d 프레임, rec %d 행' % (n, len(rows)))

    for f in range(n):
        canvas = np.full((H, W_OUT, 3), 16, np.uint8)
        cv2.putText(canvas, '%s   frame %d/%d' % (args.label, f, n - 1), (18, 15),
                    FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        for j, (sub, name) in enumerate(CAMS):
            p = osp.join(d, sub, '%04d.png' % f)
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.resize(img, (CAM_W, CAM_H))
            cv2.putText(img, name, (6, 16), FONT, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            r0, c0 = 20 + (j // 3) * CAM_H, (j % 3) * CAM_W
            canvas[r0:r0 + CAM_H, c0:c0 + CAM_W] = img

        y0 = 20 + CAM_H * 2
        bev = cv2.imread(osp.join(d, 'bev', '%04d.png' % f))
        if bev is not None:
            bw = W_OUT - TL_W
            s = min(bw / bev.shape[1], MID_H / bev.shape[0])
            bev = cv2.resize(bev, (int(bev.shape[1] * s), int(bev.shape[0] * s)))
            canvas[y0:y0 + bev.shape[0], 0:bev.shape[1]] = bev
        if rows:
            i = min(f * 10, len(rows) - 1)
            canvas[y0:y0 + MID_H, W_OUT - TL_W:] = draw_panel(rows, i, res)
            canvas[y0 + MID_H:y0 + MID_H + TIME_H] = draw_time(rows, i)
        cv2.imwrite(osp.join(tmp, '%05d.jpg' % f), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    os.makedirs(osp.dirname(osp.abspath(args.out)) or '.', exist_ok=True)
    subprocess.run([os.environ.get('FFMPEG','ffmpeg'), '-y', '-loglevel', 'error', '-framerate', str(args.fps),
                    '-i', osp.join(tmp, '%05d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '23', args.out], check=True)
    subprocess.run(['rm', '-rf', tmp])
    print('[tt] 완료 -> %s (%.1f MB)' % (args.out, osp.getsize(args.out) / 1e6))


if __name__ == '__main__':
    main()
