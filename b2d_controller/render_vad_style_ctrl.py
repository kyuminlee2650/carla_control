#!/usr/bin/env python
"""VAD-style closed-loop video with a live control-input panel.

Layout, left to right:

    [ 6 camera views, 2x3 ]   [ VAD BEV ]   [ control inputs ]
             2133x800            800x800         620x800

The first two columns are render_vad_style.py's own output, called into rather than reimplemented
-- same detections, same online map vectors, same motion forecasts, same 'winter' planning
gradient on the BEV and on CAM_FRONT, same VAD hard-coded thresholds. If that panel is ever
re-tuned, this video follows automatically.

The third column is new: what the controller was doing at that exact tick, read from meta/. It
branches on the dump's own schema -- MpcKfController logs delta_rad/a_cmd/kf_status, the stock
PIDController logs desired_speed/angle -- so the same command renders either run and the two are
directly comparable side by side.

Requires the b2d_zoo environment (cv2 + mmcv, which render_vad_style.py needs for
CustomNuscenesBox) and a dump that contains pred/ and CAM_*/ -- i.e. one written by the current
vad_b2d_agent.py or by vad_b2d_agent_vaddump.py.

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate b2d_zoo
    PYTHONPATH=<vad_demo_video>/Bench2DriveZoo:<vad_demo_video>/pydeps \
    python render_vad_style_ctrl.py --dump <frames>/<run> --out review_vad_ctrl.mp4
"""

import argparse
import importlib.util
import io
import json
import os
import os.path as osp
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TICK_DT_S = 0.05          # 20 Hz control loop, so tick index * 0.05 is the wall-clock time
PANEL_W, PANEL_H = 620, 800
DEFAULT_RVS = "/home/ailab/2026intern/kmlee/vad_demo_video/render_vad_style.py"

# cv2.VideoWriter can only be relied on for mp4v (MPEG-4 Part 2) here -- OpenCV wheels ship without
# an H.264 encoder for licensing reasons. mp4v plays in VLC but NOT in browsers, Notion, Slack or
# anything else that uses the platform's <video> decoder, so a raw render is effectively
# un-shareable. The finished file is therefore transcoded to H.264 as the last step. ffmpeg is not
# on PATH on this workstation; the pdm environment has a binary that is only ever invoked, never
# modified (see vad_demo_video/HANDOFF.md).
FFMPEG_CANDIDATES = ("/home/ailab/miniconda3/envs/pdm/bin/ffmpeg", "ffmpeg")

# Same house palette analyze_run_log.py/viz_utils use, in BGR for cv2 and hex for matplotlib.
INK, MUTED, BLUE, AQUA, RED, ORANGE = "#0b0b0b", "#898781", "#2a78d6", "#1baf7a", "#e34948", "#eb6834"

_RVS = None      # render_vad_style module, loaded once per process (see _load_rvs)
_HIST = None     # whole-run series for the rolling plot, set in the Pool initializer


def _load_rvs(path):
    global _RVS
    if _RVS is None:
        spec = importlib.util.spec_from_file_location("render_vad_style", path)
        _RVS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RVS)
    return _RVS


# --------------------------------------------------------------------------- run series

def load_history(dump_dir):
    """Every meta/*.json as arrays, for the rolling strip chart at the bottom of the panel.

    Loaded once in the parent and handed to the workers, rather than each worker re-reading ~800
    small JSON files per frame it renders.
    """
    paths = sorted(fp for fp in os.listdir(osp.join(dump_dir, "meta")) if fp.endswith(".json"))
    idx, recs = [], []
    for fp in paths:
        with open(osp.join(dump_dir, "meta", fp)) as f:
            recs.append(json.load(f))
        idx.append(int(fp[:-5]))
    get = lambda k: np.asarray([r.get(k, np.nan) for r in recs], dtype=float)
    return dict(idx=np.asarray(idx), t=np.asarray(idx) * TICK_DT_S,
                speed=get("speed"), throttle=get("throttle"), brake=get("brake"),
                steer=get("steer"), a_cmd=get("a_cmd"),
                is_mpc=bool(recs) and "delta_rad" in recs[0])


# --------------------------------------------------------------------------- control panel

def _bar(img, y, label, value, lo, hi, color_bgr):
    """One labelled horizontal bar, drawn straight onto the panel with cv2.

    Drawn rather than plotted because a matplotlib axis per bar costs more than the whole rest of
    the panel; the strip chart below is the only part that genuinely needs matplotlib.
    """
    x0, x1 = 250, PANEL_W - 30
    cv2.putText(img, f"{label}", (24, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, f"{value:+.3f}", (140, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (20, 20, 20), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y - 9), (x1, y + 9), (215, 215, 215), 1)
    zero = int(x0 + (0.0 - lo) / (hi - lo) * (x1 - x0))
    val = int(x0 + (np.clip(value, lo, hi) - lo) / (hi - lo) * (x1 - x0))
    if lo < 0:
        cv2.line(img, (zero, y - 9), (zero, y + 9), (170, 170, 170), 1)
    cv2.rectangle(img, (min(zero, val), y - 8), (max(zero, val), y + 8), color_bgr, -1)


def _strip_chart(hist, t_now, width, height, dpi=100):
    """Whole-run speed + pedal trace with a 'you are here' marker, as a BGR image."""
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#fcfcfb")
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax)
    for a in (ax, ax2):
        a.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(True, color="#e8e7e0", linewidth=0.7)
        a.tick_params(labelsize=7, colors=MUTED)
        a.axvline(t_now, color=RED, linewidth=1.4, zorder=5)

    ax.plot(hist["t"], hist["speed"], color=BLUE, linewidth=1.2)
    ax.set_ylabel("v (m/s)", fontsize=8, color=MUTED)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax2.plot(hist["t"], hist["throttle"], color=AQUA, linewidth=1.1, label="throttle")
    ax2.plot(hist["t"], hist["brake"], color=RED, linewidth=1.1, label="brake")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel("pedal", fontsize=8, color=MUTED)
    ax2.set_xlabel("t (s)", fontsize=8, color=MUTED)
    ax2.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def control_panel(meta, frame_idx, hist):
    img = np.full((PANEL_H, PANEL_W, 3), 252, np.uint8)
    t = frame_idx * TICK_DT_S
    cv2.putText(img, "CONTROL", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, f"tick {frame_idx:04d}   t = {t:6.2f} s", (24, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 120, 120), 1, cv2.LINE_AA)

    speed = float(meta.get("speed", float("nan")))
    cv2.putText(img, f"{speed:5.2f}", (24, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(img, "m/s", (185, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2, cv2.LINE_AA)

    _bar(img, 190, "throttle", float(meta.get("throttle", 0.0)), 0.0, 1.0, (122, 175, 27))   # AQUA
    _bar(img, 226, "brake", float(meta.get("brake", 0.0)), 0.0, 1.0, (72, 73, 227))          # RED
    _bar(img, 262, "steer", float(meta.get("steer", 0.0)), -1.0, 1.0, (214, 120, 42))        # BLUE

    y = 312
    if "delta_rad" in meta:
        # MpcKfController: the pedal layer tracks an ACCELERATION command, and the lateral MPC
        # emits a wheel angle -- neither has a stock-PID counterpart, so they only appear here.
        _bar(img, y, "a_cmd", float(meta.get("a_cmd", 0.0)), -4.05, 2.40, (52, 104, 235))    # ORANGE
        cv2.putText(img, "m/s2  (B2D limits)", (250, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 150, 150), 1, cv2.LINE_AA)
        y += 60
        _bar(img, y, "delta", float(np.degrees(meta["delta_rad"])), -30.0, 30.0, (214, 120, 42))
        cv2.putText(img, "deg, front wheel", (250, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 150, 150), 1, cv2.LINE_AA)
        y += 66
        st = str(meta.get("kf_status", "?"))
        ok = st in ("solved", "solved inaccurate")
        cv2.putText(img, f"lateral QP : {st}", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (20, 20, 20) if ok else (72, 73, 227), 1 if ok else 2, cv2.LINE_AA)
        y += 26
        cv2.putText(img, f"speed QP   : {meta.get('speed_mpc_status', '?')}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y += 26
        cv2.putText(img, f"gear {int(meta.get('gear', 0))}    pedal u = {meta.get('u_filtered', 0.0):+.3f}"
                         f"{'  SAT' if meta.get('pedal_saturated') else ''}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y += 34
    else:
        # stock PIDController schema
        cv2.putText(img, f"desired speed : {float(meta.get('desired_speed', float('nan'))):5.2f} m/s",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y += 28
        cv2.putText(img, f"aim angle     : {float(meta.get('angle', float('nan'))):+6.3f}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y += 40

    chart = _strip_chart(hist, t, PANEL_W - 20, PANEL_H - y - 20)
    ch, cw = chart.shape[:2]
    img[y:y + ch, 10:10 + cw] = chart
    return img


# --------------------------------------------------------------------------- frame assembly

def _find_ffmpeg():
    for cand in FFMPEG_CANDIDATES:
        path = cand if osp.isabs(cand) else shutil.which(cand)
        if path and os.access(path, os.X_OK):
            return path
    return None


def to_h264(src, crf=24, scale=None):
    """Transcode `src` in place to browser-playable H.264. Returns True on success.

    yuv420p and the High profile are what every consumer player expects; +faststart moves the
    index to the front so the file starts playing before it has fully downloaded. The source is
    replaced rather than written alongside, because two files differing only by codec is exactly
    the situation where the un-playable one gets shared by mistake.

    scale: optional output width. The panel text is sized for the full 4172 px canvas, so
    downscaling trades legibility for file size -- only worth it against a hard upload limit.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        print(f"[render] WARNING: no ffmpeg found in {FFMPEG_CANDIDATES}; leaving {src} as mp4v "
              f"(plays in VLC, will NOT play in a browser or Notion)")
        return False
    tmp = src + ".h264.mp4"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src,
           "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-crf", str(crf), "-preset", "medium", "-threads", "4",
           "-movflags", "+faststart"]
    if scale:
        # -2 keeps the height even, which H.264 requires.
        cmd += ["-vf", f"scale={int(scale)}:-2"]
    cmd += [tmp]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"[render] WARNING: H.264 transcode failed ({exc}); leaving {src} as mp4v")
        if osp.exists(tmp):
            os.remove(tmp)
        return False
    before, after = osp.getsize(src), osp.getsize(tmp)
    os.replace(tmp, src)
    print(f"[render] transcoded to H.264: {before/1e6:.1f} MB -> {after/1e6:.1f} MB"
          + (f", width {int(scale)}" if scale else ""))
    return True


def _init_worker(dump_dir, rvs_path, hist):
    global _HIST
    _HIST = hist
    _load_rvs(rvs_path)


def render_frame(idx, dump_dir, rvs_path):
    rvs = _load_rvs(rvs_path)
    base = rvs.render_frame(idx, dump_dir)        # [cameras | VAD BEV], 2933x800
    with open(osp.join(dump_dir, "meta", "%04d.json" % idx)) as f:
        meta = json.load(f)
    return cv2.hconcat([base, control_panel(meta, idx, _HIST)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="per-route dump directory (has pred/, CAM_*/, meta/)")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--render-vad-style", default=DEFAULT_RVS,
                    help="path to render_vad_style.py, whose panel this reuses")
    ap.add_argument("--stride", type=int, default=2,
                    help="keep every N-th tick (20 Hz sim; stride 2 + fps 10 = real time)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-h264", action="store_true",
                    help="skip the H.264 transcode and leave the raw mp4v file (plays in VLC, "
                         "not in a browser/Notion)")
    ap.add_argument("--crf", type=int, default=24,
                    help="H.264 quality, lower is better and larger (default 24)")
    ap.add_argument("--scale", type=int, default=None,
                    help="downscale to this output width, e.g. 1776 to halve it. Costs panel "
                         "legibility; use only against a hard upload size limit")
    args = ap.parse_args()

    hist = load_history(args.dump)
    print(f"[render] controller: {'MpcKfController' if hist['is_mpc'] else 'PIDController (stock)'}")

    frames = sorted(int(f[:-4]) for f in os.listdir(osp.join(args.dump, "pred"))
                    if f.endswith(".npz"))
    # Only ticks that have BOTH a prediction and a meta record can be rendered; mismatches mean a
    # partial dump (an interrupted run), and silently skipping them beats crashing 400 frames in.
    have_meta = {int(f[:-5]) for f in os.listdir(osp.join(args.dump, "meta")) if f.endswith(".json")}
    dropped = [i for i in frames if i not in have_meta]
    frames = [i for i in frames if i in have_meta][::args.stride]
    if dropped:
        print(f"[render] skipping {len(dropped)} tick(s) with pred/ but no meta/")
    print(f"[render] {len(frames)} frames from {args.dump}")

    size = (2933 + PANEL_W, 800)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size, True)
    worker = partial(render_frame, dump_dir=args.dump, rvs_path=args.render_vad_style)
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.dump, args.render_vad_style, hist)) as pool:
        for n, img in enumerate(pool.imap(worker, frames, chunksize=4)):
            writer.write(img)
            if n % 50 == 0:
                print("[render] %d/%d" % (n, len(frames)), flush=True)
    writer.release()
    print(f"[render] wrote {args.out}  ({len(frames)} frames, "
          f"{len(frames) / max(args.fps, 1):.1f}s at {args.fps} fps)")

    if not args.no_h264:
        to_h264(args.out, crf=args.crf, scale=args.scale)


if __name__ == "__main__":
    main()
