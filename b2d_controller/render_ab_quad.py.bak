r"""Side-by-side PID vs MPC-KF review video, as a 2x2 quad.

    +---------------------------+---------------------------+
    |  PID   6 cameras (2x3)    |  MPC   6 cameras (2x3)    |
    +---------------------------+---------------------------+
    |  PID   VAD BEV | control  |  MPC   VAD BEV | control  |
    +---------------------------+---------------------------+

Why a quad rather than two files played together: the two controllers diverge within a few
seconds of the same start, so "what did the other one do here?" is the whole question, and it
cannot be answered by scrubbing two windows to the same timestamp by hand. Both sides are indexed
by TICK, so row-for-row the two halves are the same instant of simulated time even though the runs
have different lengths and end at different places.

Four additions over render_vad_style_ctrl.py, all of them things the single-run video could not show:

  route overlay   The route planner's own path, reconstructed at 1 m resolution and projected onto
                  CAM_FRONT through the same lidar2img matrix VAD's planning overlay uses. Without
                  it the front view shows where the car went but not where it was supposed to go,
                  which is exactly the distinction every lane-departure question turns on.
  BEV legend      VAD's map vectors are drawn in six colours that mean six different things
                  (lane markings, centre line, traffic light, stop sign) and nothing in the frame
                  said which was which.
  collision flash The frame's border goes red on the side that just hit something. Collisions are
                  the single most important event in these runs and were previously invisible
                  unless you happened to be watching the right pixels.
  per-side status Tick, time and score banner per half, so a side that has already failed is
                  obvious rather than just frozen.

Requires the b2d_zoo environment (cv2 + mmcv, via render_vad_style.py). The route reconstruction
additionally needs CARLA's PythonAPI on sys.path; without it the video still renders, minus the
route overlay.

    python render_ab_quad.py --pid-dump <frames>/<pid_run> --mpc-dump <frames>/<mpc_run> \
                             --route 17569 --town Town12 --out review_ab.mp4
"""

import argparse
import importlib.util
import io
import json
import math
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

DEFAULT_RVS = "/home/ailab/2026intern/kmlee/vad_demo_video/render_vad_style.py"
DEFAULT_CTRL = "/home/ailab/carla_control/b2d_controller/render_vad_style_ctrl.py"
CARLA_API = "/home/ailab/2026intern/carla/PythonAPI/carla"
LEADERBOARD = "/home/ailab/2026intern/kmlee/vad_demo_video/Bench2Drive/leaderboard"
ROUTES_XML = LEADERBOARD + "/data/bench2drive220.xml"
XODR_DIRS = ("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{t}.xodr",
             "/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/{t}/OpenDrive/{t}.xodr")

# vad_b2d_agent's own self.lidar2img, all six cameras. Copied rather than recomputed: these are
# what VAD itself was fed, so anything projected through them lands where VAD thought it was.
LIDAR2IMG = {
 'CAM_FRONT': [[1142.51841, 800.0, 0.0, -952.0], [0.0, 450.0, -1142.51841, -809.704417],
               [0.0, 1.0, 0.0, -1.19], [0.0, 0.0, 0.0, 1.0]],
 'CAM_FRONT_LEFT': [[6.03961325e-14, 1394.75744, 0.0, -920.539908],
                    [-368.618420, 258.109396, -1142.51841, -647.296750],
                    [-0.819152044, 0.573576436, 0.0, -0.829094072], [0.0, 0.0, 0.0, 1.0]],
 'CAM_FRONT_RIGHT': [[1310.64327, -477.035138, 0.0, -406.010608],
                     [368.618420, 258.109396, -1142.51841, -647.296750],
                     [0.819152044, 0.573576436, 0.0, -0.829094072], [0.0, 0.0, 0.0, 1.0]],
 'CAM_BACK': [[-560.166031, -800.0, 0.0, -1288.0], [5.51091060e-14, -450.0, -560.166031, -858.939847],
              [1.22464680e-16, -1.0, 0.0, -1.61], [0.0, 0.0, 0.0, 1.0]],
 'CAM_BACK_LEFT': [[-1142.51841, 800.0, 0.0, -684.385123],
                   [-422.861679, -153.909064, -1142.51841, -496.004706],
                   [-0.939692621, -0.342020143, 0.0, -0.492889531], [0.0, 0.0, 0.0, 1.0]],
 'CAM_BACK_RIGHT': [[360.989788, -1347.23223, 0.0, -104.238127],
                    [422.861679, -153.909064, -1142.51841, -496.004706],
                    [0.939692621, -0.342020143, 0.0, -0.492889531], [0.0, 0.0, 0.0, 1.0]],
}
LIDAR2IMG = {k: np.array(v, dtype=float) for k, v in LIDAR2IMG.items()}

TICK_DT_S = 0.05
QUAD_W, CAM_H, BOT_H, BAR_H = 1400, 525, 525, 54
COLLISION_FLASH_TICKS = 12          # +-0.6 s around the impact
ROUTE_BGR = (60, 200, 255)          # amber; VAD's own plan is the blue-green 'winter' gradient
INK, MUTED, PANEL = (24, 24, 24), (140, 140, 140), (250, 250, 248)

_RVS = _CTRL = None
_STATE = {}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mods(rvs_path, ctrl_path):
    global _RVS, _CTRL
    if _RVS is None:
        _RVS = _load(rvs_path, "render_vad_style")
    if _CTRL is None:
        _CTRL = _load(ctrl_path, "render_vad_style_ctrl")
    return _RVS, _CTRL


# --------------------------------------------------------------------------- run data

def ego_track(dump_dir):
    """(N,2) world positions and (N,) world yaw in radians, per tick, from metric_info.json."""
    mi = json.load(open(osp.join(dump_dir, "metric_info.json")))
    ticks = sorted(mi, key=int)
    loc = np.array([mi[t]["location"][:2] for t in ticks], dtype=float)
    fwd = np.array([mi[t]["forward_vector"][:2] for t in ticks], dtype=float)
    return loc, np.arctan2(fwd[:, 1], fwd[:, 0])


def collision_ticks(dump_dir, loc):
    """Tick indices at which a collision was recorded.

    results.json gives collisions as world coordinates and no timestamp, so each one is matched to
    the tick whose ego position is closest to it. That is exact enough for a half-second flash and
    needs no extra instrumentation in the agent.
    """
    for cand in (osp.join(dump_dir, "..", "..", "results.json"),
                 osp.join(dump_dir, "..", "results.json")):
        if not osp.exists(cand):
            continue
        try:
            rec = json.load(open(cand))["_checkpoint"]["records"][0]
        except Exception:
            return [], None
        out = []
        for key in ("collisions_vehicle", "collisions_layout", "collisions_pedestrian"):
            for msg in rec["infractions"].get(key, []):
                nums = re.findall(r"[-+]?\d*\.?\d+", msg.split("(x=")[-1]) if "(x=" in msg else []
                if len(nums) >= 2 and len(loc):
                    p = np.array([float(nums[0]), float(nums[1])])
                    out.append(int(np.argmin(np.einsum("ij,ij->i", loc - p, loc - p))))
        return sorted(out), rec["scores"]
    return [], None


def build_route(route_id, town):
    """The route planner's own path at 1 m resolution, in world coordinates, or None.

    Rebuilt rather than read: what the agent is handed is downsample_route(..., 50), which for
    these routes is four to six points tens of metres apart -- unusable as an overlay. Tracing
    between them through the same GlobalRoutePlanner that built the route restores it exactly
    (measured: median 0.00 cm against the original).
    """
    try:
        if CARLA_API not in sys.path:
            sys.path.insert(0, CARLA_API)
        import xml.etree.ElementTree as ET
        import carla
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        xodr = next((p.format(t=town) for p in XODR_DIRS if osp.exists(p.format(t=town))), None)
        if xodr is None:
            print(f"[quad] {town}.xodr not found -- no route overlay")
            return None
        node = next((r for r in ET.parse(ROUTES_XML).getroot().findall("route")
                     if r.get("id") == str(route_id)), None)
        if node is None:
            return None
        keys = [carla.Location(x=float(p.get("x")), y=float(p.get("y")), z=float(p.get("z")))
                for p in node.find("waypoints").findall("position")]
        grp = GlobalRoutePlanner(carla.Map(town, open(xodr).read()), 1.0)
        pts = []
        for a, b in zip(keys[:-1], keys[1:]):
            pts += grp.trace_route(a, b)
        xy = np.array([[w.transform.location.x, w.transform.location.y] for w, _ in pts])

        # GlobalRoutePlanner represents a LANE CHANGE as a single discrete hop: the route stays on
        # one lane centre, then jumps ~3.5 m sideways in one step (route 17569 does exactly this at
        # index 37 -> 38, a 6.1 m step carrying -3.5 m of lateral offset). Projected as-is that
        # reads on camera as a right-angle kink and looks like a rendering fault. Filling the hops
        # in at the route's own 1 m spacing turns each into the diagonal it physically is. This is
        # a DISPLAY change only -- nothing else in the project consumes this polyline.
        step = np.hypot(*np.diff(xy, axis=0).T)
        gap = np.flatnonzero(step > 1.8)
        if len(gap):
            out = [xy[:gap[0] + 1]]
            for j, g in enumerate(gap):
                n = max(2, int(round(step[g])))
                t = np.linspace(0, 1, n + 1)[1:-1, None]
                out.append(xy[g] + t * (xy[g + 1] - xy[g]))
                end = gap[j + 1] + 1 if j + 1 < len(gap) else len(xy)
                out.append(xy[g + 1:end])
            xy = np.vstack(out)
        print(f"[quad] route {route_id} ({town}): {len(xy)} points, {len(keys)} keypoints, "
              f"{len(gap)} lane-change hop(s) filled")
        return xy
    except Exception as exc:
        print(f"[quad] route overlay unavailable ({type(exc).__name__}: {exc})")
        return None


# --------------------------------------------------------------------------- overlays

def _project(pts_lf, M, rvs, out_w, z=None):
    """(lateral, forward) ego points -> pixel polyline segments visible in this camera.

    Returns a list of (K,2) arrays: each is a run of consecutive points that are all IN FRONT of
    the camera. Splitting on the depth sign matters -- a polyline straddling the image plane
    otherwise gets a segment drawn straight across the frame between a point ahead and its mirror
    image behind, which looks like a real map vector and is not one.
    """
    n = len(pts_lf)
    if n < 2:
        return []
    z = rvs.GROUND_Z if z is None else z
    hom = np.concatenate([np.asarray(pts_lf, dtype=float),
                          np.full((n, 1), z), np.ones((n, 1))], axis=1)
    uvw = (M @ hom.T).T
    depth = uvw[:, 2]
    ok = depth > 0.3
    if not ok.any():
        return []
    uv = np.full((n, 2), np.nan)
    uv[ok] = uvw[ok, 0:2] / depth[ok, None]
    uv *= out_w / rvs.SRC_W
    runs, cur = [], []
    for i in range(n):
        if ok[i] and np.all(np.abs(uv[i]) < 1e5):
            cur.append(uv[i])
        elif len(cur) >= 2:
            runs.append(np.array(cur)); cur = []
        else:
            cur = []
    if len(cur) >= 2:
        runs.append(np.array(cur))
    return runs


def draw_map_vectors_on_cam(img, npz, cam, rvs):
    """VAD's predicted HD-map vectors, drawn into the camera they were predicted from.

    The BEV already shows these, but only in an abstract top-down frame where "is that Solid line
    actually on the lane edge?" cannot be answered. Projected back onto the image the prediction
    sits on top of the thing it claims to be, so a systematic offset or a hallucinated vector is
    visible directly. Colours are render_vad_style's own MAP_COLORS, so the BEV legend describes
    both panels at once.
    """
    from matplotlib.colors import to_rgb
    M = LIDAR2IMG.get(cam)
    if M is None:
        return img
    scores, labels, pts = npz["map_scores_3d"], npz["map_labels_3d"], npz["map_pts_3d"]
    for i in range(len(scores)):
        if scores[i] < rvs.MAP_CONF_TH:
            continue
        name = rvs.MAP_COLORS[int(labels[i]) % len(rvs.MAP_COLORS)]
        bgr = tuple(int(255 * v) for v in to_rgb(name))[::-1]
        for run in _project(pts[i], M, rvs, img.shape[1]):
            poly = np.round(run).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], False, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.polylines(img, [poly], False, bgr, 3, cv2.LINE_AA)
    return img


def draw_route_on_cam(img, route_xy, ego_p, ego_yaw, cam, rvs, ahead=60.0):
    """The route planner's path in whichever camera it falls into (not just CAM_FRONT)."""
    M = LIDAR2IMG.get(cam)
    if M is None or route_xy is None or not len(route_xy):
        return img
    d0 = route_xy - ego_p
    i0 = int(np.argmin(np.einsum("ij,ij->i", d0, d0)))
    seg = route_xy[max(0, i0 - 15):i0 + int(ahead) + 20]
    d = seg - ego_p
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    lf = np.stack([-d[:, 0] * s + d[:, 1] * c, d[:, 0] * c + d[:, 1] * s], axis=1)
    for run in _project(lf, M, rvs, img.shape[1]):
        poly = np.round(run).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], False, (0, 0, 0), 9, cv2.LINE_AA)
        cv2.polylines(img, [poly], False, ROUTE_BGR, 5, cv2.LINE_AA)
    return img


def draw_route_on_front(img, route_xy, ego_p, ego_yaw, rvs, ahead=60.0):
    """Project the route ahead of the ego onto CAM_FRONT, through VAD's own lidar2img.

    Same projection path as render_vad_style.draw_plan_on_front, so the route and VAD's predicted
    plan land in the same pixels when they agree -- which is what makes a disagreement legible.
    The ego frame here is (lateral, forward) with +lateral to the RIGHT, matching both CARLA's
    left-handed world and VAD's own out_truck convention.
    """
    if route_xy is None or not len(route_xy):
        return img
    # Walk the route in ITS OWN order from the point nearest the ego. Sorting the visible points
    # by forward distance instead looks equivalent on a straight road and is wrong the moment the
    # route curves: a bend puts two different stations at the same depth, and the polyline then
    # zig-zags between them instead of following the road.
    d0 = route_xy - ego_p
    i0 = int(np.argmin(np.einsum("ij,ij->i", d0, d0)))
    seg = route_xy[i0:i0 + int(ahead) + 20]
    d = seg - ego_p
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    fwd = d[:, 0] * c + d[:, 1] * s
    lat = -d[:, 0] * s + d[:, 1] * c
    keep = (fwd > 1.0) & (fwd < ahead) & (np.abs(lat) < 25.0)
    if keep.sum() < 2:
        return img
    # first contiguous run of visible points, so a later stretch of the same route coming back
    # into frame cannot be joined to this one by a straight chord across the scene
    idx = np.flatnonzero(keep)
    brk = np.flatnonzero(np.diff(idx) > 1)
    idx = idx[:brk[0] + 1] if len(brk) else idx
    lat, fwd = lat[idx], fwd[idx]

    pts = np.stack([lat, fwd, np.full(len(fwd), rvs.GROUND_Z), np.ones(len(fwd))], axis=1)
    uv = (rvs.LIDAR2IMG_FRONT @ pts.T).T
    depth = uv[:, 2]
    ok = depth > 1e-2
    uv = uv[ok, 0:2] / depth[ok, None]
    ok = (uv[:, 0] > -500) & (uv[:, 0] < rvs.SRC_W + 500) & (uv[:, 1] > 0) & (uv[:, 1] < rvs.SRC_H)
    uv = uv[ok] * (img.shape[1] / rvs.SRC_W)
    if len(uv) < 2:
        return img
    poly = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], False, (0, 0, 0), 9, cv2.LINE_AA)
    cv2.polylines(img, [poly], False, ROUTE_BGR, 5, cv2.LINE_AA)
    return img


def bev_legend(bev, rvs):
    """Name the BEV's colours. Drawn onto a widened canvas so nothing covers the scene itself."""
    from matplotlib.colors import to_rgb
    pad = 250
    out = cv2.copyMakeBorder(bev, 0, 0, 0, pad, cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
    x0, y = bev.shape[1] + 14, 30
    cv2.putText(out, "MAP VECTORS", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, INK, 1, cv2.LINE_AA)
    y += 22
    rows = [(n, tuple(int(255 * v) for v in to_rgb(c))[::-1])
            for n, c in zip(rvs.MAP_CLASSES, rvs.MAP_COLORS)]
    rows += [("", None),
             ("agent + forecast", tuple(int(255 * v) for v in to_rgb("tomato"))[::-1]),
             ("ego", tuple(int(255 * v) for v in to_rgb("mediumseagreen"))[::-1]),
             ("VAD plan", (200, 190, 60)),
             ("route planner", ROUTE_BGR)]
    for name, bgr in rows:
        if not name:
            y += 10
            continue
        cv2.line(out, (x0, y - 4), (x0 + 26, y - 4), bgr, 4, cv2.LINE_AA)
        cv2.putText(out, name, (x0 + 34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, INK, 1, cv2.LINE_AA)
        y += 24
    return out


def _letterbox(img, w, h, bg=PANEL):
    """Fit into (w, h) without distorting -- the control panel is text and must not be stretched."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), bg, np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


# --------------------------------------------------------------------------- one half

def render_half(side, idx):
    """One controller's quadrant at tick idx, or a frozen last frame once its run has ended."""
    st = _STATE[side]
    rvs, ctrl = _mods(st["rvs"], st["ctrl"])
    dump = st["dump"]
    live = idx in st["have"]
    use = idx if live else st["last"]

    npz = np.load(osp.join(dump, "pred", "%04d.npz" % use))
    cams = []
    for cam in rvs.CAMS:
        img = cv2.imread(osp.join(dump, cam, "%04d.jpg" % use))
        img = cv2.resize(img, (rvs.CAM_W, rvs.CAM_H))
        # map vectors first (they are the background layer), then the route, then VAD's own plan
        # on top -- so where all three coincide the plan stays readable.
        img = draw_map_vectors_on_cam(img, npz, cam, rvs)
        if use < len(st["loc"]):
            img = draw_route_on_cam(img, st["route"], st["loc"][use], st["yaw"][use], cam, rvs)
        if cam == "CAM_FRONT":
            img = rvs.draw_plan_on_front(img, npz)
        cv2.putText(img, cam, (12, 34), 0, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, cam, (12, 34), 0, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cams.append(img)
    grid = cv2.vconcat([cv2.hconcat(cams[0:3]), cv2.hconcat(cams[3:6])])
    grid = cv2.resize(grid, (QUAD_W, CAM_H), interpolation=cv2.INTER_AREA)

    bev = bev_legend(rvs.render_bev(npz), rvs)
    cv2.putText(bev, rvs.CMD_LIST[int(npz["command"])], (16, bev.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    bev = _letterbox(bev, int(QUAD_W * 0.52), BOT_H, bg=(255, 255, 255))

    with open(osp.join(dump, "meta", "%04d.json" % use)) as f:
        meta = json.load(f)
    panel = _letterbox(ctrl.control_panel(meta, use, st["hist"]), QUAD_W - bev.shape[1], BOT_H)
    quad = cv2.vconcat([grid, cv2.hconcat([bev, panel])])

    # title bar: which controller, where it is in time, and its final score
    bar = np.full((BAR_H, QUAD_W, 3), (32, 32, 32), np.uint8)
    cv2.putText(bar, st["label"], (16, 37), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    tail = "" if live else "   (run ended)"
    cv2.putText(bar, f"tick {use:04d}   t = {use * TICK_DT_S:6.2f} s{tail}", (330, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1, cv2.LINE_AA)
    if st["scores"]:
        cv2.putText(bar, f"DS {st['scores']['score_composed']:.2f}", (QUAD_W - 190, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (170, 230, 170), 2, cv2.LINE_AA)
    quad = cv2.vconcat([bar, quad])

    hit = any(abs(idx - c) <= COLLISION_FLASH_TICKS for c in st["hits"])
    if hit:
        cv2.rectangle(quad, (0, 0), (quad.shape[1] - 1, quad.shape[0] - 1), (60, 60, 235), 18)
        cv2.putText(quad, "COLLISION", (QUAD_W // 2 - 110, BAR_H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 235), 2, cv2.LINE_AA)
    return quad


def _init_worker(state):
    global _STATE
    _STATE = state


def render_frame(idx):
    right = render_half("mpc", idx)
    if "pid" not in _STATE:
        return right
    left = render_half("pid", idx)
    return cv2.hconcat([left, cv2.copyMakeBorder(right, 0, 0, 4, 0,
                                                 cv2.BORDER_CONSTANT, None, value=(32, 32, 32))])


# --------------------------------------------------------------------------- driver

def _ffmpeg():
    for c in ("/home/ailab/miniconda3/envs/pdm/bin/ffmpeg", "ffmpeg"):
        if osp.isabs(c) and osp.exists(c):
            return c
        w = shutil.which(c)
        if w:
            return w
    return None


def to_h264(src, crf=24):
    ff = _ffmpeg()
    if not ff:
        print("[quad] ffmpeg not found -- leaving mp4v")
        return
    tmp = src + ".h264.mp4"
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                        "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", tmp], capture_output=True)
    if r.returncode == 0 and osp.exists(tmp):
        a, b = os.path.getsize(src), os.path.getsize(tmp)
        os.replace(tmp, src)
        print(f"[quad] transcoded to H.264: {a/1e6:.1f} MB -> {b/1e6:.1f} MB")
    else:
        print("[quad] H.264 transcode failed:", r.stderr.decode()[:300])


def side_state(dump, label, route_xy, rvs_path, ctrl_path):
    _, ctrl = _mods(rvs_path, ctrl_path)
    have = {int(f[:-4]) for f in os.listdir(osp.join(dump, "pred")) if f.endswith(".npz")}
    have &= {int(f[:-5]) for f in os.listdir(osp.join(dump, "meta")) if f.endswith(".json")}
    loc, yaw = ego_track(dump)
    hits, scores = collision_ticks(dump, loc)
    return dict(dump=dump, label=label, have=have, last=max(have), loc=loc, yaw=yaw,
                hits=hits, scores=scores, route=route_xy, hist=ctrl.load_history(dump),
                rvs=rvs_path, ctrl=ctrl_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid-dump", default=None,
                    help="omit to render the MPC side alone (half-width, same overlays)")
    ap.add_argument("--mpc-dump", required=True)
    ap.add_argument("--max-tick", type=int, default=None,
                    help="stop here; useful when a run ends in a long stationary tail")
    ap.add_argument("--mpc-label", default="MPC-KF",
                    help="banner text for the right/only side -- this renderer is also used for "
                         "runs that are not MPC at all (e.g. stock PID with a different "
                         "checkpoint), and a wrong banner on a saved video is a real trap")
    ap.add_argument("--pid-label", default="PID (stock)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--route", default=None, help="route id, for the route-planner overlay")
    ap.add_argument("--town", default=None, help="town name; inferred from the dump path if absent")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rvs", default=DEFAULT_RVS)
    ap.add_argument("--ctrl", default=DEFAULT_CTRL)
    ap.add_argument("--no-h264", action="store_true")
    args = ap.parse_args()

    town = args.town
    if town is None:
        m = re.search(r"_(Town\d+\w*?)_", osp.basename(args.mpc_dump.rstrip("/")) + "_")
        town = m.group(1) if m else None
    route_id = args.route
    if route_id is None:
        m = re.search(r"RouteScenario_(\d+)_", args.mpc_dump)
        route_id = m.group(1) if m else None
    route_xy = build_route(route_id, town) if (route_id and town) else None

    state = {"mpc": side_state(args.mpc_dump, args.mpc_label, route_xy, args.rvs, args.ctrl)}
    if args.pid_dump:
        state["pid"] = side_state(args.pid_dump, args.pid_label, route_xy, args.rvs, args.ctrl)
    n = max(s["last"] for s in state.values())
    if args.max_tick is not None:
        n = min(n, args.max_tick)
    frames = [i for i in range(0, n + 1, args.stride)
              if any(i in s["have"] for s in state.values())]
    for k, s in state.items():
        print(f"[quad] {s['label']}: {s['last']+1} ticks, collisions {len(s['hits'])}")
    print(f"[quad] {len(frames)} frames"
          f"{' (single side)' if 'pid' not in state else ''}")

    _init_worker(state)
    probe = render_frame(frames[0])
    h, w = probe.shape[:2]
    os.makedirs(osp.dirname(osp.abspath(args.out)) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(state,)) as ex:
        for img in ex.map(render_frame, frames, chunksize=2):
            vw.write(img)
    vw.release()
    print(f"[quad] wrote {args.out}  ({len(frames)} frames, {w}x{h}, "
          f"{len(frames)/args.fps:.1f}s at {args.fps} fps)")
    if not args.no_h264:
        to_h264(args.out)


if __name__ == "__main__":
    main()
