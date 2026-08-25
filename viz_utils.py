"""Everything the path-tracking scripts do that isn't control: cameras, live view, metrics, plots.

Split out so the main scripts only contain the parts relevant to debugging the controller. The
former bev_view.py and video_recorder.py modules live here too -- one import for all of it.

Layout
    1. run identity         (run_name)
    2. palette + axis styling
    3. spectator camera     (follow_with_spectator)
    4. video recording      (VIEWS, VideoRecorder)
    5. live BEV view        (BevView)
    6. metrics              (rmse/max/mean, print_error_summary)
    7. plotting internals   (panel grids, series lookup, rmse badges)
    8. result figures       (plot_results)

Both VideoRecorder and BevView follow the same lifecycle: construct once after the vehicle is
spawned, call close() from the caller's finally block.

plot_results() draws three figures:
    fig 1  lateral tracking     -- cross-track error, heading error, yaw, yaw rate, yaw accel,
                                   v_y, a_y, steer
    fig 2  longitudinal tracking-- speed error, speed, a_x, jerk, total jerk magnitude,
                                   throttle + brake (combined)
    fig 3  trajectory           -- desired path vs. ego trajectory (equal aspect, so its own figure)

Panels whose series a given stack does not record are drawn as an explicit "not recorded" note
rather than crashing, so older scripts that log a smaller hist dict still plot.
"""
import datetime
import math
import multiprocessing as mp
import os
import signal
import queue
import subprocess
import sys
import threading
import time

import carla

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projection
import numpy as np

# Times New Roman body text, everywhere in this module -- falls back to a bundled serif on a
# machine without that font installed rather than erroring. mathtext.fontset="stix" (not the
# mpl default "dejavusans") is what actually matters for labels like $v_x$/$\dot\psi$: STIX is
# drawn to match a Times-like serif, so inline math doesn't visibly switch typeface mid-label the
# way DejaVu Sans math would next to Times New Roman prose.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"


# ---------------------------------------------------------------------------
# 1. run identity
# ---------------------------------------------------------------------------

# Stamped once at import, i.e. once per process, so every artifact of a run -- the figures and the
# mp4 -- carries the same <script>_<date>_<time> stem and they pair up in the output directories.
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_name(suffix="", name=None):
    """<script>_<date>_<time>[_suffix], the shared stem for this run's output files.

    name defaults to the script that was run, so stanley_PID.py writes stanley_PID_20260807_181500
    and a future controller names its own files without touching this module.
    """
    if name is None:
        main = sys.modules.get("__main__")
        path = getattr(main, "__file__", None) or sys.argv[0]
        name = os.path.splitext(os.path.basename(path))[0] or "run"
    stem = f"{name}_{RUN_STAMP}"
    return f"{stem}_{suffix}" if suffix else stem


# ---------------------------------------------------------------------------
# 2. palette + axis styling
# ---------------------------------------------------------------------------

# categorical slots from the house palette (references/palette.md), light mode
COLOR_BG = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_BLUE = "#2a78d6"      # actual / primary series
COLOR_ORANGE = "#eb6834"    # ego trajectory
COLOR_AQUA = "#1baf7a"      # throttle
COLOR_RED = "#e34948"       # brake / error
COLOR_PURPLE = "#8b5cf6"    # jerk / derived series

# cycled across runs when a plot function is handed {label: hist} instead of one hist -- index 0
# is COLOR_BLUE, so a single-run call still gets the same look it always had
COMPARE_COLORS = [COLOR_BLUE, COLOR_ORANGE, COLOR_AQUA, COLOR_RED, COLOR_PURPLE]

# Paired index-for-index with COMPARE_COLORS -- a multi-run panel tells runs apart by dash pattern
# as well as color, so it still reads once printed in grayscale or by someone color-blind to the
# blue/orange/aqua/red/purple set. (0, (3, 1, 1, 1)) is a dash-dot-dot, matplotlib's own on-off-tuple
# form since there's no named string for it the way "-"/"--"/"-."/":" cover the first four.
COMPARE_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# Shared stroke/type weights -- every plot function in this module should read these rather than
# hardcode its own numbers, so retuning one constant retunes every figure at once. Values match
# what was already hardcoded throughout this file before this was pulled out, so introducing the
# knob does not itself change how anything currently looks.
LINEWIDTH = 1.5          # primary data series
LINEWIDTH_THIN = 1.0     # reference lines: axhline/axvline, zero lines, gridlines
MARKERSIZE = 28          # scatter marker area (matplotlib's `s=`)
FONTSIZE_TITLE = 24      # figure suptitle
FONTSIZE_SUBTITLE = 20   # per-axes title
FONTSIZE_LABEL = 18      # axis labels
FONTSIZE_TICK = 16        # tick labels
FONTSIZE_LEGEND = 18      # legend text


def _style_axes(ax):
    ax.set_facecolor(COLOR_BG)
    ax.grid(True, color=COLOR_GRID, linewidth=LINEWIDTH_THIN * 0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=FONTSIZE_TICK)
    # title styling is NOT set here: Axes.set_title() unconditionally resets fontsize/fontweight/
    # color to matplotlib's rcParams defaults on every call (it applies its own `default` dict
    # before kwargs), so anything set on ax.title before the real set_title() call just gets
    # wiped out -- see _title() below, which every title in this file should go through instead.
    ax.xaxis.label.set_color(COLOR_MUTED)
    ax.yaxis.label.set_color(COLOR_MUTED)
    ax.xaxis.label.set_fontsize(FONTSIZE_LABEL)
    ax.yaxis.label.set_fontsize(FONTSIZE_LABEL)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def _title(ax, text, fontsize=FONTSIZE_SUBTITLE):
    """Every panel title in this file should be set through here, not ax.set_title() directly --
    see the note in _style_axes() for why setting title style any other way doesn't stick."""
    ax.set_title(text, fontsize=fontsize, fontweight="bold", color=COLOR_INK)


def _legend(ax, **kwargs):
    ax.legend(frameon=False, labelcolor=COLOR_INK, fontsize=FONTSIZE_LEGEND, **kwargs)


def _panel_handles(axes):
    """Every labeled artist across `axes`, first-come order, deduped by label -- the handle list a
    shared figure legend needs when the panels themselves already label their series (a per-panel
    legend would have read the same artists). matplotlib's own "_"-prefixed labels are skipped, as
    ax.legend() skips them.

    Panels that draw the same signal in the same role (several runs' "desired vel") collapse to one
    entry; that's the point of the dedup, and why order comes from the axes rather than a set.
    """
    handles, seen = [], set()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label in seen or label.startswith("_"):
                continue
            seen.add(label)
            handles.append(handle)
    return handles


def _bottom_legend(fig, handles, title=None, max_ncol=6):
    """Shared fig-level legend below every axes, sized to clear the bottom row's own xlabel
    instead of plot_comparison()'s fixed bbox_to_anchor=(0.5, -0.02) -- that offset only clears
    the xlabel when the figure has enough rows above it that one more legend row is a small
    fraction of the total height; a single-row figure (this file's newer per-speed/per-steer
    reports) needs noticeably more room, and more again once enough handles wrap the legend onto a
    second row. ncol is capped at max_ncol so a long handle list wraps instead of running off the
    figure edge or shrinking to unreadable size.
    """
    ncol = max(1, min(len(handles), max_ncol))
    rows = math.ceil(len(handles) / ncol)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
              labelcolor=COLOR_INK, fontsize=FONTSIZE_LEGEND,
              bbox_to_anchor=(0.5, -0.06 - 0.09 * rows), title=title)


# ---------------------------------------------------------------------------
# 3. spectator camera
# ---------------------------------------------------------------------------

# the framing of the CARLA 3D window, shared with VIEWS["chase"] so the two cannot drift apart
CHASE_BACK, CHASE_UP, CHASE_PITCH = 8.0, 4.0, -15.0


def follow_with_spectator(world, vehicle, back=CHASE_BACK, up=CHASE_UP, pitch=CHASE_PITCH):
    """Move the spectator to a 3rd-person chase view behind the vehicle."""
    transform = vehicle.get_transform()
    yaw = transform.rotation.yaw
    offset = carla.Location(
        x=-back * math.cos(math.radians(yaw)),
        y=-back * math.sin(math.radians(yaw)),
        z=up,
    )
    spectator_transform = carla.Transform(
        transform.location + offset,
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )
    world.get_spectator().set_transform(spectator_transform)


# ---------------------------------------------------------------------------
# 4. video recording
# ---------------------------------------------------------------------------

# Mounts are body-frame offsets from the actor origin, the same origin follow_with_spectator
# measures from -- so the chase view reproduces the CARLA 3D window and the others are the usual
# demo angles.
VIEWS = {
    "chase": (carla.Location(x=-CHASE_BACK, z=CHASE_UP), carla.Rotation(pitch=CHASE_PITCH)),
    "hood":  (carla.Location(x=1.2, z=1.3), carla.Rotation(pitch=0.0)),
    "front": (carla.Location(x=-5.5, z=2.2), carla.Rotation(pitch=-8.0)),
    "top":   (carla.Location(x=0.0, z=28.0), carla.Rotation(pitch=-90.0)),
}


class VideoRecorder:
    """Record the run to an mp4 through a CARLA RGB camera attached to the ego vehicle.

    Why a sensor and not a screen recorder: the control scripts run the world in synchronous mode,
    so the camera delivers exactly one image per world.tick() -- the video's frame rate is the
    simulation rate (1/dt), independent of how fast the loop actually runs on the wall clock. A run
    sped up with --times-run, or slowed down by a heavy MPC solve, still comes out as a correct
    real-time video.

    Encoding happens on a writer thread so the sensor callback (which runs on CARLA's own listener
    thread and must return fast) only pays for a BGRA->RGB copy. ffmpeg comes from imageio-ffmpeg,
    a static binary inside the venv -- nothing is installed system-wide.

        rec = VideoRecorder(world, vehicle, "drive.mp4", fps=1.0 / args.dt)
        ...
        rec.close()
    """

    def __init__(self, world, vehicle, out_path, fps=20.0, width=1280, height=720,
                 view="chase", fov=90.0, quality=6, preset="veryfast"):
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; pick one of {sorted(VIEWS)}")
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.out_path = out_path
        self.width, self.height = width, height
        self.frames = 0
        self._dropped = 0

        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(width))
        blueprint.set_attribute("image_size_y", str(height))
        blueprint.set_attribute("fov", str(fov))
        # no sensor_tick: leave it at 0 so the camera fires on every tick of the synchronous world

        location, rotation = VIEWS[view]
        # Rigid, not SpringArmGhost: a spring arm treats the offset as an arm and rotates it by the
        # mount's own pitch, so the chase view asking for 4 m up at -15 deg actually sat at 1.8 m
        # (and "top" ended up 28 m *behind* the car at ground level). Rigid places the camera
        # literally at the offset, which is what makes the recording match the spectator window.
        # The cost is that the camera now inherits the body's pitch and roll, so the horizon dips a
        # degree or two under braking where the spectator stays level.
        self.camera = world.spawn_actor(blueprint, carla.Transform(location, rotation),
                                        attach_to=vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)

        # 큐가 차면 프레임을 버리지 않고 센서 콜백을 그 자리에서 막는다 (queue_wait 까지). 버리면
        # 영상이 그 지점에서 시간이 압축되어 툭툭 끊겨 보이는데, 인코더는 어차피 고정 fps 로 쓰기
        # 때문에 유실분만큼 화면이 순간이동한다. 막으면 시뮬 루프가 인코더 속도에 맞춰 느려질 뿐
        # 영상은 온전하다 -- --times-run 20 처럼 20배속으로 돌릴 때 720p 인코딩이 못 따라와서
        # 생기던 문제(측정: 873프레임 중 173장 유실)의 실제 원인.
        self._queue = queue.Queue(maxsize=600)
        self._writer_thread = threading.Thread(target=self._write_loop, args=(fps, quality, preset),
                                               daemon=True)
        self._writer_thread.start()
        self.camera.listen(self._on_image)
        print(f"Recording to {out_path} ({width}x{height} @ {fps:.0f}fps, {view} view)")

    def _on_image(self, image):
        """Runs on CARLA's listener thread -- convert and hand off, nothing slow here.

        여기서 절대 블로킹하지 않는다. 한 번 blocking put 으로 바꿔봤다가 클라이언트가 통째로
        멎는 걸 확인했다 -- 이 콜백은 CARLA 자신의 리스너 스레드에서 돌기 때문에, 여기서 멈추면
        world.tick() 응답까지 함께 막혀 서버 타임아웃(TimeoutException)으로 죽는다. 역압은
        throttle() 로 메인 루프에서 건다.
        """
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        try:
            self._queue.put_nowait(bgra[:, :, [2, 1, 0]].tobytes())  # BGRA -> RGB24
        except queue.Full:
            self._dropped += 1

    def throttle(self, max_wait=30.0, high_water=0.5):
        """큐가 절반 넘게 차 있으면 빠질 때까지 메인 루프를 잡아둔다 -- 프레임 유실 방지의 핵심.

        매 world.tick() 마다 카메라가 정확히 한 장을 만들므로, 여기서 잠깐 멈춰 다음 tick 을
        미루면 인코더가 밀린 만큼 따라잡는다. 즉 시뮬이 인코더 속도까지만 빨라지고 영상은 온전히
        남는다 (--times-run 20 처럼 20배속으로 돌릴 때 720p 인코딩이 못 따라와 생기던 뚝뚝 끊김의
        해법). 이 대기는 반드시 메인 루프에서 해야 한다 -- _on_image 쪽 설명 참고.
        """
        if self.camera is None:
            return 0.0
        limit = max(1, int(self._queue.maxsize * high_water))
        start = time.time()
        while self._queue.qsize() > limit and time.time() - start < max_wait:
            time.sleep(0.005)
        return time.time() - start

    def _write_loop(self, fps, quality, preset):
        import imageio_ffmpeg

        # -preset 은 libx264 의 속도/압축률 손잡이다. imageio 기본값(medium)은 720p 를 실시간
        # 남짓으로밖에 못 뽑아서, 20배속 주행에서 큐가 차는 주된 이유였다. veryfast 는 파일이 조금
        # 커지는 대신 인코딩이 몇 배 빨라져 역압이 걸리는 구간 자체를 없앤다.
        writer = imageio_ffmpeg.write_frames(
            self.out_path, size=(self.width, self.height), fps=fps, quality=quality,
            macro_block_size=1, ffmpeg_log_level="error",
            output_params=["-preset", preset],
        )
        writer.send(None)  # seed the generator; this is what launches ffmpeg
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                writer.send(frame)
                self.frames += 1
        finally:
            writer.close()

    def close(self):
        if self.camera is not None and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()
        self.camera = None
        self._queue.put(None)          # sentinel: drains whatever is still queued, then closes ffmpeg
        self._writer_thread.join(timeout=60.0)
        if self._dropped:
            print(f"  warning: 인코더가 계속 밀려 {self._dropped} 프레임을 버렸습니다 "
                  f"(영상이 그 지점에서 끊깁니다) -- throttle() 호출이 빠졌거나 max_wait 초과")
        print(f"Recording done: {self.out_path} ({self.frames} frames)")
        return self.out_path


def _drawtext_font():
    """drawtext 가 쓸 TTF 경로. matplotlib 이 자기 폰트를 venv 안에 함께 깔기 때문에 시스템에
    폰트가 없어도 항상 하나는 있다. 못 찾으면 None -- 그때는 라벨 없이 합치기만 한다."""
    try:
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        return path if os.path.exists(path) else None
    except Exception:
        return None


def stack_videos_side_by_side(entries, out_path, fps, delete_inputs=True):
    """여러 주행 영상을 좌우로 이어붙여 하나의 mp4 로 만든다. 성공하면 out_path, 실패하면 None.

    entries: [(label, path, frames), ...] -- 왼쪽부터의 순서 그대로. 호출자가 --controller 에
    적은 순서로 넘겨주므로, 명령줄에 쓴 순서가 곧 화면 배치 순서가 된다.

    제어기들은 순차로 주행하므로 동시 녹화가 불가능하다. 그래서 각 시행을 따로 녹화해 두고
    여기서 합친다 -- 같은 시각의 두 주행이 아니라 "각자의 t=0 부터"를 나란히 놓은 것이라는 뜻
    이며, 두 주행의 길이가 다르면(한쪽이 먼저 완주하면) 짧은 쪽 마지막 프레임을 정지 화면으로
    늘려 끝을 맞춘다 (tpad=stop_mode=clone). 그냥 hstack 하면 짧은 쪽이 끝나는 순간 영상이
    통째로 끝나 긴 쪽의 남은 주행을 볼 수 없다.

    각 패널 상단에 제어기 이름을 얹는다 (drawtext). 폰트를 못 찾으면 라벨만 생략한다.
    """
    entries = [(lbl, path, fr) for lbl, path, fr in entries
               if path and os.path.exists(path) and fr > 0]
    if len(entries) < 2:
        return None

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        print(f"  ! 영상 합치기 실패: ffmpeg 를 찾지 못했습니다 ({exc})")
        return None

    font = _drawtext_font()
    longest = max(fr for _, _, fr in entries)

    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    for _, path, _ in entries:
        cmd += ["-i", path]

    chains = []
    for i, (label, _, frames) in enumerate(entries):
        steps = []
        pad = (longest - frames) / float(fps)
        if pad > 1e-3:
            steps.append(f"tpad=stop_mode=clone:stop_duration={pad:.3f}")
        if font:
            # 텍스트 안의 ' 와 : 는 필터 문법과 충돌하므로 미리 없앤다 (제어기 이름엔 없지만,
            # 임의의 label 이 들어와도 필터 그래프가 깨지지 않도록).
            safe = str(label).replace("'", "").replace(":", " ")
            steps.append(
                f"drawtext=fontfile='{font}':text='{safe}':fontcolor=white:fontsize=40"
                f":box=1:boxcolor=black@0.55:boxborderw=12:x=(w-text_w)/2:y=24")
        steps.append("setsar=1")
        chains.append(f"[{i}:v]" + ",".join(steps) + f"[v{i}]")

    inputs = "".join(f"[v{i}]" for i in range(len(entries)))
    graph = ";".join(chains) + f";{inputs}hstack=inputs={len(entries)}[out]"
    cmd += ["-filter_complex", graph, "-map", "[out]",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out_path]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    except Exception as exc:
        print(f"  ! 영상 합치기 실패 ({exc})")
        return None
    if proc.returncode != 0 or not os.path.exists(out_path):
        print(f"  ! 영상 합치기 실패 (ffmpeg exit {proc.returncode})")
        err = proc.stderr.decode("utf-8", "replace").strip()
        if err:
            print("    " + err.splitlines()[-1])
        return None

    order = " | ".join(lbl for lbl, _, _ in entries)
    print(f"영상 합치기 완료: {out_path}  (좌->우: {order})")
    if delete_inputs:
        for _, path, _ in entries:
            try:
                os.remove(path)
            except OSError:
                pass
        print(f"  개별 영상 {len(entries)}개는 삭제했습니다 (합쳐진 파일 하나만 남깁니다)")
    return out_path


def concat_videos_sequential(entries, out_path, delete_inputs=True):
    """여러 영상을 시간순으로 이어붙여 하나의 mp4 로 만든다. 성공하면 out_path, 실패하면 None.

    entries: [(label, path, frames), ...] -- 이어붙일 순서 그대로 (호출자가 시행한 순서).

    stack_videos_side_by_side() 는 여러 주행을 나란히 "동시 재생"하는 것이었고, 이건 그 반대로
    한 영상이 끝나면 다음 영상이 이어지는 "순차 재생"이다 -- 그래서 tpad 로 길이를 맞출 필요가
    없고, concat 필터로 그냥 순서대로 붙이면 된다. 각 구간 상단에 라벨(예: 그 구간에 가해진
    각충격량)을 얹는다 (drawtext). 폰트를 못 찾으면 라벨만 생략한다.
    """
    entries = [(lbl, path, fr) for lbl, path, fr in entries
               if path and os.path.exists(path) and fr > 0]
    if len(entries) < 2:
        return None

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        print(f"  ! 영상 이어붙이기 실패: ffmpeg 를 찾지 못했습니다 ({exc})")
        return None

    font = _drawtext_font()

    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    for _, path, _ in entries:
        cmd += ["-i", path]

    chains = []
    for i, (label, _, _) in enumerate(entries):
        steps = []
        if font:
            # 텍스트 안의 ' 와 : 는 필터 문법과 충돌하므로 미리 없앤다.
            safe = str(label).replace("'", "").replace(":", " ")
            steps.append(
                f"drawtext=fontfile='{font}':text='{safe}':fontcolor=white:fontsize=40"
                f":box=1:boxcolor=black@0.55:boxborderw=12:x=(w-text_w)/2:y=24")
        steps.append("setsar=1")
        chains.append(f"[{i}:v]" + ",".join(steps) + f"[v{i}]")

    inputs = "".join(f"[v{i}]" for i in range(len(entries)))
    graph = ";".join(chains) + f";{inputs}concat=n={len(entries)}:v=1:a=0[out]"
    cmd += ["-filter_complex", graph, "-map", "[out]",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out_path]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    except Exception as exc:
        print(f"  ! 영상 이어붙이기 실패 ({exc})")
        return None
    if proc.returncode != 0 or not os.path.exists(out_path):
        print(f"  ! 영상 이어붙이기 실패 (ffmpeg exit {proc.returncode})")
        err = proc.stderr.decode("utf-8", "replace").strip()
        if err:
            print("    " + err.splitlines()[-1])
        return None

    order = " -> ".join(lbl for lbl, _, _ in entries)
    print(f"영상 이어붙이기 완료: {out_path}  (순서: {order})")
    if delete_inputs:
        for _, path, _ in entries:
            try:
                os.remove(path)
            except OSError:
                pass
        print(f"  개별 영상 {len(entries)}개는 삭제했습니다 (합쳐진 파일 하나만 남깁니다)")
    return out_path


# ---------------------------------------------------------------------------
# 5. live BEV view
# ---------------------------------------------------------------------------

def _nearest_index(x, y, path_x, path_y, last_idx, search_window=30):
    """Same forward-only nearest-neighbor search as lateral_error() in the control script, kept as
    an independent copy here so BevView never has to read the control script's own last_idx."""
    lo = last_idx
    hi = min(len(path_x), last_idx + search_window)
    dists = [math.hypot(x - path_x[i], y - path_y[i]) for i in range(lo, hi)]
    return lo + dists.index(min(dists))


def _find_vehicle(world, near_x, near_y, max_dist=5.0, max_ticks=40):
    """world.get_actors() (bulk, unfiltered) only reflects a freshly-spawned actor after at least
    one tick has happened -- and the control script may not have called world.tick() yet by the
    time it constructs BevView. Since the world is already in synchronous mode by then, nudge it
    ourselves (a client other than the one driving the main loop is allowed to tick it too) until
    the actor list catches up."""
    for _ in range(max_ticks):
        best, best_d = None, None
        for actor in world.get_actors().filter("vehicle.*"):
            d = math.hypot(actor.get_location().x - near_x, actor.get_location().y - near_y)
            if best is None or d < best_d:
                best, best_d = actor, d
        if best is not None and best_d <= max_dist:
            return best
        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            time.sleep(0.1)
    raise RuntimeError("BevView: no vehicle found near the path start -- is it spawned yet?")


def _bev_process_main(vehicle_id, path_x, path_y, host, port, view_radius, trail_max_len,
                      redraw_interval_ms, stop_event, ready_event):
    """Entry point for the child process: owns its own CARLA connection and is this process's
    real main thread, so it's safe for matplotlib/Tk to live here for the process's whole life."""
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    vehicle = world.get_actor(vehicle_id)
    if vehicle is None:
        print(f"BevView: vehicle id {vehicle_id} not found in child process; exiting.")
        return

    lock = threading.Lock()
    trail_x, trail_y = [], []
    state = {"ego_xy": None, "last_idx": 0}

    def on_tick(snapshot):
        if stop_event.is_set() or not vehicle.is_alive:
            return
        try:
            loc = vehicle.get_location()
        except RuntimeError:
            return  # actor was destroyed between the is_alive check and this call
        x, y = loc.x, loc.y
        with lock:
            state["last_idx"] = _nearest_index(x, y, path_x, path_y, state["last_idx"])
            state["ego_xy"] = (x, y)
            trail_x.append(x)
            trail_y.append(y)
            if len(trail_x) > trail_max_len:
                del trail_x[: -trail_max_len]
                del trail_y[: -trail_max_len]

    tick_id = world.on_tick(on_tick)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(COLOR_BG)
    _style_axes(ax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")

    ax.plot(path_x, path_y, color=COLOR_MUTED, linewidth=1.5, linestyle="--", label="desired path")
    (trail_line,) = ax.plot([], [], color=COLOR_ORANGE, linewidth=2, label="ego trajectory")
    (ego_dot,) = ax.plot([], [], color=COLOR_BLUE, marker="o", markersize=9, linestyle="", label="ego")
    _legend(ax, loc="upper right")

    n_path = len(path_x)

    def redraw(_frame):
        if stop_event.is_set():
            plt.close(fig)
            return trail_line, ego_dot
        with lock:
            tx, ty = list(trail_x), list(trail_y)
            ego_xy = state["ego_xy"]
            last_idx = state["last_idx"]
        if ego_xy is None:
            return trail_line, ego_dot
        x, y = ego_xy
        trail_line.set_data(tx, ty)
        ego_dot.set_data([x], [y])
        _title(ax, f"last_idx = {last_idx}/{n_path - 1}")
        r = view_radius
        ax.set_xlim(x - r, x + r)
        ax.set_ylim(y - r, y + r)
        return trail_line, ego_dot

    anim = FuncAnimation(fig, redraw, interval=redraw_interval_ms, cache_frame_data=False)
    ready_event.set()
    plt.show()  # blocks until redraw() closes fig via stop_event

    try:
        world.remove_on_tick(tick_id)
    except Exception:
        pass


class BevView:
    """Live top-down (BEV) view of the desired path and the ego vehicle -- self-driving.

    Finds the spawned vehicle and refreshes itself every simulation tick via world.on_tick(); the
    control script only has to construct a BevView once and call close() -- no per-tick push needed.
    Draws exactly: desired path (static), ego trajectory (trail), current ego position, and last_idx
    (nearest desired-path index to the ego, recomputed here independently of whatever the control
    script tracks internally -- BevView never reads the control script's state).

    Process model: matplotlib/Tk GUI calls are only safe on a process's real main thread, so the
    whole plot lives in its own child process (started here, killed in close()) rather than a
    background thread of the caller's process. Inside that child process, world.on_tick() callbacks
    still arrive on CARLA's own internal listener thread, so a lock still guards the handoff to the
    redraw timer, which runs on the child process's main thread via plt.show().
    """

    def __init__(self, path_x, path_y, host="localhost", port=2000, view_radius=40.0,
                 trail_max_len=5000, redraw_interval_ms=50):
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
        vehicle = _find_vehicle(world, path_x[0], path_y[0])

        self._stop_event = mp.Event()
        ready_event = mp.Event()
        self._process = mp.Process(
            target=_bev_process_main,
            args=(vehicle.id, list(path_x), list(path_y), host, port, view_radius, trail_max_len,
                  redraw_interval_ms, self._stop_event, ready_event),
            daemon=True,
        )
        self._process.start()
        if not ready_event.wait(timeout=5.0):
            print("BevView: GUI window didn't come up within 5s, continuing anyway.")

    def close(self):
        self._stop_event.set()
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
        plt.ioff()  # the child leaves interactive mode on; plot_results()'s plt.show() must block


# ---------------------------------------------------------------------------
# 6. metrics
# ---------------------------------------------------------------------------

def error_stats(values):
    """(rmse, max|e|, mean) of a series. Returns None for an empty series."""
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    n = len(values)
    rmse = math.sqrt(sum(v * v for v in values) / n)
    return rmse, max(abs(v) for v in values), sum(values) / n


def magnitude_stats(values):
    """(mean|v|, peak|v|) of a series. Returns None for an empty series.

    For signals that oscillate around zero by design (acceleration, jerk, yaw rate) a signed
    mean/RMSE against a zero reference isn't the interesting number -- how big the swings
    typically are, and how big they get, is. error_stats() stays the right tool for anything
    that is itself a tracking error (e_y, e_theta, speed error).
    """
    values = [abs(v) for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return sum(values) / len(values), max(values)


def speed_error_series(hist, target_speed_ms):
    """Reference-minus-measured speed, using a logged per-step reference when there is one.

    The cruise scripts hold a constant target; the MPC stack sweeps v_des, so prefer hist["v_des"]
    whenever it exists -- scoring a swept reference against a constant is meaningless.
    """
    v_des = hist.get("v_des")
    if v_des:
        return [d - v for d, v in zip(v_des, hist["v_x"])]
    return [target_speed_ms - v for v in hist["v_x"]]


# B2D "Comfortness" / Driving Smoothness hard limits (2.4 Comfortness table) -- (lo, hi) per
# signal, in the same units print_error_summary already logs them in EXCEPT yaw_rate: hist logs
# that one in degrees (matches the rest of this file's dashboards), but B2D's own 0.95 rad/s limit
# is in radians, so that conversion happens at the comparison site in b2d_comfort_penalty() below,
# not here.
B2D_COMFORT_LIMITS = {
    "a_x":        (-4.05, 2.40),   # m/s^2 -- asymmetric: braking vs accelerating limits differ
    "a_y":        (-4.90, 4.90),   # m/s^2
    "yaw_rate":   (-0.95, 0.95),   # rad/s
    "yaw_acc":    (-1.93, 1.93),   # rad/s^2
    "jerk":       (-4.13, 4.13),   # m/s^3
    # Two-sided, matching the scorer's own check: _within_bound(magnitude_jerk, -8.37, +8.37).
    # It reads like a norm ("|jerk|") but the scored channel is d/dt(|accel|) -- the DERIVATIVE of a
    # magnitude, which goes negative freely (measured to -40 m/s^3 on a real run, with 70+ ticks
    # past -8.37 in a single 35 s drive). The old (0.0, 8.37) entry was right for this project's own
    # hist["jerk_total"] = hypot(jerk_x, jerk_y), a genuine non-negative norm, and drew only the
    # upper limit line -- which hid every negative-side violation once the panels switched to the
    # scored channel. For a non-negative signal the extra lower bound is inert, so _band_penalty()'s
    # use of this entry is unaffected.
    "jerk_total": (-8.37, 8.37),   # m/s^3
}

def _band_penalty(values, lo, hi):
    """Mean, band-width-normalized excess-outside-[lo,hi]: 0 if the signal never left the band,
    1 if it sat a full band-width past the limit for the entire run. Continuous rather than B2D's
    own 20-frame binary Smooth/not-Smooth segment scoring, which is fine as the paper's own
    reported number but is a flat, uninformative signal to tune against -- two configurations that
    are both "0% smooth" can still be very differently bad, and a pass/fail score can't tell them
    apart. None if `values` is empty (this hist never recorded the signal)."""
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    width = hi - lo
    excess = [max(0.0, v - hi) + max(0.0, lo - v) for v in values]
    return (sum(excess) / len(excess)) / width


def b2d_comfort_penalty(hist):
    """B2D Comfortness-style penalty terms for one run: {name: P}, P=0 perfect, growing
    unboundedly worse (no cap) the further/longer a signal sits outside its limit. "total" sums
    whatever terms this hist actually recorded (None for any that weren't, e.g. a steer=0 run has
    no a_y/yaw_rate/yaw_acc at all) -- route completion is deliberately not a term here since
    incomplete runs are eyeballed and thrown away rather than scored.

    Comfort/smoothness terms only (a_x/a_y/yaw_rate/yaw_acc/jerk/jerk_total) -- lateral_error and
    lap_time score tracking/route-progress, a different thing, and were dropped from this scoring
    entirely rather than just excluded from "total"."""
    hist = add_scored_comfort_channels(hist)
    terms = {}
    terms["a_x"] = _band_penalty(hist.get("a_x", []), *B2D_COMFORT_LIMITS["a_x"])
    terms["a_y"] = _band_penalty(hist.get("a_y", []), *B2D_COMFORT_LIMITS["a_y"])
    yaw_rate_rad = [math.radians(v) for v in hist.get("yaw_rate", [])]
    terms["yaw_rate"] = _band_penalty(yaw_rate_rad, *B2D_COMFORT_LIMITS["yaw_rate"])
    terms["yaw_acc"] = _band_penalty(hist.get("yaw_acc_scored", []), *B2D_COMFORT_LIMITS["yaw_acc"])
    terms["jerk"] = _band_penalty(hist.get("jerk_scored", []), *B2D_COMFORT_LIMITS["jerk"])
    terms["jerk_total"] = _band_penalty(hist.get("jerk_total_scored", []),
                                        *B2D_COMFORT_LIMITS["jerk_total"])

    available = [v for v in terms.values() if v is not None]
    terms["total"] = sum(available) if available else None
    return terms


# Bench2Drive 의 채점 모듈이 있는 디렉터리. b2d_controller/b2d_metrics.py 가 --tools-dir 기본값으로
# 쓰는 바로 그 경로이며, 두 곳이 같은 파일을 가리키도록 일부러 하드코딩 대신 여기 한 곳에 모아둔다.
#
# 기본값은 이 저장소에 함께 들어있는 사본(b2d_controller/comfort_metric/) -- 우분투 랩 머신의
# 절대경로였던 예전 기본값은 이 Windows 체크아웃에서는 애초에 존재하지 않는 경로라 항상 실패했다.
# os.path.join 이라 OS 에 무관하게 동작하므로 우분투에서도 그대로 쓸 수 있다. 이 사본은
# 2026-08-24 부터 랩 머신 원본(/home/ailab/2026intern/kmlee/vad_demo_video/Bench2Drive/tools,
# md5 0c65061599ff0f162777fca15ecf6dad)을 바이트 그대로 복사한 것이라 대시보드가 실제로 쓰는
# 파일과 동일하다 -- 경위와 검증 방법은 b2d_controller/comfort_metric/PROVENANCE.md 참고.
# 랩 머신 원본에서 직접 읽고 싶으면 $B2D_TOOLS_DIR 로 그 경로를 넘기면 된다 (같은 내용이 로드된다).
B2D_TOOLS_DIR = os.environ.get(
    "B2D_TOOLS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "b2d_controller", "comfort_metric"))


def b2d_comfortness(hist, tools_dir=None):
    """연구실 채점 방식 그대로의 Comfortness 점수 하나 (0~1, 높을수록 좋음). 못 구하면 None.

    계산을 여기서 다시 구현하지 않고 Bench2Drive 의 채점 모듈(efficiency_smoothness_benchmark.py)
    에서 seg_compute_comfort_metric 을 그대로 import 한다 -- b2d_controller/b2d_metrics.py 가
    쓰는 것과 같은 파일이라, 대시보드가 보고하는 숫자와 여기 숫자가 갈라질 수 없다. 정의는 20틱
    (1초) 고정 구간으로 잘라 여섯 채널(lon/lat 가속도, |jerk|, lon jerk, yaw 가속도, yaw rate)이
    전부 한계 안에 있는 구간의 비율이며, 채널별 감점을 더하던 이 파일의 예전 b2d_comfort_penalty()
    와는 다른 값이다 (그쪽은 연속적인 자체 지표였고, 이쪽은 실제 점수).

    hist 는 CARLA 의 metric_info.json 이 아니므로 그 함수가 받는 모양으로 맞춰 넣는다:

      acceleration  월드 프레임 (N,3). 차체 프레임 (a_x, a_y) 를 forward/right 벡터로 되돌린다.
                    채점 함수는 다시 forward/right 에 투영하므로 lon_acc/lat_acc 는 정확히
                    a_x/a_y 로 복원되고, |acc| 는 프레임과 무관하다 -- CARLA 좌표계 손잡이
                    규약에 의존하지 않는 유일한 구성이다.
      angular_velocity  (N,3), [:,2] 만 쓰이고 단위는 deg/s -- hist["yaw_rate"] 가 이미 deg/s.
      location/rotation 채점 함수가 첫 줄에서 버리는 인자. 길이만 맞춰 0 을 넣는다.

    가속도는 hist["a_x_raw"]/["a_y_raw"](원시 IMU)를 우선 쓴다. 채점 함수가 자기 Savitzky-Golay
    를 직접 걸기 때문에, 이미 저역통과된 hist["a_x"]/["a_y"] 를 넣으면 이중 평활이 되어 점수가
    실제보다 후하게 나온다. 원시 채널이 없는 예전 hist 는 필터본으로 폴백하되 그 사실을 알린다.
    """
    scorer, kwargs = _b2d_comfort_inputs(hist, tools_dir)
    if kwargs is None:
        return None
    try:
        return float(scorer.seg_compute_comfort_metric(**kwargs))
    except Exception as exc:
        print(f"  ! comfortness 계산 실패 ({exc})")
        return None


def _b2d_comfort_inputs(hist, tools_dir=None):
    """(채점 모듈, seg_compute_comfort_metric 에 넘길 kwargs) 또는 (모듈, None).

    b2d_comfortness() 와 b2d_comfort_report() 가 반드시 같은 입력을 보게 하려고 따로 뺐다 --
    두 곳이 각자 hist 를 풀면 한쪽만 고쳐졌을 때 "점수"와 "그 점수의 내역"이 조용히 어긋난다.
    입력 구성의 근거는 b2d_comfortness() docstring 참고.
    """
    tools_dir = tools_dir or B2D_TOOLS_DIR
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import efficiency_smoothness_benchmark as scorer
    except Exception as exc:
        print(f"  ! comfortness 계산 불가 -- {tools_dir} 에서 채점 모듈을 import 하지 못했습니다 ({exc})")
        return None, None

    yaw_deg = hist.get("yaw", [])
    a_x = hist.get("a_x_raw") or hist.get("a_x", [])
    a_y = hist.get("a_y_raw") or hist.get("a_y", [])
    n = min(len(yaw_deg), len(a_x), len(a_y), len(hist.get("yaw_rate", [])))
    if n < 2:
        return scorer, None

    yaw = np.radians(np.asarray(yaw_deg[:n], dtype=float))
    fwd = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(n)], axis=1)
    right = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros(n)], axis=1)
    accel = (np.asarray(a_x[:n], dtype=float)[:, None] * fwd
             + np.asarray(a_y[:n], dtype=float)[:, None] * right)
    ang_vel = np.zeros((n, 3))
    ang_vel[:, 2] = np.asarray(hist["yaw_rate"][:n], dtype=float)   # deg/s, 채점 함수가 직접 변환

    return scorer, dict(acceleration=accel, angular_velocity=ang_vel, forward_vector=fwd,
                        right_vector=right, location=np.zeros((n, 3)), rotation=np.zeros((n, 3)))


# (표시 이름, 상수 이름, 부호) -- compute_comfort_metric() 이 검사하는 여섯 채널의 한계를 채점
# 모듈에서 그대로 읽어오기 위한 표. 값을 여기 옮겨적지 않는 이유는 b2d_metrics 의 b2d_limits() 와
# 같다: 옮겨적는 순간 채점 모듈이 바뀌어도 이쪽은 모른 채로 남는다.
_COMFORT_CHANNELS = (
    ("lon_acc",  ("MIN_LON_ACCEL", "min_lon_accel"), ("MAX_LON_ACCEL", "max_lon_accel")),
    ("lat_acc",  ("MAX_ABS_LAT_ACCEL", "max_abs_lat_accel"), None),
    ("|jerk|",   ("MAX_ABS_MAG_JERK", "max_abs_mag_jerk"), None),
    ("lon_jerk", ("MAX_ABS_LON_JERK", "max_abs_lon_jerk"), None),
    ("yaw_acc",  ("MAX_ABS_YAW_ACCEL", "max_abs_yaw_accel"), None),
    ("yaw_rate", ("MAX_ABS_YAW_RATE", "max_abs_yaw_rate"), None),
)


def _comfort_channel_series(scorer, kw, start, stop, window_size=7, poly_order=2):
    """The six channels compute_comfort_metric() actually bound-checks, for one segment.

    Rebuilt from the scoring module's OWN _smooth()/constants rather than recomputed here, so this
    breakdown follows the module if it changes. It is still a second implementation of that
    function's body, which is exactly why b2d_comfort_report() cross-checks its own pass/fail count
    against the module's seg_compute_comfort_metric() and refuses to report a breakdown that
    disagrees -- a silent drift here would be worse than no breakdown at all.

    Returns {name: (values, lo, hi)} or None if the module doesn't expose what this needs.
    """
    try:
        smooth, dt = scorer._smooth, scorer.CARLA_TICK_SECONDS
    except AttributeError:
        return None

    accel = kw["acceleration"][start:stop]
    fwd, right = kw["forward_vector"][start:stop], kw["right_vector"][start:stop]
    window_size = min(window_size, len(accel))
    if poly_order >= window_size:
        return None

    lon = np.einsum("ij,ij->i", accel[:, :2], fwd[:, :2])
    lat = np.einsum("ij,ij->i", accel[:, :2], right[:, :2])
    mag = np.hypot(accel[:, 0], accel[:, 1])
    yaw_rate = np.deg2rad(kw["angular_velocity"][start:stop, 2])

    yaw_acc = smooth(yaw_rate, window_size, poly_order, deriv=1, delta=dt)
    yaw_rate = smooth(yaw_rate, window_size, poly_order)
    lon = smooth(lon, window_size, poly_order)
    lat = smooth(lat, window_size, poly_order)
    mag = smooth(mag, window_size, poly_order)
    mag_jerk = smooth(mag, window_size, poly_order, deriv=1, delta=dt)
    lon_jerk = smooth(lon, window_size, poly_order, deriv=1, delta=dt)

    series = dict(lon_acc=lon, lat_acc=lat, jerk=mag_jerk, lon_jerk=lon_jerk,
                  yaw_acc=yaw_acc, yaw_rate=yaw_rate)
    out = {}
    for (name, lo_names, hi_names), key in zip(_COMFORT_CHANNELS,
                                               ("lon_acc", "lat_acc", "jerk", "lon_jerk",
                                                "yaw_acc", "yaw_rate")):
        try:
            lo_v = next(getattr(scorer, n) for n in lo_names if hasattr(scorer, n))
        except StopIteration:
            return None
        if hi_names is None:                       # symmetric +/- bound from one constant
            lo, hi = -lo_v, lo_v
        else:
            hi = next(getattr(scorer, n) for n in hi_names if hasattr(scorer, n))
            lo = lo_v
        out[name] = (series[key], lo, hi)
    return out


# The three comfort channels the scoring module DIFFERENTIATES, and the hist keys the recomputed
# versions get stored under. Measured against real runs, the differentiated channels are the only
# ones where this project's own signal and the scored one disagree at all:
#
#     channel      corr(control-path, scored)   peak ratio
#     lon_acc                  1.000               1.00x     <- not differentiated
#     lat_acc                  1.000               1.00x     <- not differentiated
#     yaw_rate                 0.997               0.98x     <- not differentiated
#     yaw_acc                  0.780               1.70x     <- differentiated
#     lon_jerk                 0.708               1.51x     <- differentiated
#     |jerk|                   0.291               1.55x     <- differentiated
#
# Cause: this project derives them with a CAUSAL low-pass (ImuAcceleration's tau=0.15 filters,
# a matching tau=0.15 filter on yaw rate) because those ran inside a control loop, while the
# scorer uses a NON-CAUSAL
# Savitzky-Golay derivative over each 20-tick segment. Differentiation amplifies high frequency, so
# the filter choice dominates the result; the undifferentiated channels barely notice it.
#
# The scored version is consistently LARGER (1.19x-1.70x on the runs measured), which means a
# B2D_COMFORT_LIMITS line drawn over the control-path signal UNDER-reports violations: a curve can
# sit comfortably inside the red lines on a segment the score has already failed.
SCORED_COMFORT_KEYS = {"jerk": "jerk_scored", "jerk_total": "jerk_total_scored",
                       "yaw_acc": "yaw_acc_scored"}
_SCORED_SOURCE = {"jerk": "lon_jerk", "jerk_total": "|jerk|", "yaw_acc": "yaw_acc"}


def _scored_key(runs, key):
    """(hist key, title suffix) for a panel that carries a B2D limit line.

    Prefers the scored channel add_scored_comfort_channels() built; falls back to this project's own
    control-path series when it could not be built -- a steer=0 hist (longitudinal_PID.py) has no
    yaw/yaw_rate, so the scorer's inputs cannot be assembled from it at all. The suffix goes in the
    panel title so the two cases are never confused: without it, the same red limit line would be
    drawn over two different signals under one label, which is the exact problem this whole change
    exists to remove.
    """
    dest = SCORED_COMFORT_KEYS[key]
    if any(_get(h, dest) is not None for h in runs.values()):
        return dest, ""
    # No fallback signal exists any more -- the control-path jerk/yaw_acc channels are gone, so a
    # hist the scorer's inputs cannot be assembled from (no yaw/yaw_rate) simply has nothing to
    # draw here, and _dynamics_panel()'s "not recorded" message is the honest thing to show.
    return dest, ""


def add_scored_comfort_channels(hist, tools_dir=None, per_step=20):
    """hist + the differentiated comfort channels recomputed the way the SCORING module computes
    them, under the SCORED_COMFORT_KEYS names. Returns hist unchanged if they can't be built.

    Post-run, not per-tick, and that is not a shortcut: the scorer's derivative is a non-causal
    savgol over a 20-sample window, so tick k's value depends on ticks k+1..k+3 and simply does not
    exist yet inside a live control loop. Deriving it here from what hist already carries (yaw,
    yaw_rate, a_x, a_y) also means every run ever logged gets it retroactively, with no change to
    any controller script.

    Two properties worth knowing before plotting the result:
      * it is SEGMENT-WISE. The scorer filters each 20-tick block independently, so the series has
        a real discontinuity every second. That is what is scored, so it is drawn as-is rather than
        smoothed over -- and it makes the segment structure the metric works in visible.
      * the last partial segment (< per_step ticks) has no scored value at all; the scorer drops it.
        Those ticks are NaN here, which matplotlib skips, so the panel just ends slightly early.
    """
    scorer, kw = _b2d_comfort_inputs(hist, tools_dir)
    if kw is None:
        return hist
    n = len(kw["angular_velocity"])
    if n <= per_step:
        return hist

    out = {src: [] for src in _SCORED_SOURCE.values()}
    covered = 0
    for start in range(0, n, per_step):
        stop = start + per_step
        if stop > n:
            break
        detail = _comfort_channel_series(scorer, kw, start, stop)
        if detail is None:
            return hist
        for src in out:
            out[src].append(detail[src][0])
        covered = stop

    total = len(hist.get("t", [])) or n
    merged = dict(hist)
    for key, dest in SCORED_COMFORT_KEYS.items():
        values = np.concatenate(out[_SCORED_SOURCE[key]])
        padded = np.full(total, np.nan)
        padded[:min(covered, total)] = values[:min(covered, total)]
        merged[dest] = padded
    return merged


def b2d_comfort_report(hist, tools_dir=None, per_step=20, tick_s=0.05):
    """Why b2d_comfortness() came out the way it did: one row per scored 1-second segment.

    The score is a pass/fail ratio over fixed 20-tick segments and a segment fails the moment ONE
    sample on ONE of six channels leaves its band, so the single number can't say whether a run
    was mildly bad everywhere or catastrophically bad in one spot -- and it reads the RAW
    acceleration channels, not the low-passed ones the figures plot, so "the graph looks clean" and
    "the segment failed" are not in contradiction. This prints what actually happened.

    Returns {"score", "segments", "n_pass", "n_fail", "dropped_ticks"} or None. segments is a list
    of {"index", "t0", "t1", "passed", "channels": {name: {"min","max","lo","hi","ok"}}}.
    "channels" is {} when the scoring module doesn't expose the internals the breakdown needs.
    """
    scorer, kw = _b2d_comfort_inputs(hist, tools_dir)
    if kw is None:
        return None
    n = len(kw["angular_velocity"])
    if n <= per_step:
        return None

    segments, n_pass = [], 0
    for index, start in enumerate(range(0, n, per_step)):
        stop = start + per_step
        if stop > n:
            break                                   # the scorer drops a short tail; so do we
        passed = bool(scorer.compute_comfort_metric(
            kw["acceleration"][start:stop], kw["angular_velocity"][start:stop],
            kw["forward_vector"][start:stop], kw["right_vector"][start:stop],
            kw["location"][start:stop], kw["rotation"][start:stop]))
        n_pass += passed
        channels = {}
        detail = _comfort_channel_series(scorer, kw, start, stop)
        if detail:
            for name, (values, lo, hi) in detail.items():
                channels[name] = dict(min=float(values.min()), max=float(values.max()),
                                      lo=lo, hi=hi,
                                      ok=bool(values.min() > lo and values.max() < hi))
        segments.append(dict(index=index, t0=start * tick_s, t1=stop * tick_s,
                             passed=passed, channels=channels))

    # The breakdown is a second implementation of the module's own channel math; if it disagrees
    # with the module about even one segment, drop the per-channel detail rather than print
    # numbers that don't explain the score they claim to explain.
    official = b2d_comfortness(hist, tools_dir)
    if official is not None and segments and abs(n_pass / len(segments) - official) > 1e-9:
        print("  ! comfort report: 구간 판정이 채점 모듈과 불일치 -- 채널 내역을 생략합니다")
        for seg in segments:
            seg["channels"] = {}
    return dict(score=official, segments=segments, n_pass=n_pass,
                n_fail=len(segments) - n_pass, dropped_ticks=n - len(segments) * per_step)


def print_comfort_report(hist, target_speed_ms=None, tools_dir=None, failures_only=False):
    """b2d_comfort_report() as a table. Marks the offending channel(s) on every failed segment.

    failures_only=True prints just the failed segments -- for a long run where most pass.
    """
    report = b2d_comfort_report(hist, tools_dir)
    if report is None:
        print("  ! comfort report: 구간을 만들 만큼의 데이터가 없습니다")
        return None

    print(f"\n=== B2D Comfortness 구간 내역: {report['n_pass']}/{len(report['segments'])} 통과 "
          f"= {report['score']:.4f} ===")
    if report["dropped_ticks"]:
        print(f"  (뒤 {report['dropped_ticks']}틱은 20틱을 못 채워 채점에서 제외됨)")
    names = [name for name, _, _ in _COMFORT_CHANNELS]
    print("  seg  t (s)        " + "  ".join(f"{n:>9}" for n in names))
    for seg in report["segments"]:
        if failures_only and seg["passed"]:
            continue
        cells = []
        for name in names:
            ch = seg["channels"].get(name)
            if ch is None:
                cells.append(f"{'?':>9}")
                continue
            worst = ch["max"] if abs(ch["max"]) >= abs(ch["min"]) else ch["min"]
            cells.append(f"{worst:>8.2f}{'' if ch['ok'] else '*'}")
        mark = "PASS" if seg["passed"] else "FAIL"
        print(f"  {seg['index']:>3}  {seg['t0']:>5.1f}-{seg['t1']:<5.1f} {mark}  " + "  ".join(cells))
    # a "*" marks the channel whose own band this segment left -- the reason it failed
    print("  * = 이 채널이 한계를 벗어남 (값은 구간 내 절대값 최대 샘플)")
    print("  한계: " + ",  ".join(
        f"{name} [{seg['channels'][name]['lo']:.2f}, {seg['channels'][name]['hi']:.2f}]"
        for name in names
        for seg in report["segments"][:1] if name in seg["channels"]))
    return report


def print_error_summary(hist, target_speed_ms, lateral=True):
    """RMSE / max / mean of each tracked error, plus mean/peak magnitude of the raw longitudinal
    and lateral dynamics signals, over the whole run.

    The dynamics rows are printed as sections that appear only when this stack actually recorded
    them, and the section is skipped rather than printed empty. Pass lateral=False for a stack with
    no lateral control at all (longitudinal_mpc.py / longitudinal_PID.py pin steer at 0): those DO
    log the yaw channels, because the scored comfort channels and Comfortness need all six, so the
    "only print what was recorded" gating cannot tell them from a run that actually steers.
    """
    if not hist["t"]:
        return

    # v_y_hat - v_y (a genuine tracking error, unlike a_y/yaw_rate/... below which are raw
    # dynamics signals with no target) only exists for a run that logged a Kalman-filter v_y
    # estimate alongside ground truth (mpc_mpc_KF.py's "mpc-kf" controller) -- [] for every other
    # hist, same "row only appears if recorded" gating the dynamics sections already use.
    v_y_hat = hist.get("v_y_hat", [])
    v_y_est_err = ([hat - true for hat, true in zip(v_y_hat, hist["v_y"])]
                  if len(v_y_hat) and len(v_y_hat) == len(hist.get("v_y", [])) else [])

    print(f"\n=== error summary: {len(hist['t'])} steps, {hist['t'][-1]:.1f} s ===")
    for name, unit, series in (("cross-track", "m", hist.get("e_y", [])),
                               ("heading    ", "deg", hist.get("e_theta", [])),
                               ("speed      ", "m/s", speed_error_series(hist, target_speed_ms)),
                               ("v_y estimate", "m/s", v_y_est_err)):
        stats = error_stats(series)
        if stats is None:
            continue
        rmse, peak, bias = stats
        print(f"  {name}  RMSE={rmse:7.3f} {unit:<3}  max|e|={peak:7.3f} {unit:<3} ")

    # Every jerk/yaw-acceleration number in this project now comes from ONE place: the scoring
    # module's own derivative, rebuilt by add_scored_comfort_channels(). The figures already plot
    # those; this table used to print a differently-derived pair (a causal tau=0.15 low-pass
    # derivative) under the same names, so the printed peak and the scored peak disagreed by up to
    # 1.6x and a run could read as clear of the 4.13 limit here while the score had already failed
    # the segment. Same signal everywhere, or the numbers cannot be compared to each other.
    hist = add_scored_comfort_channels(hist)
    long_rows = (("accel a_x   ", "m/s^2", hist.get("a_x", [])),
                ("jerk        ", "m/s^3", hist.get("jerk_scored", [])),
                ("|jerk| total", "m/s^3", hist.get("jerk_total_scored", [])))
    lat_rows = (("yaw rate ", "deg/s  ", hist.get("yaw_rate", [])),
               ("yaw accel", "rad/s^2", hist.get("yaw_acc_scored", [])),
               ("accel a_y", "m/s^2  ", hist.get("a_y", [])))

    # lateral=False drops the whole lateral block for a stack that has no lateral control at all
    # (longitudinal_mpc.py pins steer at 0). Those runs DO log yaw/yaw_rate/a_y -- b2d_comfortness()
    # needs all six channels or it returns None -- so the "only print what was recorded" gating
    # below cannot tell them apart from a run that actually steers; the caller has to say.
    sections = [("longitudinal dynamics", long_rows)]
    if lateral:
        sections.append(("lateral dynamics", lat_rows))
    for section, rows in sections:
        printed_header = False
        for name, unit, series in rows:
            stats = magnitude_stats(series)
            if stats is None:
                continue
            if not printed_header:
                print(f"  -- {section} --")
                printed_header = True
            mean_abs, peak = stats
            print(f"  {name}  mean|.|={mean_abs:7.3f} {unit:<7}  peak|.|={peak:7.3f} {unit}")

    comfort = b2d_comfortness(hist)
    if comfort is not None:
        print(f"  -- B2D Comfortness (연구실 채점 방식, 1.000 = 전 구간 편안) --")
        print(f"    Comfortness   {comfort:.4f}")


# ---------------------------------------------------------------------------
# 7. plotting internals
# ---------------------------------------------------------------------------

def _panels(title, n_rows=3, n_cols=2, figsize=(15, 10), xlabel="$t$ (s)"):
    """A styled grid sharing the time axis, flattened in row-major order. title="" (or None) skips
    the suptitle entirely -- for figures (like plot_comparison()'s) where each panel's own title
    already carries the identifying info and a figure-level title would just be redundant.

    EVERY panel gets its own x tick labels and its own xlabel, not just the bottom row. sharex=True
    is kept -- it is what keeps the panels on one common x range, and what makes an interactive
    zoom/pan move all of them together -- but its side effect of blanking the tick labels on every
    row except the last is undone here. The reason is how these figures are actually read: a panel
    gets cropped into a slide or pointed at on its own, and one with a bare x axis is then
    unreadable, while counting rows up to the bottom of a 4-row grid to find the time axis is
    something a reader should not have to do. The cost is repeated identical tick rows, which is
    cheap next to that.

    xlabel: the shared x-axis label put on every panel. Defaults to time since every current caller
    is a time series; pass something else for a grid parameterized differently."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    if title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE, color=COLOR_INK, fontweight="bold")
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        _style_axes(ax)
        ax.tick_params(labelbottom=True)     # undo sharex's hiding, see docstring
        if xlabel:
            ax.set_xlabel(xlabel)
    return fig, axes


def _get(hist, key):
    """Series as an array, or None when this stack never recorded it."""
    values = hist.get(key)
    if values is None or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return arr


def _runs(data):
    """Normalize a single hist dict, or {label: hist} for a multi-controller comparison, into the
    latter. A lone hist has a top-level "t" key; a label dict does not (no controller is named
    "t"), so that's what tells the two apart."""
    return {"": data} if "t" in data else data


def _not_recorded(ax, key):
    ax.text(0.5, 0.5, f'"{key}" not recorded by this run', transform=ax.transAxes,
            ha="center", va="center", color=COLOR_MUTED, fontsize=10)


def _rmse_badge(ax, e, unit):
    """Corner box with the panel's own RMSE, so a figure is readable without the console output."""
    stats = error_stats(e)
    if stats is None:
        return
    ax.text(0.985, 0.05, f"RMSE = {stats[0]:.3f} {unit}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=COLOR_INK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=COLOR_AXIS, alpha=0.85))


def _rmse_box(ax, entries):
    """Bottom-left box holding one line of colored text per entry -- entries is [(text, color)].

    Built from matplotlib.offsetbox (TextArea/VPacker/AnnotationBbox) rather than stacked ax.text()
    calls: a single ax.text() can't mix colors within its string, and separately-bboxed lines don't
    read as one box -- VPacker stacks per-line TextAreas (each free to have its own color) inside
    one shared frame, which is what this needs.

    Corner and frame match _rmse_badge()'s, so a single-run box and a multi-run box are the same
    object in two sizes rather than two different-looking annotations.
    """
    children = [TextArea(text, textprops=dict(color=color, fontsize=9, fontweight="bold"))
                for text, color in entries]
    if not children:
        return
    packed = VPacker(children=children, pad=4, sep=3, align="left")
    box = AnnotationBbox(packed, (0.02, 0.02), xycoords="axes fraction", box_alignment=(0, 0),
                         frameon=True, pad=0.4, annotation_clip=False,
                         bboxprops=dict(boxstyle="round", facecolor="white",
                                        edgecolor=COLOR_AXIS, alpha=0.85))
    ax.add_artist(box)


def _rmse_box_multi(ax, results, colors, key, unit):
    """Bottom-left box, one line per controller: 'Label: 0.123 unit' in that controller's own
    color -- the multi-run analogue of _rmse_badge(), used by plot_comparison()'s error panels
    instead of folding the numbers into the panel title (which is what _rmse_suffix() used to do
    there; with 3+ controllers the title wrapped onto its neighbor's row). Each line's own color
    doubles as a legend, since the line's color already IDs the controller everywhere else in the
    figure -- no swatch/marker needed in the box itself, just colored text.

    Reads a stored hist key. _error_multi(rmse_box=True) builds the same box for panels whose error
    series is computed rather than stored (speed error, which is v_des - v_x).
    """
    entries = []
    for run_label, hist in results.items():
        e = _get(hist, key)
        stats = error_stats(e) if e is not None else None
        if stats is not None:
            entries.append((f"{run_label}: {stats[0]:.3f} {unit}", colors[run_label]))
    _rmse_box(ax, entries)


def _error_multi(ax, runs, colors, multi, ylabel, panel_title, unit, series_fn, legend=True,
                 linestyles=None, rmse_box=False):
    """One error-vs-time panel, for a single run or several overlaid.

    series_fn(hist) -> the error array for that run (or None if this stack never recorded it).
    Single run keeps the filled-band look; multiple runs switch to plain colored lines.

    legend=False skips this panel's own legend -- for callers (plot_comparison(),
    plot_longitudinal()) that build one shared legend for the whole figure instead of repeating it
    on every panel.

    rmse_box=True puts the RMSE in a bottom-left _rmse_box() instead of the two places it lives by
    default (bottom-right _rmse_badge() for a single run, a "(RMSE=...)" suffix on the legend label
    for several). Both defaults assume this panel has a legend of its own to carry the run names;
    once the legend moves to the figure bottom the suffix goes with it, too far from the panel to
    read as that panel's number -- so a figure with a shared bottom legend wants this on. Same
    corner and frame as _rmse_box_multi(), which does the same job for a stored hist key.

    linestyles: optional {label: linestyle}, consulted only when multi=True (see COMPARE_LINESTYLES)
    -- a run without an entry falls back to solid, same as before this parameter existed.
    """
    ax.axhline(0.0, color=COLOR_AXIS, linewidth=1, linestyle="--")
    found = False
    box_entries = []
    for label, hist in runs.items():
        e = series_fn(hist)
        if e is None:
            continue
        found = True
        t = np.asarray(hist["t"], dtype=float)
        color = colors[label]
        stats = error_stats(e)
        if not multi:
            ax.plot(t, e, color=color, linewidth=1.6, solid_capstyle="round")
            ax.fill_between(t, e, 0, color=color, alpha=0.15)
            if not rmse_box:
                _rmse_badge(ax, e, unit)
            elif stats:
                # single run: no run name to prefix, and COLOR_INK rather than the series color,
                # so it reads the same as the _rmse_badge() it replaces
                box_entries.append((f"RMSE = {stats[0]:.3f} {unit}", COLOR_INK))
        else:
            tag = label if rmse_box else (f"{label} (RMSE={stats[0]:.2f})" if stats else label)
            style = linestyles[label] if linestyles else "-"
            ax.plot(t, e, color=color, linewidth=1.4, linestyle=style, label=tag)
            if rmse_box and stats:
                box_entries.append((f"{label}: {stats[0]:.3f} {unit}", color))
    _rmse_box(ax, box_entries)
    if not found:
        _not_recorded(ax, "e")
    elif multi and legend:
        _legend(ax)
    ax.set_ylabel(ylabel)
    _title(ax, panel_title)


def _dynamics_panel(ax, runs, colors, multi, key, ylabel, panel_title, fill_color=None, ref_key=None,
                    legend=True, series_name="actual", ref_name="reference", linestyles=None):
    """One plain time-series panel (acceleration, jerk, yaw rate, ...), for a single run or
    several overlaid. Single run gets an optional fill; multiple runs get one colored line each
    plus a legend. "not recorded" if no run in `runs` logged `key` at all.

    ref_key: an optional companion series (e.g. "a_cmd") drawn as a dashed line in the same color as
    its run, right on top of the measured one -- for stacks that never computed it (stanley_PID.py
    has no a_cmd concept at all), _get() just returns None and the dashed line is silently skipped,
    so the same panel code works whether or not a given controller has a reference to show.

    series_name/ref_name: 범례에 붙는 역할 이름. ref_key 가 실제로 그려질 때만 쓰이며, 두 선이
    각각 무엇인지 말로 적어주기 위한 것이다 -- 같은 색 실선/점선만으로는 어느 쪽이 측정이고 어느
    쪽이 추정인지 알 수 없다. 기본값은 a_cmd 처럼 "명령/기준"을 겹쳐 그리는 패널용이고, 칼만
    필터 v_y 패널은 ref_name="estimated" 를 넘겨 "actual" / "estimated" 로 읽히게 한다.

    legend=False skips this panel's own legend -- for callers (plot_comparison()) that build one
    shared legend for the whole figure instead of repeating it on every panel.

    linestyles: optional {label: linestyle} (see COMPARE_LINESTYLES), consulted only when multi=True
    and this run has no ref line -- a run without an entry falls back to solid."""
    found = False
    has_ref = False
    for label, hist in runs.items():
        series = _get(hist, key)
        if series is None:
            continue
        found = True
        t = np.asarray(hist["t"], dtype=float)
        color = colors[label]
        ref = _get(hist, ref_key) if ref_key is not None else None
        series_label = label if multi else None
        if ref is not None:
            has_ref = True
            # 겹쳐 그릴 때는 실측선에도 역할 이름을 붙인다 -- 예전에는 run 라벨만 있어서
            # ("MPC-KF" vs "MPC-KF ref") 어느 쪽이 실제값인지 범례만 보고는 알 수 없었다.
            series_label = f"{label} {series_name}" if label else series_name
        # When there's a ref line to overlay, swap to LINEWIDTH_THIN (series) / LINEWIDTH (ref,
        # thicker) instead of the plain multi/non-multi widths below -- a thin solid + thick dashed
        # pair reads clearly even when the two nearly overlap (v_y_hat tracking v_y closely, a_cmd
        # tracking a_x closely, ...), which same-width same-color-plus-alpha didn't: the dashed line
        # all but disappeared under the solid one. ref is always COLOR_RED regardless of the
        # series' own color -- every current caller draws at most one (series, ref) pair per panel
        # (plot_lateral()'s ref_key uses are all multi=False; plot_kf_series() always hands this a
        # single-entry runs dict), so a fixed contrasting ref color never collides with another
        # run's own ref, and reads as "the estimate/reference" at a glance instead of just a paler
        # copy of whatever color the series happened to get. No ref -> untouched (1.3/1.5 as before).
        series_lw = LINEWIDTH_THIN if ref is not None else (1.3 if multi else 1.5)
        # Per-controller dash pattern only when there's no ref line to keep solid-vs-dashed free for
        # the actual/estimated contrast above -- see the ref_is-not-None branch's own note.
        series_style = (linestyles[label] if (multi and linestyles and ref is None) else "-")
        ax.plot(t, series, color=color, linewidth=series_lw, linestyle=series_style,
               solid_capstyle="round", label=series_label)
        if not multi and fill_color:
            ax.fill_between(t, series, 0, color=fill_color, alpha=0.15)
        if ref is not None:
            ref_label = f"{label} {ref_name}" if label else ref_name
            ax.plot(t, ref, color=COLOR_RED, linewidth=LINEWIDTH, linestyle="--", alpha=0.9,
                   label=ref_label)
    if not found:
        _not_recorded(ax, key)
    else:
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=1)
        if (multi or has_ref) and legend:
            _legend(ax)
    ax.set_ylabel(ylabel)
    _title(ax, panel_title)


# One place for the B2D limit line's stroke, so _b2d_limit_lines() and the legend swatch a shared
# figure legend needs for it (_b2d_limit_handle()) cannot drift apart. Its (5, 3) dash is longer
# and its stroke thicker than _dynamics_panel()'s "--" reference line, which is also COLOR_RED --
# that contrast is the only thing separating the two in a panel that draws both.
_B2D_LIMIT_STYLE = dict(color=COLOR_RED, linewidth=1.8, linestyle=(0, (5, 3)), alpha=0.95)


def _b2d_limit_lines(ax, lo, hi):
    """Red dashed line(s) marking a B2D_COMFORT_LIMITS band on a dynamics panel -- both bounds for
    a two-sided range, just the top one when lo==0 (a magnitude/norm signal like |jerk|, which
    never goes negative, so a line at 0 would just sit on the axis). Drawn at zorder=2.5, above the
    data lines (zorder~2 by default) and the axhline(0) reference (zorder~1), so the limit itself
    always reads clearly instead of blending into whatever data line happens to sit on top of it."""
    ax.axhline(hi, zorder=2.5, **_B2D_LIMIT_STYLE)
    if lo != 0:
        ax.axhline(lo, zorder=2.5, **_B2D_LIMIT_STYLE)


def _b2d_limit_handle(label="B2D comfort limit"):
    """Legend swatch for the _b2d_limit_lines() stroke. Only a figure with a shared bottom legend
    needs one: a per-panel legend is built from that panel's own labeled artists, and the limit
    lines are deliberately unlabeled axhlines (labeling them would repeat the same entry on every
    dynamics panel). A single figure-level legend is the one place the marking can be named once."""
    return plt.Line2D([0], [0], label=label, **_B2D_LIMIT_STYLE)


def close_all_figures(*_):
    """Close every open matplotlib figure. Bound to a key on each window and to SIGINT by _show()."""
    plt.close("all")


def _show():
    """plt.show(), but every open window closes together on `q`/`escape` or on Ctrl+C.

    Stock matplotlib makes a batch of figures tedious in exactly the way this repo produces them:
    plt.show() blocks until the LAST window is closed, its default `q` keymap closes only the
    focused one, and Ctrl+C in the terminal is swallowed by the GUI event loop -- so a run that
    drew eight panels had to be dismissed eight times before the script would exit.

    Both escapes are installed here rather than left to the caller so that every figure this module
    shows behaves the same way, whichever entry point drew it:
      * `q` / `escape` on any window  -> closes all of them (stock `q` closes just that one)
      * Ctrl+C in the terminal        -> same, via a SIGINT handler restored on the way out

    The SIGINT handler is only installable from the main thread (signal.signal raises otherwise, e.g.
    under a worker thread or some notebook kernels); that case falls back to key-only, which is why
    the whole install is guarded rather than assumed.
    """
    figs = [plt.figure(n) for n in plt.get_fignums()]
    if not figs:
        return
    for fig in figs:
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda ev: close_all_figures() if ev.key in ("q", "escape") else None)

    previous = None
    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, close_all_figures)
    except (ValueError, OSError):
        previous = None      # not the main thread -- key-only, see docstring

    print(f"  ({len(figs)} figure(s) open -- press q or esc on any window, or Ctrl+C here, "
          f"to close them all)")
    try:
        plt.show()          # the real one -- everything else in this module calls _show()
    finally:
        if previous is not None:
            try:
                signal.signal(signal.SIGINT, previous)
            except (ValueError, OSError):
                pass


def _save(fig, out_dir, stem):
    out_path = os.path.join(out_dir, f"{stem}.png")
    # bbox_inches="tight": recrops to whatever the figure actually drew, so a fig-level legend
    # placed just outside the constrained-layout axes area (plot_comparison()'s shared bottom
    # legend) doesn't get clipped off the saved PNG.
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return out_path


# ---------------------------------------------------------------------------
# 8. result figures
# ---------------------------------------------------------------------------

def plot_lateral(hist, title="Lateral tracking performance"):
    """fig 1: cross-track error, heading error, yaw pair, yaw rate, yaw acceleration, lateral
    velocity, lateral acceleration, steer -- for one controller's single run.

    hist: one run's hist dict. For comparing several controllers' runs against each other, see
    plot_comparison() instead -- it reuses the same _error_multi/_dynamics_panel helpers this
    function calls with multi=True, in its own dedicated figures rather than overlaying them here.
    """
    runs, colors = {"": add_scored_comfort_channels(hist)}, {"": COLOR_BLUE}
    fig, (ax_ey, ax_eth, ax_yaw, ax_r, ax_racc, ax_vy, ax_ay, ax_steer) = _panels(
        title, n_rows=4, figsize=(15, 13))

    _error_multi(ax_ey, runs, colors, False, "$e_y$ (m)", "Lateral error", "m",
                lambda h: _get(h, "e_y"))
    _error_multi(ax_eth, runs, colors, False, r"$e_\theta$ (deg)", "Heading error", "deg",
                lambda h: _get(h, "e_theta"))

    ax_yaw.plot(hist["t"], hist["path_yaw"], color=COLOR_MUTED, linewidth=2,
               linestyle="--", label="path yaw")
    ax_yaw.plot(hist["t"], hist["yaw"], color=COLOR_BLUE, linewidth=1.8,
               solid_capstyle="round", label="ego yaw")
    ax_yaw.set_ylabel(r"$\psi$ (deg)")
    _title(ax_yaw, "Vehicle heading vs. road heading")
    _legend(ax_yaw)

    _dynamics_panel(ax_r, runs, colors, False, "yaw_rate", r"$\dot\psi$ (deg/s)", "Yaw rate")
    _b2d_limit_lines(ax_r, *(math.degrees(v) for v in B2D_COMFORT_LIMITS["yaw_rate"]))
    _k, _sfx = _scored_key(runs, "yaw_acc")
    _dynamics_panel(ax_racc, runs, colors, False, _k, r"$\ddot\psi$ (rad/s$^2$)",
                    "Yaw acceleration" + _sfx)
    _b2d_limit_lines(ax_racc, *B2D_COMFORT_LIMITS["yaw_acc"])
    # ref_key="v_y_hat": when a run logged a Kalman-filter v_y estimate alongside ground truth
    # (mpc_mpc_KF.py's "mpc-kf" controller), it's overlaid as a dashed line in the same color --
    # see _dynamics_panel's ref_key doc. Runs that never log it (every other stack, plus this same
    # stack's own ground-truth "mpc" baseline run) just get _get()==None and the overlay is skipped,
    # so this is a no-op for every plot_lateral() caller that existed before mpc_mpc_KF.py.
    _dynamics_panel(ax_vy, runs, colors, False, "v_y", "$v_y$ (m/s)", "Lateral velocity (body frame)",
                    ref_key="v_y_hat", ref_name="estimated")
    _dynamics_panel(ax_ay, runs, colors, False, "a_y", "$a_y$ (m/s$^2$)", "Lateral acceleration (body frame)")
    _b2d_limit_lines(ax_ay, *B2D_COMFORT_LIMITS["a_y"])

    _dynamics_panel(ax_steer, runs, colors, False, "steer_deg", r"$\delta$ (deg)", "Steering angle (front wheel)")

    return fig


def plot_kf_series(runs, key, ref_key, ylabel, title):
    """One-panel time-series overlay across several runs: `key` (solid) vs. `ref_key` (dashed, if
    logged), one color per run via COMPARE_COLORS. This is the shared machinery behind plot_kf_vy()
    (v_y_hat vs. ground-truth v_y) and kalman_filter.py's clean-vs-noisy sensor channel plots (dpsi,
    a_y) alike -- reuses _dynamics_panel's own multi-run/ref_key mechanism (same one plot_lateral()'s
    single-run panels use) rather than a bespoke routine, so every one of these figures stays on the
    same LINEWIDTH/FONTSIZE/COLOR_* knobs as everything else in this file.

    runs: {label: hist}, e.g. {"5 m/s": hist_5, "10 m/s": hist_10, "15 m/s": hist_15} -- same shape
    plot_comparison() takes. A run missing `ref_key` just draws `key` alone (ref line skipped, per
    _dynamics_panel's own "not recorded" handling).
    """
    colors = {label: COMPARE_COLORS[i % len(COMPARE_COLORS)] for i, label in enumerate(runs)}
    linestyles = {label: COMPARE_LINESTYLES[i % len(COMPARE_LINESTYLES)] for i, label in enumerate(runs)}
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(title, fontsize=FONTSIZE_TITLE, color=COLOR_INK, fontweight="bold")
    _style_axes(ax)
    _dynamics_panel(ax, runs, colors, True, key, ylabel, "", ref_key=ref_key,
                    ref_name="estimated" if ref_key == "v_y_hat" else "reference",
                    linestyles=linestyles)
    ax.set_xlabel("$t$ (s)")
    return fig


def plot_kf_vy(runs, title="$v_y$: Kalman-filter estimate vs. ground truth"):
    """v_y_hat (dashed) vs. ground-truth v_y (solid) -- see plot_kf_series(), which this wraps."""
    return plot_kf_series(runs, "v_y", "v_y_hat", "$v_y$ (m/s)", title)


def plot_kf_run(hist, title=""):
    """kalman_filter.py's whole per-run report as ONE figure, 3 stacked panels sharing a time axis --
    v_y estimate vs. ground truth, dpsi clean vs. noisy sensor, a_y clean vs. noisy sensor -- instead
    of 3 separate plot_kf_series() figures/windows for the same run. hist needs "v_y_hat" (from
    replay()) and "dpsi_noisy"/"ay_noisy" (the noisy measurements replay() actually fed the filter,
    see kalman_filter.py's main()) alongside the usual "v_y"/"yaw_rate"/"a_y" ground truth.

    Single-run (not {label: hist}) on purpose, unlike plot_kf_series/plot_kf_vy -- this is one run's
    full picture, not several runs' v_y overlaid, so it reuses _dynamics_panel with multi=False (same
    convention plot_lateral()'s own single-run panels use) rather than the multi-run color cycle.
    """
    runs, colors = {"": hist}, {"": COLOR_BLUE}
    fig, (ax_vy, ax_dpsi, ax_ay) = plt.subplots(3, 1, figsize=(11, 12), sharex=True,
                                                constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    if title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE, color=COLOR_INK, fontweight="bold")
    for ax in (ax_vy, ax_dpsi, ax_ay):
        _style_axes(ax)
        ax.tick_params(labelbottom=True)     # same as _panels(): sharex must not blank these
        ax.set_xlabel("$t$ (s)")

    _dynamics_panel(ax_vy, runs, colors, False, "v_y", "$v_y$ (m/s)",
                    "$v_y$ estimate vs. ground truth", ref_key="v_y_hat", ref_name="estimated")
    # 이 두 패널은 추정/기준이 아니라 "센서 원신호 vs 노이즈 주입본" 비교이므로 범례도 그대로
    # clean/noisy -- v_y 패널의 actual/estimated 와 같은 이유로, 같은 색 실선/점선만으로는
    # 어느 쪽이 필터가 실제로 받은 신호인지 알 수 없다.
    _dynamics_panel(ax_dpsi, runs, colors, False, "yaw_rate", r"$\dot\psi$ (deg/s)",
                    r"$\dot\psi$: clean vs. noisy sensor", ref_key="dpsi_noisy",
                    series_name="clean", ref_name="noisy")
    _dynamics_panel(ax_ay, runs, colors, False, "a_y", "$a_y$ (m/s$^2$)",
                    "$a_y$: clean vs. noisy sensor", ref_key="ay_noisy",
                    series_name="clean", ref_name="noisy")
    return fig


def plot_trajectory_fit(fwd, lat, path, vx_spline, s_max_wp, s_mid, v_seg, vx_preview, kappa_preview,
                        dt, speed, title=""):
    r"""b2d_controller's single-sample deep dive: how one VAD-waypoint PathSpline+vx-spline fit
    (mpc_kf_controller.py's build_trajectory_splines()/preview_from_splines()) actually looks --
    4 panels sharing the same COLOR_*/LINEWIDTH/FONTSIZE_* knobs as every other figure in this file.
    Built for b2d_controller/inspect_one_sample.py, which computes every array this takes (it needs
    the same intermediates for its own printed formulas, so recomputing them here would just be a
    second, possibly-divergent copy).

    fwd, lat: the fit's own input points in this file's (forward, lateral) convention -- [origin,
    *waypoints], length N (7 for VAD's usual 6 waypoints; no route-command target point mixed in --
    see mpc_kf_controller.py's build_trajectory_splines() docstring for why). path: the fitted
    PathSpline. vx_spline: the fitted speed spline. s_max_wp: the last waypoint's station, now also
    equal to path.s_max (the fit no longer extends past the real waypoints). s_mid, v_seg: the
    per-interval speed samples vx_spline was fit against. vx_preview, kappa_preview:
    preview_from_splines()'s own output (length n_p). dt: control period, used only to reconstruct
    the preview's own station cursor for plotting. speed: current speed (m/s, at t=0) -- NOT what
    vx_preview/vx_spline show, which is VAD's own predicted FUTURE speed along the trajectory; the
    two can differ a lot (e.g. current speed high, predicted speed low -- VAD forecasting a
    slowdown into a turn), by design, not by mistake.

    Note kappa_preview/vx_preview only ever cover station 0 to roughly n_p*dt*vx -- a TIME horizon,
    not path.kappa(s)/vx_spline(s)'s own full spatial domain (0 to path.s_max, now the same as
    s_max_wp). At low speed that's a small fraction of the fitted curve; the dense curves are still
    fit from every input point regardless of how far the preview's own marker series happens to
    reach.
    """
    s_cursor = np.concatenate([[0.0], np.cumsum(np.asarray(vx_preview) * dt)[:-1]])
    s_dense = np.linspace(0.0, path.s_max, 300)
    fx, fy = path.xy(s_dense)
    yaw_dense = np.degrees(path.yaw(s_dense))
    kappa_dense = path.kappa(s_dense)
    s_dense_wp = np.linspace(0.0, s_max_wp, 100)

    # Legend labels below are kept short on purpose (this file's convention everywhere else) --
    # a full-sentence label on the speed panel's axhline once made that legend's rendered bbox
    # bigger than its own panel, and constrained_layout (which sizes each panel around everything
    # drawn in it, legends included) shrank the actual data area down into a corner to make room.
    # The fuller explanation lives in the caption below instead.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    if title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE, color=COLOR_INK, fontweight="bold")
    for ax in axes.ravel():
        _style_axes(ax)

    ax = axes[0, 0]
    ax.plot(fx, fy, "-", color=COLOR_BLUE, linewidth=LINEWIDTH, label="fitted path")
    ax.plot(fwd[0], lat[0], "^", color=COLOR_BLUE, markersize=10, label="ego (origin)")
    ax.plot(fwd[1:], lat[1:], "o", color=COLOR_ORANGE, markersize=8, label="VAD waypoints")
    ax.set_xlabel("forward (m)"); ax.set_ylabel("lateral (m)")
    _title(ax, "path fit"); _legend(ax, loc="best")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[0, 1]
    ax.plot(s_dense, yaw_dense, "-", color=COLOR_PURPLE, linewidth=LINEWIDTH)
    ax.set_xlabel("station $s$ (m)"); ax.set_ylabel(r"$\psi$ (deg)")
    _title(ax, "path.yaw(s)")

    ax = axes[1, 0]
    ax.plot(s_dense, kappa_dense, "-", color=COLOR_PURPLE, linewidth=LINEWIDTH, label="path.kappa(s)")
    ax.plot(s_cursor, kappa_preview, "o", color=COLOR_RED, markersize=5, label="kappa_preview")
    ax.axhline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)
    ax.set_xlabel("station $s$ (m)"); ax.set_ylabel(r"$\kappa$ (1/m)")
    _title(ax, "curvature"); _legend(ax, loc="best")

    ax = axes[1, 1]
    ax.plot(s_dense_wp, vx_spline(s_dense_wp), "-", color=COLOR_AQUA, linewidth=LINEWIDTH, label="vx_spline(s)")
    ax.plot(s_mid, v_seg, "o", color=COLOR_ORANGE, markersize=7, label="speed samples")
    ax.plot(s_cursor, vx_preview, "x", color=COLOR_RED, markersize=6, label="vx_preview")
    ax.set_xlabel("station $s$ (m)"); ax.set_ylabel("$v_x$ (m/s)")
    _title(ax, "speed fit"); _legend(ax, loc="best")

    # supxlabel (not a bare fig.text): constrained_layout reserves real space for it like any other
    # figure-level label, so it stays inside the canvas on both plt.show() and savefig() -- a plain
    # fig.text() placed below the constrained_layout-managed area gets clipped by the display
    # window and only survives savefig's separate bbox_inches="tight" recompute.
    fig.supxlabel(
        "*t=0 speed -- vx_preview/vx_spline show VAD's own predicted FUTURE speed along the path, "
        "which can differ a lot (e.g. slowing into this turn).",
        fontsize=FONTSIZE_TICK, color=COLOR_MUTED, wrap=True)

    return fig


def plot_longitudinal(hist, target_speed_ms, title="Longitudinal speed tracking performance"):
    """fig 2: speed error, speed pair, longitudinal acceleration, longitudinal jerk, total jerk
    magnitude, control input u -- for one controller's single run.

    The last panel plots u = throttle - brake, a single signed series in [-1, 1] (throttle and
    brake are mutually exclusive in every hist this repo logs, so the subtraction reconstructs the
    actual command exactly) rather than the two separate [0, 1] series, since that's the pedal
    signal a controller actually computed before it got split into carla.VehicleControl's two
    fields.

    hist: one run's hist dict. For comparing several controllers' runs against each other, see
    plot_comparison() instead (reuses _error_multi/_dynamics_panel with multi=True there).
    """
    runs, colors = {"": add_scored_comfort_channels(hist)}, {"": COLOR_BLUE}
    fig, (ax_ev, ax_v, ax_a, ax_j, ax_jtot, ax_cmd) = _panels(title)

    _error_multi(ax_ev, runs, colors, False, "$e_v$ (m/s)", "Speed error (reference - measured)",
                "m/s", lambda h: np.asarray(speed_error_series(h, target_speed_ms), dtype=float),
                legend=False, rmse_box=True)

    v_des = _get(hist, "v_des")
    if v_des is None:
        ax_v.axhline(target_speed_ms, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    else:
        ax_v.plot(hist["t"], v_des, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    ax_v.plot(hist["t"], hist["v_x"], color=COLOR_BLUE, linewidth=1.8, solid_capstyle="round",
              label="ego vel")
    ax_v.set_ylabel("$v_x$ (m/s)")
    _title(ax_v, "Speed")

    # series_name/ref_name spelled out rather than left at "actual"/"reference": those read fine in
    # a legend sitting on this panel, but the shared legend below the figure has no panel context
    # to borrow, so the entry has to name its own signal.
    _dynamics_panel(ax_a, runs, colors, False, "a_x", "$a_x$ (m/s$^2$)", "Longitudinal acceleration",
                    ref_key="a_cmd", legend=False,
                    series_name="measured $a_x$", ref_name="commanded $a_x$")
    _b2d_limit_lines(ax_a, *B2D_COMFORT_LIMITS["a_x"])
    _k, _sfx = _scored_key(runs, "jerk")
    _dynamics_panel(ax_j, runs, colors, False, _k, "jerk (m/s$^3$)",
                    "Longitudinal jerk" + _sfx, legend=False)
    _b2d_limit_lines(ax_j, *B2D_COMFORT_LIMITS["jerk"])
    _k, _sfx = _scored_key(runs, "jerk_total")
    _dynamics_panel(ax_jtot, runs, colors, False, _k, "|jerk| (m/s$^3$)",
                    "Total jerk magnitude" + _sfx, fill_color=COLOR_PURPLE, legend=False)
    _b2d_limit_lines(ax_jtot, *B2D_COMFORT_LIMITS["jerk_total"])

    ax_cmd.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    t = np.asarray(hist["t"], dtype=float)
    u = np.asarray(hist["throttle"], dtype=float) - np.asarray(hist["brake"], dtype=float)
    ax_cmd.plot(t, u, color=COLOR_BLUE, linewidth=1.6, solid_capstyle="round")
    ax_cmd.fill_between(t, u, 0, where=(u >= 0), color=COLOR_AQUA, alpha=0.15, interpolate=True)
    ax_cmd.fill_between(t, u, 0, where=(u <= 0), color=COLOR_RED, alpha=0.15, interpolate=True)
    ax_cmd.set_ylim(-1.05, 1.05)
    ax_cmd.set_ylabel("$u$")
    _title(ax_cmd, "Longitudinal control input $u$")

    _bottom_legend(fig, _panel_handles([ax_v, ax_a]) + [_b2d_limit_handle()])
    return fig


def _draw_reference_path(ax, path_x, path_y, label="desired path"):
    """The desired path, drawn as a corridor + centreline rather than one more line.

    It used to be COLOR_MUTED at linewidth 3, dashed -- a thicker, greyer version of exactly what
    the trial trajectories are, which is the one thing it must not look like: with 3+ trials
    overlapping it, "which of these is the reference" came down to spotting a grey among five
    colours. Two strokes fix that by making it a different KIND of mark:

      * a wide, very pale band UNDER everything (zorder 1) -- reads as the road/corridor, gives the
        eye the route's shape at a glance, and cannot hide a trial line because it sits below them
      * a thin near-black dashed centreline ON TOP (zorder 6) -- the exact reference, in the one
        colour COMPARE_COLORS never uses, thin enough that trials stay readable through it

    Trials are 2 px solid/dashed colour at zorder ~2, so neither stroke competes with them.
    """
    ax.plot(path_x, path_y, color=COLOR_INK, linewidth=9, alpha=0.10,
            solid_capstyle="round", solid_joinstyle="round", zorder=1)
    ax.plot(path_x, path_y, color=COLOR_INK, linewidth=1.4, linestyle=(0, (7, 4)),
            alpha=0.85, zorder=6, label=label)


def plot_trajectory(path_x, path_y, hist, title="Desired path vs. ego trajectory"):
    """fig 3: the xy view for one run's driven line against the desired path. Its own figure
    because equal aspect fights a shared time-series grid.

    hist: one run's hist dict. For overlaying several controllers' driven lines on the same
    desired path, see plot_comparison() instead.
    """
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    _style_axes(ax)

    ax.plot(hist["x"], hist["y"], color=COLOR_ORANGE, linewidth=2, solid_capstyle="round",
           label="ego trajectory")
    _draw_reference_path(ax, path_x, path_y)
    ax.scatter([path_x[0]], [path_y[0]], color=COLOR_BLUE, zorder=7, label="start")
    ax.scatter([path_x[-1]], [path_y[-1]], color=COLOR_RED, marker="*", s=140, zorder=7, label="goal")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    _title(ax, title)
    ax.set_aspect("equal", adjustable="datalim")
    _legend(ax)
    return fig


def plot_results(path_x, path_y, hist, target_speed_ms, out_dir,
                 show=True, summary=True, label="", name=None):
    """Save + show the three result figures and print the error summary, for one controller's
    single run. For comparing 2+ controllers' runs against each other, see plot_comparison().

    Files are named <script>_<date>_<time>_<kind>.png; pass name to override the script part.
    Returns the list of saved paths (lateral, longitudinal, trajectory).
    """
    if summary:
        print_error_summary(hist, target_speed_ms)

    suffix = f" - {label}" if label else ""
    figures = [
        ("lateral", plot_lateral(hist, f"Lateral tracking performance{suffix}")),
        ("longitudinal", plot_longitudinal(hist, target_speed_ms,
                                           f"Longitudinal speed tracking performance{suffix}")),
        ("trajectory", plot_trajectory(path_x, path_y, hist)),
    ]

    os.makedirs(out_dir, exist_ok=True)
    out_paths = [_save(fig, out_dir, run_name(kind, name)) for kind, fig in figures]
    for path in out_paths:
        print(f"Figure saved: {path}")

    if show:
        _show()
    for _, fig in figures:
        plt.close(fig)
    return out_paths


def plot_comparison(results, path_x, path_y, out_dir, target_speed_ms, show=True, name=None):
    """Compare 2+ controllers' runs against each other, instead of overlaying them onto
    plot_results()'s own three figures (which those are no longer built to do -- see their
    docstrings). Produces:

      1 trajectory figure   -- desired path + every trial's driven path, one color per trial.
      1 comparison figure   -- lateral error, heading error, yaw rate, yaw acceleration, a_x, a_y,
                               longitudinal jerk, total jerk (4x2 grid), each panel one line per
                               trial via the same _error_multi/_dynamics_panel helpers plot_lateral/
                               plot_longitudinal use internally, called here with multi=True.
      2 figures per trial    -- that trial's own plot_lateral()/plot_longitudinal(), labeled.

    results: {label: hist}, 2+ entries. Total figures for N trials: 1 + 1 + 2N (6 for N=2). Also
    prints print_error_summary() per trial, same as plot_results() does for a single run.
    """
    if len(results) < 2:
        raise ValueError(f"plot_comparison needs 2+ results to compare, got {len(results)}")

    for run_label, hist in results.items():
        print(f"\n### {run_label} ###")
        print_error_summary(hist, target_speed_ms)

    # every panel carrying a B2D_COMFORT_LIMITS line must plot the signal the SCORE was computed
    # from, not this project's own control-path derivative -- see add_scored_comfort_channels()
    results = {k: add_scored_comfort_channels(v) for k, v in results.items()}
    colors = dict(zip(results, COMPARE_COLORS))
    linestyles = dict(zip(results, COMPARE_LINESTYLES))
    os.makedirs(out_dir, exist_ok=True)
    figures = []   # (stem, fig) pairs, saved+closed together at the end

    # ---- 1. trajectory overlay ---- #
    fig_traj, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    fig_traj.patch.set_facecolor(COLOR_BG)
    _style_axes(ax)
    for run_label, hist in results.items():
        ax.plot(hist["x"], hist["y"], color=colors[run_label], linewidth=2,
               linestyle=linestyles[run_label], solid_capstyle="round", label=run_label)
    _draw_reference_path(ax, path_x, path_y)
    ax.scatter([path_x[0]], [path_y[0]], color=COLOR_BLUE, zorder=7, label="start")
    ax.scatter([path_x[-1]], [path_y[-1]], color=COLOR_RED, marker="*", s=140, zorder=7, label="goal")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    _title(ax, "Desired path vs. driven trajectories")
    ax.set_aspect("equal", adjustable="datalim")
    _legend(ax, ncol=min(len(results) + 2, 4))
    figures.append(("trajectory-compare", fig_traj))

    # ---- 2. 8-panel comparison figure ---- #
    # No per-panel legends -- one shared legend for the whole figure goes on at the end instead.
    fig_cmp, (ax_ey, ax_eth, ax_r, ax_racc, ax_ax, ax_ay, ax_j, ax_jtot) = _panels(
        "Performance Comparison", n_rows=4, figsize=(15, 13))

    # RMSE numbers used to live in the panel title (_rmse_suffix(), appended text) -- with 3+
    # controllers that made the title wrap onto its neighbor's row. Now a bottom-left box per panel
    # (_rmse_box_multi(), one colored line per controller) instead, and the title stays short.
    _error_multi(ax_ey, results, colors, True, "$e_y$ (m)", "Lateral error", "m",
                lambda h: _get(h, "e_y"), legend=False, linestyles=linestyles)
    _rmse_box_multi(ax_ey, results, colors, "e_y", "m")
    _error_multi(ax_eth, results, colors, True, r"$e_\psi$ (deg)", "Heading error", "deg",
                lambda h: _get(h, "e_theta"), legend=False, linestyles=linestyles)
    _rmse_box_multi(ax_eth, results, colors, "e_theta", "deg")
    _dynamics_panel(ax_r, results, colors, True, "yaw_rate", r"$\dot\psi$ (deg/s)", "Yaw rate", legend=False,
                    linestyles=linestyles)
    _b2d_limit_lines(ax_r, *(math.degrees(v) for v in B2D_COMFORT_LIMITS["yaw_rate"]))
    _k, _sfx = _scored_key(results, "yaw_acc")
    _dynamics_panel(ax_racc, results, colors, True, _k, r"$\ddot\psi$ (rad/s$^2$)",
                    "Yaw acceleration" + _sfx, legend=False, linestyles=linestyles)
    _b2d_limit_lines(ax_racc, *B2D_COMFORT_LIMITS["yaw_acc"])
    _dynamics_panel(ax_ax, results, colors, True, "a_x", "$a_x$ (m/s$^2$)", "Longitudinal acceleration",
                    legend=False, linestyles=linestyles)
    _b2d_limit_lines(ax_ax, *B2D_COMFORT_LIMITS["a_x"])
    _dynamics_panel(ax_ay, results, colors, True, "a_y", "$a_y$ (m/s$^2$)", "Lateral acceleration",
                    legend=False, linestyles=linestyles)
    _b2d_limit_lines(ax_ay, *B2D_COMFORT_LIMITS["a_y"])
    _k, _sfx = _scored_key(results, "jerk")
    _dynamics_panel(ax_j, results, colors, True, _k, "jerk (m/s$^3$)",
                    "Longitudinal jerk" + _sfx, legend=False, linestyles=linestyles)
    _b2d_limit_lines(ax_j, *B2D_COMFORT_LIMITS["jerk"])
    _k, _sfx = _scored_key(results, "jerk_total")
    _dynamics_panel(ax_jtot, results, colors, True, _k, "|jerk| (m/s$^3$)",
                    "Total jerk magnitude" + _sfx, legend=False, linestyles=linestyles)
    _b2d_limit_lines(ax_jtot, *B2D_COMFORT_LIMITS["jerk_total"])

    # one shared legend for the whole figure, bottom center -- controller-name -> color + dash
    # pattern (the RMSE numbers now live in each error panel's own bottom-left box instead)
    handles = [plt.Line2D([0], [0], color=colors[run_label], linewidth=2,
                          linestyle=linestyles[run_label], label=run_label)
              for run_label in results]
    fig_cmp.legend(handles=handles, loc="lower center", ncol=len(results), frameon=False,
                  labelcolor=COLOR_INK, fontsize=FONTSIZE_LEGEND, bbox_to_anchor=(0.5, -0.02))
    figures.append(("comparison", fig_cmp))

    # WHERE the Comfortness points went, next to WHAT the trajectories did -- same batch, so it
    # shows with the rest instead of after them (see _show()).
    fig_comfort = _comfort_breakdown_fig(results)
    if fig_comfort is not None:
        figures.append(("comfort-breakdown", fig_comfort))

    # ---- 3. each trial's own lateral/longitudinal pair ---- #
    for run_label, hist in results.items():
        suffix = f" - {run_label}"
        stem = run_label.replace(" ", "-")
        figures.append((f"lateral-{stem}",
                        plot_lateral(hist, f"Lateral tracking performance{suffix}")))
        figures.append((f"longitudinal-{stem}",
                        plot_longitudinal(hist, target_speed_ms,
                                          f"Longitudinal speed tracking performance{suffix}")))

    out_paths = [_save(fig, out_dir, run_name(stem, name)) for stem, fig in figures]
    for path in out_paths:
        print(f"Figure saved: {path}")

    if show:
        _show()
    for _, fig in figures:
        plt.close(fig)
    return out_paths


def _plot_longitudinal_multi(runs, target_speed_ms, title):
    """Longitudinal-only overlay of 2+ runs, built directly from the shared _error_multi/
    _dynamics_panel helpers (multi=True) -- plot_longitudinal() itself is single-run only (see its
    own docstring), so plot_longitudinal_result builds this comparison layout itself instead of
    delegating to it, the same way plot_comparison() does for the full lateral+longitudinal case."""
    runs = {k: add_scored_comfort_channels(v) for k, v in runs.items()}
    colors = dict(zip(runs, COMPARE_COLORS))
    linestyles = dict(zip(runs, COMPARE_LINESTYLES))
    fig, (ax_ev, ax_v, ax_a, ax_j, ax_jtot, ax_cmd) = _panels(title)

    _error_multi(ax_ev, runs, colors, True, "$e_v$ (m/s)", "Speed error (reference - measured)",
                "m/s", lambda h: np.asarray(speed_error_series(h, target_speed_ms), dtype=float),
                legend=False, rmse_box=True, linestyles=linestyles)

    first_hist = next(iter(runs.values()))
    v_des = _get(first_hist, "v_des")
    if v_des is None:
        ax_v.axhline(target_speed_ms, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    else:
        ax_v.plot(first_hist["t"], v_des, color=COLOR_MUTED, linewidth=2, linestyle="--", label="desired vel")
    for run_label, hist in runs.items():
        ax_v.plot(hist["t"], hist["v_x"], color=colors[run_label], linewidth=1.8,
                 linestyle=linestyles[run_label], solid_capstyle="round", label=run_label)
    ax_v.set_ylabel("$v_x$ (m/s)")
    _title(ax_v, "Speed")

    _dynamics_panel(ax_a, runs, colors, True, "a_x", "$a_x$ (m/s$^2$)", "Longitudinal acceleration",
                    ref_key="a_cmd", legend=False, linestyles=linestyles)
    _b2d_limit_lines(ax_a, *B2D_COMFORT_LIMITS["a_x"])
    _k, _sfx = _scored_key(runs, "jerk")
    _dynamics_panel(ax_j, runs, colors, True, _k, "jerk (m/s$^3$)",
                    "Longitudinal jerk" + _sfx, legend=False, linestyles=linestyles)
    _b2d_limit_lines(ax_j, *B2D_COMFORT_LIMITS["jerk"])
    _k, _sfx = _scored_key(runs, "jerk_total")
    _dynamics_panel(ax_jtot, runs, colors, True, _k, "|jerk| (m/s$^3$)",
                    "Total jerk magnitude" + _sfx, legend=False, linestyles=linestyles)
    _b2d_limit_lines(ax_jtot, *B2D_COMFORT_LIMITS["jerk_total"])

    ax_cmd.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    for run_label, hist in runs.items():
        t = np.asarray(hist["t"], dtype=float)
        u = np.asarray(hist["throttle"], dtype=float) - np.asarray(hist["brake"], dtype=float)
        ax_cmd.plot(t, u, color=colors[run_label], linewidth=1.4,
                   linestyle=linestyles[run_label], label=run_label)
    ax_cmd.set_ylim(-1.05, 1.05)
    ax_cmd.set_ylabel("$u$")
    _title(ax_cmd, "Longitudinal control input $u$")

    # One handle per controller (color + dash pattern), same as plot_comparison()'s shared legend,
    # plus the two markings that aren't a controller. Built by hand rather than scraped with
    # _panel_handles(): _dynamics_panel()'s a_cmd overlay labels its lines "<run> measured" /
    # "<run> commanded" per run, which would put 2*len(runs) near-duplicate entries in the legend.
    handles = [plt.Line2D([0], [0], color=colors[run_label], linewidth=2,
                          linestyle=linestyles[run_label], label=run_label)
              for run_label in runs]
    handles.append(plt.Line2D([0], [0], color=COLOR_MUTED, linewidth=2, linestyle="--",
                              label="desired vel"))
    if any(_get(h, "a_cmd") is not None for h in runs.values()):
        handles.append(plt.Line2D([0], [0], color=COLOR_RED, linewidth=LINEWIDTH, linestyle="--",
                                  label="commanded $a_x$"))
    handles.append(_b2d_limit_handle())
    _bottom_legend(fig, handles)
    return fig


def plot_longitudinal_result(data, target_speed_ms, out_dir, show=True, summary=True, label="",
                             name=None, lateral=True):
    """Like plot_results(), but only the longitudinal figure.

    For stacks with no lateral control at all (e.g. longitudinal_PID.py, steer pinned at 0) --
    there's no path or steering to put in the other two figures.

    data: a single hist dict, or {label: hist} to overlay several controllers on one figure --
    e.g. longitudinal_mpc.py's --controller both. Each run gets its own error-summary block when
    there's more than one.
    """
    runs = _runs(data)
    if summary:
        for run_label, hist in runs.items():
            if run_label:
                print(f"\n### {run_label} ###")
            print_error_summary(hist, target_speed_ms, lateral=lateral)

    suffix = f" - {label}" if label else ""
    title = f"Longitudinal speed tracking performance{suffix}"
    if len(runs) > 1:
        fig = _plot_longitudinal_multi(runs, target_speed_ms, title)
    else:
        fig = plot_longitudinal(next(iter(runs.values())), target_speed_ms, title)

    os.makedirs(out_dir, exist_ok=True)
    out_path = _save(fig, out_dir, run_name("longitudinal", name))
    print(f"Figure saved: {out_path}")

    if show:
        _show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 8. LUT results
# ---------------------------------------------------------------------------

def plot_lut_validation(hist_ff, hist_pid, mode, args, out_dir, hist_pidonly=None, show=True, name=None):
    """Overlay LUT-only vs. LUT+PID vs. (optionally) plain-PID-only tracking, for validate_lut.py.

    "LUT" is this figure's name for what the code calls the feedforward term -- they are the same
    thing (LookupController.feedforward() is a table lookup), and the figures in this project name
    the mechanism rather than its role.

    mode == "accel": reference/response is commanded longitudinal acceleration (a_cmd vs a_x).
    mode == "speed":  reference/response is vehicle speed (v_des vs v_x) -- a_cmd there is the
                      analytic derivative of v_des fed to the LUT as its feedforward target, not
                      scored directly.

    hist_pidonly is optional (validate_lut.py's speed mode doesn't run that trial) so old callers
    and 2-trial modes keep working without passing it.
    """
    if mode == "accel":
        ref_key, resp_key, unit, ylabel = "a_cmd", "a_x", "m/s$^2$", "acceleration (m/s$^2$)"
    else:
        ref_key, resp_key, unit, ylabel = "v_des", "v_x", "m/s", "speed (m/s)"

    # "LUT" rather than "feedforward" in every label this figure draws -- the feedforward term IS
    # the lookup table, and "LUT only / LUT + PID / PID only" reads as one parallel set where
    # "feedforward only / feedforward + PID / PID only" did not. These are display strings built
    # here; validate_lut.py's own results dict keys are separate and untouched.
    # colour AND dash pattern per trial, index-for-index with COMPARE_LINESTYLES the same way the
    # multi-run figures pair them: these three curves sit on top of each other for most of a run
    # (that is the point of the comparison), and colour alone stops separating them where they
    # overlap -- or in grayscale, or for a red/green-colour-blind reader.
    series = [("LUT only", hist_ff, COLOR_AQUA, COMPARE_LINESTYLES[0]),
              ("LUT + PID", hist_pid, COLOR_BLUE, COMPARE_LINESTYLES[1])]
    if hist_pidonly is not None:
        series.append(("PID only", hist_pidonly, COLOR_ORANGE, COMPARE_LINESTYLES[2]))
    ts = {label: np.asarray(hist["t"], dtype=float) for label, hist, _, _ in series}

    # The trial names used to be spelled out in the suptitle too; they are in the shared legend at
    # the bottom now, so the title states only what the run was: which a_cmd profile, tracking what.
    fig, (ax_main, ax_err, ax_u) = _panels(
        f"{args.profile} profile {mode} tracking", n_rows=3, n_cols=1, figsize=(14, 10))

    ax_main.plot(ts["LUT only"], hist_ff[ref_key], color=COLOR_MUTED, linewidth=2.2,
                linestyle="--", label="reference")
    for label, hist, color, style in series:
        ax_main.plot(ts[label], hist[resp_key], color=color, linewidth=1.6, linestyle=style,
                     label=label)
    ax_main.set_ylabel(ylabel)
    _title(ax_main, "Tracking")

    ax_err.axhline(0.0, color=COLOR_AXIS, linewidth=1, linestyle="--")
    for label, hist, color, style in series:
        err = np.asarray(hist[ref_key], dtype=float) - np.asarray(hist[resp_key], dtype=float)
        ax_err.plot(ts[label], err, color=color, linewidth=1.4, linestyle=style, label=label)
    ax_err.set_ylabel(f"error ({unit})")
    _title(ax_err, "Tracking error (reference - measured)")

    ax_u.axhline(0.0, color=COLOR_AXIS, linewidth=1)
    for label, hist, color, style in series:
        ax_u.plot(ts[label], hist["u"], color=color, linewidth=1.3, linestyle=style, label=label)
    ax_u.set_ylim(-1.15, 1.15)
    ax_u.set_ylabel("pedal $u$")
    _title(ax_u, "Control input")

    # one legend for the figure -- every panel draws the same trials in the same colour and dash,
    # so repeating it three times only cost plot area. ax_main carries the reference line too.
    _bottom_legend(fig, _panel_handles([ax_main]))

    os.makedirs(out_dir, exist_ok=True)
    out_path = _save(fig, out_dir, run_name(mode, name))
    print(f"Figure saved: {out_path}")
    if show:
        _show()
    plt.close(fig)
    return out_path


# blue (brake) <-> neutral <-> red (throttle), built from the house palette so it matches
# everything else rather than introducing its own hex codes
_UA_CMAP = LinearSegmentedColormap.from_list("brake_throttle", [COLOR_BLUE, "#f0efec", COLOR_RED])


def plot_lut_raw_distribution(raw, out_dir, trusted_ranges=None, show=True, name=None):
    """Per-gear (v_x, a_x) scatter of collect_lut_data.py's raw sweep CSV, colored by control
    input u -- for eyeballing coverage and density right after a sweep.

    raw: structured array from np.genfromtxt(..., names=True) with gear/u/v_x/a_x columns.
    trusted_ranges: optional {gear: (v_lo, v_hi) or None}, from build_lut.py's
    trusted_speed_range() -- when given, shades the speed window each gear's fit actually trusted
    (None for a gear means it had no trustworthy window at all).

    gear 0 (CARLA's mid-shift/clutch-disengaged sentinel, not a real gear) is left out -- the
    runtime controller never queries it (see LookupController.feedforward()), and its samples
    are sparse and noisy enough that trusted_speed_range() rejects them anyway.
    """
    trusted_ranges = trusted_ranges or {}
    gears = sorted(g for g in {int(x) for x in raw["gear"]} if g != 0)
    ncols = min(3, len(gears))
    nrows = math.ceil(len(gears) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False,
                             constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    subtitle = ("\nshaded band = speed range build_lut.py trusted" if trusted_ranges else "")
    fig.suptitle(f"Raw sweep data -- per-gear (v_x, a_x), colored by control input u{subtitle}",
                fontsize=12, color=COLOR_INK, fontweight="bold")

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    flat_axes = axes.ravel()
    mappable = None
    for ax, gear in zip(flat_axes, gears):
        _style_axes(ax)
        mask = raw["gear"] == gear
        v, a, u = raw["v_x"][mask], raw["a_x"][mask], raw["u"][mask]
        mappable = ax.scatter(v, a, c=u, cmap=_UA_CMAP, norm=norm, s=6, alpha=0.35, linewidths=0)
        v_range = trusted_ranges.get(gear)
        if v_range is not None:
            ax.axvspan(v_range[0], v_range[1], color=COLOR_AQUA, alpha=0.10, zorder=0)
        title = f"gear {gear}"
        if trusted_ranges and v_range is None:
            title += "  [excluded]"
        _title(ax, title, fontsize=10)
        ax.set_xlabel("$v_x$ (m/s)")
        ax.set_ylabel("$a_x$ (m/s$^2$)")
    for ax in flat_axes[len(gears):]:
        ax.axis("off")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=fig.get_axes(), shrink=0.6, pad=0.02,
                            label="u  (brake <- 0 -> throttle)")
        cbar.ax.yaxis.label.set_color(COLOR_MUTED)
        cbar.ax.tick_params(colors=COLOR_MUTED)

    os.makedirs(out_dir, exist_ok=True)
    out_path = _save(fig, out_dir, run_name("lut_raw_distribution", name))
    print(f"Figure saved: {out_path}")
    if show:
        _show()
    plt.close(fig)
    return out_path


def _style_3d_axes(ax):
    ax.set_facecolor(COLOR_BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(COLOR_BG)
        pane.set_edgecolor(COLOR_GRID)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = COLOR_GRID
        axis.label.set_color(COLOR_MUTED)
        axis.label.set_fontweight("bold")
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)


def plot_lut_surfaces(gear_tables, out_dir, raw=None, show=True, elev=25.0, azim=-60.0, name=None):
    """One (v_x, a_x) -> u 3D surface per gear, small multiples on a shared diverging color scale
    (u: brake -1 -> throttle +1) so gears stay visually comparable -- for build_lut.py to sanity
    check a fit right after producing it.

    gear_tables: {gear: (v_grid, a_grid, u_table)}, u_table shaped (len(v_grid), len(a_grid)) --
    exactly what build_lut.py's build_gear_table() returns per gear.
    raw: optional structured array (gear/u/v_x/a_x columns) overlaid as raw sweep samples -- pass
    everything build_lut.py read in (not just what a gear's trust cut kept) to see the surface
    against what got excluded too, not only what it was fit from.
    """
    gears = sorted(gear_tables)
    ncols = min(3, len(gears))
    nrows = math.ceil(len(gears) / ncols)
    fig = plt.figure(figsize=(5.2 * ncols, 4.6 * nrows), facecolor=COLOR_BG)
    fig.suptitle("Longitudinal control-input lookup table  (u: brake -1 -> throttle +1)",
                color=COLOR_INK, fontsize=13, fontweight="bold")

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    mappable = None
    for i, gear in enumerate(gears):
        v_grid, a_grid, u_table = gear_tables[gear]
        vv, aa = np.meshgrid(v_grid, a_grid, indexing="ij")

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        mappable = ax.plot_surface(vv, aa, u_table, cmap=_UA_CMAP, norm=norm,
                                   linewidth=0, antialiased=True, alpha=0.92)

        if raw is not None:
            mask = raw["gear"] == gear
            ax.scatter(raw["v_x"][mask], raw["a_x"][mask], raw["u"][mask],
                      s=4, color=COLOR_MUTED, alpha=0.12, depthshade=False)

        _title(ax, f"Gear {gear}", fontsize=11)
        ax.set_xlabel("$v_x$ (m/s)")
        ax.set_ylabel("$a_x$ (m/s$^2$)")
        ax.set_zlabel("$u$")
        ax.set_zlim(-1, 1)
        ax.view_init(elev=elev, azim=azim)
        _style_3d_axes(ax)

    cbar = fig.colorbar(mappable, ax=fig.get_axes(), shrink=0.6, pad=0.02,
                        label="u  (brake <- 0 -> throttle)")
    cbar.ax.yaxis.label.set_color(COLOR_MUTED)
    cbar.ax.tick_params(colors=COLOR_MUTED)

    os.makedirs(out_dir, exist_ok=True)
    out_path = _save(fig, out_dir, run_name("lut_surfaces", name))
    print(f"Figure saved: {out_path}")
    if show:
        _show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 9. lateral parameter identification figures
# ---------------------------------------------------------------------------

def plot_cornering_stiffness_quadrant(alpha_f, Fyf, mask_f, alpha_r, Fyr, mask_r, Cf, Cr,
                                      out_dir=None, show=True, name=None):
    """For estimate_cornering_stiffness.py: every logged (alpha, Fy) sample, front and rear axle,
    plotted directly -- no speed color-coding, no per-speed band (this pipeline no longer selects
    by speed at all; see that script's module docstring). mask_f/mask_r mark which points actually
    went into the Cf/Cr fit: True = quadrant 1/3 (sign(alpha)==sign(Fy), a real cornering force
    pushes the same way as the slip angle that caused it) = used, drawn in the usual blue; False =
    quadrant 2/4 (wrong sign relationship, cannot be a genuine C*alpha sample) = excluded, drawn
    in red so it's visible on the plot exactly which points the fit is NOT trusting -- see
    filter_by_quadrant()'s docstring for why this is the only filter applied.

    alpha_f/Fyf/alpha_r/Fyr: full pooled arrays (radians, N) -- every logged sample, kept or not.
    Cf/Cr: the final fitted slopes (from the KEPT points only) to draw the dashed line from.
    """
    import matplotlib.pyplot as plt

    fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    for ax in (ax_f, ax_r):
        _style_axes(ax)
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)
        ax.axvline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)

    for ax, alpha, Fy, mask, C, axle in (
            (ax_f, alpha_f, Fyf, mask_f, Cf, "front"),
            (ax_r, alpha_r, Fyr, mask_r, Cr, "rear")):
        sub = axle[0]                       # "f" / "r" -- the subscript shared by alpha, Fy and C
        alpha_deg = np.degrees(np.asarray(alpha))
        Fy = np.asarray(Fy)
        mask = np.asarray(mask)

        ax.scatter(alpha_deg[mask], Fy[mask], s=MARKERSIZE * 0.35, facecolor=COLOR_BLUE,
                  edgecolor="none", alpha=0.5, zorder=3,
                  label=f"used in fit ({int(mask.sum())})")
        if (~mask).any():
            ax.scatter(alpha_deg[~mask], Fy[~mask], s=MARKERSIZE * 0.35, facecolor=COLOR_RED,
                      edgecolor="none", alpha=0.7, zorder=4,
                      label=f"excluded, wrong sign ({int((~mask).sum())})")

        lo, hi = min(0.0, alpha_deg.min()), max(0.0, alpha_deg.max())
        xs = np.linspace(lo, hi, 20)
        ax.plot(xs, C * np.radians(xs), color=COLOR_INK, linewidth=LINEWIDTH * 1.4,
               linestyle="--", zorder=5, label=rf"fit $C_{sub}$ = {C:,.0f} N/rad")
        ax.set_xlabel(rf"$\alpha_{{{sub}}}$ (deg)")
        ax.set_ylabel(rf"$F_{{y{sub}}}$ (N)")
        # mathtext, matching the axis labels' own subscripts: the panel title states the law being
        # fitted rather than spelling it out in ascii ("Fy = C * alpha"), so the title, the axes and
        # the legend entry all name the same three symbols the same way.
        _title(ax, rf"{axle.capitalize()} axle:  $F_{{y{sub}}} = C_{sub}\,\alpha_{sub}$")
        _legend(ax)

    out_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = _save(fig, out_dir, run_name("cornering_stiffness_quadrant", name))
        print(f"Figure saved: {out_path}")
    if show:
        _show()
    plt.close(fig)
    return out_path


# One flat fill for every uncomfort mark in plot_comfort_breakdown(): raster cells, the time-axis
# strip and the per-channel bars all use COLOR_RED at the same alpha, so the legend swatch is
# literally the same colour the figure draws with and carries the figure's only colour meaning.
_UNCOMFORT_CMAP = LinearSegmentedColormap.from_list("uncomfort", [COLOR_RED, COLOR_RED])


def _comfort_breakdown_fig(data, tools_dir=None, tick_s=0.05):
    """WHERE a run lost its Comfortness: a channel x 1-second-segment raster per controller, plus a
    per-channel tally of how many segments each one broke.

    The score is a pass/fail ratio over fixed 20-tick segments, and a segment fails the moment ONE
    sample on ONE of six channels leaves its band -- so the single number cannot say whether a run
    was mildly bad everywhere or catastrophically bad in one spot, nor which channel is responsible.
    Read the raster down a column for "why did second 7 fail", along a row for "which channel is the
    repeat offender".

    Cells are a flat fill, not a severity gradient: the metric is a pass/fail judgement, so the
    figure states the same thing it does -- filled means that channel put that second out of
    bounds. One colour, one meaning, matching the legend swatch and the bars exactly. (For how far
    outside a channel actually went, b2d_comfort_report() carries the per-channel min/max and
    b2d_comfort_penalty() integrates the excess continuously.)

    data: one hist dict, or {label: hist} for several controllers (stacked vertically).
    """
    runs = _runs(data)
    reports = {}
    for label, hist in runs.items():
        rep = b2d_comfort_report(hist, tools_dir=tools_dir, tick_s=tick_s)
        if rep is not None and rep["segments"]:
            reports[label] = rep
    if not reports:
        print("  ! comfort breakdown: 채점 가능한 구간이 없습니다")
        return None

    names = [n for n, _, _ in _COMFORT_CHANNELS]
    n_runs = len(reports)
    fig, axes = plt.subplots(n_runs, 2, figsize=(15, 1.1 + 2.6 * n_runs), squeeze=False,
                             gridspec_kw={"width_ratios": [4, 1]}, constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle("B2D Comfortness -- where each 1 s segment was lost",
                fontsize=FONTSIZE_TITLE, color=COLOR_INK, fontweight="bold")

    for row, (label, rep) in enumerate(reports.items()):
        ax, ax_bar = axes[row][0], axes[row][1]
        segs = rep["segments"]
        grid = np.full((len(names), len(segs)), np.nan)
        for col, seg in enumerate(segs):
            for r, nm in enumerate(names):
                ch = seg["channels"].get(nm)
                if ch is not None and not ch["ok"]:
                    grid[r, col] = 1.0          # filled = uncomfort, nothing else encoded

        _style_axes(ax)
        ax.grid(False)
        ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=_UNCOMFORT_CMAP,
                  vmin=0.0, vmax=1.0, alpha=0.75,
                  extent=(segs[0]["t0"], segs[-1]["t1"], len(names) - 0.5, -0.5),
                  interpolation="nearest")
        # a failed segment is any column with at least one filled cell -- mark it on the axis so the
        # score itself (passed/total) is readable straight off the raster
        for seg in segs:
            if not seg["passed"]:
                ax.axvspan(seg["t0"], seg["t1"], ymin=0, ymax=0.035, color=COLOR_RED, alpha=0.75,
                           lw=0)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("$t$ (s)")
        title = f"{label}: " if label else ""
        _title(ax, f"{title}{rep['n_pass']}/{len(segs)} segments pass "
                   f"= {rep['score']:.4f}", fontsize=FONTSIZE_SUBTITLE - 4)

        _style_axes(ax_bar)
        counts = [int(np.isfinite(grid[r]).sum()) for r in range(len(names))]
        ax_bar.barh(range(len(names)), counts, color=COLOR_RED, alpha=0.75)
        ax_bar.set_yticks(range(len(names)))
        ax_bar.set_yticklabels([])
        ax_bar.invert_yaxis()
        # NOT the number of failed segments: a segment that breaks four channels at once is counted
        # once in each of those four rows, so these bars sum to well past the failure count in the
        # title (82 vs 25 on a measured run). Per channel, how many 1 s segments it made uncomfort.
        ax_bar.set_xlabel("uncomfort segments")
        ax_bar.set_xlim(0, max(1, len(segs)))
        for r, c in enumerate(counts):
            if c:
                ax_bar.text(c, r, f" {c}", va="center", fontsize=9, color=COLOR_INK)
        _title(ax_bar, "per channel", fontsize=FONTSIZE_SUBTITLE - 6)

    # One entry for the one colour that carries meaning here: the same COLOR_RED fills the raster
    # cells, the time-axis strip under them and the per-channel bars, and all three mean the same
    # thing -- uncomfort. (Cell shading varies in intensity with how far outside the limit the
    # channel went; any intensity at all is a violation.)
    # label kept in ASCII: matplotlib's default font ships no Hangul glyphs, so Korean text here
    # renders as boxes in the saved PNG (every other string this module draws is English too)
    _bottom_legend(fig, [plt.Rectangle((0, 0), 1, 1, color=COLOR_RED, alpha=0.75,
                                       label="uncomfort  (1 s segment with any channel out of "
                                             "bounds)")],
                   max_ncol=1)

    return fig


def plot_comfort_breakdown(data, out_dir=None, show=True, name=None, tools_dir=None, tick_s=0.05):
    """_comfort_breakdown_fig() as a standalone figure: save it, optionally show it, return the
    path. plot_comparison() calls the builder directly instead, so the breakdown lands in the same
    batch of windows as everything else it draws rather than in a second blocking show()."""
    fig = _comfort_breakdown_fig(data, tools_dir=tools_dir, tick_s=tick_s)
    if fig is None:
        return None
    out_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = _save(fig, out_dir, run_name("comfort_breakdown", name))
        print(f"Figure saved: {out_path}")
    if show:
        _show()
    else:
        plt.close(fig)
    return out_path


def plot_speed_slip_angle(queued, out_dir=None, show=True, name=None):
    """For estimate_cornering_stiffness.py: every steady-window sample's (v_x, alpha) plotted
    directly, colored by steer angle -- shows how far alpha_f/alpha_r actually swept at each speed
    (not just its fitted Cf), which is the evidence --speeds should be widened or narrowed against:
    a speed whose slip-angle range has collapsed toward the noise floor, or whose steer-angle
    "rays" have started crossing, is a speed no longer worth including in the pooled fit.
    """
    import matplotlib.pyplot as plt

    done = [tr for tr in queued if tr.get("fit") is not None]
    steers = sorted({tr["steer_deg"] for tr in done})
    cmap = plt.get_cmap("viridis")
    steer_color = {d: cmap(i / max(1, len(steers) - 1)) for i, d in enumerate(steers)}

    fig, (ax_f, ax_r) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor(COLOR_BG)
    for ax in (ax_f, ax_r):
        _style_axes(ax)
        ax.axhline(0.0, color=COLOR_AXIS, linewidth=LINEWIDTH_THIN)

    for tr in done:
        color = steer_color[tr["steer_deg"]]
        v_x = np.asarray(tr["log"]["v_x"])[slice(*tr["window"])]
        alpha_f_deg = np.degrees(tr["fit"]["alpha_f"])
        alpha_r_deg = np.degrees(tr["fit"]["alpha_r"])
        ax_f.scatter(v_x, alpha_f_deg, s=MARKERSIZE * 0.5, facecolor=color, edgecolor="none",
                    alpha=0.5, zorder=2)
        ax_r.scatter(v_x, alpha_r_deg, s=MARKERSIZE * 0.5, facecolor=color, edgecolor="none",
                    alpha=0.5, zorder=2)

    ax_f.set_xlabel("$v_x$ (m/s)"); ax_f.set_ylabel(r"$\alpha_f$ (deg)")
    ax_r.set_xlabel("$v_x$ (m/s)"); ax_r.set_ylabel(r"$\alpha_r$ (deg)")
    _title(ax_f, "Front slip angle vs speed")
    _title(ax_r, "Rear slip angle vs speed")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=steer_color[d],
                          markeredgecolor="none", markersize=8, label=f"{d:g} deg")
              for d in steers]
    _bottom_legend(fig, handles, title="steer")

    out_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = _save(fig, out_dir, run_name("speed_slip_angle", name))
        print(f"Figure saved: {out_path}")
    if show:
        _show()
    plt.close(fig)
    return out_path


def cornering_stiffness_speed_stats(queued, by_speed):
    """Per-speed sample count / Cf & Cr / std(C across that speed's own steer angles) / alpha
    range, front AND rear axle both -- the numbers behind both print_cornering_stiffness_speed_
    table() and estimate_cornering_stiffness.py's own speed-selection step (drop a speed if
    either axle's C_std_pct or alpha_max_deg exceeds a threshold), split out as its own function
    so both read the exact same numbers instead of the selection logic recomputing its own
    version of what the table already printed.

    Returns {v: {"n_samples", "Cf", "Cf_std", "Cf_std_pct", "alpha_f_min_deg", "alpha_f_max_deg",
    "Cr", "Cr_std", "Cr_std_pct", "alpha_r_min_deg", "alpha_r_max_deg"}}, skipping speeds with no
    settled trial (by_speed.get(v) is None). n_samples is shared (front/rear windows are the same
    ticks, just a different axle's slip angle/force).
    """
    done = [tr for tr in queued if tr.get("fit") is not None]
    stats = {}
    for v in sorted({tr["target_speed"] for tr in done}):
        trials_v = [tr for tr in done if tr["target_speed"] == v]
        r = by_speed.get(v)
        if r is None or not trials_v:
            continue
        n_samples = sum(len(tr["fit"]["alpha_f"]) for tr in trials_v)
        Cf_std = float(np.std([tr["fit"]["Cf"] for tr in trials_v]))
        Cr_std = float(np.std([tr["fit"]["Cr"] for tr in trials_v]))
        alpha_f_all = np.degrees(np.concatenate([tr["fit"]["alpha_f"] for tr in trials_v]))
        alpha_r_all = np.degrees(np.concatenate([tr["fit"]["alpha_r"] for tr in trials_v]))
        stats[v] = {
            "n_samples": n_samples,
            "Cf": r["Cf"], "Cf_std": Cf_std,
            "Cf_std_pct": 100.0 * Cf_std / r["Cf"] if r["Cf"] else float("inf"),
            "alpha_f_min_deg": float(alpha_f_all.min()), "alpha_f_max_deg": float(alpha_f_all.max()),
            "Cr": r["Cr"], "Cr_std": Cr_std,
            "Cr_std_pct": 100.0 * Cr_std / r["Cr"] if r["Cr"] else float("inf"),
            "alpha_r_min_deg": float(alpha_r_all.min()), "alpha_r_max_deg": float(alpha_r_all.max()),
        }
    return stats


def print_cornering_stiffness_speed_table(queued, by_speed):
    """Per-speed sample count / Cf & Cr / std(C across that speed's own steer angles) / alpha
    range, front and rear both -- purely informational (estimate_cornering_stiffness.py no
    longer selects anything by speed; see that script's module docstring for why). Useful for
    eyeballing whether a speed's own steer angles agree with each other, and roughly how wide an
    alpha range each speed swept, without that reading feeding into the final Cf/Cr at all.
    """
    stats = cornering_stiffness_speed_stats(queued, by_speed)
    print(f"\n{'v (m/s)':>8} {'n samp':>7} "
          f"{'Cf (N/rad)':>12} {'Cf std%':>8} {'alpha_f (deg)':>18} "
          f"{'Cr (N/rad)':>12} {'Cr std%':>8} {'alpha_r (deg)':>18}")
    all_speeds = sorted({tr["target_speed"] for tr in queued})
    for v in all_speeds:
        s = stats.get(v)
        if s is None:
            print(f"{v:8.1f} {'--':>7} {'--':>12} {'--':>8} {'--':>18} "
                  f"{'--':>12} {'--':>8} {'--':>18}")
            continue
        print(f"{v:8.1f} {s['n_samples']:7d} "
              f"{s['Cf']:12,.0f} {s['Cf_std_pct']:7.1f}% "
              f"[{s['alpha_f_min_deg']:+6.2f},{s['alpha_f_max_deg']:+6.2f}] "
              f"{s['Cr']:12,.0f} {s['Cr_std_pct']:7.1f}% "
              f"[{s['alpha_r_min_deg']:+6.2f},{s['alpha_r_max_deg']:+6.2f}]")
