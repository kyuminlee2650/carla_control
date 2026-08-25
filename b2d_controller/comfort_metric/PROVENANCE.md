# efficiency_smoothness_benchmark.py

Bench2Drive 의 Comfortness 채점 모듈. **공식 배포판이 아니라, 공식 파일을 이 프로젝트가 수정한
버전을 바이트 그대로 복사한 것**이다. 이 구분이 중요하므로 아래에 근거를 남긴다.

- 복사해 온 경로: `/home/ailab/2026intern/kmlee/vad_demo_video/Bench2Drive/tools/efficiency_smoothness_benchmark.py`
- 그 체크아웃의 리모트: `https://github.com/Thinklab-SJTU/Bench2Drive.git` (공식), HEAD `2645714`
- **그 파일은 해당 체크아웃에서 커밋되지 않은 로컬 수정 상태(`M`)였다** — `77 insertions(+), 274 deletions(-)`
- md5 — 공식 HEAD 원본: `93c9b4b068b667e2bf0a598aa8ae5aab` / 이 사본: `0c65061599ff0f162777fca15ecf6dad`
- 복사 시점: 2026-08-24, 원본 수정 시각: 2026-08-18 14:12

확인 방법:

```bash
md5sum b2d_controller/comfort_metric/efficiency_smoothness_benchmark.py
# 0c65061599ff0f162777fca15ecf6dad 이어야 함
```

파일에 헤더 주석 한 줄도 덧붙이지 않은 이유가 이 md5 다. 랩 체크아웃의 그 파일과 같은지 언제든
대조할 수 있어야 하기 때문이며, 고칠 일이 생기면 랩 쪽 원본을 고치고 다시 복사해 온 뒤 여기
md5 와 날짜를 갱신한다.

## 공식판과 무엇이 다른가

`git diff` 로 대조한 결과, 이 프로젝트가 `b2d_controller/b2d_metrics.py` 의 `comfort_of()`
docstring 에 기록해 둔 결함 3건과 정확히 일치한다 (+ 관련 수정 1건):

1. **자이로 단위 미변환** — CARLA 의 angular_velocity 는 deg/s 인데 한계값
   `MAX_ABS_YAW_RATE = 0.95` 는 rad/s 다. 공식판은 변환 없이 비교했다.
   `np.deg2rad(...)` 추가.
2. **yaw 가속도가 미분되지 않음** — 공식판의 `_z_yaw_acc` 는 `deriv=` 인자 없는
   `savgol_filter` 결과, 즉 yaw rate 의 평활 사본이었다. 여섯 채널 중 yaw rate 를 두 번 검사하고
   yaw 가속도는 한 번도 검사하지 않고 있었다. `deriv=1, delta=dt` 추가.
3. **미분 스텝 불일치** — `time_interval` 기본값이 0.1 s 인데 실제 샘플 간격은 0.05 s.
   `CARLA_TICK_SECONDS = 0.05` 로 교체.
4. (관련) **yaw rate 에 대한 phase-unwrap 제거** — unwrap 은 heading 같은 순환각에나 의미가
   있고 이미 연속인 rate 신호에는 틀린 처리다. 코드에 주석이 남아 있다:
   `# A rate is not an angle, so ...`

즉 **점수가 공식판과 다르게 나오는 것이 정상이고, 그것이 수정의 목적**이다.

### 부르는 이름

문서·발표에서 "공식 Bench2Drive Comfortness" 라고 쓰면 부정확하다. 정확한 표현:

> Bench2Drive 공식 Comfortness 지표를, 확인된 결함 3건(자이로 단위 미변환, yaw 가속도 미분
> 누락, 미분 스텝 불일치)을 수정하여 사용

## 라이선스

상류 Bench2Drive(Thinklab-SJTU)는 CC BY-NC-ND 4.0(저작자표시-비영리-**변경금지**)이다.
이 파일은 공식판의 무수정 복사본이 **아니라 파생물**이므로, 변경금지 조항이 배포를 허용하지
않는다. 비상업적 내부 사용에 한해 두고, 공개 리모트에 올리지 말 것. (2026-08-24 최초 작성 시
"무수정 복사본이라 파생물이 아니다" 라고 적었던 것은 사실 오류였고, 위 `git diff` 확인 후
정정했다.)

## 백업 주의

랩 체크아웃의 그 수정은 **커밋되지 않은 상태**다. 그쪽에서 `git checkout` / `git stash` 를 한 번
실행하면 사라지며, 그 경우 이 사본이 유일한 기록이 된다. 랩 쪽에 브랜치를 만들어 커밋해 두는
편이 안전하다.

## 이전 버전 (2026-08-24 에 교체됨)

2026-08-24 이전에는 이 자리에 공개판(md5 `93c9b4b0...`)에 위 수정을 직접 재구현한 로컬 패치본이
들어 있었다. 랩 머신 원본과 바이트 단위로 대조된 적이 없어 "미검증"으로 표시해 두었던 파일이다.
교체 직전에 `runs/` 의 실제 주행 12개(route17569 ~ route24367, mpckf/pidviz/d950)에 대해 두
파일의 `seg_compute_comfort_metric` 결과를 비교했고 **12개 전부 소수점 4자리까지 일치**했다. 즉
이번 교체로 점수가 바뀌지 않으며, 과거에 그 패치본으로 뽑아 둔 Comfortness 수치도 그대로
유효하다.
(옛 파일이 필요하면 `git show f13aabe:b2d_controller/comfort_metric/efficiency_smoothness_benchmark.py`)
