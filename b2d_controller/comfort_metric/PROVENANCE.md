# efficiency_smoothness_benchmark.py

Bench2Drive 의 Comfortness 채점 모듈. **이 저장소가 다시 구현한 것이 아니라 랩 머신에서 실제로
점수를 매기는 그 파일을 바이트 그대로 복사한 것**이다.

- 원본 경로: `/home/ailab/2026intern/kmlee/vad_demo_video/Bench2Drive/tools/efficiency_smoothness_benchmark.py`
  (우분투 랩 머신)
- md5: `0c65061599ff0f162777fca15ecf6dad`
- 복사 시점: 2026-08-24
- 원본 수정 시각: 2026-08-18 (이 프로젝트가 보고한 버그 3개가 반영된 버전)

파일에 헤더 주석 한 줄도 덧붙이지 않은 이유는 md5 가 그대로 유지되어야 원본과 같은 파일인지
언제든 확인할 수 있기 때문이다. 확인 방법:

```bash
md5sum b2d_controller/comfort_metric/efficiency_smoothness_benchmark.py
# 0c65061599ff0f162777fca15ecf6dad 이어야 함
```

## 이전 버전 (2026-08-24 에 교체됨)

2026-08-24 이전에는 이 자리에 공개판 tag 0.0.4(md5 `93c9b4b0...`)에 버그 3개 수정을
직접 재구현한 로컬 패치본이 들어 있었다. 랩 머신 원본과 바이트 단위로 대조된 적이 없어
"미검증"으로 표시해 두었던 파일이다. 교체 직전에 `runs/` 의 실제 주행 12개(route17569 ~
route24367, mpckf/pidviz/d950)에 대해 두 파일의 `seg_compute_comfort_metric` 결과를
비교했고 **12개 전부 소수점 4자리까지 완전히 일치**했다. 즉 이번 교체로 점수가 바뀌지 않으며,
과거에 그 패치본으로 뽑아 둔 Comfortness 수치도 그대로 유효하다.
(옛 파일이 필요하면 `git show f13aabe:b2d_controller/comfort_metric/efficiency_smoothness_benchmark.py`)

## 라이선스

상류 Bench2Drive(Thinklab-SJTU)는 CC BY-NC-ND 4.0 이다. 이 파일은 랩 머신 원본의
**무수정 복사본**이므로 파생물이 아니고, 비상업적 이용과 출처 표시 조건에서 그대로 둘 수 있다.
수정하지 말 것 -- 고칠 것이 생기면 랩 머신 원본을 고치고 다시 복사해 오고, 이 문서의 md5 와
날짜를 같이 갱신한다.
