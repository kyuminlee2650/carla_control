# B2D road boundary 조사 — 인수인계

작성 2026-08-21. 다음 세션이 이어받기 위한 문서.

---

## 0. 한 줄 요약

Bench2Drive(B2D) 학습 파이프라인에서 `PlanMapBoundLoss`가 **인덱싱 오류로 `SolidSolid`(대향차로 중앙선)를 도로 경계로 쓰고 있고**, 진짜 도로 경계는 데이터셋 단계에서 통째로 버려진다. 해결책(차선 위상 기반 `Boundary` 클래스 라벨링)을 설계·구현·검증까지 마쳤다. **아직 저장소에는 적용하지 않았다.**

---

## 1. 문제: 버그 체인 4단

### (a) 인덱스 오류
`mmcv/models/vad_utils/plan_loss.py:27` 의 `lane_bound_cls_idx=2` 는 nuScenes 클래스 리스트
`['divider','ped_crossing','boundary']` 기준 기본값이다. B2D 클래스 리스트는

```python
# adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py:129
map_classes = ['Broken','Solid','SolidSolid','Center','TrafficLight','StopSign']
```

이므로 **인덱스 2 = `SolidSolid`**. config(`:482`)에서 이 값을 한 번도 override하지 않아 조용히 기본값으로 굴러간다.

### (b) SolidSolid는 경계가 아니라 중앙선
Town03/Town05의 모든 Driving 차선 좌/우 side를 훑어, "이웃이 Driving이 아니거나 없는 side"(= 진짜 주행가능영역 가장자리)와 마킹 타입을 교차한 결과:

| | Town03 | Town05 |
|---|---|---|
| SolidSolid 중 실제 가장자리 | 293 / 1055 (28%) | 161 / 1089 (15%) |
| 실제 가장자리 중 SolidSolid가 덮는 비율 | 293 / 9034 (**3.2%**) | 161 / 10336 (**1.6%**) |

즉 손실은 대부분 0이고, 켜질 때는 엉뚱한 곳(중앙선)을 밀어낸다.

### (c) 진짜 가장자리는 데이터셋 단계에서 소멸
`mmcv/datasets/B2D_vad_dataset.py:273` 이 `map_element_class` 에 없는 타입을 `continue` 로 스킵한다.
Town03 실제 가장자리의 마킹 타입 분포:

```
Solid 3955 / NONE 3268 / SolidBroken 701 / BrokenSolid 636 / SolidSolid 293 / Broken 181
```

`NONE`(선이 아예 없는 연석·갓길 경계)이 36%인데 전부 드롭된다.

### (d) 마킹 타입으로는 원리적으로 못 찾는다
`Curb` roadMark는 .xodr에 존재하지만 **전부 비주행 차선에만** 붙어 있다:

```
curb on lane type shoulder : 264
curb on lane type sidewalk : 237
curb on lane type none     : 192
driving lane 에 붙은 roadMark: {solid: 863, broken: 403, none: 619}
```

`gen_hdmap.py` 는 `carla_map.get_topology()` 로 주행 차선만 훑으므로 `Curb` 를 볼 일이 없다.

---

## 2. 데이터 구조 (조사 결과)

### 단계 1 — `TownXX_HD_map.npz` (월드 좌표계 원본)

```
dict[road_id] -> dict[lane_id] -> list[entry]
```

**마킹 엔트리** (Town03에 1728개), 키 = `(Color, Points, Topology, Type)`:

```python
{'Points': [((4100.5833, 3616.5604, 370.9844), (0.0, -0.0678, 99.0333)), ...],   # ((x,y,z),(roll,pitch,yaw))
 'Type': 'Broken', 'Color': 'White', 'Topology': [(7197,-3),(7187,-1)]}
```

**Center 엔트리** (510개), 키 = `(Color, Left, Points, Right, Topology, TopologyType, Type)`:

```python
{'Points': [((x,y,z),(roll,pitch,yaw), False), ...],   # 3번째 원소 = is_junction (점마다)
 'Type': 'Center', 'Color': 'White', 'Topology': [...],
 'TopologyType': 'Normal',      # Normal/Junction/EnterJunction/PassJunction/...
 'Left': (57,-2), 'Right': (57,-4)}    # 이웃 차선의 (road_id, lane_id)
```

확인된 사실:

- **마킹 폴리라인 = 차선 경계선 좌표.** `gen_hdmap.py` 가 중심선 waypoint(5cm 간격)를 `±0.5*lane_width` 만큼 right vector 방향으로 민 것. 실측: 중심선까지 거리 전 구간 1.7500 m 일정, forward 성분 0.
- 페인트 자체의 기하가 아니다. 복선(SolidSolid)도 폴리라인 하나.
- **roll/pitch/yaw는 중심선 waypoint에서 그대로 복사**된다 (`마킹 rpy == Center rpy`, `np.allclose` True). yaw는 접선 방향이 맞다 (좌표에서 뽑은 접선과 평균 오차 0.07~0.21도). 램프에서는 `[-180,180]` 밖 값이 나온다 (예: -530.78).
- **인접 차선은 같은 물리적 선을 두 번 저장**한다 (lane -3의 좌측 마킹 == lane -2의 우측 마킹, 좌표 동일).
- `road_id` = OpenDRIVE `<road>`. **교차로를 지나는 경로 하나하나가 별도 road.** Town03 .xodr: 총 369개 = 일반 82 + 교차로 연결로 287, junction 35개. driving 차선 0개인 road 121개를 빼면 CARLA가 보여주는 248개와 정확히 일치.
- `lane_id` = **기준선에서 바깥쪽으로 세는 번호이지 "몇 차선"이 아니다.** 0 = 기준선(폭 없음), 음수 = `+s` 방향 주행, 양수 = 반대 방향. 비주행 차선도 번호를 차지한다. Town03에서 가장 안쪽 Driving 차선의 `|lane_id|` 분포 = `{1:224, 2:30, 3:10, 4:31}`.
- `Left`/`Right` = **주행방향 기준. 양쪽 부호 모두 Left=기준선 쪽, Right=바깥쪽.** `lane_type` 필터가 없어 Shoulder/Bidirectional 도 반환된다. 값은 세그먼트의 마지막 waypoint에서 한 번만 샘플링되지만, 구간 내부에서 경계 여부가 뒤바뀌는 경우는 Town03 344개 side 중 0개, Town05 436개 중 0개 → 안전.
- `Topology` = `waypoint.next(0.05)` 의 Driving 후속 `(road_id, lane_id)` 리스트. 무작위 waypoint에서는 98.4%가 자기 자신. **flush 지점에서 계산돼 도로 중간 세그먼트에서는 어긋난다** (`lane_id` 가 바뀌어 트리거된 경우 이미 다음 차선의 waypoint에서 계산. 예: Town03 road 1608의 lane -2, -3이 둘 다 `[(1608,-1)]`). **우리 작업엔 쓰지 않는다.**

### 단계 2 — `b2d_map_infos.pkl` (`prepare_B2D.gengrate_map()`)

**여기서 road/lane 구조가 완전히 무너진다.** `road_id`/`lane_id` 는 `for` 순회 변수일 뿐 어느 리스트에도 저장되지 않는다.

```python
map_infos['Town12'] = {
  'lane_points':        [ndarray(N,3), ...],   # y *= -1, location 만
  'lane_types':         ['Broken', 'Solid', 'Center', ...],   # 인덱스로만 대응
  'lane_sample_points': [ndarray(~N/50,3), ...],   # 50점마다 하나, 근접필터용
  'trigger_volumes_points'/'_types'/'_sample_points': [...]
}
```

Town02 실측: road 68개 / (road,lane) 키 103개 → 평평한 폴리라인 421개.
**사라지는 것**: `road_id`, `lane_id`, `Left`, `Right`, `Topology`, `TopologyType`, `Color`, rpy, `is_junction`.

→ **경계 판정에 필요한 모든 정보를 손에 쥔 마지막 지점이 이 함수다.** 수정 지점이 여기인 이유.

### 단계 3 — `B2D_VAD_Dataset.get_map_info()` (프레임마다, ego 좌표계)

1. `lane_sample_points` 로 ego 50m 이내 필터
2. `map_element_class` 에 없는 타입 `continue` (여기서 NONE/SolidBroken/BrokenSolid 소멸)
3. `world2lidar` 변환 후 `pc_range` 클리핑 (x∈[-15,15], y∈[-30,30] = 30m × 60m)
4. shapely `LineString` → `LiDARInstanceLines(fixed_num=20)`

### 단계 4 — 모델/손실이 보는 것

```python
distances = np.linspace(0, instance.length, 20)   # 호길이 등간격 20점
```

최종 텐서 `[num_vec=100, num_pts=20, 2]` + 라벨 `[100]`.
관련 config: `map_fixed_ptsnum_per_gt_line = 20`, `map_num_vec = 100`, `point_cloud_range = [-15,-30,-2,15,30,2]`.

---

## 3. 해결책

### 핵심 규칙

> 도로 경계는 마킹 타입이 아니라 **차선 인접 관계**로 정한다.
> 어떤 Driving 차선의 좌/우 side가 경계 ⟺ 그쪽 이웃이 없거나 Driving이 아니고, **교차로가 아닐 것**.

필요한 재료 3개는 전부 npz에 있다:

| 재료 | 출처 |
|---|---|
| 이웃 차선 `(road_id, lane_id)` | `Center` 엔트리의 `Left`/`Right` |
| 그 이웃이 주행 차선인가 | npz의 `(road_id, lane_id)` 키 집합 (주행 차선만 키로 존재) |
| 교차로 제외 | `Center` 엔트리의 `TopologyType` |

**교차로 제외는 필수다.** 빼먹으면 연결로는 이웃이 없는 게 정상이므로 side의 73~80%(Town03 80%, Town05 73%)가 경계로 판정되어 교차로를 가로막는 가짜 벽이 생기고 좌회전이 막힌다. 교차로 내부 waypoint의 이웃 조회는 57%가 `None` 을 반환한다.

**`Left`/`Right` 없이 `lane_id` 크기만으로 추론하면 안 된다.** lane section 정보가 npz에 없어 precision 0.52(Town03)/0.50(Town05) — 동전 던지기다.

### 코드 — `scripts/boundary_patch.py`

```python
def _is_non_driving(nb, driving_keys):
    if nb is None or nb[0] is None:
        return True
    return tuple(nb) not in driving_keys

def boundary_sides(lane_entries, driving_keys):
    """-> {'left': bool, 'right': bool}"""
    ct = next((e for e in lane_entries if e['Type'] == 'Center'), None)
    if ct is None:
        return {'left': False, 'right': False}
    if 'Junction' in ct.get('TopologyType', ''):
        return {'left': False, 'right': False}
    return {'left':  _is_non_driving(ct.get('Left'),  driving_keys),
            'right': _is_non_driving(ct.get('Right'), driving_keys)}

def marking_side(single_lane, center_entry):
    """마킹 폴리라인이 좌측인지 우측인지. Center 면 None."""
    if single_lane['Type'] == 'Center' or center_entry is None:
        return None
    P = np.asarray([rp[0]    for rp in single_lane['Points']],  float)[:, :2]
    C = np.asarray([rp[0]    for rp in center_entry['Points']], float)[:, :2]
    Y = np.asarray([rp[1][2] for rp in center_entry['Points']], float)
    i = int(np.argmin(np.linalg.norm(C - P[0], axis=1)))
    yaw = np.deg2rad(Y[i])
    right = np.array([-np.sin(yaw), np.cos(yaw)])   # CARLA get_right_vector() 와 동일 (오차 4e-7 검증)
    return 'right' if float((P[0] - C[i]) @ right) > 0 else 'left'

def resolve_lane_type(single_lane, center_entry, bsides):
    side = marking_side(single_lane, center_entry)
    return 'Boundary' if (side is not None and bsides[side]) else single_lane['Type']
```

### 적용 지점 1 — `prepare_B2D.py` 의 `gengrate_map()`

> **바로 쓸 수 있는 드롭인 교체본이 `scripts/gengrate_map_patched.py` 에 있다.**
> 판정 로직까지 한 파일에 들어 있고 Town02/Town03 스모크 테스트를 통과했다.
> 아래는 그 중 바뀐 부분만 발췌한 것이다.

평탄화 **직전에** 판정해서 타입 문자열에 실어 보낸다. 출력 형식은 그대로라 하류가 안 깨진다.

```python
# [추가] for 루프 진입 전
driving_keys = {(rid, lid) for rid, rd in map_info.items()
                           for lid in rd if lid != 'Trigger_Volumes'}

for road_id, road in map_info.items():
    for lane_id, lane in road.items():
        if lane_id == 'Trigger_Volumes':
            ...   # 그대로
        else:
            # [추가]
            center_entry = next((e for e in lane if e['Type'] == 'Center'), None)
            bsides = boundary_sides(lane, driving_keys)

            for single_lane in lane:
                points = np.array([raw_point[0] for raw_point in single_lane['Points']])
                points[:, 1] *= -1
                lane_points.append(points)
                # [변경] lane_types.append(single_lane['Type'])
                lane_types.append(resolve_lane_type(single_lane, center_entry, bsides))
                ...   # 그대로
```

### 적용 지점 2 — `B2D_vad_dataset.py:57` (체크포인트 호환 유지)

```python
# 변경 후 — 슬롯 2 를 SolidSolid -> Boundary 로 교체, SolidSolid 는 Solid 에 흡수
self.map_element_class = {'Broken':0, 'Solid':1, 'SolidSolid':1, 'Boundary':2,
                          'Center':3, 'TrafficLight':4, 'StopSign':5}
self.MAPCLASSES = ['Broken', 'Solid', 'Boundary', 'Center', 'TrafficLight', 'StopSign']
self.NUM_MAPCLASSES = len(self.MAPCLASSES)      # 6 유지
```

- `SolidSolid: 1` 로 매핑하지 않으면 `continue` 로 버려진다.
- `MAPCLASSES` 는 반드시 명시. `list(keys())` 로 두면 7개가 되어 `NUM_MAPCLASSES` 가 틀린다.
- 클래스 수 6 유지 → map head 출력 shape 불변 → **기존 체크포인트 로드 후 파인튜닝 가능**.

### 적용 지점 3 — `VAD_base_e2e_b2d.py`

```python
# 129 행
map_classes = ['Broken', 'Solid', 'Boundary', 'Center', 'TrafficLight', 'StopSign']

# 482 행 — 기본값 의존 금지
loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=1.0, dis_thresh=1.0,
                     lane_bound_cls_idx=2),
```

`PlanMapDirectionLoss` 의 `lane_div_cls_idx=0` 은 `Broken` 이라 그대로 두면 된다.

---

## 4. 검증 결과

### 4-1. 라벨링 정확도 (npz만으로 판정 → CARLA .xodr 정답과 대조)

| 타운 | 비교차로 side | precision | recall |
|---|---|---|---|
| Town03 | 344 | **1.0000** (TP 176, FP 0) | **1.0000** (FN 0) |
| Town06 | 510 | **1.0000** (TP 162, FP 0) | **1.0000** (FN 0) |

대조 실패 키 0개 (양방향). 스크립트: `scripts/verify_real.py`, `scripts/xval06.py`

### 4-2. Town03 전체 적용 결과

```
타입 분포  전 : Solid 615, Center 510, NONE 434, Broken 377, SolidSolid 207, SolidBroken 48, BrokenSolid 47
           후 : Center 510, NONE 408, Solid 407, Broken 377, Boundary 273, SolidSolid 204, SolidBroken 31, BrokenSolid 28

Boundary 273개, 각 폴리라인 중점에서 CARLA 조회 → 바깥이 비주행 273 / 주행 0  (100.00%)
(road,lane,side) 커버리지 176/176, 누락 0
중복 제거 후 184개 / 총연장 7786 m  vs  실제 비교차로 가장자리 7908 m  → 98.5%
```

`Boundary` 273개의 출처: Solid 208 + NONE 26 + BrokenSolid 19 + SolidBroken 17 + SolidSolid 3.
**어떤 단일 마킹 타입으로도 못 얻는 집합**이라는 증거.

스크립트: `scripts/validate_patch.py`, `scripts/recall.py`, `scripts/dedup.py`

### 4-3. b2d 10개 라우트 — 프레임당 경계 가용성 (핵심 지표)

대상: `run_eval10_ab.sh` 의 `ROUTES=(24367 27582 3144 2416 2715 2286 17569 2790 3373 1792)`
방법: 계획 경로를 4m 간격 보간 → 도로 스냅 → 각 자세에서 BEV 박스(x±15, y±30) 안의 인스턴스 카운트
(`get_map_info()` 와 동일하게 박스 안 점 2개 이상이어야 인스턴스)

| route | town | scenario | frm | SS %frm | SS inst | SS dist | BD %frm | BD inst | BD dist |
|---|---|---|---|---|---|---|---|---|---|
| 24367 | Town06 | ConstructionObstacle | 68 | **0.0** | 0.00 | – | 100.0 | 1.00 | 1.8 |
| 27582 | Town11 | PedestrianCrossing | 29 | 89.7 | 9.45 | 1.8 | 100.0 | 6.38 | 1.8 |
| 3144 | Town12 | VanillaSignalizedTurnEncounterRedLight | 21 | 100.0 | 2.76 | 4.9 | 100.0 | 4.19 | 1.6 |
| 2416 | Town12 | VanillaNonSignalizedTurnEncounterStopsign | 21 | **0.0** | 0.00 | – | 100.0 | 5.90 | 1.4 |
| 2715 | Town12 | StaticCutIn | 68 | 100.0 | 2.00 | 1.6 | 100.0 | 2.00 | 4.9 |
| 2286 | Town12 | HighwayCutIn | 40 | **0.0** | 0.00 | – | 100.0 | 3.25 | 5.2 |
| 17569 | Town12 | SequentialLaneChange | 24 | **0.0** | 0.00 | – | 100.0 | 2.00 | 1.8 |
| 2790 | Town12 | InvadingTurn | 52 | 100.0 | 2.73 | 1.6 | 100.0 | 3.27 | 4.9 |
| 3373 | Town13 | YieldToEmergencyVehicle | 53 | **0.0** | 0.00 | – | 100.0 | 2.00 | 5.2 |
| 1792 | Town12 | HazardAtSideLane | 68 | **0.0** | 0.00 | – | 100.0 | 2.00 | 1.7 |
| **가중평균** | | | 444 | **37.6** | 1.37 | | **100.0** | 2.68 | |

`SS` = SolidSolid(현재), `BD` = Boundary(수정 후), `dist` = ego~최근접점 거리 중앙값(m).

**10개 중 6개 라우트에서 SolidSolid가 단 한 프레임도 안 잡힌다** → 그 라우트 전 구간에서 `PlanMapBoundLoss` 가 항상 0.
route 3144는 SolidSolid가 4.9m 떨어진 중앙선인데 실제 도로 끝은 1.6m — 잘못 짚을 뿐 아니라 엉뚱하게 멀다.

산출물: `b2d10_boundary.png`, `metrics.json`. 스크립트: `scripts/metrics.py`, `scripts/viz10.py`

### 4-4. 시각 검증

- `town03_boundary.png` — Town03 전체 + 확대 4패널. 빨강=Boundary, 파랑=그 외 마킹, 회색=주행가능영역.
- `b2d10_boundary.png` — 10개 라우트 패널. 빨강=Boundary, 주황=SolidSolid, 검정=계획경로, 초록점선=ego BEV 박스.

확인 포인트: 빨강이 도로 바깥 윤곽만 따라가고 주행영역 한복판을 가로지르지 않을 것, 교차로에서 끊길 것, 다차선 도로에서 바깥 두 줄만 빨강일 것.

---

## 5. 미해결 / 다음 할 일

### (1) 저장소 적용 — **아직 안 함**
위 3장의 세 지점을 실제로 수정하는 작업. 사용자 승인 대기 중이었다.
적용 전에 **어느 `b2d_map_infos.pkl` 을 학습에 쓰는지 확인 필요** (아래 6장에 3개 경로).

### (2) Town11/12/13 npz 직접 검증 미완 — 그리고 잠재적 OOM
`Town11/12/13_HD_map.npz` (1.6/1.4/1.9 GB) 는 **언피클에 16 GB로도 부족**하다 (`MemoryError`, `scripts/memprobe.py`).
공유 워크스테이션(총 31 GB, 타인 8 GB 사용 중)이라 상한을 더 올리지 않았다.

- 그래서 10-route 지표의 Town11/12/13 부분은 npz가 아니라 **동일한 CARLA 소스(.xodr)에서 `gen_hdmap.py` 로직을 코리도 주변에만 재현**해 뽑았다 (`scripts/carla_corridor.py`). Town06 만 npz 직접 처리(`scripts/label_town.py`).
- **더 중요한 함의**: 학습 데이터 준비 시 `gengrate_map()` 이 이 파일들을 통째로 로드하므로 **같은 OOM이 날 가능성이 높다**. 지금까지 어떻게 돌렸는지 확인이 필요하다 (더 큰 장비? 스왑? 아니면 애초에 Town11/12/13을 안 썼나?). 이건 경계 문제와 별개의 잠재 이슈다.

### (3) 중복 엔트리 — **패치에 반영 완료, 다만 검토 필요**
`gen_hdmap.py` 의 flush 결함으로 완전히 동일한 폴리라인이 여러 벌 들어간다.
Town03 Boundary 273개 중 89개가 중복, 총연장이 실제의 1.7배. Town12(부분본) 160개 중 17개 중복.

`scripts/gengrate_map_patched.py` 의 `DEDUP = True` 로 처리해 뒀다. Town03 전체 2238 → 1673.
**다만 이 전역 중복 제거는 두 종류를 함께 없앤다는 점을 검토해야 한다:**
- (a) 같은 lane 안의 중복 — 순수 버그 수정. Boundary 273 → 184.
- (b) 인접 차선이 공유하는 물리적으로 같은 선 — Broken 377 → 184.
  "물리적 선 하나 = GT 인스턴스 하나" 가 옳다고 보고 함께 제거했지만, 경계 라벨링과는
  독립적인 동작 변경이다. 맵 head 학습에 미치는 영향은 아직 확인하지 않았다.
  원래 동작을 원하면 `DEDUP = False`.

### (4) `get_map_info()` 통과 후 ego 좌표계 검증 — 미착수
지금까지 검증은 전부 월드 좌표계 원본이다. 실제 학습 입력은 30×60m 클리핑 + 20점 리샘플을 거친다. 깨질 수 있는 지점:
- **클리핑**: 폴리라인이 박스를 나갔다 다시 들어오면 `points_in_lidar[mask]` 가 불연속 조각을 하나의 `LineString` 으로 이어붙여 가짜 선분이 생긴다.
- **20점 리샘플**: 342m 폴리라인이 20점이면 간격 18m. 손실은 이 20점과의 최소거리만 보므로 경계가 실제보다 멀게 계산될 수 있다.

### (5) 재학습
손실은 **예측된** map(`lane_preds`)을 쓰므로 클래스 2의 의미가 바뀐 만큼 map head 파인튜닝이 필요하다.
shape 불변이라 가중치는 로드된다. 초기에는 `PlanMapBoundLoss(perception_detach=True)` 로 지각 head와 분리해 보는 것도 방법.

---

## 6. 환경 / 경로

### 코드
```
/home/ailab/2026intern/kmlee/tt_closedloop/Bench2DriveZoo/     # 주 작업 대상
/home/ailab/2026intern/kmlee/vad_demo_video/Bench2DriveZoo/    # 동일 코드 사본
/home/ailab/2026intern/kmlee/vad_demo_video/Bench2Drive/tools/gen_hdmap.py   # 맵 생성 원본
```
관련 파일: `mmcv/models/vad_utils/plan_loss.py`, `mmcv/datasets/prepare_B2D.py`,
`mmcv/datasets/B2D_vad_dataset.py`, `mmcv/datasets/map_utils/struct.py`,
`adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py`

### 데이터
```
/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/    # TownXX_HD_map.npz 13개
/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/b2d_pkl/b2d_map_infos.pkl          # 6.3 GB
/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/cache/infos/b2d_map_infos.pkl      # 6.3 GB
/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/b2d_pkl/infos_rals/b2d_map_infos.pkl  # 6.3 GB
```
config의 `map_root = "data/bench2drive/maps"`, `map_file = "data/infos/b2d_map_infos.pkl"` 는 상대경로이고
`tt_closedloop/Bench2DriveZoo/data/` 는 비어 있다. **어느 것을 심볼릭 링크로 쓰는지 미확인.**

### CARLA (시뮬레이터 없이 .xodr만으로 `carla.Map()` 사용 가능)
```
/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{Town}.xodr        # Town01~10
/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/{Town}/OpenDrive/{Town}.xodr # Town11~15
```

### 라우트 정의
```
/home/ailab/2026intern/jsn/2026-Summer-Internship/pipeline/data/bench2drive10_abilities.xml
/home/ailab/2026intern/kmlee/vad_demo_video/run_eval10_ab.sh    # ROUTES 배열
```

### Python 환경
```
/home/ailab/miniconda3/envs/pdm/bin/python      # carla + matplotlib 3.7.5  ← 이 조사에 사용
/home/ailab/miniconda3/envs/b2d/bin/python      # carla
/home/ailab/miniconda3/envs/b2d_zoo/bin/python  # carla(egg) + matplotlib, 학습/평가용
```

---

## 7. 스크립트 (`scripts/`)

> **전체 소스는 `CODE.md` 에 인라인으로 들어 있다.** 파일에 접근할 수 없어도 그 문서만으로 재현 가능하다.
> 30개 스크립트 996줄 전부 포함.

핵심:
- `gengrate_map_patched.py` — **`gengrate_map()` 드롭인 교체본.** 판정 로직 + DEDUP 포함, 스모크 테스트 통과. 실제 수정은 이 파일만 있으면 된다.
- `boundary_patch.py` — 판정 로직 본체만 분리한 것. 다른 스크립트들이 import 한다.
- `verify_real.py` / `xval06.py` — npz 규칙을 CARLA 정답과 대조 (Town03 / Town06)
- `validate_patch.py` — Town03 전체 적용 + 폴리라인 단위 검증
- `recall.py` / `dedup.py` — 누락·중복 정량화
- `corridor.py <Town>` — 라우트 코리도 생성 (XML → 보간 → 도로 스냅)
- `carla_corridor.py <Town>` — CARLA에서 코리도 주변 폴리라인 재현 + 라벨링
- `label_town.py <Town>` — npz에서 코리도 주변 폴리라인 추출 + 라벨링 (소형 타운용)
- `metrics.py` — 10-route 지표 표 생성 → `metrics.json`
- `viz10.py` — 10-route 그림 → `b2d10_boundary.png`
- `viz_town.py` — 타운 전체 그림 → `town03_boundary.png`

조사 과정 (근거 재현용):
- `edge.py` — 가장자리 vs 내부 마킹 타입 분포
- `junc.py` / `junc2.py` / `overlap.py` — 교차로 문제 정량화
- `roadlane.py` / `roadlane2.py` — road_id/lane_id 의미 확인
- `topo.py` / `demo4.py` — Topology / Left / Right 확인
- `xodr.py` — 일반구간 vs 연결로 비교
- `nbrrule.py` / `proxy.py` / `stab.py` — 판정 규칙 후보 비교 (Left/Right 방식이 유일하게 1.000)
- `sidecheck.py` — right vector 공식 검증
- `stage2.py` — `gengrate_map()` 재현
- `memprobe.py` — npz 메모리 요구량 측정
- `hist.py` — 맵 타입 히스토그램
- `collect_traj.py` — 런 덤프에서 ego 궤적 수집

실행 예:
```bash
cd /home/ailab/carla_control/boundary_investigation/scripts
/home/ailab/miniconda3/envs/pdm/bin/python verify_real.py
/home/ailab/miniconda3/envs/pdm/bin/python corridor.py Town12
/home/ailab/miniconda3/envs/pdm/bin/python carla_corridor.py Town12
/home/ailab/miniconda3/envs/pdm/bin/python metrics.py
```

주의: `corridor.py` → `carla_corridor.py` → `metrics.py` → `viz10.py` 순서 의존.
중간 산출물(`corridor_*.npy`, `carla_*.npy`)은 스크립트 실행 디렉토리에 생긴다.

---

## 8. 사용자 맥락

- 사용자는 이 조사를 단계별로 하나씩 이해하며 진행하기를 원했다. 데이터 구조를 npz → pkl → dataset → model 순으로 짚어가며 확인했다.
- 추측이 아니라 **실측으로 뒷받침된 답**을 원한다. 숫자 없이 "그럴 것이다" 로 답하지 말 것.
- 본인이 직접 검증하기를 원해 시각화를 요청했다. 새 결과를 낼 때 그림과 수치를 같이 주는 게 좋다.
- 공유 워크스테이션이다. 메모리/디스크를 크게 쓰는 작업은 상한을 걸고 할 것.
