# 코드 부록 — 전체 소스

`HANDOFF.md` 의 부속 문서. 조사에 쓴 **모든 스크립트의 전체 소스**를 담았다.
파일에 접근할 수 없는 경우에도 이 문서만으로 재현할 수 있다.
실제 파일은 `scripts/` 에 같은 이름으로 있다.

실행 환경: `/home/ailab/miniconda3/envs/pdm/bin/python` (carla + matplotlib)

## A. 적용할 코드 (이 둘만 있으면 수정 가능)

### `gengrate_map_patched.py`

prepare_B2D.py 의 gengrate_map() 드롭인 교체본. 판정 로직 포함, 스모크 테스트 통과.

```python
"""prepare_B2D.py 의 gengrate_map() 드롭인 교체본.

원본: Bench2DriveZoo/mmcv/datasets/prepare_B2D.py 의 gengrate_map()
바뀐 곳은 [ADDED] / [CHANGED] 주석이 붙은 줄뿐이고, 출력 형식(평평한 리스트 4종)은 원본과 동일하다.
따라서 하류(get_map_info, 데이터셋, 모델)는 클래스 딕셔너리만 고치면 된다.

적용법
  1) 이 파일의 _is_non_driving / boundary_sides / marking_side / resolve_lane_type 를
     prepare_B2D.py 상단에 붙여넣는다 (또는 boundary_patch 를 import).
  2) 원본 gengrate_map() 을 아래 gengrate_map() 으로 교체한다.
  3) B2D_vad_dataset.py 의 map_element_class / MAPCLASSES 를 HANDOFF.md 3장대로 고친다.
  4) VAD_base_e2e_b2d.py 의 map_classes 와 lane_bound_cls_idx 를 고친다.

주의
  - Town11/12/13 npz 는 언피클에 16 GB 로도 부족하다 (HANDOFF.md 5-(2)). 이 함수는 원본과
    마찬가지로 npz 를 통째로 로드하므로 그 타운들에서 OOM 이 날 수 있다. 별도 대응 필요.
  - DEDUP 은 좌표가 완전히 같은 폴리라인을 하나만 남긴다. 두 종류를 함께 제거한다.
      (a) gen_hdmap.py 의 flush 결함으로 같은 lane 안에 생긴 중복
          Town03 Boundary 273 -> 184 (89개 제거)
      (b) 인접 차선이 공유하는 물리적으로 같은 선 (lane -3 의 좌측 마킹 == lane -2 의 우측 마킹)
          Town03 Broken 377 -> 184
    (a)는 순수한 버그 수정이고, (b)는 "한 개의 물리적 선은 GT 인스턴스 하나" 라는 쪽이
    옳다고 보아 함께 제거했다. 다만 경계 라벨링과는 독립적인 동작 변경이므로 토글로 뒀다.
    Town03 전체: 폴리라인 2238 -> 1673.

검증 (Town02/Town03 스모크 테스트 통과)
    Town03: Center 422, NONE 361, Solid 337, Broken 184, Boundary 184, SolidSolid 135,
            SolidBroken 26, BrokenSolid 24
    Town02: Broken 128, NONE 92, Center 88, Boundary 40
    출력 pkl 의 키 6종과 리스트 길이 정합성은 원본과 동일.
"""

import os
import pickle
from os.path import join

import numpy as np

DEDUP = True          # 완전 중복 폴리라인 제거 (HANDOFF.md 5-(3))


# ---------------------------------------------------------------------------
# 경계 판정 로직  (scripts/boundary_patch.py 와 동일)
# ---------------------------------------------------------------------------

def _is_non_driving(nb, driving_keys):
    """이웃 (road_id, lane_id) 이 주행 차선이 아닌가. npz 키에 없으면 비주행."""
    if nb is None or nb[0] is None:
        return True                                   # 이웃 자체가 없음 = 도로 끝
    return tuple(nb) not in driving_keys


def boundary_sides(lane_entries, driving_keys):
    """이 차선의 좌/우 각각이 도로 경계인지. -> {'left': bool, 'right': bool}"""
    ct = next((e for e in lane_entries if e['Type'] == 'Center'), None)
    if ct is None:
        return {'left': False, 'right': False}
    if 'Junction' in ct.get('TopologyType', ''):
        # 교차로 연결로는 이웃이 없는 게 정상이다. 제외하지 않으면 side 의 73~80% 가
        # 경계로 판정되어 교차로를 가로막는 가짜 벽이 생긴다.
        return {'left': False, 'right': False}
    return {'left':  _is_non_driving(ct.get('Left'),  driving_keys),
            'right': _is_non_driving(ct.get('Right'), driving_keys)}


def marking_side(single_lane, center_entry):
    """마킹 폴리라인이 중심선 기준 좌측인지 우측인지. Center 면 None."""
    if single_lane['Type'] == 'Center' or center_entry is None:
        return None
    P = np.asarray([rp[0]    for rp in single_lane['Points']],  float)[:, :2]
    C = np.asarray([rp[0]    for rp in center_entry['Points']], float)[:, :2]
    Y = np.asarray([rp[1][2] for rp in center_entry['Points']], float)      # yaw
    i = int(np.argmin(np.linalg.norm(C - P[0], axis=1)))                    # 최근접 중심선 점
    yaw = np.deg2rad(Y[i])
    right = np.array([-np.sin(yaw), np.cos(yaw)])   # CARLA get_right_vector() 와 동일 (오차 4e-7 검증)
    return 'right' if float((P[0] - C[i]) @ right) > 0 else 'left'


def resolve_lane_type(single_lane, center_entry, bsides):
    """이 폴리라인에 붙일 최종 타입 문자열."""
    side = marking_side(single_lane, center_entry)
    return 'Boundary' if (side is not None and bsides[side]) else single_lane['Type']


# ---------------------------------------------------------------------------
# gengrate_map()  드롭인 교체본
# ---------------------------------------------------------------------------

def gengrate_map(map_root, out_dir):
    map_infos = {}
    for file_name in os.listdir(map_root):
        if '.npz' in file_name:
            map_info = dict(np.load(join(map_root, file_name), allow_pickle=True)['arr'])
            town_name = file_name.split('_')[0]
            map_infos[town_name] = {}
            lane_points = []
            lane_types = []
            lane_sample_points = []
            trigger_volumes_points = []
            trigger_volumes_types = []
            trigger_volumes_sample_points = []

            # [ADDED] 주행 차선 키 집합. npz 는 주행 차선만 키로 갖는다.
            driving_keys = {(rid, lid) for rid, rd in map_info.items()
                                       for lid in rd if lid != 'Trigger_Volumes'}
            seen = []                                          # [ADDED] 중복 제거용

            for road_id, road in map_info.items():
                for lane_id, lane in road.items():
                    if lane_id == 'Trigger_Volumes':
                        for single_trigger_volume in lane:
                            points = np.array(single_trigger_volume['Points'])
                            points[:, 1] *= -1                 # left2right
                            trigger_volumes_points.append(points)
                            trigger_volumes_sample_points.append(points.mean(axis=0))
                            trigger_volumes_types.append(single_trigger_volume['Type'])
                    else:
                        # [ADDED] 평탄화 직전에 경계 여부를 판정한다. 여기가 road_id/lane_id/
                        #         Left/Right/TopologyType 을 손에 쥔 마지막 지점이다.
                        center_entry = next((e for e in lane if e['Type'] == 'Center'), None)
                        bsides = boundary_sides(lane, driving_keys)

                        for single_lane in lane:
                            points = np.array([raw_point[0] for raw_point in single_lane['Points']])
                            points[:, 1] *= -1

                            # [ADDED] 반드시 flip 이후에 비교할 것. points[:,1] *= -1 은
                            #         in-place 라, flip 전에 seen 에 넣으면 저장해 둔 배열까지
                            #         같이 뒤집혀 이후 비교가 전부 실패한다.
                            if DEDUP:
                                if any(q.shape == points.shape and np.allclose(q, points)
                                       for q in seen):
                                    continue
                                seen.append(points)

                            lane_points.append(points)

                            # [CHANGED] lane_types.append(single_lane['Type'])
                            lane_types.append(
                                resolve_lane_type(single_lane, center_entry, bsides))

                            lane_lenth = points.shape[0]
                            if lane_lenth % 50 != 0:
                                devide_points = [50 * i for i in range(lane_lenth // 50 + 1)]
                            else:
                                devide_points = [50 * i for i in range(lane_lenth // 50)]
                            devide_points.append(lane_lenth - 1)
                            lane_sample_points.append(points[devide_points])

            map_infos[town_name]['lane_points'] = lane_points
            map_infos[town_name]['lane_sample_points'] = lane_sample_points
            map_infos[town_name]['lane_types'] = lane_types
            map_infos[town_name]['trigger_volumes_points'] = trigger_volumes_points
            map_infos[town_name]['trigger_volumes_sample_points'] = trigger_volumes_sample_points
            map_infos[town_name]['trigger_volumes_types'] = trigger_volumes_types

            import collections
            print(f'{town_name}: 폴리라인 {len(lane_points)}  '
                  f'{dict(collections.Counter(lane_types).most_common())}', flush=True)

    with open(join(out_dir, 'b2d_map_infos.pkl'), 'wb') as f:
        pickle.dump(map_infos, f)


if __name__ == '__main__':
    import sys
    # 예: python gengrate_map_patched.py <map_root> <out_dir>
    gengrate_map(sys.argv[1], sys.argv[2])
```

### `boundary_patch.py`

경계 판정 로직 본체만 분리한 것. 다른 스크립트들이 import 한다.

```python
"""prepare_B2D.gengrate_map() 에 넣을 경계 판정 로직."""
import numpy as np


def _is_non_driving(nb, driving_keys):
    """이웃 (road_id, lane_id) 이 주행 차선이 아닌가. npz 키에 없으면 비주행."""
    if nb is None or nb[0] is None:
        return True
    return tuple(nb) not in driving_keys


def boundary_sides(lane_entries, driving_keys):
    """이 차선의 좌/우 각각이 도로 경계인지 판정. -> {'left': bool, 'right': bool}"""
    ct = next((e for e in lane_entries if e['Type'] == 'Center'), None)
    if ct is None:
        return {'left': False, 'right': False}
    if 'Junction' in ct.get('TopologyType', ''):
        return {'left': False, 'right': False}          # 교차로 내부엔 경계 없음
    return {'left':  _is_non_driving(ct.get('Left'),  driving_keys),
            'right': _is_non_driving(ct.get('Right'), driving_keys)}


def marking_side(single_lane, center_entry):
    """마킹 폴리라인이 중심선 기준 좌측인지 우측인지. Center 면 None."""
    if single_lane['Type'] == 'Center' or center_entry is None:
        return None
    P = np.asarray([rp[0] for rp in single_lane['Points']], dtype=float)[:, :2]
    C = np.asarray([rp[0] for rp in center_entry['Points']], dtype=float)[:, :2]
    Y = np.asarray([rp[1][2] for rp in center_entry['Points']], dtype=float)
    i = int(np.argmin(np.linalg.norm(C - P[0], axis=1)))    # 가장 가까운 중심선 점
    yaw = np.deg2rad(Y[i])
    right = np.array([-np.sin(yaw), np.cos(yaw)])           # CARLA get_right_vector() 와 동일 (검증됨)
    return 'right' if float((P[0] - C[i]) @ right) > 0 else 'left'


def resolve_lane_type(single_lane, center_entry, bsides):
    """이 폴리라인에 붙일 최종 타입 문자열."""
    side = marking_side(single_lane, center_entry)
    if side is not None and bsides[side]:
        return 'Boundary'
    return single_lane['Type']
```

## B. 검증

### `verify_real.py`

공식 Town03 npz 판정 -> CARLA .xodr 정답 대조. precision/recall 1.0000.

```python
import numpy as np, collections, carla

NPZ = '/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz'
m = dict(np.load(NPZ, allow_pickle=True)['arr'])

# --- npz 만으로 내리는 판정 ---
driving_keys = {(rid, lid) for rid, road in m.items() for lid in road if lid != 'Trigger_Volumes'}
pred = {}
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type'] == 'Center'), None)
        if ct is None: continue
        if 'Junction' in ct['TopologyType']: continue          # 교차로 제외
        for side in ('Left', 'Right'):
            nb = ct[side]
            pred[(rid, lid, side)] = (nb is None or nb[0] is None or tuple(nb) not in driving_keys)

# --- CARLA 정답 ---
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
truth = {}
for w in cm.generate_waypoints(1.0):
    if w.lane_type != carla.LaneType.Driving or w.is_junction: continue
    for side, nb in (('Left', w.get_left_lane()), ('Right', w.get_right_lane())):
        truth.setdefault((w.road_id, w.lane_id, side), set()).add(
            (nb is None) or (nb.lane_type != carla.LaneType.Driving))

common = set(pred) & {k for k, v in truth.items() if len(v) == 1}
st = collections.Counter((pred[k], next(iter(truth[k]))) for k in common)
tp, fp, fn, tn = st[(1,1)], st[(1,0)], st[(0,1)], st[(0,0)]
print(f"공식 Town03 npz 로만 내린 경계 판정  (비교차로, 대조 가능한 {len(common)} side)")
print(f"   TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"   precision {tp/max(tp+fp,1):.4f}   recall {tp/max(tp+fn,1):.4f}")
print(f"\n   npz 에만 있고 CARLA 대조 불가: {len(set(pred)-set(truth))}"
      f" | CARLA 에만: {len(set(truth)-set(pred))}")
print(f"   경계로 판정된 side: {sum(pred.values())} / {len(pred)}")
```

### `xval06.py`

같은 검증을 Town06 으로. precision/recall 1.0000.

```python
import numpy as np, collections, carla, boundary_patch as bp
town='Town06'
m = dict(np.load(f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{town}_HD_map.npz', allow_pickle=True)['arr'])
dk = {(r,l) for r,rd in m.items() for l in rd if l!='Trigger_Volumes'}
pred={}
for rid,road in m.items():
    for lid,lane in road.items():
        if lid=='Trigger_Volumes': continue
        ct=next((e for e in lane if e['Type']=='Center'),None)
        if ct is None or 'Junction' in ct.get('TopologyType',''): continue
        bs=bp.boundary_sides(lane,dk)
        for s in ('left','right'): pred[(rid,lid,s)]=bs[s]
cm=carla.Map(town,open(f'/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr').read())
truth={}
for w in cm.generate_waypoints(1.0):
    if w.lane_type!=carla.LaneType.Driving or w.is_junction: continue
    for s,nb in (('left',w.get_left_lane()),('right',w.get_right_lane())):
        truth.setdefault((w.road_id,w.lane_id,s),set()).add((nb is None) or (nb.lane_type!=carla.LaneType.Driving))
common=set(pred)&{k for k,v in truth.items() if len(v)==1}
st=collections.Counter((pred[k],next(iter(truth[k]))) for k in common)
tp,fp,fn,tn=st[(1,1)],st[(1,0)],st[(0,1)],st[(0,0)]
print(f"{town} npz-only 규칙 (비교차로 {len(common)} side): TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"   precision {tp/max(tp+fp,1):.4f}   recall {tp/max(tp+fn,1):.4f}")
print(f"   npz 에만 {len(set(pred)-set(truth))} | CARLA 에만 {len(set(truth)-set(pred))}")
```

### `validate_patch.py`

Town03 전체 적용 + 폴리라인 중점에서 CARLA 재조회 (273/273).

```python
import numpy as np, collections, carla, boundary_patch as bp

NPZ = '/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz'
m = dict(np.load(NPZ, allow_pickle=True)['arr'])
driving_keys = {(r, l) for r, road in m.items() for l in road if l != 'Trigger_Volumes'}

# --- 패치 적용 결과 ---
before, after = collections.Counter(), collections.Counter()
tagged = []                      # (road_id, lane_id, side, 폴리라인)
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type'] == 'Center'), None)
        bs = bp.boundary_sides(lane, driving_keys)
        for sl in lane:
            t = bp.resolve_lane_type(sl, ct, bs)
            before[sl['Type']] += 1; after[t] += 1
            if t == 'Boundary':
                tagged.append((rid, lid, bp.marking_side(sl, ct),
                               np.asarray([rp[0] for rp in sl['Points']], float)[:, :2]))

print("타입 분포  변경 전 :", dict(before.most_common()))
print("           변경 후 :", dict(after.most_common()))
print(f"\nBoundary 로 라벨된 폴리라인 {len(tagged)}개")

# --- 검증: 그 폴리라인의 바깥쪽 이웃이 정말 비주행인가 (CARLA 대조) ---
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
ok = bad = skip = 0
for rid, lid, side, P in tagged:
    mid = P[len(P)//2]
    w = cm.get_waypoint(carla.Location(x=float(mid[0]), y=float(mid[1]), z=0.0),
                        lane_type=carla.LaneType.Driving)
    if w is None: skip += 1; continue
    nb = w.get_left_lane() if side == 'left' else w.get_right_lane()
    if (nb is None) or (nb.lane_type != carla.LaneType.Driving): ok += 1
    else: bad += 1
print(f"검증(중점에서 CARLA 조회): 바깥이 비주행 {ok}   주행 {bad}   조회실패 {skip}")
print(f"   -> 정확도 {100*ok/max(ok+bad,1):.2f}%")
```

### `recall.py`

누락(false negative) 확인 + 총연장 커버리지.

```python
import numpy as np, collections, carla, boundary_patch as bp
TOWN='Town03'
m = dict(np.load(f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{TOWN}_HD_map.npz', allow_pickle=True)['arr'])
dk = {(r,l) for r,rd in m.items() for l in rd if l!='Trigger_Volumes'}

need, got = set(), set()          # (road, lane, side) 단위
blen = collections.Counter()
for rid, road in m.items():
    for lid, lane in road.items():
        if lid=='Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type']=='Center'), None)
        bs = bp.boundary_sides(lane, dk)
        for side in ('left','right'):
            if bs[side]: need.add((rid,lid,side))
        for sl in lane:
            s = bp.marking_side(sl, ct)
            if s is not None and bs[s]:
                got.add((rid,lid,s))
                P = np.asarray([rp[0] for rp in sl['Points']],float)[:,:2]
                blen['Boundary'] += np.linalg.norm(np.diff(P,axis=0),axis=1).sum()
miss = need - got
print(f"경계여야 할 (road,lane,side) : {len(need)}")
print(f"실제로 폴리라인이 라벨된 것   : {len(got)}    누락 {len(miss)}")
if miss:
    print("  누락 예시:", list(miss)[:8])
    for rid,lid,side in list(miss)[:3]:
        print(f"    road {rid} lane {lid} {side}: 이 lane 의 엔트리 타입 = {[e['Type'] for e in m[rid][lid]]}")

# 커버리지: 비교차로 도로 가장자리 실제 길이 대비
cm = carla.Map(TOWN, open(f'/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{TOWN}.xodr').read())
true_len = 0.0
for w in cm.generate_waypoints(1.0):
    if w.lane_type!=carla.LaneType.Driving or w.is_junction: continue
    for nb in (w.get_left_lane(), w.get_right_lane()):
        if (nb is None) or (nb.lane_type!=carla.LaneType.Driving): true_len += 1.0   # 1 m 간격
print(f"\n비교차로 실제 도로 가장자리 총연장 : {true_len:8.0f} m")
print(f"Boundary 라벨 폴리라인 총연장      : {blen['Boundary']:8.0f} m")
print(f"   커버리지 {100*blen['Boundary']/true_len:.1f}%")
```

### `dedup.py`

중복 폴리라인 정량화 (273 -> 184).

```python
import numpy as np, collections, boundary_patch as bp
m = dict(np.load('/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz', allow_pickle=True)['arr'])
dk = {(r,l) for r,rd in m.items() for l in rd if l!='Trigger_Volumes'}

bpolys = []
for rid, road in m.items():
    for lid, lane in road.items():
        if lid=='Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type']=='Center'), None)
        bs = bp.boundary_sides(lane, dk)
        for sl in lane:
            if bp.resolve_lane_type(sl, ct, bs) == 'Boundary':
                bpolys.append((rid, lid, np.asarray([rp[0] for rp in sl['Points']],float)[:,:2]))
L = lambda P: np.linalg.norm(np.diff(P,axis=0),axis=1).sum()

# (a) 같은 lane 안의 완전 중복
seen, uniq, dup_same = [], [], 0
for rid,lid,P in bpolys:
    if any(rid==r and lid==l and Q.shape==P.shape and np.allclose(Q,P) for r,l,Q in uniq):
        dup_same += 1
    else: uniq.append((rid,lid,P))
print(f"Boundary 폴리라인 {len(bpolys)}개 -> 같은 lane 내 완전중복 {dup_same}개 제거 후 {len(uniq)}개")
print(f"   총연장 {sum(L(P) for _,_,P in bpolys):8.0f} m  ->  {sum(L(P) for _,_,P in uniq):8.0f} m")

# (b) 서로 다른 lane 이 같은 물리적 선을 공유하는가 (인접 차선 공유 선)
dup_cross = 0; final = []
for rid,lid,P in uniq:
    if any(Q.shape==P.shape and np.allclose(Q,P) for _,_,Q in final): dup_cross += 1
    else: final.append((rid,lid,P))
print(f"   서로 다른 lane 간 중복 {dup_cross}개 -> 최종 {len(final)}개, 총연장 {sum(L(P) for _,_,P in final):.0f} m")
print(f"   (비교차로 실제 가장자리 총연장 7908 m 대비 {100*sum(L(P) for _,_,P in final)/7908:.1f}%)")
```

### `sidecheck.py`

(-sin yaw, cos yaw) == CARLA get_right_vector() 검증. 오차 4e-7.

```python
import numpy as np, carla
# yaw 로 만든 right vector 가 CARLA 의 get_right_vector() 와 같은가
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
err = []
for w in cm.generate_waypoints(5.0):
    yaw = np.deg2rad(w.transform.rotation.yaw)
    mine = np.array([-np.sin(yaw), np.cos(yaw)])
    rv = w.transform.get_right_vector()
    err.append(np.linalg.norm(mine - np.array([rv.x, rv.y])))
err = np.array(err)
print(f"right_vector 공식 (-sin yaw, cos yaw) vs CARLA get_right_vector()")
print(f"   waypoint {len(err)}개   오차 mean {err.mean():.2e}  max {err.max():.2e}")
print(f"   -> {'일치' if err.max() < 1e-5 else '불일치 (부호/축 규약 다름)'}")
```

## C. 10-route 파이프라인 (실행 순서 의존)

### `corridor.py`

1) XML 라우트 -> 4m 보간 -> 도로 스냅. 인자: <Town>. 출력 corridor_<Town>.npy

```python
import xml.etree.ElementTree as ET, numpy as np, collections, sys, carla, os

XML='/home/ailab/2026intern/jsn/2026-Summer-Internship/pipeline/data/bench2drive10_abilities.xml'
XODR=("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{t}.xodr",
      "/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/{t}/OpenDrive/{t}.xodr")
WANT={'24367','27582','3144','2416','2715','2286','17569','2790','3373','1792'}

root=ET.parse(XML).getroot()
routes={}
for rt in root.findall('.//route'):
    if rt.get('id') not in WANT: continue
    P=np.array([[float(p.get('x')),float(p.get('y')),float(p.get('z'))]
                for p in rt.findall('.//waypoints/position')])
    routes[rt.get('id')]=dict(town=rt.get('town'), xml=P,
                              scenario=[s.get('type') for s in rt.findall('.//scenarios/scenario')][0])

town=sys.argv[1]
path=next(q.format(t=town) for q in XODR if os.path.exists(q.format(t=town)))
print(f'{town}: xodr 로딩 {os.path.getsize(path)/1e6:.0f} MB ...', flush=True)
cm=carla.Map(town, open(path).read())
print(f'{town}: 로딩 완료', flush=True)

out={}
for rid,info in routes.items():
    if info['town']!=town: continue
    P=info['xml']
    # 4 m 간격 선형 보간
    seg=np.linalg.norm(np.diff(P[:,:2],axis=0),axis=1)
    dense=[]
    for i,d in enumerate(seg):
        n=max(int(d//4),1)
        for k in range(n): dense.append(P[i]+(P[i+1]-P[i])*k/n)
    dense.append(P[-1]); dense=np.array(dense)
    # 도로에 스냅
    snap=[]
    for q in dense:
        w=cm.get_waypoint(carla.Location(x=float(q[0]),y=float(q[1]),z=float(q[2])),
                          lane_type=carla.LaneType.Driving)
        if w is None: continue
        L=w.transform.location
        snap.append([L.x,L.y,w.transform.rotation.yaw, float(w.is_junction)])
    snap=np.array(snap)
    out[rid]=dict(town=town, scenario=info['scenario'], xml=P, snap=snap)
    d=np.linalg.norm(np.diff(snap[:,:2],axis=0),axis=1).sum()
    print(f'  route {rid:>6} {info["scenario"]:<42} 샘플 {len(snap):5d}  경로장 {d:8.1f} m', flush=True)
np.save(f'corridor_{town}.npy', out, allow_pickle=True)
```

### `carla_corridor.py`

2) CARLA 에서 코리도 주변 폴리라인 재현 + 라벨링. 인자: <Town>. 출력 carla_<Town>.npy

```python
"""gen_hdmap.py 의 폴리라인 생성 로직을 코리도 주변에서만 재현하고 Boundary 를 라벨링한다."""
import numpy as np, collections, sys, os, carla

XODR=("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{t}.xodr",
      "/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/{t}/OpenDrive/{t}.xodr")
R = 80.0
STEP = 1.0

town = sys.argv[1]
cor = np.load(f'corridor_{town}.npy', allow_pickle=True).item()
S = np.concatenate([v['snap'][:, :2] for v in cor.values()])
path = next(q.format(t=town) for q in XODR if os.path.exists(q.format(t=town)))
print(f'{town}: xodr 로딩 ...', flush=True)
cm = carla.Map(town, open(path).read())

def near(loc):
    return np.min(np.linalg.norm(S - np.array([loc.x, loc.y]), axis=1)) <= R

# ---- 코리도 주변 Driving waypoint 를 BFS 로 수집 ----
seen, queue, wps = set(), [], []
for x, y, yaw, isj in np.concatenate([v['snap'] for v in cor.values()]):
    w = cm.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0), lane_type=carla.LaneType.Driving)
    if w: queue.append(w)
while queue:
    w = queue.pop()
    k = (w.road_id, w.section_id, w.lane_id, round(w.s / STEP))
    if k in seen: continue
    seen.add(k)
    if not near(w.transform.location): continue
    wps.append(w)
    for nxt in (w.next(STEP) + w.previous(STEP)):
        if nxt.lane_type == carla.LaneType.Driving: queue.append(nxt)
    for nb in (w.get_left_lane(), w.get_right_lane()):
        if nb is not None and nb.lane_type == carla.LaneType.Driving: queue.append(nb)
print(f'{town}: 코리도 내 Driving waypoint {len(wps)}개', flush=True)

# ---- (road, section, lane) 별로 묶어 폴리라인 생성 ----
lanes = collections.defaultdict(list)
for w in wps: lanes[(w.road_id, w.section_id, w.lane_id)].append(w)

polys = []          # (type, Nx2, road_id, lane_id, is_junction, side)
for key, ws in lanes.items():
    ws.sort(key=lambda w: w.s)
    rid, sec, lid = key
    for side in ('left', 'right'):
        cur_t, buf = None, []
        for w in ws:
            mk = w.left_lane_marking if side == 'left' else w.right_lane_marking
            nb = w.get_left_lane()  if side == 'left' else w.get_right_lane()
            is_bd = (not w.is_junction) and ((nb is None) or (nb.lane_type != carla.LaneType.Driving))
            t = 'Boundary' if is_bd else str(mk.type)
            c, rv, hw = w.transform.location, w.transform.get_right_vector(), w.lane_width / 2
            sgn = -1.0 if side == 'left' else 1.0
            p = (c.x + rv.x * hw * sgn, c.y + rv.y * hw * sgn)
            if t != cur_t:
                if len(buf) > 1: polys.append((cur_t, np.array(buf, np.float32), rid, lid, ws[0].is_junction, side))
                cur_t, buf = t, []
            buf.append(p)
        if len(buf) > 1: polys.append((cur_t, np.array(buf, np.float32), rid, lid, ws[0].is_junction, side))
    C = np.array([[w.transform.location.x, w.transform.location.y] for w in ws], np.float32)
    if len(C) > 1: polys.append(('Center', C, rid, lid, ws[0].is_junction, ''))

bands=[]
for w in wps:
    c,rv,hw=w.transform.location,w.transform.get_right_vector(),w.lane_width/2
    bands.append([(c.x-rv.x*hw,c.y-rv.y*hw),(c.x+rv.x*hw,c.y+rv.y*hw)])
print(f'{town}: 폴리라인 {len(polys)}  타입 {dict(collections.Counter(t for t,*_ in polys).most_common())}', flush=True)
np.save(f'carla_{town}.npy', dict(polys=polys, corridor=cor, bands=np.array(bands,np.float32)), allow_pickle=True)
```

### `label_town.py`

2') npz 에서 코리도 주변 추출 (소형 타운용). 인자: <Town>.

```python
import numpy as np, collections, sys, os, resource, boundary_patch as bp
town = sys.argv[1]
R = 140.0                                   # 코리도 반경
NPZ = f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{town}_HD_map.npz'

cor = np.load(f'corridor_{town}.npy', allow_pickle=True).item()
S = np.concatenate([v['snap'][:, :2] for v in cor.values()])          # 이 타운 모든 라우트 샘플
lo, hi = S.min(0) - R, S.max(0) + R

print(f'{town}: npz 로딩 {os.path.getsize(NPZ)/1e6:.0f} MB ...', flush=True)
m = dict(np.load(NPZ, allow_pickle=True)['arr'])
print(f'{town}: road {len(m)}개, RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB', flush=True)

driving_keys = {(r, l) for r, rd in m.items() for l in rd if l != 'Trigger_Volumes'}
kept, gstat = [], collections.Counter()
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type'] == 'Center'), None)
        bs = bp.boundary_sides(lane, driving_keys)
        isj = ct is not None and 'Junction' in ct.get('TopologyType', '')
        for sl in lane:
            t = bp.resolve_lane_type(sl, ct, bs)
            gstat[t] += 1
            P = np.asarray([rp[0] for rp in sl['Points']], np.float32)[:, :2]
            if P[:, 0].max() < lo[0] or P[:, 0].min() > hi[0] or \
               P[:, 1].max() < lo[1] or P[:, 1].min() > hi[1]: continue
            if np.min(np.linalg.norm(P[::20][:, None, :] - S[None, ::3, :], axis=-1)) > R: continue
            side = bp.marking_side(sl, ct)
            kept.append((t, P, int(rid), int(lid), bool(isj), side or ''))
del m
print(f'{town}: 전체 폴리라인 {sum(gstat.values())} -> 코리도 내 {len(kept)}', flush=True)
print(f'{town}: 전체 타입 분포 {dict(gstat.most_common())}', flush=True)
np.save(f'labelled_{town}.npy',
        dict(kept=kept, global_stat=dict(gstat), corridor=cor), allow_pickle=True)
print(f'{town}: 저장 완료  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB', flush=True)
```

### `metrics.py`

3) 프레임당 경계 가용성 표 -> metrics.json

```python
import numpy as np, collections, json

TOWNS = ['Town06','Town11','Town12','Town13']
LAT, LON = 15.0, 30.0                      # pc_range: x +-15, y +-30
rows = []
for town in TOWNS:
    d = np.load(f'carla_{town}.npy', allow_pickle=True).item()
    polys, cor = d['polys'], d['corridor']
    for rid, info in cor.items():
        snap = info['snap']
        cnt = collections.defaultdict(list)          # type -> 프레임별 인스턴스 수
        dmin = {'Boundary': [], 'SolidSolid': []}
        for x, y, yaw, isj in snap:
            e = np.array([x, y]); a = np.deg2rad(yaw)
            f = np.array([np.cos(a), np.sin(a)]); r = np.array([-np.sin(a), np.cos(a)])
            per = collections.Counter()
            best = {'Boundary': np.inf, 'SolidSolid': np.inf}
            for t, P, *_ in polys:
                V = P - e
                lat, lon = V @ r, V @ f
                inb = (np.abs(lat) <= LAT) & (np.abs(lon) <= LON)
                if inb.sum() > 1:
                    per[t] += 1
                    if t in best:
                        best[t] = min(best[t], float(np.linalg.norm(V[inb], axis=1).min()))
            for k, v in per.items(): cnt[k].append(v)
            for k in dmin:
                cnt.setdefault(k, [])
                if len(cnt[k]) < len(cnt.get('Center', [])) : pass
                dmin[k].append(best[k])
        n = len(snap)
        def stat(t):
            v = np.array(cnt.get(t, []) + [0]*(n - len(cnt.get(t, []))))
            return 100*float((v > 0).mean()), float(v.mean())
        ss_pct, ss_mean = stat('SolidSolid')
        bd_pct, bd_mean = stat('Boundary')
        db = np.array(dmin['Boundary']); ds = np.array(dmin['SolidSolid'])
        rows.append(dict(route=rid, town=town, scenario=info['scenario'], frames=n,
                         ss_pct=ss_pct, ss_mean=ss_mean, bd_pct=bd_pct, bd_mean=bd_mean,
                         bd_dist=float(np.median(db[np.isfinite(db)])) if np.isfinite(db).any() else float('nan'),
                         ss_dist=float(np.median(ds[np.isfinite(ds)])) if np.isfinite(ds).any() else float('nan')))

order = ['24367','27582','3144','2416','2715','2286','17569','2790','3373','1792']
rows.sort(key=lambda r: order.index(r['route']))
json.dump(rows, open('metrics.json','w'), indent=1)

print(f"{'route':>6} {'town':>7} {'scenario':<42} {'frm':>4} | "
      f"{'SS %frm':>8} {'SS inst':>8} {'SS dist':>8} | {'BD %frm':>8} {'BD inst':>8} {'BD dist':>8}")
print('-'*128)
for r in rows:
    fd = lambda v: f'{v:8.1f}' if np.isfinite(v) else '       -'
    print(f"{r['route']:>6} {r['town']:>7} {r['scenario']:<42} {r['frames']:>4} | "
          f"{r['ss_pct']:8.1f} {r['ss_mean']:8.2f} {fd(r['ss_dist'])} | "
          f"{r['bd_pct']:8.1f} {r['bd_mean']:8.2f} {fd(r['bd_dist'])}")
print('-'*128)
w = np.array([r['frames'] for r in rows], float); w /= w.sum()
print(f"{'가중평균':>6} {'':>7} {'':<42} {int(sum(r['frames'] for r in rows)):>4} | "
      f"{sum(w[i]*rows[i]['ss_pct'] for i in range(len(rows))):8.1f} "
      f"{sum(w[i]*rows[i]['ss_mean'] for i in range(len(rows))):8.2f} {'':>8} | "
      f"{sum(w[i]*rows[i]['bd_pct'] for i in range(len(rows))):8.1f} "
      f"{sum(w[i]*rows[i]['bd_mean'] for i in range(len(rows))):8.2f}")
```

### `viz10.py`

4) 10-route 그림 -> b2d10_boundary.png

```python
import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

ORDER=['24367','27582','3144','2416','2715','2286','17569','2790','3373','1792']
M={r['route']:r for r in json.load(open('metrics.json'))}
data={t:np.load(f'carla_{t}.npy',allow_pickle=True).item() for t in ['Town06','Town11','Town12','Town13']}
loc={rid:t for t in data for rid in data[t]['corridor']}

fig,axes=plt.subplots(2,5,figsize=(30,13))
for ax,rid in zip(axes.ravel(),ORDER):
    t=loc[rid]; d=data[t]; info=d['corridor'][rid]; snap=info['snap']; m=M[rid]
    C=snap[:,:2]; cx,cy=C.mean(0); half=max(C.max(0)-C.min(0)).max()/2+55

    ax.add_collection(LineCollection(d['bands'],colors='#dcdcdc',linewidths=2.0,zorder=0))
    for ty,P,rr,ll,isj,side in d['polys']:
        if ty=='Boundary': continue
        if ty=='Center': ax.plot(P[:,0],P[:,1],color='#c4c4c4',lw=0.5,ls=(0,(4,4)),zorder=1)
        elif ty=='SolidSolid': ax.plot(P[:,0],P[:,1],color='#f5a623',lw=2.0,zorder=2)
        else: ax.plot(P[:,0],P[:,1],color='#4a90d9',lw=0.9,zorder=2)
    for ty,P,*_ in d['polys']:
        if ty=='Boundary': ax.plot(P[:,0],P[:,1],color='#e02020',lw=2.6,zorder=4)
    ax.plot(C[:,0],C[:,1],color='#111111',lw=2.0,zorder=5)
    ax.plot(C[0,0],C[0,1],'o',color='#111',ms=7,zorder=6)

    # ego box 하나 예시 (경로 중간 지점)
    i=len(snap)//2; x,y,yaw,_=snap[i]; a=np.deg2rad(yaw)
    f=np.array([np.cos(a),np.sin(a)]); r=np.array([-np.sin(a),np.cos(a)])
    box=np.array([[-15,-30],[15,-30],[15,30],[-15,30],[-15,-30]],float)
    W=np.array([x,y])+box[:,0:1]*r+box[:,1:2]*f
    ax.plot(W[:,0],W[:,1],color='#2e7d32',lw=1.6,ls='--',zorder=6)

    ax.set_xlim(cx-half,cx+half); ax.set_ylim(cy+half,cy-half); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"route {rid}  ({t})\n{m['scenario']}",fontsize=10.5)
    ax.text(0.02,0.975,f"SolidSolid (current): {m['ss_pct']:.0f}% of frames, {m['ss_mean']:.2f} inst\n"
                       f"Boundary  (fixed)  : {m['bd_pct']:.0f}% of frames, {m['bd_mean']:.2f} inst",
            transform=ax.transAxes,va='top',ha='left',fontsize=8.6,family='monospace',
            bbox=dict(fc='white',ec='#999',alpha=.92,pad=3.5))

h=[plt.Line2D([],[],color=c,lw=w,ls=s) for c,w,s in
   [('#e02020',2.6,'-'),('#f5a623',2.0,'-'),('#4a90d9',1.2,'-'),('#c4c4c4',.9,(0,(4,4))),
    ('#111111',2.0,'-'),('#2e7d32',1.6,'--')]]
fig.legend(h,['Boundary (auto-labelled)','SolidSolid (what the loss uses today)','other lane markings',
              'lane centre','planned route','ego BEV box 30x60 m'],
           loc='lower center',ncol=6,frameon=False,fontsize=11.5)
fig.suptitle('Bench2Drive 10-route set — road boundary labelling (RED) vs the current SolidSolid proxy (ORANGE)',
             fontsize=15,y=0.985)
plt.tight_layout(rect=[0,0.035,1,0.965])
plt.savefig('b2d10_boundary.png',dpi=100,bbox_inches='tight')
print('saved')
```

### `viz_town.py`

타운 전체 그림 -> town03_boundary.png (독립 실행 가능)

```python
import numpy as np, collections, carla, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import boundary_patch as bp

TOWN = 'Town03'
NPZ  = f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{TOWN}_HD_map.npz'
XODR = f'/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{TOWN}.xodr'

# ---------- 1. 라벨링 ----------
m = dict(np.load(NPZ, allow_pickle=True)['arr'])
driving_keys = {(r, l) for r, rd in m.items() for l in rd if l != 'Trigger_Volumes'}
polys = []                                     # (type, Nx2, is_junction)
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        ct = next((e for e in lane if e['Type'] == 'Center'), None)
        bs = bp.boundary_sides(lane, driving_keys)
        isj = ct is not None and 'Junction' in ct.get('TopologyType', '')
        for sl in lane:
            t = bp.resolve_lane_type(sl, ct, bs)
            P = np.asarray([rp[0] for rp in sl['Points']], float)[:, :2]
            polys.append((t, P, isj))
print('폴리라인', len(polys), '| 타입', dict(collections.Counter(t for t,_,_ in polys).most_common()))

# ---------- 2. 주행가능영역 (배경) ----------
cm = carla.Map(TOWN, open(XODR).read())
segs = []
for w in cm.generate_waypoints(1.0):
    if w.lane_type != carla.LaneType.Driving: continue
    c = w.transform.location; rv = w.transform.get_right_vector(); hw = w.lane_width/2
    segs.append([(c.x-rv.x*hw, c.y-rv.y*hw), (c.x+rv.x*hw, c.y+rv.y*hw)])
print('주행영역 segment', len(segs))

# ---------- 3. 그리기 ----------
OTHER = {'Broken':'#4a90d9', 'Solid':'#4a90d9', 'SolidSolid':'#4a90d9',
         'NONE':'#4a90d9', 'SolidBroken':'#4a90d9', 'BrokenSolid':'#4a90d9'}

def draw(ax, step=5, lw_b=2.2, lw_o=0.7, lw_c=0.4):
    ax.add_collection(LineCollection(segs, colors='#d8d8d8', linewidths=2.4, zorder=0))
    for t, P, isj in polys:
        if t == 'Boundary': continue
        Q = P[::step] if len(P) > step*3 else P
        if t == 'Center':
            ax.plot(Q[:,0], Q[:,1], color='#bbbbbb', lw=lw_c, ls=(0,(4,4)), zorder=1)
        else:
            ax.plot(Q[:,0], Q[:,1], color=OTHER.get(t,'#4a90d9'), lw=lw_o, zorder=2)
    for t, P, isj in polys:                       # 경계는 맨 위에
        if t != 'Boundary': continue
        Q = P[::step] if len(P) > step*3 else P
        ax.plot(Q[:,0], Q[:,1], color='#e02020', lw=lw_b, zorder=3)
    ax.set_aspect('equal'); ax.invert_yaxis()

# 줌 위치: 연결로가 가장 많은 교차로 + 차선 많은 일반 구간
byj = collections.defaultdict(list)
for w in cm.generate_waypoints(2.0):
    if w.is_junction: byj[w.junction_id].append((w.transform.location.x, w.transform.location.y))
jbig = max(byj, key=lambda k: len(set(map(tuple, byj[k]))))
jc = np.array(byj[jbig]).mean(axis=0)

wide = collections.defaultdict(list)
for w in cm.generate_waypoints(2.0):
    if not w.is_junction: wide[w.road_id].append((w.transform.location.x, w.transform.location.y, w.lane_id))
rbig = max(wide, key=lambda r: len({x[2] for x in wide[r]}))
rc = np.array([[x,y] for x,y,_ in wide[rbig]]).mean(axis=0)

allP = np.concatenate([P for _,P,_ in polys])
fig = plt.figure(figsize=(22, 13))
gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])

ax0 = fig.add_subplot(gs[:, 0]); draw(ax0, step=8)
ax0.set_title(f'{TOWN} full map   |   RED = Boundary (auto-labelled)   BLUE = other lane markings   GREY = drivable area', fontsize=11)
ax0.set_xlim(allP[:,0].min()-20, allP[:,0].max()+20); ax0.set_ylim(allP[:,1].max()+20, allP[:,1].min()-20)

for k, (cx, cy, r, ttl) in enumerate([
        (jc[0], jc[1], 60, f'Intersection (junction {jbig}) - no RED should cut across the junction'),
        (rc[0], rc[1], 70, f'Multi-lane road (road {rbig}) - only the two outermost lines should be RED'),
        (allP[:,0].mean(), allP[:,1].mean(), 90, 'Map centre'),
        (jc[0]+120, jc[1]+120, 80, 'Around the intersection')]):
    ax = fig.add_subplot(gs[k//2, 1+k%2]); draw(ax, step=1, lw_b=3.0, lw_o=1.2, lw_c=0.7)
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy+r, cy-r); ax.set_title(ttl, fontsize=10)

plt.tight_layout(); plt.savefig('/tmp/claude-1000/-home-ailab-carla-control/40dfe823-9875-434c-accf-6ce088910cc3/scratchpad/town03_boundary.png', dpi=110, bbox_inches='tight')
print('saved')
```

### `collect_traj.py`

런 덤프에서 실제 ego 궤적 수집 (참고용, 주행거리가 짧아 지표엔 미사용).

```python
import json, glob, os, numpy as np
ROUTES = {'24367':'Town06','27582':'Town11','3144':'Town12','2416':'Town12','2715':'Town12',
          '2286':'Town12','17569':'Town12','2790':'Town12','3373':'Town13','1792':'Town12'}
RUNS = '/home/ailab/2026intern/kmlee/vad_demo_video/runs'
out = {}
for r, town in ROUTES.items():
    cands = sorted(glob.glob(f'{RUNS}/route{r}_*/frames/*/metric_info.json'))
    if not cands:
        print(f'route {r:>6} ({town}): metric_info 없음'); continue
    p = max(cands, key=os.path.getsize)
    mi = json.load(open(p))
    ticks = sorted(mi, key=int)
    locs = np.array([mi[t]['location'] for t in ticks], float)
    out[r] = dict(town=town, locs=locs, src=p.split('/frames/')[0].split('/')[-1])
    d = np.linalg.norm(np.diff(locs[:,:2],axis=0),axis=1).sum()
    print(f'route {r:>6} ({town}): tick {len(ticks):5d}  주행거리 {d:7.1f} m  <- {out[r]["src"]}')
np.save('traj.npy', out, allow_pickle=True)
print(f'\n수집 완료 {len(out)}/10')
```

## D. 조사 과정 재현용 (근거가 된 측정들)

### `edge.py`

도로 가장자리 vs 차선 사이의 마킹 타입 분포. SolidSolid 가 경계가 아님을 보인다.

```python
import sys, collections

import carla
for town in ["Town03", "Town05", "Town12"]:
    p = f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
    try:
        cm = carla.Map(town, open(p).read())
    except Exception as e:
        print(town, "skip", e); continue
    wps = cm.generate_waypoints(2.0)
    mk = collections.Counter()          # marking type on true drivable edges
    mk_inner = collections.Counter()    # marking type between two driving lanes
    nbr = collections.Counter()         # lane_type of the neighbour on the edge side
    n_edge = n_inner = 0
    for w in wps:
        if w.lane_type != carla.LaneType.Driving: continue
        for side, nb, m in (("L", w.get_left_lane(), w.left_lane_marking),
                            ("R", w.get_right_lane(), w.right_lane_marking)):
            is_edge = (nb is None) or (nb.lane_type != carla.LaneType.Driving)
            nbr[str(nb.lane_type) if nb is not None else "None"] += 1
            if is_edge:
                n_edge += 1; mk[str(m.type) if m else "None"] += 1
            else:
                n_inner += 1; mk_inner[str(m.type) if m else "None"] += 1
    print(f"\n===== {town}  driving wps={sum(1 for w in wps if w.lane_type==carla.LaneType.Driving)} =====")
    print(f"  drivable-edge sides : {n_edge}   inner sides: {n_inner}")
    print("  marking type ON EDGE     :", dict(mk.most_common()))
    print("  marking type BETWEEN lanes:", dict(mk_inner.most_common()))
    print("  neighbour lane_type      :", dict(nbr.most_common(10)))
```

### `junc.py`

교차로를 제외하지 않으면 side 의 73~80% 가 경계로 오판됨.

```python
import collections, carla
for town in ["Town03", "Town05"]:
    cm = carla.Map(town, open(f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr").read())
    wps = [w for w in cm.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving]
    st = collections.Counter()
    for w in wps:
        j = "junction" if w.is_junction else "road"
        for nb in (w.get_left_lane(), w.get_right_lane()):
            edge = (nb is None) or (nb.lane_type != carla.LaneType.Driving)
            st[(j, "edge" if edge else "inner")] += 1
    tot_j = sum(v for k,v in st.items() if k[0]=="junction")
    tot_r = sum(v for k,v in st.items() if k[0]=="road")
    print(f"{town}: junction sides {tot_j} -> edge {st[('junction','edge')]} ({100*st[('junction','edge')]/max(tot_j,1):.0f}%)"
          f" | road sides {tot_r} -> edge {st[('road','edge')]} ({100*st[('road','edge')]/max(tot_r,1):.0f}%)")
```

### `junc2.py`

is_junction 정의, junction 당 연결로 수.

```python
import carla, collections
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(1.0)
# is_junction 의 정의 확인
agree = sum(1 for w in wps if w.is_junction == (w.junction_id != -1))
print(f"is_junction == (junction_id != -1) 인 waypoint: {agree}/{len(wps)}")
# 한 road 안에서 is_junction 이 섞이는가
byroad = collections.defaultdict(set)
for w in wps: byroad[w.road_id].add(w.is_junction)
mixed = [r for r,v in byroad.items() if len(v)>1]
print(f"한 road 안에서 is_junction 이 섞이는 road: {len(mixed)}/{len(byroad)}  -> road 단위 속성")
# junction 하나가 몇 개의 연결로(road)로 이뤄지는가
j = collections.defaultdict(set)
for w in wps:
    if w.junction_id != -1: j[w.junction_id].add(w.road_id)
print(f"junction 당 연결로 road 수: min {min(map(len,j.values()))}  max {max(map(len,j.values()))}  평균 {sum(map(len,j.values()))/len(j):.1f}")
# Topology = next(0.05) 의 Driving 후속. 분기 개수 분포
nb = collections.Counter()
for w in wps:
    nb[len([x for x in w.next(0.05) if x.lane_type==carla.LaneType.Driving])] += 1
print("next(0.05) Driving 후속 개수 분포:", dict(sorted(nb.items())))
```

### `overlap.py`

교차로 연결로가 물리적으로 겹침 + 이웃 조회 57% None.

```python
import carla, collections, numpy as np, itertools
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(0.5)

# 369 -> 248 재조정 확인
roads = {w.road_id for w in wps}
jroads = {w.road_id for w in wps if w.is_junction}
print(f"driving waypoint 가 생기는 road: {len(roads)}  (그중 교차로 연결로 {len(jroads)})")

# 한 junction 안의 연결로들이 물리적으로 겹치는가
byj = collections.defaultdict(lambda: collections.defaultdict(list))
for w in wps:
    if w.is_junction: byj[w.junction_id][w.road_id].append((w.transform.location.x, w.transform.location.y))
jid = max(byj, key=lambda k: len(byj[k]))
rs = {r: np.array(p) for r, p in byj[jid].items()}
print(f"\njunction {jid}: 연결로 {len(rs)}개")
cross = 0
for a, b in itertools.combinations(rs, 2):
    d = np.linalg.norm(rs[a][:,None,:] - rs[b][None,:,:], axis=-1).min()
    if d < 1.0: cross += 1
print(f"  연결로 쌍 {len(list(itertools.combinations(rs,2)))}개 중 중심선 거리 1 m 이내로 스치는 쌍: {cross}")

# 연결로 waypoint 의 이웃 유무
n_none = n_any = 0
for w in wps:
    if not w.is_junction: continue
    for nb in (w.get_left_lane(), w.get_right_lane()):
        n_any += 1
        if nb is None: n_none += 1
print(f"\n교차로 내부 waypoint 의 좌/우 이웃 조회 {n_any}회 중 None 반환 {n_none}회 ({100*n_none/n_any:.0f}%)")
```

### `xodr.py`

일반구간 82 vs 교차로 연결로 287 비교.

```python
import xml.etree.ElementTree as ET, collections, statistics as st
root = ET.parse("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").getroot()

norm, conn = [], []
for r in root.findall('road'):
    (conn if r.get('junction') != '-1' else norm).append(r)
print(f"Town03 .xodr:  <road> 총 {len(norm)+len(conn)}개   일반 {len(norm)}   교차로 연결로 {len(conn)}")
print(f"               <junction> 엘리먼트 {len(root.findall('junction'))}개")

def stats(rs, name):
    L   = [float(r.get('length')) for r in rs]
    nls = [len(r.findall('lanes/laneSection')) for r in rs]
    dl, both, mk = [], 0, collections.Counter()
    for r in rs:
        s0 = r.find('lanes/laneSection')
        ids = [int(l.get('id')) for side in ('left','right') for l in s0.findall(f'{side}/lane')]
        d   = [int(l.get('id')) for side in ('left','right') for l in s0.findall(f'{side}/lane') if l.get('type')=='driving']
        dl.append(len(d))
        if any(i<0 for i in d) and any(i>0 for i in d): both += 1
        for l in s0.findall('.//lane'):
            for rm in l.findall('roadMark'): mk[rm.get('type')] += 1
    print(f"\n--- {name} ({len(rs)}개) ---")
    print(f"  길이        중앙값 {st.median(L):7.1f} m   min {min(L):6.2f}   max {max(L):7.1f}")
    print(f"  laneSection 중앙값 {st.median(nls):.0f}개")
    print(f"  driving 차선 수 분포 {dict(sorted(collections.Counter(dl).items()))}")
    print(f"  양방향(±둘 다 driving) road: {both}/{len(rs)}")
    print(f"  roadMark 타입: {dict(mk.most_common(6))}")

stats(norm, "일반 구간  junction=\"-1\"")
stats(conn, "교차로 연결로  junction=\"<id>\"")

print("\n--- 연결로 predecessor/successor 는 무엇을 가리키나 ---")
c = collections.Counter()
for r in conn:
    for tag in ('predecessor','successor'):
        e = r.find(f'link/{tag}')
        c[(tag, e.get('elementType') if e is not None else None)] += 1
print("  연결로:", dict(c))
c = collections.Counter()
for r in norm:
    for tag in ('predecessor','successor'):
        e = r.find(f'link/{tag}')
        c[(tag, e.get('elementType') if e is not None else None)] += 1
print("  일반  :", dict(c))

j = root.findall('junction')[0]
print(f"\n--- junction id={j.get('id')} 의 connection 목록 (앞 8개) ---")
for cn in j.findall('connection')[:8]:
    ll = cn.find('laneLink')
    print(f"    들어오는 road {cn.get('incomingRoad'):>4}  ->  연결로 road {cn.get('connectingRoad'):>4}"
          f"   laneLink from={ll.get('from')} to={ll.get('to')}" if ll is not None else "")
print(f"    (이 junction 의 connection 총 {len(j.findall('connection'))}개)")
```

### `roadlane.py`

road_id / lane_id 의미, 가장 안쪽 Driving lane 의 |lane_id| 분포.

```python
import carla, collections
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(1.0)

roads = collections.defaultdict(set)
for w in wps:
    roads[w.road_id].add((w.section_id, w.lane_id, str(w.lane_type)))
print(f"Town03: road_id 개수 = {len(roads)}   전체 waypoint = {len(wps)}")
print(f"road_id 값 범위 = {min(roads)} ~ {max(roads)}  (연속인가? {sorted(roads)==list(range(min(roads),max(roads)+1))})")

junc = [w for w in wps if w.is_junction]
print(f"junction 내부 waypoint = {len(junc)}, 그 road_id 개수 = {len(set(w.road_id for w in junc))}")
print(f"junction_id 개수 = {len(set(w.junction_id for w in junc))}")

print("\n=== 한 road 의 lane_id 구성 (차선 안쪽->바깥쪽) ===")
for rid in sorted(roads):
    lanes = sorted(roads[rid], key=lambda t: (t[0], t[1]))
    ids = [l[1] for l in lanes if l[0] == lanes[0][0]]
    if len(ids) >= 5 and min(ids) < 0 and max(ids) > 0:
        for s, lid, lt in lanes:
            if s != lanes[0][0]: continue
            print(f"   road {rid} section {s}  lane_id {lid:+3d}   lane_type={lt}")
        break

print("\n=== '차로 번호' 가 아니라는 증거: 가장 안쪽 Driving lane 의 |lane_id| 분포 ===")
inner = collections.Counter()
for rid, ls in roads.items():
    for sign in (-1, 1):
        d = [abs(l[1]) for l in ls if l[2] == 'Driving' and (l[1] < 0) == (sign < 0)]
        if d: inner[min(d)] += 1
print("   ", dict(sorted(inner.items())), "  <- 1 이 아니면 안쪽에 비주행 차선(갓길/연석 등)이 있다는 뜻")

print("\n=== lane section 별로 lane_type 이 바뀌는 (road, lane_id) 가 있는가 ===")
byrl = collections.defaultdict(set)
for rid, ls in roads.items():
    for s, lid, lt in ls: byrl[(rid, lid)].add(lt)
mixed = {k: v for k, v in byrl.items() if len(v) > 1}
print(f"    {len(mixed)} / {len(byrl)} 개 (road,lane_id) 가 section 마다 lane_type 이 다름")
for k, v in list(mixed.items())[:5]: print("      ", k, sorted(v))
```

### `roadlane2.py`

lane_id 부호=주행방향, get_left/right_lane() 이 주행방향 기준임을 확인.

```python
import carla, collections
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(1.0)
print("generate_waypoints 가 돌려주는 lane_type:", dict(collections.Counter(str(w.lane_type) for w in wps)))

roads = collections.defaultdict(set)
for w in wps: roads[w.road_id].add((w.section_id, w.lane_id, str(w.lane_type)))

# 양방향 + 비주행 차선이 섞인 road 예시
for rid in sorted(roads):
    ls = [l for l in roads[rid] if l[0]==0]
    ids = [l[1] for l in ls]
    if len(ls) >= 4 and min(ids) < 0 and max(ids) > 0:
        print(f"\n=== road {rid} section 0 (기준선에서 바깥쪽으로) ===")
        for s, lid, lt in sorted(ls, key=lambda t: -t[1]):
            print(f"    lane_id {lid:+3d}   {lt}")
        break

# lane_id 부호 = 주행방향
print("\n=== lane_id 부호와 주행방향 ===")
for rid in sorted(roads):
    ls = [l for l in roads[rid] if l[0]==0 and l[2]=='Driving']
    if any(l[1]<0 for l in ls) and any(l[1]>0 for l in ls):
        for w in wps:
            if w.road_id==rid and w.section_id==0 and w.lane_type==carla.LaneType.Driving:
                print(f"    road {rid} lane {w.lane_id:+3d}  yaw={w.transform.rotation.yaw:8.2f}  s={w.s:6.1f}")
        break

# 이웃 조회 방향 확인
print("\n=== get_left/right_lane() 은 '주행방향 기준' 인가 ===")
for rid in sorted(roads):
    ls = [l for l in roads[rid] if l[0]==0 and l[2]=='Driving']
    if any(l[1]<0 for l in ls) and any(l[1]>0 for l in ls):
        seen=set()
        for w in wps:
            if w.road_id!=rid or w.section_id!=0 or w.lane_type!=carla.LaneType.Driving: continue
            if w.lane_id in seen: continue
            seen.add(w.lane_id)
            L,R = w.get_left_lane(), w.get_right_lane()
            f=lambda n: f"({n.road_id},{n.lane_id},{str(n.lane_type)})" if n else "None"
            print(f"    lane {w.lane_id:+3d}  Left={f(L):28s} Right={f(R)}")
        break
```

### `topo.py`

Topology 가 next(0.05) 이고 98.4% 가 자기 자신임을 확인.

```python
import numpy as np, collections, carla

# 1) waypoint.next(d) 가 실제로 뭘 돌려주는가
cm = carla.Map("Town03", open("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").read())
wps = cm.generate_waypoints(1.0)
same = diff = 0
nb = collections.Counter()
for w in wps:
    nxt = [x for x in w.next(0.05) if x.lane_type == carla.LaneType.Driving]
    nb[len(nxt)] += 1
    for x in nxt:
        if (x.road_id, x.lane_id) == (w.road_id, w.lane_id): same += 1
        else: diff += 1
print("next(0.05) 의 Driving 후속 개수 분포:", dict(sorted(nb.items())))
print(f"후속이 '자기 자신과 같은 (road_id, lane_id)' 인 경우: {same}  /  다른 차선: {diff}")
print(f"   -> {100*same/(same+diff):.1f}% 가 자기 자신")

# 2) 그럼 실제 npz 의 Topology 필드에는 뭐가 들어있나
NPZ='/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz'
m = dict(np.load(NPZ, allow_pickle=True)['arr'])
c = collections.Counter(); n = collections.Counter()
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes': continue
        for e in lane:
            t = e['Topology']
            n[len(t)] += 1
            for item in t:
                c['자기 자신' if tuple(item) == (rid, lid) else '다른 차선'] += 1
print(f"\nnpz Topology 필드: 항목 개수 분포 {dict(sorted(n.items()))}")
print(f"   내용물: {dict(c)}")
```

### `demo4.py`

Left/Right(옆) vs Topology(앞) 대조 예시.

```python
import numpy as np, carla
m = dict(np.load('/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town03_HD_map.npz',
                 allow_pickle=True)['arr'])
cm = carla.Map('Town03', open('/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr').read())
wps = cm.generate_waypoints(2.0)

for rid, road in m.items():
    lids = [l for l in road if l != 'Trigger_Volumes']
    neg = sorted([l for l in lids if l < 0], reverse=True)
    if len(neg) < 2: continue
    cts = {l: next((e for e in road[l] if e['Type']=='Center'), None) for l in neg}
    if any(c is None or c['TopologyType'] != 'Normal' for c in cts.values()): continue
    print(f"=== road {rid} (일반 구간) : 같은 방향 차선 {neg} ===\n")
    for lid in neg:
        ct = cts[lid]
        w  = next((w for w in wps if w.road_id==rid and w.lane_id==lid), None)
        L, R = (w.get_left_lane(), w.get_right_lane()) if w else (None, None)
        f = lambda n: f"({n.road_id},{n.lane_id},{str(n.lane_type)})" if n else "None"
        print(f"  lane {lid:+d}")
        print(f"     Left     = {str(ct['Left']):16s}  <- 옆(기준선 쪽)   CARLA 실제: {f(L)}")
        print(f"     Right    = {str(ct['Right']):16s}  <- 옆(바깥쪽)     CARLA 실제: {f(R)}")
        print(f"     Topology = {str([tuple(x) for x in ct['Topology']]):16s}  <- 앞(이어지는 차선)")
        print()
    break
```

### `nbrrule.py`

Left/Right + npz 키 집합 규칙의 정확도 (1.000).

```python
import carla, collections
# npz-only 추론 규칙 재현: Center 의 Left/Right 가 가리키는 (road_id, lane_id) 가
# npz 키(=Driving 차선)에 없으면 그 side 를 boundary 로 본다.
for town in ["Town03","Town05"]:
    cm = carla.Map(town, open(f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr").read())
    wps = [w for w in cm.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving]
    driving_keys = {(w.road_id, w.lane_id) for w in wps}     # npz 가 저장하는 키 집합
    st = collections.Counter()
    for w in wps:
        if w.is_junction: continue
        for nb in (w.get_left_lane(), w.get_right_lane()):
            truth = (nb is None) or (nb.lane_type != carla.LaneType.Driving)
            pred  = (nb is None) or ((nb.road_id, nb.lane_id) not in driving_keys)
            st[(pred, truth)] += 1
    tp,fp,fn,tn = st[(True,True)],st[(True,False)],st[(False,True)],st[(False,False)]
    print(f"{town}: TP={tp} FP={fp} FN={fn} TN={tn}"
          f"  -> precision {tp/max(tp+fp,1):.3f}  recall {tp/max(tp+fn,1):.3f}")
```

### `proxy.py`

lane_id 크기만으로 추론하면 precision 0.50 임을 보인다 (왜 Left/Right 가 필요한가).

```python
import carla, collections
for town in ["Town03","Town05"]:
    cm = carla.Map(town, open(f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr").read())
    wps = [w for w in cm.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving]
    # outermost driving lane per (road_id, section?, sign) -- npz has no section, mimic that
    outer = {}
    for w in wps:
        k = (w.road_id, 1 if w.lane_id > 0 else -1)
        outer[k] = max(outer.get(k, 0), abs(w.lane_id))
    st = collections.Counter()
    for w in wps:
        if w.is_junction: continue
        k = (w.road_id, 1 if w.lane_id > 0 else -1)
        pred_outer = abs(w.lane_id) == outer[k]        # npz-inferable guess: this lane's outer edge is boundary
        nb = w.get_right_lane() if w.lane_id < 0 else w.get_left_lane()   # the outward side
        truth = (nb is None) or (nb.lane_type != carla.LaneType.Driving)
        st[(pred_outer, truth)] += 1
    tp,fp,fn,tn = st[(True,True)],st[(True,False)],st[(False,True)],st[(False,False)]
    print(f"{town}: outer-edge-of-outermost-lane 규칙  TP={tp} FP={fp} FN={fn} TN={tn}"
          f"  -> precision {tp/max(tp+fp,1):.3f}  recall {tp/max(tp+fn,1):.3f}")
```

### `stab.py`

Left/Right 가 세그먼트당 1회 샘플링돼도 안전함 (불안정 0건).

```python
import carla, collections
# gen_hdmap 은 Left/Right 를 세그먼트의 '마지막 waypoint' 에서 한 번만 기록한다.
# 같은 (road_id, lane_id) 구간 안에서 이웃의 boundary 여부가 바뀌면 그 샘플은 구간 일부에 대해 틀린다.
for town in ["Town03","Town05"]:
    cm = carla.Map(town, open(f"/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr").read())
    wps = [w for w in cm.generate_waypoints(2.0) if w.lane_type == carla.LaneType.Driving and not w.is_junction]
    runs = collections.defaultdict(lambda: {'L': set(), 'R': set()})
    for w in wps:
        k = (w.road_id, w.lane_id)
        for side, nb in (('L', w.get_left_lane()), ('R', w.get_right_lane())):
            runs[k][side].add((nb is None) or (nb.lane_type != carla.LaneType.Driving))
    unstable = sum(1 for v in runs.values() for s in 'LR' if len(v[s]) > 1)
    total    = sum(1 for v in runs.values() for s in 'LR' if len(v[s]) > 0)
    print(f"{town}: (road,lane) side 수 {total} 중 구간 내부에서 boundary 여부가 뒤바뀌는 side = {unstable} ({100*unstable/total:.1f}%)")
```

### `stage2.py`

gengrate_map() 재현 — road_id/lane_id 가 버려지는 것을 보인다.

```python
import numpy as np, collections
# ---- prepare_B2D.gengrate_map() 의 루프를 그대로 재현 ----
map_info = dict(np.load('/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/Town02_HD_map.npz',
                        allow_pickle=True)['arr'])
lane_points, lane_types, lane_sample_points = [], [], []
trigger_volumes_points, trigger_volumes_types, trigger_volumes_sample_points = [], [], []

for road_id, road in map_info.items():                 # road_id : 순회 변수
    for lane_id, lane in road.items():                 # lane_id : 순회 변수
        if lane_id == 'Trigger_Volumes':
            for stv in lane:
                p = np.array(stv['Points']); p[:,1] *= -1
                trigger_volumes_points.append(p)
                trigger_volumes_sample_points.append(p.mean(axis=0))
                trigger_volumes_types.append(stv['Type'])
        else:
            for single_lane in lane:
                points = np.array([rp[0] for rp in single_lane['Points']])   # location 만
                points[:,1] *= -1
                lane_points.append(points)                                   # 평평한 리스트에 append
                lane_types.append(single_lane['Type'])                       # 문자열 하나
                n = points.shape[0]
                dp = [50*i for i in range(n//50 + (1 if n % 50 else 0))]
                dp.append(n-1)
                lane_sample_points.append(points[dp])
# ---- 여기까지가 함수 전부. road_id / lane_id 는 어디에도 저장되지 않음 ----

print("=== gengrate_map() 출력 (Town02) ===")
print(f"  lane_points        : list, len={len(lane_points)}   원소 예: ndarray{lane_points[0].shape}")
print(f"  lane_types         : list, len={len(lane_types)}   원소 예: {lane_types[:6]}")
print(f"  lane_sample_points : list, len={len(lane_sample_points)}   원소 예: ndarray{lane_sample_points[0].shape}")
print(f"  trigger_volumes_*  : len={len(trigger_volumes_points)}  types={collections.Counter(trigger_volumes_types)}")
print()
print("  입력 npz : road", len(map_info), "개 ->", sum(len(v) for v in map_info.values()), "개 (road,lane) 키")
print("  출력     : 평평한 폴리라인", len(lane_points), "개.  road/lane 경계는 흔적 없음")
print()
print("  살아남은 필드 : Points(위치만), Type")
print("  사라진 필드   : road_id, lane_id, Left, Right, Topology, TopologyType, Color,")
print("                  roll/pitch/yaw, is_junction")
```

### `memprobe.py`

Town12 npz 가 16 GB 로도 부족함을 확인.

```python
import resource, sys, numpy as np, os
CAP = 16 * 1024**3                      # 16 GB 하드 상한 (공유 장비 보호)
resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))
town = sys.argv[1]
p = f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{town}_HD_map.npz'
print(f'{town}: {os.path.getsize(p)/1e9:.2f} GB 로딩 시도 (상한 16 GB)', flush=True)
try:
    m = dict(np.load(p, allow_pickle=True)['arr'])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    npoly = sum(len(v) for k, v in m.items() for kk, v in [(kk, vv) for kk, vv in v.items()])
    print(f'  성공: road {len(m)}개, peak RSS {rss:.1f} GB')
except MemoryError:
    print(f'  MemoryError — 16 GB 로 부족')
except Exception as e:
    print(f'  실패: {type(e).__name__}: {e}')
```

### `hist.py`

맵 타입 히스토그램.

```python
import numpy as np, collections
m = dict(np.load('/home/ailab/2026intern/jsn/maps/Town12_HD_map.npz', allow_pickle=True)['arr'])
c = collections.Counter(); pts = collections.Counter(); col = collections.Counter()
for rid, road in m.items():
    for lid, lane in road.items():
        if lid == 'Trigger_Volumes':
            for t in lane: c['TV:'+t['Type']] += 1
            continue
        for sl in lane:
            c[sl['Type']] += 1
            pts[sl['Type']] += len(sl['Points'])
            col[(sl['Type'], sl.get('Color'))] += 1
print("=== lane 'Type' histogram (polyline count / total points) ===")
for k, v in c.most_common():
    print(f"{k:22s} {v:7d}   pts={pts.get(k,0):9d}")
print("\n=== (Type, Color) ===")
for k, v in col.most_common(20): print(f"{str(k):40s} {v}")
KEEP = {'Broken','Solid','SolidSolid','Center'}
tot = sum(v for k,v in c.items() if not k.startswith('TV:'))
kept = sum(v for k,v in c.items() if k in KEEP)
print(f"\nb2d dataset keeps {kept}/{tot} = {100*kept/tot:.1f}% of polylines; drops {tot-kept}")
```
