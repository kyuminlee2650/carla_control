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
