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
