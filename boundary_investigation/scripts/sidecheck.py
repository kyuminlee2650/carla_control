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
