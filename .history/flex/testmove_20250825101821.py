import mujoco, numpy as np

model = mujoco.MjModel.from_xml_path("g2_with_30lego.xml")
data = mujoco.MjData(model)

aid_rot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
jid_yaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
qadr_yaw = model.jnt_qposadr[jid_yaw]

for deg in [-180,-135,-90,-45,0,45,90,135,180]:
    data.ctrl[aid_rot] = np.deg2rad(deg)
    for _ in range(200):  # 等待收敛
        mujoco.mj_step(model, data)
    print(f"ctrl={deg:.1f}°, yaw={np.rad2deg(data.qpos[qadr_yaw]):.1f}°")
