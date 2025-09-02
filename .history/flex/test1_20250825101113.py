import mujoco, mujoco.viewer
import numpy as np, time

model = mujoco.MjModel.from_xml_path("test1.xml")
data  = mujoco.MjData(model)

aid_rot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
jid_yaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
qadr_yaw = model.jnt_qposadr[jid_yaw]

with mujoco.viewer.launch_passive(model, data) as viewer:
    for target in np.linspace(-np.pi, np.pi, 9):  # 从 -180° 到 +180° 逐步设置
        data.ctrl[aid_rot] = target
        for _ in range(200):
            mujoco.mj_step(model, data)
            viewer.sync()
        print(f"ctrl={np.degrees(target):.1f}°, yaw={np.degrees(data.qpos[qadr_yaw]):.1f}°")
        time.sleep(0.5)
