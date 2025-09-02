import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("gripper1.xml")
data  = mujoco.MjData(model)

aid_grasp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grasp")

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        if step < 300:
            data.ctrl[aid_grasp] = 0.0   # 张开
        else:
            data.ctrl[aid_grasp] = 1.0   # 闭合
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
