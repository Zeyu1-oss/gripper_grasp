import mujoco
import mujoco.viewer
import numpy as np
import time

XML = "g2_with_30lego.xml"

def name2id(model, objtype, name):
    try: return mujoco.mj_name2id(model, objtype, name)
    except: return None

def main():
    model = mujoco.MjModel.from_xml_path(XML)
    data  = mujoco.MjData(model)

    aid_rot = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
    jid_yaw = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
    qadr_yaw = model.jnt_qposadr[jid_yaw]

    loR, hiR = model.actuator_ctrlrange[aid_rot]

    # viewer 打开
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for target in np.linspace(loR, hiR, 10):  # 在 -pi 到 +pi 之间均匀取10个目标
            print(f"\n[TEST] 设置 ctrl={target:.3f} rad ({np.degrees(target):.1f}°)")
            data.ctrl[aid_rot] = target
            for _ in range(300):  # 模拟 0.3s
                mujoco.mj_step(model, data)
                viewer.sync()
            yaw_cur = float(data.qpos[qadr_yaw])
            print(f"      实际 yaw={yaw_cur:.3f} rad ({np.degrees(yaw_cur):.1f}°), "
                  f"误差={np.degrees(target-yaw_cur):.2f}°")

if __name__ == "__main__":
    main()
