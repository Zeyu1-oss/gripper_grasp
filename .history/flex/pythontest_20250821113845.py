import time
import numpy as np
import mujoco
import mujoco.viewer

XML = "1lego.xml"
TARGET = "lego_geom"

def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None

def main():
    model = mujoco.MjModel.from_xml_path(XML)
    data  = mujoco.MjData(model)

    # 找 actuator
    aid_x     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_grasp = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grasp")

    # geom id
    gid_palm   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm")
    gid_target = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TARGET)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        phase = "align"  # align -> close -> lift
        t0 = time.time()
        while viewer.is_running():
            mujoco.mj_forward(model, data)

            palm   = data.geom_xpos[gid_palm]
            target = data.geom_xpos[gid_target]

            if phase == "align":
                # 移动到目标上方
                data.ctrl[aid_x]    = target[0]
                data.ctrl[aid_y]    = target[1]
                data.ctrl[aid_lift] = target[2] + 0.05

                if abs(palm[0]-target[0])<0.01 and abs(palm[1]-target[1])<0.01:
                    phase = "close"

            elif phase == "close":
                data.ctrl[aid_grasp] = min(data.ctrl[aid_grasp] + 0.01, 1.0)
                if data.ctrl[aid_grasp] >= 1.0:
                    phase = "lift"

            elif phase == "lift":
                data.ctrl[aid_lift] = 0.4  # 往上抬

            mujoco.mj_step(model, data)
            viewer.sync()

        print("Simulation finished in %.2f s" % (time.time() - t0))

if __name__ == "__main__":
    main()
