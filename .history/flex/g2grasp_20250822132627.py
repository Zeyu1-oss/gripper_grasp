import argparse, time, numpy as np, mujoco, mujoco.viewer

DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_02_geom"

XY_ALIGN_TOL   = 0.004
STEP_XY_SETTLE = 10
DOWN_STEP      = 0.003
PRINT_PERIOD   = 0.15
LIFT_SAFE      = 0.25
LIFT_UP_VALUE  = 0.30

# 正=加紧，负=张开 —— 按你的规则
OPEN_CMD       = -0.8
CLOSE_CMD      = +0.8

CONTACT_TH      = 0.2
BOTH_CONTACT_TH = 0.5

def name2id(model, objtype, name):
    try: return mujoco.mj_name2id(model, objtype, name)
    except Exception: return None

def contact_normal_force(model, data, i):
    f6 = np.zeros(6); mujoco.mj_contactForce(model, data, i, f6)
    return abs(f6[0])

def forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_target):
    leftF = rightF = 0.0; hitL = hitR = False
    for i in range(data.ncon):
        c = data.contact[i]; g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        if (b1==bid_left  and b2==bid_target) or (b2==bid_left  and b1==bid_target):
            leftF  += contact_normal_force(model, data, i);  hitL = True
        if (b1==bid_right and b2==bid_target) or (b2==bid_right and b1==bid_target):
            rightF += contact_normal_force(model, data, i);  hitR = True
    return leftF, rightF, hitL, hitR

def wait_steps(model, data, n):
    for _ in range(n): mujoco.mj_step(model, data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # actuator ids
    aid_x     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_left  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    aid_right = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")

    # body/geom ids
    bid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    bid_left  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    gid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target)
    bid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.target.replace("_geom",""))

    if any(v is None for v in [aid_x, aid_y, aid_lift, aid_left, aid_right,
                               bid_palm, bid_left, bid_right, gid_tgt, bid_tgt]):
        print("[ERROR] 名称映射失败，检查 x/y/lift, left_joint/right_joint, palm/left_link/right_link, lego_*_geom")
        return

    # 初始：张开并抬高
    data.ctrl[aid_left]  = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    loL, hiL = model.actuator_ctrlrange[aid_lift]
    data.ctrl[aid_lift]  = float(np.clip(LIFT_SAFE, loL, hiL))
    wait_steps(model, data, 80)

    phase = "align_xy"; last_print = 0.0
    lift_cmd = float(data.ctrl[aid_lift])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)

            palm   = data.xpos[bid_palm]
            target = data.geom_xpos[gid_tgt]

            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                print(f"[{phase:9s}] palm=({palm[0]:+.3f},{palm[1]:+.3f},{palm[2]:.3f})  "
                      f"tgt=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f})  "
                      f"ctrl(x,y,lift)=({data.ctrl[aid_x]:+.3f},{data.ctrl[aid_y]:+.3f},{data.ctrl[aid_lift]:+.3f})")
                last_print = now

            leftF, rightF, hitL, hitR = forces_fingers_vs_target_by_body(
                model, data, bid_left, bid_right, bid_tgt
            )

            if phase == "align_xy":
                loX, hiX = model.actuator_ctrlrange[aid_x]
                loY, hiY = model.actuator_ctrlrange[aid_y]
                data.ctrl[aid_x] = float(np.clip(target[0], loX, hiX))
                data.ctrl[aid_y] = float(np.clip(target[1], loY, hiY))
                wait_steps(model, data, STEP_XY_SETTLE)
                if abs(palm[0]-target[0]) < XY_ALIGN_TOL and abs(palm[1]-target[1]) < XY_ALIGN_TOL:
                    phase = "descend"

            elif phase == "descend":
                loZ, hiZ = model.actuator_ctrlrange[aid_lift]
                lift_cmd = float(np.clip(lift_cmd - DOWN_STEP, loZ, hiZ))
                data.ctrl[aid_lift] = lift_cmd
                wait_steps(model, data, 2)
                if hitL or hitR:
                    phase = "close"

            elif phase == "close":
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                if not (hitL and hitR):
                    data.ctrl[aid_lift] = float(np.clip(data.ctrl[aid_lift]-0.001, loZ, hiZ))
                wait_steps(model, data, 2)
                if hitL and hitR and (leftF+rightF) >= BOTH_CONTACT_TH and leftF >= CONTACT_TH and rightF >= CONTACT_TH:
                    phase = "lift"

            elif phase == "lift":
                data.ctrl[aid_lift] = float(np.clip(LIFT_UP_VALUE, loZ, hiZ))
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                wait_steps(model, data, 2)

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
