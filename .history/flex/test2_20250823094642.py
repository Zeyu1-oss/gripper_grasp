import argparse, time, numpy as np, mujoco, mujoco.viewer

# ---------------- 配置 ----------------
DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_09_geom"

XY_ALIGN_TOL   = 0.004
STEP_XY_SETTLE = 10
DOWN_STEP_MAX  = 0.004
PRINT_PERIOD   = 0.15

LIFT_SAFE      = 0.25
LIFT_UP_VALUE  = 0.30

OPEN_CMD       = 0.4     # 张开
CLOSE_CMD      = -0.8    # 闭合

CONTACT_TH      = 0.5
BOTH_CONTACT_TH = 0.5
DOWN_SAFE_GAP   = 0.0015

ROT_STEP_MAX   = 0.06
ROT_TOL        = 0.05

# ---------------- 工具函数 ----------------
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

def body_bottom_z(model, data, bid):
    zmin = +1e9
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid: continue
        if model.geom_conaffinity[gid] == 0: continue
        half_z = float(model.geom_size[gid][2]) if model.geom_size.shape[1] >= 3 else 0.0
        zmin = min(zmin, float(data.geom_xpos[gid][2]) - half_z)
    return zmin

def target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009):
    cz = float(data.geom_xpos[gid_tgt][2])
    half_z = fallback_halfz
    if model.geom_size.shape[1] >= 3:
        hz = float(model.geom_size[gid_tgt][2])
        if hz > 1e-6: half_z = hz
    return cz - half_z

def compute_grasp_yaw(model, data, gid_target):
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]
    axis = np.argmax(size)  # 长边方向
    long_dir_world = R[:, axis]
    dir_xy = np.array([long_dir_world[0], long_dir_world[1]])
    if np.linalg.norm(dir_xy) < 1e-6: return 0.0
    dir_xy /= np.linalg.norm(dir_xy)
    return np.arctan2(dir_xy[1], dir_xy[0]) + np.pi/2

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# ---------------- 主流程 ----------------
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
    aid_rot   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")

    # body/geom ids
    bid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    bid_left  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    gid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target)
    bid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.target.replace("_geom", ""))

    if any(v is None for v in [aid_x, aid_y, aid_lift, aid_left, aid_right, aid_rot,
                               bid_palm, bid_left, bid_right, gid_tgt, bid_tgt]):
        print("[ERROR] 名称映射失败"); return

    loZ, hiZ = model.actuator_ctrlrange[aid_lift]
    loR, hiR = model.actuator_ctrlrange[aid_rot]

    # 初始：张开 + 抬高
    data.ctrl[aid_left]  = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift]  = float(np.clip(LIFT_SAFE, loZ, hiZ))
    data.ctrl[aid_rot]   = 0.0
    wait_steps(model, data, 80)

    # 标定 palm2tip
    palm0 = data.xpos[bid_palm].copy()
    finger_bottom0 = body_bottom_z(model, data, bid_left)
    palm2tip = float(palm0[2] - finger_bottom0)
    if not (0.02 <= palm2tip <= 0.30): palm2tip = 0.142
    print(f"[calib] palm2tip = {palm2tip:.4f} m")

    phase = "align_xy"
    last_print = 0.0
    lift_cmd = float(data.ctrl[aid_lift])
    desired_yaw = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)

            palm   = data.xpos[bid_palm]
            target = data.geom_xpos[gid_tgt]
            tip_z  = body_bottom_z(model, data, bid_left)
            tgt_bottom = target_bottom_z(model, data, gid_tgt)
            z_goal_tip = tgt_bottom + DOWN_SAFE_GAP
            err_tip    = tip_z - z_goal_tip

            yaw_cur = float(data.qpos[model.jnt_qposadr[name2id(model, mujoco.mjtObj.mjOBJ_JOINT,"yaw")]]) \
                        if name2id(model, mujoco.mjtObj.mjOBJ_JOINT,"yaw") is not None else 0.0

            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                print(f"tip_z={tip_z:.3f} z_goal_tip={z_goal_tip:.3f} err_tip={err_tip:.3f} "
                      f"yaw_cur={yaw_cur:+.3f} yaw_cmd={data.ctrl[aid_rot]:+.3f} "
                      f"ctrl(x,y,lift)=({data.ctrl[aid_x]:+.3f},{data.ctrl[aid_y]:+.3f},{data.ctrl[aid_lift]:+.3f}) "
                      f"target=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f})")
                last_print = now

            leftF, rightF, hitL, hitR = forces_fingers_vs_target_by_body(
                model, data, bid_left, bid_right, bid_tgt
            )

            # ---------------- 状态机 ----------------
            if phase == "align_xy":
                data.ctrl[aid_x] = np.clip(target[0], *model.actuator_ctrlrange[aid_x])
                data.ctrl[aid_y] = np.clip(target[1], *model.actuator_ctrlrange[aid_y])
                wait_steps(model, data, STEP_XY_SETTLE)
                if abs(palm[0]-target[0]) < XY_ALIGN_TOL and abs(palm[1]-target[1]) < XY_ALIGN_TOL:
                    desired_yaw = compute_grasp_yaw(model, data, gid_tgt)
                    phase = "rotate"

            elif phase == "rotate":
                err = wrap_to_pi(desired_yaw - yaw_cur)
                step = np.clip(err, -ROT_STEP_MAX, ROT_STEP_MAX)
                data.ctrl[aid_rot] = np.clip(data.ctrl[aid_rot] + step, loR, hiR)
                wait_steps(model, data, 2)
                if abs(err) < ROT_TOL: phase = "descend"

            elif phase == "descend":
                z_goal_tip = target_bottom_z(model, data, gid_tgt) + DOWN_SAFE_GAP
                desired_lift = z_goal_tip + palm2tip - 0.557
                err = desired_lift - data.ctrl[aid_lift]
                step = np.clip(err, -DOWN_STEP_MAX, DOWN_STEP_MAX)
                data.ctrl[aid_lift] = np.clip(data.ctrl[aid_lift] + step, loZ, hiZ)
                wait_steps(model, data, 2)
                if (hitL or hitR) or abs(err) < 0.002: phase = "close"

            elif phase == "close":
    # 读取当前接触与力
                    leftF, rightF, hitL, hitR = forces_fingers_vs_target_by_body(
                        model, data, bid_left, bid_right, bid_tgt
                    )

                    if not (hitL or hitR):
                        data.ctrl[aid_left]  = CLOSE_CMD
                        data.ctrl[aid_right] = CLOSE_CMD
                        wait_steps(model, data, 2)

                    elif hitL and hitR:      # 双侧接触 

                        wait_steps(model, data, 2)

                        if (leftF + rightF) >= BOTH_CONTACT_TH :
                           phase = "lift

            elif phase == "lift":
                err = LIFT_UP_VALUE - data.ctrl[aid_lift]
                step = np.clip(err, -0.003, 0.003)
                data.ctrl[aid_lift] = np.clip(data.ctrl[aid_lift] + step, loZ, hiZ)
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                wait_steps(model, data, 2)

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
