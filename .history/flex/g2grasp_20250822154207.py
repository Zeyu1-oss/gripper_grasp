import argparse, time, numpy as np, mujoco, mujoco.viewer

DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_02_geom"

# --- 参数 ---
XY_ALIGN_TOL   = 0.004
STEP_XY_SETTLE = 10
DOWN_STEP      = 0.003
PRINT_PERIOD   = 0.15
LIFT_SAFE      = 0.25
LIFT_UP_VALUE  = 0.30
DOWN_SAFE_GAP  = 0.0015

# 手指：正=加紧，负=张开
OPEN_CMD       = -0.8
CLOSE_CMD      = +0.8

# 旋转控制（每步最大角度变化）
ROT_STEP_MAX   = 0.06  # rad
ROT_TOL        = 0.05  # 接近阈值 (rad)

# 接触判据
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
            leftF  += contact_normal_force(model, data, i); hitL = True
        if (b1==bid_right and b2==bid_target) or (b2==bid_right and b1==bid_target):
            rightF += contact_normal_force(model, data, i); hitR = True
    return leftF, rightF, hitL, hitR

def wait_steps(model, data, n):
    for _ in range(n): mujoco.mj_step(model, data)

def body_bottom_z(model, data, bid):
    """某 body 的可碰撞几何的最底 z（忽略纯视觉几何）"""
    zmin = +1e9
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid: continue
        if model.geom_conaffinity[gid] == 0:  # 视觉几何
            continue
        half_z = 0.0
        if model.geom_size.shape[1] >= 3:
            half_z = float(model.geom_size[gid][2])
        zmin = min(zmin, float(data.geom_xpos[gid][2]) - half_z)
    return zmin if zmin < 1e8 else float(data.xpos[bid][2])

def target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009):
    cz = float(data.geom_xpos[gid_tgt][2])
    half_z = fallback_halfz
    if model.geom_size.shape[1] >= 3:
        hz = float(model.geom_size[gid_tgt][2])
        if hz > 1e-6: half_z = hz
    return cz - half_z

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def compute_grasp_yaw(model, data, gid_target):
    """
    根据乐高长边方向，计算让手指与长边垂直的 yaw。
    取 geom_xmat 的列向量中“半边长”最大的那根对应的世界方向；再 +90° 使手指法向对齐。
    """
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]   # 半轴 (x,y,z)
    axis = int(np.argmax(size))          # 0=x, 1=y, 2=z
    long_dir_world = R[:, axis]

    dir_xy = np.array([long_dir_world[0], long_dir_world[1]])
    n = np.linalg.norm(dir_xy)
    if n < 1e-8:      # 近似竖直，随便给个 0
        return 0.0
    dir_xy /= n
    angle = np.arctan2(dir_xy[1], dir_xy[0])
    return wrap_to_pi(angle + np.pi/2)   # 手指与长边垂直

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model); mujoco.mj_forward(model, data)

    # actuator ids
    aid_x      = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y      = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_left   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    aid_right  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")
    aid_rot    = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")  # 新增

    # joint id（为读当前角度）
    jid_yaw = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")             # 新增
    qadr_yaw = model.jnt_qposadr[jid_yaw] if jid_yaw is not None else None

    # body/geom ids
    bid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    bid_left  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    gid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target)
    bid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.target.replace("_geom",""))

    if any(v is None for v in [aid_x, aid_y, aid_lift, aid_left, aid_right, aid_rot,
                               bid_palm, bid_left, bid_right, gid_tgt, bid_tgt,
                               jid_yaw, qadr_yaw]):
        print("[ERROR] 名称映射失败：检查 x/y/lift/rotation，yaw 关节，以及 palm/left_link/right_link/lego_*")
        return

    loZ, hiZ = model.actuator_ctrlrange[aid_lift]
    loR, hiR = model.actuator_ctrlrange[aid_rot]

    # 初始：张开并抬高
    data.ctrl[aid_left]  = OPEN_CMD   # 负=张开
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift]  = float(np.clip(LIFT_SAFE, loZ, hiZ))
    # 初始 yaw 保持 0（也可留在当前值）
    data.ctrl[aid_rot]   = float(np.clip(0.0, loR, hiR))
    wait_steps(model, data, 80)

    phase = "align_xy"
    last_print = 0.0
    lift_cmd = float(data.ctrl[aid_lift])
    desired_yaw = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)

            palm   = data.xpos[bid_palm]               # 世界坐标
            target = data.geom_xpos[gid_tgt]
            tip_z  = body_bottom_z(model, data, bid_left)
            tgt_bottom = target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009)
            z_goal_tip = tgt_bottom + DOWN_SAFE_GAP
            err_tip    = tip_z - z_goal_tip

            yaw_cur = float(data.qpos[qadr_yaw])

            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                print(f"[{phase:9s}] palm=({palm[0]:+.3f},{palm[1]:+.3f},{palm[2]:.3f})  "
                      f"tgt=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f})  "
                      f"tip_z={tip_z:.3f}  z_goal_tip={z_goal_tip:.3f}  err_tip={err_tip:.3f}  "
                      f"yaw_cur={yaw_cur:+.3f} yaw_cmd={data.ctrl[aid_rot]:+.3f}  "
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
                    desired_yaw = compute_grasp_yaw(model, data, gid_tgt)
                    phase = "rotate"

            elif phase == "rotate":
                # 把 yaw 转向 desired_yaw（限速，防止过快震荡）
                err = wrap_to_pi(desired_yaw - yaw_cur)
                step = float(np.clip(err, -ROT_STEP_MAX, ROT_STEP_MAX))
                data.ctrl[aid_rot] = float(np.clip(data.ctrl[aid_rot] + step, loR, hiR))
                wait_steps(model, data, 2)
                if abs(err) < ROT_TOL:
                    phase = "descend"

            elif phase == "descend":
                step = min(DOWN_STEP, max(0.0, err_tip) * 0.5)
                lift_cmd = float(np.clip(lift_cmd - step, loZ, hiZ))
                data.ctrl[aid_lift] = lift_cmd
                wait_steps(model, data, 2)

                if err_tip <= 0.002 or (hitL or hitR):
                    phase = "close"

                if lift_cmd <= loZ + 1e-5 and err_tip > 0.005:
                    print("[ABORT] lift 已到下限仍未达到目标高度，检查几何尺寸/偏置。")
                    break

            elif phase == "close":
                # 只夹紧，不再向下压
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                wait_steps(model, data, 2)
                if hitL and hitR and (leftF + rightF) >= BOTH_CONTACT_TH and \
                   leftF >= CONTACT_TH and rightF >= CONTACT_TH:
                    phase = "lift"

            elif phase == "lift":
                data.ctrl[aid_lift]  = float(np.clip(LIFT_UP_VALUE, loZ, hiZ))
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                wait_steps(model, data, 2)

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
