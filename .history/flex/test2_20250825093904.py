import argparse, time, numpy as np, mujoco, mujoco.viewer

# ---------------- 配置 ----------------
DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_09_geom"

# 对齐 & 动作步长
XY_ALIGN_TOL   = 0.004
STEP_XY_SETTLE = 10
DOWN_STEP      = 0.003
PRINT_PERIOD   = 0.15

# 旋转控制（逐步推进）
ROT_STEP_MAX   = 0.06          # rad（~3.4°）
ROT_TOL        = 0.01          # rad（~0.57°）

# 高度
LIFT_SAFE      = 0.25
LIFT_UP_VALUE  = 0.30

# 夹爪：正=加紧，负=张开（按你的 XML）
OPEN_CMD       = 0.4
CLOSE_CMD      = -0.4

# 接触判据
CONTACT_TH      = 0.8
BOTH_CONTACT_TH = 0.7

# 与砖底面的安全缝隙
DOWN_SAFE_GAP   = 0.0015

# 姿态阈值
VERTICAL_THRESHOLD = 0.7       # |ez·Z| > 0.7 认为站着

# ---------------- 工具函数 ----------------
def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None

def contact_normal_force(model, data, i):
    f6 = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, f6)
    return abs(f6[0])

def forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_target):
    leftF = rightF = 0.0
    hitL = hitR = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        if (b1 == bid_left  and b2 == bid_target) or (b2 == bid_left  and b1 == bid_target):
            leftF  += contact_normal_force(model, data, i); hitL = True
        if (b1 == bid_right and b2 == bid_target) or (b2 == bid_right and b1 == bid_target):
            rightF += contact_normal_force(model, data, i); hitR = True
    return leftF, rightF, hitL, hitR

def wait_steps(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)

def body_bottom_z(model, data, bid):
    zmin = +1e9
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid:
            continue
        if model.geom_conaffinity[gid] == 0:
            continue
        half_z = 0.0
        if model.geom_size.shape[1] >= 3:
            half_z = float(model.geom_size[gid][2])
        zmin = min(zmin, float(data.geom_xpos[gid][2]) - half_z)
    return zmin

def target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009):
    cz = float(data.geom_xpos[gid_tgt][2])
    half_z = fallback_halfz
    if model.geom_size.shape[1] >= 3:
        hz = float(model.geom_size[gid_tgt][2])
        if hz > 1e-6:
            half_z = hz
    return cz - half_z

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def angle_diff_mod_pi(target, current):
    """
    以“π 等价（180°等价）”方式计算最短角度误差：
    适合平行夹爪（两指连线翻转180°仍可抓取）的任务。
    结果范围在 [-pi/2, +pi/2] 之间。
    """
    e = wrap_to_pi(target - current)
    if e >  np.pi/2: e -= np.pi
    if e < -np.pi/2: e += np.pi
    return e

def clamp_to_ctrlrange(model, aid, val):
    lo, hi = model.actuator_ctrlrange[aid]
    return float(np.clip(val, lo, hi))

# ---------------- 姿态分类 & 期望 yaw 计算 ----------------
def classify_vertical_or_horizontal(model, data, gid_target):
    """
    LEGO 长边 = 局部 z 轴（ez）。根据 ez 在世界系 Z 分量判断站/躺。
    返回: ('vertical' 或 'horizontal', R)，R 的列分别是 ex, ey, ez 在世界系的方向。
    """
    R = data.geom_xmat[gid_target].reshape(3, 3)
    ez = R[:, 2]
    if abs(ez[2]) > VERTICAL_THRESHOLD:
        return "vertical", R
    else:
        return "horizontal", R

def compute_desired_yaw_vertical(R):
    """
    站着：lego 的 z 与 palm 的 z 平行。
    目标：让 LEGO 的 x/y 与 palm 的 x/y 分别平行。
    做法：令 palm 的 x 轴与 LEGO 的 x 轴在 XY 对齐：yaw = atan2(ex_y, ex_x)
    """
    ex = R[:, 0]
    v  = np.array([ex[0], ex[1]])
    n  = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    v /= n
    return wrap_to_pi(float(np.arctan2(v[1], v[0])))

def compute_desired_yaw_horizontal(R):
    """
    躺着：lego 的 z 与地面平行。
    目标：让 lego 的 z 与 palm 的 y 平行（在 XY 平面对齐）。
    palm y(ψ) 的 XY 方向为 [-sinψ, cosψ]。
    令 [-sinψ, cosψ] 与 ez 的水平投影对齐：cosψ = ez_y, sinψ = -ez_x → ψ = atan2(-ez_x, ez_y)
    """
    ez = R[:, 2]
    v  = np.array([ez[0], ez[1]])
    n  = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    v /= n
    vx, vy = float(v[0]), float(v[1])
    return wrap_to_pi(float(np.arctan2(-vx, vy)))

def compute_desired_yaw(model, data, gid_target):
    state, R = classify_vertical_or_horizontal(model, data, gid_target)
    if state == "vertical":
        yaw_des = compute_desired_yaw_vertical(R)
    else:
        yaw_des = compute_desired_yaw_horizontal(R)
    return state, yaw_des

# ---------------- 主逻辑 ----------------
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
    jid_yaw   = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
    qadr_yaw  = model.jnt_qposadr[jid_yaw] if jid_yaw is not None else None

    # body/geom ids
    bid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    bid_left  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    gid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target)
    bid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.target.replace("_geom", ""))

    if any(v is None for v in [aid_x, aid_y, aid_lift, aid_left, aid_right, aid_rot,
                               bid_palm, bid_left, bid_right, gid_tgt, bid_tgt,
                               jid_yaw, qadr_yaw]):
        print("[ERROR] 名称映射失败：检查 x/y/lift/rotation，yaw 关节，以及 palm/left_link/right_link/lego_*")
        return

    # 执行器范围
    loZ, hiZ = model.actuator_ctrlrange[aid_lift]
    loR, hiR = model.actuator_ctrlrange[aid_rot]

    # 初始：张开并抬高，yaw 置 0
    data.ctrl[aid_left]  = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift]  = float(np.clip(LIFT_SAFE, loZ, hiZ))
    data.ctrl[aid_rot]   = float(np.clip(0.0, loR, hiR))
    wait_steps(model, data, 80)

    # 标定 palm→指尖偏移
    palm0 = data.xpos[bid_palm].copy()
    finger_bottom0 = body_bottom_z(model, data, bid_left)
    palm2tip = float(palm0[2] - finger_bottom0)
    if not (0.02 <= palm2tip <= 0.30):
        palm2tip = 0.142
    print(f"[calib] palm2tip = {palm2tip:.4f} m")

    phase = "align_xy"
    last_print = 0.0
    lift_cmd = float(data.ctrl[aid_lift])
    desired_yaw = 0.0
    pose_state = "unknown"

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)

            palm   = data.xpos[bid_palm]
            target = data.geom_xpos[gid_tgt]
            tip_z  = body_bottom_z(model, data, bid_left)
            tgt_bottom = target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009)
            z_goal_tip = tgt_bottom + DOWN_SAFE_GAP
            err_tip    = tip_z - z_goal_tip
            yaw_cur    = float(data.qpos[qadr_yaw])

            # 打印
            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                err_show = angle_diff_mod_pi(desired_yaw, yaw_cur) if phase == "rotate" else 0.0
                print(f"{phase} state={pose_state} des_yaw={np.degrees(desired_yaw):.1f}° "
                      f"yaw_cur={np.degrees(yaw_cur):.1f}° ctrl_rot={np.degrees(data.ctrl[aid_rot]):.1f}° "
                      f"err={np.degrees(err_show):.1f}°")
                last_print = now

            # 接触检测
            leftF, rightF, hitL, hitR = forces_fingers_vs_target_by_body(
                model, data, bid_left, bid_right, bid_tgt
            )

            # ---------------- 状态机 ----------------
            if phase == "align_xy":
                loX, hiX = model.actuator_ctrlrange[aid_x]
                loY, hiY = model.actuator_ctrlrange[aid_y]
                data.ctrl[aid_x] = float(np.clip(target[0], loX, hiX))
                data.ctrl[aid_y] = float(np.clip(target[1], loY, hiY))
                wait_steps(model, data, STEP_XY_SETTLE)

                if abs(palm[0]-target[0]) < XY_ALIGN_TOL and abs(palm[1]-target[1]) < XY_ALIGN_TOL:
                    pose_state, desired_yaw = compute_desired_yaw(model, data, gid_tgt)
                    print(f"[ALIGN] pose={pose_state}, desired_yaw={np.degrees(desired_yaw):.1f}°")
                    phase = "rotate"

            elif phase == "rotate":
                # 用“π 等价”的最短路误差推进
                err = angle_diff_mod_pi(desired_yaw, yaw_cur)
                step = float(np.clip(err, -ROT_STEP_MAX, ROT_STEP_MAX))
                data.ctrl[aid_rot] = clamp_to_ctrlrange(model, aid_rot, yaw_cur + step)
                wait_steps(model, data, 2)

                # 重新读取判断是否到位
                yaw_cur = float(data.qpos[qadr_yaw])
                if abs(angle_diff_mod_pi(desired_yaw, yaw_cur)) < ROT_TOL:
                    print(f"[ROTATE] 完成 yaw={np.degrees(yaw_cur):.1f}° → 目标={np.degrees(desired_yaw):.1f}°")
                    phase = "descend"

            elif phase == "descend":
                # 固持 yaw
                yaw_cur = float(data.qpos[qadr_yaw])
                err_hold = angle_diff_mod_pi(desired_yaw, yaw_cur)
                step_hold = float(np.clip(err_hold, -ROT_STEP_MAX, ROT_STEP_MAX))
                data.ctrl[aid_rot] = clamp_to_ctrlrange(model, aid_rot, yaw_cur + step_hold)

                # 下降到目标上方
                z_target_bottom = target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009)
                z_goal_palm = z_target_bottom + DOWN_SAFE_GAP + palm2tip
                errz = float(palm[2]) - z_goal_palm

                if errz <= 0.001:
                    print(f"[DESCEND] 已到位")
                    phase = "close"
                else:
                    step = min(DOWN_STEP, max(0.0, errz) * 0.5)
                    lift_cmd = float(np.clip(lift_cmd - step, loZ, hiZ))
                    data.ctrl[aid_lift] = lift_cmd + 0.19
                wait_steps(model, data, 2)

                if (hitL or hitR) or err_tip <= 0.001:
                    print(f"[DESCEND] 检测到接触")
                    phase = "close"

                if errz < 0.015 and not (hitL and hitR):
                    data.ctrl[aid_left]  = min(CLOSE_CMD, data.ctrl[aid_left]  + 0.1)
                    data.ctrl[aid_right] = min(CLOSE_CMD, data.ctrl[aid_right] + 0.1)

            elif phase == "close":
                # 固持 yaw
                yaw_cur = float(data.qpos[qadr_yaw])
                err_hold = angle_diff_mod_pi(desired_yaw, yaw_cur)
                step_hold = float(np.clip(err_hold, -ROT_STEP_MAX, ROT_STEP_MAX))
                data.ctrl[aid_rot] = clamp_to_ctrlrange(model, aid_rot, yaw_cur + step_hold)

                leftF, rightF, hitL, hitR = forces_fingers_vs_target_by_body(
                    model, data, bid_left, bid_right, bid_tgt
                )
                if not (hitL or hitR):
                    data.ctrl[aid_left]  = CLOSE_CMD
                    data.ctrl[aid_right] = CLOSE_CMD
                    wait_steps(model, data, 2)
                elif hitL and hitR:
                    wait_steps(model, data, 2)
                    if (leftF + rightF) >= BOTH_CONTACT_TH:
                        print(f"[CLOSE] 抓取成功，力: {leftF+rightF:.2f}N")
                        phase = "lift"

            elif phase == "lift":
                # 保持夹紧 & 固持 yaw
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                yaw_cur = float(data.qpos[qadr_yaw])
                err_hold = angle_diff_mod_pi(desired_yaw, yaw_cur)
                step_hold = float(np.clip(err_hold, -ROT_STEP_MAX, ROT_STEP_MAX))
                data.ctrl[aid_rot] = clamp_to_ctrlrange(model, aid_rot, yaw_cur + step_hold)

                # 抬起
                z_goal   = float(np.clip(LIFT_UP_VALUE, loZ, hiZ))
                cur = float(data.ctrl[aid_lift])
                errz = z_goal - cur
                step = min(0.003, abs(errz) * 0.5)
                new_val = cur + np.sign(errz) * step
                data.ctrl[aid_lift] = float(np.clip(new_val, loZ, hiZ))
                wait_steps(model, data, 2)

            # 刷新
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
