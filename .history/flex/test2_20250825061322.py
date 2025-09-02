import argparse, time, numpy as np, mujoco, mujoco.viewer

# ---------------- 配置 ----------------
DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_09_geom"

# 对齐 & 动作步长
XY_ALIGN_TOL   = 0.004         # XY 对齐阈值 (m)
STEP_XY_SETTLE = 10            # XY 每次设置后等待步数
DOWN_STEP      = 0.003         # 单步最大下降 (m)
PRINT_PERIOD   = 0.15          # 打印节流 (s)

# 旋转相关
ROT_TOL_FINGER = 0.03          # 以“两指连线角度”为准的容差 (rad)，~1.7°
ROT_SETTLE_STEPS = 2           # 每次给定目标后等待步数

# 初始与抬起高度
LIFT_SAFE      = 0.25          # 初始抬升到的 lift 目标
LIFT_UP_VALUE  = 0.30          # 抓到后抬升到的 lift 目标

# 夹爪：正=加紧，负=张开（你原本的定义）
OPEN_CMD       = 0.4
CLOSE_CMD      = -0.4

# 接触判据
CONTACT_TH      = 0.8
BOTH_CONTACT_TH = 0.7

# 与砖底面的安全缝隙
DOWN_SAFE_GAP   = 0.0015

# 姿态判断阈值
VERTICAL_THRESHOLD = 0.7       # Z分量大于此值认为是竖立状态

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

# ---------- 角度/向量小工具 ----------
def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def angle_of_xy(vec2):
    """vec2: (x,y) -> atan2(y,x) with safe norm check"""
    n = np.linalg.norm(vec2)
    if n < 1e-9:
        return 0.0
    return float(np.arctan2(vec2[1], vec2[0]))

def angle_diff_mod_pi(target, current):
    """
    返回目标与当前的“mod π”（180°对称等价）的最小差值
    这样对平行夹爪（翻转180°等价）就不会卡在+/-π
    """
    e = wrap_to_pi(target - current)
    if e >  np.pi/2: e -= np.pi
    if e < -np.pi/2: e += np.pi
    return e

def finger_line_angle(model, data, bid_left, bid_right):
    """
    两指连线的 XY 朝向角 phi_finger。
    直接用两指 body 质心的投影连线，鲁棒地反映了实际“夹爪张合方向”。
    """
    pL = data.xpos[bid_left]
    pR = data.xpos[bid_right]
    v  = np.array([pR[0]-pL[0], pR[1]-pL[1]])
    return angle_of_xy(v)

# ---------- 目标姿态判断 & 目标抓取方向 ----------
def determine_lego_orientation(model, data, gid_target):
    """判断乐高是站着(vertical)还是躺着(horizontal)，并返回 longest_axis 索引"""
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]

    longest_axis = int(np.argmax(size))
    longest_dir  = R[:, longest_axis]
    vertical_component = abs(longest_dir[2])
    if vertical_component > VERTICAL_THRESHOLD:
        return "vertical", longest_axis
    else:
        return "horizontal", longest_axis

def compute_target_finger_angle(model, data, gid_target, orientation, longest_axis):
    """
    计算“目标两指连线角” phi_target（世界系 XY 平面角度）。
    设计策略（稳 & 好夹）：
      - vertical（站立）：长轴≈Z，其余两个轴在XY。选择“在XY投影尺寸更大”的那个轴作为夹持法向 → 匹配更大接触面/更稳。
      - horizontal（躺平）：三个轴里挑“XY投影×尺寸”最大的轴（即在平面上最显著的轴）作为夹持法向。
    解释：两指连线应当与“被夹的那一对相对侧面”的法向一致；用这个角作 phi_target。
    """
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]

    # 每个轴的水平分量幅度（投影到XY）
    axes = [R[:,0], R[:,1], R[:,2]]
    horiz_amp = [np.linalg.norm([ax[0], ax[1]]) for ax in axes]
    scores = [size[i] * horiz_amp[i] for i in range(3)]  # “在XY的有效尺寸”

    if orientation == "vertical":
        # 排除近似竖直的长轴，剩下两个在XY，取在XY更“粗”的那个
        cand = [i for i in range(3) if i != longest_axis]
        best_axis = max(cand, key=lambda i: scores[i])
    else:
        # 躺平：直接取在XY最显著的轴（多半就是物体的摆放主轴）
        best_axis = int(np.argmax(scores))

    best_dir = axes[best_axis]
    dir_xy   = np.array([best_dir[0], best_dir[1]])
    if np.linalg.norm(dir_xy) < 1e-9:
        # 极端退化：如果几乎没水平分量，就不转
        return 0.0
    dir_xy /= np.linalg.norm(dir_xy)
    phi = float(np.arctan2(dir_xy[1], dir_xy[0]))
    return phi

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
        print("[ERROR] 名称映射失败")
        return

    # 执行器范围
    loZ, hiZ = model.actuator_ctrlrange[aid_lift]
    loR, hiR = model.actuator_ctrlrange[aid_rot]

    # 初始状态
    data.ctrl[aid_left]  = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift]  = float(np.clip(LIFT_SAFE, loZ, hiZ))
    # 对 position actuator：ctrl 即“目标角度”（绝对），先给 0
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

    phi_target = 0.0
    lego_orientation = "unknown"

    # --- 旋转阶段的稳健控制附加变量 ---
    rot_calibrated = False   # 是否完成符号标定
    rot_sign = +1.0          # yaw 与 phi 的方向映射（自动标定）
    prev_err_phi = None
    rot_stall_count = 0
    ROT_STALL_LIMIT = 30     # 连续若干次“没改进”后触发兜底
    ROT_TEST_DELTA = 0.05    # 标定用的小试探角（弧度）    last_print = 0.0
    lift_cmd = float(data.ctrl[aid_lift])
    # 旋转期望改为“以两指连线角”为基准
    phi_target = 0.0
    lego_orientation = "unknown"

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

            # 打印信息
            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                phi_cur = finger_line_angle(model, data, bid_left, bid_right)
                err_phi = angle_diff_mod_pi(phi_target, phi_cur) if phase == "rotate" else 0.0
                print(f"{phase} orient={lego_orientation} "
                      f"phi_target={np.degrees(phi_target):.1f}° phi_cur={np.degrees(phi_cur):.1f}° "
                      f"yaw_cur={np.degrees(yaw_cur):.1f}° err_phi={np.degrees(err_phi):.1f}°")
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
                    # 判断乐高姿态并计算“目标两指连线角”
                    lego_orientation, longest_axis = determine_lego_orientation(model, data, gid_tgt)
                    phi_target = compute_target_finger_angle(model, data, gid_tgt, lego_orientation, longest_axis)
                    print(f"[ALIGN] 乐高姿态: {lego_orientation}, 最长轴: {longest_axis}, "
                          f"phi_target={np.degrees(phi_target):.1f}°")
                    phase = "rotate"

            elif phase == "rotate":
                # 当前两指连线角 & 目标误差（考虑 π 周期）
                phi_cur = finger_line_angle(model, data, bid_left, bid_right)
                err_phi = angle_diff_mod_pi(phi_target, phi_cur)

                # 1) 首次进入：自动标定 yaw 对 phi 的符号映射
                if not rot_calibrated:
                    yaw_now = float(data.qpos[qadr_yaw])

                    # 试探：把目标设到 yaw_now + 小增量，看看 phi_cur 往哪变
                    data.ctrl[aid_rot] = float(np.clip(yaw_now + ROT_TEST_DELTA, loR, hiR))
                    wait_steps(model, data, 6)   # 给点时间响应
                    phi_after = finger_line_angle(model, data, bid_left, bid_right)

                    dphi = angle_diff_mod_pi(phi_after, phi_cur)  # 变化（mod π）
                    rot_sign = +1.0 if dphi >= 0.0 else -1.0

                    # 恢复到原始 yaw 目标（防止试探残留）
                    data.ctrl[aid_rot] = float(np.clip(yaw_now, loR, hiR))
                    wait_steps(model, data, 2)

                    rot_calibrated = True
                    # 重新计算一次当前误差（更干净）
                    phi_cur = finger_line_angle(model, data, bid_left, bid_right)
                    err_phi = angle_diff_mod_pi(phi_target, phi_cur)

                # 2) 正常控制：把“相对角”加到当前 yaw 上 → 绝对目标
                yaw_cur = float(data.qpos[qadr_yaw])
                yaw_des = wrap_to_pi(yaw_cur + rot_sign * err_phi)
                data.ctrl[aid_rot] = float(np.clip(yaw_des, loR, hiR))
                wait_steps(model, data, ROT_SETTLE_STEPS)

                # 3) 进度评估 & 兜底
                phi_new = finger_line_angle(model, data, bid_left, bid_right)
                err_new = abs(angle_diff_mod_pi(phi_target, phi_new))

                if prev_err_phi is None:
                    prev_err_phi = err_new
                else:
                    # 若误差基本不降（或变大），累计卡死计数
                    if err_new > prev_err_phi - 1e-3:
                        rot_stall_count += 1
                    else:
                        rot_stall_count = 0
                    prev_err_phi = err_new

                # 兜底：多次“无改进”直接写 qpos，强制对齐（一次性）
                if rot_stall_count >= ROT_STALL_LIMIT:
                    hard_yaw = wrap_to_pi(yaw_cur + rot_sign * err_new)
                    data.qpos[qadr_yaw] = hard_yaw
                    data.qvel[qadr_yaw] = 0.0
                    mujoco.mj_forward(model, data)
                    rot_stall_count = 0
                    prev_err_phi = None  # 重置评估

                # 终止条件：以两指连线角为准（mod π）
                if err_new < ROT_TOL_FINGER:
                    print(f"[ROTATE] 完成! phi_target={np.degrees(phi_target):.1f}°, "
                          f"phi_cur={np.degrees(phi_new):.1f}°, yaw={np.degrees(float(data.qpos[qadr_yaw])):.1f}°")
                    # 清理旋转用状态，防止下次复用旧值
                    rot_calibrated = False
                    prev_err_phi = None
                    rot_stall_count = 0
                    phase = "descend"


            elif phase == "descend":
                z_target_bottom = target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009)
                z_goal_palm = z_target_bottom + DOWN_SAFE_GAP + palm2tip
                err = float(palm[2]) - z_goal_palm

                if err <= 0.001:
                    print(f"[DESCEND] 已达到目标高度")
                    phase = "close"
                else:
                    step = min(DOWN_STEP, max(0.0, err) * 0.5)
                    lift_cmd = float(np.clip(lift_cmd - step, loZ, hiZ))
                    # 你原脚本这里有 +0.19 的偏置，如果这是标定出来的就保留
                    data.ctrl[aid_lift] = lift_cmd + 0.19

                wait_steps(model, data, 2)

                if (hitL or hitR) or err_tip <= 0.001:
                    phase = "close"
                    print(f"[DESCEND] 检测到接触")

                # 靠近时预收
                if err < 0.015 and not (hitL and hitR):
                    data.ctrl[aid_left]  = min(CLOSE_CMD, data.ctrl[aid_left]  + 0.1)
                    data.ctrl[aid_right] = min(CLOSE_CMD, data.ctrl[aid_right] + 0.1)

            elif phase == "close":
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
                        phase = "lift"
                        print(f"[CLOSE] 抓取成功，力: {leftF+rightF:.2f}N")

            elif phase == "lift":
                # 保持夹紧
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD

                loZ, hiZ = model.actuator_ctrlrange[aid_lift]
                z_goal   = float(np.clip(LIFT_UP_VALUE, loZ, hiZ))
                cur = float(data.ctrl[aid_lift])
                err = z_goal - cur
                step = min(0.003, abs(err) * 0.5)
                new_val = cur + np.sign(err) * step
                data.ctrl[aid_lift] = float(np.clip(new_val, loZ, hiZ))
                wait_steps(model, data, 2)

            # 前进一步并刷新
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
