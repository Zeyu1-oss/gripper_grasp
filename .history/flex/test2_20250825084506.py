import argparse, time, numpy as np, mujoco, mujoco.viewer

# ---------------- 配置 ----------------
DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_02_geom"

# 对齐 & 动作步长
XY_ALIGN_TOL   = 0.004         # XY 对齐阈值 (m)
STEP_XY_SETTLE = 10            # XY 每次设置后等待步数
DOWN_STEP      = 0.003         # 单步最大下降 (m)
PRINT_PERIOD   = 0.15          # 打印节流 (s)

# 旋转相关
ROT_TOL_FINGER   = 0.03        # 以“两指连线角度”为准的容差 (rad)，~1.7°
ROT_SETTLE_STEPS = 2           # 每次给定目标后等待步数
ROT_MAX_STEP     = 0.35        # 单次最大角度修正 (rad) 避免大步抖动
ROT_TEST_DELTA   = 0.05        # 符号标定试探角 (rad)
ROT_STALL_LIMIT  = 30          # 若误差长时间无改进，触发兜底

# 初始与抬起高度
LIFT_SAFE      = 0.25          # 初始抬升到的 lift 目标
LIFT_UP_VALUE  = 0.30          # 抓到后抬升到的 lift 目标

# 夹爪：正=加紧，负=张开
OPEN_CMD       = 0.4
CLOSE_CMD      = -0.4

# 接触判据
CONTACT_TH       = 0.8
BOTH_CONTACT_TH  = 0.7

# 与砖底面的安全缝隙
DOWN_SAFE_GAP    = 0.0015

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
    n = np.linalg.norm(vec2)
    if n < 1e-9:
        return 0.0
    return float(np.arctan2(vec2[1], vec2[0]))

def finger_line_angle(model, data, bid_left, bid_right):
    """
    两指连线的 XY 朝向角 phi_finger。
    """
    pL = data.xpos[bid_left]
    pR = data.xpos[bid_right]
    v  = np.array([pR[0]-pL[0], pR[1]-pL[1]])
    return angle_of_xy(v)

# ---------- 目标姿态判断 & 目标抓取方向 ----------
def determine_lego_orientation(model, data, gid_target):
    """
    判断乐高是站着(vertical)还是躺着(horizontal)，并返回最长轴索引 longest_axis。
    """
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
    目标两指连线角 phi_target（世界系 XY）：
      - horizontal（躺平）：两指连线 ⟂ 长轴 ⇒ phi_target = phi_long + 90°
      - vertical（站立）：长轴≈Z，选“非长轴的两个轴”里 更薄&&更水平 的那个轴方向
    """
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]
    axes = [R[:,0], R[:,1], R[:,2]]

    def xy_angle(v):
        vxy = np.array([v[0], v[1]])
        n = np.linalg.norm(vxy)
        if n < 1e-9:
            return 0.0
        return float(np.arctan2(vxy[1], vxy[0]))

    if orientation == "horizontal":
        # 与长轴正交
        long_dir = axes[longest_axis]
        phi_long = xy_angle(long_dir)
        phi_target = phi_long + np.pi/2.0
    else:
        # 站立：从非长轴两个中挑“更薄且更水平”的
        cand = [i for i in range(3) if i != longest_axis]

        def horiz_score(i):
            dir_i = axes[i]
            horiz = np.linalg.norm([dir_i[0], dir_i[1]])   # 越接近1越水平
            thin  = 1.0 / max(size[i], 1e-6)               # 越薄越好
            return horiz * thin

        best_axis = max(cand, key=horiz_score)
        phi_target = xy_angle(axes[best_axis])

    return wrap_to_pi(phi_target)

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
    data.ctrl[aid_rot]   = float(np.clip(0.0, loR, hiR))
    wait_steps(model, data, 80)

    # 标定 palm→指尖偏移
    palm0 = data.xpos[bid_palm].copy()
    finger_bottom0 = body_bottom_z(model, data, bid_left)
    palm2tip = float(palm0[2] - finger_bottom0)
    if not (0.02 <= palm2tip <= 0.30):
        palm2tip = 0.142
    print(f"[calib] palm2tip = {palm2tip:.4f} m")

    # 状态变量
    phase = "align_xy"
    last_print = 0.0
    lift_cmd = float(data.ctrl[aid_lift])

    phi_target = 0.0
    lego_orientation = "unknown"

    # 旋转控制增强
    rot_calibrated = False     # 是否完成符号标定
    rot_sign = +1.0            # yaw 与 phi 的方向映射（自动标定）
    prev_err_phi = None
    rot_stall_count = 0
    yaw_hold = None            # 旋转完成后固持的绝对 yaw

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

            # 日志
            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                phi_cur = finger_line_angle(model, data, bid_left, bid_right)
                # 只在 rotate 阶段显示误差；其余阶段误差=0避免干扰阅读
                phi_err_for_print = wrap_to_pi(phi_target - phi_cur) if phase == "rotate" else 0.0
                print(f"{phase} orient={lego_orientation} "
                      f"phi_target={np.degrees(phi_target):.1f}° phi_cur={np.degrees(phi_cur):.1f}° "
                      f"yaw_cur={np.degrees(yaw_cur):.1f}° err_phi={np.degrees(phi_err_for_print):.1f}°")
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
                    # 判断姿态 & 计算目标两指连线角
                    lego_orientation, longest_axis = determine_lego_orientation(model, data, gid_tgt)
                    phi_target = compute_target_finger_angle(model, data, gid_tgt, lego_orientation, longest_axis)
                    print(f"[ALIGN] 乐高姿态: {lego_orientation}, 最长轴: {longest_axis}, "
                          f"phi_target={np.degrees(phi_target):.1f}°")
                    # 清理旋转状态
                    rot_calibrated = False
                    prev_err_phi = None
                    rot_stall_count = 0
                    yaw_hold = None
                    phase = "rotate"

            elif phase == "rotate":
                # 当前两指连线角 & 完整误差（不取 mod π）
                phi_cur = finger_line_angle(model, data, bid_left, bid_right)
                phi_err_full = wrap_to_pi(phi_target - phi_cur)

                # --- 首次进入：自动标定 yaw 对 phi 的正负映射 ---
                if not rot_calibrated:
                    yaw_now = float(data.qpos[qadr_yaw])

                    # 试探：+δ，看 phi 朝哪变
                    data.ctrl[aid_rot] = float(np.clip(yaw_now + ROT_TEST_DELTA, loR, hiR))
                    wait_steps(model, data, 6)
                    phi_after = finger_line_angle(model, data, bid_left, bid_right)
                    dphi = wrap_to_pi(phi_after - phi_cur)  # 完整差值
                    rot_sign = +1.0 if dphi >= 0.0 else -1.0

                    # 还原
                    data.ctrl[aid_rot] = float(np.clip(yaw_now, loR, hiR))
                    wait_steps(model, data, 2)
                    rot_calibrated = True

                    # 重新测当前
                    phi_cur = finger_line_angle(model, data, bid_left, bid_right)
                    phi_err_full = wrap_to_pi(phi_target - phi_cur)
                    prev_err_phi = abs(phi_err_full)
                    rot_stall_count = 0

                # --- 位置执行器：把“相对误差”转换成绝对目标 ---
                yaw_cur = float(data.qpos[qadr_yaw])
                delta = np.clip(rot_sign * phi_err_full, -ROT_MAX_STEP, ROT_MAX_STEP)
                yaw_des = wrap_to_pi(yaw_cur + delta)
                data.ctrl[aid_rot] = float(np.clip(yaw_des, loR, hiR))
                wait_steps(model, data, ROT_SETTLE_STEPS)

                # 重新评估误差 & 进度
                phi_cur = finger_line_angle(model, data, bid_left, bid_right)
                phi_err_full = wrap_to_pi(phi_target - phi_cur)
                cur_err = abs(phi_err_full)

                if prev_err_phi is not None and cur_err > prev_err_phi - 1e-3:
                    rot_stall_count += 1
                else:
                    rot_stall_count = 0
                prev_err_phi = cur_err

                # 兜底：多次“无改进”直接写 qpos 强对齐一次
                if rot_stall_count >= ROT_STALL_LIMIT:
                    hard_yaw = wrap_to_pi(yaw_cur + rot_sign * phi_err_full)
                    data.qpos[qadr_yaw] = hard_yaw
                    data.qvel[qadr_yaw] = 0.0
                    mujoco.mj_forward(model, data)
                    rot_stall_count = 0
                    prev_err_phi = None

                # 收敛：只以“真目标”判定（非 mod π）
                if cur_err < ROT_TOL_FINGER:
                    yaw_hold = wrap_to_pi(float(data.qpos[qadr_yaw]))   # 固持用
                    print(f"[ROTATE] 完成! phi_target={np.degrees(phi_target):.1f}°, "
                          f"phi_cur={np.degrees(phi_cur):.1f}°, yaw={np.degrees(yaw_hold):.1f}°")
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
                    data.ctrl[aid_lift] = lift_cmd + 0.19   # 若为你的标定值，保留

                # 固持旋转角，防止被扰动拖走
                if yaw_hold is not None:
                    data.ctrl[aid_rot] = float(np.clip(yaw_hold, loR, hiR))

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
                    # 固持旋转角
                    if yaw_hold is not None:
                        data.ctrl[aid_rot] = float(np.clip(yaw_hold, loR, hiR))
                    wait_steps(model, data, 2)
                elif hitL and hitR:
                    # 固持旋转角
                    if yaw_hold is not None:
                        data.ctrl[aid_rot] = float(np.clip(yaw_hold, loR, hiR))
                    wait_steps(model, data, 2)
                    if (leftF + rightF) >= BOTH_CONTACT_TH:
                        phase = "lift"
                        print(f"[CLOSE] 抓取成功，力: {leftF+rightF:.2f}N")

            elif phase == "lift":
                # 保持夹紧 & 固持旋转角
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                if yaw_hold is not None:
                    data.ctrl[aid_rot] = float(np.clip(yaw_hold, loR, hiR))

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
