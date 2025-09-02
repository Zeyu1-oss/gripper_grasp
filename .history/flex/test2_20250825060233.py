import argparse, time, numpy as np, mujoco, mujoco.viewer

# ---------------- 配置 ----------------
DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_09_geom"

# 对齐 & 动作步长
XY_ALIGN_TOL   = 0.004         # XY 对齐阈值 (m)
STEP_XY_SETTLE = 10            # XY 每次设置后等待步数
DOWN_STEP      = 0.003         # 单步最大下降 (m)
PRINT_PERIOD   = 0.15          # 打印节流 (s)

ROT_TOL        = 0.01          # 旋转容差 (rad)
# 初始与抬起高度
LIFT_SAFE      = 0.25          # 初始抬升到的 lift 目标
LIFT_UP_VALUE  = 0.30          # 抓到后抬升到的 lift 目标

# 夹爪：正=加紧，负=张开
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

def determine_lego_orientation(model, data, gid_target):
    """判断乐高积木的姿态：站着(vertical)或躺着(horizontal)"""
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]
    
    # 找到最长轴
    longest_axis = np.argmax(size)
    longest_dir = R[:, longest_axis]
    
    # 检查最长轴的垂直分量
    vertical_component = abs(longest_dir[2])
    
    if vertical_component > VERTICAL_THRESHOLD:
        return "vertical", longest_axis
    else:
        return "horizontal", longest_axis

def compute_grasp_yaw_adaptive(model, data, gid_target, orientation, longest_axis):
    """自适应计算抓取角度"""
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]
    
    if orientation == "vertical":
        # 竖立状态：垂直于最长轴抓取
        long_dir_world = R[:, longest_axis]
        dir_xy = np.array([long_dir_world[0], long_dir_world[1]])
        if np.linalg.norm(dir_xy) < 1e-6:
            return 0.0
        dir_xy /= np.linalg.norm(dir_xy)
        angle = np.arctan2(dir_xy[1], dir_xy[0])
        return angle + np.pi/2  # 垂直于最长轴
        
    else:
        # 躺倒状态：选择最稳定的抓取方向
        # 优先选择与地面平行且较长的轴
        best_score = -1
        best_angle = 0.0
        
        for axis in range(3):
            if axis == longest_axis:
                continue  # 跳过最长轴（通常与地面平行）
                
            axis_dir = R[:, axis]
            # 计算水平程度（Z分量越小越水平）
            horizontalness = 1 - abs(axis_dir[2])
            score = size[axis] * horizontalness
            
            if score > best_score:
                best_score = score
                best_axis = axis
        
        # 计算最佳抓取方向（垂直于选定的轴）
        best_dir = R[:, best_axis]
        dir_xy = np.array([best_dir[0], best_dir[1]])
        if np.linalg.norm(dir_xy) < 1e-6:
            return 0.0
        dir_xy /= np.linalg.norm(dir_xy)
        angle = np.arctan2(dir_xy[1], dir_xy[0])
        return angle + np.pi/2

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

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

    phase = "align_xy"
    last_print = 0.0
    lift_cmd = float(data.ctrl[aid_lift])
    desired_yaw = 0.0
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
            yaw_cur = float(data.qpos[qadr_yaw])

            # 打印信息
            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                print(f"{phase} orient={lego_orientation} desire_yaw={np.degrees(desired_yaw):.1f}° "
                      f"yaw_cur={np.degrees(yaw_cur):.1f}° err={np.degrees(wrap_to_pi(desired_yaw - yaw_cur)):.1f}°")
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
                    # 判断乐高姿态并计算抓取角度
                    lego_orientation, longest_axis = determine_lego_orientation(model, data, gid_tgt)
                    desired_yaw = compute_grasp_yaw_adaptive(model, data, gid_tgt, lego_orientation, longest_axis)
                    print(f"[ALIGN] 乐高姿态: {lego_orientation}, 最长轴: {longest_axis}, 抓取角度: {np.degrees(desired_yaw):.1f}°")
                    phase = "rotate"

            elif phase == "rotate":
                err = wrap_to_pi(desired_yaw - yaw_cur)

                # 直接用 P 控制器推过去
                k_rot = 0.8
                data.ctrl[aid_rot] = float(np.clip(yaw_cur + k_rot * err, loR, hiR))

                wait_steps(model, data, 2)

                if abs(err) < ROT_TOL:
                    print(f"[ROTATE] OK! 目标={np.degrees(desired_yaw):.1f}°, 当前={np.degrees(yaw_cur):.1f}°")
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
                data.ctrl[aid_left]  = min(CLOSE_CMD, data.ctrl[aid_left]  - 0.6)
                data.ctrl[aid_right] = min(CLOSE_CMD, data.ctrl[aid_right] - 0.6)
                
                loZ, hiZ = model.actuator_ctrlrange[aid_lift]
                z_goal   = float(np.clip(LIFT_UP_VALUE, loZ, hiZ))
                cur = float(data.ctrl[aid_lift])
                err = z_goal - cur
                step = min(0.003, abs(err) * 0.5)
                new_val = cur + np.sign(err) * step
                data.ctrl[aid_lift] = float(np.clip(new_val, loZ, hiZ))
                
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                wait_steps(model, data, 2)

            # 前进一步并刷新
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()