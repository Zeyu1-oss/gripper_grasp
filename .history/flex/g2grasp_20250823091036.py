import argparse, time, numpy as np, mujoco, mujoco.viewer

# ---------------- 配置 ----------------
DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_00_geom"

# 对齐 & 动作步长
XY_ALIGN_TOL   = 0.004         # XY 对齐阈值 (m)
STEP_XY_SETTLE = 10            # XY 每次设置后等待步数
DOWN_STEP      = 0.003         # 单步最大下降 (m)
PRINT_PERIOD   = 0.15          # 打印节流 (s)
# 力均衡参数
F_BAL_TOL = 0.2    # 两侧力差小于该阈值认为已均衡 (N)
K_BAL     = 0.6    # 力差到电机指令的增益（越大均衡越快，但更容易抖）

ROT_STEP_MAX   = 0.06  # rad
ROT_TOL        = 0.05  # 
# 初始与抬起高度（注意：lift 是 position 执行器，世界z = 0.5 + qpos(lift)）
LIFT_SAFE      = 0.25          # 初始抬升到的 lift 目标（关节坐标）
LIFT_UP_VALUE  = 0.30          # 抓到后抬升到的 lift 目标（关节坐标）

# 夹爪：正=加紧，负=张开 —— 按你的 XML 约定
OPEN_CMD       = 0.4
CLOSE_CMD      = -0.8
# 接触判据（按需调）
CONTACT_TH      = 0.8
BOTH_CONTACT_TH = 0.7

# 与砖底面的安全缝隙
DOWN_SAFE_GAP   = 0.0015

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
    """取某 body 所有有碰撞功能的 geom 的最底部 z（视觉几何体被剔除）。"""
    zmin = +1e9
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid:
            continue
        # 视觉几何体（conaffinity==0）跳过
        if model.geom_conaffinity[gid] == 0:
            continue
        half_z = 0.0
        # 大部分类型 geom_size[gid] 至少有 3 项；若无则按 0 处理
        if model.geom_size.shape[1] >= 3:
            half_z = float(model.geom_size[gid][2])
        zmin = min(zmin, float(data.geom_xpos[gid][2]) - half_z)
    return zmin

def target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009):
    """估算目标几何“底面”z：center_z - half_z；若 size 不可靠，用回退半高。"""
    cz = float(data.geom_xpos[gid_tgt][2])
    half_z = fallback_halfz
    if model.geom_size.shape[1] >= 3:
        hz = float(model.geom_size[gid_tgt][2])
        if hz > 1e-6:
            half_z = hz
    return cz - half_z
def compute_grasp_yaw(model, data, gid_target):
    """yaw角度"""
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]   # 半边长 (x,y,z)

    axis = np.argmax(size)  # 0=x, 1=y, 2=z
    long_dir_world = R[:, axis]

    dir_xy = np.array([long_dir_world[0], long_dir_world[1]])
    if np.linalg.norm(dir_xy) < 1e-6:
        return 0.0
    dir_xy /= np.linalg.norm(dir_xy)

    angle = np.arctan2(dir_xy[1], dir_xy[0])
    return angle -np.pi/2   # 手指与长边垂直
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
    aid_rot    = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")  # 新增
    jid_yaw = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")             # 新增
    qadr_yaw = model.jnt_qposadr[jid_yaw] if jid_yaw is not None else None

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

    # 提前取一次 lift 的范围，后面分支都会用到
    loZ, hiZ = model.actuator_ctrlrange[aid_lift]
    loR, hiR = model.actuator_ctrlrange[aid_rot]

    # 初始：张开并抬高（注意：OPEN_CMD 是负数，代表张开）
    data.ctrl[aid_left]  = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift]  = float(np.clip(LIFT_SAFE, loZ, hiZ))
    data.ctrl[aid_rot]   = float(np.clip(0.0, loR, hiR))

    wait_steps(model, data, 80)

    # --- 标定 palm→指尖（左指）竖直偏移，用于精确停在积木上方 ---
    palm0 = data.xpos[bid_palm].copy()
    finger_bottom0 = body_bottom_z(model, data, bid_left)
    palm2tip = float(palm0[2] - finger_bottom0)
    # 如果网格尺寸或朝向导致异常，做个合理回退（约 14~16 cm 视你的模型而定）
    if not (0.02 <= palm2tip <= 0.30):
        palm2tip = 0.142
    print(f"[calib] palm2tip = {palm2tip:.4f} m")

    phase = "align_xy"
    last_print = 0.0
    lift_cmd = float(data.ctrl[aid_lift])

    desired_yaw = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(0.02) 
        while viewer.is_running():
            time.sleep(0.02) 
            mujoco.mj_forward(model, data)

            palm   = data.xpos[bid_palm]               # 世界坐标
            target = data.geom_xpos[gid_tgt]
            tip_z  = body_bottom_z(model, data, bid_left)
            tgt_bottom = target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009)
            z_goal_tip = tgt_bottom + DOWN_SAFE_GAP
            err_tip    = tip_z - z_goal_tip

            yaw_cur = float(data.qpos[qadr_yaw])
            # 观察打印
            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                print(f"tip_z={tip_z:.3f}  z_goal_tip={z_goal_tip:.3f}  err_tip={err_tip:.3f}  "
                      f"yaw_cur={yaw_cur:+.3f} yaw_cmd={data.ctrl[aid_rot]:+.3f}  "
                      f"ctrl(x,y,lift)=({data.ctrl[aid_x]:+.3f},{data.ctrl[aid_y]:+.3f},{data.ctrl[aid_lift]:+.3f})")
                last_print = now

            # 接触/力
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
                # 目标底面高度（center z - half_z）
                z_target_bottom = target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009)
                # 让“指尖底部”停在目标底面上方 DOWN_SAFE_GAP
                z_goal_palm = z_target_bottom + DOWN_SAFE_GAP + palm2tip

                # 误差（>0 说明手还高，需要下降）
                err = float(palm[2]) - z_goal_palm

                # 自适应步长，防止越过
                step = min(DOWN_STEP, max(0.0, err) * 0.5)
                lift_cmd = float(np.clip(lift_cmd - step, loZ, hiZ))
                # 再加护栏：绝不低于目标 + 安全缝 - 1mm
                # 世界 z = 0.5 + qpos(lift)，但这里直接用闭环观测 palm[2] 控制，不强行写死偏置

                data.ctrl[aid_lift] = lift_cmd+0.185
                wait_steps(model, data, 2)

                # 进入夹紧：已有接触或高度已经足够近
                if (hitL or hitR) or err_tip <= 0.002:
                    phase = "close"

                # 靠近时轻微预收，便于更快建立接触（可选）
                if err < 0.015 and not (hitL and hitR):
                    data.ctrl[aid_left]  = min(CLOSE_CMD, data.ctrl[aid_left]  + 0.1)
                    data.ctrl[aid_right] = min(CLOSE_CMD, data.ctrl[aid_right] + 0.1)
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
                           phase = "lift"

            elif phase == "lift":
                loZ, hiZ = model.actuator_ctrlrange[aid_lift]
                z_goal   = float(np.clip(LIFT_UP_VALUE, loZ, hiZ))

    # 当前 lift 值
                cur = float(data.ctrl[aid_lift])

    # 误差
                err = z_goal - cur

    # 逐步上升（越接近目标越慢）
                step = min(0.003, abs(err) * 0.5)   # 这里的 0.003 是最大步长，可以调大/调小控制速度
                new_val = cur + np.sign(err) * step
            
                data.ctrl[aid_lift] = float(np.clip(new_val, loZ, hiZ))

    # 保持夹紧
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD

                wait_steps(model, data, 2)

            # 前进一步并刷新
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
