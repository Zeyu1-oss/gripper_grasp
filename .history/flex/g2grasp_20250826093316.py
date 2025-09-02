import argparse, time, numpy as np, mujoco, mujoco.viewer

# ---------------- 配置 ----------------
DEFAULT_XML    = "g2_with_100lego.xml"
DEFAULT_TARGET = "lego_23_geom"

XY_ALIGN_TOL   = 0.001
STEP_XY_SETTLE = 10
DOWN_STEP      = 0.003
PRINT_PERIOD   = 0.15

ROT_STEP_MAX   = 0.06   # 单步最大旋转
ROT_TOL        = 0.02   # 收敛阈值 rad (~1.1°)

LIFT_SAFE      = 0.25
LIFT_UP_VALUE  = 0.30

OPEN_CMD       = 0.8
CLOSE_CMD      = -0.8

# 沉降（预滚）判据
SETTLE_SECONDS       = 5.0
SETTLE_VEL_THRESH    = 0.02
SETTLE_STABLE_STEPS  = 200

CONTACT_TH      = 0.4
BOTH_CONTACT_TH = 0.5
DOWN_SAFE_GAP   = 0.0015

VERTICAL_THRESHOLD = 0.7   # 判断竖立的 z 分量阈值

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
        if (b1 == bid_left and b2 == bid_target) or (b2 == bid_left and b1 == bid_target):
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
        if model.geom_bodyid[gid] != bid: continue
        if model.geom_conaffinity[gid] == 0: continue
        # 注意：mesh/sdf 可能没 size[2]，此处只在 box/capsule/cylinder 等有用
        half_z = float(model.geom_size[gid][2]) if model.geom_size.shape[1] >= 3 else 0.0
        zmin = min(zmin, float(data.geom_xpos[gid][2]) - half_z)
    return zmin

def target_halfz_bottom_top(model, data, gid_tgt, fallback_halfz=0.009):
    cz = float(data.geom_xpos[gid_tgt][2])
    half_z = fallback_halfz
    if model.geom_size.shape[1] >= 3:
        hz = float(model.geom_size[gid_tgt][2])
        if hz > 1e-6:
            half_z = hz
    bottom = cz - half_z
    top    = cz + half_z
    return half_z, bottom, top

def target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009):
    half_z, bottom, _ = target_halfz_bottom_top(model, data, gid_tgt, fallback_halfz)
    return bottom

def wrap_to_pi(a): return (a + np.pi) % (2*np.pi) - np.pi
def angle_diff_mod_pi(target, current):
    e = wrap_to_pi(target - current)
    if e >  np.pi/2: e -= np.pi
    if e < -np.pi/2: e += np.pi
    return e

def determine_orientation(model, data, gid_tgt):
    R = data.geom_xmat[gid_tgt].reshape(3,3)
    size = model.geom_size[gid_tgt]
    longest_axis = int(np.argmax(size))
    long_dir = R[:, longest_axis]
    if abs(long_dir[2]) > VERTICAL_THRESHOLD:
        return "vertical", longest_axis
    else:
        return "horizontal", longest_axis

def compute_grasp_yaw(model, data, gid_tgt, state, longest_axis):
    R = data.geom_xmat[gid_tgt].reshape(3, 3)
    x, y, z = R[:, 0], R[:, 1], R[:, 2]

    if state == "vertical":
        yaw = np.arctan2(x[1], x[0])
        palm_y = np.array([-np.sin(yaw), np.cos(yaw)])
        if np.dot(y[:2], palm_y) < 0:
            yaw = wrap_to_pi(yaw + np.pi)
        return yaw
    else:  # horizontal
        v = z[:2]
        n = np.linalg.norm(v)
        if n < 1e-9:
            return 0.0
        v /= n
        yaw = np.arctan2(v[1], v[0]) - np.pi / 2
        palm_y = np.array([-np.sin(yaw), np.cos(yaw)])
        if np.dot(v, palm_y) < 0:
            yaw = wrap_to_pi(yaw + np.pi)
        return yaw

# ---------------- 主程序 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # actuators
    aid_x     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_left  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    aid_right = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")
    aid_rot   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
    jid_yaw   = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
    qadr_yaw  = model.jnt_qposadr[jid_yaw]
    jid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift")
    qadr_lift = model.jnt_qposadr[jid_lift]

    # bodies
    bid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    bid_left  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")
    gid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target)
    bid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.target.replace("_geom",""))

    # init
    loZ, hiZ = model.actuator_ctrlrange[aid_lift]
    loR, hiR = model.actuator_ctrlrange[aid_rot]
    data.ctrl[aid_left] = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift] = LIFT_SAFE
    data.ctrl[aid_rot]  = 0.0
    wait_steps(model, data, 50)

    palm2tip = data.xpos[bid_palm][2] - body_bottom_z(model, data, bid_left)

    phase = "settle"          # 先沉降
    state = "unknown"
    desired_yaw = 0.0
    lift_cmd = float(data.ctrl[aid_lift])
    last_print = 0.0

    # 沉降计时 / 计步
    settle_deadline = time.time() + SETTLE_SECONDS
    settle_stable_cnt = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            palm   = data.xpos[bid_palm]
            target = data.geom_xpos[gid_tgt]
            yaw_cur = float(data.qpos[qadr_yaw])

            now = time.time()

            # ------- 分阶段打印 -------
            if now - last_print >= PRINT_PERIOD:
                if phase == "settle":
                    qv = np.linalg.norm(data.qvel)
                    print(f"[SETTLE] 速度范数={qv:.4f}, 连续稳定步={settle_stable_cnt}, "
                          f"剩余时间={max(0.0, settle_deadline-now):.2f}s")
                elif phase == "align_xy":
                    dx, dy = palm[0]-target[0], palm[1]-target[1]
                    print(f"[ALIGN_XY] 目标({target[0]:.3f},{target[1]:.3f})  "
                          f"手掌({palm[0]:.3f},{palm[1]:.3f})  "
                          f"误差: dx={dx*1000:.1f}mm, dy={dy*1000:.1f}mm  "
                          f"阈值={XY_ALIGN_TOL*1000:.1f}mm")
                elif phase == "rotate":
                    err = wrap_to_pi(desired_yaw - yaw_cur)
                    print(f"[ROTATE] des={np.degrees(desired_yaw):.1f}°, "
                          f"cur={np.degrees(yaw_cur):.1f}°, "
                          f"err={np.degrees(err):.1f}°, "
                          f"ctrl={np.degrees(data.ctrl[aid_rot]):.1f}°")
                elif phase == "descend":
                    halfz, lego_bot, lego_top = target_halfz_bottom_top(model, data, gid_tgt)
                    palm_tip_z = body_bottom_z(model, data, bid_left)
                    z_goal = lego_bot + DOWN_SAFE_GAP + palm2tip
                    dz = palm[2] - z_goal
                    print(f"[DESCEND] LEGO: bottom={lego_bot:.4f} top={lego_top:.4f} "
                          f"thick={2*halfz*1000:.1f}mm | "
                          f"PalmZ={palm[2]:.4f} TipZ={palm_tip_z:.4f} | "
                          f"z_goal={z_goal:.4f} dz={dz*1000:.1f}mm")
                elif phase == "close":
                    lF, rF, hitL, hitR = forces_fingers_vs_target_by_body(
                        model, data, bid_left, bid_right, bid_tgt)
                    print(f"[CLOSE] L={lF:.3f}N R={rF:.3f}N SUM={(lF+rF):.3f}N "
                          f"hitL={hitL} hitR={hitR}  阈值={BOTH_CONTACT_TH:.2f}N")
                elif phase == "lift":
                    lift_q = float(data.qpos[qadr_lift])
                    palm_tip_z = body_bottom_z(model, data, bid_left)
                    print(f"[LIFT] lift_q={lift_q:.4f} 目标={LIFT_UP_VALUE:.3f} "
                          f"PalmTipZ={palm_tip_z:.4f}")
                last_print = now
            # ------------------------

            # ------- 各阶段控制 -------
            if phase == "settle":
                data.ctrl[aid_left]  = OPEN_CMD
                data.ctrl[aid_right] = OPEN_CMD
                data.ctrl[aid_lift]  = LIFT_SAFE

                if np.linalg.norm(data.qvel) < SETTLE_VEL_THRESH:
                    settle_stable_cnt += 1
                else:
                    settle_stable_cnt = 0

                if (now >= settle_deadline) or (settle_stable_cnt >= SETTLE_STABLE_STEPS):
                    print("[SETTLE] 完成，开始对齐 XY。")
                    phase = "align_xy"

                mujoco.mj_step(model, data)
                viewer.sync()
                continue  # 本轮只沉降，不做后续逻辑

            if phase == "align_xy":
                data.ctrl[aid_x] = target[0]
                data.ctrl[aid_y] = target[1]
                wait_steps(model, data, STEP_XY_SETTLE)
                if abs(palm[0]-target[0]) < XY_ALIGN_TOL and abs(palm[1]-target[1]) < XY_ALIGN_TOL:
                    state, longest_axis = determine_orientation(model, data, gid_tgt)
                    desired_yaw = compute_grasp_yaw(model, data, gid_tgt, state, longest_axis)
                    phase = "rotate"

            elif phase == "rotate":
                err = wrap_to_pi(desired_yaw - yaw_cur)
                data.ctrl[aid_rot] = float(np.clip(desired_yaw, loR, hiR))
                # 观察几帧的变化
                for i in range(5):
                    mujoco.mj_step(model, data)
                yaw_cur = float(data.qpos[qadr_yaw])
                if abs(wrap_to_pi(desired_yaw - yaw_cur)) < ROT_TOL:
                    phase = "descend"

            elif phase == "descend":
                # 计算目标高度并下压
                lego_bot = target_bottom_z(model, data, gid_tgt)
                z_goal = lego_bot + DOWN_SAFE_GAP + palm2tip
                z_err = palm[2] - z_goal
                if z_err <= 0.005:
                    phase = "close"
                else:
                    step = min(DOWN_STEP, max(0.0, z_err)*0.5)
                    lift_cmd = np.clip(lift_cmd - step, loZ, hiZ)
                    # data.ctrl[aid_lift] = lift_cmd + 0.194
                    data.ctrl[aid_lift] = lift_cmd
                wait_steps(model, data, 2)

            elif phase == "close":
                lF, rF, hitL, hitR = forces_fingers_vs_target_by_body(
                    model, data, bid_left, bid_right, bid_tgt)
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                wait_steps(model, data, 5)
                if hitL and hitR and (lF + rF) > BOTH_CONTACT_TH:
                    phase = "lift"

            elif phase == "lift":
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                cur = float(data.ctrl[aid_lift])
                step = min(0.003, abs(LIFT_UP_VALUE - cur) * 0.5)
                data.ctrl[aid_lift] = np.clip(cur + step, loZ, hiZ)
                wait_steps(model, data, 2)

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
