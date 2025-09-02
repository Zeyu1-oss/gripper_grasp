import argparse, time, numpy as np, mujoco, mujoco.viewer

# ---------------- 配置 ----------------
DEFAULT_XML    = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_02_geom"

XY_ALIGN_TOL   = 0.004
STEP_XY_SETTLE = 10
DOWN_STEP      = 0.003
PRINT_PERIOD   = 0.15

ROT_TOL        = 0.05     # ~3°
ROT_MAX_STEP   = 0.02     # 每步最大旋转弧度

LIFT_SAFE      = 0.25
LIFT_UP_VALUE  = 0.30

OPEN_CMD       = 0.4
CLOSE_CMD      = -0.4

CONTACT_TH      = 0.8
BOTH_CONTACT_TH = 0.7
DOWN_SAFE_GAP   = 0.0015
VERTICAL_THRESHOLD = 0.7

# ---------------- 工具函数 ----------------
def name2id(model, objtype, name):
    try: return mujoco.mj_name2id(model, objtype, name)
    except Exception: return None

def wait_steps(model, data, n):
    for _ in range(n): mujoco.mj_step(model, data)

def body_bottom_z(model, data, bid):
    zmin = +1e9
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid: continue
        if model.geom_conaffinity[gid] == 0: continue
        half_z = model.geom_size[gid][2] if model.geom_size.shape[1] >= 3 else 0
        zmin = min(zmin, float(data.geom_xpos[gid][2]) - half_z)
    return zmin

def target_bottom_z(model, data, gid_tgt, fallback_halfz=0.009):
    cz = float(data.geom_xpos[gid_tgt][2])
    hz = model.geom_size[gid_tgt][2] if model.geom_size.shape[1] >= 3 else fallback_halfz
    return cz - (hz if hz > 1e-6 else fallback_halfz)

def wrap_to_pi(a): return (a + np.pi) % (2*np.pi) - np.pi

def clamp_to_ctrlrange(model, aid, val):
    lo, hi = model.actuator_ctrlrange[aid]
    return float(np.clip(val, lo, hi))

# ---------------- 抓取姿态判断 ----------------
def determine_lego_orientation(model, data, gid_target):
    """返回姿态类型 + 最长轴索引"""
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]
    longest_axis = int(np.argmax(size))
    vertical_component = abs(R[2, longest_axis])
    return ("vertical" if vertical_component > VERTICAL_THRESHOLD else "horizontal", longest_axis)

def compute_desired_yaw(model, data, gid_target, orientation, longest_axis):
    """根据姿态计算 desired_yaw"""
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]

    if orientation == "vertical":
        # 站立：最长轴≈Z，取另一个XY方向为夹持方向
        cand = [i for i in range(3) if i != longest_axis]
        best = max(cand, key=lambda i: size[i])
        axis_dir = R[:, best]
    else:
        # 躺平：取在XY投影最大的方向为夹爪方向
        scores = [size[i] * np.linalg.norm([R[0, i], R[1, i]]) for i in range(3)]
        best = int(np.argmax(scores))
        axis_dir = R[:, best]

    dir_xy = np.array([axis_dir[0], axis_dir[1]])
    if np.linalg.norm(dir_xy) < 1e-6: return 0.0
    dir_xy /= np.linalg.norm(dir_xy)
    return float(np.arctan2(dir_xy[1], dir_xy[0]) + np.pi/2)  # 垂直方向

# ---------------- 主逻辑 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # actuator / joint ids
    aid_x     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_left  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    aid_right = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")
    aid_rot   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
    jid_yaw   = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
    qadr_yaw  = model.jnt_qposadr[jid_yaw]

    bid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    gid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target)
    bid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.target.replace("_geom",""))

    # 初始状态
    data.ctrl[aid_left]  = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift]  = clamp_to_ctrlrange(model, aid_lift, LIFT_SAFE)
    data.ctrl[aid_rot]   = clamp_to_ctrlrange(model, aid_rot, 0.0)
    wait_steps(model, data, 50)

    palm0 = data.xpos[bid_palm].copy()
    palm2tip = palm0[2] - body_bottom_z(model, data, bid_palm)

    phase = "align_xy"
    desired_yaw = 0.0
    lego_orientation = "unknown"
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            palm   = data.xpos[bid_palm]
            target = data.geom_xpos[gid_tgt]
            yaw_cur = float(data.qpos[qadr_yaw])

            # 打印
            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                print(f"{phase} orient={lego_orientation} "
                      f"des_yaw={np.degrees(desired_yaw):.1f}° cur_yaw={np.degrees(yaw_cur):.1f}° "
                      f"err={np.degrees(wrap_to_pi(desired_yaw-yaw_cur)):.1f}°")
                last_print = now

            if phase == "align_xy":
                data.ctrl[aid_x] = clamp_to_ctrlrange(model, aid_x, target[0])
                data.ctrl[aid_y] = clamp_to_ctrlrange(model, aid_y, target[1])
                wait_steps(model, data, STEP_XY_SETTLE)

                if abs(palm[0]-target[0])<XY_ALIGN_TOL and abs(palm[1]-target[1])<XY_ALIGN_TOL:
                    lego_orientation, longest_axis = determine_lego_orientation(model, data, gid_tgt)
                    desired_yaw = compute_desired_yaw(model, data, gid_tgt, lego_orientation, longest_axis)
                    phase = "rotate"
                    print(f"[ALIGN] {lego_orientation}, axis={longest_axis}, des_yaw={np.degrees(desired_yaw):.1f}°")

            elif phase == "rotate":
                err = wrap_to_pi(desired_yaw - data.ctrl[aid_rot])
                step = np.clip(err, -ROT_MAX_STEP, ROT_MAX_STEP)
                data.ctrl[aid_rot] = clamp_to_ctrlrange(model, aid_rot, data.ctrl[aid_rot] + step)
                if abs(err) < ROT_TOL:
                    print(f"[ROTATE] OK yaw={np.degrees(yaw_cur):.1f}°")
                    phase = "descend"

            elif phase == "descend":
                # TODO: 添加下降 & 抓取逻辑
                pass

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
