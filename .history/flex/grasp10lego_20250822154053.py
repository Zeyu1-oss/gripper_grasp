import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

# ---------------- 参数 ----------------
DEFAULT_XML = "gripper1.xml"

XY_ALIGN_TOL      = 0.001      # XY 对齐阈值 (m)
DOWN_SAFE_GAP     = 0.0015     # 指尖底部相对乐高“底面”的安全缝隙 (m)
LIFT_STEP_MAX     = 0.003      # 每步最大 Z 关节改变量 (m)
CLOSE_STEP        = 0.008      # 每步夹爪改变量（带符号）
GRASP_FORCE_TH    = 1.0        # 左右指与目标的法向力阈值 (N)
PRINT_PERIOD      = 0.15       # 日志节流
LIFT_UP_VALUE     = 0.30       # 抓到后抬升到的 lift 目标（关节坐标）


# ---------- 工具函数 ----------
def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None
def compute_grasp_yaw(model, data, gid_target):
    """根据乐高长边方向，计算手应该旋转的yaw角度"""
    R = data.geom_xmat[gid_target].reshape(3, 3)
    size = model.geom_size[gid_target]   # 半边长 (x,y,z)

    axis = np.argmax(size)  # 0=x, 1=y, 2=z
    long_dir_world = R[:, axis]

    dir_xy = np.array([long_dir_world[0], long_dir_world[1]])
    if np.linalg.norm(dir_xy) < 1e-6:
        return 0.0
    dir_xy /= np.linalg.norm(dir_xy)

    angle = np.arctan2(dir_xy[1], dir_xy[0])
    return angle + np.pi/2   # 手指与长边垂直

def contact_normal_force(model, data, i):
    f6 = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, f6)
    return abs(f6[0])

def sum_finger_forces_with_target(model, data, gid_left, gid_right, target_gid):
    leftF = rightF = 0.0
    hitL = hitR = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 == gid_left and g2 == target_gid) or (g2 == gid_left and g1 == target_gid):
            leftF += contact_normal_force(model, data, i); hitL = True
        if (g1 == gid_right and g2 == target_gid) or (g2 == gid_right and g1 == target_gid):
            rightF += contact_normal_force(model, data, i); hitR = True
    return leftF, rightF, hitL, hitR

def finger_bottom_world_z(model, data, gid_finger):
    center_z = float(data.geom_xpos[gid_finger][2])
    half_h   = float(model.geom_size[gid_finger][2])
    return center_z - half_h

def clamp_to_ctrlrange(model, aid, val):
    lo, hi = model.actuator_ctrlrange[aid]
    return float(np.clip(val, lo, hi))

def _finger_gap(model, data, gid_left, gid_right):
    lx = float(data.geom_xpos[gid_left][0])
    rx = float(data.geom_xpos[gid_right][0])
    return abs(rx - lx)

def robust_grasp_close_dir(model, data, aid_grasp, gid_left, gid_right):
    """在当前值附近双向试探，选择‘让指间距变小’的方向"""
    lo, hi = model.actuator_ctrlrange[aid_grasp]
    cur = float(data.ctrl[aid_grasp])
    for _ in range(5): mujoco.mj_step(model, data)

    span = (hi - lo)
    delta = 0.15 * span if span > 1e-6 else 0.05

    # 向正方向
    data.ctrl[aid_grasp] = clamp_to_ctrlrange(model, aid_grasp, cur + delta)
    for _ in range(5): mujoco.mj_step(model, data)
    gap_plus = _finger_gap(model, data, gid_left, gid_right)

    # 向负方向
    data.ctrl[aid_grasp] = clamp_to_ctrlrange(model, aid_grasp, cur - delta)
    for _ in range(5): mujoco.mj_step(model, data)
    gap_minus = _finger_gap(model, data, gid_left, gid_right)

    # 回到初值
    data.ctrl[aid_grasp] = cur
    for _ in range(5): mujoco.mj_step(model, data)

    return +1.0 if gap_plus < gap_minus else -1.0

def wait_until_settled(model, data, bid_target, settle_steps=3000, vel_thresh=5e-3):
    """等待目标积木自由落体稳定"""
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        vel = np.linalg.norm(data.cvel[bid_target])
        if vel < vel_thresh:
            break


# ---------- 单个抓取流程 ----------
def grasp_one_block(model, data, target_name,
                    aid_x, aid_y, aid_lift, aid_grasp,
                    gid_left, gid_right, gid_palm):

    gid_target = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_name)
    bid_target = name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name.replace("_geom",""))
    if gid_target is None or bid_target is None:
        print(f"[ERROR] 找不到 {target_name}")
        return

    print(f"[INFO] 等待 {target_name} 落稳中 ...")
    wait_until_settled(model, data, bid_target)

    # 初始张开
    lo_g, hi_g = model.actuator_ctrlrange[aid_grasp]
    open_g = lo_g + 0.9*(hi_g - lo_g)
    data.ctrl[aid_grasp] = open_g
    data.ctrl[aid_lift]  = clamp_to_ctrlrange(model, aid_lift, 0.25)
    for _ in range(50): mujoco.mj_step(model, data)

    # 判定闭合方向
    close_dir = robust_grasp_close_dir(model, data, aid_grasp, gid_left, gid_right)
    print(f"[INFO] {target_name}: grasp 正方向是{'闭合' if close_dir>0 else '张开'}")

    phase = "align_xy"
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        done = False
        while viewer.is_running() and not done:
            mujoco.mj_forward(model, data)

            palm   = data.geom_xpos[gid_palm]
            target = data.geom_xpos[gid_target]
            z_origin = float(target[2] - model.geom_size[gid_target][2])
            z_goal   = z_origin + DOWN_SAFE_GAP

            # 控制
            data.ctrl[aid_x] = clamp_to_ctrlrange(model, aid_x, float(target[0]))
            data.ctrl[aid_y] = clamp_to_ctrlrange(model, aid_y, float(target[1]))
            finger_bot_z = finger_bottom_world_z(model, data, gid_left)

            # 打印
            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                print(f"[{target_name} | {phase:10s}] palm=({palm[0]:+.3f},{palm[1]:+.3f},{palm[2]:.3f}) "
                      f"z_origin={z_origin:.3f} finger_bottom={finger_bot_z:.3f} "
                      f"ctrl(lift)={data.ctrl[aid_lift]:+.3f} ctrl(g)={data.ctrl[aid_grasp]:.3f}")
                last_print = now

            # 状态机
            if phase == "align_xy":
                if abs(palm[0] - data.ctrl[aid_x]) < XY_ALIGN_TOL and \
                   abs(palm[1] - data.ctrl[aid_y]) < XY_ALIGN_TOL:
                    phase = "descend"

            elif phase == "descend":
                err = z_goal - finger_bot_z
                step = np.clip(err, -LIFT_STEP_MAX, LIFT_STEP_MAX)
                data.ctrl[aid_lift] = clamp_to_ctrlrange(model, aid_lift,
                                                        data.ctrl[aid_lift] + step)
                if abs(err) < 0.008:
                    phase = "close"

            elif phase == "close":
                data.ctrl[aid_grasp] = clamp_to_ctrlrange(
                    model, aid_grasp, data.ctrl[aid_grasp] + close_dir * CLOSE_STEP
                )
                leftF, rightF, hitL, hitR = sum_finger_forces_with_target(
                    model, data, gid_left, gid_right, gid_target
                )
                if (abs(data.ctrl[aid_grasp] - lo_g) < 1e-6 or
                    abs(data.ctrl[aid_grasp] - hi_g) < 1e-6) and (not hitL and not hitR):
                    data.ctrl[aid_lift] = clamp_to_ctrlrange(model, aid_lift,
                                                            data.ctrl[aid_lift] - 0.001)
                if hitL and hitR and leftF >= GRASP_FORCE_TH and rightF >= GRASP_FORCE_TH:
                    phase = "lift"

            elif phase == "lift":
                data.ctrl[aid_lift] = clamp_to_ctrlrange(model, aid_lift, LIFT_UP_VALUE)
                done = True

        while viewer.is_running() and not done:
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()

    print(f"[DONE] 已完成 {target_name} 的抓取")



# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # actuator + geom id
    aid_x     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_grasp = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grasp")
    gid_left  = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger")
    gid_right = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger")
    gid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm")

    # 依次抓 lego_01 到 lego_10
    targets = [f"lego_{i:02d}_geom" for i in range(1, 11)]
    for t in targets:
        grasp_one_block(model, data, t,
                        aid_x, aid_y, aid_lift, aid_grasp,
                        gid_left, gid_right, gid_palm)
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
        

if __name__ == "__main__":
    main()
