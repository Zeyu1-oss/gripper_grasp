import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ---------------- 工具 ----------------
def mat2eulerZYX(Rmat):
    # 正交化，避免数值误差
    U, _, Vt = np.linalg.svd(Rmat)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        Rm *= -1
    # 这里返回 roll, pitch, yaw （与 XML: roll(x), pitch(y), yaw(z) 对应）
    return R.from_matrix(Rm).as_euler('zyx', degrees=False)[::-1]

def aid(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0:
        raise RuntimeError(f"找不到 actuator: {name}")
    return i

def sid(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if i < 0:
        raise RuntimeError(f"找不到 site: {name}")
    return i

def tid(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    if i < 0:
        raise RuntimeError(f"找不到 tendon: {name}")
    return i

def set_gravity(model, enable):
    model.opt.gravity[:] = [0, 0, -9.81] if enable else [0, 0, 0]

def clamp_ctrl(model, aid_idx, val):
    lo, hi = model.actuator_ctrlrange[aid_idx]
    return float(np.clip(val, lo, hi))

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    xml_path  = "../g2_eval.xml"
    pose_path = "../results/6d/grasp_poses.npy"
    grasp_idx = 99            # 选用哪个抓取姿态
    open_w    = 0.03          # 张开(合计) 3 cm，对应 ctrlrange="0 0.03"
    close_w   = 0.00          # 闭合到 0
    settle_s  = 0.5           # 夹爪到位后静置 0.5 s（无重力）
    close_s   = 1.0           # 夹爪渐进夹紧时长（全程可视化）
    hold_s    = 2.0           # 开重力后观察时长
    succ_mm   = 5.0           # 成功阈值（LEGO 位移 < 5 mm）

    # 载入模型与数据
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    # 拿 actuator id
    aid_x     = aid(model, "x_actuator")
    aid_y     = aid(model, "y_actuator")
    aid_z     = aid(model, "z_actuator")
    aid_roll  = aid(model, "roll_actuator")
    aid_pitch = aid(model, "pitch_actuator")
    aid_yaw   = aid(model, "yaw_actuator")
    aid_open  = aid(model, "gripper_opening")

    # 拿 tendon id（用于读当前开口长度）
    tid_grasp = tid(model, "grasp")

    # LEGO 中心 site
    sid_lego = sid(model, "lego_center")

    # 读取抓取 6D
    poses = np.load(pose_path, allow_pickle=True)
    pose  = poses[grasp_idx]
    center = np.asarray(pose["center"], dtype=float)
    Rmat   = np.asarray(pose["rotation"], dtype=float)
    roll, pitch, yaw = mat2eulerZYX(Rmat)

    print("===== 6D 抓取姿态信息 =====")
    print(f"Center (xyz): {center}")
    print("Rotation (3x3):\n", Rmat)
    print(f"RPY (rad)   : {roll:.3f}, {pitch:.3f}, {yaw:.3f}")

    # 初始化：无重力
    mujoco.mj_resetData(model, data)
    set_gravity(model, enable=False)
    mujoco.mj_forward(model, data)

    # 设定 6D 目标（给 position 执行器 ctrl）
    data.ctrl[aid_x]     = center[0]
    data.ctrl[aid_y]     = center[1]
    data.ctrl[aid_z]     = center[2]
    data.ctrl[aid_roll]  = roll
    data.ctrl[aid_pitch] = pitch
    data.ctrl[aid_yaw]   = yaw

    # 设定初始开口（张开）
    data.ctrl[aid_open] = clamp_ctrl(model, aid_open, open_w)

    # ---------- 启动可视化 ----------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # A) 到位后静置（无重力）
        target_t = data.time + max(0.0, settle_s)
        while viewer.is_running() and data.time < target_t:
            # 保持 6D 目标不变（每步都喂 ctrl，避免被扰动）
            data.ctrl[aid_x]     = center[0]
            data.ctrl[aid_y]     = center[1]
            data.ctrl[aid_z]     = center[2]
            data.ctrl[aid_roll]  = roll
            data.ctrl[aid_pitch] = pitch
            data.ctrl[aid_yaw]   = yaw
            data.ctrl[aid_open]  = clamp_ctrl(model, aid_open, open_w)

            mujoco.mj_step(model, data)
            viewer.sync()

        # B) 渐进夹紧（全程可视化）
        steps_close = max(1, int(close_s / dt))
        print("\n开始渐进夹紧 …")
        for k in range(steps_close):
            alpha = (k + 1) / steps_close
            w = (1.0 - alpha) * open_w + alpha * close_w
            data.ctrl[aid_open] = clamp_ctrl(model, aid_open, w)

            # 6D 目标保持
            data.ctrl[aid_x]     = center[0]
            data.ctrl[aid_y]     = center[1]
            data.ctrl[aid_z]     = center[2]
            data.ctrl[aid_roll]  = roll
            data.ctrl[aid_pitch] = pitch
            data.ctrl[aid_yaw]   = yaw

            mujoco.mj_step(model, data)

            # 打印当前开口（读 tendon 长度）
            if k % max(1, steps_close // 20) == 0:
                mujoco.mj_forward(model, data)
                opening_now = float(data.ten_length[tid_grasp])
                print(f"  step {k:4d}/{steps_close}: opening={opening_now*1000:.1f} mm")

            viewer.sync()

        # C) 开启重力，统计位移
        set_gravity(model, enable=True)
        mujoco.mj_forward(model, data)
        p0 = data.site_xpos[sid_lego].copy()
        t_end = data.time + hold_s

        print("\n重力已开启，观察 LEGO 是否掉落 …")
        while viewer.is_running() and data.time < t_end:
            # 夹爪继续保持闭合 + 6D 位姿固定
            data.ctrl[aid_open]  = clamp_ctrl(model, aid_open, close_w)
            data.ctrl[aid_x]     = center[0]
            data.ctrl[aid_y]     = center[1]
            data.ctrl[aid_z]     = center[2]
            data.ctrl[aid_roll]  = roll
            data.ctrl[aid_pitch] = pitch
            data.ctrl[aid_yaw]   = yaw

            mujoco.mj_step(model, data)
            viewer.sync()

        p1 = data.site_xpos[sid_lego].copy()
        disp = float(np.linalg.norm(p1 - p0))
        print("\n===== 结果 =====")
        print(f"LEGO 位移: {disp*1000:.2f} mm  (阈值 {succ_mm:.1f} mm)")
        if disp < succ_mm * 1e-3:
            print("✅ 抓取成功：LEGO 未显著移动")
        else:
            print("❌ 抓取失败：LEGO 移动过大/掉落")
