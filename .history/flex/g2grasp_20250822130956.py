import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

# ========= 参数 =========
DEFAULT_XML = "g2_with_30lego.xml"
DEFAULT_TARGET = "lego_02_geom"   # 目标几何名：lego_xx_geom

XY_ALIGN_TOL   = 0.004        # XY 对齐阈值 (m)
DOWN_SAFE_GAP  = 0.0015       # 指尖底面距离乐高“底面”的安全缝隙 (m)
LIFT_STEP_MAX  = 0.003        # 每步 lift 最大改变量
PRINT_PERIOD   = 0.15         # 打印节流(s)
LIFT_UP_VALUE  = 0.30         # 抬升到的关节坐标 (lift ctrl)

# 手指力控（motor）参数
OPEN_CMD       = +0.8         # 张开施加的正向电机指令
CLOSE_CMD      = -0.6         # 闭合施加的负向电机指令
CONTACT_TH     = 0.4          # 单侧接触法向力阈值(N)
BOTH_CONTACT_TH= 0.8          # 双侧接触合力阈值(N)

ROT_RATE       = 0.02         # 每步最大旋转控制步长 (rad)

# ========= 工具 =========
def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None

def contact_normal_force(model, data, i):
    f6 = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, f6)
    # f6[0] 是法向（在接触坐标系）的力，取正
    return abs(f6[0])

def sum_forces_on_target_from_fingers(model, data, gid_left, gid_right, gid_target):
    leftF = rightF = 0.0
    hitL = hitR = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        # 左指与目标
        if (g1 == gid_left and g2 == gid_target) or (g2 == gid_left and g1 == gid_target):
            leftF  += contact_normal_force(model, data, i)
            hitL = True
        # 右指与目标
        if (g1 == gid_right and g2 == gid_target) or (g2 == gid_right and g1 == gid_target):
            rightF += contact_normal_force(model, data, i)
            hitR = True
    return leftF, rightF, hitL, hitR

def finger_bottom_world_z(model, data, gid_finger_box):
    """对 box 几何：底面 z = 几何中心世界 z - 半高"""
    center_z = float(data.geom_xpos[gid_finger_box][2])
    half_h   = float(model.geom_size[gid_finger_box][2])  # 对 box：size = 半边长
    return center_z - half_h

def clamp_to_ctrlrange(model, aid, val):
    lo, hi = model.actuator_ctrlrange[aid]
    return float(np.clip(val, lo, hi))

def compute_grasp_yaw(model, data, gid_target):
    """
    估计乐高长边方向的 yaw（世界系），让手指与长边垂直（夹在短边上）。
    对 mesh/sdf: model.geom_size 是缩放，不是尺寸；我们用旋转矩阵的主轴投影近似。
    """
    R = data.geom_xmat[gid_target].reshape(3,3)
    # 取 x 轴（列 0）或 y 轴（列 1）的投影中更“长”的方向
    # 简化：两者都投影到 XY，选模更大的
    dirx = R[:,0]; diry = R[:,1]
    vx = np.array([dirx[0], dirx[1]]); vy = np.array([diry[0], diry[1]])
    nx = np.linalg.norm(vx); ny = np.linalg.norm(vy)
    if nx < 1e-8 and ny < 1e-8:
        return 0.0
    v = vx if nx >= ny else vy
    v = v / (np.linalg.norm(v)+1e-12)
    yaw = np.arctan2(v[1], v[0])
    # 手指与长边垂直 -> 加 90°
    return yaw + np.pi/2.0

def wait_steps(model, data, n=20):
    for _ in range(n):
        mujoco.mj_step(model, data)

# ========= 主流程 =========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 取 actuator / geom id
    aid_x     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_rot   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
    aid_left  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    aid_right = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")

    gid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm")
    gid_lf    = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger")
    gid_rf    = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger")
    gid_tgt   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target)

    required = [aid_x, aid_y, aid_lift, aid_rot, aid_left, aid_right,
                gid_palm, gid_lf, gid_rf, gid_tgt]
    if any(v is None for v in required):
        print("[ERROR] 名称映射失败：请确认 XML 里存在 actuator x/y/lift/rotation/left_joint/right_joint"
              " 以及 geom palm/left_finger/right_finger 和目标几何名。")
        return

    # 初始：手指张开（motor 力控：正数张开），升到安全高度
    data.ctrl[aid_left]  = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift]  = clamp_to_ctrlrange(model, aid_lift, 0.25)
    wait_steps(model, data, n=60)

    phase = "align_xy"
    last_print = 0.0
    desired_yaw = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)

            palm   = data.geom_xpos[gid_palm]
            target = data.geom_xpos[gid_tgt]

            # 目标z：乐高“底面” + 安全缝隙
            z_origin = float(target[2] - model.geom_size[gid_tgt][2])  # 对 box/sdf: 下底面近似
            z_goal   = z_origin + DOWN_SAFE_GAP

            # 指尖底面 z（取左右中较低者，防抖）
            lf_bot = finger_bottom_world_z(model, data, gid_lf)
            rf_bot = finger_bottom_world_z(model, data, gid_rf)
            finger_bot = min(lf_bot, rf_bot)

            now = time.time()
            if now - last_print >= PRINT_PERIOD:
                print(f"[{phase:9s}] palm=({palm[0]:+.3f},{palm[1]:+.3f},{palm[2]:.3f})  "
                      f"tgt=({target[0]:+.3f},{target[1]:+.3f},{target[2]:.3f})  "
                      f"z_goal={z_goal:.3f} finger_bot={finger_bot:.3f}  "
                      f"ctrl(x,y,lift,rot)=({data.ctrl[aid_x]:+.3f},{data.ctrl[aid_y]:+.3f},"
                      f"{data.ctrl[aid_lift]:+.3f},{data.ctrl[aid_rot]:+.3f})")
                last_print = now

            # ===== 相位机 =====
            if phase == "align_xy":
                # 设目标到 X/Y 执行器（位置型），然后跑几步让它收敛
                data.ctrl[aid_x] = clamp_to_ctrlrange(model, aid_x, float(target[0]))
                data.ctrl[aid_y] = clamp_to_ctrlrange(model, aid_y, float(target[1]))
                wait_steps(model, data, n=10)

                if abs(palm[0] - target[0]) < XY_ALIGN_TOL and \
                   abs(palm[1] - target[1]) < XY_ALIGN_TOL:
                    desired_yaw = compute_grasp_yaw(model, data, gid_tgt)
                    phase = "rotate"

            elif phase == "rotate":
                err = desired_yaw - float(data.ctrl[aid_rot])
                step = np.clip(err, -ROT_RATE, ROT_RATE)
                data.ctrl[aid_rot] = clamp_to_ctrlrange(model, aid_rot, data.ctrl[aid_rot] + step)
                wait_steps(model, data, n=4)
                if abs(err) < 0.05:
                    phase = "descend"

            elif phase == "descend":
                # 逼近：err = 期望底面高度 - 当前指尖底面高度
                err  = z_goal - finger_bot
                step = np.clip(err, -LIFT_STEP_MAX, LIFT_STEP_MAX)
                data.ctrl[aid_lift] = clamp_to_ctrlrange(model, aid_lift, data.ctrl[aid_lift] + step)
                wait_steps(model, data, n=2)

                if abs(err) < 0.004:
                    phase = "close"

            elif phase == "close":
                # motor 力控：持续给负力闭合，直到两侧与目标都建立足够法向力
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                wait_steps(model, data, n=1)

                leftF, rightF, hitL, hitR = sum_forces_on_target_from_fingers(
                    model, data, gid_lf, gid_rf, gid_tgt
                )
                # 触到且有力，再微降一点点让接触更充分
                if not (hitL and hitR):
                    data.ctrl[aid_lift] = clamp_to_ctrlrange(model, aid_lift, data.ctrl[aid_lift] - 0.001)

                if hitL and hitR and (leftF+rightF) >= BOTH_CONTACT_TH and \
                   leftF >= CONTACT_TH and rightF >= CONTACT_TH:
                    phase = "lift"

            elif phase == "lift":
                data.ctrl[aid_lift] = clamp_to_ctrlrange(model, aid_lift, LIFT_UP_VALUE)
                # 保持夹紧
                data.ctrl[aid_left]  = CLOSE_CMD
                data.ctrl[aid_right] = CLOSE_CMD
                wait_steps(model, data, n=2)
                # 这里可以选择退出或进入下一块
                # 在本 demo 里，一直停留展示
            else:
                pass

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
