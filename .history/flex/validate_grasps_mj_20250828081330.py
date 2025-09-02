#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, math, time, argparse
import numpy as np

# 如果加了 --egl 就用 GPU/无头
if "--egl" in os.sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco

# ======== 小工具 ========
def name2id(model, objtype, name):
    try: return mujoco.mj_name2id(model, objtype, name)
    except: return -1

def req_id(i, what):
    if i is None or i < 0:
        raise RuntimeError(f"找不到 {what}")
    return i

def wait_steps(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)

def quat_to_R(qx,qy,qz,qw):
    n = math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)+1e-12
    x,y,z,w = qx/n,qy/n,qz/n,qw/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], float)

def R_to_euler_zyx(R):
    # 返回 (roll_x, pitch_y, yaw_z) 弧度，Z-Y-X 顺序
    sy = -R[2,0]
    sy = min(1.0, max(-1.0, sy))
    pitch = math.asin(sy)
    roll  = math.atan2(R[2,1], R[2,2])
    yaw   = math.atan2(R[1,0], R[0,0])
    return roll, pitch, yaw

def contact_forces_sum(model, data, bodyA, bodyB):
    total = 0.0; hit = False
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]; b2 = model.geom_bodyid[c.geom2]
        if (b1==bodyA and b2==bodyB) or (b1==bodyB and b2==bodyA):
            f6 = np.zeros(6); mujoco.mj_contactForce(model, data, i, f6)
            total += abs(float(f6[0])); hit=True
    return total, hit

# ======== 主程序 ========
def main():
    ap = argparse.ArgumentParser(description="把夹爪移动到给定6D姿态→合拢→上抬→判断LEGO是否被带起")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--target", default="lego_01", help="LEGO 几何名（如 lego_01）")
    # 6D位姿（世界系）
    ap.add_argument("--pos",  nargs=3, type=float, required=True, metavar=("X","Y","Z"))
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--quat", nargs=4, type=float, metavar=("QX","QY","QZ","QW"))
    group.add_argument("--rpy_deg", nargs=3, type=float, metavar=("ROLLdeg","PITCHdeg","YAWdeg"))
    # 夹爪开合 & 上抬
    ap.add_argument("--close", type=float, default=0.02, help="夹爪合拢控制量（你的XML里 gripper_motor 的 ctrl）")
    ap.add_argument("--prewait", type=int, default=300, help="到位后静置步数")
    ap.add_argument("--lift", type=float, default=0.06, help="合拢后上抬高度(m)")
    ap.add_argument("--hold_s", type=float, default=0.6, help="上抬后保持秒数（按仿真时间换算步数）")
    ap.add_argument("--out", default="grasp_once_result.json")
    ap.add_argument("--egl", action="store_true")
    args = ap.parse_args()

    # 读模型
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    # 查 actuator id
    aid_x   = req_id(name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x_actuator"),   "actuator:x_actuator")
    aid_y   = req_id(name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y_actuator"),   "actuator:y_actuator")
    aid_z   = req_id(name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "z_actuator"),   "actuator:z_actuator")
    aid_r   = req_id(name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_actuator"),"actuator:roll_actuator")
    aid_p   = req_id(name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_actuator"),"actuator:pitch_actuator")
    aid_yaw = req_id(name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_actuator"), "actuator:yaw_actuator")
    aid_g   = req_id(name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor"), "actuator:gripper_motor")

    gid_tgt = req_id(name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target), f"geom:{args.target}")
    bid_tgt = model.geom_bodyid[gid_tgt]
    bid_left  = req_id(name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link"), "body:left_link")
    bid_right = req_id(name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link"), "body:right_link")

    mujoco.mj_forward(model, data)

    # 6D姿态：世界 pos + 世界 R(→ZYX欧拉)
    px,py,pz = args.pos
    if args.quat:
        R = quat_to_R(*args.quat)
        roll, pitch, yaw = R_to_euler_zyx(R)
    else:
        roll, pitch, yaw = [math.radians(v) for v in args.rpy_deg]

    # 1) 直接把 6D 设为 position 执行器的目标
    data.ctrl[aid_x]   = float(px)
    data.ctrl[aid_y]   = float(py)
    data.ctrl[aid_z]   = float(pz)
    data.ctrl[aid_r]   = float(roll)
    data.ctrl[aid_p]   = float(pitch)
    data.ctrl[aid_yaw] = float(yaw)

    # 到位静置
    wait_steps(model, data, args.prewait)

    # 记录上抬前 LEGO 高度
    z_before = float(data.geom_xpos[gid_tgt][2])

    # 2) 合拢（注意：你的 XML 用的是 motor 力控；若力度小，可能合不拢）
    data.ctrl[aid_g] = float(args.close)
    wait_steps(model, data, 200)

    # 触力（可选：看看是否真夹到）
    F_L, hitL = contact_forces_sum(model, data, bid_left,  bid_tgt)
    F_R, hitR = contact_forces_sum(model, data, bid_right, bid_tgt)
    F_sum = F_L + F_R

    # 3) 轻轻上抬（最小动作验证是否“掉下去”）
    data.ctrl[aid_z] = float(pz + args.lift)
    # 用仿真步而不是墙钟
    hold_steps = max(1, int(args.hold_s / model.opt.timestep))
    wait_steps(model, data, hold_steps)

    z_after = float(data.geom_xpos[gid_tgt][2])
    dz = z_after - z_before

    # 判定：被带起来则成功（阈值取上抬量的 60%）
    success = (dz > 0.6*args.lift)

    result = {
        "xml": args.xml,
        "target": args.target,
        "pose_world": {
            "position": [px,py,pz],
            "rpy_rad":  [roll, pitch, yaw]
        },
        "gripper_cmd": args.close,
        "lift": args.lift,
        "z_before": z_before,
        "z_after":  z_after,
        "dz": dz,
        "contacts": {
            "left_F":  F_L,
            "right_F": F_R,
            "sum_F":   F_sum,
            "hitL": bool(hitL),
            "hitR": bool(hitR)
        },
        "success": bool(success)
    }

    print(f"\n=== 结果 ===")
    print(f"带起高度 Δz = {dz:.3f} m （阈值 {0.6*args.lift:.3f} m） → {'✅ 成功' if success else '❌ 失败'}")
    print(f"接触力合计 ~ {F_sum:.3f} N   命中: L={hitL} R={hitR}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"已写入 {args.out}")

if __name__ == "__main__":
    main()
