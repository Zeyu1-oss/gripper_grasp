#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, json, time, math, argparse, traceback
import numpy as np
import multiprocessing as mp
import os , sys 
if ('--egl' in sys.argv) and ('MUJOCOGL' not in os.environ):
    os.environ['MUJOCO_GL']='egl'
import mujoco

# -------------------- 可调参数（也可通过命令行） --------------------
SETTLE_STEPS        = 300         # 初始沉降步数
XY_ALIGN_TOL        = 0.001       # xy 对齐阈值 (m)
STEP_XY_SETTLE      = 10          # xy 控制后静置步数
DOWN_STEP           = 0.003       # 每次下压增量 (m)
DOWN_SAFE_GAP       = 0.0015      # 手指尖与目标“底面”安全间隙 (m)
ROT_TOL             = 0.02        # 旋转收敛阈值 (rad)
OPEN_CMD            = 0.8
CLOSE_CMD           = -0.6
BOTH_CONTACT_TH     = 0.5         # 合力阈值 (N) 判定“夹住”
APPROACH_DIST       = 0.03        # 沿抓取的接近方向退开这么远作为起始位姿 (m)
LIFT_CLEAR          = 0.08        # 成功后抬到离地高度 (m)
HOLD_TIME           = 0.6         # 抬起后保持时间 (s)
PRINT_EVERY         = 0           # >0 时周期打印调试（每多少步一条）

# -------------------- 小工具 --------------------
def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None

def wait_steps(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)

def wrap_to_pi(a): 
    return (a + np.pi) % (2*np.pi) - np.pi

def body_bottom_z(model, data, bid):
    zmin = +1e9
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid: 
            continue
        if model.geom_conaffinity[gid] == 0: 
            continue
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

def forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_target):
    leftF = rightF = 0.0
    hitL = hitR = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        if (b1 == bid_left and b2 == bid_target) or (b2 == bid_left and b1 == bid_target):
            f6 = np.zeros(6); mujoco.mj_contactForce(model, data, i, f6)
            leftF  += abs(f6[0]); hitL = True
        if (b1 == bid_right and b2 == bid_target) or (b2 == bid_right and b1 == bid_target):
            f6 = np.zeros(6); mujoco.mj_contactForce(model, data, i, f6)
            rightF += abs(f6[0]); hitR = True
    return leftF, rightF, hitL, hitR

def yaw_from_R(R):
    """从旋转矩阵取平面 yaw（假设 gripper-x 是朝前轴）"""
    return math.atan2(R[1,0], R[0,0])

def quat_xyzw_to_R(q):
    x, y, z, w = q
    # 归一
    n = math.sqrt(x*x+y*y+z*z+w*w) + 1e-12
    x/=n; y/=n; z/=n; w/=n
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=float)
    return R

# -------------------- 进程内全局（每个 worker 各一份） --------------------
_g = {}

def _worker_init(xml_path, target_geom, headless_env):
    # 每个进程各自设置渲染后端（不显示）
    if headless_env and ("MUJOCO_GL" not in os.environ):
        os.environ["MUJOCO_GL"] = headless_env  # 通常 'egl' 或 'osmesa'
    model = mujoco.MjModel.from_xml_path(xml_path)
    _g["model"] = model
    _g["target_geom"] = target_geom

    # 取各类 id
    _g["aid_x"]     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    _g["aid_y"]     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    _g["aid_lift"]  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    _g["aid_rot"]   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
    _g["aid_L"]     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    _g["aid_R"]     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")
    _g["jid_yaw"]   = name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    "yaw")
    _g["qadr_yaw"]  = model.jnt_qposadr[_g["jid_yaw"]]
    _g["jid_lift"]  = name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    "lift")
    _g["qadr_lift"] = model.jnt_qposadr[_g["jid_lift"]]
    _g["bid_palm"]  = name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "palm")
    _g["bid_left"]  = name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "left_link")
    _g["bid_right"] = name2id(model, mujoco.mjtObj.mjOBJ_BODY,     "right_link")
    _g["gid_tgt"]   = name2id(model, mujoco.mjtObj.mjOBJ_GEOM,     target_geom)
    _g["bid_tgt"]   = name2id(model, mujoco.mjtObj.mjOBJ_BODY,     target_geom.replace("_geom",""))

def _one_trial(args):
    """单次抓取试验：输入 grasp dict，返回结果 dict"""
    idx, grasp, timing = args
    model = _g["model"]
    data  = mujoco.MjData(model)

    # 便捷句柄
    aid_x, aid_y, aid_lift, aid_rot = _g["aid_x"], _g["aid_y"], _g["aid_lift"], _g["aid_rot"]
    aid_L, aid_R = _g["aid_L"], _g["aid_R"]
    qadr_yaw = _g["qadr_yaw"]
    bid_palm, bid_left, bid_right = _g["bid_palm"], _g["bid_left"], _g["bid_right"]
    gid_tgt, bid_tgt = _g["gid_tgt"], _g["bid_tgt"]

    # 初始沉降
    mujoco.mj_forward(model, data)
    wait_steps(model, data, SETTLE_STEPS)

    # 物体世界变换
    mujoco.mj_forward(model, data)
    p_obj = data.geom_xpos[gid_tgt].copy()
    R_obj = data.geom_xmat[gid_tgt].reshape(3,3).copy()

    # 将 grasp 从“网格坐标” -> 世界坐标
    p_g = np.array(grasp["position"], dtype=float)
    R_g = quat_xyzw_to_R(grasp["quaternion_xyzw"])
    p_world = p_obj + R_obj @ p_g
    R_world = R_obj @ R_g

    # 期望 yaw（绕世界Z）
    yaw_des = yaw_from_R(R_world)

    # 计算手指顶到底盘的差（用于算目标z）
    mujoco.mj_forward(model, data)
    palm2tip = data.xpos[bid_palm][2] - body_bottom_z(model, data, bid_left)

    # 起始姿态：沿抓取 z 轴“反向”（approach_dir = -R_world[:,2]）
    # 你之前的 top/side 脚本里 z 是 approach 方向（朝下），这里统一从 -z 方向退 APPROACH_DIST
    approach_dir = -R_world[:,2]
    approach_dir = approach_dir / (np.linalg.norm(approach_dir)+1e-12)
    p_start = p_world - approach_dir * APPROACH_DIST

    # 控制到起始 xy + yaw + 安全高度
    loZ, hiZ = model.actuator_ctrlrange[aid_lift]
    data.ctrl[aid_L] = OPEN_CMD
    data.ctrl[aid_R] = OPEN_CMD
    data.ctrl[aid_x] = float(p_start[0])
    data.ctrl[aid_y] = float(p_start[1])
    data.ctrl[aid_rot] = float(wrap_to_pi(yaw_des))
    # 让手指尖到达 p_start.z
    target_lift = np.clip(float(p_start[2] + palm2tip), loZ, hiZ)
    data.ctrl[aid_lift] = target_lift
    wait_steps(model, data, STEP_XY_SETTLE)

    # 下压到接近物体顶部（以底面+gap 为准）
    _, bot, top = target_halfz_bottom_top(model, data, gid_tgt)
    z_goal = bot + DOWN_SAFE_GAP + palm2tip
    # 慢慢逼近
    fail_reason = None
    for _ in range(400):  # 限最大步数，防暴走
        palm_z = float(data.xpos[bid_palm][2])
        dz = palm_z - z_goal
        if dz <= 0.002:
            break
        step = min(DOWN_STEP, max(0.0, dz)*0.5)
        data.ctrl[aid_lift] = np.clip(float(data.ctrl[aid_lift]) - step, loZ, hiZ)
        mujoco.mj_step(model, data)

    # 闭合
    data.ctrl[aid_L] = CLOSE_CMD
    data.ctrl[aid_R] = CLOSE_CMD
    wait_steps(model, data, 10)

    # 检查是否夹住
    lF, rF, hitL, hitR = forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_tgt)
    sumF = lF + rF
    if not (hitL and hitR and sumF > BOTH_CONTACT_TH):
        fail_reason = f"not_grasped: hitL={hitL}, hitR={hitR}, sumF={sumF:.3f}N"

    # 抬起并保持
    # 抬到“地面高度+LIFT_CLEAR”以上
    # 这里假设地面 z=0，如有托盘可把阈值稍增
    lift_target = max(float(data.qpos[_g["qadr_lift"]]) + 0.15, LIFT_CLEAR)
    end_time = time.time() + HOLD_TIME
    ok_hold = True
    max_drop = 0.0

    # 抬
    for _ in range(250):
        cur = float(data.ctrl[aid_lift])
        step = min(0.004, abs(lift_target-cur)*0.5)
        data.ctrl[aid_lift] = np.clip(cur + step, loZ, hiZ)
        mujoco.mj_step(model, data)

    # 保持：监测物块高度是否明显下降(掉落)
    z_min_keep = float(data.geom_xpos[gid_tgt][2])
    while time.time() < end_time:
        mujoco.mj_step(model, data)
        z_now = float(data.geom_xpos[gid_tgt][2])
        drop = z_min_keep - z_now
        if drop < 0.0:
            z_min_keep = z_now
        else:
            max_drop = max(max_drop, drop)
        # 若完全失去接触且高度明显下降，可提前判失败
        lF, rF, hitL, hitR = forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_tgt)
        if not (hitL or hitR) and (z_now < 0.01):
            ok_hold = False
            fail_reason = f"dropped: z={z_now:.3f}, contacts=F"
            break

    success = (fail_reason is None) and ok_hold
    res = {
        "index": idx,
        "success": success,
        "reason": "ok" if success else (fail_reason or "unknown"),
        "sumF": sumF,
        "final_z": float(data.geom_xpos[gid_tgt][2]),
        "max_drop": float(max_drop),
        "yaw_des_deg": float(np.degrees(wrap_to_pi(yaw_des))),
        "grasp_width": float(grasp.get("width", -1.0)),
    }
    return res


# -------------------- 主逻辑 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="MuJoCo 场景 XML含夹爪+LEGO")
    ap.add_argument("--target", required=True, help="目标几何名（如 lego_23_geom")
    ap.add_argument("--grasps", required=True, help="6D 抓取 JSON数组或 {grasps:[]}")
    ap.add_argument("--processes", type=int, default=max(1, mp.cpu_count()//2))
    ap.add_argument("--topk", type=int, default=0, help="只测前 K 个（按 score 排），0=全部")
    ap.add_argument("--egl", action="store_true", help="设置 MUJOCO_GL=egl（无显示用 GPU）")
    ap.add_argument("--out", default="grasp_eval_results.json")
    args = ap.parse_args()

    headless_env = "egl" if args.egl else os.environ.get("MUJOCO_GL", "")

    # 读取 grasps
    with open(args.grasps, "r") as f:
        G = json.load(f)
    grasps = G.get("grasps", G)
    if not isinstance(grasps, list):
        raise ValueError("JSON 格式错误：应为数组或含 'grasps' 的对象")

    # 若有 score，按 score 降序
    if len(grasps) > 0 and "score" in grasps[0]:
        grasps = sorted(grasps, key=lambda g: g["score"], reverse=True)

    if args.topk and args.topk > 0:
        grasps = grasps[:args.topk]

    print(f"[INFO] 待评测抓取数: {len(grasps)}")

    # 并行
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.processes, initializer=_worker_init,
                  initargs=(args.xml, args.target, headless_env)) as pool:
        tasks = [(i, grasps[i], None) for i in range(len(grasps))]
        results = []
        for res in pool.imap_unordered(_one_trial, tasks, chunksize=4):
            results.append(res)
            if len(results) % 10 == 0:
                ok = sum(r["success"] for r in results)
                print(f"[PROGRESS] {len(results)}/{len(grasps)} done, success={ok}")

    # 汇总
    ok = [r for r in results if r["success"]]
    bad = [r for r in results if not r["success"]]
    summary = {
        "xml": args.xml,
        "target": args.target,
        "grasps_file": args.grasps,
        "processes": args.processes,
        "success": len(ok),
        "failed": len(bad),
        "total": len(results),
        "results": sorted(results, key=lambda r: r["index"])
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] success={len(ok)}/{len(results)} → {args.out}")

if __name__ == "__main__":
    main()
