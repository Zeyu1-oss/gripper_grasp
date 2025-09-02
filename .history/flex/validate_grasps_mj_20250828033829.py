#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, argparse
import numpy as np
import mujoco

# =============== 可调参数（保持头less运行，不导入 viewer） ===============
OPEN_CMD          = 0.8
CLOSE_CMD         = -0.6
XY_ALIGN_TOL      = 0.002       # XY 对齐阈值（米）
STEP_XY_SETTLE    = 10
DOWN_SAFE_GAP     = 0.0015      # 手指尖与抓取中心的安全间隙
ROT_TOL           = 0.02        # yaw 收敛阈值（弧度）
SETTLE_SECONDS    = 2.0         # 开始前沉降
SETTLE_VEL_THR    = 0.02
SETTLE_STEPS_OK   = 150
APPROACH_TOL_DEG  = 30          # 只尝试 z轴≈世界 -Z 的 grasps（角度容差）
FORCE_OK_SUM      = 0.5         # 两指合力阈值（N），用于“已夹住”的判定
LIFT_UP_VALUE     = 0.30        # 抬起目标升降的控制值
SUCCESS_LIFT_DZ   = 0.015       # 视为成功的抬高（米）
FALLBACK_HALFZ    = 0.009       # 目标厚度未知时的半厚度估计（米）
MAX_ROT_STEPS     = 200
MAX_DESC_STEPS    = 300
MAX_CLOSE_STEPS   = 200
MAX_LIFT_STEPS    = 220

# ====================== 工具函数 ======================
def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return -1

def wait_steps(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def quat_xyzw_to_R(q):
    x,y,z,w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=float)

def pose_of_geom(model, data, gid):
    T = np.eye(4)
    T[:3,3]  = data.geom_xpos[gid]
    T[:3,:3] = data.geom_xmat[gid].reshape(3,3)
    return T

def body_bottom_z(model, data, bid):
    zmin = +1e9
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid: continue
        if model.geom_conaffinity[gid] == 0: continue
        halfz = float(model.geom_size[gid][2]) if model.geom_size.shape[1] >= 3 else 0.0
        zmin = min(zmin, float(data.geom_xpos[gid][2]) - halfz)
    return zmin

def target_halfz_bottom_top(model, data, gid, fallback_halfz=FALLBACK_HALFZ):
    cz = float(data.geom_xpos[gid][2])
    halfz = fallback_halfz
    if model.geom_size.shape[1] >= 3:
        hz = float(model.geom_size[gid][2])
        if hz > 1e-6:
            halfz = hz
    return halfz, cz - halfz, cz + halfz

def forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_target):
    def contact_normal_force(i):
        f6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f6)
        return abs(f6[0])
    leftF = rightF = 0.0
    hitL = hitR = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1 = model.geom_bodyid[g1]; b2 = model.geom_bodyid[g2]
        if (b1 == bid_left and b2 == bid_target) or (b2 == bid_left and b1 == bid_target):
            leftF  += contact_normal_force(i); hitL = True
        if (b1 == bid_right and b2 == bid_target) or (b2 == bid_right and b1 == bid_target):
            rightF += contact_normal_force(i); hitR = True
    return leftF, rightF, hitL, hitR

def find_topmost_lego_geom(model, data, prefix="lego_", suffix="_geom"):
    best_gid = -1; best_top = -1e9
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not nm: continue
        if (prefix in nm) and nm.endswith(suffix):
            _, _, top = target_halfz_bottom_top(model, data, gid)
            if top > best_top:
                best_top = top; best_gid = gid
    return best_gid

def find_freejoint_for_body(model, bid):
    """返回 (jID, qpos_adr) 若此 body 有 freejoint；否则 (-1, -1)"""
    jadr = model.body_jntadr[bid]
    jnum = model.body_jntnum[bid]
    for k in range(jnum):
        jid = jadr + k
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return jid, model.jnt_qposadr[jid]
    return -1, -1

# ====================== 主流程 ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, required=True)
    ap.add_argument("--grasps", type=str, required=True)
    ap.add_argument("--target_geom", type=str, default="", help="目标 geom（不填则自动选 Z 最高的 lego_*_geom）")
    ap.add_argument("--try_top_k", type=int, default=999999, help="最多尝试的 grasp 数（默认全试）")
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--approach_tol_deg", type=float, default=APPROACH_TOL_DEG)
    args = ap.parse_args()

    # 1) 加载 MuJoCo（不导入 viewer，纯 headless）
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 控制器/关节 id
    aid_x     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x")
    aid_y     = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y")
    aid_lift  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    aid_left  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_joint")
    aid_right = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_joint")
    aid_rot   = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotation")
    jid_yaw   = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
    qadr_yaw  = model.jnt_qposadr[jid_yaw]

    bid_palm  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    bid_left  = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link")
    bid_right = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link")

    loZ, hiZ = model.actuator_ctrlrange[aid_lift]
    loR, hiR = model.actuator_ctrlrange[aid_rot]

    # 2) 选择目标 lego
    if args.target_geom:
        gid_tgt = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target_geom)
        assert gid_tgt >= 0, f"找不到 geom: {args.target_geom}"
    else:
        gid_tgt = find_topmost_lego_geom(model, data)
        assert gid_tgt >= 0, "没找到 lego_*_geom"
        args.target_geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid_tgt)

    bid_tgt = model.geom_bodyid[gid_tgt]
    free_jid, free_qadr = find_freejoint_for_body(model, bid_tgt)

    # 记录初始状态（用于每次试抓复位）
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    ctrl0 = np.zeros_like(data.ctrl)

    # 手爪初始放到安全状态
    def init_controls():
        data.ctrl[aid_left]  = OPEN_CMD
        data.ctrl[aid_right] = OPEN_CMD
        data.ctrl[aid_lift]  = 0.25
        data.ctrl[aid_rot]   = 0.0

    init_controls()
    wait_steps(model, data, 50)

    # palm→指尖的竖向偏移
    palm2tip = data.xpos[bid_palm][2] - body_bottom_z(model, data, bid_left)

    # 3) 读取 grasps JSON（物体坐标）
    jd = json.load(open(args.grasps, "r"))
    grasps = jd.get("grasps", [])
    assert len(grasps) > 0, "JSON 里没有 grasps 字段或为空"

    # 物体→世界 的姿态（选目标 geom 的姿态做代表）
    T_obj2world = pose_of_geom(model, data, gid_tgt)

    # 4) 过滤“接近方向≈自上而下”
    up_world = np.array([0,0,1.0])
    cos_tol  = np.cos(np.deg2rad(args.approach_tol_deg))

    world_grasps = []
    for g in grasps:
        p_obj = np.asarray(g["position"], dtype=float)
        q_obj = np.asarray(g["quaternion_xyzw"], dtype=float)  # 如果你是别的字段名，在这儿改
        R_obj = quat_xyzw_to_R(q_obj)
        width = float(g.get("width", 0.01))
        score = float(g.get("score", 0.0))

        # 变换到世界
        Rw = T_obj2world[:3,:3] @ R_obj
        pw = T_obj2world[:3,:3] @ p_obj + T_obj2world[:3,3]

        # 只试接近方向 z ≈ -Z
        z_w = Rw[:,2]
        if np.dot(z_w, -up_world) < cos_tol:
            continue

        world_grasps.append({"pos": pw, "R": Rw, "width": width, "score": score})

    # 排序（按 score，如无 score 都是 0 则保持原顺序）
    world_grasps.sort(key=lambda x: x["score"], reverse=True)
    if args.try_top_k < len(world_grasps):
        world_grasps = world_grasps[:args.try_top_k]

    print(f"[INFO] 目标: {args.target_geom}, 可尝试 grasps: {len(world_grasps)}")

    # 5) 开始前沉降
    deadline = time.time() + SETTLE_SECONDS
    stable_cnt = 0
    while time.time() < deadline:
        if np.linalg.norm(data.qvel) < SETTLE_VEL_THR:
            stable_cnt += 1
        else:
            stable_cnt = 0
        mujoco.mj_step(model, data)
        if stable_cnt >= SETTLE_STEPS_OK:
            break
    print("[SETTLE] ok")

    # 6) 每个 grasp 独立验证（每次复位）
    # 复位方法：恢复 qpos/qvel，清零 ctrl，然后设定安全 ctrl
    def reset_trial():
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        data.ctrl[:] = ctrl0
        mujoco.mj_forward(model, data)
        init_controls()
        wait_steps(model, data, 30)

    # 方便拿目标初始高度（每次重置后）
    def target_z():
        return data.xpos[bid_tgt][2]

    results = []
    loR, hiR = model.actuator_ctrlrange[aid_rot]
    loZ, hiZ = model.actuator_ctrlrange[aid_lift]

    for i, g in enumerate(world_grasps):
        reset_trial()

        pw = g["pos"]; Rw = g["R"]; width = g["width"]; score = g["score"]
        xw = Rw[:,0]
        yaw_des = np.arctan2(xw[1], xw[0])

        # 1) XY 对齐
        data.ctrl[aid_x] = pw[0]
        data.ctrl[aid_y] = pw[1]
        wait_steps(model, data, STEP_XY_SETTLE)

        # 2) yaw 调整
        ok_rot = False
        for _ in range(MAX_ROT_STEPS):
            data.ctrl[aid_rot] = float(np.clip(yaw_des, loR, hiR))
            mujoco.mj_step(model, data)
            if abs(wrap_to_pi(data.qpos[name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")] - yaw_des)) < ROT_TOL:
                ok_rot = True
                break

        # 3) 下压到抓取中心高度
        z_goal = pw[2] + DOWN_SAFE_GAP + palm2tip
        ok_desc = False
        for _ in range(MAX_DESC_STEPS):
            palm_z = data.xpos[bid_palm][2]
            dz = palm_z - z_goal
            if dz <= 0.004:
                ok_desc = True
                break
            step = min(0.003, max(0.0, dz)*0.35)
            cur  = float(data.ctrl[aid_lift])
            data.ctrl[aid_lift] = np.clip(cur - step, loZ, hiZ)
            mujoco.mj_step(model, data)

        # 4) 闭合，检测接触力
        data.ctrl[aid_left]  = CLOSE_CMD
        data.ctrl[aid_right] = CLOSE_CMD
        hit_ok = False
        lF=rF=0.0
        for _ in range(MAX_CLOSE_STEPS):
            lF, rF, hitL, hitR = forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, bid_tgt)
            if hitL and hitR and (lF + rF) > FORCE_OK_SUM:
                hit_ok = True
                break
            mujoco.mj_step(model, data)

        # 5) 抬起并判定成功
        before = target_z()
        for _ in range(MAX_LIFT_STEPS):
            cur = float(data.ctrl[aid_lift])
            step = min(0.003, abs(LIFT_UP_VALUE - cur)*0.5)
            data.ctrl[aid_lift] = np.clip(cur + step, loZ, hiZ)
            mujoco.mj_step(model, data)
        after  = target_z()
        lifted = (after - before) > SUCCESS_LIFT_DZ

        results.append({
            "index": i,
            "success": bool(lifted and hit_ok and ok_rot and ok_desc),
            "lift_dz": float(after - before),
            "sum_force": float(lF + rF),
            "yaw_deg": float(np.degrees(yaw_des)),
            "width": float(width),
            "score": float(score),
            "align_ok": True,
            "rot_ok": bool(ok_rot),
            "desc_ok": bool(ok_desc),
            "contact_ok": bool(hit_ok)
        })

        # 松开，准备下一次（不必 viewer，同步更快）
        data.ctrl[aid_left]  = OPEN_CMD
        data.ctrl[aid_right] = OPEN_CMD
        wait_steps(model, data, 20)

    # 7) 保存报告
    out = {
        "xml": args.xml,
        "target_geom": args.target_geom,
        "num_tested": len(results),
        "success_count": int(sum(1 for r in results if r["success"])),
        "approach_tol_deg": args.approach_tol_deg,
        "force_ok_sum": FORCE_OK_SUM,
        "results": results
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {out['success_count']}/{out['num_tested']} 成功 → {args.out}")

if __name__ == "__main__":
    main()
