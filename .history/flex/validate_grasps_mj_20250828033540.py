#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, time, argparse
import numpy as np
import mujoco
import mujoco.viewer

# ====== 可调参数（和你现有控制一致） ======
OPEN_CMD        = 0.8
CLOSE_CMD       = -0.6
XY_ALIGN_TOL    = 0.002
STEP_XY_SETTLE  = 10
DOWN_SAFE_GAP   = 0.0015     # 手指尖与目标上表面安全间隙
PRINT_PERIOD    = 0.20
ROT_TOL         = 0.02       # yaw 收敛（弧度）
SETTLE_SECONDS  = 3.0        # 开始前沉降
SETTLE_VEL_THR  = 0.02
SETTLE_STEPS_OK = 200
APPROACH_TOL_DEG = 25        # 只尝试 z轴≈-Z 的抓取
LIFT_UP_VALUE   = 0.30       # 抓起后抬到的升降目标
FORCE_OK_SUM    = 0.5        # 夹爪两侧合力阈值（N）
FALLBACK_HALFZ  = 0.009      # SDF/mesh 没 size 时估的半厚度

# ====== 小工具 ======
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
    # q = [x, y, z, w]
    x,y,z,w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=float)
    return R

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
        if hz > 1e-6: halfz = hz
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
    return best_gid, best_top

# ====== 主流程 ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, required=True)
    ap.add_argument("--grasps", type=str, required=True)
    ap.add_argument("--try_top_k", type=int, default=50, help="尝试评分最高的前 K 个 grasp（且满足接近方向约束）")
    ap.add_argument("--target_geom", type=str, default="", help="直接指定目标 geom 名称（不指定则自动找 Z 最高的 lego）")
    args = ap.parse_args()

    # 加载 MuJoCo
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

    # 读取 grasps JSON
    jd = json.load(open(args.grasps, "r"))
    grasps = jd["grasps"]
    # 如果你的字段不是 "quaternion_xyzw"/"position"/"width"，这里改一下取值
    # grasps 元素示例：
    # {"position":[x,y,z], "quaternion_xyzw":[qx,qy,qz,qw], "width":0.012, "score":..., ...}

    # 选择目标 LEGO
    if args.target_geom:
        gid_tgt = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.target_geom)
        assert gid_tgt >= 0, f"找不到目标 geom: {args.target_geom}"
        _, _, topz = target_halfz_bottom_top(model, data, gid_tgt)
        print(f"[TARGET] 指定 geom={args.target_geom}，topZ={topz:.4f}")
    else:
        gid_tgt, topz = find_topmost_lego_geom(model, data)
        assert gid_tgt >= 0, "没找到 lego_*_geom"
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid_tgt)
        print(f"[TARGET] Z 最高的目标: {nm}, topZ={topz:.4f}")

    bid_tgt = model.geom_bodyid[gid_tgt]

    # 把 JSON grasp(物体坐标) 变到世界坐标，并过滤“从上接近”的
    T_obj2world = pose_of_geom(model, data, gid_tgt)
    up_world = np.array([0,0,1.0])
    cos_tol  = np.cos(np.deg2rad(APPROACH_TOL_DEG))

    world_grasps = []
    for g in grasps:
        pg_obj = np.asarray(g["position"], dtype=float)
        qg_obj = np.asarray(g["quaternion_xyzw"], dtype=float)
        Rg_obj = quat_xyzw_to_R(qg_obj)

        # 物体->世界
        Rw = T_obj2world[:3,:3] @ Rg_obj
        pw = T_obj2world[:3,:3] @ pg_obj + T_obj2world[:3,3]

        # 过滤：要求 grasp z 轴 ≈ 世界 -Z
        z_w = Rw[:,2]                # grasp 坐标系 z 轴（approach 方向）
        if np.dot(z_w, -up_world) < cos_tol:
            continue

        world_grasps.append({
            "pos": pw, "R": Rw,
            "width": float(g.get("width", 0.01)),
            "score": float(g.get("score", 0.0))
        })

    if not world_grasps:
        print("[WARN] 没有满足自上而下的抓取；调大 --try_top_k 或放宽 APPROACH_TOL_DEG")
        return

    # 按 score 排序，取前 K
    world_grasps.sort(key=lambda x: x["score"], reverse=True)
    world_grasps = world_grasps[:max(1, args.try_top_k)]
    print(f"[INFO] 可尝试的 grasp 数: {len(world_grasps)} (top-{args.try_top_k})")

    # 初始化手爪：张开、升到安全高度
    data.ctrl[aid_left]  = OPEN_CMD
    data.ctrl[aid_right] = OPEN_CMD
    data.ctrl[aid_lift]  = 0.25
    data.ctrl[aid_rot]   = 0.0
    wait_steps(model, data, 50)

    # 计算 palm→指尖 的竖向偏移（用于下压目标）
    palm2tip = data.xpos[bid_palm][2] - body_bottom_z(model, data, bid_left)

    # 先沉降几秒让砖块稳定
    deadline = time.time() + SETTLE_SECONDS
    stable_cnt = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and time.time() < deadline:
            if np.linalg.norm(data.qvel) < SETTLE_VEL_THR:
                stable_cnt += 1
            else:
                stable_cnt = 0
            mujoco.mj_step(model, data); viewer.sync()
            if stable_cnt >= SETTLE_STEPS_OK: break
        print("[SETTLE] 完成，开始尝试 grasps。")

        last_print = 0.0
        success = False

        for gi, g in enumerate(world_grasps):
            pw = g["pos"]; Rw = g["R"]; width = g["width"]
            # 只用 yaw：让 palm-x 对齐 grasp-x 的投影
            xw = Rw[:,0]
            yaw_des = np.arctan2(xw[1], xw[0])

            # === 1) 平移到 XY ===
            data.ctrl[aid_x] = pw[0]
            data.ctrl[aid_y] = pw[1]
            wait_steps(model, data, STEP_XY_SETTLE)

            # === 2) 调 yaw ===
            for _ in range(150):
                data.ctrl[aid_rot] = float(np.clip(yaw_des, loR, hiR))
                mujoco.mj_step(model, data)
                if abs(wrap_to_pi(data.qpos[qadr_yaw] - yaw_des)) < ROT_TOL:
                    break

            # === 3) 下压到接近 grasp 中心高度 ===
            halfz, lego_bot, lego_top = target_halfz_bottom_top(model, data, gid_tgt)
            z_goal = pw[2] + DOWN_SAFE_GAP + palm2tip
            for _ in range(300):
                palm_z = data.xpos[bid_palm][2]
                dz = palm_z - z_goal
                if dz <= 0.004:
                    break
                step = min(0.003, max(0.0, dz)*0.35)
                cur  = float(data.ctrl[aid_lift])
                data.ctrl[aid_lift] = np.clip(cur - step, loZ, hiZ)
                mujoco.mj_step(model, data); viewer.sync()

            # === 4) 闭合并检查接触力 ===
            data.ctrl[aid_left]  = CLOSE_CMD
            data.ctrl[aid_right] = CLOSE_CMD
            hit_ok = False
            for _ in range(120):
                lF, rF, hitL, hitR = forces_fingers_vs_target_by_body(model, data, bid_left, bid_right, model.geom_bodyid[gid_tgt])
                if time.time()-last_print > PRINT_PERIOD:
                    print(f"[TRY {gi:02d}] sumF={lF+rF:.3f}N hitL={hitL} hitR={hitR} yaw={np.degrees(yaw_des):.1f}°")
                    last_print = time.time()
                if hitL and hitR and (lF + rF) > FORCE_OK_SUM:
                    hit_ok = True
                    break
                mujoco.mj_step(model, data); viewer.sync()
            if not hit_ok:
                # 张开一点再换下一个 grasp
                data.ctrl[aid_left]  = OPEN_CMD
                data.ctrl[aid_right] = OPEN_CMD
                wait_steps(model, data, 40)
                continue

            # === 5) 抬起并判定是否抓起成功 ===
            before = data.xpos[model.geom_bodyid[gid_tgt]][2]
            for _ in range(200):
                cur = float(data.ctrl[aid_lift])
                step = min(0.003, abs(LIFT_UP_VALUE - cur)*0.5)
                data.ctrl[aid_lift] = np.clip(cur + step, loZ, hiZ)
                mujoco.mj_step(model, data); viewer.sync()
            after = data.xpos[model.geom_bodyid[gid_tgt]][2]

            lifted = (after - before) > 0.015  # 抬高超过 15mm 视为成功
            print(f"[RESULT {gi:02d}] lifted={lifted}  dz={after-before:.3f} m")

            if lifted:
                print("[SUCCESS] 抓取成功！")
                success = True
                break
            else:
                # 失败则松开回到安全位，尝试下一个
                data.ctrl[aid_left]  = OPEN_CMD
                data.ctrl[aid_right] = OPEN_CMD
                data.ctrl[aid_lift]  = 0.25
                wait_steps(model, data, 80)

        if not success:
            print("[FAIL] 所选 grasp 没成功。可放宽阈值或多试一些。")

if __name__ == "__main__":
    main()
