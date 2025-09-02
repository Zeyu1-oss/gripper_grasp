
import os, json, math, argparse
import numpy as np

# 可选 EGL（无头/GPU）
if '--egl' in os.sys.argv:
    os.environ['MUJOCO_GL'] = 'egl'

import mujoco

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

def quat_xyzw_to_R(q):
    x,y,z,w = q
    n = math.sqrt(x*x+y*y+z*z+w*w)+1e-12
    x,y,z,w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], float)

def R_to_quat_wxyz(R):
    # MuJoCo freejoint 的顺序是 [x y z qw qx qy qz]
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t+1.0)*2
        qw = 0.25*s
        qx = (R[2,1]-R[1,2])/s
        qy = (R[0,2]-R[2,0])/s
        qz = (R[1,0]-R[0,1])/s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qx = 0.25*s
            qy = (R[0,1]+R[1,0])/s
            qz = (R[0,2]+R[2,0])/s
            qw = (R[2,1]-R[1,2])/s
        elif i == 1:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qx = (R[0,1]+R[1,0])/s
            qy = 0.25*s
            qz = (R[1,2]+R[2,1])/s
            qw = (R[0,2]-R[2,0])/s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qx = (R[0,2]+R[2,0])/s
            qy = (R[1,2]+R[2,1])/s
            qz = 0.25*s
            qw = (R[1,0]-R[0,1])/s
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)+1e-12
    return qw/n, qx/n, qy/n, qz/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', required=True, help='你的场景 XML（含夹爪、lego freejoint、grasp腱执行器）')
    ap.add_argument('--grasps', required=True, help='抓取JSON，字段包含 position, quaternion_xyzw（在物体坐标系）')
    ap.add_argument('--index', type=int, default=0, help='使用第几个抓取（默认0）')
    ap.add_argument('--lego_body', default='lego', help='LEGO 对应的 body 名称（有 freejoint）')
    ap.add_argument('--lego_geom', default='lego_01', help='LEGO 的 geom 名（用于取位置）')
    ap.add_argument('--gripper_site', default='gripper_center', help='夹爪中心 site 名')
    ap.add_argument('--close', type=float, default=0.02, help='夹爪合拢命令（取决于你XML里的执行器类型/量纲）')
    ap.add_argument('--freeze_steps', type=int, default=200, help='冻结阶段步数（抵消重力）')
    ap.add_argument('--hold_steps', type=int, default=600, help='给重力后保持步数')
    ap.add_argument('--tol_mm', type=float, default=2.0, help='判为“移动/掉落”的位移阈值（毫米）')
    ap.add_argument('--egl', action='store_true', help='使用 EGL（无头）')
    ap.add_argument('--out', default='static_hold_result.json')
    args = ap.parse_args()

    # 读抓取
    with open(args.grasps, 'r') as f:
        G = json.load(f)
    grasps = G.get('grasps', G)
    if not isinstance(grasps, list) or len(grasps) == 0:
        raise RuntimeError('抓取JSON格式错误或为空')
    g = grasps[args.index]
    p_rel = np.array(g['position'], float)              # 物体坐标系下的抓取点（相对物体）
    R_rel = quat_xyzw_to_R(g['quaternion_xyzw'])       # 物体→夹爪 旋转

    # 读模型
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # id
    jid_free = req_id(name2id(model, mujoco.mjtObj.mjOBJ_JOINT,  f'{args.lego_body}:freejoint') \
                      or name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'freejoint'),
                      'LEGO freejoint')
    qadr = model.jnt_qposadr[jid_free]      # 起始下标（7个：xyz+quat）
    bid_lego  = req_id(name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.lego_body), f'body:{args.lego_body}')
    gid_lego  = req_id(name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.lego_geom), f'geom:{args.lego_geom}')
    aid_grip  = req_id(name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper_motor'), 'actuator:gripper_motor')
    sid_grip  = req_id(name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.gripper_site), f'site:{args.gripper_site}')

    # 取夹爪当前世界位姿（保持不动）
    mujoco.mj_forward(model, data)
    p_g_world = data.site_xpos[sid_grip].copy()
    R_g_world = data.site_xmat[sid_grip].reshape(3,3).copy()

    # 计算物体世界位姿，使得： p_g_world = p_obj + R_obj @ p_rel,  R_g_world = R_obj @ R_rel
    # => R_obj = R_g_world @ R_rel.T
    R_obj = R_g_world @ R_rel.T
    p_obj = p_g_world - R_obj @ p_rel

    # 写入 LEGO freejoint qpos（世界位姿）
    qw,qx,qy,qz = R_to_quat_wxyz(R_obj)
    data.qpos[qadr+0:qadr+3] = p_obj
    data.qpos[qadr+3:qadr+7] = np.array([qw,qx,qy,qz], float)
    mujoco.mj_forward(model, data)

    # 冻结：用 xfrc_applied 反重力
    mass = model.body_mass[bid_lego]
    gz   = model.opt.gravity[2]      # 一般是 -9.81
    anti = np.array([0,0, -mass*gz, 0,0,0], float)  # -gz 为正，抵消重力
    data.xfrc_applied[bid_lego] = anti

    # 让系统稳定一会
    wait_steps(model, data, args.freeze_steps)

    # 合拢
    data.ctrl[aid_grip] = float(args.close)
    wait_steps(model, data, 200)

    # 记录初始参考（给重力前）
    mujoco.mj_forward(model, data)
    p_ref = data.geom_xpos[gid_lego].copy()

    # 给重力：取消反力
    data.xfrc_applied[bid_lego] = 0.0

    # 观察保持
    max_disp = 0.0
    for _ in range(args.hold_steps):
        mujoco.mj_step(model, data)
        p_now = data.geom_xpos[gid_lego]
        disp = np.linalg.norm(p_now - p_ref)
        if disp > max_disp:
            max_disp = disp

    tol = args.tol_mm * 1e-3
    success = (max_disp < tol)

    result = {
        "xml": args.xml,
        "grasp_index": args.index,
        "position_rel_obj": p_rel.tolist(),
        "success": bool(success),
        "max_motion_m": float(max_disp),
        "tol_m": float(tol)
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[DONE] {'✅ 成功' if success else '❌ 失败'} | 最大位移={max_disp*1000:.2f} mm (阈值 {args.tol_mm:.1f} mm) → {args.out}")

if __name__=="__main__":
    main
