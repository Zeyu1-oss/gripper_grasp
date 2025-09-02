#!/usr/bin/env python3
import argparse, time, json, numpy as np, mujoco

def name2id(model, objtype, name):
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None

def quat_xyzw_to_wxyz(q):
    # 输入 xyzw，MuJoCo mocap_quat 需要 wxyz
    x,y,z,w = q
    return np.array([w,x,y,z], dtype=float)

def step_n(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', required=True, help='g2_one_lego_eval.xml')
    ap.add_argument('--pose', type=str, default=None,
                    help='JSON 字符串: {"pos":[x,y,z],"quat_xyzw":[x,y,z,w],"width":0.012}')
    ap.add_argument('--json', type=str, default=None,
                    help='包含同结构 grasp 的 json 文件（取第一个）')
    ap.add_argument('--open', type=float, default=0.014, help='初始开口(米)')
    ap.add_argument('--close', type=float, default=0.001, help='闭合目标开口(米)')
    ap.add_argument('--lift', type=float, default=0.12, help='上抬高度(米)')
    ap.add_argument('--settle', type=float, default=0.5, help='阶段间沉降时间(s)')
    ap.add_argument('--success_dz', type=float, default=0.02, help='成功阈值: 抬升后物体高出初始(米)')
    args = ap.parse_args()

    # 读取 grasp
    if args.pose:
        g = json.loads(args.pose)
    elif args.json:
        with open(args.json,'r') as f:
            js = json.load(f)
        g = js['grasps'][0] if 'grasps' in js and js['grasps'] else js
    else:
        raise ValueError('需要 --pose 或 --json')

    pos  = np.array(g['position'      if 'position' in g else 'pos'], dtype=float)
    quat = np.array(g['quaternion_xyzw' if 'quaternion_xyzw' in g else 'quat_xyzw'], dtype=float)
    width = float(g.get('width', args.close*2))

    # 模型与把手
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    aid_left  = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left')
    aid_right = name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right')
    bid_obj   = name2id(model, mujoco.mjtObj.mjOBJ_BODY,      'obj')
    bid_mocap = name2id(model, mujoco.mjtObj.mjOBJ_BODY,      'tcp_target')

    if None in [aid_left, aid_right, bid_obj, bid_mocap]:
        raise RuntimeError('ID 解析失败：检查 actuator/body 命名。')

    mujoco.mj_forward(model, data)

    # 初始物体高度
    obj_z0 = float(data.xpos[bid_obj][2])

    # 1) 打开手指
    data.ctrl[aid_left]  = args.open
    data.ctrl[aid_right] = args.open
    step_n(model, data, int(args.settle/ model.opt.timestep))

    # 2) Pre-grasp：把 mocap 放到抓取位姿上方 2cm
    mocap_quat = quat_xyzw_to_wxyz(quat)
    approach = np.array([0,0,1.0])   # 从上往下接近；若你的 grasp 有 approach，可替换
    pre_pos = pos + 0.02 * approach
    data.mocap_pos[0]  = pre_pos
    data.mocap_quat[0] = mocap_quat
    step_n(model, data, int(args.settle/ model.opt.timestep))

    # 3) 下压到抓取位姿
    data.mocap_pos[0] = pos
    step_n(model, data, int(args.settle/ model.opt.timestep))

    # 4) 闭合（按给定 width；若不清楚映射，直接闭合到 close）
    target_gap = max(args.close, min(args.open, 0.5*width))
    data.ctrl[aid_left]  = target_gap
    data.ctrl[aid_right] = target_gap
    step_n(model, data, int(args.settle/ model.opt.timestep))

    # 5) 上抬
    data.mocap_pos[0] = pos + np.array([0,0,args.lift])
    step_n(model, data, int(0.6/ model.opt.timestep))

    # 判定成功：物体抬升超过阈值，并在最后 0.2s 仍然高于阈值
    obj_z = float(data.xpos[bid_obj][2])
    lifted = (obj_z - obj_z0) > args.success_dz

    hold_steps = int(0.2 / model.opt.timestep)
    held = True
    for _ in range(hold_steps):
        mujoco.mj_step(model, data)
        if float(data.xpos[bid_obj][2]) - obj_z0 <= args.success_dz:
            held = False
            break

    ok = bool(lifted and held)
    print(json.dumps({
        "success": ok,
        "dz": float(obj_z - obj_z0),
        "final_z": obj_z,
        "threshold": args.success_dz
    }, indent=2))

if __name__ == '__main__':
    main()
