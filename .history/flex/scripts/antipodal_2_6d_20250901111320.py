#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

def _unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def _extract_p1p2(item):
    """兼容 (p1,p2,...) 或 dict({'p1','p2',...})"""
    if isinstance(item, dict):
        return np.asarray(item["p1"], float), np.asarray(item["p2"], float)
    else:
        p1, p2 = item[0], item[1]
        return np.asarray(p1, float), np.asarray(p2, float)

def _build_frame_random_roll(x_axis, rng):
    """
    已知 x（夹爪开合方向），随机生成 (y,z)：
    1) 先用 world_up 构造一个基准 y0
    2) 在 x 轴周围随机旋转角度 theta：y = cosθ*y0 + sinθ*(x×y0)
    3) z = x × y，保证右手系
    """
    x = _unit(x_axis)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(x, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])

    # 基准 y0：来自 world_up 在 x 垂直平面上的投影
    y0 = world_up - np.dot(world_up, x) * x
    y0 = _unit(y0)
    z0 = _unit(np.cross(x, y0))        # 与 x,y0 正交

    # 在 (y0, z0) 平面中随机转角
    theta = rng.uniform(0.0, 2.0*np.pi)
    y = np.cos(theta) * y0 + np.sin(theta) * z0
    y = _unit(y)
    z = _unit(np.cross(x, y))          # 确保右手系
    return y, z

def pair_to_grasp_poses_multi(p1, p2, per_pair=5, palm_offset=0.0755, rng=None):
    """
    对单个点对生成多个 6D 位姿（随机 z / 随机 roll around x）
    返回列表，每个元素：{'center','rotation','T','p1','p2'}
    """
    if rng is None:
        rng = np.random.default_rng()

    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    center_tip = 0.5 * (p1 + p2)

    x_axis = p2 - p1
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("接触点重合，无法定义夹爪开合方向 x。")
    x_axis = x_axis / x_norm

    poses = []
    for _ in range(int(per_pair)):
        y_axis, z_axis = _build_frame_random_roll(x_axis, rng)
        R = np.column_stack([x_axis, y_axis, z_axis])
        # 掌心位置：沿 z 正方向偏移
        center_palm = center_tip + z_axis * palm_offset

        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3,  3] = center_palm

        poses.append({
            "center": center_palm,     # t
            "rotation": R,             # 3x3
            "T": T,                    # 4x4
            "p1": p1, "p2": p2
        })
    return poses

def pairs_to_grasp_poses(pairs, per_pair=5, palm_offset=0.0755, seed=42):
    """
    将一组 antipodal 点对转换为多组 6D 位姿
    - 支持 dict/tuple 输入格式
    - 每对生成 per_pair 个位姿（roll 随机）
    """
    rng = np.random.default_rng(seed)
    all_poses = []
    err_cnt = 0

    for idx, item in enumerate(pairs):
        try:
            p1, p2 = _extract_p1p2(item)
            poses = pair_to_grasp_poses_multi(p1, p2, per_pair=per_pair,
                                              palm_offset=palm_offset, rng=rng)
            # 记录 pair 索引用于追踪
            for P in poses:
                P["pair_index"] = idx
            all_poses.extend(poses)
        except Exception as e:
            err_cnt += 1
            if err_cnt <= 5:
                print(f"[WARN] 第 {idx} 对失败：{e}")

    print(f"成功转换 {len(pairs)-err_cnt}/{len(pairs)} 个点对，共生成 {len(all_poses)} 个 6D 位姿")
    return all_poses

def main():
    ap = argparse.ArgumentParser("Antipodal pairs → 多个随机 roll 的 6D 位姿")
    ap.add_argument("--pairs_path", type=str,
                    default="../results/antipodal_pairs/antipodal_pairs_ray.npy")
    ap.add_argument("--out_dir", type=str, default="../results/6d")
    ap.add_argument("--per_pair", type=int, default=5,
                    help="每个点对生成多少个位姿")
    ap.add_argument("--palm_offset", type=float, default=0.0755,
                    help="沿 z 轴偏移量（米）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 找文件（含兜底）
    pairs_path = args.pairs_path
    if not os.path.exists(pairs_path):
        alt = "../results/antipodal_pairs.npy"
        if os.path.exists(alt):
            pairs_path = alt
            print(f"使用备用文件路径: {pairs_path}")
        else:
            raise FileNotFoundError(f"找不到 antipodal pairs 文件: {args.pairs_path}")

    pairs = np.load(pairs_path, allow_pickle=True)
    print(f"加载了 {len(pairs)} 条 antipodal 记录")

    os.makedirs(args.out_dir, exist_ok=True)
    poses = pairs_to_grasp_poses(pairs,
                                 per_pair=args.per_pair,
                                 palm_offset=args.palm_offset,
                                 seed=args.seed)

    if len(poses) > 0:
        out_path = os.path.join(args.out_dir, "grasp_poses.npy")
        np.save(out_path, poses, allow_pickle=True)
        print(f"✅ 已保存 {len(poses)} 个 6D 位姿到: {out_path}")

        # 打印一个示例
        first = poses[0]
        R = first["rotation"]
        print("\n示例位姿：")
        print("中心点(palm):", first["center"])
        print("x(开合):", R[:,0])
        print("y:", R[:,1])
        print("z(接近):", R[:,2])
        print("det(R) =", np.linalg.det(R))
    else:
        print("⚠️ 未生成任何位姿")

if __name__ == "__main__":
    main()

# import numpy as np
# import os
# def pair_to_grasp_pose(p1, p2, n1, n2, palm_offset=0.1315, mode="random"):
#     center = (p1 + p2) / 2
#     x_axis = (p2 - p1)
#     if np.linalg.norm(x_axis) < 1e-8:
#         raise ValueError("接触点重合，无法定义抓取方向")
#     x_axis /= np.linalg.norm(x_axis)

#     if mode == "avg":  # 平均法向
#         z_axis = (n1 + n2)
#         if np.linalg.norm(z_axis) < 1e-8:
#             z_axis = np.array([0,0,1])  # fallback
#     elif mode == "up":  # 固定向上
#         z_axis = np.array([0,0,1])
#         if abs(np.dot(z_axis, x_axis)) > 0.9:  # 防止平行
#             z_axis = np.array([0,1,0])
#     elif mode == "random":  # 随机正交
#         rand = np.random.randn(3)
#         z_axis = rand - np.dot(rand, x_axis) * x_axis
#     else:
#         raise ValueError(f"未知模式: {mode}")

#     z_axis /= np.linalg.norm(z_axis)
#     y_axis = np.cross(z_axis, x_axis)
#     y_axis /= np.linalg.norm(y_axis)
#     x_axis = np.cross(y_axis, z_axis)
#     x_axis /= np.linalg.norm(x_axis)

#     R = np.stack([x_axis, y_axis, z_axis], axis=1)
#     palm_center = center - z_axis * palm_offset
#     return {'center': palm_center, 'rotation': R}


# def pairs_to_grasp_poses(pairs, palm_offset=0):
#     poses = []
#     for idx, (p1, p2, n1, n2) in enumerate(pairs):
#         try:
#             pose = pair_to_grasp_pose(np.array(p1), np.array(p2), np.array(n1), np.array(n2), palm_offset)
#             poses.append(pose)
#         except Exception as e:
#             print(f"第{idx}对转换失败: {e}")
#     return poses

# if __name__ == "__main__":
#     pairs = np.load("../results/antipodal_pairs/antipodal_pairs_ray.npy", allow_pickle=True)
#     os.makedirs("../results/6d", exist_ok=True)
#     poses = pairs_to_grasp_poses(pairs, palm_offset=0.03)
#     np.save("../results/6d/grasp_poses.npy", poses)
#     if len(poses) > 0:
#         np.save("../results/6d/grasp_poses.npy", poses)
#         print(f"已保存 {len(poses)} 个夹爪指尖6D位姿到 ../results/6d/grasp_poses.npy")
        
#         # 打印第一个位姿作为示例
#         first_pose = poses[0]
#         print("\n第一个夹爪指尖位姿示例:")
#         print(f"中心点: {first_pose['center']}")
#         print(f"x轴(开合方向): {first_pose['rotation'][:, 0]}")
#         print(f"y轴: {first_pose['rotation'][:, 1]}")
#         print(f"z轴(朝上): {first_pose['rotation'][:, 2]}")
#         print(f"旋转矩阵:\n{first_pose['rotation']}")
#     else:
#         print("警告: 没有成功转换任何夹爪指尖位姿")