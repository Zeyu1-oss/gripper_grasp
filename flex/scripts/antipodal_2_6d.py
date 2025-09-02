#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

class GraspPoseGenerator:
    """
    将 antipodal 点对转换为夹爪 palm 的 6D 位姿：
    - x 轴 = p1 -> p2（开合方向）
    - y/z 轴 = 在绕 x 轴随机 roll 后得到（随机“z 方向”）
    - palm 坐标 = 指尖中心 + z * palm_offset
    """
    def __init__(self, per_pair: int = 5, palm_offset: float = 0.0755,
                 seed: int = 42, roll_range_deg: float | None = None):
        """
        per_pair      : 每个点对生成多少个位姿
        palm_offset   : 沿 z 轴的偏移（米）
        seed          : 随机种子，保证复现
        roll_range_deg: 若为 None 则 [0, 360) 随机；否则在 [-range, +range] 度内随机
        """
        self.per_pair = int(per_pair)
        self.palm_offset = float(palm_offset)
        self.rng = np.random.default_rng(seed)
        self.roll_range_deg = roll_range_deg

    # --------- 小工具 ---------
    @staticmethod
    def _unit(v, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        return v if n < eps else v / n

    @staticmethod
    def _extract_p1p2(item):
        """兼容 (p1,p2,...) 或 dict({'p1','p2',...}) 输入"""
        if isinstance(item, dict):
            return np.asarray(item["p1"], float), np.asarray(item["p2"], float)
        else:
            p1, p2 = item[0], item[1]
            return np.asarray(p1, float), np.asarray(p2, float)

    def _sample_roll_rad(self) -> float:
        """返回随机 roll 角度（弧度）"""
        if self.roll_range_deg is None:
            return self.rng.uniform(0.0, 2.0 * np.pi)
        a = np.deg2rad(self.roll_range_deg)
        return self.rng.uniform(-a, +a)

    def _build_frame_random_roll(self, x_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        已知 x（开合方向），构造随机 roll 的右手坐标系 (x,y,z)。
        1) 用 world_up 构造基准 y0（在垂直于 x 的平面投影）
        2) 在 (y0, z0) 平面内绕 x 随机旋转
        """
        x = self._unit(x_axis)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(x, world_up)) > 0.95:  # x 接近 z 时改用 y
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # 基准 y0：把 world_up 投影到 x 的正交平面
        y0 = world_up - np.dot(world_up, x) * x
        y0 = self._unit(y0)
        z0 = self._unit(np.cross(x, y0))

        theta = self._sample_roll_rad()
        y = np.cos(theta) * y0 + np.sin(theta) * z0
        y = self._unit(y)
        z = self._unit(np.cross(x, y))  # 右手系
        return x, y, z

    @staticmethod
    def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = t
        return T

    # --------- 单对点 → 多个 6D 位姿 ---------
    def poses_from_pair(self, p1: np.ndarray, p2: np.ndarray) -> list[dict]:
        """
        对单个 (p1, p2) 生成 self.per_pair 个 6D 位姿。
        返回字典列表：{'center','rotation','T','p1','p2'}
        """
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)

        x_axis = p2 - p1
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            raise ValueError("接触点重合，无法定义夹爪开合方向 x。")
        x_axis = x_axis / x_norm

        center_tip = 0.5 * (p1 + p2)
        out = []
        for _ in range(self.per_pair):
            x, y, z = self._build_frame_random_roll(x_axis)
            R = np.column_stack([x, y, z])
            center_palm = center_tip + z * self.palm_offset
            T = self._make_T(R, center_palm)
            out.append({
                "center": center_palm,
                "rotation": R,
                "T": T,
                "p1": p1,
                "p2": p2
            })
        return out

    # --------- 多对点 → 多个 6D 位姿 ---------
    def convert(self, pairs: list | np.ndarray) -> list[dict]:
        """
        pairs: list/ndarray，元素可为 (p1,p2,...) 或 {'p1','p2',...}
        返回累积的位姿列表，每个含 pair_index 字段
        """
        all_poses = []
        err_cnt = 0
        for idx, item in enumerate(pairs):
            try:
                p1, p2 = self._extract_p1p2(item)
                poses = self.poses_from_pair(p1, p2)
                for P in poses:
                    P["pair_index"] = idx
                all_poses.extend(poses)
            except Exception as e:
                err_cnt += 1
                if err_cnt <= 5:
                    print(f"[WARN] 第 {idx} 对失败: {e}")
        print(f"成功转换 {len(pairs)-err_cnt}/{len(pairs)} 对，共生成 {len(all_poses)} 个 6D 位姿")
        return all_poses

    @staticmethod
    def save(poses: list[dict], out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, poses, allow_pickle=True)
        print(f"✅ 已保存 {len(poses)} 个 6D 位姿到: {out_path}")

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser("Antipodal pairs → 多个随机 roll 的 6D 位姿（类封装）")
    ap.add_argument("--pairs_path", type=str,
                    default="../results/antipodal_pairs/lego_pairs.npy")
    ap.add_argument("--out_dir", type=str, default="../results/6d")
    ap.add_argument("--out_name", type=str, default="grasp_poses.npy")
    ap.add_argument("--per_pair", type=int, default=5)
    ap.add_argument("--palm_offset", type=float, default=0.0755)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--roll_range_deg", type=float, default=None,
                    help="若指定，则 roll∈[-range,+range]；默认 None 表示 [0,360)")
    args = ap.parse_args()

    pairs_path = args.pairs_path
    if not os.path.exists(pairs_path):
        # 兜底
        alt = "../results/antipodal_pairs.npy"
        if os.path.exists(alt):
            pairs_path = alt
            print(f"使用备用文件路径: {pairs_path}")
        else:
            raise FileNotFoundError(f"找不到 antipodal pairs 文件: {args.pairs_path}")

    pairs = np.load(pairs_path, allow_pickle=True)
    print(f"加载了 {len(pairs)} 条 antipodal 记录")

    gen = GraspPoseGenerator(per_pair=args.per_pair,
                             palm_offset=args.palm_offset,
                             seed=args.seed,
                             roll_range_deg=args.roll_range_deg)

    poses = gen.convert(pairs)

    if len(poses) > 0:
        out_path = os.path.join(args.out_dir, args.out_name)
        gen.save(poses, out_path)

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