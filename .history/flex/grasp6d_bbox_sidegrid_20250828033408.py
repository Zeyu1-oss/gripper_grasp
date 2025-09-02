#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 OBB 的确定性侧向抓取生成（6D）
- 不依赖网格 watertight 或高质量法线；
- 在 OBB 的两个成对侧面（±X、±Y）上生成规则网格的对向接触；
- 自动过滤开口宽度不在 [wmin, wmax] 的轴向；
- 给出每个抓取的 6D 姿态（中心+四元数），并附上两接触点、开口宽度、坐标系。
坐标系约定（夹爪）：
- x: 夹紧方向（从 p -> q）
- z: 接近方向，固定为 OBB 的 -Z（即从上往下接近）
- y: z × x，保证右手系
输出在“mesh坐标系”下；落在仿真里时：p_world = R_body @ p_mesh + t_body；R_world = R_body @ R_grasp。
"""

import json, argparse, math
import numpy as np
import trimesh

# ---------------- 工具 ----------------
def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else (v/n)

def mat_to_quat(R):
    """3x3 -> xyzw quaternion"""
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
            w = (m[2,1] - m[1,2]) / s
        elif i == 1:
            s = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
            w = (m[0,2] - m[2,0]) / s
        else:
            s = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
            w = (m[1,0] - m[0,1]) / s
    return [x, y, z, w]

def load_mesh(path, scale=1.0):
    mesh = trimesh.load(path, process=False)
    if scale != 1.0:
        mesh.apply_scale(scale)
    # 轻处理：不依赖 watertight
    try:
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh

def oriented_bbox(mesh: trimesh.Trimesh):
    """
    用 Trimesh 的 oriented box：
    返回：
      extents: (3,) 盒子在自身坐标系下的边长 [Ex, Ey, Ez]
      Tbox2mesh: (4,4) 从盒子坐标 -> mesh 坐标的齐次变换
    """
    obb = mesh.bounding_box_oriented
    extents    = np.asarray(obb.primitive.extents,   dtype=float)   # (3,)
    Tbox2mesh  = np.asarray(obb.primitive.transform, dtype=float)   # (4,4)
    # 形状兜底校验
    extents = extents.reshape(3,)
    Tbox2mesh = Tbox2mesh.reshape(4,4)
    return extents, Tbox2mesh

def transform_points(R, t, P):
    """P(Nx3) 从 box 系变到 mesh 系：P_mesh = R @ P_box + t"""
    return P @ R.T + t

# 生成网格参数
def linspace_margin(a_min, a_max, n, margin):
    if n <= 1:
        return np.array([(a_min + a_max) * 0.5])
    a_min2 = a_min + margin
    a_max2 = a_max - margin
    if a_max2 <= a_min2:
        a_min2 = (a_min + a_max) * 0.5 - 1e-6
        a_max2 = (a_min + a_max) * 0.5 + 1e-6
    return np.linspace(a_min2, a_max2, n)

# ---------------- 主逻辑：基于 OBB 的侧抓 ----------------
def make_grasps_bbox_side(extents, Tbox2mesh, wmin, wmax,
                          grid_nu=5, grid_nv=6,
                          margin_u=0.001, margin_v=0.001):
    """
    在两个轴向（±X、±Y）的成对侧面上生成抓取。
    - 对于 ±X 面：开口宽度 = Ex；接触点在 y-z 面铺网格
    - 对于 ±Y 面：开口宽度 = Ey；接触点在 x-z 面铺网格
    - 对向、平行面，z(夹爪) 固定取 OBB 的 -Z（从“上”往“下”接近）
    返回：列表(dict)
    """
    Rbm = Tbox2mesh[:3, :3]
    tbm = Tbox2mesh[:3, 3]
    Ex, Ey, Ez = float(extents[0]), float(extents[1]), float(extents[2])

    grasps = []

    def add_axis(axis_idx):
        # axis_idx = 0 用 ±X 面（宽=Ex），其余两个轴是网格坐标；
        # axis_idx = 1 用 ±Y 面（宽=Ey）
        widths = [Ex, Ey, Ez]
        width = widths[axis_idx]
        if not (wmin - 1e-9 <= width <= wmax + 1e-9):
            return  # 开口宽度不在范围，跳过该轴

        # 网格的两个自由轴 id
        free_axes = [0,1,2]
        free_axes.remove(axis_idx)
        u_id, v_id = free_axes[0], free_axes[1]
        # 每个自由轴的取值范围（盒子中心±一半长度）
        half = [Ex*0.5, Ey*0.5, Ez*0.5]
        u_vals = linspace_margin(-half[u_id], +half[u_id], grid_nu, margin_u)
        v_vals = linspace_margin(-half[v_id], +half[v_id], grid_nv, margin_v)

        # 法线方向（盒子系）：
        n_minus = np.zeros(3); n_minus[axis_idx] = -1.0
        n_plus  = np.zeros(3); n_plus[axis_idx]  = +1.0

        # 夹爪坐标系（盒子系）：
        # x: 从 p_minus -> p_plus 的方向（即 +axis_idx）
        # z: 固定取 -Z（盒子系），表示从“上”往“下”接近
        # y: z × x
        x_axis_box = np.zeros(3); x_axis_box[axis_idx] = +1.0
        z_axis_box = np.array([0.0, 0.0, -1.0])
        # 若 axis_idx==2（即 ±Z 面），那就是顶/底面，不算“侧抓”，这里不生成
        if axis_idx == 2:
            return

        y_axis_box = unit(np.cross(z_axis_box, x_axis_box))
        x_axis_box = unit(np.cross(y_axis_box, z_axis_box))
        R_grasp_box = np.column_stack([x_axis_box, y_axis_box, z_axis_box])  # 列向量为基

        for u in u_vals:
            for v in v_vals:
                # 两接触点（盒子系）
                p_box = np.zeros(3); q_box = np.zeros(3)
                p_box[axis_idx] = -half[axis_idx]
                q_box[axis_idx] = +half[axis_idx]
                p_box[u_id] = u; p_box[v_id] = v
                q_box[u_id] = u; q_box[v_id] = v

                # 变到 mesh 系
                p_mesh = transform_points(Rbm, tbm, p_box[None,:])[0]
                q_mesh = transform_points(Rbm, tbm, q_box[None,:])[0]
                mid_mesh = 0.5 * (p_mesh + q_mesh)
                # 旋转同样需要从盒子系变到 mesh 系
                R_grasp_mesh = Rbm @ R_grasp_box
                quat_xyzw = mat_to_quat(R_grasp_mesh)

                grasps.append({
                    "position": mid_mesh.tolist(),
                    "quaternion_xyzw": quat_xyzw,
                    "width": float(width),
                    "contacts": [p_mesh.tolist(), q_mesh.tolist()],
                    "closing_dir_mesh": (R_grasp_mesh[:,0]).tolist(),
                    "approach_dir_mesh": (R_grasp_mesh[:,2]).tolist(),  # 约为“向下”
                    "axis_from_obb": ["+X/-X","+Y/-Y","(skip Z)"][axis_idx]
                })

    # 生成 ±X 面与 ±Y 面
    add_axis(0)
    add_axis(1)
    # 不在 ±Z 面上生成抓取（顶/底），视为“非侧抓”

    return grasps

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=str, required=True)
    ap.add_argument('--scale', type=float, default=1.0)

    # 你夹爪的开口范围（米）
    ap.add_argument('--wmin', type=float, default=0.004)  # 4mm
    ap.add_argument('--wmax', type=float, default=0.018)  # 18mm

    # 网格密度与边缘留白（避免靠边/倒角）
    ap.add_argument('--grid_u', type=int, default=5)      # 每个侧面网格 U 数
    ap.add_argument('--grid_v', type=int, default=6)      # 每个侧面网格 V 数
    ap.add_argument('--margin_u', type=float, default=0.001)  # 1mm
    ap.add_argument('--margin_v', type=float, default=0.001)

    ap.add_argument('--out', type=str, default='bbox_side_grasps.json')
    args = ap.parse_args()

    mesh = load_mesh(args.mesh, scale=args.scale)
    extents, Tbox2mesh = oriented_bbox(mesh)
    print(f"[INFO] OBB extents(m): {np.array2string(extents, precision=4)}")

    grasps = make_grasps_bbox_side(
        extents, Tbox2mesh,
        wmin=args.wmin, wmax=args.wmax,
        grid_nu=args.grid_u, grid_nv=args.grid_v,
        margin_u=args.margin_u, margin_v=args.margin_v
    )

    out = {
        "mesh": args.mesh,
        "scale": args.scale,
        "obb_extents": extents.tolist(),
        "obb_T_box_to_mesh": Tbox2mesh.tolist(),
        "assumptions": {
            "side_only": True,
            "parallel_opposite_faces": True,
            "approach_from_box_minusZ": True
        },
        "grasps": grasps
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[SUCCESS] {len(grasps)} grasps → {args.out}")

    if grasps:
        g0 = grasps[0]
        print(f"[BEST-ish] pos={np.round(g0['position'],4)}, width={g0['width']:.4f} m, axis={g0['axis_from_obb']}")

if __name__ == "__main__":
    main()
