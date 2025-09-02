#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 N 个不重叠 LEGO 方块的 XML 片段（托盘内，约 1 cm 立方，mesh 名为 "lego"）
默认输出到 generated_lego.xml，可在主 XML 的 <worldbody> 中 <include> 进来。
"""

import argparse, random, math
from itertools import product

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100, help="LEGO 数量")
    ap.add_argument("--outfile", type=str, default="generated_lego.xml", help="输出文件名")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    # 几何/边界参数（按你的托盘和乐高设置）
    ap.add_argument("--tray_half", type=float, default=0.20, help="托盘半尺寸 (m)，来自 tray_floor size=0.20")
    ap.add_argument("--wall_thick", type=float, default=0.005, help="围墙厚度 (m)")
    ap.add_argument("--lego_size", type=float, default=0.01, help="乐高近似边长 (m)")
    ap.add_argument("--safety", type=float, default=0.002, help="离墙安全边距 (m)")
    ap.add_argument("--cell", type=float, default=0.020, help="栅格间距 (m)，>= 乐高直径(=size) 更安全")
    ap.add_argument("--jitter", type=float, default=0.003, help="抖动幅度 (m)，保证 cell - 2*jitter >= 2*半边长")
    ap.add_argument("--zmin", type=float, default=0.06, help="初始高度下界 (m)")
    ap.add_argument("--zmax", type=float, default=0.20, help="初始高度上界 (m)")
    args = ap.parse_args()

    random.seed(args.seed)

    # 有效平面边界（避开墙+半砖+安全余量）
    half = args.tray_half - args.wall_thick - args.lego_size/2.0 - args.safety
    xmin = ymin = -half
    xmax = ymax =  half

    # 生成规整网格中心
    def frange(a, b, step):
        # 包含 b 的“近似闭区间”步进
        x = a
        out = []
        while x <= b + 1e-9:
            out.append(round(x, 6))
            x += step
        return out

    xs = frange(xmin, xmax, args.cell)
    ys = frange(ymin, ymax, args.cell)
    grid = [(x, y) for x, y in product(xs, ys)]

    if len(grid) < args.count:
        raise SystemExit(f"[ERROR] 网格点数量 {len(grid)} < 期望数量 {args.count}，请减小 --count 或减小 --cell。")

    # 随机打乱并取前 N 个，再施加小抖动，保证最小中心距 >= cell - 2*jitter
    random.shuffle(grid)
    chosen = grid[:args.count]

    # 校验抖动是否安全：cell - 2*jitter >= lego_size
    if args.cell - 2*args.jitter < args.lego_size:
        raise SystemExit(f"[ERROR] cell - 2*jitter = {args.cell - 2*args.jitter:.4f} < lego_size={args.lego_size:.4f}，会有重叠风险。请减小 --jitter 或增大 --cell。")

    # 一些好看的随机颜色（RGBA）
    palette = [
        (0.90,0.30,0.10,1), (0.90,0.60,0.10,1), (0.20,0.70,0.20,1),
        (0.20,0.50,0.90,1), (0.60,0.30,0.90,1), (0.95,0.20,0.40,1),
        (0.20,0.80,0.80,1), (0.40,0.55,0.75,1), (0.70,0.40,0.40,1),
        (0.30,0.30,0.80,1)
    ]

    def rand_rgba():
        r,g,b,a = random.choice(palette)
        # 轻微扰动避免完全相同
        r = min(max(r + random.uniform(-0.05, 0.05), 0.05), 0.95)
        g = min(max(g + random.uniform(-0.05, 0.05), 0.05), 0.95)
        b = min(max(b + random.uniform(-0.05, 0.05), 0.05), 0.95)
        return f"{r:.3f} {g:.3f} {b:.3f} 1"

    # 输出 XML 片段
    lines = []
    lines.append("<!-- Generated LEGO bodies: paste under <worldbody> or <include this file> -->")
    for i, (cx, cy) in enumerate(chosen):
        jx = random.uniform(-args.jitter, args.jitter)
        jy = random.uniform(-args.jitter, args.jitter)
        x = min(max(cx + jx, xmin), xmax)
        y = min(max(cy + jy, ymin), ymax)
        z = random.uniform(args.zmin, args.zmax)
        rgba = rand_rgba()
        name = f"lego_{i:02d}"
        gname = f"{name}_geom"
        block = f"""  <body name="{name}" pos="{x:.3f} {y:.3f} {z:.3f}">
    <freejoint/>
    <inertial pos="0 0 0" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
    <geom type="sdf" name="{gname}" friction="2.0 0.01 0.00001" mesh="lego" rgba="{rgba}">
      <plugin instance="sdflego"/>
    </geom>
  </body>"""
        lines.append(block)

    xml_text = "\n".join(lines)

    with open(args.outfile, "w", encoding="utf-8") as f:
        f.write(xml_text)

    # 简单验证最小中心距（仅在同一高度投影的 2D 距离，保守估计）
    # 由于我们来自规则网格 + 有界抖动，理论下限 = cell - 2*jitter
    min_d = args.cell - 2*args.jitter
    print(f"[OK] 已生成 {args.count} 个 LEGO 到 {args.outfile}")
    print(f"     托盘 XY 可用范围: [{xmin:.3f},{xmax:.3f}] × [{ymin:.3f},{ymax:.3f}] m")
    print(f"     栅格间距 cell = {args.cell:.3f} m，抖动 jitter = ±{args.jitter:.3f} m")
    print(f"     理论最小中心距 ≥ {min_d:.3f} m；乐高边长 ≈ {args.lego_size:.3f} m → 不重叠 ✅")

if __name__ == "__main__":
    main()
