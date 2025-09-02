import random
import math

# ---------- 参数可按需修改 ----------
GRID_N = 10                  # 10x10
GRID_SPACING = 0.020         # 2 cm 网格间距（居中铺满）
TRAY_INNER_HALF = 0.20       # 托盘内半宽（你的 tray_floor size 是 0.20）
FLOOR_TOP_Z = 0.010          # 托盘面顶面高度：tray_floor pos z=0.005, 厚度 0.005 -> 顶面 ~0.010
LEGO_HALF_Z = 0.009          # 约 9mm 半高（按你之前模型经验值）
BASE_EPS = 0.0005            # 静态底层与地面留一点缝，避免数值抖动
BASE_Z = FLOOR_TOP_Z + LEGO_HALF_Z + BASE_EPS  # 静态底层乐高的 z

DROP_COUNT = 100             # 掉落数量
DROP_MIN_DIST = 0.022        # 掉落之间的最小间距（>2.0cm 更安全）
DROP_AREA_HALF = 0.165       # 掉落采样的 XY 半径范围（小于内侧 0.20，留边距）
DROP_Z = 0.28                # 掉落初始高度
RAND_SEED = 42               # 固定随机种子，便于复现
# -----------------------------------

random.seed(RAND_SEED)

def color():
    # 柔和随机色
    return f"{random.uniform(0.2,0.9):.2f} {random.uniform(0.2,0.9):.2f} {random.uniform(0.2,0.9):.2f} 1"

def safe_xy(x, y):
    # 确保在托盘内侧范围
    return abs(x) <= (TRAY_INNER_HALF - 0.01) and abs(y) <= (TRAY_INNER_HALF - 0.01)

def gen_grid_bases():
    out = []
    # 让网格以 (0,0) 居中
    offset = (GRID_N - 1) * GRID_SPACING * 0.5
    for r in range(GRID_N):
        for c in range(GRID_N):
            x = -offset + c * GRID_SPACING
            y = -offset + r * GRID_SPACING
            if not safe_xy(x, y):
                continue
            name = f"lego_base_{r:02d}_{c:02d}"
            gname = f"{name}_geom"
            rgba = color()
            out.append(f'''
      <body name="{name}" pos="{x:.4f} {y:.4f} {BASE_Z:.4f}">
        <!-- 静态基座：无 freejoint，焊死在世界坐标（不会动） -->
        <geom type="sdf" name="{gname}" mesh="lego" rgba="{rgba}" friction="2.0 0.01 0.00001">
          <plugin instance="sdflego"/>
        </geom>
      </body>''')
    return "\n".join(out)

def too_close(p, pts, dmin):
    for (x,y) in pts:
        dx = x - p[0]
        dy = y - p[1]
        if dx*dx + dy*dy < dmin*dmin:
            return True
    return False

def gen_drop_positions(n, half, dmin, max_try=10000):
    pts = []
    tries = 0
    while len(pts) < n and tries < max_try:
        tries += 1
        x = random.uniform(-half, half)
        y = random.uniform(-half, half)
        if not safe_xy(x, y):
            continue
        if too_close((x,y), pts, dmin):
            continue
        pts.append((x,y))
    return pts

def gen_drops():
    out = []
    pts = gen_drop_positions(DROP_COUNT, DROP_AREA_HALF, DROP_MIN_DIST)
    if len(pts) < DROP_COUNT:
        print(f"<!-- [WARN] 仅放置了 {len(pts)} / {DROP_COUNT} 个掉落体（采样空间或最小间距过紧，可调小 DROP_MIN_DIST 或增大 DROP_AREA_HALF） -->")

    for i, (x, y) in enumerate(pts):
        # 只给个随机偏航（z 轴）更稳定；如需任意姿态，可加 e.g. euler="rx ry rz"
        yaw_deg = random.uniform(0, 360.0)
        name = f"lego_drop_{i:03d}"
        gname = f"{name}_geom"
        rgba = color()
        out.append(f'''
      <body name="{name}" pos="{x:.4f} {y:.4f} {DROP_Z:.4f}" euler="0 0 {yaw_deg:.2f}">
        <freejoint/>
        <inertial pos="0 0 0" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
        <geom type="sdf" name="{gname}" mesh="lego" rgba="{rgba}" friction="2.0 0.01 0.00001">
          <plugin instance="sdflego"/>
        </geom>
      </body>''')
    return "\n".join(out)

snippet = "\n".join([
    "<!-- ===== 自动生成：10x10 静态基座 + 100 掉落乐高 ===== -->",
    gen_grid_bases(),
    gen_drops(),
    "<!-- ===== 结束 ===== -->"
])

print(snippet)
