import random, math, os

# ---------- 托盘 & 乐高参数 ----------
GRID_N = 10                  # 10x10
GRID_SPACING = 0.020         # 2 cm 间距
TRAY_INNER_HALF = 0.20       # 托盘内半宽，对应 tray_floor size="0.20 0.20 0.005"
FLOOR_TOP_Z = 0.010          # 托盘面顶面高度（pos z=0.005，厚度0.005 → 顶面 ~0.010）
LEGO_HALF_Z = 0.009          # 乐高半高（约 9mm）
BASE_EPS = 0.0005            # 与地面微小缝隙，防数值抖动
BASE_Z = FLOOR_TOP_Z + LEGO_HALF_Z + BASE_EPS

DROP_COUNT = 100
DROP_MIN_DIST = 0.022        # 掉落体之间最小间距（>2.2cm）
DROP_AREA_HALF = 0.165       # 掉落采样 XY 半径（留边距，别太靠墙）
DROP_Z = 0.28                # 掉落初始高度
RAND_SEED = 42
random.seed(RAND_SEED)

def color():
    return f"{random.uniform(0.2,0.9):.2f} {random.uniform(0.2,0.9):.2f} {random.uniform(0.2,0.9):.2f} 1"

def safe_xy(x, y):
    return abs(x) <= (TRAY_INNER_HALF - 0.01) and abs(y) <= (TRAY_INNER_HALF - 0.01)

def gen_grid_bases():
    out = []
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
            out.append(f'''  <body name="{name}" pos="{x:.4f} {y:.4f} {BASE_Z:.4f}">
    <!-- 静态：无 freejoint -->
    <geom type="sdf" name="{gname}" mesh="lego" rgba="{rgba}" friction="2.0 0.01 0.00001">
      <plugin instance="sdflego"/>
    </geom>
  </body>''')
    return "\n".join(out)

def too_close(p, pts, dmin):
    x0, y0 = p
    for x, y in pts:
        if (x - x0)**2 + (y - y0)**2 < dmin**2:
            return True
    return False

def gen_drop_positions(n, half, dmin, max_try=10000):
    pts, tries = [], 0
    while len(pts) < n and tries < max_try:
        tries += 1
        x = random.uniform(-half, half)
        y = random.uniform(-half, half)
        if not safe_xy(x, y):  # 在托盘内
            continue
        if too_close((x, y), pts, dmin):
            continue
        pts.append((x, y))
    return pts

def gen_drops():
    out = []
    pts = gen_drop_positions(DROP_COUNT, DROP_AREA_HALF, DROP_MIN_DIST)
    if len(pts) < DROP_COUNT:
        print(f"[WARN] 仅放置 {len(pts)} / {DROP_COUNT} 个（可调小 DROP_MIN_DIST 或增大 DROP_AREA_HALF）")
    for i, (x, y) in enumerate(pts):
        yaw_deg = random.uniform(0, 360.0)
        name = f"lego_drop_{i:03d}"
        gname = f"{name}_geom"
        rgba = color()
        out.append(f'''  <body name="{name}" pos="{x:.4f} {y:.4f} {DROP_Z:.4f}" euler="0 0 {yaw_deg:.2f}">
    <freejoint/>
    <inertial pos="0 0 0" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
    <geom type="sdf" name="{gname}" mesh="lego" rgba="{rgba}" friction="2.0 0.01 0.00001">
      <plugin instance="sdflego"/>
    </geom>
  </body>''')
    return "\n".join(out)

grid_xml = "<!-- 仅片段：放在 <worldbody> 内可 include -->\n" + gen_grid_bases() + "\n"
drops_xml = "<!-- 仅片段：放在 <worldbody> 内可 include -->\n" + gen_drops() + "\n"

with open("lego_grid_10x10.xml", "w", encoding="utf-8") as f:
    f.write(grid_xml)
with open("lego_drops_100.xml", "w", encoding="utf-8") as f:
    f.write(drops_xml)

print("[OK] 生成 lego_grid_10x10.xml 和 lego_drops_100.xml 完成")
print('在主 XML 的 <worldbody> 里加入：\n  <include file="lego_grid_10x10.xml"/>\n  <include file="lego_drops_100.xml"/>')
