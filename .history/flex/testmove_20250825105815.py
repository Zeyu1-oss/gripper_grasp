import random

N = 100
lego_size = 0.01  # 1cm
tray_half_size = 0.18  # 给 LEGO 留点边界 (tray=0.20, 留 2cm margin)
min_dist = 0.02   # 保证不重叠 (两倍 LEGO size)

positions = []

def is_valid(pos, positions):
    for (x, y, z) in positions:
        dx = pos[0] - x
        dy = pos[1] - y
        dz = pos[2] - z
        if (dx*dx + dy*dy + dz*dz) < (min_dist**2):
            return False
    return True

# 生成 100 个不重叠的 lego
while len(positions) < N:
    x = random.uniform(-tray_half_size, tray_half_size)
    y = random.uniform(-tray_half_size, tray_half_size)
    z = random.uniform(0.05, 0.15)  # 高度随便分布在托盘上方
    pos = (x, y, z)
    if is_valid(pos, positions):
        positions.append(pos)

# 输出 XML
for i, (x, y, z) in enumerate(positions):
    print(f"""
    <body name="lego_{i:03d}" pos="{x:.3f} {y:.3f} {z:.3f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
      <geom type="sdf" name="lego_{i:03d}_geom" 
            friction="2.0 0.01 0.00001" mesh="lego"  
            rgba="{random.random():.2f} {random.random():.2f} {random.random():.2f} 1">
        <plugin instance="sdflego"/>
      </geom>
    </body>
    """)
