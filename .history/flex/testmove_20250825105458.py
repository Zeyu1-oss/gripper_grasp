import random

N = 100
lego_size = 0.01      # 1 cm
grid_step = 0.02      # 栅格间隔 2 cm，保证不会重叠
jitter = 0.003        # 每个格子内的小扰动
z_low, z_high = 0.05, 0.15  # 高度范围

# 生成所有可用格子中心
coords = []
x_vals = [round(x, 3) for x in f range(-0.18, 0.18, grid_step)]
y_vals = [round(y, 3) for y in f range(-0.18, 0.18, grid_step)]
for x in x_vals:
    for y in y_vals:
        coords.append((x, y))

# 随机选 100 个格子
random.shuffle(coords)
positions = coords[:N]

# 加扰动并输出 XML
for i, (x, y) in enumerate(positions):
    xj = x + random.uniform(-jitter, jitter)
    yj = y + random.uniform(-jitter, jitter)
    z  = random.uniform(z_low, z_high)

    print(f"""
    <body name="lego_{i:03d}" pos="{xj:.3f} {yj:.3f} {z:.3f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
      <geom type="sdf" name="lego_{i:03d}_geom" 
            friction="2.0 0.01 0.00001" mesh="lego"  
            rgba="{random.random():.2f} {random.random():.2f} {random.random():.2f} 1">
        <plugin instance="sdflego"/>
      </geom>
    </body>
    """)

# 工具函数
def frange(start, stop, step):
    while start <= stop:
        yield start
        start += step
