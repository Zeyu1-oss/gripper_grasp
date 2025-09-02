import numpy as np

def generate_random_legos(num_legos=100):
    """生成随机位置的乐高XML代码"""
    colors = [
        "0.9 0.3 0.1 1", "0.9 0.6 0.1 1", "0.2 0.7 0.2 1", "0.2 0.5 0.9 1",
        "0.6 0.3 0.9 1", "0.3 0.8 0.5 1", "0.8 0.2 0.5 1", "0.5 0.5 0.1 1",
        "0.1 0.8 0.8 1", "0.9 0.1 0.9 1"
    ]
    
    xml_code = ""
    for i in range(num_legos):
        # 随机位置：x,y在[-0.3, 0.3]范围内，z在[1.0, 1.5]范围内
        x = np.random.uniform(-0.3, 0.3)
        y = np.random.uniform(-0.3, 0.3)
        z = np.random.uniform(1.0, 1.5)
        color = colors[i % len(colors)]
        
        xml_code += f'''
    <body name="lego_{i:02d}" pos="{x:.3f} {y:.3f} {z:.3f}">
      <freejoint/>
      <geom type="sdf" name="lego_{i:02d}_geom" friction="2.0 0.01 0.00001" mesh="lego" rgba="{color}">
        <plugin instance="sdflego"/>
      </geom>
    </body>\n'''
    
    return xml_code

# 生成100个随机乐高
lego_xml = generate_random_legos(100)
print(lego_xml)