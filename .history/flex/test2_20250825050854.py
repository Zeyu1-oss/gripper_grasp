import numpy as np
import mujoco
from mujoco import mjtObj

def name2id(model, objtype, name):
    """根据名称获取ID"""
    try:
        return mujoco.mj_name2id(model, objtype, name)
    except Exception:
        return None

def print_lego_rotation_matrices(xml_file="g2_with_30lego.xml"):
    """打印0-9号乐高积木的旋转矩阵"""
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)
    
    print("=== 乐高积木旋转矩阵到世界坐标系 ===")
    print("格式: R = [[r11, r12, r13],")
    print("          [r21, r22, r23],")
    print("          [r31, r32, r33]]")
    print("其中每一列代表物体自身坐标轴在世界坐标系中的方向")
    print("=" * 60)
    
    # 检查0-9号乐高积木
    for i in range(10):
        lego_name = f"lego_{i:02d}_geom"  # 格式: lego_00_geom, lego_01_geom, ...
        lego_body_name = f"lego_{i:02d}"   # 对应的body名称
        
        # 获取几何体ID
        gid = name2id(model, mjtObj.mjOBJ_GEOM, lego_name)
        bid = name2id(model, mjtObj.mjOBJ_BODY, lego_body_name)
        
        if gid is None:
            print(f"乐高 {i}: {lego_name} 不存在")
            continue
        
        # 前向计算获取当前状态
        mujoco.mj_forward(model, data)
        
        # 获取旋转矩阵 (3x3)
        R = data.geom_xmat[gid].reshape(3, 3)
        
        # 获取位置和尺寸
        pos = data.geom_xpos[gid]
        size = model.geom_size[gid]
        
        print(f"\n乐高 {i} ({lego_name}):")
        print(f"位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"尺寸: [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}]")
        
        # 打印旋转矩阵
        print("旋转矩阵 R (世界坐标系):")
        print(f"[{R[0,0]:+8.4f} {R[0,1]:+8.4f} {R[0,2]:+8.4f}]  ← X轴世界方向: [{R[0,0]:+.3f}, {R[0,1]:+.3f}, {R[0,2]:+.3f}]")
        print(f"[{R[1,0]:+8.4f} {R[1,1]:+8.4f} {R[1,2]:+8.4f}]  ← Y轴世界方向: [{R[1,0]:+.3f}, {R[1,1]:+.3f}, {R[1,2]:+.3f}]")
        print(f"[{R[2,0]:+8.4f} {R[2,1]:+8.4f} {R[2,2]:+8.4f}]  ← Z轴世界方向: [{R[2,0]:+.3f}, {R[2,1]:+.3f}, {R[2,2]:+.3f}]")
        
        # 计算欧拉角（可选）
        try:
            # 从旋转矩阵提取欧拉角 (ZYX顺序)
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(R[2,1], R[2,2])
                y = np.arctan2(-R[2,0], sy)
                z = np.arctan2(R[1,0], R[0,0])
            else:
                x = np.arctan2(-R[1,2], R[1,1])
                y = np.arctan2(-R[2,0], sy)
                z = 0
            
            print(f"欧拉角 (Z-Y-X): [{np.degrees(z):6.1f}°, {np.degrees(y):6.1f}°, {np.degrees(x):6.1f}°]")
        except:
            print("欧拉角计算失败")
        
        print("-" * 50)

def analyze_lego_orientation(xml_file="g2_with_30lego.xml", lego_number=0):
    """详细分析特定乐高积木的朝向"""
    
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)
    
    lego_name = f"lego_{lego_number:02d}_geom"
    gid = name2id(model, mjtObj.mjOBJ_GEOM, lego_name)
    
    if gid is None:
        print(f"乐高 {lego_name} 不存在")
        return
    
    mujoco.mj_forward(model, data)
    R = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    
    print(f"\n=== 乐高 {lego_number} 详细朝向分析 ===")
    
    # 分析每个轴的方向
    axes = ['X', 'Y', 'Z']
    for i, axis in enumerate(axes):
        axis_dir = R[:, i]
        horizontal = np.linalg.norm(axis_dir[:2])  # XY平面分量
        vertical = abs(axis_dir[2])  # Z分量
        
        print(f"\n{axis}轴 (长度: {size[i]:.4f}m):")
        print(f"  世界方向: [{axis_dir[0]:+.3f}, {axis_dir[1]:+.3f}, {axis_dir[2]:+.3f}]")
        print(f"  水平分量: {horizontal:.3f}, 垂直分量: {vertical:.3f}")
        
        if vertical > 0.9:
            print(f"  → 主要垂直方向")
        elif horizontal > 0.9:
            print(f"  → 主要水平方向")
        
        # 计算与地面的夹角
        angle_with_ground = np.degrees(np.arccos(horizontal))
        print(f"  与地面夹角: {angle_with_ground:.1f}°")
    
    # 找出最长的轴
    longest_axis = np.argmax(size)
    longest_dir = R[:, longest_axis]
    print(f"\n最长轴: {axes[longest_axis]} (长度: {size[longest_axis]:.4f}m)")
    print(f"方向: [{longest_dir[0]:+.3f}, {longest_dir[1]:+.3f}, {longest_dir[2]:+.3f}]")

if __name__ == "__main__":
    # 打印所有0-9号乐高的旋转矩阵
    print_lego_rotation_matrices()
    
    # 详细分析第一个乐高的朝向
    print("\n" + "="*60)
    analyze_lego_orientation(lego_number=0)
    
    # 如果需要分析特定乐高，可以取消注释下面这行
    # analyze_lego_orientation(lego_number=7)