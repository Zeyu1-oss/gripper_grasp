import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from pathlib import Path

def check_file_paths():
    """
    检查文件路径并给出建议
    """
    current_dir = Path.cwd()
    print(f"当前工作目录: {current_dir}")
    
    # 检查XML文件
    xml_files = ["gripper.xml", "lego.xml", "combined_scene.xml"]
    for xml_file in xml_files:
        xml_path = current_dir / xml_file
        print(f"  {xml_file}: {'存在' if xml_path.exists() else '不存在'} - {xml_path}")
    
    # 检查网格文件目录
    mesh_dirs = ["meshes", "../meshes", "assets", "../assets", "models"]
    mesh_files = ["base_link.STL", "left_link.STL", "right_link.STL", "lego.obj"]
    
    print(f"\n网格文件搜索:")
    for mesh_dir in mesh_dirs:
        mesh_path = current_dir / mesh_dir
        print(f"  检查目录: {mesh_path}")
        if mesh_path.exists():
            print(f"    目录存在，包含文件:")
            try:
                for file in mesh_path.iterdir():
                    if file.suffix.lower() in ['.stl', '.obj']:
                        print(f"      - {file.name}")
            except:
                print(f"      无法访问目录")
        else:
            print(f"    目录不存在")
    
    # 建议目录结构
    print(f"\n建议的目录结构:")
    print(f"  {current_dir}/")
    print(f"  ├── gripper.xml")
    print(f"  ├── lego.xml") 
    print(f"  ├── combined_scene.xml")
    print(f"  ├── meshes/")
    print(f"  │   ├── base_link.STL")
    print(f"  │   ├── left_link.STL")
    print(f"  │   ├── right_link.STL")
    print(f"  │   └── lego.obj")
    print(f"  └── visualization_script.py")

def create_config_xml():
    """
    根据实际路径创建配置文件
    """
    current_dir = Path.cwd()
    
    # 寻找网格文件
    mesh_files = {
        'base_link.STL': None,
        'left_link.STL': None,
        'right_link.STL': None,
        'lego.obj': None
    }
    
    search_dirs = [
        current_dir,
        current_dir / "meshes",
        current_dir / "assets", 
        current_dir / "models",
        current_dir / "..",
        current_dir / "../meshes",
        current_dir / "../assets"
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for mesh_file in mesh_files.keys():
                if mesh_files[mesh_file] is None:  # 如果还没找到
                    mesh_path = search_dir / mesh_file
                    if mesh_path.exists():
                        # 计算相对路径
                        try:
                            rel_path = os.path.relpath(mesh_path, current_dir)
                            mesh_files[mesh_file] = rel_path
                            print(f"找到 {mesh_file}: {rel_path}")
                        except:
                            mesh_files[mesh_file] = str(mesh_path)
                            print(f"找到 {mesh_file}: {mesh_path} (绝对路径)")
    
    # 显示未找到的文件
    missing_files = [f for f, path in mesh_files.items() if path is None]
    if missing_files:
        print(f"\n未找到以下文件: {missing_files}")
        return False
    
    # 生成正确的XML配置
    print(f"\n生成更新后的XML文件...")
    return mesh_files
    """
    加载并可视化抓取器和LEGO组件
    """
    
    # 方法1: 分别加载两个组件
    try:
        print("加载抓取器模型...")
        gripper_model = mujoco.MjModel.from_xml_path("gripper.xml")
        gripper_data = mujoco.MjData(gripper_model)
        
        print("加载LEGO模型...")
        lego_model = mujoco.MjModel.from_xml_path("lego.xml")
        lego_data = mujoco.MjData(lego_model)
        
        # 可视化抓取器
        print("可视化抓取器...")
        with mujoco.viewer.launch_passive(gripper_model, gripper_data) as viewer:
            # 运行仿真
            for _ in range(1000):
                mujoco.mj_step(gripper_model, gripper_data)
                viewer.sync()
                time.sleep(0.001)
        
        # 可视化LEGO
        print("可视化LEGO...")
        with mujoco.viewer.launch_passive(lego_model, lego_data) as viewer:
            # 运行仿真
            for _ in range(1000):
                mujoco.mj_step(lego_model, lego_data)
                viewer.sync()
                time.sleep(0.001)
                
    except Exception as e:
        print(f"加载独立组件失败: {e}")
    
    # 方法2: 加载组合场景
    try:
        print("\n加载组合场景...")
        combined_model = mujoco.MjModel.from_xml_path("combined_scene.xml")
        combined_data = mujoco.MjData(combined_model)
        
        print("可视化组合场景...")
        with mujoco.viewer.launch_passive(combined_model, combined_data) as viewer:
            # 运行仿真
            for step in range(5000):
                # 简单的控制示例
                if step < 1000:
                    # 移动抓取器到LEGO上方
                    combined_data.ctrl[0] = 0.0  # x
                    combined_data.ctrl[1] = 0.0  # y
                    combined_data.ctrl[2] = 0.1  # z
                elif step < 2000:
                    # 下降
                    combined_data.ctrl[2] = 0.05
                elif step < 3000:
                    # 抓取
                    combined_data.ctrl[6] = 0.5  # 夹爪
                else:
                    # 抬起
                    combined_data.ctrl[2] = 0.15
                
                mujoco.mj_step(combined_model, combined_data)
                viewer.sync()
                time.sleep(0.001)
                
    except Exception as e:
        print(f"加载组合场景失败: {e}")

def print_model_info():
    """
    打印模型信息
    """
    try:
        gripper_model = mujoco.MjModel.from_xml_path("gripper.xml")
        print(f"抓取器模型:")
        print(f"  - 关节数量: {gripper_model.njnt}")
        print(f"  - 执行器数量: {gripper_model.nu}")
        print(f"  - 传感器数量: {gripper_model.nsensor}")
        print(f"  - 几何体数量: {gripper_model.ngeom}")
        
        lego_model = mujoco.MjModel.from_xml_path("lego.xml")
        print(f"\nLEGO模型:")
        print(f"  - 关节数量: {lego_model.njnt}")
        print(f"  - 执行器数量: {lego_model.nu}")
        print(f"  - 传感器数量: {lego_model.nsensor}")
        print(f"  - 几何体数量: {lego_model.ngeom}")
        
    except Exception as e:
        print(f"获取模型信息失败: {e}")
        print("检查当前目录是否包含 gripper.xml 和 lego.xml 文件")

def interactive_control():
    """
    交互式控制示例
    """
    try:
        model = mujoco.MjModel.from_xml_path("combined_scene.xml")
        data = mujoco.MjData(model)
        
        print("交互式控制 - 使用键盘控制抓取器:")
        print("W/S: 前后移动")
        print("A/D: 左右移动")
        print("Q/E: 上下移动")
        print("空格: 开合夹爪")
        print("ESC: 退出")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)
                
    except Exception as e:
        print(f"交互式控制失败: {e}")

if __name__ == "__main__":
    print("MuJoCo 组件可视化脚本")
    print("=" * 50)
    
    # 首先检查文件路径
    check_file_paths()
    
    # 打印模型信息
    print_model_info()
    
    print("\n开始可视化...")
    load_and_visualize_components()
    
    # 可选：运行交互式控制
    # interactive_control()