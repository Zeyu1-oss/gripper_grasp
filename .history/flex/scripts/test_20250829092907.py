import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET

def merge_xml_models(gripper_xml_path, object_xml_path, output_path=None):
    """
    将两个XML模型合并到一个场景中
    
    Args:
        gripper_xml_path: 夹爪XML文件路径
        object_xml_path: 物体XML文件路径  
        output_path: 可选的输出文件路径
    
    Returns:
        合并后的XML字符串
    """
    
    # 读取夹爪XML
    gripper_tree = ET.parse(gripper_xml_path)
    gripper_root = gripper_tree.getroot()
    
    # 读取物体XML
    object_tree = ET.parse(object_xml_path)
    object_root = object_tree.getroot()
    
    # 创建新的合并XML
    merged_root = ET.Element("mujoco")
    merged_root.set("model", "Merged_Gripper_Object")
    
    # 合并所有非worldbody的元素
    sections_to_merge = ['extension', 'asset', 'default', 'option']
    
    for section in sections_to_merge:
        # 从夹爪XML添加
        gripper_section = gripper_root.find(section)
        if gripper_section is not None:
            merged_root.append(gripper_section)
        
        # 从物体XML合并asset
        if section == 'asset':
            object_section = object_root.find(section)
            if object_section is not None:
                existing_asset = merged_root.find('asset')
                if existing_asset is not None:
                    # 合并asset内容
                    for child in object_section:
                        existing_asset.append(child)
                else:
                    merged_root.append(object_section)
    
    # 创建worldbody
    worldbody = ET.Element("worldbody")
    
    # 添加夹爪body
    gripper_worldbody = gripper_root.find('worldbody')
    if gripper_worldbody is not None:
        for body in gripper_worldbody:
            worldbody.append(body)
    
    # 添加物体body
    object_worldbody = object_root.find('worldbody')
    if object_worldbody is not None:
        for body in object_worldbody:
            # 可以在这里调整物体位置
            if body.tag == 'body' and body.get('name') == 'lego':
                body.set('pos', '0 0 0.2')  # 将LEGO放在抓取器上方
            worldbody.append(body)
    
    merged_root.append(worldbody)
    
    # 添加其他部分
    other_sections = ['equality', 'tendon', 'actuator', 'sensor']
    for section in other_sections:
        gripper_section = gripper_root.find(section)
        if gripper_section is not None:
            merged_root.append(gripper_section)
            
        # 合并物体的传感器
        if section == 'sensor':
            object_section = object_root.find(section)
            if object_section is not None:
                existing_sensor = merged_root.find('sensor')
                if existing_sensor is not None:
                    for sensor in object_section:
                        existing_sensor.append(sensor)
    
    # 转换为字符串
    xml_string = ET.tostring(merged_root, encoding='unicode')
    
    # 格式化XML（添加声明）
    formatted_xml = '<?xml version="1.0"?>\n' + xml_string
    
    # 可选：保存到文件
    if output_path:
        with open(output_path, 'w') as f:
            f.write(formatted_xml)
        print(f"合并后的XML已保存到: {output_path}")
    
    return formatted_xml

def visualize_merged_scene(gripper_xml, object_xml):
    """
    可视化合并后的场景
    """
    try:
        print("正在合并XML模型...")
        merged_xml = merge_xml_models(gripper_xml, object_xml, "merged_scene.xml")
        
        print("加载合并后的模型...")
        model = mujoco.MjModel.from_xml_string(merged_xml)
        data = mujoco.MjData(model)
        
        print(f"模型信息:")
        print(f"  - 关节数量: {model.njnt}")
        print(f"  - 执行器数量: {model.nu}")
        print(f"  - 几何体数量: {model.ngeom}")
        print(f"  - 物体数量: {model.nbody}")
        
        print("\n启动可视化器...")
        print("按 ESC 或关闭窗口退出")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
        
        print("可视化完成")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查XML文件路径和内容")

if __name__ == "__main__":
    print("MuJoCo双模型加载器")
    print("=" * 30)
    
    # 设置XML文件路径
    gripper_xml = "../mjcf/gripper.xml"  # 夹爪XML路径
    object_xml = "../mjcf/lego.xml"      # 物体XML路径
    
    print(f"夹爪XML: {gripper_xml}")
    print(f"物体XML: {object_xml}")
    
    # 可视化
    visualize_merged_scene(gripper_xml, object_xml)