#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import mujoco
import time
from scipy.spatial.transform import Rotation

class GripperController:
    def __init__(self, model_path, grasp_json_path):
        """
        初始化夹爪控制器
        :param model_path: MuJoCo模型文件路径
        :param grasp_json_path: 抓取姿态JSON文件路径
        """
        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 加载抓取姿态
        self.grasp_poses = self.load_grasp_poses(grasp_json_path)
        self.current_grasp_idx = 0
        
        # 创建仿真器
        self.renderer = mujoco.Renderer(self.model)
        
        # 控制参数
        self.position_kp = 800.0
        self.orientation_kp = 200.0
        self.gripper_kp = 500.0
        
        # 状态变量
        self.phase = "approach"  # approach, grasp, lift, verify
        self.phase_start_time = 0
        self.initial_object_pos = None
        
        print(f"Loaded {len(self.grasp_poses)} grasp poses")
    
    def load_grasp_poses(self, json_path):
        """从JSON文件加载抓取姿态"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        grasps = []
        for grasp in data.get('grasps', []):
            grasps.append({
                'position': np.array(grasp['position']),
                'quaternion': np.array(grasp['quaternion_xyzw']),  # [x, y, z, w]
                'width': grasp['width'],
                'score': grasp.get('score', 0.0)
            })
        
        # 按评分排序
        grasps.sort(key=lambda x: x['score'], reverse=True)
        return grasps
    
    def quaternion_to_euler(self, quat_xyzw):
        """四元数(xyzw)转欧拉角"""
        # 转换为wxyz格式
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        rot = Rotation.from_quat(quat_wxyz)
        return rot.as_euler('xyz')
    
    def euler_to_quaternion(self, euler):
        """欧拉角转四元数(xyzw)"""
        rot = Rotation.from_euler('xyz', euler)
        quat_wxyz = rot.as_quat()  # [x, y, z, w]
        return np.array([quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]])
    
    def set_6d_pose(self, target_pos, target_quat_xyzw):
        """
        设置6D目标姿态
        :param target_pos: 目标位置 [x, y, z]
        :param target_quat_xyzw: 目标四元数 [x, y, z, w]
        """
        # 设置位置
        self.data.ctrl[0] = target_pos[0]  # x
        self.data.ctrl[1] = target_pos[1]  # y
        self.data.ctrl[2] = target_pos[2]  # z
        
        # 四元数转欧拉角进行控制
        euler = self.quaternion_to_euler(target_quat_xyzw)
        self.data.ctrl[3] = euler[0]  # roll
        self.data.ctrl[4] = euler[1]  # pitch
        self.data.ctrl[5] = euler[2]  # yaw
    
    def set_gripper_width(self, width):
        """设置夹爪宽度"""
        self.data.ctrl[6] = width
    
    def get_current_pose(self):
        """获取当前夹爪姿态"""
        pos = np.array([
            self.data.sensor('x_pos').data[0],
            self.data.sensor('y_pos').data[0],
            self.data.sensor('z_pos').data[0]
        ])
        
        euler = np.array([
            self.data.sensor('roll_pos').data[0],
            self.data.sensor('pitch_pos').data[0],
            self.data.sensor('yaw_pos').data[0]
        ])
        
        quat = self.euler_to_quaternion(euler)
        return pos, quat
    
    def get_object_position(self):
        """获取物体当前位置"""
        return self.data.sensor('lego_position').data.copy()
    
    def get_object_orientation(self):
        """获取物体当前姿态"""
        return self.data.sensor('lego_orientation').data.copy()
    
    def check_contact(self):
        """检查接触状态"""
        left_contact = self.data.sensor('left_contact').data[0] > 0.1
        right_contact = self.data.sensor('right_contact').data[0] > 0.1
        return left_contact and right_contact
    
    def check_grasp_success(self):
        """检查抓取是否成功"""
        if not self.check_contact():
            return False
        
        # 检查物体是否被提起（位置发生变化）
        current_pos = self.get_object_position()
        if self.initial_object_pos is None:
            self.initial_object_pos = current_pos.copy()
            return False
        
        height_diff = current_pos[2] - self.initial_object_pos[2]
        return height_diff > 0.02  # 至少提起2cm
    
    def approach_phase(self, grasp_pose):
        """接近阶段"""
        target_pos = grasp_pose['position'].copy()
        target_pos[2] += 0.05  # 在目标上方5cm处接近
        
        self.set_6d_pose(target_pos, grasp_pose['quaternion'])
        self.set_gripper_width(grasp_pose['width'] + 0.005)  # 稍微张开一点
        
        current_pos, _ = self.get_current_pose()
        distance = np.linalg.norm(current_pos - target_pos)
        
        if distance < 0.005:  # 接近完成
            self.phase = "grasp"
            self.phase_start_time = self.data.time
            print("Approach complete, starting grasp")
        
        return distance < 0.005
    
    def grasp_phase(self, grasp_pose):
        """抓取阶段"""
        # 缓慢下降到抓取位置
        target_pos = grasp_pose['position'].copy()
        current_pos, current_quat = self.get_current_pose()
        
        # 线性插值下降
        progress = min(1.0, (self.data.time - self.phase_start_time) / 2.0)
        approach_pos = current_pos + (target_pos - current_pos) * progress
        
        self.set_6d_pose(approach_pos, grasp_pose['quaternion'])
        
        if progress >= 0.8:
            # 开始闭合夹爪
            close_progress = min(1.0, (self.data.time - self.phase_start_time - 1.6) / 0.4)
            grip_width = grasp_pose['width'] + 0.005 * (1 - close_progress)
            self.set_gripper_width(grip_width)
        
        if progress >= 1.0 and self.check_contact():
            self.phase = "lift"
            self.phase_start_time = self.data.time
            self.initial_object_pos = self.get_object_position()
            print("Grasp complete, starting lift")
            return True
        
        return False
    
    def lift_phase(self):
        """提升阶段"""
        current_pos, current_quat = self.get_current_pose()
        lift_pos = current_pos.copy()
        lift_pos[2] += 0.01  # 每次提升1cm
        
        self.set_6d_pose(lift_pos, current_quat)
        
        # 保持夹爪闭合
        self.set_gripper_width(self.data.ctrl[6])
        
        # 检查是否达到提升高度
        if current_pos[2] > self.initial_object_pos[2] + 0.1:
            self.phase = "verify"
            self.phase_start_time = self.data.time
            print("Lift complete, starting verification")
            return True
        
        return False
    
    def verify_phase(self):
        """验证阶段"""
        # 保持当前位置，检查抓取是否稳定
        current_pos, current_quat = self.get_current_pose()
        self.set_6d_pose(current_pos, current_quat)
        self.set_gripper_width(self.data.ctrl[6])
        
        # 验证抓取是否成功
        success = self.check_grasp_success()
        
        if self.data.time - self.phase_start_time > 2.0:  # 验证2秒
            if success:
                print("Grasp verification: SUCCESS!")
            else:
                print("Grasp verification: FAILED!")
            return True
        
        return False
    
    def run_grasp_sequence(self, grasp_idx):
        """运行完整的抓取序列"""
        if grasp_idx >= len(self.grasp_poses):
            print("No more grasp poses to try")
            return False
        
        grasp_pose = self.grasp_poses[grasp_idx]
        print(f"\n=== Trying grasp {grasp_idx + 1}/{len(self.grasp_poses)} ===")
        print(f"Position: {grasp_pose['position']}")
        print(f"Width: {grasp_pose['width']:.4f}")
        print(f"Score: {grasp_pose['score']:.3f}")
        
        self.phase = "approach"
        self.phase_start_time = self.data.time
        self.initial_object_pos = None
        
        while True:
            # 步进仿真
            mujoco.mj_step(self.model, self.data)
            
            # 更新渲染
            self.renderer.update_scene(self.data)
            
            # 执行当前阶段的控制
            if self.phase == "approach":
                self.approach_phase(grasp_pose)
            elif self.phase == "grasp":
                self.grasp_phase(grasp_pose)
            elif self.phase == "lift":
                if self.lift_phase():
                    break
            elif self.phase == "verify":
                if self.verify_phase():
                    break
            
            # 添加小延迟以便观察
            time.sleep(0.001)
        
        return self.check_grasp_success()
    
    def run_all_grasps(self):
        """尝试所有抓取姿态"""
        success_count = 0
        
        for i in range(len(self.grasp_poses)):
            # 重置仿真
            mujoco.mj_resetData(self.model, self.data)
            
            # 运行抓取序列
            success = self.run_grasp_sequence(i)
            if success:
                success_count += 1
                print(f"Grasp {i+1} succeeded!")
            else:
                print(f"Grasp {i+1} failed.")
            
            # 短暂暂停
            time.sleep(1.0)
        
        print(f"\n=== Final Results ===")
        print(f"Successful grasps: {success_count}/{len(self.grasp_poses)}")
        print(f"Success rate: {success_count/len(self.grasp_poses)*100:.1f}%")

def main():
    # 配置文件路径
    model_path = "gripper_sdf.xml"  # 修改为您的XML文件路径
    grasp_json_path = "top_grasps.json"  # 修改为您的JSON文件路径
    
    # 创建控制器
    controller = GripperController(model_path, grasp_json_path)
    
    try:
        # 运行所有抓取尝试
        controller.run_all_grasps()
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        print("Experiment completed")

if __name__ == "__main__":
    main()