"""
指标收集器 - 用于收集和记录测试过程中的各种性能参数
"""
import time
import csv
import os
import numpy as np
import math
from collections import defaultdict, deque
import copy


class MetricsCollector:
    def __init__(self, map_index, env, save_dir="results/detailed_metrics"):
        self.map_index = map_index
        self.env = env
        self.save_dir = save_dir
        self.csv_filename = f"{save_dir}/map_{map_index}_metrics.csv"
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化参数
        self.init_metrics()
        
        # CSV文件头
        self.field_names = [
            'step', 'episode', 'timestamp', 'map_index',
            'robot_x', 'robot_y', 'target_x', 'target_y',
            'total_distance', 'step_distance', 'covered_area', 'exploration_rate',
            'robot_count', 'max_travel_distance', 'current_velocity', 'avg_velocity',
            'current_acceleration', 'avg_acceleration', 'current_jerk', 'avg_jerk',
            'covered_cells_count', 'total_free_cells', 'CR', 'redundant_cells',
            'total_visited_cells', 'SR', 'collision_count', 'computation_time_step',
            'total_computation_time', 'task_time', 'success_rate', 'reward'
        ]
        
        # 初始化CSV文件
        self.init_csv()
        
    def init_metrics(self):
        """初始化所有指标"""
        # 时间相关
        self.task_start_time = time.time()
        self.last_position_time = time.time()
        self.total_computation_time = 0.0
        
        # 位置和轨迹
        self.trajectory_points = []  # [(x, y, timestamp), ...]
        self.robot_positions = []
        self.last_position = None
        
        # 速度和加速度
        self.velocities = deque(maxlen=100)  # 保留最近100个速度值
        self.accelerations = deque(maxlen=100)
        self.jerks = deque(maxlen=100)
        self.last_velocity = 0.0
        self.last_acceleration = 0.0
        
        # 覆盖率相关
        self.covered_cells = set()  # 存储已覆盖的栅格坐标
        self.cell_visit_count = defaultdict(int)  # 每个栅格的访问次数
        self.robot_radius = 10  # 机器人半径（像素）
        self.cell_size = 1  # 栅格尺寸（像素）
        
        # 碰撞检测
        self.collision_count = 0
        self.last_collision_time = 0
        self.collision_cooldown = 0.1
        
        # 任务相关
        self.step_count = 0
        self.episode_count = 0
        self.success_rate = 0.0
        self.robot_count = 1  # 单机器人系统
        
        # 计算总的可移动区域
        self.total_free_cells = np.sum(self.env.ground_truth == 255)
        
    def init_csv(self):
        """初始化CSV文件"""
        # 如果文件不存在，写入表头
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.field_names)
                
    def update_position(self, new_position, reward=0.0, done=False):
        """更新机器人位置并计算相关指标"""
        current_time = time.time()
        
        # 记录轨迹点
        self.trajectory_points.append((new_position[0], new_position[1], current_time))
        self.robot_positions.append(new_position.copy())
        
        # 计算移动距离
        step_distance = 0.0
        if self.last_position is not None:
            step_distance = np.linalg.norm(new_position - self.last_position)
            
            # 计算速度
            time_diff = current_time - self.last_position_time
            if time_diff > 0:
                velocity = step_distance / time_diff
                self.velocities.append(velocity)
                
                # 计算加速度
                if len(self.velocities) >= 2:
                    acceleration = (velocity - self.last_velocity) / time_diff
                    self.accelerations.append(acceleration)
                    
                    # 计算加加速度（jerk）
                    if len(self.accelerations) >= 2:
                        jerk = (acceleration - self.last_acceleration) / time_diff
                        self.jerks.append(jerk)
                        self.last_acceleration = acceleration
                        
                    self.last_velocity = velocity
        
        # 更新覆盖的栅格
        self.update_covered_cells(new_position)
        
        # 检测碰撞
        self.check_collision(new_position, current_time)
        
        # 更新任务状态
        if done:
            self.success_rate = 1.0
            
        self.last_position = new_position.copy()
        self.last_position_time = current_time
        
        return step_distance
        
    def update_covered_cells(self, position):
        """更新机器人覆盖的栅格"""
        x, y = int(position[0]), int(position[1])
        
        # 计算机器人覆盖的栅格范围（圆形区域）
        for dx in range(-self.robot_radius, self.robot_radius + 1):
            for dy in range(-self.robot_radius, self.robot_radius + 1):
                if dx*dx + dy*dy <= self.robot_radius**2:
                    cell_x, cell_y = x + dx, y + dy
                    # 检查是否在地图范围内
                    if (0 <= cell_x < self.env.ground_truth.shape[1] and 
                        0 <= cell_y < self.env.ground_truth.shape[0]):
                        # 检查是否为可移动区域
                        if self.env.ground_truth[cell_y, cell_x] == 255:
                            cell_coord = (cell_x, cell_y)
                            self.covered_cells.add(cell_coord)
                            self.cell_visit_count[cell_coord] += 1
                            
    def check_collision(self, position, current_time):
        """检查碰撞"""
        x, y = int(position[0]), int(position[1])
        
        # 检查机器人中心位置是否碰撞到障碍物
        if (0 <= x < self.env.ground_truth.shape[1] and 
            0 <= y < self.env.ground_truth.shape[0]):
            if self.env.ground_truth[y, x] == 1:  # 障碍物
                if current_time - self.last_collision_time > self.collision_cooldown:
                    self.collision_count += 1
                    self.last_collision_time = current_time
                    
    def calculate_coverage_rate(self):
        """计算覆盖率 CR"""
        if self.total_free_cells > 0:
            return len(self.covered_cells) / self.total_free_cells
        return 0.0
        
    def calculate_redundancy_rate(self):
        """计算清扫冗余度 SR"""
        total_visited_area = len(self.cell_visit_count)
        if total_visited_area == 0:
            return 0.0
            
        redundant_area = sum(max(0, count - 1) for count in self.cell_visit_count.values())
        return redundant_area / total_visited_area
        
    def calculate_total_distance(self):
        """计算总移动距离"""
        if len(self.trajectory_points) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(1, len(self.trajectory_points)):
            p1 = self.trajectory_points[i-1]
            p2 = self.trajectory_points[i]
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_distance += distance
        return total_distance
        
    def get_current_metrics(self, step, episode, reward=0.0, computation_time=0.0):
        """获取当前所有指标"""
        current_time = time.time()
        task_time = current_time - self.task_start_time
        total_distance = self.calculate_total_distance()
        
        # 当前位置
        current_pos = self.robot_positions[-1] if self.robot_positions else [0, 0]
        
        # 计算步距离
        step_distance = 0.0
        if len(self.robot_positions) >= 2:
            step_distance = np.linalg.norm(
                np.array(self.robot_positions[-1]) - np.array(self.robot_positions[-2])
            )
        
        # 速度相关指标
        current_velocity = self.velocities[-1] if self.velocities else 0.0
        avg_velocity = np.mean(self.velocities) if self.velocities else 0.0
        
        # 加速度相关指标
        current_acceleration = self.accelerations[-1] if self.accelerations else 0.0
        avg_acceleration = np.mean(np.abs(self.accelerations)) if self.accelerations else 0.0
        
        # 加加速度相关指标
        current_jerk = self.jerks[-1] if self.jerks else 0.0
        avg_jerk = np.mean(np.abs(self.jerks)) if self.jerks else 0.0
        
        # 覆盖率指标
        coverage_rate = self.calculate_coverage_rate()
        redundancy_rate = self.calculate_redundancy_rate()
        
        # 更新计算时间
        self.total_computation_time += computation_time
        
        metrics = {
            'step': step,
            'episode': episode,
            'timestamp': current_time,
            'map_index': self.map_index,
            'robot_x': current_pos[0],
            'robot_y': current_pos[1],
            'target_x': self.env.target_position[0],
            'target_y': self.env.target_position[1],
            'total_distance': total_distance,
            'step_distance': step_distance,
            'covered_area': len(self.covered_cells),
            'exploration_rate': self.env.explored_rate,
            'robot_count': self.robot_count,
            'max_travel_distance': total_distance,  # 单机器人情况下等同于总距离
            'current_velocity': current_velocity,
            'avg_velocity': avg_velocity,
            'current_acceleration': current_acceleration,
            'avg_acceleration': avg_acceleration,
            'current_jerk': current_jerk,
            'avg_jerk': avg_jerk,
            'covered_cells_count': len(self.covered_cells),
            'total_free_cells': self.total_free_cells,
            'CR': coverage_rate,
            'redundant_cells': sum(max(0, count - 1) for count in self.cell_visit_count.values()),
            'total_visited_cells': len(self.cell_visit_count),
            'SR': redundancy_rate,
            'collision_count': self.collision_count,
            'computation_time_step': computation_time,
            'total_computation_time': self.total_computation_time,
            'task_time': task_time,
            'success_rate': self.success_rate,
            'reward': reward
        }
        
        return metrics
        
    def save_step_metrics(self, step, episode, reward=0.0, computation_time=0.0):
        """保存单步指标到CSV"""
        metrics = self.get_current_metrics(step, episode, reward, computation_time)
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [metrics[field] for field in self.field_names]
            writer.writerow(row)
            
    def print_summary(self):
        """打印指标摘要"""
        total_distance = self.calculate_total_distance()
        coverage_rate = self.calculate_coverage_rate()
        redundancy_rate = self.calculate_redundancy_rate()
        avg_velocity = np.mean(self.velocities) if self.velocities else 0.0
        avg_acceleration = np.mean(np.abs(self.accelerations)) if self.accelerations else 0.0
        avg_jerk = np.mean(np.abs(self.jerks)) if self.jerks else 0.0
        
        print(f"\n=== 地图 {self.map_index} 测试结果摘要 ===")
        print(f"总移动距离: {total_distance:.2f} 像素")
        print(f"覆盖率 (CR): {coverage_rate:.4f}")
        print(f"清扫冗余度 (SR): {redundancy_rate:.4f}")
        print(f"碰撞次数: {self.collision_count}")
        print(f"探索率: {self.env.explored_rate:.4f}")
        print(f"平均速度: {avg_velocity:.2f} 像素/秒")
        print(f"平均加速度: {avg_acceleration:.2f} 像素/秒²")
        print(f"平均加加速度: {avg_jerk:.2f} 像素/秒³")
        print(f"任务完成率: {self.success_rate:.2f}")
        print(f"总计算时间: {self.total_computation_time:.4f} 秒")
        print(f"数据已保存到: {self.csv_filename}")
        print("=" * 50)
