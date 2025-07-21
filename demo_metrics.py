#!/usr/bin/env python3
"""
演示指标收集系统的使用方法
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from metrics_collector import MetricsCollector


class DemoEnvironment:
    """演示用的模拟环境"""
    def __init__(self, size=640):
        # 创建一个简单的地图
        self.ground_truth = np.ones((size, size)) * 255  # 全部可移动
        
        # 添加一些障碍物
        self.ground_truth[200:300, 200:300] = 1  # 中央障碍物
        self.ground_truth[400:450, 100:200] = 1  # 其他障碍物
        
        # 设置起始和目标位置
        self.start_position = np.array([50, 50])
        self.target_position = np.array([550, 550])
        self.explored_rate = 0.0
        
        print(f"创建演示地图: {size}x{size}")
        print(f"起始位置: {self.start_position}")
        print(f"目标位置: {self.target_position}")


def simulate_robot_movement():
    """模拟机器人运动轨迹"""
    env = DemoEnvironment()
    collector = MetricsCollector(map_index=999, env=env, save_dir="demo_results")
    
    print("\n开始模拟机器人运动...")
    
    # 模拟一个简单的导航路径
    start = env.start_position
    target = env.target_position
    
    # 生成一个螺旋式的探索路径
    positions = []
    current_pos = start.copy().astype(float)
    
    # 参数化的螺旋路径
    steps = 100
    for i in range(steps):
        # 螺旋运动
        angle = i * 0.2
        radius = i * 2
        
        # 向目标方向移动，同时加上螺旋偏移
        direction = target - current_pos
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 10:
            direction = direction / direction_norm * 5  # 每步移动5个像素
            
            # 添加螺旋偏移
            offset_x = radius * np.cos(angle)
            offset_y = radius * np.sin(angle)
            
            next_pos = current_pos + direction + np.array([offset_x, offset_y]) * 0.1
            
            # 确保位置在地图范围内
            next_pos[0] = np.clip(next_pos[0], 10, env.ground_truth.shape[1] - 10)
            next_pos[1] = np.clip(next_pos[1], 10, env.ground_truth.shape[0] - 10)
            
            positions.append(next_pos.copy())
            current_pos = next_pos
            
            # 增加探索率（模拟）
            env.explored_rate = min(i / steps * 0.85, 0.85)
        else:
            # 到达目标
            positions.append(target.copy())
            env.explored_rate = 1.0
            break
    
    print(f"生成了 {len(positions)} 个路径点")
    
    # 模拟机器人按路径移动并收集指标
    for step, position in enumerate(positions):
        # 模拟计算时间
        computation_start = time.time()
        time.sleep(0.001)  # 模拟算法计算时间
        computation_time = time.time() - computation_start
        
        # 模拟奖励
        distance_to_target = np.linalg.norm(position - target)
        reward = -distance_to_target / 1000  # 距离越近奖励越高
        
        # 检查是否到达目标
        done = distance_to_target < 20
        
        # 更新指标收集器
        collector.update_position(position, reward=reward, done=done)
        collector.save_step_metrics(
            step=step, 
            episode=0, 
            reward=reward, 
            computation_time=computation_time
        )
        
        if done:
            print(f"在第 {step} 步到达目标！")
            break
    
    # 打印最终统计
    collector.print_summary()
    
    return collector, positions


def visualize_demo_results(collector, positions):
    """可视化演示结果"""
    print("\n生成可视化图表...")
    
    # 读取保存的CSV数据
    import pandas as pd
    try:
        data = pd.read_csv(collector.csv_filename)
    except:
        print("无法读取CSV数据，使用内存中的数据")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('指标收集系统演示结果', fontsize=16)
    
    # 1. 轨迹图
    axes[0, 0].plot([p[0] for p in positions], [p[1] for p in positions], 'b-', linewidth=2, label='轨迹')
    axes[0, 0].scatter(positions[0][0], positions[0][1], c='green', s=100, marker='o', label='起点')
    axes[0, 0].scatter(collector.env.target_position[0], collector.env.target_position[1], 
                      c='red', s=100, marker='*', label='终点')
    axes[0, 0].set_title('机器人移动轨迹')
    axes[0, 0].set_xlabel('X坐标')
    axes[0, 0].set_ylabel('Y坐标')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 覆盖率演化
    axes[0, 1].plot(data['step'], data['CR'], 'g-', linewidth=2)
    axes[0, 1].set_title('覆盖率 (CR) 演化')
    axes[0, 1].set_xlabel('步数')
    axes[0, 1].set_ylabel('覆盖率')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 速度演化
    axes[1, 0].plot(data['step'], data['current_velocity'], 'r-', alpha=0.6, label='瞬时速度')
    axes[1, 0].plot(data['step'], data['avg_velocity'], 'k-', linewidth=2, label='平均速度')
    axes[1, 0].set_title('速度演化')
    axes[1, 0].set_xlabel('步数')
    axes[1, 0].set_ylabel('速度 (像素/秒)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 累计距离
    axes[1, 1].plot(data['step'], data['total_distance'], 'purple', linewidth=2)
    axes[1, 1].set_title('累计移动距离')
    axes[1, 1].set_xlabel('步数')
    axes[1, 1].set_ylabel('总距离 (像素)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = "demo_results/demo_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {plot_file}")
    
    # 显示图表（如果支持的话）
    try:
        plt.show()
    except:
        print("无法显示图表（可能没有GUI支持）")
    
    plt.close()


def main():
    """主演示函数"""
    print("=" * 60)
    print("      指标收集系统演示")
    print("=" * 60)
    
    # 创建演示结果目录
    os.makedirs("demo_results", exist_ok=True)
    
    try:
        # 运行模拟
        collector, positions = simulate_robot_movement()
        
        # 可视化结果
        visualize_demo_results(collector, positions)
        
        print("\n" + "=" * 60)
        print("演示完成！生成的文件：")
        print(f"- 详细指标数据: {collector.csv_filename}")
        print(f"- 可视化图表: demo_results/demo_visualization.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
