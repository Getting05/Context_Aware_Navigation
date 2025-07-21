"""
指标分析工具 - 用于分析和可视化测试指标数据
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path


class MetricsAnalyzer:
    def __init__(self, metrics_dir="results/detailed_metrics"):
        self.metrics_dir = metrics_dir
        self.data = None
        self.map_data = {}
        
    def load_data(self, map_index=None):
        """加载指标数据"""
        if map_index is not None:
            # 加载特定地图的数据
            csv_file = f"{self.metrics_dir}/map_{map_index}_metrics.csv"
            if os.path.exists(csv_file):
                self.data = pd.read_csv(csv_file)
                print(f"已加载地图 {map_index} 的数据，共 {len(self.data)} 条记录")
            else:
                print(f"未找到地图 {map_index} 的数据文件")
        else:
            # 加载所有地图的数据
            csv_files = glob.glob(f"{self.metrics_dir}/map_*_metrics.csv")
            all_data = []
            
            for csv_file in csv_files:
                map_idx = int(Path(csv_file).stem.split('_')[1])
                df = pd.read_csv(csv_file)
                df['map_index'] = map_idx
                all_data.append(df)
                self.map_data[map_idx] = df
                
            if all_data:
                self.data = pd.concat(all_data, ignore_index=True)
                print(f"已加载 {len(csv_files)} 个地图的数据，总共 {len(self.data)} 条记录")
            else:
                print("未找到任何指标数据文件")
                
    def analyze_coverage_efficiency(self):
        """分析覆盖效率"""
        if self.data is None:
            print("请先加载数据")
            return
            
        print("\n=== 覆盖效率分析 ===")
        
        # 按地图分组分析
        for map_idx, df in self.map_data.items():
            if len(df) == 0:
                continue
                
            final_row = df.iloc[-1]
            
            print(f"\n地图 {map_idx}:")
            print(f"  最终覆盖率 (CR): {final_row['CR']:.4f}")
            print(f"  总移动距离: {final_row['total_distance']:.2f} 像素")
            print(f"  探索率: {final_row['exploration_rate']:.4f}")
            print(f"  清扫冗余度 (SR): {final_row['SR']:.4f}")
            print(f"  运动效率: {final_row['exploration_rate']/final_row['total_distance']*1000:.4f} (探索率/千像素)")
            print(f"  碰撞次数: {final_row['collision_count']}")
            print(f"  任务完成: {'是' if final_row['success_rate'] > 0 else '否'}")
            
    def analyze_motion_characteristics(self):
        """分析运动特征"""
        if self.data is None:
            print("请先加载数据")
            return
            
        print("\n=== 运动特征分析 ===")
        
        for map_idx, df in self.map_data.items():
            if len(df) == 0:
                continue
                
            print(f"\n地图 {map_idx}:")
            print(f"  平均速度: {df['avg_velocity'].iloc[-1]:.2f} 像素/秒")
            print(f"  最大瞬时速度: {df['current_velocity'].max():.2f} 像素/秒")
            print(f"  平均加速度幅值: {df['avg_acceleration'].iloc[-1]:.2f} 像素/秒²")
            print(f"  最大瞬时加速度: {df['current_acceleration'].abs().max():.2f} 像素/秒²")
            print(f"  平均加加速度幅值: {df['avg_jerk'].iloc[-1]:.2f} 像素/秒³")
            print(f"  最大瞬时加加速度: {df['current_jerk'].abs().max():.2f} 像素/秒³")
            
    def analyze_computation_performance(self):
        """分析计算性能"""
        if self.data is None:
            print("请先加载数据")
            return
            
        print("\n=== 计算性能分析 ===")
        
        for map_idx, df in self.map_data.items():
            if len(df) == 0:
                continue
                
            final_row = df.iloc[-1]
            avg_computation_time = df['computation_time_step'].mean()
            max_computation_time = df['computation_time_step'].max()
            
            print(f"\n地图 {map_idx}:")
            print(f"  总计算时间: {final_row['total_computation_time']:.4f} 秒")
            print(f"  平均单步计算时间: {avg_computation_time:.4f} 秒")
            print(f"  最大单步计算时间: {max_computation_time:.4f} 秒")
            print(f"  总任务时间: {final_row['task_time']:.2f} 秒")
            print(f"  计算时间占比: {final_row['total_computation_time']/final_row['task_time']*100:.2f}%")
            
    def plot_metrics_evolution(self, map_index=None, save_plots=True):
        """绘制指标演化图"""
        if self.data is None:
            print("请先加载数据")
            return
            
        if map_index is not None:
            if map_index not in self.map_data:
                print(f"未找到地图 {map_index} 的数据")
                return
            data_to_plot = {map_index: self.map_data[map_index]}
        else:
            data_to_plot = self.map_data
            
        for map_idx, df in data_to_plot.items():
            if len(df) == 0:
                continue
                
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'地图 {map_idx} 指标演化', fontsize=16)
            
            # 覆盖率演化
            axes[0, 0].plot(df['step'], df['CR'])
            axes[0, 0].set_title('覆盖率 (CR) 演化')
            axes[0, 0].set_xlabel('步数')
            axes[0, 0].set_ylabel('覆盖率')
            axes[0, 0].grid(True)
            
            # 探索率演化
            axes[0, 1].plot(df['step'], df['exploration_rate'])
            axes[0, 1].set_title('探索率演化')
            axes[0, 1].set_xlabel('步数')
            axes[0, 1].set_ylabel('探索率')
            axes[0, 1].grid(True)
            
            # 移动距离累计
            axes[0, 2].plot(df['step'], df['total_distance'])
            axes[0, 2].set_title('累计移动距离')
            axes[0, 2].set_xlabel('步数')
            axes[0, 2].set_ylabel('总距离 (像素)')
            axes[0, 2].grid(True)
            
            # 速度演化
            axes[1, 0].plot(df['step'], df['current_velocity'], alpha=0.6, label='瞬时速度')
            axes[1, 0].plot(df['step'], df['avg_velocity'], label='平均速度')
            axes[1, 0].set_title('速度演化')
            axes[1, 0].set_xlabel('步数')
            axes[1, 0].set_ylabel('速度 (像素/秒)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 冗余度演化
            axes[1, 1].plot(df['step'], df['SR'])
            axes[1, 1].set_title('清扫冗余度 (SR) 演化')
            axes[1, 1].set_xlabel('步数')
            axes[1, 1].set_ylabel('冗余度')
            axes[1, 1].grid(True)
            
            # 计算时间演化
            axes[1, 2].plot(df['step'], df['computation_time_step'])
            axes[1, 2].set_title('单步计算时间')
            axes[1, 2].set_xlabel('步数')
            axes[1, 2].set_ylabel('计算时间 (秒)')
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            
            if save_plots:
                plot_dir = f"{self.metrics_dir}/plots"
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(f"{plot_dir}/map_{map_idx}_evolution.png", dpi=300, bbox_inches='tight')
                print(f"保存图表到: {plot_dir}/map_{map_idx}_evolution.png")
            
            plt.show()
            
    def plot_trajectory(self, map_index, save_plot=True):
        """绘制轨迹图"""
        if map_index not in self.map_data:
            print(f"未找到地图 {map_index} 的数据")
            return
            
        df = self.map_data[map_index]
        
        plt.figure(figsize=(12, 10))
        
        # 绘制轨迹
        plt.plot(df['robot_x'], df['robot_y'], 'b-', linewidth=2, label='机器人轨迹')
        plt.scatter(df['robot_x'].iloc[0], df['robot_y'].iloc[0], 
                   c='green', s=100, marker='o', label='起始点', zorder=5)
        plt.scatter(df['target_x'].iloc[0], df['target_y'].iloc[0], 
                   c='red', s=100, marker='*', label='目标点', zorder=5)
        
        # 根据速度着色轨迹点
        scatter = plt.scatter(df['robot_x'], df['robot_y'], c=df['current_velocity'], 
                             cmap='viridis', s=20, alpha=0.6, label='速度分布')
        plt.colorbar(scatter, label='速度 (像素/秒)')
        
        plt.title(f'地图 {map_index} 机器人轨迹')
        plt.xlabel('X 坐标 (像素)')
        plt.ylabel('Y 坐标 (像素)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息文本
        final_row = df.iloc[-1]
        stats_text = f"""统计信息:
总距离: {final_row['total_distance']:.1f} 像素
覆盖率: {final_row['CR']:.3f}
探索率: {final_row['exploration_rate']:.3f}
冗余度: {final_row['SR']:.3f}
碰撞次数: {final_row['collision_count']}"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_plot:
            plot_dir = f"{self.metrics_dir}/plots"
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(f"{plot_dir}/map_{map_index}_trajectory.png", dpi=300, bbox_inches='tight')
            print(f"保存轨迹图到: {plot_dir}/map_{map_index}_trajectory.png")
            
        plt.show()
        
    def export_summary(self, output_file=None):
        """导出摘要报告"""
        if self.data is None:
            print("请先加载数据")
            return
            
        if output_file is None:
            output_file = f"{self.metrics_dir}/summary_report.csv"
            
        summary_data = []
        
        for map_idx, df in self.map_data.items():
            if len(df) == 0:
                continue
                
            final_row = df.iloc[-1]
            
            summary = {
                'map_index': map_idx,
                'total_steps': len(df),
                'final_CR': final_row['CR'],
                'final_exploration_rate': final_row['exploration_rate'],
                'total_distance': final_row['total_distance'],
                'final_SR': final_row['SR'],
                'collision_count': final_row['collision_count'],
                'avg_velocity': final_row['avg_velocity'],
                'avg_acceleration': final_row['avg_acceleration'],
                'avg_jerk': final_row['avg_jerk'],
                'total_computation_time': final_row['total_computation_time'],
                'task_time': final_row['task_time'],
                'success_rate': final_row['success_rate'],
                'motion_efficiency': final_row['exploration_rate'] / final_row['total_distance'] * 1000,
                'computation_time_ratio': final_row['total_computation_time'] / final_row['task_time']
            }
            summary_data.append(summary)
            
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_file, index=False)
        print(f"摘要报告已保存到: {output_file}")
        
        # 打印统计摘要
        print("\n=== 多地图统计摘要 ===")
        print(f"测试地图数量: {len(summary_data)}")
        print(f"平均覆盖率: {summary_df['final_CR'].mean():.4f} ± {summary_df['final_CR'].std():.4f}")
        print(f"平均探索率: {summary_df['final_exploration_rate'].mean():.4f} ± {summary_df['final_exploration_rate'].std():.4f}")
        print(f"平均移动距离: {summary_df['total_distance'].mean():.1f} ± {summary_df['total_distance'].std():.1f}")
        print(f"平均冗余度: {summary_df['final_SR'].mean():.4f} ± {summary_df['final_SR'].std():.4f}")
        print(f"平均碰撞次数: {summary_df['collision_count'].mean():.1f}")
        print(f"任务成功率: {summary_df['success_rate'].mean():.2f}")


def main():
    """主函数 - 演示如何使用分析工具"""
    analyzer = MetricsAnalyzer()
    
    print("指标分析工具")
    print("1. 加载所有数据")
    analyzer.load_data()
    
    print("\n2. 分析覆盖效率")
    analyzer.analyze_coverage_efficiency()
    
    print("\n3. 分析运动特征")
    analyzer.analyze_motion_characteristics()
    
    print("\n4. 分析计算性能")
    analyzer.analyze_computation_performance()
    
    print("\n5. 导出摘要报告")
    analyzer.export_summary()
    
    # 如果有数据，绘制第一个地图的指标演化
    if analyzer.map_data:
        first_map = list(analyzer.map_data.keys())[0]
        print(f"\n6. 绘制地图 {first_map} 的指标演化图")
        analyzer.plot_metrics_evolution(first_map)
        
        print(f"\n7. 绘制地图 {first_map} 的轨迹图")
        analyzer.plot_trajectory(first_map)


if __name__ == "__main__":
    main()
