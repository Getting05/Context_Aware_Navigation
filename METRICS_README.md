# Context Aware Navigation 指标收集系统

本项目实现了一个综合的指标收集系统，用于评估扫地机器人在寻路过程中的各种性能参数。

## 📊 实现的指标

### 核心性能指标

| 指标 | 名称 | 计算公式 | 说明 |
|------|------|----------|------|
| **CR** | 覆盖率 | `已覆盖栅格数 / 总可移动栅格数` | 机器人实际清扫覆盖的区域比例 |
| **SR** | 清扫冗余度 | `重复访问栅格数 / 总访问栅格数` | 重复清扫区域的比例，越低越好 |
| **ME** | 运动效率 | `探索率 / 总移动距离 × 1000` | 每千像素移动距离的探索效率 |
| **Collision** | 碰撞计数 | 累计碰撞次数 | 与障碍物的物理碰撞次数 |
| **CT** | 计算时间 | 累计算法耗时 | 路径规划算法的总计算时间 |
| **FT** | 任务总耗时 | 任务开始到结束的时间 | 完成整个探索任务的总时间 |

### 运动学指标

| 指标 | 名称 | 计算公式 | 说明 |
|------|------|----------|------|
| **Vel_avg** | 平均速度 | `总移动距离 / 任务总时间` | 机器人的平均移动速度 |
| **Acc_avg** | 平均加速度 | `∑|加速度| / 采样次数` | 平均加速度幅值 |
| **Jerk_avg** | 平均加加速度 | `∑|加加速度| / 采样次数` | 平均加加速度幅值，反映运动平滑性 |

### 扩展指标

| 指标 | 名称 | 说明 |
|------|------|------|
| **total_distance** | 总移动距离 | 机器人行驶的总距离 |
| **covered_area** | 覆盖面积 | 已清扫/覆盖的栅格数量 |
| **exploration_rate** | 探索百分比 | 当前已探索区域的比例 |
| **robot_count** | 机器人数量 | 当前系统中的机器人数量 |
| **max_travel_distance** | 最大行驶距离 | 单机器人的最大行驶距离 |

## 🏗️ 系统架构

### 核心组件

1. **MetricsCollector** (`metrics_collector.py`)
   - 实时收集各种性能指标
   - 计算覆盖率、冗余度、运动学参数
   - 保存详细的每步数据到CSV文件

2. **TestWorker** (修改后的 `test_worker.py`)
   - 集成指标收集器
   - 在测试过程中实时记录数据
   - 每步保存指标到CSV文件

3. **MetricsAnalyzer** (`analyze_metrics.py`)
   - 分析和可视化收集的指标数据
   - 生成统计报告和图表
   - 支持多地图对比分析

## 🚀 使用方法

### 1. 运行测试并收集指标

```python
# 运行单个地图测试
python test_driver.py  # 使用原有的测试驱动

# 或者使用我们的测试脚本
python test_metrics.py  # 专门用于测试指标收集
```

### 2. 分析收集的指标

```python
# 使用分析工具
python analyze_metrics.py

# 或者在代码中使用
from analyze_metrics import MetricsAnalyzer

analyzer = MetricsAnalyzer()
analyzer.load_data()  # 加载所有数据
analyzer.analyze_coverage_efficiency()  # 分析覆盖效率
analyzer.plot_metrics_evolution(map_index=0)  # 绘制指标演化
analyzer.export_summary()  # 导出摘要报告
```

### 3. 数据文件结构

```
results/
├── detailed_metrics/
│   ├── map_0_metrics.csv      # 地图0的详细指标
│   ├── map_1_metrics.csv      # 地图1的详细指标
│   ├── ...
│   ├── summary_report.csv     # 所有地图的摘要
│   └── plots/
│       ├── map_0_evolution.png    # 指标演化图
│       ├── map_0_trajectory.png   # 轨迹图
│       └── ...
```

## 📈 指标实现详情

### 覆盖率 (CR) 实现

```python
def calculate_coverage_rate(self):
    """计算覆盖率 CR = 已覆盖栅格数 / 总可移动栅格数"""
    return len(self.covered_cells) / self.total_free_cells

def update_covered_cells(self, position):
    """更新机器人覆盖的栅格（考虑机器人半径）"""
    x, y = int(position[0]), int(position[1])
    # 计算圆形覆盖区域
    for dx in range(-self.robot_radius, self.robot_radius + 1):
        for dy in range(-self.robot_radius, self.robot_radius + 1):
            if dx*dx + dy*dy <= self.robot_radius**2:
                cell_coord = (x + dx, y + dy)
                if self.is_valid_cell(cell_coord):
                    self.covered_cells.add(cell_coord)
```

### 清扫冗余度 (SR) 实现

```python
def calculate_redundancy_rate(self):
    """计算清扫冗余度 SR = 重复访问栅格面积 / 总访问栅格面积"""
    total_visited_area = len(self.cell_visit_count)
    if total_visited_area == 0:
        return 0.0
    # 计算重复访问的栅格数（访问次数-1）
    redundant_area = sum(max(0, count - 1) for count in self.cell_visit_count.values())
    return redundant_area / total_visited_area
```

### 运动学指标实现

```python
def update_kinematics(self, new_position, current_time):
    """更新速度、加速度、加加速度"""
    if self.last_position is not None:
        # 计算速度
        distance = np.linalg.norm(new_position - self.last_position)
        time_diff = current_time - self.last_position_time
        velocity = distance / time_diff if time_diff > 0 else 0
        
        # 计算加速度
        if len(self.velocities) >= 1:
            acceleration = (velocity - self.last_velocity) / time_diff
            
            # 计算加加速度
            if len(self.accelerations) >= 1:
                jerk = (acceleration - self.last_acceleration) / time_diff
                self.jerks.append(jerk)
```

### 碰撞检测实现

```python
def check_collision(self, position, current_time):
    """检查碰撞并计数"""
    x, y = int(position[0]), int(position[1])
    # 检查机器人位置是否与障碍物碰撞
    if self.env.ground_truth[y, x] == 1:  # 障碍物标记为1
        # 防止重复计数的冷却机制
        if current_time - self.last_collision_time > self.collision_cooldown:
            self.collision_count += 1
            self.last_collision_time = current_time
```

## 📊 CSV数据格式

每个地图生成一个CSV文件，包含以下字段：

| 字段名 | 类型 | 描述 |
|--------|------|------|
| step | int | 当前步数 |
| episode | int | 回合编号 |
| timestamp | float | 时间戳 |
| map_index | int | 地图索引 |
| robot_x, robot_y | float | 机器人坐标 |
| target_x, target_y | float | 目标坐标 |
| total_distance | float | 累计移动距离 |
| step_distance | float | 单步移动距离 |
| covered_area | int | 覆盖的栅格数 |
| exploration_rate | float | 探索率 |
| CR | float | 覆盖率 |
| SR | float | 清扫冗余度 |
| current_velocity | float | 瞬时速度 |
| avg_velocity | float | 平均速度 |
| current_acceleration | float | 瞬时加速度 |
| avg_acceleration | float | 平均加速度 |
| current_jerk | float | 瞬时加加速度 |
| avg_jerk | float | 平均加加速度 |
| collision_count | int | 累计碰撞次数 |
| computation_time_step | float | 单步计算时间 |
| total_computation_time | float | 累计计算时间 |
| task_time | float | 任务总时间 |
| success_rate | float | 任务完成率 |
| reward | float | 奖励值 |

## 🔧 配置选项

在 `test_parameter.py` 中可以配置：

```python
# 详细指标收集配置
SAVE_DETAILED_METRICS = True  # 是否保存详细指标
METRICS_DIR = 'results/detailed_metrics'  # 保存目录
ROBOT_RADIUS = 10  # 机器人半径（像素）
COLLISION_COOLDOWN = 0.1  # 碰撞检测冷却时间
```

## 📈 分析功能

### 1. 基本统计分析
- 多地图平均性能对比
- 标准差和变异性分析
- 成功率统计

### 2. 时序分析
- 指标随时间的演化趋势
- 收敛性分析
- 性能稳定性评估

### 3. 可视化
- 指标演化曲线图
- 机器人轨迹图（按速度着色）
- 多地图对比图表

### 4. 报告生成
- CSV格式的摘要报告
- 高质量的图表导出
- 详细的统计信息

## 🎯 指标解读指南

### 优秀性能指标参考值
- **CR (覆盖率)**: > 0.9 (90%以上覆盖)
- **SR (冗余度)**: < 0.1 (10%以下重复)
- **碰撞次数**: = 0 (无碰撞)
- **运动效率**: 越高越好
- **平均速度**: 稳定且适中
- **加速度/加加速度**: 较低值表示运动平滑

### 问题诊断
- **覆盖率低**: 路径规划不充分，可能存在死角
- **冗余度高**: 重复访问过多，效率低下
- **碰撞多**: 路径规划或障碍物检测有问题
- **速度波动大**: 运动控制不平滑
- **计算时间长**: 算法效率需要优化

## 🔍 扩展功能

系统设计为可扩展的，可以轻松添加新的指标：

1. 在 `MetricsCollector` 中添加新的计算方法
2. 更新 `field_names` 列表
3. 在 `get_current_metrics()` 中返回新指标
4. 在分析工具中添加对应的分析功能

这个指标收集系统为扫地机器人导航算法的评估和优化提供了全面的数据支持。
