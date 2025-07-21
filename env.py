import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import copy
import skimage.io
from skimage.measure import block_reduce
from scipy import *
from sensor import *
from graph_generator import *
from node import *
import time

class Env():
    def __init__(self, map_index, k_size=20, plot=False, test=False):
        self.test = test
        if self.test:
            self.map_dir = f'DungeonMaps/pp/test'
        else:
            self.map_dir = f'DungeonMaps/pp/train'
        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort()  # Sort map_list in ascending order to ensure sequential selection
        self.map_index = map_index % len(self.map_list)  # Use map_index directly to select maps sequentially
        self.ground_truth, self.start_position, self.target_position = self.import_ground_truth_pp(
            self.map_dir + '/' + self.map_list[self.map_index])
        self.ground_truth_size = np.shape(self.ground_truth)
        self.robot_belief = np.ones(self.ground_truth_size) * 127  # unexplored
        self.downsampled_belief = None
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.resolution = 4  # downsample belief
        self.sensor_range = 80
        self.explored_rate = 0
        self.frontiers = None
        self.graph_generator = Graph_generator(map_size=self.ground_truth_size, sensor_range=self.sensor_range, k_size=k_size, target_position = self.target_position, plot=plot)
        self.graph_generator.route_node.append(self.start_position)
        self.node_coords, self.graph, self.node_utility, self.indicator, self.direction_vector = None, None, None, None, None
        
        # 初始化任务目标
        self.sweeping_objects = []  # S类目标列表
        self.grasping_objects = []  # G类目标列表
        self.completed_sweeping = []  # 已完成的S类目标
        self.completed_grasping = []  # 已完成的G类目标
        self.task_completion_radius = 3  # 任务完成半径
        
        # 在begin()之后生成任务目标
        self.begin()
        
        # 设置随机种子以保证可重现性
        np.random.seed(42 + map_index)
        self.generate_task_objects()

        self.plot = plot
        self.frame_files = []
        if self.plot:
            self.xPoints = [self.start_position[0]]
            self.yPoints = [self.start_position[1]]
            self.xTarget = [self.target_position[0]]
            self.yTarget = [self.target_position[1]]

    def find_index_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        return index

    def begin(self):
        self.robot_belief = self.update_robot_belief(self.start_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        self.frontiers = self.find_frontier()
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.node_coords, self.graph, self.node_utility, self.indicator, self.direction_vector = self.graph_generator.generate_graph(
            self.start_position, self.ground_truth, self.robot_belief, self.frontiers)

    def step(self, robot_position, next_position, travel_dist):
        dist = np.linalg.norm(robot_position - next_position)
        dist_to_target = np.linalg.norm(next_position - self.target_position)
        astar_dist_cur_to_target, _ = self.graph_generator.find_shortest_path(robot_position, self.target_position, 
                                                                           self.graph_generator.ground_truth_node_coords, self.graph_generator.ground_truth_graph)
        astar_dist_next_to_target, _ = self.graph_generator.find_shortest_path(next_position, self.target_position, 
                                                                            self.graph_generator.ground_truth_node_coords, self.graph_generator.ground_truth_graph)
        travel_dist += dist
        robot_position = next_position
        self.graph_generator.route_node.append(robot_position)
        next_node_index = self.find_index_from_coords(robot_position)
        self.graph_generator.nodes_list[next_node_index].set_visited()
        self.robot_belief = self.update_robot_belief(robot_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        frontiers = self.find_frontier()
        self.explored_rate = self.evaluate_exploration_rate()
        
        # 检查任务目标完成情况
        newly_completed_s, newly_completed_g = self.check_task_completion(robot_position)
        
        reward, done = self.calculate_reward(astar_dist_cur_to_target, astar_dist_next_to_target, dist_to_target)
        
        # 为完成任务目标提供额外奖励
        reward += (newly_completed_s + newly_completed_g) * 5.0  # 每完成一个任务目标+5奖励
        
        if self.plot:
            self.xPoints.append(robot_position[0])
            self.yPoints.append(robot_position[1])
        self.node_coords, self.graph, self.node_utility, self.indicator, self.direction_vector = self.graph_generator.update_graph(
            robot_position, self.robot_belief, self.old_robot_belief, frontiers, self.frontiers)
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.frontiers = frontiers

        return reward, done, robot_position, travel_dist
    
    def import_ground_truth_pp(self, map_index):
        ground_truth = (skimage.io.imread(map_index, 1) * 255).astype(int)
        print(f"Loading map: {map_index}")  # Debug log for map path
        print(f"Unique pixel values in map: {np.unique(ground_truth)}")  # Debug log for pixel values

        robot_location = np.nonzero(ground_truth == 209)
        if len(robot_location[0]) == 0:
            print("Warning: No start position (pixel value 209) found in the map. Using default start position.")
            robot_location = np.array([0, 0])  # Default start position
        else:
            robot_location = np.array([robot_location[1][0], robot_location[0][0]])

        target_location = np.nonzero(ground_truth == 68)
        if len(target_location[0]) == 0:
            raise ValueError("No target position (pixel value 68) found in the map.")
        target_location = np.array([target_location[1][0], target_location[0][0]])

        ground_truth = (ground_truth > 150)|((ground_truth<=80)&(ground_truth>=60))
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location, target_location
    
    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        return robot_belief

    def calculate_reward(self, astar_dist_cur_to_target, astar_dist_next_to_target, dist_to_target):
        reward = 0
        done = False
        reward -= 0.5
        reward += (astar_dist_cur_to_target - astar_dist_next_to_target) / 64
        if dist_to_target == 0:
            reward += 20
            done = True
        return reward, done

    def evaluate_exploration_rate(self):
        rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)
        return rate

    def find_frontier(self):
        y_len = self.downsampled_belief.shape[0]
        x_len = self.downsampled_belief.shape[1]
        mapping = self.downsampled_belief.copy()
        belief = self.downsampled_belief.copy()
        mapping = (mapping == 127) * 1
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)
        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        f = points[ind_to]
        f = f.astype(int)
        f = f * self.resolution
        return f

    def plot_env(self, n, path, step, travel_dist):
        plt.switch_backend('agg')
        # plt.ion()
        plt.cla()
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        # for i in range(len(self.graph_generator.x)):
        #    plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'tan', zorder=1)  # plot edges will take long time
        # 不显示节点点位
        # plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=self.node_utility, zorder=5)
        # plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        
        # 保留目标点和路径，使用英文标签避免中文字体问题
        plt.plot(self.xTarget, self.yTarget, 'o', color='green', markersize=20, label='Target')
        plt.plot(self.xPoints, self.yPoints, '-', color='blue', linewidth=3, label='Path')
        plt.plot(self.xPoints[-1], self.yPoints[-1], 'o', color='magenta', markersize=10, label='Current Position')
        plt.plot(self.xPoints[0], self.yPoints[0], 'o', color='cyan', markersize=10, label='Start Position')
        
        # 绘制任务目标
        # S类目标（清扫）- 未完成的用红色正方形，已完成的用灰色正方形
        for i, s_target in enumerate(self.sweeping_objects):
            color = 'gray' if i in self.completed_sweeping else 'red'
            plt.plot(s_target[0], s_target[1], 's', color=color, markersize=8, 
                    label='S-targets (completed)' if i in self.completed_sweeping and i == 0 else 
                          ('S-targets' if i == 0 and i not in self.completed_sweeping else ''))
        
        # G类目标（抓取）- 未完成的用橙色三角形，已完成的用灰色三角形
        for i, g_target in enumerate(self.grasping_objects):
            color = 'gray' if i in self.completed_grasping else 'orange'
            plt.plot(g_target[0], g_target[1], '^', color=color, markersize=8,
                    label='G-targets (completed)' if i in self.completed_grasping and i == 0 else
                          ('G-targets' if i == 0 and i not in self.completed_grasping else ''))
        
        # 添加图例
        plt.legend(loc='upper right')
        
        # 更新标题，包含TCR信息
        tcr = self.get_tcr()
        plt.suptitle('Explored: {:.4g}  Distance: {:.4g}  TCR: {:.4g}'.format(
            self.explored_rate, travel_dist, tcr))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=150))
        plt.close()  # Close the figure to free memory instead of showing it
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)
    
    def generate_task_objects(self):
        """
        在地图的自由区域随机生成S类和G类任务目标
        """
        # 获取自由区域的坐标点
        free_cells = self.free_cells()
        
        # 从test_parameter导入参数
        from test_parameter import NUM_SWEEPING_OBJECTS, NUM_GRASPING_OBJECTS, TASK_COMPLETION_RADIUS
        
        if len(free_cells) < NUM_SWEEPING_OBJECTS + NUM_GRASPING_OBJECTS:
            print(f"警告：可用自由区域点数不足 ({len(free_cells)} < {NUM_SWEEPING_OBJECTS + NUM_GRASPING_OBJECTS})，将使用所有可用点")
            num_s_targets = min(len(free_cells) // 2, NUM_SWEEPING_OBJECTS)
            num_g_targets = min(len(free_cells) - num_s_targets, NUM_GRASPING_OBJECTS)
        else:
            num_s_targets = NUM_SWEEPING_OBJECTS
            num_g_targets = NUM_GRASPING_OBJECTS
        
        # 设置任务完成半径
        self.task_completion_radius = TASK_COMPLETION_RADIUS
        
        # 随机选择位置放置目标，确保不重复，并且远离起点和终点
        excluded_positions = [self.start_position, self.target_position]
        
        # 筛选出距离起点和终点足够远的自由区域
        valid_positions = []
        for pos in free_cells:
            too_close = False
            for exclude_pos in excluded_positions:
                if np.linalg.norm(pos - exclude_pos) < 30:  # 至少距离30像素
                    too_close = True
                    break
            if not too_close:
                valid_positions.append(pos)
        
        if len(valid_positions) < num_s_targets + num_g_targets:
            print(f"警告：有效位置不足，使用原始自由区域")
            valid_positions = free_cells.tolist()
        
        # 随机选择位置放置目标，确保不重复
        selected_indices = np.random.choice(len(valid_positions), 
                                          size=min(len(valid_positions), num_s_targets + num_g_targets), 
                                          replace=False)
        selected_positions = np.array(valid_positions)[selected_indices]
        
        # 分配S类和G类目标
        actual_s_targets = min(len(selected_positions) // 2, num_s_targets)
        self.sweeping_objects = selected_positions[:actual_s_targets].tolist()
        self.grasping_objects = selected_positions[actual_s_targets:actual_s_targets + min(len(selected_positions) - actual_s_targets, num_g_targets)].tolist()
        
        print(f"生成了 {len(self.sweeping_objects)} 个S类目标和 {len(self.grasping_objects)} 个G类目标")

    def check_task_completion(self, robot_position):
        """
        检查机器人当前位置是否完成了任何任务目标
        
        Args:
            robot_position: 机器人当前位置
            
        Returns:
            tuple: (新完成的S类目标数量, 新完成的G类目标数量)
        """
        newly_completed_s = 0
        newly_completed_g = 0
        
        # 检查S类目标
        for i, s_target in enumerate(self.sweeping_objects):
            if i not in self.completed_sweeping:
                distance = np.linalg.norm(robot_position - np.array(s_target))
                if distance <= self.task_completion_radius:
                    self.completed_sweeping.append(i)
                    newly_completed_s += 1
                    print(f"完成S类目标 {i+1}/{len(self.sweeping_objects)} 在位置 {s_target}")
        
        # 检查G类目标
        for i, g_target in enumerate(self.grasping_objects):
            if i not in self.completed_grasping:
                distance = np.linalg.norm(robot_position - np.array(g_target))
                if distance <= self.task_completion_radius:
                    self.completed_grasping.append(i)
                    newly_completed_g += 1
                    print(f"完成G类目标 {i+1}/{len(self.grasping_objects)} 在位置 {g_target}")
        
        return newly_completed_s, newly_completed_g

    def get_tcr(self):
        """
        计算任务完成率 (Task Completion Rate)
        
        Returns:
            float: TCR值 (0.0 到 1.0)
        """
        total_tasks = len(self.sweeping_objects) + len(self.grasping_objects)
        completed_tasks = len(self.completed_sweeping) + len(self.completed_grasping)
        
        if total_tasks == 0:
            return 1.0
        
        return completed_tasks / total_tasks