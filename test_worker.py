import imageio
import csv
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from env import Env
from model import PolicyNet
from test_parameter import *
from metrics_collector import MetricsCollector


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image, test=True)
        self.local_policy_net = policy_net
        self.travel_dist = 0
        self.robot_position = self.env.start_position
        self.perf_metrics = dict()
        
        # 初始化指标收集器
        self.metrics_collector = MetricsCollector(self.global_step, self.env)

    def run_episode(self, curr_episode):
        done = False
        step = 0
        total_steps = 0  # 记录总步数

        observations = self.get_observations()
        # 初始位置记录到指标收集器
        self.metrics_collector.update_position(self.robot_position)
        
        for i in range(MAX_STEPS):  # 使用MAX_STEPS替换固定的128
            # 记录计算时间开始
            computation_start = time.perf_counter()
            
            next_position, action_index = self.select_node(observations)

            # 记录计算时间结束
            computation_end = time.perf_counter()
            computation_time = computation_end - computation_start
            
            # 在当前位置和下一个位置之间插入中间点
            interpolated_positions = self.interpolate_positions(self.robot_position, next_position, steps=10)  # 增加步数
            
            # 确保所有插值位置都是有效的图节点
            valid_positions = []
            for pos in interpolated_positions:
                valid_pos = self.find_nearest_valid_node(pos)
                valid_positions.append(valid_pos)
            
            # 逐个处理有效的插值位置
            for j, intermediate_pos in enumerate(valid_positions[1:]):  # 跳过第一个点（当前位置）
                # 增加总步数
                total_steps += 1
                
                # 检查是否达到每张地图的最大步数限制
                if total_steps >= MAX_STEPS_PER_MAP:
                    print(f"达到地图最大步数限制 ({MAX_STEPS_PER_MAP})，自动结束当前地图测试")
                    done = True
                    break
                
                # 移动到中间位置
                try:
                    sub_reward, sub_done, self.robot_position, self.travel_dist = self.env.step(
                        self.robot_position, intermediate_pos, self.travel_dist
                    )
                    
                    # 只有最后一个点才使用实际reward和done，其他点使用临时值
                    if j == len(valid_positions) - 2:  # 最后一个中间点
                        reward, done = sub_reward, sub_done
                    else:
                        reward, done = 0, False
                    
                    # 更新指标收集器
                    step_distance = self.metrics_collector.update_position(
                        self.robot_position, reward=reward, done=done
                    )
                    
                    # 保存单步指标
                    self.metrics_collector.save_step_metrics(
                        step=total_steps,  # 使用总步数
                        episode=curr_episode, 
                        reward=reward, 
                        computation_time=computation_time / len(valid_positions)  # 平均分配计算时间
                    )
                    
                    # 如果到达目标或完成探索，提前退出
                    if done:
                        break
                except Exception as e:
                    print(f"处理中间点时出错: {e}")
                    continue
            
            # 如果在处理中间点过程中完成任务或达到最大步数，则退出主循环
            if done or total_steps >= MAX_STEPS_PER_MAP:
                break
                
            # 获取新的观察
            observations = self.get_observations()

            # save evaluation data
            if SAVE_TRAJECTORY:
                if not os.path.exists(trajectory_path):
                    os.makedirs(trajectory_path)
                csv_filename = f'results/trajectory/ours_trajectory_result.csv'
                new_file = False if os.path.exists(csv_filename) else True
                field_names = ['dist', 'area']
                with open(csv_filename, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.array([self.travel_dist, np.sum(self.env.robot_belief == 255)]).reshape(1, -1)
                    writer.writerows(csv_data)

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, self.travel_dist)

            if done:
                break

        # 打印指标摘要
        self.metrics_collector.print_summary()

        # 记录步数信息
        reached_step_limit = total_steps >= MAX_STEPS_PER_MAP
        if reached_step_limit:
            print(f"已达到每张地图的最大步数限制 ({MAX_STEPS_PER_MAP})，自动切换到下一张地图")
            self.perf_metrics['reached_step_limit'] = True
            # 如果达到步数限制但没有成功完成，将 done 设置为 True 以便切换到下一张地图
            if not done:
                done = True
                print("由于达到步数限制，将当前地图标记为已完成")

        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['total_steps'] = total_steps

        # save final path length
        if SAVE_LENGTH:
            if not os.path.exists(length_path):
                os.makedirs(length_path)
            csv_filename = f'results/length/ours_length_result.csv'
            new_file = False if os.path.exists(csv_filename) else True
            field_names = ['dist']
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.array([self.travel_dist]).reshape(-1,1)
                writer.writerows(csv_data)

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def get_observations(self):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        indicator = copy.deepcopy(self.env.indicator)

        direction_vector = copy.deepcopy(self.env.direction_vector)

        # normalize observations
        node_coords = node_coords / 640
        node_utility = node_utility / 50

        # transfer to node inputs tensor
        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape((n_nodes, 1))
        direction_nums = direction_vector.shape[0]
        direction_vector_inputs = direction_vector.reshape(direction_nums, 3)
        direction_vector_inputs[:, 2] /= 640
        node_inputs = np.concatenate((node_coords, node_utility_inputs, indicator, direction_vector_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)

        # calculate a mask for padded node
        node_padding_mask = None

        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        edge = edge_inputs[current_index]
        while len(edge) < self.k_size:
            edge.append(0)

        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

        edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = edge_inputs[0, 0, action_index.item()]
        next_position = self.env.node_coords[next_node_index]

        return next_position, action_index

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
        return bias_matrix

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_rate), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)

    def interpolate_positions(self, start_pos, end_pos, steps=5):
        """
        在两个位置之间插入中间点，以增加路径密度
        
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            steps: 要插入的中间点数量
            
        Returns:
            插值后的位置列表，包括起点和终点
        """
        positions = []
        # 线性插值
        for i in range(steps + 1):
            t = i / steps
            x = start_pos[0] * (1 - t) + end_pos[0] * t
            y = start_pos[1] * (1 - t) + end_pos[1] * t
            positions.append(np.array([x, y]))
        return positions

    def find_nearest_valid_node(self, position):
        """
        找到与给定位置最近的有效节点坐标
        
        Args:
            position: 要查找的位置
            
        Returns:
            最近的有效节点坐标
        """
        # 获取环境中所有节点的坐标
        node_coords = self.env.node_coords
        
        if node_coords.shape[0] == 0:
            # 如果没有有效节点，返回原始位置
            print("警告：没有找到有效节点，使用原始位置")
            return position
            
        # 计算给定位置到所有节点的距离
        distances = np.linalg.norm(node_coords - position, axis=1)
        
        # 找到距离最小的节点索引
        nearest_index = np.argmin(distances)
        
        # 返回最近节点的坐标
        return node_coords[nearest_index]

    def work(self, curr_episode):
        self.run_episode(curr_episode)
