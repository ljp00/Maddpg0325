import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy
from scipy.ndimage import gaussian_filter


class EnvironmentGenerator:
    def __init__(self, grid_size=(100, 100)):
        self.grid_size = grid_size
        self.sensitivity_map = self._generate_sensitivity()
        # 计算并存储敏感度梯度
        self.gradient_x, self.gradient_y = self._compute_gradients()

    def _generate_sensitivity(self, smooth_factor=20.0):
        sensitivity = np.random.rand(*self.grid_size)
        sensitivity = gaussian_filter(sensitivity, sigma=smooth_factor)
        sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min())
        return np.clip(sensitivity, 0, 1)

    def _compute_gradients(self):
        """计算敏感度地图的梯度"""
        gradient_y, gradient_x = np.gradient(self.sensitivity_map)
        # 归一化梯度到[-1, 1]范围
        max_grad = max(np.abs(gradient_x).max(), np.abs(gradient_y).max())
        if max_grad > 0:
            gradient_x = gradient_x / max_grad
            gradient_y = gradient_y / max_grad
        return gradient_x, gradient_y

    def get_sensitivity_and_gradient(self, x_grid, y_grid):
        """获取指定网格位置的敏感度值和梯度"""
        x_grid = np.clip(x_grid, 0, self.grid_size[0] - 1)
        y_grid = np.clip(y_grid, 0, self.grid_size[1] - 1)
        return (self.sensitivity_map[x_grid, y_grid],
                self.gradient_x[x_grid, y_grid],
                self.gradient_y[x_grid, y_grid])


class UAVEnv:
    def __init__(self, length=2, num_obstacle=3, num_agents=3, grid_size=100):
        # 基本环境参数
        self.grid_size = grid_size
        self.length = length
        self.num_obstacle = num_obstacle
        self.num_agents = num_agents  # 现在只包含实际的UAV数量

        # UAV运动参数
        self.time_step = 0.5
        self.v_max = 0.1
        self.a_max = 0.04

        # 传感器参数
        self.L_sensor = 0.2
        self.num_lasers = 16
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)]
                                     for _ in range(self.num_agents)]

        # 初始化位置和速度列表
        self.multi_current_pos = [np.zeros(2) for _ in range(self.num_agents)]
        self.multi_current_vel = [np.zeros(2) for _ in range(self.num_agents)]

        # 环境状态变量
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        self.history_positions = [[] for _ in range(num_agents)]

        # 环境敏感度相关
        self.env_gen = EnvironmentGenerator(grid_size=(self.grid_size, self.grid_size))
        self.monitor_radius = 0.3  # 监测半径（世界坐标）
        self.mu_cov = 0.4  # 覆盖奖励权重

        # 空间定义
        self.action_space = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
            for agent in self.agents
        }
        self.observation_space = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(27,))
            for agent in self.agents
        }

    def get_sensitivity_info(self, pos):
        """获取指定位置的敏感度信息"""
        scale = self.grid_size / self.length
        x_grid = int(pos[0] * scale)
        y_grid = int(pos[1] * scale)
        return self.env_gen.get_sensitivity_and_gradient(x_grid, y_grid)

    def compute_coverage_reward(self):
        """计算覆盖奖励"""
        scale = self.grid_size / self.length
        R_grid = int(self.monitor_radius * scale)
        coverage_mask = np.zeros(self.env_gen.sensitivity_map.shape, dtype=bool)
        individual_rewards = []

        for i in range(self.num_agents - 1):
            pos = self.multi_current_pos[i]
            x_grid = int(pos[0] * scale)
            y_grid = int(pos[1] * scale)

            x_low = max(0, x_grid - R_grid)
            x_high = min(self.grid_size, x_grid + R_grid + 1)
            y_low = max(0, y_grid - R_grid)
            y_high = min(self.grid_size, y_grid + R_grid + 1)

            xs, ys = np.meshgrid(np.arange(x_low, x_high),
                                 np.arange(y_low, y_high), indexing='ij')
            distances = np.sqrt((xs - x_grid) ** 2 + (ys - y_grid) ** 2)
            circle_mask = distances <= R_grid

            # 计算个体覆盖奖励
            individual_coverage = np.zeros_like(coverage_mask)
            individual_coverage[x_low:x_high, y_low:y_high] = circle_mask
            individual_rewards.append(np.sum(self.env_gen.sensitivity_map[individual_coverage]))

            # 更新总覆盖区域
            coverage_mask[x_low:x_high, y_low:y_high] |= circle_mask

        total_reward = np.sum(self.env_gen.sensitivity_map[coverage_mask])
        return total_reward, individual_rewards

    def get_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            # 1. 基础状态：位置和速度 (4维)
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max
            ]  # 4维

            # 2. 队友位置 (4维)
            S_team = []
            team_count = 0
            for j in range(self.num_agents - 1):  # 修改：只遍历非目标UAV
                if j != i:
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0] / self.length, pos_other[1] / self.length])
                    team_count += 1
                    if team_count == 2:  # 只取两个队友的位置
                        break

            # 如果队友数量不足，用0填充
            while len(S_team) < 4:
                S_team.extend([0, 0])

            # 3. 激光传感器数据 (16维)
            S_obser = self.multi_current_lasers[i][:16]  # 确保只取16个激光数据

            # 4. 敏感度信息 (3维)
            sensitivity, dx, dy = self.get_sensitivity_info(pos)
            S_sensitivity = [sensitivity, dx, dy]

            # 调试信息
            print(f"Dimensions check:")
            print(f"S_uavi: {len(S_uavi)}")  # 应该是4
            print(f"S_team: {len(S_team)}")  # 应该是4
            print(f"S_obser: {len(S_obser)}")  # 应该是16
            print(f"S_sensitivity: {len(S_sensitivity)}")  # 应该是3

            # 合并所有状态 (总计27维: 4+4+16+3)
            single_obs = S_uavi + S_team + S_obser + S_sensitivity

            # 确保维度正确
            assert len(single_obs) == 27, f"Observation dimension mismatch. Expected 27, got {len(single_obs)}"

            total_obs.append(single_obs)

        return total_obs

        return total_obs

    def reset(self):
        """重置环境状态"""
        SEED = random.randint(1, 1000)
        random.seed(SEED)

        # 重置位置和速度列表
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]

        # 为每个智能体随机初始化位置
        for _ in range(self.num_agents):
            # 在环境边界内随机初始化位置
            pos = np.random.uniform(low=0.1, high=0.4, size=(2,))
            self.multi_current_pos.append(pos)
            self.multi_current_vel.append(np.zeros(2))

        # 初始化激光传感器数据
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)]
                                     for _ in range(self.num_agents)]

        # 更新激光传感器数据和碰撞检测
        self.update_lasers_isCollied_wrapper()

        return self.get_multi_obs()

    def step(self, actions):
        """环境步进"""
        # 记录当前状态
        rewards = np.zeros(self.num_agents)

        # 更新每个智能体的位置和速度
        for i in range(self.num_agents):
            # 更新速度
            self.multi_current_vel[i] += actions[i] * self.time_step

            # 速度限制
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if vel_magnitude >= self.v_max:
                self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max

            # 更新位置
            self.multi_current_pos[i] += self.multi_current_vel[i] * self.time_step

            # 确保智能体在环境边界内
            self.multi_current_pos[i] = np.clip(self.multi_current_pos[i], 0, self.length)

        # 更新碰撞检测和奖励
        IsCollied = self.update_lasers_isCollied_wrapper()
        rewards, dones = self.cal_rewards_dones(IsCollied, None)  # 移除了last_d参数

        return self.get_multi_obs(), rewards, dones

    def update_lasers_isCollied_wrapper(self):
        """更新激光传感器数据和碰撞状态"""
        self.multi_current_lasers = []
        dones = []

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []

            # 检查与每个障碍物的碰撞
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos, obs_pos, r,
                                                      self.L_sensor, self.num_lasers,
                                                      self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)

            # 处理碰撞状态
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)

            self.multi_current_lasers.append(current_lasers)
            dones.append(done)

        return dones

    def cal_rewards_dones(self, IsCollied, last_d):
        """计算奖励和完成状态"""
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)

        # 权重设置
        mu_coverage = 0.4  # 覆盖奖励权重
        mu_collision = 0.4  # 避障奖励权重
        mu_completion = 0.2  # 任务完成奖励权重

        # 1. 覆盖奖励
        total_coverage, individual_coverage = self.compute_coverage_reward()
        for i in range(self.num_agents - 1):
            rewards[i] += mu_coverage * (0.5 * total_coverage / (self.num_agents - 1) +
                                         0.5 * individual_coverage[i])

        # 2. 避障奖励
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1) / self.L_sensor
            rewards[i] += mu_collision * r_safe

        # 3. 任务完成奖励（达到覆盖率目标且无碰撞）
        total_sensitivity = np.sum(self.env_gen.sensitivity_map)
        coverage_ratio = total_coverage / total_sensitivity
        if coverage_ratio >= 0.8 and not any(IsCollied):
            rewards[:self.num_agents - 1] += mu_completion * 10
            dones = [True] * self.num_agents

        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        """更新激光传感器数据和碰撞状态"""
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []

            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos, obs_pos, r,
                                                      self.L_sensor, self.num_lasers,
                                                      self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)

            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)

        return dones

    def render(self):
        """渲染环境状态"""
        plt.clf()
        # 绘制敏感度地图
        plt.imshow(self.env_gen.sensitivity_map, cmap='viridis',
                   extent=[0, self.length, 0, self.length])
        plt.colorbar(label='Sensitivity')

        # 绘制UAV
        uav_icon = mpimg.imread('UAV.png')
        for i in range(self.num_agents - 1):  # 只绘制非目标UAV
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)

            # 绘制监测范围
            circle = plt.Circle(pos, self.monitor_radius, color='b', fill=False, alpha=0.3)
            plt.gca().add_patch(circle)

            # 绘制UAV图标
            angle = np.arctan2(vel[1], vel[0])
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

        # 绘制障碍物
        for obs in self.obstacles:
            circle = plt.Circle(obs.position, obs.radius, color='black', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()
        plt.legend()

        # 转换为图像
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        return image


class obstacle:
    def __init__(self, length=2, is_dynamic=False):
        self.position = np.random.uniform(low=0.45, high=length - 0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        if is_dynamic:
            speed = np.random.uniform(0.01, 0.03)
            self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        else:
            self.velocity = np.zeros(2)
        self.radius = np.random.uniform(0.1, 0.15)
        self.is_dynamic = is_dynamic