from maddpg import MADDPG
from sim_env import UAVEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings('ignore')


# -------------------------------
# 环境敏感度生成模块（用于热力图与轨迹颜色映射）
# -------------------------------
class EnvironmentGenerator:
    def __init__(self, grid_size=(100, 100)):
        self.grid_size = grid_size
        self.sensitivity_map = self._generate_sensitivity()

    def _generate_sensitivity(self, smooth_factor=20.0):
        sensitivity = np.random.rand(*self.grid_size)
        sensitivity = gaussian_filter(sensitivity, sigma=smooth_factor)
        sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min())
        return np.clip(sensitivity, 0, 1)

    def get_sensitivity(self, pos):
        # pos: 数组，形状(..., 2)
        x = np.clip(pos[..., 0], 0, self.grid_size[0] - 1).astype(int)
        y = np.clip(pos[..., 1], 0, self.grid_size[1] - 1).astype(int)
        return self.sensitivity_map[x, y]


# -------------------------------
# 可视化系统：布局为两行两列
#   - 左上：无人机轨迹（采用轨迹跟踪逻辑，颜色由环境敏感度映射）
#   - 右上：热力图（显示环境敏感度）
#   - 下方：无人机速度显示（与之前保持一致）
# 为使无人机运动更明显，轨迹与热力图显示区域采用缩放（例如 0.5，即 50m×50m）
# -------------------------------
class VisualizationSystem:
    def __init__(self, grid_size=(100, 100), num_uavs=4, scale_factor=0.5):
        self.grid_size = grid_size
        self.num_uavs = num_uavs
        self.scale_factor = scale_factor
        self.env_gen = EnvironmentGenerator(grid_size=grid_size)
        # 显示区域尺寸：缩放后的尺寸
        self.display_grid = (self.grid_size[0] * self.scale_factor, self.grid_size[1] * self.scale_factor)

        # 利用 GridSpec 布局：上排2个子图（左：轨迹，右：热力图），下排1个子图（速度显示）
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        self.ax_heat = self.fig.add_subplot(gs[0, 1])
        self.ax_vel = self.fig.add_subplot(gs[1, :])

        # 初始化左上：无人机轨迹图，设置缩放后尺寸和相等比例
        self.ax_traj.set_xlim(0, self.display_grid[0])
        self.ax_traj.set_ylim(0, self.display_grid[1])
        self.ax_traj.set_title('UAV Trajectories (Scaled)')
        self.ax_traj.set_aspect('equal', adjustable='box')

        # 初始化右上：热力图，采用相同缩放后尺寸
        self.bg_layer = self.ax_heat.imshow(
            self.env_gen.sensitivity_map.T,
            cmap='RdYlGn_r',
            alpha=0.4,
            extent=[0, self.display_grid[0], 0, self.display_grid[1]],
            origin='lower',
            zorder=0
        )
        self.ax_heat.set_xlim(0, self.display_grid[0])
        self.ax_heat.set_ylim(0, self.display_grid[1])
        self.ax_heat.set_title('Environmental Sensitivity Heatmap (Scaled)')
        self.ax_heat.set_aspect('equal', adjustable='box')

        # 初始化下方：无人机速度显示
        self.velocity_lines = []
        self.velocity_texts = []
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_uavs))
        for i in range(self.num_uavs):
            label = f'UAV {i}' if i != 3 else 'Target'
            line, = self.ax_vel.plot([], [], color=colors[i], lw=1.5, label=label)
            self.velocity_lines.append(line)
            text = self.ax_vel.text(0.95, 0.85 - i * 0.07, '', transform=self.ax_vel.transAxes,
                                    color=colors[i], fontsize=9)
            self.velocity_texts.append(text)
        self.ax_vel.set_xlim(0, 200)
        self.ax_vel.set_ylim(0, 5)
        self.ax_vel.set_xlabel('Time Step')
        self.ax_vel.set_ylabel('Velocity (m/s)')
        self.ax_vel.set_title('Real-time Velocity Monitoring')
        self.ax_vel.legend(loc='upper left')

        # 初始化左上轨迹跟踪（使用 LineCollection 实现连续轨迹，颜色映射由环境敏感度提供）
        self.trajectories = [np.empty((0, 2)) for _ in range(self.num_uavs)]
        self.traj_lines = []
        for i in range(self.num_uavs):
            lc = LineCollection([], cmap='viridis', linewidths=1.5, zorder=1)
            self.ax_traj.add_collection(lc)
            self.traj_lines.append(lc)

    def update_trajectories(self, positions):
        """
        positions: ndarray, shape (num_uavs, 2) 或列表形式的相似数据
        """
        # 将传入的 positions 转换为 NumPy 数组
        positions = np.array(positions)
        # 对位置进行缩放，以便在较小显示区域内展现得更明显
        positions_scaled = positions * self.scale_factor
        for i in range(self.num_uavs):
            new_point = np.array([[positions_scaled[i][0], positions_scaled[i][1]]])
            self.trajectories[i] = np.concatenate([self.trajectories[i], new_point])
            if len(self.trajectories[i]) >= 2:
                segments = np.stack([self.trajectories[i][:-1],
                                     self.trajectories[i][1:]], axis=1)
                # 为保证颜色映射正确，将轨迹点转换回原始坐标（即除以缩放因子）计算敏感度
                sensitivity_values = self.env_gen.get_sensitivity(self.trajectories[i] / self.scale_factor)
                self.traj_lines[i].set_segments(segments)
                self.traj_lines[i].set_array(sensitivity_values)
        return self.traj_lines

    def update_velocity_curves(self, velocities, timestep):
        """
        velocities: list 或 ndarray, 每个元素为当前各智能体速度标量（例如：速度模长）
        timestep: 当前时间步
        """
        for i in range(self.num_uavs):
            current_ydata = self.velocity_lines[i].get_ydata()
            if current_ydata.size == 0:
                new_index = 0
            else:
                new_index = len(current_ydata)
            x_data = np.append(self.velocity_lines[i].get_xdata(), new_index)
            y_data = np.append(current_ydata, velocities[i])
            self.velocity_lines[i].set_data(x_data, y_data)
            self.velocity_texts[i].set_text(f'{velocities[i]:.2f} m/s')
        if timestep > 100:
            self.ax_vel.set_xlim(timestep - 100, timestep + 20)
        return self.velocity_lines + self.velocity_texts

    def render_frame(self, frame_data):
        """
        frame_data: tuple (positions, velocities, timestep)
          - positions: 当前各智能体坐标（用于更新左上轨迹）
          - velocities: 当前各智能体速度（用于下方速度曲线更新）
          - timestep: 当前时间步
        返回所有更新的艺术对象列表（用于 blit）
        """
        positions, velocities, timestep = frame_data
        traj_artists = self.update_trajectories(positions)
        vel_artists = self.update_velocity_curves(velocities, timestep)
        return traj_artists + vel_artists


# 以下为原有速度数据平滑与绘图函数（若在仿真结束后需要展示，可解除注释）
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_velocity_magnitude(time_steps, velocities_magnitude):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_magnitude)):
        if i != 3:
            plt.plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_magnitude[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("Magnitude")
    plt.title("UAV Velocity Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocity_x(time_steps, velocities_x):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_x)):
        if i != 3:
            plt.plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_x[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_x$")
    plt.title("UAV $Vel_x$")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocity_y(time_steps, velocities_y):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_y)):
        if i != 3:
            plt.plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_y[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_y$")
    plt.title("UAV $Vel_y$")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocities(velocities_magnitude, velocities_x, velocities_y):
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(len(velocities_magnitude)):
        if i != 3:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            axs[0].plot(time_steps, velocities_magnitude[i], label='Target')
    axs[0].set_title('Speed Magnitude vs Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Speed Magnitude')
    axs[0].legend()
    for i in range(len(velocities_x)):
        if i != 3:
            axs[1].plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            axs[1].plot(time_steps, velocities_x[i], label='Target')
    axs[1].set_title('Velocity X Component vs Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity X Component')
    axs[1].legend()
    for i in range(len(velocities_y)):
        if i != 3:
            axs[2].plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            axs[2].plot(time_steps, velocities_y[i], label='Target')
    axs[2].set_title('Velocity Y Component vs Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity Y Component')
    axs[2].legend()
    plt.tight_layout()
    plt.show()


# -------------------------------
# 主程序：加载 MADDPG、初始化 UAVEnv、并使用自定义可视化系统
# -------------------------------
if __name__ == '__main__':
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    # 用于记录各智能体速度数据（仿真结束后可用于绘图）
    velocities_magnitude_data = [[] for _ in range(env.num_agents)]
    velocities_x_data = [[] for _ in range(env.num_agents)]
    velocities_y_data = [[] for _ in range(env.num_agents)]

    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128, alpha=0.0001, beta=0.003,
                           scenario='UAV_Round_up', chkpt_dir='tmp/maddpg/')
    maddpg_agents.load_checkpoint()
    print('---- Evaluating ----')

    obs = env.reset()
    total_steps = 0

    # 初始化自定义可视化系统，设置缩放因子使得轨迹图和热力图显示区域更小（例如50m×50m）
    vis_system = VisualizationSystem(grid_size=(100, 100), num_uavs=env.num_agents, scale_factor=0.5)

    def update(frame):
        global obs, total_steps, velocities_magnitude_data, velocities_x_data, velocities_y_data
        total_steps += 1

        # 记录每个智能体的速度信息，修改为整体乘以10
        for i in range(env.num_agents):
            vel = env.multi_current_vel[i]  # 假设该属性为当前各智能体速度向量
            modified_vel = np.array(vel) * 100  # 速度整体乘以10
            v_x, v_y = modified_vel
            speed = np.linalg.norm(modified_vel)
            velocities_magnitude_data[i].append(speed)
            velocities_x_data[i].append(v_x)
            velocities_y_data[i].append(v_y)

        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, _, dones = env.step(actions)

        # 从环境中获取当前无人机位置（假设属性 multi_current_pos 存在）
        positions = env.multi_current_pos
        # 下方速度图显示各智能体速度模长，乘以10
        velocities = [np.linalg.norm(np.array(env.multi_current_vel[i]) * 10) for i in range(env.num_agents)]

        # 更新自定义可视化系统：更新左上轨迹和下方速度曲线（右上热力图为静态背景）
        artists = vis_system.render_frame((positions, velocities, total_steps))

        obs = obs_
        if any(dones):
            ani.event_source.stop()
            print("Round-up finished in", frame, "steps.")
            # 如有需要，可在仿真结束后对速度数据进行平滑处理并绘图
            time_steps = range(len(velocities_magnitude_data[0]))
            plot_velocity_magnitude(time_steps, velocities_magnitude_data)
            plot_velocity_x(time_steps, velocities_x_data)
            plot_velocity_y(time_steps, velocities_y_data)
        return artists

    ani = animation.FuncAnimation(vis_system.fig, update, frames=10000, interval=20, blit=True)
    plt.show()
