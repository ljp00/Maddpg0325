import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# 环境敏感度生成模块（稳定版）
class EnvironmentGenerator:
    def __init__(self, grid_size=(100, 100)):
        self.grid_size = grid_size
        self.sensitivity_map = self._generate_sensitivity()

    def _generate_sensitivity(self, smooth_factor=20.0):
        """生成带高斯平滑的敏感度地图"""
        sensitivity = np.random.rand(*self.grid_size)
        sensitivity = gaussian_filter(sensitivity, sigma=smooth_factor)
        sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min())
        return np.clip(sensitivity, 0, 1)

    def get_sensitivity(self, pos):
        """优化坐标转换逻辑"""
        x = np.clip(pos[..., 0], 0, self.grid_size[0] - 1).astype(int)
        y = np.clip(pos[..., 1], 0, self.grid_size[1] - 1).astype(int)
        return self.sensitivity_map[x, y]

# 高性能可视化系统
class VisualizationSystem:
    def __init__(self, env_generator, num_uavs=4):
        self.env = env_generator
        self.num_uavs = num_uavs

        # 使用 GridSpec 将图形分为两行两列，第一行分左右两部分，第二行跨两列
        self.fig = plt.figure(figsize=(15, 10))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1])
        self.ax_traj = self.fig.add_subplot(gs[0, 0])  # 左上：无人机运动轨迹图
        self.ax_heat = self.fig.add_subplot(gs[0, 1])   # 右上：环境敏感度热力图
        self.ax_vel = self.fig.add_subplot(gs[1, :])    # 下方：无人机速度显示

        self._init_trajectory_plot()
        self._init_heatmap_layer()
        self._init_velocity_display()
        self._init_trajectory_tracking()

    def _init_trajectory_plot(self):
        """初始化无人机运动轨迹显示区域"""
        self.ax_traj.set(
            xlim=(0, self.env.grid_size[0]),
            ylim=(0, self.env.grid_size[1]),
            title='UAV Trajectories'
        )

    def _init_heatmap_layer(self):
        """初始化环境敏感度热力图显示区域"""
        self.bg_layer = self.ax_heat.imshow(
            self.env.sensitivity_map.T,
            cmap='RdYlGn_r',
            alpha=0.4,
            extent=[0, self.env.grid_size[0], 0, self.env.grid_size[1]],
            origin='lower',
            zorder=0
        )
        self.ax_heat.set(
            xlim=(0, self.env.grid_size[0]),
            ylim=(0, self.env.grid_size[1]),
            title='Environmental Sensitivity Heatmap'
        )

    def _init_velocity_display(self):
        """初始化无人机速度显示组件"""
        self.velocity_lines = []
        self.velocity_texts = []
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_uavs))

        for i in range(self.num_uavs):
            # 初始化速度曲线
            line, = self.ax_vel.plot([], [],
                                      color=colors[i],
                                      lw=1.5,
                                      label=f'UAV {i + 1}')
            self.velocity_lines.append(line)

            # 初始化实时速度文本
            text = self.ax_vel.text(
                0.95, 0.85 - i * 0.07, '',
                transform=self.ax_vel.transAxes,
                color=colors[i],
                fontsize=9
            )
            self.velocity_texts.append(text)

        self.ax_vel.set(
            xlim=(0, 200),
            ylim=(0, 5),
            xlabel='Time Step',
            ylabel='Velocity (m/s)',
            title='Real-time Velocity Monitoring'
        )
        self.ax_vel.legend(loc='upper left')

    def _init_trajectory_tracking(self):
        """初始化轨迹跟踪组件，将 LineCollection 添加到无人机运动轨迹图中"""
        self.trajectories = [np.empty((0, 2)) for _ in range(self.num_uavs)]
        self.traj_lines = [
            LineCollection([],
                           cmap='viridis',
                           linewidths=1.5,
                           zorder=1)
            for _ in range(self.num_uavs)
        ]
        for line in self.traj_lines:
            self.ax_traj.add_collection(line)

    def _update_trajectories(self, positions):
        """轨迹更新逻辑（优化版）"""
        for i in range(self.num_uavs):
            # 更新轨迹点
            new_point = np.array([[positions[i][0], positions[i][1]]])
            self.trajectories[i] = np.concatenate([self.trajectories[i], new_point])

            # 生成线段数据并映射敏感度颜色
            if len(self.trajectories[i]) >= 2:
                segments = np.stack([
                    self.trajectories[i][:-1],
                    self.trajectories[i][1:]
                ], axis=1)
                sensitivity_values = self.env.get_sensitivity(self.trajectories[i])
                self.traj_lines[i].set_segments(segments)
                self.traj_lines[i].set_array(sensitivity_values)

    def _update_velocity_curves(self, velocities, timestep):
        """速度曲线更新逻辑（稳定版）"""
        for i in range(self.num_uavs):
            x_data = np.arange(len(self.velocity_lines[i].get_ydata()) + 1)
            y_data = np.append(self.velocity_lines[i].get_ydata(), velocities[i])
            self.velocity_lines[i].set_data(x_data, y_data)
            self.velocity_texts[i].set_text(f'{velocities[i]:.2f} m/s')

        # 自动滚动显示
        if timestep > 100:
            self.ax_vel.set_xlim(timestep - 100, timestep + 20)

    def render_frame(self, frame_data):
        """帧渲染主函数"""
        positions, velocities, timestep = frame_data
        self._update_trajectories(positions)
        self._update_velocity_curves(velocities, timestep)
        return self.traj_lines + self.velocity_lines + self.velocity_texts

# 无人机仿真系统（类型安全版）
class UAVSimulator:
    def __init__(self, grid_size=(100, 100)):
        self.positions = np.random.rand(4, 2) * grid_size[0]
        self.velocities = np.random.rand(4) * 2 + 0.5
        self.directions = np.random.randn(4, 2)
        self.directions /= np.linalg.norm(self.directions, axis=1, keepdims=True)

    def simulation_step(self):
        """仿真步进（边界处理优化）"""
        self.positions += self.directions * self.velocities[:, np.newaxis]

        # 边界反弹逻辑
        over_boundary = (self.positions < 0) | (self.positions > 100)
        self.directions[over_boundary] *= -1
        self.positions = np.clip(self.positions, 0, 100)

        return self.positions.copy(), self.velocities.copy()

# 主程序执行
if __name__ == '__main__':
    # 初始化各组件
    env = EnvironmentGenerator(grid_size=(100, 100))
    visualizer = VisualizationSystem(env)
    simulator = UAVSimulator()

    # 动画回调函数
    def animation_update(frame):
        positions, velocities = simulator.simulation_step()
        return visualizer.render_frame((positions, velocities, frame))

    # 配置动画参数
    ani = animation.FuncAnimation(
        visualizer.fig,
        animation_update,
        frames=1000,
        interval=50,
        blit=True,
        repeat=False
    )

    # 为热力图添加颜色条
    cbar = visualizer.fig.colorbar(
        visualizer.bg_layer,
        ax=visualizer.ax_heat,
        label='Environmental Sensitivity'
    )

    plt.tight_layout()
    plt.show()
