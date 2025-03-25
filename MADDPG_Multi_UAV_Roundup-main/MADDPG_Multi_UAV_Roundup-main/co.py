from maddpg import MADDPG
from sim_env_cov import UAVEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class CoverageVisualizationSystem:
    def __init__(self, grid_size=(100, 100), num_uavs=3):
        self.grid_size = grid_size
        self.num_uavs = num_uavs
        self.env_gen = None

        # 创建图形布局
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

        # 左上：UAV轨迹
        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        self.ax_traj.set_xlim(0, 2)
        self.ax_traj.set_ylim(0, 2)
        self.ax_traj.set_title('UAV Coverage Trajectories')
        self.ax_traj.grid(True)

        # 右上：敏感度热力图
        self.ax_heat = self.fig.add_subplot(gs[0, 1])
        self.ax_heat.set_title('Sensitivity Map')

        # 下方：速度和覆盖率监控
        self.ax_metrics = self.fig.add_subplot(gs[1, :])
        self.ax_metrics.set_title('Real-time Metrics')
        self.ax_metrics.set_xlabel('Time Step')
        self.ax_metrics.set_ylabel('Value')

        # 初始化轨迹
        self.trajectories = [np.empty((0, 2)) for _ in range(self.num_uavs)]
        self.traj_lines = []
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_uavs))
        for i in range(self.num_uavs):
            line, = self.ax_traj.plot([], [], color=colors[i], label=f'UAV {i}')
            self.traj_lines.append(line)
        self.ax_traj.legend()

        # 初始化指标曲线
        self.coverage_line, = self.ax_metrics.plot([], [], 'r-', label='Coverage Reward')
        self.avg_speed_line, = self.ax_metrics.plot([], [], 'b-', label='Avg Speed')
        self.ax_metrics.legend()
        self.coverage_history = []
        self.speed_history = []

        plt.tight_layout()

    def set_sensitivity_map(self, sensitivity_map):
        """设置并显示敏感度地图"""
        self.ax_heat.clear()
        im = self.ax_heat.imshow(
            sensitivity_map.T,
            cmap='RdYlGn',
            extent=[0, 2, 0, 2],
            origin='lower'
        )
        self.ax_heat.set_title('Environment Sensitivity')
        plt.colorbar(im, ax=self.ax_heat)

    def update_trajectories(self, positions):
        """更新UAV轨迹"""
        for i in range(self.num_uavs):
            self.trajectories[i] = np.vstack([self.trajectories[i], positions[i]])
            self.traj_lines[i].set_data(self.trajectories[i][:, 0], self.trajectories[i][:, 1])
        return self.traj_lines

    def update_metrics(self, coverage_reward, avg_speed, timestep):
        """更新指标曲线"""
        self.coverage_history.append(coverage_reward)
        self.speed_history.append(avg_speed)

        x_data = np.arange(len(self.coverage_history))
        self.coverage_line.set_data(x_data, self.coverage_history)
        self.avg_speed_line.set_data(x_data, self.speed_history)

        if timestep > 100:
            self.ax_metrics.set_xlim(timestep - 100, timestep + 20)
        self.ax_metrics.set_ylim(0, max(max(self.coverage_history), max(self.speed_history)) * 1.1)

        return [self.coverage_line, self.avg_speed_line]

    def render_frame(self, frame_data):
        """渲染一帧"""
        positions, coverage_reward, velocities, timestep = frame_data

        # 更新轨迹
        traj_artists = self.update_trajectories(positions)

        # 更新指标
        avg_speed = np.mean([np.linalg.norm(v) for v in velocities])
        metric_artists = self.update_metrics(coverage_reward, avg_speed, timestep)

        return traj_artists + metric_artists


def evaluate_coverage():
    # 初始化环境和智能体
    env = UAVEnv(num_agents=3)
    n_agents = env.num_agents

    # 设置MADDPG
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    n_actions = 2

    maddpg_agents = MADDPG(
        actor_dims=actor_dims,
        critic_dims=critic_dims,
        n_agents=n_agents,
        n_actions=n_actions,
        fc1=128,
        fc2=128,
        alpha=0.0001,
        beta=0.003,
        scenario='UAV_Round_up',
        chkpt_dir='tmp/maddpg/'
        #chkpt_dir = 'training_logs/20250325_021550/checkpointsUAV_Round_up'
    )

    # 加载训练好的模型
    try:
        maddpg_agents.load_checkpoint()
        print('Successfully loaded checkpoint')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 初始化可视化系统
    vis_system = CoverageVisualizationSystem(num_uavs=n_agents)

    # 重置环境并设置敏感度地图
    obs = env.reset()
    vis_system.set_sensitivity_map(env.env_gen.sensitivity_map)

    total_steps = 0

    def update(frame):
        nonlocal obs, total_steps
        total_steps += 1

        # 选择并执行动作
        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, rewards, dones = env.step(actions)

        # 计算覆盖奖励
        coverage_reward, _ = env.compute_coverage_reward()

        # 准备可视化数据
        positions = env.multi_current_pos
        velocities = env.multi_current_vel

        # 更新可视化
        artists = vis_system.render_frame((
            positions,
            coverage_reward,
            velocities,
            total_steps
        ))

        obs = obs_
        if any(dones) or total_steps >= 200:  # 最多运行200步
            ani.event_source.stop()
            print(f"Evaluation finished at step {total_steps}")
            print(f"Final coverage reward: {coverage_reward:.4f}")

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f'evaluation_results_{timestamp}'
            os.makedirs(save_dir, exist_ok=True)

            # 保存最终图像
            plt.savefig(f'{save_dir}/final_state.png')

            # 保存覆盖率历史
            np.save(f'{save_dir}/coverage_history.npy', vis_system.coverage_history)

        return artists

    # 创建动画
    ani = animation.FuncAnimation(
        vis_system.fig,
        update,
        frames=1000,
        interval=50,
        blit=True
    )

    plt.show()


if __name__ == '__main__':
    evaluate_coverage()