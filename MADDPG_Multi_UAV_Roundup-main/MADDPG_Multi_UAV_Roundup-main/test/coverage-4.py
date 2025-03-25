from maddpg import MADDPG
from sim_env import UAVEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

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
        x = np.clip(pos[..., 0], 0, self.grid_size[0] - 1).astype(int)
        y = np.clip(pos[..., 1], 0, self.grid_size[1] - 1).astype(int)
        return self.sensitivity_map[x, y]

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

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


if __name__ == '__main__':
    # 初始化环境和 MADDPG
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128, alpha=0.0001, beta=0.003,
                           scenario='UAV_Round_up', chkpt_dir='tmp/maddpg/')
    maddpg_agents.load_checkpoint()
    print("---- Evaluating ----")

    obs = env.reset()
    total_steps = 0

    # 为每个 UAV 创建轨迹记录容器
    trajectories = [[] for _ in range(n_agents)]

    # 设置左侧轨迹图的缩放因子
    zoom_scale = 0.5  # 将环境范围缩放到 0～50，便于突出 UAV 运动

    # 创建一个包含左右两个子图的图窗口
    fig, (ax_traj, ax_heat) = plt.subplots(1, 2, figsize=(15, 7))

    # 左侧子图：无人机运动轨迹（采用缩放后的坐标）
    ax_traj.set_xlim(0, 100 * zoom_scale)
    ax_traj.set_ylim(0, 100 * zoom_scale)
    ax_traj.set_title("UAV Trajectories (Zoomed)")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.set_aspect('equal', adjustable='box')
    # 为每个 UAV 创建空的线对象（增加 marker 显示 UAV 当前位置）
    lines = []
    for i in range(n_agents):
        label = f"UAV {i}" if i != 3 else "Target"
        line, = ax_traj.plot([], [], lw=2, marker='o', markersize=8, label=label)
        lines.append(line)
    ax_traj.legend()

    # 右侧子图：热力图（显示整个 100×100 环境）
    env_gen = EnvironmentGenerator(grid_size=(100, 100))
    heatmap = ax_heat.imshow(env_gen.sensitivity_map.T, cmap='RdYlGn_r', alpha=0.4,
                             extent=[0, 100, 0, 100], origin='lower')
    ax_heat.set_xlim(0, 100)
    ax_heat.set_ylim(0, 100)
    ax_heat.set_title("Environmental Sensitivity Heatmap")
    ax_heat.set_aspect('equal', adjustable='box')

    # 更新函数，每一帧更新 UAV 轨迹并推进仿真
    def update(frame):
        nonlocal total_steps, trajectories, obs
        total_steps += 1

        # 获取当前 UAV 坐标（确保 UAVEnv 提供正确的方法）
        current_positions = np.array(env.get_positions())  # 检查 `get_positions()` 是否存在
        display_positions = np.clip(current_positions * zoom_scale, 0, 100 * zoom_scale)

        for i in range(n_agents):
            trajectories[i].append(display_positions[i])
            xs = [pos[0] for pos in trajectories[i]]
            ys = [pos[1] for pos in trajectories[i]]
            lines[i].set_data(xs, ys)

        actions = maddpg_agents.choose_action(obs, evaluate=True)
        obs_, _, dones, _ = env.step(actions)  # 确保 step() 返回四个值
        obs = obs_

        if any(dones):
            ani.event_source.stop()
            print("Round-up finished in", frame, "steps.")

        return lines


    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20, blit=True)
    plt.show()
