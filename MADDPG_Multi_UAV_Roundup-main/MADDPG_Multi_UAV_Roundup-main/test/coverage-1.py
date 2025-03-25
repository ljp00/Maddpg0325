from maddpg import MADDPG
from sim_env import UAVEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')


# -------------------------------
# 环境敏感度生成模块（用于热力图背景）
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

    # 创建一个包含左右两个子图的图窗口
    fig, (ax_traj, ax_heat) = plt.subplots(1, 2, figsize=(15, 7))

    # 左侧子图：无人机运动轨迹
    ax_traj.set_xlim(0, 2.5)
    ax_traj.set_ylim(0, 2.5)
    ax_traj.set_title("UAV Trajectories")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.set_aspect('equal', adjustable='box')
    # 为每个 UAV 创建空的线对象
    lines = []
    for i in range(n_agents):
        label = f"UAV {i}" if i != 3 else "Target"
        line, = ax_traj.plot([], [], lw=2, label=label)
        lines.append(line)
    ax_traj.legend()

    # 右侧子图：热力图
    env_gen = EnvironmentGenerator(grid_size=(100, 100))
    heatmap = ax_heat.imshow(env_gen.sensitivity_map.T, cmap='RdYlGn_r', alpha=0.4,
                             extent=[0, 100, 0, 100], origin='lower')
    ax_heat.set_xlim(0, 100)
    ax_heat.set_ylim(0, 100)
    ax_heat.set_title("Environmental Sensitivity Heatmap")
    ax_heat.set_aspect('equal', adjustable='box')

    # 更新函数，每一帧更新 UAV 轨迹并推进仿真
    def update(frame):
        global obs, total_steps, trajectories
        total_steps += 1

        # 获取当前 UAV 坐标（假设返回形状为 (n_agents, 2) 的数组或列表）
        current_positions = env.multi_current_pos
        for i in range(n_agents):
            trajectories[i].append(current_positions[i])
            xs = [pos[0] for pos in trajectories[i]]
            ys = [pos[1] for pos in trajectories[i]]
            lines[i].set_data(xs, ys)

        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, _, dones = env.step(actions)
        obs = obs_
        if any(dones):
            ani.event_source.stop()
            print("Round-up finished in", frame, "steps.")
        return lines

    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20, blit=True)
    plt.show()
