import numpy as np
from maddpg import MADDPG
from sim_env_cov import UAVEnv
from buffer import MultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import shutil
from datetime import datetime

warnings.filterwarnings('ignore')


class TrainingLogger:
    def __init__(self, base_dir='training_logs'):
        self.start_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"{base_dir}/{self.start_time}"
        self.user = "ljp00"
        self.create_directories()

    def create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.base_dir,
            f"{self.base_dir}/checkpoints",
            f"{self.base_dir}/images",
            f"{self.base_dir}/backups",
            f"{self.base_dir}/plots"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def log_message(self, message, level="INFO"):
        """记录日志消息"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        with open(f"{self.base_dir}/training.log", 'a') as f:
            f.write(log_message + '\n')
        if level in ["ERROR", "WARNING"]:
            print(log_message)


def obs_list_to_state_vector(obs):
    """将观察列表转换为状态向量"""
    state = np.hstack([np.ravel(o) for o in obs])
    return state


def save_image(env_render, filename):
    """保存环境渲染图像"""
    image = Image.fromarray(env_render, 'RGBA')
    image = image.convert('RGB')
    image.save(filename)


def plot_training_progress(score_history, save_path):
    """绘制训练进度图"""
    plt.figure(figsize=(12, 6))

    # 绘制原始分数
    plt.plot(score_history, 'b-', alpha=0.3, label='Raw Scores')

    # 绘制移动平均
    window_size = 100
    moving_avg = np.convolve(score_history, np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size - 1, len(score_history)), moving_avg, 'r-', label=f'Moving Average (n={window_size})')

    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def save_training_state(save_dir, episode, score_history, avg_score, total_steps, best_score):
    """保存训练状态"""
    training_state = {
        'episode': episode,
        'score_history': score_history,
        'avg_score': avg_score,
        'total_steps': total_steps,
        'best_score': best_score,
        'timestamp': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    np.save(f"{save_dir}/training_state.npy", training_state)


def load_training_state(save_dir):
    """加载训练状态"""
    state_file = f"{save_dir}/training_state.npy"
    if os.path.exists(state_file):
        return np.load(state_file, allow_pickle=True).item()
    return None


def backup_training_data(episode, base_dir, logger):
    """创建训练数据备份"""
    backup_dir = f"{base_dir}/backups/episode_{episode}"
    try:
        os.makedirs(backup_dir, exist_ok=True)
        # 复制重要文件
        shutil.copy2(f"{base_dir}/training_state.npy", backup_dir)
        shutil.copytree(f"{base_dir}/checkpoints", f"{backup_dir}/checkpoints", dirs_exist_ok=True)
        shutil.copy2(f"{base_dir}/score_history.csv", backup_dir)
        logger.log_message(f"Successfully created backup at episode {episode}")
    except Exception as e:
        logger.log_message(f"Backup failed at episode {episode}: {str(e)}", "ERROR")


if __name__ == '__main__':
    # 初始化日志记录器
    logger = TrainingLogger()
    logger.log_message("Starting training session")

    # 初始化环境
    env = UAVEnv(num_agents=3)
    n_agents = env.num_agents

    # 记录环境信息
    logger.log_message(f"Initialized environment with {n_agents} agents")
    logger.log_message(f"Observation space dimensions: {env.observation_space['agent_0'].shape[0]}")

    # 设置智能体的观察空间和动作空间维度
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    n_actions = 2

    logger.log_message(f"Actor dimensions: {actor_dims}")
    logger.log_message(f"Critic dimensions: {critic_dims}")

    # 初始化MADDPG智能体
    maddpg_agents = MADDPG(actor_dims=actor_dims,
                           critic_dims=critic_dims,
                           n_agents=n_agents,
                           n_actions=n_actions,
                           fc1=128,
                           fc2=128,
                           alpha=0.0001,
                           beta=0.003,
                           scenario='UAV_Round_up',
                           chkpt_dir=f'{logger.base_dir}/checkpoints')

    # 初始化经验回放缓冲区
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=256)

    # 训练参数设置
    PRINT_INTERVAL = 100
    N_GAMES = 5000
    MAX_STEPS = 100
    BACKUP_INTERVAL = 500

    # 尝试加载之前的训练状态
    training_state = load_training_state(logger.base_dir)
    if training_state is not None:
        start_episode = training_state['episode'] + 1
        score_history = training_state['score_history']
        total_steps = training_state['total_steps']
        best_score = training_state['best_score']
        logger.log_message(f"Resumed training from episode {start_episode}")
    else:
        start_episode = 0
        score_history = []
        total_steps = 0
        best_score = -30
        logger.log_message("Starting new training session")

    # 记录训练开始时间
    training_start_time = time.time()

    # 主训练循环
    for i in range(start_episode, N_GAMES):
        episode_start_time = time.time()
        obs = env.reset()
        score = 0
        dones = [False] * n_agents
        episode_step = 0

        # Episode循环
        while not any(dones):
            # 选择动作并执行
            actions = maddpg_agents.choose_action(obs, episode_step, evaluate=False)
            obs_, rewards, dones = env.step(actions)

            # 存储经验
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            # 检查是否达到最大步数
            if episode_step >= MAX_STEPS:
                dones = [True] * n_agents

            # 训练
            if total_steps % 10 == 0:
                maddpg_agents.learn(memory, total_steps)

            # 更新状态
            obs = obs_
            score += sum(rewards)
            total_steps += 1
            episode_step += 1

        # 更新训练统计
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # 保存最佳模型
        if avg_score > best_score:
            maddpg_agents.save_checkpoint()
            best_score = avg_score
            logger.log_message(f"New best score: {best_score:.2f}")

        # 定期记录和备份
        if i % PRINT_INTERVAL == 0 and i > 0:
            episode_time = time.time() - episode_start_time
            elapsed_time = time.time() - training_start_time
            remaining_episodes = N_GAMES - i
            estimated_time = (elapsed_time / i) * remaining_episodes

            logger.log_message(
                f"Episode: {i}/{N_GAMES}, "
                f"Average Score: {avg_score:.2f}, "
                f"Episode Time: {episode_time:.2f}s, "
                f"Estimated Remaining Time: {estimated_time / 3600:.2f}h"
            )

            # 绘制训练进度图
            plot_training_progress(score_history, f"{logger.base_dir}/plots/training_progress.png")

            # 保存训练状态
            save_training_state(logger.base_dir, i, score_history, avg_score, total_steps, best_score)

        # 定期备份
        if i % BACKUP_INTERVAL == 0 and i > 0:
            backup_training_data(i, logger.base_dir, logger)

    # 训练结束，保存最终结果
    logger.log_message("Training completed")
    final_score_file = f"{logger.base_dir}/score_history.csv"
    pd.DataFrame(score_history).to_csv(final_score_file, header=False, index=False)

    # 绘制最终训练进度图
    plot_training_progress(score_history, f"{logger.base_dir}/plots/final_training_progress.png")

    # 记录训练总结
    total_training_time = time.time() - training_start_time
    logger.log_message(f"Total training time: {total_training_time / 3600:.2f} hours")
    logger.log_message(f"Final average score: {np.mean(score_history[-100:]):.2f}")
    logger.log_message(f"Best score achieved: {best_score:.2f}")
    logger.log_message(f"Total steps: {total_steps}")