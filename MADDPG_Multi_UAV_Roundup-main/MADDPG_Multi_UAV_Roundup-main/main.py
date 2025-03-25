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

warnings.filterwarnings('ignore')

def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state

def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency
    image.save(filename)

if __name__ == '__main__':
    # 初始化环境
    env = UAVEnv(num_agents=3)  # 修改：明确指定3个智能体
    n_agents = env.num_agents

    # 打印维度信息进行调试
    print("Number of agents:", n_agents)
    print("Observation space for each agent:", env.observation_space['agent_0'].shape[0])

    # 设置智能体的观察空间和动作空间维度
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    n_actions = 2  # 动作空间维度（x和y方向的加速度）

    print("Actor dimensions:", actor_dims)
    print("Critic dimensions:", critic_dims)

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
                          chkpt_dir='tmp/maddpg/')

    # 初始化经验回放缓冲区
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                  n_actions, n_agents, batch_size=256)

    # 训练参数设置
    PRINT_INTERVAL = 100
    N_GAMES = 5000
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    evaluate = False  # 设置为True时进行评估，False时进行训练
    best_score = -30

    # 根据模式选择是加载检查点还是开始训练
    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')

    # 主循环
    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        dones = [False] * n_agents
        episode_step = 0

        # 每个episode的步骤
        while not any(dones):
            # 评估模式下进行渲染并保存图像
            if evaluate:
                env_render = env.render()
                if episode_step % 10 == 0:
                    filename = f'images/episode_{i}_step_{episode_step}.png'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    save_image(env_render, filename)

            # 选择动作并执行
            actions = maddpg_agents.choose_action(obs, total_steps, evaluate)
            obs_, rewards, dones = env.step(actions)

            # 将观察结果转换为状态向量
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            # 检查是否达到最大步数
            if episode_step >= MAX_STEPS:
                dones = [True] * n_agents

            # 存储转换
            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            # 训练模式下定期学习
            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory, total_steps)

            # 更新状态和分数
            obs = obs_
            score += sum(rewards)  # 修改：直接使用所有智能体的奖励总和
            total_steps += 1
            episode_step += 1

        # 记录得分历史
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # 训练模式下保存最佳模型
        if not evaluate:
            if i % PRINT_INTERVAL == 0 and i > 0 and avg_score > best_score:
                print('New best score', avg_score, '>', best_score, 'saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score

        # 定期打印进度
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))

    # 保存得分历史
    file_name = 'score_history.csv'
    if not os.path.exists(file_name):
        pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)