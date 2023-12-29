import numpy as np
import time
import copy

import torch

from environment import Env
from options import args_parser
from reinforcement_models import action_unnormalized, SAC, ReplayBuffer

start_time = time.time()

is_training = True
max_episodes = 2001
max_steps = 500
average_rewards = []
rewards = []
batch_size = 256
action_dim = 3
state_dim = 8

ACTION_1_MIN = 0.0  #
ACTION_2_MIN = 1.0  #
ACTION_1_MAX = 0.  #
ACTION_2_MAX = 1.  #

replay_buffer_size = 10000

agent = SAC(state_dim, action_dim)
replay_buffer = ReplayBuffer(replay_buffer_size)

min_distance = 99  # 用来保存记录到的最小距离和对应分布
min_distribution = None

#agent.load_models(40)

print('State Dimensions: ' + str(state_dim))
print('Action Dimensions: ' + str(action_dim))

if __name__ == '__main__':
    args = args_parser()  # 加载参数
    # 将single模型那边的模型参数进行加载
    w_global_new = torch.load("./weights/epoch1client_globalClient_mnist_single_kind1_num1.pth")
    w_global_old = torch.load("./weights/epoch0client_globalClient_mnist_single_kind1_num1.pth")
    env = Env(args, w_global_new, w_global_old)
    before_training = 4
    past_action = np.array([0., 0.])

    for ep in range(max_episodes):
        done = False
        state = env.reset()

        if is_training and not ep % 10 == 0 and len(replay_buffer) > before_training * batch_size:
            print('Episode: ' + str(ep) + ' training')
        else:
            if len(replay_buffer) > before_training * batch_size:
                print('Episode: ' + str(ep) + ' evaluating')
            else:
                print('Episode: ' + str(ep) + ' adding to memory')

        rewards_current_episode = 0.

        for step in range(max_steps):
            state = np.float32(state)
            # print('state___', state)
            if is_training and not ep % 10 == 0:
                action = agent.select_action(state)
            else:
                action = agent.select_action(state, eval=True)

            if not is_training:
                action = agent.select_action(state, eval=True)
            unnorm_action = np.array([action_unnormalized(action[0], ACTION_V_MAX, ACTION_V_MIN),
                                      action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)])

            next_state, reward, distance, done = env.step(action, state)
            if distance < min_distance:
                min_distance = distance
                min_distribution = state[args.kind]

            rewards_current_episode += reward
            next_state = np.float32(next_state)
            if not ep % 10 == 0 or not len(replay_buffer) > before_training * batch_size:
                replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) > before_training * batch_size and is_training and not ep % 10 == 0:
                agent.update_parameters(replay_buffer, batch_size)
            state = copy.deepcopy(next_state)

            if done:
                break

        rewards.append(rewards_current_episode)

        print('reward per ep: ' + str(rewards_current_episode))
        print('reward average per ep: ' + str(rewards_current_episode/(ep+1)) + ' and break step: ' + str(step))