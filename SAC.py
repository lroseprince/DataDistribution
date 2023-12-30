import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
from torch.distributions import Normal
from torch.optim import Adam
from environment import Env
from options import args_parser


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# ---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))

world = 'sac'

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q1 = nn.Linear(hidden_dim, 1)

        # Q2
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        x_state_action = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1_q1(x_state_action))
        x1 = F.relu(self.linear2_q1(x1))
        x1 = F.relu(self.linear3_q1(x1))
        x1 = self.linear4_q1(x1)

        x2 = F.relu(self.linear1_q2(x_state_action))
        x2 = F.relu(self.linear2_q2(x2))
        x2 = F.relu(self.linear3_q2(x2))
        x2 = self.linear4_q2(x2)

        return x1, x2


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        #print("mean:",mean)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        #print("x_t:", x_t)
        action = torch.tanh(x_t)
        #print("action:", action)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std


class SAC(object):
    def __init__(self, state_dim,
                 action_dim, gamma=0.99,
                 tau=1e-2,
                 alpha=0.2,
                 hidden_dim=256,
                 lr=0.0003):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, state, eval=False):
        #print("state:", state)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)
        
        action = action.detach().cpu().numpy()[0]
        return action

    def update_parameters(self, memory, batch_size):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  #
        qf2_loss = F.mse_loss(qf2, next_q_value)  #
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, mean, log_std = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        # Regularization Loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        soft_update(self.critic_target, self.critic, self.tau)

    # Save model parameters
    def save_models(self, episode_count):
        torch.save(self.policy.state_dict(),
                   dirPath + '/SAC_model/' + world + '/' + str(episode_count) + '_policy_net.pth')
        torch.save(self.critic.state_dict(),
                   dirPath + '/SAC_model/' + world + '/' + str(episode_count) + 'value_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    # Load model parameters
    def load_models(self, episode):
        self.policy.load_state_dict(
            torch.load(dirPath + '/SAC_model/' + world + '/' + str(episode) + '_policy_net.pth'))
        self.critic.load_state_dict(torch.load(dirPath + '/SAC_model/' + world + '/' + str(episode) + 'value_net.pth'))
        hard_update(self.critic_target, self.critic)
        print('***Models load***')



is_training = True
max_episodes = 1001
max_steps = 10
average_rewards = []
rewards = []
batch_size = 256
action_dim = 3
state_dim = 6

ACTION_1_MIN = -0.5  #
ACTION_2_MIN = -0.5  #
ACTION_3_MIN = -0.5  #
ACTION_1_MAX = 0.5  #
ACTION_2_MAX = 0.5  #
ACTION_3_MAX = 0.5  #
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
            #print('state___', state)
            if is_training and not ep % 10 == 0:
                # print('state___', state)
                action = agent.select_action(state)
            else:
                action = agent.select_action(state, eval=True)
            if not is_training:
                action = agent.select_action(state, eval=True)
            unnorm_action = np.array([action_unnormalized(action[0], ACTION_1_MAX, ACTION_1_MIN),
                                      action_unnormalized(action[1], ACTION_2_MAX, ACTION_2_MIN),
                                      action_unnormalized(action[2], ACTION_3_MAX, ACTION_3_MIN)
                                     ])
            # print("action:", action)
            print("unnorm_action:", unnorm_action)
            next_state, reward, distance, done, best_distribution = env.step(unnorm_action, state)
            print("reward:", reward)
            if distance < min_distance:
                min_distance = distance
                min_distribution = state[args.kind]
            # print("success distribution:", best_distribution)
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

        print('episode reward: ' + str(rewards_current_episode))
        # print('reward average per ep: ' + str(rewards_current_episode/(ep+1)) + ' and break step: ' + str(step))
