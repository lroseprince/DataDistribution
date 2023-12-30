"""
@Time: 2023/12/29 15:12
@Author: lroseprince
@File:environment.py
@Description:推测数据分布任务的强化学习环境类
"""
import copy
import torch
import numpy as np
from utils import build_model, vicious_load_dataset, reinforce_control_iid_num, judge_index, get_output_params, \
    init_distribution
from Update import LocalUpdate


class Env:
    def __init__(self, args, w_global_new, w_global_old):
        """
        env类的初始化
        :param args: 设定参数的argparse
        """
        self.args = args
        self.distance_old = torch.tensor(99, dtype=torch.float32)  # 用来保存上一次的距离
        self.net = build_model(args=args)  # 恶意用户训练的模型
        self.data_train, _ = vicious_load_dataset(args=args)
        self.w_global_new = w_global_new  # 较新轮的模型参数权重
        self.w_global_old = w_global_old  # 较旧轮的模型参数权重
        self.index = judge_index()
        if len(self.index) != args.kind:
            raise ValueError("判断失败！")

    def get_state(self, distribution):
        """
        获取权重，分布，数据集的大小以及种类等状态信息，合并为列表并返回；同时需要判断任务是否完成，是则返回done
        :return:
        """
        s_1 = torch.mean(get_output_params(self.net.state_dict(), self.args), dim=1).tolist()[:self.args.kind]
        state = distribution + s_1
        return state

    def get_reward(self):
        """
        如果done，则reward+1；如果t时刻权重参数与恶意用户参数的KL散度小于t+1时刻，则reward+0.1，反之-0.1
        :return:
        """
        done = False
        reward = 0.
        tmp_output_vicious = get_output_params(self.net.state_dict(), self.args)
        tmp_output_global_new = get_output_params(self.w_global_new, self.args)
        distance = 1e2*torch.pairwise_distance(torch.mean(tmp_output_vicious,dim=1), torch.mean(tmp_output_global_new, dim=1))
        # reward = distance  # 这个值后面看看具体多大，看情况对其放缩
        print("distance:", distance)
        if distance < self.distance_old:
            reward += 0.1
        else:
            reward -= 0.1
        self.distance_old = distance
        if distance < 0.2:
            done = True
            reward = 1.
        return reward, distance, done

    def step(self, action, state):
        """
        恶意用户进行训练，得到下一步的环境
        :param state:
        :param action:
        :return:
        """
        best_distribution = []
        self.net.load_state_dict(self.w_global_old)  # 加载旧的global模型参数，在此基础上进行训练
        tmp_state = state.tolist()
        idxs_vicious = reinforce_control_iid_num(dataset=self.data_train, alpha=tmp_state[:self.args.kind], args=self.args)
        local_vicious = LocalUpdate(args=self.args, dataset=self.data_train, idxs=idxs_vicious)
        w_vicious, _ = local_vicious.train(net=copy.deepcopy(self.net).to(self.args.device))
        state_next = self.do_action(state[:self.args.kind], action) + \
                     torch.mean(get_output_params(w_vicious, self.args), dim=1).tolist()[:self.args.kind]
        #print("w_vicious:", w_vicious)
        reward, distance, done = self.get_reward()
        if done:
            best_distribution.append(state[:self.args.kind])
        return state_next, reward, distance, done, best_distribution

    def do_action(self, distribution, action):
        """
        state中的distribution+action
        :param distribution:
        :param action:
        :return:
        """
        # print("distribution:", distribution)
        for i in range(self.args.kind):
            # print("distribution[i]:",distribution[i])
            # print("action[i]:",action[i])
            distribution[i] += action[i]
            # if action[i] >= 0:
            #     if distribution[i] + action[i] <= 1.0:
            #         distribution[i] += action[i]
            # else:
            #     if distribution[i] + action[i] >= 0.0:
            #         distribution[i] += action[i]
        # print("dis:",type(distribution))
        return distribution.tolist()


    def reset(self):
        """
        如果done或者超过max_steps都需要reset，状态分布重置为初始状态，比如可以都为0.1
        :return:
        """
        return self.get_state(init_distribution(self.args))
