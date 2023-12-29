'''
@Time: 2023/12/29 15:12
@Author: lroseprince
@File:env.py
@Description:推测数据分布任务的强化学习环境类
'''
import copy

import torch
import numpy as np

from utils import build_model, vicious_load_dataset, reinforce_control_iid_num
from Update import LocalUpdate

class Env:
    def __init__(self, args):
        '''

        :param args: 设定参数的argparse
        '''
        self.state = None  # 当前状态
        self.action = None  # 动作
        self.reward = None  # 奖励
        self.done = None  # 用来标记回合结束
        self.distance_old = None  # 用来保存上一次的距离
        self.net = build_model(args=args)  # 恶意用户训练的模型
        self.data_train, _ = vicious_load_dataset(args=args)
        self.w_global_new = torch.load(args.w_global_new_path)  # 较新轮的模型参数权重
        self.w_global_old = torch.load(args.w_global_old_path)  # 较旧轮的模型参数权重
        self.data_type = args.kind_num  # 有多少种类
        self.data_size = args.data_size  # 数据量大小
        self.cuda = args.cuda  # 是否使用cuda

    # 获取权重，分布，数据集的大小以及种类等状态信息，合并为列表并返回；同时需要判断任务是否完成，是则返回done
    def getState(self, w_global_output_old, distribution, data_type, data_size):
        done = False
        s_1 = torch.mean(w_global_output_old, dim=1).tolist()
        state = s_1 + distribution + [data_type, data_size]

        return state, done

    # 如果done，则reward+1；如果t时刻权重参数与恶意用户参数的KL散度小于t+1时刻，则reward+0.1，反之-0.1
    # w_vicious_output,恶意用户训练模型的输出层参数，w_global_output，联邦学习环境较新论的模型输出层参数，distance_old,传递记录旧的距离
    def setReward(self, state, done, w_vicious_output, w_global_output, distance_old):

        distance = torch.pairwise_distance(torch.mean(w_vicious_output, dim=1),
                                           torch.mean(w_global_output, dim=1)).item()
        if distance < distance_old:
            self.reward += 0.1
        else:
            self.reward -= 0.1
        distance_old = distance
        return done, distance_old

    def step(self, args):
        '''
        恶意用户进行训练，得到下一步的环境
        :param action:
        :return:
        '''
        self.net.load_state_dict(self.w_global_old)  # 加载旧的global模型参数，在此基础上进行训练
        tmp_state = self.state.tolist()
        idxs_vicious = reinforce_control_iid_num(dataset=self.data_train, alpha=tmp_state[:args.kind], args=args)
        local_vicious = LocalUpdate(args=args, dataset=self.data_train, idxs=idxs_vicious)
        w_vicious, _ = local_vicious.train(net=copy.deepcopy(self.net).to(args.device))



    def reset(self):  # 如果done或者超过max_steps都需要reset，状态分布重置为初始状态，比如可以都为0.1
        pass






