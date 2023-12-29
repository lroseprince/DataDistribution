class Env():
    def __init__(self, action_dim=10):
        self.distribution = []
        self.data_type = 10
        self.data_size = 60000


    # 获取权重，分布，数据集的大小以及种类等状态信息，合并为列表并返回；同时需要判断任务是否完成，是则返回done
    def getState(self):
        done = False
        s_1 = torch.mean(w_global_output_old, dim=1).tolist()   # w_global_output_old为较旧的联邦学习环境输出层参数
        state = s_1 + self.distribution + [data_type, data_size]
        return state, done



    def setReward(self, state, done):           # 如果done，则reward+1；如果t时刻权重参数与恶意用户参数的KL散度小于t+1时刻，
        distance = torch.pairwise_distance(torch.mean(w_vicious_output, dim=1),
                                           torch.mean(w_global_output, dim=1)).item()
        if distance < distance_old:
            reward += 0.1
        else:
            reward -= 0.1

        # 则reward+0.1，反之-0.1
        if done:
            reward += 1
        return reward, done



    def step(self, action):                     # 执行动作环境发生的变化
        
        return np.asarray(state), reward, done


    def reset(self):                            # 如果done或者超过max_steps都需要reset，状态分布重置为初始状态，比如可以都为0.1

        return np.asarray(state)