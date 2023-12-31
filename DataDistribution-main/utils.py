"""
@Time: 2023/12/29 16:25
@Author: lroseprince
@File:utils.py
@Description: 用来构建环境的工具类
"""
from torchvision import datasets, transforms
import torch
import random
import numpy as np

from Nets import CNNMnist, CNNCifar


def reinforce_control_iid_num(dataset, alpha, args):
    """
    根据强化学习的state中的数据分布进行数据分配,为了方便起见，类只能搞连续的且在前面的类，任意挑选种类的功能后面实现
    :param dataset:
    :param alpha: alpha为state中的distribution，为list形式
    :param args:
    :return:
    """
    idxs_dict = {}
    index_clients = []  # 用来存储最终选择的数据索引
    # 将数据集分好类
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    idxs_dict = {k: idxs_dict[k] for k in sorted(idxs_dict.keys())}  # 用来记录索引

    # 将数值转换为相对值
    num_kind = []
    total = sum(alpha)
    for i in range(args.kind):
        num_kind.append(int(alpha[i] / total * 6000))
    # 根据alpha矩阵来调整数据分布
    for i in range(args.kind):
        idxs_dict[i] = idxs_dict[i][:num_kind[i]]
    # 某些种类可以不进行下发
    idxs_dict_new = dict_slice(idxs_dict, 0, args.kind)
    # 将字典中的value代表的索引放在一个列表中，方便进行选择
    for key in idxs_dict_new.keys():
        index_clients.extend(idxs_dict_new[key])
    random.shuffle(index_clients)
    return index_clients


def control_iid_num(dataset, args):
    """
    用于单体用户或者联邦学习用户进行训练时候得到数据集的索引
    :param dataset:
    :param args:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
    idxs_dict = {}
    indexOfClients = []
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    idxs_dict = {k: idxs_dict[k] for k in sorted(idxs_dict.keys())}  # 用来记录索引
    for i in range(0, 1):
        idxs_dict[i] = idxs_dict[i][:2000]
    for i in range(1, 2):
        idxs_dict[i] = idxs_dict[i][:2000]
    for i in range(2, 3):
        idxs_dict[i] = idxs_dict[i][:2000]
    # for i in range(3, 4):
    #     idxs_dict[i] = idxs_dict[i][:2000]
    # for i in range(4, 5):
    #     idxs_dict[i] = idxs_dict[i][:6000]
    # for i in range(5, 6):
    #     idxs_dict[i] = idxs_dict[i][:6000]
    # for i in range(6, 7):
    #     idxs_dict[i] = idxs_dict[i][:2000]
    # for i in range(7, 8):
    #     idxs_dict[i] = idxs_dict[i][:2000]
    # for i in range(8, 9):
    #     idxs_dict[i] = idxs_dict[i][:2000]
    # for i in range(9, 10):
    #     idxs_dict[i] = idxs_dict[i][:6000]

    # 某些种类可以不进行下发
    idxs_dict_new = dict_slice(idxs_dict, 0, 3)
    # 将字典中的value代表的索引放在一个列表中，方便进行选择
    for key in idxs_dict_new.keys():
        indexOfClients.extend(idxs_dict_new[key])
    random.shuffle(indexOfClients)
    if args.is_federated:
        num_shard_client = int(len(indexOfClients)/args.num_users)
        for i in range(args.num_users):
            rand_set = set(random.sample(indexOfClients, num_shard_client))
            indexOfClients = list(set(indexOfClients) - rand_set)
            dict_users[i] = np.concatenate((dict_users[i], np.array(list(rand_set))), axis=0)
        return dict_users
    return indexOfClients


def vicious_load_dataset(args):
    """
    为单体用户和联邦学习、恶意用户按照自定义的方式加载数据
    :param args:
    :return:
    """
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

    elif args.dataset == 'fashion-mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist/', train=False, download=True, transform=trans_mnist)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test


def build_model(args):
    """
    创建与联邦学习或者单体用户相同的模型供恶意用户学习
    :param args:
    :return:
    """
    if args.dataset == 'mnist':
        net = CNNMnist(args=args)
    elif args.dataset == 'fashion-mnist':
        net = CNNMnist(args=args)
    elif args.dataset == 'cifar':
        net = CNNCifar(args=args)
    return net.to(args.device)


def dict_slice(adict, start, end):
    """
    字典切割，只适用于连续
    :param adict:
    :param start:
    :param end:
    :return:
    """
    keys = adict.keys()
    dict_slicing = {}
    for k in list(keys)[start: end]:
        dict_slicing[k] = adict[k]
    return dict_slicing


def judge_index():
    """
    先通过模型参数的变化进行识别输出层参数的变化，联邦学习中通过哪几个种类的数据进行训练
    现在可以直接指定，先直接假定，后续再补充这部分代码  目前还不是很确定用所有的还是只用那几个
    :return: 返回索引
    """
    index = [0, 1, 2]
    return index


def get_output_params(weights, args):
    """
    输入模型参数，将输出层的参数分离出来
    :param args:
    :param weights:
    :return:
    """
    if args.dataset == "mnist" or "fashion-mnist":
        output_params = weights['fc2.weight']
    elif args.dataset == "cifar":
        output_params = weights['fc3.weight']
    return output_params


def init_distribution(args):
    """
    初始化的时候，或者done为true的时候将distribution设定为初始状态分布
    :return:
    """
    return [5 for _ in range(args.kind)]
