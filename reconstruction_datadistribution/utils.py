'''
@Time: 2023/12/29 16:25
@Author: lroseprince
@File:utils.py
@Description: 用来构建环境的工具类
'''
from torchvision import datasets, transforms
import torch
import random

from Nets import CNNMnist, CNNCifar


def reinforce_control_iid_num(dataset, alpha, args):
    '''
    根据强化学习的state中的数据分布进行数据分配,为了方便起见，类只能搞连续的且在前面的类，任意挑选种类的功能后面实现
    :param dataset_train:
    :param alpha: alpha为state中的distribution，为list形式
    :param args:
    :return:
    '''
    idxs_dict = {}
    indexOfClients = []  # 用来存储最终选择的数据索引
    # 将数据集分好类
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    idxs_dict = {k: idxs_dict[k] for k in sorted(idxs_dict.keys())}  # 用来记录索引

    # 将数值转换为相对值
    numOfKind = []
    total = sum(alpha)
    for i in range(args.kind):
        numOfKind.append(int(alpha[i] / total * 1200))
    # 根据alpha矩阵来调整数据分布
    for i in range(args.kind):
        idxs_dict[i] = idxs_dict[i][:numOfKind[i]]
    # 某些种类可以不进行下发
    idxs_dict_new = dict_slice(idxs_dict, 0, args.kind)
    # 将字典中的value代表的索引放在一个列表中，方便进行选择
    for key in idxs_dict_new.keys():
        indexOfClients.extend(idxs_dict_new[key])
    random.shuffle(indexOfClients)
    return indexOfClients


def vicious_load_dataset(args):
    '''
    为单体用户和联邦学习按照自定义的方式加载数据
    :param args:
    :return:
    '''
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

    elif args.dataset == 'fashion-mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist/', train=False, download=True, transform=trans_mnist)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test

def build_model(args):
    '''
    创建与联邦学习或者单体用户相同的模型供恶意用户学习
    :param args:
    :return:
    '''
    if args.dataset == 'mnist':
        net = CNNMnist(args=args)
    elif args.dataset == 'fashion-mnist':
        net = CNNMnist(args=args)
    elif args.dataset == 'cifar':
        net = CNNCifar(args=args)
    return net


def dict_slice(adict, start, end):
    '''
    字典切割，只适用于连续
    :param adict:
    :param start:
    :param end:
    :return:
    '''
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start: end]:
        dict_slice[k] = adict[k]
    return dict_slice
