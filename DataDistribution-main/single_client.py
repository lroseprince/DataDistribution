"""
@Time: 2023/12/30 19:40
@Author: lroseprince
@File:single_client.py
@Description: single训练得到参数
"""
import copy
import numpy as np
import torch
from torchvision import datasets, transforms
import random

from options import args_parser
from utils import vicious_load_dataset, control_iid_num, build_model
from Update import LocalUpdate
from test import test_img


if __name__ == '__main__':

    args = args_parser()

    dataset_train, dataset_test = vicious_load_dataset(args)
    data_index_single = control_iid_num(dataset_train, args)
    net_single = build_model(args)

    for i in range(args.epochs):
        local = LocalUpdate(args, dataset_train, data_index_single)
        w, loss = local.train(net=copy.deepcopy(net_single))
        torch.save(w, "./weights/epoch{}client_globalClient_mnist_single_kind3_num012_v1.pth".format(i))
        net_single.load_state_dict(copy.deepcopy(w))
        print(loss)

    net_single.eval()
    acc_train, loss_train = test_img(net_single, dataset_train, args)
    acc_test, loss_test = test_img(net_single, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))


