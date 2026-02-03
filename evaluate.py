import argparse
import copy
import logging
import os
import sys
import time
from my_node import *
from models import get_model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, binomtest
from collections import OrderedDict
from attack import SAPGD, ASSG_SAPGD, ASSG_APGD, My_APGD, ASSG_SAPGD_poi, SAPGD_poi, My_APGD_poi, ASSG_APGD_poi, ASSG_MIFGSM, ASSG_PGD, ASSG_Adam
from vggsnn import VGG_SNN
from sew_resnet import BasicBlock, SEWResNetCifar, SEWResNet
import matplotlib.pyplot as plt

from utils import evaluate_standard
from dataset import get_loaders

import torchattacks

from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_dir', default='/mnt/data/datasets', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--network', default='SEWResNet19C', type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    # parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--device_ids', default=[3], type=int, nargs='+',
                        help="List of GPU ids to use for training (e.g., --device_ids 3 4 5)")
    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save log')
    parser.add_argument('--attack_type', default='apgd')
    parser.add_argument('--time_step', default=8, type=int)
    parser.add_argument('--node_type', default='LIF', type=str)
    parser.add_argument('--encoder', default='dir', type=str)
    parser.add_argument('--bmodel', default=None, type=str)
    parser.add_argument('--relax', default=1.5, type=float)
    parser.add_argument('--beta_1', default=0.9, type=float)
    parser.add_argument('--beta_2', default=0.9, type=float)
    return parser.parse_args()

def evaluate_attack(model, test_loader, device, atk, atk_name, logger):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    model.module.sum_output = True
    test_loader = iter(test_loader)
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
    for i in pbar:
        X, y = next(test_loader)
        X, y = X.to(device), y.to(device)
        X_adv = atk(X, y)  # advtorch
        model.module.sum_output = True
        assert torch.max(X_adv - X) > atk.eps
        with torch.no_grad():
            # model.module.t_softmax = True
            output = model(X_adv)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            # model.module.t_softmax = False

    pgd_acc = test_acc / n
    pgd_loss = test_loss / n
    logger.info(atk_name)
    logger.info('ASR: %.4f \t', 1 - pgd_acc)
    model.zero_grad()
    model.module.sum_output = False

    return pgd_loss, pgd_acc

def main():
    args = get_args()
    # device = args.device
    device = torch.device(f'cuda:{args.device_ids[0]}')

    # assert type(args.pretrain) == str and os.path.exists(args.pretrain)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'svhn':
        args.num_classes = 10
    elif args.dataset == 'dvsc':
        args.num_classes = 10

    args.save_dir = os.path.join('logs', args.dataset, args.network)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log_path = os.path.join(args.save_dir, args.pretrain.split('/')[0] + f"_{args.attack_type}" + ".txt")

    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)

    logger.info(args)

    logger.info('Dataset: %s', args.dataset)

    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker, norm=False)
    node = MyLIFNode
    if args.node_type == 'PSN':
        node = PSNode
    elif args.node_type == 'IF':
        node = MyIFNode
    elif args.node_type == 'LIF2':
        node = MyLIFNode2

    model = get_model(args, device, node, normalization=dataset_normalization)
    model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    model.to(device)

    # load pretrained model
    path = os.path.join('./ckpt', args.dataset, args.network)
    args.pretrain = os.path.join(path, args.pretrain)
    pretrained_model = torch.load(args.pretrain, map_location=device, weights_only=True)
    state_dict = pretrained_model
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    _, nature_acc = evaluate_standard(test_loader, model, device)
    logger.info('Nature Acc: %.4f \t', nature_acc)

    if args.attack_type[0:3] == 'pgd':
        steps = int(''.join(filter(str.isdigit, args.attack_type)))
        model.module.set_backpropagation('atan', device, alpha=10.)
        atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=steps, random_start=True)
        evaluate_attack(model, test_loader, device, atk, args.attack_type, logger)

    elif args.attack_type == 'apgd':

        model.module.set_backpropagation('bptt', device, alpha=12, SG='atan')
        atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
        evaluate_attack(model, test_loader, device, atk, 'apgd-atan', logger)

    elif args.attack_type == 'assg-apgd':
        model.module.set_backpropagation('bptt', device, alpha=16, SG='atan')
        for a in [0.425]:
            atk = ASSG_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a)
            evaluate_attack(model, test_loader, device, atk, f'assg-apgd-{a}', logger)

    elif args.attack_type == 'sapgd':
            model.module.set_backpropagation('bptt', device, alpha=16, SG='atan')
            atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce')
            evaluate_attack(model, test_loader, device, atk, 'sapgd-atan', logger)

    elif args.attack_type == 'sapgd_dvs': # For CIFAR10-DVS
            model.module.set_backpropagation('bptt', device, alpha=16, SG='atan')
            atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', range=(-10, 10))
            evaluate_attack(model, test_loader, device, atk, 'sapgd-atan', logger)

    elif args.attack_type == 'sapgd_poi': # For Poisson encoder
        a = 10
        model.module.set_backpropagation('bptt', device, alpha=a, SG='atan')
        atk = SAPGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False, loss='ce')
        evaluate_attack(model, test_loader, device, atk, f'sapgd-atan{a}', logger)

    elif args.attack_type == 'assg-sapgd':
        model.module.set_backpropagation('bptt', device, alpha=14, SG='atan')
        left, right = 0.35, 0.49
        best_a = None
        best_pgd_acc = float('inf')
        max_iter = 8  # ternary search
        for it in range(max_iter):
            a1 = left + (right - left) / 3.0
            a2 = right - (right - left) / 3.0

            atk1 = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a1, relax=args.relax, beta_1=args.beta_1, beta_2=args.beta_2)
            _, pgd_acc1 = evaluate_attack(
                model, test_loader, device, atk1, f'R{args.relax}-{args.beta_1}-{args.beta_2}-{a1:.6f}', logger
            )

            atk2 = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a2, relax=args.relax, beta_1=args.beta_1, beta_2=args.beta_2)
            _, pgd_acc2 = evaluate_attack(
                model, test_loader, device, atk2, f'R{args.relax}-{args.beta_1}-{args.beta_2}-{a2:.6f}', logger
            )

            if pgd_acc1 < best_pgd_acc:
                best_pgd_acc = pgd_acc1
                best_a = a1
            if pgd_acc2 < best_pgd_acc:
                best_pgd_acc = pgd_acc2
                best_a = a2

            if pgd_acc1 < pgd_acc2:
                right = a2
            else:
                left = a1

        logger.info(f'R{args.relax}-{args.beta_1}-{args.beta_2}: best_a={best_a:.6f}, best_asr={1-best_pgd_acc:.4f}')

    elif args.attack_type == 'assg-sapgd_dvs':  # For CIFAR10-DVS
        model.module.set_backpropagation('bptt', device, alpha=14, SG='atan')
        for a in [0.43]:
            atk = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a, range=(-10, 10))
            evaluate_attack(model, test_loader, device, atk, f'migcSTadam-{a}', logger)

    elif args.attack_type == 'assg-sapgd_poi': # For Poisson encoder
        model.module.set_backpropagation('bptt', device, alpha=14, SG='atan')
        for a in [0.455, 0.46, 0.465]:
        # for a in [0.375]:
            atk = ASSG_SAPGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False, loss='ce', a=a, relax=args.relax, beta_1=args.beta_1, beta_2=args.beta_2)
            evaluate_attack(model, test_loader, device, atk, f'1-0.95-x+0.001b-mean-esapgd-{a}', logger)

    elif args.attack_type == 'abla_all': # Exp for BPTR,RGA,PDSG,STBP,optimal alpha Atan and ASSG

        model.module.set_backpropagation('bptt', device, alpha=2, SG='qgate')
        atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
        _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-qgate2', logger)

        atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
        _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-qgate2', logger)

        for alpha in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
            model.module.set_backpropagation('bptt', device, alpha=alpha, SG='atan')
            atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-atan{alpha}', logger)

            atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-atan{alpha}', logger)

        modes = ['rga', 'bptr', 'hart']
        for mode in modes:
            model.module.set_backpropagation(mode, device, alpha=2, SG='qgate')
            atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-{mode}', logger)

            model.module.set_backpropagation(mode, device, alpha=2, SG='qgate')
            atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-{mode}', logger)

        model.module.set_backpropagation('bptt', device, alpha=4, SG='pdsg')
        atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
        evaluate_attack(model, test_loader, device, atk, 'apgd-pdsg', logger)

        atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce')
        evaluate_attack(model, test_loader, device, atk, 'sa-pgd-pdsg', logger)

        for a in [0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445, 0.45]:
            model.module.set_backpropagation('bptt', device, alpha=4, SG='atan')
            atk = ASSG_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a)
            evaluate_attack(model, test_loader, device, atk, f'assg-apgd-{a}', logger)

            atk = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a)
            evaluate_attack(model, test_loader, device, atk, f'assg-sa-pgd-{a}', logger)

    elif args.attack_type == 'abla_light':  # Exp for BPTR,RGA,PDSG,STBP and ASSG

        model.module.set_backpropagation('bptt', device, alpha=2, SG='qgate')
        if args.node_type == 'PSN':
            model.module.set_backpropagation('bptt', device, alpha=4, SG='atan')
        atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
        _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-qgate2', logger)

        atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
        _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-qgate2', logger)

        modes = ['rga', 'bptr', 'hart']
        for mode in modes:
            model.module.set_backpropagation(mode, device, alpha=2, SG='qgate')
            if args.node_type == 'PSN':
                model.module.set_backpropagation(mode, device, alpha=4, SG='atan')
            atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-{mode}', logger)

            model.module.set_backpropagation(mode, device, alpha=2, SG='qgate')
            if args.node_type == 'PSN':
                model.module.set_backpropagation(mode, device, alpha=4, SG='atan')
            atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-{mode}', logger)

        model.module.set_backpropagation('bptt', device, alpha=4, SG='pdsg')
        atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
        evaluate_attack(model, test_loader, device, atk, 'apgd-pdsg', logger)

        atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce')
        evaluate_attack(model, test_loader, device, atk, 'sa-pgd-pdsg', logger)

        for a in [0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445, 0.45]:
            model.module.set_backpropagation('bptt', device, alpha=4, SG='atan')
            atk = ASSG_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a)
            evaluate_attack(model, test_loader, device, atk, f'assg-apgd-{a}', logger)

            atk = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a)
            evaluate_attack(model, test_loader, device, atk, f'assg-sa-pgd-{a}', logger)


    elif args.attack_type == 'abla_dvs':  # For CIFAR10-DVS

        for alpha in [2]:
            model.module.set_backpropagation('bptt', device, alpha=alpha, SG='qgate')
            atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, range=(-10., 10.))
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-qgate{alpha}', logger)

            atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, range=(-10., 10.))
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-qgate{alpha}', logger)

        modes = ['rga', 'bptr', 'hart']
        for mode in modes:
            model.module.set_backpropagation(mode, device, alpha=2, SG='qgate')
            atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, range=(-10., 10.))
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-{mode}', logger)

            model.module.set_backpropagation(mode, device, alpha=2, SG='qgate')
            atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, range=(-10., 10.))
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-{mode}', logger)

        model.module.set_backpropagation('bptt', device, alpha=2, SG='pdsg')
        atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, range=(-10., 10.))
        evaluate_attack(model, test_loader, device, atk, 'apgd-pdsg', logger)

        atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', range=(-10., 10.))
        evaluate_attack(model, test_loader, device, atk, 'sa-pgd-pdsg', logger)

        for a in [0.425, 0.43, 0.435, 0.44, 0.445, 0.45, 0.455]:
            model.module.set_backpropagation('bptt', device, alpha=4, SG='atan')
            atk = ASSG_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a, range=(-10., 10.))
            evaluate_attack(model, test_loader, device, atk, f'assg-apgd-{a}', logger)

            atk = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a, range=(-10., 10.))
            evaluate_attack(model, test_loader, device, atk, f'assg-sa-pgd-{a}', logger)

    elif args.attack_type == 'abla_poi': # For Poisson encoder

        for alpha in [2]:
            model.module.set_backpropagation('bptt', device, alpha=alpha, SG='qgate')
            atk = My_APGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-atan{alpha}', logger)

            atk = SAPGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-atan{alpha}', logger)

        modes = ['rga', 'bptr', 'hart']
        for mode in modes:
            model.module.set_backpropagation(mode, device, alpha=2, SG='qgate')
            atk = My_APGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-{mode}', logger)

            model.module.set_backpropagation(mode, device, alpha=2, SG='qgate')
            atk = SAPGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-{mode}', logger)

        model.module.set_backpropagation('bptt', device, alpha=2, SG='pdsg')
        atk = My_APGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False)
        evaluate_attack(model, test_loader, device, atk, 'apgd-pdsg', logger)

        atk = SAPGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False, loss='ce')
        evaluate_attack(model, test_loader, device, atk, 'sa-pgd-pdsg', logger)

        for a in [0.47, 0.475, 0.48, 0.485]:
            model.module.set_backpropagation('bptt', device, alpha=2, SG='atan')
            atk = ASSG_APGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False, loss='ce', a=a)
            evaluate_attack(model, test_loader, device, atk, f'assg-apgd-{a}', logger)

            atk = ASSG_SAPGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False, loss='ce', a=a)
            evaluate_attack(model, test_loader, device, atk, f'assg-sa-pgd-{a}', logger)

    elif args.attack_type == 'hart': # Search for optimal alpha for HART

        for alpha in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
            model.module.set_backpropagation('hart', device, alpha=alpha, SG='atan')
            atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-hart{alpha}', logger)

            atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-hart{alpha}', logger)

    elif args.attack_type == 'hart_dvs': # Search for optimal alpha for HART for CIFAR10-DVS

        for alpha in [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]:
            model.module.set_backpropagation('hart', device, alpha=alpha, SG='atan')
            atk = My_APGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, range=(-10., 10.))
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-hart{alpha}', logger)

            atk = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, range=(-10., 10.))
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-hart{alpha}', logger)

    elif args.attack_type == 'hart_poi': # Search for optimal alpha for HART for Poisson encoder

        for alpha in [4, 6, 8, 10, 12, 14]:
            model.module.set_backpropagation('hart', device, alpha=alpha, SG='atan')
            atk = My_APGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'apgd-hart{alpha}', logger)

            atk = SAPGD_poi(model, eps=args.epsilon / 255, steps=50, eot_iter=10, verbose=False)
            _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'sa-pgd-hart{alpha}', logger)

    elif args.attack_type == 'abla_opt':  # ablation study for attack methods

        for a in [0.435]:
            result = []
            for step in [100, 400, 700, 1000]:
                model.module.set_backpropagation('bptt', device, alpha=4, SG='atan')

                atk = ASSG_PGD(model, eps=args.epsilon / 255, alpha=args.epsilon / 255 / 16, steps=step, a=a)
                _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'ASSG_PGD-{a}-{step}', logger)
                result.append(1-pgd_acc)

                atk = ASSG_MIFGSM(model, eps=args.epsilon / 255, alpha=args.epsilon / 255 / 16, steps=step, a=a, decay=0.6)
                _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'ASSG_MIFGSM-{a}-{step}', logger)
                result.append(1 - pgd_acc)

                atk = ASSG_APGD(model, eps=args.epsilon / 255, steps=step, eot_iter=1, verbose=False, loss='ce', a=a)
                _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'ASSG_APGD-{a}-{step}', logger)
                result.append(1 - pgd_acc)

                atk = ASSG_Adam(model, eps=args.epsilon / 255, steps=step, eot_iter=1, verbose=False, loss='ce', a=a)
                _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'ASSG_Adam-{a}-{step}', logger)
                result.append(1 - pgd_acc)

                atk = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=step, eot_iter=1, verbose=False, loss='ce', a=a)
                _, pgd_acc = evaluate_attack(model, test_loader, device, atk, f'ASSG_SAPGD-{a}-{step}', logger)
                result.append(1 - pgd_acc)

                logger.info(f'ASR_{a}: %.4f \t', result)

    elif args.attack_type == 'loss': # plot loss curves
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 16
        mpl.rcParams['axes.labelsize'] = 18
        mpl.rcParams['legend.fontsize'] = 16


        style_map = {
            'STBP': dict(color='#0072B2', linestyle='-'),
            'BPTR': dict(color='#E69F00', linestyle='--'),
            'RGA': dict(color='#009E73', linestyle='-.'),
            'PDSG': dict(color='#D55E00', linestyle=':'),
            'HART': dict(color='#CC79A7', linestyle='-'),
            'ASSG': dict(color='#56B4E9', linestyle='--'),
        }

        _seen_methods = set()

        def plot_loss(step, loss, method: str):
            style = style_map.get(method, {})

            legend_label = method if method not in _seen_methods else "_nolegend_"
            x = np.arange(len(loss))
            plt.plot(x, loss, linewidth=2, label=legend_label, **style)
            _seen_methods.add(method)

        for a in [0.435]:
            result = []
            for step in [100]:
                model.module.set_backpropagation('bptt', device, alpha=2, SG='qgate')
                atk = SAPGD(model, eps=args.epsilon / 255, steps=step, eot_iter=1, verbose=True, visual=True, loss='ce', SG='STBP')
                model.eval()
                model.module.sum_output = True
                test_loader = iter(test_loader)
                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
                for i in pbar:
                    X, y = next(test_loader)
                    X, y = X.to(device), y.to(device)
                    X_adv = atk(X, y)  # advtorch
                    break
                loss = np.load(f'loss_record_STBP_SAPGD{step}.npy')
                plot_loss(step, loss, method='STBP')

                model.module.set_backpropagation('bptr', device, alpha=2, SG='qgate')
                atk = SAPGD(model, eps=args.epsilon / 255, steps=step, eot_iter=1, verbose=True, visual=True, loss='ce', SG='BPTR')
                model.eval()
                model.module.sum_output = True
                test_loader = iter(test_loader)
                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
                for i in pbar:
                    X, y = next(test_loader)
                    X, y = X.to(device), y.to(device)
                    X_adv = atk(X, y)  # advtorch
                    break
                loss = np.load(f'loss_record_BPTR_SAPGD{step}.npy')
                plot_loss(step, loss, method='BPTR')

                model.module.set_backpropagation('rga', device, alpha=2, SG='qgate')
                atk = SAPGD(model, eps=args.epsilon / 255, steps=step, eot_iter=1, verbose=True, visual=True, loss='ce', SG='RGA')
                model.eval()
                model.module.sum_output = True
                test_loader = iter(test_loader)
                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
                for i in pbar:
                    X, y = next(test_loader)
                    X, y = X.to(device), y.to(device)
                    X_adv = atk(X, y)  # advtorch
                    break
                loss = np.load(f'loss_record_RGA_SAPGD{step}.npy')
                plot_loss(step, loss, method='RGA')

                model.module.set_backpropagation('bptt', device, alpha=2, SG='pdsg')
                atk = SAPGD(model, eps=args.epsilon / 255, steps=step, eot_iter=1, verbose=True, visual=True, loss='ce', SG='PDSG')
                model.eval()
                model.module.sum_output = True
                test_loader = iter(test_loader)
                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
                for i in pbar:
                    X, y = next(test_loader)
                    X, y = X.to(device), y.to(device)
                    X_adv = atk(X, y)  # advtorch
                    break
                loss = np.load(f'loss_record_PDSG_SAPGD{step}.npy')
                plot_loss(step, loss, method='PDSG')

                model.module.set_backpropagation('hart', device, alpha=14, SG='atan')
                atk = SAPGD(model, eps=args.epsilon / 255, steps=step, eot_iter=1, verbose=True, visual=True, loss='ce', SG='HART')
                model.eval()
                model.module.sum_output = True
                test_loader = iter(test_loader)
                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
                for i in pbar:
                    X, y = next(test_loader)
                    X, y = X.to(device), y.to(device)
                    X_adv = atk(X, y)  # advtorch
                    break
                loss = np.load(f'loss_record_HART_SAPGD{step}.npy')
                plot_loss(step, loss, method='HART')

                atk = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=step, eot_iter=1, verbose=True, visual=True, loss='ce', a=a)
                test_loader = iter(test_loader)
                bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
                pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
                for i in pbar:
                    X, y = next(test_loader)
                    X, y = X.to(device), y.to(device)
                    X_adv = atk(X, y)  # advtorch
                    break
                loss = np.load(f'loss_record_ASSG_SAPGD{step}.npy')
                plot_loss(step, loss, method='ASSG')

        plt.xlabel('Iteration Numbers')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(ncol=3)
        plt.tight_layout()
        plt.show()

    elif args.attack_type == 'alpha': # plot alpha distribution
        atk = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=True, show_alpha=True, loss='ce', a=0.435)
        model.eval()
        model.module.sum_output = True
        test_loader = iter(test_loader)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
        for i in pbar:
            X, y = next(test_loader)
            X, y = X.to(device), y.to(device)
            X_adv = atk(X, y)  # advtorch
            break

    elif args.attack_type == 'assg-sg':
        model.module.set_backpropagation('bptt', device, alpha=14, SG='atan')
        left, right = 1, 6
        best_a = None
        best_pgd_acc = float('inf')
        max_iter = 12  # ternary search
        for it in range(max_iter):
            a1 = left + (right - left) / 3.0
            a2 = right - (right - left) / 3.0

            atk1 = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a1,
                              relax=args.relax, beta_1=args.beta_1, beta_2=args.beta_2, SG='assg-g')
            _, pgd_acc1 = evaluate_attack(
                model, test_loader, device, atk1, f'ASSG-G-{a1:.6f}', logger
            )

            atk2 = ASSG_SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce', a=a2,
                              relax=args.relax, beta_1=args.beta_1, beta_2=args.beta_2, SG='assg-g')
            _, pgd_acc2 = evaluate_attack(
                model, test_loader, device, atk2, f'ASSG-G-{a2:.6f}', logger
            )

            if pgd_acc1 < best_pgd_acc:
                best_pgd_acc = pgd_acc1
                best_a = a1
            if pgd_acc2 < best_pgd_acc:
                best_pgd_acc = pgd_acc2
                best_a = a2

            if pgd_acc1 < pgd_acc2:
                right = a2
            else:
                left = a1

        logger.info(f'ASSG-G: best_a={best_a:.6f}, best_asr={1 - best_pgd_acc:.4f}')

    elif args.attack_type == 'sg':
        model.module.set_backpropagation('bptt', device, alpha=14, SG='atan')
        left, right = 2, 20
        best_a = None
        best_pgd_acc = float('inf')
        max_iter = 12  # ternary search
        for it in range(max_iter):
            a1 = left + (right - left) / 3.0
            a2 = right - (right - left) / 3.0
            model.module.set_backpropagation('bptt', device, alpha=a1, SG='gaussian')
            atk1 = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce')
            _, pgd_acc1 = evaluate_attack(
                model, test_loader, device, atk1, f'G-{a1:.6f}', logger
            )
            model.module.set_backpropagation('bptt', device, alpha=a2, SG='gaussian')
            atk2 = SAPGD(model, eps=args.epsilon / 255, steps=100, eot_iter=1, verbose=False, loss='ce')
            _, pgd_acc2 = evaluate_attack(
                model, test_loader, device, atk2, f'G-{a2:.6f}', logger
            )

            if pgd_acc1 < best_pgd_acc:
                best_pgd_acc = pgd_acc1
                best_a = a1
            if pgd_acc2 < best_pgd_acc:
                best_pgd_acc = pgd_acc2
                best_a = a2

            if pgd_acc1 < pgd_acc2:
                right = a2
            else:
                left = a1

        logger.info(f'Gaussian: best_a={best_a:.6f}, best_asr={1 - best_pgd_acc:.4f}')


    logger.info('Testing done.')


if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    main()
