from utils import *
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision
import math
from tonic import DiskCachedDataset
import tonic
from einops import rearrange, repeat
from scipy.stats import norm, binomtest
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import sys

from torch import Tensor
from typing import Tuple

def get_loaders(dir_, batch_size, dataset='cifar10', worker=4, norm=True, subset=None):
    if dataset == 'cifar10' or dataset == 'cifar100':
        if norm:

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
            dataset_normalization = None

        else:

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset_normalization = NormalizeByChannelMeanStd(
                mean=cifar10_mean, std=cifar10_std)

        if dataset == 'cifar10':
            if subset is not None:
                train_dataset = datasets.CIFAR10(
                    dir_, train=True, transform=train_transform, download=False)
                test_dataset = Subset(datasets.CIFAR10(
                    dir_, train=False, transform=test_transform, download=False), subset)
            else:
                train_dataset = datasets.CIFAR10(
                    dir_, train=True, transform=train_transform, download=False)
                test_dataset = datasets.CIFAR10(
                    dir_, train=False, transform=test_transform, download=False)
        elif dataset == 'cifar100':
            if subset is not None:
                train_dataset = datasets.CIFAR100(
                    dir_, train=True, transform=train_transform, download=False)
                test_dataset = Subset(datasets.CIFAR100(
                    dir_, train=False, transform=test_transform, download=False), subset)
            else:
                train_dataset = datasets.CIFAR100(
                    dir_, train=True, transform=train_transform, download=False)
                test_dataset = datasets.CIFAR100(
                    dir_, train=False, transform=test_transform, download=False)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=worker,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=worker,
        )
        return train_loader, test_loader, dataset_normalization

    elif dataset == 'svhn':
        if norm:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(svhn_mean, svhn_std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(svhn_mean, svhn_std),
            ])
            dataset_normalization = None

        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset_normalization = NormalizeByChannelMeanStd(
                mean=svhn_mean, std=svhn_std)

        train_dataset = datasets.SVHN(
            os.path.join(dir_, 'SVHN'), split='train', transform=train_transform, download=True)
        test_dataset = datasets.SVHN(
            os.path.join(dir_, 'SVHN'), split='test', transform=test_transform, download=True)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=worker,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=worker,
        )
        return train_loader, test_loader, dataset_normalization

    elif dataset == 'dvsc':

        dataset_normalization = None
        train_loader, test_loader, _, _ = get_dvsc10_data(DATA_DIR=dir_, batch_size=batch_size, step=8)
        return train_loader, test_loader, dataset_normalization

    else:
        return None

def unpack_mix_param(args):
    mix_up = args['mix_up'] if 'mix_up' in args else False
    cut_mix = args['cut_mix'] if 'cut_mix' in args else False
    event_mix = args['event_mix'] if 'event_mix' in args else False
    beta = args['beta'] if 'beta' in args else 1.
    prob = args['prob'] if 'prob' in args else .5
    num = args['num'] if 'num' in args else 1
    num_classes = args['num_classes'] if 'num_classes' in args else 10
    noise = args['noise'] if 'noise' in args else 0.
    gaussian_n = args['gaussian_n'] if 'gaussian_n' in args else None
    return mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n

def get_dvsc10_data(DATA_DIR, batch_size, step, **kwargs):
    """
    获取DVS CIFAR10数据
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    size = kwargs['size'] if 'size' in kwargs else 48
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    train_dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=train_transform)
    test_dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=test_transform)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        transforms.RandomCrop(size, padding=size // 12),
        transforms.RandomHorizontalFlip(),
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(DATA_DIR, 'DVS/DVS_Cifar10/train_cache_{}'.format(step)),
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(DATA_DIR, 'DVS/DVS_Cifar10/test_cache_{}'.format(step)),
                                     transform=test_transform)

    num_train = len(train_dataset)
    num_per_cls = num_train // 10
    indices_train, indices_test = [], []
    portion = kwargs['portion'] if 'portion' in kwargs else .9
    for i in range(10):
        indices_train.extend(
            list(range(i * num_per_cls, round(i * num_per_cls + num_per_cls * portion))))
        indices_test.extend(
            list(range(round(i * num_per_cls + num_per_cls * portion), (i + 1) * num_per_cls)))

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train),
        pin_memory=True, drop_last=True, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
        pin_memory=True, drop_last=False, num_workers=2
    )

    return train_loader, test_loader, mixup_active, None