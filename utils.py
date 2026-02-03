# import apex.amp as amp
import os.path
import random
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

svhn_mean = (0.5, 0.5, 0.5)
svhn_std = (0.5, 0.5, 0.5)

tiny_mean = (0.4802, 0.4481, 0.3975)
tiny_std = (0.2302, 0.2265, 0.2262)

def get_norm_stat(mean, std):
    mu = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)

    return mu, std, upper_limit, lower_limit

def get_norm_stat_dvs():
    mu = torch.tensor((0., 0.)).view(2, 1, 1)
    std = torch.tensor((1., 1.)).view(2, 1, 1)

    upper_limit = ((10 - mu) / std)
    lower_limit = ((-10 - mu) / std)

    return mu, std, upper_limit, lower_limit

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    if len(tensor.shape) == 5:
        mean = mean[None, None, :, None, None]
        std = std[None, None, :, None, None]
    else:
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


# evaluate on clean images with single norm
def evaluate_standard(test_loader, model, device):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    model.module.sum_output = True
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    model.module.sum_output = False
    return test_loss/n, test_acc/n

def evaluate_pgd(test_loader, model, eps, alpha, steps, lower_limit, upper_limit, device):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    # model.module.sum_output = True
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        adv = PGD_AT(model, X, y, eps, alpha, steps, lower_limit, upper_limit, device)
        with torch.no_grad():
            X_adv = X + adv
            output = model(X_adv).mean(1)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    # model.module.sum_output = False
    return test_loss/n, test_acc/n

def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T  # L_TET
    if lamb != 0:
        # outputs = rearrange(outputs, 't b c -> b t c')
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd  # L_Total

def orthogonal_retraction(model, beta=0.002):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    weight_ = module.weight.data
                    sz = weight_.shape
                    weight_ = weight_.reshape(sz[0],-1)
                    rows = list(range(module.weight.data.shape[0]))
                elif isinstance(module, nn.Linear):
                    if module.weight.data.shape[0] < 200: # set a sample threshold for row number
                        weight_ = module.weight.data
                        sz = weight_.shape
                        weight_ = weight_.reshape(sz[0], -1)
                        rows = list(range(module.weight.data.shape[0]))
                    else:
                        rand_rows = np.random.permutation(module.weight.data.shape[0])
                        rows = rand_rows[: int(module.weight.data.shape[0] * 0.3)]
                        weight_ = module.weight.data[rows,:]
                        sz = weight_.shape
                module.weight.data[rows,:] = ((1 + beta) * weight_ - beta * weight_.matmul(weight_.t()).matmul(weight_)).reshape(sz)

def imshow(img):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def PGD_AT(model, inputs, y, eps, alpha, steps, lower_limit, upper_limit, device):
    delta = torch.zeros_like(inputs).to(device)
    if len(delta.shape) == 5:
        for j in range(len(eps)):
            delta[:, :, j, :, :].uniform_((-eps[j][0][0]).item(), (eps[j][0][0]).item())
    elif len(delta.shape) == 4:
        for j in range(len(eps)):
            delta[:, j, :, :].uniform_((-eps[j][0][0]).item(), (eps[j][0][0]).item())
    delta.data = clamp(delta, lower_limit.to(device) - inputs, upper_limit.to(device) - inputs)
    delta.requires_grad = True

    for _ in range(steps):
        output = model(inputs + delta).mean(1)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()

        delta.data = clamp(delta + alpha * torch.sign(grad), -eps, eps)
        delta.data = clamp(delta, lower_limit.to(device) - inputs, upper_limit.to(device) - inputs)
        delta.grad.zero_()

    return delta.detach()

def PGD_TR(model, inputs, y, eps, alpha, steps, lower_limit, upper_limit, device):
    # alpha = eps / 4.
    adv = torch.FloatTensor(inputs.shape).to(device)
    if len(adv.shape) == 5:
        for j in range(len(eps)):
            adv[:, :, j, :, :].uniform_((-eps[j][0][0] / 2).item(), (eps[j][0][0] / 2).item())
    else:
        for j in range(len(eps)):
            adv[:, j, :, :].uniform_((-eps[j][0][0] / 2).item(), (eps[j][0][0] / 2).item())
    # adv.data = clamp(adv, lower_limit.to(device) - inputs, upper_limit.to(device) - inputs)
    X = inputs.detach()
    outputs1 = model(X).mean(1)
    for i in range(steps):
        adv.requires_grad = True
        outputs2 = model(X+adv).mean(1)
        p_s = F.log_softmax(outputs2 / 2, dim=1)
        p_t = F.softmax(outputs1 / 2, dim=1)
        kl = F.kl_div(p_s, p_t.detach(), size_average=False) * 4 / outputs1.shape[0]
        kl.backward()
        sign_grad = adv.grad.data.sign()
        with torch.no_grad():
            adv.data = adv.data + alpha * sign_grad
            adv.data = clamp(adv.data, -eps, eps)
            adv.data = clamp(adv, lower_limit.to(device) - inputs, upper_limit.to(device) - inputs)
    adv.grad = None
    adv.requires_grad = False
    # model.zero_grad()
    return adv.detach()

def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    return -(
        x[np.arange(x.shape[0]), y]
        - x_sorted[:, -2] * ind
        - x_sorted[:, -1] * (1.0 - ind)
    ) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
