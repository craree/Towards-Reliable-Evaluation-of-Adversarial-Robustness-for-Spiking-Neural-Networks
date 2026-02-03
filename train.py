import argparse
import copy
import logging
import os
import time

import torch

from evaluate import evaluate_attack
import torchattacks
from my_node import *
from vggsnn import VGG_SNN
from sew_resnet import BasicBlock, SEWResNetCifar, SEWResNet

from utils import *
from dataset import get_loaders
from braincog.model_zoo.base_module import *
from models import get_model
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_dir', default='/mnt/data/datasets', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--network', default='SEWResNet19C', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--device', default='cuda:3', type=str)
    parser.add_argument('--device_ids', default=[3], type=int, nargs='+',
                        help="List of GPU ids to use for training (e.g., --device_ids 3 4 5)")
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--lr_schedule', default='cosine', choices=['cyclic', 'multistep', 'cosine'])
    parser.add_argument('--lr_min', default=0., type=float)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--alpha', default=4, type=float, help='Step size')
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    parser.add_argument('--seed', default=3142, type=int, help='Random seed')

    parser.add_argument('--pgd_iters', default=1, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)

    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')

    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--reg', default=0.004, type=float)
    parser.add_argument('--lamb', default=0.001, type=float)
    parser.add_argument('--adv_training', action='store_true',
                        help='if adv training')
    parser.add_argument('--time_step', default=8, type=int)
    parser.add_argument('--Robust_Loss', default='AT', type=str)
    parser.add_argument('--encoder', default='dir', type=str)
    parser.add_argument('--means', default=1.0, type=float, metavar='N',
                        help='make all the potential increment around the means (default: 1.0)')
    parser.add_argument('--node_type', default='LIF', type=str)
    parser.add_argument('--RAT', action='store_true', help='if use different norm for different layers')

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(f'cuda:{args.device_ids[0]}')
    # torch.cuda.set_device(device)
    args.num_classes = 10
    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'tiny':
        args.num_classes = 200
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        mu, std, upper_limit, lower_limit = get_norm_stat(cifar10_mean, cifar10_std)
    elif args.dataset == 'svhn':
        mu, std, upper_limit, lower_limit = get_norm_stat(svhn_mean, svhn_std)
    elif args.dataset == 'tiny':
        mu, std, upper_limit, lower_limit = get_norm_stat(tiny_mean, tiny_std)
    elif args.dataset == 'dvsc':
        mu, std, upper_limit, lower_limit = get_norm_stat_dvs()
    path = os.path.join('./ckpt', args.dataset, args.network)
    args.save_dir = os.path.join(path, args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logfile = os.path.join(args.save_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get data loader
    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker)
    if args.dataset == 'dvsc':
        train_loader_e, test_loader_e, dataset_normalization_e = train_loader, test_loader, dataset_normalization
    else:
        train_loader_e, test_loader_e, dataset_normalization_e = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker, norm=False)

    # adv training attack setting
    alpha = ((args.alpha / 255.) / std).to(device)
    eps = ((args.eps / 255.) / std).to(device)

    node = MyLIFNode
    if args.node_type == 'PSN':
        node = PSNode
    elif args.node_type == 'IF':
        node = MyIFNode
    elif args.node_type == 'LIF2':
        node = MyLIFNode2
    model = get_model(args, device, node, normalization=None)
    model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    model.to(device)
    if args.node_type == 'PSN':
        model.module.set_backpropagation('bptt', device, alpha=4., SG='atan')

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)

    if args.dataset == 'dvsc':
        dataset_normalization_e = None
    else:
        dataset_normalization_e = dataset_normalization_e.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.lr_schedule == 'cyclic':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_pgd_acc = 0
    best_clean_acc = 0
    test_acc_best_pgd = 0
    start_epoch = 0

    # Start training
    start_train_time = time.time()
    for epoch in range(start_epoch, args.epochs+1):
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        train_loss = 0
        train_acc = 0
        train_n = 0
        model.train()

        for i, (X, y) in enumerate(train_loader):
            _iters = epoch * len(train_loader) + i
            X, y = X.to(device), y.to(device)

            if args.adv_training:
                # init delta
                delta = PGD_AT(model, X, y, eps, alpha, args.pgd_iters, lower_limit, upper_limit, device)
                X_adv = X + delta
            else:
                X_adv = X

            if args.Robust_Loss == 'TRADES':
                v = PGD_TR(model, X, y, eps, alpha, args.pgd_iters, lower_limit, upper_limit, device)
                x2 = X + v
                # outputs2 = F.softmax(model(x2) / 2, dim=2).mean(1)
                outputs2 = model(x2)

                outputs = model(X)
                mean_out = F.softmax(outputs, dim=2).mean(1)

                Loss_adv = TET_loss(outputs, y, criterion, args.means, args.lamb)

                p_s = F.log_softmax(outputs2.mean(1) / 2, dim=1)
                p_t = F.softmax(outputs.mean(1) / 2, dim=1)
                kl = F.kl_div(p_s, p_t, size_average=False) * 4 / outputs.shape[0]

                loss = Loss_adv + args.beta * kl

            elif args.Robust_Loss == 'SR':
                X_adv.requires_grad_(True)
                outputs = model(X_adv)

                # mean_out = F.softmax(outputs, dim=2).mean(1)
                mean_out = outputs.mean(1)

                loss1 = TET_loss(outputs, y, criterion, args.means, args.lamb)

                out = mean_out.gather(1, y.unsqueeze(1)).squeeze()  # choose
                batch = []
                inds = []
                for j in range(len(mean_out)):
                    mm, ind = torch.cat([mean_out[j, :y[j]], mean_out[j, y[j] + 1:]], dim=0).max(0)
                    f = torch.exp(out[j]) / (torch.exp(out[j]) + torch.exp(mm) + 1e-20)
                    batch.append(f)
                    inds.append(ind.item())
                f1 = torch.stack(batch, dim=0)
                dx = torch.autograd.grad(f1, X_adv, grad_outputs=torch.ones_like(f1, device=device), retain_graph=True)[
                    0]
                X_adv.requires_grad_(False)

                v = dx.detach().sign()

                x2 = X_adv + 0.01 * v

                outputs2 = model(x2).mean(1)

                out = outputs2.gather(1, y.unsqueeze(1)).squeeze()  # choose
                batch = []
                for j in range(len(outputs2)):
                    mm = torch.cat([outputs2[j, :y[j]], outputs2[j, y[j] + 1:]], dim=0)[inds[j]]
                    f = torch.exp(out[j]) / (torch.exp(out[j]) + torch.exp(mm) + 1e-20)
                    batch.append(f)
                f2 = torch.stack(batch, dim=0)

                dl = (f2 - f1) / 0.01

                loss2 = dl.pow(2).mean()

                loss = loss1 + args.beta * loss2
                loss = loss.mean()

            else:
                outputs = model(X_adv)
                mean_out = outputs.mean(1)
                loss = TET_loss(outputs, y, criterion, args.means, args.lamb)


            opt.zero_grad()
            loss.backward()
            opt.step()
            # if args.Robust_Loss == 'RAT':
            if args.RAT:
                orthogonal_retraction(model, args.reg)

            train_loss += loss.item() * y.size(0)
            train_acc += (mean_out.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            if i % 50 == 0:
                logger.info("Iter: [{:d}][{:d}/{:d}]\t"
                            "Loss {:.3f}\t"
                            "Prec@1 {:.3f}\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    train_loss / train_n,
                    train_acc / train_n)
                )
        scheduler.step()
        model.eval()
        logger.info('Evaluating with standard images...')
        test_loss, test_acc = evaluate_standard(test_loader, model, device)
        logger.info(
            'Test Loss: %.4f  \t Test Acc: %.4f',
            test_loss, test_acc)

        if args.pretrain == None:
            if test_acc > best_clean_acc:
                best_clean_acc = test_acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'weight_c.pth'))
                logger.info('Best Acc: %.4f ',
                             best_clean_acc)

            if epoch >= int(args.epochs / 5 * 3):
                logger.info('Evaluating with PGD Attack...')
                pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, eps, eps/2, 10,
                                                 lower_limit, upper_limit, device)
                if pgd_acc > best_pgd_acc:
                    best_pgd_acc = pgd_acc
                    test_acc_best_pgd = test_acc

                    # torch.save(best_state, os.path.join(args.save_dir, 'model.pth'))
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'weight_r.pth'))
                logger.info(
                        'PGD Loss: %.4f \t PGD Acc: %.4f \n Best PGD Acc: %.4f \t Test Acc of best PGD ckpt: %.4f',
                        pgd_loss, pgd_acc, best_pgd_acc, test_acc_best_pgd)
                opt.zero_grad()

    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)


    pretrained_model = torch.load(os.path.join(args.save_dir, 'weight_c.pth'), map_location=device, weights_only=True)
    state_dict = pretrained_model
    if not list(state_dict.keys())[0].startswith('module.'):
        state_dict = {'module.' + k: v for k, v in pretrained_model.items()}
    model.load_state_dict(state_dict, strict=False)
    # model.to(args.device)
    model.eval()
    model.module.normalize = dataset_normalization_e
    model.module.set_backpropagation('bptt', device, alpha=4., SG='atan')
    from attack import ASSG_SAPGD
    atk = ASSG_SAPGD(model, eps=args.eps / 255, steps=100, eot_iter=1, a=0.435)
    evaluate_attack(model, test_loader_e, device, atk, 'esaapgd', logger)


if __name__ == "__main__":
    main()
