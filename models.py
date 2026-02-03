import torch

from sew_resnet import BasicBlock, SEWResNetCifar, SEWResNet, Bottleneck
from vggsnn import VGG_SNN, VGGCIFAR
from my_node import *

def get_model(args, device, node, normalization=None):
    if args.network == 'SEWResNet19C':

        model = SEWResNetCifar(BasicBlock, [3,3,3], cnf='ADD', node_type=node, step=args.time_step,
                               num_classes=args.num_classes, layer_by_layer=True, act_fun=QGateGrad,
                               sum_output=False, data_norm=normalization, encoder=args.encoder,
                               )

    elif args.network == 'SEWResNet31C':

        model = SEWResNetCifar(BasicBlock, [5, 5, 5], cnf='ADD', node_type=node, step=args.time_step,
                               num_classes=args.num_classes, layer_by_layer=True, act_fun=QGateGrad,
                               sum_output=False, data_norm=normalization, encoder=args.encoder,
                               )

    elif args.network == 'SEWResNet43C':

        model = SEWResNetCifar(BasicBlock, [7, 7, 7], cnf='ADD', node_type=node, step=args.time_step,
                               num_classes=args.num_classes, layer_by_layer=True, act_fun=QGateGrad,
                               sum_output=False, data_norm=normalization, encoder=args.encoder,
                               )

    elif args.network == 'SEWResNet20':

            model = SEWResNet(BasicBlock, [3, 2, 2, 2], cnf='ADD', node_type=node, step=args.time_step,
                              num_classes=args.num_classes, layer_by_layer=True, act_fun=QGateGrad,
                              sum_output=False, data_norm=normalization, encoder=args.encoder,
                              )

    elif args.network == 'SEWResNet26':

        model = SEWResNet(BasicBlock, [3, 3, 3, 3], cnf='ADD', node_type=node, step=args.time_step,
                          num_classes=args.num_classes, layer_by_layer=True, act_fun=QGateGrad,
                          sum_output=False, data_norm=normalization, encoder=args.encoder,
                          )

    elif args.network == 'SEWResNet34':

        model = SEWResNet(BasicBlock, [3, 4, 6, 3], cnf='ADD', node_type=node, step=args.time_step,
                          num_classes=args.num_classes, layer_by_layer=True, act_fun=QGateGrad,
                          sum_output=False, data_norm=normalization, encoder=args.encoder,
                          )
    elif args.network == 'VGG':

            model = VGG_SNN(node_type=node, step=args.time_step,
                            num_classes=args.num_classes, layer_by_layer=True, act_fun=QGateGrad,
                            sum_output=False, data_norm=normalization)

    elif args.network == 'VGG_C':

        model = VGGCIFAR(node_type=node, step=args.time_step,
                        num_classes=args.num_classes, layer_by_layer=True, act_fun=QGateGrad,
                        sum_output=False, data_norm=normalization)

    return model