from functools import partial

import torch
from torch.nn import functional as F
import torchvision
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.datasets import is_dvs_data
import torch.nn.init as init
from my_node import *

class VGG_SNN(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 data_norm=None,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes
        self.normalize = data_norm
        self.sum_output = kwargs["sum_output"] if "sum_output" in kwargs else True
        self.once = kwargs["once"] if "once" in kwargs else False
        self.output_one_step = False
        self.node = node_type
        self.node = partial(self.node, **kwargs, step=step)

        self.feature = nn.Sequential(
            BaseConvModule(2, 64, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node, channels=512),
            # BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            # BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
        )
        self.fc1 = nn.Linear(512 * 3 * 3, 10)
        # self.node1 = self.node()
        # self.fc2 = nn.Linear(512 * 2 * 2, num_classes)

    def forward(self, inputs):

        if self.normalize is not None:
            self.normalize.mean = self.normalize.mean.to(inputs.device)
            self.normalize.std = self.normalize.std.to(inputs.device)
            inputs = self.normalize(inputs)
        if len(inputs.shape) == 4:
            inputs = repeat(inputs, 'b c w h -> t b c w h', t=self.step)
            inputs = rearrange(inputs, 't b c w h -> (t b) c w h')
        elif len(inputs.shape) == 5:
            inputs = rearrange(inputs, 'b t c w h -> (t b) c w h')
        self.reset()

        x = self.feature(inputs)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = rearrange(x, '(t b) c -> b t c', t=self.step)
        if self.output_one_step:
            # x = x[:, self.output_one_step -1:].mean(1)
            x = x[:, self.output_one_step - 1]
        elif self.sum_output:
            # x = F.softmax(x, dim=2)
            x = x.mean(1)
        return x

    def set_backpropagation(self, mode, device=None, alpha=2., relax=1.5, beta_1=0.9, beta_2=0.9, SG='atan'):
        if SG == 'atan':  # Atan surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = Atan1Grad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'rect':  # Rect surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = RectGrad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'qgate':  # Triangular surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = QGateGrad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'sig':  # Sigmoid surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = SigGrad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'gaussian':  # Gaussian surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = GauGrad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'assg':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                             beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'assg-t':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad_T(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                               beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'assg-r':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad_R(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                               beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'assg-s':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad_S(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                               beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'assg-g':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad_G(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                               beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'pdsg':  # PDSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = PDGrad(alpha=alpha, requires_grad=False).to(device)

        for module in self.modules():
            if isinstance(module, MyBaseNode):
                module.mode = mode



class VGGCIFAR(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 data_norm=None,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes
        self.normalize = data_norm
        self.sum_output = kwargs["sum_output"] if "sum_output" in kwargs else True
        self.once = kwargs["once"] if "once" in kwargs else False
        self.output_one_step = False
        self.node = node_type
        self.node = partial(self.node, **kwargs, step=step)

        self.feature = nn.Sequential(
            BaseConvModule(3, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1),
                           node=self.node),
            nn.AvgPool2d(2),
        )
        self.fc1 = nn.Linear(512 * 4 * 4, 512 * 2 * 2)
        self.node1 = self.node()
        self.fc2 = nn.Linear(512 * 2 * 2, num_classes)

    def forward(self, inputs):

        if self.normalize is not None:
            self.normalize.mean = self.normalize.mean.to(inputs.device)
            self.normalize.std = self.normalize.std.to(inputs.device)
            inputs = self.normalize(inputs)
        if len(inputs.shape) == 4:
            inputs = repeat(inputs, 'b c w h -> t b c w h', t=self.step)
            inputs = rearrange(inputs, 't b c w h -> (t b) c w h')
        elif len(inputs.shape) == 5:
            inputs = rearrange(inputs, 't b c w h -> (t b) c w h')
        self.reset()

        x = self.feature(inputs)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(self.node1(x))
        x = rearrange(x, '(t b) c -> b t c', t=self.step)
        if self.output_one_step:
            # x = x[:, self.output_one_step -1:].mean(1)
            x = x[:, self.output_one_step - 1]
        elif self.sum_output:
            # x = F.softmax(x, dim=2)
            x = x.mean(1)
        return x

    def set_backpropagation(self, mode, device=None, alpha=2., relax=1.5, beta_1=0.9, beta_2=0.9, SG='atan'):
        if SG == 'atan':  # Atan surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = Atan1Grad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'rect':  # Rect surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = RectGrad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'qgate':  # Triangular surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = QGateGrad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'sig':  # Sigmoid surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = SigGrad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'gaussian':  # Gaussian surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = GauGrad(alpha=alpha, requires_grad=False).to(device)

        elif SG == 'assg':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                             beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'assg-t':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad_T(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                               beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'assg-r':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad_R(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                               beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'assg-s':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad_S(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                               beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'assg-g':  # ASSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = ASSGrad_G(alpha=alpha, requires_grad=False, step=self.step, relax=relax,
                                               beta_1=beta_1, beta_2=beta_2).to(device)

        elif SG == 'pdsg':  # PDSG surrogate gradient
            for module in self.modules():
                if isinstance(module, MyBaseNode):
                    module.act_fun = PDGrad(alpha=alpha, requires_grad=False).to(device)

        for module in self.modules():
            if isinstance(module, MyBaseNode):
                module.mode = mode

