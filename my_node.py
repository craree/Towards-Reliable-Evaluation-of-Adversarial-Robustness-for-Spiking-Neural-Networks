from urllib.parse import quote_plus

import math
from cgitb import reset

import torch
from braincog.base.node.node import *
from sympy.codegen.ast import continue_

class Poi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rand = torch.rand_like(x)
        out = (x>=rand).float()
        # ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

poi = Poi.apply

class Poisson(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = poi(x)
        return out

class atan1(torch.autograd.Function):
    """
    使用 Atan 作为代理梯度函数
    对应的原函数为:

    .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}
    反向传播的函数为:

    .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

    """

    @staticmethod
    def forward(ctx, inputs, alpha):
        output = inputs.gt(0.).float()
        ctx.save_for_backward(inputs, alpha)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        grad_alpha = None
        dsdv = ctx.saved_tensors[1] / 2 / \
                   (1 + (ctx.saved_tensors[1] * math.pi /
                         2 * ctx.saved_tensors[0]).square())

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * dsdv

        return grad_x, grad_alpha


class Atan1Grad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return atan1.apply(x, alpha)

    def forward(self, x):
        if len(x.shape) == 4:
            alpha = self.alpha.view(-1, 1, 1, 1)
        else:
            alpha = self.alpha.view(-1, 1)
        return self.act_fun(x, alpha)

class ASSGrad(SurrogateFunctionBase):
    def __init__(self, alpha=0.4, step=4, relax=1.5, beta_1=0.9, beta_2=0.9, requires_grad=False):
        super().__init__(alpha, requires_grad)
        """
               Adaptive Sharpness Surrogate Gradient (ASSG) module.

               This class implements a surrogate gradient function whose sharpness parameter
               adapts based on the first- and second-order moment statistics of the inputs. 

               Args:
                   alpha (float): Upper bound of the expected gradient-vanishing degree A/2
                   step (int): Number of timesteps of SNN.
                   relax (float): Relaxation coefficient for the second-order term.
                   beta_1 (float): Momentum coefficient for the first-order moment.
                   beta_2 (float): Momentum coefficient for the second-order moment.
                   
        """
        # M_t: first-order moment (EMA of |x|)
        # D_t: second-order moment (EMA of |x - M_t|)
        self.M_t = [1.] * step
        self.D_t = [0.] * step
        self.a = 2 * math.tan(alpha * math.pi) / math.pi # a = 2 * tan(pi * A / 2) / pi
        self.eot_flag = False
        self.step = step
        self.t = 0
        self.relax = relax
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    @staticmethod
    def act_fun(x, alpha):
        return atan1.apply(x, alpha)

    def forward(self, x):
        if self.eot_flag:
            # In EOT mode, skip updating statistical moments
            pass
        else:
            self.M_t[self.t] = self.beta_1 * self.M_t[self.t] + (1 - self.beta_1) * torch.abs(x).detach()
            self.D_t[self.t] = self.beta_2 * self.D_t[self.t] + (1 - self.beta_2) * torch.abs(torch.abs(x)-self.M_t[self.t]).detach()

        alpha = self.a / (self.M_t[self.t] + self.relax * self.D_t[self.t] + 0.0001)
        self.t = self.t + 1
        # Apply Atan surrogate function with adaptive sharpness alpha
        return self.act_fun(x, alpha)

    def reset(self):
        """
              Reset timestep counter to zero.

              Should be called at the beginning of each forward propagation cycle.
        """
        self.t = 0


class ASSGrad_T(SurrogateFunctionBase):
    def __init__(self, alpha=0.8, step=4, relax=1.5, beta_1=0.9, beta_2=0.9, requires_grad=False):
        super().__init__(alpha, requires_grad)
        """
               Adaptive Sharpness Surrogate Gradient (ASSG) module.

               This class implements a surrogate gradient function whose sharpness parameter
               adapts based on the first- and second-order moment statistics of the inputs. 

               Args:
                   alpha (float): Upper bound of the expected gradient-vanishing degree A/2
                   step (int): Number of timesteps of SNN.
                   relax (float): Relaxation coefficient for the second-order term.
                   beta_1 (float): Momentum coefficient for the first-order moment.
                   beta_2 (float): Momentum coefficient for the second-order moment.

        """
        # M_t: first-order moment (EMA of |x|)
        # D_t: second-order moment (EMA of |x - M_t|)
        self.M_t = [1.] * step
        self.D_t = [0.] * step
        self.a = alpha
        self.eot_flag = False
        self.step = step
        self.t = 0
        self.relax = relax
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    @staticmethod
    def act_fun(x, alpha):
        return quadratic_gate.apply(x, alpha)

    def forward(self, x):
        if self.eot_flag:
            # In EOT mode, skip updating statistical moments
            pass
        else:
            self.M_t[self.t] = self.beta_1 * self.M_t[self.t] + (1 - self.beta_1) * torch.abs(x).detach()
            self.D_t[self.t] = self.beta_2 * self.D_t[self.t] + (1 - self.beta_2) * torch.abs(
                torch.abs(x) - self.M_t[self.t]).detach()

        alpha = self.a / (self.M_t[self.t] + self.relax * self.D_t[self.t] + 0.0001)
        self.t = self.t + 1
        # Apply Atan surrogate function with adaptive sharpness alpha
        return self.act_fun(x, alpha)

    def reset(self):
        """
              Reset timestep counter to zero.

              Should be called at the beginning of each forward propagation cycle.
        """
        self.t = 0

class gate1(torch.autograd.Function):
    """
    使用 gate 作为代理梯度函数
    对应的原函数为:

    .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)
    反向传播的函数为:

    .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            grad_x = torch.where(
                x.abs() < 1. / alpha,
                alpha / 2.0 * torch.ones_like(x),
                torch.zeros_like(x)
            )
            ctx.save_for_backward(grad_x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None

class RectGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return gate1.apply(x, alpha)

class ASSGrad_R(SurrogateFunctionBase):
    def __init__(self, alpha=0.8, step=4, relax=1.5, beta_1=0.9, beta_2=0.9, requires_grad=False):
        super().__init__(alpha, requires_grad)
        """
               Adaptive Sharpness Surrogate Gradient (ASSG) module.

               This class implements a surrogate gradient function whose sharpness parameter
               adapts based on the first- and second-order moment statistics of the inputs. 

               Args:
                   alpha (float): Upper bound of the expected gradient-vanishing degree A/2
                   step (int): Number of timesteps of SNN.
                   relax (float): Relaxation coefficient for the second-order term.
                   beta_1 (float): Momentum coefficient for the first-order moment.
                   beta_2 (float): Momentum coefficient for the second-order moment.

        """
        # M_t: first-order moment (EMA of |x|)
        # D_t: second-order moment (EMA of |x - M_t|)
        self.M_t = [1.] * step
        self.D_t = [0.] * step
        self.a = alpha
        self.eot_flag = False
        self.step = step
        self.t = 0
        self.relax = relax
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    @staticmethod
    def act_fun(x, alpha):
        return gate1.apply(x, alpha)

    def forward(self, x):
        if self.eot_flag:
            # In EOT mode, skip updating statistical moments
            pass
        else:
            self.M_t[self.t] = self.beta_1 * self.M_t[self.t] + (1 - self.beta_1) * torch.abs(x).detach()
            self.D_t[self.t] = self.beta_2 * self.D_t[self.t] + (1 - self.beta_2) * torch.abs(
                torch.abs(x) - self.M_t[self.t]).detach()

        alpha = self.a / (self.M_t[self.t] + self.relax * self.D_t[self.t] + 0.0001)
        self.t = self.t + 1
        # Apply Atan surrogate function with adaptive sharpness alpha
        return self.act_fun(x, alpha)

    def reset(self):
        """
              Reset timestep counter to zero.

              Should be called at the beginning of each forward propagation cycle.
        """
        self.t = 0

class sigmoid1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            grad_x = alpha * torch.sigmoid(alpha * x) * (1 - torch.sigmoid(alpha * x))
            ctx.save_for_backward(grad_x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None

class SigGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return sigmoid1.apply(x, alpha)

class ASSGrad_S(SurrogateFunctionBase):
    def __init__(self, alpha=0.8, step=4, relax=1.5, beta_1=0.9, beta_2=0.9, requires_grad=False):
        super().__init__(alpha, requires_grad)
        """
               Adaptive Sharpness Surrogate Gradient (ASSG) module.

               This class implements a surrogate gradient function whose sharpness parameter
               adapts based on the first- and second-order moment statistics of the inputs. 

               Args:
                   alpha (float): Upper bound of the expected gradient-vanishing degree A/2
                   step (int): Number of timesteps of SNN.
                   relax (float): Relaxation coefficient for the second-order term.
                   beta_1 (float): Momentum coefficient for the first-order moment.
                   beta_2 (float): Momentum coefficient for the second-order moment.

        """
        # M_t: first-order moment (EMA of |x|)
        # D_t: second-order moment (EMA of |x - M_t|)
        self.M_t = [1.] * step
        self.D_t = [0.] * step
        self.a = alpha
        self.eot_flag = False
        self.step = step
        self.t = 0
        self.relax = relax
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    @staticmethod
    def act_fun(x, alpha):
        return sigmoid1.apply(x, alpha)

    def forward(self, x):
        if self.eot_flag:
            # In EOT mode, skip updating statistical moments
            pass
        else:
            self.M_t[self.t] = self.beta_1 * self.M_t[self.t] + (1 - self.beta_1) * torch.abs(x).detach()
            self.D_t[self.t] = self.beta_2 * self.D_t[self.t] + (1 - self.beta_2) * torch.abs(
                torch.abs(x) - self.M_t[self.t]).detach()

        alpha = self.a / (self.M_t[self.t] + self.relax * self.D_t[self.t] + 0.0001)
        self.t = self.t + 1
        # Apply Atan surrogate function with adaptive sharpness alpha
        return self.act_fun(x, alpha)

    def reset(self):
        """
              Reset timestep counter to zero.

              Should be called at the beginning of each forward propagation cycle.
        """
        self.t = 0

class gaussian1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            grad_x = torch.exp(-0.5 * (alpha * x)**2) * (alpha / math.sqrt(2 * math.pi))
            ctx.save_for_backward(grad_x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None

class GauGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return gaussian1.apply(x, alpha)

class ASSGrad_G(SurrogateFunctionBase):
    def __init__(self, alpha=0.8, step=4, relax=1.5, beta_1=0.9, beta_2=0.9, requires_grad=False):
        super().__init__(alpha, requires_grad)
        """
               Adaptive Sharpness Surrogate Gradient (ASSG) module.

               This class implements a surrogate gradient function whose sharpness parameter
               adapts based on the first- and second-order moment statistics of the inputs. 

               Args:
                   alpha (float): Upper bound of the expected gradient-vanishing degree A/2
                   step (int): Number of timesteps of SNN.
                   relax (float): Relaxation coefficient for the second-order term.
                   beta_1 (float): Momentum coefficient for the first-order moment.
                   beta_2 (float): Momentum coefficient for the second-order moment.

        """
        # M_t: first-order moment (EMA of |x|)
        # D_t: second-order moment (EMA of |x - M_t|)
        self.M_t = [1.] * step
        self.D_t = [0.] * step
        self.a = alpha
        self.eot_flag = False
        self.step = step
        self.t = 0
        self.relax = relax
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    @staticmethod
    def act_fun(x, alpha):
        return gaussian1.apply(x, alpha)

    def forward(self, x):
        if self.eot_flag:
            # In EOT mode, skip updating statistical moments
            pass
        else:
            self.M_t[self.t] = self.beta_1 * self.M_t[self.t] + (1 - self.beta_1) * torch.abs(x).detach()
            self.D_t[self.t] = self.beta_2 * self.D_t[self.t] + (1 - self.beta_2) * torch.abs(
                torch.abs(x) - self.M_t[self.t]).detach()

        alpha = self.a / (self.M_t[self.t] + self.relax * self.D_t[self.t] + 0.0001)
        self.t = self.t + 1
        # Apply Atan surrogate function with adaptive sharpness alpha
        return self.act_fun(x, alpha)

    def reset(self):
        """
              Reset timestep counter to zero.

              Should be called at the beginning of each forward propagation cycle.
        """
        self.t = 0

def piecewise_quadratic_backward(grad_output: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor):
    sigma = sigma.expand_as(x)
    grad = torch.exp(- ((x - 0.5 * sigma)/(np.sqrt(2)*sigma))**2) / (np.sqrt(2*np.pi)*sigma)
    # grad = torch.exp(- (x / (np.sqrt(2) * sigma)) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    grad_input = grad_output * grad
    return grad_input, None, None, None

class pdsg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, sigma: torch.Tensor):
        if x.requires_grad:
            ctx.save_for_backward(x, sigma)
        return x.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(grad_output, ctx.saved_tensors[0], ctx.saved_tensors[1])

class PDGrad(SurrogateFunctionBase):
    """
       Potential-Dependent Surrogate Gradient (PDSG) module.
    """
    def __init__(self, alpha=2., requires_grad=True):
        super().__init__(alpha, requires_grad)
        self.mem_pot = []

    @staticmethod
    def act_fun(x, sigma):
        return pdsg.apply(x, sigma)

    def forward(self, x):
        self.mem_pot.append(x.clone().detach())
        v = torch.stack(self.mem_pot, dim=0)
        with torch.no_grad():
            if len(x.shape) == 4:  # for conv layer
                if x.shape[2] == x.shape[3]:  # conventional conv layer
                    dim = (0, 1, 3, 4)
                else:  # transformer layer
                    dim = (0, 1, 2, 4)
            else:  # for linear layer
                dim = tuple(range(len(x.shape)))
            sigma = v.std(dim=dim, keepdim=True).squeeze(0)
            sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
        return self.act_fun(x, sigma)

    def reset(self):
        self.mem_pot = []

class GradAvg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.mean(dim=0,keepdim=True)
        grad_input = grad_input.expand_as(grad_output)
        return grad_input


class MyBaseNode(nn.Module, abc.ABC):
    """
    神经元模型的基类
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param mem_detach: 是否将上一时刻的膜电位在计算图中截断
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=.5,
                 v_reset=0.,
                 dt=1.,
                 step=8,
                 requires_thres_grad=False,
                 sigmoid_thres=False,
                 requires_fp=False,
                 layer_by_layer=False,
                 n_groups=1,
                 *args,
                 **kwargs):

        super(MyBaseNode, self).__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=requires_thres_grad)
        self.sigmoid_thres = sigmoid_thres
        self.mem = 0.
        self.spike = 0.
        self.dt = dt
        self.feature_map = []
        self.mem_collect = []
        self.requires_fp = requires_fp
        self.v_reset = v_reset
        self.step = step
        self.layer_by_layer = layer_by_layer
        self.mode = 'bptt'
        self.groups = n_groups
        self.mem_detach = kwargs['mem_detach'] if 'mem_detach' in kwargs else False
        self.requires_mem = kwargs['requires_mem'] if 'requires_mem' in kwargs else False

    @abc.abstractmethod
    def calc_spike(self, inputs):
        """
        :return: None
        """

        pass

    def get_thres(self):
        return self.threshold if not self.sigmoid_thres else self.threshold.sigmoid()

    def rearrange2node(self, inputs):
        if len(inputs.shape) == 4:
            outputs = rearrange(inputs, '(t b) c w h -> t b c w h', t=self.step)
        elif len(inputs.shape) == 2:
            outputs = rearrange(inputs, '(t b) c -> t b c', t=self.step)
        else:
            raise NotImplementedError

        return outputs

    def rearrange2op(self, inputs):
        if len(inputs.shape) == 5:
            outputs = rearrange(inputs, 't b c w h -> (t b) c w h')
        elif len(inputs.shape) == 3:
            outputs = rearrange(inputs, ' t b c -> (t b) c')
        else:
            raise NotImplementedError
        return outputs

    def forward(self, inputs):
        """
        torch.nn.Module 默认调用的函数，用于计算膜电位的输入和脉冲的输出
        在```self.requires_fp is True``` 的情况下，可以使得```self.feature_map```用于记录trace
        :param inputs: 当前输入的膜电位
        :return: 输出的脉冲
        """

        inputs = self.rearrange2node(inputs)
        outputs = self.calc_spike(inputs)
        outputs = self.rearrange2op(outputs)

        return outputs


    def n_reset(self):
        """
        神经元重置，用于模型接受两个不相关输入之间，重置神经元所有的状态
        :return: None
        """
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []

    def get_n_attr(self, attr):

        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return None

    def set_n_warm_up(self, flag):
        """
        一些训练策略会在初始的一些epoch，将神经元视作ANN的激活函数训练，此为设置是否使用该方法训练
        :param flag: True：神经元变为激活函数， False：不变
        :return: None
        """
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        """
        动态设置神经元的阈值
        :param thresh: 阈值
        :return:
        """
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)

    def set_n_tau(self, tau):
        """
        动态设置神经元的衰减系数，用于带Leaky的神经元
        :param tau: 衰减系数
        :return:
        """
        if hasattr(self, 'tau'):
            self.tau = Parameter(torch.tensor(tau, dtype=torch.float), requires_grad=False)
        else:
            raise NotImplementedError

class RateBpLIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, tau):
        mem = 0.
        th = 0.5
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem + (x[t, ...] - mem) * tau
            # mem = mem * tau + x[t, ...]
            spike = ((mem - th) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out, torch.tensor(tau))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, out, tau = ctx.saved_tensors
        x = x.mean(0, keepdim=True)

        '''
        gamma = 0.01
        exp1 = torch.exp((tau-1+x)/gamma)
        g1 = exp1/(1+exp1)
        log1 = gamma*torch.log(1+exp1)
        h = torch.log(log1/x)
        grad = (log1-x*g1)*torch.log(tau)/(x*log1*h*h)
        grad = grad*(x>0).float()*(x<1).float()
        '''
        #RGA
        gamma = 0.2
        ext = 1  #
        des = 1
        th = 0.5
        beta = 2.
        grad = (x > (1 - tau) * th).float() * (x <= th * beta).float() * (des - gamma * th + gamma * th * tau) / ((tau + beta - 1) * th)+ (
                    x <= (1 - tau) * th).float() * (x >= 0).float() * gamma
        grad_input = grad_output * grad

        return grad_input, None, None

class BPTRLIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, tau):
        mem = 0.
        th = 0.5
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem + (x[t, ...] - mem) * tau
            # mem = mem * tau + x[t, ...]
            spike = ((mem - th) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out, torch.tensor(tau))
        return out

    @staticmethod
    def backward(ctx, grad_output):

        '''
        gamma = 0.01
        exp1 = torch.exp((tau-1+x)/gamma)
        g1 = exp1/(1+exp1)
        log1 = gamma*torch.log(1+exp1)
        h = torch.log(log1/x)
        grad = (log1-x*g1)*torch.log(tau)/(x*log1*h*h)
        grad = grad*(x>0).float()*(x<1).float()
        '''

        # BPTR
        x, out, tau = ctx.saved_tensors
        T = out.shape[0]
        out = out.mean(0).unsqueeze(0)
        grad_input = grad_output * (out > 0).float()

        return grad_input, None, None

class MyLIFNode(MyBaseNode):
    """
    LIF neuron model
    parameter act_fun determine the surrogate gradient function
    """

    def __init__(self, threshold=0.5, tau=0.5, act_fun=QGateGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.w = nn.Parameter(torch.as_tensor(tau), requires_grad=False)
        self.ratebp = RateBpLIF.apply
        self.bptr = BPTRLIF.apply
        self.grad_avg = GradAvg.apply

    def calc_spike(self, inputs):

        if self.mode == 'rga': #RGA
            return self.ratebp(inputs, self.w)
        elif self.mode == 'bptr': #BPTR
            return self.bptr(inputs, self.w)
        elif self.mode == 'hart': #HART
            T = len(inputs)
            inputs = self.grad_avg(inputs)
            self.mem = torch.zeros_like(inputs[0])
            spike_pot = []
            for t in range(T):
                self.mem = self.mem.detach() + (inputs[t] - self.mem.detach()) * self.w
                # self.mem = self.mem * self.w + inputs[t]
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                self.spike = self.act_fun(self.mem - self.threshold)
                self.mem = self.mem * (1 - self.spike.detach())
                # self.mem = self.mem * (1 - self.spike)
                spike_pot.append(self.spike)
            output = torch.stack(spike_pot, dim=0)
            if self.requires_mem is True:
                self.mem_collect = torch.stack(self.mem_collect, dim=0)
            if self.requires_fp is True:
                self.feature_map = output
            return output
        else: #STBP
            T = len(inputs)
            spike_pot = []
            for t in range(T):
                self.mem = self.mem + (inputs[t] - self.mem) * self.w
                # if self.requires_mem is True:
                #     self.mem_collect.append(self.mem)
                self.mem_collect.append(self.mem)
                self.spike = self.act_fun(self.mem - self.threshold)
                self.mem = self.mem * (1 - self.spike.detach())
                spike_pot.append(self.spike)
            output = torch.stack(spike_pot, dim=0)
            # if self.requires_mem is True:
            #     self.mem_collect = torch.stack(self.mem_collect, dim=0)
            self.mem_collect = torch.stack(self.mem_collect, dim=0)
            if self.requires_fp is True:
                self.feature_map = output
            return output

        # self.mem = self.mem - self.spike.detach() * self.threshold

    def n_reset(self):
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
        if hasattr(self.act_fun, 'reset'):
            self.act_fun.reset()

class RateBpIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        mem = 0.
        th = 1.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem + x[t, ...]
            spike = ((mem - th) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, out = ctx.saved_tensors
        x = x.mean(0, keepdim=True)

        '''
        gamma = 0.01
        exp1 = torch.exp((tau-1+x)/gamma)
        g1 = exp1/(1+exp1)
        log1 = gamma*torch.log(1+exp1)
        h = torch.log(log1/x)
        grad = (log1-x*g1)*torch.log(tau)/(x*log1*h*h)
        grad = grad*(x>0).float()*(x<1).float()
        '''

        #RGA
        gamma = 0.2
        ext = 1  #
        th = 1.
        beta = 2.
        grad = (x >= 0).float() * (x <= th * beta).float() * 1/ (th * beta)
        grad_input = grad_output * grad

        return grad_input, None, None

class BPTRIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        mem = 0.
        th = 1.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem + x[t, ...]
            # mem = mem * tau + x[t, ...]
            spike = ((mem - th) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):

        '''
        gamma = 0.01
        exp1 = torch.exp((tau-1+x)/gamma)
        g1 = exp1/(1+exp1)
        log1 = gamma*torch.log(1+exp1)
        h = torch.log(log1/x)
        grad = (log1-x*g1)*torch.log(tau)/(x*log1*h*h)
        grad = grad*(x>0).float()*(x<1).float()
        '''

        #BPTR
        x, out = ctx.saved_tensors
        T = out.shape[0]
        out = out.mean(0).unsqueeze(0)
        grad_input = grad_output * (out > 0).float()

        return grad_input, None, None

class MyIFNode(MyBaseNode):
    """
    IF neuron model
    """

    def __init__(self, threshold=1, act_fun=QGateGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.ratebp = RateBpIF.apply
        self.bptr = BPTRIF.apply
        self.grad_avg = GradAvg.apply

    def calc_spike(self, inputs):

        if self.mode == 'rga':
            return self.ratebp(inputs)
        elif self.mode == 'bptr':
            return self.bptr(inputs)
        elif self.mode == 'hart':
            T = len(inputs)
            inputs = self.grad_avg(inputs)
            self.mem = torch.zeros_like(inputs[0])
            spike_pot = []
            for t in range(T):
                self.mem = self.mem.detach() + inputs[t]
                # self.mem = self.mem * self.w + inputs[t]
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                self.spike = self.act_fun(self.mem - self.threshold)
                self.mem = self.mem * (1 - self.spike.detach())
                spike_pot.append(self.spike)
            output = torch.stack(spike_pot, dim=0)
            if self.requires_mem is True:
                self.mem_collect = torch.stack(self.mem_collect, dim=0)
            if self.requires_fp is True:
                self.feature_map = output
            return output
        else:
            T = len(inputs)
            spike_pot = []
            for t in range(T):
                self.mem = self.mem + inputs[t]
                # self.mem = self.mem * self.w + inputs[t]
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                self.spike = self.act_fun(self.mem - self.threshold)
                self.mem = self.mem * (1 - self.spike.detach())
                spike_pot.append(self.spike)
            output = torch.stack(spike_pot, dim=0)
            if self.requires_mem is True:
                self.mem_collect = torch.stack(self.mem_collect, dim=0)
            if self.requires_fp is True:
                self.feature_map = output
            return output

        # self.mem = self.mem - self.spike.detach() * self.threshold

    def n_reset(self):
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
        if hasattr(self.act_fun, 'reset'):
            self.act_fun.reset()

class RateBpLIF2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, tau):
        mem = 0.
        th = 1.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * tau + x[t, ...]
            # mem = mem * tau + x[t, ...]
            spike = ((mem - th) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out, torch.tensor(tau))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, out, tau = ctx.saved_tensors
        x = x.mean(0, keepdim=True)

        #RGA
        gamma = 0.2
        ext = 1  #
        des = 1
        th = 1.
        beta = 2.
        grad = (x > (1 - tau) * th).float() * (x <= th * beta).float() * (des - gamma * th + gamma * th * tau) / ((tau + beta - 1) * th)+ (
                    x <= (1 - tau) * th).float() * (x >= 0).float() * gamma
        grad_input = grad_output * grad

        return grad_input, None, None

class BPTRLIF2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, tau):
        mem = 0.
        th = 1.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * tau + x[t, ...]
            # mem = mem * tau + x[t, ...]
            spike = ((mem - th) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out, torch.tensor(tau))
        return out

    @staticmethod
    def backward(ctx, grad_output):

        # BPTR
        x, out, tau = ctx.saved_tensors
        T = out.shape[0]
        out = out.mean(0).unsqueeze(0)
        grad_input = grad_output * (out > 0).float()

        return grad_input, None, None

class MyLIFNode2(MyBaseNode):
    """
    LIF-2 neuron model
    """

    def __init__(self, threshold=1, tau=0.5, act_fun=QGateGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.w = nn.Parameter(torch.as_tensor(tau), requires_grad=False)
        self.ratebp = RateBpLIF2.apply
        self.bptr = BPTRLIF2.apply
        self.grad_avg = GradAvg.apply

    def calc_spike(self, inputs):

        if self.mode == 'rga':
            return self.ratebp(inputs, self.w)
        elif self.mode == 'bptr':
            return self.bptr(inputs, self.w)
        elif self.mode == 'hart':
            T = len(inputs)
            inputs = self.grad_avg(inputs)
            self.mem = torch.zeros_like(inputs[0])
            spike_pot = []
            for t in range(T):
                self.mem = self.w * self.mem.detach() + inputs[t]
                # self.mem = self.mem * self.w + inputs[t]
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                self.spike = self.act_fun(self.mem - self.threshold)
                self.mem = self.mem * (1 - self.spike.detach())
                spike_pot.append(self.spike)
            output = torch.stack(spike_pot, dim=0)
            if self.requires_mem is True:
                self.mem_collect = torch.stack(self.mem_collect, dim=0)
            if self.requires_fp is True:
                self.feature_map = output
            return output
        else:
            T = len(inputs)
            spike_pot = []
            for t in range(T):
                self.mem = self.w * self.mem + inputs[t]
                # self.mem = self.mem * self.w + inputs[t]
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                self.spike = self.act_fun(self.mem - self.threshold)
                self.mem = self.mem * (1 - self.spike.detach())
                spike_pot.append(self.spike)
            output = torch.stack(spike_pot, dim=0)
            if self.requires_mem is True:
                self.mem_collect = torch.stack(self.mem_collect, dim=0)
            if self.requires_fp is True:
                self.feature_map = output
            return output

    def n_reset(self):
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
        if hasattr(self.act_fun, 'reset'):
            self.act_fun.reset()


class PSNode(MyBaseNode):
    """
    PSN neuron model
    """
    def __init__(self, threshold=0.0, act_fun=QGateGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.fc = nn.Linear(self.step, self.step, bias=True)
        nn.init.constant_(self.fc.bias, -1.0)
        self.grad_avg = GradAvg.apply

    def calc_spike(self, inputs):
        T = len(inputs)
        if self.mode == 'hart':
            inputs = self.grad_avg(inputs)
            self.mem = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, inputs.flatten(1)).view(
                inputs.shape)
            if self.requires_mem is True:
                self.mem_collect.append(self.mem)
            spike_pot = []
            for t in range(T):
                spike_pot.append(self.act_fun(self.mem[t]))
            output = torch.stack(spike_pot, dim=0)
            if self.requires_mem is True:
                self.mem_collect = torch.stack(self.mem_collect, dim=0)
            if self.requires_fp is True:
                self.feature_map = output
            return output
        else:
            self.mem = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, inputs.flatten(1)).view(
                inputs.shape)
            if self.requires_mem is True:
                self.mem_collect.append(self.mem)
            spike_pot = []
            for t in range(T):
                spike_pot.append(self.act_fun(self.mem[t]))
            output = torch.stack(spike_pot, dim=0)
            if self.requires_mem is True:
                self.mem_collect = torch.stack(self.mem_collect, dim=0)
            if self.requires_fp is True:
                self.feature_map = output
            return output

    def n_reset(self):
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
        if hasattr(self.act_fun, 'reset'):
            self.act_fun.reset()
