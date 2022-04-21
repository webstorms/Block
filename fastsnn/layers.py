import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from brainbox.models import BBModel


class BaseLIFNeurons(BBModel):

    def __init__(self, n_in, n_out, beta_init=0.9, beta_range=[0, 1], beta_diff=False, bias_init=0, bias_diff=True, surrogate_scale=100):
        super(BaseLIFNeurons, self).__init__()
        self._n_in = n_in
        self._n_out = n_out
        self._beta_init = beta_init
        self._beta_range = beta_range
        self._beta_diff = beta_diff
        self._bias_init = bias_init
        self._bias_diff = bias_diff
        self._surrogate_scale = surrogate_scale

        self._spike_function = FastSigmoid.get(surrogate_scale)
        self._betas = nn.Parameter(
            data=torch.Tensor([beta_init]),
            requires_grad=beta_diff)

    @property
    def beta(self):
        return torch.clamp(self._betas, min=self._beta_range[0], max=self._beta_range[1])

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'n_in': self._n_in, 'n_out': self._n_out, 'beta_range': self._beta_range,
                'beta_diff': self._beta_diff, 'bias_diff': self._bias_diff, 'surrogate_scale': self._surrogate_scale}


class VanillaBaseLIFNeurons(BaseLIFNeurons):

    def __init__(self, n_in, n_out, beta_init=0.9, beta_range=[0, 1], beta_diff=False, bias_init=0, bias_diff=True, surrogate_scale=100, detach_reset_grad=True, deactivate_reset=False, inf_refactory=False):
        super().__init__(n_in, n_out, beta_init, beta_range, beta_diff, bias_init, bias_diff, surrogate_scale)
        self._detach_reset_grad = detach_reset_grad
        self._deactivate_reset = deactivate_reset
        self._inf_refactory = inf_refactory

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'detach_reset_grad': self._detach_reset_grad, 'deactivate_reset': self._deactivate_reset, 'inf_refactory': self._inf_refactory}

    def input_to_current(self, pre_spikes):
        raise NotImplementedError

    def forward(self, pre_spikes):
        # pre_spikes: n_batch x n_in x n_timesteps x ...

        # Initialise membrane voltage and tracking variables
        with torch.no_grad():
            t_len = pre_spikes.shape[2]
            out_shape = self.input_to_current(pre_spikes[:, :, 0]).shape
            mem = torch.zeros(out_shape, device=pre_spikes.device, dtype=pre_spikes.dtype)
            d = torch.zeros(out_shape, device=pre_spikes.device, dtype=pre_spikes.dtype)
            mem_list = []
            post_spikes_list = []

        for t in range(t_len):
            input_current = self.input_to_current(pre_spikes[:, :, t])

            # Update membrane potential
            mem = self.beta * mem + input_current
            # mem = torch.einsum('bn...,n->bn...', mem, self.beta) + input_current

            # To spike or not to spike
            post_spikes = self._spike_function(mem - 1)

            if self._inf_refactory:
                post_spikes *= (1 - d)
                d = torch.maximum(d, post_spikes)
            post_spikes_list.append(post_spikes)

            if not self._deactivate_reset:
                # Reset membrane potential for spiked neurons
                reset = post_spikes if not self._detach_reset_grad else post_spikes.detach()
                # mem *= (1 - reset)
                mem -= reset
            mem_list.append(mem.clone())

        return torch.stack(post_spikes_list, dim=2), torch.stack(mem_list, dim=2)


class LinearFastLIFNeurons(BaseLIFNeurons):

    def __init__(self, n_in, n_out, t_len, beta_init=0.9, beta_range=[0, 1], beta_diff=False, bias_init=0, bias_diff=True, surrogate_scale=100):
        super().__init__(n_in, n_out, beta_init, beta_range, beta_diff, bias_init, bias_diff, surrogate_scale)
        self.t_len = t_len

        self.weight = nn.Parameter(data=torch.rand(n_out, n_in))
        self.bias = nn.Parameter(data=torch.rand(n_out), requires_grad=bias_diff)
        self._betas_kernel = nn.Parameter(torch.cat([torch.Tensor([beta_init]) ** (t_len-i-1) for i in range(t_len)], dim=0).T.view(1, 1, 1, t_len))
        self._sum_kernel = nn.Parameter(torch.ones(t_len).view(1, 1, 1, t_len))

        self.init_weight(self.weight, 'uniform', a=-np.sqrt(1/n_in), b=np.sqrt(1/n_in))
        self.init_weight(self.bias, 'constant', c=bias_init)

    @property
    def hyperparams(self):
        return {**super().hyperparams, 't_len': self.t_len}

    @staticmethod
    def s(s_sum):
        return F.relu(s_sum * (1 - s_sum) + 1) * s_sum

    def forward(self, pre_spikes):
        # pre_spikes: n_batch x n_in x n_timesteps

        input_current = torch.einsum('bjt,ij->bti', pre_spikes, self.weight) + self.bias

        input_current = input_current.permute(0, 2, 1)
        pad_input_current = F.pad(input_current, pad=(self.t_len-1, 0)).unsqueeze(1)
        int_mem = F.conv2d(pad_input_current, self._betas_kernel)
        int_spikes = F.pad(self._spike_function(int_mem - 1), pad=(self.t_len - 1, 0))
        sum_spikes = F.pad(F.conv2d(int_spikes, self._sum_kernel), pad=(self.t_len-1, 0))
        sum_spikes = F.conv2d(sum_spikes, self._sum_kernel)

        spikes = LinearFastLIFNeurons.s(sum_spikes).squeeze(1)

        return spikes, int_mem.squeeze(1)


class LinearLIFNeurons(VanillaBaseLIFNeurons):

    def __init__(self, n_in, n_out, beta_init=0.9, beta_range=[0, 1], beta_diff=False, bias_init=0, bias_diff=True, surrogate_scale=100, detach_reset_grad=True, deactivate_reset=False, inf_refactory=False):
        super().__init__(n_in, n_out, beta_init, beta_range, beta_diff, bias_init, bias_diff, surrogate_scale, detach_reset_grad, deactivate_reset, inf_refactory)
        self.pre_spikes_to_current = nn.Linear(n_in, n_out)
        self.pre_spikes_to_current.bias.requires_grad = bias_diff

        self.init_weight(self.pre_spikes_to_current.weight, 'uniform', a=-np.sqrt(1/n_in), b=np.sqrt(1/n_in))
        self.init_weight(self.pre_spikes_to_current.bias, 'constant', c=bias_init)

    def input_to_current(self, pre_spikes):
        return self.pre_spikes_to_current(pre_spikes)


class FastSigmoid(torch.autograd.Function):

    """
    Use the normalized negative part of a fast sigmoid for the surrogate gradient
    as done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @classmethod
    def get(cls, scale):
        cls.scale = scale

        return cls.apply

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (FastSigmoid.scale * torch.abs(input) + 1.0) ** 2

        return grad
