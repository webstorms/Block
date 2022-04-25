import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from brainbox.models import BBModel


class BaseLIFNeurons(BBModel):

    def __init__(self, n_in, n_out, beta_init=0.9, bias_init=0, surrogate_scale=100):
        super(BaseLIFNeurons, self).__init__()
        self._n_in = n_in
        self._n_out = n_out
        self._beta_init = beta_init
        self._bias_init = bias_init
        self._surrogate_scale = surrogate_scale

        self._spike_function = FastSigmoid.get(surrogate_scale)
        self.beta = nn.Parameter(
            data=torch.Tensor([beta_init]),
            requires_grad=False)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "n_in": self._n_in, "n_out": self._n_out, "surrogate_scale": self._surrogate_scale}


class VanillaBaseLIFNeurons(BaseLIFNeurons):

    def __init__(self, n_in, n_out, beta_init=0.9, bias_init=0, surrogate_scale=100, deactivate_reset=False, single_spike=False):
        super().__init__(n_in, n_out, beta_init, bias_init, surrogate_scale)
        self._deactivate_reset = deactivate_reset
        self._single_spike = single_spike

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'deactivate_reset': self._deactivate_reset}

    def input_to_current(self, pre_spikes):
        raise NotImplementedError

    def forward(self, pre_spikes):
        # pre_spikes: n_batch x n_in x n_timesteps x ...

        # Initialise membrane voltage and tracking variables
        t_len = pre_spikes.shape[2]
        spike_mask = None
        mem = None
        mem_list = []
        post_spikes_list = []

        for t in range(t_len):
            input_current = self.input_to_current(pre_spikes[:, :, t])

            # Update membrane potential
            if mem is None:
                new_mem = input_current
            else:
                new_mem = self.beta * mem + input_current

            # To spike or not to spike
            post_spikes = self._spike_function(new_mem - 1)

            if spike_mask is None:
                spike_mask = torch.zeros(post_spikes.shape)

            if self._single_spike:
                post_spikes *= (1 - spike_mask)
                spike_mask = torch.maximum(spike_mask, post_spikes)

            post_spikes_list.append(post_spikes)

            # Reset membrane potential for spiked neurons
            if not self._deactivate_reset:
                new_mem -= post_spikes

            mem_list.append(new_mem)
            mem = new_mem

        return torch.stack(post_spikes_list, dim=2), torch.stack(mem_list, dim=2)


class LinearLIFNeurons(VanillaBaseLIFNeurons):

    def __init__(self, n_in, n_out, beta_init=0.9, bias_init=0, surrogate_scale=100, deactivate_reset=False, single_spike=False):
        super().__init__(n_in, n_out, beta_init, bias_init, surrogate_scale, deactivate_reset, single_spike)
        self.pre_spikes_to_current = nn.Linear(n_in, n_out)
        self.init_weight(self.pre_spikes_to_current.weight, "uniform", a=-np.sqrt(1/n_in), b=np.sqrt(1/n_in))
        self.init_weight(self.pre_spikes_to_current.bias, "constant", c=bias_init)

    def input_to_current(self, pre_spikes):
        return self.pre_spikes_to_current(pre_spikes)


class LinearFastLIFNeurons(BaseLIFNeurons):

    def __init__(self, t_len, n_in, n_out, beta_init=0.9, bias_init=0, surrogate_scale=100):
        super().__init__(n_in, n_out, beta_init, bias_init, surrogate_scale)
        self._t_len = t_len

        self.pre_spikes_to_current = nn.Linear(n_in, n_out)
        self.init_weight(self.pre_spikes_to_current.weight, "uniform", a=-np.sqrt(1/n_in), b=np.sqrt(1/n_in))
        self.init_weight(self.pre_spikes_to_current.bias, "constant", c=bias_init)

        self._betas_kernel = nn.Parameter(torch.cat([torch.Tensor([beta_init]) ** (t_len-i-1) for i in range(t_len)], dim=0).T.view(1, 1, 1, t_len))
        self._sum_kernel = nn.Parameter(torch.ones(1, 1, 1, t_len))

    @property
    def hyperparams(self):
        return {**super().hyperparams, 't_len': self._t_len}

    def forward(self, pre_spikes):
        # pre_spikes: n_batch x n_in x n_timesteps

        # 1. Convert pre-synaptic spikes to input current
        pre_spikes = pre_spikes.permute(0, 2, 1)  # b x time x n
        input_current = self.pre_spikes_to_current(pre_spikes)

        # 2. Calculate membrane potential without reset
        input_current = input_current.permute(0, 2, 1)  # b x n x time
        pad_input_current = F.pad(input_current, pad=(self._t_len - 1, 0)).unsqueeze(1)
        int_mem = F.conv2d(pad_input_current, self._betas_kernel)

        # 3. Map no-reset membrane potentials to output spikes
        int_spikes = self._spike_function(int_mem - 1)
        pad_int_spikes = F.pad(int_spikes, pad=(self._t_len - 1, 0))
        sum_a = F.conv2d(pad_int_spikes, self._sum_kernel)
        pad_sum_a = F.pad(sum_a, pad=(self._t_len - 1, 0))
        sum_b = F.conv2d(pad_sum_a, self._sum_kernel)
        spikes = LinearFastLIFNeurons.g(sum_b).squeeze(1)

        return spikes, int_mem.squeeze(1)

    @staticmethod
    def g(s_sum):
        return F.relu(s_sum * (1 - s_sum) + 1) * s_sum


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
