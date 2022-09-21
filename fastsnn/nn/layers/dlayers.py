import torch
import torch.nn as nn
import numpy as np

from fastsnn.nn.surrogate import FastSigmoid
import fastsnn.nn.methods as methods
from fastsnn.nn.functional import cat
from fastsnn.nn.layers import standard


class BaseDNeurons(standard.BaseNeurons):

    def __init__(self, d, recurrent, method, t_len, beta_init=[0.9], beta_requires_grad=False, spike_func=FastSigmoid.apply, scale=10, **kwargs):
        super().__init__(method, t_len, beta_init, beta_requires_grad, spike_func, scale, **kwargs)
        self._d = d
        self._recurrent = recurrent

        # We repeatedly pass input data through the same block shape, but also create a last block of a different length
        # to process last bit of input data incase t_len is not evenly divisible by d
        self._t_len_block = t_len // d
        self._block = self._get_method_func(self._t_len_block, **kwargs)
        self._last_block = self._get_method_func(t_len - d * self._t_len_block)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "d": self._d, "recurrent": self._recurrent}

    def forward(self, x, return_type=methods.RETURN_SPIKES):
        # current: b x n x t
        # beta: n
        # v_init: b x n

        v_init = None
        input_current = self._get_input_current(x)
        block_responses = []

        for i in range(self._d):
            input_current_slice = input_current[:, :, i * self._t_len_block: (i+1) * self._t_len_block]
            block_response = self._block(input_current_slice, self.beta, v_init, return_type)
            block_responses.append((block_response, ) if type(block_response) == torch.Tensor else block_response)

            if self._recurrent:
                spikes = block_response if return_type == methods.RETURN_SPIKES else block_response[0]
                v_init = self._get_recurrent_input_current(spikes)

        if self._last_block.t_len != 0:
            input_current_slice = input_current[:, :, self._d * self._t_len_block:]
            block_response = self._last_block(input_current_slice, self.beta, v_init, return_type)
            block_responses.append((block_response, ) if type(block_response) == torch.Tensor else block_response)

        return cat(block_responses)

    def _get_input_current(self, x):
        return x

    def _get_recurrent_input_current(self, spikes):
        raise NotImplementedError


class LinearDNeurons(BaseDNeurons):

    def __init__(self, n_in, n_out, d, recurrent, method, t_len, detach_recurrent_spikes=True, use_recurrent_max=True, beta_init=[0.9], beta_requires_grad=False, spike_func=FastSigmoid.apply, scale=10, **kwargs):
        super().__init__(d, recurrent, method, t_len, beta_init, beta_requires_grad, spike_func, scale, **kwargs)
        self._n_in = n_in
        self._n_out = n_out
        self._detach_recurrent_spikes = detach_recurrent_spikes
        self._use_recurrent_max = use_recurrent_max

        self._to_input_current = nn.Linear(n_in, n_out)
        self.init_weight(self._to_input_current.weight, "uniform", a=-np.sqrt(1 / n_in), b=np.sqrt(1 / n_in))
        self.init_weight(self._to_input_current.bias, "constant", c=0)

        self._to_recurrent_current = nn.Linear(n_out, n_out)
        self.init_weight(self._to_recurrent_current.weight, "uniform", a=-np.sqrt(1 / n_out), b=np.sqrt(1 / n_out))
        self.init_weight(self._to_recurrent_current.bias, "constant", c=0)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "n_in": self._n_in, "n_out": self._n_out, "detach_recurrent_spikes": self._detach_recurrent_spikes, "use_recurrent_max": self._use_recurrent_max}

    def _get_input_current(self, x):
        x = x.permute(0, 2, 1)
        current = self._to_input_current(x)
        current = current.permute(0, 2, 1)

        return current

    def _get_recurrent_input_current(self, spikes):
        spikes = torch.amax(spikes, dim=2) if self._use_recurrent_max else torch.sum(spikes, dim=2)
        if self._detach_recurrent_spikes:
            spikes = spikes.detach()
        current = self._to_recurrent_current(spikes)

        return current
