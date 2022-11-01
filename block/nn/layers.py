import torch
import torch.nn as nn
import numpy as np
from brainbox.models import BBModel

from block.nn.surrogate import FastSigmoid
import block.nn.methods as methods


METHOD_STANDARD = "standard"
METHOD_FAST_NAIVE = "fast_naive"
METHOD_FAST_OPTIMISED = "fast_optimised"


class BaseNeurons(BBModel):

    def __init__(self, method, t_len, beta_init=[0.9], beta_requires_grad=False, spike_func=FastSigmoid.apply, scale=100, **kwargs):
        super().__init__()
        self._method = method
        self._t_len = t_len
        self._beta_init = beta_init
        self._beta_requires_grad = beta_requires_grad
        self._spike_func = spike_func
        self._scale = scale

        self._beta = nn.Parameter(data=torch.Tensor(beta_init), requires_grad=beta_requires_grad)
        self._method_func = self._get_method_func(t_len, **kwargs)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "method": self._method, "t_len": self._t_len, "beta_init": self._beta_init, "beta_requires_grad": self._beta_requires_grad, "spike_func": self._spike_func.__class__.__name__}

    @property
    def beta(self):
        return torch.clamp(self._beta, min=0.00, max=1)

    def forward(self, x, v_init=None, return_type=methods.RETURN_SPIKES):
        # current: b x n x t
        # beta: n
        # v_init: b x n
        return self._method_func(x, self.beta, v_init, return_type)

    def get_recurrent_current(self, spikes):
        raise NotImplementedError

    def _get_method_func(self, t_len, **kwargs):
        if self._method == METHOD_STANDARD:
            recurrent_source = self.get_recurrent_current if kwargs.get("recurrent", False) else None
            return methods.MethodStandard(t_len, self._spike_func, self._scale, kwargs.get("single_spike", False), kwargs.get("integrator", False), recurrent_source)
        elif self._method == METHOD_FAST_NAIVE:
            return methods.MethodFastNaive(t_len, self._spike_func, self._scale, self.beta)
        elif self._method == METHOD_FAST_OPTIMISED:
            raise NotImplementedError


class LinearNeurons(BaseNeurons):

    def __init__(self, n_in, n_out, method, t_len, beta_init=[0.9], beta_requires_grad=False, spike_func=FastSigmoid.apply, scale=10, **kwargs):
        super().__init__(method, t_len, beta_init, beta_requires_grad, spike_func, scale, **kwargs)
        self._n_in = n_in
        self._n_out = n_out

        self._to_current = nn.Linear(n_in, n_out)
        self._to_recurrent_current = nn.Linear(n_out, n_out)
        self.init_weight(self._to_current.weight, "uniform", a=-np.sqrt(1 / n_in), b=np.sqrt(1 / n_in))
        self.init_weight(self._to_current.bias, "constant", c=0)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "n_in": self._n_in, "n_out": self._n_out}

    def get_recurrent_current(self, spikes):
        return self._to_recurrent_current(spikes)

    def forward(self, x, v_init=None, return_type=methods.RETURN_SPIKES):
        x = x.permute(0, 2, 1)
        current = self._to_current(x)
        current = current.permute(0, 2, 1)
        spikes = super().forward(current, v_init, return_type)

        return spikes


class ConvNeurons(BaseNeurons):

    def __init__(self, n_in, n_out, kernel, stride, method, t_len, beta_init=[0.9], beta_requires_grad=False, spike_func=FastSigmoid.apply, scale=10, **kwargs):
        super().__init__(method, t_len, beta_init, beta_requires_grad, spike_func, scale, **kwargs)
        self._n_in = n_in
        self._n_out = n_out
        self._kernel = kernel
        self._stride = stride
        self._flatten = kwargs.get("flatten", False)

        self._to_current = nn.Conv3d(n_in, n_out, (1, kernel, kernel), (1, stride, stride))

        n_in = kernel * kernel
        self.init_weight(self._to_current.weight, "uniform", a=-np.sqrt(1 / n_in), b=np.sqrt(1 / n_in))
        self.init_weight(self._to_current.bias, "constant", c=0)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "n_in": self._n_in, "n_out": self._n_out, "kernel": self._kernel, "stride": self._stride}

    def forward(self, x, v_init=None, return_type=methods.RETURN_SPIKES):
        current = self._to_current(x)
        b, n, t, h, w = current.shape

        current = current.permute(0, 1, 3, 4, 2)
        current = current.flatten(start_dim=1, end_dim=3)
        spikes = super().forward(current, v_init, return_type)

        if not self._flatten:
            spikes = spikes.view(b, n, h, w, t)
            spikes = spikes.permute(0, 1, 4, 2, 3)

        return spikes

