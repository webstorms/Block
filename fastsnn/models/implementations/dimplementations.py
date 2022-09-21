import numpy as np

from fastsnn.models.builder import LinearDModel
from fastsnn.models.implementations.standard import BaseModel


class DBaseModel(BaseModel):

    def __init__(self, d, recurrent, method, t_len, n_layers=1, detach_recurrent_spikes=False, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, single_spike=True):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        self._d = d
        self._recurrent = recurrent
        self._n_layers = n_layers
        self._detach_recurrent_spikes = detach_recurrent_spikes

    @property
    def hyperparams(self):
        return {**super().hyperparams, "d": self._d, "recurrent": self._recurrent, "n_layers": self._n_layers, "detach_recurrent_spikes": self._detach_recurrent_spikes}


class SHDDModel(DBaseModel):

    def __init__(self, d, recurrent, method, t_len,  n_layers=1, detach_recurrent_spikes=False, heterogeneous_beta=True, beta_requires_grad=True, readout_max=False, single_spike=True, **kwargs):
        super().__init__(d, recurrent, method, t_len, n_layers, detach_recurrent_spikes, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        kwargs["detach_recurrent_spikes"] = detach_recurrent_spikes
        self._model = LinearDModel(d, recurrent, method, t_len, n_in=700, n_out=20, n_hidden=128, n_layers=n_layers, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, scale=10, **kwargs)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)