import numpy as np
from brainbox.models import BBModel

from block.models.builder import LinearModel, ConvModel


class BaseModel(BBModel):

    def __init__(self, method, t_len, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, single_spike=True, recurrent=False, n_layers=1, n_neurons=None):
        super().__init__()
        self._method = method
        self._t_len = t_len
        self._heterogeneous_beta = heterogeneous_beta
        self._beta_requires_grad = beta_requires_grad
        self._readout_max = readout_max
        self._single_spike = single_spike
        self._recurrent = recurrent
        self._n_layers = n_layers
        self._n_neurons = n_neurons

    @property
    def hyperparams(self):
        return {**super().hyperparams, "method": self._method, "t_len": self._t_len, "heterogeneous_beta": self._heterogeneous_beta, "beta_requires_grad": self._beta_requires_grad, "readout_max": self._readout_max, "single_spike": self._single_spike, "recurrent": self._recurrent, "n_layers": self._n_layers, "n_neurons": self._n_neurons}


class YingYangModel(BaseModel):

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=False, single_spike=True):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        self._model = LinearModel(method, t_len, n_in=4, n_out=3, n_hidden=120, n_layers=1, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)


class LinearMNINSTModel(BaseModel):

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=False, single_spike=True):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        self._model = LinearModel(method, t_len, n_in=784, n_out=10, n_hidden=800, n_layers=1, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)


class ConvMNINSTModel(BaseModel):

    _CHANNELS = [32, 64]
    _NEURONS = [25088, 12544, 3136]

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=True, single_spike=True):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        self._model = ConvModel(method, ConvMNINSTModel._CHANNELS, n_in=1, n_out=10, t_len=t_len, beta_init=1, heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, hidden_neurons=ConvMNINSTModel._NEURONS, readout_max=readout_max, single_spike=single_spike)

    def forward(self, spikes):
        return self._model(spikes)


class LinearFMNINSTModel(BaseModel):

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=False, single_spike=True):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        self._model = LinearModel(method, t_len, n_in=784, n_out=10, n_hidden=1000, n_layers=1, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)


class ConvFMNINSTModel(BaseModel):

    _CHANNELS = [32, 64]
    _NEURONS = [25088, 12544, 3136]

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=True, single_spike=True):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        self._model = ConvModel(method, ConvFMNINSTModel._CHANNELS, n_in=1, n_out=10, t_len=t_len, beta_init=1, heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, hidden_neurons=ConvFMNINSTModel._NEURONS, readout_max=readout_max, single_spike=single_spike)

    def forward(self, spikes):
        return self._model(spikes)


class NMNISTModel(BaseModel):

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=False, single_spike=True):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        self._model = LinearModel(method, t_len, n_in=1156, n_out=20, n_hidden=300, n_layers=1, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
        # Note: Whoops, erroneously did all training with n_out=20 and should have been n_out=10. Will still work with n_out=10, as the additional 10 neurons
        # just become 10 dead readout neurons...

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)


class SHDModel(BaseModel):

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=False, single_spike=True):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        self._model = LinearModel(method, t_len, n_in=700, n_out=20, n_hidden=300, n_layers=1, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)
