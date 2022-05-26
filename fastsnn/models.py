import torch
import torch.nn as nn
from brainbox.models import BBModel

from fastsnn.layers import LinearFastLIFNeurons, LinearLIFNeurons


class BaseModel(BBModel):

    def __init__(self, t_len, n_in, n_out, n_hidden, n_layers, fast_layer=False, skip_connections=False, bias=0.5, hidden_beta=0.9, readout_beta=0.9, **kwargs):
        super().__init__()
        self._t_len = t_len
        self._n_in = n_in
        self._n_out = n_out
        self._n_hidden = n_hidden
        self._n_layers = n_layers
        self._fast_layer = fast_layer
        self._skip_connections = skip_connections
        self._bias = bias
        self._hidden_beta = hidden_beta
        self._readout_beta = readout_beta

        self._layers = nn.ModuleList()

        # Build hidden layers
        for i in range(n_layers):
            n_in = n_in if i == 0 else n_hidden
            self._layers.append(self._build_layer(n_in, n_hidden, bias, hidden_beta, **kwargs))

        # Build readout layer
        if fast_layer:
            self._readout_layer = LinearFastLIFNeurons(t_len, n_hidden, n_out, readout_beta, bias)
        else:
            self._readout_layer = LinearLIFNeurons(n_hidden, n_out, readout_beta, bias, deactivate_reset=True)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "t_len": self._t_len, "n_in": self._n_in, "n_out": self._n_out,
                "n_hidden": self._n_hidden, "n_layers": self._n_layers, "fast_layer": self._fast_layer,
                "skip_connections": self._skip_connections, "bias": self._bias, "hidden_beta": self._hidden_beta,
                "readout_beta": self._readout_beta}

    def forward(self, spikes):
        spike_history = []
        mem_history = []

        for i, layer in enumerate(self._layers):
            new_spikes, mem = layer(spikes)

            if self._skip_connections and i > 0:
                spikes = torch.clamp(new_spikes + spikes, min=0, max=1)
            else:
                spikes = new_spikes

            spike_history.append(spikes)
            mem_history.append(mem)

        spikes, mem = self._readout_layer(spikes)

        spike_history.append(spikes)
        mem_history.append(mem)

        return torch.max(mem, 2)[0], spike_history, mem_history

    def _build_layer(self, n_in, n_out, bias, beta, **kwargs):
        pass


class LinearModel(BaseModel):

    def __init__(self, t_len, n_in, n_out, n_hidden, n_layers, fast_layer=False, skip_connections=False, bias=0.5, hidden_beta=0.9, readout_beta=0.9, **kwargs):
        super().__init__(t_len, n_in, n_out, n_hidden, n_layers, fast_layer, skip_connections, bias, hidden_beta, readout_beta, **kwargs)

    def _build_layer(self, n_in, n_out, bias, beta, **kwargs):
        if self._fast_layer:
            return LinearFastLIFNeurons(self._t_len, n_in, n_out, beta, bias)
        else:
            if kwargs.get("single_spike", False):
                return LinearLIFNeurons(n_in, n_out, beta, bias, deactivate_reset=True, single_spike=True)
            else:
                return LinearLIFNeurons(n_in, n_out, beta, bias)
