import numpy as np
import torch
import torch.nn as nn
from brainbox.models import BBModel

from block.nn.layers import LinearNeurons, ConvNeurons
import block.nn.methods as methods


class LinearModel(BBModel):

    def __init__(self, method, t_len, n_in, n_out, n_hidden, n_layers, skip_connections=False, hidden_beta=0.9, readout_beta=0.9, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, **kwargs):
        super().__init__()
        self._method = method
        self._t_len = t_len
        self._n_in = n_in
        self._n_out = n_out
        self._n_hidden = n_hidden
        self._n_layers = n_layers
        self._skip_connections = skip_connections
        self._hidden_beta = hidden_beta
        self._readout_beta = readout_beta
        self._heterogeneous_beta = heterogeneous_beta
        self._beta_requires_grad = beta_requires_grad
        self._readout_max = readout_max

        self._layers = nn.ModuleList()

        # Build hidden layers
        for i in range(n_layers):
            n_in = n_in if i == 0 else n_hidden
            self._layers.append(self._build_layer(n_in, n_hidden, hidden_beta, heterogeneous_beta, beta_requires_grad, **kwargs))

        self._readout_layer = self._build_layer(n_hidden, n_out, readout_beta, heterogeneous_beta, beta_requires_grad, single_spike=True, integrator=True)

    @staticmethod
    def build_beta(beta_init, n, heterogeneous_beta):
        return [beta_init for _ in range(n if heterogeneous_beta else 1)]

    @property
    def hyperparams(self):
        return {**super().hyperparams, "method": self._method,  "t_len": self._t_len, "n_in": self._n_in,
                "n_out": self._n_out, "n_hidden": self._n_hidden, "n_layers": self._n_layers,
                "skip_connections": self._skip_connections, "hidden_beta": self._hidden_beta,
                "readout_beta": self._readout_beta, "heterogeneous_beta": self._heterogeneous_beta,
                "beta_requires_grad": self._beta_requires_grad, "readout_max": self._readout_max}

    def forward(self, spikes, return_all=False, deactivate_readout=False):

        spikes_list = []

        for i, layer in enumerate(self._layers):
            new_spikes = layer(spikes)

            if self._skip_connections and i > 0:
                # spikes = torch.clamp(new_spikes + spikes, min=0, max=1)
                spikes = new_spikes + spikes
            else:
                spikes = new_spikes

            if return_all:
                spikes_list.append(new_spikes)

        if deactivate_readout:
            return spikes

        spikes, mem = self._readout_layer(spikes, return_type=methods.RETURN_SPIKES_AND_MEM)
        output = torch.max(mem, 2)[0] if self._readout_max else torch.sum(mem, 2)

        if return_all:
            return output, spikes_list

        return output

    def _build_layer(self, n_in, n_out, beta_init, heterogeneous_beta, beta_requires_grad, **kwargs):
        print("building layer..")
        beta_init = LinearModel.build_beta(beta_init, n_out, heterogeneous_beta)
        return LinearNeurons(n_in, n_out, self._method, self._t_len, beta_init, beta_requires_grad, **kwargs)


class ConvModel(BBModel):

    def __init__(self, method, channels, n_in, n_out, t_len, beta_init, heterogeneous_beta, beta_requires_grad, hidden_neurons, readout_max=True, single_spike=True):
        super().__init__()
        self._readout_max = readout_max

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.ConstantPad3d((2, 2, 2, 2, 0, 0), 0),
            ConvNeurons(n_in, channels[0], 5, 1, method, t_len, LinearModel.build_beta(beta_init, hidden_neurons[0], heterogeneous_beta), beta_requires_grad, single_spike=single_spike),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            # Block 2
            nn.ConstantPad3d((2, 2, 2, 2, 0, 0), 0),
            ConvNeurons(channels[0], channels[1], 5, 1, method, t_len, LinearModel.build_beta(beta_init, hidden_neurons[1], heterogeneous_beta), beta_requires_grad, single_spike=single_spike),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        )
        n_linear = 1000
        self.linear_layers = nn.Sequential(
            LinearNeurons(hidden_neurons[-1], n_linear, method, t_len, LinearModel.build_beta(beta_init, n_linear, heterogeneous_beta), beta_requires_grad, single_spike=single_spike),
        )
        self.readout_layer = LinearNeurons(n_linear, n_out, method, t_len, LinearModel.build_beta(beta_init, n_out, heterogeneous_beta), beta_requires_grad, single_spike=True, integrator=True)

    def forward(self, spikes):
        spikes = self.conv_layers(spikes)
        spikes = spikes.permute(0, 1, 3, 4, 2)
        spikes = spikes.flatten(start_dim=1, end_dim=3)

        spikes = self.linear_layers(spikes)
        spikes, mem = self.readout_layer(spikes, return_type=methods.RETURN_SPIKES_AND_MEM)
        output = torch.max(mem, 2)[0] if self._readout_max else torch.sum(mem, 2)

        return output


class VGG11Model(BBModel):

    def __init__(self, method, channels, n_in, n_out, t_len, beta_init, heterogeneous_beta, beta_requires_grad, hidden_neurons, readout_max=True, single_spike=True):
        super().__init__()
        self._readout_max = readout_max

        sc = 4
        scale = 1
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            ConvNeurons(n_in, 128, 3, 1, method, t_len, LinearModel.build_beta(beta_init, hidden_neurons[0], heterogeneous_beta), beta_requires_grad, scale=scale, single_spike=single_spike, sc=None),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

            # Block 2
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            ConvNeurons(128, 256, 3, 1, method, t_len, LinearModel.build_beta(beta_init, hidden_neurons[0], heterogeneous_beta), beta_requires_grad, scale=scale, single_spike=single_spike, sc=sc),
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            ConvNeurons(256, 256, 3, 1, method, t_len, LinearModel.build_beta(beta_init, hidden_neurons[0], heterogeneous_beta), beta_requires_grad, scale=scale, single_spike=single_spike, sc=sc),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

            # Block 3
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            ConvNeurons(256, 512, 3, 1, method, t_len, LinearModel.build_beta(beta_init, hidden_neurons[0], heterogeneous_beta), beta_requires_grad, scale=scale, single_spike=single_spike, sc=sc),
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            ConvNeurons(512, 512, 3, 1, method, t_len, LinearModel.build_beta(beta_init, hidden_neurons[0], heterogeneous_beta), beta_requires_grad, scale=scale, single_spike=single_spike, sc=sc),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

        )
        n_lin = 4096 // 2
        self.linear_layers = nn.Sequential(
            LinearNeurons(hidden_neurons[-1], n_lin, method, t_len, LinearModel.build_beta(beta_init, n_lin, heterogeneous_beta=False), beta_requires_grad=False, single_spike=single_spike),
            LinearNeurons(n_lin, 1000, method, t_len, LinearModel.build_beta(beta_init, n_lin, heterogeneous_beta=False), beta_requires_grad=False, single_spike=single_spike),
        )
        self.readout_layer = LinearNeurons(1000, n_out, method, t_len, LinearModel.build_beta(beta_init, n_out, heterogeneous_beta=False), beta_requires_grad=False, single_spike=True, integrator=True)

        self.linear_layers[0].init_weight(self.linear_layers[0]._to_current.weight, "uniform", a=-np.sqrt(sc / hidden_neurons[-1]), b=np.sqrt(sc / hidden_neurons[-1]))
        self.linear_layers[1].init_weight(self.linear_layers[1]._to_current.weight, "uniform", a=-np.sqrt(sc / n_lin), b=np.sqrt(sc / n_lin))
        self.readout_layer.init_weight(self.readout_layer._to_current.weight, "uniform", a=-np.sqrt(sc / 1000), b=np.sqrt(sc / 1000))

    def forward(self, spikes):
        spikes = self.conv_layers(spikes)
        spikes = spikes.permute(0, 1, 3, 4, 2)
        spikes = spikes.flatten(start_dim=1, end_dim=3)
        spikes = self.linear_layers(spikes)
        spikes, mem = self.readout_layer(spikes, return_type=methods.RETURN_SPIKES_AND_MEM)
        output = torch.max(mem, 2)[0] if self._readout_max else torch.sum(mem, 2)

        return output