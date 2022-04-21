import torch
import torch.nn as nn
from brainbox.models import BBModel

from fastsnn.layers import LinearFastLIFNeurons, LinearLIFNeurons


class LinearSNN(BBModel):

    LAYER_FAST_REFAC = 0
    LAYER_VAN_REFAC = 1
    LAYER_VANILLA = 2

    def __init__(self, layer_type, n_in, n_out, n_per_hidden, t_len, beta_init=0.9, beta_range=[0, 1], beta_diff=False, bias_init=0, bias_diff=True):
        super().__init__()
        self.layer_type = layer_type
        self.n_per_hidden = n_per_hidden
        self.beta_init = beta_init
        self.beta_range = beta_range
        self.beta_diff = beta_diff
        self.bias_init = bias_init
        self.bias_diff = bias_diff

        self._layers = nn.ModuleList()

        # Instantiate hidden layers
        for n_hidden in n_per_hidden:
            if layer_type == LinearSNN.LAYER_FAST_REFAC:
                layer = LinearFastLIFNeurons(n_in, n_hidden, t_len, beta_init, beta_range, beta_diff, bias_init, bias_diff)

            else:
                layer = LinearLIFNeurons(n_in, n_hidden, beta_init, beta_range, beta_diff, bias_init, bias_diff, inf_refactory=(layer_type == LinearSNN.LAYER_VAN_REFAC))

            self._layers.append(layer)
            n_in = n_hidden
        n_hidden = n_in

        # Instantiate readout layer
        if layer_type == LinearSNN.LAYER_FAST_REFAC:
            self._readout_layer = LinearFastLIFNeurons(n_hidden, n_out, t_len, beta_init, beta_range, beta_diff, bias_init, bias_diff)
        else:
            self._readout_layer = LinearLIFNeurons(n_hidden, n_out, beta_init, beta_range, beta_diff, bias_init, bias_diff, deactivate_reset=True)

    @property
    def hyperparams(self):
        layers_hyperparams = [layer.hyperparams for layer in self._layers]
        layers_hyperparams.extend([self._readout_layer.hyperparams])
        return {**super().hyperparams, 'layer_type': self.layer_type, 'layers': layers_hyperparams}

    def forward(self, x, return_history=False):

        spike_history = []
        mem_history = []

        for layer in self._layers:
            x, mem = layer(x)

            if return_history:
                spike_history.append(x)
                mem_history.append(mem)

        spikes, mem = self._readout_layer(x)
        spike_history.append(spikes)
        mem_history.append(mem)
        
        return torch.max(mem, 2)[0] if not return_history else [spike_history, mem_history]
