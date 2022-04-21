import sys
sys.path.append('/Users/home/PycharmProjects/BrainBox')
sys.path.append('/Users/home/PycharmProjects/FastSNN')

import torch

from fastsnn.models import LinearSNN
from fastsnn.layers import LinearFastLIFNeurons, LinearLIFNeurons


def test_snn_params():
    beta_init = 0.8
    beta_range = [0.001, 0.999]
    beta_diff = True
    bias_init = 0.2

    for i in range(3):
        fast_snn = LinearSNN(i, 10, 1, [100], 100, beta_init=beta_init, beta_range=beta_range, beta_diff=beta_diff, bias_init=bias_init)
        for layer in fast_snn._layers:
            assert abs(layer.beta.item() - beta_init) < 10e-5
            assert layer._beta_range == beta_range
            assert layer._beta_diff == beta_diff
            assert layer._beta_diff == beta_diff
            if i == 0:
                assert abs(layer.bias[0].item() - bias_init) < 10e-5
            else:
                assert abs(layer.pre_spikes_to_current.bias[0].item() - bias_init) < 10e-5


def test_layer_output():
    t_len = 500

    for _ in range(10):
        van_snn = LinearLIFNeurons(5, 1000, inf_refactory=True)
        fast_snn = LinearFastLIFNeurons(5, 1000, t_len)

        fast_snn.weight = van_snn.pre_spikes_to_current.weight
        fast_snn.bias = van_snn.pre_spikes_to_current.bias

        pre_spikes = torch.ones(100, 5, t_len)
        van_spikes, van_mem = van_snn(pre_spikes)
        fast_spikes, fast_mem = fast_snn(pre_spikes)
        assert (van_spikes.long() ^ fast_spikes.long()).sum() == 0
