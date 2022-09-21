import torch

from fastsnn.nn import methods
import fastsnn.nn.layers as layers
from fastsnn.models import builder


def test_even_divisibility():
    base_neurons = layers.BaseDNeurons(d=2, recurrent=False, method="standard", t_len=40)
    assert base_neurons._block.t_len == 20
    assert base_neurons._last_block.t_len == 0


def test_odd_divisibility():
    base_neurons = layers.BaseDNeurons(d=3, recurrent=False, method="standard", t_len=40)
    assert base_neurons._block.t_len == 13
    assert base_neurons._last_block.t_len == 1


def test_return_spikes():
    base_neurons = layers.BaseDNeurons(d=2, recurrent=False, method="standard", t_len=40)
    input_data = torch.rand(2, 10, 2**7)
    out = base_neurons(input_data)
    assert type(out) == torch.Tensor


def test_return_spikes_and_mem():
    base_neurons = layers.BaseDNeurons(d=2, recurrent=False, method="standard", t_len=40)
    input_data = torch.rand(2, 10, 2**7)
    out = base_neurons(input_data, return_type=methods.RETURN_SPIKES_AND_MEM)
    assert len(out) == 2


def test_setting_recurrent_flags():
    linear_model = builder.LinearDModel(d=2, recurrent=True, method="standard", t_len=2**7, n_in=10, n_out=2, n_hidden=100, n_layers=1, skip_connections=False, hidden_beta=0.9, readout_beta=0.9, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, detach_recurrent_spikes=False, use_recurrent_max=False)
    assert not linear_model._layers[0]._detach_recurrent_spikes
    assert not linear_model._layers[0]._detach_recurrent_spikes

    linear_model = builder.LinearDModel(d=2, recurrent=True, method="standard", t_len=2**7, n_in=10, n_out=2, n_hidden=100, n_layers=1, skip_connections=False, hidden_beta=0.9, readout_beta=0.9, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, detach_recurrent_spikes=True, use_recurrent_max=True)
    assert linear_model._layers[0]._detach_recurrent_spikes
    assert linear_model._layers[0]._detach_recurrent_spikes