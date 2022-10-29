import torch

import block.nn.layers as layers


# Test BaseNeurons

def test_beta_clamp():
    base_neurons = layers.BaseNeurons(layers.METHOD_STANDARD, 4, beta_init=[-1, 2])
    base_neurons.beta == torch.Tensor([[0.0010], [0.9990]])


def test_beta_differentiable():
    base_neurons = layers.BaseNeurons(layers.METHOD_STANDARD, 4, beta_requires_grad=False)
    assert not base_neurons.beta.requires_grad

    base_neurons = layers.BaseNeurons(layers.METHOD_STANDARD, 4, beta_requires_grad=True)
    assert base_neurons.beta.requires_grad


# Test LinearNeurons

def test_linear_layers_single_beta():
    assert _linear_models_identical(use_single_beta=True)
    assert _linear_models_identical(use_single_beta=False)


def _linear_models_identical(use_single_beta):
    # Model arguments
    n_in = 200
    n_out = 10
    t_len = 200
    beta_init = torch.rand(1) if use_single_beta else torch.rand(n_out)
    beta_requires_grad = False

    # Models
    van_linear = layers.LinearNeurons(n_in, n_out, layers.METHOD_STANDARD, t_len, beta_init, beta_requires_grad, single_spike=True)
    fast_linear = layers.LinearNeurons(n_in, n_out, layers.METHOD_FAST_NAIVE, t_len, beta_init, beta_requires_grad)
    van_linear._to_current.weight = fast_linear._to_current.weight
    van_linear._to_current.bias = fast_linear._to_current.bias

    # Sample input data
    v_init = torch.rand(128, n_out)
    spikes = torch.rand(128, n_in, t_len)
    van_linear_output = van_linear(spikes)
    fast_linear_output = fast_linear(spikes)

    return torch.allclose(van_linear_output, fast_linear_output)
