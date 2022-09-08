import torch

from fastsnn.nn.methods import MethodFastNaive
from fastsnn.nn.surrogate import FastSigmoid


# Test MethodFastNaive

def test_fast_method_beta_kernel():
    single_beta_fast_naive = MethodFastNaive(4, FastSigmoid.apply, scale=10, beta=torch.Tensor([0.1]))
    assert torch.allclose(single_beta_fast_naive._beta_kernel[0, 0, 0], torch.Tensor([0.0010, 0.0100, 0.1000, 1.0000]))

    multi_beta_fast_naive = MethodFastNaive(4, FastSigmoid.apply, scale=10, beta=torch.Tensor([0.1, 0.5]))
    assert torch.allclose(multi_beta_fast_naive._beta_kernel[0, 0, 0], torch.Tensor([0.0010, 0.0100, 0.1000, 1.0000]))
    assert torch.allclose(multi_beta_fast_naive._beta_kernel[1, 0, 0], torch.Tensor([0.1250, 0.2500, 0.5000, 1.0000]))


def test_fast_method_phi_kernel():
    fast_naive = MethodFastNaive(4, FastSigmoid.apply, scale=10, beta=torch.Tensor([0.1]))
    assert torch.allclose(fast_naive._phi_kernel, torch.Tensor([[[[4., 3., 2., 1.]]]]))


def test_fast_method_g():
    phi_spikes = torch.zeros(10, 10)
    phi_spikes[0, 0] = 1
    phi_spikes[5, 4] = 1
    assert MethodFastNaive.g(phi_spikes).sum() == 2


def test_differentiable_vars():
    fast_naive = MethodFastNaive(4, FastSigmoid.apply, scale=10, beta=torch.Tensor([0.1]))
    assert not fast_naive._beta_ident_base.requires_grad
    assert not fast_naive._beta_exp.requires_grad
    assert not fast_naive._beta_kernel.requires_grad
    assert not fast_naive._phi_kernel.requires_grad