import os
from pathlib import Path

import torch
torch.backends.cudnn.benchmark = True

from block.benchmark import LayerBenchmarker


def run_2d():
    path = os.path.join(Path(__file__).parent.parent, "results/benchmarks/2d")
    batch_sizes = [16, 32, 64, 128]
    t_lens = [2 ** i for i in range(3, 12)]
    hidden_units = [i * 100 for i in range(1, 11)]
    methods = ["standard", "fast_naive"]
    beta_options = [(True, False), (True, True), (False, False)]
    input_units = 1000

    for batch_size in batch_sizes:
        for t_len in t_lens:
            for units in hidden_units:
                for method in methods:
                    for beta_option in beta_options:
                        heterogeneous_beta, beta_requires_grad = beta_option
                        benchmarker = LayerBenchmarker(method, t_len, input_units, units, n_layers=1, heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, batch_size=batch_size)
                        benchmarker.benchmark()
                        benchmarker.save(path)


def run_large_n():
    path = os.path.join(Path(__file__).parent.parent, "results/benchmarks/units")
    batch_sizes = [16, 32, 64, 128]
    t_len = 2**7
    hidden_units = [i * 2000 for i in range(1, 11)]
    methods = ["standard", "fast_naive"]
    beta_options = [(True, False), (True, True), (False, False)]
    input_units = 1000

    for batch_size in batch_sizes:
        for units in hidden_units:
            for method in methods:
                for beta_option in beta_options:
                    heterogeneous_beta, beta_requires_grad = beta_option
                    benchmarker = LayerBenchmarker(method, t_len, input_units, units, n_layers=1, heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, batch_size=batch_size)
                    benchmarker.benchmark()
                    benchmarker.save(path)


def run_layers():
    path = os.path.join(Path(__file__).parent.parent, "results/benchmarks/layers")
    batch_sizes = [16, 32, 64, 128]
    t_len = 2**7
    hidden_units = [i * 100 for i in range(1, 11)]
    n_layers = [1, 2, 3, 4, 5]
    methods = ["standard", "fast_naive"]
    beta_options = [(True, False), (True, True), (False, False)]
    input_units = 1000

    for batch_size in batch_sizes:
        for units in hidden_units:
            for n_layer in n_layers:
                for method in methods:
                    for beta_option in beta_options:
                        heterogeneous_beta, beta_requires_grad = beta_option
                        benchmarker = LayerBenchmarker(method, t_len, input_units, units, n_layers=n_layer, heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, batch_size=batch_size)
                        benchmarker.benchmark()
                        benchmarker.save(path)


if __name__ == "__main__":
    run_2d()
    run_large_n()
    run_layers()

