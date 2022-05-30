import os

import torch

from fastsnn.benchmark import LinearLayerBenchmarker

torch.backends.cudnn.benchmark = True


def main():
    path = os.path.join(os.path.dirname(os.getcwd()), "results/benchmarks")
    batch_sizes = [16, 32, 64, 128, 256]
    t_lens = [2 ** i for i in range(3, 12)]
    hidden_units = [i * 100 for i in range(1, 11)]
    speedup_layer = [False, True]

    n_samples = 10 + 1  # +1 sample is to let cudnn run different conv algorithms which can take long
    input_units = 1000

    for batch_size in batch_sizes:
        for t_len in t_lens:
            for units in hidden_units:
                for speedup in speedup_layer:
                    benchmarker = LinearLayerBenchmarker(speedup, t_len, input_units, units, min_r=0, max_r=200, n_samples=n_samples, batch_size=batch_size)
                    benchmarker.benchmark()
                    benchmarker.save(path)


if __name__ == "__main__":
    main()
