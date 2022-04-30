import os
import sys
sys.path.append("/home/luketaylor/PycharmProjects/BrainBox")
sys.path.append("/home/luketaylor/PycharmProjects/FastSNN")
sys.path.append("/data/dpag-auditory-neuroscience/kebl6283/PycharmProjects/BrainBox")
sys.path.append("/data/dpag-auditory-neuroscience/kebl6283/PycharmProjects/FastSNN")
import ast
import argparse

import torch

from fastsnn.benchmark import LinearLayerBenchmarker, LinearModelBenchmarker

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--type", type=str, default="layer")
    parser.add_argument("--fast_layer", type=str, default="True")
    parser.add_argument("--t_len", type=int, default=100)
    parser.add_argument("--input_units", type=int, default=28*28)
    parser.add_argument("--hidden_units", type=int, default=100)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    path = args.path
    type = args.type
    fast_layer = ast.literal_eval(args.fast_layer)
    t_len = args.t_len
    input_units = args.input_units
    hidden_units = args.hidden_units
    n_layers = args.n_layers
    batch_size = args.batch_size
    n_samples = 5+1

    if type == "layer":
        benchmarker = LinearLayerBenchmarker(fast_layer, t_len, input_units, hidden_units, min_r=0, max_r=200, n_samples=n_samples, batch_size=batch_size)
    elif type == "model":
        benchmarker = LinearModelBenchmarker(fast_layer, t_len, input_units, hidden_units, n_layers, min_r=0, max_r=200, n_samples=n_samples, batch_size=batch_size)

    benchmarker.benchmark()
    benchmarker.save(path)


if __name__ == "__main__":
    main()
