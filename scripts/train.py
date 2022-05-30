import os
import ast
import argparse

import torch
import numpy as np

from fastsnn import datasets, models, trainer


def main():
    torch.backends.cudnn.benchmark = True

    # Building settings
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--n_hidden", type=int)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--fast_layer", type=str)
    parser.add_argument("--skip_connections", type=str, default="True")
    parser.add_argument("--bias", type=float, default=0)
    parser.add_argument("--hidden_tau", type=float, default=10)
    parser.add_argument("--readout_tau", type=float, default=20)
    parser.add_argument("--dt", type=float, default=1)

    # Training arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--device", type=str, default="cuda")

    # Load arguments
    args = parser.parse_args()
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    n_hidden = args.n_hidden
    n_layers = args.n_layers
    fast_layer = ast.literal_eval(args.fast_layer)
    skip_connections = ast.literal_eval(args.skip_connections)
    bias = args.bias
    hidden_tau = args.hidden_tau
    readout_tau = args.readout_tau
    dt = args.dt
    dataset = args.dataset
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    device = args.device

    # Instantiate the dataset
    print("Building dataset...")
    if args.dataset == "fmnist":
        dataset = datasets.FMNISTDataset(os.path.join(path, "data"))
    elif args.dataset == "nmnist":
        dataset = datasets.NMNISTDataset(os.path.join(path, "data", "N-MNIST"), dt=dt)
    elif args.dataset == "shd":
        dataset = datasets.SHDDataset(os.path.join(path, "data", "SHD"), dt=dt)

    # Instantiate the model
    print("Building model...")
    n_in = dataset.n_in
    n_out = dataset.n_out
    t_len = dataset.t_len
    hidden_beta = np.exp(-dt / hidden_tau)
    readout_beta = np.exp(-dt / readout_tau)
    model = models.LinearModel(t_len, n_in, n_out, n_hidden, n_layers, fast_layer, skip_connections, bias, hidden_beta, readout_beta)

    # Instantiate the trainer
    print("Started training...")
    snn_trainer = trainer.Trainer(os.path.join(path, f"results/datasets/{args.dataset}"), model, dataset, epoch, batch_size, lr, device=device)
    snn_trainer.train(save=True)


if __name__ == "__main__":
    main()
