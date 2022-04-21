import os
import sys
sys.path.append(os.path.join('/home/luketaylor/PycharmProjects', 'BrainBox'))
sys.path.append(os.path.join('/home/luketaylor/PycharmProjects', 'FastSNN'))
import ast
import argparse

import torch

from fastsnn.datasets import TimeToFirstSpike, StaticImageSpiking
from fastsnn.trainer import FastSNNTrainer
from fastsnn.models import LinearSNN


def main():

    torch.backends.cudnn.benchmark = True

    # Building settings
    parser = argparse.ArgumentParser(description='Save model outputs.')

    parser.add_argument('--path', type=str, default='.', help='project path (default: .)')

    # model arguments
    parser.add_argument('--layer_type', type=int, default=0, help='layer type (default: 0)')
    parser.add_argument('--n_per_hidden', type=str, default='[]', help='number of units per hidden layer (default: [])')
    parser.add_argument('--t_len', type=int, default=100, help='simulation duration (default: 100)')
    parser.add_argument('--beta_init', type=float, default=0.9, help='initial beta (default: 0.9)')
    parser.add_argument('--beta_range', type=str, default='[0.001, 0.999]', help='beta range (default: [0.001, 0.999])')
    parser.add_argument('--beta_diff', type=str, default='True', help='learnable beta (default: True)')
    parser.add_argument('--bias_init', type=float, default=0, help='initial bias (default: 0)')
    parser.add_argument('--bias_diff', type=str, default='True', help='learnable bias (default: True)')

    # Training arguments
    parser.add_argument('--dataset', type=str, default='FashionMNIST', help='dataset name (default: FashionMNIST)')
    parser.add_argument('--dataset_cash', type=str, default='True', help='dataset cash (default: True)')
    parser.add_argument('--epoch', type=int, default=50, help='epoch count (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=-4, help='lr (default: -4)')
    parser.add_argument('--device', type=str, default='cuda', help='device (default: cuda)')
    parser.add_argument('--dtype', type=str, default='float', help='dtype (default: float)')

    args = parser.parse_args()

    # Instantiate the dataset
    if args.dataset == 'FashionMNIST':
        enc = TimeToFirstSpike(args.t_len, thr=0.2)
        dataset = StaticImageSpiking(os.path.join(args.path, 'data'), args.dataset, train=True, transform=enc, cash=ast.literal_eval(args.dataset_cash))
        n_in, n_out = 784, 10

    # Instantiate the model
    model = LinearSNN(args.layer_type, n_in, n_out, ast.literal_eval(args.n_per_hidden), args.t_len, beta_init=args.beta_init, beta_range=ast.literal_eval(args.beta_range), beta_diff=ast.literal_eval(args.beta_diff), bias_init=args.bias_init, bias_diff=ast.literal_eval(args.bias_diff))

    # Instantiate the trainer
    dtype = torch.float if args.dtype == 'float' else torch.half
    trainer = FastSNNTrainer(os.path.join(args.path, 'results'), model, dataset, args.epoch, args.batch_size, args.lr, device=args.device, dtype=dtype)
    trainer.train(save=True)


if __name__ == '__main__':
    main()
