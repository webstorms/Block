import os
import ast
import argparse
from pathlib import Path

import torch
import numpy as np

from dblock import datasets, models, trainer
from dblock.datasets.transforms import List


def get_dataset(base_path, args):
    flatten = not ast.literal_eval(args.load_spatial_dims)
    use_augmentation = ast.literal_eval(args.use_augmentation)
    print(f"flatten = {flatten} {type(flatten)}")
    print(f"use_augmentation = {use_augmentation} {type(use_augmentation)}")

    if args.dataset == "yinyang":
        dataset = datasets.YinYangDataset(size=20000, t_len=args.t_len, transform=List.get_yingyang_transform(args.t_len))

    elif args.dataset == "mnist":
        train_transform = List.get_mnist_transform(args.t_len, flatten, use_augmentation)
        dataset = datasets.MNISTDataset(os.path.join(base_path, "data"), t_len=args.t_len, transform=train_transform)

    elif args.dataset == "fmnist":
        train_transform = List.get_fmnist_transform(args.t_len, flatten, use_augmentation)
        dataset = datasets.FMNISTDataset(os.path.join(base_path, "data"), t_len=args.t_len, transform=train_transform)

    elif args.dataset == "nmnist":
        transform = List.get_nmnist_transform(args.t_len)
        dataset = datasets.NMNISTDataset(os.path.join(base_path, "data", "N-MNIST"), dt=1, transform=transform)

    elif args.dataset == "shd":
        transform = List.get_shd_transform(args.t_len)
        dataset = datasets.SHDDataset(os.path.join(base_path, "data", "SHD"), dt=2, transform=transform)

    elif args.dataset == "ssc":
        transform = List.get_ssc_transform(args.t_len)
        dataset = datasets.SSCDataset(os.path.join(base_path, "data", "SSC"), dt=2, transform=transform)

    elif args.dataset == "cifar10":
        transform = List.get_cifar10_transform(args.t_len, use_augmentation=use_augmentation)
        dataset = datasets.CIFAR10Dataset(os.path.join(base_path, "data"), t_len=args.t_len, transform=transform)

    return dataset


def get_model(t_len, args):
    load_conv_model = ast.literal_eval(args.load_spatial_dims)
    single_spike = ast.literal_eval(args.single_spike)
    beta_requires_grad = ast.literal_eval(args.beta_requires_grad)
    readout_max = ast.literal_eval(args.readout_max)
    print(f"single_spike = {single_spike} {type(single_spike)}")
    print(f"load_conv_model = {load_conv_model} {type(load_conv_model)}")

    milestones = [-1]

    if args.dataset == "yinyang":
        model = models.YingYangModel(args.method, t_len, single_spike=single_spike)
        c = 4
        n_in = 4
        model._model._layers[0].init_weight(model._model._layers[0]._to_current.weight, "uniform", a=-c*np.sqrt(1 / n_in), b=c*np.sqrt(1 / n_in))
        milestones = [50, 100]
    elif args.dataset == "mnist":
        if load_conv_model:
            model = models.ConvMNINSTModel(args.method, t_len, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
            milestones = [30, 60, 90]
        else:
            model = models.LinearMNINSTModel(args.method, t_len, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
            milestones = [15, 90, 120]
    elif args.dataset == "fmnist":
        if load_conv_model:
            model = models.ConvFMNINSTModel(args.method, t_len, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
            milestones = [30, 60, 90]
        else:
            model = models.LinearFMNINSTModel(args.method, t_len, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
            milestones = [15, 90, 120]
    elif args.dataset == "nmnist":
        model = models.NMNISTModel(args.method, t_len, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
        milestones = [30, 60, 90]
    elif args.dataset == "shd":
        model = models.SHDModel(args.method, t_len, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
        milestones = [30, 60, 90]
    elif args.dataset == "cifar10":
        model = models.CIFAR10Model(args.method, t_len, heterogeneous_beta=False, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
        milestones = [50, 100, 120]

    return model, milestones


def main():
    torch.backends.cudnn.benchmark = True

    # Building settings
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--path", type=str, default="block")
    parser.add_argument("--method", type=str)
    parser.add_argument("--t_len", type=int)
    parser.add_argument("--beta_requires_grad", type=str)
    parser.add_argument("--readout_max", type=str)
    parser.add_argument("--single_spike", type=str)

    # Training arguments
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--load_spatial_dims", type=str)
    parser.add_argument("--use_augmentation", type=str)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--track_activity", type=str, default="False")

    # Load arguments
    args = parser.parse_args()
    base_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
    print(base_path)
    # Instantiate the dataset
    print("Building dataset...")
    dataset = get_dataset(base_path, args)

    # Instantiate the model
    print("Building model...")
    model, milestones = get_model(args.t_len, args)

    # Instantiate the trainer
    print("Started training...")
    track_activity = ast.literal_eval(args.track_activity)

    model_results_path = os.path.join(base_path, f"results/datasets/{args.dataset}" if not track_activity else f"results/datasets/robustness/{args.dataset}")

    # Ensure that activity starts at 0
    if track_activity:
        c = 0.1
        n_in = 28*28 if args.dataset == "mnist" else dataset.n_in
        model._model._layers[0].init_weight(model._model._layers[0]._to_current.weight, "uniform", a=-c*np.sqrt(1 / n_in), b=c*np.sqrt(1 / n_in))

    transform = List.get_cifar10_transform(args.t_len, use_augmentation=True)
    val_dataset = datasets.CIFAR10Dataset(os.path.join(base_path, "data"), train=False, t_len=args.t_len, transform=transform)
    snn_trainer = trainer.Trainer(model_results_path, model, dataset, args.epoch, args.batch, args.lr, milestones=milestones, gamma=args.gamma, val_dataset=val_dataset, device=args.device, track_activity=track_activity)
    snn_trainer.train(save=True)


if __name__ == "__main__":
    main()
