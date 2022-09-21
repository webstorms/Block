import torch
import torchvision
from brainbox.datasets import BBDataset


class StaticImageSpiking(BBDataset):

    def __init__(self, root, train, channel, n_out, t_len, transform):
        super().__init__(root, train, preprocess=None, transform=transform, target_transform=None, push_gpu=False)
        self._channel = channel
        self._n_out = n_out
        self._t_len = t_len
        self._transform = transform

    @property
    def hyperparams(self):
        return {**super().hyperparams, "channel": self._channel, "t_len": self._t_len}

    @property
    def channel(self):
        return self._channel

    @property
    def n_out(self):
        return self._n_out

    @property
    def t_len(self):
        return self._t_len

    @staticmethod
    def build_dataset(dataset):
        new_dataset = []
        for i in range(len(dataset)):
            new_dataset.append(dataset[i][0])

        return torch.stack(new_dataset)


class MNISTDataset(StaticImageSpiking):
    
    def __init__(self, root, train=True, t_len=64, transform=None):
        super().__init__(root, train, channel=1, n_out=10, t_len=t_len, transform=transform)

    def _load_dataset(self, train):
        dataset = torchvision.datasets.MNIST(self._root, train=train, transform=torchvision.transforms.ToTensor(), download=True)

        return StaticImageSpiking.build_dataset(dataset), dataset.targets


class FMNISTDataset(StaticImageSpiking):

    def __init__(self, root, train=True, t_len=64, transform=None):
        super().__init__(root, train, channel=1, n_out=10, t_len=t_len, transform=transform)

    def _load_dataset(self, train):
        dataset = torchvision.datasets.FashionMNIST(self._root, train=train, transform=torchvision.transforms.ToTensor(), download=True)

        return StaticImageSpiking.build_dataset(dataset), dataset.targets