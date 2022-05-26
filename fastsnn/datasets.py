import os
import tables

import torch
import torchvision
from torch.distributions.poisson import Poisson
import numpy as np

from brainbox.datasets import BBDataset
from brainbox.datasets.transforms import BBTransform


# Transforms

class ImageToTimeOfFirstSpikeEncoding(BBTransform):

    def __init__(self, n_units, t_len, dt=1, tau_mem=20, thr=0.2, epsilon=1e-7):
        self._t_len = t_len
        self._tau_mem = tau_mem
        self._thr = thr
        self._epsilon = epsilon

        self._spike_builder = SpikeTensorBuilder(n_units, t_len, dt)

    def __call__(self, x):
        units, times = self._image_to_spikes_times(x)

        return self._spike_builder(units, times)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "tau_mem": self._tau_mem, "thr": self._thr}

    def _image_to_spikes_times(self, x):
        # Flatten the image into a single dimension
        x = x.view(-1)

        # Calculating the first spike times
        idx = x < self._thr  # Indices of the neurons that will not spike
        x = np.clip(x, self._thr + self._epsilon, 1e9)  # Avoid division by zero when x=self.thr
        T = self._tau_mem * np.log(x / (x - self._thr))  # Calculating the time of first spike
        T[idx] = self._t_len
        c = T < self._t_len  # Indices of the neurons that will spike

        # Generate COO data
        n_in = len(x)
        unit_numbers = torch.arange(n_in)
        units, times = unit_numbers[c], T[c],

        return units, times


class SpikeTensorBuilder(BBTransform):

    def __init__(self, n_units, t_len, dt):
        self._n_units = n_units
        self._t_len = t_len
        self._dt = dt

    def __call__(self, units, times):
        units = units % self._n_units
        times = torch.round(times * 1000. / self._dt).int()

        # Constrain spike length
        idxs = (times < self._t_len)
        units = units[idxs]
        times = times[idxs]

        # Build COO tensor
        indices = torch.stack([torch.Tensor(units.tolist()), torch.Tensor(times.tolist())], dim=0).long()
        shape = torch.Size([self._n_units, self._t_len, ])
        spikes = torch.FloatTensor(np.ones(len(indices[0])))

        return torch.sparse.FloatTensor(indices, spikes, shape).to_dense()


# Datasets

class SyntheticSpikes(BBDataset):

    def __init__(self, t_len, n_units, min_r, max_r, n_samples):
        super().__init__(None)
        self.t_len = t_len
        self.n_units = n_units
        self.min_r = min_r
        self.max_r = max_r
        self.n_samples = n_samples

    def __getitem__(self, i):
        rate = torch.FloatTensor(1).uniform_(self.min_r, self.max_r).item()
        x = self._create_spikes(rate, self.n_units, self.t_len)

        return x

    def __len__(self):
        return self.n_samples

    def _load_dataset(self, train):
        return None, None

    def _create_spikes(self, rate, n_units, t_len):
        pois_dis = Poisson(rate/t_len)
        if type(n_units) == tuple:
            samples = pois_dis.sample(sample_shape=(*n_units, t_len))
        else:
            samples = pois_dis.sample(sample_shape=(n_units, t_len))
        samples[samples > 1] = 1

        return samples


class BaseDataset(BBDataset):

    def __init__(self, root, train, preprocess, n_in, n_out, t_len, dt):
        super().__init__(root, train, preprocess, transform=None, target_transform=None, push_gpu=False)
        self._n_in = n_in
        self._n_out = n_out
        self._t_len = t_len
        self._dt = dt

    @property
    def hyperparams(self):
        return {**super().hyperparams, "t_len": self._t_len, "dt": self._dt}

    @property
    def n_in(self):
        return self._n_in

    @property
    def n_out(self):
        return self._n_out

    @property
    def t_len(self):
        return self._t_len


class StaticImageSpiking(BaseDataset):

    def __init__(self, root, train, n_in, n_out, t_len, dt, tau_mem=20, thr=0.2, epsilon=1e-7):
        super().__init__(root, train, lambda dataset: StaticImageSpiking.preprocess(dataset, n_in, t_len, dt, tau_mem, thr, epsilon), n_in, n_out, t_len, dt)

    @staticmethod
    def preprocess(dataset, n_units, t_len, dt, tau_mem, thr, epsilon):
        img_to_spikes = ImageToTimeOfFirstSpikeEncoding(n_units, t_len, dt, tau_mem, thr, epsilon)
        processed_dataset = []

        for i in range(dataset.data.shape[0]):
            processed_dataset.append(img_to_spikes(dataset.data[i]))

        return torch.stack(processed_dataset).pin_memory()


class FMNISTDataset(StaticImageSpiking):

    def __init__(self, root, train=True, tau_mem=20, thr=0.2, epsilon=1e-7):
        super().__init__(root, train, n_in=28*28, n_out=10, t_len=100, dt=1, tau_mem=tau_mem, thr=thr, epsilon=epsilon)

    def _load_dataset(self, train):
        dataset = torchvision.datasets.FashionMNIST(self._root, train=train, transform=torchvision.transforms.ToTensor(), download=True)

        return dataset.data, dataset.targets


class H5Dataset(BaseDataset):

    def __init__(self, root, train, n_in, n_out, t_len, train_name, test_name, dt):
        self._file = None
        self._train_name = train_name
        self._test_name = test_name
        super().__init__(root, train, lambda dataset: H5Dataset.preprocess(dataset, n_in, t_len, dt), n_in, n_out, t_len, dt)
        self._file.close()

    @staticmethod
    def preprocess(dataset, n_units, t_len, dt):
        to_spikes = SpikeTensorBuilder(n_units, t_len, dt)
        processed_dataset = []

        units, times = dataset

        for i in range(len(units)):
            item_units = torch.Tensor(np.array(units[i], dtype=np.int))
            item_times = torch.Tensor(np.array(times[i], dtype=np.float))

            processed_dataset.append(to_spikes(item_units, item_times))
            if i % 50 == 0:
                print(f"{i}/{len(units)}")

        return processed_dataset

    @staticmethod
    def _open_file(hdf5_file_path):
        fileh = tables.open_file(hdf5_file_path, mode="r")
        units = fileh.root.spikes.units
        times = fileh.root.spikes.times
        labels = fileh.root.labels

        return fileh, units, times, labels

    def _load_dataset(self, train):
        name = self._train_name if train else self._test_name
        fileh, units, times, labels = H5Dataset._open_file(os.path.join(self._root, name))
        targets = torch.Tensor(labels)
        self._file = fileh

        return (units, times), targets


class NMNISTDataset(H5Dataset):

    def __init__(self, root, train=True, dt=1):
        super().__init__(root, train, n_in=1156, n_out=10, t_len=300, train_name="train.h5", test_name="test.h5", dt=dt)


class SHDDataset(H5Dataset):

    def __init__(self, root, train=True, dt=2):
        super().__init__(root, train, n_in=700, n_out=20, t_len=500, train_name="shd_train.h5", test_name="shd_test.h5", dt=dt)