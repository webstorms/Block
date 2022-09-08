import os
import tables

import torch
import numpy as np
from brainbox.datasets import BBDataset


class H5Dataset(BBDataset):

    def __init__(self, root, train, n_in, n_out, t_len, train_name, test_name, dt, transform=None):
        self._file = None
        self._train_name = train_name
        self._test_name = test_name
        super().__init__(root, train, lambda dataset: H5Dataset.preprocess(dataset), transform)
        self._file.close()

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

    @staticmethod
    def preprocess(dataset):
        processed_dataset = []
        units, times = dataset

        for i in range(len(units)):
            item_units = torch.Tensor(np.array(units[i], dtype=np.int))
            item_times = torch.Tensor(np.array(times[i], dtype=np.float))
            processed_dataset.append((item_units, item_times))

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

    def __init__(self, root, train=True, dt=1, transform=None):
        super().__init__(root, train, n_in=1156, n_out=10, t_len=400, train_name="train.h5", test_name="test.h5", dt=dt, transform=transform)


class SHDDataset(H5Dataset):

    def __init__(self, root, train=True, dt=2, transform=None):
        super().__init__(root, train, n_in=700, n_out=20, t_len=300, train_name="shd_train.h5", test_name="shd_test.h5", dt=dt, transform=transform)
