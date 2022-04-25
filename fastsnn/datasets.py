import torch
from torch.distributions.poisson import Poisson
import torchvision
import numpy as np

from brainbox.datasets import BBDataset
from brainbox.datasets.transforms import BBTransform


class TimeToFirstSpike(BBTransform):

    def __init__(self, t_len, tau_mem=20, thr=0.2, epsilon=1e-7):
        self.t_len = t_len
        self.tau_mem = tau_mem
        self.thr = thr
        self.epsilon = epsilon

    def __call__(self, x):
        indices, shape = self._get_coo(x)
        spikes = self._coo_to_spikes(indices, shape)

        return spikes

    def _get_coo(self, x):
        # Flatten the image into a single dimension
        x = x.view(-1)

        # Calculating the first spike times
        idx = x < self.thr  # Indices of the neurons that will not spike
        x = np.clip(x, self.thr + self.epsilon, 1e9)  # Avoid division by zero when x=self.thr
        T = self.tau_mem * np.log(x / (x - self.thr))  # Calculating the time of first spike
        T[idx] = self.t_len
        c = T < self.t_len  # Indices of the neurons that will spike

        # Building the COO
        n_in = len(x)
        unit_numbers = torch.arange(n_in)
        times, units = T[c], unit_numbers[c]

        indices = torch.stack([units, times], dim=0).long()
        shape = torch.Size([n_in, self.t_len, ])

        return indices, shape

    def _coo_to_spikes(self, indices, shape):
        spikes = torch.FloatTensor(np.ones(len(indices[0])))

        return torch.sparse.FloatTensor(indices, spikes, shape).to_dense()

    @property
    def hyperparams(self):
        return {**super().hyperparams, 't_len': self.t_len, 'tau_mem': self.tau_mem, 'thr': self.thr}


class StaticImageSpiking(BBDataset):

    def __init__(self, root, name, train, transform, cash=True):
        super().__init__(transform, None)
        self.name = name
        self.train = train
        self.cash = cash

        if self.name == 'FashionMNIST':
            self.dataset = torchvision.datasets.FashionMNIST(root, train=train, transform=torchvision.transforms.ToTensor(), download=True)

        if self.cash:
            cash_dataset = []
            for i in range(len(self)):
                cash_dataset.append(self.transform(self.dataset[i][0]))
            self.cash_dataset = torch.stack(cash_dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.cash:
            x = self.cash_dataset[i]
        elif self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.dataset)

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'name': self.name, 'train': self.train}


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
        pass

    def _create_spikes(self, rate, n_units, t_len):
        pois_dis = Poisson(rate/t_len)
        if type(n_units) == tuple:
            samples = pois_dis.sample(sample_shape=(*n_units, t_len))
        else:
            samples = pois_dis.sample(sample_shape=(n_units, t_len))
        samples[samples > 1] = 1

        return samples

    # def _create_spikes(self, rates, T):
    #
    #     def create_spike_vector(rate, T):
    #         element_rate = rate/T
    #         if element_rate > 0.5:
    #             raise ValueError('element_rate {0} too large.'.format(element_rate))
    #
    #         pois_dis = Poisson(element_rate)
    #         samples = pois_dis.sample(sample_shape=(1, T))
    #         samples[samples > 1] = 1
    #
    #         return samples[0]
    #
    #     N = len(rates)
    #     spike_matrix = torch.zeros([N, T], dtype=torch.float32)
    #     spike_matrix.requires_grad = False
    #
    #     for i in range(N):
    #         spike_matrix[i] = create_spike_vector(rates[i], T)
    #
    #     return spike_matrix