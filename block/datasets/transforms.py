import torch
import torchvision
import numpy as np
import brainbox
from brainbox.datasets.transforms import BBTransform


class SpikeTensorBuilder(BBTransform):

    def __init__(self, n_units, t_len, dt):
        self._n_units = n_units
        self._t_len = t_len
        self._dt = dt

    def __call__(self, args):
        units, times = args[0], args[1]
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


class JitterSpikeTimes(BBTransform):

    def __init__(self, p=0.5, max_std=0.006):
        self._p = p
        self._max_std = max_std

    def __call__(self, args):
        units, times = args[0], args[1]

        new_times = times.clone()
        if torch.rand(1) > self._p:
            std = torch.rand(1) * self._max_std
            new_times += torch.torch.normal(0, std * torch.ones_like(times))
            new_times[new_times < 0] = 10000

        return units, new_times

    @property
    def hyperparams(self):
        return {**super().hyperparams, "p": self._p, "max_std": self._max_std}


class ElongateSpikeTimes(BBTransform):

    def __init__(self, p=0.5, min_m=0.9, max_m=1.1):
        self._p = p
        self._min_m = min_m
        self._max_m = max_m

    def __call__(self, args):
        units, times = args[0], args[1]

        new_times = times.clone()
        if torch.rand(1) > self._p:
            m = ((torch.rand(1) * (self._max_m - self._min_m)) + self._min_m)
            new_times *= m
            new_times[new_times < 0] = 10000

        return units, new_times

    @property
    def hyperparams(self):
        return {**super().hyperparams, "p": self._p, "min_m": self._min_m, "max_m": self._max_m}


class AddGaussianNoise(BBTransform):

    # Adds gaussian noise to image tensor

    def __init__(self, p=0.5, max_std=0.05):
        self._p = p
        self._max_std = max_std

    def __call__(self, img):
        if torch.rand(1) > self._p:
            std = torch.rand(1) * self._max_std
            return torch.clamp((img + torch.randn(img.size()) * std), min=0, max=1)

        return img

    @property
    def hyperparams(self):
        return {**super().hyperparams, "p": self._p, "max_std": self._max_std}


class SingleSpike2DEncoding(BBTransform):

    # Takes image tensor and converts it to a spike tensor of sine spike encoding

    def __init__(self, channel, dim, t_len, max_c):
        self._channel = channel
        self._dim = dim
        self._t_len = t_len
        self._max_c = max_c

        k = dim ** 2
        self._c = torch.Tensor([[[i] * k] for i in range(self._channel)]).long().flatten()
        self._h = torch.Tensor([self._channel * [[i] * dim for i in range(dim)]]).long().flatten()
        self._w = torch.Tensor([self._channel * [[i for i in range(dim)] for _ in range(dim)]]).long().flatten()

    def __call__(self, x, thresh=0.2):
        times = self._image_to_spikes_times(x)
        indices = torch.stack([self._c, times, self._h, self._w], dim=0).long()
        shape = torch.Size([self._channel, self._t_len, self._dim, self._dim])
        spikes = torch.ones(len(indices[0]))

        assert (times >= 0).all()
        assert (spikes > 0).all()

        spike_tensor = torch.sparse.FloatTensor(indices, spikes, shape).to_dense()
        return spike_tensor
        # thresh_mask = x > thresh
        # return torch.einsum("cthw, chw->cthw", spike_tensor, thresh_mask)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "max_c": self._max_c}

    def _image_to_spikes_times(self, x):
        # Flatten the image into a single dimension
        times = ((self._max_c - x) / self._max_c) * (self._t_len - 1)

        return times.flatten()


class SingleSpike1DEncoding(BBTransform):

    # Takes 1D tensor and converts it to a spike tensor of sine spike encoding

    def __init__(self, dim, t_len, max_c):
        self._dim = dim
        self._t_len = t_len
        self._max_c = max_c

        self._d = torch.arange(dim).long()

    def __call__(self, x):
        times = self._to_spike_times(x)
        indices = torch.stack([self._d, times], dim=0).long()
        shape = torch.Size([self._dim, self._t_len])
        spikes = torch.FloatTensor(np.ones(len(indices[0])))

        return torch.sparse.FloatTensor(indices, spikes, shape).to_dense()

    @property
    def hyperparams(self):
        return {**super().hyperparams, "max_c": self._max_c}

    def _to_spike_times(self, x):
        times = ((self._max_c - x) / self._max_c) * (self._t_len - 1)

        return times.flatten()


class RandomCrop(BBTransform):

    # Randomly crop spatial dims from tensor

    def __init__(self, size, padding):
        self._size = size
        self._padding = padding

        self._crop = torchvision.transforms.RandomCrop(self._size, self._padding)

    def __call__(self, img):
        return self._crop(img)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "size": self._size, "padding": self._padding}


class RandomPerspective(BBTransform):

    # Randomly crop spatial dims from tensor

    def __init__(self, distortion_scale, p):
        self._distortion_scale = distortion_scale
        self._p = p

        self._perspective = torchvision.transforms.RandomPerspective(distortion_scale, p)

    def __call__(self, img):
        return self._perspective(img)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "p": self._p, "distortion_scale": self._distortion_scale}


class RandomHorizontalFlip(BBTransform):

    # Randomly flip img tensor horizontally

    def __init__(self, p=0.5):
        self._p = p

        self._flip = torchvision.transforms.RandomHorizontalFlip(p)

    def __call__(self, img):
        return self._flip(img)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "p": self._p}


class ToClip(BBTransform):

    def __init__(self, repeats):
        self._repeats = repeats

    def __call__(self, img):
        # img: channel x height x width
        # output: channel x time x height x width
        assert len(img.shape) == 3
        channel = img.shape[0]
        height = img.shape[1]
        width = img.shape[2]

        clip = torch.zeros(
            (
                channel,
                self._repeats,
                height,
                width,
            )
        )
        clip[:] = img.unsqueeze(1)

        return clip

    @property
    def hyperparams(self):
        hyperparams = {
            **super().hyperparams,
            "repeats": self._repeats,
        }

        return hyperparams


class Normalize(BBTransform):

    def __init__(self, mean, std):
        self._mean = mean
        self._std = std
        self._transform = torchvision.transforms.Normalize(self._mean, self._std)

    def __call__(self, img):
        return self._transform(img)

    @property
    def hyperparams(self):
        hyperparams = {
            **super().hyperparams,
            "mean": self._mean,
            "std": self._std,
        }

        return hyperparams


class Flatten(BBTransform):

    # Randomly crop spatial dims from tensor

    def __init__(self):
        pass

    def __call__(self, img):
        return img.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)


class List:

    @staticmethod
    def get_yingyang_transform(t_len):
        return SingleSpike1DEncoding(dim=4, t_len=t_len, max_c=1)

    @staticmethod
    def get_mnist_transform(t_len, flatten=False, use_augmentation=False):
        if use_augmentation:
            transform_list = [
                RandomPerspective(0.4, 0.7),
                RandomCrop(28, 3),
                AddGaussianNoise(),
                SingleSpike2DEncoding(channel=1, dim=28, t_len=t_len, max_c=1)
            ]
        else:
            transform_list = [SingleSpike2DEncoding(channel=1, dim=28, t_len=t_len, max_c=1)]

        if flatten:
            transform_list.append(Flatten())

        return brainbox.datasets.transforms.Compose(transform_list)

    @staticmethod
    def get_fmnist_transform(t_len, flatten=False, use_augmentation=False):
        if use_augmentation:
            transform_list = [
                RandomPerspective(0.1, 0.3),
                RandomHorizontalFlip(),
                RandomCrop(28, 1),
                AddGaussianNoise(),
                SingleSpike2DEncoding(channel=1, dim=28, t_len=t_len, max_c=1)
            ]
        else:
            transform_list = [SingleSpike2DEncoding(channel=1, dim=28, t_len=t_len, max_c=1)]

        if flatten:
            transform_list.append(Flatten())

        return brainbox.datasets.transforms.Compose(transform_list)

    @staticmethod
    def get_nmnist_transform(t_len, use_augmentation=False):
        if use_augmentation:
            raise NotImplementedError
        else:
            transform_list = [SpikeTensorBuilder(n_units=1156, t_len=t_len, dt=1)]

        return brainbox.datasets.transforms.Compose(transform_list)

    @staticmethod
    def get_shd_transform(t_len, use_augmentation=False):
        if use_augmentation:
            raise NotImplementedError
            # transform_list = [
            #     # TODO: Add random shift (vertical and horizontal)
            #     JitterSpikeTimes(),
            #     ElongateSpikeTimes(),
            #     SpikeTensorBuilder(n_units=700, t_len=500, dt=2)
            # ]
        else:
            transform_list = [SpikeTensorBuilder(n_units=700, t_len=t_len, dt=2)]

        return brainbox.datasets.transforms.Compose(transform_list)

    @staticmethod
    def get_ssc_transform(t_len, use_augmentation=False):
        if use_augmentation:
            raise NotImplementedError
        else:
            transform_list = [SpikeTensorBuilder(n_units=700, t_len=t_len, dt=2)]

        return brainbox.datasets.transforms.Compose(transform_list)

    @staticmethod
    def get_cifar10_transform(t_len, use_augmentation=False):
        if use_augmentation:
            # transform_list = brainbox.datasets.transforms.Compose(
            #     [
            #         RandomCrop(32, 4),
            #         RandomHorizontalFlip(),
            #         SingleSpike2DEncoding(channel=3, dim=32, t_len=t_len, max_c=1)
            #     ]
            # )
            transform_list = [
                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                RandomHorizontalFlip(),
                RandomCrop(32, 4),
                ToClip(t_len)
            ]
        else:
            raise NotImplementedError

        return brainbox.datasets.transforms.Compose(transform_list)
