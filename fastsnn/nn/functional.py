import torch
import torch.nn.functional as F


def bconv1d(x, weight, stride=1, dilation=1, padding=0):
    b, c, n, h = x.shape
    n, out_channels, in_channels, kernel_width_size = weight.shape

    out = x.view(b, c * n, h)
    weight = weight.view(n * out_channels, in_channels, kernel_width_size)

    out = F.conv1d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=n, padding=padding)

    return out.view(b, c, n, -1)


def cat(tensor_list):
    n = len(tensor_list[0])
    cat_tensors = [torch.cat([tensors[i] for tensors in tensor_list], dim=2) for i in range(n)]

    return cat_tensors[0] if n == 1 else cat_tensors
