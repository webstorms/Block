import torch.nn.functional as F


def bconv1d(x, weight, stride=1, dilation=1, padding=0):
    b, c, n, h = x.shape
    n, out_channels, in_channels, kernel_width_size = weight.shape

    out = x.view(b, c * n, h)
    weight = weight.view(n * out_channels, in_channels, kernel_width_size)

    out = F.conv1d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=n, padding=padding)

    return out.view(b, c, n, -1)
