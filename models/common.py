import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    def __init__(self):
        super(Focus, self).__init__()

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, x):
        return torch.cat(x, self.d)
