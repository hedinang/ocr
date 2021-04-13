from torch import nn
import torch


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


class Detect(nn.Module):
    def __init__(self):  # detection layer
        super(Detect, self).__init__()

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                yv, xv = torch.meshgrid(
                    [torch.arange(ny), torch.arange(nx)])
                self.grid[i] = torch.stack((xv, yv), 2).view(
                    (1, 1, ny, nx, 2)).float().to(x[i].device)
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                           self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * \
                self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))
        return torch.cat(z, 1), x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x[0]
