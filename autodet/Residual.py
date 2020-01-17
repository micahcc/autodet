import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, ichannels, ochannels, kradius=1):
        super(Residual, self).__init__()
        self.kradius = kradius
        self.kwidth = self.kradius * 2 + 1

        assert ichannels == ochannels
        self.conv = nn.Conv2d(ichannels, ochannels, self.kwidth)
        self.conv_bn = nn.BatchNorm2d(ochannels)

    def forward(self, x):
        # compute difference
        dx = self.conv_bn(self.conv(x))
        cx = x[..., self.kradius: -self.kradius, self.kradius: -self.kradius]
        x = cx + dx
        return x
