import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, ichannels, ochannels, kradius=1, project_activation=F.softsign):
        super(Residual, self).__init__()
        self.kradius = kradius
        self.kwidth = self.kradius * 2 + 1

        if ichannels != ochannels:
            self.conv_project = nn.Conv2d(ichannels, ochannels, 1)
            self.conv_project_bn = nn.BatchNorm2d(ochannels)
            self.project_activation = project_activation
        else:
            self.conv_project = None
            self.conv_project_bn = None
            self.project_activation = None

        self.conv = nn.Conv2d(ochannels, ochannels, self.kwidth)
        self.conv_bn = nn.BatchNorm2d(ochannels)

    def forward(self, x):
        # project into a dimension to match input and output channels
        if self.conv_project and self.conv_project_bn and self.project_activation:
            x = self.conv_project(x)
            x = self.conv_project_bn(x)
            x = self.project_activation(x)

        # compute difference, then crop input down the convolved version
        dx = self.conv_bn(self.conv(x))
        cx = x[..., self.kradius: -self.kradius, self.kradius: -self.kradius]
        x = cx + dx
        return x
