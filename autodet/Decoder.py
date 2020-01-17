import torch.nn as nn
import torch.nn.functional as F

from autodet.Residual import Residual


class Decoder(nn.Module):
    def __init__(self, ichannels, ochannels):
        super(Decoder, self).__init__()
        self.layer0 = nn.Conv2d(ichannels, 20, 1)
        self.layer1 = Residual(ichannels=20, ochannels=20, kradius=1)
        self.layer2 = nn.Conv2d(20, ochannels, 1)

    def forward(self, x):
        x = self.layer0(x)
        x = F.softsign(x)
        x = self.layer1(x)
        x = F.softsign(x)
        x = self.layer2(x)
        x = F.sigmoid(x)
        return x
