import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, ichannels, ochannels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(ichannels, 20, 3)
        self.conv2 = nn.Conv2d(20, ochannels, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
