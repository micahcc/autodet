import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, ichannels, ochannels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(ichannels, 20, 3)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, ochannels, 3)
        self.conv2_bn = nn.BatchNorm2d(ochannels)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        return F.relu(self.conv2_bn(self.conv2(x)))
