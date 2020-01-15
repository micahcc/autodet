import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, target, predicted):
        _, _, tgtH, tgtW = target.shape
        _, _, prdH, prdW = predicted.shape

        dH = (tgtH - prdH) // 2
        dW = (tgtW - prdW) // 2

        cropped_target = target[:, :, dH:(tgtH-dH), dW:(tgtW-dW)]
        assert cropped_target.shape == predicted.shape
        return torch.mean((cropped_target - predicted) ** 2)
