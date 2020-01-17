import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_to_match(target, predicted):
    _, _, tgtH, tgtW = target.shape
    _, _, prdH, prdW = predicted.shape

    dH = (tgtH - prdH) // 2
    dW = (tgtW - prdW) // 2

    # crop target down
    if dH < 0 and dW < 0:
        dH, dW = abs(dH), abs(dW)
        cropped_target = target
        cropped_predicted = predicted[:, :, dH:(prdH-dH), dW:(prdW-dW)]
    else:
        assert dH >= 0 and dW >= 0, 'mixed cropping not supported'
        cropped_target = target[:, :, dH:(tgtH-dH), dW:(tgtW-dW)]
        cropped_predicted = predicted

    assert cropped_target.shape == cropped_predicted.shape
    return cropped_target, cropped_predicted


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, target, predicted, real_criticism, gen_criticism):
        target, predicted = crop_to_match(target, predicted)
        reconstruction_loss_img = torch.sum(
            (target - predicted) ** 2, dim=1, keepdim=True)

        gen_criticism, real_criticism = \
            crop_to_match(gen_criticism, real_criticism)
        critic_loss_img = real_criticism - gen_criticism
        gen_loss_img = gen_criticism

        gen_loss = torch.mean(gen_loss_img)
        critic_loss = torch.mean(critic_loss_img)
        reconstruction_loss = torch.mean(reconstruction_loss_img)
        total_loss = gen_loss + critic_loss + reconstruction_loss

        return {
            'total_loss': total_loss,
            'gen_loss': gen_loss,
            'critic_loss': critic_loss,
            'reconstruction_loss': reconstruction_loss,
            'gen_loss_image': gen_loss_img,
            'critic_loss_image': critic_loss_img,
            'reconstruction_loss_image': reconstruction_loss_img,
        }
