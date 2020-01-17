import numpy as np
from torch.utils.tensorboard import SummaryWriter


def Summarize(
        outdir,
        step,
        **kwargs):
    writer = SummaryWriter(outdir)

    cast_kwargs = {}
    for k, v in kwargs.items():
        v = v.cpu().detach().numpy()

        if v.size == 1:
            writer.add_scalar(k, v, step)
        elif v.shape[1] != 3 and v.shape[1] != 1:
            # v = np.transpose(v, [0, 2, 3, 1])
            #writer.add_images(k, v, step)
            # TODO split into separate channels
            pass
        else:
            # v = np.transpose(v, [0, 2, 3, 1])
            writer.add_images(k, v, step)
