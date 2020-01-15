#!/usr/bin/python3
import argparse
import os

import matplotlib.pyplot as plt
import torch

import autodet

from autodet.UnlabeledDirectoryDataset import UnlabeledDirectoryDataset
from autodet.Encoder import Encoder
from autodet.Decoder import Decoder
from autodet.Loss import Loss


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--idir', required=True,
                        help='Input file directory')

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = UnlabeledDirectoryDataset(args.idir)
    encoder = Encoder(ichannels=3, ochannels=20)
    decoder = Decoder(ichannels=20, ochannels=3)
    losser = Loss()

    # construct an optimizer
    params = [p for p in encoder.parameters() if p.requires_grad] + \
        [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.01)
    optimizer.zero_grad()

    encoder.to(device)
    decoder.to(device)
    losser.to(device)

    #fig = plt.figure()
    for i in range(len(dataset)):
        sample = dataset[i]

        # "batch" by adding dim 0
        img = torch.tensor([sample['image']],
                           dtype=torch.float32, device=device)

        encoded = encoder(img)
        decoded = decoder(encoded)

        # loss
        loss = losser(target=img, predicted=decoded)
        loss.backward()

        optimizer.step()

        print(loss)

        #plt.imshow(sample['image'])
        #plt.show()


if __name__ == '__main__':
    main()
