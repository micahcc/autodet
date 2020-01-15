#!/usr/bin/python3
import argparse
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils import data

import autodet


from autodet.UnlabeledDirectoryDataset import UnlabeledDirectoryDataset
from autodet.Encoder import Encoder
from autodet.Decoder import Decoder
from autodet.Loss import Loss


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--in-file', required=True,
                        help='input file ')
    parser.add_argument('-c', '--checkpoint-file', required=True,
                        help='Checkpoint file for model')
    parser.add_argument('-o', '--out-file',
                        help='output file, or if none given display on the screen')

    args = parser.parse_args()

    raw = Image.open(args.in_file)
    orig = np.array(raw) / 255.0
    features = np.transpose(orig, [2, 0, 1])

    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    encoder = Encoder(ichannels=3, ochannels=20)
    decoder = Decoder(ichannels=20, ochannels=3)

    checkpoint = torch.load(args.checkpoint_file)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # need to eval once to update the batch norm params
    encoder.eval()
    decoder.eval()

    # add batch dimension as first dimension
    encoded = encoder(torch.Tensor([features]))
    decoded = decoder(encoded)

    # convert decoded back to a 3D numpy array with HWC
    decoded = decoded.detach().numpy()
    decoded = decoded[0, ...]  # drop batch
    decoded = np.transpose(decoded, [1, 2, 0])  # swap C to last dimension

    fig, (ax1, ax2) = plt.subplots(2)
    imgplot = ax1.imshow(orig)
    imgplot = ax2.imshow(decoded)
    plt.show()
    return 0


if __name__ == '__main__':
    main()
