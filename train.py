#!/usr/bin/python3
import argparse
import os
import sys

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
    parser.add_argument('-i', '--idir', required=True,
                        help='input file directory')
    parser.add_argument('-e', '--epochs', default=1, type=int,
                        help='input file directory')
    parser.add_argument('-N', '--batch-size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('-d', '--model-dir', required=True,
                        help='Model dir (save and restore from here)')

    args = parser.parse_args()
    dataset_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 6,
    }

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.isdir(args.model_dir):
        print("{} exists and isn't a directory!".format(args.model_dir))
        return -1

    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    training_set = UnlabeledDirectoryDataset(args.idir)
    training_gen = data.DataLoader(training_set, **dataset_params)
    encoder = Encoder(ichannels=3, ochannels=20)
    decoder = Decoder(ichannels=20, ochannels=3)
    losser = Loss()

    # construct an optimizer
    params = [p for p in encoder.parameters() if p.requires_grad] + \
        [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer.zero_grad()

    encoder.to(device)
    decoder.to(device)
    losser.to(device)

    #fig = plt.figure()
    for epoch in range(args.epochs):
        for sample in training_gen:

            # "batch" by adding dim 0
            img = (sample['image'] / 255.0).to(device)

            encoded = encoder(img)
            decoded = decoder(encoded)

            # loss
            loss = losser(target=img, predicted=decoded)
            loss.backward()

            optimizer.step()
            print(loss)

        import ipdb
        ipdb.set_trace()
        # at the end of each epoch save
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losser_state_dict': losser.state_dict(),
        },
            os.path.join(args.model_dir, 'checkpoint.tch'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
