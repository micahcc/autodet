#!/usr/bin/python3
import argparse
import os
import sys

import torch
from torch.utils import data

from autodet.UnlabeledDirectoryDataset import UnlabeledDirectoryDataset
from autodet.Encoder import Encoder
from autodet.Decoder import Decoder
from autodet.Loss import Loss


def LoadCheckpoint(dirname):
    names = os.listdir(dirname)
    if not names:
        return None

    names = filter(lambda x: x.startswith('checkpoint-')
                   and x.endswith('.tch'), names)
    f = sorted(names)[-1]
    return torch.load(os.path.join(dirname, f))


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

    step = 0
    epoch = 0
    encoder = Encoder(ichannels=3, ochannels=20)
    decoder = Decoder(ichannels=20, ochannels=3)
    losser = Loss()

    # construct an optimizer
    params = [p for p in encoder.parameters() if p.requires_grad] + \
        [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer.zero_grad()

    # need to move to device BEFORE loading state so that everything gets loaded
    # onto the correct device automatically
    encoder.to(device)
    decoder.to(device)
    losser.to(device)

    ckpt = LoadCheckpoint(args.model_dir)
    if ckpt:
        epoch = ckpt['epoch']
        step = ckpt['step']
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        losser.load_state_dict(ckpt['losser_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    while epoch < args.epochs:
        for sample in training_gen:

            # "batch" by adding dim 0
            img = (sample['image'] / 255.0).to(device)

            encoded = encoder(img)
            decoded = decoder(encoded)

            # loss
            loss = losser(target=img, predicted=decoded)
            loss.backward()

            optimizer.step()
            step += args.batch_size
            print(loss)
        epoch += 1

        # at the end of each epoch save
        save_dict = {
            'epoch': epoch,
            'step': step,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losser_state_dict': losser.state_dict(),
        }
        save_file = os.path.join(
            args.model_dir, 'checkpoint-e{:04}-s{:04}.tch'.format(epoch, step))
        torch.save(save_dict, save_file)

    print("Completed {} epochs ({} steps)".format(epoch, step))
    return 0


if __name__ == '__main__':
    sys.exit(main())
