#!/usr/bin/python3
import argparse
import os
import sys

import torch
from torch.utils import data
import torchvision

from autodet.UnlabeledDirectoryDataset import UnlabeledDirectoryDataset
from autodet.Encoder import Encoder
from autodet.Decoder import Decoder
from autodet.Critic import Critic
from autodet.Loss import Loss
from autodet.summary import Summarize


def LoadCheckpoint(dirname):
    names = os.listdir(dirname)
    names = list(filter(lambda x: x.startswith('checkpoint-')
                        and x.endswith('.tch'), names))

    if not names:
        return None

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
    parser.add_argument('-s', '--shape', default=512,
                        type=int, help='image shape')
    parser.add_argument('-S', '--summarize-every-step', default=100,
                        type=int, help='Summarize every this number of steps')

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

    def transform(image):
        crop = torchvision.transforms.RandomResizedCrop(args.shape)
        return crop(image)

    training_set = UnlabeledDirectoryDataset(args.idir, transform=transform)
    training_gen = data.DataLoader(training_set, **dataset_params)

    step = 0
    epoch = 0
    encoder = Encoder(ichannels=3, ochannels=20)
    decoder = Decoder(ichannels=20, ochannels=3)
    critic = Critic(ichannels=3, ochannels=1)
    losser = Loss()

    # construct an optimizer
    params = [p for p in encoder.parameters() if p.requires_grad] + \
        [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.00001)
    optimizer.zero_grad()

    # need to move to device BEFORE loading state so that everything gets loaded
    # onto the correct device automatically
    encoder.to(device)
    decoder.to(device)
    critic.to(device)
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

            gen_criticism = critic(decoded)
            real_criticism = critic(img)

            # loss
            losses = losser(target=img, predicted=decoded,
                            gen_criticism=gen_criticism, real_criticism=real_criticism)
            total_loss = losses['total_loss']
            total_loss.backward()

            if step % args.summarize_every_step == 0:
                print(total_loss)
                Summarize(
                    step=step,
                    outdir=args.model_dir,
                    total_loss=losses['total_loss'],
                    gen_loss=losses['gen_loss'],
                    critic_loss=losses['critic_loss'],
                    reconstruction_loss=losses['reconstruction_loss'],
                    input_img=img,
                    encoded_img=encoded,
                    decoded_img=decoded,
                    gen_loss_image=losses['gen_loss_image'],
                    critic_loss_image=losses['critic_loss_image'],
                    reconstruction_loss_image=losses['reconstruction_loss_image'])

            optimizer.step()
            step += args.batch_size
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
