import os

import torch
import numpy as np
from PIL import Image


class UnlabeledDirectoryDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_name)

        # transpose so the data is CHW instead of HWC
        img = np.transpose(image, [2, 0, 1])
        sample = {'image': img}

        if self.transform:
            sample = self.transform(sample)

        return sample
