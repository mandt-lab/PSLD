import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from util import data_scaler, register_module


@register_module(category="datasets", name="celeba64")
class CelebADataset(Dataset):
    """Implementation of the CelebA dataset.
    Downloaded from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """
    def __init__(self, root, norm=True, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform
        self.norm = norm

        self.images = []

        for img in tqdm(os.listdir(root)):
            self.images.append(os.path.join(self.root, img))

        # Subsample the dataset (if enabled)
        if subsample_size is not None:
            self.images = self.images[:subsample_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)

        # Normalize
        img = data_scaler(img, norm=self.norm)

        # TODO: Add the functionality to return labels to enable
        # guidance-based generation.
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.images)
