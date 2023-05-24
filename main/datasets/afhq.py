import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from util import register_module


@register_module(category="datasets", name='afhqv2')
class AFHQv2Dataset(Dataset):
    def __init__(self, root, norm=True, subsample_size=None, return_target=False, transform=None, **kwargs):
        # We only train on the AFHQ train set
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform
        self.norm = norm
        self.subsample_size = subsample_size
        self.return_target = return_target

        self.images = []
        self.labels = []

        subfolder_list = ["dog", "cat", "wild"]
        # subfolder_list = ["wild"]
        base_path = os.path.join(self.root, "train")
        for idx, subfolder in enumerate(subfolder_list):
            sub_path = os.path.join(base_path, subfolder)

            for img in tqdm(os.listdir(sub_path)):
                self.images.append(os.path.join(sub_path, img))
                self.labels.append(idx)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0

        if self.return_target:
            return torch.from_numpy(img).permute(2, 0, 1).float(), self.labels[idx]
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.images) if self.subsample_size is None else self.subsample_size


if __name__ == "__main__":
    root = "/home/pandeyk1/datasets/afhqv2/"
    dataset = AFHQv2Dataset(root)
    print(len(dataset))
