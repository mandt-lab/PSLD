import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from util import data_scaler, register_module


@register_module(category="datasets", name="cifar10")
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        root,
        norm=True,
        transform=None,
        subsample_size=None,
        return_target=False,
        **kwargs,
    ):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")

        if subsample_size is not None:
            assert isinstance(subsample_size, int)

        self.root = root
        self.norm = norm
        self.transform = transform
        self.dataset = CIFAR10(self.root, train=True, download=True, transform=None)
        self.subsample_size = subsample_size
        self.return_target = return_target

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        # Apply transform
        img_ = self.transform(img) if self.transform is not None else img

        # Normalize
        img = data_scaler(img_, norm=self.norm)

        # Return (with targets if needed for guidance-based generation)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()
        if self.return_target:
            return img_tensor, target
        return img_tensor

    def __len__(self):
        return len(self.dataset) if self.subsample_size is None else self.subsample_size
