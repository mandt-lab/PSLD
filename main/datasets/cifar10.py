import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from util import register_module, data_scaler


@register_module(category="datasets", name='cifar10')
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        root,
        norm=True,
        transform=None,
        cond_transform=None,
        subsample_size=None,
        **kwargs,
    ):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")

        if subsample_size is not None:
            assert isinstance(subsample_size, int)

        self.root = root
        self.norm = norm
        self.transform = transform
        self.cond_transform = cond_transform
        self.dataset = CIFAR10(
            self.root, train=True, download=True, transform=None, **kwargs
        )
        self.subsample_size = subsample_size

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        # Apply transform
        img_ = self.transform(img) if self.transform is not None else img

        # Normalize
        img = data_scaler(img_, norm=self.norm)

        if self.cond_transform is not None:
            c_img = self.cond_transform(img_)
            c_img = data_scaler(c_img, norm=self.norm)
            return (
                torch.tensor(img).permute(2, 0, 1).float(),
                torch.tensor(c_img).permute(2, 0, 1).float(),
            )
        return torch.tensor(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.dataset) if self.subsample_size is None else self.subsample_size


if __name__ == "__main__":
    root = "/home/pandeyk1/datasets/"
    dataset = CIFAR10Dataset(root)
    print(dataset[0].shape)
