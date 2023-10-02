import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from util import data_scaler, register_module


@register_module(category="datasets", name="afhqv2")
class AFHQv2Dataset(Dataset):
    """Implementation of the AFHQv2 dataset.
    Downloaded from https://github.com/clovaai/stargan-v2
    """
    def __init__(
        self,
        root,
        norm=True,
        subsample_size=None,
        return_target=False,
        transform=None,
        **kwargs,
    ):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform
        self.norm = norm
        self.subsample_size = subsample_size
        self.return_target = return_target

        self.images = []
        self.labels = []

        cat = kwargs.get("cat", [])
        is_train = kwargs.get("train", True)
        subfolder_list = ["dog", "cat", "wild"] if cat == [] else cat
        base_path = os.path.join(self.root, "train" if is_train else "test")
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

        # Scale images between [-1, 1] or [0, 1]
        # Normalize
        img = data_scaler(img, norm=self.norm)

        # Return Targets actually returns the class-label based on the animal category.
        # This is only helpful when using guidance using generation.
        if self.return_target:
            return torch.from_numpy(img).permute(2, 0, 1).float(), self.labels[idx]
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.images) if self.subsample_size is None else self.subsample_size
