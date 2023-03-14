import torch
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from util import register_module
from torchvision.datasets import MNIST
import torchvision.transforms as T


@register_module(category="datasets", name='inpaint')
class InpaintDataset(Dataset):
    def __init__(self, config, dataset):
        # Parent dataset (must return only images)
        self.config = config
        self.dataset = dataset

        # Used for creating masks
        t = T.Compose([T.Resize((config.data.image_size, config.data.image_size),InterpolationMode.NEAREST), T.ToTensor()])
        self.mnist = MNIST(config.data.root, train=True, download=True, transform=t)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        digit, _ = self.mnist[idx]
        digit = torch.cat([digit]*3, dim=0)
        mask = (digit > 0).type(torch.long)
        mask = 1 - mask
        assert mask.shape == img.shape
        return img, mask

    def __len__(self):
        return self.config.evaluation.n_samples