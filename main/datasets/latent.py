from torch.utils.data import Dataset
from util import register_module


@register_module(category="datasets", name="latent")
class SDELatentDataset(Dataset):
    """A dataset for generating samples from the equilibrium distribution
    of the forward SDE (useful during sampling)
    """
    def __init__(self, sde, config):
        self.sde = sde
        self.num_samples = config.evaluation.n_samples
        self.shape = [
            self.num_samples,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        ]
        self.samples = self.sde.prior_sampling(self.shape)

    def get_batch(self, shape):
        return self.sde.prior_sampling(shape)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return self.num_samples
