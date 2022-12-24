import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

_MODULES = {}


def register_module(category=None, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        local_category = category
        if local_category is None:
            local_category = cls.__name__ if name is None else name

        # Create category (if does not exist)
        if local_category not in _MODULES:
            _MODULES[local_category] = {}

        # Add module to the category
        local_name = cls.__name__ if name is None else name
        if name in _MODULES[local_category]:
            raise ValueError(
                f"Already registered module with name: {local_name} in category: {category}"
            )

        _MODULES[local_category][local_name] = cls
        return cls

    return _register


def get_module(category, name):
    module = _MODULES.get(category, dict()).get(name, None)
    if module is None:
        raise ValueError(f"No module named `{name}` found in category: `{category}`")
    return module


def configure_device(device):
    if device.startswith("gpu"):
        if not torch.cuda.is_available():
            raise Exception(
                "CUDA support is not available on your platform. Re-run using CPU or TPU mode"
            )
        gpu_id = device.split(":")[-1]
        if gpu_id == "":
            # Use all GPU's
            gpu_id = -1
        gpu_id = [int(id) for id in gpu_id.split(",")]
        return f"cuda:{gpu_id}", gpu_id
    return device


def space_timesteps(num_timesteps, desired_count, type="uniform"):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :return: a set of diffusion steps from the original process to use.
    """
    if type == "uniform":
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == desired_count:
                return range(0, num_timesteps, i)
        raise ValueError(
            f"cannot create exactly {desired_count} steps with an integer stride"
        )
    elif type == "quad":
        seq = np.linspace(0, np.sqrt(num_timesteps * 0.8), desired_count) ** 2
        seq = [int(s) for s in list(seq)]
        return seq
    else:
        raise NotImplementedError


def get_dataset(config):
    # TODO: Add support for dynamically adding **kwargs directly via config
    # Parse config
    name = config.data.name
    root = config.data.root
    image_size = config.data.image_size
    norm = config.data.norm
    flip = config.data.hflip

    # Checks
    assert isinstance(norm, bool)

    if name.lower() == "cifar10":
        assert image_size == 32

    # Construct transforms
    t_list = [T.Resize((image_size, image_size))]
    if flip:
        t_list.append(T.RandomHorizontalFlip())
    transform = T.Compose(t_list)

    # Get dataset
    dataset_cls = get_module(category="datasets", name=name.lower())
    if dataset_cls is None:
        raise ValueError(
            f"Dataset with name: {name} not found in category: `datasets`. Ensure its properly registered"
        )

    return dataset_cls(root, norm=norm, transform=transform)


def import_modules_into_registry():
    logger.info("Importing modules into registry")
    import datasets
    import losses
    import models


def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.squeeze().permute(1, 2, 0))
        ax[i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def compare_interpolations(
    interpolations_1,
    interpolations_2,
    save_path=None,
    figsize=(10, 2),
    denorm=True,
):
    assert len(interpolations_1) == len(interpolations_2)
    N = len(interpolations_1)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)

    for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
        # De-Norm
        inter_1 = 0.5 * inter_1 + 0.5 if denorm else inter_1
        # inter_2 = 0.5 * inter_2 + 0.5 if denorm else inter_2

        # Plot
        ax[0, i].imshow(inter_1.squeeze().permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.squeeze().permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=100, pad_inches=0)


def convert_to_np(obj):
    obj = obj.permute(0, 2, 3, 1).contiguous()
    obj = obj.detach().cpu().numpy()

    obj_list = []
    for _, out in enumerate(obj):
        obj_list.append(out)
    return obj_list


def normalize(obj):
    B, C, H, W = obj.shape
    for i in range(3):
        channel_val = obj[:, i, :, :].view(B, -1)
        channel_val -= channel_val.min(1, keepdim=True)[0]
        channel_val /= (
            channel_val.max(1, keepdim=True)[0] - channel_val.min(1, keepdim=True)[0]
        )
        channel_val = channel_val.view(B, H, W)
        obj[:, i, :, :] = channel_val
    return obj


def save_as_images(obj, file_name="output", denorm=True):
    # Saves predictions as png images (useful for Sample generation)
    if denorm:
        # obj = normalize(obj)
        obj = obj * 0.5 + 0.5
    obj_list = convert_to_np(obj)

    for i, out in enumerate(obj_list):
        out = (out * 255).clip(0, 255).astype(np.uint8)
        img_out = Image.fromarray(out)
        current_file_name = file_name + "_%d.png" % i
        img_out.save(current_file_name, "png")


def save_as_np(obj, file_name="output", denorm=True):
    # Saves predictions directly as numpy arrays
    if denorm:
        obj = normalize(obj)
    obj_list = convert_to_np(obj)

    for i, out in enumerate(obj_list):
        current_file_name = file_name + "_%d.npy" % i
        np.save(current_file_name, out)


def compare_samples(samples, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    ncols = len(samples)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

    for idx, (caption, img) in enumerate(samples.items()):
        ax[idx].imshow(img.permute(1, 2, 0))
        ax[idx].set_title(caption)
        ax[idx].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=100, pad_inches=0)

    plt.close()
