import logging

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

_MODULES = {}


def reshape(t, rt):
    """Adds additional dimensions corresponding to the size of the
    reference tensor rt.
    """
    if len(rt.shape) == len(t.shape):
        return t
    ones = [1] * len(rt.shape[1:])
    t_ = t.view(-1, *ones)
    assert len(t_.shape) == len(rt.shape)
    return t_


def data_scaler(img, norm=True):
    if norm:
        img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
    else:
        img = np.asarray(img).astype(np.float) / 255.0
    return img


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

    return dataset_cls(
        root,
        norm=norm,
        transform=transform,
        return_target=config.data.return_target,
    )


def import_modules_into_registry():
    logger.info("Importing modules into registry")
    import datasets
    import losses
    import models
    import samplers


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
