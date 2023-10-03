import logging
import os
import sys

# Add project directory to sys.path
p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)

from copy import deepcopy

import hydra
import pytorch_lightning as pl
from callbacks import InpaintingImageWriter
from datasets import InpaintDataset
from models.wrapper import SDEWrapper
from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import get_dataset, get_module, import_modules_into_registry

logger = logging.getLogger(__name__)


# Import all modules into registry
import_modules_into_registry()


@hydra.main(config_path=os.path.join(p, "configs"))
def inpaint(config):
    """Evaluation script for inpainting with pre-trained score models using guidance."""
    config = config.dataset.diffusion
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.evaluation.seed)

    # Setup Score Predictor
    score_fn_cls = get_module(category="score_fn", name=config.model.score_fn.name)
    score_fn = score_fn_cls(config)
    logger.info(f"Using Score fn: {score_fn_cls}")

    ema_score_fn = deepcopy(score_fn)
    for p in ema_score_fn.parameters():
        p.requires_grad = False

    score_fn.eval()
    ema_score_fn.eval()

    # Setup Score SDE
    sde_cls = get_module(category="sde", name=config.model.sde.name)
    sde = sde_cls(config)
    logger.info(f"Using SDE: {sde_cls}")

    # Setup sampler
    sampler_cls = get_module(category="samplers", name=config.evaluation.sampler.name)
    logger.info(f"Using Sampler: {sampler_cls}")

    # Setup dataset
    base_dataset = get_dataset(config)
    dataset = InpaintDataset(config, base_dataset)
    logger.info(f"Using Dataset: {dataset} with size: {len(dataset)}")

    wrapper = SDEWrapper.load_from_checkpoint(
        config.evaluation.chkpt_path,
        config=config,
        sde=sde,
        score_fn=score_fn,
        ema_score_fn=ema_score_fn,
        sampler_cls=sampler_cls,
    )

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device_type = config.evaluation.accelerator
    test_kwargs["accelerator"] = device_type
    if device_type == "gpu":
        test_kwargs["devices"] = config.evaluation.devices
        # # Disable find_unused_parameters when using DDP training for performance reasons
        # loader_kws["persistent_workers"] = True
    elif device_type == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=config.evaluation.workers,
        **loader_kws,
    )

    # Setup Image writer callback trainer
    write_callback = InpaintingImageWriter(
        config.evaluation.save_path,
        write_interval="batch",
        sample_prefix=config.evaluation.sample_prefix,
        path_prefix=config.evaluation.path_prefix,
        save_mode=config.evaluation.save_mode,
        is_augmented=config.model.sde.is_augmented,
        save_batch=True,
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config.evaluation.save_path

    # Setup Pl module and predict
    sampler = pl.Trainer(**test_kwargs)
    sampler.predict(wrapper, val_loader)


if __name__ == "__main__":
    inpaint()
