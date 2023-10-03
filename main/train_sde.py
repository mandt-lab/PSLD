import logging
import os
from copy import deepcopy

import hydra
import pytorch_lightning as pl
from callbacks import EMAWeightUpdate
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import get_dataset, get_module, import_modules_into_registry

logger = logging.getLogger(__name__)


# Import all modules into registry
import_modules_into_registry()


@hydra.main(config_path="configs")
def train(config):
    """Helper script for training a score-based generative model"""
    # Get config and setup
    config = config.dataset.diffusion
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Setup dataset
    dataset = get_dataset(config)
    logger.info(f"Using Dataset: {dataset} with size: {len(dataset)}")

    # Setup score predictor
    score_fn_cls = get_module(category="score_fn", name=config.model.score_fn.name)
    score_fn = score_fn_cls(config)
    logger.info(f"Using Score fn: {score_fn_cls}")

    # Setup target network for EMA
    ema_score_fn = deepcopy(score_fn)
    for p in ema_score_fn.parameters():
        p.requires_grad = False

    # Setup Score SDE
    sde_cls = get_module(category="sde", name=config.model.sde.name)
    sde = sde_cls(config)
    logger.info(f"Using SDE: {sde_cls} with type: {sde.type}")
    logger.info(sde)

    # Setup Loss fn
    criterion_cls = get_module(category="losses", name=config.training.loss.name)
    criterion = criterion_cls(config, sde)
    logger.info(f"Using Loss: {criterion_cls}")

    # Setup Lightning Wrapper Module
    wrapper_cls = get_module(category="pl_modules", name=config.model.pl_module)
    wrapper = wrapper_cls(
        config, sde, score_fn, ema_score_fn=ema_score_fn, criterion=criterion
    )

    # Setup Trainer
    train_kwargs = {}

    # Setup callbacks
    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"{config.model.sde.name}-{config.training.chkpt_prefix}"
        + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_top_k=-1,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]
    if config.training.use_ema:
        ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)
        train_kwargs["callbacks"].append(ema_callback)

    device_type = config.training.accelerator
    train_kwargs["accelerator"] = device_type
    loader_kws = {}
    if device_type == "gpu":
        train_kwargs["devices"] = config.training.devices

        # Disable find_unused_parameters when using DDP training for performance reasons
        # train_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device_type == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    batch_size = config.training.batch_size
    batch_size = min(len(dataset), batch_size)
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    # Trainer
    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs, strategy="ddp")

    # Restore checkpoint
    restore_path = config.training.restore_path
    if restore_path == "":
        restore_path = None
    trainer.fit(wrapper, train_dataloaders=loader, ckpt_path=restore_path)


if __name__ == "__main__":
    train()
