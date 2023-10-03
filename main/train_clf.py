import logging
import os

import hydra
import pytorch_lightning as pl
from models.clf_wrapper import TClfWrapper
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import get_dataset, get_module, import_modules_into_registry

logger = logging.getLogger(__name__)


# Import all modules into registry
import_modules_into_registry()


@hydra.main(config_path="configs")
def train_clf(config):
    """Helper script for training a noise conditioned classifier for guidance purposes"""
    # Get config and setup
    config = config.dataset
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.clf.training.seed, workers=True)

    # Setup dataset
    dataset = get_dataset(config.clf)
    logger.info(f"Using Dataset: {dataset} with size: {len(dataset)}")

    # Setup score predictor
    clf_fn_cls = get_module(category="clf_fn", name=config.clf.model.clf_fn.name)
    clf_fn = clf_fn_cls(config.clf)
    logger.info(f"Using Classifier fn: {clf_fn_cls}")

    # Setup Score SDE
    sde_cls = get_module(category="sde", name=config.diffusion.model.sde.name)
    sde = sde_cls(config.diffusion)
    logger.info(f"Using SDE: {sde_cls} with type: {sde.type}")
    logger.info(sde)

    # Setup Loss fn
    criterion_cls = get_module(category="losses", name=config.clf.training.loss.name)
    criterion = criterion_cls(config, sde)
    logger.info(f"Using Loss: {criterion_cls}")

    # Setup Lightning Wrapper Module
    wrapper = TClfWrapper(config, sde, clf_fn, score_fn=None, criterion=criterion)

    # Setup Trainer
    train_kwargs = {}

    # Setup callbacks
    results_dir = config.clf.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"{config.clf.model.clf_fn.name}-{config.diffusion.model.sde.name}-{config.clf.training.chkpt_prefix}"
        + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.clf.training.chkpt_interval,
        save_top_k=-1,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.clf.training.epochs
    train_kwargs["log_every_n_steps"] = config.clf.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    device_type = config.clf.training.accelerator
    train_kwargs["accelerator"] = device_type
    loader_kws = {}
    if device_type == "gpu":
        train_kwargs["devices"] = config.clf.training.devices

        # Disable find_unused_parameters when using DDP training for performance reasons
        train_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device_type == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config.clf.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    batch_size = config.clf.training.batch_size
    batch_size = min(len(dataset), batch_size)
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.clf.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    # Trainer
    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)

    # Restore checkpoint
    restore_path = config.clf.training.restore_path
    if restore_path == "":
        restore_path = None
    trainer.fit(wrapper, train_dataloaders=loader, ckpt_path=restore_path)


if __name__ == "__main__":
    train_clf()
