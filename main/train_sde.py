import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from callbacks import EMAWeightUpdate
from models.wrapper import SDEWrapper
from util import configure_device, get_dataset, get_module, import_modules_into_registry

logger = logging.getLogger(__name__)


# Import all modules into registry
import_modules_into_registry()


@hydra.main(config_path="configs")
def train(config):
    # Get config and setup
    config = config.dataset.diffusion
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Setup dataset
    dataset = get_dataset(config)
    logger.info(f"Using Dataset: {dataset} with size: {len(dataset)}")

    # Setup score predictor
    denoiser_cls = get_module(category="denoiser", name=config.model.denoiser.name)
    denoiser = denoiser_cls(config)
    logger.info(f"Using Denoiser Backend: {denoiser_cls}")

    # Setup Score SDE
    sde_cls = get_module(category="sde", name=config.model.sde.name)
    sde = sde_cls(config)
    logger.info(f"Using SDE Backend: {sde_cls}")

    # Setup Loss fn
    criterion_cls = get_module(category="losses", name="score_loss")
    logger.info(f"Using Loss Backend: {criterion_cls}")

    # Setup Lightning Wrapper Module
    wrapper = SDEWrapper(config, denoiser, sde, criterion_cls)

    # Setup Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path != "":
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    # Setup callbacks
    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vpsde-{config.training.chkpt_prefix}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    if config.training.use_ema:
        ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)
        train_kwargs["callbacks"].append(ema_callback)

    device = config.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
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

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(wrapper, train_dataloader=loader)


if __name__ == "__main__":
    train()
