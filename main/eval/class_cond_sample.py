import logging
import os
import sys
from copy import deepcopy

# Add project directory to sys.path
p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)

import hydra
import pytorch_lightning as pl
from callbacks import SimpleImageWriter
from datasets.latent import SDELatentDataset
from models.clf_wrapper import TClfWrapper
from models.wrapper import SDEWrapper
from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import get_module, import_modules_into_registry

logger = logging.getLogger(__name__)


# Import all modules into registry
import_modules_into_registry()


@hydra.main(config_path=os.path.join(p, "configs"))
def cc_sample(config):
    """Evaluation script for Class conditional sampling with pre-trained score models 
    using classifier guidance.
    """
    config = config.dataset
    config_sde = config.diffusion
    config_clf = config.clf
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config_clf.evaluation.seed)

    # Setup Score SDE
    sde_cls = get_module(category="sde", name=config_sde.model.sde.name)
    sde = sde_cls(config_sde)
    logger.info(f"Using SDE: {sde_cls}")

    # Setup score predictor
    score_fn_cls = get_module(category="score_fn", name=config_sde.model.score_fn.name)
    score_fn = score_fn_cls(config_sde)
    logger.info(f"Using Score fn: {score_fn_cls}")

    ema_score_fn = deepcopy(score_fn)
    for p in ema_score_fn.parameters():
        p.requires_grad = False

    wrapper = SDEWrapper.load_from_checkpoint(
        config_sde.evaluation.chkpt_path,
        config=config_sde,
        sde=sde,
        score_fn=score_fn,
        ema_score_fn=ema_score_fn,
        sampler_cls=None,
    )

    score_fn = (
        wrapper.ema_score_fn
        if config_sde.evaluation.sample_from == "target"
        else wrapper.score_fn
    )
    score_fn.eval()

    # Setup sampler
    sampler_cls = get_module(
        category="samplers", name=config_sde.evaluation.sampler.name
    )
    logger.info(
        f"Using Sampler: {sampler_cls}. Make sure the sampler supports Class-conditional sampling"
    )

    # Setup dataset
    dataset = SDELatentDataset(sde, config_sde)
    logger.info(f"Using Dataset: {dataset} with size: {len(dataset)}")

    # Setup classifier (for guidance)
    clf_fn_cls = get_module(category="clf_fn", name=config_clf.model.clf_fn.name)
    clf_fn = clf_fn_cls(config_clf)
    logger.info(f"Using Classifier fn: {clf_fn_cls}")

    wrapper = TClfWrapper.load_from_checkpoint(
        config_clf.evaluation.chkpt_path,
        config=config,
        sde=sde,
        clf_fn=clf_fn,
        score_fn=score_fn,
        sampler_cls=sampler_cls,
        strict=False,
    )
    wrapper.eval()

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device_type = config_sde.evaluation.accelerator
    test_kwargs["accelerator"] = device_type
    if device_type == "gpu":
        test_kwargs["devices"] = config_sde.evaluation.devices
        # # Disable find_unused_parameters when using DDP training for performance reasons
        # loader_kws["persistent_workers"] = True
    elif device_type == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        dataset,
        batch_size=config_sde.evaluation.batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=config_sde.evaluation.workers,
        **loader_kws,
    )

    # Setup Image writer callback trainer
    write_callback = SimpleImageWriter(
        config_sde.evaluation.save_path,
        write_interval="batch",
        sample_prefix=config_sde.evaluation.sample_prefix,
        path_prefix=config_sde.evaluation.path_prefix,
        save_mode=config_sde.evaluation.save_mode,
        is_augmented=config_sde.model.sde.is_augmented,
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config_sde.evaluation.save_path

    # Setup Pl module and predict
    sampler = pl.Trainer(**test_kwargs)
    sampler.predict(wrapper, val_loader)


if __name__ == "__main__":
    cc_sample()
