import logging
import os
from typing import Sequence, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch.nn import Module

from util import save_as_images, save_as_np

logger = logging.getLogger(__name__)


class EMAWeightUpdate(Callback):
    """EMA weight update
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[EMAWeightUpdate()])
    """

    def __init__(self, tau: float = 0.9999):
        """
        Args:
            tau: EMA decay rate
        """
        super().__init__()
        self.tau = tau
        logger.info(f"Setup EMA callback with tau: {self.tau}")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.score_fn
        target_net = pl_module.ema_score_fn

        # update weights
        self.update_weights(online_net, target_net)

    def update_weights(
        self, online_net: Union[Module, Tensor], target_net: Union[Module, Tensor]
    ) -> None:
        # apply MA weight update
        with torch.no_grad():
            for targ, src in zip(target_net.parameters(), online_net.parameters()):
                targ.mul_(self.tau).add_(src, alpha=1 - self.tau)


# TODO: Add Support for saving momentum images
class SimpleImageWriter(BasePredictionWriter):
    """Pytorch Lightning Callback for writing a batch of images to disk."""

    def __init__(
        self,
        output_dir,
        write_interval,
        sample_prefix="",
        path_prefix="",
        save_mode="image",
        is_norm=True,
        is_augmented=True,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.sample_prefix = sample_prefix
        self.path_prefix = path_prefix
        self.is_norm = is_norm
        self.is_augmented = is_augmented
        self.save_fn = save_as_images if save_mode == "image" else save_as_np

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        rank = pl_module.global_rank

        # Write output images
        # NOTE: We need to use gpu rank during saving to prevent
        # processes from overwriting images
        samples = prediction.cpu()

        # Ignore momentum states if the SDE is augmented
        if self.is_augmented:
            samples, _ = torch.chunk(samples, 2, dim=1)

        # Setup save dirs
        if self.path_prefix != "":
            base_save_path = os.path.join(self.output_dir, str(self.path_prefix))
        else:
            base_save_path = self.output_dir
        img_save_path = os.path.join(base_save_path, "images")
        os.makedirs(img_save_path, exist_ok=True)

        # Save images
        self.save_fn(
            samples,
            file_name=os.path.join(
                img_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"
            ),
            denorm=self.is_norm,
        )


class InpaintingImageWriter(BasePredictionWriter):
    """Pytorch Lightning Callback for writing a batch of images to disk.
    Specifically adapted for Image inpainting.
    """

    def __init__(
        self,
        output_dir,
        write_interval,
        eval_mode="sample",
        sample_prefix="",
        path_prefix="",
        save_mode="image",
        is_norm=True,
        is_augmented=True,
        save_batch=False,
    ):
        super().__init__(write_interval)
        assert eval_mode in ["sample", "recons"]
        self.output_dir = output_dir
        self.eval_mode = eval_mode
        self.sample_prefix = sample_prefix
        self.path_prefix = path_prefix
        self.is_norm = is_norm
        self.is_augmented = is_augmented
        self.save_fn = save_as_images if save_mode == "image" else save_as_np
        self.save_batch = save_batch

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        rank = pl_module.global_rank

        # Write output images
        # NOTE: We need to use gpu rank during saving to prevent
        # processes from overwriting images
        samples = prediction.cpu()

        if self.is_augmented:
            samples, _ = torch.chunk(samples, 2, dim=1)

        # Setup dirs
        if self.path_prefix != "":
            base_save_path = os.path.join(self.output_dir, str(self.path_prefix))
        else:
            base_save_path = self.output_dir
        img_save_path = os.path.join(base_save_path, "images")
        os.makedirs(img_save_path, exist_ok=True)

        # Save images
        self.save_fn(
            samples,
            file_name=os.path.join(
                img_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"
            ),
            denorm=self.is_norm,
        )

        # Save batch (For inpainting)
        if self.save_batch:
            batch_save_path = os.path.join(base_save_path, "batch")
            corr_save_path = os.path.join(base_save_path, "corrupt")
            os.makedirs(batch_save_path, exist_ok=True)
            os.makedirs(corr_save_path, exist_ok=True)
            img, mask = batch
            img = img * 0.5 + 0.5
            self.save_fn(
                img * mask,
                file_name=os.path.join(
                    corr_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"
                ),
                denorm=False,
            )
            self.save_fn(
                img,
                file_name=os.path.join(
                    batch_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"
                ),
                denorm=False,
            )
