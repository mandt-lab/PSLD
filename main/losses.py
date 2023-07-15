import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import get_module, register_module, reshape

logger = logging.getLogger(__name__)


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = torch.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


@register_module(category="losses", name="score_loss")
class ScoreLoss(nn.Module):
    """Loss function for training non-augmented Score based models (like VP-SDE)"""

    def __init__(self, config, sde):
        super().__init__()
        assert config.training.loss.weighting in ["nll", "fid"]
        self.sde = sde
        self.l_type = config.training.loss.l_type
        self.weighting = config.training.loss.weighting

        logger.info(f"Initialized Loss fn with weighting: {self.weighting}")

        if self.weighting == "nll" and self.l_type != "l2":
            # Use only MSE loss when maximizing for nll
            raise ValueError("l_type can only be `l2` when using nll weighting")

        self.reduce_strategy = "mean" if config.training.loss.reduce_mean else "sum"
        criterion_type = nn.MSELoss if self.l_type == "l2" else nn.L1Loss
        self.criterion = criterion_type(reduction=self.reduce_strategy)

    def forward(self, x_0, t, score_fn, eps=None):
        if eps is None:
            eps = torch.randn_like(x_0, device=x_0.device)

        assert eps.shape == x_0.shape

        # Predict epsilon
        x_t = self.sde.perturb_data(x_0, t, noise=eps)
        eps_pred = score_fn(x_t.type(torch.float32), t.type(torch.float32))

        # Use eps-prediction when optimizing for FID
        loss = self.criterion(eps, eps_pred)

        # Use g(t)**2 weighting when optimizing for NLL
        if self.weighting == "nll":
            gt_2 = reshape(self.sde.likelihood_weighting(t), x_0)
            gt_score = self.sde.get_score(eps, t)
            pred_score = self.sde.get_score(eps_pred, t)
            # loss = self.criterion(pred_score, gt_score) * gt_2
            loss = (pred_score - gt_score) ** 2 * gt_2
            loss = (
                torch.mean(loss) if self.reduce_strategy == "mean" else torch.sum(loss)
            )

        return loss


@register_module(category="losses", name="psld_score_loss")
class PSLDScoreLoss(nn.Module):
    """Loss function for training PSLD."""

    def __init__(self, config, sde):
        super().__init__()
        # TODO: Add support for likelihood training
        assert config.training.loss.weighting in ["fid"]
        assert config.training.mode in ["hsm", "dsm"]
        assert isinstance(sde, get_module("sde", "psld"))
        self.sde = sde
        self.l_type = config.training.loss.l_type
        self.weighting = config.training.loss.weighting
        self.mode = config.training.mode
        self.decomp_mode = config.model.sde.decomp_mode

        logger.info(
            f"Initialized Loss fn with weighting: {self.weighting} mode: {self.mode}"
        )

        if self.weighting == "nll" and self.l_type != "l2":
            # Use only MSE loss when maximizing for nll
            raise ValueError("l_type can only be `l2` when using nll weighting")

        self.reduce_strategy = "mean" if config.training.loss.reduce_mean else "sum"

    def forward(self, x_0, t, score_fn, eps=None):
        # Sample momentum (DSM)
        m_0 = np.sqrt(self.sde.mm_0) * torch.randn_like(x_0)
        mm_0 = 0.0

        # Update momentum (if training mode is HSM)
        if self.mode == "hsm":
            m_0 = torch.zeros_like(x_0)
            mm_0 = self.sde.mm_0

        z_0 = torch.cat([x_0, m_0], dim=1)
        xx_0 = 0

        # Sample random noise
        if eps is None:
            eps = torch.randn_like(z_0)
        assert eps.shape == z_0.shape

        # Predict epsilon
        z_t, _, _ = self.sde.perturb_data(x_0, m_0, xx_0, mm_0, t, eps=eps)
        z_t = z_t.type(torch.float32)
        eps_pred = score_fn(z_t, t.type(torch.float32))

        # Use eps-prediction when optimizing for FID
        eps_x, eps_m = torch.chunk(eps, 2, dim=1)
        if self.sde.mode == "score_m" and self.decomp_mode == "lower":
            assert eps_pred.shape == eps_m.shape
            loss = (eps_m - eps_pred) ** 2
        elif self.sde.mode == "score_x" and self.decomp_mode == "upper":
            assert eps_pred.shape == eps_x.shape
            loss = (eps_x - eps_pred) ** 2
        else:
            assert eps_pred.shape == eps.shape
            loss = (eps - eps_pred) ** 2

        loss = torch.mean(loss) if self.reduce_strategy == "mean" else torch.sum(loss)
        return loss


@register_module(category="losses", name="tce_loss")
class PSLDTimeCELoss(nn.Module):
    """Loss function for training noise conditioned classifier for guidance"""

    def __init__(self, config, sde):
        super().__init__()
        assert config.diffusion.training.mode in ["hsm", "dsm"]
        assert isinstance(sde, get_module("sde", "psld"))
        self.sde = sde
        self.l_type = config.clf.training.loss.l_type
        self.mode = config.diffusion.training.mode

        self.reduce_strategy = (
            "mean" if config.diffusion.training.loss.reduce_mean else "sum"
        )
        self.criterion = nn.CrossEntropyLoss(reduction=self.reduce_strategy)

    def forward(self, x_0, y, t, clf_fn):
        # Sample momentum (DSM)
        m_0 = np.sqrt(self.sde.mm_0) * torch.randn_like(x_0)
        mm_0 = 0.0

        # Update momentum (if training mode is HSM)
        if self.mode == "hsm":
            m_0 = torch.zeros_like(x_0)
            mm_0 = self.sde.mm_0

        u_0 = torch.cat([x_0, m_0], dim=1)
        xx_0 = 0

        # Sample random noise
        eps = torch.randn_like(u_0)

        # Perturb Data
        u_t, _, _ = self.sde.perturb_data(x_0, m_0, xx_0, mm_0, t, eps=eps)

        # Predict label
        y_pred = clf_fn(u_t.type(torch.float32), t)

        # CE loss
        loss = self.criterion(y_pred, y)

        # Top-k accuracy (for debugging)
        acc = compute_top_k(y_pred, y, 1)
        return loss, acc
