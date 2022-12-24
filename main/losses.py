import torch
import torch.nn as nn

from util import register_module


@register_module(category="losses", name="score_loss")
class ScoreLoss(nn.Module):
    def __init__(self, config, sde):
        super().__init__()
        assert config.weighting in ["nll", "fid"]
        self.sde = sde
        self.l_type = config.l_type
        self.weighting = config.weighting

        if self.weighting == "nll" and self.l_type != "l2":
            # Use only MSE loss when maximizing for nll
            raise ValueError("l_type can only be `l2` when using nll weighting")

        self.reduce_strategy = "mean" if config.reduce_mean else "sum"
        criterion_type = nn.MSELoss if self.l_type == "l2" else nn.L1Loss
        self.criterion = criterion_type(reduction=self.reduce_strategy)

    def forward(self, eps, eps_pred, t):
        # Use g(t)**2 weighting when optimizing for NLL
        if self.weighting == "nll":
            gt_2 = self.sde.sde(torch.zeros_like(eps), t)[1] ** 2
            gt_score = self.sde.get_score(eps)
            pred_score = self.sde.get_score(eps_pred)
            loss = self.criterion(pred_score, gt_score) * gt_2

        # Use eps-prediction when optimizing for FID
        loss = self.criterion(eps, eps_pred)
        return loss
