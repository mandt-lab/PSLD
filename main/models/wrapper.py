import logging

import pytorch_lightning as pl
import torch
from util import register_module

logger = logging.getLogger(__name__)


@register_module(category="pl_modules", name="sde_wrapper")
class SDEWrapper(pl.LightningModule):
    def __init__(self, config, denoiser, sde, criterion_cls):
        super().__init__()
        self.config = config
        self.denoiser = denoiser
        self.sde = sde
        self.criterion = criterion_cls(self.config.training.loss, sde)
        self.train_eps = self.config.training.train_eps

        # Disable automatic optimization
        self.automatic_optimization = False

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        x_0 = batch

        # Sample timepoints (between [eps, 1])
        t_ = torch.rand(x_0.shape[0], device=x_0.device)
        t = t_ * (self.sde.T - self.train_eps) + self.train_eps
        assert t.shape[0] == x_0.shape[0]

        # Sample noise
        eps = torch.randn_like(x_0)

        # Predict epsilon
        x_t = self.sde.perturb_data(x_0, t, noise=eps)
        eps_pred = self.denoiser(x_t, t)

        # Compute loss
        loss = self.criterion(eps, eps_pred, t)

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.denoiser.parameters(), self.config.training.optimizer.grad_clip
        )
        optim.step()

        # Scheduler step
        lr_sched.step()
        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def configure_optimizers(self):
        opt_config = self.config.training.optimizer
        opt_name = opt_config.name
        if opt_name == "Adam":
            optimizer = torch.optim.Adam(
                self.denoiser.parameters(),
                lr=opt_config.lr,
                betas=(opt_config.beta_1, opt_config.beta_2),
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer {opt_name} not supported yet!")

        # Define the LR scheduler (As in Ho et al.)
        if opt_config.warmup == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / opt_config.warmup, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }
