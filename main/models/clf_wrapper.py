import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.seed import seed_everything
from util import get_module, register_module

logger = logging.getLogger(__name__)


@register_module(category="pl_modules", name="tclf_wrapper")
class TClfWrapper(pl.LightningModule):
    """This PL module can do the following tasks:
    - train: Train a classifier to predict labels given a noisy state z_t
    - predict: Generate class-conditional samples using classifier guidance
    """

    def __init__(
        self,
        config,
        sde,
        clf_fn,
        score_fn=None,
        criterion=None,
        sampler_cls=None,
        corrector_fn=None,
    ):
        super().__init__()
        self.config = config
        self.sde = sde
        self.clf_fn = clf_fn

        # Training
        self.criterion = criterion
        self.train_eps = self.config.diffusion.training.train_eps

        # Evaluation
        self.score_fn = score_fn
        self.sampler = None

        if sampler_cls is not None:
            self.sampler = sampler_cls(
                self.config,
                self.sde,
                self.score_fn,
                self.clf_fn,
                corrector_fn=corrector_fn,
            )
        self.eval_eps = self.config.diffusion.evaluation.eval_eps
        self.denoise = self.config.diffusion.evaluation.denoise
        n_discrete_steps = self.config.diffusion.evaluation.n_discrete_steps
        self.n_discrete_steps = (
            n_discrete_steps - 1 if self.denoise else n_discrete_steps
        )
        self.val_eps = self.config.diffusion.evaluation.eval_eps
        self.stride_type = self.config.diffusion.evaluation.stride_type

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        # Images and labels
        x_0, y = batch

        # Sample timepoints (between [train_eps, 1])
        t_ = torch.rand(x_0.shape[0], device=x_0.device, dtype=torch.float64)
        t = t_ * (self.sde.T - self.train_eps) + self.train_eps
        assert t.shape[0] == x_0.shape[0]

        # Compute loss and backward
        loss, acc = self.criterion(x_0, y, t, self.clf_fn)

        self.log("loss", loss, prog_bar=True)
        self.log("Top1-Acc", acc, prog_bar=True)
        return loss

    def on_predict_start(self):
        seed = self.config.clf.evaluation.seed

        # This is done for predictions since setting a common seed
        # leads to generating same samples across gpus which affects
        # the evaluation metrics like FID negatively.
        seed_everything(seed + self.global_rank, workers=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        t_final = self.sde.T - self.eval_eps
        ts = torch.linspace(
            0,
            t_final,
            self.n_discrete_steps + 1,
            device=batch.device,
            dtype=torch.float64,
        )

        if self.stride_type == "uniform":
            pass
        elif self.stride_type == "quadratic":
            ts = t_final * torch.flip(1 - (ts / t_final) ** 2.0, dims=[0])

        return self.sampler.sample(
            batch, ts, self.n_discrete_steps, denoise=self.denoise, eps=self.eval_eps
        )

    def on_predict_end(self):
        if isinstance(self.sampler, get_module("samplers", "bb_ode")):
            print(self.sampler.mean_nfe)

    def configure_optimizers(self):
        opt_config = self.config.clf.training.optimizer
        opt_name = opt_config.name
        if opt_name == "Adam":
            optimizer = torch.optim.Adam(
                self.clf_fn.parameters(),
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
