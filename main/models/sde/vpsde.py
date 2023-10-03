import numpy as np
import torch
from util import register_module, reshape

from .base import SDE


@register_module(category="sde", name="vpsde")
class VPSDE(SDE):
    def __init__(self, config):
        """Construct a Variance Preserving (VP) SDE."""
        super().__init__(config.model.sde.n_timesteps)
        self.beta_0 = config.model.sde.beta_min
        self.beta_1 = config.model.sde.beta_max

    def beta_t(self, t):
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    @property
    def T(self):
        return 1.0

    @property
    def type(self):
        return "vpsde"

    def get_score(self, eps, t):
        return -eps / reshape(self._std(t), eps)

    def perturb_data(self, x_0, t, noise=None):
        """Add noise to input data point"""
        noise = torch.randn_like(x_0) if None else noise
        assert noise.shape == x_0.shape

        mu_t, std_t = self.cond_marginal_prob(x_0, t)
        assert mu_t.shape == x_0.shape
        assert len(std_t.shape) == len(x_0.shape)
        return mu_t + noise * std_t

    def sde(self, x_t, t):
        """Return the drift and diffusion coefficients"""
        beta_t = reshape(self.beta_t(t), x_t)
        drift = -0.5 * beta_t * x_t
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def reverse_sde(self, x_t, t, score_fn, probability_flow=False):
        """Return the drift and diffusion coefficients of the reverse sde"""
        # The reverse SDE is defines on the domain (T-t)
        t = self.T - t

        # Forward drift and diffusion
        f, g = self.sde(x_t, t)

        # scale the score by 0.5 for the prob. flow formulation
        eps_pred = score_fn(x_t.type(torch.float32), t.type(torch.float32))
        score = self.get_score(eps_pred, t)
        score = 0.5 * score if probability_flow else score

        # Reverse drift
        f_bar = -f + g**2 * score
        assert f_bar.shape == f.shape

        # Reverse diffusion (0 for prob. flow)
        g_bar = g if not probability_flow else torch.zeros_like(g)
        return f_bar, g_bar

    def cond_marginal_prob(self, x_0, t):
        """Generate samples from the perturbation kernel p(x_t|x_0)"""
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x_0
        assert mean.shape == x_0.shape

        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        std = reshape(std, x_0)
        return mean, std

    def _std(self, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        return torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))

    def prior_sampling(self, shape):
        """Generate samples from the prior p(x_T)"""
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def likelihood_weighting(self, t):
        # Return g(t)^2 for likelihood training and computation
        return self.beta_t(t)
