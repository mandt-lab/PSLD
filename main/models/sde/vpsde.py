import numpy as np
import torch
from .base import SDE
from util import register_module


@register_module(category="sde", name="vpsde")
class VPSDE(SDE):
    def __init__(self, config):
        """Construct a Variance Preserving SDE."""
        super().__init__(config.model.sde.n_timesteps)
        self.beta_0 = config.model.sde.beta_min
        self.beta_1 = config.model.sde.beta_max
        self.discrete_betas = torch.linspace(
            self.beta_0 / self.N, self.beta_1 / self.N, self.N
        )
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def get_score(self, eps, t):
        return -eps / self.std(t)

    def perturb_data(self, x_0, t, noise=None):
        """Add noise to input data point"""
        noise = torch.randn_like(x_0) if None else noise
        assert noise.shape == x_0.shape
        mu_t, std_t = self.cond_marginal_prob(x_0, t)
        return mu_t + noise * std_t[:, None, None, None]

    def sde(self, x_t, t):
        """Return the drift and diffusion coefficients"""
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x_t
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def cond_marginal_prob(self, x_0, t):
        """Generate samples from the perturbation kernel p(x_t|x_0)"""
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x_0
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def std(self, t):
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

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G
