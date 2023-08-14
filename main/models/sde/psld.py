import logging

import numpy as np
import torch
from util import register_module, reshape

from .base import SDE

logger = logging.getLogger(__name__)


@register_module(category="sde", name="psld")
class PSLD(SDE):
    def __init__(self, config):
        """Construct a Phase-Space Langevin Diffusion (PSLD) SDE."""
        super().__init__(config.model.sde.n_timesteps)
        self.beta_0 = config.model.sde.beta_min
        self.beta_1 = config.model.sde.beta_max

        #### SDE Core Parameters ####
        self.nu = config.model.sde.nu
        self.gamma = config.model.sde.gamma

        assert self.nu != 0 or self.gamma != 0
        self.m_inv = (self.gamma - self.nu) ** 2 / 4
        self.m = 1 / self.m_inv

        #### Perturbation kernel specific ####
        self.kappa = config.model.sde.kappa
        self.mm_0 = self.kappa * self.m
        self.eps = config.model.sde.numerical_eps
        self.decomp_mode = config.model.sde.decomp_mode
        assert self.decomp_mode in ["lower", "upper"]

    def __repr__(self):
        return f"Initialized SDE with m_inv:{self.m_inv}, gamma: {self.gamma}, nu: {self.nu}, Decomp mode: {self.decomp_mode}"

    def beta_t(self, t):
        # \beta(t)
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def b_t(self, t):
        # \int_0^t \beta(s) ds
        return self.beta_0 * t + 0.5 * (t**2) * (self.beta_1 - self.beta_0)

    @property
    def T(self):
        return 1.0

    @property
    def mode(self):
        if self.gamma == 0:
            return "score_m"
        elif self.nu == 0:
            return "score_x"
        return "score_xm"

    @property
    def type(self):
        return f"psld-{self.mode}"

    def _mean(self, x_0, m_0, t):
        """Returns the mean at time t of the perturbation kernel: p(z_t|z_0) (DSM) or p(z_t|x_0) (HSM)"""
        # Scaling factor
        mu_lam = (self.nu + self.gamma) / 4
        b_t = reshape(self.b_t(t), x_0)
        scaling_factor = torch.exp(-mu_lam * b_t)

        # Soln coefficients
        A_1 = (self.nu - self.gamma) / 4
        A_2 = (self.gamma - self.nu) ** 2 / 8
        C_1 = -0.5
        C_2 = (self.gamma - self.nu) / 4

        # Individual means
        mu_x = A_1 * x_0 * b_t + A_2 * m_0 * b_t + x_0
        assert mu_x.shape == x_0.shape
        mu_m = C_1 * x_0 * b_t + C_2 * m_0 * b_t + m_0
        assert mu_m.shape == m_0.shape

        # Scaling
        mu = torch.cat([mu_x, mu_m], dim=1)
        mu = mu * reshape(scaling_factor, mu)
        return mu

    def _cov(self, xx_0, mm_0, t):
        """Returns the 2 x 2 covariance matrix at time t of the perturbation kernel: p(z_t|z_0) (DSM) or p(z_t|x_0) (HSM)"""
        # Scaling factor
        cov_lam = (self.nu + self.gamma) / 2
        b_t = self.b_t(t)
        b_t2 = b_t**2
        scaling_factor = torch.exp(-cov_lam * b_t)
        inv_scaling_factor = torch.exp(cov_lam * b_t)

        # Only non-zero coefficients defined here for convenience
        A_1, A_2, A_3, A_5, A_6 = (
            self.m_inv / 4,
            self.m_inv**2 / 4,
            (self.nu - self.gamma) / 2,
            -self.m_inv / 2,
            (self.gamma - self.nu) / 2,
        )
        A_7 = inv_scaling_factor - 1
        C_1, C_2, C_3, C_4, C_5 = (
            (self.gamma - self.nu) / 8,
            self.m_inv * (self.gamma - self.nu) / 8,
            -1 / 2,
            self.m_inv / 2,
            (self.nu - self.gamma) / 4,
        )
        D_1, D_2, D_4, D_5, D_6 = (
            1 / 4,
            self.m_inv / 4,
            (self.gamma - self.nu) / 2,
            -1 / 2,
            self.m * (self.nu - self.gamma) / 2,
        )
        D_7 = self.m * (inv_scaling_factor - 1)

        # Individual covariances
        xx_t = (
            A_1 * b_t2 * xx_0
            + A_2 * b_t2 * mm_0
            + A_3 * b_t * xx_0
            + A_5 * b_t2
            + A_6 * b_t
            + A_7
            + xx_0
        ) * scaling_factor

        xm_t = (
            C_1 * b_t2 * xx_0
            + C_2 * b_t2 * mm_0
            + C_3 * b_t * xx_0
            + C_4 * b_t * mm_0
            + C_5 * b_t2
        ) * scaling_factor

        mm_t = (
            D_1 * b_t2 * xx_0
            + D_2 * b_t2 * mm_0
            + D_4 * b_t * mm_0
            + D_5 * b_t2
            + D_6 * b_t
            + D_7
            + mm_0
        ) * scaling_factor

        assert xx_t.shape == t.shape
        assert xm_t.shape == t.shape
        assert mm_t.shape == t.shape
        return xx_t + self.eps, xm_t, mm_t + self.eps

    def get_coeff(self, var):
        """Returns the scalar coefficients for the matrix decomposition of the
        covariance of the perturbation kernel
        """
        xx_t, xm_t, mm_t = var
        if self.decomp_mode == "lower":
            # Cholesky decomposition
            l11 = torch.sqrt(xx_t)
            l12 = 0.0 * torch.ones_like(xx_t)
            l21 = xm_t / l11
            l22 = torch.sqrt(mm_t - l21**2.0)

            if (
                torch.sum(torch.isnan(l11)) > 0
                or torch.sum(torch.isnan(l21)) > 0
                or torch.sum(torch.isnan(l22)) > 0
            ):
                raise ValueError("Numerical precision error.")
            return l11, l12, l21, l22

        # Upper triangular factorization (Not really Cholesky but similar)
        u22 = torch.sqrt(mm_t)
        u12 = xm_t / u22
        u11 = torch.sqrt(xx_t - u12**2.0)
        u21 = 0.0 * torch.ones_like(mm_t)

        if (
            torch.sum(torch.isnan(u22)) > 0
            or torch.sum(torch.isnan(u12)) > 0
            or torch.sum(torch.isnan(u11)) > 0
        ):
            raise ValueError("Numerical precision error.")
        return u11, u12, u21, u22

    def get_inv_coeff(self, var):
        """Returns the scalar coefficients for the inverse-transpose of the
        matrix decomposition of the covariance of the perturbation kernel
        """
        xx_t, xm_t, mm_t = var
        if self.decomp_mode == "lower":
            # Compute L_t^{-T} (Cholesky transpose inv)
            li11 = torch.sqrt(1 / xx_t)
            li12 = -xm_t / (torch.sqrt(xx_t) * torch.sqrt(xx_t * mm_t - xm_t**2))
            li21 = 0.0 * torch.ones_like(xx_t)
            li22 = torch.sqrt(xx_t / (xx_t * mm_t - xm_t**2))

            if (
                torch.sum(torch.isnan(li11)) > 0
                or torch.sum(torch.isnan(li12)) > 0
                or torch.sum(torch.isnan(li22)) > 0
            ):
                raise ValueError("Numerical precision error.")
            return li11, li12, li21, li22

        # Upper triangular factorization (inv transpose)
        ui22 = torch.sqrt(1 / mm_t)
        ui21 = -xm_t / (torch.sqrt(mm_t) * torch.sqrt(xx_t * mm_t - xm_t**2))
        ui11 = torch.sqrt(mm_t / (xx_t * mm_t - xm_t**2))
        ui12 = 0.0 * torch.ones_like(mm_t)

        if (
            torch.sum(torch.isnan(ui22)) > 0
            or torch.sum(torch.isnan(ui21)) > 0
            or torch.sum(torch.isnan(ui11)) > 0
        ):
            raise ValueError("Numerical precision error.")
        return ui11, ui12, ui21, ui22

    def cond_marginal_prob(self, x_0, m_0, xx_0, mm_0, t):
        """Returns mean and variance of the perturbation kernel"""
        mu_t = self._mean(x_0, m_0, t)
        assert mu_t.shape == torch.cat([x_0, m_0], dim=1).shape

        xx_t, xm_t, mm_t = self._cov(xx_0, mm_0, t)
        return mu_t, (xx_t, xm_t, mm_t)

    def get_score(self, eps, xx_0, mm_0, t):
        """Returns the score given epsilon (predicted or just randomly sampled)"""
        var = self._cov(xx_0, mm_0, t)

        # Compute L_t^{-T} (Get transpose inv)
        c11, c12, c21, c22 = self.get_inv_coeff(var)

        # Score = -L_t^{-T} \epsilon. If the noise is explicitly added in only
        # in the position or the momentum space, the score in the other
        # dimension is explicitly set to 0 (since it wont be used during sampling).
        if self.decomp_mode == "lower" and self.mode == "score_m":
            score_x = torch.zeros_like(eps)
            score_m = -reshape(c22, eps).type(torch.float32) * eps
            return torch.cat([score_x, score_m], dim=1)

        if self.decomp_mode == "upper" and self.mode == "score_x":
            score_m = torch.zeros_like(eps)
            score_x = -reshape(c11, eps).type(torch.float32) * eps
            return torch.cat([score_x, score_m], dim=1)

        # Case when both epsilons need to be computed
        eps_x, eps_m = torch.chunk(eps, 2, dim=1)
        score_x = (
            -reshape(c11, eps_x).type(torch.float32) * eps_x
            - reshape(c12, eps_m).type(torch.float32) * eps_m
        )
        score_m = (
            -reshape(c21, eps_x).type(torch.float32) * eps_x
            - reshape(c22, eps_m).type(torch.float32) * eps_m
        )
        return torch.cat([score_x, score_m], dim=1)

    def perturb_data(self, x_0, m_0, xx_0, mm_0, t, eps=None):
        """Add noise to input data point (x_0, m_0)"""
        u_0 = torch.cat([x_0, m_0], dim=1)
        if eps is None:
            eps = torch.randn_like(u_0)

        assert eps.shape == u_0.shape

        # Mean and Variance
        mu_t, var = self.cond_marginal_prob(x_0, m_0, xx_0, mm_0, t)

        # Get covariance factorization components
        c11, c12, c21, c22 = self.get_coeff(var)

        # Sample noise
        eps_x, eps_m = torch.chunk(eps, 2, dim=1)
        noise_x = reshape(c11, eps_x) * eps_x + reshape(c12, eps_m) * eps_m
        noise_m = reshape(c21, eps_x) * eps_x + reshape(c22, eps_m) * eps_m
        noise = torch.cat((noise_x, noise_m), dim=1)

        # Perturb
        u_t = mu_t + noise

        if eps is None:
            return u_t, eps, mu_t, var
        return u_t, mu_t, var

    def predict_x_from_eps(self, z_t, eps, t):
        """Predicts the original (x_0, m_0) pair given a noisy state and
        the predicted or randomly sampled epsilon.
        """
        var = self._cov(0.0, self.mm_0, t)
        l11, l12, l21, l22 = self.get_coeff(var)

        eps_x, eps_m = torch.chunk(eps, 2, dim=1)
        scaled_eps_x = l11 * eps_x + l12 * eps_m
        scaled_eps_m = l21 * eps_x + l22 * eps_m

        # Predict mean
        z_x, z_m = torch.chunk(z_t, 2, dim=1)
        mu_x = z_x - scaled_eps_x
        mu_m = z_m - scaled_eps_m

        # Predict reconstruction
        mu_lam = (self.nu + self.gamma) / 4
        b_t = self.b_t(t)
        scaling_factor = torch.exp(mu_lam * b_t)
        A_1 = (self.nu - self.gamma) / 4
        A_2 = (self.gamma - self.nu) ** 2 / 8
        C_1 = -0.5
        C_2 = (self.gamma - self.nu) / 4

        C = torch.tensor(
            [[A_1 * b_t + 1, A_2 * b_t], [C_1 * b_t, C_2 * b_t + 1]], device=z_t.device
        )
        C_inv = torch.linalg.inv(C)
        C_inv_sc = C_inv * scaling_factor
        c11, c12, c21, c22 = (
            C_inv_sc[0, 0],
            C_inv_sc[0, 1],
            C_inv_sc[1, 0],
            C_inv_sc[1, 1],
        )

        x_recons = c11 * mu_x + c12 * mu_m
        m_recons = c21 * mu_x + c22 * mu_m
        return x_recons, m_recons

    def sde(self, u_t, t):
        """Return the drift and diffusion coefficients"""
        x_t, m_t = torch.chunk(u_t, 2, dim=1)

        beta_t = reshape(self.beta_t(t), x_t)

        drift_x = 0.5 * beta_t * (self.m_inv * m_t - self.gamma * x_t)
        drift_v = 0.5 * beta_t * (-self.nu * m_t - x_t)

        diffusion_x = torch.sqrt(beta_t * self.gamma) * torch.ones_like(x_t)
        diffusion_v = torch.sqrt(beta_t * self.m * self.nu) * torch.ones_like(x_t)
        return torch.cat((drift_x, drift_v), dim=1), torch.cat(
            (diffusion_x, diffusion_v), dim=1
        )

    def reverse_sde(self, u_t, t, score_fn, probability_flow=False):
        """Return the drift and diffusion coefficients of the reverse sde"""
        # The reverse SDE is defines on the domain (T-t)
        t = self.T - t

        # Forward drift and diffusion
        f, g = self.sde(u_t, t)

        # scale the score by 0.5 for the prob. flow formulation
        eps_pred = score_fn(u_t.type(torch.float32), t.type(torch.float32))
        score = self.get_score(eps_pred, 0, self.mm_0, t)
        score = 0.5 * score if probability_flow else score

        # Reverse drift
        f_bar = -f + g**2 * score
        assert f_bar.shape == f.shape

        # Reverse diffusion (0 for prob. flow)
        g_bar = g if not probability_flow else torch.zeros_like(g)
        return f_bar, g_bar

    def prior_sampling(self, shape):
        """Generate samples from the prior p(x_T)"""
        p_x = torch.randn(*shape)
        p_m = torch.randn(*shape) * np.sqrt(self.m)
        return torch.cat([p_x, p_m], dim=1)

    def prior_logp(self, z):
        pass

    def likelihood_weighting(self, t):
        beta_t = self.beta_t(t)
        return beta_t * self.gamma, beta_t * self.m * self.nu
