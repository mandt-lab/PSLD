import logging
import numpy as np
import torch
from .base import SDE
from util import register_module, reshape

logger = logging.getLogger(__name__)


@register_module(category="sde", name="es3sde")
class ES3SDE(SDE):
    def __init__(self, config):
        """Construct a Extended State Space Score (ES3) SDE."""
        super().__init__(config.model.sde.n_timesteps)
        self.beta_0 = config.model.sde.beta_min
        self.beta_1 = config.model.sde.beta_max
        self.use_ms = config.model.sde.use_ms

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
        return f"es3sde-{self.mode}"

    def _mean(self, x_0, m_0, t):
        """Returns the mean at time t of the perturbation kernel: p(u_t|u_0)"""
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
        """Returns the 2 x 2 covariance matrix at time t of the perturbation kernel: p(u_t|u_0).
        One needs to run `torch.kron` on the output to get the original covariance matrix.
        Usually xx_0 will be 0 so a lot of the coefficients in the final soln will zero out"""
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
        """Returns mean and variance of the perturbation kernel p(x_t|x_0)"""
        mu_t = self._mean(x_0, m_0, t)
        assert mu_t.shape == torch.cat([x_0, m_0], dim=1).shape

        xx_t, xm_t, mm_t = self._cov(xx_0, mm_0, t)
        return mu_t, (xx_t, xm_t, mm_t)

    def get_ms(self, u_t, var):
        x_t, m_t = torch.chunk(u_t, 2, dim=1)
        _, _, mm_t = var

        # Get factorization coeffs
        l11, l12, l21, l22 = self.get_coeff(var)

        s_x = x_t / reshape(mm_t, x_t)
        s_m = m_t / reshape(mm_t, m_t)

        # Compute L_t^T * s
        ms_x = reshape(l11, x_t) * s_x + reshape(l21, m_t) * s_m
        ms_m = reshape(l12, x_t) * s_x + reshape(l22, m_t) * s_m
        return torch.cat([ms_x, ms_m], dim=1)

    def get_score(self, u_t, xx_0, mm_0, eps, t):
        """Returns the score of the perturbation kernel given eps (predicted or just noise)"""
        x_t, m_t = torch.chunk(u_t, 2, dim=1)
        var = self._cov(xx_0, mm_0, t)

        # Compute L_t^{-T} (Get transpose inv)
        c11, c12, c21, c22 = self.get_inv_coeff(var)

        ms_m = torch.zeros_like(m_t)
        ms_x = torch.zeros_like(x_t)
        if self.use_ms:
            ms_x = x_t / reshape(var[2], x_t)
            ms_m = m_t / reshape(var[2], m_t)

        # Score = -L_t^{-T} \epsilon
        if self.decomp_mode == "lower" and self.mode == "score_m":
            score_x = torch.zeros_like(eps)
            score_m = -reshape(c22, eps).type(torch.float32) * eps - ms_m
            return torch.cat([score_x, score_m], dim=1)

        if self.decomp_mode == "upper" and self.mode == "score_x":
            score_m = torch.zeros_like(eps)
            score_x = -reshape(c11, eps).type(torch.float32) * eps - ms_x
            return torch.cat([score_x, score_m], dim=1)

        # Case when both epsilons need to be computed
        eps_x, eps_m = torch.chunk(eps, 2, dim=1)
        score_x = (
            -reshape(c11, eps_x).type(torch.float32) * eps_x
            - reshape(c12, eps_m).type(torch.float32) * eps_m
        ) - ms_x
        score_m = (
            -reshape(c21, eps_x).type(torch.float32) * eps_x
            - reshape(c22, eps_m).type(torch.float32) * eps_m
        ) - ms_m
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

    def predict_x_from_eps(self, u_t, eps, t):
        var = self._cov(0.0, self.mm_0, t)
        l11, l12, l21, l22 = self.get_coeff(var)

        eps_x, eps_m = torch.chunk(eps, 2, dim=1)
        scaled_eps_x = l11 * eps_x + l12 * eps_m
        scaled_eps_m = l21 * eps_x + l22 * eps_m

        # Predict mean
        u_x, u_m = torch.chunk(u_t, 2, dim=1)
        mu_x = u_x - scaled_eps_x
        mu_m = u_m - scaled_eps_m

        # Predict reconstruction
        mu_lam = (self.nu + self.gamma) / 4
        b_t = self.b_t(t)
        scaling_factor = torch.exp(mu_lam * b_t)
        A_1 = (self.nu - self.gamma) / 4
        A_2 = (self.gamma - self.nu) ** 2 / 8
        C_1 = -0.5
        C_2 = (self.gamma - self.nu) / 4

        C = torch.tensor([[A_1 * b_t + 1, A_2 * b_t], [C_1 * b_t, C_2 * b_t + 1]], device=u_t.device)
        C_inv = torch.linalg.inv(C)
        C_inv_sc = C_inv * scaling_factor
        c11, c12, c21, c22 = C_inv_sc[0, 0], C_inv_sc[0, 1], C_inv_sc[1, 0], C_inv_sc[1, 1]

        x_recons = c11 * mu_x + c12 * mu_m
        m_recons = c21 * mu_x + c22 * mu_m

        return x_recons, m_recons

    def clip_and_predict_eps(self, z, eps, t):
        # Predict reconstruction
        x_recons, m_recons = self.predict_x_from_eps(z, eps, t)
        x_recons = x_recons.clip(-1.0, 1.0)
        m_recons = torch.zeros_like(x_recons)

        # Compute eps_pred with new reconstruction
        mu_lam = (self.nu + self.gamma) / 4
        b_t = self.b_t(t)
        scaling_factor = torch.exp(-mu_lam * b_t)
        A_1 = (self.nu - self.gamma) / 4
        A_2 = (self.gamma - self.nu) ** 2 / 8
        C_1 = -0.5
        C_2 = (self.gamma - self.nu) / 4

        S_t = torch.tensor([[A_1 * b_t + 1, A_2 * b_t], [C_1 * b_t, C_2 * b_t + 1]], device=z.device) * scaling_factor
        s11, s12, s21, s22 = S_t[0,0], S_t[0,1], S_t[1,0], S_t[1,1]
        sr_x = s11 * x_recons + s12 * m_recons
        sr_m = s21 * x_recons + s22 * m_recons
        sr = torch.cat([sr_x, sr_m], dim=1)

        diff = z - sr
        diff_x, diff_m = torch.chunk(diff, 2, dim=1)
        var_t = self._cov(0, self.mm_0, t)
        li11, li12, li21, li22 = self.get_inv_coeff(var_t)
        s_eps_x = li11 * diff_x + li21 * diff_m
        s_eps_m = li12 * diff_x + li22 * diff_m

        return s_eps_x, s_eps_m

    def sde(self, u_t, t):
        """Return the drift and diffusion coefficients"""
        x_t, m_t = torch.chunk(u_t, 2, dim=1)

        beta_t = reshape(self.beta_t(t), x_t)
        # beta_t = self.beta_t(t)

        drift_x = 0.5 * beta_t * (self.m_inv * m_t - self.gamma * x_t)
        drift_v = 0.5 * beta_t * (-self.nu * m_t - x_t)

        diffusion_x = torch.sqrt(beta_t * self.gamma) * torch.ones_like(x_t)
        diffusion_v = torch.sqrt(beta_t * self.m * self.nu) * torch.ones_like(x_t)
        return torch.cat((drift_x, drift_v), dim=1), torch.cat(
            (diffusion_x, diffusion_v), dim=1
        )

    def reverse_sde(self, u_t, t, score_fn, probability_flow=False, clip=False):
        """Return the drift and diffusion coefficients of the reverse sde"""
        # The reverse SDE is defines on the domain (T-t)
        t = self.T - t

        # Forward drift and diffusion
        f, g = self.sde(u_t, t)

        # scale the score by 0.5 for the prob. flow formulation
        eps_pred = score_fn(u_t.type(torch.float32), t.type(torch.float32))
        if clip:
            eps_x, eps_m = self.clip_and_predict_eps(u_t, eps_pred, t[0])
            eps_pred = torch.cat([eps_x, eps_m], dim=1)
        score = self.get_score(u_t, 0, self.mm_0, eps_pred, t)
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
