import torch

from util import get_module, register_module, reshape
from .base import Sampler


@register_module(category="samplers", name="em_sde")
class EulerMaruyamaSampler(Sampler):
    def __init__(self, config, sde, score_fn, corrector_fn=None):
        super().__init__(config, sde, score_fn, corrector_fn=corrector_fn)

    def predictor_update_fn(self, x, t, dt):
        f, g = self.sde.reverse_sde(
            x,
            t * torch.ones(x.shape[0], device=x.device, dtype=torch.float64),
            self.score_fn,
            probability_flow=False,
        )
        x_mean = x + f * dt
        noise = g * torch.sqrt(dt) * torch.randn_like(x)
        x = x_mean + noise
        return x, x_mean

    def denoising_fn(self, x, t, dt):
        f, _ = self.sde.reverse_sde(
            x,
            t * torch.ones(x.shape[0], device=x.device, dtype=torch.float64),
            self.score_fn,
            probability_flow=False,
        )
        x_mean = x + f * dt
        # noise = g * torch.sqrt(dt) * torch.randn_like(x)
        # x = x_mean + noise
        return x_mean

    def sample(self, batch, ts, n_discrete_steps, denoise=True, eps=1e-3):
        x = batch
        self.nfe = n_discrete_steps

        # Sample
        with torch.no_grad():
            for t_idx in range(n_discrete_steps):
                dt = reshape(ts[t_idx + 1] - ts[t_idx], x)
                # Predictor step
                x, _ = self.predictor_update_fn(x, ts[t_idx], dt)

                # Corrector_step
                x, _ = self.corrector_update_fn(x, ts[t_idx], dt)

            if denoise:
                x = self.denoising_fn(
                    x,
                    torch.tensor(self.sde.T - eps, device=x.device),
                    reshape(torch.tensor(eps, device=x.device), x),
                )
        return x


@register_module(category="samplers", name="sscs_sde")
class SSCSSampler(Sampler):
    def __init__(self, config, sde, score_fn, corrector_fn=None):
        super().__init__(config, sde, score_fn, corrector_fn=corrector_fn)

    def _mean(self, u, t, dt):
        # Chunk
        x, m = torch.chunk(u, 2, dim=1)

        # Compute \int \beta_t
        db_t = self.sde.b_t(self.sde.T - (t + dt)) - self.sde.b_t(self.sde.T - t)
        db_t = reshape(db_t, u)

        # Scaling constants
        mu_lam = (self.sde.nu + self.sde.gamma) / 4
        scaling_factor = torch.exp(mu_lam * db_t)

        # Soln coefficients
        A_1 = (self.sde.nu - self.sde.gamma) / 4
        A_2 = -((self.sde.gamma - self.sde.nu) ** 2) / 8
        C_1 = 0.5
        C_2 = (self.sde.gamma - self.sde.nu) / 4

        # Individual means
        mu_x = -A_1 * x * db_t - A_2 * m * db_t + x
        assert mu_x.shape == x.shape
        mu_m = -C_1 * x * db_t - C_2 * m * db_t + m
        assert mu_m.shape == m.shape

        # Scaling
        mu = torch.cat([mu_x, mu_m], dim=1)
        mu = mu * reshape(scaling_factor, mu)
        return mu

    def _var(self, t, dt):
        # Compute \int \beta_t
        db_t = self.sde.b_t(self.sde.T - (t + dt)) - self.sde.b_t(self.sde.T - t)
        db_t2 = db_t**2

        # Scaling constants
        cov_lam = (self.sde.nu + self.sde.gamma) / 2
        scaling_factor = torch.exp(cov_lam * db_t)
        inv_scaling_factor = torch.exp(-cov_lam * db_t)

        # NOTE: SSCS always uses a 0 initial cov for both data and momentum
        # since they are just samples from the previous step so terms involving
        # them are automatically zeroed and not defined here
        A_5, A_6 = (-self.sde.m_inv / 2, (self.sde.gamma - self.sde.nu) / 2)
        A_7 = inv_scaling_factor - 1
        C_5 = (self.sde.gamma - self.sde.nu) / 4
        D_5, D_6 = (-1 / 2, self.sde.m * (self.sde.nu - self.sde.gamma) / 2)
        D_7 = self.sde.m * (inv_scaling_factor - 1)

        # Individual covariances
        xx_t = (A_5 * db_t2 - A_6 * db_t + A_7) * scaling_factor
        xm_t = (C_5 * db_t2) * scaling_factor
        mm_t = (D_5 * db_t2 - D_6 * db_t + D_7) * scaling_factor

        assert xx_t.shape == t.shape
        assert xm_t.shape == t.shape
        assert mm_t.shape == t.shape
        return xx_t + self.sde.eps, xm_t, mm_t + self.sde.eps

    def analytical_dynamics(self, u, t, dt):
        # Mean and variance
        mu = self._mean(u, t, dt)
        var = self._var(t, dt)

        # Coefficients
        c11, c12, c21, c22 = self.sde.get_coeff(var)

        # Sample noise
        eps = torch.randn_like(u)
        eps_x, eps_m = torch.chunk(eps, 2, dim=1)

        noise_x = reshape(c11, eps_x) * eps_x + reshape(c12, eps_m) * eps_m
        noise_m = reshape(c21, eps_x) * eps_x + reshape(c22, eps_m) * eps_m
        noise = torch.cat((noise_x, noise_m), dim=1)

        # Perturb
        u_t = mu + noise
        return u_t

    def euler_score_dynamics(self, u, t, dt):
        t = self.sde.T - t
        beta_t = reshape(self.sde.beta_t(t), u)
        x, m = torch.chunk(u, 2, dim=1)

        # Predict score
        eps_pred = self.score_fn(u.type(torch.float32), t.type(torch.float32))
        score = self.sde.get_score(u, 0, self.sde.mm_0, eps_pred, t)
        score_x, score_m = torch.chunk(score, 2, dim=1)

        # Euler update
        x_bar = x + dt * self.sde.gamma * beta_t * (score_x + x)
        m_bar = m + dt * self.sde.m * self.sde.nu * beta_t * (
            score_m + self.sde.m_inv * m
        )
        return torch.cat((x_bar, m_bar), dim=1)

    def predictor_update_fn(self, u, t, dt):
        t = t * torch.ones(u.shape[0], device=u.device, dtype=torch.float64)
        u = self.analytical_dynamics(u, t, dt / 2)
        u = self.euler_score_dynamics(u, t, dt)
        u = self.analytical_dynamics(u, t, dt / 2)
        return u

    def denoising_fn(self, x, t, dt):
        f, g = self.sde.reverse_sde(
            x,
            t * torch.ones(x.shape[0], device=x.device, dtype=torch.float64),
            self.score_fn,
            probability_flow=False,
        )
        x_mean = x + f * dt
        noise = g * torch.sqrt(dt) * torch.randn_like(x)
        x = x_mean + noise
        return x_mean

    def sample(self, batch, ts, n_discrete_steps, denoise=True, eps=1e-3):
        x = batch
        self.nfe = n_discrete_steps

        # Sample
        with torch.no_grad():
            for t_idx in range(n_discrete_steps):
                dt = ts[t_idx + 1] - ts[t_idx]
                # Predictor step
                x = self.predictor_update_fn(x, ts[t_idx], dt)

                # Corrector_step
                x, _ = self.corrector_update_fn(x, ts[t_idx], dt)

            if denoise:
                x = self.denoising_fn(
                    x,
                    torch.tensor(self.sde.T - eps, device=x.device),
                    reshape(torch.tensor(eps, device=x.device), x),
                )
        return x


@register_module(category="samplers", name="msscs_sde")
class ModifiedSSCSSampler(Sampler):
    """Sampler based on an alternate splitting scheme
    """
    def __init__(self, config, sde, score_fn, corrector_fn=None):
        super().__init__(config, sde, score_fn, corrector_fn=corrector_fn)

    def analytical_dynamics(self, u, t, dt):
        pass

    def euler_score_dynamics(self, u, t, dt):
        pass

    def predictor_update_fn(self, u, t, dt):
        t = t * torch.ones(u.shape[0], device=u.device, dtype=torch.float64)
        u = self.analytical_dynamics(u, t, dt / 2)
        u = self.euler_score_dynamics(u, t, dt)
        u = self.analytical_dynamics(u, t, dt / 2)
        return u

    def denoising_fn(self, x, t, dt):
        f, g = self.sde.reverse_sde(
            x,
            t * torch.ones(x.shape[0], device=x.device, dtype=torch.float64),
            self.score_fn,
            probability_flow=False,
        )
        x_mean = x + f * dt
        noise = g * torch.sqrt(dt) * torch.randn_like(x)
        x = x_mean + noise
        return x_mean

    def sample(self, batch, ts, n_discrete_steps, denoise=True, eps=1e-3):
        x = batch
        self.nfe = n_discrete_steps

        # Sample
        with torch.no_grad():
            for t_idx in range(n_discrete_steps):
                dt = ts[t_idx + 1] - ts[t_idx]
                # Predictor step
                x = self.predictor_update_fn(x, ts[t_idx], dt)

                # Corrector_step
                x, _ = self.corrector_update_fn(x, ts[t_idx], dt)

            if denoise:
                x = self.denoising_fn(
                    x,
                    torch.tensor(self.sde.T - eps, device=x.device),
                    reshape(torch.tensor(eps, device=x.device), x),
                )
        return x
