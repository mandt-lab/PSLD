import torch

from torchdiffeq import odeint
from .base import Sampler
from util import register_module


@register_module(category="samplers", name="bb_ode")
class BBODESampler(Sampler):
    def __init__(self, config, sde, score_fn, mode="reverse"):
        super().__init__(config, sde, score_fn)
        self.mode = mode
        self.nfe = 0
        self.rtol = config.evaluation.sampler.rtol
        self.atol = config.evaluation.sampler.atol
        self.method = config.evaluation.sampler.method
        self.solver_opts = config.evaluation.sampler.solver_opts

    def predictor_update_fn(self, x, t, dt):
        f, g = self.sde.reverse_sde(x, t, self.score_fn, probability_flow=False)
        x_mean = x + f * dt
        noise = g * torch.sqrt(dt) * torch.randn_like(x)
        x = x_mean + noise
        return x, x_mean

    def sample(self, batch, ts, n_discrete_steps, denoise=True, eps=1e-3):
        def ode_fn(t, x):
            self.nfe += 1
            vec_t = torch.ones(x.shape[0], device=x.device, dtype=torch.float64) * t
            f, _ = self.sde.reverse_sde(x, vec_t, self.score_fn, probability_flow=True)
            return f

        x = batch

        time_tensor = torch.tensor(
            [0.0, self.sde.T - eps], dtype=torch.float64, device=x.device
        )
        # BB-ODE solver
        solution = odeint(
            ode_fn,
            x,
            time_tensor,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            options=self.solver_opts,
        )

        x = solution[-1]

        if denoise:
            _, x = self.predictor_update_fn(x, self.sde.T - eps, eps)
        return x
