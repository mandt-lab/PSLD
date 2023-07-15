import torch
from torchdiffeq import odeint
from util import register_module

from .base import Sampler


@register_module(category="samplers", name="bb_ode")
class BBODESampler(Sampler):
    """Black-Box ODE sampler for generating samples from the
    Probability Flow ODE.
    """

    def __init__(self, config, sde, score_fn, corrector_fn=None):
        super().__init__(config, sde, score_fn, corrector_fn=corrector_fn)
        self.nfe = 0
        self.rtol = config.evaluation.sampler.rtol
        self.atol = config.evaluation.sampler.atol
        self.solver_opts = {"solver": config.evaluation.sampler.solver}

        self._counter = 0

    @property
    def n_steps(self):
        return self.nfe

    @property
    def mean_nfe(self):
        if self._counter != 0:
            return self.nfe / self._counter
        raise ValueError("Run .sample() to compute mean_nfe")

    def predictor_update_fn(self, x, t, dt):
        pass

    def denoise_fn(self, x, t, dt):
        f, _ = self.sde.reverse_sde(x, t, self.score_fn, probability_flow=True)
        x_mean = x + f * dt
        return x_mean

    def sample(self, batch, ts, n_discrete_steps, denoise=True, eps=1e-3):
        def ode_fn(t, x):
            self.nfe += 1
            vec_t = torch.ones(x.shape[0], device=x.device, dtype=torch.float64) * t
            f, _ = self.sde.reverse_sde(x, vec_t, self.score_fn, probability_flow=True)
            return f

        x = batch
        self._counter += 1

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
            method="scipy_solver",
            options=self.solver_opts,
        )

        x = solution[-1]

        # Denoise
        if denoise:
            x = self.denoise_fn(
                x,
                torch.ones(x.shape[0], device=x.device, dtype=torch.float64)
                * (self.sde.T - eps),
                eps,
            )
            self.nfe += 1
        return x
