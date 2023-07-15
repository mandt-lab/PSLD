import abc


class Sampler(abc.ABC):
    """The abstract class for a Sampler algorithm."""

    def __init__(self, config, sde, score_fn, corrector_fn=None):
        super().__init__()
        self.config = config
        self.sde = sde
        self.score_fn = score_fn
        self.corrector_fn = corrector_fn

    @property
    def n_steps(self):
        return self.config.evaluation.n_discrete_steps

    @abc.abstractmethod
    def predictor_update_fn(self):
        raise NotImplementedError

    def corrector_update_fn(self, x, t, dt):
        if self.corrector_fn is not None:
            return self.corrector_fn(x, t, dt)

        # If no corrector is defined, Identity is default
        return x, x

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError
