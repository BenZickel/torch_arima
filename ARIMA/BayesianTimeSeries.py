import pyro
import torch as pt
from pyro.distributions import TransformedDistribution
from pyro.nn import PyroSample, PyroModule, pyro_method
from ARIMA.TimeSeries import ARIMA, VARIMA
from ARIMA.pyro_utils import make_params_pyro
from ARIMA.Innovations import NormalInnovations, MultivariateNormalInnovations

ARIMA = PyroModule[ARIMA]
VARIMA = PyroModule[VARIMA]

def BayesianARIMA(*args, obs_idx, predict_idx, innovations=NormalInnovations, **kwargs):
    return BayesianTimeSeries(ARIMA(*args, **kwargs), innovations(), obs_idx, predict_idx)

def BayesianVARIMA(*args, n, obs_idx, predict_idx, innovations=MultivariateNormalInnovations, **kwargs):
    '''
    Examples:
        >>> from ARIMA import BayesianVARIMA
        >>> import torch as pt
        >>> import pyro
        >>> obs_idx = [*range(10)]
        >>> predict_idx = [*range(10,17)]
        >>> n = 5
        >>> model = BayesianVARIMA(3, 0, 1, 0, 1, 2, 12, n=n, obs_idx=obs_idx, predict_idx=predict_idx)
        >>> with pyro.poutine.trace(): ret_val = model.predict(pt.zeros(len(obs_idx), n))
        >>> ret_val.shape
        torch.Size([17, 5])
    '''
    return BayesianTimeSeries(VARIMA([ARIMA(*args, **kwargs) for i in range(n)]), innovations(n), obs_idx, predict_idx)

class BayesianTimeSeries(PyroModule):
    def __init__(self, model, innovations, obs_idx, predict_idx):
        super().__init__()
        self.model = model
        # Creating model latent variables
        make_params_pyro(self)
        self.innovations_dist = innovations
        self.set_indices(obs_idx, predict_idx)

    def set_indices(self, obs_idx, predict_idx):
        # Validate and store indices
        self.obs_idx = [*obs_idx]
        self.predict_idx = [*predict_idx]
        if set(self.obs_idx).union(set(self.predict_idx)) != \
           set(range(len(self.obs_idx) + len(self.predict_idx))):
            raise UserWarning('Indices of observations and predictions must be complementary.')
        # Create innovations
        self.predict_innovations = PyroSample(lambda self: self.innovations_dist(len(self.predict_idx)))
        # Create observations
        self.observations = PyroSample(lambda self: self.observations_dist())
        return self

    def innovations(self):
        # Build innovations vector
        innovations = pt.empty(self.innovations_dist.shape(len(self.obs_idx) + len(self.predict_idx))).fill_(pt.nan)
        innovations[self.innovations_dist.slice(self.predict_idx)] = self.predict_innovations
        is_innovation = pt.zeros(len(self.obs_idx) + len(self.predict_idx), dtype=pt.bool)
        is_innovation[self.predict_idx] = True
        return innovations, is_innovation

    def observations_dist(self):
        combined, is_innovation = self.innovations()
        transform = self.model.get_transform(x=combined, idx=self.obs_idx)
        return TransformedDistribution(self.innovations_dist(len(self.obs_idx)), [transform])

    def forward(self):
        # Return sampled observations
        return self.observations
        
    @pyro_method
    def predict(self):
        # Return predictions combined with observations
        combined, is_innovation = self.innovations()
        combined[self.innovations_dist.slice(self.obs_idx)] = self.observations
        transform = self.model.get_transform(x=combined, idx=self.predict_idx, x_is_in_not_out=is_innovation)
        combined[self.innovations_dist.slice(self.predict_idx)] = transform(combined[self.innovations_dist.slice(self.predict_idx)])
        return combined

if __name__ == '__main__':
    import doctest
    doctest.testmod()
