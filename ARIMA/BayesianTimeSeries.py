import torch as pt
from pyro.distributions import TransformedDistribution
from pyro.nn import PyroSample, PyroModule, PyroModuleList, pyro_method
from . import TimeSeries
from .pyro_utils import make_params_pyro
from .Innovations import NormalInnovations, MultivariateNormalInnovations

ARIMA = PyroModule[TimeSeries.ARIMA]
class VARIMA(TimeSeries.VARIMATransform, PyroModule):
    def __init__(self, arimas):
        super().__init__()
        self.arimas = PyroModuleList(arimas)

def BayesianARIMA(*args, obs_idx, predict_idx, innovations=NormalInnovations, double=False, **kwargs):
    model = ARIMA(*args, **kwargs)
    if double:
        model.double()
    innovations = innovations()
    if isinstance(model.i_tail, pt.nn.Parameter):
        # Convert input tail parameter to the innovations distribution
        tail_len = model.i_tail.shape[-1]
        delattr(model, 'i_tail')
        model.i_tail = PyroSample(lambda self: innovations([*range(-tail_len, 0)]))
    # Create remaining model latent variables
    make_params_pyro(model)
    return BayesianTimeSeries(model, innovations, obs_idx, predict_idx)

def BayesianVARIMA(*args, n, obs_idx, predict_idx, innovations=MultivariateNormalInnovations, double=False, **kwargs):
    '''
    Examples:
        >>> from ARIMA import BayesianVARIMA
        >>> import torch as pt
        >>> import pyro
        >>> obs_idx = [*range(10)]
        >>> predict_idx = [*range(10,17)]
        >>> n = 5
        >>> model = BayesianVARIMA(3, 0, 1, 0, 1, 2, 12, n=n, obs_idx=obs_idx, predict_idx=predict_idx)
        >>> with pyro.poutine.trace(): ret_val = model.predict()
        >>> ret_val.shape
        torch.Size([17, 5])
    '''
    model = VARIMA([ARIMA(*args, **kwargs) for i in range(n)])
    if double:
        model.double()
    innovations = innovations(n)
    # Create remaining model latent variables
    make_params_pyro(model)
    return BayesianTimeSeries(model, innovations, obs_idx, predict_idx)

class BayesianTimeSeries(PyroModule):
    def __init__(self, model, innovations, obs_idx, predict_idx):
        super().__init__()
        self.model = model
        self.innovations_dist = innovations
        self.set_indices(obs_idx, predict_idx)

    def set_indices(self, obs_idx, predict_idx):
        # Validate and store indices
        self.obs_idx = [*obs_idx]
        self.predict_idx = [*predict_idx]
        if set(self.obs_idx).union(set(self.predict_idx)) != \
           set(range(len(self))):
            raise UserWarning('Indices of observations and predictions must be complementary.')
        # Create innovations
        self.predict_innovations = PyroSample(lambda self: self.innovations_dist(self.predict_idx))
        # Create observations
        self.observations = PyroSample(lambda self: self.observations_dist())
        return self

    def __len__(self):
        return len(self.obs_idx) + len(self.predict_idx)

    def innovations(self):
        # Build innovations vector
        innovations = self.predict_innovations.new_empty(self.innovations_dist.shape(len(self))).fill_(pt.nan)
        innovations[self.innovations_dist.slice(self.predict_idx)] = self.predict_innovations
        is_innovation = pt.zeros(len(self), dtype=pt.bool)
        is_innovation[self.predict_idx] = True
        return innovations, is_innovation

    def observations_dist(self):
        combined, is_innovation = self.innovations()
        transform = self.model.get_transform(x=combined, idx=self.obs_idx)
        return TransformedDistribution(self.innovations_dist(self.obs_idx), [transform])

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
