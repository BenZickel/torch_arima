import pyro
import torch as pt
from pyro.distributions import Normal, LogNormal, TransformedDistribution
from pyro.nn import PyroSample
from ARIMA.TimeSeries import ARIMA
from ARIMA.pyro_utils import make_params_pyro

ARIMA = pyro.nn.PyroModule[ARIMA]

class BayesianARIMA(ARIMA):
    def __init__(self, *args, obs_idx, predict_idx, sigma_prior_dist=LogNormal, sigma_prior_dist_params=dict(loc=0, scale=5), **kwargs):
        super().__init__(*args, **kwargs)
        # Creating latent variables
        make_params_pyro(self)
        self.sigma = PyroSample(sigma_prior_dist(**sigma_prior_dist_params))
        self.predict_innovations = PyroSample(lambda self: self.innovations_dist(len(self.predict_idx)))
        # Validate and store indices
        self.obs_idx = [*obs_idx]
        self.predict_idx = [*predict_idx]
        if set(self.obs_idx).union(set(self.predict_idx)) != \
           set(range(len(self.obs_idx) + len(self.predict_idx))):
            raise UserWarning('Indices of observations and predictions must be complementary.')

    def innovations_dist(self, num_samples):
        return Normal(                    pt.zeros(self.sigma.shape + (num_samples,)),
                      self.sigma[..., None].expand(self.sigma.shape + (num_samples,))).to_event(1)

    def innovations(self):
        # Build innovations vector 
        innovations = pt.zeros(self.predict_innovations.shape[:-1] + (len(self.obs_idx) + len(self.predict_idx),))
        innovations[..., self.predict_idx] = self.predict_innovations
        is_innovation = innovations.bool()
        is_innovation[..., self.predict_idx] = True
        is_innovation[..., self.obs_idx] = False
        return innovations, is_innovation

    def observations_dist(self):
        combined, is_innovation = self.innovations()
        transform = self.get_transform(x=combined, idx=self.obs_idx)
        return TransformedDistribution(self.innovations_dist(len(self.obs_idx)), [transform])

    def forward(self, observations=None):
        if observations is not None:
            self.observations = pyro.sample('observations', self.observations_dist(), obs=observations)
        else:
            # Return predictions
            combined, is_innovation = self.innovations()
            combined[..., self.obs_idx] = self.observations
            transform = self.get_transform(x=combined, idx=self.predict_idx, x_is_in_not_out=is_innovation)
            combined[..., self.predict_idx] = transform(combined[..., self.predict_idx])
            return combined
