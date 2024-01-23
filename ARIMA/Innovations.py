import torch as pt
from pyro.nn import PyroSample, PyroModule
from pyro.distributions import Normal, LogNormal, MultivariateNormal

class NormalInnovations(PyroModule):
    def __init__(self, sigma_prior_dist=LogNormal, sigma_prior_dist_params=dict(loc=0, scale=5)):
        super().__init__()
        self.sigma_prior = sigma_prior_dist(**sigma_prior_dist_params)
        self.sigma = PyroSample(self.sigma_prior)

    def shape(self, num_event_samples, shape=None):
        if shape is None: shape = list(self.sigma.shape) if len(self.sigma.shape) > 0 else [1]
        shape[min(-len(self.sigma_prior.event_shape), -1)] = num_event_samples
        return tuple(shape)

    def slice(self, event_samples_idx):
        indices = [Ellipsis] + [slice(None)] * max(1, len(self.sigma.shape))
        indices[min(-len(self.sigma_prior.event_shape), -1)] = event_samples_idx
        return indices

    def forward(self, num_samples):
        return Normal(                    pt.zeros(self.sigma.shape + (num_samples,)),
                      self.sigma[..., None].expand(self.sigma.shape + (num_samples,))).to_event(1)

class MultivariateNormalInnovations(PyroModule):
    def __init__(self, n, sqrt_cov_prior_dist=Normal, sqrt_cov_prior_dist_params=dict(loc=0, scale=5)):
        super().__init__()
        self.sqrt_cov_prior = sqrt_cov_prior_dist(**sqrt_cov_prior_dist_params).expand((n, n)).to_event(2)
        self.sqrt_cov = PyroSample(self.sqrt_cov_prior)

    def shape(self, num_event_samples, shape=None):
        if shape is None: shape = list(self.sqrt_cov.shape)
        shape[-len(self.sqrt_cov_prior.event_shape)] = num_event_samples
        return tuple(shape)

    def slice(self, event_samples_idx):
        indices = [Ellipsis] + [slice(None)] * len(self.sqrt_cov.shape)
        indices[-len(self.sqrt_cov_prior.event_shape)] = event_samples_idx
        return indices

    def forward(self, num_samples):
        cov = pt.matmul(self.sqrt_cov, pt.swapaxes(self.sqrt_cov, -1, -2))
        return MultivariateNormal(                   pt.zeros(cov.shape[:-2] + (num_samples, cov.shape[-1])),
                                  cov[..., None, :, :].expand(cov.shape[:-2] + (num_samples, cov.shape[-1], cov.shape[-1]))).to_event(1)
                                           