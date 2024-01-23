import torch as pt
from pyro.nn import PyroSample, PyroModule
from pyro.distributions import Normal, LogNormal, MultivariateNormal

class NormalInnovations(PyroModule):
    def __init__(self, sigma_prior_dist=LogNormal, sigma_prior_dist_params=dict(loc=0, scale=5)):
        super().__init__()
        self.sigma_prior = sigma_prior_dist(**sigma_prior_dist_params)
        self.sigma = PyroSample(self.sigma_prior)

    def shape(self, num_event_samples):
        return self.sigma.shape + (num_event_samples,)

    def slice(self, event_samples_idx):
        return (Ellipsis, event_samples_idx)

    def forward(self, num_samples):
        return Normal(                    pt.zeros(self.sigma.shape + (num_samples,)),
                      self.sigma[..., None].expand(self.sigma.shape + (num_samples,))).to_event(1)

class MultivariateNormalInnovations(PyroModule):
    def __init__(self, n, sqrt_cov_prior_dist=Normal, sqrt_cov_prior_dist_params=dict(loc=0, scale=5)):
        super().__init__()
        self.sqrt_cov_prior = sqrt_cov_prior_dist(**sqrt_cov_prior_dist_params).expand((n, n)).to_event(2)
        self.sqrt_cov = PyroSample(self.sqrt_cov_prior)

    def shape(self, num_event_samples):
        return self.sqrt_cov.shape[:-2] + (num_event_samples, self.sqrt_cov.shape[-1])

    def slice(self, event_samples_idx):
        return (Ellipsis, event_samples_idx, slice(None))

    def forward(self, num_samples):
        cov = pt.matmul(self.sqrt_cov, pt.swapaxes(self.sqrt_cov, -1, -2))
        return MultivariateNormal(                   pt.zeros(cov.shape[:-2] + (num_samples, cov.shape[-1])),
                                  cov[..., None, :, :].expand(cov.shape[:-2] + (num_samples, cov.shape[-1], cov.shape[-1]))).to_event(1)
                                           