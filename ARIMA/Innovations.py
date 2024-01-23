import torch as pt
from pyro.nn import PyroSample, PyroModule
from pyro.distributions import Normal, LogNormal, MultivariateNormal

class NormalInnovations(PyroModule):
    def __init__(self, sigma_prior_dist=LogNormal, sigma_prior_dist_params=dict(loc=0, scale=5)):
        super().__init__()
        self.sigma = PyroSample(sigma_prior_dist(**sigma_prior_dist_params))

    def forward(self, num_samples):
        return Normal(                    pt.zeros(self.sigma.shape + (num_samples,)),
                      self.sigma[..., None].expand(self.sigma.shape + (num_samples,))).to_event(1)

class MultivariateNormalInnovations(PyroModule):
    def __init__(self, n, sqrt_cov_prior_dist=Normal, sqrt_cov_prior_dist_params=dict(loc=0, scale=5)):
        super().__init__()
        self.sqrt_cov = PyroSample(sqrt_cov_prior_dist(**sqrt_cov_prior_dist_params).expand((n, n)).to_event(2))

    def forward(self, num_samples):
        cov = pt.matmul(self.sqrt_cov, pt.swapaxes(self.sqrt_cov, -1, -2))
        return MultivariateNormal(                   pt.zeros(cov.shape[:-2] + (num_samples, cov.shape[-1])),
                                  cov[..., None, :, :].expand(cov.shape[:-2] + (num_samples, cov.shape[-1], cov.shape[-1]))).to_event(1)
                                           