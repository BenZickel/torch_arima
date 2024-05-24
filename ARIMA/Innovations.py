import torch as pt
from pyro.nn import PyroSample, PyroModule
from pyro.distributions import Normal, LogNormal, MultivariateNormal, LKJCholesky, StableWithLogProb, Uniform


class StableInnovations(PyroModule):
    def __init__(self):
        super().__init__()
        self.sigma = PyroSample(LogNormal(loc=0, scale=5))
        self.stability = PyroSample(Uniform(0, 2))
        self.skew = PyroSample(Uniform(-1, 1))

    def shape(self, num_event_samples):
        return self.sigma.shape + (num_event_samples,)

    def slice(self, event_samples_idx):
        return (Ellipsis, event_samples_idx)

    def forward(self, num_samples):
        return StableWithLogProb(
                    self.stability[..., None].expand(self.stability.shape + (num_samples,)),
                    self.skew[..., None].expand(self.skew.shape + (num_samples,)),
                    self.sigma[..., None].expand(self.sigma.shape + (num_samples,)), 0).to_event(1)
    

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

class NormalInnovationsVector(PyroModule):
    def __init__(self, n, sigma_prior_dist=LogNormal, sigma_prior_dist_params=dict(loc=0, scale=5)):
        super().__init__()
        self.sigma_prior = sigma_prior_dist(**sigma_prior_dist_params).expand((n,)).to_event(1)
        self.sigma = PyroSample(self.sigma_prior)

    def shape(self, num_event_samples):
        return self.sigma.shape[:-1] + (num_event_samples, self.sigma.shape[-1])

    def slice(self, event_samples_idx):
        return (Ellipsis, event_samples_idx, slice(None))

    def forward(self, num_samples):
        return Normal(pt.zeros(self.sigma.shape[:-1] + (num_samples, self.sigma.shape[-1])),
                      self.sigma[..., None, :].expand(self.sigma.shape[:-1] +
                                                       (num_samples, self.sigma.shape[-1]))).to_event(2)

class MultivariateNormalInnovations(PyroModule):
    def __init__(self, n, sigma_prior_dist=LogNormal, sigma_prior_dist_params=dict(loc=0, scale=5)):
        super().__init__()
        self.sigma_prior = sigma_prior_dist(**sigma_prior_dist_params).expand((n,)).to_event(1)
        self.scale_diag = PyroSample(self.sigma_prior)
        self.scale_tril = PyroSample(LKJCholesky(n, 1))

    def shape(self, num_event_samples):
        return self.scale_tril.shape[:-2] + (num_event_samples, self.scale_tril.shape[-1])

    def slice(self, event_samples_idx):
        return (Ellipsis, event_samples_idx, slice(None))

    def forward(self, num_samples):
        sqrt_cov = self.scale_tril * self.scale_diag[..., None]
        return MultivariateNormal(pt.zeros(sqrt_cov.shape[:-2] + (num_samples, sqrt_cov.shape[-1])),
                                  scale_tril = sqrt_cov[..., None, :, :].expand(sqrt_cov.shape[:-2] +
                                                                                (num_samples, sqrt_cov.shape[-1], sqrt_cov.shape[-1]))).to_event(1)
                                           