import torch as pt
from pyro.nn import PyroSample, PyroModule
from pyro.distributions import Normal, LogNormal, MultivariateNormal, LKJCholesky


def IndexIndependent(Innovations):
    '''
    Mark input class as generating innovations which are independent of their indices.
    '''
    class IndexIndependentInnovations(Innovations):
        def forward(self, samples_idx):
            try:
                num_samples = len(samples_idx)
            except TypeError:
                num_samples = samples_idx
            return super().forward(num_samples)
    return IndexIndependentInnovations


@IndexIndependent
class NormalInnovations(PyroModule):
    def __init__(self, sigma_prior=LogNormal(loc=0, scale=5), fixed_exp=False):
        super().__init__()
        self.sigma = PyroSample(sigma_prior)
        self.fixed_exp = fixed_exp

    def shape(self, num_event_samples):
        return self.sigma.shape + (num_event_samples,)

    def slice(self, event_samples_idx):
        return (Ellipsis, event_samples_idx)

    def forward(self, num_samples):
        sigma = self.sigma[..., None].expand(self.sigma.shape + (num_samples,))
        return Normal(-0.5 * sigma * sigma
                      if self.fixed_exp else
                      pt.zeros_like(sigma), sigma).to_event(1)


@IndexIndependent
class NormalInnovationsVector(PyroModule):
    def __init__(self, n, sigma_prior=LogNormal(loc=0, scale=5), fixed_exp=False):
        super().__init__()
        self.sigma = PyroSample(sigma_prior.expand((n,)).to_event(1))
        self.fixed_exp = fixed_exp

    def shape(self, num_event_samples):
        return self.sigma.shape[:-1] + (num_event_samples, self.sigma.shape[-1])

    def slice(self, event_samples_idx):
        return (Ellipsis, event_samples_idx, slice(None))

    def forward(self, num_samples):
        sigma = self.sigma[..., None, :].expand(self.sigma.shape[:-1] + (num_samples, self.sigma.shape[-1]))
        return Normal(-0.5 * sigma * sigma
                      if self.fixed_exp else
                      pt.zeros_like(sigma), sigma).to_event(2)


@IndexIndependent
class MultivariateNormalInnovations(PyroModule):
    def __init__(self, n, sigma_prior=LogNormal(loc=0, scale=5), fixed_exp=False):
        super().__init__()
        self.scale_diag = PyroSample(sigma_prior.expand((n,)).to_event(1))
        self.scale_tril = PyroSample(LKJCholesky(n, 1))
        self.fixed_exp = fixed_exp

    def shape(self, num_event_samples):
        return self.scale_tril.shape[:-2] + (num_event_samples, self.scale_tril.shape[-1])

    def slice(self, event_samples_idx):
        return (Ellipsis, event_samples_idx, slice(None))

    def forward(self, num_samples):
        sqrt_cov = self.scale_tril * self.scale_diag[..., None]
        scale_tril = sqrt_cov[..., None, :, :].expand(sqrt_cov.shape[:-2] + 
                                                      (num_samples, sqrt_cov.shape[-1], sqrt_cov.shape[-1]))
        sigma = self.scale_diag[..., None, :].expand(sqrt_cov.shape[:-2] + 
                                                     (num_samples, sqrt_cov.shape[-1]))
        return MultivariateNormal(-0.5 * sigma * sigma
                                  if self.fixed_exp else
                                  pt.zeros_like(sigma), scale_tril = scale_tril).to_event(1)
