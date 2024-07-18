
import torch
import pytest

from ARIMA.Innovations import NormalInnovations, NormalInnovationsVector, MultivariateNormalInnovations
from pyro import distributions as dist

@pytest.mark.parametrize("innovations, num_elements",
                         [(NormalInnovations,             None),
                          (NormalInnovationsVector,       5),
                          (MultivariateNormalInnovations, 5)])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.3])
def test_fixed_exp(innovations, num_elements, sigma):
    num_samples = 100000
    num_elements = dict() if num_elements is None else dict(n=num_elements)
    innovations = innovations(sigma_prior = dist.Delta(torch.tensor(sigma)), fixed_exp=True, **num_elements)
    samples = innovations(num_samples).sample()
    samples_exp_mean = samples.exp().mean(0)
    torch.testing.assert_close(samples_exp_mean, torch.ones(samples_exp_mean.shape), atol=0.01, rtol=0.01)
