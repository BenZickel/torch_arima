import pyro
import torch as pt
from pyro.nn import PyroSample, PyroParam
from pyro.infer import Importance, Trace_ELBO
from pyro.infer.importance import vectorized_importance_weights
from typing import NamedTuple, Any

def make_params_pyro(module, sample_names=True, dist_class=pyro.distributions.Normal, dist_params=dict(loc=0, scale=5)):
    '''
    Replace all module parameters with Pyro parameters
    '''
    # In place conversion of all child modules to PyroModule
    for child_module in module.children():
        pyro.nn.module.to_pyro_module_(child_module)
    # Convert all parameters to Pyro parameters with priors
    parameters = [*module.parameters()]
    pyro_parameters = [PyroSample(dist_class(**dist_params).expand(param.shape).to_event(len(param.shape)))
                       if sample_names is True or name in sample_names else         
                       PyroParam(pt.tensor(param)) for name, param in module.named_parameters()]
    update_list = []
    for child_module in module.modules():
        for name, param in child_module.named_parameters(recurse=False):
            idx = min([n for n, p in enumerate(parameters) if p is param])
            update_list.append((child_module, name, pyro_parameters[idx]))
    for child_module, name, pyro_param in update_list:
        delattr(child_module, name)
        setattr(child_module, name, pyro_param)

class CalcObsLogProbResult(NamedTuple):
    obs_log_prob: Any
    log_weights: Any

def calc_obs_log_prob(model, guide, num_samples, args=tuple(), kwargs=dict(), vectorize=False):
    # Do sampling
    if vectorize:
        # Assumes static model structure
        trace_elbo = Trace_ELBO()
        trace_elbo._guess_max_plate_nesting(model, guide, args, kwargs)
        log_weights, model_trace, guide_trace = vectorized_importance_weights(model, guide, *args,
                                                                              num_samples=num_samples,
                                                                              max_plate_nesting=trace_elbo.max_plate_nesting, **kwargs)
    else:
        importance_sampler = Importance(model, guide, num_samples=num_samples)
        importance_sampler.run(*args, **kwargs)
        log_weights = pt.tensor(importance_sampler.log_weights)
    # Calculate observation probability
    log_num_samples = pt.log(pt.tensor(num_samples * 1.0))
    obs_log_prob = pt.logsumexp(log_weights - log_num_samples, 0)
    return CalcObsLogProbResult(obs_log_prob, log_weights)
