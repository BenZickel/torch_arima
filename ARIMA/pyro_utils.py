import torch as pt
import pyro
import dill
from io import BytesIO
from pyro import distributions, poutine
from pyro.nn import PyroSample, PyroParam
from pyro.infer import Importance, Trace_ELBO
from pyro.infer.importance import vectorized_importance_weights
from typing import NamedTuple, Any
from functools import partial

def make_params_pyro(module, param_names=[], dist_class=distributions.Normal, dist_params=dict(loc=0, scale=5)):
    '''
    Replace all module parameters with Pyro parameters
    '''
    # In place conversion of all child modules to PyroModule
    for child_module in module.children():
        pyro.nn.module.to_pyro_module_(child_module)
    # Convert all parameters to Pyro parameters with priors
    parameters = [*module.parameters()]
    pyro_parameters = [PyroSample(dist_class(**dist_params).expand(param.shape).to_event(len(param.shape)))
                       if name not in param_names else         
                       PyroParam(param) for name, param in module.named_parameters()]
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
    effective_sample_size: Any

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
    log_sum_weights = pt.logsumexp(log_weights, 0)
    log_norm_weights = log_weights - log_sum_weights
    effective_sample_size = pt.exp(-pt.logsumexp(2 * log_norm_weights, 0))
    obs_log_prob = log_sum_weights - log_num_samples
    return CalcObsLogProbResult(obs_log_prob, log_weights, effective_sample_size)

class ModifiedModelGuide:
    '''
    Class for creating a guide whose same name but different shape model nodes are taken from the model.
    
    Args:
        model: Callable.
        guide: Callable.
        args: Positional arguments of the model.
        kwargs: Keyword arguments of the model.

    Returns:
        Guide as a callable class instance.
    '''
    def __init__(self, model, guide, args=tuple(), kwargs=dict()):
        # Get guide and model nodes
        model_trace = poutine.trace(model).get_trace(*args, **kwargs)
        guide_trace = poutine.trace(guide).get_trace()
        # Create guide with same name but different shape model nodes taken from the model
        guide_block_nodes = set(node for node in guide_trace.nodes if
                                node in model_trace.nodes and
                                'value' in guide_trace.nodes[node] and
                                hasattr(guide_trace.nodes[node]['value'], 'shape') and
                                guide_trace.nodes[node]['value'].shape != model_trace.nodes[node]['value'].shape)
        model_block_nodes = set(node for node in model_trace.nodes if
                                node not in guide_trace.nodes)
        def modified_model_guide(*ignore_args, **ignore_kwargs):
            trace = poutine.trace(poutine.block(guide, hide=guide_block_nodes)).get_trace()
            return poutine.block(poutine.replay(model, trace=trace), hide=set(trace.nodes).union(model_block_nodes))(*args, **kwargs)
        self.model = model
        self.guide = guide
        self.guide_block_nodes = guide_block_nodes
        self.model_block_nodes = model_block_nodes
        self.modified_model_guide = modified_model_guide

    def __call__(self, *args, **kwargs):
        return self.modified_model_guide(*args, **kwargs)

load = partial(pt.load, pickle_module=dill)
save = partial(pt.save, pickle_module=dill)

def clone(model):
    buffer = BytesIO()
    save(model, buffer)
    buffer.seek(0)
    return load(buffer)
