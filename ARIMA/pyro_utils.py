import torch as pt
import pyro
import dill
from io import BytesIO
from pyro import distributions, poutine
from pyro.nn import PyroSample, PyroParam, PyroModule
from pyro.infer import Importance, Trace_ELBO
from pyro.infer.importance import vectorized_importance_weights
from typing import NamedTuple, Any
from functools import partial

ModuleList = PyroModule[pt.nn.ModuleList]

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
    model_trace: Any
    guide_trace: Any

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
        model_trace, guide_trace = None, None
    # Calculate observation probability
    log_num_samples = pt.log(pt.tensor(num_samples * 1.0))
    log_sum_weights = pt.logsumexp(log_weights, 0)
    log_norm_weights = log_weights - log_sum_weights
    effective_sample_size = pt.exp(-pt.logsumexp(2 * log_norm_weights, 0))
    obs_log_prob = log_sum_weights - log_num_samples
    return CalcObsLogProbResult(obs_log_prob, log_weights, effective_sample_size, model_trace, guide_trace)

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

# PyTorch load and save that supports loading and saving of inline lambda functions
load = partial(pt.load, pickle_module=dill)
save = partial(pt.save, pickle_module=dill)

# PyTorch clone that supports cloning of inline lambda functions
def clone(model):
    buffer = BytesIO()
    save(model, buffer)
    buffer.seek(0)
    return load(buffer)

class MixtureGuide(PyroModule):
    '''
    Create guide that is a weighed mixture of guides of the same type with independent parameters for each guide.
    '''
    def __init__(self, guide, model, n_components, args=tuple(), kwargs=dict(), init_fn=None):
        super().__init__()
        self.guide_list = ModuleList()
        for n_component in range(n_components):
            # Create mixture component
            self.guide_list.append(guide(model))
            # Initialize mixture component
            self.guide_list[-1](*args, **kwargs)
            if init_fn is not None:
                init_fn(self.guide_list[-1], n_component)

    def __call__(self, *args, **kwargs):
        alpha = pyro.param('alpha', pt.zeros(len(self.guide_list))).exp()
        # Select with mixture component to use
        idx = pyro.sample('idx', distributions.Categorical(alpha / sum(alpha)),
                           infer={'enumerate': 'sequential', 'is_auxiliary': True})
        # Sample selected mixture component
        self.guide_list[idx](*args, **kwargs)

def plate_log_prob_sum(trace, plate_name='num_particles_vectorized'):
    '''
    Get log probability sum from trace while keeping indexing over the specified plate.
    '''
    wd = trace.plate_to_symbol[plate_name]
    log_prob_sum = 0.0
    for site in trace.nodes.values():
        if site["type"] != "sample":
            continue
        log_prob_sum += pt.einsum(
            site["packed"]["log_prob"]._pyro_dims + "->" + wd,
            [site["packed"]["log_prob"]])
    return log_prob_sum

def weighed_quantile(input, probs, log_weights, dim=0):
    """
    Computes quantiles of weighed ``input`` samples at ``probs``.

    :param torch.Tensor input: the input tensor.
    :param list probs: quantile positions.
    :param torch.Tensor log_weights: sample weights tensor.
    :param int dim: dimension to take quantiles from ``input``.
    :returns torch.Tensor: quantiles of ``input`` at ``probs``.
    
    Example:
    >>> from ARIMA.pyro_utils import weighed_quantile
    >>> from torch import Tensor
    >>> input = Tensor([[10, 50, 40], [20, 30, 0]])
    >>> probs = Tensor([0.2, 0.8])
    >>> log_weights = Tensor([0.4, 0.5, 0.1]).log()
    >>> weighed_quantile(input, probs, log_weights, -1)
    tensor([[40.4000, 47.6000],
            [ 9.0000, 26.4000]])
    """
    dim = dim if dim >= 0 else (len(input.shape) + dim)
    if isinstance(probs, (list, tuple)):
        probs = pt.tensor(probs, dtype=input.dtype, device=input.device)
    # Calculate normalized weights
    weights = (log_weights - pt.logsumexp(log_weights, 0)).exp()
    # Sort input and weights
    sorted_input, sorting_indices = input.sort(dim)
    weights = weights[sorting_indices].cumsum(dim)
    # Scale weights to be between zero and one
    weights = weights - weights.min(dim, keepdim=True)[0]
    weights = weights / weights.max(dim, keepdim=True)[0]
    # Calculate indices
    indices_above = (weights[..., None] <= probs).sum(dim, keepdim=True).swapaxes(dim, -1).clamp(max=input.size(dim) - 1)[..., 0]
    indices_below = (indices_above - 1).clamp(min=0)
    # Calculate below and above qunatiles
    quantiles_below = sorted_input.gather(dim, indices_below)
    quantiles_above = sorted_input.gather(dim, indices_above)
    # Calculate weights for below and above quantiles
    probs_shape = [None] * len(input.shape)
    probs_shape[dim] = slice(None)
    expanded_probs_shape = list(input.shape)
    expanded_probs_shape[dim] = len(probs)
    probs = probs[probs_shape].expand(*expanded_probs_shape)
    weights_below = weights.gather(dim, indices_below)
    weights_above = weights.gather(dim, indices_above)
    weights_below = (weights_above - probs) / (weights_above - weights_below)
    weights_above = 1 - weights_below
    # Return quantiles
    return weights_below * quantiles_below + weights_above * quantiles_above

if __name__ == '__main__':
    import doctest
    doctest.testmod()
