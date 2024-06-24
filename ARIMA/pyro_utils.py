import torch as pt
import pyro
import dill
from io import BytesIO
from pyro import distributions, poutine
import pyro.infer
import pyro.infer.autoguide
from pyro.nn import PyroSample, PyroParam, PyroModule
from functools import partial
from .torch_utils import enable_jit, disable_jit, _jit_enabled

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

def render_model(model,
                 guide = None,
                 model_args = tuple(),
                 model_kwargs = dict(),
                 render_distributions = True,
                 render_params = True,
                 render_deterministic = True):
    if _jit_enabled:
        jit_enabled = True
        disable_jit()
    else:
        jit_enabled = False

    if guide is not None:
        guide_trace = poutine.trace(guide).get_trace()
        model = poutine.replay(model, trace=guide_trace)

    graph = pyro.render_model(model,
                              model_args = model_args,
                              model_kwargs = model_kwargs,
                              render_distributions = render_distributions,
                              render_params = render_params,
                              render_deterministic = render_deterministic)

    if jit_enabled:
        enable_jit()
    
    return graph

if __name__ == '__main__':
    import doctest
    doctest.testmod()
