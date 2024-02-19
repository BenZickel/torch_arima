import pyro
import torch as pt
from pyro.nn import PyroSample, PyroParam

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

