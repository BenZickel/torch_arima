import torch as pt
from torch.distributions.transforms import ComposeTransform
from torch.distributions.constraints import independent, real
from .torch_utils import jit_script

def calc_arma_transform(x, x_is_in_not_out, input, output, i_coefs, o_coefs, drift):
    assert(x_is_in_not_out.shape==x.shape[-1:])
    # Match dimensions
    input = input[[None] * (len(x.shape) - len(input.shape)) + [Ellipsis]].expand(*(x.shape[:-1] + (input.shape[-1],)))
    output = output[[None] * (len(x.shape) - len(output.shape)) + [Ellipsis]].expand(*(x.shape[:-1] + (output.shape[-1],)))
    return calc_arma_transform_core(x, x_is_in_not_out, input, output, i_coefs, o_coefs, drift)

@jit_script
def calc_arma_transform_core(x, x_is_in_not_out, input, output, i_coefs, o_coefs, drift):
    # Assume last coefficient is one
    i_coefs = i_coefs[..., :-1]
    o_coefs = o_coefs[..., :-1]
    # Trim input and output
    input = input[..., (-i_coefs.shape[-1]):]
    output = output[..., (-o_coefs.shape[-1]):]
    # Loop over input
    ret_val = []
    for n in range(x.shape[-1]):
        next_x = x[..., n][..., None]
        if x_is_in_not_out[n]:
            next_val = next_x + ((input  * i_coefs).sum(-1) - \
                                 (output * o_coefs).sum(-1))[..., None] + drift
            input  = pt.cat([input[..., 1:],  next_x], dim=-1)
            output = pt.cat([output[..., 1:], next_val], dim=-1)
        else:
            next_val = next_x + ((output * o_coefs).sum(-1) - \
                                 (input  * i_coefs).sum(-1))[..., None] - drift
            input  = pt.cat([input[..., 1:],  next_val], dim=-1)
            output = pt.cat([output[..., 1:], next_x], dim=-1)
        ret_val.append(next_val)
    return pt.cat(ret_val, dim=-1)

class ARMATransform(pt.distributions.transforms.Transform):
    '''
    Invertible ARMA transform with support for transforming only part of the samples.
    The transform has a Jacobian determinant of one even if only part of the samples are used as input.
    See a discussion with ChatGPT on the subject at https://chat.openai.com/share/55d34600-6b9d-49ea-b7de-0b70b0e2382f.
    '''
    domain = independent(real, 1)
    codomain = independent(real, 1)
    bijective = True

    def __init__(self, i_tail, o_tail, i_coefs, o_coefs, drift, x=None, idx=None, x_is_in_not_out=None):
        super().__init__()
        self.i_tail, self.o_tail, self.i_coefs, self.o_coefs, self.drift = i_tail, o_tail, i_coefs, o_coefs, drift
        self.x, self.idx, self.x_is_in_not_out = x, idx, x_is_in_not_out
        if x_is_in_not_out is not None:
            if not x_is_in_not_out[idx].all():
                raise UserWarning('Inputs must be innovations.')
    
    def log_abs_det_jacobian(self, x, y):
        return x.new_zeros(x.shape[:(-1)]) 

    def get_x(self, x):
        x_is_in_not_out = pt.tensor([True] * (x if self.x is None else self.x).shape[-1]) if self.x_is_in_not_out is None else self.x_is_in_not_out.clone()
        if self.x is not None:
            x_clone = self.x.clone()
            x_clone[..., self.idx] = x
            x = x_clone
        return x, x_is_in_not_out

    def _call(self, x):
        x, x_is_in_not_out = self.get_x(x)
        x = calc_arma_transform(x, x_is_in_not_out, self.i_tail, self.o_tail, self.i_coefs, self.o_coefs, self.drift)
        if self.x is not None:
            x = x[..., self.idx]
        return x

    def _inverse(self, x):
        x, x_is_in_not_out = self.get_x(x)
        x_is_in_not_out = ~x_is_in_not_out
        if self.x is not None:
            x_is_in_not_out[self.idx] = ~x_is_in_not_out[self.idx]
            x_is_in_not_out = ~x_is_in_not_out
        x = calc_arma_transform(x, x_is_in_not_out, self.i_tail, self.o_tail, self.i_coefs, self.o_coefs, self.drift)
        if self.x is not None:
            x = x[..., self.idx]
        return x
