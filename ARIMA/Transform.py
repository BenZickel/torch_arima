import torch as pt
from torch.distributions.transforms import ComposeTransform
from torch.distributions.constraints import independent, real

def calc_arma_transform(x, x_is_in_not_out, input, output, i_coefs, o_coefs, drift):
    # Match dimensions
    input = input[[None] * (len(x.shape) - len(input.shape)) + [Ellipsis]].expand(*(x.shape[:-1] + (input.shape[-1],)))
    output = output[[None] * (len(x.shape) - len(output.shape)) + [Ellipsis]].expand(*(x.shape[:-1] + (output.shape[-1],)))

    # Loop over input
    ret_val = []
    for n in range(x.shape[-1]):
        if x_is_in_not_out[..., n]:
            input = pt.cat([input, x[..., n][..., None]], dim=-1)
            next_val = drift
            next_val = next_val + (input[..., (-i_coefs.shape[-1]):] * i_coefs).sum(-1)[..., None]
            next_val = next_val - (output[..., (-o_coefs.shape[-1]+1):] * o_coefs[..., :-1]).sum(-1)[..., None]
            output = pt.cat([output, next_val], dim=-1)
        else:
            output = pt.cat([output, x[..., n][..., None]], dim=-1)
            next_val = -drift
            next_val = next_val + (output[..., (-o_coefs.shape[-1]):] * o_coefs).sum(-1)[..., None]
            next_val = next_val - (input[..., (-i_coefs.shape[-1]+1):] * i_coefs[..., :-1]).sum(-1)[..., None]
            input = pt.cat([input, next_val], dim=-1)
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

    def __init__(self, i_tail, o_tail, i_coefs, o_coefs, drift, x=None, idx=None, x_is_in_not_out=None, output_transforms=[]):
        super().__init__()
        self.i_tail, self.o_tail, self.i_coefs, self.o_coefs, self.drift = i_tail, o_tail, i_coefs, o_coefs, drift
        self.x, self.idx, self.x_is_in_not_out, self.output_transforms = x, idx, x_is_in_not_out, output_transforms
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
            x[..., ~x_is_in_not_out] = ComposeTransform(self.output_transforms).inv(x[..., ~x_is_in_not_out])
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
