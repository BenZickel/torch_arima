import torch as pt

def calc_arma_transform(x, x_is_in_not_out, input, output, i_coefs, o_coefs, drift):
    ret_val = []
    for v, in_not_out in zip(x, x_is_in_not_out):
        if in_not_out:
            input = pt.cat([input, v[None]])
            x = drift
            x = x + (input[(-len(i_coefs)):] * i_coefs).sum(0)
            x = x - (output[(-len(o_coefs)+1):] * o_coefs[:-1]).sum(0)
            output = pt.cat([output, x])
        else:
            output = pt.cat([output, v[None]])
            x = -drift
            x = x + (output[(-len(o_coefs)):] * o_coefs).sum(0)
            x = x - (input[(-len(i_coefs)+1):] * i_coefs[:-1]).sum(0)
            input = pt.cat([input, x])
        ret_val.append(x)
    return pt.cat(ret_val)

class ARMATransform(pt.distributions.transforms.Transform):
    '''
    Invertible ARMA transform with support for transforming only part of the samples.
    The transform has a Jacobian determinant of one even if only part of the samples are used as input.
    See a discussion with ChatGPT on the subject at https://chat.openai.com/share/55d34600-6b9d-49ea-b7de-0b70b0e2382f.
    '''
    domain = pt.distributions.constraints.real
    codomain = pt.distributions.constraints.real

    def __init__(self, i_tail, o_tail, i_coefs, o_coefs, drift, x=None, idx=None, x_is_in_not_out=None):
        super().__init__()
        self.i_tail, self.o_tail, self.i_coefs, self.o_coefs, self.drift = i_tail, o_tail, i_coefs, o_coefs, drift
        self.x, self.idx, self.x_is_in_not_out = x, idx, x_is_in_not_out

    def log_abs_det_jacobian(self, x, y):
        return x.new_zeros(x.shape[:-1])

    def _call(self, x):
        x_is_in_not_out = pt.tensor([True] * len(x if self.x is None else self.x)) if self.x_is_in_not_out is None else self.x_is_in_not_out.clone()
        if self.x is not None:
            x_clone = self.x.clone()
            x_clone[self.idx] = x
            x = x_clone
        x = calc_arma_transform(x, x_is_in_not_out, self.i_tail, self.o_tail, self.i_coefs, self.o_coefs, self.drift)
        if self.x is not None:
            x = x[self.idx]
        return x

    def _inverse(self, x):
        x_is_in_not_out = pt.tensor([False] * len(x if self.x is None else self.x)) if self.x_is_in_not_out is None else ~self.x_is_in_not_out.clone()
        if self.x is not None:
            x_clone = self.x.clone()
            x_clone[self.idx] = x
            x = x_clone
            x_is_in_not_out[self.idx] = ~x_is_in_not_out[self.idx]
            x_is_in_not_out = ~x_is_in_not_out
        x = calc_arma_transform(x, x_is_in_not_out, self.i_tail, self.o_tail, self.i_coefs, self.o_coefs, self.drift)
        if self.x is not None:
            x = x[self.idx]
        return x
