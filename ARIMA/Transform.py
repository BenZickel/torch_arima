import torch as pt

def calc_arma_transform(input, i_tail, o_tail, i_coefs, o_coefs, bias):
    input = pt.cat([i_tail, input])
    output = o_tail
    for count in range(len(input) - len(i_tail)):
        x = bias
        x = x + (input[count:(count + len(i_tail) + 1)] * i_coefs).sum(0)
        x = x - (output[count:] * o_coefs[:-1]).sum(0)
        output = pt.cat([output, x])
    return output[-(count + 1):]

class ARMATransform(pt.distributions.transforms.Transform):
    def __init__(self, i_tail, o_tail, i_coefs, o_coefs, drift):
        super().__init__()
        self.i_tail, self.o_tail, self.i_coefs, self.o_coefs, self.drift = i_tail, o_tail, i_coefs, o_coefs, drift

    def log_abs_det_jacobian(self, x, y):
        return x.new_zeros(x.shape[: self.dim])

    def _call(self, x):
        return calc_arma_transform(x, self.i_tail, self.o_tail, self.i_coefs, self.o_coefs, self.drift)
    
    def _inverse(self, x):
        return calc_arma_transform(x, self.o_tail, self.i_tail, self.o_coefs, self.i_coefs, -self.drift)
 