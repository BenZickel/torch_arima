import torch as pt
import math
from torch.distributions.transforms import ComposeTransform
from .Polynomial import BiasOnePolynomial, PD
from .Transform import ARMATransform

def calc_grads_hesss(coefs, grad_params, hess_params):
    grads = []
    hesss = []
    for n, grad_param in enumerate(grad_params):
        grads.append([])
        if n < len(hess_params):
            hesss.append([])
        for coef in coefs:
            grads[-1].append(pt.autograd.grad(coef, grad_param, create_graph=True)[0])
            if n < len(hess_params):
                hesss[-1].append([])
                for g in grads[-1][-1]:
                    hesss[-1][-1].append(pt.autograd.grad(g, hess_params[n], create_graph=True)[0])
    grads = [pt.stack(grad).detach().clone() for grad in grads]
    hesss = [pt.stack([pt.stack(h) for h in hess]).detach().clone() for hess in hesss]
    return grads, hesss

def replicate_to_length(x, length, dim=-1):
    shape = list(x.shape)
    if shape[dim] == length:
        return x
    elif shape[dim] > length:
        raise UserWarning('Length shorter than input at specified dimension.')
    else:
        sizes = [1] * len(shape)
        sizes[dim] = math.ceil(length / shape[dim])
        x = x.repeat(*sizes)
        select = [slice(None)] * len(shape)
        select[dim] = slice(length)
        return x[select]

def get_tail(tail_type, s, coefs):
    if tail_type == 'zero':
        return pt.zeros(len(coefs) - 1)
    elif tail_type == 'full':
        return pt.nn.Parameter(pt.zeros(len(coefs) - 1))
    elif tail_type == 'seasonal':
        return pt.nn.Parameter(pt.zeros(s))
    else:
        return pt.nn.Parameter(pt.zeros(tail_type))

class Transform():
    def forward(self, observations, *args, **kwargs):
        return self.get_transform(*args, **kwargs).inv(observations)
    
    def predict(self, innovations, *args, **kwargs):
        return self.get_transform(*args, **kwargs)(innovations)

class ARIMA(Transform, pt.nn.Module):
    '''
    ARIMA time series implementation in PyTorch.

    Args:
        p: Auto regressive order.
        d: Integrating order.
        q: Moving average order.
        ps: Seasonal auto regressive order.
        ds: Seasonal integrating order.
        qs: Seasonal moving average order.
        s: Seasonality
        drift: If set to ``True``, drift will be included.
        i_tail_type: Set input tail to zero if 'zero', fully parametrized if 'full', seasonally parametrized if 'seasonal', and i_tail_type parametrized if an integer.
        o_tail_type: Set output tail to zero if 'zero', fully parametrized if 'full', seasonally parametrized if 'seasonal', and i_tail_type parametrized if an integer.
    
    Examples:
        >>> from ARIMA import ARIMA
        >>> from numpy.testing import assert_array_almost_equal
        >>> from torch import randn
        >>> arima = ARIMA(2, 1, 1, 3, 0, 0, 4, True, 'full', 'full')
        >>> input = randn(7)
        >>> output = arima.predict(input)
        >>> input_from_output = arima(output)
        >>> assert_array_almost_equal(input.detach().numpy(), input_from_output.detach().numpy(), 6)
    '''
    def __init__(self, p, d, q, ps, ds, qs, s, drift=False, i_tail_type='zero', o_tail_type='full', output_transforms=[]):
        super().__init__()
        self.PD = PD(p, d)
        self.Q = BiasOnePolynomial(q)
        self.PDS = PD(ps, ds, s)
        self.QS = BiasOnePolynomial(qs, s)
        self.output_transforms = output_transforms
        
        # Calculate coefficients factors of observables
        o_coefs = list((self.PD * self.PDS).get_coefs())[::-1]
        o_grad_params = self.PD.get_params() + self.PDS.get_params()
        self.o_grads, self.o_hesss = calc_grads_hesss(o_coefs, o_grad_params, o_grad_params[1:])
        self.o_coefs = pt.cat([o_coef[None] for o_coef in o_coefs]).detach().clone()
        
        # Calculate coefficients factors of innovations
        i_coefs = list((self.Q * self.QS).get_coefs())[::-1]
        i_grad_params = self.Q.get_params() + self.QS.get_params()
        self.i_grads, self.i_hesss = calc_grads_hesss(i_coefs, i_grad_params, i_grad_params[1:])
        self.i_coefs = pt.cat([i_coef[None] for i_coef in i_coefs]).detach().clone()
        
        self.drift = pt.nn.Parameter(pt.zeros(1)) if drift else pt.zeros(1)
        self.i_tail = get_tail(i_tail_type, s, self.i_coefs)
        self.o_tail = get_tail(o_tail_type, s, self.o_coefs)
        
    def get_transform(self, x=None, idx=None, x_is_in_not_out=None):
        # Calculate coefficients of observations
        o_params = self.PD.get_params() + self.PDS.get_params()
        o_coefs = self.o_coefs
        for o_grad, o_param in zip(self.o_grads, o_params):
            o_coefs = o_coefs + pt.matmul(o_grad, o_param[...,None])[...,0]
        for o_hess, o_left, o_right in zip(self.o_hesss, o_params, o_params[1:]):
            o_coefs = o_coefs + pt.matmul(pt.matmul(o_left[..., None, None, :], o_hess), o_right[..., None, :, None])[..., 0, 0]

        # Calculate coefficients of innovations
        i_params = self.Q.get_params() + self.QS.get_params()
        i_coefs = self.i_coefs
        for i_grad, i_param in zip(self.i_grads, i_params):
            i_coefs = i_coefs + pt.matmul(i_grad, i_param[..., None])[...,0]
        for i_hess, i_left, i_right in zip(self.i_hesss, i_params, i_params[1:]):
            i_coefs = i_coefs + pt.matmul(pt.matmul(i_left[..., None, None, :], i_hess), i_right[..., None, :, None])[..., 0, 0]

        # Transform observations to their value at the ARMA transfrom output
        if x_is_in_not_out is not None:
            x = x.clone()
            output_transforms = [transform
                                 if isinstance(transform, pt.distributions.transforms.Transform) else
                                 transform(~x_is_in_not_out) for transform in self.output_transforms]
            x[..., ~x_is_in_not_out] = ComposeTransform(output_transforms).inv(x[..., ~x_is_in_not_out])

        # Set tails to correct length
        i_tail = replicate_to_length(self.i_tail, i_coefs.shape[-1] - 1)
        o_tail = replicate_to_length(self.o_tail, o_coefs.shape[-1] - 1)
        
        # Setup output transform
        output_transforms = [transform
                             if isinstance(transform, pt.distributions.transforms.Transform) else
                             transform(idx) for transform in self.output_transforms]
        
        return ComposeTransform([ARMATransform(i_tail, o_tail,
                                               i_coefs, o_coefs, self.drift,
                                               x, idx, x_is_in_not_out)] + output_transforms)

class VARIMATransform(Transform):
    def get_transform(self, x=None, idx=None, x_is_in_not_out=None):
        if x is None:
            x_vec = [None] * len(self.arimas)
        else:
            x_vec = [x[..., idx] for idx in range(x.shape[-1])]
        return pt.distributions.transforms.StackTransform([arima.get_transform(x_value, idx, x_is_in_not_out)
                                                                                for x_value, arima in zip(x_vec, self.arimas)], dim=-1)

class VARIMA(VARIMATransform, pt.nn.Module):
    '''
    Vector ARIMA time series implementation in PyTorch.

    Args:
        arimas: List of ARIMA transforms.

    Examples:
        >>> from ARIMA import ARIMA, VARIMA
        >>> from numpy.testing import assert_array_almost_equal
        >>> from torch import randn
        >>> varima = VARIMA([ARIMA(2, 1, 1, 3, 0, 0, 4, True, 'full', 'full') for count in range(5)])
        >>> input = randn(7, 5)
        >>> output = varima.predict(input)
        >>> first_output = varima.arimas[0].predict(input[:, 0])
        >>> assert_array_almost_equal(output[:, 0].detach().numpy(), first_output.detach().numpy(), 6)
        >>> input_from_output = varima(output)
        >>> assert_array_almost_equal(input.detach().numpy(), input_from_output.detach().numpy(), 6)
    '''
    def __init__(self, arimas):
        super().__init__()
        self.arimas = pt.nn.ModuleList(arimas)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
