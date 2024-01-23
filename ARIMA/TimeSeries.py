import torch as pt
from ARIMA.taylor import taylor
from ARIMA.Polynomial import BiasOnePolynomial, PD
from ARIMA.Transform import ARMATransform

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

class Transform():
    def forward(self, observations):
        return self.get_transform().inv(observations)
    
    def predict(self, innovations):
        return self.get_transform()(innovations)

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
        fix_i_tail: If set to ``True``, the input tail will be fixed to zero.
        fix_o_tail: If set to ``True``, the output tail will be fixed to zero.

    Examples:
        >>> from ARIMA import ARIMA
        >>> from numpy.testing import assert_array_almost_equal
        >>> from torch import randn
        >>> arima = ARIMA(2, 1, 1, 3, 0, 0, 4, True, True, True)
        >>> input = randn(7)
        >>> output = arima.predict(input)
        >>> input_from_output = arima(output)
        >>> assert_array_almost_equal(input.detach().numpy(), input_from_output.detach().numpy(), 6)
    '''
    def __init__(self, p, d, q, ps, ds, qs, s, drift=False, fix_i_tail=True, fix_o_tail=False):
        super().__init__()
        self.PD = PD(p, d)
        self.Q = BiasOnePolynomial(q)
        self.PDS = PD(ps, ds, s)
        self.QS = BiasOnePolynomial(qs, s)
        
        # Calculate coefficients factors of observables
        o_coefs = taylor(lambda x: self.PD(x) * self.PDS(x), self.PD.degree() + self.PDS.degree() + 1)[::-1]
        o_grad_params = self.PD.P.get_coefs() + self.PDS.P.get_coefs()
        self.o_grads, self.o_hesss = calc_grads_hesss(o_coefs, o_grad_params, o_grad_params[1:])
        self.o_coefs = pt.cat([o_coef[None] for o_coef in o_coefs]).detach().clone()
        
        # Calculate coefficients factors of innovations
        i_coefs = taylor(lambda x: self.Q(x) * self.QS(x), self.Q.degree() + self.QS.degree() + 1)[::-1]
        i_grad_params = self.Q.get_coefs() + self.QS.get_coefs()
        self.i_grads, self.i_hesss = calc_grads_hesss(i_coefs, i_grad_params, i_grad_params[1:])
        self.i_coefs = pt.cat([i_coef[None] for i_coef in i_coefs]).detach().clone()
        
        self.drift = pt.nn.Parameter(pt.zeros(1)) if drift else pt.zeros(1)
        i_tail = pt.zeros(len(self.i_coefs) - 1)
        o_tail = pt.zeros(len(self.o_coefs) - 1)
        self.i_tail = i_tail if fix_i_tail else pt.nn.Parameter(i_tail)
        self.o_tail = o_tail if fix_o_tail else pt.nn.Parameter(o_tail)
        
    def get_transform(self, *args, **kwargs):
        # Calculate coefficients of observations
        o_params = self.PD.P.get_coefs() + self.PDS.P.get_coefs()
        o_coefs = self.o_coefs
        for o_grad, o_param in zip(self.o_grads, o_params):
            o_coefs = o_coefs + pt.matmul(o_grad, o_param[...,None])[...,0]
        for o_hess, o_left, o_right in zip(self.o_hesss, o_params, o_params[1:]):
            o_coefs = o_coefs + pt.matmul(pt.matmul(o_left[..., None, None, :], o_hess), o_right[..., None, :, None])[..., 0, 0]

        # Calculate coefficients of innovations
        i_params = self.Q.get_coefs() + self.QS.get_coefs()
        i_coefs = self.i_coefs
        for i_grad, i_param in zip(self.i_grads, i_params):
            i_coefs = i_coefs + pt.matmul(i_grad, i_param[..., None])[...,0]
        for i_hess, i_left, i_right in zip(self.i_hesss, i_params, i_params[1:]):
            i_coefs = i_coefs + pt.matmul(pt.matmul(i_left[..., None, None, :], i_hess), i_right[..., None, :, None])[..., 0, 0]

        return ARMATransform(self.i_tail, self.o_tail, i_coefs, o_coefs, self.drift, *args, **kwargs)

class VARIMA(Transform, pt.nn.Module):
    '''
    Vector ARIMA time series implementation in PyTorch.

    Args:
        arimas: List of ARIMA transforms.

    Examples:
        >>> from ARIMA import ARIMA, VARIMA
        >>> from numpy.testing import assert_array_almost_equal
        >>> from torch import randn
        >>> varima = VARIMA([ARIMA(2, 1, 1, 3, 0, 0, 4, True, True, True) for count in range(5)])
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
        
    def get_transform(self, *args, x=None, **kwargs):
        if x is None:
            x_vec = [None] * len(self.arimas)
        else:
            x_vec = [x[..., idx] for idx in range(x.shape[-1])]
        return pt.distributions.transforms.CatTransform([arima.get_transform(*args, x=x_value, strip_last_idx=True, **kwargs)
                                                                                for x_value, arima in zip(x_vec, self.arimas)], dim=-1)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
