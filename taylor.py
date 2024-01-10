import torch as pt

def taylor(f, n=pt.inf, origin=0.0):
    '''
    Calculate the differentiable Taylor series of function `f` up to order `n` at point `origin'.

    Args:
        f: Function receiving one argument.
        n: Maximal order of the returned Taylor series.
            Default: ``torch.inf``
        origin: Point at which the Taylor series should be calculated.
            Default: ``0.0``

    Examples:
        >>> from taylor import taylor
        >>> def p(x): return 5*x**6 + 9*x**3 + 2.5*x**2 + 1.5*x + 8
        >>> series = taylor(p)
        >>> series = [float(s.detach().numpy()) for s in series]
        >>> print(series)
        [8.0, 1.5, 2.5, 9.0, 0.0, 0.0, 5.0]
    '''
    x = pt.nn.Parameter(pt.tensor(origin))
    series = [f(x)]
    count = 1
    while count < n:
        grad = pt.autograd.grad(series[-1], x, create_graph=True)[0] / count
        if grad.grad_fn is None:
            break
        else:
            series.append(grad)
        count += 1
    return series

if __name__ == "__main__":
    import doctest
    doctest.testmod()
