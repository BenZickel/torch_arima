import torch as pt

class BiasOnePolynomial(pt.nn.Module):
    def __init__(self, n, multiplicity=1):
        super().__init__()
        self.bias = pt.tensor(1.0)
        self.multiplicity = multiplicity
        self.coefs = pt.nn.Parameter(pt.zeros(n)) if n > 0 else None

    def forward(self, x):
        result = self.bias
        if self.coefs is not None:
            for n, coef in enumerate(self.coefs):
                result = result + coef * x ** ((n+1) * self.multiplicity)
        return result

    def degree(self):
        return len(self.coefs) * self.multiplicity if self.coefs is not None else 0

    def get_coefs(self):
        return [self.coefs] if self.coefs is not None else []

class IntegratorPolynomial(pt.nn.Module):
    def __init__(self, n, multiplicity=1):
        super().__init__()
        self.n = n
        self.multiplicity = multiplicity

    def forward(self, x):
        return (1 - x ** self.multiplicity) ** self.n
    
    def degree(self):
        return self.n * self.multiplicity
    
class PD(pt.nn.Module):
    def __init__(self, p, d, multiplicity=1):
        super().__init__()
        self.P = BiasOnePolynomial(p, multiplicity)
        self.D = IntegratorPolynomial(d, multiplicity)

    def forward(self, x):
        return self.P(x) * self.D(x)
    
    def degree(self):
        return self.P.degree() + self.D.degree()
