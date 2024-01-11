import torch as pt

class BiasOnePolynomial(pt.nn.Module):
    def __init__(self, n, multiplicity=1):
        super().__init__()
        self.bias = 1
        self.multiplicity = multiplicity
        self.coefs = pt.nn.Parameter(pt.zeros(n))

    def forward(self, x):
        result = self.bias
        for n, coef in enumerate(self.coefs):
            result = result + coef * x ** ((n+1) * self.multiplicity)
        return result

    def degree(self):
        return len(self.coefs) * self.multiplicity

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
