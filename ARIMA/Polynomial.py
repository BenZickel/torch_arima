import torch as pt

class Polynomial(pt.nn.Module):
    def __init__(self, coefs=None):
        super().__init__()
        self.coefs = coefs
    
    def get_coefs(self):
        return self.coefs

    def __mul__(self, other):
        return TwoPolynomialOperation(self, other, 'Multiply')

    def __pow__(self, power):
        retval = Polynomial(pt.Tensor([1.0]))
        for count in range(power):
            retval = retval * self
        return retval

class TwoPolynomialOperation(Polynomial):
    def __init__(self, first_poly, second_poly, operation):
        super().__init__()
        self.first_poly = first_poly
        self.second_poly = second_poly
        self.operation = operation

    def get_coefs(self):
        if self.operation == 'Multiply':
            # Do polynomial multiplication
            first_poly_coefs = self.first_poly.get_coefs()
            second_poly_coefs = self.second_poly.get_coefs()
            coefs = pt.zeros(len(first_poly_coefs) + len(second_poly_coefs) - 1)
            for n, coef in enumerate(first_poly_coefs):
                temp = pt.zeros(len(first_poly_coefs) + len(second_poly_coefs) - 1)
                temp[n:(n+len(second_poly_coefs))] = coef * second_poly_coefs
                coefs = coefs + temp
            return coefs
        else:
            raise ('Operation {} is not supported.'.format(self.operation))

class BiasOnePolynomial(Polynomial):
    def __init__(self, n, multiplicity=1):
        super().__init__()
        self.bias = pt.tensor(1.0)
        self.multiplicity = multiplicity
        self.coefs = pt.nn.Parameter(pt.zeros(n))

    def get_coefs(self):
        coefs = pt.zeros(1 + len(self.coefs) * self.multiplicity)
        coefs[0] = self.bias
        coefs[self.multiplicity::self.multiplicity] = self.coefs
        return coefs

def IntegratorPolynomial(n, multiplicity=1):
    coefs = pt.zeros(1 + n * multiplicity)
    coefs[-1] = -1.0
    return Polynomial(coefs) ** n
    
def PD(p, d, multiplicity=1):
    return BiasOnePolynomial(p, multiplicity) * IntegratorPolynomial(d, multiplicity)
