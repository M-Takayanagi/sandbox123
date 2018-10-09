import numpy as np

""" 
Funtion to generate variable Y in structural equation
  X = X (errVarX is error variable for x)
  Y = funcY(X, errVarY)
                       
  return value: Y
"""
def generate2ndVar(X, errVarY, funcY, **parmsForFuncY):
    Y = funcY(X, **parmsForFuncY) + errVarY
    return Y



""" 
Linear function of funcY
    Y = a*X + b 
"""
def linearFunc(X, a, b):
    return a*X + b

""" 
Polynomial function of funcY
    c = Coefficients for numpy.poly1d
      (example: c = [2,3,4] for 2*x^2 + 3*x + 4)
  Y = c_p-1*X^(p-1) + c_p-2*X^(p-2) + ... + c_1*X^(p-1) + c_0 
"""
def polynomialFunc(X, c):
    return np.poly1d(c)(X)


"""
Logistic function of funcY
  Y = a + (b-a)/(1+exp(c*(d-X)))
"""
def logisticFunc(X, a, b, c, d):
    return a + (b-a)/(1+np.exp(c*(d-X)))

