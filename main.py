import numpy.random as nprand
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mplt
from sklearn.preprocessing import scale as scaler

from cdt.causality.pairwise import (ANM, IGCI)
from cdt.utils.io import read_causal_pairs

import variableGenerator as vg


N = 100  # Number of data
repeat = 1000  # How many times to try the causal analysis

def repeatCausalPair(N, repeat):
  print("CORRECT DIRECTION: X->Y")
  ANMcount = 0
  IGCIcount = 0
  for i in range(repeat):
    X  = nprand.rand(N)  # X (= error varialbe for X)
    eY = nprand.rand(N)  # error variable for Y

    ## Polynomial function example
    ##   Y = f(X) + eY = -5X^4 + 2X^3 + 3X^2 + 2X + 1 + eY 
    #Y = vg.generate2ndVar(X, eY, vg.polynomialFunc,
    #                      c=[-5, 1, 3, 2, 1])

    # Polynomial function example
    #   Y = -5X^4 + 2X^3 + 3X^2 + 2X + 1
    Y = vg.generate2ndVar(X, eY, vg.polynomialFunc,
                          c=[-5, 1, 3, 2, 1])

    ## Linear function example
    ##   Y = 2X + 1 + eY
    #Y = vg.generate2ndVar(X, eY, vg.linearFunc,
    #                    a=2, b=1)

    # Write X-Y plot in the first loop
    if i == 0:
      mplt.figure()
      mplt.scatter(X, Y)
      mplt.savefig('XYplot.png')

    data = pd.Series({"X":scaler(X), "Y":scaler(Y)})
    #print(data)
    #print("CORRECT DIRECTION: X->Y")

    m = ANM()
    pred = m.predict(data)
    print(pred, "(ANM, Value : 1 if X->Y and -1 if Y->X)")
    if(pred > 0):
      ANMcount+=1

    m = IGCI()
    pred = m.predict(data)
    print(pred[0], "(IGCI, Value: >0 if X->Y and <0 if Y->X)") 
    if(pred > 0):
      IGCIcount+=1
 
  return(ANMcount, IGCIcount)


counts = repeatCausalPair(N, repeat)
print()
print(counts[0]/repeat, " for ANM")
print(counts[1]/repeat, " for IGCI")

