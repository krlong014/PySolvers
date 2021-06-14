import numpy as np
import numpy.linalg as la
import scipy.sparse as sp

def FDLaplacian1D(a, b, m):
  h = np.abs(b-a)/np.double(m+1)

  mainDiag = -2*np.ones(m)
  offDiag = np.ones(m-1)

  A = (1/h/h)*sp.diags([mainDiag, offDiag, offDiag], [0,-1,1])

  return A
