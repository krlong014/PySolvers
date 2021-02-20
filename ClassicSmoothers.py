import scipy.sparse.linalg as spla
import scipy.sparse as sp
import numpy as np

class JacobiSmoother:
  def __init__(self, A):
    self.A = A
    self.DInv = np.reciprocal(A.diagonal())

  def apply(self, f, x, nu):

    for i in range(nu):
      r = f - self.A*x
      x = x + np.multiply(self.DInv, r)

    return x


# Gauss-Seidel
class GaussSeidelSmoother:

  # Constructor: extract UT part of matrix
  def __init__(self, A):
    self.A = A
    self.U = sp.triu(A).tocsr()

  # Apply nu steps of Gauss-Seidel with f as the RHS
  def apply(self, f, x, nu):

    for i in range(nu):
      r = f - self.A*x # Residual at current step
      # Compute U^{-1}*(f - A*x)
      dx = spla.spsolve(self.U, r)
      x = x + dx 

    return x
