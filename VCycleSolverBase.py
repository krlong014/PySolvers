from UniformRefinementSequence import *
from Tab import *
from Debug import *
from ClassicSmoothers import *
from abc import ABC, abstractmethod

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy.linalg as la
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp

class VCycleSolverBase(ABC):
  def __init__(self, nuPre=3, nuPost=3,
    maxIters=100, tol=1.0e-8, verb=0):

    self.refSeq = None
    self.numLevels = 0
    self.nuPre = nuPre
    self.nuPost = nuPost
    self.maxIters = maxIters
    self.tol = tol
    self.ops = None
    self.verb = verb
    self.smoothers = None

  @abstractmethod
  def prepForSolve(self, A):
    pass

  def solve(self, A, b):

    verb = self.verb

    tab0 = Tab()
    Debug.msg1(verb, tab0, 'Setup for VCycle solve')

    # Do any required setup for solve (differs between AMG and GMG)
    self.prepForSolve(A)

    # Get sequence of coarsened operators (assumed to be produced in the
    # call to prepForSolve())
    self.ops = self.refSeq.seqA

    # Construct the smoothers at each level
    self.smoothers = self.numLevels * [None]
    for lev in range(self.numLevels):
      self.smoothers[lev] = GaussSeidelSmoother(self.ops[lev])

    tab1 = Tab()
    Debug.msg1(verb, tab0, 'Starting VCycle solve')

    # We'll use ||b|| for convergence testing
    bNorm = la.norm(b)
    # Create vectors for residual r and solution x
    r = np.copy(b)
    x = np.copy(b)

    # Short-circuit if b=0. Solution is easy.
    if bNorm == 0.0:
      Debug.msg1(verb, tab1, 'RHS is zero, returning zero solution')
      return x

    # Main loop
    for k in range(self.maxIters):
      # Run a V-cycle
      x = self.runLevel(b, x, self.numLevels-1)

      # Compute residual
      r = b - A*x
      # Check for convergence
      rNorm = la.norm(r)
      Debug.msg2(verb, tab1, 'Iter=%d relative resid=%g' % (k, rNorm/bNorm))
      if rNorm < self.tol*bNorm:
        Debug.msg1(verb, tab1, 'Converged after %d iterations' % k)
        return x

    # If we're here, the method didn't converge withing the maximum number
    # of iterations
    Debug.msg1(verb, tab0, 'Failed to converge')
    return x

  # Carry out a V-cycle from level lev
  def runLevel(self, fh, xh, lev):

    verb = self.verb
    tab0 = Tab()
    Debug.msg2(verb, tab0,
      'lev=%d, dim(fh)=%d, dim(xh)=%d' % (lev, len(fh), len(xh)))
    tab1 = Tab()

    # if at coarsest level, do a direct solve
    if lev==0:
      A_c = self.ops[0]
      xOut = spla.spsolve(A_c, fh) # use SuperLU solver in scipy
      return xOut

    # Otherwise: pre-smooth, apply recursively, and post-smooth

    # Pre-smooth
    Debug.msg2(verb, tab1, 'pre-smooth')
    xh = self.smoothers[lev].apply(fh, xh, self.nuPre)

    # Find the residual after smoothing
    Debug.msg2(verb, tab1, 'finding resid after smoothing')
    rh = fh - self.ops[lev]*xh
    # Coarsen the residual
    Debug.msg2(verb, tab1, 'coarsening resid')
    r2h = self.refSeq.downdates[lev-1]*rh

    # Recursively apply ML to solve A^{2h} e^{2h} = r^{2h}
    Debug.msg2(verb, tab1, 'recursing...')
    x2h = np.zeros_like(r2h)
    x2h = self.runLevel(r2h, x2h, lev-1)

    # Correct the solution by adding in the prolongation of the coarse-grid error
    Debug.msg2(verb, tab1, 'fine grid correction')
    xh = xh + self.refSeq.updates[lev-1]*x2h

    # Post-smooth to remove any high-frequency errors resulting from fine-grid
    # correction
    Debug.msg2(verb, tab1, 'post-smooth')
    xh = self.smoothers[lev].apply(fh, xh, self.nuPost)

    return xh
