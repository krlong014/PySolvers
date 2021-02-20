afrom UniformRefinementSequence import *
from Tab import *
from Debug import *
from ClassicSmoothers import *

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy.linalg as la
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp

class VCycleSolver:
  def __init__(self, refSeq, nuPre=3, nuPost=3,
    maxIters=100, tol=1.0e-8, verb=0):

    self.refSeq = refSeq
    self.numLevels = refSeq.numLevels()
    self.nuPre = nuPre
    self.nuPost = nuPost
    self.maxIters = maxIters
    self.tol = tol
    self.ops = None
    self.verb = verb
    self.smoothers = None

  def solve(self, A, b):

    verb = self.verb

    tab0 = Tab()
    Debug.msg1(verb, tab0, 'Setup for VCycle solve')
    # Produce sequence of coarsened operators
    self.ops = self.refSeq.makeMatrixSequence(A)

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

# Test program

if __name__=='__main__':

  # --------------------------------------------------
  # Produce a discrete Laplacian operator in 1D
  from UniformLineMesher import UniformLineMesh

  # Set number of elements for the fine mesh
  nFine = int(2**10)
  L = 2 # Number of levels

  # Find number of elements for coarse meshes
  nCoarse = nFine
  for lev in range(L-1):
    nCoarse = int(nCoarse/2)
  # Produce the coarsest mesh
  coarse = UniformLineMesh(0.0, 1.0, nCoarse)
  # Refine the coarse mesh L times
  urs = UniformRefinementSequence(coarse, L)

  # Discretize using FEM on the finest mesh
  fine = urs.meshes[L-1]

  N = len(fine.verts)
  h = 1.0/(N-1)
  A = sp.dok_matrix((N,N)) # Build as DOK matrix

  for i in range(N):
    A[i,i]=2.0/h/h
    if i>0:
      A[i,i-1] = -1.0/h/h
    if i<(N-1):
      A[i,i+1] = -1.0/h/h

  # Convert to CSR for efficiency
  A = A.tocsr()
  print('A is ', A.get_shape())

  # Create a random exact solution and create the RHS vector
  rs = RandomState(MT19937(SeedSequence(123456789)))
  xEx = rs.rand(N)
  b = A*xEx # Method of manufactured solutions

  # Construct multigrid solver
  solver = VCycleSolver(urs, verb=1, maxIters=1000, tol=1.0e-10)
  # Do the solve
  x = solver.solve(A, b)

  err = la.norm(xEx-x)/la.norm(xEx)
  print('error = ', err)
