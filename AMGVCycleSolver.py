from VCycleSolverBase import VCycleSolverBase
from Tab import *
from Debug import *
from ClassicSmoothers import *
from AMGRefinementSequence import AMGRefinementSequence

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy.linalg as la
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp

class AMGVCycleSolver(VCycleSolverBase):
  def __init__(self, nuPre=3, nuPost=3, maxIters=100,
      numLevels=4, tol=1.0e-8, theta=0.25, graph=False, verb=0):
    super().__init__(nuPre=nuPre, nuPost=nuPost, maxIters=maxIters,
      tol=tol, verb=verb)
    self.numLevels = numLevels
    self.refSeq = None
    self.theta = theta
    self.graph = graph

  def prepForSolve(self, A):
    if self.refSeq == None or A is not self.refSeq.seqA[self.numLevels-1]:
      self.refSeq = AMGRefinementSequence(A, self.numLevels,
        theta=self.theta, graph=self.graph, verb=self.verb)


# ---------------------------------------------------------------------------
# Test code



if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from GoofySquare import GoofySquare1
  from AMGCoarsen import coarsen
  from UniformRectangleMesher import UniformRectangleMesher
  from MPLMeshViewer import MPLMeshViewer

  np.set_printoptions(precision=4)

  M = 256
  mesh = UniformRectangleMesher(0.0, 1.0, M, 0.0, 1.0, M)

  beta = 0.1
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)

  numLevels = 4
  verb = 1
  nu = 3
  theta = 0.00


  # Create a random exact solution and create the RHS vector
  rs = RandomState(MT19937(SeedSequence(123456789)))
  xEx = rs.rand(A.shape[0])
  b = A*xEx # Method of manufactured solutions

  # Construct multigrid solver
  solver = AMGVCycleSolver(verb=verb, numLevels=numLevels, nuPre=nu,
    nuPost = nu, maxIters=50, tol=1.0e-10, theta=theta, graph=False)
  # Do the solve
  x = solver.solve(A, b)

  err = la.norm(xEx-x)/la.norm(xEx)
  print('error = ', err)
