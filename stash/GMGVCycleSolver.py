from VCycleSolverBase import VCycleSolverBase
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


class GMGVCycleSolver(VCycleSolverBase):
  def __init__(self, refSeq, nuPre=3, nuPost=3, maxIters=100,
      tol=1.0e-8, verb=0):
    super().__init__(nuPre=nuPre, nuPost=nuPost, maxIters=maxIters,
      tol=tol, verb=verb)
    self.refSeq = refSeq
    self.numLevels = refSeq.numLevels()

  def prepForSolve(self, A):
    self.refSeq.makeMatrixSequence(A)

# Test program

if __name__=='__main__':

  # --------------------------------------------------
  # Produce a discrete Laplacian operator in 1D
  from UniformLineMesher import UniformLineMesh

  # Set number of elements for the fine mesh
  nFine = int(2**8)
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
  solver = GMGVCycleSolver(urs, verb=1, maxIters=1000, tol=1.0e-10)
  # Do the solve
  x = solver.solve(A, b)

  err = la.norm(xEx-x)/la.norm(xEx)
  print('error = ', err)
