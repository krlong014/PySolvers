import numpy as np
import scipy.sparse as sp
from AMGCoarsen import coarsen
from AMGTransfer import smoothUpdate, makeDowndate
from Tab import Tab
from MatrixGraph import matrixGraph
from Debug import Debug
from Timer import Timer


class AMGRefinementSequence:

  def __init__(self, A_f, numLevels, theta=0.25, verb=0, graph=False):
    timer = Timer('AMG setup')
    self.seqA = [None]*numLevels
    self.updates = [None]*numLevels
    self.downdates = [None]*numLevels
    self.seqA[numLevels-1] = A_f

    tab1 = Tab()
    tab2 = Tab()
    for i in reversed(range(numLevels-1)):
      Debug.msg1(verb, '-------------------------------------------------------------')
      Debug.msg1(verb, tab1, 'coarsening level %d to level %d' % (i+1,i))
      np.set_printoptions(threshold=np.inf)
      Debug.msg4(verb, tab2, 'operator is A[%d]\n' % i, self.seqA[i+1].tolil())
      C = coarsen(self.seqA[i+1], theta = theta, verb=verb)
      Debug.msg2(verb, tab2, '#vertices: %d' % len(C))
      if graph:
        matrixGraph(self.seqA[i+1], C, name='graph-%d.gv' % i)
      Debug.msg1(verb, tab1, 'making update operator: level %d to level %d' % (i+1,i))
      I_up = smoothUpdate(self.seqA[i+1], C, verb)
      Debug.msg1(verb, tab1, 'making downdate operator: level %d to level %d' % (i,i+1))
      I_down = makeDowndate(I_up, verb)
      self.updates[i] = I_up
      self.downdates[i] = I_down
      self.seqA[i] = self.downdates[i]*(self.seqA[i+1]*self.updates[i])


  def numLevels(self):
    return len(self.seqA)

  def update(self, i):
    return self.updates[i]

  def downdate(self, i):
    return self.downdates[i]

  def matrix(self, i):
    return self.seqA[i]





# ---------------------------------------------------------------------------
# Test code



if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from GoofySquare import GoofySquare1
  from AMGCoarsen import coarsen
  from UniformRectangleMesher import UniformRectangleMesher
  from MPLMeshViewer import MPLMeshViewer

  np.set_printoptions(precision=4)

  M = 24
  mesh = UniformRectangleMesher(0.0, 1.0, M, 0.0, 1.0, M)

  beta = 0.0
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)

  numLevels = 4
  verb = 1

  AMGRS = AMGRefinementSequence(A, numLevels, verb=verb, graph=True)


  for i in range(numLevels-1):
    Debug.msg2(verb, 'I_up[%d] = \n' % i, AMGRS.update(i))
    Debug.msg2(verb, 'I_down[%d] = \n' % i, AMGRS.downdate(i))
    Debug.msg2(verb, 'A[%d] = \n' % i, AMGRS.matrix(i))
