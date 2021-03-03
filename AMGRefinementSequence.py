import numpy as np
import scipy.sparse as sp
from AMGCoarsen import coarsen
from AMGTransfer import smoothUpdate, makeDowndate
from Tab import Tab


class AMGRefinementSequence:

  def __init__(self, A_f, numLevels, verb=0):


    self.seqA = [None]*numLevels
    self.updates = [None]*numLevels
    self.downdates = [None]*numLevels
    self.seqA[numLevels-1] = A_f

    tab1 = Tab()
    tab2 = Tab()
    for i in reversed(range(numLevels-1)):
      print('-------------------------------------------------------------')
      print(tab1, 'coarsening level %d to level %d' % (i+1,i))
      print(tab2, 'operator is A[%d]\n' % i, self.seqA[i+1])
      C = coarsen(self.seqA[i+1])
      print(tab1, 'making update operator: level %d to level %d' % (i+1,i))
      I_up = smoothUpdate(self.seqA[i+1], C)
      print(tab1, 'making downdate operator: level %d to level %d' % (i,i+1))
      I_down = makeDowndate(I_up)
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

  def makeVectorSequence(self, fine_b):
    L = self.numLevels()
    seq_b = [None]*L
    seq_b[L-1]=fine_b
    for i in reversed(range(L-1)):
      seq_b[i]=self.downdates[i]*self.seq_b[i+1]
    return seq_b

# ---------------------------------------------------------------------------
# Test code



if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from GoofySquare import GoofySquare1
  from AMGCoarsen import coarsen
  from UniformRectangleMesher import UniformRectangleMesher
  from MPLMeshViewer import MPLMeshViewer

  np.set_printoptions(precision=4)

  mesh = UniformRectangleMesher(0.0, 1.0, 6, 0.0, 1.0, 6)

  beta = 0.0
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)

  numLevels = 4

  AMGRS = AMGRefinementSequence(A, numLevels)



  print('-- loading fEx and fUp')
  for i in range(numLevels-1):
    print('I_up[%d] = \n' % i, AMGRS.update(i))
    print('I_down[%d] = \n' % i, AMGRS.downdate(i))
    print('A[%d] = \n' % i, AMGRS.matrix(i))
