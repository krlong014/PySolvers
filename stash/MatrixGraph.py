from graphviz import Graph
import scipy.sparse as sp
from Debug import Debug

def matrixGraph(A, C, name='graph.gv', verb=0):

  g = Graph(filename=name)
  ip = A.indptr
  allVals = A.data
  allCols = A.indices

  M = A.shape[0]

  for i in range(M):
    Debug.msg1(verb, 'i=%d' % i)
    if i in C:
      color = 'lightblue'
    else:
      color = 'lightgrey'
    g.node('%d' % i, '%d' % i, color=color, style='filled')

  for i in range(M):
    cols = allCols[ip[i] : ip[i+1]]
    for j in cols:
      if i<j:
        g.edge('%d' % i, '%d' % j)

  g.view()

#----------------------------------------------------------------------------

if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from GoofySquare import GoofySquare1
  from UniformRectangleMesher import UniformRectangleMesher
  from MPLMeshViewer import MPLMeshViewer
  from AMGCoarsen import coarsen

  mesh = UniformRectangleMesher(0.0, 1.0, 4, 0.0, 1.0, 4)

  beta = 0.0
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)

  C = coarsen(A)

  matrixGraph(A,C)
