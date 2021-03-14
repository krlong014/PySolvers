import scipy.sparse as sp
import numpy as np
from Debug import Debug

def findNeighborSets(A, theta=0.25):

  if not sp.isspmatrix_csr(A):
    raise ValueError('findNeighborSets() expected a CSR matrix as input')

  m = A.shape[0]
  if A.shape[1] != A.shape[0]:
    raise ValueError('Non-square matrix in findNeighborSets(): size is %d by %d' %
    (A.shape[0], A.shape[1]))

  ip = A.indptr
  allVals = A.data
  allCols = A.indices

  # pre-allocate the influence sets
  S = []
  St = []
  for i in range(m):
    S.append(set([]))
    St.append(set([]))

  # Main loop over rows
  for i in range(m):
    # Extract column indices and values for this reow
    cols = allCols[ip[i] : ip[i+1]]
    vals = allVals[ip[i] : ip[i+1]]

    # Find the maximum off-diagonal |a_ij| in this row
    maxOffDiag = 0.0
    for j, a_ij in zip(cols, vals):
      if j==i:
        continue
      if np.abs(a_ij) > maxOffDiag:
        maxOffDiag = np.abs(a_ij)

    # Find the strongly-coupled pairs
    for j, a_ij in zip(cols, vals):
      if j==i:
        continue
      if np.abs(a_ij) >= theta*maxOffDiag:
        S[i].add(j)
        St[j].add(i)

  return (S, St)


class PrioritizedSet:
  def __init__(self, numNodes):
    self.size = 0
    self.maxPriority = 0
    self.priorityToIndexMap = {}
    self.indexToPriorityMap = [0]*numNodes

  def add(self, nodeIndex, priority):
    self.size += 1
    self.indexToPriorityMap[nodeIndex] = priority
    if priority > self.maxPriority:
      self.maxPriority = priority
    if priority in self.priorityToIndexMap:
      self.priorityToIndexMap[priority].add(nodeIndex)
    else:
      self.priorityToIndexMap[priority] = set([nodeIndex])

  def updatePriority(self, nodeIndex, newP):
    oldP = self.indexToPriorityMap[nodeIndex]
    if newP == oldP:
      return
    self.remove(nodeIndex)
    self.add(nodeIndex, newP)

  def remove(self, nodeIndex):
    self.size -= 1
    priority = self.indexToPriorityMap[nodeIndex]
    self.indexToPriorityMap[nodeIndex] = 0
    self.priorityToIndexMap[priority].remove(nodeIndex)
    self.updateAfterRemoval(priority)

  # Get an arbitrary element from the set with the highest priority
  def getNext(self):
    self.size -= 1
    p = self.maxPriority
    i = self.priorityToIndexMap[p].pop()
    self.updateAfterRemoval(p)

    return i

  def updateAfterRemoval(self, priority):
    if len(self.priorityToIndexMap[priority])==0:
      del self.priorityToIndexMap[priority]
      self.maxPriority = 0
      for k in self.priorityToIndexMap.keys():
        if k > self.maxPriority:
          self.maxPriority = k

  def show(self):
    K = list(self.priorityToIndexMap)
    K.sort()
    print('\tmax priority = ', K[len(K)-1])
    for p in K:
      print('\t\tpriority=', p, ', items=', self.priorityToIndexMap[p])








def coarsen(A, theta=0.25):

  S, St = findNeighborSets(A, theta)

  # Initialize sets of coarse, fine, and unassigned nodes.
  C = set([])
  F = set([])
  U = set(range(A.shape[0]))

  # compute priorities for nodes based on connectivity
  P = PrioritizedSet(len(St))
  for i,s in enumerate(St):
    P.add(i, len(s))

  while P.size > 0:
    #print('--- start iteration --- ')
    #print('\tpriorities = ')
    #P.show()
    i = P.getNext()
    U.remove(i)
    C.add(i)
    St_i = St[i]
    for j in St_i:
      if j in U:
        U.remove(j)
        P.remove(j)
        F.add(j)

    # Sweep through all unassigned neighbors of neighbors, putting into a set
    # to remove duplications
    toUpdate = set([])
    for j in St_i:
      for k in St[j]:
        if k in U:
          toUpdate.add(k)

    # Now update priorities of neighbors of neighbors
    for k in toUpdate:
      p_k = 0
      for ell in St[k]:
        if ell in U:
          p_k += 1
        elif ell in F:
          p_k += 2
      P.updatePriority(k, p_k)

    #print('--- done sweep ---')
    #print('\tC=', C)
    #print('\tF=', F)
    #print('\tfU=', U)

  return C



#----------------------------------------------------------------------------

if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from GoofySquare import GoofySquare1
  from UniformRectangleMesher import UniformRectangleMesher
  from MPLMeshViewer import MPLMeshViewer

  mesh = UniformRectangleMesher(0.0, 1.0, 8, 0.0, 1.0, 7)
  #mesh = GoofySquare1()

  beta = 0.0
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)

  C = coarsen(A)

  viewer = MPLMeshViewer(vertRad=0.05, fontSize=14)
  viewer.show(mesh, marked=C)
