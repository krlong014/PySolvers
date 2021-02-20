import scipy.sparse as sp
import numpy as np

def makeIndexMaps(C):
  '''
  Create the maps between the indices of the coarse nodes are they appear
  in the coarse and fine meshes.
  '''

  c_to_f = []
  f_to_c = {}

  for c,f in enumerate(C):
    c_to_f.append(f)
    f_to_c[f] = c

  return (c_to_f, f_to_c)

def simpleAverageUpdate(A, C):
  '''
  Create an update operator from a coarse mesh to a fine mesh, based on
  a matrix A and a set of coarse-mesh nodes C
  '''

  if not sp.isspmatrix_csr(A):
    raise ValueError('makeUpdate() expected a CSR matrix as input')

  m = A.shape[0]
  if A.shape[1] != A.shape[0]:
    raise ValueError('Non-square matrix in makeUpdate(): size is %d by %d' %
    (A.shape[0], A.shape[1]))

  c_to_f, f_to_c = makeIndexMaps(C)
  print('f2c map is ', f_to_c)

  I_up = sp.dok_matrix((m,m))

  ip = A.indptr
  allVals = A.data
  allCols = A.indices

  # Initial loop over all nodes in fine mesh
  for i in range(m):

    # If the current index is in the coarse mesh, take its value directly
    # from the fine mesh
    if i in C:
      I_up[i, f_to_c[i]] = 1.0
      continue

    # If here, the current index is in the fine mesh only. We need to compute
    # interpolation weights

    cols = allCols[ip[i] : ip[i+1]]
    vals = allVals[ip[i] : ip[i+1]]

    C_i = []
    rowSum = 0.0

    for j, a_ij in zip(cols,vals):
      if j==i: continue
      if j in C:
        C_i.append(j)
        rowSum += np.abs(a_ij)

    for j, a_ij in zip(cols,vals):
      if j==i or j not in C:
        continue
      print('processing i=%d, j=%d' % (i,j))
      I_up[i,f_to_c[j]] = np.abs(a_ij)/rowSum

  return I_up.tocsr()


#----------------------------------------------------------------------------

if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from GoofySquare import GoofySquare1

  mesh = GoofySquare1()

  beta = 0.0
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)


  C = set([1,4,5,7,9,12])

  I_up = simpleAverageUpdate(A,C)

  print(I_up.todense())
