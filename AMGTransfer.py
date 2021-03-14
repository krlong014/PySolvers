import scipy.sparse as sp
import numpy as np
from Tab import Tab
from Debug import Debug

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

def smoothUpdate(A, C, verb=0):
  '''
  Create an update operator from a coarse mesh to a fine mesh, based on
  a matrix A and a set of coarse-mesh nodes C
  '''
  tab0 = Tab()

  Debug.msg2(verb, tab0, 'C=', C)
  if not sp.isspmatrix_csr(A):
    raise ValueError('makeUpdate() expected a CSR matrix as input')

  M = A.shape[0]
  if A.shape[1] != A.shape[0]:
    raise ValueError('Non-square matrix in makeUpdate(): size is %d by %d' %
    (A.shape[0], A.shape[1]))

  c_to_f, f_to_c = makeIndexMaps(C)
  Debug.msg2(verb, tab0, 'f2c map is ', f_to_c)

  I_up = sp.dok_matrix((M,len(C)))

  ip = A.indptr
  allVals = A.data
  allCols = A.indices

  tab1 = Tab()
  tab2 = Tab()
  tab3 = Tab()
  tab4 = Tab()
  tab5 = Tab()


  # Main loop over rows
  for i in range(M):
    Debug.msg2(verb, tab0, 'row %d' % i)
    # If the current index is in the coarse mesh, take its value directly
    # from the fine mesh
    if i in C:
      I_up[i, f_to_c[i]] = 1.0
      Debug.msg2(verb, tab1, 'coarse-coarse connection, entry is 1')
      continue

    Debug.msg2(verb, tab1, 'interpolating value for fine row ', i)
    # Get arrays of column indices and values for this row
    cols_i = allCols[ip[i] : ip[i+1]]
    vals_i = allVals[ip[i] : ip[i+1]]

    # Find diagonal
    diag = vals_i[np.where(cols_i==i)]
    Debug.msg2(verb, tab1, 'diagonal is %g' % diag)

    # Diagonal must be nonzero!
    if diag == 0.0:
      raise ValueError('zero diagonal detected in row %d' % i)

    # build interpolation matrix
    for j, a_ij in zip(cols_i, vals_i):
      if j==i:
        continue
      if j not in C:
        continue
      Debug.msg2(verb, tab2, 'contrib from coarse node f=%d, c=%d' % (j, f_to_c[j]))

      w_ij = a_ij
      Debug.msg2(verb, tab2, 'direct connection: (%d,%d)=%g' % (i,j,w_ij))
      Debug.msg2(verb, tab2, 'finding indirect contributions')
      Debug.msg2(verb, tab2, 'looking at neighbors ', cols_i)
      for m, a_im in zip(cols_i, vals_i):
        if m==i: # <--- I forgot this on first pass!
          continue
        if m not in C: # Approximate values on F by averaging
          Debug.msg2(verb, tab3, 'approximating fine node m=%d' % m)
          cols_m = allCols[ip[m] : ip[m+1]]
          vals_m = allVals[ip[m] : ip[m+1]]
          denom = 0.0
          num = 0.0
          Debug.msg2(verb, tab3, 'neighbors of m=%d are ' % m, cols_m)
          count = 0
          for k,a_mk in zip(cols_m, vals_m):
            if k in C and k in cols_i:
              Debug.msg2(verb, tab4, 'indirect contribution from coarse node k=', k)
              denom += a_mk
              count += 1
            if k==j:
              num = a_im * a_mk
          Debug.msg2(verb, tab4, 'numerator=%g, denominator=%g' % (num,denom))
          if count>0:
            if denom==0.0:
              raise ValueError('zero denominator detected in (%d,%d)' %(i,j))
            Debug.msg2(verb, tab3, 'indirect connection through fine node %d: %g' % (m,num/denom))
            w_ij += num/denom
        else: # No need to approximate coarse node values
          Debug.msg2(verb, tab3, 'no need to approximate value at coarse node m=', m)
          continue
      Debug.msg2(verb, tab2, '(%d,%d), val=%g' %(i,j,w_ij))
      I_up[i, f_to_c[j]] = -w_ij/diag
    # end second pass through row i
  # end loop over rows

  return I_up.tocsr()

def makeDowndate(I_up, normalize=True, verb=0):
  I_down = I_up.transpose(copy=True).tolil()
  Debug.msg1(verb, 'I_down shape=', I_down.shape)
  if normalize:
    for r in range(I_down.shape[0]):
      row = I_down.getrowview(r)
      nrm = row.sum()
      row /= nrm
  return I_down.tocsr()

#----------------------------------------------------------------------------

if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from GoofySquare import GoofySquare1
  from AMGCoarsen import coarsen
  from UniformRectangleMesher import UniformRectangleMesher
  from MPLMeshViewer import MPLMeshViewer

  np.set_printoptions(precision=4)

  mesh = UniformRectangleMesher(0.0, 1.0, 2, 0.0, 1.0, 2)

  beta = 0.0
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)

  np.set_printoptions(precision=4)
  verb=0
  Debug.msg1(verb, 'A=\n', A.todense())


  C = coarsen(A)


  I_up = smoothUpdate(A,C)
  I_down = makeDowndate(I_up)

  Debug.msg1(verb, 'Update matrix:')
  Debug.msg1(verb, I_up.todense())

  Debug.msg1(verb, "Update row sums:")
  Debug.msg1(verb, I_up*np.ones(I_up.shape[1]))

  Debug.msg1(verb, 'Downdate matrix:')
  Debug.msg1(verb, I_down.todense())

  Debug.msg1(verb, "Downdate row sums:")
  Debug.msg1(verb, I_down*np.ones(I_down.shape[1]))

  viewer = MPLMeshViewer(vertRad=0.025, fontSize=10)
  viewer.show(mesh, marked=C)
