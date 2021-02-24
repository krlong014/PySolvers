import scipy.sparse as sp
import numpy as np
from Tab import Tab

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

def smoothUpdate(A, C):
  '''
  Create an update operator from a coarse mesh to a fine mesh, based on
  a matrix A and a set of coarse-mesh nodes C
  '''
  tab0 = Tab()

  print(tab0, 'C=', C)
  if not sp.isspmatrix_csr(A):
    raise ValueError('makeUpdate() expected a CSR matrix as input')

  M = A.shape[0]
  if A.shape[1] != A.shape[0]:
    raise ValueError('Non-square matrix in makeUpdate(): size is %d by %d' %
    (A.shape[0], A.shape[1]))

  c_to_f, f_to_c = makeIndexMaps(C)
  print(tab0, 'f2c map is ', f_to_c)

  I_up = sp.dok_matrix((M,len(C)))

  ip = A.indptr
  allVals = A.data
  allCols = A.indices

  tab1 = Tab()
  tab2 = Tab()
  tab3 = Tab()
  tab4 = Tab()


  # Main loop over rows
  for i in range(M):
    print(tab0, 'row %d' % i)
    # If the current index is in the coarse mesh, take its value directly
    # from the fine mesh
    if i in C:
      I_up[i, f_to_c[i]] = 1.0
      print(tab1, 'coarse-coarse connection, entry is 1')
      continue

    # Get arrays of column indices and values for this row
    cols_i = allCols[ip[i] : ip[i+1]]
    vals_i = allVals[ip[i] : ip[i+1]]

    # Find diagonal
    diag = vals_i[np.where(cols_i==i)]
    print(tab1, 'diagonal is %g' % diag)

    # Diagonal must be nonzero!
    if diag == 0.0:
      raise ValueError('zero diagonal detected in row %d' % i)

    # build interpolation matrix
    for j, a_ij in zip(cols_i, vals_i):
      if j==i:
        continue
      if j not in C:
        continue
      print(tab2, 'coarse node f=%d, c=%d' % (j, f_to_c[j]))

      w_ij = a_ij
      print(tab2, 'direct connection: (%d,%d)=%g' % (i,j,w_ij))
      for m, a_im in zip(cols_i, vals_i):
        if m not in C: # Approximate values on F by averaging
          print(tab3, 'fine node m=%d' % m)
          cols_m = allCols[ip[m] : ip[m+1]]
          vals_m = allVals[ip[m] : ip[m+1]]
          denom = 0.0
          for k,a_mk in zip(cols_m, vals_m):
            if k in C and k in cols_i:
              denom += a_mk
            if k==j:
              num = a_im * a_mk
          print(tab4, 'numerator=%g, denominator=%g' % (num,denom))
          if denom==0.0:
            raise ValueError('zero denominator detected in (%d,%d)' %(i,j))
          print(tab3, 'indirect connection through fine node %d: %g' % (m,num/denom))
          w_ij += num/denom
        else: # No need to approximate coarse node values
          continue
      print(tab2, '(%d,%d), val=%g' %(i,j,w_ij))
      I_up[i, f_to_c[j]] = -w_ij/diag
    # end second pass through row i
  # end loop over rows

  return I_up.tocsr()


#----------------------------------------------------------------------------

if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from GoofySquare import GoofySquare1


  mesh = GoofySquare1()

  beta = 1.0
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)

  print('A=\n', A.todense())


  C = set([1,4,5,7,9,12])

  I_up = smoothUpdate(A,C)

  print(I_up.todense())
